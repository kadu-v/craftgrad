# 18. local buffer と barrier で並列縮約する

前の節で動かしたコードに、今回使う演算と処理を追加します。各節の確認を実行してから次へ進みます。

参照するUOpの構成は [tinygrad c22edff](https://github.com/tinygrad/tinygrad/blob/c22edff14c79310a9b2df62a78819cf77c8289fa/tinygrad/uop/ops.py) です。最初から全機能を実装せず、使う節で増やします。

## 18.01 shared memory と barrier で一行を縮約する

まず幅とlocal sizeが同じ2の冪の一行を扱う。各laneが一要素を共有領域へ書き、全laneでbarrierを通してから半分ずつ足す。

`src/gpu.rs` の `pub(crate) struct Builder`を置き換える。

```rust
// file="src/gpu.rs" mode="replace" item="pub(crate) struct Builder"
pub(crate) struct Builder {
    ops: Vec<UOp>,
    seen: HashSet<u64>,
    pub(crate) next: usize,
}
```

`src/gpu.rs` の `impl Builder` 内 の `pub(crate) fn new`を置き換える。

```rust
// file="src/gpu.rs" mode="replace" within="impl Builder" item="pub(crate) fn new"
    pub(crate) fn new() -> Self {
        Self {
            ops: vec![],
            seen: HashSet::new(),
            next: 0,
        }
    }
```

`src/gpu.rs` の `impl Builder` 内を追記する。

```rust
// file="src/gpu.rs" mode="append" within="impl Builder"
    pub(crate) fn alloc(&mut self, size: usize) -> UOp {
        let id = self.next;
        self.next += 1;
        self.emit(UOp::new(
            Ops::Alloc,
            vec![],
            Arg::Allocation {
                id,
                dtype: DType::F32,
                size,
            },
        ))
    }

    pub(crate) fn barrier(&mut self, dep: UOp) -> UOp {
        self.emit(UOp::new(Ops::Barrier, vec![dep], Arg::None))
    }

    pub(crate) fn load(&mut self, b: &UOp, index: UOp, valid: UOp) -> UOp {
        self.emit(b.index(index, valid).load())
    }

    pub(crate) fn store(&mut self, b: &UOp, index: UOp, value: UOp, valid: UOp) -> UOp {
        self.emit(b.index(index, valid).store(value))
    }
```

`src/gpu.rs`を追記する。

```rust
// file="src/gpu.rs" mode="append"


pub fn local_reduction(rows:usize,width:usize,local:usize)->Result<(UOp,Launch),String>{
    if rows!=1 || width!=local || !local.is_power_of_two(){return Err("one row with power-of-two width equal to local size is required".into());}
    let mut b=Builder::new();let input=b.param(0,width);let output=b.param(1,1);let lane=b.special("lid0");
    let shared=b.alloc(local);let value=b.load(&input,lane.clone(),UOp::bool(true));
    let stored=b.store(&shared,lane.clone(),value,UOp::bool(true));let mut dep=b.barrier(stored);
    for step in (0..local.ilog2()).rev().map(|s|1usize<<s){let valid=lane.lt(&UOp::int(step as i64));
        let a=b.load(&shared.after(vec![dep.clone()]),lane.clone(),valid.clone());
        let rhs=b.load(&shared.after(vec![dep]),lane.add(&UOp::int(step as i64)),valid.clone());
        let stored=b.store(&shared,lane.clone(),a.add(&rhs),valid);dep=b.barrier(stored);}
    let value=b.load(&shared.after(vec![dep]),UOp::int(0),UOp::bool(true));b.store(&output,UOp::int(0),value,lane.eq(&UOp::int(0)));
    Ok((b.finish(),Launch{groups:[1,1,1],local:[local,1,1]}))
}
```

`src/renderers.rs` の `pub fn render`を置き換える。

```rust
// file="src/renderers.rs" mode="replace" item="pub fn render"
pub fn render(
    linear: &UOp,
    entry: &str,
    dialect: Dialect,
    launch: Launch,
) -> Result<Kernel, String> {
    ident(entry)?;
    if linear.op != Ops::Linear {
        return Err("GPU renderer requires lowered LINEAR UOps".into());
    }
    let mut params = BTreeMap::new();
    let mut values = HashMap::<u64, String>::new();
    let mut ranges = HashSet::new();
    let mut body = String::new();
    let mut declarations = String::new();
    let mut shared_bytes = 0usize;
    let core_signature = crate::codegen::signature(linear)?;
    let buffer_slots = core_signature
        .buffers
        .iter()
        .map(|(slot, u)| (u.id, *slot))
        .collect::<HashMap<_, _>>();
    for (slot, _) in &core_signature.buffers {
        params.insert(*slot, format!("p{slot}"));
    }
    for u in &linear.src {
        if let Arg::Slot { index, .. } = u.arg {
            if u.op == Ops::Param {
                params.insert(index, format!("p{index}"));
            }
        }
    }
    let mut source = match dialect {
        Dialect::Metal => "#include <metal_stdlib>\nusing namespace metal;\n".to_owned(),
        Dialect::OpenCl => String::new(),
    };
    source
        .push_str(
            "inline long chibi_div(long a,long b){long q=a/b,r=a%b;return q-((r!=0)&&((r<0)!=(b<0)));}\ninline long chibi_mod(long a,long b){long r=a%b;return r+(((r!=0)&&((r<0)!=(b<0)))?b:0);}\n",
        );
    {}
    let mut signature = params
        .iter()
        .map(|(idx, name)| match dialect {
            Dialect::Metal => format!("device float* {name} [[buffer({idx})]]"),
            Dialect::OpenCl => format!("__global float* {name}"),
        })
        .collect::<Vec<_>>();
    if dialect == Dialect::Metal {
        signature
            .extend([
                "uint3 gid [[thread_position_in_grid]]".into(),
                "uint3 lid [[thread_position_in_threadgroup]]".into(),
                "uint3 group [[threadgroup_position_in_grid]]".into(),
            ]);
    }
    for u in &linear.src {
        let name = format!("v{}", u.id);
        let get = |i: usize| -> Result<String, String> {
            let s = u
                .src
                .get(i)
                .ok_or_else(|| format!("missing {:?} source {i}", u.op))?;
            values
                .get(&s.id)
                .cloned()
                .ok_or_else(|| {
                    format!("GPU UOp {:?} source {} not linearized", u.op, s.id)
                })
        };
        let expr = match u.op {
            Ops::Const => {
                match u.arg {
                    Arg::F32(bits) => {
                        match dialect {
                            Dialect::Metal => format!("as_type<float>((uint){bits})"),
                            Dialect::OpenCl => format!("as_float((uint){bits})"),
                        }
                    }
                    Arg::I32(x) => format!("{x}"),
                    Arg::Int(x) => format!("{x}L"),
                    Arg::Bool(b) => if b { "true" } else { "false" }.into(),
                    _ => return Err("unsupported GPU constant".into()),
                }
            }
            Ops::Alloc => {
                if let Arg::Allocation { dtype, size, .. } = u.arg {
                    let address = if size > 1 {
                        shared_bytes = shared_bytes
                            .checked_add(
                                size
                                    * match dtype {
                                        DType::I64 => 8,
                                        _ => 4,
                                    },
                            )
                            .ok_or("GPU shared memory overflow")?;
                        match dialect {
                            Dialect::Metal => "threadgroup ",
                            Dialect::OpenCl => "__local ",
                        }
                    } else {
                        ""
                    };
                    declarations
                        .push_str(&format!("{address}{} {name}[{size}];\n", ty(dtype)));
                    values.insert(u.id, name);
                    continue;
                } else {
                    return Err("GPU allocation missing layout".into());
                }
            }
            Ops::Param | Ops::Buffer => {
                format!(
                    "p{}", buffer_slots.get(& u.id).ok_or("GPU buffer missing slot") ?
                )
            }
            Ops::Index => {
                if let Arg::DType(DType::F32x4) = u.arg {
                    let address = if dialect == Dialect::Metal {
                        "device "
                    } else {
                        "__global "
                    };
                    format!("((({address}float4*){})+({}))", get(0) ?, get(1) ?)
                } else {
                    format!("({}+({}))", get(0) ?, get(1) ?)
                }
            }
            Ops::Load => {
                {}
                let index = &u.src[0];
                let ptr = get(0)?;
                let cast = match u.arg {
                    Arg::DType(dt) => format!("{}*", ty(dt)),
                    _ => "float*".into(),
                };
                let addr = if dialect == Dialect::Metal {
                    "device "
                } else {
                    "__global "
                };
                let typed = if matches!(u.arg, Arg::DType(_)) {
                    format!("(({addr}{cast}){ptr})")
                } else {
                    ptr
                };
                if index.op == Ops::Index && index.src.len() > 2
                    && index.src[2] != UOp::bool(true)
                {
                    let valid = values
                        .get(&index.src[2].id)
                        .ok_or("missing GPU load mask")?;
                    format!("({valid}?*{typed}:0)")
                } else {
                    format!("*{typed}")
                }
            }
            Ops::Store => {
                let ptr = get(0)?;
                let value = get(1)?;
                {}
                let index = &u.src[0];
                if index.op == Ops::Index && index.src.len() > 2
                    && index.src[2] != UOp::bool(true)
                {
                    let valid = values
                        .get(&index.src[2].id)
                        .ok_or("missing GPU store mask")?;
                    body.push_str(&format!("if({valid}) *{ptr}={value};\n"));
                } else {
                    body.push_str(&format!("*{ptr}={value};\n"));
                }
                String::new()
            }
            Ops::Range => {
                let (id, axis) = if let Arg::Range { id, axis } = u.arg {
                    (id, axis)
                } else {
                    return Err("range missing axis".into());
                };
                match axis {
                    Axis::Global | Axis::Local => {
                        let dim = id % 3;
                        let hw = match (dialect, axis) {
                            (Dialect::Metal, Axis::Global) => format!("gid[{}]", dim),
                            (Dialect::Metal, _) => format!("lid[{}]", dim),
                            (Dialect::OpenCl, Axis::Global) => {
                                format!("get_global_id({dim})")
                            }
                            _ => format!("get_local_id({dim})"),
                        };
                        declarations.push_str(&format!("long {name};\n"));
                        body.push_str(&format!("{name}={hw};\n"));
                    }
                    _ => {
                        declarations.push_str(&format!("long {name};\n"));
                        body.push_str(
                            &format!(
                                "for({name}={};{name}<{};{name}++){{\n", get(0) ?, get(1) ?
                            ),
                        );
                        ranges.insert(u.id);
                    }
                }
                values.insert(u.id, name);
                continue;
            }
            Ops::End => {
                if u.src.last().is_some_and(|r| ranges.contains(&r.id)) {
                    body.push_str("}\n");
                }
                String::new()
            }
            Ops::Special => {
                let text = if let Arg::Text(ref t) = u.arg {
                    t.as_str()
                } else {
                    return Err("SPECIAL requires gid/lid/group and dimension".into());
                };
                let (kind, dim) = text.split_at(text.len().saturating_sub(1));
                let dim = dim.parse::<usize>().map_err(|_| "invalid SPECIAL dimension")?;
                if dim > 2 {
                    return Err("SPECIAL dimension outside grid".into());
                }
                match (dialect, kind) {
                    (Dialect::Metal, "gid") => format!("gid[{dim}]"),
                    (Dialect::Metal, "lid") => format!("lid[{dim}]"),
                    (Dialect::Metal, "group") => format!("group[{dim}]"),
                    (Dialect::OpenCl, "gid") => format!("get_global_id({dim})"),
                    (Dialect::OpenCl, "lid") => format!("get_local_id({dim})"),
                    (Dialect::OpenCl, "group") => format!("get_group_id({dim})"),
                    _ => return Err("invalid SPECIAL builtin".into()),
                }
            }
            Ops::Barrier => {
                body.push_str(
                    match dialect {
                        Dialect::Metal => {
                            "threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                        }
                        Dialect::OpenCl => "barrier(CLK_LOCAL_MEM_FENCE);\n",
                    },
                );
                String::new()
            }
            Ops::Div if matches!(u.dtype(), DType::I32 | DType::I64) => {
                format!("chibi_div({},{})", get(0) ?, get(1) ?)
            }
            Ops::Mod => format!("chibi_mod({},{})", get(0) ?, get(1) ?),
            Ops::Add
            | Ops::Sub
            | Ops::Mul
            | Ops::Div
            | Ops::Lt
            | Ops::Eq
            | Ops::And
            | Ops::Or => {
                let op = match u.op {
                    Ops::Add => "+",
                    Ops::Sub => "-",
                    Ops::Mul => "*",
                    Ops::Div => "/",
                    Ops::Mod => "%",
                    Ops::Lt => "<",
                    Ops::Eq => "==",
                    Ops::And => "&&",
                    _ => "||",
                };
                format!("({} {op} {})", get(0) ?, get(1) ?)
            }
            Ops::Max => format!("max({},{})", get(0) ?, get(1) ?),
            Ops::Neg => format!("(-{})", get(0) ?),
            Ops::Exp
            | Ops::Log
            | Ops::Sin
            | Ops::Cos
            | Ops::Sqrt
            | Ops::Exp2
            | Ops::Log2 => {
                let fun = match u.op {
                    Ops::Exp => "exp",
                    Ops::Log => "log",
                    Ops::Sin => "sin",
                    Ops::Cos => "cos",
                    Ops::Sqrt => "sqrt",
                    Ops::Exp2 => "exp2",
                    _ => "log2",
                };
                format!("{fun}({})", get(0) ?)
            }
            Ops::Where => format!("({}?{}:{})", get(0) ?, get(1) ?, get(2) ?),
            Ops::Cast => format!("(({}){})", ty(u.dtype()), get(0) ?),
            Ops::After => get(0)?,
            Ops::Sink | Ops::Linear => String::new(),
            _ => return Err(format!("unsupported lowered GPU UOp {:?}", u.op)),
        };
        if !expr.is_empty() {
            if matches!(u.op, Ops::Index | Ops::Param | Ops::Buffer | Ops::After) {
                values.insert(u.id, expr);
            } else {
                declarations.push_str(&format!("{} {name};\n", ty(u.dtype())));
                body.push_str(&format!("{name}={expr};\n"));
                values.insert(u.id, name);
            }
        }
    }
    let mut declared = HashSet::new();
    let declarations = declarations
        .lines()
        .filter(|line| declared.insert(*line))
        .collect::<Vec<_>>()
        .join("\n");
    source
        .push_str(
            &format!(
                "{} void {entry}({}){{\n{declarations}{body}}}\n", match dialect {
                Dialect::Metal => "kernel", Dialect::OpenCl => "__kernel", }, signature
                .join(",")
            ),
        );
    Ok(Kernel {
        source,
        entry: entry.into(),
        launch,
        shared_bytes,
    })
}
```

`src/uir.rs` の `fn op`を置き換える。

```rust
// file="src/uir.rs" mode="replace" item="fn op"
fn op(s: &str) -> Result<Ops> {
    [
        Ops::Invalid,
        Ops::Const,
        Ops::Barrier,
        Ops::Special,
        Ops::Buffer,
        Ops::Alloc,
        Ops::Param,
        Ops::Cast,
        Ops::Bitcast,
        Ops::Shl,
        Ops::Shr,
        Ops::Xor,
        Ops::Add,
        Ops::Sub,
        Ops::Mul,
        Ops::Div,
        Ops::Mod,
        Ops::Neg,
        Ops::Exp,
        Ops::Log,
        Ops::Sin,
        Ops::Cos,
        Ops::Sqrt,
        Ops::Exp2,
        Ops::Log2,
        Ops::Max,
        Ops::Lt,
        Ops::Eq,
        Ops::And,
        Ops::Or,
        Ops::Where,
        Ops::Reshape,
        Ops::Permute,
        Ops::Expand,
        Ops::Shrink,
        Ops::Pad,
        Ops::Flip,
        Ops::Stack,
        Ops::Gather,
        Ops::ScatterAdd,
        Ops::Reduce,
        Ops::Detach,
        Ops::Stage,
        Ops::Range,
        Ops::Index,
        Ops::Load,
        Ops::Store,
        Ops::End,
        Ops::After,
        Ops::Call,
        Ops::Sink,
        Ops::Linear,
        Ops::Program,
        Ops::Source,
        Ops::Binary,
        Ops::CustomFunction,
    ]
    .into_iter()
    .find(|o| format!("{o:?}") == s)
    .ok_or("unknown operation".into())
}
```

`src/uir.rs` の `fn decode_arg`を置き換える。

```rust
// file="src/uir.rs" mode="replace" item="fn decode_arg"
fn decode_arg(s: &str) -> Result<Arg> {
    let fields: Vec<_> = s.split(';').collect();
    Ok(match fields.as_slice() {
        ["none"] => Arg::None,
        ["f32", v] if v.len() == 8 => {
            Arg::F32(u32::from_str_radix(v, 16).map_err(|_| "invalid f32 bits")?)
        }
        ["i32", v] => Arg::I32(number(v)?),
        ["i64", v] => Arg::Int(number(v)?),
        ["bool", v] => Arg::Bool(boolean(v)?),
        ["shape", v] => Arg::Shape(list(v)?),
        ["axes", v] => Arg::Axes(list(v)?),
        ["pairs", v] => {
            let v: Vec<usize> = list(v)?;
            if v.len() % 2 != 0 {
                return Err("odd pairs list".into());
            }
            Arg::Pairs(v.chunks_exact(2).map(|p| (p[0], p[1])).collect())
        }
        ["axis", v] => Arg::Axis(number(v)?),
        ["reduction", o, v, k] => Arg::Reduction {
            op: op(o)?,
            axes: list(v)?,
            keepdim: boolean(k)?,
        },
        ["buffer", n, v, d] => Arg::Buffer {
            name: string(n)?,
            shape: list(v)?,
            data: if *d == "-" { None } else { Some(list(d)?) },
        },
        ["range", id, a] => Arg::Range {
            id: number(id)?,
            axis: match *a {
                "Loop" => Axis::Loop,
                "Local" => Axis::Local,
                        "Global" => Axis::Global,
                "Reduce" => Axis::Reduce,
                _ => return Err("unknown axis".into()),
            },
        },
        ["allocation", id, dt, size] => Arg::Allocation {
            id: number(id)?,
            dtype: dtype(dt)?,
            size: number(size)?,
        },
        ["id", v] => Arg::Id(number(v)?),
        ["dtype", dt] => Arg::DType(dtype(dt)?),
        ["text", v] => Arg::Text(string(v)?),
        ["bytes", v] => Arg::Bytes(unhex(v)?),
        ["slot", i, v] => Arg::Slot {
            index: number(i)?,
            shape: list(v)?,
        },
        ["function",name,dt] => Arg::Function{name:string(name)?,dtype:dtype(dt)?},
        ["scalar", i, dt] => Arg::ScalarParam {
            index: number(i)?,
            dtype: dtype(dt)?,
        },
        _ => return Err("invalid typed argument".into()),
    })
}
```

`src/uop.rs` の `pub enum Ops`を置き換える。

```rust
// file="src/uop.rs" mode="replace" item="pub enum Ops"
pub enum Ops {
    Invalid,
    Const,
    Buffer,
    Alloc,
    Param,
    Cast,
    Bitcast,
    Shl,
    Shr,
    Xor,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Neg,
    Exp,
    Log,
    Sin,
    Cos,
    Sqrt,
    Exp2,
    Log2,
    Max,
    Lt,
    Eq,
    And,
    Or,
    Where,
    Reshape,
    Permute,
    Expand,
    Shrink,
    Pad,
    Flip,
    Stack,
    Gather,
    ScatterAdd,
    Reduce,
    Detach,
    Stage,
    Range,
    Index,
    Load,
    Store,
    End,
    After,
    Call,
    Sink,
    Linear,
    Program,
    Source,
    Binary,
    CustomFunction,
    Special,
    Barrier,
}
```

`src/uop.rs` の `pub enum Axis`を置き換える。

```rust
// file="src/uop.rs" mode="replace" item="pub enum Axis"
pub enum Axis {
    Loop,
    Reduce,
    Global,
    Local,
}
```

`tests/gpu_reduction.rs`を保存する。

```rust
// file="tests/gpu_reduction.rs" mode="write"
#![cfg(any(all(feature="metal",target_os="macos"),feature="opencl"))]
mod support;
use chibigrad::{gpu,UOp};
fn check(run:fn(&UOp,gpu::Launch,&[Vec<f32>])->Vec<f32>) {
 let (linear,launch)=gpu::local_reduction(1,16,16).unwrap();
 let values=(0..16).map(|i|i as f32/10.).collect::<Vec<_>>();
 let result=run(&linear,launch,&[values.clone(),vec![0.;1]]);
 for row in 0..1 {assert!((result[row]-values[row*16..(row+1)*16].iter().sum::<f32>()).abs()<1e-3);}
}

#[cfg(all(feature="metal",target_os="macos"))]
#[test]
fn metal_native(){check(support::run_metal);}
#[cfg(feature="opencl")]
#[test]
fn opencl_native(){check(support::run_cl);}
```

`tests/support/mod.rs`を保存する。

```rust
// file="tests/support/mod.rs" mode="write"
use chibigrad::{
    gpu,
    renderers::{self, Dialect},
};
fn words(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}
fn floats(bytes: Vec<u8>) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
        .collect()
}
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn run_metal(linear: &chibigrad::UOp, launch: gpu::Launch, inputs: &[Vec<f32>]) -> Vec<f32> {
    use chibigrad::{backends::metal::MetalDevice, queue::HWQueue};
    let device = MetalDevice::open().unwrap();
    eprintln!("Metal device: {:?}",device.capabilities());
    let spec = renderers::render(linear, "test_kernel", Dialect::Metal, launch.clone()).unwrap();
    let program = device
        .compile(&spec)
        .unwrap_or_else(|e| panic!("{e}\n{}", spec.source));
    let buffers = inputs
        .iter()
        .map(|input| {
            let b = device.allocate(input.len() * 4).unwrap();
            b.write(0, &words(input)).unwrap();
            b
        })
        .collect::<Vec<_>>();
    let commands = device
        .command_buffer(&[(program, launch, buffers.clone())])
        .unwrap();
    let mut queue = HWQueue::new();
    queue.metal(commands, chibigrad::UOp::int(0), HWQueue::runtime_slot(0));
    let source = chibigrad::codegen::render(queue.linear(), "runtime_entry").unwrap();
    let cpu = chibigrad::cpu::CpuProgram::compile_with_symbols(
        &source,
        "runtime_entry",
        &chibigrad::queue::foreign_symbols(),
    )
    .unwrap_or_else(|e| panic!("{e}\n{source}"));
    let mut signal = [1i64];
    let mut args = (signal.as_mut_ptr() as u64).to_ne_bytes();
    unsafe { cpu.run(&mut args) }.unwrap();
    assert_eq!(chibigrad::queue::take_error(), None);
    assert_eq!(
        floats(
            buffers
                .last()
                .unwrap()
                .read(0, inputs.last().unwrap().len() * 4)
                .unwrap()
        )
        .len(),
        inputs.last().unwrap().len()
    );
    signal[0] = 2;
    std::hint::black_box(&signal);
    unsafe { cpu.run(&mut args) }.unwrap();
    assert_eq!(chibigrad::queue::take_error(), None);
    floats(
        buffers
            .last()
            .unwrap()
            .read(0, inputs.last().unwrap().len() * 4)
            .unwrap(),
    )
}
#[cfg(feature = "opencl")]
pub fn run_cl(linear: &chibigrad::UOp, launch: gpu::Launch, inputs: &[Vec<f32>]) -> Vec<f32> {
    use chibigrad::{
        backends::opencl::{Argument, OpenClDevice},
        queue::HWQueue,
    };
    let device = OpenClDevice::open(0).unwrap();
    eprintln!("OpenCL device: {:?}",device.capabilities());
    let spec = renderers::render(linear, "test_kernel", Dialect::OpenCl, launch.clone()).unwrap();
    let program = device
        .compile(&spec)
        .unwrap_or_else(|e| panic!("{e}\n{}", spec.source));
    let buffers = inputs
        .iter()
        .map(|input| {
            let b = device.allocate(input.len() * 4).unwrap();
            b.write(0, &words(input)).unwrap();
            b
        })
        .collect::<Vec<_>>();
    let command = program
        .command(
            launch,
            buffers.iter().cloned().map(Argument::Buffer).collect(),
        )
        .unwrap();
    let mut queue = HWQueue::new();
    queue.opencl(command);
    let source = chibigrad::codegen::render(queue.linear(), "runtime_entry").unwrap();
    let cpu = chibigrad::cpu::CpuProgram::compile_with_symbols(
        &source,
        "runtime_entry",
        &chibigrad::queue::foreign_symbols(),
    )
    .unwrap_or_else(|e| panic!("{e}\n{source}"));
    unsafe { cpu.run(&mut []) }.unwrap();
    unsafe { cpu.run(&mut []) }.unwrap();
    assert_eq!(chibigrad::queue::take_error(), None);
    floats(
        buffers
            .last()
            .unwrap()
            .read(0, inputs.last().unwrap().len() * 4)
            .unwrap(),
    )
}
```

```sh
# check step=18.01 requires=metal,opencl
cargo test --offline --features metal,opencl --test gpu_reduction -- --nocapture
```

MetalとOpenCLで16要素の和が参照値と一致する。

## 18.02 行数と端数のある幅を扱う

workgroupごとに行を割り当て、各laneが複数要素を累積する。最後のtileではLOADをマスクし、全laneが同じbarrierを通る。Builderへ局所ループとENDを追加する。

`src/gpu.rs` の `impl Builder` 内を追記する。

```rust
// file="src/gpu.rs" mode="append" within="impl Builder"
    pub(crate) fn range(&mut self, end: usize) -> UOp {
        let id = self.next;
        self.next += 1;
        self.emit(UOp::range(UOp::int(end as i64), id, Axis::Reduce))
    }

    pub(crate) fn end(&mut self, r: &UOp) -> UOp {
        self.emit(UOp::new(Ops::End, vec![r.clone()], Arg::None))
    }
```

`src/gpu.rs` の `pub fn local_reduction`を置き換える。

```rust
// file="src/gpu.rs" mode="replace" item="pub fn local_reduction"
pub fn local_reduction(rows: usize, width: usize, local: usize) -> Result<(UOp, Launch), String> {
    if rows == 0 || width == 0 || !local.is_power_of_two() {
        return Err("reduction requires nonzero shape and power-of-two local size".into());
    }
    let mut b = Builder::new();
    let input = b.param(0, rows.checked_mul(width).ok_or("reduction overflow")?);
    let output = b.param(1, rows);
    let lane = b.special("lid0");
    let row = b.special("group0");
    let shared = b.alloc(local);
    let acc = b.alloc(1);
    let init = b.store(&acc, UOp::int(0), UOp::f32(0.), UOp::bool(true));
    let r = b.range(width.div_ceil(local));
    let col = r.mul(&UOp::int(local as i64)).add(&lane);
    let value = b.load(
        &input,
        row.mul(&UOp::int(width as i64)).add(&col),
        col.lt(&UOp::int(width as i64)),
    );
    let old = b.load(&acc.after(vec![init]), UOp::int(0), UOp::bool(true));
    b.store(&acc, UOp::int(0), old.add(&value), UOp::bool(true));
    let end = b.end(&r);
    let acc = b.load(&acc.after(vec![end]), UOp::int(0), UOp::bool(true));
    let stored = b.store(&shared, lane.clone(), acc, UOp::bool(true));
    let mut dep = b.barrier(stored);
    for step in (0..local.ilog2()).rev().map(|s| 1usize << s) {
        let valid = lane.lt(&UOp::int(step as i64));
        let a = b.load(
            &shared.after(vec![dep.clone()]),
            lane.clone(),
            valid.clone(),
        );
        let v = b.load(
            &shared.after(vec![dep]),
            lane.add(&UOp::int(step as i64)),
            valid.clone(),
        );
        let s = b.store(&shared, lane.clone(), a.add(&v), valid);
        dep = b.barrier(s);
    }
    let result = b.load(&shared.after(vec![dep]), UOp::int(0), UOp::bool(true));
    b.store(&output, row, result, lane.eq(&UOp::int(0)));
    Ok((
        b.finish(),
        Launch {
            groups: [rows, 1, 1],
            local: [local, 1, 1],
        },
    ))
}
```

`src/gpu.rs`を追記する。

```rust
// file="src/gpu.rs" mode="append"

use crate::Axis;
```

`tests/gpu_reduction.rs` の `fn check`を置き換える。

```rust
// file="tests/gpu_reduction.rs" mode="replace" item="fn check"
fn check(run:fn(&UOp,gpu::Launch,&[Vec<f32>])->Vec<f32>) {
 let (linear,launch)=gpu::local_reduction(3,37,16).unwrap();
 let values=(0..111).map(|i|i as f32/10.).collect::<Vec<_>>();
 let result=run(&linear,launch,&[values.clone(),vec![0.;3]]);
 for row in 0..3 {assert!((result[row]-values[row*37..(row+1)*37].iter().sum::<f32>()).abs()<1e-3);}
}
```

```sh
# check step=18.02 requires=metal,opencl
cargo test --offline --features metal,opencl --test gpu_reduction -- --nocapture
```

3行37列をlocal size16で縮約し、3つの行和が参照値と一致する。

## 章末の累積確認

前の章までのテストとexampleも再ビルドします。

```sh
# check
cargo test --quiet --all-targets
```

## 要件との対応

### 18-001 LOCAL storage と BARRIER を verifier、linearize、renderer に追加する。

<!-- todo:18-001 -->

関連する実装と確認は本章の `18.01`, `18.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 18-002 各 lane が `col=chunk*L+lane` の部分和を逐次計算する。

<!-- todo:18-002 -->

関連する実装と確認は本章の `18.01`, `18.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 18-003 全 lane が LOCAL の自分の位置に一つ書く。

<!-- todo:18-003 -->

関連する実装と確認は本章の `18.01`, `18.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 18-004 barrier 後だけ LOCAL を読み、lane 0 が一出力を書く。

<!-- todo:18-004 -->

関連する実装と確認は本章の `18.01`, `18.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 18-005 Metal と OpenCL の両方に同じ IR を下げる。

<!-- todo:18-005 -->

関連する実装と確認は本章の `18.01`, `18.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 18-006 stride を L/2、L/4、…、1 として LOCAL の二要素を足す。

<!-- todo:18-006 -->

関連する実装と確認は本章の `18.01`, `18.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 18-007 `lane<stride` の分岐では加算と LOCAL STORE だけを行う。

<!-- todo:18-007 -->

関連する実装と確認は本章の `18.01`, `18.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 18-008 各 stride の後の barrier は全 lane が実行する位置に置く。

<!-- todo:18-008 -->

関連する実装と確認は本章の `18.01`, `18.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 18-009 最終結果の global STORE は lane 0 だけに限定する。

<!-- todo:18-009 -->

関連する実装と確認は本章の `18.01`, `18.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 18-010 LOCAL の read-after-write と write-after-read の依存を保つ。

<!-- todo:18-010 -->

関連する実装と確認は本章の `18.01`, `18.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 18-011 N=1、L-1、L、L+1、257 の部分和を同じ規則で作る。

<!-- todo:18-011 -->

関連する実装と確認は本章の `18.01`, `18.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 18-012 二次元の行和では group ID を row として使う。

<!-- todo:18-012 -->

関連する実装と確認は本章の `18.01`, `18.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 18-013 input の無効 LOAD と output の無効 STORE を別々に guard する。

<!-- todo:18-013 -->

関連する実装と確認は本章の `18.01`, `18.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 18-014 LOCAL を同じ kernel のループで再利用する場合、次の書込み前に barrier を入れる。

<!-- todo:18-014 -->

関連する実装と確認は本章の `18.01`, `18.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 18-015 非 2 冪の local を明示的に拒否し、無言で誤った tree を作らない。

<!-- todo:18-015 -->

関連する実装と確認は本章の `18.01`, `18.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 18-016 大きな入力を固定 chunk に分け、各 group の部分和を GLOBAL に保存する。

<!-- todo:18-016 -->

関連する実装と確認は本章の `18.01`, `18.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 18-017 部分和バッファを次 kernel の PARAM slot に bind する。

<!-- todo:18-017 -->

関連する実装と確認は本章の `18.01`, `18.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 18-018 二つの kernel の依存を schedule に明示する。

<!-- todo:18-018 -->

関連する実装と確認は本章の `18.01`, `18.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 18-019 同じ queue 上の順序と最後の読出し完了を runtime で保証する。

<!-- todo:18-019 -->

関連する実装と確認は本章の `18.01`, `18.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 18-020 単一 workgroup の barrier を全 GPU の barrier として扱わない。

<!-- todo:18-020 -->

関連する実装と確認は本章の `18.01`, `18.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 18-021 整数として正確に表せる小例は、期待値と厳密に一致する。

<!-- todo:18-021 -->

関連する実装と確認は本章の `18.01`, `18.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 18-022 通常の f32 は加算順序の差を考慮し、`1e-5+1e-5*abs(reference)` を初期目標にする。

<!-- todo:18-022 -->

関連する実装と確認は本章の `18.01`, `18.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 18-023 正負が相殺する入力と大小が混在する入力も比較する。

<!-- todo:18-023 -->

関連する実装と確認は本章の `18.01`, `18.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 18-024 barrier を通る回数が lane の ID と入力の有効性で変わらない。

<!-- todo:18-024 -->

関連する実装と確認は本章の `18.01`, `18.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 18-025 同じ入力を繰り返して偶然の一回の成功に依存していないことを確認する。

<!-- todo:18-025 -->

関連する実装と確認は本章の `18.01`, `18.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 18-026 実機未実施の backend を数値一致済みとして報告しない。

<!-- todo:18-026 -->

関連する実装と確認は本章の `18.01`, `18.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。
