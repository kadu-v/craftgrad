# 08. shape 付き Tensor を同じ UOp の添字とループへ下げる

前の節で動かしたコードに、今回使う演算と処理を追加します。各節の確認を実行してから次へ進みます。

参照するUOpの構成は [tinygrad c22edff](https://github.com/tinygrad/tinygrad/blob/c22edff14c79310a9b2df62a78819cf77c8289fa/tinygrad/uop/ops.py) です。最初から全機能を実装せず、使う節で増やします。

## 08.01 Tensor を共有 UOp の薄い包みにする

Tensor は UOp 一つだけを持つ。ここで初めて具体的な shape、要素数、ブロードキャストの検証を加える。compile は式を添字とループへ下げ、実バッファを読む C を実行する。メモリ引数も要素数から shape に移し、以前のテストは同じ境界を保つ。

`src/codegen.rs` の `use crate::`を置き換える。

```rust
// file="src/codegen.rs" mode="replace" item="use crate::"
use crate::{
    Result,
    uop::{Arg, Axis, DType, Ops, UOp, is_binary, is_unary, numel},
};
```

`src/codegen.rs` の `pub fn to_program`を置き換える。

```rust
// file="src/codegen.rs" mode="replace" item="pub fn to_program"
pub fn to_program(sink: &UOp, entry: &str) -> Result<UOp> {
    let linear = if sink
        .src
        .iter()
        .any(|s| s.op == Ops::Store && s.src.first().is_some_and(|p| p.op == Ops::Buffer))
    {
        rangeify(sink)?
    } else {
        linearize(sink)?
    };
    let source = render(&linear, entry)?;
    Ok(UOp::new(
        Ops::Program,
        vec![
            sink.clone(),
            linear,
            UOp::new(Ops::Source, vec![], Arg::Text(source)),
        ],
        Arg::Text(entry.into()),
    ))
}
```

`src/codegen.rs` の `pub fn signature`を置き換える。

```rust
// file="src/codegen.rs" mode="replace" item="pub fn signature"
pub fn signature(linear: &UOp) -> Result<Signature> {
    let mut buffers = vec![];
    let mut seen = HashSet::new();
    let mut occupied = HashSet::new();
    for u in linear.toposort() {
        if let (Ops::Param, Arg::Slot { index, .. }) = (&u.op, &u.arg) {
            occupied.insert(*index);
            buffers.push((*index, u));
        }
    }
    let mut next = 0;
    for u in linear.toposort() {
        if !seen.insert(u.id) {
            continue;
        }
        if u.op == Ops::Buffer {
            while occupied.contains(&next) {
                next += 1;
            }
            buffers.push((next, u));
            occupied.insert(next);
            next += 1;
        }
    }
    Ok(Signature { buffers })
}
```

`src/codegen.rs` の `fn render_impl`を置き換える。

```rust
// file="src/codegen.rs" mode="replace" item="fn render_impl"
fn render_impl(linear: &UOp, entry: &str, status_slot: Option<usize>) -> Result<String> {
    let mut all = vec![];
    let mut seen = HashSet::new();
    let mut effects = HashSet::new();
    let mut scopes = vec![];
    fn add(
        u: &UOp,
        out: &mut Vec<UOp>,
        seen: &mut HashSet<u64>,
        effects: &mut HashSet<u64>,
        scopes: &mut Vec<HashSet<u64>>,
    ) {
        if seen.contains(&u.id) || effects.contains(&u.id) {
            return;
        }
        for src in u.ordered_sources() {
            add(src, out, seen, effects, scopes)
        }
        if u.op == Ops::Range {
            scopes.push(seen.clone());
        }
        seen.insert(u.id);
        out.push(u.clone());
        if matches!(u.op, Ops::Range | Ops::End | Ops::Store | Ops::Call) {
            effects.insert(u.id);
        }
        if u.op == Ops::End {
            if let Some(parent) = scopes.pop() {
                *seen = parent;
            }
        }
    }
    for u in &linear.src {
        add(u, &mut all, &mut seen, &mut effects, &mut scopes)
    }
    let complete = UOp::linear(all);
    let linear = &complete;
    verify_lowered(linear)?;

    if !identifier(entry) {
        return Err("invalid C entry identifier".into());
    }
    let sig = signature(linear)?;
    let mut text = String::from(
        "extern float fmaxf(float,float);\nstatic float chibi_float(unsigned int x) { union { unsigned int u; float f; } b; b.u=x; return b.f; }\nstatic long long chibi_div(long long a,long long b) { long long q=a/b, r=a%b; return q-((r!=0)&&((r<0)!=(b<0))); }\nstatic long long chibi_mod(long long a,long long b) { long long r=a%b; return r+(((r!=0)&&((r<0)!=(b<0)))?b:0); }\n",
    );
    text.push_str(
        "static int chibi_int(unsigned int x) { union { unsigned int u; int i; } b; b.u=x; return b.i; }\nstatic unsigned int chibi_bits(float x) { union { unsigned int u; float f; } b; b.f=x; return b.u; }\nstatic int chibi_shr(int a,int b) { unsigned int u=(unsigned int)a; return chibi_int((u>>b)|((a<0&&b)?(~0u<<(32-b)):0u)); }\nstatic long long chibi_shr64(long long a,long long b) { unsigned long long u=(unsigned long long)a; return (long long)((u>>b)|((a<0&&b)?(~0ull<<(64-b)):0ull)); }\nstatic float chibi_max(float a,float b) { if(a!=a||b!=b)return chibi_float(2143289344u); if(a==0.0f&&b==0.0f)return 0.0f; return a>b?a:b; }\n",
    );
    text.push_str(&format!("void {entry}(unsigned char *args) {{\n"));
    let mut values = HashMap::new();
    let mut addresses: HashMap<u64, (String, String, String, DType)> = HashMap::new();
    for (slot, u) in &sig.buffers {
        let name = format!("b{}", u.id);
        text.push_str(&format!(
            "  void *{name}=(void*)(unsigned long long)((unsigned long long*)args)[{slot}];\n"
        ));
        values.insert(u.id, name);
    }
    let mut depth = 1;
    let mut occurrence = 0usize;
    let mut value_scopes = vec![];
    let mut address_scopes = vec![];
    for u in &linear.src {
        let get = |s: &UOp| -> Result<String> {
            values
                .get(&s.id)
                .cloned()
                .ok_or_else(|| format!("{:?} {} used before definition", s.op, s.id))
        };
        let at = |i: usize| -> Result<String> { get(u.src.get(i).ok_or("missing source")?) };
        let indent = "  ".repeat(depth);
        if let Some(slot) = status_slot {
            let reject = format!(
                "{{ ((int*)(unsigned long long)((unsigned long long*)args)[{slot}])[0]=1; return; }}"
            );
            let a = || at(0);
            let b = || at(1);
            let guard = match u.op {
                Ops::Shl | Ops::Shr => Some(format!(
                    "({}<0 || {}>={})",
                    b()?,
                    b()?,
                    if u.dtype() == DType::I32 { 32 } else { 64 }
                )),
                Ops::Div | Ops::Mod if matches!(u.dtype(), DType::I32 | DType::I64) => {
                    Some(format!(
                        "({}==0 || ({}=={} && {}==-1))",
                        b()?,
                        a()?,
                        if u.dtype() == DType::I32 {
                            "(-2147483647-1)"
                        } else {
                            "(-9223372036854775807ll-1)"
                        },
                        b()?
                    ))
                }
                Ops::Cast
                    if u.src[0].dtype() == DType::F32
                        && matches!(u.dtype(), DType::I32 | DType::I64) =>
                {
                    let limit = if u.dtype() == DType::I32 {
                        "2147483648.0f"
                    } else {
                        "9223372036854775808.0f"
                    };
                    Some(format!("!({} >= -{limit} && {} < {limit})", a()?, a()?))
                }
                Ops::Add | Ops::Sub | Ops::Mul if u.dtype() == DType::I64 => {
                    let operation = match u.op {
                        Ops::Add => "add",
                        Ops::Sub => "sub",
                        _ => "mul",
                    };
                    text.push_str(&format!("{indent}long long checked{};\n", u.id));
                    Some(format!(
                        "__builtin_{operation}_overflow({}, {}, &checked{})",
                        a()?,
                        b()?,
                        u.id
                    ))
                }
                Ops::Neg if u.dtype() == DType::I64 => {
                    Some(format!("({}==(-9223372036854775807ll-1))", a()?))
                }
                _ => None,
            };
            if let Some(condition) = guard {
                text.push_str(&format!("{indent}if ({condition}) {reject}\n"));
            }
        }
        let expression = match (&u.op, &u.arg) {
            (Ops::Const, Arg::Float(bits)) => {
                let bits = (bits.value() as f32).to_bits();
                values.insert(u.id, format!("chibi_float({bits}u)"));
                continue;
            }
            (Ops::Const, Arg::I32(v)) => {
                values.insert(u.id, format!("((int){}u)", *v as u32));
                continue;
            }
            (Ops::Const, Arg::Int(v)) => {
                values.insert(u.id, format!("((long long){}ull)", *v as u64));
                continue;
            }
            (Ops::Const, Arg::Bool(v)) => {
                values.insert(u.id, if *v { "1" } else { "0" }.into());
                continue;
            }
            (Ops::Buffer, _) | (Ops::Param, _) => continue,
            (Ops::Alloc, Arg::Allocation { dtype, size, .. }) => {
                let name = format!("a{}", u.id);
                text.push_str(&format!(
                    "{indent}{} {name}[{}];\n",
                    ctype(*dtype),
                    size.max(&1)
                ));
                values.insert(u.id, name);
                continue;
            }
            (Ops::After, _) => {
                values.insert(u.id, at(0)?);
                if let Some(addr) = addresses.get(&u.src[0].id).cloned() {
                    addresses.insert(u.id, addr);
                }
                continue;
            }
            (Ops::Index, _) => {
                let dtype = if let Arg::DType(dt) = u.arg {
                    dt
                } else {
                    DType::F32
                };
                addresses.insert(u.id, (at(0)?, at(1)?, at(2)?, dtype));
                values.insert(
                    u.id,
                    format!("(({}*){})+({})", ctype(dtype), at(0)?, at(1)?),
                );
                continue;
            }
            (Ops::Load, _) => {
                let dtype = if let Arg::DType(dt) = u.arg {
                    dt
                } else {
                    DType::F32
                };
                if let Some((base, index, gate, _)) = addresses.get(&u.src[0].id) {
                    format!(
                        "({gate} ? (({}*){base})[{index}] : ({})0)",
                        ctype(dtype),
                        ctype(dtype)
                    )
                } else {
                    format!("(({}*){})[0]", ctype(dtype), at(0)?)
                }
            }
            (Ops::Store, _) => {
                let value = at(1)?;
                if status_slot.is_some_and(
                    |slot| matches!(u.src[0].arg, Arg::Slot { index,.. } if index == slot),
                ) {
                    text.push_str(&format!(
                        "{indent}if (!({value})) {{ ((int*){})[0]=1; return; }}\n",
                        at(0)?
                    ));
                    values.insert(u.id, "0".into());
                    continue;
                }
                if let Some((base, index, gate, dtype)) = addresses.get(&u.src[0].id) {
                    text.push_str(&format!(
                        "{indent}if ({gate}) (({}*){base})[{index}]={value};\n",
                        ctype(*dtype)
                    ));
                } else {
                    let dtype = if let Arg::DType(dt) = u.arg {
                        dt
                    } else {
                        u.src[1].dtype()
                    };
                    text.push_str(&format!(
                        "{indent}(({}*){})[0]={value};\n",
                        ctype(dtype),
                        at(0)?
                    ));
                }
                values.insert(u.id, "0".into());
                continue;
            }
            (Ops::Range, _) => {
                value_scopes.push(values.clone());
                address_scopes.push(addresses.clone());
                let name = format!("r{}", u.id);
                text.push_str(&format!(
                    "{indent}for (long long {name}={}; {name}<{}; {name}++) {{\n",
                    at(0)?,
                    at(1)?
                ));
                values.insert(u.id, name);
                depth += 1;
                continue;
            }
            (Ops::End, _) => {
                if depth == 1 {
                    return Err("extra END".into());
                }
                depth -= 1;
                text.push_str(&format!("{}}}\n", "  ".repeat(depth)));
                values = value_scopes.pop().ok_or("missing scope")?;
                addresses = address_scopes.pop().ok_or("missing address scope")?;
                values.insert(u.id, "0".into());
                continue;
            }
            (Ops::Cast, Arg::DType(DType::Bool)) => format!("({}!=0)", at(0)?),
            (Ops::Cast, Arg::DType(dtype)) => {
                format!("(({})({}))", ctype(*dtype), at(0)?)
            }
            (Ops::Bitcast, Arg::DType(dt)) if u.src[0].dtype() == *dt => at(0)?,
            (Ops::Bitcast, Arg::DType(DType::I32)) => {
                format!("chibi_int(chibi_bits({}))", at(0)?)
            }
            (Ops::Bitcast, Arg::DType(DType::F32)) => {
                format!("chibi_float((unsigned int)({}))", at(0)?)
            }
            (Ops::Shl, _) => {
                if u.dtype() == DType::I32 {
                    format!("chibi_int((unsigned int)({}) << ({}))", at(0)?, at(1)?)
                } else {
                    format!(
                        "((long long)((unsigned long long)({}) << ({})))",
                        at(0)?,
                        at(1)?
                    )
                }
            }
            (Ops::Shr, _) => {
                format!(
                    "{}({},{})",
                    if u.dtype() == DType::I32 {
                        "chibi_shr"
                    } else {
                        "chibi_shr64"
                    },
                    at(0)?,
                    at(1)?
                )
            }
            (Ops::Neg, _) => {
                if u.dtype() == DType::I32 {
                    format!("((int)(0u-(unsigned int)({})))", at(0)?)
                } else {
                    format!("(-({}))", at(0)?)
                }
            }
            (Ops::Where, _) => format!("({}?{}:{})", at(0)?, at(1)?, at(2)?),
            (Ops::Max, _) if u.dtype() == DType::F32 => {
                format!("chibi_max({},{})", at(0)?, at(1)?)
            }
            (Ops::Max, _) => format!("({}>{}?{}:{})", at(0)?, at(1)?, at(0)?, at(1)?),
            (Ops::Div, _) if matches!(u.dtype(), DType::I32 | DType::I64) => {
                format!("chibi_div({},{})", at(0)?, at(1)?)
            }
            (Ops::Mod, _) => format!("chibi_mod({},{})", at(0)?, at(1)?),
            (op, _) if is_binary(*op) => {
                let symbol = match op {
                    Ops::Add => "+",
                    Ops::Sub => "-",
                    Ops::Mul => "*",
                    Ops::Div => "/",
                    Ops::Lt => "<",
                    Ops::Eq => "==",
                    Ops::And => "&",
                    Ops::Or => "|",
                    Ops::Xor => "^",
                    _ => return Err("unsupported ALU".into()),
                };
                if u.dtype() == DType::I32 && matches!(op, Ops::Add | Ops::Sub | Ops::Mul) {
                    format!(
                        "((int)((unsigned int)({}) {symbol} (unsigned int)({})))",
                        at(0)?,
                        at(1)?
                    )
                } else {
                    format!("({} {symbol} {})", at(0)?, at(1)?)
                }
            }
            (Ops::Sink, _) => continue,
            _ => return Err(format!("C renderer does not support {:?}", u.op)),
        };
        if u.dtype() == DType::Void {
            text.push_str(&format!("{indent}{expression};\n"));
            values.insert(u.id, "0".into());
        } else {
            let name = format!("v{}_{}", u.id, occurrence);
            occurrence += 1;
            text.push_str(&format!(
                "{indent}{} {name}={expression};\n",
                ctype(u.dtype())
            ));
            values.insert(u.id, name);
        }
    }
    if depth != 1 {
        return Err("unclosed C loop".into());
    }
    text.push_str("}\n");
    Ok(text)
}
```

`src/codegen.rs` の `fn pointer_size`を置き換える。

```rust
// file="src/codegen.rs" mode="replace" item="fn pointer_size"
fn pointer_size(u: &UOp) -> Option<usize> {
    match &u.arg {
        Arg::Slot { shape, .. } => Some(numel(shape)),
        Arg::Buffer { shape, .. } => Some(numel(shape)),
        _ => None,
    }
}
```

`src/codegen.rs`を追記する。

```rust
// file="src/codegen.rs" mode="append"

fn coords(linear: &UOp, shape: &[UOp]) -> Vec<UOp> {
    let mut stride = UOp::int(1);
    let mut out = vec![];
    for n in shape.iter().rev() {
        out.push(linear.div(&stride).modulo(n));
        stride = stride.mul(n);
    }
    out.reverse();
    out
}

fn offset(indices: &[UOp], shape: &[UOp]) -> UOp {
    indices
        .iter()
        .zip(shape)
        .fold(UOp::int(0), |acc, (i, n)| acc.mul(n).add(i))
}

fn broadcast(indices: &[UOp], shape: &[UOp]) -> Vec<UOp> {
    let start = indices.len() - shape.len();
    shape
        .iter()
        .enumerate()
        .map(|(i, n)| {
            if *n == UOp::int(1) {
                UOp::int(0)
            } else {
                indices[start + i].clone()
            }
        })
        .collect()
}

struct Lower {
    ops: Vec<UOp>,
    seen: HashSet<u64>,
    next_range: usize,
    cache: HashMap<(u64, Vec<u64>, u64), UOp>,
}

impl Lower {
    fn emit(&mut self, u: UOp) -> UOp {
        if self.seen.insert(u.id) {
            for s in u.ordered_sources() {
                self.emit(s.clone());
            }
            self.ops.push(u.clone());
        }
        u
    }
    fn range(&mut self, end: UOp, axis: Axis) -> UOp {
        let end = simplify(&end).unwrap_or(end);
        let r = UOp::range(end, self.next_range, axis);
        self.next_range += 1;
        self.emit(r)
    }
    fn close(&mut self, range: &UOp, body: UOp) -> UOp {
        self.emit(UOp::new(Ops::End, vec![body, range.clone()], Arg::None))
    }
    fn scalar(&mut self, u: &UOp, indices: &[UOp], valid: &UOp) -> Result<UOp> {
        let key = (u.id, indices.iter().map(|i| i.id).collect(), valid.id);
        if let Some(v) = self.cache.get(&key) {
            return Ok(v.clone());
        }
        let result = match (&u.op, &u.arg) {
            (Ops::Const, _) => self.emit(u.clone()),
            (Ops::Buffer, _) | (Ops::Param, _) => {
                let shape = u.shape_uops()?;
                let idx = simplify(&offset(indices, &shape))?;
                let gate = simplify(valid)?;
                self.emit(u.index(idx, gate).load())
            }
            (op, _) if is_unary(*op) => {
                let x = self.scalar(&u.src[0], indices, valid)?;
                self.emit(UOp::new(*op, vec![x], u.arg.clone()))
            }
            (op, _) if is_binary(*op) || *op == Ops::Where => {
                let mut src = vec![];
                for x in &u.src {
                    src.push(self.scalar(x, &broadcast(indices, &x.shape_uops()?), valid)?)
                }
                self.emit(UOp::new(*op, src, u.arg.clone()))
            }
            _ => return Err(format!("rangeify cannot lower {:?}", u.op)),
        };
        self.cache.insert(key, result.clone());
        Ok(result)
    }
}

pub fn rangeify(sink: &UOp) -> Result<UOp> {
    if sink.op != Ops::Sink {
        return Err("rangeify expects SINK".into());
    }
    let sink = normalize(sink)?;
    let mut lower = Lower {
        ops: vec![],
        seen: HashSet::new(),
        next_range: 0,
        cache: HashMap::new(),
    };
    for store in &sink.src {
        if store.op != Ops::Store || store.src.len() != 2 {
            return Err("rangeify expects STORE(buffer, tensor)".into());
        }
        let target = &store.src[0];
        let value = &store.src[1];
        let shape = value.shape_uops()?;
        if target.shape_uops()? != shape {
            return Err("output buffer shape mismatch".into());
        }
        let r = lower.range(product(&shape), Axis::Loop);
        let coords = coords(&r, &shape);
        let result = lower.scalar(value, &coords, &UOp::bool(true))?;
        let output = lower.emit(target.index(r.clone(), UOp::bool(true)).store(result));
        lower.close(&r, output);
        lower.cache.clear();
    }
    let linear = UOp::linear(lower.ops);
    verify_lowered(&linear)?;
    Ok(linear)
}

fn product(shape: &[UOp]) -> UOp {
    shape.iter().fold(UOp::int(1), |a, b| a.mul(b))
}

fn simplify(u: &UOp) -> Result<UOp> {
    let mut values: HashMap<u64, UOp> = HashMap::new();
    for original in u.toposort() {
        let id = original.id;
        let n = original;
        let n = n.replace(n.src.iter().map(|s| values[&s.id].clone()).collect());
        let result = if (is_binary(n.op) || is_unary(n.op) || n.op == Ops::Where)
            && n.src.iter().all(|s| s.op == Ops::Const)
        {
            let v = n
                .src
                .iter()
                .map(|s| crate::scalar::Value::from_constant(s).unwrap())
                .collect::<Vec<_>>();
            crate::scalar::eval_alu(n.op, &n.arg, &v)?.uop()
        } else {
            n
        };
        values.insert(id, result);
    }
    Ok(values[&u.id].clone())
}
```

`src/lib.rs`を追記する。

```rust
// file="src/lib.rs" mode="append"

pub mod runtime;

pub mod tensor;

pub use runtime::{Executable, InputSpec, compile_many};

pub use tensor::{Buffer, Tensor};
```

`src/runtime.rs`を保存する。

```rust
// file="src/runtime.rs" mode="write"
use crate::{
    Arg, Buffer, Ops, Result, Tensor, UOp, codegen::{rangeify, render, signature},
    cpu::CpuProgram,
};
use std::collections::HashMap;
#[derive(Clone, Debug)]
pub struct InputSpec {
    pub name: String,
    pub shape: Vec<usize>,
}
pub struct Executable {
    inputs: Vec<InputSpec>,
    input_nodes: Vec<UOp>,
    captured: HashMap<u64, Buffer>,
    outputs: Vec<UOp>,
    programs: Vec<UOp>,
    native: CpuProgram,
}
pub fn compile_many(outputs: &[Tensor]) -> Result<Executable> {
    if outputs.is_empty() {
        return Err("compile requires at least one output".into());
    }
    let mut result_nodes = vec![];
    let mut stores = vec![];
    for (index, t) in outputs.iter().enumerate() {
        let b = UOp::new(
            Ops::Buffer,
            vec![],
            Arg::Buffer {
                name: format!("output_{}_{}", t.uop().id, index),
                shape: t.shape()?,
                data: None,
            },
        );
        stores.push(b.store(t.uop().clone()));
        result_nodes.push(b);
    }
    let sink = UOp::sink(stores);
    let linear = rangeify(&sink)?;
    let source = render(&linear, "tensor_kernel")?;
    let native = unsafe { CpuProgram::compile(&source, "tensor_kernel") }?;
    let mut input_nodes = vec![];
    let mut inputs = vec![];
    let mut captured = HashMap::new();
    for (_, u) in signature(&linear)?.buffers {
        if result_nodes.contains(&u) {
            continue;
        }
        let Arg::Buffer { name, shape, data } = &u.arg else {
            return Err("invalid tensor buffer".into());
        };
        if let Some(data) = data {
            captured
                .insert(
                    u.id,
                    Buffer::new(
                        shape.clone(),
                        data.iter().map(|b| f32::from_bits(*b)).collect(),
                    )?,
                );
        } else {
            inputs
                .push(InputSpec {
                    name: name.clone(),
                    shape: shape.clone(),
                });
            input_nodes.push(u);
        }
    }
    let program = UOp::new(
        Ops::Program,
        vec![
            sink, linear, UOp::new(Ops::Source, vec![], Arg::Text(source)),
            UOp::new(Ops::Binary, vec![], Arg::Bytes(native.object().to_vec())),
        ],
        Arg::Text("tensor_kernel".into()),
    );
    Ok(Executable {
        inputs,
        input_nodes,
        captured,
        outputs: result_nodes,
        programs: vec![program],
        native,
    })
}
impl Executable {
    pub fn inputs(&self) -> &[InputSpec] {
        &self.inputs
    }
    pub fn programs(&self) -> &[UOp] {
        &self.programs
    }
    pub fn kernel_count(&self) -> usize {
        1
    }
    pub fn sources(&self) -> Vec<&str> {
        vec![self.native.source().unwrap()]
    }
    pub fn run(
        &self,
        inputs: &[Buffer],
        _vars: &HashMap<String, i64>,
    ) -> Result<Vec<Buffer>> {
        if inputs.len() != self.inputs.len() {
            return Err("input count mismatch".into());
        }
        let mut bindings: HashMap<u64, &Buffer> = self
            .captured
            .iter()
            .map(|(&id, b)| (id, b))
            .collect();
        for ((spec, node), buffer) in self
            .inputs
            .iter()
            .zip(&self.input_nodes)
            .zip(inputs)
        {
            if buffer.shape != spec.shape
                || crate::uop::numel(&spec.shape) != buffer.data.len()
            {
                return Err(format!("input shape mismatch for {}", spec.name));
            }
            bindings.insert(node.id, buffer);
        }
        let mut outputs = self
            .outputs
            .iter()
            .map(|u| Buffer::zeros(&u.shape()?))
            .collect::<Result<Vec<_>>>()?;
        let sig = signature(&self.programs[0].src[1])?;
        let count = sig.buffers.iter().map(|(i, _)| i + 1).max().unwrap_or(0);
        let mut packed = vec![0u8; count * 8];
        for (slot, u) in sig.buffers {
            let pointer = if let Some(i) = self.outputs.iter().position(|out| out == &u)
            {
                outputs[i].data.as_mut_ptr()
            } else {
                bindings
                    .get(&u.id)
                    .ok_or("missing tensor binding")?
                    .data
                    .as_ptr()
                    .cast_mut()
            };
            packed[slot * 8..slot * 8 + 8]
                .copy_from_slice(&(pointer as u64).to_ne_bytes());
        }
        unsafe { self.native.run(&mut packed) }?;
        Ok(outputs)
    }
}
```

`src/scalar.rs` の `pub fn compile_with_guards`を置き換える。

```rust
// file="src/scalar.rs" mode="replace" item="pub fn compile_with_guards"
pub fn compile_with_guards(root: &UOp, guards: &[UOp]) -> Result<CompiledScalar> {
    let root = crate::functions::inline_calls(root)?;
    let types = verify(&root)?;
    for guard in guards {
        if guard.dtype() != DType::Bool {
            return Err("guard must be Bool".into());
        }
        for u in guard.toposort() {
            verify_node(&u)?;
            if let Arg::ScalarParam { index, dtype } = u.arg {
                if types.get(index) != Some(&dtype) {
                    return Err("guard parameter signature mismatch".into());
                }
            }
        }
    }
    let all = UOp::sink(
        guards.iter().cloned().chain(std::iter::once(root.clone())).collect(),
    );
    let mut mapped: HashMap<u64, UOp> = HashMap::new();
    for u in all.toposort().into_iter().filter(|u| u.op != Ops::Sink) {
        let replacement = if let Arg::ScalarParam { index, dtype } = u.arg {
            let ptr = UOp::new(
                Ops::Param,
                vec![],
                Arg::Slot {
                    index: index + 1,
                    shape: vec![1],
                },
            );
            UOp::new(Ops::Load, vec![ptr], Arg::DType(dtype))
        } else {
            u.replace(u.src.iter().map(|s| mapped[&s.id].clone()).collect())
        };
        mapped.insert(u.id, replacement);
    }
    let ptr = UOp::new(
        Ops::Param,
        vec![],
        Arg::Slot {
            index: 0,
            shape: vec![1],
        },
    );
    let store = UOp::new(
        Ops::Store,
        vec![ptr, mapped[& root.id].clone()],
        Arg::DType(root.dtype()),
    );
    let status = UOp::new(
        Ops::Param,
        vec![],
        Arg::Slot {
            index: types.len() + 1,
            shape: vec![1],
        },
    );
    let mut stores: Vec<_> = guards
        .iter()
        .map(|g| {
            UOp::new(
                Ops::Store,
                vec![status.clone(), mapped[& g.id].clone()],
                Arg::DType(DType::Bool),
            )
        })
        .collect();
    stores.push(store);
    let linear = crate::codegen::linearize(&UOp::sink(stores))?;
    let source = crate::codegen::render_checked(
        &linear,
        "chibi_scalar",
        types.len() + 1,
    )?;
    let program = unsafe { crate::cpu::CpuProgram::compile(&source, "chibi_scalar") }?;
    Ok(CompiledScalar {
        source,
        program,
        types,
        output: root.dtype(),
    })
}
```

`src/tensor.rs`を保存する。

```rust
// file="src/tensor.rs" mode="write"
use crate::{
    Result,
    runtime::{Executable, compile_many},
    uop::{Arg, DType, Ops, UOp, checked_numel},
};
use std::{cell::Cell, collections::HashMap};
#[derive(Clone, Debug, PartialEq)]
pub struct Buffer {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}
impl Buffer {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        if checked_numel(&shape)? != data.len() {
            return Err("buffer length disagrees with shape".into());
        }
        Ok(Self { data, shape })
    }
    pub fn zeros(shape: &[usize]) -> Result<Self> {
        Self::new(shape.to_vec(), vec![0.; checked_numel(shape)?])
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Tensor {
    pub(crate) uop: UOp,
}
thread_local! {
    static DATA_ID : Cell < u64 >= const { Cell::new(0) };
}
impl Tensor {
    pub fn uop(&self) -> &UOp {
        &self.uop
    }
    /// # Safety
    /// The raw graph must satisfy tensor, memory and effect invariants for its target.
    pub unsafe fn from_raw_uop(uop: UOp) -> Self {
        Self::from_uop(uop)
    }
    pub(crate) fn from_uop(uop: UOp) -> Self {
        Self { uop }
    }
    pub fn from_slice(data: &[f32], shape: &[usize]) -> Self {
        let id = DATA_ID.with(|n| {
            n.set(n.get() + 1);
            n.get()
        });
        Self::from_uop(UOp::new(
            Ops::Buffer,
            vec![],
            Arg::Buffer {
                name: format!("data_{id}"),
                shape: shape.to_vec(),
                data: Some(data.iter().map(|x| x.to_bits()).collect()),
            },
        ))
    }
    pub fn parameter(name: &str, shape: &[usize]) -> Self {
        Self::from_uop(UOp::new(
            Ops::Buffer,
            vec![],
            Arg::Buffer {
                name: name.into(),
                shape: shape.to_vec(),
                data: None,
            },
        ))
    }
    pub fn scalar(value: f32) -> Self {
        Self::from_uop(UOp::f32(value))
    }
    pub fn shape(&self) -> Result<Vec<usize>> {
        self.uop.shape()
    }
    pub fn numel(&self) -> Result<usize> {
        checked_numel(&self.shape()?)
    }

    fn unary(&self, op: Ops) -> Self {
        Self::from_uop(self.uop.alu(op, &[]))
    }
    fn binary(&self, other: &Self, op: Ops) -> Self {
        Self::from_uop(self.uop.alu(op, &[other.uop.clone()]))
    }
    pub fn add(&self, other: &Self) -> Self {
        self.binary(other, Ops::Add)
    }
    pub fn sub(&self, other: &Self) -> Self {
        self.binary(other, Ops::Sub)
    }
    pub fn mul(&self, other: &Self) -> Self {
        self.binary(other, Ops::Mul)
    }
    pub fn div(&self, other: &Self) -> Self {
        self.binary(other, Ops::Div)
    }
    pub fn maximum(&self, other: &Self) -> Self {
        self.binary(other, Ops::Max)
    }
    pub fn lt(&self, other: &Self) -> Self {
        self.binary(other, Ops::Lt)
    }
    pub fn eq(&self, other: &Self) -> Self {
        self.binary(other, Ops::Eq)
    }
    pub fn neg(&self) -> Self {
        self.unary(Ops::Neg)
    }
    pub fn cast(&self, dtype: DType) -> Self {
        Self::from_uop(self.uop.cast(dtype))
    }
    pub fn where_(&self, yes: &Self, no: &Self) -> Self {
        Self::from_uop(self.uop.where_(&yes.uop, &no.uop))
    }
    pub fn compile(&self) -> Result<Executable> {
        compile_many(std::slice::from_ref(self))
    }
    pub fn realize(&self) -> Result<Buffer> {
        let executable = self.compile()?;
        let mut outputs = executable.run(&[], &HashMap::new())?;
        Ok(outputs.remove(0))
    }
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        Ok(self.realize()?.data)
    }
}
```

`src/uir.rs` の `fn op`を置き換える。

```rust
// file="src/uir.rs" mode="replace" item="fn op"
fn op(s: &str) -> Result<Ops> {
    [
        Ops::Invalid,
        Ops::Const,
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
        Ops::Max,
        Ops::Lt,
        Ops::Eq,
        Ops::And,
        Ops::Or,
        Ops::Where,
        Ops::Stack,
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
    ]
    .into_iter()
    .find(|o| format!("{o:?}") == s)
    .ok_or("unknown operation".into())
}
```

`src/uir.rs` の `fn encode_arg`を置き換える。

```rust
// file="src/uir.rs" mode="replace" item="fn encode_arg"
fn encode_arg(a: &Arg) -> String {
    match a {
        Arg::None => "none".into(),
        Arg::Float(v) => format!("float;{:016x}", v.value().to_bits()),
        Arg::I32(v) => format!("i32;{v}"),
        Arg::Int(v) => format!("i64;{v}"),
        Arg::Bool(v) => format!("bool;{}", *v as u8),
        Arg::Buffer { name, shape, data } => {
            format!(
                "buffer;{};{};{}",
                hex(name.as_bytes()),
                numbers(shape),
                data.as_ref().map(|v| numbers(v)).unwrap_or("-".into())
            )
        }
        Arg::Range { id, axis } => format!("range;{id};{axis:?}"),
        Arg::Allocation { id, dtype, size } => {
            format!("allocation;{id};{dtype:?};{size}")
        }
        Arg::Id(v) => format!("id;{v}"),
        Arg::DType(dt) => format!("dtype;{dt:?}"),
        Arg::Text(s) => format!("text;{}", hex(s.as_bytes())),
        Arg::Bytes(v) => format!("bytes;{}", hex(v)),
        Arg::Slot { index, shape } => format!("slot;{index};{}", numbers(shape)),
        Arg::ScalarParam { index, dtype } => format!("scalar;{index};{dtype:?}"),
    }
}
```

`src/uir.rs` の `fn decode_arg`を置き換える。

```rust
// file="src/uir.rs" mode="replace" item="fn decode_arg"
fn decode_arg(s: &str) -> Result<Arg> {
    let fields: Vec<_> = s.split(';').collect();
    Ok(match fields.as_slice() {
        ["none"] => Arg::None,
        ["float", v] if v.len() == 16 && v.bytes().all(|b| b.is_ascii_hexdigit()) => {
            Arg::Float(crate::uop::ConstFloat::new(f64::from_bits(
                u64::from_str_radix(v, 16).map_err(|_| "invalid float bits")?
            )))
        }
        ["i32", v] => Arg::I32(number(v)?),
        ["i64", v] => Arg::Int(number(v)?),
        ["bool", v] => Arg::Bool(boolean(v)?),
        ["buffer", n, v, d] => Arg::Buffer {
            name: string(n)?,
            shape: list(v)?,
            data: if *d == "-" { None } else { Some(list(d)?) },
        },
        ["range", id, a] => Arg::Range {
            id: number(id)?,
            axis: match *a {
                "Loop" => Axis::Loop,
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
        ["scalar", i, dt] => Arg::ScalarParam {
            index: number(i)?,
            dtype: dtype(dt)?,
        },
        _ => return Err("invalid typed argument".into()),
    })
}
```

`src/uir.rs` の `fn validate_node`を置き換える。

```rust
// file="src/uir.rs" mode="replace" item="fn validate_node"
fn validate_node(u: &UOp) -> Result<()> {
    if crate::uop::is_binary(u.op) || crate::uop::is_unary(u.op) || u.op == Ops::Where {
        if u.src.iter().all(|s| {
            matches!(
                s.dtype(),
                DType::Bool | DType::I32 | DType::I64 | DType::F32
            ) && s.shape().is_ok_and(|shape| shape.is_empty())
        }) {
            crate::scalar::verify_node(u)?;
        } else {
            u.shape()?;
        }
    }
    match (&u.op, &u.arg) {
        (Ops::Const, _) => {
            crate::scalar::verify_node(u)?;
        }
        (Ops::Param, Arg::ScalarParam { .. }) => {
            crate::scalar::verify_node(u)?;
        }
        (Ops::Param, Arg::Slot { .. })
        | (Ops::Buffer, Arg::Buffer { .. })
        | (Ops::Alloc, Arg::Allocation { .. })
            if u.src.is_empty() =>
        {
            if u.op == Ops::Buffer {
                u.shape()?;
            }
        }
        (Ops::Param | Ops::Buffer | Ops::Alloc, _) => {
            return Err("invalid leaf metadata".into());
        }
        (Ops::Range, Arg::Range { .. })
            if u.src.len() == 2 && u.src.iter().all(|s| s.dtype() == DType::I64) => {}
        (Ops::Range, _) => return Err("invalid RANGE".into()),
        (Ops::End, Arg::None)
            if !u.src.is_empty() && u.src.last().is_some_and(|r| r.op == Ops::Range) => {}
        (Ops::End, _) => return Err("invalid END".into()),
        (Ops::Source, Arg::Text(_))
        | (Ops::Binary, Arg::Bytes(_))
        | (Ops::Invalid, Arg::Text(_))
            if u.src.is_empty() => {}
        (Ops::Source | Ops::Binary | Ops::Invalid, _) => {
            return Err("invalid source payload".into());
        }
        (Ops::Load, Arg::None | Arg::DType(_))
            if u.src.len() == 1 && u.src[0].dtype() == DType::Ptr => {}
        (Ops::Store, Arg::None | Arg::DType(_))
            if u.src.len() == 2 && u.src[0].dtype() == DType::Ptr => {}
        (Ops::Load | Ops::Store, _) => return Err("invalid memory operation".into()),
        (Ops::Linear, Arg::None) => {
            crate::codegen::verify_lowered(u)?;
            validate_linear_scope(u)?;
        }
        (Ops::Linear, _) => return Err("invalid LINEAR".into()),
        (Ops::Call, Arg::Id(count))
            if u.src.first().is_some_and(|b| b.op == Ops::Stack)
                && u.src.len().checked_sub(1) == Some(*count) => {}
        (Ops::Call, _) => {
            return Err("invalid CALL".into());
        }
        _ => {}
    }
    Ok(())
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
    Max,
    Lt,
    Eq,
    And,
    Or,
    Where,
    Stack,
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
}
```

`src/uop.rs` の `pub enum Arg`を置き換える。

```rust
// file="src/uop.rs" mode="replace" item="pub enum Arg"
pub enum Arg {
    None,
    Float(ConstFloat),
    I32(i32),
    Int(i64),
    Bool(bool),
    Buffer {
        name: String,
        shape: Vec<usize>,
        data: Option<Vec<u32>>,
    },
    Range {
        id: usize,
        axis: Axis,
    },
    Allocation {
        id: usize,
        dtype: DType,
        size: usize,
    },
    Id(usize),
    DType(DType),
    Text(String),
    Bytes(Vec<u8>),
    Slot {
        index: usize,
        shape: Vec<usize>,
    },
    ScalarParam {
        index: usize,
        dtype: DType,
    },
}
```

`src/uop.rs` の `pub struct Node`を置き換える。

```rust
// file="src/uop.rs" mode="replace" item="pub struct Node"
pub struct Node {
    pub id: u64,
    pub op: Ops,
    pub src: Vec<UOp>,
    pub arg: Arg,
    shape_cache: OnceCell<Result<Vec<usize>>>,
    dtype_cache: OnceCell<DType>,
}
```

`src/uop.rs` の `impl UOp` 内 の `pub fn new`を置き換える。

```rust
// file="src/uop.rs" mode="replace" within="impl UOp" item="pub fn new"
    pub fn new(op: Ops, src: Vec<UOp>, arg: Arg) -> Self {
        INTERN.with(|cell| {
            let mut intern = cell.borrow_mut();
            let key = Key {
                op,
                src: src.iter().map(|x| x.id).collect(),
                arg: arg.clone(),
            };
            if let Some(node) = intern.nodes.get(&key).and_then(Weak::upgrade) {
                return Self(node);
            }
            intern.next += 1;
            let node = Rc::new(Node {
                id: intern.next,
                op,
                src,
                arg,
                shape_cache: OnceCell::new(),
                dtype_cache: OnceCell::new(),
            });
            intern.nodes.insert(key, Rc::downgrade(&node));
            if intern.nodes.len() % 1024 == 0 {
                intern.nodes.retain(|_, v| v.strong_count() != 0);
            }
            Self(node)
        })
    }
```

`src/uop.rs` の `impl UOp` 内 の `fn infer_dtype`を置き換える。

```rust
// file="src/uop.rs" mode="replace" within="impl UOp" item="fn infer_dtype"
    fn infer_dtype(&self) -> DType {
        match self.op {
            Ops::Const => match self.arg {
                Arg::Float(_) => DType::F32,
                Arg::I32(_) => DType::I32,
                Arg::Int(_) => DType::I64,
                Arg::Bool(_) => DType::Bool,
                _ => DType::Void,
            },
            Ops::Param => match self.arg {
                Arg::ScalarParam { dtype, .. } => dtype,
                _ => DType::Ptr,
            },
            Ops::Buffer | Ops::Alloc => DType::Ptr,
            Ops::Index => {
                if self.src.first().is_some_and(|s| s.op == Ops::Call) {
                    self.src[0].dtype()
                } else {
                    DType::Ptr
                }
            }
            Ops::Range => DType::I64,
            Ops::Lt | Ops::Eq => DType::Bool,
            Ops::Cast | Ops::Bitcast => {
                if let Arg::DType(dt) = self.arg {
                    dt
                } else {
                    DType::Void
                }
            }
            Ops::Load => match self.arg {
                Arg::DType(dt) => dt,
                _ => DType::F32,
            },
            Ops::Call => self.src.first().map_or(DType::Void, |x| x.dtype()),
            Ops::Where => self.src.get(1).map_or(DType::Void, |x| x.dtype()),
            Ops::Store
            | Ops::End
            | Ops::Sink
            | Ops::Linear
            | Ops::Program
            | Ops::Source
            | Ops::Binary
            | Ops::Invalid => DType::Void,
            _ => self.src.first().map_or(DType::Void, |x| {
                if x.dtype() == DType::Ptr {
                    DType::F32
                } else {
                    x.dtype()
                }
            }),
        }
    }
```

`src/uop.rs` の `impl UOp` 内を追記する。

```rust
// file="src/uop.rs" mode="append" within="impl UOp"
    pub(crate) fn shape_uops(&self) -> Result<Vec<UOp>> {
        Ok(self
            .shape()?
            .into_iter()
            .map(|n| UOp::int(n as i64))
            .collect())
    }

    pub fn shape(&self) -> Result<Vec<usize>> {
        self.shape_cache.get_or_init(|| self.infer_shape()).clone()
    }

    fn infer_shape(&self) -> Result<Vec<usize>> {
        let err = || format!("invalid {:?} arguments", self.op);
        let source = || self.src.first().ok_or_else(err)?.shape();
        match (&self.op, &self.arg) {
            (Ops::Invalid, Arg::Text(s)) => Err(s.clone()),
            (Ops::Buffer, Arg::Buffer { shape, data, .. }) => {
                checked_numel(shape)?;
                if data.as_ref().is_some_and(|d| d.len() != numel(shape)) {
                    return Err("buffer data length disagrees with shape".into());
                }
                Ok(shape.clone())
            }
            (Ops::Param, Arg::Slot { .. }) => Ok(vec![]),
            (Ops::Param, Arg::ScalarParam { .. }) => Ok(vec![]),
            (Ops::Const, _) | (Ops::Range, _) | (Ops::Load, _) => Ok(vec![]),
            (Ops::Where, _) => {
                if self.src.len() != 3 {
                    return Err(err());
                }
                let s = broadcast_shape(&self.src[0].shape()?, &self.src[1].shape()?)?;
                broadcast_shape(&s, &self.src[2].shape()?)
            }
            (op, _) if is_binary(*op) => {
                if self.src.len() != 2 {
                    return Err(err());
                }
                broadcast_shape(&self.src[0].shape()?, &self.src[1].shape()?)
            }
            (op, _) if is_unary(*op) || matches!(op, Ops::After) => source(),
            _ => Err(err()),
        }
    }
```

`src/uop.rs`を追記する。

```rust
// file="src/uop.rs" mode="append"

use crate::Result;

pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

pub fn checked_numel(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |a, &b| {
        a.checked_mul(b)
            .filter(|&n| n <= i64::MAX as usize)
            .ok_or_else(|| "tensor shape overflow".into())
    })
}

pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let n = a.len().max(b.len());
    (0..n)
        .map(|i| {
            let x = if i + a.len() >= n {
                a[i + a.len() - n]
            } else {
                1
            };
            let y = if i + b.len() >= n {
                b[i + b.len() - n]
            } else {
                1
            };
            if x == y {
                Ok(x)
            } else if x == 1 {
                Ok(y)
            } else if y == 1 {
                Ok(x)
            } else {
                Err(format!("cannot broadcast {a:?} and {b:?}"))
            }
        })
        .collect()
}
```

`tests/loop_memory.rs` の `fn masked_vector_load_and_store_execute_real_ranges`を置き換える。

```rust
// file="tests/loop_memory.rs" mode="replace" item="fn masked_vector_load_and_store_execute_real_ranges"
fn masked_vector_load_and_store_execute_real_ranges() {
    let input = UOp::new(
        Ops::Param,
        vec![],
        Arg::Slot {
            index: 0,
            shape: vec![3],
        },
    );
    let output = UOp::new(
        Ops::Param,
        vec![],
        Arg::Slot {
            index: 1,
            shape: vec![3],
        },
    );
    let range = UOp::range(UOp::int(4), 0, Axis::Loop);
    let valid = range.lt(&UOp::int(3));
    let load = input.index(range.clone(), valid.clone()).load();
    let store = output.index(range.clone(), valid).store(load.mul(&UOp::f32(2.)));
    let linear = UOp::linear(
        vec![range.clone(), store, UOp::new(Ops::End, vec![range], Arg::None),],
    );
    let source = codegen::render(&linear, "vector_entry").unwrap();
    let native = unsafe { CpuProgram::compile(&source, "vector_entry") }.unwrap();
    let input = [2.0f32, 3., 4.];
    let mut output = [0.0f32, 0., 0., 12345.];
    let mut args = (input.as_ptr() as u64).to_ne_bytes().to_vec();
    args.extend((output.as_mut_ptr() as u64).to_ne_bytes());
    unsafe { native.run(&mut args) }.unwrap();
    assert_eq!(output, [4., 6., 8., 12345.]);
}
```

`tests/loops_uop.rs` の `fn scope_and_unconditional_memory_bounds_are_checked`を置き換える。

```rust
// file="tests/loops_uop.rs" mode="replace" item="fn scope_and_unconditional_memory_bounds_are_checked"
fn scope_and_unconditional_memory_bounds_are_checked() {
    let first = UOp::range(UOp::int(3), 0, Axis::Loop);
    let second = UOp::range(UOp::int(4), 1, Axis::Loop);
    let invalid = UOp::linear(vec![
        first.clone(),
        second.clone(),
        UOp::new(Ops::End, vec![first.clone()], Arg::None),
    ]);
    assert!(codegen::verify_lowered(&invalid).is_err());
    assert!(codegen::verify_lowered(&UOp::linear(vec![first])).is_err());
    let buffer = UOp::new(
        Ops::Param,
        vec![],
        Arg::Slot {
            index: 0,
            shape: vec![3],
        },
    );
    let outside = buffer.index(UOp::int(3), UOp::bool(true));
    assert!(codegen::verify_lowered(&UOp::linear(vec![outside])).is_err());
}
```

`tests/memory.rs` の `fn ptr`を置き換える。

```rust
// file="tests/memory.rs" mode="replace" item="fn ptr"
fn ptr(index: usize) -> UOp {
    UOp::new(Ops::Param, vec![], Arg::Slot { index, shape: vec![1] })
}
```

`tests/reduction.rs` の `fn row_sum`を置き換える。

```rust
// file="tests/reduction.rs" mode="replace" item="fn row_sum"
fn row_sum(width: usize, values: &[f32]) -> Vec<f32> {
    let input = UOp::new(
        Ops::Param,
        vec![],
        Arg::Slot {
            index: 0,
            shape: vec![2, width],
        },
    );
    let output = UOp::new(
        Ops::Param,
        vec![],
        Arg::Slot {
            index: 1,
            shape: vec![2],
        },
    );
    let row = UOp::range(UOp::int(2), 0, Axis::Loop);
    let accumulator = UOp::new(
        Ops::Alloc,
        vec![],
        Arg::Allocation {
            id: 0,
            dtype: DType::F32,
            size: 1,
        },
    );
    let init = accumulator.store(UOp::f32(0.));
    let col = UOp::range(UOp::int(width as i64), 1, Axis::Reduce);
    let index = row.mul(&UOp::int(width as i64)).add(&col);
    let value = input.index(index, UOp::bool(true)).load();
    let state = accumulator.after(vec![init.clone(), col.clone()]);
    let update = state.store(state.load().add(&value));
    let end = UOp::new(Ops::End, vec![update.clone(), col.clone()], Arg::None);
    let result = accumulator.after(vec![end.clone()]).load();
    let store = output.index(row.clone(), UOp::bool(true)).store(result.clone());
    let outer_end = UOp::new(Ops::End, vec![store.clone(), row.clone()], Arg::None);
    let linear = UOp::linear(
        vec![row, accumulator, init, col, update, end, result, store, outer_end,],
    );
    let source = codegen::render(&linear, "row_sum").unwrap();
    let native = unsafe { CpuProgram::compile(&source, "row_sum") }.unwrap();
    let mut output = vec![0.0f32; 2];
    let mut args = (values.as_ptr() as u64).to_ne_bytes().to_vec();
    args.extend((output.as_mut_ptr() as u64).to_ne_bytes());
    unsafe { native.run(&mut args) }.unwrap();
    output
}
```

`tests/reduction_max.rs` の `fn max_reduction_and_empty_identity`を置き換える。

```rust
// file="tests/reduction_max.rs" mode="replace" item="fn max_reduction_and_empty_identity"
fn max_reduction_and_empty_identity() {
    for (width, data, expected) in [
        (3, vec![- 7.0f32, - 2., - 9., 4., 3., 2.], vec![- 2., 4.]),
        (0, vec![], vec![f32::NEG_INFINITY; 2]),
    ] {
        let p = |index, n| {
            UOp::new(Ops::Param, vec![], Arg::Slot { index, shape: vec![n] })
        };
        let linear = row_reduce(p(0, 2 * width), p(1, 2), 2, width, Ops::Max).unwrap();
        let source = codegen::render(&linear, "maximum").unwrap();
        let program = unsafe { CpuProgram::compile(&source, "maximum") }.unwrap();
        let mut out = vec![0f32; 2];
        let mut args = (data.as_ptr() as u64).to_ne_bytes().to_vec();
        args.extend((out.as_mut_ptr() as u64).to_ne_bytes());
        unsafe { program.run(&mut args) }.unwrap();
        assert_eq!(out, expected);
    }
}
```

`tests/tensor.rs`を保存する。

```rust
// file="tests/tensor.rs" mode="write"
use chibigrad::{Buffer, Tensor};
#[test]
fn lazy_tensor_shape_broadcast_and_rebinding() {
    let x = Tensor::parameter("x", &[2, 2]);
    let y = x
        .mul(&Tensor::scalar(3.))
        .add(&Tensor::from_slice(&[1., 2.], &[2]));
    assert_eq!(y.shape().unwrap(), [2, 2]);
    let p = y.compile().unwrap();
    let out = p
        .run(
            &[Buffer::new(vec![2, 2], vec![1., 2., 3., 4.]).unwrap()],
            &std::collections::HashMap::new(),
        )
        .unwrap();
    assert_eq!(out[0].data, [4., 8., 10., 14.]);
    assert!(Tensor::from_slice(&[1.], &[2]).compile().is_err());
}
```

```sh
# check step=08.01
cargo test --offline
```

[2,2] と [2] のブロードキャストが [4,8,10,14] になり、データ長の不一致を拒否する。

## 08.02 Tensor の軸を畳み込む

REDUCE に演算、軸、keepdim を持たせる。shape 検証で結果の次元を定め、各出力座標から前章の局所蓄積と内側 RANGE を生成する。

`src/codegen.rs` の `struct Lower`を置き換える。

```rust
// file="src/codegen.rs" mode="replace" item="struct Lower"
struct Lower {
    ops: Vec<UOp>,
    seen: HashSet<u64>,
    next_range: usize,
    next_alloc: usize,
    cache: HashMap<(u64, Vec<u64>, u64), UOp>,
}
```

`src/codegen.rs` の `impl Lower` 内 の `fn scalar`を置き換える。

```rust
// file="src/codegen.rs" mode="replace" within="impl Lower" item="fn scalar"
    fn scalar(&mut self, u: &UOp, indices: &[UOp], valid: &UOp) -> Result<UOp> {
        let key = (u.id, indices.iter().map(|i| i.id).collect(), valid.id);
        if let Some(v) = self.cache.get(&key) {
            return Ok(v.clone());
        }
        let result = match (&u.op, &u.arg) {
            (Ops::Const, _) => self.emit(u.clone()),
            (Ops::Buffer, _) | (Ops::Param, _) => {
                let shape = u.shape_uops()?;
                let idx = simplify(&offset(indices, &shape))?;
                let gate = simplify(valid)?;
                self.emit(u.index(idx, gate).load())
            }
            (Ops::Reduce, Arg::Reduction { op, axes, keepdim }) => {
                let shape = u.src[0].shape_uops()?;
                let mut input = vec![];
                let mut output_axis = 0;
                for i in 0..shape.len() {
                    if axes.contains(&i) {
                        input.push(UOp::int(0));
                        if *keepdim {
                            output_axis += 1;
                        }
                    } else {
                        input.push(indices[output_axis].clone());
                        output_axis += 1;
                    }
                }
                if axes.is_empty() {
                    self.scalar(&u.src[0], &input, valid)?
                } else {
                    let identity = if *op == Ops::Add {
                        0.
                    } else {
                        f32::NEG_INFINITY
                    };
                    let (alloc, init) = self.accumulator(UOp::f32(identity));
                    let saved = self.cache.clone();
                    let old_seen = self.seen.clone();
                    let mut ranges = vec![];
                    for &a in axes {
                        let r = self.range(shape[a].clone(), Axis::Reduce);
                        input[a] = r.clone();
                        ranges.push(r);
                    }
                    let value = self.scalar(&u.src[0], &input, valid)?;
                    let mut deps = vec![init];
                    deps.extend(ranges.clone());
                    let ptr = alloc.after(deps);
                    let current = self.emit(ptr.load());
                    let update = self.emit(ptr.store(current.alu(*op, &[value])));
                    let mut end = update;
                    for r in ranges.iter().rev() {
                        end = self.close(r, end);
                    }
                    self.cache = saved;
                    self.seen = old_seen;
                    self.seen.insert(alloc.id);
                    self.seen.insert(end.id);
                    self.emit(alloc.after(vec![end]).load())
                }
            }
            (op, _) if is_unary(*op) => {
                let x = self.scalar(&u.src[0], indices, valid)?;
                self.emit(UOp::new(*op, vec![x], u.arg.clone()))
            }
            (op, _) if is_binary(*op) || *op == Ops::Where => {
                let mut src = vec![];
                for x in &u.src {
                    src.push(self.scalar(x, &broadcast(indices, &x.shape_uops()?), valid)?)
                }
                self.emit(UOp::new(*op, src, u.arg.clone()))
            }
            _ => return Err(format!("rangeify cannot lower {:?}", u.op)),
        };
        self.cache.insert(key, result.clone());
        Ok(result)
    }
```

`src/codegen.rs` の `impl Lower` 内を追記する。

```rust
// file="src/codegen.rs" mode="append" within="impl Lower"
    fn accumulator(&mut self, initial: UOp) -> (UOp, UOp) {
        let alloc = UOp::new(
            Ops::Alloc,
            vec![],
            Arg::Allocation {
                id: self.next_alloc,
                dtype: DType::F32,
                size: 1,
            },
        );
        self.next_alloc += 1;
        self.emit(alloc.clone());
        let init = self.emit(alloc.store(initial));
        (alloc, init)
    }
```

`src/codegen.rs` の `pub fn rangeify`を置き換える。

```rust
// file="src/codegen.rs" mode="replace" item="pub fn rangeify"
pub fn rangeify(sink: &UOp) -> Result<UOp> {
    if sink.op != Ops::Sink {
        return Err("rangeify expects SINK".into());
    }
    let sink = normalize(sink)?;
    let mut lower = Lower {
        ops: vec![],
        seen: HashSet::new(),
        next_range: 0,
        next_alloc: 0,
        cache: HashMap::new(),
    };
    for store in &sink.src {
        if store.op != Ops::Store || store.src.len() != 2 {
            return Err("rangeify expects STORE(buffer, tensor)".into());
        }
        let target = &store.src[0];
        let value = &store.src[1];
        let shape = value.shape_uops()?;
        if target.shape_uops()? != shape {
            return Err("output buffer shape mismatch".into());
        }
        let r = lower.range(product(&shape), Axis::Loop);
        let coords = coords(&r, &shape);
        let result = lower.scalar(value, &coords, &UOp::bool(true))?;
        let output = lower.emit(target.index(r.clone(), UOp::bool(true)).store(result));
        lower.close(&r, output);
        lower.cache.clear();
    }
    let linear = UOp::linear(lower.ops);
    verify_lowered(&linear)?;
    Ok(linear)
}
```

`src/tensor.rs` の `impl Tensor` 内を追記する。

```rust
// file="src/tensor.rs" mode="append" within="impl Tensor"
    fn invalid(message: impl Into<String>) -> Self {
        Self::from_uop(UOp::invalid(message))
    }

    pub fn reduce(&self, axes: &[usize], keepdim: bool, op: Ops) -> Self {
        Self::from_uop(UOp::new(
            Ops::Reduce,
            vec![self.uop.clone()],
            Arg::Reduction {
                op,
                axes: axes.to_vec(),
                keepdim,
            },
        ))
    }

    pub fn sum(&self) -> Self {
        match self.shape() {
            Ok(s) => self.reduce(&(0..s.len()).collect::<Vec<_>>(), false, Ops::Add),
            Err(e) => Self::invalid(e),
        }
    }

    pub fn sum_axes(&self, axes: &[usize], keepdim: bool) -> Self {
        self.reduce(axes, keepdim, Ops::Add)
    }

    pub fn max(&self, axes: &[usize], keepdim: bool) -> Self {
        self.reduce(axes, keepdim, Ops::Max)
    }

    pub fn mean(&self, axes: &[usize], keepdim: bool) -> Self {
        match self.shape() {
            Ok(s) if axes.iter().all(|&a| a < s.len()) => self.sum_axes(axes, keepdim).div(
                &Self::scalar(axes.iter().map(|&a| s[a]).product::<usize>() as f32),
            ),
            _ => Self::invalid("invalid mean axes"),
        }
    }
```

`src/uir.rs` の `fn op`を置き換える。

```rust
// file="src/uir.rs" mode="replace" item="fn op"
fn op(s: &str) -> Result<Ops> {
    [
        Ops::Invalid,
        Ops::Const,
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
        Ops::Max,
        Ops::Lt,
        Ops::Eq,
        Ops::And,
        Ops::Or,
        Ops::Where,
        Ops::Stack,
        Ops::Reduce,
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
    ]
    .into_iter()
    .find(|o| format!("{o:?}") == s)
    .ok_or("unknown operation".into())
}
```

`src/uir.rs` の `fn encode_arg`を置き換える。

```rust
// file="src/uir.rs" mode="replace" item="fn encode_arg"
fn encode_arg(a: &Arg) -> String {
    match a {
        Arg::None => "none".into(),
        Arg::Float(v) => format!("float;{:016x}", v.value().to_bits()),
        Arg::I32(v) => format!("i32;{v}"),
        Arg::Int(v) => format!("i64;{v}"),
        Arg::Bool(v) => format!("bool;{}", *v as u8),
        Arg::Reduction { op, axes, keepdim } => {
            format!("reduction;{op:?};{};{}", numbers(axes), *keepdim as u8)
        }
        Arg::Buffer { name, shape, data } => {
            format!(
                "buffer;{};{};{}",
                hex(name.as_bytes()),
                numbers(shape),
                data.as_ref().map(|v| numbers(v)).unwrap_or("-".into())
            )
        }
        Arg::Range { id, axis } => format!("range;{id};{axis:?}"),
        Arg::Allocation { id, dtype, size } => {
            format!("allocation;{id};{dtype:?};{size}")
        }
        Arg::Id(v) => format!("id;{v}"),
        Arg::DType(dt) => format!("dtype;{dt:?}"),
        Arg::Text(s) => format!("text;{}", hex(s.as_bytes())),
        Arg::Bytes(v) => format!("bytes;{}", hex(v)),
        Arg::Slot { index, shape } => format!("slot;{index};{}", numbers(shape)),
        Arg::ScalarParam { index, dtype } => format!("scalar;{index};{dtype:?}"),
    }
}
```

`src/uir.rs` の `fn decode_arg`を置き換える。

```rust
// file="src/uir.rs" mode="replace" item="fn decode_arg"
fn decode_arg(s: &str) -> Result<Arg> {
    let fields: Vec<_> = s.split(';').collect();
    Ok(match fields.as_slice() {
        ["none"] => Arg::None,
        ["float", v] if v.len() == 16 && v.bytes().all(|b| b.is_ascii_hexdigit()) => {
            Arg::Float(crate::uop::ConstFloat::new(f64::from_bits(
                u64::from_str_radix(v, 16).map_err(|_| "invalid float bits")?
            )))
        }
        ["i32", v] => Arg::I32(number(v)?),
        ["i64", v] => Arg::Int(number(v)?),
        ["bool", v] => Arg::Bool(boolean(v)?),
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
    Max,
    Lt,
    Eq,
    And,
    Or,
    Where,
    Stack,
    Reduce,
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
}
```

`src/uop.rs` の `pub enum Arg`を置き換える。

```rust
// file="src/uop.rs" mode="replace" item="pub enum Arg"
pub enum Arg {
    None,
    Float(ConstFloat),
    I32(i32),
    Int(i64),
    Bool(bool),
    Reduction {
        op: Ops,
        axes: Vec<usize>,
        keepdim: bool,
    },
    Buffer {
        name: String,
        shape: Vec<usize>,
        data: Option<Vec<u32>>,
    },
    Range {
        id: usize,
        axis: Axis,
    },
    Allocation {
        id: usize,
        dtype: DType,
        size: usize,
    },
    Id(usize),
    DType(DType),
    Text(String),
    Bytes(Vec<u8>),
    Slot {
        index: usize,
        shape: Vec<usize>,
    },
    ScalarParam {
        index: usize,
        dtype: DType,
    },
}
```

`src/uop.rs` の `impl UOp` 内 の `fn infer_shape`を置き換える。

```rust
// file="src/uop.rs" mode="replace" within="impl UOp" item="fn infer_shape"
    fn infer_shape(&self) -> Result<Vec<usize>> {
        let err = || format!("invalid {:?} arguments", self.op);
        let source = || self.src.first().ok_or_else(err)?.shape();
        match (&self.op, &self.arg) {
            (Ops::Invalid, Arg::Text(s)) => Err(s.clone()),
            (Ops::Buffer, Arg::Buffer { shape, data, .. }) => {
                checked_numel(shape)?;
                if data.as_ref().is_some_and(|d| d.len() != numel(shape)) {
                    return Err("buffer data length disagrees with shape".into());
                }
                Ok(shape.clone())
            }
            (Ops::Param, Arg::Slot { .. }) => Ok(vec![]),
            (Ops::Param, Arg::ScalarParam { .. }) => Ok(vec![]),
            (Ops::Const, _) | (Ops::Range, _) | (Ops::Load, _) => Ok(vec![]),
            (Ops::Reduce, Arg::Reduction { op, axes, keepdim }) => {
                let old = source()?;
                if !matches!(op, Ops::Add | Ops::Max)
                    || axes.iter().any(|&a| a >= old.len())
                    || axes.iter().copied().collect::<HashSet<_>>().len() != axes.len()
                {
                    return Err("invalid reduction axes or operation".into());
                }
                Ok(old
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &n)| {
                        if axes.contains(&i) {
                            keepdim.then_some(1)
                        } else {
                            Some(n)
                        }
                    })
                    .collect())
            }
            (Ops::Where, _) => {
                if self.src.len() != 3 {
                    return Err(err());
                }
                let s = broadcast_shape(&self.src[0].shape()?, &self.src[1].shape()?)?;
                broadcast_shape(&s, &self.src[2].shape()?)
            }
            (op, _) if is_binary(*op) => {
                if self.src.len() != 2 {
                    return Err(err());
                }
                broadcast_shape(&self.src[0].shape()?, &self.src[1].shape()?)
            }
            (op, _) if is_unary(*op) || matches!(op, Ops::After) => source(),
            _ => Err(err()),
        }
    }
```

`tests/tensor_native.rs`を保存する。

```rust
// file="tests/tensor_native.rs" mode="write"
use chibigrad::{Buffer, Ops, Tensor};
use std::collections::HashMap;

#[test]
fn fused_reduction_rebinds_native_inputs() {
    let x = Tensor::parameter("x", &[4]);
    let w = Tensor::parameter("w", &[4]);
    let y = x.mul(&w).add(&Tensor::scalar(1.)).sum();
    let exe = y.compile().unwrap();
    assert_eq!(exe.kernel_count(), 1);
    for p in exe.programs() {
        assert!(p.src.iter().any(|s| s.op == Ops::Binary));
        assert!(!p.src[1].src.iter().any(|u| matches!(u.op, Ops::Reduce)));
    }
    let run = |xs: &[f32]| {
        let bufs = exe
            .inputs()
            .iter()
            .map(|i| {
                Buffer::new(
                    i.shape.clone(),
                    if i.name == "x" {
                        xs.to_vec()
                    } else {
                        vec![4.; 4]
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        exe.run(&bufs, &HashMap::new()).unwrap()[0].data.clone()
    };
    assert_eq!(run(&[1., 2., 3., 4.]), [44.]);
    assert_eq!(run(&[2., 2., 2., 2.]), [36.]);
}
#[test]
fn empty_reduction_has_its_identity() {
    assert_eq!(Tensor::from_slice(&[], &[0]).sum().to_vec().unwrap(), [0.]);
    assert_eq!(
        Tensor::from_slice(&[], &[0])
            .max(&[0], false)
            .to_vec()
            .unwrap(),
        [f32::NEG_INFINITY]
    );
}
```

```sh
# check step=08.02
cargo test --offline
```

入力を差し替えた積和が 44 と 36 を返し、空テンソルの和と最大値が各単位元になる。

## 章末の累積確認

前の章までのテストとexampleも再ビルドします。

```sh
# check
cargo test --quiet --all-targets
```

## 要件との対応

### 08-001 shape を scalar と rank 1 に限定して導出する。

<!-- todo:08-001 -->

関連する実装と確認は本章の `08.01`, `08.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 08-002 同じ長さの入力二つから output RANGE を作る。

<!-- todo:08-002 -->

関連する実装と確認は本章の `08.01`, `08.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 08-003 ALU の座標を入力へ伝え、INDEX と LOAD を生成する。

<!-- todo:08-003 -->

関連する実装と確認は本章の `08.01`, `08.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 08-004 高水準加算から `[1,2,3]+[10,20,30]=[11,22,33]` を得る。

<!-- todo:08-004 -->

関連する実装と確認は本章の `08.01`, `08.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 08-005 長さ不一致を kernel 生成前に拒否する。

<!-- todo:08-005 -->

関連する実装と確認は本章の `08.01`, `08.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 08-006 scalar と長さ 1 の入力に EXPAND を加える。

<!-- todo:08-006 -->

関連する実装と確認は本章の `08.01`, `08.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 08-007 入力の singleton 座標を 0 にする。

<!-- todo:08-007 -->

関連する実装と確認は本章の `08.01`, `08.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 08-008 MUL の出力を新規 buffer に保存し、次の ADD が読むようにする。

<!-- todo:08-008 -->

関連する実装と確認は本章の `08.01`, `08.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 08-009 `x*2+1` を `[3,5,7]` にする。

<!-- todo:08-009 -->

関連する実装と確認は本章の `08.01`, `08.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 08-010 `[1]` の bias `[10]` を展開し、`x+b` を `[11,12,13]` にする。

<!-- todo:08-010 -->

関連する実装と確認は本章の `08.01`, `08.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 08-011 REDUCE の入力 shape と出力 scalar を検証する。

<!-- todo:08-011 -->

関連する実装と確認は本章の `08.01`, `08.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 08-012 reduction 軸 k と accumulator の初期化を生成する。

<!-- todo:08-012 -->

関連する実装と確認は本章の `08.01`, `08.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 08-013 前章の逐次累積と同じ低水準 IR へ変換する。

<!-- todo:08-013 -->

関連する実装と確認は本章の `08.01`, `08.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 08-014 `[1,2,3]` の総和を `6` にする。

<!-- todo:08-014 -->

関連する実装と確認は本章の `08.01`, `08.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 08-015 `sum(x*2)` を中間 MUL と別 reduction kernel で `12` にする。

<!-- todo:08-015 -->

関連する実装と確認は本章の `08.01`, `08.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 08-016 REDUCE MAX の初期値を f32 の負の無限大にする。

<!-- todo:08-016 -->

関連する実装と確認は本章の `08.01`, `08.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 08-017 第三章の MAX の特殊値契約を interpreter と C で維持する。

<!-- todo:08-017 -->

関連する実装と確認は本章の `08.01`, `08.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 08-018 `max([-5,-2,-7])=-2` を既存 CPU 経路で実行する。

<!-- todo:08-018 -->

関連する実装と確認は本章の `08.01`, `08.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 08-019 同じノードを二つの出力 RANGE から lower する fixture を作る。

<!-- todo:08-019 -->

関連する実装と確認は本章の `08.01`, `08.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 08-020 軸の漏れ、違う座標の誤共有、scalar の保存長の誤りを検証する。

<!-- todo:08-020 -->

関連する実装と確認は本章の `08.01`, `08.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 08-021 高水準の `x*2+1` が `[3,5,7]` になる。

<!-- todo:08-021 -->

関連する実装と確認は本章の `08.01`, `08.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 08-022 affine の実行 kernel は二つで、中間 buffer が一つある。

<!-- todo:08-022 -->

関連する実装と確認は本章の `08.01`, `08.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 08-023 総和が `6`、`sum(x*2)` が `12`、MAX が `-2` になる。

<!-- todo:08-023 -->

関連する実装と確認は本章の `08.01`, `08.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 08-024 scalar の出力 storage は一要素、shape は `[]` になる。

<!-- todo:08-024 -->

関連する実装と確認は本章の `08.01`, `08.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 08-025 backend 入力に EXPAND と高水準 REDUCE が残らない。

<!-- todo:08-025 -->

関連する実装と確認は本章の `08.01`, `08.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 08-026 各 kernel の RANGE は閉じ、INDEX はその kernel の座標を使う。

<!-- todo:08-026 -->

関連する実装と確認は本章の `08.01`, `08.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 08-027 shape 不一致と不正 EXPAND は実行前にエラーになる。

<!-- todo:08-027 -->

関連する実装と確認は本章の `08.01`, `08.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。
