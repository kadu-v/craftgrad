# 25. 実機で行列命令と RMSNorm の capability を確認する

前の節で動かしたコードに、今回使う演算と処理を追加します。各節の確認を実行してから次へ進みます。

参照するUOpの構成は [tinygrad c22edff](https://github.com/tinygrad/tinygrad/blob/c22edff14c79310a9b2df62a78819cf77c8289fa/tinygrad/uop/ops.py) です。最初から全機能を実装せず、使う節で増やします。

## 25.01 RMSNorm を協調 reduction へ下ろす

1行の二乗和を各 lane で累積し、shared memory と barrier で木状に縮約する。平方根の逆数と learned weight を掛ける。幅37と local16で末尾の predicate も実行する。

`src/gpu.rs`を追記する。

```rust
// file="src/gpu.rs" mode="append"

pub fn rms_norm(
    rows: usize,
    width: usize,
    local: usize,
    epsilon: f32,
) -> Result<(UOp, Launch), String> {
    if rows == 0 || width == 0 || !local.is_power_of_two() || !epsilon.is_finite() || epsilon <= 0.
    {
        return Err("invalid RMSNorm shape, local size or epsilon".into());
    }
    let mut b = Builder::new();
    let input = b.param(0, rows.checked_mul(width).ok_or("RMSNorm overflow")?);
    let weights = b.param(1, width);
    let output = b.param(2, rows * width);
    let lane = b.special("lid0");
    let row = b.special("group0");
    let shared = b.alloc(local);
    let acc = b.alloc(1);
    let init = b.store(&acc, UOp::int(0), UOp::f32(0.), UOp::bool(true));
    let r = b.range(width.div_ceil(local));
    let col = r.mul(&UOp::int(local as i64)).add(&lane);
    let v = b.load(
        &input,
        row.mul(&UOp::int(width as i64)).add(&col),
        col.lt(&UOp::int(width as i64)),
    );
    let old = b.load(&acc.after(vec![init]), UOp::int(0), UOp::bool(true));
    b.store(&acc, UOp::int(0), old.add(&v.mul(&v)), UOp::bool(true));
    let end = b.end(&r);
    let sum = b.load(&acc.after(vec![end]), UOp::int(0), UOp::bool(true));
    let stored = b.store(&shared, lane.clone(), sum, UOp::bool(true));
    let mut dep = b.barrier(stored);
    for step in (0..local.ilog2()).rev().map(|s| 1usize << s) {
        let valid = lane.lt(&UOp::int(step as i64));
        let x = b.load(
            &shared.after(vec![dep.clone()]),
            lane.clone(),
            valid.clone(),
        );
        let y = b.load(
            &shared.after(vec![dep]),
            lane.add(&UOp::int(step as i64)),
            valid.clone(),
        );
        let st = b.store(&shared, lane.clone(), x.add(&y), valid);
        dep = b.barrier(st);
    }
    let sum = b.load(&shared.after(vec![dep]), UOp::int(0), UOp::bool(true));
    let scale = b.emit(
        UOp::f32(1.).div(
            &sum.div(&UOp::f32(width as f32))
                .add(&UOp::f32(epsilon))
                .alu(Ops::Sqrt, &[]),
        ),
    );
    let r = b.range(width.div_ceil(local));
    let col = r.mul(&UOp::int(local as i64)).add(&lane);
    let valid = col.lt(&UOp::int(width as i64));
    let index = row.mul(&UOp::int(width as i64)).add(&col);
    let x = b.load(&input, index.clone(), valid.clone());
    let w = b.load(&weights, col, valid.clone());
    b.store(&output, index, x.mul(&scale).mul(&w), valid);
    b.end(&r);
    Ok((
        b.finish(),
        Launch {
            groups: [rows, 1, 1],
            local: [local, 1, 1],
        },
    ))
}
```

`tests/hardware_rms.rs`を保存する。

```rust
// file="tests/hardware_rms.rs" mode="write"
mod support;
use chibigrad::gpu;
static GPU_TEST:std::sync::Mutex<()>=std::sync::Mutex::new(());
fn check_rms(run:fn(&chibigrad::UOp,gpu::Launch,&[Vec<f32>])->Vec<f32>){
 let input=(0..74).map(|i|(i%7) as f32-3.).collect::<Vec<_>>();
 let weights=(0..37).map(|i|1.+i as f32/37.).collect::<Vec<_>>();
 let expected=input.chunks(37).flat_map(|row|{let scale=(row.iter().map(|x|x*x).sum::<f32>()/37.+1e-5).sqrt().recip();row.iter().zip(&weights).map(move |(x,w)|x*scale*w)}).collect::<Vec<_>>();
 let (body,launch)=gpu::rms_norm(2,37,16,1e-5).unwrap();
 let got=run(&body,launch,&[input,weights,vec![0.;74]]);
 for (a,b) in got.iter().zip(expected){assert!((a-b).abs()<1e-5,"RMSNorm {a} vs {b}");}
 assert!(gpu::rms_norm(0,37,16,1e-5).is_err());assert!(gpu::rms_norm(2,37,15,1e-5).is_err());
}
#[cfg(all(feature="metal",target_os="macos"))]
#[test]
fn metal_rms_norm_tail_and_weight(){let _g=GPU_TEST.lock().unwrap();check_rms(support::run_metal);}
#[cfg(feature="opencl")]
#[test]
fn opencl_rms_norm_tail_and_weight(){let _g=GPU_TEST.lock().unwrap();check_rms(support::run_cl);}
```

```sh
# check step=25.01 requires=metal,opencl
cargo test --offline --features metal,opencl --test hardware_rms
```

Metal と OpenCL の2行37列の出力が独立 CPU 参照に1e-5以内で一致する。不正な shape と local size を拒否する。

## 25.02 Matrix dtype と WMMA で GEMM を実行する

8x8 matrix 値を Load と Store の dtype に持たせ、WMMA の入力・出力を共通 UOp で表す。shared memory を介して13x11x17の末尾をゼロ埋めし、Metal の simdgroup matrix 命令で積和する。

`src/codegen.rs` の `fn ctype`を置き換える。

```rust
// file="src/codegen.rs" mode="replace" item="fn ctype"
fn ctype(dt: DType) -> &'static str {
    match dt {
        DType::F32 => "float",
        DType::MatrixF32x8x8 => "matrix_float8x8",
        DType::F32x4 => "float4",
        DType::I32 => "int",
        DType::I64 => "long long",
        DType::Bool => "int",
        DType::Ptr => "void*",
        DType::Void => "void",
    }
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
    if linear.src.iter().any(|u|u.dtype()==DType::MatrixF32x8x8){return Err("matrix kernels require the Metal renderer".into());}
    let sig = signature(linear)?;
    let mut text = String::from(
        "extern float exp2f(float);\nextern float log2f(float);\nextern float sinf(float);\nextern float sqrtf(float);\nextern float fmaxf(float,float);\nstatic float chibi_float(unsigned int x) { union { unsigned int u; float f; } b; b.u=x; return b.f; }\nstatic long long chibi_div(long long a,long long b) { long long q=a/b, r=a%b; return q-((r!=0)&&((r<0)!=(b<0))); }\nstatic long long chibi_mod(long long a,long long b) { long long r=a%b; return r+(((r!=0)&&((r<0)!=(b<0)))?b:0); }\n",
    );
    text.push_str(
        "static int chibi_int(unsigned int x) { union { unsigned int u; int i; } b; b.u=x; return b.i; }\nstatic unsigned int chibi_bits(float x) { union { unsigned int u; float f; } b; b.f=x; return b.u; }\nstatic int chibi_shr(int a,int b) { unsigned int u=(unsigned int)a; return chibi_int((u>>b)|((a<0&&b)?(~0u<<(32-b)):0u)); }\nstatic long long chibi_shr64(long long a,long long b) { unsigned long long u=(unsigned long long)a; return (long long)((u>>b)|((a<0&&b)?(~0ull<<(64-b)):0ull)); }\nstatic float chibi_max(float a,float b) { if(a!=a||b!=b)return chibi_float(2143289344u); if(a==0.0f&&b==0.0f)return 0.0f; return a>b?a:b; }\n",
    );
    let mut functions: HashMap<String, (DType, Vec<DType>)> = HashMap::new();
    for u in &linear.src {
        let (body, args) = if u.op == Ops::Call {
            let b = u.src.first().ok_or("CALL lacks body")?;
            (b, &u.src[1..])
        } else if u.op == Ops::CustomFunction && !u.src.is_empty() {
            (u, &u.src[..])
        } else {
            continue;
        };
        let Arg::Function { name, dtype } = &body.arg else {
            continue;
        };
        if !identifier(name) {
            return Err("invalid C function identifier".into());
        }
        let types = args.iter().map(UOp::dtype).collect::<Vec<_>>();
        if functions
            .get(name)
            .is_some_and(|old| old != &(*dtype, types.clone()))
        {
            return Err(format!("inconsistent callback signature for {name}"));
        }
        functions.insert(name.clone(), (*dtype, types));
    }
    let mut function_names = functions.keys().collect::<Vec<_>>();
    function_names.sort();
    for name in function_names {
        let (ret, args) = &functions[name];
        text.push_str(&format!(
            "extern {} {}({});\n",
            ctype(*ret),
            name,
            if args.is_empty() {
                "void".into()
            } else {
                args.iter().map(|d| ctype(*d)).collect::<Vec<_>>().join(",")
            }
        ));
    }
    text.push_str("typedef float float4 __attribute__((ext_vector_type(4),aligned(1)));\n");
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
    for (slot, u) in &sig.variables {
        let name = format!("v{}", u.id);
        text.push_str(&format!("  long long {name}=((long long*)args)[{slot}];\n"));
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
            (Ops::Buffer, _) | (Ops::Param, _) | (Ops::Variable, _) => continue,
            (Ops::Bind, _) => at(1)?,
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
            (Ops::Exp2, _) => format!("exp2f({})", at(0)?),
            (Ops::Log2, _) => format!("log2f({})", at(0)?),
            (Ops::Sin, _) => format!("sinf({})", at(0)?),
            (Ops::Sqrt, _) => format!("sqrtf({})", at(0)?),
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
            (Ops::Call, _) => {
                let body = u.src.first().ok_or("CALL body missing")?;
                let Arg::Function { name, .. } = &body.arg else {
                    return Err("scalar CALL must target CUSTOM_FUNCTION".into());
                };
                format!(
                    "{}({})",
                    name,
                    u.src[1..]
                        .iter()
                        .map(get)
                        .collect::<Result<Vec<_>>>()?
                        .join(",")
                )
            }
            (Ops::CustomFunction, Arg::Function { name, .. }) => {
                if u.src.is_empty() {
                    values.insert(u.id, name.clone());
                    continue;
                }
                format!(
                    "{}({})",
                    name,
                    u.src.iter().map(get).collect::<Result<Vec<_>>>()?.join(",")
                )
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

`src/gpu.rs`を追記する。

```rust
// file="src/gpu.rs" mode="append"

pub fn matrix_gemm(m: usize, n: usize, k: usize) -> Result<(UOp, Launch), String> {
    if m == 0 || n == 0 || k == 0 {
        return Err("matrix GEMM requires nonzero dimensions".into());
    }
    let mut b = Builder::new();
    let a = b.param(0, m.checked_mul(k).ok_or("GEMM overflow")?);
    let rhs = b.param(1, k.checked_mul(n).ok_or("GEMM overflow")?);
    let out = b.param(2, m.checked_mul(n).ok_or("GEMM overflow")?);
    let lane = b.special("lid0");
    let gx = b.special("group0");
    let gy = b.special("group1");
    let sa = b.alloc(64);
    let sb = b.alloc(64);
    let sc = b.alloc(64);
    let id = b.next;
    b.next += 1;
    let acc = b.emit(UOp::new(
        Ops::Alloc,
        vec![],
        Arg::Allocation {
            id,
            dtype: DType::MatrixF32x8x8,
            size: 1,
        },
    ));
    let zero = b.emit(UOp::f32(0.).cast(DType::MatrixF32x8x8));
    let init = b.store(&acc, UOp::int(0), zero, UOp::bool(true));
    let block = b.range(k.div_ceil(8));
    let mut dep = block.clone();
    for part in 0..2 {
        let z = lane.add(&UOp::int(part * 32));
        let tr = z.div(&UOp::int(8));
        let tc = z.modulo(&UOp::int(8));
        let ar = gy.mul(&UOp::int(8)).add(&tr);
        let ak = block.mul(&UOp::int(8)).add(&tc);
        let bk = block.mul(&UOp::int(8)).add(&tr);
        let bc = gx.mul(&UOp::int(8)).add(&tc);
        let av = b.load(
            &a,
            ar.mul(&UOp::int(k as i64)).add(&ak),
            ar.lt(&UOp::int(m as i64)).and(&ak.lt(&UOp::int(k as i64))),
        );
        let bv = b.load(
            &rhs,
            bk.mul(&UOp::int(n as i64)).add(&bc),
            bk.lt(&UOp::int(k as i64)).and(&bc.lt(&UOp::int(n as i64))),
        );
        let st = b.store(&sa.after(vec![dep]), z.clone(), av, UOp::bool(true));
        dep = b.store(&sb.after(vec![st]), z, bv, UOp::bool(true));
    }
    let ready = b.barrier(dep);
    let left = b.emit(UOp::new(
        Ops::Load,
        vec![
            sa.after(vec![ready.clone()])
                .index(UOp::int(0), UOp::bool(true)),
            UOp::int(8),
        ],
        Arg::DType(DType::MatrixF32x8x8),
    ));
    let right = b.emit(UOp::new(
        Ops::Load,
        vec![
            sb.after(vec![ready]).index(UOp::int(0), UOp::bool(true)),
            UOp::int(8),
        ],
        Arg::DType(DType::MatrixF32x8x8),
    ));
    let current = b.emit(UOp::new(
        Ops::Load,
        vec![
            acc.after(vec![init, block.clone()])
                .index(UOp::int(0), UOp::bool(true)),
        ],
        Arg::DType(DType::MatrixF32x8x8),
    ));
    let product = b.emit(UOp::new(
        Ops::Wmma,
        vec![left, right, current],
        Arg::DType(DType::MatrixF32x8x8),
    ));
    let stored = b.store(&acc, UOp::int(0), product, UOp::bool(true));
    b.barrier(stored);
    let end = b.end(&block);
    let result = b.emit(UOp::new(
        Ops::Load,
        vec![acc.after(vec![end]).index(UOp::int(0), UOp::bool(true))],
        Arg::DType(DType::MatrixF32x8x8),
    ));
    let st = b.emit(UOp::new(
        Ops::Store,
        vec![sc.index(UOp::int(0), UOp::bool(true)), result, UOp::int(8)],
        Arg::DType(DType::MatrixF32x8x8),
    ));
    let ready = b.barrier(st);
    for part in 0..2 {
        let z = lane.add(&UOp::int(part * 32));
        let row = gy.mul(&UOp::int(8)).add(&z.div(&UOp::int(8)));
        let col = gx.mul(&UOp::int(8)).add(&z.modulo(&UOp::int(8)));
        let value = b.load(&sc.after(vec![ready.clone()]), z, UOp::bool(true));
        b.store(
            &out,
            row.mul(&UOp::int(n as i64)).add(&col),
            value,
            row.lt(&UOp::int(m as i64))
                .and(&col.lt(&UOp::int(n as i64))),
        );
    }
    Ok((
        b.finish(),
        Launch {
            groups: [n.div_ceil(8), m.div_ceil(8), 1],
            local: [32, 1, 1],
        },
    ))
}
```

`src/renderers.rs` の `fn ident`を置き換える。

```rust
// file="src/renderers.rs" mode="replace" item="fn ident"
fn ident(name: &str) -> Result<(), String> {
    if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        || name.as_bytes()[0].is_ascii_digit()
    {
        Err("invalid GPU identifier".into())
    } else {
        Ok(())
    }
}
```

`src/renderers.rs` の `fn ty`を置き換える。

```rust
// file="src/renderers.rs" mode="replace" item="fn ty"
fn ty(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "float",
        DType::F32x4 => "float4",
        DType::MatrixF32x8x8 => "simdgroup_float8x8",
        DType::I64 => "long",
        DType::Bool => "bool",
        DType::Ptr => "float*",
        DType::Void => "void",
        _ => "int",
    }
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
    let mut vars = BTreeMap::new();
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
        if let Arg::Variable { ref name, .. } = u.arg {
            ident(name)?;
            vars.insert(name.clone(), name.clone());
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
    let nparams = params.keys().copied().max().map_or(0, |slot| slot + 1);
    if linear.src.iter().any(|u| u.dtype() == DType::MatrixF32x8x8) {
        if dialect != Dialect::Metal {
            return Err("matrix WMMA unavailable on OpenCL renderer".into());
        }
        source.push_str("#include <metal_simdgroup_matrix>\n");
    }
    let mut signature = params
        .iter()
        .map(|(idx, name)| match dialect {
            Dialect::Metal => format!("device float* {name} [[buffer({idx})]]"),
            Dialect::OpenCl => format!("__global float* {name}"),
        })
        .collect::<Vec<_>>();
    if !vars.is_empty() {
        signature
            .push(
                match dialect {
                    Dialect::Metal => {
                        format!("device const long* scalars [[buffer({nparams})]]")
                    }
                    Dialect::OpenCl => "__global const long* scalars".into(),
                },
            );
    }
    if dialect == Dialect::Metal {
        signature
            .extend([
                "uint3 gid [[thread_position_in_grid]]".into(),
                "uint3 lid [[thread_position_in_threadgroup]]".into(),
                "uint3 group [[threadgroup_position_in_grid]]".into(),
            ]);
    }
    for (idx, (name, _)) in vars.iter().enumerate() {
        body.push_str(&format!("long {name}=scalars[{idx}];\n"));
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
                    Arg::Float(bits) => {
                        let bits = (bits.value() as f32).to_bits();
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
            Ops::Variable => {
                if let Arg::Variable { ref name, .. } = u.arg {
                    name.clone()
                } else {
                    return Err("GPU variable missing name".into());
                }
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
                if u.dtype() == DType::MatrixF32x8x8 && !matrix_pointer(&u.src[0]) {
                    let stride = get(1)?;
                    declarations.push_str(&format!("simdgroup_float8x8 {name};\n"));
                    body.push_str(
                        &format!(
                            "simdgroup_load({name},{},{stride},ulong2(0,0));\n", get(0) ?
                        ),
                    );
                    values.insert(u.id, name);
                    continue;
                }
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
                let typed = if matches!(u.arg, Arg::DType(_))
                    && !matrix_pointer(&u.src[0])
                {
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
                if u.src[1].dtype() == DType::MatrixF32x8x8 && !matrix_pointer(&u.src[0])
                {
                    body.push_str(
                        &format!(
                            "simdgroup_store({value},{ptr},{},ulong2(0,0));\n", get(2) ?
                        ),
                    );
                    continue;
                }
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
            Ops::Cast => {
                if u.dtype() == DType::MatrixF32x8x8 {
                    format!("simdgroup_float8x8({})", get(0) ?)
                } else {
                    format!("(({}){})", ty(u.dtype()), get(0) ?)
                }
            }
            Ops::After => get(0)?,
            Ops::Sink | Ops::Linear => String::new(),
            Ops::Wmma => {
                if dialect != Dialect::Metal || u.src.len() != 3 {
                    return Err("WMMA requires Metal 8x8 F32 matrices".into());
                }
                declarations.push_str(&format!("simdgroup_float8x8 {name};\n"));
                body.push_str(
                    &format!(
                        "simdgroup_multiply_accumulate({name},{},{},{});\n", get(0) ?,
                        get(1) ?, get(2) ?
                    ),
                );
                values.insert(u.id, name);
                continue;
            }
            _ => return Err(format!("unsupported lowered GPU UOp {:?}", u.op)),
        };
        if !expr.is_empty() {
            if matches!(
                u.op, Ops::Index | Ops::Param | Ops::Buffer | Ops::Variable | Ops::After
            ) {
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

`src/renderers.rs`を追記する。

```rust
// file="src/renderers.rs" mode="append"

fn matrix_pointer(u: &UOp) -> bool {
    match u.arg {
        Arg::Allocation { dtype: DType::MatrixF32x8x8, .. } => true,
        _ => {
            matches!(u.op, Ops::Index | Ops::After)
                && u.src.first().is_some_and(matrix_pointer)
        }
    }
}
```

`src/uir.rs` の `fn dtype`を置き換える。

```rust
// file="src/uir.rs" mode="replace" item="fn dtype"
fn dtype(s: &str) -> Result<DType> {
    [
        DType::F32,
        DType::MatrixF32x8x8,
        DType::F32x4,
        DType::I32,
        DType::I64,
        DType::Bool,
        DType::Ptr,
        DType::Void,
    ]
    .into_iter()
    .find(|d| format!("{d:?}") == s)
    .ok_or("unknown dtype".into())
}
```

`src/uir.rs` の `fn op`を置き換える。

```rust
// file="src/uir.rs" mode="replace" item="fn op"
fn op(s: &str) -> Result<Ops> {
    [
        Ops::Invalid,
        Ops::Const,
        Ops::Wmma,
        Ops::Barrier,
        Ops::Special,
        Ops::Buffer,
        Ops::Alloc,
        Ops::Param,
        Ops::Variable,
        Ops::Bind,
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

`src/uop.rs` の `pub enum DType`を置き換える。

```rust
// file="src/uop.rs" mode="replace" item="pub enum DType"
pub enum DType {
    F32,
    I32,
    I64,
    Bool,
    Ptr,
    Void,
    F32x4,
    MatrixF32x8x8,
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
    Variable,
    Bind,
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
    Wmma,
}
```

`tests/hardware_wmma.rs`を保存する。

```rust
// file="tests/hardware_wmma.rs" mode="write"
mod support;
use chibigrad::{gpu,renderers::{self,Dialect}};
static GPU_TEST:std::sync::Mutex<()>=std::sync::Mutex::new(());
#[cfg(all(feature="metal",target_os="macos"))]
#[test]
fn metal_matrix_wmma_common_uops_and_elf_runtime() {
    let _gpu = GPU_TEST.lock().unwrap();
    let caps = chibigrad::backends::metal::MetalDevice::open()
        .unwrap()
        .capabilities()
        .clone();
    assert!(caps.max_threads >= 32, "matrix kernel needs a 32-lane simdgroup");
    let (m, n, k) = (13, 11, 17);
    let a = (0..m * k).map(|i| (i % 7) as f32 - 3.).collect::<Vec<_>>();
    let b = (0..k * n).map(|i| (i % 5) as f32 - 2.).collect::<Vec<_>>();
    let expected = (0..m * n)
        .map(|i| {
            (0..k)
                .map(|j| a[i / n * k + j] * b[j * n + i % n])
                .sum::<f32>()
        })
        .collect::<Vec<_>>();
    let (linear, launch) = gpu::matrix_gemm(m, n, k).unwrap();
    assert!(linear.src.iter().any(|u| u.op == chibigrad::Ops::Wmma));
    assert_eq!(
        support::run_metal(&linear, launch, &[a, b, vec![0.; m * n]]),
        expected
    );
}

#[test]
fn opencl_rejects_matrix_instructions() {
    let (body, launch) = gpu::matrix_gemm(13, 11, 17).unwrap();
    let error = renderers::render(&body, "matrix", Dialect::OpenCl, launch).unwrap_err();
    assert!(error.contains("matrix WMMA unavailable"), "{error}");
}

#[test]
fn cpu_rejects_matrix_instructions(){let (body,_)=gpu::matrix_gemm(13,11,17).unwrap();assert!(chibigrad::codegen::render(&body,"matrix").is_err());}
```

```sh
# check step=25.02 requires=metal,wmma
cargo test --offline --features metal,opencl --test hardware_wmma
```

M3 の capability を確認し、実際の WMMA を含む13x11x17 GEMM が CPU 参照に一致する。OpenCL renderer は matrix 命令を明示的に拒否する。

## 章末の累積確認

前の章までのテストとexampleも再ビルドします。

```sh
# check
cargo test --quiet --all-targets
```

## 要件との対応

### 25-001 Metal device の family、API が返す実行幅、threadgroup 上限、local byte 上限を取得する。

<!-- todo:25-001 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-002 必要な行列型と演算を持つ小 shader をコンパイルし、失敗理由を capability に保存する。

<!-- todo:25-002 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-003 capability が使えない場合も19の f32 GEMM を実行し、同じ出力を得る。

<!-- todo:25-003 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-004 capability を test 用に無効にし、通常経路を強制して確認する。

<!-- todo:25-004 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-005 cache key に device capability、型、compiler と ABI の識別を含める。

<!-- todo:25-005 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-006 まず公開の simdgroup load と store を使い、8x8の tile を一つだけ計算する。

<!-- todo:25-006 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-007 行列の row/column 順、leading dimension、転置指定を入力 layout と照合する。

<!-- todo:25-007 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-008 shader で I@A と A@I を計算し、19の通常積和と CPU reference に比較する。

<!-- todo:25-008 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-009 行列の一箇所だけが1の basis 入力で、出力座標の対応を検証する。

<!-- todo:25-009 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-010 shader を既存の Metal compiler、buffer bind、dispatch、同期、読出しの経路で動かす。

<!-- todo:25-010 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-011 既存の `sum(A[i,k]*B[k,j], k)` の形と dtype を認識する。

<!-- todo:25-011 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-012 MatrixTile を生成し、対応しない stride、型、alignment は通常版へ戻す。

<!-- todo:25-012 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-013 K=16、24で部分和を次の行列演算の accumulator に渡す。

<!-- todo:25-013 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-014 M/N/K が8の倍数でない場合は、まず通常版へ戻して数値を保つ。

<!-- todo:25-014 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-015 次のコミットで端数を local tile にゼロ詰めし、無効 LOAD と STORE を guard する。

<!-- todo:25-015 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-016 capability が許す場合だけ、F16 入力と f32 累積を追加する。

<!-- todo:25-016 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-017 変換後の実 F16 入力値を f64 へ上げた reference と比較し、入力量子化の差を分ける。

<!-- todo:25-017 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-018 負値、ゼロ、打消し、大きさの異なる値、K の複数 tile を確認する。

<!-- todo:25-018 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-019 通常版、行列命令版、fallback の同一入力を比べ、選択経路を記録する。

<!-- todo:25-019 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-020 正しさの確認後に compile、転送、kernel、同期、全体の計測を分ける。

<!-- todo:25-020 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-021 論文の演算一つ、compiler 変換一つ、または新しい device 一つを対象として選ぶ。

<!-- todo:25-021 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-022 入力、出力、数式、比較用の既存実装、対応 dtype と shape を短い仕様にする。

<!-- todo:25-022 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-023 最初に既存 pipeline の baseline example を動かし、同じ入出力で拡張を差し替える。

<!-- todo:25-023 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-024 小入力の期待値、ランダム比較、境界値、必要なら gradient 比較を追加する。

<!-- todo:25-024 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-025 device と commit、再現コマンド、結果、未対応範囲を project の記録へ残す。

<!-- todo:25-025 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-026 identity と basis の期待値に一致し、K の部分和が失われない。

<!-- todo:25-026 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-027 f32 は `abs_err <= 1e-4 + 1e-4*abs(ref)` を初期基準として全要素を比較する。

<!-- todo:25-027 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-028 F16 入力と f32 累積は、量子化済み入力 reference に `abs_err <= 1e-3 + 1e-3*abs(ref)` で比較する。

<!-- todo:25-028 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-029 許容値が満たせない場合は誤差の原因と対象範囲を調べ、無条件で閾値を広げない。

<!-- todo:25-029 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-030 capability 無効、非対応型、不正 shape、端数のケースで安全に fallback またはエラーになる。

<!-- todo:25-030 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-031 実行経路と数値の両方を確認し、source 内の命令名だけで完了にしない。

<!-- todo:25-031 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-032 小 MLP と LLM の一回の更新を回帰確認し、lowering の変更が勾配の意味を変えない。

<!-- todo:25-032 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 25-033 独立課題は baseline と拡張を同じコマンドで比較でき、未確認の実機を明示する。

<!-- todo:25-033 -->

関連する実装と確認は本章の `25.01`, `25.02` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。
