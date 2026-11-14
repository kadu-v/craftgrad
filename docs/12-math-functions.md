# 12. 数学演算を通常 UOp の多項式へ分解する

前の節で動かしたコードに、今回使う演算と処理を追加します。各節の確認を実行してから次へ進みます。

参照するUOpの構成は [tinygrad c22edff](https://github.com/tinygrad/tinygrad/blob/c22edff14c79310a9b2df62a78819cf77c8289fa/tinygrad/uop/ops.py) です。最初から全機能を実装せず、使う節で増やします。

## 12.01 数学関数の正規化とネイティブ実行

exp・log・cos を exp2・log2・sin の組合せへ正規化し、sqrt とともに C の数学関数へ下げる。Tensor と参照評価にも同じ演算を追加する。

`src/codegen.rs` の `use crate::`を置き換える。

```rust
// file="src/codegen.rs" mode="replace" item="use crate::"
use crate::{
    Result,
    pattern::{PatternMatcher, UPat, graph_rewrite},
    uop::{Arg, Axis, DType, Ops, UOp, is_binary, is_unary, numel},
};
```

`src/codegen.rs` の `pub fn normalize`を置き換える。

```rust
// file="src/codegen.rs" mode="replace" item="pub fn normalize"
pub fn normalize(root: &UOp) -> Result<UOp> {
    let root = crate::functions::inline_calls(root)?;
    let pm = PatternMatcher::new()
        .add(UPat::op(Ops::Exp).src(vec![UPat::var("x")]), |c| {
            Some(
                c["x"]
                    .mul(&UOp::f32(std::f32::consts::LOG2_E))
                    .alu(Ops::Exp2, &[]),
            )
        })
        .add(UPat::op(Ops::Log).src(vec![UPat::var("x")]), |c| {
            Some(
                c["x"]
                    .alu(Ops::Log2, &[])
                    .mul(&UOp::f32(std::f32::consts::LN_2)),
            )
        })
        .add(UPat::op(Ops::Cos).src(vec![UPat::var("x")]), |c| {
            Some(
                UOp::f32(std::f32::consts::FRAC_PI_2)
                    .sub(&c["x"])
                    .alu(Ops::Sin, &[]),
            )
        });
    graph_rewrite(&root, &pm)
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
        "extern float exp2f(float);\nextern float log2f(float);\nextern float sinf(float);\nextern float sqrtf(float);\nextern float fmaxf(float,float);\nstatic float chibi_float(unsigned int x) { union { unsigned int u; float f; } b; b.u=x; return b.f; }\nstatic long long chibi_div(long long a,long long b) { long long q=a/b, r=a%b; return q-((r!=0)&&((r<0)!=(b<0))); }\nstatic long long chibi_mod(long long a,long long b) { long long r=a%b; return r+(((r!=0)&&((r<0)!=(b<0)))?b:0); }\n",
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

`src/scalar.rs` の `pub fn verify_node`を置き換える。

```rust
// file="src/scalar.rs" mode="replace" item="pub fn verify_node"
pub fn verify_node(u: &UOp) -> Result<()> {
    if u.op == Ops::Const {
        return Value::from_constant(u)
            .filter(|_| u.src.is_empty())
            .map(|_| ())
            .ok_or("invalid scalar constant".into());
    }
    if let (Ops::Param, Arg::ScalarParam { dtype, .. }) = (u.op, &u.arg) {
        return if scalar_type(*dtype) && u.src.is_empty() {
            Ok(())
        } else {
            Err("invalid scalar parameter".into())
        };
    }
    let types: Vec<_> = u.src.iter().map(UOp::dtype).collect();
    let first = *types.first().ok_or("missing scalar operand")?;
    let unary = crate::uop::is_unary(u.op);
    let n = if u.op == Ops::Where {
        3
    } else if unary {
        1
    } else if crate::uop::is_binary(u.op) {
        2
    } else {
        return Err(format!("unsupported scalar {:?}", u.op));
    };
    if types.len() != n || !types.iter().all(|t| scalar_type(*t)) {
        return Err("scalar arity or dtype mismatch".into());
    }
    match u.op {
        Ops::Cast => {
            if !matches!(u.arg, Arg::DType(dt) if scalar_type(dt)) {
                return Err("invalid CAST target".into());
            }
        }
        Ops::Bitcast => {
            if !matches!(
                (& u.arg, first), (Arg::DType(DType::F32 | DType::I32), DType::I32 |
                DType::F32)
            ) {
                return Err("BITCAST requires equal 32-bit widths".into());
            }
        }
        Ops::Where => {
            if first != DType::Bool || types[1] != types[2] {
                return Err("WHERE dtype mismatch".into());
            }
        }
        Ops::And | Ops::Or | Ops::Xor => {
            if !matches!(first, DType::Bool | DType::I32 | DType::I64)
                || types[1] != first
            {
                return Err("bitwise dtype mismatch".into());
            }
        }
        Ops::Shl | Ops::Shr | Ops::Mod => {
            if !matches!(first, DType::I32 | DType::I64) || types[1] != first {
                return Err("integer dtype mismatch".into());
            }
        }
        Ops::Eq | Ops::Lt => {
            if types[1] != first {
                return Err("comparison dtype mismatch".into());
            }
        }
        Ops::Exp | Ops::Log | Ops::Sin | Ops::Cos | Ops::Exp2 | Ops::Log2 | Ops::Sqrt => {
            if first != DType::F32 {
                return Err("transcendental requires F32".into());
            }
        }
        _ => {
            if first == DType::Bool || types.iter().any(|t| *t != first) {
                return Err("numeric dtype mismatch".into());
            }
        }
    }
    if u.op != Ops::Cast && u.op != Ops::Bitcast && u.arg != Arg::None {
        return Err("unexpected scalar argument".into());
    }
    Ok(())
}
```

`src/scalar.rs` の `pub fn eval_alu`を置き換える。

```rust
// file="src/scalar.rs" mode="replace" item="pub fn eval_alu"
pub fn eval_alu(op: Ops, arg: &Arg, values: &[Value]) -> Result<Value> {
    let sources = values.iter().map(|v| v.uop()).collect();
    let node = UOp::new(op, sources, arg.clone());
    verify_node(&node)?;
    let a = values[0];
    if op == Ops::Where {
        return Ok(values[if a == Value::Bool(true) { 1 } else { 2 }]);
    }
    if matches!(op, Ops::Cast | Ops::Bitcast) && *arg == Arg::DType(a.dtype()) {
        return Ok(a);
    }
    if op == Ops::Bitcast {
        return Ok(
            match a {
                Value::I32(v) => Value::F32Bits(v as u32),
                Value::F32Bits(v) => Value::I32(v as i32),
                _ => unreachable!(),
            },
        );
    }
    if op == Ops::Cast {
        let Arg::DType(dt) = arg else { unreachable!() };
        return Ok(
            match dt {
                DType::Bool => Value::Bool(a.number() != 0.),
                DType::F32 => Value::f32(a.number() as f32),
                DType::I32 => {
                    match a {
                        Value::F32Bits(v) => {
                            let x = f32::from_bits(v);
                            if !x.is_finite()
                                || !(-2147483648.0..2147483648.0).contains(&(x as f64))
                            {
                                return Err("unsafe F32 to I32 CAST".into());
                            }
                            Value::I32(x as i32)
                        }
                        Value::I64(v) => Value::I32(v as i32),
                        _ => Value::I32(a.number() as i32),
                    }
                }
                DType::I64 => {
                    match a {
                        Value::F32Bits(v) => {
                            let x = f32::from_bits(v);
                            if !x.is_finite()
                                || !(-9223372036854775808.0..9223372036854775808.0)
                                    .contains(&(x as f64))
                            {
                                return Err("unsafe F32 to I64 CAST".into());
                            }
                            Value::I64(x as i64)
                        }
                        Value::I64(v) => Value::I64(v),
                        _ => Value::I64(a.number() as i64),
                    }
                }
                _ => unreachable!(),
            },
        );
    }
    let b = values.get(1).copied();
    if matches!(op, Ops::Eq | Ops::Lt) {
        let b = b.unwrap();
        let cmp = match (a, b) {
            (Value::I64(a), Value::I64(b)) => if op == Ops::Eq { a == b } else { a < b }
            _ => {
                if op == Ops::Eq {
                    a.number() == b.number()
                } else {
                    a.number() < b.number()
                }
            }
        };
        return Ok(Value::Bool(cmp));
    }
    Ok(
        match a {
            Value::Bool(a) => {
                let Value::Bool(b) = b.unwrap() else { unreachable!() };
                Value::Bool(
                    match op {
                        Ops::And => a & b,
                        Ops::Or => a | b,
                        Ops::Xor => a ^ b,
                        _ => unreachable!(),
                    },
                )
            }
            Value::I32(a) => {
                let b = match b {
                    Some(Value::I32(v)) => v,
                    _ => 0,
                };
                if matches!(op, Ops::Shl | Ops::Shr) && !(0..32).contains(&b) {
                    return Err("unsafe I32 shift count".into());
                }
                if matches!(op, Ops::Div | Ops::Mod)
                    && (b == 0 || (a == i32::MIN && b == -1))
                {
                    return Err("unsafe I32 division".into());
                }
                Value::I32(
                    match op {
                        Ops::Add => a.wrapping_add(b),
                        Ops::Sub => a.wrapping_sub(b),
                        Ops::Mul => a.wrapping_mul(b),
                        Ops::Neg => a.wrapping_neg(),
                        Ops::And => a & b,
                        Ops::Or => a | b,
                        Ops::Xor => a ^ b,
                        Ops::Shl => a.wrapping_shl(b as u32),
                        Ops::Shr => a >> b,
                        Ops::Div => {
                            crate::scalar::floor_div(a as i64, b as i64).unwrap() as i32
                        }
                        Ops::Mod => {
                            crate::scalar::floor_mod(a as i64, b as i64).unwrap() as i32
                        }
                        Ops::Max => a.max(b),
                        _ => return Err("invalid I32 operation".into()),
                    },
                )
            }
            Value::I64(a) => {
                let b = match b {
                    Some(Value::I64(v)) => v,
                    _ => 0,
                };
                if matches!(op, Ops::Shl | Ops::Shr) && !(0..64).contains(&b) {
                    return Err("unsafe I64 shift count".into());
                }
                Value::I64(
                    match op {
                        Ops::Add => a.checked_add(b),
                        Ops::Sub => a.checked_sub(b),
                        Ops::Mul => a.checked_mul(b),
                        Ops::Neg => a.checked_neg(),
                        Ops::And => Some(a & b),
                        Ops::Or => Some(a | b),
                        Ops::Xor => Some(a ^ b),
                        Ops::Shl => Some(a.wrapping_shl(b as u32)),
                        Ops::Shr => Some(a >> b),
                        Ops::Div => crate::scalar::floor_div(a, b),
                        Ops::Mod => crate::scalar::floor_mod(a, b),
                        Ops::Max => Some(a.max(b)),
                        _ => None,
                    }
                        .ok_or("invalid or overflowing I64 operation")?,
                )
            }
            Value::F32Bits(a) => {
                let a = f32::from_bits(a);
                let b = match b {
                    Some(Value::F32Bits(v)) => f32::from_bits(v),
                    _ => 0.,
                };
                Value::f32(
                    match op {
                        Ops::Add => a + b,
                        Ops::Sub => a - b,
                        Ops::Mul => a * b,
                        Ops::Div => a / b,
                        Ops::Neg => -a,
                        Ops::Max => {
                            if a.is_nan() || b.is_nan() {
                                f32::NAN
                            } else if a == 0. && b == 0. {
                                0.
                            } else {
                                a.max(b)
                            }
                        }
                        Ops::Exp => a.exp(),
                        Ops::Log => a.ln(),
                        Ops::Sin => a.sin(),
                        Ops::Cos => a.cos(),
                        Ops::Exp2 => a.exp2(),
                        Ops::Log2 => a.log2(),
                        Ops::Sqrt => a.sqrt(),
                        _ => return Err("invalid F32 operation".into()),
                    },
                )
            }
        },
    )
}
```

`src/tensor.rs` の `impl Tensor` 内を追記する。

```rust
// file="src/tensor.rs" mode="append" within="impl Tensor"
    pub fn exp(&self) -> Self {
        self.unary(Ops::Exp)
    }

    pub fn log(&self) -> Self {
        self.unary(Ops::Log)
    }

    pub fn sin(&self) -> Self {
        self.unary(Ops::Sin)
    }

    pub fn cos(&self) -> Self {
        self.unary(Ops::Cos)
    }

    pub fn sqrt(&self) -> Self {
        self.unary(Ops::Sqrt)
    }

    pub fn exp2(&self) -> Self {
        self.unary(Ops::Exp2)
    }

    pub fn log2(&self) -> Self {
        self.unary(Ops::Log2)
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
    ]
    .into_iter()
    .find(|o| format!("{o:?}") == s)
    .ok_or("unknown operation".into())
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
}
```

`src/uop.rs` の `pub fn is_unary`を置き換える。

```rust
// file="src/uop.rs" mode="replace" item="pub fn is_unary"
pub fn is_unary(op: Ops) -> bool {
    matches!(
        op,
        Ops::Neg
            | Ops::Exp
            | Ops::Log
            | Ops::Sin
            | Ops::Cos
            | Ops::Sqrt
            | Ops::Exp2
            | Ops::Log2
            | Ops::Cast
            | Ops::Bitcast
    )
}
```

`tests/math.rs`を保存する。

```rust
// file="tests/math.rs" mode="write"
use chibigrad::Tensor;
#[test]
fn tensor_math_uses_native_functions() {
    let x = Tensor::from_slice(&[0.25, 0.5, 1., 2.], &[4]);
    for (actual, expected) in [
        (
            x.exp().to_vec().unwrap(),
            [0.25f32, 0.5, 1., 2.].map(f32::exp),
        ),
        (
            x.log().to_vec().unwrap(),
            [0.25f32, 0.5, 1., 2.].map(f32::ln),
        ),
        (
            x.sqrt().to_vec().unwrap(),
            [0.25f32, 0.5, 1., 2.].map(f32::sqrt),
        ),
    ] {
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
```

`tests/tensor_native.rs` の `fn fused_reduction_rebinds_native_inputs`を置き換える。

```rust
// file="tests/tensor_native.rs" mode="replace" item="fn fused_reduction_rebinds_native_inputs"
fn fused_reduction_rebinds_native_inputs() {
    let x = Tensor::parameter("x", &[4]);
    let w = Tensor::parameter("w", &[4]);
    let y = x.mul(&w).add(&Tensor::scalar(1.)).sum();
    let exe = y.compile().unwrap();
    assert_eq!(exe.kernel_count(), 1);
    for p in exe.programs() {
        assert!(p.src.iter().any(|s| s.op == Ops::Binary));
        assert!(
            !p.src[1]
                .src
                .iter()
                .any(|u| matches!(u.op, Ops::Reduce | Ops::Reshape | Ops::Exp | Ops::Cos))
        );
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
```

```sh
# check step=12.01
cargo test --offline
```

exp・log・sqrt の Tensor 結果が参照値に 1e-5 以内で一致する。

## 12.02 通常の UOp だけで数学関数を近似する

ビット分解、整数指数、Horner 法で exp2・log2・sin・sqrt のフォールバックを作る。近似の有効範囲をガードとして残し、範囲外はネイティブ実行からエラーを返す。

`src/lib.rs`を追記する。

```rust
// file="src/lib.rs" mode="append"


pub mod math_decomp;
```

`src/math_decomp.rs`を保存する。

```rust
// file="src/math_decomp.rs" mode="write"
//! Polynomial fallbacks built entirely from ordinary UOps.
//! Forced fallbacks reject unsupported inputs through native guards.
//! Exp2 supports [-149,127], Log2 positive finite f32, Sin [-32,32],
//! and Sqrt nonnegative finite f32, including signed zero and subnormals.
use crate::{DType, Ops, Result, UOp};
use std::collections::HashMap;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Domain {
    Exp2,
    Log2,
    Sin,
    Sqrt,
}
impl Domain {
    fn guard(self, x: &UOp) -> UOp {
        let (lo, hi) = match self {
            Self::Exp2 => (-149., 127.),
            Self::Log2 => (f32::from_bits(1), f32::MAX),
            Self::Sin => (-32., 32.),
            Self::Sqrt => (0., f32::MAX),
        };
        x.lt(&UOp::f32(lo))
            .eq(&UOp::bool(false))
            .and(&UOp::f32(hi).lt(x).eq(&UOp::bool(false)))
            .and(&x.eq(x))
    }
}
#[derive(Clone, Debug)]
pub struct MathDecomposition {
    pub root: UOp,
    pub domains: Vec<(UOp, Domain)>,
}
impl MathDecomposition {
    pub fn validate(&self, inputs: &[crate::scalar::Value]) -> Result<()> {
        let types = crate::scalar::verify(&self.root)?;
        if types.len() != inputs.len() || types.iter().zip(inputs).any(|(dt, v)| *dt != v.dtype()) {
            return Err("math input signature mismatch".into());
        }
        for (x, domain) in &self.domains {
            let guard = domain.guard(x);
            let mut constants: HashMap<u64, UOp> = HashMap::new();
            for u in guard.toposort() {
                let replacement = if let crate::Arg::ScalarParam { index, dtype } = u.arg {
                    let input = inputs.get(index).ok_or("missing math input")?;
                    if input.dtype() != dtype {
                        return Err("math input dtype mismatch".into());
                    }
                    input.uop()
                } else {
                    u.replace(u.src.iter().map(|s| constants[&s.id].clone()).collect())
                };
                constants.insert(u.id, replacement);
            }
            if crate::scalar::evaluate(&constants[&guard.id], &[])?
                != crate::scalar::Value::Bool(true)
            {
                return Err(format!("input outside {domain:?} approximation domain"));
            }
        }
        Ok(())
    }

    pub fn compile(&self) -> Result<crate::scalar::CompiledScalar> {
        let guards = self
            .domains
            .iter()
            .map(|(x, d)| d.guard(x))
            .collect::<Vec<_>>();
        crate::scalar::compile_with_guards(&self.root, &guards)
    }
}
fn horner(x: &UOp, coefficients: &[f32]) -> UOp {
    let mut p = UOp::f32(*coefficients.last().unwrap());
    for &c in coefficients[..coefficients.len() - 1].iter().rev() {
        p = p.mul(x).add(&UOp::f32(c));
    }
    p
}
fn floor(x: &UOp) -> UOp {
    let trunc = x.cast(DType::I32);
    x.lt(&trunc.cast(DType::F32))
        .where_(&trunc.sub(&UOp::i32(1)), &trunc)
}
fn pow2_integer(x: &UOp) -> UOp {
    x.add(&UOp::i32(127))
        .alu(Ops::Shl, &[UOp::i32(23)])
        .bitcast(DType::F32)
}
pub fn approximate_exp2(x: &UOp) -> UOp {
    let n = floor(x);
    let r = x.sub(&n.cast(DType::F32));
    let scale_n = n.lt(&UOp::i32(-126)).where_(&n.add(&UOp::i32(24)), &n);
    let factor = n
        .lt(&UOp::i32(-126))
        .where_(&UOp::f32(1. / 16777216.), &UOp::f32(1.));
    let l = std::f32::consts::LN_2;
    let mut coefficients = vec![1.];
    for i in 1..=10 {
        coefficients.push(coefficients[i - 1] * l / i as f32);
    }
    horner(&r, &coefficients)
        .mul(&pow2_integer(&scale_n))
        .mul(&factor)
}
fn mantissa_exponent(x: &UOp) -> (UOp, UOp) {
    let tiny = x.lt(&UOp::f32(f32::MIN_POSITIVE));
    let y = tiny.where_(&x.mul(&UOp::f32(16777216.)), x);
    let bits = y.bitcast(DType::I32);
    let exponent = bits
        .alu(Ops::Shr, &[UOp::i32(23)])
        .sub(&UOp::i32(127))
        .sub(&tiny.where_(&UOp::i32(24), &UOp::i32(0)));
    let mantissa = bits
        .and(&UOp::i32(0x7fffff))
        .alu(Ops::Or, &[UOp::i32(0x3f800000)])
        .bitcast(DType::F32);
    (mantissa, exponent)
}
pub fn approximate_log2(x: &UOp) -> UOp {
    let (m, e) = mantissa_exponent(x);
    let t = m.sub(&UOp::f32(1.)).div(&m.add(&UOp::f32(1.)));
    let coefficients = (0..10)
        .map(|n| 2. * std::f32::consts::LOG2_E / (2 * n + 1) as f32)
        .collect::<Vec<_>>();
    t.mul(&horner(&t.mul(&t), &coefficients))
        .add(&e.cast(DType::F32))
}
pub fn approximate_sin(x: &UOp) -> UOp {
    let q = floor(
        &x.mul(&UOp::f32(std::f32::consts::FRAC_1_PI / 2.))
            .add(&UOp::f32(0.5)),
    )
    .cast(DType::F32);
    let r = x.sub(&q.mul(&UOp::f32(std::f32::consts::TAU)));
    let r = UOp::f32(std::f32::consts::FRAC_PI_2)
        .lt(&r)
        .where_(&UOp::f32(std::f32::consts::PI).sub(&r), &r);
    let r = r
        .lt(&UOp::f32(-std::f32::consts::FRAC_PI_2))
        .where_(&UOp::f32(-std::f32::consts::PI).sub(&r), &r);
    r.mul(&horner(
        &r.mul(&r),
        &[
            1.,
            -1. / 6.,
            1. / 120.,
            -1. / 5040.,
            1. / 362880.,
            -1. / 39916800.,
            1. / 6227020800.,
        ],
    ))
}
pub fn approximate_sqrt(x: &UOp) -> UOp {
    let (m, e) = mantissa_exponent(x);
    let half = e.div(&UOp::i32(2));
    let m = e
        .modulo(&UOp::i32(2))
        .eq(&UOp::i32(1))
        .where_(&m.mul(&UOp::f32(2.)), &m);
    let mut q = UOp::f32(1.);
    for _ in 0..6 {
        q = q.add(&m.div(&q)).mul(&UOp::f32(0.5));
    }
    let result = q.mul(&pow2_integer(&half));
    x.eq(&UOp::f32(0.)).where_(x, &result)
}
pub fn decompose(root: &UOp, force_fallback: bool) -> Result<MathDecomposition> {
    if !force_fallback {
        return Ok(MathDecomposition {
            root: root.clone(),
            domains: vec![],
        });
    }
    let root = crate::codegen::normalize(root)?;
    let mut mapped: HashMap<u64, UOp> = HashMap::new();
    let mut domains = vec![];
    for u in root.toposort() {
        let node = u.replace(u.src.iter().map(|s| mapped[&s.id].clone()).collect());
        let rewrite = match node.op {
            Ops::Exp2 => (
                Some(Domain::Exp2),
                Some(approximate_exp2 as fn(&UOp) -> UOp),
            ),
            Ops::Log2 => (
                Some(Domain::Log2),
                Some(approximate_log2 as fn(&UOp) -> UOp),
            ),
            Ops::Sin => (Some(Domain::Sin), Some(approximate_sin as fn(&UOp) -> UOp)),
            Ops::Sqrt => (
                Some(Domain::Sqrt),
                Some(approximate_sqrt as fn(&UOp) -> UOp),
            ),
            _ => (None, None),
        };
        let result = if let (Some(domain), Some(build)) = rewrite {
            if node.src.len() != 1 || node.dtype() != DType::F32 {
                return Err("math fallback requires F32 unary UOp".into());
            }
            domains.push((node.src[0].clone(), domain));
            build(&node.src[0])
        } else {
            node
        };
        mapped.insert(u.id, result);
    }
    Ok(MathDecomposition {
        root: mapped.remove(&root.id).unwrap(),
        domains,
    })
}
```

`tests/math_decomp.rs`を保存する。

```rust
// file="tests/math_decomp.rs" mode="write"
use chibigrad::{
    DType, Ops, Result,
    functions::Function,
    math_decomp::{self, Domain},
    scalar::{self, Value},
};

#[test]
fn forced_fallback_native_accuracy_and_domains() -> Result<()> {
    let x = Function::parameter(0, DType::F32);
    let mut seed = 0x51c0ffeeu32;
    for (op, reference, domain, mut samples, tolerance) in [
        (
            Ops::Exp2,
            f32::exp2 as fn(f32) -> f32,
            Domain::Exp2,
            vec![-149., -148.5, -126., -16., -0.5, 0., 0.5, 20., 126., 127.],
            3e-7f32,
        ),
        (
            Ops::Log2,
            f32::log2,
            Domain::Log2,
            vec![
                f32::from_bits(1),
                f32::from_bits(2),
                f32::MIN_POSITIVE,
                0.1,
                0.5,
                1.,
                1.5,
                2.,
                100.,
                f32::MAX,
            ],
            3e-7,
        ),
        (
            Ops::Sin,
            f32::sin,
            Domain::Sin,
            vec![
                -32.,
                -std::f32::consts::PI,
                -0.,
                0.,
                1.,
                std::f32::consts::PI,
                32.,
            ],
            5e-6,
        ),
        (
            Ops::Sqrt,
            f32::sqrt,
            Domain::Sqrt,
            vec![
                -0.,
                0.,
                f32::from_bits(1),
                f32::from_bits(2),
                f32::MIN_POSITIVE,
                0.25,
                1.,
                2.,
                100.,
                f32::MAX,
            ],
            2e-7,
        ),
    ] {
        for _ in 0..200 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let unit = seed as f64 / u32::MAX as f64;
            samples.push(match domain {
                Domain::Exp2 => (unit * 276. - 149.) as f32,
                Domain::Log2 | Domain::Sqrt => f32::from_bits((seed % 0x7f7fffff) + 1),
                Domain::Sin => (unit * 64. - 32.) as f32,
            });
        }
        let root = x.alu(op, &[]);
        let decomposed = math_decomp::decompose(&root, true)?;
        assert!(
            !decomposed
                .root
                .toposort()
                .iter()
                .any(|u| matches!(u.op, Ops::Exp2 | Ops::Log2 | Ops::Sin | Ops::Sqrt))
        );
        let p = decomposed.compile()?;
        for x in samples {
            let input = [Value::f32(x)];
            let actual = p.run(&input)?.number() as f32;
            let expected = reference(x);
            let error = (actual - expected).abs();
            let limit = if domain == Domain::Sin {
                tolerance
            } else {
                tolerance * expected.abs() + f32::from_bits(1)
            };
            assert!(
                error <= limit,
                "{op:?} {x} actual {actual} expected {expected} error {error} limit {limit}"
            );
            let oracle = scalar::evaluate(&decomposed.root, &input)?.number() as f32;
            assert!(
                (actual - oracle).abs() <= limit,
                "native oracle disagreement {op:?} {x}"
            );
            if domain == Domain::Sqrt && x == 0. {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(p.run(&[Value::f32(value)]).is_err());
        }
        let bad = match domain {
            Domain::Exp2 => 128.,
            Domain::Log2 => 0.,
            Domain::Sin => 33.,
            Domain::Sqrt => -1.,
        };
        assert!(p.run(&[Value::f32(bad)]).is_err());
        assert_eq!(math_decomp::decompose(&root, false)?.root, root);
    }
    Ok(())
}
```

```sh
# check step=12.02
cargo test --offline
```

200 個以上の入力と端点で近似精度を確認し、NaN・範囲外・不正な型を拒否する。

## 章末の累積確認

前の章までのテストとexampleも再ビルドします。

```sh
# check
cargo test --quiet --all-targets
```

## 要件との対応

### 12-001 SQRT の dtype と入力数を verifier に追加する。

<!-- todo:12-001 -->

関連する実装と確認は本章の `12.01`, `12.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 12-002 interpreter と C renderer の sqrt をそろえる。

<!-- todo:12-002 -->

関連する実装と確認は本章の `12.01`, `12.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 12-003 コンパイルされた共有ライブラリから sqrtf を解決できるようにする。

<!-- todo:12-003 -->

関連する実装と確認は本章の `12.01`, `12.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 12-004 通常値、-0、負値、Inf、NaN を分類して比較する。

<!-- todo:12-004 -->

関連する実装と確認は本章の `12.01`, `12.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 12-005 SIN、LOG2、EXP2 を elementwise として rangeify に通す。

<!-- todo:12-005 -->

関連する実装と確認は本章の `12.01`, `12.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 12-006 C 側を sinf、log2f、exp2f に下げる。

<!-- todo:12-006 -->

関連する実装と確認は本章の `12.01`, `12.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 12-007 `exp` と自然対数を定数の積と既存 UOp の合成で追加する。

<!-- todo:12-007 -->

関連する実装と確認は本章の `12.01`, `12.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 12-008 特殊値の期待する分類と定義域を API の説明に書く。

<!-- todo:12-008 -->

関連する実装と確認は本章の `12.01`, `12.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 12-009 `exp2(log2(x))` を正の小さい配列で実行する。

<!-- todo:12-009 -->

関連する実装と確認は本章の `12.01`, `12.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 12-010 exp2 は `x=q+r`、`r∈[-0.5,0.5]` に分けて Horner 形式で評価する。

<!-- todo:12-010 -->

関連する実装と確認は本章の `12.01`, `12.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 12-011 `2^q` の再構成を定義し、整数変換の範囲を検査する。

<!-- todo:12-011 -->

関連する実装と確認は本章の `12.01`, `12.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 12-012 log2 は `x=2^e*m` に分け、m を限られた区間に正規化する。

<!-- todo:12-012 -->

関連する実装と確認は本章の `12.01`, `12.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 12-013 係数、近似区間、次数、出典または係数生成方法を実装の近くに残す。

<!-- todo:12-013 -->

関連する実装と確認は本章の `12.01`, `12.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 12-014 各段階を UOp にして既存 rewrite と renderer へ通す。

<!-- todo:12-014 -->

関連する実装と確認は本章の `12.01`, `12.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 12-015 周期と象限を求め、短い区間へ還元して符号を復元する。

<!-- todo:12-015 -->

関連する実装と確認は本章の `12.01`, `12.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 12-016 区間内だけの奇多項式を Horner 形式で評価する。

<!-- todo:12-016 -->

関連する実装と確認は本章の `12.01`, `12.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 12-017 `|x|<=32` の境界、π の倍数、その両隣をテストする。

<!-- todo:12-017 -->

関連する実装と確認は本章の `12.01`, `12.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 12-018 大きな有限値と Inf、NaN を libm 経路に分岐する。

<!-- todo:12-018 -->

関連する実装と確認は本章の `12.01`, `12.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 12-019 高水準 SIN と近似展開後のグラフを両方 dump できるようにする。

<!-- todo:12-019 -->

関連する実装と確認は本章の `12.01`, `12.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 12-020 通常値の許容誤差と特殊値の分類を分けて判定する。

<!-- todo:12-020 -->

関連する実装と確認は本章の `12.01`, `12.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 12-021 sqrt(-0) の sign bit、log2(0) の -Inf、exp2(-Inf) の 0 を検査する。

<!-- todo:12-021 -->

関連する実装と確認は本章の `12.01`, `12.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 12-022 固定 seed の有限値集合と、区間境界の隣接 f32 を比較する。

<!-- todo:12-022 -->

関連する実装と確認は本章の `12.01`, `12.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 12-023 近似対象外の値を libm に戻す分岐が生成コードに残る。

<!-- todo:12-023 -->

関連する実装と確認は本章の `12.01`, `12.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 12-024 近似を選ばない既存 example の数値が変わらない。

<!-- todo:12-024 -->

関連する実装と確認は本章の `12.01`, `12.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。
