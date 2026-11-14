# 06. RANGE と END のスコープと guard を検証する

前の節で動かしたコードに、今回使う演算と処理を追加します。各節の確認を実行してから次へ進みます。

参照するUOpの構成は [tinygrad c22edff](https://github.com/tinygrad/tinygrad/blob/c22edff14c79310a9b2df62a78819cf77c8289fa/tinygrad/uop/ops.py) です。最初から全機能を実装せず、使う節で増やします。

## 06.01 RANGE と END で反復する

RANGE を C の for、END をループ終了へ変換する。ループに入る際の値とアドレスの環境を保存し、終了時に戻す。4 回の反復に 3 要素のマスクを合わせ、末尾の番兵を保つ。

`src/codegen.rs` の `use crate::`を置き換える。

```rust
// file="src/codegen.rs" mode="replace" item="use crate::"
use crate::{
    Result,
    uop::{Arg, Axis, DType, Ops, UOp, is_binary, is_unary},
};
```

`src/codegen.rs` の `pub fn verify_lowered`を置き換える。

```rust
// file="src/codegen.rs" mode="replace" item="pub fn verify_lowered"
pub fn verify_lowered(linear: &UOp) -> Result<()> {
    if linear.op != Ops::Linear {
        return Err("renderer expects LINEAR".into());
    }
    let mut ranges = vec![];
    for u in &linear.src {
        if u.op == Ops::Range {
            if u.src.len() != 2 {
                return Err("RANGE needs start/end".into());
            }
            ranges.push(u.id)
        }
        if u.op == Ops::End {
            let id = u.src.last().ok_or("END missing RANGE")?.id;
            if ranges.pop() != Some(id) {
                return Err("END does not match active RANGE".into());
            }
        }
        if u.op == Ops::Index {
            if u.src.len() != 3 {
                return Err("INDEX needs buffer,index,valid".into());
            }
            if u.src[2] == UOp::bool(true) {
                if let (Some(size), Some((lo, hi))) =
                    (pointer_size(&u.src[0]), index_bounds(&u.src[1]))
                {
                    if lo < 0 || hi >= size as i64 {
                        return Err("unguarded index outside allocation".into());
                    }
                }
            }
        }
    }
    if !ranges.is_empty() {
        return Err("unclosed RANGE".into());
    }
    Ok(())
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
            (Ops::Param, _) => continue,
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

`src/codegen.rs` の `fn index_bounds`を置き換える。

```rust
// file="src/codegen.rs" mode="replace" item="fn index_bounds"
fn index_bounds(u: &UOp) -> Option<(i64, i64)> {
    match &u.arg {
        Arg::Int(n) => Some((*n, *n)),
        Arg::Range { .. } => {
            let a = index_bounds(&u.src[0])?.0;
            let b = index_bounds(&u.src[1])?.1;
            Some((a, b - 1))
        }
        _ => None,
    }
}
```

`src/lib.rs`を追記する。

```rust
// file="src/lib.rs" mode="append"

pub use uop::Axis;
```

`src/uir.rs` の `use crate::`を置き換える。

```rust
// file="src/uir.rs" mode="replace" item="use crate::"
use crate::{Arg, Axis, DType, Ops, Result, UOp};
```

`src/uir.rs` の `fn op`を置き換える。

```rust
// file="src/uir.rs" mode="replace" item="fn op"
fn op(s: &str) -> Result<Ops> {
    [
        Ops::Invalid,
        Ops::Const,
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
        Arg::Range { id, axis } => format!("range;{id};{axis:?}"),
        Arg::Id(v) => format!("id;{v}"),
        Arg::DType(dt) => format!("dtype;{dt:?}"),
        Arg::Text(s) => format!("text;{}", hex(s.as_bytes())),
        Arg::Bytes(v) => format!("bytes;{}", hex(v)),
        Arg::Slot { index, size } => format!("slot;{index};{size}"),
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
        ["range", id, a] => Arg::Range {
            id: number(id)?,
            axis: match *a {
                "Loop" => Axis::Loop,
                _ => return Err("unknown axis".into()),
            },
        },
        ["id", v] => Arg::Id(number(v)?),
        ["dtype", dt] => Arg::DType(dtype(dt)?),
        ["text", v] => Arg::Text(string(v)?),
        ["bytes", v] => Arg::Bytes(unhex(v)?),
        ["slot", i, v] => Arg::Slot {
            index: number(i)?,
            size: number(v)?,
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
        crate::scalar::verify_node(u)?;
    }
    match (&u.op, &u.arg) {
        (Ops::Const, _) => {
            crate::scalar::verify_node(u)?;
        }
        (Ops::Param, Arg::ScalarParam { .. }) => {
            crate::scalar::verify_node(u)?;
        }
        (Ops::Param, Arg::Slot { .. }) if u.src.is_empty() => {}
        (Ops::Param, _) => return Err("invalid leaf metadata".into()),
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

`src/uir.rs`を追記する。

```rust
// file="src/uir.rs" mode="append"

fn validate_linear_scope(linear: &UOp) -> Result<()> {
    let mut free: HashMap<u64, HashSet<u64>> = HashMap::new();
    for u in linear.toposort() {
        let mut dependencies = HashSet::new();
        for s in &u.src {
            dependencies.extend(free[&s.id].iter().copied());
        }
        if u.op == Ops::Range {
            dependencies.insert(u.id);
        }
        if u.op == Ops::End {
            dependencies.remove(&u.src.last().ok_or("END lacks range")?.id);
        }
        free.insert(u.id, dependencies);
    }
    let mut active = HashSet::new();
    for u in &linear.src {
        if u.op == Ops::Range {
            active.insert(u.id);
        }
        if !free[&u.id].is_subset(&active) {
            return Err("RANGE value escapes lexical loop scope".into());
        }
        if u.op == Ops::End {
            active.remove(&u.src.last().ok_or("END lacks range")?.id);
        }
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
    Range { id: usize, axis: Axis },
    Id(usize),
    DType(DType),
    Text(String),
    Bytes(Vec<u8>),
    Slot { index: usize, size: usize },
    ScalarParam { index: usize, dtype: DType },
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
    pub fn range(end: UOp, id: usize, axis: Axis) -> Self {
        Self::new(Ops::Range, vec![Self::int(0), end], Arg::Range { id, axis })
    }
```

`src/uop.rs`を追記する。

```rust
// file="src/uop.rs" mode="append"

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Axis {
    Loop,
}
```

`tests/loop_memory.rs`を保存する。

```rust
// file="tests/loop_memory.rs" mode="write"
use chibigrad::{Arg, Axis, Ops, UOp, codegen, cpu::CpuProgram};
#[test]
fn masked_vector_load_and_store_execute_real_ranges() {
    let input = UOp::new(Ops::Param, vec![], Arg::Slot { index: 0, size: 3 });
    let output = UOp::new(Ops::Param, vec![], Arg::Slot { index: 1, size: 3 });
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

`tests/loops_uop.rs`を保存する。

```rust
// file="tests/loops_uop.rs" mode="write"
use chibigrad::{Arg, Axis, Ops, UOp, codegen};
#[test]
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
    let buffer = UOp::new(Ops::Param, vec![], Arg::Slot { index: 0, size: 3 });
    let outside = buffer.index(UOp::int(3), UOp::bool(true));
    assert!(codegen::verify_lowered(&UOp::linear(vec![outside])).is_err());
}
```

```sh
# check step=06.01
cargo test --offline
```

入力 [2,3,4] が [4,6,8] になり、範囲外 STORE は起きない。閉じ忘れや交差した END を拒否する。

## 06.02 ループの値の寿命を検証する

各ノードが参照する RANGE を集め、LINEAR のその位置で有効なループの集合と比較する。END の対応だけでなく、ループを抜けた後の添字の使用も生成前に拒否する。

`src/codegen.rs` の `pub fn verify_lowered`を置き換える。

```rust
// file="src/codegen.rs" mode="replace" item="pub fn verify_lowered"
pub fn verify_lowered(linear: &UOp) -> Result<()> {
    if linear.op != Ops::Linear {
        return Err("renderer expects LINEAR".into());
    }
    let mut ranges = vec![];
    for u in &linear.src {
        if u.op == Ops::Range {
            if u.src.len() != 2 {
                return Err("RANGE needs start/end".into());
            }
            ranges.push(u.id)
        }
        if u.op == Ops::End {
            let id = u.src.last().ok_or("END missing RANGE")?.id;
            if ranges.pop() != Some(id) {
                return Err("END does not match active RANGE".into());
            }
        }
        if u.op == Ops::Index {
            if u.src.len() != 3 {
                return Err("INDEX needs buffer,index,valid".into());
            }
            if u.src[2] == UOp::bool(true) {
                if let (Some(size), Some((lo, hi))) =
                    (pointer_size(&u.src[0]), index_bounds(&u.src[1]))
                {
                    if lo < 0 || hi >= size as i64 {
                        return Err("unguarded index outside allocation".into());
                    }
                }
            }
        }
    }
    if !ranges.is_empty() {
        return Err("unclosed RANGE".into());
    }
    crate::uir::validate_linear_scope(linear)?;
    Ok(())
}
```

`src/uir.rs`を保存する。

```rust
// file="src/uir.rs" mode="write"
//! Canonical UIR v2. Each line is `id=Op DType [source_ids] typed_arg`.
//! IDs are dense and backward references only. Strings and bytes use hexadecimal
//! UTF-8/bytes, floats use their exact hexadecimal IEEE bits, and lists use commas.
use crate::{Arg, Axis, DType, Ops, Result, UOp};
use std::collections::{HashMap, HashSet};
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn unhex(s: &str) -> Result<Vec<u8>> {
    if s.len() % 2 != 0 || !s.is_ascii() {
        return Err("invalid hex".into());
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|_| "invalid hex".into()))
        .collect()
}
fn string(s: &str) -> Result<String> {
    String::from_utf8(unhex(s)?).map_err(|_| "invalid UTF-8".into())
}
fn number<T: std::str::FromStr>(s: &str) -> Result<T> {
    s.parse().map_err(|_| "invalid number".into())
}
fn list<T: std::str::FromStr>(s: &str) -> Result<Vec<T>> {
    if s.is_empty() {
        Ok(vec![])
    } else {
        s.split(',').map(number).collect()
    }
}
fn numbers<T: std::fmt::Display>(v: &[T]) -> String {
    v.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}
fn dtype(s: &str) -> Result<DType> {
    [
        DType::F32,
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
fn op(s: &str) -> Result<Ops> {
    [
        Ops::Invalid,
        Ops::Const,
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
fn encode_arg(a: &Arg) -> String {
    match a {
        Arg::None => "none".into(),
        Arg::Float(v) => format!("float;{:016x}", v.value().to_bits()),
        Arg::I32(v) => format!("i32;{v}"),
        Arg::Int(v) => format!("i64;{v}"),
        Arg::Bool(v) => format!("bool;{}", *v as u8),
        Arg::Range { id, axis } => format!("range;{id};{axis:?}"),
        Arg::Id(v) => format!("id;{v}"),
        Arg::DType(dt) => format!("dtype;{dt:?}"),
        Arg::Text(s) => format!("text;{}", hex(s.as_bytes())),
        Arg::Bytes(v) => format!("bytes;{}", hex(v)),
        Arg::Slot { index, size } => format!("slot;{index};{size}"),
        Arg::ScalarParam { index, dtype } => format!("scalar;{index};{dtype:?}"),
    }
}
fn boolean(s: &str) -> Result<bool> {
    match s {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err("invalid bool".into()),
    }
}
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
        ["range", id, a] => Arg::Range {
            id: number(id)?,
            axis: match *a {
                "Loop" => Axis::Loop,
                _ => return Err("unknown axis".into()),
            },
        },
        ["id", v] => Arg::Id(number(v)?),
        ["dtype", dt] => Arg::DType(dtype(dt)?),
        ["text", v] => Arg::Text(string(v)?),
        ["bytes", v] => Arg::Bytes(unhex(v)?),
        ["slot", i, v] => Arg::Slot {
            index: number(i)?,
            size: number(v)?,
        },
        ["scalar", i, dt] => Arg::ScalarParam {
            index: number(i)?,
            dtype: dtype(dt)?,
        },
        _ => return Err("invalid typed argument".into()),
    })
}
pub fn print(root: &UOp) -> String {
    let nodes = root.toposort();
    let ids: HashMap<_, _> = nodes.iter().enumerate().map(|(i, u)| (u.id, i)).collect();
    let mut text = "UIR 2\n".to_string();
    for (id, u) in nodes.iter().enumerate() {
        text.push_str(&format!(
            "{id}={:?} {:?} [{}] {}\n",
            u.op,
            u.dtype(),
            numbers(&u.src.iter().map(|s| ids[&s.id]).collect::<Vec<_>>()),
            encode_arg(&u.arg)
        ));
    }
    text.push_str(&format!("root={}\n", ids[&root.id]));
    text
}
pub fn parse(text: &str) -> Result<UOp> {
    let mut lines = text.lines();
    if lines.next() != Some("UIR 2") {
        return Err("missing UIR 2 header".into());
    }
    let mut nodes: Vec<UOp> = vec![];
    let mut root = None;
    for line in lines {
        if root.is_some() {
            return Err("text after root".into());
        }
        if let Some(id) = line.strip_prefix("root=") {
            root = Some(
                nodes
                    .get(number::<usize>(id)?)
                    .ok_or("invalid root")?
                    .clone(),
            );
            continue;
        }
        let (id, body) = line.split_once('=').ok_or("missing node id")?;
        if number::<usize>(id)? != nodes.len() {
            return Err("node IDs must be dense".into());
        }
        let mut parts = body.splitn(4, ' ');
        let operation = op(parts.next().ok_or("missing op")?)?;
        let dt = dtype(parts.next().ok_or("missing dtype")?)?;
        let src = parts.next().ok_or("missing sources")?;
        let src = src
            .strip_prefix('[')
            .and_then(|s| s.strip_suffix(']'))
            .ok_or("invalid sources")?;
        let sources = list::<usize>(src)?
            .into_iter()
            .map(|i| {
                nodes
                    .get(i)
                    .cloned()
                    .ok_or("forward or missing source".into())
            })
            .collect::<Result<Vec<_>>>()?;
        let arg = decode_arg(parts.next().ok_or("missing arg")?)?;
        let u = UOp::new(operation, sources, arg);
        if u.dtype() != dt {
            return Err("declared dtype disagrees with UOp".into());
        }
        validate_node(&u)?;
        nodes.push(u);
    }
    let root = root.ok_or("missing root")?;
    if root.toposort().len() != nodes.len() {
        return Err("unreachable UIR nodes".into());
    }
    validate_scope(&root)?;
    Ok(root)
}
fn validate_node(u: &UOp) -> Result<()> {
    if crate::uop::is_binary(u.op) || crate::uop::is_unary(u.op) || u.op == Ops::Where {
        crate::scalar::verify_node(u)?;
    }
    match (&u.op, &u.arg) {
        (Ops::Const, _) => {
            crate::scalar::verify_node(u)?;
        }
        (Ops::Param, Arg::ScalarParam { .. }) => {
            crate::scalar::verify_node(u)?;
        }
        (Ops::Param, Arg::Slot { .. }) if u.src.is_empty() => {}
        (Ops::Param, _) => return Err("invalid leaf metadata".into()),
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
fn validate_scope(root: &UOp) -> Result<()> {
    let mut stack = vec![(root.clone(), None::<Vec<DType>>)];
    let mut seen = HashSet::new();
    while let Some((u, signature)) = stack.pop() {
        if !seen.insert((u.id, signature.clone())) {
            continue;
        }
        if let (Arg::ScalarParam { index, dtype }, Some(params)) = (&u.arg, &signature) {
            if params.get(*index) != Some(dtype) {
                return Err("unbound lexical parameter".into());
            }
        }
        if u.op == Ops::Call && u.src.first().is_some_and(|b| b.op == Ops::Stack) {
            stack.push((
                u.src[0].clone(),
                Some(u.src[1..].iter().map(UOp::dtype).collect()),
            ));
            for arg in &u.src[1..] {
                stack.push((arg.clone(), signature.clone()));
            }
        } else {
            for src in &u.src {
                stack.push((src.clone(), signature.clone()));
            }
        }
    }
    Ok(())
}
pub(crate) fn validate_linear_scope(linear: &UOp) -> Result<()> {
    let mut free: HashMap<u64, HashSet<u64>> = HashMap::new();
    for u in linear.toposort() {
        let mut dependencies = HashSet::new();
        for s in &u.src {
            dependencies.extend(free[&s.id].iter().copied());
        }
        if u.op == Ops::Range {
            dependencies.insert(u.id);
        }
        if u.op == Ops::End {
            dependencies.remove(&u.src.last().ok_or("END lacks range")?.id);
        }
        free.insert(u.id, dependencies);
    }
    let mut active = HashSet::new();
    for u in &linear.src {
        if u.op == Ops::Range {
            active.insert(u.id);
        }
        if !free[&u.id].is_subset(&active) {
            return Err("RANGE value escapes lexical loop scope".into());
        }
        if u.op == Ops::End {
            active.remove(&u.src.last().ok_or("END lacks range")?.id);
        }
    }
    Ok(())
}
pub fn assert_uop(actual: &UOp, expected: &UOp) -> Result<()> {
    if print(actual) == print(expected) {
        Ok(())
    } else {
        Err(format!(
            "UOp mismatch\nactual\n{}expected\n{}",
            print(actual),
            print(expected)
        ))
    }
}
```

`tests/scope.rs`を保存する。

```rust
// file="tests/scope.rs" mode="write"
use chibigrad::{Arg, Axis, Ops, UOp, codegen};
#[test]
fn range_values_do_not_escape_their_scope() {
    let r = UOp::range(UOp::int(2), 0, Axis::Loop);
    let end = UOp::new(Ops::End, vec![r.clone()], Arg::None);
    let escaped = r.add(&UOp::int(1));
    assert!(codegen::verify_lowered(&UOp::linear(vec![r, end, escaped])).is_err());
}
```

```sh
# check step=06.02
cargo test --offline
```

閉じた RANGE を後で足し算に使うグラフを拒否し、正常なマスク付きループは実行できる。

## 章末の累積確認

前の章までのテストとexampleも再ビルドします。

```sh
# check
cargo test --quiet --all-targets
```

## 要件との対応

### 06-001 RANGE と END を検証、interpreter、C renderer へ追加する。

<!-- todo:06-001 -->

関連する実装と確認は本章の `06.01`, `06.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 06-002 body に入ると添字を束縛し、END でスコープを閉じる。

<!-- todo:06-002 -->

関連する実装と確認は本章の `06.01`, `06.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 06-003 LOAD と STORE の範囲を RANGE bounds から証明する。

<!-- todo:06-003 -->

関連する実装と確認は本章の `06.01`, `06.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 06-004 三要素加算を既存の固定 ABI で実行する。

<!-- todo:06-004 -->

関連する実装と確認は本章の `06.01`, `06.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 06-005 第五章の固定添字の例も引き続き通す。

<!-- todo:06-005 -->

関連する実装と確認は本章の `06.01`, `06.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 06-006 builder に外側 body 内で内側 RANGE を作る操作を加える。

<!-- todo:06-006 -->

関連する実装と確認は本章の `06.01`, `06.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 06-007 END が内側、外側の順に閉じることを検証する。

<!-- todo:06-007 -->

関連する実装と確認は本章の `06.01`, `06.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 06-008 `offset = row*3+col` を作り、各反復で LOAD と STORE を行う。

<!-- todo:06-008 -->

関連する実装と確認は本章の `06.01`, `06.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 06-009 `[1,2,3,4,5,6]` のコピーが同じ六要素を返すようにする。

<!-- todo:06-009 -->

関連する実装と確認は本章の `06.01`, `06.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 06-010 RANGE 0 と RANGE 1 を interpreter と C で実行する。

<!-- todo:06-010 -->

関連する実装と確認は本章の `06.01`, `06.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 06-011 空の body が入力を一度も LOAD しないようにする。

<!-- todo:06-011 -->

関連する実装と確認は本章の `06.01`, `06.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 06-012 空出力には有効なホスト所有物を用意し、ポインタを dereference しない。

<!-- todo:06-012 -->

関連する実装と確認は本章の `06.01`, `06.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 06-013 一要素加算 `2+10` を生成コードから `12` として読む。

<!-- todo:06-013 -->

関連する実装と確認は本章の `06.01`, `06.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 06-014 閉じた RANGE の値を別の body で使うグラフを拒否する。

<!-- todo:06-014 -->

関連する実装と確認は本章の `06.01`, `06.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 06-015 交差する END、重複 END、未終了 RANGE を拒否する。

<!-- todo:06-015 -->

関連する実装と確認は本章の `06.01`, `06.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 06-016 `INDEX(x,i+1)` が N 要素を越える場合を生成前に拒否する。

<!-- todo:06-016 -->

関連する実装と確認は本章の `06.01`, `06.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 06-017 bounds の積が i32 を越えるケースを拒否する。

<!-- todo:06-017 -->

関連する実装と確認は本章の `06.01`, `06.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 06-018 `--dump-kernel` に各命令の開いている軸を出す。

<!-- todo:06-018 -->

関連する実装と確認は本章の `06.01`, `06.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 06-019 単一ループの結果が `[11,22,33]` になる。

<!-- todo:06-019 -->

関連する実装と確認は本章の `06.01`, `06.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 06-020 二重ループのコピーが `[1,2,3,4,5,6]` になる。

<!-- todo:06-020 -->

関連する実装と確認は本章の `06.01`, `06.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 06-021 空は空配列、一要素は `[12]` を返す。

<!-- todo:06-021 -->

関連する実装と確認は本章の `06.01`, `06.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 06-022 END の直前には対応する RANGE が開いている。

<!-- todo:06-022 -->

関連する実装と確認は本章の `06.01`, `06.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 06-023 最外側 END の後に未束縛の軸が残らない。

<!-- todo:06-023 -->

関連する実装と確認は本章の `06.01`, `06.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 06-024 すべてのメモリアクセスを bounds と保存領域長から証明できる。

<!-- todo:06-024 -->

関連する実装と確認は本章の `06.01`, `06.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 06-025 範囲外、スコープ違反、添字 overflow は実行前に失敗する。

<!-- todo:06-025 -->

関連する実装と確認は本章の `06.01`, `06.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。
