# 07. LOAD・STORE・AFTER の累積で縮約を実行する

前の節で動かしたコードに、今回使う演算と処理を追加します。各節の確認を実行してから次へ進みます。

参照するUOpの構成は [tinygrad c22edff](https://github.com/tinygrad/tinygrad/blob/c22edff14c79310a9b2df62a78819cf77c8289fa/tinygrad/uop/ops.py) です。最初から全機能を実装せず、使う節で増やします。

## 07.01 局所領域で和を蓄積する

ALLOC でカーネル内の一要素を確保する。初期化、内側 RANGE の LOAD・加算・STORE、END 後の LOAD を AFTER で結ぶ。外側の行ループごとに初期値へ戻す。

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

`src/uir.rs` の `fn op`を置き換える。

```rust
// file="src/uir.rs" mode="replace" item="fn op"
fn op(s: &str) -> Result<Ops> {
    [
        Ops::Invalid,
        Ops::Const,
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
        Arg::Range { id, axis } => format!("range;{id};{axis:?}"),
        Arg::Allocation { id, dtype, size } => {
            format!("allocation;{id};{dtype:?};{size}")
        }
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
        (Ops::Param, Arg::Slot { .. }) | (Ops::Alloc, Arg::Allocation { .. })
            if u.src.is_empty() => {}
        (Ops::Param | Ops::Alloc, _) => return Err("invalid leaf metadata".into()),
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

`src/uop.rs` の `pub enum Axis`を置き換える。

```rust
// file="src/uop.rs" mode="replace" item="pub enum Axis"
pub enum Axis {
    Loop,
    Reduce,
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
        size: usize,
    },
    ScalarParam {
        index: usize,
        dtype: DType,
    },
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
            Ops::Alloc => DType::Ptr,
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

`tests/reduction.rs`を保存する。

```rust
// file="tests/reduction.rs" mode="write"
use chibigrad::{Arg, Axis, DType, Ops, UOp, codegen, cpu::CpuProgram};
fn row_sum(width: usize, values: &[f32]) -> Vec<f32> {
    let input = UOp::new(
        Ops::Param,
        vec![],
        Arg::Slot {
            index: 0,
            size: 2 * width,
        },
    );
    let output = UOp::new(Ops::Param, vec![], Arg::Slot { index: 1, size: 2 });
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
#[test]
fn load_store_after_form_reduction_and_reset_accumulator() {
    assert_eq!(row_sum(3, & [1., 2., 3., 4., 5., 6.]), [6., 15.]);
    assert_eq!(row_sum(0, & []), [0., 0.]);
}
```

```sh
# check step=07.01
cargo test --offline
```

2 行の和が [6,15] になり、長さ 0 の各行は加算の単位元 0 を返す。

## 07.02 最大値と空集合の単位元

行方向の蓄積を共通の UOp 構築関数にする。演算と単位元を対にし、最大値なら負の無限大から始める。和の構造をそのまま使って負の入力も扱う。

`src/kernels.rs`を保存する。

```rust
// file="src/kernels.rs" mode="write"
use crate::{Arg, Axis, DType, Ops, Result, UOp};
pub fn row_reduce(input: UOp, output: UOp, rows: usize, width: usize, op: Ops) -> Result<UOp> {
    let identity = match op {
        Ops::Add => 0.,
        Ops::Max => f32::NEG_INFINITY,
        _ => return Err("row reduction requires Add or Max".into()),
    };
    let row = UOp::range(UOp::int(rows as i64), 0, Axis::Loop);
    let acc = UOp::new(
        Ops::Alloc,
        vec![],
        Arg::Allocation {
            id: 0,
            dtype: DType::F32,
            size: 1,
        },
    );
    let init = acc.store(UOp::f32(identity));
    let col = UOp::range(UOp::int(width as i64), 1, Axis::Reduce);
    let value = input
        .index(row.mul(&UOp::int(width as i64)).add(&col), UOp::bool(true))
        .load();
    let state = acc.after(vec![init.clone(), col.clone()]);
    let update = state.store(state.load().alu(op, &[value]));
    let end = UOp::new(Ops::End, vec![update.clone(), col.clone()], Arg::None);
    let result = acc.after(vec![end.clone()]).load();
    let store = output
        .index(row.clone(), UOp::bool(true))
        .store(result.clone());
    let outer = UOp::new(Ops::End, vec![store.clone(), row.clone()], Arg::None);
    Ok(UOp::linear(vec![
        row, acc, init, col, update, end, result, store, outer,
    ]))
}
```

`src/lib.rs`を追記する。

```rust
// file="src/lib.rs" mode="append"


pub mod kernels;
```

`tests/reduction_max.rs`を保存する。

```rust
// file="tests/reduction_max.rs" mode="write"
use chibigrad::{Arg, Ops, UOp, codegen, cpu::CpuProgram, kernels::row_reduce};
#[test]
fn max_reduction_and_empty_identity() {
    for (width, data, expected) in [
        (3, vec![- 7.0f32, - 2., - 9., 4., 3., 2.], vec![- 2., 4.]),
        (0, vec![], vec![f32::NEG_INFINITY; 2]),
    ] {
        let p = |index, n| UOp::new(Ops::Param, vec![], Arg::Slot { index, size: n });
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

```sh
# check step=07.02
cargo test --offline
```

負の値を含む行最大が [-2,4]、空の行最大が負の無限大になる。

## 章末の累積確認

前の章までのテストとexampleも再ビルドします。

```sh
# check
cargo test --quiet --all-targets
```

## 要件との対応

### 07-001 REG allocation と初期化の region を実装する。

<!-- todo:07-001 -->

関連する実装と確認は本章の `07.01`, `07.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 07-002 反復内 LOAD と STORE を interpreter と C で順に実行する。

<!-- todo:07-002 -->

関連する実装と確認は本章の `07.01`, `07.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 07-003 ループ後の LOAD を END の完了へ依存させる。

<!-- todo:07-003 -->

関連する実装と確認は本章の `07.01`, `07.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 07-004 `[1,2,3,4]` の総和を生成コードで `10` にする。

<!-- todo:07-004 -->

関連する実装と確認は本章の `07.01`, `07.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 07-005 長さ 0 では初期値 `0` を返す。

<!-- todo:07-005 -->

関連する実装と確認は本章の `07.01`, `07.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 07-006 外側 RANGE の body に accumulator 初期化を置く。

<!-- todo:07-006 -->

関連する実装と確認は本章の `07.01`, `07.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 07-007 内側 RANGE で列を累積する。

<!-- todo:07-007 -->

関連する実装と確認は本章の `07.01`, `07.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 07-008 内側 END の後、外側 END の前に行の出力を STORE する。

<!-- todo:07-008 -->

関連する実装と確認は本章の `07.01`, `07.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 07-009 `[1,2,3,4,5,6]` の行和を `[6,15]` にする。

<!-- todo:07-009 -->

関連する実装と確認は本章の `07.01`, `07.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 07-010 空の列数なら各行を `0` にする。

<!-- todo:07-010 -->

関連する実装と確認は本章の `07.01`, `07.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 07-011 `i`、`j`、`k` の三軸と row-major 添字を手で作る。

<!-- todo:07-011 -->

関連する実装と確認は本章の `07.01`, `07.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 07-012 各 `(i,j)` で accumulator をゼロへ戻す。

<!-- todo:07-012 -->

関連する実装と確認は本章の `07.01`, `07.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 07-013 reduction 軸 k の END の後に出力を STORE する。

<!-- todo:07-013 -->

関連する実装と確認は本章の `07.01`, `07.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 07-014 2×3 と 3×2 の入力を既存 CPU runtime で呼ぶ。

<!-- todo:07-014 -->

関連する実装と確認は本章の `07.01`, `07.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 07-015 ホストの独立した三重ループと結果を比較する。

<!-- todo:07-015 -->

関連する実装と確認は本章の `07.01`, `07.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 07-016 出力位置 p と kernel 位置 k を RANGE で作る。

<!-- todo:07-016 -->

関連する実装と確認は本章の `07.01`, `07.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 07-017 `x[p+k]*w[k]` を累積する。

<!-- todo:07-017 -->

関連する実装と確認は本章の `07.01`, `07.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 07-018 `output_len=input_len-kernel_len+1` を checked 演算で求める。

<!-- todo:07-018 -->

関連する実装と確認は本章の `07.01`, `07.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 07-019 `x=[1,2,3,4]`, `w=[2,1]` を `[4,7,10]` にする。

<!-- todo:07-019 -->

関連する実装と確認は本章の `07.01`, `07.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 07-020 kernel が入力より長い場合を明確な未対応エラーにする。

<!-- todo:07-020 -->

関連する実装と確認は本章の `07.01`, `07.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 07-021 総和が `10`、行和が `[6,15]` になる。

<!-- todo:07-021 -->

関連する実装と確認は本章の `07.01`, `07.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 07-022 空の総和と列数 0 の行和が加算の初期値を返す。

<!-- todo:07-022 -->

関連する実装と確認は本章の `07.01`, `07.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 07-023 GEMM が `[[22,28],[49,64]]`、convolution が `[4,7,10]` になる。

<!-- todo:07-023 -->

関連する実装と確認は本章の `07.01`, `07.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 07-024 dump に高水準 REDUCE がなく、累積を行う LOAD と STORE がある。

<!-- todo:07-024 -->

関連する実装と確認は本章の `07.01`, `07.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 07-025 accumulator の初期化が正しい外側 region にある。

<!-- todo:07-025 -->

関連する実装と確認は本章の `07.01`, `07.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 07-026 入力を変えた二回目の run でも前回の accumulator が残らない。

<!-- todo:07-026 -->

関連する実装と確認は本章の `07.01`, `07.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 07-027 一般 f32 は前章までの予定許容誤差で interpreter と CPU を比較する。

<!-- todo:07-027 -->

関連する実装と確認は本章の `07.01`, `07.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。
