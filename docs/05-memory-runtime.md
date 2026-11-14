# 05. INDEX・LOAD・STORE・AFTER でメモリを扱う

前の節で動かしたコードに、今回使う演算と処理を追加します。各節の確認を実行してから次へ進みます。

参照するUOpの構成は [tinygrad c22edff](https://github.com/tinygrad/tinygrad/blob/c22edff14c79310a9b2df62a78819cf77c8289fa/tinygrad/uop/ops.py) です。最初から全機能を実装せず、使う節で増やします。

## 05.01 INDEX・LOAD・STORE を C に下げる

メモリの仮引数はポインタスロットと要素数で表す。INDEX は位置と有効条件を持ち、LOAD と STORE を生成する。スカラーのコンパイルもこの共通の LINEAR と C 生成器へ移し、前節までの整数検査を残す。

`src/codegen.rs`を保存する。

```rust
// file="src/codegen.rs" mode="write"
use crate::{
    Result,
    uop::{Arg, DType, Ops, UOp, is_binary},
};
use std::collections::{HashMap, HashSet};
pub fn normalize(root: &UOp) -> Result<UOp> {
    crate::functions::inline_calls(root)
}
pub fn linearize(sink: &UOp) -> Result<UOp> {
    let sink = crate::functions::inline_calls(sink)?;
    if sink.op == Ops::Linear {
        return Ok(sink.clone());
    }
    if sink.op != Ops::Sink {
        return Err("linearize expects SINK".into());
    }
    let linear = UOp::linear(
        sink.toposort()
            .into_iter()
            .filter(|x| x.op != Ops::Sink)
            .collect(),
    );
    verify_lowered(&linear)?;
    Ok(linear)
}
pub fn verify_lowered(linear: &UOp) -> Result<()> {
    if linear.op != Ops::Linear {
        return Err("renderer expects LINEAR".into());
    }
    for u in &linear.src {
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
    Ok(())
}
pub fn to_program(sink: &UOp, entry: &str) -> Result<UOp> {
    let linear = linearize(sink)?;
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
#[derive(Clone, Debug)]
pub struct Signature {
    pub buffers: Vec<(usize, UOp)>,
}

pub fn signature(linear: &UOp) -> Result<Signature> {
    let mut slots = std::collections::BTreeMap::new();
    for u in linear.toposort() {
        if let (Ops::Param, Arg::Slot { index, .. }) = (u.op, &u.arg) {
            if slots.get(index).is_some_and(|old| old != &u) {
                return Err("conflicting pointer parameter slot".into());
            }
            slots.insert(*index, u);
        }
    }
    Ok(Signature {
        buffers: slots.into_iter().collect(),
    })
}

fn ctype(dt: DType) -> &'static str {
    match dt {
        DType::F32 => "float",
        DType::I32 => "int",
        DType::I64 => "long long",
        DType::Bool => "int",
        DType::Ptr => "void*",
        DType::Void => "void",
    }
}
fn identifier(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c == '_' || c.is_ascii_alphabetic())
        && chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}
pub fn render(linear: &UOp, entry: &str) -> Result<String> {
    render_impl(linear, entry, None)
}
pub fn render_checked(linear: &UOp, entry: &str, status_slot: usize) -> Result<String> {
    render_impl(linear, entry, Some(status_slot))
}
fn render_impl(linear: &UOp, entry: &str, status_slot: Option<usize>) -> Result<String> {
    let mut all = vec![];
    let mut seen = HashSet::new();
    let mut effects = HashSet::new();
    fn add(u: &UOp, out: &mut Vec<UOp>, seen: &mut HashSet<u64>, effects: &mut HashSet<u64>) {
        if seen.contains(&u.id) || effects.contains(&u.id) {
            return;
        }
        for src in u.ordered_sources() {
            add(src, out, seen, effects)
        }

        seen.insert(u.id);
        out.push(u.clone());
    }
    for u in &linear.src {
        add(u, &mut all, &mut seen, &mut effects)
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
    let depth = 1;
    let mut occurrence = 0usize;
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
            (Ops::Const, Arg::F32(bits)) => {
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
fn pointer_size(u: &UOp) -> Option<usize> {
    match &u.arg {
        Arg::Slot { size, .. } => Some(*size),
        _ => None,
    }
}
fn index_bounds(u: &UOp) -> Option<(i64, i64)> {
    match &u.arg {
        Arg::Int(n) => Some((*n, *n)),
        _ => None,
    }
}
```

`src/lib.rs`を追記する。

```rust
// file="src/lib.rs" mode="append"

pub mod codegen;
```

`src/scalar.rs`を保存する。

```rust
// file="src/scalar.rs" mode="write"
//! A scalar oracle and checked native compilation over the common UOp DAG.
use crate::{Arg, DType, Ops, Result, UOp};
use std::collections::HashMap;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Value {
    Bool(bool),
    I32(i32),
    I64(i64),
    F32Bits(u32),
}
impl Value {
    pub fn f32(v: f32) -> Self {
        Self::F32Bits(v.to_bits())
    }
    pub fn dtype(self) -> DType {
        match self {
            Self::Bool(_) => DType::Bool,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::F32Bits(_) => DType::F32,
        }
    }
    pub fn number(self) -> f64 {
        match self {
            Self::Bool(v) => v as u8 as f64,
            Self::I32(v) => v as f64,
            Self::I64(v) => v as f64,
            Self::F32Bits(v) => f32::from_bits(v) as f64,
        }
    }
    pub fn uop(self) -> UOp {
        match self {
            Self::Bool(v) => UOp::bool(v),
            Self::I32(v) => UOp::i32(v),
            Self::I64(v) => UOp::int(v),
            Self::F32Bits(v) => UOp::f32(f32::from_bits(v)),
        }
    }
    pub fn from_constant(u: &UOp) -> Option<Self> {
        if u.op != Ops::Const {
            return None;
        }
        Some(
            match u.arg {
                Arg::Bool(v) => Self::Bool(v),
                Arg::I32(v) => Self::I32(v),
                Arg::Int(v) => Self::I64(v),
                Arg::F32(v) => Self::F32Bits(v),
                _ => return None,
            },
        )
    }
    fn pack(self) -> u64 {
        match self {
            Self::Bool(v) => v as u64,
            Self::I32(v) => v as u32 as u64,
            Self::I64(v) => v as u64,
            Self::F32Bits(v) => v as u64,
        }
    }
    fn unpack(dt: DType, v: u64) -> Result<Self> {
        Ok(
            match dt {
                DType::Bool => Self::Bool(v != 0),
                DType::I32 => Self::I32(v as i32),
                DType::I64 => Self::I64(v as i64),
                DType::F32 => Self::F32Bits(v as u32),
                _ => return Err("not a scalar dtype".into()),
            },
        )
    }
}
fn scalar_type(dt: DType) -> bool {
    matches!(dt, DType::Bool | DType::I32 | DType::I64 | DType::F32)
}
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
pub fn verify(root: &UOp) -> Result<Vec<DType>> {
    let mut params = HashMap::new();
    for u in root.toposort() {
        verify_node(&u)?;
        if let Arg::ScalarParam { index, dtype } = u.arg {
            if params.insert(index, dtype).is_some_and(|old| old != dtype) {
                return Err("inconsistent parameter dtype".into());
            }
        }
        if u.src.iter().all(|s| s.op == Ops::Const) && !u.src.is_empty() {
            let v = u
                .src
                .iter()
                .map(|s| Value::from_constant(s).unwrap())
                .collect::<Vec<_>>();
            eval_alu(u.op, &u.arg, &v)?;
        }
    }
    (0..params.len())
        .map(|i| {
            params.get(&i).copied().ok_or("non-dense scalar parameter slots".into())
        })
        .collect()
}
fn inputs_valid(types: &[DType], inputs: &[Value]) -> Result<()> {
    if types.len() != inputs.len()
        || types.iter().zip(inputs).any(|(t, v)| *t != v.dtype())
    {
        return Err("scalar input signature mismatch".into());
    }
    Ok(())
}
pub fn evaluate(root: &UOp, inputs: &[Value]) -> Result<Value> {
    let root = crate::functions::inline_calls(root)?;
    inputs_valid(&verify(&root)?, inputs)?;
    let mut values = HashMap::new();
    for u in root.toposort() {
        let value = if let Some(v) = Value::from_constant(&u) {
            v
        } else if let Arg::ScalarParam { index, .. } = u.arg {
            inputs[index]
        } else {
            eval_alu(
                u.op,
                &u.arg,
                &u.src.iter().map(|s| values[&s.id]).collect::<Vec<_>>(),
            )?
        };
        values.insert(u.id, value);
    }
    Ok(values[&root.id])
}
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
                        _ => return Err("invalid F32 operation".into()),
                    },
                )
            }
        },
    )
}
pub struct CompiledScalar {
    pub source: String,
    program: crate::cpu::CpuProgram,
    types: Vec<DType>,
    output: DType,
}
pub fn compile(root: &UOp) -> Result<CompiledScalar> {
    compile_with_guards(root, &[])
}
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
                    size: 1,
                },
            );
            UOp::new(Ops::Load, vec![ptr], Arg::DType(dtype))
        } else {
            u.replace(u.src.iter().map(|s| mapped[&s.id].clone()).collect())
        };
        mapped.insert(u.id, replacement);
    }
    let ptr = UOp::new(Ops::Param, vec![], Arg::Slot { index: 0, size: 1 });
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
            size: 1,
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
impl CompiledScalar {
    pub fn run(&self, inputs: &[Value]) -> Result<Value> {
        inputs_valid(&self.types, inputs)?;
        let words: Vec<_> = inputs.iter().map(|v| v.pack()).collect();
        let mut output = 0u64;
        let mut status = 0i32;
        let mut args = vec![(& mut output as * mut u64) as u64];
        args.extend(words.iter().map(|w| (w as *const u64) as u64));
        args.push((&mut status as *mut i32) as u64);
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                args.as_mut_ptr().cast::<u8>(),
                args.len() * 8,
            )
        };
        unsafe { self.program.run(bytes) }?;
        if status != 0 {
            return Err("unsafe native scalar operation".into());
        }
        Value::unpack(self.output, output)
    }
}
pub(crate) fn floor_div(a: i64, b: i64) -> Option<i64> {
    let q = a.checked_div(b)?;
    let r = a.checked_rem(b)?;
    q.checked_sub((r != 0 && (r < 0) != (b < 0)) as i64)
}
pub(crate) fn floor_mod(a: i64, b: i64) -> Option<i64> {
    let r = a.checked_rem(b)?;
    if r != 0 && (r < 0) != (b < 0) { r.checked_add(b) } else { Some(r) }
}
```

`src/uir.rs` の `fn dtype`を置き換える。

```rust
// file="src/uir.rs" mode="replace" item="fn dtype"
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
        Ops::Index,
        Ops::Load,
        Ops::Store,
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
        Arg::F32(v) => format!("f32;{v:08x}"),
        Arg::I32(v) => format!("i32;{v}"),
        Arg::Int(v) => format!("i64;{v}"),
        Arg::Bool(v) => format!("bool;{}", *v as u8),
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
        ["f32", v] if v.len() == 8 => {
            Arg::F32(u32::from_str_radix(v, 16).map_err(|_| "invalid f32 bits")?)
        }
        ["i32", v] => Arg::I32(number(v)?),
        ["i64", v] => Arg::Int(number(v)?),
        ["bool", v] => Arg::Bool(boolean(v)?),
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
    Index,
    Load,
    Store,
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
    F32(u32),
    I32(i32),
    Int(i64),
    Bool(bool),
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
                Arg::F32(_) => DType::F32,
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
    pub fn index(&self, index: UOp, valid: UOp) -> Self {
        Self::new(Ops::Index, vec![self.clone(), index, valid], Arg::None)
    }

    pub fn load(&self) -> Self {
        Self::new(Ops::Load, vec![self.clone()], Arg::None)
    }

    pub fn store(&self, value: UOp) -> Self {
        Self::new(Ops::Store, vec![self.clone(), value], Arg::None)
    }

    pub fn linear(src: Vec<UOp>) -> Self {
        Self::new(Ops::Linear, src, Arg::None)
    }
```

`tests/memory.rs`を保存する。

```rust
// file="tests/memory.rs" mode="write"
use chibigrad::{Arg, Ops, UOp, codegen, cpu::CpuProgram};
fn ptr(index: usize) -> UOp {
    UOp::new(Ops::Param, vec![], Arg::Slot { index, size: 1 })
}
fn execute(sink: UOp, buffers: &mut [Vec<f32>]) {
    let p = codegen::to_program(&sink, "memory").unwrap();
    let Arg::Text(source) = &p.src[2].arg else { panic!() };
    let native = unsafe { CpuProgram::compile(source, "memory") }.unwrap();
    let mut args = buffers
        .iter_mut()
        .flat_map(|b| (b.as_mut_ptr() as u64).to_ne_bytes())
        .collect::<Vec<_>>();
    unsafe { native.run(&mut args) }.unwrap();
}
#[test]
fn scalar_memory_and_mask() {
    let x = ptr(0);
    let y = ptr(1);
    let value = x.index(UOp::int(0), UOp::bool(true)).load().mul(&UOp::f32(2.));
    let masked = x.index(UOp::int(999), UOp::bool(false)).load();
    let stores = vec![
        y.index(UOp::int(0), UOp::bool(true)).store(value.add(& masked)), y
        .index(UOp::int(999), UOp::bool(false)).store(UOp::f32(99.)),
    ];
    let mut buffers = vec![vec![3.], vec![0., 12345.]];
    execute(UOp::sink(stores), &mut buffers);
    assert_eq!(buffers[1], [6., 12345.]);
}
```

```sh
# check step=05.01
cargo test --offline
```

LOAD と STORE が実メモリを更新し、偽のマスクは範囲外の 999 番地へ触れず番兵を保つ。

## 05.02 AFTER で書込み後の値を読む

AFTER は値と先に済ませる効果を別々の入力に持つ。走査は依存する STORE を先に並べ、その後の LOAD が新しい値を見る。アドレスと値の見かけの共有だけでは表せない実行順序を UOp に含める。

`src/codegen.rs` の `fn render_impl`を置き換える。

```rust
// file="src/codegen.rs" mode="replace" item="fn render_impl"
fn render_impl(linear: &UOp, entry: &str, status_slot: Option<usize>) -> Result<String> {
    let mut all = vec![];
    let mut seen = HashSet::new();
    let mut effects = HashSet::new();
    fn add(u: &UOp, out: &mut Vec<UOp>, seen: &mut HashSet<u64>, effects: &mut HashSet<u64>) {
        if seen.contains(&u.id) || effects.contains(&u.id) {
            return;
        }
        for src in u.ordered_sources() {
            add(src, out, seen, effects)
        }

        seen.insert(u.id);
        out.push(u.clone());
    }
    for u in &linear.src {
        add(u, &mut all, &mut seen, &mut effects)
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
    let depth = 1;
    let mut occurrence = 0usize;
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
            (Ops::Const, Arg::F32(bits)) => {
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
        Ops::Index,
        Ops::Load,
        Ops::Store,
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
    Index,
    Load,
    Store,
    After,
    Call,
    Sink,
    Linear,
    Program,
    Source,
    Binary,
}
```

`src/uop.rs` の `impl UOp` 内 の `pub fn ordered_sources`を置き換える。

```rust
// file="src/uop.rs" mode="replace" within="impl UOp" item="pub fn ordered_sources"
    pub fn ordered_sources(&self) -> Vec<&UOp> {
        if self.op == Ops::After && !self.src.is_empty() {
            self.src[1..].iter().chain(self.src[..1].iter()).collect()
        } else {
            self.src.iter().collect()
        }
    }
```

`src/uop.rs` の `impl UOp` 内を追記する。

```rust
// file="src/uop.rs" mode="append" within="impl UOp"
    pub fn after(&self, deps: Vec<UOp>) -> Self {
        let mut src = vec![self.clone()];
        src.extend(deps);
        Self::new(Ops::After, src, Arg::None)
    }
```

`tests/memory.rs`を追記する。

```rust
// file="tests/memory.rs" mode="append"
#[test]
fn after_observes_write_before_load() {
    let x = ptr(0);
    let write = x.store(UOp::f32(7.));
    let value = x.after(vec![write]).load();
    let mut buffers = vec![vec![3.], vec![0.]];
    execute(UOp::sink(vec![ptr(1).store(value)]), &mut buffers);
    assert_eq!(buffers[0], [7.]);
    assert_eq!(buffers[1], [7.]);
}
```

```sh
# check step=05.02
cargo test --offline
```

同じ領域へ 7 を書いた後の LOAD が 7 を返す。スカラー参照評価との一致も保つ。

## 章末の累積確認

前の章までのテストとexampleも再ビルドします。

```sh
# check
cargo test --quiet --all-targets
```

## 要件との対応

### 05-001 BUFFER の束縛と ALLOC の固有 ID を追加する。

<!-- todo:05-001 -->

関連する実装と確認は本章の `05.01`, `05.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 05-002 メモリ INDEX、LOAD、STORE を C と検証用 interpreter に加える。

<!-- todo:05-002 -->

関連する実装と確認は本章の `05.01`, `05.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 05-003 buffer の PARAM を固定 ABI のポインタ配列へ接続する。

<!-- todo:05-003 -->

関連する実装と確認は本章の `05.01`, `05.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 05-004 入力の長さと dtype を呼出し前に照合する。

<!-- todo:05-004 -->

関連する実装と確認は本章の `05.01`, `05.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 05-005 `[10,20,30]` から生成コードで `31` を得る。

<!-- todo:05-005 -->

関連する実装と確認は本章の `05.01`, `05.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 05-006 効果を依存順に実行し、到達可能な STORE を消さない。

<!-- todo:05-006 -->

関連する実装と確認は本章の `05.01`, `05.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 05-007 AFTER の返す storage identity と状態を区別する。

<!-- todo:05-007 -->

関連する実装と確認は本章の `05.01`, `05.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 05-008 同じ位置へ `1`、次に `2` を書き、その後に読んで `2` を返す。

<!-- todo:05-008 -->

関連する実装と確認は本章の `05.01`, `05.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 05-009 LOAD の共有キーへ状態を含め、異なる時点の読出しを共有しない。

<!-- todo:05-009 -->

関連する実装と確認は本章の `05.01`, `05.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 05-010 複数の効果を持つ CALL の出力を AFTER から読めるようにする。

<!-- todo:05-010 -->

関連する実装と確認は本章の `05.01`, `05.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 05-011 `SHRINK(x,1,3)` と base offset を導出する。

<!-- todo:05-011 -->

関連する実装と確認は本章の `05.01`, `05.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 05-012 view の INDEX 0 を元 buffer の INDEX 1 へ変換する。

<!-- todo:05-012 -->

関連する実装と確認は本章の `05.01`, `05.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 05-013 storage_base と read、write の集合を dump に出す。

<!-- todo:05-013 -->

関連する実装と確認は本章の `05.01`, `05.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 05-014 二入力 `[1,2,3]`, `[10,20,30]` を展開して `[11,22,33]` を作る。

<!-- todo:05-014 -->

関連する実装と確認は本章の `05.01`, `05.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 05-015 同じ長さの二つの ALLOC が互いを上書きしないことを確かめる。

<!-- todo:05-015 -->

関連する実装と確認は本章の `05.01`, `05.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 05-016 dtype、アドレス kind、保存領域の可変性を検証する。

<!-- todo:05-016 -->

関連する実装と確認は本章の `05.01`, `05.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 05-017 SHRINK の `0 <= start <= end <= len` を検証する。

<!-- todo:05-017 -->

関連する実装と確認は本章の `05.01`, `05.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 05-018 静的 INDEX が view の範囲内にあることを検証する。

<!-- todo:05-018 -->

関連する実装と確認は本章の `05.01`, `05.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 05-019 未初期化出力の LOAD と AFTER の循環を拒否する。

<!-- todo:05-019 -->

関連する実装と確認は本章の `05.01`, `05.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 05-020 `len*itemsize` と offset の overflow をホスト上の checked 演算で拒否する。

<!-- todo:05-020 -->

関連する実装と確認は本章の `05.01`, `05.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 05-021 固定アクセスの結果が `31`、view の結果が `20` になる。

<!-- todo:05-021 -->

関連する実装と確認は本章の `05.01`, `05.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 05-022 二入力の三要素加算が `[11,22,33]` になる。

<!-- todo:05-022 -->

関連する実装と確認は本章の `05.01`, `05.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 05-023 書込み前後の LOAD が state を区別し、最終値 `2` を読む。

<!-- todo:05-023 -->

関連する実装と確認は本章の `05.01`, `05.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 05-024 dump の base identity は view 作成前後で一致する。

<!-- todo:05-024 -->

関連する実装と確認は本章の `05.01`, `05.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 05-025 accesses が入力を read、出力を write として列挙する。

<!-- todo:05-025 -->

関連する実装と確認は本章の `05.01`, `05.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 05-026 範囲外、型不一致、未初期化、効果循環が実行前にエラーになる。

<!-- todo:05-026 -->

関連する実装と確認は本章の `05.01`, `05.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。
