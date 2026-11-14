# 04. CALL のスコープ、STACK の選択、UIR の保存と復元を実装する

前の節で動かしたコードに、今回使う演算と処理を追加します。各節の確認を実行してから次へ進みます。

参照するUOpの構成は [tinygrad c22edff](https://github.com/tinygrad/tinygrad/blob/c22edff14c79310a9b2df62a78819cf77c8289fa/tinygrad/uop/ops.py) です。最初から全機能を実装せず、使う節で増やします。

## 04.01 CALL と STACK で関数を表す

関数の出力を STACK にまとめ、CALL に本体と実引数を置く。仮引数を呼出しごとの環境で置き換え、定数 INDEX で出力を選ぶ。関数内の同じ番号の仮引数が外側の呼出しへ漏れない。

`src/functions.rs`を保存する。

```rust
// file="src/functions.rs" mode="write"
use crate::{Arg, DType, Ops, Result, UOp};
#[derive(Clone)]
pub struct Function {
    body: UOp,
    parameters: Vec<DType>,
}
impl Function {
    pub fn parameter(index: usize, dtype: DType) -> UOp {
        UOp::new(Ops::Param, vec![], Arg::ScalarParam { index, dtype })
    }
    pub fn new(parameters: &[DType], outputs: &[UOp]) -> Result<Self> {
        let first = outputs.first().ok_or("function needs an output")?;
        if outputs.iter().any(|u| u.dtype() != first.dtype()) {
            return Err("STACK outputs require one dtype".into());
        }
        for output in outputs {
            for u in output.toposort() {
                if let Arg::ScalarParam { index, dtype } = u.arg {
                    if parameters.get(index) != Some(&dtype) {
                        return Err("function parameter signature mismatch".into());
                    }
                }
            }
        }
        Ok(Self {
            body: UOp::new(Ops::Stack, outputs.to_vec(), Arg::None),
            parameters: parameters.to_vec(),
        })
    }
    pub fn call(&self, arguments: &[UOp]) -> Result<UOp> {
        if arguments.len() != self.parameters.len()
            || arguments
                .iter()
                .zip(&self.parameters)
                .any(|(u, d)| u.dtype() != *d)
        {
            return Err("function argument signature mismatch".into());
        }
        let mut sources = vec![self.body.clone()];
        sources.extend_from_slice(arguments);
        Ok(UOp::new(Ops::Call, sources, Arg::Id(self.parameters.len())))
    }
    pub fn select(&self, call: &UOp, index: &UOp) -> Result<UOp> {
        if call.op != Ops::Call || call.src.first() != Some(&self.body) {
            return Err("call belongs to another function".into());
        }
        let bounds = selector_bounds(index)?;
        if bounds.0 < 0 || bounds.1 >= self.body.src.len() as i64 {
            return Err("STACK index outside outputs".into());
        }
        Ok(UOp::new(
            Ops::Index,
            vec![call.clone(), index.clone(), UOp::bool(true)],
            Arg::None,
        ))
    }
}
pub fn inline_calls(root: &UOp) -> Result<UOp> {
    if !root
        .toposort()
        .iter()
        .any(|u| u.op == Ops::Call && u.src.first().is_some_and(|b| b.op == Ops::Stack))
    {
        return Ok(root.clone());
    }
    fn call_outputs(
        call: &UOp,
        environment: Option<&[UOp]>,
        budget: &mut usize,
    ) -> Result<Vec<UOp>> {
        let body = call.src.first().ok_or("CALL missing body")?;
        let Arg::Id(count) = call.arg else {
            return Err("pure CALL missing parameter count".into());
        };
        if body.op != Ops::Stack || call.src.len() != count + 1 {
            return Err("pure CALL signature".into());
        }
        let args = call.src[1..]
            .iter()
            .map(|u| visit(u, environment, budget))
            .collect::<Result<Vec<_>>>()?;
        body.src
            .iter()
            .map(|u| visit(u, Some(&args), budget))
            .collect()
    }
    fn visit(node: &UOp, environment: Option<&[UOp]>, budget: &mut usize) -> Result<UOp> {
        if *budget == 0 {
            return Err("function expansion budget exhausted".into());
        }
        *budget -= 1;
        if let (Ops::Param, Arg::ScalarParam { index, dtype }) = (node.op, &node.arg) {
            if let Some(args) = environment {
                let argument = args.get(*index).ok_or("missing scoped parameter")?;
                if argument.dtype() != *dtype {
                    return Err("scoped parameter dtype mismatch".into());
                }
                return Ok(argument.clone());
            }
        }
        if node.op == Ops::Index
            && node.src.first().is_some_and(|u| {
                u.op == Ops::Call && u.src.first().is_some_and(|b| b.op == Ops::Stack)
            })
        {
            let outputs = call_outputs(&node.src[0], environment, budget)?;
            let index = visit(
                node.src.get(1).ok_or("INDEX missing selector")?,
                environment,
                budget,
            )?;
            let bounds = selector_bounds(&index)?;
            if bounds.0 < 0 || bounds.1 >= outputs.len() as i64 {
                return Err("STACK selector outside outputs".into());
            }
            let mut result = outputs[0].clone();
            for (i, value) in outputs.iter().enumerate().skip(1) {
                result = index.eq(&UOp::int(i as i64)).where_(value, &result);
            }
            return Ok(result);
        }
        if node.op == Ops::Call && node.src.first().is_some_and(|body| body.op == Ops::Stack) {
            let outputs = call_outputs(node, environment, budget)?;
            if outputs.len() != 1 {
                return Err("multiple CALL outputs need INDEX".into());
            }
            return Ok(outputs[0].clone());
        }
        Ok(node.replace(
            node.src
                .iter()
                .map(|u| visit(u, environment, budget))
                .collect::<Result<Vec<_>>>()?,
        ))
    }
    visit(root, None, &mut 1_000_000)
}
fn selector_bounds(index: &UOp) -> Result<(i64, i64)> {
    match index.arg {
        Arg::Int(n) => Ok((n, n)),
        _ => Err("STACK selector must be a constant integer".into()),
    }
}
```

`src/lib.rs`を追記する。

```rust
// file="src/lib.rs" mode="append"

pub mod functions;
```

`src/pattern.rs` の `pub struct RewriteOptions`を置き換える。

```rust
// file="src/pattern.rs" mode="replace" item="pub struct RewriteOptions"
pub struct RewriteOptions {
    pub order: RewriteOrder,
    pub walk: bool,
    pub enter_calls: bool,
    pub max_steps: usize,
}
```

`src/pattern.rs` の `impl Default for RewriteOptions` 内 の `fn default`を置き換える。

```rust
// file="src/pattern.rs" mode="replace" within="impl Default for RewriteOptions" item="fn default"
    fn default() -> Self {
        Self {
            order: RewriteOrder::BottomUp,
            walk: true,
            enter_calls: false,
            max_steps: 100_000,
        }
    }
```

`src/pattern.rs` の `pub fn graph_rewrite_with_options`を置き換える。

```rust
// file="src/pattern.rs" mode="replace" item="pub fn graph_rewrite_with_options"
pub fn graph_rewrite_with_options(
    root: &UOp,
    matcher: &PatternMatcher,
    options: RewriteOptions,
) -> Result<UOp> {
    enum Frame {
        Enter(UOp),
        Finish(UOp),
        Alias(UOp, UOp),
    }
    let mut stack = vec![Frame::Enter(root.clone())];
    let mut memo: HashMap<u64, UOp> = HashMap::new();
    let mut active = HashSet::new();
    let mut remaining = options.max_steps;
    let skip_body = |u: &UOp, i: usize| !options.enter_calls && u.op == Ops::Call && i == 0;
    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Enter(u) => {
                if memo.contains_key(&u.id) {
                    continue;
                }
                if remaining == 0 || !active.insert(u.id) {
                    return Err("rewrite cycle or budget exhausted".into());
                }
                remaining -= 1;
                if options.order == RewriteOrder::TopDown {
                    if let Some(next) = matcher.rewrite(&u) {
                        stack.push(Frame::Alias(u, next.clone()));
                        stack.push(Frame::Enter(next));
                        continue;
                    }
                }
                stack.push(Frame::Finish(u.clone()));
                if options.walk {
                    for (i, src) in u.src.iter().enumerate().rev() {
                        if !skip_body(&u, i) {
                            stack.push(Frame::Enter(src.clone()));
                        }
                    }
                }
            }
            Frame::Finish(u) => {
                let src = u
                    .src
                    .iter()
                    .enumerate()
                    .map(|(i, s)| {
                        if options.walk && !skip_body(&u, i) {
                            memo.get(&s.id)
                                .cloned()
                                .ok_or("rewrite source missing".to_string())
                        } else {
                            Ok(s.clone())
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                let rewritten = u.replace(src);
                if options.order == RewriteOrder::BottomUp {
                    if let Some(next) = matcher.rewrite(&rewritten) {
                        stack.push(Frame::Alias(u, next.clone()));
                        stack.push(Frame::Enter(next));
                        continue;
                    }
                }
                active.remove(&u.id);
                memo.insert(u.id, rewritten);
            }
            Frame::Alias(original, replacement) => {
                let result = memo
                    .get(&replacement.id)
                    .ok_or("replacement missing")?
                    .clone();
                active.remove(&original.id);
                memo.insert(original.id, result);
            }
        }
    }
    memo.remove(&root.id)
        .ok_or("rewrite produced no root".into())
}
```

`src/scalar.rs` の `pub fn evaluate`を置き換える。

```rust
// file="src/scalar.rs" mode="replace" item="pub fn evaluate"
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
```

`src/scalar.rs` の `pub fn compile`を置き換える。

```rust
// file="src/scalar.rs" mode="replace" item="pub fn compile"
pub fn compile(root: &UOp) -> Result<CompiledScalar> {
    let root = crate::functions::inline_calls(root)?;
    let types = verify(&root)?;
    let mut source = String::from(
        "static float as_float(unsigned int b){union{unsigned int b;float f;}v;v.b=b;return v.f;}\nstatic unsigned int bits(float f){union{unsigned int b;float f;}v;v.f=f;return v.b;}\nstatic long long floor_div(long long a,long long b){long long q=a/b,r=a%b;return q-((r!=0)&&((r<0)!=(b<0)));}\nstatic long long floor_mod(long long a,long long b){long long r=a%b;return r+(((r!=0)&&((r<0)!=(b<0)))?b:0);}\nvoid chibi_scalar(unsigned char *packed){unsigned long long *args=(unsigned long long*)packed;int *status=(int*)args[0];*status=0;\n",
    );
    let mut values: HashMap<u64, String> = HashMap::new();
    for u in root.toposort() {
        let at = |i: usize| values[&u.src[i].id].clone();
        let reject = "{*status=1;return;}";
        if matches!(u.op, Ops::Div | Ops::Mod)
            && matches!(u.dtype(), DType::I32 | DType::I64)
        {
            let min = if u.dtype() == DType::I32 {
                "(-2147483647-1)"
            } else {
                "(-9223372036854775807ll-1)"
            };
            source
                .push_str(
                    &format!(
                        "if({}==0||({}=={}&&{}==-1)){}\n", at(1), at(0), min, at(1),
                        reject
                    ),
                );
        }
        if matches!(u.op, Ops::Shl | Ops::Shr) {
            source
                .push_str(
                    &format!(
                        "if({}<0||{}>={}){}\n", at(1), at(1), if u.dtype() == DType::I32
                        { 32 } else { 64 }, reject
                    ),
                );
        }
        if u.op == Ops::Cast && u.src[0].dtype() == DType::F32
            && matches!(u.dtype(), DType::I32 | DType::I64)
        {
            let lim = if u.dtype() == DType::I32 {
                "2147483648.0f"
            } else {
                "9223372036854775808.0f"
            };
            source
                .push_str(
                    &format!("if(!({}>=-{}&&{}<{})){}\n", at(0), lim, at(0), lim, reject),
                );
        }
        let expression = match (&u.op, &u.arg) {
            (Ops::Const, Arg::Float(n)) => format!("as_float({}u)", (n.value() as f32).to_bits()),
            (Ops::Const, Arg::I32(n)) => format!("((int){}u)", * n as u32),
            (Ops::Const, Arg::Int(n)) => format!("((long long){}ull)", * n as u64),
            (Ops::Const, Arg::Bool(n)) => (*n as u8).to_string(),
            (Ops::Param, Arg::ScalarParam { index, dtype }) => {
                if *dtype == DType::F32 {
                    format!("as_float((unsigned int)args[{}])", index + 2)
                } else {
                    format!("(({})args[{}])", ctype(* dtype), index + 2)
                }
            }
            (Ops::Where, _) => format!("({}?{}:{})", at(0), at(1), at(2)),
            (Ops::Cast, Arg::DType(dt)) => format!("(({}){})", ctype(* dt), at(0)),
            (Ops::Bitcast, Arg::DType(DType::F32)) => {
                format!("as_float((unsigned int){})", at(0))
            }
            (Ops::Bitcast, Arg::DType(DType::I32)) => format!("((int)bits({}))", at(0)),
            (Ops::Neg, _) => {
                if u.dtype() == DType::I64 {
                    source
                        .push_str(
                            &format!(
                                "if({}==(-9223372036854775807ll-1)){}\n", at(0), reject
                            ),
                        );
                }
                format!("(-{})", at(0))
            }
            (Ops::Max, _) if u.dtype() == DType::F32 => {
                format!(
                    "({0}!={0}||{1}!={1}?as_float(2143289344u):({0}==0.0f&&{1}==0.0f?0.0f:({0}>{1}?{0}:{1})))",
                    at(0), at(1)
                )
            }
            (Ops::Max, _) => format!("({0}>{1}?{0}:{1})", at(0), at(1)),
            (Ops::Div, _) if u.dtype() != DType::F32 => {
                format!("floor_div({},{})", at(0), at(1))
            }
            (Ops::Mod, _) => format!("floor_mod({},{})", at(0), at(1)),
            (op, _) => {
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
                    Ops::Shl => "<<",
                    Ops::Shr => ">>",
                    _ => return Err("unsupported scalar C expression".into()),
                };
                if u.dtype() == DType::I64
                    && matches!(op, Ops::Add | Ops::Sub | Ops::Mul)
                {
                    let op = match op {
                        Ops::Add => "add",
                        Ops::Sub => "sub",
                        _ => "mul",
                    };
                    source
                        .push_str(
                            &format!(
                                "long long checked{};if(__builtin_{}_overflow({},{},&checked{})){}\n",
                                u.id, op, at(0), at(1), u.id, reject
                            ),
                        );
                }
                if u.dtype() == DType::I32
                    && matches!(op, Ops::Add | Ops::Sub | Ops::Mul | Ops::Shl)
                {
                    format!(
                        "((int)((unsigned int){} {} (unsigned int){}))", at(0), symbol,
                        at(1)
                    )
                } else {
                    format!("({} {} {})", at(0), symbol, at(1))
                }
            }
        };
        let name = format!("v{}", u.id);
        source.push_str(&format!("{} {}={};\n", ctype(u.dtype()), name, expression));
        values.insert(u.id, name);
    }
    let result = &values[&root.id];
    source
        .push_str(
            &format!(
                "args[1]={};}}\n", if root.dtype() == DType::F32 {
                format!("bits({result})") } else {
                format!("(unsigned long long){result}") }
            ),
        );
    let program = unsafe { crate::cpu::CpuProgram::compile(&source, "chibi_scalar") }?;
    Ok(CompiledScalar {
        source,
        program,
        types,
        output: root.dtype(),
    })
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
    Call,
    Sink,
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
    Id(usize),
    DType(DType),
    Text(String),
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
                _ => DType::Void,
            },
            Ops::Index => self.src.first().map_or(DType::Void, |x| x.dtype()),
            Ops::Lt | Ops::Eq => DType::Bool,
            Ops::Cast | Ops::Bitcast => {
                if let Arg::DType(dt) = self.arg {
                    dt
                } else {
                    DType::Void
                }
            }
            Ops::Call => self
                .src
                .first()
                .and_then(|x| x.src.first())
                .map_or(DType::Void, |x| x.dtype()),
            Ops::Where => self.src.get(1).map_or(DType::Void, |x| x.dtype()),
            Ops::Sink | Ops::Invalid => DType::Void,
            _ => self.src.first().map_or(DType::Void, |x| x.dtype()),
        }
    }
```

`src/uop.rs` の `impl UOp` 内を追記する。

```rust
// file="src/uop.rs" mode="append" within="impl UOp"
    pub fn sink(src: Vec<UOp>) -> Self {
        Self::new(Ops::Sink, src, Arg::None)
    }
```

`tests/functions.rs`を保存する。

```rust
// file="tests/functions.rs" mode="write"
use chibigrad::{
    DType, UOp,
    functions::Function,
    scalar::{self, Value},
};
#[test]
fn nested_calls_and_stack() {
    let x = Function::parameter(0, DType::F32);
    let inner =
        Function::new(&[DType::F32], &[x.mul(&UOp::f32(2.)), x.add(&UOp::f32(3.))]).unwrap();
    let nested = inner.call(&[x.add(&UOp::f32(1.))]).unwrap();
    let chosen = inner.select(&nested, &UOp::int(1)).unwrap();
    let outer = Function::new(&[DType::F32], &[chosen]).unwrap();
    let value = outer.call(&[UOp::f32(4.)]).unwrap();
    assert_eq!(scalar::evaluate(&value, &[]).unwrap(), Value::f32(8.));
    assert_eq!(
        scalar::compile(&value).unwrap().run(&[]).unwrap(),
        Value::f32(8.)
    );
    assert!(outer.call(&[UOp::int(4)]).is_err());
    assert!(inner.select(&nested, &UOp::int(2)).is_err());
}
```

```sh
# check step=04.01
cargo test --offline
```

入れ子の関数が 8 を返し、引数の型違いと範囲外の出力選択を拒否する。

## 04.02 UIR の往復と検証

DAG を後方参照だけの連番へ変換し、型付き引数を文字列化する。読み込み時に番号・参照・dtype・関数の束縛を検証し、読めただけの不正グラフを受け入れない。

`src/lib.rs`を追記する。

```rust
// file="src/lib.rs" mode="append"

pub mod uir;
```

`src/uir.rs`を保存する。

```rust
// file="src/uir.rs" mode="write"
//! Canonical UIR v2. Each line is `id=Op DType [source_ids] typed_arg`.
//! IDs are dense and backward references only. Strings and bytes use hexadecimal
//! UTF-8/bytes, floats use their exact hexadecimal IEEE bits, and lists use commas.
use crate::{Arg, DType, Ops, Result, UOp};
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
    [DType::F32, DType::I32, DType::I64, DType::Bool, DType::Void]
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
        Ops::Index,
        Ops::Call,
        Ops::Sink,
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
        Arg::Id(v) => format!("id;{v}"),
        Arg::DType(dt) => format!("dtype;{dt:?}"),
        Arg::Text(s) => format!("text;{}", hex(s.as_bytes())),
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
        ["id", v] => Arg::Id(number(v)?),
        ["dtype", dt] => Arg::DType(dtype(dt)?),
        ["text", v] => Arg::Text(string(v)?),
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
        (Ops::Param, _) => return Err("invalid leaf metadata".into()),
        (Ops::Invalid, Arg::Text(_)) if u.src.is_empty() => {}
        (Ops::Invalid, _) => return Err("invalid source payload".into()),
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

`tests/uir.rs`を保存する。

```rust
// file="tests/uir.rs" mode="write"
use chibigrad::{
    DType, UOp,
    functions::Function,
    scalar::{self, Value},
};
#[test]
fn canonical_roundtrip_rejects_corruption() {
    let x = Function::parameter(0, DType::F32);
    let shared = x.add(&UOp::f32(-0.));
    let f = Function::new(&[DType::F32], &[shared.mul(&shared)]).unwrap();
    let y = f.call(&[UOp::f32(4.)]).unwrap();
    let text = chibigrad::uir::print(&y);
    let parsed = chibigrad::uir::parse(&text).unwrap();
    chibigrad::uir::assert_uop(&y, &parsed).unwrap();
    assert_eq!(scalar::evaluate(&parsed, &[]).unwrap(), Value::f32(16.));
    for bad in [
        text.replace("root=", "oops="),
        text.replace("float;8000000000000000", "float;gggggggggggggggg"),
        text.replace("0=Param", "7=Param"),
        text.replace("[0,1]", "[999,1]"),
        text.replace("F32 [] scalar;0;F32", "I32 [] scalar;0;F32"),
        format!("{text}extra\n"),
    ] {
        assert!(chibigrad::uir::parse(&bad).is_err(), "{bad}");
    }
}
```

`tests/uir.rs`を追記する。

```rust
// file="tests/uir.rs" mode="append"
#[test]
fn float_arguments_roundtrip_all_64_bits() {
    use chibigrad::{Arg, Ops, uop::ConstFloat};
    for value in [0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, 1.1, f64::MAX, f64::from_bits(1), f64::NAN] {
        let constant = ConstFloat::new(value);
        let root = UOp::new(Ops::Const, vec![], Arg::Float(constant));
        let text = chibigrad::uir::print(&root);
        assert!(text.starts_with("UIR 2\n"));
        assert!(text.contains(&format!("float;{:016x}", constant.value().to_bits())));
        let parsed = chibigrad::uir::parse(&text).unwrap();
        assert_eq!(parsed, root);
        assert_eq!(chibigrad::uir::print(&parsed), text);
        match parsed.arg {
            Arg::Float(v) => assert_eq!(v.value().to_bits(), constant.value().to_bits()),
            _ => panic!("float argument lost"),
        }
    }
    for bad in ["float;00000000", "float;00000000000000000", "float;+000000000000000", "float;gggggggggggggggg", "f32;00000000"] {
        let text = format!("UIR 2\n0=Const F32 [] {bad}\nroot=0\n");
        assert!(chibigrad::uir::parse(&text).is_err(), "{text}");
    }
    assert!(chibigrad::uir::parse("UIR 1\n0=Const F32 [] f32;00000000\nroot=0\n").is_err());
    let nan = "UIR 2\n0=Const F32 [] float;fff8000000000042\nroot=0\n";
    assert!(chibigrad::uir::print(&chibigrad::uir::parse(nan).unwrap()).contains("float;7ff8000000000000"));
}
```

```sh
# check step=04.02
cargo test --offline
```

共有を保った往復、正しい関数結果、壊れた浮動小数点ビット列や前方参照の拒否を確認する。

## 章末の累積確認

前の章までのテストとexampleも再ビルドします。

```sh
# check
cargo test --quiet --all-targets
```

## 要件との対応

### 04-001 `FunctionId` と引数署名を、実際に呼ぶ関数一つに対して追加する。

<!-- todo:04-001 -->

関連する実装と確認は本章の `04.01`, `04.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 04-002 PARAM の intern key に所属関数を含める。

<!-- todo:04-002 -->

関連する実装と確認は本章の `04.01`, `04.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 04-003 interpreter が呼出しごとに引数環境を作るようにする。

<!-- todo:04-003 -->

関連する実装と確認は本章の `04.01`, `04.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 04-004 C 生成前に body を capture なしで inline する。

<!-- todo:04-004 -->

関連する実装と確認は本章の `04.01`, `04.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 04-005 `f(2,3)=10` と `f(5,1)=12` を CPU まで通す。

<!-- todo:04-005 -->

関連する実装と確認は本章の `04.01`, `04.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 04-006 同型 scalar の STACK を interpreter と型導出へ加える。

<!-- todo:04-006 -->

関連する実装と確認は本章の `04.01`, `04.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 04-007 静的 INDEX の型と `0 <= index < len` を検証する。

<!-- todo:04-007 -->

関連する実装と確認は本章の `04.01`, `04.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 04-008 CALL の結果に固定長 STACK を許し、各要素を inline する。

<!-- todo:04-008 -->

関連する実装と確認は本章の `04.01`, `04.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 04-009 `pair(2,3)` から `5` と `-1` を取り出して C まで通す。

<!-- todo:04-009 -->

関連する実装と確認は本章の `04.01`, `04.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 04-010 対応演算だけの printer と parser を追加する。

<!-- todo:04-010 -->

関連する実装と確認は本章の `04.01`, `04.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 04-011 入力番号、関数名、dtype を検証してから arena へ入れる。

<!-- todo:04-011 -->

関連する実装と確認は本章の `04.01`, `04.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 04-012 f32 の特殊値もビット列で往復するようにする。

<!-- todo:04-012 -->

関連する実装と確認は本章の `04.01`, `04.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 04-013 `pair` の UIR を出力し、読み戻したグラフから C を生成する。

<!-- todo:04-013 -->

関連する実装と確認は本章の `04.01`, `04.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 04-014 読み戻し前後で `5` と `-1` が一致することを assertion で確かめる。

<!-- todo:04-014 -->

関連する実装と確認は本章の `04.01`, `04.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 04-015 未束縛 PARAM と他関数の PARAM の直接参照を拒否する。

<!-- todo:04-015 -->

関連する実装と確認は本章の `04.01`, `04.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 04-016 CALL の引数不足、dtype 不一致、再帰を拒否する。

<!-- todo:04-016 -->

関連する実装と確認は本章の `04.01`, `04.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 04-017 INDEX の負数と長さ以上の値を拒否する。

<!-- todo:04-017 -->

関連する実装と確認は本章の `04.01`, `04.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 04-018 `assert_uop` が意図的に誤った予定値で失敗することを確認する。

<!-- todo:04-018 -->

関連する実装と確認は本章の `04.01`, `04.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 04-019 二つの呼出しで値を混同せず、`10` と `12` を返す。

<!-- todo:04-019 -->

関連する実装と確認は本章の `04.01`, `04.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 04-020 STACK の選択結果が `5` と `-1` になる。

<!-- todo:04-020 -->

関連する実装と確認は本章の `04.01`, `04.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 04-021 UIR の出力、入力、実行が一往復する。

<!-- todo:04-021 -->

関連する実装と確認は本章の `04.01`, `04.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 04-022 異なる関数の PARAM が同じ NodeId に共有されない。

<!-- todo:04-022 -->

関連する実装と確認は本章の `04.01`, `04.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 04-023 誤った予定値と不正な UIR の assertion が実際に失敗する。

<!-- todo:04-023 -->

関連する実装と確認は本章の `04.01`, `04.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 04-024 scalar の前章までの例も同じ pipeline で成功する。

<!-- todo:04-024 -->

関連する実装と確認は本章の `04.01`, `04.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。
