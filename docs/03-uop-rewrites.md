# 03. UOpの共有・型・scalar実行・パターン書換えを追加する

前の節で動かしたコードに、今回使う演算と処理を追加します。各節の確認を実行してから次へ進みます。

参照するUOpの構成は [tinygrad c22edff](https://github.com/tinygrad/tinygrad/blob/c22edff14c79310a9b2df62a78819cf77c8289fa/tinygrad/uop/ops.py) です。最初から全機能を実装せず、使う節で増やします。

## 03.01 共有する不変ノードと型付きスカラー

Rc と Weak のインターン表で同じ op・src・arg を一つにする。F32 のビット表現、I32 の巡回演算、I64 の検査、Bool、Cast を加える。入力の型と個数を境界で確かめ、Rust の参照評価と C の実行を照合する。前章の CLI はこの型付き実行器に移し、7、20、二回の入力差替えを保つ。

`Cargo.toml`を保存する。

```toml
# file="Cargo.toml" mode="write"
[package]
name="chibigrad"
version="0.1.0"
edition="2024"
```

`examples/scalar_arithmetic.rs`を保存する。

```rust
// file="examples/scalar_arithmetic.rs" mode="write"
use chibigrad::{UOp, scalar};
fn main() -> chibigrad::Result<()> {
    let root = UOp::f32(2.).add(&UOp::f32(3.)).mul(&UOp::f32(4.));
    let plan = scalar::compile(&root)?;
    for flag in std::env::args().skip(1) {
        match flag.as_str() {
            "--dump-source" => println!("{}", plan.source),
            "--dump-graph" => {
                for n in root.toposort() {
                    println!("{:?} {:?} inputs={}", n.op, n.arg, n.src.len());
                }
            }
            _ => return Err(format!("unknown option: {flag}")),
        }
    }
    println!("{}", plan.run(&[])?.number());
    Ok(())
}
```

`examples/scalar_const.rs`を保存する。

```rust
// file="examples/scalar_const.rs" mode="write"
use chibigrad::{UOp, scalar};
fn main() -> chibigrad::Result<()> {
    let root = UOp::f32(7.);
    let plan = scalar::compile(&root)?;
    for flag in std::env::args().skip(1) {
        match flag.as_str() {
            "--dump-source" => println!("{}", plan.source),
            "--dump-graph" => {
                for n in root.toposort() {
                    println!("{:?} {:?} inputs={}", n.op, n.arg, n.src.len());
                }
            }
            _ => return Err(format!("unknown option: {flag}")),
        }
    }
    println!("{}", plan.run(&[])?.number());
    Ok(())
}
```

`examples/scalar_inputs.rs` の `use chibigrad::`を置き換える。

```rust
// file="examples/scalar_inputs.rs" mode="replace" item="use chibigrad::"
use chibigrad::{
    Arg, DType, Ops, UOp,
    scalar::{self, Value},
};
```

`examples/scalar_inputs.rs` の `fn main`を置き換える。

```rust
// file="examples/scalar_inputs.rs" mode="replace" item="fn main"
fn main() -> chibigrad::Result<()> {
    let mut stats = false;
    for flag in std::env::args().skip(1) {
        match flag.as_str() {
            "--stats" => stats = true,
            _ => return Err(format!("unknown option: {flag}")),
        }
    }
    let p = |index| {
        UOp::new(
            Ops::Param,
            vec![],
            Arg::ScalarParam {
                index,
                dtype: DType::F32,
            },
        )
    };
    let root = p(0).add(&p(1)).mul(&p(2));
    let mut compiles = 0;
    let mut runs = 0;
    let plan = scalar::compile(&root)?;
    compiles += 1;
    for row in [[2., 3., 4.], [1., 2., 4.]] {
        let inputs = row.map(Value::f32);
        println!("{}", plan.run(&inputs)?.number());
        runs += 1;
    }
    if stats {
        println!("compiles={compiles} runs={runs}");
    }
    Ok(())
}
```

`src/codegen.rs`を削除する。

```rust
// file="src/codegen.rs" mode="delete"

```

`src/cpu.rs` の `use std::`を置き換える。

```rust
// file="src/cpu.rs" mode="replace" item="use std::"
use std::{
    ffi::{CStr, CString, c_char, c_int, c_void},
    fs,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};
```

`src/cpu.rs` の `impl CpuProgram` 内 の `pub unsafe fn compile`を置き換える。

```rust
// file="src/cpu.rs" mode="replace" within="impl CpuProgram" item="pub unsafe fn compile"
    pub unsafe fn compile(source: &str, entry: &str) -> Result<Self> {
        let symbol = CString::new(entry).map_err(|e| e.to_string())?;
        let dir = Scratch::new()?;
        let input = dir.0.join("scalar.c");
        let library = dir.0.join("scalar.so");
        fs::write(&input, source).map_err(|e| e.to_string())?;
        let output = Command::new(std::env::var("CC").unwrap_or_else(|_| {
            if cfg!(target_os = "macos") {
                "/usr/bin/clang".into()
            } else {
                "clang".into()
            }
        }))
        .args(["-std=c11", "-O2", "-shared", "-fPIC"])
        .arg(&input)
        .arg("-o")
        .arg(&library)
        .output()
        .map_err(|e| e.to_string())?;
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).into());
        }
        let object = fs::read(&library).map_err(|e| e.to_string())?;
        let path = CString::new(library.to_str().ok_or("non-UTF-8 library path")?)
            .map_err(|e| e.to_string())?;
        let handle = unsafe { dlopen(path.as_ptr(), 2) };
        if handle.is_null() {
            return Err(loader_error());
        }
        unsafe {
            dlerror();
        }
        let address = unsafe { dlsym(handle, symbol.as_ptr()) };
        if address.is_null() {
            let message = loader_error();
            unsafe {
                dlclose(handle);
            }
            return Err(message);
        }
        let entry =
            unsafe { std::mem::transmute::<*mut c_void, unsafe extern "C" fn(*mut u8)>(address) };
        Ok(Self {
            handle,
            entry,
            object,
            source: source.into(),
        })
    }
```

`src/cpu.rs` の `impl CpuProgram` 内 の `pub unsafe fn run`を置き換える。

```rust
// file="src/cpu.rs" mode="replace" within="impl CpuProgram" item="pub unsafe fn run"
    pub unsafe fn run(&self, args: &mut [u8]) -> Result<()> {
        unsafe {
            (self.entry)(args.as_mut_ptr());
        }
        Ok(())
    }
```

`src/cpu.rs` の `impl CpuProgram` 内 の `pub fn object`を置き換える。

```rust
// file="src/cpu.rs" mode="replace" within="impl CpuProgram" item="pub fn object"
    pub fn object(&self) -> &[u8] {
        &self.object
    }
```

`src/cpu.rs` の `impl CpuProgram` 内 の `pub fn source`を置き換える。

```rust
// file="src/cpu.rs" mode="replace" within="impl CpuProgram" item="pub fn source"
    pub fn source(&self) -> Option<&str> {
        Some(&self.source)
    }
```

`src/cpu.rs` の `impl Drop for CpuProgram` 内 の `fn drop`を置き換える。

```rust
// file="src/cpu.rs" mode="replace" within="impl Drop for CpuProgram" item="fn drop"
    fn drop(&mut self) {
        unsafe {
            dlclose(self.handle);
        }
    }
```

`src/cpu.rs` の `fn loader_error`を置き換える。

```rust
// file="src/cpu.rs" mode="replace" item="fn loader_error"
fn loader_error() -> String {
    let error = unsafe { dlerror() };
    if error.is_null() {
        "dynamic loader failed".into()
    } else {
        unsafe { CStr::from_ptr(error) }.to_string_lossy().into()
    }
}
```

`src/cpu.rs` の `impl Scratch` 内 の `fn new`を置き換える。

```rust
// file="src/cpu.rs" mode="replace" within="impl Scratch" item="fn new"
    fn new() -> Result<Self> {
        let path = std::env::temp_dir().join(format!(
            "chibigrad-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).map_err(|e| e.to_string())?;
        Ok(Self(path))
    }
```

`src/cpu.rs` の `impl Drop for Scratch` 内 の `fn drop`を置き換える。

```rust
// file="src/cpu.rs" mode="replace" within="impl Drop for Scratch" item="fn drop"
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
```

`src/lib.rs`を保存する。

```rust
// file="src/lib.rs" mode="write"
pub type Result<T> = std::result::Result<T, String>;
pub use uop::{Arg, DType, Ops, UOp};
pub mod scalar;
pub mod uop;

pub mod cpu;
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
    let root = root.clone();
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
fn ctype(dt: DType) -> &'static str {
    match dt {
        DType::F32 => "float",
        DType::I32 => "int",
        DType::I64 => "long long",
        DType::Bool => "int",
        _ => "void",
    }
}
pub fn compile(root: &UOp) -> Result<CompiledScalar> {
    let root = root.clone();
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
            (Ops::Const, Arg::F32(n)) => format!("as_float({n}u)"),
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
impl CompiledScalar {
    pub fn run(&self, inputs: &[Value]) -> Result<Value> {
        inputs_valid(&self.types, inputs)?;
        let mut status = 0i32;
        let mut words = vec![(& mut status as * mut i32) as u64, 0];
        words.extend(inputs.iter().map(|v| v.pack()));
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(words.as_mut_ptr().cast(), words.len() * 8)
        };
        unsafe { self.program.run(bytes) }?;
        if status != 0 {
            return Err("unsafe native scalar operation".into());
        }
        Value::unpack(self.output, words[1])
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

`src/uop.rs`を保存する。

```rust
// file="src/uop.rs" mode="write"
use std::{
    cell::{OnceCell, RefCell},
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
    ops::Deref,
    rc::{Rc, Weak},
};
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    I32,
    I64,
    Bool,
    Void,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Arg {
    None,
    F32(u32),
    I32(i32),
    Int(i64),
    Bool(bool),
    DType(DType),
    Text(String),
    ScalarParam { index: usize, dtype: DType },
}
#[derive(Debug)]
pub struct Node {
    pub id: u64,
    pub op: Ops,
    pub src: Vec<UOp>,
    pub arg: Arg,
    dtype_cache: OnceCell<DType>,
}
#[derive(Clone, Debug)]
pub struct UOp(Rc<Node>);
#[derive(Clone, PartialEq, Eq, Hash)]
struct Key {
    op: Ops,
    src: Vec<u64>,
    arg: Arg,
}
#[derive(Default)]
struct Interner {
    next: u64,
    nodes: HashMap<Key, Weak<Node>>,
}
thread_local! {
    static INTERN : RefCell < Interner > = RefCell::new(Interner::default());
}
impl Node {
    fn evict(&self) {
        let key = Key {
            op: self.op,
            src: self.src.iter().map(|s| s.id).collect(),
            arg: self.arg.clone(),
        };
        let _ = INTERN.try_with(|cell| {
            if let Ok(mut intern) = cell.try_borrow_mut() {
                if intern
                    .nodes
                    .get(&key)
                    .is_some_and(|weak| std::ptr::eq(weak.as_ptr(), self))
                {
                    intern.nodes.remove(&key);
                }
            }
        });
    }
}
impl Drop for Node {
    fn drop(&mut self) {
        self.evict();
        let mut pending = std::mem::take(&mut self.src);
        while let Some(child) = pending.pop() {
            if Rc::strong_count(&child.0) == 1 {
                child.evict();
                if let Ok(mut node) = Rc::try_unwrap(child.0) {
                    pending.append(&mut node.src);
                }
            }
        }
    }
}
impl Deref for UOp {
    type Target = Node;
    fn deref(&self) -> &Node {
        &self.0
    }
}
impl PartialEq for UOp {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for UOp {}
impl Hash for UOp {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}
impl UOp {
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
                dtype_cache: OnceCell::new(),
            });
            intern.nodes.insert(key, Rc::downgrade(&node));
            if intern.nodes.len() % 1024 == 0 {
                intern.nodes.retain(|_, v| v.strong_count() != 0);
            }
            Self(node)
        })
    }
    pub fn weak(&self) -> Weak<Node> {
        Rc::downgrade(&self.0)
    }
    pub fn prune_interner() -> usize {
        INTERN.with(|c| {
            let mut c = c.borrow_mut();
            c.nodes.retain(|_, v| v.strong_count() != 0);
            c.nodes.len()
        })
    }
    pub fn interner_len() -> usize {
        INTERN.with(|cell| cell.borrow().nodes.len())
    }
    pub fn replace(&self, src: Vec<UOp>) -> Self {
        Self::new(self.op, src, self.arg.clone())
    }
    pub fn f32(x: f32) -> Self {
        Self::new(Ops::Const, vec![], Arg::F32(x.to_bits()))
    }
    pub fn i32(x: i32) -> Self {
        Self::new(Ops::Const, vec![], Arg::I32(x))
    }
    pub fn int(x: i64) -> Self {
        Self::new(Ops::Const, vec![], Arg::Int(x))
    }
    pub fn bool(x: bool) -> Self {
        Self::new(Ops::Const, vec![], Arg::Bool(x))
    }
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(Ops::Invalid, vec![], Arg::Text(message.into()))
    }
    pub fn alu(&self, op: Ops, others: &[UOp]) -> Self {
        let mut src = vec![self.clone()];
        src.extend_from_slice(others);
        Self::new(op, src, Arg::None)
    }
    pub fn add(&self, x: &Self) -> Self {
        self.alu(Ops::Add, &[x.clone()])
    }
    pub fn mul(&self, x: &Self) -> Self {
        self.alu(Ops::Mul, &[x.clone()])
    }
    pub fn sub(&self, x: &Self) -> Self {
        self.alu(Ops::Sub, &[x.clone()])
    }
    pub fn div(&self, x: &Self) -> Self {
        self.alu(Ops::Div, &[x.clone()])
    }
    pub fn modulo(&self, x: &Self) -> Self {
        self.alu(Ops::Mod, &[x.clone()])
    }
    pub fn lt(&self, x: &Self) -> Self {
        self.alu(Ops::Lt, &[x.clone()])
    }
    pub fn eq(&self, x: &Self) -> Self {
        self.alu(Ops::Eq, &[x.clone()])
    }
    pub fn and(&self, x: &Self) -> Self {
        self.alu(Ops::And, &[x.clone()])
    }
    pub fn bitcast(&self, dt: DType) -> Self {
        Self::new(Ops::Bitcast, vec![self.clone()], Arg::DType(dt))
    }
    pub fn cast(&self, dt: DType) -> Self {
        Self::new(Ops::Cast, vec![self.clone()], Arg::DType(dt))
    }
    pub fn where_(&self, a: &Self, b: &Self) -> Self {
        Self::new(
            Ops::Where,
            vec![self.clone(), a.clone(), b.clone()],
            Arg::None,
        )
    }
    pub fn dtype(&self) -> DType {
        *self.dtype_cache.get_or_init(|| self.infer_dtype())
    }
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
                _ => DType::Void,
            },
            Ops::Lt | Ops::Eq => DType::Bool,
            Ops::Cast | Ops::Bitcast => {
                if let Arg::DType(dt) = self.arg {
                    dt
                } else {
                    DType::Void
                }
            }
            Ops::Where => self.src.get(1).map_or(DType::Void, |x| x.dtype()),
            Ops::Invalid => DType::Void,
            _ => self.src.first().map_or(DType::Void, |x| x.dtype()),
        }
    }
    pub fn toposort(&self) -> Vec<UOp> {
        let mut seen = HashSet::new();
        let mut out = vec![];
        let mut stack = vec![(self.clone(), false)];
        while let Some((u, done)) = stack.pop() {
            if done {
                out.push(u);
                continue;
            }
            if !seen.insert(u.id) {
                continue;
            }
            stack.push((u.clone(), true));
            let sources = u.ordered_sources();
            for s in sources.into_iter().rev() {
                stack.push((s.clone(), false));
            }
        }
        out
    }
    pub fn ordered_sources(&self) -> Vec<&UOp> {
        self.src.iter().collect()
    }
}
pub fn is_binary(op: Ops) -> bool {
    matches!(
        op,
        Ops::Add
            | Ops::Sub
            | Ops::Mul
            | Ops::Div
            | Ops::Mod
            | Ops::Max
            | Ops::Lt
            | Ops::Eq
            | Ops::And
            | Ops::Or
            | Ops::Xor
            | Ops::Shl
            | Ops::Shr
    )
}
pub fn is_unary(op: Ops) -> bool {
    matches!(op, Ops::Neg | Ops::Cast | Ops::Bitcast)
}
```

`tests/scalar.rs`を保存する。

```rust
// file="tests/scalar.rs" mode="write"
use chibigrad::{
    Arg, DType, Ops, UOp,
    scalar::{self, Value},
};
fn param(index: usize, dtype: DType) -> UOp {
    UOp::new(Ops::Param, vec![], Arg::ScalarParam { index, dtype })
}
#[test]
fn sharing_and_values() {
    assert_eq!(UOp::f32(7.), UOp::f32(7.));
    assert_ne!(UOp::f32(0.), UOp::f32(-0.));
    assert_ne!(UOp::int(1), UOp::bool(true));
    let weak = {
        let u = param(0, DType::I64).add(&UOp::int(2));
        assert_eq!(u.dtype(), DType::I64);
        u.weak()
    };
    assert!(weak.upgrade().is_none());
}
#[test]
fn scalar_constants_and_rebinding() {
    let y = param(0, DType::F32)
        .add(&param(1, DType::F32))
        .mul(&param(2, DType::F32));
    let p = scalar::compile(&y).unwrap();
    for (a, b, c, v) in [(2., 3., 4., 20.), (1., 2., 4., 12.)] {
        let x = [Value::f32(a), Value::f32(b), Value::f32(c)];
        assert_eq!(p.run(&x).unwrap(), Value::f32(v));
        assert_eq!(scalar::evaluate(&y, &x).unwrap(), Value::f32(v));
    }
    for v in [7., 5., 20.] {
        assert_eq!(
            scalar::compile(&UOp::f32(v)).unwrap().run(&[]).unwrap(),
            Value::f32(v)
        );
    }
}

#[test]
fn native_matches_oracle_200_cases() -> chibigrad::Result<()> {
    let a = param(0, DType::I32);
    let b = param(1, DType::I32);
    let expression = a
        .mul(&b)
        .add(&a)
        .alu(Ops::Xor, &[a.alu(Ops::Shr, &[b.and(&UOp::i32(31))])]);
    let compiled = scalar::compile(&expression)?;
    let mut seed = 0x51c0ffeeu32;
    for _ in 0..200 {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let x = seed as i32;
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let values = [Value::I32(x), Value::I32(seed as i32)];
        assert_eq!(
            compiled.run(&values)?,
            scalar::evaluate(&expression, &values)?
        );
    }
    for op in [
        Ops::Div,
        Ops::Mod,
        Ops::Shl,
        Ops::Shr,
        Ops::And,
        Ops::Or,
        Ops::Xor,
    ] {
        let y = a.alu(op, &[b.clone()]);
        let plan = scalar::compile(&y)?;
        for (x, z) in [
            (-7, 3),
            (7, -3),
            (i32::MIN, -1),
            (1, 0),
            (-8, 1),
            (1, 32),
            (i32::MAX, 31),
        ] {
            let values = [Value::I32(x), Value::I32(z)];
            let expected = scalar::evaluate(&y, &values);
            let actual = plan.run(&values);
            match (expected, actual) {
                (Ok(a), Ok(b)) => assert_eq!(a, b),
                (Err(_), Err(_)) => {}
                other => panic!("{op:?} {values:?} {other:?}"),
            }
        }
    }
    let x = param(0, DType::F32);
    for dt in [DType::I32, DType::I64] {
        let y = x.cast(dt);
        let p = scalar::compile(&y)?;
        for v in [
            0.,
            -0.,
            2147483520.,
            2147483648.,
            -2147483648.,
            f32::NAN,
            f32::INFINITY,
            9223372036854775808.,
        ] {
            let values = [Value::f32(v)];
            let a = scalar::evaluate(&y, &values);
            let b = p.run(&values);
            match (a, b) {
                (Ok(a), Ok(b)) => assert_eq!(a, b),
                (Err(_), Err(_)) => {}
                v => panic!("{v:?}"),
            }
        }
    }
    let a = param(0, DType::I64);
    let b = param(1, DType::I64);
    for op in [Ops::Add, Ops::Sub, Ops::Mul, Ops::Div, Ops::Mod, Ops::Shr] {
        let y = a.alu(op, &[b.clone()]);
        let p = scalar::compile(&y)?;
        for (a, b) in [
            (i64::MAX, 1),
            (i64::MIN, -1),
            (-7, 3),
            (9007199254740993, 2),
        ] {
            let values = [Value::I64(a), Value::I64(b)];
            match (scalar::evaluate(&y, &values), p.run(&values)) {
                (Ok(a), Ok(b)) => assert_eq!(a, b),
                (Err(_), Err(_)) => {}
                v => panic!("{op:?} {v:?}"),
            }
        }
    }
    Ok(())
}
```

```sh
# check step=03.01
cargo test --offline
cargo run --offline --example scalar_inputs -- --stats
```

共有、200 個の入力でのネイティブ一致、整数境界と不正入力の検出が通る。

## 03.02 パターンで共有グラフを書き換える

UPat が演算・型・引数・入力を照合し、名前付きキャプチャを置換関数へ渡す。共有された入力をメモ化し、書換え順序と回数制限で循環を検出する。型やスカラーの実行器を変更せず規則を追加できる。

`src/lib.rs`を追記する。

```rust
// file="src/lib.rs" mode="append"

pub mod pattern;
```

`src/pattern.rs`を保存する。

```rust
// file="src/pattern.rs" mode="write"
use crate::{
    Result,
    uop::{Arg, DType, Ops, UOp},
};
use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
};
pub type Captures = HashMap<String, UOp>;
#[derive(Clone, Debug, Default)]
pub struct UPat {
    pub ops: Option<Vec<Ops>>,
    pub dtype: Option<DType>,
    pub arg: Option<Arg>,
    pub name: Option<String>,
    pub src: Option<Vec<UPat>>,
    pub commutative: bool,
    pub allow_extra: bool,
    pub repeat: Option<Box<UPat>>,
}
impl UPat {
    pub fn any() -> Self {
        Self::default()
    }
    pub fn op(op: Ops) -> Self {
        Self {
            ops: Some(vec![op]),
            ..Self::default()
        }
    }
    pub fn ops(ops: &[Ops]) -> Self {
        Self {
            ops: Some(ops.to_vec()),
            ..Self::default()
        }
    }
    pub fn var(name: &str) -> Self {
        Self::any().named(name)
    }
    pub fn constant(arg: Arg) -> Self {
        Self::op(Ops::Const).arg(arg)
    }
    pub fn named(mut self, name: &str) -> Self {
        self.name = Some(name.into());
        self
    }
    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }
    pub fn arg(mut self, arg: Arg) -> Self {
        self.arg = Some(arg);
        self
    }
    pub fn src(mut self, src: Vec<UPat>) -> Self {
        self.src = Some(src);
        self
    }
    pub fn commutative(mut self) -> Self {
        self.commutative = true;
        self
    }
    pub fn allow_extra(mut self) -> Self {
        self.allow_extra = true;
        self
    }
    pub fn repeat_src(mut self, pattern: UPat) -> Self {
        self.repeat = Some(Box::new(pattern));
        self.allow_extra = true;
        if self.src.is_none() {
            self.src = Some(vec![]);
        }
        self
    }
    pub fn matches(&self, u: &UOp) -> Option<Captures> {
        self.match_with(u, Captures::new()).into_iter().next()
    }
    fn match_with(&self, u: &UOp, mut captures: Captures) -> Vec<Captures> {
        if self.ops.as_ref().is_some_and(|ops| !ops.contains(&u.op))
            || self.dtype.is_some_and(|dt| dt != u.dtype())
            || self.arg.as_ref().is_some_and(|a| a != &u.arg)
        {
            return vec![];
        }
        if let Some(name) = &self.name {
            if captures.get(name).is_some_and(|v| v != u) {
                return vec![];
            }
            captures.insert(name.clone(), u.clone());
        }
        let Some(prefix) = &self.src else {
            return vec![captures];
        };
        let mut patterns = prefix.clone();
        if let Some(repeated) = &self.repeat {
            if u.src.len() < patterns.len() {
                return vec![];
            }
            while patterns.len() < u.src.len() {
                patterns.push((**repeated).clone());
            }
        }
        let patterns = &patterns;
        if u.src.len() < patterns.len() || (!self.allow_extra && u.src.len() != patterns.len()) {
            return vec![];
        }
        fn visit(
            patterns: &[UPat],
            nodes: &[UOp],
            depth: usize,
            used: &mut [bool],
            cap: Captures,
            commutative: bool,
            out: &mut Vec<Captures>,
        ) {
            if depth == patterns.len() {
                out.push(cap);
                return;
            }
            for i in 0..nodes.len() {
                if used[i] || (!commutative && i != depth) {
                    continue;
                }
                used[i] = true;
                for next in patterns[depth].match_with(&nodes[i], cap.clone()) {
                    visit(patterns, nodes, depth + 1, used, next, commutative, out)
                }
                used[i] = false;
            }
        }
        let mut out = vec![];
        visit(
            patterns,
            &u.src,
            0,
            &mut vec![false; u.src.len()],
            captures,
            self.commutative,
            &mut out,
        );
        out
    }
}
type Rewrite = Rc<dyn Fn(&Captures) -> Option<UOp>>;
#[derive(Default, Clone)]
pub struct PatternMatcher {
    rules: Vec<(UPat, Rewrite)>,
    by_op: HashMap<Ops, Vec<usize>>,
    wildcard: Vec<usize>,
}
impl PatternMatcher {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add(mut self, pattern: UPat, rule: impl Fn(&Captures) -> Option<UOp> + 'static) -> Self {
        let index = self.rules.len();
        if let Some(ops) = &pattern.ops {
            for op in ops {
                self.by_op.entry(*op).or_default().push(index)
            }
        } else {
            self.wildcard.push(index)
        }
        self.rules.push((pattern, Rc::new(rule)));
        self
    }
    pub fn rewrite(&self, u: &UOp) -> Option<UOp> {
        let mut indices = self.by_op.get(&u.op).cloned().unwrap_or_default();
        indices.extend_from_slice(&self.wildcard);
        indices.sort_unstable();
        for i in indices {
            let (p, f) = &self.rules[i];
            for captures in p.match_with(u, Captures::new()) {
                if let Some(v) = f(&captures) {
                    if &v != u {
                        return Some(v);
                    }
                }
            }
        }
        None
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RewriteOrder {
    TopDown,
    BottomUp,
}
#[derive(Clone, Copy, Debug)]
pub struct RewriteOptions {
    pub order: RewriteOrder,
    pub walk: bool,
    pub max_steps: usize,
}
impl Default for RewriteOptions {
    fn default() -> Self {
        Self {
            order: RewriteOrder::BottomUp,
            walk: true,
            max_steps: 100_000,
        }
    }
}
pub fn graph_rewrite(root: &UOp, matcher: &PatternMatcher) -> Result<UOp> {
    graph_rewrite_with_options(root, matcher, RewriteOptions::default())
}
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
    let skip_body = |_u: &UOp, _i: usize| false;
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

`tests/pattern.rs`を保存する。

```rust
// file="tests/pattern.rs" mode="write"
use chibigrad::{
    Arg, DType, Ops, UOp,
    pattern::{PatternMatcher, UPat, graph_rewrite},
};
#[test]
fn captures_rewrite_sharing_and_weak_lifetime() {
    let x = UOp::new(
        Ops::Param,
        vec![],
        Arg::ScalarParam {
            index: 0,
            dtype: DType::I64,
        },
    );
    let same = UPat::op(Ops::Add).src(vec![UPat::var("x"), UPat::var("x")]);
    assert!(same.matches(&x.add(&x)).is_some());
    assert!(same.matches(&x.add(&UOp::int(2))).is_none());
    let pm = PatternMatcher::new().add(
        UPat::op(Ops::Add)
            .src(vec![UPat::var("x"), UPat::constant(Arg::Int(0))])
            .commutative(),
        |c| Some(c["x"].clone()),
    );
    assert_eq!(graph_rewrite(&UOp::int(0).add(&x), &pm).unwrap(), x);
    let cycle = PatternMatcher::new()
        .add(UPat::constant(Arg::Int(0)), |_| Some(UOp::int(1)))
        .add(UPat::constant(Arg::Int(1)), |_| Some(UOp::int(0)));
    assert!(graph_rewrite(&UOp::int(0), &cycle).is_err());
}
#[test]
fn deep_graph_rewrite_and_drop_are_iterative() {
    let before = UOp::interner_len();
    let mut u = UOp::int(0);
    for i in 1..20000 {
        u = u.add(&UOp::int(i));
    }
    assert_eq!(graph_rewrite(&u, &PatternMatcher::new()).unwrap(), u);
    drop(u);
    assert_eq!(UOp::interner_len(), before);
}
```

```sh
# check step=03.02
cargo test --offline
```

x + 0 の除去、同一キャプチャの制約、循環拒否、二万段のグラフの書換えと解放を確認する。

## 章末の累積確認

前の章までのテストとexampleも再ビルドします。

```sh
# check
cargo test --quiet --all-targets
```

## 要件との対応

### 03-001 arena に `HashMap<NodeKey, NodeId>` を追加する。

<!-- todo:03-001 -->

関連する実装と確認は本章の `03.01`, `03.02` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 03-002 f32 のビットキーと演算の入力順を保持する。

<!-- todo:03-002 -->

関連する実装と確認は本章の `03.01`, `03.02` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 03-003 同じ `a+b` を二回作ると同じ番号になるようにする。

<!-- todo:03-003 -->

関連する実装と確認は本章の `03.01`, `03.02` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 03-004 `(a+b)*(a+b)` を C まで通し、入力 `2,3` で `25` を得る。

<!-- todo:03-004 -->

関連する実装と確認は本章の `03.01`, `03.02` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 03-005 dtype の導出と各演算の入力数を一か所で検証する。

<!-- todo:03-005 -->

関連する実装と確認は本章の `03.01`, `03.02` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 03-006 bool と i32 の ABI pack と C の復元処理を追加する。

<!-- todo:03-006 -->

関連する実装と確認は本章の `03.01`, `03.02` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 03-007 加減乗算、比較、WHERE、CAST を interpreter と renderer に加える。

<!-- todo:03-007 -->

関連する実装と確認は本章の `03.01`, `03.02` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 03-008 void 定数と混合 dtype を検証時に拒否する。

<!-- todo:03-008 -->

関連する実装と確認は本章の `03.01`, `03.02` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 03-009 `WHERE(a<3,a*2,a+1)` を `a=2` で `4`、`a=5` で `6` にする。

<!-- todo:03-009 -->

関連する実装と確認は本章の `03.01`, `03.02` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 03-010 XOR、AND、OR、SHL、SHR、DIV、MOD を両経路に加える。

<!-- todo:03-010 -->

関連する実装と確認は本章の `03.01`, `03.02` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 03-011 RECIP と MAX を本文の f32 契約で実行する。

<!-- todo:03-011 -->

関連する実装と確認は本章の `03.01`, `03.02` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 03-012 bounds と動的入力の範囲検証を追加する。

<!-- todo:03-012 -->

関連する実装と確認は本章の `03.01`, `03.02` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 03-013 危険な DIV、shift、CAST は安全性を証明できなければ拒否する。

<!-- todo:03-013 -->

関連する実装と確認は本章の `03.01`, `03.02` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 03-014 `((5 XOR 3) SHL 1)=12` と負数の DIV、MOD を C まで通す。

<!-- todo:03-014 -->

関連する実装と確認は本章の `03.01`, `03.02` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 03-015 子を先に再構築する rewrite を `match` で作る。

<!-- todo:03-015 -->

関連する実装と確認は本章の `03.01`, `03.02` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 03-016 検証済み ALU evaluator を定数畳み込みにも使う。

<!-- todo:03-016 -->

関連する実装と確認は本章の `03.01`, `03.02` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 03-017 `i32 x+0 → x`、`x XOR 0 → x`、`WHERE(true,a,b) → a` を加える。

<!-- todo:03-017 -->

関連する実装と確認は本章の `03.01`, `03.02` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 03-018 前後の dtype、bounds、ALU 数を dump に出す。

<!-- todo:03-018 -->

関連する実装と確認は本章の `03.01`, `03.02` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 03-019 三つの実行経路を入力 `-7,0,5` と定数例で比較する。

<!-- todo:03-019 -->

関連する実装と確認は本章の `03.01`, `03.02` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 03-020 床除算、wrap、shift、f32 の特殊値で三経路が合う。

<!-- todo:03-020 -->

関連する実装と確認は本章の `03.01`, `03.02` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 03-021 NaN は `is_nan`、符号付きゼロはビット列で比較する。

<!-- todo:03-021 -->

関連する実装と確認は本章の `03.01`, `03.02` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 03-022 有限の一般 f32 は予定許容値 `atol=1e-6, rtol=1e-6` で比較する。

<!-- todo:03-022 -->

関連する実装と確認は本章の `03.01`, `03.02` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 03-023 dtype 不一致、void 入力、範囲外入力は生成か呼出しの前に失敗する。

<!-- todo:03-023 -->

関連する実装と確認は本章の `03.01`, `03.02` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 03-024 bounds が結果を含み、証明できない式を安全と判定しない。

<!-- todo:03-024 -->

関連する実装と確認は本章の `03.01`, `03.02` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 03-025 前章までの成功例も通る。

<!-- todo:03-025 -->

関連する実装と確認は本章の `03.01`, `03.02` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。
