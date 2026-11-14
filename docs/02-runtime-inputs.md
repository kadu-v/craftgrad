# 02. 実行時入力と共有ライブラリを追加する

前の節で動かしたコードに、今回使う演算と処理を追加します。各節の確認を実行してから次へ進みます。

参照するUOpの構成は [tinygrad c22edff](https://github.com/tinygrad/tinygrad/blob/c22edff14c79310a9b2df62a78819cf77c8289fa/tinygrad/uop/ops.py) です。最初から全機能を実装せず、使う節で増やします。

## 02.01 Param に実行時の値を渡す

定数と別に Param を追加する。arg は入力番号だけを保持する。生成する main でコマンドライン引数を f32 に変換し、p0 に束縛する。式の (p0 + 3) * 4 を変えず、Rust から 2 を渡して 20 を得る。

`examples/scalar_arithmetic.rs` の `fn main`を置き換える。

```rust
// file="examples/scalar_arithmetic.rs" mode="replace" item="fn main"
fn main() -> chibigrad::Result<()> {
    let root = UOp::f32(2.0).add(&UOp::f32(3.0)).mul(&UOp::f32(4.0));
    let source = chibigrad::codegen::source(&root, 0);
    for flag in std::env::args().skip(1) {
        match flag.as_str() {
            "--dump-source" => println!("{source}"),
            "--dump-graph" => for node in root.postorder() {
                println!("{:?} {:?} inputs={}", node.op, node.arg, node.src.len());
            },
            _ => return Err(format!("unknown option: {flag}")),
        }
    }
    let result = chibigrad::codegen::run(&root, &[])?;
    println!("{result}");
    Ok(())
}
```

`examples/scalar_const.rs` の `fn main`を置き換える。

```rust
// file="examples/scalar_const.rs" mode="replace" item="fn main"
fn main() -> chibigrad::Result<()> {
    let root = UOp::f32(7.0);
    let source = chibigrad::codegen::source(&root, 0);
    for flag in std::env::args().skip(1) {
        match flag.as_str() {
            "--dump-source" => println!("{source}"),
            "--dump-graph" => for node in root.postorder() {
                println!("{:?} {:?} inputs={}", node.op, node.arg, node.src.len());
            },
            _ => return Err(format!("unknown option: {flag}")),
        }
    }
    let result = chibigrad::codegen::run(&root, &[])?;
    println!("{result}");
    Ok(())
}
```

`src/codegen.rs` の `pub fn expression`を置き換える。

```rust
// file="src/codegen.rs" mode="replace" item="pub fn expression"
pub fn expression(root: &UOp) -> String {
    let root = root.rewrite(|_| None);
    let mut terms: Vec<(UOp, String)> = Vec::new();
    for node in root.postorder() {
        let source = |index: usize| {
            &terms.iter().find(|(old, _)| old.ptr_eq(&node.src[index])).unwrap().1
        };
        let term = match node.op {
            Ops::Const => match node.arg {
                Arg::F32(bits) => format!("{:e}f", f32::from_bits(bits)),
                _ => unreachable!(),
            },
            Ops::Add => format!("({} + {})", source(0), source(1)),
            Ops::Mul => format!("({} * {})", source(0), source(1)),
            Ops::Param => match node.arg {
                Arg::ScalarParam { index } => format!("p{index}"),
                _ => unreachable!(),
            },
        };
        terms.push((node, term));
    }
    terms.pop().unwrap().1
}
```

`src/codegen.rs` の `pub fn source`を置き換える。

```rust
// file="src/codegen.rs" mode="replace" item="pub fn source"
pub fn source(root: &UOp, parameters: usize) -> String {
    let declarations = (0..parameters)
        .map(|i| format!("float p{i} = strtof(argv[{}], 0);", i + 1))
        .collect::<String>();
    format!(
        "#include <stdio.h>\n#include <stdlib.h>\nint main(int argc, char **argv) {{ {}printf(\"%.9g\\n\", (double)({})); }}\n",
        declarations, expression(root)
    )
}
```

`src/codegen.rs` の `pub fn run`を置き換える。

```rust
// file="src/codegen.rs" mode="replace" item="pub fn run"
pub fn run(root: &UOp, inputs: &[f32]) -> Result<f32> {
    let source = source(root, inputs.len());
    let dir = std::env::temp_dir().join(format!("chibigrad-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    fs::create_dir(&dir).map_err(|e| e.to_string())?;
    let result = (|| {
        let input = dir.join("scalar.c");
        let binary = dir.join("scalar");
        fs::write(&input, source).map_err(|e| e.to_string())?;
        let compiler = std::env::var_os("CC").unwrap_or_else(|| {
            if cfg!(target_os = "macos") { "/usr/bin/clang".into() } else { "clang".into() }
        });
        let compile = Command::new(compiler).args(["-std=c11", "-O2"])
            .arg(&input).arg("-o").arg(&binary).output().map_err(|e| e.to_string())?;
        if !compile.status.success() { return Err(String::from_utf8_lossy(&compile.stderr).into()); }
        let output = Command::new(binary).args(inputs.iter().map(|x| x.to_string())).output().map_err(|e| e.to_string())?;
        if !output.status.success() { return Err("scalar executable failed".into()); }
        String::from_utf8_lossy(&output.stdout).trim().parse::<f32>().map_err(|e| e.to_string())
    })();
    fs::remove_dir_all(&dir).map_err(|e| e.to_string())?;
    result
}
```

`src/uop.rs` の `pub enum Ops`を置き換える。

```rust
// file="src/uop.rs" mode="replace" item="pub enum Ops"
pub enum Ops {
    Const,
    Add,
    Mul,
    Param,
}
```

`src/uop.rs` の `pub enum Arg`を置き換える。

```rust
// file="src/uop.rs" mode="replace" item="pub enum Arg"
pub enum Arg {
    None,
    F32(u32),
    ScalarParam { index: usize },
}
```

`src/uop.rs` の `impl UOp` 内を追記する。

```rust
// file="src/uop.rs" mode="append" within="impl UOp"
    pub fn scalar_param(index: usize) -> Self {
        Self::new(Ops::Param, vec![], Arg::ScalarParam { index })
    }
```

`tests/scalar.rs` の `fn constant_runs_native_code`を置き換える。

```rust
// file="tests/scalar.rs" mode="replace" item="fn constant_runs_native_code"
fn constant_runs_native_code() {
    assert_eq!(run(&(UOp::f32(7.0)), &[]).unwrap(), 7.0);
}
```

`tests/scalar.rs` の `fn addition_runs_native_code`を置き換える。

```rust
// file="tests/scalar.rs" mode="replace" item="fn addition_runs_native_code"
fn addition_runs_native_code() {
    assert_eq!(run(&(UOp::f32(2.0).add(&UOp::f32(3.0))), &[]).unwrap(), 5.0);
}
```

`tests/scalar.rs` の `fn multiplication_runs_native_code`を置き換える。

```rust
// file="tests/scalar.rs" mode="replace" item="fn multiplication_runs_native_code"
fn multiplication_runs_native_code() {
    let sum = UOp::f32(2.0).add(&UOp::f32(3.0));
    assert_eq!(run(&(sum.mul(&UOp::f32(4.0))), &[]).unwrap(), 20.0);
}
```

`tests/scalar.rs`を追記する。

```rust
// file="tests/scalar.rs" mode="append"


#[test]
fn parameter_is_read_at_runtime() {
    let expression = UOp::scalar_param(0).add(&UOp::f32(3.0)).mul(&UOp::f32(4.0));
    assert_eq!(run(&(expression), &[2.0]).unwrap(), 20.0);
}
```

```sh
# check step=02.01
cargo test -- --nocapture
cargo run --example scalar_const -- --dump-source --dump-graph
cargo run --example scalar_arithmetic -- --dump-source --dump-graph
```

定数、加算、乗算の確認を維持し、Param を使う式も 20 を返す。

## 02.02 共有ライブラリと packed ABI

実行ファイルの起動を共有ライブラリの関数呼び出しに置き換える。入力を 4 バイトずつ詰め、末尾の 4 バイトを出力に使う。C 側は memcpy で読み書きするため、バイト配列のアラインメントに依存しない。CpuProgram が clang の成果物、関数ポインタ、ライブラリの寿命を所有する。run は生の ABI を扱う unsafe 関数で、呼び出し側が必要なバイト数とポインタの有効性を保証する。既存テストは同じ値の確認をこの呼び出しへ移す。 共有ライブラリはロード時の初期化と解放時の終了処理も実行するため、compileにもunsafe契約を付ける。教材では検証した生成元のCだけを渡す。

`examples/scalar_arithmetic.rs` の `fn main`を置き換える。

```rust
// file="examples/scalar_arithmetic.rs" mode="replace" item="fn main"
fn main() -> chibigrad::Result<()> {
    let root = UOp::f32(2.0).add(&UOp::f32(3.0)).mul(&UOp::f32(4.0));
    let source = chibigrad::codegen::emit_scalar(&root, 0);
    for flag in std::env::args().skip(1) {
        match flag.as_str() {
            "--dump-source" => println!("{source}"),
            "--dump-graph" => {
                for node in root.postorder() {
                    println!("{:?} {:?} inputs={}", node.op, node.arg, node.src.len());
                }
            }
            _ => return Err(format!("unknown option: {flag}")),
        }
    }
    let program = unsafe { chibigrad::cpu::CpuProgram::compile(&source, "scalar") }?;
    let mut bytes = 0f32.to_ne_bytes();
    unsafe {
        program.run(&mut bytes)?;
    }
    let result = f32::from_ne_bytes(bytes);
    println!("{result}");
    Ok(())
}
```

`examples/scalar_const.rs` の `fn main`を置き換える。

```rust
// file="examples/scalar_const.rs" mode="replace" item="fn main"
fn main() -> chibigrad::Result<()> {
    let root = UOp::f32(7.0);
    let source = chibigrad::codegen::emit_scalar(&root, 0);
    for flag in std::env::args().skip(1) {
        match flag.as_str() {
            "--dump-source" => println!("{source}"),
            "--dump-graph" => {
                for node in root.postorder() {
                    println!("{:?} {:?} inputs={}", node.op, node.arg, node.src.len());
                }
            }
            _ => return Err(format!("unknown option: {flag}")),
        }
    }
    let program = unsafe { chibigrad::cpu::CpuProgram::compile(&source, "scalar") }?;
    let mut bytes = 0f32.to_ne_bytes();
    unsafe {
        program.run(&mut bytes)?;
    }
    let result = f32::from_ne_bytes(bytes);
    println!("{result}");
    Ok(())
}
```

`src/codegen.rs`を保存する。

```rust
// file="src/codegen.rs" mode="write"
use crate::{Arg, Ops, UOp};

pub fn expression(root: &UOp) -> String {
    let root = root.rewrite(|_| None);
    let mut terms: Vec<(UOp, String)> = Vec::new();
    for node in root.postorder() {
        let source = |index: usize| {
            &terms.iter().find(|(old, _)| old.ptr_eq(&node.src[index])).unwrap().1
        };
        let term = match node.op {
            Ops::Const => match node.arg {
                Arg::F32(bits) => format!("{:e}f", f32::from_bits(bits)),
                _ => unreachable!(),
            },
            Ops::Add => format!("({} + {})", source(0), source(1)),
            Ops::Mul => format!("({} * {})", source(0), source(1)),
            Ops::Param => match node.arg {
                Arg::ScalarParam { index } => format!("p{index}"),
                _ => unreachable!(),
            },
        };
        terms.push((node, term));
    }
    terms.pop().unwrap().1
}

pub fn emit_scalar(root: &UOp, parameters: usize) -> String {
    let declarations = (0..parameters)
        .map(|i| format!("float p{i}; memcpy(&p{i}, args + {}, 4);\n", 4 * i))
        .collect::<String>();
    format!(
        "#include <string.h>\nvoid scalar(unsigned char *args) {{\n{declarations}float result = {};\nmemcpy(args + {}, &result, 4);\n}}\n",
        expression(root), 4 * parameters
    )
}
```

`src/cpu.rs`を保存する。

```rust
// file="src/cpu.rs" mode="write"
use crate::Result;
use std::{ffi::{c_char, c_int, c_void, CStr, CString}, fs, path::PathBuf, process::Command, sync::atomic::{AtomicUsize, Ordering}};

#[cfg_attr(target_os = "linux", link(name = "dl"))]
unsafe extern "C" {
    fn dlopen(path: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}
static NEXT: AtomicUsize = AtomicUsize::new(0);

pub struct CpuProgram {
    handle: *mut c_void,
    entry: unsafe extern "C" fn(*mut u8),
    object: Vec<u8>,
    source: String,
}

impl CpuProgram {
    /// Compile and load trusted C source.
    ///
    /// # Safety
    /// The library's initializers and finalizers must preserve Rust memory and thread safety.
    /// Loading can execute initializers immediately; dropping the program can execute finalizers.
    ///
    /// ```compile_fail
    /// #![forbid(unsafe_code)]
    /// let _ = chibigrad::cpu::CpuProgram::compile("void entry(void *args){}", "entry");
    /// ```
    pub unsafe fn compile(source: &str, entry: &str) -> Result<Self> {
        let symbol = CString::new(entry).map_err(|e| e.to_string())?;
        let dir = Scratch::new()?;
        let input = dir.0.join("scalar.c");
        let library = dir.0.join("scalar.so");
        fs::write(&input, source).map_err(|e| e.to_string())?;
        let compiler = std::env::var_os("CC").unwrap_or_else(|| {
            if cfg!(target_os = "macos") { "/usr/bin/clang".into() } else { "clang".into() }
        });
        let output = Command::new(compiler).args(["-std=c11", "-O2", "-shared", "-fPIC"])
            .arg(&input).arg("-o").arg(&library).output().map_err(|e| e.to_string())?;
        if !output.status.success() { return Err(String::from_utf8_lossy(&output.stderr).into()); }
        let object = fs::read(&library).map_err(|e| e.to_string())?;
        let path = CString::new(library.to_str().ok_or("non-UTF-8 library path")?).map_err(|e| e.to_string())?;
        let handle = unsafe { dlopen(path.as_ptr(), 2) };
        if handle.is_null() { return Err(loader_error()); }
        unsafe { dlerror(); }
        let address = unsafe { dlsym(handle, symbol.as_ptr()) };
        if address.is_null() {
            let message = loader_error();
            unsafe { dlclose(handle); }
            return Err(message);
        }
        let entry = unsafe { std::mem::transmute::<*mut c_void, unsafe extern "C" fn(*mut u8)>(address) };
        Ok(Self { handle, entry, object, source: source.into() })
    }

    /// # Safety
    /// The entry must have the C signature `void entry(unsigned char *)`.
    /// The caller must provide every byte, pointer and lifetime required by that code.
    pub unsafe fn run(&self, args: &mut [u8]) -> Result<()> {
        unsafe { (self.entry)(args.as_mut_ptr()); }
        Ok(())
    }
    pub fn object(&self) -> &[u8] { &self.object }
    pub fn source(&self) -> Option<&str> { Some(&self.source) }
}
impl Drop for CpuProgram {
    fn drop(&mut self) { unsafe { dlclose(self.handle); } }
}
fn loader_error() -> String {
    let error = unsafe { dlerror() };
    if error.is_null() { "dynamic loader failed".into() }
    else { unsafe { CStr::from_ptr(error) }.to_string_lossy().into() }
}
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Result<Self> {
        let path = std::env::temp_dir().join(format!("chibigrad-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::create_dir(&path).map_err(|e| e.to_string())?;
        Ok(Self(path))
    }
}
impl Drop for Scratch {
    fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
}
```

`src/lib.rs`を追記する。

```rust
// file="src/lib.rs" mode="append"

pub mod cpu;
```

`tests/scalar.rs`を保存する。

```rust
// file="tests/scalar.rs" mode="write"
use chibigrad::UOp;
use chibigrad::{codegen::emit_scalar, cpu::CpuProgram};
fn run(root: &UOp, inputs: &[f32]) -> f32 {
    let program = unsafe {
        CpuProgram::compile(&emit_scalar(root, inputs.len()), "scalar")
    }
        .unwrap();
    let mut args = inputs.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<_>>();
    args.extend(0f32.to_ne_bytes());
    unsafe {
        program.run(&mut args).unwrap();
    }
    f32::from_ne_bytes(args[4 * inputs.len()..].try_into().unwrap())
}
#[test]
fn constant_runs_native_code() {
    assert_eq!(run(& (UOp::f32(7.0)), & []), 7.0);
}
#[test]
fn addition_runs_native_code() {
    assert_eq!(run(& (UOp::f32(2.0).add(& UOp::f32(3.0))), & []), 5.0);
}
#[test]
fn multiplication_runs_native_code() {
    let sum = UOp::f32(2.0).add(&UOp::f32(3.0));
    assert_eq!(run(& (sum.mul(& UOp::f32(4.0))), & []), 20.0);
}
#[test]
fn parameter_is_read_at_runtime() {
    let expression = UOp::scalar_param(0).add(&UOp::f32(3.0)).mul(&UOp::f32(4.0));
    assert_eq!(run(& (expression), & [2.0]), 20.0);
}
#[test]
fn shared_program_keeps_source_and_binary() {
    let source = emit_scalar(&UOp::f32(7.0), 0);
    let program = unsafe { CpuProgram::compile(&source, "scalar") }.unwrap();
    assert_eq!(program.source(), Some(source.as_str()));
    assert!(! program.object().is_empty());
    assert!(unsafe { CpuProgram::compile(& source, "missing") } .is_err());
}
#[test]
fn postorder_and_identity_rewrite() {
    let root = UOp::f32(2.0).add(&UOp::f32(3.0)).mul(&UOp::f32(4.0));
    let order = root.postorder();
    assert!(order.last().unwrap().ptr_eq(& root));
    for (i, node) in order.iter().enumerate() {
        for child in &node.src {
            assert!(order[..i].iter().any(| earlier | earlier.ptr_eq(child)));
        }
    }
    assert!(root.rewrite(| _ | None).ptr_eq(& root));
}
```

```sh
# check step=02.02
cargo test -- --nocapture
cargo run --example scalar_const -- --dump-source --dump-graph
cargo run --example scalar_arithmetic -- --dump-source --dump-graph
cargo test --offline --doc
```

既存の数値と走査の確認を維持する。共有ライブラリから 7、5、20 を得て、元の C とバイナリも取得できる。存在しない entry はエラーになる。

## 02.03 同じバイナリを二度呼ぶ

(p0 + p1) * p2 を一度だけコンパイルする。ひとつの CpuProgram に [2, 3, 4] と [1, 2, 4] を順に渡す。再コンパイルせずに入力を束縛し直せることを、20 と 12 の結果と同じバイナリの保持で確認する。

`examples/scalar_inputs.rs`を保存する。

```rust
// file="examples/scalar_inputs.rs" mode="write"
use chibigrad::{codegen::emit_scalar, cpu::CpuProgram, UOp};
fn main() -> chibigrad::Result<()> {
    let mut show_stats = false;
    for flag in std::env::args().skip(1) {
        match flag.as_str() {
            "--stats" => show_stats = true,
            _ => return Err(format!("unknown option: {flag}")),
        }
    }
    let root = UOp::scalar_param(0)
        .add(&UOp::scalar_param(1))
        .mul(&UOp::scalar_param(2));
    let mut compiles = 0;
    let mut runs = 0;
    let program = unsafe { CpuProgram::compile(&emit_scalar(&root, 3), "scalar") }?;
    compiles += 1;
    for values in [[2.0f32, 3.0, 4.0], [1.0, 2.0, 4.0]] {
        let mut bytes = values
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect::<Vec<_>>();
        bytes.extend(0f32.to_ne_bytes());
        unsafe {
            program.run(&mut bytes)?;
        }
        runs += 1;
        println!("{}", f32::from_ne_bytes(bytes[12..16].try_into().unwrap()));
    }
    if show_stats {
        println!("compiles={compiles} runs={runs}");
    }
    Ok(())
}
```

`tests/scalar.rs`を追記する。

```rust
// file="tests/scalar.rs" mode="append"

#[test]
fn one_binary_accepts_two_bindings() {
    let expression = UOp::scalar_param(0)
        .add(&UOp::scalar_param(1))
        .mul(&UOp::scalar_param(2));
    let source = emit_scalar(&expression, 3);
    let program = unsafe { CpuProgram::compile(&source, "scalar") }.unwrap();
    let binary = program.object().to_vec();
    let mut results = Vec::new();
    for values in [[2.0f32, 3.0, 4.0], [1.0, 2.0, 4.0]] {
        let mut args = values.into_iter().flat_map(f32::to_ne_bytes).collect::<Vec<_>>();
        args.extend(0f32.to_ne_bytes());
        unsafe {
            program.run(&mut args).unwrap();
        }
        results.push(f32::from_ne_bytes(args[12..16].try_into().unwrap()));
    }
    assert_eq!(results, [20.0, 12.0]);
    assert_eq!(program.object(), binary);
    println!("same binary: {} bytes; results: {:?}", binary.len(), results);
}
```

```sh
# check step=02.03
cargo test -- --nocapture
cargo run --example scalar_const -- --dump-source --dump-graph
cargo run --example scalar_arithmetic -- --dump-source --dump-graph
cargo run --example scalar_inputs -- --stats
```

既存の確認を維持する。ログに same binary と results: [20.0, 12.0] が表示される。scalar_inputs --stats は 20、12、compiles=1 runs=2 を表示する。

## 章末の累積確認

前の章までのテストとexampleも再ビルドします。

```sh
# check
cargo test --quiet --all-targets
```

## 要件との対応

### 02-001 scalar `Param` を追加し、`slot` の有効性を検証する。

<!-- todo:02-001 -->

関連する実装と確認は本章の `02.01`, `02.02`, `02.03` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 02-002 `a+b` の C 関数を生成し、example の入力を wrapper に渡す。

<!-- todo:02-002 -->

関連する実装と確認は本章の `02.01`, `02.02`, `02.03` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 02-003 入力を C の定数にせず、実行時に受け取る。

<!-- todo:02-003 -->

関連する実装と確認は本章の `02.01`, `02.02`, `02.03` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 02-004 前章の定数実行例も同じ renderer で通す。

<!-- todo:02-004 -->

関連する実装と確認は本章の `02.01`, `02.02`, `02.03` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 02-005 固定 ABI の `chibi_entry` と wrapper を生成する。

<!-- todo:02-005 -->

関連する実装と確認は本章の `02.01`, `02.02`, `02.03` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 02-006 macOS は clang の `-dynamiclib`、対応する Linux は `-shared -fPIC` を使う。

<!-- todo:02-006 -->

関連する実装と確認は本章の `02.01`, `02.02`, `02.03` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 02-007 OS の動的 library 機構でロードし、入口を解決する。

<!-- todo:02-007 -->

関連する実装と確認は本章の `02.01`, `02.02`, `02.03` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 02-008 library の所有物を `CompiledScalar` に保持する。

<!-- todo:02-008 -->

関連する実装と確認は本章の `02.01`, `02.02`, `02.03` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 02-009 `a+b` を compile、load、run まで通し、結果 `5` を読む。

<!-- todo:02-009 -->

関連する実装と確認は本章の `02.01`, `02.02`, `02.03` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 02-010 PARAM 三つで式を作る `scalar_inputs` example を追加する。

<!-- todo:02-010 -->

関連する実装と確認は本章の `02.01`, `02.02`, `02.03` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 02-011 compile と run を別の呼出しにする。

<!-- todo:02-011 -->

関連する実装と確認は本章の `02.01`, `02.02`, `02.03` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 02-012 二回目はスロットのビット列だけを更新する。

<!-- todo:02-012 -->

関連する実装と確認は本章の `02.01`, `02.02`, `02.03` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 02-013 同じ入口と library を使ったことを example の計数で示す。

<!-- todo:02-013 -->

関連する実装と確認は本章の `02.01`, `02.02`, `02.03` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 02-014 source の dump に入力値が含まれないことを確かめる。

<!-- todo:02-014 -->

関連する実装と確認は本章の `02.01`, `02.02`, `02.03` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 02-015 不足、余剰、存在しないスロットを run 前に拒否する。

<!-- todo:02-015 -->

関連する実装と確認は本章の `02.01`, `02.02`, `02.03` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 02-016 symbol 不在は library path と入口名を含むエラーにする。

<!-- todo:02-016 -->

関連する実装と確認は本章の `02.01`, `02.02`, `02.03` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 02-017 compile 失敗と run の入力エラーを区別する。

<!-- todo:02-017 -->

関連する実装と確認は本章の `02.01`, `02.02`, `02.03` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 02-018 所有物の破棄で library と一時ファイルを解放する。

<!-- todo:02-018 -->

関連する実装と確認は本章の `02.01`, `02.02`, `02.03` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 02-019 成功例で `20`、`12` を得る。

<!-- todo:02-019 -->

関連する実装と確認は本章の `02.01`, `02.02`, `02.03` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 02-020 compile の予定計数が `1`、run が `2` になる。

<!-- todo:02-020 -->

関連する実装と確認は本章の `02.01`, `02.02`, `02.03` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 02-021 生成 C は実行時 scalar スロットを参照する。

<!-- todo:02-021 -->

関連する実装と確認は本章の `02.01`, `02.02`, `02.03` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 02-022 入力不足では FFI を実行しない。

<!-- todo:02-022 -->

関連する実装と確認は本章の `02.01`, `02.02`, `02.03` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 02-023 前章の定数例も成功する。

<!-- todo:02-023 -->

関連する実装と確認は本章の `02.01`, `02.02`, `02.03` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 02-024 library の所有物を破棄した後に関数ポインタだけが公開されない。

<!-- todo:02-024 -->

関連する実装と確認は本章の `02.01`, `02.02`, `02.03` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。
