# 01. 定数だけの UOp から始めて Add と Mul を足す

前の節で動かしたコードに、今回使う演算と処理を追加します。各節の確認を実行してから次へ進みます。

参照するUOpの構成は [tinygrad c22edff](https://github.com/tinygrad/tinygrad/blob/c22edff14c79310a9b2df62a78819cf77c8289fa/tinygrad/uop/ops.py) です。最初から全機能を実装せず、使う節で増やします。

## 01.01 定数を C にして実行する

まず値ひとつだけを表す。UOp は Rc<Node> を包み、Node は op、src、arg を持つ。Const は有限の f32 をビット列で保持する。postorder は入力から親の順にたどり、rewrite はその順序で入力を置き換えてから規則を適用する。ここでは何も置き換えない規則を通し、同じノードを保つ。コード生成も同じ走査順を使い、clang が作った実行ファイルから 7 を受け取る。 コンパイラは CC で指定できる。未指定なら macOS では /usr/bin/clang、それ以外では clang を使う。

`Cargo.toml`を保存する。

```toml
# file="Cargo.toml" mode="write"
[package]
name = "chibigrad"
version = "0.1.0"
edition = "2024"
```

`src/codegen.rs`を保存する。

```rust
// file="src/codegen.rs" mode="write"
use crate::{Arg, Ops, UOp};

pub fn expression(root: &UOp) -> String {
    let root = root.rewrite(|_| None);
    let mut terms: Vec<(UOp, String)> = Vec::new();
    for node in root.postorder() {
        let term = match node.op {
            Ops::Const => match node.arg {
                Arg::F32(bits) => format!("{:e}f", f32::from_bits(bits)),
            },
        };
        terms.push((node, term));
    }
    terms.pop().unwrap().1
}


use crate::Result;
use std::{fs, process::Command, sync::atomic::{AtomicUsize, Ordering}};
static NEXT: AtomicUsize = AtomicUsize::new(0);

pub fn source(root: &UOp) -> String {
    format!(
        "#include <stdio.h>\nint main(void) {{ printf(\"%.9g\\n\", (double)({})); }}\n",
        expression(root)
    )
}

pub fn run(root: &UOp) -> Result<f32> {
    let source = source(root);
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
        let output = Command::new(binary).output().map_err(|e| e.to_string())?;
        if !output.status.success() { return Err("scalar executable failed".into()); }
        String::from_utf8_lossy(&output.stdout).trim().parse::<f32>().map_err(|e| e.to_string())
    })();
    fs::remove_dir_all(&dir).map_err(|e| e.to_string())?;
    result
}
```

`src/lib.rs`を保存する。

```rust
// file="src/lib.rs" mode="write"
pub type Result<T> = std::result::Result<T, String>;
pub mod uop;
pub mod codegen;
pub use uop::{Arg, Ops, UOp};
```

`src/uop.rs`を保存する。

```rust
// file="src/uop.rs" mode="write"
use std::{ops::Deref, rc::Rc};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ops {
    Const,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Arg {
    F32(u32),
}
#[derive(Debug)]
pub struct Node {
    pub op: Ops,
    pub src: Vec<UOp>,
    pub arg: Arg,
}
#[derive(Clone, Debug)]
pub struct UOp(Rc<Node>);
impl Deref for UOp {
    type Target = Node;
    fn deref(&self) -> &Node { &self.0 }
}
impl UOp {
    pub fn new(op: Ops, src: Vec<UOp>, arg: Arg) -> Self {
        Self(Rc::new(Node { op, src, arg }))
    }
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
    pub fn postorder(&self) -> Vec<Self> {
        fn visit(node: &UOp, order: &mut Vec<UOp>) {
            for child in &node.src { visit(child, order); }
            order.push(node.clone());
        }
        let mut order = Vec::new();
        visit(self, &mut order);
        order
    }
    pub fn rewrite(&self, rule: impl Fn(&Self) -> Option<Self>) -> Self {
        let mut rewritten: Vec<(Self, Self)> = Vec::new();
        for node in self.postorder() {
            let src: Vec<Self> = node.src.iter().map(|child| {
                rewritten.iter().find(|(old, _)| old.ptr_eq(child)).unwrap().1.clone()
            }).collect();
            let unchanged = src.iter().zip(&node.src).all(|(a, b)| a.ptr_eq(b));
            let rebuilt = if unchanged { node.clone() }
                else { Self::new(node.op, src, node.arg.clone()) };
            let result = rule(&rebuilt).unwrap_or(rebuilt);
            rewritten.push((node, result));
        }
        rewritten.pop().unwrap().1
    }
    pub fn f32(value: f32) -> Self {
        assert!(value.is_finite());
        Self::new(Ops::Const, vec![], Arg::F32(value.to_bits()))
    }
}
```

`tests/scalar.rs`を保存する。

```rust
// file="tests/scalar.rs" mode="write"
use chibigrad::UOp;
use chibigrad::codegen::run;

#[test]
fn constant_runs_native_code() {
    assert_eq!(run(&(UOp::f32(7.0))).unwrap(), 7.0);
}

#[test]
fn postorder_and_identity_rewrite() {
    let root = UOp::f32(7.0);
    let order = root.postorder();
    assert!(order.last().unwrap().ptr_eq(&root));
    for (i, node) in order.iter().enumerate() {
        for child in &node.src {
            assert!(order[..i].iter().any(|earlier| earlier.ptr_eq(child)));
        }
    }
    assert!(root.rewrite(|_| None).ptr_eq(&root));
}
```

```sh
# check step=01.01
cargo test -- --nocapture
```

定数 7 を実行ファイルから得る。postorder と恒等書換えのテストも通る。

## 01.02 Add と二つの入力

Ops に Add、Arg に None を追加する。add は左右の UOp を src に保存する。定数で使った postorder の走査で、先に得た二つの入力の式から C の加算式を作る。定数の実行確認を残したまま 2 + 3 を実行する。

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
        };
        terms.push((node, term));
    }
    terms.pop().unwrap().1
}
```

`src/uop.rs` の `pub enum Ops`を置き換える。

```rust
// file="src/uop.rs" mode="replace" item="pub enum Ops"
pub enum Ops {
    Const,
    Add,
}
```

`src/uop.rs` の `pub enum Arg`を置き換える。

```rust
// file="src/uop.rs" mode="replace" item="pub enum Arg"
pub enum Arg {
    None,
    F32(u32),
}
```

`src/uop.rs` の `impl UOp` 内を追記する。

```rust
// file="src/uop.rs" mode="append" within="impl UOp"
    pub fn add(&self, rhs: &Self) -> Self {
        Self::new(Ops::Add, vec![self.clone(), rhs.clone()], Arg::None)
    }
```

`tests/scalar.rs` の `fn postorder_and_identity_rewrite`を置き換える。

```rust
// file="tests/scalar.rs" mode="replace" item="fn postorder_and_identity_rewrite"
fn postorder_and_identity_rewrite() {
    let root = UOp::f32(2.0).add(&UOp::f32(3.0));
    let order = root.postorder();
    assert!(order.last().unwrap().ptr_eq(&root));
    for (i, node) in order.iter().enumerate() {
        for child in &node.src {
            assert!(order[..i].iter().any(|earlier| earlier.ptr_eq(child)));
        }
    }
    assert!(root.rewrite(|_| None).ptr_eq(&root));
}
```

`tests/scalar.rs`を追記する。

```rust
// file="tests/scalar.rs" mode="append"


#[test]
fn addition_runs_native_code() {
    assert_eq!(run(&(UOp::f32(2.0).add(&UOp::f32(3.0)))).unwrap(), 5.0);
}
```

```sh
# check step=01.02
cargo test -- --nocapture
```

定数 7 と加算 5、入力が親より先に現れる走査、同じノードを保つ恒等書換えを確認する。

## 01.03 Mul を加えて式を組む

Mul も二つの入力を持つ。既存の Add を入力にして (2 + 3) * 4 を組み、生成した C を clang でコンパイルする。Rust の計算結果で代用せず、実行ファイルから 20 を受け取る。

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
        };
        terms.push((node, term));
    }
    terms.pop().unwrap().1
}
```

`src/uop.rs` の `pub enum Ops`を置き換える。

```rust
// file="src/uop.rs" mode="replace" item="pub enum Ops"
pub enum Ops {
    Const,
    Add,
    Mul,
}
```

`src/uop.rs` の `impl UOp` 内を追記する。

```rust
// file="src/uop.rs" mode="append" within="impl UOp"
    pub fn mul(&self, rhs: &Self) -> Self {
        Self::new(Ops::Mul, vec![self.clone(), rhs.clone()], Arg::None)
    }
```

`tests/scalar.rs` の `fn postorder_and_identity_rewrite`を置き換える。

```rust
// file="tests/scalar.rs" mode="replace" item="fn postorder_and_identity_rewrite"
fn postorder_and_identity_rewrite() {
    let root = UOp::f32(2.0).add(&UOp::f32(3.0)).mul(&UOp::f32(4.0));
    let order = root.postorder();
    assert!(order.last().unwrap().ptr_eq(&root));
    for (i, node) in order.iter().enumerate() {
        for child in &node.src {
            assert!(order[..i].iter().any(|earlier| earlier.ptr_eq(child)));
        }
    }
    assert!(root.rewrite(|_| None).ptr_eq(&root));
}
```

`tests/scalar.rs`を追記する。

```rust
// file="tests/scalar.rs" mode="append"


#[test]
fn multiplication_runs_native_code() {
    let sum = UOp::f32(2.0).add(&UOp::f32(3.0));
    assert_eq!(run(&(sum.mul(&UOp::f32(4.0)))).unwrap(), 20.0);
}
```

```sh
# check step=01.03
cargo test -- --nocapture
```

定数、加算、乗算、postorder と恒等書換えのテストが通り、複合式が 20 を返す。

## 01.04 生成した C とグラフを表示する

scalar_const と scalar_arithmetic を実行例にする。--dump-source で clang に渡す C を表示し、--dump-graph で postorder の演算と引数を表示する。表示した式と実行結果の対応を手元で追えるようにする。

`examples/scalar_arithmetic.rs`を保存する。

```rust
// file="examples/scalar_arithmetic.rs" mode="write"
use chibigrad::UOp;

fn main() -> chibigrad::Result<()> {
    let root = UOp::f32(2.0).add(&UOp::f32(3.0)).mul(&UOp::f32(4.0));
    let source = chibigrad::codegen::source(&root);
    for flag in std::env::args().skip(1) {
        match flag.as_str() {
            "--dump-source" => println!("{source}"),
            "--dump-graph" => for node in root.postorder() {
                println!("{:?} {:?} inputs={}", node.op, node.arg, node.src.len());
            },
            _ => return Err(format!("unknown option: {flag}")),
        }
    }
    let result = chibigrad::codegen::run(&root)?;
    println!("{result}");
    Ok(())
}
```

`examples/scalar_const.rs`を保存する。

```rust
// file="examples/scalar_const.rs" mode="write"
use chibigrad::UOp;

fn main() -> chibigrad::Result<()> {
    let root = UOp::f32(7.0);
    let source = chibigrad::codegen::source(&root);
    for flag in std::env::args().skip(1) {
        match flag.as_str() {
            "--dump-source" => println!("{source}"),
            "--dump-graph" => for node in root.postorder() {
                println!("{:?} {:?} inputs={}", node.op, node.arg, node.src.len());
            },
            _ => return Err(format!("unknown option: {flag}")),
        }
    }
    let result = chibigrad::codegen::run(&root)?;
    println!("{result}");
    Ok(())
}
```

```sh
# check step=01.04
cargo test -- --nocapture
cargo run --example scalar_const -- --dump-source --dump-graph
cargo run --example scalar_arithmetic -- --dump-source --dump-graph
```

定数のグラフと C、実行結果 7 を表示する。複合式は Const、Add、Mul の順で表示し、実行結果は 20 になる。

## 章末の累積確認

前の章までのテストとexampleも再ビルドします。

```sh
# check
cargo test --quiet --all-targets
```

## 要件との対応

### 01-001 Cargo package と `scalar_const` example を必要な分だけ追加する。

<!-- todo:01-001 -->

関連する実装と確認は本章の `01.01`, `01.02`, `01.03`, `01.04` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 01-002 `Const(f32)` と root を保持し、一つのノードを構築する。

<!-- todo:01-002 -->

関連する実装と確認は本章の `01.01`, `01.02`, `01.03`, `01.04` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 01-003 恒等 rewrite と依存順 linearize を経由して C を生成する。

<!-- todo:01-003 -->

関連する実装と確認は本章の `01.01`, `01.02`, `01.03`, `01.04` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 01-004 一時ディレクトリに C を書き、system clang を子プロセスとして起動する。

<!-- todo:01-004 -->

関連する実装と確認は本章の `01.01`, `01.02`, `01.03`, `01.04` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 01-005 生成実行ファイルを実行し、成功終了と標準出力 `7` を照合する。

<!-- todo:01-005 -->

関連する実装と確認は本章の `01.01`, `01.02`, `01.03`, `01.04` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 01-006 `Add` と `Mul` に二つの `NodeId` を持たせる。

<!-- todo:01-006 -->

関連する実装と確認は本章の `01.01`, `01.02`, `01.03`, `01.04` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 01-007 `Const(2)`, `Const(3)`, `Add`, `Const(4)`, `Mul` を組み立てる。

<!-- todo:01-007 -->

関連する実装と確認は本章の `01.01`, `01.02`, `01.03`, `01.04` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 01-008 依存先を利用前に一度だけ並べる。

<!-- todo:01-008 -->

関連する実装と確認は本章の `01.01`, `01.02`, `01.03`, `01.04` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 01-009 C renderer に加算と乗算を足し、生成コードを実行する。

<!-- todo:01-009 -->

関連する実装と確認は本章の `01.01`, `01.02`, `01.03`, `01.04` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 01-010 第 1 コミットの定数の実行例も再実行する。

<!-- todo:01-010 -->

関連する実装と確認は本章の `01.01`, `01.02`, `01.03`, `01.04` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 01-011 `--dump-graph` でノード番号、演算、入力を出す。

<!-- todo:01-011 -->

関連する実装と確認は本章の `01.01`, `01.02`, `01.03`, `01.04` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 01-012 `--dump-source` で clang に渡した C を出す。

<!-- todo:01-012 -->

関連する実装と確認は本章の `01.01`, `01.02`, `01.03`, `01.04` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 01-013 root の番号と演算の入力数を生成前に検証する。

<!-- todo:01-013 -->

関連する実装と確認は本章の `01.01`, `01.02`, `01.03`, `01.04` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 01-014 clang 不在とコンパイル失敗を区別する。

<!-- todo:01-014 -->

関連する実装と確認は本章の `01.01`, `01.02`, `01.03`, `01.04` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 01-015 stdout の値を読み、予定値と比較する実行テストを作る。

<!-- todo:01-015 -->

関連する実装と確認は本章の `01.01`, `01.02`, `01.03`, `01.04` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 01-016 `scalar_const` が生成コードを実行して `7` を返す。

<!-- todo:01-016 -->

関連する実装と確認は本章の `01.01`, `01.02`, `01.03`, `01.04` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 01-017 `scalar_arithmetic` が定数式 `(2+3)*4` を構築して `20` を返す。

<!-- todo:01-017 -->

関連する実装と確認は本章の `01.01`, `01.02`, `01.03`, `01.04` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 01-018 グラフで `ADD` の結果を `MUL` が参照している。

<!-- todo:01-018 -->

関連する実装と確認は本章の `01.01`, `01.02`, `01.03`, `01.04` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 01-019 source に実際の `+` と `*` が残っている。

<!-- todo:01-019 -->

関連する実装と確認は本章の `01.01`, `01.02`, `01.03`, `01.04` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 01-020 不正な root と入力数は clang の起動前にエラーになる。

<!-- todo:01-020 -->

関連する実装と確認は本章の `01.01`, `01.02`, `01.03`, `01.04` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 01-021 初回から恒等 rewrite と linearize を通過する。

<!-- todo:01-021 -->

関連する実装と確認は本章の `01.01`, `01.02`, `01.03`, `01.04` にある。`tests/scalar.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。
