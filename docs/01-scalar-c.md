# 01. 定数 7 を C にして実行し、加算と乗算で 20 を返す

最初のデータは `Vec<Op>` と root の番号だけである。恒等 rewrite、依存順の走査、C の出力、clang、生成実行ファイルの順に進む。`Op::Add` と `Op::Mul` は二項の enum variant なので入力数の違いは Rust の型で表現できない。入力番号と循環は `order` が検査する。

`file=... mode=write` はそのファイルの全置換、`mode=append` は末尾への追記である。すべて前章でできたプロジェクトの root で行う。`sh check` は実際の検証コマンドであり、ソースファイルには保存しない。

## 1. Cargo package と `scalar_const` example を必要な分だけ追加する。
<!-- todo:01-001 -->

```toml file=Cargo.toml mode=write
[package]
name = "chinigrad"
version = "0.1.0"
edition = "2024"

[dependencies]
```

```rust file=src/lib.rs mode=write
pub type Result<T> = std::result::Result<T, String>;
pub mod bootstrap;
```

```rust file=examples/scalar_const.rs mode=write
use chinigrad::{
	Result,
	bootstrap::{Graph, Op},
};
fn main() -> Result<()> {
	let args = std::env::args().skip(1).collect::<Vec<_>>();
	let value = args
		.iter()
		.find(|x| !x.starts_with("--"))
		.map(|s| s.parse::<f32>().map_err(|e| e.to_string()))
		.transpose()?
		.unwrap_or(7.);
	let mut g = Graph::default();
	let root = g.node(Op::Const(value));
	if args.iter().any(|s| s == "--dump-graph") {
		eprintln!("{g:#?}")
	}
	if args.iter().any(|s| s == "--dump-source") {
		eprintln!("{}", g.source(root)?)
	}
	let output = g.run(root)?;
	let actual = output.trim().parse::<f32>().map_err(|e| e.to_string())?;
	if actual != value {
		return Err("generated result mismatch".into());
	}
	print!("{output}");
	Ok(())
}
```

## 2. `Const(f32)` と root を保持し、一つのノードを構築する。
<!-- todo:01-002 -->

```rust file=src/bootstrap.rs mode=write
use crate::Result;
use std::{
	path::PathBuf,
	process::Command,
	sync::atomic::{AtomicU64, Ordering},
};
#[derive(Clone, Debug)]
pub enum Op {
	Const(f32),
	Add(usize, usize),
	Mul(usize, usize),
}
#[derive(Clone, Debug, Default)]
pub struct Graph {
	pub nodes: Vec<Op>,
}
```

## 3. 恒等 rewrite と依存順 linearize を経由して C を生成する。
<!-- todo:01-003 -->

```rust file=src/bootstrap.rs mode=append
impl Graph {
	pub fn node(&mut self, op: Op) -> usize {
		let id = self.nodes.len();
		self.nodes.push(op);
		id
	}
	pub fn rewrite(&self, root: usize) -> Result<(Self, usize)> {
		self.order(root)?;
		Ok((self.clone(), root))
	}
	pub fn order(&self, root: usize) -> Result<Vec<usize>> {
		fn visit(g: &Graph, id: usize, seen: &mut [bool], out: &mut Vec<usize>) -> Result<()> {
			let node = g.nodes.get(id).ok_or("invalid root or input")?;
			if seen[id] {
				return Ok(());
			}
			match node {
				Op::Add(a, b) | Op::Mul(a, b) => {
					if *a >= id || *b >= id {
						return Err("forward or cyclic dependency".into());
					}
					visit(g, *a, seen, out)?;
					visit(g, *b, seen, out)?
				}
				_ => {}
			}
			seen[id] = true;
			out.push(id);
			Ok(())
		}
		let mut out = vec![];
		visit(self, root, &mut vec![false; self.nodes.len()], &mut out)?;
		Ok(out)
	}
	pub fn source(&self, root: usize) -> Result<String> {
		let (g, root) = self.rewrite(root)?;
		let mut source="#include <stdio.h>\n#include <stdint.h>\n#include <string.h>\nstatic float bits(uint32_t x){float y;memcpy(&y,&x,4);return y;}\nint main(void){\n".to_string();
		for id in g.order(root)? {
			let expression = match g.nodes[id] {
				Op::Const(value) => format!("bits({}u)", value.to_bits()),
				Op::Add(a, b) => format!("v{a}+v{b}"),
				Op::Mul(a, b) => format!("v{a}*v{b}"),
			};
			source.push_str(&format!("float v{id}={expression};\n"));
		}
		source.push_str(&format!(
			"printf(\"%.9g\\n\",(double)v{root});\nreturn 0;\n}}\n"
		));
		Ok(source)
	}
	pub fn run(&self, root: usize) -> Result<String> {
		run_source(&self.source(root)?, None)
	}
}
```

## 4. 一時ディレクトリに C を書き、system clang を子プロセスとして起動する。
<!-- todo:01-004 -->

```rust file=src/bootstrap.rs mode=append
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Temporary(PathBuf);
impl Drop for Temporary {
	fn drop(&mut self) {
		let _ = std::fs::remove_dir_all(&self.0);
	}
}
pub fn run_source(source: &str, compiler: Option<&str>) -> Result<String> {
	let dir = Temporary(std::env::temp_dir().join(format!(
		"chinigrad-exe-{}-{}",
		std::process::id(),
		NEXT.fetch_add(1, Ordering::Relaxed)
	)));
	std::fs::create_dir(&dir.0).map_err(|e| e.to_string())?;
	let c = dir.0.join("program.c");
	let binary = dir.0.join("program");
	std::fs::write(&c, source).map_err(|e| e.to_string())?;
	let mut command = Command::new(compiler.unwrap_or(if cfg!(target_os = "macos") {
		"/usr/bin/clang"
	} else {
		"clang"
	}));
	command.args(["-std=c11", "-O2"]);
	if cfg!(target_os = "macos") {
		let sdk = Command::new("xcrun")
			.args(["--sdk", "macosx", "--show-sdk-path"])
			.output()
			.map_err(|e| e.to_string())?;
		if !sdk.status.success() {
			return Err("cannot locate macOS SDK".into());
		}
		let sdk = String::from_utf8_lossy(&sdk.stdout);
		command
			.arg("-isysroot")
			.arg(sdk.trim())
			.arg(format!("--sysroot={}", sdk.trim()));
	}
	let output = command
		.arg(&c)
		.arg("-o")
		.arg(&binary)
		.output()
		.map_err(|e| format!("cannot launch clang: {e}"))?;
	if !output.status.success() {
		return Err(format!(
			"clang compilation failed: {}",
			String::from_utf8_lossy(&output.stderr)
		));
	}
	let output = Command::new(&binary)
		.output()
		.map_err(|e| format!("cannot launch generated executable: {e}"))?;
	if !output.status.success() {
		return Err(format!("generated executable failed: {}", output.status));
	}
	String::from_utf8(output.stdout).map_err(|e| e.to_string())
}
```

## 5. 生成実行ファイルを実行し、成功終了と標準出力 `7` を照合する。
<!-- todo:01-005 -->

```rust file=tests/scalar_c.rs mode=write
use chinigrad::{
	Result,
	bootstrap::{Graph, Op, run_source},
};
#[test]
fn const_and_arithmetic() -> Result<()> {
	let mut g = Graph::default();
	let seven = g.node(Op::Const(7.));
	assert_eq!(g.run(seven)?.trim(), "7");
	let a = g.node(Op::Const(2.));
	let b = g.node(Op::Const(3.));
	let sum = g.node(Op::Add(a, b));
	let c = g.node(Op::Const(4.));
	let root = g.node(Op::Mul(sum, c));
	assert_eq!(g.run(root)?.trim(), "20");
	assert!(g.source(root)?.contains('+') && g.source(root)?.contains('*'));
	assert_eq!(g.order(root)?.iter().filter(|&&id| id == sum).count(), 1);
	assert_eq!(g.rewrite(root)?.1, root);
	Ok(())
}
#[test]
fn rejects_invalid_graph_and_compiler() {
	let mut g = Graph::default();
	assert!(g.source(0).is_err());
	g.nodes.push(Op::Add(0, 0));
	assert!(g.source(0).is_err());
	assert!(
		run_source("bad C", None)
			.unwrap_err()
			.contains("clang compilation failed")
	);
	assert!(
		run_source(
			"int main(void){return 0;}",
			Some("/missing/chinigrad-clang")
		)
		.unwrap_err()
		.contains("cannot launch clang")
	);
}
```

`tests/scalar_c.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test scalar_c
```

## 6. `Add` と `Mul` に二つの `NodeId` を持たせる。
<!-- todo:01-006 -->

`tests/scalar_c.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test scalar_c
```

## 7. `Const(2)`, `Const(3)`, `Add`, `Const(4)`, `Mul` を組み立てる。
<!-- todo:01-007 -->

```rust file=examples/scalar_arithmetic.rs mode=write
use chinigrad::{
	Result,
	bootstrap::{Graph, Op},
};
fn main() -> Result<()> {
	let args = std::env::args().skip(1).collect::<Vec<_>>();
	let values = args
		.iter()
		.filter(|s| !s.starts_with("--"))
		.map(|s| s.parse::<f32>().map_err(|e| e.to_string()))
		.collect::<Result<Vec<_>>>()?;
	let values = if values.is_empty() {
		vec![2., 3., 4.]
	} else {
		values
	};
	if values.len() != 3 {
		return Err("expected a b c".into());
	}
	let mut g = Graph::default();
	let a = g.node(Op::Const(values[0]));
	let b = g.node(Op::Const(values[1]));
	let sum = g.node(Op::Add(a, b));
	let c = g.node(Op::Const(values[2]));
	let root = g.node(Op::Mul(sum, c));
	if args.iter().any(|s| s == "--dump-graph") {
		eprintln!("{g:#?}")
	}
	if args.iter().any(|s| s == "--dump-source") {
		eprintln!("{}", g.source(root)?)
	}
	let out = g.run(root)?;
	assert_eq!(
		out.trim().parse::<f32>().unwrap(),
		(values[0] + values[1]) * values[2]
	);
	print!("{out}");
	Ok(())
}
```

`tests/scalar_c.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test scalar_c
```

## 8. 依存先を利用前に一度だけ並べる。
<!-- todo:01-008 -->

`tests/scalar_c.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test scalar_c
```

## 9. C renderer に加算と乗算を足し、生成コードを実行する。
<!-- todo:01-009 -->

`tests/scalar_c.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test scalar_c
```

## 10. 第 1 コミットの定数の実行例も再実行する。
<!-- todo:01-010 -->

`tests/scalar_c.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test scalar_c
```

## 11. `--dump-graph` でノード番号、演算、入力を出す。
<!-- todo:01-011 -->

`tests/scalar_c.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test scalar_c
```

## 12. `--dump-source` で clang に渡した C を出す。
<!-- todo:01-012 -->

`tests/scalar_c.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test scalar_c
```

## 13. root の番号と演算の入力数を生成前に検証する。
<!-- todo:01-013 -->

`tests/scalar_c.rs` の `rejects_invalid_graph_and_compiler` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test scalar_c rejects_invalid_graph_and_compiler
```

## 14. clang 不在とコンパイル失敗を区別する。
<!-- todo:01-014 -->

`tests/scalar_c.rs` の `rejects_invalid_graph_and_compiler` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test scalar_c rejects_invalid_graph_and_compiler
```

## 15. stdout の値を読み、予定値と比較する実行テストを作る。
<!-- todo:01-015 -->

`tests/scalar_c.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test scalar_c
```

## 16. `scalar_const` が生成コードを実行して `7` を返す。
<!-- todo:01-016 -->

`tests/scalar_c.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test scalar_c
```

## 17. `scalar_arithmetic` が定数式 `(2+3)*4` を構築して `20` を返す。
<!-- todo:01-017 -->

`tests/scalar_c.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test scalar_c
```

## 18. グラフで `ADD` の結果を `MUL` が参照している。
<!-- todo:01-018 -->

`tests/scalar_c.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test scalar_c
```

## 19. source に実際の `+` と `*` が残っている。
<!-- todo:01-019 -->

`tests/scalar_c.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test scalar_c
```

## 20. 不正な root と入力数は clang の起動前にエラーになる。
<!-- todo:01-020 -->

`tests/scalar_c.rs` の `rejects_invalid_graph_and_compiler` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test scalar_c rejects_invalid_graph_and_compiler
```

## 21. 初回から恒等 rewrite と linearize を通過する。
<!-- todo:01-021 -->

`tests/scalar_c.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test scalar_c
```

## 章末の累積確認

前章までの example と test も同じ checkout で再ビルドする。

```sh check
cargo test --quiet --all-targets
```

```sh check
cargo run --quiet --example scalar_const -- 7 --dump-source
cargo run --quiet --example scalar_arithmetic -- 2 3 4 --dump-graph --dump-source
```
