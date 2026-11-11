# 02. 同じ生成コードへ実行時入力を渡して 20 と 12 を返す

入力値は `uint64_t` の scalar 配列に f32 bits として格納する。ポインタ配列の第0要素が出力である。入口は `int chibi_entry(void*const* buffers,const uint64_t* scalars)` に統一する。元の void 入口案に対する変更は戻り値の status だけで、引数 ABI は二配列のままである。library と関数ポインタの寿命は一つの所有物に閉じる。macOS では xcrun の SDK をヘッダと linker の両方へ渡す。

`file=... mode=write` はそのファイルの全置換、`mode=append` は末尾への追記である。すべて前章でできたプロジェクトの root で行う。`# check` は実際の検証コマンドであり、ソースファイルには保存しない。

## 1. scalar `Param` を追加し、`slot` の有効性を検証する。
<!-- todo:02-001 -->

```rust
// file=src/native.rs mode=write
use crate::Result;
use std::{
	ffi::{CStr, CString, c_char, c_void},
	path::PathBuf,
	process::Command,
	sync::atomic::{AtomicU64, Ordering},
};

static NEXT: AtomicU64 = AtomicU64::new(0);
pub struct NativeLibrary {
	handle: *mut c_void,
	entry: *mut c_void,
	directory: PathBuf,
}
#[cfg_attr(target_os = "linux", link(name = "dl"))]
unsafe extern "C" {
	fn dlopen(path: *const c_char, flags: i32) -> *mut c_void;
	fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
	fn dlclose(handle: *mut c_void) -> i32;
	fn dlerror() -> *const c_char;
}
fn loader_error() -> String {
	unsafe {
		let e = dlerror();
		if e.is_null() {
			"unknown loader error".into()
		} else {
			CStr::from_ptr(e).to_string_lossy().into_owned()
		}
	}
}
```

```rust
// file=src/inputs.rs mode=write
use crate::{Result, native::NativeLibrary};
use std::{collections::HashSet, ffi::c_void};
#[derive(Clone, Debug)]
pub enum Op {
	Const(f32),
	Param(usize),
	Add(usize, usize),
	Mul(usize, usize),
}
#[derive(Clone, Debug, Default)]
pub struct Graph {
	pub nodes: Vec<Op>,
}
impl Graph {
	pub fn node(&mut self, op: Op) -> usize {
		let id = self.nodes.len();
		self.nodes.push(op);
		id
	}
	fn order(&self, root: usize) -> Result<Vec<usize>> {
		fn visit(
			g: &Graph,
			id: usize,
			seen: &mut HashSet<usize>,
			out: &mut Vec<usize>,
		) -> Result<()> {
			let node = g.nodes.get(id).ok_or("invalid scalar root/input")?;
			if seen.insert(id) {
				if let Op::Add(a, b) | Op::Mul(a, b) = node {
					if *a >= id || *b >= id {
						return Err("cyclic or forward input".into());
					}
					visit(g, *a, seen, out)?;
					visit(g, *b, seen, out)?
				}
				out.push(id)
			}
			Ok(())
		}
		let mut out = vec![];
		visit(self, root, &mut HashSet::new(), &mut out)?;
		Ok(out)
	}
	pub fn compile(&self, root: usize) -> Result<Compiled> {
		let order = self.order(root)?;
		let mut source="#include <stdint.h>\n#include <string.h>\nstatic float bits(uint32_t x){float y;memcpy(&y,&x,4);return y;}\nint chibi_entry(void*const* buffers,const uint64_t* scalars){\n".to_string();
		let mut slots = HashSet::new();
		for id in order {
			let expression = match self.nodes[id] {
				Op::Const(v) => format!("bits({}u)", v.to_bits()),
				Op::Param(slot) => {
					slots.insert(slot);
					format!("bits((uint32_t)scalars[{slot}])")
				}
				Op::Add(a, b) => format!("v{a}+v{b}"),
				Op::Mul(a, b) => format!("v{a}*v{b}"),
			};
			source.push_str(&format!("float v{id}={expression};\n"));
		}
		if (0..slots.len()).any(|s| !slots.contains(&s)) {
			return Err("PARAM slots must be dense".into());
		}
		source.push_str(&format!(
			"uint32_t output;memcpy(&output,&v{root},4);*(uint64_t*)buffers[0]=output;return 0;\n}}\n"
		));
		let library = unsafe { NativeLibrary::compile(&source, "chibi_entry") }?;
		Ok(Compiled {
			source,
			library,
			slots: slots.len(),
			runs: 0,
		})
	}
}
```

```rust
// file=src/lib.rs mode=append
pub mod inputs;
pub mod native;
```

## 2. `a+b` の C 関数を生成し、example の入力を wrapper に渡す。
<!-- todo:02-002 -->

```rust
// file=tests/runtime_inputs.rs mode=write
use chibigrad::{
	Result,
	inputs::{Graph, Op},
};
#[test]
fn same_library_two_runs() -> Result<()> {
	let mut g = Graph::default();
	let a = g.node(Op::Param(0));
	let b = g.node(Op::Param(1));
	let c = g.node(Op::Param(2));
	let sum = g.node(Op::Add(a, b));
	let root = g.node(Op::Mul(sum, c));
	let mut plan = g.compile(root)?;
	assert!(plan.source.contains("scalars[0]"));
	assert_eq!(plan.run(&[2., 3., 4.])?, 20.);
	assert_eq!(plan.run(&[5., 1., 2.])?, 12.);
	assert_eq!(plan.runs, 2);
	assert!(plan.run(&[2., 3.]).is_err());
	assert_eq!(plan.runs, 2);
	assert!(plan.run(&[2., 3., 4., 5.]).is_err());
	let mut bad = Graph::default();
	let missing = bad.node(Op::Param(1));
	assert!(bad.compile(missing).is_err());
	Ok(())
}
#[test]
fn constant_and_add() -> Result<()> {
	let mut g = Graph::default();
	let c = g.node(Op::Const(7.));
	assert_eq!(g.compile(c)?.run(&[])?, 7.);
	let a = g.node(Op::Param(0));
	let b = g.node(Op::Param(1));
	let sum = g.node(Op::Add(a, b));
	assert_eq!(g.compile(sum)?.run(&[2., 3.])?, 5.);
	Ok(())
}
```

## 3. 入力を C の定数にせず、実行時に受け取る。
<!-- todo:02-003 -->

```rust
// file=src/inputs.rs mode=append
pub fn scalar_words(values: &[f32]) -> Vec<u64> {
	values.iter().map(|v| v.to_bits() as u64).collect()
}
```

## 4. 前章の定数実行例も同じ renderer で通す。
<!-- todo:02-004 -->

```rust
// file=examples/scalar_const.rs mode=write
use chibigrad::{
	Result,
	inputs::{Graph, Op},
};
fn main() -> Result<()> {
	let args = std::env::args().skip(1).collect::<Vec<_>>();
	let value = args
		.iter()
		.find(|s| !s.starts_with("--"))
		.map(|s| s.parse::<f32>().map_err(|e| e.to_string()))
		.transpose()?
		.unwrap_or(7.);
	let mut g = Graph::default();
	let y = g.node(Op::Const(value));
	let mut plan = g.compile(y)?;
	if args.iter().any(|s| s == "--dump-source") {
		eprintln!("{}", plan.source)
	}
	let output = plan.run(&[])?;
	assert_eq!(output, value);
	println!("{output}");
	Ok(())
}
```

## 5. 固定 ABI の `chibi_entry` と wrapper を生成する。
<!-- todo:02-005 -->

```rust
// file=src/inputs.rs mode=append
pub struct Compiled {
	pub source: String,
	library: NativeLibrary,
	slots: usize,
	pub runs: usize,
}
impl Compiled {
	pub fn run(&mut self, inputs: &[f32]) -> Result<f32> {
		if inputs.len() != self.slots {
			return Err("scalar input count mismatch".into());
		}
		let words = scalar_words(inputs);
		let mut output = 0u64;
		let buffers = [(&mut output as *mut u64).cast::<c_void>()];
		type Entry = unsafe extern "C" fn(*const *mut c_void, *const u64) -> i32;
		let entry: Entry = unsafe { std::mem::transmute(self.library.symbol()) };
		let status = unsafe { entry(buffers.as_ptr(), words.as_ptr()) };
		if status != 0 {
			return Err("scalar entry rejected inputs".into());
		}
		self.runs += 1;
		Ok(f32::from_bits(output as u32))
	}
}
```

## 6. macOS は clang の `-dynamiclib`、対応する Linux は `-shared -fPIC` を使う。
<!-- todo:02-006 -->

```rust
// file=src/native.rs mode=append
impl NativeLibrary {
	pub fn directory(&self) -> &std::path::Path {
		&self.directory
	}
	/// # Safety
	/// The C source, compiler, and any library initializers must be trusted to
	/// execute safely in this process. Loading a library can run its constructors.
	pub unsafe fn compile(source: &str, entry: &str) -> Result<Self> {
		let compiler = std::env::var("CC").unwrap_or_else(|_| {
			if cfg!(target_os = "macos") {
				"/usr/bin/clang".into()
			} else {
				"clang".into()
			}
		});
		unsafe { Self::compile_with_compiler(source, entry, &compiler) }
	}
	/// # Safety
	/// The caller must trust the compiler, C source, and library initializers.
	pub unsafe fn compile_with_compiler(source: &str, entry: &str, compiler: &str) -> Result<Self> {
		let directory = std::env::temp_dir().join(format!(
			"chibigrad-{}-{}",
			std::process::id(),
			NEXT.fetch_add(1, Ordering::Relaxed)
		));
		std::fs::create_dir(&directory).map_err(|e| e.to_string())?;
		let result = (|| {
			let c = directory.join("kernel.c");
			let dylib = directory.join("kernel.dylib");
			std::fs::write(&c, source).map_err(|e| e.to_string())?;
			let mut command = Command::new(compiler);
			command.args(["-std=c11", "-O2", "-fno-fast-math", "-ffp-contract=off"]);
			if cfg!(target_os = "macos") {
				let sdk = Command::new("xcrun")
					.args(["--sdk", "macosx", "--show-sdk-path"])
					.output()
					.map_err(|e| format!("cannot locate macOS SDK: {e}"))?;
				if !sdk.status.success() {
					return Err("xcrun failed to locate macOS SDK".into());
				}
				let sdk = String::from_utf8_lossy(&sdk.stdout);
				command
					.arg("-isysroot")
					.arg(sdk.trim())
					.arg(format!("--sysroot={}", sdk.trim()));
			}
			if cfg!(target_os = "macos") {
				command.arg("-dynamiclib");
			} else {
				command.args(["-shared", "-fPIC"]);
			}
			let output = command
				.arg(&c)
				.arg("-o")
				.arg(&dylib)
				.arg("-lm")
				.output()
				.map_err(|e| format!("cannot launch clang: {e}"))?;
			if !output.status.success() {
				return Err(format!(
					"clang compilation failed: {}",
					String::from_utf8_lossy(&output.stderr)
				));
			}
			let name =
				CString::new(dylib.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
			let symbol = CString::new(entry).map_err(|e| e.to_string())?;
			unsafe {
				let handle = dlopen(name.as_ptr(), 2);
				if handle.is_null() {
					return Err(format!("dlopen {}: {}", dylib.display(), loader_error()));
				}
				let pointer = dlsym(handle, symbol.as_ptr());
				if pointer.is_null() {
					let e = format!("dlsym {} {entry}: {}", dylib.display(), loader_error());
					dlclose(handle);
					return Err(e);
				}
				Ok(Self {
					handle,
					entry: pointer,
					directory: directory.clone(),
				})
			}
		})();
		if result.is_err() {
			let _ = std::fs::remove_dir_all(directory);
		}
		result
	}
	/// Returns the resolved entry pointer while retaining library ownership.
	///
	/// # Safety
	/// Calls through this pointer must use the compiled ABI and valid allocations.
	/// The pointer must never be called after this library is dropped.
	pub unsafe fn symbol(&self) -> *mut c_void {
		self.entry
	}
}
```

`tests/runtime_inputs.rs` の正常系と境界系をまとめて再実行する。

```sh
# check
cargo test --quiet --test runtime_inputs
```

## 7. OS の動的 library 機構でロードし、入口を解決する。
<!-- todo:02-007 -->

`tests/runtime_inputs.rs` の正常系と境界系をまとめて再実行する。

`src/native.rs` の `pub unsafe fn compile_with_compiler(`の定義を次の全体で置き換えます。

```rust
// file="src/native.rs" mode="replace" item="pub unsafe fn compile_with_compiler("
	pub unsafe fn compile_with_compiler(source: &str, entry: &str, compiler: &str) -> Result<Self> {
		let directory = std::env::temp_dir().join(format!(
			"chibigrad-{}-{}",
			std::process::id(),
			NEXT.fetch_add(1, Ordering::Relaxed)
		));
		std::fs::create_dir(&directory).map_err(|e| e.to_string())?;
		let result = (|| {
			let c = directory.join("kernel.c");
			let dylib = directory.join("kernel.dylib");
			std::fs::write(&c, source).map_err(|e| e.to_string())?;
			let mut command = Command::new(compiler);
			command.args(["-std=c11", "-O2", "-fno-fast-math", "-ffp-contract=off"]);
			if cfg!(target_os = "macos") {
				let sdk = Command::new("xcrun")
					.args(["--sdk", "macosx", "--show-sdk-path"])
					.output()
					.map_err(|e| format!("cannot locate macOS SDK: {e}"))?;
				if !sdk.status.success() {
					return Err("xcrun failed to locate macOS SDK".into());
				}
				let sdk = String::from_utf8_lossy(&sdk.stdout);
				command
					.arg("-isysroot")
					.arg(sdk.trim())
					.arg(format!("--sysroot={}", sdk.trim()));
			}
			if cfg!(target_os = "macos") {
				command.arg("-dynamiclib");
			} else {
				command.args(["-shared", "-fPIC"]);
			}
			let output = command
				.arg(&c)
				.arg("-o")
				.arg(&dylib)
				.arg("-lm")
				.output()
				.map_err(|e| format!("cannot launch clang: {e}"))?;
			if !output.status.success() {
				return Err(format!(
					"clang compilation failed: {}",
					String::from_utf8_lossy(&output.stderr)
				));
			}
			let name =
				CString::new(dylib.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
			let symbol = CString::new(entry).map_err(|e| e.to_string())?;
			unsafe {
				let handle = dlopen(name.as_ptr(), 2);
				if handle.is_null() {
					return Err(format!("dlopen {}: {}", dylib.display(), loader_error()));
				}
				let pointer = dlsym(handle, symbol.as_ptr());
				if pointer.is_null() {
					let e = format!("dlsym {} {entry}: {}", dylib.display(), loader_error());
					dlclose(handle);
					return Err(e);
				}
				Ok(Self {
					handle,
					entry: pointer,
					directory: directory.clone(),
				})
			}
		})();
		if result.is_err() {
			let _ = std::fs::remove_dir_all(directory);
		}
		result
	}
```

```sh
# check
cargo test --quiet --test runtime_inputs
```

## 8. library の所有物を `CompiledScalar` に保持する。
<!-- todo:02-008 -->

`tests/runtime_inputs.rs` の正常系と境界系をまとめて再実行する。

`src/inputs.rs` の `pub struct Compiled`の定義を次の全体で置き換えます。

```rust
// file="src/inputs.rs" mode="replace" item="pub struct Compiled"
pub struct Compiled {
	pub source: String,
	library: NativeLibrary,
	slots: usize,
	pub runs: usize,
}
```

```sh
# check
cargo test --quiet --test runtime_inputs
```

## 9. `a+b` を compile、load、run まで通し、結果 `5` を読む。
<!-- todo:02-009 -->

`tests/runtime_inputs.rs` の `constant_and_add` で、この経路の値または拒否条件を確認する。

`tests/runtime_inputs.rs` の `fn constant_and_add(`の定義を次の全体で置き換えます。

```rust
// file="tests/runtime_inputs.rs" mode="replace" item="fn constant_and_add("
fn constant_and_add() -> Result<()> {
	let mut g = Graph::default();
	let c = g.node(Op::Const(7.));
	assert_eq!(g.compile(c)?.run(&[])?, 7.);
	let a = g.node(Op::Param(0));
	let b = g.node(Op::Param(1));
	let sum = g.node(Op::Add(a, b));
	assert_eq!(g.compile(sum)?.run(&[2., 3.])?, 5.);
	Ok(())
}
```

```sh
# check
cargo test --quiet --test runtime_inputs constant_and_add
```

## 10. PARAM 三つで式を作る `scalar_inputs` example を追加する。
<!-- todo:02-010 -->

```rust
// file=examples/scalar_inputs.rs mode=write
use chibigrad::{
	Result,
	inputs::{Graph, Op},
};
fn main() -> Result<()> {
	let args = std::env::args().skip(1).collect::<Vec<_>>();
	let mut before = vec![];
	let mut after = vec![];
	let mut second = false;
	for arg in &args {
		if arg == "--then" {
			second = true
		} else if arg == "--dump-source" {
		} else {
			let value = arg.parse::<f32>().map_err(|e| e.to_string())?;
			if second {
				after.push(value)
			} else {
				before.push(value)
			}
		}
	}
	if before.is_empty() {
		before = vec![2., 3., 4.];
		after = vec![5., 1., 2.]
	}
	let mut g = Graph::default();
	let a = g.node(Op::Param(0));
	let b = g.node(Op::Param(1));
	let c = g.node(Op::Param(2));
	let sum = g.node(Op::Add(a, b));
	let root = g.node(Op::Mul(sum, c));
	let mut plan = g.compile(root)?;
	if args.iter().any(|s| s == "--dump-source") {
		eprintln!("{}", plan.source)
	}
	println!("{}", plan.run(&before)?);
	if !after.is_empty() {
		println!("{}", plan.run(&after)?);
	}
	eprintln!("compile=1 run={}", plan.runs);
	Ok(())
}
```

`tests/runtime_inputs.rs` の正常系と境界系をまとめて再実行する。

```sh
# check
cargo test --quiet --test runtime_inputs
```

## 11. compile と run を別の呼出しにする。
<!-- todo:02-011 -->

`tests/runtime_inputs.rs` の正常系と境界系をまとめて再実行する。

`src/inputs.rs` の `pub fn compile(`の定義を次の全体で置き換えます。

```rust
// file="src/inputs.rs" mode="replace" item="pub fn compile("
	pub fn compile(&self, root: usize) -> Result<Compiled> {
		let order = self.order(root)?;
		let mut source="#include <stdint.h>\n#include <string.h>\nstatic float bits(uint32_t x){float y;memcpy(&y,&x,4);return y;}\nint chibi_entry(void*const* buffers,const uint64_t* scalars){\n".to_string();
		let mut slots = HashSet::new();
		for id in order {
			let expression = match self.nodes[id] {
				Op::Const(v) => format!("bits({}u)", v.to_bits()),
				Op::Param(slot) => {
					slots.insert(slot);
					format!("bits((uint32_t)scalars[{slot}])")
				}
				Op::Add(a, b) => format!("v{a}+v{b}"),
				Op::Mul(a, b) => format!("v{a}*v{b}"),
			};
			source.push_str(&format!("float v{id}={expression};\n"));
		}
		if (0..slots.len()).any(|s| !slots.contains(&s)) {
			return Err("PARAM slots must be dense".into());
		}
		source.push_str(&format!(
			"uint32_t output;memcpy(&output,&v{root},4);*(uint64_t*)buffers[0]=output;return 0;\n}}\n"
		));
		let library = unsafe { NativeLibrary::compile(&source, "chibi_entry") }?;
		Ok(Compiled {
			source,
			library,
			slots: slots.len(),
			runs: 0,
		})
	}
```

```sh
# check
cargo test --quiet --test runtime_inputs
```

## 12. 二回目はスロットのビット列だけを更新する。
<!-- todo:02-012 -->

`tests/runtime_inputs.rs` の正常系と境界系をまとめて再実行する。

`src/inputs.rs` の `pub fn run(`の定義を次の全体で置き換えます。

```rust
// file="src/inputs.rs" mode="replace" item="pub fn run("
	pub fn run(&mut self, inputs: &[f32]) -> Result<f32> {
		if inputs.len() != self.slots {
			return Err("scalar input count mismatch".into());
		}
		let words = scalar_words(inputs);
		let mut output = 0u64;
		let buffers = [(&mut output as *mut u64).cast::<c_void>()];
		type Entry = unsafe extern "C" fn(*const *mut c_void, *const u64) -> i32;
		let entry: Entry = unsafe { std::mem::transmute(self.library.symbol()) };
		let status = unsafe { entry(buffers.as_ptr(), words.as_ptr()) };
		if status != 0 {
			return Err("scalar entry rejected inputs".into());
		}
		self.runs += 1;
		Ok(f32::from_bits(output as u32))
	}
```

```sh
# check
cargo test --quiet --test runtime_inputs
```

## 13. 同じ入口と library を使ったことを example の計数で示す。
<!-- todo:02-013 -->

`tests/runtime_inputs.rs` の正常系と境界系をまとめて再実行する。

`tests/runtime_inputs.rs` の `fn same_library_two_runs(`の定義を次の全体で置き換えます。

```rust
// file="tests/runtime_inputs.rs" mode="replace" item="fn same_library_two_runs("
fn same_library_two_runs() -> Result<()> {
	let mut g = Graph::default();
	let a = g.node(Op::Param(0));
	let b = g.node(Op::Param(1));
	let c = g.node(Op::Param(2));
	let sum = g.node(Op::Add(a, b));
	let root = g.node(Op::Mul(sum, c));
	let mut plan = g.compile(root)?;
	assert!(plan.source.contains("scalars[0]"));
	assert_eq!(plan.run(&[2., 3., 4.])?, 20.);
	assert_eq!(plan.run(&[5., 1., 2.])?, 12.);
	assert_eq!(plan.runs, 2);
	assert!(plan.run(&[2., 3.]).is_err());
	assert_eq!(plan.runs, 2);
	assert!(plan.run(&[2., 3., 4., 5.]).is_err());
	let mut bad = Graph::default();
	let missing = bad.node(Op::Param(1));
	assert!(bad.compile(missing).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test runtime_inputs
```

## 14. source の dump に入力値が含まれないことを確かめる。
<!-- todo:02-014 -->

`tests/runtime_inputs.rs` の正常系と境界系をまとめて再実行する。

`tests/runtime_inputs.rs` の `fn same_library_two_runs(`の定義を次の全体で置き換えます。

```rust
// file="tests/runtime_inputs.rs" mode="replace" item="fn same_library_two_runs("
fn same_library_two_runs() -> Result<()> {
	let mut g = Graph::default();
	let a = g.node(Op::Param(0));
	let b = g.node(Op::Param(1));
	let c = g.node(Op::Param(2));
	let sum = g.node(Op::Add(a, b));
	let root = g.node(Op::Mul(sum, c));
	let mut plan = g.compile(root)?;
	assert!(plan.source.contains("scalars[0]"));
	assert_eq!(plan.run(&[2., 3., 4.])?, 20.);
	assert_eq!(plan.run(&[5., 1., 2.])?, 12.);
	assert_eq!(plan.runs, 2);
	assert!(plan.run(&[2., 3.]).is_err());
	assert_eq!(plan.runs, 2);
	assert!(plan.run(&[2., 3., 4., 5.]).is_err());
	let mut bad = Graph::default();
	let missing = bad.node(Op::Param(1));
	assert!(bad.compile(missing).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test runtime_inputs
```

## 15. 不足、余剰、存在しないスロットを run 前に拒否する。
<!-- todo:02-015 -->

`tests/runtime_inputs.rs` の正常系と境界系をまとめて再実行する。

`src/inputs.rs` の `pub fn run(`の定義を次の全体で置き換えます。

```rust
// file="src/inputs.rs" mode="replace" item="pub fn run("
	pub fn run(&mut self, inputs: &[f32]) -> Result<f32> {
		if inputs.len() != self.slots {
			return Err("scalar input count mismatch".into());
		}
		let words = scalar_words(inputs);
		let mut output = 0u64;
		let buffers = [(&mut output as *mut u64).cast::<c_void>()];
		type Entry = unsafe extern "C" fn(*const *mut c_void, *const u64) -> i32;
		let entry: Entry = unsafe { std::mem::transmute(self.library.symbol()) };
		let status = unsafe { entry(buffers.as_ptr(), words.as_ptr()) };
		if status != 0 {
			return Err("scalar entry rejected inputs".into());
		}
		self.runs += 1;
		Ok(f32::from_bits(output as u32))
	}
```

```sh
# check
cargo test --quiet --test runtime_inputs
```

## 16. symbol 不在は library path と入口名を含むエラーにする。
<!-- todo:02-016 -->

```rust
// file=tests/runtime_inputs.rs mode=append
use chibigrad::native::NativeLibrary;
#[test]
fn native_errors() -> Result<()> {
	let source = "int chibi_entry(void*const* b,const unsigned long long* s){return 0;}";
	let error = unsafe { NativeLibrary::compile(source, "absent") }
		.err()
		.unwrap();
	assert!(error.contains("absent") && error.contains("kernel.dylib"));
	assert!(
		unsafe { NativeLibrary::compile("not valid C", "chibi_entry") }
			.err()
			.unwrap()
			.contains("clang compilation failed")
	);
	assert!(
		unsafe {
			NativeLibrary::compile_with_compiler(
				source,
				"chibi_entry",
				"/definitely/missing/chibi-clang",
			)
		}
		.err()
		.unwrap()
		.contains("cannot launch clang")
	);
	Ok(())
}
```

`tests/runtime_inputs.rs` の `native_errors` で、この経路の値または拒否条件を確認する。

`src/native.rs` の `pub unsafe fn compile_with_compiler(`の定義を次の全体で置き換えます。

```rust
// file="src/native.rs" mode="replace" item="pub unsafe fn compile_with_compiler("
	pub unsafe fn compile_with_compiler(source: &str, entry: &str, compiler: &str) -> Result<Self> {
		let directory = std::env::temp_dir().join(format!(
			"chibigrad-{}-{}",
			std::process::id(),
			NEXT.fetch_add(1, Ordering::Relaxed)
		));
		std::fs::create_dir(&directory).map_err(|e| e.to_string())?;
		let result = (|| {
			let c = directory.join("kernel.c");
			let dylib = directory.join("kernel.dylib");
			std::fs::write(&c, source).map_err(|e| e.to_string())?;
			let mut command = Command::new(compiler);
			command.args(["-std=c11", "-O2", "-fno-fast-math", "-ffp-contract=off"]);
			if cfg!(target_os = "macos") {
				let sdk = Command::new("xcrun")
					.args(["--sdk", "macosx", "--show-sdk-path"])
					.output()
					.map_err(|e| format!("cannot locate macOS SDK: {e}"))?;
				if !sdk.status.success() {
					return Err("xcrun failed to locate macOS SDK".into());
				}
				let sdk = String::from_utf8_lossy(&sdk.stdout);
				command
					.arg("-isysroot")
					.arg(sdk.trim())
					.arg(format!("--sysroot={}", sdk.trim()));
			}
			if cfg!(target_os = "macos") {
				command.arg("-dynamiclib");
			} else {
				command.args(["-shared", "-fPIC"]);
			}
			let output = command
				.arg(&c)
				.arg("-o")
				.arg(&dylib)
				.arg("-lm")
				.output()
				.map_err(|e| format!("cannot launch clang: {e}"))?;
			if !output.status.success() {
				return Err(format!(
					"clang compilation failed: {}",
					String::from_utf8_lossy(&output.stderr)
				));
			}
			let name =
				CString::new(dylib.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
			let symbol = CString::new(entry).map_err(|e| e.to_string())?;
			unsafe {
				let handle = dlopen(name.as_ptr(), 2);
				if handle.is_null() {
					return Err(format!("dlopen {}: {}", dylib.display(), loader_error()));
				}
				let pointer = dlsym(handle, symbol.as_ptr());
				if pointer.is_null() {
					let e = format!("dlsym {} {entry}: {}", dylib.display(), loader_error());
					dlclose(handle);
					return Err(e);
				}
				Ok(Self {
					handle,
					entry: pointer,
					directory: directory.clone(),
				})
			}
		})();
		if result.is_err() {
			let _ = std::fs::remove_dir_all(directory);
		}
		result
	}
```

```sh
# check
cargo test --quiet --test runtime_inputs native_errors
```

## 17. compile 失敗と run の入力エラーを区別する。
<!-- todo:02-017 -->

`tests/runtime_inputs.rs` の `native_errors` で、この経路の値または拒否条件を確認する。

`src/inputs.rs` の `pub fn run(`の定義を次の全体で置き換えます。

```rust
// file="src/inputs.rs" mode="replace" item="pub fn run("
	pub fn run(&mut self, inputs: &[f32]) -> Result<f32> {
		if inputs.len() != self.slots {
			return Err("scalar input count mismatch".into());
		}
		let words = scalar_words(inputs);
		let mut output = 0u64;
		let buffers = [(&mut output as *mut u64).cast::<c_void>()];
		type Entry = unsafe extern "C" fn(*const *mut c_void, *const u64) -> i32;
		let entry: Entry = unsafe { std::mem::transmute(self.library.symbol()) };
		let status = unsafe { entry(buffers.as_ptr(), words.as_ptr()) };
		if status != 0 {
			return Err("scalar entry rejected inputs".into());
		}
		self.runs += 1;
		Ok(f32::from_bits(output as u32))
	}
```

```sh
# check
cargo test --quiet --test runtime_inputs native_errors
```

## 18. 所有物の破棄で library と一時ファイルを解放する。
<!-- todo:02-018 -->

```rust
// file=src/native.rs mode=append
impl Drop for NativeLibrary {
	fn drop(&mut self) {
		unsafe {
			dlclose(self.handle);
		}
		let _ = std::fs::remove_dir_all(&self.directory);
	}
}
```

```rust
// file=tests/runtime_inputs.rs mode=append
#[test]
fn native_lifetime() -> Result<()> {
	let source = "int chibi_entry(void*const* b,const unsigned long long* s){return 0;}";
	let library = unsafe { NativeLibrary::compile(source, "chibi_entry") }?;
	let dir = library.directory().to_path_buf();
	assert!(dir.exists());
	drop(library);
	assert!(!dir.exists());
	Ok(())
}
```

`tests/runtime_inputs.rs` の `native_lifetime` で、この経路の値または拒否条件を確認する。

```sh
# check
cargo test --quiet --test runtime_inputs native_lifetime
```

## 19. 成功例で `20`、`12` を得る。
<!-- todo:02-019 -->

`tests/runtime_inputs.rs` の正常系と境界系をまとめて再実行する。

`tests/runtime_inputs.rs` の `fn same_library_two_runs(`の定義を次の全体で置き換えます。

```rust
// file="tests/runtime_inputs.rs" mode="replace" item="fn same_library_two_runs("
fn same_library_two_runs() -> Result<()> {
	let mut g = Graph::default();
	let a = g.node(Op::Param(0));
	let b = g.node(Op::Param(1));
	let c = g.node(Op::Param(2));
	let sum = g.node(Op::Add(a, b));
	let root = g.node(Op::Mul(sum, c));
	let mut plan = g.compile(root)?;
	assert!(plan.source.contains("scalars[0]"));
	assert_eq!(plan.run(&[2., 3., 4.])?, 20.);
	assert_eq!(plan.run(&[5., 1., 2.])?, 12.);
	assert_eq!(plan.runs, 2);
	assert!(plan.run(&[2., 3.]).is_err());
	assert_eq!(plan.runs, 2);
	assert!(plan.run(&[2., 3., 4., 5.]).is_err());
	let mut bad = Graph::default();
	let missing = bad.node(Op::Param(1));
	assert!(bad.compile(missing).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test runtime_inputs
```

## 20. compile の予定計数が `1`、run が `2` になる。
<!-- todo:02-020 -->

`tests/runtime_inputs.rs` の正常系と境界系をまとめて再実行する。

`tests/runtime_inputs.rs` の `fn same_library_two_runs(`の定義を次の全体で置き換えます。

```rust
// file="tests/runtime_inputs.rs" mode="replace" item="fn same_library_two_runs("
fn same_library_two_runs() -> Result<()> {
	let mut g = Graph::default();
	let a = g.node(Op::Param(0));
	let b = g.node(Op::Param(1));
	let c = g.node(Op::Param(2));
	let sum = g.node(Op::Add(a, b));
	let root = g.node(Op::Mul(sum, c));
	let mut plan = g.compile(root)?;
	assert!(plan.source.contains("scalars[0]"));
	assert_eq!(plan.run(&[2., 3., 4.])?, 20.);
	assert_eq!(plan.run(&[5., 1., 2.])?, 12.);
	assert_eq!(plan.runs, 2);
	assert!(plan.run(&[2., 3.]).is_err());
	assert_eq!(plan.runs, 2);
	assert!(plan.run(&[2., 3., 4., 5.]).is_err());
	let mut bad = Graph::default();
	let missing = bad.node(Op::Param(1));
	assert!(bad.compile(missing).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test runtime_inputs
```

## 21. 生成 C は実行時 scalar スロットを参照する。
<!-- todo:02-021 -->

`tests/runtime_inputs.rs` の正常系と境界系をまとめて再実行する。

`tests/runtime_inputs.rs` の `fn same_library_two_runs(`の定義を次の全体で置き換えます。

```rust
// file="tests/runtime_inputs.rs" mode="replace" item="fn same_library_two_runs("
fn same_library_two_runs() -> Result<()> {
	let mut g = Graph::default();
	let a = g.node(Op::Param(0));
	let b = g.node(Op::Param(1));
	let c = g.node(Op::Param(2));
	let sum = g.node(Op::Add(a, b));
	let root = g.node(Op::Mul(sum, c));
	let mut plan = g.compile(root)?;
	assert!(plan.source.contains("scalars[0]"));
	assert_eq!(plan.run(&[2., 3., 4.])?, 20.);
	assert_eq!(plan.run(&[5., 1., 2.])?, 12.);
	assert_eq!(plan.runs, 2);
	assert!(plan.run(&[2., 3.]).is_err());
	assert_eq!(plan.runs, 2);
	assert!(plan.run(&[2., 3., 4., 5.]).is_err());
	let mut bad = Graph::default();
	let missing = bad.node(Op::Param(1));
	assert!(bad.compile(missing).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test runtime_inputs
```

## 22. 入力不足では FFI を実行しない。
<!-- todo:02-022 -->

`tests/runtime_inputs.rs` の正常系と境界系をまとめて再実行する。

`tests/runtime_inputs.rs` の `fn same_library_two_runs(`の定義を次の全体で置き換えます。

```rust
// file="tests/runtime_inputs.rs" mode="replace" item="fn same_library_two_runs("
fn same_library_two_runs() -> Result<()> {
	let mut g = Graph::default();
	let a = g.node(Op::Param(0));
	let b = g.node(Op::Param(1));
	let c = g.node(Op::Param(2));
	let sum = g.node(Op::Add(a, b));
	let root = g.node(Op::Mul(sum, c));
	let mut plan = g.compile(root)?;
	assert!(plan.source.contains("scalars[0]"));
	assert_eq!(plan.run(&[2., 3., 4.])?, 20.);
	assert_eq!(plan.run(&[5., 1., 2.])?, 12.);
	assert_eq!(plan.runs, 2);
	assert!(plan.run(&[2., 3.]).is_err());
	assert_eq!(plan.runs, 2);
	assert!(plan.run(&[2., 3., 4., 5.]).is_err());
	let mut bad = Graph::default();
	let missing = bad.node(Op::Param(1));
	assert!(bad.compile(missing).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test runtime_inputs
```

## 23. 前章の定数例も成功する。
<!-- todo:02-023 -->

`tests/runtime_inputs.rs` の `constant_and_add` で、この経路の値または拒否条件を確認する。

`tests/runtime_inputs.rs` の `fn constant_and_add(`の定義を次の全体で置き換えます。

```rust
// file="tests/runtime_inputs.rs" mode="replace" item="fn constant_and_add("
fn constant_and_add() -> Result<()> {
	let mut g = Graph::default();
	let c = g.node(Op::Const(7.));
	assert_eq!(g.compile(c)?.run(&[])?, 7.);
	let a = g.node(Op::Param(0));
	let b = g.node(Op::Param(1));
	let sum = g.node(Op::Add(a, b));
	assert_eq!(g.compile(sum)?.run(&[2., 3.])?, 5.);
	Ok(())
}
```

```sh
# check
cargo test --quiet --test runtime_inputs constant_and_add
```

## 24. library の所有物を破棄した後に関数ポインタだけが公開されない。
<!-- todo:02-024 -->

`tests/runtime_inputs.rs` の `native_lifetime` で、この経路の値または拒否条件を確認する。

`src/native.rs` の `pub unsafe fn symbol(`の定義を次の全体で置き換えます。

```rust
// file="src/native.rs" mode="replace" item="pub unsafe fn symbol("
	pub unsafe fn symbol(&self) -> *mut c_void {
		self.entry
	}
```

```sh
# check
cargo test --quiet --test runtime_inputs native_lifetime
```

## 章末の累積確認

前章までの example と test も同じ checkout で再ビルドする。

```sh
# check
cargo test --quiet --all-targets
```

```sh
# check
cargo run --quiet --example scalar_inputs -- 2 3 4 --then 5 1 2 --dump-source
```
