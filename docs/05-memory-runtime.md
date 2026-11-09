# 05. バッファの LOAD と STORE を安全な順序で CPU 実行する

メモリ IR は storage identity と view を分ける。Stmt の列が効果の順序であり、LOAD はその位置の状態を読む。AFTER は新しい保存領域ではなく、その列の先行 STORE が完了した時点に対応する。読み取り値を再利用したい場合は別 storage に STORE して保存する。LOAD をアドレスだけで intern しない。Kernel の AST は次章の RANGE も同じ型で表す。有限の静的 loop では全アクセスを列挙して初期化と境界を検証し、1千万手の証明上限を超える場合は compile を拒否する。検証成功を実行の代わりにはしない。

`file=... mode=write` はそのファイルの全置換、`mode=append` は末尾への追記である。すべて前章でできたプロジェクトの root で行う。`sh check` は実際の検証コマンドであり、ソースファイルには保存しない。

## 1. BUFFER の束縛と ALLOC の固有 ID を追加する。
<!-- todo:05-001 -->

```rust file=src/native.rs mode=append
#[derive(Clone, Debug)]
pub struct KernelBody {
	pub output: usize,
	pub body: String,
	pub dependencies: Vec<usize>,
}
pub fn render_entry(helpers: &str, slots: &[usize], kernels: &[KernelBody]) -> String {
	let mut source = helpers.to_string();
	source.push_str("int chibi_entry(void*const* buffers,const uint64_t* s){\n");
	for id in slots {
		source.push_str(&format!("float *b{id}=(float*)buffers[{id}];\n"));
	}
	for kernel in kernels {
		source.push_str(&kernel.body)
	}
	source.push_str("return 0;\n}\n");
	source
}
```

```rust file=src/kernel.rs mode=write
use crate::{Result, native::NativeLibrary};
use std::{
	collections::{HashMap, HashSet},
	ffi::c_void,
};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct View {
	pub storage: usize,
	pub offset: usize,
	pub len: usize,
}
impl View {
	pub fn shrink(self, start: usize, end: usize) -> Result<Self> {
		if start > end || end > self.len {
			return Err("invalid memory SHRINK".into());
		}
		Ok(Self {
			storage: self.storage,
			offset: self
				.offset
				.checked_add(start)
				.ok_or("view offset overflow")?,
			len: end - start,
		})
	}
}
#[derive(Clone, Debug)]
pub enum Index {
	Const(i32),
	Axis(usize),
	Add(Box<Index>, Box<Index>),
	Mul(Box<Index>, Box<Index>),
}
impl Index {
	pub fn add(self, b: Self) -> Self {
		Self::Add(Box::new(self), Box::new(b))
	}
	pub fn mul(self, b: Self) -> Self {
		Self::Mul(Box::new(self), Box::new(b))
	}
	fn value(&self, axes: &HashMap<usize, i32>) -> Result<i32> {
		match self {
			Self::Const(x) => Ok(*x),
			Self::Axis(id) => axes
				.get(id)
				.copied()
				.ok_or("free or closed RANGE axis".into()),
			Self::Add(a, b) => a
				.value(axes)?
				.checked_add(b.value(axes)?)
				.ok_or("index addition overflow".into()),
			Self::Mul(a, b) => a
				.value(axes)?
				.checked_mul(b.value(axes)?)
				.ok_or("index multiplication overflow".into()),
		}
	}
	fn source(&self) -> String {
		match self {
			Self::Const(x) => x.to_string(),
			Self::Axis(id) => format!("i{id}"),
			Self::Add(a, b) => format!("({}+{})", a.source(), b.source()),
			Self::Mul(a, b) => format!("({}*{})", a.source(), b.source()),
		}
	}
}
#[derive(Clone, Debug)]
pub enum Expr {
	Const(f32),
	Load(View, Index),
	Add(Box<Expr>, Box<Expr>),
	Mul(Box<Expr>, Box<Expr>),
}
impl Expr {
	pub fn add(self, b: Self) -> Self {
		Self::Add(Box::new(self), Box::new(b))
	}
	pub fn mul(self, b: Self) -> Self {
		Self::Mul(Box::new(self), Box::new(b))
	}
	fn source(&self) -> String {
		match self {
			Self::Const(v) => format!("chibi_bits({}u)", v.to_bits()),
			Self::Load(view, index) => {
				format!("b{}[{}+({})]", view.storage, view.offset, index.source())
			}
			Self::Add(a, b) => format!("({}+{})", a.source(), b.source()),
			Self::Mul(a, b) => format!("({}*{})", a.source(), b.source()),
		}
	}
}
#[derive(Clone, Debug)]
pub enum Stmt {
	Store {
		view: View,
		index: Index,
		value: Expr,
	},
	Range {
		axis: usize,
		len: usize,
		body: Vec<Stmt>,
	},
}
#[derive(Clone, Debug)]
pub struct Storage {
	pub len: usize,
	pub input: Option<usize>,
	pub register: bool,
}
#[derive(Clone, Debug, Default)]
pub struct Kernel {
	pub storage: Vec<Storage>,
	pub body: Vec<Stmt>,
	pub outputs: Vec<View>,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Accesses {
	pub read: HashSet<usize>,
	pub write: HashSet<usize>,
}
impl Kernel {
	pub fn input(&mut self, slot: usize, len: usize) -> Result<View> {
		self.allocate(len, Some(slot), false)
	}
	pub fn alloc(&mut self, len: usize) -> Result<View> {
		self.allocate(len, None, false)
	}
	pub fn reg(&mut self) -> Result<View> {
		self.allocate(1, None, true)
	}
	fn allocate(&mut self, len: usize, input: Option<usize>, register: bool) -> Result<View> {
		if len > i32::MAX as usize
			|| len
				.checked_mul(4)
				.filter(|&n| n <= isize::MAX as usize)
				.is_none()
		{
			return Err("storage size overflow".into());
		}
		let storage = self.storage.len();
		self.storage.push(Storage {
			len,
			input,
			register,
		});
		Ok(View {
			storage,
			offset: 0,
			len,
		})
	}
	fn address(&self, view: View, index: &Index, axes: &HashMap<usize, i32>) -> Result<usize> {
		let storage = self.storage.get(view.storage).ok_or("unknown storage")?;
		if view
			.offset
			.checked_add(view.len)
			.filter(|&n| n <= storage.len)
			.is_none()
		{
			return Err("view exceeds storage".into());
		}
		let i = usize::try_from(index.value(axes)?).map_err(|_| "negative INDEX")?;
		if i >= view.len {
			return Err("INDEX exceeds view".into());
		}
		view.offset.checked_add(i).ok_or("address overflow".into())
	}
	fn evaluate(
		&self,
		expr: &Expr,
		axes: &HashMap<usize, i32>,
		memory: &[Vec<f32>],
		initialized: &[Vec<bool>],
		access: &mut Accesses,
	) -> Result<f32> {
		match expr {
			Expr::Const(v) => Ok(*v),
			Expr::Load(view, index) => {
				let i = self.address(*view, index, axes)?;
				if !initialized[view.storage][i] {
					return Err("LOAD before initialization/AFTER".into());
				}
				access.read.insert(view.storage);
				Ok(memory[view.storage][i])
			}
			Expr::Add(a, b) => Ok(self.evaluate(a, axes, memory, initialized, access)?
				+ self.evaluate(b, axes, memory, initialized, access)?),
			Expr::Mul(a, b) => Ok(self.evaluate(a, axes, memory, initialized, access)?
				* self.evaluate(b, axes, memory, initialized, access)?),
		}
	}
	fn walk(
		&self,
		body: &[Stmt],
		axes: &mut HashMap<usize, i32>,
		memory: &mut [Vec<f32>],
		initialized: &mut [Vec<bool>],
		access: &mut Accesses,
		budget: &mut usize,
	) -> Result<()> {
		for statement in body {
			*budget = budget
				.checked_sub(1)
				.ok_or("verification work exceeds finite proof budget")?;
			match statement {
				Stmt::Store { view, index, value } => {
					if self
						.storage
						.get(view.storage)
						.ok_or("STORE unknown storage")?
						.input
						.is_some()
					{
						return Err("STORE to readonly input".into());
					}
					let address = self.address(*view, index, axes)?;
					let value = self.evaluate(value, axes, memory, initialized, access)?;
					memory[view.storage][address] = value;
					initialized[view.storage][address] = true;
					access.write.insert(view.storage);
				}
				Stmt::Range { axis, len, body } => {
					if axes.contains_key(axis) || *len > i32::MAX as usize {
						return Err("duplicate RANGE or i32 overflow".into());
					}
					for i in 0..*len {
						axes.insert(*axis, i as i32);
						self.walk(body, axes, memory, initialized, access, budget)?;
					}
					axes.remove(axis);
				}
			}
		}
		Ok(())
	}
	fn scope_check(
		body: &[Stmt],
		open: &mut HashSet<usize>,
		seen: &mut HashSet<usize>,
	) -> Result<()> {
		fn index(x: &Index, open: &HashSet<usize>) -> Result<()> {
			match x {
				Index::Axis(a) if !open.contains(a) => Err("free or closed RANGE axis".into()),
				Index::Add(a, b) | Index::Mul(a, b) => {
					index(a, open)?;
					index(b, open)
				}
				_ => Ok(()),
			}
		}
		fn expr(x: &Expr, open: &HashSet<usize>) -> Result<()> {
			match x {
				Expr::Load(_, i) => index(i, open),
				Expr::Add(a, b) | Expr::Mul(a, b) => {
					expr(a, open)?;
					expr(b, open)
				}
				_ => Ok(()),
			}
		}
		for s in body {
			match s {
				Stmt::Store {
					index: i, value, ..
				} => {
					index(i, open)?;
					expr(value, open)?
				}
				Stmt::Range { axis, len, body } => {
					if *len > i32::MAX as usize || !seen.insert(*axis) {
						return Err("RANGE reused or extent overflow".into());
					}
					open.insert(*axis);
					Self::scope_check(body, open, seen)?;
					open.remove(axis);
				}
			}
		}
		Ok(())
	}
	pub fn interpret(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
		let (memory, _) = self.execute(inputs)?;
		Ok(self
			.outputs
			.iter()
			.map(|v| memory[v.storage][v.offset..v.offset + v.len].to_vec())
			.collect())
	}
	fn execute(&self, inputs: &[Vec<f32>]) -> Result<(Vec<Vec<f32>>, Accesses)> {
		Self::scope_check(&self.body, &mut HashSet::new(), &mut HashSet::new())?;
		let mut slots = HashSet::new();
		let mut memory = vec![];
		let mut initialized = vec![];
		for storage in &self.storage {
			if let Some(slot) = storage.input {
				slots.insert(slot);
				let data = inputs.get(slot).ok_or("missing memory input")?;
				if data.len() != storage.len {
					return Err("memory input length mismatch".into());
				}
				memory.push(data.clone());
				initialized.push(vec![true; storage.len]);
			} else {
				memory.push(vec![0.; storage.len]);
				initialized.push(vec![false; storage.len]);
			}
		}
		if slots.len() != inputs.len() {
			return Err("extra memory inputs".into());
		}
		let mut access = Accesses::default();
		self.walk(
			&self.body,
			&mut HashMap::new(),
			&mut memory,
			&mut initialized,
			&mut access,
			&mut 10_000_000,
		)?;
		for view in &self.outputs {
			if view
				.offset
				.checked_add(view.len)
				.filter(|&end| end <= self.storage.get(view.storage).map(|s| s.len).unwrap_or(0))
				.is_none()
			{
				return Err("invalid output view".into());
			}
			if initialized[view.storage][view.offset..view.offset + view.len]
				.iter()
				.any(|&v| !v)
			{
				return Err("output not fully initialized".into());
			}
		}
		Ok((memory, access))
	}
	pub fn verify(&self) -> Result<Accesses> {
		let count = self
			.storage
			.iter()
			.filter_map(|s| s.input)
			.max()
			.map(|n| n + 1)
			.unwrap_or(0);
		let mut inputs = vec![vec![]; count];
		for s in &self.storage {
			if let Some(slot) = s.input {
				if !inputs[slot].is_empty() && inputs[slot].len() != s.len {
					return Err("input alias lengths differ".into());
				}
				inputs[slot] = vec![0.; s.len]
			}
		}
		Ok(self.execute(&inputs)?.1)
	}
	pub fn lower(&self) -> Result<crate::native::KernelBody> {
		self.verify()?;
		fn emit(body: &[Stmt], out: &mut String) {
			for s in body {
				match s {
					Stmt::Store { view, index, value } => out.push_str(&format!(
						"b{}[{}+({})]={};\n",
						view.storage,
						view.offset,
						index.source(),
						value.source()
					)),
					Stmt::Range { axis, len, body } => {
						out.push_str(&format!(
							"for(int32_t i{axis}=0;i{axis}<{len};++i{axis}){{\n"
						));
						emit(body, out);
						out.push_str("}\n");
					}
				}
			}
		}
		let mut body = String::new();
		emit(&self.body, &mut body);
		let accesses = self.verify()?;
		let kernel = crate::native::KernelBody {
			output: self.outputs.first().map(|v| v.storage).unwrap_or(0),
			body,
			dependencies: accesses.read.into_iter().collect(),
		};
		Ok(kernel)
	}
	pub fn render(&self) -> Result<String> {
		let kernel = self.lower()?;
		Ok(crate::native::render_entry(
			crate::scalar::SCALAR_HELPERS,
			&(0..self.storage.len()).collect::<Vec<_>>(),
			&[kernel],
		))
	}
	pub fn compile(&self) -> Result<CompiledKernel> {
		let source = self.render()?;
		let library = unsafe { NativeLibrary::compile(&source, "chibi_entry") }?;
		Ok(CompiledKernel {
			kernel: self.clone(),
			source,
			library,
		})
	}
}
pub struct CompiledKernel {
	kernel: Kernel,
	pub source: String,
	library: NativeLibrary,
}
impl CompiledKernel {
	pub fn run(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
		let mut slots = HashSet::new();
		let mut storage = self
			.kernel
			.storage
			.iter()
			.map(|s| vec![0.; s.len])
			.collect::<Vec<_>>();
		let mut pointers = storage
			.iter_mut()
			.map(|b| b.as_mut_ptr().cast::<c_void>())
			.collect::<Vec<_>>();
		for (id, s) in self.kernel.storage.iter().enumerate() {
			if let Some(slot) = s.input {
				slots.insert(slot);
				let input = inputs.get(slot).ok_or("missing kernel input")?;
				if input.len() != s.len {
					return Err("kernel input length mismatch".into());
				}
				pointers[id] = input.as_ptr().cast_mut().cast();
			}
		}
		if slots.len() != inputs.len() {
			return Err("extra kernel input".into());
		}
		type Entry = unsafe extern "C" fn(*const *mut c_void, *const u64) -> i32;
		let entry: Entry = unsafe { std::mem::transmute(self.library.symbol()) };
		let status = unsafe { entry(pointers.as_ptr(), std::ptr::null()) };
		if status != 0 {
			return Err("memory kernel failed".into());
		}
		Ok(self
			.kernel
			.outputs
			.iter()
			.map(|v| {
				if let Some(slot) = self.kernel.storage[v.storage].input {
					inputs[slot][v.offset..v.offset + v.len].to_vec()
				} else {
					storage[v.storage][v.offset..v.offset + v.len].to_vec()
				}
			})
			.collect())
	}
}
pub fn store(view: View, index: Index, value: Expr) -> Stmt {
	Stmt::Store { view, index, value }
}
pub fn range(axis: usize, len: usize, body: Vec<Stmt>) -> Stmt {
	Stmt::Range { axis, len, body }
}
impl Kernel {
	pub fn inline_effect(&mut self, callee: &Kernel, args: &[View]) -> Result<Vec<View>> {
		callee.verify()?;
		let mut mapping = vec![];
		for storage in &callee.storage {
			let view = if let Some(slot) = storage.input {
				let view = *args.get(slot).ok_or("effect CALL missing argument")?;
				if view.len != storage.len {
					return Err("effect CALL length mismatch".into());
				}
				view
			} else {
				self.allocate(storage.len, None, storage.register)?
			};
			mapping.push(view);
		}
		fn max_axis(body: &[Stmt]) -> usize {
			body.iter()
				.map(|s| match s {
					Stmt::Range { axis, body, .. } => (*axis + 1).max(max_axis(body)),
					_ => 0,
				})
				.max()
				.unwrap_or(0)
		}
		let axis_offset = max_axis(&self.body);
		fn index(x: &Index, offset: usize) -> Index {
			match x {
				Index::Const(v) => Index::Const(*v),
				Index::Axis(a) => Index::Axis(a + offset),
				Index::Add(a, b) => index(a, offset).add(index(b, offset)),
				Index::Mul(a, b) => index(a, offset).mul(index(b, offset)),
			}
		}
		fn view(v: View, map: &[View]) -> Result<View> {
			map[v.storage].shrink(
				v.offset,
				v.offset.checked_add(v.len).ok_or("effect view overflow")?,
			)
		}
		fn expr(x: &Expr, map: &[View], offset: usize) -> Result<Expr> {
			Ok(match x {
				Expr::Const(v) => Expr::Const(*v),
				Expr::Load(v, i) => Expr::Load(view(*v, map)?, index(i, offset)),
				Expr::Add(a, b) => expr(a, map, offset)?.add(expr(b, map, offset)?),
				Expr::Mul(a, b) => expr(a, map, offset)?.mul(expr(b, map, offset)?),
			})
		}
		fn statements(body: &[Stmt], map: &[View], offset: usize) -> Result<Vec<Stmt>> {
			body.iter()
				.map(|s| {
					Ok(match s {
						Stmt::Store {
							view: v,
							index: i,
							value,
						} => store(view(*v, map)?, index(i, offset), expr(value, map, offset)?),
						Stmt::Range { axis, len, body } => {
							range(axis + offset, *len, statements(body, map, offset)?)
						}
					})
				})
				.collect()
		}
		self.body
			.extend(statements(&callee.body, &mapping, axis_offset)?);
		callee.outputs.iter().map(|&v| view(v, &mapping)).collect()
	}
}
```

```rust file=src/lib.rs mode=append
pub mod kernel;
```

```rust file=tests/memory_runtime.rs mode=write
use chinigrad::{
	Result,
	kernel::{Expr as E, Index as I, Kernel, range, store},
};

#[test]
fn memory_state_and_views() -> Result<()> {
	let mut k = Kernel::default();
	let x = k.input(0, 3)?;
	let out = k.alloc(3)?;
	let state = k.alloc(1)?;
	k.body = vec![
		store(state, I::Const(0), E::Const(1.)),
		store(out, I::Const(0), E::Load(state, I::Const(0))),
		store(state, I::Const(0), E::Const(2.)),
		store(out, I::Const(1), E::Load(state, I::Const(0))),
		store(out, I::Const(2), E::Load(x.shrink(1, 3)?, I::Const(0))),
	];
	k.outputs = vec![out];
	assert_eq!(x.storage, x.shrink(1, 3)?.storage);
	assert_eq!(
		k.compile()?.run(&[vec![10., 20., 30.]])?,
		vec![vec![1., 2., 20.]]
	);
	let access = k.verify()?;
	assert!(access.read.contains(&x.storage));
	assert!(access.write.contains(&out.storage));
	let bad = store(out, I::Const(0), E::Load(k.alloc(1)?, I::Const(0)));
	k.body.insert(0, bad);
	assert!(k.compile().is_err());
	Ok(())
}
#[test]
fn fixed_accesses_and_vector_add() -> Result<()> {
	let mut k = Kernel::default();
	let x = k.input(0, 3)?;
	let out = k.alloc(1)?;
	k.body = vec![store(
		out,
		I::Const(0),
		E::Load(x, I::Const(2)).add(E::Const(1.)),
	)];
	k.outputs = vec![out];
	assert_eq!(k.compile()?.run(&[vec![10., 20., 30.]])?, vec![vec![31.]]);
	let mut k = Kernel::default();
	let x = k.input(0, 3)?;
	let y = k.input(1, 3)?;
	let out = k.alloc(3)?;
	let other = k.alloc(3)?;
	assert_ne!(out.storage, other.storage);
	k.body = (0..3)
		.map(|i| {
			store(
				out,
				I::Const(i),
				E::Load(x, I::Const(i)).add(E::Load(y, I::Const(i))),
			)
		})
		.collect();
	k.outputs = vec![out];
	assert_eq!(
		k.compile()?.run(&[vec![1., 2., 3.], vec![10., 20., 30.]])?,
		vec![vec![11., 22., 33.]]
	);
	assert!(
		k.compile()?
			.run(&[vec![1., 2.], vec![10., 20., 30.]])
			.is_err()
	);
	Ok(())
}
```

`tests/memory_runtime.rs` の `fixed_accesses_and_vector_add` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test memory_runtime fixed_accesses_and_vector_add
```

## 2. メモリ INDEX、LOAD、STORE を C と検証用 interpreter に加える。
<!-- todo:05-002 -->

`tests/memory_runtime.rs` の正常系と境界系をまとめて再実行する。

`src/kernel.rs` の `pub fn render(`の定義を次の全体で置き換えます。

```rust file="src/kernel.rs" mode="replace" item="pub fn render("
	pub fn render(&self) -> Result<String> {
		let kernel = self.lower()?;
		Ok(crate::native::render_entry(
			crate::scalar::SCALAR_HELPERS,
			&(0..self.storage.len()).collect::<Vec<_>>(),
			&[kernel],
		))
	}
```

`src/kernel.rs` の `fn evaluate(`の定義を次の全体で置き換えます。

```rust file="src/kernel.rs" mode="replace" item="fn evaluate("
	fn evaluate(
		&self,
		expr: &Expr,
		axes: &HashMap<usize, i32>,
		memory: &[Vec<f32>],
		initialized: &[Vec<bool>],
		access: &mut Accesses,
	) -> Result<f32> {
		match expr {
			Expr::Const(v) => Ok(*v),
			Expr::Load(view, index) => {
				let i = self.address(*view, index, axes)?;
				if !initialized[view.storage][i] {
					return Err("LOAD before initialization/AFTER".into());
				}
				access.read.insert(view.storage);
				Ok(memory[view.storage][i])
			}
			Expr::Add(a, b) => Ok(self.evaluate(a, axes, memory, initialized, access)?
				+ self.evaluate(b, axes, memory, initialized, access)?),
			Expr::Mul(a, b) => Ok(self.evaluate(a, axes, memory, initialized, access)?
				* self.evaluate(b, axes, memory, initialized, access)?),
		}
	}
```

```sh check
cargo test --quiet --test memory_runtime
```

## 3. buffer の PARAM を固定 ABI のポインタ配列へ接続する。
<!-- todo:05-003 -->

`tests/memory_runtime.rs` の正常系と境界系をまとめて再実行する。

`src/kernel.rs` の `pub fn run(`の定義を次の全体で置き換えます。

```rust file="src/kernel.rs" mode="replace" item="pub fn run("
	pub fn run(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
		let mut slots = HashSet::new();
		let mut storage = self
			.kernel
			.storage
			.iter()
			.map(|s| vec![0.; s.len])
			.collect::<Vec<_>>();
		let mut pointers = storage
			.iter_mut()
			.map(|b| b.as_mut_ptr().cast::<c_void>())
			.collect::<Vec<_>>();
		for (id, s) in self.kernel.storage.iter().enumerate() {
			if let Some(slot) = s.input {
				slots.insert(slot);
				let input = inputs.get(slot).ok_or("missing kernel input")?;
				if input.len() != s.len {
					return Err("kernel input length mismatch".into());
				}
				pointers[id] = input.as_ptr().cast_mut().cast();
			}
		}
		if slots.len() != inputs.len() {
			return Err("extra kernel input".into());
		}
		type Entry = unsafe extern "C" fn(*const *mut c_void, *const u64) -> i32;
		let entry: Entry = unsafe { std::mem::transmute(self.library.symbol()) };
		let status = unsafe { entry(pointers.as_ptr(), std::ptr::null()) };
		if status != 0 {
			return Err("memory kernel failed".into());
		}
		Ok(self
			.kernel
			.outputs
			.iter()
			.map(|v| {
				if let Some(slot) = self.kernel.storage[v.storage].input {
					inputs[slot][v.offset..v.offset + v.len].to_vec()
				} else {
					storage[v.storage][v.offset..v.offset + v.len].to_vec()
				}
			})
			.collect())
	}
```

```sh check
cargo test --quiet --test memory_runtime
```

## 4. 入力の長さと dtype を呼出し前に照合する。
<!-- todo:05-004 -->

`tests/memory_runtime.rs` の `fixed_accesses_and_vector_add` で、この経路の値または拒否条件を確認する。

`src/kernel.rs` の `pub fn run(`の定義を次の全体で置き換えます。

```rust file="src/kernel.rs" mode="replace" item="pub fn run("
	pub fn run(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
		let mut slots = HashSet::new();
		let mut storage = self
			.kernel
			.storage
			.iter()
			.map(|s| vec![0.; s.len])
			.collect::<Vec<_>>();
		let mut pointers = storage
			.iter_mut()
			.map(|b| b.as_mut_ptr().cast::<c_void>())
			.collect::<Vec<_>>();
		for (id, s) in self.kernel.storage.iter().enumerate() {
			if let Some(slot) = s.input {
				slots.insert(slot);
				let input = inputs.get(slot).ok_or("missing kernel input")?;
				if input.len() != s.len {
					return Err("kernel input length mismatch".into());
				}
				pointers[id] = input.as_ptr().cast_mut().cast();
			}
		}
		if slots.len() != inputs.len() {
			return Err("extra kernel input".into());
		}
		type Entry = unsafe extern "C" fn(*const *mut c_void, *const u64) -> i32;
		let entry: Entry = unsafe { std::mem::transmute(self.library.symbol()) };
		let status = unsafe { entry(pointers.as_ptr(), std::ptr::null()) };
		if status != 0 {
			return Err("memory kernel failed".into());
		}
		Ok(self
			.kernel
			.outputs
			.iter()
			.map(|v| {
				if let Some(slot) = self.kernel.storage[v.storage].input {
					inputs[slot][v.offset..v.offset + v.len].to_vec()
				} else {
					storage[v.storage][v.offset..v.offset + v.len].to_vec()
				}
			})
			.collect())
	}
```

```sh check
cargo test --quiet --test memory_runtime fixed_accesses_and_vector_add
```

## 5. `[10,20,30]` から生成コードで `31` を得る。
<!-- todo:05-005 -->

`tests/memory_runtime.rs` の `fixed_accesses_and_vector_add` で、この経路の値または拒否条件を確認する。

`tests/memory_runtime.rs` の `fn fixed_accesses_and_vector_add(`の定義を次の全体で置き換えます。

```rust file="tests/memory_runtime.rs" mode="replace" item="fn fixed_accesses_and_vector_add("
fn fixed_accesses_and_vector_add() -> Result<()> {
	let mut k = Kernel::default();
	let x = k.input(0, 3)?;
	let out = k.alloc(1)?;
	k.body = vec![store(
		out,
		I::Const(0),
		E::Load(x, I::Const(2)).add(E::Const(1.)),
	)];
	k.outputs = vec![out];
	assert_eq!(k.compile()?.run(&[vec![10., 20., 30.]])?, vec![vec![31.]]);
	let mut k = Kernel::default();
	let x = k.input(0, 3)?;
	let y = k.input(1, 3)?;
	let out = k.alloc(3)?;
	let other = k.alloc(3)?;
	assert_ne!(out.storage, other.storage);
	k.body = (0..3)
		.map(|i| {
			store(
				out,
				I::Const(i),
				E::Load(x, I::Const(i)).add(E::Load(y, I::Const(i))),
			)
		})
		.collect();
	k.outputs = vec![out];
	assert_eq!(
		k.compile()?.run(&[vec![1., 2., 3.], vec![10., 20., 30.]])?,
		vec![vec![11., 22., 33.]]
	);
	assert!(
		k.compile()?
			.run(&[vec![1., 2.], vec![10., 20., 30.]])
			.is_err()
	);
	Ok(())
}
```

```sh check
cargo test --quiet --test memory_runtime fixed_accesses_and_vector_add
```

## 6. 効果を依存順に実行し、到達可能な STORE を消さない。
<!-- todo:05-006 -->

`tests/memory_runtime.rs` の正常系と境界系をまとめて再実行する。

`src/kernel.rs` の `fn walk(`の定義を次の全体で置き換えます。

```rust file="src/kernel.rs" mode="replace" item="fn walk("
	fn walk(
		&self,
		body: &[Stmt],
		axes: &mut HashMap<usize, i32>,
		memory: &mut [Vec<f32>],
		initialized: &mut [Vec<bool>],
		access: &mut Accesses,
		budget: &mut usize,
	) -> Result<()> {
		for statement in body {
			*budget = budget
				.checked_sub(1)
				.ok_or("verification work exceeds finite proof budget")?;
			match statement {
				Stmt::Store { view, index, value } => {
					if self
						.storage
						.get(view.storage)
						.ok_or("STORE unknown storage")?
						.input
						.is_some()
					{
						return Err("STORE to readonly input".into());
					}
					let address = self.address(*view, index, axes)?;
					let value = self.evaluate(value, axes, memory, initialized, access)?;
					memory[view.storage][address] = value;
					initialized[view.storage][address] = true;
					access.write.insert(view.storage);
				}
				Stmt::Range { axis, len, body } => {
					if axes.contains_key(axis) || *len > i32::MAX as usize {
						return Err("duplicate RANGE or i32 overflow".into());
					}
					for i in 0..*len {
						axes.insert(*axis, i as i32);
						self.walk(body, axes, memory, initialized, access, budget)?;
					}
					axes.remove(axis);
				}
			}
		}
		Ok(())
	}
```

```sh check
cargo test --quiet --test memory_runtime
```

## 7. AFTER の返す storage identity と状態を区別する。
<!-- todo:05-007 -->

`tests/memory_runtime.rs` の正常系と境界系をまとめて再実行する。

`src/kernel.rs` の `fn walk(`の定義を次の全体で置き換えます。

```rust file="src/kernel.rs" mode="replace" item="fn walk("
	fn walk(
		&self,
		body: &[Stmt],
		axes: &mut HashMap<usize, i32>,
		memory: &mut [Vec<f32>],
		initialized: &mut [Vec<bool>],
		access: &mut Accesses,
		budget: &mut usize,
	) -> Result<()> {
		for statement in body {
			*budget = budget
				.checked_sub(1)
				.ok_or("verification work exceeds finite proof budget")?;
			match statement {
				Stmt::Store { view, index, value } => {
					if self
						.storage
						.get(view.storage)
						.ok_or("STORE unknown storage")?
						.input
						.is_some()
					{
						return Err("STORE to readonly input".into());
					}
					let address = self.address(*view, index, axes)?;
					let value = self.evaluate(value, axes, memory, initialized, access)?;
					memory[view.storage][address] = value;
					initialized[view.storage][address] = true;
					access.write.insert(view.storage);
				}
				Stmt::Range { axis, len, body } => {
					if axes.contains_key(axis) || *len > i32::MAX as usize {
						return Err("duplicate RANGE or i32 overflow".into());
					}
					for i in 0..*len {
						axes.insert(*axis, i as i32);
						self.walk(body, axes, memory, initialized, access, budget)?;
					}
					axes.remove(axis);
				}
			}
		}
		Ok(())
	}
```

```sh check
cargo test --quiet --test memory_runtime
```

## 8. 同じ位置へ `1`、次に `2` を書き、その後に読んで `2` を返す。
<!-- todo:05-008 -->

`tests/memory_runtime.rs` の正常系と境界系をまとめて再実行する。

`tests/memory_runtime.rs` の `fn memory_state_and_views(`の定義を次の全体で置き換えます。

```rust file="tests/memory_runtime.rs" mode="replace" item="fn memory_state_and_views("
fn memory_state_and_views() -> Result<()> {
	let mut k = Kernel::default();
	let x = k.input(0, 3)?;
	let out = k.alloc(3)?;
	let state = k.alloc(1)?;
	k.body = vec![
		store(state, I::Const(0), E::Const(1.)),
		store(out, I::Const(0), E::Load(state, I::Const(0))),
		store(state, I::Const(0), E::Const(2.)),
		store(out, I::Const(1), E::Load(state, I::Const(0))),
		store(out, I::Const(2), E::Load(x.shrink(1, 3)?, I::Const(0))),
	];
	k.outputs = vec![out];
	assert_eq!(x.storage, x.shrink(1, 3)?.storage);
	assert_eq!(
		k.compile()?.run(&[vec![10., 20., 30.]])?,
		vec![vec![1., 2., 20.]]
	);
	let access = k.verify()?;
	assert!(access.read.contains(&x.storage));
	assert!(access.write.contains(&out.storage));
	let bad = store(out, I::Const(0), E::Load(k.alloc(1)?, I::Const(0)));
	k.body.insert(0, bad);
	assert!(k.compile().is_err());
	Ok(())
}
```

```sh check
cargo test --quiet --test memory_runtime
```

## 9. LOAD の共有キーへ状態を含め、異なる時点の読出しを共有しない。
<!-- todo:05-009 -->

`tests/memory_runtime.rs` の正常系と境界系をまとめて再実行する。

`src/kernel.rs` の `fn walk(`の定義を次の全体で置き換えます。

```rust file="src/kernel.rs" mode="replace" item="fn walk("
	fn walk(
		&self,
		body: &[Stmt],
		axes: &mut HashMap<usize, i32>,
		memory: &mut [Vec<f32>],
		initialized: &mut [Vec<bool>],
		access: &mut Accesses,
		budget: &mut usize,
	) -> Result<()> {
		for statement in body {
			*budget = budget
				.checked_sub(1)
				.ok_or("verification work exceeds finite proof budget")?;
			match statement {
				Stmt::Store { view, index, value } => {
					if self
						.storage
						.get(view.storage)
						.ok_or("STORE unknown storage")?
						.input
						.is_some()
					{
						return Err("STORE to readonly input".into());
					}
					let address = self.address(*view, index, axes)?;
					let value = self.evaluate(value, axes, memory, initialized, access)?;
					memory[view.storage][address] = value;
					initialized[view.storage][address] = true;
					access.write.insert(view.storage);
				}
				Stmt::Range { axis, len, body } => {
					if axes.contains_key(axis) || *len > i32::MAX as usize {
						return Err("duplicate RANGE or i32 overflow".into());
					}
					for i in 0..*len {
						axes.insert(*axis, i as i32);
						self.walk(body, axes, memory, initialized, access, budget)?;
					}
					axes.remove(axis);
				}
			}
		}
		Ok(())
	}
```

```sh check
cargo test --quiet --test memory_runtime
```

## 10. 複数の効果を持つ CALL の出力を AFTER から読めるようにする。
<!-- todo:05-010 -->

`tests/memory_runtime.rs` の正常系と境界系をまとめて再実行する。

`src/kernel.rs` の `pub fn inline_effect(`の定義を次の全体で置き換えます。

```rust file="src/kernel.rs" mode="replace" item="pub fn inline_effect("
	pub fn inline_effect(&mut self, callee: &Kernel, args: &[View]) -> Result<Vec<View>> {
		callee.verify()?;
		let mut mapping = vec![];
		for storage in &callee.storage {
			let view = if let Some(slot) = storage.input {
				let view = *args.get(slot).ok_or("effect CALL missing argument")?;
				if view.len != storage.len {
					return Err("effect CALL length mismatch".into());
				}
				view
			} else {
				self.allocate(storage.len, None, storage.register)?
			};
			mapping.push(view);
		}
		fn max_axis(body: &[Stmt]) -> usize {
			body.iter()
				.map(|s| match s {
					Stmt::Range { axis, body, .. } => (*axis + 1).max(max_axis(body)),
					_ => 0,
				})
				.max()
				.unwrap_or(0)
		}
		let axis_offset = max_axis(&self.body);
		fn index(x: &Index, offset: usize) -> Index {
			match x {
				Index::Const(v) => Index::Const(*v),
				Index::Axis(a) => Index::Axis(a + offset),
				Index::Add(a, b) => index(a, offset).add(index(b, offset)),
				Index::Mul(a, b) => index(a, offset).mul(index(b, offset)),
			}
		}
		fn view(v: View, map: &[View]) -> Result<View> {
			map[v.storage].shrink(
				v.offset,
				v.offset.checked_add(v.len).ok_or("effect view overflow")?,
			)
		}
		fn expr(x: &Expr, map: &[View], offset: usize) -> Result<Expr> {
			Ok(match x {
				Expr::Const(v) => Expr::Const(*v),
				Expr::Load(v, i) => Expr::Load(view(*v, map)?, index(i, offset)),
				Expr::Add(a, b) => expr(a, map, offset)?.add(expr(b, map, offset)?),
				Expr::Mul(a, b) => expr(a, map, offset)?.mul(expr(b, map, offset)?),
			})
		}
		fn statements(body: &[Stmt], map: &[View], offset: usize) -> Result<Vec<Stmt>> {
			body.iter()
				.map(|s| {
					Ok(match s {
						Stmt::Store {
							view: v,
							index: i,
							value,
						} => store(view(*v, map)?, index(i, offset), expr(value, map, offset)?),
						Stmt::Range { axis, len, body } => {
							range(axis + offset, *len, statements(body, map, offset)?)
						}
					})
				})
				.collect()
		}
		self.body
			.extend(statements(&callee.body, &mapping, axis_offset)?);
		callee.outputs.iter().map(|&v| view(v, &mapping)).collect()
	}
```

```sh check
cargo test --quiet --test memory_runtime
```

## 11. `SHRINK(x,1,3)` と base offset を導出する。
<!-- todo:05-011 -->

`tests/memory_runtime.rs` の正常系と境界系をまとめて再実行する。

`src/kernel.rs` の `pub fn shrink(`の定義を次の全体で置き換えます。

```rust file="src/kernel.rs" mode="replace" item="pub fn shrink("
	pub fn shrink(self, start: usize, end: usize) -> Result<Self> {
		if start > end || end > self.len {
			return Err("invalid memory SHRINK".into());
		}
		Ok(Self {
			storage: self.storage,
			offset: self
				.offset
				.checked_add(start)
				.ok_or("view offset overflow")?,
			len: end - start,
		})
	}
```

```sh check
cargo test --quiet --test memory_runtime
```

## 12. view の INDEX 0 を元 buffer の INDEX 1 へ変換する。
<!-- todo:05-012 -->

`tests/memory_runtime.rs` の正常系と境界系をまとめて再実行する。

`src/kernel.rs` の `fn address(`の定義を次の全体で置き換えます。

```rust file="src/kernel.rs" mode="replace" item="fn address("
	fn address(&self, view: View, index: &Index, axes: &HashMap<usize, i32>) -> Result<usize> {
		let storage = self.storage.get(view.storage).ok_or("unknown storage")?;
		if view
			.offset
			.checked_add(view.len)
			.filter(|&n| n <= storage.len)
			.is_none()
		{
			return Err("view exceeds storage".into());
		}
		let i = usize::try_from(index.value(axes)?).map_err(|_| "negative INDEX")?;
		if i >= view.len {
			return Err("INDEX exceeds view".into());
		}
		view.offset.checked_add(i).ok_or("address overflow".into())
	}
```

```sh check
cargo test --quiet --test memory_runtime
```

## 13. storage_base と read、write の集合を dump に出す。
<!-- todo:05-013 -->

`tests/memory_runtime.rs` の正常系と境界系をまとめて再実行する。

`src/kernel.rs` の `pub fn verify(`の定義を次の全体で置き換えます。

```rust file="src/kernel.rs" mode="replace" item="pub fn verify("
	pub fn verify(&self) -> Result<Accesses> {
		let count = self
			.storage
			.iter()
			.filter_map(|s| s.input)
			.max()
			.map(|n| n + 1)
			.unwrap_or(0);
		let mut inputs = vec![vec![]; count];
		for s in &self.storage {
			if let Some(slot) = s.input {
				if !inputs[slot].is_empty() && inputs[slot].len() != s.len {
					return Err("input alias lengths differ".into());
				}
				inputs[slot] = vec![0.; s.len]
			}
		}
		Ok(self.execute(&inputs)?.1)
	}
```

```sh check
cargo test --quiet --test memory_runtime
```

## 14. 二入力 `[1,2,3]`, `[10,20,30]` を展開して `[11,22,33]` を作る。
<!-- todo:05-014 -->

`tests/memory_runtime.rs` の `fixed_accesses_and_vector_add` で、この経路の値または拒否条件を確認する。

`tests/memory_runtime.rs` の `fn fixed_accesses_and_vector_add(`の定義を次の全体で置き換えます。

```rust file="tests/memory_runtime.rs" mode="replace" item="fn fixed_accesses_and_vector_add("
fn fixed_accesses_and_vector_add() -> Result<()> {
	let mut k = Kernel::default();
	let x = k.input(0, 3)?;
	let out = k.alloc(1)?;
	k.body = vec![store(
		out,
		I::Const(0),
		E::Load(x, I::Const(2)).add(E::Const(1.)),
	)];
	k.outputs = vec![out];
	assert_eq!(k.compile()?.run(&[vec![10., 20., 30.]])?, vec![vec![31.]]);
	let mut k = Kernel::default();
	let x = k.input(0, 3)?;
	let y = k.input(1, 3)?;
	let out = k.alloc(3)?;
	let other = k.alloc(3)?;
	assert_ne!(out.storage, other.storage);
	k.body = (0..3)
		.map(|i| {
			store(
				out,
				I::Const(i),
				E::Load(x, I::Const(i)).add(E::Load(y, I::Const(i))),
			)
		})
		.collect();
	k.outputs = vec![out];
	assert_eq!(
		k.compile()?.run(&[vec![1., 2., 3.], vec![10., 20., 30.]])?,
		vec![vec![11., 22., 33.]]
	);
	assert!(
		k.compile()?
			.run(&[vec![1., 2.], vec![10., 20., 30.]])
			.is_err()
	);
	Ok(())
}
```

```sh check
cargo test --quiet --test memory_runtime fixed_accesses_and_vector_add
```

## 15. 同じ長さの二つの ALLOC が互いを上書きしないことを確かめる。
<!-- todo:05-015 -->

`tests/memory_runtime.rs` の `fixed_accesses_and_vector_add` で、この経路の値または拒否条件を確認する。

`tests/memory_runtime.rs` の `fn fixed_accesses_and_vector_add(`の定義を次の全体で置き換えます。

```rust file="tests/memory_runtime.rs" mode="replace" item="fn fixed_accesses_and_vector_add("
fn fixed_accesses_and_vector_add() -> Result<()> {
	let mut k = Kernel::default();
	let x = k.input(0, 3)?;
	let out = k.alloc(1)?;
	k.body = vec![store(
		out,
		I::Const(0),
		E::Load(x, I::Const(2)).add(E::Const(1.)),
	)];
	k.outputs = vec![out];
	assert_eq!(k.compile()?.run(&[vec![10., 20., 30.]])?, vec![vec![31.]]);
	let mut k = Kernel::default();
	let x = k.input(0, 3)?;
	let y = k.input(1, 3)?;
	let out = k.alloc(3)?;
	let other = k.alloc(3)?;
	assert_ne!(out.storage, other.storage);
	k.body = (0..3)
		.map(|i| {
			store(
				out,
				I::Const(i),
				E::Load(x, I::Const(i)).add(E::Load(y, I::Const(i))),
			)
		})
		.collect();
	k.outputs = vec![out];
	assert_eq!(
		k.compile()?.run(&[vec![1., 2., 3.], vec![10., 20., 30.]])?,
		vec![vec![11., 22., 33.]]
	);
	assert!(
		k.compile()?
			.run(&[vec![1., 2.], vec![10., 20., 30.]])
			.is_err()
	);
	Ok(())
}
```

```sh check
cargo test --quiet --test memory_runtime fixed_accesses_and_vector_add
```

## 16. dtype、アドレス kind、保存領域の可変性を検証する。
<!-- todo:05-016 -->

`tests/memory_runtime.rs` の `fixed_accesses_and_vector_add` で、この経路の値または拒否条件を確認する。

`src/kernel.rs` の `pub fn verify(`の定義を次の全体で置き換えます。

```rust file="src/kernel.rs" mode="replace" item="pub fn verify("
	pub fn verify(&self) -> Result<Accesses> {
		let count = self
			.storage
			.iter()
			.filter_map(|s| s.input)
			.max()
			.map(|n| n + 1)
			.unwrap_or(0);
		let mut inputs = vec![vec![]; count];
		for s in &self.storage {
			if let Some(slot) = s.input {
				if !inputs[slot].is_empty() && inputs[slot].len() != s.len {
					return Err("input alias lengths differ".into());
				}
				inputs[slot] = vec![0.; s.len]
			}
		}
		Ok(self.execute(&inputs)?.1)
	}
```

```sh check
cargo test --quiet --test memory_runtime fixed_accesses_and_vector_add
```

## 17. SHRINK の `0 <= start <= end <= len` を検証する。
<!-- todo:05-017 -->

`tests/memory_runtime.rs` の正常系と境界系をまとめて再実行する。

`src/kernel.rs` の `pub fn shrink(`の定義を次の全体で置き換えます。

```rust file="src/kernel.rs" mode="replace" item="pub fn shrink("
	pub fn shrink(self, start: usize, end: usize) -> Result<Self> {
		if start > end || end > self.len {
			return Err("invalid memory SHRINK".into());
		}
		Ok(Self {
			storage: self.storage,
			offset: self
				.offset
				.checked_add(start)
				.ok_or("view offset overflow")?,
			len: end - start,
		})
	}
```

```sh check
cargo test --quiet --test memory_runtime
```

## 18. 静的 INDEX が view の範囲内にあることを検証する。
<!-- todo:05-018 -->

`tests/memory_runtime.rs` の正常系と境界系をまとめて再実行する。

`src/kernel.rs` の `fn address(`の定義を次の全体で置き換えます。

```rust file="src/kernel.rs" mode="replace" item="fn address("
	fn address(&self, view: View, index: &Index, axes: &HashMap<usize, i32>) -> Result<usize> {
		let storage = self.storage.get(view.storage).ok_or("unknown storage")?;
		if view
			.offset
			.checked_add(view.len)
			.filter(|&n| n <= storage.len)
			.is_none()
		{
			return Err("view exceeds storage".into());
		}
		let i = usize::try_from(index.value(axes)?).map_err(|_| "negative INDEX")?;
		if i >= view.len {
			return Err("INDEX exceeds view".into());
		}
		view.offset.checked_add(i).ok_or("address overflow".into())
	}
```

```sh check
cargo test --quiet --test memory_runtime
```

## 19. 未初期化出力の LOAD と AFTER の循環を拒否する。
<!-- todo:05-019 -->

`tests/memory_runtime.rs` の正常系と境界系をまとめて再実行する。

`src/kernel.rs` の `pub fn verify(`の定義を次の全体で置き換えます。

```rust file="src/kernel.rs" mode="replace" item="pub fn verify("
	pub fn verify(&self) -> Result<Accesses> {
		let count = self
			.storage
			.iter()
			.filter_map(|s| s.input)
			.max()
			.map(|n| n + 1)
			.unwrap_or(0);
		let mut inputs = vec![vec![]; count];
		for s in &self.storage {
			if let Some(slot) = s.input {
				if !inputs[slot].is_empty() && inputs[slot].len() != s.len {
					return Err("input alias lengths differ".into());
				}
				inputs[slot] = vec![0.; s.len]
			}
		}
		Ok(self.execute(&inputs)?.1)
	}
```

```sh check
cargo test --quiet --test memory_runtime
```

## 20. `len*itemsize` と offset の overflow をホスト上の checked 演算で拒否する。
<!-- todo:05-020 -->

`tests/memory_runtime.rs` の正常系と境界系をまとめて再実行する。

`src/kernel.rs` の `fn allocate(`の定義を次の全体で置き換えます。

```rust file="src/kernel.rs" mode="replace" item="fn allocate("
	fn allocate(&mut self, len: usize, input: Option<usize>, register: bool) -> Result<View> {
		if len > i32::MAX as usize
			|| len
				.checked_mul(4)
				.filter(|&n| n <= isize::MAX as usize)
				.is_none()
		{
			return Err("storage size overflow".into());
		}
		let storage = self.storage.len();
		self.storage.push(Storage {
			len,
			input,
			register,
		});
		Ok(View {
			storage,
			offset: 0,
			len,
		})
	}
```

```sh check
cargo test --quiet --test memory_runtime
```

## 21. 固定アクセスの結果が `31`、view の結果が `20` になる。
<!-- todo:05-021 -->

`tests/memory_runtime.rs` の `fixed_accesses_and_vector_add` で、この経路の値または拒否条件を確認する。

`tests/memory_runtime.rs` の `fn fixed_accesses_and_vector_add(`の定義を次の全体で置き換えます。

```rust file="tests/memory_runtime.rs" mode="replace" item="fn fixed_accesses_and_vector_add("
fn fixed_accesses_and_vector_add() -> Result<()> {
	let mut k = Kernel::default();
	let x = k.input(0, 3)?;
	let out = k.alloc(1)?;
	k.body = vec![store(
		out,
		I::Const(0),
		E::Load(x, I::Const(2)).add(E::Const(1.)),
	)];
	k.outputs = vec![out];
	assert_eq!(k.compile()?.run(&[vec![10., 20., 30.]])?, vec![vec![31.]]);
	let mut k = Kernel::default();
	let x = k.input(0, 3)?;
	let y = k.input(1, 3)?;
	let out = k.alloc(3)?;
	let other = k.alloc(3)?;
	assert_ne!(out.storage, other.storage);
	k.body = (0..3)
		.map(|i| {
			store(
				out,
				I::Const(i),
				E::Load(x, I::Const(i)).add(E::Load(y, I::Const(i))),
			)
		})
		.collect();
	k.outputs = vec![out];
	assert_eq!(
		k.compile()?.run(&[vec![1., 2., 3.], vec![10., 20., 30.]])?,
		vec![vec![11., 22., 33.]]
	);
	assert!(
		k.compile()?
			.run(&[vec![1., 2.], vec![10., 20., 30.]])
			.is_err()
	);
	Ok(())
}
```

```sh check
cargo test --quiet --test memory_runtime fixed_accesses_and_vector_add
```

## 22. 二入力の三要素加算が `[11,22,33]` になる。
<!-- todo:05-022 -->

`tests/memory_runtime.rs` の `fixed_accesses_and_vector_add` で、この経路の値または拒否条件を確認する。

`tests/memory_runtime.rs` の `fn fixed_accesses_and_vector_add(`の定義を次の全体で置き換えます。

```rust file="tests/memory_runtime.rs" mode="replace" item="fn fixed_accesses_and_vector_add("
fn fixed_accesses_and_vector_add() -> Result<()> {
	let mut k = Kernel::default();
	let x = k.input(0, 3)?;
	let out = k.alloc(1)?;
	k.body = vec![store(
		out,
		I::Const(0),
		E::Load(x, I::Const(2)).add(E::Const(1.)),
	)];
	k.outputs = vec![out];
	assert_eq!(k.compile()?.run(&[vec![10., 20., 30.]])?, vec![vec![31.]]);
	let mut k = Kernel::default();
	let x = k.input(0, 3)?;
	let y = k.input(1, 3)?;
	let out = k.alloc(3)?;
	let other = k.alloc(3)?;
	assert_ne!(out.storage, other.storage);
	k.body = (0..3)
		.map(|i| {
			store(
				out,
				I::Const(i),
				E::Load(x, I::Const(i)).add(E::Load(y, I::Const(i))),
			)
		})
		.collect();
	k.outputs = vec![out];
	assert_eq!(
		k.compile()?.run(&[vec![1., 2., 3.], vec![10., 20., 30.]])?,
		vec![vec![11., 22., 33.]]
	);
	assert!(
		k.compile()?
			.run(&[vec![1., 2.], vec![10., 20., 30.]])
			.is_err()
	);
	Ok(())
}
```

```sh check
cargo test --quiet --test memory_runtime fixed_accesses_and_vector_add
```

## 23. 書込み前後の LOAD が state を区別し、最終値 `2` を読む。
<!-- todo:05-023 -->

`tests/memory_runtime.rs` の正常系と境界系をまとめて再実行する。

`tests/memory_runtime.rs` の `fn memory_state_and_views(`の定義を次の全体で置き換えます。

```rust file="tests/memory_runtime.rs" mode="replace" item="fn memory_state_and_views("
fn memory_state_and_views() -> Result<()> {
	let mut k = Kernel::default();
	let x = k.input(0, 3)?;
	let out = k.alloc(3)?;
	let state = k.alloc(1)?;
	k.body = vec![
		store(state, I::Const(0), E::Const(1.)),
		store(out, I::Const(0), E::Load(state, I::Const(0))),
		store(state, I::Const(0), E::Const(2.)),
		store(out, I::Const(1), E::Load(state, I::Const(0))),
		store(out, I::Const(2), E::Load(x.shrink(1, 3)?, I::Const(0))),
	];
	k.outputs = vec![out];
	assert_eq!(x.storage, x.shrink(1, 3)?.storage);
	assert_eq!(
		k.compile()?.run(&[vec![10., 20., 30.]])?,
		vec![vec![1., 2., 20.]]
	);
	let access = k.verify()?;
	assert!(access.read.contains(&x.storage));
	assert!(access.write.contains(&out.storage));
	let bad = store(out, I::Const(0), E::Load(k.alloc(1)?, I::Const(0)));
	k.body.insert(0, bad);
	assert!(k.compile().is_err());
	Ok(())
}
```

```sh check
cargo test --quiet --test memory_runtime
```

## 24. dump の base identity は view 作成前後で一致する。
<!-- todo:05-024 -->

`tests/memory_runtime.rs` の正常系と境界系をまとめて再実行する。

`tests/memory_runtime.rs` の `fn memory_state_and_views(`の定義を次の全体で置き換えます。

```rust file="tests/memory_runtime.rs" mode="replace" item="fn memory_state_and_views("
fn memory_state_and_views() -> Result<()> {
	let mut k = Kernel::default();
	let x = k.input(0, 3)?;
	let out = k.alloc(3)?;
	let state = k.alloc(1)?;
	k.body = vec![
		store(state, I::Const(0), E::Const(1.)),
		store(out, I::Const(0), E::Load(state, I::Const(0))),
		store(state, I::Const(0), E::Const(2.)),
		store(out, I::Const(1), E::Load(state, I::Const(0))),
		store(out, I::Const(2), E::Load(x.shrink(1, 3)?, I::Const(0))),
	];
	k.outputs = vec![out];
	assert_eq!(x.storage, x.shrink(1, 3)?.storage);
	assert_eq!(
		k.compile()?.run(&[vec![10., 20., 30.]])?,
		vec![vec![1., 2., 20.]]
	);
	let access = k.verify()?;
	assert!(access.read.contains(&x.storage));
	assert!(access.write.contains(&out.storage));
	let bad = store(out, I::Const(0), E::Load(k.alloc(1)?, I::Const(0)));
	k.body.insert(0, bad);
	assert!(k.compile().is_err());
	Ok(())
}
```

```sh check
cargo test --quiet --test memory_runtime
```

## 25. accesses が入力を read、出力を write として列挙する。
<!-- todo:05-025 -->

`tests/memory_runtime.rs` の正常系と境界系をまとめて再実行する。

`tests/memory_runtime.rs` の `fn memory_state_and_views(`の定義を次の全体で置き換えます。

```rust file="tests/memory_runtime.rs" mode="replace" item="fn memory_state_and_views("
fn memory_state_and_views() -> Result<()> {
	let mut k = Kernel::default();
	let x = k.input(0, 3)?;
	let out = k.alloc(3)?;
	let state = k.alloc(1)?;
	k.body = vec![
		store(state, I::Const(0), E::Const(1.)),
		store(out, I::Const(0), E::Load(state, I::Const(0))),
		store(state, I::Const(0), E::Const(2.)),
		store(out, I::Const(1), E::Load(state, I::Const(0))),
		store(out, I::Const(2), E::Load(x.shrink(1, 3)?, I::Const(0))),
	];
	k.outputs = vec![out];
	assert_eq!(x.storage, x.shrink(1, 3)?.storage);
	assert_eq!(
		k.compile()?.run(&[vec![10., 20., 30.]])?,
		vec![vec![1., 2., 20.]]
	);
	let access = k.verify()?;
	assert!(access.read.contains(&x.storage));
	assert!(access.write.contains(&out.storage));
	let bad = store(out, I::Const(0), E::Load(k.alloc(1)?, I::Const(0)));
	k.body.insert(0, bad);
	assert!(k.compile().is_err());
	Ok(())
}
```

```sh check
cargo test --quiet --test memory_runtime
```

## 26. 範囲外、型不一致、未初期化、効果循環が実行前にエラーになる。
<!-- todo:05-026 -->

`tests/memory_runtime.rs` の正常系と境界系をまとめて再実行する。

`tests/memory_runtime.rs` の `fn memory_state_and_views(`の定義を次の全体で置き換えます。

```rust file="tests/memory_runtime.rs" mode="replace" item="fn memory_state_and_views("
fn memory_state_and_views() -> Result<()> {
	let mut k = Kernel::default();
	let x = k.input(0, 3)?;
	let out = k.alloc(3)?;
	let state = k.alloc(1)?;
	k.body = vec![
		store(state, I::Const(0), E::Const(1.)),
		store(out, I::Const(0), E::Load(state, I::Const(0))),
		store(state, I::Const(0), E::Const(2.)),
		store(out, I::Const(1), E::Load(state, I::Const(0))),
		store(out, I::Const(2), E::Load(x.shrink(1, 3)?, I::Const(0))),
	];
	k.outputs = vec![out];
	assert_eq!(x.storage, x.shrink(1, 3)?.storage);
	assert_eq!(
		k.compile()?.run(&[vec![10., 20., 30.]])?,
		vec![vec![1., 2., 20.]]
	);
	let access = k.verify()?;
	assert!(access.read.contains(&x.storage));
	assert!(access.write.contains(&out.storage));
	let bad = store(out, I::Const(0), E::Load(k.alloc(1)?, I::Const(0)));
	k.body.insert(0, bad);
	assert!(k.compile().is_err());
	Ok(())
}
```

```sh check
cargo test --quiet --test memory_runtime
```

## 章末の累積確認

前章までの example と test も同じ checkout で再ビルドする。

```sh check
cargo test --quiet --all-targets
```
