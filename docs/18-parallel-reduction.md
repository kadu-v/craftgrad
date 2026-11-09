# 18. LOCAL と一様な barrier で並列縮約する

この章は前章までの `chibigrad` に追加する。掲載順に `file=` のパスへ保存する。`mode=write` はそのファイル全体を置き換え、`mode=append` は末尾へ追記する。既存の別ファイルは残す。各コードは省略なしの実装である。

CPU の実行結果と GPU の実機結果を別に記録する。`requires=metal` と `requires=opencl` の確認は、その device がある環境で実行する。機器がなければ `NOT_RUN` と記録し、source 検査の成功へ置き換えない。

`LocalOp` は barrier の条件分岐を表現しない。一様な命令列から Metal と OpenCL の source を生成する。tree の一回の加算後に全 lane が barrier を通る。全 GPU の同期は二つの kernel と runtime の完了待ちで表す。後続 program は `compile_next` で同じ device と queue を使う。OpenCL では context も共有する。実機 test は `shares_queue` で同じ API object を確認してから、部分和を次の入力へ bind する。

<a id="local-implementation"></a>

## 実装を保存する

```rust file=src/lib.rs mode=append
pub mod collective;
pub mod gpu_kernels;
```

```rust file=src/collective.rs mode=write
use crate::Result;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalOp {
	WritePartial,
	Barrier,
	Tree { stride: usize },
	SerialFinish,
	StoreLaneZero,
}
#[derive(Clone, Debug)]
pub struct Collective {
	pub lanes: usize,
	pub local_elements: usize,
	pub ops: Vec<LocalOp>,
}
impl Collective {
	pub fn row_sum(lanes: usize, tree: bool) -> Result<Self> {
		if !lanes.is_power_of_two() {
			return Err("local must be power of two".into());
		}
		let mut ops = vec![LocalOp::WritePartial, LocalOp::Barrier];
		if tree {
			let mut stride = lanes / 2;
			while stride > 0 {
				ops.extend([LocalOp::Tree { stride }, LocalOp::Barrier]);
				stride /= 2;
			}
		} else {
			ops.push(LocalOp::SerialFinish);
		}
		ops.push(LocalOp::StoreLaneZero);
		let program = Self {
			lanes,
			local_elements: lanes,
			ops,
		};
		program.verify()?;
		Ok(program)
	}
	pub fn verify(&self) -> Result<()> {
		if self.lanes == 0 || self.local_elements < self.lanes {
			return Err("LOCAL allocation shorter than lane stores".into());
		}
		let mut written = false;
		let mut visible = false;
		let mut final_store = false;
		for op in &self.ops {
			match *op {
				LocalOp::WritePartial => {
					if written && !visible {
						return Err("LOCAL reuse before barrier".into());
					}
					written = true;
					visible = false;
				}
				LocalOp::Barrier => {
					if !written {
						return Err("barrier without LOCAL write".into());
					}
					visible = true;
				}
				LocalOp::Tree { stride } => {
					if !visible
						|| stride == 0 || stride
						.checked_mul(2)
						.is_none_or(|n| n > self.local_elements)
					{
						return Err("LOCAL tree read before barrier or outside allocation".into());
					}
					visible = false;
				}
				LocalOp::SerialFinish => {
					if !visible {
						return Err("serial LOCAL read before barrier".into());
					}
				}
				LocalOp::StoreLaneZero => {
					if !visible {
						return Err("global store before LOCAL results visible".into());
					}
					final_store = true;
				}
			}
		}
		if !final_store {
			return Err("missing lane0 output".into());
		}
		Ok(())
	}
	pub fn linearize(&self) -> Result<Vec<LocalOp>> {
		self.verify()?;
		Ok(self.ops.clone())
	}
}
```

```rust file=src/gpu_kernels.rs mode=write
use crate::gpu::{Dialect, Kernel, groups, header};
fn shared(d: Dialect) -> &'static str {
	match d {
		Dialect::Metal => "threadgroup",
		Dialect::OpenCl => "__local",
	}
}
fn barrier(d: Dialect) -> &'static str {
	match d {
		Dialect::Metal => "threadgroup_barrier(mem_flags::mem_threadgroup);",
		Dialect::OpenCl => "barrier(CLK_LOCAL_MEM_FENCE);",
	}
}
fn spec(
	d: Dialect,
	lengths: Vec<usize>,
	body: String,
	group_count: usize,
	local: usize,
	bytes: usize,
) -> Result<Kernel, String> {
	if local == 0 {
		return Err("local=0".into());
	}
	group_count.checked_mul(local).ok_or("launch overflow")?;
	let output = lengths.len() - 1;
	Ok(Kernel {
		scalars: vec![],
		source: header(d, lengths.len(), &body),
		entry: "chibi".into(),
		lengths,
		output,
		groups: group_count,
		local,
		shared_bytes: bytes,
	})
}
pub fn row_sum(
	d: Dialect,
	rows: usize,
	cols: usize,
	local: usize,
	tree: bool,
) -> Result<Kernel, String> {
	if !local.is_power_of_two() {
		return Err("reduction local must be a power of two".into());
	}
	let len = rows.checked_mul(cols).ok_or("shape overflow")?;
	let program = crate::collective::Collective::row_sum(local, tree)?;
	let mut body = format!("{} float tmp[{local}];", shared(d));
	for op in program.linearize()? {
		match op {
			crate::collective::LocalOp::WritePartial => {
				body += &format!(
					"float acc=0.0f;for(size_t c=l;c<{cols};c+={local}){{acc+=b0[g*{cols}+c];}}tmp[l]=acc;"
				)
			}
			crate::collective::LocalOp::Barrier => body += barrier(d),
			crate::collective::LocalOp::Tree { stride } => {
				body += &format!("if(l<{stride}){{tmp[l]+=tmp[l+{stride}];}}")
			}
			crate::collective::LocalOp::SerialFinish => {
				body += &format!(
					"if(l==0){{float sum=0.0f;for(size_t c=0;c<{local};c++){{sum+=tmp[c];}}tmp[0]=sum;}}"
				)
			}
			crate::collective::LocalOp::StoreLaneZero => body += "if(l==0){b1[g]=tmp[0];}",
		}
	}
	spec(
		d,
		vec![len, rows],
		body,
		rows,
		local,
		local.checked_mul(4).ok_or("local byte overflow")?,
	)
}
#[derive(Clone, Debug)]
pub struct ReductionSchedule {
	pub first: Kernel,
	pub second: Kernel,
}
pub fn two_stage_sum(
	d: Dialect,
	n: usize,
	chunk: usize,
	local: usize,
) -> Result<ReductionSchedule, String> {
	if chunk == 0 {
		return Err("chunk=0".into());
	}
	let count = groups(n, chunk)?;
	let mut first = row_sum(d, count, chunk, local, true)?;
	first.lengths[0] = n;
	first.source = first.source.replace(
		&format!("acc+=b0[g*{chunk}+c];"),
		&format!("if(g*{chunk}+c<{n}){{acc+=b0[g*{chunk}+c];}}"),
	);
	Ok(ReductionSchedule {
		first,
		second: row_sum(d, 1, count, local, true)?,
	})
}
pub fn dynamic_sum(
	d: Dialect,
	n: usize,
	capacity: usize,
	local: usize,
) -> Result<ReductionSchedule, String> {
	if n > capacity {
		return Err("logical reduction length exceeds capacity".into());
	}
	if !local.is_power_of_two() {
		return Err("reduction local must be power of two".into());
	}
	let chunk = local.checked_mul(4).ok_or("chunk overflow")?;
	let count = groups(n, chunk)?;
	let mut first = row_sum(d, count, chunk, local, true)?;
	first.lengths[0] = capacity;
	first.scalars = vec![n as u64];
	first.source = first.source.replace(
		&format!("acc+=b0[g*{chunk}+c];"),
		&format!("if(g*{chunk}+c<s[0]){{acc+=b0[g*{chunk}+c];}}"),
	);
	let mut second = row_sum(d, 1, count, local, true)?;
	second.scalars = vec![count as u64];
	second.source = second
		.source
		.replace(&format!("c<{count};"), "c<s[0];")
		.replace(&format!("g*{count}+c"), "g*s[0]+c");
	Ok(ReductionSchedule { first, second })
}
```

```rust file=src/gpu_graph.rs mode=append
impl GpuPlan {
	pub fn compile_parallel(
		graph: &Graph,
		outputs: &[usize],
		backend: Backend,
		local: usize,
	) -> Result<Self> {
		let mut lowered = lower(graph, outputs, backend.dialect(), local)?;
		for (kernel, slots) in &mut lowered {
			let out = *slots.last().ok_or("output")?;
			if let Op::Reduce {
				op: crate::Reduce::Sum,
				x,
				ref axes,
				..
			} = graph.nodes[out].op
			{
				let shape = graph.shape(x);
				if shape.len() == 2 && axes == &[1] && matches!(graph.nodes[x].op, Op::Input { .. })
				{
					*kernel = crate::gpu_kernels::row_sum(
						backend.dialect(),
						shape[0],
						shape[1],
						local,
						true,
					)?;
					*slots = vec![x, out];
				}
			}
		}
		Self::from_lowered(graph, outputs, backend, lowered, true)
	}
}
```

```rust file=tests/collective.rs mode=write
use chibigrad::{
	collective::{Collective, LocalOp},
	gpu::Dialect,
	gpu_kernels::*,
};
#[test]
fn barriers_and_shader_compile() {
	assert!(
		Collective {
			lanes: 64,
			local_elements: 63,
			ops: vec![]
		}
		.verify()
		.is_err()
	);
	assert!(
		Collective {
			lanes: 64,
			local_elements: 64,
			ops: vec![
				LocalOp::WritePartial,
				LocalOp::Tree { stride: 32 },
				LocalOp::Barrier,
				LocalOp::StoreLaneZero
			]
		}
		.verify()
		.is_err()
	);
	assert!(row_sum(Dialect::OpenCl, 3, 257, 3, true).is_err());
	let mut sources = vec![];
	for cols in [1, 63, 64, 65, 257] {
		sources.push(row_sum(Dialect::OpenCl, 2, cols, 64, true).unwrap().source);
		sources.push(row_sum(Dialect::OpenCl, 2, cols, 64, false).unwrap().source);
	}
	for (i, source) in sources.into_iter().enumerate() {
		let path = std::env::temp_dir().join(format!(
			"chibi-local-{}-{:?}-{i}.cl",
			std::process::id(),
			std::thread::current().id()
		));
		std::fs::write(&path, source).unwrap();
		let output = std::process::Command::new("clang")
			.args(["-x", "cl", "-cl-std=CL1.2", "-fsyntax-only"])
			.arg(&path)
			.output()
			.unwrap();
		let _ = std::fs::remove_file(path);
		assert!(
			output.status.success(),
			"{}",
			String::from_utf8_lossy(&output.stderr)
		);
	}
}
```

```rust file=tests/reduction_hardware.rs mode=write
use chibigrad::{
	gpu::{Dialect, Kernel},
	gpu_kernels::{dynamic_sum, row_sum},
};
fn compare(actual: &[f32], reference: &[f64]) {
	for (&a, &b) in actual.iter().zip(reference) {
		assert!(a.is_finite() && (a as f64 - b).abs() <= 1e-5 + 1e-5 * b.abs());
	}
}
fn cases(d: Dialect, run: impl Fn(&Kernel, &[Vec<f32>]) -> Result<Vec<f32>, String>) {
	for n in [1, 63, 64, 65, 257] {
		for tree in [false, true] {
			let k = row_sum(d, 2, n, 64, tree).unwrap();
			let x = (0..2 * n).map(|i| (i % 7) as f32 - 3.).collect::<Vec<_>>();
			let reference = x
				.chunks(n)
				.map(|s| s.iter().map(|&x| x as f64).sum())
				.collect::<Vec<_>>();
			for _ in 0..3 {
				compare(&run(&k, &[x.clone(), vec![0.; 2]]).unwrap(), &reference);
			}
		}
	}
	for n in [0, 1, 3, 17] {
		let p = dynamic_sum(d, n, 17, 8).unwrap();
		let x = (0..17).map(|i| if i < n { 1. } else { f32::NAN }).collect();
		let part = run(&p.first, &[x, vec![0.; p.first.lengths[1]]]).unwrap();
		let out = run(&p.second, &[part, vec![0.]]).unwrap();
		assert_eq!(out, [n as f32]);
	}
}
#[test]
#[ignore = "requires Metal"]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn metal() {
	let seed = chibigrad::backends::metal::MetalProgram::compile(
		&row_sum(Dialect::Metal, 1, 1, 64, true).unwrap(),
	)
	.unwrap();
	cases(Dialect::Metal, |k, b| {
		let p = seed.compile_next(k)?;
		assert!(seed.shares_queue(&p));
		Ok(unsafe { p.run(k, b) }?.output)
	});
}

#[test]
#[ignore = "requires OpenCL"]
#[cfg(feature = "opencl")]
fn opencl() {
	let seed = chibigrad::backends::opencl::OpenClProgram::compile(
		&row_sum(Dialect::OpenCl, 1, 1, 64, true).unwrap(),
		0,
		0,
	)
	.unwrap();
	cases(Dialect::OpenCl, |k, b| {
		let p = seed.compile_next(k)?;
		assert!(seed.shares_queue(&p));
		Ok(unsafe { p.run(k, b) }?.output)
	});
}
```

```sh check requires=metal
cargo test --test reduction_hardware --features metal metal -- --ignored --exact
```

```sh check requires=opencl
cargo test --test reduction_hardware --features opencl opencl -- --ignored --exact
```

```rust file=tests/book_18.rs mode=write
mod suite0 {
	use chibigrad::{
		collective::{Collective, LocalOp},
		gpu::Dialect,
		gpu_kernels::*,
	};
	#[test]
	pub(super) fn barriers_and_shader_compile() {
		assert!(
			Collective {
				lanes: 64,
				local_elements: 63,
				ops: vec![]
			}
			.verify()
			.is_err()
		);
		assert!(
			Collective {
				lanes: 64,
				local_elements: 64,
				ops: vec![
					LocalOp::WritePartial,
					LocalOp::Tree { stride: 32 },
					LocalOp::Barrier,
					LocalOp::StoreLaneZero
				]
			}
			.verify()
			.is_err()
		);
		assert!(row_sum(Dialect::OpenCl, 3, 257, 3, true).is_err());
		let mut sources = vec![];
		for cols in [1, 63, 64, 65, 257] {
			sources.push(row_sum(Dialect::OpenCl, 2, cols, 64, true).unwrap().source);
			sources.push(row_sum(Dialect::OpenCl, 2, cols, 64, false).unwrap().source);
		}
		for (i, source) in sources.into_iter().enumerate() {
			let path = std::env::temp_dir().join(format!(
				"chibi-local-{}-{:?}-{i}.cl",
				std::process::id(),
				std::thread::current().id()
			));
			std::fs::write(&path, source).unwrap();
			let output = std::process::Command::new("clang")
				.args(["-x", "cl", "-cl-std=CL1.2", "-fsyntax-only"])
				.arg(&path)
				.output()
				.unwrap();
			let _ = std::fs::remove_file(path);
			assert!(
				output.status.success(),
				"{}",
				String::from_utf8_lossy(&output.stderr)
			);
		}
	}
}
```

## LOCAL storage と BARRIER を verifier、linearize、renderer に追加する。

<!-- todo:18-001 -->

実装は[この章の `Collective::verify、Collective::linearize、gpu_kernels::row_sum、two_stage_sum、dynamic_sum、GpuPlan::compile_parallel`](#local-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_18.rs mode=append
#[test]
fn todo_18_001() {
	suite0::barriers_and_shader_compile();
}
```

`src/collective.rs` の `pub fn verify(`の定義を次の全体で置き換えます。

```rust file="src/collective.rs" mode="replace" item="pub fn verify("
	pub fn verify(&self) -> Result<()> {
		if self.lanes == 0 || self.local_elements < self.lanes {
			return Err("LOCAL allocation shorter than lane stores".into());
		}
		let mut written = false;
		let mut visible = false;
		let mut final_store = false;
		for op in &self.ops {
			match *op {
				LocalOp::WritePartial => {
					if written && !visible {
						return Err("LOCAL reuse before barrier".into());
					}
					written = true;
					visible = false;
				}
				LocalOp::Barrier => {
					if !written {
						return Err("barrier without LOCAL write".into());
					}
					visible = true;
				}
				LocalOp::Tree { stride } => {
					if !visible
						|| stride == 0 || stride
						.checked_mul(2)
						.is_none_or(|n| n > self.local_elements)
					{
						return Err("LOCAL tree read before barrier or outside allocation".into());
					}
					visible = false;
				}
				LocalOp::SerialFinish => {
					if !visible {
						return Err("serial LOCAL read before barrier".into());
					}
				}
				LocalOp::StoreLaneZero => {
					if !visible {
						return Err("global store before LOCAL results visible".into());
					}
					final_store = true;
				}
			}
		}
		if !final_store {
			return Err("missing lane0 output".into());
		}
		Ok(())
	}
```

`src/collective.rs` の `pub fn linearize(`の定義を次の全体で置き換えます。

```rust file="src/collective.rs" mode="replace" item="pub fn linearize("
	pub fn linearize(&self) -> Result<Vec<LocalOp>> {
		self.verify()?;
		Ok(self.ops.clone())
	}
```

`src/gpu_kernels.rs` の `pub fn row_sum(`の定義を次の全体で置き換えます。

```rust file="src/gpu_kernels.rs" mode="replace" item="pub fn row_sum("
pub fn row_sum(
	d: Dialect,
	rows: usize,
	cols: usize,
	local: usize,
	tree: bool,
) -> Result<Kernel, String> {
	if !local.is_power_of_two() {
		return Err("reduction local must be a power of two".into());
	}
	let len = rows.checked_mul(cols).ok_or("shape overflow")?;
	let program = crate::collective::Collective::row_sum(local, tree)?;
	let mut body = format!("{} float tmp[{local}];", shared(d));
	for op in program.linearize()? {
		match op {
			crate::collective::LocalOp::WritePartial => {
				body += &format!(
					"float acc=0.0f;for(size_t c=l;c<{cols};c+={local}){{acc+=b0[g*{cols}+c];}}tmp[l]=acc;"
				)
			}
			crate::collective::LocalOp::Barrier => body += barrier(d),
			crate::collective::LocalOp::Tree { stride } => {
				body += &format!("if(l<{stride}){{tmp[l]+=tmp[l+{stride}];}}")
			}
			crate::collective::LocalOp::SerialFinish => {
				body += &format!(
					"if(l==0){{float sum=0.0f;for(size_t c=0;c<{local};c++){{sum+=tmp[c];}}tmp[0]=sum;}}"
				)
			}
			crate::collective::LocalOp::StoreLaneZero => body += "if(l==0){b1[g]=tmp[0];}",
		}
	}
	spec(
		d,
		vec![len, rows],
		body,
		rows,
		local,
		local.checked_mul(4).ok_or("local byte overflow")?,
	)
}
```

```sh check
cargo test --test book_18 todo_18_001 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 各 lane が `col=chunk*L+lane` の部分和を逐次計算する。

<!-- todo:18-002 -->

実装は[この章の `Collective::verify、Collective::linearize、gpu_kernels::row_sum、two_stage_sum、dynamic_sum、GpuPlan::compile_parallel`](#local-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_18.rs mode=append
#[test]
fn todo_18_002() {
	suite0::barriers_and_shader_compile();
}
```

`src/gpu_kernels.rs` の `pub fn row_sum(`の定義を次の全体で置き換えます。

```rust file="src/gpu_kernels.rs" mode="replace" item="pub fn row_sum("
pub fn row_sum(
	d: Dialect,
	rows: usize,
	cols: usize,
	local: usize,
	tree: bool,
) -> Result<Kernel, String> {
	if !local.is_power_of_two() {
		return Err("reduction local must be a power of two".into());
	}
	let len = rows.checked_mul(cols).ok_or("shape overflow")?;
	let program = crate::collective::Collective::row_sum(local, tree)?;
	let mut body = format!("{} float tmp[{local}];", shared(d));
	for op in program.linearize()? {
		match op {
			crate::collective::LocalOp::WritePartial => {
				body += &format!(
					"float acc=0.0f;for(size_t c=l;c<{cols};c+={local}){{acc+=b0[g*{cols}+c];}}tmp[l]=acc;"
				)
			}
			crate::collective::LocalOp::Barrier => body += barrier(d),
			crate::collective::LocalOp::Tree { stride } => {
				body += &format!("if(l<{stride}){{tmp[l]+=tmp[l+{stride}];}}")
			}
			crate::collective::LocalOp::SerialFinish => {
				body += &format!(
					"if(l==0){{float sum=0.0f;for(size_t c=0;c<{local};c++){{sum+=tmp[c];}}tmp[0]=sum;}}"
				)
			}
			crate::collective::LocalOp::StoreLaneZero => body += "if(l==0){b1[g]=tmp[0];}",
		}
	}
	spec(
		d,
		vec![len, rows],
		body,
		rows,
		local,
		local.checked_mul(4).ok_or("local byte overflow")?,
	)
}
```

```sh check
cargo test --test book_18 todo_18_002 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 全 lane が LOCAL の自分の位置に一つ書く。

<!-- todo:18-003 -->

実装は[この章の `Collective::verify、Collective::linearize、gpu_kernels::row_sum、two_stage_sum、dynamic_sum、GpuPlan::compile_parallel`](#local-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_18.rs mode=append
#[test]
fn todo_18_003() {
	suite0::barriers_and_shader_compile();
}
```

`src/gpu_kernels.rs` の `pub fn row_sum(`の定義を次の全体で置き換えます。

```rust file="src/gpu_kernels.rs" mode="replace" item="pub fn row_sum("
pub fn row_sum(
	d: Dialect,
	rows: usize,
	cols: usize,
	local: usize,
	tree: bool,
) -> Result<Kernel, String> {
	if !local.is_power_of_two() {
		return Err("reduction local must be a power of two".into());
	}
	let len = rows.checked_mul(cols).ok_or("shape overflow")?;
	let program = crate::collective::Collective::row_sum(local, tree)?;
	let mut body = format!("{} float tmp[{local}];", shared(d));
	for op in program.linearize()? {
		match op {
			crate::collective::LocalOp::WritePartial => {
				body += &format!(
					"float acc=0.0f;for(size_t c=l;c<{cols};c+={local}){{acc+=b0[g*{cols}+c];}}tmp[l]=acc;"
				)
			}
			crate::collective::LocalOp::Barrier => body += barrier(d),
			crate::collective::LocalOp::Tree { stride } => {
				body += &format!("if(l<{stride}){{tmp[l]+=tmp[l+{stride}];}}")
			}
			crate::collective::LocalOp::SerialFinish => {
				body += &format!(
					"if(l==0){{float sum=0.0f;for(size_t c=0;c<{local};c++){{sum+=tmp[c];}}tmp[0]=sum;}}"
				)
			}
			crate::collective::LocalOp::StoreLaneZero => body += "if(l==0){b1[g]=tmp[0];}",
		}
	}
	spec(
		d,
		vec![len, rows],
		body,
		rows,
		local,
		local.checked_mul(4).ok_or("local byte overflow")?,
	)
}
```

```sh check
cargo test --test book_18 todo_18_003 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## barrier 後だけ LOCAL を読み、lane 0 が一出力を書く。

<!-- todo:18-004 -->

実装は[この章の `Collective::verify、Collective::linearize、gpu_kernels::row_sum、two_stage_sum、dynamic_sum、GpuPlan::compile_parallel`](#local-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_18.rs mode=append
#[test]
fn todo_18_004() {
	suite0::barriers_and_shader_compile();
}
```

`src/gpu_kernels.rs` の `pub fn row_sum(`の定義を次の全体で置き換えます。

```rust file="src/gpu_kernels.rs" mode="replace" item="pub fn row_sum("
pub fn row_sum(
	d: Dialect,
	rows: usize,
	cols: usize,
	local: usize,
	tree: bool,
) -> Result<Kernel, String> {
	if !local.is_power_of_two() {
		return Err("reduction local must be a power of two".into());
	}
	let len = rows.checked_mul(cols).ok_or("shape overflow")?;
	let program = crate::collective::Collective::row_sum(local, tree)?;
	let mut body = format!("{} float tmp[{local}];", shared(d));
	for op in program.linearize()? {
		match op {
			crate::collective::LocalOp::WritePartial => {
				body += &format!(
					"float acc=0.0f;for(size_t c=l;c<{cols};c+={local}){{acc+=b0[g*{cols}+c];}}tmp[l]=acc;"
				)
			}
			crate::collective::LocalOp::Barrier => body += barrier(d),
			crate::collective::LocalOp::Tree { stride } => {
				body += &format!("if(l<{stride}){{tmp[l]+=tmp[l+{stride}];}}")
			}
			crate::collective::LocalOp::SerialFinish => {
				body += &format!(
					"if(l==0){{float sum=0.0f;for(size_t c=0;c<{local};c++){{sum+=tmp[c];}}tmp[0]=sum;}}"
				)
			}
			crate::collective::LocalOp::StoreLaneZero => body += "if(l==0){b1[g]=tmp[0];}",
		}
	}
	spec(
		d,
		vec![len, rows],
		body,
		rows,
		local,
		local.checked_mul(4).ok_or("local byte overflow")?,
	)
}
```

```sh check
cargo test --test book_18 todo_18_004 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## Metal と OpenCL の両方に同じ IR を下げる。

<!-- todo:18-005 -->

実装は[この章の `Collective::verify、Collective::linearize、gpu_kernels::row_sum、two_stage_sum、dynamic_sum、GpuPlan::compile_parallel`](#local-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_18.rs mode=append
#[test]
fn todo_18_005() {
	suite0::barriers_and_shader_compile();
}
```

`src/gpu_kernels.rs` の `pub fn row_sum(`の定義を次の全体で置き換えます。

```rust file="src/gpu_kernels.rs" mode="replace" item="pub fn row_sum("
pub fn row_sum(
	d: Dialect,
	rows: usize,
	cols: usize,
	local: usize,
	tree: bool,
) -> Result<Kernel, String> {
	if !local.is_power_of_two() {
		return Err("reduction local must be a power of two".into());
	}
	let len = rows.checked_mul(cols).ok_or("shape overflow")?;
	let program = crate::collective::Collective::row_sum(local, tree)?;
	let mut body = format!("{} float tmp[{local}];", shared(d));
	for op in program.linearize()? {
		match op {
			crate::collective::LocalOp::WritePartial => {
				body += &format!(
					"float acc=0.0f;for(size_t c=l;c<{cols};c+={local}){{acc+=b0[g*{cols}+c];}}tmp[l]=acc;"
				)
			}
			crate::collective::LocalOp::Barrier => body += barrier(d),
			crate::collective::LocalOp::Tree { stride } => {
				body += &format!("if(l<{stride}){{tmp[l]+=tmp[l+{stride}];}}")
			}
			crate::collective::LocalOp::SerialFinish => {
				body += &format!(
					"if(l==0){{float sum=0.0f;for(size_t c=0;c<{local};c++){{sum+=tmp[c];}}tmp[0]=sum;}}"
				)
			}
			crate::collective::LocalOp::StoreLaneZero => body += "if(l==0){b1[g]=tmp[0];}",
		}
	}
	spec(
		d,
		vec![len, rows],
		body,
		rows,
		local,
		local.checked_mul(4).ok_or("local byte overflow")?,
	)
}
```

`src/gpu_kernels.rs` の `fn shared(`の定義を次の全体で置き換えます。

```rust file="src/gpu_kernels.rs" mode="replace" item="fn shared("
fn shared(d: Dialect) -> &'static str {
	match d {
		Dialect::Metal => "threadgroup",
		Dialect::OpenCl => "__local",
	}
}
```

`src/gpu_kernels.rs` の `fn barrier(`の定義を次の全体で置き換えます。

```rust file="src/gpu_kernels.rs" mode="replace" item="fn barrier("
fn barrier(d: Dialect) -> &'static str {
	match d {
		Dialect::Metal => "threadgroup_barrier(mem_flags::mem_threadgroup);",
		Dialect::OpenCl => "barrier(CLK_LOCAL_MEM_FENCE);",
	}
}
```

```sh check
cargo test --test book_18 todo_18_005 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## stride を L/2、L/4、…、1 として LOCAL の二要素を足す。

<!-- todo:18-006 -->

実装は[この章の `Collective::verify、Collective::linearize、gpu_kernels::row_sum、two_stage_sum、dynamic_sum、GpuPlan::compile_parallel`](#local-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_18.rs mode=append
#[test]
fn todo_18_006() {
	suite0::barriers_and_shader_compile();
}
```

`src/collective.rs` の `pub fn row_sum(`の定義を次の全体で置き換えます。

```rust file="src/collective.rs" mode="replace" item="pub fn row_sum("
	pub fn row_sum(lanes: usize, tree: bool) -> Result<Self> {
		if !lanes.is_power_of_two() {
			return Err("local must be power of two".into());
		}
		let mut ops = vec![LocalOp::WritePartial, LocalOp::Barrier];
		if tree {
			let mut stride = lanes / 2;
			while stride > 0 {
				ops.extend([LocalOp::Tree { stride }, LocalOp::Barrier]);
				stride /= 2;
			}
		} else {
			ops.push(LocalOp::SerialFinish);
		}
		ops.push(LocalOp::StoreLaneZero);
		let program = Self {
			lanes,
			local_elements: lanes,
			ops,
		};
		program.verify()?;
		Ok(program)
	}
```

`src/gpu_kernels.rs` の `pub fn row_sum(`の定義を次の全体で置き換えます。

```rust file="src/gpu_kernels.rs" mode="replace" item="pub fn row_sum("
pub fn row_sum(
	d: Dialect,
	rows: usize,
	cols: usize,
	local: usize,
	tree: bool,
) -> Result<Kernel, String> {
	if !local.is_power_of_two() {
		return Err("reduction local must be a power of two".into());
	}
	let len = rows.checked_mul(cols).ok_or("shape overflow")?;
	let program = crate::collective::Collective::row_sum(local, tree)?;
	let mut body = format!("{} float tmp[{local}];", shared(d));
	for op in program.linearize()? {
		match op {
			crate::collective::LocalOp::WritePartial => {
				body += &format!(
					"float acc=0.0f;for(size_t c=l;c<{cols};c+={local}){{acc+=b0[g*{cols}+c];}}tmp[l]=acc;"
				)
			}
			crate::collective::LocalOp::Barrier => body += barrier(d),
			crate::collective::LocalOp::Tree { stride } => {
				body += &format!("if(l<{stride}){{tmp[l]+=tmp[l+{stride}];}}")
			}
			crate::collective::LocalOp::SerialFinish => {
				body += &format!(
					"if(l==0){{float sum=0.0f;for(size_t c=0;c<{local};c++){{sum+=tmp[c];}}tmp[0]=sum;}}"
				)
			}
			crate::collective::LocalOp::StoreLaneZero => body += "if(l==0){b1[g]=tmp[0];}",
		}
	}
	spec(
		d,
		vec![len, rows],
		body,
		rows,
		local,
		local.checked_mul(4).ok_or("local byte overflow")?,
	)
}
```

```sh check
cargo test --test book_18 todo_18_006 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## `lane<stride` の分岐では加算と LOCAL STORE だけを行う。

<!-- todo:18-007 -->

実装は[この章の `Collective::verify、Collective::linearize、gpu_kernels::row_sum、two_stage_sum、dynamic_sum、GpuPlan::compile_parallel`](#local-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_18.rs mode=append
#[test]
fn todo_18_007() {
	suite0::barriers_and_shader_compile();
}
```

`src/gpu_kernels.rs` の `pub fn row_sum(`の定義を次の全体で置き換えます。

```rust file="src/gpu_kernels.rs" mode="replace" item="pub fn row_sum("
pub fn row_sum(
	d: Dialect,
	rows: usize,
	cols: usize,
	local: usize,
	tree: bool,
) -> Result<Kernel, String> {
	if !local.is_power_of_two() {
		return Err("reduction local must be a power of two".into());
	}
	let len = rows.checked_mul(cols).ok_or("shape overflow")?;
	let program = crate::collective::Collective::row_sum(local, tree)?;
	let mut body = format!("{} float tmp[{local}];", shared(d));
	for op in program.linearize()? {
		match op {
			crate::collective::LocalOp::WritePartial => {
				body += &format!(
					"float acc=0.0f;for(size_t c=l;c<{cols};c+={local}){{acc+=b0[g*{cols}+c];}}tmp[l]=acc;"
				)
			}
			crate::collective::LocalOp::Barrier => body += barrier(d),
			crate::collective::LocalOp::Tree { stride } => {
				body += &format!("if(l<{stride}){{tmp[l]+=tmp[l+{stride}];}}")
			}
			crate::collective::LocalOp::SerialFinish => {
				body += &format!(
					"if(l==0){{float sum=0.0f;for(size_t c=0;c<{local};c++){{sum+=tmp[c];}}tmp[0]=sum;}}"
				)
			}
			crate::collective::LocalOp::StoreLaneZero => body += "if(l==0){b1[g]=tmp[0];}",
		}
	}
	spec(
		d,
		vec![len, rows],
		body,
		rows,
		local,
		local.checked_mul(4).ok_or("local byte overflow")?,
	)
}
```

```sh check
cargo test --test book_18 todo_18_007 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 各 stride の後の barrier は全 lane が実行する位置に置く。

<!-- todo:18-008 -->

実装は[この章の `Collective::verify、Collective::linearize、gpu_kernels::row_sum、two_stage_sum、dynamic_sum、GpuPlan::compile_parallel`](#local-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_18.rs mode=append
#[test]
fn todo_18_008() {
	suite0::barriers_and_shader_compile();
}
```

`src/collective.rs` の `pub fn row_sum(`の定義を次の全体で置き換えます。

```rust file="src/collective.rs" mode="replace" item="pub fn row_sum("
	pub fn row_sum(lanes: usize, tree: bool) -> Result<Self> {
		if !lanes.is_power_of_two() {
			return Err("local must be power of two".into());
		}
		let mut ops = vec![LocalOp::WritePartial, LocalOp::Barrier];
		if tree {
			let mut stride = lanes / 2;
			while stride > 0 {
				ops.extend([LocalOp::Tree { stride }, LocalOp::Barrier]);
				stride /= 2;
			}
		} else {
			ops.push(LocalOp::SerialFinish);
		}
		ops.push(LocalOp::StoreLaneZero);
		let program = Self {
			lanes,
			local_elements: lanes,
			ops,
		};
		program.verify()?;
		Ok(program)
	}
```

`src/gpu_kernels.rs` の `pub fn row_sum(`の定義を次の全体で置き換えます。

```rust file="src/gpu_kernels.rs" mode="replace" item="pub fn row_sum("
pub fn row_sum(
	d: Dialect,
	rows: usize,
	cols: usize,
	local: usize,
	tree: bool,
) -> Result<Kernel, String> {
	if !local.is_power_of_two() {
		return Err("reduction local must be a power of two".into());
	}
	let len = rows.checked_mul(cols).ok_or("shape overflow")?;
	let program = crate::collective::Collective::row_sum(local, tree)?;
	let mut body = format!("{} float tmp[{local}];", shared(d));
	for op in program.linearize()? {
		match op {
			crate::collective::LocalOp::WritePartial => {
				body += &format!(
					"float acc=0.0f;for(size_t c=l;c<{cols};c+={local}){{acc+=b0[g*{cols}+c];}}tmp[l]=acc;"
				)
			}
			crate::collective::LocalOp::Barrier => body += barrier(d),
			crate::collective::LocalOp::Tree { stride } => {
				body += &format!("if(l<{stride}){{tmp[l]+=tmp[l+{stride}];}}")
			}
			crate::collective::LocalOp::SerialFinish => {
				body += &format!(
					"if(l==0){{float sum=0.0f;for(size_t c=0;c<{local};c++){{sum+=tmp[c];}}tmp[0]=sum;}}"
				)
			}
			crate::collective::LocalOp::StoreLaneZero => body += "if(l==0){b1[g]=tmp[0];}",
		}
	}
	spec(
		d,
		vec![len, rows],
		body,
		rows,
		local,
		local.checked_mul(4).ok_or("local byte overflow")?,
	)
}
```

```sh check
cargo test --test book_18 todo_18_008 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 最終結果の global STORE は lane 0 だけに限定する。

<!-- todo:18-009 -->

実装は[この章の `Collective::verify、Collective::linearize、gpu_kernels::row_sum、two_stage_sum、dynamic_sum、GpuPlan::compile_parallel`](#local-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_18.rs mode=append
#[test]
fn todo_18_009() {
	suite0::barriers_and_shader_compile();
}
```

`src/gpu_kernels.rs` の `pub fn row_sum(`の定義を次の全体で置き換えます。

```rust file="src/gpu_kernels.rs" mode="replace" item="pub fn row_sum("
pub fn row_sum(
	d: Dialect,
	rows: usize,
	cols: usize,
	local: usize,
	tree: bool,
) -> Result<Kernel, String> {
	if !local.is_power_of_two() {
		return Err("reduction local must be a power of two".into());
	}
	let len = rows.checked_mul(cols).ok_or("shape overflow")?;
	let program = crate::collective::Collective::row_sum(local, tree)?;
	let mut body = format!("{} float tmp[{local}];", shared(d));
	for op in program.linearize()? {
		match op {
			crate::collective::LocalOp::WritePartial => {
				body += &format!(
					"float acc=0.0f;for(size_t c=l;c<{cols};c+={local}){{acc+=b0[g*{cols}+c];}}tmp[l]=acc;"
				)
			}
			crate::collective::LocalOp::Barrier => body += barrier(d),
			crate::collective::LocalOp::Tree { stride } => {
				body += &format!("if(l<{stride}){{tmp[l]+=tmp[l+{stride}];}}")
			}
			crate::collective::LocalOp::SerialFinish => {
				body += &format!(
					"if(l==0){{float sum=0.0f;for(size_t c=0;c<{local};c++){{sum+=tmp[c];}}tmp[0]=sum;}}"
				)
			}
			crate::collective::LocalOp::StoreLaneZero => body += "if(l==0){b1[g]=tmp[0];}",
		}
	}
	spec(
		d,
		vec![len, rows],
		body,
		rows,
		local,
		local.checked_mul(4).ok_or("local byte overflow")?,
	)
}
```

```sh check
cargo test --test book_18 todo_18_009 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## LOCAL の read-after-write と write-after-read の依存を保つ。

<!-- todo:18-010 -->

実装は[この章の `Collective::verify、Collective::linearize、gpu_kernels::row_sum、two_stage_sum、dynamic_sum、GpuPlan::compile_parallel`](#local-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_18.rs mode=append
#[test]
fn todo_18_010() {
	suite0::barriers_and_shader_compile();
}
```

`src/collective.rs` の `pub fn verify(`の定義を次の全体で置き換えます。

```rust file="src/collective.rs" mode="replace" item="pub fn verify("
	pub fn verify(&self) -> Result<()> {
		if self.lanes == 0 || self.local_elements < self.lanes {
			return Err("LOCAL allocation shorter than lane stores".into());
		}
		let mut written = false;
		let mut visible = false;
		let mut final_store = false;
		for op in &self.ops {
			match *op {
				LocalOp::WritePartial => {
					if written && !visible {
						return Err("LOCAL reuse before barrier".into());
					}
					written = true;
					visible = false;
				}
				LocalOp::Barrier => {
					if !written {
						return Err("barrier without LOCAL write".into());
					}
					visible = true;
				}
				LocalOp::Tree { stride } => {
					if !visible
						|| stride == 0 || stride
						.checked_mul(2)
						.is_none_or(|n| n > self.local_elements)
					{
						return Err("LOCAL tree read before barrier or outside allocation".into());
					}
					visible = false;
				}
				LocalOp::SerialFinish => {
					if !visible {
						return Err("serial LOCAL read before barrier".into());
					}
				}
				LocalOp::StoreLaneZero => {
					if !visible {
						return Err("global store before LOCAL results visible".into());
					}
					final_store = true;
				}
			}
		}
		if !final_store {
			return Err("missing lane0 output".into());
		}
		Ok(())
	}
```

`src/collective.rs` の `pub fn row_sum(`の定義を次の全体で置き換えます。

```rust file="src/collective.rs" mode="replace" item="pub fn row_sum("
	pub fn row_sum(lanes: usize, tree: bool) -> Result<Self> {
		if !lanes.is_power_of_two() {
			return Err("local must be power of two".into());
		}
		let mut ops = vec![LocalOp::WritePartial, LocalOp::Barrier];
		if tree {
			let mut stride = lanes / 2;
			while stride > 0 {
				ops.extend([LocalOp::Tree { stride }, LocalOp::Barrier]);
				stride /= 2;
			}
		} else {
			ops.push(LocalOp::SerialFinish);
		}
		ops.push(LocalOp::StoreLaneZero);
		let program = Self {
			lanes,
			local_elements: lanes,
			ops,
		};
		program.verify()?;
		Ok(program)
	}
```

`src/gpu_kernels.rs` の `pub fn row_sum(`の定義を次の全体で置き換えます。

```rust file="src/gpu_kernels.rs" mode="replace" item="pub fn row_sum("
pub fn row_sum(
	d: Dialect,
	rows: usize,
	cols: usize,
	local: usize,
	tree: bool,
) -> Result<Kernel, String> {
	if !local.is_power_of_two() {
		return Err("reduction local must be a power of two".into());
	}
	let len = rows.checked_mul(cols).ok_or("shape overflow")?;
	let program = crate::collective::Collective::row_sum(local, tree)?;
	let mut body = format!("{} float tmp[{local}];", shared(d));
	for op in program.linearize()? {
		match op {
			crate::collective::LocalOp::WritePartial => {
				body += &format!(
					"float acc=0.0f;for(size_t c=l;c<{cols};c+={local}){{acc+=b0[g*{cols}+c];}}tmp[l]=acc;"
				)
			}
			crate::collective::LocalOp::Barrier => body += barrier(d),
			crate::collective::LocalOp::Tree { stride } => {
				body += &format!("if(l<{stride}){{tmp[l]+=tmp[l+{stride}];}}")
			}
			crate::collective::LocalOp::SerialFinish => {
				body += &format!(
					"if(l==0){{float sum=0.0f;for(size_t c=0;c<{local};c++){{sum+=tmp[c];}}tmp[0]=sum;}}"
				)
			}
			crate::collective::LocalOp::StoreLaneZero => body += "if(l==0){b1[g]=tmp[0];}",
		}
	}
	spec(
		d,
		vec![len, rows],
		body,
		rows,
		local,
		local.checked_mul(4).ok_or("local byte overflow")?,
	)
}
```

```sh check
cargo test --test book_18 todo_18_010 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## N=1、L-1、L、L+1、257 の部分和を同じ規則で作る。

<!-- todo:18-011 -->

実装は[この章の `Collective::verify、Collective::linearize、gpu_kernels::row_sum、two_stage_sum、dynamic_sum、GpuPlan::compile_parallel`](#local-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_18.rs mode=append
#[test]
fn todo_18_011() {
	suite0::barriers_and_shader_compile();
}
```

```sh check
cargo test --test book_18 todo_18_011 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 二次元の行和では group ID を row として使う。

<!-- todo:18-012 -->

実装は[この章の `Collective::verify、Collective::linearize、gpu_kernels::row_sum、two_stage_sum、dynamic_sum、GpuPlan::compile_parallel`](#local-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_18.rs mode=append
#[test]
fn todo_18_012() {
	suite0::barriers_and_shader_compile();
}
```

`src/gpu_kernels.rs` の `pub fn row_sum(`の定義を次の全体で置き換えます。

```rust file="src/gpu_kernels.rs" mode="replace" item="pub fn row_sum("
pub fn row_sum(
	d: Dialect,
	rows: usize,
	cols: usize,
	local: usize,
	tree: bool,
) -> Result<Kernel, String> {
	if !local.is_power_of_two() {
		return Err("reduction local must be a power of two".into());
	}
	let len = rows.checked_mul(cols).ok_or("shape overflow")?;
	let program = crate::collective::Collective::row_sum(local, tree)?;
	let mut body = format!("{} float tmp[{local}];", shared(d));
	for op in program.linearize()? {
		match op {
			crate::collective::LocalOp::WritePartial => {
				body += &format!(
					"float acc=0.0f;for(size_t c=l;c<{cols};c+={local}){{acc+=b0[g*{cols}+c];}}tmp[l]=acc;"
				)
			}
			crate::collective::LocalOp::Barrier => body += barrier(d),
			crate::collective::LocalOp::Tree { stride } => {
				body += &format!("if(l<{stride}){{tmp[l]+=tmp[l+{stride}];}}")
			}
			crate::collective::LocalOp::SerialFinish => {
				body += &format!(
					"if(l==0){{float sum=0.0f;for(size_t c=0;c<{local};c++){{sum+=tmp[c];}}tmp[0]=sum;}}"
				)
			}
			crate::collective::LocalOp::StoreLaneZero => body += "if(l==0){b1[g]=tmp[0];}",
		}
	}
	spec(
		d,
		vec![len, rows],
		body,
		rows,
		local,
		local.checked_mul(4).ok_or("local byte overflow")?,
	)
}
```

```sh check
cargo test --test book_18 todo_18_012 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## input の無効 LOAD と output の無効 STORE を別々に guard する。

<!-- todo:18-013 -->

実装は[この章の `Collective::verify、Collective::linearize、gpu_kernels::row_sum、two_stage_sum、dynamic_sum、GpuPlan::compile_parallel`](#local-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_18.rs mode=append
#[test]
fn todo_18_013() {
	suite0::barriers_and_shader_compile();
}
```

`src/gpu_kernels.rs` の `pub fn two_stage_sum(`の定義を次の全体で置き換えます。

```rust file="src/gpu_kernels.rs" mode="replace" item="pub fn two_stage_sum("
pub fn two_stage_sum(
	d: Dialect,
	n: usize,
	chunk: usize,
	local: usize,
) -> Result<ReductionSchedule, String> {
	if chunk == 0 {
		return Err("chunk=0".into());
	}
	let count = groups(n, chunk)?;
	let mut first = row_sum(d, count, chunk, local, true)?;
	first.lengths[0] = n;
	first.source = first.source.replace(
		&format!("acc+=b0[g*{chunk}+c];"),
		&format!("if(g*{chunk}+c<{n}){{acc+=b0[g*{chunk}+c];}}"),
	);
	Ok(ReductionSchedule {
		first,
		second: row_sum(d, 1, count, local, true)?,
	})
}
```

`src/gpu_kernels.rs` の `pub fn row_sum(`の定義を次の全体で置き換えます。

```rust file="src/gpu_kernels.rs" mode="replace" item="pub fn row_sum("
pub fn row_sum(
	d: Dialect,
	rows: usize,
	cols: usize,
	local: usize,
	tree: bool,
) -> Result<Kernel, String> {
	if !local.is_power_of_two() {
		return Err("reduction local must be a power of two".into());
	}
	let len = rows.checked_mul(cols).ok_or("shape overflow")?;
	let program = crate::collective::Collective::row_sum(local, tree)?;
	let mut body = format!("{} float tmp[{local}];", shared(d));
	for op in program.linearize()? {
		match op {
			crate::collective::LocalOp::WritePartial => {
				body += &format!(
					"float acc=0.0f;for(size_t c=l;c<{cols};c+={local}){{acc+=b0[g*{cols}+c];}}tmp[l]=acc;"
				)
			}
			crate::collective::LocalOp::Barrier => body += barrier(d),
			crate::collective::LocalOp::Tree { stride } => {
				body += &format!("if(l<{stride}){{tmp[l]+=tmp[l+{stride}];}}")
			}
			crate::collective::LocalOp::SerialFinish => {
				body += &format!(
					"if(l==0){{float sum=0.0f;for(size_t c=0;c<{local};c++){{sum+=tmp[c];}}tmp[0]=sum;}}"
				)
			}
			crate::collective::LocalOp::StoreLaneZero => body += "if(l==0){b1[g]=tmp[0];}",
		}
	}
	spec(
		d,
		vec![len, rows],
		body,
		rows,
		local,
		local.checked_mul(4).ok_or("local byte overflow")?,
	)
}
```

```sh check
cargo test --test book_18 todo_18_013 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## LOCAL を同じ kernel のループで再利用する場合、次の書込み前に barrier を入れる。

<!-- todo:18-014 -->

実装は[この章の `Collective::verify、Collective::linearize、gpu_kernels::row_sum、two_stage_sum、dynamic_sum、GpuPlan::compile_parallel`](#local-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_18.rs mode=append
#[test]
fn todo_18_014() {
	suite0::barriers_and_shader_compile();
}
```

`src/collective.rs` の `pub fn verify(`の定義を次の全体で置き換えます。

```rust file="src/collective.rs" mode="replace" item="pub fn verify("
	pub fn verify(&self) -> Result<()> {
		if self.lanes == 0 || self.local_elements < self.lanes {
			return Err("LOCAL allocation shorter than lane stores".into());
		}
		let mut written = false;
		let mut visible = false;
		let mut final_store = false;
		for op in &self.ops {
			match *op {
				LocalOp::WritePartial => {
					if written && !visible {
						return Err("LOCAL reuse before barrier".into());
					}
					written = true;
					visible = false;
				}
				LocalOp::Barrier => {
					if !written {
						return Err("barrier without LOCAL write".into());
					}
					visible = true;
				}
				LocalOp::Tree { stride } => {
					if !visible
						|| stride == 0 || stride
						.checked_mul(2)
						.is_none_or(|n| n > self.local_elements)
					{
						return Err("LOCAL tree read before barrier or outside allocation".into());
					}
					visible = false;
				}
				LocalOp::SerialFinish => {
					if !visible {
						return Err("serial LOCAL read before barrier".into());
					}
				}
				LocalOp::StoreLaneZero => {
					if !visible {
						return Err("global store before LOCAL results visible".into());
					}
					final_store = true;
				}
			}
		}
		if !final_store {
			return Err("missing lane0 output".into());
		}
		Ok(())
	}
```

`src/collective.rs` の `pub fn row_sum(`の定義を次の全体で置き換えます。

```rust file="src/collective.rs" mode="replace" item="pub fn row_sum("
	pub fn row_sum(lanes: usize, tree: bool) -> Result<Self> {
		if !lanes.is_power_of_two() {
			return Err("local must be power of two".into());
		}
		let mut ops = vec![LocalOp::WritePartial, LocalOp::Barrier];
		if tree {
			let mut stride = lanes / 2;
			while stride > 0 {
				ops.extend([LocalOp::Tree { stride }, LocalOp::Barrier]);
				stride /= 2;
			}
		} else {
			ops.push(LocalOp::SerialFinish);
		}
		ops.push(LocalOp::StoreLaneZero);
		let program = Self {
			lanes,
			local_elements: lanes,
			ops,
		};
		program.verify()?;
		Ok(program)
	}
```

`src/gpu_kernels.rs` の `pub fn row_sum(`の定義を次の全体で置き換えます。

```rust file="src/gpu_kernels.rs" mode="replace" item="pub fn row_sum("
pub fn row_sum(
	d: Dialect,
	rows: usize,
	cols: usize,
	local: usize,
	tree: bool,
) -> Result<Kernel, String> {
	if !local.is_power_of_two() {
		return Err("reduction local must be a power of two".into());
	}
	let len = rows.checked_mul(cols).ok_or("shape overflow")?;
	let program = crate::collective::Collective::row_sum(local, tree)?;
	let mut body = format!("{} float tmp[{local}];", shared(d));
	for op in program.linearize()? {
		match op {
			crate::collective::LocalOp::WritePartial => {
				body += &format!(
					"float acc=0.0f;for(size_t c=l;c<{cols};c+={local}){{acc+=b0[g*{cols}+c];}}tmp[l]=acc;"
				)
			}
			crate::collective::LocalOp::Barrier => body += barrier(d),
			crate::collective::LocalOp::Tree { stride } => {
				body += &format!("if(l<{stride}){{tmp[l]+=tmp[l+{stride}];}}")
			}
			crate::collective::LocalOp::SerialFinish => {
				body += &format!(
					"if(l==0){{float sum=0.0f;for(size_t c=0;c<{local};c++){{sum+=tmp[c];}}tmp[0]=sum;}}"
				)
			}
			crate::collective::LocalOp::StoreLaneZero => body += "if(l==0){b1[g]=tmp[0];}",
		}
	}
	spec(
		d,
		vec![len, rows],
		body,
		rows,
		local,
		local.checked_mul(4).ok_or("local byte overflow")?,
	)
}
```

```sh check
cargo test --test book_18 todo_18_014 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 非 2 冪の local を明示的に拒否し、無言で誤った tree を作らない。

<!-- todo:18-015 -->

実装は[この章の `Collective::verify、Collective::linearize、gpu_kernels::row_sum、two_stage_sum、dynamic_sum、GpuPlan::compile_parallel`](#local-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_18.rs mode=append
#[test]
fn todo_18_015() {
	suite0::barriers_and_shader_compile();
}
```

`src/gpu_kernels.rs` の `pub fn row_sum(`の定義を次の全体で置き換えます。

```rust file="src/gpu_kernels.rs" mode="replace" item="pub fn row_sum("
pub fn row_sum(
	d: Dialect,
	rows: usize,
	cols: usize,
	local: usize,
	tree: bool,
) -> Result<Kernel, String> {
	if !local.is_power_of_two() {
		return Err("reduction local must be a power of two".into());
	}
	let len = rows.checked_mul(cols).ok_or("shape overflow")?;
	let program = crate::collective::Collective::row_sum(local, tree)?;
	let mut body = format!("{} float tmp[{local}];", shared(d));
	for op in program.linearize()? {
		match op {
			crate::collective::LocalOp::WritePartial => {
				body += &format!(
					"float acc=0.0f;for(size_t c=l;c<{cols};c+={local}){{acc+=b0[g*{cols}+c];}}tmp[l]=acc;"
				)
			}
			crate::collective::LocalOp::Barrier => body += barrier(d),
			crate::collective::LocalOp::Tree { stride } => {
				body += &format!("if(l<{stride}){{tmp[l]+=tmp[l+{stride}];}}")
			}
			crate::collective::LocalOp::SerialFinish => {
				body += &format!(
					"if(l==0){{float sum=0.0f;for(size_t c=0;c<{local};c++){{sum+=tmp[c];}}tmp[0]=sum;}}"
				)
			}
			crate::collective::LocalOp::StoreLaneZero => body += "if(l==0){b1[g]=tmp[0];}",
		}
	}
	spec(
		d,
		vec![len, rows],
		body,
		rows,
		local,
		local.checked_mul(4).ok_or("local byte overflow")?,
	)
}
```

```sh check
cargo test --test book_18 todo_18_015 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 大きな入力を固定 chunk に分け、各 group の部分和を GLOBAL に保存する。

<!-- todo:18-016 -->

実装は[この章の `Collective::verify、Collective::linearize、gpu_kernels::row_sum、two_stage_sum、dynamic_sum、GpuPlan::compile_parallel`](#local-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_18.rs mode=append
#[test]
fn todo_18_016() {
	suite0::barriers_and_shader_compile();
}
```

`src/gpu_kernels.rs` の `pub fn two_stage_sum(`の定義を次の全体で置き換えます。

```rust file="src/gpu_kernels.rs" mode="replace" item="pub fn two_stage_sum("
pub fn two_stage_sum(
	d: Dialect,
	n: usize,
	chunk: usize,
	local: usize,
) -> Result<ReductionSchedule, String> {
	if chunk == 0 {
		return Err("chunk=0".into());
	}
	let count = groups(n, chunk)?;
	let mut first = row_sum(d, count, chunk, local, true)?;
	first.lengths[0] = n;
	first.source = first.source.replace(
		&format!("acc+=b0[g*{chunk}+c];"),
		&format!("if(g*{chunk}+c<{n}){{acc+=b0[g*{chunk}+c];}}"),
	);
	Ok(ReductionSchedule {
		first,
		second: row_sum(d, 1, count, local, true)?,
	})
}
```

```sh check
cargo test --test book_18 todo_18_016 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 部分和バッファを次 kernel の PARAM slot に bind する。

<!-- todo:18-017 -->

実装は[この章の `Collective::verify、Collective::linearize、gpu_kernels::row_sum、two_stage_sum、dynamic_sum、GpuPlan::compile_parallel`](#local-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_18.rs mode=append
#[test]
fn todo_18_017() {
	suite0::barriers_and_shader_compile();
}
```

`src/gpu_kernels.rs` の `pub fn two_stage_sum(`の定義を次の全体で置き換えます。

```rust file="src/gpu_kernels.rs" mode="replace" item="pub fn two_stage_sum("
pub fn two_stage_sum(
	d: Dialect,
	n: usize,
	chunk: usize,
	local: usize,
) -> Result<ReductionSchedule, String> {
	if chunk == 0 {
		return Err("chunk=0".into());
	}
	let count = groups(n, chunk)?;
	let mut first = row_sum(d, count, chunk, local, true)?;
	first.lengths[0] = n;
	first.source = first.source.replace(
		&format!("acc+=b0[g*{chunk}+c];"),
		&format!("if(g*{chunk}+c<{n}){{acc+=b0[g*{chunk}+c];}}"),
	);
	Ok(ReductionSchedule {
		first,
		second: row_sum(d, 1, count, local, true)?,
	})
}
```

`src/gpu_graph.rs` の `pub fn run_with_shapes(`の定義を次の全体で置き換えます。

```rust file="src/gpu_graph.rs" mode="replace" item="pub fn run_with_shapes("
	pub fn run_with_shapes(&self, inputs: &[Buffer], shapes: &[Vec<usize>]) -> Result<Vec<Buffer>> {
		if self.specialized && self.graph.nodes.iter().map(|n| &n.shape).ne(shapes.iter()) {
			return Err("optimized GPU plan shape mismatch".into());
		}
		if shapes.len() != self.graph.nodes.len() {
			return Err("GPU shape table".into());
		}
		let mut graph = self.graph.clone();
		for (node, shape) in graph.nodes.iter_mut().zip(shapes) {
			if node.shape.len() != shape.len() {
				return Err("GPU rank change".into());
			}
			elements(shape)?;
			node.shape = shape.clone();
		}
		graph.verify(&self.outputs)?;
		let mut logical_inputs = inputs.to_vec();
		for (id, node) in graph.nodes.iter().enumerate() {
			if let Op::Input { slot } = node.op {
				let n = elements(graph.shape(id))?;
				let input = inputs.get(slot).ok_or("missing GPU input")?;
				if input.shape.len() != graph.shape(id).len() || input.data.len() < n {
					return Err("GPU input rank/capacity".into());
				}
				logical_inputs[slot] =
					Buffer::new(graph.shape(id).to_vec(), input.data[..n].to_vec())?;
			}
		}
		graph.validate_inputs(&logical_inputs, &self.outputs)?;
		graph.validate_indices(&logical_inputs, &self.outputs)?;
		let mut values = graph
			.nodes
			.iter()
			.map(|n| Buffer::zeros(n.shape.clone()))
			.collect::<Result<Vec<_>>>()?;
		for id in graph.topology(&self.outputs)? {
			if let Op::Input { slot } = graph.nodes[id].op {
				values[id] = inputs[slot].clone();
			}
		}
		for step in &self.steps {
			let mut kernel = step.kernel.clone();
			kernel.lengths = step
				.slots
				.iter()
				.enumerate()
				.map(|(index, &id)| {
					if id == usize::MAX {
						step.kernel.lengths[index]
					} else {
						values[id].data.len()
					}
				})
				.collect();
			if !self.specialized {
				kernel.scalars = shapes.iter().flatten().map(|&n| n as u64).collect();
			}
			if !self.specialized {
				kernel.groups = groups(kernel.lengths[kernel.output], kernel.local)?;
			}
			if kernel.lengths[kernel.output] == 0 {
				continue;
			}
			let buffers = step
				.slots
				.iter()
				.enumerate()
				.map(|(index, &id)| {
					if id == usize::MAX {
						vec![1.; kernel.lengths[index]]
					} else {
						values[id].data.clone()
					}
				})
				.collect::<Vec<_>>();
			let run = step.program.run(&kernel, &buffers)?;
			values[*step.slots.last().unwrap()].data = run.output;
		}
		Ok(self.outputs.iter().map(|&id| values[id].clone()).collect())
	}
```

```sh check
cargo test --test book_18 todo_18_017 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 二つの kernel の依存を schedule に明示する。

<!-- todo:18-018 -->

実装は[この章の `Collective::verify、Collective::linearize、gpu_kernels::row_sum、two_stage_sum、dynamic_sum、GpuPlan::compile_parallel`](#local-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_18.rs mode=append
#[test]
fn todo_18_018() {
	suite0::barriers_and_shader_compile();
}
```

`src/gpu_kernels.rs` の `pub fn two_stage_sum(`の定義を次の全体で置き換えます。

```rust file="src/gpu_kernels.rs" mode="replace" item="pub fn two_stage_sum("
pub fn two_stage_sum(
	d: Dialect,
	n: usize,
	chunk: usize,
	local: usize,
) -> Result<ReductionSchedule, String> {
	if chunk == 0 {
		return Err("chunk=0".into());
	}
	let count = groups(n, chunk)?;
	let mut first = row_sum(d, count, chunk, local, true)?;
	first.lengths[0] = n;
	first.source = first.source.replace(
		&format!("acc+=b0[g*{chunk}+c];"),
		&format!("if(g*{chunk}+c<{n}){{acc+=b0[g*{chunk}+c];}}"),
	);
	Ok(ReductionSchedule {
		first,
		second: row_sum(d, 1, count, local, true)?,
	})
}
```

`src/gpu_graph.rs` の `fn from_lowered(`の定義を次の全体で置き換えます。

```rust file="src/gpu_graph.rs" mode="replace" item="fn from_lowered("
	fn from_lowered(
		graph: &Graph,
		outputs: &[usize],
		backend: Backend,
		lowered: Vec<(Kernel, Vec<usize>)>,
		specialized: bool,
	) -> Result<Self> {
		let mut steps: Vec<Step> = vec![];
		let mut sources = vec![];
		for (kernel, slots) in lowered {
			sources.push(kernel.source.clone());
			let program = if kernel.lengths[kernel.output] == 0 {
				Program::Unavailable
			} else if let Some(seed) = steps
				.iter()
				.find(|step| !matches!(step.program, Program::Unavailable))
			{
				seed.program.compile_next(&kernel)?
			} else {
				match backend {
					Backend::Metal => {
						#[cfg(all(target_os = "macos", feature = "metal"))]
						{
							Program::Metal(crate::backends::metal::MetalProgram::compile(&kernel)?)
						}
						#[cfg(not(all(target_os = "macos", feature = "metal")))]
						{
							return Err("Metal feature/macOS required".into());
						}
					}
					Backend::OpenCl => {
						#[cfg(feature = "opencl")]
						{
							Program::OpenCl(crate::backends::opencl::OpenClProgram::compile(
								&kernel, 0, 0,
							)?)
						}
						#[cfg(not(feature = "opencl"))]
						{
							return Err("OpenCL feature required".into());
						}
					}
				}
			};
			steps.push(Step {
				kernel,
				slots,
				program,
			});
		}
		Ok(Self {
			graph: graph.clone(),
			outputs: outputs.to_vec(),
			sources,
			specialized,
			steps,
		})
	}
```

```sh check
cargo test --test book_18 todo_18_018 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 同じ queue 上の順序と最後の読出し完了を runtime で保証する。

<!-- todo:18-019 -->

実装は[この章の `Collective::verify、Collective::linearize、gpu_kernels::row_sum、two_stage_sum、dynamic_sum、GpuPlan::compile_parallel`](#local-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_18.rs mode=append
#[test]
fn todo_18_019() {
	suite0::barriers_and_shader_compile();
}
```

`src/backends/metal.rs` の `pub fn compile_next(`の定義を次の全体で置き換えます。

```rust file="src/backends/metal.rs" mode="replace" item="pub fn compile_next("
	pub fn compile_next(&self, kernel: &Kernel) -> Result<Self, String> {
		Self::compile_on(kernel, self.device.clone(), self.queue.clone())
	}
```

`src/backends/opencl.rs` の `pub fn compile_next(`の定義を次の全体で置き換えます。

```rust file="src/backends/opencl.rs" mode="replace" item="pub fn compile_next("
	pub fn compile_next(&self, spec: &Spec) -> Result<Self, String> {
		Self::compile_on(
			spec,
			Device::new(self.device.id()),
			self.context.clone(),
			self.queue.clone(),
		)
	}
```

`src/backends/metal.rs` の `pub unsafe fn run(&self,`の定義を次の全体で置き換えます。

```rust file="src/backends/metal.rs" mode="replace" item="pub unsafe fn run(&self,"
	pub unsafe fn run(&self, kernel: &Kernel, inputs: &[Vec<f32>]) -> Result<Run, String> {
		kernel.validate(inputs)?;
		if self.contract
			!= (
				kernel.source.clone(),
				kernel.entry.clone(),
				kernel.lengths.len(),
				kernel.scalars.len().max(1),
				kernel.local,
			) {
			return Err("compiled GPU contract mismatch".into());
		}
		let total = Instant::now();
		let mut timings = Timings::default();
		if kernel.lengths[kernel.output] == 0 {
			return Ok(Run {
				output: vec![],
				device: self.device.name().to_string(),
				timings,
			});
		}
		let (_, max_threads, max_shared) = self.limits();
		if kernel.local > max_threads {
			return Err(format!(
				"Metal local {} exceeds {max_threads}",
				kernel.local
			));
		}
		if kernel.shared_bytes > max_shared
			|| self.pipeline.staticThreadgroupMemoryLength() > max_shared
		{
			return Err("Metal threadgroup memory limit".into());
		}
		if kernel.groups == 0 {
			return Err("nonempty output with zero groups".into());
		}
		autoreleasepool(|_| {
			let allocation = Instant::now();
			let buffers: Vec<_> = inputs
				.iter()
				.map(|input| {
					let bytes = input
						.len()
						.checked_mul(4)
						.ok_or("Metal byte overflow")?
						.max(4);
					if bytes > self.device.maxBufferLength() {
						return Err("Metal buffer limit".to_owned());
					}
					self.device
						.newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
						.ok_or_else(|| "Metal buffer allocation failed".to_owned())
				})
				.collect::<Result<_, _>>()?;
			let scalar_words = if kernel.scalars.is_empty() {
				vec![0u64]
			} else {
				kernel.scalars.clone()
			};
			let scalar_buffer = self
				.device
				.newBufferWithLength_options(
					scalar_words.len() * 8,
					MTLResourceOptions::StorageModeShared,
				)
				.ok_or("Metal scalar allocation")?;
			if kernel.source.contains("device const float4")
				&& buffers
					.iter()
					.any(|buffer| !(buffer.contents().as_ptr() as usize).is_multiple_of(16))
			{
				return Err("Metal vector buffer alignment".into());
			}
			timings.allocation = allocation.elapsed();
			let transfer = Instant::now();
			for (input, buffer) in inputs.iter().zip(&buffers) {
				unsafe {
					std::ptr::copy_nonoverlapping(
						input.as_ptr(),
						buffer.contents().as_ptr().cast::<f32>(),
						input.len(),
					);
				}
			}
			unsafe {
				std::ptr::copy_nonoverlapping(
					scalar_words.as_ptr(),
					scalar_buffer.contents().as_ptr().cast::<u64>(),
					scalar_words.len(),
				);
			}
			timings.transfer = transfer.elapsed();
			let command = self
				.queue
				.commandBuffer()
				.ok_or("Metal commandBuffer returned nil")?;
			let encoder = command
				.computeCommandEncoder()
				.ok_or("Metal encoder returned nil")?;
			encoder.setComputePipelineState(&self.pipeline);
			for (index, buffer) in buffers.iter().enumerate() {
				unsafe {
					encoder.setBuffer_offset_atIndex(Some(buffer), 0, index);
				}
			}
			unsafe {
				encoder.setBuffer_offset_atIndex(Some(&scalar_buffer), 0, buffers.len());
			}
			encoder.dispatchThreadgroups_threadsPerThreadgroup(
				MTLSize {
					width: kernel.groups,
					height: 1,
					depth: 1,
				},
				MTLSize {
					width: kernel.local,
					height: 1,
					depth: 1,
				},
			);
			encoder.endEncoding();
			command.commit();
			let synchronization = Instant::now();
			command.waitUntilCompleted();
			timings.synchronization = synchronization.elapsed();
			if command.status() != MTLCommandBufferStatus::Completed {
				return Err(format!("Metal command execution: {:?}", command.error()));
			}
			let elapsed = command.GPUEndTime() - command.GPUStartTime();
			if elapsed.is_finite() && elapsed >= 0.0 {
				timings.kernel = Some(Duration::from_secs_f64(elapsed));
			}
			let transfer = Instant::now();
			let output = unsafe {
				std::slice::from_raw_parts(
					buffers[kernel.output].contents().as_ptr().cast::<f32>(),
					kernel.lengths[kernel.output],
				)
			}
			.to_vec();
			timings.transfer += transfer.elapsed();
			timings.total = total.elapsed();
			Ok(Run {
				output,
				device: self.device.name().to_string(),
				timings,
			})
		})
	}
```

`src/backends/opencl.rs` の `pub unsafe fn run(&self,`の定義を次の全体で置き換えます。

```rust file="src/backends/opencl.rs" mode="replace" item="pub unsafe fn run(&self,"
	pub unsafe fn run(&self, spec: &Spec, inputs: &[Vec<f32>]) -> Result<Run, String> {
		spec.validate(inputs)?;
		if self.contract
			!= (
				spec.source.clone(),
				spec.entry.clone(),
				spec.lengths.len(),
				spec.scalars.len().max(1),
				spec.local,
			) {
			return Err("compiled GPU contract mismatch".into());
		}
		let _execution = self
			.execution
			.lock()
			.map_err(|_| "OpenCL execution mutex poisoned")?;
		let total = Instant::now();
		let mut timings = Timings::default();
		if spec.lengths[spec.output] == 0 {
			return Ok(Run {
				output: vec![],
				device: api("clGetDeviceInfo(name)", self.device.name())?,
				timings,
			});
		}
		let max = api(
			"clGetDeviceInfo(workgroup)",
			self.device.max_work_group_size(),
		)?
		.min(api(
			"clGetKernelWorkGroupInfo",
			self.kernel.get_work_group_size(self.device.id()),
		)?);
		let axes = api(
			"clGetDeviceInfo(workitems)",
			self.device.max_work_item_sizes(),
		)?;
		if spec.local > max || spec.local > axes[0] {
			return Err(format!("OpenCL local {} exceeds limit {max}", spec.local));
		}
		let local_mem = api("clGetDeviceInfo(local)", self.device.local_mem_size())?;
		if spec.shared_bytes as u64 > local_mem
			|| api(
				"clGetKernelWorkGroupInfo(local)",
				self.kernel.get_local_mem_size(self.device.id()),
			)? > local_mem
		{
			return Err("OpenCL local memory limit".into());
		}
		if spec.groups == 0 {
			return Err("nonempty output with zero groups".into());
		}
		let allocation = Instant::now();
		let mut buffers: Vec<_> = inputs
			.iter()
			.map(|b| unsafe {
				api(
					"clCreateBuffer",
					Buffer::<f32>::create(
						&self.context,
						CL_MEM_READ_WRITE,
						b.len().max(1),
						ptr::null_mut(),
					),
				)
			})
			.collect::<Result<_, _>>()?;
		let scalar_words = if spec.scalars.is_empty() {
			vec![0u64]
		} else {
			spec.scalars.clone()
		};
		let mut scalar_buffer = unsafe {
			api(
				"clCreateBuffer(scalars)",
				Buffer::<u64>::create(
					&self.context,
					CL_MEM_READ_WRITE,
					scalar_words.len(),
					ptr::null_mut(),
				),
			)?
		};
		timings.allocation = allocation.elapsed();
		let transfer = Instant::now();
		for (input, buffer) in inputs.iter().zip(&mut buffers) {
			if !input.is_empty() {
				unsafe {
					api(
						"clEnqueueWriteBuffer",
						self.queue
							.enqueue_write_buffer(buffer, CL_BLOCKING, 0, input, &[]),
					)?;
				}
			}
		}
		unsafe {
			api(
				"clEnqueueWriteBuffer(scalars)",
				self.queue.enqueue_write_buffer(
					&mut scalar_buffer,
					CL_BLOCKING,
					0,
					&scalar_words,
					&[],
				),
			)?;
			api(
				"clSetKernelArg(scalars)",
				self.kernel.set_arg(buffers.len() as u32, &scalar_buffer),
			)?;
		}
		timings.transfer = transfer.elapsed();
		for (index, buffer) in buffers.iter().enumerate() {
			unsafe {
				api("clSetKernelArg", self.kernel.set_arg(index as u32, buffer))?;
			}
		}
		let global = [spec
			.groups
			.checked_mul(spec.local)
			.ok_or("launch overflow")?];
		let local = [spec.local];
		let event = unsafe {
			api(
				"clEnqueueNDRangeKernel",
				self.queue.enqueue_nd_range_kernel(
					self.kernel.get(),
					1,
					ptr::null(),
					global.as_ptr(),
					local.as_ptr(),
					&[],
				),
			)?
		};
		let sync = Instant::now();
		api("clWaitForEvents", event.wait())?;
		timings.synchronization = sync.elapsed();
		if let (Ok(start), Ok(end)) = (
			event.profiling_command_start(),
			event.profiling_command_end(),
		) {
			timings.kernel = Some(Duration::from_nanos(end - start));
		}
		let transfer = Instant::now();
		let mut output = vec![0.; spec.lengths[spec.output]];
		unsafe {
			api(
				"clEnqueueReadBuffer",
				self.queue.enqueue_read_buffer(
					&buffers[spec.output],
					CL_BLOCKING,
					0,
					&mut output,
					&[],
				),
			)?;
		}
		timings.transfer += transfer.elapsed();
		timings.total = total.elapsed();
		Ok(Run {
			output,
			device: api("clGetDeviceInfo(name)", self.device.name())?,
			timings,
		})
	}
```

```sh check
cargo test --test book_18 todo_18_019 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 単一 workgroup の barrier を全 GPU の barrier として扱わない。

<!-- todo:18-020 -->

実装は[この章の `Collective::verify、Collective::linearize、gpu_kernels::row_sum、two_stage_sum、dynamic_sum、GpuPlan::compile_parallel`](#local-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_18.rs mode=append
#[test]
fn todo_18_020() {
	suite0::barriers_and_shader_compile();
}
```

```sh check
cargo test --test book_18 todo_18_020 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 整数として正確に表せる小例は、期待値と厳密に一致する。

<!-- todo:18-021 -->

実装は[この章の `Collective::verify、Collective::linearize、gpu_kernels::row_sum、two_stage_sum、dynamic_sum、GpuPlan::compile_parallel`](#local-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_18.rs mode=append
#[test]
fn todo_18_021() {
	suite0::barriers_and_shader_compile();
}
```

```sh check
cargo test --test book_18 todo_18_021 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 通常の f32 は加算順序の差を考慮し、`1e-5+1e-5*abs(reference)` を初期目標にする。

<!-- todo:18-022 -->

実装は[この章の `Collective::verify、Collective::linearize、gpu_kernels::row_sum、two_stage_sum、dynamic_sum、GpuPlan::compile_parallel`](#local-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_18.rs mode=append
#[test]
fn todo_18_022() {
	suite0::barriers_and_shader_compile();
}
```

```sh check
cargo test --test book_18 todo_18_022 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 正負が相殺する入力と大小が混在する入力も比較する。

<!-- todo:18-023 -->

実装は[この章の `Collective::verify、Collective::linearize、gpu_kernels::row_sum、two_stage_sum、dynamic_sum、GpuPlan::compile_parallel`](#local-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_18.rs mode=append
#[test]
fn todo_18_023() {
	suite0::barriers_and_shader_compile();
}
```

```sh check
cargo test --test book_18 todo_18_023 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## barrier を通る回数が lane の ID と入力の有効性で変わらない。

<!-- todo:18-024 -->

実装は[この章の `Collective::verify、Collective::linearize、gpu_kernels::row_sum、two_stage_sum、dynamic_sum、GpuPlan::compile_parallel`](#local-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_18.rs mode=append
#[test]
fn todo_18_024() {
	suite0::barriers_and_shader_compile();
}
```

```sh check
cargo test --test book_18 todo_18_024 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 同じ入力を繰り返して偶然の一回の成功に依存していないことを確認する。

<!-- todo:18-025 -->

実装は[この章の `Collective::verify、Collective::linearize、gpu_kernels::row_sum、two_stage_sum、dynamic_sum、GpuPlan::compile_parallel`](#local-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_18.rs mode=append
#[test]
fn todo_18_025() {
	suite0::barriers_and_shader_compile();
}
```

```sh check
cargo test --test book_18 todo_18_025 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 実機未実施の backend を数値一致済みとして報告しない。

<!-- todo:18-026 -->

実装は[この章の `Collective::verify、Collective::linearize、gpu_kernels::row_sum、two_stage_sum、dynamic_sum、GpuPlan::compile_parallel`](#local-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_18.rs mode=append
#[test]
fn todo_18_026() {
	suite0::barriers_and_shader_compile();
}
```

```sh check
cargo test --test book_18 todo_18_026 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 確認記録

```sh check
cargo test
```

GPU の source は CPU の numerical test と分けて扱う。上の CPU 成功は GPU の実機確認を意味しない。

