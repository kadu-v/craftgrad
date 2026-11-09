# 18. LOCAL と一様な barrier で並列縮約する

この章は前章までの `chinigrad` に追加する。掲載順に `file=` のパスへ保存する。`mode=write` はそのファイル全体を置き換え、`mode=append` は末尾へ追記する。既存の別ファイルは残す。各コードは省略なしの実装である。

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
use chinigrad::{
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
use chinigrad::{
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
	let seed = chinigrad::backends::metal::MetalProgram::compile(
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
	let seed = chinigrad::backends::opencl::OpenClProgram::compile(
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
	use chinigrad::{
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

