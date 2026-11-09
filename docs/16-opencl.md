# 16. OpenCL の公開 API で同じ Graph を実行する

この章は前章までの `chinigrad` に追加する。掲載順に `file=` のパスへ保存する。`mode=write` はそのファイル全体を置き換え、`mode=append` は末尾へ追記する。既存の別ファイルは残す。各コードは省略なしの実装である。

CPU の実行結果と GPU の実機結果を別に記録する。`requires=metal` と `requires=opencl` の確認は、その device がある環境で実行する。機器がなければ `NOT_RUN` と記録し、source 検査の成功へ置き換えない。

`Cargo.toml` を次の全体へ置き換える。Metal の設定を残し、OpenCL の任意依存を追加する。OpenCL loader は `opencl3` の既定の dynamic feature を使う。

```toml file=Cargo.toml mode=write
[package]
name = "chinigrad"
version = "0.1.0"
edition = "2024"
[features]
default = []
metal = ["dep:objc2", "dep:objc2-foundation", "dep:objc2-metal"]
opencl = ["dep:opencl3"]
[dependencies]
opencl3 = { version = "=0.12.3", optional = true }
[target.'cfg(target_os = "macos")'.dependencies]
objc2 = { version = "=0.6.4", optional = true }
objc2-foundation = { version = "=0.3.2", optional = true }
objc2-metal = { version = "=0.3.2", optional = true }
```

標準の数学関数を既定にする。`MathMode::NativeApprox` を明示した場合だけ `native_sqrt`、`native_sin`、`native_log2`、`native_exp2` を生成する。

<a id="opencl-implementation"></a>

## 実装を保存する

```rust file=src/backends/mod.rs mode=append
#[cfg(feature = "opencl")]
pub mod opencl;
```

```rust file=src/backends/opencl.rs mode=write
use crate::gpu::{Kernel as Spec, Run, Timings};
use opencl3::{
	command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue},
	context::Context,
	device::{CL_DEVICE_TYPE_ALL, Device},
	kernel::Kernel,
	memory::{Buffer, CL_MEM_READ_WRITE},
	platform::get_platforms,
	program::Program,
	types::CL_BLOCKING,
};
use std::{
	ptr,
	time::{Duration, Instant},
};
fn api<T>(name: &str, value: opencl3::Result<T>) -> Result<T, String> {
	value.map_err(|e| format!("{name}: {e} ({})", e.0))
}
#[derive(Debug)]
pub struct DeviceInfo {
	pub platform: usize,
	pub device: usize,
	pub name: String,
}
pub fn devices() -> Result<Vec<DeviceInfo>, String> {
	let mut out = vec![];
	for (p, platform) in api("clGetPlatformIDs", get_platforms())?.iter().enumerate() {
		for (d, id) in api("clGetDeviceIDs", platform.get_devices(CL_DEVICE_TYPE_ALL))?
			.into_iter()
			.enumerate()
		{
			out.push(DeviceInfo {
				platform: p,
				device: d,
				name: api("clGetDeviceInfo(name)", Device::new(id).name())?,
			});
		}
	}
	Ok(out)
}
pub struct OpenClProgram {
	device: Device,
	context: std::sync::Arc<Context>,
	queue: std::sync::Arc<CommandQueue>,
	kernel: Kernel,
	compile_time: Duration,
	contract: (String, String, usize, usize, usize),
	execution: std::sync::Mutex<()>,
}
impl OpenClProgram {
	pub fn compile(
		spec: &Spec,
		platform_index: usize,
		device_index: usize,
	) -> Result<Self, String> {
		let start = Instant::now();
		let platforms = api("clGetPlatformIDs", get_platforms())?;
		let platform = platforms
			.get(platform_index)
			.ok_or("OpenCL platform unavailable")?;
		let ids = api("clGetDeviceIDs", platform.get_devices(CL_DEVICE_TYPE_ALL))?;
		let id = *ids.get(device_index).ok_or("OpenCL device unavailable")?;
		let device = Device::new(id);
		let context = std::sync::Arc::new(api("clCreateContext", Context::from_device(&device))?);
		let queue = std::sync::Arc::new(api(
			"clCreateCommandQueue",
			CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE),
		)?);
		let mut program = Self::compile_on(spec, device, context, queue)?;
		program.compile_time = start.elapsed();
		Ok(program)
	}
	pub fn compile_next(&self, spec: &Spec) -> Result<Self, String> {
		Self::compile_on(
			spec,
			Device::new(self.device.id()),
			self.context.clone(),
			self.queue.clone(),
		)
	}
	fn compile_on(
		spec: &Spec,
		device: Device,
		context: std::sync::Arc<Context>,
		queue: std::sync::Arc<CommandQueue>,
	) -> Result<Self, String> {
		let start = Instant::now();
		let program = Program::create_and_build_from_source(&context, &spec.source, "")
			.map_err(|e| format!("clBuildProgram: {e}"))?;
		let kernel = api("clCreateKernel", Kernel::create(&program, &spec.entry))?;
		Ok(Self {
			device,
			context,
			queue,
			kernel,
			compile_time: start.elapsed(),
			contract: (
				spec.source.clone(),
				spec.entry.clone(),
				spec.lengths.len(),
				spec.scalars.len().max(1),
				spec.local,
			),
			execution: std::sync::Mutex::new(()),
		})
	}
	pub fn shares_queue(&self, other: &Self) -> bool {
		std::sync::Arc::ptr_eq(&self.queue, &other.queue)
			&& std::sync::Arc::ptr_eq(&self.context, &other.context)
	}
	pub fn compile_time(&self) -> Duration {
		self.compile_time
	}
	/// # Safety
	/// The source, entry, buffer/scalar ABI, and local size must match compilation.
	/// Every access made by the shader must fit the bound buffers for these scalar
	/// values and groups. The caller must prove indexing, aliasing, and collective
	/// barrier requirements. Buffer lengths alone cannot validate arbitrary shaders.
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
}
/// # Safety
/// The source, entry, buffer/scalar ABI, and local size must match compilation.
/// Every access made by the shader must fit the bound buffers for these scalar
/// values and groups. The caller must prove indexing, aliasing, and collective
/// barrier requirements. Buffer lengths alone cannot validate arbitrary shaders.
pub unsafe fn run(spec: &Spec, inputs: &[Vec<f32>]) -> Result<Run, String> {
	spec.validate(inputs)?;
	if spec.lengths[spec.output] == 0 {
		return Ok(Run {
			output: vec![],
			device: "OpenCL API not called (empty)".into(),
			timings: Timings::default(),
		});
	}
	let total = Instant::now();
	let program = OpenClProgram::compile(spec, 0, 0)?;
	let mut result = unsafe { program.run(spec, inputs) }?;
	result.timings.compile = program.compile_time();
	result.timings.total = total.elapsed();
	Ok(result)
}
```

```rust file=src/gpu_graph.rs mode=append
#[derive(Clone, Copy, Debug)]
pub enum MathMode {
	Precise,
	NativeApprox,
}
pub fn lower_math(
	graph: &Graph,
	outputs: &[usize],
	d: Dialect,
	local: usize,
	mode: MathMode,
) -> Result<Vec<(Kernel, Vec<usize>)>> {
	let mut kernels = lower(graph, outputs, d, local)?;
	if matches!(mode, MathMode::NativeApprox) {
		if d != Dialect::OpenCl {
			return Err("native math mode is OpenCL-only".into());
		}
		for (kernel, _) in &mut kernels {
			for name in ["sqrt", "sin", "log2", "exp2"] {
				kernel.source = kernel
					.source
					.replace(&format!("{name}("), &format!("native_{name}("));
			}
		}
	}
	Ok(kernels)
}
```

```rust file=tests/book_16.rs mode=write
use chinigrad::{
	Binary, Buffer, Graph, Reduce, Unary,
	gpu::{Dialect, groups},
	gpu_graph::{Backend, GpuPlan, MathMode, lower, lower_math},
};
fn source_contract() {
	assert_eq!(groups(257, 64).unwrap() * 64, 320);
	assert!(groups(1, 0).is_err());
	let mut g = Graph::default();
	let x = g.input(0, vec![2, 3]).unwrap();
	let p = g.pad(x, vec![(1, 1), (2, 1)]).unwrap();
	let s = g.unary(Unary::Sqrt, p).unwrap();
	let y = g.unary(Unary::Sin, s).unwrap();
	let z = g.unary(Unary::Log2, y).unwrap();
	let out = g.unary(Unary::Exp2, z).unwrap();
	let precise = lower(&g, &[out], Dialect::OpenCl, 64).unwrap();
	assert!(
		precise
			.iter()
			.any(|(k, _)| k.source.contains("get_group_id(0)"))
	);
	let source = precise
		.iter()
		.map(|(k, _)| k.source.as_str())
		.collect::<String>();
	assert!(source.contains("sqrt("));
	assert!(source.contains("sin("));
	assert!(!source.contains("native_"));
	let fast = lower_math(&g, &[out], Dialect::OpenCl, 64, MathMode::NativeApprox).unwrap();
	assert!(fast.iter().any(|(k, _)| k.source.contains("native_sqrt(")));
}
#[cfg(feature = "opencl")]
fn hardware_contract() {
	use chinigrad::backends::opencl::{OpenClProgram, devices};
	let inventory = devices().unwrap();
	assert!(!inventory.is_empty());
	let first = &inventory[0];
	let mut bad = chinigrad::gpu::add(Dialect::OpenCl, 4, 4).unwrap();
	bad.source = "invalid OpenCL".into();
	assert!(
		OpenClProgram::compile(&bad, first.platform, first.device)
			.err()
			.unwrap()
			.contains("clBuildProgram")
	);
	for n in [0, 1, 63, 64, 65, 257] {
		let mut g = Graph::default();
		let a = g.input(0, vec![n]).unwrap();
		let b = g.input(1, vec![n]).unwrap();
		let out = g.binary(Binary::Add, a, b).unwrap();
		let plan = GpuPlan::compile(&g, &[out], Backend::OpenCl, 64).unwrap();
		for value in [1., 7.] {
			let inputs = [
				Buffer::new(vec![n], vec![value; n]).unwrap(),
				Buffer::new(vec![n], vec![2.; n]).unwrap(),
			];
			assert_eq!(
				plan.run(&inputs).unwrap(),
				g.interpret(&[out], &inputs).unwrap()
			);
		}
	}
	let mut g = Graph::default();
	let a = g.input(0, vec![2, 3]).unwrap();
	let b = g.input(1, vec![3, 2]).unwrap();
	let z = g.matmul(a, b).unwrap();
	let padded = g.pad(a, vec![(1, 0), (1, 2)]).unwrap();
	let movement = g.permute(padded, vec![1, 0]).unwrap();
	let loss = g.reduce(z, Reduce::Sum, vec![0, 1], false).unwrap();
	let gradient = chinigrad::autodiff::gradient(&mut g, loss, &[b], None).unwrap()[0];
	let outputs = [z, movement, gradient];
	let inputs = [
		Buffer::new(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap(),
		Buffer::new(vec![3, 2], vec![1., 2., 3., 4., 5., 6.]).unwrap(),
	];
	assert_eq!(
		GpuPlan::compile(&g, &outputs, Backend::OpenCl, 64)
			.unwrap()
			.run(&inputs)
			.unwrap(),
		g.interpret(&outputs, &inputs).unwrap()
	);
}
```

```sh check
cargo check --features opencl
```

## platform と device を列挙し、明示指定できる入口を追加する。

<!-- todo:16-001 -->

実装は[この章の `gpu_graph::lower_math、OpenClProgram::compile、OpenClProgram::run、opencl::devices`](#opencl-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_16.rs mode=append
#[test]
#[cfg(feature = "opencl")]
#[ignore = "requires real device"]
fn todo_16_001() {
	hardware_contract();
}
```

```sh check requires=opencl
cargo test --test book_16 --features opencl todo_16_001 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## f32 加算の kernel source を共通 IR から生成する。

<!-- todo:16-002 -->

実装は[この章の `gpu_graph::lower_math、OpenClProgram::compile、OpenClProgram::run、opencl::devices`](#opencl-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_16.rs mode=append
#[test]
fn todo_16_002() {
	source_contract();
}
```

```sh check
cargo test --test book_16 todo_16_002 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## program build と kernel 作成、buffer の bind を実装する。

<!-- todo:16-003 -->

実装は[この章の `gpu_graph::lower_math、OpenClProgram::compile、OpenClProgram::run、opencl::devices`](#opencl-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_16.rs mode=append
#[test]
#[cfg(feature = "opencl")]
#[ignore = "requires real device"]
fn todo_16_003() {
	hardware_contract();
}
```

```sh check requires=opencl
cargo test --test book_16 --features opencl todo_16_003 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## 入力転送、NDRange、同期読出しを一巡させる。

<!-- todo:16-004 -->

実装は[この章の `gpu_graph::lower_math、OpenClProgram::compile、OpenClProgram::run、opencl::devices`](#opencl-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_16.rs mode=append
#[test]
#[cfg(feature = "opencl")]
#[ignore = "requires real device"]
fn todo_16_004() {
	hardware_contract();
}
```

```sh check requires=opencl
cargo test --test book_16 --features opencl todo_16_004 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## 各 OpenCL エラーに API 名と戻り値を含める。

<!-- todo:16-005 -->

実装は[この章の `gpu_graph::lower_math、OpenClProgram::compile、OpenClProgram::run、opencl::devices`](#opencl-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_16.rs mode=append
#[test]
#[cfg(feature = "opencl")]
#[ignore = "requires real device"]
fn todo_16_005() {
	hardware_contract();
}
```

```sh check requires=opencl
cargo test --test book_16 --features opencl todo_16_005 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## 共通 launch から global_work_size を計算する関数を作る。

<!-- todo:16-006 -->

実装は[この章の `gpu_graph::lower_math、OpenClProgram::compile、OpenClProgram::run、opencl::devices`](#opencl-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_16.rs mode=append
#[test]
fn todo_16_006() {
	source_contract();
}
```

```sh check
cargo test --test book_16 todo_16_006 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## N=257、local=64 では global=320 を渡す。

<!-- todo:16-007 -->

実装は[この章の `gpu_graph::lower_math、OpenClProgram::compile、OpenClProgram::run、opencl::devices`](#opencl-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_16.rs mode=append
#[test]
fn todo_16_007() {
	source_contract();
}
```

```sh check
cargo test --test book_16 todo_16_007 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## `i<N` の内側でだけ LOAD と STORE を実行する。

<!-- todo:16-008 -->

実装は[この章の `gpu_graph::lower_math、OpenClProgram::compile、OpenClProgram::run、opencl::devices`](#opencl-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_16.rs mode=append
#[test]
fn todo_16_008() {
	source_contract();
}
```

```sh check
cargo test --test book_16 todo_16_008 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## N=1、L-1、L、L+1 を CPU と比較する。

<!-- todo:16-009 -->

実装は[この章の `gpu_graph::lower_math、OpenClProgram::compile、OpenClProgram::run、opencl::devices`](#opencl-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_16.rs mode=append
#[test]
#[cfg(feature = "opencl")]
#[ignore = "requires real device"]
fn todo_16_009() {
	hardware_contract();
}
```

```sh check requires=opencl
cargo test --test book_16 --features opencl todo_16_009 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## N=0 は host 側で空出力を返し、ゼロサイズ NDRange を投入しない。

<!-- todo:16-010 -->

実装は[この章の `gpu_graph::lower_math、OpenClProgram::compile、OpenClProgram::run、opencl::devices`](#opencl-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_16.rs mode=append
#[test]
fn todo_16_010() {
	source_contract();
}
```

```sh check
cargo test --test book_16 todo_16_010 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## local size が機器の上限を越えたら起動前に拒否する。

<!-- todo:16-011 -->

実装は[この章の `gpu_graph::lower_math、OpenClProgram::compile、OpenClProgram::run、opencl::devices`](#opencl-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_16.rs mode=append
#[test]
#[cfg(feature = "opencl")]
#[ignore = "requires real device"]
fn todo_16_011() {
	hardware_contract();
}
```

```sh check requires=opencl
cargo test --test book_16 --features opencl todo_16_011 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## 二次元の平坦 ID を shape の座標へ戻す。

<!-- todo:16-012 -->

実装は[この章の `gpu_graph::lower_math、OpenClProgram::compile、OpenClProgram::run、opencl::devices`](#opencl-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_16.rs mode=append
#[test]
fn todo_16_012() {
	source_contract();
}
```

```sh check
cargo test --test book_16 todo_16_012 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## broadcast、転置後 reshape、shrink、pad を共通 lowering から生成する。

<!-- todo:16-013 -->

実装は[この章の `gpu_graph::lower_math、OpenClProgram::compile、OpenClProgram::run、opencl::devices`](#opencl-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_16.rs mode=append
#[test]
fn todo_16_013() {
	source_contract();
}
```

```sh check
cargo test --test book_16 todo_16_013 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## pad の LOAD 自体を有効条件の分岐内に置く。

<!-- todo:16-014 -->

実装は[この章の `gpu_graph::lower_math、OpenClProgram::compile、OpenClProgram::run、opencl::devices`](#opencl-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_16.rs mode=append
#[test]
fn todo_16_014() {
	source_contract();
}
```

```sh check
cargo test --test book_16 todo_16_014 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## SQRT、SIN、LOG2、EXP2 を OpenCL の通常の数学関数へ下げる。

<!-- todo:16-015 -->

実装は[この章の `gpu_graph::lower_math、OpenClProgram::compile、OpenClProgram::run、opencl::devices`](#opencl-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_16.rs mode=append
#[test]
fn todo_16_015() {
	source_contract();
}
```

```sh check
cargo test --test book_16 todo_16_015 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## native 高速近似関数は明示的な別モードにする。

<!-- todo:16-016 -->

実装は[この章の `gpu_graph::lower_math、OpenClProgram::compile、OpenClProgram::run、opencl::devices`](#opencl-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_16.rs mode=append
#[test]
fn todo_16_016() {
	source_contract();
}
```

```sh check
cargo test --test book_16 todo_16_016 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 一つの work item に一行の逐次 SUM を担当させる。

<!-- todo:16-017 -->

実装は[この章の `gpu_graph::lower_math、OpenClProgram::compile、OpenClProgram::run、opencl::devices`](#opencl-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_16.rs mode=append
#[test]
#[cfg(feature = "opencl")]
#[ignore = "requires real device"]
fn todo_16_017() {
	hardware_contract();
}
```

```sh check requires=opencl
cargo test --test book_16 --features opencl todo_16_017 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## matmul の出力 `(m,n)` を一つの work item に割り当てる。

<!-- todo:16-018 -->

実装は[この章の `gpu_graph::lower_math、OpenClProgram::compile、OpenClProgram::run、opencl::devices`](#opencl-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_16.rs mode=append
#[test]
#[cfg(feature = "opencl")]
#[ignore = "requires real device"]
fn todo_16_018() {
	hardware_contract();
}
```

```sh check requires=opencl
cargo test --test book_16 --features opencl todo_16_018 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## reduction の累積を work item ごとに初期化する。

<!-- todo:16-019 -->

実装は[この章の `gpu_graph::lower_math、OpenClProgram::compile、OpenClProgram::run、opencl::devices`](#opencl-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_16.rs mode=append
#[test]
#[cfg(feature = "opencl")]
#[ignore = "requires real device"]
fn todo_16_019() {
	hardware_contract();
}
```

```sh check requires=opencl
cargo test --test book_16 --features opencl todo_16_019 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## CPU と OpenCL の出力、勾配を小さい固定値で比較する。

<!-- todo:16-020 -->

実装は[この章の `gpu_graph::lower_math、OpenClProgram::compile、OpenClProgram::run、opencl::devices`](#opencl-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_16.rs mode=append
#[test]
#[cfg(feature = "opencl")]
#[ignore = "requires real device"]
fn todo_16_020() {
	hardware_contract();
}
```

```sh check requires=opencl
cargo test --test book_16 --features opencl todo_16_020 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## 新しい重みを同じ Plan の入力 slot に渡して再実行する。

<!-- todo:16-021 -->

実装は[この章の `gpu_graph::lower_math、OpenClProgram::compile、OpenClProgram::run、opencl::devices`](#opencl-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_16.rs mode=append
#[test]
#[cfg(feature = "opencl")]
#[ignore = "requires real device"]
fn todo_16_021() {
	hardware_contract();
}
```

```sh check requires=opencl
cargo test --test book_16 --features opencl todo_16_021 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## in-order queue で前の転送と kernel の順序が守られる。

<!-- todo:16-022 -->

実装は[この章の `gpu_graph::lower_math、OpenClProgram::compile、OpenClProgram::run、opencl::devices`](#opencl-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_16.rs mode=append
#[test]
#[cfg(feature = "opencl")]
#[ignore = "requires real device"]
fn todo_16_022() {
	hardware_contract();
}
```

```sh check requires=opencl
cargo test --test book_16 --features opencl todo_16_022 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## 非同期 read を開始しただけの値を、完了した値として返さない。

<!-- todo:16-023 -->

実装は[この章の `gpu_graph::lower_math、OpenClProgram::compile、OpenClProgram::run、opencl::devices`](#opencl-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_16.rs mode=append
#[test]
#[cfg(feature = "opencl")]
#[ignore = "requires real device"]
fn todo_16_023() {
	hardware_contract();
}
```

```sh check requires=opencl
cargo test --test book_16 --features opencl todo_16_023 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## device の不在、build 失敗、起動失敗を別々の診断にする。

<!-- todo:16-024 -->

実装は[この章の `gpu_graph::lower_math、OpenClProgram::compile、OpenClProgram::run、opencl::devices`](#opencl-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_16.rs mode=append
#[test]
#[cfg(feature = "opencl")]
#[ignore = "requires real device"]
fn todo_16_024() {
	hardware_contract();
}
```

```sh check requires=opencl
cargo test --test book_16 --features opencl todo_16_024 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## 機器がないときは source 検査と未実施の実機項目を分けて記録する。

<!-- todo:16-025 -->

実装は[この章の `gpu_graph::lower_math、OpenClProgram::compile、OpenClProgram::run、opencl::devices`](#opencl-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_16.rs mode=append
#[test]
fn todo_16_025() {
	source_contract();
}
```

```sh check
cargo test --test book_16 todo_16_025 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## runtime の API 呼出し以外に OpenCL 固有のロジックを広げない。

<!-- todo:16-026 -->

実装は[この章の `gpu_graph::lower_math、OpenClProgram::compile、OpenClProgram::run、opencl::devices`](#opencl-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_16.rs mode=append
#[test]
fn todo_16_026() {
	source_contract();
}
```

```sh check
cargo test --test book_16 todo_16_026 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 空の加算をテストし、NDRange の投入なしで空の結果を得る。

<!-- todo:16-027 -->

実装は[この章の `gpu_graph::lower_math、OpenClProgram::compile、OpenClProgram::run、opencl::devices`](#opencl-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_16.rs mode=append
#[test]
fn todo_16_027() {
	source_contract();
}
```

```sh check
cargo test --test book_16 todo_16_027 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 確認記録

```sh check
cargo test
```

