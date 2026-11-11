# 15. Metal の公開 API で実際の Graph を実行する

この章は前章までの `chibigrad` に追加する。掲載順に `file=` のパスへ保存する。`mode=write` はそのファイル全体を置き換え、`mode=append` は末尾へ追記する。既存の別ファイルは残す。各コードは省略なしの実装である。

CPU の実行結果と GPU の実機結果を別に記録する。`requires=metal` と `requires=opencl` の確認は、その device がある環境で実行する。機器がなければ `NOT_RUN` と記録し、source 検査の成功へ置き換えない。

生の shader を受け取る backend の `run` は `unsafe` とする。任意の source の buffer 範囲は長さだけでは証明できない。通常は検証済み Graph から作った安全な `GpuPlan` を使う。低水準の実機 test では、この章で生成した shader と同じ shape、scalar、launch を用意してから `unsafe` block で起動する。buffer の論理 slot と scalar の論理 slot を分ける。GPU は f32 pointer 引数の後へ `constant ulong *s` を bind する。`GpuPlan` は `graph::kernel_bodies` と同じ座標変換を使い、外側の出力 loop だけを group と lane に置き換える。pad の LOAD は有効条件の分岐内に残る。GPU の gather と scatter は Input、Const、movement で作った整数添字を扱う。添字軸が 16777216 を超える場合と、数学関数で添字を計算する場合は compile 前に拒否する。

`Cargo.toml` を次の全体へ置き換える。Metal の binding は macOS と feature の両方で制限する。依存の version を固定する。

```toml
# file=Cargo.toml mode=write
[package]
name = "chibigrad"
version = "0.1.0"
edition = "2024"
[features]
default = []
metal = ["dep:objc2", "dep:objc2-foundation", "dep:objc2-metal"]
[dependencies]
[target.'cfg(target_os = "macos")'.dependencies]
objc2 = { version = "=0.6.4", optional = true }
objc2-foundation = { version = "=0.3.2", optional = true }
objc2-metal = { version = "=0.3.2", optional = true }
```

<a id="metal-implementation"></a>

## 実装を保存する

```rust
// file=src/lib.rs mode=append
pub mod backends;
pub mod gpu;
pub mod gpu_graph;
```

```rust
// file=src/backends/mod.rs mode=write
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metal;
```

```rust
// file=src/gpu.rs mode=write
use std::time::Duration;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dialect {
	Metal,
	OpenCl,
}
#[derive(Clone, Debug)]
pub struct Kernel {
	pub source: String,
	pub scalars: Vec<u64>,
	pub entry: String,
	pub lengths: Vec<usize>,
	pub output: usize,
	pub groups: usize,
	pub local: usize,
	pub shared_bytes: usize,
}
impl Kernel {
	pub fn validate(&self, buffers: &[Vec<f32>]) -> Result<(), String> {
		if self.local == 0 || self.local > u32::MAX as usize {
			return Err("local size".into());
		}
		self.groups
			.checked_mul(self.local)
			.ok_or("launch overflow")?;
		if self.groups > u32::MAX as usize {
			return Err("group ID overflow".into());
		}
		if self.output >= self.lengths.len() || buffers.len() != self.lengths.len() {
			return Err("buffer layout".into());
		}
		for (&len, data) in self.lengths.iter().zip(buffers) {
			if len != data.len() {
				return Err("buffer shape".into());
			}
			len.checked_mul(4).ok_or("buffer byte overflow")?;
		}
		Ok(())
	}
}
#[derive(Clone, Debug, Default)]
pub struct Timings {
	pub compile: Duration,
	pub allocation: Duration,
	pub transfer: Duration,
	pub kernel: Option<Duration>,
	pub synchronization: Duration,
	pub total: Duration,
}
#[derive(Clone, Debug)]
pub struct Run {
	pub output: Vec<f32>,
	pub device: String,
	pub timings: Timings,
}
pub fn groups(n: usize, local: usize) -> Result<usize, String> {
	if local == 0 {
		return Err("zero local size".into());
	}
	let groups = n / local + usize::from(!n.is_multiple_of(local));
	groups.checked_mul(local).ok_or("launch overflow")?;
	Ok(groups)
}
pub fn header(d: Dialect, count: usize, body: &str) -> String {
	let args = (0..count)
		.map(|i| match d {
			Dialect::Metal => format!("device float *b{i} [[buffer({i})]]"),
			Dialect::OpenCl => format!("__global float *b{i}"),
		})
		.collect::<Vec<_>>()
		.join(",");
	match d {
		Dialect::Metal => format!(
			"#include <metal_stdlib>\nusing namespace metal;\nkernel void chibi({args}, constant ulong *s [[buffer({count})]], uint group_id [[threadgroup_position_in_grid]], uint local_id [[thread_position_in_threadgroup]]){{size_t g=group_id,l=local_id;{body}}}"
		),
		Dialect::OpenCl => format!(
			"__kernel void chibi({args}, __constant ulong *s){{size_t g=get_group_id(0), l=get_local_id(0);{body}}}"
		),
	}
}
pub fn add(d: Dialect, n: usize, local: usize) -> Result<Kernel, String> {
	let body = format!("size_t i=g*{local}+l;if(i<{n}){{b2[i]=b0[i]+b1[i];}}");
	Ok(Kernel {
		scalars: vec![],
		source: header(d, 3, &body),
		entry: "chibi".into(),
		lengths: vec![n, n, n],
		output: 2,
		groups: groups(n, local)?,
		local,
		shared_bytes: 0,
	})
}
```

```rust
// file=src/backends/metal.rs mode=write
use crate::gpu::{Kernel, Run, Timings};
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use std::time::{Duration, Instant};
type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub struct MetalProgram {
	device: Device,
	queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
	pipeline: Pipeline,
	compile_time: Duration,
	contract: (String, String, usize, usize, usize),
}
impl MetalProgram {
	pub fn compile(kernel: &Kernel) -> Result<Self, String> {
		let device = MTLCreateSystemDefaultDevice().ok_or("Metal device unavailable")?;
		let queue = device
			.newCommandQueue()
			.ok_or("Metal newCommandQueue returned nil")?;
		Self::compile_on(kernel, device, queue)
	}
	pub fn compile_next(&self, kernel: &Kernel) -> Result<Self, String> {
		Self::compile_on(kernel, self.device.clone(), self.queue.clone())
	}
	fn compile_on(
		kernel: &Kernel,
		device: Device,
		queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
	) -> Result<Self, String> {
		autoreleasepool(|_| {
			let start = Instant::now();
			let options = MTLCompileOptions::new();
			#[allow(deprecated)]
			options.setFastMathEnabled(false);
			let library = device
				.newLibraryWithSource_options_error(
					&NSString::from_str(&kernel.source),
					Some(&options),
				)
				.map_err(|e| format!("Metal source build: {e}"))?;
			let function = library
				.newFunctionWithName(&NSString::from_str(&kernel.entry))
				.ok_or("Metal entry missing")?;
			let pipeline = device
				.newComputePipelineStateWithFunction_error(&function)
				.map_err(|e| format!("Metal pipeline build: {e}"))?;
			Ok(Self {
				device,
				queue,
				pipeline,
				compile_time: start.elapsed(),
				contract: (
					kernel.source.clone(),
					kernel.entry.clone(),
					kernel.lengths.len(),
					kernel.scalars.len().max(1),
					kernel.local,
				),
			})
		})
	}
	pub fn families(&self) -> Vec<String> {
		let known = [
			("Apple1", MTLGPUFamily::Apple1),
			("Apple2", MTLGPUFamily::Apple2),
			("Apple3", MTLGPUFamily::Apple3),
			("Apple4", MTLGPUFamily::Apple4),
			("Apple5", MTLGPUFamily::Apple5),
			("Apple6", MTLGPUFamily::Apple6),
			("Apple7", MTLGPUFamily::Apple7),
			("Apple8", MTLGPUFamily::Apple8),
			("Apple9", MTLGPUFamily::Apple9),
			("Mac2", MTLGPUFamily::Mac2),
		];
		known
			.into_iter()
			.filter(|(_, f)| self.device.supportsFamily(*f))
			.map(|(name, _)| name.to_owned())
			.collect()
	}
	pub fn shares_queue(&self, other: &Self) -> bool {
		std::ptr::addr_eq(&*self.queue, &*other.queue)
	}
	pub fn compile_time(&self) -> Duration {
		self.compile_time
	}
	pub fn limits(&self) -> (usize, usize, usize) {
		(
			self.pipeline.threadExecutionWidth(),
			self.pipeline
				.maxTotalThreadsPerThreadgroup()
				.min(self.device.maxThreadsPerThreadgroup().width),
			self.device.maxThreadgroupMemoryLength(),
		)
	}
	/// # Safety
	/// The source, entry, buffer/scalar ABI, and local size must match compilation.
	/// Every access made by the shader must fit the bound buffers for these scalar
	/// values and groups. The caller must prove indexing, aliasing, and collective
	/// barrier requirements. Buffer lengths alone cannot validate arbitrary shaders.
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
}
/// # Safety
/// The source, entry, buffer/scalar ABI, and local size must match compilation.
/// Every access made by the shader must fit the bound buffers for these scalar
/// values and groups. The caller must prove indexing, aliasing, and collective
/// barrier requirements. Buffer lengths alone cannot validate arbitrary shaders.
pub unsafe fn run(kernel: &Kernel, inputs: &[Vec<f32>]) -> Result<Run, String> {
	kernel.validate(inputs)?;
	if kernel.lengths[kernel.output] == 0 {
		return Ok(Run {
			output: vec![],
			device: "Metal API not called (empty)".into(),
			timings: Timings::default(),
		});
	}
	let total = Instant::now();
	let program = MetalProgram::compile(kernel)?;
	let mut result = unsafe { program.run(kernel, inputs) }?;
	result.timings.compile = program.compile_time();
	result.timings.total = total.elapsed();
	Ok(result)
}
```

```rust
// file=src/gpu_graph.rs mode=write
use crate::{
	Buffer, Graph, Op, Result,
	gpu::{Dialect, Kernel, groups, header},
	graph::kernel_bodies,
	runtime::elements,
};
#[derive(Clone, Copy, Debug)]
pub enum Backend {
	Metal,
	OpenCl,
}
impl Backend {
	pub fn dialect(self) -> Dialect {
		match self {
			Self::Metal => Dialect::Metal,
			Self::OpenCl => Dialect::OpenCl,
		}
	}
}
struct Step {
	kernel: Kernel,
	slots: Vec<usize>,
	program: Program,
}
enum Program {
	#[cfg(all(target_os = "macos", feature = "metal"))]
	Metal(crate::backends::metal::MetalProgram),
	#[cfg(feature = "opencl")]
	OpenCl(crate::backends::opencl::OpenClProgram),
	Unavailable,
}
impl Program {
	fn compile_next(&self, kernel: &Kernel) -> Result<Self> {
		match self {
			#[cfg(all(target_os = "macos", feature = "metal"))]
			Self::Metal(program) => Ok(Self::Metal(program.compile_next(kernel)?)),
			#[cfg(feature = "opencl")]
			Self::OpenCl(program) => Ok(Self::OpenCl(program.compile_next(kernel)?)),
			Self::Unavailable => {
				let _ = kernel;
				Err("missing initial GPU queue".into())
			}
		}
	}
	fn run(&self, kernel: &Kernel, buffers: &[Vec<f32>]) -> Result<crate::gpu::Run> {
		match self {
			#[cfg(all(target_os = "macos", feature = "metal"))]
			Self::Metal(p) => unsafe { p.run(kernel, buffers) },
			#[cfg(feature = "opencl")]
			Self::OpenCl(p) => unsafe { p.run(kernel, buffers) },
			Self::Unavailable => {
				let _ = (kernel, buffers);
				Err("GPU program unavailable".into())
			}
		}
	}
}
pub struct GpuPlan {
	graph: Graph,
	outputs: Vec<usize>,
	pub sources: Vec<String>,
	specialized: bool,
	steps: Vec<Step>,
}
fn helpers(d: Dialect) -> String {
	let bits = match d {
		Dialect::Metal => "as_type<float>(x)",
		Dialect::OpenCl => "as_float(x)",
	};
	format!(
		"inline float chibi_bits(uint x){{return {bits};}}\ninline float chibi_max(float a,float b){{if(isnan(a)||isnan(b))return NAN;if(a==0.0f&&b==0.0f)return 0.0f;return a>b?a:b;}}\n"
	)
}
pub fn lower(
	graph: &Graph,
	outputs: &[usize],
	d: Dialect,
	local: usize,
) -> Result<Vec<(Kernel, Vec<usize>)>> {
	graph.verify(outputs)?;
	for id in graph.topology(outputs)? {
		if let Op::Gather { x, indices, axis }
		| Op::ScatterAdd {
			base: x,
			indices,
			axis,
			..
		} = graph.nodes[id].op
		{
			if graph.shape(x)[axis] > 16_777_216 {
				return Err("GPU f32 index extent exceeds exact integer range".into());
			}
			for index_node in graph.topology(&[indices])? {
				if !matches!(
					graph.nodes[index_node].op,
					Op::Input { .. }
						| Op::Const { .. } | Op::Reshape { .. }
						| Op::Permute { .. }
						| Op::Expand { .. }
						| Op::Shrink { .. }
						| Op::Pad { .. } | Op::Flip { .. }
						| Op::Stack { .. } | Op::Stage { .. }
						| Op::StopGradient { .. }
				) {
					return Err("GPU computed index arithmetic is unsupported; materialize integer indices as an input".into());
				}
			}
		}
	}
	let kernels = kernel_bodies(graph, outputs, true)?;
	let dims = graph
		.nodes
		.iter()
		.flat_map(|n| n.shape.iter().map(|&v| v as u64))
		.collect::<Vec<_>>();
	let mut result = vec![];
	for k in kernels {
		let n = elements(graph.shape(k.output))?;
		let start = k.body.find('{').ok_or("missing outer loop")? + 1;
		let end = k.body.rfind('}').ok_or("missing loop end")?;
		let mut body = k.body[start..end].to_string();

		for (from, to) in [
			("expf(", "exp("),
			("logf(", "log("),
			("sinf(", "sin("),
			("cosf(", "cos("),
			("sqrtf(", "sqrt("),
			("exp2f(", "exp2("),
			("log2f(", "log2("),
			("floorf(", "floor("),
			("(double)", "(float)"),
			("uint32_t", "uint"),
		] {
			body = body.replace(from, to);
		}
		body = body.replace("return 2;", "return;");
		let mut slots = graph
			.topology(&[k.output])?
			.into_iter()
			.filter(|id| body.contains(&format!("b{id}[")))
			.collect::<Vec<_>>();
		slots.retain(|&id| id != k.output);
		slots.push(k.output);
		for (index, id) in slots.iter().enumerate() {
			body = body.replace(&format!("b{id}["), &format!("slot{index}["));
		}
		for index in 0..slots.len() {
			body = body.replace(&format!("slot{index}["), &format!("b{index}["));
		}
		let condition = k
			.body
			.split("i<")
			.nth(1)
			.and_then(|s| s.split("; ++i)").next())
			.ok_or("outer loop condition")?;
		let body = format!("size_t i=g*{local}+l;if(i<{condition}){{{body}}}");
		let mut source = header(d, slots.len(), &body);
		let helper = helpers(d);
		source = match d {
			Dialect::Metal => source.replacen("kernel void", &(helper + "kernel void"), 1),
			Dialect::OpenCl => helper + &source,
		};
		let lengths = slots
			.iter()
			.map(|&id| elements(graph.shape(id)))
			.collect::<Result<Vec<_>>>()?;
		result.push((
			Kernel {
				scalars: dims.clone(),
				source,
				entry: "chibi".into(),
				output: slots.len() - 1,
				lengths,
				groups: groups(n, local)?,
				local,
				shared_bytes: 0,
			},
			slots,
		));
	}
	Ok(result)
}
impl GpuPlan {
	pub fn compile(
		graph: &Graph,
		outputs: &[usize],
		backend: Backend,
		local: usize,
	) -> Result<Self> {
		let lowered = lower(graph, outputs, backend.dialect(), local)?;
		Self::from_lowered(graph, outputs, backend, lowered, false)
	}
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
	pub fn run(&self, inputs: &[Buffer]) -> Result<Vec<Buffer>> {
		self.run_with_shapes(
			inputs,
			&self
				.graph
				.nodes
				.iter()
				.map(|n| n.shape.clone())
				.collect::<Vec<_>>(),
		)
	}
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
}
```

```rust
// file=tests/gpu_graph.rs mode=write
use chibigrad::{
	Binary, Buffer, Graph, Reduce,
	gpu::Dialect,
	gpu_graph::{Backend, GpuPlan, lower},
};
#[test]
fn graph_source_and_empty() {
	let mut g = Graph::default();
	let x = g.input(0, vec![0]).unwrap();
	let c = g.constant(2., vec![]).unwrap();
	let y = g.binary(Binary::Add, x, c).unwrap();
	for d in [Dialect::Metal, Dialect::OpenCl] {
		let code = lower(&g, &[y], d, 64).unwrap();
		assert!(code.last().unwrap().0.source.contains("if(i<"));
		assert!(!code.last().unwrap().0.source.contains("expf("));
	}
	let plan = GpuPlan::compile(&g, &[y], Backend::Metal, 64).unwrap();
	assert_eq!(
		plan.run(&[Buffer::new(vec![0], vec![]).unwrap()]).unwrap()[0].data,
		Vec::<f32>::new()
	);
}
#[test]
fn graph_lower_opencl_compiles() {
	let mut g = Graph::default();
	let x = g.input(0, vec![2, 3]).unwrap();
	let pad = g.pad(x, vec![(1, 1), (2, 1)]).unwrap();
	let y = g.reduce(pad, Reduce::Sum, vec![1], false).unwrap();
	for (i, (kernel, _)) in lower(&g, &[y], Dialect::OpenCl, 64)
		.unwrap()
		.into_iter()
		.enumerate()
	{
		let path = std::env::temp_dir().join(format!("chibi-{}-{i}.cl", std::process::id()));
		std::fs::write(&path, kernel.source).unwrap();
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
#[test]
fn unsupported_index_math_rejected_before_compilation() {
	let mut g = Graph::default();
	let x = g.input(0, vec![4]).unwrap();
	let i = g.input(1, vec![1]).unwrap();
	let i = g.unary(chibigrad::Unary::Sin, i).unwrap();
	let y = g.gather(x, i, 0).unwrap();
	for d in [Dialect::Metal, Dialect::OpenCl] {
		assert!(
			lower(&g, &[y], d, 64)
				.unwrap_err()
				.contains("computed index")
		);
	}
}
```

```rust
// file=tests/book_15.rs mode=write
use chibigrad::{
	Binary, Buffer, Graph, Reduce,
	gpu::{Dialect, groups},
	gpu_graph::{Backend, GpuPlan, lower},
};
fn source_contract() {
	assert_eq!(groups(257, 64).unwrap(), 5);
	assert!(groups(1, 0).is_err());
	assert!(groups(usize::MAX, 2).is_err());
	let mut g = Graph::default();
	let x = g.input(0, vec![2, 3]).unwrap();
	let x = g.pad(x, vec![(1, 1), (2, 1)]).unwrap();
	let y = g.reduce(x, Reduce::Sum, vec![1], false).unwrap();
	let code = lower(&g, &[y], Dialect::Metal, 64).unwrap();
	let source = code
		.iter()
		.map(|(k, _)| k.source.as_str())
		.collect::<String>();
	assert!(source.contains("threadgroup_position_in_grid"));
	assert!(source.contains("if(i<"));
	assert!(source.contains("if ("));
	assert!(!source.contains("malloc"));
	assert!(source.contains("float "));
	assert!(source.contains("s["));
	let mut empty = Graph::default();
	let a = empty.input(0, vec![0]).unwrap();
	let b = empty.constant(2., vec![]).unwrap();
	let y = empty.binary(Binary::Add, a, b).unwrap();
	let p = GpuPlan::compile(&empty, &[y], Backend::Metal, 64).unwrap();
	assert_eq!(
		p.run(&[Buffer::new(vec![0], vec![]).unwrap()]).unwrap()[0].data,
		Vec::<f32>::new()
	);
}
#[cfg(all(target_os = "macos", feature = "metal"))]
fn hardware_contract() {
	use chibigrad::backends::metal::MetalProgram;
	let mut bad = chibigrad::gpu::add(Dialect::Metal, 4, 4).unwrap();
	bad.source = "this is invalid MSL".into();
	let error = MetalProgram::compile(&bad)
		.err()
		.expect("invalid MSL must fail");
	assert!(error.contains("build"));
	for n in [1, 63, 64, 65, 257] {
		let mut g = Graph::default();
		let a = g.input(0, vec![n]).unwrap();
		let b = g.input(1, vec![n]).unwrap();
		let out = g.binary(Binary::Add, a, b).unwrap();
		let p = GpuPlan::compile(&g, &[out], Backend::Metal, 64).unwrap();
		for value in [1., 7.] {
			let inputs = [
				Buffer::new(vec![n], vec![value; n]).unwrap(),
				Buffer::new(vec![n], vec![2.; n]).unwrap(),
			];
			assert_eq!(
				p.run(&inputs).unwrap(),
				g.compile(&[out]).unwrap().run(&inputs).unwrap()
			);
			assert_eq!(
				p.run(&inputs).unwrap(),
				g.interpret(&[out], &inputs).unwrap()
			);
		}
	}
	let mut g = Graph::default();
	let a = g.input(0, vec![2, 3]).unwrap();
	let b = g.input(1, vec![3, 2]).unwrap();
	let z = g.matmul(a, b).unwrap();
	let pad = g.pad(a, vec![(1, 0), (1, 2)]).unwrap();
	let movement = g.permute(pad, vec![1, 0]).unwrap();
	let loss = g.reduce(z, Reduce::Sum, vec![0, 1], false).unwrap();
	let gradient = chibigrad::autodiff::gradient(&mut g, loss, &[b], None).unwrap()[0];
	let outputs = [z, movement, gradient];
	let p = GpuPlan::compile(&g, &outputs, Backend::Metal, 64).unwrap();
	let inputs = [
		Buffer::new(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap(),
		Buffer::new(vec![3, 2], vec![1., 2., 3., 4., 5., 6.]).unwrap(),
	];
	assert_eq!(
		p.run(&inputs).unwrap(),
		g.interpret(&outputs, &inputs).unwrap()
	);
}
```

```sh
# check
cargo check --features metal
```

## 利用可能な Metal device を検出し、未対応時の明示的エラーを作る。

<!-- todo:15-001 -->

実装は[この章の `GpuPlan::compile、GpuPlan::run_with_shapes、gpu::groups、MetalProgram::compile、MetalProgram::run、MetalProgram::limits`](#metal-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_15.rs mode=append
#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
#[ignore = "requires real device"]
fn todo_15_001() {
	hardware_contract();
}
```

`src/backends/metal.rs` の `pub fn compile(kernel:`の定義を次の全体で置き換えます。

```rust
// file="src/backends/metal.rs" mode="replace" item="pub fn compile(kernel:"
	pub fn compile(kernel: &Kernel) -> Result<Self, String> {
		let device = MTLCreateSystemDefaultDevice().ok_or("Metal device unavailable")?;
		let queue = device
			.newCommandQueue()
			.ok_or("Metal newCommandQueue returned nil")?;
		Self::compile_on(kernel, device, queue)
	}
```

```sh
# check requires=metal
cargo test --test book_15 --features metal todo_15_001 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## 静的 f32 の加算用 MSL を共通 kernel IR から生成する。

<!-- todo:15-002 -->

実装は[この章の `GpuPlan::compile、GpuPlan::run_with_shapes、gpu::groups、MetalProgram::compile、MetalProgram::run、MetalProgram::limits`](#metal-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_15.rs mode=append
#[test]
fn todo_15_002() {
	source_contract();
}
```

`src/gpu_graph.rs` の `pub fn lower(`の定義を次の全体で置き換えます。

```rust
// file="src/gpu_graph.rs" mode="replace" item="pub fn lower("
pub fn lower(
	graph: &Graph,
	outputs: &[usize],
	d: Dialect,
	local: usize,
) -> Result<Vec<(Kernel, Vec<usize>)>> {
	graph.verify(outputs)?;
	for id in graph.topology(outputs)? {
		if let Op::Gather { x, indices, axis }
		| Op::ScatterAdd {
			base: x,
			indices,
			axis,
			..
		} = graph.nodes[id].op
		{
			if graph.shape(x)[axis] > 16_777_216 {
				return Err("GPU f32 index extent exceeds exact integer range".into());
			}
			for index_node in graph.topology(&[indices])? {
				if !matches!(
					graph.nodes[index_node].op,
					Op::Input { .. }
						| Op::Const { .. } | Op::Reshape { .. }
						| Op::Permute { .. }
						| Op::Expand { .. }
						| Op::Shrink { .. }
						| Op::Pad { .. } | Op::Flip { .. }
						| Op::Stack { .. } | Op::Stage { .. }
						| Op::StopGradient { .. }
				) {
					return Err("GPU computed index arithmetic is unsupported; materialize integer indices as an input".into());
				}
			}
		}
	}
	let kernels = kernel_bodies(graph, outputs, true)?;
	let dims = graph
		.nodes
		.iter()
		.flat_map(|n| n.shape.iter().map(|&v| v as u64))
		.collect::<Vec<_>>();
	let mut result = vec![];
	for k in kernels {
		let n = elements(graph.shape(k.output))?;
		let start = k.body.find('{').ok_or("missing outer loop")? + 1;
		let end = k.body.rfind('}').ok_or("missing loop end")?;
		let mut body = k.body[start..end].to_string();

		for (from, to) in [
			("expf(", "exp("),
			("logf(", "log("),
			("sinf(", "sin("),
			("cosf(", "cos("),
			("sqrtf(", "sqrt("),
			("exp2f(", "exp2("),
			("log2f(", "log2("),
			("floorf(", "floor("),
			("(double)", "(float)"),
			("uint32_t", "uint"),
		] {
			body = body.replace(from, to);
		}
		body = body.replace("return 2;", "return;");
		let mut slots = graph
			.topology(&[k.output])?
			.into_iter()
			.filter(|id| body.contains(&format!("b{id}[")))
			.collect::<Vec<_>>();
		slots.retain(|&id| id != k.output);
		slots.push(k.output);
		for (index, id) in slots.iter().enumerate() {
			body = body.replace(&format!("b{id}["), &format!("slot{index}["));
		}
		for index in 0..slots.len() {
			body = body.replace(&format!("slot{index}["), &format!("b{index}["));
		}
		let condition = k
			.body
			.split("i<")
			.nth(1)
			.and_then(|s| s.split("; ++i)").next())
			.ok_or("outer loop condition")?;
		let body = format!("size_t i=g*{local}+l;if(i<{condition}){{{body}}}");
		let mut source = header(d, slots.len(), &body);
		let helper = helpers(d);
		source = match d {
			Dialect::Metal => source.replacen("kernel void", &(helper + "kernel void"), 1),
			Dialect::OpenCl => helper + &source,
		};
		let lengths = slots
			.iter()
			.map(|&id| elements(graph.shape(id)))
			.collect::<Result<Vec<_>>>()?;
		result.push((
			Kernel {
				scalars: dims.clone(),
				source,
				entry: "chibi".into(),
				output: slots.len() - 1,
				lengths,
				groups: groups(n, local)?,
				local,
				shared_bytes: 0,
			},
			slots,
		));
	}
	Ok(result)
}
```

```sh
# check
cargo test --test book_15 todo_15_002 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 公開 API で library と pipeline を作り、build error を返す。

<!-- todo:15-003 -->

実装は[この章の `GpuPlan::compile、GpuPlan::run_with_shapes、gpu::groups、MetalProgram::compile、MetalProgram::run、MetalProgram::limits`](#metal-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_15.rs mode=append
#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
#[ignore = "requires real device"]
fn todo_15_003() {
	hardware_contract();
}
```

`src/backends/metal.rs` の `fn compile_on(`の定義を次の全体で置き換えます。

```rust
// file="src/backends/metal.rs" mode="replace" item="fn compile_on("
	fn compile_on(
		kernel: &Kernel,
		device: Device,
		queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
	) -> Result<Self, String> {
		autoreleasepool(|_| {
			let start = Instant::now();
			let options = MTLCompileOptions::new();
			#[allow(deprecated)]
			options.setFastMathEnabled(false);
			let library = device
				.newLibraryWithSource_options_error(
					&NSString::from_str(&kernel.source),
					Some(&options),
				)
				.map_err(|e| format!("Metal source build: {e}"))?;
			let function = library
				.newFunctionWithName(&NSString::from_str(&kernel.entry))
				.ok_or("Metal entry missing")?;
			let pipeline = device
				.newComputePipelineStateWithFunction_error(&function)
				.map_err(|e| format!("Metal pipeline build: {e}"))?;
			Ok(Self {
				device,
				queue,
				pipeline,
				compile_time: start.elapsed(),
				contract: (
					kernel.source.clone(),
					kernel.entry.clone(),
					kernel.lengths.len(),
					kernel.scalars.len().max(1),
					kernel.local,
				),
			})
		})
	}
```

```sh
# check requires=metal
cargo test --test book_15 --features metal todo_15_003 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## shared buffer の入力を書き、4 thread を dispatch する。

<!-- todo:15-004 -->

実装は[この章の `GpuPlan::compile、GpuPlan::run_with_shapes、gpu::groups、MetalProgram::compile、MetalProgram::run、MetalProgram::limits`](#metal-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_15.rs mode=append
#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
#[ignore = "requires real device"]
fn todo_15_004() {
	hardware_contract();
}
```

`src/backends/metal.rs` の `pub unsafe fn run(&self,`の定義を次の全体で置き換えます。

```rust
// file="src/backends/metal.rs" mode="replace" item="pub unsafe fn run(&self,"
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

```sh
# check requires=metal
cargo test --test book_15 --features metal todo_15_004 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## 完了待ち後に出力を読み、CPU の値と比較する。

<!-- todo:15-005 -->

実装は[この章の `GpuPlan::compile、GpuPlan::run_with_shapes、gpu::groups、MetalProgram::compile、MetalProgram::run、MetalProgram::limits`](#metal-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_15.rs mode=append
#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
#[ignore = "requires real device"]
fn todo_15_005() {
	hardware_contract();
}
```

```sh
# check requires=metal
cargo test --test book_15 --features metal todo_15_005 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## groups を `ceil(N/L)` で計算し、掛算 overflow を検査する。

<!-- todo:15-006 -->

実装は[この章の `GpuPlan::compile、GpuPlan::run_with_shapes、gpu::groups、MetalProgram::compile、MetalProgram::run、MetalProgram::limits`](#metal-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_15.rs mode=append
#[test]
fn todo_15_006() {
	source_contract();
}
```

`src/gpu.rs` の `pub fn groups(`の定義を次の全体で置き換えます。

```rust
// file="src/gpu.rs" mode="replace" item="pub fn groups("
pub fn groups(n: usize, local: usize) -> Result<usize, String> {
	if local == 0 {
		return Err("zero local size".into());
	}
	let groups = n / local + usize::from(!n.is_multiple_of(local));
	groups.checked_mul(local).ok_or("launch overflow")?;
	Ok(groups)
}
```

```sh
# check
cargo test --test book_15 todo_15_006 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## `i<N` の内側に LOAD と STORE の両方を置く。

<!-- todo:15-007 -->

実装は[この章の `GpuPlan::compile、GpuPlan::run_with_shapes、gpu::groups、MetalProgram::compile、MetalProgram::run、MetalProgram::limits`](#metal-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_15.rs mode=append
#[test]
fn todo_15_007() {
	source_contract();
}
```

`src/gpu_graph.rs` の `pub fn lower(`の定義を次の全体で置き換えます。

```rust
// file="src/gpu_graph.rs" mode="replace" item="pub fn lower("
pub fn lower(
	graph: &Graph,
	outputs: &[usize],
	d: Dialect,
	local: usize,
) -> Result<Vec<(Kernel, Vec<usize>)>> {
	graph.verify(outputs)?;
	for id in graph.topology(outputs)? {
		if let Op::Gather { x, indices, axis }
		| Op::ScatterAdd {
			base: x,
			indices,
			axis,
			..
		} = graph.nodes[id].op
		{
			if graph.shape(x)[axis] > 16_777_216 {
				return Err("GPU f32 index extent exceeds exact integer range".into());
			}
			for index_node in graph.topology(&[indices])? {
				if !matches!(
					graph.nodes[index_node].op,
					Op::Input { .. }
						| Op::Const { .. } | Op::Reshape { .. }
						| Op::Permute { .. }
						| Op::Expand { .. }
						| Op::Shrink { .. }
						| Op::Pad { .. } | Op::Flip { .. }
						| Op::Stack { .. } | Op::Stage { .. }
						| Op::StopGradient { .. }
				) {
					return Err("GPU computed index arithmetic is unsupported; materialize integer indices as an input".into());
				}
			}
		}
	}
	let kernels = kernel_bodies(graph, outputs, true)?;
	let dims = graph
		.nodes
		.iter()
		.flat_map(|n| n.shape.iter().map(|&v| v as u64))
		.collect::<Vec<_>>();
	let mut result = vec![];
	for k in kernels {
		let n = elements(graph.shape(k.output))?;
		let start = k.body.find('{').ok_or("missing outer loop")? + 1;
		let end = k.body.rfind('}').ok_or("missing loop end")?;
		let mut body = k.body[start..end].to_string();

		for (from, to) in [
			("expf(", "exp("),
			("logf(", "log("),
			("sinf(", "sin("),
			("cosf(", "cos("),
			("sqrtf(", "sqrt("),
			("exp2f(", "exp2("),
			("log2f(", "log2("),
			("floorf(", "floor("),
			("(double)", "(float)"),
			("uint32_t", "uint"),
		] {
			body = body.replace(from, to);
		}
		body = body.replace("return 2;", "return;");
		let mut slots = graph
			.topology(&[k.output])?
			.into_iter()
			.filter(|id| body.contains(&format!("b{id}[")))
			.collect::<Vec<_>>();
		slots.retain(|&id| id != k.output);
		slots.push(k.output);
		for (index, id) in slots.iter().enumerate() {
			body = body.replace(&format!("b{id}["), &format!("slot{index}["));
		}
		for index in 0..slots.len() {
			body = body.replace(&format!("slot{index}["), &format!("b{index}["));
		}
		let condition = k
			.body
			.split("i<")
			.nth(1)
			.and_then(|s| s.split("; ++i)").next())
			.ok_or("outer loop condition")?;
		let body = format!("size_t i=g*{local}+l;if(i<{condition}){{{body}}}");
		let mut source = header(d, slots.len(), &body);
		let helper = helpers(d);
		source = match d {
			Dialect::Metal => source.replacen("kernel void", &(helper + "kernel void"), 1),
			Dialect::OpenCl => helper + &source,
		};
		let lengths = slots
			.iter()
			.map(|&id| elements(graph.shape(id)))
			.collect::<Result<Vec<_>>>()?;
		result.push((
			Kernel {
				scalars: dims.clone(),
				source,
				entry: "chibi".into(),
				output: slots.len() - 1,
				lengths,
				groups: groups(n, local)?,
				local,
				shared_bytes: 0,
			},
			slots,
		));
	}
	Ok(result)
}
```

```sh
# check
cargo test --test book_15 todo_15_007 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## device と pipeline の上限を使って local size を検証する。

<!-- todo:15-008 -->

実装は[この章の `GpuPlan::compile、GpuPlan::run_with_shapes、gpu::groups、MetalProgram::compile、MetalProgram::run、MetalProgram::limits`](#metal-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_15.rs mode=append
#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
#[ignore = "requires real device"]
fn todo_15_008() {
	hardware_contract();
}
```

`src/backends/metal.rs` の `pub unsafe fn run(&self,`の定義を次の全体で置き換えます。

```rust
// file="src/backends/metal.rs" mode="replace" item="pub unsafe fn run(&self,"
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

```sh
# check requires=metal
cargo test --test book_15 --features metal todo_15_008 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## N=1、L-1、L、L+1、257 を同じ lowering で実行する。

<!-- todo:15-009 -->

実装は[この章の `GpuPlan::compile、GpuPlan::run_with_shapes、gpu::groups、MetalProgram::compile、MetalProgram::run、MetalProgram::limits`](#metal-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_15.rs mode=append
#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
#[ignore = "requires real device"]
fn todo_15_009() {
	hardware_contract();
}
```

```sh
# check requires=metal
cargo test --test book_15 --features metal todo_15_009 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## N=0 は host 側で空出力を返し、ゼロ個の group を dispatch しない。

<!-- todo:15-010 -->

実装は[この章の `GpuPlan::compile、GpuPlan::run_with_shapes、gpu::groups、MetalProgram::compile、MetalProgram::run、MetalProgram::limits`](#metal-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_15.rs mode=append
#[test]
fn todo_15_010() {
	source_contract();
}
```

`src/backends/metal.rs` の `pub unsafe fn run(kernel:`の定義を次の全体で置き換えます。

```rust
// file="src/backends/metal.rs" mode="replace" item="pub unsafe fn run(kernel:"
pub unsafe fn run(kernel: &Kernel, inputs: &[Vec<f32>]) -> Result<Run, String> {
	kernel.validate(inputs)?;
	if kernel.lengths[kernel.output] == 0 {
		return Ok(Run {
			output: vec![],
			device: "Metal API not called (empty)".into(),
			timings: Timings::default(),
		});
	}
	let total = Instant::now();
	let program = MetalProgram::compile(kernel)?;
	let mut result = unsafe { program.run(kernel, inputs) }?;
	result.timings.compile = program.compile_time();
	result.timings.total = total.elapsed();
	Ok(result)
}
```

`src/gpu_graph.rs` の `fn from_lowered(`の定義を次の全体で置き換えます。

```rust
// file="src/gpu_graph.rs" mode="replace" item="fn from_lowered("
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

```sh
# check
cargo test --test book_15 todo_15_010 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## Plan の入力 buffer を差し替えて再実行する。

<!-- todo:15-011 -->

実装は[この章の `GpuPlan::compile、GpuPlan::run_with_shapes、gpu::groups、MetalProgram::compile、MetalProgram::run、MetalProgram::limits`](#metal-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_15.rs mode=append
#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
#[ignore = "requires real device"]
fn todo_15_011() {
	hardware_contract();
}
```

`src/gpu_graph.rs` の `pub fn run_with_shapes(`の定義を次の全体で置き換えます。

```rust
// file="src/gpu_graph.rs" mode="replace" item="pub fn run_with_shapes("
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

`src/backends/metal.rs` の `pub unsafe fn run(&self,`の定義を次の全体で置き換えます。

```rust
// file="src/backends/metal.rs" mode="replace" item="pub unsafe fn run(&self,"
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

```sh
# check requires=metal
cargo test --test book_15 --features metal todo_15_011 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## 出力の平坦 ID を多次元座標へ復元する。

<!-- todo:15-012 -->

実装は[この章の `GpuPlan::compile、GpuPlan::run_with_shapes、gpu::groups、MetalProgram::compile、MetalProgram::run、MetalProgram::limits`](#metal-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_15.rs mode=append
#[test]
fn todo_15_012() {
	source_contract();
}
```

`src/graph.rs` の `fn coord(`の定義を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn coord("
	fn coord(&self, id: usize, index: &str, axis: usize) -> String {
		format!(
			"((({index}) / {}) % {})",
			self.product(id, axis + 1..self.g.shape(id).len()),
			self.dim(id, axis)
		)
	}
```

`src/graph.rs` の `fn flatten(`の定義を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn flatten("
	fn flatten(&self, id: usize, c: &[String]) -> String {
		if c.is_empty() {
			return "0".into();
		}
		format!(
			"({})",
			c.iter()
				.enumerate()
				.map(|(a, c)| format!(
					"({c}) * {}",
					self.product(id, a + 1..self.g.shape(id).len())
				))
				.collect::<Vec<_>>()
				.join(" + ")
		)
	}
```

`src/gpu_graph.rs` の `pub fn lower(`の定義を次の全体で置き換えます。

```rust
// file="src/gpu_graph.rs" mode="replace" item="pub fn lower("
pub fn lower(
	graph: &Graph,
	outputs: &[usize],
	d: Dialect,
	local: usize,
) -> Result<Vec<(Kernel, Vec<usize>)>> {
	graph.verify(outputs)?;
	for id in graph.topology(outputs)? {
		if let Op::Gather { x, indices, axis }
		| Op::ScatterAdd {
			base: x,
			indices,
			axis,
			..
		} = graph.nodes[id].op
		{
			if graph.shape(x)[axis] > 16_777_216 {
				return Err("GPU f32 index extent exceeds exact integer range".into());
			}
			for index_node in graph.topology(&[indices])? {
				if !matches!(
					graph.nodes[index_node].op,
					Op::Input { .. }
						| Op::Const { .. } | Op::Reshape { .. }
						| Op::Permute { .. }
						| Op::Expand { .. }
						| Op::Shrink { .. }
						| Op::Pad { .. } | Op::Flip { .. }
						| Op::Stack { .. } | Op::Stage { .. }
						| Op::StopGradient { .. }
				) {
					return Err("GPU computed index arithmetic is unsupported; materialize integer indices as an input".into());
				}
			}
		}
	}
	let kernels = kernel_bodies(graph, outputs, true)?;
	let dims = graph
		.nodes
		.iter()
		.flat_map(|n| n.shape.iter().map(|&v| v as u64))
		.collect::<Vec<_>>();
	let mut result = vec![];
	for k in kernels {
		let n = elements(graph.shape(k.output))?;
		let start = k.body.find('{').ok_or("missing outer loop")? + 1;
		let end = k.body.rfind('}').ok_or("missing loop end")?;
		let mut body = k.body[start..end].to_string();

		for (from, to) in [
			("expf(", "exp("),
			("logf(", "log("),
			("sinf(", "sin("),
			("cosf(", "cos("),
			("sqrtf(", "sqrt("),
			("exp2f(", "exp2("),
			("log2f(", "log2("),
			("floorf(", "floor("),
			("(double)", "(float)"),
			("uint32_t", "uint"),
		] {
			body = body.replace(from, to);
		}
		body = body.replace("return 2;", "return;");
		let mut slots = graph
			.topology(&[k.output])?
			.into_iter()
			.filter(|id| body.contains(&format!("b{id}[")))
			.collect::<Vec<_>>();
		slots.retain(|&id| id != k.output);
		slots.push(k.output);
		for (index, id) in slots.iter().enumerate() {
			body = body.replace(&format!("b{id}["), &format!("slot{index}["));
		}
		for index in 0..slots.len() {
			body = body.replace(&format!("slot{index}["), &format!("b{index}["));
		}
		let condition = k
			.body
			.split("i<")
			.nth(1)
			.and_then(|s| s.split("; ++i)").next())
			.ok_or("outer loop condition")?;
		let body = format!("size_t i=g*{local}+l;if(i<{condition}){{{body}}}");
		let mut source = header(d, slots.len(), &body);
		let helper = helpers(d);
		source = match d {
			Dialect::Metal => source.replacen("kernel void", &(helper + "kernel void"), 1),
			Dialect::OpenCl => helper + &source,
		};
		let lengths = slots
			.iter()
			.map(|&id| elements(graph.shape(id)))
			.collect::<Result<Vec<_>>>()?;
		result.push((
			Kernel {
				scalars: dims.clone(),
				source,
				entry: "chibi".into(),
				output: slots.len() - 1,
				lengths,
				groups: groups(n, local)?,
				local,
				shared_bytes: 0,
			},
			slots,
		));
	}
	Ok(result)
}
```

```sh
# check
cargo test --test book_15 todo_15_012 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## movement と broadcast を CPU と同じ座標変換から生成する。

<!-- todo:15-013 -->

実装は[この章の `GpuPlan::compile、GpuPlan::run_with_shapes、gpu::groups、MetalProgram::compile、MetalProgram::run、MetalProgram::limits`](#metal-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_15.rs mode=append
#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
#[ignore = "requires real device"]
fn todo_15_013() {
	hardware_contract();
}
```

`src/graph.rs` の `fn expr(` の `Op::Expand` 分岐を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn expr(" arm="Op::Expand"
			Op::Expand { x } => {
				let xi = self.broadcast_index(x, id, index);
				return self.expr(x, &xi, code);
			}
```

`src/graph.rs` の `fn expr(` の `Op::Permute` 分岐を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn expr(" arm="Op::Permute"
			Op::Permute { x, axes } => {
				let mut c = vec![String::new(); axes.len()];
				for (a, &old) in axes.iter().enumerate() {
					c[old] = self.coord(id, index, a)
				}
				let xi = self.flatten(x, &c);
				return self.expr(x, &xi, code);
			}
```

`src/graph.rs` の `fn expr(` の `Op::Shrink` 分岐を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn expr(" arm="Op::Shrink"
			Op::Shrink { x, ranges } => {
				let c = ranges
					.iter()
					.enumerate()
					.map(|(a, &(lo, _))| format!("({}+{lo})", self.coord(id, index, a)))
					.collect::<Vec<_>>();
				let xi = self.flatten(x, &c);
				return self.expr(x, &xi, code);
			}
```

`src/gpu_graph.rs` の `pub fn lower(`の定義を次の全体で置き換えます。

```rust
// file="src/gpu_graph.rs" mode="replace" item="pub fn lower("
pub fn lower(
	graph: &Graph,
	outputs: &[usize],
	d: Dialect,
	local: usize,
) -> Result<Vec<(Kernel, Vec<usize>)>> {
	graph.verify(outputs)?;
	for id in graph.topology(outputs)? {
		if let Op::Gather { x, indices, axis }
		| Op::ScatterAdd {
			base: x,
			indices,
			axis,
			..
		} = graph.nodes[id].op
		{
			if graph.shape(x)[axis] > 16_777_216 {
				return Err("GPU f32 index extent exceeds exact integer range".into());
			}
			for index_node in graph.topology(&[indices])? {
				if !matches!(
					graph.nodes[index_node].op,
					Op::Input { .. }
						| Op::Const { .. } | Op::Reshape { .. }
						| Op::Permute { .. }
						| Op::Expand { .. }
						| Op::Shrink { .. }
						| Op::Pad { .. } | Op::Flip { .. }
						| Op::Stack { .. } | Op::Stage { .. }
						| Op::StopGradient { .. }
				) {
					return Err("GPU computed index arithmetic is unsupported; materialize integer indices as an input".into());
				}
			}
		}
	}
	let kernels = kernel_bodies(graph, outputs, true)?;
	let dims = graph
		.nodes
		.iter()
		.flat_map(|n| n.shape.iter().map(|&v| v as u64))
		.collect::<Vec<_>>();
	let mut result = vec![];
	for k in kernels {
		let n = elements(graph.shape(k.output))?;
		let start = k.body.find('{').ok_or("missing outer loop")? + 1;
		let end = k.body.rfind('}').ok_or("missing loop end")?;
		let mut body = k.body[start..end].to_string();

		for (from, to) in [
			("expf(", "exp("),
			("logf(", "log("),
			("sinf(", "sin("),
			("cosf(", "cos("),
			("sqrtf(", "sqrt("),
			("exp2f(", "exp2("),
			("log2f(", "log2("),
			("floorf(", "floor("),
			("(double)", "(float)"),
			("uint32_t", "uint"),
		] {
			body = body.replace(from, to);
		}
		body = body.replace("return 2;", "return;");
		let mut slots = graph
			.topology(&[k.output])?
			.into_iter()
			.filter(|id| body.contains(&format!("b{id}[")))
			.collect::<Vec<_>>();
		slots.retain(|&id| id != k.output);
		slots.push(k.output);
		for (index, id) in slots.iter().enumerate() {
			body = body.replace(&format!("b{id}["), &format!("slot{index}["));
		}
		for index in 0..slots.len() {
			body = body.replace(&format!("slot{index}["), &format!("b{index}["));
		}
		let condition = k
			.body
			.split("i<")
			.nth(1)
			.and_then(|s| s.split("; ++i)").next())
			.ok_or("outer loop condition")?;
		let body = format!("size_t i=g*{local}+l;if(i<{condition}){{{body}}}");
		let mut source = header(d, slots.len(), &body);
		let helper = helpers(d);
		source = match d {
			Dialect::Metal => source.replacen("kernel void", &(helper + "kernel void"), 1),
			Dialect::OpenCl => helper + &source,
		};
		let lengths = slots
			.iter()
			.map(|&id| elements(graph.shape(id)))
			.collect::<Result<Vec<_>>>()?;
		result.push((
			Kernel {
				scalars: dims.clone(),
				source,
				entry: "chibi".into(),
				output: slots.len() - 1,
				lengths,
				groups: groups(n, local)?,
				local,
				shared_bytes: 0,
			},
			slots,
		));
	}
	Ok(result)
}
```

```sh
# check requires=metal
cargo test --test book_15 --features metal todo_15_013 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## pad の有効条件が成立した場合だけ device buffer を LOAD する。

<!-- todo:15-014 -->

実装は[この章の `GpuPlan::compile、GpuPlan::run_with_shapes、gpu::groups、MetalProgram::compile、MetalProgram::run、MetalProgram::limits`](#metal-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_15.rs mode=append
#[test]
fn todo_15_014() {
	source_contract();
}
```

`src/graph.rs` の `fn expr(` の `Op::Pad` 分岐を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn expr(" arm="Op::Pad"
			Op::Pad { x, pads } => {
				let c = (0..pads.len())
					.map(|a| self.coord(id, index, a))
					.collect::<Vec<_>>();
				let checks = pads
					.iter()
					.enumerate()
					.map(|(a, &(lo, _))| {
						format!("({} >= {lo} && {} - {lo} < {})", c[a], c[a], self.dim(x, a))
					})
					.collect::<Vec<_>>();
				let target = self.name();
				code.push_str(&format!(
					"float {target}=0.0f;\nif ({}) {{\n",
					if checks.is_empty() {
						"1".into()
					} else {
						checks.join(" && ")
					}
				));
				let xi = self.flatten(
					x,
					&c.iter()
						.zip(&pads)
						.map(|(c, &(lo, _))| format!("({c}-{lo})"))
						.collect::<Vec<_>>(),
				);
				let value = self.expr(x, &xi, code)?;
				code.push_str(&format!("{target}={value};\n}}\n"));
				return Ok(target);
			}
```

`src/gpu_graph.rs` の `pub fn lower(`の定義を次の全体で置き換えます。

```rust
// file="src/gpu_graph.rs" mode="replace" item="pub fn lower("
pub fn lower(
	graph: &Graph,
	outputs: &[usize],
	d: Dialect,
	local: usize,
) -> Result<Vec<(Kernel, Vec<usize>)>> {
	graph.verify(outputs)?;
	for id in graph.topology(outputs)? {
		if let Op::Gather { x, indices, axis }
		| Op::ScatterAdd {
			base: x,
			indices,
			axis,
			..
		} = graph.nodes[id].op
		{
			if graph.shape(x)[axis] > 16_777_216 {
				return Err("GPU f32 index extent exceeds exact integer range".into());
			}
			for index_node in graph.topology(&[indices])? {
				if !matches!(
					graph.nodes[index_node].op,
					Op::Input { .. }
						| Op::Const { .. } | Op::Reshape { .. }
						| Op::Permute { .. }
						| Op::Expand { .. }
						| Op::Shrink { .. }
						| Op::Pad { .. } | Op::Flip { .. }
						| Op::Stack { .. } | Op::Stage { .. }
						| Op::StopGradient { .. }
				) {
					return Err("GPU computed index arithmetic is unsupported; materialize integer indices as an input".into());
				}
			}
		}
	}
	let kernels = kernel_bodies(graph, outputs, true)?;
	let dims = graph
		.nodes
		.iter()
		.flat_map(|n| n.shape.iter().map(|&v| v as u64))
		.collect::<Vec<_>>();
	let mut result = vec![];
	for k in kernels {
		let n = elements(graph.shape(k.output))?;
		let start = k.body.find('{').ok_or("missing outer loop")? + 1;
		let end = k.body.rfind('}').ok_or("missing loop end")?;
		let mut body = k.body[start..end].to_string();

		for (from, to) in [
			("expf(", "exp("),
			("logf(", "log("),
			("sinf(", "sin("),
			("cosf(", "cos("),
			("sqrtf(", "sqrt("),
			("exp2f(", "exp2("),
			("log2f(", "log2("),
			("floorf(", "floor("),
			("(double)", "(float)"),
			("uint32_t", "uint"),
		] {
			body = body.replace(from, to);
		}
		body = body.replace("return 2;", "return;");
		let mut slots = graph
			.topology(&[k.output])?
			.into_iter()
			.filter(|id| body.contains(&format!("b{id}[")))
			.collect::<Vec<_>>();
		slots.retain(|&id| id != k.output);
		slots.push(k.output);
		for (index, id) in slots.iter().enumerate() {
			body = body.replace(&format!("b{id}["), &format!("slot{index}["));
		}
		for index in 0..slots.len() {
			body = body.replace(&format!("slot{index}["), &format!("b{index}["));
		}
		let condition = k
			.body
			.split("i<")
			.nth(1)
			.and_then(|s| s.split("; ++i)").next())
			.ok_or("outer loop condition")?;
		let body = format!("size_t i=g*{local}+l;if(i<{condition}){{{body}}}");
		let mut source = header(d, slots.len(), &body);
		let helper = helpers(d);
		source = match d {
			Dialect::Metal => source.replacen("kernel void", &(helper + "kernel void"), 1),
			Dialect::OpenCl => helper + &source,
		};
		let lengths = slots
			.iter()
			.map(|&id| elements(graph.shape(id)))
			.collect::<Result<Vec<_>>>()?;
		result.push((
			Kernel {
				scalars: dims.clone(),
				source,
				entry: "chibi".into(),
				output: slots.len() - 1,
				lengths,
				groups: groups(n, local)?,
				local,
				shared_bytes: 0,
			},
			slots,
		));
	}
	Ok(result)
}
```

```sh
# check
cargo test --test book_15 todo_15_014 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 無効 LOAD の値を事後に WHERE で隠す実装を拒否する。

<!-- todo:15-015 -->

実装は[この章の `GpuPlan::compile、GpuPlan::run_with_shapes、gpu::groups、MetalProgram::compile、MetalProgram::run、MetalProgram::limits`](#metal-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_15.rs mode=append
#[test]
fn todo_15_015() {
	source_contract();
}
```

`src/graph.rs` の `fn expr(` の `Op::Pad` 分岐を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn expr(" arm="Op::Pad"
			Op::Pad { x, pads } => {
				let c = (0..pads.len())
					.map(|a| self.coord(id, index, a))
					.collect::<Vec<_>>();
				let checks = pads
					.iter()
					.enumerate()
					.map(|(a, &(lo, _))| {
						format!("({} >= {lo} && {} - {lo} < {})", c[a], c[a], self.dim(x, a))
					})
					.collect::<Vec<_>>();
				let target = self.name();
				code.push_str(&format!(
					"float {target}=0.0f;\nif ({}) {{\n",
					if checks.is_empty() {
						"1".into()
					} else {
						checks.join(" && ")
					}
				));
				let xi = self.flatten(
					x,
					&c.iter()
						.zip(&pads)
						.map(|(c, &(lo, _))| format!("({c}-{lo})"))
						.collect::<Vec<_>>(),
				);
				let value = self.expr(x, &xi, code)?;
				code.push_str(&format!("{target}={value};\n}}\n"));
				return Ok(target);
			}
```

`src/gpu_graph.rs` の `pub fn lower(`の定義を次の全体で置き換えます。

```rust
// file="src/gpu_graph.rs" mode="replace" item="pub fn lower("
pub fn lower(
	graph: &Graph,
	outputs: &[usize],
	d: Dialect,
	local: usize,
) -> Result<Vec<(Kernel, Vec<usize>)>> {
	graph.verify(outputs)?;
	for id in graph.topology(outputs)? {
		if let Op::Gather { x, indices, axis }
		| Op::ScatterAdd {
			base: x,
			indices,
			axis,
			..
		} = graph.nodes[id].op
		{
			if graph.shape(x)[axis] > 16_777_216 {
				return Err("GPU f32 index extent exceeds exact integer range".into());
			}
			for index_node in graph.topology(&[indices])? {
				if !matches!(
					graph.nodes[index_node].op,
					Op::Input { .. }
						| Op::Const { .. } | Op::Reshape { .. }
						| Op::Permute { .. }
						| Op::Expand { .. }
						| Op::Shrink { .. }
						| Op::Pad { .. } | Op::Flip { .. }
						| Op::Stack { .. } | Op::Stage { .. }
						| Op::StopGradient { .. }
				) {
					return Err("GPU computed index arithmetic is unsupported; materialize integer indices as an input".into());
				}
			}
		}
	}
	let kernels = kernel_bodies(graph, outputs, true)?;
	let dims = graph
		.nodes
		.iter()
		.flat_map(|n| n.shape.iter().map(|&v| v as u64))
		.collect::<Vec<_>>();
	let mut result = vec![];
	for k in kernels {
		let n = elements(graph.shape(k.output))?;
		let start = k.body.find('{').ok_or("missing outer loop")? + 1;
		let end = k.body.rfind('}').ok_or("missing loop end")?;
		let mut body = k.body[start..end].to_string();

		for (from, to) in [
			("expf(", "exp("),
			("logf(", "log("),
			("sinf(", "sin("),
			("cosf(", "cos("),
			("sqrtf(", "sqrt("),
			("exp2f(", "exp2("),
			("log2f(", "log2("),
			("floorf(", "floor("),
			("(double)", "(float)"),
			("uint32_t", "uint"),
		] {
			body = body.replace(from, to);
		}
		body = body.replace("return 2;", "return;");
		let mut slots = graph
			.topology(&[k.output])?
			.into_iter()
			.filter(|id| body.contains(&format!("b{id}[")))
			.collect::<Vec<_>>();
		slots.retain(|&id| id != k.output);
		slots.push(k.output);
		for (index, id) in slots.iter().enumerate() {
			body = body.replace(&format!("b{id}["), &format!("slot{index}["));
		}
		for index in 0..slots.len() {
			body = body.replace(&format!("slot{index}["), &format!("b{index}["));
		}
		let condition = k
			.body
			.split("i<")
			.nth(1)
			.and_then(|s| s.split("; ++i)").next())
			.ok_or("outer loop condition")?;
		let body = format!("size_t i=g*{local}+l;if(i<{condition}){{{body}}}");
		let mut source = header(d, slots.len(), &body);
		let helper = helpers(d);
		source = match d {
			Dialect::Metal => source.replacen("kernel void", &(helper + "kernel void"), 1),
			Dialect::OpenCl => helper + &source,
		};
		let lengths = slots
			.iter()
			.map(|&id| elements(graph.shape(id)))
			.collect::<Result<Vec<_>>>()?;
		result.push((
			Kernel {
				scalars: dims.clone(),
				source,
				entry: "chibi".into(),
				output: slots.len() - 1,
				lengths,
				groups: groups(n, local)?,
				local,
				shared_bytes: 0,
			},
			slots,
		));
	}
	Ok(result)
}
```

```sh
# check
cargo test --test book_15 todo_15_015 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## CPU、interpreter、Metal の三つの値を比較する。

<!-- todo:15-016 -->

実装は[この章の `GpuPlan::compile、GpuPlan::run_with_shapes、gpu::groups、MetalProgram::compile、MetalProgram::run、MetalProgram::limits`](#metal-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_15.rs mode=append
#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
#[ignore = "requires real device"]
fn todo_15_016() {
	hardware_contract();
}
```

```sh
# check requires=metal
cargo test --test book_15 --features metal todo_15_016 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## 一つの thread が一行を担当し、列方向を逐次累積する。

<!-- todo:15-017 -->

実装は[この章の `GpuPlan::compile、GpuPlan::run_with_shapes、gpu::groups、MetalProgram::compile、MetalProgram::run、MetalProgram::limits`](#metal-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_15.rs mode=append
#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
#[ignore = "requires real device"]
fn todo_15_017() {
	hardware_contract();
}
```

`src/graph.rs` の `fn expr(` の `Op::Reduce` 分岐を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn expr(" arm="Op::Reduce"
			Op::Reduce {
				op,
				x,
				axes,
				keepdim,
			} => {
				let target = self.name();
				let k = self.name();
				code.push_str(&format!(
					"float {target} = {};\nfor (size_t {k}=0; {k}<{}; ++{k}) {{\n",
					if op == Reduce::Sum {
						"0.0f"
					} else {
						"-INFINITY"
					},
					self.product(x, axes.iter().copied())
				));
				let mut out_axis = 0;
				let c = (0..self.g.shape(x).len())
					.map(|a| {
						if let Some(ra) = axes.iter().position(|&v| v == a) {
							format!(
								"(({k} / {}) % {})",
								self.product(x, axes[ra + 1..].iter().copied()),
								self.dim(x, a)
							)
						} else {
							let oa = if keepdim {
								a
							} else {
								let v = out_axis;
								out_axis += 1;
								v
							};
							self.coord(id, index, oa)
						}
					})
					.collect::<Vec<_>>();
				let xi = self.flatten(x, &c);
				let value = self.expr(x, &xi, code)?;
				code.push_str(&format!(
					"{target} = {};\n}}\n",
					if op == Reduce::Sum {
						format!("{target} + ({value})")
					} else {
						format!("chibi_max({target},{value})")
					}
				));
				return Ok(target);
			}
```

`src/gpu_graph.rs` の `pub fn lower(`の定義を次の全体で置き換えます。

```rust
// file="src/gpu_graph.rs" mode="replace" item="pub fn lower("
pub fn lower(
	graph: &Graph,
	outputs: &[usize],
	d: Dialect,
	local: usize,
) -> Result<Vec<(Kernel, Vec<usize>)>> {
	graph.verify(outputs)?;
	for id in graph.topology(outputs)? {
		if let Op::Gather { x, indices, axis }
		| Op::ScatterAdd {
			base: x,
			indices,
			axis,
			..
		} = graph.nodes[id].op
		{
			if graph.shape(x)[axis] > 16_777_216 {
				return Err("GPU f32 index extent exceeds exact integer range".into());
			}
			for index_node in graph.topology(&[indices])? {
				if !matches!(
					graph.nodes[index_node].op,
					Op::Input { .. }
						| Op::Const { .. } | Op::Reshape { .. }
						| Op::Permute { .. }
						| Op::Expand { .. }
						| Op::Shrink { .. }
						| Op::Pad { .. } | Op::Flip { .. }
						| Op::Stack { .. } | Op::Stage { .. }
						| Op::StopGradient { .. }
				) {
					return Err("GPU computed index arithmetic is unsupported; materialize integer indices as an input".into());
				}
			}
		}
	}
	let kernels = kernel_bodies(graph, outputs, true)?;
	let dims = graph
		.nodes
		.iter()
		.flat_map(|n| n.shape.iter().map(|&v| v as u64))
		.collect::<Vec<_>>();
	let mut result = vec![];
	for k in kernels {
		let n = elements(graph.shape(k.output))?;
		let start = k.body.find('{').ok_or("missing outer loop")? + 1;
		let end = k.body.rfind('}').ok_or("missing loop end")?;
		let mut body = k.body[start..end].to_string();

		for (from, to) in [
			("expf(", "exp("),
			("logf(", "log("),
			("sinf(", "sin("),
			("cosf(", "cos("),
			("sqrtf(", "sqrt("),
			("exp2f(", "exp2("),
			("log2f(", "log2("),
			("floorf(", "floor("),
			("(double)", "(float)"),
			("uint32_t", "uint"),
		] {
			body = body.replace(from, to);
		}
		body = body.replace("return 2;", "return;");
		let mut slots = graph
			.topology(&[k.output])?
			.into_iter()
			.filter(|id| body.contains(&format!("b{id}[")))
			.collect::<Vec<_>>();
		slots.retain(|&id| id != k.output);
		slots.push(k.output);
		for (index, id) in slots.iter().enumerate() {
			body = body.replace(&format!("b{id}["), &format!("slot{index}["));
		}
		for index in 0..slots.len() {
			body = body.replace(&format!("slot{index}["), &format!("b{index}["));
		}
		let condition = k
			.body
			.split("i<")
			.nth(1)
			.and_then(|s| s.split("; ++i)").next())
			.ok_or("outer loop condition")?;
		let body = format!("size_t i=g*{local}+l;if(i<{condition}){{{body}}}");
		let mut source = header(d, slots.len(), &body);
		let helper = helpers(d);
		source = match d {
			Dialect::Metal => source.replacen("kernel void", &(helper + "kernel void"), 1),
			Dialect::OpenCl => helper + &source,
		};
		let lengths = slots
			.iter()
			.map(|&id| elements(graph.shape(id)))
			.collect::<Result<Vec<_>>>()?;
		result.push((
			Kernel {
				scalars: dims.clone(),
				source,
				entry: "chibi".into(),
				output: slots.len() - 1,
				lengths,
				groups: groups(n, local)?,
				local,
				shared_bytes: 0,
			},
			slots,
		));
	}
	Ok(result)
}
```

```sh
# check requires=metal
cargo test --test book_15 --features metal todo_15_017 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## matmul は一つの thread が一出力 `(m,n)` を担当する。

<!-- todo:15-018 -->

実装は[この章の `GpuPlan::compile、GpuPlan::run_with_shapes、gpu::groups、MetalProgram::compile、MetalProgram::run、MetalProgram::limits`](#metal-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_15.rs mode=append
#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
#[ignore = "requires real device"]
fn todo_15_018() {
	hardware_contract();
}
```

`src/graph.rs` の `pub fn matmul(`の定義を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="pub fn matmul("
	pub fn matmul(&mut self, a: NodeId, b: NodeId) -> Result<NodeId> {
		let sa = self.get(a)?.shape.clone();
		let sb = self.get(b)?.shape.clone();
		if sa.len() != 2 || sb.len() != 2 || sa[1] != sb[0] {
			return Err("matmul requires [M,K] and [K,N]".into());
		}
		let aa = self.reshape(a, vec![sa[0], 1, sa[1]])?;
		let bt = self.permute(b, vec![1, 0])?;
		let bb = self.reshape(bt, vec![1, sb[1], sb[0]])?;
		let products = self.binary(Binary::Mul, aa, bb)?;
		self.reduce(products, Reduce::Sum, vec![2], false)
	}
```

`src/graph.rs` の `fn expr(` の `Op::Reduce` 分岐を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn expr(" arm="Op::Reduce"
			Op::Reduce {
				op,
				x,
				axes,
				keepdim,
			} => {
				let target = self.name();
				let k = self.name();
				code.push_str(&format!(
					"float {target} = {};\nfor (size_t {k}=0; {k}<{}; ++{k}) {{\n",
					if op == Reduce::Sum {
						"0.0f"
					} else {
						"-INFINITY"
					},
					self.product(x, axes.iter().copied())
				));
				let mut out_axis = 0;
				let c = (0..self.g.shape(x).len())
					.map(|a| {
						if let Some(ra) = axes.iter().position(|&v| v == a) {
							format!(
								"(({k} / {}) % {})",
								self.product(x, axes[ra + 1..].iter().copied()),
								self.dim(x, a)
							)
						} else {
							let oa = if keepdim {
								a
							} else {
								let v = out_axis;
								out_axis += 1;
								v
							};
							self.coord(id, index, oa)
						}
					})
					.collect::<Vec<_>>();
				let xi = self.flatten(x, &c);
				let value = self.expr(x, &xi, code)?;
				code.push_str(&format!(
					"{target} = {};\n}}\n",
					if op == Reduce::Sum {
						format!("{target} + ({value})")
					} else {
						format!("chibi_max({target},{value})")
					}
				));
				return Ok(target);
			}
```

`src/gpu_graph.rs` の `pub fn lower(`の定義を次の全体で置き換えます。

```rust
// file="src/gpu_graph.rs" mode="replace" item="pub fn lower("
pub fn lower(
	graph: &Graph,
	outputs: &[usize],
	d: Dialect,
	local: usize,
) -> Result<Vec<(Kernel, Vec<usize>)>> {
	graph.verify(outputs)?;
	for id in graph.topology(outputs)? {
		if let Op::Gather { x, indices, axis }
		| Op::ScatterAdd {
			base: x,
			indices,
			axis,
			..
		} = graph.nodes[id].op
		{
			if graph.shape(x)[axis] > 16_777_216 {
				return Err("GPU f32 index extent exceeds exact integer range".into());
			}
			for index_node in graph.topology(&[indices])? {
				if !matches!(
					graph.nodes[index_node].op,
					Op::Input { .. }
						| Op::Const { .. } | Op::Reshape { .. }
						| Op::Permute { .. }
						| Op::Expand { .. }
						| Op::Shrink { .. }
						| Op::Pad { .. } | Op::Flip { .. }
						| Op::Stack { .. } | Op::Stage { .. }
						| Op::StopGradient { .. }
				) {
					return Err("GPU computed index arithmetic is unsupported; materialize integer indices as an input".into());
				}
			}
		}
	}
	let kernels = kernel_bodies(graph, outputs, true)?;
	let dims = graph
		.nodes
		.iter()
		.flat_map(|n| n.shape.iter().map(|&v| v as u64))
		.collect::<Vec<_>>();
	let mut result = vec![];
	for k in kernels {
		let n = elements(graph.shape(k.output))?;
		let start = k.body.find('{').ok_or("missing outer loop")? + 1;
		let end = k.body.rfind('}').ok_or("missing loop end")?;
		let mut body = k.body[start..end].to_string();

		for (from, to) in [
			("expf(", "exp("),
			("logf(", "log("),
			("sinf(", "sin("),
			("cosf(", "cos("),
			("sqrtf(", "sqrt("),
			("exp2f(", "exp2("),
			("log2f(", "log2("),
			("floorf(", "floor("),
			("(double)", "(float)"),
			("uint32_t", "uint"),
		] {
			body = body.replace(from, to);
		}
		body = body.replace("return 2;", "return;");
		let mut slots = graph
			.topology(&[k.output])?
			.into_iter()
			.filter(|id| body.contains(&format!("b{id}[")))
			.collect::<Vec<_>>();
		slots.retain(|&id| id != k.output);
		slots.push(k.output);
		for (index, id) in slots.iter().enumerate() {
			body = body.replace(&format!("b{id}["), &format!("slot{index}["));
		}
		for index in 0..slots.len() {
			body = body.replace(&format!("slot{index}["), &format!("b{index}["));
		}
		let condition = k
			.body
			.split("i<")
			.nth(1)
			.and_then(|s| s.split("; ++i)").next())
			.ok_or("outer loop condition")?;
		let body = format!("size_t i=g*{local}+l;if(i<{condition}){{{body}}}");
		let mut source = header(d, slots.len(), &body);
		let helper = helpers(d);
		source = match d {
			Dialect::Metal => source.replacen("kernel void", &(helper + "kernel void"), 1),
			Dialect::OpenCl => helper + &source,
		};
		let lengths = slots
			.iter()
			.map(|&id| elements(graph.shape(id)))
			.collect::<Result<Vec<_>>>()?;
		result.push((
			Kernel {
				scalars: dims.clone(),
				source,
				entry: "chibi".into(),
				output: slots.len() - 1,
				lengths,
				groups: groups(n, local)?,
				local,
				shared_bytes: 0,
			},
			slots,
		));
	}
	Ok(result)
}
```

```sh
# check requires=metal
cargo test --test book_15 --features metal todo_15_018 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## reduction の初期値を各 thread のローカル変数に置く。

<!-- todo:15-019 -->

実装は[この章の `GpuPlan::compile、GpuPlan::run_with_shapes、gpu::groups、MetalProgram::compile、MetalProgram::run、MetalProgram::limits`](#metal-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_15.rs mode=append
#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
#[ignore = "requires real device"]
fn todo_15_019() {
	hardware_contract();
}
```

`src/graph.rs` の `fn expr(` の `Op::Reduce` 分岐を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn expr(" arm="Op::Reduce"
			Op::Reduce {
				op,
				x,
				axes,
				keepdim,
			} => {
				let target = self.name();
				let k = self.name();
				code.push_str(&format!(
					"float {target} = {};\nfor (size_t {k}=0; {k}<{}; ++{k}) {{\n",
					if op == Reduce::Sum {
						"0.0f"
					} else {
						"-INFINITY"
					},
					self.product(x, axes.iter().copied())
				));
				let mut out_axis = 0;
				let c = (0..self.g.shape(x).len())
					.map(|a| {
						if let Some(ra) = axes.iter().position(|&v| v == a) {
							format!(
								"(({k} / {}) % {})",
								self.product(x, axes[ra + 1..].iter().copied()),
								self.dim(x, a)
							)
						} else {
							let oa = if keepdim {
								a
							} else {
								let v = out_axis;
								out_axis += 1;
								v
							};
							self.coord(id, index, oa)
						}
					})
					.collect::<Vec<_>>();
				let xi = self.flatten(x, &c);
				let value = self.expr(x, &xi, code)?;
				code.push_str(&format!(
					"{target} = {};\n}}\n",
					if op == Reduce::Sum {
						format!("{target} + ({value})")
					} else {
						format!("chibi_max({target},{value})")
					}
				));
				return Ok(target);
			}
```

`src/gpu_graph.rs` の `pub fn lower(`の定義を次の全体で置き換えます。

```rust
// file="src/gpu_graph.rs" mode="replace" item="pub fn lower("
pub fn lower(
	graph: &Graph,
	outputs: &[usize],
	d: Dialect,
	local: usize,
) -> Result<Vec<(Kernel, Vec<usize>)>> {
	graph.verify(outputs)?;
	for id in graph.topology(outputs)? {
		if let Op::Gather { x, indices, axis }
		| Op::ScatterAdd {
			base: x,
			indices,
			axis,
			..
		} = graph.nodes[id].op
		{
			if graph.shape(x)[axis] > 16_777_216 {
				return Err("GPU f32 index extent exceeds exact integer range".into());
			}
			for index_node in graph.topology(&[indices])? {
				if !matches!(
					graph.nodes[index_node].op,
					Op::Input { .. }
						| Op::Const { .. } | Op::Reshape { .. }
						| Op::Permute { .. }
						| Op::Expand { .. }
						| Op::Shrink { .. }
						| Op::Pad { .. } | Op::Flip { .. }
						| Op::Stack { .. } | Op::Stage { .. }
						| Op::StopGradient { .. }
				) {
					return Err("GPU computed index arithmetic is unsupported; materialize integer indices as an input".into());
				}
			}
		}
	}
	let kernels = kernel_bodies(graph, outputs, true)?;
	let dims = graph
		.nodes
		.iter()
		.flat_map(|n| n.shape.iter().map(|&v| v as u64))
		.collect::<Vec<_>>();
	let mut result = vec![];
	for k in kernels {
		let n = elements(graph.shape(k.output))?;
		let start = k.body.find('{').ok_or("missing outer loop")? + 1;
		let end = k.body.rfind('}').ok_or("missing loop end")?;
		let mut body = k.body[start..end].to_string();

		for (from, to) in [
			("expf(", "exp("),
			("logf(", "log("),
			("sinf(", "sin("),
			("cosf(", "cos("),
			("sqrtf(", "sqrt("),
			("exp2f(", "exp2("),
			("log2f(", "log2("),
			("floorf(", "floor("),
			("(double)", "(float)"),
			("uint32_t", "uint"),
		] {
			body = body.replace(from, to);
		}
		body = body.replace("return 2;", "return;");
		let mut slots = graph
			.topology(&[k.output])?
			.into_iter()
			.filter(|id| body.contains(&format!("b{id}[")))
			.collect::<Vec<_>>();
		slots.retain(|&id| id != k.output);
		slots.push(k.output);
		for (index, id) in slots.iter().enumerate() {
			body = body.replace(&format!("b{id}["), &format!("slot{index}["));
		}
		for index in 0..slots.len() {
			body = body.replace(&format!("slot{index}["), &format!("b{index}["));
		}
		let condition = k
			.body
			.split("i<")
			.nth(1)
			.and_then(|s| s.split("; ++i)").next())
			.ok_or("outer loop condition")?;
		let body = format!("size_t i=g*{local}+l;if(i<{condition}){{{body}}}");
		let mut source = header(d, slots.len(), &body);
		let helper = helpers(d);
		source = match d {
			Dialect::Metal => source.replacen("kernel void", &(helper + "kernel void"), 1),
			Dialect::OpenCl => helper + &source,
		};
		let lengths = slots
			.iter()
			.map(|&id| elements(graph.shape(id)))
			.collect::<Result<Vec<_>>>()?;
		result.push((
			Kernel {
				scalars: dims.clone(),
				source,
				entry: "chibi".into(),
				output: slots.len() - 1,
				lengths,
				groups: groups(n, local)?,
				local,
				shared_bytes: 0,
			},
			slots,
		));
	}
	Ok(result)
}
```

```sh
# check requires=metal
cargo test --test book_15 --features metal todo_15_019 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## CPU と同じ shape graph から reduction と matmul を作る。

<!-- todo:15-020 -->

実装は[この章の `GpuPlan::compile、GpuPlan::run_with_shapes、gpu::groups、MetalProgram::compile、MetalProgram::run、MetalProgram::limits`](#metal-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_15.rs mode=append
#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
#[ignore = "requires real device"]
fn todo_15_020() {
	hardware_contract();
}
```

`src/graph.rs` の `pub fn reduce(`の定義を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="pub fn reduce("
	pub fn reduce(
		&mut self,
		x: NodeId,
		op: Reduce,
		axes: Vec<usize>,
		keepdim: bool,
	) -> Result<NodeId> {
		let input = &self.get(x)?.shape;
		axes_valid(&axes, input.len())?;
		let shape = input
			.iter()
			.enumerate()
			.filter_map(|(i, &d)| {
				if axes.contains(&i) {
					if keepdim { Some(1) } else { None }
				} else {
					Some(d)
				}
			})
			.collect();
		self.insert(
			Op::Reduce {
				op,
				x,
				axes,
				keepdim,
			},
			shape,
		)
	}
```

`src/graph.rs` の `pub fn matmul(`の定義を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="pub fn matmul("
	pub fn matmul(&mut self, a: NodeId, b: NodeId) -> Result<NodeId> {
		let sa = self.get(a)?.shape.clone();
		let sb = self.get(b)?.shape.clone();
		if sa.len() != 2 || sb.len() != 2 || sa[1] != sb[0] {
			return Err("matmul requires [M,K] and [K,N]".into());
		}
		let aa = self.reshape(a, vec![sa[0], 1, sa[1]])?;
		let bt = self.permute(b, vec![1, 0])?;
		let bb = self.reshape(bt, vec![1, sb[1], sb[0]])?;
		let products = self.binary(Binary::Mul, aa, bb)?;
		self.reduce(products, Reduce::Sum, vec![2], false)
	}
```

`src/gpu_graph.rs` の `pub fn lower(`の定義を次の全体で置き換えます。

```rust
// file="src/gpu_graph.rs" mode="replace" item="pub fn lower("
pub fn lower(
	graph: &Graph,
	outputs: &[usize],
	d: Dialect,
	local: usize,
) -> Result<Vec<(Kernel, Vec<usize>)>> {
	graph.verify(outputs)?;
	for id in graph.topology(outputs)? {
		if let Op::Gather { x, indices, axis }
		| Op::ScatterAdd {
			base: x,
			indices,
			axis,
			..
		} = graph.nodes[id].op
		{
			if graph.shape(x)[axis] > 16_777_216 {
				return Err("GPU f32 index extent exceeds exact integer range".into());
			}
			for index_node in graph.topology(&[indices])? {
				if !matches!(
					graph.nodes[index_node].op,
					Op::Input { .. }
						| Op::Const { .. } | Op::Reshape { .. }
						| Op::Permute { .. }
						| Op::Expand { .. }
						| Op::Shrink { .. }
						| Op::Pad { .. } | Op::Flip { .. }
						| Op::Stack { .. } | Op::Stage { .. }
						| Op::StopGradient { .. }
				) {
					return Err("GPU computed index arithmetic is unsupported; materialize integer indices as an input".into());
				}
			}
		}
	}
	let kernels = kernel_bodies(graph, outputs, true)?;
	let dims = graph
		.nodes
		.iter()
		.flat_map(|n| n.shape.iter().map(|&v| v as u64))
		.collect::<Vec<_>>();
	let mut result = vec![];
	for k in kernels {
		let n = elements(graph.shape(k.output))?;
		let start = k.body.find('{').ok_or("missing outer loop")? + 1;
		let end = k.body.rfind('}').ok_or("missing loop end")?;
		let mut body = k.body[start..end].to_string();

		for (from, to) in [
			("expf(", "exp("),
			("logf(", "log("),
			("sinf(", "sin("),
			("cosf(", "cos("),
			("sqrtf(", "sqrt("),
			("exp2f(", "exp2("),
			("log2f(", "log2("),
			("floorf(", "floor("),
			("(double)", "(float)"),
			("uint32_t", "uint"),
		] {
			body = body.replace(from, to);
		}
		body = body.replace("return 2;", "return;");
		let mut slots = graph
			.topology(&[k.output])?
			.into_iter()
			.filter(|id| body.contains(&format!("b{id}[")))
			.collect::<Vec<_>>();
		slots.retain(|&id| id != k.output);
		slots.push(k.output);
		for (index, id) in slots.iter().enumerate() {
			body = body.replace(&format!("b{id}["), &format!("slot{index}["));
		}
		for index in 0..slots.len() {
			body = body.replace(&format!("slot{index}["), &format!("b{index}["));
		}
		let condition = k
			.body
			.split("i<")
			.nth(1)
			.and_then(|s| s.split("; ++i)").next())
			.ok_or("outer loop condition")?;
		let body = format!("size_t i=g*{local}+l;if(i<{condition}){{{body}}}");
		let mut source = header(d, slots.len(), &body);
		let helper = helpers(d);
		source = match d {
			Dialect::Metal => source.replacen("kernel void", &(helper + "kernel void"), 1),
			Dialect::OpenCl => helper + &source,
		};
		let lengths = slots
			.iter()
			.map(|&id| elements(graph.shape(id)))
			.collect::<Result<Vec<_>>>()?;
		result.push((
			Kernel {
				scalars: dims.clone(),
				source,
				entry: "chibi".into(),
				output: slots.len() - 1,
				lengths,
				groups: groups(n, local)?,
				local,
				shared_bytes: 0,
			},
			slots,
		));
	}
	Ok(result)
}
```

```sh
# check requires=metal
cargo test --test book_15 --features metal todo_15_020 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## 小型 MLP の前向きと勾配にも同じ backend を選べるようにする。

<!-- todo:15-021 -->

実装は[この章の `GpuPlan::compile、GpuPlan::run_with_shapes、gpu::groups、MetalProgram::compile、MetalProgram::run、MetalProgram::limits`](#metal-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_15.rs mode=append
#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
#[ignore = "requires real device"]
fn todo_15_021() {
	hardware_contract();
}
```

`src/gpu_graph.rs` の `fn from_lowered(`の定義を次の全体で置き換えます。

```rust
// file="src/gpu_graph.rs" mode="replace" item="fn from_lowered("
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

`src/gpu_graph.rs` の `pub fn run_with_shapes(`の定義を次の全体で置き換えます。

```rust
// file="src/gpu_graph.rs" mode="replace" item="pub fn run_with_shapes("
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

```sh
# check requires=metal
cargo test --test book_15 --features metal todo_15_021 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## CPU 用テストは Metal がない環境でも実行できる。

<!-- todo:15-022 -->

実装は[この章の `GpuPlan::compile、GpuPlan::run_with_shapes、gpu::groups、MetalProgram::compile、MetalProgram::run、MetalProgram::limits`](#metal-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_15.rs mode=append
#[test]
fn todo_15_022() {
	source_contract();
}
```

```sh
# check
cargo test --test book_15 todo_15_022 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 未対応機器の skip を、実機の数値比較成功として報告しない。

<!-- todo:15-023 -->

実装は[この章の `GpuPlan::compile、GpuPlan::run_with_shapes、gpu::groups、MetalProgram::compile、MetalProgram::run、MetalProgram::limits`](#metal-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_15.rs mode=append
#[test]
fn todo_15_023() {
	source_contract();
}
```

```sh
# check
cargo test --test book_15 todo_15_023 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## GPU の build error と command buffer の実行エラーを別々に伝える。

<!-- todo:15-024 -->

実装は[この章の `GpuPlan::compile、GpuPlan::run_with_shapes、gpu::groups、MetalProgram::compile、MetalProgram::run、MetalProgram::limits`](#metal-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_15.rs mode=append
#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
#[ignore = "requires real device"]
fn todo_15_024() {
	hardware_contract();
}
```

`src/backends/metal.rs` の `fn compile_on(`の定義を次の全体で置き換えます。

```rust
// file="src/backends/metal.rs" mode="replace" item="fn compile_on("
	fn compile_on(
		kernel: &Kernel,
		device: Device,
		queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
	) -> Result<Self, String> {
		autoreleasepool(|_| {
			let start = Instant::now();
			let options = MTLCompileOptions::new();
			#[allow(deprecated)]
			options.setFastMathEnabled(false);
			let library = device
				.newLibraryWithSource_options_error(
					&NSString::from_str(&kernel.source),
					Some(&options),
				)
				.map_err(|e| format!("Metal source build: {e}"))?;
			let function = library
				.newFunctionWithName(&NSString::from_str(&kernel.entry))
				.ok_or("Metal entry missing")?;
			let pipeline = device
				.newComputePipelineStateWithFunction_error(&function)
				.map_err(|e| format!("Metal pipeline build: {e}"))?;
			Ok(Self {
				device,
				queue,
				pipeline,
				compile_time: start.elapsed(),
				contract: (
					kernel.source.clone(),
					kernel.entry.clone(),
					kernel.lengths.len(),
					kernel.scalars.len().max(1),
					kernel.local,
				),
			})
		})
	}
```

`src/backends/metal.rs` の `pub unsafe fn run(&self,`の定義を次の全体で置き換えます。

```rust
// file="src/backends/metal.rs" mode="replace" item="pub unsafe fn run(&self,"
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

```sh
# check requires=metal
cargo test --test book_15 --features metal todo_15_024 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## 結果を読む前の完了待ちとリソースの寿命が runtime に集約される。

<!-- todo:15-025 -->

実装は[この章の `GpuPlan::compile、GpuPlan::run_with_shapes、gpu::groups、MetalProgram::compile、MetalProgram::run、MetalProgram::limits`](#metal-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_15.rs mode=append
#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
#[ignore = "requires real device"]
fn todo_15_025() {
	hardware_contract();
}
```

`src/backends/metal.rs` の `pub unsafe fn run(&self,`の定義を次の全体で置き換えます。

```rust
// file="src/backends/metal.rs" mode="replace" item="pub unsafe fn run(&self,"
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

```sh
# check requires=metal
cargo test --test book_15 --features metal todo_15_025 -- --exact --ignored
```

機器がある場合は実際の値と診断を確認する。機器がなければこの実機確認を `NOT_RUN` と記録する。

## MSL に未対応 UOp が残る場合はコンパイル前に拒否する。

<!-- todo:15-026 -->

実装は[この章の `GpuPlan::compile、GpuPlan::run_with_shapes、gpu::groups、MetalProgram::compile、MetalProgram::run、MetalProgram::limits`](#metal-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_15.rs mode=append
#[test]
fn todo_15_026() {
	source_contract();
}
```

`src/gpu_graph.rs` の `pub fn lower(`の定義を次の全体で置き換えます。

```rust
// file="src/gpu_graph.rs" mode="replace" item="pub fn lower("
pub fn lower(
	graph: &Graph,
	outputs: &[usize],
	d: Dialect,
	local: usize,
) -> Result<Vec<(Kernel, Vec<usize>)>> {
	graph.verify(outputs)?;
	for id in graph.topology(outputs)? {
		if let Op::Gather { x, indices, axis }
		| Op::ScatterAdd {
			base: x,
			indices,
			axis,
			..
		} = graph.nodes[id].op
		{
			if graph.shape(x)[axis] > 16_777_216 {
				return Err("GPU f32 index extent exceeds exact integer range".into());
			}
			for index_node in graph.topology(&[indices])? {
				if !matches!(
					graph.nodes[index_node].op,
					Op::Input { .. }
						| Op::Const { .. } | Op::Reshape { .. }
						| Op::Permute { .. }
						| Op::Expand { .. }
						| Op::Shrink { .. }
						| Op::Pad { .. } | Op::Flip { .. }
						| Op::Stack { .. } | Op::Stage { .. }
						| Op::StopGradient { .. }
				) {
					return Err("GPU computed index arithmetic is unsupported; materialize integer indices as an input".into());
				}
			}
		}
	}
	let kernels = kernel_bodies(graph, outputs, true)?;
	let dims = graph
		.nodes
		.iter()
		.flat_map(|n| n.shape.iter().map(|&v| v as u64))
		.collect::<Vec<_>>();
	let mut result = vec![];
	for k in kernels {
		let n = elements(graph.shape(k.output))?;
		let start = k.body.find('{').ok_or("missing outer loop")? + 1;
		let end = k.body.rfind('}').ok_or("missing loop end")?;
		let mut body = k.body[start..end].to_string();

		for (from, to) in [
			("expf(", "exp("),
			("logf(", "log("),
			("sinf(", "sin("),
			("cosf(", "cos("),
			("sqrtf(", "sqrt("),
			("exp2f(", "exp2("),
			("log2f(", "log2("),
			("floorf(", "floor("),
			("(double)", "(float)"),
			("uint32_t", "uint"),
		] {
			body = body.replace(from, to);
		}
		body = body.replace("return 2;", "return;");
		let mut slots = graph
			.topology(&[k.output])?
			.into_iter()
			.filter(|id| body.contains(&format!("b{id}[")))
			.collect::<Vec<_>>();
		slots.retain(|&id| id != k.output);
		slots.push(k.output);
		for (index, id) in slots.iter().enumerate() {
			body = body.replace(&format!("b{id}["), &format!("slot{index}["));
		}
		for index in 0..slots.len() {
			body = body.replace(&format!("slot{index}["), &format!("b{index}["));
		}
		let condition = k
			.body
			.split("i<")
			.nth(1)
			.and_then(|s| s.split("; ++i)").next())
			.ok_or("outer loop condition")?;
		let body = format!("size_t i=g*{local}+l;if(i<{condition}){{{body}}}");
		let mut source = header(d, slots.len(), &body);
		let helper = helpers(d);
		source = match d {
			Dialect::Metal => source.replacen("kernel void", &(helper + "kernel void"), 1),
			Dialect::OpenCl => helper + &source,
		};
		let lengths = slots
			.iter()
			.map(|&id| elements(graph.shape(id)))
			.collect::<Result<Vec<_>>>()?;
		result.push((
			Kernel {
				scalars: dims.clone(),
				source,
				entry: "chibi".into(),
				output: slots.len() - 1,
				lengths,
				groups: groups(n, local)?,
				local,
				shared_bytes: 0,
			},
			slots,
		));
	}
	Ok(result)
}
```

```sh
# check
cargo test --test book_15 todo_15_026 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 空の加算をテストし、Metal 呼出しなしで空の結果を得る。

<!-- todo:15-027 -->

実装は[この章の `GpuPlan::compile、GpuPlan::run_with_shapes、gpu::groups、MetalProgram::compile、MetalProgram::run、MetalProgram::limits`](#metal-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_15.rs mode=append
#[test]
fn todo_15_027() {
	source_contract();
}
```

```sh
# check
cargo test --test book_15 todo_15_027 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 確認記録

```sh
# check
cargo test
```

