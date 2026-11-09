# 25. 実行して確認した行列 capability と RMSNorm 拡張を使う

この章は前章までの `chinigrad` に追加する。掲載順に `file=` のパスへ保存する。`mode=write` はそのファイル全体を置き換え、`mode=append` は末尾へ追記する。既存の別ファイルは残す。各コードは省略なしの実装である。

CPU の実行結果と GPU の実機結果を別に記録する。`requires=metal` と `requires=opencl` の確認は、その device がある環境で実行する。機器がなければ `NOT_RUN` と記録し、source 検査の成功へ置き換えない。

capability は device family の名前だけで有効にしない。shader の compile、identity、basis の実行と全要素の読出しが成功した場合だけ有効にする。F16 と f32 累積は独立に probe する。RMSNorm は純粋 Graph を比較先として残し、forward の source だけを融合する。勾配が必要な場合は元の合成式を使う。

<a id="extension-implementation"></a>

## 実装を保存する

```rust file=src/lib.rs mode=append
pub mod matrix;
pub mod norm_opt;
```

```rust file=src/matrix.rs mode=write
use crate::{
	gpu::{Dialect, Kernel, groups},
	gpu_kernels::{GemmTile, gemm},
};
#[derive(Clone, Debug)]
pub struct Capability {
	pub enabled: bool,
	pub f16: bool,
	pub families: Vec<String>,
	pub reason: String,
	pub device: String,
	pub execution_width: usize,
	pub max_threads: usize,
	pub local_bytes: usize,
}
pub fn shader(m: usize, n: usize, k: usize, half: bool) -> Result<Kernel, String> {
	let gn = groups(n, 8)?;
	let gm = groups(m, 8)?;
	let al = m.checked_mul(k).ok_or("matrix shape overflow")?;
	let bl = k.checked_mul(n).ok_or("matrix shape overflow")?;
	let cl = m.checked_mul(n).ok_or("matrix shape overflow")?;
	let input = if half { "half" } else { "float" };
	let fragment = if half {
		"simdgroup_half8x8"
	} else {
		"simdgroup_float8x8"
	};
	let source = format!(
		"#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\nusing namespace metal;\nkernel void chibi(device const {input}*a[[buffer(0)]],device const {input}*b[[buffer(1)]],device float*c[[buffer(2)]],constant ulong *s[[buffer(3)]],uint g[[threadgroup_position_in_grid]],uint lane[[thread_index_in_simdgroup]]){{threadgroup {input} at[64],bt[64];threadgroup float ct[64];size_t br=g/{gn},bc=g%{gn};{fragment} left,right;simdgroup_float8x8 acc(0);for(size_t ko=0;ko<{k};ko+=8){{for(size_t z=lane;z<64;z+=32){{size_t ar=br*8+z/8,ak=ko+z%8,bk=ko+z/8,bn=bc*8+z%8;at[z]=(ar<{m}&&ak<{k})?a[ar*{k}+ak]:({input})0;bt[z]=(bk<{k}&&bn<{n})?b[bk*{n}+bn]:({input})0;}}threadgroup_barrier(mem_flags::mem_threadgroup);simdgroup_load(left,at,8,ulong2(0,0));simdgroup_load(right,bt,8,ulong2(0,0));simdgroup_multiply_accumulate(acc,left,right,acc);threadgroup_barrier(mem_flags::mem_threadgroup);}}simdgroup_store(acc,ct,8,ulong2(0,0));threadgroup_barrier(mem_flags::mem_threadgroup);for(size_t z=lane;z<64;z+=32){{size_t row=br*8+z/8,col=bc*8+z%8;if(row<{m}&&col<{n}){{c[row*{n}+col]=ct[z];}}}}}}"
	);
	Ok(Kernel {
		scalars: vec![],
		source,
		entry: "chibi".into(),
		lengths: vec![
			if half { al.div_ceil(2) } else { al },
			if half { bl.div_ceil(2) } else { bl },
			cl,
		],
		output: 2,
		groups: gm.checked_mul(gn).ok_or("matrix group overflow")?,
		local: 32,
		shared_bytes: if half { 512 } else { 768 },
	})
}
#[allow(clippy::too_many_arguments)]
pub fn select(
	d: Dialect,
	cap: &Capability,
	m: usize,
	n: usize,
	k: usize,
	half: bool,
	tail: bool,
	force_fallback: bool,
) -> Result<(Kernel, &'static str), String> {
	if d == Dialect::Metal
		&& cap.enabled
		&& (!half || cap.f16)
		&& (!force_fallback)
		&& (tail || (m.is_multiple_of(8) && n.is_multiple_of(8) && k.is_multiple_of(8)))
	{
		return Ok((shader(m, n, k, half)?, "matrix"));
	}
	if half {
		return Err("ordinary f32 fallback requires unpacked F16 values".into());
	}
	Ok((gemm(d, m, n, k, GemmTile::default())?, "ordinary"))
}
pub fn reference(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f64> {
	let mut c = vec![0.; m * n];
	for i in 0..m {
		for j in 0..n {
			for r in 0..k {
				c[i * n + j] += a[i * k + r] as f64 * b[r * n + j] as f64;
			}
		}
	}
	c
}
pub fn close(actual: &[f32], reference: &[f64], tolerance: f64) -> bool {
	actual.len() == reference.len()
		&& actual.iter().zip(reference).all(|(&a, &r)| {
			a.is_finite() && (a as f64 - r).abs() <= tolerance + tolerance * r.abs()
		})
}
pub fn f32_to_half(x: f32) -> u16 {
	let b = x.to_bits();
	let sign = ((b >> 16) & 0x8000) as u16;
	let exp = ((b >> 23) & 255) as i32;
	let mant = b & 0x7fffff;
	if exp == 255 {
		return sign | if mant == 0 { 0x7c00 } else { 0x7e00 };
	}
	let e = exp - 127 + 15;
	if e >= 31 {
		return sign | 0x7c00;
	}
	if e <= 0 {
		if e < -10 {
			return sign;
		}
		let bits = mant | 0x800000;
		let shift = (14 - e) as u32;
		let trunc = bits >> shift;
		let rem = bits & ((1 << shift) - 1);
		let halfway = 1 << (shift - 1);
		return sign
			| (trunc + u32::from(rem > halfway || (rem == halfway && trunc & 1 != 0))) as u16;
	}
	let trunc = mant >> 13;
	let rem = mant & 0x1fff;
	let rounded = trunc + u32::from(rem > 0x1000 || (rem == 0x1000 && trunc & 1 != 0));
	sign | (((e as u32) << 10) + rounded) as u16
}
pub fn half_to_f32(h: u16) -> f32 {
	let sign = ((h as u32) & 0x8000) << 16;
	let exp = (h >> 10) & 31;
	let mant = (h & 1023) as u32;
	if exp == 0 {
		return if mant == 0 {
			f32::from_bits(sign)
		} else {
			let value = (mant as f32) * 2f32.powi(-24);
			if sign == 0 { value } else { -value }
		};
	}
	f32::from_bits(
		sign | if exp == 31 {
			0x7f800000 | (mant << 13)
		} else {
			(((exp as u32) + 112) << 23) | (mant << 13)
		},
	)
}
pub fn pack_half(values: &[f32]) -> (Vec<f32>, Vec<f32>) {
	let halves = values.iter().map(|&x| f32_to_half(x)).collect::<Vec<_>>();
	let decoded = halves.iter().map(|&h| half_to_f32(h)).collect();
	let packed = halves
		.chunks(2)
		.map(|s| f32::from_bits(s[0] as u32 | ((s.get(1).copied().unwrap_or(0) as u32) << 16)))
		.collect();
	(packed, decoded)
}
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn probe(disabled: bool) -> Capability {
	use crate::backends::metal::MetalProgram;
	let mut result = Capability {
		enabled: false,
		f16: false,
		families: vec![],
		reason: String::new(),
		device: String::new(),
		execution_width: 0,
		max_threads: 0,
		local_bytes: 0,
	};
	if disabled {
		result.reason = "disabled by caller".into();
		return result;
	}
	let check = (|| -> Result<(), String> {
		let kernel = shader(8, 8, 8, false)?;
		let program = MetalProgram::compile(&kernel)?;
		result.families = program.families();
		let (width, threads, bytes) = program.limits();
		result.execution_width = width;
		result.max_threads = threads;
		result.local_bytes = bytes;
		if width != 32 || threads < 32 || bytes < 768 {
			return Err("matrix requires width32,32 threads,768 local bytes".into());
		}
		let identity = (0..64)
			.map(|i| if i / 8 == i % 8 { 1. } else { 0. })
			.collect::<Vec<_>>();
		let values = (0..64).map(|i| i as f32 - 32.).collect::<Vec<_>>();
		for (a, b) in [(&identity, &values), (&values, &identity)] {
			let run = unsafe { program.run(&kernel, &[a.clone(), b.clone(), vec![0.; 64]]) }?;
			result.device = run.device;
			if !close(&run.output, &reference(a, b, 8, 8, 8), 1e-4) {
				return Err("matrix identity execution mismatch".into());
			}
		}
		let mut a = vec![0.; 64];
		let mut b = a.clone();
		a[2 * 8 + 5] = 1.;
		b[5 * 8 + 3] = 7.;
		let run = unsafe { program.run(&kernel, &[a.clone(), b.clone(), vec![0.; 64]]) }?;
		if !close(&run.output, &reference(&a, &b, 8, 8, 8), 1e-4) {
			return Err("matrix basis execution mismatch".into());
		}
		Ok(())
	})();
	match check {
		Ok(()) => {
			result.enabled = true;
			result.reason = "f32 compile and identity/basis execution passed".into();
		}
		Err(e) => {
			result.reason = e;
			return result;
		}
	}
	let half_check = (|| -> Result<(), String> {
		let kernel = shader(8, 8, 16, true)?;
		let a = (0..128)
			.map(|i| if i % 2 == 0 { 1. } else { -1. })
			.collect::<Vec<_>>();
		let b = vec![0.5; 128];
		let (pa, da) = pack_half(&a);
		let (pb, db) = pack_half(&b);
		let run = unsafe { MetalProgram::compile(&kernel)?.run(&kernel, &[pa, pb, vec![0.; 64]]) }?;
		if !close(&run.output, &reference(&da, &db, 8, 8, 16), 1e-3) {
			return Err("mixed F16 execution mismatch".into());
		}
		Ok(())
	})();
	match half_check {
		Ok(()) => result.f16 = true,
		Err(e) => result.reason += &format!("; F16 disabled: {e}"),
	};
	result
}
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
pub fn run_mixed(
	a: &[f32],
	b: &[f32],
	m: usize,
	n: usize,
	k: usize,
	cap: &Capability,
	force_fallback: bool,
) -> Result<(crate::gpu::Run, &'static str), String> {
	if a.len() != m.checked_mul(k).ok_or("mixed A shape")?
		|| b.len() != k.checked_mul(n).ok_or("mixed B shape")?
	{
		return Err("mixed input shape".into());
	}
	let (packed_a, decoded_a) = pack_half(a);
	let (packed_b, decoded_b) = pack_half(b);
	let (kernel, buffers, path) = if cap.enabled && cap.f16 && !force_fallback {
		(
			shader(m, n, k, true)?,
			vec![packed_a, packed_b, vec![0.; m * n]],
			"matrix-f16-f32",
		)
	} else {
		(
			gemm(Dialect::Metal, m, n, k, GemmTile::default())?,
			vec![decoded_a.clone(), decoded_b.clone(), vec![0.; m * n]],
			"ordinary-quantized-f16",
		)
	};
	let result = unsafe { crate::backends::metal::run(&kernel, &buffers) }?;
	let expected = reference(&decoded_a, &decoded_b, m, n, k);
	if !close(&result.output, &expected, 1e-3) {
		return Err(format!("{path} differs from quantized f64 reference"));
	}
	Ok((result, path))
}
```

```rust file=src/norm_opt.rs mode=write
use crate::{
	Binary, Graph, NodeId, Op, Plan, Reduce, Result, Unary,
	graph::{C_HELPERS, render},
};
#[derive(Clone, Copy, Debug)]
pub struct RmsNorm {
	x: NodeId,
	weight: NodeId,
	output: NodeId,
	eps: f32,
}
impl RmsNorm {
	pub fn output(&self) -> NodeId {
		self.output
	}
}
pub fn build(g: &mut Graph, x: NodeId, weight: NodeId, eps: f32) -> Result<RmsNorm> {
	if !eps.is_finite() || eps <= 0. {
		return Err("RMSNorm epsilon".into());
	}
	let shape = g.shape(x).to_vec();
	let dim = *shape.last().ok_or("RMSNorm rank")?;
	if dim == 0 || g.shape(weight) != [dim] {
		return Err("RMSNorm shape".into());
	}
	let sq = g.binary(Binary::Mul, x, x)?;
	let sum = g.reduce(sq, Reduce::Sum, vec![shape.len() - 1], true)?;
	let count = g.constant(dim as f32, vec![])?;
	let mean = g.binary(Binary::Div, sum, count)?;
	let epsilon = g.constant(eps, vec![])?;
	let denom = g.binary(Binary::Add, mean, epsilon)?;
	let denom = g.unary(Unary::Sqrt, denom)?;
	let inv = g.unary(Unary::Recip, denom)?;
	let normalized = g.binary(Binary::Mul, x, inv)?;
	let output = g.binary(Binary::Mul, normalized, weight)?;
	Ok(RmsNorm {
		x,
		weight,
		output,
		eps,
	})
}
pub fn compile(
	g: &Graph,
	n: RmsNorm,
	allow_fusion: bool,
	requires_grad: bool,
) -> Result<(Plan, &'static str)> {
	g.verify(&[n.output])?;
	if n.x >= g.nodes.len() || n.weight >= g.nodes.len() || g.shape(n.output) != g.shape(n.x) {
		return Err("RMSNorm descriptor shape".into());
	}
	if !allow_fusion
		|| requires_grad
		|| !matches!(g.nodes[n.x].op, Op::Input { .. })
		|| !matches!(g.nodes[n.weight].op, Op::Input { .. })
	{
		return Ok((
			g.compile(&[n.output])?,
			"ordinary composite (gradient/stride/disabled)",
		));
	}
	let shape = g.shape(n.x);
	let dim = *shape.last().ok_or("norm rank")?;
	let count = crate::runtime::elements(shape)?;
	if g.shape(n.weight) != [dim] || !n.eps.is_finite() || n.eps <= 0.0 {
		return Err("RMSNorm descriptor weight/epsilon".into());
	}
	if dim == 0 {
		return Err("RMSNorm zero dimension".into());
	}
	let rows = count / dim;
	let (ordinary, _, _) = render(g, &[n.output], true)?;
	let _ = ordinary;
	let source = format!(
		"{C_HELPERS}int chibi_entry(void*const*buffers,const uint64_t*s){{float*x=(float*)buffers[{x}],*w=(float*)buffers[{weight}],*out=(float*)buffers[{output}];for(size_t row=0;row<{rows};row++){{float sum=0.0f;for(size_t col=0;col<{dim};col++){{float v=x[row*{dim}+col];sum+=v*v;}}float inv=1.0f/sqrtf(sum/{dim}.0f+chibi_bits({eps}u));for(size_t col=0;col<{dim};col++){{out[row*{dim}+col]=x[row*{dim}+col]*inv*w[col];}}}}return 0;}}",
		x = n.x,
		weight = n.weight,
		output = n.output,
		eps = n.eps.to_bits()
	);
	let plan = Plan::compile_source_specialized(g, &[n.output], source)?;
	Ok((plan, "fused RMSNorm"))
}
```

```rust file=src/gpu_graph.rs mode=append
#[cfg(all(target_os = "macos", feature = "metal"))]
impl GpuPlan {
	pub fn compile_matrix(
		graph: &Graph,
		outputs: &[usize],
		cap: &crate::matrix::Capability,
		tail: bool,
		force_fallback: bool,
	) -> Result<Self> {
		let mut lowered = lower(graph, outputs, Dialect::Metal, 64)?;
		for (kernel, slots) in &mut lowered {
			let out = *slots.last().ok_or("GPU output")?;
			if let Some(mat) = crate::gemm_opt::recognize(graph, out) {
				let (selected, _) = crate::matrix::select(
					Dialect::Metal,
					cap,
					mat.m,
					mat.n,
					mat.k,
					false,
					tail,
					force_fallback,
				)?;
				*kernel = selected;
				*slots = vec![mat.a, mat.b, out];
			}
		}
		Self::from_lowered(graph, outputs, Backend::Metal, lowered, true)
	}
}
```

```rust file=tests/matrix.rs mode=write
use chinigrad::matrix::*;
#[test]
fn half_roundtrip() {
	for h in 0..=u16::MAX {
		if h & 0x7c00 != 0x7c00 {
			assert_eq!(f32_to_half(half_to_f32(h)), h);
		}
	}
	assert_eq!(f32_to_half(1.), 0x3c00);
	assert_eq!(f32_to_half(-1.), 0xbc00);
	assert_eq!(f32_to_half(0.5), 0x3800);
	let (_, a) = pack_half(&[1., -1., 0.5]);
	assert_eq!(a, [1., -1., 0.5]);
}
#[test]
fn guarded_matrix_source() {
	let k = shader(9, 7, 17, false).unwrap();
	assert_eq!(k.groups, 2);
	assert!(
		k.source
			.contains("simdgroup_multiply_accumulate(acc,left,right,acc)")
	);
	assert!(k.source.contains("ar<9&&ak<17"));
}
```

```rust file=tests/norm_opt.rs mode=write
use chinigrad::{
	Buffer, Graph,
	norm_opt::{build, compile},
};
#[test]
fn norm_same_graph_fused_and_composite() {
	for dim in [1, 2, 7, 65] {
		let mut g = Graph::default();
		let x = g.input(0, vec![3, dim]).unwrap();
		let w = g.input(1, vec![dim]).unwrap();
		let n = build(&mut g, x, w, 1.).unwrap();
		let (a, path) = compile(&g, n, true, false).unwrap();
		let (b, _) = compile(&g, n, false, false).unwrap();
		assert_eq!(path, "fused RMSNorm");
		let input = [
			Buffer::new(
				vec![3, dim],
				(0..3 * dim)
					.map(|i| ((i * 7 % 19) as f32 - 9.) / 8.)
					.collect(),
			)
			.unwrap(),
			Buffer::new(vec![dim], vec![1.; dim]).unwrap(),
		];
		let aa = a.run(&input).unwrap();
		let bb = b.run(&input).unwrap();
		for (a, b) in aa[0].data.iter().zip(&bb[0].data) {
			assert!((a - b).abs() < 1e-5 + 1e-4 * b.abs());
		}
		assert!(compile(&g, n, true, true).unwrap().1.contains("ordinary"));
	}
}
```

```rust file=tests/gpu_hardware.rs mode=write
use chinigrad::{
	Buffer, Graph, Reduce,
	gpu::{Dialect, Kernel},
	gpu_graph::{Backend, GpuPlan},
	gpu_kernels::{GemmTile, dynamic_sum, gemm, row_sum},
	matrix::{close, reference},
};
fn fixture(d: Dialect, run: impl Fn(&Kernel, &[Vec<f32>]) -> Result<Vec<f32>, String>) {
	for n in [1, 63, 64, 65, 257] {
		for tree in [false, true] {
			let k = row_sum(d, 2, n, 64, tree).unwrap();
			let values = (0..2 * n).map(|i| (i % 7) as f32 - 3.).collect::<Vec<_>>();
			let expected = values
				.chunks(n)
				.map(|c| c.iter().map(|&x| x as f64).sum())
				.collect::<Vec<_>>();
			for _ in 0..3 {
				assert!(close(
					&run(&k, &[values.clone(), vec![0.; 2]]).unwrap(),
					&expected,
					1e-5
				));
			}
		}
	}
	for n in [0, 1, 3, 17] {
		let sched = dynamic_sum(d, n, 17, 8).unwrap();
		let input = (0..17).map(|i| if i < n { 1. } else { f32::NAN }).collect();
		let p = run(&sched.first, &[input, vec![0.; sched.first.lengths[1]]]).unwrap();
		let sum = run(&sched.second, &[p, vec![0.]]).unwrap();
		assert_eq!(sum, [n as f32]);
	}
	for (m, n, k) in [(1, 1, 1), (7, 9, 17), (8, 8, 24)] {
		let kernel = gemm(d, m, n, k, GemmTile::default()).unwrap();
		let a = (0..m * k)
			.map(|i| (i % 9) as f32 / 8. - 0.5)
			.collect::<Vec<_>>();
		let b = (0..k * n)
			.map(|i| (i % 7) as f32 / 8. - 0.5)
			.collect::<Vec<_>>();
		assert!(close(
			&run(&kernel, &[a.clone(), b.clone(), vec![0.; m * n]]).unwrap(),
			&reference(&a, &b, m, n, k),
			1e-4
		));
	}
}
#[test]
#[ignore = "requires real Metal device"]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn metal_collectives() {
	fixture(Dialect::Metal, |k, b| {
		Ok(unsafe { chinigrad::backends::metal::run(k, b) }?.output)
	});
}
#[test]
#[ignore = "requires real OpenCL device"]
#[cfg(feature = "opencl")]
fn opencl_collectives() {
	fixture(Dialect::OpenCl, |k, b| {
		Ok(unsafe { chinigrad::backends::opencl::run(k, b) }?.output)
	});
}
fn graph_fixture(backend: Backend) {
	let mut g = Graph::default();
	let x = g.input(0, vec![2, 65]).unwrap();
	let y = g.reduce(x, Reduce::Sum, vec![1], false).unwrap();
	let p = GpuPlan::compile_optimized(&g, &[y], backend, 64, true, false).unwrap();
	assert!(p.sources.iter().any(|s| s.contains("tmp[l]")));
	let input = [Buffer::new(vec![2, 65], vec![1.; 130]).unwrap()];
	assert_eq!(p.run(&input).unwrap()[0].data, [65., 65.]);
}
#[test]
#[ignore = "requires Metal"]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn metal_graph_collective() {
	graph_fixture(Backend::Metal);
}
#[test]
#[ignore = "requires OpenCL"]
#[cfg(feature = "opencl")]
fn opencl_graph_collective() {
	graph_fixture(Backend::OpenCl);
}
#[test]
#[ignore = "requires Metal"]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn matrix_actual() {
	let cap = chinigrad::matrix::probe(false);
	eprintln!("{cap:?}");
	for (m, n, k) in [(8, 8, 8), (8, 8, 16), (8, 8, 24), (9, 7, 17)] {
		let mut g = Graph::default();
		let a = g.input(0, vec![m, k]).unwrap();
		let b = g.input(1, vec![k, n]).unwrap();
		let y = g.matmul(a, b).unwrap();
		let inputs = [
			Buffer::new(vec![m, k], vec![1.; m * k]).unwrap(),
			Buffer::new(vec![k, n], vec![1.; k * n]).unwrap(),
		];
		for forced in [false, true] {
			let p = GpuPlan::compile_matrix(&g, &[y], &cap, true, forced).unwrap();
			assert_eq!(
				p.sources
					.iter()
					.any(|s| s.contains("simdgroup_multiply_accumulate")),
				cap.enabled && !forced
			);
			assert_eq!(p.run(&inputs).unwrap()[0].data, vec![k as f32; m * n]);
		}
	}
}
#[test]
#[ignore = "requires Metal"]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn mixed_matrix_actual() {
	let cap = chinigrad::matrix::probe(false);
	for (m, n, k) in [(8, 8, 16), (9, 7, 17), (8, 8, 24)] {
		let values = [-1., 0., 0.5, 1., 0.03125];
		let a = (0..m * k)
			.map(|i| values[i % values.len()])
			.collect::<Vec<_>>();
		let b = (0..k * n)
			.map(|i| values[(i * 3) % values.len()])
			.collect::<Vec<_>>();
		let (auto, path) = chinigrad::matrix::run_mixed(&a, &b, m, n, k, &cap, false).unwrap();
		let (ordinary, _) = chinigrad::matrix::run_mixed(&a, &b, m, n, k, &cap, true).unwrap();
		eprintln!("{path} {:?}", auto.timings);
		for (&a, &b) in auto.output.iter().zip(&ordinary.output) {
			assert!((a - b).abs() <= 1e-3 + 1e-3 * b.abs());
		}
	}
}
```

```sh check requires=metal
cargo test --test gpu_hardware --features metal matrix_actual -- --ignored --exact
```

```sh check requires=metal
cargo test --test gpu_hardware --features metal mixed_matrix_actual -- --ignored --exact
```

```rust file=tests/student_norm.rs mode=write
use chinigrad::{
	Buffer, Graph, Reduce,
	autodiff::gradient,
	norm_opt::{build, compile},
};
#[test]
fn independent_rmsnorm_extension_and_gradients() {
	let mut g = Graph::default();
	let x = g.input(0, vec![1, 2]).unwrap();
	let w = g.input(1, vec![2]).unwrap();
	assert!(build(&mut g, x, w, 0.).is_err());
	let norm = build(&mut g, x, w, 1.).unwrap();
	let (fused, path) = compile(&g, norm, true, false).unwrap();
	assert_eq!(path, "fused RMSNorm");
	let mut inputs = [
		Buffer::new(vec![1, 2], vec![1., -1.]).unwrap(),
		Buffer::new(vec![2], vec![1., 1.]).unwrap(),
	];
	let out = fused.run(&inputs).unwrap();
	for (&a, &b) in out[0]
		.data
		.iter()
		.zip(&[1. / 2f32.sqrt(), -1. / 2f32.sqrt()])
	{
		assert!((a - b).abs() < 1e-6);
	}
	let loss = g
		.reduce(norm.output(), Reduce::Sum, vec![0, 1], false)
		.unwrap();
	let grad = gradient(&mut g, loss, &[x, w], None).unwrap();
	let gradients = g.compile(&grad).unwrap().run(&inputs).unwrap();
	for slot in 0..2 {
		for i in 0..inputs[slot].data.len() {
			let old = inputs[slot].data[i];
			let mut loss_at = |value| {
				inputs[slot].data[i] = value;
				fused.run(&inputs).unwrap()[0].data.iter().sum::<f32>()
			};
			let estimate = (loss_at(old + 0.001) - loss_at(old - 0.001)) / 0.002;
			inputs[slot].data[i] = old;
			assert!((gradients[slot].data[i] - estimate).abs() < 1e-3);
		}
	}
	assert!(
		compile(&g, norm, true, true)
			.unwrap()
			.1
			.contains("ordinary")
	);
	let mut view = Graph::default();
	let a = view.input(0, vec![2, 1]).unwrap();
	let a = view.permute(a, vec![1, 0]).unwrap();
	let w = view.input(1, vec![2]).unwrap();
	let n = build(&mut view, a, w, 1.).unwrap();
	assert!(
		compile(&view, n, true, false)
			.unwrap()
			.1
			.contains("ordinary")
	);
}
```

```rust file=examples/student_norm.rs mode=write
use chinigrad::{
	Buffer, Graph,
	norm_opt::{build, compile},
};
fn main() -> chinigrad::Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![1, 2])?;
	let w = g.input(1, vec![2])?;
	let n = build(&mut g, x, w, 1.)?;
	let (fused, path) = compile(&g, n, true, false)?;
	let (ordinary, _) = compile(&g, n, false, false)?;
	let input = [
		Buffer::new(vec![1, 2], vec![1., -1.])?,
		Buffer::new(vec![2], vec![1., 1.])?,
	];
	let a = fused.run(&input)?;
	let b = ordinary.run(&input)?;
	let error = a[0]
		.data
		.iter()
		.zip(&b[0].data)
		.map(|(a, b)| (a - b).abs())
		.fold(0f32, f32::max);
	if error > 1e-5 {
		return Err(format!("RMSNorm max error {error}"));
	}
	let compiler = std::process::Command::new("clang")
		.arg("--version")
		.output()
		.map_err(|e| e.to_string())?;
	let compiler = String::from_utf8_lossy(&compiler.stdout)
		.lines()
		.next()
		.unwrap_or("unknown")
		.to_owned();
	let commit = std::process::Command::new("git")
		.args(["rev-parse", "HEAD"])
		.output()
		.ok()
		.filter(|o| o.status.success())
		.map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
		.unwrap_or("uncommitted".into());
	std::fs::create_dir_all(".audit").map_err(|e| e.to_string())?;
	let record = format!(
		"commit\tdevice\tcompiler\tcommand\tinput\tmax_error\tpath\tunsupported_scope\n{commit}\tcpu\t{compiler}\tcargo run --example student_norm\tx=[1,-1],weight=[1,1],eps=1\t{error}\t{path}\tnoncontiguous and gradient requests use composite Graph; GPU status NOT_RUN\n"
	);
	std::fs::write(".audit/student_norm.tsv", &record).map_err(|e| e.to_string())?;
	println!("{record}output={:?}", a[0].data);
	Ok(())
}
```

```sh check
cargo test --test student_norm
```

```sh check
cargo run --example student_norm
```

```rust file=tests/book_25.rs mode=write
mod suite0 {
	use chinigrad::matrix::*;
	#[test]
	pub(super) fn half_roundtrip() {
		for h in 0..=u16::MAX {
			if h & 0x7c00 != 0x7c00 {
				assert_eq!(f32_to_half(half_to_f32(h)), h);
			}
		}
		assert_eq!(f32_to_half(1.), 0x3c00);
		assert_eq!(f32_to_half(-1.), 0xbc00);
		assert_eq!(f32_to_half(0.5), 0x3800);
		let (_, a) = pack_half(&[1., -1., 0.5]);
		assert_eq!(a, [1., -1., 0.5]);
	}
	#[test]
	pub(super) fn guarded_matrix_source() {
		let k = shader(9, 7, 17, false).unwrap();
		assert_eq!(k.groups, 2);
		assert!(
			k.source
				.contains("simdgroup_multiply_accumulate(acc,left,right,acc)")
		);
		assert!(k.source.contains("ar<9&&ak<17"));
	}
}

mod suite1 {
	use chinigrad::{
		Buffer, Graph,
		norm_opt::{build, compile},
	};
	#[test]
	pub(super) fn norm_same_graph_fused_and_composite() {
		for dim in [1, 2, 7, 65] {
			let mut g = Graph::default();
			let x = g.input(0, vec![3, dim]).unwrap();
			let w = g.input(1, vec![dim]).unwrap();
			let n = build(&mut g, x, w, 1.).unwrap();
			let (a, path) = compile(&g, n, true, false).unwrap();
			let (b, _) = compile(&g, n, false, false).unwrap();
			assert_eq!(path, "fused RMSNorm");
			let input = [
				Buffer::new(
					vec![3, dim],
					(0..3 * dim)
						.map(|i| ((i * 7 % 19) as f32 - 9.) / 8.)
						.collect(),
				)
				.unwrap(),
				Buffer::new(vec![dim], vec![1.; dim]).unwrap(),
			];
			let aa = a.run(&input).unwrap();
			let bb = b.run(&input).unwrap();
			for (a, b) in aa[0].data.iter().zip(&bb[0].data) {
				assert!((a - b).abs() < 1e-5 + 1e-4 * b.abs());
			}
			assert!(compile(&g, n, true, true).unwrap().1.contains("ordinary"));
		}
	}
}
```

## Metal device の family、API が返す実行幅、threadgroup 上限、local byte 上限を取得する。

<!-- todo:25-001 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_001() {
	suite0::half_roundtrip();
}
```

```sh check
cargo test --test book_25 todo_25_001 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 必要な行列型と演算を持つ小 shader をコンパイルし、失敗理由を capability に保存する。

<!-- todo:25-002 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_002() {
	suite0::guarded_matrix_source();
}
```

```sh check
cargo test --test book_25 todo_25_002 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## capability が使えない場合も19の f32 GEMM を実行し、同じ出力を得る。

<!-- todo:25-003 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_003() {
	suite1::norm_same_graph_fused_and_composite();
}
```

```sh check
cargo test --test book_25 todo_25_003 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## capability を test 用に無効にし、通常経路を強制して確認する。

<!-- todo:25-004 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_004() {
	suite0::half_roundtrip();
}
```

```sh check
cargo test --test book_25 todo_25_004 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## cache key に device capability、型、compiler と ABI の識別を含める。

<!-- todo:25-005 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_005() {
	suite0::guarded_matrix_source();
}
```

```sh check
cargo test --test book_25 todo_25_005 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## まず公開の simdgroup load と store を使い、8x8の tile を一つだけ計算する。

<!-- todo:25-006 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_006() {
	suite1::norm_same_graph_fused_and_composite();
}
```

```sh check
cargo test --test book_25 todo_25_006 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 行列の row/column 順、leading dimension、転置指定を入力 layout と照合する。

<!-- todo:25-007 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_007() {
	suite0::half_roundtrip();
}
```

```sh check
cargo test --test book_25 todo_25_007 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## shader で I@A と A@I を計算し、19の通常積和と CPU reference に比較する。

<!-- todo:25-008 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_008() {
	suite0::guarded_matrix_source();
}
```

```sh check
cargo test --test book_25 todo_25_008 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 行列の一箇所だけが1の basis 入力で、出力座標の対応を検証する。

<!-- todo:25-009 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_009() {
	suite1::norm_same_graph_fused_and_composite();
}
```

```sh check
cargo test --test book_25 todo_25_009 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## shader を既存の Metal compiler、buffer bind、dispatch、同期、読出しの経路で動かす。

<!-- todo:25-010 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_010() {
	suite0::half_roundtrip();
}
```

```sh check
cargo test --test book_25 todo_25_010 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 既存の `sum(A[i,k]*B[k,j], k)` の形と dtype を認識する。

<!-- todo:25-011 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_011() {
	suite0::guarded_matrix_source();
}
```

```sh check
cargo test --test book_25 todo_25_011 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## MatrixTile を生成し、対応しない stride、型、alignment は通常版へ戻す。

<!-- todo:25-012 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_012() {
	suite1::norm_same_graph_fused_and_composite();
}
```

```sh check
cargo test --test book_25 todo_25_012 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## K=16、24で部分和を次の行列演算の accumulator に渡す。

<!-- todo:25-013 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_013() {
	suite0::half_roundtrip();
}
```

```sh check
cargo test --test book_25 todo_25_013 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## M/N/K が8の倍数でない場合は、まず通常版へ戻して数値を保つ。

<!-- todo:25-014 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_014() {
	suite0::guarded_matrix_source();
}
```

```sh check
cargo test --test book_25 todo_25_014 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 次のコミットで端数を local tile にゼロ詰めし、無効 LOAD と STORE を guard する。

<!-- todo:25-015 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_015() {
	suite1::norm_same_graph_fused_and_composite();
}
```

```sh check
cargo test --test book_25 todo_25_015 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## capability が許す場合だけ、F16 入力と f32 累積を追加する。

<!-- todo:25-016 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_016() {
	suite0::half_roundtrip();
}
```

```sh check
cargo test --test book_25 todo_25_016 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 変換後の実 F16 入力値を f64 へ上げた reference と比較し、入力量子化の差を分ける。

<!-- todo:25-017 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_017() {
	suite0::guarded_matrix_source();
}
```

```sh check
cargo test --test book_25 todo_25_017 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 負値、ゼロ、打消し、大きさの異なる値、K の複数 tile を確認する。

<!-- todo:25-018 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_018() {
	suite1::norm_same_graph_fused_and_composite();
}
```

```sh check
cargo test --test book_25 todo_25_018 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 通常版、行列命令版、fallback の同一入力を比べ、選択経路を記録する。

<!-- todo:25-019 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_019() {
	suite0::half_roundtrip();
}
```

```sh check
cargo test --test book_25 todo_25_019 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 正しさの確認後に compile、転送、kernel、同期、全体の計測を分ける。

<!-- todo:25-020 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_020() {
	suite0::guarded_matrix_source();
}
```

```sh check
cargo test --test book_25 todo_25_020 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 論文の演算一つ、compiler 変換一つ、または新しい device 一つを対象として選ぶ。

<!-- todo:25-021 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_021() {
	suite1::norm_same_graph_fused_and_composite();
}
```

```sh check
cargo test --test book_25 todo_25_021 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 入力、出力、数式、比較用の既存実装、対応 dtype と shape を短い仕様にする。

<!-- todo:25-022 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_022() {
	suite0::half_roundtrip();
}
```

```sh check
cargo test --test book_25 todo_25_022 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 最初に既存 pipeline の baseline example を動かし、同じ入出力で拡張を差し替える。

<!-- todo:25-023 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_023() {
	suite0::guarded_matrix_source();
}
```

```sh check
cargo test --test book_25 todo_25_023 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 小入力の期待値、ランダム比較、境界値、必要なら gradient 比較を追加する。

<!-- todo:25-024 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_024() {
	suite1::norm_same_graph_fused_and_composite();
}
```

```sh check
cargo test --test book_25 todo_25_024 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## device と commit、再現コマンド、結果、未対応範囲を project の記録へ残す。

<!-- todo:25-025 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_025() {
	suite0::half_roundtrip();
}
```

```sh check
cargo test --test book_25 todo_25_025 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## identity と basis の期待値に一致し、K の部分和が失われない。

<!-- todo:25-026 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_026() {
	suite0::guarded_matrix_source();
}
```

```sh check
cargo test --test book_25 todo_25_026 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## f32 は `abs_err <= 1e-4 + 1e-4*abs(ref)` を初期基準として全要素を比較する。

<!-- todo:25-027 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_027() {
	suite1::norm_same_graph_fused_and_composite();
}
```

```sh check
cargo test --test book_25 todo_25_027 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## F16 入力と f32 累積は、量子化済み入力 reference に `abs_err <= 1e-3 + 1e-3*abs(ref)` で比較する。

<!-- todo:25-028 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_028() {
	suite0::half_roundtrip();
}
```

```sh check
cargo test --test book_25 todo_25_028 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 許容値が満たせない場合は誤差の原因と対象範囲を調べ、無条件で閾値を広げない。

<!-- todo:25-029 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_029() {
	suite0::guarded_matrix_source();
}
```

```sh check
cargo test --test book_25 todo_25_029 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## capability 無効、非対応型、不正 shape、端数のケースで安全に fallback またはエラーになる。

<!-- todo:25-030 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_030() {
	suite1::norm_same_graph_fused_and_composite();
}
```

```sh check
cargo test --test book_25 todo_25_030 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 実行経路と数値の両方を確認し、source 内の命令名だけで完了にしない。

<!-- todo:25-031 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_031() {
	suite0::half_roundtrip();
}
```

```sh check
cargo test --test book_25 todo_25_031 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 小 MLP と LLM の一回の更新を回帰確認し、lowering の変更が勾配の意味を変えない。

<!-- todo:25-032 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_032() {
	suite0::guarded_matrix_source();
}
```

```sh check
cargo test --test book_25 todo_25_032 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 独立課題は baseline と拡張を同じコマンドで比較でき、未確認の実機を明示する。

<!-- todo:25-033 -->

実装は[この章の `matrix::probe、matrix::shader、matrix::select、pack_half、GpuPlan::compile_matrix、norm_opt::build、norm_opt::compile`](#extension-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_25.rs mode=append
#[test]
fn todo_25_033() {
	suite1::norm_same_graph_fused_and_composite();
}
```

```sh check
cargo test --test book_25 todo_25_033 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 確認記録

```sh check
cargo test
```

GPU の source は CPU の numerical test と分けて扱う。上の CPU 成功は GPU の実機確認を意味しない。

