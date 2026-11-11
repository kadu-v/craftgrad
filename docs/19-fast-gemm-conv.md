# 19. GEMM と conv のタイル、確保の再利用、compile cache を動かす

この章は前章までの `chibigrad` に追加する。掲載順に `file=` のパスへ保存する。`mode=write` はそのファイル全体を置き換え、`mode=append` は末尾へ追記する。既存の別ファイルは残す。各コードは省略なしの実装である。

CPU の実行結果と GPU の実機結果を別に記録する。`requires=metal` と `requires=opencl` の確認は、その device がある環境で実行する。機器がなければ `NOT_RUN` と記録し、source 検査の成功へ置き換えない。

GEMM は元の MUL と SUM を残して kernel だけを置き換える。認識対象は rank 2 の Input または Stage の row-major 行列である。非連続 view は通常の座標 lowering へ戻る。conv は NCHW と OIHW、groups=1、dilation=1 を扱い、stride と padding を明示する。

確保の再利用は CPU の materialized storage を対象にする。同じ byte 数と f32 alignment の領域だけを使い回す。入力、公開出力、呼出し側が指定した external storage は保護する。GPU runtime は各 dispatch で転送し完了を待つ。GPU の常駐中間 buffer pool と永続 binary cache はこの実装の対象に含めない。

<a id="gemm-implementation"></a>

## 実装を保存する

```rust
// file=src/lib.rs mode=append
pub mod conv_opt;
pub mod gemm_opt;
pub mod storage;
```

```rust
// file=src/gemm_opt.rs mode=write
use crate::{
	Binary, Graph, Op, Plan, Reduce, Result,
	graph::kernel_bodies,
	loop_opt::{Options, compile_bodies},
};
#[derive(Clone, Copy, Debug)]
pub struct Gemm {
	pub a: usize,
	pub b: usize,
	pub output: usize,
	pub m: usize,
	pub n: usize,
	pub k: usize,
}
pub fn recognize(g: &Graph, id: usize) -> Option<Gemm> {
	let Op::Reduce {
		op: Reduce::Sum,
		x,
		ref axes,
		keepdim: false,
	} = g.nodes.get(id)?.op
	else {
		return None;
	};
	if axes != &[2] {
		return None;
	}
	let Op::Binary {
		op: Binary::Mul,
		a: aa,
		b: bb,
	} = g.nodes.get(x)?.op
	else {
		return None;
	};
	let Op::Reshape { x: a } = g.nodes.get(aa)?.op else {
		return None;
	};
	let Op::Reshape { x: bt } = g.nodes.get(bb)?.op else {
		return None;
	};
	let Op::Permute { x: b, ref axes } = g.nodes.get(bt)?.op else {
		return None;
	};
	if axes != &[1, 0] {
		return None;
	}
	if !matches!(g.nodes[a].op, Op::Input { .. } | Op::Stage { .. })
		|| !matches!(g.nodes[b].op, Op::Input { .. } | Op::Stage { .. })
	{
		return None;
	}
	let sa = g.shape(a);
	let sb = g.shape(b);
	if sa.len() != 2 || sb.len() != 2 || sa[1] != sb[0] {
		return None;
	}
	Some(Gemm {
		a,
		b,
		output: id,
		m: sa[0],
		n: sb[1],
		k: sa[1],
	})
}
pub fn compile(
	g: &Graph,
	outputs: &[usize],
	tile: (usize, usize, usize),
	workers: usize,
) -> Result<(Plan, String)> {
	let (tm, tn, tk) = tile;
	if tm == 0 || tn == 0 || tk == 0 {
		return Err("GEMM zero tile".into());
	}
	tm.checked_mul(tn)
		.and_then(|n| n.checked_mul(4))
		.filter(|&n| n <= 65536)
		.ok_or("register tile too large")?;
	let mut kernels = kernel_bodies(g, outputs, true)?;
	let mut matches = 0;
	let mut offsets = vec![];
	let mut offset = 0;
	for n in &g.nodes {
		offsets.push(offset);
		offset += n.shape.len();
	}
	for kernel in &mut kernels {
		if let Some(mat) = recognize(g, kernel.output) {
			let a = mat.a;
			let b = mat.b;
			let out = mat.output;
			let m = format!("s[{}]", offsets[a]);
			let k = format!("s[{}]", offsets[a] + 1);
			let n = format!("s[{}]", offsets[b] + 1);
			let gm = format!("({m}/{tm}+({m}%{tm}!=0))");
			let gn = format!("({n}/{tn}+({n}%{tn}!=0))");
			kernel.body = format!(
				"for (size_t i=0; i<{gm}*{gn}; ++i) {{size_t br=i/{gn},bc=i%{gn};float acc[{tm}][{tn}]={{0}};for(size_t ko=0;ko<{k};){{size_t klimit={k}-ko<{tk}?{k}-ko:{tk};for(size_t ki=0;ki<klimit;ki++){{size_t r=ko+ki;for(size_t mi=0;mi<{tm};mi++){{size_t row=br*{tm}+mi;if(row<{m}){{for(size_t ni=0;ni<{tn};ni++){{size_t col=bc*{tn}+ni;if(col<{n}){{acc[mi][ni]+=b{a}[row*{k}+r]*b{b}[r*{n}+col];}}}}}}}}}}ko+=klimit;}}for(size_t mi=0;mi<{tm};mi++){{size_t row=br*{tm}+mi;for(size_t ni=0;ni<{tn};ni++){{size_t col=bc*{tn}+ni;if(row<{m}&&col<{n}){{b{out}[row*{n}+col]=acc[mi][ni];}}}}}}}}\n"
			);
			matches += 1;
		}
	}
	let (plan, report) = compile_bodies(
		g,
		outputs,
		Options {
			workers,
			..Options::default()
		},
		kernels,
	)?;
	Ok((
		plan,
		format!("GEMM matches={matches}, tile={tile:?}; {report}"),
	))
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Flops {
	pub useful: u128,
	pub padded: u128,
}
pub fn flops(m: usize, n: usize, k: usize, tile: (usize, usize, usize)) -> Result<Flops> {
	let (tm, tn, tk) = tile;
	if tm == 0 || tn == 0 || tk == 0 {
		return Err("zero tile".into());
	}
	let useful = 2u128
		.checked_mul(m as u128)
		.and_then(|v| v.checked_mul(n as u128))
		.and_then(|v| v.checked_mul(k as u128))
		.ok_or("FLOP overflow")?;
	let padded = [
		m.div_ceil(tm) as u128 * tm as u128,
		n.div_ceil(tn) as u128 * tn as u128,
		k.div_ceil(tk) as u128 * tk as u128,
	]
	.into_iter()
	.try_fold(2u128, |acc, n| {
		acc.checked_mul(n).ok_or("padded FLOP overflow")
	})?;
	Ok(Flops { useful, padded })
}
```

```rust
// file=src/conv_opt.rs mode=write
use crate::{Graph, NodeId, Result};
pub fn conv2d(
	g: &mut Graph,
	x: NodeId,
	w: NodeId,
	pad: (usize, usize),
	stride: usize,
) -> Result<NodeId> {
	if stride == 0 {
		return Err("conv stride=0".into());
	}
	let sx = g.shape(x).to_vec();
	let sw = g.shape(w).to_vec();
	if sx.len() != 4 || sw.len() != 4 || sx[1] != sw[1] || sw[2] == 0 || sw[3] == 0 {
		return Err("conv NCHW/OIHW shape".into());
	}
	let xp = g.pad(x, vec![(0, 0), (0, 0), (pad.0, pad.0), (pad.1, pad.1)])?;
	let sp = g.shape(xp).to_vec();
	let oh = sp[2].checked_sub(sw[2]).ok_or("conv kernel height")? / stride + 1;
	let ow = sp[3].checked_sub(sw[3]).ok_or("conv kernel width")? / stride + 1;
	let mut windows = vec![];
	for kh in 0..sw[2] {
		for kw in 0..sw[3] {
			let mut hs = vec![];
			for h in 0..oh {
				let index = h
					.checked_mul(stride)
					.and_then(|v| v.checked_add(kh))
					.ok_or("conv index overflow")?;
				if index > 1 << 24 {
					return Err("f32 gather index precision".into());
				}
				hs.push(g.constant(index as f32, vec![])?);
			}
			let mut ws = vec![];
			for col in 0..ow {
				let index = col
					.checked_mul(stride)
					.and_then(|v| v.checked_add(kw))
					.ok_or("conv index overflow")?;
				if index > 1 << 24 {
					return Err("f32 gather index precision".into());
				}
				ws.push(g.constant(index as f32, vec![])?);
			}
			let hi = g.stack(hs, 0)?;
			let wi = g.stack(ws, 0)?;
			let rows = g.gather(xp, hi, 2)?;
			windows.push(g.gather(rows, wi, 3)?);
		}
	}
	let patch = g.stack(windows, 4)?;
	let patch = g.permute(patch, vec![0, 2, 3, 1, 4])?;
	let reduction = sx[1]
		.checked_mul(sw[2])
		.and_then(|n| n.checked_mul(sw[3]))
		.ok_or("conv reduction shape")?;
	let positions = sx[0]
		.checked_mul(oh)
		.and_then(|n| n.checked_mul(ow))
		.ok_or("conv output shape")?;
	let matrix = g.reshape(patch, vec![positions, reduction])?;
	let matrix = g.stage(matrix)?;
	let weights = g.reshape(w, vec![sw[0], reduction])?;
	let weights = g.permute(weights, vec![1, 0])?;
	let weights = g.stage(weights)?;
	let product = g.matmul(matrix, weights)?;
	let output = g.reshape(product, vec![sx[0], oh, ow, sw[0]])?;
	g.permute(output, vec![0, 3, 1, 2])
}
```

```rust
// file=src/storage.rs mode=write
use crate::{
	Buffer, Graph, Op, Result,
	graph::kernel_bodies,
	runtime::{NativeLibrary, elements},
};
use std::{
	collections::{HashMap, HashSet},
	ffi::c_void,
	process::Command,
	rc::Rc,
	time::{Duration, Instant},
};
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CacheKey {
	pub backend: String,
	pub device: String,
	pub capability: String,
	pub source: String,
	pub compiler: String,
	pub flags: String,
	pub abi: u32,
	pub specialization: String,
}
impl CacheKey {
	pub fn cpu(source: String, specialization: String) -> Result<Self> {
		let compiler = Command::new("clang")
			.arg("--version")
			.output()
			.map_err(|e| e.to_string())?;
		if !compiler.status.success() {
			return Err("clang --version failed".into());
		}
		Ok(Self {
			backend: "cpu".into(),
			device: std::env::consts::ARCH.into(),
			capability: "f32,unaligned-vector4,pthread".into(),
			source,
			compiler: String::from_utf8_lossy(&compiler.stdout).into(),
			flags: "-std=c11 -O2 -fno-fast-math".into(),
			abi: 1,
			specialization,
		})
	}
}
#[derive(Default)]
pub struct CompileCache {
	entries: HashMap<CacheKey, Rc<NativeLibrary>>,
}
impl CompileCache {
	/// # Safety
	/// `key.source` must be trusted native C. Loading it can execute constructors.
	/// The caller must ensure its entry and every constructor are memory-safe,
	/// and must use the returned entry only with its compiled ABI and bounds.
	pub unsafe fn get(&mut self, key: CacheKey) -> Result<(Rc<NativeLibrary>, bool, Duration)> {
		let start = Instant::now();
		if let Some(entry) = self.entries.get(&key) {
			return Ok((entry.clone(), true, start.elapsed()));
		}
		if key.backend != "cpu" {
			return Err("CPU cache cannot load GPU binaries".into());
		}
		let lib = Rc::new(unsafe { NativeLibrary::compile(&key.source, "chibi_entry") }?);
		self.entries.insert(key, lib.clone());
		Ok((lib, false, start.elapsed()))
	}
}
#[derive(Clone, Debug)]
pub struct Allocation {
	pub node: usize,
	pub slot: usize,
	pub first: usize,
	pub last: usize,
	pub elements: usize,
	pub protected: bool,
}
#[derive(Debug)]
pub struct StoragePlan {
	pub allocations: Vec<Allocation>,
	pub slots: usize,
	pub slot_sizes: Vec<usize>,
}
pub fn plan(g: &Graph, outputs: &[usize], external: &[usize], reuse: bool) -> Result<StoragePlan> {
	let kernels = kernel_bodies(g, outputs, true)?;
	let mut uses: HashMap<usize, (usize, usize)> = HashMap::new();
	for (time, kernel) in kernels.iter().enumerate() {
		uses.entry(kernel.output)
			.and_modify(|v| v.1 = time)
			.or_insert((time, time));
		for &id in &kernel.dependencies {
			if kernel.body.contains(&format!("b{id}[")) {
				uses.entry(id)
					.and_modify(|v| v.1 = time)
					.or_insert((0, time));
			}
		}
	}
	let mut protected: HashSet<usize> = outputs.iter().chain(external).copied().collect();
	for id in g.topology(outputs)? {
		if matches!(g.nodes[id].op, Op::Input { .. }) {
			protected.insert(id);
		}
	}
	let mut intervals = uses.into_iter().collect::<Vec<_>>();
	intervals.sort_by_key(|&(id, (first, _))| (first, id));
	let mut allocations = vec![];
	let mut slot_sizes = vec![];
	let mut availability: Vec<usize> = vec![];
	let mut protected_slots = HashSet::new();
	for (id, (first, last)) in intervals {
		if matches!(g.nodes[id].op, Op::Input { .. }) {
			continue;
		}
		let n = elements(g.shape(id))?;
		let protect = protected.contains(&id);
		let slot = if reuse && !protect {
			(0..slot_sizes.len()).find(|&s| {
				!protected_slots.contains(&s) && availability[s] < first && slot_sizes[s] == n
			})
		} else {
			None
		};
		let slot = slot.unwrap_or_else(|| {
			let slot = slot_sizes.len();
			slot_sizes.push(n);
			availability.push(last);
			slot
		});
		availability[slot] = last;
		if protect {
			protected_slots.insert(slot);
		}
		allocations.push(Allocation {
			node: id,
			slot,
			first,
			last,
			elements: n,
			protected: protect,
		});
	}
	Ok(StoragePlan {
		slots: slot_sizes.len(),
		slot_sizes,
		allocations,
	})
}
#[derive(Debug, Default)]
pub struct Measurements {
	pub compile: Duration,
	pub allocation: Duration,
	pub transfer: Duration,
	pub native_and_sync: Duration,
	pub total: Duration,
}
pub struct MemoryPlan {
	graph: Graph,
	outputs: Vec<usize>,
	storage: StoragePlan,
	pub cache_hit: bool,
	compile: Duration,
	library: Rc<NativeLibrary>,
}
impl MemoryPlan {
	pub fn storage(&self) -> &StoragePlan {
		&self.storage
	}
	pub fn compile(
		g: &Graph,
		outputs: &[usize],
		external: &[usize],
		reuse: bool,
		cache: &mut CompileCache,
	) -> Result<Self> {
		let (source, _, _) = crate::graph::render(g, outputs, true)?;
		let key = CacheKey::cpu(source, "dynamic-shape".into())?;
		let (library, cache_hit, compile) = unsafe { cache.get(key) }?;
		Ok(Self {
			graph: g.clone(),
			outputs: outputs.to_vec(),
			storage: plan(g, outputs, external, reuse)?,
			cache_hit,
			compile,
			library,
		})
	}
	pub fn run(&self, inputs: &[Buffer]) -> Result<(Vec<Buffer>, Measurements)> {
		let total = Instant::now();
		self.graph.validate_inputs(inputs, &self.outputs)?;
		self.graph.validate_indices(inputs, &self.outputs)?;
		let allocation = Instant::now();
		let mut blocks = self
			.storage
			.slot_sizes
			.iter()
			.map(|&n| vec![0f32; n])
			.collect::<Vec<_>>();
		let mut pointers = vec![std::ptr::null_mut::<c_void>(); self.graph.nodes.len()];
		for a in &self.storage.allocations {
			pointers[a.node] = blocks[a.slot].as_mut_ptr().cast();
		}
		for id in self.graph.topology(&self.outputs)? {
			if let Op::Input { slot } = self.graph.nodes[id].op {
				pointers[id] = inputs[slot].data.as_ptr().cast_mut().cast();
			}
		}
		let mut measurements = Measurements {
			compile: self.compile,
			allocation: allocation.elapsed(),
			..Measurements::default()
		};
		let dims = self
			.graph
			.nodes
			.iter()
			.flat_map(|n| n.shape.iter().map(|&d| d as u64))
			.collect::<Vec<_>>();
		type Entry = unsafe extern "C" fn(*const *mut c_void, *const u64) -> i32;
		let entry: Entry = unsafe { std::mem::transmute(self.library.symbol()) };
		let kernel = Instant::now();
		let status = unsafe { entry(pointers.as_ptr(), dims.as_ptr()) };
		measurements.native_and_sync = kernel.elapsed();
		if status != 0 {
			return Err(format!("cached native kernel status {status}"));
		}
		let transfer = Instant::now();
		let result = self
			.outputs
			.iter()
			.map(|&id| {
				let data = if let Op::Input { slot } = self.graph.nodes[id].op {
					inputs[slot].data.clone()
				} else {
					let a = self
						.storage
						.allocations
						.iter()
						.find(|a| a.node == id)
						.ok_or("missing output allocation")?;
					blocks[a.slot].clone()
				};
				Buffer::new(self.graph.shape(id).to_vec(), data)
			})
			.collect::<Result<Vec<_>>>()?;
		measurements.transfer = transfer.elapsed();
		measurements.total = total.elapsed();
		Ok((result, measurements))
	}
}
```

```rust
// file=src/gpu_kernels.rs mode=append
#[derive(Clone, Copy, Debug)]
pub struct GemmTile {
	pub m: usize,
	pub n: usize,
	pub k: usize,
	pub pad: usize,
	pub transpose_b: bool,
}
impl Default for GemmTile {
	fn default() -> Self {
		Self {
			m: 8,
			n: 8,
			k: 8,
			pad: 1,
			transpose_b: true,
		}
	}
}
pub fn gemm_scalar(
	d: Dialect,
	m: usize,
	n: usize,
	k: usize,
	t: GemmTile,
) -> Result<Kernel, String> {
	if t.m == 0 || t.n == 0 || t.k == 0 {
		return Err("zero tile".into());
	}
	let local = t.m.checked_mul(t.n).ok_or("local overflow")?;
	let ap = t.k.checked_add(t.pad).ok_or("pitch overflow")?;
	let bp = if t.transpose_b { t.k } else { t.n }
		.checked_add(t.pad)
		.ok_or("pitch overflow")?;
	let al = t.m.checked_mul(ap).ok_or("local overflow")?;
	let bl = if t.transpose_b { t.n } else { t.k }
		.checked_mul(bp)
		.ok_or("local overflow")?;
	let bytes = al
		.checked_add(bl)
		.and_then(|n| n.checked_mul(4))
		.ok_or("local bytes overflow")?;
	let gn = groups(n, t.n)?;
	let gm = groups(m, t.m)?;
	let bload = if t.transpose_b {
		format!("bb[(z%{})*{bp}+z/{}]", t.n, t.n)
	} else {
		format!("bb[(z/{})*{bp}+z%{}]", t.n, t.n)
	};
	let bidx = if t.transpose_b {
		format!("bb[cn*{bp}+r]")
	} else {
		format!("bb[r*{bp}+cn]")
	};
	let body = format!(
		"{} float aa[{al}],bb[{bl}];size_t br=g/{gn},bc=g%{gn},rm=l/{tn},cn=l%{tn};size_t row=br*{tm}+rm,col=bc*{tn}+cn;float acc=0.0f;for(size_t ko=0;ko<{k};ko+={tk}){{for(size_t z=l;z<{tm}*{tk};z+={local}){{size_t ar=br*{tm}+z/{tk},ak=ko+z%{tk};aa[(z/{tk})*{ap}+z%{tk}]=(ar<{m}&&ak<{k})?b0[ar*{k}+ak]:0.0f;}}for(size_t z=l;z<{tk}*{tn};z+={local}){{size_t bk=ko+z/{tn},bn=bc*{tn}+z%{tn};{bload}=(bk<{k}&&bn<{n})?b1[bk*{n}+bn]:0.0f;}}{}for(size_t r=0;r<{tk};r++){{acc+=aa[rm*{ap}+r]*{bidx};}}{}}}if(row<{m}&&col<{n}){{b2[row*{n}+col]=acc;}}",
		shared(d),
		barrier(d),
		barrier(d),
		tn = t.n,
		tm = t.m,
		tk = t.k
	);
	spec(
		d,
		vec![
			m.checked_mul(k).ok_or("shape overflow")?,
			k.checked_mul(n).ok_or("shape overflow")?,
			m.checked_mul(n).ok_or("shape overflow")?,
		],
		body,
		gm.checked_mul(gn).ok_or("group overflow")?,
		local,
		bytes,
	)
}

pub fn gemm(d: Dialect, m: usize, n: usize, k: usize, t: GemmTile) -> Result<Kernel, String> {
	let mut kernel = gemm_scalar(d, m, n, k, t)?;
	if !k.is_multiple_of(4)
		|| !n.is_multiple_of(4)
		|| !t.k.is_multiple_of(4)
		|| !t.n.is_multiple_of(4)
	{
		return Ok(kernel);
	}
	let local = t.m * t.n;
	let ap = t.k + t.pad;
	let bp = if t.transpose_b { t.k } else { t.n } + t.pad;
	let aload = format!(
		"for(size_t z=l;z<{tm}*{tk};z+={local}){{size_t ar=br*{tm}+z/{tk},ak=ko+z%{tk};aa[(z/{tk})*{ap}+z%{tk}]=(ar<{m}&&ak<{k})?b0[ar*{k}+ak]:0.0f;}}",
		tm = t.m,
		tk = t.k
	);
	let bindex = if t.transpose_b {
		format!("bb[(z%{})*{bp}+z/{}]", t.n, t.n)
	} else {
		format!("bb[(z/{})*{bp}+z%{}]", t.n, t.n)
	};
	let bload = format!(
		"for(size_t z=l;z<{tk}*{tn};z+={local}){{size_t bk=ko+z/{tn},bn=bc*{tn}+z%{tn};{bindex}=(bk<{k}&&bn<{n})?b1[bk*{n}+bn]:0.0f;}}",
		tk = t.k,
		tn = t.n
	);
	let load = |pointer: &str| match d {
		Dialect::Metal => format!("*((device const float4*)({pointer}))"),
		Dialect::OpenCl => format!("vload4(0,{pointer})"),
	};
	let aa = format!(
		"for(size_t z=l*4;z<{tm}*{tk};z+={local}*4){{size_t ar=br*{tm}+z/{tk},ak=ko+z%{tk};float4 values=(float4)(0.0f);if(ar<{m}&&ak+3<{k}){{values={load};}}for(size_t c=0;c<4;c++){{aa[((z+c)/{tk})*{ap}+(z+c)%{tk}]=values[c];}}}}",
		tm = t.m,
		tk = t.k,
		load = load(&format!("b0+ar*{k}+ak"))
	);
	let store = if t.transpose_b {
		format!("bb[((z+c)%{})*{bp}+(z+c)/{}]", t.n, t.n)
	} else {
		format!("bb[((z+c)/{})*{bp}+(z+c)%{}]", t.n, t.n)
	};
	let bb = format!(
		"for(size_t z=l*4;z<{tk}*{tn};z+={local}*4){{size_t bk=ko+z/{tn},bn=bc*{tn}+z%{tn};float4 values=(float4)(0.0f);if(bk<{k}&&bn+3<{n}){{values={load};}}for(size_t c=0;c<4;c++){{{store}=values[c];}}}}",
		tk = t.k,
		tn = t.n,
		load = load(&format!("b1+bk*{n}+bn"))
	);
	kernel.source = kernel.source.replace(&aload, &aa).replace(&bload, &bb);
	Ok(kernel)
}
```

```rust
// file=src/gpu_graph.rs mode=append
impl GpuPlan {
	pub fn compile_optimized(
		graph: &Graph,
		outputs: &[usize],
		backend: Backend,
		local: usize,
		parallel_reduction: bool,
		tiled_gemm: bool,
	) -> Result<Self> {
		let mut lowered = lower(graph, outputs, backend.dialect(), local)?;
		for (kernel, slots) in &mut lowered {
			let out = *slots.last().ok_or("GPU output slot")?;
			if tiled_gemm && let Some(mat) = crate::gemm_opt::recognize(graph, out) {
				*kernel = crate::gpu_kernels::gemm(
					backend.dialect(),
					mat.m,
					mat.n,
					mat.k,
					crate::gpu_kernels::GemmTile::default(),
				)?;
				*slots = vec![mat.a, mat.b, out];
				continue;
			}
			if parallel_reduction
				&& let Op::Reduce {
					op: crate::Reduce::Sum,
					x,
					ref axes,
					keepdim: _,
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

```rust
// file=tests/gemm_opt.rs mode=write
use chibigrad::{
	Buffer, Graph,
	gemm_opt::{compile, flops, recognize},
};
#[test]
fn tiled_graph_and_rebind() {
	for (m, n, k) in [(1, 1, 1), (3, 5, 9), (9, 7, 17), (8, 8, 24)] {
		let mut g = Graph::default();
		let a = g.input(0, vec![m, k]).unwrap();
		let b = g.input(1, vec![k, n]).unwrap();
		let y = g.matmul(a, b).unwrap();
		assert!(recognize(&g, y).is_some());
		let (p, report) = compile(&g, &[y], (4, 4, 8), 3).unwrap();
		let data = [
			Buffer::new(
				vec![m, k],
				(0..m * k)
					.map(|i| ((i * 17 % 31) as f32 - 15.) / 16.)
					.collect(),
			)
			.unwrap(),
			Buffer::new(
				vec![k, n],
				(0..k * n)
					.map(|i| ((i * 11 % 29) as f32 - 14.) / 16.)
					.collect(),
			)
			.unwrap(),
		];
		let reference = g.interpret(&[y], &data).unwrap();
		assert_eq!(p.run(&data).unwrap(), reference, "{report}");
		let ones = [
			Buffer::new(vec![m, k], vec![1.; m * k]).unwrap(),
			Buffer::new(vec![k, n], vec![1.; k * n]).unwrap(),
		];
		assert_eq!(p.run(&ones).unwrap()[0].data, vec![k as f32; m * n]);
	}
}
#[test]
fn work_estimate() {
	assert_eq!(
		flops(1, 2, 5, (1, 2, 3)).unwrap(),
		chibigrad::gemm_opt::Flops {
			useful: 20,
			padded: 24
		}
	);
}
```

```rust
// file=tests/conv_opt.rs mode=write
use chibigrad::{Buffer, Graph, conv_opt::conv2d, gemm_opt};
#[test]
fn conv_stride_padding_uses_real_gemm() {
	let mut g = Graph::default();
	let x = g.input(0, vec![1, 1, 3, 3]).unwrap();
	let w = g.input(1, vec![1, 1, 2, 2]).unwrap();
	let y = conv2d(&mut g, x, w, (1, 1), 2).unwrap();
	let (p, report) = gemm_opt::compile(&g, &[y], (2, 2, 3), 2).unwrap();
	assert!(report.contains("matches=1"));
	let input = [
		Buffer::new(vec![1, 1, 3, 3], (1..=9).map(|i| i as f32).collect()).unwrap(),
		Buffer::new(vec![1, 1, 2, 2], vec![1.; 4]).unwrap(),
	];
	assert_eq!(p.run(&input).unwrap(), g.interpret(&[y], &input).unwrap());
	assert_eq!(p.run(&input).unwrap()[0].data, vec![1., 5., 11., 28.]);
}
```

```rust
// file=tests/storage.rs mode=write
use chibigrad::{
	Binary, Buffer, Graph,
	storage::{CompileCache, MemoryPlan},
};
#[test]
fn actual_reuse_and_binary_cache() {
	let mut g = Graph::default();
	let mut x = g.input(0, vec![17]).unwrap();
	let two = g.constant(2., vec![]).unwrap();
	for _ in 0..8 {
		let y = g.binary(Binary::Mul, x, two).unwrap();
		x = g.stage(y).unwrap();
	}
	let input = [Buffer::new(vec![17], vec![1.; 17]).unwrap()];
	let mut cache = CompileCache::default();
	let plain = MemoryPlan::compile(&g, &[x], &[], false, &mut cache).unwrap();
	let reuse = MemoryPlan::compile(&g, &[x], &[], true, &mut cache).unwrap();
	assert!(!plain.cache_hit);
	assert!(reuse.cache_hit);
	assert!(reuse.storage().slots < plain.storage().slots);
	assert_eq!(plain.run(&input).unwrap().0, reuse.run(&input).unwrap().0);
	assert_eq!(reuse.run(&input).unwrap().0[0].data, vec![256.; 17]);
	for a in &reuse.storage().allocations {
		for b in &reuse.storage().allocations {
			if a.node != b.node && a.slot == b.slot {
				assert!(a.last < b.first || b.last < a.first);
				assert!(!a.protected && !b.protected);
			}
		}
	}
}
```

```rust
// file=tests/gemm_source.rs mode=write
use chibigrad::{
	gpu::Dialect,
	gpu_kernels::{GemmTile, gemm},
};
#[test]
fn vector_loads_are_guarded() {
	for (m, n, k) in [(9, 16, 16), (8, 8, 24), (9, 7, 17)] {
		let kernel = gemm(Dialect::OpenCl, m, n, k, GemmTile::default()).unwrap();
		assert_eq!(kernel.source.contains("vload4("), n % 4 == 0 && k % 4 == 0);
		let path = std::env::temp_dir().join(format!(
			"gemm-v4-{}-{:?}-{m}-{n}-{k}.cl",
			std::process::id(),
			std::thread::current().id()
		));
		std::fs::write(&path, kernel.source).unwrap();
		let p = std::process::Command::new("clang")
			.args(["-x", "cl", "-cl-std=CL1.2", "-fsyntax-only"])
			.arg(&path)
			.output()
			.unwrap();
		let _ = std::fs::remove_file(path);
		assert!(p.status.success(), "{}", String::from_utf8_lossy(&p.stderr));
	}
}
```

```rust
// file=tests/gemm_hardware.rs mode=write
use chibigrad::{
	Buffer, Graph,
	gpu::{Dialect, Kernel},
	gpu_graph::{Backend, GpuPlan},
	gpu_kernels::{GemmTile, gemm},
};
fn cases(
	d: Dialect,
	backend: Backend,
	run: impl Fn(&Kernel, &[Vec<f32>]) -> Result<Vec<f32>, String>,
) {
	for (m, n, k) in [(1, 1, 1), (7, 9, 17), (8, 8, 24)] {
		let mut g = Graph::default();
		let a = g.input(0, vec![m, k]).unwrap();
		let b = g.input(1, vec![k, n]).unwrap();
		let y = g.matmul(a, b).unwrap();
		let inputs = [
			Buffer::new(
				vec![m, k],
				(0..m * k).map(|i| (i % 9) as f32 / 8. - 0.5).collect(),
			)
			.unwrap(),
			Buffer::new(
				vec![k, n],
				(0..k * n).map(|i| (i % 7) as f32 / 8. - 0.5).collect(),
			)
			.unwrap(),
		];
		let reference = g.interpret(&[y], &inputs).unwrap();
		let shader = gemm(d, m, n, k, GemmTile::default()).unwrap();
		let output = run(
			&shader,
			&[
				inputs[0].data.clone(),
				inputs[1].data.clone(),
				vec![0.; m * n],
			],
		)
		.unwrap();
		let plan = GpuPlan::compile_optimized(&g, &[y], backend, 64, false, true).unwrap();
		for actual in [output, plan.run(&inputs).unwrap()[0].data.clone()] {
			for (&a, &b) in actual.iter().zip(&reference[0].data) {
				assert!(a.is_finite() && (a - b).abs() <= 1e-4 + 1e-4 * b.abs());
			}
		}
	}
}
#[test]
#[ignore = "requires Metal"]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn metal() {
	cases(Dialect::Metal, Backend::Metal, |k, b| {
		Ok(unsafe { chibigrad::backends::metal::run(k, b) }?.output)
	});
}
#[test]
#[ignore = "requires OpenCL"]
#[cfg(feature = "opencl")]
fn opencl() {
	cases(Dialect::OpenCl, Backend::OpenCl, |k, b| {
		Ok(unsafe { chibigrad::backends::opencl::run(k, b) }?.output)
	});
}
```

```sh
# check requires=metal
cargo test --test gemm_hardware --features metal metal -- --ignored --exact
```

```sh
# check requires=opencl
cargo test --test gemm_hardware --features opencl opencl -- --ignored --exact
```

```rust
// file=tests/book_19.rs mode=write
mod suite0 {
	use chibigrad::{
		Buffer, Graph,
		gemm_opt::{compile, flops, recognize},
	};
	#[test]
	pub(super) fn tiled_graph_and_rebind() {
		for (m, n, k) in [(1, 1, 1), (3, 5, 9), (9, 7, 17), (8, 8, 24)] {
			let mut g = Graph::default();
			let a = g.input(0, vec![m, k]).unwrap();
			let b = g.input(1, vec![k, n]).unwrap();
			let y = g.matmul(a, b).unwrap();
			assert!(recognize(&g, y).is_some());
			let (p, report) = compile(&g, &[y], (4, 4, 8), 3).unwrap();
			let data = [
				Buffer::new(
					vec![m, k],
					(0..m * k)
						.map(|i| ((i * 17 % 31) as f32 - 15.) / 16.)
						.collect(),
				)
				.unwrap(),
				Buffer::new(
					vec![k, n],
					(0..k * n)
						.map(|i| ((i * 11 % 29) as f32 - 14.) / 16.)
						.collect(),
				)
				.unwrap(),
			];
			let reference = g.interpret(&[y], &data).unwrap();
			assert_eq!(p.run(&data).unwrap(), reference, "{report}");
			let ones = [
				Buffer::new(vec![m, k], vec![1.; m * k]).unwrap(),
				Buffer::new(vec![k, n], vec![1.; k * n]).unwrap(),
			];
			assert_eq!(p.run(&ones).unwrap()[0].data, vec![k as f32; m * n]);
		}
	}
	#[test]
	pub(super) fn work_estimate() {
		assert_eq!(
			flops(1, 2, 5, (1, 2, 3)).unwrap(),
			chibigrad::gemm_opt::Flops {
				useful: 20,
				padded: 24
			}
		);
	}
}

mod suite1 {
	use chibigrad::{Buffer, Graph, conv_opt::conv2d, gemm_opt};
	#[test]
	pub(super) fn conv_stride_padding_uses_real_gemm() {
		let mut g = Graph::default();
		let x = g.input(0, vec![1, 1, 3, 3]).unwrap();
		let w = g.input(1, vec![1, 1, 2, 2]).unwrap();
		let y = conv2d(&mut g, x, w, (1, 1), 2).unwrap();
		let (p, report) = gemm_opt::compile(&g, &[y], (2, 2, 3), 2).unwrap();
		assert!(report.contains("matches=1"));
		let input = [
			Buffer::new(vec![1, 1, 3, 3], (1..=9).map(|i| i as f32).collect()).unwrap(),
			Buffer::new(vec![1, 1, 2, 2], vec![1.; 4]).unwrap(),
		];
		assert_eq!(p.run(&input).unwrap(), g.interpret(&[y], &input).unwrap());
		assert_eq!(p.run(&input).unwrap()[0].data, vec![1., 5., 11., 28.]);
	}
}

mod suite2 {
	use chibigrad::{
		Binary, Buffer, Graph,
		storage::{CompileCache, MemoryPlan},
	};
	#[test]
	pub(super) fn actual_reuse_and_binary_cache() {
		let mut g = Graph::default();
		let mut x = g.input(0, vec![17]).unwrap();
		let two = g.constant(2., vec![]).unwrap();
		for _ in 0..8 {
			let y = g.binary(Binary::Mul, x, two).unwrap();
			x = g.stage(y).unwrap();
		}
		let input = [Buffer::new(vec![17], vec![1.; 17]).unwrap()];
		let mut cache = CompileCache::default();
		let plain = MemoryPlan::compile(&g, &[x], &[], false, &mut cache).unwrap();
		let reuse = MemoryPlan::compile(&g, &[x], &[], true, &mut cache).unwrap();
		assert!(!plain.cache_hit);
		assert!(reuse.cache_hit);
		assert!(reuse.storage().slots < plain.storage().slots);
		assert_eq!(plain.run(&input).unwrap().0, reuse.run(&input).unwrap().0);
		assert_eq!(reuse.run(&input).unwrap().0[0].data, vec![256.; 17]);
		for a in &reuse.storage().allocations {
			for b in &reuse.storage().allocations {
				if a.node != b.node && a.slot == b.slot {
					assert!(a.last < b.first || b.last < a.first);
					assert!(!a.protected && !b.protected);
				}
			}
		}
	}
}

mod suite3 {
	use chibigrad::{
		gpu::Dialect,
		gpu_kernels::{GemmTile, gemm},
	};
	#[test]
	pub(super) fn vector_loads_are_guarded() {
		for (m, n, k) in [(9, 16, 16), (8, 8, 24), (9, 7, 17)] {
			let kernel = gemm(Dialect::OpenCl, m, n, k, GemmTile::default()).unwrap();
			assert_eq!(kernel.source.contains("vload4("), n % 4 == 0 && k % 4 == 0);
			let path = std::env::temp_dir().join(format!(
				"gemm-v4-{}-{:?}-{m}-{n}-{k}.cl",
				std::process::id(),
				std::thread::current().id()
			));
			std::fs::write(&path, kernel.source).unwrap();
			let p = std::process::Command::new("clang")
				.args(["-x", "cl", "-cl-std=CL1.2", "-fsyntax-only"])
				.arg(&path)
				.output()
				.unwrap();
			let _ = std::fs::remove_file(path);
			assert!(p.status.success(), "{}", String::from_utf8_lossy(&p.stderr));
		}
	}
}
```

## `i,j,k` の逐次 GEMM を比較用の設定で呼び出せるようにする。

<!-- todo:19-001 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_001() {
	suite0::tiled_graph_and_rebind();
}
```

`src/runtime.rs` の `pub fn compile_unfused(`の定義を次の全体で置き換えます。

```rust
// file="src/runtime.rs" mode="replace" item="pub fn compile_unfused("
	pub fn compile_unfused(graph: &crate::Graph, outputs: &[usize]) -> Result<Self> {
		let (source, abi, kernel_count) = crate::graph::render(graph, outputs, false)?;
		let library = unsafe { NativeLibrary::compile(&source, "chibi_entry") }?;
		Ok(Self {
			sealed: (graph.nodes.clone(), outputs.to_vec(), abi.clone()),
			source,
			abi,
			graph: graph.clone(),
			outputs: outputs.to_vec(),
			kernel_count,
			library,
			specialized: false,
			allocations: crate::graph::kernel_bodies(graph, outputs, false)?
				.iter()
				.map(|k| k.output)
				.collect(),
		})
	}
```

`src/gemm_opt.rs` の `pub fn compile(`の定義を次の全体で置き換えます。

```rust
// file="src/gemm_opt.rs" mode="replace" item="pub fn compile("
pub fn compile(
	g: &Graph,
	outputs: &[usize],
	tile: (usize, usize, usize),
	workers: usize,
) -> Result<(Plan, String)> {
	let (tm, tn, tk) = tile;
	if tm == 0 || tn == 0 || tk == 0 {
		return Err("GEMM zero tile".into());
	}
	tm.checked_mul(tn)
		.and_then(|n| n.checked_mul(4))
		.filter(|&n| n <= 65536)
		.ok_or("register tile too large")?;
	let mut kernels = kernel_bodies(g, outputs, true)?;
	let mut matches = 0;
	let mut offsets = vec![];
	let mut offset = 0;
	for n in &g.nodes {
		offsets.push(offset);
		offset += n.shape.len();
	}
	for kernel in &mut kernels {
		if let Some(mat) = recognize(g, kernel.output) {
			let a = mat.a;
			let b = mat.b;
			let out = mat.output;
			let m = format!("s[{}]", offsets[a]);
			let k = format!("s[{}]", offsets[a] + 1);
			let n = format!("s[{}]", offsets[b] + 1);
			let gm = format!("({m}/{tm}+({m}%{tm}!=0))");
			let gn = format!("({n}/{tn}+({n}%{tn}!=0))");
			kernel.body = format!(
				"for (size_t i=0; i<{gm}*{gn}; ++i) {{size_t br=i/{gn},bc=i%{gn};float acc[{tm}][{tn}]={{0}};for(size_t ko=0;ko<{k};){{size_t klimit={k}-ko<{tk}?{k}-ko:{tk};for(size_t ki=0;ki<klimit;ki++){{size_t r=ko+ki;for(size_t mi=0;mi<{tm};mi++){{size_t row=br*{tm}+mi;if(row<{m}){{for(size_t ni=0;ni<{tn};ni++){{size_t col=bc*{tn}+ni;if(col<{n}){{acc[mi][ni]+=b{a}[row*{k}+r]*b{b}[r*{n}+col];}}}}}}}}}}ko+=klimit;}}for(size_t mi=0;mi<{tm};mi++){{size_t row=br*{tm}+mi;for(size_t ni=0;ni<{tn};ni++){{size_t col=bc*{tn}+ni;if(row<{m}&&col<{n}){{b{out}[row*{n}+col]=acc[mi][ni];}}}}}}}}\n"
			);
			matches += 1;
		}
	}
	let (plan, report) = compile_bodies(
		g,
		outputs,
		Options {
			workers,
			..Options::default()
		},
		kernels,
	)?;
	Ok((
		plan,
		format!("GEMM matches={matches}, tile={tile:?}; {report}"),
	))
}
```

```sh
# check
cargo test --test book_19 todo_19_001 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## `k=ko*TK+ki` とする変換を入れ、端数は `k<K` で守る。

<!-- todo:19-002 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_002() {
	suite0::tiled_graph_and_rebind();
}
```

`src/gemm_opt.rs` の `pub fn compile(`の定義を次の全体で置き換えます。

```rust
// file="src/gemm_opt.rs" mode="replace" item="pub fn compile("
pub fn compile(
	g: &Graph,
	outputs: &[usize],
	tile: (usize, usize, usize),
	workers: usize,
) -> Result<(Plan, String)> {
	let (tm, tn, tk) = tile;
	if tm == 0 || tn == 0 || tk == 0 {
		return Err("GEMM zero tile".into());
	}
	tm.checked_mul(tn)
		.and_then(|n| n.checked_mul(4))
		.filter(|&n| n <= 65536)
		.ok_or("register tile too large")?;
	let mut kernels = kernel_bodies(g, outputs, true)?;
	let mut matches = 0;
	let mut offsets = vec![];
	let mut offset = 0;
	for n in &g.nodes {
		offsets.push(offset);
		offset += n.shape.len();
	}
	for kernel in &mut kernels {
		if let Some(mat) = recognize(g, kernel.output) {
			let a = mat.a;
			let b = mat.b;
			let out = mat.output;
			let m = format!("s[{}]", offsets[a]);
			let k = format!("s[{}]", offsets[a] + 1);
			let n = format!("s[{}]", offsets[b] + 1);
			let gm = format!("({m}/{tm}+({m}%{tm}!=0))");
			let gn = format!("({n}/{tn}+({n}%{tn}!=0))");
			kernel.body = format!(
				"for (size_t i=0; i<{gm}*{gn}; ++i) {{size_t br=i/{gn},bc=i%{gn};float acc[{tm}][{tn}]={{0}};for(size_t ko=0;ko<{k};){{size_t klimit={k}-ko<{tk}?{k}-ko:{tk};for(size_t ki=0;ki<klimit;ki++){{size_t r=ko+ki;for(size_t mi=0;mi<{tm};mi++){{size_t row=br*{tm}+mi;if(row<{m}){{for(size_t ni=0;ni<{tn};ni++){{size_t col=bc*{tn}+ni;if(col<{n}){{acc[mi][ni]+=b{a}[row*{k}+r]*b{b}[r*{n}+col];}}}}}}}}}}ko+=klimit;}}for(size_t mi=0;mi<{tm};mi++){{size_t row=br*{tm}+mi;for(size_t ni=0;ni<{tn};ni++){{size_t col=bc*{tn}+ni;if(row<{m}&&col<{n}){{b{out}[row*{n}+col]=acc[mi][ni];}}}}}}}}\n"
			);
			matches += 1;
		}
	}
	let (plan, report) = compile_bodies(
		g,
		outputs,
		Options {
			workers,
			..Options::default()
		},
		kernels,
	)?;
	Ok((
		plan,
		format!("GEMM matches={matches}, tile={tile:?}; {report}"),
	))
}
```

```sh
# check
cargo test --test book_19 todo_19_002 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## `M,N` のタイル分割を追加し、最初は各タイル内の計算順を変えない。

<!-- todo:19-003 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_003() {
	suite0::tiled_graph_and_rebind();
}
```

`src/gemm_opt.rs` の `pub fn compile(`の定義を次の全体で置き換えます。

```rust
// file="src/gemm_opt.rs" mode="replace" item="pub fn compile("
pub fn compile(
	g: &Graph,
	outputs: &[usize],
	tile: (usize, usize, usize),
	workers: usize,
) -> Result<(Plan, String)> {
	let (tm, tn, tk) = tile;
	if tm == 0 || tn == 0 || tk == 0 {
		return Err("GEMM zero tile".into());
	}
	tm.checked_mul(tn)
		.and_then(|n| n.checked_mul(4))
		.filter(|&n| n <= 65536)
		.ok_or("register tile too large")?;
	let mut kernels = kernel_bodies(g, outputs, true)?;
	let mut matches = 0;
	let mut offsets = vec![];
	let mut offset = 0;
	for n in &g.nodes {
		offsets.push(offset);
		offset += n.shape.len();
	}
	for kernel in &mut kernels {
		if let Some(mat) = recognize(g, kernel.output) {
			let a = mat.a;
			let b = mat.b;
			let out = mat.output;
			let m = format!("s[{}]", offsets[a]);
			let k = format!("s[{}]", offsets[a] + 1);
			let n = format!("s[{}]", offsets[b] + 1);
			let gm = format!("({m}/{tm}+({m}%{tm}!=0))");
			let gn = format!("({n}/{tn}+({n}%{tn}!=0))");
			kernel.body = format!(
				"for (size_t i=0; i<{gm}*{gn}; ++i) {{size_t br=i/{gn},bc=i%{gn};float acc[{tm}][{tn}]={{0}};for(size_t ko=0;ko<{k};){{size_t klimit={k}-ko<{tk}?{k}-ko:{tk};for(size_t ki=0;ki<klimit;ki++){{size_t r=ko+ki;for(size_t mi=0;mi<{tm};mi++){{size_t row=br*{tm}+mi;if(row<{m}){{for(size_t ni=0;ni<{tn};ni++){{size_t col=bc*{tn}+ni;if(col<{n}){{acc[mi][ni]+=b{a}[row*{k}+r]*b{b}[r*{n}+col];}}}}}}}}}}ko+=klimit;}}for(size_t mi=0;mi<{tm};mi++){{size_t row=br*{tm}+mi;for(size_t ni=0;ni<{tn};ni++){{size_t col=bc*{tn}+ni;if(row<{m}&&col<{n}){{b{out}[row*{n}+col]=acc[mi][ni];}}}}}}}}\n"
			);
			matches += 1;
		}
	}
	let (plan, report) = compile_bodies(
		g,
		outputs,
		Options {
			workers,
			..Options::default()
		},
		kernels,
	)?;
	Ok((
		plan,
		format!("GEMM matches={matches}, tile={tile:?}; {report}"),
	))
}
```

```sh
# check
cargo test --test book_19 todo_19_003 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 小さな register 累積配列を C に出し、初期化を K ループの外に置く。

<!-- todo:19-004 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_004() {
	suite0::tiled_graph_and_rebind();
}
```

`src/gemm_opt.rs` の `pub fn compile(`の定義を次の全体で置き換えます。

```rust
// file="src/gemm_opt.rs" mode="replace" item="pub fn compile("
pub fn compile(
	g: &Graph,
	outputs: &[usize],
	tile: (usize, usize, usize),
	workers: usize,
) -> Result<(Plan, String)> {
	let (tm, tn, tk) = tile;
	if tm == 0 || tn == 0 || tk == 0 {
		return Err("GEMM zero tile".into());
	}
	tm.checked_mul(tn)
		.and_then(|n| n.checked_mul(4))
		.filter(|&n| n <= 65536)
		.ok_or("register tile too large")?;
	let mut kernels = kernel_bodies(g, outputs, true)?;
	let mut matches = 0;
	let mut offsets = vec![];
	let mut offset = 0;
	for n in &g.nodes {
		offsets.push(offset);
		offset += n.shape.len();
	}
	for kernel in &mut kernels {
		if let Some(mat) = recognize(g, kernel.output) {
			let a = mat.a;
			let b = mat.b;
			let out = mat.output;
			let m = format!("s[{}]", offsets[a]);
			let k = format!("s[{}]", offsets[a] + 1);
			let n = format!("s[{}]", offsets[b] + 1);
			let gm = format!("({m}/{tm}+({m}%{tm}!=0))");
			let gn = format!("({n}/{tn}+({n}%{tn}!=0))");
			kernel.body = format!(
				"for (size_t i=0; i<{gm}*{gn}; ++i) {{size_t br=i/{gn},bc=i%{gn};float acc[{tm}][{tn}]={{0}};for(size_t ko=0;ko<{k};){{size_t klimit={k}-ko<{tk}?{k}-ko:{tk};for(size_t ki=0;ki<klimit;ki++){{size_t r=ko+ki;for(size_t mi=0;mi<{tm};mi++){{size_t row=br*{tm}+mi;if(row<{m}){{for(size_t ni=0;ni<{tn};ni++){{size_t col=bc*{tn}+ni;if(col<{n}){{acc[mi][ni]+=b{a}[row*{k}+r]*b{b}[r*{n}+col];}}}}}}}}}}ko+=klimit;}}for(size_t mi=0;mi<{tm};mi++){{size_t row=br*{tm}+mi;for(size_t ni=0;ni<{tn};ni++){{size_t col=bc*{tn}+ni;if(row<{m}&&col<{n}){{b{out}[row*{n}+col]=acc[mi][ni];}}}}}}}}\n"
			);
			matches += 1;
		}
	}
	let (plan, report) = compile_bodies(
		g,
		outputs,
		Options {
			workers,
			..Options::default()
		},
		kernels,
	)?;
	Ok((
		plan,
		format!("GEMM matches={matches}, tile={tile:?}; {report}"),
	))
}
```

```sh
# check
cargo test --test book_19 todo_19_004 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 出力タイルの並列化は17の CPU thread 経路を使い、同じ出力要素を複数 thread が書かないようにする。

<!-- todo:19-005 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_005() {
	suite0::tiled_graph_and_rebind();
}
```

`src/gemm_opt.rs` の `pub fn compile(`の定義を次の全体で置き換えます。

```rust
// file="src/gemm_opt.rs" mode="replace" item="pub fn compile("
pub fn compile(
	g: &Graph,
	outputs: &[usize],
	tile: (usize, usize, usize),
	workers: usize,
) -> Result<(Plan, String)> {
	let (tm, tn, tk) = tile;
	if tm == 0 || tn == 0 || tk == 0 {
		return Err("GEMM zero tile".into());
	}
	tm.checked_mul(tn)
		.and_then(|n| n.checked_mul(4))
		.filter(|&n| n <= 65536)
		.ok_or("register tile too large")?;
	let mut kernels = kernel_bodies(g, outputs, true)?;
	let mut matches = 0;
	let mut offsets = vec![];
	let mut offset = 0;
	for n in &g.nodes {
		offsets.push(offset);
		offset += n.shape.len();
	}
	for kernel in &mut kernels {
		if let Some(mat) = recognize(g, kernel.output) {
			let a = mat.a;
			let b = mat.b;
			let out = mat.output;
			let m = format!("s[{}]", offsets[a]);
			let k = format!("s[{}]", offsets[a] + 1);
			let n = format!("s[{}]", offsets[b] + 1);
			let gm = format!("({m}/{tm}+({m}%{tm}!=0))");
			let gn = format!("({n}/{tn}+({n}%{tn}!=0))");
			kernel.body = format!(
				"for (size_t i=0; i<{gm}*{gn}; ++i) {{size_t br=i/{gn},bc=i%{gn};float acc[{tm}][{tn}]={{0}};for(size_t ko=0;ko<{k};){{size_t klimit={k}-ko<{tk}?{k}-ko:{tk};for(size_t ki=0;ki<klimit;ki++){{size_t r=ko+ki;for(size_t mi=0;mi<{tm};mi++){{size_t row=br*{tm}+mi;if(row<{m}){{for(size_t ni=0;ni<{tn};ni++){{size_t col=bc*{tn}+ni;if(col<{n}){{acc[mi][ni]+=b{a}[row*{k}+r]*b{b}[r*{n}+col];}}}}}}}}}}ko+=klimit;}}for(size_t mi=0;mi<{tm};mi++){{size_t row=br*{tm}+mi;for(size_t ni=0;ni<{tn};ni++){{size_t col=bc*{tn}+ni;if(row<{m}&&col<{n}){{b{out}[row*{n}+col]=acc[mi][ni];}}}}}}}}\n"
			);
			matches += 1;
		}
	}
	let (plan, report) = compile_bodies(
		g,
		outputs,
		Options {
			workers,
			..Options::default()
		},
		kernels,
	)?;
	Ok((
		plan,
		format!("GEMM matches={matches}, tile={tile:?}; {report}"),
	))
}
```

`src/loop_opt.rs` の `pub fn compile_bodies(`の定義を次の全体で置き換えます。

```rust
// file="src/loop_opt.rs" mode="replace" item="pub fn compile_bodies("
pub fn compile_bodies(
	g: &Graph,
	outputs: &[usize],
	o: Options,
	kernels: Vec<crate::graph::KernelBody>,
) -> Result<(Plan, String)> {
	o.validate()?;
	let mut source = C_HELPERS.to_string();
	source.push_str("#include <pthread.h>\ntypedef float chibi_v4 __attribute__((ext_vector_type(4)));\ntypedef struct {void*const* buffers;const uint64_t*s;size_t begin,end;int status;} Worker;\n");
	let mut calls = String::new();
	let mut report = String::new();
	for (serial, k) in kernels.iter().enumerate() {
		let (count, body) = outer(&k.body)?;
		let mut pointers = String::new();
		for id in g.topology(&[k.output])? {
			pointers += &format!("float*b{id}=(float*)w->buffers[{id}];\n");
		}
		let mut work = String::new();
		if o.vector
			&& !o.interchange
			&& let Some((a, b)) = vector_add(g, k.output)
		{
			work += &format!(
				"size_t i=w->begin;for(;w->end-i>=4;i+=4){{chibi_v4 x,y,z;memcpy(&x,b{a}+i,16);memcpy(&y,b{b}+i,16);z=x+y;memcpy(b{}+i,&z,16);}}for(;i<w->end;i++){{{body}}}",
				k.output
			);
			report += "vector=4 (unaligned memcpy); ";
		}
		if work.is_empty() {
			if o.vector {
				report += "vector=scalar fallback (noncontiguous or unsupported operation); ";
			}
			let index = if o.interchange {
				let shape = g.shape(k.output);
				if shape.len() != 2 {
					return Err("interchange needs two independent output axes".into());
				}
				format!(
					"size_t i=(flat%{})*{}+flat/{};",
					shape[0], shape[1], shape[0]
				)
			} else {
				"size_t i=flat;".into()
			};
			work += &format!(
				"for(size_t outer=w->begin;outer<w->end;){{size_t limit=w->end-outer<{tile}?w->end-outer:{tile};for(size_t inner=0;inner<limit;inner+={unroll}){{",
				tile = o.tile,
				unroll = o.unroll
			);
			for lane in 0..o.unroll {
				work += &format!(
					"{{size_t off=inner+{lane};if(off<limit){{size_t flat=outer+off;{index}{body}}}}}"
				);
			}
			work += "}outer+=limit;}";
		}
		work = work.replace("return 2;", "{w->status=2;return NULL;}");
		source += &format!(
			"static void* kernel{serial}(void*arg){{Worker*w=(Worker*)arg;const uint64_t*s=w->s;{pointers}{work}return NULL;}}\n"
		);
		let workers = o.workers;
		calls += &format!(
			"{{size_t count={count};pthread_t threads[{workers}];Worker args[{workers}];size_t created=0;int failure=0;for(size_t w=0;w<{workers};w++){{size_t q=count/{workers},r=count%{workers};args[w]=(Worker){{buffers,s,q*w+(w<r?w:r),q*(w+1)+((w+1)<r?w+1:r),0}};if(pthread_create(&threads[w],NULL,kernel{serial},&args[w])){{failure=3;break;}}created++;}}for(size_t w=0;w<created;w++){{if(pthread_join(threads[w],NULL))failure=4;if(args[w].status)failure=args[w].status;}}if(failure)return failure;}}\n"
		);
	}
	source +=
		&format!("int chibi_entry(void*const*buffers,const uint64_t*s){{{calls}return 0;}}\n");
	report += &format!(
		"tile={} unroll={} workers={} output_interchange={}",
		o.tile, o.unroll, o.workers, o.interchange
	);
	Ok((
		Plan::compile_source_specialized(g, outputs, source)?,
		report,
	))
}
```

```sh
# check
cargo test --test book_19 todo_19_005 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## Metal で A の `TM*TK`、B の `TK*TN` の local buffer を確保する。

<!-- todo:19-006 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_006() {
	suite3::vector_loads_are_guarded();
}
```

`src/gpu_kernels.rs` の `pub fn gemm_scalar(`の定義を次の全体で置き換えます。

```rust
// file="src/gpu_kernels.rs" mode="replace" item="pub fn gemm_scalar("
pub fn gemm_scalar(
	d: Dialect,
	m: usize,
	n: usize,
	k: usize,
	t: GemmTile,
) -> Result<Kernel, String> {
	if t.m == 0 || t.n == 0 || t.k == 0 {
		return Err("zero tile".into());
	}
	let local = t.m.checked_mul(t.n).ok_or("local overflow")?;
	let ap = t.k.checked_add(t.pad).ok_or("pitch overflow")?;
	let bp = if t.transpose_b { t.k } else { t.n }
		.checked_add(t.pad)
		.ok_or("pitch overflow")?;
	let al = t.m.checked_mul(ap).ok_or("local overflow")?;
	let bl = if t.transpose_b { t.n } else { t.k }
		.checked_mul(bp)
		.ok_or("local overflow")?;
	let bytes = al
		.checked_add(bl)
		.and_then(|n| n.checked_mul(4))
		.ok_or("local bytes overflow")?;
	let gn = groups(n, t.n)?;
	let gm = groups(m, t.m)?;
	let bload = if t.transpose_b {
		format!("bb[(z%{})*{bp}+z/{}]", t.n, t.n)
	} else {
		format!("bb[(z/{})*{bp}+z%{}]", t.n, t.n)
	};
	let bidx = if t.transpose_b {
		format!("bb[cn*{bp}+r]")
	} else {
		format!("bb[r*{bp}+cn]")
	};
	let body = format!(
		"{} float aa[{al}],bb[{bl}];size_t br=g/{gn},bc=g%{gn},rm=l/{tn},cn=l%{tn};size_t row=br*{tm}+rm,col=bc*{tn}+cn;float acc=0.0f;for(size_t ko=0;ko<{k};ko+={tk}){{for(size_t z=l;z<{tm}*{tk};z+={local}){{size_t ar=br*{tm}+z/{tk},ak=ko+z%{tk};aa[(z/{tk})*{ap}+z%{tk}]=(ar<{m}&&ak<{k})?b0[ar*{k}+ak]:0.0f;}}for(size_t z=l;z<{tk}*{tn};z+={local}){{size_t bk=ko+z/{tn},bn=bc*{tn}+z%{tn};{bload}=(bk<{k}&&bn<{n})?b1[bk*{n}+bn]:0.0f;}}{}for(size_t r=0;r<{tk};r++){{acc+=aa[rm*{ap}+r]*{bidx};}}{}}}if(row<{m}&&col<{n}){{b2[row*{n}+col]=acc;}}",
		shared(d),
		barrier(d),
		barrier(d),
		tn = t.n,
		tm = t.m,
		tk = t.k
	);
	spec(
		d,
		vec![
			m.checked_mul(k).ok_or("shape overflow")?,
			k.checked_mul(n).ok_or("shape overflow")?,
			m.checked_mul(n).ok_or("shape overflow")?,
		],
		body,
		gm.checked_mul(gn).ok_or("group overflow")?,
		local,
		bytes,
	)
}
```

```sh
# check
cargo test --test book_19 todo_19_006 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## lane ごとの協調 LOAD を作り、範囲外は LOAD を実行せず local にゼロを保存する。

<!-- todo:19-007 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_007() {
	suite3::vector_loads_are_guarded();
}
```

`src/gpu_kernels.rs` の `pub fn gemm_scalar(`の定義を次の全体で置き換えます。

```rust
// file="src/gpu_kernels.rs" mode="replace" item="pub fn gemm_scalar("
pub fn gemm_scalar(
	d: Dialect,
	m: usize,
	n: usize,
	k: usize,
	t: GemmTile,
) -> Result<Kernel, String> {
	if t.m == 0 || t.n == 0 || t.k == 0 {
		return Err("zero tile".into());
	}
	let local = t.m.checked_mul(t.n).ok_or("local overflow")?;
	let ap = t.k.checked_add(t.pad).ok_or("pitch overflow")?;
	let bp = if t.transpose_b { t.k } else { t.n }
		.checked_add(t.pad)
		.ok_or("pitch overflow")?;
	let al = t.m.checked_mul(ap).ok_or("local overflow")?;
	let bl = if t.transpose_b { t.n } else { t.k }
		.checked_mul(bp)
		.ok_or("local overflow")?;
	let bytes = al
		.checked_add(bl)
		.and_then(|n| n.checked_mul(4))
		.ok_or("local bytes overflow")?;
	let gn = groups(n, t.n)?;
	let gm = groups(m, t.m)?;
	let bload = if t.transpose_b {
		format!("bb[(z%{})*{bp}+z/{}]", t.n, t.n)
	} else {
		format!("bb[(z/{})*{bp}+z%{}]", t.n, t.n)
	};
	let bidx = if t.transpose_b {
		format!("bb[cn*{bp}+r]")
	} else {
		format!("bb[r*{bp}+cn]")
	};
	let body = format!(
		"{} float aa[{al}],bb[{bl}];size_t br=g/{gn},bc=g%{gn},rm=l/{tn},cn=l%{tn};size_t row=br*{tm}+rm,col=bc*{tn}+cn;float acc=0.0f;for(size_t ko=0;ko<{k};ko+={tk}){{for(size_t z=l;z<{tm}*{tk};z+={local}){{size_t ar=br*{tm}+z/{tk},ak=ko+z%{tk};aa[(z/{tk})*{ap}+z%{tk}]=(ar<{m}&&ak<{k})?b0[ar*{k}+ak]:0.0f;}}for(size_t z=l;z<{tk}*{tn};z+={local}){{size_t bk=ko+z/{tn},bn=bc*{tn}+z%{tn};{bload}=(bk<{k}&&bn<{n})?b1[bk*{n}+bn]:0.0f;}}{}for(size_t r=0;r<{tk};r++){{acc+=aa[rm*{ap}+r]*{bidx};}}{}}}if(row<{m}&&col<{n}){{b2[row*{n}+col]=acc;}}",
		shared(d),
		barrier(d),
		barrier(d),
		tn = t.n,
		tm = t.m,
		tk = t.k
	);
	spec(
		d,
		vec![
			m.checked_mul(k).ok_or("shape overflow")?,
			k.checked_mul(n).ok_or("shape overflow")?,
			m.checked_mul(n).ok_or("shape overflow")?,
		],
		body,
		gm.checked_mul(gn).ok_or("group overflow")?,
		local,
		bytes,
	)
}
```

```sh
# check
cargo test --test book_19 todo_19_007 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 保存後の barrier と、計算後に local buffer を再利用する前の barrier を入れる。

<!-- todo:19-008 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_008() {
	suite3::vector_loads_are_guarded();
}
```

`src/gpu_kernels.rs` の `pub fn gemm_scalar(`の定義を次の全体で置き換えます。

```rust
// file="src/gpu_kernels.rs" mode="replace" item="pub fn gemm_scalar("
pub fn gemm_scalar(
	d: Dialect,
	m: usize,
	n: usize,
	k: usize,
	t: GemmTile,
) -> Result<Kernel, String> {
	if t.m == 0 || t.n == 0 || t.k == 0 {
		return Err("zero tile".into());
	}
	let local = t.m.checked_mul(t.n).ok_or("local overflow")?;
	let ap = t.k.checked_add(t.pad).ok_or("pitch overflow")?;
	let bp = if t.transpose_b { t.k } else { t.n }
		.checked_add(t.pad)
		.ok_or("pitch overflow")?;
	let al = t.m.checked_mul(ap).ok_or("local overflow")?;
	let bl = if t.transpose_b { t.n } else { t.k }
		.checked_mul(bp)
		.ok_or("local overflow")?;
	let bytes = al
		.checked_add(bl)
		.and_then(|n| n.checked_mul(4))
		.ok_or("local bytes overflow")?;
	let gn = groups(n, t.n)?;
	let gm = groups(m, t.m)?;
	let bload = if t.transpose_b {
		format!("bb[(z%{})*{bp}+z/{}]", t.n, t.n)
	} else {
		format!("bb[(z/{})*{bp}+z%{}]", t.n, t.n)
	};
	let bidx = if t.transpose_b {
		format!("bb[cn*{bp}+r]")
	} else {
		format!("bb[r*{bp}+cn]")
	};
	let body = format!(
		"{} float aa[{al}],bb[{bl}];size_t br=g/{gn},bc=g%{gn},rm=l/{tn},cn=l%{tn};size_t row=br*{tm}+rm,col=bc*{tn}+cn;float acc=0.0f;for(size_t ko=0;ko<{k};ko+={tk}){{for(size_t z=l;z<{tm}*{tk};z+={local}){{size_t ar=br*{tm}+z/{tk},ak=ko+z%{tk};aa[(z/{tk})*{ap}+z%{tk}]=(ar<{m}&&ak<{k})?b0[ar*{k}+ak]:0.0f;}}for(size_t z=l;z<{tk}*{tn};z+={local}){{size_t bk=ko+z/{tn},bn=bc*{tn}+z%{tn};{bload}=(bk<{k}&&bn<{n})?b1[bk*{n}+bn]:0.0f;}}{}for(size_t r=0;r<{tk};r++){{acc+=aa[rm*{ap}+r]*{bidx};}}{}}}if(row<{m}&&col<{n}){{b2[row*{n}+col]=acc;}}",
		shared(d),
		barrier(d),
		barrier(d),
		tn = t.n,
		tm = t.m,
		tk = t.k
	);
	spec(
		d,
		vec![
			m.checked_mul(k).ok_or("shape overflow")?,
			k.checked_mul(n).ok_or("shape overflow")?,
			m.checked_mul(n).ok_or("shape overflow")?,
		],
		body,
		gm.checked_mul(gn).ok_or("group overflow")?,
		local,
		bytes,
	)
}
```

```sh
# check
cargo test --test book_19 todo_19_008 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 無効な出力 lane も barrier を通し、最後の global STORE だけを guard する。

<!-- todo:19-009 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_009() {
	suite3::vector_loads_are_guarded();
}
```

`src/gpu_kernels.rs` の `pub fn gemm_scalar(`の定義を次の全体で置き換えます。

```rust
// file="src/gpu_kernels.rs" mode="replace" item="pub fn gemm_scalar("
pub fn gemm_scalar(
	d: Dialect,
	m: usize,
	n: usize,
	k: usize,
	t: GemmTile,
) -> Result<Kernel, String> {
	if t.m == 0 || t.n == 0 || t.k == 0 {
		return Err("zero tile".into());
	}
	let local = t.m.checked_mul(t.n).ok_or("local overflow")?;
	let ap = t.k.checked_add(t.pad).ok_or("pitch overflow")?;
	let bp = if t.transpose_b { t.k } else { t.n }
		.checked_add(t.pad)
		.ok_or("pitch overflow")?;
	let al = t.m.checked_mul(ap).ok_or("local overflow")?;
	let bl = if t.transpose_b { t.n } else { t.k }
		.checked_mul(bp)
		.ok_or("local overflow")?;
	let bytes = al
		.checked_add(bl)
		.and_then(|n| n.checked_mul(4))
		.ok_or("local bytes overflow")?;
	let gn = groups(n, t.n)?;
	let gm = groups(m, t.m)?;
	let bload = if t.transpose_b {
		format!("bb[(z%{})*{bp}+z/{}]", t.n, t.n)
	} else {
		format!("bb[(z/{})*{bp}+z%{}]", t.n, t.n)
	};
	let bidx = if t.transpose_b {
		format!("bb[cn*{bp}+r]")
	} else {
		format!("bb[r*{bp}+cn]")
	};
	let body = format!(
		"{} float aa[{al}],bb[{bl}];size_t br=g/{gn},bc=g%{gn},rm=l/{tn},cn=l%{tn};size_t row=br*{tm}+rm,col=bc*{tn}+cn;float acc=0.0f;for(size_t ko=0;ko<{k};ko+={tk}){{for(size_t z=l;z<{tm}*{tk};z+={local}){{size_t ar=br*{tm}+z/{tk},ak=ko+z%{tk};aa[(z/{tk})*{ap}+z%{tk}]=(ar<{m}&&ak<{k})?b0[ar*{k}+ak]:0.0f;}}for(size_t z=l;z<{tk}*{tn};z+={local}){{size_t bk=ko+z/{tn},bn=bc*{tn}+z%{tn};{bload}=(bk<{k}&&bn<{n})?b1[bk*{n}+bn]:0.0f;}}{}for(size_t r=0;r<{tk};r++){{acc+=aa[rm*{ap}+r]*{bidx};}}{}}}if(row<{m}&&col<{n}){{b2[row*{n}+col]=acc;}}",
		shared(d),
		barrier(d),
		barrier(d),
		tn = t.n,
		tm = t.m,
		tk = t.k
	);
	spec(
		d,
		vec![
			m.checked_mul(k).ok_or("shape overflow")?,
			k.checked_mul(n).ok_or("shape overflow")?,
			m.checked_mul(n).ok_or("shape overflow")?,
		],
		body,
		gm.checked_mul(gn).ok_or("group overflow")?,
		local,
		bytes,
	)
}
```

```sh
# check
cargo test --test book_19 todo_19_009 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 同じ IR を OpenCL に出し、`groups` と `local` の積を実際の global size とする。

<!-- todo:19-010 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_010() {
	suite3::vector_loads_are_guarded();
}
```

`src/gpu_kernels.rs` の `pub fn gemm_scalar(`の定義を次の全体で置き換えます。

```rust
// file="src/gpu_kernels.rs" mode="replace" item="pub fn gemm_scalar("
pub fn gemm_scalar(
	d: Dialect,
	m: usize,
	n: usize,
	k: usize,
	t: GemmTile,
) -> Result<Kernel, String> {
	if t.m == 0 || t.n == 0 || t.k == 0 {
		return Err("zero tile".into());
	}
	let local = t.m.checked_mul(t.n).ok_or("local overflow")?;
	let ap = t.k.checked_add(t.pad).ok_or("pitch overflow")?;
	let bp = if t.transpose_b { t.k } else { t.n }
		.checked_add(t.pad)
		.ok_or("pitch overflow")?;
	let al = t.m.checked_mul(ap).ok_or("local overflow")?;
	let bl = if t.transpose_b { t.n } else { t.k }
		.checked_mul(bp)
		.ok_or("local overflow")?;
	let bytes = al
		.checked_add(bl)
		.and_then(|n| n.checked_mul(4))
		.ok_or("local bytes overflow")?;
	let gn = groups(n, t.n)?;
	let gm = groups(m, t.m)?;
	let bload = if t.transpose_b {
		format!("bb[(z%{})*{bp}+z/{}]", t.n, t.n)
	} else {
		format!("bb[(z/{})*{bp}+z%{}]", t.n, t.n)
	};
	let bidx = if t.transpose_b {
		format!("bb[cn*{bp}+r]")
	} else {
		format!("bb[r*{bp}+cn]")
	};
	let body = format!(
		"{} float aa[{al}],bb[{bl}];size_t br=g/{gn},bc=g%{gn},rm=l/{tn},cn=l%{tn};size_t row=br*{tm}+rm,col=bc*{tn}+cn;float acc=0.0f;for(size_t ko=0;ko<{k};ko+={tk}){{for(size_t z=l;z<{tm}*{tk};z+={local}){{size_t ar=br*{tm}+z/{tk},ak=ko+z%{tk};aa[(z/{tk})*{ap}+z%{tk}]=(ar<{m}&&ak<{k})?b0[ar*{k}+ak]:0.0f;}}for(size_t z=l;z<{tk}*{tn};z+={local}){{size_t bk=ko+z/{tn},bn=bc*{tn}+z%{tn};{bload}=(bk<{k}&&bn<{n})?b1[bk*{n}+bn]:0.0f;}}{}for(size_t r=0;r<{tk};r++){{acc+=aa[rm*{ap}+r]*{bidx};}}{}}}if(row<{m}&&col<{n}){{b2[row*{n}+col]=acc;}}",
		shared(d),
		barrier(d),
		barrier(d),
		tn = t.n,
		tm = t.m,
		tk = t.k
	);
	spec(
		d,
		vec![
			m.checked_mul(k).ok_or("shape overflow")?,
			k.checked_mul(n).ok_or("shape overflow")?,
			m.checked_mul(n).ok_or("shape overflow")?,
		],
		body,
		gm.checked_mul(gn).ok_or("group overflow")?,
		local,
		bytes,
	)
}
```

`src/gpu.rs` の `pub fn header(`の定義を次の全体で置き換えます。

```rust
// file="src/gpu.rs" mode="replace" item="pub fn header("
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
```

`src/backends/opencl.rs` の `pub unsafe fn run(&self,`の定義を次の全体で置き換えます。

```rust
// file="src/backends/opencl.rs" mode="replace" item="pub unsafe fn run(&self,"
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

```sh
# check
cargo test --test book_19 todo_19_010 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## まず B の local buffer のみを転置し、協調 LOAD と計算時 INDEX の両方を変更する。

<!-- todo:19-011 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_011() {
	suite3::vector_loads_are_guarded();
}
```

`src/gpu_kernels.rs` の `pub fn gemm_scalar(`の定義を次の全体で置き換えます。

```rust
// file="src/gpu_kernels.rs" mode="replace" item="pub fn gemm_scalar("
pub fn gemm_scalar(
	d: Dialect,
	m: usize,
	n: usize,
	k: usize,
	t: GemmTile,
) -> Result<Kernel, String> {
	if t.m == 0 || t.n == 0 || t.k == 0 {
		return Err("zero tile".into());
	}
	let local = t.m.checked_mul(t.n).ok_or("local overflow")?;
	let ap = t.k.checked_add(t.pad).ok_or("pitch overflow")?;
	let bp = if t.transpose_b { t.k } else { t.n }
		.checked_add(t.pad)
		.ok_or("pitch overflow")?;
	let al = t.m.checked_mul(ap).ok_or("local overflow")?;
	let bl = if t.transpose_b { t.n } else { t.k }
		.checked_mul(bp)
		.ok_or("local overflow")?;
	let bytes = al
		.checked_add(bl)
		.and_then(|n| n.checked_mul(4))
		.ok_or("local bytes overflow")?;
	let gn = groups(n, t.n)?;
	let gm = groups(m, t.m)?;
	let bload = if t.transpose_b {
		format!("bb[(z%{})*{bp}+z/{}]", t.n, t.n)
	} else {
		format!("bb[(z/{})*{bp}+z%{}]", t.n, t.n)
	};
	let bidx = if t.transpose_b {
		format!("bb[cn*{bp}+r]")
	} else {
		format!("bb[r*{bp}+cn]")
	};
	let body = format!(
		"{} float aa[{al}],bb[{bl}];size_t br=g/{gn},bc=g%{gn},rm=l/{tn},cn=l%{tn};size_t row=br*{tm}+rm,col=bc*{tn}+cn;float acc=0.0f;for(size_t ko=0;ko<{k};ko+={tk}){{for(size_t z=l;z<{tm}*{tk};z+={local}){{size_t ar=br*{tm}+z/{tk},ak=ko+z%{tk};aa[(z/{tk})*{ap}+z%{tk}]=(ar<{m}&&ak<{k})?b0[ar*{k}+ak]:0.0f;}}for(size_t z=l;z<{tk}*{tn};z+={local}){{size_t bk=ko+z/{tn},bn=bc*{tn}+z%{tn};{bload}=(bk<{k}&&bn<{n})?b1[bk*{n}+bn]:0.0f;}}{}for(size_t r=0;r<{tk};r++){{acc+=aa[rm*{ap}+r]*{bidx};}}{}}}if(row<{m}&&col<{n}){{b2[row*{n}+col]=acc;}}",
		shared(d),
		barrier(d),
		barrier(d),
		tn = t.n,
		tm = t.m,
		tk = t.k
	);
	spec(
		d,
		vec![
			m.checked_mul(k).ok_or("shape overflow")?,
			k.checked_mul(n).ok_or("shape overflow")?,
			m.checked_mul(n).ok_or("shape overflow")?,
		],
		body,
		gm.checked_mul(gn).ok_or("group overflow")?,
		local,
		bytes,
	)
}
```

```sh
# check
cargo test --test book_19 todo_19_011 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## local pitch の padding を一つ追加し、確保 byte 数と INDEX の最大値を検証する。

<!-- todo:19-012 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_012() {
	suite3::vector_loads_are_guarded();
}
```

`src/gpu_kernels.rs` の `pub fn gemm_scalar(`の定義を次の全体で置き換えます。

```rust
// file="src/gpu_kernels.rs" mode="replace" item="pub fn gemm_scalar("
pub fn gemm_scalar(
	d: Dialect,
	m: usize,
	n: usize,
	k: usize,
	t: GemmTile,
) -> Result<Kernel, String> {
	if t.m == 0 || t.n == 0 || t.k == 0 {
		return Err("zero tile".into());
	}
	let local = t.m.checked_mul(t.n).ok_or("local overflow")?;
	let ap = t.k.checked_add(t.pad).ok_or("pitch overflow")?;
	let bp = if t.transpose_b { t.k } else { t.n }
		.checked_add(t.pad)
		.ok_or("pitch overflow")?;
	let al = t.m.checked_mul(ap).ok_or("local overflow")?;
	let bl = if t.transpose_b { t.n } else { t.k }
		.checked_mul(bp)
		.ok_or("local overflow")?;
	let bytes = al
		.checked_add(bl)
		.and_then(|n| n.checked_mul(4))
		.ok_or("local bytes overflow")?;
	let gn = groups(n, t.n)?;
	let gm = groups(m, t.m)?;
	let bload = if t.transpose_b {
		format!("bb[(z%{})*{bp}+z/{}]", t.n, t.n)
	} else {
		format!("bb[(z/{})*{bp}+z%{}]", t.n, t.n)
	};
	let bidx = if t.transpose_b {
		format!("bb[cn*{bp}+r]")
	} else {
		format!("bb[r*{bp}+cn]")
	};
	let body = format!(
		"{} float aa[{al}],bb[{bl}];size_t br=g/{gn},bc=g%{gn},rm=l/{tn},cn=l%{tn};size_t row=br*{tm}+rm,col=bc*{tn}+cn;float acc=0.0f;for(size_t ko=0;ko<{k};ko+={tk}){{for(size_t z=l;z<{tm}*{tk};z+={local}){{size_t ar=br*{tm}+z/{tk},ak=ko+z%{tk};aa[(z/{tk})*{ap}+z%{tk}]=(ar<{m}&&ak<{k})?b0[ar*{k}+ak]:0.0f;}}for(size_t z=l;z<{tk}*{tn};z+={local}){{size_t bk=ko+z/{tn},bn=bc*{tn}+z%{tn};{bload}=(bk<{k}&&bn<{n})?b1[bk*{n}+bn]:0.0f;}}{}for(size_t r=0;r<{tk};r++){{acc+=aa[rm*{ap}+r]*{bidx};}}{}}}if(row<{m}&&col<{n}){{b2[row*{n}+col]=acc;}}",
		shared(d),
		barrier(d),
		barrier(d),
		tn = t.n,
		tm = t.m,
		tk = t.k
	);
	spec(
		d,
		vec![
			m.checked_mul(k).ok_or("shape overflow")?,
			k.checked_mul(n).ok_or("shape overflow")?,
			m.checked_mul(n).ok_or("shape overflow")?,
		],
		body,
		gm.checked_mul(gn).ok_or("group overflow")?,
		local,
		bytes,
	)
}
```

```sh
# check
cargo test --test book_19 todo_19_012 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## vector LOAD は連続性、alignment、範囲を確認できる内側だけに適用する。

<!-- todo:19-013 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_013() {
	suite3::vector_loads_are_guarded();
}
```

`src/gpu_kernels.rs` の `pub fn gemm(`の定義を次の全体で置き換えます。

```rust
// file="src/gpu_kernels.rs" mode="replace" item="pub fn gemm("
pub fn gemm(d: Dialect, m: usize, n: usize, k: usize, t: GemmTile) -> Result<Kernel, String> {
	let mut kernel = gemm_scalar(d, m, n, k, t)?;
	if !k.is_multiple_of(4)
		|| !n.is_multiple_of(4)
		|| !t.k.is_multiple_of(4)
		|| !t.n.is_multiple_of(4)
	{
		return Ok(kernel);
	}
	let local = t.m * t.n;
	let ap = t.k + t.pad;
	let bp = if t.transpose_b { t.k } else { t.n } + t.pad;
	let aload = format!(
		"for(size_t z=l;z<{tm}*{tk};z+={local}){{size_t ar=br*{tm}+z/{tk},ak=ko+z%{tk};aa[(z/{tk})*{ap}+z%{tk}]=(ar<{m}&&ak<{k})?b0[ar*{k}+ak]:0.0f;}}",
		tm = t.m,
		tk = t.k
	);
	let bindex = if t.transpose_b {
		format!("bb[(z%{})*{bp}+z/{}]", t.n, t.n)
	} else {
		format!("bb[(z/{})*{bp}+z%{}]", t.n, t.n)
	};
	let bload = format!(
		"for(size_t z=l;z<{tk}*{tn};z+={local}){{size_t bk=ko+z/{tn},bn=bc*{tn}+z%{tn};{bindex}=(bk<{k}&&bn<{n})?b1[bk*{n}+bn]:0.0f;}}",
		tk = t.k,
		tn = t.n
	);
	let load = |pointer: &str| match d {
		Dialect::Metal => format!("*((device const float4*)({pointer}))"),
		Dialect::OpenCl => format!("vload4(0,{pointer})"),
	};
	let aa = format!(
		"for(size_t z=l*4;z<{tm}*{tk};z+={local}*4){{size_t ar=br*{tm}+z/{tk},ak=ko+z%{tk};float4 values=(float4)(0.0f);if(ar<{m}&&ak+3<{k}){{values={load};}}for(size_t c=0;c<4;c++){{aa[((z+c)/{tk})*{ap}+(z+c)%{tk}]=values[c];}}}}",
		tm = t.m,
		tk = t.k,
		load = load(&format!("b0+ar*{k}+ak"))
	);
	let store = if t.transpose_b {
		format!("bb[((z+c)%{})*{bp}+(z+c)/{}]", t.n, t.n)
	} else {
		format!("bb[((z+c)/{})*{bp}+(z+c)%{}]", t.n, t.n)
	};
	let bb = format!(
		"for(size_t z=l*4;z<{tk}*{tn};z+={local}*4){{size_t bk=ko+z/{tn},bn=bc*{tn}+z%{tn};float4 values=(float4)(0.0f);if(bk<{k}&&bn+3<{n}){{values={load};}}for(size_t c=0;c<4;c++){{{store}=values[c];}}}}",
		tk = t.k,
		tn = t.n,
		load = load(&format!("b1+bk*{n}+bn"))
	);
	kernel.source = kernel.source.replace(&aload, &aa).replace(&bload, &bb);
	Ok(kernel)
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
# check
cargo test --test book_19 todo_19_013 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## conv の出力座標と縮約座標を GEMM と同じタイルへ写像する。

<!-- todo:19-014 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_014() {
	suite1::conv_stride_padding_uses_real_gemm();
}
```

`src/conv_opt.rs` の `pub fn conv2d(`の定義を次の全体で置き換えます。

```rust
// file="src/conv_opt.rs" mode="replace" item="pub fn conv2d("
pub fn conv2d(
	g: &mut Graph,
	x: NodeId,
	w: NodeId,
	pad: (usize, usize),
	stride: usize,
) -> Result<NodeId> {
	if stride == 0 {
		return Err("conv stride=0".into());
	}
	let sx = g.shape(x).to_vec();
	let sw = g.shape(w).to_vec();
	if sx.len() != 4 || sw.len() != 4 || sx[1] != sw[1] || sw[2] == 0 || sw[3] == 0 {
		return Err("conv NCHW/OIHW shape".into());
	}
	let xp = g.pad(x, vec![(0, 0), (0, 0), (pad.0, pad.0), (pad.1, pad.1)])?;
	let sp = g.shape(xp).to_vec();
	let oh = sp[2].checked_sub(sw[2]).ok_or("conv kernel height")? / stride + 1;
	let ow = sp[3].checked_sub(sw[3]).ok_or("conv kernel width")? / stride + 1;
	let mut windows = vec![];
	for kh in 0..sw[2] {
		for kw in 0..sw[3] {
			let mut hs = vec![];
			for h in 0..oh {
				let index = h
					.checked_mul(stride)
					.and_then(|v| v.checked_add(kh))
					.ok_or("conv index overflow")?;
				if index > 1 << 24 {
					return Err("f32 gather index precision".into());
				}
				hs.push(g.constant(index as f32, vec![])?);
			}
			let mut ws = vec![];
			for col in 0..ow {
				let index = col
					.checked_mul(stride)
					.and_then(|v| v.checked_add(kw))
					.ok_or("conv index overflow")?;
				if index > 1 << 24 {
					return Err("f32 gather index precision".into());
				}
				ws.push(g.constant(index as f32, vec![])?);
			}
			let hi = g.stack(hs, 0)?;
			let wi = g.stack(ws, 0)?;
			let rows = g.gather(xp, hi, 2)?;
			windows.push(g.gather(rows, wi, 3)?);
		}
	}
	let patch = g.stack(windows, 4)?;
	let patch = g.permute(patch, vec![0, 2, 3, 1, 4])?;
	let reduction = sx[1]
		.checked_mul(sw[2])
		.and_then(|n| n.checked_mul(sw[3]))
		.ok_or("conv reduction shape")?;
	let positions = sx[0]
		.checked_mul(oh)
		.and_then(|n| n.checked_mul(ow))
		.ok_or("conv output shape")?;
	let matrix = g.reshape(patch, vec![positions, reduction])?;
	let matrix = g.stage(matrix)?;
	let weights = g.reshape(w, vec![sw[0], reduction])?;
	let weights = g.permute(weights, vec![1, 0])?;
	let weights = g.stage(weights)?;
	let product = g.matmul(matrix, weights)?;
	let output = g.reshape(product, vec![sx[0], oh, ow, sw[0]])?;
	g.permute(output, vec![0, 3, 1, 2])
}
```

`src/gemm_opt.rs` の `pub fn compile(`の定義を次の全体で置き換えます。

```rust
// file="src/gemm_opt.rs" mode="replace" item="pub fn compile("
pub fn compile(
	g: &Graph,
	outputs: &[usize],
	tile: (usize, usize, usize),
	workers: usize,
) -> Result<(Plan, String)> {
	let (tm, tn, tk) = tile;
	if tm == 0 || tn == 0 || tk == 0 {
		return Err("GEMM zero tile".into());
	}
	tm.checked_mul(tn)
		.and_then(|n| n.checked_mul(4))
		.filter(|&n| n <= 65536)
		.ok_or("register tile too large")?;
	let mut kernels = kernel_bodies(g, outputs, true)?;
	let mut matches = 0;
	let mut offsets = vec![];
	let mut offset = 0;
	for n in &g.nodes {
		offsets.push(offset);
		offset += n.shape.len();
	}
	for kernel in &mut kernels {
		if let Some(mat) = recognize(g, kernel.output) {
			let a = mat.a;
			let b = mat.b;
			let out = mat.output;
			let m = format!("s[{}]", offsets[a]);
			let k = format!("s[{}]", offsets[a] + 1);
			let n = format!("s[{}]", offsets[b] + 1);
			let gm = format!("({m}/{tm}+({m}%{tm}!=0))");
			let gn = format!("({n}/{tn}+({n}%{tn}!=0))");
			kernel.body = format!(
				"for (size_t i=0; i<{gm}*{gn}; ++i) {{size_t br=i/{gn},bc=i%{gn};float acc[{tm}][{tn}]={{0}};for(size_t ko=0;ko<{k};){{size_t klimit={k}-ko<{tk}?{k}-ko:{tk};for(size_t ki=0;ki<klimit;ki++){{size_t r=ko+ki;for(size_t mi=0;mi<{tm};mi++){{size_t row=br*{tm}+mi;if(row<{m}){{for(size_t ni=0;ni<{tn};ni++){{size_t col=bc*{tn}+ni;if(col<{n}){{acc[mi][ni]+=b{a}[row*{k}+r]*b{b}[r*{n}+col];}}}}}}}}}}ko+=klimit;}}for(size_t mi=0;mi<{tm};mi++){{size_t row=br*{tm}+mi;for(size_t ni=0;ni<{tn};ni++){{size_t col=bc*{tn}+ni;if(row<{m}&&col<{n}){{b{out}[row*{n}+col]=acc[mi][ni];}}}}}}}}\n"
			);
			matches += 1;
		}
	}
	let (plan, report) = compile_bodies(
		g,
		outputs,
		Options {
			workers,
			..Options::default()
		},
		kernels,
	)?;
	Ok((
		plan,
		format!("GEMM matches={matches}, tile={tile:?}; {report}"),
	))
}
```

```sh
# check
cargo test --test book_19 todo_19_014 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## まず stride=1、padding=0、groups=1 の direct conv を扱い、次に padding と stride を追加する。

<!-- todo:19-015 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_015() {
	suite1::conv_stride_padding_uses_real_gemm();
}
```

`src/conv_opt.rs` の `pub fn conv2d(`の定義を次の全体で置き換えます。

```rust
// file="src/conv_opt.rs" mode="replace" item="pub fn conv2d("
pub fn conv2d(
	g: &mut Graph,
	x: NodeId,
	w: NodeId,
	pad: (usize, usize),
	stride: usize,
) -> Result<NodeId> {
	if stride == 0 {
		return Err("conv stride=0".into());
	}
	let sx = g.shape(x).to_vec();
	let sw = g.shape(w).to_vec();
	if sx.len() != 4 || sw.len() != 4 || sx[1] != sw[1] || sw[2] == 0 || sw[3] == 0 {
		return Err("conv NCHW/OIHW shape".into());
	}
	let xp = g.pad(x, vec![(0, 0), (0, 0), (pad.0, pad.0), (pad.1, pad.1)])?;
	let sp = g.shape(xp).to_vec();
	let oh = sp[2].checked_sub(sw[2]).ok_or("conv kernel height")? / stride + 1;
	let ow = sp[3].checked_sub(sw[3]).ok_or("conv kernel width")? / stride + 1;
	let mut windows = vec![];
	for kh in 0..sw[2] {
		for kw in 0..sw[3] {
			let mut hs = vec![];
			for h in 0..oh {
				let index = h
					.checked_mul(stride)
					.and_then(|v| v.checked_add(kh))
					.ok_or("conv index overflow")?;
				if index > 1 << 24 {
					return Err("f32 gather index precision".into());
				}
				hs.push(g.constant(index as f32, vec![])?);
			}
			let mut ws = vec![];
			for col in 0..ow {
				let index = col
					.checked_mul(stride)
					.and_then(|v| v.checked_add(kw))
					.ok_or("conv index overflow")?;
				if index > 1 << 24 {
					return Err("f32 gather index precision".into());
				}
				ws.push(g.constant(index as f32, vec![])?);
			}
			let hi = g.stack(hs, 0)?;
			let wi = g.stack(ws, 0)?;
			let rows = g.gather(xp, hi, 2)?;
			windows.push(g.gather(rows, wi, 3)?);
		}
	}
	let patch = g.stack(windows, 4)?;
	let patch = g.permute(patch, vec![0, 2, 3, 1, 4])?;
	let reduction = sx[1]
		.checked_mul(sw[2])
		.and_then(|n| n.checked_mul(sw[3]))
		.ok_or("conv reduction shape")?;
	let positions = sx[0]
		.checked_mul(oh)
		.and_then(|n| n.checked_mul(ow))
		.ok_or("conv output shape")?;
	let matrix = g.reshape(patch, vec![positions, reduction])?;
	let matrix = g.stage(matrix)?;
	let weights = g.reshape(w, vec![sw[0], reduction])?;
	let weights = g.permute(weights, vec![1, 0])?;
	let weights = g.stage(weights)?;
	let product = g.matmul(matrix, weights)?;
	let output = g.reshape(product, vec![sx[0], oh, ow, sw[0]])?;
	g.permute(output, vec![0, 3, 1, 2])
}
```

`src/gemm_opt.rs` の `pub fn compile(`の定義を次の全体で置き換えます。

```rust
// file="src/gemm_opt.rs" mode="replace" item="pub fn compile("
pub fn compile(
	g: &Graph,
	outputs: &[usize],
	tile: (usize, usize, usize),
	workers: usize,
) -> Result<(Plan, String)> {
	let (tm, tn, tk) = tile;
	if tm == 0 || tn == 0 || tk == 0 {
		return Err("GEMM zero tile".into());
	}
	tm.checked_mul(tn)
		.and_then(|n| n.checked_mul(4))
		.filter(|&n| n <= 65536)
		.ok_or("register tile too large")?;
	let mut kernels = kernel_bodies(g, outputs, true)?;
	let mut matches = 0;
	let mut offsets = vec![];
	let mut offset = 0;
	for n in &g.nodes {
		offsets.push(offset);
		offset += n.shape.len();
	}
	for kernel in &mut kernels {
		if let Some(mat) = recognize(g, kernel.output) {
			let a = mat.a;
			let b = mat.b;
			let out = mat.output;
			let m = format!("s[{}]", offsets[a]);
			let k = format!("s[{}]", offsets[a] + 1);
			let n = format!("s[{}]", offsets[b] + 1);
			let gm = format!("({m}/{tm}+({m}%{tm}!=0))");
			let gn = format!("({n}/{tn}+({n}%{tn}!=0))");
			kernel.body = format!(
				"for (size_t i=0; i<{gm}*{gn}; ++i) {{size_t br=i/{gn},bc=i%{gn};float acc[{tm}][{tn}]={{0}};for(size_t ko=0;ko<{k};){{size_t klimit={k}-ko<{tk}?{k}-ko:{tk};for(size_t ki=0;ki<klimit;ki++){{size_t r=ko+ki;for(size_t mi=0;mi<{tm};mi++){{size_t row=br*{tm}+mi;if(row<{m}){{for(size_t ni=0;ni<{tn};ni++){{size_t col=bc*{tn}+ni;if(col<{n}){{acc[mi][ni]+=b{a}[row*{k}+r]*b{b}[r*{n}+col];}}}}}}}}}}ko+=klimit;}}for(size_t mi=0;mi<{tm};mi++){{size_t row=br*{tm}+mi;for(size_t ni=0;ni<{tn};ni++){{size_t col=bc*{tn}+ni;if(row<{m}&&col<{n}){{b{out}[row*{n}+col]=acc[mi][ni];}}}}}}}}\n"
			);
			matches += 1;
		}
	}
	let (plan, report) = compile_bodies(
		g,
		outputs,
		Options {
			workers,
			..Options::default()
		},
		kernels,
	)?;
	Ok((
		plan,
		format!("GEMM matches={matches}, tile={tile:?}; {report}"),
	))
}
```

```sh
# check
cargo test --test book_19 todo_19_015 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## Plan の各 kernel が読む範囲と書く範囲を storage base 単位で集計する。

<!-- todo:19-016 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_016() {
	suite2::actual_reuse_and_binary_cache();
}
```

`src/storage.rs` の `pub fn plan(`の定義を次の全体で置き換えます。

```rust
// file="src/storage.rs" mode="replace" item="pub fn plan("
pub fn plan(g: &Graph, outputs: &[usize], external: &[usize], reuse: bool) -> Result<StoragePlan> {
	let kernels = kernel_bodies(g, outputs, true)?;
	let mut uses: HashMap<usize, (usize, usize)> = HashMap::new();
	for (time, kernel) in kernels.iter().enumerate() {
		uses.entry(kernel.output)
			.and_modify(|v| v.1 = time)
			.or_insert((time, time));
		for &id in &kernel.dependencies {
			if kernel.body.contains(&format!("b{id}[")) {
				uses.entry(id)
					.and_modify(|v| v.1 = time)
					.or_insert((0, time));
			}
		}
	}
	let mut protected: HashSet<usize> = outputs.iter().chain(external).copied().collect();
	for id in g.topology(outputs)? {
		if matches!(g.nodes[id].op, Op::Input { .. }) {
			protected.insert(id);
		}
	}
	let mut intervals = uses.into_iter().collect::<Vec<_>>();
	intervals.sort_by_key(|&(id, (first, _))| (first, id));
	let mut allocations = vec![];
	let mut slot_sizes = vec![];
	let mut availability: Vec<usize> = vec![];
	let mut protected_slots = HashSet::new();
	for (id, (first, last)) in intervals {
		if matches!(g.nodes[id].op, Op::Input { .. }) {
			continue;
		}
		let n = elements(g.shape(id))?;
		let protect = protected.contains(&id);
		let slot = if reuse && !protect {
			(0..slot_sizes.len()).find(|&s| {
				!protected_slots.contains(&s) && availability[s] < first && slot_sizes[s] == n
			})
		} else {
			None
		};
		let slot = slot.unwrap_or_else(|| {
			let slot = slot_sizes.len();
			slot_sizes.push(n);
			availability.push(last);
			slot
		});
		availability[slot] = last;
		if protect {
			protected_slots.insert(slot);
		}
		allocations.push(Allocation {
			node: id,
			slot,
			first,
			last,
			elements: n,
			protected: protect,
		});
	}
	Ok(StoragePlan {
		slots: slot_sizes.len(),
		slot_sizes,
		allocations,
	})
}
```

```sh
# check
cargo test --test book_19 todo_19_016 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 入力、出力、学習パラメータ、KV cache、外部へ返した view を再利用対象から外す。

<!-- todo:19-017 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_017() {
	suite2::actual_reuse_and_binary_cache();
}
```

`src/storage.rs` の `pub fn plan(`の定義を次の全体で置き換えます。

```rust
// file="src/storage.rs" mode="replace" item="pub fn plan("
pub fn plan(g: &Graph, outputs: &[usize], external: &[usize], reuse: bool) -> Result<StoragePlan> {
	let kernels = kernel_bodies(g, outputs, true)?;
	let mut uses: HashMap<usize, (usize, usize)> = HashMap::new();
	for (time, kernel) in kernels.iter().enumerate() {
		uses.entry(kernel.output)
			.and_modify(|v| v.1 = time)
			.or_insert((time, time));
		for &id in &kernel.dependencies {
			if kernel.body.contains(&format!("b{id}[")) {
				uses.entry(id)
					.and_modify(|v| v.1 = time)
					.or_insert((0, time));
			}
		}
	}
	let mut protected: HashSet<usize> = outputs.iter().chain(external).copied().collect();
	for id in g.topology(outputs)? {
		if matches!(g.nodes[id].op, Op::Input { .. }) {
			protected.insert(id);
		}
	}
	let mut intervals = uses.into_iter().collect::<Vec<_>>();
	intervals.sort_by_key(|&(id, (first, _))| (first, id));
	let mut allocations = vec![];
	let mut slot_sizes = vec![];
	let mut availability: Vec<usize> = vec![];
	let mut protected_slots = HashSet::new();
	for (id, (first, last)) in intervals {
		if matches!(g.nodes[id].op, Op::Input { .. }) {
			continue;
		}
		let n = elements(g.shape(id))?;
		let protect = protected.contains(&id);
		let slot = if reuse && !protect {
			(0..slot_sizes.len()).find(|&s| {
				!protected_slots.contains(&s) && availability[s] < first && slot_sizes[s] == n
			})
		} else {
			None
		};
		let slot = slot.unwrap_or_else(|| {
			let slot = slot_sizes.len();
			slot_sizes.push(n);
			availability.push(last);
			slot
		});
		availability[slot] = last;
		if protect {
			protected_slots.insert(slot);
		}
		allocations.push(Allocation {
			node: id,
			slot,
			first,
			last,
			elements: n,
			protected: protect,
		});
	}
	Ok(StoragePlan {
		slots: slot_sizes.len(),
		slot_sizes,
		allocations,
	})
}
```

```sh
# check
cargo test --test book_19 todo_19_017 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 単一 queue の完了順を前提に、中間 buffer の最初の利用と最後の利用を求める。

<!-- todo:19-018 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_018() {
	suite2::actual_reuse_and_binary_cache();
}
```

`src/storage.rs` の `pub fn plan(`の定義を次の全体で置き換えます。

```rust
// file="src/storage.rs" mode="replace" item="pub fn plan("
pub fn plan(g: &Graph, outputs: &[usize], external: &[usize], reuse: bool) -> Result<StoragePlan> {
	let kernels = kernel_bodies(g, outputs, true)?;
	let mut uses: HashMap<usize, (usize, usize)> = HashMap::new();
	for (time, kernel) in kernels.iter().enumerate() {
		uses.entry(kernel.output)
			.and_modify(|v| v.1 = time)
			.or_insert((time, time));
		for &id in &kernel.dependencies {
			if kernel.body.contains(&format!("b{id}[")) {
				uses.entry(id)
					.and_modify(|v| v.1 = time)
					.or_insert((0, time));
			}
		}
	}
	let mut protected: HashSet<usize> = outputs.iter().chain(external).copied().collect();
	for id in g.topology(outputs)? {
		if matches!(g.nodes[id].op, Op::Input { .. }) {
			protected.insert(id);
		}
	}
	let mut intervals = uses.into_iter().collect::<Vec<_>>();
	intervals.sort_by_key(|&(id, (first, _))| (first, id));
	let mut allocations = vec![];
	let mut slot_sizes = vec![];
	let mut availability: Vec<usize> = vec![];
	let mut protected_slots = HashSet::new();
	for (id, (first, last)) in intervals {
		if matches!(g.nodes[id].op, Op::Input { .. }) {
			continue;
		}
		let n = elements(g.shape(id))?;
		let protect = protected.contains(&id);
		let slot = if reuse && !protect {
			(0..slot_sizes.len()).find(|&s| {
				!protected_slots.contains(&s) && availability[s] < first && slot_sizes[s] == n
			})
		} else {
			None
		};
		let slot = slot.unwrap_or_else(|| {
			let slot = slot_sizes.len();
			slot_sizes.push(n);
			availability.push(last);
			slot
		});
		availability[slot] = last;
		if protect {
			protected_slots.insert(slot);
		}
		allocations.push(Allocation {
			node: id,
			slot,
			first,
			last,
			elements: n,
			protected: protect,
		});
	}
	Ok(StoragePlan {
		slots: slot_sizes.len(),
		slot_sizes,
		allocations,
	})
}
```

```sh
# check
cargo test --test book_19 todo_19_018 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## まず同じ device、dtype、サイズ、alignment の確保だけを free list で再利用する。

<!-- todo:19-019 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_019() {
	suite2::actual_reuse_and_binary_cache();
}
```

`src/storage.rs` の `pub fn plan(`の定義を次の全体で置き換えます。

```rust
// file="src/storage.rs" mode="replace" item="pub fn plan("
pub fn plan(g: &Graph, outputs: &[usize], external: &[usize], reuse: bool) -> Result<StoragePlan> {
	let kernels = kernel_bodies(g, outputs, true)?;
	let mut uses: HashMap<usize, (usize, usize)> = HashMap::new();
	for (time, kernel) in kernels.iter().enumerate() {
		uses.entry(kernel.output)
			.and_modify(|v| v.1 = time)
			.or_insert((time, time));
		for &id in &kernel.dependencies {
			if kernel.body.contains(&format!("b{id}[")) {
				uses.entry(id)
					.and_modify(|v| v.1 = time)
					.or_insert((0, time));
			}
		}
	}
	let mut protected: HashSet<usize> = outputs.iter().chain(external).copied().collect();
	for id in g.topology(outputs)? {
		if matches!(g.nodes[id].op, Op::Input { .. }) {
			protected.insert(id);
		}
	}
	let mut intervals = uses.into_iter().collect::<Vec<_>>();
	intervals.sort_by_key(|&(id, (first, _))| (first, id));
	let mut allocations = vec![];
	let mut slot_sizes = vec![];
	let mut availability: Vec<usize> = vec![];
	let mut protected_slots = HashSet::new();
	for (id, (first, last)) in intervals {
		if matches!(g.nodes[id].op, Op::Input { .. }) {
			continue;
		}
		let n = elements(g.shape(id))?;
		let protect = protected.contains(&id);
		let slot = if reuse && !protect {
			(0..slot_sizes.len()).find(|&s| {
				!protected_slots.contains(&s) && availability[s] < first && slot_sizes[s] == n
			})
		} else {
			None
		};
		let slot = slot.unwrap_or_else(|| {
			let slot = slot_sizes.len();
			slot_sizes.push(n);
			availability.push(last);
			slot
		});
		availability[slot] = last;
		if protect {
			protected_slots.insert(slot);
		}
		allocations.push(Allocation {
			node: id,
			slot,
			first,
			last,
			elements: n,
			protected: protect,
		});
	}
	Ok(StoragePlan {
		slots: slot_sizes.len(),
		slot_sizes,
		allocations,
	})
}
```

`src/storage.rs` の `pub fn run(&self,`の定義を次の全体で置き換えます。

```rust
// file="src/storage.rs" mode="replace" item="pub fn run(&self,"
	pub fn run(&self, inputs: &[Buffer]) -> Result<(Vec<Buffer>, Measurements)> {
		let total = Instant::now();
		self.graph.validate_inputs(inputs, &self.outputs)?;
		self.graph.validate_indices(inputs, &self.outputs)?;
		let allocation = Instant::now();
		let mut blocks = self
			.storage
			.slot_sizes
			.iter()
			.map(|&n| vec![0f32; n])
			.collect::<Vec<_>>();
		let mut pointers = vec![std::ptr::null_mut::<c_void>(); self.graph.nodes.len()];
		for a in &self.storage.allocations {
			pointers[a.node] = blocks[a.slot].as_mut_ptr().cast();
		}
		for id in self.graph.topology(&self.outputs)? {
			if let Op::Input { slot } = self.graph.nodes[id].op {
				pointers[id] = inputs[slot].data.as_ptr().cast_mut().cast();
			}
		}
		let mut measurements = Measurements {
			compile: self.compile,
			allocation: allocation.elapsed(),
			..Measurements::default()
		};
		let dims = self
			.graph
			.nodes
			.iter()
			.flat_map(|n| n.shape.iter().map(|&d| d as u64))
			.collect::<Vec<_>>();
		type Entry = unsafe extern "C" fn(*const *mut c_void, *const u64) -> i32;
		let entry: Entry = unsafe { std::mem::transmute(self.library.symbol()) };
		let kernel = Instant::now();
		let status = unsafe { entry(pointers.as_ptr(), dims.as_ptr()) };
		measurements.native_and_sync = kernel.elapsed();
		if status != 0 {
			return Err(format!("cached native kernel status {status}"));
		}
		let transfer = Instant::now();
		let result = self
			.outputs
			.iter()
			.map(|&id| {
				let data = if let Op::Input { slot } = self.graph.nodes[id].op {
					inputs[slot].data.clone()
				} else {
					let a = self
						.storage
						.allocations
						.iter()
						.find(|a| a.node == id)
						.ok_or("missing output allocation")?;
					blocks[a.slot].clone()
				};
				Buffer::new(self.graph.shape(id).to_vec(), data)
			})
			.collect::<Result<Vec<_>>>()?;
		measurements.transfer = transfer.elapsed();
		measurements.total = total.elapsed();
		Ok((result, measurements))
	}
```

```sh
# check
cargo test --test book_19 todo_19_019 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 同じ kernel 内で読まれる buffer を、その kernel の出力に割り当てない。

<!-- todo:19-020 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_020() {
	suite2::actual_reuse_and_binary_cache();
}
```

`src/storage.rs` の `pub fn plan(`の定義を次の全体で置き換えます。

```rust
// file="src/storage.rs" mode="replace" item="pub fn plan("
pub fn plan(g: &Graph, outputs: &[usize], external: &[usize], reuse: bool) -> Result<StoragePlan> {
	let kernels = kernel_bodies(g, outputs, true)?;
	let mut uses: HashMap<usize, (usize, usize)> = HashMap::new();
	for (time, kernel) in kernels.iter().enumerate() {
		uses.entry(kernel.output)
			.and_modify(|v| v.1 = time)
			.or_insert((time, time));
		for &id in &kernel.dependencies {
			if kernel.body.contains(&format!("b{id}[")) {
				uses.entry(id)
					.and_modify(|v| v.1 = time)
					.or_insert((0, time));
			}
		}
	}
	let mut protected: HashSet<usize> = outputs.iter().chain(external).copied().collect();
	for id in g.topology(outputs)? {
		if matches!(g.nodes[id].op, Op::Input { .. }) {
			protected.insert(id);
		}
	}
	let mut intervals = uses.into_iter().collect::<Vec<_>>();
	intervals.sort_by_key(|&(id, (first, _))| (first, id));
	let mut allocations = vec![];
	let mut slot_sizes = vec![];
	let mut availability: Vec<usize> = vec![];
	let mut protected_slots = HashSet::new();
	for (id, (first, last)) in intervals {
		if matches!(g.nodes[id].op, Op::Input { .. }) {
			continue;
		}
		let n = elements(g.shape(id))?;
		let protect = protected.contains(&id);
		let slot = if reuse && !protect {
			(0..slot_sizes.len()).find(|&s| {
				!protected_slots.contains(&s) && availability[s] < first && slot_sizes[s] == n
			})
		} else {
			None
		};
		let slot = slot.unwrap_or_else(|| {
			let slot = slot_sizes.len();
			slot_sizes.push(n);
			availability.push(last);
			slot
		});
		availability[slot] = last;
		if protect {
			protected_slots.insert(slot);
		}
		allocations.push(Allocation {
			node: id,
			slot,
			first,
			last,
			elements: n,
			protected: protect,
		});
	}
	Ok(StoragePlan {
		slots: slot_sizes.len(),
		slot_sizes,
		allocations,
	})
}
```

```sh
# check
cargo test --test book_19 todo_19_020 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## cache key に backend、device capability、生成 source、clang 等の compiler 識別、flags、ABI version を含める。

<!-- todo:19-021 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_021() {
	suite2::actual_reuse_and_binary_cache();
}
```

`src/storage.rs` の `pub fn cpu(`の定義を次の全体で置き換えます。

```rust
// file="src/storage.rs" mode="replace" item="pub fn cpu("
	pub fn cpu(source: String, specialization: String) -> Result<Self> {
		let compiler = Command::new("clang")
			.arg("--version")
			.output()
			.map_err(|e| e.to_string())?;
		if !compiler.status.success() {
			return Err("clang --version failed".into());
		}
		Ok(Self {
			backend: "cpu".into(),
			device: std::env::consts::ARCH.into(),
			capability: "f32,unaligned-vector4,pthread".into(),
			source,
			compiler: String::from_utf8_lossy(&compiler.stdout).into(),
			flags: "-std=c11 -O2 -fno-fast-math".into(),
			abi: 1,
			specialization,
		})
	}
```

```sh
# check
cargo test --test book_19 todo_19_021 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## shape 特化、tile、dtype、math 設定が source 以外で決まる場合も key に含める。

<!-- todo:19-022 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_022() {
	suite2::actual_reuse_and_binary_cache();
}
```

`src/storage.rs` の `pub fn cpu(`の定義を次の全体で置き換えます。

```rust
// file="src/storage.rs" mode="replace" item="pub fn cpu("
	pub fn cpu(source: String, specialization: String) -> Result<Self> {
		let compiler = Command::new("clang")
			.arg("--version")
			.output()
			.map_err(|e| e.to_string())?;
		if !compiler.status.success() {
			return Err("clang --version failed".into());
		}
		Ok(Self {
			backend: "cpu".into(),
			device: std::env::consts::ARCH.into(),
			capability: "f32,unaligned-vector4,pthread".into(),
			source,
			compiler: String::from_utf8_lossy(&compiler.stdout).into(),
			flags: "-std=c11 -O2 -fno-fast-math".into(),
			abi: 1,
			specialization,
		})
	}
```

```sh
# check
cargo test --test book_19 todo_19_022 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## cache hit の binary と cold compile の binary で、同じ数値結果を確認する。

<!-- todo:19-023 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_023() {
	suite2::actual_reuse_and_binary_cache();
}
```

```sh
# check
cargo test --test book_19 todo_19_023 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## Plan の入力スロットに違う A、B、重みを渡し、コードを再利用して結果が変わることを確かめる。

<!-- todo:19-024 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_024() {
	suite2::actual_reuse_and_binary_cache();
}
```

```sh
# check
cargo test --test book_19 todo_19_024 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 計測記録に compile、確保、host/device 転送、kernel、同期、全体の区間を分けて出す。

<!-- todo:19-025 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_025() {
	suite2::actual_reuse_and_binary_cache();
}
```

`src/storage.rs` の `pub fn run(&self,`の定義を次の全体で置き換えます。

```rust
// file="src/storage.rs" mode="replace" item="pub fn run(&self,"
	pub fn run(&self, inputs: &[Buffer]) -> Result<(Vec<Buffer>, Measurements)> {
		let total = Instant::now();
		self.graph.validate_inputs(inputs, &self.outputs)?;
		self.graph.validate_indices(inputs, &self.outputs)?;
		let allocation = Instant::now();
		let mut blocks = self
			.storage
			.slot_sizes
			.iter()
			.map(|&n| vec![0f32; n])
			.collect::<Vec<_>>();
		let mut pointers = vec![std::ptr::null_mut::<c_void>(); self.graph.nodes.len()];
		for a in &self.storage.allocations {
			pointers[a.node] = blocks[a.slot].as_mut_ptr().cast();
		}
		for id in self.graph.topology(&self.outputs)? {
			if let Op::Input { slot } = self.graph.nodes[id].op {
				pointers[id] = inputs[slot].data.as_ptr().cast_mut().cast();
			}
		}
		let mut measurements = Measurements {
			compile: self.compile,
			allocation: allocation.elapsed(),
			..Measurements::default()
		};
		let dims = self
			.graph
			.nodes
			.iter()
			.flat_map(|n| n.shape.iter().map(|&d| d as u64))
			.collect::<Vec<_>>();
		type Entry = unsafe extern "C" fn(*const *mut c_void, *const u64) -> i32;
		let entry: Entry = unsafe { std::mem::transmute(self.library.symbol()) };
		let kernel = Instant::now();
		let status = unsafe { entry(pointers.as_ptr(), dims.as_ptr()) };
		measurements.native_and_sync = kernel.elapsed();
		if status != 0 {
			return Err(format!("cached native kernel status {status}"));
		}
		let transfer = Instant::now();
		let result = self
			.outputs
			.iter()
			.map(|&id| {
				let data = if let Op::Input { slot } = self.graph.nodes[id].op {
					inputs[slot].data.clone()
				} else {
					let a = self
						.storage
						.allocations
						.iter()
						.find(|a| a.node == id)
						.ok_or("missing output allocation")?;
					blocks[a.slot].clone()
				};
				Buffer::new(self.graph.shape(id).to_vec(), data)
			})
			.collect::<Result<Vec<_>>>()?;
		measurements.transfer = transfer.elapsed();
		measurements.total = total.elapsed();
		Ok((result, measurements))
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

`src/backends/opencl.rs` の `pub unsafe fn run(&self,`の定義を次の全体で置き換えます。

```rust
// file="src/backends/opencl.rs" mode="replace" item="pub unsafe fn run(&self,"
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

```sh
# check
cargo test --test book_19 todo_19_025 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 浮動小数点 ALU の回数にループの反復数と launch の担当数を掛ける FLOP 見積りを追加する。

<!-- todo:19-026 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_026() {
	suite0::work_estimate();
}
```

`src/gemm_opt.rs` の `pub fn flops(`の定義を次の全体で置き換えます。

```rust
// file="src/gemm_opt.rs" mode="replace" item="pub fn flops("
pub fn flops(m: usize, n: usize, k: usize, tile: (usize, usize, usize)) -> Result<Flops> {
	let (tm, tn, tk) = tile;
	if tm == 0 || tn == 0 || tk == 0 {
		return Err("zero tile".into());
	}
	let useful = 2u128
		.checked_mul(m as u128)
		.and_then(|v| v.checked_mul(n as u128))
		.and_then(|v| v.checked_mul(k as u128))
		.ok_or("FLOP overflow")?;
	let padded = [
		m.div_ceil(tm) as u128 * tm as u128,
		n.div_ceil(tn) as u128 * tn as u128,
		k.div_ceil(tk) as u128 * tk as u128,
	]
	.into_iter()
	.try_fold(2u128, |acc, n| {
		acc.checked_mul(n).ok_or("padded FLOP overflow")
	})?;
	Ok(Flops { useful, padded })
}
```

```sh
# check
cargo test --test book_19 todo_19_026 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 数学的に必要な仕事と、padding した tile で実行する仕事の見積りを別に出す。

<!-- todo:19-027 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_027() {
	suite0::work_estimate();
}
```

`src/gemm_opt.rs` の `pub fn flops(`の定義を次の全体で置き換えます。

```rust
// file="src/gemm_opt.rs" mode="replace" item="pub fn flops("
pub fn flops(m: usize, n: usize, k: usize, tile: (usize, usize, usize)) -> Result<Flops> {
	let (tm, tn, tk) = tile;
	if tm == 0 || tn == 0 || tk == 0 {
		return Err("zero tile".into());
	}
	let useful = 2u128
		.checked_mul(m as u128)
		.and_then(|v| v.checked_mul(n as u128))
		.and_then(|v| v.checked_mul(k as u128))
		.ok_or("FLOP overflow")?;
	let padded = [
		m.div_ceil(tm) as u128 * tm as u128,
		n.div_ceil(tn) as u128 * tn as u128,
		k.div_ceil(tk) as u128 * tk as u128,
	]
	.into_iter()
	.try_fold(2u128, |acc, n| {
		acc.checked_mul(n).ok_or("padded FLOP overflow")
	})?;
	Ok(Flops { useful, padded })
}
```

```sh
# check
cargo test --test book_19 todo_19_027 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 整数値の小例は上記の期待値と一致する。

<!-- todo:19-028 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_028() {
	suite0::tiled_graph_and_rebind();
}
```

```sh
# check
cargo test --test book_19 todo_19_028 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 固定 seed の f32 入力は scalar f64 reference に対し `abs_err <= 1e-4 + 1e-4*abs(ref)` を初期基準とする。

<!-- todo:19-029 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_029() {
	suite0::tiled_graph_and_rebind();
}
```

```sh
# check
cargo test --test book_19 todo_19_029 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 大きい K では誤差を記録し、加算順の差を理由に無条件で許容値を広げない。

<!-- todo:19-030 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_030() {
	suite0::tiled_graph_and_rebind();
}
```

```sh
# check
cargo test --test book_19 todo_19_030 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## `M,N,K=1` と tile 前後の寸法、ゼロ入力、負値、非連続 view を試す。

<!-- todo:19-031 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_031() {
	suite0::tiled_graph_and_rebind();
}
```

```sh
# check
cargo test --test book_19 todo_19_031 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## dump で全 lane の barrier 到達、local buffer 内の添字、出力の単独書込みを確認する。

<!-- todo:19-032 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_032() {
	suite0::tiled_graph_and_rebind();
}
```

```sh
# check
cargo test --test book_19 todo_19_032 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 再利用を無効にした結果、再利用した結果、cache hit の結果が一致する。

<!-- todo:19-033 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_033() {
	suite2::actual_reuse_and_binary_cache();
}
```

```sh
# check
cargo test --test book_19 todo_19_033 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## まだ測っていない backend を「高速化済み」「実機確認済み」と書かない。

<!-- todo:19-034 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_034() {
	suite3::vector_loads_are_guarded();
}
```

```sh
# check
cargo test --test book_19 todo_19_034 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## FLOP の見積りが上記の20と24になり、tile の端数を増やしても有効仕事の数は変わらない。

<!-- todo:19-035 -->

実装は[この章の `gemm_opt::recognize、gemm_opt::compile、conv_opt::conv2d、storage::plan、MemoryPlan::run、CompileCache::get`](#gemm-implementation) にある。次の確認を同じファイルへ追記する。

```rust
// file=tests/book_19.rs mode=append
#[test]
fn todo_19_035() {
	suite0::work_estimate();
}
```

```sh
# check
cargo test --test book_19 todo_19_035 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 確認記録

```sh
# check
cargo test
```

GPU の source は CPU の numerical test と分けて扱う。上の CPU 成功は GPU の実機確認を意味しない。

