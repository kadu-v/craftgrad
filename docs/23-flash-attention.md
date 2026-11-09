# 23. 通常 attention と online attention を同じモデルで比べる

この章は前章までの `chibigrad` に追加する。掲載順に `file=` のパスへ保存する。`mode=write` はそのファイル全体を置き換え、`mode=append` は末尾へ追記する。既存の別ファイルは残す。各コードは省略なしの実装である。

CPU の実行結果と GPU の実機結果を別に記録する。`requires=metal` と `requires=opencl` の確認は、その device がある環境で実行する。機器がなければ `NOT_RUN` と記録し、source 検査の成功へ置き換えない。

`reference` は scalar f64、通常版は Graph の compile、online 版は C の compile を使う。全 mask 行は 0 を返し、無効な K と V を読まない。`Boundary` は同じ Graph の QKV と通常 attention を残し、推論の compile 時だけ境界の kernel を online C に置き換える。学習は通常版へ固定する。

<a id="flash-implementation"></a>

## 実装を保存する

```rust file=src/lib.rs mode=append
pub mod flash;
```

```rust file=src/flash.rs mode=write
use crate::gpu::{Dialect, Kernel, header};
#[derive(Clone, Copy, Debug)]
pub struct AttentionShape {
	pub queries: usize,
	pub keys: usize,
	pub heads: usize,
	pub kv_heads: usize,
	pub dim: usize,
	pub query_start: usize,
	pub causal: bool,
}
impl AttentionShape {
	pub fn validate(&self, q: &[f32], k: &[f32], v: &[f32], valid: &[bool]) -> Result<(), String> {
		if self.dim == 0
			|| self.heads == 0
			|| self.kv_heads == 0
			|| !self.heads.is_multiple_of(self.kv_heads)
		{
			return Err("attention heads/dim".into());
		}
		let qn = self
			.queries
			.checked_mul(self.heads)
			.and_then(|n| n.checked_mul(self.dim))
			.ok_or("attention shape overflow")?;
		let kn = self
			.keys
			.checked_mul(self.kv_heads)
			.and_then(|n| n.checked_mul(self.dim))
			.ok_or("attention shape overflow")?;
		self.query_start
			.checked_add(self.queries)
			.ok_or("query position overflow")?;
		if q.len() != qn || k.len() != kn || v.len() != kn || valid.len() != self.keys {
			return Err("attention input shape".into());
		}
		Ok(())
	}
	fn allowed(&self, t: usize, s: usize, valid: &[bool]) -> bool {
		valid[s] && (!self.causal || s <= self.query_start + t)
	}
}
#[derive(Debug)]
pub struct Reference {
	pub scores: Vec<f64>,
	pub masked_scores: Vec<f64>,
	pub probabilities: Vec<f64>,
	pub output: Vec<f64>,
}
pub fn reference(
	s: AttentionShape,
	q: &[f32],
	k: &[f32],
	v: &[f32],
	valid: &[bool],
) -> Result<Reference, String> {
	s.validate(q, k, v, valid)?;
	let mut result = Reference {
		scores: vec![0.; s.queries * s.heads * s.keys],
		masked_scores: vec![f64::NEG_INFINITY; s.queries * s.heads * s.keys],
		probabilities: vec![0.; s.queries * s.heads * s.keys],
		output: vec![0.; q.len()],
	};
	let scale = 1. / (s.dim as f64).sqrt();
	for t in 0..s.queries {
		for h in 0..s.heads {
			let row = (t * s.heads + h) * s.keys;
			let qbase = (t * s.heads + h) * s.dim;
			let kh = h / (s.heads / s.kv_heads);
			let mut max = f64::NEG_INFINITY;
			for j in 0..s.keys {
				if !s.allowed(t, j, valid) {
					continue;
				}
				let base = (j * s.kv_heads + kh) * s.dim;
				let score = (0..s.dim)
					.map(|d| q[qbase + d] as f64 * k[base + d] as f64)
					.sum::<f64>() * scale;
				result.scores[row + j] = score;
				result.masked_scores[row + j] = score;
				max = max.max(score);
			}
			if max == f64::NEG_INFINITY {
				continue;
			}
			let mut sum = 0.;
			for j in 0..s.keys {
				if s.allowed(t, j, valid) {
					result.probabilities[row + j] = (result.masked_scores[row + j] - max).exp();
					sum += result.probabilities[row + j];
				}
			}
			for j in 0..s.keys {
				if s.allowed(t, j, valid) {
					let p = result.probabilities[row + j] / sum;
					result.probabilities[row + j] = p;
					for d in 0..s.dim {
						result.output[qbase + d] += p * v[(j * s.kv_heads + kh) * s.dim + d] as f64;
					}
				}
			}
		}
	}
	Ok(result)
}
pub fn online(
	s: AttentionShape,
	q: &[f32],
	k: &[f32],
	v: &[f32],
	valid: &[bool],
	tile: usize,
) -> Result<Vec<f32>, String> {
	s.validate(q, k, v, valid)?;
	if tile == 0 {
		return Err("attention tile=0".into());
	}
	let mut output = vec![0.; q.len()];
	let scale = 1. / (s.dim as f32).sqrt();
	for t in 0..s.queries {
		for h in 0..s.heads {
			let qb = (t * s.heads + h) * s.dim;
			let kh = h / (s.heads / s.kv_heads);
			let mut m = f32::NEG_INFINITY;
			let mut l = 0f32;
			let mut u = vec![0f32; s.dim];
			for start in (0..s.keys).step_by(tile) {
				let end = start + tile.min(s.keys - start);
				let mut scores = Vec::new();
				let mut tm = f32::NEG_INFINITY;
				for j in start..end {
					if s.allowed(t, j, valid) {
						let b = (j * s.kv_heads + kh) * s.dim;
						let score = (0..s.dim).map(|d| q[qb + d] * k[b + d]).sum::<f32>() * scale;
						tm = tm.max(score);
						scores.push((j, score));
					}
				}
				if scores.is_empty() {
					continue;
				}
				let nm = m.max(tm);
				let alpha = if l == 0. { 0. } else { (m - nm).exp() };
				l *= alpha;
				for x in &mut u {
					*x *= alpha;
				}
				for (j, score) in scores {
					let p = (score - nm).exp();
					l += p;
					for d in 0..s.dim {
						u[d] += p * v[(j * s.kv_heads + kh) * s.dim + d];
					}
				}
				m = nm;
			}
			if l > 0. {
				for d in 0..s.dim {
					output[qb + d] = u[d] / l;
				}
			}
		}
	}
	Ok(output)
}
pub fn shader(d: Dialect, s: AttentionShape, tile: usize) -> Result<Kernel, String> {
	if s.dim == 0 || s.heads == 0 || s.kv_heads == 0 || !s.heads.is_multiple_of(s.kv_heads) {
		return Err("attention heads/dim".into());
	}
	s.queries
		.checked_mul(s.heads)
		.and_then(|n| n.checked_mul(s.dim))
		.ok_or("attention query shape overflow")?;
	s.keys
		.checked_mul(s.kv_heads)
		.and_then(|n| n.checked_mul(s.dim))
		.ok_or("attention key shape overflow")?;
	s.query_start
		.checked_add(s.queries)
		.ok_or("attention position overflow")?;
	if tile == 0 {
		return Err("attention tile=0".into());
	}
	let bytes = tile
		.checked_mul(s.dim)
		.and_then(|n| n.checked_mul(8))
		.ok_or("attention local overflow")?;
	let space = match d {
		Dialect::Metal => "threadgroup",
		Dialect::OpenCl => "__local",
	};
	let barrier = match d {
		Dialect::Metal => "threadgroup_barrier(mem_flags::mem_threadgroup);",
		Dialect::OpenCl => "barrier(CLK_LOCAL_MEM_FENCE);",
	};
	let allowed = if s.causal {
		format!("j<={}+t", s.query_start)
	} else {
		"1".into()
	};
	let body = format!(
		"{space} float kk[{tile}*{dim}],vv[{tile}*{dim}];size_t t=g/{heads},h=g%{heads},kh=h/{ratio},qb=g*{dim};float m=-INFINITY,den=0.0f,u[{dim}];for(size_t a=0;a<{dim};a++){{u[a]=0.0f;}}for(size_t start=0;start<{keys};start+={tile}){{for(size_t z=l;z<{tile}*{dim};z+=32){{size_t j=start+z/{dim},a=z%{dim};bool ok=j<{keys}&&({allowed})&&b3[j]!=0.0f;if(ok){{kk[z]=b1[(j*{kv}+kh)*{dim}+a];vv[z]=b2[(j*{kv}+kh)*{dim}+a];}}else{{kk[z]=0.0f;vv[z]=0.0f;}}}}{barrier}if(l==0){{for(size_t off=0;off<{tile};off++){{size_t j=start+off;if(j<{keys}&&({allowed})&&b3[j]!=0.0f){{float score=0.0f;for(size_t a=0;a<{dim};a++){{score+=b0[qb+a]*kk[off*{dim}+a];}}score*=1.0f/sqrt((float){dim});float nm=fmax(m,score);float alpha=den==0.0f?0.0f:exp(m-nm),p=exp(score-nm);den=den*alpha+p;for(size_t a=0;a<{dim};a++){{u[a]=u[a]*alpha+p*vv[off*{dim}+a];}}m=nm;}}}}}}{barrier}}}if(l==0){{for(size_t a=0;a<{dim};a++){{b4[qb+a]=den>0.0f?u[a]/den:0.0f;}}}}",
		dim = s.dim,
		heads = s.heads,
		ratio = s.heads / s.kv_heads,
		keys = s.keys,
		kv = s.kv_heads
	);
	Ok(Kernel {
		scalars: vec![],
		source: header(d, 5, &body),
		entry: "chibi".into(),
		lengths: vec![
			s.queries * s.heads * s.dim,
			s.keys * s.kv_heads * s.dim,
			s.keys * s.kv_heads * s.dim,
			s.keys,
			s.queries * s.heads * s.dim,
		],
		output: 4,
		groups: s.queries * s.heads,
		local: 32,
		shared_bytes: bytes,
	})
}
#[derive(Debug, PartialEq, Eq)]
pub enum AttentionPath {
	Online,
	Ordinary(&'static str),
}
pub fn select(
	s: AttentionShape,
	requires_grad: bool,
	local_limit: usize,
	tile: usize,
) -> AttentionPath {
	if requires_grad {
		return AttentionPath::Ordinary("training requires ordinary backward");
	}
	if tile == 0
		|| s.dim == 0
		|| s.heads == 0
		|| s.kv_heads == 0
		|| !s.heads.is_multiple_of(s.kv_heads)
	{
		return AttentionPath::Ordinary("unsupported shape");
	}
	if tile
		.checked_mul(s.dim)
		.and_then(|n| n.checked_mul(8))
		.is_none_or(|n| n > local_limit)
	{
		return AttentionPath::Ordinary("local memory limit");
	}
	AttentionPath::Online
}
pub fn graph_reference(
	g: &mut crate::Graph,
	q: usize,
	k: usize,
	v: usize,
	s: AttentionShape,
	valid: &[bool],
) -> crate::Result<usize> {
	use crate::{Binary, Reduce, Unary};
	if g.shape(q) != [s.queries, s.heads, s.dim]
		|| g.shape(k) != [s.keys, s.kv_heads, s.dim]
		|| g.shape(v) != g.shape(k)
		|| valid.len() != s.keys
	{
		return Err("ordinary attention graph shape".into());
	}
	let mut times = vec![];
	for t in 0..s.queries {
		let mut heads = vec![];
		for h in 0..s.heads {
			let allowed = (0..s.keys)
				.map(|j| s.allowed(t, j, valid))
				.collect::<Vec<_>>();
			if !allowed.iter().any(|&v| v) {
				heads.push(g.constant(0., vec![s.dim])?);
				continue;
			}
			let kh = h / (s.heads / s.kv_heads);
			let qq = g.shrink(q, vec![(t, t + 1), (h, h + 1), (0, s.dim)])?;
			let qq = g.reshape(qq, vec![1, s.dim])?;
			let kk = g.shrink(k, vec![(0, s.keys), (kh, kh + 1), (0, s.dim)])?;
			let kk = g.reshape(kk, vec![s.keys, s.dim])?;
			let vv = g.shrink(v, vec![(0, s.keys), (kh, kh + 1), (0, s.dim)])?;
			let vv = g.reshape(vv, vec![s.keys, s.dim])?;
			let mut mask = vec![];
			for &ok in &allowed {
				mask.push(g.constant(if ok { 1. } else { 0. }, vec![])?);
			}
			let mask = g.stack(mask, 0)?;
			let mask = g.reshape(mask, vec![s.keys, 1])?;
			let zero = g.constant(0., vec![])?;
			let kk = g.where_(mask, kk, zero)?;
			let vv = g.where_(mask, vv, zero)?;
			let kt = g.permute(kk, vec![1, 0])?;
			let score = g.matmul(qq, kt)?;
			let scale = g.constant(1. / (s.dim as f32).sqrt(), vec![])?;
			let score = g.binary(Binary::Mul, score, scale)?;
			let mask = g.reshape(mask, vec![1, s.keys])?;
			let neg = g.constant(f32::NEG_INFINITY, vec![])?;
			let score = g.where_(mask, score, neg)?;
			let max = g.reduce(score, Reduce::Max, vec![1], true)?;
			let max = g.stop_gradient(max)?;
			let shifted = g.binary(Binary::Sub, score, max)?;
			let exp = g.unary(Unary::Exp, shifted)?;
			let sum = g.reduce(exp, Reduce::Sum, vec![1], true)?;
			let probs = g.binary(Binary::Div, exp, sum)?;
			let out = g.matmul(probs, vv)?;
			heads.push(g.reshape(out, vec![s.dim])?);
		}
		times.push(g.stack(heads, 0)?);
	}
	if s.queries == 0 {
		return g.constant(0., vec![0, s.heads, s.dim]);
	}
	g.stack(times, 0)
}
pub struct FlashPlan {
	pub source: String,
	shape: AttentionShape,
	tile: usize,
	library: crate::runtime::NativeLibrary,
}
impl FlashPlan {
	pub fn tile(&self) -> usize {
		self.tile
	}
	pub fn compile(s: AttentionShape, tile: usize) -> Result<Self, String> {
		if tile == 0
			|| s.dim == 0
			|| s.heads == 0
			|| s.kv_heads == 0
			|| !s.heads.is_multiple_of(s.kv_heads)
		{
			return Err("flash shape/tile".into());
		}
		let causal = if s.causal {
			format!("j<={}+t", s.query_start)
		} else {
			"1".into()
		};
		let source = format!(
			"{}int chibi_entry(void*const*buffers,const uint64_t*s){{const float*q=buffers[0],*k=buffers[1],*v=buffers[2],*valid=buffers[3];float*out=buffers[4];for(size_t t=0;t<{queries};t++){{for(size_t h=0;h<{heads};h++){{size_t qb=(t*{heads}+h)*{dim},kh=h/{ratio};float m=-INFINITY,l=0.0f,u[{dim}]={{0}};for(size_t start=0;start<{keys};start+={tile}){{size_t end={keys}-start<{tile}?{keys}:start+{tile};float tm=-INFINITY,scores[{tile}];for(size_t j=start;j<end;j++){{scores[j-start]=-INFINITY;if(valid[j]!=0.0f&&({causal})){{float score=0.0f;for(size_t d=0;d<{dim};d++){{score+=q[qb+d]*k[(j*{kv}+kh)*{dim}+d];}}score/=sqrtf({dim}.0f);scores[j-start]=score;tm=fmaxf(tm,score);}}}}if(tm==-INFINITY)continue;float nm=fmaxf(m,tm),alpha=l==0.0f?0.0f:expf(m-nm);l*=alpha;for(size_t d=0;d<{dim};d++){{u[d]*=alpha;}}for(size_t j=start;j<end;j++){{if(valid[j]!=0.0f&&({causal})){{float p=expf(scores[j-start]-nm);l+=p;for(size_t d=0;d<{dim};d++){{u[d]+=p*v[(j*{kv}+kh)*{dim}+d];}}}}}}m=nm;}}for(size_t d=0;d<{dim};d++){{out[qb+d]=l>0.0f?u[d]/l:0.0f;}}}}}}return 0;}}",
			crate::graph::C_HELPERS,
			queries = s.queries,
			heads = s.heads,
			dim = s.dim,
			ratio = s.heads / s.kv_heads,
			keys = s.keys,
			kv = s.kv_heads
		);
		let library = unsafe { crate::runtime::NativeLibrary::compile(&source, "chibi_entry") }?;
		Ok(Self {
			source,
			shape: s,
			tile,
			library,
		})
	}
	pub fn run(&self, q: &[f32], k: &[f32], v: &[f32], valid: &[bool]) -> Result<Vec<f32>, String> {
		self.shape.validate(q, k, v, valid)?;
		let mask = valid
			.iter()
			.map(|&v| if v { 1f32 } else { 0f32 })
			.collect::<Vec<_>>();
		let mut output = vec![0.; q.len()];
		let pointers = [
			q.as_ptr().cast_mut().cast(),
			k.as_ptr().cast_mut().cast(),
			v.as_ptr().cast_mut().cast(),
			mask.as_ptr().cast_mut().cast(),
			output.as_mut_ptr().cast(),
		];
		type Entry = unsafe extern "C" fn(*const *mut std::ffi::c_void, *const u64) -> i32;
		let entry: Entry = unsafe { std::mem::transmute(self.library.symbol()) };
		let status = unsafe { entry(pointers.as_ptr(), [0u64].as_ptr()) };
		if status != 0 {
			return Err(format!("flash native status {status}"));
		}
		Ok(output)
	}
}
#[derive(Clone, Copy, Debug)]
pub struct Boundary {
	pub q: usize,
	pub k: usize,
	pub v: usize,
	pub output: usize,
	pub shape: AttentionShape,
	pub tile: usize,
}
pub fn compile_boundaries(
	g: &crate::Graph,
	outputs: &[usize],
	boundaries: &[Boundary],
	requires_grad: bool,
) -> crate::Result<(crate::Plan, String)> {
	use std::collections::HashSet;
	if requires_grad || boundaries.is_empty() {
		return Ok((
			g.compile(outputs)?,
			"ordinary attention (training/no boundaries)".into(),
		));
	}
	let reachable = g.topology(outputs)?.into_iter().collect::<HashSet<_>>();
	let mut skipped = HashSet::new();
	let mut replacements = std::collections::HashMap::new();
	for boundary in boundaries {
		if !reachable.contains(&boundary.output) {
			continue;
		}
		if [boundary.q, boundary.k, boundary.v, boundary.output]
			.iter()
			.any(|&id| id >= g.nodes.len())
		{
			return Err("attention boundary node".into());
		}
		if g.shape(boundary.q)
			!= [
				boundary.shape.queries,
				boundary.shape.heads,
				boundary.shape.dim,
			] || g.shape(boundary.k)
			!= [
				boundary.shape.keys,
				boundary.shape.kv_heads,
				boundary.shape.dim,
			] || g.shape(boundary.v) != g.shape(boundary.k)
			|| g.shape(boundary.output) != g.shape(boundary.q)
		{
			return Err("attention boundary shape".into());
		}
		if !matches!(g.nodes[boundary.output].op, crate::Op::Stage { .. }) {
			return Ok((
				g.compile(outputs)?,
				"ordinary attention (unstaged output)".into(),
			));
		}
		if ![boundary.q, boundary.k, boundary.v].iter().all(|&id| {
			matches!(
				g.nodes[id].op,
				crate::Op::Stage { .. } | crate::Op::Input { .. }
			)
		}) {
			return Ok((
				g.compile(outputs)?,
				"ordinary attention (unstaged QKV)".into(),
			));
		}
		let before = g
			.topology(&[boundary.q, boundary.k, boundary.v])?
			.into_iter()
			.collect::<HashSet<_>>();
		let inside = g
			.topology(&[boundary.output])?
			.into_iter()
			.filter(|id| !before.contains(id))
			.collect::<HashSet<_>>();
		skipped.extend(boundary_skips(
			g,
			outputs,
			&reachable,
			&inside,
			boundary.output,
		)?);
		let plan = FlashPlan::compile(boundary.shape, boundary.tile)?;
		let begin = plan
			.source
			.find("int chibi_entry")
			.ok_or("native attention entry")?;
		let begin = begin + plan.source[begin..].find('{').unwrap() + 1;
		let end = plan
			.source
			.rfind("return 0;")
			.ok_or("native attention return")?;
		let mut body = plan.source[begin..end]
			.replace(",*valid=buffers[3]", "")
			.replace("valid[j]!=0.0f", "1");
		for (index, id) in [
			(0, boundary.q),
			(1, boundary.k),
			(2, boundary.v),
			(4, boundary.output),
		] {
			body = body.replace(&format!("buffers[{index}]"), &format!("binding_{id}_"));
		}
		for id in [boundary.q, boundary.k, boundary.v, boundary.output] {
			body = body.replace(&format!("binding_{id}_"), &format!("buffers[{id}]"));
		}
		replacements.insert(boundary.output, format!("{{{body}}}\n"));
	}
	let kernels = crate::graph::kernel_bodies(g, outputs, true)?;
	let mut source = crate::graph::C_HELPERS.to_string();
	source += "int chibi_entry(void*const*buffers,const uint64_t*s){\n";
	for id in g.topology(outputs)? {
		source += &format!("float*b{id}=(float*)buffers[{id}];\n");
	}
	for kernel in kernels {
		if skipped.contains(&kernel.output) {
			continue;
		}
		source += replacements.get(&kernel.output).unwrap_or(&kernel.body);
	}
	source += "return 0;\n}\n";
	let plan = crate::Plan::compile_source_specialized(g, outputs, source)?;
	Ok((
		plan,
		format!(
			"online attention boundaries={} pruned kernels={}",
			replacements.len(),
			skipped.len()
		),
	))
}
#[allow(clippy::too_many_arguments)]
pub fn register(
	g: &mut crate::Graph,
	q: usize,
	k: usize,
	v: usize,
	start: usize,
	causal: bool,
	tile: usize,
	boundaries: &mut Vec<Boundary>,
) -> crate::Result<usize> {
	let q = g.stage(q)?;
	let k = g.stage(k)?;
	let v = g.stage(v)?;
	let sq = g.shape(q).to_vec();
	let sk = g.shape(k).to_vec();
	if sq.len() != 3 || sk.len() != 3 {
		return Err("attention rank".into());
	}
	let shape = AttentionShape {
		queries: sq[0],
		keys: sk[0],
		heads: sq[1],
		kv_heads: sk[1],
		dim: sq[2],
		query_start: start,
		causal,
	};
	let out = graph_reference(g, q, k, v, shape, &vec![true; shape.keys])?;
	let output = g.stage(out)?;
	boundaries.push(Boundary {
		q,
		k,
		v,
		output,
		shape,
		tile,
	});
	Ok(output)
}

pub(crate) fn boundary_skips(
	g: &crate::Graph,
	outputs: &[usize],
	reachable: &std::collections::HashSet<usize>,
	inside: &std::collections::HashSet<usize>,
	output: usize,
) -> crate::Result<std::collections::HashSet<usize>> {
	let mut kept = std::collections::HashSet::new();
	for &id in outputs {
		if inside.contains(&id) && id != output {
			kept.extend(g.topology(&[id])?);
		}
	}
	for &user in reachable {
		if !inside.contains(&user) {
			for source in g.nodes[user].op.sources() {
				if inside.contains(&source) && source != output {
					kept.extend(g.topology(&[source])?);
				}
			}
		}
	}
	Ok(inside
		.iter()
		.copied()
		.filter(|&id| id != output && !kept.contains(&id))
		.collect())
}
```

```rust file=src/gpu_graph.rs mode=append
impl GpuPlan {
	pub fn compile_flash(
		graph: &Graph,
		outputs: &[usize],
		backend: Backend,
		boundaries: &[crate::flash::Boundary],
		requires_grad: bool,
		local_bytes: usize,
	) -> Result<Self> {
		use std::collections::{HashMap, HashSet};
		if requires_grad {
			return Self::compile(graph, outputs, backend, 64);
		}
		let reachable = graph.topology(outputs)?.into_iter().collect::<HashSet<_>>();
		let mut skipped = HashSet::new();
		let mut replacements = HashMap::new();
		for boundary in boundaries {
			if [boundary.q, boundary.k, boundary.v, boundary.output]
				.iter()
				.any(|&id| id >= graph.nodes.len())
			{
				return Err("GPU attention boundary node".into());
			}
			if graph.shape(boundary.q)
				!= [
					boundary.shape.queries,
					boundary.shape.heads,
					boundary.shape.dim,
				] || graph.shape(boundary.k)
				!= [
					boundary.shape.keys,
					boundary.shape.kv_heads,
					boundary.shape.dim,
				] || graph.shape(boundary.v) != graph.shape(boundary.k)
			{
				return Err("GPU attention boundary shape".into());
			}
			if graph.shape(boundary.output) != graph.shape(boundary.q) {
				return Err("GPU attention boundary output shape".into());
			}
			if !reachable.contains(&boundary.output) {
				continue;
			}
			if crate::flash::select(boundary.shape, false, local_bytes, boundary.tile)
				!= crate::flash::AttentionPath::Online
			{
				return Self::compile(graph, outputs, backend, 64);
			}
			if !matches!(graph.nodes[boundary.output].op, Op::Stage { .. }) {
				return Self::compile(graph, outputs, backend, 64);
			}
			if ![boundary.q, boundary.k, boundary.v]
				.iter()
				.all(|&id| matches!(graph.nodes[id].op, Op::Stage { .. } | Op::Input { .. }))
			{
				return Self::compile(graph, outputs, backend, 64);
			}
			let before = graph
				.topology(&[boundary.q, boundary.k, boundary.v])?
				.into_iter()
				.collect::<HashSet<_>>();
			let inside = graph
				.topology(&[boundary.output])?
				.into_iter()
				.filter(|id| !before.contains(id))
				.collect::<HashSet<_>>();
			skipped.extend(crate::flash::boundary_skips(
				graph,
				outputs,
				&reachable,
				&inside,
				boundary.output,
			)?);
			let kernel = crate::flash::shader(backend.dialect(), boundary.shape, boundary.tile)?;
			replacements.insert(
				boundary.output,
				(
					kernel,
					vec![
						boundary.q,
						boundary.k,
						boundary.v,
						usize::MAX,
						boundary.output,
					],
				),
			);
		}
		let ordinary = lower(graph, outputs, backend.dialect(), 64)?;
		let mut lowered = vec![];
		for (kernel, slots) in ordinary {
			let id = *slots.last().ok_or("attention output")?;
			if skipped.contains(&id) {
				continue;
			}
			lowered.push(replacements.remove(&id).unwrap_or((kernel, slots)));
		}
		Self::from_lowered(graph, outputs, backend, lowered, true)
	}
}
```

```rust file=tests/flash.rs mode=write
use chibigrad::flash::*;
fn compare(a: &[f32], b: &[f64]) {
	assert_eq!(a.len(), b.len());
	for (a, b) in a.iter().zip(b) {
		assert!(
			a.is_finite() && (*a as f64 - b).abs() <= 1e-4 + 1e-4 * b.abs(),
			"{a} {b}"
		);
	}
}
#[test]
fn online_boundary_masks_gqa() {
	for keys in [0, 1, 2, 3, 7, 8, 9] {
		for ratio in [1, 2] {
			let s = AttentionShape {
				queries: 3,
				keys,
				heads: 2,
				kv_heads: 2 / ratio,
				dim: 3,
				query_start: 5,
				causal: true,
			};
			let q = (0..s.queries * s.heads * s.dim)
				.map(|i| (i as f32 - 7.) / 8.)
				.collect::<Vec<_>>();
			let mut k = (0..keys * s.kv_heads * s.dim)
				.map(|i| (i as f32 - 9.) / 9.)
				.collect::<Vec<_>>();
			let mut v = k.clone();
			let mut valid = vec![true; keys];
			if keys > 0 {
				valid[keys - 1] = false;
				for i in (keys - 1) * s.kv_heads * s.dim..k.len() {
					k[i] = f32::NAN;
					v[i] = f32::NAN;
				}
			}
			for tile in [1, 2, 3, 8] {
				let r = reference(s, &q, &k, &v, &valid).unwrap();
				compare(&online(s, &q, &k, &v, &valid, tile).unwrap(), &r.output);
			}
		}
	}
}
#[test]
fn all_masked_and_late_max() {
	let s = AttentionShape {
		queries: 1,
		keys: 3,
		heads: 1,
		kv_heads: 1,
		dim: 1,
		query_start: 5,
		causal: false,
	};
	compare(
		&online(s, &[1.], &[-100., 0., 100.], &[1., 2., 3.], &[true; 3], 1).unwrap(),
		&[3.],
	);
	assert_eq!(
		online(s, &[1.], &[f32::NAN; 3], &[f32::NAN; 3], &[false; 3], 2).unwrap(),
		[0.]
	);
	assert_eq!(
		select(s, true, 32768, 8),
		AttentionPath::Ordinary("training requires ordinary backward")
	);
}
#[test]
fn native_c_and_composite_graph() {
	for tile in [1, 2, 3, 8] {
		let s = AttentionShape {
			queries: 2,
			keys: 5,
			heads: 2,
			kv_heads: 1,
			dim: 3,
			query_start: 5,
			causal: true,
		};
		let q = vec![0.25; s.queries * s.heads * s.dim];
		let mut k = vec![0.5; s.keys * s.kv_heads * s.dim];
		let mut v = (0..k.len()).map(|i| i as f32 / 7.).collect::<Vec<_>>();
		let valid = [true, true, false, true, true];
		for i in 6..9 {
			k[i] = f32::NAN;
			v[i] = f32::NAN;
		}
		let r = reference(s, &q, &k, &v, &valid).unwrap();
		let native = FlashPlan::compile(s, tile)
			.unwrap()
			.run(&q, &k, &v, &valid)
			.unwrap();
		compare(&native, &r.output);
		let mut g = chibigrad::Graph::default();
		let qi = g.input(0, vec![s.queries, s.heads, s.dim]).unwrap();
		let ki = g.input(1, vec![s.keys, s.kv_heads, s.dim]).unwrap();
		let vi = g.input(2, vec![s.keys, s.kv_heads, s.dim]).unwrap();
		let out = graph_reference(&mut g, qi, ki, vi, s, &valid).unwrap();
		let inputs = [
			chibigrad::Buffer::new(g.shape(qi).to_vec(), q).unwrap(),
			chibigrad::Buffer::new(g.shape(ki).to_vec(), k).unwrap(),
			chibigrad::Buffer::new(g.shape(vi).to_vec(), v).unwrap(),
		];
		compare(
			&g.compile(&[out]).unwrap().run(&inputs).unwrap()[0].data,
			&r.output,
		);
	}
}
#[test]
fn model_boundary_compiler_replaces_actual_graph() {
	let s = AttentionShape {
		queries: 2,
		keys: 3,
		heads: 2,
		kv_heads: 1,
		dim: 2,
		query_start: 5,
		causal: true,
	};
	let mut g = chibigrad::Graph::default();
	let q = g.input(0, vec![2, 2, 2]).unwrap();
	let k = g.input(1, vec![3, 1, 2]).unwrap();
	let v = g.input(2, vec![3, 1, 2]).unwrap();
	let out = graph_reference(&mut g, q, k, v, s, &[true; 3]).unwrap();
	let out = g.stage(out).unwrap();
	let boundary = Boundary {
		q,
		k,
		v,
		output: out,
		shape: s,
		tile: 2,
	};
	let (p, path) = compile_boundaries(&g, &[out], &[boundary], false).unwrap();
	assert!(path.contains("boundaries=1"));
	let inputs = [
		chibigrad::Buffer::new(vec![2, 2, 2], vec![1.; 8]).unwrap(),
		chibigrad::Buffer::new(vec![3, 1, 2], vec![1.; 6]).unwrap(),
		chibigrad::Buffer::new(vec![3, 1, 2], (0..6).map(|i| i as f32).collect()).unwrap(),
	];
	let expected = g.compile(&[out]).unwrap().run(&inputs).unwrap();
	let actual = p.run(&inputs).unwrap();
	assert_eq!(actual, expected);
	assert!(
		compile_boundaries(&g, &[out], &[boundary], true)
			.unwrap()
			.1
			.contains("ordinary")
	);
}
#[test]
fn gpu_source_has_uniform_barriers() {
	use chibigrad::gpu::Dialect;
	for keys in [0, 1, 7, 8, 9] {
		let s = AttentionShape {
			queries: 3,
			keys,
			heads: 2,
			kv_heads: 1,
			dim: 3,
			query_start: 5,
			causal: true,
		};
		let kernel = shader(Dialect::OpenCl, s, 8).unwrap();
		assert_eq!(
			kernel
				.source
				.matches("barrier(CLK_LOCAL_MEM_FENCE)")
				.count(),
			2
		);
		let path = std::env::temp_dir().join(format!("flash-{}-{keys}.cl", std::process::id()));
		std::fs::write(&path, kernel.source).unwrap();
		let result = std::process::Command::new("clang")
			.args(["-x", "cl", "-cl-std=CL1.2", "-fsyntax-only"])
			.arg(&path)
			.output()
			.unwrap();
		let _ = std::fs::remove_file(path);
		assert!(
			result.status.success(),
			"{}",
			String::from_utf8_lossy(&result.stderr)
		);
	}
}
```

```rust file=tests/flash_hardware.rs mode=write
use chibigrad::{
	Buffer, Graph,
	flash::{self, AttentionShape},
	gpu::{Dialect, Kernel},
	gpu_graph::{Backend, GpuPlan},
};
fn cases(
	d: Dialect,
	backend: Backend,
	run: impl Fn(&Kernel, &[Vec<f32>]) -> Result<Vec<f32>, String>,
) {
	for keys in [1, 2, 3, 7, 8, 9] {
		let s = AttentionShape {
			queries: 3,
			keys,
			heads: 2,
			kv_heads: 1,
			dim: 3,
			query_start: 5,
			causal: true,
		};
		let q = vec![0.25; 18];
		let mut k = vec![0.5; keys * 3];
		let mut v = (0..keys * 3).map(|i| i as f32 / 7.).collect::<Vec<_>>();
		let mut valid = vec![true; keys];
		valid[keys - 1] = false;
		for i in (keys - 1) * 3..k.len() {
			k[i] = f32::NAN;
			v[i] = f32::NAN;
		}
		let reference = flash::reference(s, &q, &k, &v, &valid).unwrap();
		for tile in [1, 2, 3, 8] {
			let kernel = flash::shader(d, s, tile).unwrap();
			let mask = valid.iter().map(|&v| if v { 1. } else { 0. }).collect();
			let output = run(
				&kernel,
				&[q.clone(), k.clone(), v.clone(), mask, vec![0.; 18]],
			)
			.unwrap();
			assert!(
				chibigrad::flash::select(s, false, 32768, tile) == flash::AttentionPath::Online
			);
			for (&a, &b) in output.iter().zip(&reference.output) {
				assert!(a.is_finite() && (a as f64 - b).abs() <= 1e-4 + 1e-4 * b.abs());
			}
		}
		let mut g = Graph::default();
		let qi = g.input(0, vec![3, 2, 3]).unwrap();
		let ki = g.input(1, vec![keys, 1, 3]).unwrap();
		let vi = g.input(2, vec![keys, 1, 3]).unwrap();
		let mut boundaries = vec![];
		let out = flash::register(&mut g, qi, ki, vi, 5, true, 2, &mut boundaries).unwrap();
		let inputs = [
			Buffer::new(vec![3, 2, 3], q).unwrap(),
			Buffer::new(vec![keys, 1, 3], vec![0.5; keys * 3]).unwrap(),
			Buffer::new(vec![keys, 1, 3], vec![1.; keys * 3]).unwrap(),
		];
		let p = GpuPlan::compile_flash(&g, &[out], backend, &boundaries, false, 32768).unwrap();
		let expected = g.interpret(&[out], &inputs).unwrap();
		let actual = p.run(&inputs).unwrap();
		for (&a, &b) in actual[0].data.iter().zip(&expected[0].data) {
			assert!((a - b).abs() <= 1e-4 + 1e-4 * b.abs());
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

```sh check requires=metal
cargo test --test flash_hardware --features metal metal -- --ignored --exact
```

```sh check requires=opencl
cargo test --test flash_hardware --features opencl opencl -- --ignored --exact
```

```rust file=tests/qwen_flash.rs mode=write
use chibigrad::{
	Graph,
	flash::{compile_boundaries, register},
	nn::qwen::{
		QwenConfig, logits_all_with_attention, parameter_registry, register_graph, token_buffer,
	},
	nn::qwen::initialize,
};
#[test]
fn same_model_logits_and_cache() {
	let c = QwenConfig::tiny();
	let registry = parameter_registry(&c).unwrap();
	let mut g = Graph::default();
	let parameters = register_graph(&mut g, &registry).unwrap();
	let mut inputs = initialize(&c, 7)
		.unwrap()
		.iter()
		.map(|w| w.buffer().unwrap())
		.collect::<Vec<_>>();
	let tokens = g.input(inputs.len(), vec![3]).unwrap();
	inputs.push(token_buffer(&[1, 2, 3], &c).unwrap());
	let mut boundaries = vec![];
	let forward = logits_all_with_attention(
		&mut g,
		&c,
		&parameters,
		tokens,
		0,
		None,
		&mut |g, q, k, v, start, causal| register(g, q, k, v, start, causal, 2, &mut boundaries),
	)
	.unwrap();
	let outputs = [forward.logits];
	let expected = g.compile(&outputs).unwrap().run(&inputs).unwrap();
	let (plan, path) = compile_boundaries(&g, &outputs, &boundaries, false).unwrap();
	assert!(path.contains("boundaries=1"), "{path}");
	let actual = plan.run(&inputs).unwrap();
	for (a, b) in actual[0].data.iter().zip(&expected[0].data) {
		assert!(a.is_finite() && (a - b).abs() <= 1e-4 + 1e-4 * b.abs());
	}
	assert!(
		compile_boundaries(&g, &outputs, &boundaries, true)
			.unwrap()
			.1
			.contains("ordinary")
	);
}
#[test]
fn training_uses_composite_backward_and_update() {
	use chibigrad::{Buffer, Reduce, autodiff::gradient, flash};
	let mut g = Graph::default();
	let q = g.input(0, vec![2, 2, 2]).unwrap();
	let k = g.input(1, vec![3, 1, 2]).unwrap();
	let v = g.input(2, vec![3, 1, 2]).unwrap();
	let mut boundaries = vec![];
	let out = flash::register(&mut g, q, k, v, 0, true, 2, &mut boundaries).unwrap();
	let loss = g.reduce(out, Reduce::Sum, vec![0, 1, 2], false).unwrap();
	let gradients = gradient(&mut g, loss, &[q, k, v], None).unwrap();
	let mut outputs = vec![loss];
	outputs.extend(gradients);
	let (plan, path) = compile_boundaries(&g, &outputs, &boundaries, true).unwrap();
	assert!(path.contains("ordinary"));
	let ordinary = g.compile(&outputs).unwrap();
	let mut inputs = [
		Buffer::new(vec![2, 2, 2], vec![0.25; 8]).unwrap(),
		Buffer::new(vec![3, 1, 2], (0..6).map(|i| i as f32 / 9.).collect()).unwrap(),
		Buffer::new(vec![3, 1, 2], (0..6).map(|i| i as f32 / 7.).collect()).unwrap(),
	];
	let old = plan.run(&inputs).unwrap();
	assert_eq!(old, ordinary.run(&inputs).unwrap());
	for (x, gradient) in inputs[2].data.iter_mut().zip(&old[3].data) {
		*x -= 0.001 * gradient;
	}
	let new = plan.run(&inputs).unwrap();
	assert!(new[0].data[0] < old[0].data[0]);
	assert_eq!(new, ordinary.run(&inputs).unwrap());
}
#[test]
fn online_cache_valid_prefix_matches_ordinary() {
	use chibigrad::{Buffer, nn::qwen::KvState};
	let c = QwenConfig::tiny();
	let weights = initialize(&c, 11).unwrap();
	let mut cache = KvState::new(&c).unwrap();
	for tokens in [vec![1u32, 2], vec![3]] {
		let mut g = Graph::default();
		let registry = parameter_registry(&c).unwrap();
		let params = register_graph(&mut g, &registry).unwrap();
		let mut inputs = weights
			.iter()
			.map(|w| w.buffer().unwrap())
			.collect::<Vec<_>>();
		let ids = g.input(inputs.len(), vec![tokens.len()]).unwrap();
		inputs.push(token_buffer(&tokens, &c).unwrap());
		let mut cache_ids = vec![];
		for (k, v) in cache.keys.iter().zip(&cache.values) {
			let ki = g.input(inputs.len(), k.shape.clone()).unwrap();
			inputs.push(k.clone());
			let vi = g.input(inputs.len(), v.shape.clone()).unwrap();
			inputs.push(v.clone());
			cache_ids.push((ki, vi));
		}
		let mut boundaries = vec![];
		let f = logits_all_with_attention(
			&mut g,
			&c,
			&params,
			ids,
			cache.valid,
			Some(&cache_ids),
			&mut |g, q, k, v, start, causal| {
				register(g, q, k, v, start, causal, 2, &mut boundaries)
			},
		)
		.unwrap();
		let mut outputs = vec![f.logits];
		for (k, v) in f.cache_outputs {
			outputs.extend([k, v]);
		}
		let expected = g.compile(&outputs).unwrap().run(&inputs).unwrap();
		let (p, path) = compile_boundaries(&g, &outputs, &boundaries, false).unwrap();
		assert!(path.contains("boundaries=1"), "{path}");
		let actual = p.run(&inputs).unwrap();
		for (a, b) in actual.iter().zip(&expected) {
			for (&a, &b) in a.data.iter().zip(&b.data) {
				assert!(a.is_finite() && (a - b).abs() <= 1e-4 + 1e-4 * b.abs());
			}
		}
		cache.adopt(&actual[1..], tokens.len()).unwrap();
	}
	assert_eq!(cache.valid, 3);
	let _: Option<Buffer> = None;
}
```

```sh check
cargo test --test qwen_flash
```

```rust file=tests/book_23.rs mode=write
mod suite0 {
	use chibigrad::flash::*;
	pub(super) fn compare(a: &[f32], b: &[f64]) {
		assert_eq!(a.len(), b.len());
		for (a, b) in a.iter().zip(b) {
			assert!(
				a.is_finite() && (*a as f64 - b).abs() <= 1e-4 + 1e-4 * b.abs(),
				"{a} {b}"
			);
		}
	}
	#[test]
	pub(super) fn online_boundary_masks_gqa() {
		for keys in [0, 1, 2, 3, 7, 8, 9] {
			for ratio in [1, 2] {
				let s = AttentionShape {
					queries: 3,
					keys,
					heads: 2,
					kv_heads: 2 / ratio,
					dim: 3,
					query_start: 5,
					causal: true,
				};
				let q = (0..s.queries * s.heads * s.dim)
					.map(|i| (i as f32 - 7.) / 8.)
					.collect::<Vec<_>>();
				let mut k = (0..keys * s.kv_heads * s.dim)
					.map(|i| (i as f32 - 9.) / 9.)
					.collect::<Vec<_>>();
				let mut v = k.clone();
				let mut valid = vec![true; keys];
				if keys > 0 {
					valid[keys - 1] = false;
					for i in (keys - 1) * s.kv_heads * s.dim..k.len() {
						k[i] = f32::NAN;
						v[i] = f32::NAN;
					}
				}
				for tile in [1, 2, 3, 8] {
					let r = reference(s, &q, &k, &v, &valid).unwrap();
					compare(&online(s, &q, &k, &v, &valid, tile).unwrap(), &r.output);
				}
			}
		}
	}
	#[test]
	pub(super) fn all_masked_and_late_max() {
		let s = AttentionShape {
			queries: 1,
			keys: 3,
			heads: 1,
			kv_heads: 1,
			dim: 1,
			query_start: 5,
			causal: false,
		};
		compare(
			&online(s, &[1.], &[-100., 0., 100.], &[1., 2., 3.], &[true; 3], 1).unwrap(),
			&[3.],
		);
		assert_eq!(
			online(s, &[1.], &[f32::NAN; 3], &[f32::NAN; 3], &[false; 3], 2).unwrap(),
			[0.]
		);
		assert_eq!(
			select(s, true, 32768, 8),
			AttentionPath::Ordinary("training requires ordinary backward")
		);
	}
	#[test]
	pub(super) fn native_c_and_composite_graph() {
		for tile in [1, 2, 3, 8] {
			let s = AttentionShape {
				queries: 2,
				keys: 5,
				heads: 2,
				kv_heads: 1,
				dim: 3,
				query_start: 5,
				causal: true,
			};
			let q = vec![0.25; s.queries * s.heads * s.dim];
			let mut k = vec![0.5; s.keys * s.kv_heads * s.dim];
			let mut v = (0..k.len()).map(|i| i as f32 / 7.).collect::<Vec<_>>();
			let valid = [true, true, false, true, true];
			for i in 6..9 {
				k[i] = f32::NAN;
				v[i] = f32::NAN;
			}
			let r = reference(s, &q, &k, &v, &valid).unwrap();
			let native = FlashPlan::compile(s, tile)
				.unwrap()
				.run(&q, &k, &v, &valid)
				.unwrap();
			compare(&native, &r.output);
			let mut g = chibigrad::Graph::default();
			let qi = g.input(0, vec![s.queries, s.heads, s.dim]).unwrap();
			let ki = g.input(1, vec![s.keys, s.kv_heads, s.dim]).unwrap();
			let vi = g.input(2, vec![s.keys, s.kv_heads, s.dim]).unwrap();
			let out = graph_reference(&mut g, qi, ki, vi, s, &valid).unwrap();
			let inputs = [
				chibigrad::Buffer::new(g.shape(qi).to_vec(), q).unwrap(),
				chibigrad::Buffer::new(g.shape(ki).to_vec(), k).unwrap(),
				chibigrad::Buffer::new(g.shape(vi).to_vec(), v).unwrap(),
			];
			compare(
				&g.compile(&[out]).unwrap().run(&inputs).unwrap()[0].data,
				&r.output,
			);
		}
	}
	#[test]
	pub(super) fn model_boundary_compiler_replaces_actual_graph() {
		let s = AttentionShape {
			queries: 2,
			keys: 3,
			heads: 2,
			kv_heads: 1,
			dim: 2,
			query_start: 5,
			causal: true,
		};
		let mut g = chibigrad::Graph::default();
		let q = g.input(0, vec![2, 2, 2]).unwrap();
		let k = g.input(1, vec![3, 1, 2]).unwrap();
		let v = g.input(2, vec![3, 1, 2]).unwrap();
		let out = graph_reference(&mut g, q, k, v, s, &[true; 3]).unwrap();
		let out = g.stage(out).unwrap();
		let boundary = Boundary {
			q,
			k,
			v,
			output: out,
			shape: s,
			tile: 2,
		};
		let (p, path) = compile_boundaries(&g, &[out], &[boundary], false).unwrap();
		assert!(path.contains("boundaries=1"));
		let inputs = [
			chibigrad::Buffer::new(vec![2, 2, 2], vec![1.; 8]).unwrap(),
			chibigrad::Buffer::new(vec![3, 1, 2], vec![1.; 6]).unwrap(),
			chibigrad::Buffer::new(vec![3, 1, 2], (0..6).map(|i| i as f32).collect()).unwrap(),
		];
		let expected = g.compile(&[out]).unwrap().run(&inputs).unwrap();
		let actual = p.run(&inputs).unwrap();
		assert_eq!(actual, expected);
		assert!(
			compile_boundaries(&g, &[out], &[boundary], true)
				.unwrap()
				.1
				.contains("ordinary")
		);
	}
	#[test]
	pub(super) fn gpu_source_has_uniform_barriers() {
		use chibigrad::gpu::Dialect;
		for keys in [0, 1, 7, 8, 9] {
			let s = AttentionShape {
				queries: 3,
				keys,
				heads: 2,
				kv_heads: 1,
				dim: 3,
				query_start: 5,
				causal: true,
			};
			let kernel = shader(Dialect::OpenCl, s, 8).unwrap();
			assert_eq!(
				kernel
					.source
					.matches("barrier(CLK_LOCAL_MEM_FENCE)")
					.count(),
				2
			);
			let path = std::env::temp_dir().join(format!("flash-{}-{keys}.cl", std::process::id()));
			std::fs::write(&path, kernel.source).unwrap();
			let result = std::process::Command::new("clang")
				.args(["-x", "cl", "-cl-std=CL1.2", "-fsyntax-only"])
				.arg(&path)
				.output()
				.unwrap();
			let _ = std::fs::remove_file(path);
			assert!(
				result.status.success(),
				"{}",
				String::from_utf8_lossy(&result.stderr)
			);
		}
	}
}
```

## 通常版の score、mask 後の score、softmax、O を小入力で取り出す。

<!-- todo:23-001 -->

実装は[この章の `flash::reference、FlashPlan::compile、FlashPlan::run、flash::shader、flash::register、compile_boundaries`](#flash-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_23.rs mode=append
#[test]
fn todo_23_001() {
	suite0::native_c_and_composite_graph();
}
```

`src/flash.rs` の `pub fn reference(`の定義を次の全体で置き換えます。

```rust file="src/flash.rs" mode="replace" item="pub fn reference("
pub fn reference(
	s: AttentionShape,
	q: &[f32],
	k: &[f32],
	v: &[f32],
	valid: &[bool],
) -> Result<Reference, String> {
	s.validate(q, k, v, valid)?;
	let mut result = Reference {
		scores: vec![0.; s.queries * s.heads * s.keys],
		masked_scores: vec![f64::NEG_INFINITY; s.queries * s.heads * s.keys],
		probabilities: vec![0.; s.queries * s.heads * s.keys],
		output: vec![0.; q.len()],
	};
	let scale = 1. / (s.dim as f64).sqrt();
	for t in 0..s.queries {
		for h in 0..s.heads {
			let row = (t * s.heads + h) * s.keys;
			let qbase = (t * s.heads + h) * s.dim;
			let kh = h / (s.heads / s.kv_heads);
			let mut max = f64::NEG_INFINITY;
			for j in 0..s.keys {
				if !s.allowed(t, j, valid) {
					continue;
				}
				let base = (j * s.kv_heads + kh) * s.dim;
				let score = (0..s.dim)
					.map(|d| q[qbase + d] as f64 * k[base + d] as f64)
					.sum::<f64>() * scale;
				result.scores[row + j] = score;
				result.masked_scores[row + j] = score;
				max = max.max(score);
			}
			if max == f64::NEG_INFINITY {
				continue;
			}
			let mut sum = 0.;
			for j in 0..s.keys {
				if s.allowed(t, j, valid) {
					result.probabilities[row + j] = (result.masked_scores[row + j] - max).exp();
					sum += result.probabilities[row + j];
				}
			}
			for j in 0..s.keys {
				if s.allowed(t, j, valid) {
					let p = result.probabilities[row + j] / sum;
					result.probabilities[row + j] = p;
					for d in 0..s.dim {
						result.output[qbase + d] += p * v[(j * s.kv_heads + kh) * s.dim + d] as f64;
					}
				}
			}
		}
	}
	Ok(result)
}
```

```sh check
cargo test --test book_23 todo_23_001 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## head dimension と scale を検証し、shape 不一致を起動前に拒否する。

<!-- todo:23-002 -->

実装は[この章の `flash::reference、FlashPlan::compile、FlashPlan::run、flash::shader、flash::register、compile_boundaries`](#flash-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_23.rs mode=append
#[test]
fn todo_23_002() {
	suite0::online_boundary_masks_gqa();
}
```

`src/flash.rs` の `pub fn validate(`の定義を次の全体で置き換えます。

```rust file="src/flash.rs" mode="replace" item="pub fn validate("
	pub fn validate(&self, q: &[f32], k: &[f32], v: &[f32], valid: &[bool]) -> Result<(), String> {
		if self.dim == 0
			|| self.heads == 0
			|| self.kv_heads == 0
			|| !self.heads.is_multiple_of(self.kv_heads)
		{
			return Err("attention heads/dim".into());
		}
		let qn = self
			.queries
			.checked_mul(self.heads)
			.and_then(|n| n.checked_mul(self.dim))
			.ok_or("attention shape overflow")?;
		let kn = self
			.keys
			.checked_mul(self.kv_heads)
			.and_then(|n| n.checked_mul(self.dim))
			.ok_or("attention shape overflow")?;
		self.query_start
			.checked_add(self.queries)
			.ok_or("query position overflow")?;
		if q.len() != qn || k.len() != kn || v.len() != kn || valid.len() != self.keys {
			return Err("attention input shape".into());
		}
		Ok(())
	}
```

`src/flash.rs` の `pub fn reference(`の定義を次の全体で置き換えます。

```rust file="src/flash.rs" mode="replace" item="pub fn reference("
pub fn reference(
	s: AttentionShape,
	q: &[f32],
	k: &[f32],
	v: &[f32],
	valid: &[bool],
) -> Result<Reference, String> {
	s.validate(q, k, v, valid)?;
	let mut result = Reference {
		scores: vec![0.; s.queries * s.heads * s.keys],
		masked_scores: vec![f64::NEG_INFINITY; s.queries * s.heads * s.keys],
		probabilities: vec![0.; s.queries * s.heads * s.keys],
		output: vec![0.; q.len()],
	};
	let scale = 1. / (s.dim as f64).sqrt();
	for t in 0..s.queries {
		for h in 0..s.heads {
			let row = (t * s.heads + h) * s.keys;
			let qbase = (t * s.heads + h) * s.dim;
			let kh = h / (s.heads / s.kv_heads);
			let mut max = f64::NEG_INFINITY;
			for j in 0..s.keys {
				if !s.allowed(t, j, valid) {
					continue;
				}
				let base = (j * s.kv_heads + kh) * s.dim;
				let score = (0..s.dim)
					.map(|d| q[qbase + d] as f64 * k[base + d] as f64)
					.sum::<f64>() * scale;
				result.scores[row + j] = score;
				result.masked_scores[row + j] = score;
				max = max.max(score);
			}
			if max == f64::NEG_INFINITY {
				continue;
			}
			let mut sum = 0.;
			for j in 0..s.keys {
				if s.allowed(t, j, valid) {
					result.probabilities[row + j] = (result.masked_scores[row + j] - max).exp();
					sum += result.probabilities[row + j];
				}
			}
			for j in 0..s.keys {
				if s.allowed(t, j, valid) {
					let p = result.probabilities[row + j] / sum;
					result.probabilities[row + j] = p;
					for d in 0..s.dim {
						result.output[qbase + d] += p * v[(j * s.kv_heads + kh) * s.dim + d] as f64;
					}
				}
			}
		}
	}
	Ok(result)
}
```

```sh check
cargo test --test book_23 todo_23_002 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## causal、非 causal、padding mask、全 mask 行を別 fixture にする。

<!-- todo:23-003 -->

実装は[この章の `flash::reference、FlashPlan::compile、FlashPlan::run、flash::shader、flash::register、compile_boundaries`](#flash-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_23.rs mode=append
#[test]
fn todo_23_003() {
	suite0::all_masked_and_late_max();
}
```

```sh check
cargo test --test book_23 todo_23_003 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 位置0だけを前提にせず、query 開始位置5などの絶対位置も test に入れる。

<!-- todo:23-004 -->

実装は[この章の `flash::reference、FlashPlan::compile、FlashPlan::run、flash::shader、flash::register、compile_boundaries`](#flash-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_23.rs mode=append
#[test]
fn todo_23_004() {
	suite0::model_boundary_compiler_replaces_actual_graph();
}
```

```sh check
cargo test --test book_23 todo_23_004 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 全 mask 行を含む stable reference を Rust の scalar f64 と既存 f32 graph の両方で動かす。

<!-- todo:23-005 -->

実装は[この章の `flash::reference、FlashPlan::compile、FlashPlan::run、flash::shader、flash::register、compile_boundaries`](#flash-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_23.rs mode=append
#[test]
fn todo_23_005() {
	suite0::native_c_and_composite_graph();
}
```

`src/flash.rs` の `pub fn reference(`の定義を次の全体で置き換えます。

```rust file="src/flash.rs" mode="replace" item="pub fn reference("
pub fn reference(
	s: AttentionShape,
	q: &[f32],
	k: &[f32],
	v: &[f32],
	valid: &[bool],
) -> Result<Reference, String> {
	s.validate(q, k, v, valid)?;
	let mut result = Reference {
		scores: vec![0.; s.queries * s.heads * s.keys],
		masked_scores: vec![f64::NEG_INFINITY; s.queries * s.heads * s.keys],
		probabilities: vec![0.; s.queries * s.heads * s.keys],
		output: vec![0.; q.len()],
	};
	let scale = 1. / (s.dim as f64).sqrt();
	for t in 0..s.queries {
		for h in 0..s.heads {
			let row = (t * s.heads + h) * s.keys;
			let qbase = (t * s.heads + h) * s.dim;
			let kh = h / (s.heads / s.kv_heads);
			let mut max = f64::NEG_INFINITY;
			for j in 0..s.keys {
				if !s.allowed(t, j, valid) {
					continue;
				}
				let base = (j * s.kv_heads + kh) * s.dim;
				let score = (0..s.dim)
					.map(|d| q[qbase + d] as f64 * k[base + d] as f64)
					.sum::<f64>() * scale;
				result.scores[row + j] = score;
				result.masked_scores[row + j] = score;
				max = max.max(score);
			}
			if max == f64::NEG_INFINITY {
				continue;
			}
			let mut sum = 0.;
			for j in 0..s.keys {
				if s.allowed(t, j, valid) {
					result.probabilities[row + j] = (result.masked_scores[row + j] - max).exp();
					sum += result.probabilities[row + j];
				}
			}
			for j in 0..s.keys {
				if s.allowed(t, j, valid) {
					let p = result.probabilities[row + j] / sum;
					result.probabilities[row + j] = p;
					for d in 0..s.dim {
						result.output[qbase + d] += p * v[(j * s.kv_heads + kh) * s.dim + d] as f64;
					}
				}
			}
		}
	}
	Ok(result)
}
```

`src/flash.rs` の `pub fn graph_reference(`の定義を次の全体で置き換えます。

```rust file="src/flash.rs" mode="replace" item="pub fn graph_reference("
pub fn graph_reference(
	g: &mut crate::Graph,
	q: usize,
	k: usize,
	v: usize,
	s: AttentionShape,
	valid: &[bool],
) -> crate::Result<usize> {
	use crate::{Binary, Reduce, Unary};
	if g.shape(q) != [s.queries, s.heads, s.dim]
		|| g.shape(k) != [s.keys, s.kv_heads, s.dim]
		|| g.shape(v) != g.shape(k)
		|| valid.len() != s.keys
	{
		return Err("ordinary attention graph shape".into());
	}
	let mut times = vec![];
	for t in 0..s.queries {
		let mut heads = vec![];
		for h in 0..s.heads {
			let allowed = (0..s.keys)
				.map(|j| s.allowed(t, j, valid))
				.collect::<Vec<_>>();
			if !allowed.iter().any(|&v| v) {
				heads.push(g.constant(0., vec![s.dim])?);
				continue;
			}
			let kh = h / (s.heads / s.kv_heads);
			let qq = g.shrink(q, vec![(t, t + 1), (h, h + 1), (0, s.dim)])?;
			let qq = g.reshape(qq, vec![1, s.dim])?;
			let kk = g.shrink(k, vec![(0, s.keys), (kh, kh + 1), (0, s.dim)])?;
			let kk = g.reshape(kk, vec![s.keys, s.dim])?;
			let vv = g.shrink(v, vec![(0, s.keys), (kh, kh + 1), (0, s.dim)])?;
			let vv = g.reshape(vv, vec![s.keys, s.dim])?;
			let mut mask = vec![];
			for &ok in &allowed {
				mask.push(g.constant(if ok { 1. } else { 0. }, vec![])?);
			}
			let mask = g.stack(mask, 0)?;
			let mask = g.reshape(mask, vec![s.keys, 1])?;
			let zero = g.constant(0., vec![])?;
			let kk = g.where_(mask, kk, zero)?;
			let vv = g.where_(mask, vv, zero)?;
			let kt = g.permute(kk, vec![1, 0])?;
			let score = g.matmul(qq, kt)?;
			let scale = g.constant(1. / (s.dim as f32).sqrt(), vec![])?;
			let score = g.binary(Binary::Mul, score, scale)?;
			let mask = g.reshape(mask, vec![1, s.keys])?;
			let neg = g.constant(f32::NEG_INFINITY, vec![])?;
			let score = g.where_(mask, score, neg)?;
			let max = g.reduce(score, Reduce::Max, vec![1], true)?;
			let max = g.stop_gradient(max)?;
			let shifted = g.binary(Binary::Sub, score, max)?;
			let exp = g.unary(Unary::Exp, shifted)?;
			let sum = g.reduce(exp, Reduce::Sum, vec![1], true)?;
			let probs = g.binary(Binary::Div, exp, sum)?;
			let out = g.matmul(probs, vv)?;
			heads.push(g.reshape(out, vec![s.dim])?);
		}
		times.push(g.stack(heads, 0)?);
	}
	if s.queries == 0 {
		return g.constant(0., vec![0, s.heads, s.dim]);
	}
	g.stack(times, 0)
}
```

```sh check
cargo test --test book_23 todo_23_005 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## key 軸を小さな tile に分割し、m、l、u を C の累積値として保持する。

<!-- todo:23-006 -->

実装は[この章の `flash::reference、FlashPlan::compile、FlashPlan::run、flash::shader、flash::register、compile_boundaries`](#flash-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_23.rs mode=append
#[test]
fn todo_23_006() {
	suite0::online_boundary_masks_gqa();
}
```

`src/flash.rs` の `pub fn compile(s: AttentionShape,`の定義を次の全体で置き換えます。

```rust file="src/flash.rs" mode="replace" item="pub fn compile(s: AttentionShape,"
	pub fn compile(s: AttentionShape, tile: usize) -> Result<Self, String> {
		if tile == 0
			|| s.dim == 0
			|| s.heads == 0
			|| s.kv_heads == 0
			|| !s.heads.is_multiple_of(s.kv_heads)
		{
			return Err("flash shape/tile".into());
		}
		let causal = if s.causal {
			format!("j<={}+t", s.query_start)
		} else {
			"1".into()
		};
		let source = format!(
			"{}int chibi_entry(void*const*buffers,const uint64_t*s){{const float*q=buffers[0],*k=buffers[1],*v=buffers[2],*valid=buffers[3];float*out=buffers[4];for(size_t t=0;t<{queries};t++){{for(size_t h=0;h<{heads};h++){{size_t qb=(t*{heads}+h)*{dim},kh=h/{ratio};float m=-INFINITY,l=0.0f,u[{dim}]={{0}};for(size_t start=0;start<{keys};start+={tile}){{size_t end={keys}-start<{tile}?{keys}:start+{tile};float tm=-INFINITY,scores[{tile}];for(size_t j=start;j<end;j++){{scores[j-start]=-INFINITY;if(valid[j]!=0.0f&&({causal})){{float score=0.0f;for(size_t d=0;d<{dim};d++){{score+=q[qb+d]*k[(j*{kv}+kh)*{dim}+d];}}score/=sqrtf({dim}.0f);scores[j-start]=score;tm=fmaxf(tm,score);}}}}if(tm==-INFINITY)continue;float nm=fmaxf(m,tm),alpha=l==0.0f?0.0f:expf(m-nm);l*=alpha;for(size_t d=0;d<{dim};d++){{u[d]*=alpha;}}for(size_t j=start;j<end;j++){{if(valid[j]!=0.0f&&({causal})){{float p=expf(scores[j-start]-nm);l+=p;for(size_t d=0;d<{dim};d++){{u[d]+=p*v[(j*{kv}+kh)*{dim}+d];}}}}}}m=nm;}}for(size_t d=0;d<{dim};d++){{out[qb+d]=l>0.0f?u[d]/l:0.0f;}}}}}}return 0;}}",
			crate::graph::C_HELPERS,
			queries = s.queries,
			heads = s.heads,
			dim = s.dim,
			ratio = s.heads / s.kv_heads,
			keys = s.keys,
			kv = s.kv_heads
		);
		let library = unsafe { crate::runtime::NativeLibrary::compile(&source, "chibi_entry") }?;
		Ok(Self {
			source,
			shape: s,
			tile,
			library,
		})
	}
```

```sh check
cargo test --test book_23 todo_23_006 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## tile を読み進めるたびに m の変更で既存の l と u を補正する。

<!-- todo:23-007 -->

実装は[この章の `flash::reference、FlashPlan::compile、FlashPlan::run、flash::shader、flash::register、compile_boundaries`](#flash-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_23.rs mode=append
#[test]
fn todo_23_007() {
	suite0::all_masked_and_late_max();
}
```

`src/flash.rs` の `pub fn compile(s: AttentionShape,`の定義を次の全体で置き換えます。

```rust file="src/flash.rs" mode="replace" item="pub fn compile(s: AttentionShape,"
	pub fn compile(s: AttentionShape, tile: usize) -> Result<Self, String> {
		if tile == 0
			|| s.dim == 0
			|| s.heads == 0
			|| s.kv_heads == 0
			|| !s.heads.is_multiple_of(s.kv_heads)
		{
			return Err("flash shape/tile".into());
		}
		let causal = if s.causal {
			format!("j<={}+t", s.query_start)
		} else {
			"1".into()
		};
		let source = format!(
			"{}int chibi_entry(void*const*buffers,const uint64_t*s){{const float*q=buffers[0],*k=buffers[1],*v=buffers[2],*valid=buffers[3];float*out=buffers[4];for(size_t t=0;t<{queries};t++){{for(size_t h=0;h<{heads};h++){{size_t qb=(t*{heads}+h)*{dim},kh=h/{ratio};float m=-INFINITY,l=0.0f,u[{dim}]={{0}};for(size_t start=0;start<{keys};start+={tile}){{size_t end={keys}-start<{tile}?{keys}:start+{tile};float tm=-INFINITY,scores[{tile}];for(size_t j=start;j<end;j++){{scores[j-start]=-INFINITY;if(valid[j]!=0.0f&&({causal})){{float score=0.0f;for(size_t d=0;d<{dim};d++){{score+=q[qb+d]*k[(j*{kv}+kh)*{dim}+d];}}score/=sqrtf({dim}.0f);scores[j-start]=score;tm=fmaxf(tm,score);}}}}if(tm==-INFINITY)continue;float nm=fmaxf(m,tm),alpha=l==0.0f?0.0f:expf(m-nm);l*=alpha;for(size_t d=0;d<{dim};d++){{u[d]*=alpha;}}for(size_t j=start;j<end;j++){{if(valid[j]!=0.0f&&({causal})){{float p=expf(scores[j-start]-nm);l+=p;for(size_t d=0;d<{dim};d++){{u[d]+=p*v[(j*{kv}+kh)*{dim}+d];}}}}}}m=nm;}}for(size_t d=0;d<{dim};d++){{out[qb+d]=l>0.0f?u[d]/l:0.0f;}}}}}}return 0;}}",
			crate::graph::C_HELPERS,
			queries = s.queries,
			heads = s.heads,
			dim = s.dim,
			ratio = s.heads / s.kv_heads,
			keys = s.keys,
			kv = s.kv_heads
		);
		let library = unsafe { crate::runtime::NativeLibrary::compile(&source, "chibi_entry") }?;
		Ok(Self {
			source,
			shape: s,
			tile,
			library,
		})
	}
```

```sh check
cargo test --test book_23 todo_23_007 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 有効 key がない tile は状態を更新せず、全 mask 行を明示的に処理する。

<!-- todo:23-008 -->

実装は[この章の `flash::reference、FlashPlan::compile、FlashPlan::run、flash::shader、flash::register、compile_boundaries`](#flash-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_23.rs mode=append
#[test]
fn todo_23_008() {
	suite0::model_boundary_compiler_replaces_actual_graph();
}
```

`src/flash.rs` の `pub fn compile(s: AttentionShape,`の定義を次の全体で置き換えます。

```rust file="src/flash.rs" mode="replace" item="pub fn compile(s: AttentionShape,"
	pub fn compile(s: AttentionShape, tile: usize) -> Result<Self, String> {
		if tile == 0
			|| s.dim == 0
			|| s.heads == 0
			|| s.kv_heads == 0
			|| !s.heads.is_multiple_of(s.kv_heads)
		{
			return Err("flash shape/tile".into());
		}
		let causal = if s.causal {
			format!("j<={}+t", s.query_start)
		} else {
			"1".into()
		};
		let source = format!(
			"{}int chibi_entry(void*const*buffers,const uint64_t*s){{const float*q=buffers[0],*k=buffers[1],*v=buffers[2],*valid=buffers[3];float*out=buffers[4];for(size_t t=0;t<{queries};t++){{for(size_t h=0;h<{heads};h++){{size_t qb=(t*{heads}+h)*{dim},kh=h/{ratio};float m=-INFINITY,l=0.0f,u[{dim}]={{0}};for(size_t start=0;start<{keys};start+={tile}){{size_t end={keys}-start<{tile}?{keys}:start+{tile};float tm=-INFINITY,scores[{tile}];for(size_t j=start;j<end;j++){{scores[j-start]=-INFINITY;if(valid[j]!=0.0f&&({causal})){{float score=0.0f;for(size_t d=0;d<{dim};d++){{score+=q[qb+d]*k[(j*{kv}+kh)*{dim}+d];}}score/=sqrtf({dim}.0f);scores[j-start]=score;tm=fmaxf(tm,score);}}}}if(tm==-INFINITY)continue;float nm=fmaxf(m,tm),alpha=l==0.0f?0.0f:expf(m-nm);l*=alpha;for(size_t d=0;d<{dim};d++){{u[d]*=alpha;}}for(size_t j=start;j<end;j++){{if(valid[j]!=0.0f&&({causal})){{float p=expf(scores[j-start]-nm);l+=p;for(size_t d=0;d<{dim};d++){{u[d]+=p*v[(j*{kv}+kh)*{dim}+d];}}}}}}m=nm;}}for(size_t d=0;d<{dim};d++){{out[qb+d]=l>0.0f?u[d]/l:0.0f;}}}}}}return 0;}}",
			crate::graph::C_HELPERS,
			queries = s.queries,
			heads = s.heads,
			dim = s.dim,
			ratio = s.heads / s.kv_heads,
			keys = s.keys,
			kv = s.kv_heads
		);
		let library = unsafe { crate::runtime::NativeLibrary::compile(&source, "chibi_entry") }?;
		Ok(Self {
			source,
			shape: s,
			tile,
			library,
		})
	}
```

```sh check
cargo test --test book_23 todo_23_008 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 最後の不完全 tile を guard し、無効な K と V の LOAD を生成しない。

<!-- todo:23-009 -->

実装は[この章の `flash::reference、FlashPlan::compile、FlashPlan::run、flash::shader、flash::register、compile_boundaries`](#flash-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_23.rs mode=append
#[test]
fn todo_23_009() {
	suite0::native_c_and_composite_graph();
}
```

`src/flash.rs` の `pub fn compile(s: AttentionShape,`の定義を次の全体で置き換えます。

```rust file="src/flash.rs" mode="replace" item="pub fn compile(s: AttentionShape,"
	pub fn compile(s: AttentionShape, tile: usize) -> Result<Self, String> {
		if tile == 0
			|| s.dim == 0
			|| s.heads == 0
			|| s.kv_heads == 0
			|| !s.heads.is_multiple_of(s.kv_heads)
		{
			return Err("flash shape/tile".into());
		}
		let causal = if s.causal {
			format!("j<={}+t", s.query_start)
		} else {
			"1".into()
		};
		let source = format!(
			"{}int chibi_entry(void*const*buffers,const uint64_t*s){{const float*q=buffers[0],*k=buffers[1],*v=buffers[2],*valid=buffers[3];float*out=buffers[4];for(size_t t=0;t<{queries};t++){{for(size_t h=0;h<{heads};h++){{size_t qb=(t*{heads}+h)*{dim},kh=h/{ratio};float m=-INFINITY,l=0.0f,u[{dim}]={{0}};for(size_t start=0;start<{keys};start+={tile}){{size_t end={keys}-start<{tile}?{keys}:start+{tile};float tm=-INFINITY,scores[{tile}];for(size_t j=start;j<end;j++){{scores[j-start]=-INFINITY;if(valid[j]!=0.0f&&({causal})){{float score=0.0f;for(size_t d=0;d<{dim};d++){{score+=q[qb+d]*k[(j*{kv}+kh)*{dim}+d];}}score/=sqrtf({dim}.0f);scores[j-start]=score;tm=fmaxf(tm,score);}}}}if(tm==-INFINITY)continue;float nm=fmaxf(m,tm),alpha=l==0.0f?0.0f:expf(m-nm);l*=alpha;for(size_t d=0;d<{dim};d++){{u[d]*=alpha;}}for(size_t j=start;j<end;j++){{if(valid[j]!=0.0f&&({causal})){{float p=expf(scores[j-start]-nm);l+=p;for(size_t d=0;d<{dim};d++){{u[d]+=p*v[(j*{kv}+kh)*{dim}+d];}}}}}}m=nm;}}for(size_t d=0;d<{dim};d++){{out[qb+d]=l>0.0f?u[d]/l:0.0f;}}}}}}return 0;}}",
			crate::graph::C_HELPERS,
			queries = s.queries,
			heads = s.heads,
			dim = s.dim,
			ratio = s.heads / s.kv_heads,
			keys = s.keys,
			kv = s.kv_heads
		);
		let library = unsafe { crate::runtime::NativeLibrary::compile(&source, "chibi_entry") }?;
		Ok(Self {
			source,
			shape: s,
			tile,
			library,
		})
	}
```

`src/flash.rs` の `pub fn shader(`の定義を次の全体で置き換えます。

```rust file="src/flash.rs" mode="replace" item="pub fn shader("
pub fn shader(d: Dialect, s: AttentionShape, tile: usize) -> Result<Kernel, String> {
	if s.dim == 0 || s.heads == 0 || s.kv_heads == 0 || !s.heads.is_multiple_of(s.kv_heads) {
		return Err("attention heads/dim".into());
	}
	s.queries
		.checked_mul(s.heads)
		.and_then(|n| n.checked_mul(s.dim))
		.ok_or("attention query shape overflow")?;
	s.keys
		.checked_mul(s.kv_heads)
		.and_then(|n| n.checked_mul(s.dim))
		.ok_or("attention key shape overflow")?;
	s.query_start
		.checked_add(s.queries)
		.ok_or("attention position overflow")?;
	if tile == 0 {
		return Err("attention tile=0".into());
	}
	let bytes = tile
		.checked_mul(s.dim)
		.and_then(|n| n.checked_mul(8))
		.ok_or("attention local overflow")?;
	let space = match d {
		Dialect::Metal => "threadgroup",
		Dialect::OpenCl => "__local",
	};
	let barrier = match d {
		Dialect::Metal => "threadgroup_barrier(mem_flags::mem_threadgroup);",
		Dialect::OpenCl => "barrier(CLK_LOCAL_MEM_FENCE);",
	};
	let allowed = if s.causal {
		format!("j<={}+t", s.query_start)
	} else {
		"1".into()
	};
	let body = format!(
		"{space} float kk[{tile}*{dim}],vv[{tile}*{dim}];size_t t=g/{heads},h=g%{heads},kh=h/{ratio},qb=g*{dim};float m=-INFINITY,den=0.0f,u[{dim}];for(size_t a=0;a<{dim};a++){{u[a]=0.0f;}}for(size_t start=0;start<{keys};start+={tile}){{for(size_t z=l;z<{tile}*{dim};z+=32){{size_t j=start+z/{dim},a=z%{dim};bool ok=j<{keys}&&({allowed})&&b3[j]!=0.0f;if(ok){{kk[z]=b1[(j*{kv}+kh)*{dim}+a];vv[z]=b2[(j*{kv}+kh)*{dim}+a];}}else{{kk[z]=0.0f;vv[z]=0.0f;}}}}{barrier}if(l==0){{for(size_t off=0;off<{tile};off++){{size_t j=start+off;if(j<{keys}&&({allowed})&&b3[j]!=0.0f){{float score=0.0f;for(size_t a=0;a<{dim};a++){{score+=b0[qb+a]*kk[off*{dim}+a];}}score*=1.0f/sqrt((float){dim});float nm=fmax(m,score);float alpha=den==0.0f?0.0f:exp(m-nm),p=exp(score-nm);den=den*alpha+p;for(size_t a=0;a<{dim};a++){{u[a]=u[a]*alpha+p*vv[off*{dim}+a];}}m=nm;}}}}}}{barrier}}}if(l==0){{for(size_t a=0;a<{dim};a++){{b4[qb+a]=den>0.0f?u[a]/den:0.0f;}}}}",
		dim = s.dim,
		heads = s.heads,
		ratio = s.heads / s.kv_heads,
		keys = s.keys,
		kv = s.kv_heads
	);
	Ok(Kernel {
		scalars: vec![],
		source: header(d, 5, &body),
		entry: "chibi".into(),
		lengths: vec![
			s.queries * s.heads * s.dim,
			s.keys * s.kv_heads * s.dim,
			s.keys * s.kv_heads * s.dim,
			s.keys,
			s.queries * s.heads * s.dim,
		],
		output: 4,
		groups: s.queries * s.heads,
		local: 32,
		shared_bytes: bytes,
	})
}
```

```sh check
cargo test --test book_23 todo_23_009 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## tile 幅1、2、3、8で通常版と同じ結果を得る。

<!-- todo:23-010 -->

実装は[この章の `flash::reference、FlashPlan::compile、FlashPlan::run、flash::shader、flash::register、compile_boundaries`](#flash-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_23.rs mode=append
#[test]
fn todo_23_010() {
	suite0::online_boundary_masks_gqa();
}
```

```sh check
cargo test --test book_23 todo_23_010 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## まず一つの group が query の小タイルを担当し、K と V を順に local memory へ運ぶ。

<!-- todo:23-011 -->

実装は[この章の `flash::reference、FlashPlan::compile、FlashPlan::run、flash::shader、flash::register、compile_boundaries`](#flash-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_23.rs mode=append
#[test]
fn todo_23_011() {
	suite0::all_masked_and_late_max();
}
```

`src/flash.rs` の `pub fn shader(`の定義を次の全体で置き換えます。

```rust file="src/flash.rs" mode="replace" item="pub fn shader("
pub fn shader(d: Dialect, s: AttentionShape, tile: usize) -> Result<Kernel, String> {
	if s.dim == 0 || s.heads == 0 || s.kv_heads == 0 || !s.heads.is_multiple_of(s.kv_heads) {
		return Err("attention heads/dim".into());
	}
	s.queries
		.checked_mul(s.heads)
		.and_then(|n| n.checked_mul(s.dim))
		.ok_or("attention query shape overflow")?;
	s.keys
		.checked_mul(s.kv_heads)
		.and_then(|n| n.checked_mul(s.dim))
		.ok_or("attention key shape overflow")?;
	s.query_start
		.checked_add(s.queries)
		.ok_or("attention position overflow")?;
	if tile == 0 {
		return Err("attention tile=0".into());
	}
	let bytes = tile
		.checked_mul(s.dim)
		.and_then(|n| n.checked_mul(8))
		.ok_or("attention local overflow")?;
	let space = match d {
		Dialect::Metal => "threadgroup",
		Dialect::OpenCl => "__local",
	};
	let barrier = match d {
		Dialect::Metal => "threadgroup_barrier(mem_flags::mem_threadgroup);",
		Dialect::OpenCl => "barrier(CLK_LOCAL_MEM_FENCE);",
	};
	let allowed = if s.causal {
		format!("j<={}+t", s.query_start)
	} else {
		"1".into()
	};
	let body = format!(
		"{space} float kk[{tile}*{dim}],vv[{tile}*{dim}];size_t t=g/{heads},h=g%{heads},kh=h/{ratio},qb=g*{dim};float m=-INFINITY,den=0.0f,u[{dim}];for(size_t a=0;a<{dim};a++){{u[a]=0.0f;}}for(size_t start=0;start<{keys};start+={tile}){{for(size_t z=l;z<{tile}*{dim};z+=32){{size_t j=start+z/{dim},a=z%{dim};bool ok=j<{keys}&&({allowed})&&b3[j]!=0.0f;if(ok){{kk[z]=b1[(j*{kv}+kh)*{dim}+a];vv[z]=b2[(j*{kv}+kh)*{dim}+a];}}else{{kk[z]=0.0f;vv[z]=0.0f;}}}}{barrier}if(l==0){{for(size_t off=0;off<{tile};off++){{size_t j=start+off;if(j<{keys}&&({allowed})&&b3[j]!=0.0f){{float score=0.0f;for(size_t a=0;a<{dim};a++){{score+=b0[qb+a]*kk[off*{dim}+a];}}score*=1.0f/sqrt((float){dim});float nm=fmax(m,score);float alpha=den==0.0f?0.0f:exp(m-nm),p=exp(score-nm);den=den*alpha+p;for(size_t a=0;a<{dim};a++){{u[a]=u[a]*alpha+p*vv[off*{dim}+a];}}m=nm;}}}}}}{barrier}}}if(l==0){{for(size_t a=0;a<{dim};a++){{b4[qb+a]=den>0.0f?u[a]/den:0.0f;}}}}",
		dim = s.dim,
		heads = s.heads,
		ratio = s.heads / s.kv_heads,
		keys = s.keys,
		kv = s.kv_heads
	);
	Ok(Kernel {
		scalars: vec![],
		source: header(d, 5, &body),
		entry: "chibi".into(),
		lengths: vec![
			s.queries * s.heads * s.dim,
			s.keys * s.kv_heads * s.dim,
			s.keys * s.kv_heads * s.dim,
			s.keys,
			s.queries * s.heads * s.dim,
		],
		output: 4,
		groups: s.queries * s.heads,
		local: 32,
		shared_bytes: bytes,
	})
}
```

```sh check
cargo test --test book_23 todo_23_011 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## QK と PV は通常の f32 積和を使い、hardware matrix 命令はまだ使わない。

<!-- todo:23-012 -->

実装は[この章の `flash::reference、FlashPlan::compile、FlashPlan::run、flash::shader、flash::register、compile_boundaries`](#flash-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_23.rs mode=append
#[test]
fn todo_23_012() {
	suite0::model_boundary_compiler_replaces_actual_graph();
}
```

`src/flash.rs` の `pub fn shader(`の定義を次の全体で置き換えます。

```rust file="src/flash.rs" mode="replace" item="pub fn shader("
pub fn shader(d: Dialect, s: AttentionShape, tile: usize) -> Result<Kernel, String> {
	if s.dim == 0 || s.heads == 0 || s.kv_heads == 0 || !s.heads.is_multiple_of(s.kv_heads) {
		return Err("attention heads/dim".into());
	}
	s.queries
		.checked_mul(s.heads)
		.and_then(|n| n.checked_mul(s.dim))
		.ok_or("attention query shape overflow")?;
	s.keys
		.checked_mul(s.kv_heads)
		.and_then(|n| n.checked_mul(s.dim))
		.ok_or("attention key shape overflow")?;
	s.query_start
		.checked_add(s.queries)
		.ok_or("attention position overflow")?;
	if tile == 0 {
		return Err("attention tile=0".into());
	}
	let bytes = tile
		.checked_mul(s.dim)
		.and_then(|n| n.checked_mul(8))
		.ok_or("attention local overflow")?;
	let space = match d {
		Dialect::Metal => "threadgroup",
		Dialect::OpenCl => "__local",
	};
	let barrier = match d {
		Dialect::Metal => "threadgroup_barrier(mem_flags::mem_threadgroup);",
		Dialect::OpenCl => "barrier(CLK_LOCAL_MEM_FENCE);",
	};
	let allowed = if s.causal {
		format!("j<={}+t", s.query_start)
	} else {
		"1".into()
	};
	let body = format!(
		"{space} float kk[{tile}*{dim}],vv[{tile}*{dim}];size_t t=g/{heads},h=g%{heads},kh=h/{ratio},qb=g*{dim};float m=-INFINITY,den=0.0f,u[{dim}];for(size_t a=0;a<{dim};a++){{u[a]=0.0f;}}for(size_t start=0;start<{keys};start+={tile}){{for(size_t z=l;z<{tile}*{dim};z+=32){{size_t j=start+z/{dim},a=z%{dim};bool ok=j<{keys}&&({allowed})&&b3[j]!=0.0f;if(ok){{kk[z]=b1[(j*{kv}+kh)*{dim}+a];vv[z]=b2[(j*{kv}+kh)*{dim}+a];}}else{{kk[z]=0.0f;vv[z]=0.0f;}}}}{barrier}if(l==0){{for(size_t off=0;off<{tile};off++){{size_t j=start+off;if(j<{keys}&&({allowed})&&b3[j]!=0.0f){{float score=0.0f;for(size_t a=0;a<{dim};a++){{score+=b0[qb+a]*kk[off*{dim}+a];}}score*=1.0f/sqrt((float){dim});float nm=fmax(m,score);float alpha=den==0.0f?0.0f:exp(m-nm),p=exp(score-nm);den=den*alpha+p;for(size_t a=0;a<{dim};a++){{u[a]=u[a]*alpha+p*vv[off*{dim}+a];}}m=nm;}}}}}}{barrier}}}if(l==0){{for(size_t a=0;a<{dim};a++){{b4[qb+a]=den>0.0f?u[a]/den:0.0f;}}}}",
		dim = s.dim,
		heads = s.heads,
		ratio = s.heads / s.kv_heads,
		keys = s.keys,
		kv = s.kv_heads
	);
	Ok(Kernel {
		scalars: vec![],
		source: header(d, 5, &body),
		entry: "chibi".into(),
		lengths: vec![
			s.queries * s.heads * s.dim,
			s.keys * s.kv_heads * s.dim,
			s.keys * s.kv_heads * s.dim,
			s.keys,
			s.queries * s.heads * s.dim,
		],
		output: 4,
		groups: s.queries * s.heads,
		local: 32,
		shared_bytes: bytes,
	})
}
```

```sh check
cargo test --test book_23 todo_23_012 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 全 lane が LOAD 後と local buffer 再利用前の barrier を通る構造にする。

<!-- todo:23-013 -->

実装は[この章の `flash::reference、FlashPlan::compile、FlashPlan::run、flash::shader、flash::register、compile_boundaries`](#flash-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_23.rs mode=append
#[test]
fn todo_23_013() {
	suite0::native_c_and_composite_graph();
}
```

`src/flash.rs` の `pub fn shader(`の定義を次の全体で置き換えます。

```rust file="src/flash.rs" mode="replace" item="pub fn shader("
pub fn shader(d: Dialect, s: AttentionShape, tile: usize) -> Result<Kernel, String> {
	if s.dim == 0 || s.heads == 0 || s.kv_heads == 0 || !s.heads.is_multiple_of(s.kv_heads) {
		return Err("attention heads/dim".into());
	}
	s.queries
		.checked_mul(s.heads)
		.and_then(|n| n.checked_mul(s.dim))
		.ok_or("attention query shape overflow")?;
	s.keys
		.checked_mul(s.kv_heads)
		.and_then(|n| n.checked_mul(s.dim))
		.ok_or("attention key shape overflow")?;
	s.query_start
		.checked_add(s.queries)
		.ok_or("attention position overflow")?;
	if tile == 0 {
		return Err("attention tile=0".into());
	}
	let bytes = tile
		.checked_mul(s.dim)
		.and_then(|n| n.checked_mul(8))
		.ok_or("attention local overflow")?;
	let space = match d {
		Dialect::Metal => "threadgroup",
		Dialect::OpenCl => "__local",
	};
	let barrier = match d {
		Dialect::Metal => "threadgroup_barrier(mem_flags::mem_threadgroup);",
		Dialect::OpenCl => "barrier(CLK_LOCAL_MEM_FENCE);",
	};
	let allowed = if s.causal {
		format!("j<={}+t", s.query_start)
	} else {
		"1".into()
	};
	let body = format!(
		"{space} float kk[{tile}*{dim}],vv[{tile}*{dim}];size_t t=g/{heads},h=g%{heads},kh=h/{ratio},qb=g*{dim};float m=-INFINITY,den=0.0f,u[{dim}];for(size_t a=0;a<{dim};a++){{u[a]=0.0f;}}for(size_t start=0;start<{keys};start+={tile}){{for(size_t z=l;z<{tile}*{dim};z+=32){{size_t j=start+z/{dim},a=z%{dim};bool ok=j<{keys}&&({allowed})&&b3[j]!=0.0f;if(ok){{kk[z]=b1[(j*{kv}+kh)*{dim}+a];vv[z]=b2[(j*{kv}+kh)*{dim}+a];}}else{{kk[z]=0.0f;vv[z]=0.0f;}}}}{barrier}if(l==0){{for(size_t off=0;off<{tile};off++){{size_t j=start+off;if(j<{keys}&&({allowed})&&b3[j]!=0.0f){{float score=0.0f;for(size_t a=0;a<{dim};a++){{score+=b0[qb+a]*kk[off*{dim}+a];}}score*=1.0f/sqrt((float){dim});float nm=fmax(m,score);float alpha=den==0.0f?0.0f:exp(m-nm),p=exp(score-nm);den=den*alpha+p;for(size_t a=0;a<{dim};a++){{u[a]=u[a]*alpha+p*vv[off*{dim}+a];}}m=nm;}}}}}}{barrier}}}if(l==0){{for(size_t a=0;a<{dim};a++){{b4[qb+a]=den>0.0f?u[a]/den:0.0f;}}}}",
		dim = s.dim,
		heads = s.heads,
		ratio = s.heads / s.kv_heads,
		keys = s.keys,
		kv = s.kv_heads
	);
	Ok(Kernel {
		scalars: vec![],
		source: header(d, 5, &body),
		entry: "chibi".into(),
		lengths: vec![
			s.queries * s.heads * s.dim,
			s.keys * s.kv_heads * s.dim,
			s.keys * s.kv_heads * s.dim,
			s.keys,
			s.queries * s.heads * s.dim,
		],
		output: 4,
		groups: s.queries * s.heads,
		local: 32,
		shared_bytes: bytes,
	})
}
```

```sh check
cargo test --test book_23 todo_23_013 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## query と key の端数を別々に guard し、無効 lane に単位元を与える。

<!-- todo:23-014 -->

実装は[この章の `flash::reference、FlashPlan::compile、FlashPlan::run、flash::shader、flash::register、compile_boundaries`](#flash-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_23.rs mode=append
#[test]
fn todo_23_014() {
	suite0::online_boundary_masks_gqa();
}
```

`src/flash.rs` の `pub fn shader(`の定義を次の全体で置き換えます。

```rust file="src/flash.rs" mode="replace" item="pub fn shader("
pub fn shader(d: Dialect, s: AttentionShape, tile: usize) -> Result<Kernel, String> {
	if s.dim == 0 || s.heads == 0 || s.kv_heads == 0 || !s.heads.is_multiple_of(s.kv_heads) {
		return Err("attention heads/dim".into());
	}
	s.queries
		.checked_mul(s.heads)
		.and_then(|n| n.checked_mul(s.dim))
		.ok_or("attention query shape overflow")?;
	s.keys
		.checked_mul(s.kv_heads)
		.and_then(|n| n.checked_mul(s.dim))
		.ok_or("attention key shape overflow")?;
	s.query_start
		.checked_add(s.queries)
		.ok_or("attention position overflow")?;
	if tile == 0 {
		return Err("attention tile=0".into());
	}
	let bytes = tile
		.checked_mul(s.dim)
		.and_then(|n| n.checked_mul(8))
		.ok_or("attention local overflow")?;
	let space = match d {
		Dialect::Metal => "threadgroup",
		Dialect::OpenCl => "__local",
	};
	let barrier = match d {
		Dialect::Metal => "threadgroup_barrier(mem_flags::mem_threadgroup);",
		Dialect::OpenCl => "barrier(CLK_LOCAL_MEM_FENCE);",
	};
	let allowed = if s.causal {
		format!("j<={}+t", s.query_start)
	} else {
		"1".into()
	};
	let body = format!(
		"{space} float kk[{tile}*{dim}],vv[{tile}*{dim}];size_t t=g/{heads},h=g%{heads},kh=h/{ratio},qb=g*{dim};float m=-INFINITY,den=0.0f,u[{dim}];for(size_t a=0;a<{dim};a++){{u[a]=0.0f;}}for(size_t start=0;start<{keys};start+={tile}){{for(size_t z=l;z<{tile}*{dim};z+=32){{size_t j=start+z/{dim},a=z%{dim};bool ok=j<{keys}&&({allowed})&&b3[j]!=0.0f;if(ok){{kk[z]=b1[(j*{kv}+kh)*{dim}+a];vv[z]=b2[(j*{kv}+kh)*{dim}+a];}}else{{kk[z]=0.0f;vv[z]=0.0f;}}}}{barrier}if(l==0){{for(size_t off=0;off<{tile};off++){{size_t j=start+off;if(j<{keys}&&({allowed})&&b3[j]!=0.0f){{float score=0.0f;for(size_t a=0;a<{dim};a++){{score+=b0[qb+a]*kk[off*{dim}+a];}}score*=1.0f/sqrt((float){dim});float nm=fmax(m,score);float alpha=den==0.0f?0.0f:exp(m-nm),p=exp(score-nm);den=den*alpha+p;for(size_t a=0;a<{dim};a++){{u[a]=u[a]*alpha+p*vv[off*{dim}+a];}}m=nm;}}}}}}{barrier}}}if(l==0){{for(size_t a=0;a<{dim};a++){{b4[qb+a]=den>0.0f?u[a]/den:0.0f;}}}}",
		dim = s.dim,
		heads = s.heads,
		ratio = s.heads / s.kv_heads,
		keys = s.keys,
		kv = s.kv_heads
	);
	Ok(Kernel {
		scalars: vec![],
		source: header(d, 5, &body),
		entry: "chibi".into(),
		lengths: vec![
			s.queries * s.heads * s.dim,
			s.keys * s.kv_heads * s.dim,
			s.keys * s.kv_heads * s.dim,
			s.keys,
			s.queries * s.heads * s.dim,
		],
		output: 4,
		groups: s.queries * s.heads,
		local: 32,
		shared_bytes: bytes,
	})
}
```

```sh check
cargo test --test book_23 todo_23_014 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## Metal で検証後、同じ kernel IR を OpenCL に出して通常版と比較する。

<!-- todo:23-015 -->

実装は[この章の `flash::reference、FlashPlan::compile、FlashPlan::run、flash::shader、flash::register、compile_boundaries`](#flash-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_23.rs mode=append
#[test]
fn todo_23_015() {
	suite0::all_masked_and_late_max();
}
```

`src/flash.rs` の `pub fn shader(`の定義を次の全体で置き換えます。

```rust file="src/flash.rs" mode="replace" item="pub fn shader("
pub fn shader(d: Dialect, s: AttentionShape, tile: usize) -> Result<Kernel, String> {
	if s.dim == 0 || s.heads == 0 || s.kv_heads == 0 || !s.heads.is_multiple_of(s.kv_heads) {
		return Err("attention heads/dim".into());
	}
	s.queries
		.checked_mul(s.heads)
		.and_then(|n| n.checked_mul(s.dim))
		.ok_or("attention query shape overflow")?;
	s.keys
		.checked_mul(s.kv_heads)
		.and_then(|n| n.checked_mul(s.dim))
		.ok_or("attention key shape overflow")?;
	s.query_start
		.checked_add(s.queries)
		.ok_or("attention position overflow")?;
	if tile == 0 {
		return Err("attention tile=0".into());
	}
	let bytes = tile
		.checked_mul(s.dim)
		.and_then(|n| n.checked_mul(8))
		.ok_or("attention local overflow")?;
	let space = match d {
		Dialect::Metal => "threadgroup",
		Dialect::OpenCl => "__local",
	};
	let barrier = match d {
		Dialect::Metal => "threadgroup_barrier(mem_flags::mem_threadgroup);",
		Dialect::OpenCl => "barrier(CLK_LOCAL_MEM_FENCE);",
	};
	let allowed = if s.causal {
		format!("j<={}+t", s.query_start)
	} else {
		"1".into()
	};
	let body = format!(
		"{space} float kk[{tile}*{dim}],vv[{tile}*{dim}];size_t t=g/{heads},h=g%{heads},kh=h/{ratio},qb=g*{dim};float m=-INFINITY,den=0.0f,u[{dim}];for(size_t a=0;a<{dim};a++){{u[a]=0.0f;}}for(size_t start=0;start<{keys};start+={tile}){{for(size_t z=l;z<{tile}*{dim};z+=32){{size_t j=start+z/{dim},a=z%{dim};bool ok=j<{keys}&&({allowed})&&b3[j]!=0.0f;if(ok){{kk[z]=b1[(j*{kv}+kh)*{dim}+a];vv[z]=b2[(j*{kv}+kh)*{dim}+a];}}else{{kk[z]=0.0f;vv[z]=0.0f;}}}}{barrier}if(l==0){{for(size_t off=0;off<{tile};off++){{size_t j=start+off;if(j<{keys}&&({allowed})&&b3[j]!=0.0f){{float score=0.0f;for(size_t a=0;a<{dim};a++){{score+=b0[qb+a]*kk[off*{dim}+a];}}score*=1.0f/sqrt((float){dim});float nm=fmax(m,score);float alpha=den==0.0f?0.0f:exp(m-nm),p=exp(score-nm);den=den*alpha+p;for(size_t a=0;a<{dim};a++){{u[a]=u[a]*alpha+p*vv[off*{dim}+a];}}m=nm;}}}}}}{barrier}}}if(l==0){{for(size_t a=0;a<{dim};a++){{b4[qb+a]=den>0.0f?u[a]/den:0.0f;}}}}",
		dim = s.dim,
		heads = s.heads,
		ratio = s.heads / s.kv_heads,
		keys = s.keys,
		kv = s.kv_heads
	);
	Ok(Kernel {
		scalars: vec![],
		source: header(d, 5, &body),
		entry: "chibi".into(),
		lengths: vec![
			s.queries * s.heads * s.dim,
			s.keys * s.kv_heads * s.dim,
			s.keys * s.kv_heads * s.dim,
			s.keys,
			s.queries * s.heads * s.dim,
		],
		output: 4,
		groups: s.queries * s.heads,
		local: 32,
		shared_bytes: bytes,
	})
}
```

```sh check
cargo test --test book_23 todo_23_015 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## query head から KV head への対応を追加し、K/V の全複製を避ける。

<!-- todo:23-016 -->

実装は[この章の `flash::reference、FlashPlan::compile、FlashPlan::run、flash::shader、flash::register、compile_boundaries`](#flash-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_23.rs mode=append
#[test]
fn todo_23_016() {
	suite0::model_boundary_compiler_replaces_actual_graph();
}
```

`src/flash.rs` の `pub fn shader(`の定義を次の全体で置き換えます。

```rust file="src/flash.rs" mode="replace" item="pub fn shader("
pub fn shader(d: Dialect, s: AttentionShape, tile: usize) -> Result<Kernel, String> {
	if s.dim == 0 || s.heads == 0 || s.kv_heads == 0 || !s.heads.is_multiple_of(s.kv_heads) {
		return Err("attention heads/dim".into());
	}
	s.queries
		.checked_mul(s.heads)
		.and_then(|n| n.checked_mul(s.dim))
		.ok_or("attention query shape overflow")?;
	s.keys
		.checked_mul(s.kv_heads)
		.and_then(|n| n.checked_mul(s.dim))
		.ok_or("attention key shape overflow")?;
	s.query_start
		.checked_add(s.queries)
		.ok_or("attention position overflow")?;
	if tile == 0 {
		return Err("attention tile=0".into());
	}
	let bytes = tile
		.checked_mul(s.dim)
		.and_then(|n| n.checked_mul(8))
		.ok_or("attention local overflow")?;
	let space = match d {
		Dialect::Metal => "threadgroup",
		Dialect::OpenCl => "__local",
	};
	let barrier = match d {
		Dialect::Metal => "threadgroup_barrier(mem_flags::mem_threadgroup);",
		Dialect::OpenCl => "barrier(CLK_LOCAL_MEM_FENCE);",
	};
	let allowed = if s.causal {
		format!("j<={}+t", s.query_start)
	} else {
		"1".into()
	};
	let body = format!(
		"{space} float kk[{tile}*{dim}],vv[{tile}*{dim}];size_t t=g/{heads},h=g%{heads},kh=h/{ratio},qb=g*{dim};float m=-INFINITY,den=0.0f,u[{dim}];for(size_t a=0;a<{dim};a++){{u[a]=0.0f;}}for(size_t start=0;start<{keys};start+={tile}){{for(size_t z=l;z<{tile}*{dim};z+=32){{size_t j=start+z/{dim},a=z%{dim};bool ok=j<{keys}&&({allowed})&&b3[j]!=0.0f;if(ok){{kk[z]=b1[(j*{kv}+kh)*{dim}+a];vv[z]=b2[(j*{kv}+kh)*{dim}+a];}}else{{kk[z]=0.0f;vv[z]=0.0f;}}}}{barrier}if(l==0){{for(size_t off=0;off<{tile};off++){{size_t j=start+off;if(j<{keys}&&({allowed})&&b3[j]!=0.0f){{float score=0.0f;for(size_t a=0;a<{dim};a++){{score+=b0[qb+a]*kk[off*{dim}+a];}}score*=1.0f/sqrt((float){dim});float nm=fmax(m,score);float alpha=den==0.0f?0.0f:exp(m-nm),p=exp(score-nm);den=den*alpha+p;for(size_t a=0;a<{dim};a++){{u[a]=u[a]*alpha+p*vv[off*{dim}+a];}}m=nm;}}}}}}{barrier}}}if(l==0){{for(size_t a=0;a<{dim};a++){{b4[qb+a]=den>0.0f?u[a]/den:0.0f;}}}}",
		dim = s.dim,
		heads = s.heads,
		ratio = s.heads / s.kv_heads,
		keys = s.keys,
		kv = s.kv_heads
	);
	Ok(Kernel {
		scalars: vec![],
		source: header(d, 5, &body),
		entry: "chibi".into(),
		lengths: vec![
			s.queries * s.heads * s.dim,
			s.keys * s.kv_heads * s.dim,
			s.keys * s.kv_heads * s.dim,
			s.keys,
			s.queries * s.heads * s.dim,
		],
		output: 4,
		groups: s.queries * s.heads,
		local: 32,
		shared_bytes: bytes,
	})
}
```

```sh check
cargo test --test book_23 todo_23_016 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 対応 dtype、寸法、local memory、mask、勾配要求を条件として選択する。

<!-- todo:23-017 -->

実装は[この章の `flash::reference、FlashPlan::compile、FlashPlan::run、flash::shader、flash::register、compile_boundaries`](#flash-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_23.rs mode=append
#[test]
fn todo_23_017() {
	suite0::native_c_and_composite_graph();
}
```

`src/flash.rs` の `pub fn select(`の定義を次の全体で置き換えます。

```rust file="src/flash.rs" mode="replace" item="pub fn select("
pub fn select(
	s: AttentionShape,
	requires_grad: bool,
	local_limit: usize,
	tile: usize,
) -> AttentionPath {
	if requires_grad {
		return AttentionPath::Ordinary("training requires ordinary backward");
	}
	if tile == 0
		|| s.dim == 0
		|| s.heads == 0
		|| s.kv_heads == 0
		|| !s.heads.is_multiple_of(s.kv_heads)
	{
		return AttentionPath::Ordinary("unsupported shape");
	}
	if tile
		.checked_mul(s.dim)
		.and_then(|n| n.checked_mul(8))
		.is_none_or(|n| n > local_limit)
	{
		return AttentionPath::Ordinary("local memory limit");
	}
	AttentionPath::Online
}
```

`src/gpu_graph.rs` の `pub fn compile_flash(`の定義を次の全体で置き換えます。

```rust file="src/gpu_graph.rs" mode="replace" item="pub fn compile_flash("
	pub fn compile_flash(
		graph: &Graph,
		outputs: &[usize],
		backend: Backend,
		boundaries: &[crate::flash::Boundary],
		requires_grad: bool,
		local_bytes: usize,
	) -> Result<Self> {
		use std::collections::{HashMap, HashSet};
		if requires_grad {
			return Self::compile(graph, outputs, backend, 64);
		}
		let reachable = graph.topology(outputs)?.into_iter().collect::<HashSet<_>>();
		let mut skipped = HashSet::new();
		let mut replacements = HashMap::new();
		for boundary in boundaries {
			if [boundary.q, boundary.k, boundary.v, boundary.output]
				.iter()
				.any(|&id| id >= graph.nodes.len())
			{
				return Err("GPU attention boundary node".into());
			}
			if graph.shape(boundary.q)
				!= [
					boundary.shape.queries,
					boundary.shape.heads,
					boundary.shape.dim,
				] || graph.shape(boundary.k)
				!= [
					boundary.shape.keys,
					boundary.shape.kv_heads,
					boundary.shape.dim,
				] || graph.shape(boundary.v) != graph.shape(boundary.k)
			{
				return Err("GPU attention boundary shape".into());
			}
			if graph.shape(boundary.output) != graph.shape(boundary.q) {
				return Err("GPU attention boundary output shape".into());
			}
			if !reachable.contains(&boundary.output) {
				continue;
			}
			if crate::flash::select(boundary.shape, false, local_bytes, boundary.tile)
				!= crate::flash::AttentionPath::Online
			{
				return Self::compile(graph, outputs, backend, 64);
			}
			if !matches!(graph.nodes[boundary.output].op, Op::Stage { .. }) {
				return Self::compile(graph, outputs, backend, 64);
			}
			if ![boundary.q, boundary.k, boundary.v]
				.iter()
				.all(|&id| matches!(graph.nodes[id].op, Op::Stage { .. } | Op::Input { .. }))
			{
				return Self::compile(graph, outputs, backend, 64);
			}
			let before = graph
				.topology(&[boundary.q, boundary.k, boundary.v])?
				.into_iter()
				.collect::<HashSet<_>>();
			let inside = graph
				.topology(&[boundary.output])?
				.into_iter()
				.filter(|id| !before.contains(id))
				.collect::<HashSet<_>>();
			skipped.extend(crate::flash::boundary_skips(
				graph,
				outputs,
				&reachable,
				&inside,
				boundary.output,
			)?);
			let kernel = crate::flash::shader(backend.dialect(), boundary.shape, boundary.tile)?;
			replacements.insert(
				boundary.output,
				(
					kernel,
					vec![
						boundary.q,
						boundary.k,
						boundary.v,
						usize::MAX,
						boundary.output,
					],
				),
			);
		}
		let ordinary = lower(graph, outputs, backend.dialect(), 64)?;
		let mut lowered = vec![];
		for (kernel, slots) in ordinary {
			let id = *slots.last().ok_or("attention output")?;
			if skipped.contains(&id) {
				continue;
			}
			lowered.push(replacements.remove(&id).unwrap_or((kernel, slots)));
		}
		Self::from_lowered(graph, outputs, backend, lowered, true)
	}
```

```sh check
cargo test --test book_23 todo_23_017 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 未対応条件では22の通常 attention を選び、選択理由を dump に残す。

<!-- todo:23-018 -->

実装は[この章の `flash::reference、FlashPlan::compile、FlashPlan::run、flash::shader、flash::register、compile_boundaries`](#flash-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_23.rs mode=append
#[test]
fn todo_23_018() {
	suite0::online_boundary_masks_gqa();
}
```

`src/flash.rs` の `pub fn compile_boundaries(`の定義を次の全体で置き換えます。

```rust file="src/flash.rs" mode="replace" item="pub fn compile_boundaries("
pub fn compile_boundaries(
	g: &crate::Graph,
	outputs: &[usize],
	boundaries: &[Boundary],
	requires_grad: bool,
) -> crate::Result<(crate::Plan, String)> {
	use std::collections::HashSet;
	if requires_grad || boundaries.is_empty() {
		return Ok((
			g.compile(outputs)?,
			"ordinary attention (training/no boundaries)".into(),
		));
	}
	let reachable = g.topology(outputs)?.into_iter().collect::<HashSet<_>>();
	let mut skipped = HashSet::new();
	let mut replacements = std::collections::HashMap::new();
	for boundary in boundaries {
		if !reachable.contains(&boundary.output) {
			continue;
		}
		if [boundary.q, boundary.k, boundary.v, boundary.output]
			.iter()
			.any(|&id| id >= g.nodes.len())
		{
			return Err("attention boundary node".into());
		}
		if g.shape(boundary.q)
			!= [
				boundary.shape.queries,
				boundary.shape.heads,
				boundary.shape.dim,
			] || g.shape(boundary.k)
			!= [
				boundary.shape.keys,
				boundary.shape.kv_heads,
				boundary.shape.dim,
			] || g.shape(boundary.v) != g.shape(boundary.k)
			|| g.shape(boundary.output) != g.shape(boundary.q)
		{
			return Err("attention boundary shape".into());
		}
		if !matches!(g.nodes[boundary.output].op, crate::Op::Stage { .. }) {
			return Ok((
				g.compile(outputs)?,
				"ordinary attention (unstaged output)".into(),
			));
		}
		if ![boundary.q, boundary.k, boundary.v].iter().all(|&id| {
			matches!(
				g.nodes[id].op,
				crate::Op::Stage { .. } | crate::Op::Input { .. }
			)
		}) {
			return Ok((
				g.compile(outputs)?,
				"ordinary attention (unstaged QKV)".into(),
			));
		}
		let before = g
			.topology(&[boundary.q, boundary.k, boundary.v])?
			.into_iter()
			.collect::<HashSet<_>>();
		let inside = g
			.topology(&[boundary.output])?
			.into_iter()
			.filter(|id| !before.contains(id))
			.collect::<HashSet<_>>();
		skipped.extend(boundary_skips(
			g,
			outputs,
			&reachable,
			&inside,
			boundary.output,
		)?);
		let plan = FlashPlan::compile(boundary.shape, boundary.tile)?;
		let begin = plan
			.source
			.find("int chibi_entry")
			.ok_or("native attention entry")?;
		let begin = begin + plan.source[begin..].find('{').unwrap() + 1;
		let end = plan
			.source
			.rfind("return 0;")
			.ok_or("native attention return")?;
		let mut body = plan.source[begin..end]
			.replace(",*valid=buffers[3]", "")
			.replace("valid[j]!=0.0f", "1");
		for (index, id) in [
			(0, boundary.q),
			(1, boundary.k),
			(2, boundary.v),
			(4, boundary.output),
		] {
			body = body.replace(&format!("buffers[{index}]"), &format!("binding_{id}_"));
		}
		for id in [boundary.q, boundary.k, boundary.v, boundary.output] {
			body = body.replace(&format!("binding_{id}_"), &format!("buffers[{id}]"));
		}
		replacements.insert(boundary.output, format!("{{{body}}}\n"));
	}
	let kernels = crate::graph::kernel_bodies(g, outputs, true)?;
	let mut source = crate::graph::C_HELPERS.to_string();
	source += "int chibi_entry(void*const*buffers,const uint64_t*s){\n";
	for id in g.topology(outputs)? {
		source += &format!("float*b{id}=(float*)buffers[{id}];\n");
	}
	for kernel in kernels {
		if skipped.contains(&kernel.output) {
			continue;
		}
		source += replacements.get(&kernel.output).unwrap_or(&kernel.body);
	}
	source += "return 0;\n}\n";
	let plan = crate::Plan::compile_source_specialized(g, outputs, source)?;
	Ok((
		plan,
		format!(
			"online attention boundaries={} pruned kernels={}",
			replacements.len(),
			skipped.len()
		),
	))
}
```

`src/gpu_graph.rs` の `pub fn compile_flash(`の定義を次の全体で置き換えます。

```rust file="src/gpu_graph.rs" mode="replace" item="pub fn compile_flash("
	pub fn compile_flash(
		graph: &Graph,
		outputs: &[usize],
		backend: Backend,
		boundaries: &[crate::flash::Boundary],
		requires_grad: bool,
		local_bytes: usize,
	) -> Result<Self> {
		use std::collections::{HashMap, HashSet};
		if requires_grad {
			return Self::compile(graph, outputs, backend, 64);
		}
		let reachable = graph.topology(outputs)?.into_iter().collect::<HashSet<_>>();
		let mut skipped = HashSet::new();
		let mut replacements = HashMap::new();
		for boundary in boundaries {
			if [boundary.q, boundary.k, boundary.v, boundary.output]
				.iter()
				.any(|&id| id >= graph.nodes.len())
			{
				return Err("GPU attention boundary node".into());
			}
			if graph.shape(boundary.q)
				!= [
					boundary.shape.queries,
					boundary.shape.heads,
					boundary.shape.dim,
				] || graph.shape(boundary.k)
				!= [
					boundary.shape.keys,
					boundary.shape.kv_heads,
					boundary.shape.dim,
				] || graph.shape(boundary.v) != graph.shape(boundary.k)
			{
				return Err("GPU attention boundary shape".into());
			}
			if graph.shape(boundary.output) != graph.shape(boundary.q) {
				return Err("GPU attention boundary output shape".into());
			}
			if !reachable.contains(&boundary.output) {
				continue;
			}
			if crate::flash::select(boundary.shape, false, local_bytes, boundary.tile)
				!= crate::flash::AttentionPath::Online
			{
				return Self::compile(graph, outputs, backend, 64);
			}
			if !matches!(graph.nodes[boundary.output].op, Op::Stage { .. }) {
				return Self::compile(graph, outputs, backend, 64);
			}
			if ![boundary.q, boundary.k, boundary.v]
				.iter()
				.all(|&id| matches!(graph.nodes[id].op, Op::Stage { .. } | Op::Input { .. }))
			{
				return Self::compile(graph, outputs, backend, 64);
			}
			let before = graph
				.topology(&[boundary.q, boundary.k, boundary.v])?
				.into_iter()
				.collect::<HashSet<_>>();
			let inside = graph
				.topology(&[boundary.output])?
				.into_iter()
				.filter(|id| !before.contains(id))
				.collect::<HashSet<_>>();
			skipped.extend(crate::flash::boundary_skips(
				graph,
				outputs,
				&reachable,
				&inside,
				boundary.output,
			)?);
			let kernel = crate::flash::shader(backend.dialect(), boundary.shape, boundary.tile)?;
			replacements.insert(
				boundary.output,
				(
					kernel,
					vec![
						boundary.q,
						boundary.k,
						boundary.v,
						usize::MAX,
						boundary.output,
					],
				),
			);
		}
		let ordinary = lower(graph, outputs, backend.dialect(), 64)?;
		let mut lowered = vec![];
		for (kernel, slots) in ordinary {
			let id = *slots.last().ok_or("attention output")?;
			if skipped.contains(&id) {
				continue;
			}
			lowered.push(replacements.remove(&id).unwrap_or((kernel, slots)));
		}
		Self::from_lowered(graph, outputs, backend, lowered, true)
	}
```

```sh check
cargo test --test book_23 todo_23_018 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 学習 graph はこの章では通常版へ固定し、forward-only kernel を微分可能と扱わない。

<!-- todo:23-019 -->

実装は[この章の `flash::reference、FlashPlan::compile、FlashPlan::run、flash::shader、flash::register、compile_boundaries`](#flash-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_23.rs mode=append
#[test]
fn todo_23_019() {
	suite0::all_masked_and_late_max();
}
```

`src/flash.rs` の `pub fn compile_boundaries(`の定義を次の全体で置き換えます。

```rust file="src/flash.rs" mode="replace" item="pub fn compile_boundaries("
pub fn compile_boundaries(
	g: &crate::Graph,
	outputs: &[usize],
	boundaries: &[Boundary],
	requires_grad: bool,
) -> crate::Result<(crate::Plan, String)> {
	use std::collections::HashSet;
	if requires_grad || boundaries.is_empty() {
		return Ok((
			g.compile(outputs)?,
			"ordinary attention (training/no boundaries)".into(),
		));
	}
	let reachable = g.topology(outputs)?.into_iter().collect::<HashSet<_>>();
	let mut skipped = HashSet::new();
	let mut replacements = std::collections::HashMap::new();
	for boundary in boundaries {
		if !reachable.contains(&boundary.output) {
			continue;
		}
		if [boundary.q, boundary.k, boundary.v, boundary.output]
			.iter()
			.any(|&id| id >= g.nodes.len())
		{
			return Err("attention boundary node".into());
		}
		if g.shape(boundary.q)
			!= [
				boundary.shape.queries,
				boundary.shape.heads,
				boundary.shape.dim,
			] || g.shape(boundary.k)
			!= [
				boundary.shape.keys,
				boundary.shape.kv_heads,
				boundary.shape.dim,
			] || g.shape(boundary.v) != g.shape(boundary.k)
			|| g.shape(boundary.output) != g.shape(boundary.q)
		{
			return Err("attention boundary shape".into());
		}
		if !matches!(g.nodes[boundary.output].op, crate::Op::Stage { .. }) {
			return Ok((
				g.compile(outputs)?,
				"ordinary attention (unstaged output)".into(),
			));
		}
		if ![boundary.q, boundary.k, boundary.v].iter().all(|&id| {
			matches!(
				g.nodes[id].op,
				crate::Op::Stage { .. } | crate::Op::Input { .. }
			)
		}) {
			return Ok((
				g.compile(outputs)?,
				"ordinary attention (unstaged QKV)".into(),
			));
		}
		let before = g
			.topology(&[boundary.q, boundary.k, boundary.v])?
			.into_iter()
			.collect::<HashSet<_>>();
		let inside = g
			.topology(&[boundary.output])?
			.into_iter()
			.filter(|id| !before.contains(id))
			.collect::<HashSet<_>>();
		skipped.extend(boundary_skips(
			g,
			outputs,
			&reachable,
			&inside,
			boundary.output,
		)?);
		let plan = FlashPlan::compile(boundary.shape, boundary.tile)?;
		let begin = plan
			.source
			.find("int chibi_entry")
			.ok_or("native attention entry")?;
		let begin = begin + plan.source[begin..].find('{').unwrap() + 1;
		let end = plan
			.source
			.rfind("return 0;")
			.ok_or("native attention return")?;
		let mut body = plan.source[begin..end]
			.replace(",*valid=buffers[3]", "")
			.replace("valid[j]!=0.0f", "1");
		for (index, id) in [
			(0, boundary.q),
			(1, boundary.k),
			(2, boundary.v),
			(4, boundary.output),
		] {
			body = body.replace(&format!("buffers[{index}]"), &format!("binding_{id}_"));
		}
		for id in [boundary.q, boundary.k, boundary.v, boundary.output] {
			body = body.replace(&format!("binding_{id}_"), &format!("buffers[{id}]"));
		}
		replacements.insert(boundary.output, format!("{{{body}}}\n"));
	}
	let kernels = crate::graph::kernel_bodies(g, outputs, true)?;
	let mut source = crate::graph::C_HELPERS.to_string();
	source += "int chibi_entry(void*const*buffers,const uint64_t*s){\n";
	for id in g.topology(outputs)? {
		source += &format!("float*b{id}=(float*)buffers[{id}];\n");
	}
	for kernel in kernels {
		if skipped.contains(&kernel.output) {
			continue;
		}
		source += replacements.get(&kernel.output).unwrap_or(&kernel.body);
	}
	source += "return 0;\n}\n";
	let plan = crate::Plan::compile_source_specialized(g, outputs, source)?;
	Ok((
		plan,
		format!(
			"online attention boundaries={} pruned kernels={}",
			replacements.len(),
			skipped.len()
		),
	))
}
```

`src/gpu_graph.rs` の `pub fn compile_flash(`の定義を次の全体で置き換えます。

```rust file="src/gpu_graph.rs" mode="replace" item="pub fn compile_flash("
	pub fn compile_flash(
		graph: &Graph,
		outputs: &[usize],
		backend: Backend,
		boundaries: &[crate::flash::Boundary],
		requires_grad: bool,
		local_bytes: usize,
	) -> Result<Self> {
		use std::collections::{HashMap, HashSet};
		if requires_grad {
			return Self::compile(graph, outputs, backend, 64);
		}
		let reachable = graph.topology(outputs)?.into_iter().collect::<HashSet<_>>();
		let mut skipped = HashSet::new();
		let mut replacements = HashMap::new();
		for boundary in boundaries {
			if [boundary.q, boundary.k, boundary.v, boundary.output]
				.iter()
				.any(|&id| id >= graph.nodes.len())
			{
				return Err("GPU attention boundary node".into());
			}
			if graph.shape(boundary.q)
				!= [
					boundary.shape.queries,
					boundary.shape.heads,
					boundary.shape.dim,
				] || graph.shape(boundary.k)
				!= [
					boundary.shape.keys,
					boundary.shape.kv_heads,
					boundary.shape.dim,
				] || graph.shape(boundary.v) != graph.shape(boundary.k)
			{
				return Err("GPU attention boundary shape".into());
			}
			if graph.shape(boundary.output) != graph.shape(boundary.q) {
				return Err("GPU attention boundary output shape".into());
			}
			if !reachable.contains(&boundary.output) {
				continue;
			}
			if crate::flash::select(boundary.shape, false, local_bytes, boundary.tile)
				!= crate::flash::AttentionPath::Online
			{
				return Self::compile(graph, outputs, backend, 64);
			}
			if !matches!(graph.nodes[boundary.output].op, Op::Stage { .. }) {
				return Self::compile(graph, outputs, backend, 64);
			}
			if ![boundary.q, boundary.k, boundary.v]
				.iter()
				.all(|&id| matches!(graph.nodes[id].op, Op::Stage { .. } | Op::Input { .. }))
			{
				return Self::compile(graph, outputs, backend, 64);
			}
			let before = graph
				.topology(&[boundary.q, boundary.k, boundary.v])?
				.into_iter()
				.collect::<HashSet<_>>();
			let inside = graph
				.topology(&[boundary.output])?
				.into_iter()
				.filter(|id| !before.contains(id))
				.collect::<HashSet<_>>();
			skipped.extend(crate::flash::boundary_skips(
				graph,
				outputs,
				&reachable,
				&inside,
				boundary.output,
			)?);
			let kernel = crate::flash::shader(backend.dialect(), boundary.shape, boundary.tile)?;
			replacements.insert(
				boundary.output,
				(
					kernel,
					vec![
						boundary.q,
						boundary.k,
						boundary.v,
						usize::MAX,
						boundary.output,
					],
				),
			);
		}
		let ordinary = lower(graph, outputs, backend.dialect(), 64)?;
		let mut lowered = vec![];
		for (kernel, slots) in ordinary {
			let id = *slots.last().ok_or("attention output")?;
			if skipped.contains(&id) {
				continue;
			}
			lowered.push(replacements.remove(&id).unwrap_or((kernel, slots)));
		}
		Self::from_lowered(graph, outputs, backend, lowered, true)
	}
```

```sh check
cargo test --test book_23 todo_23_019 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## model の logits と cache 有効長を通常版に照合してから、計測用の設定を追加する。

<!-- todo:23-020 -->

実装は[この章の `flash::reference、FlashPlan::compile、FlashPlan::run、flash::shader、flash::register、compile_boundaries`](#flash-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_23.rs mode=append
#[test]
fn todo_23_020() {
	suite0::model_boundary_compiler_replaces_actual_graph();
}
```

```sh check
cargo test --test book_23 todo_23_020 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 基準例の出力は期待値と一致する。

<!-- todo:23-021 -->

実装は[この章の `flash::reference、FlashPlan::compile、FlashPlan::run、flash::shader、flash::register、compile_boundaries`](#flash-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_23.rs mode=append
#[test]
fn todo_23_021() {
	suite0::native_c_and_composite_graph();
}
```

```sh check
cargo test --test book_23 todo_23_021 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## CPU、GPU、通常版を f64 reference に対し `abs_err <= 1e-4 + 1e-4*abs(ref)` で比較する。

<!-- todo:23-022 -->

実装は[この章の `flash::reference、FlashPlan::compile、FlashPlan::run、flash::shader、flash::register、compile_boundaries`](#flash-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_23.rs mode=append
#[test]
fn todo_23_022() {
	suite0::online_boundary_masks_gqa();
}
```

```sh check
cargo test --test book_23 todo_23_022 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## query/key 長の1、tile直前、tile一致、tile直後と、GQA 比率1、2を検証する。

<!-- todo:23-023 -->

実装は[この章の `flash::reference、FlashPlan::compile、FlashPlan::run、flash::shader、flash::register、compile_boundaries`](#flash-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_23.rs mode=append
#[test]
fn todo_23_023() {
	suite0::all_masked_and_late_max();
}
```

```sh check
cargo test --test book_23 todo_23_023 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 全 mask 行、空の有効 key、後半 tile が最大の score、極端な有限 score を試す。

<!-- todo:23-024 -->

実装は[この章の `flash::reference、FlashPlan::compile、FlashPlan::run、flash::shader、flash::register、compile_boundaries`](#flash-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_23.rs mode=append
#[test]
fn todo_23_024() {
	suite0::model_boundary_compiler_replaces_actual_graph();
}
```

```sh check
cargo test --test book_23 todo_23_024 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 無効領域の NaN が有効出力へ流れず、全 mask 行は0になって NaN を含まない。

<!-- todo:23-025 -->

実装は[この章の `flash::reference、FlashPlan::compile、FlashPlan::run、flash::shader、flash::register、compile_boundaries`](#flash-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_23.rs mode=append
#[test]
fn todo_23_025() {
	suite0::native_c_and_composite_graph();
}
```

```sh check
cargo test --test book_23 todo_23_025 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## dump で全 lane の barrier と、local buffer の再利用依存を確認する。

<!-- todo:23-026 -->

実装は[この章の `flash::reference、FlashPlan::compile、FlashPlan::run、flash::shader、flash::register、compile_boundaries`](#flash-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_23.rs mode=append
#[test]
fn todo_23_026() {
	suite0::online_boundary_masks_gqa();
}
```

```sh check
cargo test --test book_23 todo_23_026 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 一つの学習 step が通常版の forward と backward を通り、未定義 gradient を隠さない。

<!-- todo:23-027 -->

実装は[この章の `flash::reference、FlashPlan::compile、FlashPlan::run、flash::shader、flash::register、compile_boundaries`](#flash-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_23.rs mode=append
#[test]
fn todo_23_027() {
	suite0::all_masked_and_late_max();
}
```

```sh check
cargo test --test book_23 todo_23_027 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 実機を使っていない backend と、未測定の速度を確認済みとして記述しない。

<!-- todo:23-028 -->

実装は[この章の `flash::reference、FlashPlan::compile、FlashPlan::run、flash::shader、flash::register、compile_boundaries`](#flash-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_23.rs mode=append
#[test]
fn todo_23_028() {
	suite0::model_boundary_compiler_replaces_actual_graph();
}
```

```sh check
cargo test --test book_23 todo_23_028 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 確認記録

```sh check
cargo test
```

GPU の source は CPU の numerical test と分けて扱う。上の CPU 成功は GPU の実機確認を意味しない。

