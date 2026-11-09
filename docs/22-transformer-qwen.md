# 22. 小さな Transformer を通し、実際の Qwen checkpoint で次の token を計算する

パラメータを全層の名前と shape で登録し、embedding、pre-norm、GQA attention、half-split RoPE、SiLU gate、残差、最終 norm と出力を通常の Graph で結ぶ。小 fixture と全24層の公開設定は別に検査する。KV cache は固定容量の読み取り専用入力と新規出力を使う。

以下は置換・追記先を指定した実装全体である。`mode=write` はそのファイル全体を置き換え、`mode=append` は末尾へ追記する。テストは節ごとに追加し、同じ名前の関数を重複して貼らない。生成 C と共有ライブラリは runtime の一時ディレクトリに作られる。

公開 Qwen の重みを使う数値比較と全 parameter 学習は、この作業では未実行である。全24層の登録、通常 attention の小モデル、勾配と再開は CPU テストで検証する。外部資産や実機がない検証は `NOT_RUN` と記録する。

公式参照は [固定 commit](https://huggingface.co/Qwen/Qwen2.5-0.5B/commit/b937cc073b55e980e6f21d5f8b7a77818efcf0eb)、[公開 config](https://huggingface.co/Qwen/Qwen2.5-0.5B/raw/b937cc073b55e980e6f21d5f8b7a77818efcf0eb/config.json)、[Qwen2 実装 v4.46.3](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/qwen2/modeling_qwen2.py) である。manifest の小ファイル SHA は取得した実データから計算し、重み SHA は公式 LFS metadata に対応する。tokenizer fixture は固定 tokenizer.json を tokenizers 0.22.2 で実行した ID である。

## 1. token ID の範囲を検査し、embedding の行選択を INDEX と LOAD へ落とす。

<!-- todo:22-001 -->

まず、この章が追加するモジュールと共有テスト補助を保存する。後続の各節は、ここに掲載した `nn::attention::{embedding,rms_norm,apply_rope,attention}` と `nn::qwen::{logits_all,parameter_registry,KvState::adopt}` の規則を、焦点を絞ったテストで確認する。

```rust file=src/lib.rs mode=append
pub mod model_assets;
```

```toml file=Cargo.toml mode=append
[dependencies.tokenizers]
version = "=0.22.2"
default-features = false
features = ["fancy-regex"]
```

```rust file=src/nn/mod.rs mode=write
pub mod attention;
pub mod qwen;
pub mod transformer;
use crate::{Binary, Graph, NodeId, Reduce, Result, Unary};
pub fn linear(g: &mut Graph, x: NodeId, weight: NodeId, bias: Option<NodeId>) -> Result<NodeId> {
	let transposed = g.permute(weight, vec![1, 0])?;
	let y = g.matmul(x, transposed)?;
	match bias {
		Some(b) => g.binary(Binary::Add, y, b),
		None => Ok(y),
	}
}
pub fn relu(g: &mut Graph, x: NodeId) -> Result<NodeId> {
	let zero = g.constant(0., vec![])?;
	let positive = g.binary(Binary::Lt, zero, x)?;
	g.where_(positive, x, zero)
}
pub fn silu(g: &mut Graph, x: NodeId) -> Result<NodeId> {
	let neg = g.unary(Unary::Neg, x)?;
	let exp = g.unary(Unary::Exp, neg)?;
	let one = g.constant(1., vec![])?;
	let denom = g.binary(Binary::Add, one, exp)?;
	g.binary(Binary::Div, x, denom)
}
pub fn log_softmax(g: &mut Graph, x: NodeId, axis: usize) -> Result<NodeId> {
	let m = g.reduce(x, Reduce::Max, vec![axis], true)?;
	let m = g.stop_gradient(m)?;
	let shift = g.binary(Binary::Sub, x, m)?;
	let ex = g.unary(Unary::Exp, shift)?;
	let sum = g.reduce(ex, Reduce::Sum, vec![axis], true)?;
	let log = g.unary(Unary::Log, sum)?;
	g.binary(Binary::Sub, shift, log)
}
pub fn cross_entropy(
	g: &mut Graph,
	logits: NodeId,
	targets: &[usize],
	valid: &[bool],
) -> Result<NodeId> {
	let shape = g.shape(logits).to_vec();
	if shape.len() != 2 || targets.len() != shape[0] || valid.len() != targets.len() {
		return Err("cross entropy shape".into());
	}
	if targets.iter().any(|&t| t >= shape[1]) {
		return Err("target outside vocabulary".into());
	}
	let count = valid.iter().filter(|&&v| v).count();
	if count == 0 {
		return Err("no valid target positions".into());
	}
	let lp = log_softmax(g, logits, 1)?;
	let mut losses = Vec::new();
	for (i, (&target, &use_it)) in targets.iter().zip(valid).enumerate() {
		if use_it {
			let p = g.shrink(lp, vec![(i, i + 1), (target, target + 1)])?;
			losses.push(g.reshape(p, vec![])?);
		}
	}
	let all = g.stack(losses, 0)?;
	let sum = g.reduce(all, Reduce::Sum, vec![0], false)?;
	let scale = g.constant(-1. / count as f32, vec![])?;
	g.binary(Binary::Mul, sum, scale)
}
pub fn mse(g: &mut Graph, prediction: NodeId, target: NodeId) -> Result<NodeId> {
	let difference = g.binary(Binary::Sub, prediction, target)?;
	let squared = g.binary(Binary::Mul, difference, difference)?;
	let shape = g.shape(squared).to_vec();
	let n: usize = shape.iter().product();
	let sum = g.reduce(squared, Reduce::Sum, (0..shape.len()).collect(), false)?;
	let scale = g.constant(1. / n as f32, vec![])?;
	g.binary(Binary::Mul, sum, scale)
}
pub fn cross_entropy_runtime(
	g: &mut Graph,
	logits: NodeId,
	targets: NodeId,
	valid: NodeId,
) -> Result<NodeId> {
	let shape = g.shape(logits).to_vec();
	if shape.len() != 2 || g.shape(targets) != [shape[0]] || g.shape(valid) != [shape[0]] {
		return Err("runtime cross entropy shape".into());
	}
	let n = shape[0];
	let total = n.checked_mul(shape[1]).ok_or("logit size overflow")?;
	if total > 16_777_216 {
		return Err("flattened token indices exceed exact f32 integer range".into());
	}
	let offsets = (0..n)
		.map(|i| g.constant((i * shape[1]) as f32, vec![]))
		.collect::<Result<Vec<_>>>()?;
	let offsets = g.stack(offsets, 0)?;
	let indices = g.binary(Binary::Add, targets, offsets)?;
	let lp = log_softmax(g, logits, 1)?;
	let flat = g.reshape(
		lp,
		vec![n.checked_mul(shape[1]).ok_or("logit size overflow")?],
	)?;
	let selected = g.gather(flat, indices, 0)?;
	let weighted = g.binary(Binary::Mul, selected, valid)?;
	let sum = g.reduce(weighted, Reduce::Sum, vec![0], false)?;
	let count = g.reduce(valid, Reduce::Sum, vec![0], false)?;
	let mean = g.binary(Binary::Div, sum, count)?;
	g.unary(Unary::Neg, mean)
}
```

```rust file=src/nn/attention.rs mode=write
use crate::{Binary, Graph, NodeId, Reduce, Result, Unary};
pub fn embedding(g: &mut Graph, weight: NodeId, tokens: NodeId) -> Result<NodeId> {
	g.gather(weight, tokens, 0)
}
pub fn rms_norm(g: &mut Graph, x: NodeId, weight: NodeId, eps: f32) -> Result<NodeId> {
	let dim = *g
		.shape(x)
		.last()
		.ok_or("RMSNorm requires a final dimension")?;
	if dim == 0 || g.shape(weight) != [dim] || eps <= 0. || !eps.is_finite() {
		return Err("RMSNorm shape/epsilon".into());
	}
	let axis = g.shape(x).len() - 1;
	let square = g.binary(Binary::Mul, x, x)?;
	let sum = g.reduce(square, Reduce::Sum, vec![axis], true)?;
	let divisor = g.constant(dim as f32, vec![])?;
	let mean = g.binary(Binary::Div, sum, divisor)?;
	let eps = g.constant(eps, vec![])?;
	let variance = g.binary(Binary::Add, mean, eps)?;
	let root = g.unary(Unary::Sqrt, variance)?;
	let norm = g.binary(Binary::Div, x, root)?;
	g.binary(Binary::Mul, norm, weight)
}
pub fn softmax(g: &mut Graph, x: NodeId, axis: usize) -> Result<NodeId> {
	let log = super::log_softmax(g, x, axis)?;
	g.unary(Unary::Exp, log)
}
/// Qwen uses half-split RoPE, pairing r with r+D/2.
pub fn apply_rope(g: &mut Graph, x: NodeId, start: usize, theta: f64) -> Result<NodeId> {
	let shape = g.shape(x).to_vec();
	if shape.len() != 3 || shape[2] % 2 != 0 || shape[2] == 0 || !theta.is_finite() || theta <= 0. {
		return Err("RoPE shape/theta".into());
	}
	let (t, h, d) = (shape[0], shape[1], shape[2]);
	let half = d / 2;
	let mut positions = Vec::new();
	for pos in 0..t {
		let absolute = start.checked_add(pos).ok_or("RoPE position overflow")?;
		let mut dimensions = Vec::new();
		for channel in 0..d {
			let r = channel % half;
			let angle = absolute as f64 / theta.powf(2. * r as f64 / d as f64);
			let (c, s) = (angle.cos() as f32, angle.sin() as f32);
			let a = g.shrink(x, vec![(pos, pos + 1), (0, h), (r, r + 1)])?;
			let b = g.shrink(x, vec![(pos, pos + 1), (0, h), (r + half, r + half + 1)])?;
			let a = g.reshape(a, vec![h])?;
			let b = g.reshape(b, vec![h])?;
			let cc = g.constant(c, vec![])?;
			let ss = g.constant(s, vec![])?;
			let output = if channel < half {
				let ac = g.binary(Binary::Mul, a, cc)?;
				let bs = g.binary(Binary::Mul, b, ss)?;
				g.binary(Binary::Sub, ac, bs)?
			} else {
				let as_ = g.binary(Binary::Mul, a, ss)?;
				let bc = g.binary(Binary::Mul, b, cc)?;
				g.binary(Binary::Add, as_, bc)?
			};
			dimensions.push(output);
		}
		positions.push(g.stack(dimensions, 1)?);
	}
	g.stack(positions, 0)
}
pub fn attention(
	g: &mut Graph,
	q: NodeId,
	k: NodeId,
	v: NodeId,
	query_start: usize,
	causal: bool,
) -> Result<NodeId> {
	let qs = g.shape(q).to_vec();
	let ks = g.shape(k).to_vec();
	if qs.len() != 3
		|| ks.len() != 3
		|| g.shape(v) != ks
		|| qs[2] != ks[2]
		|| ks[1] == 0
		|| qs[1] % ks[1] != 0
		|| qs[2] == 0
		|| ks[0] == 0
	{
		return Err("attention shape or GQA ratio".into());
	}
	let (t, h, d, s, hkv) = (qs[0], qs[1], qs[2], ks[0], ks[1]);
	let ratio = h / hkv;
	let mut heads = Vec::new();
	for head in 0..h {
		let kh = head / ratio;
		let qq = g.shrink(q, vec![(0, t), (head, head + 1), (0, d)])?;
		let qq = g.reshape(qq, vec![t, d])?;
		let kk = g.shrink(k, vec![(0, s), (kh, kh + 1), (0, d)])?;
		let kk = g.reshape(kk, vec![s, d])?;
		let kt = g.permute(kk, vec![1, 0])?;
		let scores = g.matmul(qq, kt)?;
		let scale = g.constant(1. / (d as f32).sqrt(), vec![])?;
		let scores = g.binary(Binary::Mul, scores, scale)?;
		let mut mask_values = Vec::with_capacity(t * s);
		for row in 0..t {
			let position = query_start
				.checked_add(row)
				.ok_or("query position overflow")?;
			for col in 0..s {
				mask_values.push(if !causal || col <= position {
					0.
				} else {
					f32::NEG_INFINITY
				});
			}
		}
		let mask = constant_array(g, vec![t, s], &mask_values)?;
		let scores = g.binary(Binary::Add, scores, mask)?;
		let probabilities = softmax(g, scores, 1)?;
		let vv = g.shrink(v, vec![(0, s), (kh, kh + 1), (0, d)])?;
		let vv = g.reshape(vv, vec![s, d])?;
		heads.push(g.matmul(probabilities, vv)?);
	}
	g.stack(heads, 1)
}
pub fn constant_array(g: &mut Graph, shape: Vec<usize>, values: &[f32]) -> Result<NodeId> {
	let n = shape.iter().try_fold(1usize, |a, &b| {
		a.checked_mul(b).ok_or("constant shape overflow")
	})?;
	if n != values.len() || n == 0 {
		return Err("constant array length".into());
	}
	let ids = values
		.iter()
		.map(|&v| g.constant(v, vec![]))
		.collect::<Result<Vec<_>>>()?;
	let all = g.stack(ids, 0)?;
	g.reshape(all, shape)
}
pub fn concatenate(g: &mut Graph, a: NodeId, b: NodeId, axis: usize) -> Result<NodeId> {
	let sa = g.shape(a).to_vec();
	let sb = g.shape(b).to_vec();
	if sa.len() != sb.len()
		|| axis >= sa.len()
		|| sa
			.iter()
			.zip(&sb)
			.enumerate()
			.any(|(i, (a, b))| i != axis && a != b)
	{
		return Err("concat shape".into());
	}
	let mut pa = vec![(0, 0); sa.len()];
	let mut pb = pa.clone();
	pa[axis] = (0, sb[axis]);
	pb[axis] = (sa[axis], 0);
	let a = g.pad(a, pa)?;
	let b = g.pad(b, pb)?;
	g.binary(Binary::Add, a, b)
}
```

```rust file=src/nn/qwen.rs mode=write
use super::{
	attention::{apply_rope, attention, concatenate, rms_norm},
	linear, silu,
};
use crate::{
	Binary, Buffer, Graph, NodeId, Result,
	state::{DType, Registry, Spec, TensorRecord, TensorSet},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QwenConfig {
	pub model_type: String,
	pub hidden_size: usize,
	pub intermediate_size: usize,
	pub num_hidden_layers: usize,
	pub num_attention_heads: usize,
	pub num_key_value_heads: usize,
	pub vocab_size: usize,
	pub max_position_embeddings: usize,
	pub rope_theta: f64,
	pub rms_norm_eps: f32,
	pub tie_word_embeddings: bool,
	pub hidden_act: String,
	#[serde(default)]
	pub architectures: Vec<String>,
	#[serde(default)]
	pub attention_dropout: f32,
	#[serde(default)]
	pub use_mrope: bool,
	#[serde(default)]
	pub use_sliding_window: bool,
	#[serde(default)]
	pub rope_scaling: Option<serde_json::Value>,
}
impl QwenConfig {
	pub fn tiny() -> Self {
		Self {
			model_type: "qwen2".into(),
			hidden_size: 8,
			intermediate_size: 16,
			num_hidden_layers: 1,
			num_attention_heads: 2,
			num_key_value_heads: 1,
			vocab_size: 16,
			max_position_embeddings: 8,
			rope_theta: 1_000_000.,
			rms_norm_eps: 1e-6,
			tie_word_embeddings: true,
			hidden_act: "silu".into(),
			architectures: vec!["Qwen2ForCausalLM".into()],
			attention_dropout: 0.,
			use_mrope: false,
			use_sliding_window: false,
			rope_scaling: None,
		}
	}
	pub fn validate(&self) -> Result<()> {
		if self.use_mrope
			|| self.attention_dropout != 0.
			|| (!self.architectures.is_empty() && self.architectures != ["Qwen2ForCausalLM"])
			|| self.model_type != "qwen2"
			|| self.hidden_act != "silu"
			|| self.use_sliding_window
			|| self.rope_scaling.is_some()
		{
			return Err("unsupported Qwen architecture/activation/rope/sliding window".into());
		}
		if [
			self.hidden_size,
			self.intermediate_size,
			self.num_hidden_layers,
			self.num_attention_heads,
			self.num_key_value_heads,
			self.vocab_size,
			self.max_position_embeddings,
		]
		.contains(&0)
			|| self.hidden_size % self.num_attention_heads != 0
			|| self.num_attention_heads % self.num_key_value_heads != 0
			|| (self.hidden_size / self.num_attention_heads) % 2 != 0
			|| self.vocab_size > 16_777_216
			|| self.rms_norm_eps <= 0.
			|| !self.rms_norm_eps.is_finite()
			|| self.rope_theta <= 0.
			|| !self.rope_theta.is_finite()
		{
			return Err("invalid Qwen dimensions/norm/theta".into());
		}
		Ok(())
	}
	pub fn require_public_05b(&self) -> Result<()> {
		self.validate()?;
		if (
			self.num_hidden_layers,
			self.hidden_size,
			self.intermediate_size,
			self.num_attention_heads,
			self.num_key_value_heads,
			self.vocab_size,
		) != (24, 896, 4864, 14, 2, 151936)
			|| !self.tie_word_embeddings
		{
			return Err("expected complete Qwen2.5-0.5B config".into());
		}
		Ok(())
	}
}
pub fn parameter_registry(c: &QwenConfig) -> Result<Registry> {
	c.validate()?;
	let mut registry = Registry::default();
	let mut id = 0u32;
	let mut register = |name: String, shape: Vec<usize>| -> Result<()> {
		registry.register(
			&name,
			Spec {
				id,
				slot: id as usize,
				dtype: DType::F32,
				shape,
				trainable: true,
			},
		)?;
		id = id.checked_add(1).ok_or("parameter ID overflow")?;
		Ok(())
	};
	let d = c.hidden_size;
	let kd = c.num_key_value_heads * (d / c.num_attention_heads);
	register("model.embed_tokens.weight".into(), vec![c.vocab_size, d])?;
	for layer in 0..c.num_hidden_layers {
		let p = format!("model.layers.{layer}");
		register(format!("{p}.input_layernorm.weight"), vec![d])?;
		for (name, out) in [("q_proj", d), ("k_proj", kd), ("v_proj", kd)] {
			register(format!("{p}.self_attn.{name}.weight"), vec![out, d])?;
			register(format!("{p}.self_attn.{name}.bias"), vec![out])?;
		}
		register(format!("{p}.self_attn.o_proj.weight"), vec![d, d])?;
		register(format!("{p}.post_attention_layernorm.weight"), vec![d])?;
		register(
			format!("{p}.mlp.gate_proj.weight"),
			vec![c.intermediate_size, d],
		)?;
		register(
			format!("{p}.mlp.up_proj.weight"),
			vec![c.intermediate_size, d],
		)?;
		register(
			format!("{p}.mlp.down_proj.weight"),
			vec![d, c.intermediate_size],
		)?;
	}
	register("model.norm.weight".into(), vec![d])?;
	if !c.tie_word_embeddings {
		register("lm_head.weight".into(), vec![c.vocab_size, d])?;
	}
	Ok(registry)
}
pub fn register_graph(g: &mut Graph, registry: &Registry) -> Result<BTreeMap<String, NodeId>> {
	registry
		.names
		.iter()
		.map(|(name, spec)| Ok((name.clone(), g.input(spec.slot, spec.shape.clone())?)))
		.collect()
}
pub fn load_parameters(
	c: &QwenConfig,
	loaded: &BTreeMap<String, TensorRecord>,
) -> Result<TensorSet> {
	let registry = parameter_registry(c)?;
	let mut converted = BTreeMap::new();
	for (name, tensor) in loaded {
		converted.insert(
			name.clone(),
			TensorRecord::from_buffer(&tensor.to_buffer()?),
		);
	}
	if c.tie_word_embeddings {
		if let Some(output) = converted.remove("lm_head.weight") {
			if converted.get("model.embed_tokens.weight") != Some(&output) {
				return Err("tied embedding/output payload mismatch".into());
			}
		}
	}
	registry.validate(&converted, false)
}
#[derive(Debug)]
pub struct Forward {
	pub logits: NodeId,
	pub boundaries: Vec<NodeId>,
	pub cache_outputs: Vec<(NodeId, NodeId)>,
}
pub fn logits_all(
	g: &mut Graph,
	c: &QwenConfig,
	params: &BTreeMap<String, NodeId>,
	tokens: NodeId,
	start: usize,
	cache: Option<&[(NodeId, NodeId)]>,
) -> Result<Forward> {
	logits_all_with_attention(g, c, params, tokens, start, cache, &mut attention)
}
pub fn logits_all_with_attention<F>(
	g: &mut Graph,
	c: &QwenConfig,
	params: &BTreeMap<String, NodeId>,
	tokens: NodeId,
	start: usize,
	cache: Option<&[(NodeId, NodeId)]>,
	attention_builder: &mut F,
) -> Result<Forward>
where
	F: FnMut(&mut Graph, NodeId, NodeId, NodeId, usize, bool) -> Result<NodeId>,
{
	c.validate()?;
	let shape = g.shape(tokens);
	if shape.len() != 1 || shape[0] == 0 {
		return Err("tokens must be nonempty rank one".into());
	}
	let t = shape[0];
	if start.checked_add(t).ok_or("context overflow")? > c.max_position_embeddings {
		return Err("input exceeds context capacity".into());
	}
	if cache.is_none() && start != 0 {
		return Err("nonzero position requires a cache".into());
	}
	if cache.is_some_and(|v| v.len() != c.num_hidden_layers) {
		return Err("cache layer count".into());
	}
	let get = |name: &str| {
		params
			.get(name)
			.copied()
			.ok_or_else(|| format!("missing graph parameter {name}"))
	};
	let d = c.hidden_size;
	let hd = d / c.num_attention_heads;
	let kd = c.num_key_value_heads;
	let mut x = super::attention::embedding(g, get("model.embed_tokens.weight")?, tokens)?;
	let mut boundaries = vec![x];
	let mut cache_outputs = Vec::new();
	for layer in 0..c.num_hidden_layers {
		let p = format!("model.layers.{layer}");
		let n = rms_norm(
			g,
			x,
			get(&format!("{p}.input_layernorm.weight"))?,
			c.rms_norm_eps,
		)?;
		let projection = |name: &str, heads: usize, g: &mut Graph| -> Result<NodeId> {
			let value = linear(
				g,
				n,
				get(&format!("{p}.self_attn.{name}.weight"))?,
				Some(get(&format!("{p}.self_attn.{name}.bias"))?),
			)?;
			g.reshape(value, vec![t, heads, hd])
		};
		let q = projection("q_proj", c.num_attention_heads, g)?;
		let k = projection("k_proj", kd, g)?;
		let v = projection("v_proj", kd, g)?;
		let q = apply_rope(g, q, start, c.rope_theta)?;
		let k = apply_rope(g, k, start, c.rope_theta)?;
		let (k, v) = if let Some(old) = cache {
			let (ck, cv) = old[layer];
			let expected = [c.max_position_embeddings, kd, hd];
			if g.shape(ck) != expected || g.shape(cv) != expected {
				return Err("cache capacity shape".into());
			}
			let (k, v) = if start == 0 {
				(k, v)
			} else {
				let ranges = vec![(0, start), (0, kd), (0, hd)];
				let prefix_k = g.shrink(ck, ranges.clone())?;
				let prefix_v = g.shrink(cv, ranges)?;
				(
					concatenate(g, prefix_k, k, 0)?,
					concatenate(g, prefix_v, v, 0)?,
				)
			};
			let pads = vec![(0, c.max_position_embeddings - start - t), (0, 0), (0, 0)];
			let new_k = g.pad(k, pads.clone())?;
			let new_v = g.pad(v, pads)?;
			let new_k = g.stage(new_k)?;
			let new_v = g.stage(new_v)?;
			cache_outputs.push((new_k, new_v));
			let ranges = vec![(0, start + t), (0, kd), (0, hd)];
			(g.shrink(new_k, ranges.clone())?, g.shrink(new_v, ranges)?)
		} else {
			(k, v)
		};
		let attended = attention_builder(g, q, k, v, start, true)?;
		let attended = g.reshape(attended, vec![t, d])?;
		let projected = linear(
			g,
			attended,
			get(&format!("{p}.self_attn.o_proj.weight"))?,
			None,
		)?;
		x = g.binary(Binary::Add, x, projected)?;
		let n = rms_norm(
			g,
			x,
			get(&format!("{p}.post_attention_layernorm.weight"))?,
			c.rms_norm_eps,
		)?;
		let gate = linear(g, n, get(&format!("{p}.mlp.gate_proj.weight"))?, None)?;
		let gate = silu(g, gate)?;
		let up = linear(g, n, get(&format!("{p}.mlp.up_proj.weight"))?, None)?;
		let product = g.binary(Binary::Mul, gate, up)?;
		let down = linear(g, product, get(&format!("{p}.mlp.down_proj.weight"))?, None)?;
		x = g.binary(Binary::Add, x, down)?;
		x = g.stage(x)?;
		boundaries.push(x);
	}
	let x = rms_norm(g, x, get("model.norm.weight")?, c.rms_norm_eps)?;
	boundaries.push(x);
	let output = get(if c.tie_word_embeddings {
		"model.embed_tokens.weight"
	} else {
		"lm_head.weight"
	})?;
	let logits = linear(g, x, output, None)?;
	Ok(Forward {
		logits,
		boundaries,
		cache_outputs,
	})
}
#[derive(Clone, Debug)]
pub struct KvState {
	pub keys: Vec<Buffer>,
	pub values: Vec<Buffer>,
	pub valid: usize,
	pub capacity: usize,
}
impl KvState {
	pub fn new(c: &QwenConfig) -> Result<Self> {
		let shape = vec![
			c.max_position_embeddings,
			c.num_key_value_heads,
			c.hidden_size / c.num_attention_heads,
		];
		let elements = shape.iter().product();
		let buffer = Buffer::new(shape, vec![0.; elements])?;
		Ok(Self {
			keys: vec![buffer.clone(); c.num_hidden_layers],
			values: vec![buffer; c.num_hidden_layers],
			valid: 0,
			capacity: c.max_position_embeddings,
		})
	}
	pub fn reset(&mut self) {
		self.valid = 0;
	}
	pub fn adopt(&mut self, outputs: &[Buffer], tokens: usize) -> Result<()> {
		if outputs.len() != 2 * self.keys.len()
			|| self
				.valid
				.checked_add(tokens)
				.ok_or("cache length overflow")?
				> self.capacity
		{
			return Err("cache adoption shape or capacity".into());
		}
		let mut candidate = self.clone();
		for (i, pair) in outputs.chunks_exact(2).enumerate() {
			if pair[0].shape != self.keys[i].shape || pair[1].shape != self.values[i].shape {
				return Err("cache output shape".into());
			}
			candidate.keys[i] = pair[0].clone();
			candidate.values[i] = pair[1].clone();
		}
		candidate.valid += tokens;
		*self = candidate;
		Ok(())
	}
}
pub fn token_buffer(tokens: &[u32], c: &QwenConfig) -> Result<Buffer> {
	if tokens.is_empty()
		|| tokens.len() > c.max_position_embeddings
		|| tokens.iter().any(|&t| t as usize >= c.vocab_size)
	{
		return Err("token range or context capacity".into());
	}
	Buffer::new(
		vec![tokens.len()],
		tokens.iter().map(|&t| t as f32).collect(),
	)
}
pub fn greedy(logits: &Buffer) -> Result<usize> {
	let vocab = *logits.shape.last().ok_or("logits rank")?;
	let row = logits
		.data
		.get(logits.data.len().checked_sub(vocab).ok_or("logits size")?..)
		.ok_or("logits slice")?;
	if row.is_empty() || row.iter().any(|v| !v.is_finite()) {
		return Err("nonfinite logits".into());
	}
	Ok(row
		.iter()
		.enumerate()
		.max_by(|(ai, a), (bi, b)| a.total_cmp(b).then_with(|| bi.cmp(ai)))
		.unwrap()
		.0)
}

pub fn initialize(config: &QwenConfig, seed: u64) -> Result<Vec<crate::state::BufferData>> {
	let registry = parameter_registry(config)?;
	let mut rng = seed.max(1);
	let mut values = std::collections::BTreeMap::new();
	for (name, spec) in &registry.names {
		values.entry(spec.id).or_insert_with(|| {
			let n: usize = spec.shape.iter().product();
			let norm = name.contains("layernorm") || name == "model.norm.weight";
			let bias = name.ends_with(".bias");
			let data = (0..n)
				.map(|_| {
					rng ^= rng << 13;
					rng ^= rng >> 7;
					rng ^= rng << 17;
					if norm {
						1.
					} else if bias {
						0.
					} else {
						((rng >> 40) as f32 / 16_777_216. - 0.5) * 0.1
					}
				})
				.collect();
			crate::state::BufferData {
				shape: spec.shape.clone(),
				data,
			}
		});
	}
	Ok(values.into_values().collect())
}
```

```rust file=src/nn/transformer.rs mode=write
pub use super::qwen::{
	Forward, QwenConfig as TinyConfig, logits_all, parameter_registry, register_graph,
};
```

```rust file=src/model_assets.rs mode=write
use crate::{
	Result,
	fetch::{Manifest, fetch, verified_read},
	nn::qwen::{QwenConfig, load_parameters},
	state::TensorSet,
};
use std::{collections::BTreeMap, path::Path};
pub struct ModelAssets {
	pub config: QwenConfig,
	pub state: TensorSet,
	pub tokenizer: tokenizers::Tokenizer,
	pub manifest: Manifest,
	pub eos: u32,
}
pub fn verified_files(
	manifest: &Path,
	directory: &Path,
	download: bool,
) -> Result<(Manifest, BTreeMap<String, Vec<u8>>)> {
	let bytes = std::fs::read(manifest).map_err(|e| e.to_string())?;
	let manifest: Manifest = serde_json::from_value(crate::safetensor::unique_json(&bytes)?)
		.map_err(|e| e.to_string())?;
	manifest.validate()?;
	let mut files = BTreeMap::new();
	for asset in &manifest.files {
		let bytes = if download {
			fetch(directory, asset)?
		} else {
			verified_read(&directory.join(&asset.name), asset)?
		};
		files.insert(asset.name.clone(), bytes);
	}
	Ok((manifest, files))
}
pub fn load(manifest: &Path, directory: &Path, download: bool) -> Result<ModelAssets> {
	let (manifest, files) = verified_files(manifest, directory, download)?;
	let required = |name: &str| {
		files
			.get(name)
			.ok_or_else(|| format!("manifest missing {name}"))
	};
	let config: QwenConfig =
		serde_json::from_value(crate::safetensor::unique_json(required("config.json")?)?)
			.map_err(|e| e.to_string())?;
	config.require_public_05b()?;
	let raw = if let Some(index) = files.get("model.safetensors.index.json") {
		crate::fetch::load_shards(index, &files)?
	} else {
		crate::safetensor::read(required("model.safetensors")?)?
	};
	let state = load_parameters(&config, &raw)?;
	let tokenizer = tokenizers::Tokenizer::from_bytes(required("tokenizer.json")?)
		.map_err(|e| e.to_string())?;
	let tokenizer_config = crate::safetensor::unique_json(required("tokenizer_config.json")?)?;
	if tokenizer_config
		.get("tokenizer_class")
		.and_then(|v| v.as_str())
		!= Some("Qwen2Tokenizer")
	{
		return Err("unsupported tokenizer class".into());
	}
	let generation = crate::safetensor::unique_json(required("generation_config.json")?)?;
	let eos = generation
		.get("eos_token_id")
		.and_then(|v| v.as_u64())
		.and_then(|v| u32::try_from(v).ok())
		.ok_or("EOS token ID")?;
	if eos as usize >= config.vocab_size {
		return Err("EOS vocabulary range".into());
	}
	Ok(ModelAssets {
		config,
		state,
		tokenizer,
		manifest,
		eos,
	})
}
pub fn encode(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>> {
	tokenizer
		.encode(text, false)
		.map(|e| e.get_ids().to_vec())
		.map_err(|e| e.to_string())
}
```

```rust file=examples/qwen_infer.rs mode=write
use chinigrad::{
	Graph, Result, model_assets,
	nn::qwen::{KvState, greedy, logits_all, parameter_registry, register_graph, token_buffer},
};
use std::path::Path;
fn main() -> Result<()> {
	let args = std::env::args().collect::<Vec<_>>();
	let arg = |name: &str| args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone());
	let manifest = arg("--manifest").unwrap_or("fixtures/qwen25/manifest.json".into());
	let directory = arg("--assets").unwrap_or("fixtures/qwen25".into());
	let assets = model_assets::load(
		Path::new(&manifest),
		Path::new(&directory),
		args.iter().any(|a| a == "--download"),
	)?;
	let mut c = assets.config.clone();
	c.max_position_embeddings = arg("--context")
		.unwrap_or("16".into())
		.parse::<usize>()
		.map_err(|e| e.to_string())?;
	if c.max_position_embeddings > assets.config.max_position_embeddings
		|| c.max_position_embeddings == 0
	{
		return Err("requested context capacity".into());
	}
	let count = arg("--max-new-tokens")
		.unwrap_or("1".into())
		.parse::<usize>()
		.map_err(|e| e.to_string())?;
	let text = arg("--text").unwrap_or("Hello".into());
	let mut input = model_assets::encode(&assets.tokenizer, &text)?;
	if input
		.len()
		.checked_add(count)
		.ok_or("token capacity overflow")?
		> c.max_position_embeddings
	{
		return Err("prompt plus generation exceeds context before compilation".into());
	}
	let registry = parameter_registry(&c)?;
	let weight_bindings = registry.bind(&assets.state)?;
	let slot = weight_bindings.len();
	let mut cache = KvState::new(&c)?;
	let mut generated = Vec::new();
	for _ in 0..count {
		let mut graph = Graph::default();
		let params = register_graph(&mut graph, &registry)?;
		let tokens = graph.input(slot, vec![input.len()])?;
		let mut bindings = weight_bindings.clone();
		bindings.push(token_buffer(&input, &c)?);
		let mut cache_ids = Vec::new();
		for (layer, (key, value)) in cache.keys.iter().zip(&cache.values).enumerate() {
			cache_ids.push((
				graph.input(slot + 1 + 2 * layer, key.shape.clone())?,
				graph.input(slot + 2 + 2 * layer, value.shape.clone())?,
			));
			bindings.extend([key.clone(), value.clone()]);
		}
		let forward = logits_all(
			&mut graph,
			&c,
			&params,
			tokens,
			cache.valid,
			Some(&cache_ids),
		)?;
		let mut outputs = vec![forward.logits];
		for (k, v) in forward.cache_outputs {
			outputs.extend([k, v]);
		}
		let plan = graph.compile(&outputs)?;
		let result = plan.run(&bindings)?;
		let next = greedy(&result[0])?;
		if args.iter().any(|a| a == "--dump") {
			eprintln!(
				"tokens={input:?} logits={:?} layers={} heads={}/{} position={} cache_valid={} graph_nodes={}",
				result[0].shape,
				c.num_hidden_layers,
				c.num_attention_heads,
				c.num_key_value_heads,
				cache.valid,
				cache.valid,
				graph.nodes.len()
			);
		}
		cache.adopt(&result[1..], input.len())?;
		generated.push(next as u32);
		if next as u32 == assets.eos {
			break;
		}
		input = vec![next as u32];
	}
	let output = assets
		.tokenizer
		.decode(&generated, false)
		.map_err(|e| e.to_string())?;
	println!("ids={generated:?}\n{output}");
	Ok(())
}
```

```rust file=examples/qwen_tokenize.rs mode=write
use chinigrad::{
	Result,
	fetch::{Manifest, verified_read},
	model_assets::encode,
};
use std::path::Path;
fn main() -> Result<()> {
	let manifest: Manifest = serde_json::from_slice(
		&std::fs::read("fixtures/qwen25/manifest.json").map_err(|e| e.to_string())?,
	)
	.map_err(|e| e.to_string())?;
	manifest.validate()?;
	let asset = manifest
		.files
		.iter()
		.find(|a| a.name == "tokenizer.json")
		.ok_or("tokenizer asset")?;
	let bytes = verified_read(Path::new("fixtures/qwen25/tokenizer.json"), asset)?;
	let tokenizer = tokenizers::Tokenizer::from_bytes(bytes).map_err(|e| e.to_string())?;
	for text in ["Hello", " ", "こんにちは", "Hello\nworld", "<|endoftext|>"] {
		println!(
			"{}",
			serde_json::json!({"text":text,"ids":encode(&tokenizer,text)?})
		);
	}
	Ok(())
}
```

```rust file=examples/fetch_qwen.rs mode=write
use chinigrad::{
	Result,
	fetch::{Manifest, fetch},
};
use std::path::Path;
fn main() -> Result<()> {
	let args = std::env::args().collect::<Vec<_>>();
	let manifest: Manifest = serde_json::from_slice(
		&std::fs::read("fixtures/qwen25/manifest.json").map_err(|e| e.to_string())?,
	)
	.map_err(|e| e.to_string())?;
	manifest.validate()?;
	for asset in &manifest.files {
		if args.iter().any(|a| a == "--tokenizer-only") && asset.name == "model.safetensors" {
			continue;
		}
		fetch(Path::new("fixtures/qwen25"), asset)?;
		println!("verified {} {} {}", asset.name, asset.bytes, asset.sha256);
	}
	Ok(())
}
```

```json file=fixtures/qwen25/config.json mode=write
{
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151643,
  "hidden_act": "silu",
  "hidden_size": 896,
  "initializer_range": 0.02,
  "intermediate_size": 4864,
  "max_position_embeddings": 32768,
  "max_window_layers": 24,
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.40.1",
  "use_cache": true,
  "use_mrope": false,
  "use_sliding_window": false,
  "vocab_size": 151936
}
```

```json file=fixtures/qwen25/manifest.json mode=write
{
  "revision": "b937cc073b55e980e6f21d5f8b7a77818efcf0eb",
  "files": [
    {
      "name": "config.json",
      "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/b937cc073b55e980e6f21d5f8b7a77818efcf0eb/config.json",
      "bytes": 681,
      "sha256": "479dcf0c5286339e41ad3992cd08ae88a467c4187587936248e2b7c96283484b"
    },
    {
      "name": "generation_config.json",
      "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/b937cc073b55e980e6f21d5f8b7a77818efcf0eb/generation_config.json",
      "bytes": 138,
      "sha256": "8c970692323e3ea0e9b8b0a4dca79388d31226e41f83c9fd6014804280ebf6e8"
    },
    {
      "name": "model.safetensors",
      "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/b937cc073b55e980e6f21d5f8b7a77818efcf0eb/model.safetensors",
      "bytes": 988097824,
      "sha256": "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
    },
    {
      "name": "tokenizer.json",
      "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/b937cc073b55e980e6f21d5f8b7a77818efcf0eb/tokenizer.json",
      "bytes": 7031645,
      "sha256": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
    },
    {
      "name": "tokenizer_config.json",
      "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/b937cc073b55e980e6f21d5f8b7a77818efcf0eb/tokenizer_config.json",
      "bytes": 5012,
      "sha256": "0213fe1b3f03c373e92078795dda2a2f70064fadf0cc42ce7f397d5424724a40"
    }
  ]
}
```

```json file=fixtures/qwen25/tokenizer-fixture.json mode=write
[
  {
    "text": "Hello",
    "ids": [
      9707
    ]
  },
  {
    "text": " ",
    "ids": [
      220
    ]
  },
  {
    "text": "こんにちは",
    "ids": [
      89015
    ]
  },
  {
    "text": "Hello\nworld",
    "ids": [
      9707,
      198,
      14615
    ]
  },
  {
    "text": "<|endoftext|>",
    "ids": [
      151643
    ]
  }
]
```

```json file=fixtures/tiny-transformer.json mode=write
[
  [
    0.0671042,
    0.2108996,
    0.040931597,
    -0.02865377,
    -0.12273584,
    0.007009507,
    -0.15163787,
    -0.0012681971,
    -0.003049571,
    -0.013462838,
    -0.014876825,
    0.051280003,
    -0.0127871195,
    0.1054231,
    -0.08555348,
    -0.020800315,
    0.040562876,
    0.08017206,
    0.17365754,
    -0.027789153,
    -0.06450024,
    0.1701316,
    -0.011237221,
    -0.046811216,
    -0.09734064,
    -0.077315405,
    0.05144692,
    0.031174714,
    0.03489143,
    0.036556005,
    0.019648006,
    -0.027185993,
    0.029575082,
    -0.019099629,
    -0.016163737,
    0.20257924,
    -0.07591644,
    0.007783771,
    0.16077705,
    0.09159438,
    -0.0139912935,
    0.010520314,
    0.02022292,
    -0.16471742,
    -0.02388436,
    -0.1315632,
    -0.034919187,
    0.11073867
  ],
  [
    -0.0057600974,
    -0.04117927,
    0.004650229,
    -0.035598665,
    -0.03314799,
    0.008851999,
    -0.030194765,
    -0.02623768,
    0.011505312,
    0.008218628,
    -0.03527941,
    -0.04505176,
    -0.002550459,
    -0.012281537,
    -0.015517789,
    0.008460927,
    0.02368551,
    0.00074211956,
    -0.01686806,
    0.040561885,
    0.016594727,
    0.01515717,
    -0.046978164,
    0.00081198214
  ],
  [
    -0.012583544,
    -0.044878613,
    0.003340548,
    -0.036363605,
    -0.03887511,
    -0.000274525,
    -0.034897406,
    -0.042034246,
    0.007564385,
    0.006892677,
    -0.03407039,
    -0.046927154,
    -0.00467973,
    -0.01599567,
    -0.017864577,
    -0.0027807895,
    0.021209378,
    -0.0003979832,
    -0.019190688,
    0.041704684,
    0.015254701,
    0.010885616,
    -0.048773877,
    -0.0047184853
  ],
  [
    -0.39767903,
    -1.4183035,
    0.10557169,
    -1.1492028,
    -1.2285742,
    -0.008675842,
    -1.1028663,
    -1.3284127,
    0.33494365,
    0.30520105,
    -1.5086039,
    -2.077889,
    -0.20721392,
    -0.7082728,
    -0.7910262,
    -0.123130664,
    0.82262546,
    -0.015436149,
    -0.74432874,
    1.617555,
    0.5916678,
    0.42220873,
    -1.8917402,
    -0.18301085
  ]
]
```

```python file=tools/qwen_reference.py mode=write
import argparse
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--assets", default="fixtures/qwen25")
parser.add_argument("--text", default="Hello")
parser.add_argument("--output", default="fixtures/qwen25/reference.json")
args = parser.parse_args()
assets = Path(args.assets)
manifest = json.loads((assets / "manifest.json").read_text())
for entry in manifest["files"]:
    path = assets / entry["name"]
    assert path.stat().st_size == entry["bytes"], entry["name"]
    digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
    assert digest == entry["sha256"], entry["name"]
torch.set_num_threads(1)
torch.manual_seed(1)
tokenizer = AutoTokenizer.from_pretrained(assets, local_files_only=True, trust_remote_code=False)
for fixture in json.loads((assets / "tokenizer-fixture.json").read_text()):
    assert tokenizer.encode(fixture["text"], add_special_tokens=False) == fixture["ids"]
model = AutoModelForCausalLM.from_pretrained(
    assets, local_files_only=True, trust_remote_code=False,
    torch_dtype=torch.float32, attn_implementation="eager",
).eval()
assert model.config.num_hidden_layers == 24
ids = tokenizer.encode(args.text, add_special_tokens=False)
assert 0 < len(ids) <= 16
boundaries = []
def capture(module, inputs, output):
    if isinstance(output, tuple):
        output = output[0]
    boundaries.append(output.detach().float().reshape(-1).tolist())
handles = [model.model.embed_tokens.register_forward_hook(capture)]
handles += [layer.register_forward_hook(capture) for layer in model.model.layers]
handles += [model.model.norm.register_forward_hook(capture)]
with torch.no_grad():
    logits = model(torch.tensor([ids]), use_cache=False).logits.float()
for handle in handles:
    handle.remove()
result = {
    "revision": manifest["revision"], "text": args.text, "ids": ids,
    "boundaries": boundaries, "logits": logits.reshape(-1).tolist(),
    "shape": list(logits.shape), "torch": torch.__version__,
}
Path(args.output).write_text(json.dumps(result, ensure_ascii=False) + "\n")
print("wrote", args.output, "boundaries", len(boundaries), "logits", list(logits.shape))
```

```text file=tools/qwen-reference-requirements.txt mode=write
torch==2.5.1
transformers==4.46.3
safetensors==0.4.5
tokenizers==0.20.3
```

```rust file=tests/transformer_reference.rs mode=write
use chinigrad::{
	Buffer, Graph,
	nn::qwen::initialize,
	nn::{
		attention::{apply_rope, attention, softmax},
		qwen::{KvState, QwenConfig, logits_all, parameter_registry, register_graph},
	},
	state::BufferData,
};
fn close(actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
	assert_eq!(actual.len(), expected.len());
	let mut max = 0f32;
	for (&a, &b) in actual.iter().zip(expected) {
		max = max.max((a - b).abs());
		assert!(
			a.is_finite() && b.is_finite() && (a - b).abs() <= atol + rtol * b.abs(),
			"{a} != {b}, max={max}"
		);
	}
}
fn run(
	c: &QwenConfig,
	weights: &[BufferData],
	tokens: &[usize],
	cache: Option<&KvState>,
) -> (Vec<Buffer>, Vec<Buffer>) {
	let mut g = Graph::default();
	let reg = parameter_registry(c).unwrap();
	let params = register_graph(&mut g, &reg).unwrap();
	let slot = weights.len();
	let ids = g.input(slot, vec![tokens.len()]).unwrap();
	let mut bindings = weights
		.iter()
		.map(|w| w.buffer().unwrap())
		.collect::<Vec<_>>();
	bindings.push(
		Buffer::new(
			vec![tokens.len()],
			tokens.iter().map(|&t| t as f32).collect(),
		)
		.unwrap(),
	);
	let mut cache_ids = Vec::new();
	if let Some(cache) = cache {
		for (i, (k, v)) in cache.keys.iter().zip(&cache.values).enumerate() {
			cache_ids.push((
				g.input(slot + 1 + 2 * i, k.shape.clone()).unwrap(),
				g.input(slot + 2 + 2 * i, v.shape.clone()).unwrap(),
			));
			bindings.extend([k.clone(), v.clone()]);
		}
	}
	let forward = logits_all(
		&mut g,
		c,
		&params,
		ids,
		cache.map_or(0, |c| c.valid),
		cache.map(|_| cache_ids.as_slice()),
	)
	.unwrap();
	let mut out = vec![forward.logits];
	out.extend(&forward.boundaries);
	for (k, v) in forward.cache_outputs {
		out.extend([k, v]);
	}
	let all = g.compile(&out).unwrap().run(&bindings).unwrap();
	let n = 1 + forward.boundaries.len();
	(all[..n].to_vec(), all[n..].to_vec())
}
fn reference(c: &QwenConfig, weights: &[BufferData], tokens: &[usize]) -> Vec<Vec<f32>> {
	let registry = parameter_registry(c).unwrap();
	let get = |name: &str| {
		weights[registry.names[name].slot]
			.data
			.iter()
			.map(|&v| v as f64)
			.collect::<Vec<_>>()
	};
	let d = c.hidden_size;
	let t = tokens.len();
	let linear = |x: &[f64],
	              w: &[f64],
	              rows: usize,
	              input: usize,
	              output: usize,
	              bias: Option<Vec<f64>>|
	 -> Vec<f64> {
		let mut y = vec![0.; rows * output];
		for r in 0..rows {
			for o in 0..output {
				y[r * output + o] = bias.as_ref().map_or(0., |b| b[o]);
				for i in 0..input {
					y[r * output + o] += x[r * input + i] * w[o * input + i];
				}
			}
		}
		y
	};
	let norm = |x: &[f64], weight: Vec<f64>| -> Vec<f64> {
		let mut y = x.to_vec();
		for row in y.chunks_exact_mut(d) {
			let rms =
				(row.iter().map(|v| v * v).sum::<f64>() / d as f64 + c.rms_norm_eps as f64).sqrt();
			for (i, v) in row.iter_mut().enumerate() {
				*v = *v / rms * weight[i];
			}
		}
		y
	};
	let embed = get("model.embed_tokens.weight");
	let mut x = tokens
		.iter()
		.flat_map(|&id| embed[id * d..(id + 1) * d].iter().copied())
		.collect::<Vec<_>>();
	let mut boundary = vec![x.iter().map(|&v| v as f32).collect()];
	let hd = d / c.num_attention_heads;
	let kvd = c.num_key_value_heads * hd;
	for l in 0..c.num_hidden_layers {
		let p = format!("model.layers.{l}");
		let normalized = norm(&x, get(&format!("{p}.input_layernorm.weight")));
		let project = |name: &str, out| {
			linear(
				&normalized,
				&get(&format!("{p}.self_attn.{name}.weight")),
				t,
				d,
				out,
				Some(get(&format!("{p}.self_attn.{name}.bias"))),
			)
		};
		let mut q = project("q_proj", d);
		let mut k = project("k_proj", kvd);
		let v = project("v_proj", kvd);
		for (values, heads) in [
			(&mut q, c.num_attention_heads),
			(&mut k, c.num_key_value_heads),
		] {
			for pos in 0..t {
				for head in 0..heads {
					for r in 0..hd / 2 {
						let angle = pos as f64 / c.rope_theta.powf(2. * r as f64 / hd as f64);
						let (a, b) = (
							values[(pos * heads + head) * hd + r],
							values[(pos * heads + head) * hd + r + hd / 2],
						);
						values[(pos * heads + head) * hd + r] = a * angle.cos() - b * angle.sin();
						values[(pos * heads + head) * hd + r + hd / 2] =
							a * angle.sin() + b * angle.cos();
					}
				}
			}
		}
		let mut attended = vec![0.; t * d];
		for pos in 0..t {
			for head in 0..c.num_attention_heads {
				let kh = head / (c.num_attention_heads / c.num_key_value_heads);
				let scores = (0..=pos)
					.map(|key| {
						(0..hd)
							.map(|i| {
								q[(pos * c.num_attention_heads + head) * hd + i]
									* k[(key * c.num_key_value_heads + kh) * hd + i]
							})
							.sum::<f64>() / (hd as f64).sqrt()
					})
					.collect::<Vec<_>>();
				let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
				let probabilities = scores.iter().map(|s| (s - max).exp()).collect::<Vec<_>>();
				let sum: f64 = probabilities.iter().sum();
				for key in 0..=pos {
					for i in 0..hd {
						attended[pos * d + head * hd + i] += probabilities[key] / sum
							* v[(key * c.num_key_value_heads + kh) * hd + i];
					}
				}
			}
		}
		let output = linear(
			&attended,
			&get(&format!("{p}.self_attn.o_proj.weight")),
			t,
			d,
			d,
			None,
		);
		for (a, b) in x.iter_mut().zip(output) {
			*a += b;
		}
		let n = norm(&x, get(&format!("{p}.post_attention_layernorm.weight")));
		let mut gate = linear(
			&n,
			&get(&format!("{p}.mlp.gate_proj.weight")),
			t,
			d,
			c.intermediate_size,
			None,
		);
		let up = linear(
			&n,
			&get(&format!("{p}.mlp.up_proj.weight")),
			t,
			d,
			c.intermediate_size,
			None,
		);
		for (g, u) in gate.iter_mut().zip(up) {
			*g = *g / (1. + (-*g).exp()) * u;
		}
		let down = linear(
			&gate,
			&get(&format!("{p}.mlp.down_proj.weight")),
			t,
			c.intermediate_size,
			d,
			None,
		);
		for (a, b) in x.iter_mut().zip(down) {
			*a += b;
		}
		boundary.push(x.iter().map(|&v| v as f32).collect());
	}
	x = norm(&x, get("model.norm.weight"));
	boundary.push(x.iter().map(|&v| v as f32).collect());
	let logits = linear(
		&x,
		&get(if c.tie_word_embeddings {
			"model.embed_tokens.weight"
		} else {
			"lm_head.weight"
		}),
		t,
		d,
		c.vocab_size,
		None,
	);
	boundary.insert(0, logits.iter().map(|&v| v as f32).collect());
	boundary
}
```

```rust file=tests/qwen_tokenizer.rs mode=write
use chinigrad::{
	fetch::{Manifest, verified_read},
	model_assets::encode,
};
```

```rust file=tests/qwen_public.rs mode=write
use chinigrad::{
	Graph, model_assets,
	nn::qwen::{logits_all, parameter_registry, register_graph, token_buffer},
};
use std::path::Path;
```

`tests/transformer_reference.rs::causal_rope_and_softmax` を追加する。

```rust file=tests/transformer_reference.rs mode=append
#[test]
fn causal_rope_and_softmax() {
	let mut g = Graph::default();
	let q = g.constant(0., vec![2, 2, 2]).unwrap();
	let k = g.constant(0., vec![2, 1, 2]).unwrap();
	let v = g.input(0, vec![2, 1, 2]).unwrap();
	let y = attention(&mut g, q, k, v, 0, true).unwrap();
	let rope = apply_rope(&mut g, v, 0, 10000.).unwrap();
	let scores = g.constant(0., vec![2, 2]).unwrap();
	let probabilities = softmax(&mut g, scores, 1).unwrap();
	let out = g
		.compile(&[y, rope, probabilities])
		.unwrap()
		.run(&[Buffer::new(vec![2, 1, 2], vec![2., 4., 6., 8.]).unwrap()])
		.unwrap();
	close(&out[0].data, &[2., 4., 2., 4., 4., 6., 4., 6.], 1e-5, 1e-4);
	close(&out[1].data[..2], &[2., 4.], 0., 0.);
	close(&out[2].data, &[0.5; 4], 1e-6, 0.);
}
```

実装箇所は `nn::attention::embedding` である。この要件に対する実行可能な確認は `transformer_reference::causal_rope_and_softmax` に置く。

```sh check
cargo test --offline --test transformer_reference causal_rope_and_softmax -- --exact
```

## 2. RMSNorm を `x*rsqrt(mean(x*x)+eps)*weight` で組み立てる。

<!-- todo:22-002 -->


実装箇所は `nn::attention::rms_norm` である。この要件に対する実行可能な確認は `transformer_reference::causal_rope_and_softmax` に置く。

`src/nn/attention.rs` の `pub fn rms_norm(`の定義を次の全体で置き換えます。

```rust file="src/nn/attention.rs" mode="replace" item="pub fn rms_norm("
pub fn rms_norm(g: &mut Graph, x: NodeId, weight: NodeId, eps: f32) -> Result<NodeId> {
	let dim = *g
		.shape(x)
		.last()
		.ok_or("RMSNorm requires a final dimension")?;
	if dim == 0 || g.shape(weight) != [dim] || eps <= 0. || !eps.is_finite() {
		return Err("RMSNorm shape/epsilon".into());
	}
	let axis = g.shape(x).len() - 1;
	let square = g.binary(Binary::Mul, x, x)?;
	let sum = g.reduce(square, Reduce::Sum, vec![axis], true)?;
	let divisor = g.constant(dim as f32, vec![])?;
	let mean = g.binary(Binary::Div, sum, divisor)?;
	let eps = g.constant(eps, vec![])?;
	let variance = g.binary(Binary::Add, mean, eps)?;
	let root = g.unary(Unary::Sqrt, variance)?;
	let norm = g.binary(Binary::Div, x, root)?;
	g.binary(Binary::Mul, norm, weight)
}
```

```sh check
cargo test --offline --test transformer_reference causal_rope_and_softmax -- --exact
```

## 3. softmax を `m=detach(max(scores))` の減算、exp、sum、除算で組み立て、行和を確認する。

<!-- todo:22-003 -->


実装箇所は `nn::attention::softmax` である。この要件に対する実行可能な確認は `transformer_reference::causal_rope_and_softmax` に置く。

`src/nn/mod.rs` の `pub fn log_softmax(`の定義を次の全体で置き換えます。

```rust file="src/nn/mod.rs" mode="replace" item="pub fn log_softmax("
pub fn log_softmax(g: &mut Graph, x: NodeId, axis: usize) -> Result<NodeId> {
	let m = g.reduce(x, Reduce::Max, vec![axis], true)?;
	let m = g.stop_gradient(m)?;
	let shift = g.binary(Binary::Sub, x, m)?;
	let ex = g.unary(Unary::Exp, shift)?;
	let sum = g.reduce(ex, Reduce::Sum, vec![axis], true)?;
	let log = g.unary(Unary::Log, sum)?;
	g.binary(Binary::Sub, shift, log)
}
```

```sh check
cargo test --offline --test transformer_reference causal_rope_and_softmax -- --exact
```

## 4. RoPE を偶数の head dimension と位置から作り、位置0で入力と一致させる。

<!-- todo:22-004 -->


実装箇所は `nn::attention::apply_rope` である。この要件に対する実行可能な確認は `transformer_reference::causal_rope_and_softmax` に置く。

`src/nn/attention.rs` の `pub fn apply_rope(`の定義を次の全体で置き換えます。

```rust file="src/nn/attention.rs" mode="replace" item="pub fn apply_rope("
pub fn apply_rope(g: &mut Graph, x: NodeId, start: usize, theta: f64) -> Result<NodeId> {
	let shape = g.shape(x).to_vec();
	if shape.len() != 3 || shape[2] % 2 != 0 || shape[2] == 0 || !theta.is_finite() || theta <= 0. {
		return Err("RoPE shape/theta".into());
	}
	let (t, h, d) = (shape[0], shape[1], shape[2]);
	let half = d / 2;
	let mut positions = Vec::new();
	for pos in 0..t {
		let absolute = start.checked_add(pos).ok_or("RoPE position overflow")?;
		let mut dimensions = Vec::new();
		for channel in 0..d {
			let r = channel % half;
			let angle = absolute as f64 / theta.powf(2. * r as f64 / d as f64);
			let (c, s) = (angle.cos() as f32, angle.sin() as f32);
			let a = g.shrink(x, vec![(pos, pos + 1), (0, h), (r, r + 1)])?;
			let b = g.shrink(x, vec![(pos, pos + 1), (0, h), (r + half, r + half + 1)])?;
			let a = g.reshape(a, vec![h])?;
			let b = g.reshape(b, vec![h])?;
			let cc = g.constant(c, vec![])?;
			let ss = g.constant(s, vec![])?;
			let output = if channel < half {
				let ac = g.binary(Binary::Mul, a, cc)?;
				let bs = g.binary(Binary::Mul, b, ss)?;
				g.binary(Binary::Sub, ac, bs)?
			} else {
				let as_ = g.binary(Binary::Mul, a, ss)?;
				let bc = g.binary(Binary::Mul, b, cc)?;
				g.binary(Binary::Add, as_, bc)?
			};
			dimensions.push(output);
		}
		positions.push(g.stack(dimensions, 1)?);
	}
	g.stack(positions, 0)
}
```

```sh check
cargo test --offline --test transformer_reference causal_rope_and_softmax -- --exact
```

## 5. `QK^T/sqrt(d)`、causal mask、softmax、V の積を既存 Tensor 演算で作る。

<!-- todo:22-005 -->


実装箇所は `nn::attention::attention` である。この要件に対する実行可能な確認は `transformer_reference::causal_rope_and_softmax` に置く。

`src/nn/attention.rs` の `pub fn attention(`の定義を次の全体で置き換えます。

```rust file="src/nn/attention.rs" mode="replace" item="pub fn attention("
pub fn attention(
	g: &mut Graph,
	q: NodeId,
	k: NodeId,
	v: NodeId,
	query_start: usize,
	causal: bool,
) -> Result<NodeId> {
	let qs = g.shape(q).to_vec();
	let ks = g.shape(k).to_vec();
	if qs.len() != 3
		|| ks.len() != 3
		|| g.shape(v) != ks
		|| qs[2] != ks[2]
		|| ks[1] == 0
		|| qs[1] % ks[1] != 0
		|| qs[2] == 0
		|| ks[0] == 0
	{
		return Err("attention shape or GQA ratio".into());
	}
	let (t, h, d, s, hkv) = (qs[0], qs[1], qs[2], ks[0], ks[1]);
	let ratio = h / hkv;
	let mut heads = Vec::new();
	for head in 0..h {
		let kh = head / ratio;
		let qq = g.shrink(q, vec![(0, t), (head, head + 1), (0, d)])?;
		let qq = g.reshape(qq, vec![t, d])?;
		let kk = g.shrink(k, vec![(0, s), (kh, kh + 1), (0, d)])?;
		let kk = g.reshape(kk, vec![s, d])?;
		let kt = g.permute(kk, vec![1, 0])?;
		let scores = g.matmul(qq, kt)?;
		let scale = g.constant(1. / (d as f32).sqrt(), vec![])?;
		let scores = g.binary(Binary::Mul, scores, scale)?;
		let mut mask_values = Vec::with_capacity(t * s);
		for row in 0..t {
			let position = query_start
				.checked_add(row)
				.ok_or("query position overflow")?;
			for col in 0..s {
				mask_values.push(if !causal || col <= position {
					0.
				} else {
					f32::NEG_INFINITY
				});
			}
		}
		let mask = constant_array(g, vec![t, s], &mask_values)?;
		let scores = g.binary(Binary::Add, scores, mask)?;
		let probabilities = softmax(g, scores, 1)?;
		let vv = g.shrink(v, vec![(0, s), (kh, kh + 1), (0, d)])?;
		let vv = g.reshape(vv, vec![s, d])?;
		heads.push(g.matmul(probabilities, vv)?);
	}
	g.stack(heads, 1)
}
```

```sh check
cargo test --offline --test transformer_reference causal_rope_and_softmax -- --exact
```

## 6. 小 config と明示登録した state を用意し、embedding から output projection までを通す。

<!-- todo:22-006 -->


`tests/transformer_reference.rs::zero_transformer` を追加する。

```rust file=tests/transformer_reference.rs mode=append
#[test]
fn zero_transformer() {
	let c = QwenConfig::tiny();
	let mut weights = initialize(&c, 1).unwrap();
	for w in &mut weights {
		w.data.fill(0.);
	}
	let (values, _) = run(&c, &weights, &[1, 2, 3], None);
	assert!(values[0].data.iter().all(|&x| x == 0.));
	assert_eq!(values[0].shape, [3, 16]);
}
```

実装箇所は `nn::qwen::register_graph` である。この要件に対する実行可能な確認は `transformer_reference::zero_transformer` に置く。

`src/nn/qwen.rs` の `pub fn logits_all_with_attention<F>(`の定義を次の全体で置き換えます。

```rust file="src/nn/qwen.rs" mode="replace" item="pub fn logits_all_with_attention<F>("
pub fn logits_all_with_attention<F>(
	g: &mut Graph,
	c: &QwenConfig,
	params: &BTreeMap<String, NodeId>,
	tokens: NodeId,
	start: usize,
	cache: Option<&[(NodeId, NodeId)]>,
	attention_builder: &mut F,
) -> Result<Forward>
where
	F: FnMut(&mut Graph, NodeId, NodeId, NodeId, usize, bool) -> Result<NodeId>,
{
	c.validate()?;
	let shape = g.shape(tokens);
	if shape.len() != 1 || shape[0] == 0 {
		return Err("tokens must be nonempty rank one".into());
	}
	let t = shape[0];
	if start.checked_add(t).ok_or("context overflow")? > c.max_position_embeddings {
		return Err("input exceeds context capacity".into());
	}
	if cache.is_none() && start != 0 {
		return Err("nonzero position requires a cache".into());
	}
	if cache.is_some_and(|v| v.len() != c.num_hidden_layers) {
		return Err("cache layer count".into());
	}
	let get = |name: &str| {
		params
			.get(name)
			.copied()
			.ok_or_else(|| format!("missing graph parameter {name}"))
	};
	let d = c.hidden_size;
	let hd = d / c.num_attention_heads;
	let kd = c.num_key_value_heads;
	let mut x = super::attention::embedding(g, get("model.embed_tokens.weight")?, tokens)?;
	let mut boundaries = vec![x];
	let mut cache_outputs = Vec::new();
	for layer in 0..c.num_hidden_layers {
		let p = format!("model.layers.{layer}");
		let n = rms_norm(
			g,
			x,
			get(&format!("{p}.input_layernorm.weight"))?,
			c.rms_norm_eps,
		)?;
		let projection = |name: &str, heads: usize, g: &mut Graph| -> Result<NodeId> {
			let value = linear(
				g,
				n,
				get(&format!("{p}.self_attn.{name}.weight"))?,
				Some(get(&format!("{p}.self_attn.{name}.bias"))?),
			)?;
			g.reshape(value, vec![t, heads, hd])
		};
		let q = projection("q_proj", c.num_attention_heads, g)?;
		let k = projection("k_proj", kd, g)?;
		let v = projection("v_proj", kd, g)?;
		let q = apply_rope(g, q, start, c.rope_theta)?;
		let k = apply_rope(g, k, start, c.rope_theta)?;
		let (k, v) = if let Some(old) = cache {
			let (ck, cv) = old[layer];
			let expected = [c.max_position_embeddings, kd, hd];
			if g.shape(ck) != expected || g.shape(cv) != expected {
				return Err("cache capacity shape".into());
			}
			let (k, v) = if start == 0 {
				(k, v)
			} else {
				let ranges = vec![(0, start), (0, kd), (0, hd)];
				let prefix_k = g.shrink(ck, ranges.clone())?;
				let prefix_v = g.shrink(cv, ranges)?;
				(
					concatenate(g, prefix_k, k, 0)?,
					concatenate(g, prefix_v, v, 0)?,
				)
			};
			let pads = vec![(0, c.max_position_embeddings - start - t), (0, 0), (0, 0)];
			let new_k = g.pad(k, pads.clone())?;
			let new_v = g.pad(v, pads)?;
			let new_k = g.stage(new_k)?;
			let new_v = g.stage(new_v)?;
			cache_outputs.push((new_k, new_v));
			let ranges = vec![(0, start + t), (0, kd), (0, hd)];
			(g.shrink(new_k, ranges.clone())?, g.shrink(new_v, ranges)?)
		} else {
			(k, v)
		};
		let attended = attention_builder(g, q, k, v, start, true)?;
		let attended = g.reshape(attended, vec![t, d])?;
		let projected = linear(
			g,
			attended,
			get(&format!("{p}.self_attn.o_proj.weight"))?,
			None,
		)?;
		x = g.binary(Binary::Add, x, projected)?;
		let n = rms_norm(
			g,
			x,
			get(&format!("{p}.post_attention_layernorm.weight"))?,
			c.rms_norm_eps,
		)?;
		let gate = linear(g, n, get(&format!("{p}.mlp.gate_proj.weight"))?, None)?;
		let gate = silu(g, gate)?;
		let up = linear(g, n, get(&format!("{p}.mlp.up_proj.weight"))?, None)?;
		let product = g.binary(Binary::Mul, gate, up)?;
		let down = linear(g, product, get(&format!("{p}.mlp.down_proj.weight"))?, None)?;
		x = g.binary(Binary::Add, x, down)?;
		x = g.stage(x)?;
		boundaries.push(x);
	}
	let x = rms_norm(g, x, get("model.norm.weight")?, c.rms_norm_eps)?;
	boundaries.push(x);
	let output = get(if c.tie_word_embeddings {
		"model.embed_tokens.weight"
	} else {
		"lm_head.weight"
	})?;
	let logits = linear(g, x, output, None)?;
	Ok(Forward {
		logits,
		boundaries,
		cache_outputs,
	})
}
```

```sh check
cargo test --offline --test transformer_reference zero_transformer -- --exact
```

## 7. pre-norm、attention residual、norm、gated MLP、residual、最終 norm の順を固定する。

<!-- todo:22-007 -->


実装箇所は `nn::qwen::logits_all` である。この要件に対する実行可能な確認は `transformer_reference::zero_transformer` に置く。

`src/nn/qwen.rs` の `pub fn logits_all_with_attention<F>(`の定義を次の全体で置き換えます。

```rust file="src/nn/qwen.rs" mode="replace" item="pub fn logits_all_with_attention<F>("
pub fn logits_all_with_attention<F>(
	g: &mut Graph,
	c: &QwenConfig,
	params: &BTreeMap<String, NodeId>,
	tokens: NodeId,
	start: usize,
	cache: Option<&[(NodeId, NodeId)]>,
	attention_builder: &mut F,
) -> Result<Forward>
where
	F: FnMut(&mut Graph, NodeId, NodeId, NodeId, usize, bool) -> Result<NodeId>,
{
	c.validate()?;
	let shape = g.shape(tokens);
	if shape.len() != 1 || shape[0] == 0 {
		return Err("tokens must be nonempty rank one".into());
	}
	let t = shape[0];
	if start.checked_add(t).ok_or("context overflow")? > c.max_position_embeddings {
		return Err("input exceeds context capacity".into());
	}
	if cache.is_none() && start != 0 {
		return Err("nonzero position requires a cache".into());
	}
	if cache.is_some_and(|v| v.len() != c.num_hidden_layers) {
		return Err("cache layer count".into());
	}
	let get = |name: &str| {
		params
			.get(name)
			.copied()
			.ok_or_else(|| format!("missing graph parameter {name}"))
	};
	let d = c.hidden_size;
	let hd = d / c.num_attention_heads;
	let kd = c.num_key_value_heads;
	let mut x = super::attention::embedding(g, get("model.embed_tokens.weight")?, tokens)?;
	let mut boundaries = vec![x];
	let mut cache_outputs = Vec::new();
	for layer in 0..c.num_hidden_layers {
		let p = format!("model.layers.{layer}");
		let n = rms_norm(
			g,
			x,
			get(&format!("{p}.input_layernorm.weight"))?,
			c.rms_norm_eps,
		)?;
		let projection = |name: &str, heads: usize, g: &mut Graph| -> Result<NodeId> {
			let value = linear(
				g,
				n,
				get(&format!("{p}.self_attn.{name}.weight"))?,
				Some(get(&format!("{p}.self_attn.{name}.bias"))?),
			)?;
			g.reshape(value, vec![t, heads, hd])
		};
		let q = projection("q_proj", c.num_attention_heads, g)?;
		let k = projection("k_proj", kd, g)?;
		let v = projection("v_proj", kd, g)?;
		let q = apply_rope(g, q, start, c.rope_theta)?;
		let k = apply_rope(g, k, start, c.rope_theta)?;
		let (k, v) = if let Some(old) = cache {
			let (ck, cv) = old[layer];
			let expected = [c.max_position_embeddings, kd, hd];
			if g.shape(ck) != expected || g.shape(cv) != expected {
				return Err("cache capacity shape".into());
			}
			let (k, v) = if start == 0 {
				(k, v)
			} else {
				let ranges = vec![(0, start), (0, kd), (0, hd)];
				let prefix_k = g.shrink(ck, ranges.clone())?;
				let prefix_v = g.shrink(cv, ranges)?;
				(
					concatenate(g, prefix_k, k, 0)?,
					concatenate(g, prefix_v, v, 0)?,
				)
			};
			let pads = vec![(0, c.max_position_embeddings - start - t), (0, 0), (0, 0)];
			let new_k = g.pad(k, pads.clone())?;
			let new_v = g.pad(v, pads)?;
			let new_k = g.stage(new_k)?;
			let new_v = g.stage(new_v)?;
			cache_outputs.push((new_k, new_v));
			let ranges = vec![(0, start + t), (0, kd), (0, hd)];
			(g.shrink(new_k, ranges.clone())?, g.shrink(new_v, ranges)?)
		} else {
			(k, v)
		};
		let attended = attention_builder(g, q, k, v, start, true)?;
		let attended = g.reshape(attended, vec![t, d])?;
		let projected = linear(
			g,
			attended,
			get(&format!("{p}.self_attn.o_proj.weight"))?,
			None,
		)?;
		x = g.binary(Binary::Add, x, projected)?;
		let n = rms_norm(
			g,
			x,
			get(&format!("{p}.post_attention_layernorm.weight"))?,
			c.rms_norm_eps,
		)?;
		let gate = linear(g, n, get(&format!("{p}.mlp.gate_proj.weight"))?, None)?;
		let gate = silu(g, gate)?;
		let up = linear(g, n, get(&format!("{p}.mlp.up_proj.weight"))?, None)?;
		let product = g.binary(Binary::Mul, gate, up)?;
		let down = linear(g, product, get(&format!("{p}.mlp.down_proj.weight"))?, None)?;
		x = g.binary(Binary::Add, x, down)?;
		x = g.stage(x)?;
		boundaries.push(x);
	}
	let x = rms_norm(g, x, get("model.norm.weight")?, c.rms_norm_eps)?;
	boundaries.push(x);
	let output = get(if c.tie_word_embeddings {
		"model.embed_tokens.weight"
	} else {
		"lm_head.weight"
	})?;
	let logits = linear(g, x, output, None)?;
	Ok(Forward {
		logits,
		boundaries,
		cache_outputs,
	})
}
```

```sh check
cargo test --offline --test transformer_reference zero_transformer -- --exact
```

## 8. MLP を `down(silu(gate(x))*up(x))` とし、bias の有無を fixture に固定する。

<!-- todo:22-008 -->


実装箇所は `nn::silu` である。この要件に対する実行可能な確認は `transformer_reference::zero_transformer` に置く。

`src/nn/qwen.rs` の `pub fn logits_all_with_attention<F>(`の定義を次の全体で置き換えます。

```rust file="src/nn/qwen.rs" mode="replace" item="pub fn logits_all_with_attention<F>("
pub fn logits_all_with_attention<F>(
	g: &mut Graph,
	c: &QwenConfig,
	params: &BTreeMap<String, NodeId>,
	tokens: NodeId,
	start: usize,
	cache: Option<&[(NodeId, NodeId)]>,
	attention_builder: &mut F,
) -> Result<Forward>
where
	F: FnMut(&mut Graph, NodeId, NodeId, NodeId, usize, bool) -> Result<NodeId>,
{
	c.validate()?;
	let shape = g.shape(tokens);
	if shape.len() != 1 || shape[0] == 0 {
		return Err("tokens must be nonempty rank one".into());
	}
	let t = shape[0];
	if start.checked_add(t).ok_or("context overflow")? > c.max_position_embeddings {
		return Err("input exceeds context capacity".into());
	}
	if cache.is_none() && start != 0 {
		return Err("nonzero position requires a cache".into());
	}
	if cache.is_some_and(|v| v.len() != c.num_hidden_layers) {
		return Err("cache layer count".into());
	}
	let get = |name: &str| {
		params
			.get(name)
			.copied()
			.ok_or_else(|| format!("missing graph parameter {name}"))
	};
	let d = c.hidden_size;
	let hd = d / c.num_attention_heads;
	let kd = c.num_key_value_heads;
	let mut x = super::attention::embedding(g, get("model.embed_tokens.weight")?, tokens)?;
	let mut boundaries = vec![x];
	let mut cache_outputs = Vec::new();
	for layer in 0..c.num_hidden_layers {
		let p = format!("model.layers.{layer}");
		let n = rms_norm(
			g,
			x,
			get(&format!("{p}.input_layernorm.weight"))?,
			c.rms_norm_eps,
		)?;
		let projection = |name: &str, heads: usize, g: &mut Graph| -> Result<NodeId> {
			let value = linear(
				g,
				n,
				get(&format!("{p}.self_attn.{name}.weight"))?,
				Some(get(&format!("{p}.self_attn.{name}.bias"))?),
			)?;
			g.reshape(value, vec![t, heads, hd])
		};
		let q = projection("q_proj", c.num_attention_heads, g)?;
		let k = projection("k_proj", kd, g)?;
		let v = projection("v_proj", kd, g)?;
		let q = apply_rope(g, q, start, c.rope_theta)?;
		let k = apply_rope(g, k, start, c.rope_theta)?;
		let (k, v) = if let Some(old) = cache {
			let (ck, cv) = old[layer];
			let expected = [c.max_position_embeddings, kd, hd];
			if g.shape(ck) != expected || g.shape(cv) != expected {
				return Err("cache capacity shape".into());
			}
			let (k, v) = if start == 0 {
				(k, v)
			} else {
				let ranges = vec![(0, start), (0, kd), (0, hd)];
				let prefix_k = g.shrink(ck, ranges.clone())?;
				let prefix_v = g.shrink(cv, ranges)?;
				(
					concatenate(g, prefix_k, k, 0)?,
					concatenate(g, prefix_v, v, 0)?,
				)
			};
			let pads = vec![(0, c.max_position_embeddings - start - t), (0, 0), (0, 0)];
			let new_k = g.pad(k, pads.clone())?;
			let new_v = g.pad(v, pads)?;
			let new_k = g.stage(new_k)?;
			let new_v = g.stage(new_v)?;
			cache_outputs.push((new_k, new_v));
			let ranges = vec![(0, start + t), (0, kd), (0, hd)];
			(g.shrink(new_k, ranges.clone())?, g.shrink(new_v, ranges)?)
		} else {
			(k, v)
		};
		let attended = attention_builder(g, q, k, v, start, true)?;
		let attended = g.reshape(attended, vec![t, d])?;
		let projected = linear(
			g,
			attended,
			get(&format!("{p}.self_attn.o_proj.weight"))?,
			None,
		)?;
		x = g.binary(Binary::Add, x, projected)?;
		let n = rms_norm(
			g,
			x,
			get(&format!("{p}.post_attention_layernorm.weight"))?,
			c.rms_norm_eps,
		)?;
		let gate = linear(g, n, get(&format!("{p}.mlp.gate_proj.weight"))?, None)?;
		let gate = silu(g, gate)?;
		let up = linear(g, n, get(&format!("{p}.mlp.up_proj.weight"))?, None)?;
		let product = g.binary(Binary::Mul, gate, up)?;
		let down = linear(g, product, get(&format!("{p}.mlp.down_proj.weight"))?, None)?;
		x = g.binary(Binary::Add, x, down)?;
		x = g.stage(x)?;
		boundaries.push(x);
	}
	let x = rms_norm(g, x, get("model.norm.weight")?, c.rms_norm_eps)?;
	boundaries.push(x);
	let output = get(if c.tie_word_embeddings {
		"model.embed_tokens.weight"
	} else {
		"lm_head.weight"
	})?;
	let logits = linear(g, x, output, None)?;
	Ok(Forward {
		logits,
		boundaries,
		cache_outputs,
	})
}
```

```sh check
cargo test --offline --test transformer_reference zero_transformer -- --exact
```

## 9. 最初に全重みゼロの fixture を動かし、全 logits が0になることを確認する。

<!-- todo:22-009 -->


`tests/transformer_reference.rs::nonzero_layer_boundaries_reference` を追加する。

```rust file=tests/transformer_reference.rs mode=append
#[test]
fn nonzero_layer_boundaries_reference() {
	let c = QwenConfig::tiny();
	let weights = initialize(&c, 11).unwrap();
	let (actual, _) = run(&c, &weights, &[1, 2, 3], None);
	let reference = reference(&c, &weights, &[1, 2, 3]);
	if let Ok(path) = std::env::var("CHIBI_REFERENCE_OUT") {
		std::fs::write(path, serde_json::to_vec_pretty(&reference).unwrap()).unwrap();
	}
	let saved: Vec<Vec<f32>> =
		serde_json::from_slice(include_bytes!("../fixtures/tiny-transformer.json")).unwrap();
	if std::env::var_os("CHIBI_REFERENCE_OUT").is_none() {
		assert_eq!(reference, saved);
	}

	for (a, b) in actual.iter().zip(reference) {
		close(&a.data, &b, 1e-5, 1e-4);
	}
}
```

実装箇所は `nn::qwen::logits_all` である。この要件に対する実行可能な確認は `transformer_reference::nonzero_layer_boundaries_reference` に置く。

```sh check
cargo test --offline --test transformer_reference nonzero_layer_boundaries_reference -- --exact
```

## 10. 次に固定された非ゼロの重みを使い、独立 reference の各層出力を fixture に保存する。

<!-- todo:22-010 -->


実装箇所は `transformer_reference::reference` である。この要件に対する実行可能な確認は `transformer_reference::nonzero_layer_boundaries_reference` に置く。

`tests/transformer_reference.rs` の `fn nonzero_layer_boundaries_reference(`の定義を次の全体で置き換えます。

```rust file="tests/transformer_reference.rs" mode="replace" item="fn nonzero_layer_boundaries_reference("
fn nonzero_layer_boundaries_reference() {
	let c = QwenConfig::tiny();
	let weights = initialize(&c, 11).unwrap();
	let (actual, _) = run(&c, &weights, &[1, 2, 3], None);
	let reference = reference(&c, &weights, &[1, 2, 3]);
	if let Ok(path) = std::env::var("CHIBI_REFERENCE_OUT") {
		std::fs::write(path, serde_json::to_vec_pretty(&reference).unwrap()).unwrap();
	}
	let saved: Vec<Vec<f32>> =
		serde_json::from_slice(include_bytes!("../fixtures/tiny-transformer.json")).unwrap();
	if std::env::var_os("CHIBI_REFERENCE_OUT").is_none() {
		assert_eq!(reference, saved);
	}

	for (a, b) in actual.iter().zip(reference) {
		close(&a.data, &b, 1e-5, 1e-4);
	}
}
```

```sh check
cargo test --offline --test transformer_reference nonzero_layer_boundaries_reference -- --exact
```

## 11. 21の manifest に固定 revision の全ファイルと SHA-256 を登録する。

<!-- todo:22-011 -->


`tests/transformer_reference.rs::complete_public_registration` を追加する。

```rust file=tests/transformer_reference.rs mode=append
#[test]
fn complete_public_registration() {
	let c: QwenConfig =
		serde_json::from_str(include_str!("../fixtures/qwen25/config.json")).unwrap();
	c.require_public_05b().unwrap();
	let registry = parameter_registry(&c).unwrap();
	assert_eq!(registry.names.len(), 24 * 12 + 2);
	for layer in 0..24 {
		assert!(
			registry
				.names
				.contains_key(&format!("model.layers.{layer}.self_attn.q_proj.bias"))
		);
	}
	assert_eq!(
		registry.names["model.embed_tokens.weight"].shape,
		[151936, 896]
	);
	assert!(!registry.names.contains_key("lm_head.weight"));
	let mut invalid = c;
	invalid.num_hidden_layers = 23;
	assert!(invalid.require_public_05b().is_err());
}
```

実装箇所は `fetch::Manifest::validate` である。この要件に対する実行可能な確認は `transformer_reference::complete_public_registration` に置く。

`src/model_assets.rs` の `pub fn verified_files(`の定義を次の全体で置き換えます。

```rust file="src/model_assets.rs" mode="replace" item="pub fn verified_files("
pub fn verified_files(
	manifest: &Path,
	directory: &Path,
	download: bool,
) -> Result<(Manifest, BTreeMap<String, Vec<u8>>)> {
	let bytes = std::fs::read(manifest).map_err(|e| e.to_string())?;
	let manifest: Manifest = serde_json::from_value(crate::safetensor::unique_json(&bytes)?)
		.map_err(|e| e.to_string())?;
	manifest.validate()?;
	let mut files = BTreeMap::new();
	for asset in &manifest.files {
		let bytes = if download {
			fetch(directory, asset)?
		} else {
			verified_read(&directory.join(&asset.name), asset)?
		};
		files.insert(asset.name.clone(), bytes);
	}
	Ok((manifest, files))
}
```

```sh check
cargo test --offline --test transformer_reference complete_public_registration -- --exact
```

## 12. 対象の model type と config の必要項目を検証し、未対応方式をエラーにする。

<!-- todo:22-012 -->


実装箇所は `nn::qwen::QwenConfig::validate` である。この要件に対する実行可能な確認は `transformer_reference::complete_public_registration` に置く。

`src/nn/qwen.rs` の `pub fn validate(`（`impl QwenConfig` 内）の定義を次の全体で置き換えます。

```rust file="src/nn/qwen.rs" mode="replace" item="pub fn validate(" within="impl QwenConfig"
	pub fn validate(&self) -> Result<()> {
		if self.use_mrope
			|| self.attention_dropout != 0.
			|| (!self.architectures.is_empty() && self.architectures != ["Qwen2ForCausalLM"])
			|| self.model_type != "qwen2"
			|| self.hidden_act != "silu"
			|| self.use_sliding_window
			|| self.rope_scaling.is_some()
		{
			return Err("unsupported Qwen architecture/activation/rope/sliding window".into());
		}
		if [
			self.hidden_size,
			self.intermediate_size,
			self.num_hidden_layers,
			self.num_attention_heads,
			self.num_key_value_heads,
			self.vocab_size,
			self.max_position_embeddings,
		]
		.contains(&0)
			|| self.hidden_size % self.num_attention_heads != 0
			|| self.num_attention_heads % self.num_key_value_heads != 0
			|| (self.hidden_size / self.num_attention_heads) % 2 != 0
			|| self.vocab_size > 16_777_216
			|| self.rms_norm_eps <= 0.
			|| !self.rms_norm_eps.is_finite()
			|| self.rope_theta <= 0.
			|| !self.rope_theta.is_finite()
		{
			return Err("invalid Qwen dimensions/norm/theta".into());
		}
		Ok(())
	}
```

```sh check
cargo test --offline --test transformer_reference complete_public_registration -- --exact
```

## 13. embedding、全 layer、norm、output の名前と shape を一対一に照合する。

<!-- todo:22-013 -->


実装箇所は `nn::qwen::parameter_registry` である。この要件に対する実行可能な確認は `transformer_reference::complete_public_registration` に置く。

`src/nn/qwen.rs` の `pub fn parameter_registry(`の定義を次の全体で置き換えます。

```rust file="src/nn/qwen.rs" mode="replace" item="pub fn parameter_registry("
pub fn parameter_registry(c: &QwenConfig) -> Result<Registry> {
	c.validate()?;
	let mut registry = Registry::default();
	let mut id = 0u32;
	let mut register = |name: String, shape: Vec<usize>| -> Result<()> {
		registry.register(
			&name,
			Spec {
				id,
				slot: id as usize,
				dtype: DType::F32,
				shape,
				trainable: true,
			},
		)?;
		id = id.checked_add(1).ok_or("parameter ID overflow")?;
		Ok(())
	};
	let d = c.hidden_size;
	let kd = c.num_key_value_heads * (d / c.num_attention_heads);
	register("model.embed_tokens.weight".into(), vec![c.vocab_size, d])?;
	for layer in 0..c.num_hidden_layers {
		let p = format!("model.layers.{layer}");
		register(format!("{p}.input_layernorm.weight"), vec![d])?;
		for (name, out) in [("q_proj", d), ("k_proj", kd), ("v_proj", kd)] {
			register(format!("{p}.self_attn.{name}.weight"), vec![out, d])?;
			register(format!("{p}.self_attn.{name}.bias"), vec![out])?;
		}
		register(format!("{p}.self_attn.o_proj.weight"), vec![d, d])?;
		register(format!("{p}.post_attention_layernorm.weight"), vec![d])?;
		register(
			format!("{p}.mlp.gate_proj.weight"),
			vec![c.intermediate_size, d],
		)?;
		register(
			format!("{p}.mlp.up_proj.weight"),
			vec![c.intermediate_size, d],
		)?;
		register(
			format!("{p}.mlp.down_proj.weight"),
			vec![d, c.intermediate_size],
		)?;
	}
	register("model.norm.weight".into(), vec![d])?;
	if !c.tie_word_embeddings {
		register("lm_head.weight".into(), vec![c.vocab_size, d])?;
	}
	Ok(registry)
}
```

```sh check
cargo test --offline --test transformer_reference complete_public_registration -- --exact
```

## 14. Q/K/V の bias を読み、projection 行列の出力軸と入力軸を確認する。

<!-- todo:22-014 -->


実装箇所は `nn::linear` である。この要件に対する実行可能な確認は `transformer_reference::complete_public_registration` に置く。

`src/nn/qwen.rs` の `pub fn logits_all_with_attention<F>(`の定義を次の全体で置き換えます。

```rust file="src/nn/qwen.rs" mode="replace" item="pub fn logits_all_with_attention<F>("
pub fn logits_all_with_attention<F>(
	g: &mut Graph,
	c: &QwenConfig,
	params: &BTreeMap<String, NodeId>,
	tokens: NodeId,
	start: usize,
	cache: Option<&[(NodeId, NodeId)]>,
	attention_builder: &mut F,
) -> Result<Forward>
where
	F: FnMut(&mut Graph, NodeId, NodeId, NodeId, usize, bool) -> Result<NodeId>,
{
	c.validate()?;
	let shape = g.shape(tokens);
	if shape.len() != 1 || shape[0] == 0 {
		return Err("tokens must be nonempty rank one".into());
	}
	let t = shape[0];
	if start.checked_add(t).ok_or("context overflow")? > c.max_position_embeddings {
		return Err("input exceeds context capacity".into());
	}
	if cache.is_none() && start != 0 {
		return Err("nonzero position requires a cache".into());
	}
	if cache.is_some_and(|v| v.len() != c.num_hidden_layers) {
		return Err("cache layer count".into());
	}
	let get = |name: &str| {
		params
			.get(name)
			.copied()
			.ok_or_else(|| format!("missing graph parameter {name}"))
	};
	let d = c.hidden_size;
	let hd = d / c.num_attention_heads;
	let kd = c.num_key_value_heads;
	let mut x = super::attention::embedding(g, get("model.embed_tokens.weight")?, tokens)?;
	let mut boundaries = vec![x];
	let mut cache_outputs = Vec::new();
	for layer in 0..c.num_hidden_layers {
		let p = format!("model.layers.{layer}");
		let n = rms_norm(
			g,
			x,
			get(&format!("{p}.input_layernorm.weight"))?,
			c.rms_norm_eps,
		)?;
		let projection = |name: &str, heads: usize, g: &mut Graph| -> Result<NodeId> {
			let value = linear(
				g,
				n,
				get(&format!("{p}.self_attn.{name}.weight"))?,
				Some(get(&format!("{p}.self_attn.{name}.bias"))?),
			)?;
			g.reshape(value, vec![t, heads, hd])
		};
		let q = projection("q_proj", c.num_attention_heads, g)?;
		let k = projection("k_proj", kd, g)?;
		let v = projection("v_proj", kd, g)?;
		let q = apply_rope(g, q, start, c.rope_theta)?;
		let k = apply_rope(g, k, start, c.rope_theta)?;
		let (k, v) = if let Some(old) = cache {
			let (ck, cv) = old[layer];
			let expected = [c.max_position_embeddings, kd, hd];
			if g.shape(ck) != expected || g.shape(cv) != expected {
				return Err("cache capacity shape".into());
			}
			let (k, v) = if start == 0 {
				(k, v)
			} else {
				let ranges = vec![(0, start), (0, kd), (0, hd)];
				let prefix_k = g.shrink(ck, ranges.clone())?;
				let prefix_v = g.shrink(cv, ranges)?;
				(
					concatenate(g, prefix_k, k, 0)?,
					concatenate(g, prefix_v, v, 0)?,
				)
			};
			let pads = vec![(0, c.max_position_embeddings - start - t), (0, 0), (0, 0)];
			let new_k = g.pad(k, pads.clone())?;
			let new_v = g.pad(v, pads)?;
			let new_k = g.stage(new_k)?;
			let new_v = g.stage(new_v)?;
			cache_outputs.push((new_k, new_v));
			let ranges = vec![(0, start + t), (0, kd), (0, hd)];
			(g.shrink(new_k, ranges.clone())?, g.shrink(new_v, ranges)?)
		} else {
			(k, v)
		};
		let attended = attention_builder(g, q, k, v, start, true)?;
		let attended = g.reshape(attended, vec![t, d])?;
		let projected = linear(
			g,
			attended,
			get(&format!("{p}.self_attn.o_proj.weight"))?,
			None,
		)?;
		x = g.binary(Binary::Add, x, projected)?;
		let n = rms_norm(
			g,
			x,
			get(&format!("{p}.post_attention_layernorm.weight"))?,
			c.rms_norm_eps,
		)?;
		let gate = linear(g, n, get(&format!("{p}.mlp.gate_proj.weight"))?, None)?;
		let gate = silu(g, gate)?;
		let up = linear(g, n, get(&format!("{p}.mlp.up_proj.weight"))?, None)?;
		let product = g.binary(Binary::Mul, gate, up)?;
		let down = linear(g, product, get(&format!("{p}.mlp.down_proj.weight"))?, None)?;
		x = g.binary(Binary::Add, x, down)?;
		x = g.stage(x)?;
		boundaries.push(x);
	}
	let x = rms_norm(g, x, get("model.norm.weight")?, c.rms_norm_eps)?;
	boundaries.push(x);
	let output = get(if c.tie_word_embeddings {
		"model.embed_tokens.weight"
	} else {
		"lm_head.weight"
	})?;
	let logits = linear(g, x, output, None)?;
	Ok(Forward {
		logits,
		boundaries,
		cache_outputs,
	})
}
```

```sh check
cargo test --offline --test transformer_reference complete_public_registration -- --exact
```

## 15. tied embedding を一つの ParamId へ結び、重複保存された重みがあれば整合性を確認する。

<!-- todo:22-015 -->


実装箇所は `nn::qwen::load_parameters` である。この要件に対する実行可能な確認は `transformer_reference::complete_public_registration` に置く。

`src/nn/qwen.rs` の `pub fn load_parameters(`の定義を次の全体で置き換えます。

```rust file="src/nn/qwen.rs" mode="replace" item="pub fn load_parameters("
pub fn load_parameters(
	c: &QwenConfig,
	loaded: &BTreeMap<String, TensorRecord>,
) -> Result<TensorSet> {
	let registry = parameter_registry(c)?;
	let mut converted = BTreeMap::new();
	for (name, tensor) in loaded {
		converted.insert(
			name.clone(),
			TensorRecord::from_buffer(&tensor.to_buffer()?),
		);
	}
	if c.tie_word_embeddings {
		if let Some(output) = converted.remove("lm_head.weight") {
			if converted.get("model.embed_tokens.weight") != Some(&output) {
				return Err("tied embedding/output payload mismatch".into());
			}
		}
	}
	registry.validate(&converted, false)
}
```

```sh check
cargo test --offline --test transformer_reference complete_public_registration -- --exact
```

## 16. 固定版の tokenizer library を使い、21で検証した tokenizer.json と設定を読み込む。

<!-- todo:22-016 -->


`tests/qwen_tokenizer.rs::official_tokenizer_ids` を追加する。

```rust file=tests/qwen_tokenizer.rs mode=append
#[test]
#[ignore]
fn official_tokenizer_ids() {
	let manifest: Manifest =
		serde_json::from_slice(include_bytes!("../fixtures/qwen25/manifest.json")).unwrap();
	manifest.validate().unwrap();
	let asset = manifest
		.files
		.iter()
		.find(|a| a.name == "tokenizer.json")
		.unwrap();
	let bytes = verified_read(
		std::path::Path::new("fixtures/qwen25/tokenizer.json"),
		asset,
	)
	.unwrap();
	let tokenizer = tokenizers::Tokenizer::from_bytes(bytes).unwrap();
	let fixture: serde_json::Value =
		serde_json::from_slice(include_bytes!("../fixtures/qwen25/tokenizer-fixture.json"))
			.unwrap();
	for row in fixture.as_array().unwrap() {
		let expected = row["ids"]
			.as_array()
			.unwrap()
			.iter()
			.map(|v| v.as_u64().unwrap() as u32)
			.collect::<Vec<_>>();
		assert_eq!(
			encode(&tokenizer, row["text"].as_str().unwrap()).unwrap(),
			expected
		);
	}
}
```

実装箇所は `model_assets::load` である。この要件に対する実行可能な確認は `qwen_tokenizer::official_tokenizer_ids` に置く。

`src/model_assets.rs` の `pub fn load(`の定義を次の全体で置き換えます。

```rust file="src/model_assets.rs" mode="replace" item="pub fn load("
pub fn load(manifest: &Path, directory: &Path, download: bool) -> Result<ModelAssets> {
	let (manifest, files) = verified_files(manifest, directory, download)?;
	let required = |name: &str| {
		files
			.get(name)
			.ok_or_else(|| format!("manifest missing {name}"))
	};
	let config: QwenConfig =
		serde_json::from_value(crate::safetensor::unique_json(required("config.json")?)?)
			.map_err(|e| e.to_string())?;
	config.require_public_05b()?;
	let raw = if let Some(index) = files.get("model.safetensors.index.json") {
		crate::fetch::load_shards(index, &files)?
	} else {
		crate::safetensor::read(required("model.safetensors")?)?
	};
	let state = load_parameters(&config, &raw)?;
	let tokenizer = tokenizers::Tokenizer::from_bytes(required("tokenizer.json")?)
		.map_err(|e| e.to_string())?;
	let tokenizer_config = crate::safetensor::unique_json(required("tokenizer_config.json")?)?;
	if tokenizer_config
		.get("tokenizer_class")
		.and_then(|v| v.as_str())
		!= Some("Qwen2Tokenizer")
	{
		return Err("unsupported tokenizer class".into());
	}
	let generation = crate::safetensor::unique_json(required("generation_config.json")?)?;
	let eos = generation
		.get("eos_token_id")
		.and_then(|v| v.as_u64())
		.and_then(|v| u32::try_from(v).ok())
		.ok_or("EOS token ID")?;
	if eos as usize >= config.vocab_size {
		return Err("EOS vocabulary range".into());
	}
	Ok(ModelAssets {
		config,
		state,
		tokenizer,
		manifest,
		eos,
	})
}
```

```sh check requires=qwen
cargo test --offline --test qwen_tokenizer official_tokenizer_ids -- --exact --ignored
```

巨大な tokenizer.json は本文へ埋め込まず、掲載した固定 URL、長さ、SHA-256 で取得する。chat template は適用しない。

```sh check requires=qwen
cargo run --offline --example fetch_qwen -- --tokenizer-only
cargo run --offline --example qwen_tokenize
```

## 17. 空白、日本語、改行、特殊 token の ID 列を公式 tokenizer の固定 fixture と一致させる。

<!-- todo:22-017 -->


実装箇所は `model_assets::encode` である。この要件に対する実行可能な確認は `qwen_tokenizer::official_tokenizer_ids` に置く。

`tests/qwen_tokenizer.rs` の `fn official_tokenizer_ids(`の定義を次の全体で置き換えます。

```rust file="tests/qwen_tokenizer.rs" mode="replace" item="fn official_tokenizer_ids("
fn official_tokenizer_ids() {
	let manifest: Manifest =
		serde_json::from_slice(include_bytes!("../fixtures/qwen25/manifest.json")).unwrap();
	manifest.validate().unwrap();
	let asset = manifest
		.files
		.iter()
		.find(|a| a.name == "tokenizer.json")
		.unwrap();
	let bytes = verified_read(
		std::path::Path::new("fixtures/qwen25/tokenizer.json"),
		asset,
	)
	.unwrap();
	let tokenizer = tokenizers::Tokenizer::from_bytes(bytes).unwrap();
	let fixture: serde_json::Value =
		serde_json::from_slice(include_bytes!("../fixtures/qwen25/tokenizer-fixture.json"))
			.unwrap();
	for row in fixture.as_array().unwrap() {
		let expected = row["ids"]
			.as_array()
			.unwrap()
			.iter()
			.map(|v| v.as_u64().unwrap() as u32)
			.collect::<Vec<_>>();
		assert_eq!(
			encode(&tokenizer, row["text"].as_str().unwrap()).unwrap(),
			expected
		);
	}
}
```

```sh check requires=qwen
cargo test --offline --test qwen_tokenizer official_tokenizer_ids -- --exact --ignored
```

## 18. chat template は最初は使わず、同じ生テキストと token ID を reference に渡す。

<!-- todo:22-018 -->


実装箇所は `model_assets::encode` である。この要件に対する実行可能な確認は `qwen_tokenizer::official_tokenizer_ids` に置く。

`src/model_assets.rs` の `pub fn encode(`の定義を次の全体で置き換えます。

```rust file="src/model_assets.rs" mode="replace" item="pub fn encode("
pub fn encode(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>> {
	tokenizer
		.encode(text, false)
		.map(|e| e.get_ids().to_vec())
		.map_err(|e| e.to_string())
}
```

```sh check requires=qwen
cargo test --offline --test qwen_tokenizer official_tokenizer_ids -- --exact --ignored
```

## 19. 全24 layer を f32 で実行し、embedding、層境界、最終 logits を reference と比較する。

<!-- todo:22-019 -->


`tests/qwen_public.rs::all_24_layers_reference` を追加する。

```rust file=tests/qwen_public.rs mode=append
#[test]
#[ignore]
fn all_24_layers_reference() {
	let directory = Path::new("fixtures/qwen25");
	let assets = model_assets::load(&directory.join("manifest.json"), directory, false).unwrap();
	let reference: serde_json::Value = serde_json::from_slice(
		&std::fs::read(directory.join("reference.json"))
			.expect("run tools/qwen_reference.py first"),
	)
	.unwrap();
	assert_eq!(
		reference["revision"].as_str().unwrap(),
		assets.manifest.revision
	);
	let ids = reference["ids"]
		.as_array()
		.unwrap()
		.iter()
		.map(|v| v.as_u64().unwrap() as u32)
		.collect::<Vec<_>>();
	let mut c = assets.config;
	c.max_position_embeddings = 16;
	let registry = parameter_registry(&c).unwrap();
	let mut graph = Graph::default();
	let params = register_graph(&mut graph, &registry).unwrap();
	let mut bindings = registry.bind(&assets.state).unwrap();
	let tokens = graph.input(bindings.len(), vec![ids.len()]).unwrap();
	bindings.push(token_buffer(&ids, &c).unwrap());
	let forward = logits_all(&mut graph, &c, &params, tokens, 0, None).unwrap();
	assert_eq!(forward.boundaries.len(), 26);
	let mut outputs = forward.boundaries;
	outputs.push(forward.logits);
	let actual = graph.compile(&outputs).unwrap().run(&bindings).unwrap();
	let mut expected = reference["boundaries"].as_array().unwrap().clone();
	expected.push(reference["logits"].clone());
	for (layer, (a, b)) in actual.iter().zip(expected).enumerate() {
		let values = b.as_array().unwrap();
		assert_eq!(a.data.len(), values.len());
		let mut max = 0f32;
		for (&value, reference) in a.data.iter().zip(values) {
			let reference = reference.as_f64().unwrap() as f32;
			max = max.max((value - reference).abs());
			assert!(
				value.is_finite() && (value - reference).abs() <= 1e-3 + 1e-3 * reference.abs(),
				"boundary={layer} value={value} ref={reference} max={max}"
			);
		}
		eprintln!("boundary={layer} shape={:?} max_abs_error={max}", a.shape);
	}
}
```

実装箇所は `nn::qwen::logits_all` である。この要件に対する実行可能な確認は `qwen_public::all_24_layers_reference` に置く。

```sh check requires=qwen
cargo test --offline --test qwen_public all_24_layers_reference -- --exact --ignored
```

全重みの取得後、固定版の公式実装で各層と logits の reference を作り、Rust の全24層と比較する。最大絶対誤差を層ごとに表示し、argmax だけを判定にしない。

```sh check requires=qwen
cargo run --offline --example fetch_qwen
uv run --with-requirements tools/qwen-reference-requirements.txt python tools/qwen_reference.py --assets fixtures/qwen25 --text Hello
cargo test --offline --test qwen_public all_24_layers_reference -- --exact --ignored
cargo run --offline --example qwen_infer -- --manifest fixtures/qwen25/manifest.json --context 16 --text Hello --max-new-tokens 1 --dump
```

## 20. 容量を超える入力は起動前に拒否し、まず context=16など短い容量で一つの次 token を出す。

<!-- todo:22-020 -->


実装箇所は `nn::qwen::token_buffer` である。この要件に対する実行可能な確認は `qwen_public::all_24_layers_reference` に置く。

`examples/qwen_infer.rs` の `fn main(`の定義を次の全体で置き換えます。

```rust file="examples/qwen_infer.rs" mode="replace" item="fn main("
fn main() -> Result<()> {
	let args = std::env::args().collect::<Vec<_>>();
	let arg = |name: &str| args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone());
	let manifest = arg("--manifest").unwrap_or("fixtures/qwen25/manifest.json".into());
	let directory = arg("--assets").unwrap_or("fixtures/qwen25".into());
	let assets = model_assets::load(
		Path::new(&manifest),
		Path::new(&directory),
		args.iter().any(|a| a == "--download"),
	)?;
	let mut c = assets.config.clone();
	c.max_position_embeddings = arg("--context")
		.unwrap_or("16".into())
		.parse::<usize>()
		.map_err(|e| e.to_string())?;
	if c.max_position_embeddings > assets.config.max_position_embeddings
		|| c.max_position_embeddings == 0
	{
		return Err("requested context capacity".into());
	}
	let count = arg("--max-new-tokens")
		.unwrap_or("1".into())
		.parse::<usize>()
		.map_err(|e| e.to_string())?;
	let text = arg("--text").unwrap_or("Hello".into());
	let mut input = model_assets::encode(&assets.tokenizer, &text)?;
	if input
		.len()
		.checked_add(count)
		.ok_or("token capacity overflow")?
		> c.max_position_embeddings
	{
		return Err("prompt plus generation exceeds context before compilation".into());
	}
	let registry = parameter_registry(&c)?;
	let weight_bindings = registry.bind(&assets.state)?;
	let slot = weight_bindings.len();
	let mut cache = KvState::new(&c)?;
	let mut generated = Vec::new();
	for _ in 0..count {
		let mut graph = Graph::default();
		let params = register_graph(&mut graph, &registry)?;
		let tokens = graph.input(slot, vec![input.len()])?;
		let mut bindings = weight_bindings.clone();
		bindings.push(token_buffer(&input, &c)?);
		let mut cache_ids = Vec::new();
		for (layer, (key, value)) in cache.keys.iter().zip(&cache.values).enumerate() {
			cache_ids.push((
				graph.input(slot + 1 + 2 * layer, key.shape.clone())?,
				graph.input(slot + 2 + 2 * layer, value.shape.clone())?,
			));
			bindings.extend([key.clone(), value.clone()]);
		}
		let forward = logits_all(
			&mut graph,
			&c,
			&params,
			tokens,
			cache.valid,
			Some(&cache_ids),
		)?;
		let mut outputs = vec![forward.logits];
		for (k, v) in forward.cache_outputs {
			outputs.extend([k, v]);
		}
		let plan = graph.compile(&outputs)?;
		let result = plan.run(&bindings)?;
		let next = greedy(&result[0])?;
		if args.iter().any(|a| a == "--dump") {
			eprintln!(
				"tokens={input:?} logits={:?} layers={} heads={}/{} position={} cache_valid={} graph_nodes={}",
				result[0].shape,
				c.num_hidden_layers,
				c.num_attention_heads,
				c.num_key_value_heads,
				cache.valid,
				cache.valid,
				graph.nodes.len()
			);
		}
		cache.adopt(&result[1..], input.len())?;
		generated.push(next as u32);
		if next as u32 == assets.eos {
			break;
		}
		input = vec![next as u32];
	}
	let output = assets
		.tokenizer
		.decode(&generated, false)
		.map_err(|e| e.to_string())?;
	println!("ids={generated:?}\n{output}");
	Ok(())
}
```

```sh check requires=qwen
cargo test --offline --test qwen_public all_24_layers_reference -- --exact --ignored
```

## 21. layer ごとに K と V の固定容量 buffer を持ち、読取り専用の実行時入力スロットへ渡す。

<!-- todo:22-021 -->


`tests/transformer_reference.rs::immutable_cache_matches_prefix` を追加する。

```rust file=tests/transformer_reference.rs mode=append
#[test]
fn immutable_cache_matches_prefix() {
	let c = QwenConfig::tiny();
	let weights = initialize(&c, 7).unwrap();
	let mut cache = KvState::new(&c).unwrap();
	let old = cache.keys[0].data.clone();
	let (_, outputs) = run(&c, &weights, &[1, 2], Some(&cache));
	assert_eq!(cache.keys[0].data, old);
	cache.adopt(&outputs, 2).unwrap();
	let (cached, outputs) = run(&c, &weights, &[3], Some(&cache));
	cache.adopt(&outputs, 1).unwrap();
	let (full, _) = run(&c, &weights, &[1, 2, 3], None);
	let v = c.vocab_size;
	close(&cached[0].data, &full[0].data[2 * v..3 * v], 1e-5, 1e-4);
	assert_eq!(cache.valid, 3);
	cache.reset();
	assert_eq!(cache.valid, 0);
}
```

実装箇所は `nn::qwen::logits_all` の cache 候補構築と `nn::qwen::KvState::{adopt,reset}` である。この要件に対する実行可能な確認は `transformer_reference::immutable_cache_matches_prefix` に置く。

`src/nn/qwen.rs` の `pub fn new(`（`impl KvState` 内）の定義を次の全体で置き換えます。

```rust file="src/nn/qwen.rs" mode="replace" item="pub fn new(" within="impl KvState"
	pub fn new(c: &QwenConfig) -> Result<Self> {
		let shape = vec![
			c.max_position_embeddings,
			c.num_key_value_heads,
			c.hidden_size / c.num_attention_heads,
		];
		let elements = shape.iter().product();
		let buffer = Buffer::new(shape, vec![0.; elements])?;
		Ok(Self {
			keys: vec![buffer.clone(); c.num_hidden_layers],
			values: vec![buffer; c.num_hidden_layers],
			valid: 0,
			capacity: c.max_position_embeddings,
		})
	}
```

```sh check
cargo test --offline --test transformer_reference immutable_cache_matches_prefix -- --exact
```

## 22. `start_pos+T<=context` を検証し、RoPE の絶対位置と書込み位置を一致させる。

<!-- todo:22-022 -->


実装箇所は `nn::qwen::logits_all` の cache 候補構築と `nn::qwen::KvState::{adopt,reset}` である。この要件に対する実行可能な確認は `transformer_reference::immutable_cache_matches_prefix` に置く。

`src/nn/qwen.rs` の `pub fn logits_all_with_attention<F>(`の定義を次の全体で置き換えます。

```rust file="src/nn/qwen.rs" mode="replace" item="pub fn logits_all_with_attention<F>("
pub fn logits_all_with_attention<F>(
	g: &mut Graph,
	c: &QwenConfig,
	params: &BTreeMap<String, NodeId>,
	tokens: NodeId,
	start: usize,
	cache: Option<&[(NodeId, NodeId)]>,
	attention_builder: &mut F,
) -> Result<Forward>
where
	F: FnMut(&mut Graph, NodeId, NodeId, NodeId, usize, bool) -> Result<NodeId>,
{
	c.validate()?;
	let shape = g.shape(tokens);
	if shape.len() != 1 || shape[0] == 0 {
		return Err("tokens must be nonempty rank one".into());
	}
	let t = shape[0];
	if start.checked_add(t).ok_or("context overflow")? > c.max_position_embeddings {
		return Err("input exceeds context capacity".into());
	}
	if cache.is_none() && start != 0 {
		return Err("nonzero position requires a cache".into());
	}
	if cache.is_some_and(|v| v.len() != c.num_hidden_layers) {
		return Err("cache layer count".into());
	}
	let get = |name: &str| {
		params
			.get(name)
			.copied()
			.ok_or_else(|| format!("missing graph parameter {name}"))
	};
	let d = c.hidden_size;
	let hd = d / c.num_attention_heads;
	let kd = c.num_key_value_heads;
	let mut x = super::attention::embedding(g, get("model.embed_tokens.weight")?, tokens)?;
	let mut boundaries = vec![x];
	let mut cache_outputs = Vec::new();
	for layer in 0..c.num_hidden_layers {
		let p = format!("model.layers.{layer}");
		let n = rms_norm(
			g,
			x,
			get(&format!("{p}.input_layernorm.weight"))?,
			c.rms_norm_eps,
		)?;
		let projection = |name: &str, heads: usize, g: &mut Graph| -> Result<NodeId> {
			let value = linear(
				g,
				n,
				get(&format!("{p}.self_attn.{name}.weight"))?,
				Some(get(&format!("{p}.self_attn.{name}.bias"))?),
			)?;
			g.reshape(value, vec![t, heads, hd])
		};
		let q = projection("q_proj", c.num_attention_heads, g)?;
		let k = projection("k_proj", kd, g)?;
		let v = projection("v_proj", kd, g)?;
		let q = apply_rope(g, q, start, c.rope_theta)?;
		let k = apply_rope(g, k, start, c.rope_theta)?;
		let (k, v) = if let Some(old) = cache {
			let (ck, cv) = old[layer];
			let expected = [c.max_position_embeddings, kd, hd];
			if g.shape(ck) != expected || g.shape(cv) != expected {
				return Err("cache capacity shape".into());
			}
			let (k, v) = if start == 0 {
				(k, v)
			} else {
				let ranges = vec![(0, start), (0, kd), (0, hd)];
				let prefix_k = g.shrink(ck, ranges.clone())?;
				let prefix_v = g.shrink(cv, ranges)?;
				(
					concatenate(g, prefix_k, k, 0)?,
					concatenate(g, prefix_v, v, 0)?,
				)
			};
			let pads = vec![(0, c.max_position_embeddings - start - t), (0, 0), (0, 0)];
			let new_k = g.pad(k, pads.clone())?;
			let new_v = g.pad(v, pads)?;
			let new_k = g.stage(new_k)?;
			let new_v = g.stage(new_v)?;
			cache_outputs.push((new_k, new_v));
			let ranges = vec![(0, start + t), (0, kd), (0, hd)];
			(g.shrink(new_k, ranges.clone())?, g.shrink(new_v, ranges)?)
		} else {
			(k, v)
		};
		let attended = attention_builder(g, q, k, v, start, true)?;
		let attended = g.reshape(attended, vec![t, d])?;
		let projected = linear(
			g,
			attended,
			get(&format!("{p}.self_attn.o_proj.weight"))?,
			None,
		)?;
		x = g.binary(Binary::Add, x, projected)?;
		let n = rms_norm(
			g,
			x,
			get(&format!("{p}.post_attention_layernorm.weight"))?,
			c.rms_norm_eps,
		)?;
		let gate = linear(g, n, get(&format!("{p}.mlp.gate_proj.weight"))?, None)?;
		let gate = silu(g, gate)?;
		let up = linear(g, n, get(&format!("{p}.mlp.up_proj.weight"))?, None)?;
		let product = g.binary(Binary::Mul, gate, up)?;
		let down = linear(g, product, get(&format!("{p}.mlp.down_proj.weight"))?, None)?;
		x = g.binary(Binary::Add, x, down)?;
		x = g.stage(x)?;
		boundaries.push(x);
	}
	let x = rms_norm(g, x, get("model.norm.weight")?, c.rms_norm_eps)?;
	boundaries.push(x);
	let output = get(if c.tie_word_embeddings {
		"model.embed_tokens.weight"
	} else {
		"lm_head.weight"
	})?;
	let logits = linear(g, x, output, None)?;
	Ok(Forward {
		logits,
		boundaries,
		cache_outputs,
	})
}
```

```sh check
cargo test --offline --test transformer_reference immutable_cache_matches_prefix -- --exact
```

## 23. 既存の有効 prefix と今回の K/V を新規出力 buffer へ保存し、その AFTER を attention が読むようにする。

<!-- todo:22-023 -->


実装箇所は `nn::qwen::logits_all` の cache 候補構築と `nn::qwen::KvState::{adopt,reset}` である。この要件に対する実行可能な確認は `transformer_reference::immutable_cache_matches_prefix` に置く。

`src/nn/qwen.rs` の `pub fn logits_all_with_attention<F>(`の定義を次の全体で置き換えます。

```rust file="src/nn/qwen.rs" mode="replace" item="pub fn logits_all_with_attention<F>("
pub fn logits_all_with_attention<F>(
	g: &mut Graph,
	c: &QwenConfig,
	params: &BTreeMap<String, NodeId>,
	tokens: NodeId,
	start: usize,
	cache: Option<&[(NodeId, NodeId)]>,
	attention_builder: &mut F,
) -> Result<Forward>
where
	F: FnMut(&mut Graph, NodeId, NodeId, NodeId, usize, bool) -> Result<NodeId>,
{
	c.validate()?;
	let shape = g.shape(tokens);
	if shape.len() != 1 || shape[0] == 0 {
		return Err("tokens must be nonempty rank one".into());
	}
	let t = shape[0];
	if start.checked_add(t).ok_or("context overflow")? > c.max_position_embeddings {
		return Err("input exceeds context capacity".into());
	}
	if cache.is_none() && start != 0 {
		return Err("nonzero position requires a cache".into());
	}
	if cache.is_some_and(|v| v.len() != c.num_hidden_layers) {
		return Err("cache layer count".into());
	}
	let get = |name: &str| {
		params
			.get(name)
			.copied()
			.ok_or_else(|| format!("missing graph parameter {name}"))
	};
	let d = c.hidden_size;
	let hd = d / c.num_attention_heads;
	let kd = c.num_key_value_heads;
	let mut x = super::attention::embedding(g, get("model.embed_tokens.weight")?, tokens)?;
	let mut boundaries = vec![x];
	let mut cache_outputs = Vec::new();
	for layer in 0..c.num_hidden_layers {
		let p = format!("model.layers.{layer}");
		let n = rms_norm(
			g,
			x,
			get(&format!("{p}.input_layernorm.weight"))?,
			c.rms_norm_eps,
		)?;
		let projection = |name: &str, heads: usize, g: &mut Graph| -> Result<NodeId> {
			let value = linear(
				g,
				n,
				get(&format!("{p}.self_attn.{name}.weight"))?,
				Some(get(&format!("{p}.self_attn.{name}.bias"))?),
			)?;
			g.reshape(value, vec![t, heads, hd])
		};
		let q = projection("q_proj", c.num_attention_heads, g)?;
		let k = projection("k_proj", kd, g)?;
		let v = projection("v_proj", kd, g)?;
		let q = apply_rope(g, q, start, c.rope_theta)?;
		let k = apply_rope(g, k, start, c.rope_theta)?;
		let (k, v) = if let Some(old) = cache {
			let (ck, cv) = old[layer];
			let expected = [c.max_position_embeddings, kd, hd];
			if g.shape(ck) != expected || g.shape(cv) != expected {
				return Err("cache capacity shape".into());
			}
			let (k, v) = if start == 0 {
				(k, v)
			} else {
				let ranges = vec![(0, start), (0, kd), (0, hd)];
				let prefix_k = g.shrink(ck, ranges.clone())?;
				let prefix_v = g.shrink(cv, ranges)?;
				(
					concatenate(g, prefix_k, k, 0)?,
					concatenate(g, prefix_v, v, 0)?,
				)
			};
			let pads = vec![(0, c.max_position_embeddings - start - t), (0, 0), (0, 0)];
			let new_k = g.pad(k, pads.clone())?;
			let new_v = g.pad(v, pads)?;
			let new_k = g.stage(new_k)?;
			let new_v = g.stage(new_v)?;
			cache_outputs.push((new_k, new_v));
			let ranges = vec![(0, start + t), (0, kd), (0, hd)];
			(g.shrink(new_k, ranges.clone())?, g.shrink(new_v, ranges)?)
		} else {
			(k, v)
		};
		let attended = attention_builder(g, q, k, v, start, true)?;
		let attended = g.reshape(attended, vec![t, d])?;
		let projected = linear(
			g,
			attended,
			get(&format!("{p}.self_attn.o_proj.weight"))?,
			None,
		)?;
		x = g.binary(Binary::Add, x, projected)?;
		let n = rms_norm(
			g,
			x,
			get(&format!("{p}.post_attention_layernorm.weight"))?,
			c.rms_norm_eps,
		)?;
		let gate = linear(g, n, get(&format!("{p}.mlp.gate_proj.weight"))?, None)?;
		let gate = silu(g, gate)?;
		let up = linear(g, n, get(&format!("{p}.mlp.up_proj.weight"))?, None)?;
		let product = g.binary(Binary::Mul, gate, up)?;
		let down = linear(g, product, get(&format!("{p}.mlp.down_proj.weight"))?, None)?;
		x = g.binary(Binary::Add, x, down)?;
		x = g.stage(x)?;
		boundaries.push(x);
	}
	let x = rms_norm(g, x, get("model.norm.weight")?, c.rms_norm_eps)?;
	boundaries.push(x);
	let output = get(if c.tie_word_embeddings {
		"model.embed_tokens.weight"
	} else {
		"lm_head.weight"
	})?;
	let logits = linear(g, x, output, None)?;
	Ok(Forward {
		logits,
		boundaries,
		cache_outputs,
	})
}
```

```sh check
cargo test --offline --test transformer_reference immutable_cache_matches_prefix -- --exact
```

## 24. 実行完了後に新しい cache 出力を次回の入力スロットへ束縛し、元の入力 buffer は変更しない。

<!-- todo:22-024 -->


実装箇所は `nn::qwen::logits_all` の cache 候補構築と `nn::qwen::KvState::{adopt,reset}` である。この要件に対する実行可能な確認は `transformer_reference::immutable_cache_matches_prefix` に置く。

`src/nn/qwen.rs` の `pub fn adopt(`（`impl KvState` 内）の定義を次の全体で置き換えます。

```rust file="src/nn/qwen.rs" mode="replace" item="pub fn adopt(" within="impl KvState"
	pub fn adopt(&mut self, outputs: &[Buffer], tokens: usize) -> Result<()> {
		if outputs.len() != 2 * self.keys.len()
			|| self
				.valid
				.checked_add(tokens)
				.ok_or("cache length overflow")?
				> self.capacity
		{
			return Err("cache adoption shape or capacity".into());
		}
		let mut candidate = self.clone();
		for (i, pair) in outputs.chunks_exact(2).enumerate() {
			if pair[0].shape != self.keys[i].shape || pair[1].shape != self.values[i].shape {
				return Err("cache output shape".into());
			}
			candidate.keys[i] = pair[0].clone();
			candidate.values[i] = pair[1].clone();
		}
		candidate.valid += tokens;
		*self = candidate;
		Ok(())
	}
```

```sh check
cargo test --offline --test transformer_reference immutable_cache_matches_prefix -- --exact
```

## 25. prefill と1 token decode を、cache なしで prefix 全体を再計算した logits と比較する。

<!-- todo:22-025 -->


実装箇所は `nn::qwen::logits_all` の cache 候補構築と `nn::qwen::KvState::{adopt,reset}` である。この要件に対する実行可能な確認は `transformer_reference::immutable_cache_matches_prefix` に置く。

`tests/transformer_reference.rs` の `fn immutable_cache_matches_prefix(`の定義を次の全体で置き換えます。

```rust file="tests/transformer_reference.rs" mode="replace" item="fn immutable_cache_matches_prefix("
fn immutable_cache_matches_prefix() {
	let c = QwenConfig::tiny();
	let weights = initialize(&c, 7).unwrap();
	let mut cache = KvState::new(&c).unwrap();
	let old = cache.keys[0].data.clone();
	let (_, outputs) = run(&c, &weights, &[1, 2], Some(&cache));
	assert_eq!(cache.keys[0].data, old);
	cache.adopt(&outputs, 2).unwrap();
	let (cached, outputs) = run(&c, &weights, &[3], Some(&cache));
	cache.adopt(&outputs, 1).unwrap();
	let (full, _) = run(&c, &weights, &[1, 2, 3], None);
	let v = c.vocab_size;
	close(&cached[0].data, &full[0].data[2 * v..3 * v], 1e-5, 1e-4);
	assert_eq!(cache.valid, 3);
	cache.reset();
	assert_eq!(cache.valid, 0);
}
```

```sh check
cargo test --offline --test transformer_reference immutable_cache_matches_prefix -- --exact
```

## 26. 新しい prompt の開始時に cache の有効長を初期化し、greedy 生成と EOS 終了を追加する。

<!-- todo:22-026 -->


実装箇所は `nn::qwen::logits_all` の cache 候補構築と `nn::qwen::KvState::{adopt,reset}` である。この要件に対する実行可能な確認は `transformer_reference::immutable_cache_matches_prefix` に置く。

`examples/qwen_infer.rs` の `fn main(`の定義を次の全体で置き換えます。

```rust file="examples/qwen_infer.rs" mode="replace" item="fn main("
fn main() -> Result<()> {
	let args = std::env::args().collect::<Vec<_>>();
	let arg = |name: &str| args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone());
	let manifest = arg("--manifest").unwrap_or("fixtures/qwen25/manifest.json".into());
	let directory = arg("--assets").unwrap_or("fixtures/qwen25".into());
	let assets = model_assets::load(
		Path::new(&manifest),
		Path::new(&directory),
		args.iter().any(|a| a == "--download"),
	)?;
	let mut c = assets.config.clone();
	c.max_position_embeddings = arg("--context")
		.unwrap_or("16".into())
		.parse::<usize>()
		.map_err(|e| e.to_string())?;
	if c.max_position_embeddings > assets.config.max_position_embeddings
		|| c.max_position_embeddings == 0
	{
		return Err("requested context capacity".into());
	}
	let count = arg("--max-new-tokens")
		.unwrap_or("1".into())
		.parse::<usize>()
		.map_err(|e| e.to_string())?;
	let text = arg("--text").unwrap_or("Hello".into());
	let mut input = model_assets::encode(&assets.tokenizer, &text)?;
	if input
		.len()
		.checked_add(count)
		.ok_or("token capacity overflow")?
		> c.max_position_embeddings
	{
		return Err("prompt plus generation exceeds context before compilation".into());
	}
	let registry = parameter_registry(&c)?;
	let weight_bindings = registry.bind(&assets.state)?;
	let slot = weight_bindings.len();
	let mut cache = KvState::new(&c)?;
	let mut generated = Vec::new();
	for _ in 0..count {
		let mut graph = Graph::default();
		let params = register_graph(&mut graph, &registry)?;
		let tokens = graph.input(slot, vec![input.len()])?;
		let mut bindings = weight_bindings.clone();
		bindings.push(token_buffer(&input, &c)?);
		let mut cache_ids = Vec::new();
		for (layer, (key, value)) in cache.keys.iter().zip(&cache.values).enumerate() {
			cache_ids.push((
				graph.input(slot + 1 + 2 * layer, key.shape.clone())?,
				graph.input(slot + 2 + 2 * layer, value.shape.clone())?,
			));
			bindings.extend([key.clone(), value.clone()]);
		}
		let forward = logits_all(
			&mut graph,
			&c,
			&params,
			tokens,
			cache.valid,
			Some(&cache_ids),
		)?;
		let mut outputs = vec![forward.logits];
		for (k, v) in forward.cache_outputs {
			outputs.extend([k, v]);
		}
		let plan = graph.compile(&outputs)?;
		let result = plan.run(&bindings)?;
		let next = greedy(&result[0])?;
		if args.iter().any(|a| a == "--dump") {
			eprintln!(
				"tokens={input:?} logits={:?} layers={} heads={}/{} position={} cache_valid={} graph_nodes={}",
				result[0].shape,
				c.num_hidden_layers,
				c.num_attention_heads,
				c.num_key_value_heads,
				cache.valid,
				cache.valid,
				graph.nodes.len()
			);
		}
		cache.adopt(&result[1..], input.len())?;
		generated.push(next as u32);
		if next as u32 == assets.eos {
			break;
		}
		input = vec![next as u32];
	}
	let output = assets
		.tokenizer
		.decode(&generated, false)
		.map_err(|e| e.to_string())?;
	println!("ids={generated:?}\n{output}");
	Ok(())
}
```

```sh check
cargo test --offline --test transformer_reference immutable_cache_matches_prefix -- --exact
```

## 27. 小例の causal 平均、softmax 行和、RoPE 位置0が期待どおりになる。

<!-- todo:22-027 -->


実装箇所は `transformer_reference::reference` と `nn::qwen::logits_all` である。この要件に対する実行可能な確認は `transformer_reference::immutable_cache_matches_prefix` に置く。

`tests/transformer_reference.rs` の `fn immutable_cache_matches_prefix(`の定義を次の全体で置き換えます。

```rust file="tests/transformer_reference.rs" mode="replace" item="fn immutable_cache_matches_prefix("
fn immutable_cache_matches_prefix() {
	let c = QwenConfig::tiny();
	let weights = initialize(&c, 7).unwrap();
	let mut cache = KvState::new(&c).unwrap();
	let old = cache.keys[0].data.clone();
	let (_, outputs) = run(&c, &weights, &[1, 2], Some(&cache));
	assert_eq!(cache.keys[0].data, old);
	cache.adopt(&outputs, 2).unwrap();
	let (cached, outputs) = run(&c, &weights, &[3], Some(&cache));
	cache.adopt(&outputs, 1).unwrap();
	let (full, _) = run(&c, &weights, &[1, 2, 3], None);
	let v = c.vocab_size;
	close(&cached[0].data, &full[0].data[2 * v..3 * v], 1e-5, 1e-4);
	assert_eq!(cache.valid, 3);
	cache.reset();
	assert_eq!(cache.valid, 0);
}
```

```sh check
cargo test --offline --test transformer_reference immutable_cache_matches_prefix -- --exact
```

## 28. f32 の小 fixture は `abs_err <= 1e-5 + 1e-4*abs(ref)` を初期基準に各層を比較する。

<!-- todo:22-028 -->


実装箇所は `transformer_reference::reference` と `nn::qwen::logits_all` である。この要件に対する実行可能な確認は `transformer_reference::nonzero_layer_boundaries_reference` に置く。

`tests/transformer_reference.rs` の `fn nonzero_layer_boundaries_reference(`の定義を次の全体で置き換えます。

```rust file="tests/transformer_reference.rs" mode="replace" item="fn nonzero_layer_boundaries_reference("
fn nonzero_layer_boundaries_reference() {
	let c = QwenConfig::tiny();
	let weights = initialize(&c, 11).unwrap();
	let (actual, _) = run(&c, &weights, &[1, 2, 3], None);
	let reference = reference(&c, &weights, &[1, 2, 3]);
	if let Ok(path) = std::env::var("CHIBI_REFERENCE_OUT") {
		std::fs::write(path, serde_json::to_vec_pretty(&reference).unwrap()).unwrap();
	}
	let saved: Vec<Vec<f32>> =
		serde_json::from_slice(include_bytes!("../fixtures/tiny-transformer.json")).unwrap();
	if std::env::var_os("CHIBI_REFERENCE_OUT").is_none() {
		assert_eq!(reference, saved);
	}

	for (a, b) in actual.iter().zip(reference) {
		close(&a.data, &b, 1e-5, 1e-4);
	}
}
```

```sh check
cargo test --offline --test transformer_reference nonzero_layer_boundaries_reference -- --exact
```

## 29. 実モデルは各層と logits を `abs_err <= 1e-3 + 1e-3*abs(ref)` で比較し、最大誤差を残す。

<!-- todo:22-029 -->


実装箇所は `transformer_reference::reference` と `nn::qwen::logits_all` である。この要件に対する実行可能な確認は `transformer_reference::immutable_cache_matches_prefix` に置く。

`tests/transformer_reference.rs` の `fn immutable_cache_matches_prefix(`の定義を次の全体で置き換えます。

```rust file="tests/transformer_reference.rs" mode="replace" item="fn immutable_cache_matches_prefix("
fn immutable_cache_matches_prefix() {
	let c = QwenConfig::tiny();
	let weights = initialize(&c, 7).unwrap();
	let mut cache = KvState::new(&c).unwrap();
	let old = cache.keys[0].data.clone();
	let (_, outputs) = run(&c, &weights, &[1, 2], Some(&cache));
	assert_eq!(cache.keys[0].data, old);
	cache.adopt(&outputs, 2).unwrap();
	let (cached, outputs) = run(&c, &weights, &[3], Some(&cache));
	cache.adopt(&outputs, 1).unwrap();
	let (full, _) = run(&c, &weights, &[1, 2, 3], None);
	let v = c.vocab_size;
	close(&cached[0].data, &full[0].data[2 * v..3 * v], 1e-5, 1e-4);
	assert_eq!(cache.valid, 3);
	cache.reset();
	assert_eq!(cache.valid, 0);
}
```

```sh check
cargo test --offline --test transformer_reference immutable_cache_matches_prefix -- --exact
```

## 30. argmax の一致だけで数値検証を終えず、近接した logits の差も記録する。

<!-- todo:22-030 -->


実装箇所は `transformer_reference::reference` と `nn::qwen::logits_all` である。この要件に対する実行可能な確認は `transformer_reference::immutable_cache_matches_prefix` に置く。

`tests/transformer_reference.rs` の `fn immutable_cache_matches_prefix(`の定義を次の全体で置き換えます。

```rust file="tests/transformer_reference.rs" mode="replace" item="fn immutable_cache_matches_prefix("
fn immutable_cache_matches_prefix() {
	let c = QwenConfig::tiny();
	let weights = initialize(&c, 7).unwrap();
	let mut cache = KvState::new(&c).unwrap();
	let old = cache.keys[0].data.clone();
	let (_, outputs) = run(&c, &weights, &[1, 2], Some(&cache));
	assert_eq!(cache.keys[0].data, old);
	cache.adopt(&outputs, 2).unwrap();
	let (cached, outputs) = run(&c, &weights, &[3], Some(&cache));
	cache.adopt(&outputs, 1).unwrap();
	let (full, _) = run(&c, &weights, &[1, 2, 3], None);
	let v = c.vocab_size;
	close(&cached[0].data, &full[0].data[2 * v..3 * v], 1e-5, 1e-4);
	assert_eq!(cache.valid, 3);
	cache.reset();
	assert_eq!(cache.valid, 0);
}
```

```sh check
cargo test --offline --test transformer_reference immutable_cache_matches_prefix -- --exact
```

## 31. cache 有無と chunk 分割の違いで同じ prefix の logits が許容値内になる。

<!-- todo:22-031 -->


実装箇所は `transformer_reference::reference` と `nn::qwen::logits_all` である。この要件に対する実行可能な確認は `transformer_reference::immutable_cache_matches_prefix` に置く。

`tests/transformer_reference.rs` の `fn immutable_cache_matches_prefix(`の定義を次の全体で置き換えます。

```rust file="tests/transformer_reference.rs" mode="replace" item="fn immutable_cache_matches_prefix("
fn immutable_cache_matches_prefix() {
	let c = QwenConfig::tiny();
	let weights = initialize(&c, 7).unwrap();
	let mut cache = KvState::new(&c).unwrap();
	let old = cache.keys[0].data.clone();
	let (_, outputs) = run(&c, &weights, &[1, 2], Some(&cache));
	assert_eq!(cache.keys[0].data, old);
	cache.adopt(&outputs, 2).unwrap();
	let (cached, outputs) = run(&c, &weights, &[3], Some(&cache));
	cache.adopt(&outputs, 1).unwrap();
	let (full, _) = run(&c, &weights, &[1, 2, 3], None);
	let v = c.vocab_size;
	close(&cached[0].data, &full[0].data[2 * v..3 * v], 1e-5, 1e-4);
	assert_eq!(cache.valid, 3);
	cache.reset();
	assert_eq!(cache.valid, 0);
}
```

```sh check
cargo test --offline --test transformer_reference immutable_cache_matches_prefix -- --exact
```

## 32. dump に token ID、shape、head 対応、位置、cache 有効長を出し、重みの全内容は出さない。

<!-- todo:22-032 -->


実装箇所は `transformer_reference::reference` と `nn::qwen::logits_all` である。この要件に対する実行可能な確認は `transformer_reference::immutable_cache_matches_prefix` に置く。

`examples/qwen_infer.rs` の `fn main(`の定義を次の全体で置き換えます。

```rust file="examples/qwen_infer.rs" mode="replace" item="fn main("
fn main() -> Result<()> {
	let args = std::env::args().collect::<Vec<_>>();
	let arg = |name: &str| args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone());
	let manifest = arg("--manifest").unwrap_or("fixtures/qwen25/manifest.json".into());
	let directory = arg("--assets").unwrap_or("fixtures/qwen25".into());
	let assets = model_assets::load(
		Path::new(&manifest),
		Path::new(&directory),
		args.iter().any(|a| a == "--download"),
	)?;
	let mut c = assets.config.clone();
	c.max_position_embeddings = arg("--context")
		.unwrap_or("16".into())
		.parse::<usize>()
		.map_err(|e| e.to_string())?;
	if c.max_position_embeddings > assets.config.max_position_embeddings
		|| c.max_position_embeddings == 0
	{
		return Err("requested context capacity".into());
	}
	let count = arg("--max-new-tokens")
		.unwrap_or("1".into())
		.parse::<usize>()
		.map_err(|e| e.to_string())?;
	let text = arg("--text").unwrap_or("Hello".into());
	let mut input = model_assets::encode(&assets.tokenizer, &text)?;
	if input
		.len()
		.checked_add(count)
		.ok_or("token capacity overflow")?
		> c.max_position_embeddings
	{
		return Err("prompt plus generation exceeds context before compilation".into());
	}
	let registry = parameter_registry(&c)?;
	let weight_bindings = registry.bind(&assets.state)?;
	let slot = weight_bindings.len();
	let mut cache = KvState::new(&c)?;
	let mut generated = Vec::new();
	for _ in 0..count {
		let mut graph = Graph::default();
		let params = register_graph(&mut graph, &registry)?;
		let tokens = graph.input(slot, vec![input.len()])?;
		let mut bindings = weight_bindings.clone();
		bindings.push(token_buffer(&input, &c)?);
		let mut cache_ids = Vec::new();
		for (layer, (key, value)) in cache.keys.iter().zip(&cache.values).enumerate() {
			cache_ids.push((
				graph.input(slot + 1 + 2 * layer, key.shape.clone())?,
				graph.input(slot + 2 + 2 * layer, value.shape.clone())?,
			));
			bindings.extend([key.clone(), value.clone()]);
		}
		let forward = logits_all(
			&mut graph,
			&c,
			&params,
			tokens,
			cache.valid,
			Some(&cache_ids),
		)?;
		let mut outputs = vec![forward.logits];
		for (k, v) in forward.cache_outputs {
			outputs.extend([k, v]);
		}
		let plan = graph.compile(&outputs)?;
		let result = plan.run(&bindings)?;
		let next = greedy(&result[0])?;
		if args.iter().any(|a| a == "--dump") {
			eprintln!(
				"tokens={input:?} logits={:?} layers={} heads={}/{} position={} cache_valid={} graph_nodes={}",
				result[0].shape,
				c.num_hidden_layers,
				c.num_attention_heads,
				c.num_key_value_heads,
				cache.valid,
				cache.valid,
				graph.nodes.len()
			);
		}
		cache.adopt(&result[1..], input.len())?;
		generated.push(next as u32);
		if next as u32 == assets.eos {
			break;
		}
		input = vec![next as u32];
	}
	let output = assets
		.tokenizer
		.decode(&generated, false)
		.map_err(|e| e.to_string())?;
	println!("ids={generated:?}\n{output}");
	Ok(())
}
```

```sh check
cargo test --offline --test transformer_reference immutable_cache_matches_prefix -- --exact
```

## 33. Metal と OpenCL は CPU fixture と比較し、未実行 backend は未確認と記録する。

<!-- todo:22-033 -->


実装箇所は `transformer_reference::reference` と `nn::qwen::logits_all` である。この要件に対する実行可能な確認は `transformer_reference::nonzero_layer_boundaries_reference` に置く。

`tests/transformer_reference.rs` の `fn nonzero_layer_boundaries_reference(`の定義を次の全体で置き換えます。

```rust file="tests/transformer_reference.rs" mode="replace" item="fn nonzero_layer_boundaries_reference("
fn nonzero_layer_boundaries_reference() {
	let c = QwenConfig::tiny();
	let weights = initialize(&c, 11).unwrap();
	let (actual, _) = run(&c, &weights, &[1, 2, 3], None);
	let reference = reference(&c, &weights, &[1, 2, 3]);
	if let Ok(path) = std::env::var("CHIBI_REFERENCE_OUT") {
		std::fs::write(path, serde_json::to_vec_pretty(&reference).unwrap()).unwrap();
	}
	let saved: Vec<Vec<f32>> =
		serde_json::from_slice(include_bytes!("../fixtures/tiny-transformer.json")).unwrap();
	if std::env::var_os("CHIBI_REFERENCE_OUT").is_none() {
		assert_eq!(reference, saved);
	}

	for (a, b) in actual.iter().zip(reference) {
		close(&a.data, &b, 1e-5, 1e-4);
	}
}
```

```sh check
cargo test --offline --test transformer_reference nonzero_layer_boundaries_reference -- --exact
```
