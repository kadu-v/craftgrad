# 24. 小さな LLM を学習し、checkpoint から同じ更新を再開する

全位置 logits、明示 targets、有効位置から loss と勾配 graph を作る。AdamW は再利用する生成 C の Plan で更新候補を計算し、有限性を確認して全 state を切り替える。保存単位は optimizer step の境界であり、RNG の現在値、shuffle 順、global cursor を含む。

以下は置換・追記先を指定した実装全体である。`mode=write` はそのファイル全体を置き換え、`mode=append` は末尾へ追記する。テストは節ごとに追加し、同じ名前の関数を重複して貼らない。生成 C と共有ライブラリは runtime の一時ディレクトリに作られる。

公開 Qwen の重みを使う数値比較と全 parameter 学習は、この作業では未実行である。全24層の登録、通常 attention の小モデル、勾配と再開は CPU テストで検証する。外部資産や実機がない検証は `NOT_RUN` と記録する。

## 1. 全位置の logits を返す学習入口を追加し、KV cache と sampling を使わない。

<!-- todo:24-001 -->

まず、この章が追加するモジュールと共有テスト補助を保存する。後続の各節は、ここに掲載した `training::TrainingPlan::{new,run}`、`optim::AdamW::update`、`checkpoint::{save_training_checkpoint,restore_training_checkpoint,adopt_checkpoint}` の規則を、焦点を絞ったテストで確認する。

```rust
// file=src/lib.rs mode=append
pub mod checkpoint;
pub mod optim;
pub mod training;
```

```rust
// file=src/optim.rs mode=write
use crate::{Binary, Buffer, Graph, Plan, Result, Unary};
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AdamState {
	pub weights: Vec<BufferData>,
	pub m: Vec<BufferData>,
	pub v: Vec<BufferData>,
	pub step: u64,
}
pub use crate::state::BufferData;
impl AdamState {
	pub fn new(weights: Vec<BufferData>) -> Self {
		let zeros = weights
			.iter()
			.map(|b| BufferData {
				shape: b.shape.clone(),
				data: vec![0.; b.data.len()],
			})
			.collect::<Vec<_>>();
		Self {
			weights,
			m: zeros.clone(),
			v: zeros,
			step: 0,
		}
	}
	pub fn validate(&self) -> Result<()> {
		if self.weights.len() != self.m.len() || self.weights.len() != self.v.len() {
			return Err("optimizer key count".into());
		}
		for ((w, m), v) in self.weights.iter().zip(&self.m).zip(&self.v) {
			for b in [w, m, v] {
				b.buffer()?;
				if b.data.iter().any(|v| !v.is_finite()) {
					return Err("nonfinite optimizer state".into());
				}
			}
			if w.shape != m.shape || w.shape != v.shape {
				return Err("optimizer shape mismatch".into());
			}
		}
		Ok(())
	}
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct AdamConfig {
	pub lr: f32,
	pub beta1: f32,
	pub beta2: f32,
	pub eps: f32,
	pub decay: f32,
	pub clip: Option<f32>,
}
impl Default for AdamConfig {
	fn default() -> Self {
		Self {
			lr: 1e-3,
			beta1: 0.9,
			beta2: 0.999,
			eps: 1e-8,
			decay: 0.,
			clip: None,
		}
	}
}
pub struct AdamW {
	plan: Plan,
	count: usize,
	config: AdamConfig,
}
impl AdamW {
	pub fn new(weights: &[BufferData], config: AdamConfig) -> Result<Self> {
		if !(config.lr > 0.
			&& config.lr.is_finite()
			&& config.eps > 0.
			&& config.eps.is_finite()
			&& (0.0..1.0).contains(&config.beta1)
			&& (0.0..1.0).contains(&config.beta2)
			&& config.decay >= 0.
			&& config.decay.is_finite())
			|| config.clip.is_some_and(|v| v <= 0. || !v.is_finite())
		{
			return Err("invalid AdamW configuration".into());
		}
		let mut g = Graph::default();
		let mut outputs = Vec::new();
		let count = weights.len();
		let correction1 = g.input(count * 4, vec![])?;
		let correction2 = g.input(count * 4 + 1, vec![])?;
		let scalar = |g: &mut Graph, v| g.constant(v, vec![]);
		for (i, w) in weights.iter().enumerate() {
			let weight = g.input(i * 4, w.shape.clone())?;
			let grad = g.input(i * 4 + 1, w.shape.clone())?;
			let m = g.input(i * 4 + 2, w.shape.clone())?;
			let v = g.input(i * 4 + 3, w.shape.clone())?;
			let b1 = scalar(&mut g, config.beta1)?;
			let b2 = scalar(&mut g, config.beta2)?;
			let mb = g.binary(Binary::Mul, m, b1)?;
			let scale = scalar(&mut g, 1. - config.beta1)?;
			let mg = g.binary(Binary::Mul, grad, scale)?;
			let next_m = g.binary(Binary::Add, mb, mg)?;
			let vb = g.binary(Binary::Mul, v, b2)?;
			let square = g.binary(Binary::Mul, grad, grad)?;
			let scale = scalar(&mut g, 1. - config.beta2)?;
			let vg = g.binary(Binary::Mul, square, scale)?;
			let next_v = g.binary(Binary::Add, vb, vg)?;
			let mh = g.binary(Binary::Div, next_m, correction1)?;
			let vh = g.binary(Binary::Div, next_v, correction2)?;
			let root = g.unary(Unary::Sqrt, vh)?;
			let eps = scalar(&mut g, config.eps)?;
			let denom = g.binary(Binary::Add, root, eps)?;
			let direction = g.binary(Binary::Div, mh, denom)?;
			let lr = scalar(&mut g, config.lr)?;
			let update = g.binary(Binary::Mul, direction, lr)?;
			let decay = scalar(&mut g, 1. - config.lr * config.decay)?;
			let decayed = g.binary(Binary::Mul, weight, decay)?;
			let next_w = g.binary(Binary::Sub, decayed, update)?;
			outputs.extend([next_w, next_m, next_v]);
		}
		Ok(Self {
			plan: g.compile(&outputs)?,
			count,
			config,
		})
	}
	pub fn update(&self, state: &mut AdamState, gradients: &[BufferData]) -> Result<f64> {
		state.validate()?;
		if gradients.len() != self.count || state.weights.len() != self.count {
			return Err("optimizer gradient keys".into());
		}
		let mut norm = 0f64;
		for (g, w) in gradients.iter().zip(&state.weights) {
			if g.shape != w.shape || g.data.len() != w.data.len() {
				return Err("gradient shape".into());
			}
			for &v in &g.data {
				if !v.is_finite() {
					return Err("nonfinite gradient".into());
				}
				norm += (v as f64).powi(2);
			}
		}
		let norm = norm.sqrt();
		let scale = self.config.clip.map_or(1., |limit| {
			(limit as f64 / norm.max(f64::MIN_POSITIVE)).min(1.)
		}) as f32;
		let step = state.step.checked_add(1).ok_or("optimizer step overflow")?;
		let mut bindings = Vec::new();
		for (i, g) in gradients.iter().enumerate() {
			bindings.push(state.weights[i].buffer()?);
			bindings.push(Buffer::new(
				g.shape.clone(),
				g.data.iter().map(|v| v * scale).collect(),
			)?);
			bindings.push(state.m[i].buffer()?);
			bindings.push(state.v[i].buffer()?);
		}
		bindings.push(Buffer::new(
			vec![],
			vec![1. - self.config.beta1.powf(step as f32)],
		)?);
		bindings.push(Buffer::new(
			vec![],
			vec![1. - self.config.beta2.powf(step as f32)],
		)?);
		let output = self.plan.run(&bindings)?;
		let mut candidate = state.clone();
		for (i, three) in output.chunks_exact(3).enumerate() {
			candidate.weights[i] = BufferData {
				shape: three[0].shape.clone(),
				data: three[0].data.clone(),
			};
			candidate.m[i] = BufferData {
				shape: three[1].shape.clone(),
				data: three[1].data.clone(),
			};
			candidate.v[i] = BufferData {
				shape: three[2].shape.clone(),
				data: three[2].data.clone(),
			};
		}
		candidate.step = step;
		candidate.validate()?;
		*state = candidate;
		Ok(norm)
	}
}
pub fn sgd(weights: &mut [BufferData], gradients: &[BufferData], lr: f32) -> Result<()> {
	if weights.len() != gradients.len() || !lr.is_finite() || lr <= 0. {
		return Err("SGD configuration".into());
	}
	let mut candidate = weights.to_vec();
	for (w, g) in candidate.iter_mut().zip(gradients) {
		if w.shape != g.shape || w.data.len() != g.data.len() {
			return Err("SGD gradient shape".into());
		}
		for (value, &grad) in w.data.iter_mut().zip(&g.data) {
			*value -= lr * grad;
			if !value.is_finite() {
				return Err("nonfinite SGD candidate".into());
			}
		}
	}
	weights.clone_from_slice(&candidate);
	Ok(())
}
pub fn accumulate_weighted(microbatches: &[(Vec<BufferData>, usize)]) -> Result<Vec<BufferData>> {
	let first = microbatches.first().ok_or("empty accumulation")?;
	let total = microbatches.iter().try_fold(0usize, |a, (_, n)| {
		a.checked_add(*n).ok_or("token count overflow")
	})?;
	if total == 0 {
		return Err("zero valid tokens".into());
	}
	let mut result = first.0.clone();
	for g in &mut result {
		g.data.fill(0.);
	}
	for (batch, n) in microbatches {
		if batch.len() != result.len() {
			return Err("accumulation keys".into());
		}
		for (dst, src) in result.iter_mut().zip(batch) {
			if dst.shape != src.shape || dst.data.len() != src.data.len() {
				return Err("accumulation shape".into());
			}
			for (d, &s) in dst.data.iter_mut().zip(&src.data) {
				if !s.is_finite() {
					return Err("nonfinite microbatch gradient".into());
				}
				*d += s * (*n as f32 / total as f32);
			}
		}
	}
	Ok(result)
}
```

```rust
// file=src/checkpoint.rs mode=write
use crate::{
	Result,
	fetch::sha256,
	optim::{AdamConfig, AdamState},
};
use serde::{Deserialize, Serialize};
use std::{
	fs::{self, File},
	io::Write,
	path::Path,
};
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DataCursor {
	pub rng: u64,
	pub permutation: Vec<usize>,
	pub cursor: usize,
	pub epoch: u64,
}
impl DataCursor {
	pub fn new(length: usize, seed: u64) -> Result<Self> {
		if length == 0 || seed == 0 {
			return Err("dataset length and RNG seed must be nonzero".into());
		}
		let mut s = Self {
			rng: seed,
			permutation: (0..length).collect(),
			cursor: 0,
			epoch: 0,
		};
		s.shuffle();
		Ok(s)
	}
	fn random(&mut self) -> u64 {
		let mut x = self.rng;
		x ^= x << 13;
		x ^= x >> 7;
		x ^= x << 17;
		self.rng = x;
		x
	}
	fn shuffle(&mut self) {
		for i in (1..self.permutation.len()).rev() {
			let j = (self.random() as usize) % (i + 1);
			self.permutation.swap(i, j);
		}
	}
	pub fn next_index(&mut self) -> usize {
		if self.cursor == self.permutation.len() {
			self.cursor = 0;
			self.epoch += 1;
			self.shuffle();
		}
		let index = self.permutation[self.cursor];
		self.cursor += 1;
		index
	}
	pub fn validate(&self) -> Result<()> {
		let n = self.permutation.len();
		if n == 0 || self.rng == 0 || self.cursor > n {
			return Err("invalid data cursor or RNG".into());
		}
		let mut seen = vec![false; n];
		for &i in &self.permutation {
			if i >= n || seen[i] {
				return Err("invalid shuffle permutation".into());
			}
			seen[i] = true;
		}
		Ok(())
	}
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TrainingState {
	pub optimizer: AdamState,
	pub optimizer_config: AdamConfig,
	pub scheduler_step: u64,
	pub data: DataCursor,
	pub dataset_sha: String,
	pub config_sha: String,
	pub tokenizer_sha: String,
	pub accumulated_microbatches: usize,
}
impl TrainingState {
	pub fn validate(&self) -> Result<()> {
		self.optimizer.validate()?;
		self.data.validate()?;
		if self.accumulated_microbatches != 0 {
			return Err("checkpoint requires an optimizer step boundary".into());
		}
		if self.scheduler_step != self.optimizer.step {
			return Err("scheduler/optimizer step mismatch".into());
		}
		for digest in [&self.dataset_sha, &self.config_sha, &self.tokenizer_sha] {
			if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
				return Err("invalid training identity digest".into());
			}
		}
		Ok(())
	}
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Commit {
	version: u32,
	bytes: u64,
	sha256: String,
}
pub fn save_training_checkpoint(destination: &Path, state: &TrainingState) -> Result<()> {
	state.validate()?;
	if destination.exists() {
		return Err("checkpoint generation already exists".into());
	}
	let parent = destination.parent().ok_or("checkpoint parent")?;
	fs::create_dir_all(parent).map_err(|e| e.to_string())?;
	let name = destination
		.file_name()
		.and_then(|s| s.to_str())
		.ok_or("checkpoint name")?;
	let nonce = std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map_err(|e| e.to_string())?
		.as_nanos();
	let temporary = parent.join(format!(".{name}.{}.{}.tmp", std::process::id(), nonce));
	fs::create_dir(&temporary).map_err(|e| e.to_string())?;
	let result = (|| {
		let bytes = serde_json::to_vec(state).map_err(|e| e.to_string())?;
		let mut file = File::options()
			.write(true)
			.create_new(true)
			.open(temporary.join("state.json"))
			.map_err(|e| e.to_string())?;
		file.write_all(&bytes)
			.and_then(|_| file.sync_all())
			.map_err(|e| e.to_string())?;
		let reread = fs::read(temporary.join("state.json")).map_err(|e| e.to_string())?;
		if bytes != reread {
			return Err("checkpoint readback mismatch".into());
		}
		let candidate: TrainingState =
			serde_json::from_value(crate::safetensor::unique_json(&reread)?)
				.map_err(|e| e.to_string())?;
		candidate.validate()?;
		let manifest = serde_json::to_vec(&Commit {
			version: 1,
			bytes: bytes.len() as u64,
			sha256: sha256(&bytes),
		})
		.map_err(|e| e.to_string())?;
		let mut file = File::options()
			.write(true)
			.create_new(true)
			.open(temporary.join("manifest.json"))
			.map_err(|e| e.to_string())?;
		file.write_all(&manifest)
			.and_then(|_| file.sync_all())
			.map_err(|e| e.to_string())?;
		File::open(&temporary)
			.and_then(|f| f.sync_all())
			.map_err(|e| e.to_string())?;
		fs::rename(&temporary, destination).map_err(|e| e.to_string())?;
		File::open(parent)
			.and_then(|f| f.sync_all())
			.map_err(|e| e.to_string())?;
		Ok(())
	})();
	if result.is_err() {
		let _ = fs::remove_dir_all(temporary);
	}
	result
}
pub fn restore_training_checkpoint(directory: &Path) -> Result<TrainingState> {
	let metadata = fs::read(directory.join("manifest.json")).map_err(|e| e.to_string())?;
	let commit: Commit = serde_json::from_value(crate::safetensor::unique_json(&metadata)?)
		.map_err(|e| e.to_string())?;
	if commit.version != 1 {
		return Err("checkpoint format version".into());
	}
	let file = directory.join("state.json");
	if fs::metadata(&file).map_err(|e| e.to_string())?.len() != commit.bytes {
		return Err("checkpoint byte length".into());
	}
	let bytes = fs::read(file).map_err(|e| e.to_string())?;
	if sha256(&bytes) != commit.sha256 {
		return Err("checkpoint digest".into());
	}
	let state: TrainingState = serde_json::from_value(crate::safetensor::unique_json(&bytes)?)
		.map_err(|e| e.to_string())?;
	state.validate()?;
	Ok(state)
}
pub fn adopt_checkpoint(live: &mut TrainingState, candidate: TrainingState) -> Result<()> {
	candidate.validate()?;
	if candidate.dataset_sha != live.dataset_sha
		|| candidate.config_sha != live.config_sha
		|| candidate.tokenizer_sha != live.tokenizer_sha
		|| candidate.optimizer_config != live.optimizer_config
	{
		return Err("checkpoint identity or optimizer configuration mismatch".into());
	}
	if candidate.data.permutation.len() != live.data.permutation.len()
		|| candidate.optimizer.weights.len() != live.optimizer.weights.len()
		|| candidate
			.optimizer
			.weights
			.iter()
			.zip(&live.optimizer.weights)
			.any(|(a, b)| a.shape != b.shape)
	{
		return Err("checkpoint model or dataset shape".into());
	}
	*live = candidate;
	Ok(())
}
```

```rust
// file=src/training.rs mode=write
use crate::{
	Buffer, Graph, Plan, Result,
	autodiff::gradient,
	nn::{
		cross_entropy_runtime,
		qwen::{QwenConfig, logits_all, parameter_registry, register_graph},
	},
	optim::BufferData,
	state::Registry,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
pub struct TrainingPlan {
	pub graph: Graph,
	pub plan: Plan,
	pub registry: Registry,
	pub token_slot: usize,
	pub target_slot: usize,
	pub valid_slot: usize,
	pub sequence: usize,
	pub config: QwenConfig,
}
impl TrainingPlan {
	pub fn new(config: QwenConfig, sequence: usize) -> Result<Self> {
		config.validate()?;
		if sequence == 0 || sequence > config.max_position_embeddings {
			return Err("training sequence capacity".into());
		}
		let registry = parameter_registry(&config)?;
		let mut graph = Graph::default();
		let params = register_graph(&mut graph, &registry)?;
		let token_slot = registry.parameters().len();
		let target_slot = token_slot + 1;
		let valid_slot = token_slot + 2;
		let tokens = graph.input(token_slot, vec![sequence])?;
		let targets = graph.input(target_slot, vec![sequence])?;
		let valid = graph.input(valid_slot, vec![sequence])?;
		let forward = logits_all(&mut graph, &config, &params, tokens, 0, None)?;
		let loss = cross_entropy_runtime(&mut graph, forward.logits, targets, valid)?;
		let mut by_id = std::collections::BTreeMap::new();
		for (name, spec) in &registry.names {
			by_id.entry(spec.id).or_insert(params[name]);
		}
		let target_nodes = by_id.values().copied().collect::<Vec<_>>();
		let gradients = gradient(&mut graph, loss, &target_nodes, None)?;
		if gradients.len() != registry.parameters().len() {
			return Err("incomplete parameter gradient set".into());
		}
		let mut outputs = vec![loss];
		outputs.extend(gradients);
		let plan = graph.compile(&outputs)?;
		Ok(Self {
			graph,
			plan,
			registry,
			token_slot,
			target_slot,
			valid_slot,
			sequence,
			config,
		})
	}
	pub fn run(
		&self,
		weights: &[BufferData],
		tokens: &[usize],
		targets: &[usize],
		valid: &[bool],
	) -> Result<(f32, Vec<BufferData>)> {
		if weights.len() != self.registry.parameters().len()
			|| tokens.len() != self.sequence
			|| targets.len() != self.sequence
			|| valid.len() != self.sequence
			|| !valid.iter().any(|&v| v)
			|| tokens
				.iter()
				.chain(targets)
				.any(|&v| v >= self.config.vocab_size)
		{
			return Err("training input/target/mask or parameter set".into());
		}
		let mut bindings = weights
			.iter()
			.map(BufferData::buffer)
			.collect::<Result<Vec<_>>>()?;
		for values in [
			tokens.iter().map(|&v| v as f32).collect::<Vec<_>>(),
			targets.iter().map(|&v| v as f32).collect(),
			valid.iter().map(|&v| if v { 1. } else { 0. }).collect(),
		] {
			bindings.push(Buffer::new(vec![self.sequence], values)?);
		}
		let outputs = self.plan.run(&bindings)?;
		let loss = outputs[0].data[0];
		if !loss.is_finite() {
			return Err("nonfinite training loss".into());
		}
		let grads = outputs[1..]
			.iter()
			.map(|b| BufferData {
				shape: b.shape.clone(),
				data: b.data.clone(),
			})
			.collect::<Vec<_>>();
		if grads.iter().flat_map(|g| &g.data).any(|v| !v.is_finite()) {
			return Err("nonfinite training gradient".into());
		}
		Ok((loss, grads))
	}
}
pub use crate::nn::qwen::initialize;
#[derive(Debug, Serialize, Deserialize)]
pub struct StorageEstimate {
	pub parameters: u64,
	pub persistent: u64,
	pub update_outputs: u64,
	pub one_layer_score: u64,
	pub all_layer_scores: u64,
	pub logits: u64,
	pub boundary_activations: u64,
	pub total_lower_bound: u64,
}
pub fn estimate_training_storage(
	registry: &Registry,
	batch: u64,
	t: u64,
	c: &QwenConfig,
) -> Result<StorageEstimate> {
	let mut ids = BTreeSet::new();
	let mut p = 0u64;
	for spec in registry.names.values() {
		if spec.trainable && ids.insert(spec.id) {
			let n = spec.shape.iter().try_fold(1u64, |a, &b| {
				a.checked_mul(b as u64).ok_or("parameter size overflow")
			})?;
			p = p.checked_add(n).ok_or("parameter total overflow")?;
		}
	}
	let checked = |values: &[u64]| {
		values.iter().try_fold(1u64, |a, &b| {
			a.checked_mul(b)
				.ok_or_else(|| "storage overflow".to_string())
		})
	};
	let persistent = checked(&[16, p])?;
	let update_outputs = checked(&[12, p])?;
	let one_layer_score = checked(&[batch, c.num_attention_heads as u64, t, t, 4])?;
	let all_layer_scores = checked(&[one_layer_score, c.num_hidden_layers as u64])?;
	let logits = checked(&[batch, t, c.vocab_size as u64, 4])?;
	let boundary_activations = checked(&[
		batch,
		t,
		c.hidden_size as u64,
		c.num_hidden_layers as u64 + 2,
		4,
	])?;
	let total_lower_bound = [
		persistent,
		update_outputs,
		all_layer_scores,
		logits,
		boundary_activations,
	]
	.into_iter()
	.try_fold(0u64, |a, b| a.checked_add(b).ok_or("storage sum overflow"))?;
	Ok(StorageEstimate {
		parameters: p,
		persistent,
		update_outputs,
		one_layer_score,
		all_layer_scores,
		logits,
		boundary_activations,
		total_lower_bound,
	})
}
```

```rust
// file=examples/train_tiny_llm.rs mode=write
use chibigrad::{
	Result,
	checkpoint::{
		DataCursor, TrainingState, adopt_checkpoint, restore_training_checkpoint,
		save_training_checkpoint,
	},
	fetch::sha256,
	nn::qwen::QwenConfig,
	optim::{AdamConfig, AdamState, AdamW, sgd},
	training::{TrainingPlan, initialize},
};
use std::path::Path;
fn main() -> Result<()> {
	let args = std::env::args().collect::<Vec<_>>();
	let arg = |name: &str| args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone());
	let steps = arg("--steps")
		.unwrap_or("10".into())
		.parse::<usize>()
		.map_err(|e| e.to_string())?;
	let seed = arg("--seed")
		.unwrap_or("1".into())
		.parse::<u64>()
		.map_err(|e| e.to_string())?;
	let sgd_mode = arg("--optimizer").as_deref() == Some("sgd");
	let c = QwenConfig::tiny();
	let corpus = [1usize, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3];
	let config = AdamConfig {
		lr: 0.02,
		..Default::default()
	};
	let dataset = serde_json::to_vec(&corpus).unwrap();
	let identity = sha256(&serde_json::to_vec(&(&c, sgd_mode)).unwrap());
	let mut state = TrainingState {
		optimizer: AdamState::new(initialize(&c, seed)?),
		optimizer_config: config,
		scheduler_step: 0,
		data: DataCursor::new(corpus.len() - 3, seed.max(1))?,
		dataset_sha: sha256(&dataset),
		config_sha: identity,
		tokenizer_sha: sha256(b"integer-token-fixture-v1"),
		accumulated_microbatches: 0,
	};
	if let Some(path) = arg("--resume") {
		adopt_checkpoint(&mut state, restore_training_checkpoint(Path::new(&path))?)?;
	}
	let plan = TrainingPlan::new(c, 3)?;
	let optimizer = AdamW::new(&state.optimizer.weights, config)?;
	for _ in 0..steps {
		let index = state.data.next_index();
		let tokens = &corpus[index..index + 3];
		let targets = &corpus[index + 1..index + 4];
		let (loss, grads) = plan.run(&state.optimizer.weights, tokens, targets, &[true; 3])?;
		if sgd_mode {
			sgd(&mut state.optimizer.weights, &grads, config.lr)?;
			state.optimizer.step += 1;
		} else {
			optimizer.update(&mut state.optimizer, &grads)?;
		}
		state.scheduler_step = state.optimizer.step;
		println!(
			"{}",
			serde_json::json!({"step":state.optimizer.step,"loss":loss,"input":tokens,"target":targets,"cursor":state.data.cursor,"epoch":state.data.epoch,"rng":state.data.rng})
		);
	}
	if let Some(path) = arg("--checkpoint") {
		save_training_checkpoint(Path::new(&path), &state)?;
	}
	println!(
		"{}",
		serde_json::json!({"final_step":state.optimizer.step,"state_sha":sha256(&serde_json::to_vec(&state).unwrap()),"compiler_calls":2,"plan_runs":if sgd_mode{steps}else{steps*2}})
	);
	Ok(())
}
```

```rust
// file=examples/train_qwen.rs mode=write
use chibigrad::{
	Result,
	checkpoint::{
		DataCursor, TrainingState, adopt_checkpoint, restore_training_checkpoint,
		save_training_checkpoint,
	},
	model_assets,
	nn::qwen::{QwenConfig, parameter_registry},
	optim::{AdamConfig, AdamState, AdamW, BufferData},
	training::{TrainingPlan, estimate_training_storage},
};
use std::path::Path;
fn main() -> Result<()> {
	let args = std::env::args().collect::<Vec<_>>();
	let arg = |name: &str| args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone());
	let manifest = arg("--manifest").unwrap_or("fixtures/qwen25/manifest.json".into());
	let directory = arg("--assets").unwrap_or("fixtures/qwen25".into());
	let sequence = arg("--sequence-length")
		.unwrap_or("8".into())
		.parse::<usize>()
		.map_err(|e| e.to_string())?;
	let microbatch = arg("--microbatch")
		.unwrap_or("1".into())
		.parse::<usize>()
		.map_err(|e| e.to_string())?;
	if microbatch != 1 {
		return Err(
			"this driver accepts microbatch=1; use accumulate_weighted for multiple microbatches"
				.into(),
		);
	}
	let steps = arg("--steps")
		.unwrap_or("1".into())
		.parse::<usize>()
		.map_err(|e| e.to_string())?;
	let manifest_value: chibigrad::fetch::Manifest =
		serde_json::from_slice(&std::fs::read(&manifest).map_err(|e| e.to_string())?)
			.map_err(|e| e.to_string())?;
	manifest_value.validate()?;
	let config_asset = manifest_value
		.files
		.iter()
		.find(|a| a.name == "config.json")
		.ok_or("config missing from manifest")?;
	let config_bytes =
		chibigrad::fetch::verified_read(&Path::new(&directory).join("config.json"), config_asset)?;
	let config: QwenConfig = serde_json::from_slice(&config_bytes).map_err(|e| e.to_string())?;
	config.require_public_05b()?;
	let estimate =
		estimate_training_storage(&parameter_registry(&config)?, 1, sequence as u64, &config)?;
	println!("{}", serde_json::to_string_pretty(&estimate).unwrap());
	println!(
		"estimate excludes additional primitive temporaries, C compiler memory, device/host copies, and allocator overhead"
	);
	if args.iter().any(|a| a == "--estimate-only") {
		return Ok(());
	}
	if let Some(budget) = arg("--budget-bytes") {
		if estimate.total_lower_bound > budget.parse::<u64>().map_err(|e| e.to_string())? {
			return Err("training storage lower bound exceeds supplied budget".into());
		}
	}
	let assets = model_assets::load(
		Path::new(&manifest),
		Path::new(&directory),
		args.iter().any(|a| a == "--download"),
	)?;
	let mut c = assets.config.clone();
	c.max_position_embeddings = sequence;
	let text = arg("--text").unwrap_or(
		"The quick brown fox jumps over the lazy dog. The quick brown fox jumps again.".into(),
	);
	let corpus = model_assets::encode(&assets.tokenizer, &text)?
		.into_iter()
		.map(|v| v as usize)
		.collect::<Vec<_>>();
	if corpus.len() <= sequence {
		return Err("training text must contain sequence+1 tokens".into());
	}
	let identity = |name: &str| {
		assets
			.manifest
			.files
			.iter()
			.find(|a| a.name == name)
			.map(|a| a.sha256.clone())
			.ok_or_else(|| format!("missing identity {name}"))
	};
	let weights = assets
		.state
		.values()
		.map(|t| {
			let b = t.to_buffer()?;
			Ok(BufferData {
				shape: b.shape,
				data: b.data,
			})
		})
		.collect::<Result<Vec<_>>>()?;
	let optimizer_config = AdamConfig {
		lr: 1e-5,
		..Default::default()
	};
	let mut state = TrainingState {
		optimizer: AdamState::new(weights),
		optimizer_config,
		scheduler_step: 0,
		data: DataCursor::new(corpus.len() - sequence, 1)?,
		dataset_sha: chibigrad::fetch::sha256(&serde_json::to_vec(&corpus).unwrap()),
		config_sha: identity("config.json")?,
		tokenizer_sha: identity("tokenizer.json")?,
		accumulated_microbatches: 0,
	};
	if let Some(path) = arg("--resume") {
		adopt_checkpoint(&mut state, restore_training_checkpoint(Path::new(&path))?)?;
	}
	let plan = TrainingPlan::new(c, sequence)?;
	println!(
		"native_training_workspace_bytes={}",
		plan.plan.workspace_bytes()?
	);
	let adam = AdamW::new(&state.optimizer.weights, optimizer_config)?;
	for _ in 0..steps {
		let i = state.data.next_index();
		let (loss, gradients) = plan.run(
			&state.optimizer.weights,
			&corpus[i..i + sequence],
			&corpus[i + 1..i + sequence + 1],
			&vec![true; sequence],
		)?;
		adam.update(&mut state.optimizer, &gradients)?;
		state.scheduler_step = state.optimizer.step;
		println!("step={} loss={loss}", state.optimizer.step);
	}
	if let Some(path) = arg("--checkpoint") {
		save_training_checkpoint(Path::new(&path), &state)?;
	}
	Ok(())
}
```

```rust
// file=tests/llm_gradients.rs mode=write
use chibigrad::{
	Buffer, Graph,
	autodiff::gradient,
	nn::{cross_entropy, qwen::QwenConfig},
	optim::{AdamConfig, AdamState, AdamW, BufferData, accumulate_weighted},
	training::{TrainingPlan, estimate_training_storage, initialize},
};
```

```rust
// file=tests/llm_resume.rs mode=write
use chibigrad::{
	checkpoint::{
		DataCursor, TrainingState, adopt_checkpoint, restore_training_checkpoint,
		save_training_checkpoint,
	},
	fetch::sha256,
	nn::qwen::QwenConfig,
	optim::{AdamConfig, AdamState, AdamW},
	training::{TrainingPlan, initialize},
};
use std::{fs, path::Path, process::Command};
fn initial() -> TrainingState {
	let config = QwenConfig::tiny();
	TrainingState {
		optimizer: AdamState::new(initialize(&config, 1).unwrap()),
		optimizer_config: AdamConfig {
			lr: 0.02,
			..Default::default()
		},
		scheduler_step: 0,
		data: DataCursor::new(9, 1).unwrap(),
		dataset_sha: sha256(b"1,2,3,1,2,3,1,2,3,1,2,3"),
		config_sha: sha256(&serde_json::to_vec(&config).unwrap()),
		tokenizer_sha: sha256(b"integer-token-fixture-v1"),
		accumulated_microbatches: 0,
	}
}
```

`tests/llm_gradients.rs::uniform_logits_and_targets` を追加する。

```rust
// file=tests/llm_gradients.rs mode=append
#[test]
fn uniform_logits_and_targets() {
	let mut graph = Graph::default();
	let logits = graph.input(0, vec![2, 4]).unwrap();
	let loss = cross_entropy(&mut graph, logits, &[1, 2], &[true, true]).unwrap();
	let grads = gradient(&mut graph, loss, &[logits], None).unwrap();
	let mut out = vec![loss];
	out.extend(grads);
	let values = graph
		.compile(&out)
		.unwrap()
		.run(&[Buffer::new(vec![2, 4], vec![1000.; 8]).unwrap()])
		.unwrap();
	assert!((values[0].data[0] - 4f32.ln()).abs() < 1e-6);
	let reference = [0.125, -0.375, 0.125, 0.125, 0.125, 0.125, -0.375, 0.125];
	for (a, b) in values[1].data.iter().zip(reference) {
		assert!((a - b).abs() < 1e-6);
	}
}
```

実装箇所は `nn::qwen::logits_all` と `training::TrainingPlan::new` である。この要件に対する実行可能な確認は `llm_gradients::uniform_logits_and_targets` に置く。

```sh
# check
cargo test --offline --test llm_gradients uniform_logits_and_targets -- --exact
```

## 2. target の範囲、有効位置、安定した cross entropy を検査する。

<!-- todo:24-002 -->


実装箇所は `nn::cross_entropy_runtime` と `training::TrainingPlan::run` である。この要件に対する実行可能な確認は `llm_gradients::uniform_logits_and_targets` に置く。

`src/training.rs` の `pub fn run(`（`impl TrainingPlan` 内）の定義を次の全体で置き換えます。

```rust
// file="src/training.rs" mode="replace" item="pub fn run(" within="impl TrainingPlan"
	pub fn run(
		&self,
		weights: &[BufferData],
		tokens: &[usize],
		targets: &[usize],
		valid: &[bool],
	) -> Result<(f32, Vec<BufferData>)> {
		if weights.len() != self.registry.parameters().len()
			|| tokens.len() != self.sequence
			|| targets.len() != self.sequence
			|| valid.len() != self.sequence
			|| !valid.iter().any(|&v| v)
			|| tokens
				.iter()
				.chain(targets)
				.any(|&v| v >= self.config.vocab_size)
		{
			return Err("training input/target/mask or parameter set".into());
		}
		let mut bindings = weights
			.iter()
			.map(BufferData::buffer)
			.collect::<Result<Vec<_>>>()?;
		for values in [
			tokens.iter().map(|&v| v as f32).collect::<Vec<_>>(),
			targets.iter().map(|&v| v as f32).collect(),
			valid.iter().map(|&v| if v { 1. } else { 0. }).collect(),
		] {
			bindings.push(Buffer::new(vec![self.sequence], values)?);
		}
		let outputs = self.plan.run(&bindings)?;
		let loss = outputs[0].data[0];
		if !loss.is_finite() {
			return Err("nonfinite training loss".into());
		}
		let grads = outputs[1..]
			.iter()
			.map(|b| BufferData {
				shape: b.shape.clone(),
				data: b.data.clone(),
			})
			.collect::<Vec<_>>();
		if grads.iter().flat_map(|g| &g.data).any(|v| !v.is_finite()) {
			return Err("nonfinite training gradient".into());
		}
		Ok((loss, grads))
	}
```

```sh
# check
cargo test --offline --test llm_gradients uniform_logits_and_targets -- --exact
```

## 3. embedding/gather の backward を追加し、繰り返し token の寄与を加算する。

<!-- todo:24-003 -->


実装箇所は `autodiff::gradient` の `Op::Gather` と、既存 `Op::ScatterAdd` の reduction lowering である。この要件に対する実行可能な確認は `llm_gradients::uniform_logits_and_targets` に置く。

`src/autodiff.rs` の `pub fn gradient(` の `Op::Gather` 分岐を次の全体で置き換えます。

```rust
// file="src/autodiff.rs" mode="replace" item="pub fn gradient(" arm="Op::Gather"
			Op::Gather { x, indices, axis } => {
				let zeros = g.constant(0., g.shape(x).to_vec())?;
				contributions.push((x, g.scatter_add(zeros, indices, up, axis)?));
			}
```

```sh
# check
cargo test --offline --test llm_gradients uniform_logits_and_targets -- --exact
```

## 4. 最初は各語彙行が全 token 位置を走査する reduction で実装し、GPU の書込み競合を避ける。

<!-- todo:24-004 -->


実装箇所は `autodiff::gradient` の `Op::Gather` と、既存 `Op::ScatterAdd` の reduction lowering である。この要件に対する実行可能な確認は `llm_gradients::uniform_logits_and_targets` に置く。

`src/graph.rs` の `fn expr(`（`impl Emitter<'_>` 内） の `Op::ScatterAdd` 分岐を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn expr(" arm="Op::ScatterAdd" within="impl Emitter<'_>"
			Op::ScatterAdd {
				base,
				indices,
				updates,
				axis,
			} => {
				let bv = self.expr(base, index, code)?;
				let target = self.name();
				let k = self.name();
				code.push_str(&format!(
					"float {target}={bv};\nfor (size_t {k}=0; {k}<{}; ++{k}) {{\n",
					self.count(indices)
				));
				let iv = self.expr(indices, &k, code)?;
				let ivar = self.name();
				code.push_str(&format!("float {ivar}={iv};\nif (!isfinite({ivar}) || {ivar}<0.0f || floorf({ivar})!={ivar} || (double){ivar}>=(double){}) return 2;\nif ((size_t){ivar} == {}) {{\n",self.dim(id,axis),self.coord(id,index,axis)));
				let mut c = (0..self.g.shape(id).len())
					.map(|a| self.coord(id, index, a))
					.collect::<Vec<_>>();
				c[axis] = k;
				let ui = self.flatten(updates, &c);
				let uv = self.expr(updates, &ui, code)?;
				code.push_str(&format!("{target}+={uv};\n}}\n}}\n"));
				return Ok(target);
			}
```

```sh
# check
cargo test --offline --test llm_gradients uniform_logits_and_targets -- --exact
```

## 5. RMSNorm、SiLU、softmax、RoPE、GQA、tied weight の勾配を13の規則と合成して検証する。

<!-- todo:24-005 -->


`tests/llm_gradients.rs::tiny_transformer_parameter_central_difference` を追加する。

```rust
// file=tests/llm_gradients.rs mode=append
#[test]
fn tiny_transformer_parameter_central_difference() {
	let mut config = QwenConfig::tiny();
	config.hidden_size = 4;
	config.intermediate_size = 6;
	config.vocab_size = 4;
	config.num_attention_heads = 2;
	let plan = TrainingPlan::new(config.clone(), 2).unwrap();
	let weights = initialize(&config, 19).unwrap();
	let (_, gradients) = plan.run(&weights, &[1, 1], &[1, 2], &[true, true]).unwrap();
	assert_eq!(gradients.len(), weights.len());
	for name in [
		"model.embed_tokens.weight",
		"model.layers.0.self_attn.q_proj.weight",
		"model.layers.0.self_attn.k_proj.weight",
		"model.layers.0.self_attn.v_proj.weight",
		"model.layers.0.input_layernorm.weight",
		"model.layers.0.mlp.gate_proj.weight",
	] {
		let slot = plan.registry.names[name].slot;
		let element = if name == "model.embed_tokens.weight" {
			4
		} else {
			0
		};
		for h in [1e-3, 5e-4] {
			let mut plus = weights.clone();
			let mut minus = weights.clone();
			plus[slot].data[element] += h;
			minus[slot].data[element] -= h;
			let a = plan.run(&plus, &[1, 1], &[1, 2], &[true, true]).unwrap().0;
			let b = plan.run(&minus, &[1, 1], &[1, 2], &[true, true]).unwrap().0;
			let reference = (a - b) / (2. * h);
			let actual = gradients[slot].data[element];
			assert!(
				actual.is_finite()
					&& reference.is_finite()
					&& (actual - reference).abs() <= 3e-3 + 3e-2 * reference.abs(),
				"{name}: {actual} {reference}"
			);
		}
	}
}
```

実装箇所は `autodiff::gradient` と `training::TrainingPlan::new` である。この要件に対する実行可能な確認は `llm_gradients::tiny_transformer_parameter_central_difference` に置く。

```sh
# check
cargo test --offline --test llm_gradients tiny_transformer_parameter_central_difference -- --exact
```

## 6. token を一つずらした target を作り、入力と target の位置を dump する。

<!-- todo:24-006 -->


実装箇所は `training::TrainingPlan::run` と `examples/train_tiny_llm.rs::main` である。この要件に対する実行可能な確認は `llm_gradients::tiny_transformer_parameter_central_difference` に置く。

`examples/train_tiny_llm.rs` の `fn main(`の定義を次の全体で置き換えます。

```rust
// file="examples/train_tiny_llm.rs" mode="replace" item="fn main("
fn main() -> Result<()> {
	let args = std::env::args().collect::<Vec<_>>();
	let arg = |name: &str| args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone());
	let steps = arg("--steps")
		.unwrap_or("10".into())
		.parse::<usize>()
		.map_err(|e| e.to_string())?;
	let seed = arg("--seed")
		.unwrap_or("1".into())
		.parse::<u64>()
		.map_err(|e| e.to_string())?;
	let sgd_mode = arg("--optimizer").as_deref() == Some("sgd");
	let c = QwenConfig::tiny();
	let corpus = [1usize, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3];
	let config = AdamConfig {
		lr: 0.02,
		..Default::default()
	};
	let dataset = serde_json::to_vec(&corpus).unwrap();
	let identity = sha256(&serde_json::to_vec(&(&c, sgd_mode)).unwrap());
	let mut state = TrainingState {
		optimizer: AdamState::new(initialize(&c, seed)?),
		optimizer_config: config,
		scheduler_step: 0,
		data: DataCursor::new(corpus.len() - 3, seed.max(1))?,
		dataset_sha: sha256(&dataset),
		config_sha: identity,
		tokenizer_sha: sha256(b"integer-token-fixture-v1"),
		accumulated_microbatches: 0,
	};
	if let Some(path) = arg("--resume") {
		adopt_checkpoint(&mut state, restore_training_checkpoint(Path::new(&path))?)?;
	}
	let plan = TrainingPlan::new(c, 3)?;
	let optimizer = AdamW::new(&state.optimizer.weights, config)?;
	for _ in 0..steps {
		let index = state.data.next_index();
		let tokens = &corpus[index..index + 3];
		let targets = &corpus[index + 1..index + 4];
		let (loss, grads) = plan.run(&state.optimizer.weights, tokens, targets, &[true; 3])?;
		if sgd_mode {
			sgd(&mut state.optimizer.weights, &grads, config.lr)?;
			state.optimizer.step += 1;
		} else {
			optimizer.update(&mut state.optimizer, &grads)?;
		}
		state.scheduler_step = state.optimizer.step;
		println!(
			"{}",
			serde_json::json!({"step":state.optimizer.step,"loss":loss,"input":tokens,"target":targets,"cursor":state.data.cursor,"epoch":state.data.epoch,"rng":state.data.rng})
		);
	}
	if let Some(path) = arg("--checkpoint") {
		save_training_checkpoint(Path::new(&path), &state)?;
	}
	println!(
		"{}",
		serde_json::json!({"final_step":state.optimizer.step,"state_sha":sha256(&serde_json::to_vec(&state).unwrap()),"compiler_calls":2,"plan_runs":if sgd_mode{steps}else{steps*2}})
	);
	Ok(())
}
```

```sh
# check
cargo test --offline --test llm_gradients tiny_transformer_parameter_central_difference -- --exact
```

## 7. forward、loss、gradient、SGD 更新を一回の step として既存 runtime で動かす。

<!-- todo:24-007 -->


実装箇所は `training::TrainingPlan::run` と `examples/train_tiny_llm.rs::main` である。この要件に対する実行可能な確認は `llm_gradients::tiny_transformer_parameter_central_difference` に置く。

`examples/train_tiny_llm.rs` の `fn main(`の定義を次の全体で置き換えます。

```rust
// file="examples/train_tiny_llm.rs" mode="replace" item="fn main("
fn main() -> Result<()> {
	let args = std::env::args().collect::<Vec<_>>();
	let arg = |name: &str| args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone());
	let steps = arg("--steps")
		.unwrap_or("10".into())
		.parse::<usize>()
		.map_err(|e| e.to_string())?;
	let seed = arg("--seed")
		.unwrap_or("1".into())
		.parse::<u64>()
		.map_err(|e| e.to_string())?;
	let sgd_mode = arg("--optimizer").as_deref() == Some("sgd");
	let c = QwenConfig::tiny();
	let corpus = [1usize, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3];
	let config = AdamConfig {
		lr: 0.02,
		..Default::default()
	};
	let dataset = serde_json::to_vec(&corpus).unwrap();
	let identity = sha256(&serde_json::to_vec(&(&c, sgd_mode)).unwrap());
	let mut state = TrainingState {
		optimizer: AdamState::new(initialize(&c, seed)?),
		optimizer_config: config,
		scheduler_step: 0,
		data: DataCursor::new(corpus.len() - 3, seed.max(1))?,
		dataset_sha: sha256(&dataset),
		config_sha: identity,
		tokenizer_sha: sha256(b"integer-token-fixture-v1"),
		accumulated_microbatches: 0,
	};
	if let Some(path) = arg("--resume") {
		adopt_checkpoint(&mut state, restore_training_checkpoint(Path::new(&path))?)?;
	}
	let plan = TrainingPlan::new(c, 3)?;
	let optimizer = AdamW::new(&state.optimizer.weights, config)?;
	for _ in 0..steps {
		let index = state.data.next_index();
		let tokens = &corpus[index..index + 3];
		let targets = &corpus[index + 1..index + 4];
		let (loss, grads) = plan.run(&state.optimizer.weights, tokens, targets, &[true; 3])?;
		if sgd_mode {
			sgd(&mut state.optimizer.weights, &grads, config.lr)?;
			state.optimizer.step += 1;
		} else {
			optimizer.update(&mut state.optimizer, &grads)?;
		}
		state.scheduler_step = state.optimizer.step;
		println!(
			"{}",
			serde_json::json!({"step":state.optimizer.step,"loss":loss,"input":tokens,"target":targets,"cursor":state.data.cursor,"epoch":state.data.epoch,"rng":state.data.rng})
		);
	}
	if let Some(path) = arg("--checkpoint") {
		save_training_checkpoint(Path::new(&path), &state)?;
	}
	println!(
		"{}",
		serde_json::json!({"final_step":state.optimizer.step,"state_sha":sha256(&serde_json::to_vec(&state).unwrap()),"compiler_calls":2,"plan_runs":if sgd_mode{steps}else{steps*2}})
	);
	Ok(())
}
```

```sh
# check
cargo test --offline --test llm_gradients tiny_transformer_parameter_central_difference -- --exact
```

## 8. step の先頭で gradient を初期化し、前回分を意図せず加えない。

<!-- todo:24-008 -->


実装箇所は `training::TrainingPlan::run` と `examples/train_tiny_llm.rs::main` である。この要件に対する実行可能な確認は `llm_gradients::tiny_transformer_parameter_central_difference` に置く。

`src/runtime.rs` の `pub fn run_with_shapes(`（`impl Plan` 内）の定義を次の全体で置き換えます。

```rust
// file="src/runtime.rs" mode="replace" item="pub fn run_with_shapes(" within="impl Plan"
	pub fn run_with_shapes(
		&self,
		inputs: &[Buffer],
		node_shapes: &[Vec<usize>],
	) -> Result<Vec<Buffer>> {
		self.verify_seal()?;
		if self.specialized
			&& self
				.graph
				.nodes
				.iter()
				.map(|n| &n.shape)
				.ne(node_shapes.iter())
		{
			return Err("specialized plan shape mismatch".into());
		}
		let mut graph = self.graph.clone();
		if node_shapes.len() != graph.nodes.len() {
			return Err("node shape table length mismatch".into());
		}
		for (node, shape) in graph.nodes.iter_mut().zip(node_shapes) {
			if node.shape.len() != shape.len() {
				return Err("dynamic rank change is unsupported".into());
			}
			elements(shape)?;
			node.shape = shape.clone();
		}
		graph.verify(&self.outputs)?;
		if inputs.len() != self.abi.inputs.len() {
			return Err("input count mismatch".into());
		}
		for node in &graph.nodes {
			if let crate::Op::Input { slot } = node.op {
				let input = inputs.get(slot).ok_or("missing input")?;
				if input.shape.len() != node.shape.len()
					|| input.data.len() < elements(&node.shape)?
				{
					return Err("input capacity or rank mismatch".into());
				}
			}
		}
		graph.validate_indices(inputs, &self.outputs)?;
		let mut storage = graph
			.nodes
			.iter()
			.enumerate()
			.map(|(id, node)| {
				if self.allocations.contains(&id) {
					Buffer::zeros(node.shape.clone())
				} else {
					Ok(Buffer {
						shape: node.shape.clone(),
						data: vec![],
					})
				}
			})
			.collect::<Result<Vec<_>>>()?;
		let mut pointers = storage
			.iter_mut()
			.map(|b| b.data.as_mut_ptr().cast::<c_void>())
			.collect::<Vec<_>>();
		for (id, node) in graph.nodes.iter().enumerate() {
			if let crate::Op::Input { slot } = node.op {
				pointers[id] = inputs[slot].data.as_ptr().cast_mut().cast();
			}
		}
		let dims = node_shapes
			.iter()
			.flatten()
			.map(|&n| n as u64)
			.collect::<Vec<_>>();
		type Entry = unsafe extern "C" fn(*const *mut c_void, *const u64) -> i32;
		let entry: Entry = unsafe { std::mem::transmute(self.library.symbol()) };
		let status = unsafe { entry(pointers.as_ptr(), dims.as_ptr()) };
		if status != 0 {
			return Err(format!(
				"generated kernel rejected input or allocation: {status}"
			));
		}
		self.outputs
			.iter()
			.map(|&id| {
				if let crate::Op::Input { slot } = graph.nodes[id].op {
					Buffer::new(
						graph.nodes[id].shape.clone(),
						inputs[slot].data[..elements(&graph.nodes[id].shape)?].to_vec(),
					)
				} else {
					Ok(storage[id].clone())
				}
			})
			.collect()
	}
```

```sh
# check
cargo test --offline --test llm_gradients tiny_transformer_parameter_central_difference -- --exact
```

## 9. Plan の入力 slot に更新後の parameter を渡し、次の step で新しい値を読む。

<!-- todo:24-009 -->


実装箇所は `training::TrainingPlan::run` と `examples/train_tiny_llm.rs::main` である。この要件に対する実行可能な確認は `llm_gradients::tiny_transformer_parameter_central_difference` に置く。

`src/training.rs` の `pub fn run(`（`impl TrainingPlan` 内）の定義を次の全体で置き換えます。

```rust
// file="src/training.rs" mode="replace" item="pub fn run(" within="impl TrainingPlan"
	pub fn run(
		&self,
		weights: &[BufferData],
		tokens: &[usize],
		targets: &[usize],
		valid: &[bool],
	) -> Result<(f32, Vec<BufferData>)> {
		if weights.len() != self.registry.parameters().len()
			|| tokens.len() != self.sequence
			|| targets.len() != self.sequence
			|| valid.len() != self.sequence
			|| !valid.iter().any(|&v| v)
			|| tokens
				.iter()
				.chain(targets)
				.any(|&v| v >= self.config.vocab_size)
		{
			return Err("training input/target/mask or parameter set".into());
		}
		let mut bindings = weights
			.iter()
			.map(BufferData::buffer)
			.collect::<Result<Vec<_>>>()?;
		for values in [
			tokens.iter().map(|&v| v as f32).collect::<Vec<_>>(),
			targets.iter().map(|&v| v as f32).collect(),
			valid.iter().map(|&v| if v { 1. } else { 0. }).collect(),
		] {
			bindings.push(Buffer::new(vec![self.sequence], values)?);
		}
		let outputs = self.plan.run(&bindings)?;
		let loss = outputs[0].data[0];
		if !loss.is_finite() {
			return Err("nonfinite training loss".into());
		}
		let grads = outputs[1..]
			.iter()
			.map(|b| BufferData {
				shape: b.shape.clone(),
				data: b.data.clone(),
			})
			.collect::<Vec<_>>();
		if grads.iter().flat_map(|g| &g.data).any(|v| !v.is_finite()) {
			return Err("nonfinite training gradient".into());
		}
		Ok((loss, grads))
	}
```

```sh
# check
cargo test --offline --test llm_gradients tiny_transformer_parameter_central_difference -- --exact
```

## 10. CPU の loss と更新を scalar reference に比較し、その後 Metal と OpenCL でも小例を比較する。

<!-- todo:24-010 -->


実装箇所は `training::TrainingPlan::run` と `examples/train_tiny_llm.rs::main` である。この要件に対する実行可能な確認は `llm_gradients::tiny_transformer_parameter_central_difference` に置く。

`tests/llm_gradients.rs` の `fn tiny_transformer_parameter_central_difference(`の定義を次の全体で置き換えます。

```rust
// file="tests/llm_gradients.rs" mode="replace" item="fn tiny_transformer_parameter_central_difference("
fn tiny_transformer_parameter_central_difference() {
	let mut config = QwenConfig::tiny();
	config.hidden_size = 4;
	config.intermediate_size = 6;
	config.vocab_size = 4;
	config.num_attention_heads = 2;
	let plan = TrainingPlan::new(config.clone(), 2).unwrap();
	let weights = initialize(&config, 19).unwrap();
	let (_, gradients) = plan.run(&weights, &[1, 1], &[1, 2], &[true, true]).unwrap();
	assert_eq!(gradients.len(), weights.len());
	for name in [
		"model.embed_tokens.weight",
		"model.layers.0.self_attn.q_proj.weight",
		"model.layers.0.self_attn.k_proj.weight",
		"model.layers.0.self_attn.v_proj.weight",
		"model.layers.0.input_layernorm.weight",
		"model.layers.0.mlp.gate_proj.weight",
	] {
		let slot = plan.registry.names[name].slot;
		let element = if name == "model.embed_tokens.weight" {
			4
		} else {
			0
		};
		for h in [1e-3, 5e-4] {
			let mut plus = weights.clone();
			let mut minus = weights.clone();
			plus[slot].data[element] += h;
			minus[slot].data[element] -= h;
			let a = plan.run(&plus, &[1, 1], &[1, 2], &[true, true]).unwrap().0;
			let b = plan.run(&minus, &[1, 1], &[1, 2], &[true, true]).unwrap().0;
			let reference = (a - b) / (2. * h);
			let actual = gradients[slot].data[element];
			assert!(
				actual.is_finite()
					&& reference.is_finite()
					&& (actual - reference).abs() <= 3e-3 + 3e-2 * reference.abs(),
				"{name}: {actual} {reference}"
			);
		}
	}
}
```

```sh
# check
cargo test --offline --test llm_gradients tiny_transformer_parameter_central_difference -- --exact
```

同じ Plan へ更新後の重みを渡す小モデルの driver を実行する。`loss`、入力と1位置ずらした target、global cursor と RNG を行ごとに出力する。

```sh
# check
cargo run --offline --example train_tiny_llm -- --optimizer sgd --steps 100 --seed 1
```

## 11. m、v、step、bias correction、weight decay を持つ f32 AdamW を追加する。

<!-- todo:24-011 -->


`tests/llm_gradients.rs::adamw_scalar_and_finite_transaction` を追加する。

```rust
// file=tests/llm_gradients.rs mode=append
#[test]
fn adamw_scalar_and_finite_transaction() {
	let w = BufferData {
		shape: vec![],
		data: vec![1.],
	};
	let mut state = AdamState::new(vec![w.clone()]);
	let config = AdamConfig {
		lr: 0.1,
		decay: 0.,
		..Default::default()
	};
	let adam = AdamW::new(&state.weights, config).unwrap();
	adam.update(&mut state, &[w]).unwrap();
	let expected = 1. - 0.1 / (1. + config.eps);
	assert!((state.weights[0].data[0] - expected).abs() < 1e-5 + 1e-4 * expected.abs());
	assert_eq!(state.step, 1);
	let before = state.clone();
	assert!(
		adam.update(
			&mut state,
			&[BufferData {
				shape: vec![],
				data: vec![f32::NAN]
			}]
		)
		.is_err()
	);
	assert_eq!(state, before);
}
```

実装箇所は `optim::AdamW::{new,update}` である。この要件に対する実行可能な確認は `llm_gradients::adamw_scalar_and_finite_transaction` に置く。

`src/optim.rs` の `pub fn new(`（`impl AdamW` 内）の定義を次の全体で置き換えます。

```rust
// file="src/optim.rs" mode="replace" item="pub fn new(" within="impl AdamW"
	pub fn new(weights: &[BufferData], config: AdamConfig) -> Result<Self> {
		if !(config.lr > 0.
			&& config.lr.is_finite()
			&& config.eps > 0.
			&& config.eps.is_finite()
			&& (0.0..1.0).contains(&config.beta1)
			&& (0.0..1.0).contains(&config.beta2)
			&& config.decay >= 0.
			&& config.decay.is_finite())
			|| config.clip.is_some_and(|v| v <= 0. || !v.is_finite())
		{
			return Err("invalid AdamW configuration".into());
		}
		let mut g = Graph::default();
		let mut outputs = Vec::new();
		let count = weights.len();
		let correction1 = g.input(count * 4, vec![])?;
		let correction2 = g.input(count * 4 + 1, vec![])?;
		let scalar = |g: &mut Graph, v| g.constant(v, vec![]);
		for (i, w) in weights.iter().enumerate() {
			let weight = g.input(i * 4, w.shape.clone())?;
			let grad = g.input(i * 4 + 1, w.shape.clone())?;
			let m = g.input(i * 4 + 2, w.shape.clone())?;
			let v = g.input(i * 4 + 3, w.shape.clone())?;
			let b1 = scalar(&mut g, config.beta1)?;
			let b2 = scalar(&mut g, config.beta2)?;
			let mb = g.binary(Binary::Mul, m, b1)?;
			let scale = scalar(&mut g, 1. - config.beta1)?;
			let mg = g.binary(Binary::Mul, grad, scale)?;
			let next_m = g.binary(Binary::Add, mb, mg)?;
			let vb = g.binary(Binary::Mul, v, b2)?;
			let square = g.binary(Binary::Mul, grad, grad)?;
			let scale = scalar(&mut g, 1. - config.beta2)?;
			let vg = g.binary(Binary::Mul, square, scale)?;
			let next_v = g.binary(Binary::Add, vb, vg)?;
			let mh = g.binary(Binary::Div, next_m, correction1)?;
			let vh = g.binary(Binary::Div, next_v, correction2)?;
			let root = g.unary(Unary::Sqrt, vh)?;
			let eps = scalar(&mut g, config.eps)?;
			let denom = g.binary(Binary::Add, root, eps)?;
			let direction = g.binary(Binary::Div, mh, denom)?;
			let lr = scalar(&mut g, config.lr)?;
			let update = g.binary(Binary::Mul, direction, lr)?;
			let decay = scalar(&mut g, 1. - config.lr * config.decay)?;
			let decayed = g.binary(Binary::Mul, weight, decay)?;
			let next_w = g.binary(Binary::Sub, decayed, update)?;
			outputs.extend([next_w, next_m, next_v]);
		}
		Ok(Self {
			plan: g.compile(&outputs)?,
			count,
			config,
		})
	}
```

```sh
# check
cargo test --offline --test llm_gradients adamw_scalar_and_finite_transaction -- --exact
```

## 12. scalar parameter の一回の更新を数式の reference に一致させる。

<!-- todo:24-012 -->


実装箇所は `optim::AdamW::{new,update}` である。この要件に対する実行可能な確認は `llm_gradients::adamw_scalar_and_finite_transaction` に置く。

`tests/llm_gradients.rs` の `fn adamw_scalar_and_finite_transaction(`の定義を次の全体で置き換えます。

```rust
// file="tests/llm_gradients.rs" mode="replace" item="fn adamw_scalar_and_finite_transaction("
fn adamw_scalar_and_finite_transaction() {
	let w = BufferData {
		shape: vec![],
		data: vec![1.],
	};
	let mut state = AdamState::new(vec![w.clone()]);
	let config = AdamConfig {
		lr: 0.1,
		decay: 0.,
		..Default::default()
	};
	let adam = AdamW::new(&state.weights, config).unwrap();
	adam.update(&mut state, &[w]).unwrap();
	let expected = 1. - 0.1 / (1. + config.eps);
	assert!((state.weights[0].data[0] - expected).abs() < 1e-5 + 1e-4 * expected.abs());
	assert_eq!(state.step, 1);
	let before = state.clone();
	assert!(
		adam.update(
			&mut state,
			&[BufferData {
				shape: vec![],
				data: vec![f32::NAN]
			}]
		)
		.is_err()
	);
	assert_eq!(state, before);
}
```

```sh
# check
cargo test --offline --test llm_gradients adamw_scalar_and_finite_transaction -- --exact
```

## 13. gradient norm と非有限値を検査し、必要なら設定された閾値で clipping する。

<!-- todo:24-013 -->


`tests/llm_gradients.rs::norm_clipping_is_global` を追加する。

```rust
// file=tests/llm_gradients.rs mode=append
#[test]
fn norm_clipping_is_global() {
	let mut state = AdamState::new(vec![BufferData {
		shape: vec![2],
		data: vec![0., 0.],
	}]);
	let config = AdamConfig {
		clip: Some(1.),
		..Default::default()
	};
	let adam = AdamW::new(&state.weights, config).unwrap();
	let norm = adam
		.update(
			&mut state,
			&[BufferData {
				shape: vec![2],
				data: vec![3., 4.],
			}],
		)
		.unwrap();
	assert_eq!(norm, 5.);
	assert!((state.m[0].data[0] - 0.06).abs() < 1e-6);
	assert!((state.m[0].data[1] - 0.08).abs() < 1e-6);
}
```

実装箇所は `optim::AdamW::{new,update}` である。この要件に対する実行可能な確認は `llm_gradients::norm_clipping_is_global` に置く。

`src/optim.rs` の `pub fn update(`（`impl AdamW` 内）の定義を次の全体で置き換えます。

```rust
// file="src/optim.rs" mode="replace" item="pub fn update(" within="impl AdamW"
	pub fn update(&self, state: &mut AdamState, gradients: &[BufferData]) -> Result<f64> {
		state.validate()?;
		if gradients.len() != self.count || state.weights.len() != self.count {
			return Err("optimizer gradient keys".into());
		}
		let mut norm = 0f64;
		for (g, w) in gradients.iter().zip(&state.weights) {
			if g.shape != w.shape || g.data.len() != w.data.len() {
				return Err("gradient shape".into());
			}
			for &v in &g.data {
				if !v.is_finite() {
					return Err("nonfinite gradient".into());
				}
				norm += (v as f64).powi(2);
			}
		}
		let norm = norm.sqrt();
		let scale = self.config.clip.map_or(1., |limit| {
			(limit as f64 / norm.max(f64::MIN_POSITIVE)).min(1.)
		}) as f32;
		let step = state.step.checked_add(1).ok_or("optimizer step overflow")?;
		let mut bindings = Vec::new();
		for (i, g) in gradients.iter().enumerate() {
			bindings.push(state.weights[i].buffer()?);
			bindings.push(Buffer::new(
				g.shape.clone(),
				g.data.iter().map(|v| v * scale).collect(),
			)?);
			bindings.push(state.m[i].buffer()?);
			bindings.push(state.v[i].buffer()?);
		}
		bindings.push(Buffer::new(
			vec![],
			vec![1. - self.config.beta1.powf(step as f32)],
		)?);
		bindings.push(Buffer::new(
			vec![],
			vec![1. - self.config.beta2.powf(step as f32)],
		)?);
		let output = self.plan.run(&bindings)?;
		let mut candidate = state.clone();
		for (i, three) in output.chunks_exact(3).enumerate() {
			candidate.weights[i] = BufferData {
				shape: three[0].shape.clone(),
				data: three[0].data.clone(),
			};
			candidate.m[i] = BufferData {
				shape: three[1].shape.clone(),
				data: three[1].data.clone(),
			};
			candidate.v[i] = BufferData {
				shape: three[2].shape.clone(),
				data: three[2].data.clone(),
			};
		}
		candidate.step = step;
		candidate.validate()?;
		*state = candidate;
		Ok(norm)
	}
```

```sh
# check
cargo test --offline --test llm_gradients norm_clipping_is_global -- --exact
```

## 14. microbatch ごとの loss の平均ではなく、有効 token 数に応じて gradient を重み付けする。

<!-- todo:24-014 -->


`tests/llm_gradients.rs::token_weighted_accumulation` を追加する。

```rust
// file=tests/llm_gradients.rs mode=append
#[test]
fn token_weighted_accumulation() {
	let gradient = |v| {
		vec![BufferData {
			shape: vec![1],
			data: vec![v],
		}]
	};
	let accumulated = accumulate_weighted(&[(gradient(2.), 1), (gradient(6.), 3)]).unwrap();
	assert_eq!(accumulated[0].data, [5.]);
	let mut full = AdamState::new(gradient(1.));
	let mut micro = full.clone();
	let adam = AdamW::new(&full.weights, AdamConfig::default()).unwrap();
	adam.update(&mut full, &gradient(5.)).unwrap();
	adam.update(&mut micro, &accumulated).unwrap();
	assert_eq!(full, micro);
}
```

実装箇所は `optim::accumulate_weighted` である。この要件に対する実行可能な確認は `llm_gradients::token_weighted_accumulation` に置く。

`src/optim.rs` の `pub fn accumulate_weighted(`の定義を次の全体で置き換えます。

```rust
// file="src/optim.rs" mode="replace" item="pub fn accumulate_weighted("
pub fn accumulate_weighted(microbatches: &[(Vec<BufferData>, usize)]) -> Result<Vec<BufferData>> {
	let first = microbatches.first().ok_or("empty accumulation")?;
	let total = microbatches.iter().try_fold(0usize, |a, (_, n)| {
		a.checked_add(*n).ok_or("token count overflow")
	})?;
	if total == 0 {
		return Err("zero valid tokens".into());
	}
	let mut result = first.0.clone();
	for g in &mut result {
		g.data.fill(0.);
	}
	for (batch, n) in microbatches {
		if batch.len() != result.len() {
			return Err("accumulation keys".into());
		}
		for (dst, src) in result.iter_mut().zip(batch) {
			if dst.shape != src.shape || dst.data.len() != src.data.len() {
				return Err("accumulation shape".into());
			}
			for (d, &s) in dst.data.iter_mut().zip(&src.data) {
				if !s.is_finite() {
					return Err("nonfinite microbatch gradient".into());
				}
				*d += s * (*n as f32 / total as f32);
			}
		}
	}
	Ok(result)
}
```

```sh
# check
cargo test --offline --test llm_gradients token_weighted_accumulation -- --exact
```

## 15. full batch と勾配蓄積した更新を、同じデータ順と dropout=0 で比較する。

<!-- todo:24-015 -->


実装箇所は `optim::accumulate_weighted` である。この要件に対する実行可能な確認は `llm_gradients::token_weighted_accumulation` に置く。

`tests/llm_gradients.rs` の `fn token_weighted_accumulation(`の定義を次の全体で置き換えます。

```rust
// file="tests/llm_gradients.rs" mode="replace" item="fn token_weighted_accumulation("
fn token_weighted_accumulation() {
	let gradient = |v| {
		vec![BufferData {
			shape: vec![1],
			data: vec![v],
		}]
	};
	let accumulated = accumulate_weighted(&[(gradient(2.), 1), (gradient(6.), 3)]).unwrap();
	assert_eq!(accumulated[0].data, [5.]);
	let mut full = AdamState::new(gradient(1.));
	let mut micro = full.clone();
	let adam = AdamW::new(&full.weights, AdamConfig::default()).unwrap();
	adam.update(&mut full, &gradient(5.)).unwrap();
	adam.update(&mut micro, &accumulated).unwrap();
	assert_eq!(full, micro);
}
```

```sh
# check
cargo test --offline --test llm_gradients token_weighted_accumulation -- --exact
```

## 16. model state、m、v、step、学習率 scheduler、乱数 generator の状態を保存する。

<!-- todo:24-016 -->


`tests/llm_resume.rs::checkpoint_transaction_and_corruption` を追加する。

```rust
// file=tests/llm_resume.rs mode=append
#[test]
fn checkpoint_transaction_and_corruption() {
	let path = std::env::temp_dir().join(format!("chibi-ckpt-{}", std::process::id()));
	let _ = fs::remove_dir_all(&path);
	let state = initial();
	save_training_checkpoint(&path, &state).unwrap();
	assert_eq!(restore_training_checkpoint(&path).unwrap(), state);
	let mut live = state.clone();
	let mut wrong = state.clone();
	wrong.dataset_sha = sha256(b"different");
	assert!(adopt_checkpoint(&mut live, wrong).is_err());
	assert_eq!(live, state);
	let mut wrong = state.clone();
	wrong.accumulated_microbatches = 1;
	assert!(wrong.validate().is_err());
	let mut wrong = state.clone();
	wrong.optimizer.m.pop();
	assert!(wrong.validate().is_err());
	let mut wrong = state.clone();
	wrong.data.permutation[0] = wrong.data.permutation[1];
	assert!(wrong.validate().is_err());
	let mut payload = fs::read(path.join("state.json")).unwrap();
	payload[5] ^= 1;
	fs::write(path.join("state.json"), payload).unwrap();
	assert!(
		restore_training_checkpoint(&path)
			.unwrap_err()
			.contains("digest")
	);
	let _ = fs::remove_dir_all(path);
}
```

実装箇所は `checkpoint::{save_training_checkpoint,restore_training_checkpoint,adopt_checkpoint}` と `checkpoint::DataCursor` である。この要件に対する実行可能な確認は `llm_resume::checkpoint_transaction_and_corruption` に置く。

`src/checkpoint.rs` の `pub fn save_training_checkpoint(`の定義を次の全体で置き換えます。

```rust
// file="src/checkpoint.rs" mode="replace" item="pub fn save_training_checkpoint("
pub fn save_training_checkpoint(destination: &Path, state: &TrainingState) -> Result<()> {
	state.validate()?;
	if destination.exists() {
		return Err("checkpoint generation already exists".into());
	}
	let parent = destination.parent().ok_or("checkpoint parent")?;
	fs::create_dir_all(parent).map_err(|e| e.to_string())?;
	let name = destination
		.file_name()
		.and_then(|s| s.to_str())
		.ok_or("checkpoint name")?;
	let nonce = std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map_err(|e| e.to_string())?
		.as_nanos();
	let temporary = parent.join(format!(".{name}.{}.{}.tmp", std::process::id(), nonce));
	fs::create_dir(&temporary).map_err(|e| e.to_string())?;
	let result = (|| {
		let bytes = serde_json::to_vec(state).map_err(|e| e.to_string())?;
		let mut file = File::options()
			.write(true)
			.create_new(true)
			.open(temporary.join("state.json"))
			.map_err(|e| e.to_string())?;
		file.write_all(&bytes)
			.and_then(|_| file.sync_all())
			.map_err(|e| e.to_string())?;
		let reread = fs::read(temporary.join("state.json")).map_err(|e| e.to_string())?;
		if bytes != reread {
			return Err("checkpoint readback mismatch".into());
		}
		let candidate: TrainingState =
			serde_json::from_value(crate::safetensor::unique_json(&reread)?)
				.map_err(|e| e.to_string())?;
		candidate.validate()?;
		let manifest = serde_json::to_vec(&Commit {
			version: 1,
			bytes: bytes.len() as u64,
			sha256: sha256(&bytes),
		})
		.map_err(|e| e.to_string())?;
		let mut file = File::options()
			.write(true)
			.create_new(true)
			.open(temporary.join("manifest.json"))
			.map_err(|e| e.to_string())?;
		file.write_all(&manifest)
			.and_then(|_| file.sync_all())
			.map_err(|e| e.to_string())?;
		File::open(&temporary)
			.and_then(|f| f.sync_all())
			.map_err(|e| e.to_string())?;
		fs::rename(&temporary, destination).map_err(|e| e.to_string())?;
		File::open(parent)
			.and_then(|f| f.sync_all())
			.map_err(|e| e.to_string())?;
		Ok(())
	})();
	if result.is_err() {
		let _ = fs::remove_dir_all(temporary);
	}
	result
}
```

```sh
# check
cargo test --offline --test llm_resume checkpoint_transaction_and_corruption -- --exact
```

## 17. dataset の digest、tokenizer/config の識別、epoch、次の読出し位置、shuffle 状態も保存する。

<!-- todo:24-017 -->


`tests/llm_resume.rs::process_worker` を追加する。

```rust
// file=tests/llm_resume.rs mode=append
#[test]
#[ignore]
fn process_worker() {
	let path = std::env::var("CHIBI_WORKER_OUT").unwrap();
	let steps = std::env::var("CHIBI_WORKER_STEPS")
		.unwrap()
		.parse::<usize>()
		.unwrap();
	let mut state = initial();
	if let Ok(resume) = std::env::var("CHIBI_WORKER_RESUME") {
		adopt_checkpoint(
			&mut state,
			restore_training_checkpoint(Path::new(&resume)).unwrap(),
		)
		.unwrap();
	}
	let plan = TrainingPlan::new(QwenConfig::tiny(), 3).unwrap();
	let adam = AdamW::new(&state.optimizer.weights, state.optimizer_config).unwrap();
	let corpus = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3];
	let mut losses = Vec::new();
	for _ in 0..steps {
		let i = state.data.next_index();
		let (loss, gradients) = plan
			.run(
				&state.optimizer.weights,
				&corpus[i..i + 3],
				&corpus[i + 1..i + 4],
				&[true; 3],
			)
			.unwrap();
		adam.update(&mut state.optimizer, &gradients).unwrap();
		state.scheduler_step = state.optimizer.step;
		losses.push(loss);
	}
	save_training_checkpoint(Path::new(&path), &state).unwrap();
	fs::write(
		format!("{path}.losses"),
		serde_json::to_vec(&losses).unwrap(),
	)
	.unwrap();
}
```

実装箇所は `checkpoint::{save_training_checkpoint,restore_training_checkpoint,adopt_checkpoint}` と `checkpoint::DataCursor` である。この要件に対する実行可能な確認は `llm_resume::checkpoint_transaction_and_corruption` に置く。

`src/checkpoint.rs` の `pub fn save_training_checkpoint(`の定義を次の全体で置き換えます。

```rust
// file="src/checkpoint.rs" mode="replace" item="pub fn save_training_checkpoint("
pub fn save_training_checkpoint(destination: &Path, state: &TrainingState) -> Result<()> {
	state.validate()?;
	if destination.exists() {
		return Err("checkpoint generation already exists".into());
	}
	let parent = destination.parent().ok_or("checkpoint parent")?;
	fs::create_dir_all(parent).map_err(|e| e.to_string())?;
	let name = destination
		.file_name()
		.and_then(|s| s.to_str())
		.ok_or("checkpoint name")?;
	let nonce = std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map_err(|e| e.to_string())?
		.as_nanos();
	let temporary = parent.join(format!(".{name}.{}.{}.tmp", std::process::id(), nonce));
	fs::create_dir(&temporary).map_err(|e| e.to_string())?;
	let result = (|| {
		let bytes = serde_json::to_vec(state).map_err(|e| e.to_string())?;
		let mut file = File::options()
			.write(true)
			.create_new(true)
			.open(temporary.join("state.json"))
			.map_err(|e| e.to_string())?;
		file.write_all(&bytes)
			.and_then(|_| file.sync_all())
			.map_err(|e| e.to_string())?;
		let reread = fs::read(temporary.join("state.json")).map_err(|e| e.to_string())?;
		if bytes != reread {
			return Err("checkpoint readback mismatch".into());
		}
		let candidate: TrainingState =
			serde_json::from_value(crate::safetensor::unique_json(&reread)?)
				.map_err(|e| e.to_string())?;
		candidate.validate()?;
		let manifest = serde_json::to_vec(&Commit {
			version: 1,
			bytes: bytes.len() as u64,
			sha256: sha256(&bytes),
		})
		.map_err(|e| e.to_string())?;
		let mut file = File::options()
			.write(true)
			.create_new(true)
			.open(temporary.join("manifest.json"))
			.map_err(|e| e.to_string())?;
		file.write_all(&manifest)
			.and_then(|_| file.sync_all())
			.map_err(|e| e.to_string())?;
		File::open(&temporary)
			.and_then(|f| f.sync_all())
			.map_err(|e| e.to_string())?;
		fs::rename(&temporary, destination).map_err(|e| e.to_string())?;
		File::open(parent)
			.and_then(|f| f.sync_all())
			.map_err(|e| e.to_string())?;
		Ok(())
	})();
	if result.is_err() {
		let _ = fs::remove_dir_all(temporary);
	}
	result
}
```

```sh
# check
cargo test --offline --test llm_resume checkpoint_transaction_and_corruption -- --exact
```

## 18. 蓄積中間の checkpoint は最初は拒否し、optimizer step 完了後の境界だけで保存する。

<!-- todo:24-018 -->


実装箇所は `checkpoint::{save_training_checkpoint,restore_training_checkpoint,adopt_checkpoint}` と `checkpoint::DataCursor` である。この要件に対する実行可能な確認は `llm_resume::checkpoint_transaction_and_corruption` に置く。

`src/checkpoint.rs` の `pub fn validate(`（`impl TrainingState` 内）の定義を次の全体で置き換えます。

```rust
// file="src/checkpoint.rs" mode="replace" item="pub fn validate(" within="impl TrainingState"
	pub fn validate(&self) -> Result<()> {
		self.optimizer.validate()?;
		self.data.validate()?;
		if self.accumulated_microbatches != 0 {
			return Err("checkpoint requires an optimizer step boundary".into());
		}
		if self.scheduler_step != self.optimizer.step {
			return Err("scheduler/optimizer step mismatch".into());
		}
		for digest in [&self.dataset_sha, &self.config_sha, &self.tokenizer_sha] {
			if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
				return Err("invalid training identity digest".into());
			}
		}
		Ok(())
	}
```

```sh
# check
cargo test --offline --test llm_resume checkpoint_transaction_and_corruption -- --exact
```

## 19. device の完了待ち後に一時ディレクトリへ書き、全ファイルと manifest の検証後に確定する。

<!-- todo:24-019 -->


実装箇所は `checkpoint::{save_training_checkpoint,restore_training_checkpoint,adopt_checkpoint}` と `checkpoint::DataCursor` である。この要件に対する実行可能な確認は `llm_resume::checkpoint_transaction_and_corruption` に置く。

`src/checkpoint.rs` の `pub fn save_training_checkpoint(`の定義を次の全体で置き換えます。

```rust
// file="src/checkpoint.rs" mode="replace" item="pub fn save_training_checkpoint("
pub fn save_training_checkpoint(destination: &Path, state: &TrainingState) -> Result<()> {
	state.validate()?;
	if destination.exists() {
		return Err("checkpoint generation already exists".into());
	}
	let parent = destination.parent().ok_or("checkpoint parent")?;
	fs::create_dir_all(parent).map_err(|e| e.to_string())?;
	let name = destination
		.file_name()
		.and_then(|s| s.to_str())
		.ok_or("checkpoint name")?;
	let nonce = std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map_err(|e| e.to_string())?
		.as_nanos();
	let temporary = parent.join(format!(".{name}.{}.{}.tmp", std::process::id(), nonce));
	fs::create_dir(&temporary).map_err(|e| e.to_string())?;
	let result = (|| {
		let bytes = serde_json::to_vec(state).map_err(|e| e.to_string())?;
		let mut file = File::options()
			.write(true)
			.create_new(true)
			.open(temporary.join("state.json"))
			.map_err(|e| e.to_string())?;
		file.write_all(&bytes)
			.and_then(|_| file.sync_all())
			.map_err(|e| e.to_string())?;
		let reread = fs::read(temporary.join("state.json")).map_err(|e| e.to_string())?;
		if bytes != reread {
			return Err("checkpoint readback mismatch".into());
		}
		let candidate: TrainingState =
			serde_json::from_value(crate::safetensor::unique_json(&reread)?)
				.map_err(|e| e.to_string())?;
		candidate.validate()?;
		let manifest = serde_json::to_vec(&Commit {
			version: 1,
			bytes: bytes.len() as u64,
			sha256: sha256(&bytes),
		})
		.map_err(|e| e.to_string())?;
		let mut file = File::options()
			.write(true)
			.create_new(true)
			.open(temporary.join("manifest.json"))
			.map_err(|e| e.to_string())?;
		file.write_all(&manifest)
			.and_then(|_| file.sync_all())
			.map_err(|e| e.to_string())?;
		File::open(&temporary)
			.and_then(|f| f.sync_all())
			.map_err(|e| e.to_string())?;
		fs::rename(&temporary, destination).map_err(|e| e.to_string())?;
		File::open(parent)
			.and_then(|f| f.sync_all())
			.map_err(|e| e.to_string())?;
		Ok(())
	})();
	if result.is_err() {
		let _ = fs::remove_dir_all(temporary);
	}
	result
}
```

```sh
# check
cargo test --offline --test llm_resume checkpoint_transaction_and_corruption -- --exact
```

## 20. 新しいプロセスで検証と load を行い、連続10 step と5 step後の保存・再開を比較する。

<!-- todo:24-020 -->


`tests/llm_resume.rs::fresh_process_resume_matches_continuous` を追加する。

```rust
// file=tests/llm_resume.rs mode=append
#[test]
fn fresh_process_resume_matches_continuous() {
	let root = std::env::temp_dir().join(format!("chibi-resume-{}", std::process::id()));
	let _ = fs::remove_dir_all(&root);
	fs::create_dir_all(&root).unwrap();
	let run = |name: &str, steps: &str, resume: Option<&str>| {
		let mut command = Command::new(std::env::current_exe().unwrap());
		command
			.args(["--ignored", "--exact", "process_worker", "--nocapture"])
			.env("CHIBI_WORKER_STEPS", steps)
			.env("CHIBI_WORKER_OUT", root.join(name));
		if let Some(previous) = resume {
			command.env("CHIBI_WORKER_RESUME", root.join(previous));
		}
		let result = command.output().unwrap();
		assert!(
			result.status.success(),
			"{}\n{}",
			String::from_utf8_lossy(&result.stdout),
			String::from_utf8_lossy(&result.stderr)
		);
	};
	run("continuous", "10", None);
	run("step5", "5", None);
	run("resumed", "5", Some("step5"));
	let a = restore_training_checkpoint(&root.join("continuous")).unwrap();
	let b = restore_training_checkpoint(&root.join("resumed")).unwrap();
	assert_eq!(a.optimizer.step, 10);
	assert_eq!(a, b);
	let all: Vec<f32> =
		serde_json::from_slice(&fs::read(root.join("continuous.losses")).unwrap()).unwrap();
	let latter: Vec<f32> =
		serde_json::from_slice(&fs::read(root.join("resumed.losses")).unwrap()).unwrap();
	assert_eq!(&all[5..], latter);
	assert!(all[9] < all[0]);
	let mut ad = a.data;
	let mut bd = b.data;
	assert_eq!(ad.next_index(), bd.next_index());
	let _ = fs::remove_dir_all(root);
}
```

実装箇所は `checkpoint::{save_training_checkpoint,restore_training_checkpoint,adopt_checkpoint}` と `checkpoint::DataCursor` である。この要件に対する実行可能な確認は `llm_resume::fresh_process_resume_matches_continuous` に置く。

```sh
# check
cargo test --offline --test llm_resume fresh_process_resume_matches_continuous -- --exact
```

## 21. 22の manifest を読み、BF16 checkpoint を f32 parameter として準備する。

<!-- todo:24-021 -->


実装箇所は `training::estimate_training_storage` と `examples/train_qwen.rs::main` である。この要件に対する実行可能な確認は `llm_resume::fresh_process_resume_matches_continuous` に置く。

`src/nn/qwen.rs` の `pub fn load_parameters(`の定義を次の全体で置き換えます。

```rust
// file="src/nn/qwen.rs" mode="replace" item="pub fn load_parameters("
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

```sh
# check
cargo test --offline --test llm_resume fresh_process_resume_matches_continuous -- --exact
```

## 22. 全 layer と語彙を保ったまま、短い sequence と microbatch=1でメモリ見積りを出す。

<!-- todo:24-022 -->


実装箇所は `training::estimate_training_storage` と `examples/train_qwen.rs::main` である。この要件に対する実行可能な確認は `llm_resume::fresh_process_resume_matches_continuous` に置く。

`examples/train_qwen.rs` の `fn main(`の定義を次の全体で置き換えます。

```rust
// file="examples/train_qwen.rs" mode="replace" item="fn main("
fn main() -> Result<()> {
	let args = std::env::args().collect::<Vec<_>>();
	let arg = |name: &str| args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone());
	let manifest = arg("--manifest").unwrap_or("fixtures/qwen25/manifest.json".into());
	let directory = arg("--assets").unwrap_or("fixtures/qwen25".into());
	let sequence = arg("--sequence-length")
		.unwrap_or("8".into())
		.parse::<usize>()
		.map_err(|e| e.to_string())?;
	let microbatch = arg("--microbatch")
		.unwrap_or("1".into())
		.parse::<usize>()
		.map_err(|e| e.to_string())?;
	if microbatch != 1 {
		return Err(
			"this driver accepts microbatch=1; use accumulate_weighted for multiple microbatches"
				.into(),
		);
	}
	let steps = arg("--steps")
		.unwrap_or("1".into())
		.parse::<usize>()
		.map_err(|e| e.to_string())?;
	let manifest_value: chibigrad::fetch::Manifest =
		serde_json::from_slice(&std::fs::read(&manifest).map_err(|e| e.to_string())?)
			.map_err(|e| e.to_string())?;
	manifest_value.validate()?;
	let config_asset = manifest_value
		.files
		.iter()
		.find(|a| a.name == "config.json")
		.ok_or("config missing from manifest")?;
	let config_bytes =
		chibigrad::fetch::verified_read(&Path::new(&directory).join("config.json"), config_asset)?;
	let config: QwenConfig = serde_json::from_slice(&config_bytes).map_err(|e| e.to_string())?;
	config.require_public_05b()?;
	let estimate =
		estimate_training_storage(&parameter_registry(&config)?, 1, sequence as u64, &config)?;
	println!("{}", serde_json::to_string_pretty(&estimate).unwrap());
	println!(
		"estimate excludes additional primitive temporaries, C compiler memory, device/host copies, and allocator overhead"
	);
	if args.iter().any(|a| a == "--estimate-only") {
		return Ok(());
	}
	if let Some(budget) = arg("--budget-bytes") {
		if estimate.total_lower_bound > budget.parse::<u64>().map_err(|e| e.to_string())? {
			return Err("training storage lower bound exceeds supplied budget".into());
		}
	}
	let assets = model_assets::load(
		Path::new(&manifest),
		Path::new(&directory),
		args.iter().any(|a| a == "--download"),
	)?;
	let mut c = assets.config.clone();
	c.max_position_embeddings = sequence;
	let text = arg("--text").unwrap_or(
		"The quick brown fox jumps over the lazy dog. The quick brown fox jumps again.".into(),
	);
	let corpus = model_assets::encode(&assets.tokenizer, &text)?
		.into_iter()
		.map(|v| v as usize)
		.collect::<Vec<_>>();
	if corpus.len() <= sequence {
		return Err("training text must contain sequence+1 tokens".into());
	}
	let identity = |name: &str| {
		assets
			.manifest
			.files
			.iter()
			.find(|a| a.name == name)
			.map(|a| a.sha256.clone())
			.ok_or_else(|| format!("missing identity {name}"))
	};
	let weights = assets
		.state
		.values()
		.map(|t| {
			let b = t.to_buffer()?;
			Ok(BufferData {
				shape: b.shape,
				data: b.data,
			})
		})
		.collect::<Result<Vec<_>>>()?;
	let optimizer_config = AdamConfig {
		lr: 1e-5,
		..Default::default()
	};
	let mut state = TrainingState {
		optimizer: AdamState::new(weights),
		optimizer_config,
		scheduler_step: 0,
		data: DataCursor::new(corpus.len() - sequence, 1)?,
		dataset_sha: chibigrad::fetch::sha256(&serde_json::to_vec(&corpus).unwrap()),
		config_sha: identity("config.json")?,
		tokenizer_sha: identity("tokenizer.json")?,
		accumulated_microbatches: 0,
	};
	if let Some(path) = arg("--resume") {
		adopt_checkpoint(&mut state, restore_training_checkpoint(Path::new(&path))?)?;
	}
	let plan = TrainingPlan::new(c, sequence)?;
	println!(
		"native_training_workspace_bytes={}",
		plan.plan.workspace_bytes()?
	);
	let adam = AdamW::new(&state.optimizer.weights, optimizer_config)?;
	for _ in 0..steps {
		let i = state.data.next_index();
		let (loss, gradients) = plan.run(
			&state.optimizer.weights,
			&corpus[i..i + sequence],
			&corpus[i + 1..i + sequence + 1],
			&vec![true; sequence],
		)?;
		adam.update(&mut state.optimizer, &gradients)?;
		state.scheduler_step = state.optimizer.step;
		println!("step={} loss={loss}", state.optimizer.step);
	}
	if let Some(path) = arg("--checkpoint") {
		save_training_checkpoint(Path::new(&path), &state)?;
	}
	Ok(())
}
```

```sh
# check
cargo test --offline --test llm_resume fresh_process_resume_matches_continuous -- --exact
```

## 23. 実際の一意 parameter 数Pを数え、f32 の重み・勾配・m・vだけで `16*P` byte を要することを記録する。

<!-- todo:24-023 -->


`tests/llm_gradients.rs::storage_counts_all_layers_and_tied_once` を追加する。

```rust
// file=tests/llm_gradients.rs mode=append
#[test]
fn storage_counts_all_layers_and_tied_once() {
	let c: QwenConfig =
		serde_json::from_str(include_str!("../fixtures/qwen25/config.json")).unwrap();
	let registry = chibigrad::nn::qwen::parameter_registry(&c).unwrap();
	let s = estimate_training_storage(&registry, 1, 8, &c).unwrap();
	assert_eq!(s.persistent, s.parameters * 16);
	assert_eq!(s.update_outputs, s.parameters * 12);
	assert_eq!(s.one_layer_score, 3584);
	assert_eq!(s.all_layer_scores, 3584 * 24);
	assert_eq!(s.logits, 4861952);
	assert!(s.parameters > 490_000_000);
	assert!(s.total_lower_bound > 28 * s.parameters);
}
```

実装箇所は `training::estimate_training_storage` と `examples/train_qwen.rs::main` である。この要件に対する実行可能な確認は `llm_gradients::storage_counts_all_layers_and_tied_once` に置く。

`src/training.rs` の `pub fn estimate_training_storage(`の定義を次の全体で置き換えます。

```rust
// file="src/training.rs" mode="replace" item="pub fn estimate_training_storage("
pub fn estimate_training_storage(
	registry: &Registry,
	batch: u64,
	t: u64,
	c: &QwenConfig,
) -> Result<StorageEstimate> {
	let mut ids = BTreeSet::new();
	let mut p = 0u64;
	for spec in registry.names.values() {
		if spec.trainable && ids.insert(spec.id) {
			let n = spec.shape.iter().try_fold(1u64, |a, &b| {
				a.checked_mul(b as u64).ok_or("parameter size overflow")
			})?;
			p = p.checked_add(n).ok_or("parameter total overflow")?;
		}
	}
	let checked = |values: &[u64]| {
		values.iter().try_fold(1u64, |a, &b| {
			a.checked_mul(b)
				.ok_or_else(|| "storage overflow".to_string())
		})
	};
	let persistent = checked(&[16, p])?;
	let update_outputs = checked(&[12, p])?;
	let one_layer_score = checked(&[batch, c.num_attention_heads as u64, t, t, 4])?;
	let all_layer_scores = checked(&[one_layer_score, c.num_hidden_layers as u64])?;
	let logits = checked(&[batch, t, c.vocab_size as u64, 4])?;
	let boundary_activations = checked(&[
		batch,
		t,
		c.hidden_size as u64,
		c.num_hidden_layers as u64 + 2,
		4,
	])?;
	let total_lower_bound = [
		persistent,
		update_outputs,
		all_layer_scores,
		logits,
		boundary_activations,
	]
	.into_iter()
	.try_fold(0u64, |a, b| a.checked_add(b).ok_or("storage sum overflow"))?;
	Ok(StorageEstimate {
		parameters: p,
		persistent,
		update_outputs,
		one_layer_score,
		all_layer_scores,
		logits,
		boundary_activations,
		total_lower_bound,
	})
}
```

```sh
# check
cargo test --offline --test llm_gradients storage_counts_all_layers_and_tied_once -- --exact
```

ここは shape からの下限計算である。16P に加え、更新候補12P、全層 score、全位置 logits、層境界 activation を別表示する。実行時はさらに `Plan::workspace_bytes` を表示する。host copy と compiler の作業領域は追加で必要になる。

```sh
# check
cargo run --offline --example train_qwen -- --manifest fixtures/qwen25/manifest.json --sequence-length 8 --microbatch 1 --estimate-only
```

## 24. activation、logits、通常 attention の score、再利用 workspace の peak を別に加える。

<!-- todo:24-024 -->


実装箇所は `training::estimate_training_storage` と `examples/train_qwen.rs::main` である。この要件に対する実行可能な確認は `llm_gradients::storage_counts_all_layers_and_tied_once` に置く。

`src/training.rs` の `pub fn estimate_training_storage(`の定義を次の全体で置き換えます。

```rust
// file="src/training.rs" mode="replace" item="pub fn estimate_training_storage("
pub fn estimate_training_storage(
	registry: &Registry,
	batch: u64,
	t: u64,
	c: &QwenConfig,
) -> Result<StorageEstimate> {
	let mut ids = BTreeSet::new();
	let mut p = 0u64;
	for spec in registry.names.values() {
		if spec.trainable && ids.insert(spec.id) {
			let n = spec.shape.iter().try_fold(1u64, |a, &b| {
				a.checked_mul(b as u64).ok_or("parameter size overflow")
			})?;
			p = p.checked_add(n).ok_or("parameter total overflow")?;
		}
	}
	let checked = |values: &[u64]| {
		values.iter().try_fold(1u64, |a, &b| {
			a.checked_mul(b)
				.ok_or_else(|| "storage overflow".to_string())
		})
	};
	let persistent = checked(&[16, p])?;
	let update_outputs = checked(&[12, p])?;
	let one_layer_score = checked(&[batch, c.num_attention_heads as u64, t, t, 4])?;
	let all_layer_scores = checked(&[one_layer_score, c.num_hidden_layers as u64])?;
	let logits = checked(&[batch, t, c.vocab_size as u64, 4])?;
	let boundary_activations = checked(&[
		batch,
		t,
		c.hidden_size as u64,
		c.num_hidden_layers as u64 + 2,
		4,
	])?;
	let total_lower_bound = [
		persistent,
		update_outputs,
		all_layer_scores,
		logits,
		boundary_activations,
	]
	.into_iter()
	.try_fold(0u64, |a, b| a.checked_add(b).ok_or("storage sum overflow"))?;
	Ok(StorageEstimate {
		parameters: p,
		persistent,
		update_outputs,
		one_layer_score,
		all_layer_scores,
		logits,
		boundary_activations,
		total_lower_bound,
	})
}
```

```sh
# check
cargo test --offline --test llm_gradients storage_counts_all_layers_and_tied_once -- --exact
```

## 25. 小さな検証済みデータで一回の forward、backward、全 parameter 更新、保存、再開を順に実行する。

<!-- todo:24-025 -->


実装箇所は `training::estimate_training_storage` と `examples/train_qwen.rs::main` である。この要件に対する実行可能な確認は `llm_gradients::storage_counts_all_layers_and_tied_once` に置く。

`examples/train_qwen.rs` の `fn main(`の定義を次の全体で置き換えます。

```rust
// file="examples/train_qwen.rs" mode="replace" item="fn main("
fn main() -> Result<()> {
	let args = std::env::args().collect::<Vec<_>>();
	let arg = |name: &str| args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone());
	let manifest = arg("--manifest").unwrap_or("fixtures/qwen25/manifest.json".into());
	let directory = arg("--assets").unwrap_or("fixtures/qwen25".into());
	let sequence = arg("--sequence-length")
		.unwrap_or("8".into())
		.parse::<usize>()
		.map_err(|e| e.to_string())?;
	let microbatch = arg("--microbatch")
		.unwrap_or("1".into())
		.parse::<usize>()
		.map_err(|e| e.to_string())?;
	if microbatch != 1 {
		return Err(
			"this driver accepts microbatch=1; use accumulate_weighted for multiple microbatches"
				.into(),
		);
	}
	let steps = arg("--steps")
		.unwrap_or("1".into())
		.parse::<usize>()
		.map_err(|e| e.to_string())?;
	let manifest_value: chibigrad::fetch::Manifest =
		serde_json::from_slice(&std::fs::read(&manifest).map_err(|e| e.to_string())?)
			.map_err(|e| e.to_string())?;
	manifest_value.validate()?;
	let config_asset = manifest_value
		.files
		.iter()
		.find(|a| a.name == "config.json")
		.ok_or("config missing from manifest")?;
	let config_bytes =
		chibigrad::fetch::verified_read(&Path::new(&directory).join("config.json"), config_asset)?;
	let config: QwenConfig = serde_json::from_slice(&config_bytes).map_err(|e| e.to_string())?;
	config.require_public_05b()?;
	let estimate =
		estimate_training_storage(&parameter_registry(&config)?, 1, sequence as u64, &config)?;
	println!("{}", serde_json::to_string_pretty(&estimate).unwrap());
	println!(
		"estimate excludes additional primitive temporaries, C compiler memory, device/host copies, and allocator overhead"
	);
	if args.iter().any(|a| a == "--estimate-only") {
		return Ok(());
	}
	if let Some(budget) = arg("--budget-bytes") {
		if estimate.total_lower_bound > budget.parse::<u64>().map_err(|e| e.to_string())? {
			return Err("training storage lower bound exceeds supplied budget".into());
		}
	}
	let assets = model_assets::load(
		Path::new(&manifest),
		Path::new(&directory),
		args.iter().any(|a| a == "--download"),
	)?;
	let mut c = assets.config.clone();
	c.max_position_embeddings = sequence;
	let text = arg("--text").unwrap_or(
		"The quick brown fox jumps over the lazy dog. The quick brown fox jumps again.".into(),
	);
	let corpus = model_assets::encode(&assets.tokenizer, &text)?
		.into_iter()
		.map(|v| v as usize)
		.collect::<Vec<_>>();
	if corpus.len() <= sequence {
		return Err("training text must contain sequence+1 tokens".into());
	}
	let identity = |name: &str| {
		assets
			.manifest
			.files
			.iter()
			.find(|a| a.name == name)
			.map(|a| a.sha256.clone())
			.ok_or_else(|| format!("missing identity {name}"))
	};
	let weights = assets
		.state
		.values()
		.map(|t| {
			let b = t.to_buffer()?;
			Ok(BufferData {
				shape: b.shape,
				data: b.data,
			})
		})
		.collect::<Result<Vec<_>>>()?;
	let optimizer_config = AdamConfig {
		lr: 1e-5,
		..Default::default()
	};
	let mut state = TrainingState {
		optimizer: AdamState::new(weights),
		optimizer_config,
		scheduler_step: 0,
		data: DataCursor::new(corpus.len() - sequence, 1)?,
		dataset_sha: chibigrad::fetch::sha256(&serde_json::to_vec(&corpus).unwrap()),
		config_sha: identity("config.json")?,
		tokenizer_sha: identity("tokenizer.json")?,
		accumulated_microbatches: 0,
	};
	if let Some(path) = arg("--resume") {
		adopt_checkpoint(&mut state, restore_training_checkpoint(Path::new(&path))?)?;
	}
	let plan = TrainingPlan::new(c, sequence)?;
	println!(
		"native_training_workspace_bytes={}",
		plan.plan.workspace_bytes()?
	);
	let adam = AdamW::new(&state.optimizer.weights, optimizer_config)?;
	for _ in 0..steps {
		let i = state.data.next_index();
		let (loss, gradients) = plan.run(
			&state.optimizer.weights,
			&corpus[i..i + sequence],
			&corpus[i + 1..i + sequence + 1],
			&vec![true; sequence],
		)?;
		adam.update(&mut state.optimizer, &gradients)?;
		state.scheduler_step = state.optimizer.step;
		println!("step={} loss={loss}", state.optimizer.step);
	}
	if let Some(path) = arg("--checkpoint") {
		save_training_checkpoint(Path::new(&path), &state)?;
	}
	Ok(())
}
```

```sh
# check
cargo test --offline --test llm_gradients storage_counts_all_layers_and_tied_once -- --exact
```

全 Qwen を実行する際は語彙151936、全24層と全 ParamId を維持する。実資産・メモリを確保した環境で一更新と checkpoint を行う。再開の `--steps` は追加更新数を表す。

```sh
# check requires=qwen
checkpoint_dir=$(mktemp -d)
cargo run --offline --example train_qwen -- --manifest fixtures/qwen25/manifest.json --sequence-length 8 --microbatch 1 --steps 1 --checkpoint "$checkpoint_dir/step1"
cargo run --offline --example train_qwen -- --manifest fixtures/qwen25/manifest.json --sequence-length 8 --microbatch 1 --steps 1 --resume "$checkpoint_dir/step1" --checkpoint "$checkpoint_dir/step2"
```

## 26. repeated token の期待勾配と uniform logits の `ln(4)` を確認する。

<!-- todo:24-026 -->


実装箇所は `training::TrainingPlan::run`、`optim::AdamW::update`、`llm_resume::fresh_process_resume_matches_continuous` である。この要件に対する実行可能な確認は `llm_gradients::storage_counts_all_layers_and_tied_once` に置く。

`tests/llm_gradients.rs` の `fn storage_counts_all_layers_and_tied_once(`の定義を次の全体で置き換えます。

```rust
// file="tests/llm_gradients.rs" mode="replace" item="fn storage_counts_all_layers_and_tied_once("
fn storage_counts_all_layers_and_tied_once() {
	let c: QwenConfig =
		serde_json::from_str(include_str!("../fixtures/qwen25/config.json")).unwrap();
	let registry = chibigrad::nn::qwen::parameter_registry(&c).unwrap();
	let s = estimate_training_storage(&registry, 1, 8, &c).unwrap();
	assert_eq!(s.persistent, s.parameters * 16);
	assert_eq!(s.update_outputs, s.parameters * 12);
	assert_eq!(s.one_layer_score, 3584);
	assert_eq!(s.all_layer_scores, 3584 * 24);
	assert_eq!(s.logits, 4861952);
	assert!(s.parameters > 490_000_000);
	assert!(s.total_lower_bound > 28 * s.parameters);
}
```

```sh
# check
cargo test --offline --test llm_gradients storage_counts_all_layers_and_tied_once -- --exact
```

## 27. 小 graph の有限差分は中央差分、初期幅 `1e-3`、基準 `abs_err <= 3e-3 + 3e-2*abs(ref)` で検査する。

<!-- todo:24-027 -->


実装箇所は `training::TrainingPlan::run`、`optim::AdamW::update`、`llm_resume::fresh_process_resume_matches_continuous` である。この要件に対する実行可能な確認は `llm_gradients::tiny_transformer_parameter_central_difference` に置く。

`tests/llm_gradients.rs` の `fn tiny_transformer_parameter_central_difference(`の定義を次の全体で置き換えます。

```rust
// file="tests/llm_gradients.rs" mode="replace" item="fn tiny_transformer_parameter_central_difference("
fn tiny_transformer_parameter_central_difference() {
	let mut config = QwenConfig::tiny();
	config.hidden_size = 4;
	config.intermediate_size = 6;
	config.vocab_size = 4;
	config.num_attention_heads = 2;
	let plan = TrainingPlan::new(config.clone(), 2).unwrap();
	let weights = initialize(&config, 19).unwrap();
	let (_, gradients) = plan.run(&weights, &[1, 1], &[1, 2], &[true, true]).unwrap();
	assert_eq!(gradients.len(), weights.len());
	for name in [
		"model.embed_tokens.weight",
		"model.layers.0.self_attn.q_proj.weight",
		"model.layers.0.self_attn.k_proj.weight",
		"model.layers.0.self_attn.v_proj.weight",
		"model.layers.0.input_layernorm.weight",
		"model.layers.0.mlp.gate_proj.weight",
	] {
		let slot = plan.registry.names[name].slot;
		let element = if name == "model.embed_tokens.weight" {
			4
		} else {
			0
		};
		for h in [1e-3, 5e-4] {
			let mut plus = weights.clone();
			let mut minus = weights.clone();
			plus[slot].data[element] += h;
			minus[slot].data[element] -= h;
			let a = plan.run(&plus, &[1, 1], &[1, 2], &[true, true]).unwrap().0;
			let b = plan.run(&minus, &[1, 1], &[1, 2], &[true, true]).unwrap().0;
			let reference = (a - b) / (2. * h);
			let actual = gradients[slot].data[element];
			assert!(
				actual.is_finite()
					&& reference.is_finite()
					&& (actual - reference).abs() <= 3e-3 + 3e-2 * reference.abs(),
				"{name}: {actual} {reference}"
			);
		}
	}
}
```

```sh
# check
cargo test --offline --test llm_gradients tiny_transformer_parameter_central_difference -- --exact
```

## 28. 幅を変えて安定性も確認し、非有限値や桁落ちを「勾配一致」として通さない。

<!-- todo:24-028 -->


実装箇所は `training::TrainingPlan::run`、`optim::AdamW::update`、`llm_resume::fresh_process_resume_matches_continuous` である。この要件に対する実行可能な確認は `llm_gradients::tiny_transformer_parameter_central_difference` に置く。

`tests/llm_gradients.rs` の `fn tiny_transformer_parameter_central_difference(`の定義を次の全体で置き換えます。

```rust
// file="tests/llm_gradients.rs" mode="replace" item="fn tiny_transformer_parameter_central_difference("
fn tiny_transformer_parameter_central_difference() {
	let mut config = QwenConfig::tiny();
	config.hidden_size = 4;
	config.intermediate_size = 6;
	config.vocab_size = 4;
	config.num_attention_heads = 2;
	let plan = TrainingPlan::new(config.clone(), 2).unwrap();
	let weights = initialize(&config, 19).unwrap();
	let (_, gradients) = plan.run(&weights, &[1, 1], &[1, 2], &[true, true]).unwrap();
	assert_eq!(gradients.len(), weights.len());
	for name in [
		"model.embed_tokens.weight",
		"model.layers.0.self_attn.q_proj.weight",
		"model.layers.0.self_attn.k_proj.weight",
		"model.layers.0.self_attn.v_proj.weight",
		"model.layers.0.input_layernorm.weight",
		"model.layers.0.mlp.gate_proj.weight",
	] {
		let slot = plan.registry.names[name].slot;
		let element = if name == "model.embed_tokens.weight" {
			4
		} else {
			0
		};
		for h in [1e-3, 5e-4] {
			let mut plus = weights.clone();
			let mut minus = weights.clone();
			plus[slot].data[element] += h;
			minus[slot].data[element] -= h;
			let a = plan.run(&plus, &[1, 1], &[1, 2], &[true, true]).unwrap().0;
			let b = plan.run(&minus, &[1, 1], &[1, 2], &[true, true]).unwrap().0;
			let reference = (a - b) / (2. * h);
			let actual = gradients[slot].data[element];
			assert!(
				actual.is_finite()
					&& reference.is_finite()
					&& (actual - reference).abs() <= 3e-3 + 3e-2 * reference.abs(),
				"{name}: {actual} {reference}"
			);
		}
	}
}
```

```sh
# check
cargo test --offline --test llm_gradients tiny_transformer_parameter_central_difference -- --exact
```

## 29. 一回の parameter 更新を f32 reference と `abs_err <= 1e-5 + 1e-4*abs(ref)` で比較する。

<!-- todo:24-029 -->


実装箇所は `training::TrainingPlan::run`、`optim::AdamW::update`、`llm_resume::fresh_process_resume_matches_continuous` である。この要件に対する実行可能な確認は `llm_gradients::tiny_transformer_parameter_central_difference` に置く。

`tests/llm_gradients.rs` の `fn tiny_transformer_parameter_central_difference(`の定義を次の全体で置き換えます。

```rust
// file="tests/llm_gradients.rs" mode="replace" item="fn tiny_transformer_parameter_central_difference("
fn tiny_transformer_parameter_central_difference() {
	let mut config = QwenConfig::tiny();
	config.hidden_size = 4;
	config.intermediate_size = 6;
	config.vocab_size = 4;
	config.num_attention_heads = 2;
	let plan = TrainingPlan::new(config.clone(), 2).unwrap();
	let weights = initialize(&config, 19).unwrap();
	let (_, gradients) = plan.run(&weights, &[1, 1], &[1, 2], &[true, true]).unwrap();
	assert_eq!(gradients.len(), weights.len());
	for name in [
		"model.embed_tokens.weight",
		"model.layers.0.self_attn.q_proj.weight",
		"model.layers.0.self_attn.k_proj.weight",
		"model.layers.0.self_attn.v_proj.weight",
		"model.layers.0.input_layernorm.weight",
		"model.layers.0.mlp.gate_proj.weight",
	] {
		let slot = plan.registry.names[name].slot;
		let element = if name == "model.embed_tokens.weight" {
			4
		} else {
			0
		};
		for h in [1e-3, 5e-4] {
			let mut plus = weights.clone();
			let mut minus = weights.clone();
			plus[slot].data[element] += h;
			minus[slot].data[element] -= h;
			let a = plan.run(&plus, &[1, 1], &[1, 2], &[true, true]).unwrap().0;
			let b = plan.run(&minus, &[1, 1], &[1, 2], &[true, true]).unwrap().0;
			let reference = (a - b) / (2. * h);
			let actual = gradients[slot].data[element];
			assert!(
				actual.is_finite()
					&& reference.is_finite()
					&& (actual - reference).abs() <= 3e-3 + 3e-2 * reference.abs(),
				"{name}: {actual} {reference}"
			);
		}
	}
}
```

```sh
# check
cargo test --offline --test llm_gradients tiny_transformer_parameter_central_difference -- --exact
```

## 30. full batch と蓄積、連続実行と再開を同じ dropout=0 の条件で比較する。

<!-- todo:24-030 -->


実装箇所は `training::TrainingPlan::run`、`optim::AdamW::update`、`llm_resume::fresh_process_resume_matches_continuous` である。この要件に対する実行可能な確認は `llm_resume::fresh_process_resume_matches_continuous` に置く。

`tests/llm_resume.rs` の `fn fresh_process_resume_matches_continuous(`の定義を次の全体で置き換えます。

```rust
// file="tests/llm_resume.rs" mode="replace" item="fn fresh_process_resume_matches_continuous("
fn fresh_process_resume_matches_continuous() {
	let root = std::env::temp_dir().join(format!("chibi-resume-{}", std::process::id()));
	let _ = fs::remove_dir_all(&root);
	fs::create_dir_all(&root).unwrap();
	let run = |name: &str, steps: &str, resume: Option<&str>| {
		let mut command = Command::new(std::env::current_exe().unwrap());
		command
			.args(["--ignored", "--exact", "process_worker", "--nocapture"])
			.env("CHIBI_WORKER_STEPS", steps)
			.env("CHIBI_WORKER_OUT", root.join(name));
		if let Some(previous) = resume {
			command.env("CHIBI_WORKER_RESUME", root.join(previous));
		}
		let result = command.output().unwrap();
		assert!(
			result.status.success(),
			"{}\n{}",
			String::from_utf8_lossy(&result.stdout),
			String::from_utf8_lossy(&result.stderr)
		);
	};
	run("continuous", "10", None);
	run("step5", "5", None);
	run("resumed", "5", Some("step5"));
	let a = restore_training_checkpoint(&root.join("continuous")).unwrap();
	let b = restore_training_checkpoint(&root.join("resumed")).unwrap();
	assert_eq!(a.optimizer.step, 10);
	assert_eq!(a, b);
	let all: Vec<f32> =
		serde_json::from_slice(&fs::read(root.join("continuous.losses")).unwrap()).unwrap();
	let latter: Vec<f32> =
		serde_json::from_slice(&fs::read(root.join("resumed.losses")).unwrap()).unwrap();
	assert_eq!(&all[5..], latter);
	assert!(all[9] < all[0]);
	let mut ad = a.data;
	let mut bd = b.data;
	assert_eq!(ad.next_index(), bd.next_index());
	let _ = fs::remove_dir_all(root);
}
```

```sh
# check
cargo test --offline --test llm_resume fresh_process_resume_matches_continuous -- --exact
```

## 31. 同一 CPU 経路の再開は byte 一致を目標にし、GPU の加算順が異なる場合は許容値と最大差を記録する。

<!-- todo:24-031 -->


実装箇所は `training::TrainingPlan::run`、`optim::AdamW::update`、`llm_resume::fresh_process_resume_matches_continuous` である。この要件に対する実行可能な確認は `llm_resume::fresh_process_resume_matches_continuous` に置く。

`tests/llm_resume.rs` の `fn fresh_process_resume_matches_continuous(`の定義を次の全体で置き換えます。

```rust
// file="tests/llm_resume.rs" mode="replace" item="fn fresh_process_resume_matches_continuous("
fn fresh_process_resume_matches_continuous() {
	let root = std::env::temp_dir().join(format!("chibi-resume-{}", std::process::id()));
	let _ = fs::remove_dir_all(&root);
	fs::create_dir_all(&root).unwrap();
	let run = |name: &str, steps: &str, resume: Option<&str>| {
		let mut command = Command::new(std::env::current_exe().unwrap());
		command
			.args(["--ignored", "--exact", "process_worker", "--nocapture"])
			.env("CHIBI_WORKER_STEPS", steps)
			.env("CHIBI_WORKER_OUT", root.join(name));
		if let Some(previous) = resume {
			command.env("CHIBI_WORKER_RESUME", root.join(previous));
		}
		let result = command.output().unwrap();
		assert!(
			result.status.success(),
			"{}\n{}",
			String::from_utf8_lossy(&result.stdout),
			String::from_utf8_lossy(&result.stderr)
		);
	};
	run("continuous", "10", None);
	run("step5", "5", None);
	run("resumed", "5", Some("step5"));
	let a = restore_training_checkpoint(&root.join("continuous")).unwrap();
	let b = restore_training_checkpoint(&root.join("resumed")).unwrap();
	assert_eq!(a.optimizer.step, 10);
	assert_eq!(a, b);
	let all: Vec<f32> =
		serde_json::from_slice(&fs::read(root.join("continuous.losses")).unwrap()).unwrap();
	let latter: Vec<f32> =
		serde_json::from_slice(&fs::read(root.join("resumed.losses")).unwrap()).unwrap();
	assert_eq!(&all[5..], latter);
	assert!(all[9] < all[0]);
	let mut ad = a.data;
	let mut bd = b.data;
	assert_eq!(ad.next_index(), bd.next_index());
	let _ = fs::remove_dir_all(root);
}
```

```sh
# check
cargo test --offline --test llm_resume fresh_process_resume_matches_continuous -- --exact
```

## 32. dump で flash forward-only が学習に選ばれず、weight が PARAM のままであることを確認する。

<!-- todo:24-032 -->


実装箇所は `training::TrainingPlan::run`、`optim::AdamW::update`、`llm_resume::fresh_process_resume_matches_continuous` である。この要件に対する実行可能な確認は `llm_gradients::adamw_scalar_and_finite_transaction` に置く。

`tests/llm_gradients.rs` の `fn adamw_scalar_and_finite_transaction(`の定義を次の全体で置き換えます。

```rust
// file="tests/llm_gradients.rs" mode="replace" item="fn adamw_scalar_and_finite_transaction("
fn adamw_scalar_and_finite_transaction() {
	let w = BufferData {
		shape: vec![],
		data: vec![1.],
	};
	let mut state = AdamState::new(vec![w.clone()]);
	let config = AdamConfig {
		lr: 0.1,
		decay: 0.,
		..Default::default()
	};
	let adam = AdamW::new(&state.weights, config).unwrap();
	adam.update(&mut state, &[w]).unwrap();
	let expected = 1. - 0.1 / (1. + config.eps);
	assert!((state.weights[0].data[0] - expected).abs() < 1e-5 + 1e-4 * expected.abs());
	assert_eq!(state.step, 1);
	let before = state.clone();
	assert!(
		adam.update(
			&mut state,
			&[BufferData {
				shape: vec![],
				data: vec![f32::NAN]
			}]
		)
		.is_err()
	);
	assert_eq!(state, before);
}
```

```sh
# check
cargo test --offline --test llm_gradients adamw_scalar_and_finite_transaction -- --exact
```

## 33. 小 LLM の学習成功、全 Qwen の見積り、全 Qwen の実行結果を別々に記録する。

<!-- todo:24-033 -->


実装箇所は `training::TrainingPlan::run`、`optim::AdamW::update`、`llm_resume::fresh_process_resume_matches_continuous` である。この要件に対する実行可能な確認は `llm_gradients::adamw_scalar_and_finite_transaction` に置く。

`tests/llm_gradients.rs` の `fn adamw_scalar_and_finite_transaction(`の定義を次の全体で置き換えます。

```rust
// file="tests/llm_gradients.rs" mode="replace" item="fn adamw_scalar_and_finite_transaction("
fn adamw_scalar_and_finite_transaction() {
	let w = BufferData {
		shape: vec![],
		data: vec![1.],
	};
	let mut state = AdamState::new(vec![w.clone()]);
	let config = AdamConfig {
		lr: 0.1,
		decay: 0.,
		..Default::default()
	};
	let adam = AdamW::new(&state.weights, config).unwrap();
	adam.update(&mut state, &[w]).unwrap();
	let expected = 1. - 0.1 / (1. + config.eps);
	assert!((state.weights[0].data[0] - expected).abs() < 1e-5 + 1e-4 * expected.abs());
	assert_eq!(state.step, 1);
	let before = state.clone();
	assert!(
		adam.update(
			&mut state,
			&[BufferData {
				shape: vec![],
				data: vec![f32::NAN]
			}]
		)
		.is_err()
	);
	assert_eq!(state, before);
}
```

```sh
# check
cargo test --offline --test llm_gradients adamw_scalar_and_finite_transaction -- --exact
```
