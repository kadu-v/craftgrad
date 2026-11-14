# 24. 全 parameter を学習し checkpoint から同じ更新を再開する

前の節で動かしたコードに、今回使う演算と処理を追加します。各節の確認を実行してから次へ進みます。

参照するUOpの構成は [tinygrad c22edff](https://github.com/tinygrad/tinygrad/blob/c22edff14c79310a9b2df62a78819cf77c8289fa/tinygrad/uop/ops.py) です。最初から全機能を実装せず、使う節で増やします。

## 24.01 AdamW と global norm clipping を実行する

weights と一次・二次 moment、step を AdamState に保持する。全パラメータの勾配から1つの norm を計算し、clipping、bias correction、decoupled decay を共通 Tensor graph にする。不正な更新候補は採用しない。

`src/optim.rs` の `use crate::`を置き換える。

```rust
// file="src/optim.rs" mode="replace" item="use crate::"
use crate::{Buffer, Executable, Result, Tensor, compile_many, run_named};
```

`src/optim.rs`を追記する。

```rust
// file="src/optim.rs" mode="append"

pub use crate::state::BufferData;

use serde::{Deserialize, Serialize};

use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AdamState {
    pub weights: Vec<BufferData>,
    pub m: Vec<BufferData>,
    pub v: Vec<BufferData>,
    pub step: u64,
}

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
                validate_finite(b)?;
            }
            if w.shape != m.shape || w.shape != v.shape {
                return Err("optimizer shape mismatch".into());
            }
        }
        Ok(())
    }
}

fn validate_finite(b: &BufferData) -> Result<()> {
    b.buffer()?;
    if b.data.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite optimizer state or gradient".into());
    }
    Ok(())
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
    plan: Executable,
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
        let scalar = Tensor::scalar;
        let gradients = weights
            .iter()
            .enumerate()
            .map(|(i, w)| Tensor::parameter(&format!("grad.{i}"), &w.shape))
            .collect::<Vec<_>>();
        let square_norm = gradients
            .iter()
            .fold(scalar(0.), |a, g| a.add(&g.mul(g).sum()));
        let norm = square_norm.sqrt().stage();
        let scale = if let Some(limit) = config.clip {
            let ratio = scalar(limit).div(&norm.add(&scalar(f32::MIN_POSITIVE)));
            ratio.lt(&scalar(1.)).where_(&ratio, &scalar(1.))
        } else {
            scalar(1.)
        };
        let c1 = Tensor::parameter("correction1", &[]);
        let c2 = Tensor::parameter("correction2", &[]);
        let mut outputs = vec![norm];
        for (i, w) in weights.iter().enumerate() {
            validate_finite(w)?;
            let weight = Tensor::parameter(&format!("weight.{i}"), &w.shape);
            let m = Tensor::parameter(&format!("m.{i}"), &w.shape);
            let v = Tensor::parameter(&format!("v.{i}"), &w.shape);
            let grad = gradients[i].mul(&scale);
            let next_m = m
                .mul(&scalar(config.beta1))
                .add(&grad.mul(&scalar(1. - config.beta1)))
                .stage();
            let next_v = v
                .mul(&scalar(config.beta2))
                .add(&grad.mul(&grad).mul(&scalar(1. - config.beta2)))
                .stage();
            let direction = next_m
                .div(&c1)
                .div(&next_v.div(&c2).sqrt().add(&scalar(config.eps)));
            let next_w = weight
                .mul(&scalar(1. - config.lr * config.decay))
                .sub(&direction.mul(&scalar(config.lr)));
            outputs.extend([next_w, next_m, next_v]);
        }
        Ok(Self {
            plan: compile_many(&outputs)?,
            count: weights.len(),
            config,
        })
    }
    pub fn update(&self, state: &mut AdamState, gradients: &[BufferData]) -> Result<f64> {
        state.validate()?;
        if gradients.len() != self.count || state.weights.len() != self.count {
            return Err("optimizer gradient keys".into());
        }
        let mut bindings = BTreeMap::new();
        for (i, g) in gradients.iter().enumerate() {
            validate_finite(g)?;
            if g.shape != state.weights[i].shape {
                return Err("gradient shape".into());
            }
            for (name, b) in [
                ("weight", &state.weights[i]),
                ("grad", g),
                ("m", &state.m[i]),
                ("v", &state.v[i]),
            ] {
                bindings.insert(format!("{name}.{i}"), b.buffer()?);
            }
        }
        let step = state.step.checked_add(1).ok_or("optimizer step overflow")?;
        bindings.insert(
            "correction1".into(),
            Buffer::new(vec![], vec![1. - self.config.beta1.powf(step as f32)])?,
        );
        bindings.insert(
            "correction2".into(),
            Buffer::new(vec![], vec![1. - self.config.beta2.powf(step as f32)])?,
        );
        let output = run_named(&self.plan, &bindings)?;
        let mut candidate = state.clone();
        for (i, three) in output[1..].chunks_exact(3).enumerate() {
            candidate.weights[i] = BufferData::from(&three[0]);
            candidate.m[i] = BufferData::from(&three[1]);
            candidate.v[i] = BufferData::from(&three[2]);
        }
        candidate.step = step;
        candidate.validate()?;
        let norm = output[0].data[0] as f64;
        if !norm.is_finite() {
            return Err("nonfinite global gradient norm".into());
        }
        *state = candidate;
        Ok(norm)
    }
}

pub fn sgd_step(weights: &mut [BufferData], gradients: &[BufferData], lr: f32) -> Result<()> {
    if weights.len() != gradients.len() || !lr.is_finite() || lr <= 0. {
        return Err("SGD configuration".into());
    }
    let mut bindings = BTreeMap::new();
    let mut outputs = Vec::new();
    for (i, (w, g)) in weights.iter().zip(gradients).enumerate() {
        validate_finite(w)?;
        validate_finite(g)?;
        if w.shape != g.shape {
            return Err("SGD gradient shape".into());
        }
        bindings.insert(format!("w.{i}"), w.buffer()?);
        bindings.insert(format!("g.{i}"), g.buffer()?);
        outputs.push(
            Tensor::parameter(&format!("w.{i}"), &w.shape)
                .sub(&Tensor::parameter(&format!("g.{i}"), &g.shape).mul(&Tensor::scalar(lr))),
        );
    }
    let candidate = run_named(&compile_many(&outputs)?, &bindings)?
        .iter()
        .map(BufferData::from)
        .collect::<Vec<_>>();
    for b in &candidate {
        validate_finite(b)?;
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
    let mut sums = first
        .0
        .iter()
        .map(|g| Tensor::constant(0., &g.shape))
        .collect::<Vec<_>>();
    let mut bindings = BTreeMap::new();
    for (batch_index, (batch, n)) in microbatches.iter().enumerate() {
        if batch.len() != sums.len() {
            return Err("accumulation keys".into());
        }
        for (i, g) in batch.iter().enumerate() {
            validate_finite(g)?;
            if g.shape != first.0[i].shape {
                return Err("accumulation shape".into());
            }
            let name = format!("batch.{batch_index}.{i}");
            bindings.insert(name.clone(), g.buffer()?);
            sums[i] = sums[i].add(
                &Tensor::parameter(&name, &g.shape).mul(&Tensor::scalar(*n as f32 / total as f32)),
            );
        }
    }
    Ok(run_named(&compile_many(&sums)?, &bindings)?
        .iter()
        .map(BufferData::from)
        .collect())
}
```

`tests/graph_sgd_compat.rs`を保存する。

```rust
// file="tests/graph_sgd_compat.rs" mode="write"
use chibigrad::{Buffer, Tensor, compile_many, optim::sgd};
#[test]
fn compiled_sgd_rebinds_updated_weights() {
    let w = Tensor::parameter("w", &[2]);
    let loss = w.mul(&w).sum();
    let g = loss.gradient(&w).unwrap();
    let next = sgd(&[w], &[g], 0.1).unwrap();
    let plan = compile_many(&[loss, next[0].clone()]).unwrap();
    let mut weight = Buffer::new(vec![2], vec![1., 2.]).unwrap();
    let mut previous = f32::INFINITY;
    for _ in 0..5 {
        let result = plan
            .run(&[weight], &std::collections::HashMap::new())
            .unwrap();
        assert!(result[0].data[0] < previous);
        previous = result[0].data[0];
        weight = result[1].clone();
    }
    assert!((weight.data[0] - 0.32768).abs() < 1e-5);
}
```

`tests/llm_optimizer.rs`を保存する。

```rust
// file="tests/llm_optimizer.rs" mode="write"
use chibigrad::optim::{AdamConfig,AdamState,AdamW,BufferData,accumulate_weighted};
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

```sh
# check step=24.01
cargo test --offline --test llm_optimizer --test graph_sgd_compat
```

1ステップの scalar AdamW が参照式に一致する。[3,4] の norm は5で moment は [0.06,0.08] となる。token 数重みの累積勾配は5となり、一括更新と一致する。

## 24.02 全パラメータの TrainingPlan を作る

tokens・targets・valid を実行時に束縛し、損失と Registry 内すべての parameter 勾配を1つの計画で返す。tied embedding を1回だけ数え、全 layer の score と optimizer 状態の保存量も計算する。

`src/lib.rs`を追記する。

```rust
// file="src/lib.rs" mode="append"

pub mod training;
```

`src/training.rs`を保存する。

```rust
// file="src/training.rs" mode="write"
use crate::{
    Buffer, Executable, Result, Tensor, compile_many,
    nn::{
        cross_entropy_runtime,
        qwen::{
            QwenConfig, logits_all, parameter_bindings, parameter_registry, register_parameters,
        },
    },
    optim::BufferData,
    run_named,
    state::Registry,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
pub struct TrainingPlan {
    pub plan: Executable,
    pub registry: Registry,
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
        let params = register_parameters(&registry)?;
        let tokens = Tensor::parameter("tokens", &[sequence]);
        let targets = Tensor::parameter("targets", &[sequence]);
        let valid = Tensor::parameter("valid", &[sequence]);
        let forward = logits_all(&config, &params, &tokens, 0, None)?;
        let loss = cross_entropy_runtime(&forward.logits, &targets, &valid)?;
        let mut by_id = std::collections::BTreeMap::new();
        for (name, spec) in &registry.names {
            by_id.entry(spec.id).or_insert(params[name].clone());
        }
        let gradients = loss.gradients(&by_id.into_values().collect::<Vec<_>>())?;
        if gradients.len() != registry.parameters().len() {
            return Err("incomplete parameter gradient set".into());
        }
        let mut outputs = vec![loss];
        outputs.extend(gradients);
        Ok(Self {
            plan: compile_many(&outputs)?,
            registry,
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
            || !valid.iter().any(|v| *v)
            || tokens
                .iter()
                .chain(targets)
                .any(|&v| v >= self.config.vocab_size)
        {
            return Err("training input/target/mask or parameter set".into());
        }
        let mut bindings = parameter_bindings(&self.registry, weights)?;
        for (name, values) in [
            ("tokens", tokens.iter().map(|&v| v as f32).collect()),
            ("targets", targets.iter().map(|&v| v as f32).collect()),
            (
                "valid",
                valid.iter().map(|&v| if v { 1. } else { 0. }).collect(),
            ),
        ] {
            bindings.insert(name.into(), Buffer::new(vec![self.sequence], values)?);
        }
        let outputs = run_named(&self.plan, &bindings)?;
        let loss = outputs[0].data[0];
        let grads = outputs[1..]
            .iter()
            .map(BufferData::from)
            .collect::<Vec<_>>();
        if !loss.is_finite() || grads.iter().flat_map(|g| &g.data).any(|v| !v.is_finite()) {
            return Err("nonfinite training loss or gradient".into());
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
    c.validate()?;
    if batch == 0 || t == 0 || t > c.max_position_embeddings as u64 {
        return Err("training estimate batch/context capacity".into());
    }
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

`tests/llm_training.rs`を保存する。

```rust
// file="tests/llm_training.rs" mode="write"
use chibigrad::{Buffer,Tensor,compile_many,nn::{cross_entropy,qwen::QwenConfig},training::{TrainingPlan,estimate_training_storage,initialize}};
#[test]
fn uniform_logits_and_targets() {
    let logits = Tensor::parameter("logits", &[2, 4]);
    let loss = cross_entropy(&logits, &[1, 2], &[true, true]).unwrap();
    let grads = loss.gradients(&[logits]).unwrap();
    let mut out = vec![loss];
    out.extend(grads);
    let values = compile_many(&out)
        .unwrap()
        .run(
            &[Buffer::new(vec![2, 4], vec![1000.; 8]).unwrap()],
            &Default::default(),
        )
        .unwrap();
    assert!((values[0].data[0] - 4f32.ln()).abs() < 1e-6);
    let reference = [0.125, -0.375, 0.125, 0.125, 0.125, 0.125, -0.375, 0.125];
    for (a, b) in values[1].data.iter().zip(reference) {
        assert!((a - b).abs() < 1e-6);
    }
}
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

`tests/model_contracts.rs`を保存する。

```rust
// file="tests/model_contracts.rs" mode="write"
use chibigrad::{
    Buffer,
    nn::qwen::{
        KvState, QwenConfig, greedy, initialize, load_parameters, parameter_bindings,
        parameter_registry, token_buffer,
    },
    state::TensorRecord,
    training::{TrainingPlan, estimate_training_storage},
};
#[test]
fn rejects_bad_model_and_binding_inputs_before_execution() {
    let c = QwenConfig::tiny();
    let registry = parameter_registry(&c).unwrap();
    let mut invalid = c.clone();
    invalid.num_attention_heads = 0;
    assert!(KvState::new(&invalid).is_err());
    assert!(parameter_registry(&invalid).is_err());
    assert!(TrainingPlan::new(c.clone(), 0).is_err());
    assert!(estimate_training_storage(&registry, 0, 1, &c).is_err());
    assert!(estimate_training_storage(&registry, 1, 0, &c).is_err());
    assert!(
        estimate_training_storage(&registry, 1, c.max_position_embeddings as u64 + 1, &c).is_err()
    );
    assert!(token_buffer(&[], &c).is_err());
    assert!(token_buffer(&[c.vocab_size as u32], &c).is_err());
    assert!(greedy(&Buffer::new(vec![1, 1], vec![f32::NAN]).unwrap()).is_err());
    let mut weights = initialize(&c, 1).unwrap();
    weights[0].shape = vec![weights[0].data.len()];
    assert!(parameter_bindings(&registry, &weights).is_err());
}
#[test]
fn tied_weight_load_and_cache_adoption_are_transactional() {
    let c = QwenConfig::tiny();
    let registry = parameter_registry(&c).unwrap();
    let weights = initialize(&c, 11).unwrap();
    let mut records = registry
        .names
        .iter()
        .map(|(name, spec)| {
            (
                name.clone(),
                TensorRecord::from_buffer(&weights[spec.slot].buffer().unwrap()),
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    let embedding = records["model.embed_tokens.weight"].clone();
    records.insert("lm_head.weight".into(), embedding);
    assert_eq!(load_parameters(&c, &records).unwrap().len(), weights.len());
    records.get_mut("lm_head.weight").unwrap().payload[0] ^= 1;
    assert!(load_parameters(&c, &records).unwrap_err().contains("tied"));
    let mut cache = KvState::new(&c).unwrap();
    let before = cache.keys.clone();
    assert!(
        cache
            .adopt(&vec![Buffer::new(vec![], vec![0.]).unwrap(); 2], 1)
            .is_err()
    );
    assert_eq!(cache.valid, 0);
    assert_eq!(cache.keys, before);
    assert!(
        cache
            .adopt(
                &[cache.keys[0].clone(), cache.values[0].clone()],
                c.max_position_embeddings + 1
            )
            .is_err()
    );
    assert_eq!(cache.valid, 0);
    assert_eq!(cache.keys, before);
}
```

```sh
# check step=24.02
cargo test --offline --test llm_training --test model_contracts
```

uniform logits の損失と勾配が解析値に一致する。全 weight の勾配を返し、embedding、Q/K/V、norm、MLP の中心差分が一致する。

## 24.03 checkpoint と別 process での再開を検証する

TrainingState に optimizer、scheduler step、data cursor、RNG、shuffle 順、dataset/config/tokenizer digest をまとめる。step 境界だけを一時 generation に保存し、readback と digest 検証後に公開する。別 process の再開で連続実行と完全一致を確認する。

`examples/train_qwen.rs`を保存する。

```rust
// file="examples/train_qwen.rs" mode="write"
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

`examples/train_tiny_llm.rs`を保存する。

```rust
// file="examples/train_tiny_llm.rs" mode="write"
use chibigrad::{
    Result,
    checkpoint::{
        DataCursor, TrainingState, adopt_checkpoint, restore_training_checkpoint,
        save_training_checkpoint,
    },
    fetch::sha256,
    nn::qwen::QwenConfig,
    optim::{AdamConfig, AdamState, AdamW, sgd_step},
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
            sgd_step(&mut state.optimizer.weights, &grads, config.lr)?;
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
        serde_json::json!({"final_step":state.optimizer.step,"state_sha":sha256(&serde_json::to_vec(&state).unwrap())})
    );
    Ok(())
}
```

`src/checkpoint.rs`を保存する。

```rust
// file="src/checkpoint.rs" mode="write"
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

`src/lib.rs`を追記する。

```rust
// file="src/lib.rs" mode="append"

pub mod checkpoint;
```

`tests/llm_resume.rs`を保存する。

```rust
// file="tests/llm_resume.rs" mode="write"
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

```sh
# check step=24.03
cargo test --offline --test llm_resume
cargo run --offline --example train_tiny_llm -- --steps 3 --seed 1
cargo run --offline --example train_qwen -- --estimate-only
```

10ステップ連続実行と5ステップ保存後に別 process で5ステップ実行した全状態、損失列、次の data index が完全一致する。破損や identity 不一致は状態交換前に拒否する。

## 章末の累積確認

前の章までのテストとexampleも再ビルドします。

```sh
# check
cargo test --quiet --all-targets
```

## 要件との対応

### 24-001 全位置の logits を返す学習入口を追加し、KV cache と sampling を使わない。

<!-- todo:24-001 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-002 target の範囲、有効位置、安定した cross entropy を検査する。

<!-- todo:24-002 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-003 embedding/gather の backward を追加し、繰り返し token の寄与を加算する。

<!-- todo:24-003 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-004 最初は各語彙行が全 token 位置を走査する reduction で実装し、GPU の書込み競合を避ける。

<!-- todo:24-004 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-005 RMSNorm、SiLU、softmax、RoPE、GQA、tied weight の勾配を13の規則と合成して検証する。

<!-- todo:24-005 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-006 token を一つずらした target を作り、入力と target の位置を dump する。

<!-- todo:24-006 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-007 forward、loss、gradient、SGD 更新を一回の step として既存 runtime で動かす。

<!-- todo:24-007 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-008 step の先頭で gradient を初期化し、前回分を意図せず加えない。

<!-- todo:24-008 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-009 Plan の入力 slot に更新後の parameter を渡し、次の step で新しい値を読む。

<!-- todo:24-009 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-010 CPU の loss と更新を scalar reference に比較し、その後 Metal と OpenCL でも小例を比較する。

<!-- todo:24-010 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-011 m、v、step、bias correction、weight decay を持つ f32 AdamW を追加する。

<!-- todo:24-011 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-012 scalar parameter の一回の更新を数式の reference に一致させる。

<!-- todo:24-012 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-013 gradient norm と非有限値を検査し、必要なら設定された閾値で clipping する。

<!-- todo:24-013 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-014 microbatch ごとの loss の平均ではなく、有効 token 数に応じて gradient を重み付けする。

<!-- todo:24-014 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-015 full batch と勾配蓄積した更新を、同じデータ順と dropout=0 で比較する。

<!-- todo:24-015 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-016 model state、m、v、step、学習率 scheduler、乱数 generator の状態を保存する。

<!-- todo:24-016 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-017 dataset の digest、tokenizer/config の識別、epoch、次の読出し位置、shuffle 状態も保存する。

<!-- todo:24-017 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-018 蓄積中間の checkpoint は最初は拒否し、optimizer step 完了後の境界だけで保存する。

<!-- todo:24-018 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-019 device の完了待ち後に一時ディレクトリへ書き、全ファイルと manifest の検証後に確定する。

<!-- todo:24-019 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-020 新しいプロセスで検証と load を行い、連続10 step と5 step後の保存・再開を比較する。

<!-- todo:24-020 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-021 22の manifest を読み、BF16 checkpoint を f32 parameter として準備する。

<!-- todo:24-021 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-022 全 layer と語彙を保ったまま、短い sequence と microbatch=1でメモリ見積りを出す。

<!-- todo:24-022 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-023 実際の一意 parameter 数Pを数え、f32 の重み・勾配・m・vだけで `16*P` byte を要することを記録する。

<!-- todo:24-023 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-024 activation、logits、通常 attention の score、再利用 workspace の peak を別に加える。

<!-- todo:24-024 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-025 小さな検証済みデータで一回の forward、backward、全 parameter 更新、保存、再開を順に実行する。

<!-- todo:24-025 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-026 repeated token の期待勾配と uniform logits の `ln(4)` を確認する。

<!-- todo:24-026 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-027 小 graph の有限差分は中央差分、初期幅 `1e-3`、基準 `abs_err <= 3e-3 + 3e-2*abs(ref)` で検査する。

<!-- todo:24-027 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-028 幅を変えて安定性も確認し、非有限値や桁落ちを「勾配一致」として通さない。

<!-- todo:24-028 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-029 一回の parameter 更新を f32 reference と `abs_err <= 1e-5 + 1e-4*abs(ref)` で比較する。

<!-- todo:24-029 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-030 full batch と蓄積、連続実行と再開を同じ dropout=0 の条件で比較する。

<!-- todo:24-030 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-031 同一 CPU 経路の再開は byte 一致を目標にし、GPU の加算順が異なる場合は許容値と最大差を記録する。

<!-- todo:24-031 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-032 dump で flash forward-only が学習に選ばれず、weight が PARAM のままであることを確認する。

<!-- todo:24-032 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 24-033 小 LLM の学習成功、全 Qwen の見積り、全 Qwen の実行結果を別々に記録する。

<!-- todo:24-033 -->

関連する実装と確認は本章の `24.01`, `24.02`, `24.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。
