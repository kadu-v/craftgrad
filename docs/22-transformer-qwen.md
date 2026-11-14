# 22. Qwen の全層、tokenizer、KV cache を接続する

前の節で動かしたコードに、今回使う演算と処理を追加します。各節の確認を実行してから次へ進みます。

参照するUOpの構成は [tinygrad c22edff](https://github.com/tinygrad/tinygrad/blob/c22edff14c79310a9b2df62a78819cf77c8289fa/tinygrad/uop/ops.py) です。最初から全機能を実装せず、使う節で増やします。

## 22.01 Qwen の設定と全パラメータを登録する

QwenConfig を検証し、各 layer の Q/K/V bias と重み、MLP、norm をすべて Registry へ登録する。tied embedding は1つの ParamId とし、別名の payload 不一致を拒否する。推論グラフは次節で組み立てる。

`fixtures/qwen25/config.json`を保存する。

```jsonc
// file="fixtures/qwen25/config.json" mode="write"
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

`src/nn.rs`を削除する。

```rust
// file="src/nn.rs" mode="delete"

```

`src/nn/mod.rs`を保存する。

```rust
// file="src/nn/mod.rs" mode="write"
use crate::{Result, Tensor};

pub fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let y = x.matmul(&weight.permute(&[1, 0]));
    let y = bias.map_or(y.clone(), |b| y.add(b));
    y.uop.shape()?;
    Ok(y)
}
pub fn relu(x: &Tensor) -> Tensor {
    Tensor::scalar(0.).lt(x).where_(x, &Tensor::scalar(0.))
}

pub fn log_softmax(x: &Tensor, axis: usize) -> Result<Tensor> {
    let shifted = x.sub(&x.max(&[axis], true).detach());
    let result = shifted.sub(&shifted.exp().sum_axes(&[axis], true).log());
    result.uop.shape()?;
    Ok(result)
}
pub fn mse(prediction: &Tensor, target: &Tensor) -> Result<Tensor> {
    let difference = prediction.sub(target);
    let n = crate::uop::checked_numel(&difference.uop.shape()?)?;
    if n == 0 {
        return Err("mean of an empty tensor".into());
    }
    Ok(difference
        .mul(&difference)
        .sum()
        .div(&Tensor::scalar(n as f32)))
}
pub fn cross_entropy_runtime(logits: &Tensor, targets: &Tensor, valid: &Tensor) -> Result<Tensor> {
    let shape = logits.uop.shape()?;
    if shape.len() != 2 || targets.uop.shape()? != [shape[0]] || valid.uop.shape()? != [shape[0]] {
        return Err("runtime cross entropy shape".into());
    }
    let n = shape[0];
    let total = n.checked_mul(shape[1]).ok_or("logit size overflow")?;
    if total > 16_777_216 {
        return Err("flattened token indices exceed exact f32 integer range".into());
    }
    let offsets = Tensor::from_slice(
        &(0..n).map(|i| (i * shape[1]) as f32).collect::<Vec<_>>(),
        &[n],
    );
    let indices = targets.add(&offsets);
    let selected = log_softmax(logits, 1)?
        .reshape(&[total])
        .gather(&indices, 0);
    Ok(selected.mul(valid).sum().div(&valid.sum()).neg())
}
pub fn cross_entropy(logits: &Tensor, targets: &[usize], valid: &[bool]) -> Result<Tensor> {
    let shape = logits.uop.shape()?;
    if shape.len() != 2
        || targets.len() != shape[0]
        || valid.len() != targets.len()
        || !valid.iter().any(|v| *v)
        || targets.iter().any(|&v| v >= shape[1])
    {
        return Err("cross entropy target/mask".into());
    }
    cross_entropy_runtime(
        logits,
        &Tensor::from_slice(
            &targets.iter().map(|&v| v as f32).collect::<Vec<_>>(),
            &[targets.len()],
        ),
        &Tensor::from_slice(
            &valid
                .iter()
                .map(|&v| if v { 1. } else { 0. })
                .collect::<Vec<_>>(),
            &[valid.len()],
        ),
    )
}


pub mod qwen;
```

`src/nn/qwen.rs`を保存する。

```rust
// file="src/nn/qwen.rs" mode="write"
use crate::{Buffer,Result,Tensor,state::{DType,Registry,Spec,TensorRecord,TensorSet}};
use serde::{Deserialize,Serialize};
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
pub fn register_parameters(registry: &Registry) -> Result<BTreeMap<String, Tensor>> {
    registry
        .names
        .iter()
        .map(|(name, spec)| Ok((name.clone(), Tensor::parameter(name, &spec.shape))))
        .collect()
}
pub fn parameter_bindings(
    registry: &Registry,
    weights: &[crate::state::BufferData],
) -> Result<BTreeMap<String, Buffer>> {
    if weights.len() != registry.parameters().len() {
        return Err("parameter count".into());
    }
    registry
        .names
        .iter()
        .map(|(name, spec)| {
            let buffer = weights.get(spec.slot).ok_or("parameter slot")?.buffer()?;
            if buffer.shape != spec.shape {
                return Err(format!("parameter shape {name}"));
            }
            Ok((name.clone(), buffer))
        })
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

`tests/qwen_registry.rs`を保存する。

```rust
// file="tests/qwen_registry.rs" mode="write"
use chibigrad::nn::qwen::{QwenConfig,parameter_registry,initialize,load_parameters};
use chibigrad::state::TensorRecord;
#[test]
fn tiny_registration_and_tied_payload() {
 let c=QwenConfig::tiny();let r=parameter_registry(&c).unwrap();
 assert_eq!(r.names.len(),14);assert!(!r.names.contains_key("lm_head.weight"));
 let w=initialize(&c,11).unwrap();assert_eq!(w.len(),r.parameters().len());
 let mut records=r.names.iter().map(|(n,s)|(n.clone(),TensorRecord::from_buffer(&w[s.slot].buffer().unwrap()))).collect::<std::collections::BTreeMap<_,_>>();
 records.insert("lm_head.weight".into(),records["model.embed_tokens.weight"].clone());
 assert_eq!(load_parameters(&c,&records).unwrap().len(),w.len());
 records.get_mut("lm_head.weight").unwrap().payload[0]^=1;
 assert!(load_parameters(&c,&records).unwrap_err().contains("tied"));
 let mut bad=c;bad.num_attention_heads=0;assert!(parameter_registry(&bad).is_err());
}
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

```sh
# check step=22.01
cargo test --offline --test qwen_registry
```

tiny は14項目、公開設定は290項目を登録する。24 layer の bias を含み、tied weight を重複計上しない。

## 22.02 通常の GQA と RoPE を組み立てる

embedding、RMSNorm、RoPE、causal mask、通常 softmax attention を Tensor の既存演算へ分解する。head と KV head の比率を検証する。flash 実装は含めない。

`src/nn/attention.rs`を保存する。

```rust
// file="src/nn/attention.rs" mode="write"
use crate::{Result, Tensor};

pub fn embedding(weight: &Tensor, tokens: &Tensor) -> Tensor {
    weight.gather(tokens, 0)
}
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let shape = x.uop.shape()?;
    let dim = *shape.last().ok_or("RMSNorm requires a final dimension")?;
    if dim == 0 || weight.uop.shape()? != [dim] || eps <= 0. || !eps.is_finite() {
        return Err("RMSNorm shape/epsilon".into());
    }
    let mean = x
        .mul(x)
        .sum_axes(&[shape.len() - 1], true)
        .div(&Tensor::scalar(dim as f32));
    Ok(x.div(&mean.add(&Tensor::scalar(eps)).sqrt()).mul(weight))
}
pub fn softmax(x: &Tensor, axis: usize) -> Result<Tensor> {
    Ok(super::log_softmax(x, axis)?.exp())
}
pub fn apply_rope(x: &Tensor, start: usize, theta: f64) -> Result<Tensor> {
    let shape = x.uop.shape()?;
    if shape.len() != 3 || shape[2] == 0 || shape[2] % 2 != 0 || theta <= 0. || !theta.is_finite() {
        return Err("RoPE shape/theta".into());
    }
    let (t, h, d) = (shape[0], shape[1], shape[2]);
    let half = d / 2;
    let mut cos = Vec::with_capacity(t * d);
    let mut sin = Vec::with_capacity(t * d);
    for pos in 0..t {
        let absolute = start.checked_add(pos).ok_or("RoPE position overflow")?;
        for channel in 0..d {
            let angle = absolute as f64 / theta.powf(2. * (channel % half) as f64 / d as f64);
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    let cos = Tensor::from_slice(&cos, &[t, 1, d]).expand(&[t, h, d]);
    let sin = Tensor::from_slice(&sin, &[t, 1, d]).expand(&[t, h, d]);
    let first = x.shrink(&[(0, t), (0, h), (0, half)]);
    let second = x.shrink(&[(0, t), (0, h), (half, d)]);
    let rotated = concatenate(&second.neg(), &first, 2)?;
    Ok(x.mul(&cos).add(&rotated.mul(&sin)))
}
pub fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    query_start: usize,
    causal: bool,
) -> Result<Tensor> {
    let qs = q.uop.shape()?;
    let ks = k.uop.shape()?;
    if qs.len() != 3
        || ks.len() != 3
        || v.uop.shape()? != ks
        || qs[2] != ks[2]
        || ks[1] == 0
        || qs[1] % ks[1] != 0
        || qs[2] == 0
        || ks[0] == 0
    {
        return Err("attention shape or GQA ratio".into());
    }
    let (t, h, d, s, hkv) = (qs[0], qs[1], qs[2], ks[0], ks[1]);
    let mut mask = Vec::with_capacity(t * s);
    for row in 0..t {
        let pos = query_start
            .checked_add(row)
            .ok_or("query position overflow")?;
        for col in 0..s {
            mask.push(if !causal || col <= pos {
                0.
            } else {
                f32::NEG_INFINITY
            });
        }
    }
    let mask = Tensor::from_slice(&mask, &[t, s]);
    let mut heads = Vec::new();
    for head in 0..h {
        let kh = head / (h / hkv);
        let qq = q
            .shrink(&[(0, t), (head, head + 1), (0, d)])
            .reshape(&[t, d]);
        let kk = k.shrink(&[(0, s), (kh, kh + 1), (0, d)]).reshape(&[s, d]);
        let vv = v.shrink(&[(0, s), (kh, kh + 1), (0, d)]).reshape(&[s, d]);
        let scores = qq
            .matmul(&kk.permute(&[1, 0]))
            .mul(&Tensor::scalar(1. / (d as f32).sqrt()))
            .add(&mask);
        heads.push(softmax(&scores, 1)?.matmul(&vv));
    }
    Ok(Tensor::stack(&heads, 1))
}
pub fn concatenate(a: &Tensor, b: &Tensor, axis: usize) -> Result<Tensor> {
    let sa = a.uop.shape()?;
    let sb = b.uop.shape()?;
    if sa.len() != sb.len()
        || axis >= sa.len()
        || sa
            .iter()
            .zip(&sb)
            .enumerate()
            .any(|(i, (x, y))| i != axis && x != y)
    {
        return Err("concat shape".into());
    }
    let mut pa = vec![(0, 0); sa.len()];
    let mut pb = pa.clone();
    pa[axis] = (0, sb[axis]);
    pb[axis] = (sa[axis], 0);
    Ok(a.pad(&pa).add(&b.pad(&pb)))
}
```

`src/nn/mod.rs`を追記する。

```rust
// file="src/nn/mod.rs" mode="append"

pub mod attention;



pub fn silu(x: &Tensor) -> Tensor {
    x.div(&Tensor::scalar(1.).add(&x.neg().exp()))
}
```

`tests/qwen_attention.rs`を保存する。

```rust
// file="tests/qwen_attention.rs" mode="write"
use chibigrad::{Tensor,compile_many,nn::attention::{attention,apply_rope,softmax}};
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
#[test]
fn causal_rope_and_softmax() {
    let q = Tensor::constant(0., &[2, 2, 2]);
    let k = Tensor::constant(0., &[2, 1, 2]);
    let v = Tensor::from_slice(&[2., 4., 6., 8.], &[2, 1, 2]);
    let y = attention(&q, &k, &v, 0, true).unwrap();
    let rope = apply_rope(&v, 0, 10000.).unwrap();
    let probabilities = softmax(&Tensor::constant(0., &[2, 2]), 1).unwrap();
    let out = compile_many(&[y, rope, probabilities])
        .unwrap()
        .run(&[], &Default::default())
        .unwrap();
    close(&out[0].data, &[2., 4., 2., 4., 4., 6., 4., 6.], 1e-5, 1e-4);
    close(&out[1].data[..2], &[2., 4.], 0., 0.);
    close(&out[2].data, &[0.5; 4], 1e-6, 0.);
}
```

```sh
# check step=22.02
cargo test --offline --test qwen_attention
```

GQA の各 head が正しい value を参照し、causal 出力とゼロ logits の softmax が参照値に一致する。

## 22.03 通常 attention で全 layer を実行する

登録済みパラメータから attention と SwiGLU、残差、最終 norm、tied 出力射影を組む。各 layer 境界を比較し、ゼロ重みだけでなく非ゼロの独立 f64 参照で全経路を検証する。

`fixtures/tiny-transformer.json`を保存する。

```jsonc
// file="fixtures/tiny-transformer.json" mode="write"
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

`src/nn/qwen.rs`を追記する。

```rust
// file="src/nn/qwen.rs" mode="append"

use super::{attention::{apply_rope,attention,rms_norm},linear,silu};

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

#[derive(Debug)]
pub struct Forward {pub logits:Tensor,pub boundaries:Vec<Tensor>}

pub fn logits_all(c:&QwenConfig,params:&BTreeMap<String,Tensor>,tokens:&Tensor)->Result<Forward>{
 let start=0usize;
    c.validate()?;
    let shape = tokens.uop.shape()?;
    if shape.len() != 1 || shape[0] == 0 {
        return Err("tokens must be nonempty rank one".into());
    }
    let t = shape[0];
    if start.checked_add(t).ok_or("context overflow")? > c.max_position_embeddings {
        return Err("input exceeds context capacity".into());
    }
    let get = |name: &str| {
        params
            .get(name)
            .ok_or_else(|| format!("missing parameter {name}"))
    };
    let d = c.hidden_size;
    let hd = d / c.num_attention_heads;
    let kd = c.num_key_value_heads;
    let mut x = super::attention::embedding(get("model.embed_tokens.weight")?, tokens);
    let mut boundaries = vec![x.clone()];
    for layer in 0..c.num_hidden_layers {
        let p = format!("model.layers.{layer}");
        let n = rms_norm(
            &x,
            get(&format!("{p}.input_layernorm.weight"))?,
            c.rms_norm_eps,
        )?;
        let projection = |name: &str, heads: usize| -> Result<Tensor> {
            Ok(linear(
                &n,
                get(&format!("{p}.self_attn.{name}.weight"))?,
                Some(get(&format!("{p}.self_attn.{name}.bias"))?),
            )?
            .reshape(&[t, heads, hd]))
        };
        let q = apply_rope(
            &projection("q_proj", c.num_attention_heads)?,
            start,
            c.rope_theta,
        )?;
        let k = apply_rope(&projection("k_proj", kd)?, start, c.rope_theta)?;
        let v = projection("v_proj", kd)?;
        let attended = attention(&q, &k, &v, start, true)?.reshape(&[t, d]);
        x = x.add(&linear(
            &attended,
            get(&format!("{p}.self_attn.o_proj.weight"))?,
            None,
        )?);
        let n = rms_norm(
            &x,
            get(&format!("{p}.post_attention_layernorm.weight"))?,
            c.rms_norm_eps,
        )?;
        let gate = silu(&linear(
            &n,
            get(&format!("{p}.mlp.gate_proj.weight"))?,
            None,
        )?);
        let up = linear(&n, get(&format!("{p}.mlp.up_proj.weight"))?, None)?;
        let down = linear(
            &gate.mul(&up),
            get(&format!("{p}.mlp.down_proj.weight"))?,
            None,
        )?;
        x = x.add(&down).stage();
        boundaries.push(x.clone());
    }
    let x = rms_norm(&x, get("model.norm.weight")?, c.rms_norm_eps)?;
    boundaries.push(x.clone());
    let output = get(if c.tie_word_embeddings {
        "model.embed_tokens.weight"
    } else {
        "lm_head.weight"
    })?;
    Ok(Forward {
        logits: linear(&x, output, None)?,
        boundaries,
    })
}

pub fn infer(c:&QwenConfig,weights:&[crate::state::BufferData],tokens:&[u32])->Result<Buffer>{
 let registry=parameter_registry(c)?;let params=register_parameters(&registry)?;
 let mut bindings=parameter_bindings(&registry,weights)?;
 bindings.insert("tokens".into(),token_buffer(tokens,c)?);
 let forward=logits_all(c,&params,&Tensor::parameter("tokens",&[tokens.len()]))?;
 Ok(crate::run_named(&crate::compile_many(&[forward.logits])?,&bindings)?[0].clone())
}
```

`tests/qwen_forward.rs`を保存する。

```rust
// file="tests/qwen_forward.rs" mode="write"
use chibigrad::{Buffer,Tensor,compile_many,run_named,nn::qwen::{QwenConfig,initialize,logits_all,parameter_registry,register_parameters,parameter_bindings},state::BufferData};
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
fn run(c:&QwenConfig,weights:&[BufferData],tokens:&[usize])->Vec<Buffer>{
 let reg=parameter_registry(c).unwrap();let params=register_parameters(&reg).unwrap();
 let mut bindings=parameter_bindings(&reg,weights).unwrap();
 bindings.insert("tokens".into(),Buffer::new(vec![tokens.len()],tokens.iter().map(|&t|t as f32).collect()).unwrap());
 let f=logits_all(c,&params,&Tensor::parameter("tokens",&[tokens.len()])).unwrap();
 let mut out=vec![f.logits];out.extend(f.boundaries);
 run_named(&compile_many(&out).unwrap(),&bindings).unwrap()
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
                            .sum::<f64>()
                            / (hd as f64).sqrt()
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
#[test]
fn zero_transformer() {
    let c = QwenConfig::tiny();
    let mut weights = initialize(&c, 1).unwrap();
    for w in &mut weights {
        w.data.fill(0.);
    }
    let values = run(&c, &weights, &[1, 2, 3]);
    assert!(values[0].data.iter().all(|&x| x == 0.));
    assert_eq!(values[0].shape, [3, 16]);
}
#[test]
fn nonzero_layer_boundaries_reference() {
    let c = QwenConfig::tiny();
    let weights = initialize(&c, 11).unwrap();
    let actual = run(&c, &weights, &[1, 2, 3]);
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

```sh
# check step=22.03
cargo test --offline --test qwen_forward
```

非ゼロ tiny モデルの logits と各 layer 境界が独立 f64 参照および保存済み fixture に許容誤差内で一致する。

## 22.04 検証済み assets と tokenizer を接続する

検証済み Manifest の config、safetensors、tokenizer と EOS をまとめて読む。WordLevel の局所 fixture で tokenizer 境界を実行する。公開 tokenizer と全24 layer の照合は外部 assets がある場合のみ実行する。

`Cargo.toml`を追記する。

```toml
# file="Cargo.toml" mode="append"

tokenizers={version="=0.22.2",default-features=false,features=["fancy-regex"]}
```

`examples/fetch_qwen.rs`を保存する。

```rust
// file="examples/fetch_qwen.rs" mode="write"
use chibigrad::{
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

`examples/qwen_tokenize.rs`を保存する。

```rust
// file="examples/qwen_tokenize.rs" mode="write"
use chibigrad::{
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

`fixtures/qwen25/manifest.json`を保存する。

```jsonc
// file="fixtures/qwen25/manifest.json" mode="write"
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

`fixtures/qwen25/tokenizer-fixture.json`を保存する。

```jsonc
// file="fixtures/qwen25/tokenizer-fixture.json" mode="write"
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

`src/lib.rs`を追記する。

```rust
// file="src/lib.rs" mode="append"

pub mod model_assets;
```

`src/model_assets.rs`を保存する。

```rust
// file="src/model_assets.rs" mode="write"
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

`tests/qwen_tokenizer.rs`を保存する。

```rust
// file="tests/qwen_tokenizer.rs" mode="write"
use chibigrad::{
    fetch::{Manifest, verified_read},
    model_assets::encode,
};
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
#[test]
fn tokenizer_boundary_without_external_assets() {
    let tokenizer = tokenizers::Tokenizer::from_bytes(br#"{
        "version":"1.0","truncation":null,"padding":null,"added_tokens":[],
        "normalizer":null,"pre_tokenizer":{"type":"Whitespace"},
        "post_processor":null,"decoder":null,
        "model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1,"world":2,"!":3},"unk_token":"[UNK]"}
    }"#).unwrap();
    assert_eq!(encode(&tokenizer, "hello world!").unwrap(), [1, 2, 3]);
    assert_eq!(encode(&tokenizer, "unknown").unwrap(), [0]);
    assert!(encode(&tokenizer, "").unwrap().is_empty());
}
```

```sh
# check step=22.04
cargo test --offline --test qwen_tokenizer tokenizer_boundary_without_external_assets -- --exact
```

hello world! は [1,2,3]、未知語は [0]、空文字列は空 token 列になる。公開 tokenizer の検証は assets 不在時 NOT_RUN とする。

## 22.05 KV 状態を不変入力と更新候補に分ける

固定容量の KV buffer と valid 長を分ける。forward は更新候補を返し、adopt が全 shape と容量を検証してから交換する。prefix 全再計算と分割 decode の logits を比較し、reset 後も一致を確認する。 公開重みとの比較は、fetch_qwenで取得後、tools/qwen_reference.pyで参照値を作って実行する。通常の確認では外部データの検査を実行済みと扱わない。

`examples/qwen_infer.rs`を保存する。

```rust
// file="examples/qwen_infer.rs" mode="write"
use chibigrad::{
    Result, model_assets,
    nn::qwen::{KvState, greedy, infer},
    state::BufferData,
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
    if c.max_position_embeddings == 0
        || c.max_position_embeddings > assets.config.max_position_embeddings
    {
        return Err("requested context capacity".into());
    }
    let count = arg("--max-new-tokens")
        .unwrap_or("1".into())
        .parse::<usize>()
        .map_err(|e| e.to_string())?;
    let mut input =
        model_assets::encode(&assets.tokenizer, &arg("--text").unwrap_or("Hello".into()))?;
    if input
        .len()
        .checked_add(count)
        .ok_or("token capacity overflow")?
        > c.max_position_embeddings
    {
        return Err("prompt plus generation exceeds context before compilation".into());
    }
    let weights = assets
        .state
        .values()
        .map(|v| Ok(BufferData::from(&v.to_buffer()?)))
        .collect::<Result<Vec<_>>>()?;
    let mut cache = KvState::new(&c)?;
    let mut generated = Vec::new();
    for _ in 0..count {
        let logits = infer(&c, &weights, &input, Some(&mut cache))?;
        let next = greedy(&logits)? as u32;
        generated.push(next);
        if args.iter().any(|a| a == "--dump") {
            eprintln!(
                "tokens={input:?} logits={:?} layers={} cache_valid={}",
                logits.shape, c.num_hidden_layers, cache.valid
            );
        }
        if next == assets.eos {
            break;
        }
        input = vec![next];
    }
    let text = assets
        .tokenizer
        .decode(&generated, false)
        .map_err(|e| e.to_string())?;
    println!("ids={generated:?}\n{text}");
    Ok(())
}
```

`src/nn/mod.rs`を追記する。

```rust
// file="src/nn/mod.rs" mode="append"

pub mod transformer;
```

`src/nn/qwen.rs` の `use super::`を置き換える。

```rust
// file="src/nn/qwen.rs" mode="replace" item="use super::"
use super::{
    attention::{apply_rope, attention, concatenate, rms_norm},
    linear, silu,
};
```

`src/nn/qwen.rs` の `use crate::`を置き換える。

```rust
// file="src/nn/qwen.rs" mode="replace" item="use crate::"
use crate::{
    Buffer, Result, Tensor,
    state::{DType, Registry, Spec, TensorRecord, TensorSet},
};
```

`src/nn/qwen.rs` の `use serde::`を置き換える。

```rust
// file="src/nn/qwen.rs" mode="replace" item="use serde::"
use serde::{Deserialize, Serialize};
```

`src/nn/qwen.rs` の `pub struct Forward`を置き換える。

```rust
// file="src/nn/qwen.rs" mode="replace" item="pub struct Forward"
pub struct Forward {
    pub logits: Tensor,
    pub boundaries: Vec<Tensor>,
    pub cache_outputs: Vec<(Tensor, Tensor)>,
}
```

`src/nn/qwen.rs` の `pub fn logits_all`を置き換える。

```rust
// file="src/nn/qwen.rs" mode="replace" item="pub fn logits_all"
pub fn logits_all(c:&QwenConfig,params:&BTreeMap<String,Tensor>,tokens:&Tensor,start:usize,cache:Option<&[(Tensor,Tensor)]>)->Result<Forward>{
    c.validate()?;
    let shape = tokens.uop.shape()?;
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
            .ok_or_else(|| format!("missing parameter {name}"))
    };
    let d = c.hidden_size;
    let hd = d / c.num_attention_heads;
    let kd = c.num_key_value_heads;
    let mut x = super::attention::embedding(get("model.embed_tokens.weight")?, tokens);
    let mut boundaries = vec![x.clone()];
    let mut cache_outputs = Vec::new();
    for layer in 0..c.num_hidden_layers {
        let p = format!("model.layers.{layer}");
        let n = rms_norm(
            &x,
            get(&format!("{p}.input_layernorm.weight"))?,
            c.rms_norm_eps,
        )?;
        let projection = |name: &str, heads: usize| -> Result<Tensor> {
            Ok(linear(
                &n,
                get(&format!("{p}.self_attn.{name}.weight"))?,
                Some(get(&format!("{p}.self_attn.{name}.bias"))?),
            )?
            .reshape(&[t, heads, hd]))
        };
        let q = apply_rope(
            &projection("q_proj", c.num_attention_heads)?,
            start,
            c.rope_theta,
        )?;
        let k = apply_rope(&projection("k_proj", kd)?, start, c.rope_theta)?;
        let v = projection("v_proj", kd)?;
        let (k, v) = if let Some(old) = cache {
            let (ck, cv) = &old[layer];
            let expected = [c.max_position_embeddings, kd, hd];
            if ck.uop.shape()? != expected || cv.uop.shape()? != expected {
                return Err("cache capacity shape".into());
            }
            let (k, v) = if start == 0 {
                (k, v)
            } else {
                let ranges = [(0, start), (0, kd), (0, hd)];
                (
                    concatenate(&ck.shrink(&ranges), &k, 0)?,
                    concatenate(&cv.shrink(&ranges), &v, 0)?,
                )
            };
            let pads = [(0, c.max_position_embeddings - start - t), (0, 0), (0, 0)];
            let nk = k.pad(&pads).stage();
            let nv = v.pad(&pads).stage();
            cache_outputs.push((nk.clone(), nv.clone()));
            let ranges = [(0, start + t), (0, kd), (0, hd)];
            (nk.shrink(&ranges), nv.shrink(&ranges))
        } else {
            (k, v)
        };
        let attended = attention(&q, &k, &v, start, true)?.reshape(&[t, d]);
        x = x.add(&linear(
            &attended,
            get(&format!("{p}.self_attn.o_proj.weight"))?,
            None,
        )?);
        let n = rms_norm(
            &x,
            get(&format!("{p}.post_attention_layernorm.weight"))?,
            c.rms_norm_eps,
        )?;
        let gate = silu(&linear(
            &n,
            get(&format!("{p}.mlp.gate_proj.weight"))?,
            None,
        )?);
        let up = linear(&n, get(&format!("{p}.mlp.up_proj.weight"))?, None)?;
        let down = linear(
            &gate.mul(&up),
            get(&format!("{p}.mlp.down_proj.weight"))?,
            None,
        )?;
        x = x.add(&down).stage();
        boundaries.push(x.clone());
    }
    let x = rms_norm(&x, get("model.norm.weight")?, c.rms_norm_eps)?;
    boundaries.push(x.clone());
    let output = get(if c.tie_word_embeddings {
        "model.embed_tokens.weight"
    } else {
        "lm_head.weight"
    })?;
    Ok(Forward {
        logits: linear(&x, output, None)?,
        boundaries,
        cache_outputs,
    })
}
```

`src/nn/qwen.rs` の `pub fn infer`を置き換える。

```rust
// file="src/nn/qwen.rs" mode="replace" item="pub fn infer"
pub fn infer(
    c: &QwenConfig,
    weights: &[crate::state::BufferData],
    tokens: &[u32],
    cache: Option<&mut KvState>,
) -> Result<Buffer> {
    let registry = parameter_registry(c)?;
    let parameters = register_parameters(&registry)?;
    let mut bindings = parameter_bindings(&registry, weights)?;
    bindings.insert("tokens".into(), token_buffer(tokens, c)?);
    let input = Tensor::parameter("tokens", &[tokens.len()]);
    let mut cache_tensors = Vec::new();
    if let Some(state) = cache.as_ref() {
        for (i, (key, value)) in state.keys.iter().zip(&state.values).enumerate() {
            let kn = format!("cache.key.{i}");
            let vn = format!("cache.value.{i}");
            cache_tensors.push((
                Tensor::parameter(&kn, &key.shape),
                Tensor::parameter(&vn, &value.shape),
            ));
            bindings.insert(kn, key.clone());
            bindings.insert(vn, value.clone());
        }
    }
    let start = cache.as_ref().map_or(0, |c| c.valid);
    let forward = logits_all(
        c,
        &parameters,
        &input,
        start,
        cache.as_ref().map(|_| cache_tensors.as_slice()),
    )?;
    let mut outputs = vec![forward.logits];
    for (key, value) in forward.cache_outputs {
        outputs.extend([key, value]);
    }
    let result = crate::run_named(&crate::compile_many(&outputs)?, &bindings)?;
    if let Some(cache) = cache {
        cache.adopt(&result[1..], tokens.len())?;
    }
    Ok(result[0].clone())
}
```

`src/nn/qwen.rs`を追記する。

```rust
// file="src/nn/qwen.rs" mode="append"

#[derive(Clone, Debug)]
pub struct KvState {
    pub keys: Vec<Buffer>,
    pub values: Vec<Buffer>,
    pub valid: usize,
    pub capacity: usize,
}

impl KvState {
    pub fn new(c: &QwenConfig) -> Result<Self> {
        c.validate()?;
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
```

`src/nn/transformer.rs`を保存する。

```rust
// file="src/nn/transformer.rs" mode="write"
pub use super::qwen::{
    Forward, QwenConfig as TinyConfig, logits_all, parameter_registry, register_parameters,
};
```

`tests/qwen_cache_contracts.rs`を保存する。

```rust
// file="tests/qwen_cache_contracts.rs" mode="write"
use chibigrad::{Buffer,nn::qwen::{KvState,QwenConfig,parameter_registry,initialize,load_parameters},state::TensorRecord};
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

`tests/qwen_forward.rs` の `use chibigrad::`を置き換える。

```rust
// file="tests/qwen_forward.rs" mode="replace" item="use chibigrad::"
use chibigrad::{
    Buffer, Tensor, compile_many,
    nn::qwen::initialize,
    nn::{
        attention::{apply_rope, attention, softmax},
        qwen::{
            KvState, QwenConfig, logits_all, parameter_bindings, parameter_registry,
            register_parameters,
        },
    },
    run_named,
    state::BufferData,
};
```

`tests/qwen_forward.rs` の `fn run`を置き換える。

```rust
// file="tests/qwen_forward.rs" mode="replace" item="fn run"
fn run(
    c: &QwenConfig,
    weights: &[BufferData],
    tokens: &[usize],
    cache: Option<&KvState>,
) -> (Vec<Buffer>, Vec<Buffer>) {
    let reg = parameter_registry(c).unwrap();
    let params = register_parameters(&reg).unwrap();
    let ids = Tensor::parameter("tokens", &[tokens.len()]);
    let mut bindings = parameter_bindings(&reg, weights).unwrap();
    bindings.insert(
        "tokens".into(),
        Buffer::new(
            vec![tokens.len()],
            tokens.iter().map(|&t| t as f32).collect(),
        )
        .unwrap(),
    );
    let mut cache_ids = Vec::new();
    if let Some(cache) = cache {
        for (i, (k, v)) in cache.keys.iter().zip(&cache.values).enumerate() {
            let kn = format!("cache.k.{i}");
            let vn = format!("cache.v.{i}");
            cache_ids.push((
                Tensor::parameter(&kn, &k.shape),
                Tensor::parameter(&vn, &v.shape),
            ));
            bindings.insert(kn, k.clone());
            bindings.insert(vn, v.clone());
        }
    }
    let forward = logits_all(
        c,
        &params,
        &ids,
        cache.map_or(0, |c| c.valid),
        cache.map(|_| cache_ids.as_slice()),
    )
    .unwrap();
    let mut out = vec![forward.logits];
    out.extend(forward.boundaries.iter().cloned());
    for (k, v) in forward.cache_outputs {
        out.extend([k, v]);
    }
    let all = run_named(&compile_many(&out).unwrap(), &bindings).unwrap();
    let n = 1 + forward.boundaries.len();
    (all[..n].to_vec(), all[n..].to_vec())
}
```

`tests/qwen_forward.rs` の `fn zero_transformer`を置き換える。

```rust
// file="tests/qwen_forward.rs" mode="replace" item="fn zero_transformer"
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

`tests/qwen_forward.rs` の `fn nonzero_layer_boundaries_reference`を置き換える。

```rust
// file="tests/qwen_forward.rs" mode="replace" item="fn nonzero_layer_boundaries_reference"
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

`tests/qwen_forward.rs`を追記する。

```rust
// file="tests/qwen_forward.rs" mode="append"

#[test]
fn causal_rope_and_softmax() {
    let q = Tensor::constant(0., &[2, 2, 2]);
    let k = Tensor::constant(0., &[2, 1, 2]);
    let v = Tensor::from_slice(&[2., 4., 6., 8.], &[2, 1, 2]);
    let y = attention(&q, &k, &v, 0, true).unwrap();
    let rope = apply_rope(&v, 0, 10000.).unwrap();
    let probabilities = softmax(&Tensor::constant(0., &[2, 2]), 1).unwrap();
    let out = compile_many(&[y, rope, probabilities])
        .unwrap()
        .run(&[], &Default::default())
        .unwrap();
    close(&out[0].data, &[2., 4., 2., 4., 4., 6., 4., 6.], 1e-5, 1e-4);
    close(&out[1].data[..2], &[2., 4.], 0., 0.);
    close(&out[2].data, &[0.5; 4], 1e-6, 0.);
}

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

`tests/qwen_public.rs`を保存する。

```rust
// file="tests/qwen_public.rs" mode="write"
use chibigrad::{
    Tensor, compile_many, model_assets,
    nn::qwen::{
        logits_all, parameter_bindings, parameter_registry, register_parameters, token_buffer,
    },
    run_named,
};
use std::path::Path;
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
    let params = register_parameters(&registry).unwrap();
    let weights = assets
        .state
        .values()
        .map(|v| chibigrad::state::BufferData::from(&v.to_buffer().unwrap()))
        .collect::<Vec<_>>();
    let mut bindings = parameter_bindings(&registry, &weights).unwrap();
    bindings.insert("tokens".into(), token_buffer(&ids, &c).unwrap());
    let tokens = Tensor::parameter("tokens", &[ids.len()]);
    let forward = logits_all(&c, &params, &tokens, 0, None).unwrap();
    assert_eq!(forward.boundaries.len(), 26);
    let mut outputs = forward.boundaries;
    outputs.push(forward.logits);
    let actual = run_named(&compile_many(&outputs).unwrap(), &bindings).unwrap();
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

`tools/qwen-reference-requirements.txt`を保存する。

```text
# file="tools/qwen-reference-requirements.txt" mode="write"
torch==2.5.1
transformers==4.46.3
safetensors==0.4.5
tokenizers==0.20.3
```

`tools/qwen_reference.py`を保存する。

```python
# file="tools/qwen_reference.py" mode="write"
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

```sh
# check step=22.05
cargo test --offline --test qwen_forward --test qwen_cache_contracts
```

```sh
# check requires=qwen
cargo test --offline --test qwen_tokenizer official_tokenizer_ids -- --ignored
cargo test --offline --test qwen_public all_24_layers_reference -- --ignored
```

非ゼロモデルの cached decode が完全 prefix と一致し、reset 後も一致する。不正な候補で keys と valid が変化しない。

## 章末の累積確認

前の章までのテストとexampleも再ビルドします。

```sh
# check
cargo test --quiet --all-targets
```

## 要件との対応

### 22-001 token ID の範囲を検査し、embedding の行選択を INDEX と LOAD へ落とす。

<!-- todo:22-001 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-002 RMSNorm を `x*rsqrt(mean(x*x)+eps)*weight` で組み立てる。

<!-- todo:22-002 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-003 softmax を `m=detach(max(scores))` の減算、exp、sum、除算で組み立て、行和を確認する。

<!-- todo:22-003 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-004 RoPE を偶数の head dimension と位置から作り、位置0で入力と一致させる。

<!-- todo:22-004 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-005 `QK^T/sqrt(d)`、causal mask、softmax、V の積を既存 Tensor 演算で作る。

<!-- todo:22-005 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-006 小 config と明示登録した state を用意し、embedding から output projection までを通す。

<!-- todo:22-006 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-007 pre-norm、attention residual、norm、gated MLP、residual、最終 norm の順を固定する。

<!-- todo:22-007 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-008 MLP を `down(silu(gate(x))*up(x))` とし、bias の有無を fixture に固定する。

<!-- todo:22-008 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-009 最初に全重みゼロの fixture を動かし、全 logits が0になることを確認する。

<!-- todo:22-009 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-010 次に固定された非ゼロの重みを使い、独立 reference の各層出力を fixture に保存する。

<!-- todo:22-010 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-011 21の manifest に固定 revision の全ファイルと SHA-256 を登録する。

<!-- todo:22-011 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-012 対象の model type と config の必要項目を検証し、未対応方式をエラーにする。

<!-- todo:22-012 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-013 embedding、全 layer、norm、output の名前と shape を一対一に照合する。

<!-- todo:22-013 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-014 Q/K/V の bias を読み、projection 行列の出力軸と入力軸を確認する。

<!-- todo:22-014 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-015 tied embedding を一つの ParamId へ結び、重複保存された重みがあれば整合性を確認する。

<!-- todo:22-015 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-016 固定版の tokenizer library を使い、21で検証した tokenizer.json と設定を読み込む。

<!-- todo:22-016 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-017 空白、日本語、改行、特殊 token の ID 列を公式 tokenizer の固定 fixture と一致させる。

<!-- todo:22-017 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-018 chat template は最初は使わず、同じ生テキストと token ID を reference に渡す。

<!-- todo:22-018 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-019 全24 layer を f32 で実行し、embedding、層境界、最終 logits を reference と比較する。

<!-- todo:22-019 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-020 容量を超える入力は起動前に拒否し、まず context=16など短い容量で一つの次 token を出す。

<!-- todo:22-020 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-021 layer ごとに K と V の固定容量 buffer を持ち、読取り専用の実行時入力スロットへ渡す。

<!-- todo:22-021 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-022 `start_pos+T<=context` を検証し、RoPE の絶対位置と書込み位置を一致させる。

<!-- todo:22-022 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-023 既存の有効 prefix と今回の K/V を新規出力 buffer へ保存し、その AFTER を attention が読むようにする。

<!-- todo:22-023 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-024 実行完了後に新しい cache 出力を次回の入力スロットへ束縛し、元の入力 buffer は変更しない。

<!-- todo:22-024 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-025 prefill と1 token decode を、cache なしで prefix 全体を再計算した logits と比較する。

<!-- todo:22-025 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-026 新しい prompt の開始時に cache の有効長を初期化し、greedy 生成と EOS 終了を追加する。

<!-- todo:22-026 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-027 小例の causal 平均、softmax 行和、RoPE 位置0が期待どおりになる。

<!-- todo:22-027 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-028 f32 の小 fixture は `abs_err <= 1e-5 + 1e-4*abs(ref)` を初期基準に各層を比較する。

<!-- todo:22-028 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-029 実モデルは各層と logits を `abs_err <= 1e-3 + 1e-3*abs(ref)` で比較し、最大誤差を残す。

<!-- todo:22-029 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-030 argmax の一致だけで数値検証を終えず、近接した logits の差も記録する。

<!-- todo:22-030 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-031 cache 有無と chunk 分割の違いで同じ prefix の logits が許容値内になる。

<!-- todo:22-031 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-032 dump に token ID、shape、head 対応、位置、cache 有効長を出し、重みの全内容は出さない。

<!-- todo:22-032 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 22-033 Metal と OpenCL は CPU fixture と比較し、未実行 backend は未確認と記録する。

<!-- todo:22-033 -->

関連する実装と確認は本章の `22.01`, `22.02`, `22.03`, `22.04`, `22.05` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。
