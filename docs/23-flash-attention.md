# 23. online attention を共通 UOp と協調 GPU kernel で実行する

前の節で動かしたコードに、今回使う演算と処理を追加します。各節の確認を実行してから次へ進みます。

参照するUOpの構成は [tinygrad c22edff](https://github.com/tinygrad/tinygrad/blob/c22edff14c79310a9b2df62a78819cf77c8289fa/tinygrad/uop/ops.py) です。最初から全機能を実装せず、使う節で増やします。

## 23.01 オンライン softmax で attention を分割する

最大値 m、分母 l、重み付き和 acc を tile 間で再尺度化する。通常経路と tile 経路を選ぶ引数をこの節で導入する。学習では通常の微分可能な合成演算を使う。

`examples/qwen_infer.rs` の `fn main`を置き換える。

```rust
// file="examples/qwen_infer.rs" mode="replace" item="fn main"
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
    let tile = arg("--tile")
        .map(|s| s.parse::<usize>().map_err(|e| e.to_string()))
        .transpose()?;
    let mut cache = KvState::new(&c)?;
    let mut generated = Vec::new();
    for _ in 0..count {
        let logits = infer(&c, &weights, &input, Some(&mut cache), tile)?;
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

`src/flash.rs`を保存する。

```rust
// file="src/flash.rs" mode="write"
use crate::{Result, Tensor};

pub fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    query_start: usize,
    causal: bool,
    tile: usize,
) -> Result<Tensor> {
    let qs = q.uop.shape()?;
    let ks = k.uop.shape()?;
    if tile == 0
        || qs.len() != 3
        || ks.len() != 3
        || v.uop.shape()? != ks
        || qs[2] != ks[2]
        || ks[1] == 0
        || qs[1] % ks[1] != 0
        || qs[2] == 0
        || ks[0] == 0
    {
        return Err("online attention shape, tile or GQA ratio".into());
    }
    let (t, h, d, s, hkv) = (qs[0], qs[1], qs[2], ks[0], ks[1]);
    let mut heads = Vec::new();
    for head in 0..h {
        let kh = head / (h / hkv);
        let mut rows = Vec::new();
        for row in 0..t {
            let position = query_start
                .checked_add(row)
                .ok_or("query position overflow")?;
            let valid = if causal {
                s.min(position.checked_add(1).ok_or("query position overflow")?)
            } else {
                s
            };
            let query = q
                .shrink(&[(row, row + 1), (head, head + 1), (0, d)])
                .reshape(&[1, d]);
            let mut previous: Option<(Tensor, Tensor, Tensor)> = None;
            for first in (0..valid).step_by(tile) {
                let last = (first + tile).min(valid);
                let key = k
                    .shrink(&[(first, last), (kh, kh + 1), (0, d)])
                    .reshape(&[last - first, d]);
                let value = v
                    .shrink(&[(first, last), (kh, kh + 1), (0, d)])
                    .reshape(&[last - first, d]);
                let scores = query
                    .matmul(&key.permute(&[1, 0]))
                    .mul(&Tensor::scalar(1. / (d as f32).sqrt()));
                let tile_max = scores.max(&[1], true);
                let (m, l, acc) = if let Some((old_max, old_sum, old_acc)) = previous {
                    let m = old_max.lt(&tile_max).where_(&tile_max, &old_max).stage();
                    let alpha = old_max.sub(&m).exp();
                    let probabilities = scores.sub(&m).exp();
                    let l = old_sum.mul(&alpha).add(&probabilities.sum_axes(&[1], true));
                    let acc = old_acc.mul(&alpha).add(&probabilities.matmul(&value));
                    (m, l.stage(), acc.stage())
                } else {
                    let m = tile_max.stage();
                    let probabilities = scores.sub(&m).exp();
                    let l = probabilities.sum_axes(&[1], true).stage();
                    let acc = probabilities.matmul(&value).stage();
                    (m, l, acc)
                };
                previous = Some((m, l, acc));
            }
            let (_, l, acc) = previous.ok_or("online attention has no keys")?;
            rows.push(acc.div(&l).reshape(&[d]));
        }
        heads.push(Tensor::stack(&rows, 0));
    }
    Ok(Tensor::stack(&heads, 1))
}
pub fn for_mode(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    start: usize,
    causal: bool,
    tile: usize,
    training: bool,
) -> Result<Tensor> {
    if training {
        crate::nn::attention::attention(q, k, v, start, causal)
    } else {
        attention(q, k, v, start, causal, tile)
    }
}
```

`src/lib.rs`を追記する。

```rust
// file="src/lib.rs" mode="append"

pub mod flash;
```

`src/nn/qwen.rs` の `pub fn logits_all`を置き換える。

```rust
// file="src/nn/qwen.rs" mode="replace" item="pub fn logits_all"
pub fn logits_all(
    c: &QwenConfig,
    params: &BTreeMap<String, Tensor>,
    tokens: &Tensor,
    start: usize,
    cache: Option<&[(Tensor, Tensor)]>,
) -> Result<Forward> {
    logits_all_with_attention(c, params, tokens, start, cache, &mut attention)
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
    tile: Option<usize>,
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
    let forward = logits_all_with_attention(
        c,
        &parameters,
        &input,
        start,
        cache.as_ref().map(|_| cache_tensors.as_slice()),
        &mut |q, k, v, start, causal| match tile {
            Some(tile) => crate::flash::attention(q, k, v, start, causal, tile),
            None => attention(q, k, v, start, causal),
        },
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

pub fn logits_all_with_attention<F>(
    c: &QwenConfig,
    params: &BTreeMap<String, Tensor>,
    tokens: &Tensor,
    start: usize,
    cache: Option<&[(Tensor, Tensor)]>,
    attention_builder: &mut F,
) -> Result<Forward>
where
    F: FnMut(&Tensor, &Tensor, &Tensor, usize, bool) -> Result<Tensor>,
{
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
        let attended = attention_builder(&q, &k, &v, start, true)?.reshape(&[t, d]);
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

`tests/flash.rs`を保存する。

```rust
// file="tests/flash.rs" mode="write"
use chibigrad::{
    Tensor, compile_many, flash,
    nn::{
        attention::attention,
        qwen::{KvState, QwenConfig, infer, initialize},
    },
};
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!(
            a.is_finite() && (a - b).abs() < 1e-5 + 1e-4 * b.abs(),
            "{a} != {b}"
        );
    }
}
#[test]
fn tiled_streaming_gqa_ragged_and_extreme_scores() {
    for (t, s, start, causal, tile) in [
        (3, 5, 0, true, 2),
        (2, 5, 3, true, 3),
        (3, 5, 0, false, 2),
        (1, 1, 0, true, 8),
    ] {
        let q = Tensor::from_slice(
            &(0..t * 4 * 4)
                .map(|i| ((i * 13 % 31) as f32 - 15.) * 8.)
                .collect::<Vec<_>>(),
            &[t, 4, 4],
        );
        let k = Tensor::from_slice(
            &(0..s * 2 * 4)
                .map(|i| ((i * 7 % 19) as f32 - 9.) * 4.)
                .collect::<Vec<_>>(),
            &[s, 2, 4],
        );
        let v = Tensor::from_slice(
            &(0..s * 2 * 4).map(|i| i as f32 / 9.).collect::<Vec<_>>(),
            &[s, 2, 4],
        );
        let ordinary = attention(&q, &k, &v, start, causal).unwrap();
        let tiled = flash::attention(&q, &k, &v, start, causal, tile).unwrap();
        let out = compile_many(&[ordinary, tiled])
            .unwrap()
            .run(&[], &Default::default())
            .unwrap();
        close(&out[0].data, &out[1].data);
    }
}
#[test]
fn online_model_cache_and_reset_match_full_prefix() {
    let c = QwenConfig::tiny();
    let weights = initialize(&c, 11).unwrap();
    let mut cache = KvState::new(&c).unwrap();
    for tokens in [vec![1, 2], vec![3]] {
        let prefix = if cache.valid == 0 {
            vec![1, 2]
        } else {
            vec![1, 2, 3]
        };
        let ordinary = infer(&c, &weights, &prefix, None, None).unwrap();
        let tiled = infer(&c, &weights, &tokens, Some(&mut cache), Some(2)).unwrap();
        close(
            &tiled.data,
            &ordinary.data[ordinary.data.len() - tiled.data.len()..],
        );
    }
    assert_eq!(cache.valid, 3);
    cache.reset();
    let reset = infer(&c, &weights, &[4], Some(&mut cache), Some(2)).unwrap();
    let plain = infer(&c, &weights, &[4], None, None).unwrap();
    close(&reset.data, &plain.data);
}
#[test]
fn training_uses_differentiable_composite() {
    let q = Tensor::parameter("q", &[2, 2, 2]);
    let k = Tensor::parameter("k", &[3, 1, 2]);
    let v = Tensor::parameter("v", &[3, 1, 2]);
    let y = flash::for_mode(&q, &k, &v, 0, true, 2, true).unwrap();
    let gradient = y.sum().gradients(&[q, k, v]).unwrap();
    assert_eq!(gradient.len(), 3);
    let plan = compile_many(&gradient).unwrap();
    let mut bindings = std::collections::BTreeMap::new();
    for (name, shape) in [
        ("q", vec![2, 2, 2]),
        ("k", vec![3, 1, 2]),
        ("v", vec![3, 1, 2]),
    ] {
        let n = shape.iter().product();
        bindings.insert(
            name.into(),
            chibigrad::Buffer::new(shape, vec![0.25; n]).unwrap(),
        );
    }
    let out = chibigrad::run_named(&plan, &bindings).unwrap();
    assert!(out.iter().flat_map(|b| &b.data).all(|v| v.is_finite()));
    assert!(out[2].data.iter().any(|v| *v != 0.));
}
```

```sh
# check step=23.01
cargo test --offline --test flash
```

ragged tile、GQA、大きな score、decode、reset が通常 attention と一致する。学習用経路の Q/K/V 勾配は有限で value 勾配は非ゼロになる。

## 23.02 協調 GPU attention kernel を生成する

1 workgroup が query/head を担当する。32 lane が K/V tile を shared memory へ載せ、barrier 後にオンライン softmax の状態を更新する。mask と causal 条件、末尾の tile を検査する。

`src/flash_gpu.rs`を保存する。

```rust
// file="src/flash_gpu.rs" mode="write"
use crate::{
    Arg, Axis, Ops, Result, UOp,
    gpu::{Builder, Launch},
};

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
    fn lengths(self, tile: usize) -> Result<(usize, usize, usize)> {
        if [
            self.queries,
            self.keys,
            self.heads,
            self.kv_heads,
            self.dim,
            tile,
        ]
        .contains(&0)
            || !self.heads.is_multiple_of(self.kv_heads)
        {
            return Err(
                "attention requires nonzero dimensions, tile and integral GQA ratio".into(),
            );
        }
        let product = |a: usize, b: usize, c: usize| {
            a.checked_mul(b)
                .and_then(|v| v.checked_mul(c))
                .filter(|&v| v <= i64::MAX as usize)
                .ok_or_else(|| "attention shape overflow".to_string())
        };
        let q = product(self.queries, self.heads, self.dim)?;
        let k = product(self.keys, self.kv_heads, self.dim)?;
        let local = product(tile, self.dim, 1)?;
        self.query_start
            .checked_add(self.queries)
            .filter(|&n| n <= i64::MAX as usize)
            .ok_or("attention position overflow")?;
        local
            .checked_mul(8)
            .and_then(|v| self.dim.checked_mul(4).and_then(|a| v.checked_add(a)))
            .ok_or("attention shared memory overflow")?;
        Ok((q, k, local))
    }
}
fn guarded(b: &mut Builder, condition: UOp) -> UOp {
    let id = b.next;
    b.next += 1;
    b.emit(UOp::range(
        condition.where_(&UOp::int(1), &UOp::int(0)),
        id,
        Axis::Loop,
    ))
}

/// One workgroup owns a query/head. Its 32 lanes load K/V tiles, and lane zero updates online softmax.
pub fn kernel(s: AttentionShape, tile: usize) -> Result<(UOp, Launch)> {
    let (qn, kn, tile_elements) = s.lengths(tile)?;
    let mut b = Builder::new();
    let q = b.param(0, qn);
    let k = b.param(1, kn);
    let v = b.param(2, kn);
    let mask = b.param(3, s.keys);
    let output = b.param(4, qn);
    let lane = b.special("lid0");
    let group = b.special("group0");
    let row = b.emit(group.div(&UOp::int(s.heads as i64)));
    let kh = b.emit(
        group
            .modulo(&UOp::int(s.heads as i64))
            .div(&UOp::int((s.heads / s.kv_heads) as i64)),
    );
    let qb = b.emit(group.mul(&UOp::int(s.dim as i64)));
    let leader = b.emit(lane.eq(&UOp::int(0)));
    let kk = b.alloc(tile_elements.max(2));
    let vv = b.alloc(tile_elements.max(2));
    let acc = b.alloc(s.dim.max(2));
    let max = b.alloc(1);
    let den = b.alloc(1);
    let dot = b.alloc(1);
    let max_init = b.store(
        &max,
        UOp::int(0),
        UOp::f32(f32::NEG_INFINITY),
        UOp::bool(true),
    );
    let den_init = b.store(&den, UOp::int(0), UOp::f32(0.), UOp::bool(true));
    let guard = guarded(&mut b, leader.clone());
    let a = b.range(s.dim);
    b.store(&acc, a.clone(), UOp::f32(0.), UOp::bool(true));
    b.end(&a);
    let initialized = b.end(&guard);
    let block = b.range(s.keys.div_ceil(tile));
    let chunk = b.range(tile_elements.div_ceil(32));
    let z = chunk.mul(&UOp::int(32)).add(&lane);
    let j = block
        .mul(&UOp::int(tile as i64))
        .add(&z.div(&UOp::int(s.dim as i64)));
    let bound = z
        .lt(&UOp::int(tile_elements as i64))
        .and(&j.lt(&UOp::int(s.keys as i64)));
    let valid_mask = b.load(&mask, j.clone(), bound.clone());
    let mut valid = bound.and(&valid_mask.eq(&UOp::f32(0.)).eq(&UOp::bool(false)));
    if s.causal {
        valid = valid.and(&j.lt(&row.add(&UOp::int((s.query_start + 1) as i64))));
    }
    let index = j
        .mul(&UOp::int(s.kv_heads as i64))
        .add(&kh)
        .mul(&UOp::int(s.dim as i64))
        .add(&z.modulo(&UOp::int(s.dim as i64)));
    let key = b.load(&k, index.clone(), valid.clone());
    let val = b.load(&v, index, valid);
    b.store(&kk, z.clone(), key, z.lt(&UOp::int(tile_elements as i64)));
    b.store(&vv, z.clone(), val, z.lt(&UOp::int(tile_elements as i64)));
    let loaded = b.end(&chunk);
    let ready = b.barrier(loaded);
    let guard = guarded(&mut b, leader.clone());
    let off = b.range(tile);
    let j = block.mul(&UOp::int(tile as i64)).add(&off);
    let bound = j.lt(&UOp::int(s.keys as i64));
    let msk = b.load(&mask, j.clone(), bound.clone());
    let mut allowed = bound.and(&msk.eq(&UOp::f32(0.)).eq(&UOp::bool(false)));
    if s.causal {
        allowed = allowed.and(&j.lt(&row.add(&UOp::int((s.query_start + 1) as i64))));
    }
    let key_guard = guarded(&mut b, allowed);
    let dot_init = b.store(
        &dot.after(vec![key_guard.clone()]),
        UOp::int(0),
        UOp::f32(0.),
        UOp::bool(true),
    );
    let a = b.range(s.dim);
    let qv = b.load(&q, qb.add(&a), UOp::bool(true));
    let kv = b.load(
        &kk.after(vec![ready]),
        off.mul(&UOp::int(s.dim as i64)).add(&a),
        UOp::bool(true),
    );
    let old = b.load(
        &dot.after(vec![dot_init, a.clone()]),
        UOp::int(0),
        UOp::bool(true),
    );
    b.store(&dot, UOp::int(0), old.add(&qv.mul(&kv)), UOp::bool(true));
    let dot_end = b.end(&a);
    let score = b
        .load(&dot.after(vec![dot_end]), UOp::int(0), UOp::bool(true))
        .mul(&UOp::f32(1. / (s.dim as f32).sqrt()));
    let m = b.load(
        &max.after(vec![max_init, off.clone()]),
        UOp::int(0),
        UOp::bool(true),
    );
    let l = b.load(
        &den.after(vec![den_init, off.clone()]),
        UOp::int(0),
        UOp::bool(true),
    );
    let nm = b.emit(m.alu(Ops::Max, &[score.clone()]));
    let alpha = b.emit(
        l.eq(&UOp::f32(0.))
            .where_(&UOp::f32(0.), &m.sub(&nm).alu(Ops::Exp, &[])),
    );
    let p = b.emit(score.sub(&nm).alu(Ops::Exp, &[]));
    b.store(&den, UOp::int(0), l.mul(&alpha).add(&p), UOp::bool(true));
    b.store(&max, UOp::int(0), nm, UOp::bool(true));
    let a = b.range(s.dim);
    let old = b.load(
        &acc.after(vec![initialized, a.clone()]),
        a.clone(),
        UOp::bool(true),
    );
    let value = b.load(
        &vv,
        off.mul(&UOp::int(s.dim as i64)).add(&a),
        UOp::bool(true),
    );
    b.store(
        &acc,
        a.clone(),
        old.mul(&alpha).add(&p.mul(&value)),
        UOp::bool(true),
    );
    b.end(&a);
    b.end(&key_guard);
    b.end(&off);
    let consumed = b.end(&guard);
    b.barrier(consumed);
    let finished = b.end(&block);
    let guard = guarded(&mut b, leader);
    let denominator = b.load(
        &den.after(vec![finished.clone()]),
        UOp::int(0),
        UOp::bool(true),
    );
    let a = b.range(s.dim);
    let value = b.load(&acc.after(vec![finished]), a.clone(), UOp::bool(true));
    let value = UOp::f32(0.)
        .lt(&denominator)
        .where_(&value.div(&denominator), &UOp::f32(0.));
    b.store(&output, qb.add(&a), value, UOp::bool(true));
    b.end(&a);
    b.end(&guard);
    let linear = b.finish();
    let mut ops = linear
        .src
        .iter()
        .filter(|u| u.op == Ops::Const)
        .cloned()
        .collect::<Vec<_>>();
    ops.extend(linear.src.iter().filter(|u| u.op != Ops::Const).cloned());
    Ok((
        UOp::linear(ops),
        Launch {
            groups: [s.queries * s.heads, 1, 1],
            local: [32, 1, 1],
        },
    ))
}
```

`src/lib.rs`を追記する。

```rust
// file="src/lib.rs" mode="append"

#[cfg(any(all(feature="metal",target_os="macos"),feature="opencl"))]
pub mod flash_gpu;
```

`tests/flash_kernel.rs`を保存する。

```rust
// file="tests/flash_kernel.rs" mode="write"
#![cfg(any(all(feature="metal",target_os="macos"),feature="opencl"))]
use chibigrad::{Ops,flash_gpu::{AttentionShape,kernel},renderers::{render,Dialect}};
#[test]
fn cooperative_kernel_has_barriers_and_tail_predicates(){
 let s=AttentionShape{queries:2,keys:7,heads:4,kv_heads:2,dim:3,query_start:2,causal:true};
 let (body,launch)=kernel(s,3).unwrap();
 assert_eq!(launch.local,[32,1,1]);assert_eq!(launch.groups,[8,1,1]);
 assert_eq!(body.src.iter().filter(|u|u.op==Ops::Barrier).count(),2);
 for d in [Dialect::Metal,Dialect::OpenCl]{let spec=render(&body,"attention",d,launch.clone()).unwrap();assert!(spec.shared_bytes>=2*3*3*4);assert!(!spec.source.is_empty());}
 assert!(kernel(s,0).is_err());assert!(kernel(AttentionShape{heads:3,..s},3).is_err());
}
```

```sh
# check step=23.02 requires=metal,opencl
cargo test --offline --features metal,opencl --test flash_kernel
```

共通 UOp から Metal と OpenCL の source を生成する。2つの barrier と期待する workgroup 数、shared memory が得られる。

## 23.03 custom CALL を GPU 実行計画と Qwen へ接続する

custom_attention は共通 CALL と出力 AFTER を返す。schedule が入力依存を保持し、GPU 実行計画が通常 Tensor kernel と custom kernel を同じ host queue で実行する。mask 再束縛と Qwen の cached decode を実機で比較する。

`src/flash_gpu.rs` の `use crate::`を置き換える。

```rust
// file="src/flash_gpu.rs" mode="replace" item="use crate::"
use crate::{
    Arg, Axis, Ops, Result, Tensor, UOp,
    gpu::{Builder, Launch},
};
```

`src/flash_gpu.rs`を追記する。

```rust
// file="src/flash_gpu.rs" mode="append"


/// The mask has shape [keys]; zero excludes a key. None enables every key before causal masking.
pub fn custom_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    query_start: usize,
    causal: bool,
    tile: usize,
) -> Result<Tensor> {
    let qs = q.uop.shape()?;
    let ks = k.uop.shape()?;
    if qs.len() != 3 || ks.len() != 3 || ks != v.uop.shape()? || qs[2] != ks[2] {
        return Err("GPU attention expects Q[T,H,D], K/V[S,HKV,D]".into());
    }
    let shape = AttentionShape {
        queries: qs[0],
        keys: ks[0],
        heads: qs[1],
        kv_heads: ks[1],
        dim: qs[2],
        query_start,
        causal,
    };
    let (body, launch) = kernel(shape, tile)?;
    let default_mask = Tensor::scalar(1.).reshape(&[1]).expand(&[ks[0]]);
    let mask = mask.unwrap_or(&default_mask);
    if mask.uop.shape()? != [ks[0]] {
        return Err("GPU attention mask must have shape [keys]".into());
    }
    for tensor in [q, k, v, mask] {
        if tensor
            .uop
            .logical_shape()?
            .iter()
            .zip(tensor.uop.shape()?)
            .any(|(n, size)| *n != UOp::int(size as i64))
        {
            return Err("GPU custom attention requires static dimensions".into());
        }
    }
    let output = Tensor::parameter(
        &format!("attention_output_{}_{}_{}", q.uop.id, body.id, mask.uop.id),
        &qs,
    );
    let call = UOp::new(
        Ops::Call,
        vec![
            body,
            q.stage().uop,
            k.stage().uop,
            v.stage().uop,
            mask.stage().uop,
            output.uop.clone(),
        ],
        Arg::Shape(launch.groups.into_iter().chain(launch.local).collect()),
    );
    Ok(Tensor::from_uop(output.uop.after(vec![call])))
}


#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "opencl"))]
pub fn infer_qwen(
    config: &crate::nn::qwen::QwenConfig,
    weights: &[crate::state::BufferData],
    tokens: &[u32],
    cache: Option<&mut crate::nn::qwen::KvState>,
    backend: crate::gpu_tensor::Backend,
    tile: usize,
) -> Result<crate::Buffer> {
    use crate::nn::qwen::{
        logits_all_with_attention, parameter_bindings, parameter_registry, register_parameters,
        token_buffer,
    };
    let registry = parameter_registry(config)?;
    let parameters = register_parameters(&registry)?;
    let mut bindings = parameter_bindings(&registry, weights)?;
    bindings.insert("tokens".into(), token_buffer(tokens, config)?);
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
    let forward = logits_all_with_attention(
        config,
        &parameters,
        &input,
        start,
        cache.as_ref().map(|_| cache_tensors.as_slice()),
        &mut |q, k, v, start, causal| custom_attention(q, k, v, None, start, causal, tile),
    )?;
    let mut outputs = vec![forward.logits];
    for (key, value) in forward.cache_outputs {
        outputs.extend([key, value]);
    }
    let mut executable = crate::gpu_tensor::GpuExecutable::compile(&outputs, backend)?;
    let inputs = executable
        .inputs()
        .iter()
        .map(|spec| {
            bindings
                .get(&spec.name)
                .cloned()
                .ok_or_else(|| format!("missing binding {}", spec.name))
        })
        .collect::<Result<Vec<_>>>()?;
    let result = executable.run(&inputs, &Default::default())?;
    if let Some(cache) = cache {
        cache.adopt(&result[1..], tokens.len())?;
    }
    Ok(result[0].clone())
}
```

`src/gpu_tensor.rs`を保存する。

```rust
// file="src/gpu_tensor.rs" mode="write"
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::backends::metal::MetalDevice;
#[cfg(feature = "opencl")]
use crate::backends::opencl::OpenClDevice;
use crate::{
    Arg, Axis, Buffer, Ops, Result, Tensor, UOp,
    codegen::{rangeify, signature},
    dimensions::{bind_shape, evaluate, shape_constraints},
    runtime::InputSpec,
    schedule::{Schedule, create_schedule},
    uop::numel,
};
use crate::{
    gpu::{Kernel, Launch},
    queue::{CompiledQueue, HWQueue},
    renderers::{Dialect, render},
};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Metal,
    OpenCl,
}
enum Device {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(MetalDevice),
    #[cfg(feature = "opencl")]
    OpenCl(OpenClDevice),
}
#[derive(Clone)]
enum Allocation {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(crate::backends::metal::MetalBuffer),
    #[cfg(feature = "opencl")]
    OpenCl(crate::backends::opencl::OpenClBuffer),
}
impl Device {
    fn allocate(&self, bytes: usize) -> Result<Allocation> {
        match self {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(d) => d.allocate(bytes.max(4)).map(Allocation::Metal),
            #[cfg(feature = "opencl")]
            Self::OpenCl(d) => d.allocate(bytes.max(4)).map(Allocation::OpenCl),
        }
    }
}
impl Allocation {
    fn retain(&self, queue: &mut HWQueue) -> UOp {
        match self {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(b) => queue.retain_metal_buffer(b.clone()),
            #[cfg(feature = "opencl")]
            Self::OpenCl(b) => queue.retain_opencl_buffer(b.clone()),
        }
    }
    fn transfer(&self, queue: &mut HWQueue, pointer: UOp, bytes: UOp, read: bool) {
        let buffer = self.retain(queue);
        match (self, read) {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            (Self::Metal(_), false) => queue.metal_write(buffer, pointer, bytes),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            (Self::Metal(_), true) => queue.metal_read(buffer, pointer, bytes),
            #[cfg(feature = "opencl")]
            (Self::OpenCl(_), false) => queue.opencl_write(buffer, pointer, bytes),
            #[cfg(feature = "opencl")]
            (Self::OpenCl(_), true) => queue.opencl_read(buffer, pointer, bytes),
        }
    }
}
enum HostBuffer {
    Input(usize),
    Captured(usize),
    Scalars(usize),
    Output(usize),
    Error,
}
pub struct GpuExecutable {
    inputs: Vec<InputSpec>,
    schedule: Schedule,
    programs: Vec<UOp>,
    queue: CompiledQueue,
    input_nodes: Vec<UOp>,
    captured: Vec<Buffer>,
    variables: Vec<Vec<UOp>>,
    transfers: Vec<HostBuffer>,
    sequence: i64,
    shape_checks: Vec<(String, UOp)>,
}

fn transfer(
    queue: &mut HWQueue,
    transfers: &mut Vec<HostBuffer>,
    buffer: &Allocation,
    host: HostBuffer,
    read: bool,
) {
    let index = 2 + transfers.len() * 2;
    transfers.push(host);
    buffer.transfer(
        queue,
        HWQueue::runtime_slot(index),
        HWQueue::runtime_slot(index + 1),
        read,
    );
}

fn parallelize(linear: UOp, single_output: bool) -> (UOp, Launch) {
    let mut launch = Launch {
        groups: [1; 3],
        local: [1; 3],
    };
    let ranges = linear
        .src
        .iter()
        .filter(|u| {
            matches!(
                u.arg,
                Arg::Range {
                    axis: Axis::Loop,
                    ..
                }
            )
        })
        .collect::<Vec<_>>();
    let checked_indices = linear
        .src
        .iter()
        .any(|u| matches!(&u.arg, Arg::Buffer { name, .. } if name == "__chibi_error"));
    if !single_output || ranges.len() != 1 || checked_indices {
        return (linear, launch);
    }
    let range = ranges[0];
    let Arg::Int(extent) = range.src[1].arg else {
        return (linear, launch);
    };
    if extent <= 1 {
        return (linear, launch);
    }
    launch.groups[0] = extent as usize;
    let range_id = range.id;
    let mut replacements = HashMap::new();
    let mut ops = vec![];
    for u in &linear.src {
        let sources = u
            .src
            .iter()
            .map(|s| replacements.get(&s.id).unwrap_or(s).clone())
            .collect();
        let value = if u.id == range_id {
            UOp::new(
                Ops::Range,
                sources,
                Arg::Range {
                    id: 0,
                    axis: Axis::Global,
                },
            )
        } else {
            u.replace(sources)
        };
        replacements.insert(u.id, value.clone());
        ops.push(value);
    }
    (UOp::linear(ops), launch)
}

impl GpuExecutable {
    pub fn inputs(&self) -> &[InputSpec] {
        &self.inputs
    }
    pub fn programs(&self) -> &[UOp] {
        &self.programs
    }
    pub fn schedule(&self) -> &Schedule {
        &self.schedule
    }

    pub fn compile(outputs: &[Tensor], backend: Backend) -> Result<Self> {
        let roots = outputs.iter().map(|t| t.uop.clone()).collect::<Vec<_>>();
        let shape_checks = shape_constraints(&roots)?;
        let schedule = create_schedule(&roots)?;
        let device = match backend {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal => Device::Metal(MetalDevice::open()?),
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            Backend::Metal => return Err("Metal requires macOS".into()),
            #[cfg(not(feature = "opencl"))]
            Backend::OpenCl => return Err("OpenCL feature is disabled".into()),
            #[cfg(feature = "opencl")]
            Backend::OpenCl => Device::OpenCl(OpenClDevice::open(0)?),
        };
        let dialect = match backend {
            Backend::Metal => Dialect::Metal,
            Backend::OpenCl => Dialect::OpenCl,
        };
        let mut queue = HWQueue::new();
        let mut transfers = vec![];
        let mut allocations = HashMap::new();
        let mut sizes = vec![];
        for life in &schedule.lifetimes {
            if sizes.len() <= life.slot {
                sizes.resize(life.slot + 1, 0)
            }
            sizes[life.slot] = sizes[life.slot].max(life.elements);
        }
        let slots = sizes
            .iter()
            .map(|&n| device.allocate(n * 4))
            .collect::<Result<Vec<_>>>()?;
        for life in &schedule.lifetimes {
            allocations.insert(life.buffer.id, slots[life.slot].clone());
        }
        let produced = schedule
            .lifetimes
            .iter()
            .map(|l| l.buffer.id)
            .collect::<HashSet<_>>();
        let mut inputs = vec![];
        let mut input_nodes = vec![];
        let mut captured = vec![];
        for node in schedule
            .buffers
            .iter()
            .filter(|u| !produced.contains(&u.id))
        {
            let Arg::Buffer { name, shape, data } = &node.arg else {
                return Err("invalid input buffer".into());
            };
            let allocation = device.allocate(numel(shape) * 4)?;
            let host = if let Some(bits) = data {
                let i = captured.len();
                captured.push(Buffer::new(
                    shape.clone(),
                    bits.iter().map(|&x| f32::from_bits(x)).collect(),
                )?);
                HostBuffer::Captured(i)
            } else {
                let i = inputs.len();
                inputs.push(InputSpec {
                    name: name.clone(),
                    shape: shape.clone(),
                });
                input_nodes.push(node.clone());
                HostBuffer::Input(i)
            };
            transfer(&mut queue, &mut transfers, &allocation, host, false);
            allocations.insert(node.id, allocation);
        }
        let error = device.allocate(4)?;
        transfer(&mut queue, &mut transfers, &error, HostBuffer::Error, false);
        let mut programs = vec![];
        let mut kernels: Vec<(Kernel, Vec<Allocation>)> = vec![];
        let mut variables = vec![];
        for (i, plan) in schedule.kernels.iter().enumerate() {
            let (linear, launch) = if plan.sink.op == Ops::Linear {
                let Arg::Shape(dims) = &plan.call.arg else {
                    return Err("custom GPU CALL missing launch".into());
                };
                if dims.len() != 6 {
                    return Err("custom GPU launch rank".into());
                }
                (
                    plan.sink.clone(),
                    Launch {
                        groups: [dims[0], dims[1], dims[2]],
                        local: [dims[3], dims[4], dims[5]],
                    },
                )
            } else {
                parallelize(rangeify(&plan.sink)?, plan.writes.len() == 1)
            };
            let sig = signature(&linear)?;
            let kernel = render(&linear, &format!("chibi_gpu_{i}"), dialect, launch)?;
            let mut bindings = vec![];
            for (slot, node) in sig.buffers {
                if bindings.len() != slot {
                    return Err("GPU buffer signature has a gap".into());
                }
                let buffer = if matches!(&node.arg, Arg::Buffer { name, .. } if name == "__chibi_error")
                {
                    error.clone()
                } else {
                    allocations
                        .get(&node.id)
                        .ok_or("missing GPU allocation")?
                        .clone()
                };
                bindings.push(buffer);
            }
            let mut ordered = BTreeMap::<String, UOp>::new();
            for (_, node) in sig.variables {
                let Arg::Variable { name, min, max } = &node.arg else {
                    return Err("invalid scalar variable".into());
                };
                let (mut min, mut max) = (*min, *max);
                if let Some(previous) = ordered.get(name) {
                    if let Arg::Variable { min: a, max: b, .. } = previous.arg {
                        min = min.max(a);
                        max = max.min(b);
                    }
                }
                if min > max {
                    return Err(format!("incompatible bounds for variable {name}"));
                }
                ordered.insert(name.clone(), UOp::variable(name, min, max));
            }
            let vars = ordered.into_values().collect::<Vec<_>>();
            if !vars.is_empty() {
                let scalar_buffer = device.allocate(vars.len() * 8)?;
                transfer(
                    &mut queue,
                    &mut transfers,
                    &scalar_buffer,
                    HostBuffer::Scalars(i),
                    false,
                );
                bindings.push(scalar_buffer);
            }
            variables.push(vars);
            programs.push(UOp::new(
                Ops::Program,
                vec![
                    plan.sink.clone(),
                    linear,
                    UOp::new(Ops::Source, vec![], Arg::Text(kernel.source.clone())),
                ],
                Arg::Text(kernel.entry.clone()),
            ));
            kernels.push((kernel, bindings));
        }
        match device {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Device::Metal(device) => {
                let commands = kernels
                    .iter()
                    .map(|(kernel, bindings)| {
                        let buffers = bindings
                            .iter()
                            .map(|b| match b {
                                Allocation::Metal(b) => b.clone(),
                                _ => unreachable!(),
                            })
                            .collect();
                        Ok((device.compile(kernel)?, kernel.launch.clone(), buffers))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let command = queue.retain_metal_command(device.command_buffer(&commands)?);
                for (index, (_, bindings)) in kernels.iter().enumerate() {
                    for (argument, buffer) in bindings.iter().enumerate() {
                        let buffer = buffer.retain(&mut queue);
                        queue.metal_patch(
                            command.clone(),
                            UOp::int(index as i64),
                            UOp::int(argument as i64),
                            buffer,
                            UOp::int(0),
                        );
                    }
                }
                queue.metal_submit(
                    command.clone(),
                    HWQueue::runtime_slot(0),
                    HWQueue::runtime_slot(1),
                );
                queue.metal_wait(command);
            }
            #[cfg(feature = "opencl")]
            Device::OpenCl(device) => {
                for (kernel, bindings) in &kernels {
                    let args = bindings
                        .iter()
                        .map(|b| match b {
                            Allocation::OpenCl(b) => {
                                crate::backends::opencl::Argument::Buffer(b.clone())
                            }
                            _ => unreachable!(),
                        })
                        .collect();
                    let program = device.compile(kernel)?;
                    let command =
                        queue.retain_opencl_command(program.command(kernel.launch.clone(), args)?);
                    queue.opencl_launch(command.clone());
                    queue.opencl_finish(command);
                }
            }
        }
        for (i, output) in schedule.outputs.iter().enumerate() {
            transfer(
                &mut queue,
                &mut transfers,
                &allocations[&output.id],
                HostBuffer::Output(i),
                true,
            );
        }
        transfer(&mut queue, &mut transfers, &error, HostBuffer::Error, true);
        Ok(Self {
            inputs,
            schedule,
            programs,
            queue: queue.compile("chibi_gpu_queue")?,
            input_nodes,
            captured,
            variables,
            transfers,
            sequence: 0,
            shape_checks,
        })
    }
    pub fn kernel_count(&self) -> usize {
        self.programs.len()
    }
    pub fn sources(&self) -> Vec<&str> {
        self.programs
            .iter()
            .filter_map(|p| {
                p.src.iter().find_map(|s| {
                    if let (Ops::Source, Arg::Text(text)) = (s.op, &s.arg) {
                        Some(text.as_str())
                    } else {
                        None
                    }
                })
            })
            .collect()
    }
    pub fn queue_source(&self) -> Option<&str> {
        self.queue.source()
    }
    pub fn run(&mut self, inputs: &[Buffer], vars: &HashMap<String, i64>) -> Result<Vec<Buffer>> {
        for (message, condition) in &self.shape_checks {
            if evaluate(condition, vars)? == 0 {
                return Err(message.clone());
            }
        }
        if inputs.len() != self.inputs.len() {
            return Err(format!(
                "expected {} input buffers, received {}",
                self.inputs.len(),
                inputs.len()
            ));
        }
        for ((buffer, spec), node) in inputs.iter().zip(&self.inputs).zip(&self.input_nodes) {
            let shape = bind_shape(&node.logical_shape()?, vars)?;
            if buffer.shape != shape || buffer.data.len() != numel(&shape) {
                return Err(format!(
                    "input {} expected {:?}, received {:?}",
                    spec.name, shape, buffer.shape
                ));
            }
        }
        let mut scalars = vec![];
        for variables in &self.variables {
            scalars.push(
                variables
                    .iter()
                    .map(|u| {
                        let Arg::Variable { name, min, max } = &u.arg else {
                            return Err("invalid variable".into());
                        };
                        let value = *vars
                            .get(name)
                            .ok_or_else(|| format!("unbound variable {name}"))?;
                        if value < *min || value > *max {
                            return Err(format!("variable {name}={value} outside {min}..={max}"));
                        }
                        Ok(value)
                    })
                    .collect::<Result<Vec<i64>>>()?,
            );
        }
        let mut outputs = self
            .schedule
            .outputs
            .iter()
            .map(|u| {
                let shape = bind_shape(&u.logical_shape()?, vars)?;
                Buffer::new(shape.clone(), vec![0.; numel(&shape)])
            })
            .collect::<Result<Vec<_>>>()?;
        let mut error = [0f32];
        let next = self
            .sequence
            .checked_add(1)
            .ok_or("GPU timeline overflow")?;
        let mut slots = vec![self.sequence, next];
        for host in &self.transfers {
            let (pointer, bytes) = match host {
                HostBuffer::Input(i) => {
                    (inputs[*i].data.as_ptr() as usize, inputs[*i].data.len() * 4)
                }
                HostBuffer::Captured(i) => (
                    self.captured[*i].data.as_ptr() as usize,
                    self.captured[*i].data.len() * 4,
                ),
                HostBuffer::Scalars(i) => (scalars[*i].as_ptr() as usize, scalars[*i].len() * 8),
                HostBuffer::Output(i) => (
                    outputs[*i].data.as_mut_ptr() as usize,
                    outputs[*i].data.len() * 4,
                ),
                HostBuffer::Error => (error.as_mut_ptr() as usize, 4),
            };
            slots.extend([pointer as i64, bytes as i64]);
        }
        self.sequence = next;
        unsafe { self.queue.run(&mut slots) }?;
        if error[0] != 0. {
            return Err("gather/scatter index must be finite, integral, and within bounds".into());
        }
        Ok(outputs)
    }
}
```

`src/runtime.rs`を保存する。

```rust
// file="src/runtime.rs" mode="write"
#[cfg(feature = "native")]
use crate::codegen::signature;
use crate::{
    Result,
    codegen::{rangeify, render},
    dimensions::{bind_shape, evaluate, shape_constraints},
    schedule::{Schedule, create_schedule},
    tensor::{Buffer, Tensor},
    uop::{Arg, Ops, UOp, numel},
};
use std::collections::{HashMap, HashSet};
#[derive(Clone, Debug)]
pub struct InputSpec {
    pub name: String,
    pub shape: Vec<usize>,
}
/// Compiled signatures and lifetime plans are read-only.
/// ```compile_fail
/// fn alter(e: &mut chibigrad::Executable) { e.schedule.lifetimes.clear(); }
/// ```
pub struct Executable {
    inputs: Vec<InputSpec>,
    schedule: Schedule,
    programs: Vec<UOp>,
    #[cfg(feature = "native")]
    native: Vec<crate::cpu::CpuProgram>,
    input_nodes: Vec<UOp>,
    captured: HashMap<u64, Buffer>,
    shape_checks: Vec<(String, UOp)>,
}
pub fn compile_many(outputs: &[Tensor]) -> Result<Executable> {
    unsafe { compile_many_with_transform(outputs, |linear| Ok(linear.clone())) }
}
/// # Safety
/// The transform must preserve buffer bounds, lifetimes, control-flow scope and
/// the packed signature. Native code generated from the result must be safe to execute.
pub unsafe fn compile_many_with_transform(
    outputs: &[Tensor],
    transform: impl Fn(&UOp) -> Result<UOp>,
) -> Result<Executable> {
    if outputs.iter().any(|t| {
        t.uop
            .toposort()
            .iter()
            .any(|u| u.op == Ops::Call && u.src.first().is_some_and(|b| b.op == Ops::Linear))
    }) {
        return Err(
            "GPU custom kernels require GpuExecutable; use composite attention for CPU execution"
                .into(),
        );
    }
    let shape_checks =
        shape_constraints(&outputs.iter().map(|t| t.uop.clone()).collect::<Vec<_>>())?;
    let debug = std::env::var_os("CHIBI_DEBUG").is_some();
    if debug {
        eprintln!("schedule starts");
    }
    let schedule = create_schedule(&outputs.iter().map(|x| x.uop.clone()).collect::<Vec<_>>())?;
    if debug {
        eprintln!("schedule built {} kernels", schedule.kernels.len());
    }
    let produced: HashSet<_> = schedule.lifetimes.iter().map(|l| l.buffer.id).collect();
    let mut input_nodes = vec![];
    let mut inputs = vec![];
    let mut captured = HashMap::new();
    for u in &schedule.buffers {
        if produced.contains(&u.id) {
            continue;
        }
        let Arg::Buffer { name, shape, data } = &u.arg else {
            return Err("invalid buffer argument".into());
        };
        if let Some(bits) = data {
            captured.insert(
                u.id,
                Buffer::new(
                    shape.clone(),
                    bits.iter().map(|&b| f32::from_bits(b)).collect(),
                )?,
            );
        } else {
            input_nodes.push(u.clone());
            inputs.push(InputSpec {
                name: name.clone(),
                shape: shape.clone(),
            });
        }
    }
    let mut programs = vec![];
    #[cfg(feature = "native")]
    let mut native = vec![];
    for (i, kernel) in schedule.kernels.iter().enumerate() {
        if debug {
            eprintln!("kernel {i} rangeify");
        }
        let linear = transform(&rangeify(&kernel.sink)?)?;
        if debug {
            eprintln!("kernel {i} render {} UOps", linear.src.len());
        }
        let entry = format!("chibi_kernel_{i}");
        let source = render(&linear, &entry)?;
        let stages = vec![
            kernel.sink.clone(),
            linear,
            UOp::new(Ops::Source, vec![], Arg::Text(source.clone())),
        ];
        #[cfg(feature = "native")]
        let stages = {
            if debug {
                eprintln!("kernel {i} compile {} source bytes", source.len());
            }
            let compiled = crate::compiler_cache::compile_cached(&source, &entry)?;
            let mut stages = stages;
            stages.push(UOp::new(
                Ops::Binary,
                vec![],
                Arg::Bytes(compiled.object().to_vec()),
            ));
            native.push(compiled);
            stages
        };
        programs.push(UOp::new(Ops::Program, stages, Arg::Text(entry)));
    }
    Ok(Executable {
        inputs,
        schedule,
        programs,
        #[cfg(feature = "native")]
        native,
        input_nodes,
        captured,
        shape_checks,
    })
}
impl Executable {
    pub fn inputs(&self) -> &[InputSpec] {
        &self.inputs
    }
    pub fn programs(&self) -> &[UOp] {
        &self.programs
    }
    pub fn schedule(&self) -> &Schedule {
        &self.schedule
    }

    pub fn kernel_count(&self) -> usize {
        self.programs.len()
    }
    pub fn sources(&self) -> Vec<&str> {
        self.programs
            .iter()
            .filter_map(|p| p.src.iter().find(|s| s.op == Ops::Source))
            .filter_map(|s| {
                if let Arg::Text(text) = &s.arg {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect()
    }
    pub fn run(&self, inputs: &[Buffer], vars: &HashMap<String, i64>) -> Result<Vec<Buffer>> {
        for (message, condition) in &self.shape_checks {
            if evaluate(condition, vars)? == 0 {
                return Err(message.clone());
            }
        }
        if inputs.len() != self.inputs.len() {
            return Err(format!(
                "expected {} input buffers, received {}",
                self.inputs.len(),
                inputs.len()
            ));
        }
        for ((buffer, spec), node) in inputs.iter().zip(&self.inputs).zip(&self.input_nodes) {
            let expected = bind_shape(&node.logical_shape()?, vars)?;
            if buffer.shape != expected || buffer.data.len() != numel(&expected) {
                return Err(format!(
                    "input {} expected {:?}, received {:?} with {} elements",
                    spec.name,
                    expected,
                    buffer.shape,
                    buffer.data.len()
                ));
            }
        }
        let mut external: HashMap<u64, &Buffer> =
            self.captured.iter().map(|(&id, b)| (id, b)).collect();
        for (u, b) in self.input_nodes.iter().zip(inputs) {
            external.insert(u.id, b);
        }
        let mut sizes = vec![];
        for life in &self.schedule.lifetimes {
            if sizes.len() <= life.slot {
                sizes.resize(life.slot + 1, 0)
            }
            sizes[life.slot] = sizes[life.slot].max(life.elements);
        }
        let slots: Vec<Vec<f32>> = sizes.iter().map(|&n| vec![0.; n]).collect();
        #[cfg(feature = "native")]
        let mut slots = slots;
        let locations: HashMap<_, _> = self
            .schedule
            .lifetimes
            .iter()
            .map(|l| (l.buffer.id, l.slot))
            .collect();
        #[cfg(not(feature = "native"))]
        {
            let _ = (&external, &slots, &locations, vars);
            return Err("native CPU support is disabled".into());
        }
        #[cfg(feature = "native")]
        {
            for (i, program) in self.programs.iter().enumerate() {
                let linear = &program.src[1];
                let sig = signature(linear)?;
                let count = sig
                    .buffers
                    .iter()
                    .map(|x| x.0)
                    .chain(sig.variables.iter().map(|x| x.0))
                    .max()
                    .map_or(0, |x| x + 1);
                let mut packed = vec![0u8; count * 8];
                let mut validation_status = 0f32;
                for (slot, u) in sig.buffers {
                    let pointer = if matches!(&u.arg, Arg::Buffer {name, data:Some(_), ..} if name=="__chibi_error")
                    {
                        &mut validation_status as *mut f32 as usize
                    } else if let Some(b) = external.get(&u.id) {
                        b.data.as_ptr() as usize
                    } else {
                        let loc = locations.get(&u.id).ok_or("missing runtime allocation")?;
                        slots[*loc].as_mut_ptr() as usize
                    };
                    packed[slot * 8..slot * 8 + 8].copy_from_slice(&(pointer as u64).to_ne_bytes());
                }
                for (slot, u) in sig.variables {
                    let Arg::Variable { name, min, max } = &u.arg else {
                        return Err("invalid variable".into());
                    };
                    let value = vars
                        .get(name)
                        .ok_or_else(|| format!("unbound variable {name}"))?;
                    if value < min || value > max {
                        return Err(format!("variable {name}={value} outside {min}..={max}"));
                    }
                    packed[slot * 8..slot * 8 + 8].copy_from_slice(&value.to_ne_bytes());
                }
                unsafe { self.native[i].run(&mut packed) }?;
                if validation_status != 0. {
                    return Err(
                        "gather/scatter index must be finite, integral, and within bounds".into(),
                    );
                }
            }
            self.schedule
                .outputs
                .iter()
                .map(|u| {
                    let slot = locations[&u.id];
                    let shape = bind_shape(&u.logical_shape()?, vars)?;
                    Buffer::new(shape.clone(), slots[slot][..numel(&shape)].to_vec())
                })
                .collect()
        }
    }
}
```

`src/schedule.rs` の `pub fn create_schedule`を置き換える。

```rust
// file="src/schedule.rs" mode="replace" item="pub fn create_schedule"
pub fn create_schedule(outputs: &[UOp]) -> Result<Schedule> {
    if outputs.is_empty() {
        return Err("compile requires at least one output".into());
    }
    let sink = UOp::sink(outputs.to_vec());
    let topo = tensor_toposort(&sink);
    for u in &topo {
        if !matches!(u.op, Ops::Sink | Ops::Call) {
            u.shape()?;
        }
    }
    let mut users: HashMap<u64, usize> = HashMap::new();
    for u in &topo {
        for src in &u.src {
            *users.entry(src.id).or_default() += 1;
        }
    }
    let output_ids: HashSet<_> = outputs.iter().map(|u| u.id).collect();
    let mut buffers: Vec<UOp> = topo
        .iter()
        .filter(|u| u.op == Ops::Buffer)
        .cloned()
        .collect();
    let mut replacements = HashMap::new();
    let mut producers: HashMap<u64, UOp> = HashMap::new();
    let mut kernels = vec![];
    fn add_kernel(
        values: Vec<(UOp, UOp)>,
        kernels: &mut Vec<KernelPlan>,
        producers: &mut HashMap<u64, UOp>,
    ) -> Result<()> {
        let writes: Vec<_> = values.iter().map(|(b, _)| b.clone()).collect();
        let stores = values
            .into_iter()
            .map(|(b, v)| UOp::new(Ops::Store, vec![b, v], Arg::None))
            .collect();
        let sink = UOp::sink(stores);
        let write_ids: HashSet<_> = writes.iter().map(|b| b.id).collect();
        let reads: Vec<_> = sink
            .toposort()
            .into_iter()
            .filter(|u| u.op == Ops::Buffer && !write_ids.contains(&u.id))
            .collect();
        let mut args = vec![sink.clone()];
        args.extend(reads.iter().map(|b| {
            if let Some(call) = producers.get(&b.id) {
                b.after(vec![call.clone()])
            } else {
                b.clone()
            }
        }));
        args.extend(writes.clone());
        let call = UOp::new(Ops::Call, args, Arg::Id(kernels.len()));
        for b in &writes {
            producers.insert(b.id, call.clone());
        }
        kernels.push(KernelPlan {
            sink,
            call,
            reads,
            writes,
        });
        Ok(())
    }
    for u in &topo {
        if u.op == Ops::After
            && u.src.get(1).is_some_and(|c| {
                c.op == Ops::Call && c.src.first().is_some_and(|b| b.op == Ops::Linear)
            })
        {
            if u.src.len() != 2 || u.src[0].op != Ops::Buffer {
                return Err("custom kernel requires AFTER(output buffer, CALL)".into());
            }
            let original = &u.src[1];
            let Arg::Shape(launch) = &original.arg else {
                return Err("custom kernel CALL requires launch dimensions".into());
            };
            if launch.len() != 6 || launch.contains(&0) {
                return Err("custom kernel launch requires six nonzero dimensions".into());
            }
            let args = original.src[1..]
                .iter()
                .map(|a| substitute(a, &replacements))
                .collect::<Vec<_>>();
            if args.iter().any(|a| a.op != Ops::Buffer) || args.last() != Some(&u.src[0]) {
                return Err(
                    "custom kernel arguments must be staged buffers followed by its output".into(),
                );
            }
            let mut bindings = HashMap::new();
            for param in original.src[0]
                .toposort()
                .iter()
                .filter(|p| p.op == Ops::Param)
            {
                let Arg::Slot { index, shape } = &param.arg else {
                    return Err("custom kernel requires buffer parameters".into());
                };
                let buffer = args
                    .get(*index)
                    .ok_or("custom kernel parameter index out of bounds")?;
                if numel(shape) != numel(&buffer.shape()?) {
                    return Err("custom kernel parameter shape mismatch".into());
                }
                bindings.insert(param.id, buffer.clone());
            }
            let body = substitute(&original.src[0], &bindings);
            let writes = vec![u.src[0].clone()];
            let reads = args[..args.len() - 1].to_vec();
            let mut sources = vec![body.clone()];
            sources.extend(reads.iter().map(|b| {
                producers
                    .get(&b.id)
                    .map_or_else(|| b.clone(), |p| b.after(vec![p.clone()]))
            }));
            sources.extend(writes.clone());
            let call = UOp::new(Ops::Call, sources, original.arg.clone());
            producers.insert(writes[0].id, call.clone());
            replacements.insert(u.id, writes[0].clone());
            kernels.push(KernelPlan {
                sink: body,
                call,
                reads,
                writes,
            });
            continue;
        }
        let shared = users.get(&u.id).copied().unwrap_or(0) > 1
            && (is_binary(u.op)
                || is_unary(u.op)
                || matches!(u.op, Ops::Reduce | Ops::Gather | Ops::ScatterAdd));
        let nonempty_reduce = matches!(&u.arg, Arg::Reduction{axes,..} if !axes.is_empty());
        if (u.op == Ops::Stage || shared || nonempty_reduce) && !output_ids.contains(&u.id) {
            let value = if u.op == Ops::Stage {
                substitute(&u.src[0], &replacements)
            } else {
                substitute(u, &replacements)
            };
            let buffer = UOp::new(
                Ops::Buffer,
                u.logical_shape()?,
                Arg::Buffer {
                    name: format!("temp_{}", u.id),
                    shape: u.shape()?,
                    data: None,
                },
            );
            add_kernel(vec![(buffer.clone(), value)], &mut kernels, &mut producers)?;
            buffers.push(buffer.clone());
            replacements.insert(u.id, buffer);
        }
    }
    let mut output_buffers = vec![];
    let mut final_values = vec![];
    for (i, u) in outputs.iter().enumerate() {
        let b = UOp::new(
            Ops::Buffer,
            u.logical_shape()?,
            Arg::Buffer {
                name: format!("output_{}_{}", sink.id, i),
                shape: u.shape()?,
                data: None,
            },
        );
        buffers.push(b.clone());
        output_buffers.push(b.clone());
        final_values.push((b, substitute(u, &replacements)));
    }
    add_kernel(final_values, &mut kernels, &mut producers)?;
    let dag = UOp::sink(
        output_buffers
            .iter()
            .map(|b| b.after(vec![producers[&b.id].clone()]))
            .collect(),
    );
    let linear = crate::effects::order_effects(&dag)?;
    kernels = linear
        .src
        .iter()
        .map(|call| {
            kernels
                .iter()
                .find(|kernel| kernel.call == *call)
                .cloned()
                .ok_or("scheduled CALL missing kernel".to_string())
        })
        .collect::<Result<Vec<_>>>()?;
    let output_set: HashSet<_> = output_buffers.iter().map(|b| b.id).collect();
    let mut lives = vec![];
    for b in buffers.iter().filter(|b| producers.contains_key(&b.id)) {
        let first = kernels
            .iter()
            .position(|k| k.writes.contains(b))
            .ok_or("missing buffer writer")?;
        let last = if output_set.contains(&b.id) {
            kernels.len()
        } else {
            kernels
                .iter()
                .rposition(|k| k.reads.contains(b))
                .unwrap_or(first)
        };
        lives.push(Lifetime {
            buffer: b.clone(),
            first,
            last,
            slot: 0,
            elements: numel(&b.shape()?),
        });
    }
    lives.sort_by_key(|l| l.first);
    let mut slots: Vec<(usize, usize)> = vec![];
    for life in &mut lives {
        if let Some((i, slot)) = slots
            .iter_mut()
            .enumerate()
            .find(|(_, s)| s.0 < life.first && s.1 >= life.elements)
        {
            life.slot = i;
            slot.0 = life.last;
        } else {
            life.slot = slots.len();
            slots.push((life.last, life.elements));
        }
    }
    Ok(Schedule {
        dag,
        linear,
        kernels,
        buffers,
        outputs: output_buffers,
        lifetimes: lives,
    })
}
```

`tests/flash_gpu.rs`を保存する。

```rust
// file="tests/flash_gpu.rs" mode="write"
#![cfg(any(all(feature = "metal", target_os = "macos"), feature = "opencl"))]
use chibigrad::{
    Arg, Ops, Tensor,
    flash_gpu::{AttentionShape, custom_attention, kernel},
};

#[test]
fn common_ir_boundary_and_invalid_inputs() {
    let shape = AttentionShape {
        queries: 2,
        keys: 5,
        heads: 4,
        kv_heads: 2,
        dim: 3,
        query_start: 1,
        causal: true,
    };
    let (body, launch) = kernel(shape, 3).unwrap();
    assert_eq!(body.op, Ops::Linear);
    assert_eq!(launch.local, [32, 1, 1]);
    assert_eq!(launch.groups, [8, 1, 1]);
    assert_eq!(body.src.iter().filter(|u| u.op == Ops::Barrier).count(), 2);
    for dialect in [
        chibigrad::renderers::Dialect::Metal,
        chibigrad::renderers::Dialect::OpenCl,
    ] {
        let program =
            chibigrad::renderers::render(&body, "attention", dialect, launch.clone()).unwrap();
        assert!(program.shared_bytes >= 2 * 3 * 3 * 4);
    }
    assert!(kernel(shape, 0).is_err());
    assert!(kernel(AttentionShape { heads: 3, ..shape }, 3).is_err());
    assert!(
        kernel(
            AttentionShape {
                query_start: usize::MAX,
                ..shape
            },
            3
        )
        .is_err()
    );
    let q = Tensor::parameter("q", &[2, 4, 3]);
    let k = Tensor::parameter("k", &[5, 2, 3]);
    let out = custom_attention(&q, &k, &k, None, 1, true, 3).unwrap();
    let schedule = chibigrad::schedule::create_schedule(&[out.uop().clone()]).unwrap();
    let custom = schedule
        .kernels
        .iter()
        .find(|p| p.sink.op == Ops::Linear)
        .unwrap();
    assert!(custom.sink.toposort().iter().all(|u| u.op != Ops::Param));
    assert!(matches!(&custom.call.arg, Arg::Shape(d) if d == &[8, 1, 1, 32, 1, 1]));
    assert!(
        chibigrad::compile_many(&[out])
            .err()
            .unwrap()
            .contains("GpuExecutable")
    );
    assert!(custom_attention(&q, &k, &k, Some(&Tensor::scalar(1.)), 0, true, 3).is_err());
}

#[cfg(all(feature = "metal", feature = "opencl", target_os = "macos"))]
mod native {
    use super::*;
    use chibigrad::{Backend, Buffer, GpuExecutable};
    use std::collections::{BTreeMap, HashMap};
    fn close(actual: &[f32], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
            assert!(
                a.is_finite() && (a as f64 - b).abs() < 3e-4 + b.abs() * 3e-4,
                "index={i} {a} != {b}"
            );
        }
    }
    fn oracle(s: AttentionShape, q: &[f32], k: &[f32], v: &[f32], mask: &[f32]) -> Vec<f64> {
        let mut out = vec![0.; q.len()];
        for t in 0..s.queries {
            for h in 0..s.heads {
                let kh = h / (s.heads / s.kv_heads);
                let qb = (t * s.heads + h) * s.dim;
                let allowed = (0..s.keys)
                    .filter(|&j| mask[j] != 0. && (!s.causal || j <= s.query_start + t))
                    .collect::<Vec<_>>();
                let scores = allowed
                    .iter()
                    .map(|&j| {
                        (0..s.dim)
                            .map(|a| q[qb + a] as f64 * k[(j * s.kv_heads + kh) * s.dim + a] as f64)
                            .sum::<f64>()
                            / (s.dim as f64).sqrt()
                    })
                    .collect::<Vec<_>>();
                let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights = scores.iter().map(|x| (x - m).exp()).collect::<Vec<_>>();
                let den = weights.iter().sum::<f64>();
                for (&j, p) in allowed.iter().zip(weights) {
                    for a in 0..s.dim {
                        out[qb + a] += p / den * v[(j * s.kv_heads + kh) * s.dim + a] as f64;
                    }
                }
            }
        }
        out
    }
    #[test]
    fn cooperative_attention_masks_tails_widths_gqa_and_rebinding() {
        for (dim, keys, tile, start, causal, scale) in [
            (1, 1, 1, 0, true, 1.),
            (3, 7, 3, 2, true, 1.),
            (8, 5, 8, 0, false, 50.),
            (33, 7, 3, 5, true, 1.),
            (64, 5, 2, 0, false, 1.),
        ] {
            let s = AttentionShape {
                queries: 2,
                keys,
                heads: 4,
                kv_heads: 2,
                dim,
                query_start: start,
                causal,
            };
            let qv = (0..s.queries * s.heads * dim)
                .map(|i| ((i * 13 % 31) as f32 - 15.) * scale / 17.)
                .collect::<Vec<_>>();
            let kv = (0..keys * s.kv_heads * dim)
                .map(|i| ((i * 7 % 19) as f32 - 9.) * scale / 11.)
                .collect::<Vec<_>>();
            let vv = (0..kv.len())
                .map(|i| ((i * 3 % 23) as f32 - 11.) / 7.)
                .collect::<Vec<_>>();
            let q = Tensor::parameter("q", &[s.queries, s.heads, dim]);
            let k = Tensor::parameter("k", &[keys, s.kv_heads, dim]);
            let v = Tensor::parameter("v", &[keys, s.kv_heads, dim]);
            let mask = Tensor::parameter("mask", &[keys]);
            let q = q.add(&Tensor::scalar(0.)).stage();
            let y = custom_attention(&q, &k, &v, Some(&mask), start, causal, tile).unwrap();
            let ordinary = chibigrad::nn::attention::attention(&q, &k, &v, start, causal).unwrap();
            let cpu = chibigrad::compile_many(&[ordinary]).unwrap();
            let mut bindings = BTreeMap::from([
                (
                    "q".into(),
                    Buffer::new(vec![s.queries, s.heads, dim], qv.clone()).unwrap(),
                ),
                (
                    "k".into(),
                    Buffer::new(vec![keys, s.kv_heads, dim], kv.clone()).unwrap(),
                ),
                (
                    "v".into(),
                    Buffer::new(vec![keys, s.kv_heads, dim], vv.clone()).unwrap(),
                ),
            ]);
            let expected = oracle(s, &qv, &kv, &vv, &vec![1.; keys]);
            close(
                &chibigrad::run_named(&cpu, &bindings).unwrap()[0].data,
                &expected,
            );
            for backend in [Backend::Metal, Backend::OpenCl] {
                let mut exe = GpuExecutable::compile(&[y.clone(), y.sum()], backend).unwrap();
                assert_eq!(
                    exe.schedule()
                        .kernels
                        .iter()
                        .filter(|p| p.sink.op == Ops::Linear)
                        .count(),
                    1
                );
                for masks in [
                    vec![1.; keys],
                    (0..keys)
                        .map(|i| if i % 3 == 0 { 0. } else { 1. })
                        .collect(),
                    vec![0.; keys],
                ] {
                    bindings.insert(
                        "mask".into(),
                        Buffer::new(vec![keys], masks.clone()).unwrap(),
                    );
                    let inputs = exe
                        .inputs()
                        .iter()
                        .map(|i| bindings[&i.name].clone())
                        .collect::<Vec<_>>();
                    let result = exe.run(&inputs, &HashMap::new()).unwrap();
                    let expected = oracle(s, &qv, &kv, &vv, &masks);
                    close(&result[0].data, &expected);
                    close(&result[1].data, &[expected.iter().sum()]);
                }
            }
        }
    }
}

#[cfg(all(feature = "metal", feature = "opencl", target_os = "macos"))]
#[test]
fn qwen_custom_kernel_forward_cache_and_reset() {
    use chibigrad::{
        Backend,
        flash_gpu::infer_qwen,
        nn::qwen::{KvState, QwenConfig, infer, initialize},
    };
    let mut config = QwenConfig::tiny();
    config.num_hidden_layers = 1;
    config.max_position_embeddings = 5;
    let weights = initialize(&config, 11).unwrap();
    for backend in [Backend::Metal, Backend::OpenCl] {
        let mut cache = KvState::new(&config).unwrap();
        for tokens in [vec![1, 2], vec![3]] {
            let prefix = if cache.valid == 0 {
                vec![1, 2]
            } else {
                vec![1, 2, 3]
            };
            let expected = infer(&config, &weights, &prefix, None, None).unwrap();
            let full = infer_qwen(&config, &weights, &prefix, None, backend, 2).unwrap();
            let cached =
                infer_qwen(&config, &weights, &tokens, Some(&mut cache), backend, 2).unwrap();
            for (actual, reference) in [
                (&full.data[..], &expected.data[..]),
                (
                    &cached.data[..],
                    &expected.data[expected.data.len() - cached.data.len()..],
                ),
            ] {
                for (a, b) in actual.iter().zip(reference) {
                    assert!(
                        a.is_finite() && (a - b).abs() < 3e-4 + b.abs() * 3e-4,
                        "{a} != {b}"
                    );
                }
            }
        }
        assert_eq!(cache.valid, 3);
        cache.reset();
        let reset = infer_qwen(&config, &weights, &[4], Some(&mut cache), backend, 2).unwrap();
        let expected = infer(&config, &weights, &[4], None, None).unwrap();
        for (a, b) in reset.data.iter().zip(expected.data) {
            assert!(
                a.is_finite() && (a - b).abs() < 3e-4 + b.abs() * 3e-4,
                "{a} != {b}"
            );
        }
    }
}

#[cfg(all(feature = "metal", feature = "opencl", target_os = "macos"))]
#[test]
fn gpu_rejects_runtime_reshape_and_shrink_constraints() {
    use chibigrad::{Backend, Buffer, GpuExecutable, UOp};
    use std::collections::HashMap;
    for backend in [Backend::Metal, Backend::OpenCl] {
        let n = UOp::variable("n", 1, 4);
        let x = Tensor::symbolic_parameter("x", &[n]);
        for output in [x.reshape(&[2, 2]), x.shrink(&[(0, 3)])] {
            let mut exe = GpuExecutable::compile(&[output], backend).unwrap();
            let input = Buffer::new(vec![1], vec![1.]).unwrap();
            assert!(
                exe.run(&[input], &HashMap::from([("n".into(), 1)]))
                    .unwrap_err()
                    .contains("runtime dimensions")
            );
        }
    }
}
```

```sh
# check step=23.03 requires=metal,opencl
cargo test --offline --features metal,opencl --test flash_gpu
```

Metal と OpenCL の実行結果が独立 f64 oracle に一致する。全 mask 無効、dim 1/3/8/33/64、ragged tile、再束縛、KV cache と reset を検証する。

## 章末の累積確認

前の章までのテストとexampleも再ビルドします。

```sh
# check
cargo test --quiet --all-targets
```

## 要件との対応

### 23-001 通常版の score、mask 後の score、softmax、O を小入力で取り出す。

<!-- todo:23-001 -->

関連する実装と確認は本章の `23.01`, `23.02`, `23.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 23-002 head dimension と scale を検証し、shape 不一致を起動前に拒否する。

<!-- todo:23-002 -->

関連する実装と確認は本章の `23.01`, `23.02`, `23.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 23-003 causal、非 causal、padding mask、全 mask 行を別 fixture にする。

<!-- todo:23-003 -->

関連する実装と確認は本章の `23.01`, `23.02`, `23.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 23-004 位置0だけを前提にせず、query 開始位置5などの絶対位置も test に入れる。

<!-- todo:23-004 -->

関連する実装と確認は本章の `23.01`, `23.02`, `23.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 23-005 全 mask 行を含む stable reference を Rust の scalar f64 と既存 f32 graph の両方で動かす。

<!-- todo:23-005 -->

関連する実装と確認は本章の `23.01`, `23.02`, `23.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 23-006 key 軸を小さな tile に分割し、m、l、u を C の累積値として保持する。

<!-- todo:23-006 -->

関連する実装と確認は本章の `23.01`, `23.02`, `23.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 23-007 tile を読み進めるたびに m の変更で既存の l と u を補正する。

<!-- todo:23-007 -->

関連する実装と確認は本章の `23.01`, `23.02`, `23.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 23-008 有効 key がない tile は状態を更新せず、全 mask 行を明示的に処理する。

<!-- todo:23-008 -->

関連する実装と確認は本章の `23.01`, `23.02`, `23.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 23-009 最後の不完全 tile を guard し、無効な K と V の LOAD を生成しない。

<!-- todo:23-009 -->

関連する実装と確認は本章の `23.01`, `23.02`, `23.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 23-010 tile 幅1、2、3、8で通常版と同じ結果を得る。

<!-- todo:23-010 -->

関連する実装と確認は本章の `23.01`, `23.02`, `23.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 23-011 まず一つの group が query の小タイルを担当し、K と V を順に local memory へ運ぶ。

<!-- todo:23-011 -->

関連する実装と確認は本章の `23.01`, `23.02`, `23.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 23-012 QK と PV は通常の f32 積和を使い、hardware matrix 命令はまだ使わない。

<!-- todo:23-012 -->

関連する実装と確認は本章の `23.01`, `23.02`, `23.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 23-013 全 lane が LOAD 後と local buffer 再利用前の barrier を通る構造にする。

<!-- todo:23-013 -->

関連する実装と確認は本章の `23.01`, `23.02`, `23.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 23-014 query と key の端数を別々に guard し、無効 lane に単位元を与える。

<!-- todo:23-014 -->

関連する実装と確認は本章の `23.01`, `23.02`, `23.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 23-015 Metal で検証後、同じ kernel IR を OpenCL に出して通常版と比較する。

<!-- todo:23-015 -->

関連する実装と確認は本章の `23.01`, `23.02`, `23.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 23-016 query head から KV head への対応を追加し、K/V の全複製を避ける。

<!-- todo:23-016 -->

関連する実装と確認は本章の `23.01`, `23.02`, `23.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 23-017 対応 dtype、寸法、local memory、mask、勾配要求を条件として選択する。

<!-- todo:23-017 -->

関連する実装と確認は本章の `23.01`, `23.02`, `23.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 23-018 未対応条件では22の通常 attention を選び、選択理由を dump に残す。

<!-- todo:23-018 -->

関連する実装と確認は本章の `23.01`, `23.02`, `23.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 23-019 学習 graph はこの章では通常版へ固定し、forward-only kernel を微分可能と扱わない。

<!-- todo:23-019 -->

関連する実装と確認は本章の `23.01`, `23.02`, `23.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 23-020 model の logits と cache 有効長を通常版に照合してから、計測用の設定を追加する。

<!-- todo:23-020 -->

関連する実装と確認は本章の `23.01`, `23.02`, `23.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 23-021 基準例の出力は期待値と一致する。

<!-- todo:23-021 -->

関連する実装と確認は本章の `23.01`, `23.02`, `23.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 23-022 CPU、GPU、通常版を f64 reference に対し `abs_err <= 1e-4 + 1e-4*abs(ref)` で比較する。

<!-- todo:23-022 -->

関連する実装と確認は本章の `23.01`, `23.02`, `23.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 23-023 query/key 長の1、tile直前、tile一致、tile直後と、GQA 比率1、2を検証する。

<!-- todo:23-023 -->

関連する実装と確認は本章の `23.01`, `23.02`, `23.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 23-024 全 mask 行、空の有効 key、後半 tile が最大の score、極端な有限 score を試す。

<!-- todo:23-024 -->

関連する実装と確認は本章の `23.01`, `23.02`, `23.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 23-025 無効領域の NaN が有効出力へ流れず、全 mask 行は0になって NaN を含まない。

<!-- todo:23-025 -->

関連する実装と確認は本章の `23.01`, `23.02`, `23.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 23-026 dump で全 lane の barrier と、local buffer の再利用依存を確認する。

<!-- todo:23-026 -->

関連する実装と確認は本章の `23.01`, `23.02`, `23.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 23-027 一つの学習 step が通常版の forward と backward を通り、未定義 gradient を隠さない。

<!-- todo:23-027 -->

関連する実装と確認は本章の `23.01`, `23.02`, `23.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 23-028 実機を使っていない backend と、未測定の速度を確認済みとして記述しない。

<!-- todo:23-028 -->

関連する実装と確認は本章の `23.01`, `23.02`, `23.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。
