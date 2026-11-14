# 14. 同じコンパイラで SGD と MNIST を学習する

前の節で動かしたコードに、今回使う演算と処理を追加します。各節の確認を実行してから次へ進みます。

参照するUOpの構成は [tinygrad c22edff](https://github.com/tinygrad/tinygrad/blob/c22edff14c79310a9b2df62a78819cf77c8289fa/tinygrad/uop/ops.py) です。最初から全機能を実装せず、使う節で増やします。

## 14.01 SGD の更新も Tensor としてコンパイルする

学習率とパラメータ・勾配の shape を検証し、w - lr*g を Tensor の演算で作る。損失と更新後の重みを同じ計画の出力にして、次の実行へ重みを差し替える。

`src/lib.rs`を追記する。

```rust
// file="src/lib.rs" mode="append"


pub mod optim;
```

`src/optim.rs`を保存する。

```rust
// file="src/optim.rs" mode="write"
use crate::{Result, Tensor};
pub fn sgd(parameters: &[Tensor], gradients: &[Tensor], learning_rate: f32) -> Result<Vec<Tensor>> {
    if parameters.len() != gradients.len() || !learning_rate.is_finite() || learning_rate <= 0. {
        return Err("invalid SGD inputs".into());
    }
    parameters
        .iter()
        .zip(gradients)
        .map(|(w, g)| {
            if w.shape()? != g.shape()? {
                return Err("SGD gradient shape mismatch".into());
            }
            Ok(w.sub(&g.mul(&Tensor::scalar(learning_rate))))
        })
        .collect()
}
```

`tests/sgd.rs`を保存する。

```rust
// file="tests/sgd.rs" mode="write"
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

```sh
# check step=14.01
cargo test --offline
```

同じ計画を 5 回使う二乗損失が毎回減り、最初の重みが 1 から 0.32768 になる。

## 14.02 線形層・ReLU・損失を組み立てる

linear は重みの転置と matmul、ReLU は比較と WHERE で作る。平均二乗誤差と安定化した log-softmax、実行時ラベルの交差エントロピーを合成する。名前付き入力を既存の入力仕様へ対応づける。

`src/lib.rs`を追記する。

```rust
// file="src/lib.rs" mode="append"


pub mod nn;

pub fn run_named(
    executable: &Executable,
    bindings: &std::collections::BTreeMap<String, Buffer>,
) -> Result<Vec<Buffer>> {
    let inputs = executable
        .inputs()
        .iter()
        .map(|spec| {
            let buffer = bindings
                .get(&spec.name)
                .ok_or_else(|| format!("missing binding {}", spec.name))?;
            if buffer.shape != spec.shape {
                return Err(format!("binding shape {}", spec.name));
            }
            Ok(buffer.clone())
        })
        .collect::<Result<Vec<_>>>()?;
    executable.run(&inputs, &std::collections::HashMap::new())
}
```

`src/nn.rs`を保存する。

```rust
// file="src/nn.rs" mode="write"
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
```

`tests/nn.rs`を保存する。

```rust
// file="tests/nn.rs" mode="write"
use chibigrad::{
    Buffer, Tensor, compile_many,
    nn::{linear, mse, relu},
    run_named,
};
#[test]
fn relu_mlp_gradient_matches_existing_reference() {
    let x = Tensor::parameter("x", &[2, 2]);
    let w = Tensor::parameter("w", &[2, 2]);
    let b = Tensor::parameter("b", &[2]);
    let hidden = relu(&linear(&x, &w, Some(&b)).unwrap());
    let output = linear(&hidden, &w, Some(&b)).unwrap();
    let loss = mse(&output, &Tensor::constant(0., &[2, 2])).unwrap();
    let gradient = loss.gradients(&[b]).unwrap();
    let plan = compile_many(&[loss, gradient[0].clone()]).unwrap();
    let bindings = std::collections::BTreeMap::from([
        (
            "x".into(),
            Buffer::new(vec![2, 2], vec![1., -1., 2., 1.]).unwrap(),
        ),
        (
            "w".into(),
            Buffer::new(vec![2, 2], vec![1., 0., 0., 1.]).unwrap(),
        ),
        ("b".into(), Buffer::new(vec![2], vec![0., 0.]).unwrap()),
    ]);
    let out = run_named(&plan, &bindings).unwrap();
    assert_eq!(out[0].data, [1.5]);
    assert_eq!(out[1].data, [3., 1.]);
}
```

```sh
# check step=14.02
cargo test --offline
```

二層の ReLU MLP の損失が 1.5、共有バイアスの勾配が [3,1] になる。

## 14.03 IDX データで MNIST の学習ループを動かす

IDX の magic・枚数・28×28・payload 長・ラベルを検証する。ミニバッチと重みを再束縛し、コンパイル済みの損失・推論値・SGD 更新を繰り返す。テストでは小さい決定的な IDX fixture を使う。

`examples/mnist.rs`を保存する。

```rust
// file="examples/mnist.rs" mode="write"
use chibigrad::{
    Buffer, Result, Tensor, compile_many,
    data::{Mnist, parse_idx},
    nn::{cross_entropy_runtime, linear, relu},
    run_named,
};
fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let arg = |name: &str| args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone());
    let number = |name: &str, default: &str| {
        arg(name)
            .unwrap_or(default.into())
            .parse::<usize>()
            .map_err(|e| e.to_string())
    };
    let seed = number("--seed", "1")?;
    let steps = number("--steps", "20")?;
    let subset = number("--subset", "128")?;
    let batch = number("--batch", "16")?;
    if batch == 0 || subset == 0 {
        return Err("batch/subset must be nonzero".into());
    }
    let lr = arg("--lr")
        .unwrap_or("0.01".into())
        .parse::<f32>()
        .map_err(|e| e.to_string())?;
    if !lr.is_finite() || lr <= 0. {
        return Err("learning rate".into());
    }
    let read = |images: &str, labels: &str| -> Result<Mnist> {
        parse_idx(
            &std::fs::read(arg(images).ok_or_else(|| format!("missing {images}"))?)
                .map_err(|e| e.to_string())?,
            &std::fs::read(arg(labels).ok_or_else(|| format!("missing {labels}"))?)
                .map_err(|e| e.to_string())?,
        )
    };
    let data = read("--images", "--labels")?;
    if data.images.len() < subset {
        return Err("training subset larger than dataset".into());
    }
    let x = Tensor::parameter("x", &[batch, 784]);
    let target = Tensor::parameter("target", &[batch]);
    let w1 = Tensor::parameter("w1", &[32, 784]);
    let b1 = Tensor::parameter("b1", &[32]);
    let w2 = Tensor::parameter("w2", &[10, 32]);
    let b2 = Tensor::parameter("b2", &[10]);
    let mask = Tensor::parameter("mask", &[batch]);
    let h = relu(&linear(&x, &w1, Some(&b1))?);
    let logits = linear(&h, &w2, Some(&b2))?;
    let loss = cross_entropy_runtime(&logits, &target, &mask)?;
    let parameters = [w1, b1, w2, b2];
    let grads = loss.gradients(&parameters)?;
    let mut outputs = vec![loss, logits];
    outputs.extend(
        parameters
            .iter()
            .zip(&grads)
            .map(|(w, g)| w.sub(&g.mul(&Tensor::scalar(lr)))),
    );
    let plan = compile_many(&outputs)?;
    let mut rng = (seed as u64).max(1);
    let mut weights = Vec::new();
    for (shape, bias) in [
        (vec![32, 784], false),
        (vec![32], true),
        (vec![10, 32], false),
        (vec![10], true),
    ] {
        let n = shape.iter().product();
        let values = (0..n)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                if bias {
                    0.
                } else {
                    ((rng >> 40) as f32 / 16_777_216. - 0.5) * 0.05
                }
            })
            .collect();
        weights.push(Buffer::new(shape, values)?);
    }
    println!(
        "seed={seed} lr={lr} subset={subset} batch={batch} steps={steps} order=sequential math=libm"
    );
    let binding = |dataset: &Mnist,
                   indices: &[usize],
                   weights: &[Buffer]|
     -> Result<std::collections::BTreeMap<String, Buffer>> {
        let count = indices.len();
        let mut image = vec![0.; batch * 784];
        let mut labels = vec![0.; batch];
        let mut mask = vec![0.; batch];
        for (i, &sample) in indices.iter().enumerate() {
            image[i * 784..(i + 1) * 784].copy_from_slice(&dataset.images[sample]);
            labels[i] = dataset.labels[sample] as f32;
            mask[i] = 1.;
        }
        let mut bindings = vec![
            Buffer::new(vec![batch, 784], image)?,
            Buffer::new(vec![batch], labels)?,
        ];
        bindings.extend_from_slice(weights);
        bindings.push(Buffer::new(vec![batch], mask)?);
        if count == 0 {
            return Err("empty batch".into());
        }
        Ok(["x", "target", "w1", "b1", "w2", "b2", "mask"]
            .into_iter()
            .map(String::from)
            .zip(bindings)
            .collect())
    };
    for step in 0..steps {
        let indices = (0..batch)
            .map(|i| (step * batch + i) % subset)
            .collect::<Vec<_>>();
        let result = run_named(&plan, &binding(&data, &indices, &weights)?)?;
        let value = result[0].data[0];
        if !value.is_finite() {
            return Err(format!("nonfinite loss at step {step}"));
        }
        let candidate = result[2..].to_vec();
        if candidate
            .iter()
            .flat_map(|b| &b.data)
            .any(|v| !v.is_finite())
        {
            return Err(format!("nonfinite update at step {step}"));
        }
        weights = candidate;
        println!("train step={step} loss={value}");
    }
    if arg("--test-images").is_some() {
        let evaluation = read("--test-images", "--test-labels")?;
        let n = number("--test-subset", "128")?.min(evaluation.images.len());
        if n == 0 {
            return Err("empty evaluation subset".into());
        }
        let mut loss = 0f64;
        let mut correct = 0usize;
        for start in (0..n).step_by(batch) {
            let indices = (start..(start + batch).min(n)).collect::<Vec<_>>();
            let out = run_named(&plan, &binding(&evaluation, &indices, &weights)?)?;
            loss += out[0].data[0] as f64 * indices.len() as f64;
            for (row, &sample) in indices.iter().enumerate() {
                let scores = &out[1].data[row * 10..(row + 1) * 10];
                let guess = scores
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .unwrap()
                    .0;
                correct += usize::from(guess == evaluation.labels[sample]);
            }
        }
        println!(
            "evaluation loss={} accuracy={} samples={n}",
            loss / n as f64,
            correct as f64 / n as f64
        );
    }
    println!("training_plan_runs={steps}");
    Ok(())
}
```

`examples/mnist_fixture.rs`を保存する。

```rust
// file="examples/mnist_fixture.rs" mode="write"
fn main() -> Result<(), String> {
    let directory = std::env::args().nth(1).unwrap_or("fixtures".into());
    std::fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
    let (images, labels) = chibigrad::data::idx_fixture();
    std::fs::write(format!("{directory}/mnist-images.idx"), images).map_err(|e| e.to_string())?;
    std::fs::write(format!("{directory}/mnist-labels.idx"), labels).map_err(|e| e.to_string())?;
    Ok(())
}
```

`src/data.rs`を保存する。

```rust
// file="src/data.rs" mode="write"
use crate::Result;
#[derive(Clone, Debug)]
pub struct Mnist {
    pub images: Vec<Vec<f32>>,
    pub labels: Vec<usize>,
}
fn word(bytes: &[u8], offset: usize) -> Result<usize> {
    Ok(u32::from_be_bytes(
        bytes
            .get(offset..offset + 4)
            .ok_or("short IDX header")?
            .try_into()
            .unwrap(),
    ) as usize)
}
pub fn parse_idx(images: &[u8], labels: &[u8]) -> Result<Mnist> {
    if word(images, 0)? != 2051 || word(labels, 0)? != 2049 {
        return Err("bad IDX magic".into());
    }
    let n = word(images, 4)?;
    if word(labels, 4)? != n {
        return Err("IDX count mismatch".into());
    }
    if word(images, 8)? != 28 || word(images, 12)? != 28 {
        return Err("IDX dimensions must be 28x28".into());
    }
    let bytes = n
        .checked_mul(784)
        .and_then(|v| v.checked_add(16))
        .ok_or("IDX length overflow")?;
    if images.len() != bytes || labels.len() != n.checked_add(8).ok_or("IDX label overflow")? {
        return Err("short or trailing IDX payload".into());
    }
    if labels[8..].iter().any(|&l| l >= 10) {
        return Err("IDX label range".into());
    }
    Ok(Mnist {
        images: images[16..]
            .chunks_exact(784)
            .map(|row| row.iter().map(|&v| v as f32 / 255.).collect())
            .collect(),
        labels: labels[8..].iter().map(|&l| l as usize).collect(),
    })
}
pub fn idx_fixture() -> (Vec<u8>, Vec<u8>) {
    let mut images = Vec::new();
    for w in [2051u32, 2, 28, 28] {
        images.extend(w.to_be_bytes());
    }
    images.extend(vec![0; 784]);
    images.extend(vec![255; 784]);
    let mut labels = Vec::new();
    for w in [2049u32, 2] {
        labels.extend(w.to_be_bytes());
    }
    labels.extend([0, 1]);
    (images, labels)
}
```

`src/lib.rs`を追記する。

```rust
// file="src/lib.rs" mode="append"

pub mod data;
```

`tests/mnist.rs`を保存する。

```rust
// file="tests/mnist.rs" mode="write"
use chibigrad::{
    Buffer, Tensor, compile_many,
    data::{idx_fixture, parse_idx},
    nn::{linear, mse, relu},
    run_named,
};
#[test]
fn relu_mlp_gradient_matches_existing_reference() {
    let x = Tensor::parameter("x", &[2, 2]);
    let w = Tensor::parameter("w", &[2, 2]);
    let b = Tensor::parameter("b", &[2]);
    let hidden = relu(&linear(&x, &w, Some(&b)).unwrap());
    let output = linear(&hidden, &w, Some(&b)).unwrap();
    let loss = mse(&output, &Tensor::constant(0., &[2, 2])).unwrap();
    let gradient = loss.gradients(&[b]).unwrap();
    let plan = compile_many(&[loss, gradient[0].clone()]).unwrap();
    let bindings = std::collections::BTreeMap::from([
        (
            "x".into(),
            Buffer::new(vec![2, 2], vec![1., -1., 2., 1.]).unwrap(),
        ),
        (
            "w".into(),
            Buffer::new(vec![2, 2], vec![1., 0., 0., 1.]).unwrap(),
        ),
        ("b".into(), Buffer::new(vec![2], vec![0., 0.]).unwrap()),
    ]);
    let out = run_named(&plan, &bindings).unwrap();
    assert_eq!(out[0].data, [1.5]);
    assert_eq!(out[1].data, [3., 1.]);
}
#[test]
fn idx_fixture_validation() {
    let (images, labels) = idx_fixture();
    let data = parse_idx(&images, &labels).unwrap();
    assert!(data.images[0].iter().all(|&x| x == 0.));
    assert!(data.images[1].iter().all(|&x| x == 1.));
    assert_eq!(data.labels, [0, 1]);
    let mut bad = images.clone();
    bad[3] = 0;
    assert!(parse_idx(&bad, &labels).unwrap_err().contains("magic"));
    assert!(
        parse_idx(&images[..images.len() - 1], &labels)
            .unwrap_err()
            .contains("payload")
    );
    let mut bad = labels.clone();
    bad[7] = 1;
    assert!(parse_idx(&images, &bad).unwrap_err().contains("count"));
}
```

```sh
# check step=14.03
cargo test --offline
cargo run --offline --example mnist_fixture -- fixtures
cargo run --offline --example mnist -- --images fixtures/mnist-images.idx --labels fixtures/mnist-labels.idx --subset 2 --batch 2 --steps 2
```

IDX の不正データを拒否し、2 サンプルの学習を同じ計画で 2 回実行して有限の損失を出す。

## 章末の累積確認

前の章までのテストとexampleも再ビルドします。

```sh
# check
cargo test --quiet --all-targets
```

## 要件との対応

### 14-001 パラメータのスロット番号とホスト配列を対応させる。

<!-- todo:14-001 -->

関連する実装と確認は本章の `14.01`, `14.02`, `14.03` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 14-002 scalar の平均二乗誤差を作り、w の gradient を compile する。

<!-- todo:14-002 -->

関連する実装と確認は本章の `14.01`, `14.02`, `14.03` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 14-003 SGD のホスト更新と、新しい入力バッファの bind を実装する。

<!-- todo:14-003 -->

関連する実装と確認は本章の `14.01`, `14.02`, `14.03` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 14-004 compiler 呼出し回数と Plan 実行回数を表示する。

<!-- todo:14-004 -->

関連する実装と確認は本章の `14.01`, `14.02`, `14.03` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 14-005 比較と WHERE を追加し、WHERE の値側の枝にだけ勾配を伝える。

<!-- todo:14-005 -->

関連する実装と確認は本章の `14.01`, `14.02`, `14.03` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 14-006 ReLU を WHERE で作り、正の枝だけに勾配を伝える。

<!-- todo:14-006 -->

関連する実装と確認は本章の `14.01`, `14.02`, `14.03` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 14-007 `[2,2] → [2,2] → [2,2]` の小型 fixture で MLP を組む。

<!-- todo:14-007 -->

関連する実装と確認は本章の `14.01`, `14.02`, `14.03` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 14-008 初期重みを明示配列にし、最初の loss と gradient を記録する。

<!-- todo:14-008 -->

関連する実装と確認は本章の `14.01`, `14.02`, `14.03` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 14-009 bias の unbroadcast がバッチ方向を足し戻すことを確認する。

<!-- todo:14-009 -->

関連する実装と確認は本章の `14.01`, `14.02`, `14.03` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 14-010 更新後の loss を同じ Plan で再実行する。

<!-- todo:14-010 -->

関連する実装と確認は本章の `14.01`, `14.02`, `14.03` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 14-011 `tests/fixtures` に合成した 2 件の 28×28 IDX を作る手順を置く。

<!-- todo:14-011 -->

関連する実装と確認は本章の `14.01`, `14.02`, `14.03` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 14-012 最初の画像は全 0、次は全 255、ラベルは `[0,1]` とする。

<!-- todo:14-012 -->

関連する実装と確認は本章の `14.01`, `14.02`, `14.03` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 14-013 header を検証して、正規化後の全 0 と全 1 を確認する。

<!-- todo:14-013 -->

関連する実装と確認は本章の `14.01`, `14.02`, `14.03` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 14-014 不正 magic、短い payload、件数不一致を専用エラーにする。

<!-- todo:14-014 -->

関連する実装と確認は本章の `14.01`, `14.02`, `14.03` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 14-015 実データのファイルパスと subset 件数を CLI から受け取る。

<!-- todo:14-015 -->

関連する実装と確認は本章の `14.01`, `14.02`, `14.03` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 14-016 MAX reduction、比較、WHERE を interpreter と C renderer に通す。

<!-- todo:14-016 -->

関連する実装と確認は本章の `14.01`, `14.02`, `14.03` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 14-017 log-softmax と one-hot cross entropy を既存数学関数から合成する。

<!-- todo:14-017 -->

関連する実装と確認は本章の `14.01`, `14.02`, `14.03` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 14-018 `logits=[1000,1000]` の loss が `ln(2)` になることを確認する。

<!-- todo:14-018 -->

関連する実装と確認は本章の `14.01`, `14.02`, `14.03` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 14-019 MNIST の先頭 128 件、batch=16、固定初期値で 20 step を実行する。

<!-- todo:14-019 -->

関連する実装と確認は本章の `14.01`, `14.02`, `14.03` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 14-020 初期化 seed、学習率、subset、バッチ順、数学モードをログに残す。

<!-- todo:14-020 -->

関連する実装と確認は本章の `14.01`, `14.02`, `14.03` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 14-021 test subset を学習から除き、評価の損失と正解率を別に表示する。

<!-- todo:14-021 -->

関連する実装と確認は本章の `14.01`, `14.02`, `14.03` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 14-022 fixture のテストはネットワークと外部 MNIST を必要としない。

<!-- todo:14-022 -->

関連する実装と確認は本章の `14.01`, `14.02`, `14.03` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 14-023 同じ環境、同じ seed で loss と重み更新を許容誤差内で再現する。

<!-- todo:14-023 -->

関連する実装と確認は本章の `14.01`, `14.02`, `14.03` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 14-024 前向きと backward の graph が実行後にも残る。

<!-- todo:14-024 -->

関連する実装と確認は本章の `14.01`, `14.02`, `14.03` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 14-025 Plan の再利用中に重みの差替えが結果へ反映される。

<!-- todo:14-025 -->

関連する実装と確認は本章の `14.01`, `14.02`, `14.03` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 14-026 非有限な loss が出た step を表示し、成功として扱わない。

<!-- todo:14-026 -->

関連する実装と確認は本章の `14.01`, `14.02`, `14.03` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。
