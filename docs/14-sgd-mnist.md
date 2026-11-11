# 14. 同じ実行計画を再利用して SGD と MNIST 学習を動かす

パラメータは入力 slot とホストの Buffer で持つ。Plan は値を保持せず、更新後の重みを再 bind する。MLP と安定した cross entropy を同じ graph に組み、IDX の境界検証をデータ入口に集める。

以下は置換・追記先を指定した実装全体である。`mode=write` はそのファイル全体を置き換え、`mode=append` は末尾へ追記する。テストは節ごとに追加し、同じ名前の関数を重複して貼らない。生成 C と共有ライブラリは runtime の一時ディレクトリに作られる。

## 1. パラメータのスロット番号とホスト配列を対応させる。

<!-- todo:14-001 -->

まず、この章が追加するモジュールと共有テスト補助を保存する。後続の各節は、ここに掲載した `nn::{linear,relu,mse,cross_entropy_runtime}`、`data::parse_idx`、`examples/mnist.rs::main` の規則を、焦点を絞ったテストで確認する。

```rust
// file=src/lib.rs mode=append
pub mod data;
pub mod nn;
```

```rust
// file=src/data.rs mode=write
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

```rust
// file=src/nn/mod.rs mode=write
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

```rust
// file=examples/mnist.rs mode=write
use chibigrad::{
	Buffer, Graph, Result,
	autodiff::gradient,
	data::{Mnist, parse_idx},
	nn::{cross_entropy_runtime, linear, relu},
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
	let mut g = Graph::default();
	let x = g.input(0, vec![batch, 784])?;
	let target = g.input(1, vec![batch])?;
	let w1 = g.input(2, vec![32, 784])?;
	let b1 = g.input(3, vec![32])?;
	let w2 = g.input(4, vec![10, 32])?;
	let b2 = g.input(5, vec![10])?;
	let mask = g.input(6, vec![batch])?;
	let h = linear(&mut g, x, w1, Some(b1))?;
	let h = relu(&mut g, h)?;
	let logits = linear(&mut g, h, w2, Some(b2))?;
	let loss = cross_entropy_runtime(&mut g, logits, target, mask)?;
	let grads = gradient(&mut g, loss, &[w1, b1, w2, b2], None)?;
	let mut outputs = vec![loss, logits];
	outputs.extend(grads);
	let plan = g.compile(&outputs)?;
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
	let binding = |dataset: &Mnist, indices: &[usize], weights: &[Buffer]| -> Result<Vec<Buffer>> {
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
		Ok(bindings)
	};
	for step in 0..steps {
		let indices = (0..batch)
			.map(|i| (step * batch + i) % subset)
			.collect::<Vec<_>>();
		let result = plan.run(&binding(&data, &indices, &weights)?)?;
		let value = result[0].data[0];
		if !value.is_finite() {
			return Err(format!("nonfinite loss at step {step}"));
		}
		let mut candidate = weights.clone();
		for (w, grad) in candidate.iter_mut().zip(&result[2..]) {
			for (value, &grad) in w.data.iter_mut().zip(&grad.data) {
				*value -= lr * grad;
				if !value.is_finite() {
					return Err(format!("nonfinite update at step {step}"));
				}
			}
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
			let out = plan.run(&binding(&evaluation, &indices, &weights)?)?;
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
	println!("compiler_calls=1 training_plan_runs={steps}");
	Ok(())
}
```

```rust
// file=tests/mnist.rs mode=write
use chibigrad::{
	Binary, Buffer, Graph,
	autodiff::gradient,
	data::{idx_fixture, parse_idx},
	nn::{cross_entropy, linear, mse, relu},
};
```

`tests/mnist.rs::sgd_rebind_reuses_plan` を追加する。

```rust
// file=tests/mnist.rs mode=append
#[test]
fn sgd_rebind_reuses_plan() {
	let mut g = Graph::default();
	let w = g.input(0, vec![]).unwrap();
	let x = g.input(1, vec![]).unwrap();
	let target = g.input(2, vec![]).unwrap();
	let prediction = g.binary(Binary::Mul, w, x).unwrap();
	let loss = mse(&mut g, prediction, target).unwrap();
	let dw = gradient(&mut g, loss, &[w], None).unwrap()[0];
	let plan = g.compile(&[loss, dw]).unwrap();
	let scalar = |v| Buffer::new(vec![], vec![v]).unwrap();
	let mut weight = 0f32;
	let mut losses = Vec::new();
	for _ in 0..10 {
		let out = plan.run(&[scalar(weight), scalar(2.), scalar(6.)]).unwrap();
		losses.push(out[0].data[0]);
		weight -= 0.1 * out[1].data[0];
	}
	assert!(losses[9] < losses[0]);
	assert!((weight - 3.).abs() < 1e-5);
	println!(
		"compiler_calls=1 plan_runs=10 initial={} final={}",
		losses[0], losses[9]
	);
}
```

実装箇所は `examples/mnist.rs::main` の入力 slot、SGD 更新候補、再 bind と非有限値検査 である。この要件に対する実行可能な確認は `mnist::sgd_rebind_reuses_plan` に置く。

```sh
# check
cargo test --offline --test mnist sgd_rebind_reuses_plan -- --exact
```

## 2. scalar の平均二乗誤差を作り、w の gradient を compile する。

<!-- todo:14-002 -->


実装箇所は `examples/mnist.rs::main` の入力 slot、SGD 更新候補、再 bind と非有限値検査 である。この要件に対する実行可能な確認は `mnist::sgd_rebind_reuses_plan` に置く。

`src/nn/mod.rs` の `pub fn mse(`の定義を次の全体で置き換えます。

```rust
// file="src/nn/mod.rs" mode="replace" item="pub fn mse("
pub fn mse(g: &mut Graph, prediction: NodeId, target: NodeId) -> Result<NodeId> {
	let difference = g.binary(Binary::Sub, prediction, target)?;
	let squared = g.binary(Binary::Mul, difference, difference)?;
	let shape = g.shape(squared).to_vec();
	let n: usize = shape.iter().product();
	let sum = g.reduce(squared, Reduce::Sum, (0..shape.len()).collect(), false)?;
	let scale = g.constant(1. / n as f32, vec![])?;
	g.binary(Binary::Mul, sum, scale)
}
```

```sh
# check
cargo test --offline --test mnist sgd_rebind_reuses_plan -- --exact
```

## 3. SGD のホスト更新と、新しい入力バッファの bind を実装する。

<!-- todo:14-003 -->


実装箇所は `examples/mnist.rs::main` の入力 slot、SGD 更新候補、再 bind と非有限値検査 である。この要件に対する実行可能な確認は `mnist::sgd_rebind_reuses_plan` に置く。

`examples/mnist.rs` の `fn main(`の定義を次の全体で置き換えます。

```rust
// file="examples/mnist.rs" mode="replace" item="fn main("
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
	let mut g = Graph::default();
	let x = g.input(0, vec![batch, 784])?;
	let target = g.input(1, vec![batch])?;
	let w1 = g.input(2, vec![32, 784])?;
	let b1 = g.input(3, vec![32])?;
	let w2 = g.input(4, vec![10, 32])?;
	let b2 = g.input(5, vec![10])?;
	let mask = g.input(6, vec![batch])?;
	let h = linear(&mut g, x, w1, Some(b1))?;
	let h = relu(&mut g, h)?;
	let logits = linear(&mut g, h, w2, Some(b2))?;
	let loss = cross_entropy_runtime(&mut g, logits, target, mask)?;
	let grads = gradient(&mut g, loss, &[w1, b1, w2, b2], None)?;
	let mut outputs = vec![loss, logits];
	outputs.extend(grads);
	let plan = g.compile(&outputs)?;
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
	let binding = |dataset: &Mnist, indices: &[usize], weights: &[Buffer]| -> Result<Vec<Buffer>> {
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
		Ok(bindings)
	};
	for step in 0..steps {
		let indices = (0..batch)
			.map(|i| (step * batch + i) % subset)
			.collect::<Vec<_>>();
		let result = plan.run(&binding(&data, &indices, &weights)?)?;
		let value = result[0].data[0];
		if !value.is_finite() {
			return Err(format!("nonfinite loss at step {step}"));
		}
		let mut candidate = weights.clone();
		for (w, grad) in candidate.iter_mut().zip(&result[2..]) {
			for (value, &grad) in w.data.iter_mut().zip(&grad.data) {
				*value -= lr * grad;
				if !value.is_finite() {
					return Err(format!("nonfinite update at step {step}"));
				}
			}
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
			let out = plan.run(&binding(&evaluation, &indices, &weights)?)?;
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
	println!("compiler_calls=1 training_plan_runs={steps}");
	Ok(())
}
```

```sh
# check
cargo test --offline --test mnist sgd_rebind_reuses_plan -- --exact
```

## 4. compiler 呼出し回数と Plan 実行回数を表示する。

<!-- todo:14-004 -->


実装箇所は `examples/mnist.rs::main` の入力 slot、SGD 更新候補、再 bind と非有限値検査 である。この要件に対する実行可能な確認は `mnist::sgd_rebind_reuses_plan` に置く。

`examples/mnist.rs` の `fn main(`の定義を次の全体で置き換えます。

```rust
// file="examples/mnist.rs" mode="replace" item="fn main("
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
	let mut g = Graph::default();
	let x = g.input(0, vec![batch, 784])?;
	let target = g.input(1, vec![batch])?;
	let w1 = g.input(2, vec![32, 784])?;
	let b1 = g.input(3, vec![32])?;
	let w2 = g.input(4, vec![10, 32])?;
	let b2 = g.input(5, vec![10])?;
	let mask = g.input(6, vec![batch])?;
	let h = linear(&mut g, x, w1, Some(b1))?;
	let h = relu(&mut g, h)?;
	let logits = linear(&mut g, h, w2, Some(b2))?;
	let loss = cross_entropy_runtime(&mut g, logits, target, mask)?;
	let grads = gradient(&mut g, loss, &[w1, b1, w2, b2], None)?;
	let mut outputs = vec![loss, logits];
	outputs.extend(grads);
	let plan = g.compile(&outputs)?;
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
	let binding = |dataset: &Mnist, indices: &[usize], weights: &[Buffer]| -> Result<Vec<Buffer>> {
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
		Ok(bindings)
	};
	for step in 0..steps {
		let indices = (0..batch)
			.map(|i| (step * batch + i) % subset)
			.collect::<Vec<_>>();
		let result = plan.run(&binding(&data, &indices, &weights)?)?;
		let value = result[0].data[0];
		if !value.is_finite() {
			return Err(format!("nonfinite loss at step {step}"));
		}
		let mut candidate = weights.clone();
		for (w, grad) in candidate.iter_mut().zip(&result[2..]) {
			for (value, &grad) in w.data.iter_mut().zip(&grad.data) {
				*value -= lr * grad;
				if !value.is_finite() {
					return Err(format!("nonfinite update at step {step}"));
				}
			}
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
			let out = plan.run(&binding(&evaluation, &indices, &weights)?)?;
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
	println!("compiler_calls=1 training_plan_runs={steps}");
	Ok(())
}
```

```sh
# check
cargo test --offline --test mnist sgd_rebind_reuses_plan -- --exact
```

## 5. 比較と WHERE を追加し、WHERE の値側の枝にだけ勾配を伝える。

<!-- todo:14-005 -->


`tests/mnist.rs::relu_where_and_mlp` を追加する。

```rust
// file=tests/mnist.rs mode=append
#[test]
fn relu_where_and_mlp() {
	let mut g = Graph::default();
	let x = g.input(0, vec![2, 2]).unwrap();
	let w = g.input(1, vec![2, 2]).unwrap();
	let b = g.input(2, vec![2]).unwrap();
	let hidden = linear(&mut g, x, w, Some(b)).unwrap();
	let hidden = relu(&mut g, hidden).unwrap();
	let output = linear(&mut g, hidden, w, Some(b)).unwrap();
	let target = g.constant(0., vec![2, 2]).unwrap();
	let loss = mse(&mut g, output, target).unwrap();
	let db = gradient(&mut g, loss, &[b], None).unwrap()[0];
	let plan = g.compile(&[loss, db]).unwrap();
	let buf = |shape, data| Buffer::new(shape, data).unwrap();
	let out = plan
		.run(&[
			buf(vec![2, 2], vec![1., -1., 2., 1.]),
			buf(vec![2, 2], vec![1., 0., 0., 1.]),
			buf(vec![2], vec![0., 0.]),
		])
		.unwrap();
	assert_eq!(out[0].data, [1.5]);
	assert_eq!(out[1].data, [3., 1.]);
}
```

実装箇所は `nn::relu` と `autodiff::gradient` の `Op::Where` 分岐 である。この要件に対する実行可能な確認は `mnist::relu_where_and_mlp` に置く。

`src/autodiff.rs` の `pub fn gradient(` の `Op::Where` 分岐を次の全体で置き換えます。

```rust
// file="src/autodiff.rs" mode="replace" item="pub fn gradient(" arm="Op::Where"
			Op::Where { cond, yes, no } => {
				let zero = g.constant(0., vec![])?;
				contributions.extend([
					(yes, g.where_(cond, up, zero)?),
					(no, g.where_(cond, zero, up)?),
				]);
			}
```

```sh
# check
cargo test --offline --test mnist relu_where_and_mlp -- --exact
```

## 6. ReLU を WHERE で作り、正の枝だけに勾配を伝える。

<!-- todo:14-006 -->


実装箇所は `nn::relu` と `autodiff::gradient` の `Op::Where` 分岐 である。この要件に対する実行可能な確認は `mnist::relu_where_and_mlp` に置く。

`src/nn/mod.rs` の `pub fn relu(`の定義を次の全体で置き換えます。

```rust
// file="src/nn/mod.rs" mode="replace" item="pub fn relu("
pub fn relu(g: &mut Graph, x: NodeId) -> Result<NodeId> {
	let zero = g.constant(0., vec![])?;
	let positive = g.binary(Binary::Lt, zero, x)?;
	g.where_(positive, x, zero)
}
```

```sh
# check
cargo test --offline --test mnist relu_where_and_mlp -- --exact
```

## 7. `[2,2] → [2,2] → [2,2]` の小型 fixture で MLP を組む。

<!-- todo:14-007 -->


実装箇所は `nn::linear` と `nn::mse`、`autodiff::unbroadcast` である。この要件に対する実行可能な確認は `mnist::relu_where_and_mlp` に置く。

`tests/mnist.rs` の `fn relu_where_and_mlp(`の定義を次の全体で置き換えます。

```rust
// file="tests/mnist.rs" mode="replace" item="fn relu_where_and_mlp("
fn relu_where_and_mlp() {
	let mut g = Graph::default();
	let x = g.input(0, vec![2, 2]).unwrap();
	let w = g.input(1, vec![2, 2]).unwrap();
	let b = g.input(2, vec![2]).unwrap();
	let hidden = linear(&mut g, x, w, Some(b)).unwrap();
	let hidden = relu(&mut g, hidden).unwrap();
	let output = linear(&mut g, hidden, w, Some(b)).unwrap();
	let target = g.constant(0., vec![2, 2]).unwrap();
	let loss = mse(&mut g, output, target).unwrap();
	let db = gradient(&mut g, loss, &[b], None).unwrap()[0];
	let plan = g.compile(&[loss, db]).unwrap();
	let buf = |shape, data| Buffer::new(shape, data).unwrap();
	let out = plan
		.run(&[
			buf(vec![2, 2], vec![1., -1., 2., 1.]),
			buf(vec![2, 2], vec![1., 0., 0., 1.]),
			buf(vec![2], vec![0., 0.]),
		])
		.unwrap();
	assert_eq!(out[0].data, [1.5]);
	assert_eq!(out[1].data, [3., 1.]);
}
```

```sh
# check
cargo test --offline --test mnist relu_where_and_mlp -- --exact
```

## 8. 初期重みを明示配列にし、最初の loss と gradient を記録する。

<!-- todo:14-008 -->


実装箇所は `nn::linear` と `nn::mse`、`autodiff::unbroadcast` である。この要件に対する実行可能な確認は `mnist::relu_where_and_mlp` に置く。

`tests/mnist.rs` の `fn relu_where_and_mlp(`の定義を次の全体で置き換えます。

```rust
// file="tests/mnist.rs" mode="replace" item="fn relu_where_and_mlp("
fn relu_where_and_mlp() {
	let mut g = Graph::default();
	let x = g.input(0, vec![2, 2]).unwrap();
	let w = g.input(1, vec![2, 2]).unwrap();
	let b = g.input(2, vec![2]).unwrap();
	let hidden = linear(&mut g, x, w, Some(b)).unwrap();
	let hidden = relu(&mut g, hidden).unwrap();
	let output = linear(&mut g, hidden, w, Some(b)).unwrap();
	let target = g.constant(0., vec![2, 2]).unwrap();
	let loss = mse(&mut g, output, target).unwrap();
	let db = gradient(&mut g, loss, &[b], None).unwrap()[0];
	let plan = g.compile(&[loss, db]).unwrap();
	let buf = |shape, data| Buffer::new(shape, data).unwrap();
	let out = plan
		.run(&[
			buf(vec![2, 2], vec![1., -1., 2., 1.]),
			buf(vec![2, 2], vec![1., 0., 0., 1.]),
			buf(vec![2], vec![0., 0.]),
		])
		.unwrap();
	assert_eq!(out[0].data, [1.5]);
	assert_eq!(out[1].data, [3., 1.]);
}
```

```sh
# check
cargo test --offline --test mnist relu_where_and_mlp -- --exact
```

## 9. bias の unbroadcast がバッチ方向を足し戻すことを確認する。

<!-- todo:14-009 -->


実装箇所は `nn::linear` と `nn::mse`、`autodiff::unbroadcast` である。この要件に対する実行可能な確認は `mnist::relu_where_and_mlp` に置く。

`tests/mnist.rs` の `fn relu_where_and_mlp(`の定義を次の全体で置き換えます。

```rust
// file="tests/mnist.rs" mode="replace" item="fn relu_where_and_mlp("
fn relu_where_and_mlp() {
	let mut g = Graph::default();
	let x = g.input(0, vec![2, 2]).unwrap();
	let w = g.input(1, vec![2, 2]).unwrap();
	let b = g.input(2, vec![2]).unwrap();
	let hidden = linear(&mut g, x, w, Some(b)).unwrap();
	let hidden = relu(&mut g, hidden).unwrap();
	let output = linear(&mut g, hidden, w, Some(b)).unwrap();
	let target = g.constant(0., vec![2, 2]).unwrap();
	let loss = mse(&mut g, output, target).unwrap();
	let db = gradient(&mut g, loss, &[b], None).unwrap()[0];
	let plan = g.compile(&[loss, db]).unwrap();
	let buf = |shape, data| Buffer::new(shape, data).unwrap();
	let out = plan
		.run(&[
			buf(vec![2, 2], vec![1., -1., 2., 1.]),
			buf(vec![2, 2], vec![1., 0., 0., 1.]),
			buf(vec![2], vec![0., 0.]),
		])
		.unwrap();
	assert_eq!(out[0].data, [1.5]);
	assert_eq!(out[1].data, [3., 1.]);
}
```

```sh
# check
cargo test --offline --test mnist relu_where_and_mlp -- --exact
```

## 10. 更新後の loss を同じ Plan で再実行する。

<!-- todo:14-010 -->


実装箇所は `nn::linear` と `nn::mse`、`autodiff::unbroadcast` である。この要件に対する実行可能な確認は `mnist::sgd_rebind_reuses_plan` に置く。

`examples/mnist.rs` の `fn main(`の定義を次の全体で置き換えます。

```rust
// file="examples/mnist.rs" mode="replace" item="fn main("
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
	let mut g = Graph::default();
	let x = g.input(0, vec![batch, 784])?;
	let target = g.input(1, vec![batch])?;
	let w1 = g.input(2, vec![32, 784])?;
	let b1 = g.input(3, vec![32])?;
	let w2 = g.input(4, vec![10, 32])?;
	let b2 = g.input(5, vec![10])?;
	let mask = g.input(6, vec![batch])?;
	let h = linear(&mut g, x, w1, Some(b1))?;
	let h = relu(&mut g, h)?;
	let logits = linear(&mut g, h, w2, Some(b2))?;
	let loss = cross_entropy_runtime(&mut g, logits, target, mask)?;
	let grads = gradient(&mut g, loss, &[w1, b1, w2, b2], None)?;
	let mut outputs = vec![loss, logits];
	outputs.extend(grads);
	let plan = g.compile(&outputs)?;
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
	let binding = |dataset: &Mnist, indices: &[usize], weights: &[Buffer]| -> Result<Vec<Buffer>> {
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
		Ok(bindings)
	};
	for step in 0..steps {
		let indices = (0..batch)
			.map(|i| (step * batch + i) % subset)
			.collect::<Vec<_>>();
		let result = plan.run(&binding(&data, &indices, &weights)?)?;
		let value = result[0].data[0];
		if !value.is_finite() {
			return Err(format!("nonfinite loss at step {step}"));
		}
		let mut candidate = weights.clone();
		for (w, grad) in candidate.iter_mut().zip(&result[2..]) {
			for (value, &grad) in w.data.iter_mut().zip(&grad.data) {
				*value -= lr * grad;
				if !value.is_finite() {
					return Err(format!("nonfinite update at step {step}"));
				}
			}
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
			let out = plan.run(&binding(&evaluation, &indices, &weights)?)?;
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
	println!("compiler_calls=1 training_plan_runs={steps}");
	Ok(())
}
```

```sh
# check
cargo test --offline --test mnist sgd_rebind_reuses_plan -- --exact
```

## 11. `tests/fixtures` に合成した 2 件の 28×28 IDX を作る手順を置く。

<!-- todo:14-011 -->


`tests/mnist.rs::idx_fixture_validation` を追加する。

```rust
// file=tests/mnist.rs mode=append
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

実装箇所は `data::parse_idx` と `data::idx_fixture` である。この要件に対する実行可能な確認は `mnist::idx_fixture_validation` に置く。

`src/data.rs` の `pub fn idx_fixture(`の定義を次の全体で置き換えます。

```rust
// file="src/data.rs" mode="replace" item="pub fn idx_fixture("
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

```sh
# check
cargo test --offline --test mnist idx_fixture_validation -- --exact
```

## 12. 最初の画像は全 0、次は全 255、ラベルは `[0,1]` とする。

<!-- todo:14-012 -->


実装箇所は `data::parse_idx` と `data::idx_fixture` である。この要件に対する実行可能な確認は `mnist::idx_fixture_validation` に置く。

`src/data.rs` の `pub fn idx_fixture(`の定義を次の全体で置き換えます。

```rust
// file="src/data.rs" mode="replace" item="pub fn idx_fixture("
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

```sh
# check
cargo test --offline --test mnist idx_fixture_validation -- --exact
```

## 13. header を検証して、正規化後の全 0 と全 1 を確認する。

<!-- todo:14-013 -->


実装箇所は `data::parse_idx` と `data::idx_fixture` である。この要件に対する実行可能な確認は `mnist::idx_fixture_validation` に置く。

`src/data.rs` の `pub fn parse_idx(`の定義を次の全体で置き換えます。

```rust
// file="src/data.rs" mode="replace" item="pub fn parse_idx("
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
```

```sh
# check
cargo test --offline --test mnist idx_fixture_validation -- --exact
```

## 14. 不正 magic、短い payload、件数不一致を専用エラーにする。

<!-- todo:14-014 -->


実装箇所は `data::parse_idx` と `data::idx_fixture` である。この要件に対する実行可能な確認は `mnist::idx_fixture_validation` に置く。

`src/data.rs` の `pub fn parse_idx(`の定義を次の全体で置き換えます。

```rust
// file="src/data.rs" mode="replace" item="pub fn parse_idx("
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
```

```sh
# check
cargo test --offline --test mnist idx_fixture_validation -- --exact
```

## 15. 実データのファイルパスと subset 件数を CLI から受け取る。

<!-- todo:14-015 -->


実装箇所は `data::parse_idx` と `data::idx_fixture` である。この要件に対する実行可能な確認は `mnist::idx_fixture_validation` に置く。

`examples/mnist.rs` の `fn main(`の定義を次の全体で置き換えます。

```rust
// file="examples/mnist.rs" mode="replace" item="fn main("
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
	let mut g = Graph::default();
	let x = g.input(0, vec![batch, 784])?;
	let target = g.input(1, vec![batch])?;
	let w1 = g.input(2, vec![32, 784])?;
	let b1 = g.input(3, vec![32])?;
	let w2 = g.input(4, vec![10, 32])?;
	let b2 = g.input(5, vec![10])?;
	let mask = g.input(6, vec![batch])?;
	let h = linear(&mut g, x, w1, Some(b1))?;
	let h = relu(&mut g, h)?;
	let logits = linear(&mut g, h, w2, Some(b2))?;
	let loss = cross_entropy_runtime(&mut g, logits, target, mask)?;
	let grads = gradient(&mut g, loss, &[w1, b1, w2, b2], None)?;
	let mut outputs = vec![loss, logits];
	outputs.extend(grads);
	let plan = g.compile(&outputs)?;
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
	let binding = |dataset: &Mnist, indices: &[usize], weights: &[Buffer]| -> Result<Vec<Buffer>> {
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
		Ok(bindings)
	};
	for step in 0..steps {
		let indices = (0..batch)
			.map(|i| (step * batch + i) % subset)
			.collect::<Vec<_>>();
		let result = plan.run(&binding(&data, &indices, &weights)?)?;
		let value = result[0].data[0];
		if !value.is_finite() {
			return Err(format!("nonfinite loss at step {step}"));
		}
		let mut candidate = weights.clone();
		for (w, grad) in candidate.iter_mut().zip(&result[2..]) {
			for (value, &grad) in w.data.iter_mut().zip(&grad.data) {
				*value -= lr * grad;
				if !value.is_finite() {
					return Err(format!("nonfinite update at step {step}"));
				}
			}
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
			let out = plan.run(&binding(&evaluation, &indices, &weights)?)?;
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
	println!("compiler_calls=1 training_plan_runs={steps}");
	Ok(())
}
```

```sh
# check
cargo test --offline --test mnist idx_fixture_validation -- --exact
```

## 16. MAX reduction、比較、WHERE を interpreter と C renderer に通す。

<!-- todo:14-016 -->


`tests/mnist.rs::stable_uniform_cross_entropy` を追加する。

```rust
// file=tests/mnist.rs mode=append
#[test]
fn stable_uniform_cross_entropy() {
	let mut g = Graph::default();
	let logits = g.input(0, vec![1, 2]).unwrap();
	let loss = cross_entropy(&mut g, logits, &[0], &[true]).unwrap();
	let out = g
		.compile(&[loss])
		.unwrap()
		.run(&[Buffer::new(vec![1, 2], vec![1000., 1000.]).unwrap()])
		.unwrap();
	assert!((out[0].data[0] - 2f32.ln()).abs() < 1e-6);
	assert!(cross_entropy(&mut g, logits, &[2], &[true]).is_err());
	assert!(cross_entropy(&mut g, logits, &[0], &[false]).is_err());
}
```

実装箇所は `nn::log_softmax` と `nn::cross_entropy_runtime` である。この要件に対する実行可能な確認は `mnist::stable_uniform_cross_entropy` に置く。

`src/graph.rs` の `fn expr(`（`impl Emitter<'_>` 内）の定義を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn expr(" within="impl Emitter<'_>"
	fn expr(&mut self, id: usize, index: &str, code: &mut String) -> Result<String> {
		if id != self.current && self.material.contains(&id)
			|| matches!(self.g.nodes[id].op, Op::Input { .. })
		{
			return Ok(format!("b{id}[{index}]"));
		}
		let index_name = self.name();
		code.push_str(&format!("size_t {index_name}={index};\n"));
		let index = index_name.as_str();
		let op = self.g.nodes[id].op.clone();
		let value = match op {
			Op::Input { .. } => unreachable!(),
			Op::Const { bits } => format!("chibi_bits({bits}u)"),
			Op::Unary { op, x } => {
				let a = self.expr(x, index, code)?;
				match op {
					Unary::Neg => format!("(-({a}))"),
					Unary::Recip => format!("(1.0f / ({a}))"),
					other => format!(
						"{}({a})",
						match other {
							Unary::Exp => "expf",
							Unary::Log => "logf",
							Unary::Sin => "sinf",
							Unary::Cos => "cosf",
							Unary::Sqrt => "sqrtf",
							Unary::Exp2 => "exp2f",
							Unary::Log2 => "log2f",
							_ => unreachable!(),
						}
					),
				}
			}
			Op::Binary { op, a, b } => {
				let ai = self.broadcast_index(a, id, index);
				let bi = self.broadcast_index(b, id, index);
				let av = self.expr(a, &ai, code)?;
				let bv = self.expr(b, &bi, code)?;
				match op {
					Binary::Max => format!("chibi_max({av},{bv})"),
					_ => format!(
						"(({av}) {} ({bv}))",
						match op {
							Binary::Add => "+",
							Binary::Sub => "-",
							Binary::Mul => "*",
							Binary::Div => "/",
							Binary::Lt => "<",
							Binary::Eq => "==",
							_ => unreachable!(),
						}
					),
				}
			}
			Op::Where { cond, yes, no } => {
				let ci = self.broadcast_index(cond, id, index);
				let yi = self.broadcast_index(yes, id, index);
				let ni = self.broadcast_index(no, id, index);
				let cv = self.expr(cond, &ci, code)?;
				let target = self.name();
				code.push_str(&format!("float {target};\nif ({cv} != 0.0f) {{\n"));
				let y = self.expr(yes, &yi, code)?;
				code.push_str(&format!("{target} = {y};\n}} else {{\n"));
				let n = self.expr(no, &ni, code)?;
				code.push_str(&format!("{target} = {n};\n}}\n"));
				return Ok(target);
			}
			Op::Reshape { x } | Op::Stage { x } | Op::StopGradient { x } => {
				return self.expr(x, index, code);
			}
			Op::Expand { x } => {
				let xi = self.broadcast_index(x, id, index);
				return self.expr(x, &xi, code);
			}
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
			Op::Permute { x, axes } => {
				let mut c = vec![String::new(); axes.len()];
				for (a, &old) in axes.iter().enumerate() {
					c[old] = self.coord(id, index, a)
				}
				let xi = self.flatten(x, &c);
				return self.expr(x, &xi, code);
			}
			Op::Shrink { x, ranges } => {
				let c = ranges
					.iter()
					.enumerate()
					.map(|(a, &(lo, _))| format!("({}+{lo})", self.coord(id, index, a)))
					.collect::<Vec<_>>();
				let xi = self.flatten(x, &c);
				return self.expr(x, &xi, code);
			}
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
			Op::Flip { x, axes } => {
				let c = (0..self.g.shape(x).len())
					.map(|a| {
						let c = self.coord(id, index, a);
						if axes.contains(&a) {
							format!("({}-1-{c})", self.dim(x, a))
						} else {
							c
						}
					})
					.collect::<Vec<_>>();
				let xi = self.flatten(x, &c);
				return self.expr(x, &xi, code);
			}
			Op::Stack { xs, axis } => {
				let target = self.name();
				code.push_str(&format!("float {target}=0.0f;\n"));
				let select = self.coord(id, index, axis);
				for (j, x) in xs.into_iter().enumerate() {
					code.push_str(&format!(
						"{}if ({select} == {j}) {{\n",
						if j == 0 { "" } else { "else " }
					));
					let c = (0..self.g.shape(id).len())
						.filter(|&a| a != axis)
						.map(|a| self.coord(id, index, a))
						.collect::<Vec<_>>();
					let xi = self.flatten(x, &c);
					let value = self.expr(x, &xi, code)?;
					code.push_str(&format!("{target}={value};\n}}\n"));
				}
				return Ok(target);
			}
			Op::Gather { x, indices, axis } => {
				let ii = self.coord(id, index, axis);
				let iv = self.expr(indices, &ii, code)?;
				let ivar = self.name();
				code.push_str(&format!("float {ivar}={iv};\nif (!isfinite({ivar}) || {ivar}<0.0f || floorf({ivar})!={ivar} || (double){ivar}>=(double){}) return 2;\n",self.dim(x,axis)));
				let mut c = (0..self.g.shape(x).len())
					.map(|a| self.coord(id, index, a))
					.collect::<Vec<_>>();
				c[axis] = format!("(size_t){ivar}");
				let xi = self.flatten(x, &c);
				return self.expr(x, &xi, code);
			}
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
		};
		let variable = self.name();
		code.push_str(&format!("float {variable} = {value};\n"));
		Ok(variable)
	}
```

```sh
# check
cargo test --offline --test mnist stable_uniform_cross_entropy -- --exact
```

## 17. log-softmax と one-hot cross entropy を既存数学関数から合成する。

<!-- todo:14-017 -->


実装箇所は `nn::log_softmax` と `nn::cross_entropy_runtime` である。この要件に対する実行可能な確認は `mnist::stable_uniform_cross_entropy` に置く。

`src/nn/mod.rs` の `pub fn cross_entropy_runtime(`の定義を次の全体で置き換えます。

```rust
// file="src/nn/mod.rs" mode="replace" item="pub fn cross_entropy_runtime("
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

```sh
# check
cargo test --offline --test mnist stable_uniform_cross_entropy -- --exact
```

## 18. `logits=[1000,1000]` の loss が `ln(2)` になることを確認する。

<!-- todo:14-018 -->


実装箇所は `nn::log_softmax` と `nn::cross_entropy_runtime` である。この要件に対する実行可能な確認は `mnist::stable_uniform_cross_entropy` に置く。

`tests/mnist.rs` の `fn stable_uniform_cross_entropy(`の定義を次の全体で置き換えます。

```rust
// file="tests/mnist.rs" mode="replace" item="fn stable_uniform_cross_entropy("
fn stable_uniform_cross_entropy() {
	let mut g = Graph::default();
	let logits = g.input(0, vec![1, 2]).unwrap();
	let loss = cross_entropy(&mut g, logits, &[0], &[true]).unwrap();
	let out = g
		.compile(&[loss])
		.unwrap()
		.run(&[Buffer::new(vec![1, 2], vec![1000., 1000.]).unwrap()])
		.unwrap();
	assert!((out[0].data[0] - 2f32.ln()).abs() < 1e-6);
	assert!(cross_entropy(&mut g, logits, &[2], &[true]).is_err());
	assert!(cross_entropy(&mut g, logits, &[0], &[false]).is_err());
}
```

```sh
# check
cargo test --offline --test mnist stable_uniform_cross_entropy -- --exact
```

## 19. MNIST の先頭 128 件、batch=16、固定初期値で 20 step を実行する。

<!-- todo:14-019 -->


実装箇所は `examples/mnist.rs::main` の入力 slot、SGD 更新候補、再 bind と非有限値検査 である。この要件に対する実行可能な確認は `mnist::stable_uniform_cross_entropy` に置く。

`examples/mnist.rs` の `fn main(`の定義を次の全体で置き換えます。

```rust
// file="examples/mnist.rs" mode="replace" item="fn main("
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
	let mut g = Graph::default();
	let x = g.input(0, vec![batch, 784])?;
	let target = g.input(1, vec![batch])?;
	let w1 = g.input(2, vec![32, 784])?;
	let b1 = g.input(3, vec![32])?;
	let w2 = g.input(4, vec![10, 32])?;
	let b2 = g.input(5, vec![10])?;
	let mask = g.input(6, vec![batch])?;
	let h = linear(&mut g, x, w1, Some(b1))?;
	let h = relu(&mut g, h)?;
	let logits = linear(&mut g, h, w2, Some(b2))?;
	let loss = cross_entropy_runtime(&mut g, logits, target, mask)?;
	let grads = gradient(&mut g, loss, &[w1, b1, w2, b2], None)?;
	let mut outputs = vec![loss, logits];
	outputs.extend(grads);
	let plan = g.compile(&outputs)?;
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
	let binding = |dataset: &Mnist, indices: &[usize], weights: &[Buffer]| -> Result<Vec<Buffer>> {
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
		Ok(bindings)
	};
	for step in 0..steps {
		let indices = (0..batch)
			.map(|i| (step * batch + i) % subset)
			.collect::<Vec<_>>();
		let result = plan.run(&binding(&data, &indices, &weights)?)?;
		let value = result[0].data[0];
		if !value.is_finite() {
			return Err(format!("nonfinite loss at step {step}"));
		}
		let mut candidate = weights.clone();
		for (w, grad) in candidate.iter_mut().zip(&result[2..]) {
			for (value, &grad) in w.data.iter_mut().zip(&grad.data) {
				*value -= lr * grad;
				if !value.is_finite() {
					return Err(format!("nonfinite update at step {step}"));
				}
			}
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
			let out = plan.run(&binding(&evaluation, &indices, &weights)?)?;
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
	println!("compiler_calls=1 training_plan_runs={steps}");
	Ok(())
}
```

```sh
# check
cargo test --offline --test mnist stable_uniform_cross_entropy -- --exact
```

実 MNIST の20更新は、別の評価 IDX を指定して実行する。これらの実データは教材へ同梱しない。

```sh
# check requires=mnist
cargo run --offline --example mnist -- --images "$MNIST_TRAIN_IMAGES" --labels "$MNIST_TRAIN_LABELS" --test-images "$MNIST_TEST_IMAGES" --test-labels "$MNIST_TEST_LABELS" --subset 128 --test-subset 128 --batch 16 --steps 20 --seed 1 --lr 0.01
```

## 20. 初期化 seed、学習率、subset、バッチ順、数学モードをログに残す。

<!-- todo:14-020 -->


実装箇所は `examples/mnist.rs::main` の入力 slot、SGD 更新候補、再 bind と非有限値検査 である。この要件に対する実行可能な確認は `mnist::stable_uniform_cross_entropy` に置く。

`examples/mnist.rs` の `fn main(`の定義を次の全体で置き換えます。

```rust
// file="examples/mnist.rs" mode="replace" item="fn main("
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
	let mut g = Graph::default();
	let x = g.input(0, vec![batch, 784])?;
	let target = g.input(1, vec![batch])?;
	let w1 = g.input(2, vec![32, 784])?;
	let b1 = g.input(3, vec![32])?;
	let w2 = g.input(4, vec![10, 32])?;
	let b2 = g.input(5, vec![10])?;
	let mask = g.input(6, vec![batch])?;
	let h = linear(&mut g, x, w1, Some(b1))?;
	let h = relu(&mut g, h)?;
	let logits = linear(&mut g, h, w2, Some(b2))?;
	let loss = cross_entropy_runtime(&mut g, logits, target, mask)?;
	let grads = gradient(&mut g, loss, &[w1, b1, w2, b2], None)?;
	let mut outputs = vec![loss, logits];
	outputs.extend(grads);
	let plan = g.compile(&outputs)?;
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
	let binding = |dataset: &Mnist, indices: &[usize], weights: &[Buffer]| -> Result<Vec<Buffer>> {
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
		Ok(bindings)
	};
	for step in 0..steps {
		let indices = (0..batch)
			.map(|i| (step * batch + i) % subset)
			.collect::<Vec<_>>();
		let result = plan.run(&binding(&data, &indices, &weights)?)?;
		let value = result[0].data[0];
		if !value.is_finite() {
			return Err(format!("nonfinite loss at step {step}"));
		}
		let mut candidate = weights.clone();
		for (w, grad) in candidate.iter_mut().zip(&result[2..]) {
			for (value, &grad) in w.data.iter_mut().zip(&grad.data) {
				*value -= lr * grad;
				if !value.is_finite() {
					return Err(format!("nonfinite update at step {step}"));
				}
			}
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
			let out = plan.run(&binding(&evaluation, &indices, &weights)?)?;
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
	println!("compiler_calls=1 training_plan_runs={steps}");
	Ok(())
}
```

```sh
# check
cargo test --offline --test mnist stable_uniform_cross_entropy -- --exact
```

## 21. test subset を学習から除き、評価の損失と正解率を別に表示する。

<!-- todo:14-021 -->


実装箇所は `examples/mnist.rs::main` の入力 slot、SGD 更新候補、再 bind と非有限値検査 である。この要件に対する実行可能な確認は `mnist::stable_uniform_cross_entropy` に置く。

`examples/mnist.rs` の `fn main(`の定義を次の全体で置き換えます。

```rust
// file="examples/mnist.rs" mode="replace" item="fn main("
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
	let mut g = Graph::default();
	let x = g.input(0, vec![batch, 784])?;
	let target = g.input(1, vec![batch])?;
	let w1 = g.input(2, vec![32, 784])?;
	let b1 = g.input(3, vec![32])?;
	let w2 = g.input(4, vec![10, 32])?;
	let b2 = g.input(5, vec![10])?;
	let mask = g.input(6, vec![batch])?;
	let h = linear(&mut g, x, w1, Some(b1))?;
	let h = relu(&mut g, h)?;
	let logits = linear(&mut g, h, w2, Some(b2))?;
	let loss = cross_entropy_runtime(&mut g, logits, target, mask)?;
	let grads = gradient(&mut g, loss, &[w1, b1, w2, b2], None)?;
	let mut outputs = vec![loss, logits];
	outputs.extend(grads);
	let plan = g.compile(&outputs)?;
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
	let binding = |dataset: &Mnist, indices: &[usize], weights: &[Buffer]| -> Result<Vec<Buffer>> {
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
		Ok(bindings)
	};
	for step in 0..steps {
		let indices = (0..batch)
			.map(|i| (step * batch + i) % subset)
			.collect::<Vec<_>>();
		let result = plan.run(&binding(&data, &indices, &weights)?)?;
		let value = result[0].data[0];
		if !value.is_finite() {
			return Err(format!("nonfinite loss at step {step}"));
		}
		let mut candidate = weights.clone();
		for (w, grad) in candidate.iter_mut().zip(&result[2..]) {
			for (value, &grad) in w.data.iter_mut().zip(&grad.data) {
				*value -= lr * grad;
				if !value.is_finite() {
					return Err(format!("nonfinite update at step {step}"));
				}
			}
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
			let out = plan.run(&binding(&evaluation, &indices, &weights)?)?;
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
	println!("compiler_calls=1 training_plan_runs={steps}");
	Ok(())
}
```

```sh
# check
cargo test --offline --test mnist stable_uniform_cross_entropy -- --exact
```

## 22. fixture のテストはネットワークと外部 MNIST を必要としない。

<!-- todo:14-022 -->


実装箇所は `data::parse_idx` と `data::idx_fixture` である。この要件に対する実行可能な確認は `mnist::idx_fixture_validation` に置く。

`tests/mnist.rs` の `fn idx_fixture_validation(`の定義を次の全体で置き換えます。

```rust
// file="tests/mnist.rs" mode="replace" item="fn idx_fixture_validation("
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
# check
cargo test --offline --test mnist idx_fixture_validation -- --exact
```

## 23. 同じ環境、同じ seed で loss と重み更新を許容誤差内で再現する。

<!-- todo:14-023 -->


実装箇所は `examples/mnist.rs::main` の入力 slot、SGD 更新候補、再 bind と非有限値検査 である。この要件に対する実行可能な確認は `mnist::sgd_rebind_reuses_plan` に置く。

`tests/mnist.rs` の `fn sgd_rebind_reuses_plan(`の定義を次の全体で置き換えます。

```rust
// file="tests/mnist.rs" mode="replace" item="fn sgd_rebind_reuses_plan("
fn sgd_rebind_reuses_plan() {
	let mut g = Graph::default();
	let w = g.input(0, vec![]).unwrap();
	let x = g.input(1, vec![]).unwrap();
	let target = g.input(2, vec![]).unwrap();
	let prediction = g.binary(Binary::Mul, w, x).unwrap();
	let loss = mse(&mut g, prediction, target).unwrap();
	let dw = gradient(&mut g, loss, &[w], None).unwrap()[0];
	let plan = g.compile(&[loss, dw]).unwrap();
	let scalar = |v| Buffer::new(vec![], vec![v]).unwrap();
	let mut weight = 0f32;
	let mut losses = Vec::new();
	for _ in 0..10 {
		let out = plan.run(&[scalar(weight), scalar(2.), scalar(6.)]).unwrap();
		losses.push(out[0].data[0]);
		weight -= 0.1 * out[1].data[0];
	}
	assert!(losses[9] < losses[0]);
	assert!((weight - 3.).abs() < 1e-5);
	println!(
		"compiler_calls=1 plan_runs=10 initial={} final={}",
		losses[0], losses[9]
	);
}
```

```sh
# check
cargo test --offline --test mnist sgd_rebind_reuses_plan -- --exact
```

## 24. 前向きと backward の graph が実行後にも残る。

<!-- todo:14-024 -->


実装箇所は `examples/mnist.rs::main` の入力 slot、SGD 更新候補、再 bind と非有限値検査 である。この要件に対する実行可能な確認は `mnist::sgd_rebind_reuses_plan` に置く。

`tests/mnist.rs` の `fn sgd_rebind_reuses_plan(`の定義を次の全体で置き換えます。

```rust
// file="tests/mnist.rs" mode="replace" item="fn sgd_rebind_reuses_plan("
fn sgd_rebind_reuses_plan() {
	let mut g = Graph::default();
	let w = g.input(0, vec![]).unwrap();
	let x = g.input(1, vec![]).unwrap();
	let target = g.input(2, vec![]).unwrap();
	let prediction = g.binary(Binary::Mul, w, x).unwrap();
	let loss = mse(&mut g, prediction, target).unwrap();
	let dw = gradient(&mut g, loss, &[w], None).unwrap()[0];
	let plan = g.compile(&[loss, dw]).unwrap();
	let scalar = |v| Buffer::new(vec![], vec![v]).unwrap();
	let mut weight = 0f32;
	let mut losses = Vec::new();
	for _ in 0..10 {
		let out = plan.run(&[scalar(weight), scalar(2.), scalar(6.)]).unwrap();
		losses.push(out[0].data[0]);
		weight -= 0.1 * out[1].data[0];
	}
	assert!(losses[9] < losses[0]);
	assert!((weight - 3.).abs() < 1e-5);
	println!(
		"compiler_calls=1 plan_runs=10 initial={} final={}",
		losses[0], losses[9]
	);
}
```

```sh
# check
cargo test --offline --test mnist sgd_rebind_reuses_plan -- --exact
```

## 25. Plan の再利用中に重みの差替えが結果へ反映される。

<!-- todo:14-025 -->


実装箇所は `examples/mnist.rs::main` の入力 slot、SGD 更新候補、再 bind と非有限値検査 である。この要件に対する実行可能な確認は `mnist::sgd_rebind_reuses_plan` に置く。

`tests/mnist.rs` の `fn sgd_rebind_reuses_plan(`の定義を次の全体で置き換えます。

```rust
// file="tests/mnist.rs" mode="replace" item="fn sgd_rebind_reuses_plan("
fn sgd_rebind_reuses_plan() {
	let mut g = Graph::default();
	let w = g.input(0, vec![]).unwrap();
	let x = g.input(1, vec![]).unwrap();
	let target = g.input(2, vec![]).unwrap();
	let prediction = g.binary(Binary::Mul, w, x).unwrap();
	let loss = mse(&mut g, prediction, target).unwrap();
	let dw = gradient(&mut g, loss, &[w], None).unwrap()[0];
	let plan = g.compile(&[loss, dw]).unwrap();
	let scalar = |v| Buffer::new(vec![], vec![v]).unwrap();
	let mut weight = 0f32;
	let mut losses = Vec::new();
	for _ in 0..10 {
		let out = plan.run(&[scalar(weight), scalar(2.), scalar(6.)]).unwrap();
		losses.push(out[0].data[0]);
		weight -= 0.1 * out[1].data[0];
	}
	assert!(losses[9] < losses[0]);
	assert!((weight - 3.).abs() < 1e-5);
	println!(
		"compiler_calls=1 plan_runs=10 initial={} final={}",
		losses[0], losses[9]
	);
}
```

```sh
# check
cargo test --offline --test mnist sgd_rebind_reuses_plan -- --exact
```

## 26. 非有限な loss が出た step を表示し、成功として扱わない。

<!-- todo:14-026 -->


実装箇所は `examples/mnist.rs::main` の入力 slot、SGD 更新候補、再 bind と非有限値検査 である。この要件に対する実行可能な確認は `mnist::sgd_rebind_reuses_plan` に置く。

`examples/mnist.rs` の `fn main(`の定義を次の全体で置き換えます。

```rust
// file="examples/mnist.rs" mode="replace" item="fn main("
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
	let mut g = Graph::default();
	let x = g.input(0, vec![batch, 784])?;
	let target = g.input(1, vec![batch])?;
	let w1 = g.input(2, vec![32, 784])?;
	let b1 = g.input(3, vec![32])?;
	let w2 = g.input(4, vec![10, 32])?;
	let b2 = g.input(5, vec![10])?;
	let mask = g.input(6, vec![batch])?;
	let h = linear(&mut g, x, w1, Some(b1))?;
	let h = relu(&mut g, h)?;
	let logits = linear(&mut g, h, w2, Some(b2))?;
	let loss = cross_entropy_runtime(&mut g, logits, target, mask)?;
	let grads = gradient(&mut g, loss, &[w1, b1, w2, b2], None)?;
	let mut outputs = vec![loss, logits];
	outputs.extend(grads);
	let plan = g.compile(&outputs)?;
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
	let binding = |dataset: &Mnist, indices: &[usize], weights: &[Buffer]| -> Result<Vec<Buffer>> {
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
		Ok(bindings)
	};
	for step in 0..steps {
		let indices = (0..batch)
			.map(|i| (step * batch + i) % subset)
			.collect::<Vec<_>>();
		let result = plan.run(&binding(&data, &indices, &weights)?)?;
		let value = result[0].data[0];
		if !value.is_finite() {
			return Err(format!("nonfinite loss at step {step}"));
		}
		let mut candidate = weights.clone();
		for (w, grad) in candidate.iter_mut().zip(&result[2..]) {
			for (value, &grad) in w.data.iter_mut().zip(&grad.data) {
				*value -= lr * grad;
				if !value.is_finite() {
					return Err(format!("nonfinite update at step {step}"));
				}
			}
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
			let out = plan.run(&binding(&evaluation, &indices, &weights)?)?;
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
	println!("compiler_calls=1 training_plan_runs={steps}");
	Ok(())
}
```

```sh
# check
cargo test --offline --test mnist sgd_rebind_reuses_plan -- --exact
```
