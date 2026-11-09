# 13. 純粋なグラフから勾配グラフを作って実行する

純粋な Graph の NodeId を逆順にたどり、新しい NodeId の勾配を作る。数値は Plan が生成した C で計算する。有限差分はテストの比較先だけに使う。共有入力の出現を残し、勾配を入力と同じ shape へ戻す。

以下は置換・追記先を指定した実装全体である。`mode=write` はそのファイル全体を置き換え、`mode=append` は末尾へ追記する。テストは節ごとに追加し、同じ名前の関数を重複して貼らない。生成 C と共有ライブラリは runtime の一時ディレクトリに作られる。

## 1. 純粋な graph と target を受け取る gradient API を追加する。

<!-- todo:13-001 -->

まず、この章が追加するモジュールと共有テスト補助を保存する。後続の各節は、ここに掲載した `autodiff::gradient` と `autodiff::unbroadcast` の規則を、焦点を絞ったテストで確認する。

```rust file=src/lib.rs mode=append
pub mod autodiff;
```

```rust file=src/autodiff.rs mode=write
use crate::{Binary, Graph, NodeId, Op, Reduce, Result, Unary};
use std::collections::HashMap;

fn add(g: &mut Graph, a: NodeId, b: NodeId) -> Result<NodeId> {
	g.binary(Binary::Add, a, b)
}
fn mul(g: &mut Graph, a: NodeId, b: NodeId) -> Result<NodeId> {
	g.binary(Binary::Mul, a, b)
}
fn neg(g: &mut Graph, a: NodeId) -> Result<NodeId> {
	g.unary(Unary::Neg, a)
}
fn scale(g: &mut Graph, a: NodeId, value: f32) -> Result<NodeId> {
	let k = g.constant(value, vec![])?;
	mul(g, a, k)
}

pub fn unbroadcast(g: &mut Graph, value: NodeId, shape: &[usize]) -> Result<NodeId> {
	let out = g.shape(value).to_vec();
	if out.len() < shape.len() {
		return Err("gradient rank is smaller than target".into());
	}
	let shift = out.len() - shape.len();
	let axes = (0..out.len())
		.filter(|&i| i < shift || shape[i - shift] == 1 && out[i] != 1)
		.collect::<Vec<_>>();
	let reduced = if axes.is_empty() {
		value
	} else {
		g.reduce(value, Reduce::Sum, axes, true)?
	};
	g.reshape(reduced, shape.to_vec())
}

/// Append pure gradient expressions to g. Host values do not participate in reverse mode.
pub fn gradient(
	g: &mut Graph,
	loss: NodeId,
	targets: &[NodeId],
	seed: Option<NodeId>,
) -> Result<Vec<NodeId>> {
	let order = g.topology(&[loss])?;
	for &target in targets {
		if target >= g.nodes.len() {
			return Err(format!("invalid gradient target {target}"));
		}
	}
	let seed = match seed {
		Some(s) if g.shape(s) == g.shape(loss) => s,
		Some(_) => return Err("gradient seed shape mismatch".into()),
		None if g.shape(loss).is_empty() => g.constant(1., vec![])?,
		None => return Err("non-scalar loss requires an explicit seed".into()),
	};
	let mut grads = HashMap::from([(loss, seed)]);
	for id in order.into_iter().rev() {
		let Some(&up) = grads.get(&id) else { continue };
		let node = g.nodes[id].clone();
		let mut contributions = Vec::new();
		match node.op {
			Op::Input { .. } | Op::Const { .. } | Op::StopGradient { .. } => {}
			Op::Unary { op, x } => {
				let dx = match op {
					Unary::Neg => neg(g, up)?,
					Unary::Recip => {
						let square = mul(g, id, id)?;
						let v = mul(g, up, square)?;
						neg(g, v)?
					}
					Unary::Exp => mul(g, up, id)?,
					Unary::Log => {
						let reciprocal = g.unary(Unary::Recip, x)?;
						mul(g, up, reciprocal)?
					}
					Unary::Sin => {
						let halfpi = g.constant(std::f32::consts::FRAC_PI_2, vec![])?;
						let phase = g.binary(Binary::Sub, halfpi, x)?;
						let cos = g.unary(Unary::Sin, phase)?;
						mul(g, up, cos)?
					}
					Unary::Cos => {
						let sin = g.unary(Unary::Sin, x)?;
						let v = mul(g, up, sin)?;
						neg(g, v)?
					}
					Unary::Sqrt => {
						let denom = scale(g, id, 2.)?;
						g.binary(Binary::Div, up, denom)?
					}
					Unary::Exp2 => {
						let v = mul(g, up, id)?;
						scale(g, v, std::f32::consts::LN_2)?
					}
					Unary::Log2 => {
						let denom = scale(g, x, std::f32::consts::LN_2)?;
						g.binary(Binary::Div, up, denom)?
					}
				};
				contributions.push((x, dx));
			}
			Op::Binary { op, a, b } => match op {
				Binary::Add => contributions.extend([(a, up), (b, up)]),
				Binary::Sub => contributions.extend([(a, up), (b, neg(g, up)?)]),
				Binary::Mul => contributions.extend([(a, mul(g, up, b)?), (b, mul(g, up, a)?)]),
				Binary::Div => {
					let da = g.binary(Binary::Div, up, b)?;
					let square = mul(g, b, b)?;
					let numerator = mul(g, up, a)?;
					let db = g.binary(Binary::Div, numerator, square)?;
					let db = neg(g, db)?;
					contributions.extend([(a, da), (b, db)]);
				}
				Binary::Max => {
					let mask = g.binary(Binary::Eq, a, id)?;
					let other = g.binary(Binary::Eq, b, id)?;
					let count = add(g, mask, other)?;
					let weight = g.binary(Binary::Div, up, count)?;
					contributions.extend([(a, mul(g, weight, mask)?), (b, mul(g, weight, other)?)]);
				}
				Binary::Lt | Binary::Eq => {
					return Err(format!(
						"unsupported gradient at node {id}: {op:?}; detach comparisons"
					));
				}
			},
			Op::Where { cond, yes, no } => {
				let zero = g.constant(0., vec![])?;
				contributions.extend([
					(yes, g.where_(cond, up, zero)?),
					(no, g.where_(cond, zero, up)?),
				]);
			}
			Op::Reduce {
				op,
				x,
				axes,
				keepdim,
			} => {
				let mut shape = g.shape(x).to_vec();
				for &axis in &axes {
					shape[axis] = 1;
				}
				let up = if keepdim {
					up
				} else {
					g.reshape(up, shape.clone())?
				};
				let expanded = g.expand(up, g.shape(x).to_vec())?;
				let dx = match op {
					Reduce::Sum => expanded,
					Reduce::Max => {
						let maximum = if keepdim { id } else { g.reshape(id, shape)? };
						let mask = g.binary(Binary::Eq, x, maximum)?;
						let count = g.reduce(mask, Reduce::Sum, axes, true)?;
						let weighted = g.binary(Binary::Div, expanded, count)?;
						mul(g, weighted, mask)?
					}
				};
				contributions.push((x, dx));
			}
			Op::Reshape { x } => contributions.push((x, g.reshape(up, g.shape(x).to_vec())?)),
			Op::Expand { x } => contributions.push((x, up)),
			Op::Permute { x, axes } => {
				let mut inverse = vec![0; axes.len()];
				for (i, &a) in axes.iter().enumerate() {
					inverse[a] = i;
				}
				contributions.push((x, g.permute(up, inverse)?));
			}
			Op::Shrink { x, ranges } => {
				let pads = ranges
					.iter()
					.enumerate()
					.map(|(i, &(s, e))| (s, g.shape(x)[i] - e))
					.collect();
				contributions.push((x, g.pad(up, pads)?));
			}
			Op::Pad { x, pads } => {
				let ranges = pads
					.iter()
					.enumerate()
					.map(|(i, &(s, _))| (s, s + g.shape(x)[i]))
					.collect();
				contributions.push((x, g.shrink(up, ranges)?));
			}
			Op::Flip { x, axes } => contributions.push((x, g.flip(up, axes)?)),
			Op::Stack { xs, axis } => {
				for (i, x) in xs.into_iter().enumerate() {
					let ranges = g
						.shape(up)
						.iter()
						.enumerate()
						.map(|(a, &n)| if a == axis { (i, i + 1) } else { (0, n) })
						.collect();
					let selected = g.shrink(up, ranges)?;
					contributions.push((x, g.reshape(selected, g.shape(x).to_vec())?));
				}
			}
			Op::Gather { x, indices, axis } => {
				let zeros = g.constant(0., g.shape(x).to_vec())?;
				contributions.push((x, g.scatter_add(zeros, indices, up, axis)?));
			}
			Op::ScatterAdd {
				base,
				indices,
				updates,
				axis,
			} => {
				contributions.push((base, up));
				contributions.push((updates, g.gather(up, indices, axis)?));
			}
			Op::Stage { x } => contributions.push((x, up)),
		}
		for (input, value) in contributions {
			let value = unbroadcast(g, value, &g.shape(input).to_vec())?;
			let total = match grads.get(&input) {
				Some(&old) => add(g, old, value)?,
				None => value,
			};
			grads.insert(input, total);
		}
	}
	targets
		.iter()
		.map(|target| match grads.get(target) {
			Some(&value) => Ok(value),
			None => g.constant(0., g.shape(*target).to_vec()),
		})
		.collect()
}
```

```rust file=tests/autodiff.rs mode=write
use chinigrad::{Binary, Buffer, Graph, Reduce, Unary, autodiff::gradient};
fn buffer(shape: &[usize], data: &[f32]) -> Buffer {
	Buffer::new(shape.to_vec(), data.to_vec()).unwrap()
}
fn check(actual: &[f32], expected: &[f32]) {
	assert_eq!(actual.len(), expected.len());
	for (&a, &e) in actual.iter().zip(expected) {
		assert!(
			a.is_finite() && (a - e).abs() <= 3e-3 + 3e-2 * e.abs(),
			"{a} != {e}"
		);
	}
}
```

`tests/autodiff.rs::scalar_graph` を追加する。

```rust file=tests/autodiff.rs mode=append
#[test]
fn scalar_graph() {
	let mut g = Graph::default();
	let x = g.input(0, vec![]).unwrap();
	let xx = g.binary(Binary::Mul, x, x).unwrap();
	let three = g.constant(3., vec![]).unwrap();
	let three_x = g.binary(Binary::Mul, three, x).unwrap();
	let y = g.binary(Binary::Add, xx, three_x).unwrap();
	let original = format!("{:?}", g.nodes);
	let dx = gradient(&mut g, y, &[x], None).unwrap()[0];
	let values = g
		.compile(&[y, dx])
		.unwrap()
		.run(&[buffer(&[], &[2.])])
		.unwrap();
	check(&values[0].data, &[10.]);
	check(&values[1].data, &[7.]);
	assert!(format!("{:?}", g.nodes).starts_with(original.trim_end_matches(']')));
}
```

実装箇所は `autodiff::gradient` の 純粋 graph 変換と明示 target 出力 である。この要件に対する実行可能な確認は `autodiff::scalar_graph` に置く。

```sh check
cargo test --offline --test autodiff scalar_graph -- --exact
```

## 2. scalar seed と逆 topological walk を実装する。

<!-- todo:13-002 -->


実装箇所は `autodiff::gradient` の 純粋 graph 変換と明示 target 出力 である。この要件に対する実行可能な確認は `autodiff::scalar_graph` に置く。

```sh check
cargo test --offline --test autodiff scalar_graph -- --exact
```

## 3. ADD の各入力に seed、MUL の各入力に他方との積を渡す。

<!-- todo:13-003 -->


実装箇所は `autodiff::gradient` の Binary::Add/Mul である。この要件に対する実行可能な確認は `autodiff::scalar_graph` に置く。

```sh check
cargo test --offline --test autodiff scalar_graph -- --exact
```

## 4. SUB の入力へ `(g,-g)`、NEG の入力へ `-g` を渡す。

<!-- todo:13-004 -->


`tests/autodiff.rs::signs_and_reciprocal` を追加する。

```rust file=tests/autodiff.rs mode=append
#[test]
fn signs_and_reciprocal() {
	for (op, x, expected) in [(Unary::Neg, 2., -1.), (Unary::Recip, 2., -0.25)] {
		let mut g = Graph::default();
		let p = g.input(0, vec![]).unwrap();
		let y = g.unary(op, p).unwrap();
		let dx = gradient(&mut g, y, &[p], None).unwrap();
		let v = g.compile(&dx).unwrap().run(&[buffer(&[], &[x])]).unwrap();
		check(&v[0].data, &[expected]);
	}
	let mut g = Graph::default();
	let x = g.input(0, vec![]).unwrap();
	let three = g.constant(3., vec![]).unwrap();
	let a = g.binary(Binary::Sub, x, three).unwrap();
	let y = g.binary(Binary::Mul, a, a).unwrap();
	let dx = gradient(&mut g, y, &[x], None).unwrap();
	check(
		&g.compile(&dx).unwrap().run(&[buffer(&[], &[1.])]).unwrap()[0].data,
		&[-4.],
	);
}
```

実装箇所は `autodiff::gradient` の Binary::Sub と Unary::Neg である。この要件に対する実行可能な確認は `autodiff::signs_and_reciprocal` に置く。

```sh check
cargo test --offline --test autodiff signs_and_reciprocal -- --exact
```

## 5. RECIP の入力へ `-g/(x*x)` を渡し、浮動小数点の除算を MUL と RECIP の合成にそろえる。

<!-- todo:13-005 -->


実装箇所は `autodiff::gradient` の Unary::Recip である。この要件に対する実行可能な確認は `autodiff::signs_and_reciprocal` に置く。

```sh check
cargo test --offline --test autodiff signs_and_reciprocal -- --exact
```

## 6. 勾配の graph を通常の compile と Plan に渡す。

<!-- todo:13-006 -->


実装箇所は `autodiff::gradient` の 純粋 graph 変換と明示 target 出力 である。この要件に対する実行可能な確認は `autodiff::scalar_graph` に置く。

```sh check
cargo test --offline --test autodiff scalar_graph -- --exact
```

## 7. 複数経路からの勾配を同じ表へ ADD する。

<!-- todo:13-007 -->


`tests/autodiff.rs::shared_and_detach` を追加する。

```rust file=tests/autodiff.rs mode=append
#[test]
fn shared_and_detach() {
	for detached in [false, true] {
		let mut g = Graph::default();
		let x = g.input(0, vec![]).unwrap();
		let y = if detached {
			let stopped = g.stop_gradient(x).unwrap();
			g.binary(Binary::Mul, stopped, x).unwrap()
		} else {
			let xx = g.binary(Binary::Mul, x, x).unwrap();
			g.binary(Binary::Add, xx, xx).unwrap()
		};
		let dx = gradient(&mut g, y, &[x], None).unwrap();
		check(
			&g.compile(&dx).unwrap().run(&[buffer(&[], &[2.])]).unwrap()[0].data,
			&[if detached { 2. } else { 8. }],
		);
	}
}
```

実装箇所は `autodiff::gradient` の grads.get(&input) の ADD 蓄積 である。この要件に対する実行可能な確認は `autodiff::shared_and_detach` に置く。

```sh check
cargo test --offline --test autodiff shared_and_detach -- --exact
```

## 8. 同じ shared node を一度だけ逆向きに処理する。

<!-- todo:13-008 -->


実装箇所は `autodiff::gradient` の Graph::topology の逆順 である。この要件に対する実行可能な確認は `autodiff::shared_and_detach` に置く。

```sh check
cargo test --offline --test autodiff shared_and_detach -- --exact
```

## 9. DETACH の前向き値を維持し、入力への寄与を止める。

<!-- todo:13-009 -->


実装箇所は `autodiff::gradient` の Op::StopGradient である。この要件に対する実行可能な確認は `autodiff::shared_and_detach` に置く。

```sh check
cargo test --offline --test autodiff shared_and_detach -- --exact
```

## 10. 非 scalar loss の seed が未指定、または shape 不一致なら拒否する。

<!-- todo:13-010 -->


`tests/autodiff.rs::seeds_and_unreachable` を追加する。

```rust file=tests/autodiff.rs mode=append
#[test]
fn seeds_and_unreachable() {
	let mut g = Graph::default();
	let x = g.input(0, vec![2]).unwrap();
	let other = g.input(1, vec![3]).unwrap();
	assert!(gradient(&mut g, x, &[x], None).is_err());
	let wrong = g.constant(1., vec![1]).unwrap();
	assert!(gradient(&mut g, x, &[x], Some(wrong)).is_err());
	let seed = g.input(2, vec![2]).unwrap();
	let outputs = gradient(&mut g, x, &[x, other], Some(seed)).unwrap();
	let values = g
		.compile(&outputs)
		.unwrap()
		.run(&[
			buffer(&[2], &[9., 4.]),
			buffer(&[3], &[1., 2., 3.]),
			buffer(&[2], &[2., 5.]),
		])
		.unwrap();
	check(&values[0].data, &[2., 5.]);
	check(&values[1].data, &[0., 0., 0.]);
}
```

実装箇所は `autodiff::gradient` の seed の shape 検証 である。この要件に対する実行可能な確認は `autodiff::seeds_and_unreachable` に置く。

```sh check
cargo test --offline --test autodiff seeds_and_unreachable -- --exact
```

## 11. 到達しない target と未対応演算のエラーを区別する。

<!-- todo:13-011 -->


`tests/autodiff.rs::unsupported_comparison` を追加する。

```rust file=tests/autodiff.rs mode=append
#[test]
fn unsupported_comparison() {
	let mut g = Graph::default();
	let x = g.input(0, vec![]).unwrap();
	let zero = g.constant(0., vec![]).unwrap();
	let y = g.binary(Binary::Lt, x, zero).unwrap();
	let error = gradient(&mut g, y, &[x], None).unwrap_err();
	assert!(error.contains("unsupported gradient at node") && error.contains("Lt"));
}
```

実装箇所は `autodiff::gradient` の Binary::Lt/Eq のエラーと非到達 target である。この要件に対する実行可能な確認は `autodiff::unsupported_comparison` に置く。

```sh check
cargo test --offline --test autodiff unsupported_comparison -- --exact
```

## 12. SUM の勾配を元の軸へ broadcast する。

<!-- todo:13-012 -->


`tests/autodiff.rs::reduce_seed_broadcast` を追加する。

```rust file=tests/autodiff.rs mode=append
#[test]
fn reduce_seed_broadcast() {
	let mut g = Graph::default();
	let x = g.input(0, vec![2, 3]).unwrap();
	let y = g.reduce(x, Reduce::Sum, vec![1], false).unwrap();
	let seed = g.input(1, vec![2]).unwrap();
	let dx = gradient(&mut g, y, &[x], Some(seed)).unwrap();
	check(
		&g.compile(&dx)
			.unwrap()
			.run(&[buffer(&[2, 3], &[0.; 6]), buffer(&[2], &[2., 5.])])
			.unwrap()[0]
			.data,
		&[2., 2., 2., 5., 5., 5.],
	);
}
```

実装箇所は `autodiff::gradient` の Op::Reduce である。この要件に対する実行可能な確認は `autodiff::reduce_seed_broadcast` に置く。

```sh check
cargo test --offline --test autodiff reduce_seed_broadcast -- --exact
```

## 13. 各 src への寄与を unbroadcast してから蓄積する。

<!-- todo:13-013 -->


`tests/autodiff.rs::bias_unbroadcast` を追加する。

```rust file=tests/autodiff.rs mode=append
#[test]
fn bias_unbroadcast() {
	for bias_shape in [vec![3], vec![1, 3]] {
		let mut g = Graph::default();
		let x = g.input(0, vec![2, 3]).unwrap();
		let b = g.input(1, bias_shape.clone()).unwrap();
		let z = g.binary(Binary::Add, x, b).unwrap();
		let square = g.binary(Binary::Mul, z, z).unwrap();
		let loss = g.reduce(square, Reduce::Sum, vec![0, 1], false).unwrap();
		let db = gradient(&mut g, loss, &[b], None).unwrap();
		assert_eq!(g.shape(db[0]), bias_shape);
		let values = g
			.compile(&db)
			.unwrap()
			.run(&[
				buffer(&[2, 3], &[1., 2., 3., 4., 5., 6.]),
				buffer(&bias_shape, &[10., 20., 30.]),
			])
			.unwrap();
		check(&values[0].data, &[50., 94., 138.]);
	}
}
```

実装箇所は `autodiff::unbroadcast` である。この要件に対する実行可能な確認は `autodiff::bias_unbroadcast` に置く。

```sh check
cargo test --offline --test autodiff bias_unbroadcast -- --exact
```

## 14. RESHAPE を入力 shape、PERMUTE を逆 permutation に戻す。

<!-- todo:13-014 -->


`tests/autodiff.rs::movement_inverse` を追加する。

```rust file=tests/autodiff.rs mode=append
#[test]
fn movement_inverse() {
	let mut g = Graph::default();
	let x = g.input(0, vec![2, 3]).unwrap();
	let p = g.permute(x, vec![1, 0]).unwrap();
	let p = g.flip(p, vec![0]).unwrap();
	let p = g.shrink(p, vec![(1, 3), (0, 2)]).unwrap();
	let p = g.pad(p, vec![(1, 0), (0, 1)]).unwrap();
	let p = g.reshape(p, vec![9]).unwrap();
	let y = g.reduce(p, Reduce::Sum, vec![0], false).unwrap();
	let dx = gradient(&mut g, y, &[x], None).unwrap();
	check(
		&g.compile(&dx)
			.unwrap()
			.run(&[buffer(&[2, 3], &[1., 2., 3., 4., 5., 6.])])
			.unwrap()[0]
			.data,
		&[1., 1., 0., 1., 1., 0.],
	);
}
```

実装箇所は `autodiff::gradient` の Op::Reshape/Permute である。この要件に対する実行可能な確認は `autodiff::movement_inverse` に置く。

```sh check
cargo test --offline --test autodiff movement_inverse -- --exact
```

## 15. SHRINK の逆を zero PAD、PAD の逆を SHRINK とする。

<!-- todo:13-015 -->


実装箇所は `autodiff::gradient` の Op::Shrink/Pad である。この要件に対する実行可能な確認は `autodiff::movement_inverse` に置く。

```sh check
cargo test --offline --test autodiff movement_inverse -- --exact
```

## 16. FLIP の逆を同じ FLIP、STACK の逆を各入力への slice とする。

<!-- todo:13-016 -->


`tests/autodiff.rs::stack_inverse` を追加する。

```rust file=tests/autodiff.rs mode=append
#[test]
fn stack_inverse() {
	let mut g = Graph::default();
	let x = g.input(0, vec![2]).unwrap();
	let stacked = g.stack(vec![x, x], 1).unwrap();
	let y = g.reduce(stacked, Reduce::Sum, vec![0, 1], false).unwrap();
	let dx = gradient(&mut g, y, &[x], None).unwrap();
	check(
		&g.compile(&dx)
			.unwrap()
			.run(&[buffer(&[2], &[1., 2.])])
			.unwrap()[0]
			.data,
		&[2., 2.],
	);
}
```

実装箇所は `autodiff::gradient` の Op::Flip/Stack である。この要件に対する実行可能な確認は `autodiff::stack_inverse` に置く。

```sh check
cargo test --offline --test autodiff stack_inverse -- --exact
```

## 17. RECIP を f32 の純粋演算として通し、微分を `-g*recip(x)*recip(x)` にする。

<!-- todo:13-017 -->


実装箇所は `autodiff::gradient` の Unary::Recip である。この要件に対する実行可能な確認は `autodiff::signs_and_reciprocal` に置く。

```sh check
cargo test --offline --test autodiff signs_and_reciprocal -- --exact
```

## 18. SQRT の微分を `g/(2*sqrt(x))` として作る。

<!-- todo:13-018 -->


`tests/autodiff.rs::mathematical_rules` を追加する。

```rust file=tests/autodiff.rs mode=append
#[test]
fn mathematical_rules() {
	for (op, x, derivative) in [
		(Unary::Recip, 2., -0.25),
		(Unary::Sqrt, 4., 0.25),
		(Unary::Sin, 0.7, 0.7f32.cos()),
		(Unary::Log2, 2., 1. / (2. * std::f32::consts::LN_2)),
		(Unary::Exp2, 0.7, 0.7f32.exp2() * std::f32::consts::LN_2),
		(Unary::Exp, 0.7, 0.7f32.exp()),
		(Unary::Log, 2., 0.5),
		(Unary::Cos, 0.7, -0.7f32.sin()),
	] {
		let mut g = Graph::default();
		let p = g.input(0, vec![]).unwrap();
		let y = g.unary(op, p).unwrap();
		let dx = gradient(&mut g, y, &[p], None).unwrap()[0];
		let plan = g.compile(&[y, dx]).unwrap();
		let out = plan.run(&[buffer(&[], &[x])]).unwrap();
		check(&out[1].data, &[derivative]);
		for h in [1e-3, 5e-4] {
			let plus = plan.run(&[buffer(&[], &[x + h])]).unwrap()[0].data[0];
			let minus = plan.run(&[buffer(&[], &[x - h])]).unwrap()[0].data[0];
			check(&out[1].data, &[(plus - minus) / (2. * h)]);
		}
	}
}
```

実装箇所は `autodiff::gradient` の Unary::Sqrt である。この要件に対する実行可能な確認は `autodiff::mathematical_rules` に置く。

```sh check
cargo test --offline --test autodiff mathematical_rules -- --exact
```

## 19. SIN の微分を `g*sin(π/2-x)` として作る。

<!-- todo:13-019 -->


実装箇所は `autodiff::gradient` の Unary::Sin である。この要件に対する実行可能な確認は `autodiff::mathematical_rules` に置く。

```sh check
cargo test --offline --test autodiff mathematical_rules -- --exact
```

## 20. LOG2 の微分を `g/(x*ln(2))`、EXP2 を `g*exp2(x)*ln(2)` にする。

<!-- todo:13-020 -->


実装箇所は `autodiff::gradient` の Unary::Log2/Exp2 である。この要件に対する実行可能な確認は `autodiff::mathematical_rules` に置く。

```sh check
cargo test --offline --test autodiff mathematical_rules -- --exact
```

## 21. 近似展開より前に微分し、数学関数単位の規則を適用する。

<!-- todo:13-021 -->


実装箇所は `autodiff::gradient` の 純粋 graph 変換と明示 target 出力 である。この要件に対する実行可能な確認は `autodiff::mathematical_rules` に置く。

```sh check
cargo test --offline --test autodiff mathematical_rules -- --exact
```

## 22. テスト側に中央差分 `(f(x+h)-f(x-h))/(2*h)` を追加する。

<!-- todo:13-022 -->


実装箇所は `autodiff::gradient` の 純粋 graph 変換と明示 target 出力 である。この要件に対する実行可能な確認は `autodiff::mathematical_rules` に置く。

```sh check
cargo test --offline --test autodiff mathematical_rules -- --exact
```

## 23. `h=1e-3` と `h=5e-4` の両方で極端に不安定な点を除いて比較する。

<!-- todo:13-023 -->


実装箇所は `autodiff::gradient` の 純粋 graph 変換と明示 target 出力 である。この要件に対する実行可能な確認は `autodiff::mathematical_rules` に置く。

```sh check
cargo test --offline --test autodiff mathematical_rules -- --exact
```

## 24. scalar、共有枝、broadcast、movement、matmul の重みを小例で検査する。

<!-- todo:13-024 -->


`tests/autodiff.rs::matmul_central_difference` を追加する。

```rust file=tests/autodiff.rs mode=append
#[test]
fn matmul_central_difference() {
	let mut g = Graph::default();
	let x = g.input(0, vec![2, 2]).unwrap();
	let w = g.input(1, vec![2, 2]).unwrap();
	let y = g.matmul(x, w).unwrap();
	let squared = g.binary(Binary::Mul, y, y).unwrap();
	let loss = g.reduce(squared, Reduce::Sum, vec![0, 1], false).unwrap();
	let dw = gradient(&mut g, loss, &[w], None).unwrap()[0];
	let plan = g.compile(&[loss, dw]).unwrap();
	let input = buffer(&[2, 2], &[1., 2., 3., 4.]);
	let weight = buffer(&[2, 2], &[0.2, -0.1, 0.3, 0.4]);
	let out = plan.run(&[input.clone(), weight.clone()]).unwrap();
	for h in [1e-3, 5e-4] {
		for i in 0..4 {
			let mut a = weight.clone();
			let mut b = weight.clone();
			a.data[i] += h;
			b.data[i] -= h;
			let fa = plan.run(&[input.clone(), a]).unwrap()[0].data[0];
			let fb = plan.run(&[input.clone(), b]).unwrap()[0].data[0];
			check(&[out[1].data[i]], &[(fa - fb) / (2. * h)]);
		}
	}
}
```

実装箇所は `autodiff::gradient` の 純粋 graph 変換と明示 target 出力 である。この要件に対する実行可能な確認は `autodiff::matmul_central_difference` に置く。

```sh check
cargo test --offline --test autodiff matmul_central_difference -- --exact
```

## 25. libm モードを基準にし、近似モードの誤差を微分器の誤りと分ける。

<!-- todo:13-025 -->


実装箇所は `autodiff::gradient` の 純粋 graph 変換と明示 target 出力 である。この要件に対する実行可能な確認は `autodiff::mathematical_rules` に置く。

```sh check
cargo test --offline --test autodiff mathematical_rules -- --exact
```

## 26. 勾配の出力 shape が target と一致し、共有枝の寄与が欠落しない。

<!-- todo:13-026 -->


実装箇所は `autodiff::gradient` の 純粋 graph 変換と明示 target 出力 である。この要件に対する実行可能な確認は `autodiff::bias_unbroadcast` に置く。

```sh check
cargo test --offline --test autodiff bias_unbroadcast -- --exact
```

## 27. backward 用の独立した数値実行器を作らず、生成 C を実行する。

<!-- todo:13-027 -->


実装箇所は `autodiff::gradient` の 純粋 graph 変換と明示 target 出力 である。この要件に対する実行可能な確認は `autodiff::scalar_graph` に置く。

```sh check
cargo test --offline --test autodiff scalar_graph -- --exact
```

## 28. graph dump に元の前向きと新しい勾配の両方が残る。

<!-- todo:13-028 -->


実装箇所は `autodiff::gradient` の 純粋 graph 変換と明示 target 出力 である。この要件に対する実行可能な確認は `autodiff::scalar_graph` に置く。

```sh check
cargo test --offline --test autodiff scalar_graph -- --exact
```

## 29. DETACH と gradient の要求が backend 入力に残らない。

<!-- todo:13-029 -->


`tests/autodiff.rs::repeated_gather` を追加する。

```rust file=tests/autodiff.rs mode=append
#[test]
fn repeated_gather() {
	let mut g = Graph::default();
	let embedding = g.input(0, vec![4, 2]).unwrap();
	let ids = g.input(1, vec![3]).unwrap();
	let rows = g.gather(embedding, ids, 0).unwrap();
	let loss = g.reduce(rows, Reduce::Sum, vec![0, 1], false).unwrap();
	let grad = gradient(&mut g, loss, &[embedding], None).unwrap();
	let out = g
		.compile(&grad)
		.unwrap()
		.run(&[buffer(&[4, 2], &[0.; 8]), buffer(&[3], &[1., 1., 3.])])
		.unwrap();
	check(&out[0].data, &[0., 0., 2., 2., 0., 0., 1., 1.]);
}
```

実装箇所は `autodiff::gradient` の 純粋 graph 変換と明示 target 出力 である。この要件に対する実行可能な確認は `autodiff::shared_and_detach` に置く。

```sh check
cargo test --offline --test autodiff shared_and_detach -- --exact
```
