# 10. 多次元の座標変換で movement を実行する

movement は座標の変換として lower する。pad と stack は必要な branch の中だけで入力を読む。空 shape の出力 loop は実行されないため座標の除算も実行されない。後の embedding とその勾配に備えて gather と scatter-add も同じ座標経路に含める。indices は f32 に正確に表せる非負整数に限定し、index 部分グラフを検証してから FFI を呼ぶ。

`file=... mode=write` はそのファイルの全置換、`mode=append` は末尾への追記である。すべて前章でできたプロジェクトの root で行う。`# check` は実際の検証コマンドであり、ソースファイルには保存しない。

## 1. 出力の二次元 RANGE と row-major の平坦化を追加する。
<!-- todo:10-001 -->

```rust
// file=src/graph.rs mode=append
impl Graph {
	pub fn reshape(&mut self, x: NodeId, shape: Vec<usize>) -> Result<NodeId> {
		self.insert(Op::Reshape { x }, shape)
	}
	pub fn permute(&mut self, x: NodeId, axes: Vec<usize>) -> Result<NodeId> {
		let input = &self.get(x)?.shape;
		axes_valid(&axes, input.len())?;
		let shape = axes.iter().map(|&a| input[a]).collect();
		self.insert(Op::Permute { x, axes }, shape)
	}
	pub fn shrink(&mut self, x: NodeId, ranges: Vec<(usize, usize)>) -> Result<NodeId> {
		let shape = ranges
			.iter()
			.map(|&(a, b)| b.checked_sub(a).ok_or("negative shrink range".into()))
			.collect::<Result<_>>()?;
		self.insert(Op::Shrink { x, ranges }, shape)
	}
	pub fn pad(&mut self, x: NodeId, pads: Vec<(usize, usize)>) -> Result<NodeId> {
		let input = &self.get(x)?.shape;
		if pads.len() != input.len() {
			return Err("pad rank".into());
		}
		let shape = pads
			.iter()
			.zip(input)
			.map(|(&(a, b), &n)| {
				n.checked_add(a)
					.and_then(|n| n.checked_add(b))
					.ok_or("pad overflow".into())
			})
			.collect::<Result<_>>()?;
		self.insert(Op::Pad { x, pads }, shape)
	}
	pub fn flip(&mut self, x: NodeId, axes: Vec<usize>) -> Result<NodeId> {
		self.insert(Op::Flip { x, axes }, self.get(x)?.shape.clone())
	}
	pub fn stack(&mut self, xs: Vec<NodeId>, axis: usize) -> Result<NodeId> {
		let mut shape = self.get(*xs.first().ok_or("empty stack")?)?.shape.clone();
		if axis > shape.len() {
			return Err("stack axis".into());
		}
		shape.insert(axis, xs.len());
		self.insert(Op::Stack { xs, axis }, shape)
	}
	pub fn gather(&mut self, x: NodeId, indices: NodeId, axis: usize) -> Result<NodeId> {
		let mut shape = self.get(x)?.shape.clone();
		if axis >= shape.len() || self.get(indices)?.shape.len() != 1 {
			return Err("gather axis/rank".into());
		}
		shape[axis] = self.shape(indices)[0];
		self.insert(Op::Gather { x, indices, axis }, shape)
	}
	pub fn scatter_add(
		&mut self,
		base: NodeId,
		indices: NodeId,
		updates: NodeId,
		axis: usize,
	) -> Result<NodeId> {
		self.insert(
			Op::ScatterAdd {
				base,
				indices,
				updates,
				axis,
			},
			self.get(base)?.shape.clone(),
		)
	}
}
```

```rust
// file=tests/movement.rs mode=write
use chibigrad::{Binary, Buffer, Graph, Reduce, Result, Unary};
fn buffer(s: &[usize], d: &[f32]) -> Buffer {
	Buffer::new(s.to_vec(), d.to_vec()).unwrap()
}
fn compare(g: &Graph, outs: &[usize], inputs: &[Buffer]) -> Result<Vec<Buffer>> {
	let actual = g.compile(outs)?.run(inputs)?;
	let reference = g.interpret(outs, inputs)?;
	for (a, b) in actual.iter().zip(reference) {
		assert_eq!(a.shape, b.shape);
		for (&x, &y) in a.data.iter().zip(&b.data) {
			assert!(
				x.is_nan() && y.is_nan() || (x - y).abs() <= 1e-5 + 1e-5 * y.abs(),
				"{x} != {y}"
			);
		}
	}
	Ok(actual)
}

#[test]
fn movement() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![2, 3])?;
	let t = g.permute(x, vec![1, 0])?;
	let r = g.reshape(t, vec![6])?;
	let f = g.flip(r, vec![0])?;
	let p = g.pad(f, vec![(1, 2)])?;
	let s = g.shrink(p, vec![(1, 7)])?;
	let z = g.stack(vec![r, s], 0)?;
	let out = compare(&g, &[z], &[buffer(&[2, 3], &[1., 2., 3., 4., 5., 6.])])?;
	assert_eq!(
		out[0].data,
		[1., 4., 2., 5., 3., 6., 6., 3., 5., 2., 4., 1.]
	);
	assert!(g.permute(x, vec![0, 0]).is_err());
	assert!(g.reshape(x, vec![7]).is_err());
	Ok(())
}
#[test]
fn gather_scatter() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3, 2])?;
	let ids = g.input(1, vec![3])?;
	let y = g.gather(x, ids, 0)?;
	let zero = g.constant(0., vec![3, 2])?;
	let z = g.scatter_add(zero, ids, y, 0)?;
	let inputs = [
		buffer(&[3, 2], &[1., 2., 3., 4., 5., 6.]),
		buffer(&[3], &[2., 0., 2.]),
	];
	let out = compare(&g, &[y, z], &inputs)?;
	assert_eq!(out[0].data, [5., 6., 1., 2., 5., 6.]);
	assert_eq!(out[1].data, [1., 2., 0., 0., 10., 12.]);
	let mut bad = inputs.clone();
	bad[1].data[0] = 3.;
	assert!(g.compile(&[z])?.run(&bad).is_err());
	Ok(())
}
#[test]
fn broadcast_empty_and_negative_shapes() -> Result<()> {
	let mut g = Graph::default();
	let a = g.input(0, vec![2, 3])?;
	let b = g.input(1, vec![3])?;
	let y = g.binary(Binary::Add, a, b)?;
	assert_eq!(
		g.compile(&[y])?.run(&[
			buffer(&[2, 3], &[1., 2., 3., 4., 5., 6.]),
			buffer(&[3], &[10., 20., 30.])
		])?[0]
			.data,
		[11., 22., 33., 14., 25., 36.]
	);
	let bad = g.constant(1., vec![2])?;
	assert!(g.binary(Binary::Add, a, bad).is_err());
	assert!(g.flip(a, vec![0, 0]).is_err());
	assert!(g.stack(vec![a, b], 0).is_err());
	assert!(g.shrink(a, vec![(1, 0), (0, 3)]).is_err());
	let mut g = Graph::default();
	let x = g.input(0, vec![0, 3])?;
	let p = g.permute(x, vec![1, 0])?;
	let r = g.reshape(p, vec![0])?;
	assert!(
		g.compile(&[r])?.run(&[buffer(&[0, 3], &[])])?[0]
			.data
			.is_empty()
	);
	Ok(())
}
```

`tests/movement.rs` の `broadcast_empty_and_negative_shapes` で、この経路の値または拒否条件を確認する。

```sh
# check
cargo test --quiet --test movement broadcast_empty_and_negative_shapes
```

## 2. shape を右揃えし、入力側の長さ 1 の軸を座標 0 にする。
<!-- todo:10-002 -->

`tests/movement.rs` の正常系と境界系をまとめて再実行する。

`src/graph.rs` の `fn broadcast_index(`の定義を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn broadcast_index("
	fn broadcast_index(&self, input: usize, output: usize, index: &str) -> String {
		let rank = self.g.shape(input).len();
		let outer = self.g.shape(output).len();
		self.flatten(
			input,
			&(0..rank)
				.map(|a| {
					format!(
						"({} == 1 ? 0 : {})",
						self.dim(input, a),
						self.coord(output, index, outer - rank + a)
					)
				})
				.collect::<Vec<_>>(),
		)
	}
```

```sh
# check
cargo test --quiet --test movement
```

## 3. `[2,3] + [3]` と `[2,1] + [1,3]` を CPU で実行する。
<!-- todo:10-003 -->

`tests/movement.rs` の正常系と境界系をまとめて再実行する。

`tests/movement.rs` の `fn movement(`の定義を次の全体で置き換えます。

```rust
// file="tests/movement.rs" mode="replace" item="fn movement("
fn movement() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![2, 3])?;
	let t = g.permute(x, vec![1, 0])?;
	let r = g.reshape(t, vec![6])?;
	let f = g.flip(r, vec![0])?;
	let p = g.pad(f, vec![(1, 2)])?;
	let s = g.shrink(p, vec![(1, 7)])?;
	let z = g.stack(vec![r, s], 0)?;
	let out = compare(&g, &[z], &[buffer(&[2, 3], &[1., 2., 3., 4., 5., 6.])])?;
	assert_eq!(
		out[0].data,
		[1., 4., 2., 5., 3., 6., 6., 3., 5., 2., 4., 1.]
	);
	assert!(g.permute(x, vec![0, 0]).is_err());
	assert!(g.reshape(x, vec![7]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test movement
```

## 4. `[2,3] + [2]` は broadcast 不成立として codegen 前に拒否する。
<!-- todo:10-004 -->

`tests/movement.rs` の `broadcast_empty_and_negative_shapes` で、この経路の値または拒否条件を確認する。

`src/graph.rs` の `pub fn broadcast(`の定義を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="pub fn broadcast("
pub fn broadcast(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
	let rank = a.len().max(b.len());
	let mut out = vec![1; rank];
	for k in 0..rank {
		let x = a.len().checked_sub(k + 1).map(|i| a[i]).unwrap_or(1);
		let y = b.len().checked_sub(k + 1).map(|i| b[i]).unwrap_or(1);
		out[rank - k - 1] = if x == y {
			x
		} else if x == 1 {
			y
		} else if y == 1 {
			x
		} else {
			return Err("incompatible broadcast shapes".into());
		};
	}
	elements(&out)?;
	Ok(out)
}
```

```sh
# check
cargo test --quiet --test movement broadcast_empty_and_negative_shapes
```

## 5. 要素数が等しい reshape だけを受け付ける。
<!-- todo:10-005 -->

`tests/movement.rs` の正常系と境界系をまとめて再実行する。

`src/graph.rs` の `fn verify_node(` の `Op::Reshape` 分岐を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn verify_node(" arm="Op::Reshape"
			Op::Reshape { x } => {
				if elements(s(*x))? != elements(out)? {
					return Err("reshape changes element count".into());
				}
				out.to_vec()
			}
```

```sh
# check
cargo test --quiet --test movement
```

## 6. permute の軸が重複せず、全軸を一度ずつ含むことを検証する。
<!-- todo:10-006 -->

`tests/movement.rs` の `broadcast_empty_and_negative_shapes` で、この経路の値または拒否条件を確認する。

`src/graph.rs` の `fn verify_node(` の `Op::Permute` 分岐を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn verify_node(" arm="Op::Permute"
			Op::Permute { x, axes } => {
				axes_valid(axes, s(*x).len())?;
				if axes.len() != s(*x).len() {
					return Err("permute rank mismatch".into());
				}
				axes.iter().map(|&a| s(*x)[a]).collect()
			}
```

```sh
# check
cargo test --quiet --test movement broadcast_empty_and_negative_shapes
```

## 7. 論理的な平坦化と座標の復元を rangeify に追加する。
<!-- todo:10-007 -->

`tests/movement.rs` の正常系と境界系をまとめて再実行する。

`src/graph.rs` の `fn flatten(`の定義を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn flatten("
	fn flatten(&self, id: usize, c: &[String]) -> String {
		if c.is_empty() {
			return "0".into();
		}
		format!(
			"({})",
			c.iter()
				.enumerate()
				.map(|(a, c)| format!(
					"({c}) * {}",
					self.product(id, a + 1..self.g.shape(id).len())
				))
				.collect::<Vec<_>>()
				.join(" + ")
		)
	}
```

```sh
# check
cargo test --quiet --test movement
```

## 8. 転置後の reshape を CPU 実行し、連続化した場合とも比較する。
<!-- todo:10-008 -->

`tests/movement.rs` の正常系と境界系をまとめて再実行する。

`tests/movement.rs` の `fn movement(`の定義を次の全体で置き換えます。

```rust
// file="tests/movement.rs" mode="replace" item="fn movement("
fn movement() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![2, 3])?;
	let t = g.permute(x, vec![1, 0])?;
	let r = g.reshape(t, vec![6])?;
	let f = g.flip(r, vec![0])?;
	let p = g.pad(f, vec![(1, 2)])?;
	let s = g.shrink(p, vec![(1, 7)])?;
	let z = g.stack(vec![r, s], 0)?;
	let out = compare(&g, &[z], &[buffer(&[2, 3], &[1., 2., 3., 4., 5., 6.])])?;
	assert_eq!(
		out[0].data,
		[1., 4., 2., 5., 3., 6., 6., 3., 5., 2., 4., 1.]
	);
	assert!(g.permute(x, vec![0, 0]).is_err());
	assert!(g.reshape(x, vec![7]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test movement
```

## 9. shrink の半開区間 `start..end` を入力側の offset に変換する。
<!-- todo:10-009 -->

`tests/movement.rs` の正常系と境界系をまとめて再実行する。

`src/graph.rs` の `fn expr(` の `Op::Shrink` 分岐を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn expr(" arm="Op::Shrink"
			Op::Shrink { x, ranges } => {
				let c = ranges
					.iter()
					.enumerate()
					.map(|(a, &(lo, _))| format!("({}+{lo})", self.coord(id, index, a)))
					.collect::<Vec<_>>();
				let xi = self.flatten(x, &c);
				return self.expr(x, &xi, code);
			}
```

```sh
# check
cargo test --quiet --test movement
```

## 10. `0 <= start <= end <= input_size` を各軸で検証する。
<!-- todo:10-010 -->

`tests/movement.rs` の正常系と境界系をまとめて再実行する。

`src/graph.rs` の `fn verify_node(` の `Op::Shrink` 分岐を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn verify_node(" arm="Op::Shrink"
			Op::Shrink { x, ranges } => {
				if ranges.len() != s(*x).len() {
					return Err("shrink rank mismatch".into());
				}
				ranges
					.iter()
					.enumerate()
					.map(|(a, &(lo, hi))| {
						if lo <= hi && hi <= s(*x)[a] {
							Ok(hi - lo)
						} else {
							Err("invalid shrink range".to_string())
						}
					})
					.collect::<Result<_>>()?
			}
```

```sh
# check
cargo test --quiet --test movement
```

## 11. pad の有効条件と、条件成立時だけ実行する LOAD を生成する。
<!-- todo:10-011 -->

`tests/movement.rs` の正常系と境界系をまとめて再実行する。

`src/graph.rs` の `fn expr(` の `Op::Pad` 分岐を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn expr(" arm="Op::Pad"
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
```

```sh
# check
cargo test --quiet --test movement
```

## 12. pad の外側はゼロを返し、無効なポインタへの LOAD を作らない。
<!-- todo:10-012 -->

`tests/movement.rs` の正常系と境界系をまとめて再実行する。

`src/graph.rs` の `fn expr(` の `Op::Pad` 分岐を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn expr(" arm="Op::Pad"
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
```

```sh
# check
cargo test --quiet --test movement
```

## 13. 非対称 pad と shrink の合成を interpreter と生成 C で照合する。
<!-- todo:10-013 -->

`tests/movement.rs` の正常系と境界系をまとめて再実行する。

`tests/movement.rs` の `fn movement(`の定義を次の全体で置き換えます。

```rust
// file="tests/movement.rs" mode="replace" item="fn movement("
fn movement() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![2, 3])?;
	let t = g.permute(x, vec![1, 0])?;
	let r = g.reshape(t, vec![6])?;
	let f = g.flip(r, vec![0])?;
	let p = g.pad(f, vec![(1, 2)])?;
	let s = g.shrink(p, vec![(1, 7)])?;
	let z = g.stack(vec![r, s], 0)?;
	let out = compare(&g, &[z], &[buffer(&[2, 3], &[1., 2., 3., 4., 5., 6.])])?;
	assert_eq!(
		out[0].data,
		[1., 4., 2., 5., 3., 6., 6., 3., 5., 2., 4., 1.]
	);
	assert!(g.permute(x, vec![0, 0]).is_err());
	assert!(g.reshape(x, vec![7]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test movement
```

## 14. flip の座標を `size-1-coordinate` にする。
<!-- todo:10-014 -->

`tests/movement.rs` の正常系と境界系をまとめて再実行する。

`src/graph.rs` の `fn expr(` の `Op::Flip` 分岐を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn expr(" arm="Op::Flip"
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
```

```sh
# check
cargo test --quiet --test movement
```

## 15. flip の軸の範囲と重複を検証する。
<!-- todo:10-015 -->

`tests/movement.rs` の `broadcast_empty_and_negative_shapes` で、この経路の値または拒否条件を確認する。

`src/graph.rs` の `fn verify_node(` の `Op::Flip` 分岐を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn verify_node(" arm="Op::Flip"
			Op::Flip { x, axes } => {
				axes_valid(axes, s(*x).len())?;
				s(*x).to_vec()
			}
```

```sh
# check
cargo test --quiet --test movement broadcast_empty_and_negative_shapes
```

## 16. stack の先頭座標で入力を選び、残りの座標を各入力に渡す。
<!-- todo:10-016 -->

`tests/movement.rs` の正常系と境界系をまとめて再実行する。

`src/graph.rs` の `fn expr(` の `Op::Stack` 分岐を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn expr(" arm="Op::Stack"
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
```

```sh
# check
cargo test --quiet --test movement
```

## 17. 入力選択の分岐内だけで該当入力を LOAD する。
<!-- todo:10-017 -->

`tests/movement.rs` の正常系と境界系をまとめて再実行する。

`src/graph.rs` の `fn expr(` の `Op::Stack` 分岐を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn expr(" arm="Op::Stack"
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
```

```sh
# check
cargo test --quiet --test movement
```

## 18. shape が異なる stack と、未知の軸指定を拒否する。
<!-- todo:10-018 -->

`tests/movement.rs` の `broadcast_empty_and_negative_shapes` で、この経路の値または拒否条件を確認する。

`src/graph.rs` の `fn verify_node(` の `Op::Stack` 分岐を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn verify_node(" arm="Op::Stack"
			Op::Stack { xs, axis } => {
				if xs.is_empty() {
					return Err("empty stack".into());
				}
				let base = s(xs[0]);
				if *axis > base.len() || xs.iter().any(|&x| s(x) != base) {
					return Err("stack shape/axis mismatch".into());
				}
				let mut shape = base.to_vec();
				shape.insert(*axis, xs.len());
				shape
			}
```

```sh
# check
cargo test --quiet --test movement broadcast_empty_and_negative_shapes
```

## 19. 整数で表せる上記の期待値は `f32` でも厳密に一致する。
<!-- todo:10-019 -->

`tests/movement.rs` の正常系と境界系をまとめて再実行する。

`tests/movement.rs` の `fn movement(`の定義を次の全体で置き換えます。

```rust
// file="tests/movement.rs" mode="replace" item="fn movement("
fn movement() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![2, 3])?;
	let t = g.permute(x, vec![1, 0])?;
	let r = g.reshape(t, vec![6])?;
	let f = g.flip(r, vec![0])?;
	let p = g.pad(f, vec![(1, 2)])?;
	let s = g.shrink(p, vec![(1, 7)])?;
	let z = g.stack(vec![r, s], 0)?;
	let out = compare(&g, &[z], &[buffer(&[2, 3], &[1., 2., 3., 4., 5., 6.])])?;
	assert_eq!(
		out[0].data,
		[1., 4., 2., 5., 3., 6., 6., 3., 5., 2., 4., 1.]
	);
	assert!(g.permute(x, vec![0, 0]).is_err());
	assert!(g.reshape(x, vec![7]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test movement
```

## 20. kernel IR に抽象 movement が残らず、添字式と条件付き LOAD になる。
<!-- todo:10-020 -->

`tests/movement.rs` の正常系と境界系をまとめて再実行する。

`tests/movement.rs` の `fn movement(`の定義を次の全体で置き換えます。

```rust
// file="tests/movement.rs" mode="replace" item="fn movement("
fn movement() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![2, 3])?;
	let t = g.permute(x, vec![1, 0])?;
	let r = g.reshape(t, vec![6])?;
	let f = g.flip(r, vec![0])?;
	let p = g.pad(f, vec![(1, 2)])?;
	let s = g.shrink(p, vec![(1, 7)])?;
	let z = g.stack(vec![r, s], 0)?;
	let out = compare(&g, &[z], &[buffer(&[2, 3], &[1., 2., 3., 4., 5., 6.])])?;
	assert_eq!(
		out[0].data,
		[1., 4., 2., 5., 3., 6., 6., 3., 5., 2., 4., 1.]
	);
	assert!(g.permute(x, vec![0, 0]).is_err());
	assert!(g.reshape(x, vec![7]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test movement
```

## 21. 元グラフの movement ノードは実行後にも残る。
<!-- todo:10-021 -->

`tests/movement.rs` の正常系と境界系をまとめて再実行する。

`tests/movement.rs` の `fn movement(`の定義を次の全体で置き換えます。

```rust
// file="tests/movement.rs" mode="replace" item="fn movement("
fn movement() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![2, 3])?;
	let t = g.permute(x, vec![1, 0])?;
	let r = g.reshape(t, vec![6])?;
	let f = g.flip(r, vec![0])?;
	let p = g.pad(f, vec![(1, 2)])?;
	let s = g.shrink(p, vec![(1, 7)])?;
	let z = g.stack(vec![r, s], 0)?;
	let out = compare(&g, &[z], &[buffer(&[2, 3], &[1., 2., 3., 4., 5., 6.])])?;
	assert_eq!(
		out[0].data,
		[1., 4., 2., 5., 3., 6., 6., 3., 5., 2., 4., 1.]
	);
	assert!(g.permute(x, vec![0, 0]).is_err());
	assert!(g.reshape(x, vec![7]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test movement
```

## 22. `tests/movement.rs` に不正 shape、軸重複、負の区間の拒否例を追加する。
<!-- todo:10-022 -->

`tests/movement.rs` の `broadcast_empty_and_negative_shapes` で、この経路の値または拒否条件を確認する。

`tests/movement.rs` の `fn broadcast_empty_and_negative_shapes(`の定義を次の全体で置き換えます。

```rust
// file="tests/movement.rs" mode="replace" item="fn broadcast_empty_and_negative_shapes("
fn broadcast_empty_and_negative_shapes() -> Result<()> {
	let mut g = Graph::default();
	let a = g.input(0, vec![2, 3])?;
	let b = g.input(1, vec![3])?;
	let y = g.binary(Binary::Add, a, b)?;
	assert_eq!(
		g.compile(&[y])?.run(&[
			buffer(&[2, 3], &[1., 2., 3., 4., 5., 6.]),
			buffer(&[3], &[10., 20., 30.])
		])?[0]
			.data,
		[11., 22., 33., 14., 25., 36.]
	);
	let bad = g.constant(1., vec![2])?;
	assert!(g.binary(Binary::Add, a, bad).is_err());
	assert!(g.flip(a, vec![0, 0]).is_err());
	assert!(g.stack(vec![a, b], 0).is_err());
	assert!(g.shrink(a, vec![(1, 0), (0, 3)]).is_err());
	let mut g = Graph::default();
	let x = g.input(0, vec![0, 3])?;
	let p = g.permute(x, vec![1, 0])?;
	let r = g.reshape(p, vec![0])?;
	assert!(
		g.compile(&[r])?.run(&[buffer(&[0, 3], &[])])?[0]
			.data
			.is_empty()
	);
	Ok(())
}
```

```sh
# check
cargo test --quiet --test movement broadcast_empty_and_negative_shapes
```

## 23. 同じ NodeId を異なる座標で読む例が、順序と値の両方で正しい。
<!-- todo:10-023 -->

`tests/movement.rs` の `gather_scatter` で、この経路の値または拒否条件を確認する。

`tests/movement.rs` の `fn gather_scatter(`の定義を次の全体で置き換えます。

```rust
// file="tests/movement.rs" mode="replace" item="fn gather_scatter("
fn gather_scatter() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3, 2])?;
	let ids = g.input(1, vec![3])?;
	let y = g.gather(x, ids, 0)?;
	let zero = g.constant(0., vec![3, 2])?;
	let z = g.scatter_add(zero, ids, y, 0)?;
	let inputs = [
		buffer(&[3, 2], &[1., 2., 3., 4., 5., 6.]),
		buffer(&[3], &[2., 0., 2.]),
	];
	let out = compare(&g, &[y, z], &inputs)?;
	assert_eq!(out[0].data, [5., 6., 1., 2., 5., 6.]);
	assert_eq!(out[1].data, [1., 2., 0., 0., 10., 12.]);
	let mut bad = inputs.clone();
	bad[1].data[0] = 3.;
	assert!(g.compile(&[z])?.run(&bad).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test movement gather_scatter
```

## 24. 空の reshape、permute、shrink はメモリアクセスとゼロ除算を起こさない。
<!-- todo:10-024 -->

`tests/movement.rs` の `broadcast_empty_and_negative_shapes` で、この経路の値または拒否条件を確認する。

`tests/movement.rs` の `fn broadcast_empty_and_negative_shapes(`の定義を次の全体で置き換えます。

```rust
// file="tests/movement.rs" mode="replace" item="fn broadcast_empty_and_negative_shapes("
fn broadcast_empty_and_negative_shapes() -> Result<()> {
	let mut g = Graph::default();
	let a = g.input(0, vec![2, 3])?;
	let b = g.input(1, vec![3])?;
	let y = g.binary(Binary::Add, a, b)?;
	assert_eq!(
		g.compile(&[y])?.run(&[
			buffer(&[2, 3], &[1., 2., 3., 4., 5., 6.]),
			buffer(&[3], &[10., 20., 30.])
		])?[0]
			.data,
		[11., 22., 33., 14., 25., 36.]
	);
	let bad = g.constant(1., vec![2])?;
	assert!(g.binary(Binary::Add, a, bad).is_err());
	assert!(g.flip(a, vec![0, 0]).is_err());
	assert!(g.stack(vec![a, b], 0).is_err());
	assert!(g.shrink(a, vec![(1, 0), (0, 3)]).is_err());
	let mut g = Graph::default();
	let x = g.input(0, vec![0, 3])?;
	let p = g.permute(x, vec![1, 0])?;
	let r = g.reshape(p, vec![0])?;
	assert!(
		g.compile(&[r])?.run(&[buffer(&[0, 3], &[])])?[0]
			.data
			.is_empty()
	);
	Ok(())
}
```

```sh
# check
cargo test --quiet --test movement broadcast_empty_and_negative_shapes
```

## 章末の累積確認

前章までの example と test も同じ checkout で再ビルドする。

```sh
# check
cargo test --quiet --all-targets
```
