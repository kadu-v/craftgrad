# 11. Tensor の合成演算で matmul と conv を動かす

Tensor は Graph の Rc と NodeId を持つ薄い builder である。演算メソッドは compile も run も呼ばない。matmul は reshape、permute、MUL、SUM へ、conv は SHRINK、STACK、MUL、SUM へ合成する。convolution は NCHW、stride=1、dilation=1、groups=1 のみで、他の指定は明示的なエラーになる。

`file=... mode=write` はそのファイルの全置換、`mode=append` は末尾への追記である。すべて前章でできたプロジェクトの root で行う。`sh check` は実際の検証コマンドであり、ソースファイルには保存しない。

## 1. PARAM とホスト配列を結び付ける Tensor のコンストラクタを追加する。
<!-- todo:11-001 -->

```rust file=src/graph.rs mode=append
impl Graph {
	pub fn matmul(&mut self, a: NodeId, b: NodeId) -> Result<NodeId> {
		let sa = self.get(a)?.shape.clone();
		let sb = self.get(b)?.shape.clone();
		if sa.len() != 2 || sb.len() != 2 || sa[1] != sb[0] {
			return Err("matmul requires [M,K] and [K,N]".into());
		}
		let aa = self.reshape(a, vec![sa[0], 1, sa[1]])?;
		let bt = self.permute(b, vec![1, 0])?;
		let bb = self.reshape(bt, vec![1, sb[1], sb[0]])?;
		let products = self.binary(Binary::Mul, aa, bb)?;
		self.reduce(products, Reduce::Sum, vec![2], false)
	}
	pub fn conv2d(
		&mut self,
		x: NodeId,
		w: NodeId,
		padding: (usize, usize),
		stride: usize,
		dilation: usize,
		groups: usize,
	) -> Result<NodeId> {
		if stride != 1 || dilation != 1 || groups != 1 {
			return Err("conv supports stride=dilation=groups=1".into());
		}
		let sx = self.get(x)?.shape.clone();
		let sw = self.get(w)?.shape.clone();
		if sx.len() != 4 || sw.len() != 4 || sx[1] != sw[1] || sw[2] == 0 || sw[3] == 0 {
			return Err("conv requires NCHW and OIHW".into());
		}
		let xp = self.pad(
			x,
			vec![
				(0, 0),
				(0, 0),
				(padding.0, padding.0),
				(padding.1, padding.1),
			],
		)?;
		let sp = self.shape(xp).to_vec();
		let h = sp[2]
			.checked_sub(sw[2])
			.and_then(|n| n.checked_add(1))
			.ok_or("kernel taller than input")?;
		let width = sp[3]
			.checked_sub(sw[3])
			.and_then(|n| n.checked_add(1))
			.ok_or("kernel wider than input")?;
		let mut windows = vec![];
		for kh in 0..sw[2] {
			for kw in 0..sw[3] {
				windows.push(self.shrink(
					xp,
					vec![(0, sx[0]), (0, sx[1]), (kh, kh + h), (kw, kw + width)],
				)?)
			}
		}
		let windows = self.stack(windows, 4)?;
		let windows = self.reshape(windows, vec![sx[0], 1, sx[1], h, width, sw[2], sw[3]])?;
		let weights = self.reshape(w, vec![1, sw[0], sw[1], 1, 1, sw[2], sw[3]])?;
		let product = self.binary(Binary::Mul, windows, weights)?;
		self.reduce(product, Reduce::Sum, vec![2, 5, 6], false)
	}
}
```

```rust file=src/tensor.rs mode=write
use crate::{Binary, Buffer, Graph, NodeId, Plan, Reduce, Result, Unary};
use std::{cell::RefCell, rc::Rc};
#[derive(Clone, Default)]
pub struct TensorContext {
	pub graph: Rc<RefCell<Graph>>,
	pub inputs: Rc<RefCell<Vec<Buffer>>>,
}
#[derive(Clone)]
pub struct Tensor {
	pub graph: Rc<RefCell<Graph>>,
	pub id: NodeId,
}
impl TensorContext {
	pub fn input(&self, shape: Vec<usize>, data: Vec<f32>) -> Result<Tensor> {
		let buffer = Buffer::new(shape.clone(), data)?;
		let slot = self.inputs.borrow().len();
		let id = self.graph.borrow_mut().input(slot, shape)?;
		self.inputs.borrow_mut().push(buffer);
		Ok(Tensor {
			graph: self.graph.clone(),
			id,
		})
	}
	pub fn run(&self, plan: &Plan) -> Result<Vec<Buffer>> {
		plan.run(&self.inputs.borrow())
	}
	pub fn bind(&self, slot: usize, buffer: Buffer) -> Result<()> {
		let mut inputs = self.inputs.borrow_mut();
		let target = inputs.get_mut(slot).ok_or("unknown tensor input slot")?;
		if target.shape != buffer.shape {
			return Err("replacement shape mismatch".into());
		}
		*target = buffer;
		Ok(())
	}
}
impl Tensor {
	fn wrap(&self, id: NodeId) -> Self {
		Self {
			graph: self.graph.clone(),
			id,
		}
	}
	fn binary(&self, b: &Self, op: Binary) -> Result<Self> {
		if !Rc::ptr_eq(&self.graph, &b.graph) {
			return Err("tensors belong to different graphs".into());
		}
		let id = self.graph.borrow_mut().binary(op, self.id, b.id)?;
		Ok(self.wrap(id))
	}
	pub fn add(&self, b: &Self) -> Result<Self> {
		self.binary(b, Binary::Add)
	}
	pub fn mul(&self, b: &Self) -> Result<Self> {
		self.binary(b, Binary::Mul)
	}
	pub fn sub(&self, b: &Self) -> Result<Self> {
		self.binary(b, Binary::Sub)
	}
	pub fn unary(&self, op: Unary) -> Result<Self> {
		let id = self.graph.borrow_mut().unary(op, self.id)?;
		Ok(self.wrap(id))
	}
	pub fn sum(&self, axes: Vec<usize>, keepdim: bool) -> Result<Self> {
		let id = self
			.graph
			.borrow_mut()
			.reduce(self.id, Reduce::Sum, axes, keepdim)?;
		Ok(self.wrap(id))
	}
	pub fn reshape(&self, shape: Vec<usize>) -> Result<Self> {
		let id = self.graph.borrow_mut().reshape(self.id, shape)?;
		Ok(self.wrap(id))
	}
	pub fn permute(&self, axes: Vec<usize>) -> Result<Self> {
		let id = self.graph.borrow_mut().permute(self.id, axes)?;
		Ok(self.wrap(id))
	}
	pub fn expand(&self, shape: Vec<usize>) -> Result<Self> {
		let id = self.graph.borrow_mut().expand(self.id, shape)?;
		Ok(self.wrap(id))
	}
	pub fn shrink(&self, ranges: Vec<(usize, usize)>) -> Result<Self> {
		let id = self.graph.borrow_mut().shrink(self.id, ranges)?;
		Ok(self.wrap(id))
	}
	pub fn pad(&self, pads: Vec<(usize, usize)>) -> Result<Self> {
		let id = self.graph.borrow_mut().pad(self.id, pads)?;
		Ok(self.wrap(id))
	}
	pub fn flip(&self, axes: Vec<usize>) -> Result<Self> {
		let id = self.graph.borrow_mut().flip(self.id, axes)?;
		Ok(self.wrap(id))
	}
	pub fn matmul(&self, b: &Self) -> Result<Self> {
		if !Rc::ptr_eq(&self.graph, &b.graph) {
			return Err("different graphs".into());
		}
		let id = self.graph.borrow_mut().matmul(self.id, b.id)?;
		Ok(self.wrap(id))
	}
	pub fn conv2d(
		&self,
		w: &Self,
		padding: (usize, usize),
		stride: usize,
		dilation: usize,
		groups: usize,
	) -> Result<Self> {
		if !Rc::ptr_eq(&self.graph, &w.graph) {
			return Err("different graphs".into());
		}
		let id = self
			.graph
			.borrow_mut()
			.conv2d(self.id, w.id, padding, stride, dilation, groups)?;
		Ok(self.wrap(id))
	}
	pub fn compile(&self) -> Result<Plan> {
		self.graph.borrow().compile(&[self.id])
	}
	pub fn shape(&self) -> Vec<usize> {
		self.graph.borrow().shape(self.id).to_vec()
	}
}
```

```rust file=src/lib.rs mode=append
pub mod tensor;
```

```rust file=tests/tensor_operations.rs mode=write
use chinigrad::{Binary, Buffer, Graph, Reduce, Result, Unary};
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
use chinigrad::tensor::TensorContext;

#[test]
fn matmul_conv() -> Result<()> {
	let mut g = Graph::default();
	let a = g.input(0, vec![2, 3])?;
	let b = g.input(1, vec![3, 2])?;
	let c = g.matmul(a, b)?;
	let out = compare(
		&g,
		&[c],
		&[
			buffer(&[2, 3], &[1., 2., 3., 4., 5., 6.]),
			buffer(&[3, 2], &[1., 2., 3., 4., 5., 6.]),
		],
	)?;
	assert_eq!(out[0].data, [22., 28., 49., 64.]);
	let mut g = Graph::default();
	let x = g.input(0, vec![1, 1, 1, 4])?;
	let w = g.input(1, vec![1, 1, 1, 2])?;
	let y = g.conv2d(x, w, (0, 0), 1, 1, 1)?;
	let out = compare(
		&g,
		&[y],
		&[
			buffer(&[1, 1, 1, 4], &[1., 2., 3., 4.]),
			buffer(&[1, 1, 1, 2], &[2., 1.]),
		],
	)?;
	assert_eq!(out[0].data, [4., 7., 10.]);
	Ok(())
}
#[test]
fn multi_channel_convolution_and_padding() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![1, 2, 2, 3])?;
	let w = g.input(1, vec![2, 2, 2, 2])?;
	let y = g.conv2d(x, w, (1, 0), 1, 1, 1)?;
	let values = (1..=12).map(|i| i as f32).collect::<Vec<_>>();
	let weights = (0..16).map(|i| (i % 5) as f32 - 2.).collect::<Vec<_>>();
	let inputs = [
		Buffer::new(vec![1, 2, 2, 3], values.clone())?,
		Buffer::new(vec![2, 2, 2, 2], weights.clone())?,
	];
	let actual = g.compile(&[y])?.run(&inputs)?;
	let mut expected = vec![];
	for o in 0..2 {
		for row in 0..3 {
			for col in 0..2 {
				let mut sum = 0.;
				for c in 0..2 {
					for kh in 0..2 {
						for kw in 0..2 {
							let ir = row as isize + kh as isize - 1;
							if ir >= 0 && ir < 2 {
								sum += values[(c * 2 + ir as usize) * 3 + col + kw]
									* weights[((o * 2 + c) * 2 + kh) * 2 + kw];
							}
						}
					}
				}
				expected.push(sum);
			}
		}
	}
	assert_eq!(actual[0].data, expected);
	let bias = g.input(2, vec![2])?;
	let bias = g.reshape(bias, vec![1, 2, 1, 1])?;
	let biased = g.binary(Binary::Add, y, bias)?;
	let mut bindings = inputs.to_vec();
	bindings.push(buffer(&[2], &[1., -2.]));
	let biased = g.compile(&[biased])?.run(&bindings)?;
	let expected = expected
		.iter()
		.enumerate()
		.map(|(i, &v)| v + if i < 6 { 1. } else { -2. })
		.collect::<Vec<_>>();
	assert_eq!(biased[0].data, expected);
	assert!(g.conv2d(x, w, (0, 0), 2, 1, 1).is_err());
	Ok(())
}
#[test]
fn lazy_tensor() -> Result<()> {
	let context = TensorContext::default();
	let a = context.input(vec![2, 3], vec![1., 2., 3., 4., 5., 6.])?;
	let b = context.input(vec![3, 2], vec![1., 2., 3., 4., 5., 6.])?;
	let y = a.matmul(&b)?;
	assert_eq!(y.shape(), [2, 2]);
	let plan = y.compile()?;
	assert_eq!(context.run(&plan)?[0].data, [22., 28., 49., 64.]);
	Ok(())
}
```

`tests/tensor_operations.rs` の `lazy_tensor` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test tensor_operations lazy_tensor
```

## 2. `add`、`mul`、`sum`、movement のメソッドを graph builder に委譲する。
<!-- todo:11-002 -->

`tests/tensor_operations.rs` の正常系と境界系をまとめて再実行する。

`src/tensor.rs` の `impl Tensor {`の定義を次の全体で置き換えます。

```rust file="src/tensor.rs" mode="replace" item="impl Tensor {"
impl Tensor {
	fn wrap(&self, id: NodeId) -> Self {
		Self {
			graph: self.graph.clone(),
			id,
		}
	}
```

```sh check
cargo test --quiet --test tensor_operations
```

## 3. `Tensor` 演算内で暗黙にコンパイルや実行を始めない。
<!-- todo:11-003 -->

`tests/tensor_operations.rs` の `lazy_tensor` で、この経路の値または拒否条件を確認する。

`src/tensor.rs` の `fn binary(`の定義を次の全体で置き換えます。

```rust file="src/tensor.rs" mode="replace" item="fn binary("
	fn binary(&self, b: &Self, op: Binary) -> Result<Self> {
		if !Rc::ptr_eq(&self.graph, &b.graph) {
			return Err("tensors belong to different graphs".into());
		}
		let id = self.graph.borrow_mut().binary(op, self.id, b.id)?;
		Ok(self.wrap(id))
	}
```

```sh check
cargo test --quiet --test tensor_operations lazy_tensor
```

## 4. graph を出力してから compile し、既存の CPU runtime で実行する。
<!-- todo:11-004 -->

`tests/tensor_operations.rs` の正常系と境界系をまとめて再実行する。

`src/tensor.rs` の `pub fn compile(`の定義を次の全体で置き換えます。

```rust file="src/tensor.rs" mode="replace" item="pub fn compile("
	pub fn compile(&self) -> Result<Plan> {
		self.graph.borrow().compile(&[self.id])
	}
```

```sh check
cargo test --quiet --test tensor_operations
```

## 5. 縮約次元 K の一致を検証する。
<!-- todo:11-005 -->

`tests/tensor_operations.rs` の正常系と境界系をまとめて再実行する。

`src/graph.rs` の `pub fn matmul(`の定義を次の全体で置き換えます。

```rust file="src/graph.rs" mode="replace" item="pub fn matmul("
	pub fn matmul(&mut self, a: NodeId, b: NodeId) -> Result<NodeId> {
		let sa = self.get(a)?.shape.clone();
		let sb = self.get(b)?.shape.clone();
		if sa.len() != 2 || sb.len() != 2 || sa[1] != sb[0] {
			return Err("matmul requires [M,K] and [K,N]".into());
		}
		let aa = self.reshape(a, vec![sa[0], 1, sa[1]])?;
		let bt = self.permute(b, vec![1, 0])?;
		let bb = self.reshape(bt, vec![1, sb[1], sb[0]])?;
		let products = self.binary(Binary::Mul, aa, bb)?;
		self.reduce(products, Reduce::Sum, vec![2], false)
	}
```

```sh check
cargo test --quiet --test tensor_operations
```

## 6. 上記の reshape、permute、MUL、SUM の式を Tensor メソッドで作る。
<!-- todo:11-006 -->

`tests/tensor_operations.rs` の `lazy_tensor` で、この経路の値または拒否条件を確認する。

`src/graph.rs` の `pub fn matmul(`の定義を次の全体で置き換えます。

```rust file="src/graph.rs" mode="replace" item="pub fn matmul("
	pub fn matmul(&mut self, a: NodeId, b: NodeId) -> Result<NodeId> {
		let sa = self.get(a)?.shape.clone();
		let sb = self.get(b)?.shape.clone();
		if sa.len() != 2 || sb.len() != 2 || sa[1] != sb[0] {
			return Err("matmul requires [M,K] and [K,N]".into());
		}
		let aa = self.reshape(a, vec![sa[0], 1, sa[1]])?;
		let bt = self.permute(b, vec![1, 0])?;
		let bb = self.reshape(bt, vec![1, sb[1], sb[0]])?;
		let products = self.binary(Binary::Mul, aa, bb)?;
		self.reduce(products, Reduce::Sum, vec![2], false)
	}
```

```sh check
cargo test --quiet --test tensor_operations lazy_tensor
```

## 7. 出力の shape `[M,N]` と累積の初期化位置を確認する。
<!-- todo:11-007 -->

`tests/tensor_operations.rs` の正常系と境界系をまとめて再実行する。

`tests/tensor_operations.rs` の `fn matmul_conv(`の定義を次の全体で置き換えます。

```rust file="tests/tensor_operations.rs" mode="replace" item="fn matmul_conv("
fn matmul_conv() -> Result<()> {
	let mut g = Graph::default();
	let a = g.input(0, vec![2, 3])?;
	let b = g.input(1, vec![3, 2])?;
	let c = g.matmul(a, b)?;
	let out = compare(
		&g,
		&[c],
		&[
			buffer(&[2, 3], &[1., 2., 3., 4., 5., 6.]),
			buffer(&[3, 2], &[1., 2., 3., 4., 5., 6.]),
		],
	)?;
	assert_eq!(out[0].data, [22., 28., 49., 64.]);
	let mut g = Graph::default();
	let x = g.input(0, vec![1, 1, 1, 4])?;
	let w = g.input(1, vec![1, 1, 1, 2])?;
	let y = g.conv2d(x, w, (0, 0), 1, 1, 1)?;
	let out = compare(
		&g,
		&[y],
		&[
			buffer(&[1, 1, 1, 4], &[1., 2., 3., 4.]),
			buffer(&[1, 1, 1, 2], &[2., 1.]),
		],
	)?;
	assert_eq!(out[0].data, [4., 7., 10.]);
	Ok(())
}
```

```sh check
cargo test --quiet --test tensor_operations
```

## 8. 手書きの三重ループと `M=2,N=2,K=3` の結果を比較する。
<!-- todo:11-008 -->

`tests/tensor_operations.rs` の正常系と境界系をまとめて再実行する。

`tests/tensor_operations.rs` の `fn matmul_conv(`の定義を次の全体で置き換えます。

```rust file="tests/tensor_operations.rs" mode="replace" item="fn matmul_conv("
fn matmul_conv() -> Result<()> {
	let mut g = Graph::default();
	let a = g.input(0, vec![2, 3])?;
	let b = g.input(1, vec![3, 2])?;
	let c = g.matmul(a, b)?;
	let out = compare(
		&g,
		&[c],
		&[
			buffer(&[2, 3], &[1., 2., 3., 4., 5., 6.]),
			buffer(&[3, 2], &[1., 2., 3., 4., 5., 6.]),
		],
	)?;
	assert_eq!(out[0].data, [22., 28., 49., 64.]);
	let mut g = Graph::default();
	let x = g.input(0, vec![1, 1, 1, 4])?;
	let w = g.input(1, vec![1, 1, 1, 2])?;
	let y = g.conv2d(x, w, (0, 0), 1, 1, 1)?;
	let out = compare(
		&g,
		&[y],
		&[
			buffer(&[1, 1, 1, 4], &[1., 2., 3., 4.]),
			buffer(&[1, 1, 1, 2], &[2., 1.]),
		],
	)?;
	assert_eq!(out[0].data, [4., 7., 10.]);
	Ok(())
}
```

```sh check
cargo test --quiet --test tensor_operations
```

## 9. 同じ A を転置ありと転置なしで読む式を別々の添字で下げる。
<!-- todo:11-009 -->

`tests/tensor_operations.rs` の正常系と境界系をまとめて再実行する。

`tests/tensor_operations.rs` の `fn matmul_conv(`の定義を次の全体で置き換えます。

```rust file="tests/tensor_operations.rs" mode="replace" item="fn matmul_conv("
fn matmul_conv() -> Result<()> {
	let mut g = Graph::default();
	let a = g.input(0, vec![2, 3])?;
	let b = g.input(1, vec![3, 2])?;
	let c = g.matmul(a, b)?;
	let out = compare(
		&g,
		&[c],
		&[
			buffer(&[2, 3], &[1., 2., 3., 4., 5., 6.]),
			buffer(&[3, 2], &[1., 2., 3., 4., 5., 6.]),
		],
	)?;
	assert_eq!(out[0].data, [22., 28., 49., 64.]);
	let mut g = Graph::default();
	let x = g.input(0, vec![1, 1, 1, 4])?;
	let w = g.input(1, vec![1, 1, 1, 2])?;
	let y = g.conv2d(x, w, (0, 0), 1, 1, 1)?;
	let out = compare(
		&g,
		&[y],
		&[
			buffer(&[1, 1, 1, 4], &[1., 2., 3., 4.]),
			buffer(&[1, 1, 1, 2], &[2., 1.]),
		],
	)?;
	assert_eq!(out[0].data, [4., 7., 10.]);
	Ok(())
}
```

```sh check
cargo test --quiet --test tensor_operations
```

## 10. `KH,KW` 個の SHRINK と STACK で窓を組み立てる。
<!-- todo:11-010 -->

`tests/tensor_operations.rs` の `multi_channel_convolution_and_padding` で、この経路の値または拒否条件を確認する。

`src/graph.rs` の `pub fn conv2d(`の定義を次の全体で置き換えます。

```rust file="src/graph.rs" mode="replace" item="pub fn conv2d("
	pub fn conv2d(
		&mut self,
		x: NodeId,
		w: NodeId,
		padding: (usize, usize),
		stride: usize,
		dilation: usize,
		groups: usize,
	) -> Result<NodeId> {
		if stride != 1 || dilation != 1 || groups != 1 {
			return Err("conv supports stride=dilation=groups=1".into());
		}
		let sx = self.get(x)?.shape.clone();
		let sw = self.get(w)?.shape.clone();
		if sx.len() != 4 || sw.len() != 4 || sx[1] != sw[1] || sw[2] == 0 || sw[3] == 0 {
			return Err("conv requires NCHW and OIHW".into());
		}
		let xp = self.pad(
			x,
			vec![
				(0, 0),
				(0, 0),
				(padding.0, padding.0),
				(padding.1, padding.1),
			],
		)?;
		let sp = self.shape(xp).to_vec();
		let h = sp[2]
			.checked_sub(sw[2])
			.and_then(|n| n.checked_add(1))
			.ok_or("kernel taller than input")?;
		let width = sp[3]
			.checked_sub(sw[3])
			.and_then(|n| n.checked_add(1))
			.ok_or("kernel wider than input")?;
		let mut windows = vec![];
		for kh in 0..sw[2] {
			for kw in 0..sw[3] {
				windows.push(self.shrink(
					xp,
					vec![(0, sx[0]), (0, sx[1]), (kh, kh + h), (kw, kw + width)],
				)?)
			}
		}
		let windows = self.stack(windows, 4)?;
		let windows = self.reshape(windows, vec![sx[0], 1, sx[1], h, width, sw[2], sw[3]])?;
		let weights = self.reshape(w, vec![1, sw[0], sw[1], 1, 1, sw[2], sw[3]])?;
		let product = self.binary(Binary::Mul, windows, weights)?;
		self.reduce(product, Reduce::Sum, vec![2, 5, 6], false)
	}
```

```sh check
cargo test --quiet --test tensor_operations multi_channel_convolution_and_padding
```

## 11. 座標順を明示し、window の shape を dump に出す。
<!-- todo:11-011 -->

`tests/tensor_operations.rs` の正常系と境界系をまとめて再実行する。

`src/graph.rs` の `pub fn conv2d(`の定義を次の全体で置き換えます。

```rust file="src/graph.rs" mode="replace" item="pub fn conv2d("
	pub fn conv2d(
		&mut self,
		x: NodeId,
		w: NodeId,
		padding: (usize, usize),
		stride: usize,
		dilation: usize,
		groups: usize,
	) -> Result<NodeId> {
		if stride != 1 || dilation != 1 || groups != 1 {
			return Err("conv supports stride=dilation=groups=1".into());
		}
		let sx = self.get(x)?.shape.clone();
		let sw = self.get(w)?.shape.clone();
		if sx.len() != 4 || sw.len() != 4 || sx[1] != sw[1] || sw[2] == 0 || sw[3] == 0 {
			return Err("conv requires NCHW and OIHW".into());
		}
		let xp = self.pad(
			x,
			vec![
				(0, 0),
				(0, 0),
				(padding.0, padding.0),
				(padding.1, padding.1),
			],
		)?;
		let sp = self.shape(xp).to_vec();
		let h = sp[2]
			.checked_sub(sw[2])
			.and_then(|n| n.checked_add(1))
			.ok_or("kernel taller than input")?;
		let width = sp[3]
			.checked_sub(sw[3])
			.and_then(|n| n.checked_add(1))
			.ok_or("kernel wider than input")?;
		let mut windows = vec![];
		for kh in 0..sw[2] {
			for kw in 0..sw[3] {
				windows.push(self.shrink(
					xp,
					vec![(0, sx[0]), (0, sx[1]), (kh, kh + h), (kw, kw + width)],
				)?)
			}
		}
		let windows = self.stack(windows, 4)?;
		let windows = self.reshape(windows, vec![sx[0], 1, sx[1], h, width, sw[2], sw[3]])?;
		let weights = self.reshape(w, vec![1, sw[0], sw[1], 1, 1, sw[2], sw[3]])?;
		let product = self.binary(Binary::Mul, windows, weights)?;
		self.reduce(product, Reduce::Sum, vec![2, 5, 6], false)
	}
```

```sh check
cargo test --quiet --test tensor_operations
```

## 12. 窓と重みの積を `C,KH,KW` について縮約する。
<!-- todo:11-012 -->

`tests/tensor_operations.rs` の `multi_channel_convolution_and_padding` で、この経路の値または拒否条件を確認する。

`src/graph.rs` の `pub fn conv2d(`の定義を次の全体で置き換えます。

```rust file="src/graph.rs" mode="replace" item="pub fn conv2d("
	pub fn conv2d(
		&mut self,
		x: NodeId,
		w: NodeId,
		padding: (usize, usize),
		stride: usize,
		dilation: usize,
		groups: usize,
	) -> Result<NodeId> {
		if stride != 1 || dilation != 1 || groups != 1 {
			return Err("conv supports stride=dilation=groups=1".into());
		}
		let sx = self.get(x)?.shape.clone();
		let sw = self.get(w)?.shape.clone();
		if sx.len() != 4 || sw.len() != 4 || sx[1] != sw[1] || sw[2] == 0 || sw[3] == 0 {
			return Err("conv requires NCHW and OIHW".into());
		}
		let xp = self.pad(
			x,
			vec![
				(0, 0),
				(0, 0),
				(padding.0, padding.0),
				(padding.1, padding.1),
			],
		)?;
		let sp = self.shape(xp).to_vec();
		let h = sp[2]
			.checked_sub(sw[2])
			.and_then(|n| n.checked_add(1))
			.ok_or("kernel taller than input")?;
		let width = sp[3]
			.checked_sub(sw[3])
			.and_then(|n| n.checked_add(1))
			.ok_or("kernel wider than input")?;
		let mut windows = vec![];
		for kh in 0..sw[2] {
			for kw in 0..sw[3] {
				windows.push(self.shrink(
					xp,
					vec![(0, sx[0]), (0, sx[1]), (kh, kh + h), (kw, kw + width)],
				)?)
			}
		}
		let windows = self.stack(windows, 4)?;
		let windows = self.reshape(windows, vec![sx[0], 1, sx[1], h, width, sw[2], sw[3]])?;
		let weights = self.reshape(w, vec![1, sw[0], sw[1], 1, 1, sw[2], sw[3]])?;
		let product = self.binary(Binary::Mul, windows, weights)?;
		self.reduce(product, Reduce::Sum, vec![2, 5, 6], false)
	}
```

```sh check
cargo test --quiet --test tensor_operations multi_channel_convolution_and_padding
```

## 13. 出力の保存順を NCHW にそろえる。
<!-- todo:11-013 -->

`tests/tensor_operations.rs` の正常系と境界系をまとめて再実行する。

`src/graph.rs` の `pub fn conv2d(`の定義を次の全体で置き換えます。

```rust file="src/graph.rs" mode="replace" item="pub fn conv2d("
	pub fn conv2d(
		&mut self,
		x: NodeId,
		w: NodeId,
		padding: (usize, usize),
		stride: usize,
		dilation: usize,
		groups: usize,
	) -> Result<NodeId> {
		if stride != 1 || dilation != 1 || groups != 1 {
			return Err("conv supports stride=dilation=groups=1".into());
		}
		let sx = self.get(x)?.shape.clone();
		let sw = self.get(w)?.shape.clone();
		if sx.len() != 4 || sw.len() != 4 || sx[1] != sw[1] || sw[2] == 0 || sw[3] == 0 {
			return Err("conv requires NCHW and OIHW".into());
		}
		let xp = self.pad(
			x,
			vec![
				(0, 0),
				(0, 0),
				(padding.0, padding.0),
				(padding.1, padding.1),
			],
		)?;
		let sp = self.shape(xp).to_vec();
		let h = sp[2]
			.checked_sub(sw[2])
			.and_then(|n| n.checked_add(1))
			.ok_or("kernel taller than input")?;
		let width = sp[3]
			.checked_sub(sw[3])
			.and_then(|n| n.checked_add(1))
			.ok_or("kernel wider than input")?;
		let mut windows = vec![];
		for kh in 0..sw[2] {
			for kw in 0..sw[3] {
				windows.push(self.shrink(
					xp,
					vec![(0, sx[0]), (0, sx[1]), (kh, kh + h), (kw, kw + width)],
				)?)
			}
		}
		let windows = self.stack(windows, 4)?;
		let windows = self.reshape(windows, vec![sx[0], 1, sx[1], h, width, sw[2], sw[3]])?;
		let weights = self.reshape(w, vec![1, sw[0], sw[1], 1, 1, sw[2], sw[3]])?;
		let product = self.binary(Binary::Mul, windows, weights)?;
		self.reduce(product, Reduce::Sum, vec![2, 5, 6], false)
	}
```

```sh check
cargo test --quiet --test tensor_operations
```

## 14. 手書き conv の参照ループをテスト側に置いて比較する。
<!-- todo:11-014 -->

`tests/tensor_operations.rs` の `multi_channel_convolution_and_padding` で、この経路の値または拒否条件を確認する。

`tests/tensor_operations.rs` の `fn multi_channel_convolution_and_padding(`の定義を次の全体で置き換えます。

```rust file="tests/tensor_operations.rs" mode="replace" item="fn multi_channel_convolution_and_padding("
fn multi_channel_convolution_and_padding() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![1, 2, 2, 3])?;
	let w = g.input(1, vec![2, 2, 2, 2])?;
	let y = g.conv2d(x, w, (1, 0), 1, 1, 1)?;
	let values = (1..=12).map(|i| i as f32).collect::<Vec<_>>();
	let weights = (0..16).map(|i| (i % 5) as f32 - 2.).collect::<Vec<_>>();
	let inputs = [
		Buffer::new(vec![1, 2, 2, 3], values.clone())?,
		Buffer::new(vec![2, 2, 2, 2], weights.clone())?,
	];
	let actual = g.compile(&[y])?.run(&inputs)?;
	let mut expected = vec![];
	for o in 0..2 {
		for row in 0..3 {
			for col in 0..2 {
				let mut sum = 0.;
				for c in 0..2 {
					for kh in 0..2 {
						for kw in 0..2 {
							let ir = row as isize + kh as isize - 1;
							if ir >= 0 && ir < 2 {
								sum += values[(c * 2 + ir as usize) * 3 + col + kw]
									* weights[((o * 2 + c) * 2 + kh) * 2 + kw];
							}
						}
					}
				}
				expected.push(sum);
			}
		}
	}
	assert_eq!(actual[0].data, expected);
	let bias = g.input(2, vec![2])?;
	let bias = g.reshape(bias, vec![1, 2, 1, 1])?;
	let biased = g.binary(Binary::Add, y, bias)?;
	let mut bindings = inputs.to_vec();
	bindings.push(buffer(&[2], &[1., -2.]));
	let biased = g.compile(&[biased])?.run(&bindings)?;
	let expected = expected
		.iter()
		.enumerate()
		.map(|(i, &v)| v + if i < 6 { 1. } else { -2. })
		.collect::<Vec<_>>();
	assert_eq!(biased[0].data, expected);
	assert!(g.conv2d(x, w, (0, 0), 2, 1, 1).is_err());
	Ok(())
}
```

```sh check
cargo test --quiet --test tensor_operations multi_channel_convolution_and_padding
```

## 15. conv の前に PAD を挿入し、出力 shape の計算に反映する。
<!-- todo:11-015 -->

`tests/tensor_operations.rs` の `multi_channel_convolution_and_padding` で、この経路の値または拒否条件を確認する。

`src/graph.rs` の `pub fn conv2d(`の定義を次の全体で置き換えます。

```rust file="src/graph.rs" mode="replace" item="pub fn conv2d("
	pub fn conv2d(
		&mut self,
		x: NodeId,
		w: NodeId,
		padding: (usize, usize),
		stride: usize,
		dilation: usize,
		groups: usize,
	) -> Result<NodeId> {
		if stride != 1 || dilation != 1 || groups != 1 {
			return Err("conv supports stride=dilation=groups=1".into());
		}
		let sx = self.get(x)?.shape.clone();
		let sw = self.get(w)?.shape.clone();
		if sx.len() != 4 || sw.len() != 4 || sx[1] != sw[1] || sw[2] == 0 || sw[3] == 0 {
			return Err("conv requires NCHW and OIHW".into());
		}
		let xp = self.pad(
			x,
			vec![
				(0, 0),
				(0, 0),
				(padding.0, padding.0),
				(padding.1, padding.1),
			],
		)?;
		let sp = self.shape(xp).to_vec();
		let h = sp[2]
			.checked_sub(sw[2])
			.and_then(|n| n.checked_add(1))
			.ok_or("kernel taller than input")?;
		let width = sp[3]
			.checked_sub(sw[3])
			.and_then(|n| n.checked_add(1))
			.ok_or("kernel wider than input")?;
		let mut windows = vec![];
		for kh in 0..sw[2] {
			for kw in 0..sw[3] {
				windows.push(self.shrink(
					xp,
					vec![(0, sx[0]), (0, sx[1]), (kh, kh + h), (kw, kw + width)],
				)?)
			}
		}
		let windows = self.stack(windows, 4)?;
		let windows = self.reshape(windows, vec![sx[0], 1, sx[1], h, width, sw[2], sw[3]])?;
		let weights = self.reshape(w, vec![1, sw[0], sw[1], 1, 1, sw[2], sw[3]])?;
		let product = self.binary(Binary::Mul, windows, weights)?;
		self.reduce(product, Reduce::Sum, vec![2, 5, 6], false)
	}
```

```sh check
cargo test --quiet --test tensor_operations multi_channel_convolution_and_padding
```

## 16. channel ごとの bias `[O]` を reshape して加算する。
<!-- todo:11-016 -->

`tests/tensor_operations.rs` の `multi_channel_convolution_and_padding` で、この経路の値または拒否条件を確認する。

`src/graph.rs` の `pub fn conv2d(`の定義を次の全体で置き換えます。

```rust file="src/graph.rs" mode="replace" item="pub fn conv2d("
	pub fn conv2d(
		&mut self,
		x: NodeId,
		w: NodeId,
		padding: (usize, usize),
		stride: usize,
		dilation: usize,
		groups: usize,
	) -> Result<NodeId> {
		if stride != 1 || dilation != 1 || groups != 1 {
			return Err("conv supports stride=dilation=groups=1".into());
		}
		let sx = self.get(x)?.shape.clone();
		let sw = self.get(w)?.shape.clone();
		if sx.len() != 4 || sw.len() != 4 || sx[1] != sw[1] || sw[2] == 0 || sw[3] == 0 {
			return Err("conv requires NCHW and OIHW".into());
		}
		let xp = self.pad(
			x,
			vec![
				(0, 0),
				(0, 0),
				(padding.0, padding.0),
				(padding.1, padding.1),
			],
		)?;
		let sp = self.shape(xp).to_vec();
		let h = sp[2]
			.checked_sub(sw[2])
			.and_then(|n| n.checked_add(1))
			.ok_or("kernel taller than input")?;
		let width = sp[3]
			.checked_sub(sw[3])
			.and_then(|n| n.checked_add(1))
			.ok_or("kernel wider than input")?;
		let mut windows = vec![];
		for kh in 0..sw[2] {
			for kw in 0..sw[3] {
				windows.push(self.shrink(
					xp,
					vec![(0, sx[0]), (0, sx[1]), (kh, kh + h), (kw, kw + width)],
				)?)
			}
		}
		let windows = self.stack(windows, 4)?;
		let windows = self.reshape(windows, vec![sx[0], 1, sx[1], h, width, sw[2], sw[3]])?;
		let weights = self.reshape(w, vec![1, sw[0], sw[1], 1, 1, sw[2], sw[3]])?;
		let product = self.binary(Binary::Mul, windows, weights)?;
		self.reduce(product, Reduce::Sum, vec![2, 5, 6], false)
	}
```

```sh check
cargo test --quiet --test tensor_operations multi_channel_convolution_and_padding
```

## 17. matmul の後の bias と同じ broadcast 規則を使う。
<!-- todo:11-017 -->

`tests/tensor_operations.rs` の `multi_channel_convolution_and_padding` で、この経路の値または拒否条件を確認する。

`src/graph.rs` の `pub fn conv2d(`の定義を次の全体で置き換えます。

```rust file="src/graph.rs" mode="replace" item="pub fn conv2d("
	pub fn conv2d(
		&mut self,
		x: NodeId,
		w: NodeId,
		padding: (usize, usize),
		stride: usize,
		dilation: usize,
		groups: usize,
	) -> Result<NodeId> {
		if stride != 1 || dilation != 1 || groups != 1 {
			return Err("conv supports stride=dilation=groups=1".into());
		}
		let sx = self.get(x)?.shape.clone();
		let sw = self.get(w)?.shape.clone();
		if sx.len() != 4 || sw.len() != 4 || sx[1] != sw[1] || sw[2] == 0 || sw[3] == 0 {
			return Err("conv requires NCHW and OIHW".into());
		}
		let xp = self.pad(
			x,
			vec![
				(0, 0),
				(0, 0),
				(padding.0, padding.0),
				(padding.1, padding.1),
			],
		)?;
		let sp = self.shape(xp).to_vec();
		let h = sp[2]
			.checked_sub(sw[2])
			.and_then(|n| n.checked_add(1))
			.ok_or("kernel taller than input")?;
		let width = sp[3]
			.checked_sub(sw[3])
			.and_then(|n| n.checked_add(1))
			.ok_or("kernel wider than input")?;
		let mut windows = vec![];
		for kh in 0..sw[2] {
			for kw in 0..sw[3] {
				windows.push(self.shrink(
					xp,
					vec![(0, sx[0]), (0, sx[1]), (kh, kh + h), (kw, kw + width)],
				)?)
			}
		}
		let windows = self.stack(windows, 4)?;
		let windows = self.reshape(windows, vec![sx[0], 1, sx[1], h, width, sw[2], sw[3]])?;
		let weights = self.reshape(w, vec![1, sw[0], sw[1], 1, 1, sw[2], sw[3]])?;
		let product = self.binary(Binary::Mul, windows, weights)?;
		self.reduce(product, Reduce::Sum, vec![2, 5, 6], false)
	}
```

```sh check
cargo test --quiet --test tensor_operations multi_channel_convolution_and_padding
```

## 18. stride、dilation、groups の未対応値を明示的に拒否する。
<!-- todo:11-018 -->

`tests/tensor_operations.rs` の `multi_channel_convolution_and_padding` で、この経路の値または拒否条件を確認する。

`src/graph.rs` の `pub fn conv2d(`の定義を次の全体で置き換えます。

```rust file="src/graph.rs" mode="replace" item="pub fn conv2d("
	pub fn conv2d(
		&mut self,
		x: NodeId,
		w: NodeId,
		padding: (usize, usize),
		stride: usize,
		dilation: usize,
		groups: usize,
	) -> Result<NodeId> {
		if stride != 1 || dilation != 1 || groups != 1 {
			return Err("conv supports stride=dilation=groups=1".into());
		}
		let sx = self.get(x)?.shape.clone();
		let sw = self.get(w)?.shape.clone();
		if sx.len() != 4 || sw.len() != 4 || sx[1] != sw[1] || sw[2] == 0 || sw[3] == 0 {
			return Err("conv requires NCHW and OIHW".into());
		}
		let xp = self.pad(
			x,
			vec![
				(0, 0),
				(0, 0),
				(padding.0, padding.0),
				(padding.1, padding.1),
			],
		)?;
		let sp = self.shape(xp).to_vec();
		let h = sp[2]
			.checked_sub(sw[2])
			.and_then(|n| n.checked_add(1))
			.ok_or("kernel taller than input")?;
		let width = sp[3]
			.checked_sub(sw[3])
			.and_then(|n| n.checked_add(1))
			.ok_or("kernel wider than input")?;
		let mut windows = vec![];
		for kh in 0..sw[2] {
			for kw in 0..sw[3] {
				windows.push(self.shrink(
					xp,
					vec![(0, sx[0]), (0, sx[1]), (kh, kh + h), (kw, kw + width)],
				)?)
			}
		}
		let windows = self.stack(windows, 4)?;
		let windows = self.reshape(windows, vec![sx[0], 1, sx[1], h, width, sw[2], sw[3]])?;
		let weights = self.reshape(w, vec![1, sw[0], sw[1], 1, 1, sw[2], sw[3]])?;
		let product = self.binary(Binary::Mul, windows, weights)?;
		self.reduce(product, Reduce::Sum, vec![2, 5, 6], false)
	}
```

```sh check
cargo test --quiet --test tensor_operations multi_channel_convolution_and_padding
```

## 19. pad の外側を読む可能性がある LOAD を分岐内へ保持する。
<!-- todo:11-019 -->

`tests/tensor_operations.rs` の `multi_channel_convolution_and_padding` で、この経路の値または拒否条件を確認する。

`src/graph.rs` の `fn expr(` の `Op::Pad` 分岐を次の全体で置き換えます。

```rust file="src/graph.rs" mode="replace" item="fn expr(" arm="Op::Pad"
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

```sh check
cargo test --quiet --test tensor_operations multi_channel_convolution_and_padding
```

## 20. 予定例の期待値が一致し、手書きループとも一致する。
<!-- todo:11-020 -->

`tests/tensor_operations.rs` の正常系と境界系をまとめて再実行する。

`tests/tensor_operations.rs` の `fn matmul_conv(`の定義を次の全体で置き換えます。

```rust file="tests/tensor_operations.rs" mode="replace" item="fn matmul_conv("
fn matmul_conv() -> Result<()> {
	let mut g = Graph::default();
	let a = g.input(0, vec![2, 3])?;
	let b = g.input(1, vec![3, 2])?;
	let c = g.matmul(a, b)?;
	let out = compare(
		&g,
		&[c],
		&[
			buffer(&[2, 3], &[1., 2., 3., 4., 5., 6.]),
			buffer(&[3, 2], &[1., 2., 3., 4., 5., 6.]),
		],
	)?;
	assert_eq!(out[0].data, [22., 28., 49., 64.]);
	let mut g = Graph::default();
	let x = g.input(0, vec![1, 1, 1, 4])?;
	let w = g.input(1, vec![1, 1, 1, 2])?;
	let y = g.conv2d(x, w, (0, 0), 1, 1, 1)?;
	let out = compare(
		&g,
		&[y],
		&[
			buffer(&[1, 1, 1, 4], &[1., 2., 3., 4.]),
			buffer(&[1, 1, 1, 2], &[2., 1.]),
		],
	)?;
	assert_eq!(out[0].data, [4., 7., 10.]);
	Ok(())
}
```

```sh check
cargo test --quiet --test tensor_operations
```

## 21. graph に matmul と conv 専用の実行器への抜け道がない。
<!-- todo:11-021 -->

`tests/tensor_operations.rs` の `multi_channel_convolution_and_padding` で、この経路の値または拒否条件を確認する。

`tests/tensor_operations.rs` の `fn multi_channel_convolution_and_padding(`の定義を次の全体で置き換えます。

```rust file="tests/tensor_operations.rs" mode="replace" item="fn multi_channel_convolution_and_padding("
fn multi_channel_convolution_and_padding() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![1, 2, 2, 3])?;
	let w = g.input(1, vec![2, 2, 2, 2])?;
	let y = g.conv2d(x, w, (1, 0), 1, 1, 1)?;
	let values = (1..=12).map(|i| i as f32).collect::<Vec<_>>();
	let weights = (0..16).map(|i| (i % 5) as f32 - 2.).collect::<Vec<_>>();
	let inputs = [
		Buffer::new(vec![1, 2, 2, 3], values.clone())?,
		Buffer::new(vec![2, 2, 2, 2], weights.clone())?,
	];
	let actual = g.compile(&[y])?.run(&inputs)?;
	let mut expected = vec![];
	for o in 0..2 {
		for row in 0..3 {
			for col in 0..2 {
				let mut sum = 0.;
				for c in 0..2 {
					for kh in 0..2 {
						for kw in 0..2 {
							let ir = row as isize + kh as isize - 1;
							if ir >= 0 && ir < 2 {
								sum += values[(c * 2 + ir as usize) * 3 + col + kw]
									* weights[((o * 2 + c) * 2 + kh) * 2 + kw];
							}
						}
					}
				}
				expected.push(sum);
			}
		}
	}
	assert_eq!(actual[0].data, expected);
	let bias = g.input(2, vec![2])?;
	let bias = g.reshape(bias, vec![1, 2, 1, 1])?;
	let biased = g.binary(Binary::Add, y, bias)?;
	let mut bindings = inputs.to_vec();
	bindings.push(buffer(&[2], &[1., -2.]));
	let biased = g.compile(&[biased])?.run(&bindings)?;
	let expected = expected
		.iter()
		.enumerate()
		.map(|(i, &v)| v + if i < 6 { 1. } else { -2. })
		.collect::<Vec<_>>();
	assert_eq!(biased[0].data, expected);
	assert!(g.conv2d(x, w, (0, 0), 2, 1, 1).is_err());
	Ok(())
}
```

```sh check
cargo test --quiet --test tensor_operations multi_channel_convolution_and_padding
```

## 22. reduction の初期化が出力要素ごとに一回だけ実行される。
<!-- todo:11-022 -->

`tests/tensor_operations.rs` の正常系と境界系をまとめて再実行する。

`tests/tensor_operations.rs` の `fn matmul_conv(`の定義を次の全体で置き換えます。

```rust file="tests/tensor_operations.rs" mode="replace" item="fn matmul_conv("
fn matmul_conv() -> Result<()> {
	let mut g = Graph::default();
	let a = g.input(0, vec![2, 3])?;
	let b = g.input(1, vec![3, 2])?;
	let c = g.matmul(a, b)?;
	let out = compare(
		&g,
		&[c],
		&[
			buffer(&[2, 3], &[1., 2., 3., 4., 5., 6.]),
			buffer(&[3, 2], &[1., 2., 3., 4., 5., 6.]),
		],
	)?;
	assert_eq!(out[0].data, [22., 28., 49., 64.]);
	let mut g = Graph::default();
	let x = g.input(0, vec![1, 1, 1, 4])?;
	let w = g.input(1, vec![1, 1, 1, 2])?;
	let y = g.conv2d(x, w, (0, 0), 1, 1, 1)?;
	let out = compare(
		&g,
		&[y],
		&[
			buffer(&[1, 1, 1, 4], &[1., 2., 3., 4.]),
			buffer(&[1, 1, 1, 2], &[2., 1.]),
		],
	)?;
	assert_eq!(out[0].data, [4., 7., 10.]);
	Ok(())
}
```

```sh check
cargo test --quiet --test tensor_operations
```

## 23. 学習用重みを CONST として intern せず、入力値を差し替えられる。
<!-- todo:11-023 -->

`tests/tensor_operations.rs` の正常系と境界系をまとめて再実行する。

`src/tensor.rs` の `pub fn bind(`の定義を次の全体で置き換えます。

```rust file="src/tensor.rs" mode="replace" item="pub fn bind("
	pub fn bind(&self, slot: usize, buffer: Buffer) -> Result<()> {
		let mut inputs = self.inputs.borrow_mut();
		let target = inputs.get_mut(slot).ok_or("unknown tensor input slot")?;
		if target.shape != buffer.shape {
			return Err("replacement shape mismatch".into());
		}
		*target = buffer;
		Ok(())
	}
```

```sh check
cargo test --quiet --test tensor_operations
```

## 24. 非正方行列、複数 channel、複数 output channel の小例を固定値で比較する。
<!-- todo:11-024 -->

`tests/tensor_operations.rs` の `multi_channel_convolution_and_padding` で、この経路の値または拒否条件を確認する。

`tests/tensor_operations.rs` の `fn multi_channel_convolution_and_padding(`の定義を次の全体で置き換えます。

```rust file="tests/tensor_operations.rs" mode="replace" item="fn multi_channel_convolution_and_padding("
fn multi_channel_convolution_and_padding() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![1, 2, 2, 3])?;
	let w = g.input(1, vec![2, 2, 2, 2])?;
	let y = g.conv2d(x, w, (1, 0), 1, 1, 1)?;
	let values = (1..=12).map(|i| i as f32).collect::<Vec<_>>();
	let weights = (0..16).map(|i| (i % 5) as f32 - 2.).collect::<Vec<_>>();
	let inputs = [
		Buffer::new(vec![1, 2, 2, 3], values.clone())?,
		Buffer::new(vec![2, 2, 2, 2], weights.clone())?,
	];
	let actual = g.compile(&[y])?.run(&inputs)?;
	let mut expected = vec![];
	for o in 0..2 {
		for row in 0..3 {
			for col in 0..2 {
				let mut sum = 0.;
				for c in 0..2 {
					for kh in 0..2 {
						for kw in 0..2 {
							let ir = row as isize + kh as isize - 1;
							if ir >= 0 && ir < 2 {
								sum += values[(c * 2 + ir as usize) * 3 + col + kw]
									* weights[((o * 2 + c) * 2 + kh) * 2 + kw];
							}
						}
					}
				}
				expected.push(sum);
			}
		}
	}
	assert_eq!(actual[0].data, expected);
	let bias = g.input(2, vec![2])?;
	let bias = g.reshape(bias, vec![1, 2, 1, 1])?;
	let biased = g.binary(Binary::Add, y, bias)?;
	let mut bindings = inputs.to_vec();
	bindings.push(buffer(&[2], &[1., -2.]));
	let biased = g.compile(&[biased])?.run(&bindings)?;
	let expected = expected
		.iter()
		.enumerate()
		.map(|(i, &v)| v + if i < 6 { 1. } else { -2. })
		.collect::<Vec<_>>();
	assert_eq!(biased[0].data, expected);
	assert!(g.conv2d(x, w, (0, 0), 2, 1, 1).is_err());
	Ok(())
}
```

```sh check
cargo test --quiet --test tensor_operations multi_channel_convolution_and_padding
```

## 25. graph、schedule、生成 C の順序から同じ計算であると追える。
<!-- todo:11-025 -->

`tests/tensor_operations.rs` の正常系と境界系をまとめて再実行する。

`tests/tensor_operations.rs` の `fn matmul_conv(`の定義を次の全体で置き換えます。

```rust file="tests/tensor_operations.rs" mode="replace" item="fn matmul_conv("
fn matmul_conv() -> Result<()> {
	let mut g = Graph::default();
	let a = g.input(0, vec![2, 3])?;
	let b = g.input(1, vec![3, 2])?;
	let c = g.matmul(a, b)?;
	let out = compare(
		&g,
		&[c],
		&[
			buffer(&[2, 3], &[1., 2., 3., 4., 5., 6.]),
			buffer(&[3, 2], &[1., 2., 3., 4., 5., 6.]),
		],
	)?;
	assert_eq!(out[0].data, [22., 28., 49., 64.]);
	let mut g = Graph::default();
	let x = g.input(0, vec![1, 1, 1, 4])?;
	let w = g.input(1, vec![1, 1, 1, 2])?;
	let y = g.conv2d(x, w, (0, 0), 1, 1, 1)?;
	let out = compare(
		&g,
		&[y],
		&[
			buffer(&[1, 1, 1, 4], &[1., 2., 3., 4.]),
			buffer(&[1, 1, 1, 2], &[2., 1.]),
		],
	)?;
	assert_eq!(out[0].data, [4., 7., 10.]);
	Ok(())
}
```

```sh check
cargo test --quiet --test tensor_operations
```

## 章末の累積確認

前章までの example と test も同じ checkout で再ビルドする。

```sh check
cargo test --quiet --all-targets
```
