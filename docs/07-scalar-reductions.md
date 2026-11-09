# 07. scalar の累積ループで総和、行和、GEMM と convolution を動かす

accumulator は長さ1の REG storage である。ゼロ初期化を出力位置の region 内に置き、内側 RANGE で LOAD と STORE を繰り返す。高水準の REDUCE 専用実行器は使わない。生成コードと独立 interpreter で、空の縮約、行和、三重ループ GEMM、1D convolution を比較する。

`file=... mode=write` はそのファイルの全置換、`mode=append` は末尾への追記である。すべて前章でできたプロジェクトの root で行う。`sh check` は実際の検証コマンドであり、ソースファイルには保存しない。

## 1. REG allocation と初期化の region を実装する。
<!-- todo:07-001 -->

```rust file=src/reductions.rs mode=write
use crate::{
	Result,
	kernel::{Expr as E, Index as I, Kernel, range, store},
};
pub fn row_sum(rows: usize, cols: usize) -> Result<Kernel> {
	let mut k = Kernel::default();
	let len = rows.checked_mul(cols).ok_or("row size overflow")?;
	let x = k.input(0, len)?;
	let out = k.alloc(rows)?;
	let acc = k.reg()?;
	let index = I::Axis(0)
		.mul(I::Const(
			i32::try_from(cols).map_err(|_| "column overflow")?,
		))
		.add(I::Axis(1));
	k.body = vec![range(
		0,
		rows,
		vec![
			store(acc, I::Const(0), E::Const(0.)),
			range(
				1,
				cols,
				vec![store(
					acc,
					I::Const(0),
					E::Load(acc, I::Const(0)).add(E::Load(x, index)),
				)],
			),
			store(out, I::Axis(0), E::Load(acc, I::Const(0))),
		],
	)];
	k.outputs = vec![out];
	Ok(k)
}
pub fn matmul(m: usize, n: usize, inner: usize) -> Result<Kernel> {
	let mut k = Kernel::default();
	let a = k.input(0, m.checked_mul(inner).ok_or("matrix overflow")?)?;
	let b = k.input(1, inner.checked_mul(n).ok_or("matrix overflow")?)?;
	let out = k.alloc(m.checked_mul(n).ok_or("matrix overflow")?)?;
	let acc = k.reg()?;
	let ni = i32::try_from(n).map_err(|_| "matrix dimension overflow")?;
	let ki = i32::try_from(inner).map_err(|_| "matrix dimension overflow")?;
	let ai = I::Axis(0).mul(I::Const(ki)).add(I::Axis(2));
	let bi = I::Axis(2).mul(I::Const(ni)).add(I::Axis(1));
	let oi = I::Axis(0).mul(I::Const(ni)).add(I::Axis(1));
	k.body = vec![range(
		0,
		m,
		vec![range(
			1,
			n,
			vec![
				store(acc, I::Const(0), E::Const(0.)),
				range(
					2,
					inner,
					vec![store(
						acc,
						I::Const(0),
						E::Load(acc, I::Const(0)).add(E::Load(a, ai).mul(E::Load(b, bi))),
					)],
				),
				store(out, oi, E::Load(acc, I::Const(0))),
			],
		)],
	)];
	k.outputs = vec![out];
	Ok(k)
}
pub fn conv1d(len: usize, width: usize) -> Result<Kernel> {
	if width == 0 {
		return Err("empty convolution kernel".into());
	}
	let out_len = len
		.checked_sub(width)
		.and_then(|n| n.checked_add(1))
		.ok_or("convolution kernel longer than input")?;
	let mut k = Kernel::default();
	let x = k.input(0, len)?;
	let w = k.input(1, width)?;
	let out = k.alloc(out_len)?;
	let acc = k.reg()?;
	k.body = vec![range(
		0,
		out_len,
		vec![
			store(acc, I::Const(0), E::Const(0.)),
			range(
				1,
				width,
				vec![store(
					acc,
					I::Const(0),
					E::Load(acc, I::Const(0))
						.add(E::Load(x, I::Axis(0).add(I::Axis(1))).mul(E::Load(w, I::Axis(1)))),
				)],
			),
			store(out, I::Axis(0), E::Load(acc, I::Const(0))),
		],
	)];
	k.outputs = vec![out];
	Ok(k)
}
```

```rust file=src/lib.rs mode=append
pub mod reductions;
```

```rust file=tests/scalar_reductions.rs mode=write
use chibigrad::reductions;
use chibigrad::{
	Result,
	kernel::{Expr as E, Index as I, Kernel, range, store},
};

#[test]
fn scalar_reductions() -> Result<()> {
	assert_eq!(
		reductions::row_sum(1, 4)?
			.compile()?
			.run(&[vec![1., 2., 3., 4.]])?,
		vec![vec![10.]]
	);
	let k = reductions::row_sum(2, 3)?;
	assert_eq!(
		k.compile()?.run(&[vec![1., 2., 3., 4., 5., 6.]])?,
		vec![vec![6., 15.]]
	);
	assert_eq!(
		reductions::row_sum(2, 0)?.compile()?.run(&[vec![]])?,
		vec![vec![0., 0.]]
	);
	let k = reductions::matmul(2, 2, 3)?;
	let inputs = [vec![1., 2., 3., 4., 5., 6.], vec![1., 2., 3., 4., 5., 6.]];
	assert_eq!(k.compile()?.run(&inputs)?, vec![vec![22., 28., 49., 64.]]);
	assert_eq!(k.interpret(&inputs)?, vec![vec![22., 28., 49., 64.]]);
	let mut reference = vec![0.0; 4];
	for i in 0..2 {
		for j in 0..2 {
			for inner in 0..3 {
				reference[i * 2 + j] += inputs[0][i * 3 + inner] * inputs[1][inner * 2 + j];
			}
		}
	}
	assert_eq!(k.compile()?.run(&inputs)?[0], reference);
	assert_eq!(
		reductions::conv1d(4, 2)?
			.compile()?
			.run(&[vec![1., 2., 3., 4.], vec![2., 1.]])?,
		vec![vec![4., 7., 10.]]
	);
	Ok(())
}
```

`tests/scalar_reductions.rs` の `scalar_reductions` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test scalar_reductions scalar_reductions
```

## 2. 反復内 LOAD と STORE を interpreter と C で順に実行する。
<!-- todo:07-002 -->

`tests/scalar_reductions.rs` の `scalar_reductions` で、この経路の値または拒否条件を確認する。

`src/kernel.rs` の `fn walk(`の定義を次の全体で置き換えます。

```rust file="src/kernel.rs" mode="replace" item="fn walk("
	fn walk(
		&self,
		body: &[Stmt],
		axes: &mut HashMap<usize, i32>,
		memory: &mut [Vec<f32>],
		initialized: &mut [Vec<bool>],
		access: &mut Accesses,
		budget: &mut usize,
	) -> Result<()> {
		for statement in body {
			*budget = budget
				.checked_sub(1)
				.ok_or("verification work exceeds finite proof budget")?;
			match statement {
				Stmt::Store { view, index, value } => {
					if self
						.storage
						.get(view.storage)
						.ok_or("STORE unknown storage")?
						.input
						.is_some()
					{
						return Err("STORE to readonly input".into());
					}
					let address = self.address(*view, index, axes)?;
					let value = self.evaluate(value, axes, memory, initialized, access)?;
					memory[view.storage][address] = value;
					initialized[view.storage][address] = true;
					access.write.insert(view.storage);
				}
				Stmt::Range { axis, len, body } => {
					if axes.contains_key(axis) || *len > i32::MAX as usize {
						return Err("duplicate RANGE or i32 overflow".into());
					}
					for i in 0..*len {
						axes.insert(*axis, i as i32);
						self.walk(body, axes, memory, initialized, access, budget)?;
					}
					axes.remove(axis);
				}
			}
		}
		Ok(())
	}
```

```sh check
cargo test --quiet --test scalar_reductions scalar_reductions
```

## 3. ループ後の LOAD を END の完了へ依存させる。
<!-- todo:07-003 -->

`tests/scalar_reductions.rs` の `scalar_reductions` で、この経路の値または拒否条件を確認する。

`src/kernel.rs` の `fn walk(`の定義を次の全体で置き換えます。

```rust file="src/kernel.rs" mode="replace" item="fn walk("
	fn walk(
		&self,
		body: &[Stmt],
		axes: &mut HashMap<usize, i32>,
		memory: &mut [Vec<f32>],
		initialized: &mut [Vec<bool>],
		access: &mut Accesses,
		budget: &mut usize,
	) -> Result<()> {
		for statement in body {
			*budget = budget
				.checked_sub(1)
				.ok_or("verification work exceeds finite proof budget")?;
			match statement {
				Stmt::Store { view, index, value } => {
					if self
						.storage
						.get(view.storage)
						.ok_or("STORE unknown storage")?
						.input
						.is_some()
					{
						return Err("STORE to readonly input".into());
					}
					let address = self.address(*view, index, axes)?;
					let value = self.evaluate(value, axes, memory, initialized, access)?;
					memory[view.storage][address] = value;
					initialized[view.storage][address] = true;
					access.write.insert(view.storage);
				}
				Stmt::Range { axis, len, body } => {
					if axes.contains_key(axis) || *len > i32::MAX as usize {
						return Err("duplicate RANGE or i32 overflow".into());
					}
					for i in 0..*len {
						axes.insert(*axis, i as i32);
						self.walk(body, axes, memory, initialized, access, budget)?;
					}
					axes.remove(axis);
				}
			}
		}
		Ok(())
	}
```

```sh check
cargo test --quiet --test scalar_reductions scalar_reductions
```

## 4. `[1,2,3,4]` の総和を生成コードで `10` にする。
<!-- todo:07-004 -->

`tests/scalar_reductions.rs` の `scalar_reductions` で、この経路の値または拒否条件を確認する。

`tests/scalar_reductions.rs` の `fn scalar_reductions(`の定義を次の全体で置き換えます。

```rust file="tests/scalar_reductions.rs" mode="replace" item="fn scalar_reductions("
fn scalar_reductions() -> Result<()> {
	assert_eq!(
		reductions::row_sum(1, 4)?
			.compile()?
			.run(&[vec![1., 2., 3., 4.]])?,
		vec![vec![10.]]
	);
	let k = reductions::row_sum(2, 3)?;
	assert_eq!(
		k.compile()?.run(&[vec![1., 2., 3., 4., 5., 6.]])?,
		vec![vec![6., 15.]]
	);
	assert_eq!(
		reductions::row_sum(2, 0)?.compile()?.run(&[vec![]])?,
		vec![vec![0., 0.]]
	);
	let k = reductions::matmul(2, 2, 3)?;
	let inputs = [vec![1., 2., 3., 4., 5., 6.], vec![1., 2., 3., 4., 5., 6.]];
	assert_eq!(k.compile()?.run(&inputs)?, vec![vec![22., 28., 49., 64.]]);
	assert_eq!(k.interpret(&inputs)?, vec![vec![22., 28., 49., 64.]]);
	let mut reference = vec![0.0; 4];
	for i in 0..2 {
		for j in 0..2 {
			for inner in 0..3 {
				reference[i * 2 + j] += inputs[0][i * 3 + inner] * inputs[1][inner * 2 + j];
			}
		}
	}
	assert_eq!(k.compile()?.run(&inputs)?[0], reference);
	assert_eq!(
		reductions::conv1d(4, 2)?
			.compile()?
			.run(&[vec![1., 2., 3., 4.], vec![2., 1.]])?,
		vec![vec![4., 7., 10.]]
	);
	Ok(())
}
```

```sh check
cargo test --quiet --test scalar_reductions scalar_reductions
```

## 5. 長さ 0 では初期値 `0` を返す。
<!-- todo:07-005 -->

`tests/scalar_reductions.rs` の `scalar_reductions` で、この経路の値または拒否条件を確認する。

`src/reductions.rs` の `pub fn row_sum(`の定義を次の全体で置き換えます。

```rust file="src/reductions.rs" mode="replace" item="pub fn row_sum("
pub fn row_sum(rows: usize, cols: usize) -> Result<Kernel> {
	let mut k = Kernel::default();
	let len = rows.checked_mul(cols).ok_or("row size overflow")?;
	let x = k.input(0, len)?;
	let out = k.alloc(rows)?;
	let acc = k.reg()?;
	let index = I::Axis(0)
		.mul(I::Const(
			i32::try_from(cols).map_err(|_| "column overflow")?,
		))
		.add(I::Axis(1));
	k.body = vec![range(
		0,
		rows,
		vec![
			store(acc, I::Const(0), E::Const(0.)),
			range(
				1,
				cols,
				vec![store(
					acc,
					I::Const(0),
					E::Load(acc, I::Const(0)).add(E::Load(x, index)),
				)],
			),
			store(out, I::Axis(0), E::Load(acc, I::Const(0))),
		],
	)];
	k.outputs = vec![out];
	Ok(k)
}
```

```sh check
cargo test --quiet --test scalar_reductions scalar_reductions
```

## 6. 外側 RANGE の body に accumulator 初期化を置く。
<!-- todo:07-006 -->

`tests/scalar_reductions.rs` の `scalar_reductions` で、この経路の値または拒否条件を確認する。

`src/reductions.rs` の `pub fn row_sum(`の定義を次の全体で置き換えます。

```rust file="src/reductions.rs" mode="replace" item="pub fn row_sum("
pub fn row_sum(rows: usize, cols: usize) -> Result<Kernel> {
	let mut k = Kernel::default();
	let len = rows.checked_mul(cols).ok_or("row size overflow")?;
	let x = k.input(0, len)?;
	let out = k.alloc(rows)?;
	let acc = k.reg()?;
	let index = I::Axis(0)
		.mul(I::Const(
			i32::try_from(cols).map_err(|_| "column overflow")?,
		))
		.add(I::Axis(1));
	k.body = vec![range(
		0,
		rows,
		vec![
			store(acc, I::Const(0), E::Const(0.)),
			range(
				1,
				cols,
				vec![store(
					acc,
					I::Const(0),
					E::Load(acc, I::Const(0)).add(E::Load(x, index)),
				)],
			),
			store(out, I::Axis(0), E::Load(acc, I::Const(0))),
		],
	)];
	k.outputs = vec![out];
	Ok(k)
}
```

```sh check
cargo test --quiet --test scalar_reductions scalar_reductions
```

## 7. 内側 RANGE で列を累積する。
<!-- todo:07-007 -->

`tests/scalar_reductions.rs` の `scalar_reductions` で、この経路の値または拒否条件を確認する。

`src/reductions.rs` の `pub fn row_sum(`の定義を次の全体で置き換えます。

```rust file="src/reductions.rs" mode="replace" item="pub fn row_sum("
pub fn row_sum(rows: usize, cols: usize) -> Result<Kernel> {
	let mut k = Kernel::default();
	let len = rows.checked_mul(cols).ok_or("row size overflow")?;
	let x = k.input(0, len)?;
	let out = k.alloc(rows)?;
	let acc = k.reg()?;
	let index = I::Axis(0)
		.mul(I::Const(
			i32::try_from(cols).map_err(|_| "column overflow")?,
		))
		.add(I::Axis(1));
	k.body = vec![range(
		0,
		rows,
		vec![
			store(acc, I::Const(0), E::Const(0.)),
			range(
				1,
				cols,
				vec![store(
					acc,
					I::Const(0),
					E::Load(acc, I::Const(0)).add(E::Load(x, index)),
				)],
			),
			store(out, I::Axis(0), E::Load(acc, I::Const(0))),
		],
	)];
	k.outputs = vec![out];
	Ok(k)
}
```

```sh check
cargo test --quiet --test scalar_reductions scalar_reductions
```

## 8. 内側 END の後、外側 END の前に行の出力を STORE する。
<!-- todo:07-008 -->

`tests/scalar_reductions.rs` の `scalar_reductions` で、この経路の値または拒否条件を確認する。

`src/reductions.rs` の `pub fn row_sum(`の定義を次の全体で置き換えます。

```rust file="src/reductions.rs" mode="replace" item="pub fn row_sum("
pub fn row_sum(rows: usize, cols: usize) -> Result<Kernel> {
	let mut k = Kernel::default();
	let len = rows.checked_mul(cols).ok_or("row size overflow")?;
	let x = k.input(0, len)?;
	let out = k.alloc(rows)?;
	let acc = k.reg()?;
	let index = I::Axis(0)
		.mul(I::Const(
			i32::try_from(cols).map_err(|_| "column overflow")?,
		))
		.add(I::Axis(1));
	k.body = vec![range(
		0,
		rows,
		vec![
			store(acc, I::Const(0), E::Const(0.)),
			range(
				1,
				cols,
				vec![store(
					acc,
					I::Const(0),
					E::Load(acc, I::Const(0)).add(E::Load(x, index)),
				)],
			),
			store(out, I::Axis(0), E::Load(acc, I::Const(0))),
		],
	)];
	k.outputs = vec![out];
	Ok(k)
}
```

```sh check
cargo test --quiet --test scalar_reductions scalar_reductions
```

## 9. `[1,2,3,4,5,6]` の行和を `[6,15]` にする。
<!-- todo:07-009 -->

`tests/scalar_reductions.rs` の `scalar_reductions` で、この経路の値または拒否条件を確認する。

`tests/scalar_reductions.rs` の `fn scalar_reductions(`の定義を次の全体で置き換えます。

```rust file="tests/scalar_reductions.rs" mode="replace" item="fn scalar_reductions("
fn scalar_reductions() -> Result<()> {
	assert_eq!(
		reductions::row_sum(1, 4)?
			.compile()?
			.run(&[vec![1., 2., 3., 4.]])?,
		vec![vec![10.]]
	);
	let k = reductions::row_sum(2, 3)?;
	assert_eq!(
		k.compile()?.run(&[vec![1., 2., 3., 4., 5., 6.]])?,
		vec![vec![6., 15.]]
	);
	assert_eq!(
		reductions::row_sum(2, 0)?.compile()?.run(&[vec![]])?,
		vec![vec![0., 0.]]
	);
	let k = reductions::matmul(2, 2, 3)?;
	let inputs = [vec![1., 2., 3., 4., 5., 6.], vec![1., 2., 3., 4., 5., 6.]];
	assert_eq!(k.compile()?.run(&inputs)?, vec![vec![22., 28., 49., 64.]]);
	assert_eq!(k.interpret(&inputs)?, vec![vec![22., 28., 49., 64.]]);
	let mut reference = vec![0.0; 4];
	for i in 0..2 {
		for j in 0..2 {
			for inner in 0..3 {
				reference[i * 2 + j] += inputs[0][i * 3 + inner] * inputs[1][inner * 2 + j];
			}
		}
	}
	assert_eq!(k.compile()?.run(&inputs)?[0], reference);
	assert_eq!(
		reductions::conv1d(4, 2)?
			.compile()?
			.run(&[vec![1., 2., 3., 4.], vec![2., 1.]])?,
		vec![vec![4., 7., 10.]]
	);
	Ok(())
}
```

```sh check
cargo test --quiet --test scalar_reductions scalar_reductions
```

## 10. 空の列数なら各行を `0` にする。
<!-- todo:07-010 -->

`tests/scalar_reductions.rs` の `scalar_reductions` で、この経路の値または拒否条件を確認する。

`src/reductions.rs` の `pub fn row_sum(`の定義を次の全体で置き換えます。

```rust file="src/reductions.rs" mode="replace" item="pub fn row_sum("
pub fn row_sum(rows: usize, cols: usize) -> Result<Kernel> {
	let mut k = Kernel::default();
	let len = rows.checked_mul(cols).ok_or("row size overflow")?;
	let x = k.input(0, len)?;
	let out = k.alloc(rows)?;
	let acc = k.reg()?;
	let index = I::Axis(0)
		.mul(I::Const(
			i32::try_from(cols).map_err(|_| "column overflow")?,
		))
		.add(I::Axis(1));
	k.body = vec![range(
		0,
		rows,
		vec![
			store(acc, I::Const(0), E::Const(0.)),
			range(
				1,
				cols,
				vec![store(
					acc,
					I::Const(0),
					E::Load(acc, I::Const(0)).add(E::Load(x, index)),
				)],
			),
			store(out, I::Axis(0), E::Load(acc, I::Const(0))),
		],
	)];
	k.outputs = vec![out];
	Ok(k)
}
```

```sh check
cargo test --quiet --test scalar_reductions scalar_reductions
```

## 11. `i`、`j`、`k` の三軸と row-major 添字を手で作る。
<!-- todo:07-011 -->

`tests/scalar_reductions.rs` の `scalar_reductions` で、この経路の値または拒否条件を確認する。

`src/reductions.rs` の `pub fn matmul(`の定義を次の全体で置き換えます。

```rust file="src/reductions.rs" mode="replace" item="pub fn matmul("
pub fn matmul(m: usize, n: usize, inner: usize) -> Result<Kernel> {
	let mut k = Kernel::default();
	let a = k.input(0, m.checked_mul(inner).ok_or("matrix overflow")?)?;
	let b = k.input(1, inner.checked_mul(n).ok_or("matrix overflow")?)?;
	let out = k.alloc(m.checked_mul(n).ok_or("matrix overflow")?)?;
	let acc = k.reg()?;
	let ni = i32::try_from(n).map_err(|_| "matrix dimension overflow")?;
	let ki = i32::try_from(inner).map_err(|_| "matrix dimension overflow")?;
	let ai = I::Axis(0).mul(I::Const(ki)).add(I::Axis(2));
	let bi = I::Axis(2).mul(I::Const(ni)).add(I::Axis(1));
	let oi = I::Axis(0).mul(I::Const(ni)).add(I::Axis(1));
	k.body = vec![range(
		0,
		m,
		vec![range(
			1,
			n,
			vec![
				store(acc, I::Const(0), E::Const(0.)),
				range(
					2,
					inner,
					vec![store(
						acc,
						I::Const(0),
						E::Load(acc, I::Const(0)).add(E::Load(a, ai).mul(E::Load(b, bi))),
					)],
				),
				store(out, oi, E::Load(acc, I::Const(0))),
			],
		)],
	)];
	k.outputs = vec![out];
	Ok(k)
}
```

```sh check
cargo test --quiet --test scalar_reductions scalar_reductions
```

## 12. 各 `(i,j)` で accumulator をゼロへ戻す。
<!-- todo:07-012 -->

`tests/scalar_reductions.rs` の `scalar_reductions` で、この経路の値または拒否条件を確認する。

`src/reductions.rs` の `pub fn matmul(`の定義を次の全体で置き換えます。

```rust file="src/reductions.rs" mode="replace" item="pub fn matmul("
pub fn matmul(m: usize, n: usize, inner: usize) -> Result<Kernel> {
	let mut k = Kernel::default();
	let a = k.input(0, m.checked_mul(inner).ok_or("matrix overflow")?)?;
	let b = k.input(1, inner.checked_mul(n).ok_or("matrix overflow")?)?;
	let out = k.alloc(m.checked_mul(n).ok_or("matrix overflow")?)?;
	let acc = k.reg()?;
	let ni = i32::try_from(n).map_err(|_| "matrix dimension overflow")?;
	let ki = i32::try_from(inner).map_err(|_| "matrix dimension overflow")?;
	let ai = I::Axis(0).mul(I::Const(ki)).add(I::Axis(2));
	let bi = I::Axis(2).mul(I::Const(ni)).add(I::Axis(1));
	let oi = I::Axis(0).mul(I::Const(ni)).add(I::Axis(1));
	k.body = vec![range(
		0,
		m,
		vec![range(
			1,
			n,
			vec![
				store(acc, I::Const(0), E::Const(0.)),
				range(
					2,
					inner,
					vec![store(
						acc,
						I::Const(0),
						E::Load(acc, I::Const(0)).add(E::Load(a, ai).mul(E::Load(b, bi))),
					)],
				),
				store(out, oi, E::Load(acc, I::Const(0))),
			],
		)],
	)];
	k.outputs = vec![out];
	Ok(k)
}
```

```sh check
cargo test --quiet --test scalar_reductions scalar_reductions
```

## 13. reduction 軸 k の END の後に出力を STORE する。
<!-- todo:07-013 -->

`tests/scalar_reductions.rs` の `scalar_reductions` で、この経路の値または拒否条件を確認する。

`src/reductions.rs` の `pub fn matmul(`の定義を次の全体で置き換えます。

```rust file="src/reductions.rs" mode="replace" item="pub fn matmul("
pub fn matmul(m: usize, n: usize, inner: usize) -> Result<Kernel> {
	let mut k = Kernel::default();
	let a = k.input(0, m.checked_mul(inner).ok_or("matrix overflow")?)?;
	let b = k.input(1, inner.checked_mul(n).ok_or("matrix overflow")?)?;
	let out = k.alloc(m.checked_mul(n).ok_or("matrix overflow")?)?;
	let acc = k.reg()?;
	let ni = i32::try_from(n).map_err(|_| "matrix dimension overflow")?;
	let ki = i32::try_from(inner).map_err(|_| "matrix dimension overflow")?;
	let ai = I::Axis(0).mul(I::Const(ki)).add(I::Axis(2));
	let bi = I::Axis(2).mul(I::Const(ni)).add(I::Axis(1));
	let oi = I::Axis(0).mul(I::Const(ni)).add(I::Axis(1));
	k.body = vec![range(
		0,
		m,
		vec![range(
			1,
			n,
			vec![
				store(acc, I::Const(0), E::Const(0.)),
				range(
					2,
					inner,
					vec![store(
						acc,
						I::Const(0),
						E::Load(acc, I::Const(0)).add(E::Load(a, ai).mul(E::Load(b, bi))),
					)],
				),
				store(out, oi, E::Load(acc, I::Const(0))),
			],
		)],
	)];
	k.outputs = vec![out];
	Ok(k)
}
```

```sh check
cargo test --quiet --test scalar_reductions scalar_reductions
```

## 14. 2×3 と 3×2 の入力を既存 CPU runtime で呼ぶ。
<!-- todo:07-014 -->

`tests/scalar_reductions.rs` の `scalar_reductions` で、この経路の値または拒否条件を確認する。

`tests/scalar_reductions.rs` の `fn scalar_reductions(`の定義を次の全体で置き換えます。

```rust file="tests/scalar_reductions.rs" mode="replace" item="fn scalar_reductions("
fn scalar_reductions() -> Result<()> {
	assert_eq!(
		reductions::row_sum(1, 4)?
			.compile()?
			.run(&[vec![1., 2., 3., 4.]])?,
		vec![vec![10.]]
	);
	let k = reductions::row_sum(2, 3)?;
	assert_eq!(
		k.compile()?.run(&[vec![1., 2., 3., 4., 5., 6.]])?,
		vec![vec![6., 15.]]
	);
	assert_eq!(
		reductions::row_sum(2, 0)?.compile()?.run(&[vec![]])?,
		vec![vec![0., 0.]]
	);
	let k = reductions::matmul(2, 2, 3)?;
	let inputs = [vec![1., 2., 3., 4., 5., 6.], vec![1., 2., 3., 4., 5., 6.]];
	assert_eq!(k.compile()?.run(&inputs)?, vec![vec![22., 28., 49., 64.]]);
	assert_eq!(k.interpret(&inputs)?, vec![vec![22., 28., 49., 64.]]);
	let mut reference = vec![0.0; 4];
	for i in 0..2 {
		for j in 0..2 {
			for inner in 0..3 {
				reference[i * 2 + j] += inputs[0][i * 3 + inner] * inputs[1][inner * 2 + j];
			}
		}
	}
	assert_eq!(k.compile()?.run(&inputs)?[0], reference);
	assert_eq!(
		reductions::conv1d(4, 2)?
			.compile()?
			.run(&[vec![1., 2., 3., 4.], vec![2., 1.]])?,
		vec![vec![4., 7., 10.]]
	);
	Ok(())
}
```

```sh check
cargo test --quiet --test scalar_reductions scalar_reductions
```

## 15. ホストの独立した三重ループと結果を比較する。
<!-- todo:07-015 -->

`tests/scalar_reductions.rs` の `scalar_reductions` で、この経路の値または拒否条件を確認する。

`tests/scalar_reductions.rs` の `fn scalar_reductions(`の定義を次の全体で置き換えます。

```rust file="tests/scalar_reductions.rs" mode="replace" item="fn scalar_reductions("
fn scalar_reductions() -> Result<()> {
	assert_eq!(
		reductions::row_sum(1, 4)?
			.compile()?
			.run(&[vec![1., 2., 3., 4.]])?,
		vec![vec![10.]]
	);
	let k = reductions::row_sum(2, 3)?;
	assert_eq!(
		k.compile()?.run(&[vec![1., 2., 3., 4., 5., 6.]])?,
		vec![vec![6., 15.]]
	);
	assert_eq!(
		reductions::row_sum(2, 0)?.compile()?.run(&[vec![]])?,
		vec![vec![0., 0.]]
	);
	let k = reductions::matmul(2, 2, 3)?;
	let inputs = [vec![1., 2., 3., 4., 5., 6.], vec![1., 2., 3., 4., 5., 6.]];
	assert_eq!(k.compile()?.run(&inputs)?, vec![vec![22., 28., 49., 64.]]);
	assert_eq!(k.interpret(&inputs)?, vec![vec![22., 28., 49., 64.]]);
	let mut reference = vec![0.0; 4];
	for i in 0..2 {
		for j in 0..2 {
			for inner in 0..3 {
				reference[i * 2 + j] += inputs[0][i * 3 + inner] * inputs[1][inner * 2 + j];
			}
		}
	}
	assert_eq!(k.compile()?.run(&inputs)?[0], reference);
	assert_eq!(
		reductions::conv1d(4, 2)?
			.compile()?
			.run(&[vec![1., 2., 3., 4.], vec![2., 1.]])?,
		vec![vec![4., 7., 10.]]
	);
	Ok(())
}
```

```sh check
cargo test --quiet --test scalar_reductions scalar_reductions
```

## 16. 出力位置 p と kernel 位置 k を RANGE で作る。
<!-- todo:07-016 -->

`tests/scalar_reductions.rs` の `scalar_reductions` で、この経路の値または拒否条件を確認する。

`src/reductions.rs` の `pub fn conv1d(`の定義を次の全体で置き換えます。

```rust file="src/reductions.rs" mode="replace" item="pub fn conv1d("
pub fn conv1d(len: usize, width: usize) -> Result<Kernel> {
	if width == 0 {
		return Err("empty convolution kernel".into());
	}
	let out_len = len
		.checked_sub(width)
		.and_then(|n| n.checked_add(1))
		.ok_or("convolution kernel longer than input")?;
	let mut k = Kernel::default();
	let x = k.input(0, len)?;
	let w = k.input(1, width)?;
	let out = k.alloc(out_len)?;
	let acc = k.reg()?;
	k.body = vec![range(
		0,
		out_len,
		vec![
			store(acc, I::Const(0), E::Const(0.)),
			range(
				1,
				width,
				vec![store(
					acc,
					I::Const(0),
					E::Load(acc, I::Const(0))
						.add(E::Load(x, I::Axis(0).add(I::Axis(1))).mul(E::Load(w, I::Axis(1)))),
				)],
			),
			store(out, I::Axis(0), E::Load(acc, I::Const(0))),
		],
	)];
	k.outputs = vec![out];
	Ok(k)
}
```

```sh check
cargo test --quiet --test scalar_reductions scalar_reductions
```

## 17. `x[p+k]*w[k]` を累積する。
<!-- todo:07-017 -->

`tests/scalar_reductions.rs` の `scalar_reductions` で、この経路の値または拒否条件を確認する。

`src/reductions.rs` の `pub fn conv1d(`の定義を次の全体で置き換えます。

```rust file="src/reductions.rs" mode="replace" item="pub fn conv1d("
pub fn conv1d(len: usize, width: usize) -> Result<Kernel> {
	if width == 0 {
		return Err("empty convolution kernel".into());
	}
	let out_len = len
		.checked_sub(width)
		.and_then(|n| n.checked_add(1))
		.ok_or("convolution kernel longer than input")?;
	let mut k = Kernel::default();
	let x = k.input(0, len)?;
	let w = k.input(1, width)?;
	let out = k.alloc(out_len)?;
	let acc = k.reg()?;
	k.body = vec![range(
		0,
		out_len,
		vec![
			store(acc, I::Const(0), E::Const(0.)),
			range(
				1,
				width,
				vec![store(
					acc,
					I::Const(0),
					E::Load(acc, I::Const(0))
						.add(E::Load(x, I::Axis(0).add(I::Axis(1))).mul(E::Load(w, I::Axis(1)))),
				)],
			),
			store(out, I::Axis(0), E::Load(acc, I::Const(0))),
		],
	)];
	k.outputs = vec![out];
	Ok(k)
}
```

```sh check
cargo test --quiet --test scalar_reductions scalar_reductions
```

## 18. `output_len=input_len-kernel_len+1` を checked 演算で求める。
<!-- todo:07-018 -->

`tests/scalar_reductions.rs` の `scalar_reductions` で、この経路の値または拒否条件を確認する。

`src/reductions.rs` の `pub fn conv1d(`の定義を次の全体で置き換えます。

```rust file="src/reductions.rs" mode="replace" item="pub fn conv1d("
pub fn conv1d(len: usize, width: usize) -> Result<Kernel> {
	if width == 0 {
		return Err("empty convolution kernel".into());
	}
	let out_len = len
		.checked_sub(width)
		.and_then(|n| n.checked_add(1))
		.ok_or("convolution kernel longer than input")?;
	let mut k = Kernel::default();
	let x = k.input(0, len)?;
	let w = k.input(1, width)?;
	let out = k.alloc(out_len)?;
	let acc = k.reg()?;
	k.body = vec![range(
		0,
		out_len,
		vec![
			store(acc, I::Const(0), E::Const(0.)),
			range(
				1,
				width,
				vec![store(
					acc,
					I::Const(0),
					E::Load(acc, I::Const(0))
						.add(E::Load(x, I::Axis(0).add(I::Axis(1))).mul(E::Load(w, I::Axis(1)))),
				)],
			),
			store(out, I::Axis(0), E::Load(acc, I::Const(0))),
		],
	)];
	k.outputs = vec![out];
	Ok(k)
}
```

```sh check
cargo test --quiet --test scalar_reductions scalar_reductions
```

## 19. `x=[1,2,3,4]`, `w=[2,1]` を `[4,7,10]` にする。
<!-- todo:07-019 -->

`tests/scalar_reductions.rs` の `scalar_reductions` で、この経路の値または拒否条件を確認する。

`tests/scalar_reductions.rs` の `fn scalar_reductions(`の定義を次の全体で置き換えます。

```rust file="tests/scalar_reductions.rs" mode="replace" item="fn scalar_reductions("
fn scalar_reductions() -> Result<()> {
	assert_eq!(
		reductions::row_sum(1, 4)?
			.compile()?
			.run(&[vec![1., 2., 3., 4.]])?,
		vec![vec![10.]]
	);
	let k = reductions::row_sum(2, 3)?;
	assert_eq!(
		k.compile()?.run(&[vec![1., 2., 3., 4., 5., 6.]])?,
		vec![vec![6., 15.]]
	);
	assert_eq!(
		reductions::row_sum(2, 0)?.compile()?.run(&[vec![]])?,
		vec![vec![0., 0.]]
	);
	let k = reductions::matmul(2, 2, 3)?;
	let inputs = [vec![1., 2., 3., 4., 5., 6.], vec![1., 2., 3., 4., 5., 6.]];
	assert_eq!(k.compile()?.run(&inputs)?, vec![vec![22., 28., 49., 64.]]);
	assert_eq!(k.interpret(&inputs)?, vec![vec![22., 28., 49., 64.]]);
	let mut reference = vec![0.0; 4];
	for i in 0..2 {
		for j in 0..2 {
			for inner in 0..3 {
				reference[i * 2 + j] += inputs[0][i * 3 + inner] * inputs[1][inner * 2 + j];
			}
		}
	}
	assert_eq!(k.compile()?.run(&inputs)?[0], reference);
	assert_eq!(
		reductions::conv1d(4, 2)?
			.compile()?
			.run(&[vec![1., 2., 3., 4.], vec![2., 1.]])?,
		vec![vec![4., 7., 10.]]
	);
	Ok(())
}
```

```sh check
cargo test --quiet --test scalar_reductions scalar_reductions
```

## 20. kernel が入力より長い場合を明確な未対応エラーにする。
<!-- todo:07-020 -->

`tests/scalar_reductions.rs` の `scalar_reductions` で、この経路の値または拒否条件を確認する。

`src/reductions.rs` の `pub fn conv1d(`の定義を次の全体で置き換えます。

```rust file="src/reductions.rs" mode="replace" item="pub fn conv1d("
pub fn conv1d(len: usize, width: usize) -> Result<Kernel> {
	if width == 0 {
		return Err("empty convolution kernel".into());
	}
	let out_len = len
		.checked_sub(width)
		.and_then(|n| n.checked_add(1))
		.ok_or("convolution kernel longer than input")?;
	let mut k = Kernel::default();
	let x = k.input(0, len)?;
	let w = k.input(1, width)?;
	let out = k.alloc(out_len)?;
	let acc = k.reg()?;
	k.body = vec![range(
		0,
		out_len,
		vec![
			store(acc, I::Const(0), E::Const(0.)),
			range(
				1,
				width,
				vec![store(
					acc,
					I::Const(0),
					E::Load(acc, I::Const(0))
						.add(E::Load(x, I::Axis(0).add(I::Axis(1))).mul(E::Load(w, I::Axis(1)))),
				)],
			),
			store(out, I::Axis(0), E::Load(acc, I::Const(0))),
		],
	)];
	k.outputs = vec![out];
	Ok(k)
}
```

```sh check
cargo test --quiet --test scalar_reductions scalar_reductions
```

## 21. 総和が `10`、行和が `[6,15]` になる。
<!-- todo:07-021 -->

`tests/scalar_reductions.rs` の `scalar_reductions` で、この経路の値または拒否条件を確認する。

`tests/scalar_reductions.rs` の `fn scalar_reductions(`の定義を次の全体で置き換えます。

```rust file="tests/scalar_reductions.rs" mode="replace" item="fn scalar_reductions("
fn scalar_reductions() -> Result<()> {
	assert_eq!(
		reductions::row_sum(1, 4)?
			.compile()?
			.run(&[vec![1., 2., 3., 4.]])?,
		vec![vec![10.]]
	);
	let k = reductions::row_sum(2, 3)?;
	assert_eq!(
		k.compile()?.run(&[vec![1., 2., 3., 4., 5., 6.]])?,
		vec![vec![6., 15.]]
	);
	assert_eq!(
		reductions::row_sum(2, 0)?.compile()?.run(&[vec![]])?,
		vec![vec![0., 0.]]
	);
	let k = reductions::matmul(2, 2, 3)?;
	let inputs = [vec![1., 2., 3., 4., 5., 6.], vec![1., 2., 3., 4., 5., 6.]];
	assert_eq!(k.compile()?.run(&inputs)?, vec![vec![22., 28., 49., 64.]]);
	assert_eq!(k.interpret(&inputs)?, vec![vec![22., 28., 49., 64.]]);
	let mut reference = vec![0.0; 4];
	for i in 0..2 {
		for j in 0..2 {
			for inner in 0..3 {
				reference[i * 2 + j] += inputs[0][i * 3 + inner] * inputs[1][inner * 2 + j];
			}
		}
	}
	assert_eq!(k.compile()?.run(&inputs)?[0], reference);
	assert_eq!(
		reductions::conv1d(4, 2)?
			.compile()?
			.run(&[vec![1., 2., 3., 4.], vec![2., 1.]])?,
		vec![vec![4., 7., 10.]]
	);
	Ok(())
}
```

```sh check
cargo test --quiet --test scalar_reductions scalar_reductions
```

## 22. 空の総和と列数 0 の行和が加算の初期値を返す。
<!-- todo:07-022 -->

`tests/scalar_reductions.rs` の `scalar_reductions` で、この経路の値または拒否条件を確認する。

`tests/scalar_reductions.rs` の `fn scalar_reductions(`の定義を次の全体で置き換えます。

```rust file="tests/scalar_reductions.rs" mode="replace" item="fn scalar_reductions("
fn scalar_reductions() -> Result<()> {
	assert_eq!(
		reductions::row_sum(1, 4)?
			.compile()?
			.run(&[vec![1., 2., 3., 4.]])?,
		vec![vec![10.]]
	);
	let k = reductions::row_sum(2, 3)?;
	assert_eq!(
		k.compile()?.run(&[vec![1., 2., 3., 4., 5., 6.]])?,
		vec![vec![6., 15.]]
	);
	assert_eq!(
		reductions::row_sum(2, 0)?.compile()?.run(&[vec![]])?,
		vec![vec![0., 0.]]
	);
	let k = reductions::matmul(2, 2, 3)?;
	let inputs = [vec![1., 2., 3., 4., 5., 6.], vec![1., 2., 3., 4., 5., 6.]];
	assert_eq!(k.compile()?.run(&inputs)?, vec![vec![22., 28., 49., 64.]]);
	assert_eq!(k.interpret(&inputs)?, vec![vec![22., 28., 49., 64.]]);
	let mut reference = vec![0.0; 4];
	for i in 0..2 {
		for j in 0..2 {
			for inner in 0..3 {
				reference[i * 2 + j] += inputs[0][i * 3 + inner] * inputs[1][inner * 2 + j];
			}
		}
	}
	assert_eq!(k.compile()?.run(&inputs)?[0], reference);
	assert_eq!(
		reductions::conv1d(4, 2)?
			.compile()?
			.run(&[vec![1., 2., 3., 4.], vec![2., 1.]])?,
		vec![vec![4., 7., 10.]]
	);
	Ok(())
}
```

```sh check
cargo test --quiet --test scalar_reductions scalar_reductions
```

## 23. GEMM が `[[22,28],[49,64]]`、convolution が `[4,7,10]` になる。
<!-- todo:07-023 -->

`tests/scalar_reductions.rs` の `scalar_reductions` で、この経路の値または拒否条件を確認する。

`tests/scalar_reductions.rs` の `fn scalar_reductions(`の定義を次の全体で置き換えます。

```rust file="tests/scalar_reductions.rs" mode="replace" item="fn scalar_reductions("
fn scalar_reductions() -> Result<()> {
	assert_eq!(
		reductions::row_sum(1, 4)?
			.compile()?
			.run(&[vec![1., 2., 3., 4.]])?,
		vec![vec![10.]]
	);
	let k = reductions::row_sum(2, 3)?;
	assert_eq!(
		k.compile()?.run(&[vec![1., 2., 3., 4., 5., 6.]])?,
		vec![vec![6., 15.]]
	);
	assert_eq!(
		reductions::row_sum(2, 0)?.compile()?.run(&[vec![]])?,
		vec![vec![0., 0.]]
	);
	let k = reductions::matmul(2, 2, 3)?;
	let inputs = [vec![1., 2., 3., 4., 5., 6.], vec![1., 2., 3., 4., 5., 6.]];
	assert_eq!(k.compile()?.run(&inputs)?, vec![vec![22., 28., 49., 64.]]);
	assert_eq!(k.interpret(&inputs)?, vec![vec![22., 28., 49., 64.]]);
	let mut reference = vec![0.0; 4];
	for i in 0..2 {
		for j in 0..2 {
			for inner in 0..3 {
				reference[i * 2 + j] += inputs[0][i * 3 + inner] * inputs[1][inner * 2 + j];
			}
		}
	}
	assert_eq!(k.compile()?.run(&inputs)?[0], reference);
	assert_eq!(
		reductions::conv1d(4, 2)?
			.compile()?
			.run(&[vec![1., 2., 3., 4.], vec![2., 1.]])?,
		vec![vec![4., 7., 10.]]
	);
	Ok(())
}
```

```sh check
cargo test --quiet --test scalar_reductions scalar_reductions
```

## 24. dump に高水準 REDUCE がなく、累積を行う LOAD と STORE がある。
<!-- todo:07-024 -->

`tests/scalar_reductions.rs` の `scalar_reductions` で、この経路の値または拒否条件を確認する。

`tests/scalar_reductions.rs` の `fn scalar_reductions(`の定義を次の全体で置き換えます。

```rust file="tests/scalar_reductions.rs" mode="replace" item="fn scalar_reductions("
fn scalar_reductions() -> Result<()> {
	assert_eq!(
		reductions::row_sum(1, 4)?
			.compile()?
			.run(&[vec![1., 2., 3., 4.]])?,
		vec![vec![10.]]
	);
	let k = reductions::row_sum(2, 3)?;
	assert_eq!(
		k.compile()?.run(&[vec![1., 2., 3., 4., 5., 6.]])?,
		vec![vec![6., 15.]]
	);
	assert_eq!(
		reductions::row_sum(2, 0)?.compile()?.run(&[vec![]])?,
		vec![vec![0., 0.]]
	);
	let k = reductions::matmul(2, 2, 3)?;
	let inputs = [vec![1., 2., 3., 4., 5., 6.], vec![1., 2., 3., 4., 5., 6.]];
	assert_eq!(k.compile()?.run(&inputs)?, vec![vec![22., 28., 49., 64.]]);
	assert_eq!(k.interpret(&inputs)?, vec![vec![22., 28., 49., 64.]]);
	let mut reference = vec![0.0; 4];
	for i in 0..2 {
		for j in 0..2 {
			for inner in 0..3 {
				reference[i * 2 + j] += inputs[0][i * 3 + inner] * inputs[1][inner * 2 + j];
			}
		}
	}
	assert_eq!(k.compile()?.run(&inputs)?[0], reference);
	assert_eq!(
		reductions::conv1d(4, 2)?
			.compile()?
			.run(&[vec![1., 2., 3., 4.], vec![2., 1.]])?,
		vec![vec![4., 7., 10.]]
	);
	Ok(())
}
```

```sh check
cargo test --quiet --test scalar_reductions scalar_reductions
```

## 25. accumulator の初期化が正しい外側 region にある。
<!-- todo:07-025 -->

`tests/scalar_reductions.rs` の `scalar_reductions` で、この経路の値または拒否条件を確認する。

`tests/scalar_reductions.rs` の `fn scalar_reductions(`の定義を次の全体で置き換えます。

```rust file="tests/scalar_reductions.rs" mode="replace" item="fn scalar_reductions("
fn scalar_reductions() -> Result<()> {
	assert_eq!(
		reductions::row_sum(1, 4)?
			.compile()?
			.run(&[vec![1., 2., 3., 4.]])?,
		vec![vec![10.]]
	);
	let k = reductions::row_sum(2, 3)?;
	assert_eq!(
		k.compile()?.run(&[vec![1., 2., 3., 4., 5., 6.]])?,
		vec![vec![6., 15.]]
	);
	assert_eq!(
		reductions::row_sum(2, 0)?.compile()?.run(&[vec![]])?,
		vec![vec![0., 0.]]
	);
	let k = reductions::matmul(2, 2, 3)?;
	let inputs = [vec![1., 2., 3., 4., 5., 6.], vec![1., 2., 3., 4., 5., 6.]];
	assert_eq!(k.compile()?.run(&inputs)?, vec![vec![22., 28., 49., 64.]]);
	assert_eq!(k.interpret(&inputs)?, vec![vec![22., 28., 49., 64.]]);
	let mut reference = vec![0.0; 4];
	for i in 0..2 {
		for j in 0..2 {
			for inner in 0..3 {
				reference[i * 2 + j] += inputs[0][i * 3 + inner] * inputs[1][inner * 2 + j];
			}
		}
	}
	assert_eq!(k.compile()?.run(&inputs)?[0], reference);
	assert_eq!(
		reductions::conv1d(4, 2)?
			.compile()?
			.run(&[vec![1., 2., 3., 4.], vec![2., 1.]])?,
		vec![vec![4., 7., 10.]]
	);
	Ok(())
}
```

```sh check
cargo test --quiet --test scalar_reductions scalar_reductions
```

## 26. 入力を変えた二回目の run でも前回の accumulator が残らない。
<!-- todo:07-026 -->

`tests/scalar_reductions.rs` の `scalar_reductions` で、この経路の値または拒否条件を確認する。

`tests/scalar_reductions.rs` の `fn scalar_reductions(`の定義を次の全体で置き換えます。

```rust file="tests/scalar_reductions.rs" mode="replace" item="fn scalar_reductions("
fn scalar_reductions() -> Result<()> {
	assert_eq!(
		reductions::row_sum(1, 4)?
			.compile()?
			.run(&[vec![1., 2., 3., 4.]])?,
		vec![vec![10.]]
	);
	let k = reductions::row_sum(2, 3)?;
	assert_eq!(
		k.compile()?.run(&[vec![1., 2., 3., 4., 5., 6.]])?,
		vec![vec![6., 15.]]
	);
	assert_eq!(
		reductions::row_sum(2, 0)?.compile()?.run(&[vec![]])?,
		vec![vec![0., 0.]]
	);
	let k = reductions::matmul(2, 2, 3)?;
	let inputs = [vec![1., 2., 3., 4., 5., 6.], vec![1., 2., 3., 4., 5., 6.]];
	assert_eq!(k.compile()?.run(&inputs)?, vec![vec![22., 28., 49., 64.]]);
	assert_eq!(k.interpret(&inputs)?, vec![vec![22., 28., 49., 64.]]);
	let mut reference = vec![0.0; 4];
	for i in 0..2 {
		for j in 0..2 {
			for inner in 0..3 {
				reference[i * 2 + j] += inputs[0][i * 3 + inner] * inputs[1][inner * 2 + j];
			}
		}
	}
	assert_eq!(k.compile()?.run(&inputs)?[0], reference);
	assert_eq!(
		reductions::conv1d(4, 2)?
			.compile()?
			.run(&[vec![1., 2., 3., 4.], vec![2., 1.]])?,
		vec![vec![4., 7., 10.]]
	);
	Ok(())
}
```

```sh check
cargo test --quiet --test scalar_reductions scalar_reductions
```

## 27. 一般 f32 は前章までの予定許容誤差で interpreter と CPU を比較する。
<!-- todo:07-027 -->

`tests/scalar_reductions.rs` の `scalar_reductions` で、この経路の値または拒否条件を確認する。

`tests/scalar_reductions.rs` の `fn scalar_reductions(`の定義を次の全体で置き換えます。

```rust file="tests/scalar_reductions.rs" mode="replace" item="fn scalar_reductions("
fn scalar_reductions() -> Result<()> {
	assert_eq!(
		reductions::row_sum(1, 4)?
			.compile()?
			.run(&[vec![1., 2., 3., 4.]])?,
		vec![vec![10.]]
	);
	let k = reductions::row_sum(2, 3)?;
	assert_eq!(
		k.compile()?.run(&[vec![1., 2., 3., 4., 5., 6.]])?,
		vec![vec![6., 15.]]
	);
	assert_eq!(
		reductions::row_sum(2, 0)?.compile()?.run(&[vec![]])?,
		vec![vec![0., 0.]]
	);
	let k = reductions::matmul(2, 2, 3)?;
	let inputs = [vec![1., 2., 3., 4., 5., 6.], vec![1., 2., 3., 4., 5., 6.]];
	assert_eq!(k.compile()?.run(&inputs)?, vec![vec![22., 28., 49., 64.]]);
	assert_eq!(k.interpret(&inputs)?, vec![vec![22., 28., 49., 64.]]);
	let mut reference = vec![0.0; 4];
	for i in 0..2 {
		for j in 0..2 {
			for inner in 0..3 {
				reference[i * 2 + j] += inputs[0][i * 3 + inner] * inputs[1][inner * 2 + j];
			}
		}
	}
	assert_eq!(k.compile()?.run(&inputs)?[0], reference);
	assert_eq!(
		reductions::conv1d(4, 2)?
			.compile()?
			.run(&[vec![1., 2., 3., 4.], vec![2., 1.]])?,
		vec![vec![4., 7., 10.]]
	);
	Ok(())
}
```

```sh check
cargo test --quiet --test scalar_reductions scalar_reductions
```

## 章末の累積確認

前章までの example と test も同じ checkout で再ビルドする。

```sh check
cargo test --quiet --all-targets
```
