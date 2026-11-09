# 12. 数学関数を実行し、近似と値域の責任を分ける

通常経路は sqrtf、sinf、cosf、log2f、exp2f と libm に接続する。exp と自然対数の合成も示す。近似は opt-in の別 builder で、既存ノードの意味を変更しない。exp2 は x=q+r、|r|<=0.5 の7次 Taylor、log2 は m∈[1,2) の atanh 級数11次、sin は |r|<=π/2 の13次 Taylor を Horner 形式で表す。係数は階乗または奇数分母から生成する。exp2/log2 は exponent -16..16 の範囲だけで近似し、それ以外と NaN/Inf は libm に戻す。sin は |x|<=32 のみ近似する。範囲外の branch の入力を整数へ CAST しない。

`file=... mode=write` はそのファイルの全置換、`mode=append` は末尾への追記である。すべて前章でできたプロジェクトの root で行う。`sh check` は実際の検証コマンドであり、ソースファイルには保存しない。

## 1. SQRT の dtype と入力数を verifier に追加する。
<!-- todo:12-001 -->

```rust file=src/math.rs mode=write
use crate::{Binary as B, Graph, NodeId, Result, Unary as U};
fn scalar(g: &mut Graph, x: f32) -> Result<NodeId> {
	g.constant(x, vec![])
}
fn add(g: &mut Graph, a: NodeId, b: NodeId) -> Result<NodeId> {
	g.binary(B::Add, a, b)
}
fn mul(g: &mut Graph, a: NodeId, b: NodeId) -> Result<NodeId> {
	g.binary(B::Mul, a, b)
}
fn sub(g: &mut Graph, a: NodeId, b: NodeId) -> Result<NodeId> {
	g.binary(B::Sub, a, b)
}
fn bounded(g: &mut Graph, x: NodeId, lo: f32, hi: f32) -> Result<NodeId> {
	let lo = scalar(g, lo)?;
	let hi = scalar(g, hi)?;
	let a = g.binary(B::Lt, lo, x)?;
	let b = g.binary(B::Lt, x, hi)?;
	mul(g, a, b)
}
fn horner(g: &mut Graph, x: NodeId, coefficients: &[f32]) -> Result<NodeId> {
	let mut p = scalar(g, *coefficients.last().ok_or("empty polynomial")?)?;
	for &c in coefficients[..coefficients.len() - 1].iter().rev() {
		p = mul(g, p, x)?;
		let c = scalar(g, c)?;
		p = add(g, p, c)?;
	}
	Ok(p)
}
pub fn exp(g: &mut Graph, x: NodeId) -> Result<NodeId> {
	let scale = scalar(g, std::f32::consts::LOG2_E)?;
	let y = mul(g, x, scale)?;
	g.unary(U::Exp2, y)
}
pub fn log(g: &mut Graph, x: NodeId) -> Result<NodeId> {
	let y = g.unary(U::Log2, x)?;
	let scale = scalar(g, std::f32::consts::LN_2)?;
	mul(g, y, scale)
}
pub fn approximate_exp2(g: &mut Graph, x: NodeId) -> Result<NodeId> {
	let one = scalar(g, 1.)?;
	let two = scalar(g, 2.)?;
	let mut q = scalar(g, -16.)?;
	let mut scale = scalar(g, 2.0f32.powi(-16))?;
	for n in -15..=16 {
		let threshold = scalar(g, n as f32 - 0.5)?;
		let below = g.binary(B::Lt, x, threshold)?;
		let above = sub(g, one, below)?;
		q = add(g, q, above)?;
		let factor = g.where_(above, two, one)?;
		scale = mul(g, scale, factor)?;
	}
	let r = sub(g, x, q)?;
	let l = std::f32::consts::LN_2;
	let coefficients = (0..=7)
		.map(|n| l.powi(n) / (1..=n).product::<i32>() as f32)
		.collect::<Vec<_>>();
	let p = horner(g, r, &coefficients)?;
	let approx = mul(g, p, scale)?;
	let valid = bounded(g, x, -16., 16.)?;
	let fallback = g.unary(U::Exp2, x)?;
	g.where_(valid, approx, fallback)
}
pub fn approximate_log2(g: &mut Graph, x: NodeId) -> Result<NodeId> {
	let one = scalar(g, 1.)?;
	let mut e = scalar(g, -16.)?;
	let mut scale = scalar(g, 2.0f32.powi(-16))?;
	for n in -15..=16 {
		let threshold = scalar(g, 2.0f32.powi(n))?;
		let below = g.binary(B::Lt, x, threshold)?;
		let above = sub(g, one, below)?;
		e = add(g, e, above)?;
		scale = g.where_(above, threshold, scale)?;
	}
	let m = g.binary(B::Div, x, scale)?;
	let numerator = sub(g, m, one)?;
	let denominator = add(g, m, one)?;
	let t = g.binary(B::Div, numerator, denominator)?;
	let t2 = mul(g, t, t)?;
	let coefficients = (0..6)
		.map(|n| 2.0 * std::f32::consts::LOG2_E / (2 * n + 1) as f32)
		.collect::<Vec<_>>();
	let p = horner(g, t2, &coefficients)?;
	let p = mul(g, t, p)?;
	let approx = add(g, e, p)?;
	let valid = bounded(g, x, 2.0f32.powi(-16), 2.0f32.powi(16))?;
	let fallback = g.unary(U::Log2, x)?;
	g.where_(valid, approx, fallback)
}
pub fn approximate_sin(g: &mut Graph, x: NodeId) -> Result<NodeId> {
	let one = scalar(g, 1.)?;
	let mut q = scalar(g, -5.)?;
	for n in -4..=5 {
		let threshold = scalar(g, (n as f32 - 0.5) * std::f32::consts::TAU)?;
		let below = g.binary(B::Lt, x, threshold)?;
		let above = sub(g, one, below)?;
		q = add(g, q, above)?;
	}
	let tau = scalar(g, std::f32::consts::TAU)?;
	let multiple = mul(g, q, tau)?;
	let mut r = sub(g, x, multiple)?;
	let pi = scalar(g, std::f32::consts::PI)?;
	let half = scalar(g, std::f32::consts::FRAC_PI_2)?;
	let above = g.binary(B::Lt, half, r)?;
	let reflected = sub(g, pi, r)?;
	r = g.where_(above, reflected, r)?;
	let negative_half = scalar(g, -std::f32::consts::FRAC_PI_2)?;
	let negative_pi = scalar(g, -std::f32::consts::PI)?;
	let below = g.binary(B::Lt, r, negative_half)?;
	let reflected = sub(g, negative_pi, r)?;
	r = g.where_(below, reflected, r)?;
	let squared = mul(g, r, r)?;
	let coefficients = [
		1.,
		-1. / 6.,
		1. / 120.,
		-1. / 5040.,
		1. / 362880.,
		-1. / 39916800.,
		1. / 6227020800.,
	];
	let p = horner(g, squared, &coefficients)?;
	let approx = mul(g, r, p)?;
	let valid = bounded(g, x, -32.000004, 32.000004)?;
	let fallback = g.unary(U::Sin, x)?;
	g.where_(valid, approx, fallback)
}
```

```rust file=src/lib.rs mode=append
pub mod math;
```

```rust file=tests/math_functions.rs mode=write
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
use chinigrad::math;

#[test]
fn empty_and_math() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![2, 0])?;
	let y = g.reduce(x, Reduce::Sum, vec![1], false)?;
	assert_eq!(
		g.compile(&[y])?.run(&[buffer(&[2, 0], &[])])?[0].data,
		[0., 0.]
	);
	let mut g = Graph::default();
	let x = g.input(0, vec![4])?;
	let y = g.unary(Unary::Sqrt, x)?;
	let out = g
		.compile(&[y])?
		.run(&[buffer(&[4], &[-0., 4., -1., f32::INFINITY])])?;
	assert_eq!(out[0].data[0].to_bits(), (-0.0f32).to_bits());
	assert_eq!(out[0].data[1], 2.);
	assert!(out[0].data[2].is_nan());
	assert!(out[0].data[3].is_infinite());
	Ok(())
}
#[test]
fn graph_polynomials() -> Result<()> {
	for (build, reference, values, tolerance) in [
		(
			math::approximate_exp2 as fn(&mut Graph, usize) -> Result<usize>,
			f32::exp2 as fn(f32) -> f32,
			vec![-16., -5., -0.5, 0., 0.5, 5., 15.9, 20., f32::NEG_INFINITY],
			1e-5,
		),
		(
			math::approximate_log2,
			f32::log2,
			vec![0., 0.1, 0.5, 1., 1.5, 2., 100., f32::INFINITY],
			1e-5,
		),
		(
			math::approximate_sin,
			f32::sin,
			vec![
				-32.,
				-std::f32::consts::PI,
				-0.,
				0.,
				1.,
				std::f32::consts::PI,
				32.,
				100000.,
				f32::INFINITY,
			],
			5e-6,
		),
	] {
		let mut g = Graph::default();
		let x = g.input(0, vec![values.len()])?;
		let y = build(&mut g, x)?;
		let plan = g.compile(&[y])?;
		let output = plan.run(&[Buffer::new(vec![values.len()], values.clone())?])?;
		for (&x, &actual) in values.iter().zip(&output[0].data) {
			let expected = reference(x);
			assert!(
				actual.is_nan() && expected.is_nan()
					|| actual == expected
					|| (actual - expected).abs() <= tolerance * (1. + expected.abs()),
				"x{x} actual{actual} expected{expected}"
			);
		}
	}
	Ok(())
}
#[test]
fn fixed_seed_polynomial_boundaries() -> Result<()> {
	let mut seed = 0x51c0ffeeu32;
	let mut samples = vec![
		-32.,
		-std::f32::consts::PI,
		-0.,
		0.,
		std::f32::consts::PI,
		32.,
	];
	for bound in [-32.0f32, -std::f32::consts::PI, std::f32::consts::PI, 32.] {
		samples.push(f32::from_bits(bound.to_bits() - 1));
		samples.push(f32::from_bits(bound.to_bits() + 1));
	}
	for _ in 0..128 {
		seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
		samples.push((seed as f64 / u32::MAX as f64 * 64. - 32.) as f32);
	}
	let mut g = Graph::default();
	let x = g.input(0, vec![samples.len()])?;
	let y = math::approximate_sin(&mut g, x)?;
	let plan = g.compile(&[y])?;
	let actual = plan.run(&[Buffer::new(vec![samples.len()], samples.clone())?])?;
	for (&x, &actual) in samples.iter().zip(&actual[0].data) {
		assert!((actual - x.sin()).abs() <= 5e-6, "sin({x}) actual={actual}");
	}
	assert!(plan.source.contains("sinf"));
	Ok(())
}
```

`tests/math_functions.rs` の `empty_and_math` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test math_functions empty_and_math
```

## 2. interpreter と C renderer の sqrt をそろえる。
<!-- todo:12-002 -->

`tests/math_functions.rs` の `empty_and_math` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test math_functions empty_and_math
```

## 3. コンパイルされた共有ライブラリから sqrtf を解決できるようにする。
<!-- todo:12-003 -->

`tests/math_functions.rs` の `empty_and_math` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test math_functions empty_and_math
```

## 4. 通常値、-0、負値、Inf、NaN を分類して比較する。
<!-- todo:12-004 -->

`tests/math_functions.rs` の `empty_and_math` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test math_functions empty_and_math
```

## 5. SIN、LOG2、EXP2 を elementwise として rangeify に通す。
<!-- todo:12-005 -->

`tests/math_functions.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test math_functions
```

## 6. C 側を sinf、log2f、exp2f に下げる。
<!-- todo:12-006 -->

`tests/math_functions.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test math_functions
```

## 7. `exp` と自然対数を定数の積と既存 UOp の合成で追加する。
<!-- todo:12-007 -->

`tests/math_functions.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test math_functions
```

## 8. 特殊値の期待する分類と定義域を API の説明に書く。
<!-- todo:12-008 -->

`tests/math_functions.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test math_functions
```

## 9. `exp2(log2(x))` を正の小さい配列で実行する。
<!-- todo:12-009 -->

`tests/math_functions.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test math_functions
```

## 10. exp2 は `x=q+r`、`r∈[-0.5,0.5]` に分けて Horner 形式で評価する。
<!-- todo:12-010 -->

`tests/math_functions.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test math_functions
```

## 11. `2^q` の再構成を定義し、整数変換の範囲を検査する。
<!-- todo:12-011 -->

`tests/math_functions.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test math_functions
```

## 12. log2 は `x=2^e*m` に分け、m を限られた区間に正規化する。
<!-- todo:12-012 -->

`tests/math_functions.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test math_functions
```

## 13. 係数、近似区間、次数、出典または係数生成方法を実装の近くに残す。
<!-- todo:12-013 -->

`tests/math_functions.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test math_functions
```

## 14. 各段階を UOp にして既存 rewrite と renderer へ通す。
<!-- todo:12-014 -->

`tests/math_functions.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test math_functions
```

## 15. 周期と象限を求め、短い区間へ還元して符号を復元する。
<!-- todo:12-015 -->

`tests/math_functions.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test math_functions
```

## 16. 区間内だけの奇多項式を Horner 形式で評価する。
<!-- todo:12-016 -->

`tests/math_functions.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test math_functions
```

## 17. `|x|<=32` の境界、π の倍数、その両隣をテストする。
<!-- todo:12-017 -->

`tests/math_functions.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test math_functions
```

## 18. 大きな有限値と Inf、NaN を libm 経路に分岐する。
<!-- todo:12-018 -->

`tests/math_functions.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test math_functions
```

## 19. 高水準 SIN と近似展開後のグラフを両方 dump できるようにする。
<!-- todo:12-019 -->

`tests/math_functions.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test math_functions
```

## 20. 通常値の許容誤差と特殊値の分類を分けて判定する。
<!-- todo:12-020 -->

`tests/math_functions.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test math_functions
```

## 21. sqrt(-0) の sign bit、log2(0) の -Inf、exp2(-Inf) の 0 を検査する。
<!-- todo:12-021 -->

`tests/math_functions.rs` の `empty_and_math` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test math_functions empty_and_math
```

## 22. 固定 seed の有限値集合と、区間境界の隣接 f32 を比較する。
<!-- todo:12-022 -->

`tests/math_functions.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test math_functions
```

## 23. 近似対象外の値を libm に戻す分岐が生成コードに残る。
<!-- todo:12-023 -->

`tests/math_functions.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test math_functions
```

## 24. 近似を選ばない既存 example の数値が変わらない。
<!-- todo:12-024 -->

`tests/math_functions.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test math_functions
```

## 章末の累積確認

前章までの example と test も同じ checkout で再ビルドする。

```sh check
cargo test --quiet --all-targets
```
