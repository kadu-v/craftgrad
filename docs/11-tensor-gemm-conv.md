# 11. matmul と convolution を Tensor の合成で作る

前の節で動かしたコードに、今回使う演算と処理を追加します。各節の確認を実行してから次へ進みます。

参照するUOpの構成は [tinygrad c22edff](https://github.com/tinygrad/tinygrad/blob/c22edff14c79310a9b2df62a78819cf77c8289fa/tinygrad/uop/ops.py) です。最初から全機能を実装せず、使う節で増やします。

## 11.01 行列積を展開と積和で作る

matmul はバッチ次元をそろえ、reshape・expand・permute と乗算、最後の軸の和に分解する。専用の実行器を増やさず既存の lowering を使う。

`src/tensor.rs` の `use crate::`を置き換える。

```rust
// file="src/tensor.rs" mode="replace" item="use crate::"
use crate::{
    Result,
    runtime::{Executable, compile_many},
    uop::{Arg, DType, Ops, UOp, broadcast_shape, checked_numel},
};
```

`src/tensor.rs` の `impl Tensor` 内を追記する。

```rust
// file="src/tensor.rs" mode="append" within="impl Tensor"
    pub fn matmul(&self, other: &Self) -> Self {
        let build = || -> Result<Self> {
            let a = self.shape()?;
            let b = other.shape()?;
            if a.len() < 2 || b.len() < 2 || a[a.len() - 1] != b[b.len() - 2] {
                return Err("matmul requires [...,m,k] and [...,k,n]".into());
            }
            let batch = broadcast_shape(&a[..a.len() - 2], &b[..b.len() - 2])?;
            let rank = batch.len();
            let (m, k, n) = (a[a.len() - 2], a[a.len() - 1], b[b.len() - 1]);
            let mut ar = vec![1; rank + 2 - a.len()];
            ar.extend_from_slice(&a);
            let mut ae = batch.clone();
            ae.extend([m, k]);
            let mut br = vec![1; rank + 2 - b.len()];
            br.extend_from_slice(&b);
            let mut be = batch.clone();
            be.extend([k, n]);
            let av = self.reshape(&ar).expand(&ae);
            let mut ap = batch.clone();
            ap.extend([m, 1, k]);
            let mut perm: Vec<usize> = (0..rank).collect();
            perm.extend([rank + 1, rank]);
            let bv = other.reshape(&br).expand(&be).permute(&perm);
            let mut bp = batch;
            bp.extend([1, n, k]);
            Ok(av
                .reshape(&ap)
                .mul(&bv.reshape(&bp))
                .sum_axes(&[rank + 2], false))
        };
        build().unwrap_or_else(Self::invalid)
    }
```

`tests/tensor_native.rs`を追記する。

```rust
// file="tests/tensor_native.rs" mode="append"

fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() < 1e-4, "{a} != {b}");
    }
}

#[test]
fn movement_broadcast_reduction_matmul() {
    let x = Tensor::from_slice(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    close(
        &x.permute(&[1, 0]).to_vec().unwrap(),
        &[1., 4., 2., 5., 3., 6.],
    );
    close(&x.sum_axes(&[1], false).to_vec().unwrap(), &[6., 15.]);
    close(
        &x.matmul(&x.permute(&[1, 0])).to_vec().unwrap(),
        &[14., 32., 32., 77.],
    );
    let y = x
        .pad(&[(1, 0), (1, 1)])
        .flip(&[0])
        .shrink(&[(0, 2), (1, 4)]);
    close(&y.to_vec().unwrap(), &[4., 5., 6., 1., 2., 3.]);
}
```

```sh
# check step=11.01
cargo test --offline
```

[[1,2,3],[4,5,6]] とその転置の積が [[14,32],[32,77]] になる。

## 11.02 畳込みをパッチと行列積で作る

NCHW 入力を padding と shrink でパッチへ分け、stack で並べる。パッチとフィルタを行列にして matmul し、結果を NCHW へ戻す。

`src/tensor.rs` の `impl Tensor` 内を追記する。

```rust
// file="src/tensor.rs" mode="append" within="impl Tensor"
    pub fn conv2d(&self, weight: &Self, stride: (usize, usize), padding: (usize, usize)) -> Self {
        let build = || -> Result<Self> {
            let x = self.shape()?;
            let w = weight.shape()?;
            if x.len() != 4 || w.len() != 4 || x[1] != w[1] || stride.0 == 0 || stride.1 == 0 {
                return Err("conv2d requires NCHW/OIHW and nonzero strides".into());
            }
            let padded = self.pad(&[
                (0, 0),
                (0, 0),
                (padding.0, padding.0),
                (padding.1, padding.1),
            ]);
            let p = padded.shape()?;
            if p[2] < w[2] || p[3] < w[3] {
                return Err("convolution kernel exceeds input".into());
            }
            let h = (p[2] - w[2]) / stride.0 + 1;
            let width = (p[3] - w[3]) / stride.1 + 1;
            let mut patches = vec![];
            for ky in 0..w[2] {
                for kx in 0..w[3] {
                    let mut rows = vec![];
                    for iy in 0..h {
                        let mut cols = vec![];
                        for ix in 0..width {
                            cols.push(
                                padded
                                    .shrink(&[
                                        (0, x[0]),
                                        (0, x[1]),
                                        (iy * stride.0 + ky, iy * stride.0 + ky + 1),
                                        (ix * stride.1 + kx, ix * stride.1 + kx + 1),
                                    ])
                                    .reshape(&[x[0], x[1]]),
                            );
                        }
                        rows.push(Self::stack(&cols, 2));
                    }
                    patches.push(Self::stack(&rows, 2));
                }
            }
            let columns = Self::stack(&patches, 4)
                .reshape(&[x[0], x[1], h, width, w[2], w[3]])
                .permute(&[0, 2, 3, 1, 4, 5])
                .reshape(&[x[0] * h * width, x[1] * w[2] * w[3]]);
            Ok(columns
                .matmul(&weight.reshape(&[w[0], w[1] * w[2] * w[3]]).permute(&[1, 0]))
                .reshape(&[x[0], h, width, w[0]])
                .permute(&[0, 3, 1, 2]))
        };
        build().unwrap_or_else(Self::invalid)
    }
```

`tests/convolution.rs`を保存する。

```rust
// file="tests/convolution.rs" mode="write"
use chibigrad::Tensor;
#[test]
fn composite_conv() {
    let x = Tensor::from_slice(&[1., 2., 3., 4., 5., 6., 7., 8., 9.], &[1, 1, 3, 3]);
    let w = Tensor::from_slice(&[1., 0., 0., -1.], &[1, 1, 2, 2]);
    assert_eq!(x.conv2d(&w, (1, 1), (0, 0)).to_vec().unwrap(), [-4.; 4]);
}
```

```sh
# check step=11.02
cargo test --offline
```

3×3 入力と [[1,0],[0,-1]] のフィルタが -4 の 2×2 出力を返す。

## 章末の累積確認

前の章までのテストとexampleも再ビルドします。

```sh
# check
cargo test --quiet --all-targets
```

## 要件との対応

### 11-001 PARAM とホスト配列を結び付ける Tensor のコンストラクタを追加する。

<!-- todo:11-001 -->

関連する実装と確認は本章の `11.01`, `11.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 11-002 `add`、`mul`、`sum`、movement のメソッドを graph builder に委譲する。

<!-- todo:11-002 -->

関連する実装と確認は本章の `11.01`, `11.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 11-003 `Tensor` 演算内で暗黙にコンパイルや実行を始めない。

<!-- todo:11-003 -->

関連する実装と確認は本章の `11.01`, `11.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 11-004 graph を出力してから compile し、既存の CPU runtime で実行する。

<!-- todo:11-004 -->

関連する実装と確認は本章の `11.01`, `11.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 11-005 縮約次元 K の一致を検証する。

<!-- todo:11-005 -->

関連する実装と確認は本章の `11.01`, `11.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 11-006 上記の reshape、permute、MUL、SUM の式を Tensor メソッドで作る。

<!-- todo:11-006 -->

関連する実装と確認は本章の `11.01`, `11.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 11-007 出力の shape `[M,N]` と累積の初期化位置を確認する。

<!-- todo:11-007 -->

関連する実装と確認は本章の `11.01`, `11.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 11-008 手書きの三重ループと `M=2,N=2,K=3` の結果を比較する。

<!-- todo:11-008 -->

関連する実装と確認は本章の `11.01`, `11.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 11-009 同じ A を転置ありと転置なしで読む式を別々の添字で下げる。

<!-- todo:11-009 -->

関連する実装と確認は本章の `11.01`, `11.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 11-010 `KH,KW` 個の SHRINK と STACK で窓を組み立てる。

<!-- todo:11-010 -->

関連する実装と確認は本章の `11.01`, `11.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 11-011 座標順を明示し、window の shape を dump に出す。

<!-- todo:11-011 -->

関連する実装と確認は本章の `11.01`, `11.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 11-012 窓と重みの積を `C,KH,KW` について縮約する。

<!-- todo:11-012 -->

関連する実装と確認は本章の `11.01`, `11.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 11-013 出力の保存順を NCHW にそろえる。

<!-- todo:11-013 -->

関連する実装と確認は本章の `11.01`, `11.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 11-014 手書き conv の参照ループをテスト側に置いて比較する。

<!-- todo:11-014 -->

関連する実装と確認は本章の `11.01`, `11.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 11-015 conv の前に PAD を挿入し、出力 shape の計算に反映する。

<!-- todo:11-015 -->

関連する実装と確認は本章の `11.01`, `11.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 11-016 channel ごとの bias `[O]` を reshape して加算する。

<!-- todo:11-016 -->

関連する実装と確認は本章の `11.01`, `11.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 11-017 matmul の後の bias と同じ broadcast 規則を使う。

<!-- todo:11-017 -->

関連する実装と確認は本章の `11.01`, `11.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 11-018 stride、dilation、groups の未対応値を明示的に拒否する。

<!-- todo:11-018 -->

関連する実装と確認は本章の `11.01`, `11.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 11-019 pad の外側を読む可能性がある LOAD を分岐内へ保持する。

<!-- todo:11-019 -->

関連する実装と確認は本章の `11.01`, `11.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 11-020 予定例の期待値が一致し、手書きループとも一致する。

<!-- todo:11-020 -->

関連する実装と確認は本章の `11.01`, `11.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 11-021 graph に matmul と conv 専用の実行器への抜け道がない。

<!-- todo:11-021 -->

関連する実装と確認は本章の `11.01`, `11.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 11-022 reduction の初期化が出力要素ごとに一回だけ実行される。

<!-- todo:11-022 -->

関連する実装と確認は本章の `11.01`, `11.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 11-023 学習用重みを CONST として intern せず、入力値を差し替えられる。

<!-- todo:11-023 -->

関連する実装と確認は本章の `11.01`, `11.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 11-024 非正方行列、複数 channel、複数 output channel の小例を固定値で比較する。

<!-- todo:11-024 -->

関連する実装と確認は本章の `11.01`, `11.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 11-025 graph、schedule、生成 C の順序から同じ計算であると追える。

<!-- todo:11-025 -->

関連する実装と確認は本章の `11.01`, `11.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。
