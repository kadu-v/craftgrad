# 11. Tensor の合成演算で matmul と conv を動かす

## このステップで動かすもの

[10 の movement](10-movement.md) を、薄い `Tensor` API から使う。
元計画の第 7 章の Tensor と matmul、第 11 章のモデルに必要な演算に対応する。
手書きループの GEMM と conv を、shape 付きの純粋グラフから自動生成できる状態にする。

```text
A = [[1,2,3],[4,5,6]]
B = [[1,2],[3,4],[5,6]]
A.matmul(B) = [[22,28],[49,64]]
X = [[[[1,2,3],[4,5,6],[7,8,9]]]]
W = [[[[1,1],[1,1]]]]
X.conv2d(W) = [[[[12,16],[24,28]]]]
```

これらは実装予定の API と期待値であり、現時点で実行済みの結果ではない。
CPU の既存 pipeline を使い、GEMM 専用 runtime を作らない。

## 最小実装とデータ

`Tensor` はグラフの所有先への参照、`NodeId`、shape、dtype を扱う薄い入口にする。
利用者の演算はグラフを返し、実行は `compile` と `Plan::run` の明示的な境界で行う。
実体化した出力バッファを元の演算ノードに上書きしない。
入力値と重みは PARAM スロットへ渡し、同じ Plan で値を差し替える。

```text
matmul 入力: A[M,K], B[K,N]
A → reshape[M,1,K]
B → permute[1,0] → reshape[1,N,K]
Mul → shape[M,N,K]
ReduceSum(axis=2) → shape[M,N]
```

matmul は既存の RESHAPE、PERMUTE、broadcast、MUL、REDUCE の合成にする。
最初は rank 2、静的 f32、加算累積だけを受け付ける。
kernel IR は出力 `(m,n)` と縮約 `k` の RANGE、LOAD、累積、STORE になる。
独立した MATMUL UOp を追加しなくても、この段階の動作を実現できる。

conv は NCHW の入力と OIHW の重みを使う cross-correlation とする。
まず stride=1、dilation=1、groups=1、padding=0 に限る。
出力は `[N,O,H-KH+1,W-KW+1]` になる。
窓 tensor は各 `(kh,kw)` に対応する SHRINK を STACK して構成する。
PERMUTE と RESHAPE で `[N,1,C,OH,OW,KH,KW]` にそろえる。
重みを `[1,O,C,1,1,KH,KW]` として掛け、`C,KH,KW` を足す。
窓全体の中間バッファが必要なら、最初は明示的に materialize してよい。
窓の要素を直接 LOAD する最適化は、数値が一致した後に加える。

## この章のファイル配置

src/tensor.rs に Graph を参照する薄い Tensor を追加し、matmul と conv を既存 movement/REDUCE の合成として育てる。数値参照ループは examples/book_11_*.rs に分け、統合テストから同じ期待値を確認する。

次はこの章の終わりまでに触る関連ファイルの配置案である。現在のcheckoutの一覧ではない。各コミットでは、その下に示す部分だけを追加・変更する。

```text
chibigrad/
├── examples/
│   ├── book_11_matmul_reference.rs  # 本文のコードを配置する。
│   ├── tensor_chain.rs  # Tensor から既存 CPU Plan を動かす予定例。
│   ├── tensor_conv2d.rs  # NCHW/OIHW の conv 予定実行例。
│   ├── tensor_conv_bias.rs  # padding/bias を含む予定実行例。
│   └── tensor_matmul.rs  # 合成グラフを CPU で実行する予定例。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # 安全な pad と縮約後 bias の生成 C。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rangeify.rs  # PAD の有効条件を LOAD まで維持。
│   └── tensor.rs  # padding と bias の合成、未対応引数の拒否。
├── tests/
│   └── tensor_ops.rs  # 非対称 pad、bias 再bind と不正値。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

生成Cやshaderは一時ディレクトリへ出力するため、このRustプロジェクトのツリーには含めない。コード冒頭のコメントで、生成元と生成先を確認する。

## 小さなコミットで進める

### 1. Tensor から既存の elementwise を実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── tensor_chain.rs  # [追加] Tensor から既存 CPU Plan を動かす予定例。
├── src/
│   ├── lib.rs  # [更新] tensor モジュールと利用者向け公開入口。
│   └── tensor.rs  # [追加] Graph の所有先を確認して UOp を作る Tensor。
├── tests/
│   └── tensor_ops.rs  # [追加] 遅延構築、Plan 再利用、別 Graph の拒否。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/lib.rs で pub mod tensor を宣言し、src/tensor.rs に掲載 Tensor と add/mul を置く。NodeId、Graph、Op は src/uop.rs の既存定義を参照し、broadcast_shape は src/shape.rs から使う。input、sum、movement の入口もこの impl に追加し、compile と run は既存 Plan へ委ねる。

新しいモジュールは、既存の宣言を残して次を追記する。内部の補助関数は各モジュールに置き、exampleから使う入口だけを公開する。

```rust
// src/lib.rs（モジュール宣言を追加）
pub mod tensor;
```


このコミットで足す Tensor は計算結果の配列ではなく、純粋な計算ノードへの利用者向けの参照である。候補は `src/tensor.rs` で、既存 Graph の所有方法を使い、設計案の `Tensor::input(graph,slot,shape)`、`add`、`mul`、`sum` がそれぞれ PARAM と演算ノードを作る形にする。ホスト配列の所有先は実行時の binding 側へ置く。shape と dtype を Tensor に保持する場合も Graph から導出した値と食い違う別の真実を作らず、コンストラクタの境界で検証する。別 Graph の Tensor を混ぜた場合は NodeId の番号だけを信用せず、所有先の違いとして拒否する。

入力 slot0 の x、scalar 定数1、scalar 定数2から `(x+1)*2` を作る順は、PARAM、CONST、ADD、MUL の追加である。この時点では buffer の読出しも clang の起動も行わない。出力 NodeId を明示的な compile へ渡したときに、既存 rangeify が出力長3の RANGE と broadcast を作り、schedule と C renderer を通して Plan を得る。Plan は入出力 slot の仕様とコンパイル済みの関数を持ち、今回の x のアドレスを演算ノードへ固定しない。

slot0 に `[1,2,3]` を bind すると、座標1では `2+1=3`、次に `3*2=6` となり、全体は `[4,6,8]` になる。同じ Plan の次回入力を `[4,5,6]` に替えると、同じ座標1は `(5+1)*2=12`、全体は `[10,12,14]` になる。配列を再確保した場合でも slot の dtype と shape が同じなら動き、前のアドレスを使い続けないことを検証する。

`tests/tensor_ops.rs` では式を作った直後の compiler と runtime の呼出しが0回、compile 後に1回、二回の run で compile 回数が増えないことを調べる。dump では純粋な ADD/MUL と融合した一つの出力カーネルの対応を見る。出力 allocation は入力と別に所有し、同じ shape の入力二つが同じ storage identity にならないことも確認する。Tensor の入口を追加しても、buffers/scalars の二配列 ABI と既存の実行計画をそのまま再利用する。


`src/tensor.rs` の入口は新しい計算ノードを足すだけにする。次は既存 `Graph` へ加える抜粋で、`intern(op: Op, src: &[NodeId]) -> NodeId` は03で `Arena::add` を Graph に統合した入口を使う。src は NodeId のスライスとして渡す。`shape(id) -> &[usize]` は検証済みの導出値である。

組込み用の抜粋。

既存 Graph の型を参照する Tensor の新規定義と impl。

```rust
// src/tensor.rs（既存ファイルへ組み込む抜粋）
#[derive(Clone)]
struct Tensor {
	graph_id: u64,
	id: NodeId,
}
impl Tensor {
	fn add(&self, rhs: &Self, graph: &mut Graph) -> Result<Self, String> {
		if self.graph_id != rhs.graph_id || self.graph_id != graph.id {
			return Err("different graphs".into());
		}
		let shape = broadcast_shape(graph.shape(self.id), graph.shape(rhs.id))?;
		let id = graph.intern(Op::Add, &[self.id, rhs.id]);
		debug_assert_eq!(graph.shape(id), shape);
		Ok(Self { graph_id: self.graph_id, id })
	}
	fn mul(&self, rhs: &Self, graph: &mut Graph) -> Result<Self, String> {
		if self.graph_id != rhs.graph_id || self.graph_id != graph.id {
			return Err("different graphs".into());
		}
		broadcast_shape(graph.shape(self.id), graph.shape(rhs.id))?;
		let id = graph.intern(Op::Mul, &[self.id, rhs.id]);
		Ok(Self { graph_id: self.graph_id, id })
	}
}
```

`Tensor` は NodeId と所有先の識別だけを持つ。`broadcast_shape` は10の shape 検証を再利用し、実際の添字変換は compile 後へ残す。メソッド内には host 配列、コンパイラ、runtime の呼出しがない。

同じ入力を追う例。

```text
%0 = PARAM(buffer_slot=0, shape=[3])
%1 = CONST_F32(1)
%2 = ADD(%0,%1)
%3 = CONST_F32(2)
%4 = MUL(%2,%3)
run slot0=[1,2,3] -> [4,6,8]
run slot0=[4,5,6] -> [10,12,14]
```

二回の run で変わるのは slot0 の binding であり、%0 の PARAM や %4 の演算は同じままである。

- [ ] PARAM とホスト配列を結び付ける Tensor のコンストラクタを追加する。
- [ ] `add`、`mul`、`sum`、movement のメソッドを graph builder に委譲する。
- [ ] `Tensor` 演算内で暗黙にコンパイルや実行を始めない。
- [ ] graph を出力してから compile し、既存の CPU runtime で実行する。

予定例 `tensor_chain` は `[1,2,3]` に対して `(x+1)*2` を計算し `[4,6,8]` を返す。
graph dump に ADD と MUL が見え、schedule dump に融合した一つの出力カーネルが見える。
PARAM を `[4,5,6]` に替えると同じ Plan が `[10,12,14]` を返す。

### 2. rank 2 の matmul を合成して実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_11_matmul_reference.rs  # 本文のコードを配置する。
│   └── tensor_matmul.rs  # [追加] 合成グラフを CPU で実行する予定例。
├── src/
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── tensor.rs  # [更新] rank2 matmul の shape 検証と演算合成。
├── tests/
│   └── tensor_ops.rs  # [更新] 非正方 matmul、K=0 と重み再bind。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/tensor.rs の impl Tensor に matmul を追加し、reshape、permute、mul、sum を呼んで NodeId を返す。下の三重ループと main は book_11_matmul_reference の単独例へ保存する。tests/tensor_ops.rs には main を移さず参照関数と Plan 出力の比較を追加する。


行列積は数値ループを Tensor の外で実行せず、既存の movement と REDUCE の組立てとして追加する。候補は `src/tensor.rs` の設計案 `Tensor::matmul(rhs)` である。最初に両方が rank2 の f32 で `A.shape[1]==B.shape[0]` かを確認し、`M=A[0],K=A[1],N=B[1]` を得る。A を `[M,1,K]`、B を転置 `[N,K]` の後 `[1,N,K]` に reshape し、broadcast MUL の出力 `[M,N,K]` を軸2で SUM する。SUM が軸を削除する契約なら結果はそのまま `[M,N]` であり、keepdim を使う実装なら最後の長さ1の軸を明示的に取り除く。

出力 `(m,n)` から左入力は `(m,k)`、右入力は `(k,n)` へ戻る。連続バッファ添字はそれぞれ `m*K+k` と `k*N+n` になる。rangeify は外側 m,n の各組で acc=0 を一回だけ作り、その内側の k ループで積を足し、ループ終了後に `out[m*N+n]` へ一回保存する。acc を m だけの scope へ上げると列間の和が混じるので、初期化の位置を kernel dump で確かめる。

`m=0,n=1` の場合は `1*2=2`、`2*4=8`、`3*6=18` を順に足し、累積は0,2,10,28となる。次の行の n=0 は独立した0から `4*1+5*3+6*5=49` になるので、出力は `[[22,28],[49,64]]` である。B の値を全1に rebind すれば二つの列が同じになり `[[6,6],[15,15]]` となる。学習する重み B を CONST に畳み込んでいないこともこの比較で分かる。

テスト側だけに三重ループの参照実装を置き、非正方の固定行列、K不一致、K=0の単位元を確認する。元グラフの dump は専用 MATMUL 実行器を含まず、kernel dump で上記の二つの LOAD 添字が見えることを確認する。同じ A を別の転置配置にも使う例では、座標と scope を含む memo のまま lowering する。既存 C renderer と Plan がそのままこの合成演算を実行できることが完了条件である。


合成した matmul の結果を照合する `tests/tensor_ops.rs` の参照ループを作る。入力は row-major の A と B、返すものは新規の `[M,N]` である。

単独で実行できる例。

三重ループと main を含む参照例全体。Tensor 本体の実装には使わない。

```rust
// examples/book_11_matmul_reference.rs（単独で実行する例）
fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
	assert_eq!(a.len(), m*k);
	assert_eq!(b.len(), k*n);
	let mut out = vec![0.0; m*n];
	for row in 0..m {
		for col in 0..n {
			let mut acc = 0.0;
			for reduce in 0..k {
				acc += a[row*k+reduce] * b[reduce*n+col];
			}
			out[row*n+col] = acc;
		}
	}
	out
}
fn main() {
	let a = [1.,2.,3.,4.,5.,6.];
	let b = [1.,2.,3.,4.,5.,6.];
	assert_eq!(matmul(&a,&b,2,3,2), [22.,28.,49.,64.]);
	assert_eq!(matmul(&a,&[1.;6],2,3,2), [6.,6.,15.,15.]);
	assert_eq!(matmul(&[],&[],2,0,2), [0.;4]);
}
```

`reduce` は高水準グラフの `[M,N,K]` の最後の軸に対応する。`acc` は row と col の両方が決まった後で0へ戻し、K=0でも単位元0を保存する。この参照関数はテスト専用とし、Tensor の実装をこのループで置き換えない。

同じ入力を追う例。

```text
A [2,3] -> RESHAPE [2,1,3]
B [3,2] -> PERMUTE [2,3] -> RESHAPE [1,2,3]
MUL [2,2,3] -> SUM axis=2 -> [2,2]
(row,col)=(0,1): products=[2,8,18], acc=28
```

この graph の各積の LOAD 添字が参照ループの二つの添字と一致すれば、movement と縮約の組合せを実装できている。

- [ ] 縮約次元 K の一致を検証する。
- [ ] 上記の reshape、permute、MUL、SUM の式を Tensor メソッドで作る。
- [ ] 出力の shape `[M,N]` と累積の初期化位置を確認する。
- [ ] 手書きの三重ループと `M=2,N=2,K=3` の結果を比較する。
- [ ] 同じ A を転置ありと転置なしで読む式を別々の添字で下げる。

予定例 `tensor_matmul` は `[[22,28],[49,64]]` を返す。
重みを全要素 1 の `[3,2]` に替えると `[[6,6],[15,15]]` になる。
K が異なる入力は clang を起動する前にエラーにする。

### 3. 小さな窓を作って conv を実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── tensor_conv2d.rs  # [追加] NCHW/OIHW の conv 予定実行例。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # [更新] 下がった conv のループと LOAD/STORE の出力。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rangeify.rs  # [更新] 窓の座標と縮約軸の添字生成。
│   └── tensor.rs  # [更新] SHRINK/STACK から conv2d を組む入口。
├── tests/
│   └── tensor_ops.rs  # [更新] 固定 conv と複数 channel の参照比較。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/tensor.rs の impl Tensor に conv2d を追加し、窓片を作る SHRINK と STACK、軸変換、積、SUM を組む。src/rangeify.rs では既存の座標合成が掲載 C の xi/wi と一致するか確認する。C 本体は src/renderer/c.rs の生成結果であり、conv 専用のホスト実行関数として追加しない。


conv2d は入力の小さな窓と重みを掛けて足す cross-correlation として `src/tensor.rs` の設計案 `Tensor::conv2d(weights)` に追加する。NCHW は画像の軸がバッチ、入力 channel、高さ、幅、OIHW は重みの軸が出力 channel、入力 channel、窓の高さ、幅であるという意味である。入力 `[N,C,H,W]` と重み `[O,C,KH,KW]` の channel 一致、正の KH/KW、窓が入力へ収まる条件を先に検証する。出力の高さ `OH=H-KH+1`、幅 `OW=W-KW+1` を計算してからノードを作る。

各 kh,kw の順を kh 優先、kw が内側と固定し、`SHRINK(X,[(0,N),(0,C),(kh,kh+OH),(kw,kw+OW)])` で `[N,C,OH,OW]` の窓片を作る。先頭軸の STACK は `[KH*KW,N,C,OH,OW]` になるので、reshape で `[KH,KW,N,C,OH,OW]`、permute `[2,3,4,5,0,1]` で `[N,C,OH,OW,KH,KW]` とし、長さ1の出力 channel 軸を挿入して `[N,1,C,OH,OW,KH,KW]` を得る。重みを `[1,O,C,1,1,KH,KW]` に reshape して掛ける。積の軸2,5,6、つまり C,KH,KW を同時に縮約し `[N,O,OH,OW]` を返す。軸を一つずつ削除する実装なら後ろの軸から縮約して軸番号のずれを避ける。

冒頭の入力で出力 `(0,0,0,1)` の窓は `[[2,3],[5,6]]` であり、重みが全1なら積の累積は0,2,5,10,16となる。重み `[[1,0],[0,-1]]` なら `2-6=-4` で、窓を反転していないと確認できる。直接読む kernel 添字は `((n*C+c)*H+(oh+kh))*W+(ow+kw)` となる。

テストでは window shape とこの座標を dump で追い、複数 C と O の固定値も参照ループと比較する。中間窓を materialize する初版でも、入力と別 storage に保存し schedule に境界を出す。巨大な窓を host で計算する抜け道は作らず、SHRINK/STACK から既存の compiler 全体を通す。


窓の合成を下げた C を `src/rangeify.rs` の添字生成と比較する。これは NCHW と OIHW の連続配置を受け取る kernel 本体の抜粋である。寸法とバッファ長は host 側で検証する。

組込み用の抜粋。

Tensor の conv 合成を rangeify で下げた後に renderer が出す C。

```c
// src/renderer/c.rs が生成するコード。生成先: <一時ディレクトリ>/conv_valid.c
void conv_valid(const float *x, const float *w, float *out,
                int N, int C, int H, int W, int O, int KH, int KW) {
	int OH = H-KH+1, OW = W-KW+1;
	for (int n=0; n<N; n++)
	for (int o=0; o<O; o++)
	for (int h=0; h<OH; h++)
	for (int v=0; v<OW; v++) {
		float acc = 0.0f;
		for (int c=0; c<C; c++)
		for (int kh=0; kh<KH; kh++)
		for (int kw=0; kw<KW; kw++) {
			int xi = ((n*C+c)*H+h+kh)*W+v+kw;
			int wi = ((o*C+c)*KH+kh)*KW+kw;
			acc += x[xi] * w[wi];
		}
		out[((n*O+o)*OH+h)*OW+v] = acc;
	}
}
```

出力座標は n,o,h,v、縮約座標は c,kh,kw である。`xi` では窓の原点 h,v に kh,kw を足し、`wi` ではそのまま kh,kw を使う。重みの反転は行わない。

同じ入力を追う例。

```text
SHRINK pieces: [KH*KW,N,C,OH,OW]
RESHAPE: [KH,KW,N,C,OH,OW]
PERMUTE [2,3,4,5,0,1]: [N,C,OH,OW,KH,KW]
insert axis + MUL: [N,O,C,OH,OW,KH,KW]
SUM axes=[2,5,6]: [N,O,OH,OW]
h=0,v=1 -> x positions=[1,2,4,5] -> 2+3+5+6=16
```

窓を materialize する版と直接 LOAD する版は、この出力座標から同じ四要素を読む。shape 列だけでなく `xi` も照合する。

- [ ] `KH,KW` 個の SHRINK と STACK で窓を組み立てる。
- [ ] 座標順を明示し、window の shape を dump に出す。
- [ ] 窓と重みの積を `C,KH,KW` について縮約する。
- [ ] 出力の保存順を NCHW にそろえる。
- [ ] 手書き conv の参照ループをテスト側に置いて比較する。

予定例 `tensor_conv2d` は上記の `[[[[12,16],[24,28]]]]` を返す。
重み `[[[[1,0],[0,-1]]]]` では全出力が `-4` になる。
カーネルを上下左右に反転しないことを、この重みでも確認する。

### 4. bias と zero padding を合成する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── tensor_conv_bias.rs  # [追加] padding/bias を含む予定実行例。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # [更新] 安全な pad と縮約後 bias の生成 C。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rangeify.rs  # [更新] PAD の有効条件を LOAD まで維持。
│   └── tensor.rs  # [更新] padding と bias の合成、未対応引数の拒否。
├── tests/
│   └── tensor_ops.rs  # [更新] 非対称 pad、bias 再bind と不正値。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/tensor.rs の conv2d 引数検証を更新して上下左右 padding を正規化し、入力 PAD と出力 ADD を合成する。bias は [1,O,1,1] へ reshape して既存 broadcast を使う。掲載 C の分岐と縮約後 ADD は src/rangeify.rs と src/renderer/c.rs の既存経路へ反映する。


このコミットで足す padding と bias は conv の意味を変える新しいカーネルではなく、入力側の PAD と出力側の ADD である。候補は `src/tensor.rs` の conv の引数検証と合成手順で、設計案として padding を上下左右の非負整数へ正規化する。既存例の `padding=1` は上下左右すべて1の意味とする。元の N,C を変えず、高さと幅だけを pad し、`Hp=H+top+bottom,Wp=W+left+right`、`OH=Hp-KH+1,OW=Wp-KW+1` を overflow 検査付きで計算する。stride、dilation、groups は許可値1以外をこの入口で拒否する。

PAD 後の窓から元画像を読む座標は `(n,c,oh+kh-top,ow+kw-left)` で、有効条件は元の高さと幅の範囲内である。最終的な LOAD の手前までこの条件を持ち、padding の領域では読まずにゼロを返す。bias は出力 channel ごとの `[O]` とし、`[1,O,1,1]` に reshape して `[N,O,OH,OW]` の conv 結果へ加える。rank2 の matmul の bias `[N]` は `[1,N]` とするので、どちらも既存 broadcast の規則だけで実現できる。

入力が shape `[1,1,1,1]` で値1、重みが2×2の全1、padding=1なら、pad 後の3×3画像は中央だけ1となる。四つの2×2窓はいずれもその1を一回含むので出力は `[[[[1,1],[1,1]]]]` である。元画像の負の位置を読み出してからゼロへ選び直す実装では、この例の端で不正読出しを起こす。冒頭の3×3入力で padding=0、bias=1なら、既存出力12,16,24,28へそれぞれ1が足され13,17,25,29になる。

`tests/tensor_ops.rs` では上記二例、非対称 padding、O=2の異なる bias、bias の長さ違いを確認する。kernel dump と C source で pad 条件の内側に LOAD があること、bias の添字が channel だけで決まることを確認する。Plan の入力 slot へ bias を rebind し、重みと同様に再コンパイルせず値だけ変わることを検査する。conv の合成と10の安全な pad を再利用して、今後の学習へ必要な入口を完成させる。


padding を下げる場所は入力 LOAD、bias を下げる場所は縮約後である。次は一出力を計算する生成 C の本体で、n,o,oh,ow と各寸法は外側で定義した座標と静的寸法を表す。

組込み用の抜粋。

PAD と bias を含む生成 kernel の一出力本体。外側の座標定義と結合する出力例。

```c
// src/renderer/c.rs が生成するコード。生成先: <一時ディレクトリ>/conv_bias.c
float acc = 0.0f;
for (int c=0; c<C; c++) {
	for (int kh=0; kh<KH; kh++) {
		for (int kw=0; kw<KW; kw++) {
			int ih = oh + kh - top;
			int iw = ow + kw - left;
			float value = 0.0f;
			if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
				value = x[((n*C+c)*H+ih)*W+iw];
			}
			acc += value * w[((o*C+c)*KH+kh)*KW+kw];
		}
	}
}
out[((n*O+o)*OH+oh)*OW+ow] = acc + bias[o];
```

`top,left` は PAD の前側の長さで、後側の padding は OH,OW の計算にだけ現れる。bias は c,kh,kw へ broadcast しない。縮約が終わってから出力 channel o に対して一度だけ足す。

同じ入力を追う例。

```text
X=[[[[1]]]], padding=(1,1,1,1), W=ones([1,1,2,2])
output (0,0,0,0): input candidates=(-1,-1),(-1,0),(0,-1),(0,0)
valid values=[0,0,0,1], sum=1
bias=[2] -> output=3
```

四候補のうち三つは LOAD が存在しない分岐となる。最後の一つだけが x[0] を読む。bias を窓の内部で足すと誤って四回足すことになる。

- [ ] conv の前に PAD を挿入し、出力 shape の計算に反映する。
- [ ] channel ごとの bias `[O]` を reshape して加算する。
- [ ] matmul の後の bias と同じ broadcast 規則を使う。
- [ ] stride、dilation、groups の未対応値を明示的に拒否する。
- [ ] pad の外側を読む可能性がある LOAD を分岐内へ保持する。

予定例 `tensor_conv_bias` は padding=0、bias=`[1]` で `[[[[13,17],[25,29]]]]` を返す。
`X=[[[[1]]]],W=[[[[1,1],[1,1]]]],padding=1` は `[[[[1,1],[1,1]]]]` になる。
stride 2 を黙って stride 1 として扱わない。

## 完了確認

次の example と `tests/tensor_ops.rs` は本ステップで追加する予定である。

```sh
# 作業ディレクトリ: chibigrad/
cargo test
cargo run --example tensor_chain -- --dump-graph --dump-schedule
cargo run --example tensor_matmul -- --backend cpu --dump-kernel
cargo run --example tensor_conv2d -- --backend cpu --dump-source
cargo run --example tensor_conv_bias -- --padding 1
```

- [ ] 予定例の期待値が一致し、手書きループとも一致する。
- [ ] graph に matmul と conv 専用の実行器への抜け道がない。
- [ ] reduction の初期化が出力要素ごとに一回だけ実行される。
- [ ] 学習用重みを CONST として intern せず、入力値を差し替えられる。
- [ ] 非正方行列、複数 channel、複数 output channel の小例を固定値で比較する。
- [ ] graph、schedule、生成 C の順序から同じ計算であると追える。

## tinygrad ソースとの対応

参照 commit は `c22edff14c79310a9b2df62a78819cf77c8289fa` である。

| 実在するシンボル | ソース | 役割 | chibigrad での簡略化 |
|---|---|---|---|
| `Tensor` | [tensor.py](../../tinygrad/tinygrad/tensor.py#L44) | UOp と利用者向け API を結び付ける | 静的 f32 と明示的 compile に限る |
| `OpMixin.dot` | [op.py](../../tinygrad/tinygrad/mixin/op.py#L390) | reshape、transpose、積、sum で行列積を作る | rank 2 だけを受け付ける |
| `OpMixin.matmul` | [op.py](../../tinygrad/tinygrad/mixin/op.py#L417) | dot への入口 | reverse と dtype 指定は後回しにする |
| `OpMixin.conv2d` | [op.py](../../tinygrad/tinygrad/mixin/op.py#L1585) | 窓、broadcast、積、sum で conv を作る | Winograd、image、groups を持ち込まない |
| `MovementMixin.stack` | [movement.py](../../tinygrad/tinygrad/mixin/movement.py#L269) | 複数 tensor を新しい軸へ並べる | 小さい窓の組立てに使う |

## 今は追加しないもの

batched matmul、一般 stride と dilation は必要な最小例から追加する。
速い GEMM と conv、窓バッファの削減は [19](19-fast-gemm-conv.md) で扱う。
tensor core は [25](25-hardware-extensions.md) へ分け、ここでは数値と合成を完成条件にする。
