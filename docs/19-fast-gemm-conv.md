# 19. GEMM と convolution をタイル化し、中間メモリを再利用する

## このステップで動かすもの

静的 f32 の GEMM と conv を、同じ入力を使う逐次版と比較する。
CPU、Metal、OpenCL で、一度に一つの最適化を有効にする。
[18. 並列 reduction](18-parallel-reduction.md) までの local memory と barrier を前提にする。
元計画の第10章を具体化する。第7章の合成 matmul と conv を比較用に残す。
ここに書く API、example、test、コマンドはこれから追加する予定である。
性能値はこの文書では測定していない。速さの数値を完了条件にしない。

予定の小入力は次とする。

```text
A=[[1,2,3],[4,5,6]], B=[[1,2],[3,4],[5,6]]
A@B=[[22,28],[49,64]]
x=[[[[1,2,3],[4,5,6],[7,8,9]]]], w=[[[[1,1],[1,1]]]]
conv(x,w)=[[[[12,16],[24,28]]]]
```

## 最小実装とデータ

入力は既存の multiply と REDUCE に分解された shape 付きグラフである。
出力は分割した RANGE、LOCAL の ALLOC、guard 付き INDEX、LOAD、STORE、BARRIER を持つ kernel IR である。
意味を保つ変換として `GemmTile { m, n, k }` を一つ追加する予定とする。
全演算を認識する巨大な pattern 集は作らず、最初は example が作る GEMM の形だけを認識する。
CPU は C と system clang の共有ライブラリ経路を保つ。
GPU は一つの group が出力の一つのタイルを担当する。
layout は論理 shape と分け、各 local buffer の pitch を明示する。
メモリ計画は kernel の変更とは別の runtime Plan の変換にする。

## この章のファイル配置

19章では既存の合成GEMMとconvを残し、src/schedule.rsのkernel変換、各rendererの描画、runtimeの実行準備を順に広げる。mainを持つ数値例はexamples/book_19_tiled_gemm.rsへ分離する。

次はこの章の終わりまでに触る関連ファイルの配置案である。現在のcheckoutの一覧ではない。各コミットでは、その下に示す部分だけを追加・変更する。

```text
chibigrad/
├── examples/
│   ├── book_19_tiled_gemm.rs  # 本文のコードを配置する。
│   ├── fast_conv.rs  # 合成convとtile版を比較するdriver。
│   ├── fast_gemm.rs  # cold、warmup、実行、同期を分けた記録。
│   └── memory_reuse.rs  # forkとchainのPlan割当を比較するdriver。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # 連続範囲のvector LOAD。
│   │   ├── metal.rs  # local layoutに沿うvector LOAD。
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── opencl.rs  # 同じlayoutのOpenCL描画。
│   ├── runtime/
│   │   ├── cpu.rs  # compile_keyとcompile前のbinary cache。
│   │   ├── metal.rs  # tile資源とlaunchの検査。
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── opencl.rs  # local資源と実global sizeの検査。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rangeify.rs  # b_localとconv座標のINDEX生成。
│   └── schedule.rs  # CountOpとestimate_workによる仕事量の推定。
├── tests/
│   ├── fast_gemm_conv.rs  # cache再利用と操作数20、24の確認。
│   └── memory_reuse.rs  # storage baseの別名と寿命の統合確認。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

生成Cやshaderは一時ディレクトリへ出力するため、このRustプロジェクトのツリーには含めない。コード冒頭のコメントで、生成元と生成先を確認する。

## 小さなコミットで進める

### 19a. CPU の縮約を分割する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_19_tiled_gemm.rs  # 本文のコードを配置する。
│   └── fast_gemm.rs  # [追加] Tensorから実行まで通すGEMM driver。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # [更新] 分割RANGEとregister累積のC描画。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── runtime/
│   │   ├── cpu.rs  # [更新] 出力tileを既存thread区間へ割り当てる。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── schedule.rs  # [更新] tile_gemmによるK、M、Nの分割。
├── tests/
│   └── fast_gemm_conv.rs  # [追加] 逐次版との全要素比較。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/schedule.rsの最適化設定からtile_gemmを呼び、既存RANGEをK、M、Nの順に分割する。src/renderer/c.rsにacc配列の初期化と分割ループを加える。掲載のtiledとmainは座標確認専用の別exampleへ置き、fast_gemmでは既存TensorとCPU runtimeを呼ぶ。


このコミットの入力は row-major の A[M,K] と B[K,N]、出力は C[M,N] である。編集候補 `src/schedule.rs` ではループを分割し、renderer が次の積和を C へ描く。まず座標と部分和の寿命を、単独で実行できる Rust の例で確かめる。

tiledとmainを含む数値確認用ファイル全体。fast_gemmのcompiler経路とは分けて実行する。

```rust
// examples/book_19_tiled_gemm.rs（単独で実行する例）
fn tiled(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
	assert_eq!(a.len(), m*k);
	assert_eq!(b.len(), k*n);
	let mut out = vec![0.; m*n];
	for mo in (0..m).step_by(2) {
		for no in (0..n).step_by(2) {
			let mut acc = [[0.; 2]; 2];
			for ko in (0..k).step_by(2) {
				for ki in 0..2 {
					let kk = ko + ki;
					if kk >= k { continue; }
					for mi in 0..2 {
						for ni in 0..2 {
							let (i,j) = (mo+mi,no+ni);
							if i < m && j < n {
								acc[mi][ni] += a[i*k+kk]*b[kk*n+j];
							}
						}
					}
				}
			}
			for mi in 0..2 {
				for ni in 0..2 {
					if mo+mi < m && no+ni < n {
						out[(mo+mi)*n+no+ni] = acc[mi][ni];
					}
				}
			}
		}
	}
	out
}
fn main() {
	let c = tiled(&[1.,2.,3.,4.,5.,6.], &[1.,2.,3.,4.,5.,6.], 2,2,3);
	assert_eq!(c, [22.,28.,49.,64.]);
	println!("{c:?}");
}
```

`mo,no` は出力タイルの左上、`mi,ni` は内部座標、`kk=ko+ki` は縮約座標である。`acc` は `ko` より外で作るので、`acc[0][0]` は7から22へ増える。`kk>=k` の分岐は A/B の添字を作る前に抜ける。これは CPU の guard 付き方式であり、無効な積和も実行する GPU のゼロ詰めとは仕事量が異なる。

次は同じ座標を表す予定の IR dump である。これを既存の RANGE lowering へ渡し、以下のチェック項目で生成 C と値を確認する。

```text
ALLOC_REG acc[2,2] = 0              scope=(mo,no)
RANGE ko=0..ceildiv(K,2)
  RANGE ki=0..2
    kk=2*ko+ki
    IF kk<K AND mo+mi<M AND no+ni<N
      acc[mi,ni] += LOAD A[(mo+mi)*K+kk] * LOAD B[kk*N+no+ni]
STORE C[(mo+mi)*N+no+ni] = acc[mi,ni]  guard=(mo+mi<M && no+ni<N)
expected acc00: 0 -> 7 -> 22
```

既存の合成 GEMM を入口に残し、`src/schedule.rs` の最適化設定から `tile_gemm` を呼ぶ案とする。この関数は reduction の入力が A の `[i,k]` と B の `[k,j]` の積であることを確かめ、M、N、K と対応する RANGE を取り出す。出力は元 graph の置換ではなく、その kernel 専用のループ構造である。`src/renderer/c.rs` は分割後の RANGE を通常の C ループとして描き、runtime の `buffers` と `scalars` の二配列 ABI は変更しない。

実装は K の分割だけから始める。各出力の `acc=0` を ko より外に置き、`ko=0..ceil(K/TK)`、`ki=0..TK` の順で `k=ko*TK+ki` を作る。`k<K` の時だけ A と B を読み、積を acc に加える。その後に M と N へ同じ分割を施し、出力タイルごとの `acc[TM][TN]` を一度初期化する。K tile を移るたびに初期化すると、それ以前の部分和が消えるため、初期化の scope を dump に示す。M と N の末尾は読出しと保存の両方を守る。

小例の出力 `[0,0]` は TK=2 なら最初の tile で `1*1+2*3=7`、次で `3*5=15` を足して22となる。最後の ki=1 は k=3なので LOAD しない。`[1,1]` は `4*2+5*4=28` から `6*6` を足して64になる。この二つの途中値を逐次版と比較する。CPU thread は出力 tile の番号にだけ割り当て、同じ acc や STORE 先を共有しない。端数入力では全出力座標の被覆が一回ずつであることと、生成 C に K より外側の初期化があることを検査する。既存の compiler と共有ライブラリ経路を通した値まで一致すれば、この変換単独の確認になる。

- [ ] `i,j,k` の逐次 GEMM を比較用の設定で呼び出せるようにする。
- [ ] `k=ko*TK+ki` とする変換を入れ、端数は `k<K` で守る。
- [ ] `M,N` のタイル分割を追加し、最初は各タイル内の計算順を変えない。
- [ ] 小さな register 累積配列を C に出し、初期化を K ループの外に置く。
- [ ] 出力タイルの並列化は17の CPU thread 経路を使い、同じ出力要素を複数 thread が書かないようにする。

予定例 `fast_gemm --backend cpu --tile 2,2,2 --case small` は上記の `[[22,28],[49,64]]` を返す。
`M=3,N=5,K=7` も動かし、端数を含む出力の全要素を逐次版と比較する。
各コミットで Tensor、schedule、C renderer、共有ライブラリ、固定 ABI まで通す。

### 19b. GPU の A と B を local memory に置く

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── fast_gemm.rs  # [更新] GPUのtails条件を選ぶ。
├── src/
│   ├── renderer/
│   │   ├── metal.rs  # [更新] 同じ搬入と同期のMSL描画。
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── opencl.rs  # [更新] LOCAL、BARRIER、guardのOpenCL描画。
│   ├── runtime/
│   │   ├── metal.rs  # [更新] tile資源とlaunchの検査。
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── opencl.rs  # [更新] local資源と実global sizeの検査。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── schedule.rs  # [更新] stage_gemm_tilesと協調搬入のIR。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/schedule.rsにstage_gemm_tilesを追加し、搬入、barrier、積和、barrierの順を作る。両GPU rendererの既存LOCALとBARRIER描画を広げる。掲載gemm8はOpenCL rendererの生成結果であり、runtimeにはshader本文を貼らずslotとlaunchだけを渡す。


入力は19aと同じ A/B とタイル設定、出力は各 group が別の C タイルを保存する kernel である。`stage_gemm_tiles` の生成結果を、組込み用の抜粋として OpenCL kernel で示す。TM=TN=TK=8、local size=64を host で検査し、`groups=(ceil(N/8),ceil(M/8))` とする。実機実行は未確認である。

stage_gemm_tilesのIRを描いたOpenCL C。Rustのruntimeファイルへ貼らない。

```c
// src/renderer/opencl.rs が生成するコード。生成先: <一時ディレクトリ>/gemm8.cl
__kernel void gemm8(__global const float *a, __global const float *b,
                    __global float *c, uint M, uint N, uint K) {
	uint lane = get_local_id(0);
	uint row = get_group_id(1)*8 + lane/8;
	uint col = get_group_id(0)*8 + lane%8;
	__local float at[64], bt[64];
	float acc = 0.0f;
	for (uint ko=0; ko<K; ko+=8) {
		for (uint q=lane; q<64; q+=64) {
			uint ar=get_group_id(1)*8+q/8, ak=ko+q%8;
			uint bk=ko+q/8, bc=get_group_id(0)*8+q%8;
			at[q] = (ar<M && ak<K) ? a[ar*K+ak] : 0.0f;
			bt[q] = (bk<K && bc<N) ? b[bk*N+bc] : 0.0f;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		for (uint ki=0; ki<8; ++ki)
			acc += at[(lane/8)*8+ki]*bt[ki*8+lane%8];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (row<M && col<N) c[row*N+col]=acc;
}
```

`lane/8,lane%8` が出力内座標である。条件演算子は選ばれた側だけを評価するので、範囲外の global LOAD をしない。`at,bt` は毎回64要素すべてを書き直す。二つの barrier は出力 guard の外にあり、無効出力の lane も次の搬入まで待つ。Metal renderer では LOCAL を `threadgroup`、barrier を `threadgroup_barrier(mem_flags::mem_threadgroup)` に描く。

```text
M=8,N=8,K=9, A=B=ones, local=64
ko=0: at/bt valid 64 elements, acc=8
ko=8: A local columns 1..7=0, B local rows 1..7=0, acc=9
OpenCL groups=(1,1), global=(64,1), local=(64,1)
output[0,0]=9; barrier visits per lane=4
```

CPU の tile 座標を保ち、`src/schedule.rs` に local tile の搬入を作る `stage_gemm_tiles`、Metal と OpenCL の renderer に LOCAL と BARRIER の描画を足す案である。入力は TM、TN、TK、group の出力起点と lane 番号、出力は協調保存、同期、積和、再同期の順を持つ kernel IR とする。group の lane 数を L とすると、A の平坦な要素番号を `q=lane, lane+L, ...` と巡回し、`r=q/TK, c=q%TK` へ戻して搬入する。B も同様に全要素を一回ずつ担当させる。

K tile ごとに local の全要素へ必ず値を書き、有効な global 座標だけを LOAD、範囲外は0を STOREする。搬入完了の barrier の後に acc へ積和を加え、全 lane が計算を終えた barrier の後に次の搬入へ進む。前者は未初期化値の読出し、後者は別 lane が読んでいる値の上書きを防ぐ。出力の有効性で group 全体を途中 return させない。M や N の端の lane も local 搬入と二つの barrier に参加し、最後の global STORE だけを条件付きにする。

たとえば TK=8、K=9では、二番目の tile の c=0だけが k=8を表す。c=1..7には0を書き、最初の tile の k=1..7の値を残さない。A と B が全て1なら各有効出力は8から9へ増え、16にはならない。検査では local の各添字が確保要素数未満であること、各 barrier が lane 条件の外にあることを IR と source の両方で読む。Metal の group 数と OpenCL の実 global size を dump し、後者が group 数と local size の積になっているか確かめる。同じ二配列 ABI の論理 slot から backend 引数へ設定し、入力を変更した再実行でも同じ搬入構造を使う。

- [ ] Metal で A の `TM*TK`、B の `TK*TN` の local buffer を確保する。
- [ ] lane ごとの協調 LOAD を作り、範囲外は LOAD を実行せず local にゼロを保存する。
- [ ] 保存後の barrier と、計算後に local buffer を再利用する前の barrier を入れる。
- [ ] 無効な出力 lane も barrier を通し、最後の global STORE だけを guard する。
- [ ] 同じ IR を OpenCL に出し、`groups` と `local` の積を実際の global size とする。

予定例 `fast_gemm --backend metal --tile 8,8,8 --case tails` を動かす。
Metal が使えない環境では CPU 比較までを通し、Metal 実機確認は未完了として記録する。
OpenCL も実機を検出してから同じ例を動かし、非対応を成功扱いにしない。
`K=1,7,8,9` で、最後の local tile に前の tile の値が残らないことを確認する。

### 19c. layout と convolution を一つずつ変える

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── fast_conv.rs  # [追加] 合成convとtile版を比較するdriver。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # [更新] 連続範囲のvector LOAD。
│   │   ├── metal.rs  # [更新] local layoutに沿うvector LOAD。
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── opencl.rs  # [更新] 同じlayoutのOpenCL描画。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rangeify.rs  # [更新] b_localとconv座標のINDEX生成。
│   └── schedule.rs  # [更新] 転置とpaddingを含むlocal layout。
├── tests/
│   └── fast_gemm_conv.rs  # [更新] direct_conv_oneを用いるscalar reference。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/rangeify.rsにb_localを置き、搬入と積和の添字を同じlayoutから作る。direct_conv_oneは実計算の代替ではなくtests/fast_gemm_conv.rsの比較用helperへ置く。compiler側では同関数のxi、weight、ifに対応するINDEXとLOAD guardをrangeifyへ加える。layout変更と各rendererのvector化を順に確認する。


入力は B の論理座標と NCHW/OIHW の conv 座標、出力は local offset または入力 offset である。`src/rangeify.rs` へ置く組込み用の抜粋は添字計算を明示する。`stride>0`、各寸法の積と符号付き座標が host の整数幅に収まることは起動前の検査とする。

b_localはlocal添字生成へ、direct_conv_oneはintegration test内のscalar比較helperへ追加する。

```rust
// src/rangeify.rs（既存ファイルへ組み込む抜粋）
fn b_local(j: usize, ki: usize, tk: usize) -> usize {
	let pitch = tk + 1;
	j*pitch + ki
}
// tests/fast_gemm_conv.rs（既存ファイルへ組み込む抜粋）
fn direct_conv_one(
	x: &[f32], w: &[f32], ci: usize, hi: usize, wi: usize,
	kh: usize, kw: usize, oc: usize, oy: usize, ox: usize,
	stride: usize, pad: isize,
) -> f32 {
	let mut acc = 0.;
	for ic in 0..ci {
		for ky in 0..kh {
			for kx in 0..kw {
				let iy = (oy*stride+ky) as isize-pad;
				let ix = (ox*stride+kx) as isize-pad;
				if iy>=0 && iy<hi as isize && ix>=0 && ix<wi as isize {
					let xi = (ic*hi+iy as usize)*wi+ix as usize;
					let weight = ((oc*ci+ic)*kh+ky)*kw+kx;
					acc += x[xi]*w[weight];
				}
			}
		}
	}
	acc
}
```

`b_local` は搬入側と消費側の両方で使う式で、`j=1,ki=0,TK=2` は3になる。`direct_conv_one` は batch 一つの slice を受ける座標 reference である。実際の graph では `xi` の式を INDEX へ、`if` を LOAD guard へ変換する。`weight` の `oc` は保存形式 OIHW の最外軸であり、入力のチャネルと交換しない。

```text
B local: TN=2,TK=2,pitch=3, allocation=6*f32=24 bytes
(j,ki)=(0,0),(0,1),(1,0),(1,1) -> offset=0,1,3,4
conv output(oy=0,ox=0), stride=1,pad=0:
  x offsets=0,1,3,4; values=1,2,4,5; w=1 -> 12
conv output(oy=1,ox=1): values=5,6,8,9 -> 28
```

この見出し内も既存方針どおり、B の転置、pitch の padding、vector 化、conv の順で別々に確認する。変更先は `src/schedule.rs` の tile layout、`src/rangeify.rs` の添字生成、各 renderer の vector LOAD とする案である。layout は論理座標を local の要素番号へ写す小さな記録とし、shape の変更とは分ける。通常の B は `ki*TN+j`、転置後は `j*TK+ki` を使う。LOAD 側だけを転置すると積和側が別要素を読むので、搬入時と消費時の式を同じ layout から作る。

転置した B に padding=1を入れるなら pitch は `TK+1`、確保要素数は `TN*pitch`、アクセスは `j*pitch+ki` である。TN=2、TK=2なら確保は6要素で、有効添字は0、1、3、4となる。2と5は padding なので計算に使わない。必要 byte 数は要素数に4を掛け、最大の有効添字4が確保範囲内であることを検査する。vector LOAD はこの添字列の連続部分に限り、padding をまたがず、末尾と alignment を満たす場合だけ選ぶ。

conv は NCHW の入力と OIHW の重みを固定し、出力の `(n,oc,oy,ox)` と縮約の `(ic,ky,kx)` から `iy=oy*stride+ky-pad`、`ix=ox*stride+kx-pad` を作る。最初の2x2例では出力左上が入力1、2、4、5を読み、和12になる。縮約座標を平坦化してもこの読出し順と対応は保持する。padding を導入した時は iy と ix の有効判定を LOAD の前に置く。全体 im2col は作らず、現在の tile に必要な座標だけを搬入する。小例とチャネル端数で11の合成版、layout 変更前、変更後を比較し、値の差が出た時にどの変換で変わったかを特定できるようにする。

- [ ] まず B の local buffer のみを転置し、協調 LOAD と計算時 INDEX の両方を変更する。
- [ ] local pitch の padding を一つ追加し、確保 byte 数と INDEX の最大値を検証する。
- [ ] vector LOAD は連続性、alignment、範囲を確認できる内側だけに適用する。
- [ ] conv の出力座標と縮約座標を GEMM と同じタイルへ写像する。
- [ ] まず stride=1、padding=0、groups=1 の direct conv を扱い、次に padding と stride を追加する。

予定例 `fast_conv --backend cpu --case ones2x2` は `[[12,16],[24,28]]` を返す。
conv は11の合成版を比較先とする。全体の im2col buffer は必須にしない。
stride=2、境界 padding、チャネル数3、出力チャネル数5を一条件ずつ追加して比較する。
layout、vector 化、conv 変換を一つのコミットにまとめない。

### 19d. 生存期間が重ならない中間 buffer を再利用する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── memory_reuse.rs  # [追加] forkとchainのPlan割当を比較するdriver。
├── src/
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── schedule.rs  # [更新] Class、Lifeとplan_storage_reuse。
├── tests/
│   └── memory_reuse.rs  # [追加] storage baseの別名と寿命の統合確認。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

候補のうちsrc/schedule.rsへClass、Life、plan_storage_reuseをまとめる。Planの実行順が確定した後にstorage baseごとの寿命を集計し、返されたallocation表を既存runtimeへ渡す。NodeIdや元のgraphは置き換えない。


入力は storage base ごとに view と AFTER の利用区間を合併した寿命表、出力は論理 buffer ごとの allocation ID である。次は `plan_storage_reuse` の組込み用の抜粋である。入力 `lives` は first の昇順、IDは一意、held は入力・出力・parameter・cache・外部 view を含む。`class` は device、dtype、容量、alignment の完全一致キーとする。

既存Planのメモリ計画部分へ型と割当関数を追加する。

```rust
// src/schedule.rs（既存ファイルへ組み込む抜粋）
#[derive(Clone, PartialEq, Eq)]
struct Class { device: String, dtype: u8, bytes: usize, align: usize }
struct Life { first: usize, last: usize, class: Class, held: bool }
fn plan_storage_reuse(lives: &[Life]) -> Vec<usize> {
	let mut allocations: Vec<(Class, usize, bool)> = Vec::new();
	let mut mapping = Vec::new();
	for life in lives {
		assert!(life.first<=life.last);
		let reusable = if life.held { None } else {
			allocations.iter().position(|(class,last,held)|
				!held && *last<life.first && *class==life.class)
		};
		let slot = match reusable {
			Some(i) => {
				allocations[i].1 = life.last;
				i
			}
			None => {
				allocations.push((life.class.clone(),life.last,life.held));
				allocations.len()-1
			}
		};
		mapping.push(slot);
	}
	mapping
}
```

`allocations` は実領域、`mapping` は論理 buffer の表であり、NodeIdを変えない。`last<first` の厳密不等号によって同じ kernel 内の入力を出力で上書きしない。held 領域は新規にも再利用先にもならず、以後も保持する。実行順が単一 queue の場合だけこの整数区間で十分であり、非同期 queue へ広げる際は完了依存が必要になる。

```text
fork: a=[0,2], b=[1,3], c=[2,3] -> allocation=[0,1,2]
chain: a=[0,1], b=[1,2], c=[2,3] -> allocation=[0,1,0]
first=2 で last=2 の領域は不可、last=1 の領域は可
x=[1,2,3]; fork out=(x+1)*2+(x+1)*3=[10,15,20]
```

kernel の tile 化を済ませた Plan を入力にし、`src/schedule.rs` またはそのメモリ計画部分へ `plan_storage_reuse` を追加する案である。各論理 buffer について storage base、device、dtype、容量、alignment、外部保持の有無、最初と最後の kernel 番号を記録する。view の先頭ポインタでは別領域と判定せず、同じ storage base を使う全 view の利用区間を合併する。返すのは論理 buffer から実際の確保先への割当表であり、元 graph の NodeId や parameter の識別は変えない。

kernel を実行順に走査し、番号 i の入出力を割り当てる前には `last_use<i` の中間領域だけを free list に戻す。`last_use==i` の領域は、その kernel がまだ読む可能性があるので戻さない。新しい出力には条件が全て同じ空き領域を選び、なければ確保する。入力、出力、parameter、cache、外部 view は保持集合として最初から対象外にする。AFTER が示す書込み完了と読出しの依存も利用記録へ入れる。

fork 例を4 kernel とすれば、a は kernel0で生成し1と2が読み、区間は[0,2]になる。b は[1,3]、c は[2,3]である。a を c の出力へ割り当てると、kernel2の入力と出力が重なり禁止条件に当たる。この小例では、保守的な規則によって再利用が起きなくても正しい。再利用の成功例は `a=x+1; b=a*2; c=b+3; out=c*4` のような別 test に置き、最後の読出しを終えた a の領域だけを c に使えることを確かめる。各実行前の入力と各出力の値、割当表、実確保の peak を再利用無効版と照合する。確保削減を仮定せず、数値と寿命の両方からこの規則を確認する。

- [ ] Plan の各 kernel が読む範囲と書く範囲を storage base 単位で集計する。
- [ ] 入力、出力、学習パラメータ、KV cache、外部へ返した view を再利用対象から外す。
- [ ] 単一 queue の完了順を前提に、中間 buffer の最初の利用と最後の利用を求める。
- [ ] まず同じ device、dtype、サイズ、alignment の確保だけを free list で再利用する。
- [ ] 同じ kernel 内で読まれる buffer を、その kernel の出力に割り当てない。

予定例 `memory_reuse --case fork` は `a=x+1; b=a*2; c=a*3; out=b+c` を計算する。
`x=[1,2,3]` の出力は `[10,15,20]` である。a の最後の読出しまで再利用を禁止する。
再利用前後で実際の数値と peak 確保 byte 数を比較する。
view の別名と `AFTER` の依存も最後の利用に含め、古い値を次回実行へ持ち越さない。

### 19e. コンパイル cache と計測区間を固定する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── fast_gemm.rs  # [更新] cold、warmup、実行、同期を分けた記録。
├── src/
│   ├── runtime/
│   │   ├── cpu.rs  # [更新] compile_keyとcompile前のbinary cache。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── schedule.rs  # [更新] CountOpとestimate_workによる仕事量の推定。
├── tests/
│   └── fast_gemm_conv.rs  # [更新] cache再利用と操作数20、24の確認。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/runtime/cpu.rsのcompile呼出し直前へcompile_keyとcache照合を加える。CountOpとestimate_workはsrc/schedule.rsの線形kernel解析へ置き、計測driverが有効仕事と実行仕事を読む。SHA-256は21dでsrc/fetch.rsに置くDigest256の検証済みlibrary adapterへ接続し、19章の時点でも選定済み実装でdigestを得る。入力値はcache keyへ入れない。


入力は生成 source とコンパイル条件、出力は曖昧な連結をしない cache key の byte 列である。組込み用の抜粋を `src/runtime/cpu.rs` の compile 前に置く。SHA-256 は21dと同じ検証済み library adapterへ渡し、暗号関数そのものを教材で実装しない。

compile_keyをcompile前へ追加し、CountOpとestimate_workをscheduleのkernel解析へ分ける。

```rust
// src/runtime/cpu.rs（既存ファイルへ組み込む抜粋）
fn compile_key(fields: &[&[u8]]) -> Vec<u8> {
	let mut encoded = Vec::new();
	for field in fields {
		encoded.extend_from_slice(&(field.len() as u64).to_le_bytes());
		encoded.extend_from_slice(field);
	}
	encoded
}
// src/schedule.rs（既存ファイルへ組み込む抜粋）
#[derive(Clone, Copy)]
enum CountOp { Range(u64), End, Add, Mul, Fma, Index, Load, Store }
fn estimate_work(ops: &[CountOp], launch_owners: u64) -> Option<u64> {
	let (mut multiplier, mut count) = (launch_owners,0u64);
	let mut stack = Vec::new();
	for op in ops {
		match op {
			CountOp::Range(n) => {
				stack.push(multiplier);
				multiplier = multiplier.checked_mul(*n)?;
			}
			CountOp::End => multiplier=stack.pop()?,
			CountOp::Add | CountOp::Mul => count=count.checked_add(multiplier)?,
			CountOp::Fma => count=count.checked_add(multiplier.checked_mul(2)?)?,
			CountOp::Index | CountOp::Load | CountOp::Store => {}
		}
	}
	if stack.is_empty() { Some(count) } else { None }
}
```

`fields` は backend、device capability、source、compiler識別、flags、ABI、sourceに現れない特化条件を順序固定で渡す。入力値とポインタは入れない。`launch_owners` はRANGEにまだ含まれない担当数だけである。`estimate_work` に guard を渡す前に、有効反復数が既知の guard を trip count へ反映する。動的で数えられない guard は不明と記録し、この関数から正確な値を得たとしない。

```text
key(["ab","c"]) starts 02 00 00 00 00 00 00 00 61 62 01...
key(["a","bc"]) starts 01 00 00 00 00 00 00 00 61 02...
y=x*2+1: [Range(10),Mul,Add,End], launch_owners=1 -> 20
GEMM logical: M=2,N=2,K=3 -> 2*2*2*3=24
GPU padded:  M=2,N=2,Kpad=4 -> 2*2*2*4=32
CPU guarded: valid k=3 per output -> 24
```

`src/runtime/cpu.rs` の compile 呼出しの前に cache を置き、共通の `CompileKey` と `estimate_work` を必要な範囲で追加する案とする。key は文字列の曖昧な連結で作らず、各項目を長さ付きで直列化して digest を取る。cache の binary には key と形式 version を対応させ、破損や不一致なら再コンパイルする。入力の実データや生ポインタは key に含めず、A と B の slot を再 bind して同じ binary の別入力実行を確かめる。device と compiler の異なる binary を取り違えないことも test にする。

仕事量の推定は線形化した kernel IR を順に読み、RANGE の開始で現在の反復倍率を stack に保存して範囲を掛け、END で元に戻す。浮動小数点の ADD、MUL は倍率分、FMA はその2倍を足す。launch の group と lane の担当数も掛けるが、すでに RANGE で表した軸は二重に掛けない。整数添字、比較、LOAD と STORE はこのカウンタから外す。有効仕事は元の論理 shape から別に計算する。

M=N=2、K=3の小例は有効仕事24である。TM=TN=TK=2で、K の最後を0詰めして4回の積和を実行する方式なら padded 仕事は32になる。一方、k<Kで積和自体を省く CPU 版は24であり、単に tile 数だけを掛けて32とはしない。dump には採用した guard と両方の見積りを残す。計測は初回 compile と warmup を別にし、各反復で必要な同期と結果の読出しを保証する。記録中の kernel 時間がどの仕事量と対応するかを固定してから FLOP/s を計算する。cache hit の数値、二回目の入力変更、20と24の操作数を、このコミット内の小 test として確認する。

- [ ] cache key に backend、device capability、生成 source、clang 等の compiler 識別、flags、ABI version を含める。
- [ ] shape 特化、tile、dtype、math 設定が source 以外で決まる場合も key に含める。
- [ ] cache hit の binary と cold compile の binary で、同じ数値結果を確認する。
- [ ] Plan の入力スロットに違う A、B、重みを渡し、コードを再利用して結果が変わることを確かめる。
- [ ] 計測記録に compile、確保、host/device 転送、kernel、同期、全体の区間を分けて出す。
- [ ] 浮動小数点 ALU の回数にループの反復数と launch の担当数を掛ける FLOP 見積りを追加する。
- [ ] 数学的に必要な仕事と、padding した tile で実行する仕事の見積りを別に出す。

予定例 `fast_gemm --backend cpu --bench --warmup 3 --repeat 20` を追加する。
予定の回数は計測方法の初期設定であり、測定結果ではない。
GPU の kernel 時間は完了待ちを含む host 時間と区別する。
サイズ、tile、device、compiler、実行回数、誤差、最小値と中央値を同じ記録に残す。
03の共有済みノード数を FLOP 数として使わない。整数の添字演算、LOAD、STORE も浮動小数点演算に数えない。
ADD と MUL を各1、FMAを2とする積和の規約なら、`M=2,N=2,K=3` の GEMM の有効仕事は `2*M*N*K=24 FLOPs` である。
`y=x*2+1,N=10` は20 FLOPsになる。RANGE と END の入れ子を反映し、tail guard と無効 lane を区別する。
特殊関数は数え方を明示し、未対応なら不明とする。padded な仕事量は実行条件を反映した見積りであり、hardware counter の測定値とは呼ばない。
FLOP/s は実測した kernel 時間と同じ仕事の FLOP 見積りから計算し、実測前は操作数の推定だけを表示する。

## 完了確認

以下はこのステップで作る予定の example と test のコマンドである。

```sh
# 作業ディレクトリ: chibigrad/
cargo test
cargo test --test fast_gemm_conv
cargo test --test memory_reuse
cargo run --example fast_gemm -- --backend cpu --case tails
cargo run --example fast_conv -- --backend cpu --case ones2x2
cargo run --example memory_reuse -- --case fork --dump-plan
cargo run --example fast_gemm -- --backend metal --case tails --dump-source
cargo run --example fast_gemm -- --backend opencl --case tails --dump-source
```

- [ ] 整数値の小例は上記の期待値と一致する。
- [ ] 固定 seed の f32 入力は scalar f64 reference に対し `abs_err <= 1e-4 + 1e-4*abs(ref)` を初期基準とする。
- [ ] 大きい K では誤差を記録し、加算順の差を理由に無条件で許容値を広げない。
- [ ] `M,N,K=1` と tile 前後の寸法、ゼロ入力、負値、非連続 view を試す。
- [ ] dump で全 lane の barrier 到達、local buffer 内の添字、出力の単独書込みを確認する。
- [ ] 再利用を無効にした結果、再利用した結果、cache hit の結果が一致する。
- [ ] まだ測っていない backend を「高速化済み」「実機確認済み」と書かない。
- [ ] FLOP の見積りが上記の20と24になり、tile の端数を増やしても有効仕事の数は変わらない。

## tinygrad ソースとの対応

参照は commit `c22edff14c79310a9b2df62a78819cf77c8289fa` の実コードである。

| 実在シンボルと相対リンク | tinygrad の役割 | chibigrad で簡略化する点 |
|---|---|---|
| [`Scheduler.shift_to`](../../tinygrad/tinygrad/codegen/opt/postrange.py#L65) | RANGE の分割と添字置換 | 固定 tile の明示変換から始める |
| [`Scheduler.apply_opt`](../../tinygrad/tinygrad/codegen/opt/postrange.py#L87) | 分割、padding、資源制約を検証 | 変換を一つずつ指定する |
| [`OpMixin.conv2d`](../../tinygrad/tinygrad/mixin/op.py#L1585) | movement、積、縮約で conv を表現 | groups=1 の direct conv を先に扱う |
| [`memory_plan_rewrite`](../../tinygrad/tinygrad/schedule/memory.py#L19) | 生存期間と device 別の領域割当 | 同サイズの free list から始める |
| [`Compiler.compile_cached`](../../tinygrad/tinygrad/device.py#L333) | source を key に compiler cache を使う | compiler、ABI、device 情報も明示して管理する |
| [`Estimates.from_uops`](../../tinygrad/tinygrad/renderer/__init__.py#L26) | ループ範囲と型を含めた操作数・アクセス量を見積もる | 有効 FLOP と padding を含む仕事の推定を分ける |

## 今は追加しないもの

BEAM search、自動 tile 探索、複数 queue、非同期転送、tensor core は追加しない。
tensor core は[25](25-hardware-extensions.md)で扱う。
非同期実行を後に加えるときは、host の投入順ではなく device の完了依存で生存期間を再定義する。
