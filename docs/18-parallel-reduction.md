# 18. GPU の local memory と barrier で並列 reduction を実行する

## このステップで動かすもの

[17 のループ変換](17-loop-optimizations.md) を使い、GPU の SUM を workgroup 内で並列化する。
元計画の第 10 章の「GPU の並列 reduction」に対応する。
[15](15-metal.md) と [16](16-opencl.md) の逐次 reduction を比較用に残す。

```text
x=[1,2,3,4,5,6,7,8,9,10], local=8
laneの部分和=[10,12,3,4,5,6,7,8]
local memoryへ保存し、全laneでbarrier
最初はlane0がlocal memoryを逐次加算 → out=55
次にtree reductionで加算 → out=55
row_sum([[1,2,3],[4,5,6]]) = [6,15]
```

入力数 10 を 8 lane で分けると、lane 0 と 1 がそれぞれ二つの要素を読む。
無効な入力はゼロとして扱い、LOAD 自体を実行しない。
以下の API、local size、example、テスト、コマンドは実装予定である。
Metal と OpenCL の実機で試した項目だけを確認済みとして記録する。

## 最小実装とデータ

IR の storage に GLOBAL と区別した LOCAL を追加する。
LOCAL buffer は一つの workgroup の全 lane が共有し、group 間では共有しない。
`BARRIER` は workgroup の制御同期と LOCAL memory の可視性を表す。
STORE と LOAD の依存を AFTER 相当で持ち、barrier を越えて移動させない。

```text
入力IR: REDUCE_ADD(x[row,col], col=0..N)
GPU IR:
  row=group_id, lane=local_id
  acc=0
  for chunk in 0..ceil(N/L):
    col=chunk*L+lane
    if col<N: acc += LOAD(x,row*N+col)
  STORE(local,lane,acc)
  BARRIER
  if lane==0: SUM(local[0..L]) → STORE(out,row)
```

一つの group が一つの行を担当し、launch は `groups=rows,local=L` にする。
L はまず 2 の冪だけを受け付け、device の thread 数と LOCAL byte 数を検証する。
Metal は `threadgroup` と `threadgroup_barrier(mem_flags::mem_threadgroup)` を出す。
OpenCL は `__local` と `barrier(CLK_LOCAL_MEM_FENCE)` を出す。
最初はコンパイル時の固定長 LOCAL 配列でよい。

出力の guard と barrier の位置を、単純な elementwise kernel と分ける。
無効な lane も `acc=0` を LOCAL に保存し、同じ回数の barrier を通る。
barrier の手前で `if(col>=N)return` を生成しない。
barrier を `if(lane<stride)` の内側に置かない。
複数 group の同期はこの命令では実現できないため、別 kernel の起動境界を使う。

## この章のファイル配置

並列 SUM の IR 構築を新規 src/codegen/reduction.rs に置き、LOCAL/BARRIER の型と検証、両 GPU renderer、runtime の上限検査をつなぐ。group 間の依存は src/schedule.rs で二つの kernel に分ける。

次はこの章の終わりまでに触る関連ファイルの配置案である。現在のcheckoutの一覧ではない。各コミットでは、その下に示す部分だけを追加・変更する。

```text
chibigrad/
├── examples/
│   ├── book_18_group_sum_reference.rs  # 本文のコードを配置する。
│   ├── gpu_reduce_local.rs  # lane0 集約の予定実行例。
│   ├── gpu_reduce_multipass.rs  # 二段 SUM の予定実行例。
│   ├── gpu_reduce_rows.rs  # 一 group 一行の予定実行例。
│   └── gpu_reduce_tree.rs  # tree と逐次版の予定比較例。
├── src/
│   ├── codegen/
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── reduction.rs  # chunk 部分和と二段目 SUM の lowering。
│   ├── renderer/
│   │   ├── metal.rs  # 同じ chunk 分割の MSL 出力。
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── opencl.rs  # chunk_sum の OpenCL C 出力。
│   ├── runtime/
│   │   ├── metal.rs  # 二 kernel の順序と中間 buffer の寿命。
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── opencl.rs  # 同じ in-order queue 上の二 kernel と同期 read。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── linearize.rs  # 段間の RAW/WAR 順序。
│   ├── schedule.rs  # split_global_sum の二 kernel と中間 storage の依存。
│   ├── uop.rs  # LOCAL storage、BARRIER と効果依存。
│   └── verify.rs  # barrier が lane 条件外にあることの検証。
├── tests/
│   └── parallel_reduce.rs  # partial 値、schedule 依存、最終値。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

生成Cやshaderは一時ディレクトリへ出力するため、このRustプロジェクトのツリーには含めない。コード冒頭のコメントで、生成元と生成先を確認する。

## 小さなコミットで進める

### 1. LOCAL へ部分和を書き、lane 0 がまとめる

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── gpu_reduce_local.rs  # [追加] lane0 集約の予定実行例。
├── src/
│   ├── codegen/
│   │   ├── mod.rs  # [更新] reduction モジュールの宣言。
│   │   └── reduction.rs  # [追加] lower_group_sum と LOCAL 部分和の IR。
│   ├── renderer/
│   │   ├── metal.rs  # [更新] threadgroup と MSL barrier。
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── opencl.rs  # [更新] __local と OpenCL barrier。
│   ├── runtime/
│   │   ├── metal.rs  # [更新] local size と LOCAL byte の上限。
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── opencl.rs  # [更新] workgroup と LOCAL 容量の上限。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── linearize.rs  # [更新] STORE/BARRIER/LOAD の順序保持。
│   ├── uop.rs  # [更新] LOCAL storage、BARRIER と効果依存。
│   └── verify.rs  # [更新] LOCAL アクセスと全 lane の BARRIER 検証。
├── tests/
│   └── parallel_reduce.rs  # [追加] 部分和、同期構造と実機の逐次比較。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/codegen/mod.rs に mod reduction を宣言し、src/codegen/reduction.rs に lower_group_sum を置く。src/uop.rs の既存 storage/Op に LOCAL/BARRIER を追加し、verify/linearize で効果順と全 lane の参加を保つ。両 renderer に共有配列と barrier の表記を足し、両 runtime で L と L*4 byte を検査する。掲載 OpenCL は renderer の生成先である。

新しいモジュールは、既存の宣言を残して次を追記する。内部の補助関数は各モジュールに置き、exampleから使う入口だけを公開する。

```rust
// src/codegen/mod.rs（モジュール宣言を追加）
pub mod reduction;
```


このコミットは一つの行のSUMをgroup内の複数laneへ分け、結果の集約だけをlane0が行う。候補は共通GPU codegenの設計案 `lower_group_sum(row,N,L)`、LOCALのstorage型、BARRIERのverifierと両rendererである。GLOBALはdevice全体から見えるbuffer、LOCALは一workgroupだけの共有buffer、privateなaccは一laneだけの値として区別する。LOCALのallocationはgroupごとに別の実体を持ち、同じ添字0でも別groupの値とは共有されない。launchはgroups=rows,local=Lとし、row=group_id,lane=local_idを明示する。

各laneはacc=0から始め、col=lane,lane+L,lane+2L,…の有効な入力だけを読む。N=10,L=8ならlane0は1と9を足して10、lane1は2と10で12、lane2..7は一要素ずつ3..8を得る。全laneがLOCAL[lane]へ一回STOREした後、全laneが同じBARRIERを通る。BARRIERは全員の到着を待つ制御同期と、先行LOCAL書込みを後続読出しへ見せるメモリ同期を表す。この二つが揃って初めてlane0はLOCALの8要素を逐次加算でき、10+12+3+4+5+6+7+8=55をout[row]へ一回書く。

入力colのvalidが偽でも、lane自体はacc=0をLOCALへ保存してbarrierへ到着する。elementwiseで使った範囲外threadのreturnをここへ流用しない。IRではLOCAL STORE→BARRIER→LOCAL LOADの順を効果依存で結び、通常の純粋ALUの共有や並替えでbarrierを越えさせない。MSLはthreadgroup配列とmem_threadgroupのbarrier、OpenCLは__local配列とCLK_LOCAL_MEM_FENCEを出す。固定長Lのf32配列なら必要byte数はL*4である。

局所検証は55に加え、debug用GLOBAL出力へ部分和 `[10,12,3,4,5,6,7,8]` を保存して確認する。通常経路のSUMをhostで代行しない。deviceとkernelのthread上限、LOCAL容量、Lが2の冪かをdispatch前に検査する。sourceではbarrierが入力valid条件の外にあり、最終STOREだけがlane0条件の内にあることを両backendで確認する。


`src/codegen` の並列 SUM が出す最初の OpenCL kernel は、部分和だけを lane に分ける。固定 L=8、一 group が一行を担当し、groups=rows,local=8 で起動する。

組込み用の抜粋。実際の OpenCL C kernel。実機では未検証。

src/codegen/reduction.rs の部分和 IR を OpenCL renderer が出す kernel。

```opencl
// src/renderer/opencl.rs が生成するコード。生成先: <一時ディレクトリ>/sum_local.cl
__kernel void sum_local(__global const float *x,
                        __global float *out,
                        const uint columns) {
	__local float partial[8];
	uint row = get_group_id(0);
	uint lane = get_local_id(0);
	uint chunks = columns/8 + (columns%8!=0);
	float acc = 0.0f;
	for (uint chunk=0; chunk<chunks; chunk++) {
		uint col = chunk*8+lane;
		if (col < columns) {
			acc += x[row*columns+col];
		}
	}
	partial[lane] = acc;
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lane == 0) {
		float total = 0.0f;
		for (uint j=0; j<8; j++) { total += partial[j]; }
		out[row] = total;
	}
}
```

`acc` は private、`partial` は workgroup ごとに独立した LOCAL 配列である。入力を一つも読まない lane も0を書き、同じ barrier へ来る。barrier の後にだけ lane0 が他 lane の partial を読む。

同じ入力を追う例。

```text
columns=10,L=8
lane0: cols=[0,8], values=[1,9], acc=10
lane1: cols=[1,9], values=[2,10], acc=12
partial=[10,12,3,4,5,6,7,8]
barrier -> lane0 sums partial -> 55
```

LOCAL の32byteと local size=8を host で検証する。入力が短くても lane 単位の return はなく、barrier に全員が到着する。

- [ ] LOCAL storage と BARRIER を verifier、linearize、renderer に追加する。
- [ ] 各 lane が `col=chunk*L+lane` の部分和を逐次計算する。
- [ ] 全 lane が LOCAL の自分の位置に一つ書く。
- [ ] barrier 後だけ LOCAL を読み、lane 0 が一出力を書く。
- [ ] Metal と OpenCL の両方に同じ IR を下げる。

予定例 `gpu_reduce_local` は x=1..10、L=8 で 55 を返す。
LOCAL の部分和を debug 用出力へ書き出すと `[10,12,3,4,5,6,7,8]` になる。
debug 出力は検査時だけ使い、通常のアルゴリズムを host 側で代行しない。

### 2. 全 lane が barrier を通る tree を実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── gpu_reduce_tree.rs  # [追加] tree と逐次版の予定比較例。
├── src/
│   ├── codegen/
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── reduction.rs  # [更新] reduce_local_tree と段ごとの効果依存。
│   ├── renderer/
│   │   ├── metal.rs  # [更新] tree 版 MSL の生成。
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── opencl.rs  # [更新] 同じ tree の OpenCL C 生成。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── linearize.rs  # [更新] 段間の RAW/WAR 順序。
│   └── verify.rs  # [更新] barrier が lane 条件外にあることの検証。
├── tests/
│   └── parallel_reduce.rs  # [更新] 各段の prefix と L=1、反復実行。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/codegen/reduction.rs に reduce_local_tree を追加し、lane0 逐次集約だけを tree の IR へ置き換える。各段の barrier は lane 条件の外に生成する。src/verify.rs と src/linearize.rs の効果依存を更新し、掲載 sum_tree と同じ構造を両 renderer へ出す。


lane0の逐次集約を、共有配列の有効範囲を半分ずつ縮めるtree reductionへ置き換える。候補は並列SUM loweringの設計案 `reduce_local_tree(local,L,lane)` と、LOCAL効果依存の検証である。初期の部分和STOREと最初のBARRIERは前コミットを再利用する。次にstride=L/2から始め、stride>0の間、lane<strideのlaneだけが `local[lane]+local[lane+stride]` を読み、local[lane]へ保存する。その分岐が終わった位置で全laneがBARRIERを実行し、それからstrideを半分にして次の段へ進む。Lは2の冪に限定しているため、端に残る要素の特例は要らない。

初期LOCALが `[10,12,3,4,5,6,7,8]` のL=8を追う。stride=4ではlane0が10+5=15、lane1が12+6=18、lane2が3+7=10、lane3が4+8=12を先頭4位置へ保存する。全laneの同期後、stride=2で先頭は15+10=25と18+12=30になり、次の同期後、stride=1で25+30=55となる。最後のglobal STOREはlane0だけが実行する。lane4..7も加算はしないが、三段すべてのbarrierへ到着する。

同じ段では書込先 `[0,stride)` と右側読出し元 `[stride,2*stride)` が重ならず、各書込先は一laneが所有する。次の段が前の書込みを読むため、段間のbarrierを省けない。さらに次段の書込みが他laneの前段の読出しを追い越さないよう、read-after-writeだけでなくwrite-after-readの順序も保つ。LOCAL stateを古いSTORE以前へ戻してLOADを共有するrewriteは不正とする。barrierを `if(lane<stride)` 内に入れると参加数が変わり停止し得るので、verifierで構造を検査する。

局所検証は各段の有効な先頭だけをdebug出力し、上記の4,2,1要素列を比較する。L=1なら段loopがなくても初期部分和とlane0 STOREで正しいことを確認する。同じ入力を繰り返し実行し、偶然古いLOCAL値が残った一回の成功に依存しないかを調べる。逐次版との比較では浮動小数点の加算順が変わることを踏まえ、正確な整数小例と許容誤差を使う通常例を分ける。


tree 版の `src/renderer/metal.rs` の出力を示す。部分和の作り方は同じで、stride ごとに加算する lane を半減し、全 lane の同期を分岐の外に置く。

組込み用の抜粋。実際の MSL kernel。実機では未検証。

共通 tree IR を Metal renderer が出す MSL。

```metal
// src/renderer/metal.rs が生成するコード。生成先: <一時ディレクトリ>/sum_tree.metal
#include <metal_stdlib>
using namespace metal;
kernel void sum_tree(device const float *x [[buffer(0)]],
                     device float *out [[buffer(1)]],
                     constant uint &columns [[buffer(2)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint lane [[thread_position_in_threadgroup]]) {
	threadgroup float partial[8];
	float acc = 0.0f;
	uint chunks = columns/8+(columns%8!=0);
	for (uint chunk=0; chunk<chunks; chunk++) {
		uint col = chunk*8+lane;
		if (col<columns) { acc += x[row*columns+col]; }
	}
	partial[lane] = acc;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for (uint stride=4; stride>0; stride/=2) {
		if (lane<stride) {
			partial[lane] += partial[lane+stride];
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
	if (lane==0) { out[row] = partial[0]; }
}
```

同じ段の書込先は `[0,stride)`、右側の読出し元は `[stride,2*stride)` であり、異なる lane の書込先が重ならない。次の段の読出しは前段の STORE に依存するため、全員の barrier を一段ごとに残す。

同じ入力を追う例。

```text
initial prefix=[10,12,3,4,5,6,7,8]
stride4: active prefix=[15,18,10,12], all 8 lanes barrier
stride2: active prefix=[25,30], all 8 lanes barrier
stride1: active prefix=[55], all 8 lanes barrier
lane0 writes out[row]=55
```

lane4〜7 は初期 STORE の後に加算しなくても、全ての barrier へ参加する。GPU 実機の検証時はこの列を debug 出力し、逐次版と値を比較する。

- [ ] stride を L/2、L/4、…、1 として LOCAL の二要素を足す。
- [ ] `lane<stride` の分岐では加算と LOCAL STORE だけを行う。
- [ ] 各 stride の後の barrier は全 lane が実行する位置に置く。
- [ ] 最終結果の global STORE は lane 0 だけに限定する。
- [ ] LOCAL の read-after-write と write-after-read の依存を保つ。

予定例 `gpu_reduce_tree` は同じ x で 55 を返す。
L=8 の最初の加算後の有効な先頭 4 要素は `[15,18,10,12]` になる。
次は `[25,30]`、最後は `[55]` になり、逐次版と照合できる。
生成 shader で barrier が lane 条件の外側にあることも確認する。

### 3. 端数、複数行、LOCAL の再利用を実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── gpu_reduce_rows.rs  # [追加] 一 group 一行の予定実行例。
├── src/
│   ├── codegen/
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── reduction.rs  # [更新] partial_sum の guard と LOCAL 再利用の末尾 barrier。
│   ├── renderer/
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── opencl.rs  # [更新] 二反復の LOCAL 再利用テスト用 shader の出力。
│   ├── runtime/
│   │   ├── metal.rs  # [更新] N=0/rows=0 と合法 local の分岐。
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── opencl.rs  # [更新] 空出力とゼロ単位元、上限検査。
│   └── lib.rs  # 本体のモジュール宣言と公開APIの入口。
├── tests/
│   └── parallel_reduce.rs  # [更新] 端数、複数行、二反復の効果依存。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/codegen/reduction.rs の partial_sum を端数と複数行へ広げ、LOCAL 再利用の専用テストでは次反復の STORE 前に barrier を追加する。掲載 sum_two_rows は src/renderer/opencl.rs のテスト向け生成出力として扱い、通常の一 group 一行の割当は保つ。両 runtime の起動前に N=0 のゼロ出力と rows=0 の空出力を処理する。


ここではtreeの構造を変えず、入力の端数と複数groupの所有範囲を明確にする。候補は並列reductionの入力guard、launch検証、LOCALを再利用するloopの効果依存である。設計案の `partial_sum(row,lane,N,L)` は `chunks=N/L+(N%L!=0)` 回だけ進み、col=chunk*L+laneがN未満のときだけGLOBAL LOADを行う。colが無効なlaneもacc=0からの計算を終え、LOCALへ書いて同じtreeとbarrierを通る。guardを入力のvalidと出力rowのvalidへ分け、barrierの手前でlane単位に終了しない。

2×3入力でL=8ならgroup0のLOCALは `[1,2,3,0,0,0,0,0]`、group1は `[4,5,6,0,0,0,0,0]` となる。同じLOCAL添字を使ってもgroup間で配列は独立し、結果はout[0]=6,out[1]=15である。N=9ならlane0だけがcol0と8を読み、他laneは各一要素となる。N=257,L=8のonesではlane0が33要素、他laneが32要素を担当し、合計257になる。chunk数が異なるlaneを作るのではなく、全laneに同じchunk loopと有効LOAD条件を持たせれば同期の構造が分かりやすい。

一つのgroupがloopで複数の行やtileへ同じLOCAL配列を再利用する設計へ進む場合は、前のLOCALを読む全laneが終わってから次のSTOREを始めるbarrierも必要になる。最初のread前の同期だけでは、速いlaneが次の反復で配列を上書きし、遅いlaneが古い値を読む途中に壊す可能性がある。初版はgroup一行の割当を保ち、この再利用を行う専用小例だけで反復末尾のbarrierを検証する。rowが無効ならgroup全体が同じ判断で処理を省くか、正確なgroups数で起動し、laneごとの分岐と混同しない。

N=0のSUMはhostでrows個の0を作り、rows=0なら空出力とする。ゼロサイズdispatchや未初期化出力を返さない。局所テストはN=1,L-1,L,L+1,257、複数行、LOCAL再利用の二反復を逐次版と比較する。非2冪L、容量超過、負の設定は明示的に拒否し、barrierの実行回数が入力validやlane IDで変わらないsourceを確認する。


LOCAL 配列を二反復で再利用する場合を `src/codegen` の効果依存テストにする。次は一 group が二行を処理する専用 OpenCL 小例で、実運用の初版は一 group 一行のままとする。

組込み用の抜粋。実際の OpenCL C kernel。実機では未検証。

LOCAL 再利用の効果依存を検査する二反復 kernel の生成例。

```opencl
// src/renderer/opencl.rs が生成するコード。生成先: <一時ディレクトリ>/sum_two_rows.cl
__kernel void sum_two_rows(__global const float *x,
                           __global float *out,
                           const uint columns) {
	__local float partial[8];
	uint lane = get_local_id(0);
	uint first_row = get_group_id(0)*2;
	for (uint iteration=0; iteration<2; iteration++) {
		uint row = first_row+iteration;
		float acc = 0.0f;
		uint chunks = columns/8+(columns%8!=0);
		for (uint chunk=0; chunk<chunks; chunk++) {
			uint col = chunk*8+lane;
			if (col<columns) { acc += x[row*columns+col]; }
		}
		partial[lane] = acc;
		barrier(CLK_LOCAL_MEM_FENCE);
		for (uint stride=4; stride>0; stride/=2) {
			if (lane<stride) { partial[lane] += partial[lane+stride]; }
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		if (lane==0) { out[row] = partial[0]; }
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}
```

入力行数は `2*groups`、local は8とする。最後の barrier は lane0 の partial[0] 読出しが終わるまで次の iteration の STORE を止める。tree 最後の barrier だけでは、その後の読出しと次の書込みの間を同期できない。

同じ入力を追う例。

```text
columns=3,group=0
iteration0 partial=[1,2,3,0,0,0,0,0] -> out[0]=6
end barrier: all reads complete before reuse
iteration1 partial=[4,5,6,0,0,0,0,0] -> out[1]=15
columns=0: production host returns zeros without dispatch
```

全 lane の chunk 数と iteration 数は同じで、入力の有効性で変化しない。N=0 の通常経路は host でゼロを用意し、この専用再利用テストの kernel を呼ぶ必要はない。

- [ ] N=1、L-1、L、L+1、257 の部分和を同じ規則で作る。
- [ ] 二次元の行和では group ID を row として使う。
- [ ] input の無効 LOAD と output の無効 STORE を別々に guard する。
- [ ] LOCAL を同じ kernel のループで再利用する場合、次の書込み前に barrier を入れる。
- [ ] 非 2 冪の local を明示的に拒否し、無言で誤った tree を作らない。

予定例 `gpu_reduce_rows` は `[[1,2,3],[4,5,6]]` に L=8 を使って `[6,15]` を返す。
各行が ones(257) の 3 行入力なら `[257,257,257]` になる。
N=0 の SUM は 06 までの契約どおり単位元 0 を返す。
host 側で出力のゼロ初期化を行い、ゼロサイズ dispatch を発行しない。

### 4. group をまたぐ SUM は二つの kernel で実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_18_group_sum_reference.rs  # 本文のコードを配置する。
│   └── gpu_reduce_multipass.rs  # [追加] 二段 SUM の予定実行例。
├── src/
│   ├── codegen/
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── reduction.rs  # [更新] chunk 部分和と二段目 SUM の lowering。
│   ├── renderer/
│   │   ├── metal.rs  # [更新] 同じ chunk 分割の MSL 出力。
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── opencl.rs  # [更新] chunk_sum の OpenCL C 出力。
│   ├── runtime/
│   │   ├── metal.rs  # [更新] 二 kernel の順序と中間 buffer の寿命。
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── opencl.rs  # [更新] 同じ in-order queue 上の二 kernel と同期 read。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── schedule.rs  # [更新] split_global_sum の二 kernel と中間 storage の依存。
├── tests/
│   └── parallel_reduce.rs  # [更新] partial 値、schedule 依存、最終値。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/schedule.rs に split_global_sum を追加し、K1 の partial 書込み後に K2 の読出しを置く。src/codegen/reduction.rs は各段の SUM を共通 lowering へ渡し、両 runtime は中間 allocation を K2 完了まで保持する。掲載 chunk_sum は renderer が出す shader、掲載 Rust の group_sum/main は book_18_group_sum_reference の数値参照である。


workgroupのbarrierは別groupの完了を待たないため、大きなSUMを二つのkernelへ分ける。候補はscheduleの中間bufferと依存管理、並列reductionの設計案 `split_global_sum(input,chunk)` である。第1kernelは固定chunkごとに一groupを割り当て、各group内は前コミットまでの部分和とtreeを実行する。group gの有効範囲は `[g*chunk,min((g+1)*chunk,N))` で、一つの結果をGLOBALの中間buffer partial[g]へ書く。第2kernelはこのpartialを通常のPARAMとして読み、同じSUM loweringで一つの最終値へ縮約する。

ones(1000)、chunk=256ならgroup数は4で、担当区間は `[0,256),[256,512),[512,768),[768,1000)` となる。第1kernelの出力は `[256,256,256,232]` で、第2kernelはそれを足して1000を返す。例えばL=8なら最初のgroupの各laneは32要素を担当し、最後のgroupの各laneは29要素を担当する。ここでは二段目に入る要素数4が一groupの処理範囲へ収まる設定を選び、無制限な再帰分割や自動探索は追加しない。

scheduleにはK1のwrite(partial)、K2のread(partial)とwrite(out)を明示し、K2をK1後の状態のpartialへbindする。partialのStorageIdは一つだが、K1以前の未初期化状態とK1完了後の状態を混同しない。Planのallocation表は中間bufferを両kernelの間で生存させ、K2が読み終えるまで別出力の領域へ再利用しない。入力は読取専用、新規partialと最終outは別allocationにする。

同じin-order queueへK1、K2を順に投入すれば、hostでpartialを読み返して再転送する必要はない。Metalではcommandとencoderの順序が依存を保証する形でencodeし、OpenCLではin-order queueの順を使う。最後のread完了まで全bufferと実行物を保持する。局所検証はschedule dumpの二kernelと依存edge、debug時のpartial4要素、最終1000をそれぞれ確認する。第1kernelに全GPU同期のようなbarrierやspin待ちを置かない。N=0や一chunkだけの入力も既存の単位元と単一group経路へ収束させ、GPU実機ごとに反復実行で比較する。


`src/schedule.rs` が作る第1 kernel の chunk 範囲を明示する。次の OpenCL C は固定 chunk=256,local=8 の部分和で、第2 kernel は同じ SUM lowering を partial に対して使う。

組込み用の抜粋。実際の OpenCL C kernel。実機では未検証。

schedule が作る第1段の IR を OpenCL C に出す例。

```opencl
// src/renderer/opencl.rs が生成するコード。生成先: <一時ディレクトリ>/chunk_sum.cl
__kernel void chunk_sum(__global const float *x,
                        __global float *partial_out,
                        const uint n) {
	__local float partial[8];
	uint lane = get_local_id(0);
	uint group = get_group_id(0);
	uint begin = group*256;
	uint remaining = n-begin;
	uint length = min((uint)256,remaining);
	float acc = 0.0f;
	for (uint offset=lane; offset<length; offset+=8) {
		acc += x[begin+offset];
	}
	partial[lane] = acc;
	barrier(CLK_LOCAL_MEM_FENCE);
	for (uint stride=4; stride>0; stride/=2) {
		if (lane<stride) { partial[lane] += partial[lane+stride]; }
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (lane==0) { partial_out[group] = partial[0]; }
}
```

groups=ceil(n/256) を正確に起動するので begin<n が全 group で成立する。`remaining` から length を切り、最後の group だけ短い chunk を読む。group ごとの一出力は新規 GLOBAL storage へ書く。

同じ入力を追う例。

```text
N=1000 -> groups=4,chunk=256,local=8
K1 ranges=[0,256),[256,512),[512,768),[768,1000)
K1 writes partial@state1=[256,256,256,232]
K2 binds partial@state1 -> SUM -> out=1000
schedule edge: K1 write(partial@state1) before K2 read(partial@state1)
```

同じ StorageId でも未初期化の state0 を K2 へ bind しない。中間 allocation は K2 の読出し完了まで保持し、二つの kernel は同じ in-order queue または明示依存を持つ command へ投入する。


単独で実行できる例。GPU の各段を std の配列で照合する参照器であり、実機上の同期を検証するものではない。

各段の配列値を照合する host 参照器全体。GPU 同期の実装として貼らない。

```rust
// examples/book_18_group_sum_reference.rs（単独で実行する例）
fn group_sum(x: &[f32], local: usize) -> f32 {
	assert!(local.is_power_of_two());
	let mut partial = vec![0.0;local];
	let chunks = x.len()/local+usize::from(x.len()%local!=0);
	for (lane,acc) in partial.iter_mut().enumerate() {
		for chunk in 0..chunks {
			let i = chunk*local+lane;
			if i<x.len() { *acc += x[i]; }
		}
	}
	let mut stride = local/2;
	while stride>0 {
		for lane in 0..stride { partial[lane] += partial[lane+stride]; }
		stride /= 2;
	}
	partial[0]
}
fn main() {
	let x = vec![1.0;1000];
	let partial: Vec<_> = x.chunks(256).map(|chunk|group_sum(chunk,8)).collect();
	assert_eq!(partial,[256.,256.,256.,232.]);
	assert_eq!(group_sum(&partial,8),1000.);
	for n in [0,1,7,8,9,257] { assert_eq!(group_sum(&vec![1.0;n],8),n as f32); }
	assert_eq!(group_sum(&[1.,2.,3.,4.,5.,6.,7.,8.,9.,10.],8),55.);
}
```

`partial` を新しい Vec に確保してから二段目へ渡すため、段の順序と値を個別に検査できる。GPU 版はこの host の加算を実行するのではなく、上の実 kernel を二つの起動として同じ依存関係へ置く。

- [ ] 大きな入力を固定 chunk に分け、各 group の部分和を GLOBAL に保存する。
- [ ] 部分和バッファを次 kernel の PARAM slot に bind する。
- [ ] 二つの kernel の依存を schedule に明示する。
- [ ] 同じ queue 上の順序と最後の読出し完了を runtime で保証する。
- [ ] 単一 workgroup の barrier を全 GPU の barrier として扱わない。

予定例 `gpu_reduce_multipass` は ones(1000)、chunk=256 で部分和 `[256,256,256,232]` を作る。
第 2 kernel の出力は 1000 になる。
第 1 kernel から未完了の部分和を host 経由で先読みせず、Plan の依存でつなぐ。

## 完了確認

次の example と `tests/parallel_reduce.rs` は本ステップで追加する予定である。
各 backend の実機確認を独立して記録する。

```sh
# 作業ディレクトリ: chibigrad/
cargo test
cargo run --example gpu_reduce_local -- --backend metal --local 8 --dump-source
cargo run --example gpu_reduce_tree -- --backend opencl --local 8 --compare serial
cargo run --example gpu_reduce_rows -- --backend metal --cols 257 --dump-kernel
cargo run --example gpu_reduce_multipass -- --backend opencl --size 1000 --chunk 256 --dump-schedule
```

- [ ] 整数として正確に表せる小例は、期待値と厳密に一致する。
- [ ] 通常の f32 は加算順序の差を考慮し、`1e-5+1e-5*abs(reference)` を初期目標にする。
- [ ] 正負が相殺する入力と大小が混在する入力も比較する。
- [ ] barrier を通る回数が lane の ID と入力の有効性で変わらない。
- [ ] 同じ入力を繰り返して偶然の一回の成功に依存していないことを確認する。
- [ ] 実機未実施の backend を数値一致済みとして報告しない。

## tinygrad ソースとの対応

参照 commit は `c22edff14c79310a9b2df62a78819cf77c8289fa` である。

| 実在するシンボル | ソース | 役割 | chibigrad での簡略化 |
|---|---|---|---|
| `fix_group_for_reduce` | [codegen/__init__.py](../../tinygrad/tinygrad/codegen/__init__.py#L157) | LOCAL 軸の部分和を LOCAL buffer と最終縮約へ分ける | SUM と固定 L の二段階から始める |
| `lower_reduce` | [codegen/__init__.py](../../tinygrad/tinygrad/codegen/__init__.py#L192) | reduction を累積用 REG とループに下げる | 各 lane の部分和の生成に使う考え方を採る |
| `add_local_buffer` | [codegen/__init__.py](../../tinygrad/tinygrad/codegen/__init__.py#L227) | LOCAL 保存先と lane ごとの添字を作る | 一つの固定長配列に限る |
| `add_raw_barrier` と `add_war_barrier` | [codegen/__init__.py](../../tinygrad/tinygrad/codegen/__init__.py#L245) | LOCAL の RAW とループ内 WAR に同期を加える | 初版は構造化された明示 barrier にする |
| `group_gpudims` | [gpudims.py](../../tinygrad/tinygrad/codegen/gpudims.py#L44) | hardware 軸を割り当て重複出力を mask する | row=group、lane=local に固定する |
| `OpenCLRenderer.barrier` と `MetalRenderer.barrier` | [OpenCL](../../tinygrad/tinygrad/renderer/cstyle.py#L339)、[Metal](../../tinygrad/tinygrad/renderer/cstyle.py#L377) | backend ごとの LOCAL memory barrier の表記 | 同じ BARRIER の意味を保つ |

## 今は追加しないもの

atomic、warp shuffle、任意 local size の tree、自動 reduction 分割は後の拡張にする。
tiled GEMM と conv の共有メモリは [19](19-fast-gemm-conv.md) で別々に検証する。
flash attention は [23](23-flash-attention.md) でオンライン softmax と合わせて導入する。
