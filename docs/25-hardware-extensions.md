# 25. Metal の行列命令を使い、論文または hardware の独立課題へ進む

## このステップで動かすもの

同じ GEMM を19の通常積和と Metal の simdgroup matrix 経路で動かし、数値を比較する。
[24](24-llm-training.md)までの学習例を回帰確認として残す。
行列命令の実装自体は19の GEMM と15の Metal runtime に依存する。
元計画第10章の最適化を拡張し、初版で保留した tensor core 相当の経路を具体化する。
最後に、学習者が選ぶ論文実装または hardware 対応を、再現できる独立課題としてまとめる。
本文の capability、IR、example、test、コマンドは今後実装する予定である。
GPU 実行と性能は未確認であり、source を生成できるだけで実機検証済みとはしない。

最初の予定例は `8x8` の identity 行列 I と、`A[i,j]=8*i+j` の行列である。
`I@A=A` と、`A@I=A` を確認して operand と lane の配置を切り分ける。
次に K=16として二つの縮約 tile を計算し、前の部分和が次へ渡ることを確認する。

## 最小実装とデータ

最初の対象は Metal の `simdgroup_float8x8` と `simdgroup_multiply_accumulate` を使う経路である。
Metal の行列 API 名を、NVIDIA の tensor core と同じ物理ユニット名だとは説明しない。
入力 IR は積と ADD reduction による GEMM、出力 IR は capability で選ばれた行列 tile 演算と LOAD、STORE である。
予定の IR は `MatrixTile { m, n, k, input_dtype, accum_dtype }` を一つ持つ。
まず f32 入力と f32 累積で動かし、F16 入力は独立した後続コミットにする。
CPU と標準の OpenCL 経路は19の通常積和を使う。
OpenCL を有効にしただけで tensor core を使えることは前提にしない。

capability は backend 名だけで決めず、device、OS、compiler、対応型、simdgroup 幅、資源制限から作る。
小さい検査 kernel のコンパイルと実行も成功してから、対応経路を選ぶ。
通常積和への fallback を常に保持し、非対応条件は理由付きで表示する。

## この章のファイル配置

MatrixTileの認識とloweringはsrc/schedule.rs、MSLの生成はsrc/renderer/metal.rs、probeと引数設定はsrc/runtime/metal.rsへ置く。scalarのmain付き例はexamples、比較用helperはtestsへ分け、shader本文はRustファイルへ貼らない。

次はこの章の終わりまでに触る関連ファイルの配置案である。現在のcheckoutの一覧ではない。各コミットでは、その下に示す部分だけを追加・変更する。

```text
chibigrad/
├── examples/
│   ├── book_25_matrix_fallback.rs  # 本文のコードを配置する。
│   ├── book_25_norm_reference.rs  # 本文のコードを配置する。
│   ├── matrix_capabilities.rs  # capability表示と既存GEMMの実行。
│   ├── matrix_gemm.rs  # dtype、比較、計測区間の選択。
│   ├── matrix_tile.rs  # identityとbasisの実機比較driver。
│   └── student_norm.rs  # 選んだRMSNorm課題のbaselineと拡張driver。
├── src/
│   ├── renderer/
│   │   ├── metal.rs  # half fragmentとfloat accumulatorの生成。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── runtime/
│   │   ├── metal.rs  # dtype組ごとのprobe。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── schedule.rs  # RMSNormを選んだ場合の一つのfusion変換。
├── tests/
│   ├── matrix_tiles.rs  # half_referenceによる量子化済みbytesの比較。
│   └── student_norm.rs  # 合成版、拡張、未対応strideの比較。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

生成Cやshaderは一時ディレクトリへ出力するため、このRustプロジェクトのツリーには含めない。コード冒頭のコメントで、生成元と生成先を確認する。

## 小さなコミットで進める

### 25a. capability と fallback を実際の GEMM に接続する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_25_matrix_fallback.rs  # 本文のコードを配置する。
│   └── matrix_capabilities.rs  # [追加] capability表示と既存GEMMの実行。
├── src/
│   ├── runtime/
│   │   ├── metal.rs  # [更新] probe_matrix_capabilityと実機検査結果。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── schedule.rs  # [更新] select_gemm_pathと通常GEMMへのfallback。
├── tests/
│   └── matrix_tiles.rs  # [追加] 強制無効、shape条件とfallbackの比較。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/runtime/metal.rsにprobe_matrix_capabilityを追加し、既存buffer bind、dispatch、wait、readbackでI@Iを検査する。src/schedule.rsの選択にselect_gemm_pathを加える。掲載Capability、select_matrix、ordinary、mainはexamples/book_25_matrix_fallback.rsの単独例へ置き、実際のfallbackは19章のGEMMを使う。


入力はprobe結果とM/N/K、出力はmatrix経路か通常GEMMである。単独で実行できる例は、実機probeがまだない場合に必ず通常積和へ戻ることと、その結果を確認する。`src/runtime/metal.rs` のprobeはI@Iの読出しまで成功した時だけ `executed_ok=true` を渡す。以下のfalseは実機能力の測定値ではない。

Capabilityとmainを含むfallbackの数値確認例。実機probeと本体GEMMの代替として使わない。

```rust
// examples/book_25_matrix_fallback.rs（単独で実行する例）
struct Capability { compiled_ok: bool, executed_ok: bool, simd_width: usize }
fn select_matrix(c: &Capability, m: usize, n: usize, k: usize, force_off: bool)
	-> (&'static str,&'static str) {
	if force_off { return ("ordinary","forced fallback"); }
	if !c.compiled_ok || !c.executed_ok { return ("ordinary","probe incomplete"); }
	if c.simd_width!=32 { return ("ordinary","unsupported SIMD width"); }
	if m%8!=0 || n%8!=0 || k%8!=0 { return ("ordinary","tile tails"); }
	("matrix8","supported")
}
fn ordinary(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
	assert_eq!(a.len(),m*k); assert_eq!(b.len(),k*n);
	let mut out=vec![0.;m*n];
	for i in 0..m {
		for j in 0..n {
			for r in 0..k { out[i*n+j]+=a[i*k+r]*b[r*n+j]; }
		}
	}
	out
}
fn main() {
	let cap=Capability{compiled_ok:false,executed_ok:false,simd_width:32};
	assert_eq!(select_matrix(&cap,2,2,3,false).0,"ordinary");
	let result=ordinary(&[1.,2.,3.,4.,5.,6.],&[1.,2.,3.,4.,5.,6.],2,2,3);
	assert_eq!(result,[22.,28.,49.,64.]);
	assert_eq!(select_matrix(&cap,8,8,8,true).1,"forced fallback");
	println!("fallback={result:?}");
}
```

`compiled_ok` と `executed_ok` を分けることで、source生成だけで選択しない。`simd_width=32` はこの実装の起動条件で、全Metal deviceの一般則として埋め込まない。実際のcapabilityはdevice/OS/compiler、local資源、対応型も保持する。Planのdispatcherはこの返値に従って19の通常kernelまたは25bのmatrix kernelを起動し、どちらも既存ABIと同期経路を使う。

```text
probe: compile shader -> pipeline -> dispatch I@I -> wait -> read64 values -> validate
compile success, runtime unknown -> ordinary, reason=probe incomplete
force_off=true -> ordinary even if probe passed
M=9,N=7,K=17 before guarded tiles -> ordinary, reason=tile tails
```

19の通常 GEMM を残したまま、`src/runtime/metal.rs` に `probe_matrix_capability`、選択側に `select_gemm_path` を足す案とする。capability は使えるという真偽値だけでなく、deviceとOS、compiler識別、実行幅、local資源、対応する入力と累積の型、probeの成否と理由を保持する。APIから取得した上限を確認し、必要な型と命令を含む小shaderをcompileしてpipelineを作り、実際に起動して読出した値も検査する。compile成功だけの段階と実行まで成功した段階を区別する。

probeの最小計算は8x8のI@Iで、対角が1、その他が0になることを期待する。buffer bind、group設定、完了待ち、CPU読出しは15のMetal runtimeを再利用する。probeが失敗したら理由を保存し、同じ利用者入力の通常GEMMへ戻す。test用の強制無効化も同じ選択関数を通し、fallback専用の別計算器を作らない。CPUと標準OpenCLは初めから通常経路を選び、backend名からmatrix対応を推測しない。

19の小入力をcapability有効と強制無効の両方で渡し、対応実機ならどちらも[[22,28],[49,64]]になることを比べる。ただし形状がtile条件を満たさない場合も理由付きfallbackが正しい。選択dumpにはcapability結果とshape条件を別々に示し、「deviceは対応するが今回shapeは非対応」を判別できるようにする。cacheはprobe対象のdevice、compiler、型、ABIを含め、別deviceの成功記録を流用しない。probeを実行できない環境では通常GEMMの数値だけを確認し、行列命令を未確認として残す。これで情報表示から既存Planの実行までつながり、次のtile実装を追加する場所が固定される。

- [ ] Metal device の family、API が返す実行幅、threadgroup 上限、local byte 上限を取得する。
- [ ] 必要な行列型と演算を持つ小 shader をコンパイルし、失敗理由を capability に保存する。
- [ ] capability が使えない場合も19の f32 GEMM を実行し、同じ出力を得る。
- [ ] capability を test 用に無効にし、通常経路を強制して確認する。
- [ ] cache key に device capability、型、compiler と ABI の識別を含める。

予定例 `matrix_capabilities --backend metal --run-fallback` を追加する。
使えない実機では fallback の数値成功と行列命令の未確認を別々に報告する。
このコミットは情報表示だけで終えず、既存 GEMM の起動と読出しまで通す。
新しい kernel module や独自 PCI driver は導入しない。

### 25b. 一つの行列 tile を generated shader で計算する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── matrix_tile.rs  # [追加] identityとbasisの実機比較driver。
├── src/
│   ├── renderer/
│   │   ├── metal.rs  # [更新] 公開simdgroup load、積和、storeの描画。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── runtime/
│   │   ├── metal.rs  # [更新] 32lane、一groupで既存pipelineを起動する。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── schedule.rs  # [更新] 一つのMatrixTileを持つkernel記述。
├── tests/
│   └── matrix_tiles.rs  # [更新] pipeline識別と全64要素の確認。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/schedule.rsのkernel用記述にMatrixTileを追加し、src/renderer/metal.rsで掲載matrix8のMSLを生成する。src/runtime/metal.rsはそのsourceのpipelineを既存経路で起動する。掲載MSLをRustファイル本文へ貼らず、公開load、multiply、storeを出すrenderer処理を実装する。


入力はrow-majorのA/B各64要素、出力は8×8のCである。以下は `src/renderer/metal.rs` が出すMSL kernelの実コードを示す組込み用の抜粋。`extra/gemm/metal_matmul.py` の `simdgroup_load`、`simdgroup_store` の4引数形とaccumulator初期化を縮小した。MSLのcompileと実機起動はこの文書では未確認である。

一tileのMatrixTileを描いたMSL出力。既存Metal runtimeがcompile、dispatchする。

```metal
// src/renderer/metal.rs が生成するコード。生成先: <一時ディレクトリ>/matrix8.metal
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void matrix8(device const float *a [[buffer(0)]],
                    device const float *b [[buffer(1)]],
                    device float *c [[buffer(2)]]) {
	simdgroup_float8x8 left;
	simdgroup_float8x8 right;
	simdgroup_float8x8 acc(0);
	simdgroup_load(left,a,8,ulong2(0,0));
	simdgroup_load(right,b,8,ulong2(0,0));
	simdgroup_multiply_accumulate(acc,left,right,acc);
	simdgroup_store(acc,c,8,ulong2(0,0));
}
```

hostは一つのthreadgroup、一つの32lane simdgroupだけを起動し、capabilityで幅32と集団APIの実行を確認する。leading dimensionの8はbyteでなく要素数である。全laneが同じload、multiply、storeを呼ぶ。fragmentのlane内配置を推測して `acc` の要素へ直接書かず、公開load/storeへ任せる。

```text
A=I, B[r,c]=8*r+c -> C[2,3]=19; then swap A/B -> same result
basis: A[2,5]=1, B[5,3]=7, others0 -> only C[2,3]=7
dispatch groups=(1,1,1), local=(32,1,1), source contains matrix8
log pipeline identifier from this source + read back all64 results
```

`src/renderer/metal.rs` に一つの `MatrixTile` を公開simdgroup load、積和、storeへ描く処理を追加する案である。最初のkernelは一つのsimdgroupが8x8の出力全体を担当し、capabilityが示す実行幅に合うlane数で起動する。入力AとBはrow-majorの連続64要素、leading dimensionは8要素と定義する。byte幅と要素strideを混同しない。matrix accumulatorを0に初期化し、A/Bをloadして積和を一回行い、全体を出力bufferへstoreする。

fragmentは集団で持つ行列tileの一部分であり、各laneが通常の配列の何行目を持つかを推測しない。公開load/storeを使う限り、その配置をrendererの外へ公開する必要はない。内部要素へ直接アクセスする変更が必要になった場合だけ、laneと要素番号から論理座標(r,c)への対応表を作り、全64座標が重複なく被覆されるかを検査する。別deviceのfragment表を名前だけで共有しない。

I@Aなら出力[2,3]はI[2,2]*A[2,3]=19であり、A@Iも同じ19になる。片方だけ合う場合はAとBの転置指定やloadのleading dimensionのどちらかが違う可能性を調べる。より細かいbasis testではAの(2,5)だけを1、Bの(5,3)だけを7にすると、出力(2,3)だけが7になる。要素の行列位置とoperandの対応をこの検査で特定する。通常積和とf64 referenceにも同じbytesを渡し、全64要素を比べる。sourceの命令名を確認した上で、そのsource由来のpipeline識別を実行ログへ残し、通常版を誤って起動しても通るtestにしない。新しいruntimeやABIを作らず、既存Metalの入力転送から完了待ちまでをそのまま通す。

- [ ] まず公開の simdgroup load と store を使い、8x8の tile を一つだけ計算する。
- [ ] 行列の row/column 順、leading dimension、転置指定を入力 layout と照合する。
- [ ] shader で I@A と A@I を計算し、19の通常積和と CPU reference に比較する。
- [ ] 行列の一箇所だけが1の basis 入力で、出力座標の対応を検証する。
- [ ] shader を既存の Metal compiler、buffer bind、dispatch、同期、読出しの経路で動かす。

予定例 `matrix_tile --backend metal --case identity --shape 8,8,8` を追加する。
lane が保持する fragment を直接扱う必要が出た時だけ、lane と要素から行列座標への表を追加する。
最初から全 device の fragment を表す汎用 DSL は作らない。
source に行列演算が含まれ、実行時にその pipeline を起動したことの両方を記録する。

### 25c. GEMM の認識と K 累積を compiler に入れる

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── matrix_gemm.rs  # [追加] 倍数条件と端数条件の実機driver。
├── src/
│   ├── renderer/
│   │   ├── metal.rs  # [更新] K累積と端数を含むmatrix_tailの生成。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── runtime/
│   │   ├── metal.rs  # [更新] 資源、uint範囲、16byteのshape引数pack。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── schedule.rs  # [更新] match_matrix_gemmとlower_matrix_tiles。
├── tests/
│   └── matrix_tiles.rs  # [更新] 複数K tileと9×7×17の比較。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/schedule.rsへmatch_matrix_gemmとlower_matrix_tilesを加え、純粋graphを残してkernelだけをMatrixTileへ下げる。src/renderer/metal.rsでK累積と協調搬入を描く。src/runtime/metal.rsの引数packは既存scalar slotsのM、N、Kからu32の4要素配列を作り、16byteをshape引数へ渡す。端数対応前は同じ選択入口から通常GEMMへ戻す。


入力は通常GEMMの認識結果とA/B、出力は端数をゼロ詰めした8×8のC tileである。`lower_matrix_tiles` はA[i,k]×B[k,j]のADD reductionとdtype/stride/alignmentを確認した時だけ選ぶ。次は端数対応後のMSL kernelを示す組込み用の抜粋。simdgroup一つ、32lane、local合計768byteの起動条件をhostで検査する。実機未確認である。

lower_matrix_tilesから出す端数対応MSL。hostのshape packはruntime/metal.rsへ実装する。

```metal
// src/renderer/metal.rs が生成するコード。生成先: <一時ディレクトリ>/matrix_tail.metal
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void matrix_tail(device const float *a [[buffer(0)]],
                        device const float *b [[buffer(1)]],
                        device float *c [[buffer(2)]],
                        constant uint3 &shape [[buffer(3)]],
                        uint3 group [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
	uint M=shape.x,N=shape.y,K=shape.z;
	threadgroup float at[64],bt[64],ct[64];
	simdgroup_float8x8 acc(0),left,right;
	for (uint ko=0;ko<K;ko+=8) {
		for (uint z=lane;z<64;z+=32) {
			uint ar=group.y*8+z/8,ak=ko+z%8;
			uint bk=ko+z/8,bc=group.x*8+z%8;
			at[z]=(ar<M && ak<K)?a[ar*K+ak]:0.0f;
			bt[z]=(bk<K && bc<N)?b[bk*N+bc]:0.0f;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		simdgroup_load(left,at,8,ulong2(0,0));
		simdgroup_load(right,bt,8,ulong2(0,0));
		simdgroup_multiply_accumulate(acc,left,right,acc);
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
	simdgroup_store(acc,ct,8,ulong2(0,0));
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for (uint z=lane;z<64;z+=32) {
		uint row=group.y*8+z/8,col=group.x*8+z%8;
		if (row<M && col<N) c[row*N+col]=ct[z];
	}
}
```

`acc` はkoより外側なのでK=16なら部分和8から16へ進む。`at,bt` の各要素は毎tile必ず更新し、無効LOADは条件分岐の未選択側となる。集団storeは全laneでctへ行い、その後だけglobal STOREをguardする。hostはshapeの積、ko加算と添字のuint範囲も検査する。buffers/scalarsの論理slotは既存runtimeがbackend引数0〜3へpackする。`shape` のuint3は16byte alignmentを持つので、hostは論理scalar slots 0〜2をu32の `[M,N,K,0]` へpackし、`setBytes(length=16,index=3)` 相当でbuffer引数3へ渡す。12byteの配列をそのまま渡さない。

```text
M=9,N=7,K=17, A=B=ones -> groups=(1,2,1), local=(32,1,1)
K tiles: ko=0 acc=8; ko=8 acc=16; ko=16 acc=17
all63 valid outputs=17, not24; invalid output coordinates never stored
logical graph retains MUL+REDUCE; only kernel lowering contains MatrixTile
```

手書きtileの次に、`src/schedule.rs` の `match_matrix_gemm` と `lower_matrix_tiles` を追加する案とする。recognizerはreduction軸kに対しAが(i,k)、Bが(k,j)を読む積で、他に未対応の演算やmaskが挟まっていないことを調べる。shape、dtype、stride、alignment、capabilityを確認し、成立した時だけMatrixTile{m:8,n:8,k:8,...}へ落とす。不成立なら元の通常GEMMのloweringを使い、純粋graphを書き換えたり微分規則へmatrix命令を追加したりしない。

ループは出力tileごとのacc初期化、K tileの順次loadとmatrix積和、最終storeの順にする。K=16で全入力1なら最初のtile後のaccは8、次は旧accを渡して16になる。毎回0のaccを渡すと結果8なので、この入力で累積の受渡しを検査できる。K=24では24となる。同じsimdgroupの全laneが集団演算を実行し、一部laneだけが命令を飛ばす条件分岐は生成しない。

最初の認識範囲はM/N/Kが8の倍数に限り、9x7x17の例は理由付きで通常版を選ぶ。端数用の次の変更では、A/Bを8x8のlocal tileへ協調搬入し、無効global座標はLOADせず0を保存する。全laneのbarrier後にtileを集団loadして積和し、再利用前も同期する。出力も一度localへ集団storeした後、laneごとの有効座標だけをglobalへ保存すれば、集団store自体を部分laneの分岐へ入れずに済む。全入力1の9x7x17なら全63出力が17であり、paddingを含め24にならないことを通常版と照合する。IRにaccの寿命、localのpitch、二つのguardと同期を出し、fallbackと端数対応後の両段階を別々に確認する。

- [ ] 既存の `sum(A[i,k]*B[k,j], k)` の形と dtype を認識する。
- [ ] MatrixTile を生成し、対応しない stride、型、alignment は通常版へ戻す。
- [ ] K=16、24で部分和を次の行列演算の accumulator に渡す。
- [ ] M/N/K が8の倍数でない場合は、まず通常版へ戻して数値を保つ。
- [ ] 次のコミットで端数を local tile にゼロ詰めし、無効 LOAD と STORE を guard する。

予定例 `matrix_gemm --backend metal --shape 9,7,17 --dump-kernel --dump-source` を追加する。
端数実装前は fallback、実装後は guarded tile を選び、同じ結果を返す。
一つの simdgroup 内の集団演算を lane ごとの条件分岐の片側だけで実行しない。
行列命令は lowering の最適化なので、純粋な元 graph と13の自動微分規則は保つ。

### 25d. dtype と測定を一条件ずつ広げる

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── matrix_gemm.rs  # [更新] dtype、比較、計測区間の選択。
├── src/
│   ├── renderer/
│   │   ├── metal.rs  # [更新] half fragmentとfloat accumulatorの生成。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── runtime/
│   │   ├── metal.rs  # [更新] dtype組ごとのprobe。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── schedule.rs  # [更新] MatrixTileのF16入力とF32累積条件。
├── tests/
│   └── matrix_tiles.rs  # [更新] half_referenceによる量子化済みbytesの比較。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

MatrixTileの入力dtypeをsrc/schedule.rsで広げ、Metal rendererとruntimeのprobeにF16、F32の組を追加する。掲載half_referenceはtests/matrix_tiles.rsのscalar helperへ置き、21eのsrc/state.rsのf16_to_f32を公開する必要最小限のAPIから参照する。MSLの抜粋はmetal rendererの生成部分であり、量子化前のf32値を比較先に使わない。


入力はF16へ量子化した実byte列、出力はその値に対する通常積和のreferenceとmatrix出力の比較である。まず型を切り替える組込み用MSLの核を示す。25b/25cのload、同期、tail guardはそのまま使い、probeがこの型の組を受理して実行できた場合だけ有効にする。以下もMSL実機未確認である。

F16入力とF32累積を描くMSLの生成部分。25bまたは25cのkernel内へ出力する。

```metal
// src/renderer/metal.rs が生成するコード。生成先: <一時ディレクトリ>/matrix_gemm_f16.metal
/* 組込み用。a,bはdevice const half*、cはdevice float*。
   8x8 row-major、全32laneが同じ制御で呼ぶ。 */
simdgroup_half8x8 left;
simdgroup_half8x8 right;
simdgroup_float8x8 acc(0);
for (uint ko=0;ko<K;ko+=8) {
	simdgroup_load(left,a+row_start*K+ko,K,ulong2(0,0));
	simdgroup_load(right,b+ko*N+col_start,N,ulong2(0,0));
	simdgroup_multiply_accumulate(acc,left,right,acc);
}
simdgroup_store(acc,c+row_start*N+col_start,N,ulong2(0,0));
```

`left/right` はF16、`acc` はF32である。この抜粋はM/N/Kが8の倍数の内側tileに限り、端数なら25cのhalf local tileへ搬入する。混合型APIをcompilerが受理しない場合はcapability失敗として通常版を選ぶ。次の組込み用Rust抜粋は21eの `f16_to_f32` を再利用し、量子化後の値からf64 referenceを作る。

実際に転送するF16 bit列のscalar比較helperへ追加する。f16_to_f32は21eのstateモジュールを参照する。

```rust
// tests/matrix_tiles.rs（既存ファイルへ組み込む抜粋）
fn half_reference(a: &[u16], b: &[u16], m: usize, n: usize, k: usize) -> Vec<f64> {
	assert_eq!(a.len(),m*k); assert_eq!(b.len(),k*n);
	let mut out=vec![0.;m*n];
	for i in 0..m {
		for j in 0..n {
			let mut acc=0.0f64;
			for r in 0..k {
				let av=f16_to_f32(a[i*k+r]) as f64;
				let bv=f16_to_f32(b[r*n+j]) as f64;
				acc+=av*bv;
			}
			out[i*n+j]=acc;
		}
	}
	out
}
```

`a,b` はGPUへ渡したものと同じ16bit列であり、量子化前のf32を比較先にしない。速度を測る前に全要素の誤差を確認する。コンパイル、転送、kernel、同期を分け、測定するkernel時間と同じ仕事のFLOPsを対応させる。

```text
F16 A=[0x3c00,0xbc00] -> [1,-1], B=[0x3800,0x3800] -> [0.5,0.5]
f64 reference dot=0.5-0.5=0
M=N=K=64 -> useful FLOPs=2*64^3=524288, FLOP/s未測定
API generated+executed != physical unit utilization observed by profiler
```

25bと25cのf32経路を基準に、`MatrixTile` の入力dtypeとMetal rendererのload型をF16へ広げる案である。capabilityはF16入力とf32累積の組を個別に検査し、f32が使えたという理由だけで許可しない。入力のf32値を一度F16 bytesへ変換し、その実bytesを命令版と通常版に渡す。referenceは元f32値ではなく、F16を復号した値をf64へ上げて積和する。これで入力量子化と累積・演算順の差を分けて評価できる。

最初はF16で正確に表せる1、-1、1/2を使う。例えばAの一行が[1,-1]、Bの一列が[1/2,1/2]なら二つの積は1/2と-1/2で和0になる。K tileの境界をまたいで正負を配置したtestも作り、前tileのaccの消失や符号の配置を検査する。量子化が必要な入力では元値、F16の16bit表現、復号後の値を小fixtureへ残す。誤差が大きい時は量子化済みreferenceとの差から調べ、閾値を広げる前にstride、転置、K累積、overflowを切り分ける。

経路の比較は同じshape、dtype、入力bytes、出力読出しで行い、通常版、matrix版、強制fallbackをそれぞれログへ出す。測定を始める時はcompileと初回pipeline作成をcold区間に分け、warmup後のkernelをdevice側の完了が分かる方法で計る。転送、同期、host全体の時間も別に残し、各反復の結果が期待値と一致することを確認する。M=N=K=64なら有効仕事は2*64*64*64=524288 FLOPsだが、時間を測る前にFLOP/sは出さない。公開APIを生成して起動した記録と、特定の物理unitの稼働をprofilerで観測した記録は別にする。未観測のunit利用率や速度の数字は追加しない。

- [ ] capability が許す場合だけ、F16 入力と f32 累積を追加する。
- [ ] 変換後の実 F16 入力値を f64 へ上げた reference と比較し、入力量子化の差を分ける。
- [ ] 負値、ゼロ、打消し、大きさの異なる値、K の複数 tile を確認する。
- [ ] 通常版、行列命令版、fallback の同一入力を比べ、選択経路を記録する。
- [ ] 正しさの確認後に compile、転送、kernel、同期、全体の計測を分ける。

予定例 `matrix_gemm --backend metal --shape 64,64,64 --dtype f16-f32 --bench` を追加する。
速度を測る時は結果を読み、必要な計算が省略されていないことも確認する。
MSL の行列 API を使った事実と、物理ユニットの稼働を profiler で確認した事実は区別する。
後者を確認していない場合に、特定 hardware ユニットの利用率や性能を主張しない。

### 25e. 一つの独立課題を同じ検証経路で完成させる

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_25_norm_reference.rs  # 本文のコードを配置する。
│   └── student_norm.rs  # [追加] 選んだRMSNorm課題のbaselineと拡張driver。
├── src/
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── schedule.rs  # [更新] RMSNormを選んだ場合の一つのfusion変換。
├── tests/
│   └── student_norm.rs  # [追加] 合成版、拡張、未対応strideの比較。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

ここでは候補からnorm融合を選ぶ配置例とし、examples/student_norm.rsへ本体Planと新kernelを比較するdriverを追加する。掲載norm_reference、compare、mainはexamples/book_25_norm_reference.rsの単独確認例へ分ける。融合の条件と変換はsrc/schedule.rsへ一つだけ加え、tests/student_norm.rsで既存Tensor合成版と比較する。他の候補を選ぶ場合はそのexampleとtestだけを作り、未選択の空moduleを先に増やさない。


入力はxとnorm weight、出力は既存合成式と融合kernelの比較結果である。独立課題をnorm一つに絞った場合のscalar基準を、単独で実行できる例として示す。`examples/student_norm.rs` では同じfixtureを既存Planと新kernelへ渡す。ここで確認するのは数式と境界であり、新device実機の成功ではない。

norm_reference、compare、mainを含む数式確認の独立例。student_normは別のcompiler経路の比較driverにする。

```rust
// examples/book_25_norm_reference.rs（単独で実行する例）
fn norm_reference(x: &[f64], weight: &[f64], eps: f64) -> Result<Vec<f64>, &'static str> {
	if x.is_empty() || x.len()!=weight.len() { return Err("shape"); }
	if eps<=0. || !eps.is_finite() { return Err("epsilon"); }
	if x.iter().chain(weight).any(|v|!v.is_finite()) { return Err("input"); }
	let mut squares=0.;
	for &v in x { squares+=v*v; }
	let denominator=(squares/x.len() as f64+eps).sqrt();
	if !denominator.is_finite() { return Err("overflow"); }
	let mut out=Vec::with_capacity(x.len());
	for i in 0..x.len() { out.push(x[i]/denominator*weight[i]); }
	Ok(out)
}
fn compare(reference: &[f64], actual: &[f64]) -> bool {
	reference.len()==actual.len() && reference.iter().zip(actual)
		.all(|(r,a)|a.is_finite() && (r-a).abs()<=1e-5+1e-4*r.abs())
}
fn main() {
	let reference=norm_reference(&[1.,-1.],&[1.,1.],1.).unwrap();
	let expected=[1./2f64.sqrt(),-1./2f64.sqrt()];
	assert!(compare(&reference,&expected));
	assert!(norm_reference(&[1.],&[1.,1.],1.).is_err());
	assert!(norm_reference(&[0.],&[1.],0.).is_err());
	println!("norm={reference:?}");
}
```

`squares/len` は最後の軸の二乗平均、`denominator` はその値にepsを加えた平方根である。fusion版は同じ式を一つのkernelで行い、未対応strideなら既存合成版へ戻す。必要なgradientは13の合成版を比較先にし、forward一致だけで学習可能としない。独立課題の仕様、入力、選択理由、誤差、再現コマンドは以下の記録形式に揃える。

```text
operation=RMSNorm, dtype=f32, last_dim=2, eps=1 (hand-computed fixture)
input=[1,-1],weight=[1,1] -> expected=[0.70710678,-0.70710678]
contiguous supported -> extension; unsupported stride -> ordinary composite graph
record: commit,device,compiler,command,input,max_error,path,unsupported_scope
GPU source generation only -> real-device status=unverified
```

独立課題は既存の比較先を一つ持つ小さな仕様から始める。`examples/student_norm.rs` 等と対応testを作る案とし、対象の演算、入力dtypeとshape、対応するmaskやgradient、出力の意味、fallback条件をそのexampleに近い場所へ記録する。論文名だけを仕様にせず、今回実装する数式と範囲を切り出す。まず拡張なしの既存Planが同じ入出力で動くことを確かめ、その次にcompilerの一変換またはbackendの一経路だけを差し替える。課題のために元の共通graphや二配列ABIを作り直さない。

normを選ぶなら既存の `x*rsqrt(mean(x*x)+eps)*weight` を比較先にできる。小入力x=[1,-1]、weight=[1,1]、eps=1なら二乗平均は1、分母はsqrt(2)、出力は[1/sqrt(2),-1/sqrt(2)]となる。eps=1は手計算用fixtureの値であり、Qwenのconfigを変更する意味ではない。次に固定乱数の入力、dimの端数、ゼロ、非連続viewを比較し、必要なら同じlossを通した入力とweightのgradientも照合する。対応できないstride等では既存版へ戻ることを確認する。

attention backwardを選ぶ場合はforwardのOだけで終えず、同じ小lossに対するdQ/dK/dVとGQA共有headの寄与和、24の一回の更新まで比較する。新deviceなら加算、縮約、GEMMをCPUの同じPlanと比べ、実機を使っていない段階はsourceまたはsimulator確認として分ける。どの課題も仕様を満たす入力と、前提を一つ破る入力を用意し、正しい値と拒否またはfallbackの両方を検査する。最終記録は対象commit、deviceとcompiler、再現コマンド、実入力、誤差、選択経路、未対応範囲を含める。一つを完了してから次の候補へ進み、全候補の空moduleや未検証の最適化を先に追加しない。

- [ ] 論文の演算一つ、compiler 変換一つ、または新しい device 一つを対象として選ぶ。
- [ ] 入力、出力、数式、比較用の既存実装、対応 dtype と shape を短い仕様にする。
- [ ] 最初に既存 pipeline の baseline example を動かし、同じ入出力で拡張を差し替える。
- [ ] 小入力の期待値、ランダム比較、境界値、必要なら gradient 比較を追加する。
- [ ] device と commit、再現コマンド、結果、未対応範囲を project の記録へ残す。

予定の課題と対応は次とする。全候補を一度に実装せず、一つを選んで完了させる。

| 独立課題 | 既存の比較先 | 最小の実行結果 |
|---|---|---|
| 論文中の norm または位置変換の融合 | 12と22の Tensor 合成版 | 同じ前向き出力と、必要な入力・重みの勾配 |
| attention の backward 再計算 | 23の通常 attention と24の学習 | dQ/dK/dV、GQA 勾配、同じ一回の更新 |
| 新しい compiler 変換 | 最適化を無効にした IR | 同じ値と、変換条件を破る入力での拒否または fallback |
| 新しい accelerator backend | CPU の同じ Plan | 転送、加算、縮約、GEMM の順に実機で同じ値 |

予定例名を `student_norm`、`student_attention_backward`、`student_transform`、`student_device` とする。
論文の全実験を再現したとする前に、実際に対応した演算と条件を明記する。
未所有の hardware は source と simulator の確認までとし、実機成功には数えない。

## 完了確認

以下はこの章で作る予定の example と test のコマンドである。

```sh
# 作業ディレクトリ: chibigrad/
cargo test
cargo test --test matrix_tiles
cargo run --example matrix_capabilities -- --backend metal --run-fallback
cargo run --example matrix_tile -- --backend metal --case identity --shape 8,8,8
cargo run --example matrix_gemm -- --backend metal --shape 9,7,17 --dump-source
cargo run --example matrix_gemm -- --backend metal --shape 64,64,64 --dtype f16-f32 --bench
cargo run --example student_norm -- --case small --compare-reference
```

最後のコマンドは norm を選んだ場合だけ作成する予定の例である。他の選択では対応する example を作る。

- [ ] identity と basis の期待値に一致し、K の部分和が失われない。
- [ ] f32 は `abs_err <= 1e-4 + 1e-4*abs(ref)` を初期基準として全要素を比較する。
- [ ] F16 入力と f32 累積は、量子化済み入力 reference に `abs_err <= 1e-3 + 1e-3*abs(ref)` で比較する。
- [ ] 許容値が満たせない場合は誤差の原因と対象範囲を調べ、無条件で閾値を広げない。
- [ ] capability 無効、非対応型、不正 shape、端数のケースで安全に fallback またはエラーになる。
- [ ] 実行経路と数値の両方を確認し、source 内の命令名だけで完了にしない。
- [ ] 小 MLP と LLM の一回の更新を回帰確認し、lowering の変更が勾配の意味を変えない。
- [ ] 独立課題は baseline と拡張を同じコマンドで比較でき、未確認の実機を明示する。

## tinygrad ソースとの対応

参照 commit は `c22edff14c79310a9b2df62a78819cf77c8289fa` である。

| 実在シンボルと相対リンク | tinygrad の役割 | chibigrad で簡略化する点 |
|---|---|---|
| [`TensorCore.frag_coords`](../../tinygrad/tinygrad/renderer/tc.py#L23) | lane/element と tile 座標を対応させる | まず一つの Metal tile に限定する |
| [`MetalRenderer.render_kernel`](../../tinygrad/tinygrad/renderer/cstyle.py#L400) | simdgroup matrix の積和 wrapper を出す | 公開 load/store の小 kernel から始める |
| [`Scheduler._apply_tc_opt`](../../tinygrad/tinygrad/codegen/opt/postrange.py#L164) | GEMM の形と型を認識し WMMA へ変換する | 条件を狭くして fallback を残す |
| [`MetalDevice.__init__`](../../tinygrad/tinygrad/runtime/ops_metal.py#L200) | device family を取得して target を設定する | family に加えて shader の実行も検査する |
| [`helper_tc_allclose`](../../tinygrad/test/runtime/test_tensor_cores.py#L62) | 命令の存在と実行後の数値を確認する | 同じ入力で通常版と f64 reference も比較する |
| [`prog` の生成](../../tinygrad/extra/gemm/metal_matmul.py#L29) | simdgroup load、積和、store の実験例 | 大きな固定設定を移植せず8x8から通す |

`tc.metal` の dtype 候補は capability の証拠そのものではない。
実機の利用可否と、現在の compiler がその型を受理することを個別に確認する。

## 今は追加しないもの

全 vendor 共通の行列命令、FP8、複数 GPU、自動探索、独自 driver は必須にしない。
CUDA や AMD の行列命令を選ぶ課題では、その device 固有の公開経路と fragment を別に実装する。
OpenCL の vendor 拡張を使う場合も、標準経路への fallback と対応 device の検出を付ける。
