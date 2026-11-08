# 15. Metal の公開 API で一つの出力を一つの thread に割り当てる

## このステップで動かすもの

[14 の CPU 学習](14-sgd-mnist.md) までのグラフを、まず小さい配列から Metal で実行する。
元計画の第 9 章「Metal、次に OpenCL」の Metal 部分に対応する。
生成 MSL のコンパイル、buffer の準備、起動、完了待ち、読出しまでつなぐ。

```text
x=[1,2,3,4], y=[10,20,30,40]
x+y = [11,22,33,44]
ones(257)+2 = [3 を257個]
row_sum([[1,2,3],[4,5,6]]) = [6,15]
[[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
```

GPU 上では出力一要素を一つの thread が担当する。
reduction の内側は逐次ループのまま残す。
以下は実装予定の API と検証手順であり、Metal 実機での確認済み結果ではない。
利用できる機器がない場合はソース生成の確認と実機確認を分けて記録する。

## 最小実装とデータ

CPU と共通の kernel IR から、外側の出力 RANGE を hardware ID に変換する。
launch 情報は `Launch { groups, local }` とし、どちらも三次元の整数にする。
MSL では group と local の ID を使い `i=group_id.x*local_size.x+local_id.x` を作る。
`thread_position_in_grid` を直接使う案でもよいが、host 側の寸法の意味を統一する。

```text
入力IR: RANGE(i,257), LOAD(x,i), ADD(2), STORE(out,i)
GPU IR: GROUP_ID, LOCAL_ID, i=g*L+l, IF(i<257), LOAD/ADD/STORE
launch: local=[64,1,1], groups=[5,1,1]
実際のthread数: 320、最後の63 threadはLOADもSTOREもしない
```

MSL の buffer 引数は `device const float*` と `device float*` から始める。
PARAM の slot 順に bind し、offset は最初は 0 に固定する。
実装予定の runtime は公開 Metal API だけで組み立てる。
device、command queue、source からの library、function、compute pipeline を作る。
`newLibraryWithSource` 相当の公開経路で MSL をコンパイルする。
command buffer と compute command encoder に buffer と pipeline を設定する。
`dispatchThreadgroups(groups, threadsPerThreadgroup=local)` を呼び、commit 後に完了を待つ。
初版は shared storage の buffer に限定し、読出し前の完了待ちを必須にする。
device と pipeline、buffer の寿命を未完了の command buffer より長く保つ。

参照版の private compiler service と HCQ2 のホスト UOp 化は移植しない。
MSL の source、build error、launch 寸法を独立して表示できるようにする。
local size は pipeline と device の上限に照らし合わせ、使える値を選ぶ。
64 を全機器で使える前提にはしない。

## この章のファイル配置

共通 kernel IR の GPU 軸割当を src/codegen/gpu.rs に加え、MSL の出力は src/renderer/metal.rs、公開 API と所有権は src/runtime/metal.rs に置く。掲載 MSL は一時出力であり、Rust の本体へ直接貼らない。

次はこの章の終わりまでに触る関連ファイルの配置案である。現在のcheckoutの一覧ではない。各コミットでは、その下に示す部分だけを追加・変更する。

```text
chibigrad/
├── examples/
│   ├── book_15_metal_launch.rs  # 本文のコードを配置する。
│   ├── metal_add.rs  # Tensor から Metal の加算を動かす予定例。
│   ├── metal_movement.rs  # broadcast と pad の予定実行例。
│   ├── metal_reduce_matmul.rs  # 逐次縮約と matmul の予定実行例。
│   └── metal_tail.rs  # N=257 の予定実行例。
├── src/
│   ├── codegen/
│   │   ├── gpu.rs  # 出力軸だけを thread ID へ割り当てる map_output_axes。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── renderer/
│   │   ├── metal.rs  # private acc と逐次 RANGE/END の MSL。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── runtime/
│   │   ├── metal.rs  # validate_launch と機器/pipeline 上限、空出力。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── rangeify.rs  # 既存 movement の座標と valid 条件。
├── tests/
│   └── backend_metal.rs  # 行和、matmul、MLP 勾配と再bind。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

生成Cやshaderは一時ディレクトリへ出力するため、このRustプロジェクトのツリーには含めない。コード冒頭のコメントで、生成元と生成先を確認する。

## 小さなコミットで進める

### 1. 加算を source compile から読出しまで通す

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── metal_add.rs  # [追加] Tensor から Metal の加算を動かす予定例。
├── src/
│   ├── codegen/
│   │   ├── gpu.rs  # [追加] 出力 RANGE を hardware ID と guard に変換。
│   │   └── mod.rs  # [更新] gpu モジュールの宣言。
│   ├── renderer/
│   │   ├── metal.rs  # [追加] 共通 IR からの MSL と引数列。
│   │   └── mod.rs  # [更新] metal モジュールの宣言。
│   ├── runtime/
│   │   ├── metal.rs  # [追加] MetalProgram、BindingLayout、buffer の所有と同期。
│   │   └── mod.rs  # [更新] Metal runtime の条件付き宣言。
│   └── lib.rs  # 本体のモジュール宣言と公開APIの入口。
├── tests/
│   └── backend_metal.rs  # [追加] 生成 source と条件付き実機比較。
└── Cargo.toml  # [更新] 選んだ Metal binding の任意依存と feature。
```

src/codegen/mod.rs で mod gpu、src/renderer/mod.rs で mod metal を宣言する。src/runtime/mod.rs の metal 宣言と Cargo.toml の binding 依存は対応 OS/feature に限定する。src/codegen/gpu.rs は GPU lowering と共通 Launch/BindingLayout の定義を持ち、src/runtime/metal.rs がそれを参照して MetalProgram と run を実装する。掲載 MSL は src/renderer/metal.rs の生成目標、Objective-C 風 text は runtime の呼出し契約として読む。

新しいモジュールは、既存の宣言を残して次を追記する。内部の補助関数は各モジュールに置き、exampleから使う入口だけを公開する。

```rust
// src/codegen/mod.rs（モジュール宣言を追加）
pub mod gpu;
// src/renderer/mod.rs（モジュール宣言を追加）
pub mod metal;
// src/runtime/mod.rs（モジュール宣言を追加）
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metal;
```

CPUだけのビルドがGPU runtimeを要求しないよう、既存の `Cargo.toml` の `[features]` に次のキーを追加する。テーブルがまだなければ作り、既にあればその中へキーだけを加える。bindingを選んだ際は、そのoptional依存もこのfeatureから有効にする。

```toml
# Cargo.toml（既存のfeaturesへ統合する）
[features]
metal = []
```

この章以降で metal の予定コマンドを実行するときは、`cargo run` または `cargo test` に `--features metal` を追加する。通常のCPUテストはfeatureを付けずに行う。

GPU専用のintegration testも、次をファイルの先頭に置いて同じ条件で有効にする。

```rust
// tests/backend_metal.rs（ファイル先頭へ追加）
#![cfg(all(target_os = "macos", feature = "metal"))]
```

`cargo test` はexampleもビルドするため、このGPU専用exampleにはCargoの必要featureも設定する。次の項目を既存の `Cargo.toml` へ一度だけ追加する。

```toml
# Cargo.toml（このexampleの登録）
[[example]]
name = "metal_add"
path = "examples/metal_add.rs"
required-features = ["metal"]
```

このexampleのMetalを使う既存のmainには `#[cfg(target_os = "macos")]` を付ける。Metal専用のuseと補助関数にも同じcfgを付けるか、それらをcfg付きモジュールへまとめる。macOS以外でも `--all-features` のビルド時にMetal APIを参照しないよう、次のmainを同じファイルへ追加する。

```rust
// examples/metal_add.rs（非macOS用の入口を追加）
#[cfg(not(target_os = "macos"))]
fn main() {
	eprintln!("Metal requires macOS");
	std::process::exit(1);
}
```


このコミットは共通kernel IRの一番外側の出力軸だけをGPUのthreadへ割り当て、runtimeの公開API経路を一巡させる。候補は `src/renderer/metal.rs` と `src/runtime/metal.rs` である。設計案の `MetalProgram` がdevice、queue、library、function、pipelineを保持し、`run(buffers,scalars,launch)` が今回の引数と出力を結び付ける。CPUの `chibi_entry` の二配列ABIは変更しない。GPUでも呼出し側はbuffersとscalarsを分け、`BindingLayout` の予定表がbuffer slotとscalar slotを別々にMSLの引数indexへ対応させる。scalarがあれば下位32bitから元の型へ戻してsetBytes相当で渡し、ポインタ列へ混ぜない。

初期化は `MTLCreateSystemDefaultDevice`、`newCommandQueue` の順とし、生成したMSLを `newLibraryWithSource:options:error:` でcompileする。次に `newFunctionWithName:`、`newComputePipelineStateWithFunction:error:` で実行物を得る。入力と新規出力は `newBufferWithLength:options:` でshared storageとして確保し、入力のcontentsへ4byte単位のf32を書き込む。起動時はqueueからcommandBuffer、その中にcomputeCommandEncoderを作り、pipelineと各bufferを `setBuffer:offset:atIndex:` で設定する。dispatch後にendEncoding、commit、waitUntilCompletedの順で完了させ、statusとerrorを見てから出力contentsを読む。

長さ4、local=4、groups=1の加算ならthread2がxの3とyの30を読み、33を出力2へ書く。四つの出力は `[11,22,33,44]` である。入力引数は `device const float*`、出力は可変pointerとし、同じshapeの別入力を別のMetal bufferへbindする。出力を入力bufferへ上書きしない。

FFIのunsafeはruntimeに閉じ、CPUメモリのsliceを作るときはbyte長、alignment、待機完了を検査する。Plan、bufferとcommand bufferの所有物は待機が終わるまで保持する。局所テストはCPU比較に加え、不正MSLのbuild診断と実行時command errorを別に確認する。手書きshaderだけを動かすテストで終えず、Tensor→schedule→共通IR→MSLの経路で同じ期待値を得ることを実機で確認する。


`src/renderer/metal.rs` が作る加算 kernel を先に固定する。buffer 引数0,1が入力、2が出力、scalar slot0の N は binding 表で引数3へ写す。

組込み用の抜粋。実際の MSL kernel。実機では未検証。

共通 GPU IR を MSL にした加算 kernel。Metal runtime はこの source をコンパイルする。

```metal
// src/renderer/metal.rs が生成するコード。生成先: <一時ディレクトリ>/add_f32.metal
#include <metal_stdlib>
using namespace metal;
kernel void add_f32(device const float *x [[buffer(0)]],
                    device const float *y [[buffer(1)]],
                    device float *out [[buffer(2)]],
                    constant uint &n [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
	if (i < n) {
		float a = x[i];
		float b = y[i];
		out[i] = a + b;
	}
}
```

`device const` は入力を変更しない契約を表す。`n` はポインタ配列へ埋め込まず、独立した scalar binding から setBytes で渡す。thread ID が範囲内のときだけ三つのメモリアクセスを行う。

同じ入力を追う例。

```text
inputs x=[1,2,3,4],y=[10,20,30,40],N=4
thread i=2 -> a=3,b=30 -> out[2]=33
BindingLayout buffers=[0->0,1->1,2->2], scalars=[0->3]
```

二配列 ABI の論理 slot と MSL 引数 index は同じものではない。対応表を runtime に保持して毎回 bind する。


外部APIの疑似コード。Objective-C Metal 公開 API の呼出し順であり、Rust binding は未選定である。各 error と null を直後に検査し、全オブジェクトを read 完了まで保持する。

Metal 公開 API の接続順の疑似コード。選んだ Rust binding の呼出しへ置換し、各失敗と寿命を扱う。

```text
// src/runtime/metal.rs（外部APIの疑似コード）
device = MTLCreateSystemDefaultDevice()
queue = [device newCommandQueue]
library = [device newLibraryWithSource:source options:nil error:&error]
function = [library newFunctionWithName:@"add_f32"]
pipeline = [device newComputePipelineStateWithFunction:function error:&error]
x = [device newBufferWithLength:16 options:MTLResourceStorageModeShared]
y = [device newBufferWithLength:16 options:MTLResourceStorageModeShared]
out = [device newBufferWithLength:16 options:MTLResourceStorageModeShared]
memcpy([x contents], input_x, 16); memcpy([y contents], input_y, 16)
command = [queue commandBuffer]
encoder = [command computeCommandEncoder]
[encoder setComputePipelineState:pipeline]
[encoder setBuffer:x offset:0 atIndex:0]
[encoder setBuffer:y offset:0 atIndex:1]
[encoder setBuffer:out offset:0 atIndex:2]
[encoder setBytes:&N length:4 atIndex:3]
[encoder dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(4,1,1)]
[encoder endEncoding]; [command commit]; [command waitUntilCompleted]
if command.status != MTLCommandBufferStatusCompleted: return command.error
memcpy(host_output, [out contents], 16)
```

library と pipeline はコンパイル済み Plan に保持できる。x,y,out と N の binding は各 run で設定する。最後の memcpy は command の完了と成功の確認後にだけ行う。この疑似コード自体も実機では未検証である。

- [ ] 利用可能な Metal device を検出し、未対応時の明示的エラーを作る。
- [ ] 静的 f32 の加算用 MSL を共通 kernel IR から生成する。
- [ ] 公開 API で library と pipeline を作り、build error を返す。
- [ ] shared buffer の入力を書き、4 thread を dispatch する。
- [ ] 完了待ち後に出力を読み、CPU の値と比較する。

予定例 `metal_add` は `[11,22,33,44]` を返す。
source の構文誤りを意図的に渡す runtime テストでは、MSL の診断を取得する。
このコミットから実際の Tensor graph と schedule を通し、手書き shader 専用経路にしない。

### 2. group と local を分けて端数を実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_15_metal_launch.rs  # 本文のコードを配置する。
│   └── metal_tail.rs  # [追加] N=257 の予定実行例。
├── src/
│   ├── codegen/
│   │   ├── gpu.rs  # [更新] group/local ID と出力 guard の lowering。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── runtime/
│   │   ├── metal.rs  # [更新] validate_launch と機器/pipeline 上限、空出力。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   └── lib.rs  # 本体のモジュール宣言と公開APIの入口。
├── tests/
│   └── backend_metal.rs  # [更新] 端数と再bind、空入力。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/runtime/metal.rs に validate_launch を足し、整数の検証結果を Metal の MTLSize へ変換する。掲載 launch/main は book_15_metal_launch へ保存し、実際の device/pipeline 上限検査を runtime に実装する。src/codegen/gpu.rs の ID 生成で i=g*L+l と guard を作り、encode ごとに引数を bind する。

`cargo test` はexampleもビルドするため、このGPU専用exampleにはCargoの必要featureも設定する。次の項目を既存の `Cargo.toml` へ一度だけ追加する。

```toml
# Cargo.toml（このexampleの登録）
[[example]]
name = "metal_tail"
path = "examples/metal_tail.rs"
required-features = ["metal"]
```

このexampleのMetalを使う既存のmainには `#[cfg(target_os = "macos")]` を付ける。Metal専用のuseと補助関数にも同じcfgを付けるか、それらをcfg付きモジュールへまとめる。macOS以外でも `--all-features` のビルド時にMetal APIを参照しないよう、次のmainを同じファイルへ追加する。

```rust
// examples/metal_tail.rs（非macOS用の入口を追加）
#[cfg(not(target_os = "macos"))]
fn main() {
	eprintln!("Metal requires macOS");
	std::process::exit(1);
}
```


この変更では論理出力数Nと実際に起動するthread数を分離する。候補はGPU軸のloweringとMetal runtimeの設計案 `validate_launch(N,requested_local,pipeline)` である。localをLとし、N>0ならgroupsを `N/L+(N%L!=0)` で求める。この式なら `N+L-1` 自体のoverflowを避けられる。deviceの各次元のthread上限、pipelineのmaxTotalThreadsPerThreadgroup、全次元の積を照合してからdispatchする。実行幅の推奨値は速さの参考であり、必須の倍数という条件と取り違えない。L=0と積のoverflowはAPIを呼ぶ前の入力エラーにする。

MSLでは `threadgroup_position_in_grid` と `thread_position_in_threadgroup` から `i=g*L+l` を作り、`i<N` のときだけLOADとSTOREを行う。N=257,L=64ならgroups=5、総数320で、group4のlane0だけがi=256を担当し、lane1..63は入力も出力も読まない。onesの位置256から1を読み2を加えるので最後の値も3となる。この時点はBARRIERがないため無効threadのreturnを使えるが、将来の並列reductionへ同じreturnをコピーしない。

N=0ではホストで空の出力を確定し、ゼロbyteのMetal buffer確保やゼロgroupのdispatchを必要としない経路にする。N>0の入力はrunごとにBindingLayoutから実際のbufferを引き直す。同じshapeの別配列へ差し替えた後に、前回のsetBufferだけを再利用すると古い入力を読むため、encodeのたびにbindする。compile済みpipelineと現在のbufferの寿命は別に管理する。

局所検証はN=1,L-1,L,L+1,257でCPUと一致させ、guardを外した場合に番兵領域が壊れる条件をsourceでも確認する。入力shapeと実bufferの要素数の照合を先に行い、GPUのfaultをbounds検証の代用にしない。予定exampleのdumpにはN、groups、local、実thread数を別々に表示する。再bindした二回目の実行とN=0のAPI未呼出しを確認すれば、Plan再利用と端数処理が既存の入力契約のまま成立したと分かる。


`src/runtime/metal.rs` の launch 寸法は Metal の型へ変換する前に整数だけで検証できる。次は一次元版で、0要素では None を返して API 呼出しを省く。

単独で実行できる例。

機器 API を呼ばない整数の launch 検証と main 全体。

```rust
// examples/book_15_metal_launch.rs（単独で実行する例）
fn launch(n: usize, local: usize, max_threads: usize) -> Result<Option<(usize,usize)>,String> {
	if local==0 || local>max_threads { return Err("invalid local size".into()); }
	if n==0 { return Ok(None); }
	let groups = n/local + usize::from(n%local!=0);
	let total = groups.checked_mul(local).ok_or("thread count overflow")?;
	Ok(Some((groups,total)))
}
fn main() {
	let (groups,total) = launch(257,64,1024).unwrap().unwrap();
	assert_eq!((groups,total),(5,320));
	let mut writes = vec![0;257];
	for group in 0..groups { for lane in 0..64 {
		let i = group*64+lane;
		if i<257 { writes[i]+=1; }
	}}
	assert!(writes.iter().all(|&n|n==1));
	assert_eq!(launch(0,64,1024).unwrap(),None);
	assert!(launch(1,0,1024).is_err());
}
```

`groups` は group の個数、`total` は全 thread 数である。実装では max_threads に device と pipeline の制限の小さい方を渡す。三次元化する場合は各軸の上限と積を両方検査する。

同じ入力を追う例。

```text
N=257,L=64 -> groups=5,total=320
last group g=4,lane=0 -> i=256 -> active
last group g=4,lane=1..63 -> i=257..319 -> no LOAD/STORE
```

この単独例は host の寸法と所有範囲を確認するものであり、Metal の実機実行を示さない。N=0 は shader ではなく host の分岐で処理する。

- [ ] groups を `ceil(N/L)` で計算し、掛算 overflow を検査する。
- [ ] `i<N` の内側に LOAD と STORE の両方を置く。
- [ ] device と pipeline の上限を使って local size を検証する。
- [ ] N=1、L-1、L、L+1、257 を同じ lowering で実行する。
- [ ] N=0 は host 側で空出力を返し、ゼロ個の group を dispatch しない。
- [ ] Plan の入力 buffer を差し替えて再実行する。

予定例 `metal_tail` は ones(257)+2 の全要素を 3 にする。
使用した `groups`、`local`、総 thread 数を表示する。
共有メモリと barrier はまだ無いので、無効な thread の早期終了は可能である。
[18](18-parallel-reduction.md) で barrier を加えるときは、この条件を作り直す。

### 3. broadcast と pad を安全に実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── metal_movement.rs  # [追加] broadcast と pad の予定実行例。
├── src/
│   ├── codegen/
│   │   ├── gpu.rs  # [更新] 平坦 ID から多次元出力座標への復元。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── renderer/
│   │   ├── metal.rs  # [更新] 条件付き device LOAD、比較/WHERE と標準数学関数。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── rangeify.rs  # [更新] 既存 movement の座標と valid 条件。
├── tests/
│   └── backend_metal.rs  # [更新] movement/math の CPU 比較。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/codegen/gpu.rs に unflatten_output を加え、既存 src/rangeify.rs の movement 写像へ復元座標を渡す。src/renderer/metal.rs の render 分岐へ guarded LOAD と math を追加する。掲載 padded_math はこの生成先の MSL であり、src/runtime/metal.rs のホストコードではない。

`cargo test` はexampleもビルドするため、このGPU専用exampleにはCargoの必要featureも設定する。次の項目を既存の `Cargo.toml` へ一度だけ追加する。

```toml
# Cargo.toml（このexampleの登録）
[[example]]
name = "metal_movement"
path = "examples/metal_movement.rs"
required-features = ["metal"]
```

このexampleのMetalを使う既存のmainには `#[cfg(target_os = "macos")]` を付ける。Metal専用のuseと補助関数にも同じcfgを付けるか、それらをcfg付きモジュールへまとめる。macOS以外でも `--all-features` のビルド時にMetal APIを参照しないよう、次のmainを同じファイルへ追加する。

```rust
// examples/metal_movement.rs（非macOS用の入口を追加）
#[cfg(not(target_os = "macos"))]
fn main() {
	eprintln!("Metal requires macOS");
	std::process::exit(1);
}
```


ここではGPU固有のmovement実装を作らず、10の出力座標から入力座標への写像をMSLへ渡す。候補は共通GPU loweringの設計案 `unflatten_output(i,shape)` とMSL rendererの条件付きLOADである。row-majorなshape `[s0,s1,…]` の座標は右端から `c[j]=p%s[j];p/=s[j]` と復元し、その後broadcast、reshape、permute、shrink、padを順に適用する。出力要素数0はhostで除外済みなので、長さ0の軸で剰余を計算しない。memoはNodeId、座標式、scope、有効条件、buffer stateを区別し、CPU側で成立した安全性をshader側でも保つ。

`[2,3]+[3]` の平坦thread IDが4なら出力座標は `(1,1)`、左の入力位置は4から5、右の位置は1から20で、結果25を出力4へ書く。ID5は `(1,2)` から6と30を読み36になる。bufferのbyte offsetは最初は0のまま、element添字の平坦化で位置を決めるので、shape変換のたびにhostがview bufferを作る必要はない。

pad `[1,2]`、before=1、after=2では出力ID0の入力候補は-1である。MSLの制御分岐でvalidを判定し、falseなら定数0、trueの枝だけでdevice pointerをLOADする。`select`の引数として二つの値を先に計算する形では範囲外LOADが残ることがあるため、その変換は行わない。kernel全体の `i<N` とpad由来のvalidは別の条件で、前者が真でも後者は偽になり得る。出力ID3は存在する出力要素なのでゼロをSTOREする。

以後の勾配や損失も通すため、必要な比較、WHERE、数学演算のMSL対応をこのrendererの通常命令へ追加する。sqrt、sin、log2、exp2はMetal標準ライブラリの対応関数を使い、CPUのsinf等の名前をそのまま出さない。局所テストはbroadcastとpadの値をinterpreter、CPU、Metalで比較し、sourceで二段のguardとLOAD位置を見る。転置とreshapeの合成にも同じ写像を使えば、backendごとに座標式が食い違う箇所を増やさずに済む。


pad の分岐位置を `src/renderer/metal.rs` で確認する。次は before=1,after=2 と SQRT+1 を融合した実際の MSL で、入力 x は非負値を契約とする。

組込み用の抜粋。実際の MSL kernel。実機では未検証。

PAD/SQRT/ADD を MSL にした生成例。ホスト runtime へ貼らない。

```metal
// src/renderer/metal.rs が生成するコード。生成先: <一時ディレクトリ>/padded_math.metal
#include <metal_stdlib>
using namespace metal;
kernel void padded_math(device const float *x [[buffer(0)]],
                        device float *out [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
	if (i < 5) {
		float value = 0.0f;
		int c = int(i)-1;
		if (c >= 0 && c < 2) {
			value = x[c];
		}
		out[i] = sqrt(value)+1.0f;
	}
}
```

外側の条件は出力が存在するか、内側は元入力を読めるかを判断する。i=3 は出力範囲内だが c=2 が入力範囲外なので、LOAD をせず sqrt(0)+1 を保存する。

同じ入力を追う例。

```text
x=[1,4]
PAD -> [0,1,4,0,0]
SQRT -> [0,1,2,0,0]
ADD 1 -> [1,2,3,1,1]
```

PAD のゼロをどの位置で作るかによって後続演算の結果も変わる。出力の範囲外 thread と padding の領域を同じ早期 return にしない。

- [ ] 出力の平坦 ID を多次元座標へ復元する。
- [ ] movement と broadcast を CPU と同じ座標変換から生成する。
- [ ] pad の有効条件が成立した場合だけ device buffer を LOAD する。
- [ ] 無効 LOAD の値を事後に WHERE で隠す実装を拒否する。
- [ ] CPU、interpreter、Metal の三つの値を比較する。

予定例 `metal_movement` は `[[1,2,3],[4,5,6]]+[10,20,30]` を実行する。
出力は `[[11,22,33],[14,25,36]]` とする。
pad([1,2],[(1,2)]) は `[0,1,2,0,0]` になり、範囲外 LOAD を発行しない。

### 4. 逐次 reduction と matmul を実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── metal_reduce_matmul.rs  # [追加] 逐次縮約と matmul の予定実行例。
├── src/
│   ├── codegen/
│   │   ├── gpu.rs  # [更新] 出力軸だけを thread ID へ割り当てる map_output_axes。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── renderer/
│   │   ├── metal.rs  # [更新] private acc と逐次 RANGE/END の MSL。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   └── lib.rs  # 本体のモジュール宣言と公開APIの入口。
├── tests/
│   └── backend_metal.rs  # [更新] 行和、matmul、MLP 勾配と再bind。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/codegen/gpu.rs の map_output_axes を更新し、縮約 RANGE は逐次のまま残す。src/renderer/metal.rs は既存の累積用 REG を thread 固有の float として出し、初期化と STORE の位置を掲載 gemm_serial に合わせる。前向き/勾配は src/autodiff.rs の共通変換から受け取る。

`cargo test` はexampleもビルドするため、このGPU専用exampleにはCargoの必要featureも設定する。次の項目を既存の `Cargo.toml` へ一度だけ追加する。

```toml
# Cargo.toml（このexampleの登録）
[[example]]
name = "metal_reduce_matmul"
path = "examples/metal_reduce_matmul.rs"
required-features = ["metal"]
```

このexampleのMetalを使う既存のmainには `#[cfg(target_os = "macos")]` を付ける。Metal専用のuseと補助関数にも同じcfgを付けるか、それらをcfg付きモジュールへまとめる。macOS以外でも `--all-features` のビルド時にMetal APIを参照しないよう、次のmainを同じファイルへ追加する。

```rust
// examples/metal_reduce_matmul.rs（非macOS用の入口を追加）
#[cfg(not(target_os = "macos"))]
fn main() {
	eprintln!("Metal requires macOS");
	std::process::exit(1);
}
```


この段階のreductionはGPU内でも一threadが一出力を最後まで計算する。候補は共通GPU軸割当とMSLのRANGE、局所REG、ENDの出力である。設計案の `map_output_axes` は縮約されない出力軸だけを平坦thread IDへ置換し、reduction軸は通常の逐次loopとして残す。SUMの累積はMSLのthread固有のfloat変数で、threadgroup共有領域へ置かない。初期値0は各出力に一回、全ての縮約loopの外に置き、global STOREはloopが終わった後に一回だけ行う。

行和なら論理出力数2で、thread0がrow0の1,2,3を読み、accは0→1→3→6となる。thread1は独立した0から4,5,6を足して15となる。matmulの出力shape `[2,2]` ならID3を `(m,n)=(1,1)` へ復元し、k=0,1,2で `4*2,5*4,6*6` を足すのでaccは0→8→28→64となる。他のthreadは別の出力位置へ書き、同じaccを共有しない。LOCAL軸は起動のlaneを表していても、縮約kを複数laneに割り当てる意味ではない。

小型MLPでは前向き出力、scalar loss、各重みの勾配を既存の純粋graphから作り、同じMetal backendでcompileする。gradient APIはCPUでgraphを変換する処理なので、Metal専用の微分規則を追加しない。scalar lossは一出力threadがbatchやclassのSUMを逐次実行できる。MAX、WHERE、通常の数学関数が必要な場合はrenderer capabilityを検査し、未対応opをCPUで黙って代行せず診断する。安定化用DETACHは微分後に除去し、GPU入力へ残さない。

局所検証では行和と非正方matmulの整数期待値を先に比較し、次に一回のMLP前向きと全gradientをCPUと許容誤差で比較する。生成MSLでacc初期化がkの外、出力guardの内側にあるか見る。重みslotを再bindして値が変わることも確認する。精度差の原因を切り分けるため、入力転送、各kernel出力、最終読出しを順に照合し、この時点で速度や並列reductionの成功を主張しない。


matmul の縮約軸を逐次のまま残した `src/renderer/metal.rs` の生成例である。A[M,K],B[K,N] の各出力一要素を一 thread が所有する。

組込み用の抜粋。実際の MSL kernel。実機では未検証。

出力一要素を一 thread が所有する逐次 GEMM の生成 MSL。

```metal
// src/renderer/metal.rs が生成するコード。生成先: <一時ディレクトリ>/gemm_serial.metal
#include <metal_stdlib>
using namespace metal;
kernel void gemm_serial(device const float *a [[buffer(0)]],
                        device const float *b [[buffer(1)]],
                        device float *out [[buffer(2)]],
                        constant uint &M [[buffer(3)]],
                        constant uint &K [[buffer(4)]],
                        constant uint &N [[buffer(5)]],
                        uint i [[thread_position_in_grid]]) {
	if (i < M*N) {
		uint m = i/N, n = i%N;
		float acc = 0.0f;
		for (uint k=0; k<K; k++) {
			acc += a[m*K+k]*b[k*N+n];
		}
		out[i] = acc;
	}
}
```

`acc` は thread 固有の private な変数である。M,N と byte 長を host で検証し、N=0 は dispatch しない。K=0 では loop が空になり0を保存する。

同じ入力を追う例。

```text
M=2,K=3,N=2,i=3 -> m=1,n=1
k=0 -> 4*2=8
k=1 -> 5*4=20, acc=28
k=2 -> 6*6=36, acc=64
out[3]=64
```

LOCAL ID が存在しても k を lane へ分割したことにはならない。ここに共有 memory と barrier はなく、18の並列版の比較先となる。

- [ ] 一つの thread が一行を担当し、列方向を逐次累積する。
- [ ] matmul は一つの thread が一出力 `(m,n)` を担当する。
- [ ] reduction の初期値を各 thread のローカル変数に置く。
- [ ] CPU と同じ shape graph から reduction と matmul を作る。
- [ ] 小型 MLP の前向きと勾配にも同じ backend を選べるようにする。

予定例 `metal_reduce_matmul` は `[6,15]` と `[[22,28],[49,64]]` を返す。
数値比較には `abs_error<=1e-5+1e-5*abs(cpu)` を初期目標にする。
学習の前に一回の前向きと勾配を CPU と比較し、差の発生箇所を特定する。

## 完了確認

次の例と `tests/backend_metal.rs` は追加予定である。
Metal 実機を必要とする項目は、実行機器名と実行の有無を記録する。

```sh
# 作業ディレクトリ: chibigrad/
cargo test
cargo run --example metal_add -- --dump-source
cargo run --example metal_tail -- --size 257 --local 64 --dump-launch
cargo run --example metal_movement -- --compare cpu
cargo run --example metal_reduce_matmul -- --compare cpu --dump-kernel
```

- [ ] CPU 用テストは Metal がない環境でも実行できる。
- [ ] 未対応機器の skip を、実機の数値比較成功として報告しない。
- [ ] GPU の build error と command buffer の実行エラーを別々に伝える。
- [ ] 結果を読む前の完了待ちとリソースの寿命が runtime に集約される。
- [ ] MSL に未対応 UOp が残る場合はコンパイル前に拒否する。
- [ ] 空の加算をテストし、Metal 呼出しなしで空の結果を得る。

## tinygrad ソースとの対応

参照 commit は `c22edff14c79310a9b2df62a78819cf77c8289fa` である。

| 実在するシンボル | ソース | 役割 | chibigrad での簡略化 |
|---|---|---|---|
| `MetalRenderer` | [cstyle.py](../../tinygrad/tinygrad/renderer/cstyle.py#L367) | MSL の型、buffer、group と local ID を出す | f32、一次元 launch、通常の引数 bind に限る |
| `get_grouped_dims` | [gpudims.py](../../tinygrad/tinygrad/codegen/gpudims.py#L28) | 論理軸を hardware の次元へ対応させる | 最初は出力軸を一次元へ平坦化する |
| `MetalAllocator._alloc` | [ops_metal.py](../../tinygrad/tinygrad/runtime/ops_metal.py#L172) | shared storage の Metal buffer を確保する | residency と外部ポインタは扱わない |
| `MetalDevice.__init__` | [ops_metal.py](../../tinygrad/tinygrad/runtime/ops_metal.py#L200) | device と command queue を作る | 同期する単一 queue に限る |
| `MetalCompiler` | [ops_metal.py](../../tinygrad/tinygrad/runtime/ops_metal.py#L29) | private service でソースをコンパイルする | この経路は移植せず公開 source compile を使う |
| `MetalQueue` | [ops_metal.py](../../tinygrad/tinygrad/runtime/ops_metal.py#L128) | HCQ2 と間接 command を構築する | 通常の encoder を直接呼ぶ |

## 今は追加しないもの

shared memory と並列 reduction は [18](18-parallel-reduction.md) に進める。
次は [16 の OpenCL](16-opencl.md) で同じ数値例を別 backend に通す。
tensor core と simdgroup matrix は [25](25-hardware-extensions.md) に分ける。
