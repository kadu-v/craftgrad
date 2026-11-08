# 16. 通常の OpenCL API で同じ graph を実行する

## このステップで動かすもの

[15 の Metal](15-metal.md) と同じ shape graph を OpenCL C に下げる。
元計画の第 9 章「Metal、次に OpenCL」の OpenCL 部分に対応する。
CPU、Metal、OpenCL の違いを renderer と runtime の境界に収める。

```text
[1,2,3,4] + [10,20,30,40] = [11,22,33,44]
ones(257)*2+1 = [3 を257個]
row_sum([[1,2,3],[4,5,6]]) = [6,15]
[[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
```

API、example、テスト、コマンドは追加予定である。
利用可能な OpenCL platform と device は実行時に調べる。
OpenCL device がない場合の source 確認を、実機の動作確認と区別する。

## 最小実装とデータ

OpenCL 用 kernel の入口は `__kernel void` とし、buffer は `__global float*` を使う。
共通の launch は `Launch { groups, local }` のままにする。
OpenCL API に渡す `global_work_size` は各次元で `groups[d]*local[d]` とする。
group 数をそのまま global_work_size として渡さない。

```text
論理出力N=257
Launch: groups=[5,1,1], local=[64,1,1]
OpenCL呼出し: work_dim=1, global_work_size=[320], local_work_size=[64]
kernel: i=get_global_id(0); if(i<257){ out[i]=x[i]*2+1; }
```

初版は単一 context、単一 device、in-order command queue に限る。
`clCreateProgramWithSource`、`clBuildProgram`、`clCreateKernel` を使う。
build 失敗時には `CL_PROGRAM_BUILD_LOG` を取得して source と一緒に表示する。
`clCreateBuffer` で確保し、blocking な `clEnqueueWriteBuffer` で入力を転送する。
引数は PARAM slot 順に `clSetKernelArg` で設定する。
`clEnqueueNDRangeKernel` 後は blocking read、または `clFinish` 後の read で結果を得る。
`cl_mem`、kernel、program、queue、context を所有型にまとめ、使用後に解放する。
HCQ2、ホスト側 UOp、image、SVM、out-of-order queue は持ち込まない。

local size は device の最大 workgroup size と kernel 固有の制限を満たす値にする。
一つの次元の上限、全次元の積の上限、global の積の overflow を別々に調べる。
ソース上の ID は get_global_id でも `get_group_id*L+get_local_id` でも同じ意味にする。
まず一次元 launch に固定し、三次元への展開は必要な例ができてから行う。

## この章のファイル配置

Metal と共通の src/codegen/gpu.rs を再利用し、OpenCL C の表記だけを src/renderer/opencl.rs、OpenCL handle と同期を src/runtime/opencl.rs に追加する。shader とホスト API の疑似コードは別々の配置として読む。

次はこの章の終わりまでに触る関連ファイルの配置案である。現在のcheckoutの一覧ではない。各コミットでは、その下に示す部分だけを追加・変更する。

```text
chibigrad/
├── examples/
│   ├── book_16_opencl_ndrange.rs  # 本文のコードを配置する。
│   ├── opencl_add.rs  # Tensor から OpenCL 加算を動かす予定例。
│   ├── opencl_movement_math.rs  # 共通 shape graph を実行する予定例。
│   ├── opencl_reduce_matmul.rs  # 逐次基準の予定実行例。
│   └── opencl_tail.rs  # groups/local/global を表示する予定例。
├── src/
│   ├── codegen/
│   │   ├── gpu.rs  # 出力軸のみを work item へ割り当てる処理。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── renderer/
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── opencl.rs  # private な acc と逐次縮約 loop。
│   ├── runtime/
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── opencl.rs  # to_ndrange、device/kernel 上限と空入力処理。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── rangeify.rs  # movement の座標と入力 valid 条件。
├── tests/
│   └── backend_opencl.rs  # 行和、matmul、dB と再bind の CPU 比較。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

生成Cやshaderは一時ディレクトリへ出力するため、このRustプロジェクトのツリーには含めない。コード冒頭のコメントで、生成元と生成先を確認する。

## 小さなコミットで進める

### 1. platform 選択と加算を最後まで実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── opencl_add.rs  # [追加] Tensor から OpenCL 加算を動かす予定例。
├── src/
│   ├── renderer/
│   │   ├── mod.rs  # [更新] opencl モジュールの宣言。
│   │   └── opencl.rs  # [追加] 共通 GPU IR から OpenCL C を生成。
│   ├── runtime/
│   │   ├── mod.rs  # [更新] OpenCL runtime の feature 付き宣言。
│   │   └── opencl.rs  # [追加] device、program、handle 所有型と同期 run。
│   └── lib.rs  # 本体のモジュール宣言と公開APIの入口。
├── tests/
│   └── backend_opencl.rs  # [追加] source と条件付き実機比較。
└── Cargo.toml  # [更新] 選んだ OpenCL binding の任意依存と feature。
```

src/renderer/mod.rs に mod opencl を追加し、src/runtime/mod.rs の OpenCL 宣言と binding 依存は feature で有効にする。OpenClDevice/OpenClProgram と clRelease を持つ所有型は src/runtime/opencl.rs へ置き、Launch/BindingLayout は src/codegen/gpu.rs の共通型を参照する。掲載 kernel は renderer の生成出力、API 疑似text は runtime の呼出し順として使う。

新しいモジュールは、既存の宣言を残して次を追記する。内部の補助関数は各モジュールに置き、exampleから使う入口だけを公開する。

```rust
// src/renderer/mod.rs（モジュール宣言を追加）
pub mod opencl;
// src/runtime/mod.rs（モジュール宣言を追加）
#[cfg(feature = "opencl")]
pub mod opencl;
```

CPUだけのビルドがGPU runtimeを要求しないよう、既存の `Cargo.toml` の `[features]` に次のキーを追加する。テーブルがまだなければ作り、既にあればその中へキーだけを加える。bindingを選んだ際は、そのoptional依存もこのfeatureから有効にする。

```toml
# Cargo.toml（既存のfeaturesへ統合する）
[features]
opencl = []
```

この章以降で opencl の予定コマンドを実行するときは、`cargo run` または `cargo test` に `--features opencl` を追加する。通常のCPUテストはfeatureを付けずに行う。

GPU専用のintegration testも、次をファイルの先頭に置いて同じ条件で有効にする。

```rust
// tests/backend_opencl.rs（ファイル先頭へ追加）
#![cfg(feature = "opencl")]
```

`cargo test` はexampleもビルドするため、このGPU専用exampleにはCargoの必要featureも設定する。次の項目を既存の `Cargo.toml` へ一度だけ追加する。

```toml
# Cargo.toml（このexampleの登録）
[[example]]
name = "opencl_add"
path = "examples/opencl_add.rs"
required-features = ["opencl"]
```


このコミットはMetalで用意した共通GPU IRをOpenCL Cへrenderし、通常APIで同期実行する。候補は `src/renderer/opencl.rs` と `src/runtime/opencl.rs` である。設計案の `OpenClDevice` はcontextとin-order queueを所有し、`OpenClProgram` はprogram、kernelとbinding表を持つ。初期化は `clGetPlatformIDs`、`clGetDeviceIDs` で列挙し、利用者の指定と対応するdeviceを一つ選んで `clCreateContext`、対応版の `clCreateCommandQueue` または `clCreateCommandQueueWithProperties` を呼ぶ。queueのin-order条件を既定値として維持する。

source生成後は `clCreateProgramWithSource`、`clBuildProgram`、`clCreateKernel` の順に処理する。build失敗時は `clGetProgramBuildInfo(...,CL_PROGRAM_BUILD_LOG,...)` で選んだdeviceのlogを取得する。入力と新規出力はclCreateBufferで別々に確保し、blockingのclEnqueueWriteBufferで入力を転送する。入力は `__global const float*` とCL_MEM_READ_ONLY、出力は可変pointerと適切なwrite権限で扱う。次にclSetKernelArgで各cl_mem handleを設定し、NDRangeをenqueueする。blockingのclEnqueueReadBufferが完了してからホスト配列を返す。

長さ4、local=4ならglobal=4で、ID2は3と30から33を作り、全出力 `[11,22,33,44]` になる。論理的なruntime入力はbuffersとscalarsの二配列を維持する。設計案のBindingLayoutが各buffer slotとscalar slotをkernelの引数番号へ写し、scalarは下位bitから型へ復元した値をclSetKernelArgで渡す。cl_memの引数sizeは `sizeof(cl_mem)` であり、buffer本体のbyte数ではない。scalarのsizeも宣言した型と一致させる。

FFIのunsafeはこのruntimeに閉じ、API名と戻り値を全ての失敗へ付ける。buffer、kernel、program、queue、contextは対応するclRelease関数を持つ所有型にし、最終read完了まで保持する。局所テストでは同じTensor graphのCPU比較、構文不正sourceのbuild log、platform不在とdevice指定違いを確認する。機器なしのsource検査を実機の加算成功として数えない。


`src/renderer/opencl.rs` は同じ buffer と scalar の対応を OpenCL C の引数列へ出す。四つの f32 を足す最初の kernel も N の guard を持たせる。

組込み用の抜粋。実際の OpenCL C kernel。実機では未検証。

共通 GPU IR から出す加算の OpenCL C。

```opencl
// src/renderer/opencl.rs が生成するコード。生成先: <一時ディレクトリ>/add_f32.cl
__kernel void add_f32(__global const float *x,
                      __global const float *y,
                      __global float *out,
                      const uint n) {
	size_t i = get_global_id(0);
	if (i < n) {
		float a = x[i];
		float b = y[i];
		float result = a+b;
		out[i] = result;
	}
}
```

buffer 引数0,1,2は cl_mem handle、引数3は uint の値である。`get_global_id` は全 work item の一次元位置を返すので、group 数をそのまま i の範囲にしない。

同じ入力を追う例。

```text
cl_mem arguments: index=0 x, index=1 y, index=2 out
uint argument: index=3 N=4
get_global_id(0)=2 -> x[2]=3,y[2]=30 -> out[2]=33
```

同じ論理 BindingLayout を利用しても、clSetKernelArg に渡すサイズは buffer の byte 長ではなく handle または scalar の型サイズになる。


外部APIの疑似コード。OpenCL の公開関数を呼ぶ順序であり、Rust FFI と RAII wrapper は未選定である。各呼出しの status を直後に検査し、エラーに API 名を付ける。

OpenCL 公開 API の接続順。選んだ Rust FFI と所有型へ置き換える疑似コード。

```text
// src/runtime/opencl.rs（外部APIの疑似コード）
platforms = clGetPlatformIDs()
devices = clGetDeviceIDs(selected_platform, CL_DEVICE_TYPE_ALL)
device = select_requested_device(devices)
context = clCreateContext(device)
queue = clCreateCommandQueue(context, device, properties=0)
program = clCreateProgramWithSource(context, source)
status = clBuildProgram(program, device, options="")
if status != CL_SUCCESS:
    return clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG)
kernel = clCreateKernel(program, "add_f32")
x = clCreateBuffer(context, CL_MEM_READ_ONLY, 16)
y = clCreateBuffer(context, CL_MEM_READ_ONLY, 16)
out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 16)
clEnqueueWriteBuffer(queue, x, CL_TRUE, offset=0, size=16, input_x)
clEnqueueWriteBuffer(queue, y, CL_TRUE, offset=0, size=16, input_y)
clSetKernelArg(kernel, 0, sizeof(cl_mem), &x)
clSetKernelArg(kernel, 1, sizeof(cl_mem), &y)
clSetKernelArg(kernel, 2, sizeof(cl_mem), &out)
clSetKernelArg(kernel, 3, sizeof(cl_uint), &N)
clEnqueueNDRangeKernel(queue, kernel, work_dim=1, offset=NULL, global=[4], local=[4])
clEnqueueReadBuffer(queue, out, CL_TRUE, offset=0, size=16, host_output)
clReleaseMemObject(x); clReleaseMemObject(y); clReleaseMemObject(out)
clReleaseKernel(kernel); clReleaseProgram(program)
clReleaseCommandQueue(queue); clReleaseContext(context)
```

in-order queue は write、kernel、read の順を守る。最後の blocking read の成功後にだけ結果を返し、その前に handle を解放しない。途中で失敗した場合にも作成済み handle を解放する責務は RAII 型へ置く。この接続部も実機では未検証である。

- [ ] platform と device を列挙し、明示指定できる入口を追加する。
- [ ] f32 加算の kernel source を共通 IR から生成する。
- [ ] program build と kernel 作成、buffer の bind を実装する。
- [ ] 入力転送、NDRange、同期読出しを一巡させる。
- [ ] 各 OpenCL エラーに API 名と戻り値を含める。

予定例 `opencl_add` は `[11,22,33,44]` を返す。
device 名、driver、使用した platform をログに残す。
build に失敗する source を runtime のテストから渡し、build log が表示されることを確認する。

### 2. groups と local の変換、端数を実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_16_opencl_ndrange.rs  # 本文のコードを配置する。
│   └── opencl_tail.rs  # [追加] groups/local/global を表示する予定例。
├── src/
│   ├── codegen/
│   │   ├── gpu.rs  # [更新] 共通の i<N guard。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── runtime/
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── opencl.rs  # [更新] to_ndrange、device/kernel 上限と空入力処理。
│   └── lib.rs  # 本体のモジュール宣言と公開APIの入口。
├── tests/
│   └── backend_opencl.rs  # [更新] 単位変換、端数、再bind と空入力。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/runtime/opencl.rs の起動直前に to_ndrange を追加し、groups*local を checked 演算で API の global_work_size に変換する。掲載 to_ndrange/main は book_16_opencl_ndrange の単独例に保存する。runtime 版は device/kernel の問い合わせ結果を受け取り、N=0 はその前で空出力へ分岐する。

`cargo test` はexampleもビルドするため、このGPU専用exampleにはCargoの必要featureも設定する。次の項目を既存の `Cargo.toml` へ一度だけ追加する。

```toml
# Cargo.toml（このexampleの登録）
[[example]]
name = "opencl_tail"
path = "examples/opencl_tail.rs"
required-features = ["opencl"]
```


ここでは共通のgroups/localとOpenCLのglobal_work_sizeの単位の違いを一箇所へ閉じ込める。候補はOpenCL runtimeの設計案 `to_ndrange(launch)` と共通の出力guard生成である。groupsはworkgroupの個数、localは各group内のwork item数、globalは全work item数なので、各次元のglobalをcheckedな掛算 `groups[d]*local[d]` で作る。初版はwork_dim=1とし、使用しない次元は1で統一する。N>0ならgroupsは切上げ `N/L+(N%L!=0)` で決め、L=0や整数overflowを起動前に拒否する。

N=257,L=64ではgroups=5,global=320となる。get_global_id(0)=256のwork itemは最後の有効な入力1を読み、`1*2+1=3` を出力256へ書く。ID257..319は `i<N` を満たさず、LOADもSTOREも実行しない。groupsの5をglobalへそのまま渡すと最初の5要素しか処理できないので、単に末尾の値を見るだけでなく全要素と保存範囲を検証する。予定のdumpはgroups、local、globalを別々に表示し、この単位間違いが見える形にする。

起動検証はdeviceの `CL_DEVICE_MAX_WORK_ITEM_SIZES`、`CL_DEVICE_MAX_WORK_GROUP_SIZE` とkernelの `CL_KERNEL_WORK_GROUP_SIZE` を確認する。各軸の上限とlocalの積を両方検査し、64が使えない場合は合法なLを選ぶか明示指定を拒否する。推奨倍数は必須制約と区別する。N=0は空結果をhostで返し、ゼロbyteのbuffer作成やglobal=0のNDRangeへ流さない。

局所テストでは1,L-1,L,L+1,257で出力を番兵値から開始し、全有効位置が3になったことをCPUと比較する。buffer長とshapeをdispatch前に照合し、入力の範囲外読出しをGPU側のエラー待ちで検出する方式にはしない。新しい入力bufferへ再bindした二回目にもclSetKernelArgを更新し、古いcl_memを読み続けないことを確認する。これで15と同じLaunchを使いながら、OpenCLのAPIだけに必要な寸法変換をruntimeの一関数で説明できる。


`src/runtime/opencl.rs` の `to_ndrange` は起動単位を明示する。次の単独例では各軸の device 上限と全 workgroup の kernel 上限を入力として受け取る。

単独で実行できる例。

API を呼ばない launch 単位変換と main 全体。

```rust
// examples/book_16_opencl_ndrange.rs（単独で実行する例）
fn to_ndrange(groups: [usize;3], local: [usize;3], axis_max: [usize;3], kernel_max: usize)
	-> Result<[usize;3],String> {
	let mut total_local = 1_usize;
	let mut global = [0;3];
	for d in 0..3 {
		if groups[d]==0 || local[d]==0 || local[d]>axis_max[d] {
			return Err("invalid launch dimension".into());
		}
		total_local = total_local.checked_mul(local[d]).ok_or("local overflow")?;
		global[d] = groups[d].checked_mul(local[d]).ok_or("global overflow")?;
	}
	if total_local>kernel_max { return Err("kernel workgroup limit".into()); }
	Ok(global)
}
fn main() {
	let global = to_ndrange([5,1,1],[64,1,1],[1024,1024,64],256).unwrap();
	assert_eq!(global,[320,1,1]);
	assert!(to_ndrange([5,1,1],[64,1,1],[32,1024,64],256).is_err());
	assert!(to_ndrange([1,1,1],[64,4,2],[1024;3],256).is_err());
	assert!(to_ndrange([usize::MAX,1,1],[2,1,1],[1024;3],256).is_err());
}
```

`total_local` は全軸の積、`global[d]` はその軸の group 数と local 数の積である。一次元の初版では未使用の軸を1にそろえる。N=0 はこの関数へ渡す前に空結果を返す。

同じ入力を追う例。

```text
groups=[5,1,1],local=[64,1,1] -> global=[320,1,1]
API work_dim=1 -> global_work_size points to [320]
logical N=257 -> active IDs 0..256, inactive IDs 257..319
```

未使用の軸の1と、論理的な0要素は意味が違う。後者を global=0 の NDRange にして渡さない。

- [ ] 共通 launch から global_work_size を計算する関数を作る。
- [ ] N=257、local=64 では global=320 を渡す。
- [ ] `i<N` の内側でだけ LOAD と STORE を実行する。
- [ ] N=1、L-1、L、L+1 を CPU と比較する。
- [ ] N=0 は host 側で空出力を返し、ゼロサイズ NDRange を投入しない。
- [ ] local size が機器の上限を越えたら起動前に拒否する。

予定例 `opencl_tail` は ones(257)*2+1 の全要素を 3 にする。
起動ログは `groups=5,local=64,global_work_size=320` を区別して表示する。
local=64 を使えない機器では適切な値を選び、実際の launch を報告する。

### 3. movement、broadcast、math を同じ座標変換に通す

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── opencl_movement_math.rs  # [追加] 共通 shape graph を実行する予定例。
├── src/
│   ├── codegen/
│   │   ├── gpu.rs  # [更新] 共通の座標復元と GPU 出力 guard。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── renderer/
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── opencl.rs  # [更新] movement、guarded LOAD、通常 math の OpenCL 表記。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── rangeify.rs  # [更新] movement の座標と入力 valid 条件。
├── tests/
│   └── backend_opencl.rs  # [更新] 転置後 reshape、pad と数学関数。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/renderer/opencl.rs の render_op/render_guarded_load を追加し、共通 rangeify と GPU lowering が返す添字・条件をそのまま OpenCL C へ出す。二つの掲載 shader はこの renderer の出力例で、後者は外側 guard の内部へ出す本体である。src/runtime/opencl.rs に座標変換を重複実装しない。

`cargo test` はexampleもビルドするため、このGPU専用exampleにはCargoの必要featureも設定する。次の項目を既存の `Cargo.toml` へ一度だけ追加する。

```toml
# Cargo.toml（このexampleの登録）
[[example]]
name = "opencl_movement_math"
path = "examples/opencl_movement_math.rs"
required-features = ["opencl"]
```


movementと数学関数の追加はTensorの意味の再実装ではなく、共通loweringが作った添字式と演算のOpenCL表記を埋める作業にする。候補は共通の座標復元処理とOpenCL rendererの設計案 `render_op`、`render_guarded_load` である。平坦IDを出力shapeの座標へ戻し、reshapeは論理的な平坦化と除算・剰余、permuteは逆の軸対応、broadcastは長さ1の軸を0にするという規則をそのまま使う。GPUだから入力bufferの物理strideだけで転置後reshapeを処理できる、と仮定しない。

2×3入力を転置してshape `[3,2]` にし、reshape `[6]` した場合、ID1は転置側座標 `(0,1)`、元入力 `(1,0)` へ戻るので4を読む。ID3は転置側 `(1,1)`、元入力 `(1,1)` から5を読み、全体は `[1,4,2,5,3,6]` となる。pad `[1,2]` の出力ID0では元座標-1が無効なため0を返し、ID1だけが元位置0から1を読む。条件外のpointer LOADを先に計算してselectで隠す形にはせず、OpenCL Cのif分岐内へLOADを置く。出力の有効thread条件とpadの入力valid条件は別に維持する。

SQRT、SIN、LOG2、EXP2はOpenCL Cの `sqrt`、`sin`、`log2`、`exp2` に対応させる。float入力なのでCのsqrtf等をそのまま出す必要はない。例えばID1が入力4を読めばsqrtが2を返し、出力 `[1,2,3]` を得る。OpenCLの `native_*` は速度と精度の別契約なので既定にせず、後で明示モードにする場合も通常関数のテストを保持する。特殊値はCPUのbit完全一致ではなく、定めた分類と許容誤差で比較する。

`tests/backend_opencl.rs` では転置後reshape、非対称pad、broadcast、数学関数をCPUと比較し、sourceでLOADの条件位置と型を見る。添字memoはNodeIdだけでなく座標、scope、有効条件、buffer stateを区別する。未対応opはrenderer入力のverifierで拒否し、OpenCL build失敗へ意味のないsourceを渡さない。これで前向きと勾配の両方が使う共通の演算を、同じPlanのbinding方式で実行できるようになる。


`src/renderer/opencl.rs` の movement と数学演算を一つの kernel で示す。2×3 の入力を転置後 reshape し、sqrt を適用する生成例である。

組込み用の抜粋。実際の OpenCL C kernel。実機では未検証。

転置/reshape の座標と SQRT を出した OpenCL C。

```opencl
// src/renderer/opencl.rs が生成するコード。生成先: <一時ディレクトリ>/transposed_sqrt.cl
__kernel void transposed_sqrt(__global const float *x,
                              __global float *out) {
	size_t i = get_global_id(0);
	if (i < 6) {
		uint permuted_row = (uint)i/2;
		uint permuted_col = (uint)i%2;
		uint original_row = permuted_col;
		uint original_col = permuted_row;
		uint input_index = original_row*3+original_col;
		float value = x[input_index];
		out[i] = sqrt(value);
	}
}
```

i を `[3,2]` の座標へ復元してから `[1,0]` の軸対応を逆に適用する。sqrt は OpenCL の通常関数を使い、native_sqrt の精度契約は持ち込まない。

同じ入力を追う例。

```text
x=[[1,4,9],[16,25,36]]
transpose+reshape -> [1,16,4,25,9,36]
sqrt              -> [1,4,2,5,3,6]
i=1 -> (0,1) -> original (1,0) -> sqrt(x[3])=4
```

この写像は10の CPU と15の Metal に共通である。renderer は算術式の表記だけを変え、物理 stride だけで転置後 reshape を処理したことにしない。


組込み用の抜粋。PAD の場合の OpenCL C の一出力本体。`i<5` の外側 guard の中へ置き、x は長さ2とする。

PAD 用の一出力本体。生成 kernel の i<5 guard 内に出す。

```opencl
// src/renderer/opencl.rs が生成するコード。生成先: <一時ディレクトリ>/padded_load.cl
float value = 0.0f;
int input_coordinate = (int)i - 1;
if (input_coordinate >= 0 && input_coordinate < 2) {
	value = x[input_coordinate];
}
out[i] = value;
```

入力候補が負ならポインタを読まずゼロを保存する。上の sqrt 例と合成する場合も、この if の中へだけ LOAD を置く。GPU 実機での範囲検査は未実施である。

- [ ] 二次元の平坦 ID を shape の座標へ戻す。
- [ ] broadcast、転置後 reshape、shrink、pad を共通 lowering から生成する。
- [ ] pad の LOAD 自体を有効条件の分岐内に置く。
- [ ] SQRT、SIN、LOG2、EXP2 を OpenCL の通常の数学関数へ下げる。
- [ ] native 高速近似関数は明示的な別モードにする。

予定例 `opencl_movement_math` は転置後 reshape の `[1,4,2,5,3,6]` を返す。
pad([1,2],[(1,2)]) は `[0,1,2,0,0]`、sqrt([1,4,9]) は `[1,2,3]` になる。
最初から native_sin などを既定値にして CPU と違う精度を暗黙に許さない。

### 4. 逐次 reduction と matmul を実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── opencl_reduce_matmul.rs  # [追加] 逐次基準の予定実行例。
├── src/
│   ├── codegen/
│   │   ├── gpu.rs  # [更新] 出力軸のみを work item へ割り当てる処理。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── renderer/
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── opencl.rs  # [更新] private な acc と逐次縮約 loop。
│   └── lib.rs  # 本体のモジュール宣言と公開APIの入口。
├── tests/
│   └── backend_opencl.rs  # [更新] 行和、matmul、dB と再bind の CPU 比較。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/codegen/gpu.rs の map_output_axes を再利用し、src/renderer/opencl.rs に REG、逐次 RANGE/END、最終 STORE の出力をそろえる。掲載 row_sum の private acc を基準に、matmul の内側 LOAD を出す。勾配は既存 src/autodiff.rs の graph を同じ compile/run に渡す。

`cargo test` はexampleもビルドするため、このGPU専用exampleにはCargoの必要featureも設定する。次の項目を既存の `Cargo.toml` へ一度だけ追加する。

```toml
# Cargo.toml（このexampleの登録）
[[example]]
name = "opencl_reduce_matmul"
path = "examples/opencl_reduce_matmul.rs"
required-features = ["opencl"]
```


このコミットではreduction軸を通常loopとして保持したまま、OpenCLの各work itemへ出力を割り当てる。候補は共通GPU loweringとOpenCL rendererのRANGE、局所変数、ENDである。設計案の出力軸割当は、SUMで削除された軸をNDRangeへ割り当てず、残る出力shapeだけを平坦IDへ写す。累積floatはwork itemごとのprivate変数とし、毎出力の開始時に0を設定する。__localの共有変数を使うと別laneの加算が競合するため、この段階では追加しない。

行和の出力ID0は入力の位置0,1,2を順に読み、acc=0→1→3→6となる。ID1は位置3,4,5から15を作る。matmulの出力shape `[M,N]` は `m=i/N,n=i%N` へ復元し、kのloopで `A[m*K+k]*B[k*N+n]` を加える。例のID2ならm=1,n=0で4*1、5*3、6*5を足して49となる。outの位置は2であり、groupやlaneの番号で別の出力を上書きしない。kの初期化と終了までをwork itemが単独で所有する。

勾配も13の高水準graph変換を一度行ってから同じOpenCL compilerへ渡す。前向き用とbackward用に異なる数値実行器を作らない。例えばloss=sum(A@B)なら、Bの各行の勾配はAの対応する列和になり、例のAでは `[[5,5],[7,7],[9,9]]` と比較できる。Bの全要素を1へ差し替えた次回runでは前向き出力が `[[6,6],[15,15]]` へ変わる。Planのprogram/kernelは再利用しても入力cl_memとscalar値はrunのたびにbindし直す。

局所テストは行和、matmul、上記の勾配と再bindをCPUで同じgraphから計算して照合する。整数小例の期待値から始め、通常f32では所定の許容誤差を使う。kernel sourceでacc初期化がkのloop外にあり、最終STOREが一度だけかを見る。in-order queueでは先行kernelの書込みが後続kernelへ渡り、最後のblocking readが完了するまで中間bufferを保持する。ここでLOCALやbarrierを入れず、18の並列版と比較できる逐次基準を残す。


`src/renderer/opencl.rs` の逐次縮約は private な acc と通常の for を使う。次は行和 kernel で、行数 rows だけを出力 ID に割り当てる。

組込み用の抜粋。実際の OpenCL C kernel。実機では未検証。

private acc で逐次縮約する OpenCL C の出力。

```opencl
// src/renderer/opencl.rs が生成するコード。生成先: <一時ディレクトリ>/row_sum.cl
__kernel void row_sum(__global const float *x,
                      __global float *out,
                      const uint rows,
                      const uint columns) {
	size_t row = get_global_id(0);
	if (row < rows) {
		float acc = 0.0f;
		for (uint col=0; col<columns; col++) {
			float value = x[row*columns+col];
			acc += value;
		}
		out[row] = acc;
	}
}
```

`acc` は各 work item に一つあり、`col` を読み終えるまで他の lane と共有しない。columns=0 なら0を保存する。matmul では row を `(m,n)` へ復元し、内側の LOAD を A[m*K+k] と B[k*N+n] の積へ替える。

同じ入力を追う例。

```text
row_sum x=[[1,2,3],[4,5,6]] -> out=[6,15]
GEMM output i=2,N=2 -> m=1,n=0 -> products=[4,15,30] -> 49
loss=sum(A@B), A=[[1,2,3],[4,5,6]]
dB=[[5,5],[7,7],[9,9]]
```

dB は各 k について A の列を足した値が二列へ broadcast される。同じ入力 graph の CPU 勾配と比較すれば、前向きだけでなく unbroadcast と縮約も検査できる。

- [ ] 一つの work item に一行の逐次 SUM を担当させる。
- [ ] matmul の出力 `(m,n)` を一つの work item に割り当てる。
- [ ] reduction の累積を work item ごとに初期化する。
- [ ] CPU と OpenCL の出力、勾配を小さい固定値で比較する。
- [ ] 新しい重みを同じ Plan の入力 slot に渡して再実行する。

予定例 `opencl_reduce_matmul` は `[6,15]` と `[[22,28],[49,64]]` を返す。
数値は `abs_error<=1e-5+1e-5*abs(cpu)` を初期目標として比較する。
共有メモリと barrier を入れず、Metal 初版と同じ仕事の割当てを維持する。

## 完了確認

以下は将来追加する example と `tests/backend_opencl.rs` の予定コマンドである。
Metal と OpenCL の両方が同じ機器で利用できるとは仮定しない。

```sh
# 作業ディレクトリ: chibigrad/
cargo test
cargo run --example opencl_add -- --dump-source
cargo run --example opencl_tail -- --size 257 --local 64 --dump-launch
cargo run --example opencl_movement_math -- --compare cpu
cargo run --example opencl_reduce_matmul -- --compare cpu --dump-kernel
```

- [ ] in-order queue で前の転送と kernel の順序が守られる。
- [ ] 非同期 read を開始しただけの値を、完了した値として返さない。
- [ ] device の不在、build 失敗、起動失敗を別々の診断にする。
- [ ] 機器がないときは source 検査と未実施の実機項目を分けて記録する。
- [ ] runtime の API 呼出し以外に OpenCL 固有のロジックを広げない。
- [ ] 空の加算をテストし、NDRange の投入なしで空の結果を得る。

## tinygrad ソースとの対応

参照 commit は `c22edff14c79310a9b2df62a78819cf77c8289fa` である。

| 実在するシンボル | ソース | 役割 | chibigrad での簡略化 |
|---|---|---|---|
| `OpenCLRenderer` | [cstyle.py](../../tinygrad/tinygrad/renderer/cstyle.py#L333) | OpenCL C と group、local ID を生成する | f32 buffer と一次元 launch に限る |
| `CLCompiler.compile` | [ops_cl.py](../../tinygrad/tinygrad/runtime/ops_cl.py#L28) | source から program を build し診断を得る | binary cache より source 実行を先に作る |
| `CLQueue.exec` | [ops_cl.py](../../tinygrad/tinygrad/runtime/ops_cl.py#L53) | kernel 引数と groups×local の寸法を設定する | 通常 API を直接呼び HCQ2 を省く |
| `CLAllocator` | [ops_cl.py](../../tinygrad/tinygrad/runtime/ops_cl.py#L82) | cl_mem の確保と blocking 転送 | image と host allocator の統合を省く |
| `CLDevice.__init__` | [ops_cl.py](../../tinygrad/tinygrad/runtime/ops_cl.py#L99) | context と command queue を作る | 一つの platform と device で始める |

## 今は追加しないもの

非同期転送、profiling event の高度な処理、複数 device は後の拡張にする。
ループ分割と CPU の thread は [17](17-loop-optimizations.md)、並列 reduction は [18](18-parallel-reduction.md) で扱う。
tensor core 相当の機能は [25](25-hardware-extensions.md) で capability を検査してから追加する。
