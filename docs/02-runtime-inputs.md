# 02. 同じ生成コードへ実行時入力を渡して 20 と 12 を返す

## このステップで動かすもの

[01-scalar-c.md](01-scalar-c.md) の `(2+3)*4` を `(a+b)*c` に変える。
入力 `2,3,4` で `20`、入力 `5,1,2` で `12` を返す。
一度ロードした共有ライブラリを二回呼び、入力だけを変える。
元計画の第 1 章の実行時入力と、第 3 章の CPU runtime の一部に対応する。
バッファ UOp は [05](05-memory-runtime.md) まで導入しない。

以下の API、コマンド、数値は未実装の作業計画である。
この章で作る example が、予定する入出力の検証役になる。

## 最小実装とデータ

`Param { slot: usize }` を scalar 入力として追加する。
`f32` 以外の dtype はまだ扱わない。
入力値はグラフのキーや生成 C に埋め込まない。

予定する入力 IR は次の形になる。

```text
%0 = PARAM f32 slot=0
%1 = PARAM f32 slot=1
%2 = PARAM f32 slot=2
%3 = ADD %0 %1
%4 = MUL %3 %2
root = %4
```

共有ライブラリの入口は、一つの固定 C ABI にする。

共有ライブラリ向け renderer が生成する chibi_entry の C 宣言である。Rust の extern 宣言として貼らない。

```c
// src/renderer/c.rs が生成するコード。生成先: <一時ディレクトリ>/kernel.c
void chibi_entry(void *const *buffers, const uint64_t *scalars);
```

`scalars[slot]` の下位 32 bit に `f32` のビット列を置く。
生成 C は `uint32_t word = (uint32_t)scalars[slot]` を作り、
`memcpy` で word の 4 byte を `float` に戻す。
64 bit スロットの先頭 4 byte を直接読む方法は使わない。
出力は `buffers[0]` が指すホスト上の `float` 一つへ書く。
入出力処理は wrapper の仕事で、`LOAD` や `STORE` UOp としては扱わない。
[05](05-memory-runtime.md) で同じ ABI をメモリ UOp に接続する。

Rust 側は `CompiledScalar` に library 所有物と入口を保持する案でよい。
入口の関数ポインタより先に library を解放しない。
`run` は入力数を検証し、スロット列と出力用ホスト領域を用意する。
`unsafe` と symbol 解決は CPU runtime にまとめる。

## この章のファイル配置

PARAM を加えるときに、01 の example で実際に使っている処理だけを src へ抽出する。モジュール名は uop、verify、rewrite、linearize、renderer::c、runtime::cpu とし、空ファイルを先に作らない。単独の ABI 変換例は本体と別の examples に置く。

次はこの章の終わりまでに触る関連ファイルの配置案である。現在のcheckoutの一覧ではない。各コミットでは、その下に示す部分だけを追加・変更する。

```text
chibigrad/
├── examples/
│   ├── book_02_scalar_words.rs  # std だけで f32 bits の往復を確かめる独立例。
│   ├── scalar_arithmetic.rs  # 共有処理を呼ぶ定数式の入口へ変更する。
│   ├── scalar_const.rs  # 共有処理を呼ぶ CONST 7 の入口へ変更する。
│   └── scalar_inputs.rs  # build_scalar_inputs と compile 一回、run 二回の確認。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # 三つの PARAM を読む計算式。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── runtime/
│   │   ├── cpu.rs  # ScalarSignature、呼出し前検証と library の解放。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── linearize.rs  # 依存順 DFS を移す。
│   ├── rewrite.rs  # 実行経路にある恒等 rewrite を移す。
│   ├── uop.rs  # 01 の NodeId、Op、UOp、Graph を移し、Param と必要入力数を加える。
│   └── verify.rs  # 到達可能な PARAM slot と宣言数をコンパイル前に照合する。
├── tests/
│   └── scalar_inputs.rs  # 共有ライブラリを使う再実行、入力不足、symbol 不在の統合確認。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

生成Cやshaderは一時ディレクトリへ出力するため、このRustプロジェクトのツリーには含めない。コード冒頭のコメントで、生成元と生成先を確認する。

## 小さなコミットで進める

### コミット 1. PARAM を実行ファイル wrapper の引数につなぐ

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── scalar_arithmetic.rs  # [変更] 共有処理を呼ぶ定数式の入口へ変更する。
│   ├── scalar_const.rs  # [変更] 共有処理を呼ぶ CONST 7 の入口へ変更する。
│   └── scalar_inputs.rs  # [追加] 実行時引数で同じ生成物へ値を渡す入口。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # [追加] render_c、c_expression と実行ファイル wrapper の生成。
│   │   └── mod.rs  # [追加] c モジュールの宣言。
│   ├── runtime/
│   │   ├── cpu.rs  # [追加] 01 の一時ファイル、clang と実行ファイルの起動を移す。
│   │   └── mod.rs  # [追加] cpu モジュールの宣言。
│   ├── lib.rs  # [追加] 共有したモジュールを宣言し、example が使うグラフ構築と実行経路を公開する。
│   ├── linearize.rs  # [追加] 依存順 DFS を移す。
│   ├── rewrite.rs  # [追加] 実行経路にある恒等 rewrite を移す。
│   ├── uop.rs  # [追加] 01 の NodeId、Op、UOp、Graph を移し、Param と必要入力数を加える。
│   └── verify.rs  # [追加] 01 の verify_graph を移し、PARAM slot の検証を加える。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

01 の二つの example から実装済みの型と関数を各 src ファイルへ移す。src/lib.rs に uop、verify、rewrite、linearize、renderer、runtime の宣言を加え、renderer/mod.rs と runtime/mod.rs はそれぞれ c と cpu を宣言する。example から必要な型、builder と実行 API だけを公開し、use の crate 名には既存 Cargo.toml の設定を使う。Param は src/uop.rs、入力数検査は verify.rs、render_executable_wrapper は renderer/c.rs、引数を渡す run_executable は runtime/cpu.rs に追加する。examples/scalar_inputs.rs はグラフ構築と二回の起動を担当する。

新しいモジュールは、既存の宣言を残して次を追記する。内部の補助関数は各モジュールに置き、exampleから使う入口だけを公開する。

```rust
// src/lib.rs（モジュール宣言を追加）
pub mod linearize;
pub mod renderer;
pub mod rewrite;
pub mod runtime;
pub mod uop;
pub mod verify;
// src/renderer/mod.rs（モジュール宣言を追加）
pub mod c;
// src/runtime/mod.rs（モジュール宣言を追加）
pub mod cpu;
```


定数だけを持つグラフへ、値を持たない入力位置を追加する。変更先候補は前章の実装ファイルで、設計案の `Param { slot }`、`required_scalars(root)`、実行時引数を受ける wrapper を置く。最初に `%a=PARAM(0)`、`%b=PARAM(1)`、`ADD(a,b)` を作る。renderer は PARAM を現在の入力値に置換せず、引数を読む式へ置換する。CONST 2 と PARAM slot 0 は、たとえ今回の値が 2 でも異なるノードである。

共有ライブラリ化の前なので、実行ファイルの `main` が受ける文字列を検証して f32 に変換し、計算関数へ渡す構成でよい。設計案の `render_executable_wrapper` が必要な入力数を source に含め、`run_executable(args)` が子プロセスの実行時引数として値を渡す。既存の定数 root の入力数は 0 となり、その呼出しも同じ renderer で成立させる。引数を C source の生成時に連結する処理は作らない。

一つの source を一度コンパイルし、同じ実行ファイルへ `2 3`、次に `5 1` を渡すと、slot 0 と 1 の値は順に `(2,3)`、`(5,1)` になり、ADD は 5、6 を返す予定である。局所テストは二回の source 文字列が同じことと、結果が変わることを一緒に調べる。入力不足と数値に変換できない引数は計算関数へ入る前にエラーにする。出力の読み方と clang の失敗処理は前章を再利用でき、今回変わるのは値の供給時点である。


組込み用の抜粋である。`src/renderer/c.rs` が生成する実コードを示す。PARAM 0 と 1 に対応する `a` と `b` は、コンパイル後に `argv` から供給される。

render_executable_wrapper が生成する C である。runtime/cpu.rs が書き出して clang に渡す。

```c
// src/renderer/c.rs が生成するコード。生成先: <一時ディレクトリ>/kernel.c
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
static float chibi_value(const float *params) {
	float a = params[0];
	float b = params[1];
	return a + b;
}
int main(int argc, char **argv) {
	if (argc != 3) { fprintf(stderr, "expected 2 scalars\n"); return 2; }
	float params[2];
	for (int i = 0; i < 2; i++) {
		char *end = NULL;
		errno = 0;
		params[i] = strtof(argv[i + 1], &end);
		if (errno || end == argv[i + 1] || *end != '\0') {
			fprintf(stderr, "invalid scalar slot %d\n", i); return 2;
		}
	}
	printf("%.9g\n", (double)chibi_value(params));
	return 0;
}
```

`strtof` の末尾ポインタで未消費の文字も検出する。値は `params` に入り、生成済みの `params[0]+params[1]` は変わらない。`argc` を先に検査するので、入力不足で `argv[2]` を読むこともない。CONST の場合にはこの計算関数が `params` を参照しない形で生成し、必要な引数数を 0 にする。

この入力の途中経過を並べる。

```text
compile(source) -> executable
executable 2 3 -> params=[2,3] -> 5
executable 5 1 -> params=[5,1] -> 6
executable 2 xyz -> invalid scalar slot 1
```

- [ ] scalar `Param` を追加し、`slot` の有効性を検証する。
- [ ] `a+b` の C 関数を生成し、example の入力を wrapper に渡す。
- [ ] 入力を C の定数にせず、実行時に受け取る。
- [ ] 前章の定数実行例も同じ renderer で通す。

ここでは前章の実行ファイル経路を使ってよい。
予定例 `a=2,b=3` の結果は `5`。
`a=5,b=1` に変えると、同じ生成 source から `6` を得る。

### コミット 2. 一つの入口を持つ共有ライブラリを呼ぶ

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_02_scalar_words.rs  # [追加] std だけで f32 bits の往復を確かめる独立例。
│   └── scalar_inputs.rs  # [変更] 共有ライブラリをコンパイルしてロードする実行例。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # [変更] 二配列 ABI と scalar_f32 helper の C 生成。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── runtime/
│   │   ├── cpu.rs  # [変更] compile_scalar、OwnedLibrary と CompiledScalar::load による共有ライブラリ所有。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   └── lib.rs  # 本体のモジュール宣言と公開APIの入口。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

runtime/cpu.rs に compile_scalar、OwnedLibrary、Entry、CompiledScalar と load を追加し、既存の clang 起動を共有ライブラリ生成にも使う。renderer/c.rs は scalar_f32 と chibi_entry を生成する。pack の単独例は main を含めて examples/book_02_scalar_words.rs へ保存し、その pack_f32 の規則だけを runtime/cpu.rs の実装へ組み込む。unpack_f32 と検証用 main を runtime モジュールへ貼らない。


実行ファイルの文字列引数を、ホストメモリの二配列 ABI に置き換える。実装先候補を `src/runtime/cpu.rs` と `src/renderer/c.rs` に分けるのは、実際に library 所有権と C wrapper が増えるこの時点でよい。設計案の `compile_scalar(source)` は共有 library と入口を一緒に持つ `CompiledScalar` を返す。`run(&self, inputs)` は scalar word 列と出力の f32 一要素を用意し、`buffers[0]` にその出力アドレスを入れて入口を同期呼出しする。

pack は数値の整数変換ではなく、ビット列の移送である。例えば `2.0f32.to_bits()` は `0x40000000` なので、`scalars[0]` に数値 2 ではなく `0x0000000040000000` を置く。C では下位 32 bit を `uint32_t` へ取り出し、その 4 byte を `memcpy` で float に戻す。これにより 2.0 が復元される。64 bit word のアドレスを float pointer として読むと aliasing と byte 順に依存するため、この復元規則を renderer の一か所へまとめる。

実装順は共有 library の生成、ロード、`chibi_entry` の解決、型を一致させた関数ポインタの保持、run の順とする。symbol の借用だけを外へ返さず、library が関数ポインタより長く生きる所有構造を作る。局所テストでは `(2,3)` を pack して ADD の出力 5 を読み、scalar word の dump も照合する。library の読み込み失敗と symbol 不在は別に報告する。グラフの rewrite と linearize は前章と同じであり、計算本体を FFI 専用に作り直す必要はない。


単独で実行できる例である。`src/runtime/cpu.rs` に置く pack と、C の復元と同じ「下位 32 bit を選ぶ」規則を std だけで確かめる。ここでは library のロードは実行せず、ABI のデータ変換を検証する。

pack と復元のデータ規則を確かめる独立例で、library のロードは行わない。

```rust
// examples/book_02_scalar_words.rs（単独で実行する例）
fn pack_f32(values: &[f32]) -> Vec<u64> {
	values.iter().map(|x| u64::from(x.to_bits())).collect()
}
fn unpack_f32(word: u64) -> f32 {
	f32::from_bits(word as u32)
}
fn main() {
	let input = [2.0_f32, 3.0, -0.0, f32::from_bits(0x7fc00042)];
	let words = pack_f32(&input);
	assert_eq!(words[0], 0x0000000040000000);
	for (value, word) in input.iter().zip(&words) {
		assert_eq!(value.to_bits(), unpack_f32(*word).to_bits());
	}
	assert_eq!(unpack_f32(words[0]) + unpack_f32(words[1]), 5.0);
	println!("slot0={:016x}, result=5", words[0]);
}
```

`as u64` による数値変換なら 2.0 は整数 2 になるが、ここでは `to_bits` が先なので word は `0x40000000` になる。NaN payload と負のゼロも同じ経路で往復する。共有 library が出す次の本体は同じ word を `uint32_t` へ絞ってから `memcpy` し、aliasing とホストの byte 順への依存を避ける。

この入力の途中経過を並べる。

```text
f32 2.0 -> bits 0x40000000 -> u64 0x0000000040000000
f32 3.0 -> bits 0x40400000 -> u64 0x0000000040400000
buffers=[address_of_output], scalars=[0x40000000,0x40400000]
output=5
```


組込み用の抜粋である。以下は renderer が出す ABI wrapper の実 C であり、先ほどの Rust 検証と同じ復元規則を使う。

共有ライブラリ用に生成する実 C の ABI wrapper である。

```c
// src/renderer/c.rs が生成するコード。生成先: <一時ディレクトリ>/kernel.c
#include <stdint.h>
#include <string.h>
static float scalar_f32(uint64_t slot) {
	uint32_t word = (uint32_t)slot;
	float value;
	memcpy(&value, &word, sizeof(value));
	return value;
}
void chibi_entry(void *const *buffers, const uint64_t *scalars) {
	float a = scalar_f32(scalars[0]);
	float b = scalar_f32(scalars[1]);
	*(float *)buffers[0] = a + b;
}
```

ロード接続では OS の `dlopen(path, RTLD_NOW)` と `dlsym(handle,"chibi_entry")` を使用し、両方の null を検査する。入口型は `unsafe extern "C" fn(*const *mut std::ffi::c_void, *const u64)` である。`CompiledScalar` はその入口と handle の所有物を一緒に保持し、所有物の Drop が `dlclose` を呼ぶ。パスと symbol を含むロードエラーは、02 のコンパイルエラーとは別に返す。

- [ ] 固定 ABI の `chibi_entry` と wrapper を生成する。
- [ ] macOS は clang の `-dynamiclib`、対応する Linux は `-shared -fPIC` を使う。
- [ ] OS の動的 library 機構でロードし、入口を解決する。
- [ ] library の所有物を `CompiledScalar` に保持する。
- [ ] `a+b` を compile、load、run まで通し、結果 `5` を読む。

対象ホストの一つの設定を動かせばよい。
未使用 OS の runtime を先に完成させない。
C と Rust で `f32`、スロット幅、入口の型を照合する。


組込み用の抜粋である。`src/runtime/cpu.rs` に置く macOS と Linux のロード部分を示す。`CompiledScalar` は library と入口を同じ所有物にし、外へはコミット 3 の `run` を公開する。共有 library の生成は macOS なら `clang -dynamiclib`、Linux なら `clang -shared -fPIC` の成功後にこの `load` へ渡す。

OwnedLibrary と CompiledScalar の定義、および loader の関数を追加する。

```rust
// src/runtime/cpu.rs（既存ファイルへ組み込む抜粋）
use std::{ffi::{c_char,c_void,CStr,CString},os::unix::ffi::OsStrExt,path::Path};
type Entry = unsafe extern "C" fn(*const *mut c_void,*const u64);
#[cfg_attr(target_os="linux",link(name="dl"))]
extern "C" {
	fn dlopen(path: *const c_char, flags: i32) -> *mut c_void;
	fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
	fn dlclose(handle: *mut c_void) -> i32;
	fn dlerror() -> *const c_char;
}
struct OwnedLibrary(*mut c_void);
impl Drop for OwnedLibrary {
	fn drop(&mut self) { unsafe { dlclose(self.0); } }
}
fn loader_error() -> String {
	let p = unsafe { dlerror() };
	if p.is_null() { "loader returned null".into() }
	else { unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned() }
}
struct CompiledScalar { _library: OwnedLibrary, entry: Entry, scalar_count: usize }
impl CompiledScalar {
	fn load(path: &Path, scalar_count: usize) -> Result<Self,String> {
		let cpath = CString::new(path.as_os_str().as_bytes()).map_err(|_| "NUL in library path")?;
		let handle = unsafe { dlopen(cpath.as_ptr(),2) }; // RTLD_NOW=2 on macOS/Linux。
		if handle.is_null() { return Err(format!("load {}: {}",path.display(),loader_error())); }
		let library = OwnedLibrary(handle);
		let name = CString::new("chibi_entry").unwrap();
		unsafe { dlerror(); } // この lookup より前の loader error を消す。
		let symbol = unsafe { dlsym(library.0,name.as_ptr()) };
		if symbol.is_null() { return Err(format!("symbol chibi_entry in {}: {}",path.display(),loader_error())); }
		// 対象 ABI の dlsym が返す関数アドレスだけを Entry へ変換する。
		let entry = unsafe { std::mem::transmute::<*mut c_void,Entry>(symbol) };
		Ok(Self { _library:library,entry,scalar_count })
	}
}
```

`OwnedLibrary` を symbol 解決前に作るので、欠落 symbol で早期 return しても handle が解放される。成功時はその所有物を `CompiledScalar` へ移し、入口の裸のポインタだけを返さない。`load` は ABI が上の C wrapper と一致する生成物だけに使い、`run` 中は `&self` が library の寿命を保つ。この抜粋は Unix の動的 loader へ接続する実装であり、Windows の loader を実装済みとするものではない。

### コミット 3. `(a+b)*c` を一度コンパイルして二回使う

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── scalar_inputs.rs  # [変更] build_scalar_inputs と compile 一回、run 二回の確認。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # [変更] 三つの PARAM を読む計算式。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── runtime/
│   │   ├── cpu.rs  # [変更] 入力だけを pack して同じ入口を呼ぶ CompiledScalar::run。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   └── lib.rs  # 本体のモジュール宣言と公開APIの入口。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

掲載した run を runtime/cpu.rs の既存 impl CompiledScalar に追加する。examples/scalar_inputs.rs の build_scalar_inputs で三つの PARAM を作り、コンパイルとロードは二回の run の外に一度だけ置く。renderer/c.rs は各 PARAM を scalar slot へ対応させ、入力値ではなく slot を生成する。


ここでは PARAM の数を三つに増やすだけでなく、生成物の寿命を入力の寿命から分ける。設計案の `build_scalar_inputs()` が slot 0、1、2 を使う `(a+b)*c` の root を返し、example は `let compiled = compile(root)` 相当の操作を一度だけ行う。その後で `compiled.run([2,3,4])` と `compiled.run([5,1,2])` を順に行う構造にする。新しい値を渡すたびに graph を再構築したり clang を起動したりする処理は、run の内部へ入れない。

一回目の word 列は 2、3、4 の f32 bits であり、C の中間値は 5 と 20 になる。二回目には同じ三つの slot に 5、1、2 の bits を詰め、C の中間値が 6 と 12 になる。更新するのは呼出し用配列だけで、PARAM の payload は slot のままである。出力領域は呼出しごとに用意するか明確に初期化し、前回の 20 が残っているだけの成功を避ける。C は出力アドレスを保存せず、呼出し中にだけ書く。

局所テストでは example が通過する compile と run の実際の呼出し箇所に計数を置き、compile=1、run=2 と二つの数値を同時に照合する。source に PARAM の参照式が三つあることを見れば、グラフが入力値へ特殊化されていないかを確認できる。数値 2 自体は別の記述に現れ得るため、単なる文字検索より PARAM のノードと復元箇所の対応を見る。既存の定数 example と二引数の ADD も同じ `CompiledScalar` の仕組みで動かし、特別な二回実行経路を作らない。


組込み用の抜粋である。`CompiledScalar` は前コミットの library 所有物と `entry` を持ち、さらに `scalar_count` を保持する。次の `run` に clang 呼出しや graph 構築を入れない。

既存の CompiledScalar に run を追加する。

```rust
// src/runtime/cpu.rs（既存ファイルへ組み込む抜粋）
impl CompiledScalar {
	fn run(&self, values: &[f32]) -> Result<f32, String> {
		if values.len() != self.scalar_count {
			return Err(format!("expected {} scalars", self.scalar_count));
		}
		let scalars: Vec<u64> = values.iter().map(|v| u64::from(v.to_bits())).collect();
		let mut output = f32::NAN;
		let buffers = [(&mut output as *mut f32).cast::<std::ffi::c_void>()];
		// entry は library 所有物が生きている間だけ有効。呼出しは同期的。
		unsafe { (self.entry)(buffers.as_ptr(), scalars.as_ptr()); }
		Ok(output)
	}
}
```

各呼出しで `scalars` と `output` が新しくなる一方、`self.entry` は同じである。`buffers`、`scalars`、出力の寿命は unsafe 呼出しの完了を覆う。生成する計算本体を `(scalar_f32(scalars[0])+scalar_f32(scalars[1]))*scalar_f32(scalars[2])` にすれば、二回目にも現在の三つの word が読まれる。入力の可変性を `CompiledScalar` の可変性へ結び付ける必要はない。

この入力の途中経過を並べる。

```text
compile(root): count=1
run([2,3,4]): words=[0x40000000,0x40400000,0x40800000], output=20
run([5,1,2]): words=[0x40a00000,0x3f800000,0x40000000], output=12
同じ library と entry、run count=2
```

- [ ] PARAM 三つで式を作る `scalar_inputs` example を追加する。
- [ ] compile と run を別の呼出しにする。
- [ ] 二回目はスロットのビット列だけを更新する。
- [ ] 同じ入口と library を使ったことを example の計数で示す。
- [ ] source の dump に入力値が含まれないことを確かめる。

予定する出力は順に `20` と `12`。
一般的なコンパイルキャッシュはまだ作らない。
定数グラフも従来の `7`、`20` を保つ。

### コミット 4. 呼出し前の検証と library の寿命を確認する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── src/
│   ├── runtime/
│   │   ├── cpu.rs  # [変更] ScalarSignature、呼出し前検証と library の解放。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── verify.rs  # [変更] 到達可能な PARAM slot と宣言数をコンパイル前に照合する。
├── tests/
│   └── scalar_inputs.rs  # [追加] 共有ライブラリを使う再実行、入力不足、symbol 不在の統合確認。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

ScalarSignature とその impl を runtime/cpu.rs に追加し、CompiledScalar の run は validate 成功後にだけ pack と入口呼出しを行う。verify.rs で集めた slot を from_slots に渡してコンパイル前に署名を確定する。tests/scalar_inputs.rs は公開 API で compile と run を通し、不足と余剰で入口が呼ばれないことを確かめる。library と一時ディレクトリの所有物は CompiledScalar の寿命へまとめる。


成功する FFI 呼出しの手前に、署名を使う検証を集中させる。実装先候補は `src/runtime/cpu.rs` の設計案 `ScalarSignature` と `validate_inputs` である。署名には必要な scalar 数と、各 PARAM の slot を記録する。この章の example は連続する slot 0〜2 を使い、run に渡す配列も長さ 3 を要求する。slot 3 を参照するのに署名が 3 個しかないグラフは compile 前、不足や余剰の値は run 前に拒否する。存在する値のアドレスを渡してから C の側で数を推測しない。

実装順はグラフから署名を作る検証、run の長さ検証、pack、出力確保、FFI とする。`[2,3]` が来た場合は最初の長さ比較で `expected 3 scalars` を返し、入口を一度も呼ばない。library path と symbol 名はロードエラーに残し、clang が source を処理できなかったエラーと区別する。出力用ホスト領域と引数配列の所有物は同期呼出しが戻るまで保持する。

局所テストは不足と余剰を一つずつ試し、入口呼出しの計数が変わらないことを確認する。symbol 名を変えた library も fixture にして、欠落した入口名がエラーに含まれることを調べる。破棄時には `CompiledScalar` が入口を使用できる状態で外へ残さず、library と一時ディレクトリをまとめて解放する。解放済みポインタを実行して失敗させるテストにはせず、安全な公開 API の所有関係を確認する。これで同じ compile/run の境界を次章の dtype 追加にも利用できる。


組込み用の抜粋である。`src/runtime/cpu.rs` に追加する署名検査を示す。`slots` は root から到達する PARAM の slot を DFS で集めた列であり、値の列ではない。

ScalarSignature とその検証メソッドを追加する。

```rust
// src/runtime/cpu.rs（既存ファイルへ組み込む抜粋）
struct ScalarSignature { count: usize }
impl ScalarSignature {
	fn from_slots(slots: &[usize], declared: usize) -> Result<Self, String> {
		let mut seen = vec![false; declared];
		for &slot in slots {
			let entry = seen.get_mut(slot).ok_or_else(|| format!("invalid slot {slot}"))?;
			*entry = true;
		}
		if seen.iter().any(|used| !used) {
			return Err("scalar slots must be contiguous".into());
		}
		Ok(Self { count: declared })
	}
	fn validate(&self, values: &[f32]) -> Result<(), String> {
		if values.len() != self.count { return Err(format!("expected {} scalars", self.count)); }
		Ok(())
	}
}
```

`get_mut` は slot=3 を declared=3 の配列へ入れる前に拒否する。使用済み表で連続スロットの契約も検査し、run は `validate` 成功後にだけ pack と FFI を行う。`&self` の借用中は library の所有物が解放されないので、公開するのはこの `run` であり裸の入口関数ではない。入力不足のテストはこの戻りエラーと入口呼出し数 0 を照合する。

この入力の途中経過を並べる。

```text
from_slots([0,1,2],3) -> signature.count=3
validate([2,3]) -> expected 3 scalars -> FFI calls=0
validate([2,3,4,5]) -> expected 3 scalars -> FFI calls=0
from_slots([0,3],3) -> invalid slot 3
```

- [ ] 不足、余剰、存在しないスロットを run 前に拒否する。
- [ ] symbol 不在は library path と入口名を含むエラーにする。
- [ ] compile 失敗と run の入力エラーを区別する。
- [ ] 所有物の破棄で library と一時ファイルを解放する。

入力が二つだけなら、FFI を呼ばず `expected 3 scalars` を返す予定にする。
ホストの出力領域は呼出しが戻るまで有効に保つ。
C 側は呼出し終了後の出力ポインタを保存しない。

## 完了確認

以下は新しく作る example とテストの予定コマンドである。
`--then` は二回目の入力を指定する予定の example 専用引数とする。

```sh
# 作業ディレクトリ: chibigrad/
cargo test
cargo run --example scalar_inputs -- 2 3 4 --then 5 1 2 --dump-source
cargo run --example scalar_inputs -- 2 3
```

- [ ] 成功例で `20`、`12` を得る。
- [ ] compile の予定計数が `1`、run が `2` になる。
- [ ] 生成 C は実行時 scalar スロットを参照する。
- [ ] 入力不足では FFI を実行しない。
- [ ] 前章の定数例も成功する。
- [ ] library の所有物を破棄した後に関数ポインタだけが公開されない。

一般の小数値の許容誤差は [03](03-uop-rewrites.md) で定義する。
この章は `f32` で正確に表せる値を使う。

## tinygrad ソースとの対応

参照 commit は `c22edff14c79310a9b2df62a78819cf77c8289fa`。
各リンクの行は、その checkout で確認した。

| 実在するシンボル | ソース | 役割 | chibigrad で簡略化する点 |
|---|---|---|---|
| `UOp.param` | [ops.py](../../tinygrad/tinygrad/uop/ops.py#L1184) | スロットと dtype を持つ入力 | `f32` scalar から始める |
| `UOp.param_like` | [ops.py](../../tinygrad/tinygrad/uop/ops.py#L1194) | 値と呼出し側の情報を分離 | 値を持たない PARAM を使う |
| `pack_args` | [codegen](../../tinygrad/tinygrad/codegen/__init__.py#L437) | scalar とポインタをまとめる | 独自の二配列 ABI に固定する |
| `CPUProgram.__call__` | [ops_cpu.py](../../tinygrad/tinygrad/runtime/ops_cpu.py#L46) | 引数を pack して呼ぶ | 同期呼出しにする |
| `CPUProgram._load` | [ops_cpu.py](../../tinygrad/tinygrad/runtime/ops_cpu.py#L18) | ELF などをロードする | 移植せず OS の共有 library を使う |

## 今は追加しないもの

dtype、入力の範囲、interpreter は [03](03-uop-rewrites.md) へ進める。
スコープと UIR は [04](04-functions-stack.md)、メモリ UOp は [05](05-memory-runtime.md) で加える。
JIT cache、独自 ELF ローダー、非同期 runtime は追加しない。
