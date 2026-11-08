# 17. ループ分割、unroll、vector、CPU thread を一つずつ動かす

## このステップで動かすもの

[16 の OpenCL](16-opencl.md) までで共通の演算を各 backend に通した後、明示指定の最適化を追加する。
元計画の第 10 章「最適化を一つずつ追加する」に対応する。
まず正しさと変換の可視化を完成させ、速くなったという主張は計測後に行う。

```text
y=x*2+1, x=[0,1,2,3,4,5,6,7,8,9]
基準出力=[1,3,5,7,9,11,13,15,17,19]
split(tile=4): outer=0..3, inner=0..4, i=outer*4+inner
i<10 のときだけ LOAD/STORE
unroll(inner=4): inner の4反復を静的に展開
vector(width=4): 連続する4要素を一つの値として扱う
CPU threads=3: 出力区間 [0,3),[3,6),[6,10) を別workerで実行
```

API、axis 名、example、test、コマンドは追加予定である。
`CPU_THREAD` または `THREAD` は chibigrad 側で導入する割当ての案である。
参照 commit の `AxisType` に同名の軸があるという意味ではない。

## 最小実装とデータ

最適化の入力は、rangeify 済みの RANGE と添字式を持つ kernel IR にする。
出力は同じ入出力契約を持つ別の kernel IR と launch または CPU task 情報とする。
元の純粋な Tensor graph は変えず、基準版を再コンパイルできるように保つ。
初版の変換指定は CLI とテストの固定値でよく、自動探索は要らない。

```text
入力IR: RANGE(i,10), out[i]=x[i]*2+1
split後: RANGE(o,3), RANGE(t,4), i=o*4+t, IF(i<10), ...
unroll後: i=o*4+0, ..., i=o*4+3 を各guard付きで実行
vector後: 本体は i=0,4 の float4、tailは i=8,9 のscalar
thread後: 同じkernelをbegin/end入力slotで別workerにbind
```

UPCAST は小さい静的な要素の束への展開として導入する。
unroll は命令複製、vector は型と LOAD/STORE の幅、CPU thread は runtime の並列実行である。
UPCAST を指定しただけで、SIMD 命令や CPU の複数 thread が使われるとは説明しない。
vector は連続、幅、alignment、mask、backend の対応型を検査する。
条件が成立しなければ scalar へ戻すか、明示指定を診断付きで拒否する。

CPU の初期並列化は出力 elementwise の互いに重ならない区間だけに限る。
固定 C ABI の関数型を変えず、区間 begin/end を既存の入力 slot に追加する。
worker ごとの引数表と区間メタデータは別に所有する。
入力は共有して読み、出力は一つの配列の異なる区間にだけ書く。
ロードした共有ライブラリを全 worker の join まで保持する。
GPU の LOCAL 軸を CPU thread pool と同一視しない。

## この章のファイル配置

明示的な IR 変換を新規 src/codegen/loop_opts.rs に集め、C の vector 表記は src/renderer/c.rs、CPU worker の実行は src/runtime/cpu.rs に置く。単独 main は参照例へ分け、予定 opt_* example で compiler 全体を確認する。

次はこの章の終わりまでに触る関連ファイルの配置案である。現在のcheckoutの一覧ではない。各コミットでは、その下に示す部分だけを追加・変更する。

```text
chibigrad/
├── examples/
│   ├── book_17_capability_ranges.rs  # 本文のコードを配置する。
│   ├── book_17_cpu_threads.rs  # 本文のコードを配置する。
│   ├── book_17_split_reference.rs  # 本文のコードを配置する。
│   ├── opt_compare.rs  # 基準、単体、組合せと fallback を表示する予定例。
│   ├── opt_cpu_threads.rs  # 共有ライブラリを区間実行する予定例。
│   ├── opt_split_tail.rs  # split 指定で既存 pipeline を実行する予定例。
│   ├── opt_unroll.rs  # unroll の予定実行例。
│   └── opt_vector.rs  # UPCAST と vector を区別して実行する予定例。
├── src/
│   ├── codegen/
│   │   ├── loop_opts.rs  # apply_opts と区間ごとの vector_ranges。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── renderer/
│   │   ├── c.rs  # 入力 i と出力 i-begin の区間 kernel。
│   │   ├── metal.rs  # 対応可能な vector 表記と scalar fallback。
│   │   ├── mod.rs  # vector/CPU worker/LOCAL/shared を分ける Capabilities。
│   │   └── opencl.rs  # 対応可能な vector 表記と scalar fallback。
│   ├── runtime/
│   │   ├── cpu.rs  # 区間ごとの実際の変換と実行情報。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── uop.rs  # 必要な vector 値/アクセス幅の型。
│   └── verify.rs  # 幅、dtype、連続性と mask の検証。
├── tests/
│   └── loop_opts.rs  # capability 拒否と変換単体/組合せの比較。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

生成Cやshaderは一時ディレクトリへ出力するため、このRustプロジェクトのツリーには含めない。コード冒頭のコメントで、生成元と生成先を確認する。

## 小さなコミットで進める

### 1. 端数を持つ loop split を実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_17_split_reference.rs  # 本文のコードを配置する。
│   └── opt_split_tail.rs  # [追加] split 指定で既存 pipeline を実行する予定例。
├── src/
│   ├── codegen/
│   │   ├── loop_opts.rs  # [追加] split_loop と添字/scope の置換。
│   │   └── mod.rs  # [更新] loop_opts モジュールの宣言。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── verify.rs  # [更新] 変換後 kernel の scope と境界。
├── tests/
│   └── loop_opts.rs  # [追加] split 前後の値と不正 tile。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/codegen/mod.rs に mod loop_opts を宣言し、src/codegen/loop_opts.rs に split_loop を追加する。kernel 検証は既存 src/verify.rs を更新する。掲載 split_map/main は book_17_split_reference へ保存し、その添字計算を本体では RANGE と UOp の置換表として実装する。

新しいモジュールは、既存の宣言を残して次を追記する。内部の補助関数は各モジュールに置き、exampleから使う入口だけを公開する。

```rust
// src/codegen/mod.rs（モジュール宣言を追加）
pub mod loop_opts;
```


このコミットはTensorの式を変えず、rangeify後の一つのRANGEを二つへ分ける。候補は `src/codegen/loop_opts.rs` の設計案 `split_loop(kernel,axis,tile)` とkernel verifierである。まずaxisが存在すること、tile>0、元の範囲Nと計算結果が添字型へ収まることを検証する。元の `i=0..N` を外側 `o=0..ceil(N/tile)`、内側 `t=0..tile` へ置き換え、元iの全利用位置を `o*tile+t` へ写す。この置換にはLOAD、STORE、比較、値の計算だけでなく、それらが依存するscope情報も含める。元のkernelは保持し、新しいIRを返して基準版を再実行できるようにする。

長さ10、tile=4なら外側は0,1,2の3反復になる。o=0ではi=0..3、o=1では4..7、o=2では8,9,10,11を作る。元の有効範囲 `i<10` を最後のtileだけに特別扱いせず全反復のguardとして置けば、i=8のx=8から17、i=9のx=9から19を作り、i=10,11はLOADもSTOREもしない。出力 `[1,3,5,7,9,11,13,15,17,19]` には欠落も重複も生じない。ceilの算出は `N/tile+(N%tile!=0)` とし、N+tile-1のoverflowを避ける。

新しいguardは値の計算だけを囲むものではなく、無効pointerを作ってdereferenceするLOADを支配する制御条件にする。pad等の入力valid条件が元にあれば、iの置換後もその条件を残す。外側と内側のRANGEには新しい識別子を割り当て、元iをscope外に残さない。設計案の変換memoはNodeIdだけでなく置換表とscopeを区別し、別のtileの式を同じ変数へ誤ってつながない。

局所テストはN=1,3,4,5,10,257で未最適化版と生成Cの全要素を比較する。N=0は既存の空出力規約を再利用し、tile=0や添字overflowはcompile前のエラーとする。kernel dumpではiの式、outerの切上げ範囲、tail guard、ENDの対応を見る。単に値が同じだけでなく、最後の無効位置へアクセスしないことまで確認してから、次のunrollへ進む。


`src/codegen/loop_opts.rs` の split が保存する添字を、standalone な整数 loop で確かめる。中核は outer と inner の二つから元の i を復元し、全アクセスを guard する部分である。

単独で実行できる例。

整数 loop による split の数値参照と main 全体。

```rust
// examples/book_17_split_reference.rs（単独で実行する例）
fn split_map(x: &[f32], tile: usize) -> Vec<f32> {
	assert!(tile>0);
	let outer_count = x.len()/tile + usize::from(x.len()%tile!=0);
	assert!(outer_count.checked_mul(tile).is_some());
	let mut out = vec![f32::NAN;x.len()];
	for outer in 0..outer_count {
		for inner in 0..tile {
			let i = outer*tile+inner;
			if i<x.len() {
				out[i] = x[i]*2.0+1.0;
			}
		}
	}
	out
}
fn main() {
	for n in [0,1,3,4,5,10,257] {
		let x: Vec<_> = (0..n).map(|i|i as f32).collect();
		let reference: Vec<_> = x.iter().map(|v|v*2.0+1.0).collect();
		assert_eq!(split_map(&x,4),reference);
	}
}
```

`outer_count` は切上げで、積の overflow を先に拒否する。i が範囲外でも値を0へ置き換えるのではなく、LOAD と STORE の両方を実行しない。この本体の i を UOp の置換式として用いる。

同じ入力を追う例。

```text
N=10,tile=4
outer=2,inner=0 -> i=8 -> 17
outer=2,inner=1 -> i=9 -> 19
outer=2,inner=2 -> i=10 -> skip
outer=2,inner=3 -> i=11 -> skip
```

元の loop と同じ10個の出力だけを一回ずつ書く。新規 RANGE の scope とこの guard を合わせて kernel verifier へ渡す。

- [ ] `i=outer*tile+inner` への置換を kernel IR に追加する。
- [ ] outer の範囲を切上げ、元範囲の guard を作る。
- [ ] 無効な要素の LOAD と STORE を guard の内側に置く。
- [ ] tile=0、負値、添字の overflow を起動前に拒否する。
- [ ] split 前と split 後を同じ入力で実行する。

予定例 `opt_split_tail` は長さ 10、tile=4 で基準出力と一致する。
長さ 1、3、4、5、257 でも値を照合する。
dump で最後の tile の無効 2 要素がロードされないことを確認する。

### 2. 安全な loop interchange と小さい unroll を実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── opt_unroll.rs  # [追加] unroll の予定実行例。
├── src/
│   ├── codegen/
│   │   ├── loop_opts.rs  # [更新] interchange_outputs と unroll。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── renderer/
│   │   ├── c.rs  # [更新] 展開済み IR の C 出力。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── verify.rs  # [更新] 複製後の定義利用と RANGE/END。
├── tests/
│   └── loop_opts.rs  # [更新] 独立出力の並替えと split+unroll。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/codegen/loop_opts.rs に interchange_outputs と unroll を追加し、複製先ごとの NodeId/scope の写像を作る。src/verify.rs で各複製の定義と利用を検証する。掲載 C は src/renderer/c.rs が変換済み IR を出す目標形で、Rust ソースへ関数として追加しない。


loop interchangeは独立した出力軸の実行順だけを変え、unrollは小さいloopの本体を定数回複製する。候補はloop最適化モジュールの設計案 `interchange_outputs(kernel,a,b)` と `unroll(kernel,axis,count)` である。初版は出力elementwiseの軸だけに適用し、STOREの保存位置が各座標で一意で、入力が読取専用であることを確認する。reduction軸の順序変更や、異なる反復間に状態依存があるloopは拒否する。浮動小数点の加算順を変える最適化を、この独立出力の並替えへ紛れ込ませない。

2×3入力へ1を足す例では、元はrを外、cを内として位置0,1,2,3,4,5へ書く。軸を入れ替えると `(r,c)=(0,0),(1,0),(0,1),(1,1),(0,2),(1,2)` の順で書くが、保存位置は常にr*3+cである。したがって訪問順が変わっても結果は `[[2,3,4],[5,6,7]]` になる。STOREの添字を新しいloop順の連番に変えると転置した値を返してしまうので、元の座標式の意味を保つ。

tile=4のinnerをunrollするときはt=0,1,2,3の各定数へ置換し、本体を四つ作る。o=2の複製はi=8,9,10,11であり、guardが残るので有効な二つだけが17,19を保存する。各本体の一時値や局所REGには別の識別子とscopeを与える。外側にあるaccや引数は必要な共有を保ち、inner内にある一時変数だけを複製する。変数名を文字列で置き換えるのではなく、元RANGEとローカル定義の写像を複製先ごとに持つ。

テストはinterchange単独、unroll単独、split後unrollの順に比較する。kernel verifierでRANGEとENDが一対一で閉じ、定義前利用やscope外参照がないことを検査する。source dumpでは四つの定数座標とそれぞれのtail guardを確認する。reductionの外側の独立出力軸へ適用する場合も、acc初期化が各出力の縮約loop直前に一度だけ残る必要がある。これで17の基準kernelと同じcompiler/runtimeを使ったまま、命令複製の意味を追えるようになる。


unroll 後の `src/renderer/c.rs` の出力を示す。inner の四反復を別のローカル変数へ複製し、それぞれに元の有効条件を残す。

組込み用の抜粋。

unroll 後 IR の生成 C。

```c
// src/renderer/c.rs が生成するコード。生成先: <一時ディレクトリ>/unrolled.c
void unrolled(const float *x, float *out, int n) {
	int tiles = n/4 + (n%4!=0);
	for (int outer=0; outer<tiles; outer++) {
		int i0 = outer*4+0;
		if (i0<n) { float v0=x[i0]; out[i0]=v0*2.0f+1.0f; }
		int i1 = outer*4+1;
		if (i1<n) { float v1=x[i1]; out[i1]=v1*2.0f+1.0f; }
		int i2 = outer*4+2;
		if (i2<n) { float v2=x[i2]; out[i2]=v2*2.0f+1.0f; }
		int i3 = outer*4+3;
		if (i3<n) { float v3=x[i3]; out[i3]=v3*2.0f+1.0f; }
	}
}
```

i0〜i3 と v0〜v3 は複製先ごとに別の定義である。IR でも文字列の名前置換だけにせず、各元 NodeId から複製先 NodeId への表を scope ごとに作る。外側の引数と outer は共有できる。

同じ入力を追う例。

```text
interchange before store order=[0,1,2,3,4,5]
interchange after store order =[0,3,1,4,2,5]
store index always r*3+c
unroll outer=2: i0=8,i1=9 active; i2=10,i3=11 inactive
```

interchange は訪問順を変え、unroll は定数座標で本体を複製する。どちらも出力添字を新しい訪問順の連番へ変更しない。

- [ ] 独立した出力軸の順序を入れ替える変換を追加する。
- [ ] reduction 軸の再順序化は初版では明示的に禁止する。
- [ ] 小さい inner RANGE を定数座標ごとに複製する。
- [ ] RANGE とローカル変数を複製先の scope ごとに作り直す。
- [ ] 展開後にも tail guard と累積の初期化位置を維持する。

予定例 `opt_unroll` は tile=4 の内側を展開し、同じ `[1,3,...,19]` を返す。
二次元入力 `[[1,2,3],[4,5,6]]` に 1 を足した出力は loop 順序を替えても `[[2,3,4],[5,6,7]]` になる。
共有 NodeId の式を別 scope に誤って再利用しない。

### 3. UPCAST と vector を別々に実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── opt_vector.rs  # [追加] UPCAST と vector を区別して実行する予定例。
├── src/
│   ├── codegen/
│   │   ├── loop_opts.rs  # [更新] expand_upcast と pack_vector_accesses の合法性。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── renderer/
│   │   ├── c.rs  # [更新] Clang vector 型、memcpy と scalar tail。
│   │   ├── metal.rs  # [更新] 対応可能な vector 表記と scalar fallback。
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── opencl.rs  # [更新] 対応可能な vector 表記と scalar fallback。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── uop.rs  # [更新] 必要な vector 値/アクセス幅の型。
│   └── verify.rs  # [更新] 幅、dtype、連続性と mask の検証。
├── tests/
│   └── loop_opts.rs  # [更新] width4 と不完全 vector、非整列アクセス。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/codegen/loop_opts.rs に scalar 展開の expand_upcast とアクセスを束ねる pack_vector_accesses を別関数として足す。src/uop.rs/verify.rs に必要な幅と型を追加し、各 renderer の対応幅を確認する。掲載 C は src/renderer/c.rs が作る memcpy による安全な vector 本体と scalar tail の出力例である。


UPCASTとvectorは異なる変換として実装する。候補はloop最適化モジュールの設計案 `expand_upcast(axis,width)`、`pack_vector_accesses(width)` とrendererのvector型である。UPCASTは長さの小さい静的軸を固定個数のscalar式へ展開し、例えばwidth=4なら値v0,v1,v2,v3を別の値として持つ。vector化は、その四つが連続した同種の演算とアクセスであると確認した後、一つの四要素型へまとめる処理である。UPCASTだけを指定したdumpでscalarが四つ残っていても、それ自体は仕様どおりとする。

長さ10の `x*2+1` はfull vectorの本体範囲を `floor(10/4)*4=8` と決め、i=0,4で四要素ずつ処理する。i=4ならLOADは `[4,5,6,7]`、乗算で `[8,10,12,14]`、加算で `[9,11,13,15]` になり、その四つを出力4..7へ保存する。i=8,9はscalar tailとして17,19を作る。残り二要素を四要素LOADしてからmaskで隠す方法は、先に範囲外を読むので認めない。PADで有効性がlaneごとに異なるアクセスも、最初はscalarへ戻す。

合法性は連続添字、dtype、width、backend対応、bufferの残長とalignmentを確認する。生成Cでfloat pointerをvector pointerへ強制castするだけでは、alignmentやaliasingの条件を満たさない可能性がある。初版は対応vector型の局所変数へmemcpyで16byteを移す方法、またはbackendが保証する非整列LOADを選び、必要条件をrendererで明示する。STOREも同じ長さと境界を検査する。対応しないbackendはscalar版を生成するか、明示指定なら理由付きで拒否する。

局所テストはwidth4とN=1,3,4,5,10で基準版の値と比較し、非整列の開始位置を持つ入力も調べる。source dumpではvector本体と二要素のtail、UPCASTだけの版では四つのscalar式が見えることを確認する。変換後にSIMD assemblyが出たという報告は別の調査事項とし、ここでは正しいvector型と安全なメモリアクセスを生成できたかだけを完成条件にする。


vector 化の本体は `src/renderer/c.rs` で明示する。次は Clang の四要素 vector 型を使う生成 C で、入力と出力は重ならない別 allocation とする。

組込み用の抜粋。

幅4の vector と scalar tail を出す C renderer の生成例。

```c
// src/renderer/c.rs が生成するコード。生成先: <一時ディレクトリ>/vector_map.c
#include <stddef.h>
#include <string.h>
typedef float float4 __attribute__((ext_vector_type(4)));
void vector_map(const float *x, float *out, size_t n) {
	size_t full = n/4*4;
	for (size_t i=0; i<full; i+=4) {
		float4 v;
		memcpy(&v,x+i,sizeof(v));
		float4 result = v*2.0f+1.0f;
		memcpy(out+i,&result,sizeof(result));
	}
	for (size_t i=full; i<n; i++) {
		out[i] = x[i]*2.0f+1.0f;
	}
}
```

`full` は完全な四要素だけを含む範囲である。memcpy は vector pointer の alignment や aliasing を入力 pointer へ要求しない。幅4の LOAD は必ず full の内側へ収まり、残りは scalar で読む。

同じ入力を追う例。

```text
UPCAST only: v0=x[i],v1=x[i+1],v2=x[i+2],v3=x[i+3]
vector width4: v=LOAD4(x,i), result=v*2+1
N=10: vector offsets=[0,4], scalar offsets=[8,9]
i=4: [4,5,6,7] -> [9,11,13,15]
```

UPCAST は四つの scalar を作る段階で、vector 化はアクセスの連続性を確認してそれを一型へまとめる段階である。生成 assembly の SIMD 化はこの source の検証とは別に扱う。

- [ ] UPCAST 軸を固定個数の scalar 値へ展開する。
- [ ] 連続した width=4 の f32 を vector 型へまとめる変換を別に追加する。
- [ ] vector LOAD/STORE の合法性を調べ、misalignment には安全な方法を使う。
- [ ] 端数は scalar loop に残し、full vector の範囲外 LOAD を避ける。
- [ ] vector 非対応 backend は scalar へ戻す。

予定例 `opt_vector` は長さ 10 で 8 要素を vector 本体、2 要素を scalar tail にする。
最終値は基準出力と一致し、生成 C または shader に vector 型と tail が見える。
clang が最終的に SIMD を出したかは、生成 assembly を調べた場合だけ報告する。

### 4. CPU の出力を複数 worker に分けて実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_17_cpu_threads.rs  # 本文のコードを配置する。
│   └── opt_cpu_threads.rs  # [追加] 共有ライブラリを区間実行する予定例。
├── src/
│   ├── codegen/
│   │   ├── loop_opts.rs  # [更新] begin/end slot を使う CPU 区間 lowering。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── renderer/
│   │   ├── c.rs  # [更新] 入力 i と出力 i-begin の区間 kernel。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── runtime/
│   │   ├── cpu.rs  # [更新] run_partitioned、非重複出力 slice と全 worker の join。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   └── lib.rs  # 本体のモジュール宣言と公開APIの入口。
├── tests/
│   └── loop_opts.rs  # [更新] worker 区間、番兵、失敗と空区間。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/runtime/cpu.rs に run_partitioned を追加し、compile/load 後に出力を split_at_mut で worker ごとへ渡す。区間 begin/end の lowering は src/codegen/loop_opts.rs に置き、C の出力添字を i-begin にする。掲載 threaded_map/main は book_17_cpu_threads へ保存する所有権の参照例であり、統合版では worker 内で既存の固定 ABI を呼ぶ。


CPUの並列化はcompiler内でloopを同時実行する処理ではなく、compile済みelementwise kernelを互いに重ならない出力区間で呼ぶruntimeの変更にする。候補は `src/runtime/cpu.rs` の設計案 `run_partitioned(plan,bindings,threads)` とCPU用の区間loweringである。対象は読取専用入力と一つの新規elementwise出力に限定し、reductionのaccや一つの可変状態をworker間で共有するkernelは拒否する。固定C ABIは変えず、begin/endを既存のscalars slotへ追加して区間を指定する。

N=10,T=3なら境界 `floor(t*N/T)` により `[0,3),[3,6),[6,10)` を作る。worker1は入力のi=3,4,5から7,9,11だけを担当し、他のworkerの位置へ書かない。出力配列はRust側で非重複の可変sliceへ分割し、各workerのbuffers表には自分のsliceの先頭pointerを渡せる。設計案の区間kernelでは入力添字はグローバルi、出力slice内の添字は `i-begin` として、このpointerの意味を明示する。全workerが出力全体への可変参照を持つ方式や、raw pointerを整数へ変換してSend検査を回避する方式は採らない。

compileとlibraryのloadはworker起動前に済ませる。共有するのは不変な実行物と読取専用入力であり、buffers/scalarsの引数表と区間値はworkerごとに別所有にする。libraryと入力配列を全joinまで保持し、FFIは検証済み区間だけを書くunsafeの小さい境界へ閉じる。初版はscoped threadでもよく、今必要のない常駐poolを先に作らない。workerのpanicや実行失敗があっても全workerをjoinしてから失敗を返し、部分的に埋まった出力を成功値にしない。

テストでは全出力を番兵値にし、区間の和集合が0..Nで交差がないことを別に確認する。T=1,2,3,Nより大きい値を比較し、空区間には呼出しを行わない。T=0と境界算出のoverflowは起動前に拒否する。生成Cでbegin/endが値slotから来ること、run後に全要素が基準出力へ一致することを調べれば、同じ計算の安全な所有範囲を実行結果とともに確認できる。


`src/runtime/cpu.rs` の出力区間の所有を、std の scoped thread で確認する。C ABI を呼ぶ前にも同じ分割を行い、worker ごとの slice と引数表を持つ。

単独で実行できる例。

scoped thread の所有区間を確認する単独プログラム全体。

```rust
// examples/book_17_cpu_threads.rs（単独で実行する例）
fn threaded_map(x: &[f32], threads: usize) -> Vec<f32> {
	assert!(threads>0);
	let n = x.len();
	let boundary = |t: usize| -> usize {
		((t as u128*n as u128)/threads as u128) as usize
	};
	let mut out = vec![f32::NAN;n];
	std::thread::scope(|scope| {
		let mut remaining = out.as_mut_slice();
		let mut workers = Vec::new();
		for t in 0..threads {
			let begin = boundary(t);
			let end = boundary(t+1);
			let (owned,rest) = remaining.split_at_mut(end-begin);
			remaining = rest;
			if owned.is_empty() { continue; }
			workers.push(scope.spawn(move || {
				for (offset,dst) in owned.iter_mut().enumerate() {
					let i = begin+offset;
					*dst = x[i]*2.0+1.0;
				}
			}));
		}
		let failed = workers.into_iter().map(|w|w.join().is_err()).fold(false,|a,b|a|b);
		assert!(!failed,"worker failed");
	});
	out
}
fn main() {
	let x: Vec<_> = (0..10).map(|i|i as f32).collect();
	for t in [1,2,3,20] { assert_eq!(threaded_map(&x,t),[1.,3.,5.,7.,9.,11.,13.,15.,17.,19.]); }
	assert!(threaded_map(&[],3).is_empty());
}
```

境界の乗算は u128 で行い、usize の t*N の overflow を避ける。`split_at_mut` で切り取った owned はその worker にだけ渡す。全 join を実行した後で失敗を返すため、ある worker の panic で他の実行中の slice を破棄しない。

同じ入力を追う例。

```text
N=10,T=3 -> boundaries=[0,3,6,10]
worker0 output slice length3, input indices0..3
worker1 output slice length3, input indices3..6
worker2 output slice length4, input indices6..10
C ABI: scalars[begin_slot]=begin,scalars[end_slot]=end
C STORE: owned_out[i-begin]
```

Rust の単独例は実際に3 worker の disjoint slice を動かす。共有ライブラリ経路では同じ所有区間で関数を呼び、library と入力を join が終わるまで保持する。

- [ ] CPU runtime に固定数の worker と join を追加する。
- [ ] 出力区間を重複と欠落がないように分割する。
- [ ] 区間の begin/end と引数表を worker ごとに持つ。
- [ ] worker が同じ compiler やキャッシュへ同時に書かないよう、compile を先に完了する。
- [ ] thread 数 1、2、3、出力数より多い場合を比較する。

予定例 `opt_cpu_threads` は threads=3 でも基準出力と一致する。
全要素を番兵値で初期化し、各要素がちょうど一つの担当区間に属することを検査する。
途中の worker の失敗は join 後に返し、部分出力を成功値として扱わない。

### 5. capability と変換ごとの比較をそろえる

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_17_capability_ranges.rs  # 本文のコードを配置する。
│   └── opt_compare.rs  # [追加] 基準、単体、組合せと fallback を表示する予定例。
├── src/
│   ├── codegen/
│   │   ├── loop_opts.rs  # [更新] apply_opts と区間ごとの vector_ranges。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── renderer/
│   │   └── mod.rs  # [更新] vector/CPU worker/LOCAL/shared を分ける Capabilities。
│   ├── runtime/
│   │   ├── cpu.rs  # [更新] 区間ごとの実際の変換と実行情報。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   └── lib.rs  # 本体のモジュール宣言と公開APIの入口。
├── tests/
│   └── loop_opts.rs  # [更新] capability 拒否と変換単体/組合せの比較。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

Capabilities の本体を src/renderer/mod.rs に置き、src/codegen/loop_opts.rs の apply_opts と vector_ranges が参照する。掲載の小さい Capabilities/main は book_17_capability_ranges の参照例に分ける。実装版は GPU LOCAL/shared と CPU worker を別項目にし、examples/opt_compare.rs から要求と適用結果を並べて確認する。


このコミットは新たな最適化を増やすより、変換を適用できる条件と比較結果を一つの入口で確認できるようにする。候補はbackend capabilityの型、loop最適化の検証、予定example `opt_compare` である。設計案の `Capabilities` はvectorの対応幅、CPU worker利用可否、GPU LOCALとshared memoryの利用可否を別の項目にする。`apply_opts(base,requested,capabilities)` は指定順に合法性を調べ、適用後IRと実際に適用した変換の列を返す。GPUのLOCAL対応だけを理由にCPU thread数を受け付けず、未対応指定を黙って別の意味へ置き換えない。

比較は同じ入力x=0..9と同じ元graphから基準Planを作り、splitだけ、unrollだけ、vectorだけ、threadだけを順に実行する。どの版も出力 `[1,3,5,7,9,11,13,15,17,19]` と全位置で一致することを確かめる。その後split(tile4)+unroll4、vector4+thread等の組合せを作る。threadとvectorを組み合わせる場合は、各workerの区間長と開始位置でvector本体とtailを再計算する。全体N=10で安全な四要素LOADがworker境界を越えてよいとは限らないので、区間を独立した安全境界として扱う。

例えば区間 `[3,6)` は長さ3しかなく、width4のfull vectorは存在しない。ここは全てscalarで7,9,11を返す。区間 `[6,10)` は四要素を一つにできても、開始位置のalignment条件を満たさなければ非整列の安全な方法かscalarを選ぶ。この小例を加えると、変換単体では正しいが組合せで壊れるアクセスを検出できる。出力dumpには要求した変換と実際のfallback理由を分けて表示する。

局所検証はcapability不足の拒否とfallback、基準版との全要素比較、scopeとbuffer境界のverifierを含む。計測点はcompile、転送、起動、実行、同期を混同しない位置に置くが、このコミットだけでspeedupを主張しない。時間を記録する場合は同じ入力と同じ同期範囲を使い、warmupと反復数を明記する。変換の正しさを先に比較できる入口を残すことで、次の並列reductionやtile化の基準を失わずに進める。


vector と thread の組合せでは worker の区間が新しいアクセス境界になる。次の `src/codegen/loop_opts.rs` 用の計画関数は能力不足を理由付きで返し、適用する幅を明示する。

単独で実行できる例。

簡略 Capabilities と区間計画の単独検証。実装本体の型へ重複追加しない。

```rust
// examples/book_17_capability_ranges.rs（単独で実行する例）
struct Capabilities { vector4: bool, cpu_threads: bool }
fn vector_ranges(begin: usize, end: usize, width: usize,
                 threaded: bool, caps: &Capabilities) -> Result<(usize,usize),String> {
	if begin>end { return Err("invalid interval".into()); }
	if threaded && !caps.cpu_threads { return Err("CPU threads unsupported".into()); }
	if width!=1 && width!=4 { return Err("unsupported vector width".into()); }
	if width==4 && !caps.vector4 { return Err("vector4 unsupported".into()); }
	let count = end-begin;
	let full_count = count/width*width;
	let tail_begin = begin+full_count;
	Ok((tail_begin,end))
}
fn main() {
	let cpu = Capabilities { vector4:true,cpu_threads:true };
	assert_eq!(vector_ranges(3,6,4,true,&cpu).unwrap(),(3,6));
	assert_eq!(vector_ranges(6,10,4,true,&cpu).unwrap(),(10,10));
	let gpu = Capabilities { vector4:true,cpu_threads:false };
	assert!(vector_ranges(0,10,4,true,&gpu).is_err());
}
```

返り値は scalar tail の半開区間である。begin=3,end=6 は三要素しかなく vector 本体は空になる。begin=6,end=10 は四要素を持つが、renderer はさらに alignment と mask の合法性を検査する。

同じ入力を追う例。

```text
requested vector4+threads3, N=10
worker [0,3): vector=[], tail=[0,3)
worker [3,6): vector=[], tail=[3,6)
worker [6,10): vector=[6,10), tail=[]
GPU LOCAL capability does not imply cpu_threads=true
```

この計画を基準版、各単体、組合せ版の出力比較と一緒に dump する。能力不足を黙って別の最適化へ置き換えず、fallback を選ぶ場合は実際の幅も表示する。

- [ ] backend ごとに vector 幅、CPU thread、LOCAL、shared memory の可否を明示する。
- [ ] 未対応変換を黙って別の意味で実行しない。
- [ ] split、unroll、vector、thread を単独で検証した後に組み合わせる。
- [ ] compile、転送、実行、同期を別に測れる計測点だけを追加する。

予定例 `opt_compare` は各変換の出力と基準との差を表示する。
固定値の elementwise 例では全要素一致を要求する。
実行時間を記録する場合は warmup、反復数、同期地点、機器、変換指定を添える。
この文書には未測定の speedup を書かない。

## 完了確認

以下は将来追加する例と `tests/loop_opts.rs` の予定コマンドである。

```sh
# 作業ディレクトリ: chibigrad/
cargo test
cargo run --example opt_split_tail -- --size 10 --tile 4 --dump-kernel
cargo run --example opt_unroll -- --tile 4 --dump-source
cargo run --example opt_vector -- --width 4 --size 10
cargo run --example opt_cpu_threads -- --threads 3 --size 10
cargo run --example opt_compare -- --backend cpu --check-all
```

- [ ] split と unroll の後にも scope と buffer 境界の verifier が通る。
- [ ] 未対応 vector の fallback と CPU thread 数 1 が基準版と一致する。
- [ ] 出力ごとの所有区間が明確で、reduction の累積を共有しない。
- [ ] GPU local と CPU worker の capability を独立して表示する。

## tinygrad ソースとの対応

参照 commit は `c22edff14c79310a9b2df62a78819cf77c8289fa` である。

| 実在するシンボル | ソース | 役割 | chibigrad での簡略化 |
|---|---|---|---|
| `Scheduler.shift_to` | [postrange.py](../../tinygrad/tinygrad/codegen/opt/postrange.py#L65) | 軸を分割し元添字を置き換える | 割り切れない場合の guard を最初から扱う |
| `Scheduler.apply_opt` | [postrange.py](../../tinygrad/tinygrad/codegen/opt/postrange.py#L87) | SPLIT などの指定と capability を検証する | 自動探索を持たず明示指定にする |
| `AxisType` | [ops.py](../../tinygrad/tinygrad/uop/ops.py#L18) | GLOBAL、LOCAL、UPCAST などの軸の区別 | CPU_THREAD は chibigrad の独自計画として追加する |
| `build_range_map` と `expander` | [codegen/__init__.py](../../tinygrad/tinygrad/codegen/__init__.py#L42) | UPCAST の軸を固定個数の値へ展開する | 小さな scalar unroll から始める |
| `Renderer` の capability | [renderer/__init__.py](../../tinygrad/tinygrad/renderer/__init__.py#L69) | float4、local、shared の可否を持つ | CPU worker を別の capability にする |
| `ClangRenderer` | [cstyle.py](../../tinygrad/tinygrad/renderer/cstyle.py#L288) | C vector 型を持ち has_local=False とする | thread pool は runtime 側に独立して実装する |

## 今は追加しないもの

BEAM 探索、CPU の並列 reduction、浮動小数点の積極的な再結合は後の拡張にする。
GPU の並列 reduction は [18](18-parallel-reduction.md)、tiled GEMM と conv は [19](19-fast-gemm-conv.md) へ進める。
速度より先に、各変換の値とメモリアクセスの意味を確認する。
