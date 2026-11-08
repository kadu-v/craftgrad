# 12. 数学関数を実行し、近似と値域の責任を分ける

## このステップで動かすもの

[11 の Tensor](11-tensor-gemm-conv.md) に `sqrt`、`sin`、`log2`、`exp2` を追加する。
元計画の第 8 章の微分と第 11 章の学習モデルに必要な数学関数を補う段階である。
最初は CPU の libm を呼び、次に範囲を絞った近似を compiler 内で作る。

```text
sqrt([0,1,4,9]) = [0,1,2,3]
sin([0,π/2,π]) ≈ [0,1,0]
log2([0.5,1,2,8]) = [-1,0,1,3]
exp2([-1,0,1,3]) = [0.5,1,2,8]
exp([0, ln(2)]) ≈ [1,2]
```

π は実行時に渡す `f32` の値であり、sin(π) の出力に厳密な 0 を要求しない。
以下の API、近似モード、example、誤差目標、コマンドは追加予定である。
ここで記す誤差は実測済みの性能や精度ではなく、実装後に検証する基準である。

## 最小実装とデータ

高水準 IR に `SQRT`、`SIN`、`LOG2`、`EXP2` を加える。
入力と出力は f32 の同形 tensor にする。
interpreter はホストの対応関数を使い、CPU renderer は `<math.h>` の関数へ下げる。

```text
入力IR: EXP2(PARAM(x:[4]))
kernel IR: RANGE(i,4), LOAD(x,i), EXP2, STORE(out,i), END
生成C初版: out[i] = exp2f(x[i]);
生成C近似版: out[i] = scale_pow2(poly(x[i]-round(x[i])), round(x[i]));
```

clang の共有ライブラリ作成では libm が必要な環境の link flag を runtime で扱う。
`-ffast-math` はこのステップの既定値にしない。
`exp(x)=exp2(x*log2(e))` と `log(x)=log2(x)*ln(2)` は合成で作る。
負の平方根と負の log2 は NaN、log2(±0) は -Inf と定める。
exp2 の overflow は +Inf、underflow は 0 または subnormal と定める。
NaN の payload の一致までは求めない。

近似は `libm` と明示的な `bounded-poly` の二つを選べる案にする。
初版の近似対象は sin の有限な `|x|<=32`、exp2 の `-16<=x<=16`、log2 の正規数に絞る。
対象外の入力は libm へ分岐し、巨大値の sin を不十分な range reduction に通さない。
GPU に移す際は backend の native 関数を対象外値の経路にできる。
全 f32 で libm を不要にするなら、巨大値の引数還元を別の最小例で完成させる。

## この章のファイル配置

数学 UOp の定義・検証・評価・C 出力を既存モジュールへ加える。近似展開だけを新規 src/codegen/math.rs に置き、数値検証の main は examples/book_12_*.rs に分ける。

次はこの章の終わりまでに触る関連ファイルの配置案である。現在のcheckoutの一覧ではない。各コミットでは、その下に示す部分だけを追加・変更する。

```text
chibigrad/
├── examples/
│   ├── book_12_exp_log_poly.rs  # 本文のコードを配置する。
│   ├── book_12_sin_poly.rs  # 本文のコードを配置する。
│   ├── math_exp_log_poly.rs  # 近似と libm を比べる予定例。
│   ├── math_libm.rs  # libm モードの予定実行例。
│   ├── math_sin_poly.rs  # sin 近似を libm と比べる予定例。
│   └── math_sqrt.rs  # SQRT を CPU Plan で動かす予定例。
├── src/
│   ├── codegen/
│   │   ├── math.rs  # decompose_sin と bounded な引数還元。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── renderer/
│   │   ├── c.rs  # sinf/log2f/exp2f の出力。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── runtime/
│   │   ├── cpu.rs  # 環境に必要な libm の link 引数と診断。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── interpret.rs  # 各ホスト数学関数。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rangeify.rs  # 同形 elementwise としての lowering。
│   ├── tensor.rs  # sin/log2/exp2 と自然指数・対数の合成。
│   ├── uop.rs  # 丸め、bitcast、shift が必要な場合の型付き定義。
│   └── verify.rs  # 新規整数/bit 演算の検証。
├── tests/
│   └── math_functions.rs  # π付近、±32、巨大値と特殊値。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

生成Cやshaderは一時ディレクトリへ出力するため、このRustプロジェクトのツリーには含めない。コード冒頭のコメントで、生成元と生成先を確認する。

## 小さなコミットで進める

### 1. sqrt と libm の接続を実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── math_sqrt.rs  # [追加] SQRT を CPU Plan で動かす予定例。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # [更新] math.h と sqrtf の生成。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── runtime/
│   │   ├── cpu.rs  # [更新] 環境に必要な libm の link 引数と診断。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── interpret.rs  # [更新] ホスト sqrt による比較。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rangeify.rs  # [更新] SQRT の elementwise lowering。
│   ├── tensor.rs  # [更新] sqrt の入口。
│   ├── uop.rs  # [更新] SQRT の Op と builder。
│   └── verify.rs  # [更新] 入力数と f32 の検証。
├── tests/
│   └── math_functions.rs  # [追加] 通常値と特殊値分類。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/uop.rs/verify.rs に SQRT を足し、src/interpret.rs と src/rangeify.rs の既存演算分岐を更新する。src/tensor.rs の sqrt はノードだけを作る。掲載 C は src/renderer/c.rs が出力し、sqrtf の解決に必要な link 引数は src/runtime/cpu.rs の共有ライブラリ compile 処理へ追加する。


ここでは SQRT という高水準の意味を一つ加え、数値計算そのものはホストと C の数学ライブラリへ任せる。候補は UOp 定義と verifier、interpreter、C renderer、共有ライブラリを作る compiler 呼出しである。設計案の `sqrt(node)` は f32 の入力一つを要求し、入力と同じ shape を返す。rangeify は座標ごとの入力値を一つ読んで SQRT を作るだけで、別のループ方式を増やさない。C では `<math.h>` を出力し、`sqrtf` の宣言と共有ライブラリからの解決をそろえる。必要な環境では link 引数に `-lm` を加え、全 backend 共通の引数へ無条件に混ぜない。

入力 `[0,1,4,9]` の座標2では LOAD から f32 の4を得て、`sqrtf(4)` が2を返し、出力位置2へ保存する。`sqrt(x)+1` を同時に compile すれば、同じ座標で sqrt の後に ADD を置き、`[1,2,3,4]` を一つの出力経路で作れる。Tensor のメソッド内で host 計算してから定数ノードを作ると、入力の差替えに追従しなくなるので、別の `[1,4,9,16]` をこの `sqrt(x)+1` の Plan へ bind して `[2,3,4,5]` になるかも確認する。

特殊値の判定は誤差計算へ入る前に行う。負の有限値は NaN、+Inf は +Inf、NaN は NaN、-0 は符号付きの-0を保つものとし、NaN の payload の一致は求めない。`abs(actual-reference)` では NaN が常に比較不能になるため、通常値の誤差と特殊値の分類をテスト側で分ける。-0 は数値比較だけなら+0と等しいので、bits の符号を別に調べる。`-ffast-math` はこれらの意味を変え得るため初版では有効にしない。

`tests/math_functions.rs` では同じ入力を interpreter と生成 C に渡し、整数平方の厳密値、-0の bit、負値の分類を確認する。共有ライブラリの未解決 symbol は runtime の一般実行失敗ではなく、compile/link の診断として source とともに返す。SQRT のための独立した数値実行器は追加せず、既存の LOAD、elementwise、STORE と固定 ABI を最後まで通す。


`src/renderer/c.rs` は SQRT を次の本体へ出力する。固定 ABI の buffers[0] が入力、buffers[1] が新規出力、scalars[0] の下位32bitが長さ N である。

組込み用の抜粋。

SQRT+ADD を固定 ABI の C へ出す例。

```c
// src/renderer/c.rs が生成するコード。生成先: <一時ディレクトリ>/math_sqrt.c
#include <math.h>
#include <stdint.h>
void chibi_entry(void **buffers, const uint64_t *scalars) {
	const float *x = (const float *)buffers[0];
	float *out = (float *)buffers[1];
	uint32_t n = (uint32_t)scalars[0];
	for (uint32_t i=0; i<n; i++) {
		float loaded = x[i];
		float root = sqrtf(loaded);
		out[i] = root + 1.0f;
	}
}
```

`loaded` は実行時入力なので、次の run の値もそのまま反映する。`sqrtf` の呼出し結果だけを `root` とし、ADD と融合する。runtime はこのシンボルの解決に必要な環境だけ link 引数を足す。

同じ入力を追う例。

```text
buffers[0]=[0,1,4,9], scalars[0]=4 -> [1,2,3,4]
sqrt(-0.0f) bits = 0x80000000
sqrt(-1.0f) classification = NaN
```

通常値は数値で照合し、-0 は sign bit、負値は NaN 分類で別々に確認する。この生成コードは -ffast-math を要求しない。

- [ ] SQRT の dtype と入力数を verifier に追加する。
- [ ] interpreter と C renderer の sqrt をそろえる。
- [ ] コンパイルされた共有ライブラリから sqrtf を解決できるようにする。
- [ ] 通常値、-0、負値、Inf、NaN を分類して比較する。

予定例 `math_sqrt` は `[0,1,2,3]` を返す。
負値で NaN を返すことと、-0 の符号を維持することを別々に検査する。
SQRT を ADD と融合した場合も固定 ABI から最後まで実行する。

### 2. sin、log2、exp2 を libm で動かす

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── math_libm.rs  # [追加] libm モードの予定実行例。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # [更新] sinf/log2f/exp2f の出力。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── interpret.rs  # [更新] 各ホスト数学関数。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rangeify.rs  # [更新] 同形 elementwise としての lowering。
│   ├── tensor.rs  # [更新] sin/log2/exp2 と自然指数・対数の合成。
│   ├── uop.rs  # [更新] SIN/LOG2/EXP2 の定義。
│   └── verify.rs  # [更新] 数学演算の入力検証。
├── tests/
│   └── math_functions.rs  # [更新] 合成と特殊値分類。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

SQRT と同じ各モジュールの match に SIN/LOG2/EXP2 を追加する。src/tensor.rs の exp/log は f32 定数と既存演算で組み、src/renderer/c.rs だけで C の関数名へ変換する。掲載 math_libm の C を Rust の関数として移植しない。


SQRT の接続を再利用し、SIN、LOG2、EXP2 も一入力、同形 f32 の純粋な演算として追加する。変更先は UOp の検証、Tensor の入口、interpreter と C renderer で、設計案の `sin`、`log2`、`exp2` が対応するノードを作る。自然な指数と対数は独立命令にせず、`exp(x)=exp2(x*log2(e))`、`log(x)=log2(x)*ln(2)` を graph builder で組む。定数は f32 として値を明示し、添字用 i32 を誤って数学関数へ流す場合は compile 前に拒否する。

C は入力の LOAD、`sinf`、`log2f`、`exp2f` の呼出し、STORE の順に出す。`exp2(log2(x))` の入力0.5なら、LOADで0.5、log2fで-1、exp2fで0.5となり、入力8なら3を経て8へ戻る。これにより関数の単独結果だけでなく、二つの数学演算を同じ座標で融合しても順序と型が保たれるかが分かる。`exp([0,ln(2)])` は積が概ね0と1になってから exp2 へ入り、結果が概ね1と2になる。πやln(2)は f32 の近似値なので、三角関数の節点で厳密な0を期待しない。

定義域と特殊値を演算ごとに定める。log2 は正の有限値に通常値を返し、±0は-Inf、負値はNaN、+Infは+Infである。exp2 は-Infで0、+Infで+Inf、有限の大きい値では overflow、小さい値ではsubnormalまたは0となる。sin は±InfでNaNとなる。NaN入力はどの演算でもNaNとして分類する。例外的な値を0へ置換して成功扱いにする処理は加えない。

局所検証では冒頭の入力を単独関数と合成の両方で実行し、通常値には許容誤差、特殊値には分類を使う。`tests/math_functions.rs` はホスト関数を参照にできるが、libm同士の下位bit差を一律の完全一致で拒否しない。source dump に関数呼出しと必要なヘッダがあり、同じ Plan に異なる入力を渡せることも確認する。この時点の libm モードを保存しておけば、次の近似展開の比較元を別経路として使える。


同じ LOAD と STORE の間へ三つの libm 呼出しを追加する。以下は `src/renderer/c.rs` の出力例で、三種類の入力をそれぞれ別の読取専用 buffer として渡す。

組込み用の抜粋。

数学 UOp の C libm 呼出しを示す生成ソース。

```c
// src/renderer/c.rs が生成するコード。生成先: <一時ディレクトリ>/math_libm.c
#include <math.h>
void math_libm(const float *angles, const float *positive,
              const float *powers, float *s, float *l, float *e, int n) {
	for (int i=0; i<n; i++) {
		float angle = angles[i];
		float p = positive[i];
		float exponent = powers[i];
		s[i] = sinf(angle);
		l[i] = log2f(p);
		e[i] = exp2f(exponent);
	}
}
```

`angle,p,exponent` はすべて f32 の値であり、i32 添字の i と区別する。三関数とも同じ反復の値を返す elementwise として扱い、特殊値の分岐はここでは libm に任せる。

同じ入力を追う例。

```text
LOG(x) -> MUL(LOG2(x), CONST_F32(ln(2)))
EXP(x) -> EXP2(MUL(x, CONST_F32(log2(e))))
positive=[0.5,1,2,8] -> l=[-1,0,1,3]
powers=[-1,0,1,3]   -> e=[0.5,1,2,8]
```

自然対数と自然指数は graph の合成であり、新しい backend 呼出しではない。定数も f32 として確定し、二つの数学関数の順番を dump で確認する。

- [ ] SIN、LOG2、EXP2 を elementwise として rangeify に通す。
- [ ] C 側を sinf、log2f、exp2f に下げる。
- [ ] `exp` と自然対数を定数の積と既存 UOp の合成で追加する。
- [ ] 特殊値の期待する分類と定義域を API の説明に書く。
- [ ] `exp2(log2(x))` を正の小さい配列で実行する。

予定例 `math_libm` は冒頭の四つの配列を表示する。
`exp2(log2([0.5,1,2,8]))` は各要素について絶対誤差 `2e-6` 以内を初期目標にする。
libm 間の最下位 bit の差を、直ちに compiler の誤りと扱わない。

### 3. exp2 と log2 の範囲を縮めて多項式に下げる

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_12_exp_log_poly.rs  # 本文のコードを配置する。
│   └── math_exp_log_poly.rs  # [追加] 近似と libm を比べる予定例。
├── src/
│   ├── codegen/
│   │   ├── math.rs  # [追加] decompose_math と exp2/log2 の UOp 展開。
│   │   └── mod.rs  # [追加] math モジュールの宣言と codegen への接続。
│   ├── lib.rs  # [更新] codegen モジュールの宣言。
│   ├── uop.rs  # [更新] 丸め、bitcast、shift が必要な場合の型付き定義。
│   └── verify.rs  # [更新] 新規整数/bit 演算の検証。
├── tests/
│   └── math_functions.rs  # [更新] 近似区間、境界、fallback の検証。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/lib.rs で mod codegen、src/codegen/mod.rs で mod math を宣言し、src/codegen/math.rs に decompose_math を追加する。本文の数値版は book_12_exp_log_poly へ保存し、本体では同じ Horner 計算を UOp の生成へ移す。微分後の codegen から呼び、範囲外は既存 libm 経路へ分岐する。

新しいモジュールは、既存の宣言を残して次を追記する。内部の補助関数は各モジュールに置き、exampleから使う入口だけを公開する。

```rust
// src/codegen/mod.rs（モジュール宣言を追加）
pub mod math;
// src/lib.rs（モジュール宣言を追加）
pub mod codegen;
```


このコミットでは既存 libm 経路を残し、明示指定した bounded-poly だけを通常の UOp の積と和へ展開する。候補は `src/codegen/math.rs` の設計案 `decompose_math(node,mode)` と、その検証用テストである。範囲判定を先に行い、対象外を整数変換や bit 操作へ流さず libm 分岐へ渡す。新たな整数丸め、bitcast、shift が必要なら、その意味と verifier を同じコミットで追加する。bitcast は数値の変換でなく同じ32bit列を別の型として読む操作であり、f32とi32を数値 CAST する処理と区別する。

exp2 は有限で `-16<=x<=16` のときだけ qを最近接整数、`r=x-q` とし、`|r|<=0.5` を確認する。係数を捏造しない初版の案は、`P(r)=Σ(k=0..7) (ln(2)*r)^k/k!` を Horner 形式で評価する方法である。係数は高精度でこの式から生成して一度 f32 へ丸め、生成手順と次数を残す。qの範囲なら `2^q` は正規数なので指数部 `q+127` を23bit左へ shift して f32 へ bitcast でき、Pと掛けて返す。x=3.25はq=3,r=0.25、Pは概ね1.189207、再構成は概ね9.513657になる。目標誤差は実際の f32 評価で検証し、式の打切り誤差だけで達成済みとしない。

log2 は正の有限な正規数だけで指数eと仮数 `m∈[1,2)` を取り、`z=(m-1)/(m+1)` とする。`log2(x)=e+(2/ln(2))*(z+z^3/3+z^5/5+…+z^13/13)` の奇数次数を使えば `z∈[0,1/3)` に範囲が絞れる。係数は1,1/3,…という生成式を記録し、`z*z` の Horner 評価へ下げる。x=8ならe=3,m=1,z=0から厳密に3、x=1.5ならe=0,z=0.2から概ね0.584963となる。

テストは格子、固定seed、境界の隣接f32を libm と比較し、0付近のlog2には絶対誤差も使う。subnormal、負値、Inf、NaN、範囲外exp2が fallback へ入ることを source と値で確認する。高水準ノードは元グラフに保持し、微分に先立って近似式へ置換しない。


`src/codegen/math.rs` に組み込む Horner 評価の数値版を先に検証する。入力を範囲で分け、通常の積と和で exp2 と log2 を計算する。

単独で実行できる例。

数値の Horner 評価と main 全体。src/codegen/math.rs には UOp を作る版を実装する。

```rust
// examples/book_12_exp_log_poly.rs（単独で実行する例）
fn exp2_poly(x: f32) -> f32 {
	if !x.is_finite() || !(-16.0..=16.0).contains(&x) { return x.exp2(); }
	let q = x.round() as i32;
	let r = x - q as f32;
	let t = r * std::f32::consts::LN_2;
	let mut p = 1.0 / 5040.0; // 1/7!
	for denominator in [720.0,120.0,24.0,6.0,2.0,1.0,1.0] {
		p = p*t + 1.0/denominator;
	}
	let scale = f32::from_bits(((q+127) as u32)<<23);
	p * scale
}
fn log2_poly(x: f32) -> f32 {
	if !x.is_normal() || x <= 0.0 { return x.log2(); }
	let bits = x.to_bits();
	let e = ((bits>>23)&255) as i32 - 127;
	let m = f32::from_bits((bits&0x7fffff) | (127<<23));
	let z = (m-1.0)/(m+1.0);
	let zz = z*z;
	let mut p = 1.0/13.0;
	for denominator in [11.0,9.0,7.0,5.0,3.0,1.0] {
		p = p*zz + 1.0/denominator;
	}
	e as f32 + (2.0/std::f32::consts::LN_2)*z*p
}
fn main() {
	for i in -1600..=1600 {
		let x = i as f32 / 100.0;
		let reference = x.exp2();
		assert!((exp2_poly(x)-reference).abs() <= 2e-6+2e-5*reference.abs());
		let p = 2.0_f32.powf(x);
		assert!((log2_poly(p)-p.log2()).abs() <= 2e-6+2e-5*p.log2().abs());
	}
	assert_eq!(log2_poly(0.0), f32::NEG_INFINITY);
	assert!(exp2_poly(f32::NAN).is_nan());
}
```

exp2 の `t` は ln(2)r であり、係数はその冪に対する 1/k! である。log2 の指数 `e` と仮数 `m` は bit 列から分離し、`z*z` を使って奇数項を Horner 評価する。対象外では整数変換や指数 bit の組立てより先に host の対応関数へ戻る。

同じ入力を追う例。

```text
exp2(3.25): q=3,r=0.25,t≈0.1732868,P≈1.189207,scale=8
log2(1.5): e=0,m=1.5,z=0.2,zz=0.04,result≈0.584963
log2(subnormal): fallback, exponent extraction is skipped
```

このコードは数値版の検証器である。compiler の本体では同じ積と和を UOp にし、元の高水準 EXP2/LOG2 は微分が終わるまで保つ。

- [ ] exp2 は `x=q+r`、`r∈[-0.5,0.5]` に分けて Horner 形式で評価する。
- [ ] `2^q` の再構成を定義し、整数変換の範囲を検査する。
- [ ] log2 は `x=2^e*m` に分け、m を限られた区間に正規化する。
- [ ] 係数、近似区間、次数、出典または係数生成方法を実装の近くに残す。
- [ ] 各段階を UOp にして既存 rewrite と renderer へ通す。

予定例 `math_exp_log_poly` は libm と近似版の最大絶対誤差、最大相対誤差を別々に表示する。
近似区間内の初期目標は `abs_error <= 2e-6 + 2e-5*abs(reference)` とする。
log2 が 0 に近い場所では相対誤差だけを使わない。
範囲外、subnormal、特殊値は libm 経路を実際に実行して確認する。

### 4. sin の引数還元と近似を実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_12_sin_poly.rs  # 本文のコードを配置する。
│   └── math_sin_poly.rs  # [追加] sin 近似を libm と比べる予定例。
├── src/
│   ├── codegen/
│   │   ├── math.rs  # [更新] decompose_sin と bounded な引数還元。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   └── lib.rs  # 本体のモジュール宣言と公開APIの入口。
├── tests/
│   └── math_functions.rs  # [更新] π付近、±32、巨大値と特殊値。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/codegen/math.rs に decompose_sin を追加し、前節の数学モード選択から呼ぶ。掲載 sin_poly と main は book_12_sin_poly の単独数値例へ保存する。compiler 本体では還元、符号、fallback を UOp と分岐へ置き換え、高水準 SIN を微分前に書き換えない。


SIN の近似は入力を短い区間へ戻してから行う。候補は数学分解モジュールの設計案 `decompose_sin(node)` と `tests/math_functions.rs` で、libm を選ぶ既定経路は変えない。まず有限かつ `|x|<=32` を判定し、それ以外は sinf へ分岐する。範囲内では `q=round(x/π)`、`r=x-q*π` として `r∈[-π/2,π/2]` を作り、`sin(x)=(-1)^q*sin(r)` で符号を復元する案を使える。ここで q は半周期の番号であり、qの偶奇が二つの象限ずつをまとめた符号を決める。qの負数に対しても偶奇判定が同じになる整数処理を定義する。

πを一個の粗い f32 として引くだけでは節点付近の誤差が目立つので、高精度πを `pi_hi` と残差 `pi_lo` へ分け、`r=(x-q*pi_hi)-q*pi_lo` と計算する。二つの定数の生成精度と丸め方を記録する。区間内の多項式は、まず係数が明確な `r-r^3/3!+r^5/5!-r^7/7!+r^9/9!-r^11/11!` を設計案にできる。`r*(1+r²*(-1/3!+r²*(…)))` の順で評価し、係数は式から生成する。Horner形式とはこのように高い次数側から掛算と加算を繰り返し、冪を毎回作り直さない評価順のことである。

xを f32 のπとした場合、q=1、rはほぼ0となり、奇数qの符号を掛けてほぼ0を返す。x=π/2ならqの丸めの規則により±π/2のいずれかへ還元されても、復元後はほぼ1になる必要がある。境界の丸めが変わる場所ではq,r,符号,多項式値を debug dump に出して、間違った象限と近似誤差を区別する。-0は必要なら入力をそのまま返す分岐で符号を保つ。

局所検証は0、±π/2、πの倍数、±32、その隣接f32、固定格子を libm と比較し、絶対誤差2e-5を目標にする。次数だけで誤差達成を断言せず、還元とf32評価を含めて調べる。巨大値が整数変換を通らず fallback へ行くこと、元の SIN graph と展開後の graph の両方が dump できることを確認する。


SIN の bounded-poly を同じモジュールに加える。次の例は range reduction と符号復元まで含み、有限な [-32,32] の外では host 関数へ戻る。

単独で実行できる例。

引数還元の数値検証と main 全体を保存する。

```rust
// examples/book_12_sin_poly.rs（単独で実行する例）
fn sin_poly(x: f32) -> f32 {
	if !x.is_finite() || x.abs() > 32.0 { return x.sin(); }
	if x == 0.0 { return x; }
	let q = (x/std::f32::consts::PI).round() as i32;
	let pi_hi = std::f32::consts::PI;
	let pi_lo = (std::f64::consts::PI-pi_hi as f64) as f32;
	let r = (x-q as f32*pi_hi)-q as f32*pi_lo;
	let rr = r*r;
	let mut p = -1.0/39916800.0; // -1/11!
	for coefficient in [1.0/362880.0,-1.0/5040.0,1.0/120.0,-1.0/6.0,1.0] {
		p = p*rr + coefficient;
	}
	let reduced = r*p;
	if q&1 == 0 { reduced } else { -reduced }
}
fn main() {
	for i in -32000..=32000 {
		let x = i as f32 / 1000.0;
		assert!((sin_poly(x)-x.sin()).abs() <= 2e-5, "x={x}");
	}
	assert_eq!(sin_poly(-0.0).to_bits(), (-0.0_f32).to_bits());
	assert!(sin_poly(f32::INFINITY).is_nan());
	assert_eq!(sin_poly(1e20).to_bits(), 1e20_f32.sin().to_bits());
}
```

`pi_hi` を f32 に丸めた残差を f64 のπから生成して `pi_lo` とする。`q` が半周期の番号、`r` が短い区間の値、`p` が sin(r)/r の偶多項式である。負の q でも `q&1` で偶奇を判定できる。

同じ入力を追う例。

```text
x=f32(π) -> q=1 -> r≈8.742278e-8 -> sign=-1
x=-0 -> return original -0 bits
x=1e20 -> fallback before integer conversion
```

πの入力値自身が近似値なので、ほぼゼロの残差にも意味がある。特殊値と巨大値をこの引数還元へ通さないことを確認する。

- [ ] 周期と象限を求め、短い区間へ還元して符号を復元する。
- [ ] 区間内だけの奇多項式を Horner 形式で評価する。
- [ ] `|x|<=32` の境界、π の倍数、その両隣をテストする。
- [ ] 大きな有限値と Inf、NaN を libm 経路に分岐する。
- [ ] 高水準 SIN と近似展開後のグラフを両方 dump できるようにする。

予定例 `math_sin_poly` は固定格子と境界付近で libm との差を表示する。
近似区間内の絶対誤差 `2e-5` 以下を初期目標にする。
未還元の全実数へ Taylor 展開を直接適用する実装は採用しない。
大きな sin に対応する場合は Payne-Hanek 型の還元を別途検証する。

## 完了確認

以下は将来追加する例と `tests/math_functions.rs` の予定コマンドである。

```sh
# 作業ディレクトリ: chibigrad/
cargo test
cargo run --example math_sqrt -- --dump-source
cargo run --example math_libm -- --backend cpu
cargo run --example math_exp_log_poly -- --math bounded-poly --compare libm
cargo run --example math_sin_poly -- --math bounded-poly --dump-kernel
```

- [ ] 通常値の許容誤差と特殊値の分類を分けて判定する。
- [ ] sqrt(-0) の sign bit、log2(0) の -Inf、exp2(-Inf) の 0 を検査する。
- [ ] 固定 seed の有限値集合と、区間境界の隣接 f32 を比較する。
- [ ] 近似対象外の値を libm に戻す分岐が生成コードに残る。
- [ ] 近似を選ばない既存 example の数値が変わらない。

## tinygrad ソースとの対応

参照 commit は `c22edff14c79310a9b2df62a78819cf77c8289fa` である。

| 実在するシンボル | ソース | 役割 | chibigrad での簡略化 |
|---|---|---|---|
| `ElementwiseMixin.sqrt` | [elementwise.py](../../tinygrad/tinygrad/mixin/elementwise.py#L481) | SQRT を生成する | f32 だけを扱う |
| `ClangRenderer.code_for_op` | [cstyle.py](../../tinygrad/tinygrad/renderer/cstyle.py#L302) | sqrt などを C の builtin へ下げる | 初版は四関数とも libm を使う |
| `xsin` | [transcendental.py](../../tinygrad/tinygrad/codegen/decomp/transcendental.py#L170) | 引数還元と多項式、特殊値を合成する | bounded な範囲以外は libm へ戻す |
| `xexp2` | [transcendental.py](../../tinygrad/tinygrad/codegen/decomp/transcendental.py#L193) | 整数部と残差を分離して近似する | 誤差目標を独自に検証する |
| `xlog2` | [transcendental.py](../../tinygrad/tinygrad/codegen/decomp/transcendental.py#L219) | 指数と仮数を分離して近似する | subnormal は初版で libm に委ねる |
| `pm_cast_float_alu` | [codegen/__init__.py](../../tinygrad/tinygrad/codegen/__init__.py#L238) | 分解前に浮動小数点の入力を明示する | integer 入力を暗黙に近似へ流さない |

## 今は追加しないもの

f64、全 f32 域で保証された独自 libm、正しく丸める関数は完成条件に含めない。
微分は近似多項式の内部ではなく、数学関数の高水準 UOp に対して [13](13-autodiff.md) で定義する。
softmax と学習用の損失は [14](14-sgd-mnist.md) で最小例から加える。
