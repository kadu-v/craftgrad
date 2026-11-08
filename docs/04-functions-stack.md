# 04. 純粋関数を CALL し、STACK と INDEX を UIR から実行する

## このステップで動かすもの

[03-uop-rewrites.md](03-uop-rewrites.md) の scalar グラフに関数境界を加える。
`f(a,b)=(a+b)*2` を二回呼び、入力 `2,3` と `5,1` で `10` と `12` を返す。
同じ関数の PARAM と、呼出し側の PARAM を混同しない。
元計画第 2 章の UOp と検証を補い、第 3 章以降の kernel 境界に備える。
以下の名前、文法、コマンド、数値は未実装の予定である。

## 最小実装とデータ

`FunctionId` ごとに引数の dtype、body の root、ローカルな PARAM を持つ。
PARAM の意味は `(FunctionId, slot)` で決め、slot だけをグローバルキーにしない。
`CALL(function, args...)` は、この章では副作用のない scalar 関数だけを呼ぶ。
再帰と外部関数は拒否し、interpreter と C に同じ引数束縛を使う。
C renderer は最初に body を inline してよい。
別名の PARAM を置換するときも、関数の所属を確認する。

`STACK` は同じ dtype の scalar 値をまとめた固定長の列にする。
`INDEX(STACK, const_i32)` はその要素を選ぶ純粋な値操作である。
この INDEX はアドレス生成でもメモリ読出しでもない。
異なる型を混ぜる一般 tuple と、動的長の列はまだ扱わない。
空 STACK は値のない `void` として扱い、数値 ALU へ渡さない。

予定する関数と呼出しの概形は次になる。

```text
fn pair(%a:f32, %b:f32) -> stack<f32,2> {
	%s = ADD %a %b
	%d = SUB %a %b
	return STACK %s %d
}
%r = CALL pair(2.0, 3.0)
%v = INDEX %r 0
```

この文法は chibigrad の最小 UIR 案で、tinygrad の入力文法を名乗らない。
UIR は行単位の `CONST`、`PARAM`、ALU、`STACK`、静的 `INDEX`、関数と `CALL` を扱う。
出力はノード番号と dtype を含む決定的な順序にする。
f32 定数はビット列または hex 表記で往復させる。
parser は未定義参照、重複定義、不正 dtype、誤った CALL 引数を拒否する。

テスト用に `assert_uop(expression, expected)` という helper を作る案にする。
これは interpreter と生成 C の結果を検証する Rust の assertion である。
backend に実行時 ASSERT 命令を追加する意味ではない。
UIR の assertion fixture もホスト上のテスト入力と予定値の組として保持する。

## この章のファイル配置

関数の ID と型は uop、引数の評価は interpret、inline と静的 INDEX の解消は rewrite に追加する。UIR の入出力が初めて必要になるコミット 3 でだけ uir モジュールを追加する。独立した inline の main は本体から分ける。

次はこの章の終わりまでに触る関連ファイルの配置案である。現在のcheckoutの一覧ではない。各コミットでは、その下に示す部分だけを追加・変更する。

```text
chibigrad/
├── examples/
│   ├── book_04_inline.rs  # 関数ごとの PARAM 置換を確かめる std の独立例。
│   └── function_stack.rs  # 手書き builder と UIR roundtrip の入口。
├── src/
│   ├── interpret.rs  # Value enum と eval_stack。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rewrite.rs  # inline 後の resolve_index。
│   ├── uir.rs  # render_uir、parse_uir と関数内の名前表。
│   ├── uop.rs  # ValueKind、STACK、静的 INDEX の型と長さ。
│   └── verify.rs  # verify_function、CALL と PARAM 所属の検証。
├── tests/
│   ├── common/
│   │   └── mod.rs  # 検証後の interpreter と CPU を比較する assert_uop。
│   ├── fixtures/
│   │   └── pair.uir  # 成功する UIR roundtrip の比較入力。
│   └── functions_stack.rs  # 署名違反、再帰、INDEX と assertion 失敗の統合確認。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

生成Cやshaderは一時ディレクトリへ出力するため、このRustプロジェクトのツリーには含めない。コード冒頭のコメントで、生成元と生成先を確認する。

## 小さなコミットで進める

### コミット 1. PARAM を関数のスコープへ束縛して CALL する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_04_inline.rs  # [追加] 関数ごとの PARAM 置換を確かめる std の独立例。
│   └── function_stack.rs  # [追加] 同じ関数を二組の入力で CPU 実行する入口。
├── src/
│   ├── interpret.rs  # [変更] 呼出し単位の引数環境と eval_call。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rewrite.rs  # [変更] C 生成前の inline_call とローカル PARAM 置換。
│   ├── uop.rs  # [変更] FunctionId、Function、Signature と所属を持つ PARAM。
│   └── verify.rs  # [変更] 関数署名と自由 PARAM の事前検証。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

型は src/uop.rs に追加し、既存の PARAM キーを FunctionId と slot の組へ広げる。eval_call は interpret.rs、inline_call は rewrite.rs に追加し、verify.rs の署名検証後に inline する。掲載した std の例は examples/book_04_inline.rs へ全体を保存する。本体へ組み込む際はその inline の処理を既存 Graph::intern と NodeId に合わせ、局所 Op や eval、main の重複定義を src へ持ち込まない。


今までは全 PARAM が一つの入口に属していたため、同じ slot 0 を複数の関数で使うと区別できない。候補ファイル `src/uop.rs` に設計案の `Function { id, signature, root }` を足し、PARAM の key を `(FunctionId, slot, dtype)` に変える。`src/interpret.rs` の設計案 `eval_call` は、呼出し側で引数値を評価してから、呼出し先 ID に対応する新しい引数環境を作って body を評価する。body の PARAM はその環境だけを読み、呼出し元の slot へ暗黙に戻らない。

C 生成に渡す前には、設計案の `inline_call` が呼出し先 PARAM から呼出し元の引数ノードへの置換表を作る。`f(a,b)=(a+b)*2` の f の slot 0 を最初の引数へ、slot 1 を二番目へ置換し、子を先に body を再構築する。別関数の PARAM を見つけたら置換せず、自由変数の違反として拒否する。inline 用の memo は呼出し単位で持つか、関数 ID と引数列まで key に含める。body の NodeId だけで共有すると、一回目の引数が二回目にも残ってしまう。

`f(2,3)` の置換後は `(2+3)*2` で中間値 5、結果 10、`f(5,1)` は `(5+1)*2` で中間値 6、結果 12 となる。局所テストでは二つの CALL を同じグラフへ置き、interpreter と C の結果を比較する。さらに関数 f と g の slot 0 が別番号になること、inline 後に f の PARAM が残らないことを dump で確認する。既存の ABI は入口側の PARAM だけに使い、内部関数ごとに FFI を増やさない。


単独で実行できる例である。`src/rewrite.rs` の inline を、関数 f のローカル PARAM を置換する小さなグラフで動かす。関数 ID は `u32`、NodeId はこの例だけ `usize` で表す。

局所の型と main を含む全体を一つの独立例へ保存する。

```rust
// examples/book_04_inline.rs（単独で実行する例）
use std::collections::HashMap;
#[derive(Clone)]
enum Op { Const(f32), Param { owner: u32, slot: usize }, Add, Mul }
#[derive(Clone)]
struct UOp { op: Op, src: Vec<usize> }
fn inline(g: &mut Vec<UOp>, id: usize, owner: u32, args: &[usize], memo: &mut HashMap<usize,usize>) -> Result<usize,String> {
	if let Some(&result) = memo.get(&id) { return Ok(result); }
	let n = g[id].clone();
	let result = match n.op {
		Op::Param { owner: found, slot } => {
			if found != owner { return Err("foreign PARAM capture".into()); }
			*args.get(slot).ok_or("missing argument")?
		}
		_ => {
			let src = n.src.into_iter().map(|s| inline(g,s,owner,args,memo)).collect::<Result<Vec<_>,_>>()?;
			let result = g.len();
			g.push(UOp { op: n.op, src });
			result
		}
	};
	memo.insert(id,result);
	Ok(result)
}
fn eval(g: &[UOp], id: usize) -> f32 {
	let n = &g[id];
	match n.op {
		Op::Const(x) => x,
		Op::Add => eval(g,n.src[0]) + eval(g,n.src[1]),
		Op::Mul => eval(g,n.src[0]) * eval(g,n.src[1]),
		Op::Param { .. } => panic!("unbound PARAM"),
	}
}
fn main() {
	let mut g = vec![
		UOp { op: Op::Param { owner: 1, slot: 0 }, src: vec![] },
		UOp { op: Op::Param { owner: 1, slot: 1 }, src: vec![] },
		UOp { op: Op::Add, src: vec![0,1] },
		UOp { op: Op::Const(2.0), src: vec![] },
		UOp { op: Op::Mul, src: vec![2,3] },
	];
	for x in [2.0,3.0,5.0,1.0] { g.push(UOp { op: Op::Const(x), src: vec![] }); }
	let first = inline(&mut g,4,1,&[5,6],&mut HashMap::new()).unwrap();
	let second = inline(&mut g,4,1,&[7,8],&mut HashMap::new()).unwrap();
	assert_eq!((eval(&g,first),eval(&g,second)), (10.0,12.0));
	assert!(inline(&mut g,4,2,&[5,6],&mut HashMap::new()).is_err());
}
```

`owner=1` の二つの PARAM だけを `args` で置換する。body 内の同じ ID を再訪したときはその呼出しの `memo` を使うが、二回目の CALL には新しい表を渡す。これで一回目の引数 `[5,6]` に対応した置換が、二回目 `[7,8]` へ混入しない。実装へ組み込む際は、inline 前に署名の型と引数数を検証し、純粋ノードの追加を 03 の intern に置き換える。

この入力の途中経過を並べる。

```text
f PARAMs: (owner=1,slot=0), (owner=1,slot=1)
CALL f(2,3) -> ADD(2,3)*2 -> 10
CALL f(5,1) -> ADD(5,1)*2 -> 12
owner=2 の body として同じ PARAM を使う -> foreign PARAM capture
```

- [ ] `FunctionId` と引数署名を、実際に呼ぶ関数一つに対して追加する。
- [ ] PARAM の intern key に所属関数を含める。
- [ ] interpreter が呼出しごとに引数環境を作るようにする。
- [ ] C 生成前に body を capture なしで inline する。
- [ ] `f(2,3)=10` と `f(5,1)=12` を CPU まで通す。

同じ slot 0 でも、別の関数の PARAM は別物になる。
inline 後の dump に呼出し先の自由 PARAM が残らないことを確認する。

### コミット 2. STACK と静的 INDEX で複数の結果を選ぶ

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── function_stack.rs  # [変更] pair の INDEX 0 と INDEX 1 を scalar root として実行する。
├── src/
│   ├── interpret.rs  # [変更] Value enum と eval_stack。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rewrite.rs  # [変更] inline 後の resolve_index。
│   ├── uop.rs  # [変更] ValueKind、STACK、静的 INDEX の型と長さ。
│   └── verify.rs  # [変更] STACK の同型性と INDEX の範囲検証。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

ValueKind は uop.rs、評価結果の Value は interpret.rs に追加する。抜粋の resolve_index は rewrite.rs、eval_stack は interpret.rs に置く。verify.rs は選択前に STACK 全体を検証し、examples/function_stack.rs が選んだ要素を既存の scalar ABI へ渡す。


複数値をメモリへ置く前に、グラフ上の値としてまとめる。候補ファイル `src/uop.rs` に設計案の `ValueKind::Stack { element_dtype, len }` を加え、interpreter の評価結果も scalar と固定長列を区別する。STACK の入力を左から順に評価し、同じ dtype の Scalar 列を返す。INDEX はこの列と定数 I32 の位置を受け取り、一つの scalar を返す。前章の scalar dtype の意味を変えて、STACK を一つの f32 と見なすことはしない。

`pair(a,b)` の body は `s=ADD(a,b)`、`d=SUB(a,b)`、`STACK(s,d)` とする。入力 2、3 では内部の列が `[5,-1]` になり、INDEX 0 は 5、INDEX 1 は -1 を返す。inline は STACK の各要素へ同じ PARAM 置換表を適用する。C renderer の前で `INDEX(STACK(s,d),0)` を s へ解決できるので、C に列型やメモリを導入する必要はない。選ばなかった要素も型が有効かは検証し、INDEX だけで不正な body を隠さない。

実装順は STACK の型と長さの導出、interpreter の列値、INDEX の範囲検査、CALL 戻り値の扱い、inline 後の選択解消とする。局所テストでは同じ pair の二つの選択をそれぞれ scalar root にして既存 ABI から読み、5 と -1 を比較する。長さ 2 に対する INDEX 2、F32 と I32 を混ぜた STACK を拒否する。空 STACK は値のない Void として数値演算へ渡さない。これで外部入口の二配列 ABI は維持し、関数内部の複数結果を検証できる。


組込み用の抜粋である。`src/rewrite.rs` で inline 後の静的 INDEX を解く。`g.kind(id)` は scalar、stack、void を区別する前段の型導出で、STACK の全要素の dtype と長さを検証済みとする。INDEX の位置は `Op::Index(i32)` の payload とする。

静的 INDEX の解消と列値の評価を分ける。

```rust
// src/rewrite.rs（既存ファイルへ組み込む抜粋）
fn resolve_index(g: &Graph, id: NodeId) -> Result<NodeId, String> {
	let node = &g.nodes[id.0];
	let index = match node.op { Op::Index(i) => i, _ => return Ok(id) };
	if index < 0 { return Err("INDEX must be nonnegative".into()); }
	let stack = &g.nodes[node.src[0].0];
	if !matches!(stack.op, Op::Stack) { return Err("INDEX needs an inlined STACK".into()); }
	stack.src.get(index as usize).copied().ok_or_else(|| "INDEX exceeds STACK length".into())
}
// src/interpret.rs（既存ファイルへ組み込む抜粋）
fn eval_stack(values: Vec<Scalar>) -> Result<Value, String> {
	if values.is_empty() { return Ok(Value::Void); }
	let dtype = values[0].dtype();
	if values.iter().any(|v| v.dtype() != dtype) { return Err("mixed STACK dtype".into()); }
	Ok(Value::Stack(values))
}
```

`Value` は `Scalar(Scalar)`、`Stack(Vec<Scalar>)`、`Void` の評価結果 enum とする。`resolve_index` は STACK の列から元のノード番号を返すので、値をメモリへ書く処理は増えない。`pair(2,3)` を inline した STACK の入力は ADD と SUB の二つであり、INDEX 0 を ADD へ、INDEX 1 を SUB へ解消できる。型検証は選択前に済ませ、未選択の不正要素を隠さない。

この入力の途中経過を並べる。

```text
inline pair(2,3): STACK(ADD(2,3),SUB(2,3))
INDEX 0 -> ADD(2,3) -> scalar 5 -> scalar ABI
INDEX 1 -> SUB(2,3) -> scalar -1 -> scalar ABI
INDEX 2 -> INDEX exceeds STACK length
```

- [ ] 同型 scalar の STACK を interpreter と型導出へ加える。
- [ ] 静的 INDEX の型と `0 <= index < len` を検証する。
- [ ] CALL の結果に固定長 STACK を許し、各要素を inline する。
- [ ] `pair(2,3)` から `5` と `-1` を取り出して C まで通す。

外部 ABI は scalar のまま維持してよい。
example は必要な要素を root にした入口をそれぞれ呼ぶ。
STACK 全体の ABI をここで設計しない。

### コミット 3. dump した最小 UIR を読み戻して同じ計算をする

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── function_stack.rs  # [変更] 手書き builder と UIR roundtrip の入口。
├── src/
│   ├── lib.rs  # [変更] uir モジュールと roundtrip 用 API の公開。
│   └── uir.rs  # [追加] render_uir、parse_uir と関数内の名前表。
├── tests/
│   └── fixtures/
│       └── pair.uir  # [追加] pair の dump と読み戻しに使う入力 fixture。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/uir.rs を追加し、parse_body_line と render_uir、parse_uir を置く。lib.rs に uir の宣言を追加し、外から必要な parse と render の入口を公開する。parser は uop.rs の既存の型と intern、verify.rs の binary_type を使う。pair のテキストは tests/fixtures/pair.uir へ置き、function_stack の --roundtrip から読み戻したグラフを同じ実行経路に渡す。

新しいモジュールは、既存の宣言を残して次を追記する。内部の補助関数は各モジュールに置き、exampleから使う入口だけを公開する。

```rust
// src/lib.rs（モジュール宣言を追加）
pub mod uir;
```


テキストからも同じグラフを構築できるようにする。候補ファイル `src/uir.rs` の設計案 `render_uir` と `parse_uir` を対にし、今ある演算と関数だけの小さい文法を定める。printer は関数を決定的な順に並べ、各 body では依存先を先に番号付けする。番号はファイル内の参照名であり、arena に読み込んだ後の NodeId と一致する必要はない。parser は関数ごとの名前表を持ち、読み取った `%s` をその関数で作った実ノードへ対応させる。

処理順は行の分解、演算名と dtype の読取り、参照名の解決、署名の検証、intern への登録とする。例えば pair の `%s=ADD %a %b` を読む時点では `%a` と `%b` が同じ関数の PARAM として登録済みでなければならない。CALL の呼出し先は関数表から解決し、署名がそろってから body と引数をつなぐ。定数 `-0.0` を十進数だけで往復させず、F32Bits の 8 桁 hex を保存する形なら符号付きゼロと NaN payload を保持できる。

局所テストは pair を dump し、その文字列だけから別 arena を作り、INDEX 0 と 1 の root を再生成して interpreter と C で 5 と -1 を得る。NodeId の数値が一致したかではなく、引数束縛と結果の意味が一致したかを確かめる。未定義名と重複した `%s` は元の行番号で拒否する。fixture に保存する期待値はホスト assertion の入力であり、UIR に backend の ASSERT 演算を追加する契約ではない。前の手書き builder と新 parser が同じ verifier を使うようにする。


組込み用の抜粋である。`src/uir.rs` に加える関数内の一行 parser を示す。説明を絞るため `CONST f32bits` と `ADD` を示し、PARAM、SUB、STACK、INDEX、CALL も同じローカルな名前解決を通す。`Graph::intern(op, &[NodeId]) -> NodeId` は 03、関数表と署名はコミット 1 のものを使う。

関数ごとの名前表を使う parser の実装へ追加する。

```rust
// src/uir.rs（既存ファイルへ組み込む抜粋）
fn parse_body_line(g: &mut Graph, names: &mut std::collections::HashMap<String,NodeId>, line: &str) -> Result<(),String> {
	let (lhs,rhs) = line.split_once('=').ok_or("expected =")?;
	let name = lhs.trim();
	if !name.starts_with('%') || names.contains_key(name) { return Err("invalid or duplicate name".into()); }
	let words: Vec<_> = rhs.split_whitespace().collect();
	let (op,src) = match words.as_slice() {
		["CONST","f32bits",hex] => {
			let bits = u32::from_str_radix(hex.trim_start_matches("0x"),16).map_err(|_| "bad bits")?;
			(Op::Const(Scalar::F32Bits(bits)),vec![])
		}
		["ADD",a,b] => {
			let a = *names.get(*a).ok_or("undefined local name")?;
			let b = *names.get(*b).ok_or("undefined local name")?;
			binary_type(&Op::Add,g.dtype(a),g.dtype(b))?;
			(Op::Add,vec![a,b])
		}
		_ => return Err("unsupported or malformed body line".into()),
	};
	let id = g.intern(op,&src);
	names.insert(name.to_owned(),id);
	Ok(())
}
```

`names` は関数 body を読むたびに空から作り、署名の PARAM だけを最初に登録する。外側の名前表を親として検索しないので、他関数の `%a` が偶然見つかることもない。f32 の printer は `format!("0x{bits:08x}")` を使い、この parser が元の bits を復元する。行番号は呼出し側の `enumerate` の番号をエラーへ付け、関数の verifier を通してから inline する。

この入力の途中経過を並べる。

```text
fn f の names: {%a -> PARAM(f,0), %b -> PARAM(f,1)}
%s = ADD %a %b -> local lookup -> ADD(PARAM(f,0),PARAM(f,1))
fn g の names: 新規の表。f の %s は見えない
%c = CONST f32bits 0x80000000 -> -0.0 bits を保持
```

- [ ] 対応演算だけの printer と parser を追加する。
- [ ] 入力番号、関数名、dtype を検証してから arena へ入れる。
- [ ] f32 の特殊値もビット列で往復するようにする。
- [ ] `pair` の UIR を出力し、読み戻したグラフから C を生成する。
- [ ] 読み戻し前後で `5` と `-1` が一致することを assertion で確かめる。

fixture はこの章で新しく作る予定のファイルにする。
文字列が似ているだけでなく、読み戻した計算を実行する。

### コミット 4. 境界の違反を assertion とエラーで止める

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── src/
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── verify.rs  # [変更] verify_function、CALL と PARAM 所属の検証。
├── tests/
│   ├── common/
│   │   └── mod.rs  # [追加] 検証後の interpreter と CPU を比較する assert_uop。
│   ├── fixtures/
│   │   └── pair.uir  # [確認] 成功する UIR roundtrip の比較入力。
│   └── functions_stack.rs  # [追加] 署名違反、再帰、INDEX と assertion 失敗の統合確認。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

verify_function を src/verify.rs に追加し、UIR 入口と builder 入口の両方から inline 前に呼ぶ。assert_uop は backend 命令にせず tests/common/mod.rs のホスト helper とし、tests/functions_stack.rs から mod common で使う。誤った expected を渡すテストと、不正な関数を公開 compile API へ渡すテストを分け、clang 起動前の拒否も確かめる。


このコミットでは、関数の型と所属の検証を実行前の一つの境界にまとめる。候補ファイル `src/verify.rs` の設計案 `verify_function` は、body から到達する PARAM がその FunctionId に所属し、slot が署名内にあり、dtype も一致するかを調べる。別関数の PARAM を直接指す body は、slot が偶然合っていても拒否する。CALL には引数数、各 dtype、戻り値の kind を照合する。再帰は関数の呼出し辺をたどる DFS の訪問中集合で検出する。

検証は body の inline より先に行う。誤った `pair(2)` を inline してから未定義 slot を見つけるのではなく、署名の 2 と実引数の 1 を比べて止める。`INDEX(STACK(1,2),2)` も長さ 2 の半開範囲 `[0,2)` に入らないので、interpreter が列を参照する前に拒否する。parser を入口にした場合は UIR 行番号、builder を入口にした場合は関数名と NodeId をエラーに含めれば、同じ検証結果を入力へ戻せる。

設計案 `assert_uop` は、検証済み root を interpreter と C で実行して両方を expected と比較するホストの test helper にする。局所テストでは pair の結果 5 に対し expected=6 を渡したテストが意図通り失敗することを、失敗を期待するテスト機構で確認する。成功だけを返す空 helper を通さないためである。誤った引数数、他関数 PARAM、負 INDEX、再帰も別 fixture で試し、clang の呼出しに到達しないことを確認する。成功する pair の roundtrip と scalar の既存経路は、そのまま再利用できる。


組込み用の抜粋である。`src/verify.rs` で PARAM 所属と CALL の再帰を検証する。`functions[f]` は `Function { id, signature, root }`、`Signature.args` は `Vec<DType>` とする。`linearize` は関数 body の src 辺だけをたどり、CALL の body は別の関数として検査する。

既存の関数検証へ所属と呼出し辺の検査を追加する。

```rust
// src/verify.rs（既存ファイルへ組み込む抜粋）
fn verify_function(f: usize, functions: &[Function], g: &Graph, marks: &mut [u8]) -> Result<(),String> {
	if marks[f] == 1 { return Err("recursive CALL".into()); }
	if marks[f] == 2 { return Ok(()); }
	marks[f] = 1;
	let fun = &functions[f];
	for id in linearize(g,fun.root) {
		let node = &g.nodes[id.0];
		match node.op {
			Op::Param { owner, slot, dtype } => {
				if owner != fun.id || fun.signature.args.get(slot) != Some(&dtype) {
					return Err(format!("unbound PARAM at {}",id.0));
				}
			}
			Op::Call(callee) => {
				let target = functions.get(callee).ok_or("unknown function")?;
				if target.signature.args.len() != node.src.len() { return Err("CALL arity".into()); }
				for (&arg,&dtype) in node.src.iter().zip(&target.signature.args) {
					if g.dtype(arg) != dtype { return Err("CALL dtype".into()); }
				}
				verify_function(callee,functions,g,marks)?;
			}
			_ => {}
		}
	}
	marks[f] = 2;
	Ok(())
}
```

CALL の実引数は caller の body に属し、callee の PARAM 所属は再帰先の検証で調べる。この二つを同じ環境に混ぜない。関数の印が 1 の間に再びその関数へ到達した場合だけ再帰であり、複数回呼ばれる完了済み関数は許される。INDEX の範囲検証と STACK の戻り kind の照合も前コミットの型導出で済ませ、`assert_uop` は検証後の interpreter と C の両結果を expected と比較するホスト helper にする。

この入力の途中経過を並べる。

```text
pair signature=(f32,f32)->stack<f32,2>
CALL pair(2) -> CALL arity
body f -> PARAM(g,0) -> unbound PARAM
f CALL g, g CALL f -> marks[f]=1 -> recursive CALL
assert_uop(INDEX(pair(2,3),0),6) -> expected 6, actual 5 -> host test failure
```

- [ ] 未束縛 PARAM と他関数の PARAM の直接参照を拒否する。
- [ ] CALL の引数不足、dtype 不一致、再帰を拒否する。
- [ ] INDEX の負数と長さ以上の値を拒否する。
- [ ] `assert_uop` が意図的に誤った予定値で失敗することを確認する。

予定する負例は `INDEX(STACK(1,2),2)` と、引数一つの `pair` 呼出し。
エラーには関数名または UIR の行番号と対象ノードを含める。

## 完了確認

以下はこの章で作る example、fixture とテストの予定コマンドである。

```sh
# 作業ディレクトリ: chibigrad/
cargo test
cargo run --example function_stack -- --case pair --dump-graph
cargo run --example function_stack -- --roundtrip --dump-source
```

- [ ] 二つの呼出しで値を混同せず、`10` と `12` を返す。
- [ ] STACK の選択結果が `5` と `-1` になる。
- [ ] UIR の出力、入力、実行が一往復する。
- [ ] 異なる関数の PARAM が同じ NodeId に共有されない。
- [ ] 誤った予定値と不正な UIR の assertion が実際に失敗する。
- [ ] scalar の前章までの例も同じ pipeline で成功する。

## tinygrad ソースとの対応

参照 commit は `c22edff14c79310a9b2df62a78819cf77c8289fa`。
リンクの行番号は checkout で確認した。

| 実在するシンボル | ソース | 役割 | chibigrad で簡略化する点 |
|---|---|---|---|
| `UOp.call` | [ops.py](../../tinygrad/tinygrad/uop/ops.py#L1214) | opaque body を引数付きで呼ぶ | この章は純粋 scalar body を inline する |
| `UOp.call_with_outputs` | [ops.py](../../tinygrad/tinygrad/uop/ops.py#L1227) | ALLOC 出力と STORE body を作る | 出力 buffer CALL はメモリ導入後に接続する |
| `dtype_from_uop` | [ops.py](../../tinygrad/tinygrad/uop/ops.py#L116) | STACK の dtype を導出 | 同型の固定長列に限る |
| `UOp.index` | [ops.py](../../tinygrad/tinygrad/uop/ops.py#L547) | STACK の静的選択を含む INDEX | 値の選択だけを先に扱う |
| `render_uir` | [render.py](../../tinygrad/tinygrad/uop/render.py#L59) | UOp をテキストへ出す | 独自の小さい reader も追加する |
| `type_verify` | [spec.py](../../tinygrad/tinygrad/uop/spec.py#L34) | 各ノードを仕様と照合 | 少数演算の Rust match と assertion にする |

## 今は追加しないもの

メモリアドレスの INDEX、出力領域と効果付き CALL は [05](05-memory-runtime.md) で加える。
一般的な文法、パターン DSL、closure、再帰、外部関数 linker は加えない。
