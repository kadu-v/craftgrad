# 03. scalar の型と意味を決め、interpreter と rewrite を比較する

## このステップで動かすもの

[02-runtime-inputs.md](02-runtime-inputs.md) の scalar を三つの経路で評価する。
書換え前 interpreter、書換え後 interpreter、CPU 生成コードを比較する。
式の共有、dtype の検証、整数式の簡約をここで加える。
元計画の第 2 章に対応し、後の添字計算用の scalar ALU をそろえる。
以下の API、例名、コマンドと数値は未実装の作業計画である。

## 最小実装とデータ

`DType` は `Bool`、`I32`、`F32`、`Void` に限定する。
`Scalar` は `Bool(bool)`、`I32(i32)`、`F32Bits(u32)` を使う案にする。
`void` は効果の型であり、定数値や ALU 入力には使えない。
`NodeKey` は演算、dtype、payload、入力番号から作る。
f32 のキーはビット列とし、`+0` と `-0` を区別する。
NaN もビット列を保持し、数値の `==` で intern しない。

予定する入出力 IR は次の形になる。

```text
before: ADD(PARAM i32 slot=0, CONST i32 0)
after:  PARAM i32 slot=0
```

`ADD`、`SUB`、`MUL`、`NEG` は同じ数値 dtype のみを受け取る。
整数の加減乗算と左シフトは 32 bit の二の補数で wrap する契約にする。
C は `uint32_t` で計算し、ビット列を i32 に戻して signed overflow を避ける。
`XOR`、`AND`、`OR` は i32 または bool の同型二項にする。
`SHL` と `SHR` は i32 とし、シフト数は `0 <= count < 32` に限る。
右シフトは算術シフトとし、C の負整数 shift に頼らない helper を使う。
`DIV` と `MOD` は床除算と対応する剰余にする。
`-7 DIV 3 = -3`、`-7 MOD 3 = 2` が予定値になる。
C の `/` と `%` をそのまま出すと負数で違うため、補正 helper を生成する。
除数ゼロと `i32::MIN DIV -1` は定数なら検証時に拒否する。
動的 DIV は宣言範囲から非ゼロと例外不在を証明できる式に限る。
呼出し前に実際の入力が宣言範囲内かを検証する。
`CMPLT`、`CMPEQ`、`CMPNE` は同型数値を受け取り bool を返す。
`WHERE` は bool と同型二値を受け取り、その値の dtype を返す。
`RECIP` は f32、`MAX` は同型数値にする。
f32 MAX はどちらかが NaN なら NaN、両方ゼロなら `+0` を優先する契約にする。
これは tinygrad の Python `max` の全挙動をそのまま移す契約ではない。
`CAST` は明示操作とし、暗黙の i32 と f32 の混合は拒否する。
f32 から i32 は有限かつ範囲内の値をゼロ方向に丸める。
動的 cast の安全性を宣言範囲から証明できなければ、この版では拒否する。

導出情報は `dtype(id)`、`bounds(id)`、`cost(root)` から始める。
整数 bounds は保守的な区間とし、wrap の可能性があれば i32 全域へ広げる。
f32 の区間だけで NaN が存在しないと扱わない。
`cost` は到達可能な共有済み ALU 数を数え、実行時間とは呼ばない。

## この章のファイル配置

02 で抽出した Graph と pipeline を育てる。実行できる部品検証の main は book_03_intern に分け、Graph::intern、型導出、interpreter、rewrite の本体はそれぞれ既存モジュールへ追加する。

次はこの章の終わりまでに触る関連ファイルの配置案である。現在のcheckoutの一覧ではない。各コミットでは、その下に示す部分だけを追加・変更する。

```text
chibigrad/
├── examples/
│   ├── book_03_intern.rs  # HashMap による共有と f32 bits を確認する std の独立例。
│   └── scalar_rewrite.rs  # 書換え前後と C の三経路比較。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # 整数 wrap、床除算、算術右 shift と f32 helper の生成。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── runtime/
│   │   ├── cpu.rs  # 宣言された入力範囲の呼出し前検証。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── interpret.rs  # 書換えから再利用する既存 ALU evaluator。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rewrite.rs  # bottom-up rewrite_add と呼出しごとの memo。
│   ├── uop.rs  # 到達可能な ALU を数える cost と is_alu。
│   └── verify.rs  # 動的 DIV、shift、CAST の安全性を生成前に検証する。
├── tests/
│   └── scalar_rewrite.rs  # 型、境界、特殊値と元グラフ保持の統合確認。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

生成Cやshaderは一時ディレクトリへ出力するため、このRustプロジェクトのツリーには含めない。コード冒頭のコメントで、生成元と生成先を確認する。

## 小さなコミットで進める

### コミット 1. 同じ式を共有し、従来の CPU 経路で動かす

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_03_intern.rs  # [追加] HashMap による共有と f32 bits を確認する std の独立例。
│   └── scalar_rewrite.rs  # [追加] 共有された式を従来の CPU 経路で実行する入口。
├── src/
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── uop.rs  # [変更] Graph の構造キーと Graph::intern。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

単独例は examples/book_03_intern.rs へ全体を置く。本体は src/uop.rs の Graph に intern 表を追加し、掲載した Graph::intern を実装する。既存のノード構築をこの入口へそろえ、Op と UOp の payload を f32 bits に変更する。examples/scalar_rewrite.rs では共有した (a+b)*(a+b) を作り、02 の compile と run へ渡す。


既存の arena では同じ演算を二度作ると二つの番号になるため、今回はノードを構造で検索してから追加する。変更先候補は `src/uop.rs` の設計案 `Arena::intern(key)` である。key は演算種別、f32 dtype、定数なら `to_bits()`、PARAM なら slot、ALU なら順序付き入力番号列を持つ。HashMap に同じ key があれば既存の `NodeId` を返し、なければ Vec の末尾へ UOp を追加してから表へ登録する。値を評価して等しいかどうかを調べる処理ではない。

例えば `%0=PARAM(0)`、`%1=PARAM(1)` に対して ADD(0,1) を二回作ると、最初の呼出しだけが `%2` を追加し、二回目も 2 を返す。その二つを掛ける MUL は入力 `[2,2]` になる。linearize は `%2` を一度だけ出力し、入力 2、3 なら ADD の中間値は 5、MUL の結果は 25 になる。ADD(1,0) は別 key とし、交換則による並べ替えはここへ混ぜない。

f32 の key に数値の等値判定を使わない点を局所テストで固定する。`+0` と `-0` は bits が違うので別ノード、同じ NaN payload を二度作れば同じノード、違う payload なら別ノードになる予定である。ノード共有と CPU 実行の両方を確認し、HashMap の戻り番号だけで完了にしない。既存の dtype 固定の検証と C renderer はそのまま使い、以降の rewrite が新しい式を作るときにもこの入口を通るようにする。


単独で実行できる例である。`src/uop.rs` の intern を最小の arena として切り出す。payload を `Op` に持たせた `UOp { op, src }` 自体が、この純粋部分集合の構造キーになる。

Arena、局所 Op と main を含むブロック全体を独立例として保存する。本体の Graph 定義とは別である。

```rust
// examples/book_03_intern.rs（単独で実行する例）
use std::collections::HashMap;
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Op { Const(u32), Param(usize), Add, Mul }
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct UOp { op: Op, src: Vec<usize> }
#[derive(Default)]
struct Arena { nodes: Vec<UOp>, intern: HashMap<UOp, usize> }
impl Arena {
	fn add(&mut self, op: Op, src: Vec<usize>) -> usize {
		let node = UOp { op, src };
		if let Some(&id) = self.intern.get(&node) { return id; }
		let id = self.nodes.len();
		self.nodes.push(node.clone());
		self.intern.insert(node, id);
		id
	}
}
fn main() {
	let mut a = Arena::default();
	let x = a.add(Op::Param(0), vec![]);
	let y = a.add(Op::Param(1), vec![]);
	let sum = a.add(Op::Add, vec![x,y]);
	assert_eq!(sum, a.add(Op::Add, vec![x,y]));
	assert_ne!(sum, a.add(Op::Add, vec![y,x]));
	let square = a.add(Op::Mul, vec![sum,sum]);
	assert_eq!(a.nodes[square].src, [sum,sum]);
	assert_ne!(a.add(Op::Const(0.0f32.to_bits()), vec![]), a.add(Op::Const((-0.0f32).to_bits()), vec![]));
	let nan = a.add(Op::Const(0x7fc00042), vec![]);
	assert_eq!(nan, a.add(Op::Const(0x7fc00042), vec![]));
}
```

`nodes` に追加する前の検索が、二度目の ADD を既存の番号へ戻す。キーの `src` は列なので `[x,y]` と `[y,x]` は別になる。`Const(u32)` はこのコミットの f32 bits 専用 payload で、次では `Const(Scalar)` に拡張する。効果がまだ存在しないので全ノードを intern できるが、05 の STORE と ALLOC にはこの規則をそのまま適用しない。

この入力の途中経過を並べる。

```text
PARAM(0)->%0, PARAM(1)->%1
ADD(%0,%1)->%2, ADD(%0,%1)->%2
MUL(%2,%2)->%4（逆順 ADD を作ったため %3 は別ノード）
+0 bits=0x00000000, -0 bits=0x80000000 -> distinct ids
```

- [ ] arena に `HashMap<NodeKey, NodeId>` を追加する。
- [ ] f32 のビットキーと演算の入力順を保持する。
- [ ] 同じ `a+b` を二回作ると同じ番号になるようにする。
- [ ] `(a+b)*(a+b)` を C まで通し、入力 `2,3` で `25` を得る。

このコミットでは f32 のままでよい。
`a+b` と `b+a` の自動的な同一化はまだ行わない。


組込み用の抜粋である。上の単独例を本体へ統合するときは、01 の `Graph.nodes: Vec<UOp>` を維持して、`intern: HashMap<UOp,NodeId>` を足す。ここから後の章では `Graph::intern(op, &[NodeId]) -> NodeId` に名前と署名を統一する。`NodeId` は `Copy + Eq + Hash`、`Op` と `UOp` は `Clone + Eq + Hash` を導出し、f32 payload は引き続き bits で持つ。

既存 Graph の impl に intern を追加する。

```rust
// src/uop.rs（既存ファイルへ組み込む抜粋）
impl Graph {
	fn intern(&mut self, op: Op, src: &[NodeId]) -> NodeId {
		let node = UOp { op, src: src.to_vec() };
		if let Some(&id) = self.intern.get(&node) { return id; }
		let id = NodeId(self.nodes.len());
		self.nodes.push(node.clone());
		self.intern.insert(node,id);
		id
	}
}
```

単独例の `Arena::add(Op::Add,vec![x,y])` は、本体では `graph.intern(Op::Add,&[x,y])` になる。src を借用する入口が必要な列だけを所有化するので、書換えや parser は同じ interner を呼べる。ALLOC と STORE を追加する 05 ではこの純粋ノード用入口と、固有 ID を発行する効果用入口を分ける。

### コミット 2. 四つの dtype と interpreter を CPU にそろえる

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── scalar_rewrite.rs  # [変更] WHERE の両分岐と interpreter、CPU の比較。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # [変更] 検証済み dtype ごとの式と wrapper。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── runtime/
│   │   ├── cpu.rs  # [変更] Bool と I32 の pack と型付き出力。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── interpret.rs  # [追加] 型付き値表、eval_add と eval_where。
│   ├── lib.rs  # [変更] interpret の宣言と比較に使う API の公開。
│   ├── uop.rs  # [変更] DType、Scalar、演算 payload と Scalar::dtype。
│   └── verify.rs  # [変更] binary_type と型・入力数の検証。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

DType と Scalar は src/uop.rs に追加する。抜粋の binary_type は verify.rs、eval_add と eval_where は新しい interpret.rs に置き、src/lib.rs へ interpret の宣言を加える。interpreter が使う型と evaluator は crate 内で共用し、example が必要とする評価入口だけを公開する。renderer/c.rs と runtime/cpu.rs も同じ型規則と ABI を使うように広げる。

新しいモジュールは、既存の宣言を残して次を追記する。内部の補助関数は各モジュールに置き、exampleから使う入口だけを公開する。

```rust
// src/lib.rs（モジュール宣言を追加）
pub mod interpret;
```


演算の意味を Rust の interpreter と C 生成でそろえる。候補ファイルは `src/uop.rs` の `DType` と `Scalar`、`src/verify.rs` の設計案 `infer_type`、`src/interpret.rs` の `eval_alu` である。先に演算ごとの引数数と入力型、結果型の表を match で定める。例えば CMPLT は同型数値二つから Bool、WHERE は Bool 一つと同型値二つからその値の dtype を返す。型が合わないノードを interpreter や renderer が別々に補正しない。

interpreter は既存の依存順を使い、各 NodeId の結果を型付き Scalar の表へ置く。`WHERE(a<3,a*2,a+1)` で a=2 なら比較は true、二枝は 4 と 3、選択結果は 4 になる。a=5 なら比較は false、二枝は 10 と 6、結果は 6 となる。純粋な scalar DAG の段階では両枝を評価してよいが、各枝自体が有効な演算でなければならない。後の guard 付きメモリアクセスをこの性質だけで実装しない。

ABI は f32 の bits 規則を保ち、I32 は下位 32 bit の二の補数、Bool は 0 または 1 と定め、C でも対応する型へ戻す。出力 wrapper も root の dtype に合わせて読む。局所テストでは WHERE の二入力に加え、I32 と F32 の ADD、Void の定数を拒否する。F32 の演算ごとに f32 へ丸め、C の fast-math と積和縮約を止めて比較する。これで型検証、interpreter、CPU のいずれも同じ仕様に対して確認できる。


組込み用の抜粋である。`src/verify.rs` の二項型導出と `src/interpret.rs` の評価を示す。`DType` と `Scalar` は本文の四型・三値型、`Op` は型付き payload を持つ。ここでは ADD と比較の差を取り出し、SUB、MUL、NEG も同じ型判定の数値集合へ追加する。

binary_type と評価関数を責務ごとに分けて追加する。

```rust
// src/verify.rs（既存ファイルへ組み込む抜粋）
fn binary_type(op: &Op, a: DType, b: DType) -> Result<DType, String> {
	if a != b || !matches!(a, DType::I32 | DType::F32) {
		return Err("expected two values of the same numeric dtype".into());
	}
	match op {
		Op::Add | Op::Sub | Op::Mul | Op::Max => Ok(a),
		Op::CmpLt | Op::CmpEq | Op::CmpNe => Ok(DType::Bool),
		_ => Err("not a binary numeric operation".into()),
	}
}
// src/interpret.rs（既存ファイルへ組み込む抜粋）
fn eval_add(a: Scalar, b: Scalar) -> Result<Scalar, String> {
	match (a,b) {
		(Scalar::I32(x), Scalar::I32(y)) => Ok(Scalar::I32(x.wrapping_add(y))),
		(Scalar::F32Bits(x), Scalar::F32Bits(y)) => {
			let value: f32 = f32::from_bits(x) + f32::from_bits(y);
			Ok(Scalar::F32Bits(value.to_bits()))
		}
		_ => Err("ADD dtype mismatch".into()),
	}
}
fn eval_where(c: Scalar, a: Scalar, b: Scalar) -> Result<Scalar, String> {
	if a.dtype() != b.dtype() { return Err("WHERE branch dtype mismatch".into()); }
	match c {
		Scalar::Bool(true) => Ok(a),
		Scalar::Bool(false) => Ok(b),
		_ => Err("WHERE condition must be bool".into()),
	}
}
```

`Scalar::dtype(&self)->DType` は variant をそのまま対応させる導出関数である。ADD の整数枝だけが wrap を選び、f32 枝は各命令で f32 の結果を bits に戻す。WHERE の条件と枝は別の型規則を持つので、数値二項演算の検証を使い回して bool を数値化しない。interpreter の値表にはこの `Scalar` を保存し、C でも同じ検証済み型に応じて式を選ぶ。

この入力の途中経過を並べる。

```text
a=2: CMPLT(2,3)=true, MUL(2,2)=4, ADD(2,1)=3 -> WHERE=4
a=5: CMPLT(5,3)=false, MUL(5,2)=10, ADD(5,1)=6 -> WHERE=6
ADD(I32(2),F32Bits(2.0bits)) -> dtype mismatch
```

- [ ] dtype の導出と各演算の入力数を一か所で検証する。
- [ ] bool と i32 の ABI pack と C の復元処理を追加する。
- [ ] 加減乗算、比較、WHERE、CAST を interpreter と renderer に加える。
- [ ] void 定数と混合 dtype を検証時に拒否する。
- [ ] `WHERE(a<3,a*2,a+1)` を `a=2` で `4`、`a=5` で `6` にする。

f32 interpreter は演算ごとに f32 へ丸める。
C は fast-math を無効にし、積和の意図しない縮約を避ける。

### コミット 3. 添字用 ALU と境界の契約を実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── scalar_rewrite.rs  # [変更] 境界入力を interpreter と生成 C へ渡す。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # [変更] 整数 wrap、床除算、算術右 shift と f32 helper の生成。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── runtime/
│   │   ├── cpu.rs  # [変更] 宣言された入力範囲の呼出し前検証。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── interpret.rs  # [変更] floor_div_mod、arithmetic_shr と特殊値の評価。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── uop.rs  # [変更] RangeI32、bounds と prove_division。
│   └── verify.rs  # [変更] 動的 DIV、shift、CAST の安全性を生成前に検証する。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

floor_div_mod と arithmetic_shr は interpret.rs に追加し、RangeI32 と prove_division は uop.rs の bounds 導出の近くに置く。renderer/c.rs は同じ契約の C helper を生成する。verify.rs は範囲証明を呼び、runtime/cpu.rs は pack より前に実値が宣言範囲内かを検査する。


添字に使う整数演算は、Rust や C の既定動作をそのまま採用せず、本文の契約を `eval_alu` と C helper に実装する。候補は `src/interpret.rs` の設計案 `floor_div_mod`、`src/renderer/c.rs` の整数 helper、`src/uop.rs` の `bounds` である。最初に wrap する加減乗算を unsigned bits でそろえ、次に bit 演算と shift、最後に DIV と MOD を足す。算術右 shift は負値を符号拡張する helper とし、シフト数 32 を実行してから扱うことはしない。

床除算の例では C の切捨て除算で `-7/3=-2`、余り `-1` をまず得る。余りが非ゼロで除数と符号が違う場合に商を 1 減らし、余りへ除数を足すと、商 -3、余り 2 となり `-7=(-3)*3+2` を保つ。ゼロ除数と最小整数を -1 で割る例は、その helper を呼ぶ前の検証対象にする。`(5 XOR 3) SHL 1` は bits 6 を左へ一つ動かして 12 になる。

動的入力には宣言区間を付け、bounds を子から導出する。除数が `[1,3]` ならゼロを含まないが `[-1,1]` なら拒否する。run でも実値が宣言区間内か調べ、証明の前提を維持する。局所テストでは整数最大値への加算、負の DIV と MOD、不正 shift、範囲外入力を別々に比較する。RECIP の負のゼロと MAX の NaN も追加し、NaN は数値等値ではなく分類、ゼロは bits で照合する。これらの evaluator を次の定数畳み込みでも再利用する。


組込み用の抜粋である。`src/interpret.rs` の床除算と `src/uop.rs` の範囲証明を、実行前のエラーも含めて示す。`RangeI32` は閉区間 `[lo,hi]` を持ち、生成前に `lo<=hi` が検証されている。

evaluator と bounds の証明を分ける。

```rust
// src/interpret.rs（既存ファイルへ組み込む抜粋）
fn floor_div_mod(a: i32, b: i32) -> Result<(i32,i32), String> {
	if b == 0 || (a == i32::MIN && b == -1) { return Err("invalid division".into()); }
	let (mut q, mut r) = (a / b, a % b);
	if r != 0 && (r < 0) != (b < 0) {
		q -= 1;
		r += b;
	}
	Ok((q,r))
}
// src/uop.rs（既存ファイルへ組み込む抜粋）
fn prove_division(a: RangeI32, b: RangeI32) -> Result<(), String> {
	if b.lo <= 0 && 0 <= b.hi { return Err("divisor may be zero".into()); }
	if a.lo == i32::MIN && b.lo <= -1 && -1 <= b.hi {
		return Err("division may overflow".into());
	}
	Ok(())
}
// src/interpret.rs（既存ファイルへ組み込む抜粋）
fn arithmetic_shr(x: i32, count: u32) -> Result<i32, String> {
	if count >= 32 { return Err("shift count must be below 32".into()); }
	if count == 0 { return Ok(x); }
	let bits = x as u32;
	let sign = if x < 0 { u32::MAX << (32-count) } else { 0 };
	Ok(((bits >> count) | sign) as i32)
}
```

切捨て商と余りを求め、余りと除数の符号が違うときだけ床の方向へ一段動かす。`-7,3` では `q=-2,r=-1` が `q=-3,r=2` になる。`prove_division` は実値を知らない時点で危険な組が区間に含まれるかを調べ、run の区間検証がその前提を守る。C helper も先に例外を排除して同じ補正を行い、wrap の加減乗算は `uint32_t` 上で計算した bits を `memcpy` で `int32_t` に戻す。

この入力の途中経過を並べる。

```text
floor_div_mod(-7,3) -> (-3,2), (-3)*3+2=-7
floor_div_mod(7,-3) -> (-3,-2), (-3)*(-3)-2=7
prove divisor [1,3] -> safe; [-1,1] -> reject
arithmetic_shr(-8,1) -> -4; count=32 -> reject
```

- [ ] XOR、AND、OR、SHL、SHR、DIV、MOD を両経路に加える。
- [ ] RECIP と MAX を本文の f32 契約で実行する。
- [ ] bounds と動的入力の範囲検証を追加する。
- [ ] 危険な DIV、shift、CAST は安全性を証明できなければ拒否する。
- [ ] `((5 XOR 3) SHL 1)=12` と負数の DIV、MOD を C まで通す。

境界の予定例は `i32::MAX+1 = i32::MIN`、シフト数 `32` はエラー。
`RECIP(-0.0)` は負の無限大、`CMPLT(NaN,1)` は false を確認する。

### コミット 4. bottom-up rewrite と cost を観測する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── scalar_rewrite.rs  # [変更] 書換え前後と C の三経路比較。
├── src/
│   ├── interpret.rs  # [確認] 書換えから再利用する既存 ALU evaluator。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rewrite.rs  # [変更] bottom-up rewrite_add と呼出しごとの memo。
│   └── uop.rs  # [変更] 到達可能な ALU を数える cost と is_alu。
├── tests/
│   └── scalar_rewrite.rs  # [追加] 型、境界、特殊値と元グラフ保持の統合確認。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

02 で移した恒等 rewrite を src/rewrite.rs で bottom-up の実装に置き換え、掲載した rewrite_add を置く。cost は src/uop.rs の導出情報として追加し、既存の linearize を使って到達可能な ALU だけを数える。eval_add は interpret.rs の実装を参照し、畳み込み用に複製しない。tests/scalar_rewrite.rs は公開された構築、書換え、評価、CPU 実行を組み合わせて比較する。


rewrite の入力は元の root と不変なノード群、出力は新しい root と追加されたノードである。候補ファイル `src/rewrite.rs` の設計案 `rewrite_node(old, memo)` は、まず old の全入力を再帰的に書き換え、その戻り番号で同じ演算を intern し直す。次にその再構築ノードへ局所規則を適用する。`memo[old_id]=new_id` を今回の rewrite 呼出し内だけで保持すると、共有される部分式を重複処理せず、元の番号との対応も dump できる。

`ADD(PARAM i32 slot=0, CONST i32 0)` なら PARAM と定数の処理後に ADD を検査し、I32 で右入力が 0 なので PARAM の番号を返す。旧 root は残るが、新 root から到達する ALU は一つ減る。`ADD(CONST 2,CONST 3)` は、両入力が定数であることを確認してから検証済み `eval_alu` で 5 を求め、CONST 5 を intern する。定数演算の意味を rewrite 内へもう一組書かない。f32 の零に関する恒等規則は使わない。

設計案 `cost(root)` は visited 集合を使って到達可能な ALU だけを一度ずつ数える。arena 全体には古い ADD が残るので Vec の長さを cost としない。局所テストは入力 -7、0、5 に対し旧 root の interpreter、新 root の interpreter、新 root の C 実行を比較し、I32 x+0 の cost が 1 から 0 になることを調べる。dtype が同じこと、bounds が各実値を含むこと、旧 root の dump が変更されないことも確認する。値が合っていても元グラフを破壊する書換えは、この契約を満たさない。


組込み用の抜粋である。`src/rewrite.rs` へ置く整数 ADD の bottom-up 書換えを示す。`Graph::intern(op,src)->NodeId` はコミット 1 の intern で、`Op::Const(Scalar)` と `Op::Add` を使う。`Scalar` は Copy、`NodeId` は番号として比較できる。

rewrite_add は書換え、cost は Graph の導出情報として追加する。

```rust
// src/rewrite.rs（既存ファイルへ組み込む抜粋）
fn rewrite_add(g: &mut Graph, old: NodeId, memo: &mut std::collections::HashMap<NodeId,NodeId>) -> NodeId {
	if let Some(&id) = memo.get(&old) { return id; }
	let node = g.nodes[old.0].clone();
	let src: Vec<_> = node.src.iter().map(|&s| rewrite_add(g,s,memo)).collect();
	let mut result = g.intern(node.op.clone(), &src);
	if matches!(node.op, Op::Add) {
		match (&g.nodes[src[0].0].op, &g.nodes[src[1].0].op) {
			(Op::Const(a), Op::Const(b)) => {
				let value = eval_add(*a,*b).expect("verified ADD");
				result = g.intern(Op::Const(value), &[]);
			}
			(_, Op::Const(Scalar::I32(0))) if g.dtype(src[0]) == DType::I32 => result = src[0],
			_ => {}
		}
	}
	memo.insert(old,result);
	result
}
// src/uop.rs（既存ファイルへ組み込む抜粋）
fn cost(g: &Graph, root: NodeId) -> usize {
	linearize(g,root).iter().filter(|id| g.nodes[id.0].op.is_alu()).count()
}
```

`g.dtype(id)` はコミット 2 の型導出、`is_alu` は ADD、SUB、MUL などの scalar ALU を判別する match、`linearize` は 01 の共有を一度だけ訪れる DFS である。子を置換した後で局所規則を適用するため、子の定数化が親へ伝わる。`g.nodes[old]` には代入せず、新しいノードを追加するだけなので旧 root もそのまま評価できる。`eval_add` を再利用する定数枝と、I32 に限定した零の枝を分けることで f32 の符号付き零を保つ。

この入力の途中経過を並べる。

```text
old: ADD(PARAM i32 %0, CONST i32 0), cost=1
new: PARAM i32 %0, cost=0
inputs -7,0,5 -> old interpreter == new interpreter == C
old nodes: unchanged; arena length: unchanged or larger, reachable ALU: smaller
```

- [ ] 子を先に再構築する rewrite を `match` で作る。
- [ ] 検証済み ALU evaluator を定数畳み込みにも使う。
- [ ] `i32 x+0 → x`、`x XOR 0 → x`、`WHERE(true,a,b) → a` を加える。
- [ ] 前後の dtype、bounds、ALU 数を dump に出す。
- [ ] 三つの実行経路を入力 `-7,0,5` と定数例で比較する。

f32 の `x*0 → 0` と `x+0 → x` はまだ加えない。
`i32 x+0` の cost は `1` から `0` へ減る予定になる。
元グラフを変更せず、新しい root を返す。

## 完了確認

以下はこの章で作る example とテストの予定コマンドである。

```sh
# 作業ディレクトリ: chibigrad/
cargo test
cargo run --example scalar_rewrite -- --case integer --dump-graph
cargo run --example scalar_rewrite -- --case boundaries --dump-source
```

- [ ] 床除算、wrap、shift、f32 の特殊値で三経路が合う。
- [ ] NaN は `is_nan`、符号付きゼロはビット列で比較する。
- [ ] 有限の一般 f32 は予定許容値 `atol=1e-6, rtol=1e-6` で比較する。
- [ ] dtype 不一致、void 入力、範囲外入力は生成か呼出しの前に失敗する。
- [ ] bounds が結果を含み、証明できない式を安全と判定しない。
- [ ] 前章までの成功例も通る。

## tinygrad ソースとの対応

参照 commit は `c22edff14c79310a9b2df62a78819cf77c8289fa`。
行番号は checkout で確認した。意味の違いは右欄に記す。

| 実在するシンボル | ソース | 役割 | chibigrad で簡略化する点 |
|---|---|---|---|
| `UOpMetaClass.__call__` | [ops.py](../../tinygrad/tinygrad/uop/ops.py#L181) | ノードを構造で共有 | Rust arena の intern 表にする |
| `dtype_from_uop` | [ops.py](../../tinygrad/tinygrad/uop/ops.py#L116) | dtype の導出 | 四型と明示 cast にする |
| `python_alu`, `exec_alu` | [ops.py](../../tinygrad/tinygrad/uop/ops.py#L1382) | ALU の参照評価 | MAX と整数 wrap は本文の契約を使う |
| `UOp._min_max` | [ops.py](../../tinygrad/tinygrad/uop/ops.py#L1054) | 演算結果の区間 | 静的 i32 区間から始める |
| `fold_const_alu`, `no_overflow` | [symbolic.py](../../tinygrad/tinygrad/uop/symbolic.py#L30) | 畳み込みと整数境界 | DSL を作らず `match` にする |
| `Estimates.from_uops` | [renderer](../../tinygrad/tinygrad/renderer/__init__.py#L26) | 操作数とアクセス量 | scalar ALU 数だけを出す |

## 今は追加しないもの

純粋関数と UIR は [04-functions-stack.md](04-functions-stack.md) で加える。
メモリ操作の共有規則は [05](05-memory-runtime.md) で別途定義する。
超越関数群は第 12 段階、symbolic shape は第 20 段階で加える。
