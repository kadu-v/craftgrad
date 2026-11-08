# 08. 一次元の shape と REDUCE を添字付きループへ変換する

## このステップで動かすもの

[07-scalar-reductions.md](07-scalar-reductions.md) の手書き kernel を compiler が作る。
入力 `[1,2,3]` の `x*2+1` を `[3,5,7]` にする。
高水準 `REDUCE(ADD,x)` から総和 `6` の累積ループを生成する。
元計画第 5 章の shape、REDUCE、rangeify に対応する。
以下の API、IR、例名、コマンドと数値は未実装の作業計画である。

## 最小実装とデータ

shape は scalar の `[]` と rank 1 の `[N]` だけにする。
N は静的で正の整数とし、要素数の積と byte 数は checked 演算で求める。
空配列のループは前章で扱ったが、この初版の shaped storage は N が 1 以上に限る。
高水準ノードに `shape(id)` を導出し、scalar と配列を区別する。
同じ長さの二項 ALU と、明示的な `EXPAND(scalar,[N])` を扱う。
`[1]` から `[N]` への expand も、読む入力要素が 0 の場合に限って許す。
`[2]` を `[3]` へ expand する操作は拒否する。

予定する高水準入力は次の形になる。

```text
%x = PARAM f32 shape=[3]
%m = MUL %x EXPAND(CONST f32 2, [3])
%y = ADD %m EXPAND(CONST f32 1, [3])
```

まず演算ごとに中間バッファと kernel を作る。
上の例の予定出力は二つの kernel になる。

```text
kernel0: for i in 0..3 { tmp[i] = x[i] * 2 }
kernel1: for j in 0..3 { out[j] = tmp[j] + 1 }
```

rangeify は `value_at(node,coordinate,scope)` のような処理から始める。
出力 RANGE の座標を、ALU の同じ座標へ伝える。
PARAM に到達したら INDEX と LOAD にし、EXPAND の入力には座標 0 を伝える。
scalar からの EXPAND は LOAD を増やさず、同じ scalar 値を返す。
最終 STORE が出力バッファを埋め、END が出力 RANGE を閉じる。

NodeId だけをキーに添字付きの式をキャッシュしない。
同じ高水準ノードでも、`x[i]` と `x[j]` は違う計算位置になる。
初版はこのキャッシュを持たず、必要な場所で変換する案でよい。
追加するなら座標式と region、読む state まで含める。

`REDUCE(ADD,x)` は shape `[N]` を scalar `[]` にする。
出力座標と独立の reduction RANGE k を作る。
値 `x[k]` を前章の accumulator に足し、END 後の値を一要素出力へ STORE する。
shape 変換の REDUCE と、低水準の逐次 loop を区別して dump する。

## この章のファイル配置

shape 導出と座標 lowering が必要になるここで、shape と rangeify を追加する。既存の loop、interpreter、CPU runtime を使い、二 kernel の実行順はまだ example が明示する。独立した座標変換の main は別ファイルにする。

次はこの章の終わりまでに触る関連ファイルの配置案である。現在のcheckoutの一覧ではない。各コミットでは、その下に示す部分だけを追加・変更する。

```text
chibigrad/
├── examples/
│   ├── book_08_expand_coordinates.rs  # EXPAND の入力座標を確認する std の独立例。
│   └── shape_rank1.rs  # MAX と異なる座標・束縛環境の比較。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # 既存の MAX の特殊値 helper。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── interpret.rs  # 既存 scalar MAX と一致する max_f32 と比較用 reduce_max。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── linearize.rs  # 既存 Region による初期化、更新と最終 STORE。
│   ├── rangeify.rs  # ADD と MAX の単位元を選ぶ lower_reduce。
│   ├── shape.rs  # reduce_shape による [N] から [] への導出。
│   ├── uop.rs  # REDUCE ADD と対象軸の op payload。
│   └── verify.rs  # backend 入力へ抽象 REDUCE を残さない検証。
├── tests/
│   └── rangeify.rs  # shape、EXPAND、reduction と座標混入の統合確認。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

生成Cやshaderは一時ディレクトリへ出力するため、このRustプロジェクトのツリーには含めない。コード冒頭のコメントで、生成元と生成先を確認する。

## 小さなコミットで進める

### コミット 1. shape 付き加算一つを既存 kernel へ落とす

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── shape_rank1.rs  # [追加] shape 付き加算を構築して既存 CPU で実行する入口。
├── src/
│   ├── lib.rs  # [変更] shape と rangeify の宣言と lowering の公開入口。
│   ├── linearize.rs  # [変更] Rank1Kernel の式を既存 Region の STORE と END へ接続する。
│   ├── rangeify.rs  # [追加] Coordinate、ScalarExpr、Rank1Kernel と value_at、lower_add。
│   ├── shape.rs  # [追加] Graph::shape と scalar、rank 1 の検証。
│   └── uop.rs  # [変更] PARAM の宣言 shape と高水準ノードの情報。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/shape.rs と src/rangeify.rs を必要な実装と一緒に追加し、lib.rs に両モジュールの宣言を加える。Graph の型定義は uop.rs に残し、Graph::shape の導出実装は shape.rs に置く。Coordinate、ScalarExpr、Rank1Kernel、value_at と lower_add は rangeify.rs にまとめ、ScopeId は既存 linearize の region と対応させる。examples/shape_rank1.rs は出力を既存の loop emitter と CPU 実行へ渡す。

新しいモジュールは、既存の宣言を残して次を追記する。内部の補助関数は各モジュールに置き、exampleから使う入口だけを公開する。

```rust
// src/lib.rs（モジュール宣言を追加）
pub mod rangeify;
pub mod shape;
```


example が手書きしていた添字を、shape 付きノードから自動生成する。候補ファイル `src/shape.rs` の設計案 `shape(id)` は PARAM の宣言 shape と演算の入力 shape から結果を導出し、`src/rangeify.rs` の `lower_elementwise(root)` が出力 loop を作る。ここでは rank 1 の二入力 ADD 一つを対象にする。入力 shape `[3]` と `[3]` が一致した後で、長さ 3 の新規出力、RANGE i、STORE、END を組み立てる。

中心になる設計案 `value_at(node,coordinate,scope)` は、高水準ノードの指定要素を求める低水準の scalar ノードを返す。ADD の位置 i を要求されたら左右の入力にも i を渡し、PARAM に到達したら、その kernel の buffer PARAM に対する INDEX と LOAD を作る。例えば i=1 では x[1]=2、y[1]=20 を読み、ADD の scalar 値 22 を out[1] へ書く。shape の `[3]` 自体を C の scalar 演算へ渡さず、実際の添字を持つ LOAD と ADD へ変えることが lowering である。

実装順は shape 検証、出力 allocation と軸の生成、座標伝播、STORE と END、既存 kernel verifier の順とする。局所テストでは高水準 builder に `[1,2,3]` と `[10,20,30]` を束縛して `[11,22,33]` を得て、dump に三つの手書き STORE でなく一本の RANGE が現れることを見る。長さ 2 と 3 の ADD は allocation や clang の前に拒否する。低水準 interpreter と CPU runtime は変更せず使えるため、高水準 shape の間違いと低水準実行の間違いを別々に追える。


組込み用の抜粋である。`src/rangeify.rs` で「指定された一要素」を低水準の式へ変える。`ScalarExpr` は `Load { slot, coordinate, scope }`、`Add(Box<Self>,Box<Self>)` を持つ式の木、`Coordinate` は scalar 整数式、`ScopeId` は region の番号である。

指定要素の lowering と rank 1 kernel の構築を同じモジュールへ追加する。

```rust
// src/rangeify.rs（既存ファイルへ組み込む抜粋）
fn value_at(g: &Graph, id: NodeId, coordinate: Coordinate, scope: ScopeId) -> Result<ScalarExpr,String> {
	let node = &g.nodes[id.0];
	match node.op {
		Op::Param { slot, .. } => Ok(ScalarExpr::Load { slot, coordinate, scope }),
		Op::Add => {
			let a = value_at(g,node.src[0],coordinate.clone(),scope)?;
			let b = value_at(g,node.src[1],coordinate,scope)?;
			Ok(ScalarExpr::Add(Box::new(a),Box::new(b)))
		}
		_ => Err("unsupported rank1 expression".into()),
	}
}
fn lower_add(g: &Graph, root: NodeId, axis: AxisId, scope: ScopeId) -> Result<Rank1Kernel,String> {
	let left = g.shape(g.nodes[root.0].src[0])?;
	let right = g.shape(g.nodes[root.0].src[1])?;
	if left != right || left.len() != 1 { return Err("expected matching rank1 inputs".into()); }
	let len = left[0];
	if len == 0 || len > i32::MAX as usize { return Err("invalid shaped length".into()); }
	len.checked_mul(4).ok_or("output byte size overflow")?;
	let body = value_at(g,root,Coordinate::Axis(axis),scope)?;
	Ok(Rank1Kernel { axis,scope,len,body })
}
```

`Graph::shape(id)->Result<Vec<usize>,String>` は PARAM の宣言を返し、ADD では子の一致を検証する導出関数である。`Rank1Kernel` の body は out[axis] に STORE する scalar 式で、06 の region emitter が `RANGE(axis,len) → STORE(body) → END(axis)` に変える。左右の再帰へ同じ coordinate を渡すことが rangeify の中核であり、shape の長さを値の代わりに ADD へ渡さない。ここには memo を持たせず、scope と座標の混入を避ける。

この入力の途中経過を並べる。

```text
high: ADD(PARAM x shape=[3],PARAM y shape=[3])
value_at(root,i,scope0) -> Add(Load(x,i,scope0),Load(y,i,scope0))
low: RANGE i=0..3; STORE out[i],x[i]+y[i]; END i
i=1 -> LOAD 2,20 -> 22
```

- [ ] shape を scalar と rank 1 に限定して導出する。
- [ ] 同じ長さの入力二つから output RANGE を作る。
- [ ] ALU の座標を入力へ伝え、INDEX と LOAD を生成する。
- [ ] 高水準加算から `[1,2,3]+[10,20,30]=[11,22,33]` を得る。
- [ ] 長さ不一致を kernel 生成前に拒否する。

下位の RANGE、LOAD、STORE、CPU runtime は前章までのものを使う。
shape の module だけを作って終わるコミットにはしない。

### コミット 2. EXPAND と二演算を別 kernel で実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_08_expand_coordinates.rs  # [追加] EXPAND の入力座標を確認する std の独立例。
│   └── shape_rank1.rs  # [変更] 二 kernel を同期順で呼び、tmp と最終出力を比較する。
├── src/
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rangeify.rs  # [変更] EXPAND の座標変換と保存済み中間入力への置換。
│   ├── shape.rs  # [変更] verify_expand による scalar と singleton の検証。
│   └── uop.rs  # [変更] EXPAND の op と出力 shape の情報。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

verify_expand は src/shape.rs、value_at の EXPAND 分岐と scalar 式の拡張は rangeify.rs に追加する。掲載した局所型と main を持つ例は examples/book_08_expand_coordinates.rs に全体を保存し、そのまま本体の型定義へ貼らない。shape_rank1.rs は MUL の保存と ADD の実行を明示的に二回呼び、まだ schedule モジュールを追加しない。


長さの異なる値を暗黙に合わせず、EXPAND を座標変換として追加する。候補ファイル `src/shape.rs` の設計案 `verify_expand` は入力が scalar `[]` または `[1]` であることを調べ、`src/rangeify.rs` の `value_at` が EXPAND の出力座標を入力へ写す。scalar 入力には配列座標が不要なので CONST や scalar PARAM をそのまま返し、長さ 1 の入力には常に位置 0 を渡す。出力位置 i をそのまま bias[i] として読む処理にはしない。

`x*2+1` は、まず MUL だけを root とする kernel を作り、tmp の長さ 3 を確保する。i=0、1、2 の結果 2、4、6 を tmp へ書いた後、その完了 state を次の ADD kernel の入力に束縛する。ADD は独立した軸 j を持ち、tmp[j]+1 を計算して 3、5、7 を out へ書く。この段階では MUL の式を ADD の body へ再帰的に埋め込まず、materialized な入力として読むことで二つの kernel を保つ。

局所テストでは tmp の `[2,4,6]` と最終 `[3,5,7]` を分けて調べ、source にそれぞれの loop があることを確認する。bias `[10]` の EXPAND ならすべての位置で bias[0]=10 を読み、`[11,12,13]` となる。`[2]` から `[3]` への EXPAND は未対応として拒否する。実行順は example 内で第一 kernel の同期完了後に第二 kernel を呼ぶ明示列でよく、一般の Plan はまだ作らない。入力の長さと中間 state の検証を既存のメモリ経路で維持する。


単独で実行できる例である。EXPAND の出力座標を入力の 0 へ写す過程を、生成した scalar 式の評価まで含めて示す。`src` は演算の入力番号、`Op::Param` の len=1 は singleton の入力配列を表す。

局所 Op、式の評価器と main を含む独立した座標変換の例として保存する。

```rust
// examples/book_08_expand_coordinates.rs（単独で実行する例）
#[derive(Clone,Copy)]
enum Op { Param { slot: usize, len: usize }, Const(f32), Add, Mul, Expand }
struct UOp { op: Op, src: Vec<usize> }
#[derive(Clone,Copy)]
enum Coord { Axis(usize), Zero }
enum Expr { Const(f32), Load(usize,Coord), Add(Box<Expr>,Box<Expr>), Mul(Box<Expr>,Box<Expr>) }
fn value_at(g: &[UOp], id: usize, at: Coord) -> Expr {
	let n = &g[id];
	match n.op {
		Op::Param { slot,.. } => Expr::Load(slot,at),
		Op::Const(x) => Expr::Const(x),
		Op::Add => Expr::Add(Box::new(value_at(g,n.src[0],at)),Box::new(value_at(g,n.src[1],at))),
		Op::Mul => Expr::Mul(Box::new(value_at(g,n.src[0],at)),Box::new(value_at(g,n.src[1],at))),
		Op::Expand => {
			let src = n.src[0];
			match g[src].op {
				Op::Const(_) | Op::Param { len:1,.. } => value_at(g,src,Coord::Zero),
				_ => panic!("EXPAND input must be scalar or singleton"),
			}
		}
	}
}
fn eval(e: &Expr, inputs: &[&[f32]], axes: &[usize]) -> f32 {
	match e {
		Expr::Const(x) => *x,
		Expr::Load(slot,at) => inputs[*slot][match at { Coord::Axis(a)=>axes[*a],Coord::Zero=>0 }],
		Expr::Add(a,b) => eval(a,inputs,axes)+eval(b,inputs,axes),
		Expr::Mul(a,b) => eval(a,inputs,axes)*eval(b,inputs,axes),
	}
}
fn main() {
	let g = vec![
		UOp { op:Op::Param { slot:0,len:3 },src:vec![] },
		UOp { op:Op::Const(2.0),src:vec![] },
		UOp { op:Op::Expand,src:vec![1] },
		UOp { op:Op::Mul,src:vec![0,2] },
		UOp { op:Op::Param { slot:1,len:1 },src:vec![] },
		UOp { op:Op::Expand,src:vec![4] },
		UOp { op:Op::Add,src:vec![3,5] },
	];
	let inputs: [&[f32];2] = [&[1.0,2.0,3.0],&[1.0]];
	// 最初の kernel の MUL を保存し、次の kernel の ADD と比較する。
	let mul = value_at(&g,3,Coord::Axis(0));
	let tmp: Vec<_> = (0..3).map(|i| eval(&mul,&inputs,&[i])).collect();
	let out: Vec<_> = tmp.iter().map(|x| x+inputs[1][0]).collect();
	assert_eq!(tmp,[2.0,4.0,6.0]);
	assert_eq!(out,[3.0,5.0,7.0]);
	let fused = value_at(&g,6,Coord::Axis(1));
	for i in 0..3 { assert_eq!(eval(&fused,&inputs,&[99,i]),out[i]); }
}
```

`Coord::Axis(1)` で下げた式は axes[1] を読み、EXPAND された slot 1 だけは常に 0 を読む。axes[0]=99 は使われないので、別座標を誤共有した場合の検出にもなる。この単独例では shape 検証の成功した入力だけを与える。実装の `verify_expand` は scalar `[]` または `[1]` から正の `[N]` への写像だけを受理する。末尾の fused 式は座標の意味を比較する確認用であり、この章の実行計画では tmp を保存する二 kernel を維持する。

この入力の途中経過を並べる。

```text
output coordinate i=2
MUL: x[2]*CONST 2 -> 3*2=6
EXPAND(bias[1]): requested i=2 -> input coordinate 0 -> bias[0]=1
ADD kernel: tmp[2]+bias[0] -> 7
tmp=[2,4,6], out=[3,5,7]
```

- [ ] scalar と長さ 1 の入力に EXPAND を加える。
- [ ] 入力の singleton 座標を 0 にする。
- [ ] MUL の出力を新規 buffer に保存し、次の ADD が読むようにする。
- [ ] `x*2+1` を `[3,5,7]` にする。
- [ ] `[1]` の bias `[10]` を展開し、`x+b` を `[11,12,13]` にする。

このコミットの実行順は、example 内の単純な kernel 列でよい。
融合と汎用 Plan は次章で加える。

### コミット 3. REDUCE ADD を累積ループへ変換する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── shape_rank1.rs  # [変更] sum と中間 MUL を保存した sum(x*2) の実行。
├── src/
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── linearize.rs  # [確認] 既存 Region による初期化、更新と最終 STORE。
│   ├── rangeify.rs  # [変更] lower_reduce_add と一要素の物理出力。
│   ├── shape.rs  # [変更] reduce_shape による [N] から [] への導出。
│   ├── uop.rs  # [変更] REDUCE ADD と対象軸の op payload。
│   └── verify.rs  # [変更] backend 入力へ抽象 REDUCE を残さない検証。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

REDUCE の型と shape は uop.rs と shape.rs に追加する。lower_reduce_add の抜粋は src/rangeify.rs に置き、07 の REG と Region を使う。ScalarExpr と後の抜粋で Expr と呼ぶ型は同じ scalar 式の型として育てる。shape_rank1.rs は中間 MUL の出力を先に保存し、reduction へその入力 PARAM を渡す。


高水準 REDUCE は「どの軸を集めるか」を表し、backend に渡す命令ではない。候補ファイル `src/shape.rs` の設計案 `reduce_shape` は `[N]` の軸 0 を取り除いて `[]` を返し、`src/rangeify.rs` の `lower_reduce_add` が独立した RANGE k と REG accumulator を生成する。出力は scalar shape でも保存領域は一要素必要なので、ALLOC len=1 と位置 0 への最終 STORE を作る。scalar shape を要素数 0 と取り違えない。

処理は、acc=0 の初期化、k の開始、`value_at(input,k,inner_scope)` の生成、その scalar 値による acc 更新、END k、acc の読出し、out[0] への STORE の順になる。入力 `[1,2,3]` の場合は acc が 0、1、3、6 と変わり、REDUCE の値として 6 を得る。出力には rank 1 の軸がないため、k を出力座標の代用にはしない。k は reduction の内部でだけ有効であり、最終出力を STORE する場所へ自由な値として漏らさない。

`sum(x*2)` では、前コミットと同じ MUL kernel が先に `[2,4,6]` を保存し、別の reduction kernel がそれを読んで 12 にする。入力融合の規則は次章まで加えない。局所テストでは手書き総和 kernel と同じ低水準形になったかを dump で比較し、C に抽象 REDUCE が残らないことを backend verifier で確かめる。値は interpreter と生成 C で 6 と 12 を比較する。初期化、反復、END 後の読出しの region は前章の builder を再利用し、高水準の導入によって状態の寿命を変えない。


組込み用の抜粋である。`src/rangeify.rs` が REDUCE ADD を構造化 region へ変える。`Region::Seq` と `Loop` は 06 の入れ子、`StoreReg` と `StoreOutput` は 07 の STORE、`Expr::Reg(acc)` はその反復時点の REG LOAD を表す。`value_at` は前コミットの座標 lowering を使う。

REDUCE ADD を既存の REG と Region へ変換する関数を追加する。

```rust
// src/rangeify.rs（既存ファイルへ組み込む抜粋）
fn lower_reduce_add(g: &Graph, input: NodeId, axis: AxisId, scope: ScopeId, acc: RegId) -> Result<Region,String> {
	let shape = g.shape(input)?;
	if shape.len() != 1 || shape[0] == 0 { return Err("REDUCE expects positive rank1 input".into()); }
	let n = shape[0];
	if n > i32::MAX as usize { return Err("reduction bound exceeds i32".into()); }
	let term = value_at(g,input,Coordinate::Axis(axis),scope)?;
	let update = Expr::Add(Box::new(Expr::Reg(acc)),Box::new(term));
	Ok(Region::Seq(vec![
		Region::AllocReg(acc),
		Region::StoreReg { reg:acc,value:Expr::Const(0.0) },
		Region::Loop { axis,bound:n,body:vec![Region::StoreReg { reg:acc,value:update }] },
		Region::StoreOutput { index:0,value:Expr::Reg(acc) },
	]))
}
```

ここで `Expr` は前の `ScalarExpr` に REG と定数を追加した同じ scalar 式の型であり、`RegId` はこの kernel 内の一意 ID である。外側 Seq の初期化、Loop の更新、最後の STORE の順を emitter が保つ。出力には loop 軸を作らず、物理的な一要素 allocation へ index=0 で書く。`sum(x*2)` の input は、まず前コミットの MUL kernel が保存した tmp の PARAM に置換して渡すので、この版では input 全体の再帰による融合を行わない。

この入力の途中経過を並べる。

```text
high shape: REDUCE ADD [3] -> []
physical output len: 1
ALLOC REG acc; STORE acc,0
RANGE k=0..3; STORE acc,LOAD(acc)+LOAD(x[k]); END k
STORE out[0],LOAD(acc)
x=[1,2,3] -> 6; materialized tmp=[2,4,6] -> 12
```

- [ ] REDUCE の入力 shape と出力 scalar を検証する。
- [ ] reduction 軸 k と accumulator の初期化を生成する。
- [ ] 前章の逐次累積と同じ低水準 IR へ変換する。
- [ ] `[1,2,3]` の総和を `6` にする。
- [ ] `sum(x*2)` を中間 MUL と別 reduction kernel で `12` にする。

この時点で reduction 入力の融合はしない。
生成コードに抽象的な REDUCE が残っていたら backend の入力検証で拒否する。

### コミット 4. MAX と座標別の lower 結果を検証する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── shape_rank1.rs  # [変更] MAX と異なる座標・束縛環境の比較。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # [確認] 既存の MAX の特殊値 helper。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── interpret.rs  # [変更] 既存 scalar MAX と一致する max_f32 と比較用 reduce_max。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── rangeify.rs  # [変更] ADD と MAX の単位元を選ぶ lower_reduce。
├── tests/
│   └── rangeify.rs  # [追加] shape、EXPAND、reduction と座標混入の統合確認。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

max_f32 と reduce_max は src/interpret.rs に置き、03 ですでに実装した scalar MAX があれば同じ関数を再利用する。src/rangeify.rs の lower_reduce はその契約の単位元と更新 op を選ぶ。抜粋末尾の LowerKey の予定先も rangeify.rs だが、これは将来 memo を入れる場合に必要な情報を示す参考型である。このコミットではキャッシュを追加せず、example と tests/rangeify.rs で座標と kernel ごとの環境を分けて照合する。


ADD 以外の reduction と、同じ元ノードを別座標で下げる動作を同時に確認する。候補ファイル `src/rangeify.rs` の設計案 `lower_reduce(op,input)` では、ADD の単位元 0 と更新 ADD の組を、MAX なら負の無限大と scalar MAX の組へ切り替える。MAX 自体の意味は第三章の evaluator と C helper を使い、NaN が一つでもあれば NaN、両方が零なら +0 を優先する規則を別実装で変えない。

`[-5,-2,-7]` なら acc は最初 -inf、次に -5、-2、-2 となるので結果は -2 である。初期値を 0 にするとすべて負の入力で 0 を返すため、この例で単位元の誤りを検出できる。局所テストには NaN を含む入力と符号付き零の組も置き、NaN は分類、零はビット列で比較する。出力 shape は `[]`、allocation 長は 1 のまま維持する。

座標の fixture は一つの高水準 `t=x*2` を、kernel A の軸 i と kernel B の軸 j からそれぞれ lower する。期待する式は LOAD(xA,i)*2 と LOAD(xB,j)*2 である。元 t の NodeId だけで memo すると A の軸や buffer PARAM が B に混入するため、まず lower のキャッシュを使わず、各 kernel の変換環境を分ける。入力 xA=[1,2,3]、xB=[4,5,6] なら出力は `[2,4,6]` と `[8,10,12]` に分かれる。dump で軸と入力 slot の所属を照合し、元 t の構造が変わらないことも確認する。将来 memo を足すなら座標式、region、読む state をすべて区別する必要がある。


組込み用の抜粋である。`src/interpret.rs` と reduction lowering が共有する MAX の単位元と更新規則を示す。NaN の payload 一致は求めず、NaN 分類の一致を契約にする。

MAX の評価と将来の lowering 用キャッシュキーを分ける。LowerKey はキャッシュ導入時の参考で、今は追加不要である。

```rust
// src/interpret.rs（既存ファイルへ組み込む抜粋）
fn max_f32(a: f32, b: f32) -> f32 {
	if a.is_nan() || b.is_nan() { return f32::NAN; }
	if a == 0.0 && b == 0.0 {
		return if a.is_sign_positive() || b.is_sign_positive() { 0.0 } else { -0.0 };
	}
	if a > b { a } else { b }
}
fn reduce_max(x: &[f32]) -> f32 {
	let mut acc = f32::NEG_INFINITY;
	for &term in x { acc = max_f32(acc,term); }
	acc
}
// src/rangeify.rs（既存ファイルへ組み込む抜粋）
#[derive(Clone,PartialEq,Eq,Hash)]
struct LowerKey {
	node: NodeId,
	coordinate: Coordinate,
	scope: ScopeId,
	state: BufferStateId,
	binding_environment: usize,
}
```

負の値しかない配列でも初期値 -inf から最大を選べる。零だけの比較では、どちらかが +0 なら +0、両方 -0 なら -0 を返す。この規則を 03 の scalar MAX と C helper にも使う。`LowerKey` は将来 memo を導入するときの必要情報を示す型であり、この章の実装はキャッシュなしでもよい。state と binding 環境まで区別することで、別 kernel の PARAM や過去の入力を誤って使う余地を減らす。

この入力の途中経過を並べる。

```text
reduce_max([-5,-2,-7]): -inf -> -5 -> -2 -> -2
max_f32(-0,+0).to_bits() = 0x00000000
max_f32(-0,-0).to_bits() = 0x80000000
value_at(t,i,scopeA,stateA,bindA) != value_at(t,j,scopeB,stateB,bindB)
xA*2=[2,4,6], xB*2=[8,10,12]
```

- [ ] REDUCE MAX の初期値を f32 の負の無限大にする。
- [ ] 第三章の MAX の特殊値契約を interpreter と C で維持する。
- [ ] `max([-5,-2,-7])=-2` を既存 CPU 経路で実行する。
- [ ] 同じノードを二つの出力 RANGE から lower する fixture を作る。
- [ ] 軸の漏れ、違う座標の誤共有、scalar の保存長の誤りを検証する。

二つの出力に異なる入力スロットを渡す例も追加する。
これは後の fusion のために、元グラフを変えずに lowering できることを確かめる。

## 完了確認

以下はこの章で作る example とテストの予定コマンドである。

```sh
# 作業ディレクトリ: chibigrad/
cargo test
cargo run --example shape_rank1 -- --case affine --dump-kernel
cargo run --example shape_rank1 -- --case reduce --dump-source
cargo run --example shape_rank1 -- --case coordinates --dump-graph
```

- [ ] 高水準の `x*2+1` が `[3,5,7]` になる。
- [ ] affine の実行 kernel は二つで、中間 buffer が一つある。
- [ ] 総和が `6`、`sum(x*2)` が `12`、MAX が `-2` になる。
- [ ] scalar の出力 storage は一要素、shape は `[]` になる。
- [ ] backend 入力に EXPAND と高水準 REDUCE が残らない。
- [ ] 各 kernel の RANGE は閉じ、INDEX はその kernel の座標を使う。
- [ ] shape 不一致と不正 EXPAND は実行前にエラーになる。

## tinygrad ソースとの対応

参照 commit は `c22edff14c79310a9b2df62a78819cf77c8289fa`。
リンクの行番号は checkout で確認した。

| 実在するシンボル | ソース | 役割 | chibigrad で簡略化する点 |
|---|---|---|---|
| `IndexingContext.new_range` | [indexing.py](../../tinygrad/tinygrad/schedule/indexing.py#L18) | shape の軸から RANGE を作る | 静的 rank 1 と scalar に限る |
| `broadcast_rngs` | [indexing.py](../../tinygrad/tinygrad/schedule/indexing.py#L60) | singleton の座標を 0 にする | 明示 EXPAND から始める |
| `convert_reduce_to_reduce_with_ranges` | [indexing.py](../../tinygrad/tinygrad/schedule/indexing.py#L104) | shape 軸を reduction RANGE にする | 軸 0 の ADD と MAX に限る |
| `run_rangeify` | [indexing.py](../../tinygrad/tinygrad/schedule/indexing.py#L181) | 座標伝播と入力の indexing | 演算ごとに別 kernel を作る |
| `lower_reduce` | [codegen](../../tinygrad/tinygrad/codegen/__init__.py#L192) | reduction を局所累積へ落とす | 第七章の loop builder を再利用する |
| `bufferize_to_store` | [rangeify.py](../../tinygrad/tinygrad/schedule/rangeify.py#L205) | STAGE を ALLOC と STORE にする | output と中間 buffer を明示確保する |

## 今は追加しないもの

[09-fusion-fission.md](09-fusion-fission.md) で融合、分割、Plan の再実行を加える。
多次元 shape と一般 movement は第 10 段階で追加する。
高水準 Tensor API と matmul は第 11 段階に分ける。
