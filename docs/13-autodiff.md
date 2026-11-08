# 13. 純粋なグラフから勾配グラフを作って実行する

## このステップで動かすもの

[12 の数学関数](12-math-functions.md) までの shape 付きグラフを逆向きに微分する。
元計画の第 8 章「自動微分と小さな学習」の前半に対応する。
前向きと勾配を同じ compiler と CPU runtime で実行する。

```text
loss = sum(x*x), x=[1,2,3]
loss = 14, grad(loss,x) = [2,4,6]
u=x*x, loss=sum(u+u), x=[1,2,3]
loss = 28, grad(loss,x) = [4,8,12]
loss=sum((x+b)*(x+b))
x=[[1,2,3],[4,5,6]], b=[10,20,30]
grad(loss,b) = [50,94,138]
```

API とコマンドは追加予定であり、これらの数値は検証に使う期待値である。
実行後にも元の前向きグラフを保持し、微分可能な構造を残す。

## 最小実装とデータ

公開入口は `gradient(loss, targets, seed)` の関数にする。
`GRADIENT` はこの frontend API の役割を表す呼び名とし、backend に残る命令にはしない。
scalar loss の seed は 1、非 scalar の場合は同じ shape の seed を必須にする。
戻り値は target ごとの新しい `NodeId` とする。
利用者の全 Tensor を探索して暗黙に `.grad` を更新する仕組みは初版には要らない。

```text
入力IR: REDUCE_ADD(MUL(x,x), axis=0), target=x
微分後IR: ADD(MUL(BROADCAST(1),x), MUL(BROADCAST(1),x))
kernel IR: RANGE(i,3), LOAD(x,i), MUL(2,x[i]), STORE(dx,i), END
```

元グラフを一度 topological sort し、逆順に勾配を伝える。
`HashMap<NodeId,NodeId>` を勾配の蓄積表とし、同じ入力への寄与は ADD で足す。
勾配表の shape は入力の shape と等しくする。
unbroadcast は増えた先頭軸と、入力側が 1 だった軸を SUM し、元 shape に reshape する。
`[1,3]` と `[3]` を同じ shape として扱わない。

`DETACH` は前向きには恒等、微分では伝播を止める高水準 UOp とする。
微分の後、rangeify より前に前向きの DETACH を除去する。
未対応演算を通る微分要求は、ゼロを返す代わりに演算名と node を含むエラーにする。
target が loss から到達できない場合だけ、同形のゼロ勾配を返す。
微分器は LOAD、STORE、AFTER、実行順のグラフを入力として受け付けない。

## この章のファイル配置

新規 src/autodiff.rs が純粋な Graph を受け、勾配の NodeId を返す。数値 backward や有限差分の小プログラムは examples/book_13_*.rs に保存し、本体のグラフ変換と分けて読む。

次はこの章の終わりまでに触る関連ファイルの配置案である。現在のcheckoutの一覧ではない。各コミットでは、その下に示す部分だけを追加・変更する。

```text
chibigrad/
├── examples/
│   ├── autodiff_finite_diff.rs  # case ごとの誤差を表示する予定統合例。
│   ├── autodiff_math.rs  # libm 基準の数学関数勾配予定例。
│   ├── autodiff_scalar.rs  # 前向きと勾配の二出力を CPU で実行する予定例。
│   ├── autodiff_shapes.rs  # shape 付き勾配の予定例。
│   ├── autodiff_shared.rs  # 共有枝と DETACH を動かす予定例。
│   ├── book_13_central_difference.rs  # 本文のコードを配置する。
│   ├── book_13_math_derivative.rs  # 本文のコードを配置する。
│   └── book_13_shared_detach.rs  # 本文のコードを配置する。
├── src/
│   ├── autodiff.rs  # 高水準 SQRT/SIN/LOG2/EXP2 の微分規則。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rangeify.rs  # 微分後の前向き DETACH 除去。
│   ├── tensor.rs  # 明示 target/seed を渡す薄い入口。
│   ├── uop.rs  # DETACH の定義と builder。
│   └── verify.rs  # DETACH と seed の形の検証。
├── tests/
│   └── autodiff.rs  # Plan を evaluate として使う中央差分比較。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

生成Cやshaderは一時ディレクトリへ出力するため、このRustプロジェクトのツリーには含めない。コード冒頭のコメントで、生成元と生成先を確認する。

## 小さなコミットで進める

### 1. scalar の四則に必要な演算を微分して実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── autodiff_scalar.rs  # [追加] 前向きと勾配の二出力を CPU で実行する予定例。
├── src/
│   ├── autodiff.rs  # [追加] gradient、逆順走査、勾配の蓄積。
│   ├── lib.rs  # [更新] autodiff モジュールと gradient の公開入口。
│   └── tensor.rs  # [更新] 明示 target/seed を渡す薄い入口。
├── tests/
│   └── autodiff.rs  # [追加] 四則の勾配と元グラフの保持。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/lib.rs に pub mod autodiff を宣言し、src/autodiff.rs に gradient と掲載 scalar_grad を追加する。Graph/NodeId/Op/Scalar は src/uop.rs の既存型、HashMap は標準ライブラリから参照する。topological 順は元グラフだけから作り、戻った勾配 NodeId を既存 compile/Plan へ渡す。

新しいモジュールは、既存の宣言を残して次を追記する。内部の補助関数は各モジュールに置き、exampleから使う入口だけを公開する。

```rust
// src/lib.rs（モジュール宣言を追加）
pub mod autodiff;
```


このコミットで追加するのは数値の backward 実行器ではなく、前向きの式から新しい式を作る変換である。候補は `src/autodiff.rs` の設計案 `gradient(graph,loss,targets,seed)` と、Tensor の薄い入口である。入力は rangeify 前の純粋な NodeId、target の列、必要なら seed で、出力は target と同じ順の勾配 NodeId の列とする。最初に loss から到達するノードを子から親への topological 順へ並べ、その順を固定してから逆に走査する。走査中に追加した勾配ノードを同じ走査列へ入れると、今回求めない高階微分まで始まるので入れない。

勾配表 `grads[node]` は、そのノードへ上流から到着した寄与の合計を表す。loss にseed=1を登録し、表に値がある演算だけで局所規則を適用する。ADD は両方へg、SUBは左へgと右へ-g、NEGは-g、MULは左へg*右入力と右へg*左入力を渡す。RECIPは `-g*recip(x)*recip(x)` として入力へ渡す。各 src の「出現」ごとに寄与を作り、同じ NodeId が二回現れても一回にまとめて捨てない。CONSTへは伝えず、PARAMで結果を取り出す。

`x=2,y=x*x+3*x` を追うと、ADDから二つの積へg=1が届く。最初のMULの左右はどちらもxなので、xへ2と2の二寄与を足す。次のMULからxへ3が届き、合計7となる。前向き値10とこの7を通常の compile に二出力として渡し、生成 C を実行する。逆走査ではMULの規則を一度適用し、その規則内で二つのsrcを処理する点を区別する。

`tests/autodiff.rs` にはこの例とSUB、NEG、RECIPの符号を分離した例を置く。微分前後で元ノードのop/srcが変わらず、実体化済みの前向きを再利用しても同じ勾配 graph を作れるかを確認する。初版はscalarで寄与の形を確定し、形状の一般化は次のコミットへ進める。backendへ gradient 命令を送らず、生成した ADD/MUL/RECIP を既存の rewrite、C renderer、Plan に渡せることが、この変更の実行可能な結果である。


`src/autodiff.rs` の逆走査を具体的に書く。`Graph::intern(op: Op, src: &[NodeId]) -> NodeId` は03で Graph へ統合した入口を使う。src の各 NodeId を含む構造キーでノードを共有し、結果を NodeId で返す契約である。`Op` の payload は既存の型付き定義を使う。`topo` は子が先の一意な到達ノード列で、今回追加するノードを含まない。

組込み用の抜粋。

既存 Graph を使って勾配の UOp を作る組込み関数。

```rust
// src/autodiff.rs（既存ファイルへ組み込む抜粋）
fn scalar_grad(g: &mut Graph, topo: &[NodeId], loss: NodeId) -> Result<HashMap<NodeId,NodeId>,String> {
	let one = g.intern(Op::Const(Scalar::F32Bits(1.0_f32.to_bits())), &[]);
	let mut grads = HashMap::from([(loss, one)]);
	for &id in topo.iter().rev() {
		let Some(&up) = grads.get(&id) else { continue };
		let u = g.nodes[id.0 as usize].clone();
		let contribution = match u.op {
			Op::Add => vec![(u.src[0],up),(u.src[1],up)],
			Op::Sub => {
				let neg = g.intern(Op::Neg, &[up]);
				vec![(u.src[0],up),(u.src[1],neg)]
			}
			Op::Neg => vec![(u.src[0],g.intern(Op::Neg,&[up]))],
			Op::Mul => {
				let a = g.intern(Op::Mul,&[up,u.src[1]]);
				let b = g.intern(Op::Mul,&[up,u.src[0]]);
				vec![(u.src[0],a),(u.src[1],b)]
			}
			Op::Recip => {
				let square = g.intern(Op::Mul,&[id,id]);
				let scaled = g.intern(Op::Mul,&[up,square]);
				vec![(u.src[0],g.intern(Op::Neg,&[scaled]))]
			}
			Op::Const(_) | Op::Param { .. } => Vec::new(),
			_ => return Err(format!("unsupported gradient at {id:?}: {:?}",u.op)),
		};
		for (input, value) in contribution {
			let total = match grads.get(&input) {
				Some(&old) => g.intern(Op::Add,&[old,value]),
				None => value,
			};
			grads.insert(input,total);
		}
	}
	Ok(grads)
}
```

`up` は既に集まった勾配の NodeId で、数値ではない。MUL は src の二つの出現それぞれへ寄与を作る。RECIP の id は前向きの 1/x を表すので、id*id に -up を掛ける。各寄与を表へ追加する最後のループで、共有入力への ADD を必ず作る。

同じ入力を追う例。

```text
forward: %0=PARAM(x), %1=MUL(%0,%0), %2=CONST(3), %3=MUL(%2,%0), %4=ADD(%1,%3)
reverse %4: grads[%1]=1,grads[%3]=1
reverse %3: grads[%0]=3
reverse %1: grads[%0]=ADD(ADD(3,x),x)
x=2 -> forward=10, gradient=7
```

単に訪問済み集合で %0 への二つの寄与を消すと x*x の係数2が欠ける。元の %0〜%4 は変更していない。

- [ ] 純粋な graph と target を受け取る gradient API を追加する。
- [ ] scalar seed と逆 topological walk を実装する。
- [ ] ADD の各入力に seed、MUL の各入力に他方との積を渡す。
- [ ] SUB の入力へ `(g,-g)`、NEG の入力へ `-g` を渡す。
- [ ] RECIP の入力へ `-g/(x*x)` を渡し、浮動小数点の除算を MUL と RECIP の合成にそろえる。
- [ ] 勾配の graph を通常の compile と Plan に渡す。

予定例 `autodiff_scalar` は `y=x*x+3*x,x=2` に対して `y=10,dx=7` を返す。
`x*x` の二つの src が同じ NodeId でも、寄与 2 個を足す。
前向きの計算を実体化してから勾配 graph を作っても、同じ結果になる。
`y=(x-3)*(x-3),x=1` は `y=4,dx=-4`、`y=-x,x=2` は `dx=-1`、`y=1/x,x=2` は `dx=-0.25` を確認する。
03で導入した primitive を未対応のまま損失や norm の微分へ流さない。

### 2. 共有ノードと DETACH を実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── autodiff_shared.rs  # [追加] 共有枝と DETACH を動かす予定例。
│   └── book_13_shared_detach.rs  # 本文のコードを配置する。
├── src/
│   ├── autodiff.rs  # [更新] add_grad と共有ノード/DETACH の局所規則。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rangeify.rs  # [更新] 微分後の前向き DETACH 除去。
│   ├── uop.rs  # [更新] DETACH の定義と builder。
│   └── verify.rs  # [更新] DETACH と seed の形の検証。
├── tests/
│   └── autodiff.rs  # [更新] 共有寄与、停止、seed と非到達 target。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/autodiff.rs の寄与蓄積を add_grad にそろえ、DETACH 分岐では src へ寄与を返さない。src/uop.rs と src/verify.rs に同形の DETACH を追加し、src/rangeify.rs への入口で微分後の DETACH を恒等へ下げる。掲載数値 backward は book_13_shared_detach に保存し、本体の UOp 生成を置換しない。


共有を含む graph では、一つのノードへ全消費先からの勾配が集まってから、そのノードを一度だけ逆伝播する必要がある。候補は自動微分モジュールの走査と勾配表、および UOp の DETACH の検証である。設計案の `add_grad(dst,contribution)` は未登録なら寄与を保存し、登録済みなら古い値と新しい値の ADD を作って表を更新する。topological 順を逆に進むことで、共有ノードより先にすべての消費先を処理できる。訪問済み集合を使うのはノードの走査を重複させないためであり、同一入力へ届く寄与を消すためではない。

`u=x*x,y=u+u,x=2` なら、yのADDから同じuへ1が二回届き `grads[u]=1+1=2` となる。その後uのMULを一度処理し、xへ `2*x=4` が二回届いて合計8になる。uを最初の寄与だけで処理してしまうと4に落ち、uを二度処理すると二重加算になる。テストでy=8とdx=8を同時に見ると、この二つの誤りを特定しやすい。

DETACH は入力と同じ値とshapeを返す前向き演算だが、逆伝播の規則は入力へ何も渡さないものとする。`y=detach(x)*x,x=2` のMULからは、通常のxへdetach(x)=2が届く。DETACH自身へ届いたg=2はそこで止まり、結果はdx=2になる。DETACH除去はこの逆走査の後にだけ行い、実行用の graph で恒等写像へ置き換える。前向きの式を簡約する一般rewriteに先に消させない。

seed はlossのshapeと同じものを要求し、scalarだけ省略値1を許す。到達しないtargetはそのshapeのゼロを返す。未対応opのうち勾配が実際に流れるものはop名とNodeIdを含むエラーにし、DETACHの奥へ流れないopへ不必要に微分規則を要求しない。局所検証は共有枝、DETACH、非scalarのseedなしと形状違い、到達しないtargetを分ける。graph dump と通常の生成 C の両方で、元の共有を保ちつつ勾配だけを新たに組み立てたことを確認する。


`DETACH` の停止と共有の集約は同じ逆順走査で扱う。次の単独例は走査順を数値で検証する小さな参照器であり、実装本体は前コミットのように勾配 UOp を返す。

単独で実行できる例。

数値 backward と専用 Op/UOp、main を含む参照例全体。src/uop.rs の型へ上書きしない。

```rust
// examples/book_13_shared_detach.rs（単独で実行する例）
#[derive(Clone, Copy)]
enum Op { Input, Add, Mul, Detach }
struct UOp { op: Op, src: Vec<usize> }
fn backward(nodes: &[UOp], values: &[f32]) -> Vec<f32> {
	let mut grad = vec![0.0; nodes.len()];
	*grad.last_mut().unwrap() = 1.0;
	for id in (0..nodes.len()).rev() {
		let up = grad[id];
		let u = &nodes[id];
		match u.op {
			Op::Input | Op::Detach => {},
			Op::Add => for &s in &u.src { grad[s] += up; },
			Op::Mul => {
				grad[u.src[0]] += up*values[u.src[1]];
				grad[u.src[1]] += up*values[u.src[0]];
			}
		}
	}
	grad
}
fn main() {
	let node = |op,src: &[usize]| UOp { op,src: src.to_vec() };
	let shared = [node(Op::Input,&[]),node(Op::Mul,&[0,0]),node(Op::Add,&[1,1])];
	assert_eq!(backward(&shared,&[2.,4.,8.])[0],8.);
	let stopped = [node(Op::Input,&[]),node(Op::Detach,&[0]),node(Op::Mul,&[1,0])];
	assert_eq!(backward(&stopped,&[2.,2.,4.])[0],2.);
}
```

最終ノードへ1を置き、ノード番号の逆順へ進む。この例では番号が topological 順である。ADD は同じ src=1 へ二回加えるため、共有 MUL を処理する直前の grad[1] は2になる。DETACH は前向き値を持つが入力へ寄与を渡さない。

同じ入力を追う例。

```text
shared: grad[y]=1 -> grad[u]=2 -> grad[x]=4+4=8
stopped: grad[y]=1 -> grad[detach(x)]=2,grad[x]=2
DETACH -> no propagation -> final grad[x]=2
```

停止後の2は解析的な期待値である。detach を前向きで再計算する通常の有限差分では4になり、この演算の契約を検査できない。

- [ ] 複数経路からの勾配を同じ表へ ADD する。
- [ ] 同じ shared node を一度だけ逆向きに処理する。
- [ ] DETACH の前向き値を維持し、入力への寄与を止める。
- [ ] 非 scalar loss の seed が未指定、または shape 不一致なら拒否する。
- [ ] 到達しない target と未対応演算のエラーを区別する。

予定例 `autodiff_shared` は `u=x*x,y=u+u,x=2` で `y=8,dx=8` を返す。
`y=detach(x)*x,x=2` では `y=4,dx=2` を返す。
この値を、前向きで detach を消してから微分した誤った `dx=4` と区別する。

### 3. SUM、broadcast、movement を逆にたどる

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── autodiff_shapes.rs  # [追加] shape 付き勾配の予定例。
├── src/
│   ├── autodiff.rs  # [更新] unbroadcast、inverse_order、SUM/movement の勾配規則。
│   └── lib.rs  # 本体のモジュール宣言と公開APIの入口。
├── tests/
│   └── autodiff.rs  # [更新] bias shape、slice、flip/stack と転置の逆写像。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/autodiff.rs に掲載 unbroadcast と inverse_order を追加し、各 src の寄与を蓄積する直前に unbroadcast を呼ぶ。SUM と movement の match 分岐は既存 Tensor/Graph builder で逆演算を作る。両関数は同じ微分モジュールへ置き、数値コピー用の別 runtime は作らない。


shape を持つ勾配では各ノードの表へ保存する前に、必ず入力と同じshapeへ戻す。変更先は自動微分モジュールの設計案 `unbroadcast(g,input_shape)` と各movementの局所規則である。unbroadcast は出力shapeを右揃えし、入力に存在しなかった先頭軸をSUMする。次に入力側の長さが1で出力側で増えた軸をkeepdimでSUMし、最後に元のinput_shapeへreshapeする。軸を順に削除する場合は番号のずれを避けて降順に処理する。`[1,3]` と `[3]` は同じ値数でも別のshapeとして返す。

`z=x+b` でxが `[2,3]`、bが `[3]`、lossがsum(z*z)なら、bへ渡す前の勾配は `2*z=[[22,44,66],[28,50,72]]` である。bにはない先頭軸0を足すと `[50,94,138]` になり、元の `[3]` と一致する。bが `[1,3]` なら同じ足算でもkeepdimで `[[50,94,138]]` に戻す。この差を確かめると、shapeを値数だけで同一視する誤りを検出できる。SUMの逆は削除された各軸を長さ1で挿入し、入力shapeへbroadcastするので、行和のseed `[2,5]` は元 `[2,3]` へ `[[2,2,2],[5,5,5]]` と戻る。

RESHAPEは入力shapeへのreshape、PERMUTEは逆permutationを使う。SHRINK `[start,end)` の逆は前にstart、後ろにinput_size-endをzero padする。PADの逆はその有効区間をshrinkする。FLIPは同じ軸を再反転し、STACKは先頭軸の各入力位置を長さ1でshrinkして、その軸をreshapeで除く。stack(x,x)なら二つのsliceから来る寄与を同じxへ両方足す。

局所検証はbiasの例、`sum(shrink(x,[1,3)))` の `[0,1,1,0]`、stackの全2を生成Cで確認する。転置後reshapeは逆順にreshapeと逆permutationを適用し、元の論理順へ戻るか小さい固定配列で見る。各局所規則の出力shapeを検証してから勾配表へ入れれば、後段の添字生成に誤ったshapeを渡さずに済む。前向きと同じmovement builderとcompilerを使い、勾配専用のコピー処理は作らない。


`unbroadcast` の SUM 軸を `src/autodiff.rs` で列挙する。`reduce_sum(axes,keepdim)` と `reshape` は08、10の graph builder であり、数値の host 処理ではない。

組込み用の抜粋。

unbroadcast と inverse_order を同じ微分モジュールへ追加する。

```rust
// src/autodiff.rs（既存ファイルへ組み込む抜粋）
fn unbroadcast(g: Tensor, input: &[usize]) -> Result<Tensor,String> {
	let output = g.shape().to_vec();
	if input.len() > output.len() { return Err("gradient rank".into()); }
	let leading = output.len()-input.len();
	let mut axes: Vec<usize> = (0..leading).collect();
	for (j,&size) in input.iter().enumerate() {
		let expanded = output[leading+j];
		if size == 1 {
			if expanded != 1 { axes.push(leading+j); }
		} else if size != expanded {
			return Err("gradient shape".into());
		}
	}
	let reduced = if axes.is_empty() { g } else { g.reduce_sum(&axes,true)? };
	reduced.reshape(input)
}
fn inverse_order(order: &[usize]) -> Vec<usize> {
	let mut inverse = vec![0;order.len()];
	for (out_axis,&in_axis) in order.iter().enumerate() {
		inverse[in_axis] = out_axis;
	}
	inverse
}
```

`leading` 個の軸と、入力側で1だった増加軸を一つの SUM にまとめる。keepdim=true なので軸番号は途中でずれず、最後の reshape だけで `[3]` と `[1,3]` を区別できる。`inverse_order` の入力は10で検証済みの permutation に限る。

同じ入力を追う例。

```text
incoming g=[[22,44,66],[28,50,72]], output shape=[2,3]
input=[3]   -> axes=[0], keepdim=[1,3], reshape=[3]
input=[1,3] -> axes=[0], keepdim=[1,3], reshape=[1,3]
SUM values=[50,94,138]
SHRINK input [4], interval [1,3): backward PAD(before=1,after=1) -> [0,1,1,0]
```

同じ三要素でも shape の契約を維持する。movement の逆規則も既存 PAD/SHRINK/RESHAPE/PERMUTE を生成して通常の compiler へ送る。

- [ ] SUM の勾配を元の軸へ broadcast する。
- [ ] 各 src への寄与を unbroadcast してから蓄積する。
- [ ] RESHAPE を入力 shape、PERMUTE を逆 permutation に戻す。
- [ ] SHRINK の逆を zero PAD、PAD の逆を SHRINK とする。
- [ ] FLIP の逆を同じ FLIP、STACK の逆を各入力への slice とする。

予定例 `autodiff_shapes` は冒頭の `[2,4,6]` と bias の `[50,94,138]` を返す。
`sum(shrink(x,[(1,3)])),x=[1,2,3,4]` は `[0,1,1,0]` を返す。
`sum(stack(x,x))` は x と同形の全要素 2 の勾配になる。
転置後 reshape の逆向きも、実行値を有限差分と比較する。

### 4. 数学関数の導関数を高水準で追加する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── autodiff_math.rs  # [追加] libm 基準の数学関数勾配予定例。
│   └── book_13_math_derivative.rs  # 本文のコードを配置する。
├── src/
│   ├── autodiff.rs  # [更新] 高水準 SQRT/SIN/LOG2/EXP2 の微分規則。
│   └── lib.rs  # 本体のモジュール宣言と公開APIの入口。
├── tests/
│   └── autodiff.rs  # [更新] 非1 seed、数学関数、特異点の区別。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/autodiff.rs の局所規則の match に四つの数学関数を追加する。掲載 local_derivative は book_13_math_derivative へ保存する数値参照であり、本体は同じ係数を既存の UOp で組む。src/codegen/math.rs の近似展開は gradient の後に呼ぶ。


この変更では数学関数を近似式へ展開する前のノードに対して微分規則を定義する。候補は `src/autodiff.rs` のopごとの分岐で、12の高水準 SQRT、SIN、LOG2、EXP2と、既存RECIPを使って新しい graph を作る。設計案の `differentiate_node(node,g)` は局所の寄与列を返し、その後のunbroadcastと加算は共通処理へ任せる。RECIPの規則は最初のscalarコミットで加えた `-g*recip(x)*recip(x)` をそのまま再利用し、別の符号や演算を持つ二重実装を増やさない。

SQRTには `g*recip(2*sqrt(x))`、SINには `g*sin(π/2-x)`、LOG2には `g*recip(x*ln(2))`、EXP2には `g*exp2(x)*ln(2)` を入力への寄与とする。sqrtやexp2の前向きノードは純粋なので、同じ入力とopなら参照を再利用できる。sinの導関数に必要なcosは新しいbackend命令を足さず恒等式で表せる。dtypeはすべてf32とし、ln(2)やπ/2の定数も数学関数と同じ型にする。

x=4のsqrtでは、前向き値2、分母 `2*2=4`、seed1との積からdx=0.25を得る。x=0のsinでは `sin(π/2)` がほぼ1なのでdxもほぼ1となる。x=2のlog2は分母が `2*ln(2)`、x=0のexp2は前向き値1を使ってdxがln(2)になる。複合式で上流gを2にしたときはすべての寄与も二倍になることを確認し、seedを誤って1に固定した規則を見付ける。

数学モードをbounded-polyへ変えるのはgradient graphを作った後のcodegenだけにする。ここで求めるのは理想的な数学関数の導関数であり、丸めや分岐を含む近似実装そのものを逆向きに微分する仕様ではない。`tests/autodiff.rs` ではlibmで上記の通常値と非1のseedを比較し、負値のsqrtやlog2、sqrt(0)の特異点を通常の有限差分から分ける。dumpでは高水準の微分後graphと近似展開後のkernelを分けて保存し、誤差がどちらの段階で増えたか追えるようにする。


数学関数の局所係数をまず数値で確認する。`src/autodiff.rs` は下の各式と同じ UOp を作り、上流 g を必ず掛ける。

単独で実行できる例。

局所導関数の数値確認用の Math と main 全体。

```rust
// examples/book_13_math_derivative.rs（単独で実行する例）
#[derive(Clone, Copy)]
enum Math { Sqrt, Sin, Log2, Exp2, Recip }
fn local_derivative(op: Math, x: f32, upstream: f32) -> f32 {
	match op {
		Math::Sqrt => upstream/(2.0*x.sqrt()),
		Math::Sin => upstream*(std::f32::consts::FRAC_PI_2-x).sin(),
		Math::Log2 => upstream/(x*std::f32::consts::LN_2),
		Math::Exp2 => upstream*x.exp2()*std::f32::consts::LN_2,
		Math::Recip => -upstream*x.recip()*x.recip(),
	}
}
fn main() {
	assert_eq!(local_derivative(Math::Sqrt,4.0,2.0),0.5);
	assert!((local_derivative(Math::Sin,0.0,2.0)-2.0).abs()<1e-6);
	assert!((local_derivative(Math::Log2,2.0,2.0)-1.0/std::f32::consts::LN_2).abs()<1e-6);
	assert!((local_derivative(Math::Exp2,0.0,2.0)-2.0*std::f32::consts::LN_2).abs()<1e-6);
	assert_eq!(local_derivative(Math::Recip,2.0,2.0),-0.5);
}
```

SIN の係数は cos 命令を増やさず sin(π/2-x) で表す。seed=2 のテストによって、g を無視する実装を検出する。これは局所式の検証用であり、compiler 本体へ別の backward 実行器を追加する意味ではない。

同じ入力を追う例。

```text
SQRT(x) backward: MUL(g,RECIP(MUL(CONST(2),SQRT(x))))
LOG2(x) backward: MUL(g,RECIP(MUL(x,CONST(ln(2)))))
EXP2(x) backward: MUL(MUL(g,EXP2(x)),CONST(ln(2)))
x=4,op=SQRT,g=2 -> denominator=4 -> dx=0.5
```

分解前の高水準関数に微分規則を適用し、それから bounded-poly に下げる。近似実装の分岐や bit 操作自体を微分しない。

- [ ] RECIP を f32 の純粋演算として通し、微分を `-g*recip(x)*recip(x)` にする。
- [ ] SQRT の微分を `g/(2*sqrt(x))` として作る。
- [ ] SIN の微分を `g*sin(π/2-x)` として作る。
- [ ] LOG2 の微分を `g/(x*ln(2))`、EXP2 を `g*exp2(x)*ln(2)` にする。
- [ ] 近似展開より前に微分し、数学関数単位の規則を適用する。

予定例 `autodiff_math` は `sqrt(4)` の勾配 `0.25`、`sin(0)` の勾配 `1` を返す。
`log2(2)` の勾配は `1/(2*ln(2))`、`exp2(0)` は `ln(2)` と比較する。
定義域外と sqrt(0) の特異点は、通常値の有限差分検査から分ける。

### 5. 有限差分と既存 pipeline の両方で検証する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── autodiff_finite_diff.rs  # [追加] case ごとの誤差を表示する予定統合例。
│   └── book_13_central_difference.rs  # 本文のコードを配置する。
├── tests/
│   └── autodiff.rs  # [更新] Plan を evaluate として使う中央差分比較。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

掲載 central_difference と main は book_13_central_difference の単独例にする。tests/autodiff.rs へ main を除いた検査関数を移し、evaluate に前向き Plan の bind/run を接続する。examples/autodiff_finite_diff.rs は同じ Plan 経路で case 名と要素ごとの誤差を表示する。


このコミットでは、微分規則とは独立した数値差分をテスト側だけに作り、生成Cで計算した勾配の符号、係数、shapeを検査する。候補は `tests/autodiff.rs` と予定example `autodiff_finite_diff` で、設計案の `check_gradient(plan,inputs,target_slot,h)` が一要素ずつ入力を複製して摂動する。まず基準入力でlossと勾配を実行し、次にtargetの要素jだけを+x方向と-x方向へh動かして、同じ前向きPlanを二回実行する。差分 `d_j=(loss_plus-loss_minus)/(2*h)` と勾配のj要素を比べる。摂動のたびに入力を元へ戻し、他の要素や重みを累積して変更しない。

scalarの `f(x)=x*x+3*x,x=2,h=0.001` なら、理想値で `f(2+h)=10.007001`、`f(2-h)=9.993001` となり、差を0.002で割ると7になる。f32ではこの二つの減算に丸めが入るので、誤差判定を使う。hを半分にした検査も行い、値が大きく動く場合は差分の打消しや特異点を調べる。biasの例ではbの一要素だけを動かすので、二つの行を経由した効果が一つの数値差分へ合計され、unbroadcastの抜けも検出できる。

matmulの小例ではAを固定し、Bの各要素を動かしてscalar lossの差分を取る。movementの入力も同様に一要素ずつ調べれば、座標の逆変換を微分器と別の経路で比較できる。一方DETACHは、入力値を動かして前向きを再計算するとdetach側の値も変わるため、通常の有限差分と一致しない。この演算は解析的な期待値のテストを使うか、detach部分の基準値を固定した別の関数で比較する。有限差分の不一致を理由にdx=2を誤った4へ変更しない。

libmを基準にし、通常点で絶対誤差と相対誤差の合成基準を適用する。出力はopまたはcase名、要素番号、h、解析勾配、差分、誤差を持たせ、最大値だけで失敗位置を隠さない。近似モードは別に比較し、数学関数の理想的導関数との差として記録する。これで同じcompile/runtime経路を使った実行と、独立した微分の検証を同時に確認できる。


`tests/autodiff.rs` の差分検査は、一要素ずつ独立した入力コピーを作る。`evaluate` は通常の前向き Plan を bind/run して scalar loss を返すクロージャとして渡せる。

単独で実行できる例。

数値評価クロージャで有限差分を試す単独例全体。

```rust
// examples/book_13_central_difference.rs（単独で実行する例）
fn central_difference(x: &[f32], h: f32, evaluate: impl Fn(&[f32])->f32) -> Vec<f32> {
	assert!(h.is_finite() && h > 0.0);
	(0..x.len()).map(|j| {
		let mut plus = x.to_vec();
		let mut minus = x.to_vec();
		plus[j] += h;
		minus[j] -= h;
		(evaluate(&plus)-evaluate(&minus))/(2.0*h)
	}).collect()
}
fn main() {
	let x = [1.0,2.0,3.0];
	for h in [1e-3,5e-4] {
		let numeric = central_difference(&x,h,|v|v.iter().map(|z|z*z).sum());
		for (j,(&actual,&input)) in numeric.iter().zip(&x).enumerate() {
			let reference = 2.0*input;
			assert!((actual-reference).abs() <= 2e-3+2e-3*reference.abs(),"j={j},h={h}");
		}
	}
	let frozen = 2.0_f32; // DETACH の基準値を固定
	let d = central_difference(&[2.0],1e-3,|v|frozen*v[0]);
	assert!((d[0]-2.0).abs()<1e-3);
}
```

`plus` と `minus` は各 j の開始時に x から作り直す。前要素への摂動は残らない。最後の例は detach 側を frozen=2 に固定した関数であり、停止した勾配2と比較できる。

同じ入力を追う例。

```text
ordinary f(x)=x*x at x=2 -> numerical derivative≈4
frozen f(x)=2*x at x=2   -> numerical derivative≈2
detach(x)*x backward contract -> 2
```

通常の二乗関数と frozen 関数を混ぜて判定しない。実装後の統合テストでは evaluate を C 実行 Plan に替え、同じ検査器を使う。

- [ ] テスト側に中央差分 `(f(x+h)-f(x-h))/(2*h)` を追加する。
- [ ] `h=1e-3` と `h=5e-4` の両方で極端に不安定な点を除いて比較する。
- [ ] scalar、共有枝、broadcast、movement、matmul の重みを小例で検査する。
- [ ] libm モードを基準にし、近似モードの誤差を微分器の誤りと分ける。

予定例 `autodiff_finite_diff` は項目ごとの最大誤差と判定を表示する。
通常値では `abs_error <= 2e-3 + 2e-3*abs(reference)` を初期目標にする。
近似モードの導関数は近似式自体の厳密な導関数とは限らないため、別の誤差基準を持つ。

## 完了確認

次の example と `tests/autodiff.rs` を追加してから実行する予定である。

```sh
# 作業ディレクトリ: chibigrad/
cargo test
cargo run --example autodiff_scalar -- --dump-graph
cargo run --example autodiff_shared -- --dump-kernel
cargo run --example autodiff_shapes -- --backend cpu
cargo run --example autodiff_math -- --math libm
cargo run --example autodiff_finite_diff -- --step 0.001
```

- [ ] 勾配の出力 shape が target と一致し、共有枝の寄与が欠落しない。
- [ ] backward 用の独立した数値実行器を作らず、生成 C を実行する。
- [ ] graph dump に元の前向きと新しい勾配の両方が残る。
- [ ] DETACH と gradient の要求が backend 入力に残らない。

## tinygrad ソースとの対応

参照 commit は `c22edff14c79310a9b2df62a78819cf77c8289fa` である。

| 実在するシンボル | ソース | 役割 | chibigrad での簡略化 |
|---|---|---|---|
| `compute_gradient` | [gradient.py](../../tinygrad/tinygrad/mixin/gradient.py#L114) | 逆順の微分、shape の足し戻し、共有勾配の加算 | 純粋な静的 f32 だけに限る |
| `_deepwalk` | [gradient.py](../../tinygrad/tinygrad/mixin/gradient.py#L107) | target に至る経路を選び DETACH を除く | 通常の逆 walk と到達性表で始める |
| `pm_gradient` | [gradient.py](../../tinygrad/tinygrad/mixin/gradient.py#L65) | 演算ごとの勾配 graph を作る | match 文で必要な規則だけを書く |
| `OpMixin.gradient` | [op.py](../../tinygrad/tinygrad/mixin/op.py#L473) | 明示 target と seed を受ける API | `.grad` への暗黙の代入を避ける |
| `Tensor.backward` | [tensor.py](../../tinygrad/tinygrad/tensor.py#L447) | Tensor の grad を蓄積する入口 | 初版では参考にとどめる |

## 今は追加しないもの

高階微分、破壊的更新の微分、独自 CALL の微分は必要な段階まで追加しない。
SGD は [14](14-sgd-mnist.md) で実行時の重みを差し替えて行う。
LLM の学習は [24](24-llm-training.md) に分け、ここでは小さい graph の正確さを確定する。
