# 14. 同じ実行計画を再利用して SGD と MNIST 学習を動かす

## このステップで動かすもの

[13 の自動微分](13-autodiff.md) を使い、重みを PARAM スロットで差し替えて学習する。
元計画の第 8 章の SGD と線形回帰、第 11 章の小さな MLP と MNIST に対応する。
CPU で前向き、損失、勾配、重み更新を一巡させる。

```text
線形回帰: x=[1,2], target=[2,4], y=w*x
初期 w=0, loss=mean((y-target)^2)=10
dw=-10, learning_rate=0.1
更新後 w=1, loss=2.5
```

MNIST は最初に小さいローカル fixture で読込みと学習の配線を確認する。
次に利用者が用意した IDX ファイルで、固定 subset の MLP を学習する。
fixture の結果を MNIST 本体の認識精度として報告しない。
ここで示す API、example、test、値、コマンドは実装予定である。

## 最小実装とデータ

学習用 Plan の入力は `images,labels,W1,b1,W2,b2` の固定スロットにする。
重みを CONST にせず、前向きと勾配 graph を一度作る。
初版の SGD 更新はホスト側の `new_weight=weight-lr*grad` でよい。
更新済み重みを次回の入力スロットに渡し、compiler を呼び直さない。
各 iteration の勾配バッファを独立した出力として扱う。
前回の勾配へ意図せず足し続けない。

```text
入力IR: PARAM(images), PARAM(weights) → matmul → bias → loss
微分IR: gradient(loss, weights)
実行出力: scalar loss と weight ごとの gradient buffer
ホスト更新: weight_next = weight - lr * gradient
次の実行: 同じ Plan に weight_next を bind
```

MLP は `[B,784] → [B,16] → ReLU → [B,10]` に固定する。
ReLU 用の比較と WHERE、または MAX をこのステップで明示的に追加する。
ReLU の x=0 の導関数は 0 と定める。
最初は one-hot target の二乗誤差で、学習の一巡を検証する。
続いて stable log-softmax と cross entropy を追加する。
`m=max(logits)`、`logsumexp=m+log(sum(exp(logits-m)))` を各行で計算する。
MAX reduction の一般微分には頼らず、安定化用 m は必ず DETACH する。
13では一般の REDUCE MAX の微分を追加していないため、この条件を損失の合成規則に含める。
ラベルをホストで one-hot f32 に変換すれば gather の追加は要らない。

IDX reader は image magic=2051、label magic=2049 の big-endian header を検証する。
件数、28×28、ラベル 0..9、ファイル長、画像とラベルの対応を調べる。
画素は `f32(pixel)/255` とし、バッチ順は固定にする。
初版は圧縮前のローカル IDX を読み、ネットワークなしでテストできるようにする。
取得ツールを後で作る場合も、固定 URL、checksum、保存先を明記して別コマンドにする。

## この章のファイル配置

学習 loop とホスト SGD は examples/ の実行例から始め、比較/WHERE と損失合成は src/tensor.rs、微分規則は src/autodiff.rs へ加える。IDX の検証だけを新規 src/data/idx.rs に置く。

次はこの章の終わりまでに触る関連ファイルの配置案である。現在のcheckoutの一覧ではない。各コミットでは、その下に示す部分だけを追加・変更する。

```text
chibigrad/
├── examples/
│   ├── book_14_cross_entropy_reference.rs  # 本文のコードを配置する。
│   ├── book_14_idx_reference.rs  # 本文のコードを配置する。
│   ├── book_14_linear_reference.rs  # 本文のコードを配置する。
│   ├── book_14_mlp_reference.rs  # 本文のコードを配置する。
│   ├── mnist_idx.rs  # ローカルパスと subset を受け取る予定例。
│   ├── train_linear.rs  # Plan を一度作り重みを交換する学習 loop と SGD helper。
│   ├── train_mlp_fixture.rs  # Tensor と Plan で二層 fixture を動かす予定例。
│   └── train_mnist.rs  # 固定 subset の一度 compile する学習 loop。
├── src/
│   ├── data/
│   │   ├── idx.rs  # IDX header/payload の検証とファイル読込み。
│   │   └── mod.rs  # idx モジュールの宣言。
│   ├── renderer/
│   │   ├── c.rs  # MAX と既存数学関数の C 出力。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── autodiff.rs  # WHERE 値側の勾配と比較への停止。
│   ├── interpret.rs  # 逐次 MAX の参照評価。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rangeify.rs  # MAX の逐次 lowering。
│   ├── tensor.rs  # REDUCE MAX と stable cross entropy の合成。
│   ├── uop.rs  # MAX reduction の定義。
│   └── verify.rs  # MAX の軸と dtype の検証。
├── tests/
│   ├── fixtures/
│   │   └── mnist.rs  # 合成2件の IDX を作る fixture helper。
│   └── training.rs  # stable loss/gradient と compile 回数。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

生成Cやshaderは一時ディレクトリへ出力するため、このRustプロジェクトのツリーには含めない。コード冒頭のコメントで、生成元と生成先を確認する。

## 小さなコミットで進める

### 1. 線形回帰の一回の更新を実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_14_linear_reference.rs  # 本文のコードを配置する。
│   └── train_linear.rs  # [追加] Plan を一度作り重みを交換する学習 loop と SGD helper。
├── tests/
│   └── training.rs  # [追加] 一回の更新、二回の run と compile 回数。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

examples/train_linear.rs に予定の graph 構築、二出力 compile、run、sgd_step、再bind の順を実装する。掲載 loss_and_grad/sgd/main は book_14_linear_reference の数値参照として別保存し、本番の loss/grad は Plan の出力を使う。SGD helper はこの段階では例のファイル内へ置き、専用 optimizer 階層は増やさない。


ここではgradientの出力を次回入力へ結び、コンパイル済みのPlanと学習中の値を分離する。候補は `examples/train_linear.rs` と小さい学習用helperで、設計案の `sgd_step(weights,grads,lr)` は新しいホスト配列を返す。x、target、wのslot番号を固定し、lossとdwを同じgraphから出力としてcompileする。Planが保持するのはslotのshape/dtype、schedule、実行コードであり、その時点のwの値やアドレスをグラフへ焼き込まない。runごとにbuffers/scalarsの引数表を組み直し、lossとgradientの出力allocationを入力とは別に用意する。

x=`[1,2]`、target=`[2,4]`、w=0なら、予測は `[0,0]`、差は `[-2,-4]`、二乗は `[4,16]` なので、SUMを要素数2で割ってloss=10となる。wへの勾配は `mean(2*(w*x-target)*x)` で、寄与 `[-4,-16]` の平均が-10である。ホスト更新 `w_next=0-0.1*(-10)` は1となる。次のrunでw_nextを同じslotへbindすると予測 `[1,2]`、二乗 `[1,4]`、loss=2.5になる。新しいgradientは-5であり、前回の-10へ加算して-15にしてはいけない。

更新の順番は、全gradientの実行完了、必要な出力の読出し、全new_weightの計算、次のbindingの交換とする。複数重みへ広げるときも、一部の重みだけを先に更新した値で残りのgradientを計算しない。旧入力bufferは今回のrunの完了まで保持し、new_weightのstorage identityを旧weightと同一視しない。学習率0では新しい配列を使っても値が変わらず、同じlossとなる必要がある。

`tests/training.rs` ではloss=10,dw=-10,w_next=1,loss_next=2.5を数値で確認し、compile回数1、run回数2を別に数える。sourceやPlanを作る場所へ更新loopを入れていないことを、二回目の入力アドレス変更でも確認する。前向きと勾配の純粋graphは保持したまま、既存のPlan実行とホスト配列演算だけで一回の学習が完成する。


`examples/train_linear.rs` の更新順を数値で固定する。次の単独例の `loss_and_grad` は13で生成する二出力を照合する参照式であり、実装後には同じ値を Plan から得る。

単独で実行できる例。

数値の loss/gradient と SGD 順序だけを確認する単独例全体。

```rust
// examples/book_14_linear_reference.rs（単独で実行する例）
fn loss_and_grad(x: &[f32], target: &[f32], w: f32) -> (f32,f32) {
	assert_eq!(x.len(),target.len());
	assert!(!x.is_empty());
	let mut loss = 0.0;
	let mut grad = 0.0;
	for (&input,&expected) in x.iter().zip(target) {
		let error = w*input-expected;
		loss += error*error;
		grad += 2.0*error*input;
	}
	(loss/x.len() as f32,grad/x.len() as f32)
}
fn sgd(weights: &[f32], grads: &[f32], lr: f32) -> Vec<f32> {
	assert_eq!(weights.len(),grads.len());
	weights.iter().zip(grads).map(|(&w,&g)|w-lr*g).collect()
}
fn main() {
	let (loss,dw) = loss_and_grad(&[1.,2.],&[2.,4.],0.);
	let next = sgd(&[0.],&[dw],0.1);
	let (next_loss,next_grad) = loss_and_grad(&[1.,2.],&[2.,4.],next[0]);
	assert_eq!((loss,dw,next[0],next_loss,next_grad),(10.,-10.,1.,2.5,-5.));
	assert_eq!(sgd(&next,&[next_grad],0.0),next);
}
```

loss と grad は各呼出しの先頭で0へ戻る。`sgd` は古い weights を変更せず新規配列を返すので、今回の実行が古い入力を使い終わるまで保持できる。

同じ入力を追う例。

```text
compile once: outputs=[loss,dw], inputs=[x,target,w]
run 0: w=0 -> loss=10,dw=-10
host: w_next=0-0.1*(-10)=1
run 1: bind w_next -> loss=2.5,dw=-5
```

数値例は compile の回数を検証してはいない。統合テストではこの期待値と合わせて compile=1、run=2 を数える。

- [ ] パラメータのスロット番号とホスト配列を対応させる。
- [ ] scalar の平均二乗誤差を作り、w の gradient を compile する。
- [ ] SGD のホスト更新と、新しい入力バッファの bind を実装する。
- [ ] compiler 呼出し回数と Plan 実行回数を表示する。

予定例 `train_linear` は冒頭の `loss=10,dw=-10,w_next=1,loss_next=2.5` を返す。
2 回目の実行でも compile 回数は 1 のままとする。
学習率 0 なら w が変わらず、同じ loss が返ることも確認する。

### 2. 小さな MLP の前向きと更新を実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_14_mlp_reference.rs  # 本文のコードを配置する。
│   └── train_mlp_fixture.rs  # [追加] Tensor と Plan で二層 fixture を動かす予定例。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # [更新] 比較/WHERE の C 出力。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── autodiff.rs  # [更新] WHERE 値側の勾配と比較への停止。
│   ├── interpret.rs  # [更新] 比較/WHERE の参照評価。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rangeify.rs  # [更新] 各座標の条件選択。
│   ├── tensor.rs  # [更新] 比較、where、relu の builder。
│   ├── uop.rs  # [更新] 比較と WHERE の Op。
│   └── verify.rs  # [更新] bool 条件と broadcast の検証。
├── tests/
│   └── training.rs  # [更新] 固定二層 MLP の全勾配と更新。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/tensor.rs の impl に比較/where/relu を加え、src/autodiff.rs の WHERE 分岐から値側だけへ条件付き寄与を返す。対応する Op、検証、評価、lowering、C 出力も既存 match に追加する。掲載 main は book_14_mlp_reference へ保存し、tests/training.rs では同じ中間期待値を Plan の結果と比較する。


このコミットは二層MLPの結合を確認するため、MNISTより先に全値を手計算できる固定fixtureを作る。候補は Tensor の比較とWHEREの入口、自動微分のWHERE規則、`tests/training.rs`、予定exampleである。設計案の `where(cond,a,b)` は同じ出力shapeへbroadcastし、前向きでcondが真の位置からa、偽の位置からbを選ぶ。条件はboolで、比較には微分しない。値側への寄与はaへwhere(cond,g,0)、bへwhere(cond,0,g)とし、それぞれunbroadcastしてから蓄積する。ReLUはwhere(x>0,x,0)で作るのでx=0の勾配は0となる。

まずXを2×2の単位行列、Wを単位行列、biasを0、targetをXにすると一層の予測がtargetと一致し、mean二乗誤差が0になる。この配線を確認した後、二層の固定例として `W1=I,b1=0,W2=0,b2=0,target=X` を使える。中間はReLU(X)=X、予測は全0、4要素のmean二乗誤差は0.5である。出力への勾配は `2*(prediction-target)/4` なので対角だけ-0.5、W2の勾配は対角-0.5、b2は列ごとの和で `[-0.5,-0.5]` となる。この段階のW1の勾配はW2=0を通るため0である。

学習率0.1ならW2の対角が0.05、b2が `[0.05,0.05]` になる。同じPlanへ全更新重みをbindすると予測は `[[0.1,0.05],[0.05,0.1]]`、lossは `(0.81+0.0025+0.0025+0.81)/4=0.40625` へ下がる。この固定値は乱数seedなしで、biasのbatch方向へのunbroadcastと新入力の反映を一度に確かめられる。ReLUの負値、0、正値の勾配は別の小例で確認する。

テストでは全パラメータのshape、初回lossとgradient、更新後lossを保存し、少なくともW2が変わることを確認する。WHEREの値選択はpaddingの安全なLOAD分岐を外へ移してよい理由にはならない。条件だけを非微分として、通常のmatmul、broadcast、gradientと同じcompilerを通す。小型fixtureが通ってからだけ784→16→10のshapeへ拡大する。


`tests/training.rs` で二層 fixture の全結合、ReLU、出力勾配、bias の和を直接確かめる。小さいため転記した期待値だけでなく、各中間値も追える。

単独で実行できる例。

二層 fixture の中間値を追う単独 main。tests/training.rs の全体ではない。

```rust
// examples/book_14_mlp_reference.rs（単独で実行する例）
fn main() {
	let x = [[1.0_f32,0.0],[0.0,1.0]];
	let w1 = [[1.0,0.0],[0.0,1.0]];
	let mut hidden = [[0.0;2];2];
	for b in 0..2 { for h in 0..2 {
		let pre = x[b][0]*w1[0][h]+x[b][1]*w1[1][h];
		hidden[b][h] = if pre>0.0 { pre } else { 0.0 };
	}}
	let mut dw2 = [[0.0;2];2];
	let mut db2 = [0.0;2];
	for b in 0..2 { for c in 0..2 {
		let dy = 2.0*(0.0-x[b][c])/4.0;
		db2[c] += dy;
		for h in 0..2 { dw2[h][c] += hidden[b][h]*dy; }
	}}
	let w2 = dw2.map(|row|row.map(|g|-0.1*g));
	let b2 = db2.map(|g|-0.1*g);
	let mut loss = 0.0;
	for b in 0..2 { for c in 0..2 {
		let prediction = hidden[b][0]*w2[0][c]+hidden[b][1]*w2[1][c]+b2[c];
		loss += (prediction-x[b][c]).powi(2)/4.0;
	}}
	assert_eq!(dw2,[[-0.5,0.0],[0.0,-0.5]]);
	assert_eq!(db2,[-0.5,-0.5]);
	assert!((loss-0.40625).abs()<1e-6);
}
```

最初の W2,b2 は0なので dy の式の予測値も0である。`db2` は batch 軸を足し戻し、`dw2` は hidden と dy の積を加える。ReLU の pre=0 は偽の枝に入り、導関数も0とする。

同じ入力を追う例。

```text
initial hidden=[[1,0],[0,1]], prediction=[[0,0],[0,0]], loss=0.5
dW2=[[-0.5,0],[0,-0.5]], db2=[-0.5,-0.5], dW1=0
updated prediction=[[0.1,0.05],[0.05,0.1]], loss=0.40625
```

本体の gradient graph はこれらの参照式と比較する。W1 勾配が0なのは ReLU のためだけではなく、初期 W2=0 を逆方向へ掛けるためである。

- [ ] 比較と WHERE を追加し、WHERE の値側の枝にだけ勾配を伝える。
- [ ] ReLU を WHERE で作り、正の枝だけに勾配を伝える。
- [ ] `[2,2] → [2,2] → [2,2]` の小型 fixture で MLP を組む。
- [ ] 初期重みを明示配列にし、最初の loss と gradient を記録する。
- [ ] bias の unbroadcast がバッチ方向を足し戻すことを確認する。
- [ ] 更新後の loss を同じ Plan で再実行する。

予定例 `train_mlp_fixture` は seed に依存しない固定配列を使う。
入力 `[[1,0],[0,1]]`、target を同じ配列にし、まず全結合一層で loss=0 の場合を作る。
続いて二層の非ゼロ損失を使い、少なくとも一つの重みが変わることを確認する。
損失減少の判定には固定初期値と小さい学習率をテストに保存する。

### 3. IDX fixture と MNIST のローカル読込みを実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_14_idx_reference.rs  # 本文のコードを配置する。
│   └── mnist_idx.rs  # [追加] ローカルパスと subset を受け取る予定例。
├── src/
│   ├── data/
│   │   ├── idx.rs  # [追加] IDX header/payload の検証とファイル読込み。
│   │   └── mod.rs  # [追加] idx モジュールの宣言。
│   └── lib.rs  # [更新] data モジュールの公開入口。
├── tests/
│   ├── fixtures/
│   │   └── mnist.rs  # [追加] 合成2件の IDX を作る fixture helper。
│   └── training.rs  # [更新] reader の正常値と不正データ。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/lib.rs に pub mod data、src/data/mod.rs に pub mod idx を宣言する。掲載 read_idx/main は book_14_idx_reference に保存して検証し、read_idx 関数だけを src/data/idx.rs へ移す。ファイルを読む read_mnist は同じモジュールで byte parser を呼ぶ。合成 fixture helper は tests/fixtures/mnist.rs に置き、tests/training.rs から明示的に読み込む。

新しいモジュールは、既存の宣言を残して次を追記する。内部の補助関数は各モジュールに置き、exampleから使う入口だけを公開する。

```rust
// src/data/mod.rs（モジュール宣言を追加）
pub mod idx;
// src/lib.rs（モジュール宣言を追加）
pub mod data;
```


IDX readerは学習計算と分けた入力境界として追加する。候補は `src/data/idx.rs`、`tests/fixtures` の生成手順、予定example `mnist_idx` である。設計案の `read_mnist(images_path,labels_path,limit)` は画像をrow-majorなf32の `[count,784]`、ラベルを整数列として返す。最初にファイルを開いて必要なheader長を満たすか調べ、各4byteをbig-endianのu32として読む。画像はmagic=2051、件数、rows、colsの16byte、ラベルはmagic=2049と件数の8byteである。rowsとcolsが28で件数が一致することを確認する前に大きな配列を確保しない。

期待する画像ファイル長は `16+count*28*28`、ラベルは `8+count` で、掛算と加算をchecked演算で検査する。短いpayloadと余分なpayloadを区別して診断し、壊れたcountによる巨大allocationを避ける。limitはheaderと全長の整合性を検査した後に適用し、先頭の同じ件数を画像とラベルの両方から選ぶ。ラベルは全て0..9の範囲を検証する。画素byte p は `p as f32 / 255.0` として変換し、画像番号iの画素jは出力 `i*784+j` へ入れる。

fixtureは件数2、shape28×28、1枚目の784byteが0、2枚目が255、ラベルが0,1という内容で生成できる。画像headerの先頭4byteは `[0,0,8,3]`、ラベルは `[0,0,8,1]` なので、little-endianで読む誤りも検出できる。正規化後は前半784要素が0、後半が1、平均はそれぞれ0と1になる。返却したラベルを学習入力へ渡すときにだけ10クラスのone-hot f32へ変換し、reader自体へlossの都合を持ち込まない。

局所検証はこのfixtureの値とshapeに加え、magic違い、header途中、payload一byte不足、件数不一致、28以外の寸法、ラベル10をそれぞれ専用caseにする。ネットワークや実MNISTを必要としない。CLIは指定されたローカルパスとsubset数をログへ出し、ファイルなしなら必要なパスを診断して終了する。このreaderが返した配列を既存のimages/labels slotへbindすれば、学習Planはデータ取得方法を知らずに再利用できる。


`src/data/idx.rs` の入力境界を byte slice だけで検証する。ファイルの読込みはこの関数の外で行い、header と全長を確認してから画素を確保する。

単独で実行できる例。

byte parser と合成入力の main 全体。検証後に parser 関数だけを src/data/idx.rs へ移す。

```rust
// examples/book_14_idx_reference.rs（単独で実行する例）
fn read_idx(images: &[u8], labels: &[u8]) -> Result<(Vec<f32>,Vec<u8>),String> {
	if images.len()<16 || labels.len()<8 { return Err("short header".into()); }
	let be = |v: &[u8]| u32::from_be_bytes(v.try_into().unwrap()) as usize;
	if be(&images[0..4])!=2051 || be(&labels[0..4])!=2049 {
		return Err("bad magic".into());
	}
	let count = be(&images[4..8]);
	if count!=be(&labels[4..8]) { return Err("count mismatch".into()); }
	if be(&images[8..12])!=28 || be(&images[12..16])!=28 {
		return Err("expected 28x28".into());
	}
	let bytes = count.checked_mul(784).and_then(|n|n.checked_add(16))
		.ok_or("image length overflow")?;
	let label_bytes = count.checked_add(8).ok_or("label length overflow")?;
	if images.len()!=bytes || labels.len()!=label_bytes { return Err("payload length".into()); }
	if labels[8..].iter().any(|&l|l>=10) { return Err("label out of range".into()); }
	let pixels = images[16..].iter().map(|&p|p as f32/255.0).collect();
	Ok((pixels,labels[8..].to_vec()))
}
fn main() {
	let mut images = Vec::new();
	for n in [2051_u32,2,28,28] { images.extend(n.to_be_bytes()); }
	images.extend([0;784]); images.extend([255;784]);
	let mut labels = Vec::new();
	for n in [2049_u32,2] { labels.extend(n.to_be_bytes()); }
	labels.extend([0,1]);
	let (pixels,classes) = read_idx(&images,&labels).unwrap();
	assert_eq!(classes,[0,1]);
	assert!(pixels[..784].iter().all(|&p|p==0.0));
	assert!(pixels[784..].iter().all(|&p|p==1.0));
	images.pop(); assert!(read_idx(&images,&labels).is_err());
}
```

`be` が受け取る slice は呼出し側で必ず4byteに切っている。count を使う確保は厳密な全長確認の後なので、短い payload を巨大な Vec に展開しない。label の範囲は limit をかける前の全体で検証する。

同じ入力を追う例。

```text
image prefix: 00 00 08 03 00 00 00 02 00 00 00 1c 00 00 00 1c
image length=16+2*784=1584
label prefix: 00 00 08 01 00 00 00 02
label length=8+2=10
result pixels.shape=[2,784], labels=[0,1]
```

二つの magic の末尾03と01、件数2、28の0x1cを header 内の位置と対応させる。little-endian で読む誤りは magic の段階で拒否できる。

- [ ] `tests/fixtures` に合成した 2 件の 28×28 IDX を作る手順を置く。
- [ ] 最初の画像は全 0、次は全 255、ラベルは `[0,1]` とする。
- [ ] header を検証して、正規化後の全 0 と全 1 を確認する。
- [ ] 不正 magic、短い payload、件数不一致を専用エラーにする。
- [ ] 実データのファイルパスと subset 件数を CLI から受け取る。

予定例 `mnist_idx` は fixture で `count=2,shape=[2,784],labels=[0,1]` を返す。
画素の平均はそれぞれ 0 と 1 になる。
実データ未配置時は必要なファイルパスを表示して終了し、黙って download しない。

### 4. stable cross entropy と固定 subset 学習を実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_14_cross_entropy_reference.rs  # 本文のコードを配置する。
│   └── train_mnist.rs  # [追加] 固定 subset の一度 compile する学習 loop。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # [更新] MAX と既存数学関数の C 出力。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── interpret.rs  # [更新] 逐次 MAX の参照評価。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rangeify.rs  # [更新] MAX の逐次 lowering。
│   ├── tensor.rs  # [更新] REDUCE MAX と stable cross entropy の合成。
│   ├── uop.rs  # [更新] MAX reduction の定義。
│   └── verify.rs  # [更新] MAX の軸と dtype の検証。
├── tests/
│   └── training.rs  # [更新] stable loss/gradient と compile 回数。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/tensor.rs に cross_entropy を追加し、detach(max(...)) と既存数学 builder で loss を作る。REDUCE MAX は src/uop.rs、verify.rs、interpret.rs、rangeify.rs、renderer/c.rs の既存経路へ足す。掲載数値版は book_14_cross_entropy_reference に分ける。examples/train_mnist.rs の Plan は一度 compile し、batch と全更新重みだけを再bind する。


このコミットでは損失を安定したcross entropyへ替え、固定subsetの学習loopを仕上げる。候補は Tensor のMAX reductionと損失の合成helper、`examples/train_mnist.rs`、学習テストである。設計案の `cross_entropy(logits,one_hot)` は logits `[B,C]` の各行で `m=detach(max(logits,axis=1,keepdim=true))` を作る。続いて `shifted=logits-m`、`s=sum(exp(shifted),axis=1,keepdim=true)`、`log_probs=shifted-log(s)`、`loss=-sum(one_hot*log_probs)/B` の順に組む。これは `logsumexp=m+log(s)` と同じ意味だが、大きいmを足してから引く丸めを避けやすい。

MAXは前向きの比較と逐次reductionとしてinterpreter、lowering、Cへ追加する。一般のMAX微分は今回追加せず、安定化用のmを必ずDETACHする。gradientの逆走査はDETACHで止まり、MAXへgを渡さない。logits `[1000,1000]`、one-hot `[1,0]` ならm=1000、shifted=`[0,0]`、exp=`[1,1]`、s=2となり、loss=ln(2)を得る。勾配はsoftmax-targetで `[-0.5,0.5]` となる。MAXの同点をどう微分するかを要求せず、この値まで通常の数学関数と逆伝播で確認する。

学習は先頭128件を16件ずつ固定順に切り、8batchを巡回して20step行う。`[B,784]→[B,16]→ReLU→[B,10]` の入力、重み、biasのslotを固定して一度compileし、各stepでbatchと更新重みをrebindする。one-hotはB×10のf32へhostで作る。全gradientを読み終えてから全重みを更新し、lossとgradientに非有限値があればstep番号を付けて停止する。更新前のloss、一定step後に同じ固定学習subsetを評価したloss、学習に使わないtest subsetの指標を分けて記録する。

`tests/training.rs` は1000の小例のlossと勾配、MAXがDETACHの奥にあるgraph、Planのcompile回数を確認する。実MNISTの20stepはseed、lr、subset、batch順、数学モードを明記して再現する。精度百分率を固定した完成条件は設けず、同じsubsetで有限値を保ちlossが下がることを確認する。失敗時はfixtureで検証したgradient、データ正規化、学習率を順に調べる。


`src/tensor.rs` の損失合成を数値で照合する。次は一行の loss と logits 勾配を計算する参照例で、MAX の値を固定して使う stable log-softmax の式が中心である。

単独で実行できる例。

一行の数値参照式と main 全体。Tensor の損失合成は別に UOp で実装する。

```rust
// examples/book_14_cross_entropy_reference.rs（単独で実行する例）
fn cross_entropy(logits: &[f32], target: &[f32]) -> (f32,Vec<f32>) {
	assert_eq!(logits.len(),target.len());
	assert!(!logits.is_empty());
	let m = logits.iter().copied().fold(f32::NEG_INFINITY,f32::max);
	let shifted: Vec<_> = logits.iter().map(|&x|x-m).collect();
	let exps: Vec<_> = shifted.iter().map(|&x|x.exp()).collect();
	let sum: f32 = exps.iter().sum();
	let log_sum = sum.ln();
	let loss = shifted.iter().zip(target).map(|(&x,&t)|-t*(x-log_sum)).sum();
	let grad = exps.iter().zip(target).map(|(&e,&t)|e/sum-t).collect();
	(loss,grad)
}
fn main() {
	let (loss,grad) = cross_entropy(&[1000.,1000.],&[1.,0.]);
	assert!((loss-std::f32::consts::LN_2).abs()<1e-6);
	assert_eq!(grad,[-0.5,0.5]);
	let (shifted_loss,_) = cross_entropy(&[0.,0.],&[1.,0.]);
	assert_eq!(loss,shifted_loss);
}
```

`shifted` は大きな元値へ戻さずに保持する。`log_sum` との差を取るので1000を足してから再び引く丸めを避ける。target は和1の one-hot を契約とし、batch 版では最後に B で割る。

同じ入力を追う例。

```text
m = DETACH(REDUCE_MAX(logits, axis=1,keepdim=true))
shifted = SUB(logits,m)
s = REDUCE_SUM(EXP2(MUL(shifted,log2(e))),axis=1,keepdim=true)
log_probs = SUB(shifted,MUL(LOG2(s),ln(2)))
loss = MUL(NEG(REDUCE_SUM(MUL(one_hot,log_probs))),RECIP(B))
```

MAX の一般微分は追加せず、graph に DETACH を残して逆走査を止める。勾配を作ってから DETACH を恒等写像へ下げ、通常の backend で前向きと勾配を実行する。

- [ ] MAX reduction、比較、WHERE を interpreter と C renderer に通す。
- [ ] log-softmax と one-hot cross entropy を既存数学関数から合成する。
- [ ] `logits=[1000,1000]` の loss が `ln(2)` になることを確認する。
- [ ] MNIST の先頭 128 件、batch=16、固定初期値で 20 step を実行する。
- [ ] 初期化 seed、学習率、subset、バッチ順、数学モードをログに残す。
- [ ] test subset を学習から除き、評価の損失と正解率を別に表示する。

予定例 `train_mnist` は `784→16→10` の MLP と lr=0.01 から始める。
完成条件は有限な loss と gradient、再現できる更新、固定 subset 上の loss の低下とする。
精度の固定百分率や SOTA 性能は完成条件にしない。
20 step で低下しなければ gradient、データ、学習率を検査し、基準を緩めて通さない。

## 完了確認

以下は将来の example と `tests/training.rs` の予定コマンドである。
`data/mnist` は利用者が配置するローカルデータの予定保存先である。

```sh
# 作業ディレクトリ: chibigrad/
cargo test
cargo run --example train_linear -- --steps 2 --lr 0.1
cargo run --example train_mlp_fixture -- --dump-schedule
cargo run --example mnist_idx -- --fixture
cargo run --example train_mnist -- --data-dir data/mnist --subset 128 --batch 16 --steps 20 --seed 1
```

- [ ] fixture のテストはネットワークと外部 MNIST を必要としない。
- [ ] 同じ環境、同じ seed で loss と重み更新を許容誤差内で再現する。
- [ ] 前向きと backward の graph が実行後にも残る。
- [ ] Plan の再利用中に重みの差替えが結果へ反映される。
- [ ] 非有限な loss が出た step を表示し、成功として扱わない。

## tinygrad ソースとの対応

参照 commit は `c22edff14c79310a9b2df62a78819cf77c8289fa` である。

| 実在するシンボル | ソース | 役割 | chibigrad での簡略化 |
|---|---|---|---|
| `SGD` | [optim.py](../../tinygrad/tinygrad/nn/optim.py#L74) | SGD を LARS の設定として提供する | momentum なしのホスト更新で始める |
| `Optimizer.schedule_step` | [optim.py](../../tinygrad/tinygrad/nn/optim.py#L41) | 勾配から更新を作り実体化対象を返す | 純粋 graph と入力バッファ差替えを維持する |
| `Optimizer._apply_update` | [optim.py](../../tinygrad/tinygrad/nn/optim.py#L60) | detach した重みから update を引く | ホスト側で weight_next を作る |
| `mnist` | [datasets.py](../../tinygrad/tinygrad/nn/datasets.py#L4) | gzip IDX を取得して header 部分を除き shape を作る | ローカル reader と厳密な header 検証を先に作る |
| `OpMixin.gradient` | [op.py](../../tinygrad/tinygrad/mixin/op.py#L473) | 明示 target の勾配を返す | パラメータ一覧だけに勾配を要求する |

## 今は追加しないもの

checkpoint と一般的な重み I/O は [21](21-model-io.md) に分ける。
GPU 学習は [15](15-metal.md) と [16](16-opencl.md) の数値比較後に同じモデルへ適用する。
大規模データ並列、Adam、LLM 学習は [24](24-llm-training.md) で必要性を判断する。
