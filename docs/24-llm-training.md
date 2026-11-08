# 24. 小さな LLM を学習し、checkpoint から同じ更新を再開する

## このステップで動かすもの

小さな causal Transformer を次 token 予測で学習し、中断前後で同じ更新を再現する。
[23](23-flash-attention.md)までの推論と、13の自動微分、14の SGD、21の I/O を使う。
元計画第8章の学習ループと第11章のモデル検証を、LLM 学習まで広げる。
最初の学習は22の通常 attention と f32 を使う。
23の flash kernel は backward 未実装のため、学習時に選択しない。
本文の fixture、test、example、設定値、コマンドは今後作る予定である。
ここでは学習実行、loss の低下、公開 Qwen の更新を確認済みとはしていない。

小 fixture は語彙4、dim8、hidden16、query head2、KV head1、1 layer、context8とする予定である。
token 列 `[1,2,1,2,1,2,1,2,1]` から、先頭8個を入力、後ろ8個を target にする。
初期重みは固定 seed の非ゼロ値とし、短い反復データで loss の低下を確かめる。
実モデルの課題では22で固定した公開 checkpoint の全 layer を対象にする。
モデルの品質や SOTA の throughput は完了条件にしない。

## 最小実装とデータ

学習入力は token、target、有効位置 mask、parameter、optimizer state である。
推論の最後の token だけの logits とは別に、全位置の logits `[B,T,V]` を出す。
loss は有効 token ごとの `logsumexp(logits)-target_logit` の総和を、有効 token 数で割る。
`logsumexp` は14と同じく `m=detach(max(logits))` を引いて安定化し、空の有効集合はエラーとする。
softmax してから小さい確率の log を取る計算にはしない。
入力 IR は純粋な shape 付き学習 graph、出力 IR は既存 pipeline で実行する forward と gradient の graph である。
parameter、gradient、AdamW の m と v を別 buffer に置き、更新前の値を必要な演算が読み終えてから更新する。
元 graph と実体化表を分け、一回目の実行で微分用の式を失わないようにする。

## この章のファイル配置

既存のsrc/autodiff.rsとTensor graphを使い、損失はsrc/nn/loss.rs、SGDとAdamWはsrc/nn/optim.rsへ置く。学習反復はexamples、再開単位の保存と一括採用はsrc/checkpoint.rsへ分ける。本文のautograd.rs表記は13章のautodiff.rsへ読み替える。

次はこの章の終わりまでに触る関連ファイルの配置案である。現在のcheckoutの一覧ではない。各コミットでは、その下に示す部分だけを追加・変更する。

```text
chibigrad/
├── examples/
│   ├── adamw_scalar.rs  # 本体の更新と数式referenceの比較driver。
│   ├── book_24_adamw.rs  # 本文のコードを配置する。
│   ├── book_24_embedding_ce.rs  # 本文のコードを配置する。
│   ├── llm_gradcheck.rs  # 生成Cによるrepeated tokenのdriver。
│   ├── train_qwen.rs  # estimate_training_storageと全Qwen学習driver。
│   └── train_tiny_llm.rs  # step境界の保存、追加stepとしてのresume。
├── src/
│   ├── nn/
│   │   ├── loss.rs  # causal_cross_entropyのTensor graph。
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   ├── optim.rs  # AdamWのm、v、step、平均、clip_global。
│   │   ├── qwen.rs  # 全layerと一意ParamIdの学習登録。
│   │   └── transformer.rs  # 学習でも全位置のlogits_allを使う。
│   ├── autodiff.rs  # 既存微分規則へgatherの寄与を追加する。
│   ├── checkpoint.rs  # TrainingStateとsave、restore、adopt_checkpoint。
│   ├── fetch.rs  # 全ファイルの長さとdigest検証を再利用する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── state.rs  # tensor metadataとpayloadの既存検証を再利用する。
├── tests/
│   ├── llm_gradients.rs  # 全parameter集合と一更新の照合。
│   └── llm_resume.rs  # Qwen設定を含む保存、復元の照合。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

生成Cやshaderは一時ディレクトリへ出力するため、このRustプロジェクトのツリーには含めない。コード冒頭のコメントで、生成元と生成先を確認する。

## 小さなコミットで進める

### 24a. loss と embedding の backward を最小入力で通す

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_24_embedding_ce.rs  # 本文のコードを配置する。
│   └── llm_gradcheck.rs  # [追加] 生成Cによるrepeated tokenのdriver。
├── src/
│   ├── nn/
│   │   ├── loss.rs  # [追加] causal_cross_entropyのTensor graph。
│   │   ├── mod.rs  # [更新] lossモジュールの宣言。
│   │   └── transformer.rs  # [更新] 学習でも全位置のlogits_allを使う。
│   ├── autodiff.rs  # [更新] 既存微分規則へgatherの寄与を追加する。
│   └── lib.rs  # 本体のモジュール宣言と公開APIの入口。
├── tests/
│   └── llm_gradients.rs  # [追加] CE、gather、GQA、tied weightの有限差分。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

13章のsrc/autodiff.rsへgatherのgraph微分規則を加える。本文のsrc/autograd.rsという候補名で重複moduleを作らない。src/nn/mod.rsにlossを宣言し、src/nn/loss.rsへcausal_cross_entropyを追加する。掲載embedding_grad、cross_entropy、mainはexamples/book_24_embedding_ce.rsのf64単独例に置き、llm_gradcheckは本体graphと生成Cを通す。

新しいモジュールは、既存の宣言を残して次を追記する。内部の補助関数は各モジュールに置き、exampleから使う入口だけを公開する。

```rust
// src/nn/mod.rs（モジュール宣言を追加）
pub mod loss;
```


入力はtoken列とembeddingの上流勾配、出力は語彙行ごとに寄与を足した勾配である。次は `src/autodiff.rs` のgather規則を検査する単独で実行できる例。logitsのcross entropyもscalarで確認する。実際の学習はこの式をgraphへ作り、既存の自動微分と生成Cを通す。

embedding_grad、cross_entropy、mainを含むf64の独立例。本体の微分規則は13章のsrc/autodiff.rsへ追加する。

```rust
// examples/book_24_embedding_ce.rs（単独で実行する例）
fn embedding_grad(tokens: &[usize], g: &[[f64;2]], vocab: usize) -> Vec<[f64;2]> {
	assert_eq!(tokens.len(),g.len());
	assert!(tokens.iter().all(|id|*id<vocab));
	let mut dw=vec![[0.;2];vocab];
	for row in 0..vocab {
		for c in 0..2 {
			let mut acc=0.;
			for t in 0..tokens.len() {
				if tokens[t]==row { acc+=g[t][c]; }
			}
			dw[row][c]=acc;
		}
	}
	dw
}
fn cross_entropy(logits: &[f64], target: usize) -> (f64,Vec<f64>) {
	assert!(target<logits.len() && logits.iter().all(|v|v.is_finite()));
	let m=logits.iter().copied().fold(f64::NEG_INFINITY,f64::max);
	let e: Vec<_>=logits.iter().map(|v|(v-m).exp()).collect();
	let z=e.iter().sum::<f64>();
	let mut grad: Vec<_>=e.iter().map(|v|v/z).collect();
	grad[target]-=1.;
	(m+z.ln()-logits[target],grad)
}
fn main() {
	let dw=embedding_grad(&[1,1,2],&[[1.;2];3],4);
	assert_eq!(dw,[[0.,0.],[2.,2.],[1.,1.],[0.,0.]]);
	let (loss,g)=cross_entropy(&[0.;4],2);
	assert_eq!(g,[0.25,0.25,-0.75,0.25]);
	assert!((loss-4f64.ln()).abs()<1e-12);
	println!("loss={loss}, dweight={dw:?}");
}
```

`row,c` は一つの出力担当、`t` は縮約軸である。繰返しID=1が二度現れるので `dw[1]` は2になる。GPUでも出力ごとの担当を一意にしてscatterの競合を避ける。`cross_entropy` のmはgraph版でDETACHし、logsumexpとtarget-logitの差を作る。13のSUB、NEG、RECIPの微分規則もこの分解へ必要になる。

```text
tokens=[1,1,2], sum(embedding) -> g[t,c]=1 -> dW row1=[2,2],row2=[1,1]
logits=[0,0,0,0],target=2 -> loss=ln(4), dlogits=[1/4,1/4,-3/4,1/4]
tied ParamId: dW_total=dW_embedding+dW_output_projection; update once
masked CE: total_loss=sum(valid losses)/Nvalid, reject Nvalid=0
```

22の model に全位置の学習入口 `logits_all` を保ち、`src/nn/loss.rs` の `causal_cross_entropy` と `src/autodiff.rs` の gather 規則を追加する案である。入力は logits[B,T,V]、target[B,T]、有効 mask で、出力は有効 token の平均 loss になる。target の範囲と有効数N>0を検査し、各行で `m=detach(max(logits))`、`lse=m+log(sum(exp(logits-m)))`、`loss=lse-logits[target]` の順に graph を組む。無効位置は分子とNの両方から除く。KV cache や最後の位置だけの projection は使わない。

gather の逆向きは weight の各座標(v,c)を一つの出力担当にする。全 token 位置tを巡り、token[t]==vなら上流勾配 g[t,c]を加え、最後に dweight[v,c]へ一度だけ保存する。この reduction は繰返し ID も自然に加算し、複数 lane が同じ語彙行へ STOREする競合を作らない。式は `sum_t select(token[t]==v,g[t,c],0)` であり、入力 weight の現在値を書き換える命令ではない。tied output projection から届く勾配も同じ ParamId の graph へ足すが、optimizer の更新対象には一回だけ登録する。

sum(embedding([1,1,2]))では上流 g が全成分1なので、行1の途中和は1から2、行2は1、他行は0となる。別 test で1 token、語彙4、logits=0なら exp は全て1、lse=ln(4)、loss=ln(4)となり、logits 勾配は target 成分-3/4、他成分1/4になる。これらの期待値を生成 C で確認してから、norm、RoPE、GQAを含む小 graph の中央差分へ広げる。softmax の安定化 max は勾配経路から外し、勾配が届くのは元 logits の式であることを dump する。flash forward-only が選択されていないことも同じ test の Plan で確認する。

- [ ] 全位置の logits を返す学習入口を追加し、KV cache と sampling を使わない。
- [ ] target の範囲、有効位置、安定した cross entropy を検査する。
- [ ] embedding/gather の backward を追加し、繰り返し token の寄与を加算する。
- [ ] 最初は各語彙行が全 token 位置を走査する reduction で実装し、GPU の書込み競合を避ける。
- [ ] RMSNorm、SiLU、softmax、RoPE、GQA、tied weight の勾配を13の規則と合成して検証する。

予定例 `llm_gradcheck --case repeated-tokens` を CPU の生成 C まで通す。
`sum(embedding([1,1,2]))` の weight 勾配は、行1が各成分2、行2が各成分1、他行が0である。
logits が全て0、語彙4、有効 target がある場合の loss は `ln(4)` になる。
小さい head と語彙で有限差分と照合し、共有 embedding と出力 projection の両方から勾配が届くことを確認する。

### 24b. SGD で一つの batch を反復する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── train_tiny_llm.rs  # [追加] next_token_batchと学習の反復制御。
├── src/
│   ├── nn/
│   │   ├── mod.rs  # [更新] optimモジュールの宣言。
│   │   └── optim.rs  # [追加] 既存SGDを移す更新入口とsgd_candidates。
│   └── lib.rs  # 本体のモジュール宣言と公開APIの入口。
├── tests/
│   └── llm_gradients.rs  # [更新] stepごとの勾配初期化と更新境界の比較。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

next_token_batchをexamples/train_tiny_llm.rs、sgd_candidatesをsrc/nn/optim.rsへ分ける。BTreeMapのuseもoptimizer側へ置く。src/nn/mod.rsにoptimを宣言し、14章でexample内に置いたsgd_step等の再利用部分をここへ移し、既存exampleの呼出しも同じ入口へ向ける。driverは同じsnapshotのforwardとgradientを実行し、全候補と同期の成功後だけslot表を採用する。

新しいモジュールは、既存の宣言を残して次を追記する。内部の補助関数は各モジュールに置き、exampleから使う入口だけを公開する。

```rust
// src/nn/mod.rs（モジュール宣言を追加）
pub mod optim;
```


入力は現在parameterとそのsnapshotについて計算したgradient、出力は新しいparameter候補である。`examples/train_tiny_llm.rs` のデータ整形と `src/nn/optim.rs` の更新境界を、組込み用抜粋で示す。`weights` と `gradients` は同じ一意ParamId集合、各Vecは検査済みの同じshapeである。

BTreeMapのimportと更新候補はoptimizerへ、batch整形は学習exampleへ分けて追加する。

```rust
// src/nn/optim.rs（既存ファイルへ組み込む抜粋）
use std::collections::BTreeMap;
// examples/train_tiny_llm.rs（既存ファイルへ組み込む抜粋）
fn next_token_batch(tokens: &[usize]) -> (&[usize],&[usize]) {
	assert!(tokens.len()>1);
	(&tokens[..tokens.len()-1],&tokens[1..])
}
// src/nn/optim.rs（既存ファイルへ組み込む抜粋）
fn sgd_candidates(
	weights: &BTreeMap<u32,Vec<f32>>, gradients: &BTreeMap<u32,Vec<f32>>, lr: f32,
) -> Result<BTreeMap<u32,Vec<f32>>, &'static str> {
	if weights.keys().ne(gradients.keys()) { return Err("gradient set"); }
	let mut next=BTreeMap::new();
	for (id,w) in weights {
		let g=&gradients[id];
		if w.len()!=g.len() { return Err("gradient shape"); }
		let mut updated=Vec::with_capacity(w.len());
		for (&value,&grad) in w.iter().zip(g) {
			let candidate=value-lr*grad;
			if !candidate.is_finite() { return Err("non-finite update"); }
			updated.push(candidate);
		}
		next.insert(*id,updated);
	}
	Ok(next)
}
```

`next_token_batch` は元列を一つずらす。更新関数は旧weightへ一度も書かない。stepごとの勾配表は空から作り、同じstep内の共有経路だけを加算する。全候補の生成とdevice同期が成功したらPlanのparameter slotを一括rebindする。候補を計算している途中に次stepのforwardを始めない。

```text
tokens=[1,2,1,2,1,2,1,2,1]
input =[1,2,1,2,1,2,1,2]
target=[2,1,2,1,2,1,2,1]
w0=1,g0=2,lr=0.1 -> w1=0.8
new gradient table g1=2 -> w2=0.6; accidental accumulated g1=4 would give0.4
```

`examples/train_tiny_llm.rs` にデータ整形と反復制御を置き、optimizer の更新 graph は既存14の SGD を再利用する案とする。入力列9個から input=token[0..8]、target=token[1..9]を作り、各位置で「今の token から次の token」を予測する。今回の列なら位置0は入力1と target2、位置1は入力2と target1となる。両方を同じ切出しにすると現在 token の復元問題になるため、最初の batch の ID と位置を dump して検査する。

一回の step は現在 parameter の snapshot を Plan に bindし、forward と loss、同じ snapshot についての全 gradient、更新候補の順に実行する。新しい step の勾配表は空から始め、同一 step 内の共有経路だけを加算する。物理 gradient buffer を再利用する時は、前回値へ加えるのか全要素を上書きするのかを固定する。SGD は `w_next=w-lr*g` を新規出力に計算し、必要な旧 weight の読出しが全て終わってから、全更新候補を次回の入力 slot へ一括 rebindする。元の純粋な学習 graph と入力 buffer は変更しない。

更新境界の小 test は scalar の w=1、g=2、lr=0.1なら w_next=0.8になることから始める。同じ g=2の二回目なら0.6であり、前回 gradient を残して g=4にすると0.4になってしまう。この違いを勾配初期化の検査に使い、LLM では CPU scalar reference と一回の loss、全 parameter の最大差を比較する。loss の記録は更新前の forward 値か更新後の評価値かを明示し、100 stepの先頭と最後を同じ定義で比べる。反復による低下は reference で成立する初期値と学習率を得て fixture に固定してから確認する。実行していない低下量は書かない。CPU の graph、schedule、runtimeをそのまま使い、同じ小 batch を利用可能な GPU へ移して更新まで照合する。

- [ ] token を一つずらした target を作り、入力と target の位置を dump する。
- [ ] forward、loss、gradient、SGD 更新を一回の step として既存 runtime で動かす。
- [ ] step の先頭で gradient を初期化し、前回分を意図せず加えない。
- [ ] Plan の入力 slot に更新後の parameter を渡し、次の step で新しい値を読む。
- [ ] CPU の loss と更新を scalar reference に比較し、その後 Metal と OpenCL でも小例を比較する。

予定例 `train_tiny_llm --optimizer sgd --steps 100 --seed 1` を追加する。
100は予定の実行回数であり、既に得られた学習結果ではない。
固定データで最初の loss より最後の loss が下がる設定を reference で確認し、その設定を fixture に固定する。
学習の確認は数値と更新で行い、生成文が自然に見えることだけでは判定しない。

### 24c. AdamW と勾配蓄積を追加する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── adamw_scalar.rs  # [追加] 本体の更新と数式referenceの比較driver。
│   ├── book_24_adamw.rs  # 本文のコードを配置する。
│   └── train_tiny_llm.rs  # [更新] microbatch蓄積とoptimizer選択。
├── src/
│   ├── nn/
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── optim.rs  # [更新] AdamWのm、v、step、平均、clip_global。
│   └── lib.rs  # 本体のモジュール宣言と公開APIの入口。
├── tests/
│   └── llm_gradients.rs  # [更新] full batchとの同値と非有限値での状態保持。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

掲載Adam、adamw、weighted_gradientとmainはexamples/book_24_adamw.rsへ置くf64単独例とする。本体はsrc/nn/optim.rsのSGD入口にAdamW候補生成を追加し、同じ更新境界を使う。clip_globalは同ファイルのhost検証helperへ加え、全一意parameterの平均済みgradientへ一度だけ適用する。実f32更新graphと状態の一括採用は既存runtimeで行う。


入力は平均済みgradientと旧weight/m/v/step、出力はAdamWの次状態である。`src/nn/optim.rs` のscalar更新を単独で実行できる例にする。clipは蓄積した全parameterのgradientを平均した後に一回だけ行い、その結果をこの更新へ渡す。

Adam、adamw、weighted_gradient、mainを一つの独立例として保存する。本体optimizerの型とは分ける。

```rust
// examples/book_24_adamw.rs（単独で実行する例）
#[derive(Clone, Copy)]
struct Adam { w: f64, m: f64, v: f64, step: i32 }
fn adamw(old: Adam, g: f64, lr: f64, decay: f64, eps: f64) -> Option<Adam> {
	let (b1,b2)=(0.9f64,0.999f64);
	if ![old.w,old.m,old.v,g,lr,decay,eps].iter().all(|x|x.is_finite())
		|| old.v<0. || eps<=0. || old.step<0 { return None; }
	let step=old.step.checked_add(1)?;
	let m=b1*old.m+(1.-b1)*g;
	let v=b2*old.v+(1.-b2)*g*g;
	let mhat=m/(1.-b1.powi(step));
	let vhat=v/(1.-b2.powi(step));
	let w=old.w*(1.-lr*decay)-lr*mhat/(vhat.sqrt()+eps);
	if ![w,m,v].iter().all(|x|x.is_finite()) { return None; }
	Some(Adam{w,m,v,step})
}
fn weighted_gradient(means: &[(usize,f64)]) -> Option<f64> {
	let mut count=0usize;
	let mut sum=0.;
	for &(n,g) in means {
		count=count.checked_add(n)?;
		if !g.is_finite() { return None; }
		sum+=n as f64*g;
	}
	if count==0 || !sum.is_finite() { None } else { Some(sum/count as f64) }
}
fn main() {
	let old=Adam{w:1.,m:0.,v:0.,step:0};
	let next=adamw(old,1.,0.1,0.,1e-8).unwrap();
	assert!((next.w-(1.-0.1/(1.+1e-8))).abs()<1e-12);
	assert_eq!(next.step,1); assert_eq!(old.step,0);
	assert_eq!(weighted_gradient(&[(1,2.),(3,6.)]),Some(5.));
	assert!(adamw(old,f64::NAN,0.1,0.,1e-8).is_none());
	println!("next weight={}, weighted gradient=5",next.w);
}
```

`step` は一回だけ増やし、その値をbias correctionの指数に使う。`eps` はsqrtの外側に足す。weight decayは旧wへの独立項であり、gへ加えない。`weighted_gradient` は有効token数の異なるmicrobatchの平均を `n_i/N` で重み付けする。各parameterの平均gradientが揃った後、以下の組込み用抜粋で全体normを求めて同じscaleを掛ける。

全parameterの平均gradientを一度ずつ並べたhost候補へ適用する検証helperへ追加する。

```rust
// src/nn/optim.rs（既存ファイルへ組み込む抜粋）
fn clip_global(gradient: &mut [f64], threshold: f64) -> Result<(), &'static str> {
	if threshold<=0. || !threshold.is_finite() { return Err("clip threshold"); }
	let mut norm=0.0f64;
	for &g in gradient.iter() {
		if !g.is_finite() { return Err("non-finite gradient"); }
		norm=norm.hypot(g);
	}
	if !norm.is_finite() { return Err("norm overflow"); }
	let scale=if norm>threshold { threshold/norm } else { 1. };
	for g in gradient { *g*=scale; }
	Ok(())
}
```

`hypot` は単純な二乗和よりoverflowを避けやすい。ここでのsliceは全parameterのgradientを一度ずつ並べた表である。失敗時は候補を捨て、weight/m/v、step、schedulerの全てを進めない。

```text
microbatch valid counts=[1,3], mean gradients=[2,6]
accumulated sum=2+18=20, valid count=4 -> g=5 (not4)
gradient=[3,4], threshold=2.5 -> norm5, scale0.5 -> [1.5,2]
AdamW step increments once after all microbatches
```

`src/nn/optim.rs` に ParamId ごとの m、vと全体のstepを持つ AdamW の状態を追加する案である。入力は旧 weight、平均済み gradient、旧 m/v、step、学習率と係数、出力は新しい weight/m/vとstepである。まず非有限値を検査し、必要な clipping を施してから `m'=beta1*m+(1-beta1)*g`、`v'=beta2*v+(1-beta2)*g*g` を計算する。s=step+1として `mhat=m'/(1-beta1^s)`、`vhat=v'/(1-beta2^s)` を作り、`w'=w*(1-lr*decay)-lr*mhat/(sqrt(vhat)+eps)` とする。weight decay はgへ足さず、旧wに対する独立項にする。

w=1、g=1、m=v=0、decay=0なら m'=1-beta1、v'=1-beta2で、補正後はいずれも1になる。したがって指定された scalar 例の更新は `1-0.1/(1+eps)` となる。最初からstepを1進めた後さらに補正の指数を増やす誤りや、sqrtの内側へepsを置く誤りはこの式との比較で検出する。m/vとweightは読取り専用入力として扱い、全候補が成功した時だけ次状態へ切り替える。非有限値で中止した場合はstepもschedulerも進めない。

勾配蓄積は各 microbatch の有効 token loss の和を足し、全有効数Nで割る方式にする。平均勾配しか返せない入口なら `n_i/N` で重み付けする。有効数1と3、平均勾配2と6なら全体は `(1*2+3*6)/4=5` であり、単純平均4ではない。この小 test と、同じ全 batch の更新を比較する。clip は蓄積と平均の後に一回だけ行い、`scale=min(1,threshold/norm)` を全 gradient に共通で掛ける。m/vの更新とstep加算も全 microbatch の後に一回だけ行う。dropout=0とデータ順を固定し、既存の生成 kernel と runtime でparameterと状態の両方を照合する。

- [ ] m、v、step、bias correction、weight decay を持つ f32 AdamW を追加する。
- [ ] scalar parameter の一回の更新を数式の reference に一致させる。
- [ ] gradient norm と非有限値を検査し、必要なら設定された閾値で clipping する。
- [ ] microbatch ごとの loss の平均ではなく、有効 token 数に応じて gradient を重み付けする。
- [ ] full batch と勾配蓄積した更新を、同じデータ順と dropout=0 で比較する。

予定例 `adamw_scalar --param 1 --grad 1 --lr 0.1 --weight-decay 0` を追加する。
m=v=0 の初回更新は bias correction 後に `1-0.1/(1+eps)` になる。
勾配蓄積中に parameter と optimizer state を更新せず、全 microbatch の終了時だけ step を進める。
GPU に scatter-add を追加する場合も、24aの競合しない reduction を比較先として残す。

### 24d. optimizer とデータ位置を含めて再開する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── train_tiny_llm.rs  # [更新] step境界の保存、追加stepとしてのresume。
├── src/
│   ├── checkpoint.rs  # [追加] TrainingStateとsave、restore、adopt_checkpoint。
│   ├── fetch.rs  # [更新] 全ファイルの長さとdigest検証を再利用する。
│   ├── lib.rs  # [更新] checkpointモジュールと保存、復元入口の公開。
│   └── state.rs  # [更新] tensor metadataとpayloadの既存検証を再利用する。
├── tests/
│   └── llm_resume.rs  # [追加] 別プロセスの連続実行と再開の照合。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/lib.rsにcheckpointを宣言し、src/checkpoint.rsへTrainingState、adopt_checkpoint、save_training_checkpoint、restore_training_checkpointを追加する。TensorSetは21章のdtype、shape、payload記録を一意ParamIdで束ねる表とし、重複するtensor形式を作らない。外部JSONとファイルadapterもこのモジュールの保存、復元入口へ接続し、src/state.rsのpayload検査とsrc/fetch.rsのdigestを呼ぶ。mainの反復やoptimizer計算を保存adapterへ入れない。

新しいモジュールは、既存の宣言を残して次を追記する。内部の補助関数は各モジュールに置き、exampleから使う入口だけを公開する。

```rust
// src/lib.rs（モジュール宣言を追加）
pub mod checkpoint;
```


入力はoptimizer step境界の全状態、出力は検証済みの復元候補と一括採用である。次は `src/checkpoint.rs` の組込み用抜粋。`TensorSet` は一意ParamIdからdtype/shape/payloadへの表とし、21のbyte検証済みとする。RNGはseedだけでなく実際の内部状態、shuffle順列も保存する。

学習状態全体の候補と一括採用処理へ追加する。TensorSetはstateモジュールの検証済みtensor記録を使う。

```rust
// src/checkpoint.rs（既存ファイルへ組み込む抜粋）
struct TrainingState {
	weights: TensorSet, m: TensorSet, v: TensorSet, step: u64,
	scheduler_step: u64, rng_state: Vec<u8>, permutation: Vec<usize>,
	cursor: usize, epoch: u64, dataset_sha: [u8;32], config_sha: [u8;32],
	tokenizer_sha: [u8;32], accumulated_microbatches: usize,
}
fn adopt_checkpoint(live: &mut TrainingState, candidate: TrainingState)
	-> Result<(), &'static str> {
	if candidate.accumulated_microbatches!=0 { return Err("not a step boundary"); }
	if candidate.dataset_sha!=live.dataset_sha
		|| candidate.config_sha!=live.config_sha
		|| candidate.tokenizer_sha!=live.tokenizer_sha { return Err("data identity"); }
	if candidate.weights.keys().ne(candidate.m.keys())
		|| candidate.weights.keys().ne(candidate.v.keys()) { return Err("optimizer keys"); }
	for (id,w) in &candidate.weights {
		if w.shape!=candidate.m[id].shape || w.shape!=candidate.v[id].shape {
			return Err("optimizer shape");
		}
	}
	if candidate.scheduler_step!=candidate.step { return Err("scheduler step"); }
	if candidate.cursor>candidate.permutation.len() { return Err("data cursor"); }
	*live=candidate;
	Ok(())
}
```

`candidate` は全shard、全digest、RNG形式、順列の一意性と範囲、model登録仕様を検証した後に渡す。`scheduler_step` は本教材の一更新一進行方式の契約である。保存側は外部JSON/ファイルadapterへ次の順で接続する。`write_manifest` は各ファイル名、長さ、SHA-256を含む最後の書込みで、最初に公開しない。

save_training_checkpointとrestore_training_checkpointの外部JSON、ファイル操作の接続順。各疑似名を既存state、digestと選定libraryの実APIへ置き換える。

```text
// src/checkpoint.rs（外部APIの疑似コード）
require accumulated_microbatches==0
wait_all_devices()
snapshot weights,m,v,step,scheduler,RNG,epoch,permutation,cursor,identities
create_unique_temporary_directory(on_same_filesystem_as_destination)
write_each_tensor_and_metadata(); flush_and_sync()
read_back_and_verify_all_lengths_digests_keys_shapes_steps()
write_manifest_last(); flush_and_sync()
rename_to_new_checkpoint_generation()   # old generation remains valid
restore: verify_every_file -> build_candidate -> transfer_and_wait -> adopt_checkpoint
```

```text
continuous A: initial -> 10 updates -> step10
resumed B: initial -> 5 updates -> save(next cursor, RNG current state)
new process -> restore step5 -> 5 additional updates -> step10
compare weights,m,v, losses6..10,next batch; missing m -> old live state retained
```

21の一括 load を、学習状態全体の transaction へ広げる。`src/checkpoint.rs` の `save_training_checkpoint` と `restore_training_checkpoint` を作る案である。保存の入力は optimizer step 完了後の model、m/v、step、scheduler、RNG の現在状態、dataset の識別と次読出し位置になる。蓄積中の gradient や一部更新の状態は初版では保存せず、保存要求時に step 境界かを検査する。全 device の完了を待ってから同じ時点の値を host へ読み、別々の step の weight とmが混ざらないようにする。

確定先と同じ filesystem に一意な一時 directory を作り、tensor payload と metadata を書く。各ファイルの byte 数、digest、dtype、shape、全 key、step の整合性を読み直して検証し、最後に manifest を書いて確定名へ rename する。すでに確定した checkpoint を壊す上書きはせず、別の世代として保存する。restore は全ファイルと dataset、tokenizer、config の識別を検査して仮状態を作り、全件が正しい時だけ model、optimizer、データ読み出し位置を同時に採用する。途中の shard の失敗時は以前の state を保つ。

比較 test は同じ初期状態からAを10更新、Bを5更新して保存する。Bを終了し、新プロセスで復元して追加5更新し、最終step=10の全parameter、m/v、loss列の後半、次batchをAへ照合する。step5の保存には次がstep6であることと、そのstepが読むデータ位置を記録する。乱数はseedだけでなく内部状態と進行位置を保存し、shuffle済み順列を保存する方式ならその順列とcursorも復元する。corrupt manifest、欠落m、異なるdatasetで拒否するtestを置く。連続実行と再開の差が出た時は最初の再開batch、学習率、乱数、gradientの順に比較できるdumpを残し、保存ファイルの存在だけを再開成功の判定にしない。

- [ ] model state、m、v、step、学習率 scheduler、乱数 generator の状態を保存する。
- [ ] dataset の digest、tokenizer/config の識別、epoch、次の読出し位置、shuffle 状態も保存する。
- [ ] 蓄積中間の checkpoint は最初は拒否し、optimizer step 完了後の境界だけで保存する。
- [ ] device の完了待ち後に一時ディレクトリへ書き、全ファイルと manifest の検証後に確定する。
- [ ] 新しいプロセスで検証と load を行い、連続10 step と5 step後の保存・再開を比較する。

予定例 `train_tiny_llm --steps 5 --checkpoint /tmp/chibigrad-llm-ckpt` の後に `--resume` を使う。
再開時の `--steps 5` は追加で5更新する意味と定義し、最終 step が10になることを確認する。
同じ backend、同じ設定、dropout=0 の loss、全 parameter、m、v、次 batch の一致を検証する。
seed の再設定だけで乱数 generator の現在位置を復元したことにはしない。

### 24e. 同じ学習経路を実際の Qwen の全 layer に広げる

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── train_qwen.rs  # [追加] estimate_training_storageと全Qwen学習driver。
├── src/
│   ├── nn/
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── qwen.rs  # [更新] 全layerと一意ParamIdの学習登録。
│   └── lib.rs  # 本体のモジュール宣言と公開APIの入口。
├── tests/
│   ├── llm_gradients.rs  # [更新] 全parameter集合と一更新の照合。
│   └── llm_resume.rs  # [更新] Qwen設定を含む保存、復元の照合。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

estimate_training_storageとStorageEstimateはまずexamples/train_qwen.rsの確保前検査へ置く。driverは22章のQwen登録表、24aの損失、既存autodiff、24cのoptimizer、24dのcheckpointを組み合わせる。Qwen固有の微分器や別parameter表は作らず、実際の一意ParamId集合から各領域を見積もる。


入力は一意ParamIdの要素数表とbatch/sequence設定、出力は各保存領域のbyte見積りである。`estimate_training_storage` の組込み用抜粋ではtied parameterをIDで一度だけ数える。候補の新weight/m/vを同時に保持する区間は16Pの外に加算する。

全Qwen学習driverの確保前見積りへ追加する。実Planのactivation等は別に加算する。

```rust
// examples/train_qwen.rs（既存ファイルへ組み込む抜粋）
use std::collections::BTreeMap;
struct StorageEstimate { persistent: u64, update_outputs: u64, score: u64, logits: u64 }
fn estimate_training_storage(
	registered: &[(u32,u64)], batch: u64, t: u64, heads: u64, vocab: u64,
) -> Option<StorageEstimate> {
	let mut unique=BTreeMap::new();
	for &(id,elements) in registered {
		if let Some(old)=unique.insert(id,elements) {
			if old!=elements { return None; }
		}
	}
	let p=unique.values().try_fold(0u64,|n,v|n.checked_add(*v))?;
	let persistent=p.checked_mul(16)?;
	let update_outputs=p.checked_mul(12)?;
	let score=batch.checked_mul(heads)?.checked_mul(t)?.checked_mul(t)?.checked_mul(4)?;
	let logits=batch.checked_mul(t)?.checked_mul(vocab)?.checked_mul(4)?;
	Some(StorageEstimate{persistent,update_outputs,score,logits})
}
```

`persistent` は旧weight、gradient、m、vの各4P、`update_outputs` は新weight/m/vの各4Pである。実Planがこれらを同時に保持するなら合計28Pにactivation等が加わる。`score` は一層の値であり全24層の寿命を勝手に一つへ縮めない。host転送copy、workspace、activation再計算の有無も実Planから別に加える。

```text
B=1,T=8,Hq=14,V=151936
score one layer =1*14*8*8*4=3584 bytes
all-position logits=1*8*151936*4=4861952 bytes
registered embed ID=3 and tied output ID=3 -> P counts ID3 once
16P is persistent lower bound; no measured peak or successful Qwen training claimed
```

小モデルの更新と再開を済ませた経路へ、22の全 Qwen state を渡す。変更先は `examples/train_qwen.rs` とメモリ見積り関数 `estimate_training_storage` とする案であり、Qwen 専用の微分器は追加しない。manifest の検証と BF16 から f32 への変換後、全 layer と tied weight を同じ ParamId の登録表へ入れる。Pは各一意 ParamId の要素数を一回ずつ足して求め、tied embedding と出力を二重に数えない。全24 layerと語彙151936を保持していることも登録表で検査する。

最初の下限は weight、gradient、m、vが各4P byteで計16P byteである。ただし読取り専用入力と新規更新出力を同時に保持する区間、device転送用hostコピー、activation、workspaceは別に加える。16Pだけを全peakと呼ばない。通常attentionの各scoreは `B*Hq*T*T*4` byte、全位置logitsは `B*T*V*4` byteになる。B=1、T=8、Hq=14なら一つのscoreは3584 byte、V=151936のlogitsは4861952 byteである。これはshapeからの計算値であり、全層を同時保持する量と寿命によるpeakは実Planから別に求める。

gradientの要素数がPと対応するか、更新対象のParamId集合がmodelの全集合と一致するかを検査し、ゼロ勾配も正当な値として保持する。一回のforward、通常attentionによるbackward、更新候補の計算、全stateの切替、checkpoint保存、別プロセスの復元を順に実行する。BF16変換後の同じf32値とデータをreferenceにも渡し、推論の次token一致だけでは学習成功にしない。メモリが足りない場合は確保前の見積りと止まった段階を記録する。block再計算を後に入れる場合も、小モデルで通常版の全gradientと一致してから使う。flashはbackward未実装なので通常版へ戻し、推論用score削減をこの学習の実測値へ混ぜない。

- [ ] 22の manifest を読み、BF16 checkpoint を f32 parameter として準備する。
- [ ] 全 layer と語彙を保ったまま、短い sequence と microbatch=1でメモリ見積りを出す。
- [ ] 実際の一意 parameter 数Pを数え、f32 の重み・勾配・m・vだけで `16*P` byte を要することを記録する。
- [ ] activation、logits、通常 attention の score、再利用 workspace の peak を別に加える。
- [ ] 小さな検証済みデータで一回の forward、backward、全 parameter 更新、保存、再開を順に実行する。

予定例 `train_qwen --manifest fixtures/qwen25/manifest.json --sequence-length 8 --microbatch 1 --steps 1` を追加する。
一部 layer のみの更新、LoRA、語彙の切捨ては全 parameter 学習とは区別する。
実機メモリが足りなければ見積りと未実行箇所を記録し、全 Qwen 学習の実行済みとは書かない。
activation checkpointing を追加する場合は block 単位の再計算から始め、gradient を通常版と照合する。
flash backward を追加していない段階では23の通常版 fallback を保ち、推論用のメモリ削減を学習の数値に流用しない。

## 完了確認

次はこの章で作る予定の example と test のコマンドである。

```sh
# 作業ディレクトリ: chibigrad/
cargo test
cargo test --test llm_gradients
cargo test --test llm_resume
cargo run --example llm_gradcheck -- --case repeated-tokens
cargo run --example adamw_scalar -- --param 1 --grad 1 --lr 0.1 --weight-decay 0
cargo run --example train_tiny_llm -- --optimizer sgd --steps 100 --seed 1
cargo run --example train_tiny_llm -- --steps 5 --checkpoint /tmp/chibigrad-llm-ckpt
cargo run --example train_tiny_llm -- --steps 5 --resume /tmp/chibigrad-llm-ckpt
cargo run --example train_qwen -- --manifest fixtures/qwen25/manifest.json --sequence-length 8 --microbatch 1 --steps 1
```

- [ ] repeated token の期待勾配と uniform logits の `ln(4)` を確認する。
- [ ] 小 graph の有限差分は中央差分、初期幅 `1e-3`、基準 `abs_err <= 3e-3 + 3e-2*abs(ref)` で検査する。
- [ ] 幅を変えて安定性も確認し、非有限値や桁落ちを「勾配一致」として通さない。
- [ ] 一回の parameter 更新を f32 reference と `abs_err <= 1e-5 + 1e-4*abs(ref)` で比較する。
- [ ] full batch と蓄積、連続実行と再開を同じ dropout=0 の条件で比較する。
- [ ] 同一 CPU 経路の再開は byte 一致を目標にし、GPU の加算順が異なる場合は許容値と最大差を記録する。
- [ ] dump で flash forward-only が学習に選ばれず、weight が PARAM のままであることを確認する。
- [ ] 小 LLM の学習成功、全 Qwen の見積り、全 Qwen の実行結果を別々に記録する。

## tinygrad ソースとの対応

参照 commit は `c22edff14c79310a9b2df62a78819cf77c8289fa` である。

| 実在シンボルと相対リンク | tinygrad の役割 | chibigrad で簡略化する点 |
|---|---|---|
| [`compute_gradient`](../../tinygrad/tinygrad/mixin/gradient.py#L114) | 逆順に勾配を作り共有経路の寄与を足す | pure graph と既存規則を使う |
| [`_embedding_fwd`](../../tinygrad/tinygrad/nn/__init__.py#L360) | 比較と reduction で行選択を構成する | backward も競合しない reduction で始める |
| [`OpMixin.sparse_categorical_crossentropy`](../../tinygrad/tinygrad/mixin/op.py#L1760) | logits と log-softmax から loss を作る | smoothing なし、有効 token の平均に固定する |
| [`Optimizer.schedule_step`](../../tinygrad/tinygrad/nn/optim.py#L41) | gradient から更新と実体化対象を作る | fused optimizer を後回しにする |
| [`LAMB._step`](../../tinygrad/tinygrad/nn/optim.py#L161) | AdamW の m、v と bias correction を計算する | trust ratio を使わず AdamW の式に限定する |
| [`GPT.__call__`](../../tinygrad/examples/llm.c/train_gpt2.py#L100) | 学習時は全位置の logits から loss を作る | GPT-2 の構造は移植せず、Qwen の計算を使う |

`tinygrad/llm/model.py` の推論入口は最後の token だけを projection する。
その入口をそのまま teacher forcing の学習に使わない。

## 今は追加しないもの

大規模データの事前学習、分散 optimizer、mixed precision の自動化、SOTA 品質は要求しない。
小 LLM の正しい更新と再開を先に完成させ、全 Qwen の実行をその後の確認単位にする。
学習用 flash backward と新しい optimizer は、同じ数値検証を備えた独立課題として追加する。
