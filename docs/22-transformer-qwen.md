# 22. 小さな Transformer を通し、実際の Qwen checkpoint で次の token を計算する

## このステップで動かすもの

固定した1層の小モデルから始め、検証済みの公開 checkpoint で greedy 推論を動かす。
[21](21-model-io.md)の state と取得、20の上限付き shape、12の数学関数を使う。
元計画第11章の Transformer 拡張を、実際の Qwen 推論まで具体化する。
前半の MLP と MNIST の完了点は維持する。Qwen を初期コンパイラの前提にはしない。
本文の型、fixture、example、test とコマンドは今後作る予定である。
公開重みのダウンロード、推論、GPU 実機実行はこの文書の作成時には実施していない。

公開モデルは `Qwen/Qwen2.5-0.5B` の dense `Qwen2ForCausalLM` 一種類を最初の対象にする。
revision は `b937cc073b55e980e6f21d5f8b7a77818efcf0eb` を初期候補とする。
この revision には safetensors、config、tokenizer がある。[公式 commit](https://huggingface.co/Qwen/Qwen2.5-0.5B/commit/b937cc073b55e980e6f21d5f8b7a77818efcf0eb)
Qwen3、MoE、hybrid、画像入力を同じ名前の範囲として扱わない。

## 最小実装とデータ

予定の小 fixture は `layers=1, dim=8, hidden=16, heads=2, kv_heads=1, head_dim=4, vocab=16, context=8` とする。
入力は token ID `[1,2,3]`、出力は各位置の logits `[1,3,16]` である。
最初は token 列を直接与え、文字列の tokenization と計算を別々に検証する。
入力 IR は embedding の行選択、RMSNorm、GEMM、RoPE、causal attention、SiLU、residual の組合せである。
出力 IR は既存の LOAD、STORE、数学演算、RANGE、REDUCE に落ちる。
最初の計算は全て f32 とし、checkpoint の BF16 は21の明示変換で読み込む。

公開 config では layer 数24、hidden size896、query head14、KV head2、語彙151936である。
`head_dim=896/14=64` と GQA 比率7を検証し、`tie_word_embeddings=true` を反映する。
RoPE の theta と norm epsilon を config から読む。[公式 config](https://huggingface.co/Qwen/Qwen2.5-0.5B/raw/b937cc073b55e980e6f21d5f8b7a77818efcf0eb/config.json)
Q/K/V projection の bias、half-split RoPE、SiLU gate は対象アーキテクチャに合わせる。
対応する計算は [Qwen2 の公式実装](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/qwen2/modeling_qwen2.py) と照合する。

## この章のファイル配置

Tensor合成のattention部品はsrc/nn/attention.rs、block配線はsrc/nn/transformer.rs、Qwen固有のconfig、重み名、KV状態はsrc/nn/qwen.rsへ置く。f64の単独例とtokenizer接続をmodelの計算本体から分ける。

次はこの章の終わりまでに触る関連ファイルの配置案である。現在のcheckoutの一覧ではない。各コミットでは、その下に示す部分だけを追加・変更する。

```text
chibigrad/
├── examples/
│   ├── attention_reference.rs  # 既存CPU pipelineによるzero-QKの確認。
│   ├── book_22_rope_gqa.rs  # 本文のコードを配置する。
│   ├── qwen_infer.rs  # 同期成功後のcache候補採用とgreedy反復。
│   ├── qwen_inspect.rs  # 全key、shape、共有ParamIdの表示。
│   └── tiny_transformer.rs  # ゼロと非ゼロfixtureのdriver。
├── src/
│   ├── nn/
│   │   ├── attention.rs  # embedding、rms_norm、apply_rope、attention_referenceのgraph版。
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   ├── qwen.rs  # KvState、extend_cache、decode_mask、decode_with_cache。
│   │   └── transformer.rs  # TinyConfig、BlockParams、blockと全位置logits。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── state.rs  # Qwen登録を既存slot照合へ渡す。
├── tests/
│   ├── qwen_tokenizer.rs  # 公式ID fixtureの照合。
│   └── transformer_reference.rs  # cacheなし、chunk、decodeの数値と旧cache保持。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

生成Cやshaderは一時ディレクトリへ出力するため、このRustプロジェクトのツリーには含めない。コード冒頭のコメントで、生成元と生成先を確認する。

## 小さなコミットで進める

### 22a. embedding、norm、attention を小入力で通す

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── attention_reference.rs  # [追加] 既存CPU pipelineによるzero-QKの確認。
│   └── book_22_rope_gqa.rs  # 本文のコードを配置する。
├── src/
│   ├── nn/
│   │   ├── attention.rs  # [追加] embedding、rms_norm、apply_rope、attention_referenceのgraph版。
│   │   └── mod.rs  # [追加] attentionモジュールの宣言。
│   └── lib.rs  # [更新] nnモジュールと必要な公開APIの宣言。
├── tests/
│   └── transformer_reference.rs  # [追加] causal、RoPE、GQAと部品出力の比較。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

候補のうちsrc/nn/attention.rsにgraph版の各部品を置く。nnが初登場ならsrc/lib.rsにnnを、src/nn/mod.rsにattentionを宣言し、既存nnがあればその宣言へ足す。掲載rope、rms_norm、mainはf64の部品確認用としてexamples/book_22_rope_gqa.rsへ一括保存する。attention_reference exampleはそれと分け、既存Tensor APIから生成Cまで実行する。

新しいモジュールは、既存の宣言を残して次を追記する。内部の補助関数は各モジュールに置き、exampleから使う入口だけを公開する。

```rust
// src/lib.rs（モジュール宣言を追加）
pub mod nn;
// src/nn/mod.rs（モジュール宣言を追加）
pub mod attention;
```


入力はheadごとのベクトルと絶対位置、出力はhalf-split方式で回転したベクトルである。`src/nn/attention.rs` の座標referenceを単独で実行できる例にする。Qwenの完成プログラムではなく、RoPEとGQAの対応を検査する部品である。

f64のrope、rms_normとmainを含む独立した座標確認例。Tensor graphの実装はattentionモジュールへ別に組み込む。

```rust
// examples/book_22_rope_gqa.rs（単独で実行する例）
fn rope(x: &[f64], position: usize, theta: f64) -> Vec<f64> {
	assert!(!x.is_empty() && x.len()%2==0 && theta>0.);
	let half=x.len()/2;
	let mut out=vec![0.;x.len()];
	for r in 0..half {
		let angle=position as f64/theta.powf(2.*r as f64/x.len() as f64);
		let (sin,cos)=angle.sin_cos();
		let (a,b)=(x[r],x[r+half]);
		out[r]=a*cos-b*sin;
		out[r+half]=a*sin+b*cos;
	}
	out
}
fn rms_norm(x: &[f64], weight: &[f64], eps: f64) -> Vec<f64> {
	assert_eq!(x.len(),weight.len());
	assert!(!x.is_empty() && eps>0.);
	let mean=x.iter().map(|v|v*v).sum::<f64>()/x.len() as f64;
	x.iter().zip(weight).map(|(v,w)|v/(mean+eps).sqrt()*w).collect()
}
fn main() {
	let x=[1.,2.,3.,4.];
	assert_eq!(rope(&x,0,10000.),x);
	let y=rope(&x,1,10000.);
	let norm=|v:&[f64]|v.iter().map(|a|a*a).sum::<f64>();
	assert!((norm(&x)-norm(&y)).abs()<1e-12);
	let mapping: Vec<_>=(0..4).map(|hq|hq/(4/2)).collect();
	assert_eq!(mapping,[0,0,1,1]);
	let n=rms_norm(&[1.,-1.],&[1.,1.],1.);
	assert!((n[0]-1./2f64.sqrt()).abs()<1e-12);
	println!("rope={y:?}, GQA={mapping:?}");
}
```

`half=2` なので組は `(x[0],x[2])` と `(x[1],x[3])` になる。`position` はchunk内の位置でなく絶対位置 `S+t` である。`mapping` の比率2はquery head4、KV head2のときだけ使い、head数の割切れをhostで検査する。graph版のnormは同じ式を12の数学演算へ分解し、通常attentionのsoftmaxには `detach(max)` を使う。

```text
embedding token=[1,2,3], weight[V,D] -> out[t,c]=weight[token[t],c]
Q=K=0, scale=1/sqrt(D), V=[[2,4],[6,8]]
query_pos=0: valid keys={0}, probs=[1,0], O=[2,4]
query_pos=1: valid keys={0,1}, probs=[0.5,0.5], O=[4,6]
```

`src/tensor.rs` の既存演算を使い、`src/nn/attention.rs` または最初の example 内に `embedding`、`rms_norm`、`apply_rope`、`attention_reference` を置く案とする。入力の token は整数で0以上 vocab 未満かを host で検査する。embedding は token 位置tと成分cから weight[token[t],c]を読む。norm は最後の dim ごとに二乗和を dim で割り、eps を足して逆平方根を掛ける。softmax は行最大を DETACH した m を引き、exp、行和、除算の順に作り、比較用に各中間 Tensor を返せるようにする。

RoPE は head ごとに d/2個の周波数を作る。対象の half-split では成分 r と r+d/2を一組とし、角度を `position/theta^(2*r/d)` とする。d=4の[a,b,c,d]なら最初の組は(a,c)、次は(b,d)であり、隣り合う(a,b)を組にしない。各組を `(a*cos-c*sin, a*sin+c*cos)` の順で元の成分位置へ戻す。位置0なら全角度0で元の値になる。回転対象の d が偶数であること、Q と K で同じ絶対位置を使うことを入口で確認する。

attention は Q と K の内積を sqrt(d) で割り、key_pos<=query_posだけを有効にする。GQA 比率をRとすると query head h は KV head `h/R` を読む。小設定の query2、KV1なら両 head ともKV0を選ぶ。Q=K=0では最初の query の確率は[1,0]で出力[2,4]、次は[1/2,1/2]で[4,6]になる。未来 V を変更して最初の出力が変わらない test と、norm の二乗平均、RoPE の組、head 対応を dump する test を用意する。各部品は新しい専用計算器ではなく graph として作り、通常の rangeify と生成 C の結果まで比較する。

- [ ] token ID の範囲を検査し、embedding の行選択を INDEX と LOAD へ落とす。
- [ ] RMSNorm を `x*rsqrt(mean(x*x)+eps)*weight` で組み立てる。
- [ ] softmax を `m=detach(max(scores))` の減算、exp、sum、除算で組み立て、行和を確認する。
- [ ] RoPE を偶数の head dimension と位置から作り、位置0で入力と一致させる。
- [ ] `QK^T/sqrt(d)`、causal mask、softmax、V の積を既存 Tensor 演算で作る。

予定例 `attention_reference --case zero-qk` を CPU pipeline で実行する。
Q=K=0、V=`[[2,4],[6,8]]` の causal 出力は `[[2,4],[4,6]]` である。
GQA は query head から対応する KV head を添字で選び、まず小さい head 数で確かめる。
時刻tの mask は `key_pos<=query_pos` とし、未来 token を変えても過去の出力が変わらないことを確認する。

### 22b. 1層の Transformer から logits を返す

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── tiny_transformer.rs  # [追加] ゼロと非ゼロfixtureのdriver。
├── src/
│   ├── nn/
│   │   ├── mod.rs  # [更新] transformerモジュールの宣言。
│   │   └── transformer.rs  # [追加] TinyConfig、BlockParams、blockと全位置logits。
│   └── lib.rs  # 本体のモジュール宣言と公開APIの入口。
├── tests/
│   └── transformer_reference.rs  # [更新] 各層境界とlogitsの比較。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/nn/mod.rsにtransformerを宣言し、src/nn/transformer.rsへTinyConfig、BlockParamsと掲載blockを追加する。blockはTransformerBlock::forwardから呼ぶ本文用の関数とし、logits_allでembedding、全block、最終norm、outputを接続する。rmsとropeは22aのgraph版を参照し、単独例のf64関数をTensor処理へ貼り込まない。

新しいモジュールは、既存の宣言を残して次を追記する。内部の補助関数は各モジュールに置き、exampleから使う入口だけを公開する。

```rust
// src/nn/mod.rs（モジュール宣言を追加）
pub mod transformer;
```


入力はx[B,T,dim]とParamId参照、出力はresidual後の同じshapeである。次は `src/nn/transformer.rs` の組込み用抜粋で、既存Tensor graph APIを使う。`linear_oi` は22cの式、`rms` と `rope` は22aのgraph版、`attention` は `softmax(QK^T/sqrt(D)+mask)V` の通常graphを返す。これらは値の即時計算や専用runtimeではない。

既存Tensor graph APIを使うblockの配線へ追加する。

```rust
// src/nn/transformer.rs（既存ファイルへ組み込む抜粋）
fn block(x: Tensor, p: &BlockParams, c: &TinyConfig) -> Tensor {
	let (b,t,d)=(x.shape()[0],x.shape()[1],c.dim);
	let hd=d/c.heads;
	let normalized=rms(x.clone(),p.input_norm.clone(),c.eps);
	let q=linear_oi(normalized.clone(),p.q.clone(),Some(p.q_bias.clone()))
		.reshape([b,t,c.heads,hd]);
	let k=linear_oi(normalized.clone(),p.k.clone(),Some(p.k_bias.clone()))
		.reshape([b,t,c.kv_heads,hd]);
	let v=linear_oi(normalized,p.v.clone(),Some(p.v_bias.clone()))
		.reshape([b,t,c.kv_heads,hd]);
	let q=rope(q,c.positions.clone(),c.theta).permute([0,2,1,3]);
	let k=rope(k,c.positions.clone(),c.theta).permute([0,2,1,3]);
	let v=v.permute([0,2,1,3]);
	let attended=attention(q,k,v,c.positions.clone(),true)
		.permute([0,2,1,3]).reshape([b,t,d]);
	let residual=x+linear_oi(attended,p.o.clone(),None);
	let normalized=rms(residual.clone(),p.post_norm.clone(),c.eps);
	let gate=linear_oi(normalized.clone(),p.gate.clone(),None);
	let up=linear_oi(normalized,p.up.clone(),None);
	let silu=gate.clone()/(1.0+(-gate).exp());
	residual+linear_oi(silu*up,p.down.clone(),None)
}
```

`BlockParams` は登録済みPARAMをTensorとして引く記録、`TinyConfig.positions` は今回の絶対位置Tensorである。`residual` は最初のnorm前のxへattentionを足す。MLPはgateとupの別projectionを作り、`silu*up` をdownでdimへ戻す。最終logitsは全blockの後で `linear_oi(rms(x,final_norm,eps), embedding_weight, None)` とし、語彙行を出力軸に使う。

```text
x [1,3,8] -> Q [1,3,2,4] -> permute [1,2,3,4]
K/V [1,3,1,4] -> GQA attention -> attended [1,2,3,4]
merge -> [1,3,8] -> gate/up [1,3,16] -> down [1,3,8]
final projection weight [16,8] -> logits [1,3,16]
all weights+biases zero -> 48 logits all0 -> argmax=min_id=0
```

22aの部品が単独で動いた後、`src/nn/transformer.rs` に `TinyConfig`、`TransformerBlock::forward`、`Transformer::logits_all` を置く案である。forward は Tensor と state の ParamId 参照を受け、値と必要な比較用の層境界を返す。重みをメソッド内の定数へ取り込まず、model の明示登録から PARAM を引く。config は dim が heads で割り切れること、query head が KV head の整数倍であること、token の長さが context 以下であることを検査する。

入力[1,2,3]は embedding 後[1,3,8]となる。norm 後の projection を `[B,T,H,D]` へ reshape し、Q は[1,3,2,4]、K と V は[1,3,1,4]とする。RoPE を適用し、attention の前に `[B,H,T,D]` へ permute する。attention の結果は逆 permute で `[B,T,H,D]` に戻し、[1,3,8]へ reshape して o projection と最初の residual を計算する。次の norm から gate と up の[1,3,16]を作り、`silu(gate)*up` を down で[1,3,8]へ戻して residual に足す。全 block 後の norm と embedding の転置を使う output projection から[1,3,16]の logits を返す。residual の入力を norm 後の値に取り違えないよう、境界名を記録する。

全重みゼロでは embedding が0、各 norm は eps によって有限、projection と residual も0なので logits の48要素は全て0になる。argmax は各位置で token0を返す。これは配線と shape の検査であり、非ゼロの演算の正しさを保証しないため、続く固定非ゼロ fixture では embedding、attention 後、MLP 後、最終 norm、logits を独立 reference と保存比較する。最初に差の出る境界から調べられるようにする。fixture の乱数と dtype と bias 設定を固定し、予定の example を CPU の既存 pipeline で通してから、同じ state を Metal と OpenCL へ bind する。

- [ ] 小 config と明示登録した state を用意し、embedding から output projection までを通す。
- [ ] pre-norm、attention residual、norm、gated MLP、residual、最終 norm の順を固定する。
- [ ] MLP を `down(silu(gate(x))*up(x))` とし、bias の有無を fixture に固定する。
- [ ] 最初に全重みゼロの fixture を動かし、全 logits が0になることを確認する。
- [ ] 次に固定された非ゼロの重みを使い、独立 reference の各層出力を fixture に保存する。

予定例 `tiny_transformer --fixture fixtures/transformer/tiny.json` は shape `[1,3,16]` を返す。
ゼロ logits に対する argmax の同値処理は最小 token ID を選ぶと定義する。
非ゼロ fixture の logits は reference を実行してから記録し、本文で未測定の値を作らない。
小モデルの段階でも C、Metal、OpenCL の既存 compiler pipeline を使う。

### 22c. 公開 Qwen の config と重み名を対応させる

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── qwen_inspect.rs  # [追加] 全key、shape、共有ParamIdの表示。
├── src/
│   ├── nn/
│   │   ├── mod.rs  # [更新] qwenモジュールの宣言。
│   │   └── qwen.rs  # [追加] QwenConfig::parse、map_qwen_stateとprojection shape照合。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── state.rs  # [更新] Qwen登録を既存slot照合へ渡す。
├── tests/
│   └── transformer_reference.rs  # [更新] OI方向とhead設定の確認。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/nn/mod.rsにqwenを宣言し、src/nn/qwen.rsへQwenConfig::parse、map_qwen_state、qwen_projection_shapesを置く。configのJSON library adapterはQwenConfig::parseの入口へ接続する。projection_oiは同じファイル内の小さい座標referenceとして使い、実modelのlinear_oiはTensorのINDEXとREDUCEを作る。21cのstate照合へ登録表を渡し、別のloaderやparameter型を作らない。

新しいモジュールは、既存の宣言を残して次を追記する。内部の補助関数は各モジュールに置き、exampleから使う入口だけを公開する。

```rust
// src/nn/mod.rs（モジュール宣言を追加）
pub mod qwen;
```


入力はcheckpointのOI行列と入力ベクトル、出力は出力軸ごとの積和である。`src/nn/qwen.rs` のshape検査後に用いるprojection referenceを組込み用抜粋で示す。Tensor版も同じ `w[o*input+i]` のINDEXを作る。

Qwen固有のOI座標referenceとconfigのshape検査へ追加する。projection_oiをmodel実行器として使わない。

```rust
// src/nn/qwen.rs（既存ファイルへ組み込む抜粋）
fn projection_oi(x: &[f32], w: &[f32], bias: Option<&[f32]>, outputs: usize)
	-> Vec<f32> {
	let inputs=x.len();
	assert_eq!(w.len(),outputs*inputs);
	if let Some(b)=bias { assert_eq!(b.len(),outputs); }
	let mut y=vec![0.;outputs];
	for o in 0..outputs {
		let mut acc=bias.map_or(0.,|b|b[o]);
		for i in 0..inputs { acc+=x[i]*w[o*inputs+i]; }
		y[o]=acc;
	}
	y
}
fn qwen_projection_shapes(dim: usize, heads: usize, kv_heads: usize)
	-> Result<[(usize,usize);4], &'static str> {
	if heads==0 || kv_heads==0 || dim%heads!=0 || heads%kv_heads!=0 {
		return Err("head configuration");
	}
	let kv=kv_heads.checked_mul(dim/heads).ok_or("width overflow")?;
	Ok([(dim,dim),(kv,dim),(kv,dim),(dim,dim)])
}
```

`outputs` は保存行列の行数で、transpose済みとして読むことはしない。`qwen_projection_shapes(896,14,2)` のKV幅は128である。config parserの外部JSON adapterは型付きフィールドを返し、dense Qwen2、rope設定、bias設定、tied embeddingの検査を通してからこの関数を呼ぶ。stateの名前照合と同一ParamIdの検査には21cを再利用する。

```text
x=[5,6], w_oi=[[1,2],[3,4]], no bias -> [17,39]
Q/O weight=[896,896], K/V weight=[128,896]
Q bias=[896], K/V bias=[128]
embed_tokens.weight [151936,896] -> shared ParamId -> output projection
```

21の検証済み bytes から `QwenConfig::parse` と `map_qwen_state` を作る案である。置き場所は `src/nn/qwen.rs` とし、汎用の全モデル loader を先に作らない。config parser は dense Qwen2、24 layer、dim896、query14、KV2、vocab151936という今回の対象を照合し、intermediate_size、theta、eps、bias、tied weight など計算に使う値を明示欄へ取り出す。別 architecture の似た key を黙って受理せず、対象外の rope scaling 等があれば、その設定を支えるまで拒否する。

checkpoint の linear weight は[出力幅,入力幅]として扱い、計算は `y[...,o]=sum_i x[...,i]*w[o,i]+bias[o]` にする。Q は[896,896]、K と V は[128,896]、各 bias はそれぞれ[896]と[128]、o projection は[896,896]を期待する。K/V 幅128は KV2と head_dim64の積である。MLP の中間幅は config の値を使い、gate と up が[中間幅,896]、down が[896,中間幅]かを検査する。保存行列を shape だけ合わせて転置せず、Tensor 側の添字式で出力軸と入力軸を明示する。

小さな検査用 weight を `w=[[1,2],[3,4]]`、x=[5,6]にすると出力は[17,39]であり、[23,34]なら保存軸を逆に読んでいる。この検査を projection の実装と共通にする。公開 state は各 layer の番号0..23を列挙して key と shape を全件照合し、embedding[151936,896]を tied output と同じ ParamId へ登録する。独立 lm_head が存在する場合は同じ値かを検査し、欠落していても tied の仕様から解決する。loader の結果は通常の state slot 表であり、重みを graph の CONST にしない。inspect の dump には key、用途、shape、共有 ParamId を出し、実 bytes と reference をまだ計算していない logits 値は記載しない。

- [ ] 21の manifest に固定 revision の全ファイルと SHA-256 を登録する。
- [ ] 対象の model type と config の必要項目を検証し、未対応方式をエラーにする。
- [ ] embedding、全 layer、norm、output の名前と shape を一対一に照合する。
- [ ] Q/K/V の bias を読み、projection 行列の出力軸と入力軸を確認する。
- [ ] tied embedding を一つの ParamId へ結び、重複保存された重みがあれば整合性を確認する。

予定の名前の対応は次とする。実ファイルの key 集合と照合してから使用する。

| checkpoint の名前 | model 内の用途 |
|---|---|
| `model.embed_tokens.weight` | embedding と tied output projection |
| `model.layers.i.self_attn.{q,k,v}_proj.{weight,bias}` | Q、K、V projection |
| `model.layers.i.self_attn.o_proj.weight` | attention output projection |
| `model.layers.i.{input_layernorm,post_attention_layernorm}.weight` | 各 pre-norm |
| `model.layers.i.mlp.{gate,up,down}_proj.weight` | dense gated MLP |
| `model.norm.weight` | 最終 norm |

予定例 `qwen_inspect --manifest fixtures/qwen25/manifest.json` は不足 key と余剰 key を報告する。
重みの shape を切り詰めて小モデルへ入れる処理は、公開 checkpoint 推論の代用にしない。

### 22d. tokenizer と全 layer の prefill を照合する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── qwen_infer.rs  # [追加] tokenizer初期化、prefill、logits読出しとargmax。
├── src/
│   ├── nn/
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── qwen.rs  # [更新] checked_prefill_bytesとモデルの事前容量検査。
│   └── lib.rs  # 本体のモジュール宣言と公開APIの入口。
├── tests/
│   ├── qwen_tokenizer.rs  # [追加] 公式ID fixtureの照合。
│   └── transformer_reference.rs  # [更新] 全layer境界とlogitsの比較。
└── Cargo.toml  # [更新] 固定版tokenizer libraryの依存。
```

tokenizerの疑似接続はexamples/qwen_infer.rsのmainから呼ぶencode_promptへ実装し、libraryは検証済みtokenizer bytesと設定だけを受け取る。掲載smallest_argmaxも同exampleの出力処理へ置く。checked_prefill_bytesはsrc/nn/qwen.rsへ分け、確保前検査から呼ぶ。fetch、state、logits_allは既存公開APIを使い、疑似API名をそのままRustの実関数とみなさない。


入力は検証済みtokenizerと生テキスト、出力は全layerのlogitsを経た最後の有効位置のIDである。tokenizer libraryは未選定なので、接続部を外部APIの疑似コードとして示す。`encode` は固定版tokenizer.jsonと追加token設定を読み、referenceと同じ特殊token方針でID列を返す。モデル計算の中核は22bであり、この接続部へ隠さない。

encode_promptとmainの接続順を示す疑似コード。Tokenizer等は選定libraryと既存fetch、state、Plan APIへ置き換える。

```text
// examples/qwen_infer.rs（外部APIの疑似コード）
verified = fetch_and_verify_every_manifest_file(manifest)
tokenizer = Tokenizer.from_verified_json(verified["tokenizer.json"])
ids = tokenizer.encode(prompt, add_special_tokens=false)
require 0 < len(ids) <= context_capacity
require every id in [0,vocab)
state = validate_and_transfer_all_qwen_state(verified)
plan.bind_buffers(state.slots, token_ids=ids)
plan.bind_scalars(T=len(ids), start_pos=0)
result = plan.run_and_wait()        # 全24blockは22bのgraphを反復
all_logits = result.read("logits") # shape=[1,T,151936]
compare_saved_boundaries(result, independent_reference)
last = all_logits[0,len(ids)-1,:]
next_id = smallest_index_of_max(last)
```

最小ID優先のargmaxは、値が同じ時にindexを更新しなければよい。以下は組込み用のRust抜粋で、有限logitsのみを受け取る。`checked_prefill_bytes` はlogitsの確保前計算であり、stateやactivationの総peakは別表へ足す。

最小IDのargmaxは推論driverへ、容量検査はQwenモデル側へ分けて追加する。

```rust
// examples/qwen_infer.rs（既存ファイルへ組み込む抜粋）
fn smallest_argmax(logits: &[f32]) -> Option<usize> {
	if logits.is_empty() || logits.iter().any(|v|!v.is_finite()) { return None; }
	let mut best=0;
	for i in 1..logits.len() {
		if logits[i]>logits[best] { best=i; }
	}
	Some(best)
}
// src/nn/qwen.rs（既存ファイルへ組み込む抜粋）
fn checked_prefill_bytes(t: usize, vocab: usize, heads: usize)
	-> Option<(usize,usize)> {
	let logits=t.checked_mul(vocab)?.checked_mul(4)?;
	let scores=heads.checked_mul(t)?.checked_mul(t)?.checked_mul(4)?;
	Some((logits,scores))
}
```

`best` は0から始めるのでゼロfixtureは0になる。T=16ではlogitsだけで9,723,904byte、一層の通常attention scoreは14,336byteを要する。実Qwenのtoken IDと次token値は推論後に記録し、このコードから推測しない。

```text
T=16,V=151936,Hq=14 -> logits=16*151936*4=9723904 bytes
score per layer=14*16*16*4=14336 bytes
zero fixture logits=[0,0,0,0] -> next_id=0
real prompt Hello -> IDs/reference/logits: 実行後にfixtureを作成、本文で未測定
```

tokenizer と model 計算を別の比較境界にする。`examples/qwen_infer.rs` は manifest 検証、tokenizer の初期化、ID 列の作成、容量検査、state の準備、prefill、logits の読出しの順で実行する案である。`encode_prompt` は検証済み tokenizer と生テキストから ID 列を返す。特殊 token を自動追加するかも reference と同じ設定に固定し、Hello の具体的な ID を推測して本文へ書かない。空白、日本語、改行、特殊 token の fixture は公式 tokenizer の出力を実際に得てから保存する。

同じ ID を22bと同じ全位置 logits の graph へ渡し、全24 block を順に通す。公開設定では token 数Tに対して embedding[1,T,896]、query[1,14,T,64]、key と value[1,2,T,64]、最終 logits[1,T,151936]となる。GQA の head0..6は KV0、head7..13は KV1を読む。推論の次 token は最後の有効位置 T-1の logits で選ぶが、比較では途中層と全位置の値を保持できる小さな T を使う。tied output の保存軸は語彙行なので、最後の内積も checkpoint の向きに合わせる。

ここで固定できる具体値はゼロ fixture の logits=0と、その時の最小 ID の argmax=0であり、実 Qwen の次 token は実行してから記録する。reference も同じ BF16 bytes を f32 へ変換し、同じ eps、theta、token ID、推論 mode、eager attention で動かす。比較表は embedding、各 block、norm、logits の順に最大誤差を出し、argmax だけ一致しても完了にしない。確保前には全 state の f32 byte 数、通常 attention の score、activation と logits を列挙する。容量16を超える入力なら kernel 前に拒否する。既存の compiler と runtime で短い prefill が完了した時だけその backend を確認済みとし、tokenizer fixture の成功だけで全モデルの成功とは扱わない。

- [ ] 固定版の tokenizer library を使い、21で検証した tokenizer.json と設定を読み込む。
- [ ] 空白、日本語、改行、特殊 token の ID 列を公式 tokenizer の固定 fixture と一致させる。
- [ ] chat template は最初は使わず、同じ生テキストと token ID を reference に渡す。
- [ ] 全24 layer を f32 で実行し、embedding、層境界、最終 logits を reference と比較する。
- [ ] 容量を超える入力は起動前に拒否し、まず context=16など短い容量で一つの次 token を出す。

予定例 `qwen_infer --manifest fixtures/qwen25/manifest.json --prompt Hello --max-new-tokens 1 --no-cache` を動かす。
reference 側も同じ checkpoint を f32 化し、eager attention、推論 mode、同じ token ID を使う。
事前に必要な state、activation、logits の byte 数を計算し、確保可能な device を選ぶ。
実モデルをまだ動かせていない場合は22dを未完了とし、小 fixture の成功と区別する。

### 22e. 新しい cache 出力を束縛して逐次生成する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── qwen_infer.rs  # [更新] 同期成功後のcache候補採用とgreedy反復。
├── src/
│   ├── nn/
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── qwen.rs  # [更新] KvState、extend_cache、decode_mask、decode_with_cache。
│   └── lib.rs  # 本体のモジュール宣言と公開APIの入口。
├── tests/
│   └── transformer_reference.rs  # [更新] cacheなし、chunk、decodeの数値と旧cache保持。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/nn/qwen.rsへKvStateと掲載cache座標helperを追加し、decode_with_cacheのgraph生成で同じprefix範囲、絶対位置、maskを使う。extend_cacheのhost配列計算は座標referenceであり、実行時のコピーと新K/VのSTOREは既存Tensor graphとAFTERで作る。qwen_infer側が全layerの実行と同期成功後だけ候補slotを次回入力へ一括採用する。


入力は読取り専用の旧cacheと今回の回転済みK/V、出力はprefixをコピーした新cacheである。`src/nn/qwen.rs` のcache座標を組込み用抜粋で示す。ここでは一つのlayer、KV head一つを平坦な `[context,D]` で表す。他head/layerにも同じ範囲計算を使う。

KvState周辺へ座標referenceとmask helperを追加する。実cacheのコピーとSTOREは既存graphへ組み立てる。

```rust
// src/nn/qwen.rs（既存ファイルへ組み込む抜粋）
fn extend_cache(old: &[f32], current: &[f32], start: usize,
                context: usize, d: usize) -> Result<(Vec<f32>,usize), &'static str> {
	if d==0 || current.len()%d!=0 { return Err("head width"); }
	let capacity=context.checked_mul(d).ok_or("capacity overflow")?;
	if old.len()!=capacity { return Err("old capacity"); }
	let t=current.len()/d;
	let end=start.checked_add(t).ok_or("position overflow")?;
	if end>context { return Err("context exhausted"); }
	let prefix=start.checked_mul(d).ok_or("prefix overflow")?;
	let mut next=vec![0.;capacity];
	next[..prefix].copy_from_slice(&old[..prefix]);
	next[prefix..end*d].copy_from_slice(current);
	Ok((next,end))
}
fn decode_mask(start: usize, query_t: usize, key: usize, valid: usize) -> bool {
	key<valid && key<=start+query_t
}
```

`prefix` は要素数である。`next` は旧入力と別の所有領域を持ち、Kのprefixは再度RoPEを掛けずにコピーする。新K/VのSTORE後をAFTERで表してattentionに渡す。`decode_mask` は上で検査した `start+T` の範囲内だけ呼ぶ。全layerのK/Vとlogitsの実行・同期が終わってから、KvState候補の全slotを次runへ一括rebindする。

```text
S=1,T=1,D=2, oldV=[2,4,0,0,0,0], currentV=[6,8]
nextV=[2,4,6,8,0,0], next_valid=2, oldV unchanged
query absolute position=1 -> keys0,1 valid -> Q=K=0 -> O=[4,6]
new prompt: valid=0 -> prefix length0; stale bytes never LOAD
```

推論状態を graph とは別に保持するため、`src/nn/qwen.rs` に `KvState` と `decode_with_cache`、runtime 側に実行後の state 採用処理を足す案とする。KvState は layer ごとの K と V の入力 BufferSlot、context 容量、有効長を持つ。入力は読取り専用であり、decode は新しい logits、次の K/V 出力、有効長の候補を返す。全 layer の実行と同期が成功した後だけ候補を現在 state に採用する。失敗途中の layer だけ cache を進めない。

start_pos=S、今回長Tなら checked 加算で S+T<=context を確認する。各 layer は旧 cache の[0,S)を新規 buffer へコピーし、今回の回転済み K と V を[S,S+T)へ保存する。その STORE 完了を AFTER で表し、attention は新規 cache の[0,S+T)だけを読む。Q と K の RoPE は今回 token の絶対位置 S+tを使う。保存済み K はすでに回転しているので、prefix コピー時に再度回転しない。未使用の容量は mask 以前に LOAD guard で排除する。

例えば3 tokenの prefill 後は S=3となり、次の1 tokenは位置3に保存し、有効 key は0、1、2、3の4個になる。query の位置が0ではないため、T=1だけを見て key0以外を mask するのは誤りである。Q=K=0の小例で prefix V=[2,4]、今回 V=[6,8]なら、S=1の query は両方を読み[4,6]になる。この値と cache なしの2 token再計算を照合する。さらに一括 prefill と複数 chunk の logits を比較し、旧 cache bytes が実行前後で同じかを確認する。次回は新しい出力を既存 Plan の入力へ rebindし、以前の入力や純粋 graph を変更しない。新しい prompt では有効長を0へ戻し、古い capacity 内の値が出力へ混入しない test を置く。

- [ ] layer ごとに K と V の固定容量 buffer を持ち、読取り専用の実行時入力スロットへ渡す。
- [ ] `start_pos+T<=context` を検証し、RoPE の絶対位置と書込み位置を一致させる。
- [ ] 既存の有効 prefix と今回の K/V を新規出力 buffer へ保存し、その AFTER を attention が読むようにする。
- [ ] 実行完了後に新しい cache 出力を次回の入力スロットへ束縛し、元の入力 buffer は変更しない。
- [ ] prefill と1 token decode を、cache なしで prefix 全体を再計算した logits と比較する。
- [ ] 新しい prompt の開始時に cache の有効長を初期化し、greedy 生成と EOS 終了を追加する。

予定例 `qwen_infer --manifest fixtures/qwen25/manifest.json --prompt Hello --max-new-tokens 4` を動かす。
20の動的長さを使う前に、静的 prefill と T=1 の別 Plan でもよい。
実体化した cache を純粋 graph へ埋め込まず、元 graph と cache の現在値を分ける。
初版は有効 prefix をコピーする方式にし、05と09の読取り専用入力・新規出力の契約を保つ。
in-place cache 更新はこの段階には追加しない。コピー削減は別の変更として、更新可能な入力の契約と読み書き依存を導入してから行う。

## 完了確認

以下の example と test はこの章で追加する予定である。

```sh
# 作業ディレクトリ: chibigrad/
cargo test
cargo test --test transformer_reference
cargo test --test qwen_tokenizer
cargo run --example attention_reference -- --case zero-qk
cargo run --example tiny_transformer -- --fixture fixtures/transformer/tiny.json
cargo run --example qwen_inspect -- --manifest fixtures/qwen25/manifest.json
cargo run --example qwen_infer -- --manifest fixtures/qwen25/manifest.json --prompt Hello --max-new-tokens 4
```

- [ ] 小例の causal 平均、softmax 行和、RoPE 位置0が期待どおりになる。
- [ ] f32 の小 fixture は `abs_err <= 1e-5 + 1e-4*abs(ref)` を初期基準に各層を比較する。
- [ ] 実モデルは各層と logits を `abs_err <= 1e-3 + 1e-3*abs(ref)` で比較し、最大誤差を残す。
- [ ] argmax の一致だけで数値検証を終えず、近接した logits の差も記録する。
- [ ] cache 有無と chunk 分割の違いで同じ prefix の logits が許容値内になる。
- [ ] dump に token ID、shape、head 対応、位置、cache 有効長を出し、重みの全内容は出さない。
- [ ] Metal と OpenCL は CPU fixture と比較し、未実行 backend は未確認と記録する。

## tinygrad ソースとの対応

参照は commit `c22edff14c79310a9b2df62a78819cf77c8289fa` である。

| 実在シンボルと相対リンク | tinygrad の役割 | chibigrad で簡略化する点 |
|---|---|---|
| [`precompute_freqs_cis` と `apply_rope`](../../tinygrad/tinygrad/llm/model.py#L28) | RoPE の周波数と回転を計算 | YaRN を使わず対象の half-split に固定する |
| [`TransformerConfig`](../../tinygrad/tinygrad/llm/model.py#L142) | 多数のモデルの設定を保持する | dense Qwen2 の必要項目だけ持つ |
| [`TransformerBlock._attention`](../../tinygrad/tinygrad/llm/model.py#L353) | projection、RoPE、cache、GQA を接続する | f32 と通常 attention から始める |
| [`RandMixin.scaled_dot_product_attention`](../../tinygrad/tinygrad/mixin/rand.py#L298) | GEMM と softmax の合成による基準経路 | dropout=0、causal に限定する |
| [`SimpleTokenizer.from_hf`](../../tinygrad/tinygrad/llm/cli.py#L65) | tokenizer 設定と語彙を読む | 固定版 library を使い ID fixture で照合する |
| [`TestAttention.test_independent_positions_jit`](../../tinygrad/test/runtime/test_attention.py#L38) | 位置と未使用 cache の影響を検証する | 単一 sequence の cache から始める |

この参照版の `Transformer.from_hf` は GLM DSA だけを許可する。
その関数を Qwen 対応 loader と説明しない。上記の計算部を参考に、Qwen 用の名前対応を追加する。

## 今は追加しないもの

sampling、量子化、長文最適化、MoE、複数 GPU は追加しない。
通常 attention の正しさを固定してから[23](23-flash-attention.md)でメモリ使用量を減らす。
学習時は KV cache を使わず、[24](24-llm-training.md)の純粋な全 sequence graph を使う。
