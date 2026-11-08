# 23. attention の score 行列を全保存せずに前向きを計算する

## このステップで動かすもの

[22](22-transformer-qwen.md)の通常 attention と同じ出力を、タイル単位の online softmax で返す。
最初は CPU の1行、次に Metal と OpenCL の複数行へ進める。
元計画第10章のタイル化と第11章の Transformer を、flash attention へ拡張する。
19の local layout と20の実行時長さの検証を再利用する。
本文の kernel、example、test、測定コマンドは今後作る予定である。
速度と GPU 実機動作は未測定であり、SOTA 性能を完了条件にしない。

予定の基準例は次とする。

```text
Q=K=0、V=[[2,4],[6,8]]
causal=True  -> O=[[2,4],[4,6]]
causal=False -> O=[[4,6],[4,6]]
```

## 最小実装とデータ

意味は `O=softmax(QK^T/sqrt(d)+mask)V` のまま保つ。
入力は Q、K、V と、query の絶対開始位置、有効 key 長、causal 設定である。
まず f32、dropout=0、単一 batch、同じ query/KV head 数に限定する。
出力 IR は Q/K/V のタイル LOAD、max と sum、exp、row ごとの累積、barrier と STORE である。
タイル化で外部メモリへの読み書きを減らす方針は[FlashAttention 論文](https://arxiv.org/abs/2205.14135)に基づく。
論文の速度の数値を chibigrad の結果として転載しない。

各 query 行について `m=-inf, l=0, u[d]=0` を保つ。
次の key tile に有効な score が一つ以上ある場合だけ、以下を更新する。

```text
m_new = max(m, max(valid_scores))
alpha = if l==0 then 0 else exp(m-m_new)
p_j   = if valid_j then exp(score_j-m_new) else 0
l_new = alpha*l + sum(p_j)
u_new = alpha*u + sum(p_j*V_j)
O     = if l>0 then u/l else 0
```

無効な分岐の `-inf-(-inf)` と範囲外 LOAD は評価しない。
全 mask 行は O=0、保存する場合の logsumexp=-inf と定義する。
比較用の通常 attention にも、全 mask 行だけ同じ明示分岐を入れる。
単に softmax の NaN を後から0へ変える実装にはしない。

## この章のファイル配置

通常attentionとdispatcherはsrc/nn/attention.rs、前向きonline kernelの記述と割当はsrc/schedule.rs、出力言語への描画は各rendererへ置く。f64のmain付きreferenceは一つずつexamplesへ分け、tests/online_attention.rsには本体との比較を置く。

次はこの章の終わりまでに触る関連ファイルの配置案である。現在のcheckoutの一覧ではない。各コミットでは、その下に示す部分だけを追加・変更する。

```text
chibigrad/
├── examples/
│   ├── attention_dispatch.rs  # inferenceとtrainingの経路と理由を表示する。
│   ├── attention_parts.rs  # 通常graphでscoreからOまでを取り出すdriver。
│   ├── book_23_attention_reference.rs  # 本文のコードを配置する。
│   ├── book_23_online_softmax.rs  # 本文のコードを配置する。
│   └── online_attention.rs  # GPUのtile、端数指定。
├── src/
│   ├── nn/
│   │   ├── attention.rs  # AttentionRequest、select_attention、kv_head。
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── qwen.rs  # 同じ位置とcache有効長をdispatcherへ渡す。
│   ├── renderer/
│   │   ├── c.rs  # emit_online_rowによるm、l、uの描画。
│   │   ├── metal.rs  # 同じOnlineAttention IRからMSLを生成する。
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── opencl.rs  # online4のOpenCL出力。
│   ├── runtime/
│   │   ├── metal.rs  # 資源と整数範囲、local sizeの検査。
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── opencl.rs  # head範囲のbind、NDRangeと資源検査。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── schedule.rs  # query tile、local layoutと協調搬入。
├── tests/
│   └── online_attention.rs  # GQA、未対応条件とforward-onlyの確認。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

生成Cやshaderは一時ディレクトリへ出力するため、このRustプロジェクトのツリーには含めない。コード冒頭のコメントで、生成元と生成先を確認する。

## 小さなコミットで進める

### 23a. attention の各中間値を基準として固定する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── attention_parts.rs  # [追加] 通常graphでscoreからOまでを取り出すdriver。
│   └── book_23_attention_reference.rs  # 本文のコードを配置する。
├── src/
│   ├── nn/
│   │   ├── attention.rs  # [更新] attention_referenceの中間出力と全mask分岐。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   └── lib.rs  # 本体のモジュール宣言と公開APIの入口。
├── tests/
│   └── online_attention.rs  # [追加] scalar f64と本体の各中間値の照合。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/nn/attention.rsの既存attention_referenceに比較用の中間出力と全mask行の明示分岐を加える。掲載referenceとmainはexamples/book_23_attention_reference.rsへ一つの単独例として保存する。tests/online_attention.rsはmainの置き場にせず、同じ式のreference_row helperと実際のTensor、生成Cの結果を比較するintegration testを追加する。


入力は一行分の有限score、有効mask、V、出力は安定した確率とOである。`tests/online_attention.rs` のscalar referenceを単独で実行できる例にする。QKの内積とscaleを終えたscoreを受けるので、後のonline版とも同じ値で比較できる。

referenceとmainを含む一行のf64確認例。integration testと通常attention graphは別に作る。

```rust
// examples/book_23_attention_reference.rs（単独で実行する例）
fn reference(scores: &[f64], valid: &[bool], v: &[[f64;2]])
	-> (Vec<f64>,[f64;2],f64) {
	assert_eq!(scores.len(),valid.len());
	assert_eq!(scores.len(),v.len());
	let mut maximum=f64::NEG_INFINITY;
	let mut any=false;
	for j in 0..scores.len() {
		if valid[j] { maximum=maximum.max(scores[j]); any=true; }
	}
	let mut p=vec![0.;scores.len()];
	if !any { return (p,[0.;2],f64::NEG_INFINITY); }
	let mut denominator=0.;
	for j in 0..scores.len() {
		if valid[j] { p[j]=(scores[j]-maximum).exp(); denominator+=p[j]; }
	}
	let mut out=[0.;2];
	for j in 0..scores.len() {
		if !valid[j] { continue; }
		p[j]/=denominator;
		for c in 0..2 { out[c]+=p[j]*v[j][c]; }
	}
	(p,out,maximum+denominator.ln())
}
fn main() {
	let v=[[2.,4.],[6.,8.]];
	let (p,o,_) = reference(&[0.,0.],&[true,true],&v);
	assert_eq!(p,[0.5,0.5]); assert_eq!(o,[4.,6.]);
	let (p,o,lse) = reference(&[f64::NAN;2],&[false,false],&v);
	assert_eq!(p,[0.,0.]); assert_eq!(o,[0.,0.]); assert_eq!(lse,f64::NEG_INFINITY);
	let (p,o,_) = reference(&[1000.,1001.],&[true,true],&v);
	assert!((p.iter().sum::<f64>()-1.).abs()<1e-12 && o[0].is_finite());
	println!("stable probabilities={p:?}");
}
```

`any` は有効scoreが一つでもあるかを記録する。全maskでは指数計算へ進まないのでNaNのscoreさえ読まない。非空行の `maximum` はgraph版ではDETACHしたmaxであり、値のreferenceには微分経路はない。`denominator` は最大値基準の指数和、返す第三要素はlogsumexpである。

```text
Q=[0,0], K0=K1=[0,0], scale=1/sqrt(2) -> scores=[0,0]
query_abs=0, key_abs=[0,1] -> valid=[true,false] -> p=[1,0],O=[2,4]
query_abs=5, key_abs=[4,5,6] -> valid=[true,true,false]
scores=[1000,1001] -> m=1001, exps=[exp(-1),1], finite denominator
```

通常 attention を比較できる状態にしてから score の保存を消す。`src/nn/attention.rs` の `attention_reference` に小入力用の中間出力を加え、`tests/online_attention.rs` に scalar f64 の `reference_row` を置く案である。reference_row は一つの Q 行、K/V の有効範囲、query と key の絶対位置、mask を受け、score、確率、O、必要なら logsumexp を返す。最初に shape と有効長を検査し、score の内積、scale、mask の有効判定、行最大、指数、行和、V の加重和の順に計算する。

有効 key がない行は、max と指数の計算へ入る前に O=0と logsumexp=-infを返す。非空行の graph 版は `m=detach(max(valid_scores))` により安定化し、mask された要素の指数を0とする。全 mask を通常の `-inf` の softmax へそのまま流すと `-inf-(-inf)` が NaN になるため、この分岐は reference と graph の両方で明示する。無効な K/V の bytes を先に読まないことも、中間値の正しさとは別に生成 source で検査する。

Q=K=0の2行では score は全て0、causal の有効集合は順に{0}と{0,1}となる。指数の有効部分は1なので分母は1と2、確率は[1,0]と[1/2,1/2]、出力は[2,4]と[4,6]になる。score が[1000,1001]なら m=1001を引いて[-1,0]から指数を作り、大きな exp(1001)を直接計算しない。絶対 query 位置5、key 位置4、5、6の例では最後だけを無効にする。各段階の fixture は mask と位置も保存し、最終 O だけ一致して途中の mask が誤っている状況を見逃さない。f64 reference と既存 f32 graph と生成 C を並べて比較し、この段階ではまだ通常の score buffer を残す。

- [ ] 通常版の score、mask 後の score、softmax、O を小入力で取り出す。
- [ ] head dimension と scale を検証し、shape 不一致を起動前に拒否する。
- [ ] causal、非 causal、padding mask、全 mask 行を別 fixture にする。
- [ ] 位置0だけを前提にせず、query 開始位置5などの絶対位置も test に入れる。
- [ ] 全 mask 行を含む stable reference を Rust の scalar f64 と既存 f32 graph の両方で動かす。

予定例 `attention_parts --case zero-qk --causal` は上記の `[[2,4],[4,6]]` を返す。
`[1000,1001]` の score でも softmax の行和が1、出力が有限になることを確認する。
このコミットの計算は通常の graph、rangeify、renderer、runtime を通す。

### 23b. CPU で1行の online softmax を実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_23_online_softmax.rs  # 本文のコードを配置する。
│   └── online_attention.rs  # [追加] CPUと通常版を比較するdriver。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # [更新] emit_online_rowによるm、l、uの描画。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── schedule.rs  # [更新] OnlineAttentionのkernel記述とCPUの行割当。
├── tests/
│   └── online_attention.rs  # [更新] 後半最大、空tile、再実行の照合。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/schedule.rsに前向き専用OnlineAttentionを加え、C rendererのemit_online_rowへ渡す。m、l、uをkey tileより外の行scopeで描く。掲載onlineとmainはexamples/book_23_online_softmax.rsへ置く数値確認例であり、rendererのRustホスト処理へ丸ごと貼らない。online_attentionは実際のkernel source生成とCPU runtimeを通す別exampleにする。


入力は23aと同じscore/mask/Vとtile幅、出力はOとlogsumexpである。次は単独で実行できるonline更新の例。`src/renderer/c.rs` の `emit_online_row` はこのループをCへ描き、scoreはQ/Kの内積からtile内だけに生成する。例はm/l/uの数値検査に絞る。

onlineとmainを含むm、l、uの数値確認例。CPU rendererには対応するC生成を実装する。

```rust
// examples/book_23_online_softmax.rs（単独で実行する例）
fn online(scores: &[f64], valid: &[bool], v: &[[f64;2]], tile: usize)
	-> ([f64;2],f64) {
	assert!(tile>0 && scores.len()==valid.len() && scores.len()==v.len());
	let (mut m,mut l,mut u)=(f64::NEG_INFINITY,0.0f64,[0.;2]);
	for begin in (0..scores.len()).step_by(tile) {
		let end=(begin+tile).min(scores.len());
		let mut tile_max=f64::NEG_INFINITY;
		let mut any=false;
		for j in begin..end {
			if valid[j] { tile_max=tile_max.max(scores[j]); any=true; }
		}
		if !any { continue; }
		let new_m=m.max(tile_max);
		let alpha=if l==0. { 0. } else { (m-new_m).exp() };
		l*=alpha;
		for c in 0..2 { u[c]*=alpha; }
		for j in begin..end {
			if !valid[j] { continue; }
			let p=(scores[j]-new_m).exp();
			l+=p;
			for c in 0..2 { u[c]+=p*v[j][c]; }
		}
		m=new_m;
	}
	if l==0. { ([0.;2],f64::NEG_INFINITY) }
	else { ([u[0]/l,u[1]/l],m+l.ln()) }
}
fn main() {
	for tile in [1,2,3,8] {
		let (o,_) = online(&[0.,2f64.ln()],&[true,true],&[[2.,4.],[6.,8.]],tile);
		assert!((o[0]-14./3.).abs()<1e-12 && (o[1]-20./3.).abs()<1e-12);
	}
	let (o,lse)=online(&[f64::NAN],&[false],&[[f64::NAN;2]],1);
	assert_eq!(o,[0.;2]); assert_eq!(lse,f64::NEG_INFINITY);
	println!("online correction and all-mask passed");
}
```

`m` はこれまでの最大score、`l` はその基準の指数和、`u` は同じ指数重みを掛けたVの和である。新しい最大を見つけると `alpha` で両方を同時に縮める。`l=0` の場合は差を計算しないので `-inf-(-inf)` が発生しない。全tileが空なら最後まで初期状態を保つ。

```text
tile0 score=0:       m=0,     l=1,   u=[2,4]
tile1 score=ln(2):   m=ln(2), alpha=1/2
old state rescaled:           l=1/2, u=[1,2]
new contribution p=1,V=[6,8]: l=3/2, u=[7,10]
O=[14/3,20/3], logsumexp=ln(3)
```

比較用 graph は残し、`src/schedule.rs` に前向き専用の `OnlineAttention` kernel 記述、C renderer に `emit_online_row` を追加する案とする。入口は一つの Q 行、K/V の slot、有効長、位置、tile 幅であり、出力は O の一行だけである。m、l、u はその行を開始する時に初期化し、key tile の外側で寿命を持たせる。tile ごとに有効 score を作り、有効集合が空なら K/V の無効 LOAD や max を評価せず、状態をそのまま次へ渡す。

有効 tile では新しい最大 m_newを決め、旧状態の指数の基準を合わせる alpha を計算する。旧 l=0なら alpha=0とし、m=-infの差を計算しない。続いて各有効 score の p を新しい基準で計算し、旧 l と u に alpha を掛けてから今回の sum(p)と sum(p*V)を足す。全 tile が終わって l>0なら u/lを保存し、そうでなければ0を保存する。この更新の l は確率の総和ではなく、現在の最大で正規化する前の指数和である。

補正の小例は d=2、Q=[sqrt(2),0]、K=[[0,0],[ln(2),0]]、V=[[2,4],[6,8]]、tile幅1とする。scale 後の score は[0,ln(2)]になる。最初は m=0、l=1、u=[2,4]となる。次は m_new=ln(2)、alpha=1/2、今回のp=1なので l=3/2、u=[7,10]、O=[14/3,20/3]になる。alpha を省いた平均[4,6]とは異なるため、この例で後半最大の誤りを検出できる。全 mask の場合は全 tile を飛ばして m=-inf、l=0、u=0を保ち、最後だけ0を保存する。最後の tile の端数、tile 幅変更、同じ Plan の別長でこの状態の初期化と guard を確認する。生成 C の loop と acc scope を dump し、score の全行列を確保していないことを実 buffer 一覧でも確かめる。

- [ ] key 軸を小さな tile に分割し、m、l、u を C の累積値として保持する。
- [ ] tile を読み進めるたびに m の変更で既存の l と u を補正する。
- [ ] 有効 key がない tile は状態を更新せず、全 mask 行を明示的に処理する。
- [ ] 最後の不完全 tile を guard し、無効な K と V の LOAD を生成しない。
- [ ] tile 幅1、2、3、8で通常版と同じ結果を得る。

予定例 `online_attention --backend cpu --case zero-qk --tile-n 1` を動かす。
max が後の tile で増える入力を入れ、補正を省いた実装なら失敗する fixture を用意する。
異なる長さの連続実行で前回の m、l、u が残らないことも確かめる。
入力の有効領域は有限値に限定し、その外側に NaN を置いて guard を検査する。

### 23c. GPU の local memory に Q/K/V の tile を置く

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── online_attention.rs  # [更新] GPUのtile、端数指定。
├── src/
│   ├── renderer/
│   │   ├── metal.rs  # [更新] 同じOnlineAttention IRからMSLを生成する。
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── opencl.rs  # [更新] online4のOpenCL出力。
│   ├── runtime/
│   │   ├── metal.rs  # [更新] 資源と整数範囲、local sizeの検査。
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── opencl.rs  # [更新] head範囲のbind、NDRangeと資源検査。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── schedule.rs  # [更新] query tile、local layoutと協調搬入。
├── tests/
│   └── online_attention.rs  # [更新] GPUのguardと同期、通常版の比較。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/schedule.rsのOnlineAttentionにBM、BN、K/V local tileの搬入と二つのbarrierを加える。MetalとOpenCL rendererで同じ同期順序を描き、runtimeはlocal byte数、lane数、uint範囲を起動前に確認する。掲載online4はOpenCL rendererが生成するshaderであり、Rust側にはIR変換とsource生成の処理を実装する。


入力はD=4のQ/K/V、query開始位置、有効長、出力はquery二行ずつのOである。以下は組込み用の抜粋として示すOpenCL kernelで、local size=2、BM=2、BN=3に固定する。K/Vを搬入し、各担当laneが一行のonline状態を更新する。実機未確認で、Metalには同じkernel IRを描く。

OnlineAttentionの固定D=4 kernelを描いたOpenCL C。Metal側は同じIRからMSLを生成する。

```c
// src/renderer/opencl.rs が生成するコード。生成先: <一時ディレクトリ>/online4.cl
__kernel void online4(__global const float *q, __global const float *k,
                     __global const float *v, __global float *o,
                     uint Tq, uint Tk, uint query_start, uint causal) {
	uint lane=get_local_id(0), row=get_group_id(0)*2+lane;
	__local float kt[12], vt[12];
	float qr[4]={0,0,0,0}, u[4]={0,0,0,0};
	float m=-INFINITY, l=0.0f;
	if (row<Tq) for (uint c=0;c<4;++c) qr[c]=q[row*4+c];
	for (uint begin=0;begin<Tk;begin+=3) {
		for (uint z=lane;z<12;z+=2) {
			uint key=begin+z/4, c=z%4;
			kt[z]=(key<Tk)?k[key*4+c]:0.0f;
			vt[z]=(key<Tk)?v[key*4+c]:0.0f;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if (row<Tq) for (uint j=0;j<3;++j) {
			uint key=begin+j;
			if (key>=Tk || (causal && key>query_start+row)) continue;
			float score=0.0f;
			for (uint c=0;c<4;++c) score+=qr[c]*kt[j*4+c];
			score*=0.5f;
			float nm=fmax(m,score), alpha=(l==0.0f)?0.0f:exp(m-nm);
			float p=exp(score-nm);
			l=alpha*l+p;
			for (uint c=0;c<4;++c) u[c]=alpha*u[c]+p*vt[j*4+c];
			m=nm;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (row<Tq) for (uint c=0;c<4;++c) o[row*4+c]=(l>0.0f)?u[c]/l:0.0f;
}
```

`kt,vt` の24個のf32は96byte、Qとuは各laneのprivate配列である。この実装はBN個のscoreを保存せず、搬入tile内をscore一つずつ更新するので23bのtile幅1と同じ更新式になる。無効queryのlaneも搬入と両barrierを実行する。`query_start+Tq` とすべての添字がuintに収まることをhostで検査する。keyの絶対開始位置は0に固定する。任意padding mask、別のkey開始位置、GQAはまだこのkernelの対応外とする。

```text
Tq=5,Tk=7,BM=2,BN=3,D=4 -> groups=3,key tiles=3
last query group: lane0=row4 valid,lane1=row5 only stages+barriers
last key tile: key6 valid,key7/8 staged0 but excluded from max/exp
private Q/u + local K/V=96 bytes; no full score allocation
```

CPU の online 状態の意味をそのまま GPU へ移す。`src/schedule.rs` に query tile の割当と local layout を足し、Metal と OpenCL の renderer が同じ同期順序を出す案とする。最初は有効 query 行ごとに一つの担当 lane が d の内積と u[d] の更新を逐次行い、全 lane は K/V の協調搬入へ参加する。この限定なら、m、l、u の書込み担当を一意にできる。内積や行和を lane 間に分割する最適化は、その後に reduction の同期を別に検証して加える。

group は BM行の Q を一度搬入し、BN行ずつの K/V を順に搬入する。各 tile は有効な global LOAD または0の local STORE、barrier、QK と online 更新、barrier の順で進める。無効 query 行の lane も二つの barrier を通り、最終 O の STORE だけを省く。無効 key は score の max に加えず、exp の確率寄与も0とする。0詰めされた K の score=0を有効 score と見なすと、全て負の有効 score より大きくなり得るため、データの0と有効 mask は別に保持する。

予定 shape のTq=5、Tk=7、BM=2、BN=3なら query group は3個、key tile は3回になる。最後の query group は行4だけが有効で、最後の key tile は key6だけが有効である。D=4で Q、K、V と score tile を全て local に置く構成なら、それぞれ8、12、12、6個の f32で計152 byteとなり、row 累積等の追加領域は別に数える。実際に確保した配列と lane 数から資源量を検査し、固定の係数で推測しない。query 開始位置5なら各行の位置は5..9であり、key の絶対位置との比較で causal を決める。source で同期と範囲を確認し、同じ mask と入力を使う通常版へ数値を照合した後に、利用可能な GPU ごとの結果を記録する。

- [ ] まず一つの group が query の小タイルを担当し、K と V を順に local memory へ運ぶ。
- [ ] QK と PV は通常の f32 積和を使い、hardware matrix 命令はまだ使わない。
- [ ] 全 lane が LOAD 後と local buffer 再利用前の barrier を通る構造にする。
- [ ] query と key の端数を別々に guard し、無効 lane に単位元を与える。
- [ ] Metal で検証後、同じ kernel IR を OpenCL に出して通常版と比較する。

予定例 `online_attention --backend metal --shape 1,2,5,7,4 --tile-m 2 --tile-n 3` を追加する。
shape は batch、heads、query 長、key 長、head dimension の順とする。
causal 判定は key の絶対位置を query の絶対位置と比べる。
1 token decode と prefill で同じ左上基準の mask を使い回さない。
local byte 数と lane 数が device の上限を超える場合は、起動前に通常版を選ぶ。

### 23d. GQA と dispatcher を model へ接続する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── attention_dispatch.rs  # [追加] inferenceとtrainingの経路と理由を表示する。
├── src/
│   ├── nn/
│   │   ├── attention.rs  # [更新] AttentionRequest、select_attention、kv_head。
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── qwen.rs  # [更新] 同じ位置とcache有効長をdispatcherへ渡す。
│   └── lib.rs  # 本体のモジュール宣言と公開APIの入口。
├── tests/
│   └── online_attention.rs  # [更新] GQA、未対応条件とforward-onlyの確認。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/nn/attention.rsの通常版入口の前へAttentionRequest、select_attention、kv_headを加える。src/nn/qwen.rsの呼出しから同じ絶対位置とcache有効長を渡し、未対応条件では既存attention_referenceのgraphを返す。D=4、headごとの範囲bind等の条件だけを登録し、勾配要求は通常版へ戻す。


入力はdtype、head数、mask、gradient要求と資源検査結果、出力は選択経路と理由である。`src/nn/attention.rs` に置く組込み用抜粋では、forward-onlyの条件を選択関数自身に持たせる。`fits_local` は実際の配列byte数とlane数をdevice上限と比べた結果で、backend名から推測しない。

既存attention入口へdispatcherとGQAのhead対応を追加する。

```rust
// src/nn/attention.rs（既存ファイルへ組み込む抜粋）
struct AttentionRequest {
	hq: usize, hkv: usize, f32_input: bool, contiguous: bool,
	head_dim: usize, key_start: usize,
	per_head_bindings: bool, index_bounds_ok: bool,
	causal_or_none: bool, needs_gradient: bool, fits_local: bool,
	probe_ok: bool,
}
fn select_attention(r: &AttentionRequest) -> Result<(&'static str,&'static str), &'static str> {
	if r.hkv==0 || r.hq==0 || r.hq%r.hkv!=0 { return Err("invalid GQA"); }
	let fallback = if r.needs_gradient { Some("backward requires ordinary graph") }
		else if !r.f32_input { Some("dtype") }
		else if r.head_dim!=4 { Some("head dimension requires generic kernel") }
		else if r.key_start!=0 { Some("key position requires generic kernel") }
		else if !r.contiguous { Some("stride") }
		else if !r.per_head_bindings { Some("head binding") }
		else if !r.index_bounds_ok { Some("kernel uint index range") }
		else if !r.causal_or_none { Some("mask") }
		else if !r.fits_local { Some("local resources") }
		else if !r.probe_ok { Some("backend unverified") }
		else { None };
	Ok(match fallback {
		Some(reason) => ("ordinary",reason),
		None => ("online","supported forward"),
	})
}
fn kv_head(hq: usize, query_heads: usize, kv_heads: usize) -> usize {
	assert!(kv_heads>0 && query_heads%kv_heads==0 && hq<query_heads);
	hq/(query_heads/kv_heads)
}
```

`needs_gradient` を最初に検査するので、推論用kernelへ学習graphを流してDETACHで隠す余地がない。この時点の23cはD=4、keyの絶対開始位置0に限るため、D=64や開始位置が0以外なら通常版へ戻す。一般化したkernelを後で追加した時だけ、その対応条件を広げる。`index_bounds_ok` はquery_start+Tq、配列の最大添字、tileの丸上げがkernelのuintへ収まることをchecked計算した結果である。

`kv_head` はK/Vの先頭offsetを求める時に使い、全query head分のK/Vを複製しない。23cのkernel自体は一つのheadしか受け取らないので、初版は各batch/query headについて一回ずつ起動する。連続した `[B,H,T,D]` なら、Q/Oの先頭は `(b*Hq+hq)*Tq*D`、K/Vの先頭は `(b*Hkv+hkv)*Tk*D` 要素である。これらをchecked計算し、deviceのoffsetとalignment条件を満たすhead範囲だけをbindする。キャッシュの確保容量strideがTkと異なる場合や、backendがそのhead範囲をbindできない場合、`per_head_bindings=false` で通常版へ戻す。modelへ接続する際は通常版とonline版が同じ絶対位置とcache有効長を受け取る。

```text
Hq=4,Hkv=2 -> hkv=[0,0,1,1]
change only KV0 values -> query0/1 change; query2/3 unchanged
needs_gradient=true -> ordinary, reason="backward requires ordinary graph"
head_dim=64 -> ordinary, reason="head dimension requires generic kernel"
key_start=5 -> ordinary, reason="key position requires generic kernel"
B=1,Hq=2,Tq=5,Tk=7: ordinary score=70*f32=280 bytes
online: full score slot absent, tile/private accumulators listed separately
```

`src/nn/attention.rs` に `select_attention` を置き、入力の dtype、shape、stride、mask 方式、位置、device 資源、gradient 要求を一度に判定する案である。返すのは通常版または online 版と、その選択理由である。model の呼出し位置を変えず、推論でも条件を満たさなければ通常の Tensor 合成 graph を作る。学習は勾配が必要な入力や parameter を検出した時点で通常版へ戻し、未定義 backward を0や DETACH に置換して通さない。forward-only の意味を選択関数と test の両方に固定する。

GQA は query head 数Hqが KV head 数Hkvの整数倍であることを検査し、R=Hq/Hkv、hkv=hq/Rで K/V の先頭位置を求める。query4、KV2なら対応は[0,0,1,1]であり、K/V を4 headの実 buffer へ複製しない。KV0の V だけを変える小 test で query0と1だけが変わり、2と3は変わらないことを確認する。shape の対応だけでなく model の logits と cache 有効長を22の通常版に比べ、kernel の局所的な成功が model の選択位置の誤りを隠さないようにする。

一時領域の確認は Plan の実確保表で行う。通常版が B*Hq*Tq*Tk個の score を保持する設定なら、予定のB=1、Hq=2、Tq=5、Tk=7は70個である。online 版には全体 score の確保がなく、代わりに tile と行の m/l/uが存在する。Q/K/V と O の領域や runtime の再利用領域をこの削減に混ぜず、peak は寿命を含めて比較する。test 用の capability 無効化、未対応 mask、training を各々入力し、理由付きで通常版が選ばれ同じ値になることを確かめる。計測設定はこの後に加え、同一入力と同期条件で時間と実 byte 数を記録する。計測していない速度や score 以外の削減を主張しない。

- [ ] query head から KV head への対応を追加し、K/V の全複製を避ける。
- [ ] 対応 dtype、寸法、local memory、mask、勾配要求を条件として選択する。
- [ ] 未対応条件では22の通常 attention を選び、選択理由を dump に残す。
- [ ] 学習 graph はこの章では通常版へ固定し、forward-only kernel を微分可能と扱わない。
- [ ] model の logits と cache 有効長を通常版に照合してから、計測用の設定を追加する。

予定例 `attention_dispatch --fixture fixtures/transformer/tiny.json --mode inference` を動かす。
`--mode training` は通常版へ戻り、24でその graph の自動微分を使う。
score の全体 `[B,H,Tq,Tk]` を確保しないことを Plan の実 buffer 一覧で確認する。
一時領域の計算量は tile と row 累積に依存し、出力 O と入力 Q/K/V の領域とは区別する。
compile、転送、kernel、同期の時間と、peak 確保 byte 数を通常版と同じ条件で記録する。

## 完了確認

以下の example と test はこの章で作る予定である。

```sh
# 作業ディレクトリ: chibigrad/
cargo test
cargo test --test online_attention
cargo run --example attention_parts -- --case zero-qk --causal
cargo run --example online_attention -- --backend cpu --case zero-qk --tile-n 1
cargo run --example online_attention -- --backend metal --shape 1,2,5,7,4 --tile-m 2 --tile-n 3
cargo run --example online_attention -- --backend opencl --shape 1,2,5,7,4 --tile-m 2 --tile-n 3
cargo run --example attention_dispatch -- --fixture fixtures/transformer/tiny.json --mode inference --dump-plan
```

- [ ] 基準例の出力は期待値と一致する。
- [ ] CPU、GPU、通常版を f64 reference に対し `abs_err <= 1e-4 + 1e-4*abs(ref)` で比較する。
- [ ] query/key 長の1、tile直前、tile一致、tile直後と、GQA 比率1、2を検証する。
- [ ] 全 mask 行、空の有効 key、後半 tile が最大の score、極端な有限 score を試す。
- [ ] 無効領域の NaN が有効出力へ流れず、全 mask 行は0になって NaN を含まない。
- [ ] dump で全 lane の barrier と、local buffer の再利用依存を確認する。
- [ ] 一つの学習 step が通常版の forward と backward を通り、未定義 gradient を隠さない。
- [ ] 実機を使っていない backend と、未測定の速度を確認済みとして記述しない。

## tinygrad ソースとの対応

参照は commit `c22edff14c79310a9b2df62a78819cf77c8289fa` である。

| 実在シンボルと相対リンク | tinygrad の役割 | chibigrad で簡略化する点 |
|---|---|---|
| [`RandMixin.scaled_dot_product_attention`](../../tinygrad/tinygrad/mixin/rand.py#L298) | GEMM と softmax の通常 attention | 比較用に同じ分解を残す |
| [`OpMixin._softmax`](../../tinygrad/tinygrad/mixin/op.py#L680) | max 減算による安定化 | 全 mask 行の動作も明示する |
| [`_amd_flash_attention`](../../tinygrad/tinygrad/llm/kernels/amd.py#L309) | local tile と online 状態で前向きを計算 | AMD WMMA を移植せず f32 積和から始める |
| [`_amd_flash_decode_combine`](../../tinygrad/tinygrad/llm/kernels/amd.py#L258) | 部分結果を max と正規化係数で結合する | 最初は key tile を一 group 内で順番に処理する |
| [`flash_attention`](../../tinygrad/tinygrad/llm/kernels/amd.py#L396) | 寸法条件の検査と通常版への fallback | Metal、OpenCL の能力を個別に検証する |
| [`amd_custom_kernels_supported`](../../tinygrad/tinygrad/llm/kernels/amd.py#L19) | device と renderer で利用可否を判定する | backend 名だけで最適化を選ばない |

AMD の source が存在することは、Metal と OpenCL で実行できた証拠にはならない。

## 今は追加しないもの

dropout、FP8、split-K、複数 GPU、hardware matrix 命令は追加しない。
この章の flash kernel は forward-only とし、24の学習は通常 attention の backward を使う。
学習用 flash を追加する場合は、行統計から P を再計算する backward を独立した次の課題にする。
dQ、dK、dV と GQA の勾配加算を有限差分と通常版で確認してから、学習 dispatcher に登録する。
その段階まで、学習時にも score 保存を除去できたとは主張しない。
