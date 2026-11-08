# 09. 融合と分割で閉じた kernel graph を作り、Plan を再実行する

## このステップで動かすもの

[08-shapes-rangeify.md](08-shapes-rangeify.md) の別 kernel の連鎖を必要な範囲で融合する。
`(x+1)*2` に `[1,2,3]` を渡し、単一 kernel で `[4,6,8]` を返す。
明示 STAGE を挟むと二つの kernel に分割して同じ値を返す。
元計画第 6 章の fusion、materialization と実行順に対応する。
以下の API、IR、コマンド、数値は未実装の作業計画である。

## 最小実装とデータ

純粋な shape 付き元グラフは不変に保つ。
「この値を保存する」という決定を、別の materialization 表に保持する。
出力 storage の実体も、元ノードを BUFFER へ置換せず別の束縛表へ置く。
学習で必要になる重みを CONST に変えて保存しない。

最初の融合対象は、単一の消費先を持つ要素 ALU の連鎖に限る。
出力座標を入力へ伝え、中間 MUL や ADD の値をそのまま式に埋める。
生成する最終 STORE と output RANGE は変えない。

```text
before:
  kernel0: tmp[i] = x[i] + 1
  kernel1: out[j] = tmp[j] * 2
after:
  kernel0: out[i] = (x[i] + 1) * 2
```

`STAGE(value)` は明示的な保存境界にする。
STAGE の producer を別 kernel にし、consumer は保存済み buffer を読む。
二つ以上の消費先を持つ計算も、初版では一度保存する。
reduction の出力を別 reduction の内側へ再計算する融合は行わない。

kernel body は閉じた関数にする。
外部 storage はすべて PARAM へ置き換え、CALL の引数に元の buffer state を並べる。
各 body の RANGE は END で閉じ、他 kernel の軸を捕捉しない。
scalar PARAM も明示署名に含め、呼出し側から束縛する。
ここで「閉じる」は、許可した PARAM 以外の自由入力がないことを意味する。

`Plan` は compiled kernel、各 kernel の引数スロット、依存順、新規 allocation を持つ。
`run(inputs)` は入力 BUFFER を束縛し、保存領域を確保して同期的に順番に呼ぶ。
Plan を二回実行するときは、新しい入力と重みをスロットへ渡す。
各 run は中間 state を作り直し、前回の実体化値を返さない。
compiled library は Plan が所有し、再コンパイルせずに使う。

添字付き式のキャッシュは、NodeId のみでは作らない。
融合先の座標、region、読む state が違えば lower 結果も違う。
まずキャッシュなしで正しさを確認する。
カーネル間の依存は新規出力を読む純粋 DAG なので、トポロジカル順で十分になる。

## この章のファイル配置

保存境界、閉じた body と Plan はこの章で追加する schedule にまとめる。座標を伝える処理は既存 rangeify、compiled library の所有と FFI は既存 runtime/cpu を使う。Plan の std 例と、実 library を使う fusion_plan は別に保存する。

次はこの章の終わりまでに触る関連ファイルの配置案である。現在のcheckoutの一覧ではない。各コミットでは、その下に示す部分だけを追加・変更する。

```text
chibigrad/
├── examples/
│   ├── book_09_plan_bindings.rs  # 依存順と毎回の束縛をホスト関数で確かめる std の独立例。
│   └── fusion_plan.rs  # 実 library を一度作り、二回の入力で Plan を実行する。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # 融合された MUL、ADD、acc 更新の C 生成。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── runtime/
│   │   ├── cpu.rs  # 既存の compiled library を Plan の同期呼出しへ接続する。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rangeify.rs  # kernel ごとに分けた座標と束縛環境。
│   ├── schedule.rs  # Plan 全体の入力検査と run 単位の実体化表。
│   ├── uop.rs  # STAGE と保存境界の op。
│   └── verify.rs  # verify_closed_kernel と verify_kernel_graph。
├── tests/
│   └── fusion_plan.rs  # 境界違反、異なる座標、再実行と元グラフ保持の統合確認。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

生成Cやshaderは一時ディレクトリへ出力するため、このRustプロジェクトのツリーには含めない。コード冒頭のコメントで、生成元と生成先を確認する。

## 小さなコミットで進める

### コミット 1. 単一消費先の要素演算を融合する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── fusion_plan.rs  # [追加] 非融合と融合の出力、kernel 数と中間 buffer を比較する入口。
├── src/
│   ├── lib.rs  # [変更] schedule モジュールと保存境界選択 API の公開。
│   ├── rangeify.rs  # [変更] 保存しない要素演算へ座標を伝える。
│   └── schedule.rs  # [追加] collect_consumers と choose_materializations。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/schedule.rs を追加し、collect_consumers と choose_materializations を置く。lib.rs に schedule の宣言と example に必要な入口を追加する。rangeify.rs は選ばれた保存境界だけで再帰を止め、融合を行う。掲載抜粋の STAGE 判定はコミット 3 の境界処理まで含めた形であり、STAGE の builder と分割処理はそこで追加する。examples/fusion_plan.rs で既存の二 kernel の比較経路を維持する。

新しいモジュールは、既存の宣言を残して次を追記する。内部の補助関数は各モジュールに置き、exampleから使う入口だけを公開する。

```rust
// src/lib.rs（モジュール宣言を追加）
pub mod schedule;
```


前章では ADD の結果を必ず保存していたため、今回は保存する場所を lower の前に選ぶ。候補ファイル `src/schedule.rs` の設計案 `collect_consumers(roots)` が到達可能な DAG の使用辺を集計し、`choose_materializations` が最終出力と保存が必要な値の表を作る。要素 ALU の単一消費先の連鎖では中間を表へ入れず、元ノード自体は変更しない。最初の実装では使用辺を数える保守的な方法でよく、同じ consumer の二入力に使う値の再計算判断は最適化しない。

`(x+1)*2` の最終 MUL の位置 i を lower すると、入力 ADD が保存境界でないため `value_at(ADD,i,scope)` を呼ぶ。そこから x[i] の LOAD と 1 の scalar 値を作り、その ADD の戻り番号を MUL へ直接渡す。入力 x=[1,2,3] なら i=0 で 1+1=2、それを掛けて 4、続いて 6 と 8 を得る。中間 ADD の出力用 ALLOC、STORE、独立 RANGE は生成しない。

局所テストでは前章の二 kernel 経路を比較先として残し、融合した一 kernel の結果と全要素を照合する。schedule dump では kernel が一つ、中間 buffer が 0、入力 x と新規出力だけがあることを調べる。kernel dump では ADD と MUL が同じ軸 i を使い、元グラフの dump には両ノードが残ることを見る。library と固定 ABI の経路は既存のままであり、値を得るための処理は中間メモリの読書きから同じ body 内の scalar 式へ変わる。kernel 数だけで速度が上がったと結論しない。


組込み用の抜粋である。`src/schedule.rs` の保存境界の選択を示す。`linearize` は 01 の到達可能ノードを一度だけ並べる DFS、NodeId は `Copy + Eq + Hash` である。元 graph は借用するだけで変更しない。

保存境界の選択を schedule へ追加する。STAGE を含む判定はコミット 3 の分割処理まで含めた抜粋である。

```rust
// src/schedule.rs（既存ファイルへ組み込む抜粋）
fn choose_materializations(g: &Graph, roots: &[NodeId]) -> std::collections::HashSet<NodeId> {
	let mut reachable = std::collections::HashSet::new();
	for &root in roots { reachable.extend(linearize(g,root)); }
	let mut uses = std::collections::HashMap::<NodeId,usize>::new();
	for &id in &reachable {
		for &src in &g.nodes[id.0].src { *uses.entry(src).or_default() += 1; }
	}
	let mut save: std::collections::HashSet<_> = roots.iter().copied().collect();
	for &id in &reachable {
		let op = &g.nodes[id.0].op;
		let compute = matches!(op,Op::Add | Op::Mul | Op::Reduce(_) | Op::Stage);
		if matches!(op,Op::Stage | Op::Reduce(_)) || (compute && uses.get(&id).copied().unwrap_or(0) > 1) {
			save.insert(id);
		}
	}
	save
}
```

使用数は consumer ノードを一度だけ訪ね、その src 辺を全部数える。`MUL(t,t)` なら二本の辺として数える保守的な規則である。root は必ず保存し、単一使用の ADD と MUL は境界にしない。reduction の結果は常に保存するので、別の出力位置ごとに reduction を再実行する融合は起きない。STAGE の保存ノードを生産するときはその入力を計算し、consumer では STAGE の保存済み値を読む。

この入力の途中経過を並べる。

```text
x -> ADD(x,1)=t -> MUL(t,2)=y
uses[t]=1, roots={y}, save={y} -> one kernel
t -> MUL(t,2)=y, ADD(t,3)=z
uses[t]=2, roots={y,z}, save={t,y,z} -> three kernels
```

- [ ] 元グラフから consumer 数を集める。
- [ ] 単一消費先の ADD と MUL の境界を materialize しない。
- [ ] lower で同じ出力座標を producer へ伝える。
- [ ] `(x+1)*2` を単一 kernel で `[4,6,8]` にする。
- [ ] 前章の非融合経路とも出力を比較する。

kernel 数の削減は構造確認であり、速度向上を測定したとは扱わない。
入力に変更を加えず、中間 allocation が消えることを dump で確認する。

### コミット 2. reduction の入力側を融合する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── fusion_plan.rs  # [変更] sum(x*w+1) を非融合経路と照合する。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # [変更] 融合された MUL、ADD、acc 更新の C 生成。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── rangeify.rs  # [変更] LowerEnv と fused_value_at、reduction 内部軸への座標伝播。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

LowerEnv と fused_value_at は src/rangeify.rs に追加し、08 の lower_reduce_add の term 構築をこの関数に接続する。buffer slot と scalar slot の環境を分け、reduction の scope を使う。renderer/c.rs は生成された scalar 式を既存の逐次 loop 内に出力し、掲載した C はその出力として確認する。


要素演算の座標伝播を、reduction が読む一要素にも適用する。候補ファイル `src/rangeify.rs` の設計案 `lower_reduction_input` は、REDUCE の内部軸 k を使って `value_at(input,k,inner_scope)` を呼ぶ。最終出力は scalar `[]` なので出力用の配列軸はなく、k は accumulator を更新する内側 region だけに属する。単一消費先で純粋な MUL と ADD なら、その入力に到達するまで再帰的に式を生成する。

`sum(x*w+1)` では k ごとに x[k] と w[k] を LOAD し、MUL、1 を足す ADD、acc の更新を同じ body に置く。指定入力の積は 2、6、12、20、足す値は 3、7、13、21 であり、acc は 0、3、10、23、44 と進む。MUL と ADD の全配列を保存する必要がなく、REG acc と最終一要素出力だけを作れる。加算順は k=0 からの逐次順を維持し、式の融合を理由に reduction の順を変えない。

局所テストは非融合の中間 `[2,6,12,20]`、`[3,7,13,21]` を使う経路の 44 と比較する。dump では入力 LOAD、MUL、ADD、REG の更新が k の中にあり、最終 STORE が END k の後にあることを確認する。reduction 結果を EXPAND して他の要素計算へ使う場合は保存境界にし、各位置で同じ reduction を再実行しない。中間を消せる条件は純粋な入力側だけに限定する。既存の accumulator 初期化と state の検証をそのまま使うことで、fusion が loop の寿命や LOAD の時点を変えていないことを確認できる。


組込み用の抜粋である。`src/rangeify.rs` の融合した一要素の lowering を示す。`env.slots` は別 kernel で保存する値または外部入力の NodeId から、この kernel のローカル buffer slot への対応表である。`env.root` は今回だけ再帰を止めない最終値、`env.scope` は現在の region である。scalar 入力は別の `env.scalar_slots` に登録し、`Expr::ScalarParam { slot, owner }` は scalar PARAM の値を表す。buffer slot と scalar slot はそれぞれ二配列 ABI 内の番号である。

kernel ごとの LowerEnv を使う融合の座標 lowering を追加する。

```rust
// src/rangeify.rs（既存ファイルへ組み込む抜粋）
fn fused_value_at(g: &Graph, id: NodeId, at: Coordinate, env: &LowerEnv) -> Result<Expr,String> {
	if id != env.root {
		if let Some(&slot) = env.slots.get(&id) {
			return Ok(Expr::Load { slot,coordinate:at,scope:env.scope });
		}
	}
	if let Some(&slot) = env.scalar_slots.get(&id) {
		return Ok(Expr::ScalarParam { slot,owner:env.owner });
	}
	let n = &g.nodes[id.0];
	match n.op {
		Op::Const(Scalar::F32Bits(bits)) => Ok(Expr::Const(f32::from_bits(bits))),
		Op::Add | Op::Mul => {
			let a = fused_value_at(g,n.src[0],at.clone(),env)?;
			let b = fused_value_at(g,n.src[1],at,env)?;
			Ok(if matches!(n.op,Op::Add) { Expr::Add(Box::new(a),Box::new(b)) } else { Expr::Mul(Box::new(a),Box::new(b)) })
		}
		Op::Expand => fused_value_at(g,n.src[0],Coordinate::Const(0),env),
		Op::Stage if id == env.root => fused_value_at(g,n.src[0],at,env),
		Op::Reduce(_) => Err("reduction output needs a materialization boundary".into()),
		_ => Err("missing explicit input binding".into()),
	}
}
```

reduction の入力にも同じ `at` を渡すだけで要素演算がその body に入る。`sum(x*w+1)` の lowering は 08 の `lower_reduce_add` の term 生成だけをこの関数へ替え、`at=k`、scope=内側 region にする。scalar CONST の EXPAND は座標 0 を渡しても LOAD を増やさない。保存境界に入った時点では再帰せずローカル PARAM の LOAD を返すので、reduction 出力の融合禁止も守れる。

この入力の途中経過を並べる。

```text
k=0: x=1,w=2, term=3, acc=3
k=1: x=2,w=3, term=7, acc=10
k=2: x=3,w=4, term=13, acc=23
k=3: x=4,w=5, term=21, acc=44
intermediate arrays=0; final output len=1
```


組込み用の抜粋である。上の式の renderer が出す逐次 C は次の形になる。端末の加算順を変えず、clang の fast-math と積和縮約を無効にして比較する。

融合した reduction の renderer が出す逐次 C である。

```c
// src/renderer/c.rs が生成するコード。生成先: <一時ディレクトリ>/kernel.c
float acc = 0.0f;
for (int32_t k = 0; k < 4; k++) {
	float product = x[k] * w[k];
	float term = product + 1.0f;
	acc = acc + term;
}
out[0] = acc;
```

- [ ] `sum(x*w+1)` の reduction 座標を要素演算へ伝える。
- [ ] reduction の外側出力と内側 k のスコープを区別する。
- [ ] LOAD、MUL、ADD、accumulator 更新を同じ body に作る。
- [ ] `x=[1,2,3,4]`, `w=[2,3,4,5]` の結果を `44` にする。
- [ ] 入力側の一時配列がなく、一つの reduction kernel になることを確認する。

逐次 reduction の演算順は維持する。
reduction の結果を expand して再利用する式は、保守的に保存境界を置く。

### コミット 3. STAGE と複数消費先を実際に分割する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── fusion_plan.rs  # [変更] STAGE と fanout の実行入力と dump。
├── src/
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rangeify.rs  # [変更] 境界の値をローカル PARAM の LOAD に変える。
│   ├── schedule.rs  # [変更] materialize、close_body と境界ごとの StorageId。
│   └── uop.rs  # [変更] STAGE と保存境界の op。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

STAGE の構築口を src/uop.rs に追加し、src/schedule.rs に materialize と close_body の抜粋を置く。materialization 表、閉じた Graph と元 Graph を分ける。rangeify.rs は consumer の保存境界を新しいローカル PARAM から読み、examples/fusion_plan.rs で producer と各 consumer の呼出しを確認する。


保存境界を明示する STAGE と、複数の消費先に渡す値を実際の kernel へ分ける。候補ファイル `src/schedule.rs` の設計案 `materialize(node)` は、対象ノード一つに一つの出力 StorageId を割り当て、その値を作る producer kernel を生成する。`src/rangeify.rs` の consumer 側では境界に到達したら再帰を止め、その producer 完了後の state を受け取る buffer PARAM から LOAD する。STAGE を取り除いて式だけをつなぐ処理にはしない。

`STAGE(x+1)*2` なら producer が t=[2,3,4] を書いて END し、AFTER(t,producer) を consumer の入力へ渡す。consumer はその t を読んで `[4,6,8]` を書く。fanout の `t=x+1; y=t*2; z=t+3` でも t を一度保存し、その一つの state を y と z の二つの署名へ対応させる。出力は y=[4,6,8]、z=[5,6,7]、kernel は t、y、z の三つになる。y と z の実行順は互いに依存しないが、両方とも t の後に置く。

局所テストでは STAGE の二 kernel、fanout の三 kernel と t の allocation 一つを dump から数える。t を consumer ごとに二回割り当てたり、同じ StorageId の別 allocation として扱ったりしない。y と z の出力は別の StorageId にし、同じ長さでも所有物を共有しない。値は従来の非融合経路と比較し、producer の計算回数も一回であることを確認する。入力 x は読取り専用のままにする。この章の fission は、保存する値を選ぶ表を変え、閉じた kernel を生成することで実現し、元の純粋グラフを書き換えない。


組込み用の抜粋である。`src/schedule.rs` の分割では、保存境界を kernel の PARAM へ置き換えた新しい body を作る。`inputs` はこの root 以外の保存境界と外部入力を探索して集め、番号順に並べた NodeId 列である。`Graph::intern(op,&src)` は純粋ノードを共有し、`g.kind(id)` は dtype と shape を含む値の kind を返す。

元グラフからローカル PARAM を持つ閉じた body をコピーする関数を追加する。

```rust
// src/schedule.rs（既存ファイルへ組み込む抜粋）
fn close_body(g: &Graph, root: NodeId, owner: FunctionId, inputs: &[NodeId]) -> (Graph,NodeId) {
	fn copy(g: &Graph, body: &mut Graph, id: NodeId, params: &std::collections::HashMap<NodeId,NodeId>, memo: &mut std::collections::HashMap<NodeId,NodeId>) -> NodeId {
		if let Some(&p) = params.get(&id) { return p; }
		if let Some(&n) = memo.get(&id) { return n; }
		let old = &g.nodes[id.0];
		let src: Vec<_> = old.src.iter().map(|&s| copy(g,body,s,params,memo)).collect();
		let new = body.intern(old.op.clone(),&src);
		memo.insert(id,new);
		new
	}
	let mut body = Graph::default();
	let mut params = std::collections::HashMap::new();
	for (slot,&input) in inputs.iter().enumerate() {
		let p = body.intern(Op::Param { owner,slot,kind:g.kind(input) },&[]);
		params.insert(input,p);
	}
	let producer = if matches!(g.nodes[root.0].op,Op::Stage) { g.nodes[root.0].src[0] } else { root };
	let local_root = copy(g,&mut body,producer,&params,&mut std::collections::HashMap::new());
	(body,local_root)
}
```

探索は input の境界で止まり、その先を body へコピーしない。STAGE 自体が producer root の場合だけ入力の計算を body にし、consumer では STAGE の NodeId を PARAM にする。`inputs` と同じ順で CALL 引数を並べれば、ローカル slot と元の AFTER state が一対一になる。materialization 表は境界ごとに一つの StorageId を発行し、閉じた body をその出力へ STORE する kernel に lower する。body 内の RANGE は lower のたびに新しく作り、元 graph の軸を持ち込まない。

この入力の途中経過を並べる。

```text
STAGE(x+1)=t: producer body PARAM0=x; STORE t[i],PARAM0[i]+1
consumer: PARAM0=AFTER(t,producer); STORE out[i],PARAM0[i]*2
fanout: t storage allocated once -> y PARAM0 and z PARAM0 both bind its completed state
CALL order: t before y; t before z; outputs y=[4,6,8],z=[5,6,7]
```

- [ ] 明示 STAGE を非融合境界として扱う。
- [ ] producer の出力を ALLOC、STORE、AFTER に変える。
- [ ] consumer が新しい PARAM から保存済みの値を読むようにする。
- [ ] `(STAGE(x+1))*2` を二 kernel で `[4,6,8]` にする。
- [ ] `t=x+1; y=t*2; z=t+3` を三 kernel で実行する。

最後の予定結果は `y=[4,6,8]`, `z=[5,6,7]`。
共有 producer t は一度だけ計算し、二つの consumer に明示的に渡す。
分割の際に同じ storage ID の別々の ALLOC を作らない。

### コミット 4. 閉じた kernel graph と Plan を動かす

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_09_plan_bindings.rs  # [追加] 依存順と毎回の束縛をホスト関数で確かめる std の独立例。
│   └── fusion_plan.rs  # [変更] 実 library を一度作り、二回の入力で Plan を実行する。
├── src/
│   ├── runtime/
│   │   ├── cpu.rs  # [変更] 既存の compiled library を Plan の同期呼出しへ接続する。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── schedule.rs  # [変更] KernelSpec、Plan、依存順と run ごとの束縛表。
│   └── verify.rs  # [変更] kernel の署名と CALL 引数の検証。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

KernelSpec と実際の Plan は src/schedule.rs に追加し、runtime/cpu.rs の compiled library 所有物を kernel ごとに保持する。掲載した std の Plan と main は examples/book_09_plan_bindings.rs に全体を置く。本体へは依存順と呼出しごとの束縛の規則を移し、ホスト関数の entry を実 kernel の代用にしない。fusion_plan.rs は compile と二回の run を分けて 44 と 14 を比較する。


example 内の手動実行列を、入力を差し替えて再実行できる Plan にする。候補ファイル `src/schedule.rs` の設計案 `KernelSpec` は body、署名、CALL 引数、出力 state を持ち、`Plan` は compiled library 群、kernel の依存順、allocation の記述子を保持する。まず body が参照する外部 storage と scalar を収集し、決定的な順で局所 PARAM に置換する。CALL の引数列はその順に元の buffer state と scalar を並べ、所属しない自由入力を残さない。

producer の出力 state を読む consumer へ辺を張り、入次数 0 の kernel から取り出すトポロジカルソートで実行順を作る。全 kernel を取り出せなければ循環である。compile は Plan 構築時に各固有 kernel を一度行い、run は全入力の署名を検証してから今回の入力束縛と新規 allocation を作り、順番に同期呼出しする。入力値はコンパイルキーや元ノードへ埋め込まない。

一 kernel に融合した `sum(x*w+1)` で x=[1,2,3,4] を固定すると、一回目 w=[2,3,4,5] は 44、二回目 w=[1,1,1,1] は加算項 2、3、4、5 の和で 14 になる。局所テストではこの二結果、compile 計数 1、run 計数 2 を同時に確認する。run の出力所有物を返した後も、二回目の新規領域と別に保持し、一回目の 44 が変わらないことを見る。Plan は compiled library を所有し、各 run の実体化表は呼出し内へ限定する。前の入力で保存した値を NodeId にひも付けて次回へ流用しない。


単独で実行できる例である。`src/schedule.rs` の Plan が行う依存順と毎回の束縛を、ホスト関数を kernel 入口にした std の例で検証する。これは動的 library のロードを検証する例ではなく、保持済みの入口に対する Plan の意味を確認する。実装では `entry` を 02 の compiled library 所有物の `run` に置き換える。

ホスト関数を入口にした std の Plan 例全体を別ファイルに保存する。

```rust
// examples/book_09_plan_bindings.rs（単独で実行する例）
use std::collections::VecDeque;
struct Kernel { inputs: Vec<usize>, output: usize, entry: fn(&[&[f32]]) -> Vec<f32> }
struct Plan { lengths: Vec<usize>, input_slots: Vec<usize>, kernels: Vec<Kernel>, order: Vec<usize> }
impl Plan {
	fn new(lengths: Vec<usize>, input_slots: Vec<usize>, kernels: Vec<Kernel>) -> Result<Self,String> {
		let mut input_seen = std::collections::HashSet::new();
		for &slot in &input_slots {
			if slot >= lengths.len() || !input_seen.insert(slot) { return Err("invalid or duplicate external input slot".into()); }
		}
		let mut producer = vec![None;lengths.len()];
		for (k,node) in kernels.iter().enumerate() {
			if node.output >= lengths.len() || input_slots.contains(&node.output) { return Err("invalid output slot".into()); }
			if producer[node.output].replace(k).is_some() { return Err("multiple producers".into()); }
		}
		let mut degree = vec![0;kernels.len()];
		let mut next = vec![vec![];kernels.len()];
		for (k,node) in kernels.iter().enumerate() {
			for &slot in &node.inputs {
				if slot >= lengths.len() { return Err("invalid input slot".into()); }
				if let Some(p) = producer[slot] { degree[k]+=1; next[p].push(k); }
				else if !input_slots.contains(&slot) { return Err("unbound slot".into()); }
			}
		}
		let mut ready: VecDeque<_> = (0..kernels.len()).filter(|&k| degree[k]==0).collect();
		let mut order = vec![];
		while let Some(k) = ready.pop_front() {
			order.push(k);
			for &consumer in &next[k] { degree[consumer]-=1; if degree[consumer]==0 { ready.push_back(consumer); } }
		}
		if order.len()!=kernels.len() { return Err("kernel dependency cycle".into()); }
		Ok(Self { lengths,input_slots,kernels,order })
	}
	fn run(&self, inputs: &[Vec<f32>]) -> Result<Vec<Option<Vec<f32>>>,String> {
		if inputs.len()!=self.input_slots.len() { return Err("input count mismatch".into()); }
		for (&slot,value) in self.input_slots.iter().zip(inputs) {
			if value.len()!=self.lengths[slot] { return Err("input length mismatch".into()); }
		}
		let mut bindings = vec![None;self.lengths.len()];
		for (&slot,value) in self.input_slots.iter().zip(inputs) { bindings[slot]=Some(value.clone()); }
		for &k in &self.order {
			let node = &self.kernels[k];
			let args: Vec<&[f32]> = node.inputs.iter().map(|&s| bindings[s].as_ref().unwrap().as_slice()).collect();
			let output = (node.entry)(&args);
			if output.len()!=self.lengths[node.output] { return Err("kernel output length mismatch".into()); }
			bindings[node.output]=Some(output);
		}
		Ok(bindings)
	}
}
fn dot_plus_one(args: &[&[f32]]) -> Vec<f32> {
	let mut acc = 0.0;
	for i in 0..args[0].len() { acc += args[0][i]*args[1][i]+1.0; }
	vec![acc]
}
fn main() {
	let plan = Plan::new(vec![4,4,1],vec![0,1],vec![Kernel { inputs:vec![0,1],output:2,entry:dot_plus_one }]).unwrap();
	let first = plan.run(&[vec![1.0,2.0,3.0,4.0],vec![2.0,3.0,4.0,5.0]]).unwrap();
	let second = plan.run(&[vec![1.0,2.0,3.0,4.0],vec![1.0;4]]).unwrap();
	assert_eq!(first[2],Some(vec![44.0]));
	assert_eq!(second[2],Some(vec![14.0]));
	assert_eq!(first[2],Some(vec![44.0]));
	assert!(plan.run(&[vec![1.0],vec![1.0;4]]).is_err());
	let add_one: fn(&[&[f32]]) -> Vec<f32> = |a| a[0].iter().map(|x| x+1.0).collect();
	let times_two: fn(&[&[f32]]) -> Vec<f32> = |a| a[0].iter().map(|x| x*2.0).collect();
	// consumer を先に登録しても、実行順は producer の後になる。
	let chain = Plan::new(vec![3,3,3],vec![0],vec![
		Kernel { inputs:vec![1],output:2,entry:times_two },
		Kernel { inputs:vec![0],output:1,entry:add_one },
	]).unwrap();
	assert_eq!(chain.order,[1,0]);
	assert_eq!(chain.run(&[vec![1.0,2.0,3.0]]).unwrap()[2],Some(vec![4.0,6.0,8.0]));
	assert!(Plan::new(vec![3,3],vec![],vec![
		Kernel { inputs:vec![1],output:0,entry:add_one },
		Kernel { inputs:vec![0],output:1,entry:add_one },
	]).is_err());
}
```

`producer` が slot の書き手を一意にし、consumer ごとに入次数と後続辺を作る。Kahn の走査が全 kernel を取り出せた場合だけ Plan を返す。`run` は最初に全入力長を検査し、その後で今回だけの `bindings` を作る。二回目の実行後にも first の所有する 44 は変わらない。実装の Plan は compiled library を構築時に一度保持し、各 run の出力を今回の allocation に bind する。実 library を使う example では compile=1、run=2 の計数も追加で照合する。

この入力の途中経過を並べる。

```text
Plan created once: input slots=[0,1], output slot=2, order=[0]
run1 bindings: x,w1,new out -> output44
run2 bindings: x,w2,new out -> output14
first output remains44; graph remains unchanged
input length mismatch -> no entry call
```

- [ ] 各 body の外部 storage と scalar を PARAM に正規化する。
- [ ] kernel の署名、CALL 引数、出力 state を照合する verifier を追加する。
- [ ] producer と consumer の辺から実行順を作り、循環なら拒否する。
- [ ] 既存の CPU compile と run を Plan の kernel 列へ接続する。
- [ ] 同じ Plan に新しい重みを渡して、二回目の出力を確認する。

予定例は `sum(x*w+1)` で x を固定し、二回目の w を `[1,1,1,1]` にする。
出力は初回 `44`、二回目 `14`、compile の予定計数は一回になる。
run ごとの allocation は独立させ、前回の出力所有物も壊さない。

### コミット 5. 分割の漏れと古い実体化の再利用を検出する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── src/
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rangeify.rs  # [変更] kernel ごとに分けた座標と束縛環境。
│   ├── schedule.rs  # [変更] Plan 全体の入力検査と run 単位の実体化表。
│   └── verify.rs  # [変更] verify_closed_kernel と verify_kernel_graph。
├── tests/
│   └── fusion_plan.rs  # [追加] 境界違反、異なる座標、再実行と元グラフ保持の統合確認。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

verify_closed_kernel の抜粋は src/verify.rs に追加し、verify_kernel_graph の各 body 検査から呼ぶ。既存の verify_regions で軸の所属と閉じ忘れも検査する。src/schedule.rs は全入力の署名を最初の入口呼出し前に検証し、実体化表を run 内で作る。tests/fusion_plan.rs は実 CPU 経路で二回の出力と以前の出力の保持を照合する。


最後に、分割が作った関数境界を破る fixture を用意し、誤った Plan が実行へ進まないことを確かめる。候補ファイル `src/verify.rs` の設計案 `verify_kernel_graph` は、各 body をその署名の PARAM からだけ閉じたものとして検査する。body の buffer 参照はローカル PARAM かローカル REG とし、外側の BUFFER NodeId を直接参照していれば拒否する。STORE 先が署名の出力に含まれることと、CALL の引数の dtype、長さ、可変性、state の順序も照合する。

RANGE については各 body の region をたどり、他 kernel の軸を使う INDEX や未終了軸を拒否する。座標 fixture は同じ producer を一方で i、他方で j から lower し、異なる入力値も束縛する。例えば xA=[1,2,3] と xB=[4,5,6] の t=x+1 を使えば、結果 `[2,3,4]` と `[5,6,7]` は別になるため、NodeId だけの memo による入力と軸の混入を数値で検出できる。lower の環境と memo は kernel 単位へ限定し、座標を省いた共有を作らない。

局所テストでは不正な body だけを verifier に渡し、library 呼出しが 0 のまま失敗することを見る。通常の Plan では run の前後の元グラフ dump を比較し、materialization 表が別に管理されていることを確認する。入力の長さや dtype の検証は最初の kernel を起動する前に Plan 全体で済ませ、途中まで書いてから入力不一致に気付く流れを避ける。rebind の 44 と 14、以前の出力 44 の保持も再確認する。値が同じになる入力を選ばず、古い実体化を再利用すると必ず差が出る条件で契約を固定する。


組込み用の抜粋である。`src/verify.rs` で、閉じた body の所属と署名を実行前に照合する。`KernelSpec` の signature は各 slot の kind と writable を持ち、`Op::Store { output_slot }` はこの抜粋では lowering 済み STORE の保存先 PARAM slot を表す。REG STORE は別 variant `StoreReg` として既存の region 検証に渡す。

閉じた body と署名の検査を追加する。

```rust
// src/verify.rs（既存ファイルへ組み込む抜粋）
fn verify_closed_kernel(g: &Graph, spec: &KernelSpec) -> Result<(),String> {
	for id in linearize(g,spec.root) {
		let node = &g.nodes[id.0];
		match &node.op {
			Op::Param { owner,slot,kind } => {
				if *owner != spec.owner { return Err("foreign PARAM".into()); }
				if spec.signature.get(*slot).map(|p| &p.kind) != Some(kind) { return Err("PARAM signature mismatch".into()); }
			}
			Op::Buffer { .. } => return Err("free external BUFFER in kernel body".into()),
			Op::Range { owner,.. } if *owner != spec.owner => return Err("free external RANGE".into()),
			Op::Store { output_slot } => {
				if !spec.signature.get(*output_slot).map(|p| p.writable).unwrap_or(false) {
					return Err("STORE outside declared outputs".into());
				}
			}
			_ => {}
		}
	}
	if spec.arguments.len()!=spec.signature.len() { return Err("CALL signature length mismatch".into()); }
	for (argument,param) in spec.arguments.iter().zip(&spec.signature) {
		if argument.kind!=param.kind || (param.writable && !argument.writable) {
			return Err("CALL argument kind or mutability mismatch".into());
		}
	}
	verify_regions(&spec.scope_instructions)
}
```

scope の列は 06 の `verify_regions` と同じ検証を通し、別 kernel の軸と閉じ忘れを拒否する。通常の INDEX/STORE 表現では 05 の `storage_base` をたどって保存先 PARAM を求め、`output_slot` の検査と同じ判定を行う。CALL の引数には producer 完了後の state を渡し、09 の依存辺がその順序を保証する。run ごとの入力署名検査はこの静的検証に追加して全 kernel の起動前に行い、元 graph の NodeId に実体化値を保存しない。

この入力の途中経過を並べる。

```text
body PARAM(owner=A,slot0), kernel owner=B -> foreign PARAM
body references outer BUFFER -> free external BUFFER
STORE slot0 where signature[0].writable=false -> reject
RANGE from kernelA used in kernelB -> free external RANGE
normal rebind: output44 -> new weight binding -> output14
```

- [ ] body の未束縛 PARAM、自由 RANGE、STORE 先の漏れを拒否する。
- [ ] 外側 kernel の NodeId が body に直接残った例を拒否する。
- [ ] 二つの違う座標で同じ producer を使う fixture を実行する。
- [ ] 元グラフを run の前後で dump し、構造が変わらないことを確認する。
- [ ] 入力の長さと dtype が署名と違う場合、実行開始前に拒否する。

Plan の再利用テストは、出力値が変わる重みを選ぶ。
結果がたまたま同じになる入力では、古い実体化の誤利用を検出できない。

## 完了確認

以下はこの章で新しく作る example とテストの予定コマンドである。

```sh
# 作業ディレクトリ: chibigrad/
cargo test
cargo run --example fusion_plan -- --case chain --dump-schedule
cargo run --example fusion_plan -- --case stage --dump-kernel
cargo run --example fusion_plan -- --case fanout --dump-schedule
cargo run --example fusion_plan -- --case rebind --dump-graph
```

- [ ] chain は一 kernel、STAGE は二 kernel、fanout は三 kernel になる。
- [ ] 非融合と融合の値が `[4,6,8]` で一致する。
- [ ] fanout の二出力が `[4,6,8]` と `[5,6,7]` になる。
- [ ] 同じ Plan の再実行で `44` から `14` へ変わり、再コンパイルしない。
- [ ] 閉じた body に未束縛の storage と RANGE が残らない。
- [ ] dump で各 consumer が producer 完了後の state を読む。
- [ ] 純粋な元グラフが run 前後で同じ構造を保つ。

## tinygrad ソースとの対応

参照 commit は `c22edff14c79310a9b2df62a78819cf77c8289fa`。
リンクの行番号は checkout で読んで確認した。

| 実在するシンボル | ソース | 役割 | chibigrad で簡略化する点 |
|---|---|---|---|
| `remove_bufferize` | [rangeify.py](../../tinygrad/tinygrad/schedule/rangeify.py#L48) | 保存境界の削除可否と座標置換 | 単一消費先だけを融合する |
| `bufferize_to_store` | [rangeify.py](../../tinygrad/tinygrad/schedule/rangeify.py#L205) | 境界を allocation と STORE にする | GLOBAL の新規出力だけにする |
| `split_store` | [rangeify.py](../../tinygrad/tinygrad/schedule/rangeify.py#L323) | 閉じた効果を kernel CALL にする | 全外部入力を明示 PARAM 化する |
| `get_kernel_graph` | [rangeify.py](../../tinygrad/tinygrad/schedule/rangeify.py#L341) | rangeify、境界処理、kernel 分割 | 保守的な規則だけを順に適用する |
| `create_schedule` | [schedule](../../tinygrad/tinygrad/schedule/__init__.py#L31) | buffer state の依存から順序を作る | 破壊的更新なしの DAG 順だけを扱う |
| `spec_kernel_graph` | [spec.py](../../tinygrad/tinygrad/uop/spec.py#L252) | kernel graph の許可ノードを検証 | 自由軸と署名の一致を必須にする |

## 今は追加しないもの

多次元 shape と movement は [10-movement.md](10-movement.md) で加える。
保存と再計算を選ぶ高度な cost model、buffer 再利用は第 19 段階へ送る。
破壊的更新の読み書き hazard、複数 device、非同期実行は今は加えない。
