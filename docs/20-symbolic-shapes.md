# 20. 上限付きの shape を実行時に束縛して同じ kernel を動かす

## このステップで動かすもの

長さ `N` を実行時入力にし、一つの Plan で `N=1,3,17` を順に計算する。
`y=x*2+1` と `sum(x)` を、同じ値の静的 shape 版と比較する。
[19](19-fast-gemm-conv.md)までの検証と cache を前提にする。
元計画の「初版では実装しない symbolic shape」を後半の追加課題として実装する。
初期の静的 f32 の経路を残し、全ての shape を一度に symbolic にしない。
本文の型、API、example、test とコマンドは今後実装する予定である。

```text
N∈[1,17], x の物理容量は17要素
bind(N=3), x=[1,2,3,...] -> y の論理値=[3,5,7], sum=6
bind(N=1), x=[4,...]     -> y の論理値=[9],     sum=4
bind(N=18)              -> 起動前に BoundError
```

## 最小実装とデータ

20aでは shape の各次元を `Dim::Static(usize)` または `Dim::Var(VarId)` とする予定である。
`VarSpec { id, name, min, max }` はグラフ側、`Bindings` は一回の実行側に置く。
名前だけで変数を同一視せず、異なるスコープの `N` に別の VarId を与える。
入力 IR は変数を含む shape、出力 IR は同じ変数を上限とする RANGE と scalar PARAM を持つ。
最初の式は変数、非負定数、加算、正の定数倍、正の定数での除算と剰余に制限する。
整数演算の幅と overflow を検査し、負の shape とゼロ除算を受け付けない。
必要なバッファ容量は上限 shape から決め、論理 shape は束縛値から決める。
可変次元同士の積を一般の記号代数として解くことは要求しない。

実行方式は二つを別に扱う。

| 方式 | kernel と cache | 起動時の検査 |
|---|---|---|
| generic | `N` を scalar 引数にして範囲内で再利用 | 上下限、容量、shape 関係を検査 |
| specialized | `N=16` などを定数にして最適化 | 特化条件が成立する時だけ選択 |

## この章のファイル配置

shapeの定義と式はsrc/shape.rs、一回のBindingsから引数とlaunchを作る処理は各runtimeへ置く。単独の境界確認をexamplesへ分け、元のTensor graphへ今回値を書き戻さない。

次はこの章の終わりまでに触る関連ファイルの配置案である。現在のcheckoutの一覧ではない。各コミットでは、その下に示す部分だけを追加・変更する。

```text
chibigrad/
├── examples/
│   ├── book_20_bindings.rs  # 本文のコードを配置する。
│   ├── symbolic_dispatch.rs  # 16、17、3の選択と全要素比較。
│   ├── symbolic_movement.rs  # 可変reshapeと行和のdriver。
│   └── symbolic_vector.rs  # Metal、OpenCLの長さとlocal指定。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # 実行時のRANGE上限を描く。
│   │   ├── metal.rs  # 動的NのPARAM、LOAD guard、同期。
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── opencl.rs  # sum_groupsの生成。
│   ├── runtime/
│   │   ├── cpu.rs  # genericと特化条件を含むcache key。
│   │   ├── metal.rs  # snapshotからscalar引数とprepare_launchを作る。
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── opencl.rs  # 同じsnapshotの引数設定とNDRange。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rangeify.rs  # ValueAtKeyによる一回のlowering内のmemo。
│   ├── schedule.rs  # DispatchVariantとselect_variant。
│   └── shape.rs  # canonical_varsを使うshape署名。
├── tests/
│   └── symbolic_shapes.rs  # 署名と座標、scope、stateの分離。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

生成Cやshaderは一時ディレクトリへ出力するため、このRustプロジェクトのツリーには含めない。コード冒頭のコメントで、生成元と生成先を確認する。

## 小さなコミットで進める

### 20a. 単一次元を束縛して CPU の加算を動かす

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_20_bindings.rs  # 本文のコードを配置する。
│   └── symbolic_vector.rs  # [追加] 同じPlanで長さを替えるdriver。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # [更新] 実行時のRANGE上限を描く。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── runtime/
│   │   ├── cpu.rs  # [更新] validate_bindingsと実行snapshotのscalar化。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rangeify.rs  # [更新] 可変次元をscalar PARAMへ変換する。
│   ├── schedule.rs  # [更新] VarIdとScalarSlot、容量と論理長の記録。
│   └── shape.rs  # [更新] Dim、VarId、VarSpec、Bindingsの定義。
├── tests/
│   └── symbolic_shapes.rs  # [追加] 未束縛、重複、範囲外と再実行の確認。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

掲載validateとmainはexamples/book_20_bindings.rsの独立した確認例へ置く。本体ではVarId、VarSpec、Bindingsをsrc/shape.rsに追加し、検査をsrc/runtime/cpu.rsのvalidate_bindingsへ接続する。rangeifyはPlanのVarIdからscalar slotを引き、C rendererはそのPARAMをRANGE上限に使う。既存shape型は拡張し、別のTensor型を作らない。


入力は許可した VarSpec と今回の指定、出力は検査後の変更不能な Bindings である。`src/shape.rs` の検査部分を単独で実行できる例にする。実プロジェクトでは検査済みの `snapshot` からだけ `scalars[slot]` を作る。

validateとmainを含むhost境界の独立例。本体のshape型とruntime検査へ丸ごと重ねて貼らない。

```rust
// examples/book_20_bindings.rs（単独で実行する例）
use std::collections::BTreeMap;
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct VarId(u32);
struct VarSpec { id: VarId, min: u64, max: u64 }
struct Bindings(BTreeMap<VarId,u64>);
fn validate(specs: &[VarSpec], pairs: &[(VarId,u64)]) -> Result<Bindings,String> {
	if specs.iter().any(|s| s.min>s.max || s.max>i32::MAX as u64) {
		return Err("shape bounds exceed the kernel i32 domain".into());
	}
	let mut values = BTreeMap::new();
	for &(id,value) in pairs {
		let spec = specs.iter().find(|s|s.id==id).ok_or("unknown variable")?;
		if value<spec.min || value>spec.max { return Err("bounds".into()); }
		if values.insert(id,value).is_some() { return Err("duplicate".into()); }
	}
	for spec in specs {
		if !values.contains_key(&spec.id) { return Err("unbound".into()); }
	}
	Ok(Bindings(values))
}
fn main() {
	let n = VarId(7);
	let specs = [VarSpec{id:n,min:1,max:17}];
	let snapshot = validate(&specs,&[(n,3)]).unwrap();
	let logical = snapshot.0[&n] as usize;
	let x = [1.,2.,3.,f32::NAN];
	let y: Vec<_> = x[..logical].iter().map(|v|v*2.+1.).collect();
	assert_eq!(y,[3.,5.,7.]);
	assert!(validate(&specs,&[(n,18)]).is_err());
	assert!(validate(&specs,&[(n,3),(n,3)]).is_err());
	assert!(validate(&specs,&[]).is_err());
	assert!(validate(&[VarSpec{id:n,min:0,max:u64::MAX}],&[(n,3)]).is_err());
	println!("{y:?}");
}
```

`VarId(7)` は名前Nとは別の識別子である。`values.insert` が既存値を返したら二重指定として候補全体を捨てる。`logical=3` なので NaN の要素を参照しない。例の `map` は host の境界検査用であり、実際の Tensor 計算の置換ではない。以下は二配列 ABI を保った生成 C の本体で、scalar 型とslot幅は02のABI定義に合わせる。

scalar Nを受け取るC rendererの出力。共有ライブラリへcompileする対象でありRustではない。

```c
// src/renderer/c.rs が生成するコード。生成先: <一時ディレクトリ>/symbolic_vector.c
#include <stdint.h>
/* 組込み用。buffers[0]=x, buffers[1]=y, scalars[0]=検査済みN。 */
void chibi_entry(void *const *buffers, const uint64_t *scalars) {
	const float *x = (const float *)buffers[0];
	float *y = (float *)buffers[1];
	int32_t n = (int32_t)(uint32_t)scalars[0];
	for (int32_t i=0; i<n; ++i) y[i]=2.0f*x[i]+1.0f;
}
/* n=3 -> y[0..3]=[3,5,7], capacity=17, graph上のNはVarのまま */
```

shape値は非負のi32範囲に制限する。ホストはNをu64 wordの下位32bitへ入れ、上位32bitを0にする。Cはその下位bitsを取り出す。Nが `i32::MAX` 以下という起動前検査があるため、最後のint32_t変換は表現可能である。F32のbits復元とは別の整数slotであり、scalar wordにunionのメンバは持たせない。

ここまでは Plan が静的な長さを持つ。今回は `src/shape.rs` に VarId、VarSpec、Bindings、`src/runtime/cpu.rs` の実行準備に `validate_bindings` を置く案とし、変数の定義と一回の値を分離する。VarSpec は N の許される範囲[1,17]を保持し、Bindings は VarId に3などの整数を対応させる。検査関数は必要な変数集合と利用者の指定を照合し、欠落、二重指定、未知変数、範囲外を報告した後、変更不能な実行用 snapshot を返す。検査中に入力が変わって kernel と容量検査が別の N を見ることを防ぐ。

次に rangeify が長さを host の定数へ落とす箇所を、変数なら scalar PARAM を参照する処理へ変える。Plan の `VarId -> ScalarSlot` 表を一度決め、二配列 ABI の scalars の整数 slot に N を格納する。buffer は従来どおり buffers に渡し、buffer ポインタへ長さを埋め込まない。生成 C のループは0から scalars の N までを回す。確保容量17と論理長Nは別の欄に保持し、戻り値の Tensor は論理長の範囲だけを公開する。

N=3で i=0、1、2が x の1、2、3を読み、3、5、7を保存する。次に同じ Plan を N=1、x の先頭4で起動すると i=0だけが9を保存する。前回の出力の2、3要素目が物理領域に残っても、新しい Tensor の値として読めてはいけない。N=18と未束縛では C 関数の呼出し回数が増えないことを test にする。dump で VarId、slot、今回値、容量を並べ、元 graph 内の N が実行後も Var のままであることを確認する。静的版と同じ compiler と runtime を通し、shape 仕様だけを加えた実行として比較する。

- [ ] VarSpec と Bindings を追加し、未束縛、二重指定、範囲外を起動前に拒否する。
- [ ] 02と03の固定 ABI の scalar slot を使い、VarId から slot への対応を追加する。
- [ ] `RANGE(N)` を C の実行時ループ上限へ出す。
- [ ] 容量17の buffer に N=1、3、17を渡し、毎回異なる論理長の結果を返す。
- [ ] 純粋な元グラフを置換せず、束縛値を runtime の実行記録に置く。

予定例 `symbolic_vector --backend cpu --length 3` は `[3,5,7]` を返す。
同じ Plan に N=1 を渡した二回目に、前回の長さ3を保持しないことを確認する。
このコミットでも graph、rangeify、C renderer、共有ライブラリ、固定 ABI を通す。

### 20b. 縮約と movement の制約を検証する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── symbolic_movement.rs  # [追加] 可変reshapeと行和のdriver。
├── src/
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rangeify.rs  # [更新] movementの制約とLOAD guardの生成。
│   ├── schedule.rs  # [更新] 起動前に評価するshape制約。
│   └── shape.rs  # [更新] DimExprとcheckedなeval_dim。
├── tests/
│   └── symbolic_shapes.rs  # [更新] reshape_rowsによる行和reference。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/shape.rsへDimExprとeval_dimを追加し、式の参照IDはTensorのNodeIdと区別する。reshape_rowsはtests/symbolic_shapes.rsの比較helperへ置く。実際のreshapeとsumは既存Tensor graphからrangeifyへ流し、要素数の等式をPlanの実行前検査に加える。


入力は `DimExpr` と20aの Bindings、出力は overflow を検査した非負整数である。`src/shape.rs` へ組み込む抜粋を示す。ここでは読みやすさのため Box で表し、intern版では Box の位置を shape専用 `DimExprId` に替える。TensorのNodeIdとは型を共有しない。

DimExprとeval_dimをshapeへ加え、reshape_rowsはintegration testのreference helperへ分ける。

```rust
// src/shape.rs（既存ファイルへ組み込む抜粋）
enum DimExpr {
	Const(u64), Var(VarId), Add(Box<DimExpr>,Box<DimExpr>),
	MulConst(Box<DimExpr>,u64), DivConst(Box<DimExpr>,u64),
	ModConst(Box<DimExpr>,u64),
}
fn eval_dim(e: &DimExpr, b: &Bindings) -> Option<u64> {
	match e {
		DimExpr::Const(v) => Some(*v),
		DimExpr::Var(id) => b.0.get(id).copied(),
		DimExpr::Add(x,y) => eval_dim(x,b)?.checked_add(eval_dim(y,b)?),
		DimExpr::MulConst(x,c) => eval_dim(x,b)?.checked_mul(*c),
		DimExpr::DivConst(x,c) => eval_dim(x,b)?.checked_div(*c),
		DimExpr::ModConst(x,c) => eval_dim(x,b)?.checked_rem(*c),
	}
}
// tests/symbolic_shapes.rs（既存ファイルへ組み込む抜粋）
fn reshape_rows(input: &[f32], n: u64) -> Result<Vec<f32>, &'static str> {
	let needed = n.checked_mul(2).ok_or("size overflow")?;
	if u64::try_from(input.len()).ok()!=Some(needed) { return Err("reshape"); }
	let mut rows = Vec::new();
	for row in input.chunks_exact(2) { rows.push(row[0]+row[1]); }
	Ok(rows)
}
```

`eval_dim` は変数値を graphへ書き戻さない。`checked_div(0)` と未束縛は `None` になる。`reshape_rows` は `2*N` と入力長が等しい場合だけ読み始める。一般のmovementでも先に制約を評価し、その後に座標を LOAD へ渡す。padの範囲外では `if valid { load } else { 0 }` の順を守る。

```text
DimExpr=MulConst(Var(7),2), Bindings={7:3} -> 6
input=[1,2,3,4,5,6], reshape=[3,2], rows=[3,7,11]
input.len=5, needed=6 -> Err("reshape"), LOAD count=0
pad-left=1: out i=0 -> source=-1 -> constant0, LOAD count=0
```

加算だけでなく、長さの式と shape 間の関係を実行前に評価する。`src/shape.rs` に式ノードを intern する小さな DimExpr の領域と `eval_dim` を追加する案である。`Dim::Expr(NodeId)` の NodeId は shape 用の整数式を指すと型で区別し、Tensor のデータノードと混同しない。式は Const、Var、Add、定数倍、正の定数による除算と剰余だけから始める。子を先に評価してから checked 演算を使い、値域の上限評価にも同じ overflow 検査を行う。変数同士の積は要素数の実行時検査で計算できても、一般の式簡約として導入する必要はない。

movement の関数は変換後の shape に加え、実行時に必要な等式と guard を Plan へ返す。reshape([N,2])なら入力の論理要素数と `2*N` を比較し、broadcast なら各軸が1または同値という条件を検査する。同名の別 VarId を一つに置換せず、その回の値を比較する。shrink は開始と終了が入力の論理範囲内かを確認し、pad は出力座標から pad 幅を引いた入力座標が有効かを INDEX の条件にする。LOAD してから mask を掛ける順にはしない。

入力[1,2,3,4,5,6]と N=3なら式2*Nは6となり、reshape の `(r,c)` は平坦位置 `r*2+c` を読む。行和は1+2、3+4、5+6から[3,7,11]となる。入力長5では6との不一致を実行準備で返し、縮約 acc も buffer も起動しない。sum(x) の acc は各実行で0に戻し、N=3の6を次の N=1へ持ち越さない。テストでは式の値、制約の左右、guard の順を dump し、静的 movement と全出力を比べる。一般の代数ソルバを用意する前に、許可した式の検査と既存の座標変換が一緒に動くところまで完成させる。

- [ ] `Dim::Expr(NodeId)` を必要な次元式だけに追加し、`2*N` を式のまま運ぶ。
- [ ] `sum(x)` の縮約 RANGE に N を使い、初期値を実行ごとに設定する。
- [ ] `reshape([N,2])` の必要要素数と入力の論理要素数を実行時に検証する。
- [ ] 同じ変数を使った broadcast は許可し、別変数の等値は Bindings で確認する。
- [ ] 動的 shrink と pad の有効範囲を guard として INDEX に渡す。
- [ ] 確定できないアクセスを安全とみなさず、guard を生成するか未対応として拒否する。

予定例 `symbolic_movement --rows 3` は `[1,2,3,4,5,6]` を `[3,2]` に reshape して行和 `[3,7,11]` を返す。
要素数5で同じ reshape を要求した場合は、LOAD の前に shape エラーを返す。
`N=3` の転置、pad、shrink を静的版と比較する。
未束縛の式を host の usize に暗黙変換しない。

### 20c. GPU の端数を束縛値で守る

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── symbolic_vector.rs  # [更新] Metal、OpenCLの長さとlocal指定。
├── src/
│   ├── renderer/
│   │   ├── metal.rs  # [更新] 動的NのPARAM、LOAD guard、同期。
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── opencl.rs  # [更新] sum_groupsの生成。
│   ├── runtime/
│   │   ├── metal.rs  # [更新] snapshotからscalar引数とprepare_launchを作る。
│   │   ├── mod.rs  # 子モジュールを宣言する。
│   │   └── opencl.rs  # [更新] 同じsnapshotの引数設定とNDRange。
│   └── lib.rs  # 本体のモジュール宣言と公開APIの入口。
├── tests/
│   └── symbolic_shapes.rs  # [更新] 無効領域NaNと末尾guardの確認。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

各runtimeのprepare_launchへ検証済みBindingsを渡し、同じNからscalar引数とgroup数を作る。各rendererのPARAM描画に整数Nを加え、reductionのLOADだけをguardする。掲載sum_groupsはsrc/renderer/opencl.rsの生成出力であり、partialの確保と第二kernelの接続は既存Planで行う。


入力は同じ snapshot のNと容量、出力は端数laneを含むgroup reductionである。以下は renderer が出す OpenCL kernel の組込み用の抜粋。local size=8、`partial` はceil(N/8)要素をhost側で確保する。group間の総和は18の第二kernelで行う。実機未確認である。

動的長さのgroup reductionを描いたOpenCL C。Metal版はmetal rendererで同じIRから生成する。

```c
// src/renderer/opencl.rs が生成するコード。生成先: <一時ディレクトリ>/sum_groups.cl
__kernel void sum_groups(__global const float *x,
                         __global float *partial, uint N) {
	uint lane=get_local_id(0);
	uint i=get_group_id(0)*8+lane;
	__local float scratch[8];
	float value=0.0f;
	if (i<N) value=x[i];
	scratch[lane]=value;
	barrier(CLK_LOCAL_MEM_FENCE);
	for (uint offset=4; offset>0; offset/=2) {
		if (lane<offset) scratch[lane]+=scratch[lane+offset];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (lane==0) partial[get_group_id(0)]=scratch[0];
}
```

`i<N` はLOADだけを囲む。`scratch[lane]` と四回のbarrierは全laneが通る。`N=3` ならscratchは[1,2,3,0,0,0,0,0]であり、容量内のNaNを読まない。hostは `N<=u32::MAX`、容量、local資源を検査し、OpenCL globalを `ceil(N/8)*8` とする。Metalにも同じNをbuffer引数領域へ転送する。

```text
snapshot N=17, capacity=17, local=8, groups=3, global=24
last group lanes: i=16(valid),17..23(no LOAD)
snapshot N=3, x=[1,2,3,NaN,NaN,NaN,NaN,NaN] -> partial[0]=6
```

CPU で検査した実行用 Bindings を、そのまま Metal と OpenCL の引数設定と launch 計算の入力にする。変更先は各 runtime の `prepare_launch` と renderer の scalar PARAM 描画とする案である。Plan の論理 ABI は buffers と scalars のまま保持し、Metal は scalar 用の引数領域、OpenCL は対応する scalar kernel 引数へ snapshot の整数を渡す。引数の整数幅と添字演算の幅を一致させ、device 側の型に収まらない値は起動前に拒否する。

長さ17、local=8なら group 数は `ceil(17/8)=3`、実 lane 数は24になる。各 lane の i を `group*8+lane` で作り、i<Nの時だけ x[i] を LOAD して y[i] を STOREする。group2では lane0だけが i=16を担当し、lane1..7は LOAD 自体を実行しない。容量が17ある事実だけでは N=3の時の i=3..16を読んでよいことにならない。通常の要素演算なら条件分岐で保存まで省略できる。

local reduction の場合は全 lane が部分和の保存と barrier に参加する。無効 lane の部分和を0とし、有効 LOAD を条件分岐の内部に置いてから local へ保存する。N=3、x=[1,2,3,NaN,...]、local=8なら部分和は[1,2,3,0,0,0,0,0]、結果は6になる。NaN を読んでから0と選択すると source 上の LOAD guard が欠けるため不合格にする。境界 test は論理末尾以降の NaN と物理末尾外の検査領域を別に設け、値と書込み範囲を確認する。dump には論理長、容量、group、local、実 global size を出す。既存の転送と同期と読出しを再利用し、Metal と OpenCL の実行結果は利用できる device ごとに記録する。

- [ ] Metal の引数領域に N を置き、論理長から group 数を決める。
- [ ] `i<N` を LOAD と STORE の両方へ伝え、容量だけを guard に使わない。
- [ ] local reduction では無効 lane に0を与え、全 lane を barrier に到達させる。
- [ ] OpenCL の scalar 引数と NDRange を同じ束縛値で設定する。
- [ ] GPU に送る値と host で検証した値が同じになるよう、一回の Bindings を固定する。

予定例 `symbolic_vector --backend metal --length 17 --local 8` を動かす。
物理容量17の直後に検査領域を置く test と、論理末尾以降を NaN にした test を追加する。
無効 lane が NaN を LOAD してから0を選ぶ生成コードは不合格にする。
Metal と OpenCL の実機実行は、各 backend を使える環境で別々に確認する。

### 20d. generic と specialized の cache を分ける

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── symbolic_dispatch.rs  # [追加] 16、17、3の選択と全要素比較。
├── src/
│   ├── runtime/
│   │   ├── cpu.rs  # [更新] genericと特化条件を含むcache key。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rangeify.rs  # [更新] ValueAtKeyによる一回のlowering内のmemo。
│   ├── schedule.rs  # [更新] DispatchVariantとselect_variant。
│   └── shape.rs  # [更新] canonical_varsを使うshape署名。
├── tests/
│   └── symbolic_shapes.rs  # [更新] 署名と座標、scope、stateの分離。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

canonical_varsをsrc/shape.rs、select_variantをsrc/schedule.rs、ValueAtKeyをsrc/rangeify.rsへ分ける。ValueAtKeyは08のLowerKeyを拡張する扱いとし、二つのmemo表を重ねて作らない。runtimeのcompile keyへ正規化したshape制約と特化条件を加え、variant選択は通常のBindings検査後だけ行う。


入力はgraph上の変数列と制約、出力は値を含めないgeneric署名と今回のvariant選択である。`canonical_shape_signature` の組込み用抜粋では、変数が初めて現れた順に番号を振る。`occurrences` はshape式の決めた走査順、`specs` はVarIdから上下限への表である。制約式も同じ番号表で直列化して19eのkeyへ渡す。

shape署名、variant選択、loweringのmemo keyをそれぞれ既存の責務へ分ける。

```rust
// src/shape.rs（既存ファイルへ組み込む抜粋）
use std::collections::BTreeMap;
fn canonical_vars(
	occurrences: &[VarId], specs: &BTreeMap<VarId,(u64,u64)>,
) -> Result<Vec<(usize,u64,u64)>, &'static str> {
	let mut ids = BTreeMap::new();
	let mut result = Vec::new();
	for id in occurrences {
		let next = ids.len();
		let normalized = *ids.entry(*id).or_insert(next);
		let &(min,max) = specs.get(id).ok_or("missing spec")?;
		result.push((normalized,min,max));
	}
	Ok(result)
}
// src/schedule.rs（既存ファイルへ組み込む抜粋）
fn select_variant(n: u64) -> &'static str {
	if n==16 { "specialized:N=16" } else { "generic" }
}
// src/rangeify.rs（既存ファイルへ組み込む抜粋）
#[derive(PartialEq, Eq, Hash)]
struct ValueAtKey {
	node: u32,
	coordinates: Vec<u32>,
	range_scope: u32,
	buffer_state: Option<u32>,
	binding_environment: u32,
}
```

`normalized` は同じ変数の再出現を保持する。二つの独立変数を同じNに潰さない。`select_variant` は20aの検査後だけ呼ぶので18をgenericへ逃がして検査を省略しない。`ValueAtKey` は08の `LowerKey` と同じ情報を持つrangeify内の値のmemoであり、binary cacheとは別である。`coordinates` はintern済み座標式のID、`range_scope` は反復scopeのIDとする。`buffer_state` は05のLOADが読むstateのノードIDで、純粋な値ではNoneにする。同じstorageと座標でも、書込み前後のstateを同じキーにしない。`binding_environment` は04の関数PARAMを実引数へ対応させる環境のIDである。同じ関数bodyを別の引数でloweringした結果も共有しない。このmemoは一回のlowering内だけに置き、別のPlan実行へ値を持ち越さない。

```text
vars=[Var7,Var9,Var7] -> [(0,1,17),(1,1,17),(0,1,17)]
vars=[Var2,Var4,Var2] -> 同じ署名、Bindings値はkeyに入らない
vars=[Var7,Var7,Var7] -> 別の署名
N=16,17,3 -> specialized,generic,generic
value_at(x,[i,j],scope0,s1) != value_at(x,[j,i],scope0,s1)
value_at(x,[i,j],scope0,s1) != value_at(x,[i,j],scope0,s2)
```

コード再利用の単位と、その場の shape 値を分ける。`src/schedule.rs` に `DispatchVariant` と選択条件、runtime の cache key 作成に `canonical_shape_signature` を足す案である。generic の署名では生の VarId の生成順に依存しないよう、graph 中で初めて現れた変数から0、1と番号を振り、同じ変数の再出現は同じ番号を使う。範囲と制約もその番号で表し、別変数二つを同じ一変数の式にまとめない。今回の Bindings の値は binary key へ入れない。

specialized は N=16を前提に定数化した kernel と、その前提を確認する predicate を一組として持つ。実行時は先に通常の bounds と容量を検査し、snapshot の N が16なら特化版、それ以外は generic を選ぶ。特化条件、tile と dtype は特化 key に含める。N=17の時に N=16のコードを選んで末尾を落とす誤りは、選択ログだけでなく出力17要素の比較で検出する。同じ名前で上限だけ変えた変数の key も分かれるようにする。

入力の先頭を1、2、3、...とすると、長さ16、17、3の順で最終要素は33、35、7になる。generic を最初に17で compile した後、3では同じコードを使いながら最後の論理位置が2へ変わることを確かめる。rangeify の値の memo は実行 cache と別物であり、`(node, 出力座標式, RANGE scope, buffer state)` をキーにする案とする。たとえば x とその転置が同じ元ノードを読む場合、座標(0,1)の x[0,1]と転置側の x[1,0]を共有しない。各 variant の source digest、選択理由、今回の slot 値を dump し、異なる座標と異なる変数制約の test で安全な共有範囲を固定する。

- [ ] generic kernel の key に VarId の正規化情報、上下限、制約、ABI を含める。
- [ ] generic kernel の key に毎回の N の値を混ぜず、1、3、17で同じコードを再利用する。
- [ ] `N=16` を条件とする特化版を一つ追加し、条件不成立なら generic に戻す。
- [ ] 特化 key には N=16 と tile 条件を含め、N=17でその binary を選ばないようにする。
- [ ] 添字式の memo key に node、出力座標、RANGE の scope を含めるか、scope 内に cache を閉じる。

予定例 `symbolic_dispatch --lengths 16,17,3` は特化、generic、generic の選択を dump に出す。
全出力は各 N の静的版と一致する。
shape の範囲を変えた同名変数や、同じ node を別座標で利用する例を作る。
cache が一致すべき場合と一致してはいけない場合を、それぞれ値で検証する。

## 完了確認

次の example と test はこの章で作成する予定である。

```sh
# 作業ディレクトリ: chibigrad/
cargo test
cargo test --test symbolic_shapes
cargo run --example symbolic_vector -- --backend cpu --length 3
cargo run --example symbolic_movement -- --rows 3 --dump-graph
cargo run --example symbolic_dispatch -- --lengths 16,17,3 --dump-plan
cargo run --example symbolic_vector -- --backend metal --length 17 --local 8
cargo run --example symbolic_vector -- --backend opencl --length 17 --local 8
```

- [ ] N の下限、上限、その外側、未束縛で期待どおり成功またはエラーになる。
- [ ] 整数の小例は期待値と完全一致する。
- [ ] f32 の generic、specialized、静的版は `abs_err <= 1e-5 + 1e-5*abs(ref)` を初期基準として比較する。
- [ ] RANGE の上限と buffer の容量を区別して dump に表示する。
- [ ] 動的 reshape の積、GEMM の K、broadcast の等値を起動前に検証する。
- [ ] Plan を連続再実行しても、束縛値と累積値が前回から残らない。
- [ ] 同じ node を転置前後で利用して、添字 cache が座標を混同しない。
- [ ] GPU での失敗は CPU の成功で置き換えず、backend 別に確認結果を残す。

## tinygrad ソースとの対応

参照 commit は `c22edff14c79310a9b2df62a78819cf77c8289fa` である。

| 実在シンボルと相対リンク | tinygrad の役割 | chibigrad で簡略化する点 |
|---|---|---|
| [`UOp.variable`](../../tinygrad/tinygrad/uop/ops.py#L974) | 名前と値域を持つ scalar PARAM を作る | shape 用の非負整数に限定する |
| [`UOp.bind`](../../tinygrad/tinygrad/uop/ops.py#L989) | 値域内の定数を変数へ束縛する | 元グラフとは別の Bindings に保持する |
| [`UOp.unbind_all`](../../tinygrad/tinygrad/uop/ops.py#L1001) | 束縛済みグラフから変数と値を分離する | 最初から仕様と実行値を分ける |
| [`no_overflow`](../../tinygrad/tinygrad/uop/symbolic.py#L34) | signed の symbolic 変換が値域内か確認する | 許可した整数式だけを検査する |
| [`TestSymbolicJit.test_plus1`](../../tinygrad/test/runtime/test_symbolic_jit.py#L9) | 長さを変えた JIT 再利用を検証する | JIT 全体ではなく Plan の再実行で始める |

## 今は追加しないもの

任意の symbolic 方程式、データの値で決まる出力長、一般の動的 rank は追加しない。
Transformer の sequence 長と cache 位置は[22](22-transformer-qwen.md)で、この範囲内の変数として扱う。
19の memory reuse は最大容量を確保し、論理長だけを理由に生存中の領域を重ねない。
