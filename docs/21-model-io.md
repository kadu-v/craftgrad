# 21. モデルの重みを保存し、検証したファイルから再実行する

## このステップで動かすもの

小さな線形層の重みをファイルへ保存し、別プロセスで読み込んで同じ結果を返す。
同じ Plan の入力スロットへ重みを差し替え、再コンパイルせずに出力を変える。
[20](20-symbolic-shapes.md)までの型と境界検証を前提にする。
元計画第8章の重みの差替えと第11章のモデル実行を、重み I/O へ拡張する。
最初はローカル fixture だけを使い、その後に URL、SHA-256、safetensors を追加する。
本文の関数、fixture、コマンドは今後実装する予定であり、既存 API の説明ではない。

```text
x=[2,3], weight=[4,5], bias=[1] -> y=24
load した同じ state             -> y=24
weight=[1,1], bias=[0] に差替え   -> y=5
```

## 最小実装とデータ

予定の model は `Linear { weight: ParamId, bias: ParamId }` とする。
`StateDict` は名前から Tensor の dtype、shape、連続 byte 列への対応である。
学習対象の parameter と、走行統計などの永続 buffer を区別して登録する。
両方を state に保存し、gradient や compiler の一時 buffer は保存しない。
入力 IR は weight と bias の PARAM を持つ純粋な graph、出力はその PARAM を参照する既存 Plan である。
state の入替えで純粋 graph と kernel source を変更しない。
複数の名前が同じ parameter を指す場合は ParamId を共有し、読み込み時の矛盾を検査する。

fixture には名前、shape、dtype、期待値、ファイル全体の SHA-256 を記した manifest を付ける。
manifest の digest は fixture 作成時に実 byte 列から算出する。架空の digest は本文に置かない。
読み込みの順序は次で固定する。

```text
file bytes -> byte 長と SHA-256 -> header の検査
           -> tensor 範囲の検査 -> state の型と名前の照合
           -> buffer の作成 -> device 転送 -> Plan の入力に設定
```

## この章のファイル配置

stateの登録と保存形式、safetensorsの検査をsrc/state.rsへまとめ、取得とSHA-256検証は21dでsrc/fetch.rsへ分ける。JSON、HTTP、SHA-256のlibraryはこの二つの入力境界から接続し、Tensorの計算をloaderへ移さない。

次はこの章の終わりまでに触る関連ファイルの配置案である。現在のcheckoutの一覧ではない。各コミットでは、その下に示す部分だけを追加・変更する。

```text
chibigrad/
├── examples/
│   ├── book_21_safetensors_range.rs  # 本文のコードを配置する。
│   ├── fetch_weights.rs  # manifest指定で小fixtureを取得する。
│   ├── safetensors_cast.rs  # 混合dtype fixtureの変換とf32 Plan実行。
│   ├── safetensors_linear.rs  # 独立writerのfixtureからLinearを実行する。
│   ├── state_roundtrip.rs  # 保存と別プロセスloadからLinearを実行する。
│   └── state_slots.rs  # 同じPlanへ二つのstateをbindする。
├── src/
│   ├── fetch.rs  # index、全shard、config、tokenizerの取得検証。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── state.rs  # F16、BF16変換とcollect_shards。
├── tests/
│   └── model_io.rs  # 特殊値と全shardの候補採用。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

生成Cやshaderは一時ディレクトリへ出力するため、このRustプロジェクトのツリーには含めない。コード冒頭のコメントで、生成元と生成先を確認する。

## 小さなコミットで進める

### 21a. ローカルの小さな state を往復する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── state_roundtrip.rs  # [追加] 保存と別プロセスloadからLinearを実行する。
├── src/
│   ├── lib.rs  # [更新] stateモジュールとexampleが使うAPIの公開。
│   └── state.rs  # [追加] TensorBytes、StateDict、登録、保存、検証、一括採用。
├── tests/
│   └── model_io.rs  # [追加] byte往復と失敗時の旧state保持。
└── Cargo.toml  # [更新] 教育用metadataを読むJSON libraryの依存。
```

src/lib.rsへstateのモジュール宣言を加え、src/state.rsにTensorBytes、StateDict、commit_state、f32_payloadを置く。登録とsave_state、read_stateも同じファイルへ加える。JSON writerとparserはここから選定したlibraryへ接続し、重複key、shape、byte範囲を検査した候補だけを一括採用する。examplesは公開する登録、保存、loadのAPIを使い、package名は実際のCargo.tomlから読む。

新しいモジュールは、既存の宣言を残して次を追記する。内部の補助関数は各モジュールに置き、exampleから使う入口だけを公開する。

```rust
// src/lib.rs（モジュール宣言を追加）
pub mod state;
```


入力は検証済みの名前付きTensor候補、出力は全項目が整合した次のstateである。`src/state.rs` に置く組込み用の抜粋では、候補を最後まで確認するまでlive stateを触らない。`TensorBytes` のdtypeは今回はF32固定で、shapeの積とbyte列を照合する。取得bytesの長さとdigestは21dの入口で先に検証する。

新しいstateモジュールの型と候補採用へ追加する。

```rust
// src/state.rs（既存ファイルへ組み込む抜粋）
use std::collections::BTreeMap;
#[derive(Clone)]
struct TensorBytes { shape: Vec<usize>, bytes: Vec<u8> }
type StateDict = BTreeMap<String,TensorBytes>;
fn commit_state(live: &mut StateDict, candidate: StateDict) -> Result<(),String> {
	if live.keys().ne(candidate.keys()) { return Err("state key set".into()); }
	for (name,tensor) in &candidate {
		let old = &live[name];
		if tensor.shape!=old.shape { return Err(format!("shape: {name}")); }
		let count = tensor.shape.iter().try_fold(1usize,|n,d|n.checked_mul(*d))
			.ok_or("element overflow")?;
		let bytes = count.checked_mul(4).ok_or("byte overflow")?;
		if tensor.bytes.len()!=bytes { return Err(format!("payload: {name}")); }
	}
	*live = candidate;
	Ok(())
}
fn f32_payload(values: &[f32]) -> Vec<u8> {
	values.iter().flat_map(|v|v.to_le_bytes()).collect()
}
```

`candidate` は所有権を持つ仮の辞書である。検査途中の `Err` は候補を捨て、`*live=candidate` は全件成功時だけ実行する。deviceの場合は、この代入より前に全転送の完了を待ち、辞書の代わりに `ParamId -> BufferSlot` の候補表を切り替える。serializationのJSON writerは名前とshapeを保存するadapterで、計算やstate採用の判断を持たせない。

```text
weight shape=[2], bytes=f32_payload([4,5]); bias shape=[1], bytes=f32_payload([1])
Plan inputs x=[2,3] -> products=[8,15] -> sum=23 -> output=24
candidate bias shape=[2], bytes=4 bytes -> reject, live output remains24
valid candidate weight=[1,1],bias=[0] -> atomic commit -> output5
```

既存の Linear は PARAM の入力値を runtime で受け取れる。このコミットでは `src/state.rs` を必要になった時点で追加し、`registered_state`、`save_state`、`read_state`、`commit_state` を置く案とする。登録項目は名前、ParamId または永続 buffer の ID、dtype、shape、役割を持つ。登録順を保存形式の暗黙の契約にせず、名前で照合する。保存対象の device 完了を待ち、view は論理順に連続化した byte 列へしてから metadata と payload を作る。gradient や一時領域を再帰的に拾う仕組みにはしない。

教育用形式でも offset と byte 長を明示する。weight shape[2]の8 byte、bias shape[1]の4 byteを並べるなら、weight は[0,8)、bias は[8,12)である。read_state は fixture の byte 長と manifest の SHA-256 を照合してから metadata を読み、shape の積、dtype の幅、範囲、名前を checked 演算で検査し、仮の StateDict を返す。全項目が揃った後にだけ commit_state が live な slot 表を一括で差し替える。二番目の項目が壊れていても最初の weight だけを変更しない。保存も一時出力へ完了した内容を検査してから確定名へ置く。

x=[2,3]、weight=[4,5]、bias=[1]なら LOAD した積は8と15、reduction は23、bias を足した STORE は24になる。保存と別プロセスの読込みで f32 の byte 列とこの途中値を照合する。その後 weight=[1,1]、bias=[0]を同じ Plan へ bind すると2+3で5になる。source が同じで slot の実 buffer だけが変わることを dump に示す。失敗 test では切れた payload と shape 不一致を入れ、load 前の state の byte と出力24が残ることを確認する。I/O の検査と既存生成 C の実行を一組で確認し、専用の線形層計算を loader 内へ置かない。

- [ ] 名前と ParamId を明示登録し、weight、bias、永続 buffer の一覧を得る。
- [ ] 最初は教育用の短い JSON metadata と little-endian f32 payload を保存する。
- [ ] dtype、shape、要素数、byte 数を検査してから新しい CPU buffer を作る。
- [ ] state 全体の検証が終わってから入れ替え、不完全な load でモデルを半分更新しない。
- [ ] 元の値24、load 後の24、差替え後の5を生成 C の実行結果で比較する。

予定例 `state_roundtrip --dir /tmp/chibigrad-state` を動かす。
保存する host byte 列は device の完了待ち後に読み出す。
このコミットでも通常の graph、schedule、C compiler、runtime を経由する。
独立した「モデル専用計算器」を追加しない。

### 21b. safetensors の f32 を読む

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_21_safetensors_range.rs  # 本文のコードを配置する。
│   └── safetensors_linear.rs  # [追加] 独立writerのfixtureからLinearを実行する。
├── src/
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── state.rs  # [更新] read_safetensorsとheader、tensor範囲検査。
├── tests/
│   └── model_io.rs  # [更新] header、offset、重複key、重なりの確認。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

掲載header_end、tensor_range、mainはexamples/book_21_safetensors_range.rsの一つの独立例へ置く。本体のread_safetensorsはsrc/state.rsへ追加し、同じ境界検査をJSON adapterが返すmetadataへ適用する。JSON adapterも同ファイルのreader入口で呼び、重複keyを保持せず消してしまうparser設定を使わない。safetensors_linearは通常のPlan実行まで通す別exampleにする。


入力はfile bytesとJSON parserが返したshape/offset、出力はそのtensorの検査済みファイル範囲である。次は単独で実行できる例で、JSONの解析と独立に `D=8+H` とoffsetの基準を確認する。実readerのJSON adapterは重複keyを拒否し、F32、非負shape、二個の整数offsetを返す契約とする。

header、相対offset、mainを含む範囲検査の独立例。JSON readerやLinear Planの完成例ではない。

```rust
// examples/book_21_safetensors_range.rs（単独で実行する例）
use std::ops::Range;
fn header_end(file: &[u8]) -> Option<usize> {
	let h = u64::from_le_bytes(file.get(..8)?.try_into().ok()?);
	let h = usize::try_from(h).ok()?;
	if h>1_048_576 { return None; }
	let d = 8usize.checked_add(h)?;
	if d>file.len() { None } else { Some(d) }
}
fn tensor_range(file_len: usize, d: usize, shape: &[usize],
                offsets: [usize;2]) -> Option<Range<usize>> {
	let elements = shape.iter().try_fold(1usize,|n,s|n.checked_mul(*s))?;
	let bytes = elements.checked_mul(4)?;
	let [begin,end] = offsets;
	if end.checked_sub(begin)?!=bytes { return None; }
	let start = d.checked_add(begin)?;
	let stop = d.checked_add(end)?;
	if stop>file_len { None } else { Some(start..stop) }
}
fn main() {
	let header = br#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
	let mut file = (header.len() as u64).to_le_bytes().to_vec();
	file.extend_from_slice(header);
	for v in [4.0f32,5.0] { file.extend_from_slice(&v.to_le_bytes()); }
	let d = header_end(&file).unwrap();
	let r = tensor_range(file.len(),d,&[2],[0,8]).unwrap();
	let values: Vec<_> = file[r].chunks_exact(4)
		.map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect();
	assert_eq!(values,[4.,5.]);
	assert!(tensor_range(file.len(),d,&[3],[0,8]).is_none());
	assert!(tensor_range(file.len(),d,&[2],[4,12]).is_none());
	assert_eq!(file.len(),8+header.len()+8);
	println!("D={d}, values={values:?}");
}
```

`h` はheaderだけの長さ、`d` はファイル上のdata開始位置である。`offsets=[0,8]` は `file[d..d+8]` を意味する。`checked_sub` が逆順を拒否し、shape積と4の積が範囲長に一致することを確かめる。全tensorの範囲は開始位置順に並べ、前のendより次のbeginが小さければ重なりを拒否する。空tensorの長さ0と隣接範囲は許す。

```text
header bytes [8,D), payload bytes [D,file.len)
weight offsets=[0,8], bias offsets=[8,12] -> disjoint
weight offsets=[0,8], bias offsets=[4,8] -> overlap -> reject
file.len=8+H+12; manifest digest covers all of these bytes
```

21aの検証済み StateDict を作る入口として、`src/state.rs` の `read_safetensors` を追加する案である。入力は検証対象の file bytes、出力は名前ごとの dtype、shape、payload の byte 範囲を持つ仮 metadata とする。まず先頭8 byteを little-endian の非負長 H として読み、`8+H` の overflow とファイル終端を検査する。JSON は byte 範囲[8,8+H)、data の開始 D は8+Hである。header の長さ上限も設け、異常な H だけで巨大な確保を行わない。

JSON の object は重複 key を検出できる方法で読む。`__metadata__` は tensor と区別し、各 tensor は F32、shape、二つの offset を要求する。data_offsets はファイル先頭からではなく D からの相対位置で、終端は含まない。shape[2]、offset[0,8]の weight は file[D..D+8]を表す。shape[1]の bias が[8,12]なら隣接してよいが、[4,8]なら重なりを拒否する。要素数と4の積が範囲長に等しいこと、全範囲が data 内に収まることを buffer 作成前に確認する。

weight=[4,5]の payload は16進 byteで `00 00 80 40 00 00 a0 40` になる。8 byteを一つの整数として読むのではなく、各4 byteを f32 に復号して4と5を得る。metadata の shape が[3]なら12 byteを要するため、この payload を読んで推測補完せずエラーにする。ヘッダ長、offset の基準、dtype 幅のそれぞれを一箇所だけ壊した fixture を用意し、失敗位置を分けて確認する。独立 writer の作った正常 fixture は同じ StateDict へ変換し、通常の Linear Plan の24まで実行する。safetensors の parse 成功と model の state 照合は別の関数境界として残し、正常な別 shape のファイルを無理に対象モデルへ入れない。

- [ ] 8 byte の header 長、JSON header、data 部分の開始位置を検査する。
- [ ] 各 tensor の `dtype`、`shape`、`data_offsets` を読み、積と byte 計算の overflow を拒否する。
- [ ] payload 内の範囲、長さの一致、不正な重なり、重複した名前を検証する。
- [ ] 未知 dtype と不正な header を明示エラーにし、f32 に推測変換しない。
- [ ] Rust の safetensors reader と独立に作った小 fixture で byte と数値を照合する。

予定例 `safetensors_linear --fixture fixtures/io/linear.manifest.json` は24を返す。
`[4,5]` は little-endian f32 の8 byteであることを検査する。
壊れた offset、切れた header、異常に大きい shape は buffer 確保前に拒否する。
信頼済みの小 fixture の作成には既存の safetensors writer を使ってよい。

### 21c. model へ厳密に state を入れる

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── state_slots.rs  # [追加] 同じPlanへ二つのstateをbindする。
├── src/
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── state.rs  # [更新] Registeredとresolve_state_slots、strictなmodel照合。
├── tests/
│   └── model_io.rs  # [更新] missing、extra、tied値不一致の確認。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/state.rsの登録仕様へRegisteredとresolve_state_slotsを加え、既存ParamIdとBufferSlotへ解決する。掲載u32は既存ParamIdの識別を表す抜粋として読み、新しいparameter ID体系を作らない。validate_state_for_modelで全差分を集めた後、全転送が完了した候補slot表だけをcommit_stateの採用処理へ渡す。


入力はモデルの登録仕様と仮StateDict、出力は全差分の一覧またはslot候補である。`src/state.rs` の組込み用の抜粋では、21aのF32 stateを使ってshapeとkeyを全件比較する。dtypeを広げた際は同じ箇所へdtype比較も足す。名前からParamIdへの表はmodel登録時に一度作り、tied名は同じIDを持つ。

明示登録仕様とstrictなslot解決へ追加する。

```rust
// src/state.rs（既存ファイルへ組み込む抜粋）
struct Registered { name: String, param: u32, shape: Vec<usize> }
fn resolve_state_slots(
	specs: &[Registered], state: &StateDict,
) -> Result<BTreeMap<u32,Vec<u8>>,Vec<String>> {
	let mut errors = Vec::new();
	let mut slots = BTreeMap::new();
	for spec in specs {
		match state.get(&spec.name) {
			None => errors.push(format!("missing {}",spec.name)),
			Some(t) if t.shape!=spec.shape => errors.push(format!("shape {}",spec.name)),
			Some(t) => {
				if let Some(old) = slots.get(&spec.param) {
					if old!=&t.bytes { errors.push(format!("tied value {}",spec.name)); }
				} else { slots.insert(spec.param,t.bytes.clone()); }
			}
		}
	}
	for name in state.keys() {
		if !specs.iter().any(|s|&s.name==name) { errors.push(format!("extra {name}")); }
	}
	if errors.is_empty() { Ok(slots) } else { Err(errors) }
}
```

`slots` はまだliveなPlanではない。複数名が同じ `param` を使う時はbyte一致を要求し、任意の片方を採用しない。全候補をdeviceへ転送して成功した後に、同じParamIdのPlan slotをまとめてrebindする。永続bufferは別の登録役割を持ち、optimizer集合には入れない。

```text
registered: embed.weight -> ParamId(3), lm_head.weight -> ParamId(3)
same bytes -> one buffer, two references
different bytes -> errors=["tied value lm_head.weight"], no rebind
missing bias + extra unrelated -> report both, old output24 remains
```

ファイルの境界を検証できても、目的の model に合うとは限らない。`src/state.rs` に `validate_state_for_model` と `resolve_state_slots` を足す案とし、前者は model の登録仕様と仮 StateDict を入力に、全差分の一覧または確定候補を返す。必要 key の集合差、余剰 key、各 shape、dtype を走査し、最初の一件で止めず同じ報告へ集める。partial load を明示した場合も、存在する項目の不正な shape や dtype は許可しない。欠落した項目を残す方針を明示し、未初期化 buffer を slot へ渡さない。

後者は `name -> ParamId -> BufferSlot` を解決し、全候補の検証と device 転送が終わってから slot 表を差し替える。同じ ParamId が embedding と出力 projection の二つの名前に現れるなら、両方は同じ確定 buffer に結び付ける。二つの保存値が矛盾する時はどちらかを任意に採用せず load を拒否する。ParamId はモデル中の論理的な共有、storage identity は今回の保存領域であり、load により storage が変わっても parameter の意味は変わらない。永続 buffer も slot を持てるが optimizer の parameter 集合へ混ぜない。

小例では first state の weight と bias を検証し、Plan の二つの slot を結ぶと8+15+1で24になる。second state の候補を全件確認して一括切替すると2+3+0で5になる。この間に source digest と compile 回数が変わらず、二つの PARAM が CONST になっていないことを確認する。bias だけ欠けた strict load を挟み、出力が24のままかも検査する。device 版は CPU の検証を済ませた候補だけを転送し、転送失敗でも live state を切り替えない。生成 kernel の実行、完了待ち、読出しを既存 backend に任せ、CPU と利用可能な各 device で同じ二状態の出力を比べる。

- [ ] 必須 key の欠落、想定外 key、shape 不一致、dtype 不一致を一度に報告する。
- [ ] strict load を標準にし、一部 load を明示要求した時だけ欠落を許す。
- [ ] 名前を Plan の入力 slot に解決し、同じコンパイル済み Plan を再実行する。
- [ ] 永続 buffer を保存と load に含め、optimizer が更新する parameter 一覧からは外す。
- [ ] Metal と OpenCL では検証済み CPU state を転送し、CPU 版の出力と比較する。

予定例 `state_slots --backend cpu --weights first,second` は順に24と5を返す。
同じ ParamId を共有する二つの参照が、load 後も一つの値を見ることを確認する。
backend を使えない環境の device test は未確認と記録する。
I/O の成功だけで device 上の推論成功とは判定しない。

### 21d. URL と SHA-256 を付けて取得する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── fetch_weights.rs  # [追加] manifest指定で小fixtureを取得する。
├── src/
│   ├── fetch.rs  # [追加] Digest256、fetch_to_cache、fetch_verifiedと検証済みpath。
│   ├── lib.rs  # [更新] fetchモジュールと取得入口の公開。
│   └── state.rs  # [更新] 取得検証済みpathからreaderを呼ぶ入口。
├── tests/
│   └── model_io.rs  # [更新] ローカルHTTP失敗とcache再検証。
└── Cargo.toml  # [更新] HTTPと標準SHA-256 libraryの依存。
```

候補をsrc/fetch.rsへ分ける構成に固定し、src/lib.rsにfetchを宣言する。Digest256、fetch_to_cacheとHTTP streamを作るadapterを同ファイルへ置き、選定したSHA-256 libraryのadapterもここで実装する。src/state.rsのreaderへはfetch_verifiedの成功pathだけを渡す。19eのcompile cacheもこのdigest adapterを使い、HTTP処理やJSON解析をruntimeへ入れない。

新しいモジュールは、既存の宣言を残して次を追記する。内部の補助関数は各モジュールに置き、exampleから使う入口だけを公開する。

```rust
// src/lib.rs（モジュール宣言を追加）
pub mod fetch;
```


入力は固定manifestと受信stream、出力は長さとSHA-256の両方が一致した確定pathである。次は `src/fetch.rs` の組込み用の抜粋であり、SHA-256 libraryに依存する。`Digest256` adapterは標準SHA-256を初期化済みで受け取り、`update` は入力全byteを順に消費し、`finish` は32byteを返す。自作hashや短縮checksumで代用しない。

fetchの新モジュールへ追加する。Digest256は選定する標準SHA-256 libraryで実装する接続契約である。

```rust
// src/fetch.rs（既存ファイルへ組み込む抜粋）
use std::{fs::{self,OpenOptions}, io::{self,Read,Write}, path::Path};
trait Digest256 { fn update(&mut self, bytes: &[u8]); fn finish(self) -> [u8;32]; }
fn fetch_to_cache<R: Read,H: Digest256>(
	mut stream: R, mut hash: H, expected_len: u64, expected_sha: [u8;32],
	temporary: &Path, destination: &Path,
) -> io::Result<()> {
	let mut file = OpenOptions::new().write(true).create_new(true).open(temporary)?;
	let result = (|| {
		let mut count = 0u64;
		let mut block = [0u8;8192];
		loop {
			let n = stream.read(&mut block)?;
			if n==0 { break; }
			count = count.checked_add(n as u64).ok_or(io::ErrorKind::InvalidData)?;
			if count>expected_len { return Err(io::ErrorKind::InvalidData.into()); }
			hash.update(&block[..n]);
			file.write_all(&block[..n])?;
		}
		if count!=expected_len || hash.finish()!=expected_sha {
			return Err(io::ErrorKind::InvalidData.into());
		}
		file.sync_all()?;
		fs::rename(temporary,destination)?;
		Ok(())
	})();
	if result.is_err() { let _ = fs::remove_file(temporary); }
	result
}
```

`temporary` は同じfilesystem内の一意名であり、`create_new` が共有を拒否する。確定先名は期待digestに結び付け、既存cacheは同じ長さとdigestを再検査する。`count` は受信中に増え、期待長を超えた瞬間に停止する。parserはこの関数の成功後だけ確定pathを受け取る。HTTP status、timeout、redirect制限はstreamを作るadapterの契約に置き、ローカルpathも同じ全byte検証へ通す。

```text
expected_len=8+H+12, expected_sha=manifestの実測SHA-256
stream truncated       -> no rename, no parser call
same length, byte flip -> SHA mismatch, no rename
cache hit              -> rehash full file -> verified path -> safetensors -> y=24
```

取得処理は parser の前に置く。`src/state.rs` と分ける必要が生じたら `src/fetch.rs` に `fetch_verified` を作る案で、入力は固定 revision の URL、期待長、期待 SHA-256、保存名、出力は検証を終えた path である。manifest の digest は実際の fixture bytes から一度算出して固定する。URL の末尾や HTTP の成功コードだけでは内容を保証しない。ローカル path も同じ検証関数を通し、URL と path の二経路で安全条件を変えない。

cache があれば先に長さと全内容の digest を検査する。一致すればその path を返し、不一致なら model loader へ渡さない。新規取得は cache と同じ filesystem の一意な一時名へ書き、到着 byte 数と SHA-256 を更新する。期待長を超えた時と途中切断は失敗とし、完了後に長さと digest の両方を照合してから確定名へ rename する。必要なら書込みの flush と同期を済ませ、成功前の一時名は公開しない。複数取得が同じ cache 名を狙っても一時名を共有せず、すでに確定した内容も digest で検査する。

正常なsafetensors fixtureのpayloadが12 byte、JSON header長がHなら、ファイル全体は `8+H+12` byteである。manifestにはこの全体長と、headerも含む全ファイルのSHA-256を記録する。受信が全体長に届かない途中状態はparserへ渡さない。全体長が一致していても一つのbyteが変わればSHA-256が異なり、headerの読込みやbuffer作成へ進まない。test server は正常、途中切断、長さ超過、同じ長さで内容変更の応答を切り替え、どの失敗でも検証済み path が返らないことを調べる。さらに正常 cache を一度取得してからネットワークを止め、再検証して再利用できることを確認する。戻された正常 path だけを21bの reader へ渡し、Linear の24まで通す。取得 test の成功を公開モデルの download 成功と書かず、小さな固定 bytes の転送と検証が成立した事実として記録する。

- [ ] manifest に固定 revision を含む URL、期待 byte 数、SHA-256、保存名を記録する。
- [ ] まずローカル HTTP fixture server から小ファイルを取得する test を作る。
- [ ] 一時ファイルへ保存し、byte 数と SHA-256 が一致した後だけ cache 名に rename する。
- [ ] 既存 cache と直接指定されたローカルファイルも、load の前に同じ digest 検査を通す。
- [ ] 通信失敗、途中切断、digest 不一致では未検証ファイルを model loader に渡さない。

予定例 `fetch_weights --manifest fixtures/io/http.manifest.json` は検証済み path を返す。
後続の safetensors reader は、その検証を通った path だけを受け取る。
URL 自体や cache 名を信頼性の根拠にしない。
CI はローカル fixture で完結し、公開モデルを毎回ダウンロードしない。

### 21e. 次章の checkpoint に必要な dtype と shard を加える

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── safetensors_cast.rs  # [追加] 混合dtype fixtureの変換とf32 Plan実行。
├── src/
│   ├── fetch.rs  # [更新] index、全shard、config、tokenizerの取得検証。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── state.rs  # [更新] F16、BF16変換とcollect_shards。
├── tests/
│   └── model_io.rs  # [更新] 特殊値と全shardの候補採用。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/state.rsのread_safetensorsに保存dtypeを加え、f16_to_f32、bf16_to_f32、convert_halfをcast_payload_to_f32から呼ぶ。collect_shardsをload_sharded_stateの候補結合に使う。indexのJSON adapterは同ファイルに置き、src/fetch.rsで全ファイルの検証を終えてから変換、結合、21cのmodel照合、一括採用へ進む。


入力はdtype付きの16bit payload、出力は明示変換したf32値である。`cast_payload_to_f32` の中核として、組込み用の抜粋を `src/state.rs` に置く。F16のsubnormalを通常値の指数式へ流さない。BF16は上位16bitへ置く方式である。

cast_payload_to_f32から使う保存dtypeの復号helperへ追加する。

```rust
// src/state.rs（既存ファイルへ組み込む抜粋）
fn f16_to_f32(bits: u16) -> f32 {
	let sign = if bits&0x8000==0 { 1.0f32 } else { -1.0 };
	let exponent = (bits>>10)&31;
	let fraction = bits&1023;
	match exponent {
		0 => sign*(fraction as f32)*2.0f32.powi(-24),
		31 => {
			let raw = ((bits as u32&0x8000)<<16)|0x7f800000|((fraction as u32)<<13);
			f32::from_bits(raw)
		}
		e => sign*(1.0+fraction as f32/1024.0)*2.0f32.powi(e as i32-15),
	}
}
fn bf16_to_f32(bits: u16) -> f32 { f32::from_bits((bits as u32)<<16) }
fn convert_half(bytes: &[u8], bf16: bool) -> Result<Vec<f32>, &'static str> {
	if bytes.len()%2!=0 { return Err("odd payload length"); }
	let mut out = Vec::with_capacity(bytes.len()/2);
	for pair in bytes.chunks_exact(2) {
		let bits = u16::from_le_bytes([pair[0],pair[1]]);
		out.push(if bf16 { bf16_to_f32(bits) } else { f16_to_f32(bits) });
	}
	Ok(out)
}
```

`exponent=0` では `fraction*2^-24` がF16の非正規数を表し、fraction=0ではsignを保ったゼロになる。`31` ではInfinityとNaNの分類をbitから保つ。shardの読込みは変換の成功だけでは確定しない。組込み時の `load_sharded_state` は、digest検証済みshard辞書を入力にして次の照合を行う。`index` のkey集合は21cのmodel登録表にも照合する。

load_sharded_state内で、全検証済みshardから候補StateDictを組むhelperへ追加する。

```rust
// src/state.rs（既存ファイルへ組み込む抜粋）
fn collect_shards(
	index: &BTreeMap<String,String>, shards: &BTreeMap<String,StateDict>,
) -> Result<StateDict,String> {
	let mut result = StateDict::new();
	for (shard_name,state) in shards {
		for (name,tensor) in state {
			if index.get(name)!=Some(shard_name) { return Err(format!("shard key {name}")); }
			if result.insert(name.clone(),tensor.clone()).is_some() {
				return Err(format!("duplicate {name}"));
			}
		}
	}
	if result.keys().ne(index.keys()) { return Err("missing shard tensor".into()); }
	Ok(result)
}
```

`result` は全shardの候補で、最後のmissing検査までlive stateを更新しない。BF16の `80 3f` とF16の `00 3c` はどちらも1だが、dtypeが違うので同じdecoderには渡さない。

```text
F16 0x0001 -> 2^-24; 0x8000 -> negative zero; 0x7c00 -> +inf
F16 0x3c00 -> 1; BF16 0x3f80 -> 1
index weight->part1,bias->part2; absent part2 -> missing shard tensor
normal two shards -> validate_state_for_model -> one atomic rebind -> y=24
```

既存の f32 loader を保ち、`read_safetensors` の dtype 記述と明示変換関数 `cast_payload_to_f32` を広げる案である。metadata は F16 と BF16 の名前と幅2 byteを保持し、保存 byte を f32 の4 byteと見なさない。復号と変換を分け、BF16 は16 bitを f32 の上位16 bitへ配置し、F16 は符号、指数、仮数を取り出して値を作る。F16 の subnormal と特殊値には通常値と別の処理を設ける。各関数の入力は元 dtype と連続 bytes、出力は shape を保った f32 bytes であり、計算 graph の dtype はここでは変えない。

たとえば little-endian の `80 3f` は BF16 の1、`00 3c` は F16 の1である。同じ2 byteを型の情報なしに読むと別の値になるため、未知の dtype 名から推測しない。変換 test は正負ゼロ、通常値、最小付近の subnormal、無限大、NaN を別々に作り、有限値は独立 reference と照合する。ゼロの符号と無限大の符号を調べ、NaN は数値の等号で判定せず分類を確認する。変換後の1と線形層の積和は既存 f32 pipeline まで通して確認する。

shard では `load_sharded_state` が index の `tensor名 -> shard名` を読み、必要な shard の集合を重複なく列挙する。全ファイルの長さと digest、config、tokenizer の検証を先に完了し、各 shard の key と index の対応を照合してから仮 StateDict を組む。ある key の指定先が違う、二つの shard に同じ key がある、最後の shard が欠ける場合は state 全体を公開しない。最後に21cの厳密な model 照合と一括切替を使う。小さな二 shard fixture で正常時の24と、後半失敗時に以前の24が残ることを確認する。全 checkpoint を読み切る前に一部の layer を live state へ登録しないため、大きなモデルでも load の成否を一つの境界に保てる。

- [ ] F16 と BF16 の payload を読む場合は型を保持し、最初の実行用に明示的な f32 変換を用意する。
- [ ] ゼロ、有限値、非正規数、無限大、NaN を使って各変換を独立の reference と比較する。
- [ ] shard index から必要ファイルを列挙し、全 shard の digest を確認した後に state を構築する。
- [ ] index 内の名前と各 shard の metadata、必要 key の集合を照合する。
- [ ] config と tokenizer の各ファイルも manifest の検証対象にする。

予定例 `safetensors_cast --fixture fixtures/io/mixed-dtypes.manifest.json` を追加する。
これは計算 IR 全体を half 化する変更ではない。変換後は既存 f32 pipeline で線形層を実行する。
F16 と BF16 のうち、選んだ checkpoint にない型はその checkpoint の必須実装にしない。

## 完了確認

以下はこの章で作る予定の example と test のコマンドである。

```sh
# 作業ディレクトリ: chibigrad/
cargo test
cargo test --test model_io
cargo run --example state_roundtrip -- --dir /tmp/chibigrad-state
cargo run --example safetensors_linear -- --fixture fixtures/io/linear.manifest.json
cargo run --example state_slots -- --backend cpu --weights first,second
cargo run --example fetch_weights -- --manifest fixtures/io/http.manifest.json
cargo run --example safetensors_cast -- --fixture fixtures/io/mixed-dtypes.manifest.json
```

- [ ] f32 は roundtrip 前後の byte と数値が一致し、線形層が24を返す。
- [ ] SHA-256 不一致では header 解析、buffer 確保、device 転送を始めない。
- [ ] dtype、名前、shape、offset の破損 fixture をそれぞれ拒否する。
- [ ] dump で weight の PARAM slot を確認し、重みが定数に折り畳まれていない。
- [ ] device の小例は CPU reference に対し `abs_err <= 1e-5 + 1e-5*abs(ref)` とする。
- [ ] 大きなモデルなしで、保存、load、取得、破損時の経路を検証できる。

## tinygrad ソースとの対応

参照 commit は `c22edff14c79310a9b2df62a78819cf77c8289fa` である。

| 実在シンボルと相対リンク | tinygrad の役割 | chibigrad で簡略化する点 |
|---|---|---|
| [`safe_load_metadata` と `safe_load`](../../tinygrad/tinygrad/nn/state.py#L44) | safetensors header と tensor を読む | 外部 byte 列の境界検査を先に行う |
| [`safe_save`](../../tinygrad/tinygrad/nn/state.py#L64) | dtype、shape、offset と data を保存する | CPU の連続 f32 から始める |
| [`get_state_dict`](../../tinygrad/tinygrad/nn/state.py#L89) | model の属性を再帰的に列挙する | Rust の明示登録を使う |
| [`load_state_dict`](../../tinygrad/tinygrad/nn/state.py#L128) | shape と device を照合して model に入れる | 全件検証後に slot を一括更新する |
| [`fetch`](../../tinygrad/tinygrad/helpers.py#L460) | HTTP cache と任意の SHA-256 検証 | local path でも digest 検証を必須にする |
| [`hf_load`](../../tinygrad/tinygrad/llm/hf.py#L9) | shard と重み名の対応を処理する | 単一 device、量子化なしで始める |

`fetch` はこの参照版ではローカル path を即座に返す。そこに SHA 検証があるとは扱わない。

## 今は追加しないもの

pickle や任意 Python object の復元、量子化形式、mmap の zero-copy、分散 checkpoint は追加しない。
[22](22-transformer-qwen.md)で検証済みの config と重みを組み合わせる。
optimizer と乱数状態まで保存する学習 checkpoint は[24](24-llm-training.md)で追加する。
