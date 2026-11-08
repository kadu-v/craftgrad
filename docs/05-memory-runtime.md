# 05. バッファの LOAD と STORE を安全な順序で CPU 実行する

## このステップで動かすもの

[04-functions-stack.md](04-functions-stack.md) の値のグラフにメモリ操作を加える。
入力 `[10,20,30]` の添字 2 の要素を読み、`out[0] = x[2] + 1` で `31` を返す。
次に短い配列を展開して読み書きし、複数入力と新規出力を扱う。
元計画第 3 章のメモリ、実行順、CPU runtime に対応する。
以下の IR、API、コマンドと期待値は未実装の作業計画である。

## 最小実装とデータ

`BUFFER` は実行時の保存領域との束縛、`ALLOC` は新しい保存領域の宣言にする。
同じ長さと dtype の ALLOC を二つ作っても別の `StorageId` にする。
ALLOC は呼出しごとに新しい領域を得る。グラフ作成時には値を確保しない。
ホスト側の束縛表が所有物、長さ、dtype、可変性を保持する。

`addrspace` は scalar 値の `ALU`、外部保存領域の `GLOBAL`、局所変数の `REG` から始める。
LOCAL は GPU の章まで未対応として拒否する。
前章の INDEX は STACK の値選択、今回の INDEX はメモリ上の要素位置である。
両方を区別できる kind を導出し、メモリ INDEX を ALU に直接渡さない。
`LOAD(INDEX(...))` だけが保存領域から値を読む。
`STORE(INDEX(...),value)` は `void` の効果になる。

`SHRINK(base,start,end)` は連続した一次元の保存領域の view に限定する。
ここでの start と end は byte 数ではなく要素数で、範囲は半開区間とする。
view を作っても新規 allocation は行わない。
`storage_base` は SHRINK と AFTER をたどり、元の StorageId を返す。
INDEX のアドレスは base offset と要素位置から求める。

予定する入力と出力依存は次の形になる。

```text
%x = BUFFER f32 len=3 binding=input0
%o = ALLOC f32 len=1 GLOBAL
%v = LOAD(INDEX(%x, 2))
%s = STORE(INDEX(%o, 0), ADD(%v, 1.0))
%result = AFTER(%o, %s)
```

`AFTER(storage,effect...)` は、同じ保存領域を効果完了後に参照する状態にする。
効果を伴う STORE は純粋 ALU と同じ intern 規則で共有しない。
LOAD の共有はアドレスに加えて読む buffer state を含む場合だけ許す。
古い状態の LOAD を書込み後の LOAD に置換しない。
最初の外部入力は読取り専用、新規出力だけを可変にする。
同じ保存領域への別名入力と破壊的更新は、局所 REG の場合を除き拒否する。

`accesses(kernel)` は StorageId ごとの read と write を導出する予定関数である。
この名前は chibigrad の設計案で、tinygrad の同名 property を示さない。
効果付き CALL は明示 input と output の PARAM を受け取り `void` を返す。
出力を読む側は `AFTER(output,CALL(...))` を参照する。
前章の純粋 scalar CALL は inline のまま残す。
外部入口には [02](02-runtime-inputs.md) の二配列 ABI を使う。

## この章のファイル配置

StorageId と view の意味は uop に置き、保存領域の所有と FFI は既存 runtime/cpu にまとめる。interpreter と verifier をメモリに対応させ、順序だけを確かめる main は独立した example にする。

次はこの章の終わりまでに触る関連ファイルの配置案である。現在のcheckoutの一覧ではない。各コミットでは、その下に示す部分だけを追加・変更する。

```text
chibigrad/
├── examples/
│   ├── book_05_state_order.rs  # read-before-overwrite の辺を確かめる std の独立例。
│   └── memory_scalar.rs  # shrink と三要素の展開済み加算。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # 合成した要素 offset を使うアドレスの生成。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── runtime/
│   │   ├── cpu.rs  # 検証済み BufferDesc と実際の束縛の照合。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── interpret.rs  # 同じ依存順で状態と効果を実行する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── linearize.rs  # 値、state、sequence_before の順序辺を含む走査。
│   ├── rewrite.rs  # 置換後も順序辺の参照先を更新する。
│   ├── uop.rs  # View、SHRINK、storage_base、view_offset と accesses。
│   └── verify.rs  # InitState と verify_memory、循環とサイズの検査。
├── tests/
│   └── memory_runtime.rs  # LOAD、STORE、view、状態と不正入力の統合確認。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

生成Cやshaderは一時ディレクトリへ出力するため、このRustプロジェクトのツリーには含めない。コード冒頭のコメントで、生成元と生成先を確認する。

## 小さなコミットで進める

### コミット 1. 一つの入力から一つの出力へ書く

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── memory_scalar.rs  # [追加] 一入力一出力のグラフ構築と生成コードの実行。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # [変更] buffer PARAM とアドレスの C 生成。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── runtime/
│   │   ├── cpu.rs  # [変更] StorageId の束縛表、出力 allocation と buffer 署名検証。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── interpret.rs  # [変更] Address と固定添字の load、store。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── uop.rs  # [変更] StorageId、BufferDesc、メモリ INDEX、LOAD、STORE と kind。
│   └── verify.rs  # [変更] メモリ kind、長さと可変性の検証。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

StorageId と BufferDesc は src/uop.rs に追加する。Address、eval_index、load、store の抜粋は interpret.rs に置き、scalar_memory_example は同ファイル内の検証用 helper として使う。実際の example の main は examples/memory_scalar.rs に置き、ホスト計算で代用せずグラフを renderer/c.rs と runtime/cpu.rs へ渡す。runtime の既存入口を buffer 配列に広げ、別の FFI 経路は作らない。


従来の scalar の値に、保存先の参照と読書きの効果を追加する。候補ファイル `src/uop.rs` に設計案の `StorageId`、`BufferDesc { dtype, len, mutability }`、メモリ用 INDEX を置く。BUFFER は入力 slot と記述子を持ち、ALLOC は同じ記述子でも毎回新しい StorageId を発行する。`src/runtime/cpu.rs` の束縛表が StorageId にホスト配列の所有物を対応させる。graph の構築時には出力の実アドレスを決めない。

実装順はアドレス kind の導出、固定 INDEX、LOAD、STORE、実行順の接続とする。INDEX(x,2) は f32 の値ではなく x の要素位置であり、LOAD が初めて値 30 を返す。ADD が 31 を作り、STORE が出力の INDEX 0 へ書く。最初は builder の明示的な効果順で STORE を一つ実行すればよい。C の本体は `out[0]=x[2]+1.0f` となるが、Rust 側は入力の長さ 3 と出力の長さ 1 を検証してからアドレスを渡す。

固定 ABI の buffers 配列には kernel の署名順で入力 x と新規出力 out を並べる。scalars 配列は従来の実行時 scalar のために残し、ポインタを scalar word に混ぜない。interpreter も同じ BufferDesc とホスト値の表を読み書きし、メモリ操作の比較先にする。局所テストでは `[10,20,30]` の結果 31 に加え、入力が変更されていないこと、長さ 2 の束縛を呼出し前に拒否することを確認する。前章までの scalar 出力 wrapper と library 所有権を使い、STORE のために別の runtime を作らない。


組込み用の抜粋である。`src/interpret.rs` の固定添字実行を示す。ここでは `Op::Buffer(slot)`、`Alloc { storage, len }`、`Index(i32)`、`Load`、`Store` を使い、src の役割を明示する。`Memory` は初期化済み f32 領域の表で、実 runtime の署名検証と allocation の後に使う。

アドレスと値と効果の評価を interpret.rs に追加する。scalar_memory_example は同じファイル内の検証用 helper で、実 example の main は別に作る。

```rust
// src/interpret.rs（既存ファイルへ組み込む抜粋）
struct Address { storage: usize, offset: usize }
fn eval_index(base: usize, len: usize, index: i32) -> Result<Address,String> {
	let offset = usize::try_from(index).map_err(|_| "negative INDEX")?;
	if offset >= len { return Err("INDEX exceeds buffer length".into()); }
	Ok(Address { storage: base, offset })
}
fn load(memory: &[Vec<f32>], address: &Address) -> f32 {
	memory[address.storage][address.offset]
}
fn store(memory: &mut [Vec<f32>], address: &Address, value: f32, writable: &[bool]) -> Result<(),String> {
	if !writable[address.storage] { return Err("STORE to read-only input".into()); }
	memory[address.storage][address.offset] = value;
	Ok(())
}
fn scalar_memory_example(input: &[f32]) -> Result<f32,String> {
	if input.len() != 3 { return Err("expected input length 3".into()); }
	let mut memory = vec![input.to_vec(), vec![f32::NAN;1]];
	let read = eval_index(0,3,2)?;
	let write = eval_index(1,1,0)?;
	let value = load(&memory,&read) + 1.0;
	store(&mut memory,&write,value,&[false,true])?;
	Ok(memory[1][0])
}
```

INDEX はアドレス、LOAD は値、STORE は効果という違いを関数の型に出す。`scalar_memory_example` の `memory[0]` は入力のコピーを用いる interpreter の所有物であり、生成 C では元の read-only 入力を直接読む。二配列 ABI の buffers は `[x_pointer,out_pointer]`、scalars は別配列のままにする。この例の `NAN` はホスト上の検出補助であり、未初期化 LOAD を許す根拠にはしない。

この入力の途中経過を並べる。

```text
INDEX(storage=0,len=3,index=2) -> Address(0,2)
LOAD -> 30; ADD 1 -> 31
STORE Address(1,0),31 -> out=[31]
input=[10,20,30] は変更されない
```


組込み用の抜粋である。同じ演算の生成 C 本体を示す。署名の長さと可変性は Rust が呼出し前に検証する。

LOAD と STORE から renderer が生成する実 C の入口である。

```c
// src/renderer/c.rs が生成するコード。生成先: <一時ディレクトリ>/kernel.c
#include <stdint.h>
void chibi_entry(void *const *buffers, const uint64_t *scalars) {
	const float *x = (const float *)buffers[0];
	float *out = (float *)buffers[1];
	(void)scalars;
	float value = x[2];
	out[0] = value + 1.0f;
}
```

- [ ] BUFFER の束縛と ALLOC の固有 ID を追加する。
- [ ] メモリ INDEX、LOAD、STORE を C と検証用 interpreter に加える。
- [ ] buffer の PARAM を固定 ABI のポインタ配列へ接続する。
- [ ] 入力の長さと dtype を呼出し前に照合する。
- [ ] `[10,20,30]` から生成コードで `31` を得る。

最初のコミットは固定添字だけでよい。
scalar 入出力の wrapper も維持し、前章までの例を通す。

### コミット 2. AFTER で書込み後の状態を読む

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_05_state_order.rs  # [追加] read-before-overwrite の辺を確かめる std の独立例。
│   └── memory_scalar.rs  # [変更] 同じ領域の二つの LOAD を実 CPU で比較する。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # [変更] 先行 LOAD の一時変数と STORE の順序を保つ。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── interpret.rs  # [変更] 同じ依存順で状態と効果を実行する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── linearize.rs  # [変更] 値、state、sequence_before の順序辺を含む走査。
│   ├── rewrite.rs  # [変更] 置換後も順序辺の参照先を更新する。
│   ├── uop.rs  # [変更] BufferState、AFTER と状態を含む LOAD キー。
│   └── verify.rs  # [変更] verify_state_chain と明示順序辺の検査。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

BufferState と sequence_before の構築口は uop.rs に追加し、順序辺を linearize.rs と interpret.rs の走査へ接続する。verify_state_chain は境界検査として verify.rs に置く。独立例の visit と main は examples/book_05_state_order.rs に保存する。本体の既存 DFS をその局所型で置き換えず、明示辺をたどる規則だけを反映する。memory_scalar では二つの読出しを別の出力位置に保存する。


保存領域の同一性と、その領域のどの時点を読むかを分ける。候補ファイル `src/uop.rs` と `src/linearize.rs` の設計案 `BufferState { storage, predecessor }`、`verify_state_chain` を使い、AFTER は新しい StorageId を発行せず、効果完了後を示す state を返す。ALLOC が領域 S と未初期化 state s0 を作り、STORE1 の後を s1、STORE2 の後を s2 と呼ぶと、三つの state の storage_base はすべて S になる。

例を `STORE1(S,0,1)`、`s1=AFTER(S,STORE1)`、`r1=LOAD(INDEX(s1,0))`、`STORE2(s1,0,2)`、`s2=AFTER(s1,STORE2)`、`r2=LOAD(INDEX(s2,0))` と組む。必要な順序は `STORE1 → r1 → STORE2 → r2` である。state を LOAD の intern key に含めるだけでは、r1をSTORE2より前に実行する保証はない。AFTERは過去の値をコピーしたsnapshotではなく、同じ保存領域への順序を表す。

この初版では、builderに設計案の `sequence_before(r1,store2)` を用意し、「この読出しを済ませてから次の書込みへ進む」という順序辺を明示する。bodyが持つ順序辺を値の依存辺と合わせ、interpreterとlinearizeの両方でたどる。rewriteでノードを置換した際も辺の参照先を更新する。古いstateを読むノードと、それを上書きするノードの間に順序辺がなければ、この版のverifierは未対応として拒否する。汎用の読み書きハザード解析を先に作る必要はない。Cではr1をscalarの一時変数へ保存した後で領域Sを2へ更新し、その後r2を読む。二つのLOADの共有キーにはそれぞれs1とs2を使い、計算済みのr1を後のLOADへ置換しない。STOREも同じ値と位置という理由で消さない。

state の直列性は、同じ s0 から独立した二つの STORE が出ていないかを検証して保つ。分岐した場合に番号順で勝手に順序を決めない。効果付き CALL は出力 PARAM を含む署名と Void の body を持ち、外側は AFTER(output,CALL) を通じて結果を読む。局所テストはr1とr2の両方を別の新規出力へ保存し、interpreterと生成Cが `[1,2]` を返すことを調べる。最終値2だけではr1の遅延読出しを検出できない。同じStorageId、異なるLOAD state、r1からSTORE2への順序辺もdumpで見る。値と効果と順序辺を含むrootから実行列を作り、到達可能なSTOREや先行読出しを落とさない。


単独で実行できる例である。`src/linearize.rs` の順序を一要素の局所保存領域 S で検証する。ここでは src に値・state・明示順序の辺を合わせて持つ最小例にし、`Store(2)` の二本目の辺が read-before-overwrite を表す。

局所型、visit と main を一つの独立例として保存する。

```rust
// examples/book_05_state_order.rs（単独で実行する例）
#[derive(Clone,Copy)]
enum Op { Store(i32), After, Load, Results }
struct UOp { op: Op, src: Vec<usize> }
fn visit(g: &[UOp], id: usize, done: &mut [bool], order: &mut Vec<usize>) {
	if done[id] { return; }
	for &s in &g[id].src { visit(g,s,done,order); }
	done[id] = true;
	order.push(id);
}
fn main() {
	let g = vec![
		UOp { op: Op::Store(1), src: vec![] },
		UOp { op: Op::After, src: vec![0] },
		UOp { op: Op::Load, src: vec![1] },
		UOp { op: Op::Store(2), src: vec![1,2] },
		UOp { op: Op::After, src: vec![1,3] },
		UOp { op: Op::Load, src: vec![4] },
		UOp { op: Op::Results, src: vec![5,2] },
	];
	let mut order = Vec::new();
	visit(&g,6,&mut vec![false;g.len()],&mut order);
	let mut storage = None;
	let mut values = vec![None;g.len()];
	for &id in &order {
		match g[id].op {
			Op::Store(x) => storage = Some(x),
			Op::Load => values[id] = Some(storage.expect("initialized S")),
			Op::After | Op::Results => {}
		}
	}
	assert_eq!((values[2],values[5]),(Some(1),Some(2)));
	assert!(order.iter().position(|&x| x==2) < order.iter().position(|&x| x==3));
	println!("r1={}, r2={}",values[2].unwrap(),values[5].unwrap());
}
```

root は意図的に r2 を先に要求する。DFS が r2 の依存をたどっても、Store2 の明示辺 `[1,2]` が r1 を先に評価させる。AFTER 自体は `storage` をコピーせず、値を読むのは LOAD の時点である。もし Store2 の r1 辺を消すと、root のこの順序では r1 も 2 を読み、テストが失敗する。実装ではメモリ INDEX と StorageId を追加し、LOAD の key を `(address,state)`、順序辺を別表にしても同じ走査でたどる。

この入力の途中経過を並べる。

```text
root sources=[r2,r1]
DFS: Store1 -> s1 -> r1 -> Store2 -> s2 -> r2 -> root
S: uninitialized -> 1 -> 2
LOAD(s1)=r1=1, LOAD(s2)=r2=2
AFTER の数=2、StorageId の数=1
```

- [ ] 効果を依存順に実行し、到達可能な STORE を消さない。
- [ ] AFTER の返す storage identity と状態を区別する。
- [ ] 同じ位置へ `1`、次に `2` を書き、その後に読んで `2` を返す。
- [ ] LOAD の共有キーへ状態を含め、異なる時点の読出しを共有しない。
- [ ] 複数の効果を持つ CALL の出力を AFTER から読めるようにする。

state の遷移は直列な chain に限る。
同じ古い state から別々の書込みが分岐した場合は、順序不明として拒否する。

### コミット 3. SHRINK と accesses を実際の読書きに使う

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── memory_scalar.rs  # [変更] shrink と三要素の展開済み加算。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # [変更] 合成した要素 offset を使うアドレスの生成。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── runtime/
│   │   ├── cpu.rs  # [変更] 異なる ALLOC の所有物と署名順の束縛。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── uop.rs  # [変更] View、SHRINK、storage_base、view_offset と accesses。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

View、shrink、byte_offset、MemoryUse、accesses の抜粋は src/uop.rs の保存領域の導出情報の近くに追加する。storage_base と view_offset もここで AFTER と SHRINK をたどる。renderer/c.rs は得られた offset を使い、runtime/cpu.rs は StorageId ごとに異なる所有物を束縛する。三要素のグラフ builder と dump 表示は examples/memory_scalar.rs に加える。


view の追加では、所有する領域を増やさずに読める区間だけを変える。候補ファイル `src/uop.rs` の設計案 `storage_base` と `view_offset` は、SHRINK をたどって base StorageId と要素 offset を合成する。`SHRINK(x,1,3)` は長さ 2、offset 1 の view なので、INDEX(view,0) は x の位置 `1+0=1` となる。C の byte アドレスを直接足す前に、要素 offset の加算を checked 演算で求め、最後に itemsize を掛ける。

`accesses(kernel)` は LOAD と STORE を走査し、そのアドレスから `storage_base` を求めて read と write の集合へ登録する設計案にする。view のノードを別 allocation として数えると、x の一部を読んでいる事実が消えるため、集合の key は StorageId にする。AFTER を通る参照でも保存元は変えない。ただし読む時点は state 側に残し、accesses の集計で LOAD の共有条件まで弱めない。

二入力加算はまだ RANGE を使わず、位置 0、1、2 の三つの STORE を明示順で作る。中間値は 1+10=11、2+20=22、3+30=33 で、新規出力だけへ書く。局所テストでは shrink の位置 0 から 20 を読み、dump に入力 x の StorageId が read として一度現れることを確認する。長さも dtype も同じ ALLOC を二つ作る fixture では異なる初期値を書き、一方を更新しても他方が変わらないことを見る。これで構造が同じ領域宣言を intern してしまう誤りを検出できる。


組込み用の抜粋である。`src/uop.rs` の view 合成と `accesses` の集計を示す。`View { storage, offset, len }` の offset は要素単位である。`MemoryUse` は検証済み LOAD と STORE から集めたアドレス使用で、読む state は元のノードに残す。

保存元の導出と view のアクセス集計を既存 uop に追加する。

```rust
// src/uop.rs（既存ファイルへ組み込む抜粋）
#[derive(Clone,Copy)]
struct View { storage: usize, offset: usize, len: usize }
fn shrink(base: View, start: usize, end: usize) -> Result<View,String> {
	if start > end || end > base.len { return Err("invalid SHRINK interval".into()); }
	let offset = base.offset.checked_add(start).ok_or("offset overflow")?;
	Ok(View { storage: base.storage, offset, len: end-start })
}
fn byte_offset(view: View, index: usize, itemsize: usize) -> Result<usize,String> {
	if index >= view.len { return Err("INDEX outside view".into()); }
	view.offset.checked_add(index).and_then(|n| n.checked_mul(itemsize)).ok_or_else(|| "byte offset overflow".into())
}
struct MemoryUse { view: View, write: bool }
fn accesses(uses: &[MemoryUse]) -> std::collections::BTreeMap<usize,(bool,bool)> {
	let mut result = std::collections::BTreeMap::new();
	for access in uses {
		let flags = result.entry(access.view.storage).or_insert((false,false));
		if access.write { flags.1 = true; } else { flags.0 = true; }
	}
	result
}
```

`shrink` は `storage` をコピーし、保存領域を増やさず offset と長さだけを変える。二段の view でも offset が加算される。`byte_offset` は view 内の範囲検証を先に行うので、元領域に収まるという理由で view 外の INDEX を受理しない。`accesses` の key は view の番号でなく元の storage であり、同じ入力を複数回読んでも read は一つのフラグにまとまる。ALLOC の storage は intern を通さず単調な ID 発行で作る。

この入力の途中経過を並べる。

```text
base=(storage=7,offset=0,len=3)
SHRINK(base,1,3)=(storage=7,offset=1,len=2)
INDEX(view,0): element=1, byte_offset=4 -> x[1]=20
accesses: storage7 -> (read=true,write=false)
ALLOC len3 -> storage8; ALLOC len3 -> storage9
```

- [ ] `SHRINK(x,1,3)` と base offset を導出する。
- [ ] view の INDEX 0 を元 buffer の INDEX 1 へ変換する。
- [ ] storage_base と read、write の集合を dump に出す。
- [ ] 二入力 `[1,2,3]`, `[10,20,30]` を展開して `[11,22,33]` を作る。
- [ ] 同じ長さの二つの ALLOC が互いを上書きしないことを確かめる。

view から読む予定例は `[10,20,30]` の shrink の 0 番目で `20`。
ループは次章なので、三要素の加算は STORE を三つ作る。

### コミット 4. メモリ verifier を生成前に通す

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── src/
│   ├── runtime/
│   │   ├── cpu.rs  # [変更] 検証済み BufferDesc と実際の束縛の照合。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── verify.rs  # [変更] InitState と verify_memory、循環とサイズの検査。
├── tests/
│   └── memory_runtime.rs  # [追加] LOAD、STORE、view、状態と不正入力の統合確認。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

InitState と impl は src/verify.rs に追加し、verify_memory の state ごとの初期化検査に使う。runtime/cpu.rs は呼出し前に実配列と検証済みの記述子を照合する。tests/memory_runtime.rs で公開 builder と compile/run を通し、範囲外、未初期化、循環が C 生成前に失敗することを確認する。


メモリ操作の検証を renderer の前へ置き、不正なアドレスを生成しない。候補ファイル `src/verify.rs` の設計案 `verify_memory` は、各ノードに値、列、アドレス、効果の kind を導出する。LOAD の入力はアドレス、STORE はアドレスと同 dtype の値、返り値は Void とする。BUFFER の読取り専用指定も束縛表だけに任せず署名へ含め、入力への STORE は静的検証で拒否する。

範囲検証は最初に SHRINK の半開区間を調べ、その長さを INDEX の許容範囲に使う。長さ 3 の x なら位置 2 は有効、3 と -1 は無効である。view の offset を足す前にも view 内の位置を確認するので、元領域の中に偶然収まる view 外アクセスを通さない。allocation byte 数と合成 offset は checked なホスト整数で求め、overflow した値を小さなアドレスとして受理しない。

未初期化の検証では state ごとに確定して書かれた位置の集合を追う案にする。ALLOC 直後は空、固定 STORE 0 の後は `{0}` となり、その state の LOAD 0 だけを許す。入力 BUFFER は検証済みの長さの範囲が初期化済みである。AFTER の依存は DFS の訪問中集合で循環を検出し、効果が完了しない state を作らない。局所テストには x[-1]、x[3]、読取り専用 STORE、未初期化 LOAD、overflow と循環を分けて置き、compile 計数が増えないことを調べる。成功例 31 と view の 20 は、同じ verifier を通ってから既存 CPU 経路で実行する。


組込み用の抜粋である。`src/verify.rs` の固定位置に対する初期化検証を示す。`InitState` は各 AFTER state の「確実に書かれた位置」の集合であり、領域全体に一つだけ置く集合ではない。

state ごとの初期化検証を追加する。

```rust
// src/verify.rs（既存ファイルへ組み込む抜粋）
#[derive(Clone)]
struct InitState { len: usize, initialized: std::collections::BTreeSet<usize> }
impl InitState {
	fn allocated(len: usize, itemsize: usize) -> Result<Self,String> {
		len.checked_mul(itemsize).ok_or("allocation size overflow")?;
		Ok(Self { len, initialized: std::collections::BTreeSet::new() })
	}
	fn after_store(&self, index: i32, writable: bool) -> Result<Self,String> {
		if !writable { return Err("STORE to read-only input".into()); }
		let i = usize::try_from(index).map_err(|_| "negative INDEX")?;
		if i >= self.len { return Err("INDEX exceeds buffer length".into()); }
		let mut next = self.clone();
		next.initialized.insert(i);
		Ok(next)
	}
	fn verify_load(&self, index: i32) -> Result<(),String> {
		let i = usize::try_from(index).map_err(|_| "negative INDEX")?;
		if i >= self.len { return Err("INDEX exceeds buffer length".into()); }
		if !self.initialized.contains(&i) { return Err("uninitialized LOAD".into()); }
		Ok(())
	}
}
```

ALLOC の集合は空で、`after_store` は現在の集合を変更せず次の state を返す。したがって s1 の位置 0 が初期化済みでも s0 の LOAD 0 は拒否される。BUFFER の入口では検証済みの全位置を初期化済みとして登録する。AFTER の循環は 01 の三色 DFS を state と効果の辺にも適用し、同じ旧 state に複数の書込み後継がある場合や、旧 state の読出しから上書きへの明示辺がない場合も拒否する。

この入力の途中経過を並べる。

```text
s0=allocated(3,4): initialized={}
s0.verify_load(0) -> uninitialized LOAD
s1=s0.after_store(0,true): initialized={0}
s1.verify_load(0) -> ok; s1.verify_load(1) -> uninitialized LOAD
after_store(-1,true) / after_store(3,true) / after_store(0,false) -> error
```

- [ ] dtype、アドレス kind、保存領域の可変性を検証する。
- [ ] SHRINK の `0 <= start <= end <= len` を検証する。
- [ ] 静的 INDEX が view の範囲内にあることを検証する。
- [ ] 未初期化出力の LOAD と AFTER の循環を拒否する。
- [ ] `len*itemsize` と offset の overflow をホスト上の checked 演算で拒否する。

予定する負例は `x[-1]`、`x[3]`、読取り専用入力への STORE。
静的な証明ができないアクセスは、まだ生成せず未対応エラーにする。
不正なポインタを生成 C に渡してから失敗を観測するテストにはしない。

## 完了確認

以下はこの章で新しく作る example とテストの予定コマンドである。

```sh
# 作業ディレクトリ: chibigrad/
cargo test
cargo run --example memory_scalar -- --case load-store --dump-source
cargo run --example memory_scalar -- --case views --dump-kernel
cargo run --example memory_scalar -- --case invalid-index
```

- [ ] 固定アクセスの結果が `31`、view の結果が `20` になる。
- [ ] 二入力の三要素加算が `[11,22,33]` になる。
- [ ] 書込み前後の LOAD が state を区別し、最終値 `2` を読む。
- [ ] dump の base identity は view 作成前後で一致する。
- [ ] accesses が入力を read、出力を write として列挙する。
- [ ] 範囲外、型不一致、未初期化、効果循環が実行前にエラーになる。

## tinygrad ソースとの対応

参照 commit は `c22edff14c79310a9b2df62a78819cf77c8289fa`。
各リンクの行は checkout で読んで確認した。

| 実在するシンボル | ソース | 役割 | chibigrad で簡略化する点 |
|---|---|---|---|
| `UOp.load`, `UOp.store`, `UOp.after` | [ops.py](../../tinygrad/tinygrad/uop/ops.py#L583) | 値と効果の依存を作る | 新規出力と直列 state に限る |
| `UOp.storage_base` | [ops.py](../../tinygrad/tinygrad/uop/ops.py#L753) | view と AFTER の保存元を求める | SHRINK と AFTER だけから始める |
| `UOp.addrspace` | [ops.py](../../tinygrad/tinygrad/uop/ops.py#L870) | 値と保存場所を区別する | ALU、GLOBAL、REG だけを受け付ける |
| `validate_index`, `type_verify` | [spec.py](../../tinygrad/tinygrad/uop/spec.py#L11) | アクセスとノードを検証する | Z3 は導入せず静的区間で検証する |
| `ProgramInfo.from_sink` | [ops.py](../../tinygrad/tinygrad/uop/ops.py#L1326) | 入出力の slot を収集する | 独自 accesses で StorageId ごとに収集する |
| `CStyleLanguage.render_index`, `render_access` | [cstyle.py](../../tinygrad/tinygrad/renderer/cstyle.py#L178) | アドレスと dereference を C にする | scalar の連続アクセスだけにする |

## 今は追加しないもの

動的な添字は [06-loops.md](06-loops.md) の RANGE bounds で扱う。
一般 movement は第 10 段階、メモリ再利用は第 19 段階で加える。
破壊的な Tensor 更新、atomic、非同期実行は今は加えない。
