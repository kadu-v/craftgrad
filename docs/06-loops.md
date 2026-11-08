# 06. RANGE と END から安全なループを生成する

## このステップで動かすもの

[05-memory-runtime.md](05-memory-runtime.md) の展開済み加算を一本のループにする。
`x=[1,2,3]`, `y=[10,20,30]` から `[11,22,33]` を返す。
次に二重ループで 2 行 3 列をコピーし、スコープと添字を確認する。
元計画第 4 章の単一ループとループ構造に対応する。
以下の IR、API、コマンド、数値は今後実装する予定である。

## 最小実装とデータ

`RANGE(bound,axis_id)` は i32 の `0 <= i < bound` を表す。
`END(effect,range)` は、その範囲の反復を終える void の効果にする。
値だけを END で閉じて loop 外へ持ち出すことは許さない。
ループ後に必要な値は STORE し、AFTER を通じて読む。

予定する kernel IR は次の形になる。

```text
%i = RANGE(CONST i32 3, axis=0)
%xv = LOAD(INDEX(%x,%i))
%yv = LOAD(INDEX(%y,%i))
%s = STORE(INDEX(%out,%i), ADD(%xv,%yv))
%e = END(%s,%i)
%result = AFTER(%out,%e)
```

予定する C の本体は次になる。

RANGE と END から生成する C の loop body である。Rust の関数へ貼らない。

```c
// src/renderer/c.rs が生成するコード。生成先: <一時ディレクトリ>/kernel.c
for (int32_t i = 0; i < 3; i++) {
	out[i] = x[i] + y[i];
}
```

最初は builder が構造化された body を持ち、そこから RANGE と END の列を作ってよい。
scalar の単純なトポロジカル順だけでループの入れ子を推測しない。
各命令の `open_ranges` と定義先の region を導出する。
linearize の際は body の開始、効果、終了を構造に沿って出力する。

軸 ID は関数内で一意とし、別関数の軸とは区別する。
範囲は静的な `0 <= N <= i32::MAX` から始める。
`N=0` は反復しない。RANGE の空区間を非空と見なさない。
添字は第三章の一般 scalar 整数とは別に、overflow しないことを検証する。
wrap した添字を小さな安全値と誤判定しない。
`row*cols+col` の最大値を checked な広いホスト整数で評価する。

## この章のファイル配置

構造化 Region と命令の列は linearize に追加し、反復ごとの値と効果の環境は interpret に置く。範囲とスコープの証明は verify へ集める。ループを確かめる std の main と、実 compiler を使う range_copy は別に保存する。

次はこの章の終わりまでに触る関連ファイルの配置案である。現在のcheckoutの一覧ではない。各コミットでは、その下に示す部分だけを追加・変更する。

```text
chibigrad/
├── examples/
│   ├── book_06_iteration_scope.rs  # 各反復の値と効果の環境を確認する std の独立例。
│   └── range_copy.rs  # --dump-kernel のスコープ表示。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # RANGE と END を for と閉じ括弧へ変換する。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── runtime/
│   │   ├── cpu.rs  # 長さ 0 を持つ出力所有物と引数の寿命。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── interpret.rs  # bound 0 で body を評価しない eval_loop。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── linearize.rs  # 命令ごとの open_ranges を dump に渡す。
│   ├── uop.rs  # RANGE、END と関数内で一意な AxisId。
│   └── verify.rs  # ScopeInst と verify_regions によるスコープ検証。
├── tests/
│   └── loops.rs  # 空、二重 loop とスコープ・overflow 違反の統合確認。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

生成Cやshaderは一時ディレクトリへ出力するため、このRustプロジェクトのツリーには含めない。コード冒頭のコメントで、生成元と生成先を確認する。

## 小さなコミットで進める

### コミット 1. 展開済みの三要素加算を単一 RANGE に変える

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_06_iteration_scope.rs  # [追加] 各反復の値と効果の環境を確認する std の独立例。
│   └── range_copy.rs  # [追加] RANGE を持つ三要素加算を実行する入口。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # [変更] RANGE と END を for と閉じ括弧へ変換する。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── interpret.rs  # [変更] eval_region と反復ごとの値・効果環境。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── linearize.rs  # [変更] Region::Loop と emit_region による構造化された順序。
│   ├── uop.rs  # [変更] RANGE、END と関数内で一意な AxisId。
│   └── verify.rs  # [変更] RANGE bounds とループ完了後の初期化の検証。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

Region と Inst は src/linearize.rs に追加し、既存 DFS の命令列と構造化した loop の出力を区別する。eval_region は interpret.rs に追加する。std の add_loop と main は examples/book_06_iteration_scope.rs へ保存し、range_copy.rs の main はグラフを作って従来の CPU 経路を呼ぶ。uop.rs の RANGE と END、verify.rs の範囲証明、renderer/c.rs の for 生成を同じ変更で接続する。


固定添字の三つの STORE を、同じ命令を三回実行する body に変える。候補ファイル `src/linearize.rs` の設計案 `Region::Loop { range, body }` と `emit_region`、`src/interpret.rs` の `eval_region` を足す。builder が RANGE i とそれに所属する LOAD、ADD、STORE を一つの body に保持し、END がその body を閉じる。graph の一般的な依存順だけから C の波括弧の位置を決めない。

RANGE 3 を interpreter が実行すると、i の環境は 0、1、2 と変わる。各反復で `x[i]` と `y[i]` を読み、その場で出力 i に書く。i=0 では 1 と 10 から 11、i=1 では 2 と 20 から 22、i=2 では 3 と 30 から 33 を得る。LOAD を最初の反復だけで評価して memo に固定すると、全出力が 11 になってしまう。ループに依存する値の表は反復単位で作るか、再評価する命令列として扱う。効果の実行済み集合も各 loop の呼出しと反復の中だけに持ち、NodeId の全体集合で STORE を一度限りにしない。

範囲証明では `0<=i<3` を INDEX へ伝え、長さ 3 の入力と出力へ収まると判定する。END は全反復の効果完了を返し、外側の AFTER(out,END) が最終出力を表す。局所テストは interpreter と C の `[11,22,33]` を比較し、source が一本の for と一つの STORE を持つことを調べる。下位の buffer ABI、StorageId、library 所有権をそのまま使い、ループを理由に入出力の契約を変えない。初期化済み領域の判定も、この単純な全範囲 STORE が完了した時点で出力全体へ広げる。


単独で実行できる例である。`src/interpret.rs` の反復スコープを、同じ LOAD・ADD・STORE の ID を再利用する小さい命令列で示す。例の ID 0、1、2 は値、3 は効果である。

std だけで反復ごとの環境を確かめる一つの example とする。

```rust
// examples/book_06_iteration_scope.rs（単独で実行する例）
use std::collections::{HashMap,HashSet};
fn add_loop(x: &[f32], y: &[f32]) -> (Vec<f32>,usize) {
	assert_eq!(x.len(),y.len());
	let mut out = vec![0.0;x.len()];
	let mut writes = 0;
	for i in 0..x.len() {
		// RANGE の一反復に入るたびに新しい環境を作る。
		let mut values = HashMap::<usize,f32>::new();
		let mut effects = HashSet::<usize>::new();
		values.insert(0,x[i]);
		values.insert(1,y[i]);
		values.insert(2,values[&0]+values[&1]);
		if effects.insert(3) {
			out[i] = values[&2];
			writes += 1;
		}
		assert!(!effects.insert(3)); // 同じ反復内の STORE だけは一回。
	}
	(out,writes)
}
fn main() {
	assert_eq!(add_loop(&[1.0,2.0,3.0],&[10.0,20.0,30.0]),(vec![11.0,22.0,33.0],3));
	assert_eq!(add_loop(&[],&[]),(vec![],0));
	assert_eq!(add_loop(&[2.0],&[10.0]),(vec![12.0],1));
}
```

`values` と `effects` の宣言が for の内側にあることが要点である。ID 3 が一度実行済みになっても、その記録は次の i へ持ち越されない。実装の `eval_region` ではこの短い固定命令列を region の命令列で置き換え、RANGE の bound と環境から添字を求める。外側の不変値を別の環境へ置く最適化は後で加えられるが、反復に依存する LOAD の結果を共有してはならない。

この入力の途中経過を並べる。

```text
iteration i=0: values[2]=11, effects={3}
iteration i=1: values={} から開始、values[2]=22, effects={3}
iteration i=2: values[2]=33
END: writes=3, AFTER(out,END) -> [11,22,33]
```

- [ ] RANGE と END を検証、interpreter、C renderer へ追加する。
- [ ] body に入ると添字を束縛し、END でスコープを閉じる。
- [ ] LOAD と STORE の範囲を RANGE bounds から証明する。
- [ ] 三要素加算を既存の固定 ABI で実行する。
- [ ] 第五章の固定添字の例も引き続き通す。

予定出力は `[11,22,33]`。
生成 C に STORE 三つの展開がなく、一本の for があることを確認する。

### コミット 2. 二重ループと内側の添字を動かす

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── range_copy.rs  # [変更] row と col の二重 loop の builder。
├── src/
│   ├── interpret.rs  # [変更] 内側ループへ入るごとの新しい環境。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── linearize.rs  # [変更] RegionId、open_ranges と入れ子の emit_region。
│   └── verify.rs  # [変更] checked_row_offset と広いホスト整数での添字証明。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

emit_region の抜粋は src/linearize.rs に置き、RegionId と open_ranges もこの構造に対応させる。checked_row_offset は添字の安全性を証明する helper として src/verify.rs に置く。examples/range_copy.rs に二重 loop のグラフを追加し、interpret.rs の内側環境と生成 C の両方で六要素のコピーを照合する。


単一 body の中へ別の loop region を入れ、各軸の有効範囲を明示する。候補ファイル `src/linearize.rs` の設計案 `RegionId` と `open_ranges(node)` を使い、row の loop body 内に col の loop body を作る。命令列は RANGE row、RANGE col、offset の演算、LOAD、STORE、END col、END row の順となる。RANGE の axis_id は同じ関数内で別番号にし、境界値が同じでも軸を intern して一つにしない。

offset は `row*3+col` とする。row=0 では col=0、1、2 から位置 0、1、2 を読み、row=1 では位置 3、4、5 を読む。内側 loop に入るたびに col を 0 へ戻すため、前の行の col=3 を次の行へ持ち越さない。値の依存する open_ranges は offset と LOAD では `{row,col}`、内側 END の後の効果では `{row}` となる。外側 END の後にはどちらの軸も自由な値として残らない。

添字証明は row の最大 1 と col の最大 2 から `1*3+2=5` をホストの広い整数で求め、長さ 6 未満と確認する。この計算自体を I32 の wrap 演算に任せない。局所テストでは `[1,2,3,4,5,6]` のコピー結果を全要素照合し、kernel dump の END が col、row の順に閉じることを見る。C の二つの for の入れ子と interpreter の環境を対応させれば、値が合わない際に軸の束縛とアドレスのどちらが誤ったかを分けて調べられる。高水準 shape を使わずに、既存の region と scalar ALU だけでこの動作を組み立てる。


組込み用の抜粋である。`src/linearize.rs` の構造化 region を順序列へ出す。`Region` は `Loop { axis, bound, body }` と `Effect(NodeId)`、`Inst` は `Range`、`Effect`、`End` を持つ一時的な enum であり、最終的には同じ順の UOp にする。

region の出力と添字の安全性検査を分けて追加する。

```rust
// src/linearize.rs（既存ファイルへ組み込む抜粋）
fn emit_region(region: &Region, out: &mut Vec<Inst>) {
	match region {
		Region::Effect(id) => out.push(Inst::Effect(*id)),
		Region::Loop { axis,bound,body } => {
			out.push(Inst::Range { axis:*axis, bound:*bound });
			for child in body { emit_region(child,out); }
			out.push(Inst::End(*axis));
		}
	}
}
// src/verify.rs（既存ファイルへ組み込む抜粋）
fn checked_row_offset(row: usize, col: usize, cols: usize, len: usize) -> Result<usize,String> {
	if col >= cols { return Err("column outside row".into()); }
	let offset = row.checked_mul(cols).and_then(|n| n.checked_add(col)).ok_or("offset overflow")?;
	if offset >= len || offset > i32::MAX as usize { return Err("index may exceed buffer length".into()); }
	Ok(offset)
}
```

外側 Loop を処理する間に内側 Loop の開始と終了が両方出力されるので、END は自然に col、row の順になる。`checked_row_offset` は scalar の wrap 演算と違い、ホストの checked 演算で安全性を確認する。列の範囲と総容量を別々に調べることで、row 1 の col 3 がたまたま領域内に見える状況も拒否する。内側 body の値環境は row が変わるたびにも作り直す。

この入力の途中経過を並べる。

```text
RANGE row=0..2
  RANGE col=0..3
    offset=row*3+col; LOAD x[offset]; STORE out[offset]
  END col
END row
row0 offsets=0,1,2; row1 offsets=3,4,5
```

- [ ] builder に外側 body 内で内側 RANGE を作る操作を加える。
- [ ] END が内側、外側の順に閉じることを検証する。
- [ ] `offset = row*3+col` を作り、各反復で LOAD と STORE を行う。
- [ ] `[1,2,3,4,5,6]` のコピーが同じ六要素を返すようにする。

形状付きグラフはまだ存在しない。
2 行 3 列という情報は example が持ち、要素位置を手で組み立てる。
内側の添字を外側の次の行で再利用しないことを確認する。

### コミット 3. 空と一要素の範囲を実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── range_copy.rs  # [変更] 空配列と一要素の実行入力。
├── src/
│   ├── runtime/
│   │   ├── cpu.rs  # [変更] 長さ 0 を持つ出力所有物と引数の寿命。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── interpret.rs  # [変更] bound 0 で body を評価しない eval_loop。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── verify.rs  # [変更] AxisBounds、range_bounds と空範囲の添字証明。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

AxisBounds と二つの検証関数を src/verify.rs に追加し、通常区間から空範囲を分ける。interpret.rs の eval_loop は同じ bound の制御で空を扱う。runtime/cpu.rs では長さ 0 の所有物も保持し、range_copy の empty ケースは実際に生成 kernel を呼んで比較する。


空の反復を特別な実行成功値で代用せず、同じ loop の制御で扱う。候補ファイル `src/interpret.rs` の設計案 `eval_loop` は bound が 0 なら body を一度も評価せず END の完了を返す。C でも `for(i=0;i<N;i++)` の条件を維持し、N=0 のとき INDEX と LOAD へ到達しない。N=1 なら i=0 の一回だけであり、終了時の i=1 を body 内の有効値として使わない。

空出力のホスト表現には、長さ 0 を記録した所有物を作る。固定 ABI に pointer slot が必要でも、その pointer に対する dereference は生成される body の実行で一度も行わない。検証では空範囲に対して最大添字 N-1 を通常の unsigned 減算で作らず、空と非空を区別した区間を返す。空 body の「有効なアクセス範囲」を `[0,-1]` のような逆転した普通の区間に押し込まない。

局所テストは N=0 の空配列と N=1 の x=[2]、y=[10] を使う。interpreter のメモリアクセス箇所に置く検査用計数では空入力の LOAD と STORE がともに 0、一要素では出力が `[12]` となる予定である。C の結果も照合し、空配列のために架空の一要素を返さない。以前の N=3 と二重 loop を同じ比較関数へ通せば、空の例だけの分岐が既存処理を壊していないかを確認できる。後の累積では空 loop の外に初期化があるため、空の総和を作る前提にもなる。


組込み用の抜粋である。`src/verify.rs` が扱う軸区間に、空を専用の variant として加える。普通の区間の最大値を 0-1 で作らず、非空のときだけ最大添字を検査する。

空と非空の範囲を表す型と検証関数を追加する。

```rust
// src/verify.rs（既存ファイルへ組み込む抜粋）
enum AxisBounds { Empty, Nonempty { lo: i64, hi: i64 } }
fn range_bounds(n: i64) -> Result<AxisBounds,String> {
	if !(0..=i32::MAX as i64).contains(&n) { return Err("invalid RANGE bound".into()); }
	if n == 0 { Ok(AxisBounds::Empty) } else { Ok(AxisBounds::Nonempty { lo:0,hi:n-1 }) }
}
fn verify_identity_index(bounds: AxisBounds, len: usize) -> Result<(),String> {
	match bounds {
		AxisBounds::Empty => Ok(()),
		AxisBounds::Nonempty { lo,hi } => {
			if lo < 0 || hi as u64 >= len as u64 { return Err("index may exceed buffer length".into()); }
			Ok(())
		}
	}
}
```

空軸では実行されるアクセスが存在しないため、identity INDEX の証明はその枝で完了する。これは不正な dtype や閉じていない RANGE を無視する意味ではなく、アクセスの値域だけの規則である。`N=1` の区間は `[0,0]` で、一度だけ位置 0 を使う。空配列のホスト所有物も作り、生成 C の for 条件が false になることで dereference が実行されないことを確認する。

この入力の途中経過を並べる。

```text
N=0 -> AxisBounds::Empty -> body executions=0 -> writes=0
N=1 -> Nonempty{lo:0,hi:0} -> x[0]+y[0]=12
N=3 -> Nonempty{lo:0,hi:2} -> max index 2 < len3
N=-1 -> invalid RANGE bound
```

- [ ] RANGE 0 と RANGE 1 を interpreter と C で実行する。
- [ ] 空の body が入力を一度も LOAD しないようにする。
- [ ] 空出力には有効なホスト所有物を用意し、ポインタを dereference しない。
- [ ] 一要素加算 `2+10` を生成コードから `12` として読む。

`N=0` の結果は空配列、書込み回数は `0` が予定値になる。
境界を `i<=N` にした実装は、一要素例でも範囲外になり検証で止まる。

### コミット 4. スコープ違反と添字 overflow を拒否する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── range_copy.rs  # [変更] --dump-kernel のスコープ表示。
├── src/
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── linearize.rs  # [変更] 命令ごとの open_ranges を dump に渡す。
│   └── verify.rs  # [変更] ScopeInst と verify_regions によるスコープ検証。
├── tests/
│   └── loops.rs  # [追加] 空、二重 loop とスコープ・overflow 違反の統合確認。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

ScopeInst と verify_regions の抜粋を src/verify.rs に追加し、linearize.rs が持つ構造化順序から検証列を渡す。kernel dump は range_copy.rs の入口で各命令と開いている軸を表示する。tests/loops.rs に公開実行経路を通る成功例と生成前に拒否される負例を置く。


region を持った builder が通常は正しい構造を作るとしても、UIR や変換後のグラフは境界で検証する。候補ファイル `src/verify.rs` の設計案 `verify_regions` は、構造化順序をたどりながら開いた RANGE の stack を持つ。RANGE で push、END で同じ最上位軸を pop し、末尾に残りがあれば未終了とする。別の軸を閉じる END は交差、すでに閉じた軸を閉じる END は重複として拒否する。

各値の定義 region と利用 region も調べる。row 内で作った col はその内側 body でだけ使え、END col 後の STORE が col の値を参照していたら `range escapes scope` とする。外側で作った定数や入力 BUFFER を内側で使うことは許す。END は効果を閉じる操作なので、col を値として返す手段にはしない。`--dump-kernel` には命令と open_ranges を並べ、誤った fixture のどこで軸が漏れたかを読めるようにする。

添字の検証は region の束縛から範囲を得て行う。長さ N に i+1 を使うと、非空の最大値が N となるため拒否する。row*cols+col の最大値を checked な広いホスト整数で求め、I32 に収まらないなら C を生成しない。局所テストでは scope 漏れ、交差 END、重複 END、未終了、i+1、積の overflow を独立に作り、C コンパイルのエラーへ流れないことを確認する。正しい単一・二重・空 loop もこの検証を通し、構造と値の両方を同じ pipeline で確認する。


組込み用の抜粋である。`src/verify.rs` のスコープ検査で `open` を一つの stack として扱う。`ScopeInst` は `Range(axis)`、`Use { required }`、`End(axis)` の検証用列で、required は値の導出時に集めた自由軸集合である。

構造化した命令列の scope 検査へ追加する。

```rust
// src/verify.rs（既存ファイルへ組み込む抜粋）
fn verify_regions(code: &[ScopeInst]) -> Result<(),String> {
	let mut open = Vec::new();
	let mut declared = std::collections::HashSet::new();
	for inst in code {
		match inst {
			ScopeInst::Range(axis) => {
				if !declared.insert(*axis) { return Err("duplicate axis id".into()); }
				open.push(*axis);
			}
			ScopeInst::Use { required } => {
				if required.iter().any(|a| !open.contains(a)) { return Err("range escapes scope".into()); }
			}
			ScopeInst::End(axis) => {
				if open.pop() != Some(*axis) { return Err("crossed or duplicate END".into()); }
			}
		}
	}
	if !open.is_empty() { return Err("unclosed RANGE".into()); }
	Ok(())
}
```

`declared` は同一関数の軸 ID の重複を止め、`open` は現在の利用可能性を表す。内側 END の後で required に col が含まれる命令はそこで拒否される。整数値の bounds はこれと別に 03 から導出し、添字の `row*cols+col` は 06 の checked な広い整数で証明する。値域が安全でもスコープが違う軸は使えない。

この入力の途中経過を並べる。

```text
Range(row),Range(col),Use{row,col},End(col),Use{row},End(row) -> ok
Range(row),Range(col),End(row) -> crossed END
Range(col),End(col),Use{col} -> range escapes scope
Range(row) のみ -> unclosed RANGE
```

- [ ] 閉じた RANGE の値を別の body で使うグラフを拒否する。
- [ ] 交差する END、重複 END、未終了 RANGE を拒否する。
- [ ] `INDEX(x,i+1)` が N 要素を越える場合を生成前に拒否する。
- [ ] bounds の積が i32 を越えるケースを拒否する。
- [ ] `--dump-kernel` に各命令の開いている軸を出す。

予定するエラーは `range escapes scope` と `index may exceed buffer length`。
グラフが誤っているときは、C の構文エラーまで進めない。

## 完了確認

以下はこの章で新しく作る example とテストの予定コマンドである。

```sh
# 作業ディレクトリ: chibigrad/
cargo test
cargo run --example range_copy -- --case add3 --dump-source
cargo run --example range_copy -- --case nested --dump-kernel
cargo run --example range_copy -- --case empty
```

- [ ] 単一ループの結果が `[11,22,33]` になる。
- [ ] 二重ループのコピーが `[1,2,3,4,5,6]` になる。
- [ ] 空は空配列、一要素は `[12]` を返す。
- [ ] END の直前には対応する RANGE が開いている。
- [ ] 最外側 END の後に未束縛の軸が残らない。
- [ ] すべてのメモリアクセスを bounds と保存領域長から証明できる。
- [ ] 範囲外、スコープ違反、添字 overflow は実行前に失敗する。

C の for の見た目だけでは完了としない。
生成コードを共有 library として呼び、実際の出力を比較する。

## tinygrad ソースとの対応

参照 commit は `c22edff14c79310a9b2df62a78819cf77c8289fa`。
リンクの行は checkout で読んで確認した。

| 実在するシンボル | ソース | 役割 | chibigrad で簡略化する点 |
|---|---|---|---|
| `UOp.range`, `UOp.end` | [ops.py](../../tinygrad/tinygrad/uop/ops.py#L613) | 軸と反復終了を構築する | 静的な有界ループだけにする |
| `spec_shared` の RANGE と END 規則 | [spec.py](../../tinygrad/tinygrad/uop/spec.py#L81) | bound と void 効果の終了を検証 | 構造化 body と対応表を検証する |
| `CFGContext` | [linearizer.py](../../tinygrad/tinygrad/codegen/late/linearizer.py#L50) | 軸の入れ子と順序を解析する | 明示的な入れ子をそのまま使う |
| `linearize` | [linearizer.py](../../tinygrad/tinygrad/codegen/late/linearizer.py#L8) | スコープを考慮して順序を作る | 汎用優先度は追加しない |
| `base_rewrite` の RANGE と END | [cstyle.py](../../tinygrad/tinygrad/renderer/cstyle.py#L18) | for と閉じ括弧を出す | C の逐次ループに固定する |

## 今は追加しないもの

累積、行和、手書き GEMM と convolution は [07-scalar-reductions.md](07-scalar-reductions.md) で加える。
shape からループを作る rangeify は [08](08-shapes-rangeify.md) まで導入しない。
while、break、動的 shape、ループ最適化は後の段階へ送る。
