# 01. 定数 7 を C にして実行し、加算と乗算で 20 を返す

## このステップで動かすもの

Rust で作った scalar グラフから C を生成し、system clang で実行ファイルを作る。
最初のコミットは `CONST(7.0f32)` 一つで、標準出力の予定値は `7`。
次のコミットで `(2.0 + 3.0) * 4.0` を作り、予定値 `20` を得る。
実行時入力は [02-runtime-inputs.md](02-runtime-inputs.md) で加える。
元計画の「段階的な実装手順」第 1 章を、最初の実行までさらに分割する。

この文書は未実装の作業計画である。
コード片、コマンド、例名、数値は今後実装して確認する予定を示す。
既存の chibigrad にこれらの API が存在すると仮定しない。
初回は Rust と system clang の利用可能性を確認してから始める。

## 最小実装とデータ

最初に必要なのは、ノード一つと root の番号だけである。
`NodeId` と `Vec<UOp>` を同じ Rust ファイルに置いてよい。
最初の `Op` は `Const(f32)` 一種類にする。
ノード共有、汎用 visitor、backend trait はまだ作らない。

予定する最初の入力 IR は次の形になる。

```text
%0 = CONST f32 7
root = %0
```

予定する出力 C は、scalar 式を返す関数と wrapper で構成する。

examples/scalar_const.rs の render_c が生成する C の予定形である。Rust ソースへ貼らず、一時ディレクトリの kernel.c として確認する。

```c
// examples/scalar_const.rs が生成するコード。生成先: <一時ディレクトリ>/kernel.c
#include <stdio.h>
static float chibi_value(void) {
	return 7.0f;
}
int main(void) {
	printf("%.9g\n", (double)chibi_value());
	return 0;
}
```

`rewrite` は root をそのまま返す関数でよい。
`linearize` は root の依存先を先に並べる関数にする。
初回から `graph → rewrite → linearize → render C → clang → run` を通す。
空の別モジュールを先に作る作業は含めない。
次に `Add` と `Mul` を追加し、型はすべて `f32` に固定する。
最適化はせず、生成 C に加算と乗算が残ることを確かめる。

## この章のファイル配置

この章は examples 内の小プログラムから始める。scalar_const と scalar_arithmetic は別の main を持ち、共通処理の src への抽出は実行時入力が必要になる 02 で行う。ここで示すツリーは今後の編集先であり、既存の Cargo.toml と src/main.rs を変更する指示ではない。

次はこの章の終わりまでに触る関連ファイルの配置案である。現在のcheckoutの一覧ではない。各コミットでは、その下に示す部分だけを追加・変更する。

```text
chibigrad/
├── examples/
│   ├── scalar_arithmetic.rs  # グラフ検証、dump、外部プロセスのエラー表示。
│   └── scalar_const.rs  # 同じ検証とエラー処理を通す CONST 7 の例。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

生成Cやshaderは一時ディレクトリへ出力するため、このRustプロジェクトのツリーには含めない。コード冒頭のコメントで、生成元と生成先を確認する。

## 小さなコミットで進める

### コミット 1. CONST 7 をビルドして、その場で実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── scalar_const.rs  # [追加] CONST 7 のグラフ構築から clang 実行までを持つ最初の小プログラム。
└── Cargo.toml  # [確認] 既存の package 設定を確認し、examples の自動検出を使う。
```

examples/scalar_const.rs を追加し、掲載した単独例の全体を置く。NodeId、Op、UOp、Graph、恒等 rewrite、linearize、render_c と main を同じファイルに置く。本文の二候補から examples 内だけで動く案を選ぶ。この時点で src/lib.rs や空のモジュールは追加しない。


最初に、`examples/scalar_const.rs` か同じファイルから呼ぶ `src/lib.rs` に、設計案の `NodeId(usize)`、`UOp { op: Op, src: Vec<NodeId> }`、`Graph { nodes, root }` を置く。最初の Op は `Op::Const(f32)` とし、その src は空にする。`build_const(7.0)` は `nodes[0]` を作り root に 0 を返すだけにする。ここでの入力は Rust が読む example 引数であり、clang が作るプログラムには入力口を用意しない。まず `rewrite(root)` が同じ番号を返し、`linearize(root)` が `[0]` を返すところを確かめる。この二つを省略せず、後で処理を足す場所を実際の実行経路に置く。

次に設計案の `render_c(graph, order)` が順序列を読み、`chibi_value` と `main` を含む文字列を返す。定数 7 は `7.0f` として出せる。`compile_and_run(source)` は一時ディレクトリに `kernel.c` を書き、`Command` の個別引数として source path と `-o` と出力 path を clang に渡す。コンパイル成功後にだけ実行ファイルを起動し、終了コードと stdout を Rust へ返す。Rust 自身で 7 を計算して表示するだけでは、この変更の確認にならない。

処理を追うと、`7` という引数、`CONST(7)` のノード、順序 `[0]`、`return 7.0f`、子プロセスの stdout `7\n` の順になる。局所テストでは stdout を数値へ読み直して 7 と照合し、source に返却式と wrapper があることも見る。最初は C の特殊な浮動小数点定数まで広げず、この有限値一つを端から端まで通す。一時ディレクトリの所有物を関数末尾まで保持すれば、コンパイル直後に実行ファイルが消える誤りも避けられる。


単独で実行できる例である。最初の編集先 `examples/scalar_const.rs` の全体をこの大きさから始める。標準ライブラリだけで Rust の入口から生成実行ファイルの stdout まで通る。

このブロック全体を一つの example として保存する。

```rust
// examples/scalar_const.rs（単独で実行する例）
use std::{fs, process::Command};
#[derive(Clone, Copy)]
struct NodeId(usize);
enum Op { Const(f32) }
struct UOp { op: Op, src: Vec<NodeId> }
struct Graph { nodes: Vec<UOp>, root: NodeId }
fn rewrite(root: NodeId) -> NodeId { root }
fn linearize(g: &Graph, root: NodeId) -> Vec<NodeId> {
	assert!(g.nodes[root.0].src.is_empty());
	vec![root]
}
fn render_c(g: &Graph, order: &[NodeId], root: NodeId) -> String {
	let mut c = String::from("#include <stdio.h>\nstatic float chibi_value(void) {\n");
	for &NodeId(id) in order {
		match g.nodes[id].op {
			Op::Const(x) => c.push_str(&format!("\tfloat v{id} = {x:.1}f;\n")),
		}
	}
	c.push_str(&format!("\treturn v{};\n}}\n", root.0));
	c.push_str("int main(void) { printf(\"%.9g\\n\", (double)chibi_value()); return 0; }\n");
	c
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
	let g = Graph { nodes: vec![UOp { op: Op::Const(7.0), src: vec![] }], root: NodeId(0) };
	let root = rewrite(g.root);
	let order = linearize(&g, root);
	let c = render_c(&g, &order, root);
	let dir = std::env::temp_dir().join(format!("chibi-scalar-{}", std::process::id()));
	fs::create_dir(&dir)?;
	let result = (|| -> Result<(), Box<dyn std::error::Error>> {
		let source = dir.join("kernel.c");
		let binary = dir.join("kernel");
		fs::write(&source, &c)?;
		let mut compiler = Command::new("clang");
		#[cfg(target_os="macos")]
		{
			let sdk = Command::new("xcrun").arg("--show-sdk-path").output()?;
			if !sdk.status.success() { return Err("xcrun could not locate the macOS SDK".into()); }
			let sdk = String::from_utf8(sdk.stdout)?;
			let sdk = sdk.trim();
			compiler.arg("-isysroot").arg(sdk).arg(format!("--sysroot={sdk}"));
		}
		let compile = compiler.arg(&source).arg("-o").arg(&binary).output()?;
		if !compile.status.success() {
			return Err(format!("clang: {}", String::from_utf8_lossy(&compile.stderr)).into());
		}
		let run = Command::new(&binary).output()?;
		if !run.status.success() { return Err(format!("run: {}", run.status).into()); }
		let stdout = String::from_utf8(run.stdout)?;
		assert_eq!(stdout.trim().parse::<f32>()?, 7.0);
		print!("{stdout}");
		Ok(())
	})();
	fs::remove_dir_all(&dir)?;
	result
}
```

`Graph` の root は 0、恒等 `rewrite` の戻り値も 0 である。`linearize` が返す番号を `render_c` が変数 `v0` に変える。`Command::output` は子プロセス終了まで待つため、`dir` を削除するのは実行と stdout の照合の後になる。この例の renderer は有限値 7 専用であり、特殊値の出力規則は 03 で加える。macOS では `xcrun --show-sdk-path` から SDK を取得し、ヘッダ用の `-isysroot` と linker 用の `--sysroot` の両方を clang に渡すため、Homebrew clang でも標準ヘッダを見つけられる。保存した本文のこのブロックを `rustc --edition=2021` でビルドして実行できる。

この入力の途中経過を並べる。

```text
Graph: nodes[0] = UOp { op: Const(7.0), src: [] }, root=0
rewrite: 0 -> 0
linearize: [0]
C: float v0 = 7.0f; return v0;
clang -> kernel -> stdout: 7
```

- [ ] Cargo package と `scalar_const` example を必要な分だけ追加する。
- [ ] `Const(f32)` と root を保持し、一つのノードを構築する。
- [ ] 恒等 rewrite と依存順 linearize を経由して C を生成する。
- [ ] 一時ディレクトリに C を書き、system clang を子プロセスとして起動する。
- [ ] 生成実行ファイルを実行し、成功終了と標準出力 `7` を照合する。

このコミットで予定コマンド `cargo run --example scalar_const -- 7` を通す。
引数 `7` は example が定数グラフを作る値で、生成コードの実行時入力ではない。
子プロセスは引数配列で起動し、clang の終了コードと stderr を保存する。

### コミット 2. ADD と MUL を通して 20 を返す

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── scalar_arithmetic.rs  # [追加] ADD と MUL を持つ定数式の実行例。
│   └── scalar_const.rs  # [確認] 最初の CONST 7 の独立した実行例。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

scalar_const.rs の小さな構成を別の examples/scalar_arithmetic.rs へ写し、こちらの Op に Add と Mul を追加する。掲載した linearize でこのファイルの関数を置き換え、c_expression を同じファイルに追加して render_c の右辺から呼ぶ。main では (2+3)*4 を構築する。scalar_const の main と連結しない。


前コミットには依存のない CONST しかないため、今回は依存を持つグラフを同じ生成経路へ入れる。変更先は引き続き最初の Rust ファイルでよく、設計案の `Op::Add` と `Op::Mul`、入力列を返す `sources(id)` を加える。二つの入力番号は UOp の src に順序付きで置く。`build_arithmetic(2,3,4)` は `%0=2`、`%1=3`、`%2=ADD(0,1)`、`%3=4`、`%4=MUL(2,3)` を作り、root を 4 にする。ここでノード番号の昇順を実行順の規則にしてしまわない。

`linearize` は root から深さ優先に入力をたどり、全入力を処理した後でそのノードを順序列へ足す。これを後行順と呼ぶ。処理済み集合を `NodeId` で持つことで、共有される入力を二回出力しない。この例では `[0,1,2,3,4]` になり、renderer は先に `float v2 = v0 + v1`、次に `float v4 = v2 * v3` を出せる。式を再帰的に埋め込む実装でも数値は合うが、共有ノードを一度計算することを観察できる一時変数の形から始めるとよい。

中間値は `v0=2`、`v1=3`、`v2=5`、`v3=4`、`v4=20` になる。局所テストには `%2` を左右両方に使う `MUL(%2,%2)` も置き、順序列内の `%2` が一つだけであることを調べる。この時点では hash-consing はなく、同じ番号を参照する共有だけを扱う。source に `+` と `*` が残ること、CONST 7 の経路も通ることを確認し、Rust 側の定数畳み込みでこの観察を消さない。


組込み用の抜粋である。`Op` に `Add` と `Mul` を足した後、前の `linearize` を次の DFS に置き換え、renderer の各ノードの式もここで選ぶ。`NodeId`、`Graph`、`UOp` はコミット 1 の型を使う。

linearize を置き換え、c_expression を追加する。両方ともこの章では同じ example 内に置く。

```rust
// examples/scalar_arithmetic.rs（既存ファイルへ組み込む抜粋）
fn linearize(g: &Graph, root: NodeId) -> Vec<NodeId> {
	fn visit(g: &Graph, id: NodeId, done: &mut [bool], out: &mut Vec<NodeId>) {
		if done[id.0] { return; }
		for &child in &g.nodes[id.0].src { visit(g, child, done, out); }
		done[id.0] = true;
		out.push(id);
	}
	let mut out = Vec::new();
	visit(g, root, &mut vec![false; g.nodes.len()], &mut out);
	out
}
fn c_expression(node: &UOp) -> String {
	match node.op {
		Op::Const(x) => format!("{x:.1}f"),
		Op::Add => format!("v{} + v{}", node.src[0].0, node.src[1].0),
		Op::Mul => format!("v{} * v{}", node.src[0].0, node.src[1].0),
	}
}
```

DFS は入力をすべて訪ねてから現在の番号を追加する。`done` を番号ごとに持つので、`MUL(%2,%2)` の二本目の辺は `%2` を再出力しない。renderer はコミット 1 の `float v{id} = ...;` の右辺に `c_expression` の戻り値を使う。これで演算の入力と生成変数の対応が保たれ、式の共有が C でも一回の計算になる。

この入力の途中経過を並べる。

```text
%0=CONST 2; %1=CONST 3; %2=ADD %0 %1; %3=CONST 4; %4=MUL %2 %3
DFS postorder: [0,1,2,3,4]
v2 = v0 + v1 = 5
v4 = v2 * v3 = 20
MUL(%2,%2) の場合: %2 の出力回数=1、結果=25
```

- [ ] `Add` と `Mul` に二つの `NodeId` を持たせる。
- [ ] `Const(2)`, `Const(3)`, `Add`, `Const(4)`, `Mul` を組み立てる。
- [ ] 依存先を利用前に一度だけ並べる。
- [ ] C renderer に加算と乗算を足し、生成コードを実行する。
- [ ] 第 1 コミットの定数の実行例も再実行する。

予定する生成式は `((2.0f + 3.0f) * 4.0f)`、結果は `20`。
root の型は `f32` のままにする。
この時点で定数畳み込みは行わない。

### コミット 3. 同じ経路の失敗と dump を見えるようにする

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── scalar_arithmetic.rs  # [変更] グラフ検証、dump、外部プロセスのエラー表示。
│   └── scalar_const.rs  # [変更] 同じ検証とエラー処理を通す CONST 7 の例。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

verify_graph の抜粋は examples/scalar_arithmetic.rs に追加し、生成前に呼ぶ。この章では本文の src/verify.rs 候補を採らず、dump_graph、dump_source とエラー型も example 内へ置く。scalar_const.rs にも必要な検証と dump を反映し、各ファイル内の #[cfg(test)] テストで不正 root と入力数を確かめる。02 で実際に共用するときにこれらを src へ移す。


ここまでの成功経路を、失敗の場所まで追えるようにする。最初のファイル内の関数を使ってよいので、設計案の `verify_graph`、`dump_graph`、`dump_source` と、生成・コンパイル・実行を区別するエラー型を足す。`verify_graph` は arena を読む前に root が長さ未満かを調べ、各参照先も確認する。CONST の src は空、ADD と MUL の src は長さ 2 に限定する。入力列の長さを確認してから renderer が src[0] と src[1] を読む順にすれば、不足時の配列アクセスを避けられる。

走査には未訪問、訪問中、完了の状態を使う案にすれば、訪問中ノードへの戻りを循環として検出できる。正しい DAG では入力を先に表示した dump と linearize の対応を読める。`--dump-source` は renderer の文字列を別に再生成せず、実際にファイルへ書いた文字列を出す。clang を起動できなかった場合と、起動した clang が非ゼロで終了した場合を分け、後者には stderr を添える。実行ファイルの非ゼロ終了もコンパイル成功とは別の失敗である。

局所テストでは root=99 の一ノードグラフを渡し、clang の呼出し回数が 0 のまま拒否されることを調べる。入力不足を表現する fixture と、存在しない clang path も独立に試す。成功時は `(2+3)*4` の dump の ADD 参照が MUL に入り、実際の stdout から 20 を読めることを確認する。これで値が違う場合に、グラフ構築、順序列、C、外部プロセスのどこから調べるかを判断できる。


組込み用の抜粋である。`src/verify.rs` の候補として、既存 `Graph` に対する入力数・番号・循環の検査を一つの関数にする。ここで `Op` は Const、Add、Mul の三種類である。

この章では verify_graph を example 内へ追加する。02 の抽出後に同じ関数を src/verify.rs へ移す。

```rust
// examples/scalar_arithmetic.rs（既存ファイルへ組み込む抜粋）
fn verify_graph(g: &Graph) -> Result<(), String> {
	fn visit(g: &Graph, id: NodeId, mark: &mut [u8]) -> Result<(), String> {
		let node = g.nodes.get(id.0).ok_or_else(|| format!("missing node {}", id.0))?;
		if mark[id.0] == 1 { return Err(format!("cycle at {}", id.0)); }
		if mark[id.0] == 2 { return Ok(()); }
		let arity = match node.op { Op::Const(_) => 0, Op::Add | Op::Mul => 2 };
		if node.src.len() != arity {
			return Err(format!("node {}: expected {arity} sources", id.0));
		}
		mark[id.0] = 1;
		for &src in &node.src { visit(g, src, mark)?; }
		mark[id.0] = 2;
		Ok(())
	}
	visit(g, g.root, &mut vec![0; g.nodes.len()])
}
```

`get` が root=99 も子の欠落も同じ入口で拒否する。入力数を調べた後にだけ再帰するため、後段は `src[0]` と `src[1]` を安全に読むことができる。印 1 は今回の探索路上、2 は完了であり、共有された完了ノードと循環を区別する。pipeline の最初に `verify_graph(&g)?` を置き、成功した同じ `c` 文字列を dump とファイル書込みの双方へ渡す。

この入力の途中経過を並べる。

```text
root=99 -> missing node 99 -> clang calls=0
ADD src=[0] -> expected 2 sources -> clang calls=0
%0=ADD(%1,%1), %1=ADD(%0,%0) -> cycle at 0
正しい DAG -> verify -> rewrite -> DFS -> C -> compile -> run
```

- [ ] `--dump-graph` でノード番号、演算、入力を出す。
- [ ] `--dump-source` で clang に渡した C を出す。
- [ ] root の番号と演算の入力数を生成前に検証する。
- [ ] clang 不在とコンパイル失敗を区別する。
- [ ] stdout の値を読み、予定値と比較する実行テストを作る。

予定する負例は「存在しない root」「Add の入力不足」「clang 起動失敗」。
一時ディレクトリの寿命は、実行ファイルの終了まで保つ。
成功例は必ず同じ pipeline を通し、Rust で予定値だけを返す分岐を作らない。

## 完了確認

以下はこの章で新しく作る example とテストの予定コマンドである。

```sh
# 作業ディレクトリ: chibigrad/
cargo test
cargo run --example scalar_const -- 7 --dump-source
cargo run --example scalar_arithmetic -- 2 3 4 --dump-graph --dump-source
```

- [ ] `scalar_const` が生成コードを実行して `7` を返す。
- [ ] `scalar_arithmetic` が定数式 `(2+3)*4` を構築して `20` を返す。
- [ ] グラフで `ADD` の結果を `MUL` が参照している。
- [ ] source に実際の `+` と `*` が残っている。
- [ ] 不正な root と入力数は clang の起動前にエラーになる。
- [ ] 初回から恒等 rewrite と linearize を通過する。

`7` と `20` は `f32` で正確に表せるため、一致を直接確認する。
性能、共有ライブラリ化、ノード数の削減は完了条件にしない。

## tinygrad ソースとの対応

参照 commit は `c22edff14c79310a9b2df62a78819cf77c8289fa`。
以下の行番号は、その checkout の実装を読んで確認した。

| 実在するシンボル | ソース | 役割 | chibigrad で簡略化する点 |
|---|---|---|---|
| `UOp.const`, `UOp.alu` | [ops.py](../../tinygrad/tinygrad/uop/ops.py#L598) | 定数と演算ノードを作る | 明示的 `f32` と `Vec` の番号だけにする |
| `linearize` | [linearizer.py](../../tinygrad/tinygrad/codegen/late/linearizer.py#L8) | 依存と優先度から命令順を作る | scalar DAG の依存順走査にする |
| `CStyleLanguage.code_for_op` | [cstyle.py](../../tinygrad/tinygrad/renderer/cstyle.py#L152) | ALU を C の式へ変換する | `ADD` と `MUL` から始める |
| `do_render`, `do_compile` | [codegen](../../tinygrad/tinygrad/codegen/__init__.py#L468) | source と binary をつなぐ | example から関数を順に呼ぶ |

## 今は追加しないもの

runtime の `PARAM` と共有ライブラリは [02](02-runtime-inputs.md) で追加する。
hash-consing、interpreter、rewrite と dtype は [03](03-uop-rewrites.md) で追加する。
Tensor、メモリ UOp、GPU はまだ必要ない。
CPU の独自 ELF ローダーは後の章でも移植しない。
