# 07. scalar の累積ループで総和、行和、GEMM と convolution を動かす

## このステップで動かすもの

[06-loops.md](06-loops.md) のループに局所 accumulator を加える。
`sum([1,2,3,4])=10`、2 行 3 列の行和 `[6,15]` を返す。
続いて添字を手で組み立てた小さい GEMM と一次元 convolution を動かす。
元計画第 4 章の累積と手書き GEMM に対応する。
convolution もここでは配列の積和の例として加える。
以下の IR、API、コマンドと数値は未実装の作業計画である。

## 最小実装とデータ

高水準 REDUCE はまだ導入しない。
`ALLOC REG len=1` の accumulator、LOAD、STORE、AFTER と RANGE を組み合わせる。
総和では accumulator を 0、積では 1、MAX では負の無限大で初期化する。
最初に実装するのは f32 の逐次 ADD の総和だけでよい。

予定する総和の動作は次になる。

```text
%acc = ALLOC REG f32 len=1
init: STORE(INDEX(%acc,0), 0.0)
%i = RANGE(4, axis=0)
body: STORE(INDEX(%acc,0), LOAD(INDEX(%acc,0)) + LOAD(INDEX(%x,%i)))
end: END(body,%i)
result: LOAD(INDEX(AFTER(%acc,end),0))
```

この略記の `init`、`body`、`end` は構造化 region の順序を表す。
実装では初期化から RANGE への依存と、body 内の LOAD から STORE への依存を付ける。
REG の LOAD は各反復時点の値を読む。ループ外へ hoist しない。
一回の実行に対する buffer state を、反復回数分の cyclic NodeId で表現しない。
前章の構造化 body が、各反復で同じ命令列を順に実行する。
初期化、反復の更新、ループ後の読出しを区別する region 情報を使う。

行和の予定 C は次になる。

行和の Region を renderer が変換する C の body である。

```c
// src/renderer/c.rs が生成するコード。生成先: <一時ディレクトリ>/kernel.c
for (int32_t row = 0; row < 2; row++) {
	float acc = 0.0f;
	for (int32_t col = 0; col < 3; col++) {
		acc = acc + x[row * 3 + col];
	}
	out[row] = acc;
}
```

行ごとの初期化を外側ループの外へ移すと、二行目が誤る。
LOAD のキャッシュには状態と region を含め、各反復の値を固定しない。
GEMM は `A[i*K+k] * B[k*N+j]` の逐次積和にする。
convolution は重みを反転しない ML の cross-correlation と定義する。
最初は一次元、単一入出力 channel、stride 1、padding 0、dilation 1 に限る。

## この章のファイル配置

累積の仕組みは既存の linearize、interpret、renderer に加える。行和、GEMM と convolution の手書きグラフ builder は examples に置き、高水準 Tensor API や REDUCE のモジュールを先に作らない。

次はこの章の終わりまでに触る関連ファイルの配置案である。現在のcheckoutの一覧ではない。各コミットでは、その下に示す部分だけを追加・変更する。

```text
chibigrad/
├── examples/
│   ├── book_07_reg_sum.rs  # 一要素 REG の状態遷移を確かめる std の独立例。
│   ├── scalar_gemm_conv.rs  # build_conv1d、比較関数 conv1d と case の切替え。
│   └── scalar_reduce.rs  # build_row_sum と rows、cols の実行入力。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # row と col の間へ accumulator 初期化を出力する。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── interpret.rs  # row の反復ごとに REG と効果環境を初期化する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── linearize.rs  # 行ごとの REG 初期化を外側 body 内に保持する。
│   └── verify.rs  # 三つの保存領域長と row-major 添字を既存の証明で検証する。
├── tests/
│   └── scalar_reductions.rs  # 総和、行和、GEMM と cross-correlation の統合確認。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

生成Cやshaderは一時ディレクトリへ出力するため、このRustプロジェクトのツリーには含めない。コード冒頭のコメントで、生成元と生成先を確認する。

## 小さなコミットで進める

### コミット 1. accumulator を更新して総和を返す

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_07_reg_sum.rs  # [追加] 一要素 REG の状態遷移を確かめる std の独立例。
│   └── scalar_reduce.rs  # [追加] 手書き総和グラフと既存 CPU 実行の入口。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # [変更] REG を局所 accumulator として生成する。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── interpret.rs  # [変更] REG の所有と反復中の LOAD、STORE の評価。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── linearize.rs  # [変更] build_sum_loop と REG の初期化、更新、終了後読出しの region。
│   └── verify.rs  # [変更] REG の初期化と反復前後の利用を検証する。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/linearize.rs に build_sum_loop と必要な REG region を追加する。interpret.rs の eval_region と renderer/c.rs の出力を、初期化、反復内更新、END 後の LOAD へ広げる。掲載した sum_with_trace と main は examples/book_07_reg_sum.rs に全体を保存し、examples/scalar_reduce.rs はそのホスト関数で答えを返すのではなく手書きグラフを構築して CPU へ渡す。


前章では各反復の出力が別位置にあったが、総和では同じ局所変数を更新する。候補ファイル `src/linearize.rs` の設計案 `build_sum_loop` は、REG の一要素領域、初期化 region、RANGE の更新 region、END 後の読出しを順に作る。REG は C の局所 `float acc` に対応させてよく、外部 ABI の buffers へ追加しない。初期化 STORE が完了してから RANGE を開始し、反復内は現時点の acc を LOAD してから ADD と STORE を行う。

入力 `[1,2,3,4]` では初期値 0 から、i=0 で 0+1=1、i=1 で 1+2=3、i=2 で 3+3=6、i=3 で 6+4=10 と変わる。loop 外へ返す LOAD は END 後の state を読むので最終値 10 を得る。値の依存グラフに自分自身を指す NodeId を追加する必要はない。構造化 body の LOAD と STORE が、各反復で順番に同じ REG を読み書きすることで状態を進める。

interpreter の候補 `eval_region` は loop の各呼出しで局所 REG の所有物を作り、反復ごとに body の値 memo と効果の実行済み集合を更新する。最初に読んだ acc=0 を NodeId だけで全反復へキャッシュしたり、STORE を全実行で一回に制限したりしない。局所テストは trace の `0,1,3,6,10` と C の 10 を比較し、長さ 0 では body が実行されず初期化値 0 が残ることを調べる。二回の run でも初期化を通し、前回の acc を引き継がない。これで前章の loop と第五章の state 表現を累積へ接続できる。


単独で実行できる例である。`src/interpret.rs` の REG 一要素を、初期化・反復内 LOAD・STORE・END 後 LOAD に分けて追う。実装時にもこの順を region に残す。

sum_with_trace と main の全体を独立例として保存する。

```rust
// examples/book_07_reg_sum.rs（単独で実行する例）
fn sum_with_trace(x: &[f32]) -> (f32,Vec<f32>) {
	let mut reg = [0.0_f32]; // 初期化 region の STORE。
	let mut trace = vec![reg[0]];
	for i in 0..x.len() {
		// 反復の値 memo を持ち越さず、現在の REG を読む。
		let old = reg[0];
		let term = x[i];
		let next = old + term;
		reg[0] = next;
		trace.push(next);
	}
	// END 後の state から読む。
	(reg[0],trace)
}
fn main() {
	let (value,trace) = sum_with_trace(&[1.0,2.0,3.0,4.0]);
	assert_eq!(trace,[0.0,1.0,3.0,6.0,10.0]);
	assert_eq!(value,10.0);
	assert_eq!(sum_with_trace(&[]),(0.0,vec![0.0]));
	assert_eq!(sum_with_trace(&[2.0]),(2.0,vec![0.0,2.0]));
	println!("{trace:?} -> {value}");
}
```

`reg` は run のたびに新しく初期化される。`old` はその反復で読み直す値であり、初期値 0 の別名として固定しない。次の反復は前の STORE の結果を読み、END 後には 10 が残る。REG を更新する仕組みを循環した NodeId にする必要はなく、06 の region が同じ命令を繰り返す。この例は interpreter の状態遷移を検証し、C の生成と実行は既存の CPU 経路で別に確認する。

この入力の途中経過を並べる。

```text
init STORE acc=0
i=0: LOAD 0 -> ADD 1 -> STORE 1
i=1: LOAD 1 -> ADD 2 -> STORE 3
i=2: LOAD 3 -> ADD 3 -> STORE 6
i=3: LOAD 6 -> ADD 4 -> STORE 10
END -> LOAD acc -> 10
```

- [ ] REG allocation と初期化の region を実装する。
- [ ] 反復内 LOAD と STORE を interpreter と C で順に実行する。
- [ ] ループ後の LOAD を END の完了へ依存させる。
- [ ] `[1,2,3,4]` の総和を生成コードで `10` にする。
- [ ] 長さ 0 では初期値 `0` を返す。

このコミットでは手書き loop builder から kernel を作る。
shape 付きの sum API を先に作る必要はない。

### コミット 2. 行ごとに accumulator を初期化する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── scalar_reduce.rs  # [変更] build_row_sum と rows、cols の実行入力。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # [変更] row と col の間へ accumulator 初期化を出力する。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── interpret.rs  # [変更] row の反復ごとに REG と効果環境を初期化する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── linearize.rs  # [変更] 行ごとの REG 初期化を外側 body 内に保持する。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

examples/scalar_reduce.rs に build_row_sum(rows,cols) を追加し、06 の Region を使って外側 row の body に REG 初期化を置く。linearize.rs と interpret.rs はその所属と実行順を保つ。掲載した row_sum の C は renderer/c.rs が出す出力の見本であり、Rust の example へ関数本体として貼らない。


総和の region を、行ごとに作り直す構造へ広げる。候補ファイル `examples/scalar_reduce.rs` の設計案 `build_row_sum(rows,cols)` は、外側 row の body に REG の宣言と初期化を置き、その後に内側 col の body を置く。初期化の open_ranges は `{row}`、更新式は `{row,col}`、出力 STORE は内側 END の後なので `{row}` となる。外側 END の後にしか最終出力全体を返さない。

row=0 では acc を 0 に戻し、位置 `0*3+0`、1、2 の 1、2、3 を読み、acc が 1、3、6 と変わった後に out[0]=6 を書く。row=1 では前の 6 を使わず再び 0 を入れ、位置 3、4、5 の 4、5、6 で acc が 4、9、15 になり out[1]=15 を書く。REG の宣言を C のどこに置くかにかかわらず、初期化の効果は外側の各反復で必ず実行する。interpreter の「実行済み STORE」は row の次の反復へ持ち越さない。

局所テストは `[6,15]` の数値と、生成 C の初期化が二つの for の間にあることを照合する。初期化を一回だけにした誤った fixture なら二行目は 6+4+5+6=21 になるので、この入力は region の誤りを見つけられる。列数 0 の場合にも各 row の初期化と out[row] の STORE は実行し、出力は `[0,0]` にする。内側の反復を省くことと外側の効果を省くことを区別する。既存の checked な `row*cols+col` の範囲証明と固定 ABI を使い、shape の自動導出はここへ先取りしない。


組込み用の抜粋である。`examples/scalar_reduce.rs` の `build_row_sum` が生成する実 C を、関数の形で示す。`rows*cols` と出力長はホストの checked 演算で検証し、入力と出力は別保存領域として束縛する。

scalar_reduce の build_row_sum から renderer が生成する C である。

```c
// src/renderer/c.rs が生成するコード。生成先: <一時ディレクトリ>/kernel.c
#include <stdint.h>
static void row_sum(const float *x, float *out, int32_t rows, int32_t cols) {
	for (int32_t row = 0; row < rows; row++) {
		float acc = 0.0f;
		for (int32_t col = 0; col < cols; col++) {
			int32_t offset = row * cols + col;
			float old = acc;
			float term = x[offset];
			acc = old + term;
		}
		out[row] = acc;
	}
}
```

`float acc=0.0f` は row の body に入るたびに実行され、col の body の外側にある。列数 0 でも外側 row と最後の STORE は残るので、各行に 0 を出す。kernel builder では REG 宣言と初期化 STORE を row の region に置き、col の END 後に REG の LOAD と出力 STORE を置く。この C の字下げと region の所属が一致することを dump で確認する。

この入力の途中経過を並べる。

```text
row0: acc 0 -> 1 -> 3 -> 6 -> out[0]=6
row1: acc 0 -> 4 -> 9 -> 15 -> out[1]=15
cols=0: row0 acc0 -> out0=0; row1 acc0 -> out1=0
初期化を row の外に出した誤実装: out=[6,21]
```

- [ ] 外側 RANGE の body に accumulator 初期化を置く。
- [ ] 内側 RANGE で列を累積する。
- [ ] 内側 END の後、外側 END の前に行の出力を STORE する。
- [ ] `[1,2,3,4,5,6]` の行和を `[6,15]` にする。
- [ ] 空の列数なら各行を `0` にする。

予定する間違いの検出例は、二行目が `21` になる初期化位置の誤り。
生成 C の位置と実際の `[6,15]` の両方を確認する。

### コミット 3. 三重ループの手書き GEMM を動かす

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── scalar_gemm_conv.rs  # [追加] build_gemm、独立した比較関数 gemm と main。
├── src/
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   └── verify.rs  # [確認] 三つの保存領域長と row-major 添字を既存の証明で検証する。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

examples/scalar_gemm_conv.rs を追加し、掲載した gemm をホストの比較用関数として置く。同じファイルに build_gemm と一つの main を追加し、既存の loop builder で i、j、k の kernel を作って CPU へ渡す。比較用 gemm を compiler の生成結果として返さず、生成コードの出力と照合する。


GEMM は新しい低水準命令ではなく、今ある三つの RANGE と累積を組み合わせる example として作る。候補ファイル `examples/scalar_gemm_conv.rs` の設計案 `build_gemm(m,n,k)` は i、j、k の順で region を作り、i と j が決まった場所で acc を初期化する。内側は A の位置 `i*K+k` と B の位置 `k*N+j` を LOAD し、積を acc に足す。k の END が終わってから `out[i*N+j]` へ書き、次の j では acc を再び 0 にする。

M=2、K=3、N=2 のとき、出力 (0,0) の積は `1*1=1`、`2*3=6`、`3*5=15` で、acc は 1、7、22 となる。(0,1) は `1*2=2`、`2*4=8`、`3*6=18` から 28 となる。行 1 の出力は同じ規則で 49 と 64 になる。B を `j*K+k` で読んでしまう誤りは、この非対称な入力と長方形の寸法なら結果へ現れる。

実装前に各 buffer 長と最大 offset を checked なホスト整数で求め、A は M*K、B は K*N、out は M*N の容量を検証する。局所テストの比較先は同じ kernel builder を解釈した結果だけにせず、ホストの独立した三重 loop でも積和を求める。dump では acc の初期化が i と j の内側、k の外側に一つあり、出力 STORE が k の外にあることを見る。最初は k の昇順に逐次加算し、結果を変える並べ替えや tiling を加えない。CPU の compile/run と loop verifier は既存経路を使える。


組込み用の抜粋である。`examples/scalar_gemm_conv.rs` で kernel の比較先として使う、row-major の三重ループを示す。この関数の三つの loop と acc の位置を、そのまま i、j、k の region として builder に写す。

この example のホスト比較用 gemm を追加する。compiler 本体の新命令にはしない。

```rust
// examples/scalar_gemm_conv.rs（既存ファイルへ組み込む抜粋）
fn gemm(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Result<Vec<f32>,String> {
	let alen = m.checked_mul(k).ok_or("A size overflow")?;
	let blen = k.checked_mul(n).ok_or("B size overflow")?;
	let olen = m.checked_mul(n).ok_or("output size overflow")?;
	if a.len() != alen || b.len() != blen { return Err("GEMM input length mismatch".into()); }
	if [alen,blen,olen].iter().any(|&len| len > i32::MAX as usize) {
		return Err("GEMM index exceeds i32".into());
	}
	let mut out = vec![0.0;olen];
	for i in 0..m {
		for j in 0..n {
			let mut acc = 0.0_f32;
			for q in 0..k {
				let av = a[i*k+q];
				let bv = b[q*n+j];
				let product = av * bv;
				acc += product;
			}
			out[i*n+j] = acc;
		}
	}
	Ok(out)
}
```

A の横移動は q、B の縦移動は `q*n` である。acc は i と j が決まってから初期化し、q が全部終わってから一度だけ out に書く。サイズの積を先に checked 演算で求め、添字を i32 で生成できる範囲に制限する。kernel builder はこの参照関数を呼んで答えを作るのではなく、同じ位置式の LOAD、MUL、ADD、STORE を構築し、生成 C の結果をこの関数と照合する。

この入力の途中経過を並べる。

```text
M=2,K=3,N=2
(i,j)=(0,0): products=[1,6,15], acc=[1,7,22]
(i,j)=(0,1): products=[2,8,18], acc=[2,10,28]
(i,j)=(1,0): products=[4,15,30] -> 49
(i,j)=(1,1): products=[8,20,36] -> 64
```

- [ ] `i`、`j`、`k` の三軸と row-major 添字を手で作る。
- [ ] 各 `(i,j)` で accumulator をゼロへ戻す。
- [ ] reduction 軸 k の END の後に出力を STORE する。
- [ ] 2×3 と 3×2 の入力を既存 CPU runtime で呼ぶ。
- [ ] ホストの独立した三重ループと結果を比較する。

予定入力は `A=[[1,2,3],[4,5,6]]`、`B=[[1,2],[3,4],[5,6]]`。
予定結果は `[[22,28],[49,64]]`。
出力要素ごとの accumulator を共有してしまう誤りも検出する。

### コミット 4. 二重ループの小さい convolution を動かす

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── scalar_gemm_conv.rs  # [変更] build_conv1d、比較関数 conv1d と case の切替え。
├── tests/
│   └── scalar_reductions.rs  # [追加] 総和、行和、GEMM と cross-correlation の統合確認。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

conv1d の抜粋を examples/scalar_gemm_conv.rs に追加し、既存の main から --case conv1d で選べるようにする。build_conv1d もこの example に置き、位置式を低水準グラフとして作る。tests/scalar_reductions.rs では公開された構築と実行 API を使って、空の総和、行和と生成 kernel の積和を独立したホスト計算に照合する。


一次元 convolution も、出力位置ごとの短い積和として手で記述する。候補ファイル `examples/scalar_gemm_conv.rs` に設計案の `build_conv1d(input_len,kernel_len)` を加える。最初に kernel 長が正で入力長以下かを調べ、checked な減算と加算で出力長を求める。その後で外側 p を出力長、内側 k を kernel 長とする二重 loop を作り、p ごとの REG に `x[p+k]*w[k]` を足す。この章の定義は重みを反転しないので w の位置は `kernel_len-1-k` にしない。

入力長 4、kernel 長 2 なら出力長は 3 になる。p=0 では `1*2=2` と `2*1=2` の和が 4、p=1 では `2*2=4` と `3*1=3` の和が 7、p=2 では `3*2=6` と `4*1=4` の和が 10 となる。k の END 後にだけ out[p] を STORE し、次の p の初期化を通る。最大の入力位置は `(3-1)+(2-1)=3` で、長さ 4 の有効範囲に収まる。

局所テストではこの trace と `[4,7,10]` を interpreter、生成 C、ホストの独立した二重 loop で比較する。w=[2,1] は非対称なので、重みを反転した実装なら先頭が 5 となって違いを検出できる。kernel 長が入力より長い負例は出力長の unsigned underflow より前に拒否する。入力と重みは read-only のスロットとして束縛し、別の値で再実行して acc が残らないことも見る。padding や stride を追加せず、既存の RANGE とアドレス検証の組合せが動くことをこのコミットの到達点にする。


組込み用の抜粋である。`examples/scalar_gemm_conv.rs` に置く一次元 cross-correlation の比較用関数を示す。kernel builder の外側軸 p と内側軸 q も、この出力長と位置式から作る。

同じ example へ比較用 conv1d を追加し、main は既存の一つを使う。

```rust
// examples/scalar_gemm_conv.rs（既存ファイルへ組み込む抜粋）
fn conv1d(x: &[f32], w: &[f32]) -> Result<Vec<f32>,String> {
	if w.is_empty() || w.len() > x.len() { return Err("invalid kernel length".into()); }
	let len = x.len().checked_sub(w.len()).and_then(|n| n.checked_add(1)).ok_or("output length overflow")?;
	if x.len() > i32::MAX as usize { return Err("convolution index exceeds i32".into()); }
	let mut out = vec![0.0;len];
	for p in 0..len {
		let mut acc = 0.0_f32;
		for q in 0..w.len() {
			let input = x[p+q];
			let weight = w[q];
			acc += input * weight;
		}
		out[p] = acc;
	}
	Ok(out)
}
```

`w[q]` をそのまま使うため重みは反転しない。長さ 4 と 2 なら len=3、最大 p+q は 2+1=3 になり、入力の最後の位置に収まる。空 kernel と長すぎる kernel は減算前に拒否する。各 p の acc 初期化、q の逐次積和、q の END 後の STORE が、手書き IR の構造そのものになる。非対称な重み `[2,1]` で先頭が 4 になることを確認すると、反転した convolution との差も見える。

この入力の途中経過を並べる。

```text
p=0: x[0]*w[0]+x[1]*w[1] = 1*2+2*1 = 4
p=1: 2*2+3*1 = 7
p=2: 3*2+4*1 = 10
weights を反転した誤実装: 最初の要素=1*1+2*2=5
```

- [ ] 出力位置 p と kernel 位置 k を RANGE で作る。
- [ ] `x[p+k]*w[k]` を累積する。
- [ ] `output_len=input_len-kernel_len+1` を checked 演算で求める。
- [ ] `x=[1,2,3,4]`, `w=[2,1]` を `[4,7,10]` にする。
- [ ] kernel が入力より長い場合を明確な未対応エラーにする。

padding はまだないため、guard 付き LOAD は必要ない。
入力と重みは BUFFER の入力スロットから受け取り、定数に固定しない。

## 完了確認

以下はこの章で新しく作る example とテストの予定コマンドである。

```sh
# 作業ディレクトリ: chibigrad/
cargo test
cargo run --example scalar_reduce -- --case sum --dump-kernel
cargo run --example scalar_reduce -- --case rows --dump-source
cargo run --example scalar_gemm_conv -- --case gemm
cargo run --example scalar_gemm_conv -- --case conv1d
```

- [ ] 総和が `10`、行和が `[6,15]` になる。
- [ ] 空の総和と列数 0 の行和が加算の初期値を返す。
- [ ] GEMM が `[[22,28],[49,64]]`、convolution が `[4,7,10]` になる。
- [ ] dump に高水準 REDUCE がなく、累積を行う LOAD と STORE がある。
- [ ] accumulator の初期化が正しい外側 region にある。
- [ ] 入力を変えた二回目の run でも前回の accumulator が残らない。
- [ ] 一般 f32 は前章までの予定許容誤差で interpreter と CPU を比較する。

この章では最適化せず、逐次の演算順も参照実装とそろえる。
GEMM の速度や大きなモデルの実行は完了条件にしない。

## tinygrad ソースとの対応

参照 commit は `c22edff14c79310a9b2df62a78819cf77c8289fa`。
リンクの行番号は checkout で確認した。

| 実在するシンボル | ソース | 役割 | chibigrad で簡略化する点 |
|---|---|---|---|
| `identity_element` | [ops.py](../../tinygrad/tinygrad/uop/ops.py#L58) | reduction の初期値 | まず f32 ADD の 0 を使う |
| `lower_reduce` | [codegen](../../tinygrad/tinygrad/codegen/__init__.py#L192) | REG と初期化、更新、END を作る | 高水準 REDUCE はまだなく、この出力相当を手で作る |
| `UOp.alloc_like` | [ops.py](../../tinygrad/tinygrad/uop/ops.py#L1161) | 同型の保存領域を宣言する | REG の scalar 一要素に限定する |
| `CFGContext` | [linearizer.py](../../tinygrad/tinygrad/codegen/late/linearizer.py#L50) | 反復の依存と入れ子 | region で初期化と更新を明示する |
| `base_rewrite` の LOAD と STORE | [cstyle.py](../../tinygrad/tinygrad/renderer/cstyle.py#L57) | メモリ上の値を C で読み書きする | accumulator は普通の局所変数でよい |

## 今は追加しないもの

[08-shapes-rangeify.md](08-shapes-rangeify.md) で形状から同じ累積ループを自動生成する。
多次元 Tensor API の matmul と convolution は第 11 段階へ進める。
並列 reduction、tiling、速い GEMM は第 18、19 段階で加える。
