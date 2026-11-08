# 10. 多次元の座標変換で movement を実行する

## このステップで動かすもの

[09 の fusion と fission](09-fusion-fission.md) の shape 付きグラフを拡張する。
元計画の第 7 章「多次元 shape と movement」に対応する。
まず CPU だけで動かし、出力座標から入力座標への変換を確認する。

```text
x = [[1,2,3],[4,5,6]]
x + [10,20,30] = [[11,22,33],[14,25,36]]
x.permute([1,0]).reshape([6]) = [1,4,2,5,3,6]
x.shrink([(0,2),(1,3)]) = [[2,3],[5,6]]
pad([1,2], [(1,2)]) = [0,1,2,0,0]
flip([1,2,3], [0]) = [3,2,1]
stack([1,2], [3,4], axis=0) = [[1,2],[3,4]]
```

記載する Rust API、example、テスト名、コマンドはこれから実装する予定である。
Tensor の使いやすい入口は [11](11-tensor-gemm-conv.md) で追加する。
ここでは graph builder から movement を組み立てて実行する。

## 最小実装とデータ

shape は静的な非負整数の列とし、要素数と stride の積を検査する。
06 までの空 tensor の契約を維持し、reshape の除算でゼロ除算を起こさない。
出力の要素数が 0 なら添字を評価せず、空の結果を返す。

```text
入力IR: PERMUTE(RESHAPE(PARAM(x), [2,3]), [1,0])
出力shape: [3,2]
出力座標: (r,c)
入力座標: (c,r)
kernel IR: LOAD(x, c*3+r) → STORE(out, r*2+c)
```

追加する操作は `RESHAPE`、`PERMUTE`、`SHRINK`、`PAD`、`FLIP`、`STACK` とする。
`FLIP` は chibigrad 側の導入案であり、参照版に同名 UOp があるという意味ではない。
`STACK` は同形の入力を新しい先頭軸で並べる形から始める。
pad の値はまず `f32` のゼロに限る。

reshape は出力座標を論理的な一次元位置へ畳み、入力 shape で除算と剰余を取る。
転置後の reshape も、この写像を順に合成すれば扱える。
非連続な配列に stride を一つ付けるだけでは上記の順序を表せない。
最初のコミットで難しければ、その箇所を明示的に materialize する。
その場合もコピー前の論理順序を維持し、schedule dump に境界を出す。

添字式のキャッシュキーには `NodeId` と出力座標式、カーネルのスコープを含める。
同じ入力を二つの座標で読む場合を、同じ LOAD にまとめない。
純粋な元グラフは保持し、実体化先の表を別に持つ。

## この章のファイル配置

既存の src/shape.rs で shape を検証し、src/rangeify.rs の lower_at に出力から入力への座標変換を足す。整数の単独例は examples/book_10_*.rs、生成 C は renderer の出力として読む。

次はこの章の終わりまでに触る関連ファイルの配置案である。現在のcheckoutの一覧ではない。各コミットでは、その下に示す部分だけを追加・変更する。

```text
chibigrad/
├── examples/
│   ├── book_10_broadcast_coords.rs  # 本文のコードを配置する。
│   ├── book_10_reshape_permute.rs  # 本文のコードを配置する。
│   ├── movement_broadcast.rs  # Graph から二つの broadcast を実行する予定例。
│   ├── movement_flip_stack.rs  # flip と stack の予定実行例。
│   ├── movement_pad.rs  # 非対称 pad を実行する予定例。
│   └── movement_reshape.rs  # 転置後 reshape を CPU Plan で動かす予定例。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # 入力選択に従う条件付き LOAD の出力。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rangeify.rs  # 反転座標と入力選択の分岐。
│   ├── shape.rs  # flip 軸と stack 入力 shape の検証。
│   └── uop.rs  # FLIP と先頭軸 STACK の builder。
├── tests/
│   └── movement.rs  # 二重 flip、stack、不正入力。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

生成Cやshaderは一時ディレクトリへ出力するため、このRustプロジェクトのツリーには含めない。コード冒頭のコメントで、生成元と生成先を確認する。

## 小さなコミットで進める

### 1. 二次元の加算と broadcast を実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_10_broadcast_coords.rs  # 本文のコードを配置する。
│   └── movement_broadcast.rs  # [追加] Graph から二つの broadcast を実行する予定例。
├── src/
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rangeify.rs  # [更新] 各出力軸の RANGE と lower_at の入力座標。
│   └── shape.rs  # [更新] 多次元 broadcast の検証と入力座標。
├── tests/
│   └── movement.rs  # [追加] interpreter と C の多次元比較。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/shape.rs に broadcast_shape を追加し、整数版 input_coords は単独例で確認する。src/rangeify.rs の既存 lower_at を多次元の座標列へ広げ、NodeId・座標・scope を memo のキーにする。tests/movement.rs と予定 example では既存 Graph と CPU Plan の公開入口を使う。


この変更では、08までの一次元の添字生成を、出力の各軸を一つずつ巡る処理へ広げる。変更先は既存の shape 検証と rangeify のファイルとし、まだ分離していなければ同じモジュールでよい。設計案の `broadcast_shape(a,b)` は右端から軸を照合し、同値または片方が1の場合だけ出力長を決める。欠けた先頭軸は長さ1として扱うが、出力から入力へ戻すときはその軸を取り除く。`lower_at(node,coords,scope)` は指定座標の値を返す予定の入口であり、Tensor 全体をコピーして返す関数にはしない。

`[2,3]+[3]` なら出力 RANGE は `r=0..2,c=0..3`、出力の保存位置は `r*3+c` になる。左入力の座標は `(r,c)`、右入力は `(c)` と決めてから、それぞれの連続バッファ添字を生成する。出力 `(1,2)` では左の位置5から6、右の位置2から30を読み、36を出力位置5へ保存する。`[2,1]+[1,3]` では同じ出力座標を左 `(1,0)`、右 `(0,2)` に戻すので、2と30から32になる。軸長1の入力を出力座標のまま読む誤りを、この一要素で見分けられる。

先に全軸の整合性と要素数の積の overflow を検証し、その後だけ RANGE と LOAD を作る。`[2,3]+[2]` は末尾3と2が合わない時点で止め、部分的なカーネルを作らない。座標の値が変わる式を `NodeId` だけで memo 化しない。出力座標式と scope を含むキーを使い、同一の読出し条件の場合だけ共有する。`tests/movement.rs` では二つの broadcast 例を interpreter と生成 C で比較し、kernel dump の左右の LOAD 添字まで確認する。入力 slot と buffers/scalars の固定 ABI はそのまま再利用でき、このコミットで多次元の elementwise 全体が既存 runtime までつながる。


`src/shape.rs` の入力座標の計算を、まず整数で実行する。入力は右揃え可能な shape と出力座標、返すものはその入力だけの座標である。

単独で実行できる例。

入力座標の整数参照例全体を保存する。src/shape.rs へ main ごと貼らない。

```rust
// examples/book_10_broadcast_coords.rs（単独で実行する例）
fn input_coords(shape: &[usize], output: &[usize]) -> Vec<usize> {
	assert!(shape.len() <= output.len());
	let skip = output.len() - shape.len();
	shape.iter().enumerate().map(|(axis, &size)| {
		let c = output[skip + axis];
		if size == 1 { 0 } else { assert!(c < size); c }
	}).collect()
}
fn main() {
	let x = [1,2,3,4,5,6];
	let b = [10,20,30];
	let mut out = [0;6];
	for r in 0..2 {
		for c in 0..3 {
			let a = input_coords(&[2,3], &[r,c]);
			let q = input_coords(&[3], &[r,c]);
			out[r*3+c] = x[a[0]*3+a[1]] + b[q[0]];
		}
	}
	assert_eq!(out, [11,22,33,14,25,36]);
	assert_eq!(input_coords(&[2,1], &[1,2]), [1,0]);
}
```

`skip` は入力にない先頭軸の個数である。長さ1の軸だけを0へ固定し、残りの座標をそのまま渡す。ここでは全 shape の互換性は既存 verifier が先に検査する。実数値で確かめた後、この整数計算を添字用 UOp の生成へ移す。

同じ入力を追う例。

```text
output=(1,2)
x:[2,3] -> (1,2) -> offset=5 -> 6
b:[3]   -> (2)   -> offset=2 -> 30
out[5]=36
```

二つの入力で同じ出力座標から異なる長さの座標列を得る。memo のキーにもこの違いを残す。


組込み用の抜粋。`src/shape.rs` へ足す shape 検証。後の Tensor の `broadcast_shape(a,b)` はこの関数を再利用する。

既存 shape 検証へ broadcast_shape を追加する抜粋。

```rust
// src/shape.rs（既存ファイルへ組み込む抜粋）
fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>,String> {
	let rank = a.len().max(b.len());
	let mut shape = vec![1;rank];
	for from_right in 0..rank {
		let left = a.len().checked_sub(from_right+1).map(|j|a[j]).unwrap_or(1);
		let right = b.len().checked_sub(from_right+1).map(|j|b[j]).unwrap_or(1);
		let size = if left==right { left }
			else if left==1 { right }
			else if right==1 { left }
			else { return Err(format!("broadcast mismatch {left} vs {right}")); };
		shape[rank-1-from_right] = size;
	}
	let mut count = 1_usize;
	for &size in &shape {
		count = count.checked_mul(size).ok_or("element count overflow")?;
	}
	Ok(shape)
}
```

`from_right` は末尾から数える軸番号で、入力に軸がなければ1を補う。0と1の組では0を返し、空出力を維持するため `max(left,right)` へ置き換えない。全 shape が確定した後で要素数の overflow を検査し、成功したときだけ rangeify へ渡す。

同じ関数に渡す shape の例。

```text
broadcast_shape([2,3],[3])   -> Ok([2,3])
broadcast_shape([2,1],[1,3]) -> Ok([2,3])
broadcast_shape([0,3],[1,3]) -> Ok([0,3])
broadcast_shape([2,3],[2])   -> Err("broadcast mismatch 3 vs 2")
```

先頭軸を補う処理と長さ1の軸の拡張をここで決め、前の `input_coords` がそれを元入力の座標へ戻す。0要素の結果では input_coords を呼び出さず、空出力を返す。

- [ ] 出力の二次元 RANGE と row-major の平坦化を追加する。
- [ ] shape を右揃えし、入力側の長さ 1 の軸を座標 0 にする。
- [ ] `[2,3] + [3]` と `[2,1] + [1,3]` を CPU で実行する。
- [ ] `[2,3] + [2]` は broadcast 不成立として codegen 前に拒否する。

予定例 `movement_broadcast` は上記の `[[11,22,33],[14,25,36]]` を返す。
`[2,1]` の `[[1],[2]]` と `[1,3]` の `[[10,20,30]]` は `[[11,21,31],[12,22,32]]` になる。
各出力について入力の添字が別々に導かれることを dump で見る。

### 2. reshape と permute を合成する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   ├── book_10_reshape_permute.rs  # 本文のコードを配置する。
│   └── movement_reshape.rs  # [追加] 転置後 reshape を CPU Plan で動かす予定例。
├── src/
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rangeify.rs  # [更新] map_reshape と map_permute の添字 UOp。
│   ├── shape.rs  # [更新] 要素数と permutation の検証。
│   └── uop.rs  # [更新] RESHAPE と PERMUTE の payload と builder。
├── tests/
│   └── movement.rs  # [更新] 転置後 reshape と不正軸。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/uop.rs の既存 Op と builder に RESHAPE/PERMUTE を足し、src/shape.rs で要素数と軸列を検証する。本文の候補は、整数による比較先を src/shape.rs、添字 UOp を返す map_reshape/map_permute を src/rangeify.rs に置く案へそろえる。単独例をそのまま本体へ貼らず、検証後に写像を lower_at の分岐へ組み込む。


ここでは 論理的な要素順を保って shape を変える reshape と、軸順を換える permute を、出力から入力へ向かう座標関数として実装する。候補は shape 検証と rangeify の movement 分岐で、設計案の `map_reshape(in_shape,out_shape,coords)` と `map_permute(order,coords)` をその場に置く。reshape は両側の要素数を先に比較し、permute は長さが rank と等しく、各軸番号が一度だけ現れることを検査する。変換元の純粋なノードは書き換えず、今回のカーネル用の添字式だけを返す。

reshape の出力 shape を `t`、座標を `c` とすると、論理位置は `p=Σ c[j]*product(t[j+1..])` である。入力 shape `s` の座標は右端から `a[j]=p%s[j]; p=p/s[j]` と復元する。permute の出力軸 j が入力軸 `order[j]` に対応するので、逆向きには `a[order[j]]=c[j]` と埋める。軸列をそのまま入力座標へ適用する方法では、三軸以上の循環 permutation を誤るので、逆 permutation を必ず構成する。

`x:[2,3]→permute([1,0]):[3,2]→reshape([6])` の出力位置3を追う。reshape から `(3/2,3%2)=(1,1)` を得て、permute で入力 `(1,1)` へ戻し、元バッファ位置4の5を読む。出力位置1は転置側 `(0,1)`、元入力 `(1,0)` から4になり、全体は `[1,4,2,5,3,6]` となる。この例は元バッファを連続した6要素として読む実装との違いが見える。

空の shape は先に空出力へ分岐し、長さ0での剰余や除算を作らない。memo は元ノード、変換後の座標式、scope の組にし、同じ PARAM を通常配置と転置配置で使う式でも LOAD を混同しない。局所検証はこの trace、要素数違い、重複軸、三軸の循環 permutation を含める。materialize を選ぶ場合は論理順で中間配列へ書き、schedule の明示境界を越えて同じ期待値になることを確認する。


`map_reshape` の中心は平坦化と復元、`map_permute` の中心は入力軸への代入である。次の整数版を `src/shape.rs` の写像の比較先にする。

単独で実行できる例。

二つの座標写像と main を一つの単独参照例として保存する。

```rust
// examples/book_10_reshape_permute.rs（単独で実行する例）
fn reshape_coords(input: &[usize], output: &[usize], c: &[usize]) -> Vec<usize> {
	assert_eq!(input.iter().product::<usize>(), output.iter().product());
	assert_eq!(c.len(), output.len());
	assert!(input.iter().all(|&s| s > 0)); // 空出力は呼出し前に除外
	let mut p = 0;
	for (&size, &coordinate) in output.iter().zip(c) {
		assert!(coordinate < size);
		p = p * size + coordinate;
	}
	let mut a = vec![0; input.len()];
	for axis in (0..input.len()).rev() {
		a[axis] = p % input[axis];
		p /= input[axis];
	}
	a
}
fn permute_coords(order: &[usize], c: &[usize]) -> Vec<usize> {
	let mut a = vec![0; order.len()];
	let mut seen = vec![false; order.len()];
	for (j, &axis) in order.iter().enumerate() {
		assert!(axis < order.len() && !seen[axis]);
		seen[axis] = true;
		a[axis] = c[j];
	}
	a
}
fn main() {
	let x = [1,2,3,4,5,6];
	let y: Vec<_> = (0..6).map(|i| {
		let c = reshape_coords(&[3,2], &[6], &[i]);
		let a = permute_coords(&[1,0], &c);
		x[a[0]*3+a[1]]
	}).collect();
	assert_eq!(y, [1,4,2,5,3,6]);
	assert_eq!(permute_coords(&[1,2,0], &[2,3,1]), [1,2,3]);
}
```

最初のループが出力の論理位置 `p` を作る。逆順のループは入力 shape の末尾から余りを取り、座標を復元する。permute は `c[order[j]]` と読むのではなく、入力軸 `order[j]` へ出力座標 `c[j]` を書く。

同じ入力を追う例。

```text
reshape output i=1 -> permuted coordinates (0,1)
permute order [1,0] -> original coordinates (1,0)
LOAD x[1*3+0] = 4
```

この順に二つの写像を合成すると転置の論理順を維持できる。単なる元バッファの平坦な読出しでは位置1が2になってしまう。

- [ ] 要素数が等しい reshape だけを受け付ける。
- [ ] permute の軸が重複せず、全軸を一度ずつ含むことを検証する。
- [ ] 論理的な平坦化と座標の復元を rangeify に追加する。
- [ ] 転置後の reshape を CPU 実行し、連続化した場合とも比較する。

予定例 `movement_reshape` の出力は `[1,4,2,5,3,6]` とする。
`reshape([5])` と `permute([0,0])` はそれぞれ専用のエラーを返す。
同じ PARAM を元配置と転置配置で読む式も同じ生成カーネルで実行する。

### 3. shrink と安全な pad を実行する

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── movement_pad.rs  # [追加] 非対称 pad を実行する予定例。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # [更新] valid の内側だけに LOAD を出す分岐。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rangeify.rs  # [更新] MappedIndex の coords/valid と条件付き LOAD。
│   ├── shape.rs  # [更新] 半開区間、padding 長と出力 shape。
│   └── uop.rs  # [更新] SHRINK と PAD の定義と builder。
├── tests/
│   └── movement.rs  # [更新] pad と shrink の値と安全性。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/rangeify.rs に MappedIndex { coords, valid } を置き、SHRINK と PAD の lower_at 分岐で座標と有効条件を合成する。src/uop.rs と src/shape.rs に対応する操作と検証を足す。掲載 C は src/renderer/c.rs が生成する分岐の目標形であり、Rust ソースへ貼らない。


shrink と pad はどちらも座標を平行移動するが、pad では入力座標の有効性も値の一部として扱う。候補は movement の shape 検証と rangeify、条件付き LOAD を出す C renderer である。設計案の座標変換の結果を `MappedIndex { coords, valid }` のような組にし、valid はその入力を安全に読める条件式とする。shrink の出力長は `end-start`、入力座標は `out+start` であり、全区間を検証してから式を作る。pad の出力長は `before+size+after`、入力座標は `out-before`、valid は全軸について `before<=out<before+size` を満たす条件の積にする。

長さ2の `[1,2]` に before=1、after=2 を付けると出力長5になる。座標0では入力座標-1と valid=false を得て、LOAD を実行せずゼロを返す。座標1では入力0から1、座標2では入力1から2、座標3と4は再びゼロになる。これを shrink の区間 `[1,4)` で切る合成なら、出力座標0,1,2を pad 側1,2,3へ戻し、結果 `[1,2,0]` を得る。条件と座標の両方を順に合成する必要がある。

C の選択式を作る前に LOAD を共通の一時変数へ持ち上げると、valid=false でも負の位置を読む。最初は `if(valid){value=LOAD(...)}else{value=0}` という制御分岐で表現し、その内側にだけ INDEX と LOAD を置く。メモリ状態を表す依存も分岐の内部へ保持する。同じ座標でも異なる valid 条件を持つ読出しを共有しないよう、添字 memo の条件も区別する。

`tests/movement.rs` に非対称 pad、空 shrink、全領域が padding の出力位置、不正区間を加え、interpreter と生成 C の両方で値を照合する。`--check-bounds` の予定経路では条件外の LOAD が実際にないことを確かめ、source dump でも分岐内に読出しがあるかを見る。既存の出力 RANGE、STORE、固定 ABI を再利用し、pad 専用の host コピーには逃がさない。


`src/renderer/c.rs` に出す分岐を、長さ2への pad とその shrink に固定して見る。入力は read-only、出力だけを更新する生成 C の関数本体である。

組込み用の抜粋。

rangeify が作る条件付き LOAD を C renderer が出力する目標形。

```c
// src/renderer/c.rs が生成するコード。生成先: <一時ディレクトリ>/pad_then_shrink.c
void pad_then_shrink(const float *x, float *out) {
	/* PAD(x, before=1, after=2) の範囲 [1,4) を SHRINK */
	for (int i = 0; i < 3; i++) {
		int pad_coordinate = i + 1;
		int input_coordinate = pad_coordinate - 1;
		float value = 0.0f;
		if (pad_coordinate >= 1 && pad_coordinate < 3) {
			value = x[input_coordinate];
		}
		out[i] = value;
	}
}
```

`pad_coordinate` は shrink の開始 offset を戻した位置で、`input_coordinate` はさらに pad の before を引いた位置になる。i=2 は pad 側3へ戻るが valid が偽なので x[2] を読まない。

同じ入力を追う例。

```text
x=[1,2]
PAD:       [0,1,2,0,0]
SHRINK 1:4 [1,2,0]
i=2 -> pad_coordinate=3 -> valid=false -> value=0
```

`value` の初期値をゼロにし、LOAD は if の内部だけに置く。条件外のアドレスを読んだ後にゼロへ選び直すコードへ変えない。

- [ ] shrink の半開区間 `start..end` を入力側の offset に変換する。
- [ ] `0 <= start <= end <= input_size` を各軸で検証する。
- [ ] pad の有効条件と、条件成立時だけ実行する LOAD を生成する。
- [ ] pad の外側はゼロを返し、無効なポインタへの LOAD を作らない。
- [ ] 非対称 pad と shrink の合成を interpreter と生成 C で照合する。

予定例 `movement_pad` は `[0,1,2,0,0]` を返す。
`x[-1]` を一旦 LOAD してから値だけゼロに選び直すコードは不合格にする。
CPU の安全性検査を有効にしても端の要素で範囲外読出しを起こさないことを確認する。

### 4. flip と stack を同じ経路に通す

このコミットで触るファイルは次のとおり。既存の他のファイルは省略している。

```text
chibigrad/
├── examples/
│   └── movement_flip_stack.rs  # [追加] flip と stack の予定実行例。
├── src/
│   ├── renderer/
│   │   ├── c.rs  # [更新] 入力選択に従う条件付き LOAD の出力。
│   │   └── mod.rs  # 子モジュールを宣言する。
│   ├── lib.rs  # 本体のモジュール宣言と公開APIの入口。
│   ├── rangeify.rs  # [更新] 反転座標と入力選択の分岐。
│   ├── shape.rs  # [更新] flip 軸と stack 入力 shape の検証。
│   └── uop.rs  # [更新] FLIP と先頭軸 STACK の builder。
├── tests/
│   └── movement.rs  # [更新] 二重 flip、stack、不正入力。
└── Cargo.toml  # 既存のpackage設定。必要な依存だけを追加する。
```

src/uop.rs の Op と builder に FLIP/STACK を足し、src/shape.rs で軸と入力列を検証する。src/rangeify.rs の lower_at に反転と先頭座標による入力選択を追加する。掲載 C に合わせて src/renderer/c.rs は選ばれた入力の分岐内だけに LOAD を出す。


このコミットでは、添字を反転する FLIP と、新しい軸で入力を選ぶ STACK を同じ座標 lowering に追加する。変更先は graph builder、shape 検証、movement の分岐とする。設計案の `flip(node,axes)` は入力と同じ shape を返し、`stack(inputs,axis=0)` は全入力の shape と dtype が一致したときだけ先頭へ入力数を追加する。最初は先頭以外の axis を拒否し、任意軸に対応したような API にしない。入力列の順序は意味を持つので、ノードの共有処理でも並べ替えない。

flip の対象軸 j では出力座標 c を `s[j]-1-c`、それ以外は c のまま入力へ渡す。長さ3の `[1,2,3]` なら出力0は入力2から3、出力1は入力1から2、出力2は入力0から1を読む。同じ軸をもう一度反転すると `s-1-(s-1-c)=c` に戻り、追加のコピーなしで元の値になる。重複した flip 軸は相殺に任せず入力エラーにし、指定の取り違えを早く見付ける。

二入力 a=`[1,2]`、b=`[3,4]` の stack は shape `[2,2]` である。出力 `(q,c)` の q=0 なら a、q=1 なら b を選び、残りの `(c)` でその入力だけを読む。出力 `(1,0)` は b の位置0から3になり、保存先は `1*2+0=2` である。初版は q ごとの分岐を作り、選ばれない入力からの LOAD を分岐外へ出さない。各入力の pad 条件も、その入力が選ばれた分岐の中で評価する。

`tests/movement.rs` では flip、二重 flip、stack、flip を含む入力の stack を小配列で比較する。異なる shape の入力、空の入力列、範囲外 axis の診断も確認する。要素数0では座標反転の式を評価せず空を返す。graph dump に FLIP と STACK が残り、kernel dump では入力選択と添字へ下がったことを確認すれば、元グラフを壊して実体化していないと判断できる。以後の Tensor メソッドはこの builder を呼ぶだけで同じ C 実行経路を使える。


`src/rangeify.rs` の FLIP と先頭軸 STACK を下げた結果を、二入力の C で確認する。ここでは stack の二番目の入力だけを flip する。

組込み用の抜粋。

FLIP/STACK の lowering 結果を C renderer が出力する例。

```c
// src/renderer/c.rs が生成するコード。生成先: <一時ディレクトリ>/flip_stack.c
void flip_stack(const float *a, const float *b, float *out) {
	for (int q = 0; q < 2; q++) {
		for (int c = 0; c < 2; c++) {
			float value;
			if (q == 0) {
				value = a[c];
			} else {
				int flipped = 2 - 1 - c;
				value = b[flipped];
			}
			out[q * 2 + c] = value;
		}
	}
}
```

`q` は stack の新しい軸、`c` は元入力の座標である。q=0 なら a の読出しだけ、q=1 なら b の反転した読出しだけを行う。反転の `2-1-c` は元の軸長から作る。

同じ入力を追う例。

```text
a=[1,2], b=[3,4]
STACK(a, FLIP(b)) -> [[1,2],[4,3]]
(q,c)=(1,0) -> b[1] -> out[2]=4
```

STACK の選択と FLIP の座標変換は別の処理であり、選ばれない入力の valid 条件や LOAD を評価する必要はない。

- [ ] flip の座標を `size-1-coordinate` にする。
- [ ] flip の軸の範囲と重複を検証する。
- [ ] stack の先頭座標で入力を選び、残りの座標を各入力に渡す。
- [ ] 入力選択の分岐内だけで該当入力を LOAD する。
- [ ] shape が異なる stack と、未知の軸指定を拒否する。

予定例 `movement_flip_stack` は `[3,2,1]` と `[[1,2],[3,4]]` を表示する。
二重 flip は元の `[1,2,3]` に戻る。
各コミットで graph、rangeify、linearize、C、clang、固定 ABI の経路を通す。

## 完了確認

以下は追加予定の example とテストを実装した後に実行するコマンドである。

```sh
# 作業ディレクトリ: chibigrad/
cargo test
cargo run --example movement_broadcast -- --backend cpu --dump-kernel
cargo run --example movement_reshape -- --dump-schedule --dump-source
cargo run --example movement_pad -- --check-bounds
cargo run --example movement_flip_stack -- --dump-graph
```

- [ ] 整数で表せる上記の期待値は `f32` でも厳密に一致する。
- [ ] kernel IR に抽象 movement が残らず、添字式と条件付き LOAD になる。
- [ ] 元グラフの movement ノードは実行後にも残る。
- [ ] `tests/movement.rs` に不正 shape、軸重複、負の区間の拒否例を追加する。
- [ ] 同じ NodeId を異なる座標で読む例が、順序と値の両方で正しい。
- [ ] 空の reshape、permute、shrink はメモリアクセスとゼロ除算を起こさない。

## tinygrad ソースとの対応

参照 commit は `c22edff14c79310a9b2df62a78819cf77c8289fa` である。

| 実在するシンボル | ソース | 役割 | chibigrad での簡略化 |
|---|---|---|---|
| `_apply_reshape` | [indexing.py](../../tinygrad/tinygrad/schedule/indexing.py#L147) | 座標を平坦化して入力 shape に復元する | 静的 shape の除算と剰余から始める |
| `apply_movement_op` | [indexing.py](../../tinygrad/tinygrad/schedule/indexing.py#L163) | shrink、permute、pad、reshape の添字変換 | zero pad と半開区間に限定する |
| `MovementMixin.reshape` | [movement.py](../../tinygrad/tinygrad/mixin/movement.py#L163) | shape 検証と RESHAPE の生成 | `-1` 推論は後から加える |
| `MovementMixin.flip` | [movement.py](../../tinygrad/tinygrad/mixin/movement.py#L232) | pad と reshape を合成して反転する | 専用 FLIP を添字反転へ下げてもよい |
| `MovementMixin.stack` | [movement.py](../../tinygrad/tinygrad/mixin/movement.py#L269) | STACK と permute で新しい軸に並べる | 先頭軸だけから始める |

## 今は追加しないもの

symbolic shape、任意 stride の外部 view、負の step の一般 slice は [20](20-symbolic-shapes.md) 以降の拡張とする。
Tensor API と matmul、conv は [11](11-tensor-gemm-conv.md) に進める。
movement の微分は [13](13-autodiff.md) で純粋グラフに対して追加する。
