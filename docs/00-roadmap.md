# chibigrad を小さな実行可能なステップで作る

Rust で教育用 ML コンパイラを作るための TODO である。まず小さな UOp を C にして実行する。その経路を残したまま、配列、ループ、shape、自動微分、GPU、実モデルへ広げる。

全体の背景は [全体計画](chibigrad-plan.md) に残す。このファイルを入口に、番号順の詳細手順を進める。各詳細手順のチェックボックスは、実装と確認が完了してから記録する。この文書作成時点では chibigrad の実装はない。

## 一つのコミットで実行結果まで確認する

1. 対象ステップの前提と tinygrad 対応表を読む。
2. 最初の小コミットだけを実装する。
3. 既存テストと、そのコミットの入力例を実行する。
4. 出力、エラー、IR dump を確認し、実行記録を残す。
5. 次の小コミットへ進む。

全ファイルの enum、trait、空の pass を先に作らない。最初の実装は `Const(7)` を生成 C 経由で実行できればよい。後の機能が必要とするデータは、その機能を実行できる変更と一緒に追加する。

ここでいう全体のパイプラインは、その時点で対応する入力から実行結果までの経路を意味する。初回から GPU や高水準 Tensor API を実装する意味ではない。

各詳細手順の `cargo run --example ...`、テストファイル、Rust API は、そのステップで作る予定のものである。現在存在する実行コマンドや検証済み実装として扱わない。手順の実装時には、記載されたコマンドが実際に動く状態にしてから完了とする。

## コードを読みながら一つずつ動かす

各コミットは、追加する処理の導入、コード、コードの読み方、小さい入力での確認、作業チェックの順で読む。本文では変数、座標、保存領域の状態がどう変わるかを追う。チェック項目へ進む前に、掲載した入力がどの値を経て出力になるかを説明できることを確かめる。

コードブロックには次の区別がある。

- **単独で実行できる例。** 必要な型と `main`、または test を含む Rust の小プログラムである。それぞれを別のファイルに保存して実行する。複数の例の `main` を一つのファイルへ連結しない。
- **組込み用の抜粋。** その章までに作った `Graph`、builder、runtime へ追加する型や関数である。直前の説明にある型と補助関数を使う。ファイル全体ではないため、そのブロックだけをコンパイルする使い方はしない。
- **外部 API の疑似コード。** Metal、OpenCL、ファイル形式の parser などを接続する順序と契約を示す。選ぶ binding やライブラリの正確な関数名へ置き換える。疑似コードがそのままビルドできるとは扱わない。

Rust の単独例は、たとえば次のコマンドで確認する。これは実装予定の `cargo run --example ...` とは別に、本文中のコードを読むための使い方である。

```sh
rustc --edition=2021 /tmp/chibi_example.rs -o /tmp/chibi_example
/tmp/chibi_example
```

`#[test]` を含む単独例は `rustc --edition=2021 --test` で test 実行ファイルを作る。assertion が通ったことと、章全体の compiler pipeline が完成したことは区別する。C、MSL、OpenCL のコードは、それぞれ対応する生成先の言語であり、Rust のファイルへ直接貼らない。

先に単独例でループや座標変換の意味を確かめ、その処理を組込み用の抜粋の位置へ移す。最後に既存の graph、lowering、renderer、runtime を通す確認へ戻る。GPU の単独ホスト例が通っても、GPU kernel の実機確認は対応する章の手順で別に行う。

## ファイルツリーと配置コメントを読む

各章の「この章のファイル配置」は、その章の終わりまでに触る関連ファイルを示す。各コミット直下のツリーは、その変更で編集・追加・確認する部分だけを示す。前の章で作った他のファイルは残る。ツリーは教材の配置案であり、現在のcheckoutに全ファイルが存在するという意味ではない。

01では `examples/scalar_const.rs` の中だけで小さな全体を動かす。02で実行時入力と共有ライブラリが必要になった時に、共通処理を `src` へ移す。後の章でも、新しい処理が動くコミットと一緒にモジュールを追加する。各章のツリーを見て、将来の空ファイルを先に用意する必要はない。

コードの先頭にあるパスを、配置先として読む。

```text
// src/uop.rs（既存ファイルへ組み込む抜粋）
// examples/book_03_intern.rs（単独で実行する例）
// src/renderer/c.rs が生成するコード。生成先: <一時ディレクトリ>/kernel.c
// src/runtime/metal.rs（外部APIの疑似コード）
```

`src` の抜粋は、そのファイルの型や関数へ追加する。置換する関数は直前の説明に従う。一つのブロック内で配置コメントが変わる場合は、次のコメントまでをそれぞれのファイルに分ける。単独例の `main` や検証専用の小さい型を、本体のモジュールへ重複して貼り付けない。`examples/book_*.rs` は部品の意味を確かめるプログラムであり、通常のコンパイラ経路を動かす `examples/scalar_inputs.rs` などとは分けて保存する。

新規モジュールは `src/lib.rs`、または `src/runtime/mod.rs` のような親の `mod.rs` から宣言する。初登場するコミットに追記する宣言を掲載する。既存の宣言や関数は残す。`src/main.rs` はCLIが必要になるまでそのままでよく、同じ本体を `main.rs` と `lib.rs` の両方から別々の `mod` 宣言で読み込まない。exampleはライブラリの公開入口を使い、内部の補助関数は各モジュールへ置く。

全体計画の概略ツリーにある `autograd.rs` は、詳細手順では `src/autodiff.rs` に相当する。同じ微分器を二つのファイルへ分けず、13章で作った `src/autodiff.rs` を24章でも拡張する。

15章と16章ではruntimeを `metal`、`opencl` のCargo featureで有効にする。以降、GPUの予定コマンドには `--features metal` または `--features opencl` を追加する。Metalのruntimeと専用testはmacOSでもあることを条件にする。CPUだけの通常テストはこれらのfeatureを付けずに行い、rendererの文字列生成とGPU APIの接続を分けて確認する。

GPU専用exampleは追加するコミットで `Cargo.toml` の `[[example]]` に `required-features` を設定する。通常の `cargo test` がそのexampleをビルドする場合にも、GPUの依存を要求しないためである。Metal専用exampleはmacOS用のmainと、それ以外で利用不可を表示するmainを条件で分ける。専用のuseと補助関数も同じOS条件で囲む。後のCPU/GPU共通exampleでは、GPU runtimeのimportと呼出しを同じfeature・OS条件の内側に置き、無効なbackendを選んだときは利用不可のエラーを返す。ホスト上の寸法計算などを試す `book_*.rs` は標準ライブラリだけで動くため、このGPU用featureを要求しない。

生成C、MSL、OpenCLのブロックは、コメントにあるrendererが作る出力である。生成された文字列を一時ファイルやbackendのsource compileへ渡す。RustのソースファイルへCやshader本体をそのまま貼らない。外部APIの疑似コードは指定runtimeなどへ実装する呼出し順を示しており、選んだbindingの型とエラー処理へ書き直す。

IR、期待値、実行出力の `text` ブロックは確認用であり、配置先のRustファイルへ貼るコードではない。シェルブロックは冒頭のコメントにあるプロジェクトルートで実行する。本文の `rustc` 単独例だけを試す場合は、保存したexampleのパスへコマンドを合わせる。`Cargo.toml` の既存package設定は維持し、ライブラリの公開入口をexampleからimportするときは実際のcrate名を使う。


## 実装する順序

元計画の11章は維持し、次の25ステップに分割する。追加した後半は元計画の12〜15章に対応する。番号順に読むと、各ステップの前提を先に実装できる。

| TODO | 動くようにするもの | 元計画の章 |
|---|---|---|
| [ ] [01. scalar の C 生成](01-scalar-c.md) | `7`、次に `(2+3)*4 = 20` をコンパイルして実行する | 1 |
| [ ] [02. 実行時の入力](02-runtime-inputs.md) | 同じ生成コードへ異なる値を渡す | 1 |
| [ ] [03. UOp と書き換え](03-uop-rewrites.md) | 型、共有、interpreter、整数式と定数の簡約を加える | 2 |
| [ ] [04. 関数と STACK](04-functions-stack.md) | PARAM、CALL、STACK、INDEX と小さな UIR を扱う | 2 |
| [ ] [05. メモリと runtime](05-memory-runtime.md) | LOAD、STORE、AFTER と共有ライブラリ実行を扱う | 3 |
| [ ] [06. ループ](06-loops.md) | RANGE、END と入れ子のスコープを扱う | 4 |
| [ ] [07. 累積と手書きカーネル](07-scalar-reductions.md) | 行和、GEMM、convolution を低水準 IR で書く | 4 |
| [ ] [08. shape と rangeify](08-shapes-rangeify.md) | 一次元の式と REDUCE からループを生成する | 5 |
| [ ] [09. fusion と fission](09-fusion-fission.md) | カーネル境界、中間保存、実行順を決める | 6 |
| [ ] [10. movement](10-movement.md) | 多次元の座標を元の入力座標に変換する | 7 |
| [ ] [11. Tensor と GEMM・conv](11-tensor-gemm-conv.md) | 高水準の式から行列積と畳み込みをコンパイルする | 7 |
| [ ] [12. 数学関数](12-math-functions.md) | SQRT、SIN、LOG2、EXP2 の実装と分解を扱う | 7の補足 |
| [ ] [13. 自動微分](13-autodiff.md) | 純粋なグラフから勾配グラフを作る | 8 |
| [ ] [14. SGD と MNIST](14-sgd-mnist.md) | 再利用する実行計画で小さなモデルを学習する | 8・11 |
| [ ] [15. Metal](15-metal.md) | 出力要素をスレッドへ割り当てて実機で実行する | 9 |
| [ ] [16. OpenCL](16-opencl.md) | 同じ計算を OpenCL の通常 API で実行する | 9 |
| [ ] [17. ループ最適化](17-loop-optimizations.md) | 分割、UPCAST、ベクトル化、CPU の THREAD を扱う | 10 |
| [ ] [18. 並列 reduction](18-parallel-reduction.md) | local memory と barrier を使う | 10 |
| [ ] [19. 高速 GEMM・conv](19-fast-gemm-conv.md) | tile、メモリ階層、再利用、測定を扱う | 10 |
| [ ] [20. symbolic shape](20-symbolic-shapes.md) | 制約付きの実行時サイズと境界を扱う | 12 |
| [ ] [21. モデル I/O](21-model-io.md) | 重み保存、URL、SHA-256、checkpoint を扱う | 13 |
| [ ] [22. Transformer と Qwen](22-transformer-qwen.md) | 小モデルから実チェックポイントの推論へ広げる | 14 |
| [ ] [23. flash attention](23-flash-attention.md) | attention 行列を全面保存しない実装へ変換する | 14 |
| [ ] [24. LLM 学習](24-llm-training.md) | 小さな言語モデルの学習と再開を通す | 14 |
| [ ] [25. ハードウェア拡張](25-hardware-extensions.md) | 行列アクセラレータと独自の拡張課題を扱う | 15 |

14で静的 CPU の小さな学習系、16でCPU・Metal・OpenCLの基本経路、19で基本最適化までが動く。20〜25は Gist の後半の内容を扱う追加段階であり、前半の開始条件にはしない。後半もこの計画の対象に含める。

## Gist の内容と対応を確認する

参照するのは [Compilers for Machine Learning](https://gist.github.com/geohot/4768597d9dc536446ee2d5de1f29e89d) の内容である。授業の日程、成績配分、SOTA 性能の達成はこの教材の受入条件にしない。扱う内容を以下の手順へ対応付ける。

| 扱う内容 | 詳細手順 |
|---|---|
| UOp、型、共有、定数畳み込み、記号的な簡約 | 01〜04、20 |
| PARAM と CALL、STACK と INDEX、テスト用 ASSERT | 02〜04 |
| メモリ、所有権、addrspace、accesses、storage_base | 05 |
| LOAD と STORE の順序、AFTER、アクセス境界 | 05〜07 |
| RANGE と END、入れ子、GEMM と conv | 06〜07、11 |
| 一次元 broadcast、REDUCE、rangeify | 08 |
| STAGE、materialization、fusion と fission | 09 |
| 多次元 movement、合成演算 | 10〜11 |
| 数学関数と近似・命令分解 | 12 |
| GRADIENT に相当するグラフ変換と DETACH | 13 |
| MNIST と演算量の見積り | 14、17、19 |
| URL と SHA-256 を持つ入力・重みの取得 | 21 |
| GPU、GLOBAL と LOCAL、共有メモリ | 15〜16、18〜19 |
| UPCAST、CPU の並列化、メモリ階層 | 17〜19 |
| 動的な記号的サイズ | 20 |
| 実チェックポイントの Qwen 推論 | 21〜22 |
| flash attention | 23 |
| LLM の学習と再開 | 24 |
| 行列アクセラレータ、別ハードウェアや論文の拡張 | 25 |

同じ意味を API や test helper で実現する場合は、詳細手順に表現上の違いを書く。例えば現在の tinygrad に `ASSERT` や `GRADIENT` という演算がなくても、存在するかのように引用しない。テキスト UIR の完全な互換性も、内容の対応とは別の要件として扱う。

## tinygrad を読みながら実装する

参照する checkout は `../../tinygrad`、解析対象の commit は `c22edff14c79310a9b2df62a78819cf77c8289fa` である。各詳細手順の対応表に、実際に読んだシンボルと行番号を記す。後で参照先が移動した場合は、この commit に戻って確認する。

| 読む場所 | 注目する責務 |
|---|---|
| [UOp](../../tinygrad/tinygrad/uop/ops.py#L181) | ノードの共有、導出情報、依存関係 |
| [UOp の検証](../../tinygrad/tinygrad/uop/spec.py#L34) | 段階ごとに許される形を確認する |
| [UIR の表示](../../tinygrad/tinygrad/uop/render.py#L59) | 人が読める graph dump を作る |
| [rangeify](../../tinygrad/tinygrad/schedule/rangeify.py#L341) | 添字変換とカーネル境界 |
| [添字変換](../../tinygrad/tinygrad/schedule/indexing.py#L146) | movement を座標の式に変換する |
| [codegen](../../tinygrad/tinygrad/codegen/__init__.py#L269) | 最適化と lowering を分ける |
| [実行](../../tinygrad/tinygrad/engine/realize.py#L284) | 計算とバッファの binding を結び付ける |
| [数学 API](../../tinygrad/tinygrad/mixin/op.py#L390) | 少数の演算から高水準の演算を組み立てる |

各手順では「tinygrad にある責務」「chibigrad に採用する仕組み」「今は移植しない複雑さ」を分ける。現在の tinygrad にない小さな parser や CPU threading を追加する場合は、教育実装側の設計として明記する。

全体計画の既存ソースリンクは、この docs ディレクトリから sibling の tinygrad を指す相対リンクへ修正する。シンボルの位置を読むためのリンクであり、chibigrad に同名ファイルが存在するという意味ではない。

## 毎回の完了条件

- [ ] 各小コミットで `cargo test` が通り、その時点の生成コードが実行できる。
- [ ] 新しい機能の最小入力と期待値を確認する。
- [ ] 不正な型、shape、添字のうち今回扱うケースを拒否できる。
- [ ] dump で、どの変換が何を取り除いたか確認する。
- [ ] interpreter、未最適化経路、tinygrad など、実装と異なる経路を比較先に使う。
- [ ] CPU と GPU の確認記録を分ける。CPU の成功で GPU を完了にしない。
- [ ] 性能を扱う手順では、コンパイル、転送、起動、実行、同期を分けて記録する。
- [ ] ステップの実装が終わるまで、この計画のチェックボックスを完了にしない。

CPU だけの通常テストが GPU SDK を要求しない構成にする。Metal、OpenCL、行列アクセラレータの確認は対応する機器で実行し、その環境を記録する。機器がない場合は未確認と残す。

## 元計画から変更した箇所

- 第1章を、定数のコード生成と実行時入力の二段階へ細分化する。
- 元の11章を詳細手順01〜19へ対応付け、章の説明は残す。
- 数学関数と convolution の実装を明示し、Gist の内容との抜けを埋める。
- 元の対象外項目のうち symbolic shape と行列アクセラレータは、後半の追加段階へ移す。
- 第12〜15章として symbolic shape、I/O、実モデルと attention、ハードウェア拡張を追記する。

最初のステップで、汎用的な IR framework、全 backend、全 dtype、メモリアリーナを先に完成させない。必要な機能を実行例と一緒に追加する方針は変えない。

## 参照する資料

- [全体計画](chibigrad-plan.md)。元の tinygrad 解析と設計方針を読む。
- [Compilers for Machine Learning](https://gist.github.com/geohot/4768597d9dc536446ee2d5de1f29e89d)。扱う内容の範囲を確認する。
- [chibicc の説明](https://github.com/rui314/chibicc/blob/main/README.md)。小さな入力から機能を追加する進め方を参照する。履歴の書換え方針はこの計画には採用しない。

Foundational Thinking に沿って、各ステップで追加するデータと不変条件を先に記す。Sequence work into verifiable units に沿って、小さな実装と実行確認を一つの単位にする。
