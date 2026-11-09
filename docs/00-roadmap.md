# chibigrad を番号順に写経して作る

01章から25章までのコードを保存し、各確認コマンドを実行して、一つのRustプロジェクトを育てます。定数7のC生成から始め、実行時入力、配列、Tensor、自動微分、GPU、モデル保存、Qwenの推論と学習へ進みます。

このcheckoutの `src/main.rs` と `Cargo.toml` は写経を始める前の状態です。教材の完成コードは各章のファイル指定付きブロックにあります。写経先には別の空ディレクトリを用意します。crate名は既存の `chinigrad` を使います。

## 保存先と操作を読む

コードブロックの `file=` が写経先プロジェクトからの相対パスです。親ディレクトリがなければ作ります。

| 指定 | 操作 |
|---|---|
| `mode=write` | 指定ファイル全体をコードで置き換える |
| `mode=append` | 指定ファイルの末尾へコードを追加する |
| `mode=replace item="fn name("` | 指定の同名関数全体を置き換える。直前の `#[test]` などの属性は残す |
| `sh check` | 直前までの保存を終えてから、写経先プロジェクトのrootで実行する |
| `sh check requires=metal` | Metalの実機がある場合に実行する |

置換対象の `item` は一箇所だけに一致します。同じ関数名がある場合は、掲載した宣言の先頭全体で対象を特定します。生成C、IR、図、期待出力のブロックはRustファイルへ保存しません。`file=` のないコマンドや確認出力もソースへ貼りません。

同じ機能を異なるTODOの観点から説明する節では、既出の関数を再掲して置き換えます。実装するTODOには実装関数を、期待値や拒否条件を確かめるTODOにはテスト関数を掲載します。テストの補助関数と型は前の節で保存済みです。確認コマンドはコードが揃う位置に置いてあります。

## 実行環境を用意する

Rust 1.98.1とC11対応Clangで確認します。CPU runtimeは64bitのmacOSとLinuxを対象にします。macOSではXcode Command Line ToolsとmacOS SDKが必要です。掲載したdriverが `xcrun` でSDKを取得します。LinuxではClang、標準C開発ヘッダと動的loaderが必要です。

15章以降はCargoが固定versionの依存を取得するためネットワークが必要です。Metalは対応するmacOSとdevice、OpenCLは利用可能なplatform、device、driverを別に用意します。featureを有効にしてビルドできても実機計算の確認にはなりません。

実データを使う確認は条件付きです。MNIST、公開Qwen重み、tokenizer、公式reference用Python環境の準備は対応章に掲載します。Qwenの重みは約0.99GBです。BF16をf32へ展開し、コピーとworkspaceも作るので、ファイル容量以上のRAMが必要です。全parameterのAdamW学習は24章の見積りを先に実行します。

## 章を番号順に進める

- [01. 定数 7 を C にして実行し、加算と乗算で 20 を返す](01-scalar-c.md)
- [02. 同じ生成コードへ実行時入力を渡して 20 と 12 を返す](02-runtime-inputs.md)
- [03. scalar の型と意味を決め、interpreter と rewrite を比較する](03-uop-rewrites.md)
- [04. 純粋関数を CALL し、STACK と INDEX を UIR から実行する](04-functions-stack.md)
- [05. バッファの LOAD と STORE を安全な順序で CPU 実行する](05-memory-runtime.md)
- [06. RANGE と END から安全なループを生成する](06-loops.md)
- [07. scalar の累積ループで総和、行和、GEMM と convolution を動かす](07-scalar-reductions.md)
- [08. 一次元の shape と REDUCE を添字付きループへ変換する](08-shapes-rangeify.md)
- [09. 融合と分割で閉じた kernel graph を作り、Plan を再実行する](09-fusion-fission.md)
- [10. 多次元の座標変換で movement を実行する](10-movement.md)
- [11. Tensor の合成演算で matmul と conv を動かす](11-tensor-gemm-conv.md)
- [12. 数学関数を実行し、近似と値域の責任を分ける](12-math-functions.md)
- [13. 純粋なグラフから勾配グラフを作って実行する](13-autodiff.md)
- [14. 同じ実行計画を再利用して SGD と MNIST 学習を動かす](14-sgd-mnist.md)
- [15. Metal の公開 API で実際の Graph を実行する](15-metal.md)
- [16. OpenCL の公開 API で同じ Graph を実行する](16-opencl.md)
- [17. loop split、unroll、vector、CPU worker を実行する](17-loop-optimizations.md)
- [18. LOCAL と一様な barrier で並列縮約する](18-parallel-reduction.md)
- [19. GEMM と conv のタイル、確保の再利用、compile cache を動かす](19-fast-gemm-conv.md)
- [20. 上限付きの shape を束縛して同じ kernel を動かす](20-symbolic-shapes.md)
- [21. モデルの重みを保存し、検証したファイルから再実行する](21-model-io.md)
- [22. 小さな Transformer を通し、実際の Qwen checkpoint で次の token を計算する](22-transformer-qwen.md)
- [23. 通常 attention と online attention を同じモデルで比べる](23-flash-attention.md)
- [24. 小さな LLM を学習し、checkpoint から同じ更新を再開する](24-llm-training.md)
- [25. 実行して確認した行列 capability と RMSNorm 拡張を使う](25-hardware-extensions.md)

## 写経した結果を自動で再現する

この節のコマンドは、教材checkoutのrootで実行します。`tools/replay_docs.py` はMarkdownの `file=` ブロックだけを、指定した別ディレクトリへ順に適用します。出力先には空のディレクトリを指定します。既存ファイルを持つ出力先と、このcheckout内への出力は拒否します。

```sh
uv run python tools/replay_docs.py --self-test
uv run python tools/replay_docs.py --lint
uv run python tools/replay_docs.py --output /tmp/chibigrad-book --check --report /tmp/chibigrad-book-report.json
```

3行目はCPUの確認を実行します。`requires` のある確認は、指定した能力を持つ環境で `--capability metal`、`--capability opencl`、`--capability mnist`、`--capability qwen`、`--capability loopback` を追加して実行します。複数ある場合はオプションを繰り返します。各章のデータ準備コマンドと実行時の環境変数も先に設定します。実機・データがなければ結果は `NOT_RUN` です。

`--through 02` なら02章までを展開します。途中の状態を比較する場合は、別の空ディレクトリを指定します。`--lint` の697項目は見出しとコード配置の構造検査です。数値の正しさは、実行したテストと外部referenceとの比較で判断します。

## 全章に適用する確認

以下の8節は各章の実装を終えた後に使います。コマンドは、特記しない限り写経先のrootで実行します。

### 00-001 各小コミットで `cargo test` が通り、その時点の生成コードが実行できる。

<!-- todo:00-001 -->

章末にそれまでのテストをまとめて実行します。本文中の期待値をホストの代入だけで作らず、生成Cを実行するfixtureを通します。

```sh
cargo test --all-targets
```

### 00-002 新しい機能の最小入力と期待値を確認する。

<!-- todo:00-002 -->

01章の小入力を再実行します。式の入力値と実際のstdoutを照合する例です。

```sh
cargo run --example scalar_const -- 7
cargo run --example scalar_arithmetic -- 2 3 4
```

### 00-003 不正な型、shape、添字のうち今回扱うケースを拒否できる。

<!-- todo:00-003 -->

全テストには型、shape、添字、署名、容量、ファイル形式の拒否条件を含みます。正常系だけで確認を終えません。

```sh
cargo test --all-targets
```

### 00-004 dump で、どの変換が何を取り除いたか確認する。

<!-- todo:00-004 -->

変換前のgraphと生成Cを表示します。後の章ではPlanのkernel数、signature、workspace、実行時shapeも確認します。

```sh
cargo run --example scalar_arithmetic -- 2 3 4 --dump-graph --dump-source
```

### 00-005 interpreter、未最適化経路、tinygrad など、実装と異なる経路を比較先に使う。

<!-- todo:00-005 -->

Tensorのテストは独立interpreterと生成nativeコードを比較します。Qwenでは固定revisionの公式実装と全layer出力・logitsを比較します。

```sh
cargo test --all-targets
```

### 00-006 CPU と GPU の確認記録を分ける。CPU の成功で GPU を完了にしない。

<!-- todo:00-006 -->

CPUのテストとGPUのテストを別に実行します。デバイスがなければ後の2行を未実行と記録します。

```sh
cargo test --all-targets
cargo test --features metal --test book_15 -- --ignored
cargo test --features opencl --test book_16 -- --ignored
```

### 00-007 性能を扱う手順では、コンパイル、転送、起動、実行、同期を分けて記録する。

<!-- todo:00-007 -->

15、16章の `gpu::Timings` はcompile、allocation、transfer、kernel、synchronization、totalを区別します。device eventが得られないkernel時間は `None` のままです。計測値の検査と実機実行を分けます。

```sh
cargo test --test book_15
cargo test --test book_16
```

### 00-008 ステップの実装が終わるまで、この計画のチェックボックスを完了にしない。

<!-- todo:00-008 -->

完了は検証結果で記録します。教材checkoutのrootで構造検査を実行し、続いて全章を空ディレクトリへ再現します。条件付き確認の `NOT_RUN` を完了に読み替えません。

```sh
uv run python tools/replay_docs.py --lint
uv run python tools/replay_docs.py --output /tmp/chibigrad-final-check --check --report /tmp/chibigrad-final-check.json
```

詳細な責務と設計上の選択は [全体構成](chibigrad-plan.md) にまとめます。
