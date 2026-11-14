# UOpを小さく作り、計算できる範囲を増やす

Rustで共通のUOpを作り、01〜25章を順に拡張します。各節で使う演算・型・関数だけを実装します。01.01はConstだけ、01.02でAdd、01.03でMulを追加します。

前の節のコードに追記・置換し、確認を実行してから次へ進みます。後半のenum variant、shape、scheduler、GPU、ELFを先に保存する手順はありません。掲載コードが実装の正本です。このcheckoutのsrc/main.rsとCargo.tomlは写経前の状態を保ちます。

参照するtinygradは `c22edff14c79310a9b2df62a78819cf77c8289fa` です。共通UOpの構成を参考にし、教材では動く最小形から育てます。

## 保存操作を読む

フェンスには言語名だけを書きます。先頭コメントの `file=` が写経先からの相対パスです。Rustとjsoncでは `//`、TOML・Python・textでは `#` を使います。保存するときはメタデータ行を除きます。

| 指定 | 操作 |
|---|---|
| `mode=write` | ファイル全体を保存する |
| `mode=append` | ファイル末尾へ追記する |
| `mode=append within="impl UOp"` | 指定したimplの閉じ括弧の直前へ追記する |
| `mode=replace item="pub enum Ops"` | 指定した定義を置き換える |
| `within=` と `item=` | 指定したimpl内の関数を置き換える |
| `mode=delete` | 不要になったファイルを削除する |
| `# check step=01.01` | その節の保存後に、写経先rootで確認する |
| `requires=metal` | Metalを実行できる環境で確認する |

確認ブロックは一行ごとに独立したコマンドです。複数のcargo testはそれぞれ実際にテストが実行されたかを確認します。確認ブロック間でshell変数を引き継ぎません。

## 実行環境を用意する

Rust 1.98.1とClangで確認しています。01章は生成実行ファイル、02章から17.03までは共有ライブラリを使います。macOSでは既定でsystem clangを選び、CCで変更できます。共有ライブラリのロードは初期化コードも実行するため、生のCpuProgram::compileとrunにはunsafe契約があります。

17章でClangのELF objectを読み込むloaderへ移行します。macOSでもELF targetを指定します。CPUは64bit macOS・Linux、objectとrelocationはx86-64・AArch64を扱います。実行検証はApple Silicon上で行い、x86-64は生成objectとfixtureで検査します。

Metalは対応するmacOS device、OpenCLは利用できるplatformとdeviceが必要です。15・16章はRustから直接転送・起動し、17章でhost queueもUOpへ移します。25章のWMMAは対応deviceで別に確認します。

外部Rust依存は21章以降で追加します。初回の依存取得にはネットワークが必要です。実MNIST、公式tokenizer、公開Qwen重みは外部データです。fixtureの成功と実データの成功を分けて記録します。

## 章を順に進める

- [01. 定数だけの UOp から始めて Add と Mul を足す](01-scalar-c.md)
- [02. 実行時入力と共有ライブラリを追加する](02-runtime-inputs.md)
- [03. UOpの共有・型・scalar実行・パターン書換えを追加する](03-uop-rewrites.md)
- [04. CALL のスコープ、STACK の選択、UIR の保存と復元を実装する](04-functions-stack.md)
- [05. INDEX・LOAD・STORE・AFTER でメモリを扱う](05-memory-runtime.md)
- [06. RANGE と END のスコープと guard を検証する](06-loops.md)
- [07. LOAD・STORE・AFTER の累積で縮約を実行する](07-scalar-reductions.md)
- [08. shape 付き Tensor を同じ UOp の添字とループへ下げる](08-shapes-rangeify.md)
- [09. kernel の境界、依存、schedule とメモリ計画を作る](09-fusion-fission.md)
- [10. movement・broadcast・gather・scatter を添字へ変換する](10-movement.md)
- [11. matmul と convolution を Tensor の合成で作る](11-tensor-gemm-conv.md)
- [12. 数学演算を通常 UOp の多項式へ分解する](12-math-functions.md)
- [13. UOp の逆伝播から勾配の式を作る](13-autodiff.md)
- [14. 同じコンパイラで SGD と MNIST を学習する](14-sgd-mnist.md)
- [15. Metalの転送とカーネル実行を追加する](15-metal.md)
- [16. OpenCLを追加し、同じTensorをGPU実行する](16-opencl.md)
- [17. ELFとhost queueを作り、loop・vector・CPU workerを追加する](17-loop-optimizations.md)
- [18. local buffer と barrier で並列縮約する](18-parallel-reduction.md)
- [19. GPUのGEMM・畳込みとcompile cacheを追加する](19-fast-gemm-conv.md)
- [20. symbolic shape を束縛し generic と特化版を再利用する](20-symbolic-shapes.md)
- [21. モデル state・safetensors・manifest を検証する](21-model-io.md)
- [22. Qwen の全層、tokenizer、KV cache を接続する](22-transformer-qwen.md)
- [23. online attention を共通 UOp と協調 GPU kernel で実行する](23-flash-attention.md)
- [24. 全 parameter を学習し checkpoint から同じ更新を再開する](24-llm-training.md)
- [25. 実機で行列命令と RMSNorm の capability を確認する](25-hardware-extensions.md)

## 空のディレクトリへ再現する

教材checkoutのrootで実行します。節の途中まで再現できます。

```sh
uv run python tools/replay.py --self-test
uv run python tools/replay.py --lint
book_dir=$(mktemp -d /tmp/chibigrad-book.XXXXXX)
uv run python tools/replay.py --output "$book_dir" --through-step 01.01 --check
```

01.01のsrc/uop.rsにはConstだけがあります。別の空ディレクトリに01.02まで再現するとAdd、01.03までならMulも加わります。未知の節番号と、節の順序の重複はエラーです。

全章を確認する場合は `--through-step` を省略します。Metal・OpenCLを実行できる場合は `--capability metal --capability opencl` を追加します。対応する行列命令を実行できる場合は `--capability wmma` も追加します。指定しなかった能力の確認はNOT_RUNです。CPUの成功でGPUの成功を代用しません。

各章末の累積状態を別々に調べるには、次を実行します。

```sh
uv run python tools/verify_tutorial.py --through 25 --report /tmp/chibigrad-prefixes.json
```

tools/replay.pyは各節でOps・Arg・DType・Axisとmoduleの導入位置を検査します。tools/publish_book.pyはMarkdownから章・節・要件の索引を作ります。完成crateを読み込んだり、掲載コードを書き換えたりしません。

```sh
uv run python tools/publish_book.py --write
uv run python tools/publish_book.py --check
```

元の697件の要件IDとTODO文はrequirements.jsonに保存します。各章末の対応付けは、697件すべてを個別に数値検証したという意味ではありません。各節の実行結果と、条件付き確認の記録を併せて判断します。

## 要件との対応

### 00-001 各小コミットで `cargo test` が通り、その時点の生成コードが実行できる。

<!-- todo:00-001 -->

`tools/replay.py --self-test` と `--lint` が保存操作・節番号・導入位置・テスト実行数を検査する。各章の数値結果は隣接する確認コマンドで調べる。

### 00-002 新しい機能の最小入力と期待値を確認する。

<!-- todo:00-002 -->

`tools/replay.py --self-test` と `--lint` が保存操作・節番号・導入位置・テスト実行数を検査する。各章の数値結果は隣接する確認コマンドで調べる。

### 00-003 不正な型、shape、添字のうち今回扱うケースを拒否できる。

<!-- todo:00-003 -->

`tools/replay.py --self-test` と `--lint` が保存操作・節番号・導入位置・テスト実行数を検査する。各章の数値結果は隣接する確認コマンドで調べる。

### 00-004 dump で、どの変換が何を取り除いたか確認する。

<!-- todo:00-004 -->

`tools/replay.py --self-test` と `--lint` が保存操作・節番号・導入位置・テスト実行数を検査する。各章の数値結果は隣接する確認コマンドで調べる。

### 00-005 interpreter、未最適化経路、tinygrad など、実装と異なる経路を比較先に使う。

<!-- todo:00-005 -->

`tools/replay.py --self-test` と `--lint` が保存操作・節番号・導入位置・テスト実行数を検査する。各章の数値結果は隣接する確認コマンドで調べる。

### 00-006 CPU と GPU の確認記録を分ける。CPU の成功で GPU を完了にしない。

<!-- todo:00-006 -->

`tools/replay.py --self-test` と `--lint` が保存操作・節番号・導入位置・テスト実行数を検査する。各章の数値結果は隣接する確認コマンドで調べる。

### 00-007 性能を扱う手順では、コンパイル、転送、起動、実行、同期を分けて記録する。

<!-- todo:00-007 -->

`tools/replay.py --self-test` と `--lint` が保存操作・節番号・導入位置・テスト実行数を検査する。各章の数値結果は隣接する確認コマンドで調べる。

### 00-008 ステップの実装が終わるまで、この計画のチェックボックスを完了にしない。

<!-- todo:00-008 -->

`tools/replay.py --self-test` と `--lint` が保存操作・節番号・導入位置・テスト実行数を検査する。各章の数値結果は隣接する確認コマンドで調べる。
