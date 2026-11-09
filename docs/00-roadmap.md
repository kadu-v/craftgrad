# chibigrad を番号順に写経して作る

01章から25章までのコードを保存し、各確認コマンドを実行して、一つのRustプロジェクトを育てます。定数7のC生成から始め、実行時入力、配列、Tensor、自動微分、GPU、モデル保存、Qwenの推論と学習へ進みます。

このcheckoutの `src/main.rs` と `Cargo.toml` は写経を始める前の状態です。教材の完成コードは各章のファイル指定付きブロックにあります。写経先には別の空ディレクトリを用意します。crate名は `chibigrad` です。

## 保存先と操作を読む

コードブロックの `file=` が写経先プロジェクトからの相対パスです。親ディレクトリがなければ作ります。

| 指定 | 操作 |
|---|---|
| `mode=write` | 指定ファイル全体をコードで置き換える |
| `mode=append` | 指定ファイルの末尾へコードを追加する |
| `mode=replace item="fn name("` | 指定した関数・型・定数の定義全体を置き換える。直前の属性は残す |
| `within="impl Type"` | 指定したimpl内で置換対象を探す |
| `arm="Op::Reduce"` | 指定した関数内のmatch分岐を、パターンとbodyごと置き換える |
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

この節のコマンドは、教材checkoutのrootで実行します。下のスクリプトを `/tmp/chibigrad_replay.py` に保存します。リポジトリへスクリプトファイルを追加する必要はありません。`/tmp/chibigrad_replay.py` はMarkdownの `file=` ブロックだけを、指定した別ディレクトリへ順に適用します。出力先には空のディレクトリを指定します。既存ファイルを持つ出力先と、このcheckout内への出力は拒否します。

```sh
uv run python /tmp/chibigrad_replay.py --self-test
uv run python /tmp/chibigrad_replay.py --lint
uv run python /tmp/chibigrad_replay.py --output /tmp/chibigrad-book --check --report /tmp/chibigrad-book-report.json
```

3行目はCPUの確認を実行します。`requires` のある確認は、指定した能力を持つ環境で `--capability metal`、`--capability opencl`、`--capability mnist`、`--capability qwen`、`--capability loopback` を追加して実行します。複数ある場合はオプションを繰り返します。各章のデータ準備コマンドと実行時の環境変数も先に設定します。大きいモデルのコンパイルを確認する場合は `--timeout 3600` のように確認一件の上限秒数を指定します。実機・データがなければ結果は `NOT_RUN` です。

`--through 02` なら02章までを展開します。途中の状態を比較する場合は、別の空ディレクトリを指定します。`--lint` の697項目は見出しとコード配置の構造検査です。数値の正しさは、実行したテストと外部referenceとの比較で判断します。

## この改訂で実行した確認

2026年11月9日に、掲載コードを空ディレクトリへ番号順に適用しました。25章、1,112回のファイル操作を通し、CPUの確認コマンド673件が成功しました。展開した最終プロジェクトの `cargo test --all-features --all-targets` は301件成功、失敗0件、ignored 50件でした。Markdown内の検証スクリプトから独立に展開した108ファイルも、テストしたファイルと全バイトが一致しました。

条件付きコマンド53件は、この通し実行では `NOT_RUN` です。内訳はMetal 23件、OpenCL 20件、Qwenデータ8件、MNIST 1件、loopback 1件です。feature付きビルドの成功やignoredを実機テストの成功として数えません。

別の検証用プロジェクトでは、固定した実Qwen重みを使い、1 tokenと2 tokenの入力について公式referenceの全24層・最終正規化・全logitsと照合しました。この確認後にcoreの安全性検査を改訂しているため、最終掲載コードそのものを使った実Qwenの再照合は未実行です。実GPU、実MNIST、実Qwenのcached decode・CLI生成・全parameter学習も未実行です。各章には、それらを実行するコードと条件を掲載しています。

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
uv run python /tmp/chibigrad_replay.py --lint
uv run python /tmp/chibigrad_replay.py --output /tmp/chibigrad-final-check --check --report /tmp/chibigrad-final-check.json
```

詳細な責務と設計上の選択は [全体構成](chibigrad-plan.md) にまとめます。

## 写経確認用スクリプト

次のコードを `/tmp/chibigrad_replay.py` に保存します。Python標準ライブラリだけを使います。`--self-test` は曖昧な置換、文字列内の波括弧、出力先の逸脱、存在しない追記先、テスト0件を検査します。`--check` のsource操作は01〜25章だけへ適用し、この入口の案内コードをプロジェクトへ書き込みません。

```python
"""Replay the printed chibigrad tutorial into a separate empty project."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

ROOT = Path.cwd()
RAW = re.compile(r'(?:br|r)(#*)"')
CHAR = re.compile(r"'(?:\\(?:[nrt0\\'\"]|x[0-9a-fA-F]{2}|u\{[0-9a-fA-F_]+\})|[^\\'\n])'")


def skip_rust_literal(source, index):
	if source.startswith("//", index):
		end = source.find("\n", index)
		return len(source) if end < 0 else end + 1
	if source.startswith("/*", index):
		depth, cursor = 1, index + 2
		while cursor < len(source) and depth:
			if source.startswith("/*", cursor):
				depth += 1
				cursor += 2
			elif source.startswith("*/", cursor):
				depth -= 1
				cursor += 2
			else:
				cursor += 1
		if depth:
			raise ValueError("unterminated Rust comment")
		return cursor
	raw = RAW.match(source, index) if source[index] in "rb" else None
	if raw and (index == 0 or not (source[index-1].isalnum() or source[index-1] == "_")):
		delimiter = '"' + raw[1]
		end = source.find(delimiter, raw.end())
		if end < 0:
			raise ValueError("unterminated Rust raw string")
		return end + len(delimiter)
	if source[index] == '"':
		cursor = index + 1
		while cursor < len(source):
			if source[cursor] == "\\":
				cursor += 2
			elif source[cursor] == '"':
				return cursor + 1
			else:
				cursor += 1
		raise ValueError("unterminated Rust string")
	if source[index] == "'":
		char = CHAR.match(source, index)
		if char:
			return char.end()
	return None


def rust_delimited_end(source, start, opening="{", closing="}"):
	if source[start] != opening:
		raise ValueError("expected Rust delimiter")
	depth, index = 1, start + 1
	while index < len(source):
		skipped = skip_rust_literal(source, index)
		if skipped is not None:
			index = skipped
			continue
		if source[index] == opening:
			depth += 1
		elif source[index] == closing:
			depth -= 1
			if depth == 0:
				return index + 1
		index += 1
	raise ValueError("unclosed Rust item")


def rust_item_span(source, item):
	boundary = r"(?!\w)" if item[-1:].isalnum() or item.endswith("_") else ""
	matches = list(re.finditer(r"(?m)^[\t ]*" + re.escape(item) + boundary, source))
	if len(matches) != 1:
		raise ValueError(f"item must match once, found {len(matches)}: {item}")
	cursor = matches[0].end()
	while cursor < len(source):
		skipped = skip_rust_literal(source, cursor)
		if skipped is not None:
			cursor = skipped
		elif source[cursor] == "{":
			end = rust_delimited_end(source, cursor)
			if source[end:end+1] == ";":
				end += 1
			return matches[0].start(), end + (source[end:end+1] == "\n")
		elif source[cursor] == ";":
			if re.match(r"(?:pub(?:\([^)]*\))?\s+)?(?:const|static|type|struct)\b", item):
				end = cursor + 1
				return matches[0].start(), end + (source[end:end+1] == "\n")
			break
		else:
			cursor += 1
	raise ValueError(f"item has no body: {item}")


def rust_arm_span(source, item, arm):
	begin, end = rust_item_span(source, item)
	matches = list(re.finditer(r"(?m)^[\t ]*" + re.escape(arm) + r"(?=[\s{])", source[begin:end]))
	if len(matches) != 1:
		raise ValueError(f"arm must match once, found {len(matches)}: {arm}")
	cursor = begin + matches[0].end()
	while cursor < end and not source.startswith("=>", cursor):
		skipped = skip_rust_literal(source, cursor)
		cursor = skipped if skipped is not None else cursor + 1
	if cursor == end:
		raise ValueError("match arm has no arrow")
	cursor += 2
	while source[cursor].isspace():
		cursor += 1
	if source[cursor] != "{":
		raise ValueError("match arm needs a block body")
	finish = rust_delimited_end(source, cursor)
	if source[finish:finish+1] == ",":
		finish += 1
	if source[finish:finish+1] == "\n":
		finish += 1
	return begin + matches[0].start(), finish


def rust_replacement_span(source, options):
	offset, scope = 0, source
	if "within" in options:
		offset, end = rust_item_span(source, options["within"])
		scope = source[offset:end]
	a, b = (rust_arm_span(scope, options["item"], options["arm"]) if "arm" in options
			else rust_item_span(scope, options["item"]))
	return offset + a, offset + b


def blocks(path):
	lines = path.read_text().splitlines(keepends=True)
	index = 0
	while index < len(lines):
		match = re.fullmatch(r"```([^\n]*)\n?", lines[index])
		if not match:
			index += 1
			continue
		start, info = index + 1, shlex.split(match[1])
		body = []
		index += 1
		while index < len(lines) and lines[index].strip() != "```":
			body.append(lines[index])
			index += 1
		if index == len(lines):
			raise ValueError(f"{path}:{start}: unclosed fence")
		index += 1
		pairs = [part.split("=", 1) for part in info[1:] if "=" in part]
		options = dict(pairs)
		if len(pairs) != len(options):
			raise ValueError(f"{path}:{start}: duplicate attribute")
		yield start, info[0] if info else "", options, "".join(body), "check" in info[1:]


def destination(project, name):
	relative = Path(name)
	target = (project / relative).resolve()
	if relative.is_absolute() or target == project.resolve() or not target.is_relative_to(project.resolve()):
		raise ValueError(f"file escapes replay directory: {name}")
	return target


def apply_source(project, language, options, body):
	target = destination(project, options["file"])
	mode = options.get("mode")
	if set(options) - {"file", "mode", "item", "arm", "within"}:
		raise ValueError("unknown source attribute")
	if mode not in {"write", "append", "replace"}:
		raise ValueError("mode must be write, append, or replace")
	if mode != "write" and not target.exists():
		raise ValueError(f"missing file: {options['file']}")
	target.parent.mkdir(parents=True, exist_ok=True)
	if mode == "replace":
		if language != "rust" or not options.get("item"):
			raise ValueError("replacement requires rust and item")
		if not body.lstrip().startswith(options.get("arm", options["item"])):
			raise ValueError("replacement changes the selected item")
		previous = target.read_text()
		a, b = rust_replacement_span(previous, options)
		target.write_text(previous[:a] + body.rstrip() + "\n" + previous[b:])
	else:
		if set(options) & {"item", "arm", "within"}:
			raise ValueError("selectors apply only to replacement")
		with target.open("w" if mode == "write" else "a") as output:
			output.write(body)


def require_executed_tests(command, stdout):
	if re.search(r"\bcargo\s+test\b", command) and "--no-run" not in command:
		count = sum(int(x) for x in re.findall(r"test result: ok\. (\d+) passed;", stdout))
		if count == 0:
			raise ValueError("Cargo succeeded but no test passed; check the filter")


def replay(chapters, project, run_checks=False, capabilities=(), timeout=300):
	project.mkdir(parents=True, exist_ok=True)
	if any(project.iterdir()):
		raise ValueError(f"output must be empty: {project}")
	results = []
	for chapter in chapters:
		written, checks = 0, []
		for line, language, options, body, check in blocks(chapter):
			context = f"{chapter.name}:{line}"
			if "file" in options:
				apply_source(project, language, options, body)
				written += 1
			if check:
				if language != "sh" or set(options) - {"requires"}:
					raise ValueError(f"{context}: invalid check")
				needed = set(options.get("requires", "").split(",")) - {""}
				missing = needed - set(capabilities)
				result = dict(line=line, command=body, requires=sorted(needed), status="NOT_RUN", reason="checks disabled")
				if run_checks and not missing:
					print(f"CHECK {context}", flush=True)
					done = subprocess.run(["/bin/sh", "-eu", "-c", body], cwd=project, capture_output=True, text=True, timeout=timeout)
					result.update(status="PASS" if done.returncode == 0 else "FAIL", reason="", exit_code=done.returncode, stdout=done.stdout, stderr=done.stderr)
					if done.returncode:
						print(done.stdout + done.stderr, flush=True)
						raise subprocess.CalledProcessError(done.returncode, body)
					require_executed_tests(body, done.stdout)
				elif run_checks:
					result["reason"] = "missing capability: " + ", ".join(sorted(missing))
					print(f"NOT_RUN {context}: {result['reason']}", flush=True)
				checks.append(result)
		if not written or not checks:
			raise ValueError(f"{chapter.name}: source and checks required")
		results.append(dict(chapter=chapter.name, file_blocks=written, checks=checks))
		print(f"{chapter.name}: {written} source operations, {len(checks)} checks", flush=True)
	return results


def lint(chapters):
	inventory = json.loads((ROOT / "docs/requirements.json").read_text())
	selected = {str(path.relative_to(ROOT)) for path in chapters}
	expected = {row["id"]: row for row in inventory if row["file"] in selected}
	found, errors = {}, []
	for chapter in chapters:
		source = chapter.read_text()
		for match in re.finditer(r"<!-- todo:(\d{2}-\d{3}) -->", source):
			identity = match[1]
			if identity in found:
				errors.append(f"duplicate {identity}")
			found[identity] = chapter
			end = re.search(r"^#{1,6} ", source[match.end():], re.M)
			section = source[match.end():match.end()+end.start()] if end else source[match.end():]
			headings = list(re.finditer(r"^#{2,6} (.+)$", source[:match.start()], re.M))
			title = headings[-1][1] if headings else ""
			if identity in expected and expected[identity]["todo"].rstrip("。") not in title:
				errors.append(f"{identity}: original heading missing")
			pattern = r"^```(?:rust|toml)\b[^\n]*\bfile=" if not identity.startswith("00-") else r"^```(?:rust|toml|sh)\b"
			if not re.search(pattern, section, re.M):
				errors.append(f"{identity}: no source operation")
		for line, language, options, body, check in blocks(chapter):
			if "file" in options:
				destination(Path("/tmp/chibi-lint-project"), options["file"])
				if re.search(r"\b(?:todo!|unimplemented!)\s*\(", body):
					errors.append(f"{chapter.name}:{line}: unfinished source")
	errors += [f"missing {identity}" for identity in expected.keys()-found.keys()]
	errors += [f"unknown {identity}" for identity in found.keys()-expected.keys()]
	for error in errors:
		print(error)
	print(f"structural coverage: {len(found.keys() & expected.keys())}/{len(expected)} requirements")
	return not errors


def self_test():
	sample = '#[test]\nfn f(){let r=r#"{ }"#;let c=\'}\';/* { /* } */ } */}\nfn g(){}\n'
	a, b = rust_item_span(sample, "fn f(")
	assert sample[:a] == "#[test]\n" and sample[a:b].endswith("*/}\n")
	try:
		rust_item_span("fn f(){}\nfn f(){}\n", "fn f(")
	except ValueError:
		pass
	else:
		raise AssertionError("ambiguous replacement accepted")
	arms = 'fn f(){match x{\n Op::Pad{x}=>{println!("}");x}\n Op::Other{x}=>{x}\n}}\n'
	a, b = rust_arm_span(arms, "fn f(", "Op::Pad")
	assert arms[a:b] == ' Op::Pad{x}=>{println!("}");x}\n'
	try:
		require_executed_tests("cargo test typo", "test result: ok. 0 passed; 0 failed;")
	except ValueError:
		pass
	else:
		raise AssertionError("zero-test success accepted")
	with tempfile.TemporaryDirectory() as folder:
		base = Path(folder)
		chapter = base / "01.md"
		chapter.write_text('```text file=x mode=write\na\n```\n```text file=x mode=append\nb\n```\n```sh check\ntest -s x\n```\n')
		replay([chapter], base / "output", True)
		assert (base / "output/x").read_text() == "a\nb\n"
		for options in [dict(file="../escape", mode="write"), dict(file="missing", mode="append")]:
			try:
				apply_source(base, "text", options, "x")
			except ValueError:
				pass
			else:
				raise AssertionError("invalid source operation accepted")
	print("self-test passed")


def main():
	global ROOT
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--docs", type=Path, default=Path.cwd()/"docs")
	parser.add_argument("--output", type=Path)
	parser.add_argument("--through", type=int, default=25)
	parser.add_argument("--check", action="store_true")
	parser.add_argument("--lint", action="store_true")
	parser.add_argument("--self-test", action="store_true")
	parser.add_argument("--capability", action="append", default=[])
	parser.add_argument("--timeout", type=int, default=300)
	parser.add_argument("--report", type=Path)
	args = parser.parse_args()
	if args.self_test:
		self_test()
		return
	docs = args.docs.resolve()
	ROOT = docs.parent
	chapters = sorted(p for p in docs.glob("[0-9][0-9]-*.md") if (0 if args.lint else 1) <= int(p.name[:2]) <= args.through)
	if not chapters:
		parser.error("no tutorial chapters found")
	if args.lint:
		raise SystemExit(0 if lint(chapters) else 1)
	if args.output is None or args.timeout <= 0:
		parser.error("--output and a positive timeout are required")
	output = args.output.resolve()
	if output == ROOT or output.is_relative_to(ROOT):
		parser.error("output must be outside the tutorial checkout")
	if args.report:
		args.report.write_text(json.dumps(dict(status="RUNNING", output=str(output)))+"\n")
	try:
		results = replay(chapters, output, args.check, args.capability, args.timeout)
	except Exception as error:
		if args.report:
			args.report.write_text(json.dumps(dict(status="FAIL", error=str(error)), indent=2)+"\n")
		raise
	skipped = sum(check["status"] == "NOT_RUN" for chapter in results for check in chapter["checks"])
	status = "EXTRACTED" if not args.check else ("CPU_PASS_WITH_NOT_RUN" if skipped else "PASS")
	if args.report:
		args.report.write_text(json.dumps(dict(status=status, chapters=results), indent=2)+"\n")
	print(f"{status}: {len(results)} chapters, {skipped} checks not run")


if __name__ == "__main__":
	main()
```
