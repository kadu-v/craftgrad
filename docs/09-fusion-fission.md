# 09. 融合と分割で閉じた kernel graph を作り、Plan を再実行する

Graph::compile が consumer 数、STAGE、reduction を見て materialize 位置を選ぶ。定数は出力以外では inline する。Plan は閉じた kernel の順序と binding 署名を持ち、入力値をキャッシュしない。実体化する kernel の出力だけを確保し、融合で消えた broadcast product をホストに確保しない。公開 dump metadata の変更は seal の検証で実行前に拒否する。

`file=... mode=write` はそのファイルの全置換、`mode=append` は末尾への追記である。すべて前章でできたプロジェクトの root で行う。`sh check` は実際の検証コマンドであり、ソースファイルには保存しない。

## 1. 元グラフから consumer 数を集める。
<!-- todo:09-001 -->

この章の閉じたkernel表現を追加します。`KernelSignature` と `KernelCall` を比較し、producerの完了stateを要求します。body検査はこのcompilerの生成する構造化Cの部分言語に限定します。任意の外部Cのメモリ安全性を証明する関数ではありません。添字の範囲はGraphと実行時shapeの検証も使います。

```rust file=src/lib.rs mode=append
pub mod closure;
```

```rust file=src/closure.rs mode=write
use crate::{Graph, Op, Result, native::KernelBody};
use std::collections::HashSet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelSignature {
	pub buffers: Vec<usize>,
	pub scalar_words: usize,
	pub output: usize,
}
#[derive(Clone, Debug)]
pub struct KernelCall {
	pub target: usize,
	pub buffers: Vec<usize>,
	pub scalar_words: usize,
	pub after: Vec<usize>,
	pub output_state: usize,
}

pub fn signatures(graph: &Graph, kernels: &[KernelBody]) -> Result<Vec<KernelSignature>> {
	let scalar_words = graph.nodes.iter().try_fold(0usize, |n, node| {
		n.checked_add(node.shape.len())
			.ok_or("scalar signature overflow")
	})?;
	kernels
		.iter()
		.map(|kernel| {
			if kernel.output >= graph.nodes.len() {
				return Err("kernel output outside graph".into());
			}
			if matches!(graph.nodes[kernel.output].op, Op::Input { .. }) {
				return Err("kernel cannot STORE into a read-only input".into());
			}
			let mut buffers = kernel.dependencies.clone();
			buffers.push(kernel.output);
			let unique: HashSet<_> = buffers.iter().copied().collect();
			if unique.len() != buffers.len() || buffers.iter().any(|&id| id >= graph.nodes.len()) {
				return Err("kernel signature has duplicate or unknown storage".into());
			}
			Ok(KernelSignature {
				buffers,
				scalar_words,
				output: kernel.output,
			})
		})
		.collect()
}

pub fn calls(graph: &Graph, kernels: &[KernelBody]) -> Result<Vec<KernelCall>> {
	let signatures = signatures(graph, kernels)?;
	let mut produced = std::collections::HashMap::new();
	let mut result = Vec::new();
	for (index, signature) in signatures.iter().enumerate() {
		let mut after = Vec::new();
		for &id in &kernels[index].dependencies {
			if !matches!(graph.nodes[id].op, Op::Input { .. }) {
				after.push(
					*produced
						.get(&id)
						.ok_or("producer must complete before consumer")?,
				);
			}
		}
		after.sort_unstable();
		after.dedup();
		let state = index + 1;
		if produced.insert(signature.output, state).is_some() {
			return Err("multiple producers for one output state".into());
		}
		result.push(KernelCall {
			target: index,
			buffers: signature.buffers.clone(),
			scalar_words: signature.scalar_words,
			after,
			output_state: state,
		});
	}
	verify_calls(graph, kernels, &result)?;
	Ok(result)
}

fn tokens(body: &str) -> Result<Vec<String>> {
	let bytes = body.as_bytes();
	let mut result = Vec::new();
	let mut i = 0;
	while i < bytes.len() {
		if bytes[i].is_ascii_whitespace() {
			i += 1;
			continue;
		}
		if bytes[i] == b'"' || bytes[i] == b'\'' || !bytes[i].is_ascii() {
			return Err("unsupported token in generated kernel body".into());
		}
		if bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'.' {
			let start = i;
			i += 1;
			while i < bytes.len()
				&& (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'.')
			{
				i += 1;
			}
			result.push(body[start..i].into());
		} else {
			result.push((bytes[i] as char).to_string());
			i += 1;
		}
	}
	Ok(result)
}

fn mutation(code: &[String], position: usize, end: usize) -> (bool, bool) {
	let at = |i: usize| code.get(i).map(String::as_str);
	let assignment = at(end) == Some("=") && at(end + 1) != Some("=");
	let compound = matches!(at(end), Some("+" | "-" | "*" | "/" | "%" | "&" | "|" | "^"))
		&& at(end + 1) == Some("=");
	let shift = matches!(
		(at(end), at(end + 1), at(end + 2)),
		(Some("<"), Some("<"), Some("=")) | (Some(">"), Some(">"), Some("="))
	);
	let suffix = matches!(
		(at(end), at(end + 1)),
		(Some("+"), Some("+")) | (Some("-"), Some("-"))
	);
	let prefix = position >= 2
		&& matches!(
			(at(position - 2), at(position - 1)),
			(Some("+"), Some("+")) | (Some("-"), Some("-"))
		);
	(
		assignment || compound || shift || suffix || prefix,
		assignment,
	)
}

pub fn verify_body(body: &str, signature: &KernelSignature) -> Result<()> {
	let code = tokens(body)?;
	let mut scopes: Vec<(HashSet<String>, bool)> = vec![(HashSet::new(), false)];
	let mut pending_for = false;
	let mut stored = false;
	let types = [
		"float", "double", "size_t", "int32_t", "uint32_t", "uint64_t",
	];
	let fixed = [
		"for",
		"if",
		"else",
		"return",
		"const",
		"s",
		"NAN",
		"INFINITY",
		"chibi_bits",
		"chibi_max",
		"expf",
		"logf",
		"sinf",
		"cosf",
		"sqrtf",
		"exp2f",
		"log2f",
		"floorf",
		"isfinite",
	];
	for (position, word) in code.iter().enumerate() {
		if word == "for" {
			if pending_for {
				return Err("nested incomplete RANGE header".into());
			}
			scopes.push((HashSet::new(), false));
			pending_for = true;
		} else if word == "{" {
			scopes.push((HashSet::new(), pending_for));
			pending_for = false;
		} else if word == "}" {
			let (_, closes_for) = scopes.pop().ok_or("unmatched kernel scope")?;
			if closes_for {
				scopes.pop().ok_or("missing RANGE scope")?;
			}
			if scopes.is_empty() {
				return Err("kernel scope escaped body".into());
			}
		}
		if let Some(digits) = word
			.strip_prefix('b')
			.filter(|s| !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit()))
		{
			let id: usize = digits.parse().map_err(|_| "storage ID overflow")?;
			if !signature.buffers.contains(&id) {
				return Err(format!("unbound storage PARAM b{id}"));
			}
			if code.get(position + 1).map(String::as_str) != Some("[") {
				return Err("storage must be indexed".into());
			}
			let mut nesting = 1;
			let mut end = position + 2;
			while end < code.len() && nesting != 0 {
				match code[end].as_str() {
					"[" => nesting += 1,
					"]" => nesting -= 1,
					_ => {}
				}
				end += 1;
			}
			if nesting != 0 {
				return Err("unclosed storage index".into());
			}
			let (write, assignment) = mutation(&code, position, end);
			if write {
				if id != signature.output {
					return Err("STORE targets a non-output PARAM".into());
				}
				if !assignment {
					return Err("output update reads uninitialized output".into());
				}
				stored = true;
			} else if id == signature.output {
				return Err("output LOAD is not allowed before its STORE".into());
			}
			continue;
		}
		if word == "s" {
			if code.get(position + 1).map(String::as_str) != Some("[") {
				return Err("scalar PARAM must use a literal slot".into());
			}
			let slot: usize = code
				.get(position + 2)
				.ok_or("missing scalar slot")?
				.parse()
				.map_err(|_| "nonliteral scalar slot")?;
			if slot >= signature.scalar_words
				|| code.get(position + 3).map(String::as_str) != Some("]")
			{
				return Err("unbound scalar PARAM".into());
			}
			if mutation(&code, position, position + 4).0 {
				return Err("scalar PARAM is read-only".into());
			}
			continue;
		}
		let identifier = word.as_bytes()[0].is_ascii_alphabetic() || word.starts_with('_');
		if !identifier || types.contains(&word.as_str()) || fixed.contains(&word.as_str()) {
			continue;
		}
		let declaration = position > 0 && types.contains(&code[position - 1].as_str());
		if declaration {
			if !scopes.last_mut().unwrap().0.insert(word.clone()) {
				return Err("duplicate local declaration".into());
			}
		} else if !scopes.iter().rev().any(|scope| scope.0.contains(word)) {
			return Err(format!("free or closed RANGE/local {word}"));
		}
	}
	if pending_for || scopes.len() != 1 {
		return Err("unclosed RANGE or kernel scope".into());
	}
	if !stored {
		return Err("kernel is missing its output STORE".into());
	}
	Ok(())
}

pub fn verify_calls(graph: &Graph, kernels: &[KernelBody], calls: &[KernelCall]) -> Result<()> {
	let signatures = signatures(graph, kernels)?;
	if calls.len() != kernels.len() {
		return Err("kernel CALL count mismatch".into());
	}
	let mut initialized: HashSet<usize> = graph
		.nodes
		.iter()
		.enumerate()
		.filter_map(|(id, node)| matches!(node.op, Op::Input { .. }).then_some(id))
		.collect();
	let mut states = std::collections::HashMap::new();
	for (index, (signature, call)) in signatures.iter().zip(calls).enumerate() {
		if call.target != index
			|| call.buffers != signature.buffers
			|| call.scalar_words != signature.scalar_words
		{
			return Err("kernel signature and CALL arguments differ".into());
		}
		if call.output_state != index + 1 {
			return Err("kernel output state mismatch".into());
		}
		if initialized.contains(&signature.output) {
			return Err("duplicate producer STORE".into());
		}
		let mut required = Vec::new();
		for &id in &kernels[index].dependencies {
			if !initialized.contains(&id) {
				return Err("LOAD before producer STORE".into());
			}
			if let Some(state) = states.get(&id) {
				required.push(*state);
			}
		}
		required.sort_unstable();
		required.dedup();
		if call.after != required {
			return Err("CALL does not depend on the producer state".into());
		}
		verify_body(&kernels[index].body, signature)?;
		initialized.insert(signature.output);
		states.insert(signature.output, call.output_state);
	}
	Ok(())
}
```


`src/graph.rs` の同名関数全体を置き換えます。08章の非融合の選択を、consumer 数、STAGE、縮約に応じた選択へ変更します。

```rust file=src/graph.rs mode=replace item="pub fn kernel_bodies("
pub fn kernel_bodies(graph: &Graph, outputs: &[NodeId], fuse: bool) -> Result<Vec<KernelBody>> {
	graph.verify(outputs)?;
	let order = graph.topology(outputs)?;
	let mut users = vec![0; graph.nodes.len()];
	for &id in &order {
		for input in graph.nodes[id].op.sources() {
			users[input] += 1
		}
	}
	let material: HashSet<_> = order
		.iter()
		.copied()
		.filter(|&id| {
			outputs.contains(&id)
				|| (!matches!(graph.nodes[id].op, Op::Const { .. }) && (!fuse || users[id] > 1))
				|| matches!(
					graph.nodes[id].op,
					Op::Input { .. }
						| Op::Stage { .. } | Op::Reduce { .. }
						| Op::Gather { .. }
						| Op::ScatterAdd { .. }
				)
		})
		.collect();
	let mut offsets = vec![];
	let mut offset = 0;
	for node in &graph.nodes {
		offsets.push(offset);
		offset += node.shape.len()
	}
	let mut emitter = Emitter {
		g: graph,
		offsets,
		material,
		current: 0,
		serial: 0,
	};
	let mut kernels = vec![];
	for id in order {
		if !emitter.material.contains(&id) || matches!(graph.nodes[id].op, Op::Input { .. }) {
			continue;
		}
		emitter.current = id;
		let mut body = format!("for (size_t i=0; i<{}; ++i) {{\n", emitter.count(id));
		let value = emitter.expr(id, "i", &mut body)?;
		body.push_str(&format!("b{id}[i]={value};\n}}\n"));
		let dependencies = graph
			.topology(&[id])?
			.into_iter()
			.filter(|&x| x != id && emitter.material.contains(&x))
			.collect();
		kernels.push(KernelBody {
			output: id,
			body,
			dependencies,
		});
	}
	crate::closure::calls(graph, &kernels)?;
	Ok(kernels)
}
```


```rust file=tests/fusion_plan.rs mode=write
use chinigrad::{Binary, Buffer, Graph, Reduce, Result, Unary};
fn buffer(s: &[usize], d: &[f32]) -> Buffer {
	Buffer::new(s.to_vec(), d.to_vec()).unwrap()
}
fn compare(g: &Graph, outs: &[usize], inputs: &[Buffer]) -> Result<Vec<Buffer>> {
	let actual = g.compile(outs)?.run(inputs)?;
	let reference = g.interpret(outs, inputs)?;
	for (a, b) in actual.iter().zip(reference) {
		assert_eq!(a.shape, b.shape);
		for (&x, &y) in a.data.iter().zip(&b.data) {
			assert!(
				x.is_nan() && y.is_nan() || (x - y).abs() <= 1e-5 + 1e-5 * y.abs(),
				"{x} != {y}"
			);
		}
	}
	Ok(actual)
}

#[test]
fn arithmetic_fusion() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let one = g.constant(1.0, vec![])?;
	let two = g.constant(2.0, vec![])?;
	let a = g.binary(Binary::Add, x, one)?;
	let y = g.binary(Binary::Mul, a, two)?;
	let p = g.compile(&[y])?;
	assert_eq!(p.kernel_count, 1);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [4., 6., 8.]);
	let staged = g.stage(a)?;
	let z = g.binary(Binary::Mul, staged, two)?;
	assert_eq!(g.compile(&[z])?.kernel_count, 2);
	let three = g.constant(3., vec![])?;
	let z = g.binary(Binary::Add, a, three)?;
	assert_eq!(g.compile(&[y, z])?.kernel_count, 3);
	let result = compare(&g, &[y, z], &[buffer(&[3], &[1., 2., 3.])])?;
	assert_eq!(result[1].data, [5., 6., 7.]);
	Ok(())
}
#[test]
fn reductions_and_rebinding() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![4])?;
	let w = g.input(1, vec![4])?;
	let one = g.constant(1., vec![])?;
	let p = g.binary(Binary::Mul, x, w)?;
	let p = g.binary(Binary::Add, p, one)?;
	let y = g.reduce(p, Reduce::Sum, vec![0], false)?;
	let p = g.compile(&[y])?;
	assert_eq!(p.kernel_count, 1);
	assert_eq!(
		p.run(&[
			buffer(&[4], &[1., 2., 3., 4.]),
			buffer(&[4], &[2., 3., 4., 5.])
		])?[0]
			.data,
		[44.]
	);
	assert_eq!(
		p.run(&[
			buffer(&[4], &[1., 2., 3., 4.]),
			buffer(&[4], &[1., 1., 1., 1.])
		])?[0]
			.data,
		[14.]
	);
	Ok(())
}
#[test]
fn dynamic_metadata() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let two = g.constant(2., vec![])?;
	let y = g.binary(Binary::Mul, x, two)?;
	let p = g.compile(&[y])?;
	let shapes = vec![vec![2], vec![], vec![2]];
	assert_eq!(
		p.run_with_shapes(&[buffer(&[4], &[3., 4., 9., 9.])], &shapes)?[0].data,
		[6., 8.]
	);
	assert!(
		chinigrad::Plan::compile_specialized(&g, &[y])?
			.run_with_shapes(&[buffer(&[4], &[3., 4., 9., 9.])], &shapes)
			.is_err()
	);
	Ok(())
}
#[test]
fn compiled_plan_metadata_is_sealed() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![2])?;
	let y = g.unary(Unary::Neg, x)?;
	let mut plan = g.compile(&[y])?;
	plan.graph.nodes[x].shape = vec![1];
	assert!(plan.run(&[buffer(&[1], &[1.])]).is_err());
	Ok(())
}
```

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test fusion_plan
```

## 2. 単一消費先の ADD と MUL の境界を materialize しない。
<!-- todo:09-002 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

`src/graph.rs` の `pub fn kernel_bodies(`の定義を次の全体で置き換えます。

```rust file="src/graph.rs" mode="replace" item="pub fn kernel_bodies("
pub fn kernel_bodies(graph: &Graph, outputs: &[NodeId], fuse: bool) -> Result<Vec<KernelBody>> {
	graph.verify(outputs)?;
	let order = graph.topology(outputs)?;
	let mut users = vec![0; graph.nodes.len()];
	for &id in &order {
		for input in graph.nodes[id].op.sources() {
			users[input] += 1
		}
	}
	let material: HashSet<_> = order
		.iter()
		.copied()
		.filter(|&id| {
			outputs.contains(&id)
				|| (!matches!(graph.nodes[id].op, Op::Const { .. }) && (!fuse || users[id] > 1))
				|| matches!(
					graph.nodes[id].op,
					Op::Input { .. }
						| Op::Stage { .. } | Op::Reduce { .. }
						| Op::Gather { .. }
						| Op::ScatterAdd { .. }
				)
		})
		.collect();
	let mut offsets = vec![];
	let mut offset = 0;
	for node in &graph.nodes {
		offsets.push(offset);
		offset += node.shape.len()
	}
	let mut emitter = Emitter {
		g: graph,
		offsets,
		material,
		current: 0,
		serial: 0,
	};
	let mut kernels = vec![];
	for id in order {
		if !emitter.material.contains(&id) || matches!(graph.nodes[id].op, Op::Input { .. }) {
			continue;
		}
		emitter.current = id;
		let mut body = format!("for (size_t i=0; i<{}; ++i) {{\n", emitter.count(id));
		let value = emitter.expr(id, "i", &mut body)?;
		body.push_str(&format!("b{id}[i]={value};\n}}\n"));
		let dependencies = graph
			.topology(&[id])?
			.into_iter()
			.filter(|&x| x != id && emitter.material.contains(&x))
			.collect();
		kernels.push(KernelBody {
			output: id,
			body,
			dependencies,
		});
	}
	crate::closure::calls(graph, &kernels)?;
	Ok(kernels)
}
```

```sh check
cargo test --quiet --test fusion_plan
```

## 3. lower で同じ出力座標を producer へ伝える。
<!-- todo:09-003 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

`src/graph.rs` の `fn expr(` の `Op::Binary` 分岐を次の全体で置き換えます。

```rust file="src/graph.rs" mode="replace" item="fn expr(" arm="Op::Binary"
			Op::Binary { op, a, b } => {
				let ai = self.broadcast_index(a, id, index);
				let bi = self.broadcast_index(b, id, index);
				let av = self.expr(a, &ai, code)?;
				let bv = self.expr(b, &bi, code)?;
				match op {
					Binary::Max => format!("chibi_max({av},{bv})"),
					_ => format!(
						"(({av}) {} ({bv}))",
						match op {
							Binary::Add => "+",
							Binary::Sub => "-",
							Binary::Mul => "*",
							Binary::Div => "/",
							Binary::Lt => "<",
							Binary::Eq => "==",
							_ => unreachable!(),
						}
					),
				}
			}
```

```sh check
cargo test --quiet --test fusion_plan
```

## 4. `(x+1)*2` を単一 kernel で `[4,6,8]` にする。
<!-- todo:09-004 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

`tests/fusion_plan.rs` の `fn arithmetic_fusion(`の定義を次の全体で置き換えます。

```rust file="tests/fusion_plan.rs" mode="replace" item="fn arithmetic_fusion("
fn arithmetic_fusion() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let one = g.constant(1.0, vec![])?;
	let two = g.constant(2.0, vec![])?;
	let a = g.binary(Binary::Add, x, one)?;
	let y = g.binary(Binary::Mul, a, two)?;
	let p = g.compile(&[y])?;
	assert_eq!(p.kernel_count, 1);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [4., 6., 8.]);
	let staged = g.stage(a)?;
	let z = g.binary(Binary::Mul, staged, two)?;
	assert_eq!(g.compile(&[z])?.kernel_count, 2);
	let three = g.constant(3., vec![])?;
	let z = g.binary(Binary::Add, a, three)?;
	assert_eq!(g.compile(&[y, z])?.kernel_count, 3);
	let result = compare(&g, &[y, z], &[buffer(&[3], &[1., 2., 3.])])?;
	assert_eq!(result[1].data, [5., 6., 7.]);
	Ok(())
}
```

```sh check
cargo test --quiet --test fusion_plan
```

## 5. 前章の非融合経路とも出力を比較する。
<!-- todo:09-005 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

`tests/fusion_plan.rs` の `fn arithmetic_fusion(`の定義を次の全体で置き換えます。

```rust file="tests/fusion_plan.rs" mode="replace" item="fn arithmetic_fusion("
fn arithmetic_fusion() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let one = g.constant(1.0, vec![])?;
	let two = g.constant(2.0, vec![])?;
	let a = g.binary(Binary::Add, x, one)?;
	let y = g.binary(Binary::Mul, a, two)?;
	let p = g.compile(&[y])?;
	assert_eq!(p.kernel_count, 1);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [4., 6., 8.]);
	let staged = g.stage(a)?;
	let z = g.binary(Binary::Mul, staged, two)?;
	assert_eq!(g.compile(&[z])?.kernel_count, 2);
	let three = g.constant(3., vec![])?;
	let z = g.binary(Binary::Add, a, three)?;
	assert_eq!(g.compile(&[y, z])?.kernel_count, 3);
	let result = compare(&g, &[y, z], &[buffer(&[3], &[1., 2., 3.])])?;
	assert_eq!(result[1].data, [5., 6., 7.]);
	Ok(())
}
```

```sh check
cargo test --quiet --test fusion_plan
```

## 6. `sum(x*w+1)` の reduction 座標を要素演算へ伝える。
<!-- todo:09-006 -->

`tests/fusion_plan.rs` の `reductions_and_rebinding` で、この経路の値または拒否条件を確認する。

`src/graph.rs` の `fn expr(` の `Op::Reduce` 分岐を次の全体で置き換えます。

```rust file="src/graph.rs" mode="replace" item="fn expr(" arm="Op::Reduce"
			Op::Reduce {
				op,
				x,
				axes,
				keepdim,
			} => {
				let target = self.name();
				let k = self.name();
				code.push_str(&format!(
					"float {target} = {};\nfor (size_t {k}=0; {k}<{}; ++{k}) {{\n",
					if op == Reduce::Sum {
						"0.0f"
					} else {
						"-INFINITY"
					},
					self.product(x, axes.iter().copied())
				));
				let mut out_axis = 0;
				let c = (0..self.g.shape(x).len())
					.map(|a| {
						if let Some(ra) = axes.iter().position(|&v| v == a) {
							format!(
								"(({k} / {}) % {})",
								self.product(x, axes[ra + 1..].iter().copied()),
								self.dim(x, a)
							)
						} else {
							let oa = if keepdim {
								a
							} else {
								let v = out_axis;
								out_axis += 1;
								v
							};
							self.coord(id, index, oa)
						}
					})
					.collect::<Vec<_>>();
				let xi = self.flatten(x, &c);
				let value = self.expr(x, &xi, code)?;
				code.push_str(&format!(
					"{target} = {};\n}}\n",
					if op == Reduce::Sum {
						format!("{target} + ({value})")
					} else {
						format!("chibi_max({target},{value})")
					}
				));
				return Ok(target);
			}
```

```sh check
cargo test --quiet --test fusion_plan reductions_and_rebinding
```

## 7. reduction の外側出力と内側 k のスコープを区別する。
<!-- todo:09-007 -->

`tests/fusion_plan.rs` の `reductions_and_rebinding` で、この経路の値または拒否条件を確認する。

`src/graph.rs` の `fn expr(` の `Op::Reduce` 分岐を次の全体で置き換えます。

```rust file="src/graph.rs" mode="replace" item="fn expr(" arm="Op::Reduce"
			Op::Reduce {
				op,
				x,
				axes,
				keepdim,
			} => {
				let target = self.name();
				let k = self.name();
				code.push_str(&format!(
					"float {target} = {};\nfor (size_t {k}=0; {k}<{}; ++{k}) {{\n",
					if op == Reduce::Sum {
						"0.0f"
					} else {
						"-INFINITY"
					},
					self.product(x, axes.iter().copied())
				));
				let mut out_axis = 0;
				let c = (0..self.g.shape(x).len())
					.map(|a| {
						if let Some(ra) = axes.iter().position(|&v| v == a) {
							format!(
								"(({k} / {}) % {})",
								self.product(x, axes[ra + 1..].iter().copied()),
								self.dim(x, a)
							)
						} else {
							let oa = if keepdim {
								a
							} else {
								let v = out_axis;
								out_axis += 1;
								v
							};
							self.coord(id, index, oa)
						}
					})
					.collect::<Vec<_>>();
				let xi = self.flatten(x, &c);
				let value = self.expr(x, &xi, code)?;
				code.push_str(&format!(
					"{target} = {};\n}}\n",
					if op == Reduce::Sum {
						format!("{target} + ({value})")
					} else {
						format!("chibi_max({target},{value})")
					}
				));
				return Ok(target);
			}
```

```sh check
cargo test --quiet --test fusion_plan reductions_and_rebinding
```

## 8. LOAD、MUL、ADD、accumulator 更新を同じ body に作る。
<!-- todo:09-008 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

`src/graph.rs` の `fn expr(` の `Op::Reduce` 分岐を次の全体で置き換えます。

```rust file="src/graph.rs" mode="replace" item="fn expr(" arm="Op::Reduce"
			Op::Reduce {
				op,
				x,
				axes,
				keepdim,
			} => {
				let target = self.name();
				let k = self.name();
				code.push_str(&format!(
					"float {target} = {};\nfor (size_t {k}=0; {k}<{}; ++{k}) {{\n",
					if op == Reduce::Sum {
						"0.0f"
					} else {
						"-INFINITY"
					},
					self.product(x, axes.iter().copied())
				));
				let mut out_axis = 0;
				let c = (0..self.g.shape(x).len())
					.map(|a| {
						if let Some(ra) = axes.iter().position(|&v| v == a) {
							format!(
								"(({k} / {}) % {})",
								self.product(x, axes[ra + 1..].iter().copied()),
								self.dim(x, a)
							)
						} else {
							let oa = if keepdim {
								a
							} else {
								let v = out_axis;
								out_axis += 1;
								v
							};
							self.coord(id, index, oa)
						}
					})
					.collect::<Vec<_>>();
				let xi = self.flatten(x, &c);
				let value = self.expr(x, &xi, code)?;
				code.push_str(&format!(
					"{target} = {};\n}}\n",
					if op == Reduce::Sum {
						format!("{target} + ({value})")
					} else {
						format!("chibi_max({target},{value})")
					}
				));
				return Ok(target);
			}
```

```sh check
cargo test --quiet --test fusion_plan
```

## 9. `x=[1,2,3,4]`, `w=[2,3,4,5]` の結果を `44` にする。
<!-- todo:09-009 -->

`tests/fusion_plan.rs` の `reductions_and_rebinding` で、この経路の値または拒否条件を確認する。

`tests/fusion_plan.rs` の `fn reductions_and_rebinding(`の定義を次の全体で置き換えます。

```rust file="tests/fusion_plan.rs" mode="replace" item="fn reductions_and_rebinding("
fn reductions_and_rebinding() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![4])?;
	let w = g.input(1, vec![4])?;
	let one = g.constant(1., vec![])?;
	let p = g.binary(Binary::Mul, x, w)?;
	let p = g.binary(Binary::Add, p, one)?;
	let y = g.reduce(p, Reduce::Sum, vec![0], false)?;
	let p = g.compile(&[y])?;
	assert_eq!(p.kernel_count, 1);
	assert_eq!(
		p.run(&[
			buffer(&[4], &[1., 2., 3., 4.]),
			buffer(&[4], &[2., 3., 4., 5.])
		])?[0]
			.data,
		[44.]
	);
	assert_eq!(
		p.run(&[
			buffer(&[4], &[1., 2., 3., 4.]),
			buffer(&[4], &[1., 1., 1., 1.])
		])?[0]
			.data,
		[14.]
	);
	Ok(())
}
```

```sh check
cargo test --quiet --test fusion_plan reductions_and_rebinding
```

## 10. 入力側の一時配列がなく、一つの reduction kernel になることを確認する。
<!-- todo:09-010 -->

`tests/fusion_plan.rs` の `reductions_and_rebinding` で、この経路の値または拒否条件を確認する。

`tests/fusion_plan.rs` の `fn reductions_and_rebinding(`の定義を次の全体で置き換えます。

```rust file="tests/fusion_plan.rs" mode="replace" item="fn reductions_and_rebinding("
fn reductions_and_rebinding() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![4])?;
	let w = g.input(1, vec![4])?;
	let one = g.constant(1., vec![])?;
	let p = g.binary(Binary::Mul, x, w)?;
	let p = g.binary(Binary::Add, p, one)?;
	let y = g.reduce(p, Reduce::Sum, vec![0], false)?;
	let p = g.compile(&[y])?;
	assert_eq!(p.kernel_count, 1);
	assert_eq!(
		p.run(&[
			buffer(&[4], &[1., 2., 3., 4.]),
			buffer(&[4], &[2., 3., 4., 5.])
		])?[0]
			.data,
		[44.]
	);
	assert_eq!(
		p.run(&[
			buffer(&[4], &[1., 2., 3., 4.]),
			buffer(&[4], &[1., 1., 1., 1.])
		])?[0]
			.data,
		[14.]
	);
	Ok(())
}
```

```sh check
cargo test --quiet --test fusion_plan reductions_and_rebinding
```

## 11. 明示 STAGE を非融合境界として扱う。
<!-- todo:09-011 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

`src/graph.rs` の `pub fn kernel_bodies(`の定義を次の全体で置き換えます。

```rust file="src/graph.rs" mode="replace" item="pub fn kernel_bodies("
pub fn kernel_bodies(graph: &Graph, outputs: &[NodeId], fuse: bool) -> Result<Vec<KernelBody>> {
	graph.verify(outputs)?;
	let order = graph.topology(outputs)?;
	let mut users = vec![0; graph.nodes.len()];
	for &id in &order {
		for input in graph.nodes[id].op.sources() {
			users[input] += 1
		}
	}
	let material: HashSet<_> = order
		.iter()
		.copied()
		.filter(|&id| {
			outputs.contains(&id)
				|| (!matches!(graph.nodes[id].op, Op::Const { .. }) && (!fuse || users[id] > 1))
				|| matches!(
					graph.nodes[id].op,
					Op::Input { .. }
						| Op::Stage { .. } | Op::Reduce { .. }
						| Op::Gather { .. }
						| Op::ScatterAdd { .. }
				)
		})
		.collect();
	let mut offsets = vec![];
	let mut offset = 0;
	for node in &graph.nodes {
		offsets.push(offset);
		offset += node.shape.len()
	}
	let mut emitter = Emitter {
		g: graph,
		offsets,
		material,
		current: 0,
		serial: 0,
	};
	let mut kernels = vec![];
	for id in order {
		if !emitter.material.contains(&id) || matches!(graph.nodes[id].op, Op::Input { .. }) {
			continue;
		}
		emitter.current = id;
		let mut body = format!("for (size_t i=0; i<{}; ++i) {{\n", emitter.count(id));
		let value = emitter.expr(id, "i", &mut body)?;
		body.push_str(&format!("b{id}[i]={value};\n}}\n"));
		let dependencies = graph
			.topology(&[id])?
			.into_iter()
			.filter(|&x| x != id && emitter.material.contains(&x))
			.collect();
		kernels.push(KernelBody {
			output: id,
			body,
			dependencies,
		});
	}
	crate::closure::calls(graph, &kernels)?;
	Ok(kernels)
}
```

```sh check
cargo test --quiet --test fusion_plan
```

## 12. producer の出力を ALLOC、STORE、AFTER に変える。
<!-- todo:09-012 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

`src/graph.rs` の `pub fn kernel_bodies(`の定義を次の全体で置き換えます。

```rust file="src/graph.rs" mode="replace" item="pub fn kernel_bodies("
pub fn kernel_bodies(graph: &Graph, outputs: &[NodeId], fuse: bool) -> Result<Vec<KernelBody>> {
	graph.verify(outputs)?;
	let order = graph.topology(outputs)?;
	let mut users = vec![0; graph.nodes.len()];
	for &id in &order {
		for input in graph.nodes[id].op.sources() {
			users[input] += 1
		}
	}
	let material: HashSet<_> = order
		.iter()
		.copied()
		.filter(|&id| {
			outputs.contains(&id)
				|| (!matches!(graph.nodes[id].op, Op::Const { .. }) && (!fuse || users[id] > 1))
				|| matches!(
					graph.nodes[id].op,
					Op::Input { .. }
						| Op::Stage { .. } | Op::Reduce { .. }
						| Op::Gather { .. }
						| Op::ScatterAdd { .. }
				)
		})
		.collect();
	let mut offsets = vec![];
	let mut offset = 0;
	for node in &graph.nodes {
		offsets.push(offset);
		offset += node.shape.len()
	}
	let mut emitter = Emitter {
		g: graph,
		offsets,
		material,
		current: 0,
		serial: 0,
	};
	let mut kernels = vec![];
	for id in order {
		if !emitter.material.contains(&id) || matches!(graph.nodes[id].op, Op::Input { .. }) {
			continue;
		}
		emitter.current = id;
		let mut body = format!("for (size_t i=0; i<{}; ++i) {{\n", emitter.count(id));
		let value = emitter.expr(id, "i", &mut body)?;
		body.push_str(&format!("b{id}[i]={value};\n}}\n"));
		let dependencies = graph
			.topology(&[id])?
			.into_iter()
			.filter(|&x| x != id && emitter.material.contains(&x))
			.collect();
		kernels.push(KernelBody {
			output: id,
			body,
			dependencies,
		});
	}
	crate::closure::calls(graph, &kernels)?;
	Ok(kernels)
}
```

`src/runtime.rs` の `pub fn run_with_shapes(`の定義を次の全体で置き換えます。

```rust file="src/runtime.rs" mode="replace" item="pub fn run_with_shapes("
	pub fn run_with_shapes(
		&self,
		inputs: &[Buffer],
		node_shapes: &[Vec<usize>],
	) -> Result<Vec<Buffer>> {
		self.verify_seal()?;
		if self.specialized
			&& self
				.graph
				.nodes
				.iter()
				.map(|n| &n.shape)
				.ne(node_shapes.iter())
		{
			return Err("specialized plan shape mismatch".into());
		}
		let mut graph = self.graph.clone();
		if node_shapes.len() != graph.nodes.len() {
			return Err("node shape table length mismatch".into());
		}
		for (node, shape) in graph.nodes.iter_mut().zip(node_shapes) {
			if node.shape.len() != shape.len() {
				return Err("dynamic rank change is unsupported".into());
			}
			elements(shape)?;
			node.shape = shape.clone();
		}
		graph.verify(&self.outputs)?;
		if inputs.len() != self.abi.inputs.len() {
			return Err("input count mismatch".into());
		}
		for node in &graph.nodes {
			if let crate::Op::Input { slot } = node.op {
				let input = inputs.get(slot).ok_or("missing input")?;
				if input.shape.len() != node.shape.len()
					|| input.data.len() < elements(&node.shape)?
				{
					return Err("input capacity or rank mismatch".into());
				}
			}
		}
		graph.validate_indices(inputs, &self.outputs)?;
		let mut storage = graph
			.nodes
			.iter()
			.enumerate()
			.map(|(id, node)| {
				if self.allocations.contains(&id) {
					Buffer::zeros(node.shape.clone())
				} else {
					Ok(Buffer {
						shape: node.shape.clone(),
						data: vec![],
					})
				}
			})
			.collect::<Result<Vec<_>>>()?;
		let mut pointers = storage
			.iter_mut()
			.map(|b| b.data.as_mut_ptr().cast::<c_void>())
			.collect::<Vec<_>>();
		for (id, node) in graph.nodes.iter().enumerate() {
			if let crate::Op::Input { slot } = node.op {
				pointers[id] = inputs[slot].data.as_ptr().cast_mut().cast();
			}
		}
		let dims = node_shapes
			.iter()
			.flatten()
			.map(|&n| n as u64)
			.collect::<Vec<_>>();
		type Entry = unsafe extern "C" fn(*const *mut c_void, *const u64) -> i32;
		let entry: Entry = unsafe { std::mem::transmute(self.library.symbol()) };
		let status = unsafe { entry(pointers.as_ptr(), dims.as_ptr()) };
		if status != 0 {
			return Err(format!(
				"generated kernel rejected input or allocation: {status}"
			));
		}
		self.outputs
			.iter()
			.map(|&id| {
				if let crate::Op::Input { slot } = graph.nodes[id].op {
					Buffer::new(
						graph.nodes[id].shape.clone(),
						inputs[slot].data[..elements(&graph.nodes[id].shape)?].to_vec(),
					)
				} else {
					Ok(storage[id].clone())
				}
			})
			.collect()
	}
```

```sh check
cargo test --quiet --test fusion_plan
```

## 13. consumer が新しい PARAM から保存済みの値を読むようにする。
<!-- todo:09-013 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

`src/graph.rs` の `fn expr(`の定義を次の全体で置き換えます。

```rust file="src/graph.rs" mode="replace" item="fn expr("
	fn expr(&mut self, id: usize, index: &str, code: &mut String) -> Result<String> {
		if id != self.current && self.material.contains(&id)
			|| matches!(self.g.nodes[id].op, Op::Input { .. })
		{
			return Ok(format!("b{id}[{index}]"));
		}
		let index_name = self.name();
		code.push_str(&format!("size_t {index_name}={index};\n"));
		let index = index_name.as_str();
		let op = self.g.nodes[id].op.clone();
		let value = match op {
			Op::Input { .. } => unreachable!(),
			Op::Const { bits } => format!("chibi_bits({bits}u)"),
			Op::Unary { op, x } => {
				let a = self.expr(x, index, code)?;
				match op {
					Unary::Neg => format!("(-({a}))"),
					Unary::Recip => format!("(1.0f / ({a}))"),
					other => format!(
						"{}({a})",
						match other {
							Unary::Exp => "expf",
							Unary::Log => "logf",
							Unary::Sin => "sinf",
							Unary::Cos => "cosf",
							Unary::Sqrt => "sqrtf",
							Unary::Exp2 => "exp2f",
							Unary::Log2 => "log2f",
							_ => unreachable!(),
						}
					),
				}
			}
			Op::Binary { op, a, b } => {
				let ai = self.broadcast_index(a, id, index);
				let bi = self.broadcast_index(b, id, index);
				let av = self.expr(a, &ai, code)?;
				let bv = self.expr(b, &bi, code)?;
				match op {
					Binary::Max => format!("chibi_max({av},{bv})"),
					_ => format!(
						"(({av}) {} ({bv}))",
						match op {
							Binary::Add => "+",
							Binary::Sub => "-",
							Binary::Mul => "*",
							Binary::Div => "/",
							Binary::Lt => "<",
							Binary::Eq => "==",
							_ => unreachable!(),
						}
					),
				}
			}
			Op::Where { cond, yes, no } => {
				let ci = self.broadcast_index(cond, id, index);
				let yi = self.broadcast_index(yes, id, index);
				let ni = self.broadcast_index(no, id, index);
				let cv = self.expr(cond, &ci, code)?;
				let target = self.name();
				code.push_str(&format!("float {target};\nif ({cv} != 0.0f) {{\n"));
				let y = self.expr(yes, &yi, code)?;
				code.push_str(&format!("{target} = {y};\n}} else {{\n"));
				let n = self.expr(no, &ni, code)?;
				code.push_str(&format!("{target} = {n};\n}}\n"));
				return Ok(target);
			}
			Op::Reshape { x } | Op::Stage { x } | Op::StopGradient { x } => {
				return self.expr(x, index, code);
			}
			Op::Expand { x } => {
				let xi = self.broadcast_index(x, id, index);
				return self.expr(x, &xi, code);
			}
			Op::Reduce {
				op,
				x,
				axes,
				keepdim,
			} => {
				let target = self.name();
				let k = self.name();
				code.push_str(&format!(
					"float {target} = {};\nfor (size_t {k}=0; {k}<{}; ++{k}) {{\n",
					if op == Reduce::Sum {
						"0.0f"
					} else {
						"-INFINITY"
					},
					self.product(x, axes.iter().copied())
				));
				let mut out_axis = 0;
				let c = (0..self.g.shape(x).len())
					.map(|a| {
						if let Some(ra) = axes.iter().position(|&v| v == a) {
							format!(
								"(({k} / {}) % {})",
								self.product(x, axes[ra + 1..].iter().copied()),
								self.dim(x, a)
							)
						} else {
							let oa = if keepdim {
								a
							} else {
								let v = out_axis;
								out_axis += 1;
								v
							};
							self.coord(id, index, oa)
						}
					})
					.collect::<Vec<_>>();
				let xi = self.flatten(x, &c);
				let value = self.expr(x, &xi, code)?;
				code.push_str(&format!(
					"{target} = {};\n}}\n",
					if op == Reduce::Sum {
						format!("{target} + ({value})")
					} else {
						format!("chibi_max({target},{value})")
					}
				));
				return Ok(target);
			}
			Op::Permute { x, axes } => {
				let mut c = vec![String::new(); axes.len()];
				for (a, &old) in axes.iter().enumerate() {
					c[old] = self.coord(id, index, a)
				}
				let xi = self.flatten(x, &c);
				return self.expr(x, &xi, code);
			}
			Op::Shrink { x, ranges } => {
				let c = ranges
					.iter()
					.enumerate()
					.map(|(a, &(lo, _))| format!("({}+{lo})", self.coord(id, index, a)))
					.collect::<Vec<_>>();
				let xi = self.flatten(x, &c);
				return self.expr(x, &xi, code);
			}
			Op::Pad { x, pads } => {
				let c = (0..pads.len())
					.map(|a| self.coord(id, index, a))
					.collect::<Vec<_>>();
				let checks = pads
					.iter()
					.enumerate()
					.map(|(a, &(lo, _))| {
						format!("({} >= {lo} && {} - {lo} < {})", c[a], c[a], self.dim(x, a))
					})
					.collect::<Vec<_>>();
				let target = self.name();
				code.push_str(&format!(
					"float {target}=0.0f;\nif ({}) {{\n",
					if checks.is_empty() {
						"1".into()
					} else {
						checks.join(" && ")
					}
				));
				let xi = self.flatten(
					x,
					&c.iter()
						.zip(&pads)
						.map(|(c, &(lo, _))| format!("({c}-{lo})"))
						.collect::<Vec<_>>(),
				);
				let value = self.expr(x, &xi, code)?;
				code.push_str(&format!("{target}={value};\n}}\n"));
				return Ok(target);
			}
			Op::Flip { x, axes } => {
				let c = (0..self.g.shape(x).len())
					.map(|a| {
						let c = self.coord(id, index, a);
						if axes.contains(&a) {
							format!("({}-1-{c})", self.dim(x, a))
						} else {
							c
						}
					})
					.collect::<Vec<_>>();
				let xi = self.flatten(x, &c);
				return self.expr(x, &xi, code);
			}
			Op::Stack { xs, axis } => {
				let target = self.name();
				code.push_str(&format!("float {target}=0.0f;\n"));
				let select = self.coord(id, index, axis);
				for (j, x) in xs.into_iter().enumerate() {
					code.push_str(&format!(
						"{}if ({select} == {j}) {{\n",
						if j == 0 { "" } else { "else " }
					));
					let c = (0..self.g.shape(id).len())
						.filter(|&a| a != axis)
						.map(|a| self.coord(id, index, a))
						.collect::<Vec<_>>();
					let xi = self.flatten(x, &c);
					let value = self.expr(x, &xi, code)?;
					code.push_str(&format!("{target}={value};\n}}\n"));
				}
				return Ok(target);
			}
			Op::Gather { x, indices, axis } => {
				let ii = self.coord(id, index, axis);
				let iv = self.expr(indices, &ii, code)?;
				let ivar = self.name();
				code.push_str(&format!("float {ivar}={iv};\nif (!isfinite({ivar}) || {ivar}<0.0f || floorf({ivar})!={ivar} || (double){ivar}>=(double){}) return 2;\n",self.dim(x,axis)));
				let mut c = (0..self.g.shape(x).len())
					.map(|a| self.coord(id, index, a))
					.collect::<Vec<_>>();
				c[axis] = format!("(size_t){ivar}");
				let xi = self.flatten(x, &c);
				return self.expr(x, &xi, code);
			}
			Op::ScatterAdd {
				base,
				indices,
				updates,
				axis,
			} => {
				let bv = self.expr(base, index, code)?;
				let target = self.name();
				let k = self.name();
				code.push_str(&format!(
					"float {target}={bv};\nfor (size_t {k}=0; {k}<{}; ++{k}) {{\n",
					self.count(indices)
				));
				let iv = self.expr(indices, &k, code)?;
				let ivar = self.name();
				code.push_str(&format!("float {ivar}={iv};\nif (!isfinite({ivar}) || {ivar}<0.0f || floorf({ivar})!={ivar} || (double){ivar}>=(double){}) return 2;\nif ((size_t){ivar} == {}) {{\n",self.dim(id,axis),self.coord(id,index,axis)));
				let mut c = (0..self.g.shape(id).len())
					.map(|a| self.coord(id, index, a))
					.collect::<Vec<_>>();
				c[axis] = k;
				let ui = self.flatten(updates, &c);
				let uv = self.expr(updates, &ui, code)?;
				code.push_str(&format!("{target}+={uv};\n}}\n}}\n"));
				return Ok(target);
			}
		};
		let variable = self.name();
		code.push_str(&format!("float {variable} = {value};\n"));
		Ok(variable)
	}
```

```sh check
cargo test --quiet --test fusion_plan
```

## 14. `(STAGE(x+1))*2` を二 kernel で `[4,6,8]` にする。
<!-- todo:09-014 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

`tests/fusion_plan.rs` の `fn arithmetic_fusion(`の定義を次の全体で置き換えます。

```rust file="tests/fusion_plan.rs" mode="replace" item="fn arithmetic_fusion("
fn arithmetic_fusion() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let one = g.constant(1.0, vec![])?;
	let two = g.constant(2.0, vec![])?;
	let a = g.binary(Binary::Add, x, one)?;
	let y = g.binary(Binary::Mul, a, two)?;
	let p = g.compile(&[y])?;
	assert_eq!(p.kernel_count, 1);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [4., 6., 8.]);
	let staged = g.stage(a)?;
	let z = g.binary(Binary::Mul, staged, two)?;
	assert_eq!(g.compile(&[z])?.kernel_count, 2);
	let three = g.constant(3., vec![])?;
	let z = g.binary(Binary::Add, a, three)?;
	assert_eq!(g.compile(&[y, z])?.kernel_count, 3);
	let result = compare(&g, &[y, z], &[buffer(&[3], &[1., 2., 3.])])?;
	assert_eq!(result[1].data, [5., 6., 7.]);
	Ok(())
}
```

```sh check
cargo test --quiet --test fusion_plan
```

## 15. `t=x+1; y=t*2; z=t+3` を三 kernel で実行する。
<!-- todo:09-015 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

`tests/fusion_plan.rs` の `fn arithmetic_fusion(`の定義を次の全体で置き換えます。

```rust file="tests/fusion_plan.rs" mode="replace" item="fn arithmetic_fusion("
fn arithmetic_fusion() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let one = g.constant(1.0, vec![])?;
	let two = g.constant(2.0, vec![])?;
	let a = g.binary(Binary::Add, x, one)?;
	let y = g.binary(Binary::Mul, a, two)?;
	let p = g.compile(&[y])?;
	assert_eq!(p.kernel_count, 1);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [4., 6., 8.]);
	let staged = g.stage(a)?;
	let z = g.binary(Binary::Mul, staged, two)?;
	assert_eq!(g.compile(&[z])?.kernel_count, 2);
	let three = g.constant(3., vec![])?;
	let z = g.binary(Binary::Add, a, three)?;
	assert_eq!(g.compile(&[y, z])?.kernel_count, 3);
	let result = compare(&g, &[y, z], &[buffer(&[3], &[1., 2., 3.])])?;
	assert_eq!(result[1].data, [5., 6., 7.]);
	Ok(())
}
```

```sh check
cargo test --quiet --test fusion_plan
```

## 16. 各 body の外部 storage と scalar を PARAM に正規化する。
<!-- todo:09-016 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

`src/graph.rs` の `pub fn render(`の定義を次の全体で置き換えます。

```rust file="src/graph.rs" mode="replace" item="pub fn render("
pub fn render(graph: &Graph, outputs: &[NodeId], fuse: bool) -> Result<(String, Abi, usize)> {
	let kernels = kernel_bodies(graph, outputs, fuse)?;
	let source = crate::native::render_entry(C_HELPERS, &graph.topology(outputs)?, &kernels);
	let mut inputs = vec![];
	for node in &graph.nodes {
		if let Op::Input { slot } = node.op {
			let binding = Binding {
				slot,
				shape: node.shape.clone(),
			};
			if !inputs.contains(&binding) {
				inputs.push(binding)
			}
		}
	}
	inputs.sort_by_key(|b| b.slot);
	let outputs = outputs
		.iter()
		.map(|&id| Binding {
			slot: id,
			shape: graph.shape(id).to_vec(),
		})
		.collect();
	Ok((source, Abi { inputs, outputs }, kernels.len()))
}
```

`src/closure.rs` の `pub fn signatures(`の定義を次の全体で置き換えます。

```rust file="src/closure.rs" mode="replace" item="pub fn signatures("
pub fn signatures(graph: &Graph, kernels: &[KernelBody]) -> Result<Vec<KernelSignature>> {
	let scalar_words = graph.nodes.iter().try_fold(0usize, |n, node| {
		n.checked_add(node.shape.len())
			.ok_or("scalar signature overflow")
	})?;
	kernels
		.iter()
		.map(|kernel| {
			if kernel.output >= graph.nodes.len() {
				return Err("kernel output outside graph".into());
			}
			if matches!(graph.nodes[kernel.output].op, Op::Input { .. }) {
				return Err("kernel cannot STORE into a read-only input".into());
			}
			let mut buffers = kernel.dependencies.clone();
			buffers.push(kernel.output);
			let unique: HashSet<_> = buffers.iter().copied().collect();
			if unique.len() != buffers.len() || buffers.iter().any(|&id| id >= graph.nodes.len()) {
				return Err("kernel signature has duplicate or unknown storage".into());
			}
			Ok(KernelSignature {
				buffers,
				scalar_words,
				output: kernel.output,
			})
		})
		.collect()
}
```

`src/closure.rs` の `pub fn calls(`の定義を次の全体で置き換えます。

```rust file="src/closure.rs" mode="replace" item="pub fn calls("
pub fn calls(graph: &Graph, kernels: &[KernelBody]) -> Result<Vec<KernelCall>> {
	let signatures = signatures(graph, kernels)?;
	let mut produced = std::collections::HashMap::new();
	let mut result = Vec::new();
	for (index, signature) in signatures.iter().enumerate() {
		let mut after = Vec::new();
		for &id in &kernels[index].dependencies {
			if !matches!(graph.nodes[id].op, Op::Input { .. }) {
				after.push(
					*produced
						.get(&id)
						.ok_or("producer must complete before consumer")?,
				);
			}
		}
		after.sort_unstable();
		after.dedup();
		let state = index + 1;
		if produced.insert(signature.output, state).is_some() {
			return Err("multiple producers for one output state".into());
		}
		result.push(KernelCall {
			target: index,
			buffers: signature.buffers.clone(),
			scalar_words: signature.scalar_words,
			after,
			output_state: state,
		});
	}
	verify_calls(graph, kernels, &result)?;
	Ok(result)
}
```

```sh check
cargo test --quiet --test fusion_plan
```

## 17. kernel の署名、CALL 引数、出力 state を照合する verifier を追加する。
<!-- todo:09-017 -->

次のテストは署名・CALL・stateの不一致を個別に作り、生成コードの実行も確認します。

```rust file=tests/closed_kernels.rs mode=write
use chinigrad::{
	Binary, Buffer, Graph, Result,
	closure::{calls, signatures, verify_body, verify_calls},
};

fn fixture() -> Result<(Graph, Vec<chinigrad::native::KernelBody>)> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let _outside = g.input(1, vec![3])?;
	let one = g.constant(1., vec![])?;
	let a = g.binary(Binary::Add, x, one)?;
	let stage = g.stage(a)?;
	let out = g.binary(Binary::Mul, stage, one)?;
	let bodies = chinigrad::graph::kernel_bodies(&g, &[out], true)?;
	Ok((g, bodies))
}

#[test]
fn signature_call_and_output_state_are_checked() -> Result<()> {
	let (graph, bodies) = fixture()?;
	let schedule = calls(&graph, &bodies)?;
	verify_calls(&graph, &bodies, &schedule)?;
	let mut wrong = schedule.clone();
	wrong[0].buffers.swap(0, 1);
	assert!(
		verify_calls(&graph, &bodies, &wrong)
			.unwrap_err()
			.contains("arguments")
	);
	let mut wrong = schedule.clone();
	wrong[0].scalar_words += 1;
	assert!(verify_calls(&graph, &bodies, &wrong).is_err());
	let mut wrong = schedule.clone();
	wrong[0].output_state += 1;
	assert!(
		verify_calls(&graph, &bodies, &wrong)
			.unwrap_err()
			.contains("output state")
	);
	let mut wrong = schedule.clone();
	wrong[1].after.clear();
	assert!(
		verify_calls(&graph, &bodies, &wrong)
			.unwrap_err()
			.contains("producer state")
	);
	let reversed = bodies.into_iter().rev().collect::<Vec<_>>();
	assert!(calls(&graph, &reversed).is_err());
	Ok(())
}

#[test]
fn unbound_storage_scalar_range_and_missing_store_are_rejected() -> Result<()> {
	let (graph, bodies) = fixture()?;
	let signature = &signatures(&graph, &bodies)?[0];
	let body = &bodies[0].body;
	assert!(
		verify_body(&(body.clone() + "float v999=b1[0];\n"), signature)
			.unwrap_err()
			.contains("unbound storage")
	);
	let scalar = format!("float v999=s[{}];\n", signature.scalar_words);
	assert!(
		verify_body(&(body.clone() + &scalar), signature)
			.unwrap_err()
			.contains("scalar")
	);
	let closed_range = format!("b{}[0]=i;\n", signature.output);
	assert!(
		verify_body(&(body.clone() + &closed_range), signature)
			.unwrap_err()
			.contains("RANGE")
	);
	assert!(
		verify_body("for(size_t i=0;i<1;++i){float v0=0.0f;}\n", signature)
			.unwrap_err()
			.contains("STORE")
	);
	let wrong_store = body.replace(&format!("b{}[i]", signature.output), "b0[i]");
	assert!(
		verify_body(&wrong_store, signature)
			.unwrap_err()
			.contains("non-output")
	);
	for update in ["b0[i]++", "--b0[i]", "b0[i]<<=1", "s[0]=0", "s[0]++"] {
		let forged = format!(
			"for(size_t i=0;i<s[0];++i){{{update};b{}[i]=b0[i];}}",
			signature.output
		);
		assert!(
			verify_body(&forged, signature).is_err(),
			"accepted {update}"
		);
	}
	let uninitialized = format!(
		"for(size_t i=0;i<s[0];++i){{b{0}[i]=b{0}[i]+b0[i];}}",
		signature.output
	);
	assert!(
		verify_body(&uninitialized, signature)
			.unwrap_err()
			.contains("output LOAD")
	);
	Ok(())
}

#[test]
fn closing_checks_preserve_native_execution_and_original_graph() -> Result<()> {
	let (graph, bodies) = fixture()?;
	let original = format!("{graph:?}");
	let output = bodies.last().unwrap().output;
	let plan = graph.compile(&[output])?;
	let input = [
		Buffer::new(vec![3], vec![1., 2., 3.])?,
		Buffer::new(vec![3], vec![9., 9., 9.])?,
	];
	assert_eq!(plan.run(&input)?[0].data, vec![2., 3., 4.]);
	assert_eq!(plan.run(&input)?[0].data, vec![2., 3., 4.]);
	assert_eq!(format!("{graph:?}"), original);
	Ok(())
}
```

`src/closure.rs` の `pub fn verify_calls(`の定義を次の全体で置き換えます。

```rust file="src/closure.rs" mode="replace" item="pub fn verify_calls("
pub fn verify_calls(graph: &Graph, kernels: &[KernelBody], calls: &[KernelCall]) -> Result<()> {
	let signatures = signatures(graph, kernels)?;
	if calls.len() != kernels.len() {
		return Err("kernel CALL count mismatch".into());
	}
	let mut initialized: HashSet<usize> = graph
		.nodes
		.iter()
		.enumerate()
		.filter_map(|(id, node)| matches!(node.op, Op::Input { .. }).then_some(id))
		.collect();
	let mut states = std::collections::HashMap::new();
	for (index, (signature, call)) in signatures.iter().zip(calls).enumerate() {
		if call.target != index
			|| call.buffers != signature.buffers
			|| call.scalar_words != signature.scalar_words
		{
			return Err("kernel signature and CALL arguments differ".into());
		}
		if call.output_state != index + 1 {
			return Err("kernel output state mismatch".into());
		}
		if initialized.contains(&signature.output) {
			return Err("duplicate producer STORE".into());
		}
		let mut required = Vec::new();
		for &id in &kernels[index].dependencies {
			if !initialized.contains(&id) {
				return Err("LOAD before producer STORE".into());
			}
			if let Some(state) = states.get(&id) {
				required.push(*state);
			}
		}
		required.sort_unstable();
		required.dedup();
		if call.after != required {
			return Err("CALL does not depend on the producer state".into());
		}
		verify_body(&kernels[index].body, signature)?;
		initialized.insert(signature.output);
		states.insert(signature.output, call.output_state);
	}
	Ok(())
}
```

```sh check
cargo test --test closed_kernels
```


`tests/fusion_plan.rs` の `dynamic_metadata` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test fusion_plan dynamic_metadata
```

## 18. producer と consumer の辺から実行順を作り、循環なら拒否する。
<!-- todo:09-018 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

`src/graph.rs` の `pub fn topology(`の定義を次の全体で置き換えます。

```rust file="src/graph.rs" mode="replace" item="pub fn topology("
	pub fn topology(&self, outputs: &[NodeId]) -> Result<Vec<NodeId>> {
		fn visit(g: &Graph, id: usize, marks: &mut [u8], order: &mut Vec<usize>) -> Result<()> {
			g.get(id)?;
			if marks[id] == 1 {
				return Err("graph cycle".into());
			}
			if marks[id] == 2 {
				return Ok(());
			}
			marks[id] = 1;
			for child in g.nodes[id].op.sources() {
				visit(g, child, marks, order)?
			}
			marks[id] = 2;
			order.push(id);
			Ok(())
		}
		if outputs.is_empty() {
			return Err("no outputs".into());
		}
		let mut marks = vec![0; self.nodes.len()];
		let mut order = vec![];
		for &id in outputs {
			visit(self, id, &mut marks, &mut order)?
		}
		Ok(order)
	}
```

`src/closure.rs` の `pub fn calls(`の定義を次の全体で置き換えます。

```rust file="src/closure.rs" mode="replace" item="pub fn calls("
pub fn calls(graph: &Graph, kernels: &[KernelBody]) -> Result<Vec<KernelCall>> {
	let signatures = signatures(graph, kernels)?;
	let mut produced = std::collections::HashMap::new();
	let mut result = Vec::new();
	for (index, signature) in signatures.iter().enumerate() {
		let mut after = Vec::new();
		for &id in &kernels[index].dependencies {
			if !matches!(graph.nodes[id].op, Op::Input { .. }) {
				after.push(
					*produced
						.get(&id)
						.ok_or("producer must complete before consumer")?,
				);
			}
		}
		after.sort_unstable();
		after.dedup();
		let state = index + 1;
		if produced.insert(signature.output, state).is_some() {
			return Err("multiple producers for one output state".into());
		}
		result.push(KernelCall {
			target: index,
			buffers: signature.buffers.clone(),
			scalar_words: signature.scalar_words,
			after,
			output_state: state,
		});
	}
	verify_calls(graph, kernels, &result)?;
	Ok(result)
}
```

```sh check
cargo test --quiet --test fusion_plan
```

## 19. 既存の CPU compile と run を Plan の kernel 列へ接続する。
<!-- todo:09-019 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

`src/runtime.rs` の `pub fn run_with_shapes(`の定義を次の全体で置き換えます。

```rust file="src/runtime.rs" mode="replace" item="pub fn run_with_shapes("
	pub fn run_with_shapes(
		&self,
		inputs: &[Buffer],
		node_shapes: &[Vec<usize>],
	) -> Result<Vec<Buffer>> {
		self.verify_seal()?;
		if self.specialized
			&& self
				.graph
				.nodes
				.iter()
				.map(|n| &n.shape)
				.ne(node_shapes.iter())
		{
			return Err("specialized plan shape mismatch".into());
		}
		let mut graph = self.graph.clone();
		if node_shapes.len() != graph.nodes.len() {
			return Err("node shape table length mismatch".into());
		}
		for (node, shape) in graph.nodes.iter_mut().zip(node_shapes) {
			if node.shape.len() != shape.len() {
				return Err("dynamic rank change is unsupported".into());
			}
			elements(shape)?;
			node.shape = shape.clone();
		}
		graph.verify(&self.outputs)?;
		if inputs.len() != self.abi.inputs.len() {
			return Err("input count mismatch".into());
		}
		for node in &graph.nodes {
			if let crate::Op::Input { slot } = node.op {
				let input = inputs.get(slot).ok_or("missing input")?;
				if input.shape.len() != node.shape.len()
					|| input.data.len() < elements(&node.shape)?
				{
					return Err("input capacity or rank mismatch".into());
				}
			}
		}
		graph.validate_indices(inputs, &self.outputs)?;
		let mut storage = graph
			.nodes
			.iter()
			.enumerate()
			.map(|(id, node)| {
				if self.allocations.contains(&id) {
					Buffer::zeros(node.shape.clone())
				} else {
					Ok(Buffer {
						shape: node.shape.clone(),
						data: vec![],
					})
				}
			})
			.collect::<Result<Vec<_>>>()?;
		let mut pointers = storage
			.iter_mut()
			.map(|b| b.data.as_mut_ptr().cast::<c_void>())
			.collect::<Vec<_>>();
		for (id, node) in graph.nodes.iter().enumerate() {
			if let crate::Op::Input { slot } = node.op {
				pointers[id] = inputs[slot].data.as_ptr().cast_mut().cast();
			}
		}
		let dims = node_shapes
			.iter()
			.flatten()
			.map(|&n| n as u64)
			.collect::<Vec<_>>();
		type Entry = unsafe extern "C" fn(*const *mut c_void, *const u64) -> i32;
		let entry: Entry = unsafe { std::mem::transmute(self.library.symbol()) };
		let status = unsafe { entry(pointers.as_ptr(), dims.as_ptr()) };
		if status != 0 {
			return Err(format!(
				"generated kernel rejected input or allocation: {status}"
			));
		}
		self.outputs
			.iter()
			.map(|&id| {
				if let crate::Op::Input { slot } = graph.nodes[id].op {
					Buffer::new(
						graph.nodes[id].shape.clone(),
						inputs[slot].data[..elements(&graph.nodes[id].shape)?].to_vec(),
					)
				} else {
					Ok(storage[id].clone())
				}
			})
			.collect()
	}
```

```sh check
cargo test --quiet --test fusion_plan
```

## 20. 同じ Plan に新しい重みを渡して、二回目の出力を確認する。
<!-- todo:09-020 -->

`tests/fusion_plan.rs` の `reductions_and_rebinding` で、この経路の値または拒否条件を確認する。

`tests/fusion_plan.rs` の `fn reductions_and_rebinding(`の定義を次の全体で置き換えます。

```rust file="tests/fusion_plan.rs" mode="replace" item="fn reductions_and_rebinding("
fn reductions_and_rebinding() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![4])?;
	let w = g.input(1, vec![4])?;
	let one = g.constant(1., vec![])?;
	let p = g.binary(Binary::Mul, x, w)?;
	let p = g.binary(Binary::Add, p, one)?;
	let y = g.reduce(p, Reduce::Sum, vec![0], false)?;
	let p = g.compile(&[y])?;
	assert_eq!(p.kernel_count, 1);
	assert_eq!(
		p.run(&[
			buffer(&[4], &[1., 2., 3., 4.]),
			buffer(&[4], &[2., 3., 4., 5.])
		])?[0]
			.data,
		[44.]
	);
	assert_eq!(
		p.run(&[
			buffer(&[4], &[1., 2., 3., 4.]),
			buffer(&[4], &[1., 1., 1., 1.])
		])?[0]
			.data,
		[14.]
	);
	Ok(())
}
```

```sh check
cargo test --quiet --test fusion_plan reductions_and_rebinding
```

## 21. body の未束縛 PARAM、自由 RANGE、STORE 先の漏れを拒否する。
<!-- todo:09-021 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

`src/closure.rs` の `pub fn verify_body(`の定義を次の全体で置き換えます。

```rust file="src/closure.rs" mode="replace" item="pub fn verify_body("
pub fn verify_body(body: &str, signature: &KernelSignature) -> Result<()> {
	let code = tokens(body)?;
	let mut scopes: Vec<(HashSet<String>, bool)> = vec![(HashSet::new(), false)];
	let mut pending_for = false;
	let mut stored = false;
	let types = [
		"float", "double", "size_t", "int32_t", "uint32_t", "uint64_t",
	];
	let fixed = [
		"for",
		"if",
		"else",
		"return",
		"const",
		"s",
		"NAN",
		"INFINITY",
		"chibi_bits",
		"chibi_max",
		"expf",
		"logf",
		"sinf",
		"cosf",
		"sqrtf",
		"exp2f",
		"log2f",
		"floorf",
		"isfinite",
	];
	for (position, word) in code.iter().enumerate() {
		if word == "for" {
			if pending_for {
				return Err("nested incomplete RANGE header".into());
			}
			scopes.push((HashSet::new(), false));
			pending_for = true;
		} else if word == "{" {
			scopes.push((HashSet::new(), pending_for));
			pending_for = false;
		} else if word == "}" {
			let (_, closes_for) = scopes.pop().ok_or("unmatched kernel scope")?;
			if closes_for {
				scopes.pop().ok_or("missing RANGE scope")?;
			}
			if scopes.is_empty() {
				return Err("kernel scope escaped body".into());
			}
		}
		if let Some(digits) = word
			.strip_prefix('b')
			.filter(|s| !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit()))
		{
			let id: usize = digits.parse().map_err(|_| "storage ID overflow")?;
			if !signature.buffers.contains(&id) {
				return Err(format!("unbound storage PARAM b{id}"));
			}
			if code.get(position + 1).map(String::as_str) != Some("[") {
				return Err("storage must be indexed".into());
			}
			let mut nesting = 1;
			let mut end = position + 2;
			while end < code.len() && nesting != 0 {
				match code[end].as_str() {
					"[" => nesting += 1,
					"]" => nesting -= 1,
					_ => {}
				}
				end += 1;
			}
			if nesting != 0 {
				return Err("unclosed storage index".into());
			}
			let (write, assignment) = mutation(&code, position, end);
			if write {
				if id != signature.output {
					return Err("STORE targets a non-output PARAM".into());
				}
				if !assignment {
					return Err("output update reads uninitialized output".into());
				}
				stored = true;
			} else if id == signature.output {
				return Err("output LOAD is not allowed before its STORE".into());
			}
			continue;
		}
		if word == "s" {
			if code.get(position + 1).map(String::as_str) != Some("[") {
				return Err("scalar PARAM must use a literal slot".into());
			}
			let slot: usize = code
				.get(position + 2)
				.ok_or("missing scalar slot")?
				.parse()
				.map_err(|_| "nonliteral scalar slot")?;
			if slot >= signature.scalar_words
				|| code.get(position + 3).map(String::as_str) != Some("]")
			{
				return Err("unbound scalar PARAM".into());
			}
			if mutation(&code, position, position + 4).0 {
				return Err("scalar PARAM is read-only".into());
			}
			continue;
		}
		let identifier = word.as_bytes()[0].is_ascii_alphabetic() || word.starts_with('_');
		if !identifier || types.contains(&word.as_str()) || fixed.contains(&word.as_str()) {
			continue;
		}
		let declaration = position > 0 && types.contains(&code[position - 1].as_str());
		if declaration {
			if !scopes.last_mut().unwrap().0.insert(word.clone()) {
				return Err("duplicate local declaration".into());
			}
		} else if !scopes.iter().rev().any(|scope| scope.0.contains(word)) {
			return Err(format!("free or closed RANGE/local {word}"));
		}
	}
	if pending_for || scopes.len() != 1 {
		return Err("unclosed RANGE or kernel scope".into());
	}
	if !stored {
		return Err("kernel is missing its output STORE".into());
	}
	Ok(())
}
```

`src/closure.rs` の `fn mutation(`の定義を次の全体で置き換えます。

```rust file="src/closure.rs" mode="replace" item="fn mutation("
fn mutation(code: &[String], position: usize, end: usize) -> (bool, bool) {
	let at = |i: usize| code.get(i).map(String::as_str);
	let assignment = at(end) == Some("=") && at(end + 1) != Some("=");
	let compound = matches!(at(end), Some("+" | "-" | "*" | "/" | "%" | "&" | "|" | "^"))
		&& at(end + 1) == Some("=");
	let shift = matches!(
		(at(end), at(end + 1), at(end + 2)),
		(Some("<"), Some("<"), Some("=")) | (Some(">"), Some(">"), Some("="))
	);
	let suffix = matches!(
		(at(end), at(end + 1)),
		(Some("+"), Some("+")) | (Some("-"), Some("-"))
	);
	let prefix = position >= 2
		&& matches!(
			(at(position - 2), at(position - 1)),
			(Some("+"), Some("+")) | (Some("-"), Some("-"))
		);
	(
		assignment || compound || shift || suffix || prefix,
		assignment,
	)
}
```

```sh check
cargo test --quiet --test fusion_plan
```

## 22. 外側 kernel の NodeId が body に直接残った例を拒否する。
<!-- todo:09-022 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

`src/closure.rs` の `pub fn verify_body(`の定義を次の全体で置き換えます。

```rust file="src/closure.rs" mode="replace" item="pub fn verify_body("
pub fn verify_body(body: &str, signature: &KernelSignature) -> Result<()> {
	let code = tokens(body)?;
	let mut scopes: Vec<(HashSet<String>, bool)> = vec![(HashSet::new(), false)];
	let mut pending_for = false;
	let mut stored = false;
	let types = [
		"float", "double", "size_t", "int32_t", "uint32_t", "uint64_t",
	];
	let fixed = [
		"for",
		"if",
		"else",
		"return",
		"const",
		"s",
		"NAN",
		"INFINITY",
		"chibi_bits",
		"chibi_max",
		"expf",
		"logf",
		"sinf",
		"cosf",
		"sqrtf",
		"exp2f",
		"log2f",
		"floorf",
		"isfinite",
	];
	for (position, word) in code.iter().enumerate() {
		if word == "for" {
			if pending_for {
				return Err("nested incomplete RANGE header".into());
			}
			scopes.push((HashSet::new(), false));
			pending_for = true;
		} else if word == "{" {
			scopes.push((HashSet::new(), pending_for));
			pending_for = false;
		} else if word == "}" {
			let (_, closes_for) = scopes.pop().ok_or("unmatched kernel scope")?;
			if closes_for {
				scopes.pop().ok_or("missing RANGE scope")?;
			}
			if scopes.is_empty() {
				return Err("kernel scope escaped body".into());
			}
		}
		if let Some(digits) = word
			.strip_prefix('b')
			.filter(|s| !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit()))
		{
			let id: usize = digits.parse().map_err(|_| "storage ID overflow")?;
			if !signature.buffers.contains(&id) {
				return Err(format!("unbound storage PARAM b{id}"));
			}
			if code.get(position + 1).map(String::as_str) != Some("[") {
				return Err("storage must be indexed".into());
			}
			let mut nesting = 1;
			let mut end = position + 2;
			while end < code.len() && nesting != 0 {
				match code[end].as_str() {
					"[" => nesting += 1,
					"]" => nesting -= 1,
					_ => {}
				}
				end += 1;
			}
			if nesting != 0 {
				return Err("unclosed storage index".into());
			}
			let (write, assignment) = mutation(&code, position, end);
			if write {
				if id != signature.output {
					return Err("STORE targets a non-output PARAM".into());
				}
				if !assignment {
					return Err("output update reads uninitialized output".into());
				}
				stored = true;
			} else if id == signature.output {
				return Err("output LOAD is not allowed before its STORE".into());
			}
			continue;
		}
		if word == "s" {
			if code.get(position + 1).map(String::as_str) != Some("[") {
				return Err("scalar PARAM must use a literal slot".into());
			}
			let slot: usize = code
				.get(position + 2)
				.ok_or("missing scalar slot")?
				.parse()
				.map_err(|_| "nonliteral scalar slot")?;
			if slot >= signature.scalar_words
				|| code.get(position + 3).map(String::as_str) != Some("]")
			{
				return Err("unbound scalar PARAM".into());
			}
			if mutation(&code, position, position + 4).0 {
				return Err("scalar PARAM is read-only".into());
			}
			continue;
		}
		let identifier = word.as_bytes()[0].is_ascii_alphabetic() || word.starts_with('_');
		if !identifier || types.contains(&word.as_str()) || fixed.contains(&word.as_str()) {
			continue;
		}
		let declaration = position > 0 && types.contains(&code[position - 1].as_str());
		if declaration {
			if !scopes.last_mut().unwrap().0.insert(word.clone()) {
				return Err("duplicate local declaration".into());
			}
		} else if !scopes.iter().rev().any(|scope| scope.0.contains(word)) {
			return Err(format!("free or closed RANGE/local {word}"));
		}
	}
	if pending_for || scopes.len() != 1 {
		return Err("unclosed RANGE or kernel scope".into());
	}
	if !stored {
		return Err("kernel is missing its output STORE".into());
	}
	Ok(())
}
```

`src/closure.rs` の `fn mutation(`の定義を次の全体で置き換えます。

```rust file="src/closure.rs" mode="replace" item="fn mutation("
fn mutation(code: &[String], position: usize, end: usize) -> (bool, bool) {
	let at = |i: usize| code.get(i).map(String::as_str);
	let assignment = at(end) == Some("=") && at(end + 1) != Some("=");
	let compound = matches!(at(end), Some("+" | "-" | "*" | "/" | "%" | "&" | "|" | "^"))
		&& at(end + 1) == Some("=");
	let shift = matches!(
		(at(end), at(end + 1), at(end + 2)),
		(Some("<"), Some("<"), Some("=")) | (Some(">"), Some(">"), Some("="))
	);
	let suffix = matches!(
		(at(end), at(end + 1)),
		(Some("+"), Some("+")) | (Some("-"), Some("-"))
	);
	let prefix = position >= 2
		&& matches!(
			(at(position - 2), at(position - 1)),
			(Some("+"), Some("+")) | (Some("-"), Some("-"))
		);
	(
		assignment || compound || shift || suffix || prefix,
		assignment,
	)
}
```

```sh check
cargo test --quiet --test fusion_plan
```

## 23. 二つの違う座標で同じ producer を使う fixture を実行する。
<!-- todo:09-023 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

`tests/fusion_plan.rs` の `fn arithmetic_fusion(`の定義を次の全体で置き換えます。

```rust file="tests/fusion_plan.rs" mode="replace" item="fn arithmetic_fusion("
fn arithmetic_fusion() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let one = g.constant(1.0, vec![])?;
	let two = g.constant(2.0, vec![])?;
	let a = g.binary(Binary::Add, x, one)?;
	let y = g.binary(Binary::Mul, a, two)?;
	let p = g.compile(&[y])?;
	assert_eq!(p.kernel_count, 1);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [4., 6., 8.]);
	let staged = g.stage(a)?;
	let z = g.binary(Binary::Mul, staged, two)?;
	assert_eq!(g.compile(&[z])?.kernel_count, 2);
	let three = g.constant(3., vec![])?;
	let z = g.binary(Binary::Add, a, three)?;
	assert_eq!(g.compile(&[y, z])?.kernel_count, 3);
	let result = compare(&g, &[y, z], &[buffer(&[3], &[1., 2., 3.])])?;
	assert_eq!(result[1].data, [5., 6., 7.]);
	Ok(())
}
```

```sh check
cargo test --quiet --test fusion_plan
```

## 24. 元グラフを run の前後で dump し、構造が変わらないことを確認する。
<!-- todo:09-024 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

`tests/fusion_plan.rs` の `fn arithmetic_fusion(`の定義を次の全体で置き換えます。

```rust file="tests/fusion_plan.rs" mode="replace" item="fn arithmetic_fusion("
fn arithmetic_fusion() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let one = g.constant(1.0, vec![])?;
	let two = g.constant(2.0, vec![])?;
	let a = g.binary(Binary::Add, x, one)?;
	let y = g.binary(Binary::Mul, a, two)?;
	let p = g.compile(&[y])?;
	assert_eq!(p.kernel_count, 1);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [4., 6., 8.]);
	let staged = g.stage(a)?;
	let z = g.binary(Binary::Mul, staged, two)?;
	assert_eq!(g.compile(&[z])?.kernel_count, 2);
	let three = g.constant(3., vec![])?;
	let z = g.binary(Binary::Add, a, three)?;
	assert_eq!(g.compile(&[y, z])?.kernel_count, 3);
	let result = compare(&g, &[y, z], &[buffer(&[3], &[1., 2., 3.])])?;
	assert_eq!(result[1].data, [5., 6., 7.]);
	Ok(())
}
```

```sh check
cargo test --quiet --test fusion_plan
```

## 25. 入力の長さと dtype が署名と違う場合、実行開始前に拒否する。
<!-- todo:09-025 -->

`tests/fusion_plan.rs` の `dynamic_metadata` で、この経路の値または拒否条件を確認する。

`src/graph.rs` の `pub fn validate_inputs(`の定義を次の全体で置き換えます。

```rust file="src/graph.rs" mode="replace" item="pub fn validate_inputs("
	pub fn validate_inputs(&self, inputs: &[Buffer], _outputs: &[NodeId]) -> Result<()> {
		let mut slots = HashMap::new();
		for id in 0..self.nodes.len() {
			if let Op::Input { slot } = self.nodes[id].op {
				if let Some(old) = slots.insert(slot, &self.nodes[id].shape)
					&& old != &self.nodes[id].shape
				{
					return Err("one input slot has inconsistent shapes".into());
				}
				let b = inputs.get(slot).ok_or("missing input slot")?;
				if b.shape != self.nodes[id].shape || b.data.len() != elements(&b.shape)? {
					return Err(format!("input {slot} shape mismatch"));
				}
			}
		}
		if slots.len() != inputs.len() || (0..inputs.len()).any(|slot| !slots.contains_key(&slot)) {
			return Err("input slots must be dense and exact".into());
		}
		Ok(())
	}
```

```sh check
cargo test --quiet --test fusion_plan dynamic_metadata
```

## 26. chain は一 kernel、STAGE は二 kernel、fanout は三 kernel になる。
<!-- todo:09-026 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

`tests/fusion_plan.rs` の `fn arithmetic_fusion(`の定義を次の全体で置き換えます。

```rust file="tests/fusion_plan.rs" mode="replace" item="fn arithmetic_fusion("
fn arithmetic_fusion() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let one = g.constant(1.0, vec![])?;
	let two = g.constant(2.0, vec![])?;
	let a = g.binary(Binary::Add, x, one)?;
	let y = g.binary(Binary::Mul, a, two)?;
	let p = g.compile(&[y])?;
	assert_eq!(p.kernel_count, 1);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [4., 6., 8.]);
	let staged = g.stage(a)?;
	let z = g.binary(Binary::Mul, staged, two)?;
	assert_eq!(g.compile(&[z])?.kernel_count, 2);
	let three = g.constant(3., vec![])?;
	let z = g.binary(Binary::Add, a, three)?;
	assert_eq!(g.compile(&[y, z])?.kernel_count, 3);
	let result = compare(&g, &[y, z], &[buffer(&[3], &[1., 2., 3.])])?;
	assert_eq!(result[1].data, [5., 6., 7.]);
	Ok(())
}
```

```sh check
cargo test --quiet --test fusion_plan
```

## 27. 非融合と融合の値が `[4,6,8]` で一致する。
<!-- todo:09-027 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

`tests/fusion_plan.rs` の `fn arithmetic_fusion(`の定義を次の全体で置き換えます。

```rust file="tests/fusion_plan.rs" mode="replace" item="fn arithmetic_fusion("
fn arithmetic_fusion() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let one = g.constant(1.0, vec![])?;
	let two = g.constant(2.0, vec![])?;
	let a = g.binary(Binary::Add, x, one)?;
	let y = g.binary(Binary::Mul, a, two)?;
	let p = g.compile(&[y])?;
	assert_eq!(p.kernel_count, 1);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [4., 6., 8.]);
	let staged = g.stage(a)?;
	let z = g.binary(Binary::Mul, staged, two)?;
	assert_eq!(g.compile(&[z])?.kernel_count, 2);
	let three = g.constant(3., vec![])?;
	let z = g.binary(Binary::Add, a, three)?;
	assert_eq!(g.compile(&[y, z])?.kernel_count, 3);
	let result = compare(&g, &[y, z], &[buffer(&[3], &[1., 2., 3.])])?;
	assert_eq!(result[1].data, [5., 6., 7.]);
	Ok(())
}
```

```sh check
cargo test --quiet --test fusion_plan
```

## 28. fanout の二出力が `[4,6,8]` と `[5,6,7]` になる。
<!-- todo:09-028 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

`tests/fusion_plan.rs` の `fn arithmetic_fusion(`の定義を次の全体で置き換えます。

```rust file="tests/fusion_plan.rs" mode="replace" item="fn arithmetic_fusion("
fn arithmetic_fusion() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let one = g.constant(1.0, vec![])?;
	let two = g.constant(2.0, vec![])?;
	let a = g.binary(Binary::Add, x, one)?;
	let y = g.binary(Binary::Mul, a, two)?;
	let p = g.compile(&[y])?;
	assert_eq!(p.kernel_count, 1);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [4., 6., 8.]);
	let staged = g.stage(a)?;
	let z = g.binary(Binary::Mul, staged, two)?;
	assert_eq!(g.compile(&[z])?.kernel_count, 2);
	let three = g.constant(3., vec![])?;
	let z = g.binary(Binary::Add, a, three)?;
	assert_eq!(g.compile(&[y, z])?.kernel_count, 3);
	let result = compare(&g, &[y, z], &[buffer(&[3], &[1., 2., 3.])])?;
	assert_eq!(result[1].data, [5., 6., 7.]);
	Ok(())
}
```

```sh check
cargo test --quiet --test fusion_plan
```

## 29. 同じ Plan の再実行で `44` から `14` へ変わり、再コンパイルしない。
<!-- todo:09-029 -->

`tests/fusion_plan.rs` の `reductions_and_rebinding` で、この経路の値または拒否条件を確認する。

`tests/fusion_plan.rs` の `fn reductions_and_rebinding(`の定義を次の全体で置き換えます。

```rust file="tests/fusion_plan.rs" mode="replace" item="fn reductions_and_rebinding("
fn reductions_and_rebinding() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![4])?;
	let w = g.input(1, vec![4])?;
	let one = g.constant(1., vec![])?;
	let p = g.binary(Binary::Mul, x, w)?;
	let p = g.binary(Binary::Add, p, one)?;
	let y = g.reduce(p, Reduce::Sum, vec![0], false)?;
	let p = g.compile(&[y])?;
	assert_eq!(p.kernel_count, 1);
	assert_eq!(
		p.run(&[
			buffer(&[4], &[1., 2., 3., 4.]),
			buffer(&[4], &[2., 3., 4., 5.])
		])?[0]
			.data,
		[44.]
	);
	assert_eq!(
		p.run(&[
			buffer(&[4], &[1., 2., 3., 4.]),
			buffer(&[4], &[1., 1., 1., 1.])
		])?[0]
			.data,
		[14.]
	);
	Ok(())
}
```

```sh check
cargo test --quiet --test fusion_plan reductions_and_rebinding
```

## 30. 閉じた body に未束縛の storage と RANGE が残らない。
<!-- todo:09-030 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

`tests/fusion_plan.rs` の `fn arithmetic_fusion(`の定義を次の全体で置き換えます。

```rust file="tests/fusion_plan.rs" mode="replace" item="fn arithmetic_fusion("
fn arithmetic_fusion() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let one = g.constant(1.0, vec![])?;
	let two = g.constant(2.0, vec![])?;
	let a = g.binary(Binary::Add, x, one)?;
	let y = g.binary(Binary::Mul, a, two)?;
	let p = g.compile(&[y])?;
	assert_eq!(p.kernel_count, 1);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [4., 6., 8.]);
	let staged = g.stage(a)?;
	let z = g.binary(Binary::Mul, staged, two)?;
	assert_eq!(g.compile(&[z])?.kernel_count, 2);
	let three = g.constant(3., vec![])?;
	let z = g.binary(Binary::Add, a, three)?;
	assert_eq!(g.compile(&[y, z])?.kernel_count, 3);
	let result = compare(&g, &[y, z], &[buffer(&[3], &[1., 2., 3.])])?;
	assert_eq!(result[1].data, [5., 6., 7.]);
	Ok(())
}
```

```sh check
cargo test --quiet --test fusion_plan
```

## 31. dump で各 consumer が producer 完了後の state を読む。
<!-- todo:09-031 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

`tests/fusion_plan.rs` の `fn arithmetic_fusion(`の定義を次の全体で置き換えます。

```rust file="tests/fusion_plan.rs" mode="replace" item="fn arithmetic_fusion("
fn arithmetic_fusion() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let one = g.constant(1.0, vec![])?;
	let two = g.constant(2.0, vec![])?;
	let a = g.binary(Binary::Add, x, one)?;
	let y = g.binary(Binary::Mul, a, two)?;
	let p = g.compile(&[y])?;
	assert_eq!(p.kernel_count, 1);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [4., 6., 8.]);
	let staged = g.stage(a)?;
	let z = g.binary(Binary::Mul, staged, two)?;
	assert_eq!(g.compile(&[z])?.kernel_count, 2);
	let three = g.constant(3., vec![])?;
	let z = g.binary(Binary::Add, a, three)?;
	assert_eq!(g.compile(&[y, z])?.kernel_count, 3);
	let result = compare(&g, &[y, z], &[buffer(&[3], &[1., 2., 3.])])?;
	assert_eq!(result[1].data, [5., 6., 7.]);
	Ok(())
}
```

```sh check
cargo test --quiet --test fusion_plan
```

## 32. 純粋な元グラフが run 前後で同じ構造を保つ。
<!-- todo:09-032 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

`tests/fusion_plan.rs` の `fn arithmetic_fusion(`の定義を次の全体で置き換えます。

```rust file="tests/fusion_plan.rs" mode="replace" item="fn arithmetic_fusion("
fn arithmetic_fusion() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let one = g.constant(1.0, vec![])?;
	let two = g.constant(2.0, vec![])?;
	let a = g.binary(Binary::Add, x, one)?;
	let y = g.binary(Binary::Mul, a, two)?;
	let p = g.compile(&[y])?;
	assert_eq!(p.kernel_count, 1);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [4., 6., 8.]);
	let staged = g.stage(a)?;
	let z = g.binary(Binary::Mul, staged, two)?;
	assert_eq!(g.compile(&[z])?.kernel_count, 2);
	let three = g.constant(3., vec![])?;
	let z = g.binary(Binary::Add, a, three)?;
	assert_eq!(g.compile(&[y, z])?.kernel_count, 3);
	let result = compare(&g, &[y, z], &[buffer(&[3], &[1., 2., 3.])])?;
	assert_eq!(result[1].data, [5., 6., 7.]);
	Ok(())
}
```

```sh check
cargo test --quiet --test fusion_plan
```

## 章末の累積確認

前章までの example と test も同じ checkout で再ビルドする。

```sh check
cargo test --quiet --all-targets
```
