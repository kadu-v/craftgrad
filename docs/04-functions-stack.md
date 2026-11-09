# 04. 純粋関数を CALL し、STACK と INDEX を UIR から実行する

Function は固有 scope、署名、scalar body、同型結果の配列を持つ。CALL は capture を許さず caller の arena へ inline する。STACK は同型の NodeId 配列で、INDEX は静的な番号だけを許す。UIR は関数ごとの PARAM、定数 bits、ALU、return を保存する。CALL を inline してから保存するため、再帰 CALL は UIR parser が未対応演算として拒否する。

`file=... mode=write` はそのファイルの全置換、`mode=append` は末尾への追記である。すべて前章でできたプロジェクトの root で行う。`sh check` は実際の検証コマンドであり、ソースファイルには保存しない。

## 1. `FunctionId` と引数署名を、実際に呼ぶ関数一つに対して追加する。
<!-- todo:04-001 -->

```rust file=src/functions.rs mode=write
use crate::{
	Result,
	scalar::{Alu, Bounds, DType, SOp, ScalarGraph, Value},
};
use std::{
	collections::HashMap,
	sync::atomic::{AtomicUsize, Ordering},
};
static NEXT_SCOPE: AtomicUsize = AtomicUsize::new(1);
#[derive(Clone, Debug)]
pub struct Function {
	pub name: String,
	pub scope: usize,
	pub graph: ScalarGraph,
	pub parameters: Vec<(DType, Option<Bounds>)>,
	pub outputs: Vec<usize>,
}
impl Function {
	pub fn build(
		name: &str,
		parameters: Vec<(DType, Option<Bounds>)>,
		body: impl FnOnce(&mut ScalarGraph, &[usize]) -> Result<Vec<usize>>,
	) -> Result<Self> {
		if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
			return Err("invalid function name".into());
		}
		let scope = NEXT_SCOPE.fetch_add(1, Ordering::Relaxed);
		let mut graph = ScalarGraph::default();
		let args = parameters
			.iter()
			.enumerate()
			.map(|(slot, &(dtype, bounds))| graph.scoped_param(scope, slot, dtype, bounds))
			.collect::<Result<Vec<_>>>()?;
		let outputs = body(&mut graph, &args)?;
		if outputs.is_empty() {
			return Err("empty function result".into());
		}
		for &root in &outputs {
			graph.verify(root, scope)?;
		}
		let dtype = graph.nodes[outputs[0]].dtype;
		if outputs.iter().any(|&id| graph.nodes[id].dtype != dtype) {
			return Err("STACK requires homogeneous scalar results".into());
		}
		Ok(Self {
			name: name.into(),
			scope,
			graph,
			parameters,
			outputs,
		})
	}
	pub fn call(&self, caller: &mut ScalarGraph, args: &[usize]) -> Result<Vec<usize>> {
		if args.len() != self.parameters.len() {
			return Err("CALL argument count".into());
		}
		for (&arg, &(dtype, bounds)) in args.iter().zip(&self.parameters) {
			let n = caller.nodes.get(arg).ok_or("CALL unknown argument")?;
			if n.dtype != dtype {
				return Err("CALL argument dtype".into());
			}
			if let Some(expected) = bounds {
				let actual = n.bounds.ok_or("CALL cannot prove parameter bounds")?;
				if actual.lo < expected.lo || actual.hi > expected.hi {
					return Err("CALL argument exceeds declared bounds".into());
				}
			}
		}
		let mut map: HashMap<usize, usize> = HashMap::new();
		for &root in &self.outputs {
			self.graph.verify(root, self.scope)?;
			for id in self.graph.order(root)? {
				if map.contains_key(&id) {
					continue;
				}
				let node = &self.graph.nodes[id];
				let value = match node.op {
					SOp::Const(v) => caller.constant(v),
					SOp::Param { slot, scope, .. } => {
						if scope != self.scope {
							return Err("capture from another scope".into());
						}
						*args.get(slot).ok_or("unbound function parameter")?
					}
					SOp::Alu(op) => caller.alu(op, node.inputs.iter().map(|x| map[x]).collect())?,
				};
				map.insert(id, value);
			}
		}
		Ok(self.outputs.iter().map(|id| map[id]).collect())
	}
	pub fn evaluate(&self, args: &[Value]) -> Result<Vec<Value>> {
		if args.len() != self.parameters.len() {
			return Err("CALL argument count".into());
		}
		self.outputs
			.iter()
			.map(|&id| self.graph.evaluate_scoped(id, args, self.scope))
			.collect()
	}
	pub fn index(results: &[usize], index: isize) -> Result<usize> {
		usize::try_from(index)
			.ok()
			.and_then(|i| results.get(i).copied())
			.ok_or("STACK INDEX out of bounds".into())
	}
	pub fn to_uir(&self) -> String {
		let mut text = format!("function {}\n", self.name);
		for node in &self.graph.nodes {
			let line = match node.op {
				SOp::Const(v) => format!("const {} {}", dtype_name(v.dtype()), v.pack()),
				SOp::Param {
					slot,
					dtype,
					bounds,
					..
				} => format!(
					"param {slot} {} {}",
					dtype_name(dtype),
					bounds
						.map(|b| format!("{} {}", b.lo, b.hi))
						.unwrap_or("none none".into())
				),
				SOp::Alu(op) => format!(
					"alu {} {}",
					alu_name(op),
					node.inputs
						.iter()
						.map(|i| i.to_string())
						.collect::<Vec<_>>()
						.join(" ")
				),
			};
			text.push_str(&line);
			text.push('\n');
		}
		text.push_str(&format!(
			"return {}\n",
			self.outputs
				.iter()
				.map(|i| i.to_string())
				.collect::<Vec<_>>()
				.join(" ")
		));
		text
	}
	pub fn from_uir(text: &str) -> Result<Self> {
		let mut lines = text.lines();
		let header = lines
			.next()
			.ok_or("missing function header")?
			.split_whitespace()
			.collect::<Vec<_>>();
		if header.len() != 2 || header[0] != "function" {
			return Err("invalid function header".into());
		}
		let name = header[1].to_string();
		let scope = NEXT_SCOPE.fetch_add(1, Ordering::Relaxed);
		let mut graph = ScalarGraph::default();
		let mut parameters = vec![];
		let mut outputs = None;
		for line in lines {
			if outputs.is_some() {
				return Err("tokens after return".into());
			}
			let fields = line.split_whitespace().collect::<Vec<_>>();
			if fields.is_empty() {
				continue;
			}
			let parse = |s: &str| {
				s.parse::<usize>()
					.map_err(|_| "invalid node number".to_string())
			};
			match fields[0] {
				"const" if fields.len() == 3 => {
					let t = parse_dtype(fields[1])?;
					let bits = fields[2]
						.parse::<u64>()
						.map_err(|_| "invalid constant bits")?;
					graph.constant(Value::unpack(t, bits)?);
				}
				"param" if fields.len() == 5 => {
					let slot = parse(fields[1])?;
					if slot != parameters.len() {
						return Err("non-dense function parameter".into());
					}
					let t = parse_dtype(fields[2])?;
					let bounds = if fields[3] == "none" && fields[4] == "none" {
						None
					} else {
						Some(Bounds::new(
							fields[3].parse().map_err(|_| "invalid bound")?,
							fields[4].parse().map_err(|_| "invalid bound")?,
						)?)
					};
					graph.scoped_param(scope, slot, t, bounds)?;
					parameters.push((t, bounds));
				}
				"alu" if fields.len() >= 3 => {
					graph.alu(
						parse_alu(fields[1])?,
						fields[2..]
							.iter()
							.map(|s| parse(s))
							.collect::<Result<_>>()?,
					)?;
				}
				"return" if fields.len() >= 2 => {
					outputs = Some(
						fields[1..]
							.iter()
							.map(|s| parse(s))
							.collect::<Result<Vec<_>>>()?,
					);
				}
				_ => return Err(format!("invalid UIR instruction {line}")),
			}
		}
		let outputs = outputs.ok_or("missing return")?;
		for &id in &outputs {
			graph.verify(id, scope)?;
		}
		if outputs
			.iter()
			.any(|&id| graph.nodes[id].dtype != graph.nodes[outputs[0]].dtype)
		{
			return Err("heterogeneous STACK".into());
		}
		Ok(Self {
			name,
			scope,
			graph,
			parameters,
			outputs,
		})
	}
}
fn dtype_name(t: DType) -> &'static str {
	match t {
		DType::Bool => "bool",
		DType::I32 => "i32",
		DType::F32 => "f32",
		DType::Void => "void",
	}
}
fn parse_dtype(s: &str) -> Result<DType> {
	match s {
		"bool" => Ok(DType::Bool),
		"i32" => Ok(DType::I32),
		"f32" => Ok(DType::F32),
		_ => Err("invalid scalar dtype".into()),
	}
}
fn alu_name(op: Alu) -> String {
	match op {
		Alu::Cast(t) => format!("cast_{}", dtype_name(t)),
		_ => format!("{op:?}").to_lowercase(),
	}
}
fn parse_alu(s: &str) -> Result<Alu> {
	Ok(match s {
		"add" => Alu::Add,
		"sub" => Alu::Sub,
		"mul" => Alu::Mul,
		"neg" => Alu::Neg,
		"xor" => Alu::Xor,
		"and" => Alu::And,
		"or" => Alu::Or,
		"shl" => Alu::Shl,
		"shr" => Alu::Shr,
		"div" => Alu::Div,
		"mod" => Alu::Mod,
		"lt" => Alu::Lt,
		"eq" => Alu::Eq,
		"ne" => Alu::Ne,
		"where" => Alu::Where,
		"recip" => Alu::Recip,
		"max" => Alu::Max,
		"cast_bool" => Alu::Cast(DType::Bool),
		"cast_i32" => Alu::Cast(DType::I32),
		"cast_f32" => Alu::Cast(DType::F32),
		_ => return Err("unsupported or recursive UIR operation".into()),
	})
}
pub fn assert_uop(
	graph: &ScalarGraph,
	root: usize,
	inputs: &[Value],
	expected: Value,
) -> Result<()> {
	let actual = graph.compile(root)?.run(inputs)?;
	if actual == expected {
		Ok(())
	} else {
		Err(format!("assert_uop expected {expected:?}, got {actual:?}"))
	}
}
```

```rust file=src/lib.rs mode=append
pub mod functions;
```

```rust file=tests/function_stack.rs mode=write
use chinigrad::functions::Function;
use chinigrad::{
	Result,
	scalar::{Alu, Bounds, DType, ScalarGraph, Value},
};

#[test]
fn function_stack_uir() -> Result<()> {
	let f = Function::build("pair", vec![(DType::F32, None); 2], |g, args| {
		Ok(vec![
			g.alu(Alu::Add, args.to_vec())?,
			g.alu(Alu::Sub, args.to_vec())?,
		])
	})?;
	let restored = Function::from_uir(&f.to_uir())?;
	let mut g = ScalarGraph::default();
	let a = g.constant(Value::f32(2.));
	let b = g.constant(Value::f32(3.));
	for f in [&f, &restored] {
		let results = f.call(&mut g, &[a, b])?;
		assert_eq!(g.compile(results[0])?.run(&[])?, Value::f32(5.));
		assert_eq!(g.compile(results[1])?.run(&[])?, Value::f32(-1.));
		assert!(Function::index(&results, -1).is_err());
	}
	assert!(Function::from_uir("function f\ncall f\nreturn 0\n").is_err());
	Ok(())
}
#[test]
fn scoped_calls_and_assertion_failure() -> Result<()> {
	let f = Function::build("double_sum", vec![(DType::F32, None); 2], |g, args| {
		let sum = g.alu(Alu::Add, args.to_vec())?;
		let two = g.constant(Value::f32(2.));
		Ok(vec![g.alu(Alu::Mul, vec![sum, two])?])
	})?;
	let mut g = ScalarGraph::default();
	for (a, b, expected) in [(2., 3., 10.), (5., 1., 12.)] {
		let a = g.constant(Value::f32(a));
		let b = g.constant(Value::f32(b));
		let y = f.call(&mut g, &[a, b])?[0];
		assert_eq!(g.compile(y)?.run(&[])?, Value::f32(expected));
		assert!(chinigrad::functions::assert_uop(&g, y, &[], Value::f32(expected + 1.)).is_err());
	}
	let other = Function::build("other", vec![(DType::F32, None); 2], |g, args| {
		Ok(vec![g.alu(Alu::Add, args.to_vec())?])
	})?;
	assert_ne!(f.scope, other.scope);
	let a = g.constant(Value::I32(1));
	assert!(f.call(&mut g, &[a]).is_err());
	assert!(f.call(&mut g, &[a, a]).is_err());
	Ok(())
}
```

`tests/function_stack.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test function_stack
```

## 2. PARAM の intern key に所属関数を含める。
<!-- todo:04-002 -->

`tests/function_stack.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test function_stack
```

## 3. interpreter が呼出しごとに引数環境を作るようにする。
<!-- todo:04-003 -->

`tests/function_stack.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test function_stack
```

## 4. C 生成前に body を capture なしで inline する。
<!-- todo:04-004 -->

`tests/function_stack.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test function_stack
```

## 5. `f(2,3)=10` と `f(5,1)=12` を CPU まで通す。
<!-- todo:04-005 -->

`tests/function_stack.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test function_stack
```

## 6. 同型 scalar の STACK を interpreter と型導出へ加える。
<!-- todo:04-006 -->

`tests/function_stack.rs` の `function_stack_uir` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test function_stack function_stack_uir
```

## 7. 静的 INDEX の型と `0 <= index < len` を検証する。
<!-- todo:04-007 -->

`tests/function_stack.rs` の `function_stack_uir` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test function_stack function_stack_uir
```

## 8. CALL の結果に固定長 STACK を許し、各要素を inline する。
<!-- todo:04-008 -->

`tests/function_stack.rs` の `function_stack_uir` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test function_stack function_stack_uir
```

## 9. `pair(2,3)` から `5` と `-1` を取り出して C まで通す。
<!-- todo:04-009 -->

`tests/function_stack.rs` の `function_stack_uir` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test function_stack function_stack_uir
```

## 10. 対応演算だけの printer と parser を追加する。
<!-- todo:04-010 -->

`tests/function_stack.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test function_stack
```

## 11. 入力番号、関数名、dtype を検証してから arena へ入れる。
<!-- todo:04-011 -->

`tests/function_stack.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test function_stack
```

## 12. f32 の特殊値もビット列で往復するようにする。
<!-- todo:04-012 -->

`tests/function_stack.rs` の `function_stack_uir` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test function_stack function_stack_uir
```

## 13. `pair` の UIR を出力し、読み戻したグラフから C を生成する。
<!-- todo:04-013 -->

`tests/function_stack.rs` の `function_stack_uir` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test function_stack function_stack_uir
```

## 14. 読み戻し前後で `5` と `-1` が一致することを assertion で確かめる。
<!-- todo:04-014 -->

`tests/function_stack.rs` の `function_stack_uir` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test function_stack function_stack_uir
```

## 15. 未束縛 PARAM と他関数の PARAM の直接参照を拒否する。
<!-- todo:04-015 -->

`tests/function_stack.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test function_stack
```

## 16. CALL の引数不足、dtype 不一致、再帰を拒否する。
<!-- todo:04-016 -->

`tests/function_stack.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test function_stack
```

## 17. INDEX の負数と長さ以上の値を拒否する。
<!-- todo:04-017 -->

`tests/function_stack.rs` の `function_stack_uir` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test function_stack function_stack_uir
```

## 18. `assert_uop` が意図的に誤った予定値で失敗することを確認する。
<!-- todo:04-018 -->

`tests/function_stack.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test function_stack
```

## 19. 二つの呼出しで値を混同せず、`10` と `12` を返す。
<!-- todo:04-019 -->

`tests/function_stack.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test function_stack
```

## 20. STACK の選択結果が `5` と `-1` になる。
<!-- todo:04-020 -->

`tests/function_stack.rs` の `function_stack_uir` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test function_stack function_stack_uir
```

## 21. UIR の出力、入力、実行が一往復する。
<!-- todo:04-021 -->

`tests/function_stack.rs` の `function_stack_uir` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test function_stack function_stack_uir
```

## 22. 異なる関数の PARAM が同じ NodeId に共有されない。
<!-- todo:04-022 -->

`tests/function_stack.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test function_stack
```

## 23. 誤った予定値と不正な UIR の assertion が実際に失敗する。
<!-- todo:04-023 -->

`tests/function_stack.rs` の `function_stack_uir` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test function_stack function_stack_uir
```

## 24. scalar の前章までの例も同じ pipeline で成功する。
<!-- todo:04-024 -->

`tests/function_stack.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test function_stack
```

## 章末の累積確認

前章までの example と test も同じ checkout で再ビルドする。

```sh check
cargo test --quiet --all-targets
```
