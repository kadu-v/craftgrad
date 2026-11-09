# 20. 上限付きの shape を束縛して同じ kernel を動かす

19章までのプロジェクトへ、上限17の長さを持つ vector を追加します。同じ compiled Plan に長さ1、3、17を渡し、`x*2+1` と総和を返します。shape の仕様は元の `Graph` と別に保持し、一回の値を `Bindings` に保存します。新しい Tensor 型は作りません。

各 `file` ブロックを指定パスへ保存します。`mode="write"` は全体置換、`mode="append"` は末尾への追加です。確認コマンドは直前までのブロックを保存してから実行します。CPU のコードはmacOSとLinuxの64bit環境を対象にします。Metal、OpenCLの実機確認は条件付きです。

## 20-001 VarSpec と Bindings を追加し、未束縛、二重指定、範囲外を起動前に拒否する。

<!-- todo:20-001 -->

`VarId` は scope と番号を持ちます。同名の `N` を別 scope から渡すと `ShapeEnv::bind` が拒否します。次の3節で一つのモジュールを組み立て、4節で実行します。

```rust file="src/symbolic.rs" mode="write"
use crate::{
	Result,
	graph::{Graph, NodeId, Op},
	runtime::{Buffer, Plan, elements},
};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VarId {
	scope: u64,
	serial: u64,
}
#[derive(Clone, Debug)]
pub struct VarSpec {
	pub id: VarId,
	pub name: String,
	pub min: usize,
	pub max: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExprId {
	scope: u64,
	serial: u64,
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DimExpr {
	Const(usize),
	Var(VarId),
	Add(ExprId, ExprId),
	MulConst(ExprId, usize),
	DivConst(ExprId, usize),
	ModConst(ExprId, usize),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dim {
	Static(usize),
	Var(VarId),
	Expr(ExprId),
}
#[derive(Clone, Debug)]
pub struct Bindings(BTreeMap<VarId, usize>);
impl Bindings {
	pub fn get(&self, id: VarId) -> Result<usize> {
		self.0
			.get(&id)
			.copied()
			.ok_or_else(|| format!("unbound variable {id:?}"))
	}
}
#[derive(Clone, Debug)]
pub struct ShapeEnv {
	scope: u64,
	pub specs: Vec<VarSpec>,
	expressions: HashMap<ExprId, DimExpr>,
	intern: HashMap<DimExpr, ExprId>,
}
impl Default for ShapeEnv {
	fn default() -> Self {
		static NEXT: AtomicU64 = AtomicU64::new(1);
		Self {
			scope: NEXT.fetch_add(1, Ordering::Relaxed),
			specs: vec![],
			expressions: HashMap::new(),
			intern: HashMap::new(),
		}
	}
}
impl ShapeEnv {
	pub fn variable(&mut self, name: &str, min: usize, max: usize) -> Result<VarId> {
		if min > max || max > i32::MAX as usize {
			return Err("shape bounds exceed nonnegative i32".into());
		}
		static NEXT_VAR: AtomicU64 = AtomicU64::new(1);
		let id = VarId {
			scope: self.scope,
			serial: NEXT_VAR.fetch_add(1, Ordering::Relaxed),
		};
		self.specs.push(VarSpec {
			id,
			name: name.into(),
			min,
			max,
		});
		Ok(id)
	}
	pub fn bind(&self, pairs: &[(VarId, usize)]) -> Result<Bindings> {
		let mut values = BTreeMap::new();
		for &(id, value) in pairs {
			let spec = self
				.specs
				.iter()
				.find(|s| s.id == id)
				.ok_or("unknown shape variable")?;
			if value < spec.min || value > spec.max {
				return Err(format!(
					"{} out of bounds [{},{}]",
					spec.name, spec.min, spec.max
				));
			}
			if values.insert(id, value).is_some() {
				return Err("duplicate shape binding".into());
			}
		}
		if values.len() != self.specs.len() {
			return Err("unbound shape variable".into());
		}
		Ok(Bindings(values))
	}
	pub fn expression(&mut self, expression: DimExpr) -> Result<ExprId> {
		match expression {
			DimExpr::Var(id) => {
				if !self.specs.iter().any(|s| s.id == id) {
					return Err("unknown expression variable".into());
				}
			}
			DimExpr::Add(a, b) => {
				self.expression_at(a)?;
				self.expression_at(b)?;
			}
			DimExpr::MulConst(a, c) | DimExpr::DivConst(a, c) | DimExpr::ModConst(a, c) => {
				self.expression_at(a)?;
				if c == 0 {
					return Err("shape expression requires positive constant".into());
				}
			}
			DimExpr::Const(_) => {}
		}
		if let Some(&id) = self.intern.get(&expression) {
			return Ok(id);
		}
		static NEXT_EXPR: AtomicU64 = AtomicU64::new(1);
		let id = ExprId {
			scope: self.scope,
			serial: NEXT_EXPR.fetch_add(1, Ordering::Relaxed),
		};
		self.expressions.insert(id, expression.clone());
		self.intern.insert(expression, id);
		Ok(id)
	}
	fn expression_at(&self, id: ExprId) -> Result<&DimExpr> {
		if id.scope != self.scope {
			return Err("foreign shape expression".into());
		}
		self.expressions
			.get(&id)
			.ok_or_else(|| "unknown shape expression".into())
	}
	pub fn eval(&self, dimension: Dim, bindings: &Bindings) -> Result<usize> {
		let value = match dimension {
			Dim::Static(n) => n,
			Dim::Var(id) => bindings.get(id)?,
			Dim::Expr(id) => match *self.expression_at(id)? {
				DimExpr::Const(n) => n,
				DimExpr::Var(id) => bindings.get(id)?,
				DimExpr::Add(a, b) => self
					.eval(Dim::Expr(a), bindings)?
					.checked_add(self.eval(Dim::Expr(b), bindings)?)
					.ok_or("shape addition overflow")?,
				DimExpr::MulConst(a, c) => self
					.eval(Dim::Expr(a), bindings)?
					.checked_mul(c)
					.ok_or("shape multiplication overflow")?,
				DimExpr::DivConst(a, c) => self
					.eval(Dim::Expr(a), bindings)?
					.checked_div(c)
					.ok_or("shape division by zero")?,
				DimExpr::ModConst(a, c) => self
					.eval(Dim::Expr(a), bindings)?
					.checked_rem(c)
					.ok_or("shape remainder by zero")?,
			},
		};
		if value > i32::MAX as usize {
			return Err("dimension exceeds kernel i32 domain".into());
		}
		Ok(value)
	}
	fn dimensions(&self, dims: &[Dim], bindings: &Bindings) -> Result<Vec<usize>> {
		let shape: Vec<_> = dims
			.iter()
			.map(|&dim| self.eval(dim, bindings))
			.collect::<Result<_>>()?;
		elements(&shape)?;
		Ok(shape)
	}
}
```

```rust file="src/lib.rs" mode="append"
pub mod symbolic;
```

## 20-002 02と03の固定 ABI の scalar slot を使い、VarId から slot への対応を追加する。

<!-- todo:20-002 -->

`ShapeProgram` は元の `Graph` に shape 式を対応させます。`scalar_slots` は入力次元の scalar word の位置を返します。`prepare` は束縛を一度検証し、既存の builder と verifier で全ノードの論理 shape を導出します。物理領域では論理要素を先頭から密に配置します。複数軸の最大 shape の stride を保つ部分配列とは区別します。

```rust file="src/symbolic.rs" mode="append"
#[derive(Clone, Debug)]
pub enum Constraint {
	Equal(Dim, Dim),
	Divisible(Dim, usize),
}
#[derive(Clone)]
pub struct ShapeProgram {
	pub graph: Graph,
	pub outputs: Vec<NodeId>,
	pub env: ShapeEnv,
	pub input_dims: BTreeMap<usize, Vec<Dim>>,
	pub target_dims: BTreeMap<NodeId, Vec<Dim>>,
	pub constraints: Vec<Constraint>,
}
pub struct Snapshot {
	pub bindings: Bindings,
	pub graph: Graph,
	pub node_shapes: Vec<Vec<usize>>,
	pub logical_input_shapes: BTreeMap<usize, Vec<usize>>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScalarSlot(pub usize);
impl ShapeProgram {
	pub fn new(graph: Graph, outputs: Vec<NodeId>, env: ShapeEnv) -> Result<Self> {
		graph.verify(&outputs)?;
		Ok(Self {
			graph,
			outputs,
			env,
			input_dims: BTreeMap::new(),
			target_dims: BTreeMap::new(),
			constraints: vec![],
		})
	}
	pub fn input(&mut self, slot: usize, dims: Vec<Dim>) -> Result<()> {
		let shape = self
			.graph
			.nodes
			.iter()
			.find_map(|node| match node.op {
				Op::Input { slot: s } if s == slot => Some(&node.shape),
				_ => None,
			})
			.ok_or("symbolic input slot absent")?;
		if shape.len() != dims.len() {
			return Err("symbolic rank differs from graph".into());
		}
		self.input_dims.insert(slot, dims);
		Ok(())
	}
	pub fn scalar_slots(&self) -> BTreeMap<VarId, Vec<ScalarSlot>> {
		let mut result: BTreeMap<VarId, Vec<ScalarSlot>> = BTreeMap::new();
		let mut offset = 0;
		for node in &self.graph.nodes {
			if let Op::Input { slot } = node.op {
				if let Some(dimensions) = self.input_dims.get(&slot) {
					for (axis, dimension) in dimensions.iter().enumerate() {
						if let Dim::Var(id) = dimension {
							result
								.entry(*id)
								.or_default()
								.push(ScalarSlot(offset + axis));
						}
					}
				}
			}
			offset += node.shape.len();
		}
		result
	}
	pub fn target(&mut self, node: NodeId, dims: Vec<Dim>) -> Result<()> {
		let target = self.graph.nodes.get(node).ok_or("symbolic target absent")?;
		if !matches!(
			target.op,
			Op::Reshape { .. } | Op::Expand { .. } | Op::Const { .. }
		) {
			return Err("symbolic target must be reshape, expand, or constant".into());
		}
		if target.shape.len() != dims.len() {
			return Err("symbolic target rank".into());
		}
		self.target_dims.insert(node, dims);
		Ok(())
	}
	pub fn prepare(&self, inputs: &[Buffer], pairs: &[(VarId, usize)]) -> Result<Snapshot> {
		let bindings = self.env.bind(pairs)?;
		for constraint in &self.constraints {
			match *constraint {
				Constraint::Equal(a, b) => {
					if self.env.eval(a, &bindings)? != self.env.eval(b, &bindings)? {
						return Err("shape equality constraint".into());
					}
				}
				Constraint::Divisible(a, c) => {
					if c == 0 || self.env.eval(a, &bindings)? % c != 0 {
						return Err("shape divisibility constraint".into());
					}
				}
			}
		}
		let mut inferred = Graph::default();
		let mut mapping = Vec::with_capacity(self.graph.nodes.len());
		let mut actual = self.graph.clone();
		let mut logical_input_shapes = BTreeMap::new();
		for (id, node) in self.graph.nodes.iter().enumerate() {
			let target = if let Some(dims) = self.target_dims.get(&id) {
				self.env.dimensions(dims, &bindings)?
			} else {
				node.shape.clone()
			};
			let next = match &node.op {
				Op::Input { slot } => {
					let shape = if let Some(dims) = self.input_dims.get(slot) {
						self.env.dimensions(dims, &bindings)?
					} else {
						node.shape.clone()
					};
					{
						let buffer = inputs.get(*slot).ok_or("missing symbolic input")?;
						if shape.len() != node.shape.len()
							|| shape.iter().zip(&node.shape).any(|(n, max)| n > max)
						{
							return Err("logical shape exceeds compiled capacity".into());
						}
						if buffer.shape.len() != shape.len()
							|| elements(&shape)? > buffer.data.len()
						{
							return Err("insufficient symbolic buffer capacity".into());
						}
						if let Some(old) = logical_input_shapes.insert(*slot, shape.clone()) {
							if old != shape {
								return Err("one slot has inconsistent logical shapes".into());
							}
						}
					}
					inferred.input(*slot, shape)?
				}
				Op::Const { bits } => inferred.constant(f32::from_bits(*bits), target)?,
				Op::Unary { op, x } => inferred.unary(*op, mapping[*x])?,
				Op::Binary { op, a, b } => inferred.binary(*op, mapping[*a], mapping[*b])?,
				Op::Where { cond, yes, no } => {
					inferred.where_(mapping[*cond], mapping[*yes], mapping[*no])?
				}
				Op::Reduce {
					op,
					x,
					axes,
					keepdim,
				} => inferred.reduce(mapping[*x], *op, axes.clone(), *keepdim)?,
				Op::Reshape { x } => inferred.reshape(mapping[*x], target)?,
				Op::Permute { x, axes } => inferred.permute(mapping[*x], axes.clone())?,
				Op::Expand { x } => inferred.expand(mapping[*x], target)?,
				Op::Shrink { x, ranges } => inferred.shrink(mapping[*x], ranges.clone())?,
				Op::Pad { x, pads } => inferred.pad(mapping[*x], pads.clone())?,
				Op::Flip { x, axes } => inferred.flip(mapping[*x], axes.clone())?,
				Op::Stack { xs, axis } => {
					inferred.stack(xs.iter().map(|&x| mapping[x]).collect(), *axis)?
				}
				Op::Gather { x, indices, axis } => {
					inferred.gather(mapping[*x], mapping[*indices], *axis)?
				}
				Op::ScatterAdd {
					base,
					indices,
					updates,
					axis,
				} => inferred.scatter_add(
					mapping[*base],
					mapping[*indices],
					mapping[*updates],
					*axis,
				)?,
				Op::Stage { x } => inferred.stage(mapping[*x])?,
				Op::StopGradient { x } => inferred.stop_gradient(mapping[*x])?,
			};
			actual.nodes[id].shape = inferred.shape(next).to_vec();
			if actual.nodes[id].shape.len() != node.shape.len() {
				return Err("runtime rank changed".into());
			}
			mapping.push(next);
		}
		if logical_input_shapes.len() != inputs.len()
			|| (0..inputs.len()).any(|slot| !logical_input_shapes.contains_key(&slot))
		{
			return Err("symbolic input slots must be dense and exact".into());
		}
		actual.verify(&self.outputs)?;
		let node_shapes = actual.nodes.iter().map(|node| node.shape.clone()).collect();
		Ok(Snapshot {
			bindings,
			graph: actual,
			node_shapes,
			logical_input_shapes,
		})
	}
	pub fn compile(self) -> Result<SymbolicPlan> {
		let generic = self.graph.compile(&self.outputs)?;
		Ok(SymbolicPlan {
			program: self,
			generic,
			specialized: None,
		})
	}
}
```

## 20-003 `RANGE(N)` を C の実行時ループ上限へ出す。

<!-- todo:20-003 -->

`SymbolicPlan` は native library を所有します。`run_with_shapes` へ毎回の shape 表を渡し、08章の同じ二配列 ABI と C の `s[slot]` を使います。`run` 内で Clang を起動しません。特化を登録しない間は常に generic を選びます。

```rust file="src/symbolic.rs" mode="append"
struct Specialization {
	variable: VarId,
	value: usize,
	tile: usize,
	plan: Plan,
}
pub struct SymbolicPlan {
	program: ShapeProgram,
	generic: Plan,
	specialized: Option<Specialization>,
}
#[derive(Clone, Debug)]
pub struct RunRecord {
	pub variant: &'static str,
	pub bindings: Vec<(VarId, usize)>,
	pub logical_shapes: Vec<Vec<usize>>,
	pub capacities: Vec<usize>,
	pub scalar_dimensions: Vec<u64>,
}
impl SymbolicPlan {
	pub fn program(&self) -> &ShapeProgram {
		&self.program
	}
	pub fn generic(&self) -> &Plan {
		&self.generic
	}
	pub fn run(
		&self,
		inputs: &[Buffer],
		pairs: &[(VarId, usize)],
	) -> Result<(Vec<Buffer>, RunRecord)> {
		let snapshot = self.program.prepare(inputs, pairs)?;
		let selected = self
			.specialized
			.as_ref()
			.filter(|s| snapshot.bindings.0.get(&s.variable) == Some(&s.value));
		let (plan, variant) = selected
			.map(|s| (&s.plan, "specialized"))
			.unwrap_or((&self.generic, "generic"));
		let output = plan.run_with_shapes(inputs, &snapshot.node_shapes)?;
		let scalar_dimensions = snapshot
			.node_shapes
			.iter()
			.flatten()
			.map(|&d| d as u64)
			.collect();
		let record = RunRecord {
			variant,
			bindings: snapshot.bindings.0.into_iter().collect(),
			logical_shapes: output.iter().map(|b| b.shape.clone()).collect(),
			capacities: inputs.iter().map(|b| b.data.len()).collect(),
			scalar_dimensions,
		};
		Ok((output, record))
	}
}
```

## 20-004 容量17の buffer に N=1、3、17を渡し、毎回異なる論理長の結果を返す。

<!-- todo:20-004 -->

`generic_reuses_binary_for_three_lengths_and_ignores_nan_tail` は生成 C を実行します。容量17を維持し、論理末尾以降を NaN にして長さ3、1、17を同じ Plan へ渡します。

```rust file="tests/symbolic.rs" mode="write"
use chinigrad::symbolic::{Constraint, Dim, DimExpr, ShapeEnv, ShapeProgram};
use chinigrad::{Binary, Buffer, Graph, Reduce};

fn vector() -> (
	chinigrad::symbolic::SymbolicPlan,
	chinigrad::symbolic::VarId,
) {
	let mut env = ShapeEnv::default();
	let n = env.variable("N", 1, 17).unwrap();
	let mut graph = Graph::default();
	let x = graph.input(0, vec![17]).unwrap();
	let two = graph.constant(2., vec![]).unwrap();
	let one = graph.constant(1., vec![]).unwrap();
	let mul = graph.binary(Binary::Mul, x, two).unwrap();
	let y = graph.binary(Binary::Add, mul, one).unwrap();
	let sum = graph.reduce(x, Reduce::Sum, vec![0], false).unwrap();
	let mut program = ShapeProgram::new(graph, vec![y, sum], env).unwrap();
	program.input(0, vec![Dim::Var(n)]).unwrap();
	(program.compile().unwrap(), n)
}

#[test]
fn generic_reuses_binary_for_three_lengths_and_ignores_nan_tail() {
	let (plan, n) = vector();
	let source = plan.generic().source.clone();
	let original = format!("{:?}", plan.program().graph);
	for len in [3, 1, 17, 3] {
		let mut data = vec![f32::NAN; 17];
		for (i, x) in data[..len].iter_mut().enumerate() {
			*x = (i + 1) as f32;
		}
		let input = Buffer::new(vec![17], data).unwrap();
		let (out, record) = plan.run(&[input], &[(n, len)]).unwrap();
		assert_eq!(out[0].shape, vec![len]);
		assert_eq!(
			out[0].data,
			(1..=len).map(|x| (2 * x + 1) as f32).collect::<Vec<_>>()
		);
		assert_eq!(out[1].data, vec![(len * (len + 1) / 2) as f32]);
		assert_eq!(record.capacities, vec![17]);
		assert_eq!(record.variant, "generic");
		assert_eq!(plan.generic().source, source);
		assert_eq!(format!("{:?}", plan.program().graph), original);
	}
	assert!(source.contains("s["));
}
```

```sh check
cargo test --test symbolic generic_reuses_binary_for_three_lengths_and_ignores_nan_tail
```

## 20-005 純粋な元グラフを置換せず、束縛値を runtime の実行記録に置く。

<!-- todo:20-005 -->

直前のテストは実行前の graph の文字列表現と source を保存し、全実行後も一致することを検査しています。束縛値は `RunRecord` にだけ残ります。

```sh check
cargo test --test symbolic generic_reuses_binary_for_three_lengths_and_ignores_nan_tail
```

## 20-006 `Dim::Expr(NodeId)` を必要な次元式だけに追加し、`2*N` を式のまま運ぶ。

<!-- todo:20-006 -->

shape 式は Tensor の `NodeId` と別型の `ExprId` で intern します。1節の `ShapeEnv::expression` と `eval` をこのテストで確認します。0による除算と overflow はエラーです。

```rust file="tests/symbolic.rs" mode="append"
#[test]
fn dimensions_are_checked_and_expression_ids_do_not_alias_nodes() {
	let mut env = ShapeEnv::default();
	let n = env.variable("N", 1, 17).unwrap();
	let var = env.expression(DimExpr::Var(n)).unwrap();
	let twice = env.expression(DimExpr::MulConst(var, 2)).unwrap();
	assert_eq!(twice, env.expression(DimExpr::MulConst(var, 2)).unwrap());
	let b = env.bind(&[(n, 3)]).unwrap();
	assert_eq!(env.eval(Dim::Expr(twice), &b).unwrap(), 6);
	assert!(env.expression(DimExpr::DivConst(var, 0)).is_err());
	let huge = env.expression(DimExpr::MulConst(var, usize::MAX)).unwrap();
	assert!(env.eval(Dim::Expr(huge), &b).is_err());
}
```

```rust file="tests/symbolic.rs" mode="append"
#[test]
fn expression_and_variable_handles_reject_foreign_and_diverged_clones() {
	let mut a = ShapeEnv::default();
	let three = a.expression(DimExpr::Const(3)).unwrap();
	let mut b = ShapeEnv::default();
	let seventeen = b.expression(DimExpr::Const(17)).unwrap();
	let bindings = b.bind(&[]).unwrap();
	assert_eq!(b.eval(Dim::Expr(seventeen), &bindings).unwrap(), 17);
	assert!(b.eval(Dim::Expr(three), &bindings).is_err());
	assert!(b.expression(DimExpr::Add(three, seventeen)).is_err());
	let mut clone = a.clone();
	let a_new = a.expression(DimExpr::Const(5)).unwrap();
	let b_new = clone.expression(DimExpr::Const(7)).unwrap();
	assert_ne!(a_new, b_new);
	assert!(
		clone
			.eval(Dim::Expr(a_new), &clone.bind(&[]).unwrap())
			.is_err()
	);
	let a_var = a.variable("N", 1, 17).unwrap();
	let b_var = clone.variable("N", 1, 17).unwrap();
	assert_ne!(a_var, b_var);
	assert!(clone.bind(&[(a_var, 3)]).is_err());
}
```

```sh check
cargo test --test symbolic dimensions_are_checked_and_expression_ids_do_not_alias_nodes
```

```sh check
cargo test --test symbolic expression_and_variable_handles_reject_foreign_and_diverged_clones
```

## 20-007 `sum(x)` の縮約 RANGE に N を使い、初期値を実行ごとに設定する。

<!-- todo:20-007 -->

4節の graph は `sum(x)` も出力します。テストは長さを変えた各実行で期待総和を比較し、前回の累積値が残らないことを検査します。

```sh check
cargo test --test symbolic generic_reuses_binary_for_three_lengths_and_ignores_nan_tail
```

## 20-008 `reshape([N,2])` の必要要素数と入力の論理要素数を実行時に検証する。

<!-- todo:20-008 -->

`target` で reshape の出力次元を `[N,2]` に指定します。`prepare` が同じ `Graph::reshape` を使うため、要素数が違えば native 呼出し前に失敗します。

```rust file="tests/symbolic.rs" mode="append"
#[test]
fn dynamic_reshape_rows_and_movement_match_static_graph() {
	let mut env = ShapeEnv::default();
	let n = env.variable("N", 1, 3).unwrap();
	let var = env.expression(DimExpr::Var(n)).unwrap();
	let twice = env.expression(DimExpr::MulConst(var, 2)).unwrap();
	let mut graph = Graph::default();
	let x = graph.input(0, vec![6]).unwrap();
	let rows = graph.reshape(x, vec![3, 2]).unwrap();
	let transposed = graph.permute(rows, vec![1, 0]).unwrap();
	let sum = graph.reduce(rows, Reduce::Sum, vec![1], false).unwrap();
	let padded = graph.pad(rows, vec![(1, 1), (0, 0)]).unwrap();
	let mut program = ShapeProgram::new(graph, vec![sum, transposed, padded], env).unwrap();
	program.input(0, vec![Dim::Expr(twice)]).unwrap();
	program
		.target(rows, vec![Dim::Var(n), Dim::Static(2)])
		.unwrap();
	let plan = program.compile().unwrap();
	let input = Buffer::new(vec![6], vec![1., 2., 3., 4., 5., 6.]).unwrap();
	for len in [3, 1, 2] {
		let (out, _) = plan.run(&[input.clone()], &[(n, len)]).unwrap();
		assert_eq!(
			out[0].data,
			(0..len).map(|r| (4 * r + 3) as f32).collect::<Vec<_>>()
		);
		let snapshot = plan
			.program()
			.prepare(&[input.clone()], &[(n, len)])
			.unwrap();
		let static_plan = snapshot.graph.compile(&plan.program().outputs).unwrap();
		let trimmed = Buffer::new(vec![len * 2], input.data[..len * 2].to_vec()).unwrap();
		let expected = static_plan.run(&[trimmed]).unwrap();
		assert_eq!(out, expected);
	}
	let short = Buffer::new(vec![5], vec![1.; 5]).unwrap();
	assert!(
		plan.run(&[short], &[(n, 3)])
			.unwrap_err()
			.contains("capacity")
	);
}
```

```sh check
cargo test --test symbolic dynamic_reshape_rows_and_movement_match_static_graph
```

## 20-009 同じ変数を使った broadcast は許可し、別変数の等値は Bindings で確認する。

<!-- todo:20-009 -->

別の `VarId` を持つ二つの入力に等値制約を追加します。名前が同じでも値を自動で同一視しません。制約を使わない通常の broadcast は既存 builder が検証します。

```rust file="tests/symbolic.rs" mode="append"
#[test]
fn broadcast_equality_constraints_fail_before_native_run() {
	let mut env = ShapeEnv::default();
	let n = env.variable("N", 1, 3).unwrap();
	let m = env.variable("N", 1, 3).unwrap();
	let mut graph = Graph::default();
	let a = graph.input(0, vec![3]).unwrap();
	let b = graph.input(1, vec![3]).unwrap();
	let y = graph.binary(Binary::Add, a, b).unwrap();
	let mut program = ShapeProgram::new(graph, vec![y], env).unwrap();
	program.input(0, vec![Dim::Var(n)]).unwrap();
	program.input(1, vec![Dim::Var(m)]).unwrap();
	program
		.constraints
		.push(Constraint::Equal(Dim::Var(n), Dim::Var(m)));
	let inputs = [
		Buffer::new(vec![3], vec![1.; 3]).unwrap(),
		Buffer::new(vec![3], vec![2.; 3]).unwrap(),
	];
	let plan = program.compile().unwrap();
	assert!(
		plan.run(&inputs, &[(n, 3), (m, 2)])
			.unwrap_err()
			.contains("equality")
	);
	assert_eq!(
		plan.run(&inputs, &[(n, 2), (m, 2)]).unwrap().0[0].data,
		vec![3., 3.]
	);
}
```

```sh check
cargo test --test symbolic broadcast_equality_constraints_fail_before_native_run
```

## 20-010 動的 shrink と pad の有効範囲を guard として INDEX に渡す。

<!-- todo:20-010 -->

shrink の開始と終了、pad 幅は graph の定数です。入力の長さが変わるたびに、shrink が現在の論理範囲内か検査します。pad の C は既存の有効条件を通ってから LOAD します。開始位置そのものを実行時変数にする形は、この版では受け付けません。

```rust file="tests/symbolic.rs" mode="append"
#[test]
fn dynamic_shrink_rejects_outside_logical_range() {
	let mut env = ShapeEnv::default();
	let n = env.variable("N", 1, 4).unwrap();
	let mut graph = Graph::default();
	let x = graph.input(0, vec![4]).unwrap();
	let shrunk = graph.shrink(x, vec![(1, 3)]).unwrap();
	let pad = graph.pad(shrunk, vec![(1, 1)]).unwrap();
	let mut program = ShapeProgram::new(graph, vec![pad], env).unwrap();
	program.input(0, vec![Dim::Var(n)]).unwrap();
	let plan = program.compile().unwrap();
	let input = Buffer::new(vec![4], vec![10., 20., 30., f32::NAN]).unwrap();
	assert_eq!(
		plan.run(&[input.clone()], &[(n, 3)]).unwrap().0[0].data,
		vec![0., 20., 30., 0.]
	);
	assert!(plan.run(&[input], &[(n, 2)]).is_err());
}
```

```sh check
cargo test --test symbolic dynamic_shrink_rejects_outside_logical_range
```

## 20-011 確定できないアクセスを安全とみなさず、guard を生成するか未対応として拒否する。

<!-- todo:20-011 -->

gather の添字検査も論理長に限定します。元の領域の NaN tail を読まず、今回の有効添字に NaN が入れば拒否することを確かめます。

```rust file="tests/symbolic.rs" mode="append"
#[test]
fn gather_validates_only_the_logical_index_prefix() {
	let mut env = ShapeEnv::default();
	let n = env.variable("N", 1, 4).unwrap();
	let mut graph = Graph::default();
	let weights = graph.input(0, vec![3]).unwrap();
	let ids = graph.input(1, vec![4]).unwrap();
	let gathered = graph.gather(weights, ids, 0).unwrap();
	let mut program = ShapeProgram::new(graph, vec![gathered], env).unwrap();
	program.input(1, vec![Dim::Var(n)]).unwrap();
	let plan = program.compile().unwrap();
	let input = [
		Buffer::new(vec![3], vec![10., 20., 30.]).unwrap(),
		Buffer::new(vec![4], vec![2., 0., f32::NAN, f32::NAN]).unwrap(),
	];
	assert_eq!(
		plan.run(&input, &[(n, 2)]).unwrap().0[0].data,
		vec![30., 10.]
	);
	assert!(plan.run(&input, &[(n, 3)]).is_err());
}
```

```sh check
cargo test --test symbolic gather_validates_only_the_logical_index_prefix
```

## 20-012 Metal の引数領域に N を置き、論理長から group 数を決める。

<!-- todo:20-012 -->

GPU も同じ `ShapeProgram::prepare` を使います。次のモジュールは15、16章の `GpuPlan` に検証済みの shape 表を渡します。Metal の scalar buffer と group 数はこの表から作られます。

```rust file="src/symbolic_gpu.rs" mode="write"
use crate::{
	Buffer, Result,
	gpu_graph::{Backend, GpuPlan},
	symbolic::{RunRecord, ShapeProgram, VarId},
};

pub struct SymbolicGpuPlan {
	program: ShapeProgram,
	gpu: GpuPlan,
}
impl ShapeProgram {
	pub fn compile_gpu(self, backend: Backend, local: usize) -> Result<SymbolicGpuPlan> {
		let gpu = GpuPlan::compile(&self.graph, &self.outputs, backend, local)?;
		Ok(SymbolicGpuPlan { program: self, gpu })
	}
}
impl SymbolicGpuPlan {
	pub fn program(&self) -> &ShapeProgram {
		&self.program
	}
	pub fn sources(&self) -> &[String] {
		&self.gpu.sources
	}
	pub fn run(
		&self,
		inputs: &[Buffer],
		pairs: &[(VarId, usize)],
	) -> Result<(Vec<Buffer>, RunRecord)> {
		let snapshot = self.program.prepare(inputs, pairs)?;
		let output = self.gpu.run_with_shapes(inputs, &snapshot.node_shapes)?;
		let record = RunRecord {
			variant: "generic",
			bindings: pairs.to_vec(),
			logical_shapes: output.iter().map(|buffer| buffer.shape.clone()).collect(),
			capacities: inputs.iter().map(|buffer| buffer.data.len()).collect(),
			scalar_dimensions: snapshot
				.node_shapes
				.iter()
				.flatten()
				.map(|&n| n as u64)
				.collect(),
		};
		Ok((output, record))
	}
}
```

```rust file="src/lib.rs" mode="append"
pub mod symbolic_gpu;
```

```sh check
cargo check --all-features
```

## 20-013 `i<N` を LOAD と STORE の両方へ伝え、容量だけを guard に使わない。

<!-- todo:20-013 -->

このテストファイルを追加します。CPU だけで生成 source の条件分岐を確認できます。GPU の4テストはデバイスが必要なので `--ignored` で明示的に実行します。未実行を成功として数えません。

```rust file="tests/symbolic_gpu.rs" mode="write"
use chinigrad::{
	Binary, Buffer, Graph, Reduce,
	gpu::{Dialect, Kernel},
	gpu_graph::Backend,
	gpu_kernels::dynamic_sum,
	symbolic::{Dim, ShapeEnv, ShapeProgram},
};

fn program() -> (ShapeProgram, chinigrad::symbolic::VarId) {
	let mut env = ShapeEnv::default();
	let n = env.variable("N", 1, 17).unwrap();
	let mut graph = Graph::default();
	let x = graph.input(0, vec![17]).unwrap();
	let two = graph.constant(2., vec![]).unwrap();
	let y = graph.binary(Binary::Mul, x, two).unwrap();
	let sum = graph.reduce(x, Reduce::Sum, vec![0], false).unwrap();
	let mut program = ShapeProgram::new(graph, vec![y, sum], env).unwrap();
	program.input(0, vec![Dim::Var(n)]).unwrap();
	(program, n)
}
fn verify_graph(backend: Backend) {
	let (program, n) = program();
	let plan = program.compile_gpu(backend, 8).unwrap();
	let original_sources = plan.sources().to_vec();
	for length in [17, 3, 1] {
		let mut input = vec![f32::NAN; 17];
		for (index, value) in input[..length].iter_mut().enumerate() {
			*value = (index + 1) as f32;
		}
		let (out, record) = plan
			.run(&[Buffer::new(vec![17], input).unwrap()], &[(n, length)])
			.unwrap();
		assert_eq!(
			out[0].data,
			(1..=length).map(|x| 2. * x as f32).collect::<Vec<_>>()
		);
		assert_eq!(out[1].data, vec![(length * (length + 1) / 2) as f32]);
		assert_eq!(record.capacities, vec![17]);
		assert_eq!(plan.sources(), original_sources);
	}
}

#[test]
fn dynamic_reduction_source_guards_before_load_and_shares_snapshot() {
	for dialect in [Dialect::Metal, Dialect::OpenCl] {
		let reference = dynamic_sum(dialect, 17, 17, 8).unwrap();
		for n in [1, 3, 17] {
			let plan = dynamic_sum(dialect, n, 17, 8).unwrap();
			assert_eq!(plan.first.scalars, vec![n as u64]);
			assert_eq!(plan.first.source, reference.first.source);
			assert!(
				plan.first
					.source
					.contains("if(g*32+c<s[0]){acc+=b0[g*32+c];}")
			);
			assert_eq!(plan.first.local, 8);
			assert!(plan.first.source.contains("tmp[l]=acc;"));
			assert!(!plan.first.source.contains("if(l<s[0])"));
		}
	}
	assert!(dynamic_sum(Dialect::OpenCl, 18, 17, 8).is_err());
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
#[ignore = "requires a Metal device"]
fn metal_uses_one_compilation_for_three_shapes() {
	verify_graph(Backend::Metal);
}

#[cfg(feature = "opencl")]
#[test]
#[ignore = "requires an OpenCL device"]
fn opencl_uses_one_compilation_for_three_shapes() {
	verify_graph(Backend::OpenCl);
}

fn reduction_buffers(kernel: &Kernel, input: Vec<f32>) -> Vec<Vec<f32>> {
	vec![input, vec![f32::NAN; kernel.lengths[kernel.output]]]
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
#[ignore = "requires a Metal device"]
fn metal_local_sum_ignores_nan_tail() {
	use chinigrad::backends::metal::MetalProgram;
	let reference = dynamic_sum(Dialect::Metal, 17, 17, 8).unwrap();
	let first = MetalProgram::compile(&reference.first).unwrap();
	let second = MetalProgram::compile(&reference.second).unwrap();
	for n in [1, 3, 17] {
		let kernels = dynamic_sum(Dialect::Metal, n, 17, 8).unwrap();
		let mut input = vec![f32::NAN; 17];
		input[..n].fill(1.);
		let partial = first
			.run(&kernels.first, &reduction_buffers(&kernels.first, input))
			.unwrap()
			.output;
		let out = second
			.run(
				&kernels.second,
				&reduction_buffers(&kernels.second, partial),
			)
			.unwrap()
			.output;
		assert_eq!(out, vec![n as f32]);
	}
}

#[cfg(feature = "opencl")]
#[test]
#[ignore = "requires an OpenCL device"]
fn opencl_local_sum_ignores_nan_tail() {
	use chinigrad::backends::opencl::OpenClProgram;
	let reference = dynamic_sum(Dialect::OpenCl, 17, 17, 8).unwrap();
	let first = OpenClProgram::compile(&reference.first, 0, 0).unwrap();
	let second = OpenClProgram::compile(&reference.second, 0, 0).unwrap();
	for n in [1, 3, 17] {
		let kernels = dynamic_sum(Dialect::OpenCl, n, 17, 8).unwrap();
		let mut input = vec![f32::NAN; 17];
		input[..n].fill(1.);
		let partial = first
			.run(&kernels.first, &reduction_buffers(&kernels.first, input))
			.unwrap()
			.output;
		let out = second
			.run(
				&kernels.second,
				&reduction_buffers(&kernels.second, partial),
			)
			.unwrap()
			.output;
		assert_eq!(out, vec![n as f32]);
	}
}
```

```sh check
cargo test --test symbolic_gpu dynamic_reduction_source_guards_before_load_and_shares_snapshot
```

## 20-014 local reduction では無効 lane に0を与え、全 lane を barrier に到達させる。

<!-- todo:20-014 -->

18章の `dynamic_sum` は無効 lane の `acc=0` を local 領域に書き、全 lane で barrier を通ります。Metal のテストは2段の program を一度 compile して、N=1、3、17を再実行します。

```sh check requires=metal
cargo test --features metal --test symbolic_gpu metal_local_sum_ignores_nan_tail -- --ignored
```

## 20-015 OpenCL の scalar 引数と NDRange を同じ束縛値で設定する。

<!-- todo:20-015 -->

OpenCL は同じ snapshot の scalar words を `__constant ulong *s` に束縛します。NDRange も同じ論理 shape から計算します。

```sh check requires=opencl
cargo test --features opencl --test symbolic_gpu opencl_uses_one_compilation_for_three_shapes -- --ignored
```

```sh check requires=opencl
cargo test --features opencl --test symbolic_gpu opencl_local_sum_ignores_nan_tail -- --ignored
```

## 20-016 GPU に送る値と host で検証した値が同じになるよう、一回の Bindings を固定する。

<!-- todo:20-016 -->

`Bindings` の map は公開せず、検証済みの snapshot から引数を作ります。`scalar_slots`、実行記録の word、論理 shape、容量を照合します。

```rust file="tests/symbolic.rs" mode="append"
#[test]
fn scalar_slots_and_metadata_use_the_same_validated_value() {
	let (plan, n) = vector();
	assert_eq!(
		plan.program().scalar_slots()[&n],
		vec![chinigrad::symbolic::ScalarSlot(0)]
	);
	let input = Buffer::new(vec![17], vec![1.; 17]).unwrap();
	let (_, record) = plan.run(&[input], &[(n, 3)]).unwrap();
	assert_eq!(record.scalar_dimensions[0], 3);
	assert_eq!(record.logical_shapes[0], vec![3]);
	assert_eq!(record.capacities, vec![17]);
}
```

```sh check
cargo test --test symbolic scalar_slots_and_metadata_use_the_same_validated_value
```

## 20-017 generic kernel の key に VarId の正規化情報、上下限、制約、ABI を含める。

<!-- todo:20-017 -->

generic の key は今回の束縛値を受け取りません。変数を shape 式内の初出順で正規化し、範囲、制約、graph、出力、ABI、backend、compile options を含めます。名前と生の VarId は含めません。

```rust file="src/symbolic.rs" mode="append"
impl ShapeProgram {
	pub fn generic_key(&self, backend: &str, options: &str) -> Result<Vec<u8>> {
		let mut ids = BTreeMap::new();
		let mut signature = format!("abi=2;dtype=f32;backend={backend:?};options={options:?};");
		fn variable(env: &ShapeEnv, id: VarId, ids: &mut BTreeMap<VarId, usize>) -> Result<String> {
			let spec = env
				.specs
				.iter()
				.find(|spec| spec.id == id)
				.ok_or("unknown key variable")?;
			let next = ids.len();
			let normalized = *ids.entry(id).or_insert(next);
			Ok(format!("v{normalized}[{},{}]", spec.min, spec.max))
		}
		fn dim(env: &ShapeEnv, d: Dim, ids: &mut BTreeMap<VarId, usize>) -> Result<String> {
			Ok(match d {
				Dim::Static(n) => format!("{n}"),
				Dim::Var(id) => variable(env, id, ids)?,
				Dim::Expr(id) => match *env.expression_at(id)? {
					DimExpr::Const(n) => format!("{n}"),
					DimExpr::Var(id) => variable(env, id, ids)?,
					DimExpr::Add(a, b) => format!(
						"({}+{})",
						dim(env, Dim::Expr(a), ids)?,
						dim(env, Dim::Expr(b), ids)?
					),
					DimExpr::MulConst(a, c) => format!("({}*{c})", dim(env, Dim::Expr(a), ids)?),
					DimExpr::DivConst(a, c) => format!("({}/{c})", dim(env, Dim::Expr(a), ids)?),
					DimExpr::ModConst(a, c) => format!("({}%{c})", dim(env, Dim::Expr(a), ids)?),
				},
			})
		}
		for (slot, dims) in &self.input_dims {
			signature.push_str(&format!("input{slot}="));
			for &d in dims {
				signature.push_str(&dim(&self.env, d, &mut ids)?);
				signature.push(',');
			}
		}
		for (node, dims) in &self.target_dims {
			signature.push_str(&format!("target{node}="));
			for &d in dims {
				signature.push_str(&dim(&self.env, d, &mut ids)?);
				signature.push(',');
			}
		}
		for constraint in &self.constraints {
			signature.push_str(&match *constraint {
				Constraint::Equal(a, b) => format!(
					"eq({},{})",
					dim(&self.env, a, &mut ids)?,
					dim(&self.env, b, &mut ids)?
				),
				Constraint::Divisible(a, c) => {
					format!("divisible({},{c})", dim(&self.env, a, &mut ids)?)
				}
			});
		}
		signature.push_str(&format!(
			"graph={:?};outputs={:?}",
			self.graph.nodes, self.outputs
		));
		Ok(signature.into_bytes())
	}
}
```

```rust file="tests/symbolic.rs" mode="append"
#[test]
fn canonical_keys_preserve_repeated_variables_and_bounds() {
	let (a, _) = vector();
	let (b, _) = vector();
	assert_eq!(
		a.program().generic_key("cpu", "").unwrap(),
		b.program().generic_key("cpu", "").unwrap()
	);
	assert_ne!(
		a.program().generic_key("cpu", "").unwrap(),
		b.program().generic_key("metal", "").unwrap()
	);
	let mut changed = b.program().clone();
	changed.env.specs[0].max = 16;
	assert_ne!(
		a.program().generic_key("cpu", "").unwrap(),
		changed.generic_key("cpu", "").unwrap()
	);
}
```

```sh check
cargo test --test symbolic canonical_keys_preserve_repeated_variables_and_bounds
```

## 20-018 generic kernel の key に毎回の N の値を混ぜず、1、3、17で同じコードを再利用する。

<!-- todo:20-018 -->

`generic_key` と source は Plan の作成時に固定されます。各実行では同じ library のまま shape word を変更します。

```sh check
cargo test --test symbolic generic_reuses_binary_for_three_lengths_and_ignores_nan_tail
```

```sh check
cargo test --test symbolic canonical_keys_preserve_repeated_variables_and_bounds
```

## 20-019 `N=16` を条件とする特化版を一つ追加し、条件不成立なら generic に戻す。

<!-- todo:20-019 -->

`add_specialization` は一つの shape 変数と tile 条件を検証し、定数 shape を持つ別の native library を一度作ります。`run` は通常の境界検査後だけ predicate を評価します。

```rust file="src/symbolic.rs" mode="append"
impl SymbolicPlan {
	pub fn add_specialization(
		&mut self,
		variable: VarId,
		value: usize,
		tile: usize,
		inputs: &[Buffer],
		pairs: &[(VarId, usize)],
	) -> Result<()> {
		if tile == 0 || value % tile != 0 {
			return Err("specialization tile does not divide value".into());
		}
		if self.program.env.specs.len() != 1 {
			return Err("this specialization supports one shape variable".into());
		}
		let snapshot = self.program.prepare(inputs, pairs)?;
		if snapshot.bindings.get(variable)? != value {
			return Err("specialization binding differs".into());
		}
		let plan = Plan::compile_specialized(&snapshot.graph, &self.program.outputs)?;
		self.specialized = Some(Specialization {
			variable,
			value,
			tile,
			plan,
		});
		Ok(())
	}
}
```

```rust file="src/symbolic.rs" mode="append"
impl SymbolicPlan {
	pub fn specialization_key(&self, backend: &str, options: &str) -> Result<Option<Vec<u8>>> {
		self.specialized
			.as_ref()
			.map(|s| {
				let mut key = self.program.generic_key(backend, options)?;
				key.extend_from_slice(
					format!(";specialized={};tile={}", s.value, s.tile).as_bytes(),
				);
				Ok(key)
			})
			.transpose()
	}
}
```

```rust file="tests/symbolic.rs" mode="append"
#[test]
fn specialization_is_distinct_and_selected_only_at_sixteen() {
	let (mut plan, n) = vector();
	let input = Buffer::new(vec![17], (1..=17).map(|x| x as f32).collect()).unwrap();
	let generic_key = plan.program().generic_key("cpu", "tile=1").unwrap();
	plan.add_specialization(n, 16, 8, &[input.clone()], &[(n, 16)])
		.unwrap();
	let specialized_key = plan.specialization_key("cpu", "tile=1").unwrap().unwrap();
	assert_ne!(specialized_key, generic_key);
	for (length, variant) in [(16, "specialized"), (17, "generic"), (3, "generic")] {
		let (out, record) = plan.run(&[input.clone()], &[(n, length)]).unwrap();
		assert_eq!(record.variant, variant);
		assert_eq!(out[0].data.last(), Some(&((2 * length + 1) as f32)));
		let snapshot = plan
			.program()
			.prepare(&[input.clone()], &[(n, length)])
			.unwrap();
		let expected = plan
			.generic()
			.run_with_shapes(&[input.clone()], &snapshot.node_shapes)
			.unwrap();
		assert_eq!(out, expected);
	}
	assert!(plan.run(&[input], &[(n, 18)]).is_err());
}
```

```sh check
cargo test --test symbolic specialization_is_distinct_and_selected_only_at_sixteen
```

## 20-020 特化 key には N=16 と tile 条件を含め、N=17でその binary を選ばないようにする。

<!-- todo:20-020 -->

特化 key に値16と tile=8を加えます。直前のテストは16、17、3の順に実行して、特化版、generic、genericを選び、各末尾も比較します。次の example で選択を表示します。

```rust file="examples/symbolic_dispatch.rs" mode="write"
use chinigrad::{
	Binary, Buffer, Graph,
	symbolic::{Dim, ShapeEnv, ShapeProgram},
};
fn main() -> Result<(), String> {
	let mut env = ShapeEnv::default();
	let n = env.variable("N", 1, 17)?;
	let mut graph = Graph::default();
	let x = graph.input(0, vec![17])?;
	let two = graph.constant(2., vec![])?;
	let one = graph.constant(1., vec![])?;
	let product = graph.binary(Binary::Mul, x, two)?;
	let y = graph.binary(Binary::Add, product, one)?;
	let mut program = ShapeProgram::new(graph, vec![y], env)?;
	program.input(0, vec![Dim::Var(n)])?;
	let mut plan = program.compile()?;
	let input = Buffer::new(vec![17], (1..=17).map(|x| x as f32).collect())?;
	plan.add_specialization(n, 16, 8, &[input.clone()], &[(n, 16)])?;
	for length in [16, 17, 3] {
		let (out, record) = plan.run(&[input.clone()], &[(n, length)])?;
		println!(
			"N={length} variant={} len={} last={}",
			record.variant,
			out[0].data.len(),
			out[0].data[length - 1]
		);
	}
	Ok(())
}
```

```sh check
cargo run --example symbolic_dispatch
```

## 20-021 添字式の memo key に node、出力座標、RANGE の scope を含めるか、scope 内に cache を閉じる。

<!-- todo:20-021 -->

graph emitter は座標評価を一回の lowering に閉じ、異なる座標や RANGE をまたぐ値の memo を持ちません。同じ入力と転置を足すテストで、共有ノードを同じ座標として扱わないことを確認します。

```rust file="tests/symbolic.rs" mode="append"
#[test]
fn shared_node_before_and_after_transpose_keeps_coordinates_separate() {
	let mut env = ShapeEnv::default();
	let n = env.variable("N", 1, 2).unwrap();
	let mut graph = Graph::default();
	let x = graph.input(0, vec![2, 2]).unwrap();
	let t = graph.permute(x, vec![1, 0]).unwrap();
	let y = graph.binary(Binary::Add, x, t).unwrap();
	let mut program = ShapeProgram::new(graph, vec![y], env).unwrap();
	program.input(0, vec![Dim::Var(n), Dim::Var(n)]).unwrap();
	let plan = program.compile().unwrap();
	let data = Buffer::new(vec![2, 2], vec![1., 2., 3., 4.]).unwrap();
	assert_eq!(
		plan.run(&[data.clone()], &[(n, 2)]).unwrap().0[0].data,
		vec![2., 5., 5., 8.]
	);
	assert_eq!(plan.run(&[data], &[(n, 1)]).unwrap().0[0].data, vec![2.]);
}
```

```sh check
cargo test --test symbolic shared_node_before_and_after_transpose_keeps_coordinates_separate
```

## 20-022 N の下限、上限、その外側、未束縛で期待どおり成功またはエラーになる。

<!-- todo:20-022 -->

`ShapeEnv::bind` の境界条件をまとめて実行します。未知 scope、欠落、二重指定、0、18を拒否し、1と17を受け付けます。

```rust file="tests/symbolic.rs" mode="append"
#[test]
fn bindings_reject_missing_duplicate_unknown_and_outside() {
	let mut env = ShapeEnv::default();
	let n = env.variable("N", 1, 17).unwrap();
	let mut another = ShapeEnv::default();
	let other_n = another.variable("N", 1, 17).unwrap();
	assert_ne!(n, other_n);
	assert!(env.bind(&[]).unwrap_err().contains("unbound"));
	assert!(
		env.bind(&[(n, 3), (n, 3)])
			.unwrap_err()
			.contains("duplicate")
	);
	assert!(env.bind(&[(other_n, 3)]).unwrap_err().contains("unknown"));
	for value in [0, 18] {
		assert!(env.bind(&[(n, value)]).is_err());
	}
	for value in [1, 17] {
		assert_eq!(env.bind(&[(n, value)]).unwrap().get(n).unwrap(), value);
	}
}
```

```sh check
cargo test --test symbolic bindings_reject_missing_duplicate_unknown_and_outside
```

## 20-023 整数の小例は期待値と完全一致する。

<!-- todo:20-023 -->

shape 式の整数演算は checked 演算です。整数で正確に表せる vector と総和も `assert_eq!` で照合します。

```sh check
cargo test --test symbolic dimensions_are_checked_and_expression_ids_do_not_alias_nodes
```

```sh check
cargo test --test symbolic generic_reuses_binary_for_three_lengths_and_ignores_nan_tail
```

## 20-024 f32 の generic、specialized、静的版は `abs_err <= 1e-5 + 1e-5*abs(ref)` を初期基準として比較する。

<!-- todo:20-024 -->

整数以外の f32 入力で generic、特化版、静的版を比較します。比較前に有限値も検査します。

```rust file="tests/symbolic.rs" mode="append"
#[test]
fn generic_specialized_and_static_f32_match_with_declared_tolerance() {
	let (mut plan, n) = vector();
	let input = Buffer::new(vec![17], (0..17).map(|i| (i as f32 * 0.17).sin()).collect()).unwrap();
	plan.add_specialization(n, 16, 8, &[input.clone()], &[(n, 16)])
		.unwrap();
	for length in [16, 17, 3] {
		let (output, _) = plan.run(&[input.clone()], &[(n, length)]).unwrap();
		for (got, &x) in output[0].data.iter().zip(&input.data) {
			let expected = 2. * x + 1.;
			assert!(got.is_finite() && (got - expected).abs() <= 1e-5 + 1e-5 * expected.abs());
		}
		let snapshot = plan
			.program()
			.prepare(&[input.clone()], &[(n, length)])
			.unwrap();
		let static_plan =
			chinigrad::Plan::compile_specialized(&snapshot.graph, &plan.program().outputs).unwrap();
		let expected = static_plan
			.run_with_shapes(&[input.clone()], &snapshot.node_shapes)
			.unwrap();
		for (a, b) in output.iter().zip(expected) {
			for (&actual, &reference) in a.data.iter().zip(&b.data) {
				assert!(actual.is_finite() && reference.is_finite());
				assert!((actual - reference).abs() <= 1e-5 + 1e-5 * reference.abs());
			}
		}
	}
}
```

```sh check
cargo test --test symbolic generic_specialized_and_static_f32_match_with_declared_tolerance
```

## 20-025 RANGE の上限と buffer の容量を区別して dump に表示する。

<!-- todo:20-025 -->

この example は Plan をループの外で作り、容量、論理 shape、値、総和を表示します。バックエンドは明示的に指定し、利用できなければエラーを返します。

```rust file="examples/symbolic_vector.rs" mode="write"
use chinigrad::{
	Binary, Buffer, Graph, Reduce,
	gpu_graph::Backend,
	symbolic::{Dim, ShapeEnv, ShapeProgram},
};

fn main() -> Result<(), String> {
	let args = std::env::args().skip(1).collect::<Vec<_>>();
	let mut backend = "cpu".to_owned();
	let mut lengths = vec![1, 3, 17];
	let mut local = 8;
	let mut i = 0;
	while i < args.len() {
		let key = &args[i];
		let value = args
			.get(i + 1)
			.ok_or_else(|| format!("missing value for {key}"))?;
		match key.as_str() {
			"--backend" => backend = value.clone(),
			"--length" => lengths = vec![value.parse().map_err(|_| "invalid length")?],
			"--lengths" => {
				lengths = value
					.split(',')
					.map(|x| x.parse().map_err(|_| "invalid lengths"))
					.collect::<Result<Vec<usize>, _>>()?
			}
			"--local" => local = value.parse().map_err(|_| "invalid local size")?,
			_ => return Err(format!("unknown option {key}")),
		}
		i += 2;
	}
	let mut env = ShapeEnv::default();
	let n = env.variable("N", 1, 17)?;
	let mut graph = Graph::default();
	let x = graph.input(0, vec![17])?;
	let two = graph.constant(2., vec![])?;
	let one = graph.constant(1., vec![])?;
	let product = graph.binary(Binary::Mul, x, two)?;
	let y = graph.binary(Binary::Add, product, one)?;
	let sum = graph.reduce(x, Reduce::Sum, vec![0], false)?;
	let mut program = ShapeProgram::new(graph, vec![y, sum], env)?;
	program.input(0, vec![Dim::Var(n)])?;
	let cpu = if backend == "cpu" {
		Some(program.clone().compile()?)
	} else {
		None
	};
	let gpu = match backend.as_str() {
		"cpu" => None,
		"metal" => Some(program.compile_gpu(Backend::Metal, local)?),
		"opencl" => Some(program.compile_gpu(Backend::OpenCl, local)?),
		_ => return Err("backend must be cpu, metal, or opencl".into()),
	};
	for length in lengths {
		if !(1..=17).contains(&length) {
			return Err("length outside [1,17]".into());
		}
		let mut data = vec![f32::NAN; 17];
		for (i, value) in data[..length].iter_mut().enumerate() {
			*value = (i + 1) as f32;
		}
		let input = Buffer::new(vec![17], data)?;
		let (out, record) = if let Some(plan) = &cpu {
			plan.run(&[input], &[(n, length)])?
		} else {
			gpu.as_ref().unwrap().run(&[input], &[(n, length)])?
		};
		println!(
			"backend={backend} N={length} capacity={:?} logical={:?} y={:?} sum={:?}",
			record.capacities, record.logical_shapes, out[0].data, out[1].data
		);
	}
	Ok(())
}
```

```sh check
cargo run --example symbolic_vector -- --backend cpu --lengths 1,3,17
```

## 20-026 動的 reshape の積、GEMM の K、broadcast の等値を起動前に検証する。

<!-- todo:20-026 -->

matmul は MUL と REDUCE の合成なので、K の等値制約を明示して保存します。二つの K を別 VarId にし、1対2が broadcast として通らないようにします。

```rust file="tests/symbolic.rs" mode="append"
#[test]
fn symbolic_gemm_k_is_an_explicit_prelaunch_equality() {
	use chinigrad::Op;
	let mut env = ShapeEnv::default();
	let ka = env.variable("K", 1, 3).unwrap();
	let kb = env.variable("K", 1, 3).unwrap();
	let mut graph = Graph::default();
	let a = graph.input(0, vec![2, 3]).unwrap();
	let b = graph.input(1, vec![3, 2]).unwrap();
	let y = graph.matmul(a, b).unwrap();
	let aa = graph
		.nodes
		.iter()
		.position(|node| matches!(node.op,Op::Reshape{x} if x==a))
		.unwrap();
	let bt = graph
		.nodes
		.iter()
		.position(|node| matches!(node.op,Op::Permute{x,..} if x==b))
		.unwrap();
	let bb = graph
		.nodes
		.iter()
		.position(|node| matches!(node.op,Op::Reshape{x} if x==bt))
		.unwrap();
	let mut program = ShapeProgram::new(graph, vec![y], env).unwrap();
	program
		.input(0, vec![Dim::Static(2), Dim::Var(ka)])
		.unwrap();
	program
		.input(1, vec![Dim::Var(kb), Dim::Static(2)])
		.unwrap();
	program
		.target(aa, vec![Dim::Static(2), Dim::Static(1), Dim::Var(ka)])
		.unwrap();
	program
		.target(bb, vec![Dim::Static(1), Dim::Static(2), Dim::Var(kb)])
		.unwrap();
	program
		.constraints
		.push(Constraint::Equal(Dim::Var(ka), Dim::Var(kb)));
	let plan = program.compile().unwrap();
	let inputs = [
		Buffer::new(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap(),
		Buffer::new(vec![3, 2], vec![1., 2., 3., 4., 5., 6.]).unwrap(),
	];
	assert!(plan.run(&inputs, &[(ka, 1), (kb, 2)]).is_err());
	assert_eq!(
		plan.run(&inputs, &[(ka, 2), (kb, 2)]).unwrap().0[0].data,
		vec![7., 10., 15., 22.]
	);
	assert_eq!(
		plan.run(&inputs, &[(ka, 3), (kb, 3)]).unwrap().0[0].data,
		vec![22., 28., 49., 64.]
	);
}
```

```sh check
cargo test --test symbolic symbolic_gemm_k_is_an_explicit_prelaunch_equality
```

```sh check
cargo test --test symbolic dynamic_reshape_rows_and_movement_match_static_graph
```

```sh check
cargo test --test symbolic broadcast_equality_constraints_fail_before_native_run
```

## 20-027 Plan を連続再実行しても、束縛値と累積値が前回から残らない。

<!-- todo:20-027 -->

同じ Plan の再実行を繰り返し、論理長と縮約の初期値が戻ることを確認します。

```sh check
cargo test --test symbolic
```

## 20-028 同じ node を転置前後で利用して、添字 cache が座標を混同しない。

<!-- todo:20-028 -->

共有した入力を転置前後で使うテストを再実行します。shape の変化と座標の変化を同時に含めます。

```sh check
cargo test --test symbolic shared_node_before_and_after_transpose_keeps_coordinates_separate
```

## 20-029 GPU での失敗は CPU の成功で置き換えず、backend 別に確認結果を残す。

<!-- todo:20-029 -->

CPU の結果とは別に各 backend のテストを実行します。デバイスがない環境では、この二つを未実行として記録します。

```sh check requires=metal
cargo test --features metal --test symbolic_gpu metal_uses_one_compilation_for_three_shapes -- --ignored
```

```sh check requires=opencl
cargo test --features opencl --test symbolic_gpu opencl_uses_one_compilation_for_three_shapes -- --ignored
```

## tinygrad の対応箇所

変数の範囲と束縛の分離は [`UOp.variable` と `UOp.bind`](../../tinygrad/tinygrad/uop/ops.py)、範囲内での簡約は [`symbolic.py`](../../tinygrad/tinygrad/uop/symbolic.py) に対応します。この教材では shape 式を非負定数、変数、加算、正の定数倍、正の定数での除算と剰余に限定します。一般の方程式ソルバ、可変 rank、データの値で変わる出力長は扱いません。

次は [21章のモデル保存と読込み](21-model-io.md) です。
