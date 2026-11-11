# 08. 一次元の shape と REDUCE を添字付きループへ変換する

ここから Tensor 用の純粋な shape 付き Graph を作る。scalar IR と異なり、Tensor の入力署名は到達した PARAM だけでなく Graph に宣言したすべての Input slot を保持する。勾配が定数になっても元の入力配列を再束縛できるためである。Graph の要素演算は f32 で、比較は0/1を返す。今章では非定数の演算をそれぞれ別 kernel にする。consumer 数による融合は09章で追加する。Kernel AST と Graph は二つの frontend であり、いずれも `native::KernelBody`、`render_entry`、二配列 ABI、NativeLibrary に合流する。Graph は座標付き C body を直接構築するので、Kernel AST を通過したとは主張しない。今後の movement を含む Op enum と renderer の全体をここで固定し、公開 builder を10、11章で追加する。

`file=... mode=write` はそのファイルの全置換、`mode=append` は末尾への追記である。すべて前章でできたプロジェクトの root で行う。`# check` は実際の検証コマンドであり、ソースファイルには保存しない。

## 1. shape を scalar と rank 1 に限定して導出する。
<!-- todo:08-001 -->

```rust
// file=src/graph.rs mode=write
use crate::{
	Result,
	runtime::{Abi, Binding, Buffer, Plan, elements},
};
use std::collections::{HashMap, HashSet};
pub type NodeId = usize;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Unary {
	Neg,
	Recip,
	Exp,
	Log,
	Sin,
	Cos,
	Sqrt,
	Exp2,
	Log2,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Binary {
	Add,
	Sub,
	Mul,
	Div,
	Max,
	Lt,
	Eq,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Reduce {
	Sum,
	Max,
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Op {
	Input {
		slot: usize,
	},
	Const {
		bits: u32,
	},
	Unary {
		op: Unary,
		x: NodeId,
	},
	Binary {
		op: Binary,
		a: NodeId,
		b: NodeId,
	},
	Where {
		cond: NodeId,
		yes: NodeId,
		no: NodeId,
	},
	Reduce {
		op: Reduce,
		x: NodeId,
		axes: Vec<usize>,
		keepdim: bool,
	},
	Reshape {
		x: NodeId,
	},
	Permute {
		x: NodeId,
		axes: Vec<usize>,
	},
	Expand {
		x: NodeId,
	},
	Shrink {
		x: NodeId,
		ranges: Vec<(usize, usize)>,
	},
	Pad {
		x: NodeId,
		pads: Vec<(usize, usize)>,
	},
	Flip {
		x: NodeId,
		axes: Vec<usize>,
	},
	Stack {
		xs: Vec<NodeId>,
		axis: usize,
	},
	Gather {
		x: NodeId,
		indices: NodeId,
		axis: usize,
	},
	ScatterAdd {
		base: NodeId,
		indices: NodeId,
		updates: NodeId,
		axis: usize,
	},
	Stage {
		x: NodeId,
	},
	StopGradient {
		x: NodeId,
	},
}
impl Op {
	pub fn sources(&self) -> Vec<NodeId> {
		match self {
			Self::Input { .. } | Self::Const { .. } => vec![],
			Self::Unary { x, .. }
			| Self::Reduce { x, .. }
			| Self::Reshape { x }
			| Self::Permute { x, .. }
			| Self::Expand { x }
			| Self::Shrink { x, .. }
			| Self::Pad { x, .. }
			| Self::Flip { x, .. }
			| Self::Stage { x }
			| Self::StopGradient { x } => vec![*x],
			Self::Binary { a, b, .. } => vec![*a, *b],
			Self::Where { cond, yes, no } => vec![*cond, *yes, *no],
			Self::Stack { xs, .. } => xs.clone(),
			Self::Gather { x, indices, .. } => vec![*x, *indices],
			Self::ScatterAdd {
				base,
				indices,
				updates,
				..
			} => vec![*base, *indices, *updates],
		}
	}
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Node {
	pub op: Op,
	pub shape: Vec<usize>,
}
#[derive(Clone, Debug, Default)]
pub struct Graph {
	pub nodes: Vec<Node>,
}
pub fn broadcast(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
	let rank = a.len().max(b.len());
	let mut out = vec![1; rank];
	for k in 0..rank {
		let x = a.len().checked_sub(k + 1).map(|i| a[i]).unwrap_or(1);
		let y = b.len().checked_sub(k + 1).map(|i| b[i]).unwrap_or(1);
		out[rank - k - 1] = if x == y {
			x
		} else if x == 1 {
			y
		} else if y == 1 {
			x
		} else {
			return Err("incompatible broadcast shapes".into());
		};
	}
	elements(&out)?;
	Ok(out)
}
fn axes_valid(axes: &[usize], rank: usize) -> Result<()> {
	let unique: HashSet<_> = axes.iter().copied().collect();
	if unique.len() != axes.len() || axes.iter().any(|&a| a >= rank) {
		return Err("invalid or repeated axis".into());
	}
	Ok(())
}
impl Graph {
	pub fn shape(&self, id: NodeId) -> &[usize] {
		&self.nodes[id].shape
	}
	fn get(&self, id: NodeId) -> Result<&Node> {
		self.nodes
			.get(id)
			.ok_or_else(|| format!("unknown node {id}"))
	}
	pub fn insert(&mut self, op: Op, shape: Vec<usize>) -> Result<NodeId> {
		elements(&shape)?;
		let node = Node { op, shape };
		self.verify_node(&node, self.nodes.len())?;
		if let Some(id) = self.nodes.iter().position(|n| n == &node) {
			return Ok(id);
		}
		let id = self.nodes.len();
		self.nodes.push(node);
		Ok(id)
	}
	pub fn input(&mut self, slot: usize, shape: Vec<usize>) -> Result<NodeId> {
		self.insert(Op::Input { slot }, shape)
	}
	pub fn constant(&mut self, value: f32, shape: Vec<usize>) -> Result<NodeId> {
		self.insert(
			Op::Const {
				bits: value.to_bits(),
			},
			shape,
		)
	}
	pub fn unary(&mut self, op: Unary, x: NodeId) -> Result<NodeId> {
		self.insert(Op::Unary { op, x }, self.get(x)?.shape.clone())
	}
	pub fn binary(&mut self, op: Binary, a: NodeId, b: NodeId) -> Result<NodeId> {
		let shape = broadcast(&self.get(a)?.shape, &self.get(b)?.shape)?;
		self.insert(Op::Binary { op, a, b }, shape)
	}
	pub fn where_(&mut self, cond: NodeId, yes: NodeId, no: NodeId) -> Result<NodeId> {
		let shape = broadcast(
			&broadcast(&self.get(cond)?.shape, &self.get(yes)?.shape)?,
			&self.get(no)?.shape,
		)?;
		self.insert(Op::Where { cond, yes, no }, shape)
	}
	pub fn reduce(
		&mut self,
		x: NodeId,
		op: Reduce,
		axes: Vec<usize>,
		keepdim: bool,
	) -> Result<NodeId> {
		let input = &self.get(x)?.shape;
		axes_valid(&axes, input.len())?;
		let shape = input
			.iter()
			.enumerate()
			.filter_map(|(i, &d)| {
				if axes.contains(&i) {
					if keepdim { Some(1) } else { None }
				} else {
					Some(d)
				}
			})
			.collect();
		self.insert(
			Op::Reduce {
				op,
				x,
				axes,
				keepdim,
			},
			shape,
		)
	}
	pub fn stage(&mut self, x: NodeId) -> Result<NodeId> {
		self.insert(Op::Stage { x }, self.get(x)?.shape.clone())
	}
	pub fn stop_gradient(&mut self, x: NodeId) -> Result<NodeId> {
		self.insert(Op::StopGradient { x }, self.get(x)?.shape.clone())
	}
	pub fn compile(&self, outputs: &[NodeId]) -> Result<Plan> {
		Plan::compile(self, outputs)
	}
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
	pub fn verify(&self, outputs: &[NodeId]) -> Result<()> {
		for id in self.topology(outputs)? {
			self.verify_node(&self.nodes[id], id)?;
		}
		Ok(())
	}
	fn verify_node(&self, node: &Node, id: usize) -> Result<()> {
		elements(&node.shape)?;
		for source in node.op.sources() {
			if source >= id {
				return Err("node source must precede its user".into());
			}
			self.get(source)?;
		}
		let s = |x: usize| self.nodes[x].shape.as_slice();
		let out = node.shape.as_slice();
		let expected = match &node.op {
			Op::Input { .. } | Op::Const { .. } => node.shape.clone(),
			Op::Unary { x, .. } | Op::Stage { x } | Op::StopGradient { x } => s(*x).to_vec(),
			Op::Binary { a, b, .. } => broadcast(s(*a), s(*b))?,
			Op::Where { cond, yes, no } => broadcast(&broadcast(s(*cond), s(*yes))?, s(*no))?,
			Op::Reduce {
				x, axes, keepdim, ..
			} => {
				axes_valid(axes, s(*x).len())?;
				s(*x)
					.iter()
					.enumerate()
					.filter_map(|(i, &d)| {
						if axes.contains(&i) {
							if *keepdim { Some(1) } else { None }
						} else {
							Some(d)
						}
					})
					.collect()
			}
			Op::Reshape { x } => {
				if elements(s(*x))? != elements(out)? {
					return Err("reshape changes element count".into());
				}
				out.to_vec()
			}
			Op::Permute { x, axes } => {
				axes_valid(axes, s(*x).len())?;
				if axes.len() != s(*x).len() {
					return Err("permute rank mismatch".into());
				}
				axes.iter().map(|&a| s(*x)[a]).collect()
			}
			Op::Expand { x } => {
				if s(*x).len() > out.len() || broadcast(s(*x), out)? != out {
					return Err("invalid expand".into());
				}
				out.to_vec()
			}
			Op::Shrink { x, ranges } => {
				if ranges.len() != s(*x).len() {
					return Err("shrink rank mismatch".into());
				}
				ranges
					.iter()
					.enumerate()
					.map(|(a, &(lo, hi))| {
						if lo <= hi && hi <= s(*x)[a] {
							Ok(hi - lo)
						} else {
							Err("invalid shrink range".to_string())
						}
					})
					.collect::<Result<_>>()?
			}
			Op::Pad { x, pads } => {
				if pads.len() != s(*x).len() {
					return Err("pad rank mismatch".into());
				}
				pads.iter()
					.enumerate()
					.map(|(a, &(lo, hi))| {
						s(*x)[a]
							.checked_add(lo)
							.and_then(|d| d.checked_add(hi))
							.ok_or("pad overflow".into())
					})
					.collect::<Result<_>>()?
			}
			Op::Flip { x, axes } => {
				axes_valid(axes, s(*x).len())?;
				s(*x).to_vec()
			}
			Op::Stack { xs, axis } => {
				if xs.is_empty() {
					return Err("empty stack".into());
				}
				let base = s(xs[0]);
				if *axis > base.len() || xs.iter().any(|&x| s(x) != base) {
					return Err("stack shape/axis mismatch".into());
				}
				let mut shape = base.to_vec();
				shape.insert(*axis, xs.len());
				shape
			}
			Op::Gather { x, indices, axis } => {
				if *axis >= s(*x).len() || s(*indices).len() != 1 {
					return Err("gather axis or indices rank".into());
				}
				let mut shape = s(*x).to_vec();
				shape[*axis] = s(*indices)[0];
				shape
			}
			Op::ScatterAdd {
				base,
				indices,
				updates,
				axis,
			} => {
				if *axis >= s(*base).len() || s(*indices).len() != 1 {
					return Err("scatter axis or indices rank".into());
				}
				let mut shape = s(*base).to_vec();
				shape[*axis] = s(*indices)[0];
				if shape != s(*updates) {
					return Err("scatter update shape".into());
				}
				s(*base).to_vec()
			}
		};
		if expected != out {
			return Err(format!("node {id} shape {:?}, expected {expected:?}", out));
		}
		Ok(())
	}
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
}

pub fn max_value(a: f32, b: f32) -> f32 {
	if a.is_nan() || b.is_nan() {
		f32::NAN
	} else if a == 0.0 && b == 0.0 {
		0.0
	} else {
		a.max(b)
	}
}
fn coords(mut i: usize, shape: &[usize]) -> Vec<usize> {
	let mut c = vec![0; shape.len()];
	for a in (0..shape.len()).rev() {
		c[a] = i % shape[a];
		i /= shape[a]
	}
	c
}
fn flat(c: &[usize], shape: &[usize]) -> usize {
	c.iter().zip(shape).fold(0, |n, (&i, &d)| n * d + i)
}
fn bindex(i: usize, out: &[usize], input: &[usize]) -> usize {
	let c = coords(i, out);
	flat(
		&input
			.iter()
			.enumerate()
			.map(|(a, &d)| {
				if d == 1 {
					0
				} else {
					c[out.len() - input.len() + a]
				}
			})
			.collect::<Vec<_>>(),
		input,
	)
}
impl Graph {
	fn interpret_node(
		&self,
		id: usize,
		inputs: &[Buffer],
		memo: &mut HashMap<usize, Buffer>,
	) -> Result<Buffer> {
		if let Some(value) = memo.get(&id) {
			return Ok(value.clone());
		}
		let node = self.get(id)?;
		let mut values = HashMap::new();
		for source in node.op.sources() {
			values.insert(source, self.interpret_node(source, inputs, memo)?);
		}
		let get = |x: usize, i: usize| -> f32 {
			let b = &values[&x];
			b.data[bindex(i, &node.shape, &b.shape)]
		};
		let mut data = vec![0.0; elements(&node.shape)?];
		match &node.op {
			Op::Input { slot } => {
				let input = inputs.get(*slot).ok_or("missing interpreter input")?;
				if input.data.len() < data.len() {
					return Err("short interpreter input".into());
				}
				let len = data.len();
				data.copy_from_slice(&input.data[..len])
			}
			Op::Const { bits } => data.fill(f32::from_bits(*bits)),
			Op::Unary { op, x } => {
				for (i, v) in data.iter_mut().enumerate() {
					let a = get(*x, i);
					*v = match op {
						Unary::Neg => -a,
						Unary::Recip => a.recip(),
						Unary::Exp => a.exp(),
						Unary::Log => a.ln(),
						Unary::Sin => a.sin(),
						Unary::Cos => a.cos(),
						Unary::Sqrt => a.sqrt(),
						Unary::Exp2 => a.exp2(),
						Unary::Log2 => a.log2(),
					}
				}
			}
			Op::Binary { op, a, b } => {
				for (i, v) in data.iter_mut().enumerate() {
					let (a, b) = (get(*a, i), get(*b, i));
					*v = match op {
						Binary::Add => a + b,
						Binary::Sub => a - b,
						Binary::Mul => a * b,
						Binary::Div => a / b,
						Binary::Max => max_value(a, b),
						Binary::Lt => (a < b) as u8 as f32,
						Binary::Eq => (a == b) as u8 as f32,
					}
				}
			}
			Op::Where { cond, yes, no } => {
				for (i, v) in data.iter_mut().enumerate() {
					*v = if get(*cond, i) != 0.0 {
						get(*yes, i)
					} else {
						get(*no, i)
					}
				}
			}
			Op::Reduce {
				op,
				x,
				axes,
				keepdim,
			} => {
				data.fill(if *op == Reduce::Sum {
					0.0
				} else {
					f32::NEG_INFINITY
				});
				let input = &values[x];
				for (i, &v) in input.data.iter().enumerate() {
					let c = coords(i, &input.shape);
					let out = c
						.iter()
						.enumerate()
						.filter_map(|(a, &c)| {
							if axes.contains(&a) {
								if *keepdim { Some(0) } else { None }
							} else {
								Some(c)
							}
						})
						.collect::<Vec<_>>();
					let j = flat(&out, &node.shape);
					data[j] = if *op == Reduce::Sum {
						data[j] + v
					} else {
						max_value(data[j], v)
					}
				}
			}
			Op::Reshape { x } | Op::Stage { x } | Op::StopGradient { x } => {
				data.copy_from_slice(&values[x].data)
			}
			Op::Expand { x } => {
				for (i, v) in data.iter_mut().enumerate() {
					*v = get(*x, i)
				}
			}
			Op::Permute { x, axes } => {
				for (i, v) in data.iter_mut().enumerate() {
					let out = coords(i, &node.shape);
					let mut c = vec![0; out.len()];
					for (a, &old) in axes.iter().enumerate() {
						c[old] = out[a]
					}
					*v = values[x].data[flat(&c, &values[x].shape)]
				}
			}
			Op::Shrink { x, ranges } => {
				for (i, v) in data.iter_mut().enumerate() {
					let c = coords(i, &node.shape)
						.iter()
						.zip(ranges)
						.map(|(&c, &(lo, _))| c + lo)
						.collect::<Vec<_>>();
					*v = values[x].data[flat(&c, &values[x].shape)]
				}
			}
			Op::Pad { x, pads } => {
				for (i, v) in data.iter_mut().enumerate() {
					let c = coords(i, &node.shape);
					if c.iter()
						.zip(pads)
						.zip(&values[x].shape)
						.all(|((&c, &(lo, _)), &n)| c >= lo && c - lo < n)
					{
						let c = c
							.iter()
							.zip(pads)
							.map(|(&c, &(lo, _))| c - lo)
							.collect::<Vec<_>>();
						*v = values[x].data[flat(&c, &values[x].shape)]
					}
				}
			}
			Op::Flip { x, axes } => {
				for (i, v) in data.iter_mut().enumerate() {
					let mut c = coords(i, &node.shape);
					for &a in axes {
						c[a] = node.shape[a] - 1 - c[a]
					}
					*v = values[x].data[flat(&c, &node.shape)]
				}
			}
			Op::Stack { xs, axis } => {
				for (i, v) in data.iter_mut().enumerate() {
					let mut c = coords(i, &node.shape);
					let x = xs[c.remove(*axis)];
					*v = values[&x].data[flat(&c, &values[&x].shape)]
				}
			}
			Op::Gather { x, indices, axis } => {
				for (i, v) in data.iter_mut().enumerate() {
					let mut c = coords(i, &node.shape);
					let index = values[indices].data[c[*axis]];
					let limit = values[x].shape[*axis];
					c[*axis] = checked_index(index, limit)?;
					*v = values[x].data[flat(&c, &values[x].shape)]
				}
			}
			Op::ScatterAdd {
				base,
				indices,
				updates,
				axis,
			} => {
				data.copy_from_slice(&values[base].data);
				for (i, &v) in values[updates].data.iter().enumerate() {
					let mut c = coords(i, &values[updates].shape);
					c[*axis] = checked_index(values[indices].data[c[*axis]], node.shape[*axis])?;
					data[flat(&c, &node.shape)] += v
				}
			}
		}
		let value = Buffer::new(node.shape.clone(), data)?;
		memo.insert(id, value.clone());
		Ok(value)
	}
	pub fn interpret(&self, outputs: &[NodeId], inputs: &[Buffer]) -> Result<Vec<Buffer>> {
		self.verify(outputs)?;
		self.validate_inputs(inputs, outputs)?;
		let mut memo = HashMap::new();
		outputs
			.iter()
			.map(|&id| self.interpret_node(id, inputs, &mut memo))
			.collect()
	}
	pub fn validate_indices(&self, inputs: &[Buffer], outputs: &[NodeId]) -> Result<()> {
		let mut memo = HashMap::new();
		for id in self.topology(outputs)? {
			let (indices, limit) = match self.nodes[id].op {
				Op::Gather { x, indices, axis } => (indices, self.shape(x)[axis]),
				Op::ScatterAdd {
					base,
					indices,
					axis,
					..
				} => (indices, self.shape(base)[axis]),
				_ => continue,
			};
			for value in self.interpret_node(indices, inputs, &mut memo)?.data {
				checked_index(value, limit)?;
			}
		}
		Ok(())
	}
}
fn checked_index(value: f32, limit: usize) -> Result<usize> {
	if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value as f64 >= limit as f64 {
		return Err("index must be an exact in-range nonnegative integer".into());
	}
	Ok(value as usize)
}

pub use crate::native::KernelBody;
struct Emitter<'a> {
	g: &'a Graph,
	offsets: Vec<usize>,
	material: HashSet<usize>,
	current: usize,
	serial: usize,
}
impl Emitter<'_> {
	fn dim(&self, id: usize, axis: usize) -> String {
		format!("s[{}]", self.offsets[id] + axis)
	}
	fn product(&self, id: usize, axes: impl Iterator<Item = usize>) -> String {
		let dims = axes.map(|a| self.dim(id, a)).collect::<Vec<_>>();
		if dims.is_empty() {
			"1".into()
		} else {
			format!("({})", dims.join(" * "))
		}
	}
	fn count(&self, id: usize) -> String {
		self.product(id, 0..self.g.shape(id).len())
	}
	fn coord(&self, id: usize, index: &str, axis: usize) -> String {
		format!(
			"((({index}) / {}) % {})",
			self.product(id, axis + 1..self.g.shape(id).len()),
			self.dim(id, axis)
		)
	}
	fn flatten(&self, id: usize, c: &[String]) -> String {
		if c.is_empty() {
			return "0".into();
		}
		format!(
			"({})",
			c.iter()
				.enumerate()
				.map(|(a, c)| format!(
					"({c}) * {}",
					self.product(id, a + 1..self.g.shape(id).len())
				))
				.collect::<Vec<_>>()
				.join(" + ")
		)
	}
	fn broadcast_index(&self, input: usize, output: usize, index: &str) -> String {
		let rank = self.g.shape(input).len();
		let outer = self.g.shape(output).len();
		self.flatten(
			input,
			&(0..rank)
				.map(|a| {
					format!(
						"({} == 1 ? 0 : {})",
						self.dim(input, a),
						self.coord(output, index, outer - rank + a)
					)
				})
				.collect::<Vec<_>>(),
		)
	}
	fn name(&mut self) -> String {
		let name = format!("v{}", self.serial);
		self.serial += 1;
		name
	}
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
}
pub fn kernel_bodies(graph: &Graph, outputs: &[NodeId], _fuse: bool) -> Result<Vec<KernelBody>> {
	graph.verify(outputs)?;
	let order = graph.topology(outputs)?;
	let material: HashSet<_> = order
		.iter()
		.copied()
		.filter(|&id| outputs.contains(&id) || !matches!(graph.nodes[id].op, Op::Const { .. }))
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
	Ok(kernels)
}
pub const C_HELPERS: &str = "#include <stdint.h>\n#include <stddef.h>\n#include <math.h>\n#include <string.h>\nstatic float chibi_bits(uint32_t x){float y;memcpy(&y,&x,4);return y;}\nstatic float chibi_max(float a,float b){if(isnan(a)||isnan(b))return NAN;if(a==0.0f&&b==0.0f)return 0.0f;return a>b?a:b;}\n";
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

impl Graph {
	pub fn expand(&mut self, x: NodeId, shape: Vec<usize>) -> Result<NodeId> {
		self.insert(Op::Expand { x }, shape)
	}
}
```

```rust
// file=src/runtime.rs mode=write
use crate::Result;
pub use crate::native::NativeLibrary;
use std::ffi::c_void;

#[derive(Clone, Debug, PartialEq)]
pub struct Buffer {
	pub shape: Vec<usize>,
	pub data: Vec<f32>,
}
impl Buffer {
	pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self> {
		if elements(&shape)? != data.len() {
			return Err("buffer shape/data length mismatch".into());
		}
		Ok(Self { shape, data })
	}
	pub fn zeros(shape: Vec<usize>) -> Result<Self> {
		let n = elements(&shape)?;
		Self::new(shape, vec![0.0; n])
	}
}
pub fn elements(shape: &[usize]) -> Result<usize> {
	let n = shape.iter().try_fold(1usize, |a, &b| {
		a.checked_mul(b).ok_or("shape product overflow")
	})?;
	if n > isize::MAX as usize / 4 {
		return Err("allocation exceeds addressable size".into());
	}
	Ok(n)
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Binding {
	pub slot: usize,
	pub shape: Vec<usize>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Abi {
	pub inputs: Vec<Binding>,
	pub outputs: Vec<Binding>,
}

pub struct Plan {
	pub source: String,
	pub abi: Abi,
	pub graph: crate::Graph,
	pub outputs: Vec<usize>,
	pub kernel_count: usize,
	library: NativeLibrary,
	specialized: bool,
	allocations: Vec<usize>,
	sealed: (Vec<crate::Node>, Vec<usize>, Abi),
}
impl Plan {
	pub fn compile(graph: &crate::Graph, outputs: &[usize]) -> Result<Self> {
		let (source, abi, kernel_count) = crate::graph::render(graph, outputs, true)?;
		let library = unsafe { NativeLibrary::compile(&source, "chibi_entry") }?;
		Ok(Self {
			sealed: (graph.nodes.clone(), outputs.to_vec(), abi.clone()),
			source,
			abi,
			graph: graph.clone(),
			outputs: outputs.to_vec(),
			kernel_count,
			library,
			specialized: false,
			allocations: crate::graph::kernel_bodies(graph, outputs, true)?
				.iter()
				.map(|k| k.output)
				.collect(),
		})
	}
	pub fn compile_unfused(graph: &crate::Graph, outputs: &[usize]) -> Result<Self> {
		let (source, abi, kernel_count) = crate::graph::render(graph, outputs, false)?;
		let library = unsafe { NativeLibrary::compile(&source, "chibi_entry") }?;
		Ok(Self {
			sealed: (graph.nodes.clone(), outputs.to_vec(), abi.clone()),
			source,
			abi,
			graph: graph.clone(),
			outputs: outputs.to_vec(),
			kernel_count,
			library,
			specialized: false,
			allocations: crate::graph::kernel_bodies(graph, outputs, false)?
				.iter()
				.map(|k| k.output)
				.collect(),
		})
	}
	pub(crate) fn compile_source(
		graph: &crate::Graph,
		outputs: &[usize],
		source: String,
	) -> Result<Self> {
		let (_, abi, kernel_count) = crate::graph::render(graph, outputs, true)?;
		let library = unsafe { NativeLibrary::compile(&source, "chibi_entry") }?;
		Ok(Self {
			sealed: (graph.nodes.clone(), outputs.to_vec(), abi.clone()),
			source,
			abi,
			graph: graph.clone(),
			outputs: outputs.to_vec(),
			kernel_count,
			library,
			specialized: false,
			allocations: crate::graph::kernel_bodies(graph, outputs, true)?
				.iter()
				.map(|k| k.output)
				.collect(),
		})
	}
	pub(crate) fn compile_source_specialized(
		graph: &crate::Graph,
		outputs: &[usize],
		source: String,
	) -> Result<Self> {
		let mut plan = Self::compile_source(graph, outputs, source)?;
		plan.specialized = true;
		Ok(plan)
	}
	pub fn compile_specialized(graph: &crate::Graph, outputs: &[usize]) -> Result<Self> {
		let (mut source, _, _) = crate::graph::render(graph, outputs, true)?;
		source = source.replace("buffers[", "buffer_table[");
		for (slot, dim) in graph.nodes.iter().flat_map(|n| n.shape.iter()).enumerate() {
			source = source.replace(&format!("s[{slot}]"), &format!("{dim}ull"));
		}
		source = source.replace("buffer_table[", "buffers[");
		Self::compile_source_specialized(graph, outputs, source)
	}
	fn verify_seal(&self) -> Result<()> {
		if self.graph.nodes != self.sealed.0
			|| self.outputs != self.sealed.1
			|| self.abi != self.sealed.2
		{
			return Err("compiled Plan metadata was modified".into());
		}
		Ok(())
	}
	pub fn workspace_bytes(&self) -> Result<usize> {
		self.verify_seal()?;
		self.allocations.iter().try_fold(0usize, |sum, &id| {
			sum.checked_add(
				elements(&self.graph.nodes[id].shape)?
					.checked_mul(4)
					.ok_or("workspace size overflow")?,
			)
			.ok_or("workspace size overflow".into())
		})
	}
	pub fn run(&self, inputs: &[Buffer]) -> Result<Vec<Buffer>> {
		self.verify_seal()?;
		self.graph.validate_inputs(inputs, &self.outputs)?;
		self.run_with_shapes(
			inputs,
			&self
				.graph
				.nodes
				.iter()
				.map(|n| n.shape.clone())
				.collect::<Vec<_>>(),
		)
	}
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
}
```

```rust
// file=src/lib.rs mode=append
pub mod graph;
```

```rust
// file=src/lib.rs mode=append
pub mod runtime;
```

```rust
// file=src/lib.rs mode=append
pub use graph::{Binary, Graph, Node, NodeId, Op, Reduce, Unary};
pub use runtime::{Abi, Binding, Buffer, Plan};
```

```rust
// file=tests/shapes_rangeify.rs mode=write
use chibigrad::{Binary, Buffer, Graph, Reduce, Result, Unary};
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
fn unfused_affine_and_reduce_max() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let two = g.constant(2., vec![])?;
	let one = g.constant(1., vec![])?;
	let mul = g.binary(Binary::Mul, x, two)?;
	let affine = g.binary(Binary::Add, mul, one)?;
	let p = chibigrad::Plan::compile_unfused(&g, &[affine])?;
	assert_eq!(p.kernel_count, 2);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [3., 5., 7.]);
	let sum = g.reduce(x, Reduce::Sum, vec![0], false)?;
	let scaled = g.reduce(mul, Reduce::Sum, vec![0], false)?;
	let max = g.reduce(x, Reduce::Max, vec![0], false)?;
	assert_eq!(
		g.compile(&[sum, scaled])?
			.run(&[buffer(&[3], &[1., 2., 3.])])?
			.iter()
			.map(|b| b.data[0])
			.collect::<Vec<_>>(),
		[6., 12.]
	);
	assert_eq!(
		g.compile(&[max])?.run(&[buffer(&[3], &[-5., -2., -7.])])?[0].data,
		[-2.]
	);
	let bias = g.constant(10., vec![1])?;
	let biased = g.binary(Binary::Add, x, bias)?;
	assert_eq!(
		g.compile(&[biased])?.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data,
		[11., 12., 13.]
	);
	assert!(g.expand(x, vec![4]).is_err());
	Ok(())
}
```

`tests/shapes_rangeify.rs` の `unfused_affine_and_reduce_max` で、この経路の値または拒否条件を確認する。

```sh
# check
cargo test --quiet --test shapes_rangeify unfused_affine_and_reduce_max
```

## 2. 同じ長さの入力二つから output RANGE を作る。
<!-- todo:08-002 -->

`tests/shapes_rangeify.rs` の `unfused_affine_and_reduce_max` で、この経路の値または拒否条件を確認する。

`src/graph.rs` の `pub fn kernel_bodies(`の定義を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="pub fn kernel_bodies("
pub fn kernel_bodies(graph: &Graph, outputs: &[NodeId], _fuse: bool) -> Result<Vec<KernelBody>> {
	graph.verify(outputs)?;
	let order = graph.topology(outputs)?;
	let material: HashSet<_> = order
		.iter()
		.copied()
		.filter(|&id| outputs.contains(&id) || !matches!(graph.nodes[id].op, Op::Const { .. }))
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
	Ok(kernels)
}
```

```sh
# check
cargo test --quiet --test shapes_rangeify unfused_affine_and_reduce_max
```

## 3. ALU の座標を入力へ伝え、INDEX と LOAD を生成する。
<!-- todo:08-003 -->

`tests/shapes_rangeify.rs` の `unfused_affine_and_reduce_max` で、この経路の値または拒否条件を確認する。

`src/graph.rs` の `fn expr(` の `Op::Binary` 分岐を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn expr(" arm="Op::Binary"
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

```sh
# check
cargo test --quiet --test shapes_rangeify unfused_affine_and_reduce_max
```

## 4. 高水準加算から `[1,2,3]+[10,20,30]=[11,22,33]` を得る。
<!-- todo:08-004 -->

`tests/shapes_rangeify.rs` の `unfused_affine_and_reduce_max` で、この経路の値または拒否条件を確認する。

`tests/shapes_rangeify.rs` の `fn unfused_affine_and_reduce_max(`の定義を次の全体で置き換えます。

```rust
// file="tests/shapes_rangeify.rs" mode="replace" item="fn unfused_affine_and_reduce_max("
fn unfused_affine_and_reduce_max() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let two = g.constant(2., vec![])?;
	let one = g.constant(1., vec![])?;
	let mul = g.binary(Binary::Mul, x, two)?;
	let affine = g.binary(Binary::Add, mul, one)?;
	let p = chibigrad::Plan::compile_unfused(&g, &[affine])?;
	assert_eq!(p.kernel_count, 2);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [3., 5., 7.]);
	let sum = g.reduce(x, Reduce::Sum, vec![0], false)?;
	let scaled = g.reduce(mul, Reduce::Sum, vec![0], false)?;
	let max = g.reduce(x, Reduce::Max, vec![0], false)?;
	assert_eq!(
		g.compile(&[sum, scaled])?
			.run(&[buffer(&[3], &[1., 2., 3.])])?
			.iter()
			.map(|b| b.data[0])
			.collect::<Vec<_>>(),
		[6., 12.]
	);
	assert_eq!(
		g.compile(&[max])?.run(&[buffer(&[3], &[-5., -2., -7.])])?[0].data,
		[-2.]
	);
	let bias = g.constant(10., vec![1])?;
	let biased = g.binary(Binary::Add, x, bias)?;
	assert_eq!(
		g.compile(&[biased])?.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data,
		[11., 12., 13.]
	);
	assert!(g.expand(x, vec![4]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test shapes_rangeify unfused_affine_and_reduce_max
```

## 5. 長さ不一致を kernel 生成前に拒否する。
<!-- todo:08-005 -->

`tests/shapes_rangeify.rs` の `unfused_affine_and_reduce_max` で、この経路の値または拒否条件を確認する。

`src/graph.rs` の `pub fn broadcast(`の定義を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="pub fn broadcast("
pub fn broadcast(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
	let rank = a.len().max(b.len());
	let mut out = vec![1; rank];
	for k in 0..rank {
		let x = a.len().checked_sub(k + 1).map(|i| a[i]).unwrap_or(1);
		let y = b.len().checked_sub(k + 1).map(|i| b[i]).unwrap_or(1);
		out[rank - k - 1] = if x == y {
			x
		} else if x == 1 {
			y
		} else if y == 1 {
			x
		} else {
			return Err("incompatible broadcast shapes".into());
		};
	}
	elements(&out)?;
	Ok(out)
}
```

```sh
# check
cargo test --quiet --test shapes_rangeify unfused_affine_and_reduce_max
```

## 6. scalar と長さ 1 の入力に EXPAND を加える。
<!-- todo:08-006 -->

`tests/shapes_rangeify.rs` の `unfused_affine_and_reduce_max` で、この経路の値または拒否条件を確認する。

`src/graph.rs` の `fn expr(` の `Op::Expand` 分岐を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn expr(" arm="Op::Expand"
			Op::Expand { x } => {
				let xi = self.broadcast_index(x, id, index);
				return self.expr(x, &xi, code);
			}
```

```sh
# check
cargo test --quiet --test shapes_rangeify unfused_affine_and_reduce_max
```

## 7. 入力の singleton 座標を 0 にする。
<!-- todo:08-007 -->

`tests/shapes_rangeify.rs` の `unfused_affine_and_reduce_max` で、この経路の値または拒否条件を確認する。

`src/graph.rs` の `fn broadcast_index(`の定義を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn broadcast_index("
	fn broadcast_index(&self, input: usize, output: usize, index: &str) -> String {
		let rank = self.g.shape(input).len();
		let outer = self.g.shape(output).len();
		self.flatten(
			input,
			&(0..rank)
				.map(|a| {
					format!(
						"({} == 1 ? 0 : {})",
						self.dim(input, a),
						self.coord(output, index, outer - rank + a)
					)
				})
				.collect::<Vec<_>>(),
		)
	}
```

```sh
# check
cargo test --quiet --test shapes_rangeify unfused_affine_and_reduce_max
```

## 8. MUL の出力を新規 buffer に保存し、次の ADD が読むようにする。
<!-- todo:08-008 -->

`tests/shapes_rangeify.rs` の `unfused_affine_and_reduce_max` で、この経路の値または拒否条件を確認する。

`src/graph.rs` の `pub fn kernel_bodies(`の定義を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="pub fn kernel_bodies("
pub fn kernel_bodies(graph: &Graph, outputs: &[NodeId], _fuse: bool) -> Result<Vec<KernelBody>> {
	graph.verify(outputs)?;
	let order = graph.topology(outputs)?;
	let material: HashSet<_> = order
		.iter()
		.copied()
		.filter(|&id| outputs.contains(&id) || !matches!(graph.nodes[id].op, Op::Const { .. }))
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
	Ok(kernels)
}
```

```sh
# check
cargo test --quiet --test shapes_rangeify unfused_affine_and_reduce_max
```

## 9. `x*2+1` を `[3,5,7]` にする。
<!-- todo:08-009 -->

`tests/shapes_rangeify.rs` の `unfused_affine_and_reduce_max` で、この経路の値または拒否条件を確認する。

`tests/shapes_rangeify.rs` の `fn unfused_affine_and_reduce_max(`の定義を次の全体で置き換えます。

```rust
// file="tests/shapes_rangeify.rs" mode="replace" item="fn unfused_affine_and_reduce_max("
fn unfused_affine_and_reduce_max() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let two = g.constant(2., vec![])?;
	let one = g.constant(1., vec![])?;
	let mul = g.binary(Binary::Mul, x, two)?;
	let affine = g.binary(Binary::Add, mul, one)?;
	let p = chibigrad::Plan::compile_unfused(&g, &[affine])?;
	assert_eq!(p.kernel_count, 2);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [3., 5., 7.]);
	let sum = g.reduce(x, Reduce::Sum, vec![0], false)?;
	let scaled = g.reduce(mul, Reduce::Sum, vec![0], false)?;
	let max = g.reduce(x, Reduce::Max, vec![0], false)?;
	assert_eq!(
		g.compile(&[sum, scaled])?
			.run(&[buffer(&[3], &[1., 2., 3.])])?
			.iter()
			.map(|b| b.data[0])
			.collect::<Vec<_>>(),
		[6., 12.]
	);
	assert_eq!(
		g.compile(&[max])?.run(&[buffer(&[3], &[-5., -2., -7.])])?[0].data,
		[-2.]
	);
	let bias = g.constant(10., vec![1])?;
	let biased = g.binary(Binary::Add, x, bias)?;
	assert_eq!(
		g.compile(&[biased])?.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data,
		[11., 12., 13.]
	);
	assert!(g.expand(x, vec![4]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test shapes_rangeify unfused_affine_and_reduce_max
```

## 10. `[1]` の bias `[10]` を展開し、`x+b` を `[11,12,13]` にする。
<!-- todo:08-010 -->

`tests/shapes_rangeify.rs` の `unfused_affine_and_reduce_max` で、この経路の値または拒否条件を確認する。

`tests/shapes_rangeify.rs` の `fn unfused_affine_and_reduce_max(`の定義を次の全体で置き換えます。

```rust
// file="tests/shapes_rangeify.rs" mode="replace" item="fn unfused_affine_and_reduce_max("
fn unfused_affine_and_reduce_max() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let two = g.constant(2., vec![])?;
	let one = g.constant(1., vec![])?;
	let mul = g.binary(Binary::Mul, x, two)?;
	let affine = g.binary(Binary::Add, mul, one)?;
	let p = chibigrad::Plan::compile_unfused(&g, &[affine])?;
	assert_eq!(p.kernel_count, 2);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [3., 5., 7.]);
	let sum = g.reduce(x, Reduce::Sum, vec![0], false)?;
	let scaled = g.reduce(mul, Reduce::Sum, vec![0], false)?;
	let max = g.reduce(x, Reduce::Max, vec![0], false)?;
	assert_eq!(
		g.compile(&[sum, scaled])?
			.run(&[buffer(&[3], &[1., 2., 3.])])?
			.iter()
			.map(|b| b.data[0])
			.collect::<Vec<_>>(),
		[6., 12.]
	);
	assert_eq!(
		g.compile(&[max])?.run(&[buffer(&[3], &[-5., -2., -7.])])?[0].data,
		[-2.]
	);
	let bias = g.constant(10., vec![1])?;
	let biased = g.binary(Binary::Add, x, bias)?;
	assert_eq!(
		g.compile(&[biased])?.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data,
		[11., 12., 13.]
	);
	assert!(g.expand(x, vec![4]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test shapes_rangeify unfused_affine_and_reduce_max
```

## 11. REDUCE の入力 shape と出力 scalar を検証する。
<!-- todo:08-011 -->

`tests/shapes_rangeify.rs` の `unfused_affine_and_reduce_max` で、この経路の値または拒否条件を確認する。

`src/graph.rs` の `pub fn reduce(`の定義を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="pub fn reduce("
	pub fn reduce(
		&mut self,
		x: NodeId,
		op: Reduce,
		axes: Vec<usize>,
		keepdim: bool,
	) -> Result<NodeId> {
		let input = &self.get(x)?.shape;
		axes_valid(&axes, input.len())?;
		let shape = input
			.iter()
			.enumerate()
			.filter_map(|(i, &d)| {
				if axes.contains(&i) {
					if keepdim { Some(1) } else { None }
				} else {
					Some(d)
				}
			})
			.collect();
		self.insert(
			Op::Reduce {
				op,
				x,
				axes,
				keepdim,
			},
			shape,
		)
	}
```

```sh
# check
cargo test --quiet --test shapes_rangeify unfused_affine_and_reduce_max
```

## 12. reduction 軸 k と accumulator の初期化を生成する。
<!-- todo:08-012 -->

`tests/shapes_rangeify.rs` の `unfused_affine_and_reduce_max` で、この経路の値または拒否条件を確認する。

`src/graph.rs` の `fn expr(` の `Op::Reduce` 分岐を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn expr(" arm="Op::Reduce"
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

```sh
# check
cargo test --quiet --test shapes_rangeify unfused_affine_and_reduce_max
```

## 13. 前章の逐次累積と同じ低水準 IR へ変換する。
<!-- todo:08-013 -->

07章までの `Kernel::lower` と今回の `graph::kernel_bodies` は、どちらも `native::KernelBody` を返します。`Stmt` と `Expr` はメモリ・ループfrontendの表現として残し、TensorのGraphをそのASTへ変換したとは扱いません。共通の低水準表現とentry生成を、型とnative実行で照合します。

```rust
// file=tests/shared_kernel_ir.rs mode=write
use chibigrad::{
	Graph, Result,
	kernel::{Expr, Index, Kernel, store},
	native::KernelBody,
};
#[test]
fn frontends_lower_to_the_same_kernel_body() -> Result<()> {
	let mut low = Kernel::default();
	let output = low.alloc(1)?;
	low.body = vec![store(output, Index::Const(0), Expr::Const(7.))];
	low.outputs = vec![output];
	let body: KernelBody = low.lower()?;
	assert_eq!(body.output, output.storage);
	assert!(body.dependencies.is_empty());
	let mut high = Graph::default();
	let root = high.constant(7., vec![])?;
	let bodies: Vec<KernelBody> = chibigrad::graph::kernel_bodies(&high, &[root], false)?;
	assert_eq!(bodies.len(), 1);
	assert_eq!(low.compile()?.run(&[])?, vec![vec![7.]]);
	assert_eq!(high.compile(&[root])?.run(&[])?[0].data, vec![7.]);
	Ok(())
}
```

`src/kernel.rs` の `pub fn lower(`の定義を次の全体で置き換えます。

```rust
// file="src/kernel.rs" mode="replace" item="pub fn lower("
	pub fn lower(&self) -> Result<crate::native::KernelBody> {
		self.verify()?;
		fn emit(body: &[Stmt], out: &mut String) {
			for s in body {
				match s {
					Stmt::Store { view, index, value } => out.push_str(&format!(
						"b{}[{}+({})]={};\n",
						view.storage,
						view.offset,
						index.source(),
						value.source()
					)),
					Stmt::Range { axis, len, body } => {
						out.push_str(&format!(
							"for(int32_t i{axis}=0;i{axis}<{len};++i{axis}){{\n"
						));
						emit(body, out);
						out.push_str("}\n");
					}
				}
			}
		}
		let mut body = String::new();
		emit(&self.body, &mut body);
		let accesses = self.verify()?;
		let kernel = crate::native::KernelBody {
			output: self.outputs.first().map(|v| v.storage).unwrap_or(0),
			body,
			dependencies: accesses.read.into_iter().collect(),
		};
		Ok(kernel)
	}
```

`src/graph.rs` の `pub fn kernel_bodies(`の定義を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="pub fn kernel_bodies("
pub fn kernel_bodies(graph: &Graph, outputs: &[NodeId], _fuse: bool) -> Result<Vec<KernelBody>> {
	graph.verify(outputs)?;
	let order = graph.topology(outputs)?;
	let material: HashSet<_> = order
		.iter()
		.copied()
		.filter(|&id| outputs.contains(&id) || !matches!(graph.nodes[id].op, Op::Const { .. }))
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
	Ok(kernels)
}
```

`src/native.rs` の `pub fn render_entry(`の定義を次の全体で置き換えます。

```rust
// file="src/native.rs" mode="replace" item="pub fn render_entry("
pub fn render_entry(helpers: &str, slots: &[usize], kernels: &[KernelBody]) -> String {
	let mut source = helpers.to_string();
	source.push_str("int chibi_entry(void*const* buffers,const uint64_t* s){\n");
	for id in slots {
		source.push_str(&format!("float *b{id}=(float*)buffers[{id}];\n"));
	}
	for kernel in kernels {
		source.push_str(&kernel.body)
	}
	source.push_str("return 0;\n}\n");
	source
}
```

```sh
# check
cargo test --test shared_kernel_ir
```


`tests/shapes_rangeify.rs` の `unfused_affine_and_reduce_max` で、この経路の値または拒否条件を確認する。

```sh
# check
cargo test --quiet --test shapes_rangeify unfused_affine_and_reduce_max
```

## 14. `[1,2,3]` の総和を `6` にする。
<!-- todo:08-014 -->

`tests/shapes_rangeify.rs` の `unfused_affine_and_reduce_max` で、この経路の値または拒否条件を確認する。

`tests/shapes_rangeify.rs` の `fn unfused_affine_and_reduce_max(`の定義を次の全体で置き換えます。

```rust
// file="tests/shapes_rangeify.rs" mode="replace" item="fn unfused_affine_and_reduce_max("
fn unfused_affine_and_reduce_max() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let two = g.constant(2., vec![])?;
	let one = g.constant(1., vec![])?;
	let mul = g.binary(Binary::Mul, x, two)?;
	let affine = g.binary(Binary::Add, mul, one)?;
	let p = chibigrad::Plan::compile_unfused(&g, &[affine])?;
	assert_eq!(p.kernel_count, 2);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [3., 5., 7.]);
	let sum = g.reduce(x, Reduce::Sum, vec![0], false)?;
	let scaled = g.reduce(mul, Reduce::Sum, vec![0], false)?;
	let max = g.reduce(x, Reduce::Max, vec![0], false)?;
	assert_eq!(
		g.compile(&[sum, scaled])?
			.run(&[buffer(&[3], &[1., 2., 3.])])?
			.iter()
			.map(|b| b.data[0])
			.collect::<Vec<_>>(),
		[6., 12.]
	);
	assert_eq!(
		g.compile(&[max])?.run(&[buffer(&[3], &[-5., -2., -7.])])?[0].data,
		[-2.]
	);
	let bias = g.constant(10., vec![1])?;
	let biased = g.binary(Binary::Add, x, bias)?;
	assert_eq!(
		g.compile(&[biased])?.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data,
		[11., 12., 13.]
	);
	assert!(g.expand(x, vec![4]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test shapes_rangeify unfused_affine_and_reduce_max
```

## 15. `sum(x*2)` を中間 MUL と別 reduction kernel で `12` にする。
<!-- todo:08-015 -->

`tests/shapes_rangeify.rs` の `unfused_affine_and_reduce_max` で、この経路の値または拒否条件を確認する。

`tests/shapes_rangeify.rs` の `fn unfused_affine_and_reduce_max(`の定義を次の全体で置き換えます。

```rust
// file="tests/shapes_rangeify.rs" mode="replace" item="fn unfused_affine_and_reduce_max("
fn unfused_affine_and_reduce_max() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let two = g.constant(2., vec![])?;
	let one = g.constant(1., vec![])?;
	let mul = g.binary(Binary::Mul, x, two)?;
	let affine = g.binary(Binary::Add, mul, one)?;
	let p = chibigrad::Plan::compile_unfused(&g, &[affine])?;
	assert_eq!(p.kernel_count, 2);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [3., 5., 7.]);
	let sum = g.reduce(x, Reduce::Sum, vec![0], false)?;
	let scaled = g.reduce(mul, Reduce::Sum, vec![0], false)?;
	let max = g.reduce(x, Reduce::Max, vec![0], false)?;
	assert_eq!(
		g.compile(&[sum, scaled])?
			.run(&[buffer(&[3], &[1., 2., 3.])])?
			.iter()
			.map(|b| b.data[0])
			.collect::<Vec<_>>(),
		[6., 12.]
	);
	assert_eq!(
		g.compile(&[max])?.run(&[buffer(&[3], &[-5., -2., -7.])])?[0].data,
		[-2.]
	);
	let bias = g.constant(10., vec![1])?;
	let biased = g.binary(Binary::Add, x, bias)?;
	assert_eq!(
		g.compile(&[biased])?.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data,
		[11., 12., 13.]
	);
	assert!(g.expand(x, vec![4]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test shapes_rangeify unfused_affine_and_reduce_max
```

## 16. REDUCE MAX の初期値を f32 の負の無限大にする。
<!-- todo:08-016 -->

`tests/shapes_rangeify.rs` の `unfused_affine_and_reduce_max` で、この経路の値または拒否条件を確認する。

`src/graph.rs` の `fn expr(` の `Op::Reduce` 分岐を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="fn expr(" arm="Op::Reduce"
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

```sh
# check
cargo test --quiet --test shapes_rangeify unfused_affine_and_reduce_max
```

## 17. 第三章の MAX の特殊値契約を interpreter と C で維持する。
<!-- todo:08-017 -->

`tests/shapes_rangeify.rs` の `unfused_affine_and_reduce_max` で、この経路の値または拒否条件を確認する。

`src/graph.rs` の `pub fn max_value(`の定義を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="pub fn max_value("
pub fn max_value(a: f32, b: f32) -> f32 {
	if a.is_nan() || b.is_nan() {
		f32::NAN
	} else if a == 0.0 && b == 0.0 {
		0.0
	} else {
		a.max(b)
	}
}
```

`src/graph.rs` の `pub const C_HELPERS`の定義を次の全体で置き換えます。

```rust
// file="src/graph.rs" mode="replace" item="pub const C_HELPERS"
pub const C_HELPERS: &str = "#include <stdint.h>\n#include <stddef.h>\n#include <math.h>\n#include <string.h>\nstatic float chibi_bits(uint32_t x){float y;memcpy(&y,&x,4);return y;}\nstatic float chibi_max(float a,float b){if(isnan(a)||isnan(b))return NAN;if(a==0.0f&&b==0.0f)return 0.0f;return a>b?a:b;}\n";
```

```sh
# check
cargo test --quiet --test shapes_rangeify unfused_affine_and_reduce_max
```

## 18. `max([-5,-2,-7])=-2` を既存 CPU 経路で実行する。
<!-- todo:08-018 -->

`tests/shapes_rangeify.rs` の `unfused_affine_and_reduce_max` で、この経路の値または拒否条件を確認する。

`tests/shapes_rangeify.rs` の `fn unfused_affine_and_reduce_max(`の定義を次の全体で置き換えます。

```rust
// file="tests/shapes_rangeify.rs" mode="replace" item="fn unfused_affine_and_reduce_max("
fn unfused_affine_and_reduce_max() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let two = g.constant(2., vec![])?;
	let one = g.constant(1., vec![])?;
	let mul = g.binary(Binary::Mul, x, two)?;
	let affine = g.binary(Binary::Add, mul, one)?;
	let p = chibigrad::Plan::compile_unfused(&g, &[affine])?;
	assert_eq!(p.kernel_count, 2);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [3., 5., 7.]);
	let sum = g.reduce(x, Reduce::Sum, vec![0], false)?;
	let scaled = g.reduce(mul, Reduce::Sum, vec![0], false)?;
	let max = g.reduce(x, Reduce::Max, vec![0], false)?;
	assert_eq!(
		g.compile(&[sum, scaled])?
			.run(&[buffer(&[3], &[1., 2., 3.])])?
			.iter()
			.map(|b| b.data[0])
			.collect::<Vec<_>>(),
		[6., 12.]
	);
	assert_eq!(
		g.compile(&[max])?.run(&[buffer(&[3], &[-5., -2., -7.])])?[0].data,
		[-2.]
	);
	let bias = g.constant(10., vec![1])?;
	let biased = g.binary(Binary::Add, x, bias)?;
	assert_eq!(
		g.compile(&[biased])?.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data,
		[11., 12., 13.]
	);
	assert!(g.expand(x, vec![4]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test shapes_rangeify unfused_affine_and_reduce_max
```

## 19. 同じノードを二つの出力 RANGE から lower する fixture を作る。
<!-- todo:08-019 -->

`tests/shapes_rangeify.rs` の `unfused_affine_and_reduce_max` で、この経路の値または拒否条件を確認する。

`tests/shapes_rangeify.rs` の `fn unfused_affine_and_reduce_max(`の定義を次の全体で置き換えます。

```rust
// file="tests/shapes_rangeify.rs" mode="replace" item="fn unfused_affine_and_reduce_max("
fn unfused_affine_and_reduce_max() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let two = g.constant(2., vec![])?;
	let one = g.constant(1., vec![])?;
	let mul = g.binary(Binary::Mul, x, two)?;
	let affine = g.binary(Binary::Add, mul, one)?;
	let p = chibigrad::Plan::compile_unfused(&g, &[affine])?;
	assert_eq!(p.kernel_count, 2);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [3., 5., 7.]);
	let sum = g.reduce(x, Reduce::Sum, vec![0], false)?;
	let scaled = g.reduce(mul, Reduce::Sum, vec![0], false)?;
	let max = g.reduce(x, Reduce::Max, vec![0], false)?;
	assert_eq!(
		g.compile(&[sum, scaled])?
			.run(&[buffer(&[3], &[1., 2., 3.])])?
			.iter()
			.map(|b| b.data[0])
			.collect::<Vec<_>>(),
		[6., 12.]
	);
	assert_eq!(
		g.compile(&[max])?.run(&[buffer(&[3], &[-5., -2., -7.])])?[0].data,
		[-2.]
	);
	let bias = g.constant(10., vec![1])?;
	let biased = g.binary(Binary::Add, x, bias)?;
	assert_eq!(
		g.compile(&[biased])?.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data,
		[11., 12., 13.]
	);
	assert!(g.expand(x, vec![4]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test shapes_rangeify unfused_affine_and_reduce_max
```

## 20. 軸の漏れ、違う座標の誤共有、scalar の保存長の誤りを検証する。
<!-- todo:08-020 -->

`tests/shapes_rangeify.rs` の `unfused_affine_and_reduce_max` で、この経路の値または拒否条件を確認する。

`tests/shapes_rangeify.rs` の `fn unfused_affine_and_reduce_max(`の定義を次の全体で置き換えます。

```rust
// file="tests/shapes_rangeify.rs" mode="replace" item="fn unfused_affine_and_reduce_max("
fn unfused_affine_and_reduce_max() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let two = g.constant(2., vec![])?;
	let one = g.constant(1., vec![])?;
	let mul = g.binary(Binary::Mul, x, two)?;
	let affine = g.binary(Binary::Add, mul, one)?;
	let p = chibigrad::Plan::compile_unfused(&g, &[affine])?;
	assert_eq!(p.kernel_count, 2);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [3., 5., 7.]);
	let sum = g.reduce(x, Reduce::Sum, vec![0], false)?;
	let scaled = g.reduce(mul, Reduce::Sum, vec![0], false)?;
	let max = g.reduce(x, Reduce::Max, vec![0], false)?;
	assert_eq!(
		g.compile(&[sum, scaled])?
			.run(&[buffer(&[3], &[1., 2., 3.])])?
			.iter()
			.map(|b| b.data[0])
			.collect::<Vec<_>>(),
		[6., 12.]
	);
	assert_eq!(
		g.compile(&[max])?.run(&[buffer(&[3], &[-5., -2., -7.])])?[0].data,
		[-2.]
	);
	let bias = g.constant(10., vec![1])?;
	let biased = g.binary(Binary::Add, x, bias)?;
	assert_eq!(
		g.compile(&[biased])?.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data,
		[11., 12., 13.]
	);
	assert!(g.expand(x, vec![4]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test shapes_rangeify unfused_affine_and_reduce_max
```

## 21. 高水準の `x*2+1` が `[3,5,7]` になる。
<!-- todo:08-021 -->

`tests/shapes_rangeify.rs` の `unfused_affine_and_reduce_max` で、この経路の値または拒否条件を確認する。

`tests/shapes_rangeify.rs` の `fn unfused_affine_and_reduce_max(`の定義を次の全体で置き換えます。

```rust
// file="tests/shapes_rangeify.rs" mode="replace" item="fn unfused_affine_and_reduce_max("
fn unfused_affine_and_reduce_max() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let two = g.constant(2., vec![])?;
	let one = g.constant(1., vec![])?;
	let mul = g.binary(Binary::Mul, x, two)?;
	let affine = g.binary(Binary::Add, mul, one)?;
	let p = chibigrad::Plan::compile_unfused(&g, &[affine])?;
	assert_eq!(p.kernel_count, 2);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [3., 5., 7.]);
	let sum = g.reduce(x, Reduce::Sum, vec![0], false)?;
	let scaled = g.reduce(mul, Reduce::Sum, vec![0], false)?;
	let max = g.reduce(x, Reduce::Max, vec![0], false)?;
	assert_eq!(
		g.compile(&[sum, scaled])?
			.run(&[buffer(&[3], &[1., 2., 3.])])?
			.iter()
			.map(|b| b.data[0])
			.collect::<Vec<_>>(),
		[6., 12.]
	);
	assert_eq!(
		g.compile(&[max])?.run(&[buffer(&[3], &[-5., -2., -7.])])?[0].data,
		[-2.]
	);
	let bias = g.constant(10., vec![1])?;
	let biased = g.binary(Binary::Add, x, bias)?;
	assert_eq!(
		g.compile(&[biased])?.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data,
		[11., 12., 13.]
	);
	assert!(g.expand(x, vec![4]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test shapes_rangeify unfused_affine_and_reduce_max
```

## 22. affine の実行 kernel は二つで、中間 buffer が一つある。
<!-- todo:08-022 -->

`tests/shapes_rangeify.rs` の `unfused_affine_and_reduce_max` で、この経路の値または拒否条件を確認する。

`tests/shapes_rangeify.rs` の `fn unfused_affine_and_reduce_max(`の定義を次の全体で置き換えます。

```rust
// file="tests/shapes_rangeify.rs" mode="replace" item="fn unfused_affine_and_reduce_max("
fn unfused_affine_and_reduce_max() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let two = g.constant(2., vec![])?;
	let one = g.constant(1., vec![])?;
	let mul = g.binary(Binary::Mul, x, two)?;
	let affine = g.binary(Binary::Add, mul, one)?;
	let p = chibigrad::Plan::compile_unfused(&g, &[affine])?;
	assert_eq!(p.kernel_count, 2);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [3., 5., 7.]);
	let sum = g.reduce(x, Reduce::Sum, vec![0], false)?;
	let scaled = g.reduce(mul, Reduce::Sum, vec![0], false)?;
	let max = g.reduce(x, Reduce::Max, vec![0], false)?;
	assert_eq!(
		g.compile(&[sum, scaled])?
			.run(&[buffer(&[3], &[1., 2., 3.])])?
			.iter()
			.map(|b| b.data[0])
			.collect::<Vec<_>>(),
		[6., 12.]
	);
	assert_eq!(
		g.compile(&[max])?.run(&[buffer(&[3], &[-5., -2., -7.])])?[0].data,
		[-2.]
	);
	let bias = g.constant(10., vec![1])?;
	let biased = g.binary(Binary::Add, x, bias)?;
	assert_eq!(
		g.compile(&[biased])?.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data,
		[11., 12., 13.]
	);
	assert!(g.expand(x, vec![4]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test shapes_rangeify unfused_affine_and_reduce_max
```

## 23. 総和が `6`、`sum(x*2)` が `12`、MAX が `-2` になる。
<!-- todo:08-023 -->

`tests/shapes_rangeify.rs` の `unfused_affine_and_reduce_max` で、この経路の値または拒否条件を確認する。

`tests/shapes_rangeify.rs` の `fn unfused_affine_and_reduce_max(`の定義を次の全体で置き換えます。

```rust
// file="tests/shapes_rangeify.rs" mode="replace" item="fn unfused_affine_and_reduce_max("
fn unfused_affine_and_reduce_max() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let two = g.constant(2., vec![])?;
	let one = g.constant(1., vec![])?;
	let mul = g.binary(Binary::Mul, x, two)?;
	let affine = g.binary(Binary::Add, mul, one)?;
	let p = chibigrad::Plan::compile_unfused(&g, &[affine])?;
	assert_eq!(p.kernel_count, 2);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [3., 5., 7.]);
	let sum = g.reduce(x, Reduce::Sum, vec![0], false)?;
	let scaled = g.reduce(mul, Reduce::Sum, vec![0], false)?;
	let max = g.reduce(x, Reduce::Max, vec![0], false)?;
	assert_eq!(
		g.compile(&[sum, scaled])?
			.run(&[buffer(&[3], &[1., 2., 3.])])?
			.iter()
			.map(|b| b.data[0])
			.collect::<Vec<_>>(),
		[6., 12.]
	);
	assert_eq!(
		g.compile(&[max])?.run(&[buffer(&[3], &[-5., -2., -7.])])?[0].data,
		[-2.]
	);
	let bias = g.constant(10., vec![1])?;
	let biased = g.binary(Binary::Add, x, bias)?;
	assert_eq!(
		g.compile(&[biased])?.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data,
		[11., 12., 13.]
	);
	assert!(g.expand(x, vec![4]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test shapes_rangeify unfused_affine_and_reduce_max
```

## 24. scalar の出力 storage は一要素、shape は `[]` になる。
<!-- todo:08-024 -->

`tests/shapes_rangeify.rs` の `unfused_affine_and_reduce_max` で、この経路の値または拒否条件を確認する。

`tests/shapes_rangeify.rs` の `fn unfused_affine_and_reduce_max(`の定義を次の全体で置き換えます。

```rust
// file="tests/shapes_rangeify.rs" mode="replace" item="fn unfused_affine_and_reduce_max("
fn unfused_affine_and_reduce_max() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let two = g.constant(2., vec![])?;
	let one = g.constant(1., vec![])?;
	let mul = g.binary(Binary::Mul, x, two)?;
	let affine = g.binary(Binary::Add, mul, one)?;
	let p = chibigrad::Plan::compile_unfused(&g, &[affine])?;
	assert_eq!(p.kernel_count, 2);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [3., 5., 7.]);
	let sum = g.reduce(x, Reduce::Sum, vec![0], false)?;
	let scaled = g.reduce(mul, Reduce::Sum, vec![0], false)?;
	let max = g.reduce(x, Reduce::Max, vec![0], false)?;
	assert_eq!(
		g.compile(&[sum, scaled])?
			.run(&[buffer(&[3], &[1., 2., 3.])])?
			.iter()
			.map(|b| b.data[0])
			.collect::<Vec<_>>(),
		[6., 12.]
	);
	assert_eq!(
		g.compile(&[max])?.run(&[buffer(&[3], &[-5., -2., -7.])])?[0].data,
		[-2.]
	);
	let bias = g.constant(10., vec![1])?;
	let biased = g.binary(Binary::Add, x, bias)?;
	assert_eq!(
		g.compile(&[biased])?.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data,
		[11., 12., 13.]
	);
	assert!(g.expand(x, vec![4]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test shapes_rangeify unfused_affine_and_reduce_max
```

## 25. backend 入力に EXPAND と高水準 REDUCE が残らない。
<!-- todo:08-025 -->

`tests/shapes_rangeify.rs` の `unfused_affine_and_reduce_max` で、この経路の値または拒否条件を確認する。

`tests/shapes_rangeify.rs` の `fn unfused_affine_and_reduce_max(`の定義を次の全体で置き換えます。

```rust
// file="tests/shapes_rangeify.rs" mode="replace" item="fn unfused_affine_and_reduce_max("
fn unfused_affine_and_reduce_max() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let two = g.constant(2., vec![])?;
	let one = g.constant(1., vec![])?;
	let mul = g.binary(Binary::Mul, x, two)?;
	let affine = g.binary(Binary::Add, mul, one)?;
	let p = chibigrad::Plan::compile_unfused(&g, &[affine])?;
	assert_eq!(p.kernel_count, 2);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [3., 5., 7.]);
	let sum = g.reduce(x, Reduce::Sum, vec![0], false)?;
	let scaled = g.reduce(mul, Reduce::Sum, vec![0], false)?;
	let max = g.reduce(x, Reduce::Max, vec![0], false)?;
	assert_eq!(
		g.compile(&[sum, scaled])?
			.run(&[buffer(&[3], &[1., 2., 3.])])?
			.iter()
			.map(|b| b.data[0])
			.collect::<Vec<_>>(),
		[6., 12.]
	);
	assert_eq!(
		g.compile(&[max])?.run(&[buffer(&[3], &[-5., -2., -7.])])?[0].data,
		[-2.]
	);
	let bias = g.constant(10., vec![1])?;
	let biased = g.binary(Binary::Add, x, bias)?;
	assert_eq!(
		g.compile(&[biased])?.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data,
		[11., 12., 13.]
	);
	assert!(g.expand(x, vec![4]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test shapes_rangeify unfused_affine_and_reduce_max
```

## 26. 各 kernel の RANGE は閉じ、INDEX はその kernel の座標を使う。
<!-- todo:08-026 -->

`tests/shapes_rangeify.rs` の `unfused_affine_and_reduce_max` で、この経路の値または拒否条件を確認する。

`tests/shapes_rangeify.rs` の `fn unfused_affine_and_reduce_max(`の定義を次の全体で置き換えます。

```rust
// file="tests/shapes_rangeify.rs" mode="replace" item="fn unfused_affine_and_reduce_max("
fn unfused_affine_and_reduce_max() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let two = g.constant(2., vec![])?;
	let one = g.constant(1., vec![])?;
	let mul = g.binary(Binary::Mul, x, two)?;
	let affine = g.binary(Binary::Add, mul, one)?;
	let p = chibigrad::Plan::compile_unfused(&g, &[affine])?;
	assert_eq!(p.kernel_count, 2);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [3., 5., 7.]);
	let sum = g.reduce(x, Reduce::Sum, vec![0], false)?;
	let scaled = g.reduce(mul, Reduce::Sum, vec![0], false)?;
	let max = g.reduce(x, Reduce::Max, vec![0], false)?;
	assert_eq!(
		g.compile(&[sum, scaled])?
			.run(&[buffer(&[3], &[1., 2., 3.])])?
			.iter()
			.map(|b| b.data[0])
			.collect::<Vec<_>>(),
		[6., 12.]
	);
	assert_eq!(
		g.compile(&[max])?.run(&[buffer(&[3], &[-5., -2., -7.])])?[0].data,
		[-2.]
	);
	let bias = g.constant(10., vec![1])?;
	let biased = g.binary(Binary::Add, x, bias)?;
	assert_eq!(
		g.compile(&[biased])?.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data,
		[11., 12., 13.]
	);
	assert!(g.expand(x, vec![4]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test shapes_rangeify unfused_affine_and_reduce_max
```

## 27. shape 不一致と不正 EXPAND は実行前にエラーになる。
<!-- todo:08-027 -->

`tests/shapes_rangeify.rs` の `unfused_affine_and_reduce_max` で、この経路の値または拒否条件を確認する。

`tests/shapes_rangeify.rs` の `fn unfused_affine_and_reduce_max(`の定義を次の全体で置き換えます。

```rust
// file="tests/shapes_rangeify.rs" mode="replace" item="fn unfused_affine_and_reduce_max("
fn unfused_affine_and_reduce_max() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let two = g.constant(2., vec![])?;
	let one = g.constant(1., vec![])?;
	let mul = g.binary(Binary::Mul, x, two)?;
	let affine = g.binary(Binary::Add, mul, one)?;
	let p = chibigrad::Plan::compile_unfused(&g, &[affine])?;
	assert_eq!(p.kernel_count, 2);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [3., 5., 7.]);
	let sum = g.reduce(x, Reduce::Sum, vec![0], false)?;
	let scaled = g.reduce(mul, Reduce::Sum, vec![0], false)?;
	let max = g.reduce(x, Reduce::Max, vec![0], false)?;
	assert_eq!(
		g.compile(&[sum, scaled])?
			.run(&[buffer(&[3], &[1., 2., 3.])])?
			.iter()
			.map(|b| b.data[0])
			.collect::<Vec<_>>(),
		[6., 12.]
	);
	assert_eq!(
		g.compile(&[max])?.run(&[buffer(&[3], &[-5., -2., -7.])])?[0].data,
		[-2.]
	);
	let bias = g.constant(10., vec![1])?;
	let biased = g.binary(Binary::Add, x, bias)?;
	assert_eq!(
		g.compile(&[biased])?.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data,
		[11., 12., 13.]
	);
	assert!(g.expand(x, vec![4]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test shapes_rangeify unfused_affine_and_reduce_max
```

## 章末の累積確認

前章までの example と test も同じ checkout で再ビルドする。

```sh
# check
cargo test --quiet --all-targets
```
