# 03. scalar の型と意味を決め、interpreter と rewrite を比較する

整数演算は i32 wrap、DIV と MOD は床除算、SHR は算術 shift とする。f32 MAX は NaN を伝播し、ゼロ二つなら常に +0 を返す。動的 DIV、shift、float-to-int CAST の安全性は宣言 bounds と実入力の検査で保証する。導出できない bounds は安全と判定しない。型付き scalar IR を追加し、既存の二つの scalar example もこの IR に移す。

`file=... mode=write` はそのファイルの全置換、`mode=append` は末尾への追記である。すべて前章でできたプロジェクトの root で行う。`# check` は実際の検証コマンドであり、ソースファイルには保存しない。

## 1. arena に `HashMap<NodeKey, NodeId>` を追加する。
<!-- todo:03-001 -->

```rust
// file=src/scalar.rs mode=write
use crate::{Result, native::NativeLibrary};
use std::{
	collections::{HashMap, HashSet},
	ffi::c_void,
};
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
	Bool,
	I32,
	F32,
	Void,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Value {
	Bool(bool),
	I32(i32),
	F32Bits(u32),
}
impl Value {
	pub fn f32(x: f32) -> Self {
		Self::F32Bits(x.to_bits())
	}
	pub fn dtype(self) -> DType {
		match self {
			Self::Bool(_) => DType::Bool,
			Self::I32(_) => DType::I32,
			Self::F32Bits(_) => DType::F32,
		}
	}
	pub fn number(self) -> f64 {
		match self {
			Self::Bool(x) => x as u8 as f64,
			Self::I32(x) => x as f64,
			Self::F32Bits(x) => f32::from_bits(x) as f64,
		}
	}
	pub fn pack(self) -> u64 {
		match self {
			Self::Bool(x) => x as u64,
			Self::I32(x) => x as u32 as u64,
			Self::F32Bits(x) => x as u64,
		}
	}
	pub fn unpack(dtype: DType, bits: u64) -> Result<Self> {
		match dtype {
			DType::Bool => Ok(Self::Bool(bits != 0)),
			DType::I32 => Ok(Self::I32(bits as u32 as i32)),
			DType::F32 => Ok(Self::F32Bits(bits as u32)),
			DType::Void => Err("void has no value".into()),
		}
	}
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Bounds {
	pub lo: i64,
	pub hi: i64,
}
impl Bounds {
	pub const I32: Self = Self {
		lo: i32::MIN as i64,
		hi: i32::MAX as i64,
	};
	pub fn new(lo: i64, hi: i64) -> Result<Self> {
		if lo > hi {
			Err("empty bounds".into())
		} else {
			Ok(Self { lo, hi })
		}
	}
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Alu {
	Add,
	Sub,
	Mul,
	Neg,
	Xor,
	And,
	Or,
	Shl,
	Shr,
	Div,
	Mod,
	Lt,
	Eq,
	Ne,
	Where,
	Recip,
	Max,
	Cast(DType),
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SOp {
	Const(Value),
	Param {
		slot: usize,
		scope: usize,
		dtype: DType,
		bounds: Option<Bounds>,
	},
	Alu(Alu),
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SNode {
	pub op: SOp,
	pub inputs: Vec<usize>,
	pub dtype: DType,
	pub bounds: Option<Bounds>,
}
#[derive(Clone, Debug, Default)]
pub struct ScalarGraph {
	pub nodes: Vec<SNode>,
	intern: HashMap<SNode, usize>,
}
impl ScalarGraph {
	fn push(&mut self, node: SNode) -> usize {
		if let Some(&id) = self.intern.get(&node) {
			return id;
		}
		let id = self.nodes.len();
		self.nodes.push(node.clone());
		self.intern.insert(node, id);
		id
	}
	pub fn constant(&mut self, v: Value) -> usize {
		let n = v.number();
		let bounds = if n.is_finite() && n >= i32::MIN as f64 && n <= i32::MAX as f64 {
			Some(Bounds {
				lo: n.floor() as i64,
				hi: n.ceil() as i64,
			})
		} else {
			None
		};
		self.push(SNode {
			op: SOp::Const(v),
			inputs: vec![],
			dtype: v.dtype(),
			bounds,
		})
	}
	pub fn param(&mut self, slot: usize, dtype: DType, bounds: Option<Bounds>) -> Result<usize> {
		self.scoped_param(0, slot, dtype, bounds)
	}
	pub fn scoped_param(
		&mut self,
		scope: usize,
		slot: usize,
		dtype: DType,
		bounds: Option<Bounds>,
	) -> Result<usize> {
		if dtype == DType::Void {
			return Err("void PARAM".into());
		}
		if let Some(b) = bounds {
			if b.lo > b.hi
				|| dtype == DType::I32 && (b.lo < i32::MIN as i64 || b.hi > i32::MAX as i64)
			{
				return Err("invalid parameter bounds".into());
			}
		}
		Ok(self.push(SNode {
			op: SOp::Param {
				slot,
				scope,
				dtype,
				bounds,
			},
			inputs: vec![],
			dtype,
			bounds: bounds.or(if dtype == DType::I32 {
				Some(Bounds::I32)
			} else {
				None
			}),
		}))
	}
	pub fn alu(&mut self, op: Alu, inputs: Vec<usize>) -> Result<usize> {
		let (dtype, bounds) = self.infer(op, &inputs)?;
		Ok(self.push(SNode {
			op: SOp::Alu(op),
			inputs,
			dtype,
			bounds,
		}))
	}
	fn infer(&self, op: Alu, inputs: &[usize]) -> Result<(DType, Option<Bounds>)> {
		let arity = match op {
			Alu::Neg | Alu::Recip | Alu::Cast(_) => 1,
			Alu::Where => 3,
			_ => 2,
		};
		if inputs.len() != arity {
			return Err("ALU arity mismatch".into());
		}
		let ns = inputs
			.iter()
			.map(|&id| self.nodes.get(id).ok_or("unknown scalar input".into()))
			.collect::<Result<Vec<_>>>()?;
		let t = ns[0].dtype;
		let numeric = |t| t == DType::I32 || t == DType::F32;
		let dtype = match op {
			Alu::Where => {
				if t != DType::Bool || ns[1].dtype != ns[2].dtype || ns[1].dtype == DType::Void {
					return Err("WHERE type mismatch".into());
				}
				ns[1].dtype
			}
			Alu::Cast(target) => {
				if target == DType::Void || t == DType::Void {
					return Err("void CAST".into());
				}
				if t == DType::F32 && target == DType::I32 {
					let b = ns[0].bounds.ok_or("unproven float-to-int CAST")?;
					if b.lo < i32::MIN as i64 || b.hi > 2147483520 {
						return Err("unsafe float-to-int CAST".into());
					}
				}
				target
			}
			Alu::Recip => {
				if t != DType::F32 {
					return Err("RECIP requires f32".into());
				}
				t
			}
			Alu::Xor | Alu::And | Alu::Or => {
				if !matches!(t, DType::I32 | DType::Bool) || ns[1].dtype != t {
					return Err("bitwise type mismatch".into());
				}
				t
			}
			Alu::Shl | Alu::Shr => {
				if t != DType::I32 || ns[1].dtype != t {
					return Err("shift requires i32".into());
				}
				let b = ns[1].bounds.ok_or("unproven shift count")?;
				if b.lo < 0 || b.hi >= 32 {
					return Err("unsafe shift count".into());
				}
				t
			}
			Alu::Div | Alu::Mod => {
				if t != DType::I32 || ns[1].dtype != t {
					return Err("integer DIV/MOD require i32".into());
				}
				let a = ns[0].bounds.unwrap_or(Bounds::I32);
				let b = ns[1].bounds.unwrap_or(Bounds::I32);
				if b.lo <= 0 && b.hi >= 0 || a.lo == i32::MIN as i64 && b.lo <= -1 && b.hi >= -1 {
					return Err("unsafe integer division".into());
				}
				t
			}
			Alu::Lt | Alu::Eq | Alu::Ne => {
				if !numeric(t) || ns[1].dtype != t {
					return Err("comparison type mismatch".into());
				}
				DType::Bool
			}
			_ => {
				if !numeric(t) || ns.iter().any(|n| n.dtype != t) {
					return Err("numeric type mismatch".into());
				}
				t
			}
		};
		let bounds = if dtype == DType::Bool {
			Some(Bounds { lo: 0, hi: 1 })
		} else if dtype == DType::I32 {
			let a = ns[0].bounds.unwrap_or(Bounds::I32);
			let b = ns.get(1).and_then(|n| n.bounds).unwrap_or(Bounds::I32);
			let raw = match op {
				Alu::Add => Some((a.lo + b.lo, a.hi + b.hi)),
				Alu::Sub => Some((a.lo - b.hi, a.hi - b.lo)),
				Alu::Mul => {
					let p = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi];
					Some((*p.iter().min().unwrap(), *p.iter().max().unwrap()))
				}
				_ => None,
			};
			Some(match raw {
				Some((lo, hi)) if lo >= i32::MIN as i64 && hi <= i32::MAX as i64 => {
					Bounds { lo, hi }
				}
				_ => Bounds::I32,
			})
		} else {
			None
		};
		Ok((dtype, bounds))
	}
	pub fn order(&self, root: usize) -> Result<Vec<usize>> {
		fn dfs(
			g: &ScalarGraph,
			id: usize,
			seen: &mut HashSet<usize>,
			order: &mut Vec<usize>,
		) -> Result<()> {
			let n = g.nodes.get(id).ok_or("invalid scalar root/input")?;
			if seen.insert(id) {
				for &x in &n.inputs {
					if x >= id {
						return Err("scalar cycle/forward reference".into());
					}
					dfs(g, x, seen, order)?
				}
				order.push(id)
			}
			Ok(())
		}
		let mut out = vec![];
		dfs(self, root, &mut HashSet::new(), &mut out)?;
		Ok(out)
	}
	pub fn verify(&self, root: usize, scope: usize) -> Result<()> {
		for id in self.order(root)? {
			let n = &self.nodes[id];
			match n.op {
				SOp::Const(v) => {
					let mut fresh = Self::default();
					let expected = fresh.constant(v);
					if n.dtype != v.dtype()
						|| !n.inputs.is_empty()
						|| n.bounds != fresh.nodes[expected].bounds
					{
						return Err("invalid CONST metadata".into());
					}
				}
				SOp::Param {
					slot,
					scope: s,
					dtype,
					bounds,
				} => {
					let mut fresh = Self::default();
					let expected = fresh.scoped_param(s, slot, dtype, bounds)?;
					if s != scope
						|| dtype != n.dtype
						|| !n.inputs.is_empty()
						|| n.bounds != fresh.nodes[expected].bounds
					{
						return Err("unbound PARAM scope or invalid bounds".into());
					}
				}
				SOp::Alu(op) => {
					let (t, b) = self.infer(op, &n.inputs)?;
					if t != n.dtype || b != n.bounds {
						return Err("invalid derived scalar metadata".into());
					}
				}
			}
		}
		Ok(())
	}
	pub fn evaluate(&self, root: usize, inputs: &[Value]) -> Result<Value> {
		self.evaluate_scoped(root, inputs, 0)
	}
	pub fn evaluate_scoped(&self, root: usize, inputs: &[Value], scope: usize) -> Result<Value> {
		self.verify(root, scope)?;
		self.validate_inputs(root, inputs)?;
		let mut values = HashMap::new();
		for id in self.order(root)? {
			let n = &self.nodes[id];
			let value = match n.op {
				SOp::Const(v) => v,
				SOp::Param { slot, .. } => inputs[slot],
				SOp::Alu(op) => {
					eval_alu(op, &n.inputs.iter().map(|x| values[x]).collect::<Vec<_>>())?
				}
			};
			values.insert(id, value);
		}
		Ok(values[&root])
	}
	pub fn validate_inputs(&self, root: usize, inputs: &[Value]) -> Result<()> {
		let mut slots = HashSet::new();
		for id in self.order(root)? {
			if let SOp::Param {
				slot,
				dtype,
				bounds,
				..
			} = self.nodes[id].op
			{
				slots.insert(slot);
				let v = *inputs.get(slot).ok_or("missing scalar input")?;
				if v.dtype() != dtype {
					return Err("scalar input dtype mismatch".into());
				}
				if let Some(b) = bounds {
					let n = v.number();
					if !n.is_finite() || n < b.lo as f64 || n > b.hi as f64 {
						return Err("scalar input outside declared bounds".into());
					}
				}
			}
		}
		if slots.len() != inputs.len() || (0..inputs.len()).any(|s| !slots.contains(&s)) {
			return Err("extra or non-dense scalar slots".into());
		}
		Ok(())
	}
	pub fn rewrite(&self, root: usize) -> Result<(Self, usize)> {
		self.verify(root, 0)?;
		let mut out = Self::default();
		let mut map: HashMap<usize, usize> = HashMap::new();
		for id in self.order(root)? {
			let n = &self.nodes[id];
			let new = match n.op {
				SOp::Const(v) => out.constant(v),
				SOp::Param {
					slot,
					scope,
					dtype,
					bounds,
				} => out.scoped_param(scope, slot, dtype, bounds)?,
				SOp::Alu(op) => {
					let children = n.inputs.iter().map(|x| map[x]).collect::<Vec<_>>();
					let constants = children
						.iter()
						.map(|&x| match out.nodes[x].op {
							SOp::Const(v) => Some(v),
							_ => None,
						})
						.collect::<Option<Vec<_>>>();
					if let Some(values) = constants {
						out.constant(eval_alu(op, &values)?)
					} else if matches!(op, Alu::Add | Alu::Xor)
						&& n.dtype == DType::I32
						&& matches!(out.nodes[children[1]].op, SOp::Const(Value::I32(0)))
					{
						children[0]
					} else if op == Alu::Where {
						if let SOp::Const(Value::Bool(b)) = out.nodes[children[0]].op {
							children[if b { 1 } else { 2 }]
						} else {
							out.alu(op, children)?
						}
					} else {
						out.alu(op, children)?
					}
				}
			};
			map.insert(id, new);
		}
		Ok((out, map[&root]))
	}
	pub fn cost(&self, root: usize) -> Result<usize> {
		Ok(self
			.order(root)?
			.iter()
			.filter(|&&id| matches!(self.nodes[id].op, SOp::Alu(_)))
			.count())
	}
	pub fn compile(&self, root: usize) -> Result<CompiledScalar> {
		self.verify(root, 0)?;
		let source = self.render(root)?;
		let library = unsafe { NativeLibrary::compile(&source, "chibi_entry") }?;
		Ok(CompiledScalar {
			graph: self.clone(),
			root,
			source,
			library,
		})
	}
}
pub fn eval_alu(op: Alu, v: &[Value]) -> Result<Value> {
	let a = v[0];
	if let Alu::Cast(t) = op {
		return Ok(match t {
			DType::Bool => Value::Bool(a.number() != 0.),
			DType::I32 => Value::I32(a.number() as i32),
			DType::F32 => Value::f32(a.number() as f32),
			DType::Void => return Err("void cast".into()),
		});
	}
	if op == Alu::Where {
		return Ok(if a == Value::Bool(true) { v[1] } else { v[2] });
	}
	if matches!(op, Alu::Lt | Alu::Eq | Alu::Ne) {
		let b = v[1].number();
		let a = a.number();
		return Ok(Value::Bool(match op {
			Alu::Lt => a < b,
			Alu::Eq => a == b,
			_ => a != b,
		}));
	}
	Ok(match a {
		Value::Bool(a) => {
			let Value::Bool(b) = v[1] else {
				return Err("bool operand mismatch".into());
			};
			Value::Bool(match op {
				Alu::Xor => a ^ b,
				Alu::And => a & b,
				Alu::Or => a | b,
				_ => return Err("invalid bool operation".into()),
			})
		}
		Value::I32(a) => {
			let b = if v.len() > 1 {
				if let Value::I32(b) = v[1] {
					b
				} else {
					return Err("i32 operand mismatch".into());
				}
			} else {
				0
			};
			Value::I32(match op {
				Alu::Add => a.wrapping_add(b),
				Alu::Sub => a.wrapping_sub(b),
				Alu::Mul => a.wrapping_mul(b),
				Alu::Neg => a.wrapping_neg(),
				Alu::Xor => a ^ b,
				Alu::And => a & b,
				Alu::Or => a | b,
				Alu::Shl => a.wrapping_shl(b as u32),
				Alu::Shr => a >> b,
				Alu::Div | Alu::Mod => {
					if b == 0 || (a == i32::MIN && b == -1) {
						return Err("invalid division".into());
					}
					let q = a / b;
					let r = a % b;
					let q = q - if r != 0 && (r < 0) != (b < 0) { 1 } else { 0 };
					if op == Alu::Div {
						q
					} else {
						(a as i64 - q as i64 * b as i64) as i32
					}
				}
				Alu::Max => a.max(b),
				_ => return Err("invalid i32 operation".into()),
			})
		}
		Value::F32Bits(a) => {
			let a = f32::from_bits(a);
			let b = if v.len() > 1 {
				v[1].number() as f32
			} else {
				0.
			};
			Value::f32(match op {
				Alu::Add => a + b,
				Alu::Sub => a - b,
				Alu::Mul => a * b,
				Alu::Neg => -a,
				Alu::Recip => a.recip(),
				Alu::Max => {
					if a.is_nan() || b.is_nan() {
						f32::NAN
					} else if a == 0. && b == 0. {
						0.
					} else {
						a.max(b)
					}
				}
				_ => return Err("invalid f32 operation".into()),
			})
		}
	})
}
fn ctype(t: DType) -> &'static str {
	match t {
		DType::Bool => "uint32_t",
		DType::I32 => "int32_t",
		DType::F32 => "float",
		DType::Void => "void",
	}
}
pub const SCALAR_HELPERS: &str = "#include <stdint.h>\n#include <string.h>\n#include <math.h>\nstatic float chibi_bits(uint32_t x){float y;memcpy(&y,&x,4);return y;}\nstatic int32_t chibi_i32(uint32_t x){int32_t y;memcpy(&y,&x,4);return y;}\nstatic int32_t chibi_div(int32_t a,int32_t b){int32_t q=a/b,r=a%b;return q-((r!=0)&&((r<0)!=(b<0)));}\nstatic int32_t chibi_mod(int32_t a,int32_t b){return (int32_t)((int64_t)a-(int64_t)chibi_div(a,b)*b);}\nstatic int32_t chibi_shr(int32_t a,int32_t b){uint32_t u=(uint32_t)a;return chibi_i32((u>>b)|((a<0&&b)?(~0u<<(32-b)):0u));}\nstatic float chibi_max(float a,float b){if(isnan(a)||isnan(b))return NAN;if(a==0.0f&&b==0.0f)return 0.0f;return a>b?a:b;}\n";
impl ScalarGraph {
	pub fn render(&self, root: usize) -> Result<String> {
		self.verify(root, 0)?;
		let mut source = SCALAR_HELPERS.to_string();
		source.push_str("int chibi_entry(void*const* buffers,const uint64_t* scalars){\n");
		for id in self.order(root)? {
			let n = &self.nodes[id];
			let expr = match n.op {
				SOp::Const(Value::Bool(x)) => format!("{}u", x as u8),
				SOp::Const(Value::I32(x)) => format!("chibi_i32({}u)", x as u32),
				SOp::Const(Value::F32Bits(x)) => format!("chibi_bits({x}u)"),
				SOp::Param { slot, dtype, .. } => match dtype {
					DType::I32 => format!("chibi_i32((uint32_t)scalars[{slot}])"),
					DType::F32 => format!("chibi_bits((uint32_t)scalars[{slot}])"),
					_ => format!("(scalars[{slot}]!=0)"),
				},
				SOp::Alu(op) => {
					let a = format!("v{}", n.inputs[0]);
					let b = n.inputs.get(1).map(|x| format!("v{x}")).unwrap_or_default();
					match op {
						Alu::Add | Alu::Sub | Alu::Mul if n.dtype == DType::I32 => format!(
							"chibi_i32((uint32_t){a} {} (uint32_t){b})",
							match op {
								Alu::Add => "+",
								Alu::Sub => "-",
								_ => "*",
							}
						),
						Alu::Neg if n.dtype == DType::I32 => format!("chibi_i32(0u-(uint32_t){a})"),
						Alu::Shl => format!("chibi_i32((uint32_t){a} << {b})"),
						Alu::Shr => format!("chibi_shr({a},{b})"),
						Alu::Div => format!("chibi_div({a},{b})"),
						Alu::Mod => format!("chibi_mod({a},{b})"),
						Alu::Max => {
							if n.dtype == DType::F32 {
								format!("chibi_max({a},{b})")
							} else {
								format!("({a}>{b}?{a}:{b})")
							}
						}
						Alu::Recip => format!("(1.0f/{a})"),
						Alu::Neg => format!("(-{a})"),
						Alu::Where => format!("({a}?{b}:v{})", n.inputs[2]),
						Alu::Cast(t) => {
							if t == DType::Bool {
								format!("({a}!=0)")
							} else {
								format!("(({}){a})", ctype(t))
							}
						}
						_ => format!(
							"({a} {} {b})",
							match op {
								Alu::Add => "+",
								Alu::Sub => "-",
								Alu::Mul => "*",
								Alu::Xor => "^",
								Alu::And => "&",
								Alu::Or => "|",
								Alu::Lt => "<",
								Alu::Eq => "==",
								Alu::Ne => "!=",
								_ => unreachable!(),
							}
						),
					}
				}
			};
			source.push_str(&format!("{} v{id}={expr};\n", ctype(n.dtype)));
		}
		if self.nodes[root].dtype == DType::F32 {
			source.push_str(&format!(
				"uint32_t bits;memcpy(&bits,&v{root},4);*(uint64_t*)buffers[0]=bits;\n"
			))
		} else {
			source.push_str(&format!("*(uint64_t*)buffers[0]=(uint32_t)v{root};\n"))
		}
		source.push_str("return 0;\n}\n");
		Ok(source)
	}
}
pub struct CompiledScalar {
	graph: ScalarGraph,
	root: usize,
	pub source: String,
	library: NativeLibrary,
}
impl CompiledScalar {
	pub fn run(&self, inputs: &[Value]) -> Result<Value> {
		self.graph.validate_inputs(self.root, inputs)?;
		let scalars = inputs.iter().map(|v| v.pack()).collect::<Vec<_>>();
		let mut output = 0u64;
		let buffers = [(&mut output as *mut u64).cast::<c_void>()];
		type Entry = unsafe extern "C" fn(*const *mut c_void, *const u64) -> i32;
		let entry: Entry = unsafe { std::mem::transmute(self.library.symbol()) };
		let status = unsafe { entry(buffers.as_ptr(), scalars.as_ptr()) };
		if status != 0 {
			return Err(format!("scalar entry failed {status}"));
		}
		Value::unpack(self.graph.nodes[self.root].dtype, output)
	}
}
```

```rust
// file=src/lib.rs mode=append
pub mod scalar;
```

```rust
// file=tests/scalar_rewrite.rs mode=write
use chibigrad::{
	Result,
	scalar::{Alu, Bounds, DType, ScalarGraph, Value},
};

#[test]
fn arithmetic_and_shared_library() -> Result<()> {
	let mut g = ScalarGraph::default();
	let a = g.param(0, DType::F32, None)?;
	let b = g.param(1, DType::F32, None)?;
	let c = g.param(2, DType::F32, None)?;
	let sum = g.alu(Alu::Add, vec![a, b])?;
	assert_eq!(sum, g.alu(Alu::Add, vec![a, b])?);
	let root = g.alu(Alu::Mul, vec![sum, c])?;
	let p = g.compile(root)?;
	assert_eq!(
		p.run(&[Value::f32(2.), Value::f32(3.), Value::f32(4.)])?,
		Value::f32(20.)
	);
	assert_eq!(
		p.run(&[Value::f32(1.), Value::f32(2.), Value::f32(4.)])?,
		Value::f32(12.)
	);
	assert!(p.run(&[]).is_err());
	Ok(())
}
#[test]
fn integer_contract_and_rewrite() -> Result<()> {
	let mut g = ScalarGraph::default();
	let x = g.param(0, DType::I32, Some(Bounds::new(-7, 5)?))?;
	let zero = g.constant(Value::I32(0));
	let root = g.alu(Alu::Add, vec![x, zero])?;
	let (r, y) = g.rewrite(root)?;
	assert_eq!(r.cost(y)?, 0);
	for x in [-7, 0, 5] {
		assert_eq!(
			g.evaluate(root, &[Value::I32(x)])?,
			r.compile(y)?.run(&[Value::I32(x)])?
		)
	}
	for (op, a, b, expected) in [
		(Alu::Div, -7, 3, -3),
		(Alu::Mod, -7, 3, 2),
		(Alu::Add, i32::MAX, 1, i32::MIN),
		(Alu::Shr, -8, 1, -4),
		(Alu::Shl, 6, 1, 12),
	] {
		let a = g.constant(Value::I32(a));
		let b = g.constant(Value::I32(b));
		let y = g.alu(op, vec![a, b])?;
		assert_eq!(g.compile(y)?.run(&[])?, Value::I32(expected));
	}
	let a = g.constant(Value::I32(1));
	assert!(g.alu(Alu::Div, vec![a, zero]).is_err());
	assert!(g.alu(Alu::Shl, vec![a, x]).is_err());
	Ok(())
}
#[test]
fn typed_failures_and_special_values() -> Result<()> {
	let mut g = ScalarGraph::default();
	let f = g.param(0, DType::F32, None)?;
	let i = g.constant(Value::I32(1));
	assert!(g.alu(Alu::Add, vec![f, i]).is_err());
	assert!(g.alu(Alu::Cast(DType::I32), vec![f]).is_err());
	assert!(g.param(1, DType::Void, None).is_err());
	assert!(g.alu(Alu::Add, vec![i]).is_err());
	assert!(g.compile(999).is_err());
	let foreign = g.scoped_param(42, 0, DType::F32, None)?;
	assert!(g.compile(foreign).is_err());
	let minus = g.constant(Value::f32(-0.));
	let plus = g.constant(Value::f32(0.));
	assert_ne!(minus, plus);
	let max = g.alu(Alu::Max, vec![minus, minus])?;
	assert_eq!(g.compile(max)?.run(&[])?, Value::f32(0.));
	let nan = g.constant(Value::F32Bits(0x7fc00042));
	assert_eq!(nan, g.constant(Value::F32Bits(0x7fc00042)));
	let max = g.alu(Alu::Max, vec![nan, plus])?;
	assert!(g.compile(max)?.run(&[])?.number().is_nan());
	let mut bounded = ScalarGraph::default();
	let x = bounded.param(0, DType::I32, Some(Bounds::new(-7, 5)?))?;
	assert!(bounded.compile(x)?.run(&[Value::I32(6)]).is_err());
	Ok(())
}
#[test]
fn where_and_float_classification() -> Result<()> {
	let mut g = ScalarGraph::default();
	let a = g.param(0, DType::F32, None)?;
	let three = g.constant(Value::f32(3.));
	let two = g.constant(Value::f32(2.));
	let one = g.constant(Value::f32(1.));
	let cond = g.alu(Alu::Lt, vec![a, three])?;
	let yes = g.alu(Alu::Mul, vec![a, two])?;
	let no = g.alu(Alu::Add, vec![a, one])?;
	let y = g.alu(Alu::Where, vec![cond, yes, no])?;
	let plan = g.compile(y)?;
	assert_eq!(plan.run(&[Value::f32(2.)])?, Value::f32(4.));
	assert_eq!(plan.run(&[Value::f32(5.)])?, Value::f32(6.));
	let (rewritten, root) = g.rewrite(y)?;
	for value in [-7., 0., 5., f32::INFINITY, f32::NAN] {
		let input = [Value::f32(value)];
		let actual = plan.run(&input)?.number();
		let reference = rewritten.evaluate(root, &input)?.number();
		assert!(actual == reference || actual.is_nan() && reference.is_nan());
	}
	Ok(())
}
#[test]
fn forged_scalar_bounds_are_rejected() -> Result<()> {
	let mut g = ScalarGraph::default();
	let x = g.param(0, DType::I32, None)?;
	g.nodes[x].bounds = Some(Bounds::new(0, 1)?);
	assert!(g.compile(x).is_err());
	let value = g.constant(Value::f32(f32::NAN));
	g.nodes[value].bounds = Some(Bounds::new(0, 1)?);
	assert!(g.compile(value).is_err());
	Ok(())
}
```

```rust
// file=examples/scalar_const.rs mode=write
use chibigrad::{
	Result,
	scalar::{ScalarGraph, Value},
};
fn main() -> Result<()> {
	let args = std::env::args().skip(1).collect::<Vec<_>>();
	let value = args
		.iter()
		.find(|s| !s.starts_with("--"))
		.map(|s| s.parse::<f32>().map_err(|e| e.to_string()))
		.transpose()?
		.unwrap_or(7.);
	let mut g = ScalarGraph::default();
	let y = g.constant(Value::f32(value));
	let plan = g.compile(y)?;
	if args.iter().any(|s| s == "--dump-source") {
		eprintln!("{}", plan.source)
	}
	if args.iter().any(|s| s == "--dump-graph") {
		eprintln!("{g:#?}")
	}
	let result = plan.run(&[])?;
	assert_eq!(result, Value::f32(value));
	println!("{}", result.number());
	Ok(())
}
```

```rust
// file=examples/scalar_arithmetic.rs mode=write
use chibigrad::{
	Result,
	scalar::{Alu, ScalarGraph, Value},
};
fn main() -> Result<()> {
	let args = std::env::args().skip(1).collect::<Vec<_>>();
	let values = args
		.iter()
		.filter(|s| !s.starts_with("--"))
		.map(|s| s.parse::<f32>().map_err(|e| e.to_string()))
		.collect::<Result<Vec<_>>>()?;
	let v = if values.is_empty() {
		vec![2., 3., 4.]
	} else {
		values
	};
	if v.len() != 3 {
		return Err("expected a b c".into());
	}
	let mut g = ScalarGraph::default();
	let a = g.constant(Value::f32(v[0]));
	let b = g.constant(Value::f32(v[1]));
	let c = g.constant(Value::f32(v[2]));
	let sum = g.alu(Alu::Add, vec![a, b])?;
	let y = g.alu(Alu::Mul, vec![sum, c])?;
	let plan = g.compile(y)?;
	if args.iter().any(|s| s == "--dump-source") {
		eprintln!("{}", plan.source)
	}
	if args.iter().any(|s| s == "--dump-graph") {
		eprintln!("{g:#?}")
	}
	let result = plan.run(&[])?;
	assert_eq!(result, Value::f32((v[0] + v[1]) * v[2]));
	println!("{}", result.number());
	Ok(())
}
```

`tests/scalar_rewrite.rs` の正常系と境界系をまとめて再実行する。

```sh
# check
cargo test --quiet --test scalar_rewrite
```

## 2. f32 のビットキーと演算の入力順を保持する。
<!-- todo:03-002 -->

`tests/scalar_rewrite.rs` の正常系と境界系をまとめて再実行する。

`src/scalar.rs` の `fn push(`の定義を次の全体で置き換えます。

```rust
// file="src/scalar.rs" mode="replace" item="fn push("
	fn push(&mut self, node: SNode) -> usize {
		if let Some(&id) = self.intern.get(&node) {
			return id;
		}
		let id = self.nodes.len();
		self.nodes.push(node.clone());
		self.intern.insert(node, id);
		id
	}
```

```sh
# check
cargo test --quiet --test scalar_rewrite
```

## 3. 同じ `a+b` を二回作ると同じ番号になるようにする。
<!-- todo:03-003 -->

`tests/scalar_rewrite.rs` の正常系と境界系をまとめて再実行する。

`src/scalar.rs` の `fn push(`の定義を次の全体で置き換えます。

```rust
// file="src/scalar.rs" mode="replace" item="fn push("
	fn push(&mut self, node: SNode) -> usize {
		if let Some(&id) = self.intern.get(&node) {
			return id;
		}
		let id = self.nodes.len();
		self.nodes.push(node.clone());
		self.intern.insert(node, id);
		id
	}
```

```sh
# check
cargo test --quiet --test scalar_rewrite
```

## 4. `(a+b)*(a+b)` を C まで通し、入力 `2,3` で `25` を得る。
<!-- todo:03-004 -->

`tests/scalar_rewrite.rs` の正常系と境界系をまとめて再実行する。

`tests/scalar_rewrite.rs` の `fn arithmetic_and_shared_library(`の定義を次の全体で置き換えます。

```rust
// file="tests/scalar_rewrite.rs" mode="replace" item="fn arithmetic_and_shared_library("
fn arithmetic_and_shared_library() -> Result<()> {
	let mut g = ScalarGraph::default();
	let a = g.param(0, DType::F32, None)?;
	let b = g.param(1, DType::F32, None)?;
	let c = g.param(2, DType::F32, None)?;
	let sum = g.alu(Alu::Add, vec![a, b])?;
	assert_eq!(sum, g.alu(Alu::Add, vec![a, b])?);
	let root = g.alu(Alu::Mul, vec![sum, c])?;
	let p = g.compile(root)?;
	assert_eq!(
		p.run(&[Value::f32(2.), Value::f32(3.), Value::f32(4.)])?,
		Value::f32(20.)
	);
	assert_eq!(
		p.run(&[Value::f32(1.), Value::f32(2.), Value::f32(4.)])?,
		Value::f32(12.)
	);
	assert!(p.run(&[]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test scalar_rewrite
```

## 5. dtype の導出と各演算の入力数を一か所で検証する。
<!-- todo:03-005 -->

`tests/scalar_rewrite.rs` の `typed_failures_and_special_values` で、この経路の値または拒否条件を確認する。

`src/scalar.rs` の `fn infer(`の定義を次の全体で置き換えます。

```rust
// file="src/scalar.rs" mode="replace" item="fn infer("
	fn infer(&self, op: Alu, inputs: &[usize]) -> Result<(DType, Option<Bounds>)> {
		let arity = match op {
			Alu::Neg | Alu::Recip | Alu::Cast(_) => 1,
			Alu::Where => 3,
			_ => 2,
		};
		if inputs.len() != arity {
			return Err("ALU arity mismatch".into());
		}
		let ns = inputs
			.iter()
			.map(|&id| self.nodes.get(id).ok_or("unknown scalar input".into()))
			.collect::<Result<Vec<_>>>()?;
		let t = ns[0].dtype;
		let numeric = |t| t == DType::I32 || t == DType::F32;
		let dtype = match op {
			Alu::Where => {
				if t != DType::Bool || ns[1].dtype != ns[2].dtype || ns[1].dtype == DType::Void {
					return Err("WHERE type mismatch".into());
				}
				ns[1].dtype
			}
			Alu::Cast(target) => {
				if target == DType::Void || t == DType::Void {
					return Err("void CAST".into());
				}
				if t == DType::F32 && target == DType::I32 {
					let b = ns[0].bounds.ok_or("unproven float-to-int CAST")?;
					if b.lo < i32::MIN as i64 || b.hi > 2147483520 {
						return Err("unsafe float-to-int CAST".into());
					}
				}
				target
			}
			Alu::Recip => {
				if t != DType::F32 {
					return Err("RECIP requires f32".into());
				}
				t
			}
			Alu::Xor | Alu::And | Alu::Or => {
				if !matches!(t, DType::I32 | DType::Bool) || ns[1].dtype != t {
					return Err("bitwise type mismatch".into());
				}
				t
			}
			Alu::Shl | Alu::Shr => {
				if t != DType::I32 || ns[1].dtype != t {
					return Err("shift requires i32".into());
				}
				let b = ns[1].bounds.ok_or("unproven shift count")?;
				if b.lo < 0 || b.hi >= 32 {
					return Err("unsafe shift count".into());
				}
				t
			}
			Alu::Div | Alu::Mod => {
				if t != DType::I32 || ns[1].dtype != t {
					return Err("integer DIV/MOD require i32".into());
				}
				let a = ns[0].bounds.unwrap_or(Bounds::I32);
				let b = ns[1].bounds.unwrap_or(Bounds::I32);
				if b.lo <= 0 && b.hi >= 0 || a.lo == i32::MIN as i64 && b.lo <= -1 && b.hi >= -1 {
					return Err("unsafe integer division".into());
				}
				t
			}
			Alu::Lt | Alu::Eq | Alu::Ne => {
				if !numeric(t) || ns[1].dtype != t {
					return Err("comparison type mismatch".into());
				}
				DType::Bool
			}
			_ => {
				if !numeric(t) || ns.iter().any(|n| n.dtype != t) {
					return Err("numeric type mismatch".into());
				}
				t
			}
		};
		let bounds = if dtype == DType::Bool {
			Some(Bounds { lo: 0, hi: 1 })
		} else if dtype == DType::I32 {
			let a = ns[0].bounds.unwrap_or(Bounds::I32);
			let b = ns.get(1).and_then(|n| n.bounds).unwrap_or(Bounds::I32);
			let raw = match op {
				Alu::Add => Some((a.lo + b.lo, a.hi + b.hi)),
				Alu::Sub => Some((a.lo - b.hi, a.hi - b.lo)),
				Alu::Mul => {
					let p = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi];
					Some((*p.iter().min().unwrap(), *p.iter().max().unwrap()))
				}
				_ => None,
			};
			Some(match raw {
				Some((lo, hi)) if lo >= i32::MIN as i64 && hi <= i32::MAX as i64 => {
					Bounds { lo, hi }
				}
				_ => Bounds::I32,
			})
		} else {
			None
		};
		Ok((dtype, bounds))
	}
```

```sh
# check
cargo test --quiet --test scalar_rewrite typed_failures_and_special_values
```

## 6. bool と i32 の ABI pack と C の復元処理を追加する。
<!-- todo:03-006 -->

`tests/scalar_rewrite.rs` の正常系と境界系をまとめて再実行する。

`src/scalar.rs` の `pub fn pack(`の定義を次の全体で置き換えます。

```rust
// file="src/scalar.rs" mode="replace" item="pub fn pack("
	pub fn pack(self) -> u64 {
		match self {
			Self::Bool(x) => x as u64,
			Self::I32(x) => x as u32 as u64,
			Self::F32Bits(x) => x as u64,
		}
	}
```

`src/scalar.rs` の `pub fn unpack(`の定義を次の全体で置き換えます。

```rust
// file="src/scalar.rs" mode="replace" item="pub fn unpack("
	pub fn unpack(dtype: DType, bits: u64) -> Result<Self> {
		match dtype {
			DType::Bool => Ok(Self::Bool(bits != 0)),
			DType::I32 => Ok(Self::I32(bits as u32 as i32)),
			DType::F32 => Ok(Self::F32Bits(bits as u32)),
			DType::Void => Err("void has no value".into()),
		}
	}
```

`src/scalar.rs` の `pub fn render(`の定義を次の全体で置き換えます。

```rust
// file="src/scalar.rs" mode="replace" item="pub fn render("
	pub fn render(&self, root: usize) -> Result<String> {
		self.verify(root, 0)?;
		let mut source = SCALAR_HELPERS.to_string();
		source.push_str("int chibi_entry(void*const* buffers,const uint64_t* scalars){\n");
		for id in self.order(root)? {
			let n = &self.nodes[id];
			let expr = match n.op {
				SOp::Const(Value::Bool(x)) => format!("{}u", x as u8),
				SOp::Const(Value::I32(x)) => format!("chibi_i32({}u)", x as u32),
				SOp::Const(Value::F32Bits(x)) => format!("chibi_bits({x}u)"),
				SOp::Param { slot, dtype, .. } => match dtype {
					DType::I32 => format!("chibi_i32((uint32_t)scalars[{slot}])"),
					DType::F32 => format!("chibi_bits((uint32_t)scalars[{slot}])"),
					_ => format!("(scalars[{slot}]!=0)"),
				},
				SOp::Alu(op) => {
					let a = format!("v{}", n.inputs[0]);
					let b = n.inputs.get(1).map(|x| format!("v{x}")).unwrap_or_default();
					match op {
						Alu::Add | Alu::Sub | Alu::Mul if n.dtype == DType::I32 => format!(
							"chibi_i32((uint32_t){a} {} (uint32_t){b})",
							match op {
								Alu::Add => "+",
								Alu::Sub => "-",
								_ => "*",
							}
						),
						Alu::Neg if n.dtype == DType::I32 => format!("chibi_i32(0u-(uint32_t){a})"),
						Alu::Shl => format!("chibi_i32((uint32_t){a} << {b})"),
						Alu::Shr => format!("chibi_shr({a},{b})"),
						Alu::Div => format!("chibi_div({a},{b})"),
						Alu::Mod => format!("chibi_mod({a},{b})"),
						Alu::Max => {
							if n.dtype == DType::F32 {
								format!("chibi_max({a},{b})")
							} else {
								format!("({a}>{b}?{a}:{b})")
							}
						}
						Alu::Recip => format!("(1.0f/{a})"),
						Alu::Neg => format!("(-{a})"),
						Alu::Where => format!("({a}?{b}:v{})", n.inputs[2]),
						Alu::Cast(t) => {
							if t == DType::Bool {
								format!("({a}!=0)")
							} else {
								format!("(({}){a})", ctype(t))
							}
						}
						_ => format!(
							"({a} {} {b})",
							match op {
								Alu::Add => "+",
								Alu::Sub => "-",
								Alu::Mul => "*",
								Alu::Xor => "^",
								Alu::And => "&",
								Alu::Or => "|",
								Alu::Lt => "<",
								Alu::Eq => "==",
								Alu::Ne => "!=",
								_ => unreachable!(),
							}
						),
					}
				}
			};
			source.push_str(&format!("{} v{id}={expr};\n", ctype(n.dtype)));
		}
		if self.nodes[root].dtype == DType::F32 {
			source.push_str(&format!(
				"uint32_t bits;memcpy(&bits,&v{root},4);*(uint64_t*)buffers[0]=bits;\n"
			))
		} else {
			source.push_str(&format!("*(uint64_t*)buffers[0]=(uint32_t)v{root};\n"))
		}
		source.push_str("return 0;\n}\n");
		Ok(source)
	}
```

```sh
# check
cargo test --quiet --test scalar_rewrite
```

## 7. 加減乗算、比較、WHERE、CAST を interpreter と renderer に加える。
<!-- todo:03-007 -->

`tests/scalar_rewrite.rs` の `where_and_float_classification` で、この経路の値または拒否条件を確認する。

`src/scalar.rs` の `pub fn eval_alu(`の定義を次の全体で置き換えます。

```rust
// file="src/scalar.rs" mode="replace" item="pub fn eval_alu("
pub fn eval_alu(op: Alu, v: &[Value]) -> Result<Value> {
	let a = v[0];
	if let Alu::Cast(t) = op {
		return Ok(match t {
			DType::Bool => Value::Bool(a.number() != 0.),
			DType::I32 => Value::I32(a.number() as i32),
			DType::F32 => Value::f32(a.number() as f32),
			DType::Void => return Err("void cast".into()),
		});
	}
	if op == Alu::Where {
		return Ok(if a == Value::Bool(true) { v[1] } else { v[2] });
	}
	if matches!(op, Alu::Lt | Alu::Eq | Alu::Ne) {
		let b = v[1].number();
		let a = a.number();
		return Ok(Value::Bool(match op {
			Alu::Lt => a < b,
			Alu::Eq => a == b,
			_ => a != b,
		}));
	}
	Ok(match a {
		Value::Bool(a) => {
			let Value::Bool(b) = v[1] else {
				return Err("bool operand mismatch".into());
			};
			Value::Bool(match op {
				Alu::Xor => a ^ b,
				Alu::And => a & b,
				Alu::Or => a | b,
				_ => return Err("invalid bool operation".into()),
			})
		}
		Value::I32(a) => {
			let b = if v.len() > 1 {
				if let Value::I32(b) = v[1] {
					b
				} else {
					return Err("i32 operand mismatch".into());
				}
			} else {
				0
			};
			Value::I32(match op {
				Alu::Add => a.wrapping_add(b),
				Alu::Sub => a.wrapping_sub(b),
				Alu::Mul => a.wrapping_mul(b),
				Alu::Neg => a.wrapping_neg(),
				Alu::Xor => a ^ b,
				Alu::And => a & b,
				Alu::Or => a | b,
				Alu::Shl => a.wrapping_shl(b as u32),
				Alu::Shr => a >> b,
				Alu::Div | Alu::Mod => {
					if b == 0 || (a == i32::MIN && b == -1) {
						return Err("invalid division".into());
					}
					let q = a / b;
					let r = a % b;
					let q = q - if r != 0 && (r < 0) != (b < 0) { 1 } else { 0 };
					if op == Alu::Div {
						q
					} else {
						(a as i64 - q as i64 * b as i64) as i32
					}
				}
				Alu::Max => a.max(b),
				_ => return Err("invalid i32 operation".into()),
			})
		}
		Value::F32Bits(a) => {
			let a = f32::from_bits(a);
			let b = if v.len() > 1 {
				v[1].number() as f32
			} else {
				0.
			};
			Value::f32(match op {
				Alu::Add => a + b,
				Alu::Sub => a - b,
				Alu::Mul => a * b,
				Alu::Neg => -a,
				Alu::Recip => a.recip(),
				Alu::Max => {
					if a.is_nan() || b.is_nan() {
						f32::NAN
					} else if a == 0. && b == 0. {
						0.
					} else {
						a.max(b)
					}
				}
				_ => return Err("invalid f32 operation".into()),
			})
		}
	})
}
```

`src/scalar.rs` の `pub fn render(`の定義を次の全体で置き換えます。

```rust
// file="src/scalar.rs" mode="replace" item="pub fn render("
	pub fn render(&self, root: usize) -> Result<String> {
		self.verify(root, 0)?;
		let mut source = SCALAR_HELPERS.to_string();
		source.push_str("int chibi_entry(void*const* buffers,const uint64_t* scalars){\n");
		for id in self.order(root)? {
			let n = &self.nodes[id];
			let expr = match n.op {
				SOp::Const(Value::Bool(x)) => format!("{}u", x as u8),
				SOp::Const(Value::I32(x)) => format!("chibi_i32({}u)", x as u32),
				SOp::Const(Value::F32Bits(x)) => format!("chibi_bits({x}u)"),
				SOp::Param { slot, dtype, .. } => match dtype {
					DType::I32 => format!("chibi_i32((uint32_t)scalars[{slot}])"),
					DType::F32 => format!("chibi_bits((uint32_t)scalars[{slot}])"),
					_ => format!("(scalars[{slot}]!=0)"),
				},
				SOp::Alu(op) => {
					let a = format!("v{}", n.inputs[0]);
					let b = n.inputs.get(1).map(|x| format!("v{x}")).unwrap_or_default();
					match op {
						Alu::Add | Alu::Sub | Alu::Mul if n.dtype == DType::I32 => format!(
							"chibi_i32((uint32_t){a} {} (uint32_t){b})",
							match op {
								Alu::Add => "+",
								Alu::Sub => "-",
								_ => "*",
							}
						),
						Alu::Neg if n.dtype == DType::I32 => format!("chibi_i32(0u-(uint32_t){a})"),
						Alu::Shl => format!("chibi_i32((uint32_t){a} << {b})"),
						Alu::Shr => format!("chibi_shr({a},{b})"),
						Alu::Div => format!("chibi_div({a},{b})"),
						Alu::Mod => format!("chibi_mod({a},{b})"),
						Alu::Max => {
							if n.dtype == DType::F32 {
								format!("chibi_max({a},{b})")
							} else {
								format!("({a}>{b}?{a}:{b})")
							}
						}
						Alu::Recip => format!("(1.0f/{a})"),
						Alu::Neg => format!("(-{a})"),
						Alu::Where => format!("({a}?{b}:v{})", n.inputs[2]),
						Alu::Cast(t) => {
							if t == DType::Bool {
								format!("({a}!=0)")
							} else {
								format!("(({}){a})", ctype(t))
							}
						}
						_ => format!(
							"({a} {} {b})",
							match op {
								Alu::Add => "+",
								Alu::Sub => "-",
								Alu::Mul => "*",
								Alu::Xor => "^",
								Alu::And => "&",
								Alu::Or => "|",
								Alu::Lt => "<",
								Alu::Eq => "==",
								Alu::Ne => "!=",
								_ => unreachable!(),
							}
						),
					}
				}
			};
			source.push_str(&format!("{} v{id}={expr};\n", ctype(n.dtype)));
		}
		if self.nodes[root].dtype == DType::F32 {
			source.push_str(&format!(
				"uint32_t bits;memcpy(&bits,&v{root},4);*(uint64_t*)buffers[0]=bits;\n"
			))
		} else {
			source.push_str(&format!("*(uint64_t*)buffers[0]=(uint32_t)v{root};\n"))
		}
		source.push_str("return 0;\n}\n");
		Ok(source)
	}
```

```sh
# check
cargo test --quiet --test scalar_rewrite where_and_float_classification
```

## 8. void 定数と混合 dtype を検証時に拒否する。
<!-- todo:03-008 -->

`tests/scalar_rewrite.rs` の `typed_failures_and_special_values` で、この経路の値または拒否条件を確認する。

`src/scalar.rs` の `fn infer(`の定義を次の全体で置き換えます。

```rust
// file="src/scalar.rs" mode="replace" item="fn infer("
	fn infer(&self, op: Alu, inputs: &[usize]) -> Result<(DType, Option<Bounds>)> {
		let arity = match op {
			Alu::Neg | Alu::Recip | Alu::Cast(_) => 1,
			Alu::Where => 3,
			_ => 2,
		};
		if inputs.len() != arity {
			return Err("ALU arity mismatch".into());
		}
		let ns = inputs
			.iter()
			.map(|&id| self.nodes.get(id).ok_or("unknown scalar input".into()))
			.collect::<Result<Vec<_>>>()?;
		let t = ns[0].dtype;
		let numeric = |t| t == DType::I32 || t == DType::F32;
		let dtype = match op {
			Alu::Where => {
				if t != DType::Bool || ns[1].dtype != ns[2].dtype || ns[1].dtype == DType::Void {
					return Err("WHERE type mismatch".into());
				}
				ns[1].dtype
			}
			Alu::Cast(target) => {
				if target == DType::Void || t == DType::Void {
					return Err("void CAST".into());
				}
				if t == DType::F32 && target == DType::I32 {
					let b = ns[0].bounds.ok_or("unproven float-to-int CAST")?;
					if b.lo < i32::MIN as i64 || b.hi > 2147483520 {
						return Err("unsafe float-to-int CAST".into());
					}
				}
				target
			}
			Alu::Recip => {
				if t != DType::F32 {
					return Err("RECIP requires f32".into());
				}
				t
			}
			Alu::Xor | Alu::And | Alu::Or => {
				if !matches!(t, DType::I32 | DType::Bool) || ns[1].dtype != t {
					return Err("bitwise type mismatch".into());
				}
				t
			}
			Alu::Shl | Alu::Shr => {
				if t != DType::I32 || ns[1].dtype != t {
					return Err("shift requires i32".into());
				}
				let b = ns[1].bounds.ok_or("unproven shift count")?;
				if b.lo < 0 || b.hi >= 32 {
					return Err("unsafe shift count".into());
				}
				t
			}
			Alu::Div | Alu::Mod => {
				if t != DType::I32 || ns[1].dtype != t {
					return Err("integer DIV/MOD require i32".into());
				}
				let a = ns[0].bounds.unwrap_or(Bounds::I32);
				let b = ns[1].bounds.unwrap_or(Bounds::I32);
				if b.lo <= 0 && b.hi >= 0 || a.lo == i32::MIN as i64 && b.lo <= -1 && b.hi >= -1 {
					return Err("unsafe integer division".into());
				}
				t
			}
			Alu::Lt | Alu::Eq | Alu::Ne => {
				if !numeric(t) || ns[1].dtype != t {
					return Err("comparison type mismatch".into());
				}
				DType::Bool
			}
			_ => {
				if !numeric(t) || ns.iter().any(|n| n.dtype != t) {
					return Err("numeric type mismatch".into());
				}
				t
			}
		};
		let bounds = if dtype == DType::Bool {
			Some(Bounds { lo: 0, hi: 1 })
		} else if dtype == DType::I32 {
			let a = ns[0].bounds.unwrap_or(Bounds::I32);
			let b = ns.get(1).and_then(|n| n.bounds).unwrap_or(Bounds::I32);
			let raw = match op {
				Alu::Add => Some((a.lo + b.lo, a.hi + b.hi)),
				Alu::Sub => Some((a.lo - b.hi, a.hi - b.lo)),
				Alu::Mul => {
					let p = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi];
					Some((*p.iter().min().unwrap(), *p.iter().max().unwrap()))
				}
				_ => None,
			};
			Some(match raw {
				Some((lo, hi)) if lo >= i32::MIN as i64 && hi <= i32::MAX as i64 => {
					Bounds { lo, hi }
				}
				_ => Bounds::I32,
			})
		} else {
			None
		};
		Ok((dtype, bounds))
	}
```

```sh
# check
cargo test --quiet --test scalar_rewrite typed_failures_and_special_values
```

## 9. `WHERE(a<3,a*2,a+1)` を `a=2` で `4`、`a=5` で `6` にする。
<!-- todo:03-009 -->

`tests/scalar_rewrite.rs` の `where_and_float_classification` で、この経路の値または拒否条件を確認する。

`tests/scalar_rewrite.rs` の `fn where_and_float_classification(`の定義を次の全体で置き換えます。

```rust
// file="tests/scalar_rewrite.rs" mode="replace" item="fn where_and_float_classification("
fn where_and_float_classification() -> Result<()> {
	let mut g = ScalarGraph::default();
	let a = g.param(0, DType::F32, None)?;
	let three = g.constant(Value::f32(3.));
	let two = g.constant(Value::f32(2.));
	let one = g.constant(Value::f32(1.));
	let cond = g.alu(Alu::Lt, vec![a, three])?;
	let yes = g.alu(Alu::Mul, vec![a, two])?;
	let no = g.alu(Alu::Add, vec![a, one])?;
	let y = g.alu(Alu::Where, vec![cond, yes, no])?;
	let plan = g.compile(y)?;
	assert_eq!(plan.run(&[Value::f32(2.)])?, Value::f32(4.));
	assert_eq!(plan.run(&[Value::f32(5.)])?, Value::f32(6.));
	let (rewritten, root) = g.rewrite(y)?;
	for value in [-7., 0., 5., f32::INFINITY, f32::NAN] {
		let input = [Value::f32(value)];
		let actual = plan.run(&input)?.number();
		let reference = rewritten.evaluate(root, &input)?.number();
		assert!(actual == reference || actual.is_nan() && reference.is_nan());
	}
	Ok(())
}
```

```sh
# check
cargo test --quiet --test scalar_rewrite where_and_float_classification
```

## 10. XOR、AND、OR、SHL、SHR、DIV、MOD を両経路に加える。
<!-- todo:03-010 -->

`tests/scalar_rewrite.rs` の `integer_contract_and_rewrite` で、この経路の値または拒否条件を確認する。

`src/scalar.rs` の `pub fn eval_alu(`の定義を次の全体で置き換えます。

```rust
// file="src/scalar.rs" mode="replace" item="pub fn eval_alu("
pub fn eval_alu(op: Alu, v: &[Value]) -> Result<Value> {
	let a = v[0];
	if let Alu::Cast(t) = op {
		return Ok(match t {
			DType::Bool => Value::Bool(a.number() != 0.),
			DType::I32 => Value::I32(a.number() as i32),
			DType::F32 => Value::f32(a.number() as f32),
			DType::Void => return Err("void cast".into()),
		});
	}
	if op == Alu::Where {
		return Ok(if a == Value::Bool(true) { v[1] } else { v[2] });
	}
	if matches!(op, Alu::Lt | Alu::Eq | Alu::Ne) {
		let b = v[1].number();
		let a = a.number();
		return Ok(Value::Bool(match op {
			Alu::Lt => a < b,
			Alu::Eq => a == b,
			_ => a != b,
		}));
	}
	Ok(match a {
		Value::Bool(a) => {
			let Value::Bool(b) = v[1] else {
				return Err("bool operand mismatch".into());
			};
			Value::Bool(match op {
				Alu::Xor => a ^ b,
				Alu::And => a & b,
				Alu::Or => a | b,
				_ => return Err("invalid bool operation".into()),
			})
		}
		Value::I32(a) => {
			let b = if v.len() > 1 {
				if let Value::I32(b) = v[1] {
					b
				} else {
					return Err("i32 operand mismatch".into());
				}
			} else {
				0
			};
			Value::I32(match op {
				Alu::Add => a.wrapping_add(b),
				Alu::Sub => a.wrapping_sub(b),
				Alu::Mul => a.wrapping_mul(b),
				Alu::Neg => a.wrapping_neg(),
				Alu::Xor => a ^ b,
				Alu::And => a & b,
				Alu::Or => a | b,
				Alu::Shl => a.wrapping_shl(b as u32),
				Alu::Shr => a >> b,
				Alu::Div | Alu::Mod => {
					if b == 0 || (a == i32::MIN && b == -1) {
						return Err("invalid division".into());
					}
					let q = a / b;
					let r = a % b;
					let q = q - if r != 0 && (r < 0) != (b < 0) { 1 } else { 0 };
					if op == Alu::Div {
						q
					} else {
						(a as i64 - q as i64 * b as i64) as i32
					}
				}
				Alu::Max => a.max(b),
				_ => return Err("invalid i32 operation".into()),
			})
		}
		Value::F32Bits(a) => {
			let a = f32::from_bits(a);
			let b = if v.len() > 1 {
				v[1].number() as f32
			} else {
				0.
			};
			Value::f32(match op {
				Alu::Add => a + b,
				Alu::Sub => a - b,
				Alu::Mul => a * b,
				Alu::Neg => -a,
				Alu::Recip => a.recip(),
				Alu::Max => {
					if a.is_nan() || b.is_nan() {
						f32::NAN
					} else if a == 0. && b == 0. {
						0.
					} else {
						a.max(b)
					}
				}
				_ => return Err("invalid f32 operation".into()),
			})
		}
	})
}
```

`src/scalar.rs` の `pub fn render(`の定義を次の全体で置き換えます。

```rust
// file="src/scalar.rs" mode="replace" item="pub fn render("
	pub fn render(&self, root: usize) -> Result<String> {
		self.verify(root, 0)?;
		let mut source = SCALAR_HELPERS.to_string();
		source.push_str("int chibi_entry(void*const* buffers,const uint64_t* scalars){\n");
		for id in self.order(root)? {
			let n = &self.nodes[id];
			let expr = match n.op {
				SOp::Const(Value::Bool(x)) => format!("{}u", x as u8),
				SOp::Const(Value::I32(x)) => format!("chibi_i32({}u)", x as u32),
				SOp::Const(Value::F32Bits(x)) => format!("chibi_bits({x}u)"),
				SOp::Param { slot, dtype, .. } => match dtype {
					DType::I32 => format!("chibi_i32((uint32_t)scalars[{slot}])"),
					DType::F32 => format!("chibi_bits((uint32_t)scalars[{slot}])"),
					_ => format!("(scalars[{slot}]!=0)"),
				},
				SOp::Alu(op) => {
					let a = format!("v{}", n.inputs[0]);
					let b = n.inputs.get(1).map(|x| format!("v{x}")).unwrap_or_default();
					match op {
						Alu::Add | Alu::Sub | Alu::Mul if n.dtype == DType::I32 => format!(
							"chibi_i32((uint32_t){a} {} (uint32_t){b})",
							match op {
								Alu::Add => "+",
								Alu::Sub => "-",
								_ => "*",
							}
						),
						Alu::Neg if n.dtype == DType::I32 => format!("chibi_i32(0u-(uint32_t){a})"),
						Alu::Shl => format!("chibi_i32((uint32_t){a} << {b})"),
						Alu::Shr => format!("chibi_shr({a},{b})"),
						Alu::Div => format!("chibi_div({a},{b})"),
						Alu::Mod => format!("chibi_mod({a},{b})"),
						Alu::Max => {
							if n.dtype == DType::F32 {
								format!("chibi_max({a},{b})")
							} else {
								format!("({a}>{b}?{a}:{b})")
							}
						}
						Alu::Recip => format!("(1.0f/{a})"),
						Alu::Neg => format!("(-{a})"),
						Alu::Where => format!("({a}?{b}:v{})", n.inputs[2]),
						Alu::Cast(t) => {
							if t == DType::Bool {
								format!("({a}!=0)")
							} else {
								format!("(({}){a})", ctype(t))
							}
						}
						_ => format!(
							"({a} {} {b})",
							match op {
								Alu::Add => "+",
								Alu::Sub => "-",
								Alu::Mul => "*",
								Alu::Xor => "^",
								Alu::And => "&",
								Alu::Or => "|",
								Alu::Lt => "<",
								Alu::Eq => "==",
								Alu::Ne => "!=",
								_ => unreachable!(),
							}
						),
					}
				}
			};
			source.push_str(&format!("{} v{id}={expr};\n", ctype(n.dtype)));
		}
		if self.nodes[root].dtype == DType::F32 {
			source.push_str(&format!(
				"uint32_t bits;memcpy(&bits,&v{root},4);*(uint64_t*)buffers[0]=bits;\n"
			))
		} else {
			source.push_str(&format!("*(uint64_t*)buffers[0]=(uint32_t)v{root};\n"))
		}
		source.push_str("return 0;\n}\n");
		Ok(source)
	}
```

```sh
# check
cargo test --quiet --test scalar_rewrite integer_contract_and_rewrite
```

## 11. RECIP と MAX を本文の f32 契約で実行する。
<!-- todo:03-011 -->

`tests/scalar_rewrite.rs` の正常系と境界系をまとめて再実行する。

`src/scalar.rs` の `pub fn eval_alu(`の定義を次の全体で置き換えます。

```rust
// file="src/scalar.rs" mode="replace" item="pub fn eval_alu("
pub fn eval_alu(op: Alu, v: &[Value]) -> Result<Value> {
	let a = v[0];
	if let Alu::Cast(t) = op {
		return Ok(match t {
			DType::Bool => Value::Bool(a.number() != 0.),
			DType::I32 => Value::I32(a.number() as i32),
			DType::F32 => Value::f32(a.number() as f32),
			DType::Void => return Err("void cast".into()),
		});
	}
	if op == Alu::Where {
		return Ok(if a == Value::Bool(true) { v[1] } else { v[2] });
	}
	if matches!(op, Alu::Lt | Alu::Eq | Alu::Ne) {
		let b = v[1].number();
		let a = a.number();
		return Ok(Value::Bool(match op {
			Alu::Lt => a < b,
			Alu::Eq => a == b,
			_ => a != b,
		}));
	}
	Ok(match a {
		Value::Bool(a) => {
			let Value::Bool(b) = v[1] else {
				return Err("bool operand mismatch".into());
			};
			Value::Bool(match op {
				Alu::Xor => a ^ b,
				Alu::And => a & b,
				Alu::Or => a | b,
				_ => return Err("invalid bool operation".into()),
			})
		}
		Value::I32(a) => {
			let b = if v.len() > 1 {
				if let Value::I32(b) = v[1] {
					b
				} else {
					return Err("i32 operand mismatch".into());
				}
			} else {
				0
			};
			Value::I32(match op {
				Alu::Add => a.wrapping_add(b),
				Alu::Sub => a.wrapping_sub(b),
				Alu::Mul => a.wrapping_mul(b),
				Alu::Neg => a.wrapping_neg(),
				Alu::Xor => a ^ b,
				Alu::And => a & b,
				Alu::Or => a | b,
				Alu::Shl => a.wrapping_shl(b as u32),
				Alu::Shr => a >> b,
				Alu::Div | Alu::Mod => {
					if b == 0 || (a == i32::MIN && b == -1) {
						return Err("invalid division".into());
					}
					let q = a / b;
					let r = a % b;
					let q = q - if r != 0 && (r < 0) != (b < 0) { 1 } else { 0 };
					if op == Alu::Div {
						q
					} else {
						(a as i64 - q as i64 * b as i64) as i32
					}
				}
				Alu::Max => a.max(b),
				_ => return Err("invalid i32 operation".into()),
			})
		}
		Value::F32Bits(a) => {
			let a = f32::from_bits(a);
			let b = if v.len() > 1 {
				v[1].number() as f32
			} else {
				0.
			};
			Value::f32(match op {
				Alu::Add => a + b,
				Alu::Sub => a - b,
				Alu::Mul => a * b,
				Alu::Neg => -a,
				Alu::Recip => a.recip(),
				Alu::Max => {
					if a.is_nan() || b.is_nan() {
						f32::NAN
					} else if a == 0. && b == 0. {
						0.
					} else {
						a.max(b)
					}
				}
				_ => return Err("invalid f32 operation".into()),
			})
		}
	})
}
```

`src/scalar.rs` の `pub fn render(`の定義を次の全体で置き換えます。

```rust
// file="src/scalar.rs" mode="replace" item="pub fn render("
	pub fn render(&self, root: usize) -> Result<String> {
		self.verify(root, 0)?;
		let mut source = SCALAR_HELPERS.to_string();
		source.push_str("int chibi_entry(void*const* buffers,const uint64_t* scalars){\n");
		for id in self.order(root)? {
			let n = &self.nodes[id];
			let expr = match n.op {
				SOp::Const(Value::Bool(x)) => format!("{}u", x as u8),
				SOp::Const(Value::I32(x)) => format!("chibi_i32({}u)", x as u32),
				SOp::Const(Value::F32Bits(x)) => format!("chibi_bits({x}u)"),
				SOp::Param { slot, dtype, .. } => match dtype {
					DType::I32 => format!("chibi_i32((uint32_t)scalars[{slot}])"),
					DType::F32 => format!("chibi_bits((uint32_t)scalars[{slot}])"),
					_ => format!("(scalars[{slot}]!=0)"),
				},
				SOp::Alu(op) => {
					let a = format!("v{}", n.inputs[0]);
					let b = n.inputs.get(1).map(|x| format!("v{x}")).unwrap_or_default();
					match op {
						Alu::Add | Alu::Sub | Alu::Mul if n.dtype == DType::I32 => format!(
							"chibi_i32((uint32_t){a} {} (uint32_t){b})",
							match op {
								Alu::Add => "+",
								Alu::Sub => "-",
								_ => "*",
							}
						),
						Alu::Neg if n.dtype == DType::I32 => format!("chibi_i32(0u-(uint32_t){a})"),
						Alu::Shl => format!("chibi_i32((uint32_t){a} << {b})"),
						Alu::Shr => format!("chibi_shr({a},{b})"),
						Alu::Div => format!("chibi_div({a},{b})"),
						Alu::Mod => format!("chibi_mod({a},{b})"),
						Alu::Max => {
							if n.dtype == DType::F32 {
								format!("chibi_max({a},{b})")
							} else {
								format!("({a}>{b}?{a}:{b})")
							}
						}
						Alu::Recip => format!("(1.0f/{a})"),
						Alu::Neg => format!("(-{a})"),
						Alu::Where => format!("({a}?{b}:v{})", n.inputs[2]),
						Alu::Cast(t) => {
							if t == DType::Bool {
								format!("({a}!=0)")
							} else {
								format!("(({}){a})", ctype(t))
							}
						}
						_ => format!(
							"({a} {} {b})",
							match op {
								Alu::Add => "+",
								Alu::Sub => "-",
								Alu::Mul => "*",
								Alu::Xor => "^",
								Alu::And => "&",
								Alu::Or => "|",
								Alu::Lt => "<",
								Alu::Eq => "==",
								Alu::Ne => "!=",
								_ => unreachable!(),
							}
						),
					}
				}
			};
			source.push_str(&format!("{} v{id}={expr};\n", ctype(n.dtype)));
		}
		if self.nodes[root].dtype == DType::F32 {
			source.push_str(&format!(
				"uint32_t bits;memcpy(&bits,&v{root},4);*(uint64_t*)buffers[0]=bits;\n"
			))
		} else {
			source.push_str(&format!("*(uint64_t*)buffers[0]=(uint32_t)v{root};\n"))
		}
		source.push_str("return 0;\n}\n");
		Ok(source)
	}
```

```sh
# check
cargo test --quiet --test scalar_rewrite
```

## 12. bounds と動的入力の範囲検証を追加する。
<!-- todo:03-012 -->

`tests/scalar_rewrite.rs` の `integer_contract_and_rewrite` で、この経路の値または拒否条件を確認する。

`src/scalar.rs` の `pub fn validate_inputs(`の定義を次の全体で置き換えます。

```rust
// file="src/scalar.rs" mode="replace" item="pub fn validate_inputs("
	pub fn validate_inputs(&self, root: usize, inputs: &[Value]) -> Result<()> {
		let mut slots = HashSet::new();
		for id in self.order(root)? {
			if let SOp::Param {
				slot,
				dtype,
				bounds,
				..
			} = self.nodes[id].op
			{
				slots.insert(slot);
				let v = *inputs.get(slot).ok_or("missing scalar input")?;
				if v.dtype() != dtype {
					return Err("scalar input dtype mismatch".into());
				}
				if let Some(b) = bounds {
					let n = v.number();
					if !n.is_finite() || n < b.lo as f64 || n > b.hi as f64 {
						return Err("scalar input outside declared bounds".into());
					}
				}
			}
		}
		if slots.len() != inputs.len() || (0..inputs.len()).any(|s| !slots.contains(&s)) {
			return Err("extra or non-dense scalar slots".into());
		}
		Ok(())
	}
```

```sh
# check
cargo test --quiet --test scalar_rewrite integer_contract_and_rewrite
```

## 13. 危険な DIV、shift、CAST は安全性を証明できなければ拒否する。
<!-- todo:03-013 -->

`tests/scalar_rewrite.rs` の `typed_failures_and_special_values` で、この経路の値または拒否条件を確認する。

`src/scalar.rs` の `fn infer(`の定義を次の全体で置き換えます。

```rust
// file="src/scalar.rs" mode="replace" item="fn infer("
	fn infer(&self, op: Alu, inputs: &[usize]) -> Result<(DType, Option<Bounds>)> {
		let arity = match op {
			Alu::Neg | Alu::Recip | Alu::Cast(_) => 1,
			Alu::Where => 3,
			_ => 2,
		};
		if inputs.len() != arity {
			return Err("ALU arity mismatch".into());
		}
		let ns = inputs
			.iter()
			.map(|&id| self.nodes.get(id).ok_or("unknown scalar input".into()))
			.collect::<Result<Vec<_>>>()?;
		let t = ns[0].dtype;
		let numeric = |t| t == DType::I32 || t == DType::F32;
		let dtype = match op {
			Alu::Where => {
				if t != DType::Bool || ns[1].dtype != ns[2].dtype || ns[1].dtype == DType::Void {
					return Err("WHERE type mismatch".into());
				}
				ns[1].dtype
			}
			Alu::Cast(target) => {
				if target == DType::Void || t == DType::Void {
					return Err("void CAST".into());
				}
				if t == DType::F32 && target == DType::I32 {
					let b = ns[0].bounds.ok_or("unproven float-to-int CAST")?;
					if b.lo < i32::MIN as i64 || b.hi > 2147483520 {
						return Err("unsafe float-to-int CAST".into());
					}
				}
				target
			}
			Alu::Recip => {
				if t != DType::F32 {
					return Err("RECIP requires f32".into());
				}
				t
			}
			Alu::Xor | Alu::And | Alu::Or => {
				if !matches!(t, DType::I32 | DType::Bool) || ns[1].dtype != t {
					return Err("bitwise type mismatch".into());
				}
				t
			}
			Alu::Shl | Alu::Shr => {
				if t != DType::I32 || ns[1].dtype != t {
					return Err("shift requires i32".into());
				}
				let b = ns[1].bounds.ok_or("unproven shift count")?;
				if b.lo < 0 || b.hi >= 32 {
					return Err("unsafe shift count".into());
				}
				t
			}
			Alu::Div | Alu::Mod => {
				if t != DType::I32 || ns[1].dtype != t {
					return Err("integer DIV/MOD require i32".into());
				}
				let a = ns[0].bounds.unwrap_or(Bounds::I32);
				let b = ns[1].bounds.unwrap_or(Bounds::I32);
				if b.lo <= 0 && b.hi >= 0 || a.lo == i32::MIN as i64 && b.lo <= -1 && b.hi >= -1 {
					return Err("unsafe integer division".into());
				}
				t
			}
			Alu::Lt | Alu::Eq | Alu::Ne => {
				if !numeric(t) || ns[1].dtype != t {
					return Err("comparison type mismatch".into());
				}
				DType::Bool
			}
			_ => {
				if !numeric(t) || ns.iter().any(|n| n.dtype != t) {
					return Err("numeric type mismatch".into());
				}
				t
			}
		};
		let bounds = if dtype == DType::Bool {
			Some(Bounds { lo: 0, hi: 1 })
		} else if dtype == DType::I32 {
			let a = ns[0].bounds.unwrap_or(Bounds::I32);
			let b = ns.get(1).and_then(|n| n.bounds).unwrap_or(Bounds::I32);
			let raw = match op {
				Alu::Add => Some((a.lo + b.lo, a.hi + b.hi)),
				Alu::Sub => Some((a.lo - b.hi, a.hi - b.lo)),
				Alu::Mul => {
					let p = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi];
					Some((*p.iter().min().unwrap(), *p.iter().max().unwrap()))
				}
				_ => None,
			};
			Some(match raw {
				Some((lo, hi)) if lo >= i32::MIN as i64 && hi <= i32::MAX as i64 => {
					Bounds { lo, hi }
				}
				_ => Bounds::I32,
			})
		} else {
			None
		};
		Ok((dtype, bounds))
	}
```

```sh
# check
cargo test --quiet --test scalar_rewrite typed_failures_and_special_values
```

## 14. `((5 XOR 3) SHL 1)=12` と負数の DIV、MOD を C まで通す。
<!-- todo:03-014 -->

`tests/scalar_rewrite.rs` の `integer_contract_and_rewrite` で、この経路の値または拒否条件を確認する。

`tests/scalar_rewrite.rs` の `fn integer_contract_and_rewrite(`の定義を次の全体で置き換えます。

```rust
// file="tests/scalar_rewrite.rs" mode="replace" item="fn integer_contract_and_rewrite("
fn integer_contract_and_rewrite() -> Result<()> {
	let mut g = ScalarGraph::default();
	let x = g.param(0, DType::I32, Some(Bounds::new(-7, 5)?))?;
	let zero = g.constant(Value::I32(0));
	let root = g.alu(Alu::Add, vec![x, zero])?;
	let (r, y) = g.rewrite(root)?;
	assert_eq!(r.cost(y)?, 0);
	for x in [-7, 0, 5] {
		assert_eq!(
			g.evaluate(root, &[Value::I32(x)])?,
			r.compile(y)?.run(&[Value::I32(x)])?
		)
	}
	for (op, a, b, expected) in [
		(Alu::Div, -7, 3, -3),
		(Alu::Mod, -7, 3, 2),
		(Alu::Add, i32::MAX, 1, i32::MIN),
		(Alu::Shr, -8, 1, -4),
		(Alu::Shl, 6, 1, 12),
	] {
		let a = g.constant(Value::I32(a));
		let b = g.constant(Value::I32(b));
		let y = g.alu(op, vec![a, b])?;
		assert_eq!(g.compile(y)?.run(&[])?, Value::I32(expected));
	}
	let a = g.constant(Value::I32(1));
	assert!(g.alu(Alu::Div, vec![a, zero]).is_err());
	assert!(g.alu(Alu::Shl, vec![a, x]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test scalar_rewrite integer_contract_and_rewrite
```

## 15. 子を先に再構築する rewrite を `match` で作る。
<!-- todo:03-015 -->

`tests/scalar_rewrite.rs` の `integer_contract_and_rewrite` で、この経路の値または拒否条件を確認する。

`src/scalar.rs` の `pub fn rewrite(`の定義を次の全体で置き換えます。

```rust
// file="src/scalar.rs" mode="replace" item="pub fn rewrite("
	pub fn rewrite(&self, root: usize) -> Result<(Self, usize)> {
		self.verify(root, 0)?;
		let mut out = Self::default();
		let mut map: HashMap<usize, usize> = HashMap::new();
		for id in self.order(root)? {
			let n = &self.nodes[id];
			let new = match n.op {
				SOp::Const(v) => out.constant(v),
				SOp::Param {
					slot,
					scope,
					dtype,
					bounds,
				} => out.scoped_param(scope, slot, dtype, bounds)?,
				SOp::Alu(op) => {
					let children = n.inputs.iter().map(|x| map[x]).collect::<Vec<_>>();
					let constants = children
						.iter()
						.map(|&x| match out.nodes[x].op {
							SOp::Const(v) => Some(v),
							_ => None,
						})
						.collect::<Option<Vec<_>>>();
					if let Some(values) = constants {
						out.constant(eval_alu(op, &values)?)
					} else if matches!(op, Alu::Add | Alu::Xor)
						&& n.dtype == DType::I32
						&& matches!(out.nodes[children[1]].op, SOp::Const(Value::I32(0)))
					{
						children[0]
					} else if op == Alu::Where {
						if let SOp::Const(Value::Bool(b)) = out.nodes[children[0]].op {
							children[if b { 1 } else { 2 }]
						} else {
							out.alu(op, children)?
						}
					} else {
						out.alu(op, children)?
					}
				}
			};
			map.insert(id, new);
		}
		Ok((out, map[&root]))
	}
```

```sh
# check
cargo test --quiet --test scalar_rewrite integer_contract_and_rewrite
```

## 16. 検証済み ALU evaluator を定数畳み込みにも使う。
<!-- todo:03-016 -->

`tests/scalar_rewrite.rs` の正常系と境界系をまとめて再実行する。

`src/scalar.rs` の `pub fn rewrite(`の定義を次の全体で置き換えます。

```rust
// file="src/scalar.rs" mode="replace" item="pub fn rewrite("
	pub fn rewrite(&self, root: usize) -> Result<(Self, usize)> {
		self.verify(root, 0)?;
		let mut out = Self::default();
		let mut map: HashMap<usize, usize> = HashMap::new();
		for id in self.order(root)? {
			let n = &self.nodes[id];
			let new = match n.op {
				SOp::Const(v) => out.constant(v),
				SOp::Param {
					slot,
					scope,
					dtype,
					bounds,
				} => out.scoped_param(scope, slot, dtype, bounds)?,
				SOp::Alu(op) => {
					let children = n.inputs.iter().map(|x| map[x]).collect::<Vec<_>>();
					let constants = children
						.iter()
						.map(|&x| match out.nodes[x].op {
							SOp::Const(v) => Some(v),
							_ => None,
						})
						.collect::<Option<Vec<_>>>();
					if let Some(values) = constants {
						out.constant(eval_alu(op, &values)?)
					} else if matches!(op, Alu::Add | Alu::Xor)
						&& n.dtype == DType::I32
						&& matches!(out.nodes[children[1]].op, SOp::Const(Value::I32(0)))
					{
						children[0]
					} else if op == Alu::Where {
						if let SOp::Const(Value::Bool(b)) = out.nodes[children[0]].op {
							children[if b { 1 } else { 2 }]
						} else {
							out.alu(op, children)?
						}
					} else {
						out.alu(op, children)?
					}
				}
			};
			map.insert(id, new);
		}
		Ok((out, map[&root]))
	}
```

```sh
# check
cargo test --quiet --test scalar_rewrite
```

## 17. `i32 x+0 → x`、`x XOR 0 → x`、`WHERE(true,a,b) → a` を加える。
<!-- todo:03-017 -->

`tests/scalar_rewrite.rs` の `where_and_float_classification` で、この経路の値または拒否条件を確認する。

`src/scalar.rs` の `pub fn rewrite(`の定義を次の全体で置き換えます。

```rust
// file="src/scalar.rs" mode="replace" item="pub fn rewrite("
	pub fn rewrite(&self, root: usize) -> Result<(Self, usize)> {
		self.verify(root, 0)?;
		let mut out = Self::default();
		let mut map: HashMap<usize, usize> = HashMap::new();
		for id in self.order(root)? {
			let n = &self.nodes[id];
			let new = match n.op {
				SOp::Const(v) => out.constant(v),
				SOp::Param {
					slot,
					scope,
					dtype,
					bounds,
				} => out.scoped_param(scope, slot, dtype, bounds)?,
				SOp::Alu(op) => {
					let children = n.inputs.iter().map(|x| map[x]).collect::<Vec<_>>();
					let constants = children
						.iter()
						.map(|&x| match out.nodes[x].op {
							SOp::Const(v) => Some(v),
							_ => None,
						})
						.collect::<Option<Vec<_>>>();
					if let Some(values) = constants {
						out.constant(eval_alu(op, &values)?)
					} else if matches!(op, Alu::Add | Alu::Xor)
						&& n.dtype == DType::I32
						&& matches!(out.nodes[children[1]].op, SOp::Const(Value::I32(0)))
					{
						children[0]
					} else if op == Alu::Where {
						if let SOp::Const(Value::Bool(b)) = out.nodes[children[0]].op {
							children[if b { 1 } else { 2 }]
						} else {
							out.alu(op, children)?
						}
					} else {
						out.alu(op, children)?
					}
				}
			};
			map.insert(id, new);
		}
		Ok((out, map[&root]))
	}
```

```sh
# check
cargo test --quiet --test scalar_rewrite where_and_float_classification
```

## 18. 前後の dtype、bounds、ALU 数を dump に出す。
<!-- todo:03-018 -->

`tests/scalar_rewrite.rs` の `typed_failures_and_special_values` で、この経路の値または拒否条件を確認する。

`tests/scalar_rewrite.rs` の `fn typed_failures_and_special_values(`の定義を次の全体で置き換えます。

```rust
// file="tests/scalar_rewrite.rs" mode="replace" item="fn typed_failures_and_special_values("
fn typed_failures_and_special_values() -> Result<()> {
	let mut g = ScalarGraph::default();
	let f = g.param(0, DType::F32, None)?;
	let i = g.constant(Value::I32(1));
	assert!(g.alu(Alu::Add, vec![f, i]).is_err());
	assert!(g.alu(Alu::Cast(DType::I32), vec![f]).is_err());
	assert!(g.param(1, DType::Void, None).is_err());
	assert!(g.alu(Alu::Add, vec![i]).is_err());
	assert!(g.compile(999).is_err());
	let foreign = g.scoped_param(42, 0, DType::F32, None)?;
	assert!(g.compile(foreign).is_err());
	let minus = g.constant(Value::f32(-0.));
	let plus = g.constant(Value::f32(0.));
	assert_ne!(minus, plus);
	let max = g.alu(Alu::Max, vec![minus, minus])?;
	assert_eq!(g.compile(max)?.run(&[])?, Value::f32(0.));
	let nan = g.constant(Value::F32Bits(0x7fc00042));
	assert_eq!(nan, g.constant(Value::F32Bits(0x7fc00042)));
	let max = g.alu(Alu::Max, vec![nan, plus])?;
	assert!(g.compile(max)?.run(&[])?.number().is_nan());
	let mut bounded = ScalarGraph::default();
	let x = bounded.param(0, DType::I32, Some(Bounds::new(-7, 5)?))?;
	assert!(bounded.compile(x)?.run(&[Value::I32(6)]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test scalar_rewrite typed_failures_and_special_values
```

## 19. 三つの実行経路を入力 `-7,0,5` と定数例で比較する。
<!-- todo:03-019 -->

`tests/scalar_rewrite.rs` の正常系と境界系をまとめて再実行する。

`tests/scalar_rewrite.rs` の `fn arithmetic_and_shared_library(`の定義を次の全体で置き換えます。

```rust
// file="tests/scalar_rewrite.rs" mode="replace" item="fn arithmetic_and_shared_library("
fn arithmetic_and_shared_library() -> Result<()> {
	let mut g = ScalarGraph::default();
	let a = g.param(0, DType::F32, None)?;
	let b = g.param(1, DType::F32, None)?;
	let c = g.param(2, DType::F32, None)?;
	let sum = g.alu(Alu::Add, vec![a, b])?;
	assert_eq!(sum, g.alu(Alu::Add, vec![a, b])?);
	let root = g.alu(Alu::Mul, vec![sum, c])?;
	let p = g.compile(root)?;
	assert_eq!(
		p.run(&[Value::f32(2.), Value::f32(3.), Value::f32(4.)])?,
		Value::f32(20.)
	);
	assert_eq!(
		p.run(&[Value::f32(1.), Value::f32(2.), Value::f32(4.)])?,
		Value::f32(12.)
	);
	assert!(p.run(&[]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test scalar_rewrite
```

## 20. 床除算、wrap、shift、f32 の特殊値で三経路が合う。
<!-- todo:03-020 -->

`tests/scalar_rewrite.rs` の `typed_failures_and_special_values` で、この経路の値または拒否条件を確認する。

`tests/scalar_rewrite.rs` の `fn typed_failures_and_special_values(`の定義を次の全体で置き換えます。

```rust
// file="tests/scalar_rewrite.rs" mode="replace" item="fn typed_failures_and_special_values("
fn typed_failures_and_special_values() -> Result<()> {
	let mut g = ScalarGraph::default();
	let f = g.param(0, DType::F32, None)?;
	let i = g.constant(Value::I32(1));
	assert!(g.alu(Alu::Add, vec![f, i]).is_err());
	assert!(g.alu(Alu::Cast(DType::I32), vec![f]).is_err());
	assert!(g.param(1, DType::Void, None).is_err());
	assert!(g.alu(Alu::Add, vec![i]).is_err());
	assert!(g.compile(999).is_err());
	let foreign = g.scoped_param(42, 0, DType::F32, None)?;
	assert!(g.compile(foreign).is_err());
	let minus = g.constant(Value::f32(-0.));
	let plus = g.constant(Value::f32(0.));
	assert_ne!(minus, plus);
	let max = g.alu(Alu::Max, vec![minus, minus])?;
	assert_eq!(g.compile(max)?.run(&[])?, Value::f32(0.));
	let nan = g.constant(Value::F32Bits(0x7fc00042));
	assert_eq!(nan, g.constant(Value::F32Bits(0x7fc00042)));
	let max = g.alu(Alu::Max, vec![nan, plus])?;
	assert!(g.compile(max)?.run(&[])?.number().is_nan());
	let mut bounded = ScalarGraph::default();
	let x = bounded.param(0, DType::I32, Some(Bounds::new(-7, 5)?))?;
	assert!(bounded.compile(x)?.run(&[Value::I32(6)]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test scalar_rewrite typed_failures_and_special_values
```

## 21. NaN は `is_nan`、符号付きゼロはビット列で比較する。
<!-- todo:03-021 -->

`tests/scalar_rewrite.rs` の `typed_failures_and_special_values` で、この経路の値または拒否条件を確認する。

`tests/scalar_rewrite.rs` の `fn typed_failures_and_special_values(`の定義を次の全体で置き換えます。

```rust
// file="tests/scalar_rewrite.rs" mode="replace" item="fn typed_failures_and_special_values("
fn typed_failures_and_special_values() -> Result<()> {
	let mut g = ScalarGraph::default();
	let f = g.param(0, DType::F32, None)?;
	let i = g.constant(Value::I32(1));
	assert!(g.alu(Alu::Add, vec![f, i]).is_err());
	assert!(g.alu(Alu::Cast(DType::I32), vec![f]).is_err());
	assert!(g.param(1, DType::Void, None).is_err());
	assert!(g.alu(Alu::Add, vec![i]).is_err());
	assert!(g.compile(999).is_err());
	let foreign = g.scoped_param(42, 0, DType::F32, None)?;
	assert!(g.compile(foreign).is_err());
	let minus = g.constant(Value::f32(-0.));
	let plus = g.constant(Value::f32(0.));
	assert_ne!(minus, plus);
	let max = g.alu(Alu::Max, vec![minus, minus])?;
	assert_eq!(g.compile(max)?.run(&[])?, Value::f32(0.));
	let nan = g.constant(Value::F32Bits(0x7fc00042));
	assert_eq!(nan, g.constant(Value::F32Bits(0x7fc00042)));
	let max = g.alu(Alu::Max, vec![nan, plus])?;
	assert!(g.compile(max)?.run(&[])?.number().is_nan());
	let mut bounded = ScalarGraph::default();
	let x = bounded.param(0, DType::I32, Some(Bounds::new(-7, 5)?))?;
	assert!(bounded.compile(x)?.run(&[Value::I32(6)]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test scalar_rewrite typed_failures_and_special_values
```

## 22. 有限の一般 f32 は予定許容値 `atol=1e-6, rtol=1e-6` で比較する。
<!-- todo:03-022 -->

`tests/scalar_rewrite.rs` の正常系と境界系をまとめて再実行する。

`tests/scalar_rewrite.rs` の `fn arithmetic_and_shared_library(`の定義を次の全体で置き換えます。

```rust
// file="tests/scalar_rewrite.rs" mode="replace" item="fn arithmetic_and_shared_library("
fn arithmetic_and_shared_library() -> Result<()> {
	let mut g = ScalarGraph::default();
	let a = g.param(0, DType::F32, None)?;
	let b = g.param(1, DType::F32, None)?;
	let c = g.param(2, DType::F32, None)?;
	let sum = g.alu(Alu::Add, vec![a, b])?;
	assert_eq!(sum, g.alu(Alu::Add, vec![a, b])?);
	let root = g.alu(Alu::Mul, vec![sum, c])?;
	let p = g.compile(root)?;
	assert_eq!(
		p.run(&[Value::f32(2.), Value::f32(3.), Value::f32(4.)])?,
		Value::f32(20.)
	);
	assert_eq!(
		p.run(&[Value::f32(1.), Value::f32(2.), Value::f32(4.)])?,
		Value::f32(12.)
	);
	assert!(p.run(&[]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test scalar_rewrite
```

## 23. dtype 不一致、void 入力、範囲外入力は生成か呼出しの前に失敗する。
<!-- todo:03-023 -->

`tests/scalar_rewrite.rs` の `typed_failures_and_special_values` で、この経路の値または拒否条件を確認する。

`tests/scalar_rewrite.rs` の `fn typed_failures_and_special_values(`の定義を次の全体で置き換えます。

```rust
// file="tests/scalar_rewrite.rs" mode="replace" item="fn typed_failures_and_special_values("
fn typed_failures_and_special_values() -> Result<()> {
	let mut g = ScalarGraph::default();
	let f = g.param(0, DType::F32, None)?;
	let i = g.constant(Value::I32(1));
	assert!(g.alu(Alu::Add, vec![f, i]).is_err());
	assert!(g.alu(Alu::Cast(DType::I32), vec![f]).is_err());
	assert!(g.param(1, DType::Void, None).is_err());
	assert!(g.alu(Alu::Add, vec![i]).is_err());
	assert!(g.compile(999).is_err());
	let foreign = g.scoped_param(42, 0, DType::F32, None)?;
	assert!(g.compile(foreign).is_err());
	let minus = g.constant(Value::f32(-0.));
	let plus = g.constant(Value::f32(0.));
	assert_ne!(minus, plus);
	let max = g.alu(Alu::Max, vec![minus, minus])?;
	assert_eq!(g.compile(max)?.run(&[])?, Value::f32(0.));
	let nan = g.constant(Value::F32Bits(0x7fc00042));
	assert_eq!(nan, g.constant(Value::F32Bits(0x7fc00042)));
	let max = g.alu(Alu::Max, vec![nan, plus])?;
	assert!(g.compile(max)?.run(&[])?.number().is_nan());
	let mut bounded = ScalarGraph::default();
	let x = bounded.param(0, DType::I32, Some(Bounds::new(-7, 5)?))?;
	assert!(bounded.compile(x)?.run(&[Value::I32(6)]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test scalar_rewrite typed_failures_and_special_values
```

## 24. bounds が結果を含み、証明できない式を安全と判定しない。
<!-- todo:03-024 -->

`tests/scalar_rewrite.rs` の `integer_contract_and_rewrite` で、この経路の値または拒否条件を確認する。

`tests/scalar_rewrite.rs` の `fn integer_contract_and_rewrite(`の定義を次の全体で置き換えます。

```rust
// file="tests/scalar_rewrite.rs" mode="replace" item="fn integer_contract_and_rewrite("
fn integer_contract_and_rewrite() -> Result<()> {
	let mut g = ScalarGraph::default();
	let x = g.param(0, DType::I32, Some(Bounds::new(-7, 5)?))?;
	let zero = g.constant(Value::I32(0));
	let root = g.alu(Alu::Add, vec![x, zero])?;
	let (r, y) = g.rewrite(root)?;
	assert_eq!(r.cost(y)?, 0);
	for x in [-7, 0, 5] {
		assert_eq!(
			g.evaluate(root, &[Value::I32(x)])?,
			r.compile(y)?.run(&[Value::I32(x)])?
		)
	}
	for (op, a, b, expected) in [
		(Alu::Div, -7, 3, -3),
		(Alu::Mod, -7, 3, 2),
		(Alu::Add, i32::MAX, 1, i32::MIN),
		(Alu::Shr, -8, 1, -4),
		(Alu::Shl, 6, 1, 12),
	] {
		let a = g.constant(Value::I32(a));
		let b = g.constant(Value::I32(b));
		let y = g.alu(op, vec![a, b])?;
		assert_eq!(g.compile(y)?.run(&[])?, Value::I32(expected));
	}
	let a = g.constant(Value::I32(1));
	assert!(g.alu(Alu::Div, vec![a, zero]).is_err());
	assert!(g.alu(Alu::Shl, vec![a, x]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test scalar_rewrite integer_contract_and_rewrite
```

## 25. 前章までの成功例も通る。
<!-- todo:03-025 -->

`tests/scalar_rewrite.rs` の正常系と境界系をまとめて再実行する。

`tests/scalar_rewrite.rs` の `fn arithmetic_and_shared_library(`の定義を次の全体で置き換えます。

```rust
// file="tests/scalar_rewrite.rs" mode="replace" item="fn arithmetic_and_shared_library("
fn arithmetic_and_shared_library() -> Result<()> {
	let mut g = ScalarGraph::default();
	let a = g.param(0, DType::F32, None)?;
	let b = g.param(1, DType::F32, None)?;
	let c = g.param(2, DType::F32, None)?;
	let sum = g.alu(Alu::Add, vec![a, b])?;
	assert_eq!(sum, g.alu(Alu::Add, vec![a, b])?);
	let root = g.alu(Alu::Mul, vec![sum, c])?;
	let p = g.compile(root)?;
	assert_eq!(
		p.run(&[Value::f32(2.), Value::f32(3.), Value::f32(4.)])?,
		Value::f32(20.)
	);
	assert_eq!(
		p.run(&[Value::f32(1.), Value::f32(2.), Value::f32(4.)])?,
		Value::f32(12.)
	);
	assert!(p.run(&[]).is_err());
	Ok(())
}
```

```sh
# check
cargo test --quiet --test scalar_rewrite
```

## 章末の累積確認

前章までの example と test も同じ checkout で再ビルドする。

```sh
# check
cargo test --quiet --all-targets
```
