# 09. kernel の境界、依存、schedule とメモリ計画を作る

前の節で動かしたコードに、今回使う演算と処理を追加します。各節の確認を実行してから次へ進みます。

参照するUOpの構成は [tinygrad c22edff](https://github.com/tinygrad/tinygrad/blob/c22edff14c79310a9b2df62a78819cf77c8289fa/tinygrad/uop/ops.py) です。最初から全機能を実装せず、使う節で増やします。

## 09.01 共有する計算と効果からスケジュールを作る

利用回数と STAGE を調べ、再利用する値を一度だけ中間バッファへ出す。CALL と AFTER の依存を並べ、読み書きの順序が曖昧なグラフを拒否する。

`src/codegen.rs` の `impl Lower` 内 の `fn scalar`を置き換える。

```rust
// file="src/codegen.rs" mode="replace" within="impl Lower" item="fn scalar"
    fn scalar(&mut self, u: &UOp, indices: &[UOp], valid: &UOp) -> Result<UOp> {
        let key = (u.id, indices.iter().map(|i| i.id).collect(), valid.id);
        if let Some(v) = self.cache.get(&key) {
            return Ok(v.clone());
        }
        let result = match (&u.op, &u.arg) {
            (Ops::Const, _) => self.emit(u.clone()),
            (Ops::Buffer, _) | (Ops::Param, _) => {
                let shape = u.shape_uops()?;
                let idx = simplify(&offset(indices, &shape))?;
                let gate = simplify(valid)?;
                self.emit(u.index(idx, gate).load())
            }
            (Ops::Reduce, Arg::Reduction { op, axes, keepdim }) => {
                let shape = u.src[0].shape_uops()?;
                let mut input = vec![];
                let mut output_axis = 0;
                for i in 0..shape.len() {
                    if axes.contains(&i) {
                        input.push(UOp::int(0));
                        if *keepdim {
                            output_axis += 1;
                        }
                    } else {
                        input.push(indices[output_axis].clone());
                        output_axis += 1;
                    }
                }
                if axes.is_empty() {
                    self.scalar(&u.src[0], &input, valid)?
                } else {
                    let identity = if *op == Ops::Add {
                        0.
                    } else {
                        f32::NEG_INFINITY
                    };
                    let (alloc, init) = self.accumulator(UOp::f32(identity));
                    let saved = self.cache.clone();
                    let old_seen = self.seen.clone();
                    let mut ranges = vec![];
                    for &a in axes {
                        let r = self.range(shape[a].clone(), Axis::Reduce);
                        input[a] = r.clone();
                        ranges.push(r);
                    }
                    let value = self.scalar(&u.src[0], &input, valid)?;
                    let mut deps = vec![init];
                    deps.extend(ranges.clone());
                    let ptr = alloc.after(deps);
                    let current = self.emit(ptr.load());
                    let update = self.emit(ptr.store(current.alu(*op, &[value])));
                    let mut end = update;
                    for r in ranges.iter().rev() {
                        end = self.close(r, end);
                    }
                    self.cache = saved;
                    self.seen = old_seen;
                    self.seen.insert(alloc.id);
                    self.seen.insert(end.id);
                    self.emit(alloc.after(vec![end]).load())
                }
            }
            (Ops::Stage, _) => self.scalar(&u.src[0], indices, valid)?,
            (op, _) if is_unary(*op) => {
                let x = self.scalar(&u.src[0], indices, valid)?;
                self.emit(UOp::new(*op, vec![x], u.arg.clone()))
            }
            (op, _) if is_binary(*op) || *op == Ops::Where => {
                let mut src = vec![];
                for x in &u.src {
                    src.push(self.scalar(x, &broadcast(indices, &x.shape_uops()?), valid)?)
                }
                self.emit(UOp::new(*op, src, u.arg.clone()))
            }
            _ => return Err(format!("rangeify cannot lower {:?}", u.op)),
        };
        self.cache.insert(key, result.clone());
        Ok(result)
    }
```

`src/effects.rs`を保存する。

```rust
// file="src/effects.rs" mode="write"
use crate::{Ops, Result, UOp};
use std::collections::{HashMap, HashSet, VecDeque};
fn state(mut node: UOp) -> Result<UOp> {
    while !matches!(node.op, Ops::Buffer | Ops::Alloc | Ops::Param | Ops::After) {
        node = node
            .src
            .first()
            .ok_or("kernel argument does not refer to storage")?
            .clone();
    }
    Ok(node)
}
pub fn order_effects(sink: &UOp) -> Result<UOp> {
    let mut visited = HashSet::new();
    let mut order = Vec::new();
    let mut stack = vec![(sink.clone(), false)];
    while let Some((u, done)) = stack.pop() {
        if done {
            order.push(u);
            continue;
        }
        if !visited.insert(u.id) {
            continue;
        }
        stack.push((u.clone(), true));
        let sources = if u.op == Ops::Call {
            &u.src[1..]
        } else {
            &u.src[..]
        };
        for src in sources.iter().rev() {
            stack.push((src.clone(), false));
        }
    }
    let mut nodes = HashMap::<u64, UOp>::new();
    let mut sequence = Vec::new();
    let mut writes = HashMap::<u64, (u64, Vec<u64>)>::new();
    let mut reads = Vec::<(u64, u64, u64)>::new();
    let mut edges = HashSet::<(u64, u64)>::new();
    for after in order.iter().filter(|u| u.op == Ops::After) {
        let previous = state(after.src.first().ok_or("AFTER missing storage")?.clone())?;
        let calls = after.src[1..]
            .iter()
            .filter(|u| u.op == Ops::Call)
            .cloned()
            .collect::<Vec<_>>();
        if after.src[1..]
            .iter()
            .any(|u| !matches!(u.op, Ops::Call | Ops::After))
        {
            return Err("schedule requires STORE effects to be split into CALLs".into());
        }
        if let Some((old, _)) = writes.insert(
            previous.id,
            (after.id, calls.iter().map(|u| u.id).collect()),
        ) {
            if old != after.id {
                return Err("sibling AFTERs supersede one storage state".into());
            }
        }
        if nodes.insert(after.id, after.clone()).is_none() {
            sequence.push(after.id);
        }
        if previous.op == Ops::After {
            edges.insert((previous.id, after.id));
        }
        for dep in after.src[1..].iter().filter(|u| u.op == Ops::After) {
            edges.insert((dep.id, after.id));
        }
        for call in calls {
            if nodes.insert(call.id, call.clone()).is_none() {
                sequence.push(call.id);
            }
            edges.insert((call.id, after.id));
            if previous.op == Ops::After {
                edges.insert((previous.id, call.id));
            }
            for argument in &call.src[1..] {
                let source = state(argument.clone())?;
                reads.push((after.id, call.id, source.id));
                if source.op == Ops::After {
                    edges.insert((source.id, call.id));
                }
            }
            for dep in after.src[1..].iter().filter(|u| u.op == Ops::After) {
                edges.insert((dep.id, call.id));
            }
        }
    }
    for (owner, reader, read_state) in reads {
        if let Some((next, writers)) = writes.get(&read_state) {
            if *next == owner {
                continue;
            }
            for &writer in writers {
                if reader != writer {
                    edges.insert((reader, writer));
                }
            }
        }
    }
    let mut children = HashMap::<u64, Vec<u64>>::new();
    let mut degree = nodes
        .keys()
        .map(|&id| (id, 0usize))
        .collect::<HashMap<_, _>>();
    let mut edges = edges.into_iter().collect::<Vec<_>>();
    edges.sort_unstable();
    for (before, after) in edges {
        if !nodes.contains_key(&before) || !nodes.contains_key(&after) {
            return Err("missing schedule dependency".into());
        }
        children.entry(before).or_default().push(after);
        *degree.get_mut(&after).unwrap() += 1;
    }
    let mut queue = sequence
        .iter()
        .filter(|id| degree[id] == 0)
        .copied()
        .collect::<VecDeque<_>>();
    let mut calls = Vec::new();
    let mut count = 0;
    while let Some(id) = queue.pop_front() {
        count += 1;
        let node = &nodes[&id];
        if node.op == Ops::Call {
            calls.push(node.clone());
        }
        for child in children.get(&id).into_iter().flatten() {
            let d = degree.get_mut(child).unwrap();
            *d -= 1;
            if *d == 0 {
                queue.push_back(*child);
            }
        }
    }
    if count != nodes.len() {
        return Err("cycle in storage read/write dependencies".into());
    }
    Ok(UOp::linear(calls))
}
```

`src/lib.rs`を追記する。

```rust
// file="src/lib.rs" mode="append"

pub mod effects;

pub mod schedule;
```

`src/runtime.rs`を保存する。

```rust
// file="src/runtime.rs" mode="write"
use crate::codegen::signature;
use crate::{
    Result, codegen::{rangeify, render},
    schedule::{Schedule, create_schedule},
    tensor::{Buffer, Tensor},
    uop::{Arg, Ops, UOp, numel},
};
use std::collections::{HashMap, HashSet};
#[derive(Clone, Debug)]
pub struct InputSpec {
    pub name: String,
    pub shape: Vec<usize>,
}
/// Compiled signatures and lifetime plans are read-only.
/// ```compile_fail
/// fn alter(e: &mut chibigrad::Executable) { e.schedule.lifetimes.clear(); }
/// ```
pub struct Executable {
    inputs: Vec<InputSpec>,
    schedule: Schedule,
    programs: Vec<UOp>,
    native: Vec<crate::cpu::CpuProgram>,
    input_nodes: Vec<UOp>,
    captured: HashMap<u64, Buffer>,
}
pub fn compile_many(outputs: &[Tensor]) -> Result<Executable> {
    unsafe { compile_many_with_transform(outputs, |linear| Ok(linear.clone())) }
}
/// # Safety
/// The transform must preserve buffer bounds, lifetimes, control-flow scope and
/// the packed signature. Native code generated from the result must be safe to execute.
pub unsafe fn compile_many_with_transform(
    outputs: &[Tensor],
    transform: impl Fn(&UOp) -> Result<UOp>,
) -> Result<Executable> {
    let debug = std::env::var_os("CHIBI_DEBUG").is_some();
    if debug {
        eprintln!("schedule starts");
    }
    let schedule = create_schedule(
        &outputs.iter().map(|x| x.uop.clone()).collect::<Vec<_>>(),
    )?;
    if debug {
        eprintln!("schedule built {} kernels", schedule.kernels.len());
    }
    let produced: HashSet<_> = schedule.lifetimes.iter().map(|l| l.buffer.id).collect();
    let mut input_nodes = vec![];
    let mut inputs = vec![];
    let mut captured = HashMap::new();
    for u in &schedule.buffers {
        if produced.contains(&u.id) {
            continue;
        }
        let Arg::Buffer { name, shape, data } = &u.arg else {
            return Err("invalid buffer argument".into());
        };
        if let Some(bits) = data {
            captured
                .insert(
                    u.id,
                    Buffer::new(
                        shape.clone(),
                        bits.iter().map(|&b| f32::from_bits(b)).collect(),
                    )?,
                );
        } else {
            input_nodes.push(u.clone());
            inputs
                .push(InputSpec {
                    name: name.clone(),
                    shape: shape.clone(),
                });
        }
    }
    let mut programs = vec![];
    let mut native = vec![];
    for (i, kernel) in schedule.kernels.iter().enumerate() {
        if debug {
            eprintln!("kernel {i} rangeify");
        }
        let linear = transform(&rangeify(&kernel.sink)?)?;
        if debug {
            eprintln!("kernel {i} render {} UOps", linear.src.len());
        }
        let entry = format!("chibi_kernel_{i}");
        let source = render(&linear, &entry)?;
        let stages = vec![
            kernel.sink.clone(), linear, UOp::new(Ops::Source, vec![], Arg::Text(source
            .clone())),
        ];
        let stages = {
            if debug {
                eprintln!("kernel {i} compile {} source bytes", source.len());
            }
            let compiled = unsafe { crate::cpu::CpuProgram::compile(&source, &entry) }?;
            let mut stages = stages;
            stages
                .push(
                    UOp::new(Ops::Binary, vec![], Arg::Bytes(compiled.object().to_vec())),
                );
            native.push(compiled);
            stages
        };
        programs.push(UOp::new(Ops::Program, stages, Arg::Text(entry)));
    }
    Ok(Executable {
        inputs,
        schedule,
        programs,
        native,
        input_nodes,
        captured,
    })
}
impl Executable {
    pub fn inputs(&self) -> &[InputSpec] {
        &self.inputs
    }
    pub fn programs(&self) -> &[UOp] {
        &self.programs
    }
    pub fn schedule(&self) -> &Schedule {
        &self.schedule
    }
    pub fn kernel_count(&self) -> usize {
        self.programs.len()
    }
    pub fn sources(&self) -> Vec<&str> {
        self.programs
            .iter()
            .filter_map(|p| p.src.iter().find(|s| s.op == Ops::Source))
            .filter_map(|s| {
                if let Arg::Text(text) = &s.arg { Some(text.as_str()) } else { None }
            })
            .collect()
    }
    pub fn run(
        &self,
        inputs: &[Buffer],
        _vars: &HashMap<String, i64>,
    ) -> Result<Vec<Buffer>> {
        if inputs.len() != self.inputs.len() {
            return Err(
                format!(
                    "expected {} input buffers, received {}", self.inputs.len(), inputs
                    .len()
                ),
            );
        }
        for ((buffer, spec), node) in inputs
            .iter()
            .zip(&self.inputs)
            .zip(&self.input_nodes)
        {
            let expected = node.shape()?;
            if buffer.shape != expected || buffer.data.len() != numel(&expected) {
                return Err(
                    format!(
                        "input {} expected {:?}, received {:?} with {} elements", spec
                        .name, expected, buffer.shape, buffer.data.len()
                    ),
                );
            }
        }
        let mut external: HashMap<u64, &Buffer> = self
            .captured
            .iter()
            .map(|(&id, b)| (id, b))
            .collect();
        for (u, b) in self.input_nodes.iter().zip(inputs) {
            external.insert(u.id, b);
        }
        let mut sizes = vec![];
        for life in &self.schedule.lifetimes {
            if sizes.len() <= life.slot {
                sizes.resize(life.slot + 1, 0)
            }
            sizes[life.slot] = sizes[life.slot].max(life.elements);
        }
        let slots: Vec<Vec<f32>> = sizes.iter().map(|&n| vec![0.; n]).collect();
        let mut slots = slots;
        let locations: HashMap<_, _> = self
            .schedule
            .lifetimes
            .iter()
            .map(|l| (l.buffer.id, l.slot))
            .collect();
        {
            for (i, program) in self.programs.iter().enumerate() {
                let linear = &program.src[1];
                let sig = signature(linear)?;
                let count = sig.buffers.iter().map(|x| x.0).max().map_or(0, |x| x + 1);
                let mut packed = vec![0u8; count * 8];
                let mut validation_status = 0f32;
                for (slot, u) in sig.buffers {
                    let pointer = if matches!(
                        & u.arg, Arg::Buffer { name, data : Some(_), .. } if name ==
                        "__chibi_error"
                    ) {
                        &mut validation_status as *mut f32 as usize
                    } else if let Some(b) = external.get(&u.id) {
                        b.data.as_ptr() as usize
                    } else {
                        let loc = locations
                            .get(&u.id)
                            .ok_or("missing runtime allocation")?;
                        slots[*loc].as_mut_ptr() as usize
                    };
                    packed[slot * 8..slot * 8 + 8]
                        .copy_from_slice(&(pointer as u64).to_ne_bytes());
                }
                unsafe { self.native[i].run(&mut packed) }?;
                if validation_status != 0. {
                    return Err(
                        "gather/scatter index must be finite, integral, and within bounds"
                            .into(),
                    );
                }
            }
            self.schedule
                .outputs
                .iter()
                .map(|u| {
                    let slot = locations[&u.id];
                    let shape = u.shape()?;
                    Buffer::new(shape.clone(), slots[slot][..numel(&shape)].to_vec())
                })
                .collect()
        }
    }
}
```

`src/schedule.rs`を保存する。

```rust
// file="src/schedule.rs" mode="write"
use crate::{
    Result,
    uop::{Arg, Ops, UOp, is_binary, is_unary, numel},
};
use std::collections::{HashMap, HashSet};
#[derive(Clone, Debug)]
pub struct KernelPlan {
    pub sink: UOp,
    pub call: UOp,
    pub reads: Vec<UOp>,
    pub writes: Vec<UOp>,
}
#[derive(Clone, Debug)]
pub struct Lifetime {
    pub buffer: UOp,
    pub first: usize,
    pub last: usize,
    pub slot: usize,
    pub elements: usize,
}
#[derive(Clone, Debug)]
pub struct Schedule {
    pub dag: UOp,
    pub linear: UOp,
    pub kernels: Vec<KernelPlan>,
    pub buffers: Vec<UOp>,
    pub outputs: Vec<UOp>,
    pub lifetimes: Vec<Lifetime>,
}
fn substitute(root: &UOp, replacements: &HashMap<u64, UOp>) -> UOp {
    let mut memo: HashMap<u64, UOp> = HashMap::new();
    for u in root.toposort() {
        let v = replacements
            .get(&u.id)
            .cloned()
            .unwrap_or_else(|| u.replace(u.src.iter().map(|s| memo[&s.id].clone()).collect()));
        memo.insert(u.id, v);
    }
    memo[&root.id].clone()
}
pub(crate) fn tensor_toposort(root: &UOp) -> Vec<UOp> {
    let mut seen = HashSet::new();
    let mut order = Vec::new();
    let mut stack = vec![(root.clone(), false)];
    while let Some((u, done)) = stack.pop() {
        if done {
            order.push(u);
            continue;
        }
        if !seen.insert(u.id) {
            continue;
        }
        stack.push((u.clone(), true));
        let sources = if u.op == Ops::Call {
            u.src.get(1..).unwrap_or_default().iter().collect()
        } else {
            u.ordered_sources()
        };
        for s in sources.into_iter().rev() {
            stack.push((s.clone(), false));
        }
    }
    order
}
pub fn create_schedule(outputs: &[UOp]) -> Result<Schedule> {
    if outputs.is_empty() {
        return Err("compile requires at least one output".into());
    }
    let sink = UOp::sink(outputs.to_vec());
    let topo = tensor_toposort(&sink);
    for u in &topo {
        if !matches!(u.op, Ops::Sink | Ops::Call) {
            u.shape()?;
        }
    }
    let mut users: HashMap<u64, usize> = HashMap::new();
    for u in &topo {
        for src in &u.src {
            *users.entry(src.id).or_default() += 1;
        }
    }
    let output_ids: HashSet<_> = outputs.iter().map(|u| u.id).collect();
    let mut buffers: Vec<UOp> = topo
        .iter()
        .filter(|u| u.op == Ops::Buffer)
        .cloned()
        .collect();
    let mut replacements = HashMap::new();
    let mut producers: HashMap<u64, UOp> = HashMap::new();
    let mut kernels = vec![];
    fn add_kernel(
        values: Vec<(UOp, UOp)>,
        kernels: &mut Vec<KernelPlan>,
        producers: &mut HashMap<u64, UOp>,
    ) -> Result<()> {
        let writes: Vec<_> = values.iter().map(|(b, _)| b.clone()).collect();
        let stores = values
            .into_iter()
            .map(|(b, v)| UOp::new(Ops::Store, vec![b, v], Arg::None))
            .collect();
        let sink = UOp::sink(stores);
        let write_ids: HashSet<_> = writes.iter().map(|b| b.id).collect();
        let reads: Vec<_> = sink
            .toposort()
            .into_iter()
            .filter(|u| u.op == Ops::Buffer && !write_ids.contains(&u.id))
            .collect();
        let mut args = vec![sink.clone()];
        args.extend(reads.iter().map(|b| {
            if let Some(call) = producers.get(&b.id) {
                b.after(vec![call.clone()])
            } else {
                b.clone()
            }
        }));
        args.extend(writes.clone());
        let call = UOp::new(Ops::Call, args, Arg::Id(kernels.len()));
        for b in &writes {
            producers.insert(b.id, call.clone());
        }
        kernels.push(KernelPlan {
            sink,
            call,
            reads,
            writes,
        });
        Ok(())
    }
    for u in &topo {
        let shared = users.get(&u.id).copied().unwrap_or(0) > 1
            && (is_binary(u.op) || is_unary(u.op) || matches!(u.op, Ops::Reduce));
        let nonempty_reduce = matches!(
            & u.arg, Arg::Reduction { axes,.. } if ! axes.is_empty()
        );
        if (u.op == Ops::Stage || shared || nonempty_reduce) && !output_ids.contains(&u.id) {
            let value = if u.op == Ops::Stage {
                substitute(&u.src[0], &replacements)
            } else {
                substitute(u, &replacements)
            };
            let buffer = UOp::new(
                Ops::Buffer,
                vec![],
                Arg::Buffer {
                    name: format!("temp_{}", u.id),
                    shape: u.shape()?,
                    data: None,
                },
            );
            add_kernel(vec![(buffer.clone(), value)], &mut kernels, &mut producers)?;
            buffers.push(buffer.clone());
            replacements.insert(u.id, buffer);
        }
    }
    let mut output_buffers = vec![];
    let mut final_values = vec![];
    for (i, u) in outputs.iter().enumerate() {
        let b = UOp::new(
            Ops::Buffer,
            vec![],
            Arg::Buffer {
                name: format!("output_{}_{}", sink.id, i),
                shape: u.shape()?,
                data: None,
            },
        );
        buffers.push(b.clone());
        output_buffers.push(b.clone());
        final_values.push((b, substitute(u, &replacements)));
    }
    add_kernel(final_values, &mut kernels, &mut producers)?;
    let dag = UOp::sink(
        output_buffers
            .iter()
            .map(|b| b.after(vec![producers[&b.id].clone()]))
            .collect(),
    );
    let linear = crate::effects::order_effects(&dag)?;
    kernels = linear
        .src
        .iter()
        .map(|call| {
            kernels
                .iter()
                .find(|kernel| kernel.call == *call)
                .cloned()
                .ok_or("scheduled CALL missing kernel".to_string())
        })
        .collect::<Result<Vec<_>>>()?;
    let output_set: HashSet<_> = output_buffers.iter().map(|b| b.id).collect();
    let mut lives = vec![];
    for b in buffers.iter().filter(|b| producers.contains_key(&b.id)) {
        let first = kernels
            .iter()
            .position(|k| k.writes.contains(b))
            .ok_or("missing buffer writer")?;
        let last = if output_set.contains(&b.id) {
            kernels.len()
        } else {
            kernels
                .iter()
                .rposition(|k| k.reads.contains(b))
                .unwrap_or(first)
        };
        lives.push(Lifetime {
            buffer: b.clone(),
            first,
            last,
            slot: 0,
            elements: numel(&b.shape()?),
        });
    }
    lives.sort_by_key(|l| l.first);
    for (i, life) in lives.iter_mut().enumerate() {
        life.slot = i;
    }
    Ok(Schedule {
        dag,
        linear,
        kernels,
        buffers,
        outputs: output_buffers,
        lifetimes: lives,
    })
}
```

`src/tensor.rs` の `impl Tensor` 内を追記する。

```rust
// file="src/tensor.rs" mode="append" within="impl Tensor"
    pub fn stage(&self) -> Self {
        Self::from_uop(UOp::new(Ops::Stage, vec![self.uop.clone()], Arg::None))
    }
```

`src/uir.rs` の `fn op`を置き換える。

```rust
// file="src/uir.rs" mode="replace" item="fn op"
fn op(s: &str) -> Result<Ops> {
    [
        Ops::Invalid,
        Ops::Const,
        Ops::Buffer,
        Ops::Alloc,
        Ops::Param,
        Ops::Cast,
        Ops::Bitcast,
        Ops::Shl,
        Ops::Shr,
        Ops::Xor,
        Ops::Add,
        Ops::Sub,
        Ops::Mul,
        Ops::Div,
        Ops::Mod,
        Ops::Neg,
        Ops::Max,
        Ops::Lt,
        Ops::Eq,
        Ops::And,
        Ops::Or,
        Ops::Where,
        Ops::Stack,
        Ops::Reduce,
        Ops::Stage,
        Ops::Range,
        Ops::Index,
        Ops::Load,
        Ops::Store,
        Ops::End,
        Ops::After,
        Ops::Call,
        Ops::Sink,
        Ops::Linear,
        Ops::Program,
        Ops::Source,
        Ops::Binary,
    ]
    .into_iter()
    .find(|o| format!("{o:?}") == s)
    .ok_or("unknown operation".into())
}
```

`src/uop.rs` の `pub enum Ops`を置き換える。

```rust
// file="src/uop.rs" mode="replace" item="pub enum Ops"
pub enum Ops {
    Invalid,
    Const,
    Buffer,
    Alloc,
    Param,
    Cast,
    Bitcast,
    Shl,
    Shr,
    Xor,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Neg,
    Max,
    Lt,
    Eq,
    And,
    Or,
    Where,
    Stack,
    Reduce,
    Stage,
    Range,
    Index,
    Load,
    Store,
    End,
    After,
    Call,
    Sink,
    Linear,
    Program,
    Source,
    Binary,
}
```

`src/uop.rs` の `impl UOp` 内 の `fn infer_shape`を置き換える。

```rust
// file="src/uop.rs" mode="replace" within="impl UOp" item="fn infer_shape"
    fn infer_shape(&self) -> Result<Vec<usize>> {
        let err = || format!("invalid {:?} arguments", self.op);
        let source = || self.src.first().ok_or_else(err)?.shape();
        match (&self.op, &self.arg) {
            (Ops::Invalid, Arg::Text(s)) => Err(s.clone()),
            (Ops::Buffer, Arg::Buffer { shape, data, .. }) => {
                checked_numel(shape)?;
                if data.as_ref().is_some_and(|d| d.len() != numel(shape)) {
                    return Err("buffer data length disagrees with shape".into());
                }
                Ok(shape.clone())
            }
            (Ops::Param, Arg::Slot { .. }) => Ok(vec![]),
            (Ops::Param, Arg::ScalarParam { .. }) => Ok(vec![]),
            (Ops::Const, _) | (Ops::Range, _) | (Ops::Load, _) => Ok(vec![]),
            (Ops::Reduce, Arg::Reduction { op, axes, keepdim }) => {
                let old = source()?;
                if !matches!(op, Ops::Add | Ops::Max)
                    || axes.iter().any(|&a| a >= old.len())
                    || axes.iter().copied().collect::<HashSet<_>>().len() != axes.len()
                {
                    return Err("invalid reduction axes or operation".into());
                }
                Ok(old
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &n)| {
                        if axes.contains(&i) {
                            keepdim.then_some(1)
                        } else {
                            Some(n)
                        }
                    })
                    .collect())
            }
            (Ops::Where, _) => {
                if self.src.len() != 3 {
                    return Err(err());
                }
                let s = broadcast_shape(&self.src[0].shape()?, &self.src[1].shape()?)?;
                broadcast_shape(&s, &self.src[2].shape()?)
            }
            (op, _) if is_binary(*op) => {
                if self.src.len() != 2 {
                    return Err(err());
                }
                broadcast_shape(&self.src[0].shape()?, &self.src[1].shape()?)
            }
            (op, _) if is_unary(*op) || matches!(op, Ops::Stage | Ops::After) => source(),
            _ => Err(err()),
        }
    }
```

`tests/schedule_effects.rs`を保存する。

```rust
// file="tests/schedule_effects.rs" mode="write"
use chibigrad::{Arg, Ops, UOp, effects::order_effects};
fn buffer(name: &str) -> UOp {
    UOp::new(
        Ops::Buffer,
        vec![],
        Arg::Buffer {
            name: name.into(),
            shape: vec![1],
            data: None,
        },
    )
}
fn call(id: usize, args: Vec<UOp>) -> UOp {
    let mut sources = vec![UOp::sink(vec![])];
    sources.extend(args);
    UOp::new(Ops::Call, sources, Arg::Id(id))
}
#[test]
fn raw_and_war_dependencies_override_incidental_graph_order() {
    let x = buffer("x");
    let y = buffer("y");
    let write = call(0, vec![x.clone()]);
    let new_x = x.after(vec![write.clone()]);
    let read = call(1, vec![new_x.clone(), y.clone()]);
    let new_y = y.after(vec![read.clone()]);
    assert_eq!(
        order_effects(&UOp::sink(vec![new_y, new_x])).unwrap().src,
        [write.clone(), read]
    );
    let old_reader = call(2, vec![x.clone(), y.clone()]);
    let read_done = y.after(vec![old_reader.clone()]);
    let overwritten = x.after(vec![write.clone()]);
    assert_eq!(
        order_effects(&UOp::sink(vec![overwritten, read_done]))
            .unwrap()
            .src,
        [old_reader, write]
    );
}
#[test]
fn sibling_writes_and_derived_cycles_are_rejected() {
    let x = buffer("x");
    let a = call(10, vec![x.clone()]);
    let b = call(11, vec![x.clone()]);
    assert!(
        order_effects(&UOp::sink(vec![
            x.after(vec![a.clone()]),
            x.after(vec![b.clone()])
        ]))
        .is_err()
    );
    let first = x.after(vec![a]);
    let second = first.after(vec![b]);
    assert!(order_effects(&UOp::sink(vec![second])).is_err());
}
```

`tests/schedule_stages.rs`を保存する。

```rust
// file="tests/schedule_stages.rs" mode="write"
use chibigrad::Tensor;
#[test]
fn stages_run_in_dependency_order() {
    let x = Tensor::from_slice(&[1., 2.], &[2]);
    let a = x.add(&Tensor::scalar(1.)).stage();
    let y = a.mul(&a).stage().add(&a);
    let p = y.compile().unwrap();
    assert!(p.kernel_count() >= 3);
    assert_eq!(
        p.run(&[], &std::collections::HashMap::new()).unwrap()[0].data,
        [6., 12.]
    );
}
```

```sh
# check step=09.01
cargo test --offline
```

複数のカーネルを依存順に実行し、RAW・WAR の順序、競合書込みと循環の拒否を確認する。

## 09.02 使い終えた一時領域を再利用する

各バッファの最初と最後の利用カーネルを記録する。寿命が重ならず容量が足りるスロットだけを再利用する。出力の寿命は実行終了まで延ばす。

`src/schedule.rs` の `pub fn create_schedule`を置き換える。

```rust
// file="src/schedule.rs" mode="replace" item="pub fn create_schedule"
pub fn create_schedule(outputs: &[UOp]) -> Result<Schedule> {
    if outputs.is_empty() {
        return Err("compile requires at least one output".into());
    }
    let sink = UOp::sink(outputs.to_vec());
    let topo = tensor_toposort(&sink);
    for u in &topo {
        if !matches!(u.op, Ops::Sink | Ops::Call) {
            u.shape()?;
        }
    }
    let mut users: HashMap<u64, usize> = HashMap::new();
    for u in &topo {
        for src in &u.src {
            *users.entry(src.id).or_default() += 1;
        }
    }
    let output_ids: HashSet<_> = outputs.iter().map(|u| u.id).collect();
    let mut buffers: Vec<UOp> = topo
        .iter()
        .filter(|u| u.op == Ops::Buffer)
        .cloned()
        .collect();
    let mut replacements = HashMap::new();
    let mut producers: HashMap<u64, UOp> = HashMap::new();
    let mut kernels = vec![];
    fn add_kernel(
        values: Vec<(UOp, UOp)>,
        kernels: &mut Vec<KernelPlan>,
        producers: &mut HashMap<u64, UOp>,
    ) -> Result<()> {
        let writes: Vec<_> = values.iter().map(|(b, _)| b.clone()).collect();
        let stores = values
            .into_iter()
            .map(|(b, v)| UOp::new(Ops::Store, vec![b, v], Arg::None))
            .collect();
        let sink = UOp::sink(stores);
        let write_ids: HashSet<_> = writes.iter().map(|b| b.id).collect();
        let reads: Vec<_> = sink
            .toposort()
            .into_iter()
            .filter(|u| u.op == Ops::Buffer && !write_ids.contains(&u.id))
            .collect();
        let mut args = vec![sink.clone()];
        args.extend(reads.iter().map(|b| {
            if let Some(call) = producers.get(&b.id) {
                b.after(vec![call.clone()])
            } else {
                b.clone()
            }
        }));
        args.extend(writes.clone());
        let call = UOp::new(Ops::Call, args, Arg::Id(kernels.len()));
        for b in &writes {
            producers.insert(b.id, call.clone());
        }
        kernels.push(KernelPlan {
            sink,
            call,
            reads,
            writes,
        });
        Ok(())
    }
    for u in &topo {
        let shared = users.get(&u.id).copied().unwrap_or(0) > 1
            && (is_binary(u.op) || is_unary(u.op) || matches!(u.op, Ops::Reduce));
        let nonempty_reduce = matches!(
            & u.arg, Arg::Reduction { axes,.. } if ! axes.is_empty()
        );
        if (u.op == Ops::Stage || shared || nonempty_reduce) && !output_ids.contains(&u.id) {
            let value = if u.op == Ops::Stage {
                substitute(&u.src[0], &replacements)
            } else {
                substitute(u, &replacements)
            };
            let buffer = UOp::new(
                Ops::Buffer,
                vec![],
                Arg::Buffer {
                    name: format!("temp_{}", u.id),
                    shape: u.shape()?,
                    data: None,
                },
            );
            add_kernel(vec![(buffer.clone(), value)], &mut kernels, &mut producers)?;
            buffers.push(buffer.clone());
            replacements.insert(u.id, buffer);
        }
    }
    let mut output_buffers = vec![];
    let mut final_values = vec![];
    for (i, u) in outputs.iter().enumerate() {
        let b = UOp::new(
            Ops::Buffer,
            vec![],
            Arg::Buffer {
                name: format!("output_{}_{}", sink.id, i),
                shape: u.shape()?,
                data: None,
            },
        );
        buffers.push(b.clone());
        output_buffers.push(b.clone());
        final_values.push((b, substitute(u, &replacements)));
    }
    add_kernel(final_values, &mut kernels, &mut producers)?;
    let dag = UOp::sink(
        output_buffers
            .iter()
            .map(|b| b.after(vec![producers[&b.id].clone()]))
            .collect(),
    );
    let linear = crate::effects::order_effects(&dag)?;
    kernels = linear
        .src
        .iter()
        .map(|call| {
            kernels
                .iter()
                .find(|kernel| kernel.call == *call)
                .cloned()
                .ok_or("scheduled CALL missing kernel".to_string())
        })
        .collect::<Result<Vec<_>>>()?;
    let output_set: HashSet<_> = output_buffers.iter().map(|b| b.id).collect();
    let mut lives = vec![];
    for b in buffers.iter().filter(|b| producers.contains_key(&b.id)) {
        let first = kernels
            .iter()
            .position(|k| k.writes.contains(b))
            .ok_or("missing buffer writer")?;
        let last = if output_set.contains(&b.id) {
            kernels.len()
        } else {
            kernels
                .iter()
                .rposition(|k| k.reads.contains(b))
                .unwrap_or(first)
        };
        lives.push(Lifetime {
            buffer: b.clone(),
            first,
            last,
            slot: 0,
            elements: numel(&b.shape()?),
        });
    }
    lives.sort_by_key(|l| l.first);
    let mut slots: Vec<(usize, usize)> = vec![];
    for life in &mut lives {
        if let Some((i, slot)) = slots
            .iter_mut()
            .enumerate()
            .find(|(_, s)| s.0 < life.first && s.1 >= life.elements)
        {
            life.slot = i;
            slot.0 = life.last;
        } else {
            life.slot = slots.len();
            slots.push((life.last, life.elements));
        }
    }
    Ok(Schedule {
        dag,
        linear,
        kernels,
        buffers,
        outputs: output_buffers,
        lifetimes: lives,
    })
}
```

`tests/schedule.rs`を保存する。

```rust
// file="tests/schedule.rs" mode="write"
use chibigrad::{Tensor, compile_many};
#[test]
fn schedule_stages_shared_values_and_reuses_dead_slots() {
    let x = Tensor::from_slice(&[1., 2., 3., 4.], &[4]);
    let a = x.add(&Tensor::scalar(1.)).stage();
    let b = a.mul(&Tensor::scalar(2.)).stage();
    let c = b.add(&Tensor::scalar(3.)).stage();
    let d = c.mul(&Tensor::scalar(4.)).stage();
    let plan = compile_many(&[d]).unwrap();
    assert!(plan.kernel_count() >= 4);
    let lives = &plan.schedule().lifetimes;
    assert!(lives.iter().enumerate().any(|(i, a)| {
        lives[i + 1..]
            .iter()
            .any(|b| a.slot == b.slot && a.last < b.first)
    }));
    assert_eq!(
        plan.run(&[], &std::collections::HashMap::new()).unwrap()[0].data,
        [28., 36., 44., 52.]
    );
}
```

```sh
# check step=09.02
cargo test --offline
```

4 段の STAGE が [28,36,44,52] を返し、重ならない中間バッファだけが同じスロットを共有する。

## 章末の累積確認

前の章までのテストとexampleも再ビルドします。

```sh
# check
cargo test --quiet --all-targets
```

## 要件との対応

### 09-001 元グラフから consumer 数を集める。

<!-- todo:09-001 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-002 単一消費先の ADD と MUL の境界を materialize しない。

<!-- todo:09-002 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-003 lower で同じ出力座標を producer へ伝える。

<!-- todo:09-003 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-004 `(x+1)*2` を単一 kernel で `[4,6,8]` にする。

<!-- todo:09-004 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-005 前章の非融合経路とも出力を比較する。

<!-- todo:09-005 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-006 `sum(x*w+1)` の reduction 座標を要素演算へ伝える。

<!-- todo:09-006 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-007 reduction の外側出力と内側 k のスコープを区別する。

<!-- todo:09-007 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-008 LOAD、MUL、ADD、accumulator 更新を同じ body に作る。

<!-- todo:09-008 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-009 `x=[1,2,3,4]`, `w=[2,3,4,5]` の結果を `44` にする。

<!-- todo:09-009 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-010 入力側の一時配列がなく、一つの reduction kernel になることを確認する。

<!-- todo:09-010 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-011 明示 STAGE を非融合境界として扱う。

<!-- todo:09-011 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-012 producer の出力を ALLOC、STORE、AFTER に変える。

<!-- todo:09-012 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-013 consumer が新しい PARAM から保存済みの値を読むようにする。

<!-- todo:09-013 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-014 `(STAGE(x+1))*2` を二 kernel で `[4,6,8]` にする。

<!-- todo:09-014 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-015 `t=x+1; y=t*2; z=t+3` を三 kernel で実行する。

<!-- todo:09-015 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-016 各 body の外部 storage と scalar を PARAM に正規化する。

<!-- todo:09-016 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-017 kernel の署名、CALL 引数、出力 state を照合する verifier を追加する。

<!-- todo:09-017 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-018 producer と consumer の辺から実行順を作り、循環なら拒否する。

<!-- todo:09-018 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-019 既存の CPU compile と run を Plan の kernel 列へ接続する。

<!-- todo:09-019 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-020 同じ Plan に新しい重みを渡して、二回目の出力を確認する。

<!-- todo:09-020 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-021 body の未束縛 PARAM、自由 RANGE、STORE 先の漏れを拒否する。

<!-- todo:09-021 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-022 外側 kernel の NodeId が body に直接残った例を拒否する。

<!-- todo:09-022 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-023 二つの違う座標で同じ producer を使う fixture を実行する。

<!-- todo:09-023 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-024 元グラフを run の前後で dump し、構造が変わらないことを確認する。

<!-- todo:09-024 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-025 入力の長さと dtype が署名と違う場合、実行開始前に拒否する。

<!-- todo:09-025 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-026 chain は一 kernel、STAGE は二 kernel、fanout は三 kernel になる。

<!-- todo:09-026 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-027 非融合と融合の値が `[4,6,8]` で一致する。

<!-- todo:09-027 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-028 fanout の二出力が `[4,6,8]` と `[5,6,7]` になる。

<!-- todo:09-028 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-029 同じ Plan の再実行で `44` から `14` へ変わり、再コンパイルしない。

<!-- todo:09-029 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-030 閉じた body に未束縛の storage と RANGE が残らない。

<!-- todo:09-030 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-031 dump で各 consumer が producer 完了後の state を読む。

<!-- todo:09-031 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 09-032 純粋な元グラフが run 前後で同じ構造を保つ。

<!-- todo:09-032 -->

関連する実装と確認は本章の `09.01`, `09.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。
