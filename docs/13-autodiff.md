# 13. UOp の逆伝播から勾配の式を作る

前の節で動かしたコードに、今回使う演算と処理を追加します。各節の確認を実行してから次へ進みます。

参照するUOpの構成は [tinygrad c22edff](https://github.com/tinygrad/tinygrad/blob/c22edff14c79310a9b2df62a78819cf77c8289fa/tinygrad/uop/ops.py) です。最初から全機能を実装せず、使う節で増やします。

## 13.01 逆向き走査で勾配の UOp を作る

出力の勾配を 1 とし、DAG を逆順に歩いて各入力へ寄与を加える。共有されたノードの寄与は加算し、ブロードキャストした軸は元の shape に和を取って戻す。勾配も Tensor なので同じコンパイラで実行できる。

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
            (Ops::Reshape, _) => {
                let linear = offset(indices, &u.shape_uops()?);
                self.scalar(&u.src[0], &coords(&linear, &u.src[0].shape_uops()?), valid)?
            }
            (Ops::Permute, Arg::Axes(axes)) => {
                let mut input = indices.to_vec();
                for (i, &a) in axes.iter().enumerate() {
                    input[a] = indices[i].clone();
                }
                self.scalar(&u.src[0], &input, valid)?
            }
            (Ops::Expand, _) => {
                let shape = u.src[0].shape_uops()?;
                self.scalar(&u.src[0], &broadcast(indices, &shape), valid)?
            }
            (Ops::Shrink, Arg::Pairs(pairs)) => {
                let input: Vec<_> = indices
                    .iter()
                    .zip(pairs)
                    .map(|(i, &(start, _))| i.add(&UOp::int(start as i64)))
                    .collect();
                self.scalar(&u.src[0], &input, valid)?
            }
            (Ops::Pad, Arg::Pairs(pairs)) => {
                let shape = u.src[0].shape_uops()?;
                let input: Vec<_> = indices
                    .iter()
                    .zip(pairs)
                    .map(|(i, &(start, _))| i.sub(&UOp::int(start as i64)))
                    .collect();
                let gate = input.iter().zip(shape).fold(valid.clone(), |v, (i, n)| {
                    v.and(&UOp::int(-1).lt(i)).and(&i.lt(&n))
                });
                let x = self.scalar(&u.src[0], &input, &gate)?;
                self.emit(gate.where_(&x, &UOp::f32(0.)))
            }
            (Ops::Flip, Arg::Axes(axes)) => {
                let shape = u.shape_uops()?;
                let input = indices
                    .iter()
                    .enumerate()
                    .map(|(i, x)| {
                        if axes.contains(&i) {
                            shape[i].sub(&UOp::int(1)).sub(x)
                        } else {
                            x.clone()
                        }
                    })
                    .collect::<Vec<_>>();
                self.scalar(&u.src[0], &input, valid)?
            }
            (Ops::Stack, Arg::Axis(axis)) => {
                let mut input = indices.to_vec();
                let chosen = input.remove(*axis);
                let mut value = UOp::f32(0.);
                for (i, src) in u.src.iter().enumerate() {
                    let gate = chosen.eq(&UOp::int(i as i64));
                    let part = self.scalar(src, &input, &valid.and(&gate))?;
                    value = self.emit(gate.where_(&part, &value));
                }
                value
            }
            (Ops::Gather, Arg::Axis(axis)) => {
                let ishape = u.src[1].shape_uops()?;
                let raw = self.scalar(&u.src[1], &indices[*axis..axis + ishape.len()], valid)?;
                let shape = u.src[0].shape_uops()?;
                let index = self.checked_index(raw, shape[*axis].clone(), valid);
                let gate = valid
                    .and(&UOp::int(-1).lt(&index))
                    .and(&index.lt(&shape[*axis]));
                let mut input = indices[..*axis].to_vec();
                input.push(index);
                input.extend_from_slice(&indices[axis + ishape.len()..]);
                self.scalar(&u.src[0], &input, &gate)?
            }
            (Ops::ScatterAdd, Arg::Axis(axis)) => {
                let base = self.scalar(&u.src[0], indices, valid)?;
                let (alloc, init) = self.accumulator(base);
                let saved = self.cache.clone();
                let old_seen = self.seen.clone();
                let shape = u.src[1].shape_uops()?;
                let r = self.range(product(&shape), Axis::Reduce);
                let ic = coords(&r, &shape);
                let raw = self.scalar(&u.src[1], &ic, valid)?;
                let index = self.checked_index(raw, u.src[0].shape_uops()?[*axis].clone(), valid);
                let mut vi = indices[..*axis].to_vec();
                vi.extend(ic);
                vi.extend_from_slice(&indices[axis + 1..]);
                let value = self.scalar(&u.src[2], &vi, valid)?;
                let picked = index.eq(&indices[*axis]).where_(&value, &UOp::f32(0.));
                let ptr = alloc.after(vec![init, r.clone()]);
                let current = self.emit(ptr.load());
                let update = self.emit(ptr.store(current.add(&picked)));
                let end = self.close(&r, update);
                self.cache = saved;
                self.seen = old_seen;
                self.seen.insert(alloc.id);
                self.seen.insert(end.id);
                self.emit(alloc.after(vec![end]).load())
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
            (Ops::Detach | Ops::Stage, _) => self.scalar(&u.src[0], indices, valid)?,
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

`src/tensor.rs` の `impl Tensor` 内を追記する。

```rust
// file="src/tensor.rs" mode="append" within="impl Tensor"
    pub fn detach(&self) -> Self {
        Self::from_uop(UOp::new(Ops::Detach, vec![self.uop.clone()], Arg::None))
    }

    pub fn gradient(&self, target: &Self) -> Result<Self> {
        Ok(self.gradients(std::slice::from_ref(target))?.remove(0))
    }

    pub fn gradients(&self, targets: &[Self]) -> Result<Vec<Self>> {
        self.shape()?;
        for target in targets {
            target.shape()?;
        }
        let nodes = self.uop.toposort();
        let mut grads = HashMap::from([(self.uop.id, Self::constant(1., &self.shape()?))]);
        for node in nodes.into_iter().rev() {
            let Some(g) = grads.get(&node.id).cloned() else {
                continue;
            };
            let x: Vec<Self> = node.src.iter().cloned().map(Self::from_uop).collect();
            let y = Self::from_uop(node.clone());
            let contributions: Vec<(usize, Self)> = match node.op {
                Ops::Buffer | Ops::Const | Ops::Detach => vec![],
                Ops::Add => vec![(0, g.clone()), (1, g)],
                Ops::Sub => vec![(0, g.clone()), (1, g.neg())],
                Ops::Mul => vec![(0, g.mul(&x[1])), (1, g.mul(&x[0]))],
                Ops::Div => {
                    vec![
                        (0, g.div(&x[1])),
                        (1, g.mul(&x[0]).div(&x[1].mul(&x[1])).neg()),
                    ]
                }
                Ops::Neg => vec![(0, g.neg())],
                Ops::Exp => vec![(0, g.mul(&y))],
                Ops::Log => vec![(0, g.div(&x[0]))],
                Ops::Exp2 => {
                    vec![(0, g.mul(&y).mul(&Self::scalar(std::f32::consts::LN_2)))]
                }
                Ops::Log2 => {
                    vec![(0, g.div(&x[0].mul(&Self::scalar(std::f32::consts::LN_2))))]
                }
                Ops::Sin => vec![(0, g.mul(&x[0].cos()))],
                Ops::Cos => vec![(0, g.mul(&x[0].sin()).neg())],
                Ops::Sqrt => vec![(0, g.div(&y.mul(&Self::scalar(2.))))],
                Ops::Max => {
                    let left = x[0].eq(&y).cast(DType::F32);
                    let right = x[1].eq(&y).cast(DType::F32);
                    let den = left.add(&right);
                    vec![(0, g.mul(&left).div(&den)), (1, g.mul(&right).div(&den))]
                }
                Ops::Where => {
                    vec![
                        (1, x[0].where_(&g, &Self::scalar(0.))),
                        (2, x[0].where_(&Self::scalar(0.), &g)),
                    ]
                }
                Ops::Lt | Ops::Eq | Ops::And | Ops::Or => vec![],
                Ops::Cast => {
                    if matches!(x[0].uop.dtype(), DType::F32 | DType::Ptr)
                        && node.dtype() == DType::F32
                    {
                        vec![(0, g)]
                    } else {
                        vec![]
                    }
                }
                Ops::Reshape => vec![(0, g.reshape(&x[0].shape()?))],
                Ops::Expand => vec![(0, unbroadcast(&g, &x[0].shape()?)?)],

                Ops::Reduce => {
                    let Arg::Reduction { op, axes, .. } = &node.arg else {
                        return Err("invalid reduce".into());
                    };
                    let shape = x[0].shape()?;
                    let reduced: Vec<_> = shape
                        .iter()
                        .enumerate()
                        .map(|(i, &n)| if axes.contains(&i) { 1 } else { n })
                        .collect();
                    let gg = g.reshape(&reduced).expand(&shape);
                    if *op == Ops::Add {
                        vec![(0, gg)]
                    } else {
                        let mask = x[0]
                            .eq(&y.reshape(&reduced).expand(&shape))
                            .cast(DType::F32);
                        vec![(
                            0,
                            gg.mul(&mask).div(&mask.sum_axes(axes, true).expand(&shape)),
                        )]
                    }
                }
                Ops::Stage => vec![(0, g)],
                _ => return Err(format!("no gradient rule for {:?}", node.op)),
            };
            for (i, contribution) in contributions {
                let contribution = unbroadcast(&contribution, &x[i].shape()?)?;
                let id = node.src[i].id;
                grads
                    .entry(id)
                    .and_modify(|old| *old = old.add(&contribution))
                    .or_insert(contribution);
            }
        }
        targets
            .iter()
            .map(|target| {
                Ok(grads
                    .get(&target.uop.id)
                    .cloned()
                    .unwrap_or_else(|| Self::constant(0., &target.shape().unwrap_or_default())))
            })
            .collect()
    }
```

`src/tensor.rs`を追記する。

```rust
// file="src/tensor.rs" mode="append"

fn unbroadcast(g: &Tensor, shape: &[usize]) -> Result<Tensor> {
    let gs = g.shape()?;
    if gs == shape {
        return Ok(g.clone());
    }
    if gs.len() < shape.len() {
        return Err("gradient broadcast rank mismatch".into());
    }
    let extra = gs.len() - shape.len();
    let axes: Vec<_> = (0..gs.len())
        .filter(|&i| i < extra || (shape[i - extra] == 1 && gs[i] != 1))
        .collect();
    Ok(g.sum_axes(&axes, true).reshape(shape))
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
        Ops::Exp,
        Ops::Log,
        Ops::Sin,
        Ops::Cos,
        Ops::Sqrt,
        Ops::Exp2,
        Ops::Log2,
        Ops::Max,
        Ops::Lt,
        Ops::Eq,
        Ops::And,
        Ops::Or,
        Ops::Where,
        Ops::Reshape,
        Ops::Permute,
        Ops::Expand,
        Ops::Shrink,
        Ops::Pad,
        Ops::Flip,
        Ops::Stack,
        Ops::Gather,
        Ops::ScatterAdd,
        Ops::Reduce,
        Ops::Detach,
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
    Exp,
    Log,
    Sin,
    Cos,
    Sqrt,
    Exp2,
    Log2,
    Max,
    Lt,
    Eq,
    And,
    Or,
    Where,
    Reshape,
    Permute,
    Expand,
    Shrink,
    Pad,
    Flip,
    Stack,
    Gather,
    ScatterAdd,
    Reduce,
    Detach,
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
            (Ops::Reshape, Arg::Shape(shape)) => {
                if checked_numel(shape)? != checked_numel(&source()?)? {
                    return Err("reshape changes element count".into());
                }
                Ok(shape.clone())
            }
            (Ops::Expand, Arg::Shape(shape)) => {
                let old = source()?;
                if shape.len() != old.len()
                    || old.iter().zip(shape).any(|(&a, &b)| a != b && a != 1)
                {
                    return Err("invalid expand dimensions".into());
                }
                checked_numel(shape)?;
                Ok(shape.clone())
            }
            (Ops::Permute, Arg::Axes(axes)) => {
                let old = source()?;
                let set: HashSet<_> = axes.iter().copied().collect();
                if axes.len() != old.len()
                    || set.len() != old.len()
                    || axes.iter().any(|&a| a >= old.len())
                {
                    return Err("invalid permutation".into());
                }
                Ok(axes.iter().map(|&a| old[a]).collect())
            }
            (Ops::Shrink, Arg::Pairs(pairs)) => {
                let old = source()?;
                if pairs.len() != old.len()
                    || pairs.iter().zip(old).any(|(&(a, b), n)| a > b || b > n)
                {
                    return Err("shrink bounds outside shape".into());
                }
                Ok(pairs.iter().map(|&(a, b)| b - a).collect())
            }
            (Ops::Pad, Arg::Pairs(pairs)) => {
                let old = source()?;
                if pairs.len() != old.len() {
                    return Err("pad rank mismatch".into());
                }
                old.iter()
                    .zip(pairs)
                    .map(|(&n, &(a, b))| {
                        n.checked_add(a)
                            .and_then(|x| x.checked_add(b))
                            .ok_or_else(|| "pad overflow".into())
                    })
                    .collect()
            }
            (Ops::Flip, Arg::Axes(axes)) => {
                let s = source()?;
                if axes.iter().any(|&a| a >= s.len()) {
                    return Err("flip axis outside rank".into());
                }
                Ok(s)
            }
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
            (Ops::Stack, Arg::Axis(axis)) => {
                let old = source()?;
                if *axis > old.len()
                    || self
                        .src
                        .iter()
                        .any(|x| x.shape().ok().as_ref() != Some(&old))
                {
                    return Err("stack shapes differ".into());
                }
                let mut s = old;
                s.insert(*axis, self.src.len());
                Ok(s)
            }
            (Ops::Gather, Arg::Axis(axis)) => {
                let s = source()?;
                let idx = self.src.get(1).ok_or_else(err)?.shape()?;
                if *axis >= s.len() {
                    return Err("gather axis outside rank".into());
                }
                let mut out = s[..*axis].to_vec();
                out.extend(idx);
                out.extend_from_slice(&s[axis + 1..]);
                Ok(out)
            }
            (Ops::ScatterAdd, Arg::Axis(axis)) => {
                let s = source()?;
                let idx = self.src.get(1).ok_or_else(err)?.shape()?;
                let values = self.src.get(2).ok_or_else(err)?.shape()?;
                if *axis >= s.len() {
                    return Err("scatter axis outside rank".into());
                }
                let mut expected = s[..*axis].to_vec();
                expected.extend(idx);
                expected.extend_from_slice(&s[axis + 1..]);
                if values != expected {
                    return Err("scatter values shape mismatch".into());
                }
                Ok(s)
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
            (op, _) if is_unary(*op) || matches!(op, Ops::Detach | Ops::Stage | Ops::After) => {
                source()
            }
            _ => Err(err()),
        }
    }
```

`tests/tensor_native.rs` の `use chibigrad::`を置き換える。

```rust
// file="tests/tensor_native.rs" mode="replace" item="use chibigrad::"
use chibigrad::{Buffer, Ops, Tensor, compile_many};
```

`tests/tensor_native.rs`を追記する。

```rust
// file="tests/tensor_native.rs" mode="append"

#[test]
fn native_gradients_and_multiple_outputs() {
    let x = Tensor::from_slice(&[1., 2., 3., 4.], &[2, 2]);
    let w = Tensor::from_slice(&[2., 3.], &[2]);
    let loss = x.mul(&w).mul(&x).sum();
    let grads = loss.gradients(&[x.clone(), w]).unwrap();
    let exe = compile_many(&[loss, grads[0].clone(), grads[1].clone()]).unwrap();
    let out = exe.run(&[], &HashMap::new()).unwrap();
    close(&out[0].data, &[80.]);
    close(&out[1].data, &[4., 12., 12., 24.]);
    close(&out[2].data, &[10., 20.]);
}
```

```sh
# check step=13.01
cargo test --offline
```

損失 80、入力勾配 [4,12,12,24]、重み勾配 [10,20] を複数出力の一回の計画で得る。

## 13.02 移動・選択・集約の勾配を戻す

転置には逆置換、切出しには padding、gather には scatter-add を使う。最大値の同値な入力には寄与を分配し、detach で逆伝播を止める。

`src/tensor.rs` の `impl Tensor` 内 の `pub fn gradients`を置き換える。

```rust
// file="src/tensor.rs" mode="replace" within="impl Tensor" item="pub fn gradients"
    pub fn gradients(&self, targets: &[Self]) -> Result<Vec<Self>> {
        self.shape()?;
        for target in targets {
            target.shape()?;
        }
        let nodes = self.uop.toposort();
        let mut grads = HashMap::from([(self.uop.id, Self::constant(1., &self.shape()?))]);
        for node in nodes.into_iter().rev() {
            let Some(g) = grads.get(&node.id).cloned() else {
                continue;
            };
            let x: Vec<Self> = node.src.iter().cloned().map(Self::from_uop).collect();
            let y = Self::from_uop(node.clone());
            let contributions: Vec<(usize, Self)> = match node.op {
                Ops::Buffer | Ops::Const | Ops::Detach => vec![],
                Ops::Add => vec![(0, g.clone()), (1, g)],
                Ops::Sub => vec![(0, g.clone()), (1, g.neg())],
                Ops::Mul => vec![(0, g.mul(&x[1])), (1, g.mul(&x[0]))],
                Ops::Div => {
                    vec![
                        (0, g.div(&x[1])),
                        (1, g.mul(&x[0]).div(&x[1].mul(&x[1])).neg()),
                    ]
                }
                Ops::Neg => vec![(0, g.neg())],
                Ops::Exp => vec![(0, g.mul(&y))],
                Ops::Log => vec![(0, g.div(&x[0]))],
                Ops::Exp2 => {
                    vec![(0, g.mul(&y).mul(&Self::scalar(std::f32::consts::LN_2)))]
                }
                Ops::Log2 => {
                    vec![(0, g.div(&x[0].mul(&Self::scalar(std::f32::consts::LN_2))))]
                }
                Ops::Sin => vec![(0, g.mul(&x[0].cos()))],
                Ops::Cos => vec![(0, g.mul(&x[0].sin()).neg())],
                Ops::Sqrt => vec![(0, g.div(&y.mul(&Self::scalar(2.))))],
                Ops::Max => {
                    let left = x[0].eq(&y).cast(DType::F32);
                    let right = x[1].eq(&y).cast(DType::F32);
                    let den = left.add(&right);
                    vec![(0, g.mul(&left).div(&den)), (1, g.mul(&right).div(&den))]
                }
                Ops::Where => {
                    vec![
                        (1, x[0].where_(&g, &Self::scalar(0.))),
                        (2, x[0].where_(&Self::scalar(0.), &g)),
                    ]
                }
                Ops::Lt | Ops::Eq | Ops::And | Ops::Or => vec![],
                Ops::Cast => {
                    if matches!(x[0].uop.dtype(), DType::F32 | DType::Ptr)
                        && node.dtype() == DType::F32
                    {
                        vec![(0, g)]
                    } else {
                        vec![]
                    }
                }
                Ops::Reshape => vec![(0, g.reshape(&x[0].shape()?))],
                Ops::Expand => vec![(0, unbroadcast(&g, &x[0].shape()?)?)],
                Ops::Permute => {
                    let Arg::Axes(axes) = &node.arg else {
                        return Err("invalid permutation".into());
                    };
                    let mut inverse = vec![0; axes.len()];
                    for (i, &a) in axes.iter().enumerate() {
                        inverse[a] = i;
                    }
                    vec![(0, g.permute(&inverse))]
                }
                Ops::Shrink => {
                    let Arg::Pairs(pairs) = &node.arg else {
                        return Err("invalid shrink".into());
                    };
                    let shape = x[0].shape()?;
                    vec![(
                        0,
                        g.pad(
                            &pairs
                                .iter()
                                .zip(shape)
                                .map(|(&(a, b), n)| (a, n - b))
                                .collect::<Vec<_>>(),
                        ),
                    )]
                }
                Ops::Pad => {
                    let Arg::Pairs(pairs) = &node.arg else {
                        return Err("invalid pad".into());
                    };
                    let shape = x[0].shape()?;
                    vec![(
                        0,
                        g.shrink(
                            &pairs
                                .iter()
                                .zip(shape)
                                .map(|(&(a, _), n)| (a, a + n))
                                .collect::<Vec<_>>(),
                        ),
                    )]
                }
                Ops::Flip => {
                    let Arg::Axes(axes) = &node.arg else {
                        return Err("invalid flip".into());
                    };
                    vec![(0, g.flip(axes))]
                }
                Ops::Stack => {
                    let Arg::Axis(axis) = node.arg else {
                        return Err("invalid stack".into());
                    };
                    let shape = g.shape()?;
                    x.iter()
                        .enumerate()
                        .map(|(i, v)| {
                            let mut bounds = shape.iter().map(|&n| (0, n)).collect::<Vec<_>>();
                            bounds[axis] = (i, i + 1);
                            Ok((i, g.shrink(&bounds).reshape(&v.shape()?)))
                        })
                        .collect::<Result<_>>()?
                }
                Ops::Gather => {
                    let Arg::Axis(axis) = node.arg else {
                        return Err("invalid gather".into());
                    };
                    vec![(
                        0,
                        Self::constant(0., &x[0].shape()?).scatter_add(&x[1], &g, axis),
                    )]
                }
                Ops::ScatterAdd => {
                    let Arg::Axis(axis) = node.arg else {
                        return Err("invalid scatter".into());
                    };
                    vec![(0, g.clone()), (2, g.gather(&x[1], axis))]
                }
                Ops::Reduce => {
                    let Arg::Reduction { op, axes, .. } = &node.arg else {
                        return Err("invalid reduce".into());
                    };
                    let shape = x[0].shape()?;
                    let reduced: Vec<_> = shape
                        .iter()
                        .enumerate()
                        .map(|(i, &n)| if axes.contains(&i) { 1 } else { n })
                        .collect();
                    let gg = g.reshape(&reduced).expand(&shape);
                    if *op == Ops::Add {
                        vec![(0, gg)]
                    } else {
                        let mask = x[0]
                            .eq(&y.reshape(&reduced).expand(&shape))
                            .cast(DType::F32);
                        vec![(
                            0,
                            gg.mul(&mask).div(&mask.sum_axes(axes, true).expand(&shape)),
                        )]
                    }
                }
                Ops::Stage => vec![(0, g)],
                _ => return Err(format!("no gradient rule for {:?}", node.op)),
            };
            for (i, contribution) in contributions {
                let contribution = unbroadcast(&contribution, &x[i].shape()?)?;
                let id = node.src[i].id;
                grads
                    .entry(id)
                    .and_modify(|old| *old = old.add(&contribution))
                    .or_insert(contribution);
            }
        }
        targets
            .iter()
            .map(|target| {
                Ok(grads
                    .get(&target.uop.id)
                    .cloned()
                    .unwrap_or_else(|| Self::constant(0., &target.shape().unwrap_or_default())))
            })
            .collect()
    }
```

`tests/autodiff.rs`を保存する。

```rust
// file="tests/autodiff.rs" mode="write"
use chibigrad::{Tensor, compile_many};
use std::collections::HashMap;
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() < 1e-4, "{a} != {b}");
    }
}
#[test]
fn native_gradients_and_multiple_outputs() {
    let x = Tensor::from_slice(&[1., 2., 3., 4.], &[2, 2]);
    let w = Tensor::from_slice(&[2., 3.], &[2]);
    let loss = x.mul(&w).mul(&x).sum();
    let grads = loss.gradients(&[x.clone(), w]).unwrap();
    let exe = compile_many(&[loss, grads[0].clone(), grads[1].clone()]).unwrap();
    let out = exe.run(&[], &HashMap::new()).unwrap();
    close(&out[0].data, &[80.]);
    close(&out[1].data, &[4., 12., 12., 24.]);
    close(&out[2].data, &[10., 20.]);
}
```

`tests/tensor_native.rs`を追記する。

```rust
// file="tests/tensor_native.rs" mode="append"

#[test]
fn gather_scatter_stage() {
    let x = Tensor::from_slice(&[1., 2., 3., 4., 5., 6.], &[3, 2]);
    let i = Tensor::from_slice(&[2., 0., 2.], &[3]);
    let y = x.gather(&i, 0).stage();
    close(&y.to_vec().unwrap(), &[5., 6., 1., 2., 5., 6.]);
    close(
        &y.sum().gradient(&x).unwrap().to_vec().unwrap(),
        &[1., 1., 0., 0., 2., 2.],
    );
}

#[test]
fn convolution_gradient() {
    let x = Tensor::from_slice(&[1., 2., 3., 4., 5., 6., 7., 8., 9.], &[1, 1, 3, 3]);
    let w = Tensor::from_slice(&[1., 0., 0., -1.], &[1, 1, 2, 2]);
    let y = x.conv2d(&w, (1, 1), (0, 0));
    close(&y.to_vec().unwrap(), &[-4.; 4]);
    close(
        &y.sum().gradient(&w).unwrap().to_vec().unwrap(),
        &[12., 16., 24., 28.],
    );
}
```

```sh
# check step=13.02
cargo test --offline
```

gather の重複添字の勾配が [1,1,0,0,2,2]、畳込みフィルタの勾配が [12,16,24,28] になる。

## 章末の累積確認

前の章までのテストとexampleも再ビルドします。

```sh
# check
cargo test --quiet --all-targets
```

## 要件との対応

### 13-001 純粋な graph と target を受け取る gradient API を追加する。

<!-- todo:13-001 -->

関連する実装と確認は本章の `13.01`, `13.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 13-002 scalar seed と逆 topological walk を実装する。

<!-- todo:13-002 -->

関連する実装と確認は本章の `13.01`, `13.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 13-003 ADD の各入力に seed、MUL の各入力に他方との積を渡す。

<!-- todo:13-003 -->

関連する実装と確認は本章の `13.01`, `13.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 13-004 SUB の入力へ `(g,-g)`、NEG の入力へ `-g` を渡す。

<!-- todo:13-004 -->

関連する実装と確認は本章の `13.01`, `13.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 13-005 RECIP の入力へ `-g/(x*x)` を渡し、浮動小数点の除算を MUL と RECIP の合成にそろえる。

<!-- todo:13-005 -->

関連する実装と確認は本章の `13.01`, `13.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 13-006 勾配の graph を通常の compile と Plan に渡す。

<!-- todo:13-006 -->

関連する実装と確認は本章の `13.01`, `13.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 13-007 複数経路からの勾配を同じ表へ ADD する。

<!-- todo:13-007 -->

関連する実装と確認は本章の `13.01`, `13.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 13-008 同じ shared node を一度だけ逆向きに処理する。

<!-- todo:13-008 -->

関連する実装と確認は本章の `13.01`, `13.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 13-009 DETACH の前向き値を維持し、入力への寄与を止める。

<!-- todo:13-009 -->

関連する実装と確認は本章の `13.01`, `13.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 13-010 非 scalar loss の seed が未指定、または shape 不一致なら拒否する。

<!-- todo:13-010 -->

関連する実装と確認は本章の `13.01`, `13.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 13-011 到達しない target と未対応演算のエラーを区別する。

<!-- todo:13-011 -->

関連する実装と確認は本章の `13.01`, `13.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 13-012 SUM の勾配を元の軸へ broadcast する。

<!-- todo:13-012 -->

関連する実装と確認は本章の `13.01`, `13.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 13-013 各 src への寄与を unbroadcast してから蓄積する。

<!-- todo:13-013 -->

関連する実装と確認は本章の `13.01`, `13.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 13-014 RESHAPE を入力 shape、PERMUTE を逆 permutation に戻す。

<!-- todo:13-014 -->

関連する実装と確認は本章の `13.01`, `13.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 13-015 SHRINK の逆を zero PAD、PAD の逆を SHRINK とする。

<!-- todo:13-015 -->

関連する実装と確認は本章の `13.01`, `13.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 13-016 FLIP の逆を同じ FLIP、STACK の逆を各入力への slice とする。

<!-- todo:13-016 -->

関連する実装と確認は本章の `13.01`, `13.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 13-017 RECIP を f32 の純粋演算として通し、微分を `-g*recip(x)*recip(x)` にする。

<!-- todo:13-017 -->

関連する実装と確認は本章の `13.01`, `13.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 13-018 SQRT の微分を `g/(2*sqrt(x))` として作る。

<!-- todo:13-018 -->

関連する実装と確認は本章の `13.01`, `13.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 13-019 SIN の微分を `g*sin(π/2-x)` として作る。

<!-- todo:13-019 -->

関連する実装と確認は本章の `13.01`, `13.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 13-020 LOG2 の微分を `g/(x*ln(2))`、EXP2 を `g*exp2(x)*ln(2)` にする。

<!-- todo:13-020 -->

関連する実装と確認は本章の `13.01`, `13.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 13-021 近似展開より前に微分し、数学関数単位の規則を適用する。

<!-- todo:13-021 -->

関連する実装と確認は本章の `13.01`, `13.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 13-022 テスト側に中央差分 `(f(x+h)-f(x-h))/(2*h)` を追加する。

<!-- todo:13-022 -->

関連する実装と確認は本章の `13.01`, `13.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 13-023 `h=1e-3` と `h=5e-4` の両方で極端に不安定な点を除いて比較する。

<!-- todo:13-023 -->

関連する実装と確認は本章の `13.01`, `13.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 13-024 scalar、共有枝、broadcast、movement、matmul の重みを小例で検査する。

<!-- todo:13-024 -->

関連する実装と確認は本章の `13.01`, `13.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 13-025 libm モードを基準にし、近似モードの誤差を微分器の誤りと分ける。

<!-- todo:13-025 -->

関連する実装と確認は本章の `13.01`, `13.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 13-026 勾配の出力 shape が target と一致し、共有枝の寄与が欠落しない。

<!-- todo:13-026 -->

関連する実装と確認は本章の `13.01`, `13.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 13-027 backward 用の独立した数値実行器を作らず、生成 C を実行する。

<!-- todo:13-027 -->

関連する実装と確認は本章の `13.01`, `13.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 13-028 graph dump に元の前向きと新しい勾配の両方が残る。

<!-- todo:13-028 -->

関連する実装と確認は本章の `13.01`, `13.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 13-029 DETACH と gradient の要求が backend 入力に残らない。

<!-- todo:13-029 -->

関連する実装と確認は本章の `13.01`, `13.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。
