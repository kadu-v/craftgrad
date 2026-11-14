# 10. movement・broadcast・gather・scatter を添字へ変換する

前の節で動かしたコードに、今回使う演算と処理を追加します。各節の確認を実行してから次へ進みます。

参照するUOpの構成は [tinygrad c22edff](https://github.com/tinygrad/tinygrad/blob/c22edff14c79310a9b2df62a78819cf77c8289fa/tinygrad/uop/ops.py) です。最初から全機能を実装せず、使う節で増やします。

## 10.01 移動演算を添字変換として下げる

reshape、expand、permute、shrink、pad、flip、stack を加える。中間の全データを移す代わりに、出力座標から入力座標と有効条件を作る。

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

`src/tensor.rs` の `impl Tensor` 内を追記する。

```rust
// file="src/tensor.rs" mode="append" within="impl Tensor"
    pub fn constant(value: f32, shape: &[usize]) -> Self {
        Self::scalar(value)
            .reshape(&vec![1; shape.len()])
            .expand(shape)
    }

    pub fn reshape(&self, shape: &[usize]) -> Self {
        Self::from_uop(UOp::new(
            Ops::Reshape,
            vec![self.uop.clone()],
            Arg::Shape(shape.to_vec()),
        ))
    }

    pub fn permute(&self, axes: &[usize]) -> Self {
        Self::from_uop(UOp::new(
            Ops::Permute,
            vec![self.uop.clone()],
            Arg::Axes(axes.to_vec()),
        ))
    }

    pub fn expand(&self, shape: &[usize]) -> Self {
        Self::from_uop(UOp::new(
            Ops::Expand,
            vec![self.uop.clone()],
            Arg::Shape(shape.to_vec()),
        ))
    }

    pub fn shrink(&self, bounds: &[(usize, usize)]) -> Self {
        Self::from_uop(UOp::new(
            Ops::Shrink,
            vec![self.uop.clone()],
            Arg::Pairs(bounds.to_vec()),
        ))
    }

    pub fn pad(&self, pads: &[(usize, usize)]) -> Self {
        Self::from_uop(UOp::new(
            Ops::Pad,
            vec![self.uop.clone()],
            Arg::Pairs(pads.to_vec()),
        ))
    }

    pub fn flip(&self, axes: &[usize]) -> Self {
        Self::from_uop(UOp::new(
            Ops::Flip,
            vec![self.uop.clone()],
            Arg::Axes(axes.to_vec()),
        ))
    }

    pub fn stack(values: &[Self], axis: usize) -> Self {
        Self::from_uop(UOp::new(
            Ops::Stack,
            values.iter().map(|x| x.uop.clone()).collect(),
            Arg::Axis(axis),
        ))
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
        Ops::Reshape,
        Ops::Permute,
        Ops::Expand,
        Ops::Shrink,
        Ops::Pad,
        Ops::Flip,
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

`src/uir.rs` の `fn encode_arg`を置き換える。

```rust
// file="src/uir.rs" mode="replace" item="fn encode_arg"
fn encode_arg(a: &Arg) -> String {
    match a {
        Arg::None => "none".into(),
        Arg::F32(v) => format!("f32;{v:08x}"),
        Arg::I32(v) => format!("i32;{v}"),
        Arg::Int(v) => format!("i64;{v}"),
        Arg::Bool(v) => format!("bool;{}", *v as u8),
        Arg::Shape(v) => format!("shape;{}", numbers(v)),
        Arg::Axes(v) => format!("axes;{}", numbers(v)),
        Arg::Pairs(v) => {
            format!(
                "pairs;{}",
                numbers(&v.iter().flat_map(|(a, b)| [a, b]).collect::<Vec<_>>())
            )
        }
        Arg::Axis(v) => format!("axis;{v}"),
        Arg::Reduction { op, axes, keepdim } => {
            format!("reduction;{op:?};{};{}", numbers(axes), *keepdim as u8)
        }
        Arg::Buffer { name, shape, data } => {
            format!(
                "buffer;{};{};{}",
                hex(name.as_bytes()),
                numbers(shape),
                data.as_ref().map(|v| numbers(v)).unwrap_or("-".into())
            )
        }
        Arg::Range { id, axis } => format!("range;{id};{axis:?}"),
        Arg::Allocation { id, dtype, size } => {
            format!("allocation;{id};{dtype:?};{size}")
        }
        Arg::Id(v) => format!("id;{v}"),
        Arg::DType(dt) => format!("dtype;{dt:?}"),
        Arg::Text(s) => format!("text;{}", hex(s.as_bytes())),
        Arg::Bytes(v) => format!("bytes;{}", hex(v)),
        Arg::Slot { index, shape } => format!("slot;{index};{}", numbers(shape)),
        Arg::ScalarParam { index, dtype } => format!("scalar;{index};{dtype:?}"),
    }
}
```

`src/uir.rs` の `fn decode_arg`を置き換える。

```rust
// file="src/uir.rs" mode="replace" item="fn decode_arg"
fn decode_arg(s: &str) -> Result<Arg> {
    let fields: Vec<_> = s.split(';').collect();
    Ok(match fields.as_slice() {
        ["none"] => Arg::None,
        ["f32", v] if v.len() == 8 => {
            Arg::F32(u32::from_str_radix(v, 16).map_err(|_| "invalid f32 bits")?)
        }
        ["i32", v] => Arg::I32(number(v)?),
        ["i64", v] => Arg::Int(number(v)?),
        ["bool", v] => Arg::Bool(boolean(v)?),
        ["shape", v] => Arg::Shape(list(v)?),
        ["axes", v] => Arg::Axes(list(v)?),
        ["pairs", v] => {
            let v: Vec<usize> = list(v)?;
            if v.len() % 2 != 0 {
                return Err("odd pairs list".into());
            }
            Arg::Pairs(v.chunks_exact(2).map(|p| (p[0], p[1])).collect())
        }
        ["axis", v] => Arg::Axis(number(v)?),
        ["reduction", o, v, k] => Arg::Reduction {
            op: op(o)?,
            axes: list(v)?,
            keepdim: boolean(k)?,
        },
        ["buffer", n, v, d] => Arg::Buffer {
            name: string(n)?,
            shape: list(v)?,
            data: if *d == "-" { None } else { Some(list(d)?) },
        },
        ["range", id, a] => Arg::Range {
            id: number(id)?,
            axis: match *a {
                "Loop" => Axis::Loop,
                "Reduce" => Axis::Reduce,
                _ => return Err("unknown axis".into()),
            },
        },
        ["allocation", id, dt, size] => Arg::Allocation {
            id: number(id)?,
            dtype: dtype(dt)?,
            size: number(size)?,
        },
        ["id", v] => Arg::Id(number(v)?),
        ["dtype", dt] => Arg::DType(dtype(dt)?),
        ["text", v] => Arg::Text(string(v)?),
        ["bytes", v] => Arg::Bytes(unhex(v)?),
        ["slot", i, v] => Arg::Slot {
            index: number(i)?,
            shape: list(v)?,
        },
        ["scalar", i, dt] => Arg::ScalarParam {
            index: number(i)?,
            dtype: dtype(dt)?,
        },
        _ => return Err("invalid typed argument".into()),
    })
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
    Reshape,
    Permute,
    Expand,
    Shrink,
    Pad,
    Flip,
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

`src/uop.rs` の `pub enum Arg`を置き換える。

```rust
// file="src/uop.rs" mode="replace" item="pub enum Arg"
pub enum Arg {
    None,
    F32(u32),
    I32(i32),
    Int(i64),
    Bool(bool),
    Shape(Vec<usize>),
    Axes(Vec<usize>),
    Pairs(Vec<(usize, usize)>),
    Axis(usize),
    Reduction {
        op: Ops,
        axes: Vec<usize>,
        keepdim: bool,
    },
    Buffer {
        name: String,
        shape: Vec<usize>,
        data: Option<Vec<u32>>,
    },
    Range {
        id: usize,
        axis: Axis,
    },
    Allocation {
        id: usize,
        dtype: DType,
        size: usize,
    },
    Id(usize),
    DType(DType),
    Text(String),
    Bytes(Vec<u8>),
    Slot {
        index: usize,
        shape: Vec<usize>,
    },
    ScalarParam {
        index: usize,
        dtype: DType,
    },
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

`tests/movement.rs`を保存する。

```rust
// file="tests/movement.rs" mode="write"
use chibigrad::Tensor;
#[test]
fn index_maps_preserve_values() {
    let x = Tensor::from_slice(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    assert_eq!(
        x.permute(&[1, 0]).to_vec().unwrap(),
        [1., 4., 2., 5., 3., 6.]
    );
    assert_eq!(
        x.pad(&[(1, 0), (1, 1)])
            .flip(&[0])
            .shrink(&[(0, 2), (1, 4)])
            .to_vec()
            .unwrap(),
        [4., 5., 6., 1., 2., 3.]
    );
    assert!(x.reshape(&[7]).compile().is_err());
    assert!(x.permute(&[1]).compile().is_err());
    let a = Tensor::from_slice(&[2., 3.], &[2, 1]).expand(&[2, 3]);
    assert_eq!(a.to_vec().unwrap(), [2., 2., 2., 3., 3., 3.]);
}
```

`tests/tensor_native.rs` の `fn fused_reduction_rebinds_native_inputs`を置き換える。

```rust
// file="tests/tensor_native.rs" mode="replace" item="fn fused_reduction_rebinds_native_inputs"
fn fused_reduction_rebinds_native_inputs() {
    let x = Tensor::parameter("x", &[4]);
    let w = Tensor::parameter("w", &[4]);
    let y = x.mul(&w).add(&Tensor::scalar(1.)).sum();
    let exe = y.compile().unwrap();
    assert_eq!(exe.kernel_count(), 1);
    for p in exe.programs() {
        assert!(p.src.iter().any(|s| s.op == Ops::Binary));
        assert!(
            !p.src[1]
                .src
                .iter()
                .any(|u| matches!(u.op, Ops::Reduce | Ops::Reshape))
        );
    }
    let run = |xs: &[f32]| {
        let bufs = exe
            .inputs()
            .iter()
            .map(|i| {
                Buffer::new(
                    i.shape.clone(),
                    if i.name == "x" {
                        xs.to_vec()
                    } else {
                        vec![4.; 4]
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        exe.run(&bufs, &HashMap::new()).unwrap()[0].data.clone()
    };
    assert_eq!(run(&[1., 2., 3., 4.]), [44.]);
    assert_eq!(run(&[2., 2., 2., 2.]), [36.]);
}
```

```sh
# check step=10.01
cargo test --offline
```

転置・切出し・反転・パディング・展開の値が一致し、不正な reshape と permutation を拒否する。

## 10.02 実行時の gather と scatter-add

添字テンソルを読み、有限・整数・範囲内の三条件を確かめる。エラーバッファで失敗を呼出し元へ返す。scatter-add は重複した添字への寄与を蓄積する。

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

`src/codegen.rs` の `impl Lower` 内を追記する。

```rust
// file="src/codegen.rs" mode="append" within="impl Lower"
    fn checked_index(&mut self, raw: UOp, extent: UOp, active: &UOp) -> UOp {
        let finite = UOp::f32(-1.)
            .lt(&raw)
            .and(&raw.lt(&extent.cast(DType::F32)));
        let index = finite.where_(&raw, &UOp::f32(0.)).cast(DType::I64);
        let valid = finite.and(&index.cast(DType::F32).eq(&raw));
        let error = UOp::new(
            Ops::Buffer,
            vec![],
            Arg::Buffer {
                name: "__chibi_error".into(),
                shape: vec![1],
                data: Some(vec![0]),
            },
        );
        self.emit(
            error
                .index(UOp::int(0), active.and(&valid.eq(&UOp::bool(false))))
                .store(UOp::f32(1.)),
        );
        self.emit(index)
    }
```

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
            && (is_binary(u.op)
                || is_unary(u.op)
                || matches!(u.op, Ops::Reduce | Ops::Gather | Ops::ScatterAdd));
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

`src/tensor.rs` の `impl Tensor` 内を追記する。

```rust
// file="src/tensor.rs" mode="append" within="impl Tensor"
    pub fn gather(&self, indices: &Self, axis: usize) -> Self {
        Self::from_uop(UOp::new(
            Ops::Gather,
            vec![self.uop.clone(), indices.uop.clone()],
            Arg::Axis(axis),
        ))
    }

    pub fn scatter_add(&self, indices: &Self, values: &Self, axis: usize) -> Self {
        Self::from_uop(UOp::new(
            Ops::ScatterAdd,
            vec![self.uop.clone(), indices.uop.clone(), values.uop.clone()],
            Arg::Axis(axis),
        ))
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

`src/uop.rs` の `impl UOp` 内 の `fn infer_dtype`を置き換える。

```rust
// file="src/uop.rs" mode="replace" within="impl UOp" item="fn infer_dtype"
    fn infer_dtype(&self) -> DType {
        match self.op {
            Ops::Const => match self.arg {
                Arg::F32(_) => DType::F32,
                Arg::I32(_) => DType::I32,
                Arg::Int(_) => DType::I64,
                Arg::Bool(_) => DType::Bool,
                _ => DType::Void,
            },
            Ops::Param => match self.arg {
                Arg::ScalarParam { dtype, .. } => dtype,
                _ => DType::Ptr,
            },
            Ops::Buffer | Ops::Alloc => DType::Ptr,
            Ops::Index => {
                if self.src.first().is_some_and(|s| s.op == Ops::Call) {
                    self.src[0].dtype()
                } else {
                    DType::Ptr
                }
            }
            Ops::Range => DType::I64,
            Ops::Lt | Ops::Eq => DType::Bool,
            Ops::Cast | Ops::Bitcast => {
                if let Arg::DType(dt) = self.arg {
                    dt
                } else {
                    DType::Void
                }
            }
            Ops::Load => match self.arg {
                Arg::DType(dt) => dt,
                _ => DType::F32,
            },
            Ops::Gather | Ops::ScatterAdd => DType::F32,
            Ops::Call => self.src.first().map_or(DType::Void, |x| x.dtype()),
            Ops::Where => self.src.get(1).map_or(DType::Void, |x| x.dtype()),
            Ops::Store
            | Ops::End
            | Ops::Sink
            | Ops::Linear
            | Ops::Program
            | Ops::Source
            | Ops::Binary
            | Ops::Invalid => DType::Void,
            _ => self.src.first().map_or(DType::Void, |x| {
                if x.dtype() == DType::Ptr {
                    DType::F32
                } else {
                    x.dtype()
                }
            }),
        }
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
            (op, _) if is_unary(*op) || matches!(op, Ops::Stage | Ops::After) => source(),
            _ => Err(err()),
        }
    }
```

`tests/tensor_native.rs`を追記する。

```rust
// file="tests/tensor_native.rs" mode="append"
#[test]
fn runtime_gather_rejects_invalid_indices() {
    let x = Tensor::from_slice(&[3., 4., 5.], &[3]);
    let i = Tensor::parameter("i", &[1]);
    let exe = x.gather(&i, 0).compile().unwrap();
    for bad in [-1., 3., 0.5, f32::NAN, f32::INFINITY] {
        assert!(
            exe.run(&[Buffer::new(vec![1], vec![bad]).unwrap()], &HashMap::new())
                .is_err(),
            "accepted {bad}"
        );
    }
    assert_eq!(
        exe.run(&[Buffer::new(vec![1], vec![2.]).unwrap()], &HashMap::new())
            .unwrap()[0]
            .data,
        [5.]
    );
}
```

```sh
# check step=10.02
cargo test --offline
```

-1、範囲外、0.5、NaN、無限大の gather 添字を拒否し、有効な添字 2 は値 5 を返す。

## 章末の累積確認

前の章までのテストとexampleも再ビルドします。

```sh
# check
cargo test --quiet --all-targets
```

## 要件との対応

### 10-001 出力の二次元 RANGE と row-major の平坦化を追加する。

<!-- todo:10-001 -->

関連する実装と確認は本章の `10.01`, `10.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 10-002 shape を右揃えし、入力側の長さ 1 の軸を座標 0 にする。

<!-- todo:10-002 -->

関連する実装と確認は本章の `10.01`, `10.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 10-003 `[2,3] + [3]` と `[2,1] + [1,3]` を CPU で実行する。

<!-- todo:10-003 -->

関連する実装と確認は本章の `10.01`, `10.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 10-004 `[2,3] + [2]` は broadcast 不成立として codegen 前に拒否する。

<!-- todo:10-004 -->

関連する実装と確認は本章の `10.01`, `10.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 10-005 要素数が等しい reshape だけを受け付ける。

<!-- todo:10-005 -->

関連する実装と確認は本章の `10.01`, `10.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 10-006 permute の軸が重複せず、全軸を一度ずつ含むことを検証する。

<!-- todo:10-006 -->

関連する実装と確認は本章の `10.01`, `10.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 10-007 論理的な平坦化と座標の復元を rangeify に追加する。

<!-- todo:10-007 -->

関連する実装と確認は本章の `10.01`, `10.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 10-008 転置後の reshape を CPU 実行し、連続化した場合とも比較する。

<!-- todo:10-008 -->

関連する実装と確認は本章の `10.01`, `10.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 10-009 shrink の半開区間 `start..end` を入力側の offset に変換する。

<!-- todo:10-009 -->

関連する実装と確認は本章の `10.01`, `10.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 10-010 `0 <= start <= end <= input_size` を各軸で検証する。

<!-- todo:10-010 -->

関連する実装と確認は本章の `10.01`, `10.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 10-011 pad の有効条件と、条件成立時だけ実行する LOAD を生成する。

<!-- todo:10-011 -->

関連する実装と確認は本章の `10.01`, `10.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 10-012 pad の外側はゼロを返し、無効なポインタへの LOAD を作らない。

<!-- todo:10-012 -->

関連する実装と確認は本章の `10.01`, `10.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 10-013 非対称 pad と shrink の合成を interpreter と生成 C で照合する。

<!-- todo:10-013 -->

関連する実装と確認は本章の `10.01`, `10.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 10-014 flip の座標を `size-1-coordinate` にする。

<!-- todo:10-014 -->

関連する実装と確認は本章の `10.01`, `10.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 10-015 flip の軸の範囲と重複を検証する。

<!-- todo:10-015 -->

関連する実装と確認は本章の `10.01`, `10.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 10-016 stack の先頭座標で入力を選び、残りの座標を各入力に渡す。

<!-- todo:10-016 -->

関連する実装と確認は本章の `10.01`, `10.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 10-017 入力選択の分岐内だけで該当入力を LOAD する。

<!-- todo:10-017 -->

関連する実装と確認は本章の `10.01`, `10.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 10-018 shape が異なる stack と、未知の軸指定を拒否する。

<!-- todo:10-018 -->

関連する実装と確認は本章の `10.01`, `10.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 10-019 整数で表せる上記の期待値は `f32` でも厳密に一致する。

<!-- todo:10-019 -->

関連する実装と確認は本章の `10.01`, `10.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 10-020 kernel IR に抽象 movement が残らず、添字式と条件付き LOAD になる。

<!-- todo:10-020 -->

関連する実装と確認は本章の `10.01`, `10.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 10-021 元グラフの movement ノードは実行後にも残る。

<!-- todo:10-021 -->

関連する実装と確認は本章の `10.01`, `10.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 10-022 `tests/movement.rs` に不正 shape、軸重複、負の区間の拒否例を追加する。

<!-- todo:10-022 -->

関連する実装と確認は本章の `10.01`, `10.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 10-023 同じ NodeId を異なる座標で読む例が、順序と値の両方で正しい。

<!-- todo:10-023 -->

関連する実装と確認は本章の `10.01`, `10.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 10-024 空の reshape、permute、shrink はメモリアクセスとゼロ除算を起こさない。

<!-- todo:10-024 -->

関連する実装と確認は本章の `10.01`, `10.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。
