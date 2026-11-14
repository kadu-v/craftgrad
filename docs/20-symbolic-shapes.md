# 20. symbolic shape を束縛し generic と特化版を再利用する

前の節で動かしたコードに、今回使う演算と処理を追加します。各節の確認を実行してから次へ進みます。

参照するUOpの構成は [tinygrad c22edff](https://github.com/tinygrad/tinygrad/blob/c22edff14c79310a9b2df62a78819cf77c8289fa/tinygrad/uop/ops.py) です。最初から全機能を実装せず、使う節で増やします。

## 20.01 記号付きスカラーと範囲を扱う

Variable は名前と最小・最大値を持ち、Bind はその範囲内の値を結ぶ。演算ごとの区間推定をキャッシュし、証明できる比較や整数の恒等式をパターンで簡約する。STACK の選択は I64 の範囲を証明してから WHERE に展開する。CPU の引数領域と GPU のスカラーバッファへ変数を渡し、既存のワーカー制御バッファとは分けて再束縛する。この節ではテンソルの shape はまだ固定である。

`src/codegen.rs`を保存する。

```rust
// file="src/codegen.rs" mode="write"
use crate::{
    Result,
    pattern::{PatternMatcher, UPat, graph_rewrite},
    symbolic::simplify,
    uop::{Arg, Axis, DType, Ops, UOp, is_binary, is_unary, numel},
};
use std::collections::{HashMap, HashSet};
pub fn normalize(root: &UOp) -> Result<UOp> {
    let root = crate::functions::inline_calls(root)?;
    let pm = PatternMatcher::new()
        .add(UPat::op(Ops::Exp).src(vec![UPat::var("x")]), |c| {
            Some(
                c["x"]
                    .mul(&UOp::f32(std::f32::consts::LOG2_E))
                    .alu(Ops::Exp2, &[]),
            )
        })
        .add(UPat::op(Ops::Log).src(vec![UPat::var("x")]), |c| {
            Some(
                c["x"]
                    .alu(Ops::Log2, &[])
                    .mul(&UOp::f32(std::f32::consts::LN_2)),
            )
        })
        .add(UPat::op(Ops::Cos).src(vec![UPat::var("x")]), |c| {
            Some(
                UOp::f32(std::f32::consts::FRAC_PI_2)
                    .sub(&c["x"])
                    .alu(Ops::Sin, &[]),
            )
        });
    graph_rewrite(&root, &pm)
}
fn coords(linear: &UOp, shape: &[UOp]) -> Vec<UOp> {
    let mut stride = UOp::int(1);
    let mut out = vec![];
    for n in shape.iter().rev() {
        out.push(linear.div(&stride).modulo(n));
        stride = stride.mul(n);
    }
    out.reverse();
    out
}
fn offset(indices: &[UOp], shape: &[UOp]) -> UOp {
    indices
        .iter()
        .zip(shape)
        .fold(UOp::int(0), |acc, (i, n)| acc.mul(n).add(i))
}
fn broadcast(indices: &[UOp], shape: &[UOp]) -> Vec<UOp> {
    let start = indices.len() - shape.len();
    shape
        .iter()
        .enumerate()
        .map(|(i, n)| {
            if *n == UOp::int(1) {
                UOp::int(0)
            } else {
                indices[start + i].clone()
            }
        })
        .collect()
}
struct Lower {
    ops: Vec<UOp>,
    seen: HashSet<u64>,
    next_range: usize,
    next_alloc: usize,
    cache: HashMap<(u64, Vec<u64>, u64), UOp>,
}
impl Lower {
    fn emit(&mut self, u: UOp) -> UOp {
        if self.seen.insert(u.id) {
            for s in u.ordered_sources() {
                self.emit(s.clone());
            }
            self.ops.push(u.clone());
        }
        u
    }
    fn range(&mut self, end: UOp, axis: Axis) -> UOp {
        let end = simplify(&end).unwrap_or(end);
        let r = UOp::range(end, self.next_range, axis);
        self.next_range += 1;
        self.emit(r)
    }
    fn close(&mut self, range: &UOp, body: UOp) -> UOp {
        self.emit(UOp::new(Ops::End, vec![body, range.clone()], Arg::None))
    }
    fn scalar(&mut self, u: &UOp, indices: &[UOp], valid: &UOp) -> Result<UOp> {
        let key = (u.id, indices.iter().map(|i| i.id).collect(), valid.id);
        if let Some(v) = self.cache.get(&key) {
            return Ok(v.clone());
        }
        let result = match (&u.op, &u.arg) {
            (Ops::Const, _) | (Ops::Variable, _) => self.emit(u.clone()),
            (Ops::Bind, _) => self.emit(u.src[1].clone()),
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
    fn accumulator(&mut self, initial: UOp) -> (UOp, UOp) {
        let alloc = UOp::new(
            Ops::Alloc,
            vec![],
            Arg::Allocation {
                id: self.next_alloc,
                dtype: DType::F32,
                size: 1,
            },
        );
        self.next_alloc += 1;
        self.emit(alloc.clone());
        let init = self.emit(alloc.store(initial));
        (alloc, init)
    }
}
pub fn rangeify(sink: &UOp) -> Result<UOp> {
    if sink.op != Ops::Sink {
        return Err("rangeify expects SINK".into());
    }
    let sink = normalize(sink)?;
    let mut lower = Lower {
        ops: vec![],
        seen: HashSet::new(),
        next_range: 0,
        next_alloc: 0,
        cache: HashMap::new(),
    };
    for store in &sink.src {
        if store.op != Ops::Store || store.src.len() != 2 {
            return Err("rangeify expects STORE(buffer, tensor)".into());
        }
        let target = &store.src[0];
        let value = &store.src[1];
        let shape = value.shape_uops()?;
        if target.shape_uops()? != shape {
            return Err("output buffer shape mismatch".into());
        }
        let r = lower.range(product(&shape), Axis::Loop);
        let coords = coords(&r, &shape);
        let result = lower.scalar(value, &coords, &UOp::bool(true))?;
        let output = lower.emit(target.index(r.clone(), UOp::bool(true)).store(result));
        lower.close(&r, output);
        lower.cache.clear();
    }
    let linear = UOp::linear(lower.ops);
    verify_lowered(&linear)?;
    Ok(linear)
}
pub fn linearize(sink: &UOp) -> Result<UOp> {
    let sink = crate::functions::inline_calls(sink)?;
    if sink.op == Ops::Linear {
        return Ok(sink.clone());
    }
    if sink.op != Ops::Sink {
        return Err("linearize expects SINK".into());
    }
    let linear = UOp::linear(
        sink.toposort()
            .into_iter()
            .filter(|x| x.op != Ops::Sink)
            .collect(),
    );
    verify_lowered(&linear)?;
    Ok(linear)
}
pub fn verify_lowered(linear: &UOp) -> Result<()> {
    if linear.op != Ops::Linear {
        return Err("renderer expects LINEAR".into());
    }
    let mut ranges = vec![];
    for u in &linear.src {
        if u.op == Ops::Range {
            if u.src.len() != 2 {
                return Err("RANGE needs start/end".into());
            }
            ranges.push(u.id)
        }
        if u.op == Ops::End {
            let id = u.src.last().ok_or("END missing RANGE")?.id;
            if ranges.pop() != Some(id) {
                return Err("END does not match active RANGE".into());
            }
        }
        if u.op == Ops::Index {
            if u.src.len() != 3 {
                return Err("INDEX needs buffer,index,valid".into());
            }
            if u.src[2] == UOp::bool(true) {
                if let (Some(size), Some((lo, hi))) =
                    (pointer_size(&u.src[0]), index_bounds(&u.src[1]))
                {
                    if lo < 0 || hi >= size as i64 {
                        return Err("unguarded index outside allocation".into());
                    }
                }
            }
        }
    }
    if !ranges.is_empty() {
        return Err("unclosed RANGE".into());
    }
    crate::uir::validate_linear_scope(linear)?;
    Ok(())
}
pub fn to_program(sink: &UOp, entry: &str) -> Result<UOp> {
    let linear = if sink
        .src
        .iter()
        .any(|s| s.op == Ops::Store && s.src.first().is_some_and(|p| p.op == Ops::Buffer))
    {
        rangeify(sink)?
    } else {
        linearize(sink)?
    };
    let source = render(&linear, entry)?;
    Ok(UOp::new(
        Ops::Program,
        vec![
            sink.clone(),
            linear,
            UOp::new(Ops::Source, vec![], Arg::Text(source)),
        ],
        Arg::Text(entry.into()),
    ))
}
#[derive(Clone, Debug)]
pub struct Signature {
    pub buffers: Vec<(usize, UOp)>,
    pub variables: Vec<(usize, UOp)>,
}
pub fn signature(linear: &UOp) -> Result<Signature> {
    let mut buffers = vec![];
    let mut variables = vec![];
    let mut seen = HashSet::new();
    let mut occupied = HashSet::new();
    for u in linear.toposort() {
        if let (Ops::Param, Arg::Slot { index, .. }) = (&u.op, &u.arg) {
            occupied.insert(*index);
            buffers.push((*index, u));
        }
    }
    let mut next = 0;
    for u in linear.toposort() {
        if !seen.insert(u.id) {
            continue;
        }
        if u.op == Ops::Buffer {
            while occupied.contains(&next) {
                next += 1
            }
            buffers.push((next, u));
            occupied.insert(next);
            next += 1;
        }
    }
    next = occupied.iter().copied().max().map_or(0, |x| x + 1);
    for u in linear.toposort() {
        if u.op == Ops::Variable {
            variables.push((next, u));
            next += 1;
        }
    }
    Ok(Signature { buffers, variables })
}
fn ctype(dt: DType) -> &'static str {
    match dt {
        DType::F32 => "float",
        DType::F32x4 => "float4",
        DType::I32 => "int",
        DType::I64 => "long long",
        DType::Bool => "int",
        DType::Ptr => "void*",
        DType::Void => "void",
    }
}
fn identifier(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c == '_' || c.is_ascii_alphabetic())
        && chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}
pub fn render(linear: &UOp, entry: &str) -> Result<String> {
    render_impl(linear, entry, None)
}
pub fn render_checked(linear: &UOp, entry: &str, status_slot: usize) -> Result<String> {
    render_impl(linear, entry, Some(status_slot))
}
fn render_impl(linear: &UOp, entry: &str, status_slot: Option<usize>) -> Result<String> {
    let mut all = vec![];
    let mut seen = HashSet::new();
    let mut effects = HashSet::new();
    let mut scopes = vec![];
    fn add(
        u: &UOp,
        out: &mut Vec<UOp>,
        seen: &mut HashSet<u64>,
        effects: &mut HashSet<u64>,
        scopes: &mut Vec<HashSet<u64>>,
    ) {
        if seen.contains(&u.id) || effects.contains(&u.id) {
            return;
        }
        for src in u.ordered_sources() {
            add(src, out, seen, effects, scopes)
        }
        if u.op == Ops::Range {
            scopes.push(seen.clone());
        }
        seen.insert(u.id);
        out.push(u.clone());
        if matches!(u.op, Ops::Range | Ops::End | Ops::Store | Ops::Call) {
            effects.insert(u.id);
        }
        if u.op == Ops::End {
            if let Some(parent) = scopes.pop() {
                *seen = parent;
            }
        }
    }
    for u in &linear.src {
        add(u, &mut all, &mut seen, &mut effects, &mut scopes)
    }
    let complete = UOp::linear(all);
    let linear = &complete;
    verify_lowered(linear)?;

    if !identifier(entry) {
        return Err("invalid C entry identifier".into());
    }
    let sig = signature(linear)?;
    let mut text = String::from(
        "extern float exp2f(float);\nextern float log2f(float);\nextern float sinf(float);\nextern float sqrtf(float);\nextern float fmaxf(float,float);\nstatic float chibi_float(unsigned int x) { union { unsigned int u; float f; } b; b.u=x; return b.f; }\nstatic long long chibi_div(long long a,long long b) { long long q=a/b, r=a%b; return q-((r!=0)&&((r<0)!=(b<0))); }\nstatic long long chibi_mod(long long a,long long b) { long long r=a%b; return r+(((r!=0)&&((r<0)!=(b<0)))?b:0); }\n",
    );
    text.push_str(
        "static int chibi_int(unsigned int x) { union { unsigned int u; int i; } b; b.u=x; return b.i; }\nstatic unsigned int chibi_bits(float x) { union { unsigned int u; float f; } b; b.f=x; return b.u; }\nstatic int chibi_shr(int a,int b) { unsigned int u=(unsigned int)a; return chibi_int((u>>b)|((a<0&&b)?(~0u<<(32-b)):0u)); }\nstatic long long chibi_shr64(long long a,long long b) { unsigned long long u=(unsigned long long)a; return (long long)((u>>b)|((a<0&&b)?(~0ull<<(64-b)):0ull)); }\nstatic float chibi_max(float a,float b) { if(a!=a||b!=b)return chibi_float(2143289344u); if(a==0.0f&&b==0.0f)return 0.0f; return a>b?a:b; }\n",
    );
    let mut functions: HashMap<String, (DType, Vec<DType>)> = HashMap::new();
    for u in &linear.src {
        let (body, args) = if u.op == Ops::Call {
            let b = u.src.first().ok_or("CALL lacks body")?;
            (b, &u.src[1..])
        } else if u.op == Ops::CustomFunction && !u.src.is_empty() {
            (u, &u.src[..])
        } else {
            continue;
        };
        let Arg::Function { name, dtype } = &body.arg else {
            continue;
        };
        if !identifier(name) {
            return Err("invalid C function identifier".into());
        }
        let types = args.iter().map(UOp::dtype).collect::<Vec<_>>();
        if functions
            .get(name)
            .is_some_and(|old| old != &(*dtype, types.clone()))
        {
            return Err(format!("inconsistent callback signature for {name}"));
        }
        functions.insert(name.clone(), (*dtype, types));
    }
    let mut function_names = functions.keys().collect::<Vec<_>>();
    function_names.sort();
    for name in function_names {
        let (ret, args) = &functions[name];
        text.push_str(&format!(
            "extern {} {}({});\n",
            ctype(*ret),
            name,
            if args.is_empty() {
                "void".into()
            } else {
                args.iter().map(|d| ctype(*d)).collect::<Vec<_>>().join(",")
            }
        ));
    }
    text.push_str("typedef float float4 __attribute__((ext_vector_type(4),aligned(1)));\n");
    text.push_str(&format!("void {entry}(unsigned char *args) {{\n"));
    let mut values = HashMap::new();
    let mut addresses: HashMap<u64, (String, String, String, DType)> = HashMap::new();
    for (slot, u) in &sig.buffers {
        let name = format!("b{}", u.id);
        text.push_str(&format!(
            "  void *{name}=(void*)(unsigned long long)((unsigned long long*)args)[{slot}];\n"
        ));
        values.insert(u.id, name);
    }
    for (slot, u) in &sig.variables {
        let name = format!("v{}", u.id);
        text.push_str(&format!("  long long {name}=((long long*)args)[{slot}];\n"));
        values.insert(u.id, name);
    }
    let mut depth = 1;
    let mut occurrence = 0usize;
    let mut value_scopes = vec![];
    let mut address_scopes = vec![];
    for u in &linear.src {
        let get = |s: &UOp| -> Result<String> {
            values
                .get(&s.id)
                .cloned()
                .ok_or_else(|| format!("{:?} {} used before definition", s.op, s.id))
        };
        let at = |i: usize| -> Result<String> { get(u.src.get(i).ok_or("missing source")?) };
        let indent = "  ".repeat(depth);
        if let Some(slot) = status_slot {
            let reject = format!(
                "{{ ((int*)(unsigned long long)((unsigned long long*)args)[{slot}])[0]=1; return; }}"
            );
            let a = || at(0);
            let b = || at(1);
            let guard = match u.op {
                Ops::Shl | Ops::Shr => Some(format!(
                    "({}<0 || {}>={})",
                    b()?,
                    b()?,
                    if u.dtype() == DType::I32 { 32 } else { 64 }
                )),
                Ops::Div | Ops::Mod if matches!(u.dtype(), DType::I32 | DType::I64) => {
                    Some(format!(
                        "({}==0 || ({}=={} && {}==-1))",
                        b()?,
                        a()?,
                        if u.dtype() == DType::I32 {
                            "(-2147483647-1)"
                        } else {
                            "(-9223372036854775807ll-1)"
                        },
                        b()?
                    ))
                }
                Ops::Cast
                    if u.src[0].dtype() == DType::F32
                        && matches!(u.dtype(), DType::I32 | DType::I64) =>
                {
                    let limit = if u.dtype() == DType::I32 {
                        "2147483648.0f"
                    } else {
                        "9223372036854775808.0f"
                    };
                    Some(format!("!({} >= -{limit} && {} < {limit})", a()?, a()?))
                }
                Ops::Add | Ops::Sub | Ops::Mul if u.dtype() == DType::I64 => {
                    let operation = match u.op {
                        Ops::Add => "add",
                        Ops::Sub => "sub",
                        _ => "mul",
                    };
                    text.push_str(&format!("{indent}long long checked{};\n", u.id));
                    Some(format!(
                        "__builtin_{operation}_overflow({}, {}, &checked{})",
                        a()?,
                        b()?,
                        u.id
                    ))
                }
                Ops::Neg if u.dtype() == DType::I64 => {
                    Some(format!("({}==(-9223372036854775807ll-1))", a()?))
                }
                _ => None,
            };
            if let Some(condition) = guard {
                text.push_str(&format!("{indent}if ({condition}) {reject}\n"));
            }
        }
        let expression = match (&u.op, &u.arg) {
            (Ops::Const, Arg::Float(bits)) => {
                let bits = (bits.value() as f32).to_bits();
                values.insert(u.id, format!("chibi_float({bits}u)"));
                continue;
            }
            (Ops::Const, Arg::I32(v)) => {
                values.insert(u.id, format!("((int){}u)", *v as u32));
                continue;
            }
            (Ops::Const, Arg::Int(v)) => {
                values.insert(u.id, format!("((long long){}ull)", *v as u64));
                continue;
            }
            (Ops::Const, Arg::Bool(v)) => {
                values.insert(u.id, if *v { "1" } else { "0" }.into());
                continue;
            }
            (Ops::Buffer, _) | (Ops::Param, _) | (Ops::Variable, _) => continue,
            (Ops::Bind, _) => at(1)?,
            (Ops::Alloc, Arg::Allocation { dtype, size, .. }) => {
                let name = format!("a{}", u.id);
                text.push_str(&format!(
                    "{indent}{} {name}[{}];\n",
                    ctype(*dtype),
                    size.max(&1)
                ));
                values.insert(u.id, name);
                continue;
            }
            (Ops::After, _) => {
                values.insert(u.id, at(0)?);
                if let Some(addr) = addresses.get(&u.src[0].id).cloned() {
                    addresses.insert(u.id, addr);
                }
                continue;
            }
            (Ops::Index, _) => {
                let dtype = if let Arg::DType(dt) = u.arg {
                    dt
                } else {
                    DType::F32
                };
                addresses.insert(u.id, (at(0)?, at(1)?, at(2)?, dtype));
                values.insert(
                    u.id,
                    format!("(({}*){})+({})", ctype(dtype), at(0)?, at(1)?),
                );
                continue;
            }
            (Ops::Load, _) => {
                let dtype = if let Arg::DType(dt) = u.arg {
                    dt
                } else {
                    DType::F32
                };
                if let Some((base, index, gate, _)) = addresses.get(&u.src[0].id) {
                    format!(
                        "({gate} ? (({}*){base})[{index}] : ({})0)",
                        ctype(dtype),
                        ctype(dtype)
                    )
                } else {
                    format!("(({}*){})[0]", ctype(dtype), at(0)?)
                }
            }
            (Ops::Store, _) => {
                let value = at(1)?;
                if status_slot.is_some_and(
                    |slot| matches!(u.src[0].arg, Arg::Slot { index,.. } if index == slot),
                ) {
                    text.push_str(&format!(
                        "{indent}if (!({value})) {{ ((int*){})[0]=1; return; }}\n",
                        at(0)?
                    ));
                    values.insert(u.id, "0".into());
                    continue;
                }
                if let Some((base, index, gate, dtype)) = addresses.get(&u.src[0].id) {
                    text.push_str(&format!(
                        "{indent}if ({gate}) (({}*){base})[{index}]={value};\n",
                        ctype(*dtype)
                    ));
                } else {
                    let dtype = if let Arg::DType(dt) = u.arg {
                        dt
                    } else {
                        u.src[1].dtype()
                    };
                    text.push_str(&format!(
                        "{indent}(({}*){})[0]={value};\n",
                        ctype(dtype),
                        at(0)?
                    ));
                }
                values.insert(u.id, "0".into());
                continue;
            }
            (Ops::Range, _) => {
                value_scopes.push(values.clone());
                address_scopes.push(addresses.clone());
                let name = format!("r{}", u.id);
                text.push_str(&format!(
                    "{indent}for (long long {name}={}; {name}<{}; {name}++) {{\n",
                    at(0)?,
                    at(1)?
                ));
                values.insert(u.id, name);
                depth += 1;
                continue;
            }
            (Ops::End, _) => {
                if depth == 1 {
                    return Err("extra END".into());
                }
                depth -= 1;
                text.push_str(&format!("{}}}\n", "  ".repeat(depth)));
                values = value_scopes.pop().ok_or("missing scope")?;
                addresses = address_scopes.pop().ok_or("missing address scope")?;
                values.insert(u.id, "0".into());
                continue;
            }
            (Ops::Cast, Arg::DType(DType::Bool)) => format!("({}!=0)", at(0)?),
            (Ops::Cast, Arg::DType(dtype)) => {
                format!("(({})({}))", ctype(*dtype), at(0)?)
            }
            (Ops::Bitcast, Arg::DType(dt)) if u.src[0].dtype() == *dt => at(0)?,
            (Ops::Bitcast, Arg::DType(DType::I32)) => {
                format!("chibi_int(chibi_bits({}))", at(0)?)
            }
            (Ops::Bitcast, Arg::DType(DType::F32)) => {
                format!("chibi_float((unsigned int)({}))", at(0)?)
            }
            (Ops::Shl, _) => {
                if u.dtype() == DType::I32 {
                    format!("chibi_int((unsigned int)({}) << ({}))", at(0)?, at(1)?)
                } else {
                    format!(
                        "((long long)((unsigned long long)({}) << ({})))",
                        at(0)?,
                        at(1)?
                    )
                }
            }
            (Ops::Shr, _) => {
                format!(
                    "{}({},{})",
                    if u.dtype() == DType::I32 {
                        "chibi_shr"
                    } else {
                        "chibi_shr64"
                    },
                    at(0)?,
                    at(1)?
                )
            }
            (Ops::Neg, _) => {
                if u.dtype() == DType::I32 {
                    format!("((int)(0u-(unsigned int)({})))", at(0)?)
                } else {
                    format!("(-({}))", at(0)?)
                }
            }
            (Ops::Exp2, _) => format!("exp2f({})", at(0)?),
            (Ops::Log2, _) => format!("log2f({})", at(0)?),
            (Ops::Sin, _) => format!("sinf({})", at(0)?),
            (Ops::Sqrt, _) => format!("sqrtf({})", at(0)?),
            (Ops::Where, _) => format!("({}?{}:{})", at(0)?, at(1)?, at(2)?),
            (Ops::Max, _) if u.dtype() == DType::F32 => {
                format!("chibi_max({},{})", at(0)?, at(1)?)
            }
            (Ops::Max, _) => format!("({}>{}?{}:{})", at(0)?, at(1)?, at(0)?, at(1)?),
            (Ops::Div, _) if matches!(u.dtype(), DType::I32 | DType::I64) => {
                format!("chibi_div({},{})", at(0)?, at(1)?)
            }
            (Ops::Mod, _) => format!("chibi_mod({},{})", at(0)?, at(1)?),
            (op, _) if is_binary(*op) => {
                let symbol = match op {
                    Ops::Add => "+",
                    Ops::Sub => "-",
                    Ops::Mul => "*",
                    Ops::Div => "/",
                    Ops::Lt => "<",
                    Ops::Eq => "==",
                    Ops::And => "&",
                    Ops::Or => "|",
                    Ops::Xor => "^",
                    _ => return Err("unsupported ALU".into()),
                };
                if u.dtype() == DType::I32 && matches!(op, Ops::Add | Ops::Sub | Ops::Mul) {
                    format!(
                        "((int)((unsigned int)({}) {symbol} (unsigned int)({})))",
                        at(0)?,
                        at(1)?
                    )
                } else {
                    format!("({} {symbol} {})", at(0)?, at(1)?)
                }
            }
            (Ops::Call, _) => {
                let body = u.src.first().ok_or("CALL body missing")?;
                let Arg::Function { name, .. } = &body.arg else {
                    return Err("scalar CALL must target CUSTOM_FUNCTION".into());
                };
                format!(
                    "{}({})",
                    name,
                    u.src[1..]
                        .iter()
                        .map(get)
                        .collect::<Result<Vec<_>>>()?
                        .join(",")
                )
            }
            (Ops::CustomFunction, Arg::Function { name, .. }) => {
                if u.src.is_empty() {
                    values.insert(u.id, name.clone());
                    continue;
                }
                format!(
                    "{}({})",
                    name,
                    u.src.iter().map(get).collect::<Result<Vec<_>>>()?.join(",")
                )
            }
            (Ops::Sink, _) => continue,
            _ => return Err(format!("C renderer does not support {:?}", u.op)),
        };
        if u.dtype() == DType::Void {
            text.push_str(&format!("{indent}{expression};\n"));
            values.insert(u.id, "0".into());
        } else {
            let name = format!("v{}_{}", u.id, occurrence);
            occurrence += 1;
            text.push_str(&format!(
                "{indent}{} {name}={expression};\n",
                ctype(u.dtype())
            ));
            values.insert(u.id, name);
        }
    }
    if depth != 1 {
        return Err("unclosed C loop".into());
    }
    text.push_str("}\n");
    Ok(text)
}
fn product(shape: &[UOp]) -> UOp {
    shape.iter().fold(UOp::int(1), |a, b| a.mul(b))
}

fn pointer_size(u: &UOp) -> Option<usize> {
    match &u.arg {
        Arg::Slot { shape, .. } => Some(numel(shape)),
        Arg::Buffer { shape, .. } => Some(numel(shape)),
        _ => None,
    }
}

fn index_bounds(u: &UOp) -> Option<(i64, i64)> {
    let b = u.bounds()?;
    Some((b.min, b.max))
}
```

`src/functions.rs` の `fn selector_bounds`を置き換える。

```rust
// file="src/functions.rs" mode="replace" item="fn selector_bounds"
fn selector_bounds(index: &UOp) -> Result<(i64, i64)> {
    if index.dtype() != DType::I64 {
        return Err("STACK selector must have I64 dtype".into());
    }
    let b = index.bounds().ok_or("STACK selector needs proven bounds")?;
    Ok((b.min, b.max))
}
```

`src/gpu.rs`を保存する。

```rust
// file="src/gpu.rs" mode="write"
use crate::Axis;
use crate::{Arg, DType, Ops, UOp};
use std::collections::HashSet;
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Launch {
    pub groups: [usize; 3],
    pub local: [usize; 3],
}
impl Launch {
    pub fn validate(&self, max_threads: usize) -> Result<(), String> {
        let n = self
            .local
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .ok_or("local size overflow")?;
        if n == 0 || n > max_threads || self.groups.contains(&0) {
            return Err("invalid launch dimensions".into());
        }
        for i in 0..3 {
            self.groups[i]
                .checked_mul(self.local[i])
                .ok_or("global size overflow")?;
        }
        Ok(())
    }
}
#[derive(Clone, Debug)]
pub struct Kernel {
    pub source: String,
    pub entry: String,
    pub launch: Launch,
    pub shared_bytes: usize,
}
#[derive(Clone, Debug, Default)]
pub struct Capabilities {
    pub name: String,
    pub max_threads: usize,
    pub local_memory: usize,
    pub indirect_commands: bool,
    pub families: Vec<String>,
    pub extensions: Vec<String>,
}
pub(crate) fn span(size: usize, offset: usize, length: usize) -> Result<(), String> {
    if offset.checked_add(length).is_none_or(|n| n > size) {
        Err("buffer access outside allocation".into())
    } else {
        Ok(())
    }
}

pub(crate) struct Builder {
    ops: Vec<UOp>,
    seen: HashSet<u64>,
    pub(crate) next: usize,
}
impl Builder {
    pub(crate) fn new() -> Self {
        Self {
            ops: vec![],
            seen: HashSet::new(),
            next: 0,
        }
    }
    pub(crate) fn emit(&mut self, u: UOp) -> UOp {
        for s in &u.src {
            if !self.seen.contains(&s.id) {
                self.emit(s.clone());
            }
        }
        if self.seen.insert(u.id) {
            self.ops.push(u.clone());
        }
        u
    }
    pub(crate) fn special(&mut self, name: &str) -> UOp {
        self.emit(UOp::new(Ops::Special, vec![], Arg::Text(name.into())))
    }
    pub(crate) fn param(&mut self, index: usize, size: usize) -> UOp {
        self.emit(UOp::new(
            Ops::Param,
            vec![],
            Arg::Slot {
                index,
                shape: vec![size],
            },
        ))
    }
    pub(crate) fn alloc(&mut self, size: usize) -> UOp {
        let id = self.next;
        self.next += 1;
        self.emit(UOp::new(
            Ops::Alloc,
            vec![],
            Arg::Allocation {
                id,
                dtype: DType::F32,
                size,
            },
        ))
    }
    pub(crate) fn range(&mut self, end: usize) -> UOp {
        let id = self.next;
        self.next += 1;
        self.emit(UOp::range(UOp::int(end as i64), id, Axis::Reduce))
    }
    pub(crate) fn end(&mut self, r: &UOp) -> UOp {
        self.emit(UOp::new(Ops::End, vec![r.clone()], Arg::None))
    }
    pub(crate) fn barrier(&mut self, dep: UOp) -> UOp {
        self.emit(UOp::new(Ops::Barrier, vec![dep], Arg::None))
    }
    pub(crate) fn load(&mut self, b: &UOp, index: UOp, valid: UOp) -> UOp {
        self.emit(b.index(index, valid).load())
    }
    pub(crate) fn store(&mut self, b: &UOp, index: UOp, value: UOp, valid: UOp) -> UOp {
        self.emit(b.index(index, valid).store(value))
    }
    pub(crate) fn finish(self) -> UOp {
        UOp::linear(self.ops)
    }
}
/// Strided lane accumulation and shared-memory tree reduction, expressed only with common UOps.

pub fn vector_add(count: usize, local: usize) -> Result<(UOp, Launch), String> {
    if count == 0 || !count.is_multiple_of(4) || local == 0 {
        return Err("vector add requires a nonzero multiple of four elements".into());
    }
    let mut b = Builder::new();
    let a = b.param(0, count);
    let rhs = b.param(1, count);
    let out = b.param(2, count);
    let gid = b.special("gid0");
    let valid = gid.lt(&UOp::int((count / 4) as i64));
    let index = |p: UOp| {
        UOp::new(
            Ops::Index,
            vec![p, gid.clone(), valid.clone()],
            Arg::DType(DType::F32x4),
        )
    };
    let a = b.emit(UOp::new(
        Ops::Load,
        vec![index(a)],
        Arg::DType(DType::F32x4),
    ));
    let rhs = b.emit(UOp::new(
        Ops::Load,
        vec![index(rhs)],
        Arg::DType(DType::F32x4),
    ));
    b.emit(index(out).store(a.add(&rhs)));
    Ok((
        b.finish(),
        Launch {
            groups: [(count / 4).div_ceil(local), 1, 1],
            local: [local, 1, 1],
        },
    ))
}

pub fn local_reduction(rows: usize, width: usize, local: usize) -> Result<(UOp, Launch), String> {
    if rows == 0 || width == 0 || !local.is_power_of_two() {
        return Err("reduction requires nonzero shape and power-of-two local size".into());
    }
    let mut b = Builder::new();
    let input = b.param(0, rows.checked_mul(width).ok_or("reduction overflow")?);
    let output = b.param(1, rows);
    let lane = b.special("lid0");
    let row = b.special("group0");
    let shared = b.alloc(local);
    let acc = b.alloc(1);
    let init = b.store(&acc, UOp::int(0), UOp::f32(0.), UOp::bool(true));
    let r = b.range(width.div_ceil(local));
    let col = r.mul(&UOp::int(local as i64)).add(&lane);
    let value = b.load(
        &input,
        row.mul(&UOp::int(width as i64)).add(&col),
        col.lt(&UOp::int(width as i64)),
    );
    let old = b.load(&acc.after(vec![init]), UOp::int(0), UOp::bool(true));
    b.store(&acc, UOp::int(0), old.add(&value), UOp::bool(true));
    let end = b.end(&r);
    let acc = b.load(&acc.after(vec![end]), UOp::int(0), UOp::bool(true));
    let stored = b.store(&shared, lane.clone(), acc, UOp::bool(true));
    let mut dep = b.barrier(stored);
    for step in (0..local.ilog2()).rev().map(|s| 1usize << s) {
        let valid = lane.lt(&UOp::int(step as i64));
        let a = b.load(
            &shared.after(vec![dep.clone()]),
            lane.clone(),
            valid.clone(),
        );
        let v = b.load(
            &shared.after(vec![dep]),
            lane.add(&UOp::int(step as i64)),
            valid.clone(),
        );
        let s = b.store(&shared, lane.clone(), a.add(&v), valid);
        dep = b.barrier(s);
    }
    let result = b.load(&shared.after(vec![dep]), UOp::int(0), UOp::bool(true));
    b.store(&output, row, result, lane.eq(&UOp::int(0)));
    Ok((
        b.finish(),
        Launch {
            groups: [rows, 1, 1],
            local: [local, 1, 1],
        },
    ))
}

pub fn tiled_gemm(m: usize, n: usize, k: usize, tile: usize) -> Result<(UOp, Launch), String> {
    if m == 0 || n == 0 || k == 0 || tile == 0 || tile.checked_mul(tile).is_none() {
        return Err("GEMM dimensions must be nonzero and representable".into());
    }
    let mut b = Builder::new();
    let a = b.param(0, m.checked_mul(k).ok_or("GEMM overflow")?);
    let rhs = b.param(1, k.checked_mul(n).ok_or("GEMM overflow")?);
    let out = b.param(2, m.checked_mul(n).ok_or("GEMM overflow")?);
    let x = b.special("lid0");
    let y = b.special("lid1");
    let gx = b.special("group0");
    let gy = b.special("group1");
    let row = gy.mul(&UOp::int(tile as i64)).add(&y);
    let col = gx.mul(&UOp::int(tile as i64)).add(&x);
    let sa = b.alloc(tile * tile);
    let sb = b.alloc(tile * tile);
    let acc = b.alloc(1);
    let init = b.store(&acc, UOp::int(0), UOp::f32(0.), UOp::bool(true));
    let block = b.range(k.div_ceil(tile));
    let ka = block.mul(&UOp::int(tile as i64)).add(&x);
    let kb = block.mul(&UOp::int(tile as i64)).add(&y);
    let av = b.load(
        &a,
        row.mul(&UOp::int(k as i64)).add(&ka),
        row.lt(&UOp::int(m as i64)).and(&ka.lt(&UOp::int(k as i64))),
    );
    let bv = b.load(
        &rhs,
        kb.mul(&UOp::int(n as i64)).add(&col),
        kb.lt(&UOp::int(k as i64)).and(&col.lt(&UOp::int(n as i64))),
    );
    let idx = y.mul(&UOp::int(tile as i64)).add(&x);
    let ast = b.store(&sa, idx.clone(), av, UOp::bool(true));
    let bst = b.store(&sb.after(vec![ast]), idx, bv, UOp::bool(true));
    let ready = b.barrier(bst);
    let t = b.range(tile);
    let av = b.load(
        &sa.after(vec![ready.clone()]),
        y.mul(&UOp::int(tile as i64)).add(&t),
        UOp::bool(true),
    );
    let bv = b.load(
        &sb.after(vec![ready]),
        t.mul(&UOp::int(tile as i64)).add(&x),
        UOp::bool(true),
    );
    let old = b.load(&acc.after(vec![init]), UOp::int(0), UOp::bool(true));
    b.store(&acc, UOp::int(0), old.add(&av.mul(&bv)), UOp::bool(true));
    let tend = b.end(&t);
    b.barrier(tend);
    let end = b.end(&block);
    let final_value = b.load(&acc.after(vec![end]), UOp::int(0), UOp::bool(true));
    b.store(
        &out,
        row.mul(&UOp::int(n as i64)).add(&col),
        final_value,
        row.lt(&UOp::int(m as i64))
            .and(&col.lt(&UOp::int(n as i64))),
    );
    Ok((
        b.finish(),
        Launch {
            groups: [n.div_ceil(tile), m.div_ceil(tile), 1],
            local: [tile, tile, 1],
        },
    ))
}

pub fn convolution(
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    outputs: usize,
    kh: usize,
    kw: usize,
    local: usize,
) -> Result<(UOp, Launch), String> {
    if [batch, channels, height, width, outputs, kh, kw, local].contains(&0)
        || kh > height
        || kw > width
    {
        return Err("invalid convolution dimensions".into());
    }
    let oh = height - kh + 1;
    let ow = width - kw + 1;
    let count = batch
        .checked_mul(outputs)
        .and_then(|x| x.checked_mul(oh))
        .and_then(|x| x.checked_mul(ow))
        .ok_or("convolution overflow")?;
    let mut b = Builder::new();
    let input = b.param(0, batch * channels * height * width);
    let weights = b.param(1, outputs * channels * kh * kw);
    let output = b.param(2, count);
    let gid = b.special("gid0");
    let ox = gid.modulo(&UOp::int(ow as i64));
    let oy = gid.div(&UOp::int(ow as i64)).modulo(&UOp::int(oh as i64));
    let oc = gid
        .div(&UOp::int((ow * oh) as i64))
        .modulo(&UOp::int(outputs as i64));
    let bn = gid.div(&UOp::int((ow * oh * outputs) as i64));
    let acc = b.alloc(1);
    let init = b.store(&acc, UOp::int(0), UOp::f32(0.), UOp::bool(true));
    let r = b.range(channels * kh * kw);
    let kx = r.modulo(&UOp::int(kw as i64));
    let ky = r.div(&UOp::int(kw as i64)).modulo(&UOp::int(kh as i64));
    let c = r.div(&UOp::int((kw * kh) as i64));
    let ai = bn
        .mul(&UOp::int(channels as i64))
        .add(&c)
        .mul(&UOp::int(height as i64))
        .add(&oy.add(&ky))
        .mul(&UOp::int(width as i64))
        .add(&ox.add(&kx));
    let wi = oc.mul(&UOp::int((channels * kh * kw) as i64)).add(&r);
    let valid = gid.lt(&UOp::int(count as i64));
    let a = b.load(&input, ai, valid.clone());
    let w = b.load(&weights, wi, valid.clone());
    let old = b.load(&acc.after(vec![init]), UOp::int(0), UOp::bool(true));
    b.store(&acc, UOp::int(0), old.add(&a.mul(&w)), UOp::bool(true));
    let end = b.end(&r);
    let value = b.load(&acc.after(vec![end]), UOp::int(0), UOp::bool(true));
    b.store(&output, gid, value, valid);
    Ok((
        b.finish(),
        Launch {
            groups: [count.div_ceil(local), 1, 1],
            local: [local, 1, 1],
        },
    ))
}
```

`src/gpu_tensor.rs`を保存する。

```rust
// file="src/gpu_tensor.rs" mode="write"
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::backends::metal::MetalDevice;
#[cfg(feature = "opencl")]
use crate::backends::opencl::OpenClDevice;
use crate::{
    Arg, Axis, Buffer, Ops, Result, Tensor, UOp,
    codegen::{rangeify, signature},
    runtime::InputSpec,
    schedule::{Schedule, create_schedule},
    uop::numel,
};
use crate::{
    gpu::{Kernel, Launch},
    queue::{CompiledQueue, HWQueue},
    renderers::{Dialect, render},
};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Metal,
    OpenCl,
}
enum Device {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(MetalDevice),
    #[cfg(feature = "opencl")]
    OpenCl(OpenClDevice),
}
#[derive(Clone)]
enum Allocation {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(crate::backends::metal::MetalBuffer),
    #[cfg(feature = "opencl")]
    OpenCl(crate::backends::opencl::OpenClBuffer),
}
impl Device {
    fn allocate(&self, bytes: usize) -> Result<Allocation> {
        match self {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(d) => d.allocate(bytes.max(4)).map(Allocation::Metal),
            #[cfg(feature = "opencl")]
            Self::OpenCl(d) => d.allocate(bytes.max(4)).map(Allocation::OpenCl),
        }
    }
}
impl Allocation {
    fn retain(&self, queue: &mut HWQueue) -> UOp {
        match self {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(b) => queue.retain_metal_buffer(b.clone()),
            #[cfg(feature = "opencl")]
            Self::OpenCl(b) => queue.retain_opencl_buffer(b.clone()),
        }
    }
    fn transfer(&self, queue: &mut HWQueue, pointer: UOp, bytes: UOp, read: bool) {
        let buffer = self.retain(queue);
        match (self, read) {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            (Self::Metal(_), false) => queue.metal_write(buffer, pointer, bytes),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            (Self::Metal(_), true) => queue.metal_read(buffer, pointer, bytes),
            #[cfg(feature = "opencl")]
            (Self::OpenCl(_), false) => queue.opencl_write(buffer, pointer, bytes),
            #[cfg(feature = "opencl")]
            (Self::OpenCl(_), true) => queue.opencl_read(buffer, pointer, bytes),
        }
    }
}
enum HostBuffer {
    Input(usize),
    Captured(usize),
    Scalars(usize),
    Output(usize),
    Error,
}
pub struct GpuExecutable {
    inputs: Vec<InputSpec>,
    schedule: Schedule,
    programs: Vec<UOp>,
    queue: CompiledQueue,
    input_nodes: Vec<UOp>,
    captured: Vec<Buffer>,
    variables: Vec<Vec<UOp>>,
    transfers: Vec<HostBuffer>,
    sequence: i64,
}

fn transfer(
    queue: &mut HWQueue,
    transfers: &mut Vec<HostBuffer>,
    buffer: &Allocation,
    host: HostBuffer,
    read: bool,
) {
    let index = 2 + transfers.len() * 2;
    transfers.push(host);
    buffer.transfer(
        queue,
        HWQueue::runtime_slot(index),
        HWQueue::runtime_slot(index + 1),
        read,
    );
}

fn parallelize(linear: UOp, single_output: bool) -> (UOp, Launch) {
    let mut launch = Launch {
        groups: [1; 3],
        local: [1; 3],
    };
    let ranges = linear
        .src
        .iter()
        .filter(|u| {
            matches!(
                u.arg,
                Arg::Range {
                    axis: Axis::Loop,
                    ..
                }
            )
        })
        .collect::<Vec<_>>();
    let checked_indices = linear
        .src
        .iter()
        .any(|u| matches!(&u.arg, Arg::Buffer { name, .. } if name == "__chibi_error"));
    if !single_output || ranges.len() != 1 || checked_indices {
        return (linear, launch);
    }
    let range = ranges[0];
    let Arg::Int(extent) = range.src[1].arg else {
        return (linear, launch);
    };
    if extent <= 1 {
        return (linear, launch);
    }
    launch.groups[0] = extent as usize;
    let range_id = range.id;
    let mut replacements = HashMap::new();
    let mut ops = vec![];
    for u in &linear.src {
        let sources = u
            .src
            .iter()
            .map(|s| replacements.get(&s.id).unwrap_or(s).clone())
            .collect();
        let value = if u.id == range_id {
            UOp::new(
                Ops::Range,
                sources,
                Arg::Range {
                    id: 0,
                    axis: Axis::Global,
                },
            )
        } else {
            u.replace(sources)
        };
        replacements.insert(u.id, value.clone());
        ops.push(value);
    }
    (UOp::linear(ops), launch)
}

impl GpuExecutable {
    pub fn inputs(&self) -> &[InputSpec] {
        &self.inputs
    }
    pub fn programs(&self) -> &[UOp] {
        &self.programs
    }
    pub fn schedule(&self) -> &Schedule {
        &self.schedule
    }

    pub fn compile(outputs: &[Tensor], backend: Backend) -> Result<Self> {
        let roots = outputs.iter().map(|t| t.uop.clone()).collect::<Vec<_>>();
        let schedule = create_schedule(&roots)?;
        let device = match backend {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal => Device::Metal(MetalDevice::open()?),
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            Backend::Metal => return Err("Metal requires macOS".into()),
            #[cfg(not(feature = "opencl"))]
            Backend::OpenCl => return Err("OpenCL feature is disabled".into()),
            #[cfg(feature = "opencl")]
            Backend::OpenCl => Device::OpenCl(OpenClDevice::open(0)?),
        };
        let dialect = match backend {
            Backend::Metal => Dialect::Metal,
            Backend::OpenCl => Dialect::OpenCl,
        };
        let mut queue = HWQueue::new();
        let mut transfers = vec![];
        let mut allocations = HashMap::new();
        let mut sizes = vec![];
        for life in &schedule.lifetimes {
            if sizes.len() <= life.slot {
                sizes.resize(life.slot + 1, 0)
            }
            sizes[life.slot] = sizes[life.slot].max(life.elements);
        }
        let slots = sizes
            .iter()
            .map(|&n| device.allocate(n * 4))
            .collect::<Result<Vec<_>>>()?;
        for life in &schedule.lifetimes {
            allocations.insert(life.buffer.id, slots[life.slot].clone());
        }
        let produced = schedule
            .lifetimes
            .iter()
            .map(|l| l.buffer.id)
            .collect::<HashSet<_>>();
        let mut inputs = vec![];
        let mut input_nodes = vec![];
        let mut captured = vec![];
        for node in schedule
            .buffers
            .iter()
            .filter(|u| !produced.contains(&u.id))
        {
            let Arg::Buffer { name, shape, data } = &node.arg else {
                return Err("invalid input buffer".into());
            };
            let allocation = device.allocate(numel(shape) * 4)?;
            let host = if let Some(bits) = data {
                let i = captured.len();
                captured.push(Buffer::new(
                    shape.clone(),
                    bits.iter().map(|&x| f32::from_bits(x)).collect(),
                )?);
                HostBuffer::Captured(i)
            } else {
                let i = inputs.len();
                inputs.push(InputSpec {
                    name: name.clone(),
                    shape: shape.clone(),
                });
                input_nodes.push(node.clone());
                HostBuffer::Input(i)
            };
            transfer(&mut queue, &mut transfers, &allocation, host, false);
            allocations.insert(node.id, allocation);
        }
        let error = device.allocate(4)?;
        transfer(&mut queue, &mut transfers, &error, HostBuffer::Error, false);
        let mut programs = vec![];
        let mut kernels: Vec<(Kernel, Vec<Allocation>)> = vec![];
        let mut variables = vec![];
        for (i, plan) in schedule.kernels.iter().enumerate() {
            let (linear, launch) = parallelize(rangeify(&plan.sink)?, plan.writes.len() == 1);
            let sig = signature(&linear)?;
            let kernel = render(&linear, &format!("chibi_gpu_{i}"), dialect, launch)?;
            let mut bindings = vec![];
            for (slot, node) in sig.buffers {
                if bindings.len() != slot {
                    return Err("GPU buffer signature has a gap".into());
                }
                let buffer = if matches!(&node.arg, Arg::Buffer { name, .. } if name == "__chibi_error")
                {
                    error.clone()
                } else {
                    allocations
                        .get(&node.id)
                        .ok_or("missing GPU allocation")?
                        .clone()
                };
                bindings.push(buffer);
            }
            let mut ordered = BTreeMap::<String, UOp>::new();
            for (_, node) in sig.variables {
                let Arg::Variable { name, min, max } = &node.arg else {
                    return Err("invalid scalar variable".into());
                };
                let (mut min, mut max) = (*min, *max);
                if let Some(previous) = ordered.get(name) {
                    if let Arg::Variable { min: a, max: b, .. } = previous.arg {
                        min = min.max(a);
                        max = max.min(b);
                    }
                }
                if min > max {
                    return Err(format!("incompatible bounds for variable {name}"));
                }
                ordered.insert(name.clone(), UOp::variable(name, min, max));
            }
            let vars = ordered.into_values().collect::<Vec<_>>();
            if !vars.is_empty() {
                let scalar_buffer = device.allocate(vars.len() * 8)?;
                transfer(
                    &mut queue,
                    &mut transfers,
                    &scalar_buffer,
                    HostBuffer::Scalars(i),
                    false,
                );
                bindings.push(scalar_buffer);
            }
            variables.push(vars);
            programs.push(UOp::new(
                Ops::Program,
                vec![
                    plan.sink.clone(),
                    linear,
                    UOp::new(Ops::Source, vec![], Arg::Text(kernel.source.clone())),
                ],
                Arg::Text(kernel.entry.clone()),
            ));
            kernels.push((kernel, bindings));
        }
        match device {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Device::Metal(device) => {
                let commands = kernels
                    .iter()
                    .map(|(kernel, bindings)| {
                        let buffers = bindings
                            .iter()
                            .map(|b| match b {
                                Allocation::Metal(b) => b.clone(),
                                _ => unreachable!(),
                            })
                            .collect();
                        Ok((device.compile(kernel)?, kernel.launch.clone(), buffers))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let command = queue.retain_metal_command(device.command_buffer(&commands)?);
                for (index, (_, bindings)) in kernels.iter().enumerate() {
                    for (argument, buffer) in bindings.iter().enumerate() {
                        let buffer = buffer.retain(&mut queue);
                        queue.metal_patch(
                            command.clone(),
                            UOp::int(index as i64),
                            UOp::int(argument as i64),
                            buffer,
                            UOp::int(0),
                        );
                    }
                }
                queue.metal_submit(
                    command.clone(),
                    HWQueue::runtime_slot(0),
                    HWQueue::runtime_slot(1),
                );
                queue.metal_wait(command);
            }
            #[cfg(feature = "opencl")]
            Device::OpenCl(device) => {
                for (kernel, bindings) in &kernels {
                    let args = bindings
                        .iter()
                        .map(|b| match b {
                            Allocation::OpenCl(b) => {
                                crate::backends::opencl::Argument::Buffer(b.clone())
                            }
                            _ => unreachable!(),
                        })
                        .collect();
                    let program = device.compile(kernel)?;
                    let command =
                        queue.retain_opencl_command(program.command(kernel.launch.clone(), args)?);
                    queue.opencl_launch(command.clone());
                    queue.opencl_finish(command);
                }
            }
        }
        for (i, output) in schedule.outputs.iter().enumerate() {
            transfer(
                &mut queue,
                &mut transfers,
                &allocations[&output.id],
                HostBuffer::Output(i),
                true,
            );
        }
        transfer(&mut queue, &mut transfers, &error, HostBuffer::Error, true);
        Ok(Self {
            inputs,
            schedule,
            programs,
            queue: queue.compile("chibi_gpu_queue")?,
            input_nodes,
            captured,
            variables,
            transfers,
            sequence: 0,
        })
    }
    pub fn kernel_count(&self) -> usize {
        self.programs.len()
    }
    pub fn sources(&self) -> Vec<&str> {
        self.programs
            .iter()
            .filter_map(|p| {
                p.src.iter().find_map(|s| {
                    if let (Ops::Source, Arg::Text(text)) = (s.op, &s.arg) {
                        Some(text.as_str())
                    } else {
                        None
                    }
                })
            })
            .collect()
    }
    pub fn queue_source(&self) -> Option<&str> {
        self.queue.source()
    }
    pub fn run(&mut self, inputs: &[Buffer], vars: &HashMap<String, i64>) -> Result<Vec<Buffer>> {
        if inputs.len() != self.inputs.len() {
            return Err(format!(
                "expected {} input buffers, received {}",
                self.inputs.len(),
                inputs.len()
            ));
        }
        for ((buffer, spec), node) in inputs.iter().zip(&self.inputs).zip(&self.input_nodes) {
            let shape = node.shape()?;
            if buffer.shape != shape || buffer.data.len() != numel(&shape) {
                return Err(format!(
                    "input {} expected {:?}, received {:?}",
                    spec.name, shape, buffer.shape
                ));
            }
        }
        let mut scalars = vec![];
        for variables in &self.variables {
            scalars.push(
                variables
                    .iter()
                    .map(|u| {
                        let Arg::Variable { name, min, max } = &u.arg else {
                            return Err("invalid variable".into());
                        };
                        let value = *vars
                            .get(name)
                            .ok_or_else(|| format!("unbound variable {name}"))?;
                        if value < *min || value > *max {
                            return Err(format!("variable {name}={value} outside {min}..={max}"));
                        }
                        Ok(value)
                    })
                    .collect::<Result<Vec<i64>>>()?,
            );
        }
        let mut outputs = self
            .schedule
            .outputs
            .iter()
            .map(|u| {
                let shape = u.shape()?;
                Buffer::new(shape.clone(), vec![0.; numel(&shape)])
            })
            .collect::<Result<Vec<_>>>()?;
        let mut error = [0f32];
        let next = self
            .sequence
            .checked_add(1)
            .ok_or("GPU timeline overflow")?;
        let mut slots = vec![self.sequence, next];
        for host in &self.transfers {
            let (pointer, bytes) = match host {
                HostBuffer::Input(i) => {
                    (inputs[*i].data.as_ptr() as usize, inputs[*i].data.len() * 4)
                }
                HostBuffer::Captured(i) => (
                    self.captured[*i].data.as_ptr() as usize,
                    self.captured[*i].data.len() * 4,
                ),
                HostBuffer::Scalars(i) => (scalars[*i].as_ptr() as usize, scalars[*i].len() * 8),
                HostBuffer::Output(i) => (
                    outputs[*i].data.as_mut_ptr() as usize,
                    outputs[*i].data.len() * 4,
                ),
                HostBuffer::Error => (error.as_mut_ptr() as usize, 4),
            };
            slots.extend([pointer as i64, bytes as i64]);
        }
        self.sequence = next;
        unsafe { self.queue.run(&mut slots) }?;
        if error[0] != 0. {
            return Err("gather/scatter index must be finite, integral, and within bounds".into());
        }
        Ok(outputs)
    }
}
```

`src/lib.rs`を保存する。

```rust
// file="src/lib.rs" mode="write"
pub type Result<T> = std::result::Result<T, String>;
pub use uop::Axis;
pub use uop::{Arg, DType, Ops, UOp};
pub mod codegen;
pub mod cpu;
pub mod data;
pub mod effects;
pub mod functions;
pub mod pattern;
pub mod runtime;
pub mod scalar;
pub mod schedule;
pub mod tensor;
pub mod uir;
pub mod uop;
pub use runtime::{Executable, InputSpec, compile_many};
pub use tensor::{Buffer, Tensor};

pub mod math_decomp;

pub mod nn;
pub mod optim;
pub fn run_named(
    executable: &Executable,
    bindings: &std::collections::BTreeMap<String, Buffer>,
) -> Result<Vec<Buffer>> {
    let inputs = executable
        .inputs()
        .iter()
        .map(|spec| {
            let buffer = bindings
                .get(&spec.name)
                .ok_or_else(|| format!("missing binding {}", spec.name))?;
            if buffer.shape != spec.shape {
                return Err(format!("binding shape {}", spec.name));
            }
            Ok(buffer.clone())
        })
        .collect::<Result<Vec<_>>>()?;
    executable.run(&inputs, &std::collections::HashMap::new())
}

pub mod kernels;

pub mod gpu;

#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "opencl"))]
pub mod backends;

pub mod renderers;

#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "opencl"))]
pub mod gpu_tensor;

#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "opencl"))]
pub use gpu_tensor::{Backend, GpuExecutable};

pub mod cpu_object;

pub mod cpu_format;

pub mod cpu_memory;

pub mod cpu_relocation;

#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "opencl"))]
pub mod queue;

pub mod optimization;

pub mod parallel;

pub mod compiler_cache;

pub mod symbolic;
```

`src/parallel.rs` の `pub struct ParallelProgram`を置き換える。

```rust
// file="src/parallel.rs" mode="replace" item="pub struct ParallelProgram"
pub struct ParallelProgram {
    object: Arc<Vec<u8>>,
    entry: String,
    buffers: Vec<(usize, Vec<usize>)>,
    output_slot: usize,
    bounds_slot: usize,
    variables: Vec<(usize, String, i64, i64)>,
    extent: i64,
    workers: usize,
}
```

`src/parallel.rs` の `impl ParallelProgram` 内 の `pub fn compile`を置き換える。

```rust
// file="src/parallel.rs" mode="replace" within="impl ParallelProgram" item="pub fn compile"
    pub fn compile(linear: &UOp, axis: usize, workers: usize) -> Result<Self> {
        let linear = crate::optimization::complete(linear)?;
        if workers == 0 || workers > 256 {
            return Err("workers must be in 1..=256".into());
        }
        let range = linear
            .src
            .iter()
            .find(|u| u.op == Ops::Range && matches!(u.arg,Arg::Range { id, .. } if id == axis))
            .ok_or("unknown worker axis")?;
        if !matches!(
            range.arg,
            Arg::Range {
                axis: Axis::Loop,
                ..
            }
        ) || range.src[0] != UOp::int(0)
        {
            return Err("worker axis must be a zero-based output loop".into());
        }
        let Arg::Int(extent) = range.src[1].arg else {
            return Err("worker partition requires a constant output count".into());
        };
        if extent < 0 {
            return Err("negative output count".into());
        }
        let begin = linear.src.iter().position(|u| u == range).unwrap();
        let finish = linear
            .src
            .iter()
            .position(|u| u.op == Ops::End && u.src.last() == Some(range))
            .ok_or("missing worker range END")?;
        if linear.src[..begin]
            .iter()
            .any(|u| matches!(u.op, Ops::Range | Ops::Store | Ops::Call))
            || linear.src[finish + 1..]
                .iter()
                .any(|u| matches!(u.op, Ops::Range | Ops::Store | Ops::Call))
        {
            return Err("worker axis must enclose every kernel effect".into());
        }
        let body = &linear.src[begin + 1..finish];
        let all = linear.toposort();
        if all.iter().any(|u| matches!(u.op, Ops::Call)) {
            return Err("unknown effects prevent parallel execution".into());
        }
        let mut stores = vec![];
        for u in &all {
            if u.op != Ops::Store {
                continue;
            }
            let mut pointer = &u.src[0];
            while pointer.op == Ops::After {
                pointer = &pointer.src[0];
            }
            if pointer.op == Ops::Alloc {
                if !body.iter().any(|node| node == pointer) {
                    return Err("loop-carried accumulator prevents parallel execution".into());
                }
                continue;
            }
            if !matches!(pointer.arg, Arg::None | Arg::DType(crate::DType::F32)) {
                return Err("worker output requires scalar F32 stores".into());
            }
            if pointer.op != Ops::Index || pointer.src[1] != *range {
                return Err("worker output must be indexed directly by the partition axis".into());
            }
            stores.push(pointer.src[0].clone());
        }
        if stores.len() != 1 {
            return Err("worker kernel must have one contiguous output store".into());
        }
        let output = &stores[0];
        for u in &all {
            if u.op == Ops::Load {
                let mut pointer = &u.src[0];
                while matches!(pointer.op, Ops::After | Ops::Index) {
                    pointer = &pointer.src[0];
                }
                if pointer == output {
                    return Err("workers cannot read their output buffer".into());
                }
            }
        }
        let sig = codegen::signature(&linear)?;
        let output_slot = sig
            .buffers
            .iter()
            .find(|(_, u)| u == output)
            .ok_or("output missing from signature")?
            .0;
        let buffers = sig
            .buffers
            .iter()
            .map(|(slot, u)| Ok((*slot, buffer_shape(u)?)))
            .collect::<Result<Vec<_>>>()?;
        if crate::uop::numel(&buffer_shape(output)?) != extent as usize {
            return Err("output shape must match worker axis extent".into());
        }
        let bounds_slot = sig
            .buffers
            .iter()
            .map(|(slot, _)| *slot)
            .max()
            .map_or(0, |slot| slot + 1);
        let bounds = UOp::new(
            Ops::Param,
            vec![],
            Arg::Slot {
                index: bounds_slot,
                shape: vec![2],
            },
        );
        let bound = |index| {
            let pointer = UOp::new(
                Ops::Index,
                vec![bounds.clone(), UOp::int(index), UOp::bool(true)],
                Arg::DType(crate::DType::I64),
            );
            UOp::new(Ops::Load, vec![pointer], Arg::DType(crate::DType::I64))
        };
        let start = bound(0);
        let stop = bound(1);
        let replacement = UOp::new(
            Ops::Range,
            vec![start.clone(), stop.clone()],
            range.arg.clone(),
        );
        fn rewrite(u: &UOp, map: &mut HashMap<u64, UOp>) -> UOp {
            if let Some(v) = map.get(&u.id) {
                return v.clone();
            }
            let sources = u.src.iter().map(|s| rewrite(s, map)).collect();
            let v = UOp::new(u.op, sources, u.arg.clone());
            map.insert(u.id, v.clone());
            v
        }
        let mut map = HashMap::from([(range.id, replacement)]);
        let partitioned = UOp::linear(linear.src.iter().map(|u| rewrite(u, &mut map)).collect());
        codegen::verify_lowered(&partitioned)?;
        let partition_sig = codegen::signature(&partitioned)?;
        for (slot, original) in &sig.buffers {
            let replacement = partition_sig
                .buffers
                .iter()
                .find(|(_, u)| u == original)
                .ok_or("buffer binding changed")?;
            if replacement.0 != *slot {
                return Err("worker partition changed buffer ABI".into());
            }
        }
        let variables = partition_sig
            .variables
            .iter()
            .map(|(slot, u)| {
                let Arg::Variable { name, min, max } = &u.arg else {
                    return Err("invalid worker variable".into());
                };
                Ok((*slot, name.clone(), *min, *max))
            })
            .collect::<Result<Vec<_>>>()?;
        let entry = "chibi_parallel_entry".to_owned();
        let source = codegen::render(&partitioned, &entry)?;
        let object = Arc::new(cpu::compile_object(&source)?);
        Ok(Self {
            object,
            entry,
            buffers,
            output_slot,
            bounds_slot,
            variables,
            extent,
            workers,
        })
    }
```

`src/parallel.rs` の `impl ParallelProgram` 内 の `pub fn run`を置き換える。

```rust
// file="src/parallel.rs" mode="replace" within="impl ParallelProgram" item="pub fn run"
    pub fn run(&self, inputs: &[(usize, Buffer)]) -> Result<WorkerRun> {
        self.run_with_vars(inputs, &HashMap::new())
    }
```

`src/parallel.rs` の `impl ParallelProgram` 内を追記する。

```rust
// file="src/parallel.rs" mode="append" within="impl ParallelProgram"
    pub fn run_with_vars(
        &self,
        inputs: &[(usize, Buffer)],
        vars: &HashMap<String, i64>,
    ) -> Result<WorkerRun> {
        let scalars = self
            .variables
            .iter()
            .map(|(slot, name, min, max)| {
                let value = *vars
                    .get(name)
                    .ok_or_else(|| format!("unbound worker variable {name}"))?;
                if value < *min || value > *max {
                    return Err(format!("worker variable {name} outside bounds"));
                }
                Ok((*slot, value))
            })
            .collect::<Result<Vec<_>>>()?;
        let input_count = inputs.len();
        let inputs = inputs
            .iter()
            .map(|(slot, buffer)| (*slot, buffer.clone()))
            .collect::<HashMap<_, _>>();
        if inputs.len() != input_count
            || inputs.len() != self.buffers.len() - 1
            || inputs.contains_key(&self.output_slot)
        {
            return Err("worker inputs must bind every read buffer exactly once".into());
        }
        for (slot, shape) in &self.buffers {
            if *slot == self.output_slot {
                continue;
            }
            let b = inputs.get(slot).ok_or("missing worker buffer")?;
            if b.shape != *shape || b.data.len() != crate::uop::numel(shape) {
                return Err(format!("worker buffer {slot} shape mismatch"));
            }
        }
        let count = self.worker_count();
        let output_shape = self
            .buffers
            .iter()
            .find(|(slot, _)| *slot == self.output_slot)
            .unwrap()
            .1
            .clone();
        let mut output = Buffer::new(output_shape, vec![0.; self.extent as usize])?;
        if count == 0 {
            return Ok(WorkerRun {
                output,
                thread_ids: vec![],
                partitions: vec![],
            });
        }
        let inputs = Arc::new(inputs);
        let mut handles = vec![];
        for worker in 0..count {
            let lo = self.extent as usize * worker / count;
            let hi = self.extent as usize * (worker + 1) / count;
            let object = self.object.clone();
            let entry = self.entry.clone();
            let input = inputs.clone();
            let output_slot = self.output_slot;
            let bounds_slot = self.bounds_slot;
            let extent = self.extent as usize;
            let scalars = scalars.clone();
            let arg_count = self
                .buffers
                .iter()
                .map(|b| b.0)
                .chain([bounds_slot])
                .chain(self.variables.iter().map(|v| v.0))
                .max()
                .unwrap()
                + 1;
            handles.push(std::thread::spawn(move || -> Result<_> {
                let native = cpu::CpuProgram::load(&object, &entry)?;
                let mut output = vec![0f32; extent];
                let mut packed = vec![0u8; arg_count * 8];
                for (&slot, b) in input.iter() {
                    packed[slot * 8..slot * 8 + 8]
                        .copy_from_slice(&(b.data.as_ptr() as u64).to_ne_bytes());
                }
                packed[output_slot * 8..output_slot * 8 + 8]
                    .copy_from_slice(&(output.as_mut_ptr() as u64).to_ne_bytes());
                let bounds = [lo as i64, hi as i64];
                packed[bounds_slot * 8..bounds_slot * 8 + 8]
                    .copy_from_slice(&(bounds.as_ptr() as u64).to_ne_bytes());
                for (slot, value) in scalars {
                    packed[slot * 8..slot * 8 + 8].copy_from_slice(&value.to_ne_bytes());
                }
                unsafe { native.run(&mut packed) }?;
                Ok((lo..hi, output, std::thread::current().id()))
            }));
        }
        let mut partitions = vec![];
        let mut thread_ids = vec![];
        let mut error = None;
        for handle in handles {
            match handle
                .join()
                .map_err(|_| "CPU worker panicked".to_owned())
                .and_then(|result| result)
            {
                Ok((range, data, id)) => {
                    output.data[range.clone()].copy_from_slice(&data[range.clone()]);
                    partitions.push(range);
                    thread_ids.push(id);
                }
                Err(message) => {
                    error.get_or_insert(message);
                }
            }
        }
        if let Some(message) = error {
            return Err(message);
        }
        Ok(WorkerRun {
            output,
            thread_ids,
            partitions,
        })
    }
```

`src/renderers.rs` の `fn ident`を置き換える。

```rust
// file="src/renderers.rs" mode="replace" item="fn ident"
fn ident(name: &str) -> Result<(), String> {
    if name.is_empty()
        || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        || name.as_bytes()[0].is_ascii_digit()
    {
        Err("invalid GPU identifier".into())
    } else {
        Ok(())
    }
}
```

`src/renderers.rs` の `pub fn render`を置き換える。

```rust
// file="src/renderers.rs" mode="replace" item="pub fn render"
pub fn render(
    linear: &UOp,
    entry: &str,
    dialect: Dialect,
    launch: Launch,
) -> Result<Kernel, String> {
    ident(entry)?;
    if linear.op != Ops::Linear {
        return Err("GPU renderer requires lowered LINEAR UOps".into());
    }
    let mut params = BTreeMap::new();
    let mut vars = BTreeMap::new();
    let mut values = HashMap::<u64, String>::new();
    let mut ranges = HashSet::new();
    let mut body = String::new();
    let mut declarations = String::new();
    let mut shared_bytes = 0usize;
    let core_signature = crate::codegen::signature(linear)?;
    let buffer_slots = core_signature
        .buffers
        .iter()
        .map(|(slot, u)| (u.id, *slot))
        .collect::<HashMap<_, _>>();
    for (slot, _) in &core_signature.buffers {
        params.insert(*slot, format!("p{slot}"));
    }
    for u in &linear.src {
        if let Arg::Slot { index, .. } = u.arg {
            if u.op == Ops::Param {
                params.insert(index, format!("p{index}"));
            }
        }
    }
    for u in &linear.src {
        if let Arg::Variable { ref name, .. } = u.arg {
            ident(name)?;
            vars.insert(name.clone(), name.clone());
        }
    }
    let nparams = params.keys().copied().max().map_or(0, |slot| slot + 1);
    let mut source = match dialect {
        Dialect::Metal => "#include <metal_stdlib>\nusing namespace metal;\n".to_owned(),
        Dialect::OpenCl => String::new(),
    };
    source
        .push_str(
            "inline long chibi_div(long a,long b){long q=a/b,r=a%b;return q-((r!=0)&&((r<0)!=(b<0)));}\ninline long chibi_mod(long a,long b){long r=a%b;return r+(((r!=0)&&((r<0)!=(b<0)))?b:0);}\n",
        );
    {}
    let mut signature = params
        .iter()
        .map(|(idx, name)| match dialect {
            Dialect::Metal => format!("device float* {name} [[buffer({idx})]]"),
            Dialect::OpenCl => format!("__global float* {name}"),
        })
        .collect::<Vec<_>>();
    if !vars.is_empty() {
        signature.push(match dialect {
            Dialect::Metal => format!("device const long* scalars [[buffer({nparams})]]"),
            Dialect::OpenCl => "__global const long* scalars".into(),
        });
    }
    for (idx, (name, _)) in vars.iter().enumerate() {
        body.push_str(&format!("long {name}=scalars[{idx}];\n"));
    }
    if dialect == Dialect::Metal {
        signature.extend([
            "uint3 gid [[thread_position_in_grid]]".into(),
            "uint3 lid [[thread_position_in_threadgroup]]".into(),
            "uint3 group [[threadgroup_position_in_grid]]".into(),
        ]);
    }
    for u in &linear.src {
        let name = format!("v{}", u.id);
        let get = |i: usize| -> Result<String, String> {
            let s = u
                .src
                .get(i)
                .ok_or_else(|| format!("missing {:?} source {i}", u.op))?;
            values
                .get(&s.id)
                .cloned()
                .ok_or_else(|| format!("GPU UOp {:?} source {} not linearized", u.op, s.id))
        };
        let expr = match u.op {
            Ops::Const => match u.arg {
                Arg::Float(bits) => {
                    let bits = (bits.value() as f32).to_bits();
                    match dialect {
                        Dialect::Metal => format!("as_type<float>((uint){bits})"),
                        Dialect::OpenCl => format!("as_float((uint){bits})"),
                    }
                },
                Arg::I32(x) => format!("{x}"),
                Arg::Int(x) => format!("{x}L"),
                Arg::Bool(b) => if b { "true" } else { "false" }.into(),
                _ => return Err("unsupported GPU constant".into()),
            },
            Ops::Alloc => {
                if let Arg::Allocation { dtype, size, .. } = u.arg {
                    let address = if size > 1 {
                        shared_bytes = shared_bytes
                            .checked_add(
                                size * match dtype {
                                    DType::I64 => 8,
                                    _ => 4,
                                },
                            )
                            .ok_or("GPU shared memory overflow")?;
                        match dialect {
                            Dialect::Metal => "threadgroup ",
                            Dialect::OpenCl => "__local ",
                        }
                    } else {
                        ""
                    };
                    declarations.push_str(&format!("{address}{} {name}[{size}];\n", ty(dtype)));
                    values.insert(u.id, name);
                    continue;
                } else {
                    return Err("GPU allocation missing layout".into());
                }
            }
            Ops::Param | Ops::Buffer => {
                format!(
                    "p{}",
                    buffer_slots.get(&u.id).ok_or("GPU buffer missing slot")?
                )
            }
            Ops::Variable => {
                if let Arg::Variable { ref name, .. } = u.arg {
                    name.clone()
                } else {
                    return Err("GPU variable missing name".into());
                }
            }
            Ops::Bind => get(1)?,
            Ops::Index => {
                if let Arg::DType(DType::F32x4) = u.arg {
                    let address = if dialect == Dialect::Metal {
                        "device "
                    } else {
                        "__global "
                    };
                    format!("((({address}float4*){})+({}))", get(0)?, get(1)?)
                } else {
                    format!("({}+({}))", get(0)?, get(1)?)
                }
            }
            Ops::Load => {
                {}
                let index = &u.src[0];
                let ptr = get(0)?;
                let cast = match u.arg {
                    Arg::DType(dt) => format!("{}*", ty(dt)),
                    _ => "float*".into(),
                };
                let addr = if dialect == Dialect::Metal {
                    "device "
                } else {
                    "__global "
                };
                let typed = if matches!(u.arg, Arg::DType(_)) {
                    format!("(({addr}{cast}){ptr})")
                } else {
                    ptr
                };
                if index.op == Ops::Index && index.src.len() > 2 && index.src[2] != UOp::bool(true)
                {
                    let valid = values
                        .get(&index.src[2].id)
                        .ok_or("missing GPU load mask")?;
                    format!("({valid}?*{typed}:0)")
                } else {
                    format!("*{typed}")
                }
            }
            Ops::Store => {
                let ptr = get(0)?;
                let value = get(1)?;
                {}
                let index = &u.src[0];
                if index.op == Ops::Index && index.src.len() > 2 && index.src[2] != UOp::bool(true)
                {
                    let valid = values
                        .get(&index.src[2].id)
                        .ok_or("missing GPU store mask")?;
                    body.push_str(&format!("if({valid}) *{ptr}={value};\n"));
                } else {
                    body.push_str(&format!("*{ptr}={value};\n"));
                }
                String::new()
            }
            Ops::Range => {
                let (id, axis) = if let Arg::Range { id, axis } = u.arg {
                    (id, axis)
                } else {
                    return Err("range missing axis".into());
                };
                match axis {
                    Axis::Global | Axis::Local => {
                        let dim = id % 3;
                        let hw = match (dialect, axis) {
                            (Dialect::Metal, Axis::Global) => format!("gid[{}]", dim),
                            (Dialect::Metal, _) => format!("lid[{}]", dim),
                            (Dialect::OpenCl, Axis::Global) => {
                                format!("get_global_id({dim})")
                            }
                            _ => format!("get_local_id({dim})"),
                        };
                        declarations.push_str(&format!("long {name};\n"));
                        body.push_str(&format!("{name}={hw};\n"));
                    }
                    _ => {
                        declarations.push_str(&format!("long {name};\n"));
                        body.push_str(&format!(
                            "for({name}={};{name}<{};{name}++){{\n",
                            get(0)?,
                            get(1)?
                        ));
                        ranges.insert(u.id);
                    }
                }
                values.insert(u.id, name);
                continue;
            }
            Ops::End => {
                if u.src.last().is_some_and(|r| ranges.contains(&r.id)) {
                    body.push_str("}\n");
                }
                String::new()
            }
            Ops::Special => {
                let text = if let Arg::Text(ref t) = u.arg {
                    t.as_str()
                } else {
                    return Err("SPECIAL requires gid/lid/group and dimension".into());
                };
                let (kind, dim) = text.split_at(text.len().saturating_sub(1));
                let dim = dim
                    .parse::<usize>()
                    .map_err(|_| "invalid SPECIAL dimension")?;
                if dim > 2 {
                    return Err("SPECIAL dimension outside grid".into());
                }
                match (dialect, kind) {
                    (Dialect::Metal, "gid") => format!("gid[{dim}]"),
                    (Dialect::Metal, "lid") => format!("lid[{dim}]"),
                    (Dialect::Metal, "group") => format!("group[{dim}]"),
                    (Dialect::OpenCl, "gid") => format!("get_global_id({dim})"),
                    (Dialect::OpenCl, "lid") => format!("get_local_id({dim})"),
                    (Dialect::OpenCl, "group") => format!("get_group_id({dim})"),
                    _ => return Err("invalid SPECIAL builtin".into()),
                }
            }
            Ops::Barrier => {
                body.push_str(match dialect {
                    Dialect::Metal => "threadgroup_barrier(mem_flags::mem_threadgroup);\n",
                    Dialect::OpenCl => "barrier(CLK_LOCAL_MEM_FENCE);\n",
                });
                String::new()
            }
            Ops::Div if matches!(u.dtype(), DType::I32 | DType::I64) => {
                format!("chibi_div({},{})", get(0)?, get(1)?)
            }
            Ops::Mod => format!("chibi_mod({},{})", get(0)?, get(1)?),
            Ops::Add | Ops::Sub | Ops::Mul | Ops::Div | Ops::Lt | Ops::Eq | Ops::And | Ops::Or => {
                let op = match u.op {
                    Ops::Add => "+",
                    Ops::Sub => "-",
                    Ops::Mul => "*",
                    Ops::Div => "/",
                    Ops::Mod => "%",
                    Ops::Lt => "<",
                    Ops::Eq => "==",
                    Ops::And => "&&",
                    _ => "||",
                };
                format!("({} {op} {})", get(0)?, get(1)?)
            }
            Ops::Max => format!("max({},{})", get(0)?, get(1)?),
            Ops::Neg => format!("(-{})", get(0)?),
            Ops::Exp | Ops::Log | Ops::Sin | Ops::Cos | Ops::Sqrt | Ops::Exp2 | Ops::Log2 => {
                let fun = match u.op {
                    Ops::Exp => "exp",
                    Ops::Log => "log",
                    Ops::Sin => "sin",
                    Ops::Cos => "cos",
                    Ops::Sqrt => "sqrt",
                    Ops::Exp2 => "exp2",
                    _ => "log2",
                };
                format!("{fun}({})", get(0)?)
            }
            Ops::Where => format!("({}?{}:{})", get(0)?, get(1)?, get(2)?),
            Ops::Cast => format!("(({}){})", ty(u.dtype()), get(0)?),
            Ops::After => get(0)?,
            Ops::Sink | Ops::Linear => String::new(),
            _ => return Err(format!("unsupported lowered GPU UOp {:?}", u.op)),
        };
        if !expr.is_empty() {
            if matches!(u.op, Ops::Index | Ops::Param | Ops::Buffer | Ops::After) {
                values.insert(u.id, expr);
            } else {
                declarations.push_str(&format!("{} {name};\n", ty(u.dtype())));
                body.push_str(&format!("{name}={expr};\n"));
                values.insert(u.id, name);
            }
        }
    }
    let mut declared = HashSet::new();
    let declarations = declarations
        .lines()
        .filter(|line| declared.insert(*line))
        .collect::<Vec<_>>()
        .join("\n");
    source.push_str(&format!(
        "{} void {entry}({}){{\n{declarations}{body}}}\n",
        match dialect {
            Dialect::Metal => "kernel",
            Dialect::OpenCl => "__kernel",
        },
        signature.join(",")
    ));
    Ok(Kernel {
        source,
        entry: entry.into(),
        launch,
        shared_bytes,
    })
}
```

`src/runtime.rs` の `pub struct Executable`を置き換える。

```rust
// file="src/runtime.rs" mode="replace" item="pub struct Executable"
pub struct Executable {
    inputs: Vec<InputSpec>,
    schedule: Schedule,
    programs: Vec<UOp>,

    native: Vec<crate::cpu::CpuProgram>,
    input_nodes: Vec<UOp>,
    captured: HashMap<u64, Buffer>,
}
```

`src/runtime.rs` の `pub unsafe fn compile_many_with_transform`を置き換える。

```rust
// file="src/runtime.rs" mode="replace" item="pub unsafe fn compile_many_with_transform"
pub unsafe fn compile_many_with_transform(
    outputs: &[Tensor],
    transform: impl Fn(&UOp) -> Result<UOp>,
) -> Result<Executable> {
    let debug = std::env::var_os("CHIBI_DEBUG").is_some();
    if debug {
        eprintln!("schedule starts");
    }
    let schedule = create_schedule(&outputs.iter().map(|x| x.uop.clone()).collect::<Vec<_>>())?;
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
            captured.insert(
                u.id,
                Buffer::new(
                    shape.clone(),
                    bits.iter().map(|&b| f32::from_bits(b)).collect(),
                )?,
            );
        } else {
            input_nodes.push(u.clone());
            inputs.push(InputSpec {
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
            kernel.sink.clone(),
            linear,
            UOp::new(Ops::Source, vec![], Arg::Text(source.clone())),
        ];

        let stages = {
            if debug {
                eprintln!("kernel {i} compile {} source bytes", source.len());
            }
            let compiled = crate::compiler_cache::compile_cached(&source, &entry)?;
            let mut stages = stages;
            stages.push(UOp::new(
                Ops::Binary,
                vec![],
                Arg::Bytes(compiled.object().to_vec()),
            ));
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
```

`src/runtime.rs` の `impl Executable` 内 の `pub fn run`を置き換える。

```rust
// file="src/runtime.rs" mode="replace" within="impl Executable" item="pub fn run"
    pub fn run(&self, inputs: &[Buffer], vars: &HashMap<String, i64>) -> Result<Vec<Buffer>> {
        if inputs.len() != self.inputs.len() {
            return Err(format!(
                "expected {} input buffers, received {}",
                self.inputs.len(),
                inputs.len()
            ));
        }
        for ((buffer, spec), node) in inputs.iter().zip(&self.inputs).zip(&self.input_nodes) {
            let expected = node.shape()?;
            if buffer.shape != expected || buffer.data.len() != numel(&expected) {
                return Err(format!(
                    "input {} expected {:?}, received {:?} with {} elements",
                    spec.name,
                    expected,
                    buffer.shape,
                    buffer.data.len()
                ));
            }
        }
        let mut external: HashMap<u64, &Buffer> =
            self.captured.iter().map(|(&id, b)| (id, b)).collect();
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
                let count = sig
                    .buffers
                    .iter()
                    .map(|x| x.0)
                    .chain(sig.variables.iter().map(|x| x.0))
                    .max()
                    .map_or(0, |x| x + 1);
                let mut packed = vec![0u8; count * 8];
                let mut validation_status = 0f32;
                for (slot, u) in sig.buffers {
                    let pointer = if matches!(&u.arg, Arg::Buffer {name, data:Some(_), ..} if name=="__chibi_error")
                    {
                        &mut validation_status as *mut f32 as usize
                    } else if let Some(b) = external.get(&u.id) {
                        b.data.as_ptr() as usize
                    } else {
                        let loc = locations.get(&u.id).ok_or("missing runtime allocation")?;
                        slots[*loc].as_mut_ptr() as usize
                    };
                    packed[slot * 8..slot * 8 + 8].copy_from_slice(&(pointer as u64).to_ne_bytes());
                }
                for (slot, u) in sig.variables {
                    let Arg::Variable { name, min, max } = &u.arg else {
                        return Err("invalid variable".into());
                    };
                    let value = vars
                        .get(name)
                        .ok_or_else(|| format!("unbound variable {name}"))?;
                    if value < min || value > max {
                        return Err(format!("variable {name}={value} outside {min}..={max}"));
                    }
                    packed[slot * 8..slot * 8 + 8].copy_from_slice(&value.to_ne_bytes());
                }
                unsafe { self.native[i].run(&mut packed) }?;
                if validation_status != 0. {
                    return Err(
                        "gather/scatter index must be finite, integral, and within bounds".into(),
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
```

`src/scalar.rs`を保存する。

```rust
// file="src/scalar.rs" mode="write"
//! A scalar oracle and checked native compilation over the common UOp DAG.
use crate::{Arg, DType, Ops, Result, UOp};
use std::collections::HashMap;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Value {
    Bool(bool),
    I32(i32),
    I64(i64),
    F32Bits(u32),
}
impl Value {
    pub fn f32(v: f32) -> Self {
        Self::F32Bits(v.to_bits())
    }
    pub fn dtype(self) -> DType {
        match self {
            Self::Bool(_) => DType::Bool,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::F32Bits(_) => DType::F32,
        }
    }
    pub fn number(self) -> f64 {
        match self {
            Self::Bool(v) => v as u8 as f64,
            Self::I32(v) => v as f64,
            Self::I64(v) => v as f64,
            Self::F32Bits(v) => f32::from_bits(v) as f64,
        }
    }
    pub fn uop(self) -> UOp {
        match self {
            Self::Bool(v) => UOp::bool(v),
            Self::I32(v) => UOp::i32(v),
            Self::I64(v) => UOp::int(v),
            Self::F32Bits(v) => UOp::f32(f32::from_bits(v)),
        }
    }
    pub fn from_constant(u: &UOp) -> Option<Self> {
        if u.op != Ops::Const {
            return None;
        }
        Some(match u.arg {
            Arg::Bool(v) => Self::Bool(v),
            Arg::I32(v) => Self::I32(v),
            Arg::Int(v) => Self::I64(v),
            Arg::Float(v) => Self::F32Bits((v.value() as f32).to_bits()),
            _ => return None,
        })
    }
    fn pack(self) -> u64 {
        match self {
            Self::Bool(v) => v as u64,
            Self::I32(v) => v as u32 as u64,
            Self::I64(v) => v as u64,
            Self::F32Bits(v) => v as u64,
        }
    }
    fn unpack(dt: DType, v: u64) -> Result<Self> {
        Ok(match dt {
            DType::Bool => Self::Bool(v != 0),
            DType::I32 => Self::I32(v as i32),
            DType::I64 => Self::I64(v as i64),
            DType::F32 => Self::F32Bits(v as u32),
            _ => return Err("not a scalar dtype".into()),
        })
    }
}
fn scalar_type(dt: DType) -> bool {
    matches!(dt, DType::Bool | DType::I32 | DType::I64 | DType::F32)
}
pub fn verify_node(u: &UOp) -> Result<()> {
    if u.op == Ops::Const {
        return Value::from_constant(u)
            .filter(|_| u.src.is_empty())
            .map(|_| ())
            .ok_or("invalid scalar constant".into());
    }
    if let (Ops::Param, Arg::ScalarParam { dtype, .. }) = (u.op, &u.arg) {
        return if scalar_type(*dtype) && u.src.is_empty() {
            Ok(())
        } else {
            Err("invalid scalar parameter".into())
        };
    }
    let types: Vec<_> = u.src.iter().map(UOp::dtype).collect();
    let first = *types.first().ok_or("missing scalar operand")?;
    let unary = crate::uop::is_unary(u.op);
    let n = if u.op == Ops::Where {
        3
    } else if unary {
        1
    } else if crate::uop::is_binary(u.op) {
        2
    } else {
        return Err(format!("unsupported scalar {:?}", u.op));
    };
    if types.len() != n || !types.iter().all(|t| scalar_type(*t)) {
        return Err("scalar arity or dtype mismatch".into());
    }
    match u.op {
        Ops::Cast => {
            if !matches!(u.arg, Arg::DType(dt) if scalar_type(dt)) {
                return Err("invalid CAST target".into());
            }
        }
        Ops::Bitcast => {
            if !matches!(
                (&u.arg, first),
                (Arg::DType(DType::F32 | DType::I32), DType::I32 | DType::F32)
            ) {
                return Err("BITCAST requires equal 32-bit widths".into());
            }
        }
        Ops::Where => {
            if first != DType::Bool || types[1] != types[2] {
                return Err("WHERE dtype mismatch".into());
            }
        }
        Ops::And | Ops::Or | Ops::Xor => {
            if !matches!(first, DType::Bool | DType::I32 | DType::I64) || types[1] != first {
                return Err("bitwise dtype mismatch".into());
            }
        }
        Ops::Shl | Ops::Shr | Ops::Mod => {
            if !matches!(first, DType::I32 | DType::I64) || types[1] != first {
                return Err("integer dtype mismatch".into());
            }
        }
        Ops::Eq | Ops::Lt => {
            if types[1] != first {
                return Err("comparison dtype mismatch".into());
            }
        }
        Ops::Exp | Ops::Log | Ops::Sin | Ops::Cos | Ops::Exp2 | Ops::Log2 | Ops::Sqrt => {
            if first != DType::F32 {
                return Err("transcendental requires F32".into());
            }
        }
        _ => {
            if first == DType::Bool || types.iter().any(|t| *t != first) {
                return Err("numeric dtype mismatch".into());
            }
        }
    }
    if u.op != Ops::Cast && u.op != Ops::Bitcast && u.arg != Arg::None {
        return Err("unexpected scalar argument".into());
    }
    Ok(())
}
pub fn verify(root: &UOp) -> Result<Vec<DType>> {
    let mut params = HashMap::new();
    for u in root.toposort() {
        verify_node(&u)?;
        if let Arg::ScalarParam { index, dtype } = u.arg {
            if params.insert(index, dtype).is_some_and(|old| old != dtype) {
                return Err("inconsistent parameter dtype".into());
            }
        }
        if u.src.iter().all(|s| s.op == Ops::Const) && !u.src.is_empty() {
            let v = u
                .src
                .iter()
                .map(|s| Value::from_constant(s).unwrap())
                .collect::<Vec<_>>();
            eval_alu(u.op, &u.arg, &v)?;
        }
    }
    (0..params.len())
        .map(|i| {
            params
                .get(&i)
                .copied()
                .ok_or("non-dense scalar parameter slots".into())
        })
        .collect()
}
fn inputs_valid(types: &[DType], inputs: &[Value]) -> Result<()> {
    if types.len() != inputs.len() || types.iter().zip(inputs).any(|(t, v)| *t != v.dtype()) {
        return Err("scalar input signature mismatch".into());
    }
    Ok(())
}
pub fn evaluate(root: &UOp, inputs: &[Value]) -> Result<Value> {
    let root = crate::functions::inline_calls(root)?;
    let root = crate::symbolic::simplify(&root)?;
    inputs_valid(&verify(&root)?, inputs)?;
    let mut values = HashMap::new();
    for u in root.toposort() {
        let value = if let Some(v) = Value::from_constant(&u) {
            v
        } else if let Arg::ScalarParam { index, .. } = u.arg {
            inputs[index]
        } else {
            eval_alu(
                u.op,
                &u.arg,
                &u.src.iter().map(|s| values[&s.id]).collect::<Vec<_>>(),
            )?
        };
        values.insert(u.id, value);
    }
    Ok(values[&root.id])
}
pub fn eval_alu(op: Ops, arg: &Arg, values: &[Value]) -> Result<Value> {
    let sources = values.iter().map(|v| v.uop()).collect();
    let node = UOp::new(op, sources, arg.clone());
    verify_node(&node)?;
    let a = values[0];
    if op == Ops::Where {
        return Ok(values[if a == Value::Bool(true) { 1 } else { 2 }]);
    }
    if matches!(op, Ops::Cast | Ops::Bitcast) && *arg == Arg::DType(a.dtype()) {
        return Ok(a);
    }
    if op == Ops::Bitcast {
        return Ok(match a {
            Value::I32(v) => Value::F32Bits(v as u32),
            Value::F32Bits(v) => Value::I32(v as i32),
            _ => unreachable!(),
        });
    }
    if op == Ops::Cast {
        let Arg::DType(dt) = arg else { unreachable!() };
        return Ok(match dt {
            DType::Bool => Value::Bool(a.number() != 0.),
            DType::F32 => Value::f32(a.number() as f32),
            DType::I32 => match a {
                Value::F32Bits(v) => {
                    let x = f32::from_bits(v);
                    if !x.is_finite() || !(-2147483648.0..2147483648.0).contains(&(x as f64)) {
                        return Err("unsafe F32 to I32 CAST".into());
                    }
                    Value::I32(x as i32)
                }
                Value::I64(v) => Value::I32(v as i32),
                _ => Value::I32(a.number() as i32),
            },
            DType::I64 => match a {
                Value::F32Bits(v) => {
                    let x = f32::from_bits(v);
                    if !x.is_finite()
                        || !(-9223372036854775808.0..9223372036854775808.0).contains(&(x as f64))
                    {
                        return Err("unsafe F32 to I64 CAST".into());
                    }
                    Value::I64(x as i64)
                }
                Value::I64(v) => Value::I64(v),
                _ => Value::I64(a.number() as i64),
            },
            _ => unreachable!(),
        });
    }
    let b = values.get(1).copied();
    if matches!(op, Ops::Eq | Ops::Lt) {
        let b = b.unwrap();
        let cmp = match (a, b) {
            (Value::I64(a), Value::I64(b)) => {
                if op == Ops::Eq {
                    a == b
                } else {
                    a < b
                }
            }
            _ => {
                if op == Ops::Eq {
                    a.number() == b.number()
                } else {
                    a.number() < b.number()
                }
            }
        };
        return Ok(Value::Bool(cmp));
    }
    Ok(match a {
        Value::Bool(a) => {
            let Value::Bool(b) = b.unwrap() else {
                unreachable!()
            };
            Value::Bool(match op {
                Ops::And => a & b,
                Ops::Or => a | b,
                Ops::Xor => a ^ b,
                _ => unreachable!(),
            })
        }
        Value::I32(a) => {
            let b = match b {
                Some(Value::I32(v)) => v,
                _ => 0,
            };
            if matches!(op, Ops::Shl | Ops::Shr) && !(0..32).contains(&b) {
                return Err("unsafe I32 shift count".into());
            }
            if matches!(op, Ops::Div | Ops::Mod) && (b == 0 || (a == i32::MIN && b == -1)) {
                return Err("unsafe I32 division".into());
            }
            Value::I32(match op {
                Ops::Add => a.wrapping_add(b),
                Ops::Sub => a.wrapping_sub(b),
                Ops::Mul => a.wrapping_mul(b),
                Ops::Neg => a.wrapping_neg(),
                Ops::And => a & b,
                Ops::Or => a | b,
                Ops::Xor => a ^ b,
                Ops::Shl => a.wrapping_shl(b as u32),
                Ops::Shr => a >> b,
                Ops::Div => crate::symbolic::floor_div(a as i64, b as i64).unwrap() as i32,
                Ops::Mod => crate::symbolic::floor_mod(a as i64, b as i64).unwrap() as i32,
                Ops::Max => a.max(b),
                _ => return Err("invalid I32 operation".into()),
            })
        }
        Value::I64(a) => {
            let b = match b {
                Some(Value::I64(v)) => v,
                _ => 0,
            };
            if matches!(op, Ops::Shl | Ops::Shr) && !(0..64).contains(&b) {
                return Err("unsafe I64 shift count".into());
            }
            Value::I64(
                match op {
                    Ops::Add => a.checked_add(b),
                    Ops::Sub => a.checked_sub(b),
                    Ops::Mul => a.checked_mul(b),
                    Ops::Neg => a.checked_neg(),
                    Ops::And => Some(a & b),
                    Ops::Or => Some(a | b),
                    Ops::Xor => Some(a ^ b),
                    Ops::Shl => Some(a.wrapping_shl(b as u32)),
                    Ops::Shr => Some(a >> b),
                    Ops::Div => crate::symbolic::floor_div(a, b),
                    Ops::Mod => crate::symbolic::floor_mod(a, b),
                    Ops::Max => Some(a.max(b)),
                    _ => None,
                }
                .ok_or("invalid or overflowing I64 operation")?,
            )
        }
        Value::F32Bits(a) => {
            let a = f32::from_bits(a);
            let b = match b {
                Some(Value::F32Bits(v)) => f32::from_bits(v),
                _ => 0.,
            };
            Value::f32(match op {
                Ops::Add => a + b,
                Ops::Sub => a - b,
                Ops::Mul => a * b,
                Ops::Div => a / b,
                Ops::Neg => -a,
                Ops::Max => {
                    if a.is_nan() || b.is_nan() {
                        f32::NAN
                    } else if a == 0. && b == 0. {
                        0.
                    } else {
                        a.max(b)
                    }
                }
                Ops::Exp => a.exp(),
                Ops::Log => a.ln(),
                Ops::Sin => a.sin(),
                Ops::Cos => a.cos(),
                Ops::Exp2 => a.exp2(),
                Ops::Log2 => a.log2(),
                Ops::Sqrt => a.sqrt(),
                _ => return Err("invalid F32 operation".into()),
            })
        }
    })
}
pub struct CompiledScalar {
    pub source: String,
    program: crate::cpu::CpuProgram,
    types: Vec<DType>,
    output: DType,
}
pub fn compile(root: &UOp) -> Result<CompiledScalar> {
    compile_with_guards(root, &[])
}
pub fn compile_with_guards(root: &UOp, guards: &[UOp]) -> Result<CompiledScalar> {
    let root = crate::functions::inline_calls(root)?;
    let root = crate::symbolic::simplify(&root)?;
    let types = verify(&root)?;
    for guard in guards {
        if guard.dtype() != DType::Bool {
            return Err("guard must be Bool".into());
        }
        for u in guard.toposort() {
            verify_node(&u)?;
            if let Arg::ScalarParam { index, dtype } = u.arg {
                if types.get(index) != Some(&dtype) {
                    return Err("guard parameter signature mismatch".into());
                }
            }
        }
    }
    let all = UOp::sink(
        guards
            .iter()
            .cloned()
            .chain(std::iter::once(root.clone()))
            .collect(),
    );
    let mut mapped: HashMap<u64, UOp> = HashMap::new();
    for u in all.toposort().into_iter().filter(|u| u.op != Ops::Sink) {
        let replacement = if let Arg::ScalarParam { index, dtype } = u.arg {
            let ptr = UOp::new(
                Ops::Param,
                vec![],
                Arg::Slot {
                    index: index + 1,
                    shape: vec![1],
                },
            );
            UOp::new(Ops::Load, vec![ptr], Arg::DType(dtype))
        } else {
            u.replace(u.src.iter().map(|s| mapped[&s.id].clone()).collect())
        };
        mapped.insert(u.id, replacement);
    }
    let ptr = UOp::new(
        Ops::Param,
        vec![],
        Arg::Slot {
            index: 0,
            shape: vec![1],
        },
    );
    let store = UOp::new(
        Ops::Store,
        vec![ptr, mapped[&root.id].clone()],
        Arg::DType(root.dtype()),
    );
    let status = UOp::new(
        Ops::Param,
        vec![],
        Arg::Slot {
            index: types.len() + 1,
            shape: vec![1],
        },
    );
    let mut stores: Vec<_> = guards
        .iter()
        .map(|g| {
            UOp::new(
                Ops::Store,
                vec![status.clone(), mapped[&g.id].clone()],
                Arg::DType(DType::Bool),
            )
        })
        .collect();
    stores.push(store);
    let linear = crate::codegen::linearize(&UOp::sink(stores))?;
    let source = crate::codegen::render_checked(&linear, "chibi_scalar", types.len() + 1)?;
    let program = crate::cpu::CpuProgram::compile(&source, "chibi_scalar")?;
    Ok(CompiledScalar {
        source,
        program,
        types,
        output: root.dtype(),
    })
}
impl CompiledScalar {
    pub fn run(&self, inputs: &[Value]) -> Result<Value> {
        inputs_valid(&self.types, inputs)?;
        let words: Vec<_> = inputs.iter().map(|v| v.pack()).collect();
        let mut output = 0u64;
        let mut status = 0i32;
        let mut args = vec![(&mut output as *mut u64) as u64];
        args.extend(words.iter().map(|w| (w as *const u64) as u64));
        args.push((&mut status as *mut i32) as u64);
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(args.as_mut_ptr().cast::<u8>(), args.len() * 8)
        };
        unsafe { self.program.run(bytes) }?;
        if status != 0 {
            return Err("unsafe native scalar operation".into());
        }
        Value::unpack(self.output, output)
    }
}
```

`src/symbolic.rs`を保存する。

```rust
// file="src/symbolic.rs" mode="write"
use crate::{
    Result,
    pattern::{PatternMatcher, UPat, graph_rewrite},
    uop::{Arg, DType, Ops, UOp, is_binary, is_unary},
};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bounds {
    pub min: i64,
    pub max: i64,
}
impl UOp {
    pub fn bounds(&self) -> Option<Bounds> {
        self.bounds_cache
            .get_or_init(|| self.infer_bounds().map(|b| (b.min, b.max)))
            .map(|(min, max)| Bounds { min, max })
    }
    fn infer_bounds(&self) -> Option<Bounds> {
        let b = |i: usize| self.src.get(i)?.bounds();
        match (&self.op, &self.arg) {
            (Ops::Const, Arg::Int(v)) => Some(Bounds { min: *v, max: *v }),
            (Ops::Const, Arg::I32(v)) => Some(Bounds {
                min: *v as i64,
                max: *v as i64,
            }),
            (Ops::Const, Arg::Bool(v)) => Some(Bounds {
                min: *v as i64,
                max: *v as i64,
            }),
            (Ops::Variable, Arg::Variable { min, max, .. }) => Some(Bounds {
                min: *min,
                max: *max,
            }),
            (Ops::Bind, _) => b(1),
            (Ops::Range, _) => Some(Bounds {
                min: b(0)?.min,
                max: b(1)?.max.checked_sub(1)?,
            }),
            (Ops::Lt | Ops::Eq, _) => Some(Bounds { min: 0, max: 1 }),
            (Ops::Cast, Arg::DType(DType::I64)) if self.src.first()?.dtype() != DType::F32 => b(0),
            _ if self.dtype() == DType::I32 => Some(Bounds {
                min: i32::MIN as i64,
                max: i32::MAX as i64,
            }),
            (Ops::Add, _) => Some(Bounds {
                min: b(0)?.min.checked_add(b(1)?.min)?,
                max: b(0)?.max.checked_add(b(1)?.max)?,
            }),
            (Ops::Sub, _) => Some(Bounds {
                min: b(0)?.min.checked_sub(b(1)?.max)?,
                max: b(0)?.max.checked_sub(b(1)?.min)?,
            }),
            (Ops::Mul, _) => {
                let a = b(0)?;
                let c = b(1)?;
                let vals = [
                    a.min.checked_mul(c.min)?,
                    a.min.checked_mul(c.max)?,
                    a.max.checked_mul(c.min)?,
                    a.max.checked_mul(c.max)?,
                ];
                Some(Bounds {
                    min: *vals.iter().min()?,
                    max: *vals.iter().max()?,
                })
            }
            (Ops::Div, _) => {
                let a = b(0)?;
                let c = b(1)?;
                if c.min <= 0 && c.max >= 0 {
                    return None;
                }
                let vals = [
                    floor_div(a.min, c.min)?,
                    floor_div(a.min, c.max)?,
                    floor_div(a.max, c.min)?,
                    floor_div(a.max, c.max)?,
                ];
                Some(Bounds {
                    min: *vals.iter().min()?,
                    max: *vals.iter().max()?,
                })
            }
            (Ops::Mod, _) => {
                let c = b(1)?;
                if c.min <= 0 {
                    return None;
                }
                Some(Bounds {
                    min: 0,
                    max: c.max - 1,
                })
            }
            (Ops::Neg, _) => Some(Bounds {
                min: b(0)?.max.checked_neg()?,
                max: b(0)?.min.checked_neg()?,
            }),
            (Ops::Where, _) => Some(Bounds {
                min: b(1)?.min.min(b(2)?.min),
                max: b(1)?.max.max(b(2)?.max),
            }),
            (Ops::Max, _) => Some(Bounds {
                min: b(0)?.min.max(b(1)?.min),
                max: b(0)?.max.max(b(1)?.max),
            }),
            _ => None,
        }
    }
}
pub fn floor_div(a: i64, b: i64) -> Option<i64> {
    let q = a.checked_div(b)?;
    let r = a.checked_rem(b)?;
    q.checked_sub(((r != 0) && ((r < 0) != (b < 0))) as i64)
}
pub fn floor_mod(a: i64, b: i64) -> Option<i64> {
    let r = a.checked_rem(b)?;
    if r != 0 && (r < 0) != (b < 0) {
        r.checked_add(b)
    } else {
        Some(r)
    }
}
pub fn constant_fold(u: &UOp) -> Option<UOp> {
    if !(is_unary(u.op) || is_binary(u.op) || u.op == Ops::Where)
        || u.src.iter().any(|s| s.op != Ops::Const)
    {
        return None;
    }
    let values = u
        .src
        .iter()
        .map(crate::scalar::Value::from_constant)
        .collect::<Option<Vec<_>>>()?;
    crate::scalar::eval_alu(u.op, &u.arg, &values)
        .ok()
        .map(|v| v.uop())
}

pub fn symbolic_matcher() -> PatternMatcher {
    let int = |v| UPat::constant(Arg::Int(v));
    PatternMatcher::new()
        .add(
            UPat::op(Ops::Bind).src(vec![UPat::any(), UPat::var("value")]),
            |c| Some(c["value"].clone()),
        )
        .add(UPat::any().named("u"), |c| constant_fold(&c["u"]))
        .add(
            UPat::op(Ops::Add)
                .src(vec![UPat::var("x").dtype(DType::I64), int(0)])
                .commutative(),
            |c| Some(c["x"].clone()),
        )
        .add(
            UPat::op(Ops::Mul)
                .src(vec![UPat::var("x").dtype(DType::I64), int(1)])
                .commutative(),
            |c| Some(c["x"].clone()),
        )
        .add(
            UPat::op(Ops::Mul)
                .src(vec![UPat::var("x").dtype(DType::I64), int(0)])
                .commutative(),
            |_| Some(UOp::int(0)),
        )
        .add(
            UPat::op(Ops::Sub).src(vec![UPat::var("x").dtype(DType::I64), UPat::var("x")]),
            |_| Some(UOp::int(0)),
        )
        .add(
            UPat::op(Ops::Div).src(vec![UPat::var("x").dtype(DType::I64), int(1)]),
            |c| Some(c["x"].clone()),
        )
        .add(
            UPat::op(Ops::Mod).src(vec![UPat::var("x").dtype(DType::I64), int(1)]),
            |_| Some(UOp::int(0)),
        )
        .add(
            UPat::op(Ops::Where).src(vec![
                UPat::constant(Arg::Bool(true)),
                UPat::var("x"),
                UPat::any(),
            ]),
            |c| Some(c["x"].clone()),
        )
        .add(
            UPat::op(Ops::Where).src(vec![
                UPat::constant(Arg::Bool(false)),
                UPat::any(),
                UPat::var("x"),
            ]),
            |c| Some(c["x"].clone()),
        )
        .add(
            UPat::op(Ops::And)
                .src(vec![UPat::var("x"), UPat::constant(Arg::Bool(true))])
                .commutative(),
            |c| Some(c["x"].clone()),
        )
        .add(
            UPat::op(Ops::And)
                .src(vec![UPat::any(), UPat::constant(Arg::Bool(false))])
                .commutative(),
            |_| Some(UOp::bool(false)),
        )
        .add(UPat::op(Ops::Lt).named("u"), |c| {
            let u = &c["u"];
            let a = u.src[0].bounds()?;
            let b = u.src[1].bounds()?;
            if a.max < b.min {
                Some(UOp::bool(true))
            } else if a.min >= b.max {
                Some(UOp::bool(false))
            } else {
                None
            }
        })
}
pub fn simplify(u: &UOp) -> Result<UOp> {
    graph_rewrite(u, &symbolic_matcher())
}
```

`src/tensor.rs` の `impl Tensor` 内を追記する。

```rust
// file="src/tensor.rs" mode="append" within="impl Tensor"
    pub fn variable(name: &str, min: i64, max: i64) -> Self {
        Self::from_uop(UOp::variable(name, min, max).cast(DType::F32))
    }
```

`src/uir.rs` の `fn op`を置き換える。

```rust
// file="src/uir.rs" mode="replace" item="fn op"
fn op(s: &str) -> Result<Ops> {
    [
        Ops::Invalid,
        Ops::Const,
        Ops::Barrier,
        Ops::Special,
        Ops::Buffer,
        Ops::Alloc,
        Ops::Param,
        Ops::Variable,
        Ops::Bind,
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
        Ops::CustomFunction,
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
        Arg::Variable { name, min, max } => {
            format!("variable;{};{min};{max}", hex(name.as_bytes()))
        }
        Arg::None => "none".into(),
        Arg::Float(v) => format!("float;{:016x}", v.value().to_bits()),
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
        Arg::Function { name, dtype } => format!("function;{};{dtype:?}", hex(name.as_bytes())),
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
        ["variable", n, a, b] => {
            let (min, max) = (number(a)?, number(b)?);
            if min > max {
                return Err("reversed variable bounds".into());
            }
            Arg::Variable {
                name: string(n)?,
                min,
                max,
            }
        }
        ["float", v] if v.len() == 16 && v.bytes().all(|b| b.is_ascii_hexdigit()) => {
            Arg::Float(crate::uop::ConstFloat::new(f64::from_bits(
                u64::from_str_radix(v, 16).map_err(|_| "invalid float bits")?
            )))
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
                "Local" => Axis::Local,
                "Global" => Axis::Global,
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
        ["function", name, dt] => Arg::Function {
            name: string(name)?,
            dtype: dtype(dt)?,
        },
        ["scalar", i, dt] => Arg::ScalarParam {
            index: number(i)?,
            dtype: dtype(dt)?,
        },
        _ => return Err("invalid typed argument".into()),
    })
}
```

`src/uir.rs` の `fn validate_node`を置き換える。

```rust
// file="src/uir.rs" mode="replace" item="fn validate_node"
fn validate_node(u: &UOp) -> Result<()> {
    if crate::uop::is_binary(u.op) || crate::uop::is_unary(u.op) || u.op == Ops::Where {
        if u.src.iter().all(|s| {
            matches!(
                s.dtype(),
                DType::Bool | DType::I32 | DType::I64 | DType::F32
            ) && s.shape().is_ok_and(|shape| shape.is_empty())
        }) {
            crate::scalar::verify_node(u)?;
        } else {
            u.shape()?;
        }
    }
    if u.op == Ops::Variable {
        return match &u.arg {
            Arg::Variable { min, max, .. } if min <= max && u.src.is_empty() => Ok(()),
            _ => Err("invalid variable".into()),
        };
    }
    if u.op == Ops::Bind {
        return if u.src.len() == 2
            && u.src[0].op == Ops::Variable
            && matches!(u.src[1].arg,Arg::Int(n) if u.src[0].bounds().is_some_and(|b| n>=b.min&&n<=b.max))
        {
            Ok(())
        } else {
            Err("invalid variable binding".into())
        };
    }
    match (&u.op, &u.arg) {
        (Ops::Const, _) => {
            crate::scalar::verify_node(u)?;
        }
        (Ops::Param, Arg::ScalarParam { .. }) => {
            crate::scalar::verify_node(u)?;
        }
        (Ops::Param, Arg::Slot { .. })
        | (Ops::Buffer, Arg::Buffer { .. })
        | (Ops::Alloc, Arg::Allocation { .. })
            if u.src.is_empty() =>
        {
            if u.op == Ops::Buffer {
                u.shape()?;
            }
        }
        (Ops::Param | Ops::Buffer | Ops::Alloc, _) => {
            return Err("invalid leaf metadata".into());
        }
        (Ops::Range, Arg::Range { .. })
            if u.src.len() == 2 && u.src.iter().all(|s| s.dtype() == DType::I64) => {}
        (Ops::Range, _) => return Err("invalid RANGE".into()),
        (Ops::End, Arg::None)
            if !u.src.is_empty() && u.src.last().is_some_and(|r| r.op == Ops::Range) => {}
        (Ops::End, _) => return Err("invalid END".into()),
        (Ops::Source, Arg::Text(_))
        | (Ops::Binary, Arg::Bytes(_))
        | (Ops::Invalid, Arg::Text(_))
            if u.src.is_empty() => {}
        (Ops::Source | Ops::Binary | Ops::Invalid, _) => {
            return Err("invalid source payload".into());
        }
        (Ops::Load, Arg::None | Arg::DType(_))
            if u.src.len() == 1 && u.src[0].dtype() == DType::Ptr => {}
        (Ops::Store, Arg::None | Arg::DType(_))
            if u.src.len() == 2 && u.src[0].dtype() == DType::Ptr => {}
        (Ops::Load | Ops::Store, _) => return Err("invalid memory operation".into()),
        (Ops::Linear, Arg::None) => {
            crate::codegen::verify_lowered(u)?;
            validate_linear_scope(u)?;
        }
        (Ops::Linear, _) => return Err("invalid LINEAR".into()),
        (Ops::Call, Arg::Id(count))
            if u.src.first().is_some_and(|b| b.op == Ops::Stack)
                && u.src.len().checked_sub(1) == Some(*count) => {}
        (Ops::Call, Arg::None) if u.src.first().is_some_and(|b| b.op == Ops::CustomFunction) => {}
        (Ops::CustomFunction, Arg::Function { .. }) => {}
        (Ops::Call, _) => {
            return Err("invalid CALL".into());
        }
        _ => {}
    }
    Ok(())
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
    Variable,
    Bind,
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
    CustomFunction,
    Special,
    Barrier,
}
```

`src/uop.rs` の `pub enum Arg`を置き換える。

```rust
// file="src/uop.rs" mode="replace" item="pub enum Arg"
pub enum Arg {
    Variable {
        name: String,
        min: i64,
        max: i64,
    },
    None,
    Float(ConstFloat),
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
    Function {
        name: String,
        dtype: DType,
    },
}
```

`src/uop.rs` の `pub struct Node`を置き換える。

```rust
// file="src/uop.rs" mode="replace" item="pub struct Node"
pub struct Node {
    pub id: u64,
    pub op: Ops,
    pub src: Vec<UOp>,
    pub arg: Arg,
    shape_cache: OnceCell<Result<Vec<usize>>>,
    dtype_cache: OnceCell<DType>,
    pub(crate) bounds_cache: OnceCell<Option<(i64, i64)>>,
}
```

`src/uop.rs` の `impl UOp` 内 の `pub fn new`を置き換える。

```rust
// file="src/uop.rs" mode="replace" within="impl UOp" item="pub fn new"
    pub fn new(op: Ops, src: Vec<UOp>, arg: Arg) -> Self {
        INTERN.with(|cell| {
            let mut intern = cell.borrow_mut();
            let key = Key {
                op,
                src: src.iter().map(|x| x.id).collect(),
                arg: arg.clone(),
            };
            if let Some(node) = intern.nodes.get(&key).and_then(Weak::upgrade) {
                return Self(node);
            }
            intern.next += 1;
            let node = Rc::new(Node {
                id: intern.next,
                op,
                src,
                arg,
                shape_cache: OnceCell::new(),
                dtype_cache: OnceCell::new(),
                bounds_cache: OnceCell::new(),
            });
            intern.nodes.insert(key, Rc::downgrade(&node));
            if intern.nodes.len() % 1024 == 0 {
                intern.nodes.retain(|_, v| v.strong_count() != 0);
            }
            Self(node)
        })
    }
```

`src/uop.rs` の `impl UOp` 内 の `fn infer_dtype`を置き換える。

```rust
// file="src/uop.rs" mode="replace" within="impl UOp" item="fn infer_dtype"
    fn infer_dtype(&self) -> DType {
        match self.op {
            Ops::Const => match self.arg {
                Arg::Float(_) => DType::F32,
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
            Ops::Variable | Ops::Range | Ops::Special => DType::I64,
            Ops::Lt | Ops::Eq => DType::Bool,
            Ops::Cast | Ops::Bitcast => {
                if let Arg::DType(dt) = self.arg {
                    dt
                } else {
                    DType::Void
                }
            }
            Ops::CustomFunction => {
                if let Arg::Function { dtype, .. } = self.arg {
                    dtype
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
            (Ops::Const, _)
            | (Ops::Variable, _)
            | (Ops::Bind, _)
            | (Ops::Range, _)
            | (Ops::Load, _)
            | (Ops::CustomFunction, _) => Ok(vec![]),
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

`src/uop.rs` の `impl UOp` 内を追記する。

```rust
// file="src/uop.rs" mode="append" within="impl UOp"
    pub fn variable(name: &str, min: i64, max: i64) -> Self {
        if min > max {
            return Self::invalid("variable minimum exceeds maximum");
        }
        Self::new(
            Ops::Variable,
            vec![],
            Arg::Variable {
                name: name.into(),
                min,
                max,
            },
        )
    }

    pub fn bind(&self, value: i64) -> Self {
        if let Arg::Variable { min, max, .. } = self.arg {
            if value < min || value > max {
                return Self::invalid("binding lies outside variable bounds");
            }
        }
        Self::new(Ops::Bind, vec![self.clone(), Self::int(value)], Arg::None)
    }
```

`tests/cpu_workers.rs`を保存する。

```rust
// file="tests/cpu_workers.rs" mode="write"
use chibigrad::optimization::{reorder_ranges, split_range, unroll_range};
use chibigrad::parallel::ParallelProgram;
use chibigrad::{Arg, Axis, Buffer, DType, Ops, Tensor, UOp, codegen};
use std::collections::{HashMap, HashSet};
fn param(index: usize, shape: &[usize]) -> UOp {
    UOp::new(
        Ops::Param,
        vec![],
        Arg::Slot {
            index,
            shape: shape.to_vec(),
        },
    )
}
fn end(body: UOp, range: UOp) -> UOp {
    UOp::new(Ops::End, vec![body, range], Arg::None)
}

#[test]
fn workers_execute_native_rowsum_and_rebind_without_recompilation() {
    let x = Tensor::parameter("x", &[7, 65]);
    let y = x.sum_axes(&[1], false);
    let executable = y.compile().unwrap();
    let linear = &executable.programs()[0].src[1];
    let input_slot = codegen::signature(linear)
        .unwrap()
        .buffers
        .into_iter()
        .find(|(_, u)| matches!(&u.arg,Arg::Buffer{name,..} if name=="x"))
        .unwrap()
        .0;
    for workers in [1, 2, 3, 16] {
        let program = ParallelProgram::compile(linear, 0, workers).unwrap();
        let object = program.object().to_vec();
        for offset in [0., 3., -2.] {
            let buffer = Buffer::new(
                vec![7, 65],
                (0..455).map(|i| (i % 11) as f32 + offset).collect(),
            )
            .unwrap();
            let expected = buffer
                .data
                .chunks(65)
                .map(|r| r.iter().sum::<f32>())
                .collect::<Vec<_>>();
            let result = program.run(&[(input_slot, buffer)]).unwrap();
            assert_eq!(result.output.data, expected);
            assert_eq!(
                result.thread_ids.iter().collect::<HashSet<_>>().len(),
                workers.min(7)
            );
            assert!(
                result
                    .thread_ids
                    .iter()
                    .all(|&id| id != std::thread::current().id())
            );
            assert_eq!(
                result
                    .partitions
                    .iter()
                    .flat_map(|r| r.clone())
                    .collect::<Vec<_>>(),
                (0..7).collect::<Vec<_>>()
            );
            assert_eq!(program.object(), object);
        }
    }
}

#[test]
fn workers_execute_flattened_gemm_with_private_reduction_state() {
    let a = Tensor::parameter("a", &[3, 7]);
    let b = Tensor::parameter("b", &[7, 5]);
    let executable = a.matmul(&b).compile().unwrap();
    let linear = &executable.programs()[0].src[1];
    let av = (0..21).map(|i| (i % 7) as f32 - 2.).collect::<Vec<_>>();
    let bv = (0..35).map(|i| (i % 11) as f32 - 5.).collect::<Vec<_>>();
    let mut buffers = vec![];
    for (slot, u) in codegen::signature(linear).unwrap().buffers {
        if let Arg::Buffer { name, shape, .. } = &u.arg {
            if name == "a" {
                buffers.push((slot, Buffer::new(shape.clone(), av.clone()).unwrap()));
            }
            if name == "b" {
                buffers.push((slot, Buffer::new(shape.clone(), bv.clone()).unwrap()));
            }
        }
    }
    let expected = (0..15)
        .map(|p| {
            (0..7)
                .map(|k| av[p / 5 * 7 + k] * bv[k * 5 + p % 5])
                .sum::<f32>()
        })
        .collect::<Vec<_>>();
    for workers in [2, 4, 32] {
        let program = ParallelProgram::compile(linear, 0, workers).unwrap();
        assert_eq!(program.run(&buffers).unwrap().output.data, expected);
    }
}

#[test]
fn splitting_and_workers_reject_a_cross_iteration_accumulator() {
    let i = UOp::range(UOp::int(3), 0, Axis::Loop);
    let j = UOp::range(UOp::int(3), 1, Axis::Loop);
    let alloc = UOp::new(
        Ops::Alloc,
        vec![],
        Arg::Allocation {
            id: 0,
            dtype: DType::F32,
            size: 1,
        },
    );
    let init = alloc.store(UOp::f32(0.));
    let update = alloc.store(alloc.load().add(&UOp::f32(1.)));
    let ej = end(update.clone(), j.clone());
    let output = param(0, &[3])
        .index(i.clone(), UOp::bool(true))
        .store(alloc.after(vec![ej.clone()]).load());
    let ei = end(output.clone(), i.clone());
    let linear = UOp::linear(vec![alloc, init, i, j, update, ej, output, ei]);
    assert!(split_range(&linear, 0, 2).is_err());
    assert!(reorder_ranges(&linear, &[1, 0]).is_err());
    assert!(ParallelProgram::compile(&linear, 0, 2).is_err());
}

#[test]
fn unknown_effects_and_overlapping_output_addresses_are_rejected() {
    let i = UOp::range(UOp::int(3), 0, Axis::Loop);
    let j = UOp::range(UOp::int(3), 1, Axis::Loop);
    let output = param(0, &[9]);
    let store = output.index(i.add(&j), UOp::bool(true)).store(UOp::f32(1.));
    let ej = end(store.clone(), j.clone());
    let ei = end(ej.clone(), i.clone());
    let overlap = UOp::linear(vec![i.clone(), j, store, ej, ei]);
    assert!(reorder_ranges(&overlap, &[1, 0]).is_err());
    let call = UOp::new(Ops::Call, vec![], Arg::Text("opaque_effect".to_owned()));
    let effect = UOp::linear(vec![i.clone(), call.clone(), end(call, i)]);
    assert!(split_range(&effect, 0, 2).is_err());
    assert!(unroll_range(&effect, 0, 2).is_err());
    assert!(ParallelProgram::compile(&effect, 0, 2).is_err());
}

#[test]
fn worker_count_is_bounded_by_outputs_and_empty_outputs_do_not_spawn() {
    for n in [0, 1] {
        let output = param(0, &[n]);
        let r = UOp::range(UOp::int(n as i64), 0, Axis::Loop);
        let store = output.index(r.clone(), UOp::bool(true)).store(UOp::f32(7.));
        let linear = UOp::linear(vec![r.clone(), store.clone(), end(store, r)]);
        let program = ParallelProgram::compile(&linear, 0, 16).unwrap();
        assert_eq!(program.worker_count(), n);
        let result = program.run(&[]).unwrap();
        assert_eq!(result.output.data, vec![7.; n]);
        assert_eq!(result.thread_ids.len(), n);
    }
}

fn matrix_kernel(m: i64, n: i64, k: i64) -> UOp {
    let a = param(0, &[m as usize, k as usize]);
    let b = param(1, &[k as usize, n as usize]);
    let c = param(2, &[m as usize, n as usize]);
    let i = UOp::range(UOp::int(m), 0, Axis::Loop);
    let j = UOp::range(UOp::int(n), 1, Axis::Loop);
    let r = UOp::range(UOp::int(k), 2, Axis::Reduce);
    let alloc = UOp::new(
        Ops::Alloc,
        vec![],
        Arg::Allocation {
            id: 0,
            dtype: DType::F32,
            size: 1,
        },
    );
    let init = alloc.store(UOp::f32(0.));
    let ptr = alloc.after(vec![init.clone(), r.clone()]);
    let av = a.index(i.mul(&UOp::int(k)).add(&r), UOp::bool(true)).load();
    let bv = b.index(r.mul(&UOp::int(n)).add(&j), UOp::bool(true)).load();
    let update = ptr.store(ptr.load().add(&av.mul(&bv)));
    let ek = end(update.clone(), r.clone());
    let store = c
        .index(i.mul(&UOp::int(n)).add(&j), UOp::bool(true))
        .store(alloc.after(vec![ek.clone()]).load());
    let ej = end(store.clone(), j.clone());
    let ei = end(ej.clone(), i.clone());
    UOp::linear(vec![i, j, alloc, init, r, update, ek, store, ej, ei])
}
#[test]
fn worker_scalars_rebind_and_invalid_execution_is_rejected() {
    let input = param(0, &[17]);
    let output = param(1, &[17]);
    let r = UOp::range(UOp::int(17), 0, Axis::Loop);
    let factor = UOp::variable("factor", -5, 5);
    let value = input
        .index(r.clone(), UOp::bool(true))
        .load()
        .mul(&factor.cast(DType::F32));
    let store = output.index(r.clone(), UOp::bool(true)).store(value);
    let linear = UOp::linear(vec![r.clone(), store.clone(), end(store, r)]);
    let program = ParallelProgram::compile(&linear, 0, 3).unwrap();
    let input = Buffer::new(vec![17], (0..17).map(|i| i as f32).collect()).unwrap();
    for factor in [-3, 2, 5] {
        let run = program
            .run_with_vars(
                &[(0, input.clone())],
                &HashMap::from([("factor".to_owned(), factor)]),
            )
            .unwrap();
        assert_eq!(
            run.output.data,
            input
                .data
                .iter()
                .map(|v| v * factor as f32)
                .collect::<Vec<_>>()
        );
    }
    assert!(program.run_with_vars(&[], &HashMap::new()).is_err());
    assert!(
        program
            .run_with_vars(&[(0, input.clone())], &HashMap::new())
            .is_err()
    );
    assert!(
        program
            .run_with_vars(&[(0, input)], &HashMap::from([("factor".to_owned(), 6)]))
            .is_err()
    );
    assert!(ParallelProgram::compile(&linear, 0, 0).is_err());
    assert!(ParallelProgram::compile(&linear, 900, 3).is_err());
    assert!(ParallelProgram::compile(&matrix_kernel(3, 5, 7), 0, 3).is_err());
}
```

`tests/gpu_conv.rs`を保存する。

```rust
// file="tests/gpu_conv.rs" mode="write"
#![cfg(any(feature = "metal", feature = "opencl"))]
mod support;
use chibigrad::{UOp, gpu};
fn check(run: fn(&UOp, gpu::Launch, &[Vec<f32>]) -> Vec<f32>) {
    let (linear, launch) = gpu::convolution(1, 1, 3, 3, 1, 2, 2, 4).unwrap();
    let result = run(
        &linear,
        launch,
        &[
            (1..=9).map(|n| n as f32).collect(),
            vec![1., 0., 0., -1.],
            vec![0.; 4],
        ],
    );
    assert_eq!(result, vec![-4.; 4]);
}
#[cfg(feature = "metal")]
#[test]
fn metal_native() {
    check(support::run_metal);
}
#[cfg(feature = "opencl")]
#[test]
fn opencl_native() {
    check(support::run_cl);
}
```

`tests/gpu_gemm.rs`を保存する。

```rust
// file="tests/gpu_gemm.rs" mode="write"
#![cfg(any(all(feature = "metal", target_os = "macos"), feature = "opencl"))]
mod support;
use chibigrad::{UOp, gpu};
fn check(run: fn(&UOp, gpu::Launch, &[Vec<f32>]) -> Vec<f32>) {
    let (m, n, k) = (5, 7, 9);
    let a = (0..m * k).map(|i| i as f32 / 10.).collect::<Vec<_>>();
    let b = (0..k * n).map(|i| i as f32 / 13.).collect::<Vec<_>>();
    let (linear, launch) = gpu::tiled_gemm(m, n, k, 4).unwrap();
    let result = run(&linear, launch, &[a.clone(), b.clone(), vec![0.; m * n]]);
    for i in 0..m {
        for j in 0..n {
            let expected = (0..k).map(|r| a[i * k + r] * b[r * n + j]).sum::<f32>();
            assert!((result[i * n + j] - expected).abs() < 1e-3);
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_native() {
    check(support::run_metal);
}
#[cfg(feature = "opencl")]
#[test]
fn opencl_native() {
    check(support::run_cl);
}
```

`tests/gpu_reduction.rs`を保存する。

```rust
// file="tests/gpu_reduction.rs" mode="write"
#![cfg(any(all(feature = "metal", target_os = "macos"), feature = "opencl"))]
mod support;
use chibigrad::{UOp, gpu};
fn check(run: fn(&UOp, gpu::Launch, &[Vec<f32>]) -> Vec<f32>) {
    let (linear, launch) = gpu::local_reduction(3, 37, 16).unwrap();
    let values = (0..111).map(|i| i as f32 / 10.).collect::<Vec<_>>();
    let result = run(&linear, launch, &[values.clone(), vec![0.; 3]]);
    for row in 0..3 {
        assert!((result[row] - values[row * 37..(row + 1) * 37].iter().sum::<f32>()).abs() < 1e-3);
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_native() {
    check(support::run_metal);
}
#[cfg(feature = "opencl")]
#[test]
fn opencl_native() {
    check(support::run_cl);
}
```

`tests/support/mod.rs` の `pub fn run_metal`を置き換える。

```rust
// file="tests/support/mod.rs" mode="replace" item="pub fn run_metal"
pub fn run_metal(linear: &chibigrad::UOp, launch: gpu::Launch, inputs: &[Vec<f32>]) -> Vec<f32> {
    use chibigrad::{backends::metal::MetalDevice, queue::HWQueue};
    let device = MetalDevice::open().unwrap();
    eprintln!("Metal device: {:?}", device.capabilities());
    let spec = renderers::render(linear, "test_kernel", Dialect::Metal, launch.clone()).unwrap();
    let program = device
        .compile(&spec)
        .unwrap_or_else(|e| panic!("{e}\n{}", spec.source));
    let buffers = inputs
        .iter()
        .map(|input| {
            let b = device.allocate(input.len() * 4).unwrap();
            b.write(0, &words(input)).unwrap();
            b
        })
        .collect::<Vec<_>>();
    let commands = device
        .command_buffer(&[(program, launch, buffers.clone())])
        .unwrap();
    let mut queue = HWQueue::new();
    queue.metal(commands, chibigrad::UOp::int(0), HWQueue::runtime_slot(0));
    let source = chibigrad::codegen::render(queue.linear(), "runtime_entry").unwrap();
    let cpu = chibigrad::cpu::CpuProgram::compile_with_symbols(
        &source,
        "runtime_entry",
        &chibigrad::queue::foreign_symbols(),
    )
    .unwrap_or_else(|e| panic!("{e}\n{source}"));
    let mut signal = [1i64];
    let mut args = (signal.as_mut_ptr() as u64).to_ne_bytes();
    unsafe { cpu.run(&mut args) }.unwrap();
    assert_eq!(chibigrad::queue::take_error(), None);
    assert_eq!(
        floats(
            buffers
                .last()
                .unwrap()
                .read(0, inputs.last().unwrap().len() * 4)
                .unwrap()
        )
        .len(),
        inputs.last().unwrap().len()
    );
    signal[0] = 2;
    std::hint::black_box(&signal);
    unsafe { cpu.run(&mut args) }.unwrap();
    assert_eq!(chibigrad::queue::take_error(), None);
    floats(
        buffers
            .last()
            .unwrap()
            .read(0, inputs.last().unwrap().len() * 4)
            .unwrap(),
    )
}
```

`tests/support/mod.rs` の `pub fn run_cl`を置き換える。

```rust
// file="tests/support/mod.rs" mode="replace" item="pub fn run_cl"
pub fn run_cl(linear: &chibigrad::UOp, launch: gpu::Launch, inputs: &[Vec<f32>]) -> Vec<f32> {
    use chibigrad::{
        backends::opencl::{Argument, OpenClDevice},
        queue::HWQueue,
    };
    let device = OpenClDevice::open(0).unwrap();
    eprintln!("OpenCL device: {:?}", device.capabilities());
    let spec = renderers::render(linear, "test_kernel", Dialect::OpenCl, launch.clone()).unwrap();
    let program = device
        .compile(&spec)
        .unwrap_or_else(|e| panic!("{e}\n{}", spec.source));
    let buffers = inputs
        .iter()
        .map(|input| {
            let b = device.allocate(input.len() * 4).unwrap();
            b.write(0, &words(input)).unwrap();
            b
        })
        .collect::<Vec<_>>();
    let command = program
        .command(
            launch,
            buffers.iter().cloned().map(Argument::Buffer).collect(),
        )
        .unwrap();
    let mut queue = HWQueue::new();
    queue.opencl(command);
    let source = chibigrad::codegen::render(queue.linear(), "runtime_entry").unwrap();
    let cpu = chibigrad::cpu::CpuProgram::compile_with_symbols(
        &source,
        "runtime_entry",
        &chibigrad::queue::foreign_symbols(),
    )
    .unwrap_or_else(|e| panic!("{e}\n{source}"));
    unsafe { cpu.run(&mut []) }.unwrap();
    unsafe { cpu.run(&mut []) }.unwrap();
    assert_eq!(chibigrad::queue::take_error(), None);
    floats(
        buffers
            .last()
            .unwrap()
            .read(0, inputs.last().unwrap().len() * 4)
            .unwrap(),
    )
}
```

`tests/symbolic_scalar.rs`を保存する。

```rust
// file="tests/symbolic_scalar.rs" mode="write"
use chibigrad::{Arg, DType, Ops, Tensor, UOp, functions::Function, symbolic::simplify};
use std::collections::HashMap;
#[test]
fn intervals_bind_and_simplify() {
    let x = UOp::variable("x", 1, 8);
    let b = x.mul(&UOp::int(2)).bounds().unwrap();
    assert_eq!((b.min, b.max), (2, 16));
    assert!(x.div(&UOp::variable("d", -1, 1)).bounds().is_none());
    assert_eq!(simplify(&x.mul(&UOp::int(1)).add(&UOp::int(0))).unwrap(), x);
    assert_eq!(simplify(&x.lt(&UOp::int(9))).unwrap(), UOp::bool(true));
    assert_eq!(simplify(&x.bind(4)).unwrap(), UOp::int(4));
    assert_eq!(x.bind(9).op, Ops::Invalid);
}
#[test]
fn dynamic_stack_selection() {
    let p = Function::parameter(0, DType::F32);
    let f = Function::new(&[DType::F32], &[p.clone(), p.mul(&UOp::f32(3.))]).unwrap();
    let call = f.call(&[UOp::f32(2.)]).unwrap();
    let selected = f.select(&call, &UOp::variable("pick", 0, 1)).unwrap();
    let output = UOp::new(
        Ops::Param,
        vec![],
        Arg::Slot {
            index: 0,
            shape: vec![1],
        },
    );
    let linear = chibigrad::codegen::linearize(&UOp::sink(vec![output.store(selected)])).unwrap();
    let source = chibigrad::codegen::render(&linear, "select_stack").unwrap();
    let native = chibigrad::cpu::CpuProgram::compile(&source, "select_stack").unwrap();
    for (pick, want) in [(0i64, 2.), (1, 6.)] {
        let mut value = 0f32;
        let mut bytes = (&mut value as *mut f32 as u64).to_ne_bytes().to_vec();
        bytes.extend(pick.to_ne_bytes());
        unsafe { native.run(&mut bytes) }.unwrap();
        assert_eq!(value, want);
    }
    assert!(f.select(&call, &UOp::variable("bad", 0, 2)).is_err());
    assert!(f.select(&call, &UOp::i32(1)).is_err());
    assert!(f.select(&call, &UOp::bool(true)).is_err());
}
#[test]
fn native_scalar_tensor_rebinding() {
    let x = Tensor::from_slice(&[2., 3.], &[2]);
    let p = x.mul(&Tensor::variable("scale", 1, 4)).compile().unwrap();
    for (n, want) in [(2, vec![4., 6.]), (3, vec![6., 9.])] {
        assert_eq!(
            p.run(&[], &HashMap::from([("scale".into(), n)])).unwrap()[0].data,
            want
        );
    }
    assert!(p.run(&[], &HashMap::new()).is_err());
    assert!(p.run(&[], &HashMap::from([("scale".into(), 5)])).is_err());
}

#[test]
fn scalar_symbol_uir_and_bound_evaluation() {
    let x = UOp::variable("n", 1, 8);
    let expression = x.bind(4).mul(&UOp::int(2));
    let text = chibigrad::uir::print(&expression);
    let parsed = chibigrad::uir::parse(&text).unwrap();
    chibigrad::uir::assert_uop(&expression, &parsed).unwrap();
    assert_eq!(
        chibigrad::scalar::evaluate(&parsed, &[]).unwrap(),
        chibigrad::scalar::Value::I64(8)
    );
    assert_eq!(
        chibigrad::scalar::compile(&parsed)
            .unwrap()
            .run(&[])
            .unwrap(),
        chibigrad::scalar::Value::I64(8)
    );
    assert!(chibigrad::uir::parse(&text.replace("variable;6e;1;8", "variable;6e;9;8")).is_err());
}
```

`tests/tensor_gpu.rs`を追記する。

```rust
// file="tests/tensor_gpu.rs" mode="append"

#[test]
fn integer_scalar_max_and_rebinding() {
    let x = Tensor::from_slice(&[1., 2., 3., 4.], &[4]);
    let output = x.maximum(&Tensor::variable("floor", 1, 4));
    for backend in backends() {
        let mut exe = GpuExecutable::compile(&[output.clone()], backend).unwrap();
        for value in [1, 3] {
            let vars = HashMap::from([("floor".into(), value)]);
            close(
                &exe.run(&[], &vars).unwrap()[0].data,
                &[1f32, 2., 3., 4.].map(|x| x.max(value as f32)),
            );
        }
        assert!(exe.run(&[], &HashMap::from([("floor".into(), 5)])).is_err());
    }
}
```

```sh
# check step=20.01
cargo test --offline
```

```sh
# check requires=metal,opencl
cargo test --offline --features metal,opencl --test tensor_gpu
```

範囲外の束縛を拒否し、STACK の選択を変えて 2 と 6、scale を変えて [4,6] と [6,9] を同じ実行計画から得る。並列 CPU と Metal・OpenCL のスカラー再束縛も通る。

## 20.02 最大容量と実行時の次元を分ける

shape は最大容量、logical_shape は実行時の次元を表す UOp 列として分ける。symbolic_parameter と reshape_symbolic がその列を保持し、スケジューラも出力バッファへ引き継ぐ。CPU と GPU は実際の次元から入力長、reshape の要素数、shrink の範囲を検証して実行する。勾配は固定次元だけを受け付け、容量分の誤った勾配を返さない。UIR にも次元グラフを残す。

`src/codegen.rs`を保存する。

```rust
// file="src/codegen.rs" mode="write"
use crate::{
    Result,
    dimensions::product,
    pattern::{PatternMatcher, UPat, graph_rewrite},
    symbolic::simplify,
    uop::{Arg, Axis, DType, Ops, UOp, is_binary, is_unary, numel},
};
use std::collections::{HashMap, HashSet};
pub fn normalize(root: &UOp) -> Result<UOp> {
    let root = crate::functions::inline_calls(root)?;
    let pm = PatternMatcher::new()
        .add(UPat::op(Ops::Exp).src(vec![UPat::var("x")]), |c| {
            Some(
                c["x"]
                    .mul(&UOp::f32(std::f32::consts::LOG2_E))
                    .alu(Ops::Exp2, &[]),
            )
        })
        .add(UPat::op(Ops::Log).src(vec![UPat::var("x")]), |c| {
            Some(
                c["x"]
                    .alu(Ops::Log2, &[])
                    .mul(&UOp::f32(std::f32::consts::LN_2)),
            )
        })
        .add(UPat::op(Ops::Cos).src(vec![UPat::var("x")]), |c| {
            Some(
                UOp::f32(std::f32::consts::FRAC_PI_2)
                    .sub(&c["x"])
                    .alu(Ops::Sin, &[]),
            )
        });
    graph_rewrite(&root, &pm)
}
fn coords(linear: &UOp, shape: &[UOp]) -> Vec<UOp> {
    let mut stride = UOp::int(1);
    let mut out = vec![];
    for n in shape.iter().rev() {
        out.push(linear.div(&stride).modulo(n));
        stride = stride.mul(n);
    }
    out.reverse();
    out
}
fn offset(indices: &[UOp], shape: &[UOp]) -> UOp {
    indices
        .iter()
        .zip(shape)
        .fold(UOp::int(0), |acc, (i, n)| acc.mul(n).add(i))
}
fn broadcast(indices: &[UOp], shape: &[UOp]) -> Vec<UOp> {
    let start = indices.len() - shape.len();
    shape
        .iter()
        .enumerate()
        .map(|(i, n)| {
            if *n == UOp::int(1) {
                UOp::int(0)
            } else {
                indices[start + i].clone()
            }
        })
        .collect()
}
struct Lower {
    ops: Vec<UOp>,
    seen: HashSet<u64>,
    next_range: usize,
    next_alloc: usize,
    cache: HashMap<(u64, Vec<u64>, u64), UOp>,
}
impl Lower {
    fn emit(&mut self, u: UOp) -> UOp {
        if self.seen.insert(u.id) {
            for s in u.ordered_sources() {
                self.emit(s.clone());
            }
            self.ops.push(u.clone());
        }
        u
    }
    fn range(&mut self, end: UOp, axis: Axis) -> UOp {
        let end = simplify(&end).unwrap_or(end);
        let r = UOp::range(end, self.next_range, axis);
        self.next_range += 1;
        self.emit(r)
    }
    fn close(&mut self, range: &UOp, body: UOp) -> UOp {
        self.emit(UOp::new(Ops::End, vec![body, range.clone()], Arg::None))
    }
    fn scalar(&mut self, u: &UOp, indices: &[UOp], valid: &UOp) -> Result<UOp> {
        let key = (u.id, indices.iter().map(|i| i.id).collect(), valid.id);
        if let Some(v) = self.cache.get(&key) {
            return Ok(v.clone());
        }
        let result = match (&u.op, &u.arg) {
            (Ops::Const, _) | (Ops::Variable, _) => self.emit(u.clone()),
            (Ops::Bind, _) => self.emit(u.src[1].clone()),
            (Ops::Buffer, _) | (Ops::Param, _) => {
                let shape = u.logical_shape()?;
                let idx = simplify(&offset(indices, &shape))?;
                let gate = simplify(valid)?;
                self.emit(u.index(idx, gate).load())
            }
            (Ops::Reshape, _) => {
                let linear = offset(indices, &u.logical_shape()?);
                self.scalar(
                    &u.src[0],
                    &coords(&linear, &u.src[0].logical_shape()?),
                    valid,
                )?
            }
            (Ops::Permute, Arg::Axes(axes)) => {
                let mut input = indices.to_vec();
                for (i, &a) in axes.iter().enumerate() {
                    input[a] = indices[i].clone();
                }
                self.scalar(&u.src[0], &input, valid)?
            }
            (Ops::Expand, _) => {
                let shape = u.src[0].logical_shape()?;
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
                let shape = u.src[0].logical_shape()?;
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
                let shape = u.logical_shape()?;
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
                let ishape = u.src[1].logical_shape()?;
                let raw = self.scalar(&u.src[1], &indices[*axis..axis + ishape.len()], valid)?;
                let shape = u.src[0].logical_shape()?;
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
                let shape = u.src[1].logical_shape()?;
                let r = self.range(product(&shape), Axis::Reduce);
                let ic = coords(&r, &shape);
                let raw = self.scalar(&u.src[1], &ic, valid)?;
                let index =
                    self.checked_index(raw, u.src[0].logical_shape()?[*axis].clone(), valid);
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
                let shape = u.src[0].logical_shape()?;
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
                    src.push(self.scalar(x, &broadcast(indices, &x.logical_shape()?), valid)?)
                }
                self.emit(UOp::new(*op, src, u.arg.clone()))
            }
            _ => return Err(format!("rangeify cannot lower {:?}", u.op)),
        };
        self.cache.insert(key, result.clone());
        Ok(result)
    }
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
    fn accumulator(&mut self, initial: UOp) -> (UOp, UOp) {
        let alloc = UOp::new(
            Ops::Alloc,
            vec![],
            Arg::Allocation {
                id: self.next_alloc,
                dtype: DType::F32,
                size: 1,
            },
        );
        self.next_alloc += 1;
        self.emit(alloc.clone());
        let init = self.emit(alloc.store(initial));
        (alloc, init)
    }
}
pub fn rangeify(sink: &UOp) -> Result<UOp> {
    if sink.op != Ops::Sink {
        return Err("rangeify expects SINK".into());
    }
    let sink = normalize(sink)?;
    let mut lower = Lower {
        ops: vec![],
        seen: HashSet::new(),
        next_range: 0,
        next_alloc: 0,
        cache: HashMap::new(),
    };
    for store in &sink.src {
        if store.op != Ops::Store || store.src.len() != 2 {
            return Err("rangeify expects STORE(buffer, tensor)".into());
        }
        let target = &store.src[0];
        let value = &store.src[1];
        let shape = value.logical_shape()?;
        if target.logical_shape()? != shape {
            return Err("output buffer shape mismatch".into());
        }
        let r = lower.range(product(&shape), Axis::Loop);
        let coords = coords(&r, &shape);
        let result = lower.scalar(value, &coords, &UOp::bool(true))?;
        let output = lower.emit(target.index(r.clone(), UOp::bool(true)).store(result));
        lower.close(&r, output);
        lower.cache.clear();
    }
    let linear = UOp::linear(lower.ops);
    verify_lowered(&linear)?;
    Ok(linear)
}
pub fn linearize(sink: &UOp) -> Result<UOp> {
    let sink = crate::functions::inline_calls(sink)?;
    if sink.op == Ops::Linear {
        return Ok(sink.clone());
    }
    if sink.op != Ops::Sink {
        return Err("linearize expects SINK".into());
    }
    let linear = UOp::linear(
        sink.toposort()
            .into_iter()
            .filter(|x| x.op != Ops::Sink)
            .collect(),
    );
    verify_lowered(&linear)?;
    Ok(linear)
}
pub fn verify_lowered(linear: &UOp) -> Result<()> {
    if linear.op != Ops::Linear {
        return Err("renderer expects LINEAR".into());
    }
    let mut ranges = vec![];
    for u in &linear.src {
        if u.op == Ops::Range {
            if u.src.len() != 2 {
                return Err("RANGE needs start/end".into());
            }
            ranges.push(u.id)
        }
        if u.op == Ops::End {
            let id = u.src.last().ok_or("END missing RANGE")?.id;
            if ranges.pop() != Some(id) {
                return Err("END does not match active RANGE".into());
            }
        }
        if u.op == Ops::Index {
            if u.src.len() != 3 {
                return Err("INDEX needs buffer,index,valid".into());
            }
            if u.src[2] == UOp::bool(true) {
                if let (Some(size), Some((lo, hi))) =
                    (pointer_size(&u.src[0]), index_bounds(&u.src[1]))
                {
                    if lo < 0 || hi >= size as i64 {
                        return Err("unguarded index outside allocation".into());
                    }
                }
            }
        }
    }
    if !ranges.is_empty() {
        return Err("unclosed RANGE".into());
    }
    crate::uir::validate_linear_scope(linear)?;
    Ok(())
}
pub fn to_program(sink: &UOp, entry: &str) -> Result<UOp> {
    let linear = if sink
        .src
        .iter()
        .any(|s| s.op == Ops::Store && s.src.first().is_some_and(|p| p.op == Ops::Buffer))
    {
        rangeify(sink)?
    } else {
        linearize(sink)?
    };
    let source = render(&linear, entry)?;
    Ok(UOp::new(
        Ops::Program,
        vec![
            sink.clone(),
            linear,
            UOp::new(Ops::Source, vec![], Arg::Text(source)),
        ],
        Arg::Text(entry.into()),
    ))
}
#[derive(Clone, Debug)]
pub struct Signature {
    pub buffers: Vec<(usize, UOp)>,
    pub variables: Vec<(usize, UOp)>,
}
pub fn signature(linear: &UOp) -> Result<Signature> {
    let mut buffers = vec![];
    let mut variables = vec![];
    let mut seen = HashSet::new();
    let mut occupied = HashSet::new();
    for u in linear.toposort() {
        if let (Ops::Param, Arg::Slot { index, .. }) = (&u.op, &u.arg) {
            occupied.insert(*index);
            buffers.push((*index, u));
        }
    }
    let mut next = 0;
    for u in linear.toposort() {
        if !seen.insert(u.id) {
            continue;
        }
        if u.op == Ops::Buffer {
            while occupied.contains(&next) {
                next += 1
            }
            buffers.push((next, u));
            occupied.insert(next);
            next += 1;
        }
    }
    next = occupied.iter().copied().max().map_or(0, |x| x + 1);
    for u in linear.toposort() {
        if u.op == Ops::Variable {
            variables.push((next, u));
            next += 1;
        }
    }
    Ok(Signature { buffers, variables })
}
fn ctype(dt: DType) -> &'static str {
    match dt {
        DType::F32 => "float",
        DType::F32x4 => "float4",
        DType::I32 => "int",
        DType::I64 => "long long",
        DType::Bool => "int",
        DType::Ptr => "void*",
        DType::Void => "void",
    }
}
fn identifier(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c == '_' || c.is_ascii_alphabetic())
        && chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}
pub fn render(linear: &UOp, entry: &str) -> Result<String> {
    render_impl(linear, entry, None)
}
pub fn render_checked(linear: &UOp, entry: &str, status_slot: usize) -> Result<String> {
    render_impl(linear, entry, Some(status_slot))
}
fn render_impl(linear: &UOp, entry: &str, status_slot: Option<usize>) -> Result<String> {
    let mut all = vec![];
    let mut seen = HashSet::new();
    let mut effects = HashSet::new();
    let mut scopes = vec![];
    fn add(
        u: &UOp,
        out: &mut Vec<UOp>,
        seen: &mut HashSet<u64>,
        effects: &mut HashSet<u64>,
        scopes: &mut Vec<HashSet<u64>>,
    ) {
        if seen.contains(&u.id) || effects.contains(&u.id) {
            return;
        }
        for src in u.ordered_sources() {
            add(src, out, seen, effects, scopes)
        }
        if u.op == Ops::Range {
            scopes.push(seen.clone());
        }
        seen.insert(u.id);
        out.push(u.clone());
        if matches!(u.op, Ops::Range | Ops::End | Ops::Store | Ops::Call) {
            effects.insert(u.id);
        }
        if u.op == Ops::End {
            if let Some(parent) = scopes.pop() {
                *seen = parent;
            }
        }
    }
    for u in &linear.src {
        add(u, &mut all, &mut seen, &mut effects, &mut scopes)
    }
    let complete = UOp::linear(all);
    let linear = &complete;
    verify_lowered(linear)?;

    if !identifier(entry) {
        return Err("invalid C entry identifier".into());
    }
    let sig = signature(linear)?;
    let mut text = String::from(
        "extern float exp2f(float);\nextern float log2f(float);\nextern float sinf(float);\nextern float sqrtf(float);\nextern float fmaxf(float,float);\nstatic float chibi_float(unsigned int x) { union { unsigned int u; float f; } b; b.u=x; return b.f; }\nstatic long long chibi_div(long long a,long long b) { long long q=a/b, r=a%b; return q-((r!=0)&&((r<0)!=(b<0))); }\nstatic long long chibi_mod(long long a,long long b) { long long r=a%b; return r+(((r!=0)&&((r<0)!=(b<0)))?b:0); }\n",
    );
    text.push_str(
        "static int chibi_int(unsigned int x) { union { unsigned int u; int i; } b; b.u=x; return b.i; }\nstatic unsigned int chibi_bits(float x) { union { unsigned int u; float f; } b; b.f=x; return b.u; }\nstatic int chibi_shr(int a,int b) { unsigned int u=(unsigned int)a; return chibi_int((u>>b)|((a<0&&b)?(~0u<<(32-b)):0u)); }\nstatic long long chibi_shr64(long long a,long long b) { unsigned long long u=(unsigned long long)a; return (long long)((u>>b)|((a<0&&b)?(~0ull<<(64-b)):0ull)); }\nstatic float chibi_max(float a,float b) { if(a!=a||b!=b)return chibi_float(2143289344u); if(a==0.0f&&b==0.0f)return 0.0f; return a>b?a:b; }\n",
    );
    let mut functions: HashMap<String, (DType, Vec<DType>)> = HashMap::new();
    for u in &linear.src {
        let (body, args) = if u.op == Ops::Call {
            let b = u.src.first().ok_or("CALL lacks body")?;
            (b, &u.src[1..])
        } else if u.op == Ops::CustomFunction && !u.src.is_empty() {
            (u, &u.src[..])
        } else {
            continue;
        };
        let Arg::Function { name, dtype } = &body.arg else {
            continue;
        };
        if !identifier(name) {
            return Err("invalid C function identifier".into());
        }
        let types = args.iter().map(UOp::dtype).collect::<Vec<_>>();
        if functions
            .get(name)
            .is_some_and(|old| old != &(*dtype, types.clone()))
        {
            return Err(format!("inconsistent callback signature for {name}"));
        }
        functions.insert(name.clone(), (*dtype, types));
    }
    let mut function_names = functions.keys().collect::<Vec<_>>();
    function_names.sort();
    for name in function_names {
        let (ret, args) = &functions[name];
        text.push_str(&format!(
            "extern {} {}({});\n",
            ctype(*ret),
            name,
            if args.is_empty() {
                "void".into()
            } else {
                args.iter().map(|d| ctype(*d)).collect::<Vec<_>>().join(",")
            }
        ));
    }
    text.push_str("typedef float float4 __attribute__((ext_vector_type(4),aligned(1)));\n");
    text.push_str(&format!("void {entry}(unsigned char *args) {{\n"));
    let mut values = HashMap::new();
    let mut addresses: HashMap<u64, (String, String, String, DType)> = HashMap::new();
    for (slot, u) in &sig.buffers {
        let name = format!("b{}", u.id);
        text.push_str(&format!(
            "  void *{name}=(void*)(unsigned long long)((unsigned long long*)args)[{slot}];\n"
        ));
        values.insert(u.id, name);
    }
    for (slot, u) in &sig.variables {
        let name = format!("v{}", u.id);
        text.push_str(&format!("  long long {name}=((long long*)args)[{slot}];\n"));
        values.insert(u.id, name);
    }
    let mut depth = 1;
    let mut occurrence = 0usize;
    let mut value_scopes = vec![];
    let mut address_scopes = vec![];
    for u in &linear.src {
        let get = |s: &UOp| -> Result<String> {
            values
                .get(&s.id)
                .cloned()
                .ok_or_else(|| format!("{:?} {} used before definition", s.op, s.id))
        };
        let at = |i: usize| -> Result<String> { get(u.src.get(i).ok_or("missing source")?) };
        let indent = "  ".repeat(depth);
        if let Some(slot) = status_slot {
            let reject = format!(
                "{{ ((int*)(unsigned long long)((unsigned long long*)args)[{slot}])[0]=1; return; }}"
            );
            let a = || at(0);
            let b = || at(1);
            let guard = match u.op {
                Ops::Shl | Ops::Shr => Some(format!(
                    "({}<0 || {}>={})",
                    b()?,
                    b()?,
                    if u.dtype() == DType::I32 { 32 } else { 64 }
                )),
                Ops::Div | Ops::Mod if matches!(u.dtype(), DType::I32 | DType::I64) => {
                    Some(format!(
                        "({}==0 || ({}=={} && {}==-1))",
                        b()?,
                        a()?,
                        if u.dtype() == DType::I32 {
                            "(-2147483647-1)"
                        } else {
                            "(-9223372036854775807ll-1)"
                        },
                        b()?
                    ))
                }
                Ops::Cast
                    if u.src[0].dtype() == DType::F32
                        && matches!(u.dtype(), DType::I32 | DType::I64) =>
                {
                    let limit = if u.dtype() == DType::I32 {
                        "2147483648.0f"
                    } else {
                        "9223372036854775808.0f"
                    };
                    Some(format!("!({} >= -{limit} && {} < {limit})", a()?, a()?))
                }
                Ops::Add | Ops::Sub | Ops::Mul if u.dtype() == DType::I64 => {
                    let operation = match u.op {
                        Ops::Add => "add",
                        Ops::Sub => "sub",
                        _ => "mul",
                    };
                    text.push_str(&format!("{indent}long long checked{};\n", u.id));
                    Some(format!(
                        "__builtin_{operation}_overflow({}, {}, &checked{})",
                        a()?,
                        b()?,
                        u.id
                    ))
                }
                Ops::Neg if u.dtype() == DType::I64 => {
                    Some(format!("({}==(-9223372036854775807ll-1))", a()?))
                }
                _ => None,
            };
            if let Some(condition) = guard {
                text.push_str(&format!("{indent}if ({condition}) {reject}\n"));
            }
        }
        let expression = match (&u.op, &u.arg) {
            (Ops::Const, Arg::Float(bits)) => {
                let bits = (bits.value() as f32).to_bits();
                values.insert(u.id, format!("chibi_float({bits}u)"));
                continue;
            }
            (Ops::Const, Arg::I32(v)) => {
                values.insert(u.id, format!("((int){}u)", *v as u32));
                continue;
            }
            (Ops::Const, Arg::Int(v)) => {
                values.insert(u.id, format!("((long long){}ull)", *v as u64));
                continue;
            }
            (Ops::Const, Arg::Bool(v)) => {
                values.insert(u.id, if *v { "1" } else { "0" }.into());
                continue;
            }
            (Ops::Buffer, _) | (Ops::Param, _) | (Ops::Variable, _) => continue,
            (Ops::Bind, _) => at(1)?,
            (Ops::Alloc, Arg::Allocation { dtype, size, .. }) => {
                let name = format!("a{}", u.id);
                text.push_str(&format!(
                    "{indent}{} {name}[{}];\n",
                    ctype(*dtype),
                    size.max(&1)
                ));
                values.insert(u.id, name);
                continue;
            }
            (Ops::After, _) => {
                values.insert(u.id, at(0)?);
                if let Some(addr) = addresses.get(&u.src[0].id).cloned() {
                    addresses.insert(u.id, addr);
                }
                continue;
            }
            (Ops::Index, _) => {
                let dtype = if let Arg::DType(dt) = u.arg {
                    dt
                } else {
                    DType::F32
                };
                addresses.insert(u.id, (at(0)?, at(1)?, at(2)?, dtype));
                values.insert(
                    u.id,
                    format!("(({}*){})+({})", ctype(dtype), at(0)?, at(1)?),
                );
                continue;
            }
            (Ops::Load, _) => {
                let dtype = if let Arg::DType(dt) = u.arg {
                    dt
                } else {
                    DType::F32
                };
                if let Some((base, index, gate, _)) = addresses.get(&u.src[0].id) {
                    format!(
                        "({gate} ? (({}*){base})[{index}] : ({})0)",
                        ctype(dtype),
                        ctype(dtype)
                    )
                } else {
                    format!("(({}*){})[0]", ctype(dtype), at(0)?)
                }
            }
            (Ops::Store, _) => {
                let value = at(1)?;
                if status_slot.is_some_and(
                    |slot| matches!(u.src[0].arg, Arg::Slot { index,.. } if index == slot),
                ) {
                    text.push_str(&format!(
                        "{indent}if (!({value})) {{ ((int*){})[0]=1; return; }}\n",
                        at(0)?
                    ));
                    values.insert(u.id, "0".into());
                    continue;
                }
                if let Some((base, index, gate, dtype)) = addresses.get(&u.src[0].id) {
                    text.push_str(&format!(
                        "{indent}if ({gate}) (({}*){base})[{index}]={value};\n",
                        ctype(*dtype)
                    ));
                } else {
                    let dtype = if let Arg::DType(dt) = u.arg {
                        dt
                    } else {
                        u.src[1].dtype()
                    };
                    text.push_str(&format!(
                        "{indent}(({}*){})[0]={value};\n",
                        ctype(dtype),
                        at(0)?
                    ));
                }
                values.insert(u.id, "0".into());
                continue;
            }
            (Ops::Range, _) => {
                value_scopes.push(values.clone());
                address_scopes.push(addresses.clone());
                let name = format!("r{}", u.id);
                text.push_str(&format!(
                    "{indent}for (long long {name}={}; {name}<{}; {name}++) {{\n",
                    at(0)?,
                    at(1)?
                ));
                values.insert(u.id, name);
                depth += 1;
                continue;
            }
            (Ops::End, _) => {
                if depth == 1 {
                    return Err("extra END".into());
                }
                depth -= 1;
                text.push_str(&format!("{}}}\n", "  ".repeat(depth)));
                values = value_scopes.pop().ok_or("missing scope")?;
                addresses = address_scopes.pop().ok_or("missing address scope")?;
                values.insert(u.id, "0".into());
                continue;
            }
            (Ops::Cast, Arg::DType(DType::Bool)) => format!("({}!=0)", at(0)?),
            (Ops::Cast, Arg::DType(dtype)) => {
                format!("(({})({}))", ctype(*dtype), at(0)?)
            }
            (Ops::Bitcast, Arg::DType(dt)) if u.src[0].dtype() == *dt => at(0)?,
            (Ops::Bitcast, Arg::DType(DType::I32)) => {
                format!("chibi_int(chibi_bits({}))", at(0)?)
            }
            (Ops::Bitcast, Arg::DType(DType::F32)) => {
                format!("chibi_float((unsigned int)({}))", at(0)?)
            }
            (Ops::Shl, _) => {
                if u.dtype() == DType::I32 {
                    format!("chibi_int((unsigned int)({}) << ({}))", at(0)?, at(1)?)
                } else {
                    format!(
                        "((long long)((unsigned long long)({}) << ({})))",
                        at(0)?,
                        at(1)?
                    )
                }
            }
            (Ops::Shr, _) => {
                format!(
                    "{}({},{})",
                    if u.dtype() == DType::I32 {
                        "chibi_shr"
                    } else {
                        "chibi_shr64"
                    },
                    at(0)?,
                    at(1)?
                )
            }
            (Ops::Neg, _) => {
                if u.dtype() == DType::I32 {
                    format!("((int)(0u-(unsigned int)({})))", at(0)?)
                } else {
                    format!("(-({}))", at(0)?)
                }
            }
            (Ops::Exp2, _) => format!("exp2f({})", at(0)?),
            (Ops::Log2, _) => format!("log2f({})", at(0)?),
            (Ops::Sin, _) => format!("sinf({})", at(0)?),
            (Ops::Sqrt, _) => format!("sqrtf({})", at(0)?),
            (Ops::Where, _) => format!("({}?{}:{})", at(0)?, at(1)?, at(2)?),
            (Ops::Max, _) if u.dtype() == DType::F32 => {
                format!("chibi_max({},{})", at(0)?, at(1)?)
            }
            (Ops::Max, _) => format!("({}>{}?{}:{})", at(0)?, at(1)?, at(0)?, at(1)?),
            (Ops::Div, _) if matches!(u.dtype(), DType::I32 | DType::I64) => {
                format!("chibi_div({},{})", at(0)?, at(1)?)
            }
            (Ops::Mod, _) => format!("chibi_mod({},{})", at(0)?, at(1)?),
            (op, _) if is_binary(*op) => {
                let symbol = match op {
                    Ops::Add => "+",
                    Ops::Sub => "-",
                    Ops::Mul => "*",
                    Ops::Div => "/",
                    Ops::Lt => "<",
                    Ops::Eq => "==",
                    Ops::And => "&",
                    Ops::Or => "|",
                    Ops::Xor => "^",
                    _ => return Err("unsupported ALU".into()),
                };
                if u.dtype() == DType::I32 && matches!(op, Ops::Add | Ops::Sub | Ops::Mul) {
                    format!(
                        "((int)((unsigned int)({}) {symbol} (unsigned int)({})))",
                        at(0)?,
                        at(1)?
                    )
                } else {
                    format!("({} {symbol} {})", at(0)?, at(1)?)
                }
            }
            (Ops::Call, _) => {
                let body = u.src.first().ok_or("CALL body missing")?;
                let Arg::Function { name, .. } = &body.arg else {
                    return Err("scalar CALL must target CUSTOM_FUNCTION".into());
                };
                format!(
                    "{}({})",
                    name,
                    u.src[1..]
                        .iter()
                        .map(get)
                        .collect::<Result<Vec<_>>>()?
                        .join(",")
                )
            }
            (Ops::CustomFunction, Arg::Function { name, .. }) => {
                if u.src.is_empty() {
                    values.insert(u.id, name.clone());
                    continue;
                }
                format!(
                    "{}({})",
                    name,
                    u.src.iter().map(get).collect::<Result<Vec<_>>>()?.join(",")
                )
            }
            (Ops::Sink, _) => continue,
            _ => return Err(format!("C renderer does not support {:?}", u.op)),
        };
        if u.dtype() == DType::Void {
            text.push_str(&format!("{indent}{expression};\n"));
            values.insert(u.id, "0".into());
        } else {
            let name = format!("v{}_{}", u.id, occurrence);
            occurrence += 1;
            text.push_str(&format!(
                "{indent}{} {name}={expression};\n",
                ctype(u.dtype())
            ));
            values.insert(u.id, name);
        }
    }
    if depth != 1 {
        return Err("unclosed C loop".into());
    }
    text.push_str("}\n");
    Ok(text)
}

fn pointer_size(u: &UOp) -> Option<usize> {
    match &u.arg {
        Arg::Slot { shape, .. } => Some(numel(shape)),
        Arg::Buffer { shape, .. } => Some(numel(shape)),
        _ => None,
    }
}

fn index_bounds(u: &UOp) -> Option<(i64, i64)> {
    let b = u.bounds()?;
    Some((b.min, b.max))
}
```

`src/dimensions.rs`を保存する。

```rust
// file="src/dimensions.rs" mode="write"
use crate::{
    Result,
    symbolic::{floor_div, floor_mod, simplify},
    uop::{Arg, Ops, UOp, is_binary, is_unary},
};
use std::collections::HashMap;
pub fn product(shape: &[UOp]) -> UOp {
    shape.iter().fold(UOp::int(1), |p, n| p.mul(n))
}
pub fn evaluate(expr: &UOp, vars: &HashMap<String, i64>) -> Result<i64> {
    let expr = simplify(expr)?;
    let mut values: HashMap<u64, i64> = HashMap::new();
    for u in expr.toposort() {
        let arg = |i: usize| {
            values
                .get(&u.src[i].id)
                .copied()
                .ok_or_else(|| "missing symbolic value".to_string())
        };
        let value = match (&u.op, &u.arg) {
            (Ops::Const, Arg::Int(n)) => *n,
            (Ops::Const, Arg::Bool(n)) => *n as i64,
            (Ops::Variable, Arg::Variable { name, min, max }) => {
                let n = *vars
                    .get(name)
                    .ok_or_else(|| format!("unbound variable {name}"))?;
                if n < *min || n > *max {
                    return Err(format!("variable {name} outside {min}..={max}"));
                }
                n
            }
            (Ops::Bind, _) => arg(1)?,
            (Ops::Add, _) => arg(0)?.checked_add(arg(1)?).ok_or("dimension overflow")?,
            (Ops::Sub, _) => arg(0)?.checked_sub(arg(1)?).ok_or("dimension overflow")?,
            (Ops::Mul, _) => arg(0)?.checked_mul(arg(1)?).ok_or("dimension overflow")?,
            (Ops::Div, _) => floor_div(arg(0)?, arg(1)?).ok_or("invalid symbolic division")?,
            (Ops::Mod, _) => floor_mod(arg(0)?, arg(1)?).ok_or("invalid symbolic modulo")?,
            (Ops::Max, _) => arg(0)?.max(arg(1)?),
            (Ops::Lt, _) => (arg(0)? < arg(1)?) as i64,
            (Ops::Eq, _) => (arg(0)? == arg(1)?) as i64,
            (Ops::Where, _) => {
                if arg(0)? != 0 {
                    arg(1)?
                } else {
                    arg(2)?
                }
            }
            _ => return Err(format!("{:?} is not a symbolic integer expression", u.op)),
        };
        values.insert(u.id, value);
    }
    values
        .get(&expr.id)
        .copied()
        .ok_or("empty symbolic expression".into())
}
pub fn bind_shape(shape: &[UOp], vars: &HashMap<String, i64>) -> Result<Vec<usize>> {
    shape
        .iter()
        .map(|n| usize::try_from(evaluate(n, vars)?).map_err(|_| "negative dimension".into()))
        .collect()
}
impl UOp {
    pub fn logical_shape(&self) -> Result<Vec<UOp>> {
        self.logical_shape_cache
            .get_or_init(|| {
                self.infer_logical_shape()
                    .and_then(|dims| dims.iter().map(simplify).collect())
            })
            .clone()
    }
    fn infer_logical_shape(&self) -> Result<Vec<UOp>> {
        let capacity = self.shape()?;
        let constants = || {
            capacity
                .iter()
                .map(|&n| UOp::int(n as i64))
                .collect::<Vec<_>>()
        };
        match (&self.op, &self.arg) {
            (Ops::Buffer, _) if !self.src.is_empty() => {
                if self.src.len() != capacity.len() {
                    return Err("symbolic buffer dimension rank mismatch".into());
                }
                Ok(self.src.clone())
            }
            (Ops::Reshape, _) if self.src.len() > 1 => Ok(self.src[1..].to_vec()),
            (Ops::Reshape, _) => {
                let before = self.src[0].logical_shape()?;
                let old = self.src[0].shape()?;
                let dynamic: Vec<_> = before
                    .iter()
                    .zip(&old)
                    .enumerate()
                    .filter(|(_, (u, n))| u.arg != Arg::Int(**n as i64))
                    .map(|(i, _)| i)
                    .collect();
                let mut dims = constants();
                if dynamic.len() == 1 {
                    let axis = dynamic[0];
                    if let Some(out) = capacity.iter().position(|&n| n == old[axis]) {
                        dims[out] = before[axis].clone();
                    } else if !capacity.is_empty() {
                        let rest: usize = capacity[1..].iter().product();
                        dims[0] = simplify(&product(&before).div(&UOp::int(rest as i64)))?;
                    }
                } else if !dynamic.is_empty() {
                    return Err(
                        "reshape with multiple symbolic dimensions requires reshape_symbolic"
                            .into(),
                    );
                }
                Ok(dims)
            }
            (Ops::Permute, Arg::Axes(axes)) => {
                let dims = self.src[0].logical_shape()?;
                Ok(axes.iter().map(|&a| dims[a].clone()).collect())
            }
            (Ops::Expand, _) => {
                let before = self.src[0].logical_shape()?;
                let old = self.src[0].shape()?;
                Ok(capacity
                    .iter()
                    .enumerate()
                    .map(|(i, &n)| {
                        if old[i] == n {
                            before[i].clone()
                        } else {
                            UOp::int(n as i64)
                        }
                    })
                    .collect())
            }
            (Ops::Pad, Arg::Pairs(pads)) => Ok(self.src[0]
                .logical_shape()?
                .iter()
                .zip(pads)
                .map(|(n, &(a, b))| n.add(&UOp::int((a + b) as i64)))
                .collect()),
            (Ops::Reduce, Arg::Reduction { axes, keepdim, .. }) => Ok(self.src[0]
                .logical_shape()?
                .iter()
                .enumerate()
                .filter_map(|(i, n)| {
                    if axes.contains(&i) {
                        keepdim.then(|| UOp::int(1))
                    } else {
                        Some(n.clone())
                    }
                })
                .collect()),
            (Ops::Stack, Arg::Axis(axis)) => {
                let mut s = self.src[0].logical_shape()?;
                s.insert(*axis, UOp::int(self.src.len() as i64));
                Ok(s)
            }
            (Ops::Gather, Arg::Axis(axis)) => {
                let before = self.src[0].logical_shape()?;
                let mut s = before[..*axis].to_vec();
                s.extend(self.src[1].logical_shape()?);
                s.extend_from_slice(&before[axis + 1..]);
                Ok(s)
            }
            (Ops::ScatterAdd | Ops::Flip | Ops::Stage | Ops::Detach | Ops::After, _) => {
                self.src[0].logical_shape()
            }
            (op, _) if is_unary(*op) => self.src[0].logical_shape(),
            (op, _) if is_binary(*op) || *op == Ops::Where => {
                let mut result = vec![UOp::int(1); capacity.len()];
                for src in &self.src {
                    let dims = src.logical_shape()?;
                    let offset = capacity.len() - dims.len();
                    for (i, dim) in dims.into_iter().enumerate() {
                        if dim != UOp::int(1) {
                            let slot = &mut result[offset + i];
                            if *slot != UOp::int(1) && simplify(slot)? != simplify(&dim)? {
                                return Err("incompatible symbolic broadcast dimensions".into());
                            }
                            *slot = dim;
                        }
                    }
                }
                Ok(result)
            }
            _ => Ok(constants()),
        }
    }
}

pub fn shape_constraints(roots: &[UOp]) -> Result<Vec<(String, UOp)>> {
    let mut checks = vec![];
    for node in crate::schedule::tensor_toposort(&UOp::sink(roots.to_vec())) {
        let mut conditions = vec![];
        match (&node.op, &node.arg) {
            (Ops::Reshape, _) => conditions
                .push(product(&node.src[0].logical_shape()?).eq(&product(&node.logical_shape()?))),
            (Ops::Shrink, Arg::Pairs(pairs)) => {
                for (dim, &(_, end)) in node.src[0].logical_shape()?.iter().zip(pairs) {
                    conditions.push(UOp::int(end as i64).lt(&dim.add(&UOp::int(1))));
                }
            }
            _ => continue,
        }
        for condition in conditions {
            let condition = simplify(&condition)?;
            if condition == UOp::bool(false) {
                return Err(format!("invalid {:?} shape", node.op));
            }
            if condition != UOp::bool(true) {
                checks.push((
                    format!("{:?} shape is invalid for runtime dimensions", node.op),
                    condition,
                ));
            }
        }
    }
    Ok(checks)
}
```

`src/gpu_tensor.rs`を保存する。

```rust
// file="src/gpu_tensor.rs" mode="write"
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::backends::metal::MetalDevice;
#[cfg(feature = "opencl")]
use crate::backends::opencl::OpenClDevice;
use crate::{
    Arg, Axis, Buffer, Ops, Result, Tensor, UOp,
    codegen::{rangeify, signature},
    dimensions::{bind_shape, evaluate, shape_constraints},
    runtime::InputSpec,
    schedule::{Schedule, create_schedule},
    uop::numel,
};
use crate::{
    gpu::{Kernel, Launch},
    queue::{CompiledQueue, HWQueue},
    renderers::{Dialect, render},
};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Metal,
    OpenCl,
}
enum Device {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(MetalDevice),
    #[cfg(feature = "opencl")]
    OpenCl(OpenClDevice),
}
#[derive(Clone)]
enum Allocation {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(crate::backends::metal::MetalBuffer),
    #[cfg(feature = "opencl")]
    OpenCl(crate::backends::opencl::OpenClBuffer),
}
impl Device {
    fn allocate(&self, bytes: usize) -> Result<Allocation> {
        match self {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(d) => d.allocate(bytes.max(4)).map(Allocation::Metal),
            #[cfg(feature = "opencl")]
            Self::OpenCl(d) => d.allocate(bytes.max(4)).map(Allocation::OpenCl),
        }
    }
}
impl Allocation {
    fn retain(&self, queue: &mut HWQueue) -> UOp {
        match self {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(b) => queue.retain_metal_buffer(b.clone()),
            #[cfg(feature = "opencl")]
            Self::OpenCl(b) => queue.retain_opencl_buffer(b.clone()),
        }
    }
    fn transfer(&self, queue: &mut HWQueue, pointer: UOp, bytes: UOp, read: bool) {
        let buffer = self.retain(queue);
        match (self, read) {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            (Self::Metal(_), false) => queue.metal_write(buffer, pointer, bytes),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            (Self::Metal(_), true) => queue.metal_read(buffer, pointer, bytes),
            #[cfg(feature = "opencl")]
            (Self::OpenCl(_), false) => queue.opencl_write(buffer, pointer, bytes),
            #[cfg(feature = "opencl")]
            (Self::OpenCl(_), true) => queue.opencl_read(buffer, pointer, bytes),
        }
    }
}
enum HostBuffer {
    Input(usize),
    Captured(usize),
    Scalars(usize),
    Output(usize),
    Error,
}
pub struct GpuExecutable {
    inputs: Vec<InputSpec>,
    schedule: Schedule,
    programs: Vec<UOp>,
    queue: CompiledQueue,
    input_nodes: Vec<UOp>,
    captured: Vec<Buffer>,
    variables: Vec<Vec<UOp>>,
    transfers: Vec<HostBuffer>,
    sequence: i64,
    shape_checks: Vec<(String, UOp)>,
}

fn transfer(
    queue: &mut HWQueue,
    transfers: &mut Vec<HostBuffer>,
    buffer: &Allocation,
    host: HostBuffer,
    read: bool,
) {
    let index = 2 + transfers.len() * 2;
    transfers.push(host);
    buffer.transfer(
        queue,
        HWQueue::runtime_slot(index),
        HWQueue::runtime_slot(index + 1),
        read,
    );
}

fn parallelize(linear: UOp, single_output: bool) -> (UOp, Launch) {
    let mut launch = Launch {
        groups: [1; 3],
        local: [1; 3],
    };
    let ranges = linear
        .src
        .iter()
        .filter(|u| {
            matches!(
                u.arg,
                Arg::Range {
                    axis: Axis::Loop,
                    ..
                }
            )
        })
        .collect::<Vec<_>>();
    let checked_indices = linear
        .src
        .iter()
        .any(|u| matches!(&u.arg, Arg::Buffer { name, .. } if name == "__chibi_error"));
    if !single_output || ranges.len() != 1 || checked_indices {
        return (linear, launch);
    }
    let range = ranges[0];
    let Arg::Int(extent) = range.src[1].arg else {
        return (linear, launch);
    };
    if extent <= 1 {
        return (linear, launch);
    }
    launch.groups[0] = extent as usize;
    let range_id = range.id;
    let mut replacements = HashMap::new();
    let mut ops = vec![];
    for u in &linear.src {
        let sources = u
            .src
            .iter()
            .map(|s| replacements.get(&s.id).unwrap_or(s).clone())
            .collect();
        let value = if u.id == range_id {
            UOp::new(
                Ops::Range,
                sources,
                Arg::Range {
                    id: 0,
                    axis: Axis::Global,
                },
            )
        } else {
            u.replace(sources)
        };
        replacements.insert(u.id, value.clone());
        ops.push(value);
    }
    (UOp::linear(ops), launch)
}

impl GpuExecutable {
    pub fn inputs(&self) -> &[InputSpec] {
        &self.inputs
    }
    pub fn programs(&self) -> &[UOp] {
        &self.programs
    }
    pub fn schedule(&self) -> &Schedule {
        &self.schedule
    }

    pub fn compile(outputs: &[Tensor], backend: Backend) -> Result<Self> {
        let roots = outputs.iter().map(|t| t.uop.clone()).collect::<Vec<_>>();
        let shape_checks = shape_constraints(&roots)?;
        let schedule = create_schedule(&roots)?;
        let device = match backend {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal => Device::Metal(MetalDevice::open()?),
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            Backend::Metal => return Err("Metal requires macOS".into()),
            #[cfg(not(feature = "opencl"))]
            Backend::OpenCl => return Err("OpenCL feature is disabled".into()),
            #[cfg(feature = "opencl")]
            Backend::OpenCl => Device::OpenCl(OpenClDevice::open(0)?),
        };
        let dialect = match backend {
            Backend::Metal => Dialect::Metal,
            Backend::OpenCl => Dialect::OpenCl,
        };
        let mut queue = HWQueue::new();
        let mut transfers = vec![];
        let mut allocations = HashMap::new();
        let mut sizes = vec![];
        for life in &schedule.lifetimes {
            if sizes.len() <= life.slot {
                sizes.resize(life.slot + 1, 0)
            }
            sizes[life.slot] = sizes[life.slot].max(life.elements);
        }
        let slots = sizes
            .iter()
            .map(|&n| device.allocate(n * 4))
            .collect::<Result<Vec<_>>>()?;
        for life in &schedule.lifetimes {
            allocations.insert(life.buffer.id, slots[life.slot].clone());
        }
        let produced = schedule
            .lifetimes
            .iter()
            .map(|l| l.buffer.id)
            .collect::<HashSet<_>>();
        let mut inputs = vec![];
        let mut input_nodes = vec![];
        let mut captured = vec![];
        for node in schedule
            .buffers
            .iter()
            .filter(|u| !produced.contains(&u.id))
        {
            let Arg::Buffer { name, shape, data } = &node.arg else {
                return Err("invalid input buffer".into());
            };
            let allocation = device.allocate(numel(shape) * 4)?;
            let host = if let Some(bits) = data {
                let i = captured.len();
                captured.push(Buffer::new(
                    shape.clone(),
                    bits.iter().map(|&x| f32::from_bits(x)).collect(),
                )?);
                HostBuffer::Captured(i)
            } else {
                let i = inputs.len();
                inputs.push(InputSpec {
                    name: name.clone(),
                    shape: shape.clone(),
                });
                input_nodes.push(node.clone());
                HostBuffer::Input(i)
            };
            transfer(&mut queue, &mut transfers, &allocation, host, false);
            allocations.insert(node.id, allocation);
        }
        let error = device.allocate(4)?;
        transfer(&mut queue, &mut transfers, &error, HostBuffer::Error, false);
        let mut programs = vec![];
        let mut kernels: Vec<(Kernel, Vec<Allocation>)> = vec![];
        let mut variables = vec![];
        for (i, plan) in schedule.kernels.iter().enumerate() {
            let (linear, launch) = parallelize(rangeify(&plan.sink)?, plan.writes.len() == 1);
            let sig = signature(&linear)?;
            let kernel = render(&linear, &format!("chibi_gpu_{i}"), dialect, launch)?;
            let mut bindings = vec![];
            for (slot, node) in sig.buffers {
                if bindings.len() != slot {
                    return Err("GPU buffer signature has a gap".into());
                }
                let buffer = if matches!(&node.arg, Arg::Buffer { name, .. } if name == "__chibi_error")
                {
                    error.clone()
                } else {
                    allocations
                        .get(&node.id)
                        .ok_or("missing GPU allocation")?
                        .clone()
                };
                bindings.push(buffer);
            }
            let mut ordered = BTreeMap::<String, UOp>::new();
            for (_, node) in sig.variables {
                let Arg::Variable { name, min, max } = &node.arg else {
                    return Err("invalid scalar variable".into());
                };
                let (mut min, mut max) = (*min, *max);
                if let Some(previous) = ordered.get(name) {
                    if let Arg::Variable { min: a, max: b, .. } = previous.arg {
                        min = min.max(a);
                        max = max.min(b);
                    }
                }
                if min > max {
                    return Err(format!("incompatible bounds for variable {name}"));
                }
                ordered.insert(name.clone(), UOp::variable(name, min, max));
            }
            let vars = ordered.into_values().collect::<Vec<_>>();
            if !vars.is_empty() {
                let scalar_buffer = device.allocate(vars.len() * 8)?;
                transfer(
                    &mut queue,
                    &mut transfers,
                    &scalar_buffer,
                    HostBuffer::Scalars(i),
                    false,
                );
                bindings.push(scalar_buffer);
            }
            variables.push(vars);
            programs.push(UOp::new(
                Ops::Program,
                vec![
                    plan.sink.clone(),
                    linear,
                    UOp::new(Ops::Source, vec![], Arg::Text(kernel.source.clone())),
                ],
                Arg::Text(kernel.entry.clone()),
            ));
            kernels.push((kernel, bindings));
        }
        match device {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Device::Metal(device) => {
                let commands = kernels
                    .iter()
                    .map(|(kernel, bindings)| {
                        let buffers = bindings
                            .iter()
                            .map(|b| match b {
                                Allocation::Metal(b) => b.clone(),
                                _ => unreachable!(),
                            })
                            .collect();
                        Ok((device.compile(kernel)?, kernel.launch.clone(), buffers))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let command = queue.retain_metal_command(device.command_buffer(&commands)?);
                for (index, (_, bindings)) in kernels.iter().enumerate() {
                    for (argument, buffer) in bindings.iter().enumerate() {
                        let buffer = buffer.retain(&mut queue);
                        queue.metal_patch(
                            command.clone(),
                            UOp::int(index as i64),
                            UOp::int(argument as i64),
                            buffer,
                            UOp::int(0),
                        );
                    }
                }
                queue.metal_submit(
                    command.clone(),
                    HWQueue::runtime_slot(0),
                    HWQueue::runtime_slot(1),
                );
                queue.metal_wait(command);
            }
            #[cfg(feature = "opencl")]
            Device::OpenCl(device) => {
                for (kernel, bindings) in &kernels {
                    let args = bindings
                        .iter()
                        .map(|b| match b {
                            Allocation::OpenCl(b) => {
                                crate::backends::opencl::Argument::Buffer(b.clone())
                            }
                            _ => unreachable!(),
                        })
                        .collect();
                    let program = device.compile(kernel)?;
                    let command =
                        queue.retain_opencl_command(program.command(kernel.launch.clone(), args)?);
                    queue.opencl_launch(command.clone());
                    queue.opencl_finish(command);
                }
            }
        }
        for (i, output) in schedule.outputs.iter().enumerate() {
            transfer(
                &mut queue,
                &mut transfers,
                &allocations[&output.id],
                HostBuffer::Output(i),
                true,
            );
        }
        transfer(&mut queue, &mut transfers, &error, HostBuffer::Error, true);
        Ok(Self {
            inputs,
            schedule,
            programs,
            queue: queue.compile("chibi_gpu_queue")?,
            input_nodes,
            captured,
            variables,
            transfers,
            sequence: 0,
            shape_checks,
        })
    }
    pub fn kernel_count(&self) -> usize {
        self.programs.len()
    }
    pub fn sources(&self) -> Vec<&str> {
        self.programs
            .iter()
            .filter_map(|p| {
                p.src.iter().find_map(|s| {
                    if let (Ops::Source, Arg::Text(text)) = (s.op, &s.arg) {
                        Some(text.as_str())
                    } else {
                        None
                    }
                })
            })
            .collect()
    }
    pub fn queue_source(&self) -> Option<&str> {
        self.queue.source()
    }
    pub fn run(&mut self, inputs: &[Buffer], vars: &HashMap<String, i64>) -> Result<Vec<Buffer>> {
        for (message, condition) in &self.shape_checks {
            if evaluate(condition, vars)? == 0 {
                return Err(message.clone());
            }
        }
        if inputs.len() != self.inputs.len() {
            return Err(format!(
                "expected {} input buffers, received {}",
                self.inputs.len(),
                inputs.len()
            ));
        }
        for ((buffer, spec), node) in inputs.iter().zip(&self.inputs).zip(&self.input_nodes) {
            let shape = bind_shape(&node.logical_shape()?, vars)?;
            if buffer.shape != shape || buffer.data.len() != numel(&shape) {
                return Err(format!(
                    "input {} expected {:?}, received {:?}",
                    spec.name, shape, buffer.shape
                ));
            }
        }
        let mut scalars = vec![];
        for variables in &self.variables {
            scalars.push(
                variables
                    .iter()
                    .map(|u| {
                        let Arg::Variable { name, min, max } = &u.arg else {
                            return Err("invalid variable".into());
                        };
                        let value = *vars
                            .get(name)
                            .ok_or_else(|| format!("unbound variable {name}"))?;
                        if value < *min || value > *max {
                            return Err(format!("variable {name}={value} outside {min}..={max}"));
                        }
                        Ok(value)
                    })
                    .collect::<Result<Vec<i64>>>()?,
            );
        }
        let mut outputs = self
            .schedule
            .outputs
            .iter()
            .map(|u| {
                let shape = bind_shape(&u.logical_shape()?, vars)?;
                Buffer::new(shape.clone(), vec![0.; numel(&shape)])
            })
            .collect::<Result<Vec<_>>>()?;
        let mut error = [0f32];
        let next = self
            .sequence
            .checked_add(1)
            .ok_or("GPU timeline overflow")?;
        let mut slots = vec![self.sequence, next];
        for host in &self.transfers {
            let (pointer, bytes) = match host {
                HostBuffer::Input(i) => {
                    (inputs[*i].data.as_ptr() as usize, inputs[*i].data.len() * 4)
                }
                HostBuffer::Captured(i) => (
                    self.captured[*i].data.as_ptr() as usize,
                    self.captured[*i].data.len() * 4,
                ),
                HostBuffer::Scalars(i) => (scalars[*i].as_ptr() as usize, scalars[*i].len() * 8),
                HostBuffer::Output(i) => (
                    outputs[*i].data.as_mut_ptr() as usize,
                    outputs[*i].data.len() * 4,
                ),
                HostBuffer::Error => (error.as_mut_ptr() as usize, 4),
            };
            slots.extend([pointer as i64, bytes as i64]);
        }
        self.sequence = next;
        unsafe { self.queue.run(&mut slots) }?;
        if error[0] != 0. {
            return Err("gather/scatter index must be finite, integral, and within bounds".into());
        }
        Ok(outputs)
    }
}
```

`src/lib.rs`を追記する。

```rust
// file="src/lib.rs" mode="append"

pub mod dimensions;
```

`src/runtime.rs`を保存する。

```rust
// file="src/runtime.rs" mode="write"
use crate::codegen::signature;
use crate::{
    Result,
    codegen::{rangeify, render},
    dimensions::{bind_shape, evaluate, shape_constraints},
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
    shape_checks: Vec<(String, UOp)>,
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
    let shape_checks =
        shape_constraints(&outputs.iter().map(|t| t.uop.clone()).collect::<Vec<_>>())?;
    let debug = std::env::var_os("CHIBI_DEBUG").is_some();
    if debug {
        eprintln!("schedule starts");
    }
    let schedule = create_schedule(&outputs.iter().map(|x| x.uop.clone()).collect::<Vec<_>>())?;
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
            captured.insert(
                u.id,
                Buffer::new(
                    shape.clone(),
                    bits.iter().map(|&b| f32::from_bits(b)).collect(),
                )?,
            );
        } else {
            input_nodes.push(u.clone());
            inputs.push(InputSpec {
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
            kernel.sink.clone(),
            linear,
            UOp::new(Ops::Source, vec![], Arg::Text(source.clone())),
        ];

        let stages = {
            if debug {
                eprintln!("kernel {i} compile {} source bytes", source.len());
            }
            let compiled = crate::compiler_cache::compile_cached(&source, &entry)?;
            let mut stages = stages;
            stages.push(UOp::new(
                Ops::Binary,
                vec![],
                Arg::Bytes(compiled.object().to_vec()),
            ));
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
        shape_checks,
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
                if let Arg::Text(text) = &s.arg {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect()
    }
    pub fn run(&self, inputs: &[Buffer], vars: &HashMap<String, i64>) -> Result<Vec<Buffer>> {
        for (message, condition) in &self.shape_checks {
            if evaluate(condition, vars)? == 0 {
                return Err(message.clone());
            }
        }
        if inputs.len() != self.inputs.len() {
            return Err(format!(
                "expected {} input buffers, received {}",
                self.inputs.len(),
                inputs.len()
            ));
        }
        for ((buffer, spec), node) in inputs.iter().zip(&self.inputs).zip(&self.input_nodes) {
            let expected = bind_shape(&node.logical_shape()?, vars)?;
            if buffer.shape != expected || buffer.data.len() != numel(&expected) {
                return Err(format!(
                    "input {} expected {:?}, received {:?} with {} elements",
                    spec.name,
                    expected,
                    buffer.shape,
                    buffer.data.len()
                ));
            }
        }
        let mut external: HashMap<u64, &Buffer> =
            self.captured.iter().map(|(&id, b)| (id, b)).collect();
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
                let count = sig
                    .buffers
                    .iter()
                    .map(|x| x.0)
                    .chain(sig.variables.iter().map(|x| x.0))
                    .max()
                    .map_or(0, |x| x + 1);
                let mut packed = vec![0u8; count * 8];
                let mut validation_status = 0f32;
                for (slot, u) in sig.buffers {
                    let pointer = if matches!(&u.arg, Arg::Buffer {name, data:Some(_), ..} if name=="__chibi_error")
                    {
                        &mut validation_status as *mut f32 as usize
                    } else if let Some(b) = external.get(&u.id) {
                        b.data.as_ptr() as usize
                    } else {
                        let loc = locations.get(&u.id).ok_or("missing runtime allocation")?;
                        slots[*loc].as_mut_ptr() as usize
                    };
                    packed[slot * 8..slot * 8 + 8].copy_from_slice(&(pointer as u64).to_ne_bytes());
                }
                for (slot, u) in sig.variables {
                    let Arg::Variable { name, min, max } = &u.arg else {
                        return Err("invalid variable".into());
                    };
                    let value = vars
                        .get(name)
                        .ok_or_else(|| format!("unbound variable {name}"))?;
                    if value < min || value > max {
                        return Err(format!("variable {name}={value} outside {min}..={max}"));
                    }
                    packed[slot * 8..slot * 8 + 8].copy_from_slice(&value.to_ne_bytes());
                }
                unsafe { self.native[i].run(&mut packed) }?;
                if validation_status != 0. {
                    return Err(
                        "gather/scatter index must be finite, integral, and within bounds".into(),
                    );
                }
            }
            self.schedule
                .outputs
                .iter()
                .map(|u| {
                    let slot = locations[&u.id];
                    let shape = bind_shape(&u.logical_shape()?, vars)?;
                    Buffer::new(shape.clone(), slots[slot][..numel(&shape)].to_vec())
                })
                .collect()
        }
    }
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
                u.logical_shape()?,
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
            u.logical_shape()?,
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

`src/tensor.rs` の `impl Tensor` 内 の `pub fn gradients`を置き換える。

```rust
// file="src/tensor.rs" mode="replace" within="impl Tensor" item="pub fn gradients"
    pub fn gradients(&self, targets: &[Self]) -> Result<Vec<Self>> {
        self.shape()?;
        for target in targets {
            target.shape()?;
        }
        let nodes = self.uop.toposort();
        for node in nodes.iter().chain(targets.iter().map(|target| &target.uop)) {
            if node
                .logical_shape()?
                .iter()
                .any(|dim| !matches!(dim.arg, Arg::Int(_)))
            {
                return Err("gradients require fixed tensor dimensions; symbolic scalar values are supported".into());
            }
        }
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

`src/tensor.rs` の `impl Tensor` 内を追記する。

```rust
// file="src/tensor.rs" mode="append" within="impl Tensor"

    pub fn symbolic_parameter(name: &str, dimensions: &[UOp]) -> Self {
        let shape = dimensions
            .iter()
            .map(|d| {
                d.bounds()
                    .filter(|b| b.min >= 0)
                    .and_then(|b| usize::try_from(b.max).ok())
            })
            .collect::<Option<Vec<_>>>();
        match shape {
            Some(shape) => Self::from_uop(UOp::new(
                Ops::Buffer,
                dimensions.to_vec(),
                Arg::Buffer {
                    name: name.into(),
                    shape,
                    data: None,
                },
            )),
            None => Self::invalid("symbolic dimensions require finite nonnegative bounds"),
        }
    }

    pub fn reshape_symbolic(&self, dimensions: &[UOp]) -> Self {
        let shape = dimensions
            .iter()
            .map(|d| {
                d.bounds()
                    .filter(|b| b.min >= 0)
                    .and_then(|b| usize::try_from(b.max).ok())
            })
            .collect::<Option<Vec<_>>>();
        match shape {
            Some(shape) => {
                let mut src = vec![self.uop.clone()];
                src.extend_from_slice(dimensions);
                Self::from_uop(UOp::new(Ops::Reshape, src, Arg::Shape(shape)))
            }
            None => Self::invalid("invalid symbolic reshape"),
        }
    }

    pub fn logical_shape(&self) -> Result<Vec<UOp>> {
        self.uop.logical_shape()
    }
```

`src/uir.rs` の `fn validate_node`を置き換える。

```rust
// file="src/uir.rs" mode="replace" item="fn validate_node"
fn validate_node(u: &UOp) -> Result<()> {
    if u.op == Ops::Buffer {
        u.shape()?;
        u.logical_shape()?;
        return Ok(());
    }
    if crate::uop::is_binary(u.op) || crate::uop::is_unary(u.op) || u.op == Ops::Where {
        if u.src.iter().all(|s| {
            matches!(
                s.dtype(),
                DType::Bool | DType::I32 | DType::I64 | DType::F32
            ) && s.shape().is_ok_and(|shape| shape.is_empty())
        }) {
            crate::scalar::verify_node(u)?;
        } else {
            u.shape()?;
        }
    }
    if u.op == Ops::Variable {
        return match &u.arg {
            Arg::Variable { min, max, .. } if min <= max && u.src.is_empty() => Ok(()),
            _ => Err("invalid variable".into()),
        };
    }
    if u.op == Ops::Bind {
        return if u.src.len() == 2
            && u.src[0].op == Ops::Variable
            && matches!(u.src[1].arg,Arg::Int(n) if u.src[0].bounds().is_some_and(|b| n>=b.min&&n<=b.max))
        {
            Ok(())
        } else {
            Err("invalid variable binding".into())
        };
    }
    match (&u.op, &u.arg) {
        (Ops::Const, _) => {
            crate::scalar::verify_node(u)?;
        }
        (Ops::Param, Arg::ScalarParam { .. }) => {
            crate::scalar::verify_node(u)?;
        }
        (Ops::Param, Arg::Slot { .. })
        | (Ops::Buffer, Arg::Buffer { .. })
        | (Ops::Alloc, Arg::Allocation { .. })
            if u.src.is_empty() =>
        {
            if u.op == Ops::Buffer {
                u.shape()?;
            }
        }
        (Ops::Param | Ops::Buffer | Ops::Alloc, _) => {
            return Err("invalid leaf metadata".into());
        }
        (Ops::Range, Arg::Range { .. })
            if u.src.len() == 2 && u.src.iter().all(|s| s.dtype() == DType::I64) => {}
        (Ops::Range, _) => return Err("invalid RANGE".into()),
        (Ops::End, Arg::None)
            if !u.src.is_empty() && u.src.last().is_some_and(|r| r.op == Ops::Range) => {}
        (Ops::End, _) => return Err("invalid END".into()),
        (Ops::Source, Arg::Text(_))
        | (Ops::Binary, Arg::Bytes(_))
        | (Ops::Invalid, Arg::Text(_))
            if u.src.is_empty() => {}
        (Ops::Source | Ops::Binary | Ops::Invalid, _) => {
            return Err("invalid source payload".into());
        }
        (Ops::Load, Arg::None | Arg::DType(_))
            if u.src.len() == 1 && u.src[0].dtype() == DType::Ptr => {}
        (Ops::Store, Arg::None | Arg::DType(_))
            if u.src.len() == 2 && u.src[0].dtype() == DType::Ptr => {}
        (Ops::Load | Ops::Store, _) => return Err("invalid memory operation".into()),
        (Ops::Linear, Arg::None) => {
            crate::codegen::verify_lowered(u)?;
            validate_linear_scope(u)?;
        }
        (Ops::Linear, _) => return Err("invalid LINEAR".into()),
        (Ops::Call, Arg::Id(count))
            if u.src.first().is_some_and(|b| b.op == Ops::Stack)
                && u.src.len().checked_sub(1) == Some(*count) => {}
        (Ops::Call, Arg::None) if u.src.first().is_some_and(|b| b.op == Ops::CustomFunction) => {}
        (Ops::CustomFunction, Arg::Function { .. }) => {}
        (Ops::Call, _) => {
            return Err("invalid CALL".into());
        }
        _ => {}
    }
    Ok(())
}
```

`src/uop.rs`を保存する。

```rust
// file="src/uop.rs" mode="write"
use crate::Result;
use std::{
    cell::{OnceCell, RefCell},
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
    ops::Deref,
    rc::{Rc, Weak},
};
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    I32,
    I64,
    Bool,
    Ptr,
    Void,
    F32x4,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ops {
    Invalid,
    Const,
    Buffer,
    Alloc,
    Param,
    Variable,
    Bind,
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
    CustomFunction,
    Special,
    Barrier,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Axis {
    Loop,
    Reduce,
    Global,
    Local,
}
#[derive(Clone, Copy, Debug)]
pub struct ConstFloat(f64);
impl ConstFloat {
    pub fn new(value: f64) -> Self {
        Self(if value.is_nan() {
            f64::from_bits(0x7ff8_0000_0000_0000)
        } else {
            value
        })
    }
    pub fn value(self) -> f64 { self.0 }
}
impl PartialEq for ConstFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 || (self.0.is_nan() && other.0.is_nan())
    }
}
impl Eq for ConstFloat {}
impl Hash for ConstFloat {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let bits = if self.0 == 0.0 { 0 } else { self.0.to_bits() };
        bits.hash(state);
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Arg {
    Variable {
        name: String,
        min: i64,
        max: i64,
    },
    None,
    Float(ConstFloat),
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
    Function {
        name: String,
        dtype: DType,
    },
}
#[derive(Debug)]
pub struct Node {
    pub id: u64,
    pub op: Ops,
    pub src: Vec<UOp>,
    pub arg: Arg,
    shape_cache: OnceCell<Result<Vec<usize>>>,
    pub(crate) logical_shape_cache: OnceCell<Result<Vec<UOp>>>,
    dtype_cache: OnceCell<DType>,
    pub(crate) bounds_cache: OnceCell<Option<(i64, i64)>>,
}
#[derive(Clone, Debug)]
pub struct UOp(Rc<Node>);
#[derive(Clone)]
struct Key {
    op: Ops,
    src: Vec<u64>,
    arg: Arg,
}
impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.op == other.op && self.src == other.src && match (&self.arg, &other.arg) {
            (Arg::Float(a), Arg::Float(b)) => a.value().to_bits() == b.value().to_bits(),
            (a, b) => a == b,
        }
    }
}
impl Eq for Key {}
impl Hash for Key {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.op.hash(state);
        self.src.hash(state);
        std::mem::discriminant(&self.arg).hash(state);
        match &self.arg {
            Arg::Float(value) => value.value().to_bits().hash(state),
            other => other.hash(state),
        }
    }
}
#[derive(Default)]
struct Interner {
    next: u64,
    nodes: HashMap<Key, Weak<Node>>,
}
thread_local! {
    static INTERN : RefCell < Interner > = RefCell::new(Interner::default());
}
impl Node {
    fn evict(&self) {
        let key = Key {
            op: self.op,
            src: self.src.iter().map(|s| s.id).collect(),
            arg: self.arg.clone(),
        };
        let _ = INTERN.try_with(|cell| {
            if let Ok(mut intern) = cell.try_borrow_mut() {
                if intern
                    .nodes
                    .get(&key)
                    .is_some_and(|weak| std::ptr::eq(weak.as_ptr(), self))
                {
                    intern.nodes.remove(&key);
                }
            }
        });
    }
}
impl Drop for Node {
    fn drop(&mut self) {
        self.evict();
        let mut pending = std::mem::take(&mut self.src);
        while let Some(child) = pending.pop() {
            if Rc::strong_count(&child.0) == 1 {
                child.evict();
                if let Ok(mut node) = Rc::try_unwrap(child.0) {
                    pending.append(&mut node.src);
                }
            }
        }
    }
}
impl Deref for UOp {
    type Target = Node;
    fn deref(&self) -> &Node {
        &self.0
    }
}
impl PartialEq for UOp {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for UOp {}
impl Hash for UOp {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}
impl UOp {
    pub fn new(op: Ops, src: Vec<UOp>, arg: Arg) -> Self {
        INTERN.with(|cell| {
            let mut intern = cell.borrow_mut();
            let key = Key {
                op,
                src: src.iter().map(|x| x.id).collect(),
                arg: arg.clone(),
            };
            if let Some(node) = intern.nodes.get(&key).and_then(Weak::upgrade) {
                return Self(node);
            }
            intern.next += 1;
            let node = Rc::new(Node {
                id: intern.next,
                op,
                src,
                arg,
                shape_cache: OnceCell::new(),
                logical_shape_cache: OnceCell::new(),
                dtype_cache: OnceCell::new(),
                bounds_cache: OnceCell::new(),
            });
            intern.nodes.insert(key, Rc::downgrade(&node));
            if intern.nodes.len() % 1024 == 0 {
                intern.nodes.retain(|_, v| v.strong_count() != 0);
            }
            Self(node)
        })
    }
    pub fn weak(&self) -> Weak<Node> {
        Rc::downgrade(&self.0)
    }
    pub fn prune_interner() -> usize {
        INTERN.with(|c| {
            let mut c = c.borrow_mut();
            c.nodes.retain(|_, v| v.strong_count() != 0);
            c.nodes.len()
        })
    }
    pub fn interner_len() -> usize {
        INTERN.with(|cell| cell.borrow().nodes.len())
    }
    pub fn replace(&self, src: Vec<UOp>) -> Self {
        Self::new(self.op, src, self.arg.clone())
    }
    pub fn f32(x: f32) -> Self {
        Self::new(Ops::Const, vec![], Arg::Float(ConstFloat::new(f64::from(x))))
    }
    pub fn i32(x: i32) -> Self {
        Self::new(Ops::Const, vec![], Arg::I32(x))
    }
    pub fn int(x: i64) -> Self {
        Self::new(Ops::Const, vec![], Arg::Int(x))
    }
    pub fn bool(x: bool) -> Self {
        Self::new(Ops::Const, vec![], Arg::Bool(x))
    }
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(Ops::Invalid, vec![], Arg::Text(message.into()))
    }
    pub fn variable(name: &str, min: i64, max: i64) -> Self {
        if min > max {
            return Self::invalid("variable minimum exceeds maximum");
        }
        Self::new(
            Ops::Variable,
            vec![],
            Arg::Variable {
                name: name.into(),
                min,
                max,
            },
        )
    }
    pub fn bind(&self, value: i64) -> Self {
        if let Arg::Variable { min, max, .. } = self.arg {
            if value < min || value > max {
                return Self::invalid("binding lies outside variable bounds");
            }
        }
        Self::new(Ops::Bind, vec![self.clone(), Self::int(value)], Arg::None)
    }
    pub fn alu(&self, op: Ops, others: &[UOp]) -> Self {
        let mut src = vec![self.clone()];
        src.extend_from_slice(others);
        Self::new(op, src, Arg::None)
    }
    pub fn add(&self, x: &Self) -> Self {
        self.alu(Ops::Add, &[x.clone()])
    }
    pub fn mul(&self, x: &Self) -> Self {
        self.alu(Ops::Mul, &[x.clone()])
    }
    pub fn sub(&self, x: &Self) -> Self {
        self.alu(Ops::Sub, &[x.clone()])
    }
    pub fn div(&self, x: &Self) -> Self {
        self.alu(Ops::Div, &[x.clone()])
    }
    pub fn modulo(&self, x: &Self) -> Self {
        self.alu(Ops::Mod, &[x.clone()])
    }
    pub fn lt(&self, x: &Self) -> Self {
        self.alu(Ops::Lt, &[x.clone()])
    }
    pub fn eq(&self, x: &Self) -> Self {
        self.alu(Ops::Eq, &[x.clone()])
    }
    pub fn and(&self, x: &Self) -> Self {
        self.alu(Ops::And, &[x.clone()])
    }
    pub fn bitcast(&self, dt: DType) -> Self {
        Self::new(Ops::Bitcast, vec![self.clone()], Arg::DType(dt))
    }
    pub fn cast(&self, dt: DType) -> Self {
        Self::new(Ops::Cast, vec![self.clone()], Arg::DType(dt))
    }
    pub fn where_(&self, a: &Self, b: &Self) -> Self {
        Self::new(
            Ops::Where,
            vec![self.clone(), a.clone(), b.clone()],
            Arg::None,
        )
    }
    pub fn range(end: UOp, id: usize, axis: Axis) -> Self {
        Self::new(Ops::Range, vec![Self::int(0), end], Arg::Range { id, axis })
    }
    pub fn index(&self, index: UOp, valid: UOp) -> Self {
        Self::new(Ops::Index, vec![self.clone(), index, valid], Arg::None)
    }
    pub fn load(&self) -> Self {
        Self::new(Ops::Load, vec![self.clone()], Arg::None)
    }
    pub fn store(&self, value: UOp) -> Self {
        Self::new(Ops::Store, vec![self.clone(), value], Arg::None)
    }
    pub fn after(&self, deps: Vec<UOp>) -> Self {
        let mut src = vec![self.clone()];
        src.extend(deps);
        Self::new(Ops::After, src, Arg::None)
    }
    pub fn sink(src: Vec<UOp>) -> Self {
        Self::new(Ops::Sink, src, Arg::None)
    }
    pub fn linear(src: Vec<UOp>) -> Self {
        Self::new(Ops::Linear, src, Arg::None)
    }
    pub fn custom(name: &str, dtype: DType, src: Vec<UOp>) -> Self {
        Self::new(
            Ops::CustomFunction,
            src,
            Arg::Function {
                name: name.into(),
                dtype,
            },
        )
    }
    pub fn dtype(&self) -> DType {
        *self.dtype_cache.get_or_init(|| self.infer_dtype())
    }
    fn infer_dtype(&self) -> DType {
        match self.op {
            Ops::Const => match self.arg {
                Arg::Float(_) => DType::F32,
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
            Ops::Variable | Ops::Range | Ops::Special => DType::I64,
            Ops::Lt | Ops::Eq => DType::Bool,
            Ops::Cast | Ops::Bitcast => {
                if let Arg::DType(dt) = self.arg {
                    dt
                } else {
                    DType::Void
                }
            }
            Ops::CustomFunction => {
                if let Arg::Function { dtype, .. } = self.arg {
                    dtype
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

    pub fn shape(&self) -> Result<Vec<usize>> {
        self.shape_cache.get_or_init(|| self.infer_shape()).clone()
    }
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
            (Ops::Const, _)
            | (Ops::Variable, _)
            | (Ops::Bind, _)
            | (Ops::Range, _)
            | (Ops::Load, _)
            | (Ops::CustomFunction, _) => Ok(vec![]),
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
    pub fn toposort(&self) -> Vec<UOp> {
        let mut seen = HashSet::new();
        let mut out = vec![];
        let mut stack = vec![(self.clone(), false)];
        while let Some((u, done)) = stack.pop() {
            if done {
                out.push(u);
                continue;
            }
            if !seen.insert(u.id) {
                continue;
            }
            stack.push((u.clone(), true));
            let sources = u.ordered_sources();
            for s in sources.into_iter().rev() {
                stack.push((s.clone(), false));
            }
        }
        out
    }
    pub fn ordered_sources(&self) -> Vec<&UOp> {
        if self.op == Ops::After && !self.src.is_empty() {
            self.src[1..].iter().chain(self.src[..1].iter()).collect()
        } else {
            self.src.iter().collect()
        }
    }
}
pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}
pub fn checked_numel(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |a, &b| {
        a.checked_mul(b)
            .filter(|&n| n <= i64::MAX as usize)
            .ok_or_else(|| "tensor shape overflow".into())
    })
}
pub fn is_binary(op: Ops) -> bool {
    matches!(
        op,
        Ops::Add
            | Ops::Sub
            | Ops::Mul
            | Ops::Div
            | Ops::Mod
            | Ops::Max
            | Ops::Lt
            | Ops::Eq
            | Ops::And
            | Ops::Or
            | Ops::Xor
            | Ops::Shl
            | Ops::Shr
    )
}
pub fn is_unary(op: Ops) -> bool {
    matches!(
        op,
        Ops::Neg
            | Ops::Exp
            | Ops::Log
            | Ops::Sin
            | Ops::Cos
            | Ops::Sqrt
            | Ops::Exp2
            | Ops::Log2
            | Ops::Cast
            | Ops::Bitcast
    )
}
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let n = a.len().max(b.len());
    (0..n)
        .map(|i| {
            let x = if i + a.len() >= n {
                a[i + a.len() - n]
            } else {
                1
            };
            let y = if i + b.len() >= n {
                b[i + b.len() - n]
            } else {
                1
            };
            if x == y {
                Ok(x)
            } else if x == 1 {
                Ok(y)
            } else if y == 1 {
                Ok(x)
            } else {
                Err(format!("cannot broadcast {a:?} and {b:?}"))
            }
        })
        .collect()
}
```

`tests/symbolic.rs`を保存する。

```rust
// file="tests/symbolic.rs" mode="write"
use chibigrad::{Buffer, Tensor, compile_many};
use std::collections::HashMap;
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() < 1e-4, "{a} != {b}");
    }
}
#[test]
fn dynamic_scalar_binding_and_errors() {
    let x = Tensor::from_slice(&[2., 3.], &[2]);
    let y = x.mul(&Tensor::variable("scale", 1, 4));
    let exe = y.compile().unwrap();
    close(
        &exe.run(&[], &HashMap::from([("scale".into(), 2)])).unwrap()[0].data,
        &[4., 6.],
    );
    assert!(exe.run(&[], &HashMap::new()).is_err());
    assert!(exe.run(&[], &HashMap::from([("scale".into(), 5)])).is_err());
    assert!(x.reshape(&[3]).compile().is_err());
    assert!(x.permute(&[1]).compile().is_err());
}

#[test]
fn symbolic_extents_reuse_one_native_program() {
    use chibigrad::UOp;
    let n = UOp::variable("n", 1, 5);
    let x = Tensor::symbolic_parameter("x", &[n.clone(), UOp::int(2)]);
    let y = x.mul(&Tensor::scalar(3.)).sum_axes(&[1], false);
    let flat = x.reshape_symbolic(&[n.mul(&UOp::int(2))]);
    let exe = compile_many(&[y, flat]).unwrap();
    for len in [1, 3, 5] {
        let data = (0..len * 2).map(|x| x as f32 + 1.).collect::<Vec<_>>();
        let outputs = exe
            .run(
                &[Buffer::new(vec![len, 2], data.clone()).unwrap()],
                &HashMap::from([("n".into(), len as i64)]),
            )
            .unwrap();
        assert_eq!(outputs[0].shape, [len]);
        assert_eq!(
            outputs[0].data,
            data.chunks(2)
                .map(|x| (x[0] + x[1]) * 3.)
                .collect::<Vec<_>>()
        );
        assert_eq!(outputs[1].shape, [len * 2]);
        assert_eq!(outputs[1].data, data);
    }
    assert!(
        exe.run(
            &[Buffer::new(vec![2, 2], vec![1.; 4]).unwrap()],
            &HashMap::from([("n".into(), 3)])
        )
        .is_err()
    );
}

#[test]
fn dynamic_shrink_checks_logical_bounds() {
    use chibigrad::UOp;
    let x = Tensor::symbolic_parameter("x", &[UOp::variable("n", 1, 5)]);
    let exe = x.shrink(&[(0, 4)]).compile().unwrap();
    assert!(
        exe.run(
            &[Buffer::new(vec![3], vec![1.; 3]).unwrap()],
            &HashMap::from([("n".into(), 3)])
        )
        .is_err()
    );
    assert_eq!(
        exe.run(
            &[Buffer::new(vec![5], vec![2.; 5]).unwrap()],
            &HashMap::from([("n".into(), 5)])
        )
        .unwrap()[0]
            .data,
        [2.; 4]
    );
}

#[test]
fn symbolic_extents_reject_gradients_instead_of_returning_capacity_shape() {
    use chibigrad::UOp;
    let x = Tensor::symbolic_parameter("x", &[UOp::variable("n", 1, 5)]);
    let error = x
        .sum()
        .gradient(&x)
        .err()
        .expect("dynamic gradient must be rejected");
    assert!(error.contains("fixed tensor dimensions"), "{error}");
    let fixed = Tensor::parameter("fixed", &[2]);
    let gradient = fixed
        .mul(&Tensor::variable("scale", 1, 4))
        .sum()
        .gradient(&fixed)
        .unwrap();
    let output = gradient
        .compile()
        .unwrap()
        .run(&[], &HashMap::from([("scale".into(), 3)]))
        .unwrap();
    assert_eq!(output[0].shape, [2]);
    assert_eq!(output[0].data, [3., 3.]);
}

#[test]
fn symbolic_buffer_uir_preserves_dimension_graph() {
    let n = chibigrad::UOp::variable("n", 1, 5);
    let x = Tensor::symbolic_parameter("x", &[n.clone(), chibigrad::UOp::int(2)]);
    let text = chibigrad::uir::print(x.uop());
    let parsed = chibigrad::uir::parse(&text).unwrap();
    chibigrad::uir::assert_uop(x.uop(), &parsed).unwrap();
    assert_eq!(parsed.logical_shape().unwrap(), [n, chibigrad::UOp::int(2)]);
}
```

`tests/tensor_gpu.rs`を追記する。

```rust
// file="tests/tensor_gpu.rs" mode="append"

#[test]
fn symbolic_extents_and_runtime_shape_guards() {
    let n = chibigrad::UOp::variable("n", 1, 5);
    let x = Tensor::symbolic_parameter("x", &[n.clone(), chibigrad::UOp::int(2)]);
    let y = x.mul(&Tensor::scalar(3.)).sum_axes(&[1], false);
    let flat = x.reshape_symbolic(&[n.mul(&chibigrad::UOp::int(2))]);
    for backend in backends() {
        let mut exe = GpuExecutable::compile(&[y.clone(), flat.clone()], backend).unwrap();
        let programs = exe.programs().to_vec();
        for len in [1, 3, 5] {
            let values = (0..len * 2).map(|i| i as f32 + 1.).collect::<Vec<_>>();
            let out = exe
                .run(
                    &[Buffer::new(vec![len, 2], values.clone()).unwrap()],
                    &HashMap::from([("n".into(), len as i64)]),
                )
                .unwrap();
            assert_eq!(out[0].shape, [len]);
            assert_eq!(
                out[0].data,
                values
                    .chunks(2)
                    .map(|r| (r[0] + r[1]) * 3.)
                    .collect::<Vec<_>>()
            );
            assert_eq!(out[1].data, values);
            assert_eq!(exe.programs(), programs);
        }
        assert!(
            exe.run(
                &[Buffer::new(vec![2, 2], vec![1.; 4]).unwrap()],
                &HashMap::from([("n".into(), 3)])
            )
            .is_err()
        );
    }
}
```

```sh
# check step=20.02
cargo test --offline
```

```sh
# check requires=metal,opencl
cargo test --offline --features metal,opencl --test tensor_gpu
```

N=1,3,5 を CPU・Metal・OpenCL の同じコンパイル済み計画へ渡し、正しい行和と flatten の shape を得る。入力 shape の不一致、実次元を超える shrink、動的次元の勾配を拒否し、固定次元のスカラー変数付き勾配は [3,3] になる。

## 20.03 N=16 の特殊化と汎用経路を選ぶ

Specializer は汎用の計画を一つ作り、指定した値で Variable を定数へ置き換えた計画も先にコンパイルする。実行時は N が 16 ならその計画、それ以外なら汎用計画を選ぶ。どちらも以前に作ったプログラムを再利用し、実行回数を数える。

`src/lib.rs`を追記する。

```rust
// file="src/lib.rs" mode="append"

pub mod specialization;
```

`src/specialization.rs`を保存する。

```rust
// file="src/specialization.rs" mode="write"
use crate::{Arg, Buffer, Executable, Ops, Result, Tensor, UOp, compile_many};
use std::collections::{BTreeMap, HashMap};

pub struct Specializer {
    generic: Executable,
    variants: BTreeMap<i64, Executable>,
    variable: String,
    pub generic_runs: usize,
    pub specialized_runs: usize,
}
fn bind(root: &UOp, name: &str, value: i64) -> Result<UOp> {
    let mut replacements = HashMap::<u64, UOp>::new();
    for node in root.toposort() {
        let result =
            if let (Ops::Variable, Arg::Variable { name: n, min, max }) = (node.op, &node.arg) {
                if n == name {
                    if value < *min || value > *max {
                        return Err("specialization value outside variable bounds".into());
                    }
                    UOp::int(value)
                } else {
                    node.clone()
                }
            } else {
                node.replace(
                    node.src
                        .iter()
                        .map(|s| replacements[&s.id].clone())
                        .collect(),
                )
            };
        replacements.insert(node.id, result);
    }
    Ok(replacements[&root.id].clone())
}
impl Specializer {
    pub fn compile(outputs: &[Tensor], variable: &str, values: &[i64]) -> Result<Self> {
        let generic = compile_many(outputs)?;
        if !UOp::sink(outputs.iter().map(|t| t.uop.clone()).collect())
            .toposort()
            .iter()
            .any(|u| matches!(&u.arg,Arg::Variable{name,..} if name==variable))
        {
            return Err("specialization variable not present".into());
        }
        let mut variants = BTreeMap::new();
        for &value in values {
            if variants.contains_key(&value) {
                return Err("duplicate specialization value".into());
            }
            let specialized = outputs
                .iter()
                .map(|t| bind(&t.uop, variable, value).map(Tensor::from_uop))
                .collect::<Result<Vec<_>>>()?;
            variants.insert(value, compile_many(&specialized)?);
        }
        Ok(Self {
            generic,
            variants,
            variable: variable.into(),
            generic_runs: 0,
            specialized_runs: 0,
        })
    }
    pub fn run(&mut self, inputs: &[Buffer], vars: &HashMap<String, i64>) -> Result<Vec<Buffer>> {
        let value = *vars
            .get(&self.variable)
            .ok_or("unbound specialization variable")?;
        if let Some(executable) = self.variants.get(&value) {
            let mut bindings = vars.clone();
            bindings.remove(&self.variable);
            let result = executable.run(inputs, &bindings)?;
            self.specialized_runs += 1;
            Ok(result)
        } else {
            let result = self.generic.run(inputs, vars)?;
            self.generic_runs += 1;
            Ok(result)
        }
    }
    pub fn generic(&self) -> &Executable {
        &self.generic
    }
    pub fn variant(&self, value: i64) -> Option<&Executable> {
        self.variants.get(&value)
    }
}
```

`tests/specialization.rs`を保存する。

```rust
// file="tests/specialization.rs" mode="write"
use chibigrad::{Buffer, Ops, Tensor, UOp, specialization::Specializer};
use std::collections::HashMap;
#[test]
fn n16_specialization_and_generic_path_share_semantics() {
    let n = UOp::variable("N", 1, 17);
    let x = Tensor::symbolic_parameter("x", &[n]);
    let y = x.mul(&Tensor::scalar(2.)).add(&Tensor::scalar(1.));
    let mut dispatch = Specializer::compile(&[y], "N", &[16]).unwrap();
    let generic_programs = dispatch.generic().programs().to_vec();
    let specialized_programs = dispatch.variant(16).unwrap().programs().to_vec();
    for n in [1usize, 3, 16, 17] {
        let data = (0..n).map(|i| i as f32).collect::<Vec<_>>();
        let buffer = Buffer::new(vec![n], data.clone()).unwrap();
        let vars = HashMap::from([("N".into(), n as i64)]);
        let expected = dispatch.generic().run(&[buffer.clone()], &vars).unwrap();
        let actual = dispatch.run(&[buffer], &vars).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(
            actual[0].data,
            data.iter().map(|v| v * 2. + 1.).collect::<Vec<_>>()
        );
    }
    assert_eq!((dispatch.generic_runs, dispatch.specialized_runs), (3, 1));
    assert_eq!(dispatch.generic().programs(), generic_programs);
    assert_eq!(
        dispatch.variant(16).unwrap().programs(),
        specialized_programs
    );
    assert!(
        dispatch
            .variant(16)
            .unwrap()
            .programs()
            .iter()
            .all(|p| p.toposort().iter().all(|u| u.op != Ops::Variable))
    );
}
```

```sh
# check step=20.03
cargo test --offline
```

```sh
# check requires=metal,opencl
cargo test --offline --features metal,opencl --test tensor_gpu --test specialization
```

N=1,3,16,17 の出力が汎用経路と一致し、汎用 3 回・特殊化 1 回になる。N=16 のプログラムから Variable が消え、実行前後の両プログラムが同一である。

## 章末の累積確認

前の章までのテストとexampleも再ビルドします。

```sh
# check
cargo test --quiet --all-targets
```

## 要件との対応

### 20-001 VarSpec と Bindings を追加し、未束縛、二重指定、範囲外を起動前に拒否する。

<!-- todo:20-001 -->

関連する実装と確認は本章の `20.01`, `20.02`, `20.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 20-002 02と03の固定 ABI の scalar slot を使い、VarId から slot への対応を追加する。

<!-- todo:20-002 -->

関連する実装と確認は本章の `20.01`, `20.02`, `20.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 20-003 `RANGE(N)` を C の実行時ループ上限へ出す。

<!-- todo:20-003 -->

関連する実装と確認は本章の `20.01`, `20.02`, `20.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 20-004 容量17の buffer に N=1、3、17を渡し、毎回異なる論理長の結果を返す。

<!-- todo:20-004 -->

関連する実装と確認は本章の `20.01`, `20.02`, `20.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 20-005 純粋な元グラフを置換せず、束縛値を runtime の実行記録に置く。

<!-- todo:20-005 -->

関連する実装と確認は本章の `20.01`, `20.02`, `20.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 20-006 `Dim::Expr(NodeId)` を必要な次元式だけに追加し、`2*N` を式のまま運ぶ。

<!-- todo:20-006 -->

関連する実装と確認は本章の `20.01`, `20.02`, `20.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 20-007 `sum(x)` の縮約 RANGE に N を使い、初期値を実行ごとに設定する。

<!-- todo:20-007 -->

関連する実装と確認は本章の `20.01`, `20.02`, `20.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 20-008 `reshape([N,2])` の必要要素数と入力の論理要素数を実行時に検証する。

<!-- todo:20-008 -->

関連する実装と確認は本章の `20.01`, `20.02`, `20.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 20-009 同じ変数を使った broadcast は許可し、別変数の等値は Bindings で確認する。

<!-- todo:20-009 -->

関連する実装と確認は本章の `20.01`, `20.02`, `20.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 20-010 動的 shrink と pad の有効範囲を guard として INDEX に渡す。

<!-- todo:20-010 -->

関連する実装と確認は本章の `20.01`, `20.02`, `20.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 20-011 確定できないアクセスを安全とみなさず、guard を生成するか未対応として拒否する。

<!-- todo:20-011 -->

関連する実装と確認は本章の `20.01`, `20.02`, `20.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 20-012 Metal の引数領域に N を置き、論理長から group 数を決める。

<!-- todo:20-012 -->

関連する実装と確認は本章の `20.01`, `20.02`, `20.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 20-013 `i<N` を LOAD と STORE の両方へ伝え、容量だけを guard に使わない。

<!-- todo:20-013 -->

関連する実装と確認は本章の `20.01`, `20.02`, `20.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 20-014 local reduction では無効 lane に0を与え、全 lane を barrier に到達させる。

<!-- todo:20-014 -->

関連する実装と確認は本章の `20.01`, `20.02`, `20.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 20-015 OpenCL の scalar 引数と NDRange を同じ束縛値で設定する。

<!-- todo:20-015 -->

関連する実装と確認は本章の `20.01`, `20.02`, `20.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 20-016 GPU に送る値と host で検証した値が同じになるよう、一回の Bindings を固定する。

<!-- todo:20-016 -->

関連する実装と確認は本章の `20.01`, `20.02`, `20.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 20-017 generic kernel の key に VarId の正規化情報、上下限、制約、ABI を含める。

<!-- todo:20-017 -->

関連する実装と確認は本章の `20.01`, `20.02`, `20.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 20-018 generic kernel の key に毎回の N の値を混ぜず、1、3、17で同じコードを再利用する。

<!-- todo:20-018 -->

関連する実装と確認は本章の `20.01`, `20.02`, `20.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 20-019 `N=16` を条件とする特化版を一つ追加し、条件不成立なら generic に戻す。

<!-- todo:20-019 -->

関連する実装と確認は本章の `20.01`, `20.02`, `20.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 20-020 特化 key には N=16 と tile 条件を含め、N=17でその binary を選ばないようにする。

<!-- todo:20-020 -->

関連する実装と確認は本章の `20.01`, `20.02`, `20.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 20-021 添字式の memo key に node、出力座標、RANGE の scope を含めるか、scope 内に cache を閉じる。

<!-- todo:20-021 -->

関連する実装と確認は本章の `20.01`, `20.02`, `20.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 20-022 N の下限、上限、その外側、未束縛で期待どおり成功またはエラーになる。

<!-- todo:20-022 -->

関連する実装と確認は本章の `20.01`, `20.02`, `20.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 20-023 整数の小例は期待値と完全一致する。

<!-- todo:20-023 -->

関連する実装と確認は本章の `20.01`, `20.02`, `20.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 20-024 f32 の generic、specialized、静的版は `abs_err <= 1e-5 + 1e-5*abs(ref)` を初期基準として比較する。

<!-- todo:20-024 -->

関連する実装と確認は本章の `20.01`, `20.02`, `20.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 20-025 RANGE の上限と buffer の容量を区別して dump に表示する。

<!-- todo:20-025 -->

関連する実装と確認は本章の `20.01`, `20.02`, `20.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 20-026 動的 reshape の積、GEMM の K、broadcast の等値を起動前に検証する。

<!-- todo:20-026 -->

関連する実装と確認は本章の `20.01`, `20.02`, `20.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 20-027 Plan を連続再実行しても、束縛値と累積値が前回から残らない。

<!-- todo:20-027 -->

関連する実装と確認は本章の `20.01`, `20.02`, `20.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 20-028 同じ node を転置前後で利用して、添字 cache が座標を混同しない。

<!-- todo:20-028 -->

関連する実装と確認は本章の `20.01`, `20.02`, `20.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 20-029 GPU での失敗は CPU の成功で置き換えず、backend 別に確認結果を残す。

<!-- todo:20-029 -->

関連する実装と確認は本章の `20.01`, `20.02`, `20.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。
