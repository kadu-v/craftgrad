# 19. GPUのGEMM・畳込みとcompile cacheを追加する

前の節で動かしたコードに、今回使う演算と処理を追加します。各節の確認を実行してから次へ進みます。

参照するUOpの構成は [tinygrad c22edff](https://github.com/tinygrad/tinygrad/blob/c22edff14c79310a9b2df62a78819cf77c8289fa/tinygrad/uop/ops.py) です。最初から全機能を実装せず、使う節で増やします。

## 19.01 二つのshared tileでGEMMを実行する

入力行列のtileを共有領域に読み、barrierで揃えてから積和を蓄積する。M・N・Kの端数をLOADとSTOREの条件で処理する。

`src/gpu.rs`を追記する。

```rust
// file="src/gpu.rs" mode="append"


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
```

`tests/gpu_gemm.rs`を保存する。

```rust
// file="tests/gpu_gemm.rs" mode="write"
#![cfg(any(all(feature="metal",target_os="macos"),feature="opencl"))]
mod support;
use chibigrad::{gpu,UOp};
fn check(run:fn(&UOp,gpu::Launch,&[Vec<f32>])->Vec<f32>) {
 let (m,n,k)=(5,7,9);
 let a=(0..m*k).map(|i|i as f32/10.).collect::<Vec<_>>();
 let b=(0..k*n).map(|i|i as f32/13.).collect::<Vec<_>>();
 let (linear,launch)=gpu::tiled_gemm(m,n,k,4).unwrap();
 let result=run(&linear,launch,&[a.clone(),b.clone(),vec![0.;m*n]]);
 for i in 0..m {for j in 0..n {let expected=(0..k).map(|r|a[i*k+r]*b[r*n+j]).sum::<f32>();assert!((result[i*n+j]-expected).abs()<1e-3);}}
}

#[cfg(all(feature="metal",target_os="macos"))]
#[test]
fn metal_native(){check(support::run_metal);}
#[cfg(feature="opencl")]
#[test]
fn opencl_native(){check(support::run_cl);}
```

```sh
# check step=19.01 requires=metal,opencl
cargo test --offline --features metal,opencl --test gpu_gemm -- --nocapture
```

5×9と9×7の積が独立したCPUの三重ループと一致する。

## 19.02 NCHW畳込みをGPUカーネルへ組み立てる

11章のTensor合成との比較用に、各出力位置でフィルタ領域を積和するUOpを組む。出力の端数laneをマスクして同じGPU実行経路へ渡す。

`src/gpu.rs`を追記する。

```rust
// file="src/gpu.rs" mode="append"


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

`tests/gpu_conv.rs`を保存する。

```rust
// file="tests/gpu_conv.rs" mode="write"
#![cfg(any(feature="metal",feature="opencl"))]
mod support;
use chibigrad::{gpu,UOp};
fn check(run:fn(&UOp,gpu::Launch,&[Vec<f32>])->Vec<f32>){
    let (linear,launch)=gpu::convolution(1,1,3,3,1,2,2,4).unwrap();
    let result=run(&linear,launch,&[(1..=9).map(|n|n as f32).collect(),vec![1.,0.,0.,-1.],vec![0.;4]]);
    assert_eq!(result,vec![-4.;4]);
}
#[cfg(feature="metal")] #[test] fn metal_native(){check(support::run_metal);}
#[cfg(feature="opencl")] #[test] fn opencl_native(){check(support::run_cl);}
```

```sh
# check step=19.02 requires=metal,opencl
cargo test --offline --features metal,opencl --test gpu_conv -- --nocapture
```

3×3入力と2×2フィルタの出力4要素がすべて-4になる。

## 19.03 コンパイラの出力objectを再利用する

C sourceとCC設定をキーにしてELF objectを保存する。cache hitでも実行領域をロードし直し、実行時の入力はcacheへ含めない。

`src/compiler_cache.rs`を保存する。

```rust
// file="src/compiler_cache.rs" mode="write"
use crate::{Result, cpu::CpuProgram};
use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    ffi::OsString,
};

#[derive(Clone, PartialEq, Eq, Hash)]
struct Key {
    source: String,
    compiler: OsString,
}
#[derive(Default)]
struct Cache {
    objects: HashMap<Key, Vec<u8>>,
    order: VecDeque<Key>,
    compiles: usize,
    hits: usize,
}
thread_local! {static CACHE:RefCell<Cache>=RefCell::new(Cache::default());}

pub fn compile_cached(source: &str, entry: &str) -> Result<CpuProgram> {
    let key = Key {
        source: source.into(),
        compiler: std::env::var_os("CC").unwrap_or_else(|| "clang".into()),
    };
    let cached = CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let bytes = cache.objects.get(&key).cloned();
        if bytes.is_some() {
            cache.hits += 1;
        }
        bytes
    });
    if let Some(bytes) = cached {
        return CpuProgram::load(&bytes, entry);
    }
    let program = CpuProgram::compile(source, entry)?;
    CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        cache.compiles += 1;
        while cache.order.len() >= 128 {
            if let Some(old) = cache.order.pop_front() {
                cache.objects.remove(&old);
            }
        }
        cache.order.push_back(key.clone());
        cache.objects.insert(key, program.object().to_vec());
    });
    Ok(program)
}
pub fn stats() -> (usize, usize) {
    CACHE.with(|cache| {
        let cache = cache.borrow();
        (cache.compiles, cache.hits)
    })
}
pub fn clear() {
    CACHE.with(|cache| *cache.borrow_mut() = Cache::default());
}
```

`src/lib.rs`を追記する。

```rust
// file="src/lib.rs" mode="append"

pub mod compiler_cache;
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

`tests/compiler_cache.rs`を保存する。

```rust
// file="tests/compiler_cache.rs" mode="write"
use chibigrad::compiler_cache::{clear, compile_cached, stats};
#[test]
fn cached_objects_reload_and_use_fresh_runtime_arguments() {
    clear();
    let source = "void cache_entry(void *args){float *out=*(float**)args;out[0]=out[1]*2;}";
    for (input, expected) in [(3.0f32, 6.0f32), (7., 14.)] {
        let program = compile_cached(source, "cache_entry").unwrap();
        let mut data = [0.0f32, input];
        let mut args = (data.as_mut_ptr() as u64).to_ne_bytes();
        unsafe { program.run(&mut args) }.unwrap();
        assert_eq!(data[0], expected);
    }
    assert_eq!(stats(), (1, 1));
    assert!(compile_cached(source, "missing_entry").is_err());
}
```

```sh
# check step=19.03
cargo test --offline --test compiler_cache
```

同じsourceを二回要求するとコンパイル1回・cache hit1回になり、別々の実行時入力から正しい値を返す。

## 章末の累積確認

前の章までのテストとexampleも再ビルドします。

```sh
# check
cargo test --quiet --all-targets
```

## 要件との対応

### 19-001 `i,j,k` の逐次 GEMM を比較用の設定で呼び出せるようにする。

<!-- todo:19-001 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-002 `k=ko*TK+ki` とする変換を入れ、端数は `k<K` で守る。

<!-- todo:19-002 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-003 `M,N` のタイル分割を追加し、最初は各タイル内の計算順を変えない。

<!-- todo:19-003 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-004 小さな register 累積配列を C に出し、初期化を K ループの外に置く。

<!-- todo:19-004 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-005 出力タイルの並列化は17の CPU thread 経路を使い、同じ出力要素を複数 thread が書かないようにする。

<!-- todo:19-005 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-006 Metal で A の `TM*TK`、B の `TK*TN` の local buffer を確保する。

<!-- todo:19-006 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-007 lane ごとの協調 LOAD を作り、範囲外は LOAD を実行せず local にゼロを保存する。

<!-- todo:19-007 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-008 保存後の barrier と、計算後に local buffer を再利用する前の barrier を入れる。

<!-- todo:19-008 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-009 無効な出力 lane も barrier を通し、最後の global STORE だけを guard する。

<!-- todo:19-009 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-010 同じ IR を OpenCL に出し、`groups` と `local` の積を実際の global size とする。

<!-- todo:19-010 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-011 まず B の local buffer のみを転置し、協調 LOAD と計算時 INDEX の両方を変更する。

<!-- todo:19-011 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-012 local pitch の padding を一つ追加し、確保 byte 数と INDEX の最大値を検証する。

<!-- todo:19-012 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-013 vector LOAD は連続性、alignment、範囲を確認できる内側だけに適用する。

<!-- todo:19-013 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-014 conv の出力座標と縮約座標を GEMM と同じタイルへ写像する。

<!-- todo:19-014 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-015 まず stride=1、padding=0、groups=1 の direct conv を扱い、次に padding と stride を追加する。

<!-- todo:19-015 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-016 Plan の各 kernel が読む範囲と書く範囲を storage base 単位で集計する。

<!-- todo:19-016 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-017 入力、出力、学習パラメータ、KV cache、外部へ返した view を再利用対象から外す。

<!-- todo:19-017 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-018 単一 queue の完了順を前提に、中間 buffer の最初の利用と最後の利用を求める。

<!-- todo:19-018 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-019 まず同じ device、dtype、サイズ、alignment の確保だけを free list で再利用する。

<!-- todo:19-019 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-020 同じ kernel 内で読まれる buffer を、その kernel の出力に割り当てない。

<!-- todo:19-020 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-021 cache key に backend、device capability、生成 source、clang 等の compiler 識別、flags、ABI version を含める。

<!-- todo:19-021 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-022 shape 特化、tile、dtype、math 設定が source 以外で決まる場合も key に含める。

<!-- todo:19-022 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-023 cache hit の binary と cold compile の binary で、同じ数値結果を確認する。

<!-- todo:19-023 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-024 Plan の入力スロットに違う A、B、重みを渡し、コードを再利用して結果が変わることを確かめる。

<!-- todo:19-024 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-025 計測記録に compile、確保、host/device 転送、kernel、同期、全体の区間を分けて出す。

<!-- todo:19-025 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-026 浮動小数点 ALU の回数にループの反復数と launch の担当数を掛ける FLOP 見積りを追加する。

<!-- todo:19-026 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-027 数学的に必要な仕事と、padding した tile で実行する仕事の見積りを別に出す。

<!-- todo:19-027 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-028 整数値の小例は上記の期待値と一致する。

<!-- todo:19-028 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-029 固定 seed の f32 入力は scalar f64 reference に対し `abs_err <= 1e-4 + 1e-4*abs(ref)` を初期基準とする。

<!-- todo:19-029 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-030 大きい K では誤差を記録し、加算順の差を理由に無条件で許容値を広げない。

<!-- todo:19-030 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-031 `M,N,K=1` と tile 前後の寸法、ゼロ入力、負値、非連続 view を試す。

<!-- todo:19-031 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-032 dump で全 lane の barrier 到達、local buffer 内の添字、出力の単独書込みを確認する。

<!-- todo:19-032 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-033 再利用を無効にした結果、再利用した結果、cache hit の結果が一致する。

<!-- todo:19-033 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-034 まだ測っていない backend を「高速化済み」「実機確認済み」と書かない。

<!-- todo:19-034 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 19-035 FLOP の見積りが上記の20と24になり、tile の端数を増やしても有効仕事の数は変わらない。

<!-- todo:19-035 -->

関連する実装と確認は本章の `19.01`, `19.02`, `19.03` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。
