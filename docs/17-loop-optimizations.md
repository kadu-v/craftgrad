# 17. loop split、unroll、vector、CPU worker を実行する

この章は前章までの `chinigrad` に追加する。掲載順に `file=` のパスへ保存する。`mode=write` はそのファイル全体を置き換え、`mode=append` は末尾へ追記する。既存の別ファイルは残す。各コードは省略なしの実装である。

CPU の実行結果と GPU の実機結果を別に記録する。`requires=metal` と `requires=opencl` の確認は、その device がある環境で実行する。機器がなければ `NOT_RUN` と記録し、source 検査の成功へ置き換えない。

出力一要素の計算が一つの所有区間に入る。worker を作る前に C の compile を完了する。各 kernel の全 worker を join した後、次の kernel を始める。vector の対象は同じ shape の contiguous 入力二つの加算である。その他は理由を表示して scalar に戻す。

<a id="loop-implementation"></a>

## 実装を保存する

```rust file=src/lib.rs mode=append
pub mod loop_opt;
```

```rust file=src/loop_opt.rs mode=write
use crate::{
	Binary, Graph, Op, Plan, Result,
	graph::{C_HELPERS, kernel_bodies},
};
#[derive(Clone, Copy, Debug)]
pub struct Options {
	pub tile: i64,
	pub unroll: usize,
	pub vector: bool,
	pub workers: usize,
	pub interchange: bool,
}
impl Default for Options {
	fn default() -> Self {
		Self {
			tile: 64,
			unroll: 1,
			vector: false,
			workers: 1,
			interchange: false,
		}
	}
}
#[derive(Clone, Copy, Debug)]
pub struct Capability {
	pub vector_width: usize,
	pub cpu_workers: bool,
	pub local: bool,
	pub shared_bytes: usize,
}
pub fn capability(backend: &str) -> Result<Capability> {
	match backend {
		"cpu" => Ok(Capability {
			vector_width: 4,
			cpu_workers: true,
			local: false,
			shared_bytes: 0,
		}),
		"metal" | "opencl" => Ok(Capability {
			vector_width: 4,
			cpu_workers: false,
			local: true,
			shared_bytes: 0,
		}),
		_ => Err("unknown backend".into()),
	}
}
impl Options {
	pub fn validate(&self) -> Result<()> {
		if self.tile <= 0 || self.unroll == 0 || self.workers == 0 || self.workers > 256 {
			return Err("tile/unroll/workers must be positive; workers <=256".into());
		}
		if self.unroll > self.tile as usize {
			return Err("unroll exceeds tile".into());
		}
		(self.tile as usize)
			.checked_add(self.unroll)
			.ok_or("tile index overflow")?;
		Ok(())
	}
}
fn outer(body: &str) -> Result<(String, String)> {
	let begin = body.find('{').ok_or("outer loop")? + 1;
	let end = body.rfind('}').ok_or("outer scope")?;
	let count = body
		.split("i<")
		.nth(1)
		.and_then(|s| s.split("; ++i)").next())
		.ok_or("outer count")?;
	Ok((count.into(), body[begin..end].into()))
}
fn vector_add(g: &Graph, id: usize) -> Option<(usize, usize)> {
	match g.nodes[id].op {
		Op::Binary {
			op: Binary::Add,
			a,
			b,
		} if matches!(g.nodes[a].op, Op::Input { .. })
			&& matches!(g.nodes[b].op, Op::Input { .. })
			&& g.shape(a) == g.shape(b)
			&& g.shape(a) == g.shape(id) =>
		{
			Some((a, b))
		}
		_ => None,
	}
}
pub fn compile(g: &Graph, outputs: &[usize], o: Options) -> Result<(Plan, String)> {
	let kernels = kernel_bodies(g, outputs, true)?;
	compile_bodies(g, outputs, o, kernels)
}
pub fn compile_bodies(
	g: &Graph,
	outputs: &[usize],
	o: Options,
	kernels: Vec<crate::graph::KernelBody>,
) -> Result<(Plan, String)> {
	o.validate()?;
	let mut source = C_HELPERS.to_string();
	source.push_str("#include <pthread.h>\ntypedef float chibi_v4 __attribute__((ext_vector_type(4)));\ntypedef struct {void*const* buffers;const uint64_t*s;size_t begin,end;int status;} Worker;\n");
	let mut calls = String::new();
	let mut report = String::new();
	for (serial, k) in kernels.iter().enumerate() {
		let (count, body) = outer(&k.body)?;
		let mut pointers = String::new();
		for id in g.topology(&[k.output])? {
			pointers += &format!("float*b{id}=(float*)w->buffers[{id}];\n");
		}
		let mut work = String::new();
		if o.vector
			&& !o.interchange
			&& let Some((a, b)) = vector_add(g, k.output)
		{
			work += &format!(
				"size_t i=w->begin;for(;w->end-i>=4;i+=4){{chibi_v4 x,y,z;memcpy(&x,b{a}+i,16);memcpy(&y,b{b}+i,16);z=x+y;memcpy(b{}+i,&z,16);}}for(;i<w->end;i++){{{body}}}",
				k.output
			);
			report += "vector=4 (unaligned memcpy); ";
		}
		if work.is_empty() {
			if o.vector {
				report += "vector=scalar fallback (noncontiguous or unsupported operation); ";
			}
			let index = if o.interchange {
				let shape = g.shape(k.output);
				if shape.len() != 2 {
					return Err("interchange needs two independent output axes".into());
				}
				format!(
					"size_t i=(flat%{})*{}+flat/{};",
					shape[0], shape[1], shape[0]
				)
			} else {
				"size_t i=flat;".into()
			};
			work += &format!(
				"for(size_t outer=w->begin;outer<w->end;){{size_t limit=w->end-outer<{tile}?w->end-outer:{tile};for(size_t inner=0;inner<limit;inner+={unroll}){{",
				tile = o.tile,
				unroll = o.unroll
			);
			for lane in 0..o.unroll {
				work += &format!(
					"{{size_t off=inner+{lane};if(off<limit){{size_t flat=outer+off;{index}{body}}}}}"
				);
			}
			work += "}outer+=limit;}";
		}
		work = work.replace("return 2;", "{w->status=2;return NULL;}");
		source += &format!(
			"static void* kernel{serial}(void*arg){{Worker*w=(Worker*)arg;const uint64_t*s=w->s;{pointers}{work}return NULL;}}\n"
		);
		let workers = o.workers;
		calls += &format!(
			"{{size_t count={count};pthread_t threads[{workers}];Worker args[{workers}];size_t created=0;int failure=0;for(size_t w=0;w<{workers};w++){{size_t q=count/{workers},r=count%{workers};args[w]=(Worker){{buffers,s,q*w+(w<r?w:r),q*(w+1)+((w+1)<r?w+1:r),0}};if(pthread_create(&threads[w],NULL,kernel{serial},&args[w])){{failure=3;break;}}created++;}}for(size_t w=0;w<created;w++){{if(pthread_join(threads[w],NULL))failure=4;if(args[w].status)failure=args[w].status;}}if(failure)return failure;}}\n"
		);
	}
	source +=
		&format!("int chibi_entry(void*const*buffers,const uint64_t*s){{{calls}return 0;}}\n");
	report += &format!(
		"tile={} unroll={} workers={} output_interchange={}",
		o.tile, o.unroll, o.workers, o.interchange
	);
	Ok((
		Plan::compile_source_specialized(g, outputs, source)?,
		report,
	))
}
pub fn interchange_axes(output_axes: &[usize], reduction_axes: &[usize]) -> Result<Vec<usize>> {
	if output_axes.iter().any(|a| reduction_axes.contains(a)) {
		return Err("reduction-axis reordering is disabled".into());
	}
	let mut axes = output_axes.to_vec();
	axes.reverse();
	Ok(axes)
}
```

```rust file=tests/loop_opt.rs mode=write
use chinigrad::{
	Binary, Buffer, Graph, Reduce,
	loop_opt::{Options, compile, interchange_axes},
};
#[test]
fn transforms_real_compiled_graph() {
	let mut g = Graph::default();
	let a = g.input(0, vec![2, 129]).unwrap();
	let b = g.input(1, vec![2, 129]).unwrap();
	let y = g.binary(Binary::Add, a, b).unwrap();
	let data = vec![
		Buffer::new(vec![2, 129], (0..258).map(|i| i as f32).collect()).unwrap(),
		Buffer::new(vec![2, 129], vec![2.; 258]).unwrap(),
	];
	let expected = g.interpret(&[y], &data).unwrap();
	for workers in [1, 2, 3, 300] {
		let workers = workers.min(256);
		for (unroll, vector, interchange) in [
			(1, false, false),
			(4, false, false),
			(1, true, false),
			(3, false, true),
		] {
			let (p, report) = compile(
				&g,
				&[y],
				Options {
					tile: 63,
					unroll,
					vector,
					workers,
					interchange,
				},
			)
			.unwrap();
			assert_eq!(p.run(&data).unwrap(), expected, "{report}");
		}
	}
	assert!(
		compile(
			&g,
			&[y],
			Options {
				tile: 0,
				..Options::default()
			}
		)
		.is_err()
	);
	assert!(interchange_axes(&[0, 1], &[1]).is_err());
}
#[test]
fn reduction_private_accumulators() {
	let mut g = Graph::default();
	let a = g.input(0, vec![7, 65]).unwrap();
	let y = g.reduce(a, Reduce::Sum, vec![1], false).unwrap();
	let data = [Buffer::new(vec![7, 65], vec![1.; 455]).unwrap()];
	let (p, _) = compile(
		&g,
		&[y],
		Options {
			workers: 3,
			unroll: 3,
			..Options::default()
		},
	)
	.unwrap();
	assert_eq!(p.run(&data).unwrap()[0].data, vec![65.; 7]);
}
#[test]
fn more_workers_than_outputs() {
	let mut g = Graph::default();
	let x = g.input(0, vec![1]).unwrap();
	let y = g.constant(2., vec![]).unwrap();
	let z = g.binary(Binary::Add, x, y).unwrap();
	let (p, _) = compile(
		&g,
		&[z],
		Options {
			workers: 3,
			..Options::default()
		},
	)
	.unwrap();
	assert_eq!(
		p.run(&[Buffer::new(vec![1], vec![1.]).unwrap()]).unwrap()[0].data,
		[3.]
	);
}
```

```rust file=tests/book_17.rs mode=write
mod suite0 {
	use chinigrad::{
		Binary, Buffer, Graph, Reduce,
		loop_opt::{Options, compile, interchange_axes},
	};
	#[test]
	pub(super) fn transforms_real_compiled_graph() {
		let mut g = Graph::default();
		let a = g.input(0, vec![2, 129]).unwrap();
		let b = g.input(1, vec![2, 129]).unwrap();
		let y = g.binary(Binary::Add, a, b).unwrap();
		let data = vec![
			Buffer::new(vec![2, 129], (0..258).map(|i| i as f32).collect()).unwrap(),
			Buffer::new(vec![2, 129], vec![2.; 258]).unwrap(),
		];
		let expected = g.interpret(&[y], &data).unwrap();
		for workers in [1, 2, 3, 300] {
			let workers = workers.min(256);
			for (unroll, vector, interchange) in [
				(1, false, false),
				(4, false, false),
				(1, true, false),
				(3, false, true),
			] {
				let (p, report) = compile(
					&g,
					&[y],
					Options {
						tile: 63,
						unroll,
						vector,
						workers,
						interchange,
					},
				)
				.unwrap();
				assert_eq!(p.run(&data).unwrap(), expected, "{report}");
			}
		}
		assert!(
			compile(
				&g,
				&[y],
				Options {
					tile: 0,
					..Options::default()
				}
			)
			.is_err()
		);
		assert!(interchange_axes(&[0, 1], &[1]).is_err());
	}
	#[test]
	pub(super) fn reduction_private_accumulators() {
		let mut g = Graph::default();
		let a = g.input(0, vec![7, 65]).unwrap();
		let y = g.reduce(a, Reduce::Sum, vec![1], false).unwrap();
		let data = [Buffer::new(vec![7, 65], vec![1.; 455]).unwrap()];
		let (p, _) = compile(
			&g,
			&[y],
			Options {
				workers: 3,
				unroll: 3,
				..Options::default()
			},
		)
		.unwrap();
		assert_eq!(p.run(&data).unwrap()[0].data, vec![65.; 7]);
	}
	#[test]
	pub(super) fn more_workers_than_outputs() {
		let mut g = Graph::default();
		let x = g.input(0, vec![1]).unwrap();
		let y = g.constant(2., vec![]).unwrap();
		let z = g.binary(Binary::Add, x, y).unwrap();
		let (p, _) = compile(
			&g,
			&[z],
			Options {
				workers: 3,
				..Options::default()
			},
		)
		.unwrap();
		assert_eq!(
			p.run(&[Buffer::new(vec![1], vec![1.]).unwrap()]).unwrap()[0].data,
			[3.]
		);
	}
}
```

## `i=outer*tile+inner` への置換を kernel IR に追加する。

<!-- todo:17-001 -->

実装は[この章の `loop_opt::compile、Options::validate、interchange_axes`](#loop-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_17.rs mode=append
#[test]
fn todo_17_001() {
	suite0::transforms_real_compiled_graph();
}
```

```sh check
cargo test --test book_17 todo_17_001 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## outer の範囲を切上げ、元範囲の guard を作る。

<!-- todo:17-002 -->

実装は[この章の `loop_opt::compile、Options::validate、interchange_axes`](#loop-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_17.rs mode=append
#[test]
fn todo_17_002() {
	suite0::reduction_private_accumulators();
}
```

```sh check
cargo test --test book_17 todo_17_002 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 無効な要素の LOAD と STORE を guard の内側に置く。

<!-- todo:17-003 -->

実装は[この章の `loop_opt::compile、Options::validate、interchange_axes`](#loop-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_17.rs mode=append
#[test]
fn todo_17_003() {
	suite0::transforms_real_compiled_graph();
}
```

```sh check
cargo test --test book_17 todo_17_003 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## tile=0、負値、添字の overflow を起動前に拒否する。

<!-- todo:17-004 -->

実装は[この章の `loop_opt::compile、Options::validate、interchange_axes`](#loop-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_17.rs mode=append
#[test]
fn todo_17_004() {
	suite0::reduction_private_accumulators();
}
```

```sh check
cargo test --test book_17 todo_17_004 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## split 前と split 後を同じ入力で実行する。

<!-- todo:17-005 -->

実装は[この章の `loop_opt::compile、Options::validate、interchange_axes`](#loop-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_17.rs mode=append
#[test]
fn todo_17_005() {
	suite0::transforms_real_compiled_graph();
}
```

```sh check
cargo test --test book_17 todo_17_005 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 独立した出力軸の順序を入れ替える変換を追加する。

<!-- todo:17-006 -->

実装は[この章の `loop_opt::compile、Options::validate、interchange_axes`](#loop-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_17.rs mode=append
#[test]
fn todo_17_006() {
	suite0::reduction_private_accumulators();
}
```

```sh check
cargo test --test book_17 todo_17_006 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## reduction 軸の再順序化は初版では明示的に禁止する。

<!-- todo:17-007 -->

実装は[この章の `loop_opt::compile、Options::validate、interchange_axes`](#loop-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_17.rs mode=append
#[test]
fn todo_17_007() {
	suite0::transforms_real_compiled_graph();
}
```

```sh check
cargo test --test book_17 todo_17_007 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 小さい inner RANGE を定数座標ごとに複製する。

<!-- todo:17-008 -->

実装は[この章の `loop_opt::compile、Options::validate、interchange_axes`](#loop-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_17.rs mode=append
#[test]
fn todo_17_008() {
	suite0::reduction_private_accumulators();
}
```

```sh check
cargo test --test book_17 todo_17_008 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## RANGE とローカル変数を複製先の scope ごとに作り直す。

<!-- todo:17-009 -->

実装は[この章の `loop_opt::compile、Options::validate、interchange_axes`](#loop-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_17.rs mode=append
#[test]
fn todo_17_009() {
	suite0::transforms_real_compiled_graph();
}
```

```sh check
cargo test --test book_17 todo_17_009 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 展開後にも tail guard と累積の初期化位置を維持する。

<!-- todo:17-010 -->

実装は[この章の `loop_opt::compile、Options::validate、interchange_axes`](#loop-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_17.rs mode=append
#[test]
fn todo_17_010() {
	suite0::reduction_private_accumulators();
}
```

```sh check
cargo test --test book_17 todo_17_010 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## UPCAST 軸を固定個数の scalar 値へ展開する。

<!-- todo:17-011 -->

実装は[この章の `loop_opt::compile、Options::validate、interchange_axes`](#loop-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_17.rs mode=append
#[test]
fn todo_17_011() {
	suite0::transforms_real_compiled_graph();
}
```

```sh check
cargo test --test book_17 todo_17_011 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 連続した width=4 の f32 を vector 型へまとめる変換を別に追加する。

<!-- todo:17-012 -->

実装は[この章の `loop_opt::compile、Options::validate、interchange_axes`](#loop-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_17.rs mode=append
#[test]
fn todo_17_012() {
	suite0::reduction_private_accumulators();
}
```

```sh check
cargo test --test book_17 todo_17_012 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## vector LOAD/STORE の合法性を調べ、misalignment には安全な方法を使う。

<!-- todo:17-013 -->

実装は[この章の `loop_opt::compile、Options::validate、interchange_axes`](#loop-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_17.rs mode=append
#[test]
fn todo_17_013() {
	suite0::transforms_real_compiled_graph();
}
```

```sh check
cargo test --test book_17 todo_17_013 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 端数は scalar loop に残し、full vector の範囲外 LOAD を避ける。

<!-- todo:17-014 -->

実装は[この章の `loop_opt::compile、Options::validate、interchange_axes`](#loop-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_17.rs mode=append
#[test]
fn todo_17_014() {
	suite0::reduction_private_accumulators();
}
```

```sh check
cargo test --test book_17 todo_17_014 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## vector 非対応 backend は scalar へ戻す。

<!-- todo:17-015 -->

実装は[この章の `loop_opt::compile、Options::validate、interchange_axes`](#loop-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_17.rs mode=append
#[test]
fn todo_17_015() {
	suite0::transforms_real_compiled_graph();
}
```

```sh check
cargo test --test book_17 todo_17_015 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## CPU runtime に固定数の worker と join を追加する。

<!-- todo:17-016 -->

実装は[この章の `loop_opt::compile、Options::validate、interchange_axes`](#loop-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_17.rs mode=append
#[test]
fn todo_17_016() {
	suite0::reduction_private_accumulators();
}
```

```sh check
cargo test --test book_17 todo_17_016 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 出力区間を重複と欠落がないように分割する。

<!-- todo:17-017 -->

実装は[この章の `loop_opt::compile、Options::validate、interchange_axes`](#loop-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_17.rs mode=append
#[test]
fn todo_17_017() {
	suite0::transforms_real_compiled_graph();
}
```

```sh check
cargo test --test book_17 todo_17_017 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 区間の begin/end と引数表を worker ごとに持つ。

<!-- todo:17-018 -->

実装は[この章の `loop_opt::compile、Options::validate、interchange_axes`](#loop-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_17.rs mode=append
#[test]
fn todo_17_018() {
	suite0::reduction_private_accumulators();
}
```

```sh check
cargo test --test book_17 todo_17_018 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## worker が同じ compiler やキャッシュへ同時に書かないよう、compile を先に完了する。

<!-- todo:17-019 -->

実装は[この章の `loop_opt::compile、Options::validate、interchange_axes`](#loop-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_17.rs mode=append
#[test]
fn todo_17_019() {
	suite0::transforms_real_compiled_graph();
}
```

```sh check
cargo test --test book_17 todo_17_019 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## thread 数 1、2、3、出力数より多い場合を比較する。

<!-- todo:17-020 -->

実装は[この章の `loop_opt::compile、Options::validate、interchange_axes`](#loop-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_17.rs mode=append
#[test]
fn todo_17_020() {
	suite0::reduction_private_accumulators();
}
```

```sh check
cargo test --test book_17 todo_17_020 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## backend ごとに vector 幅、CPU thread、LOCAL、shared memory の可否を明示する。

<!-- todo:17-021 -->

実装は[この章の `loop_opt::compile、Options::validate、interchange_axes`](#loop-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_17.rs mode=append
#[test]
fn todo_17_021() {
	suite0::transforms_real_compiled_graph();
}
```

```sh check
cargo test --test book_17 todo_17_021 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 未対応変換を黙って別の意味で実行しない。

<!-- todo:17-022 -->

実装は[この章の `loop_opt::compile、Options::validate、interchange_axes`](#loop-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_17.rs mode=append
#[test]
fn todo_17_022() {
	suite0::reduction_private_accumulators();
}
```

```sh check
cargo test --test book_17 todo_17_022 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## split、unroll、vector、thread を単独で検証した後に組み合わせる。

<!-- todo:17-023 -->

実装は[この章の `loop_opt::compile、Options::validate、interchange_axes`](#loop-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_17.rs mode=append
#[test]
fn todo_17_023() {
	suite0::transforms_real_compiled_graph();
}
```

```sh check
cargo test --test book_17 todo_17_023 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## compile、転送、実行、同期を別に測れる計測点だけを追加する。

<!-- todo:17-024 -->

実装は[この章の `loop_opt::compile、Options::validate、interchange_axes`](#loop-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_17.rs mode=append
#[test]
fn todo_17_024() {
	suite0::reduction_private_accumulators();
}
```

```sh check
cargo test --test book_17 todo_17_024 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## split と unroll の後にも scope と buffer 境界の verifier が通る。

<!-- todo:17-025 -->

実装は[この章の `loop_opt::compile、Options::validate、interchange_axes`](#loop-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_17.rs mode=append
#[test]
fn todo_17_025() {
	suite0::transforms_real_compiled_graph();
}
```

```sh check
cargo test --test book_17 todo_17_025 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 未対応 vector の fallback と CPU thread 数 1 が基準版と一致する。

<!-- todo:17-026 -->

実装は[この章の `loop_opt::compile、Options::validate、interchange_axes`](#loop-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_17.rs mode=append
#[test]
fn todo_17_026() {
	suite0::reduction_private_accumulators();
}
```

```sh check
cargo test --test book_17 todo_17_026 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 出力ごとの所有区間が明確で、reduction の累積を共有しない。

<!-- todo:17-027 -->

実装は[この章の `loop_opt::compile、Options::validate、interchange_axes`](#loop-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_17.rs mode=append
#[test]
fn todo_17_027() {
	suite0::transforms_real_compiled_graph();
}
```

```sh check
cargo test --test book_17 todo_17_027 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## GPU local と CPU worker の capability を独立して表示する。

<!-- todo:17-028 -->

実装は[この章の `loop_opt::compile、Options::validate、interchange_axes`](#loop-implementation) にある。次の確認を同じファイルへ追記する。

```rust file=tests/book_17.rs mode=append
#[test]
fn todo_17_028() {
	suite0::reduction_private_accumulators();
}
```

```sh check
cargo test --test book_17 todo_17_028 -- --exact
```

assertion がすべて通れば、この入力の期待値と拒否条件を確認できる。

## 確認記録

```sh check
cargo test
```

