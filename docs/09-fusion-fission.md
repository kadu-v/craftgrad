# 09. 融合と分割で閉じた kernel graph を作り、Plan を再実行する

Graph::compile が consumer 数、STAGE、reduction を見て materialize 位置を選ぶ。定数は出力以外では inline する。Plan は閉じた kernel の順序と binding 署名を持ち、入力値をキャッシュしない。実体化する kernel の出力だけを確保し、融合で消えた broadcast product をホストに確保しない。公開 dump metadata の変更は seal の検証で実行前に拒否する。

`file=... mode=write` はそのファイルの全置換、`mode=append` は末尾への追記である。すべて前章でできたプロジェクトの root で行う。`sh check` は実際の検証コマンドであり、ソースファイルには保存しない。

## 1. 元グラフから consumer 数を集める。
<!-- todo:09-001 -->

`src/graph.rs` の同名関数全体を置き換えます。08章の非融合の選択を、consumer 数、STAGE、縮約に応じた選択へ変更します。

```rust file=src/graph.rs mode=replace item="pub fn kernel_bodies("
pub fn kernel_bodies(graph: &Graph, outputs: &[NodeId], fuse: bool) -> Result<Vec<KernelBody>> {
	graph.verify(outputs)?;
	let order = graph.topology(outputs)?;
	let mut users = vec![0; graph.nodes.len()];
	for &id in &order {
		for input in graph.nodes[id].op.sources() {
			users[input] += 1
		}
	}
	let material: HashSet<_> = order
		.iter()
		.copied()
		.filter(|&id| {
			outputs.contains(&id)
				|| (!matches!(graph.nodes[id].op, Op::Const { .. }) && (!fuse || users[id] > 1))
				|| matches!(
					graph.nodes[id].op,
					Op::Input { .. }
						| Op::Stage { .. } | Op::Reduce { .. }
						| Op::Gather { .. }
						| Op::ScatterAdd { .. }
				)
		})
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


```rust file=tests/fusion_plan.rs mode=write
use chinigrad::{Binary, Buffer, Graph, Reduce, Result, Unary};
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
fn arithmetic_fusion() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let one = g.constant(1.0, vec![])?;
	let two = g.constant(2.0, vec![])?;
	let a = g.binary(Binary::Add, x, one)?;
	let y = g.binary(Binary::Mul, a, two)?;
	let p = g.compile(&[y])?;
	assert_eq!(p.kernel_count, 1);
	assert_eq!(p.run(&[buffer(&[3], &[1., 2., 3.])])?[0].data, [4., 6., 8.]);
	let staged = g.stage(a)?;
	let z = g.binary(Binary::Mul, staged, two)?;
	assert_eq!(g.compile(&[z])?.kernel_count, 2);
	let three = g.constant(3., vec![])?;
	let z = g.binary(Binary::Add, a, three)?;
	assert_eq!(g.compile(&[y, z])?.kernel_count, 3);
	let result = compare(&g, &[y, z], &[buffer(&[3], &[1., 2., 3.])])?;
	assert_eq!(result[1].data, [5., 6., 7.]);
	Ok(())
}
#[test]
fn reductions_and_rebinding() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![4])?;
	let w = g.input(1, vec![4])?;
	let one = g.constant(1., vec![])?;
	let p = g.binary(Binary::Mul, x, w)?;
	let p = g.binary(Binary::Add, p, one)?;
	let y = g.reduce(p, Reduce::Sum, vec![0], false)?;
	let p = g.compile(&[y])?;
	assert_eq!(p.kernel_count, 1);
	assert_eq!(
		p.run(&[
			buffer(&[4], &[1., 2., 3., 4.]),
			buffer(&[4], &[2., 3., 4., 5.])
		])?[0]
			.data,
		[44.]
	);
	assert_eq!(
		p.run(&[
			buffer(&[4], &[1., 2., 3., 4.]),
			buffer(&[4], &[1., 1., 1., 1.])
		])?[0]
			.data,
		[14.]
	);
	Ok(())
}
#[test]
fn dynamic_metadata() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![3])?;
	let two = g.constant(2., vec![])?;
	let y = g.binary(Binary::Mul, x, two)?;
	let p = g.compile(&[y])?;
	let shapes = vec![vec![2], vec![], vec![2]];
	assert_eq!(
		p.run_with_shapes(&[buffer(&[4], &[3., 4., 9., 9.])], &shapes)?[0].data,
		[6., 8.]
	);
	assert!(
		chinigrad::Plan::compile_specialized(&g, &[y])?
			.run_with_shapes(&[buffer(&[4], &[3., 4., 9., 9.])], &shapes)
			.is_err()
	);
	Ok(())
}
#[test]
fn compiled_plan_metadata_is_sealed() -> Result<()> {
	let mut g = Graph::default();
	let x = g.input(0, vec![2])?;
	let y = g.unary(Unary::Neg, x)?;
	let mut plan = g.compile(&[y])?;
	plan.graph.nodes[x].shape = vec![1];
	assert!(plan.run(&[buffer(&[1], &[1.])]).is_err());
	Ok(())
}
```

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test fusion_plan
```

## 2. 単一消費先の ADD と MUL の境界を materialize しない。
<!-- todo:09-002 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test fusion_plan
```

## 3. lower で同じ出力座標を producer へ伝える。
<!-- todo:09-003 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test fusion_plan
```

## 4. `(x+1)*2` を単一 kernel で `[4,6,8]` にする。
<!-- todo:09-004 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test fusion_plan
```

## 5. 前章の非融合経路とも出力を比較する。
<!-- todo:09-005 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test fusion_plan
```

## 6. `sum(x*w+1)` の reduction 座標を要素演算へ伝える。
<!-- todo:09-006 -->

`tests/fusion_plan.rs` の `reductions_and_rebinding` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test fusion_plan reductions_and_rebinding
```

## 7. reduction の外側出力と内側 k のスコープを区別する。
<!-- todo:09-007 -->

`tests/fusion_plan.rs` の `reductions_and_rebinding` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test fusion_plan reductions_and_rebinding
```

## 8. LOAD、MUL、ADD、accumulator 更新を同じ body に作る。
<!-- todo:09-008 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test fusion_plan
```

## 9. `x=[1,2,3,4]`, `w=[2,3,4,5]` の結果を `44` にする。
<!-- todo:09-009 -->

`tests/fusion_plan.rs` の `reductions_and_rebinding` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test fusion_plan reductions_and_rebinding
```

## 10. 入力側の一時配列がなく、一つの reduction kernel になることを確認する。
<!-- todo:09-010 -->

`tests/fusion_plan.rs` の `reductions_and_rebinding` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test fusion_plan reductions_and_rebinding
```

## 11. 明示 STAGE を非融合境界として扱う。
<!-- todo:09-011 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test fusion_plan
```

## 12. producer の出力を ALLOC、STORE、AFTER に変える。
<!-- todo:09-012 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test fusion_plan
```

## 13. consumer が新しい PARAM から保存済みの値を読むようにする。
<!-- todo:09-013 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test fusion_plan
```

## 14. `(STAGE(x+1))*2` を二 kernel で `[4,6,8]` にする。
<!-- todo:09-014 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test fusion_plan
```

## 15. `t=x+1; y=t*2; z=t+3` を三 kernel で実行する。
<!-- todo:09-015 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test fusion_plan
```

## 16. 各 body の外部 storage と scalar を PARAM に正規化する。
<!-- todo:09-016 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test fusion_plan
```

## 17. kernel の署名、CALL 引数、出力 state を照合する verifier を追加する。
<!-- todo:09-017 -->

`tests/fusion_plan.rs` の `dynamic_metadata` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test fusion_plan dynamic_metadata
```

## 18. producer と consumer の辺から実行順を作り、循環なら拒否する。
<!-- todo:09-018 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test fusion_plan
```

## 19. 既存の CPU compile と run を Plan の kernel 列へ接続する。
<!-- todo:09-019 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test fusion_plan
```

## 20. 同じ Plan に新しい重みを渡して、二回目の出力を確認する。
<!-- todo:09-020 -->

`tests/fusion_plan.rs` の `reductions_and_rebinding` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test fusion_plan reductions_and_rebinding
```

## 21. body の未束縛 PARAM、自由 RANGE、STORE 先の漏れを拒否する。
<!-- todo:09-021 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test fusion_plan
```

## 22. 外側 kernel の NodeId が body に直接残った例を拒否する。
<!-- todo:09-022 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test fusion_plan
```

## 23. 二つの違う座標で同じ producer を使う fixture を実行する。
<!-- todo:09-023 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test fusion_plan
```

## 24. 元グラフを run の前後で dump し、構造が変わらないことを確認する。
<!-- todo:09-024 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test fusion_plan
```

## 25. 入力の長さと dtype が署名と違う場合、実行開始前に拒否する。
<!-- todo:09-025 -->

`tests/fusion_plan.rs` の `dynamic_metadata` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test fusion_plan dynamic_metadata
```

## 26. chain は一 kernel、STAGE は二 kernel、fanout は三 kernel になる。
<!-- todo:09-026 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test fusion_plan
```

## 27. 非融合と融合の値が `[4,6,8]` で一致する。
<!-- todo:09-027 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test fusion_plan
```

## 28. fanout の二出力が `[4,6,8]` と `[5,6,7]` になる。
<!-- todo:09-028 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test fusion_plan
```

## 29. 同じ Plan の再実行で `44` から `14` へ変わり、再コンパイルしない。
<!-- todo:09-029 -->

`tests/fusion_plan.rs` の `reductions_and_rebinding` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test fusion_plan reductions_and_rebinding
```

## 30. 閉じた body に未束縛の storage と RANGE が残らない。
<!-- todo:09-030 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test fusion_plan
```

## 31. dump で各 consumer が producer 完了後の state を読む。
<!-- todo:09-031 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test fusion_plan
```

## 32. 純粋な元グラフが run 前後で同じ構造を保つ。
<!-- todo:09-032 -->

`tests/fusion_plan.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test fusion_plan
```

## 章末の累積確認

前章までの example と test も同じ checkout で再ビルドする。

```sh check
cargo test --quiet --all-targets
```
