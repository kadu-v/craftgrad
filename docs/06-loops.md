# 06. RANGE と END から安全なループを生成する

RANGE と END は `Stmt::Range { body }` の開始と閉じ括弧で表す。交差した END や未終了 RANGE はこの木では構築できない。軸の番号の重複とスコープ外参照は別に検査する。body に出入りすると interpreter の軸環境も更新する。空の RANGE は LOAD を実行しない。

`file=... mode=write` はそのファイルの全置換、`mode=append` は末尾への追記である。すべて前章でできたプロジェクトの root で行う。`sh check` は実際の検証コマンドであり、ソースファイルには保存しない。

## 1. RANGE と END を検証、interpreter、C renderer へ追加する。
<!-- todo:06-001 -->

```rust file=tests/loops.rs mode=write
use chinigrad::{
	Result,
	kernel::{Expr as E, Index as I, Kernel, range, store},
};

#[test]
fn loops_and_scope() -> Result<()> {
	for len in [0, 1, 3] {
		let mut k = Kernel::default();
		let a = k.input(0, len)?;
		let b = k.input(1, len)?;
		let out = k.alloc(len)?;
		k.body = vec![range(
			0,
			len,
			vec![store(
				out,
				I::Axis(0),
				E::Load(a, I::Axis(0)).add(E::Load(b, I::Axis(0))),
			)],
		)];
		k.outputs = vec![out];
		let inputs = [vec![2.; len], vec![10.; len]];
		assert_eq!(k.compile()?.run(&inputs)?, vec![vec![12.; len]]);
		assert_eq!(k.interpret(&inputs)?, vec![vec![12.; len]]);
	}
	let mut k = Kernel::default();
	let out = k.alloc(1)?;
	k.body = vec![store(out, I::Axis(0), E::Const(1.))];
	assert!(k.compile().is_err());
	Ok(())
}
#[test]
fn nested_loops_effect_calls_and_rejections() -> Result<()> {
	let mut k = Kernel::default();
	let x = k.input(0, 6)?;
	let out = k.alloc(6)?;
	let index = I::Axis(0).mul(I::Const(3)).add(I::Axis(1));
	k.body = vec![range(
		0,
		2,
		vec![range(
			1,
			3,
			vec![store(out, index.clone(), E::Load(x, index))],
		)],
	)];
	k.outputs = vec![out];
	assert_eq!(
		k.compile()?.run(&[vec![1., 2., 3., 4., 5., 6.]])?,
		vec![vec![1., 2., 3., 4., 5., 6.]]
	);
	let mut caller = Kernel::default();
	let input = caller.input(0, 6)?;
	let results = caller.inline_effect(&k, &[input])?;
	caller.outputs = results;
	assert_eq!(
		caller.compile()?.run(&[vec![1., 2., 3., 4., 5., 6.]])?,
		vec![vec![1., 2., 3., 4., 5., 6.]]
	);
	assert!(x.shrink(2, 7).is_err());
	let mut bad = k.clone();
	bad.body = vec![store(out, I::Const(0), E::Load(x, I::Const(6)))];
	assert!(bad.compile().is_err());
	bad.body = vec![store(x, I::Const(0), E::Const(1.))];
	assert!(bad.compile().is_err());
	bad.body = vec![range(0, 1, vec![range(0, 1, vec![])])];
	assert!(bad.compile().is_err());
	bad.body = vec![store(
		out,
		I::Const(i32::MAX).add(I::Const(1)),
		E::Const(1.),
	)];
	assert!(bad.compile().is_err());
	Ok(())
}
```

`tests/loops.rs` の `nested_loops_effect_calls_and_rejections` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test loops nested_loops_effect_calls_and_rejections
```

## 2. body に入ると添字を束縛し、END でスコープを閉じる。
<!-- todo:06-002 -->

`tests/loops.rs` の `nested_loops_effect_calls_and_rejections` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test loops nested_loops_effect_calls_and_rejections
```

## 3. LOAD と STORE の範囲を RANGE bounds から証明する。
<!-- todo:06-003 -->

`tests/loops.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test loops
```

## 4. 三要素加算を既存の固定 ABI で実行する。
<!-- todo:06-004 -->

`tests/loops.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test loops
```

## 5. 第五章の固定添字の例も引き続き通す。
<!-- todo:06-005 -->

`tests/loops.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test loops
```

## 6. builder に外側 body 内で内側 RANGE を作る操作を加える。
<!-- todo:06-006 -->

`tests/loops.rs` の `nested_loops_effect_calls_and_rejections` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test loops nested_loops_effect_calls_and_rejections
```

## 7. END が内側、外側の順に閉じることを検証する。
<!-- todo:06-007 -->

`tests/loops.rs` の `nested_loops_effect_calls_and_rejections` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test loops nested_loops_effect_calls_and_rejections
```

## 8. `offset = row*3+col` を作り、各反復で LOAD と STORE を行う。
<!-- todo:06-008 -->

`tests/loops.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test loops
```

## 9. `[1,2,3,4,5,6]` のコピーが同じ六要素を返すようにする。
<!-- todo:06-009 -->

`tests/loops.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test loops
```

## 10. RANGE 0 と RANGE 1 を interpreter と C で実行する。
<!-- todo:06-010 -->

`tests/loops.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test loops
```

## 11. 空の body が入力を一度も LOAD しないようにする。
<!-- todo:06-011 -->

`tests/loops.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test loops
```

## 12. 空出力には有効なホスト所有物を用意し、ポインタを dereference しない。
<!-- todo:06-012 -->

`tests/loops.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test loops
```

## 13. 一要素加算 `2+10` を生成コードから `12` として読む。
<!-- todo:06-013 -->

`tests/loops.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test loops
```

## 14. 閉じた RANGE の値を別の body で使うグラフを拒否する。
<!-- todo:06-014 -->

`tests/loops.rs` の `nested_loops_effect_calls_and_rejections` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test loops nested_loops_effect_calls_and_rejections
```

## 15. 交差する END、重複 END、未終了 RANGE を拒否する。
<!-- todo:06-015 -->

`tests/loops.rs` の `nested_loops_effect_calls_and_rejections` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test loops nested_loops_effect_calls_and_rejections
```

## 16. `INDEX(x,i+1)` が N 要素を越える場合を生成前に拒否する。
<!-- todo:06-016 -->

`tests/loops.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test loops
```

## 17. bounds の積が i32 を越えるケースを拒否する。
<!-- todo:06-017 -->

`tests/loops.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test loops
```

## 18. `--dump-kernel` に各命令の開いている軸を出す。
<!-- todo:06-018 -->

`tests/loops.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test loops
```

## 19. 単一ループの結果が `[11,22,33]` になる。
<!-- todo:06-019 -->

`tests/loops.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test loops
```

## 20. 二重ループのコピーが `[1,2,3,4,5,6]` になる。
<!-- todo:06-020 -->

`tests/loops.rs` の `nested_loops_effect_calls_and_rejections` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test loops nested_loops_effect_calls_and_rejections
```

## 21. 空は空配列、一要素は `[12]` を返す。
<!-- todo:06-021 -->

`tests/loops.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test loops
```

## 22. END の直前には対応する RANGE が開いている。
<!-- todo:06-022 -->

`tests/loops.rs` の `nested_loops_effect_calls_and_rejections` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test loops nested_loops_effect_calls_and_rejections
```

## 23. 最外側 END の後に未束縛の軸が残らない。
<!-- todo:06-023 -->

`tests/loops.rs` の `nested_loops_effect_calls_and_rejections` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test loops nested_loops_effect_calls_and_rejections
```

## 24. すべてのメモリアクセスを bounds と保存領域長から証明できる。
<!-- todo:06-024 -->

`tests/loops.rs` の正常系と境界系をまとめて再実行する。

```sh check
cargo test --quiet --test loops
```

## 25. 範囲外、スコープ違反、添字 overflow は実行前に失敗する。
<!-- todo:06-025 -->

`tests/loops.rs` の `nested_loops_effect_calls_and_rejections` で、この経路の値または拒否条件を確認する。

```sh check
cargo test --quiet --test loops nested_loops_effect_calls_and_rejections
```

## 章末の累積確認

前章までの example と test も同じ checkout で再ビルドする。

```sh check
cargo test --quiet --all-targets
```
