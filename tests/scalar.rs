// file="tests/scalar.rs" mode="write"
use chibigrad::UOp;
use chibigrad::codegen::run;

#[test]
fn constant_runs_native_code() {
    assert_eq!(run(&(UOp::f32(7.0))).unwrap(), 7.0);
    for value in [
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.1,
        f32::from_bits(1),
        f32::MAX,
    ] {
        assert_eq!(run(&UOp::f32(value)).unwrap().to_bits(), value.to_bits());
    }
    let value = f32::NAN;
    assert!(run(&UOp::f32(value)).unwrap().is_nan());
}

#[test]
fn postorder_and_identity_rewrite() {
    let root = UOp::f32(7.0);
    let order = root.postorder();
    assert!(order.last().unwrap().ptr_eq(&root));
    for (i, node) in order.iter().enumerate() {
        for child in &node.src {
            assert!(order[..i].iter().any(|earlier| earlier.ptr_eq(child)));
        }
    }
    assert!(root.rewrite(|_| None).ptr_eq(&root));
}
