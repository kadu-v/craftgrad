use chibigrad::{Arg, uop::ConstFloat};

#[test]
fn arguments_compare_values_and_keep_signed_zero() {
    let positive = ConstFloat::new(0.0);
    let negative = ConstFloat::new(-0.0);
    assert_eq!(positive, negative);
    assert_eq!(Arg::Float(positive), Arg::Float(negative));
    assert_ne!(positive.value().to_bits(), negative.value().to_bits());
    assert_eq!(ConstFloat::new(1.1).value(), 1.1);
    assert_ne!(
        ConstFloat::new(f64::INFINITY),
        ConstFloat::new(f64::NEG_INFINITY)
    );
}

#[test]
fn all_nan_payloads_are_canonical() {
    let canonical = ConstFloat::new(f64::NAN);
    for bits in [0x7ff0_0000_0000_0001, 0xfff8_0000_0000_0042] {
        let value = ConstFloat::new(f64::from_bits(bits));
        assert_eq!(value, canonical);
        assert_eq!(value.value().to_bits(), 0x7ff8_0000_0000_0000);
    }
}
