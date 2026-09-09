//! Known-vector gate for the NVFP4 CPU reference. These are the numbers the
//! GPU arms are differential-tested against, so they are asserted, not
//! round-tripped.

use half::bf16;
use infer_quant::{cpu_ref_fp4, e2m1_to_f32, e4m3_to_f32};

#[test]
fn e2m1_lut_known_values() {
    assert_eq!(e2m1_to_f32(0x0), 0.0);
    assert_eq!(e2m1_to_f32(0x1), 0.5);
    assert_eq!(e2m1_to_f32(0x2), 1.0);
    assert_eq!(e2m1_to_f32(0x7), 6.0);
    // -0.0: sign bit set, value zero
    assert!(e2m1_to_f32(0x8).is_sign_negative() && e2m1_to_f32(0x8) == 0.0);
    assert_eq!(e2m1_to_f32(0xf), -6.0);
    // High nibble is masked off.
    assert_eq!(e2m1_to_f32(0xa1), e2m1_to_f32(0x01));
}

#[test]
fn e4m3_known_values() {
    assert_eq!(e4m3_to_f32(0x00), 0.0);
    assert_eq!(e4m3_to_f32(0x38), 1.0);
    assert_eq!(e4m3_to_f32(0xb8), -1.0);
    assert_eq!(e4m3_to_f32(0x7e), 448.0);
    // 0x7f/0xff is the format NaN, clamped to +/-448 (matches device decoder).
    assert_eq!(e4m3_to_f32(0x7f), 448.0);
    assert_eq!(e4m3_to_f32(0xff), -448.0);
    // Subnormal: mantissa 3 -> (3/8) * 2^-6.
    assert_eq!(e4m3_to_f32(0x03), 0.005859375);
    assert_eq!(e4m3_to_f32(0x83), -0.005859375);
}

#[test]
fn cpu_ref_matches_hand_computed() {
    // n=1, k=16, one group. nibbles 0x2 (1.0) and 0x1 (0.5) on the first two
    // columns, e4m3 0x38 = 1.0, global 0.25, x = [1, 2, 0..].
    // (1.0*1.0 + 0.5*2.0) * 1.0 * 0.25 = 0.5.
    let mut packed = vec![0u8; 8];
    packed[0] = 0x12; // lo nibble 0x2, hi nibble 0x1
    let scales = vec![0x38u8];
    let mut x = vec![bf16::ZERO; 16];
    x[0] = bf16::from_f32(1.0);
    x[1] = bf16::from_f32(2.0);
    let out = cpu_ref_fp4(&packed, &scales, 0.25, &x, 1, 1, 16);
    assert!((out[0] - 0.5).abs() < 1e-6, "got {}", out[0]);
}
