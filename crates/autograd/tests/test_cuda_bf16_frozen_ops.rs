#![cfg(all(feature = "cuda", not(feature = "no-cuda")))]

use autograd::backend::{bf16_bits_to_f32, cpu_embedding_forward, cpu_matmul_bt_forward};
use autograd::backend_cuda::CudaBackend;
use autograd::Backend;

fn f32_to_bf16_bits_round_nearest_even(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7fff + lsb;
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

fn assert_close(actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(actual.len(), expected.len());
    let mut worst = (0.0_f32, 0_usize, 0.0_f32, 0.0_f32);
    for (idx, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        let abs = (got - want).abs();
        let tol = atol + rtol * want.abs();
        let excess = abs / tol;
        if excess > worst.0 {
            worst = (excess, idx, got, want);
        }
    }
    assert!(
        worst.0 <= 1.0,
        "worst excess={} idx={} got={} want={}",
        worst.0,
        worst.1,
        worst.2,
        worst.3
    );
}

#[test]
fn cuda_bf16_upload_readback_roundtrips_as_f32() {
    let backend = CudaBackend::new(0).expect("cuda backend");
    let src = [0.0_f32, 1.25, -2.5, 3.75, -0.03125, 64.5];
    let bits: Vec<u16> = src
        .iter()
        .map(|&value| f32_to_bf16_bits_round_nearest_even(value))
        .collect();
    let handle = backend
        .upload_bf16_bits(&bits, &[2, 3])
        .expect("upload bf16");
    let got = backend.readback(&handle).expect("readback bf16");
    let expected: Vec<f32> = bits.iter().map(|&bits| bf16_bits_to_f32(bits)).collect();
    assert_eq!(got, expected);
}

#[test]
fn cuda_matmul_bt_accepts_frozen_bf16_rhs() {
    let backend = CudaBackend::new(0).expect("cuda backend");
    let a_shape = [2, 4];
    let b_shape = [3, 4];
    let a = [0.25_f32, -1.0, 2.0, 0.5, 1.5, -0.75, 0.125, -2.0];
    let b_f32 = [
        -0.5_f32, 0.75, 1.25, -1.5, 2.0, -0.25, 0.5, 1.0, -1.0, -0.125, 0.875, 1.75,
    ];
    let b_bits: Vec<u16> = b_f32
        .iter()
        .map(|&value| f32_to_bf16_bits_round_nearest_even(value))
        .collect();
    let a_quantized: Vec<f32> = a
        .iter()
        .map(|&value| bf16_bits_to_f32(f32_to_bf16_bits_round_nearest_even(value)))
        .collect();
    let b_quantized: Vec<f32> = b_bits.iter().map(|&bits| bf16_bits_to_f32(bits)).collect();

    let a_handle = backend.upload(&a, &a_shape).expect("upload lhs");
    let b_handle = backend
        .upload_bf16_bits(&b_bits, &b_shape)
        .expect("upload rhs bf16");
    let (out_handle, out_shape) = backend
        .matmul_bt(&a_handle, &a_shape, &b_handle, &b_shape)
        .expect("bf16 rhs matmul_bt");
    assert_eq!(out_shape, vec![2, 3]);

    let got = backend.readback(&out_handle).expect("readback output");
    let expected_bf16_bits: Vec<u16> =
        cpu_matmul_bt_forward(&a_quantized, &a_shape, &b_quantized, &b_shape)
            .expect("cpu reference")
            .0
            .into_iter()
            .map(f32_to_bf16_bits_round_nearest_even)
            .collect();
    let expected: Vec<f32> = expected_bf16_bits
        .into_iter()
        .map(bf16_bits_to_f32)
        .collect();
    let expected_shape = vec![2, 3];
    assert_eq!(expected_shape, out_shape);
    assert_close(&got, &expected, 2e-3, 2e-3);
}

#[test]
fn cuda_embedding_accepts_frozen_bf16_table() {
    let backend = CudaBackend::new(0).expect("cuda backend");
    let table_shape = [5, 3];
    let table_f32 = [
        0.0_f32, 0.25, -0.5, 1.0, -1.25, 2.5, 3.0, 4.0, -5.0, 0.125, 0.5, 0.875, -2.0, -3.0, -4.0,
    ];
    let table_bits: Vec<u16> = table_f32
        .iter()
        .map(|&value| f32_to_bf16_bits_round_nearest_even(value))
        .collect();
    let table_quantized: Vec<f32> = table_bits
        .iter()
        .map(|&bits| bf16_bits_to_f32(bits))
        .collect();
    let ids = [3_i32, 1, 4, 0];

    let table = backend
        .upload_bf16_bits(&table_bits, &table_shape)
        .expect("upload bf16 embedding");
    let out = backend
        .embedding(&table, &table_shape, &ids)
        .expect("embedding bf16 table");
    let got = backend.readback(&out).expect("readback embedding");
    let expected = cpu_embedding_forward(&table_quantized, table_shape[0], table_shape[1], &ids)
        .expect("cpu embedding");
    assert_eq!(got, expected);
}

#[test]
fn cuda_embedding_from_f32_ids_accepts_frozen_bf16_table() {
    let backend = CudaBackend::new(0).expect("cuda backend");
    let table_shape = [4, 4];
    let table_f32 = [
        0.0_f32, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, -1.0, -1.25, -1.5, -1.75, 2.0, 2.25, 2.5,
        2.75,
    ];
    let table_bits: Vec<u16> = table_f32
        .iter()
        .map(|&value| f32_to_bf16_bits_round_nearest_even(value))
        .collect();
    let table_quantized: Vec<f32> = table_bits
        .iter()
        .map(|&bits| bf16_bits_to_f32(bits))
        .collect();
    let ids_f32 = [2.0_f32, 0.0, 3.0];
    let ids_i32 = [2_i32, 0, 3];

    let table = backend
        .upload_bf16_bits(&table_bits, &table_shape)
        .expect("upload bf16 embedding");
    let ids = backend.upload(&ids_f32, &[3]).expect("upload f32 ids");
    let out = backend
        .embedding_from_f32_ids(&table, &table_shape, &ids, ids_f32.len())
        .expect("embedding bf16 table from f32 ids");
    let got = backend.readback(&out).expect("readback embedding");
    let expected =
        cpu_embedding_forward(&table_quantized, table_shape[0], table_shape[1], &ids_i32)
            .expect("cpu embedding");
    assert_eq!(got, expected);
}
