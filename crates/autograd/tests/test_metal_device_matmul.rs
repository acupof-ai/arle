//! Metal-vs-CPU parity for the on-device matmul entry points that back the
//! lazy tape (`matmul_bt` and the three device backward/input-grad methods).
//! Mirrors the CUDA coverage in test_cuda_bf16_frozen_ops.rs. All outputs are
//! f32; MLX and the CPU reference differ only in accumulation order, so the
//! tolerance matches the other Metal parity tests.
#![cfg(feature = "metal")]

use std::sync::Mutex;

use autograd::Backend;
use autograd::backend::{
    cpu_matmul_backward, cpu_matmul_bt_backward, cpu_matmul_bt_forward, cpu_matmul_forward,
};
use autograd::backend_metal::MetalBackend;

// MLX's C++ allocator is not re-entrant across threads; serialize the tests.
static METAL_MATMUL_LOCK: Mutex<()> = Mutex::new(());

fn make_rows(shape: &[usize], seed: u64) -> Vec<f32> {
    let size: usize = shape.iter().product();
    let mut s = seed.max(1);
    (0..size)
        .map(|i| {
            // Deterministic LCG in a modest range so backward GEMMs stay close.
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = (s >> 33) as f32 / (1u64 << 31) as f32 - 1.0;
            x * 0.5 + i as f32 * 1e-4
        })
        .collect()
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    let worst = got
        .iter()
        .zip(want.iter())
        .enumerate()
        .map(|(i, (&a, &b))| ((a - b).abs() / b.abs().max(1.0), i, a, b))
        .fold((0f32, 0usize, 0f32, 0f32), |acc, x| {
            if x.0 > acc.0 { x } else { acc }
        });
    assert!(
        worst.0 <= tol,
        "{label}: idx {} got {} want {} rel {}",
        worst.1,
        worst.2,
        worst.3,
        worst.0
    );
}

#[test]
fn metal_matmul_bt_matches_cpu_2d() {
    let _lock = METAL_MATMUL_LOCK.lock().unwrap();
    let backend = MetalBackend;
    let (m, k, n) = (4usize, 8usize, 5usize);
    let a_shape = [m, k];
    let b_shape = [n, k];
    let a = make_rows(&[m, k], 101);
    let b = make_rows(&[n, k], 202);

    let a_h = backend.upload(&a, &a_shape).unwrap();
    let b_h = backend.upload(&b, &b_shape).unwrap();
    let (out_h, out_shape) = backend
        .matmul_bt(&a_h, &a_shape, &b_h, &b_shape)
        .expect("metal matmul_bt");
    assert_eq!(out_shape, vec![m, n]);
    let got = backend.readback(&out_h).unwrap();
    let (want, want_shape) = cpu_matmul_bt_forward(&a, &a_shape, &b, &b_shape).unwrap();
    assert_eq!(want_shape, vec![m, n]);
    assert_close(&got, &want, 1e-3, "matmul_bt 2d");
}

#[test]
fn metal_matmul_backward_device_matches_cpu_2d() {
    let _lock = METAL_MATMUL_LOCK.lock().unwrap();
    let backend = MetalBackend;
    let (m, k, n) = (4usize, 8usize, 6usize);
    let a_shape = [m, k];
    let b_shape = [k, n];
    let g_shape = [m, n];
    let a = make_rows(&[m, k], 303);
    let b = make_rows(&[k, n], 404);
    let g = make_rows(&[m, n], 505);

    let a_h = backend.upload(&a, &a_shape).unwrap();
    let b_h = backend.upload(&b, &b_shape).unwrap();
    let g_h = backend.upload(&g, &g_shape).unwrap();
    let (ga_h, gb_h) = backend
        .matmul_backward_device(&a_h, &a_shape, &b_h, &b_shape, &g_h, &g_shape, true, true)
        .expect("metal matmul backward");
    let (want_a, want_b) =
        cpu_matmul_backward(&a, &a_shape, &b, &b_shape, &g, &g_shape, true, true).expect("cpu ref");
    assert_close(
        &backend.readback(ga_h.as_ref().unwrap()).unwrap(),
        &want_a,
        1e-3,
        "backward grad_a 2d",
    );
    assert_close(
        &backend.readback(gb_h.as_ref().unwrap()).unwrap(),
        &want_b,
        1e-3,
        "backward grad_b 2d",
    );
}

#[test]
fn metal_matmul_backward_device_matches_cpu_3d() {
    let _lock = METAL_MATMUL_LOCK.lock().unwrap();
    let backend = MetalBackend;
    // Rank-3 exercises the transpose-axes ([0,2,1]) path, which a plain
    // reversal would break silently on a batched matrix.
    let (batch, m, k, n) = (3usize, 4usize, 7usize, 5usize);
    let a_shape = [batch, m, k];
    let b_shape = [batch, k, n];
    let g_shape = [batch, m, n];
    let a = make_rows(&[batch, m, k], 606);
    let b = make_rows(&[batch, k, n], 707);
    let g = make_rows(&[batch, m, n], 808);

    let a_h = backend.upload(&a, &a_shape).unwrap();
    let b_h = backend.upload(&b, &b_shape).unwrap();
    let g_h = backend.upload(&g, &g_shape).unwrap();
    let (ga_h, gb_h) = backend
        .matmul_backward_device(&a_h, &a_shape, &b_h, &b_shape, &g_h, &g_shape, true, true)
        .expect("metal matmul backward 3d");
    let (want_a, want_b) =
        cpu_matmul_backward(&a, &a_shape, &b, &b_shape, &g, &g_shape, true, true).expect("cpu ref");
    assert_close(
        &backend.readback(ga_h.as_ref().unwrap()).unwrap(),
        &want_a,
        1e-3,
        "backward grad_a 3d",
    );
    assert_close(
        &backend.readback(gb_h.as_ref().unwrap()).unwrap(),
        &want_b,
        1e-3,
        "backward grad_b 3d",
    );
}

#[test]
fn metal_matmul_backward_device_respects_need_flags() {
    let _lock = METAL_MATMUL_LOCK.lock().unwrap();
    let backend = MetalBackend;
    let a_shape = [2usize, 4usize];
    let b_shape = [4usize, 3usize];
    let g_shape = [2usize, 3usize];
    let a = make_rows(&a_shape, 909);
    let b = make_rows(&b_shape, 1001);
    let g = make_rows(&g_shape, 1102);
    let a_h = backend.upload(&a, &a_shape).unwrap();
    let b_h = backend.upload(&b, &b_shape).unwrap();
    let g_h = backend.upload(&g, &g_shape).unwrap();
    let (ga_h, gb_h) = backend
        .matmul_backward_device(&a_h, &a_shape, &b_h, &b_shape, &g_h, &g_shape, true, false)
        .expect("metal backward flags");
    assert!(ga_h.is_some(), "need_grad_a must return grad_a");
    assert!(gb_h.is_none(), "need_grad_b=false must skip grad_b");
    let (none_a, none_b) = backend
        .matmul_backward_device(&a_h, &a_shape, &b_h, &b_shape, &g_h, &g_shape, false, false)
        .expect("metal backward none");
    assert!(none_a.is_none() && none_b.is_none());
}

#[test]
fn metal_matmul_bt_backward_device_matches_cpu_2d() {
    let _lock = METAL_MATMUL_LOCK.lock().unwrap();
    let backend = MetalBackend;
    let (m, k, n) = (4usize, 8usize, 5usize);
    let a_shape = [m, k];
    let b_shape = [n, k];
    let g_shape = [m, n];
    let a = make_rows(&[m, k], 1203);
    let b = make_rows(&[n, k], 1304);
    let g = make_rows(&[m, n], 1405);

    let a_h = backend.upload(&a, &a_shape).unwrap();
    let b_h = backend.upload(&b, &b_shape).unwrap();
    let g_h = backend.upload(&g, &g_shape).unwrap();
    let (ga_h, gb_h) = backend
        .matmul_bt_backward_device(&a_h, &a_shape, &b_h, &b_shape, &g_h, &g_shape, true, true)
        .expect("metal matmul_bt backward");
    let (want_a, want_b) =
        cpu_matmul_bt_backward(&a, &a_shape, &b, &b_shape, &g, &g_shape, true, true)
            .expect("cpu ref");
    assert_close(
        &backend.readback(ga_h.as_ref().unwrap()).unwrap(),
        &want_a,
        1e-3,
        "bt backward grad_a 2d",
    );
    assert_close(
        &backend.readback(gb_h.as_ref().unwrap()).unwrap(),
        &want_b,
        1e-3,
        "bt backward grad_b 2d",
    );
}

#[test]
fn metal_matmul_bt_input_grad_device_matches_cpu_2d() {
    let _lock = METAL_MATMUL_LOCK.lock().unwrap();
    let backend = MetalBackend;
    let (m, k, n) = (4usize, 8usize, 5usize);
    let b_shape = [n, k];
    let g_shape = [m, n];
    let input_shape = [m, k];
    let b = make_rows(&[n, k], 1506);
    let g = make_rows(&[m, n], 1607);

    let b_h = backend.upload(&b, &b_shape).unwrap();
    let g_h = backend.upload(&g, &g_shape).unwrap();
    let out_h = backend
        .matmul_bt_input_grad_device(&b_h, &b_shape, &g_h, &g_shape, &input_shape)
        .expect("metal matmul_bt input grad");
    let want = cpu_matmul_forward(&g, &g_shape, &b, &b_shape)
        .expect("cpu ref")
        .0;
    assert_close(
        &backend.readback(&out_h).unwrap(),
        &want,
        1e-3,
        "bt input grad 2d",
    );
}
