//! Backend matmul parity tests. The CPU reference is authoritative; each
//! gated backend must match it to within `1e-3` relative tolerance on the
//! three shapes we actually hit in Transformer training: small 2D, square 2D,
//! and batched rank-3.

use autograd::{
    CpuBackend,
    backend::{
        Backend, cpu_embedding_forward, cpu_exp_forward, cpu_gelu_forward,
        cpu_log_softmax_forward_last_axis, cpu_matmul_forward, cpu_mean_last_axis_forward,
        cpu_mul_forward, cpu_mul_scalar_forward, cpu_neg_forward, cpu_rms_norm_forward,
        cpu_silu_forward, cpu_softmax_forward_last_axis, cpu_sum_last_axis_forward,
    },
};

#[allow(dead_code)]
fn _touch_refs() {
    // Keep the reference imports live on builds where the CUDA test block is
    // gated off (e.g. `--features cuda,no-cuda` — types check but tests skip).
    let _ = cpu_softmax_forward_last_axis;
    let _ = cpu_log_softmax_forward_last_axis;
    let _ = cpu_mul_forward;
    let _ = cpu_mul_scalar_forward;
    let _ = cpu_exp_forward;
    let _ = cpu_neg_forward;
    let _ = cpu_gelu_forward;
    let _ = cpu_silu_forward;
    let _ = cpu_rms_norm_forward;
    let _ = cpu_embedding_forward;
    let _ = cpu_sum_last_axis_forward;
    let _ = cpu_mean_last_axis_forward;
}

fn make_rows(shape: &[usize], seed: u64) -> Vec<f32> {
    let size: usize = shape.iter().product();
    let mut out = Vec::with_capacity(size);
    let mut s = seed;
    for i in 0..size {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let normalised = ((s >> 32) as u32 as f32) / (u32::MAX as f32);
        out.push((normalised - 0.5) * 2.0 + (i as f32) * 1e-4);
    }
    out
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        let denom = b.abs().max(1.0);
        let rel = (a - b).abs() / denom;
        assert!(rel <= tol, "{label}: idx {i} got {a} want {b} rel {rel}",);
    }
}

fn run_lazy_matmul<B: Backend>(
    backend: &B,
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> autograd::Result<(Vec<f32>, Vec<usize>)> {
    let a_handle = backend.upload(a, a_shape)?;
    let b_handle = backend.upload(b, b_shape)?;
    let (out_handle, out_shape) = backend.matmul(&a_handle, a_shape, &b_handle, b_shape)?;
    backend.eval(&[&out_handle])?;
    let out = backend.readback(&out_handle)?;
    Ok((out, out_shape))
}

#[test]
fn cpu_backend_matches_reference_2d() {
    let backend = CpuBackend;
    let a = make_rows(&[8, 16], 1);
    let b = make_rows(&[16, 32], 2);
    let (got, got_shape) =
        run_lazy_matmul(&backend, &a, &[8, 16], &b, &[16, 32]).expect("cpu matmul");
    let (want, want_shape) = cpu_matmul_forward(&a, &[8, 16], &b, &[16, 32]).expect("ref");
    assert_eq!(got_shape, want_shape);
    assert_close(&got, &want, 1e-6, "cpu 2d");
}

#[cfg(feature = "metal")]
#[test]
fn metal_backend_matches_cpu_small_2d() {
    use autograd::backend_metal::MetalBackend;

    let backend = MetalBackend;
    let a = make_rows(&[8, 16], 11);
    let b = make_rows(&[16, 32], 22);
    let (got, got_shape) =
        run_lazy_matmul(&backend, &a, &[8, 16], &b, &[16, 32]).expect("metal matmul");
    let (want, _) = cpu_matmul_forward(&a, &[8, 16], &b, &[16, 32]).expect("ref");
    assert_eq!(got_shape, vec![8, 32]);
    assert_close(&got, &want, 1e-3, "metal 2d small");
}

#[cfg(feature = "metal")]
#[test]
fn metal_backend_matches_cpu_square_2d() {
    use autograd::backend_metal::MetalBackend;

    let backend = MetalBackend;
    let a = make_rows(&[4, 64], 33);
    let b = make_rows(&[64, 64], 44);
    let (got, got_shape) =
        run_lazy_matmul(&backend, &a, &[4, 64], &b, &[64, 64]).expect("metal matmul");
    let (want, _) = cpu_matmul_forward(&a, &[4, 64], &b, &[64, 64]).expect("ref");
    assert_eq!(got_shape, vec![4, 64]);
    assert_close(&got, &want, 1e-3, "metal 2d square");
}

#[cfg(feature = "metal")]
#[test]
fn metal_backend_matches_cpu_batched_3d() {
    use autograd::backend_metal::MetalBackend;

    let backend = MetalBackend;
    let a = make_rows(&[3, 8, 16], 55);
    let b = make_rows(&[3, 16, 32], 66);
    let (got, got_shape) =
        run_lazy_matmul(&backend, &a, &[3, 8, 16], &b, &[3, 16, 32]).expect("metal matmul");
    let (want, _) = cpu_matmul_forward(&a, &[3, 8, 16], &b, &[3, 16, 32]).expect("ref");
    assert_eq!(got_shape, vec![3, 8, 32]);
    assert_close(&got, &want, 1e-3, "metal 3d batched");
}

fn run_lazy_add<B: Backend>(
    backend: &B,
    a: &[f32],
    b: &[f32],
    shape: &[usize],
) -> autograd::Result<Vec<f32>> {
    let a_handle = backend.upload(a, shape)?;
    let b_handle = backend.upload(b, shape)?;
    let out_handle = backend.add(&a_handle, &b_handle, shape)?;
    backend.eval(&[&out_handle])?;
    backend.readback(&out_handle)
}

fn reference_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

#[test]
fn cpu_backend_add_matches_reference() {
    let backend = CpuBackend;
    let a = make_rows(&[4, 16], 7);
    let b = make_rows(&[4, 16], 8);
    let got = run_lazy_add(&backend, &a, &b, &[4, 16]).expect("cpu add");
    assert_close(&got, &reference_add(&a, &b), 1e-6, "cpu add 2d");
}

#[cfg(feature = "metal")]
#[test]
fn metal_backend_add_matches_cpu_2d() {
    use autograd::backend_metal::MetalBackend;

    let backend = MetalBackend;
    let a = make_rows(&[8, 32], 101);
    let b = make_rows(&[8, 32], 202);
    let got = run_lazy_add(&backend, &a, &b, &[8, 32]).expect("metal add");
    assert_close(&got, &reference_add(&a, &b), 1e-3, "metal add 2d");
}

#[cfg(feature = "metal")]
#[test]
fn metal_backend_add_matches_cpu_3d() {
    use autograd::backend_metal::MetalBackend;

    let backend = MetalBackend;
    let a = make_rows(&[3, 8, 16], 303);
    let b = make_rows(&[3, 8, 16], 404);
    let got = run_lazy_add(&backend, &a, &b, &[3, 8, 16]).expect("metal add");
    assert_close(&got, &reference_add(&a, &b), 1e-3, "metal add 3d");
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_backend_matches_cpu_small_2d() {
    use autograd::backend_cuda::CudaBackend;

    let backend = CudaBackend::new(0).expect("cuda ctx");
    let a = make_rows(&[8, 16], 11);
    let b = make_rows(&[16, 32], 22);
    let (got, got_shape) = backend
        .matmul_forward(&a, &[8, 16], &b, &[16, 32])
        .expect("cuda matmul");
    let (want, _) = cpu_matmul_forward(&a, &[8, 16], &b, &[16, 32]).expect("ref");
    assert_eq!(got_shape, vec![8, 32]);
    assert_close(&got, &want, 1e-3, "cuda 2d small");
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_backend_matmul_matches_cpu_small_2d() {
    use autograd::backend_cuda::CudaBackend;

    let backend = CudaBackend::new(0).expect("cuda ctx");
    let a = make_rows(&[8, 16], 77);
    let b = make_rows(&[16, 32], 88);
    let (got, got_shape) =
        run_lazy_matmul(&backend, &a, &[8, 16], &b, &[16, 32]).expect("cuda lazy matmul");
    let (want, _) = cpu_matmul_forward(&a, &[8, 16], &b, &[16, 32]).expect("ref");
    assert_eq!(got_shape, vec![8, 32]);
    assert_close(&got, &want, 1e-3, "cuda lazy matmul 2d");
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_backend_matches_cpu_batched_3d() {
    use autograd::backend_cuda::CudaBackend;

    let backend = CudaBackend::new(0).expect("cuda ctx");
    let a = make_rows(&[3, 8, 16], 55);
    let b = make_rows(&[3, 16, 32], 66);
    let (got, got_shape) = backend
        .matmul_forward(&a, &[3, 8, 16], &b, &[3, 16, 32])
        .expect("cuda matmul");
    let (want, _) = cpu_matmul_forward(&a, &[3, 8, 16], &b, &[3, 16, 32]).expect("ref");
    assert_eq!(got_shape, vec![3, 8, 32]);
    assert_close(&got, &want, 1e-3, "cuda 3d batched");
}

#[cfg(feature = "metal")]
#[test]
fn metal_backend_softmax_matches_cpu_2d() {
    use autograd::backend_metal::MetalBackend;

    let backend = MetalBackend;
    let x = make_rows(&[4, 32], 909);
    let got = backend
        .softmax_forward_last_axis(&x, &[4, 32])
        .expect("metal softmax");
    let want = cpu_softmax_forward_last_axis(&x, &[4, 32]).expect("ref");
    assert_close(&got, &want, 1e-3, "metal softmax 2d");
}

#[cfg(feature = "metal")]
#[test]
fn metal_backend_log_softmax_matches_cpu_2d() {
    use autograd::backend_metal::MetalBackend;

    let backend = MetalBackend;
    let x = make_rows(&[4, 32], 808);
    let got = backend
        .log_softmax_forward_last_axis(&x, &[4, 32])
        .expect("metal log_softmax");
    let want = cpu_log_softmax_forward_last_axis(&x, &[4, 32]).expect("ref");
    assert_close(&got, &want, 1e-3, "metal log_softmax 2d");
}

#[cfg(feature = "metal")]
#[test]
fn metal_backend_log_softmax_matches_cpu_wide_vocab() {
    use autograd::backend_metal::MetalBackend;

    // Stresses the actual hot path: log_softmax over a realistic vocab
    // dimension from pretrain (vocab≈150k). 4096 is a shrunken proxy that
    // still exercises the full reduction + broadcast path.
    let backend = MetalBackend;
    let x = make_rows(&[8, 4096], 707);
    let got = backend
        .log_softmax_forward_last_axis(&x, &[8, 4096])
        .expect("metal log_softmax wide");
    let want = cpu_log_softmax_forward_last_axis(&x, &[8, 4096]).expect("ref");
    assert_close(&got, &want, 1e-3, "metal log_softmax wide");
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_backend_add_matches_cpu_2d() {
    use autograd::backend_cuda::CudaBackend;

    let backend = CudaBackend::new(0).expect("cuda ctx");
    let a = make_rows(&[8, 32], 505);
    let b = make_rows(&[8, 32], 606);
    let got = run_lazy_add(&backend, &a, &b, &[8, 32]).expect("cuda add");
    assert_close(&got, &reference_add(&a, &b), 1e-3, "cuda add 2d");
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_backend_softmax_matches_cpu_2d() {
    use autograd::backend::Backend;
    use autograd::backend_cuda::CudaBackend;

    let backend = CudaBackend::new(0).expect("cuda ctx");
    let x = make_rows(&[4, 32], 919);
    let got = backend
        .softmax_forward_last_axis(&x, &[4, 32])
        .expect("cuda softmax");
    let want = cpu_softmax_forward_last_axis(&x, &[4, 32]).expect("ref");
    assert_close(&got, &want, 1e-3, "cuda softmax 2d");
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_backend_log_softmax_matches_cpu_2d() {
    use autograd::backend::Backend;
    use autograd::backend_cuda::CudaBackend;

    let backend = CudaBackend::new(0).expect("cuda ctx");
    let x = make_rows(&[4, 32], 828);
    let got = backend
        .log_softmax_forward_last_axis(&x, &[4, 32])
        .expect("cuda log_softmax");
    let want = cpu_log_softmax_forward_last_axis(&x, &[4, 32]).expect("ref");
    assert_close(&got, &want, 1e-3, "cuda log_softmax 2d");
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_backend_log_softmax_matches_cpu_wide_vocab() {
    use autograd::backend::Backend;
    use autograd::backend_cuda::CudaBackend;

    let backend = CudaBackend::new(0).expect("cuda ctx");
    let x = make_rows(&[8, 4096], 727);
    let got = backend
        .log_softmax_forward_last_axis(&x, &[8, 4096])
        .expect("cuda log_softmax wide");
    let want = cpu_log_softmax_forward_last_axis(&x, &[8, 4096]).expect("ref");
    assert_close(&got, &want, 1e-3, "cuda log_softmax wide");
}

// ──────────────────────────────────────────────────────────────────────
// CPU reference self-parity tests (no backend feature required).
// Ensures the newly-added CPU fns compile and stay consistent with the
// autograd::ops::* CPU paths they mirror.
// ──────────────────────────────────────────────────────────────────────

#[test]
fn cpu_mul_scalar_matches_elementwise() {
    let x = make_rows(&[6, 5], 91);
    let got = cpu_mul_scalar_forward(&x, 0.25).unwrap();
    let want: Vec<f32> = x.iter().map(|v| v * 0.25).collect();
    assert_close(&got, &want, 1e-6, "cpu mul_scalar");
}

#[test]
fn cpu_silu_matches_ref() {
    let x = make_rows(&[4, 8], 311);
    let got = cpu_silu_forward(&x).unwrap();
    for (i, &v) in x.iter().enumerate() {
        let want = v * (1.0 / (1.0 + (-v).exp()));
        assert!(
            (got[i] - want).abs() < 1e-6,
            "idx {i}: {} vs {}",
            got[i],
            want
        );
    }
}

#[test]
fn cpu_rms_norm_matches_ref() {
    let shape = &[3, 8];
    let x = make_rows(shape, 19);
    let weight: Vec<f32> = (0..8).map(|i| 0.5 + (i as f32) * 0.1).collect();
    let got = cpu_rms_norm_forward(&x, &weight, shape, 1e-6).unwrap();
    // Reference: per-row rsqrt(mean(x^2)+eps) * x * weight
    let mut want = vec![0.0_f32; 24];
    for row in 0..3 {
        let base = row * 8;
        let mean_sq = x[base..base + 8].iter().map(|v| v * v).sum::<f32>() / 8.0;
        let inv_rms = (mean_sq + 1e-6).sqrt().recip();
        for col in 0..8 {
            want[base + col] = x[base + col] * inv_rms * weight[col];
        }
    }
    assert_close(&got, &want, 1e-6, "cpu rms_norm");
}

#[test]
fn cpu_embedding_gather_and_oob() {
    let weight: Vec<f32> = (0..(5 * 4)).map(|i| i as f32).collect();
    let ids = [0_i32, 2, 4, -1, 10];
    let got = cpu_embedding_forward(&weight, 5, 4, &ids).unwrap();
    assert_eq!(&got[0..4], &[0.0, 1.0, 2.0, 3.0]);
    assert_eq!(&got[4..8], &[8.0, 9.0, 10.0, 11.0]);
    assert_eq!(&got[8..12], &[16.0, 17.0, 18.0, 19.0]);
    assert_eq!(&got[12..16], &[0.0, 0.0, 0.0, 0.0]); // id=-1 zero row
    assert_eq!(&got[16..20], &[0.0, 0.0, 0.0, 0.0]); // id=10 oob zero row
}

#[test]
fn cpu_sum_and_mean_last_axis() {
    let shape = &[2, 5];
    let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0];
    let sum = cpu_sum_last_axis_forward(&x, shape).unwrap();
    assert_eq!(sum, vec![15.0, 150.0]);
    let mean = cpu_mean_last_axis_forward(&x, shape).unwrap();
    assert_eq!(mean, vec![3.0, 30.0]);
}

// ──────────────────────────────────────────────────────────────────────
// CUDA parity tests — PENDING REMOTE CUDA VERIFICATION. Compile on Mac
// under `--features cuda,no-cuda`; run on a real GPU box with
// `cargo test -p autograd --features cuda --test test_backend`.
// ──────────────────────────────────────────────────────────────────────

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_backend_mul_matches_cpu() {
    use autograd::backend::Backend;
    use autograd::backend_cuda::CudaBackend;
    let backend = CudaBackend::new(0).expect("cuda ctx");
    let a = make_rows(&[3, 17], 111);
    let b = make_rows(&[3, 17], 222);
    let got = backend.mul_forward(&a, &b).expect("cuda mul");
    let want = cpu_mul_forward(&a, &b).unwrap();
    assert_close(&got, &want, 1e-5, "cuda mul");
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_backend_mul_scalar_matches_cpu() {
    use autograd::backend::Backend;
    use autograd::backend_cuda::CudaBackend;
    let backend = CudaBackend::new(0).expect("cuda ctx");
    let a = make_rows(&[4, 9], 77);
    let got = backend
        .mul_scalar_forward(&a, -0.5)
        .expect("cuda mul_scalar");
    let want = cpu_mul_scalar_forward(&a, -0.5).unwrap();
    assert_close(&got, &want, 1e-6, "cuda mul_scalar");
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_backend_exp_matches_cpu() {
    use autograd::backend::Backend;
    use autograd::backend_cuda::CudaBackend;
    let backend = CudaBackend::new(0).expect("cuda ctx");
    let a = make_rows(&[2, 128], 3);
    let got = backend.exp_forward(&a).expect("cuda exp");
    let want = cpu_exp_forward(&a).unwrap();
    assert_close(&got, &want, 1e-4, "cuda exp");
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_backend_neg_matches_cpu() {
    use autograd::backend::Backend;
    use autograd::backend_cuda::CudaBackend;
    let backend = CudaBackend::new(0).expect("cuda ctx");
    let a = make_rows(&[4, 16], 5);
    let got = backend.neg_forward(&a).expect("cuda neg");
    let want = cpu_neg_forward(&a).unwrap();
    assert_close(&got, &want, 1e-6, "cuda neg");
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_backend_gelu_matches_cpu() {
    use autograd::backend::Backend;
    use autograd::backend_cuda::CudaBackend;
    let backend = CudaBackend::new(0).expect("cuda ctx");
    let a = make_rows(&[4, 128], 9);
    let got = backend.gelu_forward(&a).expect("cuda gelu");
    let want = cpu_gelu_forward(&a).unwrap();
    assert_close(&got, &want, 1e-4, "cuda gelu");
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_backend_silu_matches_cpu() {
    use autograd::backend::Backend;
    use autograd::backend_cuda::CudaBackend;
    let backend = CudaBackend::new(0).expect("cuda ctx");
    let a = make_rows(&[4, 128], 13);
    let got = backend.silu_forward(&a).expect("cuda silu");
    let want = cpu_silu_forward(&a).unwrap();
    assert_close(&got, &want, 1e-4, "cuda silu");
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_backend_rms_norm_matches_cpu() {
    use autograd::backend::Backend;
    use autograd::backend_cuda::CudaBackend;
    let backend = CudaBackend::new(0).expect("cuda ctx");
    let shape = &[4, 64];
    let x = make_rows(shape, 33);
    let weight: Vec<f32> = (0..64).map(|i| 0.5 + (i as f32) * 0.01).collect();
    let got = backend
        .rms_norm_forward(&x, &weight, shape, 1e-6)
        .expect("cuda rms_norm");
    let want = cpu_rms_norm_forward(&x, &weight, shape, 1e-6).unwrap();
    assert_close(&got, &want, 1e-4, "cuda rms_norm");
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_backend_embedding_matches_cpu() {
    use autograd::backend::Backend;
    use autograd::backend_cuda::CudaBackend;
    let backend = CudaBackend::new(0).expect("cuda ctx");
    let vocab = 64_usize;
    let dim = 32_usize;
    let weight = make_rows(&[vocab, dim], 17);
    let ids = [0_i32, 5, 10, 63, -1, 99, 7];
    let got = backend
        .embedding_forward(&weight, vocab, dim, &ids)
        .expect("cuda embed");
    let want = cpu_embedding_forward(&weight, vocab, dim, &ids).unwrap();
    assert_close(&got, &want, 1e-6, "cuda embedding");
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_backend_sum_last_axis_matches_cpu() {
    use autograd::backend::Backend;
    use autograd::backend_cuda::CudaBackend;
    let backend = CudaBackend::new(0).expect("cuda ctx");
    let shape = &[6, 257];
    let x = make_rows(shape, 41);
    let got = backend.sum_last_axis_forward(&x, shape).expect("cuda sum");
    let want = cpu_sum_last_axis_forward(&x, shape).unwrap();
    assert_close(&got, &want, 1e-3, "cuda sum");
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_backend_mean_last_axis_matches_cpu() {
    use autograd::backend::Backend;
    use autograd::backend_cuda::CudaBackend;
    let backend = CudaBackend::new(0).expect("cuda ctx");
    let shape = &[6, 257];
    let x = make_rows(shape, 43);
    let got = backend
        .mean_last_axis_forward(&x, shape)
        .expect("cuda mean");
    let want = cpu_mean_last_axis_forward(&x, shape).unwrap();
    assert_close(&got, &want, 1e-5, "cuda mean");
}
