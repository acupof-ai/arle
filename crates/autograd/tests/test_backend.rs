//! Backend matmul parity tests. The CPU reference is authoritative; each
//! gated backend must match it to within `1e-3` relative tolerance on the
//! three shapes we actually hit in Transformer training: small 2D, square 2D,
//! and batched rank-3.

use autograd::{
    CpuBackend,
    backend::{
        Backend, cpu_embedding_forward, cpu_exp_forward, cpu_gather_last_dim_forward,
        cpu_gelu_forward, cpu_log_softmax_forward_last_axis, cpu_matmul_backward,
        cpu_matmul_forward, cpu_mean_last_axis_forward, cpu_mul_forward, cpu_mul_scalar_forward,
        cpu_neg_forward, cpu_rms_norm_forward, cpu_rope_forward, cpu_scatter_add_rows_forward,
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
    let _ = cpu_rope_forward;
    let _ = cpu_gather_last_dim_forward;
    let _ = cpu_scatter_add_rows_forward;
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

#[test]
fn cpu_gather_last_dim_basic() {
    // src shape [3, 5], pick one element per row.
    let src: Vec<f32> = (0..15).map(|i| i as f32).collect();
    let ids = [0_i32, 2, 4];
    let got = cpu_gather_last_dim_forward(&src, &[3, 5], &ids).unwrap();
    assert_eq!(got, vec![0.0, 7.0, 14.0]);

    // Out-of-range → error.
    let bad = [0_i32, 2, 5];
    assert!(cpu_gather_last_dim_forward(&src, &[3, 5], &bad).is_err());
}

#[test]
fn cpu_rope_matches_ops() {
    // Cross-check the Backend trait default (`cpu_rope_forward`) against the
    // original `ops::rope::rope` implementation on a small Qwen3-shaped input.
    use autograd::Tape;
    use autograd::TensorStore;
    use autograd::ops;
    use autograd::tensor::Tensor;
    let batch = 2_usize;
    let heads = 3_usize;
    let seq = 4_usize;
    let head_dim = 8_usize;
    let half_dim = head_dim / 2;
    let shape = &[batch, heads, seq, head_dim];
    let x = make_rows(shape, 91);
    let mut cos = Vec::with_capacity(seq * half_dim);
    let mut sin = Vec::with_capacity(seq * half_dim);
    for t in 0..seq {
        for i in 0..half_dim {
            let theta = (t as f32) * (0.02_f32 + (i as f32) * 0.01_f32);
            cos.push(theta.cos());
            sin.push(theta.sin());
        }
    }
    let want = cpu_rope_forward(&x, shape, &cos, &sin).unwrap();

    // Route through ops::rope::rope so we catch any drift between the two.
    let mut store = TensorStore::default();
    let x_id = store.alloc(Tensor::new(x.clone(), shape.to_vec(), false).unwrap());
    let cos_id = store.alloc(Tensor::new(cos.clone(), vec![seq, half_dim], false).unwrap());
    let sin_id = store.alloc(Tensor::new(sin.clone(), vec![seq, half_dim], false).unwrap());
    let mut tape = Tape::default();
    let out_id = ops::rope::rope(x_id, cos_id, sin_id, &mut store, &mut tape).unwrap();
    let ops_out = store.get(out_id).unwrap().data.clone();
    assert_close(&want, &ops_out, 1e-6, "cpu_rope vs ops::rope");
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

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_backend_gather_last_dim_matches_cpu() {
    use autograd::backend::Backend;
    use autograd::backend_cuda::CudaBackend;
    let backend = CudaBackend::new(0).expect("cuda ctx");
    let shape = &[4_usize, 128_usize];
    let src = make_rows(shape, 71);
    let ids = [5_i32, 0, 127, 42];
    let got = backend
        .gather_last_dim_forward(&src, shape, &ids)
        .expect("cuda gather");
    let want = cpu_gather_last_dim_forward(&src, shape, &ids).unwrap();
    assert_close(&got, &want, 1e-6, "cuda gather");
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_backend_rope_matches_cpu() {
    use autograd::backend::Backend;
    use autograd::backend_cuda::CudaBackend;
    let backend = CudaBackend::new(0).expect("cuda ctx");
    let batch = 2_usize;
    let heads = 4_usize;
    let seq = 16_usize;
    let head_dim = 64_usize;
    let half_dim = head_dim / 2;
    let shape = &[batch, heads, seq, head_dim];
    let x = make_rows(shape, 55);
    let mut cos = Vec::with_capacity(seq * half_dim);
    let mut sin = Vec::with_capacity(seq * half_dim);
    for t in 0..seq {
        for i in 0..half_dim {
            let theta = (t as f32) * (0.02_f32 + (i as f32) * 0.01_f32);
            cos.push(theta.cos());
            sin.push(theta.sin());
        }
    }
    let got = backend
        .rope_forward(&x, shape, &cos, &sin)
        .expect("cuda rope");
    let want = cpu_rope_forward(&x, shape, &cos, &sin).unwrap();
    assert_close(&got, &want, 1e-4, "cuda rope");
}

// ──────────────────────────────────────────────────────────────────────
// Metal parity tests for the 12 newly-added Backend trait methods.
// Mirrors the CUDA block above: upload → backend op → compare vs CPU ref.
// ──────────────────────────────────────────────────────────────────────

#[cfg(feature = "metal")]
#[test]
fn metal_backend_mul_matches_cpu() {
    use autograd::backend::Backend;
    use autograd::backend_metal::MetalBackend;
    let backend = MetalBackend;
    let a = make_rows(&[3, 17], 111);
    let b = make_rows(&[3, 17], 222);
    let got = backend.mul_forward(&a, &b).expect("metal mul");
    let want = cpu_mul_forward(&a, &b).unwrap();
    assert_close(&got, &want, 1e-5, "metal mul");
}

#[cfg(feature = "metal")]
#[test]
fn metal_backend_mul_scalar_matches_cpu() {
    use autograd::backend::Backend;
    use autograd::backend_metal::MetalBackend;
    let backend = MetalBackend;
    let a = make_rows(&[4, 9], 77);
    let got = backend
        .mul_scalar_forward(&a, -0.5)
        .expect("metal mul_scalar");
    let want = cpu_mul_scalar_forward(&a, -0.5).unwrap();
    assert_close(&got, &want, 1e-6, "metal mul_scalar");
}

#[cfg(feature = "metal")]
#[test]
fn metal_backend_exp_matches_cpu() {
    use autograd::backend::Backend;
    use autograd::backend_metal::MetalBackend;
    let backend = MetalBackend;
    let a = make_rows(&[2, 128], 3);
    let got = backend.exp_forward(&a).expect("metal exp");
    let want = cpu_exp_forward(&a).unwrap();
    assert_close(&got, &want, 1e-3, "metal exp");
}

#[cfg(feature = "metal")]
#[test]
fn metal_backend_neg_matches_cpu() {
    use autograd::backend::Backend;
    use autograd::backend_metal::MetalBackend;
    let backend = MetalBackend;
    let a = make_rows(&[4, 16], 5);
    let got = backend.neg_forward(&a).expect("metal neg");
    let want = cpu_neg_forward(&a).unwrap();
    assert_close(&got, &want, 1e-6, "metal neg");
}

#[cfg(feature = "metal")]
#[test]
fn metal_backend_gelu_matches_cpu() {
    use autograd::backend::Backend;
    use autograd::backend_metal::MetalBackend;
    let backend = MetalBackend;
    let a = make_rows(&[4, 128], 9);
    let got = backend.gelu_forward(&a).expect("metal gelu");
    let want = cpu_gelu_forward(&a).unwrap();
    assert_close(&got, &want, 1e-3, "metal gelu");
}

#[cfg(feature = "metal")]
#[test]
fn metal_backend_silu_matches_cpu() {
    use autograd::backend::Backend;
    use autograd::backend_metal::MetalBackend;
    let backend = MetalBackend;
    let a = make_rows(&[4, 128], 13);
    let got = backend.silu_forward(&a).expect("metal silu");
    let want = cpu_silu_forward(&a).unwrap();
    assert_close(&got, &want, 1e-4, "metal silu");
}

#[cfg(feature = "metal")]
#[test]
fn metal_backend_rms_norm_matches_cpu() {
    use autograd::backend::Backend;
    use autograd::backend_metal::MetalBackend;
    let backend = MetalBackend;
    let shape = &[4, 64];
    let x = make_rows(shape, 33);
    let weight: Vec<f32> = (0..64).map(|i| 0.5 + (i as f32) * 0.01).collect();
    let got = backend
        .rms_norm_forward(&x, &weight, shape, 1e-6)
        .expect("metal rms_norm");
    let want = cpu_rms_norm_forward(&x, &weight, shape, 1e-6).unwrap();
    assert_close(&got, &want, 1e-4, "metal rms_norm");
}

#[cfg(feature = "metal")]
#[test]
fn metal_backend_embedding_matches_cpu() {
    use autograd::backend::Backend;
    use autograd::backend_metal::MetalBackend;
    let backend = MetalBackend;
    let vocab = 64_usize;
    let dim = 32_usize;
    let weight = make_rows(&[vocab, dim], 17);
    // Mirror the CUDA embedding test: the CPU reference zero-fills both negative
    // and out-of-bounds ids, so the Metal impl is expected to do the same. If a
    // mismatch surfaces here, the parallel Metal impl owns the divergence.
    let ids = [0_i32, 5, 10, 63, -1, 99, 7];
    let got = backend
        .embedding_forward(&weight, vocab, dim, &ids)
        .expect("metal embed");
    let want = cpu_embedding_forward(&weight, vocab, dim, &ids).unwrap();
    assert_close(&got, &want, 1e-6, "metal embedding");
}

#[cfg(feature = "metal")]
#[test]
fn metal_backend_sum_last_axis_matches_cpu() {
    use autograd::backend::Backend;
    use autograd::backend_metal::MetalBackend;
    let backend = MetalBackend;
    let shape = &[6, 257];
    let x = make_rows(shape, 41);
    let got = backend.sum_last_axis_forward(&x, shape).expect("metal sum");
    let want = cpu_sum_last_axis_forward(&x, shape).unwrap();
    assert_close(&got, &want, 1e-3, "metal sum");
}

#[cfg(feature = "metal")]
#[test]
fn metal_backend_mean_last_axis_matches_cpu() {
    use autograd::backend::Backend;
    use autograd::backend_metal::MetalBackend;
    let backend = MetalBackend;
    let shape = &[6, 257];
    let x = make_rows(shape, 43);
    let got = backend
        .mean_last_axis_forward(&x, shape)
        .expect("metal mean");
    let want = cpu_mean_last_axis_forward(&x, shape).unwrap();
    assert_close(&got, &want, 1e-5, "metal mean");
}

#[cfg(feature = "metal")]
#[test]
fn metal_backend_rope_matches_cpu() {
    use autograd::backend::Backend;
    use autograd::backend_metal::MetalBackend;
    let backend = MetalBackend;
    let batch = 2_usize;
    let heads = 4_usize;
    let seq = 16_usize;
    let head_dim = 64_usize;
    let half_dim = head_dim / 2;
    let shape = &[batch, heads, seq, head_dim];
    let x = make_rows(shape, 55);
    let mut cos = Vec::with_capacity(seq * half_dim);
    let mut sin = Vec::with_capacity(seq * half_dim);
    for t in 0..seq {
        for i in 0..half_dim {
            let theta = (t as f32) * (0.02_f32 + (i as f32) * 0.01_f32);
            cos.push(theta.cos());
            sin.push(theta.sin());
        }
    }
    let got = backend
        .rope_forward(&x, shape, &cos, &sin)
        .expect("metal rope");
    let want = cpu_rope_forward(&x, shape, &cos, &sin).unwrap();
    assert_close(&got, &want, 1e-4, "metal rope");
}

#[cfg(feature = "metal")]
#[test]
fn metal_backend_gather_last_dim_matches_cpu() {
    use autograd::backend::Backend;
    use autograd::backend_metal::MetalBackend;
    let backend = MetalBackend;
    let shape = &[4_usize, 128_usize];
    let src = make_rows(shape, 71);
    let ids = [5_i32, 0, 127, 42];
    let got = backend
        .gather_last_dim_forward(&src, shape, &ids)
        .expect("metal gather");
    let want = cpu_gather_last_dim_forward(&src, shape, &ids).unwrap();
    assert_close(&got, &want, 1e-6, "metal gather");
}

// ──────────────────────────────────────────────────────────────────────
// scatter_add_rows_forward: CPU self-parity + GPU backend parity.
// Covers the two call-site shapes:
//   embedding_backward → feature_dim = hidden, aliased indices (token ids
//   repeating across positions → atomicAdd on GPU).
//   gather_last_dim_backward → feature_dim = 1, remapped flat indices
//   `i * vocab + original_indices[i]` (all unique by construction).
// ──────────────────────────────────────────────────────────────────────

#[test]
fn cpu_scatter_add_rows_embedding_shape() {
    // 5 prefix rows, feature_dim = 4, vocab = 3. Index 0 is hit by rows 0
    // and 2 so rows 0 and 2 of upstream must sum into bin 0 (exercises the
    // aliasing path the CUDA atomicAdd is there for).
    let prefix_rows = 5_usize;
    let feature_dim = 4_usize;
    let vocab = 3_usize;
    let upstream: Vec<f32> = (0..(prefix_rows * feature_dim) as i32)
        .map(|i| i as f32 * 0.1)
        .collect();
    let indices = [0_i32, 1, 0, 2, -1];
    let got =
        cpu_scatter_add_rows_forward(&upstream, prefix_rows, feature_dim, &indices, vocab).unwrap();
    let mut want = vec![0.0_f32; vocab * feature_dim];
    for (row, &id) in indices.iter().enumerate() {
        if id < 0 || (id as usize) >= vocab {
            continue;
        }
        let src_base = row * feature_dim;
        let dst_base = (id as usize) * feature_dim;
        for col in 0..feature_dim {
            want[dst_base + col] += upstream[src_base + col];
        }
    }
    assert_close(&got, &want, 1e-6, "cpu scatter_add_rows embedding");
}

#[test]
fn cpu_scatter_add_rows_gather_shape() {
    // gather_last_dim_backward: feature_dim = 1, flat_vocab = prefix * vocab.
    let prefix_rows = 4_usize;
    let vocab = 5_usize;
    let flat_vocab = prefix_rows * vocab;
    let upstream = [1.5_f32, -2.0, 0.25, 3.5];
    let original = [2_usize, 0, 4, 1];
    let flat_ids: Vec<i32> = original
        .iter()
        .enumerate()
        .map(|(i, &id)| (i * vocab + id) as i32)
        .collect();
    let got =
        cpu_scatter_add_rows_forward(&upstream, prefix_rows, 1, &flat_ids, flat_vocab).unwrap();
    let mut want = vec![0.0_f32; flat_vocab];
    for (i, &id) in original.iter().enumerate() {
        want[i * vocab + id] += upstream[i];
    }
    assert_close(&got, &want, 1e-6, "cpu scatter_add_rows gather");
}

#[test]
fn cpu_scatter_add_rows_oob_skips() {
    // Negative and out-of-range ids must be silently dropped (matches the
    // CUDA kernel's early-return and the pre-existing inline scatter in
    // embedding_backward).
    let upstream = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let indices = [-1_i32, 10, 0];
    let got = cpu_scatter_add_rows_forward(&upstream, 3, 2, &indices, 3).unwrap();
    // Only row 2 (id=0) contributed, adding [5.0, 6.0] into bin 0.
    let want = [5.0_f32, 6.0, 0.0, 0.0, 0.0, 0.0];
    assert_close(&got, &want, 1e-6, "cpu scatter_add_rows oob");
}

#[test]
fn cpu_backend_trait_scatter_add_rows_matches_reference() {
    // Confirms CpuBackend's trait method dispatches to the CPU reference
    // (no override) and gives identical output on the embedding shape.
    let backend = CpuBackend;
    let prefix_rows = 6_usize;
    let feature_dim = 3_usize;
    let vocab = 4_usize;
    let upstream: Vec<f32> = (0..(prefix_rows * feature_dim))
        .map(|i| (i as f32) * 0.5 - 1.0)
        .collect();
    let indices = [0_i32, 3, 3, 1, -1, 2];
    let got = backend
        .scatter_add_rows_forward(&upstream, prefix_rows, feature_dim, &indices, vocab)
        .unwrap();
    let want =
        cpu_scatter_add_rows_forward(&upstream, prefix_rows, feature_dim, &indices, vocab).unwrap();
    assert_close(&got, &want, 1e-6, "cpu backend scatter_add_rows");
}

#[cfg(feature = "metal")]
#[test]
fn metal_backend_scatter_add_rows_matches_cpu() {
    use autograd::backend::Backend;
    use autograd::backend_metal::MetalBackend;
    // Exercises the MLX `scatter_add` binding via `mlx_scatter_add_rows_f32`:
    // aliased indices (2 appears twice) confirm additive semantics; -1 and 7
    // exercise the host-side OOB/negative filter before hitting MLX.
    let backend = MetalBackend;
    let prefix_rows = 7_usize;
    let feature_dim = 8_usize;
    let vocab = 5_usize;
    let upstream = make_rows(&[prefix_rows, feature_dim], 4321);
    let indices = [0_i32, 4, 2, 2, -1, 7, 1];
    let got = backend
        .scatter_add_rows_forward(&upstream, prefix_rows, feature_dim, &indices, vocab)
        .expect("metal scatter_add_rows");
    let want =
        cpu_scatter_add_rows_forward(&upstream, prefix_rows, feature_dim, &indices, vocab).unwrap();
    assert_close(&got, &want, 1e-5, "metal scatter_add_rows");
}

#[cfg(feature = "metal")]
#[test]
fn metal_backend_scatter_add_rows_gather_shape() {
    // gather_last_dim_backward shape: feature_dim = 1, flat indices unique.
    // Exercises the `feature_dim == 1` edge case through the MLX path.
    use autograd::backend::Backend;
    use autograd::backend_metal::MetalBackend;
    let backend = MetalBackend;
    let prefix_rows = 4_usize;
    let vocab = 5_usize;
    let flat_vocab = prefix_rows * vocab;
    let upstream = [1.5_f32, -2.0, 0.25, 3.5];
    let flat_ids: Vec<i32> = [2_usize, 0, 4, 1]
        .iter()
        .enumerate()
        .map(|(i, &id)| (i * vocab + id) as i32)
        .collect();
    let got = backend
        .scatter_add_rows_forward(&upstream, prefix_rows, 1, &flat_ids, flat_vocab)
        .expect("metal scatter_add_rows gather shape");
    let want =
        cpu_scatter_add_rows_forward(&upstream, prefix_rows, 1, &flat_ids, flat_vocab).unwrap();
    assert_close(&got, &want, 1e-5, "metal scatter_add_rows gather shape");
}

#[cfg(feature = "metal")]
#[test]
fn metal_backend_scatter_add_rows_all_oob() {
    // Every index out of range → MLX is never called (host filter returns
    // zeros early). Confirms the empty-valid path doesn't regress.
    use autograd::backend::Backend;
    use autograd::backend_metal::MetalBackend;
    let backend = MetalBackend;
    let upstream = [1.0_f32, 2.0, 3.0, 4.0];
    let indices = [-1_i32, 10];
    let got = backend
        .scatter_add_rows_forward(&upstream, 2, 2, &indices, 3)
        .expect("metal scatter_add_rows all oob");
    assert_eq!(got, vec![0.0_f32; 6]);
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_backend_scatter_add_rows_matches_cpu() {
    use autograd::backend::Backend;
    use autograd::backend_cuda::CudaBackend;
    let backend = CudaBackend::new(0).expect("cuda ctx");
    let prefix_rows = 9_usize;
    let feature_dim = 16_usize;
    let vocab = 6_usize;
    let upstream = make_rows(&[prefix_rows, feature_dim], 9876);
    // Mix of in-range, OOB, and negative ids plus aliased indices (5 hit
    // three times) so the atomicAdd path is exercised.
    let indices = [0_i32, 5, 5, 2, -1, 8, 1, 3, 5];
    let got = backend
        .scatter_add_rows_forward(&upstream, prefix_rows, feature_dim, &indices, vocab)
        .expect("cuda scatter_add_rows");
    let want =
        cpu_scatter_add_rows_forward(&upstream, prefix_rows, feature_dim, &indices, vocab).unwrap();
    assert_close(&got, &want, 1e-4, "cuda scatter_add_rows");
}

// -----------------------------------------------------------------------------
// add_broadcast_forward: CPU reference + GPU backend parity.

// Naive reference that does not go through the backend trait — used to
// cross-check the trait default and every backend override.
fn reference_add_broadcast(a: &[f32], a_shape: &[usize], b: &[f32], b_shape: &[usize]) -> Vec<f32> {
    let a_size: usize = a_shape.iter().product();
    let rank_offset = a_shape.len() - b_shape.len();

    // Right-aligned contiguous strides for b, zero on broadcast axes.
    let mut b_strides = vec![0_usize; a_shape.len()];
    let mut stride = 1_usize;
    for i in (0..b_shape.len()).rev() {
        let dim = b_shape[i];
        b_strides[rank_offset + i] = if dim == 1 { 0 } else { stride };
        stride *= dim;
    }

    let mut out = vec![0.0_f32; a_size];
    for (i, slot) in out.iter_mut().enumerate() {
        // Unravel i in a_shape (row-major).
        let mut coords = vec![0_usize; a_shape.len()];
        let mut linear = i;
        for d in (0..a_shape.len()).rev() {
            coords[d] = linear % a_shape[d];
            linear /= a_shape[d];
        }
        let mut b_off = 0_usize;
        for d in 0..a_shape.len() {
            b_off += coords[d] * b_strides[d];
        }
        *slot = a[i] + b[b_off];
    }
    out
}

#[test]
fn cpu_add_broadcast_matches_reference() {
    use autograd::backend::Backend;
    let backend = CpuBackend;
    let cases: &[(&[usize], &[usize])] = &[
        (&[4, 8], &[8]),
        (&[2, 3, 4], &[1, 3, 4]),
        (&[3, 1, 5], &[5]),
    ];
    for (ai, (a_shape, b_shape)) in cases.iter().enumerate() {
        let a = make_rows(a_shape, 100 + ai as u64);
        let b = make_rows(b_shape, 200 + ai as u64);
        let got = backend
            .add_broadcast_forward(&a, a_shape, &b, b_shape)
            .expect("cpu add_broadcast");
        let want = reference_add_broadcast(&a, a_shape, &b, b_shape);
        assert_close(&got, &want, 1e-6, "cpu add_broadcast");
    }
}

#[cfg(feature = "metal")]
#[test]
fn metal_backend_add_broadcast_matches_cpu() {
    use autograd::backend::Backend;
    use autograd::backend_metal::MetalBackend;
    let backend = MetalBackend;
    let cases: &[(&[usize], &[usize])] = &[
        (&[4, 8], &[8]),
        (&[2, 3, 4], &[1, 3, 4]),
        (&[3, 1, 5], &[5]),
    ];
    for (ai, (a_shape, b_shape)) in cases.iter().enumerate() {
        let a = make_rows(a_shape, 1000 + ai as u64);
        let b = make_rows(b_shape, 2000 + ai as u64);
        let got = backend
            .add_broadcast_forward(&a, a_shape, &b, b_shape)
            .expect("metal add_broadcast");
        let want = reference_add_broadcast(&a, a_shape, &b, b_shape);
        assert_close(&got, &want, 1e-5, "metal add_broadcast");
    }
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_backend_add_broadcast_matches_cpu() {
    use autograd::backend::Backend;
    use autograd::backend_cuda::CudaBackend;
    let backend = CudaBackend::new(0).expect("cuda ctx");
    let cases: &[(&[usize], &[usize])] = &[
        (&[4, 8], &[8]),
        (&[2, 3, 4], &[1, 3, 4]),
        (&[3, 1, 5], &[5]),
    ];
    for (ai, (a_shape, b_shape)) in cases.iter().enumerate() {
        let a = make_rows(a_shape, 10_000 + ai as u64);
        let b = make_rows(b_shape, 20_000 + ai as u64);
        let got = backend
            .add_broadcast_forward(&a, a_shape, &b, b_shape)
            .expect("cuda add_broadcast");
        let want = reference_add_broadcast(&a, a_shape, &b, b_shape);
        assert_close(&got, &want, 1e-5, "cuda add_broadcast");
    }
}

// ──────────────────────────────────────────────────────────────────────
// matmul_backward parity tests.
// Three shapes, mirroring the forward parity block:
//   [8,16] @ [16,32]       — small 2D
//   [4,64] @ [64,64]       — square 2D
//   [3,8,16] @ [3,16,32]   — batched 3D
// Each case validates `grad_a = dC @ B^T` and `grad_b = A^T @ dC` against
// the CPU reference (`cpu_matmul_backward`).
// ──────────────────────────────────────────────────────────────────────

#[allow(clippy::type_complexity)]
fn matmul_backward_cases() -> Vec<(Vec<usize>, Vec<usize>, Vec<usize>, u64, u64, u64)> {
    vec![
        (
            vec![8, 16],
            vec![16, 32],
            vec![8, 32],
            0xA1A1,
            0xB2B2,
            0xC3C3,
        ),
        (
            vec![4, 64],
            vec![64, 64],
            vec![4, 64],
            0xA4A4,
            0xB5B5,
            0xC6C6,
        ),
        (
            vec![3, 8, 16],
            vec![3, 16, 32],
            vec![3, 8, 32],
            0xA7A7,
            0xB8B8,
            0xC9C9,
        ),
    ]
}

#[test]
fn cpu_backend_matmul_backward_matches_reference() {
    let backend = CpuBackend;
    for (a_shape, b_shape, c_shape, sa, sb, sc) in matmul_backward_cases() {
        let a = make_rows(&a_shape, sa);
        let b = make_rows(&b_shape, sb);
        let grad_out = make_rows(&c_shape, sc);
        let (got_a, got_b) = backend
            .matmul_backward(&a, &a_shape, &b, &b_shape, &grad_out, &c_shape, true, true)
            .expect("cpu backward");
        let (want_a, want_b) =
            cpu_matmul_backward(&a, &a_shape, &b, &b_shape, &grad_out, &c_shape, true, true)
                .expect("cpu ref");
        assert_close(
            &got_a,
            &want_a,
            1e-5,
            &format!("cpu backward grad_a {a_shape:?}"),
        );
        assert_close(
            &got_b,
            &want_b,
            1e-5,
            &format!("cpu backward grad_b {a_shape:?}"),
        );
    }
}

#[test]
fn cpu_matmul_backward_skips_unneeded_sides() {
    // need_grad_a=false → grad_a is empty; need_grad_b=false → grad_b empty.
    let a = make_rows(&[4, 6], 1);
    let b = make_rows(&[6, 5], 2);
    let grad_out = make_rows(&[4, 5], 3);
    let (grad_a, grad_b) =
        cpu_matmul_backward(&a, &[4, 6], &b, &[6, 5], &grad_out, &[4, 5], true, false).unwrap();
    assert_eq!(grad_a.len(), 24);
    assert!(grad_b.is_empty());
    let (grad_a, grad_b) =
        cpu_matmul_backward(&a, &[4, 6], &b, &[6, 5], &grad_out, &[4, 5], false, true).unwrap();
    assert!(grad_a.is_empty());
    assert_eq!(grad_b.len(), 30);
    let (grad_a, grad_b) =
        cpu_matmul_backward(&a, &[4, 6], &b, &[6, 5], &grad_out, &[4, 5], false, false).unwrap();
    assert!(grad_a.is_empty());
    assert!(grad_b.is_empty());
}

#[cfg(feature = "metal")]
#[test]
fn metal_backend_matmul_backward_matches_cpu() {
    use autograd::backend_metal::MetalBackend;
    let backend = MetalBackend;
    for (a_shape, b_shape, c_shape, sa, sb, sc) in matmul_backward_cases() {
        let a = make_rows(&a_shape, sa ^ 0xFEED);
        let b = make_rows(&b_shape, sb ^ 0xFEED);
        let grad_out = make_rows(&c_shape, sc ^ 0xFEED);
        let (got_a, got_b) = backend
            .matmul_backward(&a, &a_shape, &b, &b_shape, &grad_out, &c_shape, true, true)
            .expect("metal backward");
        let (want_a, want_b) =
            cpu_matmul_backward(&a, &a_shape, &b, &b_shape, &grad_out, &c_shape, true, true)
                .expect("cpu ref");
        assert_close(
            &got_a,
            &want_a,
            1e-3,
            &format!("metal backward grad_a {a_shape:?}"),
        );
        assert_close(
            &got_b,
            &want_b,
            1e-3,
            &format!("metal backward grad_b {a_shape:?}"),
        );
    }
}

#[cfg(feature = "metal")]
#[test]
fn metal_backend_matmul_backward_skip_sides() {
    // `need_grad_*=false` on one side must yield an empty vec; the other
    // side still matches the CPU reference exactly (mod tolerance).
    use autograd::backend_metal::MetalBackend;
    let backend = MetalBackend;
    let a = make_rows(&[4, 16], 1);
    let b = make_rows(&[16, 8], 2);
    let grad_out = make_rows(&[4, 8], 3);
    let (got_a, got_b) = backend
        .matmul_backward(&a, &[4, 16], &b, &[16, 8], &grad_out, &[4, 8], true, false)
        .expect("metal backward");
    assert_eq!(got_a.len(), 4 * 16);
    assert!(got_b.is_empty());
    let (want_a, _) =
        cpu_matmul_backward(&a, &[4, 16], &b, &[16, 8], &grad_out, &[4, 8], true, false).unwrap();
    assert_close(&got_a, &want_a, 1e-3, "metal backward skip grad_b");
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_backend_matmul_backward_matches_cpu() {
    use autograd::backend_cuda::CudaBackend;
    let backend = CudaBackend::new(0).expect("cuda ctx");
    for (a_shape, b_shape, c_shape, sa, sb, sc) in matmul_backward_cases() {
        let a = make_rows(&a_shape, sa ^ 0xBEEF);
        let b = make_rows(&b_shape, sb ^ 0xBEEF);
        let grad_out = make_rows(&c_shape, sc ^ 0xBEEF);
        let (got_a, got_b) = backend
            .matmul_backward(&a, &a_shape, &b, &b_shape, &grad_out, &c_shape, true, true)
            .expect("cuda backward");
        let (want_a, want_b) =
            cpu_matmul_backward(&a, &a_shape, &b, &b_shape, &grad_out, &c_shape, true, true)
                .expect("cpu ref");
        assert_close(
            &got_a,
            &want_a,
            1e-3,
            &format!("cuda backward grad_a {a_shape:?}"),
        );
        assert_close(
            &got_b,
            &want_b,
            1e-3,
            &format!("cuda backward grad_b {a_shape:?}"),
        );
    }
}
