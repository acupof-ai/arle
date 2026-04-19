//! Backend matmul parity tests. The CPU reference is authoritative; each
//! gated backend must match it to within `1e-3` relative tolerance on the
//! three shapes we actually hit in TinyLM training: small 2D, square 2D,
//! and batched rank-3.

use autograd::{
    CpuBackend,
    backend::{Backend, cpu_matmul_forward},
};

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
