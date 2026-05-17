//! CUDA M5.3b device-lazy parity gate for the CE-loss chain ops.
//!
//! `Backend::softmax_last_axis`, `Backend::log_softmax_last_axis` and
//! `Backend::gather_last_dim` on `CudaBackend` reuse the existing NVRTC
//! kernels (`softmax_last_axis_f32` / `log_softmax_last_axis_f32` /
//! `gather_last_dim_f32`) but skip the host `readback → compute →
//! upload` roundtrip that the default trait fallback does. This file is
//! the numerical gate that says "device-lazy result equals CPU
//! reference, on the exact production shape we expect to bench against".
//!
//! Shape rationale (`[B=2, S=512, V=248070]`): matches
//! `--preset small-25m --batch 2 --seq 512 --grad-accum-steps 16`
//! per [the G3 wins entry](../../docs/experience/wins/2026-05-17-bench-pretrain-g3-cuda-adamw-step.md).
//! `1 015 808` rows × `248 070` cols ≈ `1 GB` of fp32 logits — exactly
//! the host-readback chain this milestone targets.
//!
//! Tolerance follows the AdamW gate (combined `atol=1e-6 + rtol=1e-4`,
//! mirrors `torch.allclose`). Pure-relative gates fail on tiny values
//! where the GPU kernel's `__expf` / `__logf` intrinsics differ from
//! libm by ~1 ULP; the absolute floor keeps that case green.

#![cfg(all(feature = "cuda", not(feature = "no-cuda")))]

use autograd::backend::{
    cpu_gather_last_dim_forward, cpu_log_softmax_forward_last_axis, cpu_softmax_forward_last_axis,
};
use autograd::backend_cuda::CudaBackend;
use autograd::{Backend, DeviceHandle};

/// Deterministic LCG → uniform `(-half_range, half_range)` floats.
/// Same seed → same sequence → host vs device replay identically.
fn rng_vec(seed: u64, n: usize, half_range: f32) -> Vec<f32> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((s >> 32) as u32 as f32) / (u32::MAX as f32);
        out.push((u - 0.5) * 2.0 * half_range);
    }
    out
}

/// Deterministic LCG → uniform int32 in `[0, upper)`.
fn rng_ids(seed: u64, n: usize, upper: i32) -> Vec<i32> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (s >> 32) as u32;
        out.push((u % (upper as u32)) as i32);
    }
    out
}

/// Combined `atol=1e-6 + rtol=1e-4` tolerance — matches `torch.allclose`
/// and the AdamW parity gate. Returns the worst `|diff| / tol` excess
/// ratio so the assert message can name the failing index.
fn max_err(dev: &[f32], host: &[f32]) -> (f32, f32, usize) {
    const ATOL: f32 = 1e-6;
    const RTOL: f32 = 1e-4;
    let mut worst_excess = 0.0_f32;
    let mut worst_abs = 0.0_f32;
    let mut worst_idx = 0_usize;
    for (i, (d, h)) in dev.iter().zip(host.iter()).enumerate() {
        let abs_diff = (d - h).abs();
        let tol = ATOL + RTOL * h.abs();
        let excess = abs_diff / tol;
        if excess > worst_excess {
            worst_excess = excess;
            worst_abs = abs_diff;
            worst_idx = i;
        }
    }
    (worst_excess, worst_abs, worst_idx)
}

#[test]
fn cuda_softmax_last_axis_device_lazy_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_softmax_last_axis_device_lazy_matches_cpu: no CUDA device");
        return;
    };

    // Production shape: `[B=2, S=512, V=248070]`. ~1 GB of fp32 logits,
    // matching the small-25m Qwen3.5 CE-loss path.
    let shape: Vec<usize> = vec![2, 512, 248_070];
    let size: usize = shape.iter().product();

    let x = rng_vec(0xA11CE, size, 4.0);

    // Host reference.
    let host_out = cpu_softmax_forward_last_axis(&x, &shape).expect("cpu softmax");

    // Device-lazy path.
    let x_h: DeviceHandle = backend.upload(&x, &shape).expect("upload x");
    let out_h = backend
        .softmax_last_axis(&x_h, &shape)
        .expect("cuda softmax_last_axis (device lazy)");
    backend.eval(&[&out_h]).expect("cuda eval");
    let dev_out = backend.readback(&out_h).expect("dev softmax readback");

    let (excess, abs, idx) = max_err(&dev_out, &host_out);
    assert!(
        excess <= 1.0,
        "softmax_last_axis exceeds atol=1e-6 + rtol=1e-4 at idx {idx} \
         (|diff|={abs}, dev={}, host={}, excess_ratio={excess})",
        dev_out[idx],
        host_out[idx]
    );
}

#[test]
fn cuda_log_softmax_last_axis_device_lazy_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_log_softmax_last_axis_device_lazy_matches_cpu: no CUDA device");
        return;
    };

    let shape: Vec<usize> = vec![2, 512, 248_070];
    let size: usize = shape.iter().product();

    let x = rng_vec(0xBEEF, size, 4.0);

    let host_out = cpu_log_softmax_forward_last_axis(&x, &shape).expect("cpu log_softmax");

    let x_h: DeviceHandle = backend.upload(&x, &shape).expect("upload x");
    let out_h = backend
        .log_softmax_last_axis(&x_h, &shape)
        .expect("cuda log_softmax_last_axis (device lazy)");
    backend.eval(&[&out_h]).expect("cuda eval");
    let dev_out = backend.readback(&out_h).expect("dev log_softmax readback");

    let (excess, abs, idx) = max_err(&dev_out, &host_out);
    assert!(
        excess <= 1.0,
        "log_softmax_last_axis exceeds atol=1e-6 + rtol=1e-4 at idx {idx} \
         (|diff|={abs}, dev={}, host={}, excess_ratio={excess})",
        dev_out[idx],
        host_out[idx]
    );
}

#[test]
fn cuda_gather_last_dim_device_lazy_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_gather_last_dim_device_lazy_matches_cpu: no CUDA device");
        return;
    };

    let shape: Vec<usize> = vec![2, 512, 248_070];
    let prefix: usize = shape[..shape.len() - 1].iter().product();
    let vocab = *shape.last().unwrap();
    let size: usize = shape.iter().product();

    let src = rng_vec(0xC0FFEE, size, 4.0);
    let ids = rng_ids(0xD00D, prefix, vocab as i32);

    // Host reference.
    let host_out = cpu_gather_last_dim_forward(&src, &shape, &ids).expect("cpu gather");

    // Device-lazy path: only `ids` crosses PCIe; `src` stays on-device.
    let src_h: DeviceHandle = backend.upload(&src, &shape).expect("upload src");
    let out_h = backend
        .gather_last_dim(&src_h, &shape, &ids)
        .expect("cuda gather_last_dim (device lazy)");
    backend.eval(&[&out_h]).expect("cuda eval");
    let dev_out = backend.readback(&out_h).expect("dev gather readback");

    // Gather is a point-copy — exact match expected modulo any
    // host/device fp identity. Keep the same combined-tolerance gate so a
    // single drift on `__ldg` would be caught.
    assert_eq!(dev_out.len(), host_out.len(), "gather output length");
    let (excess, abs, idx) = max_err(&dev_out, &host_out);
    assert!(
        excess <= 1.0,
        "gather_last_dim exceeds atol=1e-6 + rtol=1e-4 at idx {idx} \
         (|diff|={abs}, dev={}, host={}, excess_ratio={excess})",
        dev_out[idx],
        host_out[idx]
    );
}
