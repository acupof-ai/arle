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
    cpu_concat_axis2, cpu_embedding_forward, cpu_gather_last_dim_backward,
    cpu_gather_last_dim_forward, cpu_log_softmax_backward, cpu_log_softmax_forward_last_axis,
    cpu_matmul_backward, cpu_matmul_bt_backward, cpu_matmul_bt_forward, cpu_rms_norm_forward,
    cpu_scatter_add_rows_forward, cpu_slice, cpu_softmax_backward, cpu_softmax_forward_last_axis,
    cpu_transpose_swap,
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

fn cpu_argmax_last_dim(x: &[f32], shape: &[usize]) -> autograd::Result<Vec<f32>> {
    let vocab = *shape.last().ok_or(autograd::AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    let rows = shape.iter().product::<usize>() / vocab;
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let base = row * vocab;
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (idx, &value) in x[base..base + vocab].iter().enumerate() {
            if value > best_val {
                best_val = value;
                best_idx = idx;
            }
        }
        out.push(best_idx as f32);
    }
    Ok(out)
}

/// Combined `atol=1e-6 + rtol=1e-4` tolerance — matches `torch.allclose`
/// and the AdamW parity gate. Returns the worst `|diff| / tol` excess
/// ratio so the assert message can name the failing index.
fn max_err(dev: &[f32], host: &[f32]) -> (f32, f32, usize) {
    max_err_with_tol(dev, host, 1e-6, 1e-4)
}

/// Tolerance-parameterised variant of `max_err`. The log_softmax
/// backward path sums `upstream` across `vocab = 248 070` elements per
/// row inside the kernel, so accumulated `__expf` / `__fadd` rounding
/// is bounded by ~`sqrt(vocab) * f32_eps ≈ 5e-5`. Combined with the
/// `expf` vs `__expf` ~1-2 ULP gap on the per-element multiply, that
/// pushes the worst absolute diff to ~1e-5 at small grad magnitudes —
/// the strict AdamW gate (`atol=1e-6`) would false-positive on those
/// near-zero entries even though the relative error is well below
/// 1e-4. Keep the forward gates strict (their outputs are
/// probabilities normalized to 1, not the cancellation result of a
/// vocab-wide sum) and only relax the absolute floor for the
/// backward where the math demands it.
fn max_err_with_tol(dev: &[f32], host: &[f32], atol: f32, rtol: f32) -> (f32, f32, usize) {
    let mut worst_excess = 0.0_f32;
    let mut worst_abs = 0.0_f32;
    let mut worst_idx = 0_usize;
    for (i, (d, h)) in dev.iter().zip(host.iter()).enumerate() {
        let abs_diff = (d - h).abs();
        let tol = atol + rtol * h.abs();
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

#[test]
fn cuda_embedding_device_lazy_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_embedding_device_lazy_matches_cpu: no CUDA device");
        return;
    };

    let table_shape = vec![97, 32];
    let ids = vec![3, 7, 3, 96, 0];
    let table = rng_vec(0xEBD1_0001, table_shape.iter().product(), 1.0);
    let host_out =
        cpu_embedding_forward(&table, table_shape[0], table_shape[1], &ids).expect("cpu embedding");

    let table_h = backend.upload(&table, &table_shape).expect("upload table");
    let out_h = backend
        .embedding(&table_h, &table_shape, &ids)
        .expect("cuda embedding device");
    backend.eval(&[&out_h]).expect("cuda eval");
    let dev_out = backend.readback(&out_h).expect("embedding readback");

    let (excess, abs, idx) = max_err(&dev_out, &host_out);
    assert!(
        excess <= 1.0,
        "embedding device exceeds atol=1e-6 + rtol=1e-4 at idx {idx} \
         (|diff|={abs}, dev={}, host={}, excess_ratio={excess})",
        dev_out[idx],
        host_out[idx]
    );
}

#[test]
fn cuda_embedding_from_f32_ids_device_lazy_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_embedding_from_f32_ids_device_lazy_matches_cpu: no CUDA device");
        return;
    };

    let table_shape = vec![97, 32];
    let ids = vec![3, 7, 3, 96, 0];
    let ids_f32 = ids.iter().map(|&id| id as f32).collect::<Vec<_>>();
    let table = rng_vec(0xEBD1_0002, table_shape.iter().product(), 1.0);
    let host_out =
        cpu_embedding_forward(&table, table_shape[0], table_shape[1], &ids).expect("cpu embedding");

    let table_h = backend.upload(&table, &table_shape).expect("upload table");
    let ids_h = backend.upload(&ids_f32, &[ids.len()]).expect("upload ids");
    let out_h = backend
        .embedding_from_f32_ids(&table_h, &table_shape, &ids_h, ids.len())
        .expect("cuda embedding from f32 ids device");
    backend.eval(&[&out_h]).expect("cuda eval");
    let dev_out = backend
        .readback(&out_h)
        .expect("embedding f32 ids readback");

    let (excess, abs, idx) = max_err(&dev_out, &host_out);
    assert!(
        excess <= 1.0,
        "embedding_from_f32_ids device exceeds atol=1e-6 + rtol=1e-4 at idx {idx} \
         (|diff|={abs}, dev={}, host={}, excess_ratio={excess})",
        dev_out[idx],
        host_out[idx]
    );
}

#[test]
fn cuda_argmax_last_dim_device_lazy_matches_cpu_tie_breaking() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!(
            "skipping cuda_argmax_last_dim_device_lazy_matches_cpu_tie_breaking: no CUDA device"
        );
        return;
    };

    let shape = vec![2, 3, 257];
    let mut logits = rng_vec(0xA6A6_0001, shape.iter().product(), 2.0);
    let vocab = shape[2];
    logits[5] = 9.0;
    logits[17] = 9.0;
    logits[vocab + 128] = 8.0;
    let expected = cpu_argmax_last_dim(&logits, &shape).expect("cpu argmax");

    let logits_h = backend.upload(&logits, &shape).expect("upload logits");
    let out_h = backend
        .argmax_last_dim(&logits_h, &shape)
        .expect("cuda argmax_last_dim");
    backend.eval(&[&out_h]).expect("cuda eval");
    let dev_out = backend.readback(&out_h).expect("argmax readback");

    assert_eq!(dev_out, expected);
    assert_eq!(dev_out[0], 5.0, "ties must choose the smallest vocab index");
}

#[test]
fn cuda_write_scalar_at_device_lazy_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_write_scalar_at_device_lazy_matches_cpu: no CUDA device");
        return;
    };

    let dest = vec![0.0, 0.0, 0.0, 0.0, 0.0];
    let src = vec![42.0];
    let dest_h = backend.upload(&dest, &[dest.len()]).expect("upload dest");
    let src_h = backend.upload(&src, &[src.len()]).expect("upload src");
    let out_h = backend
        .write_scalar_at(&dest_h, &src_h, dest.len(), 3)
        .expect("cuda write scalar");
    backend.eval(&[&out_h]).expect("cuda eval");
    let dev_out = backend.readback(&out_h).expect("write readback");

    assert_eq!(dev_out, vec![0.0, 0.0, 0.0, 42.0, 0.0]);
}

#[test]
fn cuda_matmul_bt_device_lazy_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_matmul_bt_device_lazy_matches_cpu: no CUDA device");
        return;
    };

    let a_shape: Vec<usize> = vec![17, 19];
    let b_shape: Vec<usize> = vec![23, 19];
    let a = rng_vec(0xA17B19, a_shape.iter().product(), 1.0);
    let b = rng_vec(0xB23B19, b_shape.iter().product(), 1.0);

    let (host_out, host_shape) =
        cpu_matmul_bt_forward(&a, &a_shape, &b, &b_shape).expect("cpu matmul_bt");
    let a_h = backend.upload(&a, &a_shape).expect("upload a");
    let b_h = backend.upload(&b, &b_shape).expect("upload b");
    let (out_h, out_shape) = backend
        .matmul_bt(&a_h, &a_shape, &b_h, &b_shape)
        .expect("cuda matmul_bt");
    assert_eq!(out_shape, host_shape);
    backend.eval(&[&out_h]).expect("cuda eval");
    let dev_out = backend.readback(&out_h).expect("matmul_bt readback");

    let (excess, abs, idx) = max_err_with_tol(&dev_out, &host_out, 1e-4, 1e-4);
    assert!(
        excess <= 1.0,
        "matmul_bt exceeds atol=1e-4 + rtol=1e-4 at idx {idx} \
         (|diff|={abs}, dev={}, host={}, excess_ratio={excess})",
        dev_out[idx],
        host_out[idx]
    );
}

#[test]
fn cuda_log_softmax_last_axis_backward_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_log_softmax_last_axis_backward_matches_cpu: no CUDA device");
        return;
    };

    // Production shape: `[B=2, S=512, V=248070]`. The saved log_softmax
    // output (`y`) feeds the backward identity
    // `grad = upstream - exp(y) * sum(upstream, axis=-1)`. This is the
    // exact `1 015 MB` tensor that nsys identified as the single largest
    // DtoH per training step (see
    // `docs/research/2026-05-17-cuda-training-step-nsys-attribution.md`);
    // benching against the production shape catches reductions /
    // intrinsics drift the smaller unit-test shapes miss.
    let shape: Vec<usize> = vec![2, 512, 248_070];
    let size: usize = shape.iter().product();

    let x = rng_vec(0xDEC0DE, size, 4.0);
    let upstream = rng_vec(0xF00DCAFE, size, 1.0);

    // Host reference: cpu_log_softmax produces the saved `y`, then
    // cpu_log_softmax_backward consumes it with the upstream gradient.
    let log_softmax_output =
        cpu_log_softmax_forward_last_axis(&x, &shape).expect("cpu log_softmax forward");
    let host_grad = cpu_log_softmax_backward(&upstream, &log_softmax_output, &shape)
        .expect("cpu log_softmax_backward");

    // Device-resident path: forward stays lazy via M5.3b override; the
    // Wave 1 backward consumes the saved device handle without a host
    // roundtrip on either tensor.
    let x_h: DeviceHandle = backend.upload(&x, &shape).expect("upload x");
    let y_h = backend
        .log_softmax_last_axis(&x_h, &shape)
        .expect("cuda log_softmax forward (device lazy)");
    let upstream_h: DeviceHandle = backend.upload(&upstream, &shape).expect("upload upstream");
    let grad_h = backend
        .log_softmax_last_axis_backward(&upstream_h, &y_h, &shape)
        .expect("cuda log_softmax_last_axis_backward");
    backend.eval(&[&grad_h]).expect("cuda eval");
    let dev_grad = backend
        .readback(&grad_h)
        .expect("dev log_softmax grad readback");

    // Backward sums upstream across the full vocab inside the kernel
    // (`sqrt(248070)`-bounded `__fadd` rounding) and then multiplies by
    // `__expf(saved_output)`, so the cancellation result at small grad
    // magnitudes carries a few ULP of accumulated drift. Keep the
    // relative tolerance at the standard 1e-4 but bump the absolute
    // floor by 10× so near-zero entries don't false-positive — see the
    // `max_err_with_tol` rationale comment.
    let (excess, abs, idx) = max_err_with_tol(&dev_grad, &host_grad, 1e-5, 1e-4);
    assert!(
        excess <= 1.0,
        "log_softmax_last_axis_backward exceeds atol=1e-5 + rtol=1e-4 at idx {idx} \
         (|diff|={abs}, dev={}, host={}, excess_ratio={excess})",
        dev_grad[idx],
        host_grad[idx]
    );
}

#[test]
fn cuda_softmax_last_axis_backward_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_softmax_last_axis_backward_matches_cpu: no CUDA device");
        return;
    };

    // Attention-sized tile for the OPD moderate path after GQA repeat:
    // `[batch * heads, seq, seq] = [8, 5, 5]`. This is small in bytes but
    // load-bearing because a host fallback here demotes every upstream
    // attention projection grad before AdamW.
    let shape: Vec<usize> = vec![8, 5, 5];
    let size: usize = shape.iter().product();
    let x = rng_vec(0x50F7_0001, size, 2.0);
    let upstream = rng_vec(0x50F7_0002, size, 1.0);

    let softmax_output = cpu_softmax_forward_last_axis(&x, &shape).expect("cpu softmax");
    let host_grad =
        cpu_softmax_backward(&upstream, &softmax_output, &shape).expect("cpu softmax backward");

    let x_h: DeviceHandle = backend.upload(&x, &shape).expect("upload x");
    let y_h = backend
        .softmax_last_axis(&x_h, &shape)
        .expect("cuda softmax forward");
    let upstream_h = backend.upload(&upstream, &shape).expect("upload upstream");
    let grad_h = backend
        .softmax_last_axis_backward(&upstream_h, &y_h, &shape)
        .expect("cuda softmax backward");
    backend.eval(&[&grad_h]).expect("cuda eval");
    let dev_grad = backend.readback(&grad_h).expect("softmax grad readback");

    let (excess, abs, idx) = max_err_with_tol(&dev_grad, &host_grad, 1e-5, 1e-4);
    assert!(
        excess <= 1.0,
        "softmax_last_axis_backward exceeds atol=1e-5 + rtol=1e-4 at idx {idx} \
         (|diff|={abs}, dev={}, host={}, excess_ratio={excess})",
        dev_grad[idx],
        host_grad[idx]
    );
}

#[test]
fn cuda_gather_last_dim_backward_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_gather_last_dim_backward_matches_cpu: no CUDA device");
        return;
    };

    // Production shape: `[B=2, S=512, V=248070]`. The backward
    // zero-fills the full `[B, S, V]` grad and scatters the per-prefix
    // upstream scalar at `(row, ids[row])` — the same `1 GB` device
    // grad that feeds `log_softmax_last_axis_backward` in the
    // post-Wave-1 chain.
    let shape: Vec<usize> = vec![2, 512, 248_070];
    let prefix: usize = shape[..shape.len() - 1].iter().product();
    let vocab = *shape.last().unwrap();

    let upstream = rng_vec(0x501ACE, prefix, 1.0);
    let ids = rng_ids(0xBEAF, prefix, vocab as i32);

    let host_grad = cpu_gather_last_dim_backward(&upstream, &ids, &shape)
        .expect("cpu gather_last_dim_backward");

    // Device-resident path: upstream uploads (4 KB), the `[B, S, V]`
    // grad stays on-device after `alloc_zeros` + scatter kernel.
    let upstream_h: DeviceHandle = backend
        .upload(&upstream, &[shape[0], shape[1]])
        .expect("upload upstream");
    let grad_h = backend
        .gather_last_dim_backward(&upstream_h, &ids, &shape)
        .expect("cuda gather_last_dim_backward");
    backend.eval(&[&grad_h]).expect("cuda eval");
    let dev_grad = backend.readback(&grad_h).expect("dev gather grad readback");

    assert_eq!(
        dev_grad.len(),
        host_grad.len(),
        "gather backward grad length"
    );
    let (excess, abs, idx) = max_err(&dev_grad, &host_grad);
    assert!(
        excess <= 1.0,
        "gather_last_dim_backward exceeds atol=1e-6 + rtol=1e-4 at idx {idx} \
         (|diff|={abs}, dev={}, host={}, excess_ratio={excess})",
        dev_grad[idx],
        host_grad[idx]
    );
}

#[test]
fn cuda_matmul_backward_device_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_matmul_backward_device_matches_cpu: no CUDA device");
        return;
    };

    // [B*S=1024, K=2560] @ [K=2560, N=512] — production-scale rank-2 matmul
    // for the post-G3 device-resident gradient tape foundation. The forward
    // produces a `[1024, 512]` activation; backward computes
    // `grad_a:[1024, 2560]` and `grad_b:[2560, 512]` via two SGEMMs each.
    let a_shape: Vec<usize> = vec![1024, 2560];
    let b_shape: Vec<usize> = vec![2560, 512];
    let g_shape: Vec<usize> = vec![1024, 512];

    let a = rng_vec(0x12345, a_shape.iter().product(), 1.0);
    let b = rng_vec(0x67890, b_shape.iter().product(), 1.0);
    let g = rng_vec(0xABCDE, g_shape.iter().product(), 1.0);

    // Host reference: identical math to the device override, just via the
    // physically-transposed `cpu_matmul_forward` chain.
    let (host_grad_a, host_grad_b) =
        cpu_matmul_backward(&a, &a_shape, &b, &b_shape, &g, &g_shape, true, true)
            .expect("cpu matmul_backward");

    // Device-resident path: all three operands stay on-device, both grads
    // come back as unevaluated `DeviceHandle::Cuda`. Terminal eval performs
    // the single host fence per the M5.3b.11 batched-eval contract.
    let a_h: DeviceHandle = backend.upload(&a, &a_shape).expect("upload a");
    let b_h: DeviceHandle = backend.upload(&b, &b_shape).expect("upload b");
    let g_h: DeviceHandle = backend.upload(&g, &g_shape).expect("upload g");
    let (grad_a_h, grad_b_h) = backend
        .matmul_backward_device(&a_h, &a_shape, &b_h, &b_shape, &g_h, &g_shape, true, true)
        .expect("cuda matmul_backward_device");
    let grad_a_h = grad_a_h.expect("need_grad_a -> Some");
    let grad_b_h = grad_b_h.expect("need_grad_b -> Some");
    backend.eval(&[&grad_a_h, &grad_b_h]).expect("cuda eval");
    let dev_grad_a = backend.readback(&grad_a_h).expect("grad_a readback");
    let dev_grad_b = backend.readback(&grad_b_h).expect("grad_b readback");

    // cuBLAS double-SGEMM accumulates more rounding than a single sgemm on a
    // K=2560 reduction (~sqrt(K) * f32_eps ≈ 6e-6 per dot, scaled by the
    // input magnitude bound). Use combined `atol=1e-4 + rtol=1e-4` — mirrors
    // the Metal matmul-backward gate (Wave 1 reference).
    let (excess_a, abs_a, idx_a) = max_err_with_tol(&dev_grad_a, &host_grad_a, 1e-4, 1e-4);
    assert!(
        excess_a <= 1.0,
        "matmul_backward_device grad_a exceeds atol=1e-4 + rtol=1e-4 at idx {idx_a} \
         (|diff|={abs_a}, dev={}, host={}, excess_ratio={excess_a})",
        dev_grad_a[idx_a],
        host_grad_a[idx_a]
    );
    let (excess_b, abs_b, idx_b) = max_err_with_tol(&dev_grad_b, &host_grad_b, 1e-4, 1e-4);
    assert!(
        excess_b <= 1.0,
        "matmul_backward_device grad_b exceeds atol=1e-4 + rtol=1e-4 at idx {idx_b} \
         (|diff|={abs_b}, dev={}, host={}, excess_ratio={excess_b})",
        dev_grad_b[idx_b],
        host_grad_b[idx_b]
    );

    // need_grad_a=false / need_grad_b=true short-circuit: grad_a comes back
    // as `None` so the unused SGEMM is never launched.
    let (grad_a_h2, grad_b_h2) = backend
        .matmul_backward_device(&a_h, &a_shape, &b_h, &b_shape, &g_h, &g_shape, false, true)
        .expect("cuda matmul_backward_device (need_grad_a=false)");
    assert!(
        grad_a_h2.is_none(),
        "need_grad_a=false must short-circuit to None"
    );
    let grad_b_h2 = grad_b_h2.expect("need_grad_b -> Some");
    backend.eval(&[&grad_b_h2]).expect("cuda eval");
    let dev_grad_b2 = backend
        .readback(&grad_b_h2)
        .expect("grad_b readback (short)");
    let (excess_b2, abs_b2, idx_b2) = max_err_with_tol(&dev_grad_b2, &host_grad_b, 1e-4, 1e-4);
    assert!(
        excess_b2 <= 1.0,
        "matmul_backward_device grad_b (need_grad_a=false) exceeds atol=1e-4 + rtol=1e-4 at idx {idx_b2} \
         (|diff|={abs_b2}, dev={}, host={}, excess_ratio={excess_b2})",
        dev_grad_b2[idx_b2],
        host_grad_b[idx_b2]
    );
}

#[test]
fn cuda_matmul_bt_backward_device_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_matmul_bt_backward_device_matches_cpu: no CUDA device");
        return;
    };

    let a_shape: Vec<usize> = vec![64, 96];
    let b_shape: Vec<usize> = vec![128, 96];
    let g_shape: Vec<usize> = vec![64, 128];
    let a = rng_vec(0xABCD_0001, a_shape.iter().product(), 1.0);
    let b = rng_vec(0xABCD_0002, b_shape.iter().product(), 1.0);
    let g = rng_vec(0xABCD_0003, g_shape.iter().product(), 1.0);

    let (host_grad_a, host_grad_b) =
        cpu_matmul_bt_backward(&a, &a_shape, &b, &b_shape, &g, &g_shape, true, true)
            .expect("cpu matmul_bt_backward");

    let a_h = backend.upload(&a, &a_shape).expect("upload a");
    let b_h = backend.upload(&b, &b_shape).expect("upload b");
    let g_h = backend.upload(&g, &g_shape).expect("upload g");
    let (grad_a_h, grad_b_h) = backend
        .matmul_bt_backward_device(&a_h, &a_shape, &b_h, &b_shape, &g_h, &g_shape, true, true)
        .expect("cuda matmul_bt_backward_device");
    let grad_a_h = grad_a_h.expect("need_grad_a -> Some");
    let grad_b_h = grad_b_h.expect("need_grad_b -> Some");
    backend.eval(&[&grad_a_h, &grad_b_h]).expect("cuda eval");
    let dev_grad_a = backend.readback(&grad_a_h).expect("grad_a readback");
    let dev_grad_b = backend.readback(&grad_b_h).expect("grad_b readback");

    let (excess_a, abs_a, idx_a) = max_err_with_tol(&dev_grad_a, &host_grad_a, 1e-4, 1e-4);
    assert!(
        excess_a <= 1.0,
        "matmul_bt_backward_device grad_a exceeds atol=1e-4 + rtol=1e-4 at idx {idx_a} \
         (|diff|={abs_a}, dev={}, host={}, excess_ratio={excess_a})",
        dev_grad_a[idx_a],
        host_grad_a[idx_a]
    );
    let (excess_b, abs_b, idx_b) = max_err_with_tol(&dev_grad_b, &host_grad_b, 1e-4, 1e-4);
    assert!(
        excess_b <= 1.0,
        "matmul_bt_backward_device grad_b exceeds atol=1e-4 + rtol=1e-4 at idx {idx_b} \
         (|diff|={abs_b}, dev={}, host={}, excess_ratio={excess_b})",
        dev_grad_b[idx_b],
        host_grad_b[idx_b]
    );
}

#[test]
fn cuda_rms_norm_device_lazy_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_rms_norm_device_lazy_matches_cpu: no CUDA device");
        return;
    };

    let shape: Vec<usize> = vec![3, 5, 64];
    let hidden = *shape.last().unwrap();
    let x = rng_vec(0xCAFE_0001, shape.iter().product(), 1.0);
    let weight = rng_vec(0xCAFE_0002, hidden, 1.0);
    let eps = 1.0e-6;

    let host_out = cpu_rms_norm_forward(&x, &weight, &shape, eps).expect("cpu rms_norm");
    let x_h = backend.upload(&x, &shape).expect("upload x");
    let out_h = backend
        .rms_norm(&x_h, &weight, &shape, eps)
        .expect("cuda rms_norm");
    backend.eval(&[&out_h]).expect("cuda eval");
    let dev_out = backend.readback(&out_h).expect("rms_norm readback");

    let (excess, abs, idx) = max_err_with_tol(&dev_out, &host_out, 1e-5, 1e-4);
    assert!(
        excess <= 1.0,
        "rms_norm lazy forward exceeds atol=1e-5 + rtol=1e-4 at idx {idx} \
         (|diff|={abs}, dev={}, host={}, excess_ratio={excess})",
        dev_out[idx],
        host_out[idx]
    );
}

#[test]
fn cuda_layout_device_lazy_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_layout_device_lazy_matches_cpu: no CUDA device");
        return;
    };

    let shape: Vec<usize> = vec![2, 3, 4, 5];
    let x = rng_vec(0x1A70_0001, shape.iter().product(), 1.0);
    let x_h = backend.upload(&x, &shape).expect("upload x");

    let (host_transpose, transposed_shape) =
        cpu_transpose_swap(&x, &shape, 1, 2).expect("cpu transpose");
    let (transpose_h, transpose_shape) = backend
        .transpose_axes_swap(&x_h, &shape, 1, 2)
        .expect("cuda transpose");
    assert_eq!(transpose_shape, transposed_shape);
    backend.eval(&[&transpose_h]).expect("cuda eval transpose");
    let dev_transpose = backend.readback(&transpose_h).expect("transpose readback");
    let (excess_t, abs_t, idx_t) = max_err(&dev_transpose, &host_transpose);
    assert!(
        excess_t <= 1.0,
        "transpose exceeds atol=1e-6 + rtol=1e-4 at idx {idx_t} \
         (|diff|={abs_t}, dev={}, host={}, excess_ratio={excess_t})",
        dev_transpose[idx_t],
        host_transpose[idx_t]
    );

    let starts = [0, 1, 1, 0];
    let ends = [2, 3, 4, 5];
    let (host_slice, slice_shape) = cpu_slice(&x, &shape, &starts, &ends).expect("cpu slice");
    let slice_h = backend
        .slice(&x_h, &shape, &starts, &ends)
        .expect("cuda slice");
    backend.eval(&[&slice_h]).expect("cuda eval slice");
    let dev_slice = backend.readback(&slice_h).expect("slice readback");
    assert_eq!(dev_slice.len(), slice_shape.iter().product());
    let (excess_s, abs_s, idx_s) = max_err(&dev_slice, &host_slice);
    assert!(
        excess_s <= 1.0,
        "slice exceeds atol=1e-6 + rtol=1e-4 at idx {idx_s} \
         (|diff|={abs_s}, dev={}, host={}, excess_ratio={excess_s})",
        dev_slice[idx_s],
        host_slice[idx_s]
    );
}

#[test]
fn cuda_concat_axis2_device_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_concat_axis2_device_matches_cpu: no CUDA device");
        return;
    };

    let a_shape: Vec<usize> = vec![2, 3, 4, 5];
    let b_shape: Vec<usize> = vec![2, 3, 2, 5];
    let a = rng_vec(0xC0CA_0001, a_shape.iter().product(), 1.0);
    let b = rng_vec(0xC0CA_0002, b_shape.iter().product(), 1.0);
    let (host, out_shape) = cpu_concat_axis2(&a, &a_shape, &b, &b_shape).expect("cpu concat_axis2");

    let a_h = backend.upload(&a, &a_shape).expect("upload a");
    let b_h = backend.upload(&b, &b_shape).expect("upload b");
    let (out_h, dev_shape) = backend
        .concat_axis2(&a_h, &a_shape, &b_h, &b_shape)
        .expect("cuda concat_axis2");
    assert_eq!(dev_shape, out_shape);
    backend.eval(&[&out_h]).expect("cuda eval concat_axis2");
    let dev = backend.readback(&out_h).expect("concat_axis2 readback");

    let (excess, abs, idx) = max_err(&dev, &host);
    assert!(
        excess <= 1.0,
        "concat_axis2 exceeds atol=1e-6 + rtol=1e-4 at idx {idx} \
         (|diff|={abs}, dev={}, host={}, excess_ratio={excess})",
        dev[idx],
        host[idx]
    );
}

#[test]
fn cuda_slice_backward_device_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_slice_backward_device_matches_cpu: no CUDA device");
        return;
    };

    let input_shape: Vec<usize> = vec![2, 3, 4, 5];
    let starts = [0, 1, 1, 0];
    let ends = [2, 3, 4, 5];
    let upstream_shape: Vec<usize> = starts
        .iter()
        .zip(ends.iter())
        .map(|(&start, &end)| end - start)
        .collect();
    let upstream = rng_vec(0x1A70_BACC, upstream_shape.iter().product(), 1.0);

    let mut host_grad = vec![0.0; input_shape.iter().product()];
    for (out_index, &value) in upstream.iter().enumerate() {
        let mut coords = vec![0usize; upstream_shape.len()];
        let mut linear = out_index;
        for axis in (0..upstream_shape.len()).rev() {
            coords[axis] = linear % upstream_shape[axis];
            linear /= upstream_shape[axis];
        }
        let mut input_index = 0usize;
        let mut stride = 1usize;
        for axis in (0..input_shape.len()).rev() {
            input_index += (coords[axis] + starts[axis]) * stride;
            stride *= input_shape[axis];
        }
        host_grad[input_index] = value;
    }

    let upstream_h = backend
        .upload(&upstream, &upstream_shape)
        .expect("upload upstream");
    let grad_h = backend
        .slice_backward_device(&upstream_h, &input_shape, &starts, &ends)
        .expect("cuda slice_backward_device");
    backend.eval(&[&grad_h]).expect("cuda eval");
    let dev_grad = backend.readback(&grad_h).expect("slice grad readback");
    let (excess, abs, idx) = max_err(&dev_grad, &host_grad);
    assert!(
        excess <= 1.0,
        "slice_backward exceeds atol=1e-6 + rtol=1e-4 at idx {idx} \
         (|diff|={abs}, dev={}, host={}, excess_ratio={excess})",
        dev_grad[idx],
        host_grad[idx]
    );
}

#[test]
fn cuda_mean_backward_device_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_mean_backward_device_matches_cpu: no CUDA device");
        return;
    };

    // P3 production-derivative tile: the CE-loss chain produces a rank-0
    // scalar `mean` whose backward broadcasts `upstream / N` across
    // `[B=2, S=512]`. Smaller than the `[B, S, V]` softmax tile but still
    // multi-block (1024 / 256 = 4 blocks) so the launch grid is exercised.
    let output_shape: Vec<usize> = vec![2, 512];
    let elem_count: usize = output_shape.iter().product();

    // Single-element rank-0 upstream gradient. Deterministic value so the
    // host vs device math compares to identity.
    let upstream_scalar: f32 = 0.375_f32;
    let upstream_host = vec![upstream_scalar];

    // Host reference: broadcast `upstream / N` across `elem_count` slots.
    let inv_n = 1.0_f32 / elem_count as f32;
    let host_grad: Vec<f32> = vec![upstream_scalar * inv_n; elem_count];

    // Device path: rank-0 scalar upload, then `mean_backward_device`
    // broadcasts on-device via the 1D NVRTC kernel.
    let upstream_h: DeviceHandle = backend
        .upload(&upstream_host, &[])
        .expect("upload upstream scalar");
    let grad_h = backend
        .mean_backward_device(&upstream_h, &output_shape, elem_count)
        .expect("cuda mean_backward_device");
    backend.eval(&[&grad_h]).expect("cuda eval");
    let dev_grad = backend.readback(&grad_h).expect("dev mean_bwd readback");

    assert_eq!(dev_grad.len(), host_grad.len(), "mean_backward grad length");
    let (excess, abs, idx) = max_err(&dev_grad, &host_grad);
    assert!(
        excess <= 1.0,
        "mean_backward_device exceeds atol=1e-6 + rtol=1e-4 at idx {idx} \
         (|diff|={abs}, dev={}, host={}, excess_ratio={excess})",
        dev_grad[idx],
        host_grad[idx]
    );
}

#[test]
fn cuda_mul_scalar_backward_device_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_mul_scalar_backward_device_matches_cpu: no CUDA device");
        return;
    };

    // [N=10000] — small but multi-block (10000/256 ≈ 40 blocks) so we
    // exercise the launch grid the same way the `add_into_device` test does.
    let shape: Vec<usize> = vec![10_000];
    let size: usize = shape.iter().product();
    let scale: f32 = 0.5;

    let upstream = rng_vec(0xFEED, size, 2.0);
    let host_grad: Vec<f32> = upstream.iter().map(|u| u * scale).collect();

    let upstream_h: DeviceHandle = backend.upload(&upstream, &shape).expect("upload upstream");
    let grad_h = backend
        .mul_scalar_backward_device(&upstream_h, scale, &shape)
        .expect("cuda mul_scalar_backward_device");
    backend.eval(&[&grad_h]).expect("cuda eval");
    let dev_grad = backend
        .readback(&grad_h)
        .expect("dev mul_scalar_bwd readback");

    assert_eq!(
        dev_grad.len(),
        host_grad.len(),
        "mul_scalar_backward grad length"
    );
    let (excess, abs, idx) = max_err(&dev_grad, &host_grad);
    assert!(
        excess <= 1.0,
        "mul_scalar_backward_device exceeds atol=1e-6 + rtol=1e-4 at idx {idx} \
         (|diff|={abs}, dev={}, host={}, excess_ratio={excess})",
        dev_grad[idx],
        host_grad[idx]
    );
}

#[test]
fn cuda_add_into_device_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_add_into_device_matches_cpu: no CUDA device");
        return;
    };

    // [N=10000] — small but large enough to span multiple 256-thread blocks
    // (10000/256 ≈ 40 blocks) so we exercise the launch grid.
    let shape: Vec<usize> = vec![10_000];
    let size: usize = shape.iter().product();

    let dest = rng_vec(0xACC, size, 1.0);
    let src = rng_vec(0x9AB, size, 1.0);

    let host_sum: Vec<f32> = dest.iter().zip(src.iter()).map(|(d, s)| d + s).collect();

    let dest_h: DeviceHandle = backend.upload(&dest, &shape).expect("upload dest");
    let src_h: DeviceHandle = backend.upload(&src, &shape).expect("upload src");
    let out_h = backend
        .add_into_device(&dest_h, &src_h, &shape)
        .expect("cuda add_into_device");
    backend.eval(&[&out_h]).expect("cuda eval");
    let dev_sum = backend.readback(&out_h).expect("dev add_into readback");

    let (excess, abs, idx) = max_err(&dev_sum, &host_sum);
    assert!(
        excess <= 1.0,
        "add_into_device exceeds atol=1e-6 + rtol=1e-4 at idx {idx} \
         (|diff|={abs}, dev={}, host={}, excess_ratio={excess})",
        dev_sum[idx],
        host_sum[idx]
    );
}

#[test]
fn cuda_embedding_backward_device_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_embedding_backward_device_matches_cpu: no CUDA device");
        return;
    };

    // Production shape — matches the pretrain `small-25m` Qwen3.5 config:
    // `[B=2, S=512, H=160]` upstream → `[V=248070, H=160]` table grad.
    // 1024 token positions across vocab 248 070 → typical "sparse-into-
    // wide" scatter pattern. The kernel keeps the table grad on-device
    // via `atomicAdd`.
    const B: usize = 2;
    const S: usize = 512;
    const H: usize = 160;
    const V: usize = 248_070;
    let upstream_shape: Vec<usize> = vec![1, B * S, H];
    let upstream = rng_vec(0xE_B_07, B * S * H, 1.0);

    // Part A — uniformly random indices (production-ish). Most are unique
    // because B*S=1024 << V=248070, but the kernel must remain correct under
    // ANY id distribution. Use `rng_ids` for a deterministic LCG sequence.
    let ids_random = rng_ids(0x1d5, B * S, V as i32);
    let ids_random_i32: Vec<i32> = ids_random.iter().map(|&i| i).collect();
    let ids_random_usize: Vec<usize> = ids_random.iter().map(|&i| i as usize).collect();

    // Host reference uses the existing scatter_add reference path.
    let host_grad_random = cpu_scatter_add_rows_forward(&upstream, B * S, H, &ids_random_i32, V)
        .expect("cpu scatter_add (random ids)");

    let upstream_h: DeviceHandle = backend
        .upload(&upstream, &upstream_shape)
        .expect("upload upstream");
    let grad_h = backend
        .embedding_backward_device(&upstream_h, &ids_random_i32, V, H)
        .expect("cuda embedding_backward_device (random)");
    backend.eval(&[&grad_h]).expect("cuda eval (random)");
    let dev_grad_random = backend
        .readback(&grad_h)
        .expect("dev embedding_backward readback (random)");

    assert_eq!(
        dev_grad_random.len(),
        host_grad_random.len(),
        "embedding_backward grad length (random)"
    );
    // Allow 1e-4 absolute + 1e-4 relative to absorb the atomicAdd
    // accumulation order divergence vs the host scatter loop (host
    // accumulates in row-order; the atomicAdd order on-GPU is
    // nondeterministic across blocks, so identical sum semantics yield
    // identical results only modulo f32 round-off).
    let (excess, abs, idx) = max_err_with_tol(&dev_grad_random, &host_grad_random, 1e-4, 1e-4);
    assert!(
        excess <= 1.0,
        "embedding_backward_device (random ids) exceeds atol=1e-4 + rtol=1e-4 at idx {idx} \
         (|diff|={abs}, dev={}, host={}, excess_ratio={excess})",
        dev_grad_random[idx],
        host_grad_random[idx]
    );

    // Part B — deliberately duplicated indices (the atomicAdd correctness
    // gate). All B*S=1024 token positions hit the same 4 vocab slots, with
    // many duplicates per slot. If atomicAdd were absent (race condition
    // between blocks), the resulting grad would have lost increments and
    // diverge well beyond our tolerance. The canonical "the appears 800
    // times" stress case.
    let _ = ids_random_usize; // (only needed for the comment above)
    let dup_targets: [i32; 4] = [0, 42, 1337, (V - 1) as i32];
    let ids_dup_i32: Vec<i32> = (0..(B * S))
        .map(|i| dup_targets[i % dup_targets.len()])
        .collect();

    let host_grad_dup = cpu_scatter_add_rows_forward(&upstream, B * S, H, &ids_dup_i32, V)
        .expect("cpu scatter_add (duplicates)");

    let grad_h_dup = backend
        .embedding_backward_device(&upstream_h, &ids_dup_i32, V, H)
        .expect("cuda embedding_backward_device (duplicates)");
    backend
        .eval(&[&grad_h_dup])
        .expect("cuda eval (duplicates)");
    let dev_grad_dup = backend
        .readback(&grad_h_dup)
        .expect("dev embedding_backward readback (duplicates)");

    // Spot-check: each of the 4 duplicate target rows received B*S/4=256
    // increments. Sum the first column across the 4 target rows in both
    // host and device output and assert they agree to atol=1e-3 (256
    // f32 sums × `half_range=1.0` upstream ≈ `sqrt(256) * f32_eps * 1.0
    // = 2e-6` per element; the cross-block reordering bumps that by ~10×
    // in the worst case).
    let host_dup_sum: f32 = dup_targets
        .iter()
        .map(|&t| host_grad_dup[t as usize * H])
        .sum();
    let dev_dup_sum: f32 = dup_targets
        .iter()
        .map(|&t| dev_grad_dup[t as usize * H])
        .sum();
    assert!(
        (host_dup_sum - dev_dup_sum).abs() <= 1e-3,
        "embedding_backward_device duplicate-ids stress: host_sum={host_dup_sum} \
         dev_sum={dev_dup_sum} diff={}",
        (host_dup_sum - dev_dup_sum).abs()
    );

    let (excess_d, abs_d, idx_d) = max_err_with_tol(&dev_grad_dup, &host_grad_dup, 1e-4, 1e-4);
    assert!(
        excess_d <= 1.0,
        "embedding_backward_device (duplicate ids) exceeds atol=1e-4 + rtol=1e-4 at idx {idx_d} \
         (|diff|={abs_d}, dev={}, host={}, excess_ratio={excess_d})",
        dev_grad_dup[idx_d],
        host_grad_dup[idx_d]
    );
}

#[test]
fn cuda_add_broadcast_backward_device_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_add_broadcast_backward_device_matches_cpu: no CUDA device");
        return;
    };

    // Production shape: `[B=2, S=512, H=160]` upstream → `[H=160]` reduce.
    // The grad_b reduction sums B*S=1024 elements per output position,
    // sqrt(1024) ≈ 32, so accumulated f32 round-off is ~32 * f32_eps ≈ 4e-6
    // per output element (multiplied by `half_range=1.0` upstream
    // magnitude). atol=1e-5 + rtol=1e-4 absorbs that with margin.
    const B: usize = 2;
    const S: usize = 512;
    const H: usize = 160;
    let a_shape: Vec<usize> = vec![B, S, H];
    let b_shape: Vec<usize> = vec![H];

    let upstream = rng_vec(0xAB_BC_DE, B * S * H, 1.0);

    // Host reference: for each h ∈ [0, H), grad_b[h] = sum_{b,s} upstream[b,s,h].
    let mut host_grad_b = vec![0.0_f32; H];
    for b_idx in 0..B {
        for s_idx in 0..S {
            for h_idx in 0..H {
                host_grad_b[h_idx] += upstream[(b_idx * S + s_idx) * H + h_idx];
            }
        }
    }

    let upstream_h: DeviceHandle = backend
        .upload(&upstream, &a_shape)
        .expect("upload upstream");
    let grad_b_h = backend
        .add_broadcast_backward_device(&upstream_h, &a_shape, &b_shape)
        .expect("cuda add_broadcast_backward_device");
    backend.eval(&[&grad_b_h]).expect("cuda eval");
    let dev_grad_b = backend
        .readback(&grad_b_h)
        .expect("dev add_broadcast_backward readback");

    assert_eq!(
        dev_grad_b.len(),
        host_grad_b.len(),
        "add_broadcast_backward grad_b length"
    );
    let (excess, abs, idx) = max_err_with_tol(&dev_grad_b, &host_grad_b, 1e-5, 1e-4);
    assert!(
        excess <= 1.0,
        "add_broadcast_backward_device exceeds atol=1e-5 + rtol=1e-4 at idx {idx} \
         (|diff|={abs}, dev={}, host={}, excess_ratio={excess})",
        dev_grad_b[idx],
        host_grad_b[idx]
    );
}

/// Wave 2.0 parity gate: `Backend::adamw_step_device` (device-resident
/// gradient handle) must match `Backend::adamw_step` (host-slice gradient)
/// bit-for-bit on the same numerical inputs. The only thing that changes
/// between the two paths is where `grad` lives before the kernel launch —
/// the kernel itself (`adamw_step_f32`) is shared, so any numerical
/// divergence here is a real bug (e.g. wrong slice passed, dtod seed
/// missed, eval contract violated). Tolerance: `atol=1e-6 + rtol=1e-4`
/// (same as `cuda_adamw_step_matches_cpu_5_steps`).
#[test]
fn cuda_adamw_step_device_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_adamw_step_device_matches_cpu: no CUDA device");
        return;
    };

    use autograd::backend::cpu_adamw_step_in_place;

    let shape = vec![128, 64];
    let size: usize = shape.iter().product();
    const STEPS: usize = 5;
    const LR: f32 = 3e-4;
    const BETA1: f32 = 0.9;
    const BETA2: f32 = 0.95;
    const EPS: f32 = 1e-8;
    const WD: f32 = 0.01;

    let param_init = rng_vec(0xA11CE, size, 0.1);
    let grads_per_step: Vec<Vec<f32>> = (0..STEPS)
        .map(|step| {
            rng_vec(
                0xBEEF ^ (step as u64).wrapping_mul(0x9E3779B97F4A7C15),
                size,
                0.02,
            )
        })
        .collect();

    // Host reference (identical to `cuda_adamw_step_matches_cpu_5_steps`).
    let mut host_param = param_init.clone();
    let mut host_m = vec![0.0_f32; size];
    let mut host_v = vec![0.0_f32; size];
    for (step, grad) in grads_per_step.iter().enumerate().take(STEPS) {
        let t = step as i32 + 1;
        let bc1 = 1.0 - BETA1.powi(t);
        let bc2 = 1.0 - BETA2.powi(t);
        cpu_adamw_step_in_place(
            &mut host_param,
            &mut host_m,
            &mut host_v,
            grad,
            LR,
            BETA1,
            BETA2,
            EPS,
            WD,
            bc1,
            bc2,
        );
    }

    // Device chain: same as the host-slice test, except the gradient is
    // pre-uploaded as a DeviceHandle::Cuda and fed through
    // `adamw_step_device`. No `clone_htod` happens inside the kernel
    // launch in this path.
    let mut param_h: DeviceHandle = backend
        .upload(&param_init, &shape)
        .expect("upload initial param");
    let mut m_h: DeviceHandle = backend
        .upload(&vec![0.0_f32; size], &shape)
        .expect("upload zero m");
    let mut v_h: DeviceHandle = backend
        .upload(&vec![0.0_f32; size], &shape)
        .expect("upload zero v");

    for (step, grad) in grads_per_step.iter().enumerate().take(STEPS) {
        let t = step as i32 + 1;
        let bc1 = 1.0 - BETA1.powi(t);
        let bc2 = 1.0 - BETA2.powi(t);
        // Upload the per-step grad as a device handle (simulates a
        // device-resident grad produced by embedding_backward_device /
        // add_broadcast_backward_device upstream).
        let grad_h: DeviceHandle = backend
            .upload(grad, &shape)
            .expect("upload grad as device handle");
        let (new_param, new_m, new_v) = backend
            .adamw_step_device(
                &param_h, &m_h, &v_h, &grad_h, &shape, LR, BETA1, BETA2, EPS, WD, bc1, bc2,
            )
            .expect("cuda adamw_step_device");
        param_h = new_param;
        m_h = new_m;
        v_h = new_v;
    }

    backend
        .eval(&[&param_h, &m_h, &v_h])
        .expect("cuda eval after adamw_step_device chain");

    let dev_param = backend.readback(&param_h).expect("dev param readback");
    let dev_m = backend.readback(&m_h).expect("dev m readback");
    let dev_v = backend.readback(&v_h).expect("dev v readback");

    let (param_excess, param_abs, param_idx) = max_err(&dev_param, &host_param);
    let (m_excess, m_abs, m_idx) = max_err(&dev_m, &host_m);
    let (v_excess, v_abs, v_idx) = max_err(&dev_v, &host_v);

    assert!(
        param_excess <= 1.0,
        "param exceeds atol=1e-6 + rtol=1e-4 at idx {param_idx} \
         (|diff|={param_abs}, dev={}, host={}, excess_ratio={param_excess}) \
         after {STEPS} cuda adamw_step_device calls",
        dev_param[param_idx],
        host_param[param_idx]
    );
    assert!(
        m_excess <= 1.0,
        "m exceeds atol=1e-6 + rtol=1e-4 at idx {m_idx} \
         (|diff|={m_abs}, dev={}, host={}, excess_ratio={m_excess}) \
         after {STEPS} cuda adamw_step_device calls",
        dev_m[m_idx],
        host_m[m_idx]
    );
    assert!(
        v_excess <= 1.0,
        "v exceeds atol=1e-6 + rtol=1e-4 at idx {v_idx} \
         (|diff|={v_abs}, dev={}, host={}, excess_ratio={v_excess}) \
         after {STEPS} cuda adamw_step_device calls",
        dev_v[v_idx],
        host_v[v_idx]
    );
}

// ============================================================================
// Wave 2.1 — atomic batch port of 7 host-only backward ops to device-lazy.
// Each test mirrors the production-representative shape per the wave-2.1
// brief. Tolerance follows `max_err_with_tol(atol=1e-5, rtol=1e-4)` for
// the multi-pass / trig ops (rms_norm, rope) which accumulate enough
// rounding to need the absolute floor; pure elementwise (silu, gelu,
// sigmoid, exp, mul) stay on the strict `max_err` tolerance.
// ============================================================================

/// SiLU backward parity: `dx[i] = upstream[i] * silu'(x[i])` where
/// `silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))`. Per-layer
/// activation shape `[B=2, S=512, H=160]` from the wave-2.1 brief.
#[test]
fn cuda_silu_backward_device_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_silu_backward_device_matches_cpu: no CUDA device");
        return;
    };

    let shape: Vec<usize> = vec![2, 512, 160];
    let size: usize = shape.iter().product();
    let x = rng_vec(0x511, size, 4.0);
    let upstream = rng_vec(0x511_511, size, 1.0);

    let host_grad: Vec<f32> = x
        .iter()
        .zip(upstream.iter())
        .map(|(&xv, &up)| {
            let sigmoid = 1.0_f32 / (1.0 + (-xv).exp());
            let deriv = sigmoid + (xv * sigmoid * (1.0 - sigmoid));
            up * deriv
        })
        .collect();

    let x_h: DeviceHandle = backend.upload(&x, &shape).expect("upload x");
    let up_h: DeviceHandle = backend.upload(&upstream, &shape).expect("upload upstream");
    let grad_h = backend
        .silu_backward_device(&up_h, &x_h, &shape)
        .expect("cuda silu_backward_device");
    backend.eval(&[&grad_h]).expect("cuda eval");
    let dev_grad = backend.readback(&grad_h).expect("silu_bwd readback");

    let (excess, abs, idx) = max_err(&dev_grad, &host_grad);
    assert!(
        excess <= 1.0,
        "silu_backward_device exceeds atol=1e-6 + rtol=1e-4 at idx {idx} \
         (|diff|={abs}, dev={}, host={}, excess_ratio={excess})",
        dev_grad[idx],
        host_grad[idx]
    );
}

/// GELU (erf form) backward parity. Per-layer activation shape.
#[test]
fn cuda_gelu_backward_device_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_gelu_backward_device_matches_cpu: no CUDA device");
        return;
    };

    const INV_SQRT_2: f32 = 0.707_106_77;
    const INV_SQRT_2PI: f32 = 0.398_942_3;

    let shape: Vec<usize> = vec![2, 512, 160];
    let size: usize = shape.iter().product();
    let x = rng_vec(0x6e1, size, 4.0);
    let upstream = rng_vec(0x6e1_6e1, size, 1.0);

    let host_grad: Vec<f32> = x
        .iter()
        .zip(upstream.iter())
        .map(|(&xv, &up)| {
            let erf_term = libm::erff(xv * INV_SQRT_2);
            let exp_term = (-0.5_f32 * xv * xv).exp();
            let deriv = 0.5 * (1.0 + erf_term) + (xv * INV_SQRT_2PI * exp_term);
            up * deriv
        })
        .collect();

    let x_h: DeviceHandle = backend.upload(&x, &shape).expect("upload x");
    let up_h: DeviceHandle = backend.upload(&upstream, &shape).expect("upload upstream");
    let grad_h = backend
        .gelu_backward_device(&up_h, &x_h, &shape)
        .expect("cuda gelu_backward_device");
    backend.eval(&[&grad_h]).expect("cuda eval");
    let dev_grad = backend.readback(&grad_h).expect("gelu_bwd readback");

    // erff intrinsic vs libm::erff: ~1-2 ULP gap. Use the strict gate
    // (1e-6 + 1e-4) — well within the absolute floor.
    let (excess, abs, idx) = max_err(&dev_grad, &host_grad);
    assert!(
        excess <= 1.0,
        "gelu_backward_device exceeds atol=1e-6 + rtol=1e-4 at idx {idx} \
         (|diff|={abs}, dev={}, host={}, excess_ratio={excess})",
        dev_grad[idx],
        host_grad[idx]
    );
}

/// Sigmoid backward parity: consumes the saved output `y`.
#[test]
fn cuda_sigmoid_backward_device_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_sigmoid_backward_device_matches_cpu: no CUDA device");
        return;
    };

    let shape: Vec<usize> = vec![2, 512, 160];
    let size: usize = shape.iter().product();
    // Saved y = sigmoid(x); generate by sampling x and computing sigmoid
    // host-side so we feed the kernel exactly the same `y` the device
    // sigmoid forward would have produced.
    let x = rng_vec(0x515, size, 4.0);
    let y: Vec<f32> = x.iter().map(|&v| 1.0_f32 / (1.0 + (-v).exp())).collect();
    let upstream = rng_vec(0x515_515, size, 1.0);

    let host_grad: Vec<f32> = y
        .iter()
        .zip(upstream.iter())
        .map(|(&yv, &up)| up * yv * (1.0 - yv))
        .collect();

    let y_h: DeviceHandle = backend.upload(&y, &shape).expect("upload y");
    let up_h: DeviceHandle = backend.upload(&upstream, &shape).expect("upload upstream");
    let grad_h = backend
        .sigmoid_backward_device(&up_h, &y_h, &shape)
        .expect("cuda sigmoid_backward_device");
    backend.eval(&[&grad_h]).expect("cuda eval");
    let dev_grad = backend.readback(&grad_h).expect("sigmoid_bwd readback");

    let (excess, abs, idx) = max_err(&dev_grad, &host_grad);
    assert!(
        excess <= 1.0,
        "sigmoid_backward_device exceeds atol=1e-6 + rtol=1e-4 at idx {idx} \
         (|diff|={abs}, dev={}, host={}, excess_ratio={excess})",
        dev_grad[idx],
        host_grad[idx]
    );
}

/// Exp backward parity: consumes the saved output `y = exp(x)`.
#[test]
fn cuda_exp_backward_device_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_exp_backward_device_matches_cpu: no CUDA device");
        return;
    };

    let shape: Vec<usize> = vec![2, 512, 160];
    let size: usize = shape.iter().product();
    // Bound x so exp doesn't overflow into the inf band that would make
    // the parity check pointless (any test value > ~88 returns inf in f32).
    let x = rng_vec(0xE17, size, 2.0);
    let y: Vec<f32> = x.iter().map(|&v| v.exp()).collect();
    let upstream = rng_vec(0xE17_E17, size, 1.0);

    let host_grad: Vec<f32> = y
        .iter()
        .zip(upstream.iter())
        .map(|(&yv, &up)| up * yv)
        .collect();

    let y_h: DeviceHandle = backend.upload(&y, &shape).expect("upload y");
    let up_h: DeviceHandle = backend.upload(&upstream, &shape).expect("upload upstream");
    let grad_h = backend
        .exp_backward_device(&up_h, &y_h, &shape)
        .expect("cuda exp_backward_device");
    backend.eval(&[&grad_h]).expect("cuda eval");
    let dev_grad = backend.readback(&grad_h).expect("exp_bwd readback");

    let (excess, abs, idx) = max_err(&dev_grad, &host_grad);
    assert!(
        excess <= 1.0,
        "exp_backward_device exceeds atol=1e-6 + rtol=1e-4 at idx {idx} \
         (|diff|={abs}, dev={}, host={}, excess_ratio={excess})",
        dev_grad[idx],
        host_grad[idx]
    );
}

/// Mul backward parity: `grad_a = upstream * b`, `grad_b = upstream * a`.
#[test]
fn cuda_mul_backward_device_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_mul_backward_device_matches_cpu: no CUDA device");
        return;
    };

    let shape: Vec<usize> = vec![2, 512, 160];
    let size: usize = shape.iter().product();
    let a = rng_vec(0x77a, size, 2.0);
    let b = rng_vec(0x77b, size, 2.0);
    let upstream = rng_vec(0x77c, size, 1.0);

    let host_grad_a: Vec<f32> = upstream
        .iter()
        .zip(b.iter())
        .map(|(&up, &bv)| up * bv)
        .collect();
    let host_grad_b: Vec<f32> = upstream
        .iter()
        .zip(a.iter())
        .map(|(&up, &av)| up * av)
        .collect();

    let a_h: DeviceHandle = backend.upload(&a, &shape).expect("upload a");
    let b_h: DeviceHandle = backend.upload(&b, &shape).expect("upload b");
    let up_h: DeviceHandle = backend.upload(&upstream, &shape).expect("upload upstream");

    // Both sides.
    let (grad_a_h, grad_b_h) = backend
        .mul_backward_device(&up_h, &a_h, &b_h, &shape, true, true)
        .expect("cuda mul_backward_device");
    let grad_a_h = grad_a_h.expect("need_grad_a -> Some");
    let grad_b_h = grad_b_h.expect("need_grad_b -> Some");
    backend.eval(&[&grad_a_h, &grad_b_h]).expect("cuda eval");
    let dev_grad_a = backend.readback(&grad_a_h).expect("grad_a readback");
    let dev_grad_b = backend.readback(&grad_b_h).expect("grad_b readback");

    let (excess_a, abs_a, idx_a) = max_err(&dev_grad_a, &host_grad_a);
    assert!(
        excess_a <= 1.0,
        "mul_backward_device grad_a exceeds atol=1e-6 + rtol=1e-4 at idx {idx_a} \
         (|diff|={abs_a}, dev={}, host={}, excess_ratio={excess_a})",
        dev_grad_a[idx_a],
        host_grad_a[idx_a]
    );
    let (excess_b, abs_b, idx_b) = max_err(&dev_grad_b, &host_grad_b);
    assert!(
        excess_b <= 1.0,
        "mul_backward_device grad_b exceeds atol=1e-6 + rtol=1e-4 at idx {idx_b} \
         (|diff|={abs_b}, dev={}, host={}, excess_ratio={excess_b})",
        dev_grad_b[idx_b],
        host_grad_b[idx_b]
    );

    // need_grad_a=false short-circuit.
    let (grad_a_h2, grad_b_h2) = backend
        .mul_backward_device(&up_h, &a_h, &b_h, &shape, false, true)
        .expect("cuda mul_backward_device (need_grad_a=false)");
    assert!(
        grad_a_h2.is_none(),
        "need_grad_a=false must short-circuit to None"
    );
    let grad_b_h2 = grad_b_h2.expect("need_grad_b -> Some");
    backend.eval(&[&grad_b_h2]).expect("cuda eval");
    let dev_grad_b2 = backend.readback(&grad_b_h2).expect("grad_b short readback");
    let (excess_b2, _, idx_b2) = max_err(&dev_grad_b2, &host_grad_b);
    assert!(excess_b2 <= 1.0, "short-circuit grad_b idx {idx_b2}");
}

/// RMSNorm backward parity: both `grad_x` and `grad_w`. Production-ish
/// per-layer norm shape `[B=2, S=512, H=160]` + weight `[160]`. Tolerance
/// bumped to `atol=1e-5` because the two-pass reduce + per-row sqrt
/// accumulates ~`sqrt(H)*f32_eps` ≈ 1.5e-6 per output element, scaled by
/// the host loop's row-major order vs the kernel's parallel reduction
/// order.
#[test]
fn cuda_rms_norm_backward_device_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_rms_norm_backward_device_matches_cpu: no CUDA device");
        return;
    };

    let shape: Vec<usize> = vec![2, 512, 160];
    let hidden = 160_usize;
    let size: usize = shape.iter().product();
    let eps = 1e-6_f32;

    let x = rng_vec(0x88a, size, 1.0);
    let weight = rng_vec(0x88b, hidden, 1.0);
    let upstream = rng_vec(0x88c, size, 1.0);

    // Host reference (matches `cpu_rmsnorm_backward` math + the new
    // `Backend::rms_norm_backward_device` default impl).
    let rows = size / hidden;
    let mut inv_rms = vec![0.0_f32; rows];
    for r in 0..rows {
        let base = r * hidden;
        let mut sum_sq = 0.0_f32;
        for c in 0..hidden {
            let v = x[base + c];
            sum_sq += v * v;
        }
        inv_rms[r] = 1.0 / ((sum_sq / hidden as f32) + eps).sqrt();
    }
    let mut host_grad_x = vec![0.0_f32; size];
    for r in 0..rows {
        let base = r * hidden;
        let inv = inv_rms[r];
        let mut dot = 0.0_f32;
        for c in 0..hidden {
            dot += upstream[base + c] * weight[c] * x[base + c];
        }
        let correction = inv * inv * dot / hidden as f32;
        for c in 0..hidden {
            host_grad_x[base + c] =
                (inv * upstream[base + c] * weight[c]) - (x[base + c] * inv * correction);
        }
    }
    let mut host_grad_w = vec![0.0_f32; hidden];
    for r in 0..rows {
        let base = r * hidden;
        let inv = inv_rms[r];
        for c in 0..hidden {
            host_grad_w[c] += upstream[base + c] * x[base + c] * inv;
        }
    }

    let x_h: DeviceHandle = backend.upload(&x, &shape).expect("upload x");
    let w_h: DeviceHandle = backend.upload(&weight, &[hidden]).expect("upload w");
    let up_h: DeviceHandle = backend.upload(&upstream, &shape).expect("upload upstream");

    let (grad_x_h, grad_w_h) = backend
        .rms_norm_backward_device(&up_h, &x_h, &w_h, &shape, eps, true, true)
        .expect("cuda rms_norm_backward_device");
    let grad_x_h = grad_x_h.expect("need_grad_x -> Some");
    let grad_w_h = grad_w_h.expect("need_grad_w -> Some");
    backend.eval(&[&grad_x_h, &grad_w_h]).expect("cuda eval");
    let dev_grad_x = backend.readback(&grad_x_h).expect("grad_x readback");
    let dev_grad_w = backend.readback(&grad_w_h).expect("grad_w readback");

    let (excess_x, abs_x, idx_x) = max_err_with_tol(&dev_grad_x, &host_grad_x, 1e-5, 1e-4);
    assert!(
        excess_x <= 1.0,
        "rms_norm_backward_device grad_x exceeds atol=1e-5 + rtol=1e-4 at idx {idx_x} \
         (|diff|={abs_x}, dev={}, host={}, excess_ratio={excess_x})",
        dev_grad_x[idx_x],
        host_grad_x[idx_x]
    );
    let (excess_w, abs_w, idx_w) = max_err_with_tol(&dev_grad_w, &host_grad_w, 1e-5, 1e-4);
    assert!(
        excess_w <= 1.0,
        "rms_norm_backward_device grad_w exceeds atol=1e-5 + rtol=1e-4 at idx {idx_w} \
         (|diff|={abs_w}, dev={}, host={}, excess_ratio={excess_w})",
        dev_grad_w[idx_w],
        host_grad_w[idx_w]
    );
}

/// RoPE forward device-lazy parity. OPD rollout decode depends on this path
/// staying device-resident; the default `Backend::rope` fallback readbacks
/// the activation and breaks CUDA Graph capture.
#[test]
fn cuda_rope_device_lazy_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_rope_device_lazy_matches_cpu: no CUDA device");
        return;
    };

    use autograd::backend::cpu_rope_forward;
    let batch = 2_usize;
    let heads = 5_usize;
    let seq = 64_usize;
    let head_dim = 32_usize;
    let half_dim = head_dim / 2;
    let shape: Vec<usize> = vec![batch, heads, seq, head_dim];
    let size: usize = shape.iter().product();
    let cache_len = seq * half_dim;

    let x = rng_vec(0x69d, size, 1.0);
    let cos = rng_vec(0x69c, cache_len, 1.0);
    let sin = rng_vec(0x69b, cache_len, 1.0);

    let host_out = cpu_rope_forward(&x, &shape, &cos, &sin).expect("cpu rope forward");

    let x_h: DeviceHandle = backend.upload(&x, &shape).expect("upload x");
    let out_h = backend
        .rope(&x_h, &shape, &cos, &sin)
        .expect("cuda rope device lazy");
    backend.eval(&[&out_h]).expect("cuda eval");
    let dev_out = backend.readback(&out_h).expect("rope readback");

    let (excess, abs, idx) = max_err_with_tol(&dev_out, &host_out, 1e-5, 1e-4);
    assert!(
        excess <= 1.0,
        "rope device lazy exceeds atol=1e-5 + rtol=1e-4 at idx {idx} \
         (|diff|={abs}, dev={}, host={}, excess_ratio={excess})",
        dev_out[idx],
        host_out[idx]
    );
}

/// RoPE backward parity. Per the wave-2.1 brief: `[B=2, n_heads=5,
/// S=512, head_dim=32]` (NeoX layout, full-head rotation).
#[test]
fn cuda_rope_backward_device_matches_cpu() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_rope_backward_device_matches_cpu: no CUDA device");
        return;
    };

    use autograd::backend::cpu_rope_forward;
    let batch = 2_usize;
    let heads = 5_usize;
    let seq = 512_usize;
    let head_dim = 32_usize;
    let half_dim = head_dim / 2;
    let shape: Vec<usize> = vec![batch, heads, seq, head_dim];
    let size: usize = shape.iter().product();
    let cache_len = seq * half_dim;

    let upstream = rng_vec(0x69e, size, 1.0);
    let cos = rng_vec(0x69c, cache_len, 1.0);
    let sin = rng_vec(0x69b, cache_len, 1.0);

    // Host reference: cpu_rope_forward(upstream, cos, -sin).
    let neg_sin: Vec<f32> = sin.iter().map(|&v| -v).collect();
    let host_grad = cpu_rope_forward(&upstream, &shape, &cos, &neg_sin).expect("cpu rope backward");

    let up_h: DeviceHandle = backend.upload(&upstream, &shape).expect("upload upstream");
    let grad_h = backend
        .rope_backward_device(&up_h, &shape, &cos, &sin)
        .expect("cuda rope_backward_device");
    backend.eval(&[&grad_h]).expect("cuda eval");
    let dev_grad = backend.readback(&grad_h).expect("rope_bwd readback");

    // Trig accumulation: one mul-add per element, no reduction. atol=1e-5
    // + rtol=1e-4 absorbs the cos/sin intrinsic ULP gap with margin.
    let (excess, abs, idx) = max_err_with_tol(&dev_grad, &host_grad, 1e-5, 1e-4);
    assert!(
        excess <= 1.0,
        "rope_backward_device exceeds atol=1e-5 + rtol=1e-4 at idx {idx} \
         (|diff|={abs}, dev={}, host={}, excess_ratio={excess})",
        dev_grad[idx],
        host_grad[idx]
    );
}
