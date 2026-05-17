//! CUDA `Backend::adamw_step` parity gate.
//!
//! The fused NVRTC kernel in `backend_cuda/kernels/adamw.cu` must match
//! the host reference (`cpu_adamw_step_in_place` in
//! `crates/autograd/src/backend.rs`) to **≤1e-4 relative error** across
//! 5 sequential steps on a `[hidden=128, batch=64]` random shape. This
//! is the numerical gate for the #1 CUDA training-throughput
//! optimization (kill the AdamW host-readback fallback) per the
//! optimization roadmap in
//! `docs/experience/wins/2026-05-17-bench-pretrain-qwen35-25m-cuda-baseline.md`.

#![cfg(all(feature = "cuda", not(feature = "no-cuda")))]

use autograd::backend::cpu_adamw_step_in_place;
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

#[test]
fn cuda_adamw_step_matches_cpu_5_steps() {
    // Skip cleanly if no CUDA device is reachable — the test binary is
    // compiled even when remote machine isn't available; this keeps the
    // Mac type-check path green.
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping cuda_adamw_step_matches_cpu_5_steps: no CUDA device");
        return;
    };

    // Shape: [hidden=128, batch=64] flattened = 8192 elements. Bigger
    // than 1 block (256 threads) so we exercise the multi-block grid
    // path; small enough to keep allocations cheap on shared CI hosts.
    let shape = vec![128, 64];
    let size: usize = shape.iter().product();
    const STEPS: usize = 5;
    const LR: f32 = 3e-4;
    const BETA1: f32 = 0.9;
    const BETA2: f32 = 0.95;
    const EPS: f32 = 1e-8;
    const WD: f32 = 0.01;

    // Same RNG seeds drive both replays.
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

    // ---- Host reference: run cpu_adamw_step_in_place STEPS times ----
    let mut host_param = param_init.clone();
    let mut host_m = vec![0.0_f32; size];
    let mut host_v = vec![0.0_f32; size];
    for step in 0..STEPS {
        let t = step as i32 + 1;
        let bc1 = 1.0 - BETA1.powi(t);
        let bc2 = 1.0 - BETA2.powi(t);
        cpu_adamw_step_in_place(
            &mut host_param,
            &mut host_m,
            &mut host_v,
            &grads_per_step[step],
            LR,
            BETA1,
            BETA2,
            EPS,
            WD,
            bc1,
            bc2,
        );
    }

    // ---- Device: chain STEPS calls to backend.adamw_step ----
    let mut param_h: DeviceHandle = backend
        .upload(&param_init, &shape)
        .expect("upload initial param");
    let mut m_h: DeviceHandle = backend
        .upload(&vec![0.0_f32; size], &shape)
        .expect("upload zero m");
    let mut v_h: DeviceHandle = backend
        .upload(&vec![0.0_f32; size], &shape)
        .expect("upload zero v");

    for step in 0..STEPS {
        let t = step as i32 + 1;
        let bc1 = 1.0 - BETA1.powi(t);
        let bc2 = 1.0 - BETA2.powi(t);
        let (new_param, new_m, new_v) = backend
            .adamw_step(
                &param_h,
                &m_h,
                &v_h,
                &grads_per_step[step],
                &shape,
                LR,
                BETA1,
                BETA2,
                EPS,
                WD,
                bc1,
                bc2,
            )
            .expect("cuda adamw_step");
        param_h = new_param;
        m_h = new_m;
        v_h = new_v;
    }

    // Single terminal sync (mirrors AdamW::step_device's batched eval).
    backend
        .eval(&[&param_h, &m_h, &v_h])
        .expect("cuda eval after adamw chain");

    let dev_param = backend.readback(&param_h).expect("dev param readback");
    let dev_m = backend.readback(&m_h).expect("dev m readback");
    let dev_v = backend.readback(&v_h).expect("dev v readback");

    // Combined absolute + relative tolerance (industry-standard
    // `torch.allclose` / `numpy.isclose`). Pure relative gate explodes
    // on tiny values near zero where fp32 FMA contraction in the NVRTC
    // kernel can differ by ~1 ULP from the host's separate mul+add
    // (e.g. host=-3.09e-7 / dev=-3.09e-7 has |diff|=7.6e-11 = 0.025%
    // relative). The absolute floor (1e-6) keeps that case green while
    // the relative gate (1e-4) still catches any real divergence on
    // meaningful magnitudes.
    fn max_err(dev: &[f32], host: &[f32]) -> (f32, f32, usize) {
        const ATOL: f32 = 1e-6;
        const RTOL: f32 = 1e-4;
        let mut worst_excess = 0.0_f32;
        let mut worst_abs = 0.0_f32;
        let mut worst_idx = 0_usize;
        for (i, (d, h)) in dev.iter().zip(host.iter()).enumerate() {
            let abs_diff = (d - h).abs();
            let tol = ATOL + (RTOL * h.abs());
            let excess = abs_diff / tol; // > 1 means failed
            if excess > worst_excess {
                worst_excess = excess;
                worst_abs = abs_diff;
                worst_idx = i;
            }
        }
        (worst_excess, worst_abs, worst_idx)
    }

    let (param_excess, param_abs, param_idx) = max_err(&dev_param, &host_param);
    let (m_excess, m_abs, m_idx) = max_err(&dev_m, &host_m);
    let (v_excess, v_abs, v_idx) = max_err(&dev_v, &host_v);

    assert!(
        param_excess <= 1.0,
        "param exceeds atol=1e-6 + rtol=1e-4 at idx {param_idx} \
         (|diff|={param_abs}, dev={}, host={}, excess_ratio={param_excess}) \
         after {STEPS} cuda adamw_step calls",
        dev_param[param_idx],
        host_param[param_idx]
    );
    assert!(
        m_excess <= 1.0,
        "m exceeds atol=1e-6 + rtol=1e-4 at idx {m_idx} \
         (|diff|={m_abs}, dev={}, host={}, excess_ratio={m_excess}) \
         after {STEPS} cuda adamw_step calls",
        dev_m[m_idx],
        host_m[m_idx]
    );
    assert!(
        v_excess <= 1.0,
        "v exceeds atol=1e-6 + rtol=1e-4 at idx {v_idx} \
         (|diff|={v_abs}, dev={}, host={}, excess_ratio={v_excess}) \
         after {STEPS} cuda adamw_step calls",
        dev_v[v_idx],
        host_v[v_idx]
    );
}
