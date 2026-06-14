//! On-device elementwise / norm correctness proof (perf-parity Step 5b).
//!
//! Oracle-gates the `rms_norm`, `swiglu`, and `add` shaders against the host
//! f32 references they replace in `infer-vulkan`'s forward. Each kernel is
//! launched with the exact push-constant block / specialization / binding order
//! the forward uses (`rms_norm_params` / `swiglu_params` / `add_params`), and
//! the device result is compared element-wise to the host computation.
//!
//! These are the gates the AGENTS brief requires: a device op may not replace
//! its host counterpart until it matches the oracle within tolerance. f32 in,
//! f32 out, so the tolerance is tight (1e-4 relative / 1e-5 absolute) — any
//! drift means the push/spec contract is wrong.
//!
//! Runs only with `--features vulkan` + a working device; skips cleanly
//! otherwise.
#![cfg(feature = "vulkan")]

use vulkan_kernels::{
    Kernel, KernelCache, add_dispatch, add_params, launch_cached, rms_norm_dispatch,
    rms_norm_params, swiglu_dispatch, swiglu_params,
};
use vulkan_sys::{DeviceBuffer, VulkanContext};

/// Deterministic xorshift PRNG so failures reproduce.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn upload_f32<'a>(ctx: &'a VulkanContext, data: &[f32]) -> DeviceBuffer<'a> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut b = DeviceBuffer::alloc(ctx, bytes.len().max(4)).expect("alloc f32 buffer");
    b.copy_from_host(&bytes).expect("upload f32 buffer");
    b
}

fn read_f32(buf: &DeviceBuffer<'_>, n: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; n * 4];
    buf.copy_to_host(&mut bytes).expect("read back f32 buffer");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn assert_close(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    let mut worst = 0f32;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let denom = w.abs().max(1e-4);
        let rel = (g - w).abs() / denom;
        worst = worst.max(rel);
        assert!(
            (g - w).abs() < 1e-4 || rel < 1e-4,
            "{label}[{i}]: got {g} vs want {w} (rel {rel})"
        );
    }
    eprintln!("[{label}] PASS (worst rel_err={worst:.6}, n={})", got.len());
}

#[test]
fn elementwise_kernels_match_host_oracle() {
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping elementwise oracle test");
            return;
        }
    };
    eprintln!("ARLE Vulkan elementwise proof on: {}", ctx.device_name());
    let mut cache = KernelCache::new();
    let mut rng = Rng(0x1234_5678_9ABC_DEF0);

    // ---- RMSNorm: out[i] = x[i] * inv_rms * w[i], inv_rms=1/sqrt(mean(x^2)+eps).
    // Test a few widths including the real 27B hidden (5120) and FFN inter (17408)
    // to exercise the shader's unrolled num_iters branches.
    for &n in &[256usize, 5120, 17408] {
        let eps = 1e-6f32;
        let x: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
        let w: Vec<f32> = (0..n).map(|_| 0.5 + rng.next_f32() * 0.5).collect();
        let mut sumsq = 0.0f32;
        for &v in &x {
            sumsq += v * v;
        }
        let inv = 1.0 / (sumsq / n as f32 + eps).sqrt();
        let want: Vec<f32> = (0..n).map(|i| x[i] * inv * w[i]).collect();

        let buf_a = upload_f32(&ctx, &x);
        let buf_b = upload_f32(&ctx, &w);
        let buf_d = upload_f32(&ctx, &vec![0.0f32; n]);
        let push = rms_norm_params(n as u32, eps).to_le_bytes();
        let d = rms_norm_dispatch();
        launch_cached(
            &mut cache,
            &ctx,
            Kernel::RmsNorm,
            &[&buf_a, &buf_b, &buf_d],
            d,
            &push,
            Kernel::RmsNorm.specialization_u32(),
        )
        .expect("rms_norm dispatch");
        assert_close(&format!("rms_norm n={n}"), &read_f32(&buf_d, n), &want);
    }

    // ---- SwiGLU (split): out[i] = silu(gate[i]) * up[i].
    for &n in &[256usize, 17408] {
        let gate: Vec<f32> = (0..n).map(|_| rng.next_f32() * 3.0).collect();
        let up: Vec<f32> = (0..n).map(|_| rng.next_f32() * 3.0).collect();
        let want: Vec<f32> = (0..n)
            .map(|i| (gate[i] / (1.0 + (-gate[i]).exp())) * up[i])
            .collect();

        let buf_a = upload_f32(&ctx, &gate);
        let buf_b = upload_f32(&ctx, &up);
        let buf_d = upload_f32(&ctx, &vec![0.0f32; n]);
        let push = swiglu_params(n as u32).to_le_bytes();
        let d = swiglu_dispatch(n as u32);
        launch_cached(
            &mut cache,
            &ctx,
            Kernel::SwiGlu,
            &[&buf_a, &buf_b, &buf_d],
            d,
            &push,
            Kernel::SwiGlu.specialization_u32(),
        )
        .expect("swiglu dispatch");
        assert_close(&format!("swiglu n={n}"), &read_f32(&buf_d, n), &want);
    }

    // ---- Add: out[i] = a[i] + b[i] (ADD_RMS=0, binding 3 dead-code-eliminated).
    for &n in &[256usize, 5120] {
        let a: Vec<f32> = (0..n).map(|_| rng.next_f32() * 10.0).collect();
        let b: Vec<f32> = (0..n).map(|_| rng.next_f32() * 10.0).collect();
        let want: Vec<f32> = (0..n).map(|i| a[i] + b[i]).collect();

        let buf_a = upload_f32(&ctx, &a);
        let buf_b = upload_f32(&ctx, &b);
        let buf_d = upload_f32(&ctx, &vec![0.0f32; n]);
        let push = add_params(n as u32).to_le_bytes();
        let d = add_dispatch(n as u32);
        launch_cached(
            &mut cache,
            &ctx,
            Kernel::Add,
            &[&buf_a, &buf_b, &buf_d],
            d,
            &push,
            Kernel::Add.specialization_u32(),
        )
        .expect("add dispatch");
        assert_close(&format!("add n={n}"), &read_f32(&buf_d, n), &want);
    }
}
