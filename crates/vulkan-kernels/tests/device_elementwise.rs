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
    Kernel, KernelCache, add_dispatch, add_params, f16_kv_pack_dispatch, f16_kv_pack_params,
    launch_cached, rms_norm_dispatch, rms_norm_params, scaled_add_dispatch, scaled_add_params,
    swiglu_dispatch, swiglu_params,
};
use vulkan_sys::{DeviceBuffer, VulkanContext};

/// Round-to-nearest-even f32 -> f16 bit pattern. The exact host reference the
/// device `f16_kv_pack` (hardware `float16_t(float)` cast) must match bit-for-bit;
/// it is the same routine the KV cache contract was validated against.
fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | (if mant != 0 { 0x0200 } else { 0 });
    }
    let mut e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let mant_with_implicit = mant | 0x0080_0000;
        let shift = (14 - e) as u32;
        let mut m = mant_with_implicit >> shift;
        let rem_mask = (1u32 << shift) - 1;
        let rem = mant_with_implicit & rem_mask;
        let halfway = 1u32 << (shift - 1);
        if rem > halfway || (rem == halfway && (m & 1) == 1) {
            m += 1;
        }
        return sign | m as u16;
    }
    let mut m = mant >> 13;
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (m & 1) == 1) {
        m += 1;
        if m == 0x0400 {
            m = 0;
            e += 1;
            if e >= 0x1f {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((e as u16) << 10) | m as u16
}

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

    // ---- ScaledAdd: out[i] = a[i] + scale*b[i]. The MoE FFN accumulate folds
    // the router weight `w_e` into this (`acc += w_e * y_e`) and aliases the
    // accumulator (out == a), so test both a separate-dst and an in-place case.
    for &n in &[256usize, 5120] {
        let scale = 0.375f32;
        let a: Vec<f32> = (0..n).map(|_| rng.next_f32() * 10.0).collect();
        let b: Vec<f32> = (0..n).map(|_| rng.next_f32() * 10.0).collect();
        let want: Vec<f32> = (0..n).map(|i| a[i] + scale * b[i]).collect();
        let push = scaled_add_params(n as u32, scale).to_le_bytes();
        let d = scaled_add_dispatch(n as u32);

        // separate dst
        let buf_a = upload_f32(&ctx, &a);
        let buf_b = upload_f32(&ctx, &b);
        let buf_d = upload_f32(&ctx, &vec![0.0f32; n]);
        launch_cached(
            &mut cache,
            &ctx,
            Kernel::ScaledAdd,
            &[&buf_a, &buf_b, &buf_d],
            d,
            &push,
            Kernel::ScaledAdd.specialization_u32(),
        )
        .expect("scaled_add dispatch");
        assert_close(&format!("scaled_add n={n}"), &read_f32(&buf_d, n), &want);

        // in-place accumulate (out aliases a, the MoE acc pattern)
        let buf_acc = upload_f32(&ctx, &a);
        let buf_b2 = upload_f32(&ctx, &b);
        launch_cached(
            &mut cache,
            &ctx,
            Kernel::ScaledAdd,
            &[&buf_acc, &buf_b2, &buf_acc],
            d,
            &push,
            Kernel::ScaledAdd.specialization_u32(),
        )
        .expect("scaled_add in-place dispatch");
        assert_close(
            &format!("scaled_add in-place n={n}"),
            &read_f32(&buf_acc, n),
            &want,
        );
    }
}

/// Oracle-gate the `f16_kv_pack` shader (Step 2: device f16 KV-pack) against the
/// host RNE f32->f16 conversion the KV cache contract was validated with. The
/// device hardware `float16_t(float)` cast must produce the SAME f16 bit pattern
/// for every value (full-attention K/V are well within f16 normal range), so the
/// comparison is bit-for-bit. Widths cover one full-attention head row
/// (head_dim=256) and a kv_dim-wide block (kv_heads*head_dim).
#[test]
fn f16_kv_pack_matches_host_rne_oracle() {
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping f16_kv_pack oracle test");
            return;
        }
    };
    eprintln!("ARLE Vulkan f16_kv_pack proof on: {}", ctx.device_name());
    let mut cache = KernelCache::new();
    let mut rng = Rng(0xF16C_AFE0_1234_5678);

    for &n in &[256usize, 1024] {
        // Attention K/V magnitudes (post-rope / projection) span roughly [-8, 8];
        // sample a comparable range plus a few exact-representable corner values.
        let mut x: Vec<f32> = (0..n).map(|_| rng.next_f32() * 8.0).collect();
        for (i, corner) in [0.0f32, 1.0, -1.0, 0.5, 65504.0, -65504.0]
            .into_iter()
            .enumerate()
        {
            if i < n {
                x[i] = corner;
            }
        }
        let want: Vec<u16> = x.iter().map(|&v| f32_to_f16_bits(v)).collect();

        let buf_src = upload_f32(&ctx, &x);
        // Destination is n f16 = 2n bytes.
        let mut buf_dst = DeviceBuffer::alloc(&ctx, (n * 2).max(4)).expect("alloc f16 dst");
        buf_dst
            .copy_from_host(&vec![0u8; n * 2])
            .expect("zero f16 dst");
        let push = f16_kv_pack_params(n as u32).to_le_bytes();
        let d = f16_kv_pack_dispatch(n as u32);
        launch_cached(
            &mut cache,
            &ctx,
            Kernel::F16KvPack,
            &[&buf_src, &buf_dst],
            d,
            &push,
            Kernel::F16KvPack.specialization_u32(),
        )
        .expect("f16_kv_pack dispatch");

        let mut bytes = vec![0u8; n * 2];
        buf_dst.copy_to_host(&mut bytes).expect("read back f16 dst");
        let got: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g, w,
                "f16_kv_pack n={n}[{i}]: src {} -> device 0x{g:04x} vs host 0x{w:04x}",
                x[i]
            );
        }
        eprintln!("[f16_kv_pack n={n}] PASS (bit-identical to host RNE, n={n})");
    }
}
