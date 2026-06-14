//! On-device full-attention sub-op correctness proofs (full-attention on-device,
//! Step 6).
//!
//! Oracle-gates the three device kernels the dense 27B full-attention block
//! needs — `rope_neox`, per-head `rms_norm`, and `flash_attn` — against the
//! exact host f32 references they replace in `infer-vulkan`'s `full_attention`
//! (`forward.rs`). Each kernel is launched with the precise push-constant block /
//! specialization / binding order the forward will use, and the device result is
//! compared element-wise to the host computation.
//!
//! These are the gates the AGENTS brief requires: a device op may not replace
//! its host counterpart until it matches the oracle within tolerance. Q/RoPE are
//! f32-in/f32-out (tight tolerance); flash_attn stores K/V as f16 and accumulates
//! the online softmax in f32, so its tolerance is looser (f16 K/V rounding).
//!
//! Runs only with `--features vulkan` + a working device; skips cleanly
//! otherwise.
#![cfg(feature = "vulkan")]

use vulkan_kernels::{
    FlashAttentionSpec, Kernel, KernelCache, KernelParams, flash_attn_dispatch, flash_attn_params,
    launch_cached, rms_norm_dispatch, rms_norm_params, rope_neox_dispatch, rope_neox_params,
    sigmoid_mul_dispatch, sigmoid_mul_params,
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

fn upload_i32<'a>(ctx: &'a VulkanContext, data: &[i32]) -> DeviceBuffer<'a> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut b = DeviceBuffer::alloc(ctx, bytes.len().max(4)).expect("alloc i32 buffer");
    b.copy_from_host(&bytes).expect("upload i32 buffer");
    b
}

fn upload_f16<'a>(ctx: &'a VulkanContext, data: &[f32]) -> DeviceBuffer<'a> {
    let bytes: Vec<u8> = data
        .iter()
        .flat_map(|&v| f32_to_f16_bits(v).to_le_bytes())
        .collect();
    let mut b = DeviceBuffer::alloc(ctx, bytes.len().max(4)).expect("alloc f16 buffer");
    b.copy_from_host(&bytes).expect("upload f16 buffer");
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

/// Round-to-nearest-even f32 -> f16 bit pattern (host reference for what the
/// device sees when it stores K/V as float16_t — so the oracle compares the
/// host SDPA computed over the SAME f16-rounded K/V).
fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 0xff {
        // inf / nan
        return sign | 0x7c00 | (if mant != 0 { 0x0200 } else { 0 }) as u16;
    }
    let mut e = exp - 127 + 15;
    if e >= 0x1f {
        // overflow -> inf
        return sign | 0x7c00;
    }
    if e <= 0 {
        // subnormal / underflow
        if e < -10 {
            return sign;
        }
        let mant_with_implicit = mant | 0x0080_0000;
        let shift = (14 - e) as u32;
        let mut m = mant_with_implicit >> shift;
        // round to nearest even
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

/// f16 bits -> f32 (so the host SDPA reference reads K/V at the SAME precision
/// the device flash_attn does, isolating algorithm error from f16 rounding).
fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // subnormal
            let mut e = -1i32;
            let mut m = mant;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            let fe = (e + 127 - 15) as u32;
            (sign << 31) | (fe << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        let fe = exp + 127 - 15;
        (sign << 31) | (fe << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

fn f16_round(v: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits(v))
}

fn assert_close(label: &str, got: &[f32], want: &[f32], abs_tol: f32, rel_tol: f32) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    let mut worst = 0f32;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let denom = w.abs().max(1e-4);
        let rel = (g - w).abs() / denom;
        worst = worst.max(rel.min((g - w).abs()));
        assert!(
            (g - w).abs() < abs_tol || rel < rel_tol,
            "{label}[{i}]: got {g} vs want {w} (abs {}, rel {rel})",
            (g - w).abs()
        );
    }
    eprintln!(
        "[{label}] PASS (worst min(abs,rel)={worst:.6}, n={})",
        got.len()
    );
}

// ── Host references (transcribed verbatim from forward.rs full_attention). ──

fn host_rope_neox(x: &[f32], pos: usize, rotary_dim: usize, theta: f32) -> Vec<f32> {
    let half = rotary_dim / 2;
    let mut out = x.to_vec();
    for d in 0..half {
        let inv_freq = theta.powf(-(2.0 * d as f32) / rotary_dim as f32);
        let angle = pos as f32 * inv_freq;
        let (s, c) = angle.sin_cos();
        let x0 = x[d];
        let x1 = x[d + half];
        out[d] = x0 * c - x1 * s;
        out[d + half] = x0 * s + x1 * c;
    }
    out
}

fn host_rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut sumsq = 0.0f32;
    for &v in x {
        sumsq += v * v;
    }
    let inv = 1.0 / (sumsq / n as f32 + eps).sqrt();
    (0..n).map(|i| x[i] * inv * w[i]).collect()
}

#[test]
fn rope_neox_matches_host_oracle() {
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping rope_neox oracle test");
            return;
        }
    };
    eprintln!("ARLE Vulkan rope_neox proof on: {}", ctx.device_name());
    let mut cache = KernelCache::new();
    let mut rng = Rng(0xA1B2_C3D4_E5F6_0718);

    let head_dim = 256usize;
    let rotary_dim = 256usize;
    let theta = 1.0e7f32;

    // Exercise several absolute positions (theta_base = pos * theta_scale^d).
    for &pos in &[0usize, 1, 7, 63, 511, 4095] {
        // A few independent head vectors batched as rows (nrows>1 path).
        let nrows = 3usize;
        let x: Vec<f32> = (0..nrows * head_dim).map(|_| rng.next_f32()).collect();

        let mut want = vec![0.0f32; nrows * head_dim];
        for r in 0..nrows {
            let rotated = host_rope_neox(
                &x[r * head_dim..r * head_dim + head_dim],
                pos,
                rotary_dim,
                theta,
            );
            want[r * head_dim..r * head_dim + head_dim].copy_from_slice(&rotated);
        }

        let buf_x = upload_f32(&ctx, &x);
        // pos buffer: ne02=1 so every row reads rope_data_pos[0]; size to nrows
        // for safety. int[].
        let buf_pos = upload_i32(&ctx, &vec![pos as i32; nrows]);
        let buf_ff = upload_f32(&ctx, &[0.0f32]); // has_ff=0: unread dummy
        let buf_d = upload_f32(&ctx, &vec![0.0f32; nrows * head_dim]);
        let buf_idx = upload_i32(&ctx, &[0, 0]); // set_rows_stride=0: unread dummy (uvec2)

        let push =
            rope_neox_params(head_dim as u32, rotary_dim as u32, nrows as u32, theta).to_le_bytes();
        let d = rope_neox_dispatch(rotary_dim as u32, nrows as u32);
        launch_cached(
            &mut cache,
            &ctx,
            Kernel::RopeNeox,
            &[&buf_x, &buf_pos, &buf_ff, &buf_d, &buf_idx],
            d,
            &push,
            Kernel::RopeNeox.specialization_u32(),
        )
        .expect("rope_neox dispatch");
        // Tolerance 5e-4: at large `pos` (e.g. 4095) the GPU and host `sin/cos`
        // differ by a few ULPs in the rotation angle, drifting the rotated pair
        // by ~2e-4. A wrong push/binding contract would be off by O(1), so this
        // still gates the contract tightly.
        assert_close(
            &format!("rope_neox pos={pos}"),
            &read_f32(&buf_d, nrows * head_dim),
            &want,
            5e-4,
            5e-4,
        );
    }
}

#[test]
fn per_head_rms_norm_matches_host_oracle() {
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping per-head rms_norm oracle test");
            return;
        }
    };
    eprintln!(
        "ARLE Vulkan per-head rms_norm proof on: {}",
        ctx.device_name()
    );
    let mut cache = KernelCache::new();
    let mut rng = Rng(0x0F1E_2D3C_4B5A_6978);

    // The full-attn q/k norm is a PLAIN per-head RMSNorm over head_dim=256 with
    // a head_dim-wide weight (the same weight broadcast across heads).
    let head_dim = 256usize;
    let eps = 1e-6f32;
    for _ in 0..4 {
        let x: Vec<f32> = (0..head_dim).map(|_| rng.next_f32()).collect();
        let w: Vec<f32> = (0..head_dim).map(|_| 0.8 + rng.next_f32() * 0.4).collect();
        let want = host_rms_norm(&x, &w, eps);

        let buf_a = upload_f32(&ctx, &x);
        let buf_b = upload_f32(&ctx, &w);
        let buf_d = upload_f32(&ctx, &vec![0.0f32; head_dim]);
        let push = rms_norm_params(head_dim as u32, eps).to_le_bytes();
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
        .expect("per-head rms_norm dispatch");
        assert_close(
            "per_head_rms_norm n=256",
            &read_f32(&buf_d, head_dim),
            &want,
            1e-4,
            1e-4,
        );
    }
}

/// Host causal single-query SDPA over a cached K/V (the exact `full_attention`
/// inner loop), reading K/V at f16 precision so the comparison isolates the
/// flash-attn algorithm from f16 storage rounding. One query head `q` [hd]
/// against `kv_len` cached K/V rows [hd], scale, softmax, weighted V sum.
fn host_sdpa_f16kv(q: &[f32], k: &[Vec<f32>], v: &[Vec<f32>], scale: f32) -> Vec<f32> {
    let hd = q.len();
    let kv_len = k.len();
    let mut scores = vec![0.0f32; kv_len];
    let mut max_s = f32::NEG_INFINITY;
    for (t, score) in scores.iter_mut().enumerate() {
        let mut dot = 0.0f32;
        for d in 0..hd {
            dot += q[d] * f16_round(k[t][d]);
        }
        let s = dot * scale;
        *score = s;
        if s > max_s {
            max_s = s;
        }
    }
    let mut denom = 0.0f32;
    for s in &mut scores {
        *s = (*s - max_s).exp();
        denom += *s;
    }
    let inv = 1.0 / denom;
    let mut out = vec![0.0f32; hd];
    for (t, &sw) in scores.iter().enumerate() {
        let w = sw * inv;
        for d in 0..hd {
            out[d] += w * f16_round(v[t][d]);
        }
    }
    out
}

#[test]
fn sigmoid_mul_matches_host_oracle() {
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping sigmoid_mul oracle test");
            return;
        }
    };
    eprintln!("ARLE Vulkan sigmoid_mul proof on: {}", ctx.device_name());
    let mut cache = KernelCache::new();
    let mut rng = Rng(0x5151_5151_A0A0_A0A0);

    // The full-attn gate: out[i] = sigmoid(gate[i]) * attn[i], over the per-head
    // attention output (q_dim wide; test head_dim and the full 24*256 q_dim).
    for &n in &[256usize, 6144] {
        let gate: Vec<f32> = (0..n).map(|_| rng.next_f32() * 4.0).collect();
        let val: Vec<f32> = (0..n).map(|_| rng.next_f32() * 4.0).collect();
        let want: Vec<f32> = (0..n)
            .map(|i| (1.0 / (1.0 + (-gate[i]).exp())) * val[i])
            .collect();

        // separate dst
        let buf_a = upload_f32(&ctx, &gate);
        let buf_b = upload_f32(&ctx, &val);
        let buf_d = upload_f32(&ctx, &vec![0.0f32; n]);
        let push = sigmoid_mul_params(n as u32).to_le_bytes();
        let d = sigmoid_mul_dispatch(n as u32);
        launch_cached(
            &mut cache,
            &ctx,
            Kernel::SigmoidMul,
            &[&buf_a, &buf_b, &buf_d],
            d,
            &push,
            Kernel::SigmoidMul.specialization_u32(),
        )
        .expect("sigmoid_mul dispatch");
        assert_close(
            &format!("sigmoid_mul n={n}"),
            &read_f32(&buf_d, n),
            &want,
            1e-4,
            1e-4,
        );

        // in-place (out aliases val — the forward gates attn in place)
        let buf_g2 = upload_f32(&ctx, &gate);
        let buf_v2 = upload_f32(&ctx, &val);
        launch_cached(
            &mut cache,
            &ctx,
            Kernel::SigmoidMul,
            &[&buf_g2, &buf_v2, &buf_v2],
            d,
            &push,
            Kernel::SigmoidMul.specialization_u32(),
        )
        .expect("sigmoid_mul in-place dispatch");
        assert_close(
            &format!("sigmoid_mul in-place n={n}"),
            &read_f32(&buf_v2, n),
            &want,
            1e-4,
            1e-4,
        );
    }
}

#[test]
fn flash_attn_matches_host_sdpa_oracle() {
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping flash_attn oracle test");
            return;
        }
    };
    eprintln!("ARLE Vulkan flash_attn proof on: {}", ctx.device_name());
    let (sg, sg_min, sg_max) = ctx.subgroup_size();
    eprintln!("  subgroup_size={sg} (control min={sg_min}, max={sg_max})");
    let mut cache = KernelCache::new();
    let mut rng = Rng(0xDEAD_BEEF_1234_5678);

    let hd = 256usize; // head_dim = HSK = HSV
    let scale = 1.0f32 / (hd as f32).sqrt();

    // Single query head (N=1), one KV head (gqa_ratio=1 for the oracle — the
    // forward will dispatch per query head). Test several cache lengths.
    for &kv_len in &[1usize, 2, 8, 33, 64, 65, 200] {
        let q: Vec<f32> = (0..hd).map(|_| rng.next_f32()).collect();
        let k: Vec<Vec<f32>> = (0..kv_len)
            .map(|_| (0..hd).map(|_| rng.next_f32()).collect())
            .collect();
        let v: Vec<Vec<f32>> = (0..kv_len)
            .map(|_| (0..hd).map(|_| rng.next_f32()).collect())
            .collect();

        let want = host_sdpa_f16kv(&q, &k, &v, scale);

        // Device buffers. Q is f32 [hd]; K/V are f16 [kv_len, hd] row-major.
        let buf_q = upload_f32(&ctx, &q);
        let k_flat: Vec<f32> = k.iter().flatten().copied().collect();
        let v_flat: Vec<f32> = v.iter().flatten().copied().collect();
        let buf_k = upload_f16(&ctx, &k_flat);
        let buf_v = upload_f16(&ctx, &v_flat);
        // mask (binding 3, f16) — unused (Flags has no MASK bit), dummy.
        let buf_mask = upload_f16(&ctx, &[0.0f32]);
        // sinks (binding 4, f32) — unused, dummy.
        let buf_sinks = upload_f32(&ctx, &[0.0f32]);
        let buf_out = upload_f32(&ctx, &vec![0.0f32; hd]);
        // mask_opt (binding 6, uint) — unused, dummy.
        let buf_mask_opt = upload_i32(&ctx, &[0]);

        let spec = FlashAttentionSpec::f32_f16(hd as u32);
        let push = flash_attn_params(hd as u32, hd as u32, kv_len as u32, scale).to_le_bytes();
        let d = flash_attn_dispatch();
        launch_cached(
            &mut cache,
            &ctx,
            Kernel::FlashAttn,
            &[
                &buf_q,
                &buf_k,
                &buf_v,
                &buf_mask,
                &buf_sinks,
                &buf_out,
                &buf_mask_opt,
            ],
            d,
            &push,
            spec.specialization_u32(),
        )
        .expect("flash_attn dispatch");

        // f16 K/V + f32 online-softmax accumulation: loosen to f16-scale tol.
        assert_close(
            &format!("flash_attn kv_len={kv_len}"),
            &read_f32(&buf_out, hd),
            &want,
            2e-3,
            5e-3,
        );
    }
    let _ = KernelParams::empty();
}
