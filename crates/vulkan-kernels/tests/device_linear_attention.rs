//! On-device gated-delta linear-attention sub-op correctness proofs (linear-
//! attention on-device port).
//!
//! Oracle-gates the two model-specific serial shaders the dense 27B / 35B-MoE
//! linear-attention block needs — `qwen35_ssm_conv` (depthwise causal conv1d +
//! SiLU + ring advance) and `qwen35_gated_delta_net` (the recurrent gated-delta
//! state update) — against the EXACT host f32 routines they replace in
//! `infer-vulkan`'s `linear_attention` (`forward.rs`). Each kernel is launched
//! with the precise push-constant block / binding order the forward uses, and
//! the device result is compared element-wise to a host reference transcribed
//! line-for-line from `linear_attention`.
//!
//! These are the gates the AGENTS brief requires: a device op may not replace
//! its host counterpart until it matches the oracle within tolerance. The conv
//! pre-activation is bf16-rounded on BOTH sides (host `round_to_bf16` ==
//! shader `round_to_bf16`), so the conv tolerance is tight; the gated-delta
//! recurrence is pure f32 in/out, also tight.
//!
//! Runs only with `--features vulkan` + a working device; skips cleanly
//! otherwise.
#![cfg(feature = "vulkan")]

use vulkan_kernels::{
    Kernel, KernelCache, launch_cached, qwen35_gated_delta_net_dispatch,
    qwen35_gated_delta_net_params, qwen35_ssm_conv_dispatch, qwen35_ssm_conv_params,
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

fn assert_close(label: &str, got: &[f32], want: &[f32], abs_tol: f32, rel_tol: f32) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    let mut worst = 0f32;
    let mut worst_i = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let denom = w.abs().max(1e-4);
        let rel = (g - w).abs() / denom;
        if rel > worst {
            worst = rel;
            worst_i = i;
        }
        assert!(
            (g - w).abs() < abs_tol || rel < rel_tol,
            "{label}[{i}]: got {g} vs want {w} (abs {} rel {rel})",
            (g - w).abs()
        );
    }
    eprintln!(
        "[{label}] PASS (worst rel_err={worst:.3e} @ {worst_i}, n={})",
        got.len()
    );
}

// ── Host references transcribed line-for-line from forward.rs::linear_attention.

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}
fn round_to_bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    f32::from_bits((bits.wrapping_add(0x7fff).wrapping_add(lsb)) & 0xffff_0000)
}

/// Host depthwise causal conv1d + SiLU over a full sequence, advancing the ring
/// in place — the multi-token generalization of the forward's single-token conv
/// (and the exact semantics of `qwen35_ssm_conv.comp`). `x_seq` is row-major
/// `[seq_len*num_channels]`; `ring` is `[num_channels*(kernel-1)]`.
fn host_conv1d(
    x_seq: &[f32],
    conv_w: &[f32],
    ring: &mut [f32],
    out: &mut [f32],
    num_channels: usize,
    seq_len: usize,
    kernel: usize,
) {
    let state_width = kernel - 1;
    for c in 0..num_channels {
        // Snapshot the old ring (reads precede the in-place overwrite, matching
        // the shader's `old_state` register staging).
        let mut old = [0.0f32; 8];
        for i in 0..state_width.min(8) {
            old[i] = ring[c * state_width + i];
        }
        for t in 0..seq_len {
            let mut sum = 0.0f32;
            for k in 0..kernel {
                let src_t = t as isize - state_width as isize + k as isize;
                let value = if src_t < 0 {
                    let si = state_width as isize + src_t;
                    if (0..8).contains(&si) {
                        old[si as usize]
                    } else {
                        0.0
                    }
                } else {
                    x_seq[src_t as usize * num_channels + c]
                };
                sum += value * conv_w[c * kernel + k];
            }
            out[t * num_channels + c] = silu(round_to_bf16(sum));
        }
        // Ring advance: the last `state_width` inputs of the sequence.
        for i in 0..state_width {
            let src_t = seq_len as isize - state_width as isize + i as isize;
            ring[c * state_width + i] = if src_t >= 0 {
                x_seq[src_t as usize * num_channels + c]
            } else {
                let si = state_width as isize + src_t;
                if (0..8).contains(&si) {
                    old[si as usize]
                } else {
                    0.0
                }
            };
        }
    }
}

/// Host recurrent gated-delta over a sequence (matches the forward's per-token
/// loop and `qwen35_gated_delta_net.comp`). `qkv` is post-conv `[q|k|v]`
/// token-major; `gdr` is `[nv*kd*vd]` (val contiguous), read+written.
// The j/val range loops index sibling state/key arrays by the same variable, one
// to one with the kernel's index math — clearer than zipped iterators here.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn host_gated_delta(
    qkv: &[f32],
    b_proj: &[f32],
    a_proj: &[f32],
    dt_bias: &[f32],
    a_log: &[f32],
    gdr: &mut [f32],
    out: &mut [f32],
    nk: usize,
    nv: usize,
    kd: usize,
    vd: usize,
    seq_len: usize,
) {
    let q_dim_total = nk * kd;
    let k_dim_total = nk * kd;
    let v_dim_total = nv * vd;
    let qkv_stride = q_dim_total + k_dim_total + v_dim_total;
    let out_stride = nv * vd;
    for token in 0..seq_len {
        let tb = token * qkv_stride;
        for vh in 0..nv {
            let kh = vh % nk; // TILE, not block-broadcast.
            let q_head = &qkv[tb + kh * kd..tb + kh * kd + kd];
            let k_head = &qkv[tb + q_dim_total + kh * kd..tb + q_dim_total + kh * kd + kd];
            let v_head = &qkv[tb + q_dim_total + k_dim_total + vh * vd
                ..tb + q_dim_total + k_dim_total + vh * vd + vd];

            let mut q_sumsq = 0.0f32;
            let mut k_sumsq = 0.0f32;
            for j in 0..kd {
                q_sumsq += q_head[j] * q_head[j];
                k_sumsq += k_head[j] * k_head[j];
            }
            let q_norm = 1.0 / (q_sumsq + 1.0e-6).sqrt();
            let k_norm = 1.0 / (k_sumsq + 1.0e-6).sqrt();
            let q_scale = q_norm * (1.0 / (kd as f32).sqrt());

            let a_val = a_proj[token * nv + vh];
            let b_val = b_proj[token * nv + vh];
            let exp_g = (a_log[vh] * softplus(a_val + dt_bias[vh])).exp();
            let beta = sigmoid(b_val);

            let base = vh * kd * vd;
            let mut kv_mem = vec![0.0f32; vd];
            for j in 0..kd {
                let kj = k_head[j] * k_norm;
                for val in 0..vd {
                    let idx = base + j * vd + val;
                    let decayed = gdr[idx] * exp_g;
                    gdr[idx] = decayed;
                    kv_mem[val] += decayed * kj;
                }
            }
            let o = &mut out[token * out_stride + vh * vd..token * out_stride + vh * vd + vd];
            for val in 0..vd {
                let delta = (v_head[val] - kv_mem[val]) * beta;
                let mut acc = 0.0f32;
                for j in 0..kd {
                    let idx = base + j * vd + val;
                    let kj = k_head[j] * k_norm;
                    let updated = gdr[idx] + delta * kj;
                    gdr[idx] = updated;
                    acc += updated * (q_head[j] * q_scale);
                }
                o[val] = acc;
            }
        }
    }
}

#[test]
fn qwen35_ssm_conv_matches_host_oracle() {
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping ssm_conv oracle test");
            return;
        }
    };
    eprintln!("ARLE Vulkan ssm_conv proof on: {}", ctx.device_name());
    let mut cache = KernelCache::new();
    let mut rng = Rng(0xC0FF_EE00_1234_5678);

    let kernel = 4usize; // Qwen3.5 conv kernel dim.
    // A small channel count (exercises the ring) and the real 27B qkv width
    // (16*128*2 + 48*128 = 10240) to exercise the full per-channel loop.
    for &num_channels in &[7usize, 10240] {
        // Single-token decode (what the forward records each step) and a short
        // prefill (seq_len > 1) to exercise the multi-tap + ring advance.
        for &seq_len in &[1usize, 5] {
            let state_width = kernel - 1;
            let x_seq: Vec<f32> = (0..seq_len * num_channels)
                .map(|_| rng.next_f32())
                .collect();
            let conv_w: Vec<f32> = (0..num_channels * kernel).map(|_| rng.next_f32()).collect();
            // A non-zero starting ring so the causal taps actually read state.
            let ring0: Vec<f32> = (0..num_channels * state_width)
                .map(|_| rng.next_f32())
                .collect();

            let mut host_ring = ring0.clone();
            let mut host_out = vec![0.0f32; seq_len * num_channels];
            host_conv1d(
                &x_seq,
                &conv_w,
                &mut host_ring,
                &mut host_out,
                num_channels,
                seq_len,
                kernel,
            );

            let buf_x = upload_f32(&ctx, &x_seq);
            let buf_w = upload_f32(&ctx, &conv_w);
            let mut buf_state = upload_f32(&ctx, &ring0);
            let buf_out = upload_f32(&ctx, &vec![0.0f32; seq_len * num_channels]);
            let push = qwen35_ssm_conv_params(num_channels as u32, seq_len as u32, kernel as u32)
                .to_le_bytes();
            let d = qwen35_ssm_conv_dispatch(num_channels as u32);
            launch_cached(
                &mut cache,
                &ctx,
                Kernel::Qwen35SsmConv,
                &[&buf_x, &buf_w, &mut buf_state, &buf_out],
                d,
                &push,
                Kernel::Qwen35SsmConv.specialization_u32(),
            )
            .expect("ssm_conv dispatch");

            let label = format!("ssm_conv nc={num_channels} seq={seq_len}");
            assert_close(
                &format!("{label} out"),
                &read_f32(&buf_out, seq_len * num_channels),
                &host_out,
                1e-5,
                1e-5,
            );
            assert_close(
                &format!("{label} ring"),
                &read_f32(&buf_state, num_channels * state_width),
                &host_ring,
                1e-6,
                1e-6,
            );
        }
    }
}

#[test]
fn qwen35_gated_delta_net_matches_host_oracle() {
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping gated_delta oracle test");
            return;
        }
    };
    eprintln!("ARLE Vulkan gated_delta proof on: {}", ctx.device_name());
    let mut cache = KernelCache::new();
    let mut rng = Rng(0xDEAD_BEEF_0BAD_F00D);

    // Real Qwen3.5 linear-attention dims: 16 key heads, 48 value heads (GQA tile
    // factor 3), key/val head dim 128.
    let nk = 16usize;
    let nv = 48usize;
    let kd = 128usize;
    let vd = 128usize;
    let q_dim_total = nk * kd;
    let k_dim_total = nk * kd;
    let v_dim_total = nv * vd;
    let qkv_stride = q_dim_total + k_dim_total + v_dim_total;

    // Two tokens against a NON-ZERO starting state so the decay + carried memory
    // matter (a single token from a zero state would never exercise the decay).
    for &seq_len in &[1usize, 2] {
        let qkv: Vec<f32> = (0..seq_len * qkv_stride)
            .map(|_| rng.next_f32() * 0.5)
            .collect();
        let b_proj: Vec<f32> = (0..seq_len * nv).map(|_| rng.next_f32()).collect();
        let a_proj: Vec<f32> = (0..seq_len * nv).map(|_| rng.next_f32()).collect();
        let dt_bias: Vec<f32> = (0..nv).map(|_| rng.next_f32()).collect();
        // ssm_a = -exp(A_log): strictly negative, like the GGUF tensor on-box.
        let a_log: Vec<f32> = (0..nv).map(|_| -(rng.next_f32().abs() + 0.1)).collect();
        let state0: Vec<f32> = (0..nv * kd * vd).map(|_| rng.next_f32() * 0.1).collect();

        let mut host_state = state0.clone();
        let mut host_out = vec![0.0f32; seq_len * v_dim_total];
        host_gated_delta(
            &qkv,
            &b_proj,
            &a_proj,
            &dt_bias,
            &a_log,
            &mut host_state,
            &mut host_out,
            nk,
            nv,
            kd,
            vd,
            seq_len,
        );

        let buf_qkv = upload_f32(&ctx, &qkv);
        let buf_b = upload_f32(&ctx, &b_proj);
        let buf_a = upload_f32(&ctx, &a_proj);
        let buf_dt = upload_f32(&ctx, &dt_bias);
        let buf_alog = upload_f32(&ctx, &a_log);
        let mut buf_state = upload_f32(&ctx, &state0);
        let buf_out = upload_f32(&ctx, &vec![0.0f32; seq_len * v_dim_total]);
        let push = qwen35_gated_delta_net_params(
            nk as u32,
            nv as u32,
            kd as u32,
            vd as u32,
            seq_len as u32,
        )
        .to_le_bytes();
        let d = qwen35_gated_delta_net_dispatch(nv as u32);
        launch_cached(
            &mut cache,
            &ctx,
            Kernel::Qwen35GatedDeltaNet,
            &[
                &buf_qkv,
                &buf_b,
                &buf_a,
                &buf_dt,
                &buf_alog,
                &mut buf_state,
                &buf_out,
            ],
            d,
            &push,
            Kernel::Qwen35GatedDeltaNet.specialization_u32(),
        )
        .expect("gated_delta dispatch");

        let label = format!("gated_delta seq={seq_len}");
        assert_close(
            &format!("{label} out"),
            &read_f32(&buf_out, seq_len * v_dim_total),
            &host_out,
            1e-4,
            1e-4,
        );
        assert_close(
            &format!("{label} state"),
            &read_f32(&buf_state, nv * kd * vd),
            &host_state,
            1e-4,
            1e-4,
        );
    }
}
