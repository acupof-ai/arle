//! Numeric single-token forward for the **dense Qwen3.5/3.6 27B** (`arch =
//! qwen35`, dense FFN) on the Vulkan device.
//!
//! Design (correctness first; see the AGENTS brief): the heavy matmuls — every
//! projection, the dense FFN, and the LM head — run **on the GPU** through the
//! PROVEN quantized GEMV (`q8_0_gemv_with_params` fed by `q8_1_quantize`,
//! validated in `vulkan-kernels/tests/device_gemv.rs`). The lighter
//! per-element / reduction ops (RMSNorm, RoPE, attention softmax, the per-head
//! sigmoid gate, SwiGLU, depthwise conv1d, the gated-delta recurrence, residual
//! add) run **on the host in f32**, transcribed line-for-line from the
//! authoritative CUDA reference (`crates/infer-cuda/src/qwen35.rs` +
//! `crates/cuda-kernels/csrc/{attention/prefill_attention_hd256.cu,
//! misc/gated_delta_rule.cu, misc/conv1d.cu, misc/norm.cu}`). f32 host math is
//! strictly more accurate than the bf16 device path and is trivially finite, so
//! this lane delivers FINITE, sane logits end-to-end while we (separately) bring
//! up the elementwise device kernels' push-constant contracts.
//!
//! Numeric contract distilled from the reference (all verified against the real
//! 27B GGUF dims):
//!   hidden=5120, layers=64 (LLL F interleave from `config.layer_types`),
//!   heads=24, kv_heads=4, head_dim=256, rope_theta=1e7, rotary_dim=256,
//!   rms_eps=1e-6, dense FFN intermediate=17408.
//!   FULL layer: attn_q `[5120,12288]` (out=24*256*2 = [query|gate] per head),
//!   attn_k/v `[5120,1024]`, attn_q_norm/attn_k_norm `[256]` f32 (USED WITH THE
//!   `(1+w)` OFFSET), attn_output `[6144,5120]`. Per-head q/k RMSNorm → NeoX
//!   RoPE → causal SDPA (scale 1/sqrt(head_dim)) → per-head `*sigmoid(gate)` →
//!   o_proj.
//!   LINEAR layer: attn_qkv `[5120,10240]` ([q=2048|k=2048|v=6144]), attn_gate
//!   (z) `[5120,6144]`, ssm_alpha (a) / ssm_beta (b) `[5120,48]`, ssm_a (A_log)
//!   `[48]` f32, ssm_dt.bias `[48]` f32, ssm_conv1d `[4,10240]` f32
//!   (per-channel depthwise, SiLU), ssm_norm `[128]` f32 (gated RMSNorm, PLAIN
//!   weight, ×silu(z)), ssm_out `[6144,5120]`.
//!   All three block RMSNorms (input / post-attention / final) use the `(1+w)`
//!   offset; the gated output RMSNorm uses the plain weight.
//!
//! State: this lane runs the **uncached full-prefix** path for a single slot —
//! a forward of one token at `start_pos` recomputes nothing it does not own.
//! Per-slot KV cache (full layers) and recurrent + conv state (linear layers)
//! are owned by [`crate::model_qwen35::VulkanQwen35Model`] and advanced in place
//! here, matching the reference's owned-state contract.

use anyhow::{Context, Result, anyhow, bail};

use qwen35_spec::{LayerType, Qwen35Config};
use vulkan_kernels::{
    BLOCK_Q8_1_BYTES, Dispatch, KernelParams, Q8_1_X4_VALUES_PER_GROUP, gemv_dispatch, gemv_params,
    q8_0_gemv_with_params, q8_1_quantize, q8_1_quantize_dispatch, q8_1_quantize_params,
};
use vulkan_sys::{DeviceBuffer, VulkanContext};

use crate::loader::Residency;
use crate::loader::upload::{DeviceTensor, ResidentWeights};

/// Per-slot recurrent / cache state carried across forward calls for one
/// sequence. Sized from the config's local (single-GPU) widths.
pub struct Qwen35ForwardState {
    /// One growing K cache per FULL-attention layer, row-major `[pos, kv_dim]`
    /// f32 (kv_dim = kv_heads*head_dim). RoPE already applied at write time.
    pub k_cache: Vec<Vec<f32>>,
    /// One growing V cache per FULL-attention layer, `[pos, kv_dim]` f32.
    pub v_cache: Vec<Vec<f32>>,
    /// Gated-delta recurrent state per LINEAR layer, `[v_head, key_dim,
    /// val_dim]` f32 (val contiguous), matching `gated_delta_rule.cu`.
    pub gdr_state: Vec<Vec<f32>>,
    /// Depthwise conv ring per LINEAR layer, `[channel, kernel-1]` f32 —
    /// the last `kernel-1` inputs per channel (oldest first).
    pub conv_ring: Vec<Vec<f32>>,
    /// Materialized sequence length (tokens already in the caches / consumed by
    /// the recurrence). Must equal the forward's `start_pos`.
    pub seq_len: usize,
}

impl Qwen35ForwardState {
    pub fn new(config: &Qwen35Config) -> Self {
        let num_full = config
            .layer_types
            .iter()
            .filter(|&&t| t == LayerType::FullAttention)
            .count();
        let num_linear = config.layer_types.len() - num_full;
        let qkv_dim = 2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim;
        let gdr_len = config.linear_num_value_heads
            * config.linear_key_head_dim
            * config.linear_value_head_dim;
        let conv_len = qkv_dim * config.linear_conv_kernel_dim.saturating_sub(1);
        Self {
            k_cache: vec![Vec::new(); num_full],
            v_cache: vec![Vec::new(); num_full],
            gdr_state: vec![vec![0.0f32; gdr_len]; num_linear],
            conv_ring: vec![vec![0.0f32; conv_len]; num_linear],
            seq_len: 0,
        }
    }

    /// Reset for a fresh generation (zeros recurrent + conv state, clears caches).
    pub fn reset(&mut self) {
        for c in &mut self.k_cache {
            c.clear();
        }
        for c in &mut self.v_cache {
            c.clear();
        }
        for s in &mut self.gdr_state {
            s.iter_mut().for_each(|v| *v = 0.0);
        }
        for r in &mut self.conv_ring {
            r.iter_mut().for_each(|v| *v = 0.0);
        }
        self.seq_len = 0;
    }
}

/// Reusable device scratch for the on-device GEMV path. One activation-quantize
/// buffer + one f32 destination buffer, both grown to the largest shape seen.
struct GemvScratch<'a> {
    /// q8_1_x4 quantized activations (input vector). Sized for the widest GEMV
    /// input (FFN down has 17408 cols).
    quant: DeviceBuffer<'a>,
    quant_cap_cols: usize,
    /// f32 destination rows. Sized for the widest GEMV output (lm_head vocab).
    dst: DeviceBuffer<'a>,
    dst_cap_rows: usize,
}

impl<'a> GemvScratch<'a> {
    fn new(ctx: &'a VulkanContext, max_cols: usize, max_rows: usize) -> Result<Self> {
        let quant_bytes = q8_1_x4_bytes(max_cols);
        let quant = DeviceBuffer::alloc(ctx, quant_bytes)
            .map_err(|e| anyhow!("alloc gemv quant scratch ({quant_bytes} B): {e}"))?;
        let dst_bytes = max_rows * std::mem::size_of::<f32>();
        let dst = DeviceBuffer::alloc(ctx, dst_bytes)
            .map_err(|e| anyhow!("alloc gemv dst scratch ({dst_bytes} B): {e}"))?;
        Ok(Self {
            quant,
            quant_cap_cols: max_cols,
            dst,
            dst_cap_rows: max_rows,
        })
    }
}

/// Bytes for the q8_1_x4 quantized form of an `ncols`-element activation vector
/// (the shader groups 128 values into one x4 super-block of 4×36 B).
fn q8_1_x4_bytes(ncols: usize) -> usize {
    let num_x4 = ncols.div_ceil(Q8_1_X4_VALUES_PER_GROUP as usize);
    num_x4 * 4 * BLOCK_Q8_1_BYTES
}

/// The on-device numeric forward for one token. Returns logits `[vocab]` f32.
///
/// `state.seq_len` must equal `start_pos` (the uncached full-prefix contract).
/// Mutates the per-slot caches / recurrent state in place and advances
/// `state.seq_len`.
pub fn forward_token(
    ctx: &VulkanContext,
    config: &Qwen35Config,
    weights: &ResidentWeights<'_>,
    state: &mut Qwen35ForwardState,
    token: u32,
    start_pos: usize,
) -> Result<Vec<f32>> {
    if start_pos != state.seq_len {
        bail!(
            "forward_token start_pos {start_pos} != materialized seq_len {} \
             (this lane runs the uncached full-prefix path: feed tokens in order)",
            state.seq_len
        );
    }
    let h = config.hidden_size;
    let eps = config.rms_norm_eps;

    // Widest GEMV shapes for scratch sizing: input cols max = FFN intermediate
    // (down proj has 17408 input cols); output rows max = vocab (lm_head).
    let max_cols = config.intermediate_size.max(h).max(config.vocab_size);
    let max_rows = config
        .vocab_size
        .max(config.intermediate_size)
        .max(2 * config.num_attention_heads * config.head_dim);
    let mut scratch = GemvScratch::new(ctx, max_cols, max_rows)?;

    // ── Token embedding (host gather + dequant of one Q8_0 row). ──
    let mut hidden = weights
        .embedding
        .embed_row(token)
        .with_context(|| format!("embed token {token}"))?;
    if hidden.len() != h {
        bail!("embedding row width {} != hidden {h}", hidden.len());
    }

    let mut full_idx = 0usize;
    let mut linear_idx = 0usize;
    for (layer, &layer_type) in config.layer_types.iter().enumerate() {
        // input_layernorm: (1+w) offset RMSNorm.
        let attn_norm = norm_weight(weights, layer, "attn_norm")?;
        let normed = rms_norm_offset(&hidden, &attn_norm, eps);

        let attn_out = match layer_type {
            LayerType::FullAttention => {
                let out = full_attention(
                    ctx,
                    config,
                    weights,
                    state,
                    &mut scratch,
                    layer,
                    full_idx,
                    &normed,
                    start_pos,
                )?;
                full_idx += 1;
                out
            }
            LayerType::LinearAttention => {
                let out = linear_attention(
                    ctx,
                    config,
                    weights,
                    state,
                    &mut scratch,
                    layer,
                    linear_idx,
                    &normed,
                )?;
                linear_idx += 1;
                out
            }
        };

        // Post-attention residual add + post_attention_layernorm ((1+w) offset).
        let post_sum = add_vec(&hidden, &attn_out);
        let post_norm_w = norm_weight(weights, layer, "post_attention_norm")?;
        let mlp_in = rms_norm_offset(&post_sum, &post_norm_w, eps);

        // Dense FFN: down( silu(gate(x)) * up(x) ).
        let gate = gemv_layer(ctx, weights, &mut scratch, layer, "ffn_gate", &mlp_in)?;
        let up = gemv_layer(ctx, weights, &mut scratch, layer, "ffn_up", &mlp_in)?;
        let act = swiglu(&gate, &up);
        let mlp_out = gemv_layer(ctx, weights, &mut scratch, layer, "ffn_down", &act)?;

        // MLP residual add into the next layer's residual stream.
        hidden = add_vec(&post_sum, &mlp_out);
    }

    // Final norm ((1+w) offset) + LM head GEMV → logits.
    let final_norm = weights
        .get("output_norm.weight")
        .ok_or_else(|| anyhow!("missing output_norm.weight"))?;
    let final_norm_w = dequant_f32(final_norm, h)?;
    let normed = rms_norm_offset(&hidden, &final_norm_w, eps);
    let logits = gemv_global(
        ctx,
        weights,
        &mut scratch,
        "output.weight",
        &normed,
        config.vocab_size,
    )?;

    state.seq_len += 1;
    Ok(logits)
}

// ─────────────────────────────────────────────────────────────────────────────
// Full-attention layer (gated q_proj, per-head q/k norm + NeoX RoPE, causal SDPA)
// ─────────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn full_attention(
    ctx: &VulkanContext,
    config: &Qwen35Config,
    weights: &ResidentWeights<'_>,
    state: &mut Qwen35ForwardState,
    scratch: &mut GemvScratch<'_>,
    layer: usize,
    full_idx: usize,
    normed: &[f32],
    start_pos: usize,
) -> Result<Vec<f32>> {
    let hd = config.head_dim;
    let nq = config.num_attention_heads;
    let nkv = config.num_key_value_heads;
    let q_dim = nq * hd;
    let kv_dim = nkv * hd;
    let half = config.rotary_dim / 2;
    let theta = config.rope_theta;
    let eps = config.rms_norm_eps;
    let pos = start_pos;

    // q_proj → [query|gate] per head (out = 2*nq*hd); k/v_proj → [nkv*hd].
    let q_full = gemv_layer(ctx, weights, scratch, layer, "attn_q", normed)?; // 2*q_dim
    let k_in = gemv_layer(ctx, weights, scratch, layer, "attn_k", normed)?; // kv_dim
    let v_in = gemv_layer(ctx, weights, scratch, layer, "attn_v", normed)?; // kv_dim

    let q_norm = norm_weight(weights, layer, "attn_q_norm")?; // [hd]
    let k_norm = norm_weight(weights, layer, "attn_k_norm")?; // [hd]

    // Build this token's query heads (q/k norm + RoPE) and the K/V rows.
    // Query layout in q_full per head h: [h*2*hd .. h*2*hd+hd] = query,
    // [h*2*hd+hd .. h*2*hd+2*hd] = gate (consumed after attention).
    let mut q = vec![0.0f32; q_dim];
    for hh in 0..nq {
        let src = &q_full[hh * 2 * hd..hh * 2 * hd + hd];
        let normed_head = rms_norm_offset(src, &q_norm, eps); // (1+w) per-head
        let rotated = rope_neox(&normed_head, pos, half, hd, theta);
        q[hh * hd..hh * hd + hd].copy_from_slice(&rotated);
    }
    let mut k_row = vec![0.0f32; kv_dim];
    for hh in 0..nkv {
        let src = &k_in[hh * hd..hh * hd + hd];
        let normed_head = rms_norm_offset(src, &k_norm, eps);
        let rotated = rope_neox(&normed_head, pos, half, hd, theta);
        k_row[hh * hd..hh * hd + hd].copy_from_slice(&rotated);
    }

    // Append K/V to the per-slot cache (RoPE already applied to K).
    state.k_cache[full_idx].extend_from_slice(&k_row);
    state.v_cache[full_idx].extend_from_slice(&v_in);
    let kv_len = pos + 1;

    // Causal scaled-dot-product attention, GQA (group = nq/nkv query heads per
    // kv head). scale = 1/sqrt(head_dim).
    let scale = 1.0f32 / (hd as f32).sqrt();
    let group = nq / nkv;
    let kc = &state.k_cache[full_idx];
    let vc = &state.v_cache[full_idx];
    let mut attn = vec![0.0f32; q_dim];
    for hh in 0..nq {
        let kv_h = hh / group;
        let qh = &q[hh * hd..hh * hd + hd];
        // scores over all cached positions, softmax, weighted sum of V.
        let mut scores = vec![0.0f32; kv_len];
        let mut max_s = f32::NEG_INFINITY;
        for (t, score) in scores.iter_mut().enumerate() {
            let krow = &kc[t * kv_dim + kv_h * hd..t * kv_dim + kv_h * hd + hd];
            let mut dot = 0.0f32;
            for d in 0..hd {
                dot += qh[d] * krow[d];
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
        let out = &mut attn[hh * hd..hh * hd + hd];
        for (t, &sw) in scores.iter().enumerate() {
            let w = sw * inv;
            let vrow = &vc[t * kv_dim + kv_h * hd..t * kv_dim + kv_h * hd + hd];
            for d in 0..hd {
                out[d] += w * vrow[d];
            }
        }
    }

    // Per-head sigmoid gate from q_full's gate half: attn[h,d] *= sigmoid(gate).
    for hh in 0..nq {
        let gate = &q_full[hh * 2 * hd + hd..hh * 2 * hd + 2 * hd];
        let out = &mut attn[hh * hd..hh * hd + hd];
        for d in 0..hd {
            out[d] *= sigmoid(gate[d]);
        }
    }

    // o_proj: [q_dim -> hidden].
    gemv_layer(ctx, weights, scratch, layer, "attn_output", &attn)
}

// ─────────────────────────────────────────────────────────────────────────────
// Linear (gated-delta) layer: in-proj → conv1d(SiLU) → recurrent delta → gated
// RMSNorm → out-proj. Transcribed from gated_delta_rule.cu / conv1d.cu.
// ─────────────────────────────────────────────────────────────────────────────

// Several inner loops index sibling arrays by the same loop variable (the
// depthwise conv taps, the gated-delta state passes) — a range loop mirrors the
// CUDA reference's index math one-to-one and is clearer than a zipped iterator.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn linear_attention(
    ctx: &VulkanContext,
    config: &Qwen35Config,
    weights: &ResidentWeights<'_>,
    state: &mut Qwen35ForwardState,
    scratch: &mut GemvScratch<'_>,
    layer: usize,
    linear_idx: usize,
    normed: &[f32],
) -> Result<Vec<f32>> {
    let kd = config.linear_key_head_dim; // 128
    let vd = config.linear_value_head_dim; // 128
    let nk = config.linear_num_key_heads; // 16
    let nv = config.linear_num_value_heads; // 48
    let kernel = config.linear_conv_kernel_dim; // 4
    let q_dim_total = nk * kd; // 2048
    let k_dim_total = nk * kd; // 2048
    let v_dim_total = nv * vd; // 6144
    let qkv_dim = q_dim_total + k_dim_total + v_dim_total; // 10240
    let eps = config.rms_norm_eps;

    // in projections.
    let qkv = gemv_layer(ctx, weights, scratch, layer, "attn_qkv", normed)?; // [qkv_dim]
    let z = gemv_layer(ctx, weights, scratch, layer, "attn_gate", normed)?; // [v_dim_total]
    let a_proj = gemv_layer(ctx, weights, scratch, layer, "ssm_alpha", normed)?; // [nv]
    let b_proj = gemv_layer(ctx, weights, scratch, layer, "ssm_beta", normed)?; // [nv]

    // Depthwise conv1d over all qkv channels (kernel taps = [ring | x]), then
    // SiLU. Matches qwen35_ssm_conv.comp / conv1d.cu: the bf16-round of the
    // pre-activation is applied before SiLU.
    let conv_w = dequant_f32(
        weights
            .get(&format!("blk.{layer}.ssm_conv1d.weight"))
            .ok_or_else(|| anyhow!("missing blk.{layer}.ssm_conv1d.weight"))?,
        kernel * qkv_dim,
    )?; // [channel*kernel] row-major (ne0=kernel)
    let ring = &mut state.conv_ring[linear_idx];
    let state_width = kernel - 1;
    let mut qkv_conv = vec![0.0f32; qkv_dim];
    for c in 0..qkv_dim {
        let mut sum = 0.0f32;
        for k in 0..kernel {
            // src_t = t - state_width + k, with t=0 (single token): taps the
            // ring for k<state_width, the current input for k==state_width.
            let value = if k < state_width {
                ring[c * state_width + k]
            } else {
                qkv[c]
            };
            sum += value * conv_w[c * kernel + k];
        }
        qkv_conv[c] = silu(round_to_bf16(sum));
    }
    // Advance the ring (qwen35_ssm_conv.comp tail, seq_len=1): the new
    // `state_width` taps per channel are the sequence's last `state_width`
    // inputs = a left-shift of the old ring with the current input appended:
    // `[old[1], old[2], ..., old[sw-1], qkv[c]]`.
    if state_width > 0 {
        for c in 0..qkv_dim {
            let base = c * state_width;
            for i in 0..state_width - 1 {
                ring[base + i] = ring[base + i + 1];
            }
            ring[base + state_width - 1] = qkv[c];
        }
    }

    // A_log / dt_bias (f32 GGUF tensors).
    let a_log = dequant_f32(
        weights
            .get(&format!("blk.{layer}.ssm_a"))
            .ok_or_else(|| anyhow!("missing blk.{layer}.ssm_a"))?,
        nv,
    )?;
    let dt_bias = dequant_f32(
        weights
            .get(&format!("blk.{layer}.ssm_dt.bias"))
            .ok_or_else(|| anyhow!("missing blk.{layer}.ssm_dt.bias"))?,
        nv,
    )?;

    // Recurrent gated-delta for one token. Per value head v (key head k =
    // v*nk/nv): l2-normalize q/k over key_dim (eps 1e-12), scale q by
    // rsqrt(key_dim); g = -exp(A_log)*softplus(a+dt_bias); beta = sigmoid(b).
    // Two-pass state update (decay then rank-1) matching the kernel.
    let gdr = &mut state.gdr_state[linear_idx];
    let mut gdr_out = vec![0.0f32; v_dim_total];
    for vh in 0..nv {
        let kh = vh * nk / nv;
        let q_head = &qkv_conv[kh * kd..kh * kd + kd];
        let k_head = &qkv_conv[q_dim_total + kh * kd..q_dim_total + kh * kd + kd];
        let v_head = &qkv_conv
            [q_dim_total + k_dim_total + vh * vd..q_dim_total + k_dim_total + vh * vd + vd];

        let mut q_sumsq = 0.0f32;
        let mut k_sumsq = 0.0f32;
        for j in 0..kd {
            q_sumsq += q_head[j] * q_head[j];
            k_sumsq += k_head[j] * k_head[j];
        }
        let q_norm = 1.0 / (q_sumsq + 1.0e-12).sqrt();
        let k_norm = 1.0 / (k_sumsq + 1.0e-12).sqrt();
        let q_scale = q_norm * (1.0 / (kd as f32).sqrt());

        let a_val = a_proj[vh];
        let b_val = b_proj[vh];
        let exp_g = (-(a_log[vh].exp()) * softplus(a_val + dt_bias[vh])).exp();
        let beta = sigmoid(b_val);

        let base = vh * kd * vd;
        // Pass 1: decay state, accumulate kv_mem[val] = sum_j state*k.
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
        // Pass 2: rank-1 update + output.
        let out = &mut gdr_out[vh * vd..vh * vd + vd];
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
            out[val] = acc;
        }
    }

    // Gated output RMSNorm: per value head over val_dim, PLAIN f32 weight
    // (broadcast across heads), × silu(z). Matches rms_norm_gated_kernel.
    let ssm_norm = dequant_f32(
        weights
            .get(&format!("blk.{layer}.ssm_norm.weight"))
            .ok_or_else(|| anyhow!("missing blk.{layer}.ssm_norm.weight"))?,
        vd,
    )?;
    let mut normed_out = vec![0.0f32; v_dim_total];
    for vh in 0..nv {
        let x = &gdr_out[vh * vd..vh * vd + vd];
        let gate = &z[vh * vd..vh * vd + vd];
        let mut sumsq = 0.0f32;
        for &xv in x {
            sumsq += xv * xv;
        }
        let inv = 1.0 / (sumsq / vd as f32 + eps).sqrt();
        let out = &mut normed_out[vh * vd..vh * vd + vd];
        for d in 0..vd {
            out[d] = x[d] * inv * ssm_norm[d] * silu(gate[d]);
        }
    }

    // out_proj: [v_dim_total -> hidden].
    gemv_layer(ctx, weights, scratch, layer, "ssm_out", &normed_out)
}

// ─────────────────────────────────────────────────────────────────────────────
// On-device GEMV helpers (the proven q8_0 path).
// ─────────────────────────────────────────────────────────────────────────────

/// Run one quantized GEMV `y[out] = W[out,in] · x[in]` on the device for a
/// per-layer weight tensor `blk.{layer}.{suffix}.weight`.
fn gemv_layer(
    ctx: &VulkanContext,
    weights: &ResidentWeights<'_>,
    scratch: &mut GemvScratch<'_>,
    layer: usize,
    suffix: &str,
    x: &[f32],
) -> Result<Vec<f32>> {
    let name = format!("blk.{layer}.{suffix}.weight");
    let w = weights
        .get(&name)
        .ok_or_else(|| anyhow!("missing weight {name}"))?;
    gemv_device(ctx, w, scratch, x, &name)
}

/// Run a GEMV for a global (non-layer) weight tensor with a known output width.
fn gemv_global(
    ctx: &VulkanContext,
    weights: &ResidentWeights<'_>,
    scratch: &mut GemvScratch<'_>,
    name: &str,
    x: &[f32],
    _expected_out: usize,
) -> Result<Vec<f32>> {
    let w = weights
        .get(name)
        .ok_or_else(|| anyhow!("missing weight {name}"))?;
    gemv_device(ctx, w, scratch, x, name)
}

/// Core GEMV: quantize `x` to q8_1_x4 on device, dispatch the Q8_0 GEMV, read
/// back `[nrows]` f32. `nrows`/`ncols` are derived from the weight's GGUF dims
/// (`dims = [in, out]`, row-major `[out, in]` bytes).
fn gemv_device(
    ctx: &VulkanContext,
    weight: &DeviceTensor<'_>,
    scratch: &mut GemvScratch<'_>,
    x: &[f32],
    name: &str,
) -> Result<Vec<f32>> {
    let (ncols, nrows) = weight_dims(weight, name)?;
    if x.len() != ncols {
        bail!(
            "{name}: GEMV input width {} != weight in-dim (ncols) {ncols}",
            x.len()
        );
    }
    if !matches!(weight.residency, Residency::KeepQuant(_)) {
        bail!(
            "{name}: GEMV expects a packed-quant weight, got {:?}",
            weight.residency
        );
    }

    // The scratch is pre-sized in `forward_token` to the widest GEMV shape, so
    // every dispatch fits without reallocation (which would otherwise tie the
    // scratch buffers' lifetime to this call's `ctx` borrow).
    if ncols > scratch.quant_cap_cols {
        bail!(
            "{name}: GEMV ncols {ncols} exceeds pre-sized quant scratch ({})",
            scratch.quant_cap_cols
        );
    }
    if nrows > scratch.dst_cap_rows {
        bail!(
            "{name}: GEMV nrows {nrows} exceeds pre-sized dst scratch ({})",
            scratch.dst_cap_rows
        );
    }

    // 1. Upload + quantize the activation vector to q8_1_x4.
    let quant_bytes = q8_1_x4_bytes(ncols);
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut x_in = DeviceBuffer::alloc(ctx, x_bytes.len())
        .map_err(|e| anyhow!("{name}: alloc gemv x input: {e}"))?;
    x_in.copy_from_host(&x_bytes)
        .map_err(|e| anyhow!("{name}: upload gemv x: {e}"))?;
    // Zero the quant region we will write (the x4 grouping may round ncols up).
    let zero = vec![0u8; quant_bytes];
    scratch
        .quant
        .copy_from_host(&zero)
        .map_err(|e| anyhow!("{name}: zero gemv quant: {e}"))?;
    let qparams = q8_1_quantize_params(ncols as u32);
    let qdispatch = q8_1_quantize_dispatch(ncols as u32);
    q8_1_quantize(ctx, &[&x_in, &scratch.quant], qdispatch, &qparams)
        .map_err(|e| anyhow!("{name}: q8_1_quantize dispatch: {e}"))?;

    // 2. Dispatch the Q8_0 GEMV. Bindings: [A weights, B q8_1_x4, D f32 dst,
    //    Fuse0, Fuse1] (3/4 are dummies, fusion_flags=0).
    let dst_bytes = nrows * std::mem::size_of::<f32>();
    scratch
        .dst
        .copy_from_host(&vec![0u8; dst_bytes])
        .map_err(|e| anyhow!("{name}: zero gemv dst: {e}"))?;
    let mut f0 = DeviceBuffer::alloc(ctx, 4).map_err(|e| anyhow!("{name}: alloc fuse0: {e}"))?;
    f0.copy_from_host(&[0u8; 4]).ok();
    let mut f1 = DeviceBuffer::alloc(ctx, 4).map_err(|e| anyhow!("{name}: alloc fuse1: {e}"))?;
    f1.copy_from_host(&[0u8; 4]).ok();

    let params: KernelParams = gemv_params(ncols as u32, nrows as u32);
    let dispatch: Dispatch = gemv_dispatch(nrows as u32);
    q8_0_gemv_with_params(
        ctx,
        &[&weight.buffer, &scratch.quant, &scratch.dst, &f0, &f1],
        dispatch,
        &params,
    )
    .map_err(|e| anyhow!("{name}: q8_0 GEMV dispatch: {e}"))?;

    // 3. Read back the f32 result rows.
    let mut out_bytes = vec![0u8; dst_bytes];
    scratch
        .dst
        .copy_to_host(&mut out_bytes[..])
        .map_err(|e| anyhow!("{name}: read back gemv dst: {e}"))?;
    let out: Vec<f32> = out_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(out)
}

/// `(ncols=in, nrows=out)` from a weight's recorded GGUF dims. GGUF stores
/// `dims = [ne0=in, ne1=out]` and the bytes are row-major `[out, in]`, which is
/// exactly the GEMV's `[nrows, ncols]` contract. The loader records these at
/// upload time ([`DeviceTensor::gemv_dims`]).
fn weight_dims(weight: &DeviceTensor<'_>, name: &str) -> Result<(usize, usize)> {
    weight
        .gemv_dims
        .ok_or_else(|| anyhow!("{name}: resident tensor has no GEMV dims recorded"))
}

// ─────────────────────────────────────────────────────────────────────────────
// Host f32 elementwise / norm primitives (transcribed from the CUDA reference).
// ─────────────────────────────────────────────────────────────────────────────

/// `(1+w)` offset RMSNorm: `out[i] = x[i] * inv_rms * (1 + w[i])`,
/// `inv_rms = 1/sqrt(mean(x^2) + eps)`. Qwen3.5 input / post / final / q-k norms.
fn rms_norm_offset(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut sumsq = 0.0f32;
    for &v in x {
        sumsq += v * v;
    }
    let inv = 1.0 / (sumsq / n as f32 + eps).sqrt();
    (0..n).map(|i| x[i] * inv * (1.0 + w[i])).collect()
}

/// NeoX-style RoPE on one head vector `[head_dim]`. Pairs `(x[d], x[d+half])`
/// rotate by `angle = pos * theta^(-2d/rotary_dim)` for `d < half`; dims beyond
/// `rotary_dim` pass through (here rotary_dim == head_dim so none).
fn rope_neox(x: &[f32], pos: usize, half: usize, head_dim: usize, theta: f32) -> Vec<f32> {
    let mut out = x.to_vec();
    let rotary_dim = half * 2;
    for d in 0..half {
        let inv_freq = theta.powf(-(2.0 * d as f32) / rotary_dim as f32);
        let angle = pos as f32 * inv_freq;
        let (s, c) = angle.sin_cos();
        let x0 = x[d];
        let x1 = x[d + half];
        out[d] = x0 * c - x1 * s;
        out[d + half] = x0 * s + x1 * c;
    }
    // dims >= rotary_dim already copied through by to_vec().
    let _ = head_dim;
    out
}

/// SwiGLU: `silu(gate) * up` elementwise.
fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter().zip(up).map(|(&g, &u)| silu(g) * u).collect()
}

fn add_vec(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(&x, &y)| x + y).collect()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// Round an f32 to bf16 precision (truncate-to-nearest-even on the low 16 bits),
/// matching the conv1d shader's `round_to_bf16`.
fn round_to_bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    f32::from_bits((bits.wrapping_add(0x7fff).wrapping_add(lsb)) & 0xffff_0000)
}

// ─────────────────────────────────────────────────────────────────────────────
// Norm-weight dequant helpers.
// ─────────────────────────────────────────────────────────────────────────────

/// Dequantize a per-layer norm/SSM-param tensor `blk.{layer}.{suffix}.weight`
/// to `[len]` f32. These are F32 on device (`DequantF32`), so this is a copy.
fn norm_weight(weights: &ResidentWeights<'_>, layer: usize, suffix: &str) -> Result<Vec<f32>> {
    let name = format!("blk.{layer}.{suffix}.weight");
    let t = weights
        .get(&name)
        .ok_or_else(|| anyhow!("missing norm weight {name}"))?;
    let len = t
        .gemv_dims
        .map(|(c, _r)| c)
        .unwrap_or_else(|| t.buffer.len() / 4);
    dequant_f32(t, len)
}

/// Read a device-resident F32 tensor's first `len` elements to host.
fn dequant_f32(t: &DeviceTensor<'_>, len: usize) -> Result<Vec<f32>> {
    if !matches!(t.residency, Residency::DequantF32) {
        bail!(
            "{}: expected F32-resident tensor, got {:?}",
            t.name,
            t.residency
        );
    }
    let want = len * std::mem::size_of::<f32>();
    if t.buffer.len() < want {
        bail!(
            "{}: F32 tensor buffer {} B < {len} f32 needed",
            t.name,
            t.buffer.len()
        );
    }
    let mut bytes = vec![0u8; want];
    t.buffer
        .copy_to_host(&mut bytes[..])
        .map_err(|e| anyhow!("{}: read back F32 tensor: {e}", t.name))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
