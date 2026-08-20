//! **Batched prefill** for the Qwen3.5 dense lane on Vulkan.
//!
//! Decode ([`crate::forward::forward_token`]) runs one token at a time, so every
//! projection is a GEMV: a `[rows, cols]` weight read once per token to produce a
//! single output vector. That is memory-bound by construction — at 16 GiB of
//! weights the whole model streams from LPDDR5X for each token. Prefill has `T`
//! tokens available at once, so the SAME weight read produces `T` output vectors:
//! the arithmetic intensity rises `T×` and the projection becomes compute-bound.
//! llama.cpp does exactly this (`ggml_vk_mul_mat_q` → `mul_mmq_*`), which is why
//! its `pp256` is ~70× our per-token prefill loop on the same box.
//!
//! This module records the same shapes through the **already-validated**
//! `mul_mmq_{q4_k,q5_k,q6_k,q8_0}` kernels (see
//! `vulkan-kernels/tests/device_mmq.rs`), plus the batched forms of every
//! surrounding elementwise/norm/rope/attention op. The numerics are the decode
//! path's, token-for-token: the only change is that each op covers `T` rows.
//!
//! ## Layout contract
//!
//! Everything in the prefill arena is **token-major**: an activation of width `w`
//! for a chunk of `T` tokens is `[T][w]` row-major, so token `t`'s vector starts
//! at element `t*w`. That is exactly:
//!   * what `mul_mmq` produces (`D` is `[n, m]` with `stride_d = m`, i.e. one
//!     contiguous `m`-wide row per token — see [`mmq_params`]),
//!   * what `qwen35_ssm_conv.comp` / `qwen35_gated_delta_net.comp` already expect
//!     for `seq_len > 1` (both index `token * stride + channel`), and
//!   * what the batched flash-attention writes (`ne1 = n_q_heads` makes `O`
//!     `[T][head][head_dim]`).
//!
//! so no transposes are ever recorded.
//!
//! ## Scope
//!
//! v1 covers the DENSE Qwen3.5 stack (interleaved gated-delta + full attention).
//! MoE layers are NOT batched; [`prefill_supported`] reports that up front so the
//! caller keeps the per-token loop for those models rather than this module
//! bailing mid-record.

use anyhow::{Context, Result, anyhow, bail};

use qwen35_spec::{LayerType, Qwen35Config};
use vulkan_kernels::{
    CoopmatShape, FlashAttentionSpec, Kernel, MmSpec, MmqSpec, add_dispatch, add_params,
    f16_kv_pack_dispatch_rows, f16_kv_pack_params, f16_kv_pack_params_rows,
    flash_attn_dispatch_batched, flash_attn_params_batched, mm_dispatch, mmq_dispatch, mmq_params,
    q8_1_quantize_dispatch, q8_1_quantize_params, qwen35_gated_delta_net_dispatch,
    qwen35_gated_delta_net_params, qwen35_ssm_conv_dispatch, qwen35_ssm_conv_params,
    qwen36_router_gemv_dispatch, qwen36_router_gemv_params, rms_norm_dispatch_rows,
    rms_norm_params_rows, rope_neox_dispatch_batched, rope_neox_params_batched,
    sigmoid_mul_dispatch, sigmoid_mul_params_strided, swiglu_dispatch, swiglu_params,
};
use vulkan_sys::{DeviceBuffer, VulkanContext};

use crate::forward::{
    DecodeResources, KV_CACHE_MAX_SEQ, Qwen35ForwardState, Slot, align_up, final_norm_lm_head,
    packed_layer_weight, packed_or_f32_norm, q8_1_x4_bytes, ssm_weight_buffer, submit_dispatch_cap,
    weight_dims,
};
use crate::loader::Residency;
use crate::loader::upload::{DeviceTensor, ResidentWeights};
use infer_gguf::gguf::GgmlType;

/// Default tokens per batched-prefill chunk. Sizes the [`PrefillArena`] (≈40 MB
/// for the 27B at 64) and the causal mask. Bigger chunks amortize the weight read
/// over more tokens and, past 64, let [`MmqSpec::choose`] pick its LARGE tile
/// instead of the medium one — but they grow the scratch linearly and lengthen a
/// single command buffer.
pub const PREFILL_CHUNK_TOKENS: usize = 64;

/// The chunk width to build the arena for. `ARLE_VULKAN_PREFILL_CHUNK` overrides
/// [`PREFILL_CHUNK_TOKENS`] so the width can be swept against a real model
/// without a rebuild — it trades scratch bytes for arithmetic intensity, and the
/// best value is a property of the device, not of this code.
#[must_use]
pub fn prefill_chunk_tokens() -> usize {
    std::env::var("ARLE_VULKAN_PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(PREFILL_CHUNK_TOKENS)
}

/// f16 bit patterns for the causal mask. The flash kernel adds the mask value to
/// the pre-softmax score, so `0` = allowed and `-inf` = masked. Every query row
/// has at least its own diagonal allowed, so a row is never all `-inf` (which
/// would make the softmax denominator 0).
const F16_ZERO: u16 = 0x0000;
const F16_NEG_INF: u16 = 0xFC00;

// ─────────────────────────────────────────────────────────────────────────────
// PrefillArena — the decode arena's `T`-token sibling.
// ─────────────────────────────────────────────────────────────────────────────

/// Batched-prefill scratch: one wide UMA buffer of `T`-token slots, each aligned
/// to `minStorageBufferOffsetAlignment` so it binds as a storage-buffer
/// sub-range.
///
/// Deliberately SEPARATE from [`crate::forward::DeviceArena`] rather than a
/// widening of it: the decode arena's slots are sized from
/// `widest_gemv`, which folds in `vocab_size` (151936). Scaling that by `T` would
/// allocate gigabytes for a scratch the batched path never uses — the LM head
/// runs on the LAST token only, through the existing decode
/// [`final_norm_lm_head`]. Keeping the two apart also means the decode path is
/// bit-for-bit untouched by this module.
pub struct PrefillArena<'a> {
    buffer: DeviceBuffer<'a>,
    /// Max tokens per chunk (the `T` every slot is sized for).
    max_tokens: usize,
    /// q8_1_x4 of a `[T][max_cols]` activation (the `B` operand of every `mmq`),
    /// where `max_cols` is the widest single-token activation any batched op
    /// touches (the FFN intermediate for the 27B). `vocab_size` is deliberately
    /// NOT folded in — see the type doc.
    quant: Slot,
    /// 4-byte fuse dummies for the GEMV/rope/flash unread bindings.
    fuse0: Slot,
    fuse1: Slot,
    /// General-purpose `[T][max_cols]` f32 work slots (the dense FFN's live set).
    work: [Slot; 4],
    /// `[T][hidden]` residual stream.
    hid: Slot,
    /// `[T][hidden]` input/post RMSNorm output.
    normed: Slot,
    /// `[T][hidden]` attention-block output (consumed by the FFN's post-add).
    attn_res: Slot,
    /// `[T][n_q_heads][2*head_dim]` gated q-projection (`[query|gate]` per head).
    attn_qkv: Slot,
    /// `[T][n_q_heads][head_dim]` normed+roped query.
    attn_q: Slot,
    /// `[T][n_kv_heads][head_dim]` normed+roped key (then f16 → cache).
    attn_k: Slot,
    /// `[T][n_q_heads][head_dim]` flash output, gated in place.
    attn_out: Slot,
    /// `[T]` i32 absolute RoPE positions.
    pos: Slot,
    /// `[T][KV_CACHE_MAX_SEQ]` f16 causal mask (row stride is the CHUNK's
    /// `kv_len`, not the cap — the tail of the allocation is simply unused).
    mask: Slot,
    /// `[T][qkv_dim]` raw linear in-projection (conv input).
    lin_xseq: Slot,
    /// `[T][qkv_dim]` post-conv `silu(conv)` (gated-delta input).
    lin_qkv_conv: Slot,
    /// `[T][n_value_heads]` `ssm_alpha` / `ssm_beta` projections.
    lin_a: Slot,
    lin_b: Slot,
    /// `[T][n_value_heads*value_head_dim]` gated-delta output.
    lin_out: Slot,
}

impl<'a> PrefillArena<'a> {
    /// Allocate the chunk scratch for `max_tokens` tokens of `config`.
    pub fn new(ctx: &'a VulkanContext, config: &Qwen35Config, max_tokens: usize) -> Result<Self> {
        let t = max_tokens.max(1);
        let align = ctx.min_storage_buffer_offset_alignment().max(1) as usize;
        let f32_b = std::mem::size_of::<f32>();

        let h = config.hidden_size;
        let hd = config.head_dim;
        let q_dim = config.num_attention_heads * hd;
        let kv_dim = config.num_key_value_heads * hd;
        let v_dim_total = config.linear_num_value_heads * config.linear_value_head_dim;
        let qkv_dim = 2 * config.linear_num_key_heads * config.linear_key_head_dim + v_dim_total;
        // Widest activation the batched path quantizes or holds in a work slot.
        // NOT `vocab_size` — see the struct doc.
        let max_cols = config
            .intermediate_size
            .max(h)
            .max(2 * q_dim)
            .max(qkv_dim)
            .max(v_dim_total)
            .max(1);

        let mut cursor = 0u64;
        let mut place = |len: usize| -> Slot {
            let offset = cursor;
            cursor += align_up(len.max(4), align) as u64;
            Slot { offset, len }
        };
        // The B-operand staging slot. `mul_mmq` wants q8_1_x4 (1.125 B/elem);
        // the coopmat `mul_mm` wants plain f16 (2 B/elem), which is strictly
        // larger. Size for the max so the route can be chosen per device (and
        // flipped by `ARLE_VK_DISABLE_COOPMAT`) without resizing the arena.
        let quant = place(q8_1_x4_bytes(t * max_cols).max(t * max_cols * 2));
        let fuse0 = place(4);
        let fuse1 = place(4);
        let work_len = t * max_cols * f32_b;
        let work = [
            place(work_len),
            place(work_len),
            place(work_len),
            place(work_len),
        ];
        let hid = place(t * h * f32_b);
        let normed = place(t * h * f32_b);
        let attn_res = place(t * h * f32_b);
        let attn_qkv = place(t * 2 * q_dim * f32_b);
        let attn_q = place(t * q_dim * f32_b);
        let attn_k = place(t * kv_dim * f32_b);
        let attn_out = place(t * q_dim * f32_b);
        let pos = place(t * std::mem::size_of::<i32>());
        let mask = place(t * KV_CACHE_MAX_SEQ * std::mem::size_of::<u16>());
        let lin_xseq = place(t * qkv_dim * f32_b);
        let lin_qkv_conv = place(t * qkv_dim * f32_b);
        let lin_a = place(t * config.linear_num_value_heads.max(1) * f32_b);
        let lin_b = place(t * config.linear_num_value_heads.max(1) * f32_b);
        let lin_out = place(t * v_dim_total.max(1) * f32_b);
        let total = cursor as usize;

        let buffer = DeviceBuffer::alloc_uma(ctx, total.max(4))
            .map_err(|e| anyhow!("alloc prefill arena ({total} B): {e}"))?;
        Ok(Self {
            buffer,
            max_tokens: t,
            quant,
            fuse0,
            fuse1,
            work,
            hid,
            normed,
            attn_res,
            attn_qkv,
            attn_q,
            attn_k,
            attn_out,
            pos,
            mask,
            lin_xseq,
            lin_qkv_conv,
            lin_a,
            lin_b,
            lin_out,
        })
    }

    /// Tokens per chunk this arena is sized for.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    fn write_f32_at(&mut self, off: u64, data: &[f32]) -> Result<()> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.buffer
            .copy_from_host_at(off, &bytes)
            .map_err(|e| anyhow!("write prefill arena @{off}: {e}"))
    }

    fn write_bytes_at(&mut self, off: u64, bytes: &[u8]) -> Result<()> {
        self.buffer
            .copy_from_host_at(off, bytes)
            .map_err(|e| anyhow!("write prefill arena @{off}: {e}"))
    }

    fn read_f32_at(&self, off: u64, len: usize) -> Result<Vec<f32>> {
        let mut bytes = vec![0u8; len * std::mem::size_of::<f32>()];
        self.buffer
            .copy_to_host_at(off, &mut bytes[..])
            .map_err(|e| anyhow!("read prefill arena @{off}: {e}"))?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry points.
// ─────────────────────────────────────────────────────────────────────────────

/// Whether this `(config, weights)` pair can run the batched path.
///
/// The caller checks ONCE, before any recording, and falls back to the per-token
/// loop on `false` — a mid-record bail would leave the recorder open with a
/// partially-applied chunk (the KV cache and the gated-delta state are advanced
/// in place, so it is not restartable).
///
/// Excluded for v1:
/// * **MoE layers** — the batched expert gather (`mul_mat_id` over `T` tokens
///   with per-token routing) is a separate kernel contract from `mul_mmq`.
///
/// F32-resident `ssm_alpha`/`ssm_beta` are NOT excluded: the loader dequantizes
/// those two on the 27B, and `router_gemv` has no batched form, so
/// [`pf_router_gemv`] runs `T` iterations of it. They are `[hidden, 48]` — the
/// smallest weights in the layer — so the loop costs nothing next to the
/// projections around it.
pub fn prefill_supported(config: &Qwen35Config, weights: &ResidentWeights<'_>) -> bool {
    prefill_unsupported_reason(config, weights).is_none()
}

/// Why [`prefill_supported`] would say `false`, or `None` when it says `true`.
///
/// Split out so the fallback is diagnosable: a silent `Ok(None)` from the batched
/// entry point is indistinguishable from "the batched path ran and was slow",
/// which is exactly the confusion that costs an A/B run.
pub fn prefill_unsupported_reason(
    config: &Qwen35Config,
    weights: &ResidentWeights<'_>,
) -> Option<String> {
    for (layer, &layer_type) in config.layer_types.iter().enumerate() {
        if config.is_moe_layer(layer) {
            return Some(format!("layer {layer} is MoE"));
        }
        if layer_type == LayerType::LinearAttention {
            for suffix in ["ssm_alpha", "ssm_beta"] {
                let name = format!("blk.{layer}.{suffix}.weight");
                if weights.get(&name).is_none() {
                    return Some(format!("{name} is missing"));
                }
            }
        }
    }
    None
}

/// Batched prefill of `tokens` starting at `start_pos`. Returns the logits
/// `[vocab]` of the LAST token — the same value the per-token loop's final
/// [`crate::forward::forward_token`] would return, and the only logits a prefill
/// consumer needs.
///
/// Advances the device KV cache, the gated-delta recurrent state and
/// `state.seq_len` by `tokens.len()`, exactly as the per-token loop would.
/// `tokens` is processed in chunks of [`PrefillArena::max_tokens`]; each chunk is
/// ONE command buffer (subject to the `--vulkan-submit-cap` TDR valve).
pub fn forward_prefill<'a>(
    ctx: &'a VulkanContext,
    config: &Qwen35Config,
    weights: &ResidentWeights<'_>,
    res: &mut DecodeResources<'a>,
    state: &mut Qwen35ForwardState,
    tokens: &[u32],
    start_pos: usize,
) -> Result<Vec<f32>> {
    if tokens.is_empty() {
        bail!("forward_prefill: empty token chunk");
    }
    if start_pos != state.seq_len {
        bail!(
            "forward_prefill start_pos {start_pos} != materialized seq_len {} \
             (feed tokens in order)",
            state.seq_len
        );
    }
    if !prefill_supported(config, weights) {
        bail!("forward_prefill: model not supported by the batched path (MoE / F32 ssm a,b)");
    }
    let end = start_pos + tokens.len();
    if end > KV_CACHE_MAX_SEQ {
        bail!("forward_prefill: position {end} exceeds device KV cache cap {KV_CACHE_MAX_SEQ}");
    }

    let chunk_tokens = res.prefill.max_tokens();
    let mut pos = start_pos;
    let mut last_hidden: Vec<f32> = Vec::new();
    let t0 = std::time::Instant::now();
    for chunk in tokens.chunks(chunk_tokens) {
        last_hidden = forward_prefill_chunk(ctx, config, weights, res, chunk, pos)?;
        pos += chunk.len();
    }
    let secs = t0.elapsed().as_secs_f64();
    log::info!(
        "vulkan batched prefill: {} tok @ {start_pos} in {:.3}s ({:.1} tok/s, chunk={chunk_tokens})",
        tokens.len(),
        secs,
        tokens.len() as f64 / secs.max(f64::MIN_POSITIVE),
    );
    state.seq_len = end;

    final_norm_lm_head(ctx, weights, res, &last_hidden, config.rms_norm_eps)
}

/// One chunk: records every layer over the whole `[T]` chunk into ONE command
/// buffer and returns the LAST token's hidden state `[hidden]`.
fn forward_prefill_chunk<'a>(
    ctx: &'a VulkanContext,
    config: &Qwen35Config,
    weights: &ResidentWeights<'_>,
    res: &mut DecodeResources<'a>,
    tokens: &[u32],
    start_pos: usize,
) -> Result<Vec<f32>> {
    let t = tokens.len();
    let h = config.hidden_size;
    let eps = config.rms_norm_eps;
    let kv_len = start_pos + t;

    res.reset_rings();

    // ── Host staging: embeddings, RoPE positions, the causal mask. ──
    let mut embed = Vec::with_capacity(t * h);
    for &token in tokens {
        let row = weights
            .embedding
            .embed_row(token)
            .with_context(|| format!("embed token {token}"))?;
        if row.len() != h {
            bail!("embedding row width {} != hidden {h}", row.len());
        }
        embed.extend_from_slice(&row);
    }
    let positions: Vec<u8> = (0..t)
        .flat_map(|i| ((start_pos + i) as i32).to_le_bytes())
        .collect();
    // Causal mask `[T][kv_len]` f16, shared by every head and every full layer:
    // query row `r` is at absolute position `start_pos + r`, so key `c` is
    // allowed iff `c <= start_pos + r`. `m_stride = KV`, and the flash kernel
    // bounds-checks `j*Bc + c < KV`, so the row stride is exactly `kv_len`.
    let mut mask = Vec::with_capacity(t * kv_len * 2);
    for r in 0..t {
        let limit = start_pos + r;
        for c in 0..kv_len {
            let bits = if c <= limit { F16_ZERO } else { F16_NEG_INF };
            mask.extend_from_slice(&bits.to_le_bytes());
        }
    }

    let hid_off = res.prefill.hid.offset;
    let normed_off = res.prefill.normed.offset;
    let attn_off = res.prefill.attn_res.offset;
    let pos_off = res.prefill.pos.offset;
    let mask_off = res.prefill.mask.offset;
    res.prefill.write_f32_at(hid_off, &embed)?;
    res.prefill.write_bytes_at(pos_off, &positions)?;
    res.prefill.write_bytes_at(mask_off, &mask)?;

    let cap = submit_dispatch_cap();
    // Ring slots one more layer can possibly need. ring3 is the outlier: a
    // linear layer with F32 `ssm_alpha`/`ssm_beta` records `2*T` sets there for
    // the per-token a/b fallback on top of its ~10 mmq/norm sets. Every other
    // ring stays in single digits per layer (ring2 peaks at `2*n_kv + 4` f16
    // packs and quantizes on a full-attention layer).
    let ring3_headroom = 2 * t + 32;
    let ring_headroom = 16;
    res.recorder
        .begin()
        .map_err(|e| anyhow!("prefill chunk begin: {e}"))?;
    let mut full_idx = 0usize;
    let mut linear_idx = 0usize;
    for (layer, &layer_type) in config.layer_types.iter().enumerate() {
        let attn_norm = packed_or_f32_norm(weights, layer, "attn_norm")?;
        // input_layernorm over all T rows in ONE dispatch (`nrows = T`, source
        // row stride = hidden, so `hid` is read packed and `normed` written packed).
        pf_rms_norm_rows(
            ctx,
            res,
            &attn_norm.buffer,
            "prefill_input_norm",
            h,
            t,
            h,
            eps,
            hid_off,
            normed_off,
        )?;
        res.recorder.barrier();
        match layer_type {
            LayerType::FullAttention => {
                pf_full_attention(
                    ctx, config, weights, res, layer, full_idx, t, start_pos, normed_off, attn_off,
                )?;
                full_idx += 1;
            }
            LayerType::LinearAttention => {
                pf_linear_attention(
                    ctx, config, weights, res, layer, linear_idx, t, normed_off, attn_off,
                )?;
                linear_idx += 1;
            }
        }
        res.recorder.barrier();
        pf_dense_ffn(ctx, config, weights, res, layer, t, hid_off, attn_off)?;
        res.recorder.barrier();

        // Flush on either valve: the TDR dispatch cap, or a descriptor ring
        // about to wrap. `next_updated` wraps SILENTLY — it would rebind a set
        // an already-recorded dispatch still points at — and the F32
        // `ssm_alpha`/`ssm_beta` fallback records `2*T` ring3 sets per linear
        // layer, so a 64-layer chunk can outrun a ring sized for one token.
        // `submit_and_wait` has fenced the batch, so rewinding here is safe.
        let ring_low = res.ring3.remaining() < ring3_headroom
            || res.ring2.remaining() < ring_headroom
            || res.ring4.remaining() < ring_headroom
            || res.ring5.remaining() < ring_headroom
            || res.ring7.remaining() < ring_headroom;
        if res.recorder.dispatches_in_batch() as usize >= cap || ring_low {
            res.recorder
                .submit_and_wait()
                .map_err(|e| anyhow!("prefill layer[{layer}]: cap-flush submit: {e}"))?;
            res.reset_rings();
            res.recorder
                .begin()
                .map_err(|e| anyhow!("prefill layer[{layer}]: cap-flush begin: {e}"))?;
        }
    }
    res.recorder
        .submit_and_wait()
        .map_err(|e| anyhow!("prefill chunk submit: {e}"))?;

    let gpu_prof = res.recorder.take_gpu_profile();
    if !gpu_prof.is_empty() {
        let total: f64 = gpu_prof.iter().map(|(_, _, ms)| *ms).sum();
        let parts: Vec<String> = gpu_prof
            .iter()
            .map(|(label, count, ms)| format!("{label} {ms:.2}ms/{count}"))
            .collect();
        eprintln!(
            "  GPU OP PROFILE (prefill {t} tok @ {start_pos}): total {total:.2}ms | {}",
            parts.join(" | ")
        );
    }

    // Only the LAST token's hidden feeds the LM head.
    let last_off = hid_off + ((t - 1) * h * std::mem::size_of::<f32>()) as u64;
    res.prefill.read_f32_at(last_off, h)
}

// ─────────────────────────────────────────────────────────────────────────────
// Batched full attention.
// ─────────────────────────────────────────────────────────────────────────────

/// Record the whole full-attention block over a `[T]` chunk:
///
/// ```text
///   quantize(normed)                            [T][hidden]      -> quant
///   mmq(attn_q)  -> attn_qkv                    [T][2*q_dim]
///   mmq(attn_k)  -> attn_k                      [T][kv_dim]
///   mmq(attn_v)  -> work0                       [T][kv_dim]
///   rms_norm rows (nrows = T*nq, src stride 2*head_dim)  attn_qkv -> attn_q
///   rms_norm rows (nrows = T*nkv, in place)              attn_k
///   rope batched (heads = nq / nkv, tokens = T)
///   f16 kv-pack, one dispatch per kv head (rows = T)     -> KV cache [start_pos..]
///   flash-attn, ONE dispatch for all heads and tokens    -> attn_out
///   sigmoid gate, ONE strided dispatch                   attn_out *= σ(gate half)
///   quantize(attn_out) + mmq(attn_output)                -> out_off
/// ```
///
/// The query-half extraction of the interleaved `[query|gate]` q-projection is
/// FREE: it is just the `src_row_stride = 2*head_dim` of the batched RMSNorm.
#[allow(clippy::too_many_arguments)]
fn pf_full_attention<'a>(
    ctx: &'a VulkanContext,
    config: &Qwen35Config,
    weights: &ResidentWeights<'_>,
    res: &mut DecodeResources<'a>,
    layer: usize,
    full_idx: usize,
    t: usize,
    start_pos: usize,
    normed_off: u64,
    out_off_param: u64,
) -> Result<()> {
    let h = config.hidden_size;
    let hd = config.head_dim;
    let nq = config.num_attention_heads;
    let nkv = config.num_key_value_heads;
    let q_dim = nq * hd;
    let kv_dim = nkv * hd;
    let eps = config.rms_norm_eps;
    let kv_len = start_pos + t;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let f32_b = std::mem::size_of::<f32>() as u64;

    let q_norm = packed_or_f32_norm(weights, layer, "attn_q_norm")?;
    let k_norm = packed_or_f32_norm(weights, layer, "attn_k_norm")?;
    let q_w = packed_layer_weight(weights, layer, "attn_q")?;
    let k_w = packed_layer_weight(weights, layer, "attn_k")?;
    let v_w = packed_layer_weight(weights, layer, "attn_v")?;
    let o_w = packed_layer_weight(weights, layer, "attn_output")?;
    let (qw_in, qw_out) = weight_dims(q_w, "attn_q")?;
    let (kw_in, kw_out) = weight_dims(k_w, "attn_k")?;
    let (vw_in, vw_out) = weight_dims(v_w, "attn_v")?;
    let (ow_in, ow_out) = weight_dims(o_w, "attn_output")?;
    if qw_in != h || kw_in != h || vw_in != h {
        bail!("prefill full_attn[{layer}]: QKV in-dim q{qw_in}/k{kw_in}/v{vw_in} != hidden {h}");
    }
    if qw_out != 2 * q_dim || kw_out != kv_dim || vw_out != kv_dim {
        bail!(
            "prefill full_attn[{layer}]: QKV out-dim q{qw_out}/k{kw_out}/v{vw_out} != {}/{kv_dim}/{kv_dim}",
            2 * q_dim
        );
    }
    if ow_in != q_dim || ow_out != h {
        bail!("prefill full_attn[{layer}]: o_proj [{ow_in},{ow_out}] != [{q_dim},{h}]");
    }

    let qkv_off = res.prefill.attn_qkv.offset;
    let q_off = res.prefill.attn_q.offset;
    let k_off = res.prefill.attn_k.offset;
    let v_off = res.prefill.work[0].offset;
    let out_off = res.prefill.attn_out.offset;
    let quant_off = res.prefill.quant.offset;
    let pos_off = res.prefill.pos.offset;
    let mask_off = res.prefill.mask.offset;

    // q/k/v project the SAME `[T][hidden]` normed input: quantize once, three GEMMs.
    pf_pack_b(ctx, res, "prefill_attn_qin", t * h, normed_off, quant_off)?;
    res.recorder.barrier();
    pf_mm(
        ctx,
        res,
        q_w,
        "prefill_attn_q",
        h,
        2 * q_dim,
        t,
        quant_off,
        qkv_off,
    )?;
    pf_mm(
        ctx,
        res,
        k_w,
        "prefill_attn_k",
        h,
        kv_dim,
        t,
        quant_off,
        k_off,
    )?;
    pf_mm(
        ctx,
        res,
        v_w,
        "prefill_attn_v",
        h,
        kv_dim,
        t,
        quant_off,
        v_off,
    )?;
    res.recorder.barrier();

    // Per-head q RMSNorm over all (token, head) rows at once. Row `r = t*nq + hh`
    // reads `attn_qkv` at `r * 2*head_dim` — the QUERY half of head `hh` of token
    // `t` — and writes the packed `[T][nq][head_dim]` `attn_q`.
    pf_rms_norm_rows(
        ctx,
        res,
        &q_norm.buffer,
        "prefill_attn_q_norm",
        hd,
        t * nq,
        2 * hd,
        eps,
        qkv_off,
        q_off,
    )?;
    // k RMSNorm in place (source row stride == ncols, so aliasing is safe).
    pf_rms_norm_rows(
        ctx,
        res,
        &k_norm.buffer,
        "prefill_attn_k_norm",
        hd,
        t * nkv,
        hd,
        eps,
        k_off,
        k_off,
    )?;
    res.recorder.barrier();

    pf_rope(
        ctx,
        res,
        "prefill_attn_q_rope",
        hd,
        config.rotary_dim,
        nq,
        t,
        config.rope_theta,
        pos_off,
        q_off,
        q_off,
    )?;
    pf_rope(
        ctx,
        res,
        "prefill_attn_k_rope",
        hd,
        config.rotary_dim,
        nkv,
        t,
        config.rope_theta,
        pos_off,
        k_off,
        k_off,
    )?;
    res.recorder.barrier();

    // f16 KV pack: one kv head's `T` rows are strided by `nkv*head_dim` in the
    // arena but land CONTIGUOUSLY at `[start_pos .. start_pos+T]` in the cache
    // plane — one dispatch per (layer, kv_head) instead of one per token.
    for kvh in 0..nkv {
        let ksrc = k_off + (kvh * hd) as u64 * f32_b;
        let vsrc = v_off + (kvh * hd) as u64 * f32_b;
        pf_kv_pack(
            ctx,
            res,
            "prefill_kv_pack_k",
            hd,
            t,
            nkv * hd,
            ksrc,
            full_idx,
            kvh,
            start_pos,
            false,
        )?;
        pf_kv_pack(
            ctx,
            res,
            "prefill_kv_pack_v",
            hd,
            t,
            nkv * hd,
            vsrc,
            full_idx,
            kvh,
            start_pos,
            true,
        )?;
    }
    res.recorder.barrier();

    // ONE masked flash dispatch for the whole chunk: grid `(T, nq, 1)` with
    // `Br = 1`, `nek2 = nkv` so `ik2 = iq2 / (nq/nkv)` is the GQA kv head, and the
    // K/V bindings cover the layer's whole `[nkv][max_seq][head_dim]` slab.
    pf_flash(
        ctx,
        res,
        "prefill_flash",
        hd,
        t,
        kv_len,
        nq,
        nkv,
        full_idx,
        scale,
        q_off,
        mask_off,
        out_off,
    )?;
    res.recorder.barrier();

    // Per-head sigmoid gate for the whole chunk in ONE dispatch: value element
    // `i` (packed `[T][nq][head_dim]`) is gated by the gate half of the
    // interleaved `[T][nq][2*head_dim]` q-projection.
    pf_sigmoid_mul(
        ctx,
        res,
        "prefill_full_gate",
        t * q_dim,
        hd,
        2 * hd,
        hd,
        qkv_off,
        out_off,
        out_off,
    )?;
    res.recorder.barrier();

    pf_pack_b(ctx, res, "prefill_attn_oqin", t * q_dim, out_off, quant_off)?;
    res.recorder.barrier();
    pf_mm(
        ctx,
        res,
        o_w,
        "prefill_attn_output",
        q_dim,
        h,
        t,
        quant_off,
        out_off_param,
    )?;
    res.gemv_count += 4;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Batched linear (gated-delta) attention.
// ─────────────────────────────────────────────────────────────────────────────

/// Record the whole gated-delta block over a `[T]` chunk. The two model-specific
/// shaders (`qwen35_ssm_conv.comp`, `qwen35_gated_delta_net.comp`) ALREADY
/// implement the `seq_len > 1` recurrence against token-major buffers — decode
/// simply pins `seq_len = 1`. So the only change from the decode recipe is
/// `seq_len = T` in both push blocks, batched projections, and the per-value-head
/// `ssm_norm` loop collapsing into one `nrows = T*n_value_heads` dispatch.
#[allow(clippy::too_many_arguments)]
fn pf_linear_attention<'a>(
    ctx: &'a VulkanContext,
    config: &Qwen35Config,
    weights: &ResidentWeights<'_>,
    res: &mut DecodeResources<'a>,
    layer: usize,
    linear_idx: usize,
    t: usize,
    normed_off: u64,
    out_off_param: u64,
) -> Result<()> {
    let kd = config.linear_key_head_dim;
    let vd = config.linear_value_head_dim;
    let nk = config.linear_num_key_heads;
    let nv = config.linear_num_value_heads;
    let conv_kernel = config.linear_conv_kernel_dim;
    let v_dim_total = nv * vd;
    let qkv_dim = 2 * nk * kd + v_dim_total;
    let eps = config.rms_norm_eps;
    let h = config.hidden_size;

    let qkv_w = packed_layer_weight(weights, layer, "attn_qkv")?;
    let gate_w = packed_layer_weight(weights, layer, "attn_gate")?;
    let a_w = layer_weight(weights, layer, "ssm_alpha")?;
    let b_w = layer_weight(weights, layer, "ssm_beta")?;
    let ssm_out_w = packed_layer_weight(weights, layer, "ssm_out")?;
    let (qkv_in, qkv_out) = weight_dims(qkv_w, "attn_qkv")?;
    let (gate_in, gate_out) = weight_dims(gate_w, "attn_gate")?;
    let (a_in, a_out) = weight_dims(a_w, "ssm_alpha")?;
    let (b_in, b_out) = weight_dims(b_w, "ssm_beta")?;
    let (so_in, so_out) = weight_dims(ssm_out_w, "ssm_out")?;
    if qkv_in != h || gate_in != h || a_in != h || b_in != h {
        bail!("prefill linear[{layer}]: in-proj in-dim != hidden {h}");
    }
    if qkv_out != qkv_dim || gate_out != v_dim_total || a_out != nv || b_out != nv {
        bail!("prefill linear[{layer}]: in-proj out-dims disagree with the config");
    }
    if so_in != v_dim_total || so_out != h {
        bail!("prefill linear[{layer}]: ssm_out [{so_in},{so_out}] != [{v_dim_total},{h}]");
    }

    let conv_w_buf = ssm_weight_buffer(weights, layer, "ssm_conv1d.weight", conv_kernel * qkv_dim)?;
    let a_log_buf = ssm_weight_buffer(weights, layer, "ssm_a", nv)?;
    let dt_bias_buf = ssm_weight_buffer(weights, layer, "ssm_dt.bias", nv)?;
    let ssm_norm_buf = ssm_weight_buffer(weights, layer, "ssm_norm.weight", vd)?;

    let quant_off = res.prefill.quant.offset;
    let xseq_off = res.prefill.lin_xseq.offset;
    let qkv_conv_off = res.prefill.lin_qkv_conv.offset;
    let a_off = res.prefill.lin_a.offset;
    let b_off = res.prefill.lin_b.offset;
    let out_off = res.prefill.lin_out.offset;
    let z_off = res.prefill.work[1].offset;

    // Every in-projection reads the same `[T][hidden]` normed input.
    pf_pack_b(ctx, res, "prefill_lin_qin", t * h, normed_off, quant_off)?;
    res.recorder.barrier();
    pf_mm(
        ctx,
        res,
        qkv_w,
        "prefill_attn_qkv",
        h,
        qkv_dim,
        t,
        quant_off,
        xseq_off,
    )?;
    pf_mm(
        ctx,
        res,
        gate_w,
        "prefill_attn_gate",
        h,
        v_dim_total,
        t,
        quant_off,
        z_off,
    )?;
    // a/b (`[hidden -> nv]`, nv = 48) follow the decode path's residency split:
    // packed → `mul_mmq` over the chunk, F32-resident (the 27B dequantizes these
    // two) → `T` iterations of the F32 `router_gemv`. The loop is not a
    // regression worth avoiding: at `[5120,48]` the weight is ~1 MB, so 64
    // re-reads is ~63 MB against the ~16 GiB the chunk already streams.
    for (w, name, dst_off) in [
        (a_w, "prefill_ssm_alpha", a_off),
        (b_w, "prefill_ssm_beta", b_off),
    ] {
        if matches!(w.residency, Residency::KeepQuant(_)) {
            pf_mm(ctx, res, w, name, h, nv, t, quant_off, dst_off)?;
        } else {
            for token in 0..t {
                pf_router_gemv(
                    ctx,
                    res,
                    &w.buffer,
                    name,
                    nv,
                    h,
                    normed_off + (token * h * 4) as u64,
                    dst_off + (token * nv * 4) as u64,
                )?;
            }
        }
    }
    res.recorder.barrier();
    res.gemv_count += 4;

    // Depthwise conv1d + SiLU over the whole chunk, advancing the resident ring
    // exactly `T` steps (`seq_len = T`; the shader reads
    // `x_seq[t*num_channels + c]`, i.e. our token-major `[T][qkv_dim]`).
    let conv_push =
        qwen35_ssm_conv_params(qkv_dim as u32, t as u32, conv_kernel as u32).to_le_bytes();
    let conv_groups = {
        let d = qwen35_ssm_conv_dispatch(qkv_dim as u32);
        [d.x, d.y, d.z]
    };
    let xseq_bytes = (t * qkv_dim * std::mem::size_of::<f32>()) as u64;
    let conv_off = res.lin_conv_stride * linear_idx as u64;
    let conv_len = res.lin_conv_stride;
    {
        let DecodeResources {
            cache,
            recorder,
            prefill,
            ring4,
            lin_conv_state,
            ..
        } = res;
        let buf = &prefill.buffer;
        let (pipeline, _) = cache
            .get(
                ctx,
                Kernel::Qwen35SsmConv,
                Kernel::Qwen35SsmConv.specialization_u32(),
                conv_push.len() as u32,
                4,
            )
            .map_err(|e| anyhow!("prefill linear[{layer}]: build conv pipeline: {e}"))?;
        let set = ring4
            .next_updated(&[
                (buf, xseq_off, xseq_bytes),
                (conv_w_buf, 0, conv_w_buf.len() as u64),
                (lin_conv_state, conv_off, conv_len),
                (buf, qkv_conv_off, xseq_bytes),
            ])
            .map_err(|e| anyhow!("prefill linear[{layer}]: bind conv set: {e}"))?;
        recorder.label_next("lin_conv");
        recorder.dispatch_raw(pipeline, set, &conv_push, conv_groups);
    }
    res.recorder.barrier();

    // Gated-delta recurrence over the whole chunk (`seq_len = T`): one workgroup
    // per value head, each stepping the resident `[key_dim, val_dim]` state
    // through all `T` tokens in order — the same serial recurrence the per-token
    // loop performs, without the `T-1` intervening submits.
    let gd_push =
        qwen35_gated_delta_net_params(nk as u32, nv as u32, kd as u32, vd as u32, t as u32)
            .to_le_bytes();
    let gd_groups = {
        let d = qwen35_gated_delta_net_dispatch(nv as u32);
        [d.x, d.y, d.z]
    };
    let ab_bytes = (t * nv * std::mem::size_of::<f32>()) as u64;
    let out_bytes = (t * v_dim_total * std::mem::size_of::<f32>()) as u64;
    let gdr_off = res.lin_gdr_stride * linear_idx as u64;
    let gdr_len = res.lin_gdr_stride;
    {
        let DecodeResources {
            cache,
            recorder,
            prefill,
            ring7,
            lin_gdr_state,
            ..
        } = res;
        let buf = &prefill.buffer;
        let (pipeline, _) = cache
            .get(
                ctx,
                Kernel::Qwen35GatedDeltaNet,
                Kernel::Qwen35GatedDeltaNet.specialization_u32(),
                gd_push.len() as u32,
                7,
            )
            .map_err(|e| anyhow!("prefill linear[{layer}]: build gated-delta pipeline: {e}"))?;
        let set = ring7
            .next_updated(&[
                (buf, qkv_conv_off, xseq_bytes),
                (buf, b_off, ab_bytes),
                (buf, a_off, ab_bytes),
                (dt_bias_buf, 0, dt_bias_buf.len() as u64),
                (a_log_buf, 0, a_log_buf.len() as u64),
                (lin_gdr_state, gdr_off, gdr_len),
                (buf, out_off, out_bytes),
            ])
            .map_err(|e| anyhow!("prefill linear[{layer}]: bind gated-delta set: {e}"))?;
        // Split from `lin_conv` deliberately: these two share a layer but not a
        // scaling law — the conv is `T`-parallel, the gated-delta recurrence is
        // `T`-serial by construction (one workgroup per value head, stepping
        // tokens in order). Once the GEMM stopped dominating, the combined
        // `linear` category was 39% of a 256-token chunk with no way to tell
        // which half to attack.
        recorder.label_next("lin_gdr");
        recorder.dispatch_raw(pipeline, set, &gd_push, gd_groups);
    }
    res.recorder.barrier();

    // Gated output RMSNorm × silu(z): the decode path's per-value-head loop
    // collapses into ONE `nrows = T*n_value_heads` dispatch (the output is packed
    // `[T][nv][val_dim]`, so the row stride IS `val_dim`).
    pf_rms_norm_rows(
        ctx,
        res,
        ssm_norm_buf,
        "prefill_ssm_norm",
        vd,
        t * nv,
        vd,
        eps,
        out_off,
        qkv_conv_off,
    )?;
    res.recorder.barrier();
    pf_swiglu(
        ctx,
        res,
        "prefill_ssm_gated",
        t * v_dim_total,
        z_off,
        qkv_conv_off,
        out_off,
    )?;
    res.recorder.barrier();

    pf_pack_b(
        ctx,
        res,
        "prefill_ssm_out_qin",
        t * v_dim_total,
        out_off,
        quant_off,
    )?;
    res.recorder.barrier();
    pf_mm(
        ctx,
        res,
        ssm_out_w,
        "prefill_ssm_out",
        v_dim_total,
        h,
        t,
        quant_off,
        out_off_param,
    )?;
    res.gemv_count += 1;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Batched dense FFN.
// ─────────────────────────────────────────────────────────────────────────────

/// The dense FFN over a `[T]` chunk. Every op except the post-attention norm is
/// elementwise, so it runs FLAT over `T*n` — the per-token boundaries do not
/// matter. The norm is the one reduction, so it takes `nrows = T`.
#[allow(clippy::too_many_arguments)]
fn pf_dense_ffn<'a>(
    ctx: &'a VulkanContext,
    config: &Qwen35Config,
    weights: &ResidentWeights<'_>,
    res: &mut DecodeResources<'a>,
    layer: usize,
    t: usize,
    hid_off: u64,
    attn_off: u64,
) -> Result<()> {
    let h = config.hidden_size;
    let inter = config.intermediate_size;
    let eps = config.rms_norm_eps;

    let post_norm_w = weights
        .get(&format!("blk.{layer}.post_attention_norm.weight"))
        .ok_or_else(|| anyhow!("missing blk.{layer}.post_attention_norm.weight"))?;
    let gate_w = packed_layer_weight(weights, layer, "ffn_gate")?;
    let up_w = packed_layer_weight(weights, layer, "ffn_up")?;
    let down_w = packed_layer_weight(weights, layer, "ffn_down")?;
    let (gate_in, gate_out) = weight_dims(gate_w, "ffn_gate")?;
    let (down_in, down_out) = weight_dims(down_w, "ffn_down")?;
    if gate_in != h || gate_out != inter || down_in != inter || down_out != h {
        bail!(
            "prefill ffn[{layer}]: weight dims gate[{gate_in},{gate_out}] \
             down[{down_in},{down_out}] disagree with hidden {h} / inter {inter}"
        );
    }

    let w0 = res.prefill.work[0].offset; // post_sum (lives the whole FFN)
    let w1 = res.prefill.work[1].offset; // mlp_in -> act
    let w2 = res.prefill.work[2].offset; // gate -> mlp_out
    let w3 = res.prefill.work[3].offset; // up
    let quant_off = res.prefill.quant.offset;

    pf_add(
        ctx,
        res,
        "prefill_ffn_post_add",
        t * h,
        hid_off,
        attn_off,
        w0,
    )?;
    res.recorder.barrier();
    pf_rms_norm_rows(
        ctx,
        res,
        &post_norm_w.buffer,
        "prefill_ffn_post_norm",
        h,
        t,
        h,
        eps,
        w0,
        w1,
    )?;
    res.recorder.barrier();
    pf_pack_b(ctx, res, "prefill_ffn_qin", t * h, w1, quant_off)?;
    res.recorder.barrier();
    pf_mm(
        ctx,
        res,
        gate_w,
        "prefill_ffn_gate",
        h,
        inter,
        t,
        quant_off,
        w2,
    )?;
    pf_mm(ctx, res, up_w, "prefill_ffn_up", h, inter, t, quant_off, w3)?;
    res.recorder.barrier();
    pf_swiglu(ctx, res, "prefill_ffn_swiglu", t * inter, w2, w3, w1)?;
    res.recorder.barrier();
    pf_pack_b(ctx, res, "prefill_ffn_dqin", t * inter, w1, quant_off)?;
    res.recorder.barrier();
    pf_mm(
        ctx,
        res,
        down_w,
        "prefill_ffn_down",
        inter,
        h,
        t,
        quant_off,
        w2,
    )?;
    res.recorder.barrier();
    pf_add(ctx, res, "prefill_ffn_mlp_add", t * h, w0, w2, hid_off)?;
    res.gemv_count += 3;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Record helpers. Each is the `T`-row sibling of a `forward.rs` `record_*`, with
// the prefill arena bound in place of the one-token decode arena.
// ─────────────────────────────────────────────────────────────────────────────

/// Record a dispatch whose bindings all live inside the prefill arena.
/// `ranges` are `(byte offset, byte length)` pairs in binding order; the
/// descriptor ring is picked by binding count (2 = quantize, 3 = mmq / norm /
/// elementwise, 5 = rope).
fn pf_arena_dispatch<'a>(
    ctx: &'a VulkanContext,
    res: &mut DecodeResources<'a>,
    name: &str,
    label: &'static str,
    kernel: Kernel,
    spec: &[(u32, u32)],
    ranges: &[(u64, u64)],
    push: &[u8],
    groups: [u32; 3],
) -> Result<()> {
    let DecodeResources {
        cache,
        recorder,
        prefill,
        ring2,
        ring3,
        ring5,
        ..
    } = res;
    let buf = &prefill.buffer;
    let (pipeline, _) = cache
        .get(ctx, kernel, spec, push.len() as u32, ranges.len())
        .map_err(|e| anyhow!("{name}: build {kernel:?} pipeline: {e}"))?;
    let binds: Vec<(&DeviceBuffer<'_>, u64, u64)> =
        ranges.iter().map(|&(off, len)| (buf, off, len)).collect();
    let ring = match ranges.len() {
        2 => ring2,
        3 => ring3,
        5 => ring5,
        n => bail!("{name}: no prefill descriptor ring for {n} bindings"),
    };
    let set = ring
        .next_updated(&binds)
        .map_err(|e| anyhow!("{name}: bind {kernel:?} ring set: {e}"))?;
    recorder.label_next(label);
    recorder.dispatch_raw(pipeline, set, push, groups);
    Ok(())
}

/// Does this device run the batched GEMM on cooperative matrices?
///
/// The answer must be the SAME for every projection in a chunk, because one
/// [`pf_pack_b`] often feeds several [`pf_mm`] calls (q/k/v share one packed
/// input) and the two routes disagree on the staging format — q8_1_x4 vs f16.
/// A per-call fallback would therefore hand one GEMM the other's bytes and
/// produce silent garbage, so the decision is made once, here, from facts that
/// do not vary with `m`/`n`:
///
///   * the device advertises an `f16 x f16 -> f32` subgroup matrix shape, and
///   * that shape tiles the narrowest warptile ([`MmSpec::choose`]'s
///     divisibility test is independent of the output shape, so a shape that
///     tiles at `n = 32` tiles every projection in the model, and one that
///     does not tiles none of them).
///
/// `ARLE_VK_DISABLE_COOPMAT` forces the `mul_mmq` route, mirroring llama.cpp's
/// `GGML_VK_DISABLE_COOPMAT`. It exists to A/B the two lanes on one binary.
fn coopmat_route(ctx: &VulkanContext) -> Option<CoopmatShape> {
    if std::env::var_os("ARLE_VK_DISABLE_COOPMAT").is_some() {
        return None;
    }
    let shape = ctx.coopmat()?;
    let warp = ctx.subgroup_size().0;
    MmSpec::choose(shape, warp, 32, ctx.max_compute_shared_memory_size()).map(|_| shape)
}

/// Stage a flat `[ne]` f32 activation as the batched GEMM's `B` operand, in
/// whichever format [`coopmat_route`] selected.
///
/// Both formats are token-major over the same `[T][k]` block: `mul_mmq` reads
/// q8_1_x4 with `stride_b = k` (every `k` here is a multiple of 128, so each
/// token's row starts on an x4 group boundary), and the coopmat `mul_mm` reads
/// plain row-major f16 with the same stride. The f16 pack is a straight
/// widthless convert, so `f16_kv_pack` covers it with `rows = 1`.
fn pf_pack_b<'a>(
    ctx: &'a VulkanContext,
    res: &mut DecodeResources<'a>,
    name: &str,
    ne: usize,
    in_off: u64,
    quant_off: u64,
) -> Result<()> {
    if coopmat_route(ctx).is_some() {
        let push = f16_kv_pack_params(ne as u32).to_le_bytes();
        let groups = {
            let d = f16_kv_pack_dispatch_rows(ne as u32, 1);
            [d.x, d.y, d.z]
        };
        return pf_arena_dispatch(
            ctx,
            res,
            name,
            "packf16",
            Kernel::F16KvPack,
            Kernel::F16KvPack.specialization_u32(),
            &[(in_off, (ne * 4) as u64), (quant_off, (ne * 2) as u64)],
            &push,
            groups,
        );
    }
    let push = q8_1_quantize_params(ne as u32).to_le_bytes();
    let groups = {
        let d = q8_1_quantize_dispatch(ne as u32);
        [d.x, d.y, d.z]
    };
    pf_arena_dispatch(
        ctx,
        res,
        name,
        "quant",
        Kernel::QuantizeQ8_1,
        Kernel::QuantizeQ8_1.specialization_u32(),
        &[
            (in_off, (ne * 4) as u64),
            (quant_off, q8_1_x4_bytes(ne) as u64),
        ],
        &push,
        groups,
    )
}

/// The batched projection: `D[T][m] = A[m][k] · Bᵀ[T][k]`. `A` is the
/// packed-quant weight bound whole, `B` the staged activation at `quant_off`
/// (see [`pf_pack_b`]), `D` the f32 output at `dst_off`.
///
/// Two kernels can serve this, chosen by [`coopmat_route`]:
///
/// * `mul_mm` COOPMAT — dequantizes `A` into f16 shared memory and multiplies
///   through the device's matrix cores. Worth a measured **3.32x** on this box
///   (llama.cpp on a dense Qwen3.6-27B-Q8_0: 61.96 t/s vs 18.64 t/s with
///   `GGML_VK_DISABLE_COOPMAT=1`).
/// * `mul_mmq` — the integer-dot fallback, for devices with no matrix cores.
///
/// The push-constant block is byte-identical between the two (`mul_mm.comp:75`
/// and `mul_mmq.comp` declare the same 16 `uint`s, and both take `stride_a` /
/// `stride_b` in *elements*), so [`mmq_params`] serves both. Only the tile
/// geometry and the `B` element type differ.
///
/// Bindings are `[A, B, D]` — 3, so this shares `ring3` with the norm/elementwise
/// family.
#[allow(clippy::too_many_arguments)]
fn pf_mm<'a>(
    ctx: &'a VulkanContext,
    res: &mut DecodeResources<'a>,
    weight: &DeviceTensor<'_>,
    name: &str,
    k: usize,
    m: usize,
    n: usize,
    quant_off: u64,
    dst_off: u64,
) -> Result<()> {
    let mmq = mmq_kernel_for(weight, name)?;
    let shared = ctx.max_compute_shared_memory_size();

    // `b_bytes` must match whatever `pf_pack_b` wrote for this chunk; both
    // branches key off `coopmat_route`, so they cannot disagree.
    let (kernel, spec, groups, b_bytes) = match coopmat_route(ctx) {
        Some(shape) => {
            let kernel = mmq
                .coopmat_variant()
                .ok_or_else(|| anyhow!("{name}: {mmq:?} has no cooperative-matrix variant"))?;
            let warp = ctx.subgroup_size().0;
            let spec = MmSpec::choose(shape, warp, n as u32, shared).ok_or_else(|| {
                anyhow!("{name}: {shape:?} tiles no {kernel:?} warptile within {shared} B")
            })?;
            let d = mm_dispatch(m as u32, n as u32, &spec);
            let groups = [d.x, d.y, d.z];
            (
                kernel,
                spec.specialization_u32().to_vec(),
                groups,
                (n * k * 2) as u64,
            )
        }
        None => {
            let spec = MmqSpec::choose(mmq, m as u32, n as u32, shared).ok_or_else(|| {
                anyhow!("{name}: no {mmq:?} tile fits {shared} B of shared memory")
            })?;
            let d = mmq_dispatch(m as u32, n as u32, &spec);
            let groups = [d.x, d.y, d.z];
            (
                mmq,
                spec.specialization_u32().to_vec(),
                groups,
                q8_1_x4_bytes(n * k) as u64,
            )
        }
    };
    let push = mmq_params(m as u32, n as u32, k as u32).to_le_bytes();
    let dst_bytes = (n * m * std::mem::size_of::<f32>()) as u64;

    let DecodeResources {
        cache,
        recorder,
        prefill,
        ring3,
        ..
    } = res;
    let buf = &prefill.buffer;
    let (pipeline, _) = cache
        .get(ctx, kernel, &spec, push.len() as u32, 3)
        .map_err(|e| anyhow!("{name}: build {kernel:?} pipeline: {e}"))?;
    let set = ring3
        .next_updated(&[
            (&weight.buffer, 0, weight.buffer.len() as u64),
            (buf, quant_off, b_bytes),
            (buf, dst_off, dst_bytes),
        ])
        .map_err(|e| anyhow!("{name}: bind {kernel:?} ring set: {e}"))?;
    recorder.label_next("mmq");
    recorder.dispatch_raw(pipeline, set, &push, groups);
    Ok(())
}

/// One token's F32 `router_gemv` (`y[e] = Σ_c W[e,c]·x[c]`, no sigmoid) inside
/// the prefill buffer. The batched sibling of `forward::record_router_gemv`,
/// differing only in which buffer the in/out offsets index.
///
/// There is no batched form of this kernel and no need for one: its only callers
/// are `ssm_alpha`/`ssm_beta`, `[hidden, 48]` each. 3 bindings → `ring3`.
#[allow(clippy::too_many_arguments)]
fn pf_router_gemv<'a>(
    ctx: &'a VulkanContext,
    res: &mut DecodeResources<'a>,
    w_buf: &DeviceBuffer<'_>,
    name: &str,
    n_out: usize,
    hidden: usize,
    in_off: u64,
    out_off: u64,
) -> Result<()> {
    let spec = Kernel::Qwen36RouterGemv.specialization_u32();
    let push = qwen36_router_gemv_params(n_out as u32, hidden as u32, false).to_le_bytes();
    let groups = {
        let d = qwen36_router_gemv_dispatch(n_out as u32);
        [d.x, d.y, d.z]
    };
    let in_row = (hidden * std::mem::size_of::<f32>()) as u64;
    let out_row = (n_out * std::mem::size_of::<f32>()) as u64;

    let DecodeResources {
        cache,
        recorder,
        prefill,
        ring3,
        ..
    } = res;
    let buf = &prefill.buffer;
    let (pipeline, _) = cache
        .get(ctx, Kernel::Qwen36RouterGemv, spec, push.len() as u32, 3)
        .map_err(|e| anyhow!("{name}: build router_gemv pipeline: {e}"))?;
    let set = ring3
        .next_updated(&[
            (buf, in_off, in_row),
            (w_buf, 0, w_buf.len() as u64),
            (buf, out_off, out_row),
        ])
        .map_err(|e| anyhow!("{name}: bind router_gemv ring set: {e}"))?;
    recorder.label_next("gemv");
    recorder.dispatch_raw(pipeline, set, &push, groups);
    Ok(())
}

/// A resident layer weight of any residency, with its dims validated by the
/// caller. Unlike [`packed_layer_weight`] this does NOT require packed quant —
/// for `ssm_alpha`/`ssm_beta`, which the 27B dequantizes to F32.
fn layer_weight<'w>(
    weights: &'w ResidentWeights<'_>,
    layer: usize,
    suffix: &str,
) -> Result<&'w DeviceTensor<'w>> {
    let name = format!("blk.{layer}.{suffix}.weight");
    weights
        .get(&name)
        .ok_or_else(|| anyhow!("missing weight {name}"))
}

/// The `mul_mmq` kernel for a packed-quant weight's GGUF type — the batched
/// sibling of `forward::gemv_kernel_for`.
fn mmq_kernel_for(weight: &DeviceTensor<'_>, name: &str) -> Result<Kernel> {
    let Residency::KeepQuant(ty) = weight.residency else {
        bail!(
            "{name}: batched GEMM expects a packed-quant weight, got {:?}",
            weight.residency
        );
    };
    Ok(match ty {
        GgmlType::Q4K => Kernel::MmqQ4K,
        GgmlType::Q5K => Kernel::MmqQ5K,
        GgmlType::Q6K => Kernel::MmqQ6K,
        GgmlType::Q8_0 => Kernel::MmqQ8_0,
        other => bail!("{name}: no registered mul_mmq kernel for packed type {other:?}"),
    })
}

/// Multi-row RMSNorm: `nrows` independent rows of `ncols`, row `r` read at
/// `r * src_row_stride` and written PACKED at `r * ncols`. `src_row_stride >
/// ncols` extracts a strided sub-block for free (the `[query|gate]` q-projection
/// half). In-place is safe only when `src_row_stride == ncols`.
#[allow(clippy::too_many_arguments)]
fn pf_rms_norm_rows<'a>(
    ctx: &'a VulkanContext,
    res: &mut DecodeResources<'a>,
    w_buf: &DeviceBuffer<'_>,
    name: &str,
    ncols: usize,
    nrows: usize,
    src_row_stride: usize,
    eps: f32,
    in_off: u64,
    out_off: u64,
) -> Result<()> {
    let push =
        rms_norm_params_rows(ncols as u32, nrows as u32, src_row_stride as u32, eps).to_le_bytes();
    let groups = {
        let d = rms_norm_dispatch_rows(nrows as u32);
        [d.x, d.y, d.z]
    };
    let src_bytes = (nrows * src_row_stride * 4) as u64;
    let dst_bytes = (nrows * ncols * 4) as u64;
    let DecodeResources {
        cache,
        recorder,
        prefill,
        ring3,
        ..
    } = res;
    let buf = &prefill.buffer;
    let (pipeline, _) = cache
        .get(
            ctx,
            Kernel::RmsNorm,
            Kernel::RmsNorm.specialization_u32(),
            push.len() as u32,
            3,
        )
        .map_err(|e| anyhow!("{name}: build rms_norm pipeline: {e}"))?;
    let set = ring3
        .next_updated(&[
            (buf, in_off, src_bytes),
            (w_buf, 0, w_buf.len() as u64),
            (buf, out_off, dst_bytes),
        ])
        .map_err(|e| anyhow!("{name}: bind rms_norm ring set: {e}"))?;
    recorder.label_next("norm");
    recorder.dispatch_raw(pipeline, set, &push, groups);
    Ok(())
}

/// Batched NeoX RoPE over a `[T][heads][head_dim]` block, applying `pos[t]` to
/// every head of token `t` from the `[T]` i32 buffer at `pos_off`.
#[allow(clippy::too_many_arguments)]
fn pf_rope<'a>(
    ctx: &'a VulkanContext,
    res: &mut DecodeResources<'a>,
    name: &str,
    head_dim: usize,
    rotary_dim: usize,
    heads: usize,
    tokens: usize,
    theta: f32,
    pos_off: u64,
    in_off: u64,
    out_off: u64,
) -> Result<()> {
    let push = rope_neox_params_batched(
        head_dim as u32,
        rotary_dim as u32,
        heads as u32,
        tokens as u32,
        theta,
    )
    .to_le_bytes();
    let groups = {
        let d = rope_neox_dispatch_batched(rotary_dim as u32, heads as u32, tokens as u32);
        [d.x, d.y, d.z]
    };
    let block = (tokens * heads * head_dim * 4) as u64;
    let pos_bytes = (tokens * 4) as u64;
    let fuse0 = (res.prefill.fuse0.offset, res.prefill.fuse0.len as u64);
    pf_arena_dispatch(
        ctx,
        res,
        name,
        "rope",
        Kernel::RopeNeox,
        Kernel::RopeNeox.specialization_u32(),
        &[
            (in_off, block),             // 0: X input
            (pos_off, pos_bytes),        // 1: Y positions (i32 per token)
            (fuse0.0, fuse0.1),          // 2: Z freq factors (has_ff = 0, unread)
            (out_off, block),            // 3: D output
            (pos_off, pos_bytes.max(8)), // 4: I indices (set_rows_stride = 0, unread)
        ],
        &push,
        groups,
    )
}

/// Pack `rows` f32 head rows into the f16 KV cache starting at `pos`. The source
/// rows are strided (`src_stride` f32 elements apart — the `[token][kv_head]`
/// interleave); the destinations are contiguous in the cache plane.
#[allow(clippy::too_many_arguments)]
fn pf_kv_pack<'a>(
    ctx: &'a VulkanContext,
    res: &mut DecodeResources<'a>,
    name: &str,
    head_dim: usize,
    rows: usize,
    src_stride: usize,
    src_off: u64,
    full_idx: usize,
    kv_head: usize,
    pos: usize,
    is_v: bool,
) -> Result<()> {
    let push = f16_kv_pack_params_rows(
        head_dim as u32,
        rows as u32,
        src_stride as u32,
        head_dim as u32,
    )
    .to_le_bytes();
    let groups = {
        let d = f16_kv_pack_dispatch_rows(head_dim as u32, rows as u32);
        [d.x, d.y, d.z]
    };
    // The source spans `rows` strided rows; the last row ends at
    // `(rows-1)*src_stride + head_dim`.
    let src_bytes = (((rows - 1) * src_stride + head_dim) * 4) as u64;
    let DecodeResources {
        cache,
        recorder,
        prefill,
        ring2,
        kv_cache,
        ..
    } = res;
    let (dst_off, row_bytes) = kv_cache.row_dst(full_idx, kv_head, pos, is_v);
    let dst_bytes = row_bytes * rows as u64;
    let buf = &prefill.buffer;
    let kv_buf = &kv_cache.buffer;
    let (pipeline, _) = cache
        .get(
            ctx,
            Kernel::F16KvPack,
            Kernel::F16KvPack.specialization_u32(),
            push.len() as u32,
            2,
        )
        .map_err(|e| anyhow!("{name}: build f16_kv_pack pipeline: {e}"))?;
    let set = ring2
        .next_updated(&[(buf, src_off, src_bytes), (kv_buf, dst_off, dst_bytes)])
        .map_err(|e| anyhow!("{name}: bind f16_kv_pack ring set: {e}"))?;
    recorder.label_next("kvpack");
    recorder.dispatch_raw(pipeline, set, &push, groups);
    Ok(())
}

/// ONE causal-masked flash-attention dispatch for every (token, query head) of a
/// chunk. Q is `[T][n_q_heads][head_dim]`; K/V bind the layer's whole
/// `[n_kv_heads][max_seq][head_dim]` f16 cache slab (`nb12 = plane_bytes` selects
/// the GQA kv head from `ik2 = iq2 / (n_q_heads / n_kv_heads)`); the output is
/// `[T][n_q_heads][head_dim]`, i.e. the o-projection's input layout.
#[allow(clippy::too_many_arguments)]
fn pf_flash<'a>(
    ctx: &'a VulkanContext,
    res: &mut DecodeResources<'a>,
    name: &str,
    head_dim: usize,
    n: usize,
    kv_len: usize,
    nq: usize,
    nkv: usize,
    full_idx: usize,
    scale: f32,
    q_off: u64,
    mask_off: u64,
    o_off: u64,
) -> Result<()> {
    let spec = FlashAttentionSpec::f32_f16_masked(head_dim as u32, head_dim as u32);
    let plane_bytes = res.kv_cache.plane_bytes;
    let push = flash_attn_params_batched(
        head_dim as u32,
        head_dim as u32,
        n as u32,
        kv_len as u32,
        nq as u32,
        nkv as u32,
        plane_bytes as u32,
        plane_bytes as u32,
        scale,
    )
    .to_le_bytes();
    let groups = {
        let d = flash_attn_dispatch_batched(n as u32, nq as u32);
        [d.x, d.y, d.z]
    };
    let qo_bytes = (n * nq * head_dim * 4) as u64;
    let slab_bytes = plane_bytes * nkv as u64;
    let mask_bytes = (n * kv_len * std::mem::size_of::<u16>()) as u64;
    let fuse0 = (res.prefill.fuse0.offset, res.prefill.fuse0.len as u64);
    let fuse1 = (res.prefill.fuse1.offset, res.prefill.fuse1.len as u64);
    let DecodeResources {
        cache,
        recorder,
        prefill,
        ring7,
        kv_cache,
        ..
    } = res;
    let k_base = kv_cache.k_plane_off(full_idx, 0);
    let v_base = kv_cache.v_plane_off(full_idx, 0);
    let buf = &prefill.buffer;
    let kv_buf = &kv_cache.buffer;
    let (pipeline, _) = cache
        .get(
            ctx,
            Kernel::FlashAttn,
            spec.specialization_u32(),
            push.len() as u32,
            7,
        )
        .map_err(|e| anyhow!("{name}: build flash_attn pipeline: {e}"))?;
    let set = ring7
        .next_updated(&[
            (buf, q_off, qo_bytes),         // 0: Q f32 [T][nq][hd]
            (kv_buf, k_base, slab_bytes),   // 1: K f16 [nkv][max_seq][hd]
            (kv_buf, v_base, slab_bytes),   // 2: V f16 (same shape)
            (buf, mask_off, mask_bytes),    // 3: M causal mask f16 [T][KV]
            (buf, fuse1.0, fuse1.1.max(8)), // 4: S sinks (dummy)
            (buf, o_off, qo_bytes),         // 5: O f32 [T][nq][hd]
            (buf, fuse0.0, fuse0.1.max(8)), // 6: MO mask_opt (unread: USE_MASK_OPT off)
        ])
        .map_err(|e| anyhow!("{name}: bind flash_attn ring set: {e}"))?;
    recorder.label_next("flash");
    recorder.dispatch_raw(pipeline, set, &push, groups);
    Ok(())
}

/// Strided sigmoid gate: `out[i] = sigmoid(gate[(i/inner)*gate_stride +
/// gate_elem_off + i%inner]) * val[i]`.
#[allow(clippy::too_many_arguments)]
fn pf_sigmoid_mul<'a>(
    ctx: &'a VulkanContext,
    res: &mut DecodeResources<'a>,
    name: &str,
    n: usize,
    inner: usize,
    gate_stride: usize,
    gate_elem_off: usize,
    gate_base: u64,
    val_off: u64,
    out_off: u64,
) -> Result<()> {
    let push = sigmoid_mul_params_strided(
        n as u32,
        inner as u32,
        gate_stride as u32,
        gate_elem_off as u32,
    )
    .to_le_bytes();
    let groups = {
        let d = sigmoid_mul_dispatch(n as u32);
        [d.x, d.y, d.z]
    };
    let rows = n / inner.max(1);
    let gate_bytes = (rows * gate_stride * 4) as u64;
    let val_bytes = (n * 4) as u64;
    pf_arena_dispatch(
        ctx,
        res,
        name,
        "sigmoid",
        Kernel::SigmoidMul,
        Kernel::SigmoidMul.specialization_u32(),
        &[
            (gate_base, gate_bytes),
            (val_off, val_bytes),
            (out_off, val_bytes),
        ],
        &push,
        groups,
    )
}

/// Flat SwiGLU over `n` elements (`n = T * width`; the op is elementwise, so the
/// token boundaries are irrelevant).
fn pf_swiglu<'a>(
    ctx: &'a VulkanContext,
    res: &mut DecodeResources<'a>,
    name: &str,
    n: usize,
    gate_off: u64,
    up_off: u64,
    out_off: u64,
) -> Result<()> {
    let push = swiglu_params(n as u32).to_le_bytes();
    let groups = {
        let d = swiglu_dispatch(n as u32);
        [d.x, d.y, d.z]
    };
    let row = (n * 4) as u64;
    pf_arena_dispatch(
        ctx,
        res,
        name,
        "swiglu",
        Kernel::SwiGlu,
        Kernel::SwiGlu.specialization_u32(),
        &[(gate_off, row), (up_off, row), (out_off, row)],
        &push,
        groups,
    )
}

/// Flat residual add over `n` elements.
fn pf_add<'a>(
    ctx: &'a VulkanContext,
    res: &mut DecodeResources<'a>,
    name: &str,
    n: usize,
    a_off: u64,
    b_off: u64,
    out_off: u64,
) -> Result<()> {
    let push = add_params(n as u32).to_le_bytes();
    let groups = {
        let d = add_dispatch(n as u32);
        [d.x, d.y, d.z]
    };
    let row = (n * 4) as u64;
    pf_arena_dispatch(
        ctx,
        res,
        name,
        "add",
        Kernel::Add,
        Kernel::Add.specialization_u32(),
        &[(a_off, row), (b_off, row), (out_off, row)],
        &push,
        groups,
    )
}
