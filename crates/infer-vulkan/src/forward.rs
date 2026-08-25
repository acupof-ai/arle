//! Numeric whole-token forward for the **Qwen3.5/3.6** family (`arch =
//! qwen35` / `qwen36`) on the Vulkan device — both the dense-FFN variant and
//! the MoE variant (qwen36 router / top-k / per-expert fused `mul_mat_vec_id`
//! + weighted accumulate).
//!
//! Design: the **default path records every op on-device**. The heavy matmuls —
//! every projection, the dense FFN or the fused MoE experts, and the LM head —
//! run on the GPU through the PROVEN quantized GEMV (`q8_0_gemv_with_params` fed
//! by `q8_1_quantize`, validated in `vulkan-kernels/tests/device_gemv.rs`). The
//! lighter per-element / reduction ops (RMSNorm, RoPE, attention softmax, the
//! per-head sigmoid gate, SwiGLU, depthwise conv1d, the gated-delta recurrence,
//! MoE router / top-k / weighted accumulate, residual add) also run as device
//! dispatches, threaded through device-resident scratch slots so a whole token
//! stays on-device. The numerics are transcribed line-for-line from the
//! authoritative CUDA reference (`crates/infer-cuda/src/qwen35.rs` +
//! `crates/cuda-kernels/csrc/{attention/prefill_attention_hd256.cu,
//! misc/gated_delta_rule.cu, misc/conv1d.cu, misc/norm.cu}`). A host f32
//! fallback remains for ops whose device kernel is absent, but it is not the
//! default lane.
//!
//! Perf-parity Steps 3+4: the GEMV
//! path no longer allocates scratch + round-trips per op. A **[`DeviceArena`]**
//! of named (offset,len) sub-ranges (input activation, q8_1 quantized
//! activation, f32 dst, the two fuse dummies) is allocated **once** on the model
//! and bound through offset-aware ranged descriptors. The quantize+GEMV pair is
//! recorded into a persistent **`CommandRecorder`** (one submit, one fence, no
//! per-op `queue_wait_idle`) using pipelines built **once** by a persistent
//! **`KernelCache`** (no per-dispatch SPIR-V re-read / pipeline rebuild). The
//! host elementwise/norm/attention math stays AS-IS between GEMVs (Step 5 ports
//! it to device kernels).
//!
//! Numeric contract distilled from the reference (all verified against the real
//! 27B GGUF dims):
//!   hidden=5120, layers=64 (LLL F interleave from `config.layer_types`),
//!   heads=24, kv_heads=4, head_dim=256, rope_theta=1e7, rotary_dim=256,
//!   rms_eps=1e-6, dense FFN intermediate=17408.
//!   FULL layer: attn_q `[5120,12288]` (out=24*256*2 = [query|gate] per head),
//!   attn_k/v `[5120,1024]`, attn_q_norm/attn_k_norm `[256]` f32 (PLAIN weight —
//!   see `rms_norm_weight`), attn_output `[6144,5120]`. Per-head q/k RMSNorm → NeoX
//!   RoPE → causal SDPA (scale 1/sqrt(head_dim)) → per-head `*sigmoid(gate)` →
//!   o_proj.
//!   LINEAR layer: attn_qkv `[5120,10240]` ([q=2048|k=2048|v=6144]), attn_gate
//!   (z) `[5120,6144]`, ssm_alpha (a) / ssm_beta (b) `[5120,48]`, ssm_a (A_log)
//!   `[48]` f32, ssm_dt.bias `[48]` f32, ssm_conv1d `[4,10240]` f32
//!   (per-channel depthwise, SiLU), ssm_norm `[128]` f32 (gated RMSNorm, PLAIN
//!   weight, ×silu(z)), ssm_out `[6144,5120]`.
//!   All RMSNorms (input / post-attention / final / q / k / gated output) use
//!   the PLAIN GGUF weight `x*inv_rms*w` (the GGUF converter already folded the
//!   `+1` of the HF zero-centered scale — see `rms_norm_weight`).
//!
//! State: this lane runs the **uncached full-prefix** path for a single slot —
//! a forward of one token at `start_pos` recomputes nothing it does not own.
//! Per-slot KV cache (full layers) and recurrent + conv state (linear layers)
//! are owned by [`crate::model_qwen35::VulkanQwen35Model`] and advanced in place
//! here, matching the reference's owned-state contract.

use anyhow::{Context, Result, anyhow, bail};

use qwen35_spec::{LayerType, Qwen35Config};
use vulkan_kernels::{
    BLOCK_Q8_1_BYTES, FlashAttentionSpec, Kernel, KernelCache, KernelParams,
    Q8_1_X4_VALUES_PER_GROUP, add_dispatch, add_params, f16_kv_pack_dispatch, f16_kv_pack_params,
    flash_attn_dispatch, flash_attn_params, gemv_dispatch, gemv_id_dispatch, gemv_id_params,
    gemv_params, q8_1_quantize_dispatch, q8_1_quantize_params, qwen35_gated_delta_net_dispatch,
    qwen35_gated_delta_net_params, qwen35_ssm_conv_dispatch, qwen35_ssm_conv_params,
    qwen36_moe_weighted_accum_dispatch, qwen36_moe_weighted_accum_params,
    qwen36_router_gemv_dispatch, qwen36_router_gemv_params, qwen36_router_topk_dispatch,
    qwen36_router_topk_params, rms_norm_dispatch, rms_norm_params, rope_neox_dispatch,
    rope_neox_params, sigmoid_mul_dispatch, sigmoid_mul_params, swiglu_dispatch, swiglu_params,
};
use vulkan_sys::{
    CommandRecorder, DescriptorSetLayout, DescriptorSetRing, DeviceBuffer, VulkanContext,
};

use crate::loader::Residency;
use crate::loader::upload::{DeviceTensor, ResidentWeights};
use infer_gguf::gguf::GgmlType;

/// Per-slot recurrent / cache state carried across forward calls for one
/// sequence. Sized from the config's local (single-GPU) widths.
pub struct Qwen35ForwardState {
    // NOTE: the full-attention K/V cache is now **device-resident f16**
    // ([`DeviceKvCache`], owned by [`DecodeResources`]); RoPE is applied to K on
    // the device at write time. The host K/V Vecs are gone (the device flash-attn
    // reads the cache directly). Only the gated-delta recurrent + conv state stay
    // host-side (that is the NEXT chunk to port).
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
            gdr_state: vec![vec![0.0f32; gdr_len]; num_linear],
            conv_ring: vec![vec![0.0f32; conv_len]; num_linear],
            seq_len: 0,
        }
    }

    /// Reset for a fresh generation (zeros the linear recurrent + conv state).
    /// The device KV cache is positional (overwritten at each `pos`), so a fresh
    /// generation starting at pos 0 naturally reuses its planes — no explicit
    /// clear needed.
    pub fn reset(&mut self) {
        for s in &mut self.gdr_state {
            s.iter_mut().for_each(|v| *v = 0.0);
        }
        for r in &mut self.conv_ring {
            r.iter_mut().for_each(|v| *v = 0.0);
        }
        self.seq_len = 0;
    }
}

/// Bytes for the q8_1_x4 quantized form of an `ncols`-element activation vector
/// (the shader groups 128 values into one x4 super-block of 4×36 B).
pub(crate) fn q8_1_x4_bytes(ncols: usize) -> usize {
    let num_x4 = ncols.div_ceil(Q8_1_X4_VALUES_PER_GROUP as usize);
    num_x4 * 4 * BLOCK_Q8_1_BYTES
}

/// Round `n` up to the next multiple of `align` (a power-of-two device limit).
pub(crate) fn align_up(n: usize, align: usize) -> usize {
    if align == 0 {
        return n;
    }
    n.div_ceil(align) * align
}

/// Zero a UMA device buffer's first `len` bytes (host write through the mapped
/// pointer — no staging). Used to reset the resident gated-delta + conv state.
fn zero_device_buffer(buf: &mut DeviceBuffer<'_>, len: usize) -> Result<()> {
    let zeros = vec![0u8; len];
    buf.copy_from_host(&zeros)
        .map_err(|e| anyhow!("zero device buffer ({len} B): {e}"))
}

/// Decode-time cap on the per-slot full-attention KV cache (positions). The full
/// 32k context would need ~17 GB of f16 K+V across the 16 full layers; the decode
/// path (prefill + a generation run) stays well under this, and a forward that
/// exceeds it bails loud rather than corrupting. A production path would page the
/// KV; this single-slot lane sizes a fixed window.
pub const KV_CACHE_MAX_SEQ: usize = 8192;

/// The per-slot full-attention KV cache, **device-resident f16**. One wide UMA
/// buffer laid out as `[K block | V block]`, each block indexed
/// `[full_layer, kv_head, pos, head_dim]` (f16). The flash-attn kernel reads a
/// query head's K/V directly from its `(layer, kv_head)` `[max_seq, head_dim]`
/// sub-range (row stride = head_dim elements, `nb11 = head_dim`), so a per-head
/// dispatch needs only the head's base offset. RoPE is applied to K at write
/// time (matching the host cache contract); V is stored raw.
pub struct DeviceKvCache<'a> {
    pub(crate) buffer: DeviceBuffer<'a>,
    /// Byte offset of the V block (K block starts at 0).
    v_base: u64,
    head_dim: usize,
    /// Bytes per `(layer, kv_head)` `[max_seq, head_dim]` f16 plane.
    pub(crate) plane_bytes: u64,
    /// Bytes per `(layer)` slab (`n_kv_heads` planes).
    layer_bytes: u64,
}

impl<'a> DeviceKvCache<'a> {
    /// Allocate the f16 K+V cache for `n_full_layers` full-attention layers.
    pub fn new(
        ctx: &'a VulkanContext,
        n_full_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> Result<Self> {
        let f16 = std::mem::size_of::<u16>();
        let plane_bytes = (max_seq * head_dim * f16) as u64;
        let layer_bytes = plane_bytes * n_kv_heads as u64;
        let block_bytes = layer_bytes * n_full_layers.max(1) as u64;
        let total = (block_bytes * 2) as usize; // K + V
        let buffer = DeviceBuffer::alloc_uma(ctx, total.max(4))
            .map_err(|e| anyhow!("alloc device KV cache ({total} B): {e}"))?;
        Ok(Self {
            buffer,
            v_base: block_bytes,
            head_dim,
            plane_bytes,
            layer_bytes,
        })
    }

    /// Byte offset of the K plane base for `(full_idx, kv_head)`.
    pub(crate) fn k_plane_off(&self, full_idx: usize, kv_head: usize) -> u64 {
        full_idx as u64 * self.layer_bytes + kv_head as u64 * self.plane_bytes
    }

    /// Byte offset of the V plane base for `(full_idx, kv_head)`.
    pub(crate) fn v_plane_off(&self, full_idx: usize, kv_head: usize) -> u64 {
        self.v_base + self.k_plane_off(full_idx, kv_head)
    }

    /// Byte offset of one position row `[head_dim]` f16 inside a plane.
    fn pos_row_off(&self, plane_off: u64, pos: usize) -> u64 {
        plane_off + (pos * self.head_dim * std::mem::size_of::<u16>()) as u64
    }

    /// Byte offset + byte length of one `(full_idx, kv_head, pos)` head row
    /// `[head_dim]` f16 in the K (or V) block — the destination a device
    /// `f16_kv_pack` dispatch binds to write this token's roped K / raw V without
    /// a host readback — the same `[K block | V block]` × `[layer, kv_head, pos,
    /// head_dim]` address math the host pack used, exposed so the on-device pack
    /// lands the row at the identical cache position.
    pub(crate) fn row_dst(
        &self,
        full_idx: usize,
        kv_head: usize,
        pos: usize,
        is_v: bool,
    ) -> (u64, u64) {
        let plane = if is_v {
            self.v_plane_off(full_idx, kv_head)
        } else {
            self.k_plane_off(full_idx, kv_head)
        };
        let off = self.pos_row_off(plane, pos);
        (off, (self.head_dim * std::mem::size_of::<u16>()) as u64)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DeviceArena — the per-GEMV scratch as named, offset-aligned sub-ranges of ONE
// wide UMA buffer, allocated once on the model (perf-parity Step 3).
// ─────────────────────────────────────────────────────────────────────────────

/// One `(offset, len)` named slot inside the arena's backing buffer.
#[derive(Clone, Copy)]
pub(crate) struct Slot {
    pub(crate) offset: u64,
    pub(crate) len: usize,
}

/// A single `DeviceLocal|HostVisible|HostCoherent` (UMA) buffer holding all the
/// per-GEMV scratch slots, each aligned to `minStorageBufferOffsetAlignment` so
/// every slot can be bound directly as a storage-buffer descriptor sub-range.
///
/// Allocated **once** on [`crate::model_qwen35::VulkanQwen35Model`] and sized to
/// the widest GEMV the forward will run. This replaces the deleted `GemvScratch`
/// and its per-call `DeviceBuffer::alloc`/`copy_to_host` churn: the host reads
/// the result back from `dst`, while the quantize/GEMV dispatches read/write the
/// slots in place.
///
/// Perf-parity Step 5b adds named **elementwise work slots** (`work0..work2`,
/// each `max_cols` f32 wide) so the device RMSNorm / SwiGLU / residual-Add can
/// read/write the arena directly. The dense FFN sub-sequence (norm → gate/up →
/// swiglu → down → residual add) chains through these slots **device-resident**:
/// only the FFN's host inputs land once and its single output reads back once,
/// killing the per-GEMV host round-trip that dominated decode.
pub struct DeviceArena<'a> {
    buffer: DeviceBuffer<'a>,
    /// q8_1_x4 quantized activation.
    quant: Slot,
    /// f32 destination rows (widest GEMV output rows).
    dst: Slot,
    /// 4-byte fuse-dummy bindings (3/4 of the GEMV; binding 3 of `add`); only
    /// read when `fusion_flags != 0` / the RMS-add reduction is enabled (never).
    fuse0: Slot,
    fuse1: Slot,
    /// General-purpose f32 elementwise work slots (each `max_cols` wide). The
    /// device RMSNorm/SwiGLU/Add and the fused FFN thread their intermediate
    /// activations through these without a host hop. Four slots is the fused
    /// dense FFN's peak live set: {post_sum, mlp_in, gate, up} coexist after the
    /// up-proj GEMV (before swiglu frees mlp_in/gate).
    work: [Slot; 4],
    // ── Fused MoE (`mul_mat_vec_id`) slots: the per-layer 8×3 per-expert GEMVs
    // collapse into 3 fused dispatches, each producing ALL top-k experts'
    // gate/up/down at once into back-to-back per-expert blocks. Zero-sized on a
    // dense-only model. ──
    /// q8_1_x4 of the shared MoE input `mlp_in` (`hidden`-wide; gate+up read it).
    moe_in_quant: Slot,
    /// f32 `[top_k * moe_inter]` — every selected expert's gate projection.
    moe_gate: Slot,
    /// f32 `[top_k * moe_inter]` — every selected expert's up projection (then
    /// reused for the swiglu output `act_all`).
    moe_up: Slot,
    /// q8_1_x4 of `act_all` (`top_k * moe_inter`-wide; the down dispatch reads it
    /// per-expert via `ne11 = top_k`, `stride_b = moe_inter`).
    moe_act_quant: Slot,
    /// f32 `[top_k * hidden]` — every selected expert's down projection.
    moe_down: Slot,
    /// i32 `[top_k]` — the routed expert-id list (binding 5 of `mul_mat_vec_id`).
    moe_ids: Slot,
    /// f32 `[n_expert]` — router logits (resident MoE: `qwen36_router_gemv` out,
    /// `qwen36_router_topk` in). Zero-sized on a dense-only model.
    moe_logits: Slot,
    /// f32 `[top_k]` — the device routing weights (`qwen36_router_topk` out,
    /// `qwen36_moe_weighted_accum` weights). Resident MoE keeps routing on-device.
    moe_weights: Slot,
    /// f32 `[1]` — the shared-expert sigmoid gate scalar (resident MoE:
    /// `qwen36_router_gemv` sigmoid out, shared `qwen36_moe_weighted_accum` weight).
    moe_shgate: Slot,
    /// MoE fused caps: top-k experts and the widest expert intermediate.
    moe_top_k: usize,
    moe_inter_cap: usize,
    // ── Full-attention slots (on-device norm/rope/flash/gate). The q-projection
    // is `[query|gate]` per head (`2*q_dim` wide); the gate half stays here for
    // the device sigmoid-mul. The per-head q/k RMSNorm + NeoX RoPE write into
    // `attn_q`/`attn_k`; flash-attn writes per-head into `attn_out`, which is
    // then gated in place. K is read back (post-rope) to land f16 in the KV
    // cache; the flash output is read back once for the o-proj GEMV. ──
    /// f32 `[2*q_dim]` — the gated q-projection `[query|gate]` per head.
    attn_qkv: Slot,
    /// f32 `[q_dim]` — per-head normed+roped query.
    attn_q: Slot,
    /// f32 `[kv_dim]` — per-head normed+roped key (then f16->cache).
    attn_k: Slot,
    /// f32 `[q_dim]` — per-head flash-attn output, gated in place.
    attn_out: Slot,
    /// 8 bytes — the RoPE position (i32 at offset 0; also serves as the unread
    /// `uvec2` set-rows-indices dummy for binding 4).
    attn_pos: Slot,
    // ── Linear-attention (gated-delta) slots. The conv1d + recurrent state
    // update run device-resident against persistent state buffers; these slots
    // hold the per-token activations the two serial shaders read/write. The raw
    // in-proj `qkv` lands in `lin_xseq`; the depthwise conv writes
    // `lin_qkv_conv`; the gated-delta reads it (+ the `lin_a`/`lin_b`
    // projections) and writes `lin_out`. ──
    /// f32 `[qkv_dim]` — the raw in-proj `[q|k|v]` (conv input `XSeq`).
    lin_xseq: Slot,
    /// f32 `[qkv_dim]` — the post-conv `silu(conv)` `[q|k|v]` (conv `OutSeq`,
    /// gated-delta `Qkv` input).
    lin_qkv_conv: Slot,
    /// f32 `[nv]` — the `ssm_alpha` projection (gated-delta `AProj`).
    lin_a: Slot,
    /// f32 `[nv]` — the `ssm_beta` projection (gated-delta `BProj`).
    lin_b: Slot,
    /// f32 `[v_dim_total]` — the gated-delta recurrence output (`Output`),
    /// read back for the gated RMSNorm + out-proj.
    lin_out: Slot,
    // ── Residual-resident layer-loop slots (Step 3). The hidden state lives in
    // `hid` device-resident across the WHOLE layer (input-norm → attention →
    // residual-add → post-norm → FFN → residual-add): the embedding uploads it
    // once, every sub-op reads/writes it on-device, and only the final `[vocab]`
    // logits read back. The attention block writes its o_proj output into
    // `attn_resident` so the FFN's post-add consumes it without a host hop. ──
    /// f32 `[hidden]` — the resident hidden / residual stream.
    hid: Slot,
    /// f32 `[hidden]` — the attention block's o_proj output (consumed by the
    /// FFN's post-attention residual add, device-resident).
    attn_resident: Slot,
    /// f32 `[hidden]` — the device-resident input/post RMSNorm output (the
    /// attention block / FFN read it as their `normed` input).
    normed_resident: Slot,
    /// Pre-sized caps so a GEMV can bail loud rather than silently corrupt.
    max_cols: usize,
    max_rows: usize,
}

impl<'a> DeviceArena<'a> {
    /// Allocate the arena sized to the widest GEMV (`max_cols` input width,
    /// `max_rows` output rows), aligning every slot to the device's storage
    /// buffer offset granularity. `config` sizes the fused-MoE slots (zero on a
    /// dense-only model).
    pub fn new(
        ctx: &'a VulkanContext,
        config: &Qwen35Config,
        max_cols: usize,
        max_rows: usize,
    ) -> Result<Self> {
        let align = ctx.min_storage_buffer_offset_alignment().max(1) as usize;
        let quant_len = q8_1_x4_bytes(max_cols);
        let dst_len = max_rows * std::mem::size_of::<f32>();
        let fuse_len = 4usize;
        // Elementwise work slots: the widest single-vector activation a fused FFN
        // op touches is `max_cols` f32 (the FFN intermediate). dst can also hold a
        // hidden-wide vector, but max_cols already covers hidden.
        let work_len = max_cols * std::mem::size_of::<f32>();

        // Fused-MoE slots: the 3 fused `mul_mat_vec_id` dispatches produce ALL
        // top-k experts' gate/up/down at once. The shared-expert dense path reuses
        // these too (its 1-expert "id list" is a single dispatch), so cap the
        // intermediate at the wider of the routed / shared expert width.
        let moe_top_k = config.num_experts_per_tok;
        let moe_inter_cap = config
            .moe_intermediate_size
            .max(config.shared_expert_intermediate_size);
        let h = config.hidden_size;
        // gate/up/act are `[top_k * moe_inter]`; down is `[top_k * hidden]`.
        let moe_gate_len = moe_top_k * moe_inter_cap * std::mem::size_of::<f32>();
        let moe_down_len = moe_top_k * h * std::mem::size_of::<f32>();
        // q8_1 of the hidden-wide shared input, and of the `top_k*moe_inter` act.
        let moe_in_quant_len = q8_1_x4_bytes(h);
        let moe_act_quant_len = q8_1_x4_bytes(moe_top_k * moe_inter_cap);

        // Full-attention slot widths (f32).
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let attn_qkv_len = 2 * q_dim * std::mem::size_of::<f32>();
        let attn_q_len = q_dim * std::mem::size_of::<f32>();
        let attn_k_len = kv_dim * std::mem::size_of::<f32>();
        let attn_out_len = q_dim * std::mem::size_of::<f32>();

        // Linear-attention (gated-delta) slot widths (f32). `qkv_dim` =
        // `2*nk*kd + nv*vd`; `lin_out` is `nv*vd`.
        let lin_qkv_dim = 2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim;
        let lin_nv = config.linear_num_value_heads;
        let lin_vout = config.linear_num_value_heads * config.linear_value_head_dim;
        let lin_xseq_len = lin_qkv_dim * std::mem::size_of::<f32>();
        let lin_a_len = lin_nv.max(1) * std::mem::size_of::<f32>();
        let lin_out_len = lin_vout.max(1) * std::mem::size_of::<f32>();

        let mut cursor = 0u64;
        let mut place = |len: usize| -> Slot {
            let offset = cursor;
            cursor += align_up(len, align) as u64;
            Slot { offset, len }
        };
        let quant = place(quant_len);
        let dst = place(dst_len);
        let fuse0 = place(fuse_len);
        let fuse1 = place(fuse_len);
        let work = [
            place(work_len),
            place(work_len),
            place(work_len),
            place(work_len),
        ];
        let moe_in_quant = place(moe_in_quant_len);
        let moe_gate = place(moe_gate_len);
        let moe_up = place(moe_gate_len);
        let moe_act_quant = place(moe_act_quant_len);
        let moe_down = place(moe_down_len);
        let moe_ids = place(moe_top_k.max(1) * std::mem::size_of::<i32>());
        // Resident-MoE routing slots: router logits [n_expert], device routing
        // weights [top_k], shared-expert sigmoid gate [1]. Zero-ish on dense.
        let moe_logits = place(config.num_experts.max(1) * std::mem::size_of::<f32>());
        let moe_weights = place(moe_top_k.max(1) * std::mem::size_of::<f32>());
        let moe_shgate = place(std::mem::size_of::<f32>());
        let attn_qkv = place(attn_qkv_len);
        let attn_q = place(attn_q_len);
        let attn_k = place(attn_k_len);
        let attn_out = place(attn_out_len);
        let attn_pos = place(8);
        let lin_xseq = place(lin_xseq_len);
        let lin_qkv_conv = place(lin_xseq_len);
        let lin_a = place(lin_a_len);
        let lin_b = place(lin_a_len);
        let lin_out = place(lin_out_len);
        // Residual-resident layer-loop slots (Step 3), each hidden-wide f32.
        let hid_len = config.hidden_size * std::mem::size_of::<f32>();
        let hid = place(hid_len);
        let attn_resident = place(hid_len);
        let normed_resident = place(hid_len);
        let total = cursor as usize;

        let buffer = DeviceBuffer::alloc_uma(ctx, total)
            .map_err(|e| anyhow!("alloc GEMV device arena ({total} B): {e}"))?;
        Ok(Self {
            buffer,
            quant,
            dst,
            fuse0,
            fuse1,
            work,
            moe_in_quant,
            moe_gate,
            moe_up,
            moe_act_quant,
            moe_down,
            moe_ids,
            moe_logits,
            moe_weights,
            moe_shgate,
            moe_top_k,
            moe_inter_cap,
            attn_qkv,
            attn_q,
            attn_k,
            attn_out,
            attn_pos,
            lin_xseq,
            lin_qkv_conv,
            lin_a,
            lin_b,
            lin_out,
            hid,
            attn_resident,
            normed_resident,
            max_cols,
            max_rows,
        })
    }

    /// Write `data` (f32) into the arena at byte `off` (UMA, no staging).
    fn write_at(&mut self, off: u64, data: &[f32]) -> Result<()> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.buffer
            .copy_from_host_at(off, &bytes)
            .map_err(|e| anyhow!("write arena @{off}: {e}"))
    }

    /// Read `len` f32 from the arena at byte `off` (UMA, no staging).
    fn read_at(&self, off: u64, len: usize) -> Result<Vec<f32>> {
        let mut bytes = vec![0u8; len * std::mem::size_of::<f32>()];
        self.buffer
            .copy_to_host_at(off, &mut bytes[..])
            .map_err(|e| anyhow!("read arena @{off}: {e}"))?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    /// Map+write `data` (f32) into work slot `i` (UMA, no staging).
    fn write_work(&mut self, i: usize, data: &[f32]) -> Result<()> {
        debug_assert!(data.len() <= self.max_cols);
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let off = self.work[i].offset;
        self.buffer
            .copy_from_host_at(off, &bytes)
            .map_err(|e| anyhow!("write arena work[{i}]: {e}"))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Persistent decode resources: the arena + the compile-once KernelCache + the
// record-many/submit-once CommandRecorder, all owned by the model and threaded
// into the forward. ONE canonical flow — no transient launch path on decode.
// ─────────────────────────────────────────────────────────────────────────────

/// The persistent per-token decode resources, built once in
/// [`crate::model_qwen35::VulkanQwen35Model::load`] and borrowed mutably by each
/// [`forward_token`] call. Bundling them lets the forward thread one `&mut` and
/// keeps the lifetimes (`'a` = the model's `'static` context) in one place.
pub struct DecodeResources<'a> {
    pub arena: DeviceArena<'a>,
    pub cache: KernelCache<'a>,
    pub recorder: CommandRecorder<'a>,
    /// Batched-prefill scratch (a whole token chunk per slot). Separate from the
    /// one-token `arena` so the decode path is untouched.
    pub(crate) prefill: crate::prefill::PrefillArena<'a>,
    /// Persistent storage-buffer descriptor-set layouts, one per binding count
    /// used on the decode path (2 = q8_1 quantize, 3 = rms_norm / swiglu / add,
    /// 5 = GEMV). Built once; the rings allocate their sets against them. Kept
    /// alive (alongside the rings) so the layouts outlive the pipelines that were
    /// built with the same binding-count layout in the `KernelCache`.
    _layouts: Vec<DescriptorSetLayout<'a>>,
    /// Round-robin descriptor-set rings keyed by binding count. Each
    /// [`DescriptorSetRing::next_updated`] only runs `vkUpdateDescriptorSets` on a
    /// pre-allocated set — no per-dispatch `VkDescriptorPool` create/destroy
    /// (perf-parity Step 5a). Reset per token via [`Self::reset_rings`].
    pub(crate) ring2: DescriptorSetRing<'a>,
    pub(crate) ring3: DescriptorSetRing<'a>,
    /// 4-binding ring for the depthwise conv1d ([XSeq, ConvWeight, ConvState,
    /// OutSeq]). One dispatch per linear layer.
    pub(crate) ring4: DescriptorSetRing<'a>,
    pub(crate) ring5: DescriptorSetRing<'a>,
    /// 6-binding ring for the fused MoE `mul_mat_vec_id` ([A,B,D,F0,F1,IDS]).
    pub(crate) ring6: DescriptorSetRing<'a>,
    /// 7-binding ring for flash-attn ([Q,K,V,M,S,O,MO]) AND the gated-delta net
    /// ([Qkv,BProj,AProj,DtBias,ALog,State,Output]). Both record one dispatch per
    /// submit on the linear path; the full-attn records one per query head, so it
    /// is sized to `num_attention_heads`.
    pub(crate) ring7: DescriptorSetRing<'a>,
    /// Per-slot full-attention KV cache (device-resident f16). RoPE is applied to
    /// K at write time; V is stored raw. flash-attn reads each head's plane.
    pub(crate) kv_cache: DeviceKvCache<'a>,
    /// Persistent device-resident gated-delta linear-attention state, one block
    /// per LINEAR layer (the conv ring + the recurrent S matrix). Resident across
    /// tokens, read+written each token by the conv / gated-delta dispatches;
    /// zeroed for a fresh generation via [`Self::reset_linear_state`]. Sized
    /// `[n_linear * qkv_dim * (kernel-1)]` and `[n_linear * nv * kd * vd]` f32.
    pub(crate) lin_conv_state: DeviceBuffer<'a>,
    pub(crate) lin_gdr_state: DeviceBuffer<'a>,
    /// Per-linear-layer byte strides into the two state buffers above.
    pub(crate) lin_conv_stride: u64,
    pub(crate) lin_gdr_stride: u64,
    /// Lightweight per-call accumulators (nanoseconds) so a decode loop can
    /// attribute time between the GPU GEMV submits and the surrounding host
    /// prep/readback. Drained + printed via [`Self::take_profile`].
    gemv_submit_ns: u128,
    gemv_other_ns: u128,
    pub(crate) gemv_count: u64,
}

impl<'a> DecodeResources<'a> {
    pub fn new(ctx: &'a VulkanContext, config: &Qwen35Config) -> Result<Self> {
        let (max_cols, max_rows) = widest_gemv(config);
        let arena = DeviceArena::new(ctx, config, max_cols, max_rows)?;
        let cache = KernelCache::new();
        let recorder =
            CommandRecorder::new(ctx).map_err(|e| anyhow!("create decode CommandRecorder: {e}"))?;

        // One persistent layout + ring per decode binding count. A ring of N sets
        // permits N live dispatches before a slot is reused; `next_updated` wraps via
        // `next % N` (vulkan-sys), so reusing a slot whose dispatch is still in flight
        // SILENTLY aliases its descriptor set and corrupts the residual stream.
        //
        // WHOLE-TOKEN batching (perf-parity Lever B / Step 4): the residual-resident
        // dense loop records EVERY layer into ONE command buffer and `submit_and_wait`s
        // ONCE per token — eliminating the old per-layer fence-wait that idled the GPU
        // 64×/token (~0.29 s of pure submit serialization). So every layer's dispatches
        // are simultaneously in flight at the single submit, and each ring must cover
        // the WHOLE TOKEN, not one layer. Size each ring to its per-layer worst case ×
        // n_layers + headroom for the final norm/LM-head. Descriptor sets are tiny (≤7
        // storage buffers each); ~20k of them is a few MB on the 96 GB box. The MoE path
        // stays host-bridged (per-block submit) so it never needs the whole-token depth,
        // but sharing the (over-sized) rings is harmless.
        let n_heads = config.num_attention_heads;
        let n_kv = config.num_key_value_heads;
        let n_vheads = config.linear_num_value_heads;
        let n_layers = config.layer_types.len();
        // Per-layer worst case per ring (full-layer vs linear-layer, take the max).
        // ring3: full (2*(nq+nkv) q/k+rope-mate norms + nq sigmoid gates + 4 FFN) vs
        // linear (nv rms_norm + 1 swiglu + 1 input-norm + 4 FFN).
        let ring3_layer = (2 * (n_heads + n_kv) + n_heads + 8).max(n_vheads + 8) + 16;
        // ring5 (GEMV + RoPE): full-layer q/k/v/o GEMV (4) + `nq+nkv` rope + 3 FFN.
        let ring5_layer = n_heads + n_kv + 16;
        // ring2 (quantize + f16 pack): q/k/v/o quant + `2*nkv` KV-packs + 3 FFN quant.
        let ring2_layer = 2 * n_kv + 16;
        // ring7 (flash-attn): one dispatch per query head on a full-attention layer.
        let ring7_layer = n_heads + 4;
        // Resident MoE FFN per-layer ring usage (the MoE path is now ALSO
        // residual-resident / whole-token, so its dispatches share the token's
        // single submit): ~10 ring3 (post-add/post-norm/router_gemv/topk/swiglu/
        // weighted-accum ×2/shared-gate/mlp-add), ~3 ring6 (routed gate/up/down
        // mul_mat_vec_id), ~3 ring2 (qin/qact/shared-down quant), ~3 ring5
        // (shared gate/up/down GEMV). Add a generous per-MoE-layer allowance on
        // top of the attention budget (descriptor sets are cheap). ring6 was a
        // FIXED 16 (host-bridged MoE never needed whole-token depth) — it MUST
        // now scale with the routed expert GEMVs or it silently aliases.
        let n_moe = (0..n_layers).filter(|&l| config.is_moe_layer(l)).count();
        // Whole-token depths: per-layer worst case × n_layers + the MoE FFN
        // allowance + slack for the final norm + LM-head GEMV in the last batch.
        let ring3_size = (ring3_layer * n_layers + 12 * n_moe + 32).max(64);
        let ring5_size = (ring5_layer * n_layers + 4 * n_moe + 32).max(48);
        let ring2_size = (ring2_layer * n_layers + 4 * n_moe + 32).max(32);
        let ring6_size = (4 * n_moe + 16).max(16);
        let ring7_size = (ring7_layer * n_layers + 16).max(8);
        // ring4 (depthwise conv1d): one dispatch per LINEAR layer per token.
        let ring4_size = (n_layers + 16).max(16);
        let mk = |binding_count: usize,
                  size: usize|
         -> Result<(DescriptorSetLayout<'a>, DescriptorSetRing<'a>)> {
            let layout = DescriptorSetLayout::storage_buffers(ctx, binding_count)
                .map_err(|e| anyhow!("build descriptor layout ({binding_count} bindings): {e}"))?;
            let ring = DescriptorSetRing::new(ctx, &layout, binding_count, size)
                .map_err(|e| anyhow!("build descriptor ring ({binding_count} bindings): {e}"))?;
            Ok((layout, ring))
        };
        let (l2, ring2) = mk(2, ring2_size)?;
        let (l3, ring3) = mk(3, ring3_size)?;
        let (l4, ring4) = mk(4, ring4_size)?;
        let (l5, ring5) = mk(5, ring5_size)?;
        let (l6, ring6) = mk(6, ring6_size)?;
        let (l7, ring7) = mk(7, ring7_size)?;

        // Device KV cache for the full-attention layers.
        let n_full = config
            .layer_types
            .iter()
            .filter(|&&t| t == LayerType::FullAttention)
            .count();
        let kv_cache = DeviceKvCache::new(ctx, n_full, n_kv, config.head_dim, KV_CACHE_MAX_SEQ)?;

        // Persistent device-resident gated-delta state, one block per linear
        // layer (conv ring + recurrent S matrix). Sized exactly like the host
        // `Qwen35ForwardState` Vecs it replaces, but resident across tokens.
        let n_linear = config.layer_types.len() - n_full;
        let qkv_dim = 2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim;
        let conv_per_layer = qkv_dim * config.linear_conv_kernel_dim.saturating_sub(1);
        let gdr_per_layer = config.linear_num_value_heads
            * config.linear_key_head_dim
            * config.linear_value_head_dim;
        let lin_conv_stride = (conv_per_layer * std::mem::size_of::<f32>()) as u64;
        let lin_gdr_stride = (gdr_per_layer * std::mem::size_of::<f32>()) as u64;
        // At least 4 bytes so a dense (no-linear-layer) config still allocs a
        // valid buffer; the linear path never binds it then.
        let conv_total = (lin_conv_stride * n_linear as u64).max(4);
        let gdr_total = (lin_gdr_stride * n_linear as u64).max(4);
        let mut lin_conv_state = DeviceBuffer::alloc_uma(ctx, conv_total as usize)
            .map_err(|e| anyhow!("alloc linear conv state ({conv_total} B): {e}"))?;
        let mut lin_gdr_state = DeviceBuffer::alloc_uma(ctx, gdr_total as usize)
            .map_err(|e| anyhow!("alloc linear gdr state ({gdr_total} B): {e}"))?;
        zero_device_buffer(&mut lin_conv_state, conv_total as usize)?;
        zero_device_buffer(&mut lin_gdr_state, gdr_total as usize)?;

        // Batched-prefill scratch: one whole token chunk per activation slot.
        // Separate from `arena` because `widest_gemv` folds in `vocab_size`
        // (151936), and scaling THAT by the chunk width would allocate GBs for a
        // quantize slot the batched path never uses — the LM head runs on the
        // last token only, through the decode arena.
        let prefill =
            crate::prefill::PrefillArena::new(ctx, config, crate::prefill::prefill_chunk_tokens())?;

        Ok(Self {
            arena,
            cache,
            recorder,
            prefill,
            _layouts: vec![l2, l3, l4, l5, l6, l7],
            ring2,
            ring3,
            ring4,
            ring5,
            ring6,
            ring7,
            kv_cache,
            lin_conv_state,
            lin_gdr_state,
            lin_conv_stride,
            lin_gdr_stride,
            gemv_submit_ns: 0,
            gemv_other_ns: 0,
            gemv_count: 0,
        })
    }

    /// Zero the device-resident gated-delta + conv state for a fresh generation
    /// (mirrors [`Qwen35ForwardState::reset`] for the on-device path).
    pub fn reset_linear_state(&mut self) -> Result<()> {
        let conv_len = self.lin_conv_state.len();
        let gdr_len = self.lin_gdr_state.len();
        zero_device_buffer(&mut self.lin_conv_state, conv_len)?;
        zero_device_buffer(&mut self.lin_gdr_state, gdr_len)?;
        Ok(())
    }

    /// Rewind every descriptor-set ring's round-robin cursor. Called once at the
    /// start of each token so its dispatches reuse the rings from slot 0 (the
    /// prior token's submissions have all fence-completed).
    pub fn reset_rings(&mut self) {
        self.ring2.reset();
        self.ring3.reset();
        self.ring4.reset();
        self.ring5.reset();
        self.ring6.reset();
        self.ring7.reset();
    }

    /// Total `vkQueueSubmit` calls the decode recorder has issued — the
    /// submits/token instrument. Snapshot before/after a decode run to get the
    /// per-token submit count (perf-parity Step 4).
    pub fn submit_count(&self) -> u64 {
        self.recorder.submit_count()
    }

    /// Drain the accumulated GEMV timing as
    /// `(submit_secs, other_secs, gemv_count)` and reset the counters. `other`
    /// is the host-side prep + descriptor build + readback around the submit.
    pub fn take_profile(&mut self) -> (f64, f64, u64) {
        let s = self.gemv_submit_ns as f64 / 1e9;
        let o = self.gemv_other_ns as f64 / 1e9;
        let n = self.gemv_count;
        self.gemv_submit_ns = 0;
        self.gemv_other_ns = 0;
        self.gemv_count = 0;
        (s, o, n)
    }
}

/// Widest GEMV shapes for arena sizing: input cols max = FFN intermediate (down
/// proj has the most input cols); output rows max = vocab (lm_head) or the
/// gated q-proj `[query|gate]` width.
fn widest_gemv(config: &Qwen35Config) -> (usize, usize) {
    let h = config.hidden_size;
    // MoE down-proj consumes `moe_intermediate_size` cols; the shared expert
    // consumes `shared_expert_intermediate_size`. Both gate/up project TO those
    // widths (rows). Fold them in so the arena is wide enough for either the
    // dense or the MoE FFN (and the 122B's larger expert width).
    let ffn_inter = config
        .intermediate_size
        .max(config.moe_intermediate_size)
        .max(config.shared_expert_intermediate_size);
    let max_cols = ffn_inter.max(h).max(config.vocab_size);
    let max_rows = config
        .vocab_size
        .max(ffn_inter)
        .max(2 * config.num_attention_heads * config.head_dim);
    (max_cols, max_rows)
}

/// The on-device numeric forward for one token. Returns logits `[vocab]` f32.
///
/// `state.seq_len` must equal `start_pos` (the uncached full-prefix contract).
/// Mutates the per-slot caches / recurrent state in place and advances
/// `state.seq_len`. `res` carries the persistent arena/cache/recorder (built
/// once at model load); every GEMV records into them.
///
/// **Submit batching (perf-parity Step 4).** A token's dispatches are recorded
/// into the persistent [`CommandRecorder`] and submitted only at genuine
/// device→host boundaries — collapsing the old per-op submit+fence round-trips
/// (~600/token) to ~3-4 submits/layer (≈210/token for the 27B). Each batch is a
/// device-resident block that ends where the host must read a value back. A FULL
/// layer is 4 batches: (1) the 3 QKV projection GEMVs + per-head q/k RMSNorm +
/// NeoX RoPE [read back roped K + raw V for the f16 KV cache]; (2) per-head
/// flash-attn + the sigmoid gate [read back the gated activation]; (3) the o-proj
/// GEMV; (4) the fused dense FFN (or the MoE FFN). A LINEAR layer is 3 batches:
/// (1) the qkv/gate/(packed a,b) in-proj GEMVs + conv1d + the gated-delta
/// recurrence [read back z + the recurrence output]; (2) the out-proj GEMV;
/// (3) the FFN. End of token is 1 batch: the final RMSNorm + LM-head GEMV [read
/// back logits].
///
/// No single batch exceeds ~60 tiny dispatches (the full-attn norm/rope batch is
/// the widest), well under the ggml-vulkan ~100-node / APU-TDR cadence cap, so a
/// command buffer never grows large enough to trip the watchdog — the submit
/// boundaries ARE the cadence. The arena slots a batch writes are read back (or
/// consumed by the next device op) before the next batch overwrites them; every
/// in-batch dependency carries a `vkCmdPipelineBarrier`.
pub fn forward_token<'a>(
    ctx: &'a VulkanContext,
    config: &Qwen35Config,
    weights: &ResidentWeights<'_>,
    res: &mut DecodeResources<'a>,
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

    // Rewind the descriptor-set rings for this token (the prior token's
    // submissions have all fence-completed — every GEMV / FFN submit waits).
    res.reset_rings();

    // ── Token embedding (host gather + dequant of one Q8_0 row). ──
    let embed = weights
        .embedding
        .embed_row(token)
        .with_context(|| format!("embed token {token}"))?;
    if embed.len() != h {
        bail!("embedding row width {} != hidden {h}", embed.len());
    }

    // RESIDUAL-RESIDENT layer loop (Step 3): keep the hidden state in the arena
    // `hid` slot across each WHOLE layer (device input-norm → attention →
    // residual-add → post-norm → FFN → residual-add) and submit ONCE per token;
    // only the final `[vocab]` logits read back. MoE layers route on-device too
    // (router GEMV → top-k → fused experts → device-weighted accumulate), so they
    // collapse into the same single submit. This is the only forward path.
    let hidden = forward_layers_resident(ctx, config, weights, res, &embed, start_pos)?;

    // Final norm (plain RMSNorm) + LM head GEMV → logits, recorded into ONE submit:
    // the rms_norm writes work[1], a barrier, then quantize+GEMV reads it against
    // `output.weight`, and only the [vocab] logits read back.
    let logits = final_norm_lm_head(ctx, weights, res, &hidden, eps)?;

    // Per-op GPU timestamp breakdown (ARLE_GPU_TIMESTAMPS): drain the per-category
    // device times accumulated across this token's submits and print them, to
    // compare our op-type cost directly against llama.cpp's GGML_VK_PERF_LOGGER.
    let gpu_prof = res.recorder.take_gpu_profile();
    if !gpu_prof.is_empty() {
        let total: f64 = gpu_prof.iter().map(|(_, _, ms)| *ms).sum();
        let parts: Vec<String> = gpu_prof
            .iter()
            .map(|(label, count, ms)| format!("{label} {ms:.2}ms/{count}"))
            .collect();
        eprintln!(
            "  GPU OP PROFILE (pos {start_pos}): total {total:.2}ms | {}",
            parts.join(" | ")
        );
    }

    state.seq_len += 1;
    Ok(logits)
}

/// The RESIDUAL-RESIDENT dense layer loop (Step 3). The hidden state lives in the
/// arena `hid` slot across the whole stack: the embedding uploads it once, every
/// layer records input-norm → attention → residual-add → post-norm → FFN →
/// residual-add into ONE submit reading/writing `hid` device-resident, and the
/// final hidden reads back once (for the LM head). No per-layer host bridge — the
/// residual stream never leaves the device between layers.
fn forward_layers_resident<'a>(
    ctx: &'a VulkanContext,
    config: &Qwen35Config,
    weights: &ResidentWeights<'_>,
    res: &mut DecodeResources<'a>,
    embed: &[f32],
    start_pos: usize,
) -> Result<Vec<f32>> {
    let h = config.hidden_size;
    let eps = config.rms_norm_eps;
    let hid_off = res.arena.hid.offset;
    let normed_off = res.arena.normed_resident.offset;
    let attn_off = res.arena.attn_resident.offset;

    // Upload the token embedding into the resident hidden slot ONCE.
    res.arena.write_at(hid_off, embed)?;

    // WHOLE-TOKEN batch (perf-parity Lever B / Step 4): open ONE command buffer,
    // record EVERY layer (input-norm → attention → FFN, hidden state arena-resident
    // throughout — no host bridge), then `submit_and_wait` ONCE at token end. This
    // collapses the old 64 per-layer fence-wait GPU stalls into a single submit.
    // The descriptor rings are sized to a whole token (see `DecodeResources::new`),
    // so no slot is reused while its dispatch is still in flight. `--vulkan-submit-cap`
    // (default: whole token) caps the per-batch dispatch count as a TDR safety valve;
    // a flush at a layer boundary stays numerically identical because the `hid`
    // hand-off across the flush is fence-ordered by the next `begin()`.
    let cap = submit_dispatch_cap();
    // Per-layer-type GPU timing probe (ARLE_PROFILE_LAYERS=1): submit + wait after
    // each layer and bucket the elapsed by attention kind, to attribute the decode
    // floor across the full-attention vs gated-delta layers. Reintroduces per-layer
    // serialization (faithful RATIO, inflated absolute); numerically identical.
    let profile = std::env::var("ARLE_PROFILE_LAYERS").is_ok();
    let mut full_ns: u128 = 0;
    let mut lin_ns: u128 = 0;
    // Section probe (ARLE_PROFILE_SECTIONS=1): submit after the attention block
    // and again after the FFN block, bucketing the two — to attribute the decode
    // floor across attention vs FFN (the MoE FFN's expert GEMVs especially).
    // Per-section serialization inflates the absolute but keeps the RATIO faithful.
    let profile_sections = std::env::var("ARLE_PROFILE_SECTIONS").is_ok();
    let mut attn_ns: u128 = 0;
    let mut ffn_ns: u128 = 0;
    let t_record = std::time::Instant::now();
    res.recorder
        .begin()
        .map_err(|e| anyhow!("resident token begin: {e}"))?;
    let mut full_idx = 0usize;
    let mut linear_idx = 0usize;
    for (layer, &layer_type) in config.layer_types.iter().enumerate() {
        let attn_norm = packed_or_f32_norm(weights, layer, "attn_norm")?;
        // input_layernorm: device rms_norm(hid) -> normed_resident.
        record_rms_norm(
            ctx,
            res,
            &attn_norm.buffer,
            "input_norm",
            h,
            eps,
            hid_off,
            normed_off,
        )?;
        res.recorder.barrier();
        // attention block (full or linear): reads normed_resident, writes attn_off.
        match layer_type {
            LayerType::FullAttention => {
                record_full_attention(
                    ctx, config, weights, res, layer, full_idx, normed_off, attn_off, start_pos,
                )?;
                full_idx += 1;
            }
            LayerType::LinearAttention => {
                record_linear_attention(
                    ctx, config, weights, res, layer, linear_idx, normed_off, attn_off,
                )?;
                linear_idx += 1;
            }
        }
        res.recorder.barrier();
        if profile_sections {
            let t = std::time::Instant::now();
            res.recorder
                .submit_and_wait()
                .map_err(|e| anyhow!("layer[{layer}]: section attn submit: {e}"))?;
            res.recorder
                .begin()
                .map_err(|e| anyhow!("layer[{layer}]: section attn begin: {e}"))?;
            attn_ns += t.elapsed().as_nanos();
        }
        // FFN: post-add(hid + attn_off), post-norm, gate/up/swiglu/down, residual
        // add -> hid. Dense FFN, or the residual-resident MoE FFN (router GEMV →
        // on-device top-k → fused expert gather → device-weighted accumulate +
        // shared expert) for MoE layers — both keep the residual stream resident.
        if config.is_moe_layer(layer) {
            record_fused_moe_ffn(ctx, config, weights, res, layer, hid_off, attn_off)?;
        } else {
            record_fused_dense_ffn(ctx, config, weights, res, layer, hid_off, attn_off)?;
        }
        // Inter-layer barrier: the next layer's input-norm reads `hid`, which this
        // layer's FFN residual-add just wrote (the per-layer submit used to enforce
        // this ordering implicitly).
        res.recorder.barrier();

        if profile_sections {
            let t = std::time::Instant::now();
            res.recorder
                .submit_and_wait()
                .map_err(|e| anyhow!("layer[{layer}]: section ffn submit: {e}"))?;
            res.recorder
                .begin()
                .map_err(|e| anyhow!("layer[{layer}]: section ffn begin: {e}"))?;
            ffn_ns += t.elapsed().as_nanos();
        } else if profile {
            let t = std::time::Instant::now();
            res.recorder
                .submit_and_wait()
                .map_err(|e| anyhow!("layer[{layer}]: profile submit: {e}"))?;
            res.recorder
                .begin()
                .map_err(|e| anyhow!("layer[{layer}]: profile begin: {e}"))?;
            let dt = t.elapsed().as_nanos();
            match layer_type {
                LayerType::FullAttention => full_ns += dt,
                LayerType::LinearAttention => lin_ns += dt,
            }
        } else if res.recorder.dispatches_in_batch() as usize >= cap {
            // TDR safety valve: if the open command buffer hit the dispatch cap,
            // flush at this (clean) layer boundary and reopen. Default cap = whole
            // token, so the common path records all layers and submits once below.
            res.recorder
                .submit_and_wait()
                .map_err(|e| anyhow!("layer[{layer}]: resident cap-flush submit: {e}"))?;
            res.recorder
                .begin()
                .map_err(|e| anyhow!("layer[{layer}]: resident cap-flush begin: {e}"))?;
        }
    }
    if profile && (full_idx + linear_idx) > 0 {
        let fm = full_ns as f64 / 1e6;
        let lm = lin_ns as f64 / 1e6;
        eprintln!(
            "  LAYER PROFILE: full-attn {fm:.1}ms / {full_idx} layers = {:.2}ms/layer | \
             linear {lm:.1}ms / {linear_idx} layers = {:.2}ms/layer",
            fm / full_idx.max(1) as f64,
            lm / linear_idx.max(1) as f64,
        );
    }
    if profile_sections {
        let n = (full_idx + linear_idx).max(1) as f64;
        eprintln!(
            "  SECTION PROFILE: attn {:.1}ms ({:.2}ms/layer) | ffn {:.1}ms ({:.2}ms/layer)",
            attn_ns as f64 / 1e6,
            attn_ns as f64 / 1e6 / n,
            ffn_ns as f64 / 1e6,
            ffn_ns as f64 / 1e6 / n,
        );
    }

    // Single submit for the whole token (or the final partial batch when capped).
    let t_submit = std::time::Instant::now();
    res.recorder
        .submit_and_wait()
        .map_err(|e| anyhow!("resident token submit: {e}"))?;
    let submit_ns = t_submit.elapsed().as_nanos();
    res.gemv_submit_ns += submit_ns;
    res.gemv_other_ns += t_record.elapsed().as_nanos().saturating_sub(submit_ns);

    // Read the resident hidden back once (for the final norm + LM head).
    res.arena.read_at(hid_off, h)
}

/// Record the final plain RMSNorm + the LM-head GEMV into ONE submit and read
/// back only the `[vocab]` logits. `hidden` lands in work[0]; rms_norm writes
/// work[1]; quantize(work[1]) -> quant; GEMV(output.weight) -> dst; the dst
/// `[vocab]` rows read back. The end-of-token boundary is the only host
/// dependency, so collapsing the standalone norm submit + the GEMV submit into
/// one saves a submit/token (perf-parity Step 4).
pub(crate) fn final_norm_lm_head<'a>(
    ctx: &'a VulkanContext,
    weights: &ResidentWeights<'_>,
    res: &mut DecodeResources<'a>,
    hidden: &[f32],
    eps: f32,
) -> Result<Vec<f32>> {
    let h = hidden.len();
    let norm_w = weights
        .get("output_norm.weight")
        .ok_or_else(|| anyhow!("missing output_norm.weight"))?;
    if !matches!(norm_w.residency, Residency::DequantF32) {
        bail!(
            "final_norm: output_norm.weight must be F32-resident, got {:?}",
            norm_w.residency
        );
    }
    let lm_head = weights
        .get("output.weight")
        .ok_or_else(|| anyhow!("missing output.weight"))?;
    let (lm_in, vocab) = weight_dims(lm_head, "output.weight")?;
    if lm_in != h {
        bail!("final_norm: lm_head in-dim {lm_in} != hidden {h}");
    }
    if vocab > res.arena.max_rows {
        bail!(
            "final_norm: vocab {vocab} exceeds arena rows ({})",
            res.arena.max_rows
        );
    }

    let in_off = res.arena.work[0].offset;
    let normed_off = res.arena.work[1].offset;
    let dst_off = res.arena.dst.offset;
    res.arena.write_work(0, hidden)?;

    let t_start = std::time::Instant::now();
    res.recorder
        .begin()
        .map_err(|e| anyhow!("final_norm: recorder begin: {e}"))?;
    record_rms_norm(
        ctx,
        res,
        &norm_w.buffer,
        "final_norm",
        h,
        eps,
        in_off,
        normed_off,
    )?;
    res.recorder.barrier();
    record_quantize_gemv(
        ctx,
        res,
        lm_head,
        "output.weight",
        h,
        vocab,
        0,
        lm_head.buffer.len() as u64,
        normed_off,
        dst_off,
    )?;
    let t_submit = std::time::Instant::now();
    res.recorder
        .submit_and_wait()
        .map_err(|e| anyhow!("final_norm: submit: {e}"))?;
    let submit_ns = t_submit.elapsed().as_nanos();

    let logits = res.arena.read_at(dst_off, vocab)?;
    let total_ns = t_start.elapsed().as_nanos();
    res.gemv_submit_ns += submit_ns;
    res.gemv_other_ns += total_ns.saturating_sub(submit_ns);
    res.gemv_count += 1;
    Ok(logits)
}

// ─────────────────────────────────────────────────────────────────────────────
// Full-attention layer (gated q_proj, per-head q/k norm + NeoX RoPE, KV-cached
// flash-attention + sigmoid gate). The q/k RMSNorm, NeoX RoPE, KV-cached
// flash-attention, and per-head sigmoid gate now run **ON THE DEVICE**
// (full-attention on-device, Step 6), each oracle-gated against the host f32
// reference in `crates/vulkan-kernels/tests/device_full_attention.rs`. Only the
// projections (already device GEMV) bracket this block; the heavy host SDPA
// triple-loop and the per-element norm/rope/gate host math are gone.
// ─────────────────────────────────────────────────────────────────────────────

/// Record (NO begin/submit) the WHOLE full-attention block device-resident into
/// the caller's OPEN recorder (Step 3 residual-resident): reads the input-normed
/// activation from arena byte `normed_off`, writes the o_proj output `[hidden]`
/// to arena byte `out_off`. The whole block — 3 QKV projection GEMVs → per-head
/// q/k RMSNorm → NeoX RoPE → DEVICE f16 KV-pack of roped K + raw V → per-head
/// KV-cached flash-attn → per-head sigmoid gate → o_proj GEMV — records with NO
/// host hop. The caller brackets it with `begin()`/`submit_and_wait()` and chains
/// the FFN into the same submit, so the residual stream never leaves the device.
#[allow(clippy::too_many_arguments)]
fn record_full_attention<'a>(
    ctx: &'a VulkanContext,
    config: &Qwen35Config,
    weights: &ResidentWeights<'_>,
    res: &mut DecodeResources<'a>,
    layer: usize,
    full_idx: usize,
    normed_off: u64,
    out_off_param: u64,
    start_pos: usize,
) -> Result<()> {
    let h = config.hidden_size;
    let hd = config.head_dim;
    let nq = config.num_attention_heads;
    let nkv = config.num_key_value_heads;
    let q_dim = nq * hd;
    let kv_dim = nkv * hd;
    let rotary_dim = config.rotary_dim;
    let theta = config.rope_theta;
    let eps = config.rms_norm_eps;
    let pos = start_pos;
    let group = nq / nkv;
    let scale = 1.0f32 / (hd as f32).sqrt();

    if pos >= KV_CACHE_MAX_SEQ {
        bail!(
            "full_attention[{layer}]: position {pos} exceeds device KV cache cap {KV_CACHE_MAX_SEQ}"
        );
    }

    // q/k RMSNorm weights are F32-resident device tensors (PLAIN per-head, head_dim
    // wide, broadcast across heads). Bind them directly into the device rms_norm.
    let q_norm = packed_or_f32_norm(weights, layer, "attn_q_norm")?;
    let k_norm = packed_or_f32_norm(weights, layer, "attn_k_norm")?;
    let q_w = packed_layer_weight(weights, layer, "attn_q")?;
    let k_w = packed_layer_weight(weights, layer, "attn_k")?;
    let v_w = packed_layer_weight(weights, layer, "attn_v")?;
    let (qw_in, qw_out) = weight_dims(q_w, "attn_q")?;
    let (kw_in, kw_out) = weight_dims(k_w, "attn_k")?;
    let (vw_in, vw_out) = weight_dims(v_w, "attn_v")?;
    if qw_in != h || kw_in != h || vw_in != h {
        bail!("full_attn[{layer}]: QKV in-dim q{qw_in}/k{kw_in}/v{vw_in} != hidden {h}");
    }
    if qw_out != 2 * q_dim || kw_out != kv_dim || vw_out != kv_dim {
        bail!(
            "full_attn[{layer}]: QKV out-dim q{qw_out}/k{kw_out}/v{vw_out} != expected \
             {}/{kv_dim}/{kv_dim}",
            2 * q_dim
        );
    }

    // Arena slots: q_full ([query|gate] per head) -> attn_qkv, k_in -> attn_k,
    // v_in -> work[0]. q/k normed+roped land in attn_q/attn_k; the gated flash
    // output lands in attn_out, then the o_proj writes `out_off_param`.
    let qkv_off = res.arena.attn_qkv.offset;
    let q_off = res.arena.attn_q.offset;
    let k_off = res.arena.attn_k.offset;
    let v_off = res.arena.work[0].offset;
    let out_off = res.arena.attn_out.offset;
    let f32_b = std::mem::size_of::<f32>() as u64;

    let o_w = packed_layer_weight(weights, layer, "attn_output")?;
    let (ow_in, ow_out) = weight_dims(o_w, "attn_output")?;
    if ow_in != q_dim {
        bail!("full_attn[{layer}]: o_proj in-dim {ow_in} != q_dim {q_dim}");
    }
    if ow_out != h {
        bail!("full_attn[{layer}]: o_proj out-dim {ow_out} != hidden {h}");
    }
    let kv_len = pos + 1;
    let x_in_off = normed_off;
    // q/k/v all project the SAME `h`-wide normed input. Quantize it to q8_1 ONCE
    // (into the shared `quant` slot), barrier, then run the three projections as
    // GEMV-only dispatches reading that one activation — instead of re-quantizing
    // the identical input three times. Cuts the per-layer attention quantize +
    // barrier count from 3 to 1. The three GEMVs write disjoint dst slots
    // (qkv_off / k_off / v_off) so they need no barriers between them; one
    // trailing barrier separates them from the q/k norm that reads them.
    let quant_off = res.arena.quant.offset;
    record_quantize(ctx, res, "attn_qkv_qin", h, x_in_off, quant_off)?;
    res.recorder.barrier();
    // q_proj -> attn_qkv (2*q_dim).
    record_gemv_only(
        ctx,
        res,
        q_w,
        "attn_q",
        h,
        2 * q_dim,
        0,
        q_w.buffer.len() as u64,
        quant_off,
        qkv_off,
    )?;
    // k_proj -> attn_k (kv_dim).
    record_gemv_only(
        ctx,
        res,
        k_w,
        "attn_k",
        h,
        kv_dim,
        0,
        k_w.buffer.len() as u64,
        quant_off,
        k_off,
    )?;
    // v_proj -> work[0] (kv_dim).
    record_gemv_only(
        ctx,
        res,
        v_w,
        "attn_v",
        h,
        kv_dim,
        0,
        v_w.buffer.len() as u64,
        quant_off,
        v_off,
    )?;
    res.recorder.barrier();
    res.gemv_count += 3;
    for hh in 0..nq {
        // query half of head hh in q_full (stride 2*hd): bytes [hh*2*hd .. +hd].
        let qsrc = qkv_off + (hh * 2 * hd) as u64 * f32_b;
        let qdst = q_off + (hh * hd) as u64 * f32_b;
        record_rms_norm(ctx, res, &q_norm.buffer, "attn_q_norm", hd, eps, qsrc, qdst)?;
    }
    res.recorder.barrier();
    for hh in 0..nq {
        let qdst = q_off + (hh * hd) as u64 * f32_b;
        record_rope_neox(
            ctx,
            res,
            "attn_q_rope",
            hd,
            rotary_dim,
            pos,
            theta,
            qdst,
            qdst,
        )?;
    }
    for hh in 0..nkv {
        let ksrc = k_off + (hh * hd) as u64 * f32_b;
        record_rms_norm(ctx, res, &k_norm.buffer, "attn_k_norm", hd, eps, ksrc, ksrc)?;
    }
    res.recorder.barrier();
    for hh in 0..nkv {
        let kdst = k_off + (hh * hd) as u64 * f32_b;
        record_rope_neox(
            ctx,
            res,
            "attn_k_rope",
            hd,
            rotary_dim,
            pos,
            theta,
            kdst,
            kdst,
        )?;
    }
    // Barrier: the f16 KV-pack reads the roped K (attn_k) + raw V (work[0]).
    res.recorder.barrier();
    // DEVICE f16 KV-pack: roped K (attn_k) + raw V (work[0]) -> cache at `pos`.
    for kvh in 0..nkv {
        let ksrc = k_off + (kvh * hd) as u64 * f32_b;
        let vsrc = v_off + (kvh * hd) as u64 * f32_b;
        record_f16_kv_pack(ctx, res, "kv_pack_k", hd, ksrc, full_idx, kvh, pos, false)?;
        record_f16_kv_pack(ctx, res, "kv_pack_v", hd, vsrc, full_idx, kvh, pos, true)?;
    }
    // Barrier: flash reads the cache rows the pack just wrote (+ the roped Q).
    res.recorder.barrier();
    // gqa_ratio=1: each query head reads its kv head's plane directly.
    for hh in 0..nq {
        let kvh = hh / group;
        let qh_off = q_off + (hh * hd) as u64 * f32_b;
        let oh_off = out_off + (hh * hd) as u64 * f32_b;
        let k_plane = res.kv_cache.k_plane_off(full_idx, kvh);
        let v_plane = res.kv_cache.v_plane_off(full_idx, kvh);
        record_flash_attn(
            ctx,
            res,
            "full_flash",
            hd,
            kv_len,
            scale,
            qh_off,
            k_plane,
            v_plane,
            oh_off,
        )?;
    }
    res.recorder.barrier();
    for hh in 0..nq {
        // gate half of head hh in q_full: bytes [hh*2*hd+hd .. +hd]. Sigmoid-mul
        // folds the per-head gate onto the flash output in place.
        let gate_off = qkv_off + (hh * 2 * hd + hd) as u64 * f32_b;
        let oh_off = out_off + (hh * hd) as u64 * f32_b;
        record_sigmoid_mul(ctx, res, "full_gate", hd, gate_off, oh_off, oh_off)?;
    }
    // Barrier: the o_proj GEMV reads the gated attention output (attn_out).
    res.recorder.barrier();
    // o_proj: [q_dim -> hidden], chained into the SAME submit. The device KV
    // cache already owns this token's K/V; only the o_proj result reads back.
    record_quantize_gemv(
        ctx,
        res,
        o_w,
        "attn_output",
        q_dim,
        ow_out,
        0,
        o_w.buffer.len() as u64,
        out_off,
        out_off_param,
    )?;
    res.gemv_count += 1;
    Ok(())
}

/// Max compute dispatches recorded into one command buffer before the
/// residual-resident loop flushes (submit + re-begin) at the next layer boundary.
/// Default `usize::MAX` = a whole token in ONE submit (the perf-parity target). A
/// flush is numerically transparent — the `hid` hand-off is fence-ordered by the
/// reopening `begin()` — so `--vulkan-submit-cap <n>` is a pure TDR/latency safety
/// valve (lower → more submits, smaller command buffers) with no effect on output.
static SUBMIT_DISPATCH_CAP: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);

/// `--vulkan-submit-cap`, set once pre-load (values `> 0`).
pub fn set_submit_cap(cap: usize) {
    if cap > 0 {
        SUBMIT_DISPATCH_CAP.store(cap, std::sync::atomic::Ordering::Relaxed);
    }
}

pub(crate) fn submit_dispatch_cap() -> usize {
    SUBMIT_DISPATCH_CAP.load(std::sync::atomic::Ordering::Relaxed)
}

/// Record (NO begin/submit) the WHOLE linear (gated-delta) attention block
/// device-resident into the caller's OPEN recorder (Step 3 residual-resident):
/// reads the input-normed activation from arena byte `normed_off`, writes the
/// out-proj output `[hidden]` to arena byte `out_off`. The depthwise conv1d + the
/// recurrent gated-delta state update run on the two model-specific serial shaders
/// against device-resident state (the conv ring + the `[v_head, key_dim, val_dim]`
/// S matrix per linear layer); the gated output RMSNorm × silu(z) and the out-proj
/// GEMV chain in too — ALL with no host hop.
///
/// `ssm_alpha`/`ssm_beta` record device-resident regardless of residency: packed
/// (dense) via the quantized GEMV, F32-resident (35B MoE) via the F32
/// `router_gemv` kernel. So the whole linear block — including the 35B MoE's F32
/// a/b — records into the caller's submit with no host hop.
#[allow(clippy::too_many_arguments)]
fn record_linear_attention<'a>(
    ctx: &'a VulkanContext,
    config: &Qwen35Config,
    weights: &ResidentWeights<'_>,
    res: &mut DecodeResources<'a>,
    layer: usize,
    linear_idx: usize,
    normed_off: u64,
    out_off_param: u64,
) -> Result<()> {
    let kd = config.linear_key_head_dim; // 128
    let vd = config.linear_value_head_dim; // 128
    let nk = config.linear_num_key_heads; // 16
    let nv = config.linear_num_value_heads; // 48
    let kernel = config.linear_conv_kernel_dim; // 4
    let v_dim_total = nv * vd; // 6144
    let qkv_dim = 2 * nk * kd + v_dim_total; // 10240
    let eps = config.rms_norm_eps;
    let h = config.hidden_size;

    let qkv_w = packed_layer_weight(weights, layer, "attn_qkv")?;
    let gate_w = packed_layer_weight(weights, layer, "attn_gate")?;
    let (qkv_in, qkv_out) = weight_dims(qkv_w, "attn_qkv")?;
    let (gate_in, gate_out) = weight_dims(gate_w, "attn_gate")?;
    if qkv_in != h || gate_in != h {
        bail!("linear[{layer}]: in-proj in-dim qkv{qkv_in}/gate{gate_in} != hidden {h}");
    }
    if qkv_out != qkv_dim || gate_out != v_dim_total {
        bail!(
            "linear[{layer}]: in-proj out-dim qkv{qkv_out}/gate{gate_out} != {qkv_dim}/{v_dim_total}"
        );
    }
    let z_off = res.arena.work[1].offset;

    // a/b projections (`ssm_alpha`/`ssm_beta`, `[hidden -> nv]`) land in
    // lin_a/lin_b. When PACKED (dense configs) they record into the SAME merged
    // submit as qkv/gate/conv/gdr. When F32-resident (the 35B MoE) they are tiny
    // host dots staged into the slots before the submit (needs `host_normed`).
    let a_w = weights
        .get(&format!("blk.{layer}.ssm_alpha.weight"))
        .ok_or_else(|| anyhow!("missing blk.{layer}.ssm_alpha.weight"))?;
    let b_w = weights
        .get(&format!("blk.{layer}.ssm_beta.weight"))
        .ok_or_else(|| anyhow!("missing blk.{layer}.ssm_beta.weight"))?;
    let a_packed = matches!(a_w.residency, Residency::KeepQuant(_));
    let b_packed = matches!(b_w.residency, Residency::KeepQuant(_));
    // a/b (`ssm_alpha`/`ssm_beta`, `[hidden -> nv]`) record device-resident into
    // the merged submit below — packed via the quantized GEMV, F32-resident (35B
    // MoE) via the F32 `router_gemv` kernel — so the linear block needs no host
    // hop and the whole token stays in one submit.

    // ── Resolve the F32-resident SSM weight buffers (bound directly). ──
    let conv_w_buf = ssm_weight_buffer(weights, layer, "ssm_conv1d.weight", kernel * qkv_dim)?;
    let a_log_buf = ssm_weight_buffer(weights, layer, "ssm_a", nv)?;
    let dt_bias_buf = ssm_weight_buffer(weights, layer, "ssm_dt.bias", nv)?;

    // Per-linear-layer state sub-ranges (resident across tokens).
    let conv_off = res.lin_conv_stride * linear_idx as u64;
    let conv_len = res.lin_conv_stride;
    let gdr_off = res.lin_gdr_stride * linear_idx as u64;
    let gdr_len = res.lin_gdr_stride;

    let conv_push = qwen35_ssm_conv_params(qkv_dim as u32, 1, kernel as u32).to_le_bytes();
    let conv_groups = {
        let d = qwen35_ssm_conv_dispatch(qkv_dim as u32);
        [d.x, d.y, d.z]
    };
    let xseq_off = res.arena.lin_xseq.offset;
    let qkv_conv_off = res.arena.lin_qkv_conv.offset;
    let xseq_len = (qkv_dim * std::mem::size_of::<f32>()) as u64;

    // The input-normed activation is already device-resident at `normed_off`
    // (the resident loop's device input-norm wrote it); the in-proj GEMVs read it.
    let x_in_off = normed_off;
    // qkv -> lin_xseq (conv input).
    record_quantize_gemv(
        ctx,
        res,
        qkv_w,
        "attn_qkv",
        qkv_in,
        qkv_dim,
        0,
        qkv_w.buffer.len() as u64,
        x_in_off,
        xseq_off,
    )?;
    res.recorder.barrier();
    // gate z -> work[1] (read by the device gated norm's swiglu).
    record_quantize_gemv(
        ctx,
        res,
        gate_w,
        "attn_gate",
        gate_in,
        v_dim_total,
        0,
        gate_w.buffer.len() as u64,
        x_in_off,
        z_off,
    )?;
    res.recorder.barrier();
    res.gemv_count += 2;
    // a/b projections recorded into the SAME submit (-> lin_a / lin_b): packed
    // via the quantized GEMV, F32-resident (35B MoE) via the F32 router_gemv
    // (`y[e] = Σ_c W[e,c]·normed[c]`, reading the device-resident normed input).
    {
        let (a_in, a_out) = weight_dims(a_w, "ssm_alpha")?;
        if a_in != h || a_out != nv {
            bail!("linear[{layer}]: ssm_alpha dims [{a_in},{a_out}] != [{h},{nv}]");
        }
        if a_packed {
            record_quantize_gemv(
                ctx,
                res,
                a_w,
                "ssm_alpha",
                a_in,
                nv,
                0,
                a_w.buffer.len() as u64,
                x_in_off,
                res.arena.lin_a.offset,
            )?;
        } else {
            record_router_gemv(
                ctx,
                res,
                &a_w.buffer,
                "ssm_alpha",
                nv,
                h,
                false,
                x_in_off,
                res.arena.lin_a.offset,
            )?;
        }
        res.recorder.barrier();
        res.gemv_count += 1;
    }
    {
        let (b_in, b_out) = weight_dims(b_w, "ssm_beta")?;
        if b_in != h || b_out != nv {
            bail!("linear[{layer}]: ssm_beta dims [{b_in},{b_out}] != [{h},{nv}]");
        }
        if b_packed {
            record_quantize_gemv(
                ctx,
                res,
                b_w,
                "ssm_beta",
                b_in,
                nv,
                0,
                b_w.buffer.len() as u64,
                x_in_off,
                res.arena.lin_b.offset,
            )?;
        } else {
            record_router_gemv(
                ctx,
                res,
                &b_w.buffer,
                "ssm_beta",
                nv,
                h,
                false,
                x_in_off,
                res.arena.lin_b.offset,
            )?;
        }
        res.recorder.barrier();
        res.gemv_count += 1;
    }
    // ── conv1d + SiLU (single token), advancing the resident ring; reads lin_xseq.
    {
        let DecodeResources {
            cache,
            recorder,
            arena,
            ring4,
            lin_conv_state,
            ..
        } = res;
        let arena_buf = &arena.buffer;
        let (pipeline, _) = cache
            .get(
                ctx,
                Kernel::Qwen35SsmConv,
                Kernel::Qwen35SsmConv.specialization_u32(),
                conv_push.len() as u32,
                4,
            )
            .map_err(|e| anyhow!("linear[{layer}]: build conv pipeline: {e}"))?;
        let set = ring4
            .next_updated(&[
                (arena_buf, xseq_off, xseq_len),          // 0: XSeq (raw qkv)
                (conv_w_buf, 0, conv_w_buf.len() as u64), // 1: ConvWeight
                (lin_conv_state, conv_off, conv_len),     // 2: ConvState (ring)
                (arena_buf, qkv_conv_off, xseq_len),      // 3: OutSeq
            ])
            .map_err(|e| anyhow!("linear[{layer}]: bind conv set: {e}"))?;
        recorder.label_next("linear");
        recorder.dispatch_raw(pipeline, set, &conv_push, conv_groups);
    }
    // Barrier: the gated-delta net (next dispatch) reads the conv's `lin_qkv_conv`
    // output. conv + gdr now record into the SAME open recorder and submit ONCE
    // (was 2 submits) — the conv ring (resident) is written by conv, the S matrix
    // (resident) by gdr; both are distinct device buffers so the single submit is
    // hazard-free with this one barrier.
    res.recorder.barrier();

    // ── 2. Recurrent gated-delta state update (single token), read+writes the
    //       resident S matrix. Reads `qkv_conv` (binding 0) + the a/b projections.
    let gd_push =
        qwen35_gated_delta_net_params(nk as u32, nv as u32, kd as u32, vd as u32, 1).to_le_bytes();
    let gd_groups = {
        let d = qwen35_gated_delta_net_dispatch(nv as u32);
        [d.x, d.y, d.z]
    };
    let a_off = res.arena.lin_a.offset;
    let b_off = res.arena.lin_b.offset;
    let out_off = res.arena.lin_out.offset;
    let z_slot_off = res.arena.work[1].offset; // gate z (written by the gate GEMV)
    let a_len = (nv * std::mem::size_of::<f32>()) as u64;
    let out_len = (v_dim_total * std::mem::size_of::<f32>()) as u64;
    {
        let DecodeResources {
            cache,
            recorder,
            arena,
            ring7,
            lin_gdr_state,
            ..
        } = res;
        let arena_buf = &arena.buffer;
        let (pipeline, _) = cache
            .get(
                ctx,
                Kernel::Qwen35GatedDeltaNet,
                Kernel::Qwen35GatedDeltaNet.specialization_u32(),
                gd_push.len() as u32,
                7,
            )
            .map_err(|e| anyhow!("linear[{layer}]: build gated-delta pipeline: {e}"))?;
        let set = ring7
            .next_updated(&[
                (arena_buf, qkv_conv_off, xseq_len),        // 0: Qkv (post-conv)
                (arena_buf, b_off, a_len),                  // 1: BProj
                (arena_buf, a_off, a_len),                  // 2: AProj
                (dt_bias_buf, 0, dt_bias_buf.len() as u64), // 3: DtBias
                (a_log_buf, 0, a_log_buf.len() as u64),     // 4: ALog (ssm_a)
                (lin_gdr_state, gdr_off, gdr_len),          // 5: State (S matrix)
                (arena_buf, out_off, out_len),              // 6: Output
            ])
            .map_err(|e| anyhow!("linear[{layer}]: bind gated-delta set: {e}"))?;
        recorder.label_next("linear");
        recorder.dispatch_raw(pipeline, set, &gd_push, gd_groups);
    }

    // ── DEVICE GATED OUTPUT RMSNorm × silu(z) (Step 1: no z / recurrence-out
    //    readback). Per value head over val_dim: `out = rms_norm(gdr_out_head) *
    //    ssm_norm * silu(z_head)`. The PLAIN f32 `ssm_norm` weight (vd-wide,
    //    broadcast across heads) binds directly to the device rms_norm, which the
    //    per-head dispatch applies (`x*inv_rms*w`); the per-element `* silu(z)`
    //    folds into a single SwiGLU pass (gate=z, up=normed) over the whole
    //    [v_dim_total] block. All recorded into the SAME submit as conv+gdr, then
    //    the out-proj GEMV is chained too — collapsing the linear path to ONE
    //    submit and removing both readbacks. ──
    let ssm_norm_buf = ssm_weight_buffer(weights, layer, "ssm_norm.weight", vd)?;
    // rms_norm(gdr_out)*ssm_norm lands in the lin_qkv_conv slot (free after the
    // conv→gdr chain consumed it; qkv_dim ≥ v_dim_total so it fits).
    let gated_normed_off = res.arena.lin_qkv_conv.offset;
    res.recorder.barrier(); // gated-norm reads the gdr Output written above.
    for vh in 0..nv {
        let src = out_off + (vh * vd) as u64 * std::mem::size_of::<f32>() as u64;
        let dst = gated_normed_off + (vh * vd) as u64 * std::mem::size_of::<f32>() as u64;
        record_rms_norm(ctx, res, ssm_norm_buf, "ssm_norm", vd, eps, src, dst)?;
    }
    res.recorder.barrier(); // the swiglu reads every head's normed output + z.
    // out = silu(z) * normed over [v_dim_total]; reuse lin_out for the result.
    record_swiglu(
        ctx,
        res,
        "ssm_gated",
        v_dim_total,
        z_slot_off,
        gated_normed_off,
        out_off,
    )?;
    res.recorder.barrier(); // out-proj GEMV reads the gated norm output.

    // ── out_proj: [v_dim_total -> hidden] (device GEMV), chained into the SAME
    //    submit. Writes the layer's attention output `[hidden]` to `out_off_param`
    //    (the residual-resident slot the FFN's post-add consumes). ──
    let ssm_out_w = packed_layer_weight(weights, layer, "ssm_out")?;
    let (so_in, so_out) = weight_dims(ssm_out_w, "ssm_out")?;
    if so_in != v_dim_total {
        bail!("linear[{layer}]: ssm_out in-dim {so_in} != v_dim_total {v_dim_total}");
    }
    if so_out != h {
        bail!("linear[{layer}]: ssm_out out-dim {so_out} != hidden {h}");
    }
    record_quantize_gemv(
        ctx,
        res,
        ssm_out_w,
        "ssm_out",
        v_dim_total,
        so_out,
        0,
        ssm_out_w.buffer.len() as u64,
        out_off,
        out_off_param,
    )?;
    res.gemv_count += 1;
    Ok(())
}

/// Borrow an F32-resident SSM weight tensor's device buffer (`DequantF32`
/// residency), checking it holds at least `len` f32. The conv1d weight, `ssm_a`,
/// and `ssm_dt.bias` are stored on-device as plain f32 in the exact layout the
/// `qwen35_*` shaders bind, so they bind directly with no host round-trip.
pub(crate) fn ssm_weight_buffer<'b>(
    weights: &'b ResidentWeights<'_>,
    layer: usize,
    suffix: &str,
    len: usize,
) -> Result<&'b DeviceBuffer<'b>> {
    let name = format!("blk.{layer}.{suffix}");
    let t = weights
        .get(&name)
        .ok_or_else(|| anyhow!("missing {name}"))?;
    if !matches!(t.residency, Residency::DequantF32) {
        bail!(
            "{name}: expected F32-resident SSM weight, got {:?}",
            t.residency
        );
    }
    let want = len * std::mem::size_of::<f32>();
    if t.buffer.len() < want {
        bail!("{name}: F32 buffer {} B < {len} f32 needed", t.buffer.len());
    }
    Ok(&t.buffer)
}

// ─────────────────────────────────────────────────────────────────────────────
// On-device GEMV helpers (the proven q8_0 path), now recording the quantize+GEMV
// pair into the persistent recorder/cache against arena sub-buffers.
// ─────────────────────────────────────────────────────────────────────────────

/// Record (no submit) the q8_1 quantize + Q8_0/K-quant GEMV pair into the OPEN
/// recorder: read the f32 input activation from arena byte `x_in_off`
/// (`ncols` wide), quantize it into the `quant` slot, barrier, then GEMV the
/// `[nrows, ncols]` weight sub-range into arena byte `dst_off` (`nrows` f32).
///
/// Descriptor sets come from the persistent [`DescriptorSetRing`]s (no per-call
/// `VkDescriptorPool` churn — Step 5a); pipelines from the compile-once
/// [`KernelCache`]. The caller `begin()`s the recorder and `submit_and_wait()`s
/// after one or more recorded ops (the fused FFN records several before one
/// submit). Field-level borrows keep cache/ring/recorder/arena disjoint.
#[allow(clippy::too_many_arguments)]
fn record_quantize_gemv<'a>(
    ctx: &'a VulkanContext,
    res: &mut DecodeResources<'a>,
    weight: &DeviceTensor<'_>,
    name: &str,
    ncols: usize,
    nrows: usize,
    weight_offset: u64,
    weight_len: u64,
    x_in_off: u64,
    dst_off: u64,
) -> Result<()> {
    let quant_bytes = q8_1_x4_bytes(ncols);
    let dst_bytes = nrows * std::mem::size_of::<f32>();
    let quant_off = res.arena.quant.offset;
    let fuse0 = (res.arena.fuse0.offset, res.arena.fuse0.len as u64);
    let fuse1 = (res.arena.fuse1.offset, res.arena.fuse1.len as u64);

    let q_spec = Kernel::QuantizeQ8_1.specialization_u32();
    let q_push = q8_1_quantize_params(ncols as u32).to_le_bytes();
    let q_groups = {
        let d = q8_1_quantize_dispatch(ncols as u32);
        [d.x, d.y, d.z]
    };
    // Pick the GEMV kernel from the weight's packed quant type: the K-quant and
    // Q8_0 `mul_mat_vecq_*` shaders all consume the SAME q8_1_x4 activation and
    // the SAME 13-uint `gemv_params`; only the weight-decode shader differs.
    let gemv_kernel = gemv_kernel_for(weight, name)?;
    let g_spec = gemv_kernel.specialization_u32();
    let g_push = gemv_params(ncols as u32, nrows as u32).to_le_bytes();
    let g_groups = {
        let d = gemv_dispatch(nrows as u32);
        [d.x, d.y, d.z]
    };

    // Disjoint field borrows: cache (pipeline), ring (set), recorder (record),
    // arena.buffer (binding targets) are different fields of `res`.
    let DecodeResources {
        cache,
        recorder,
        arena,
        ring2,
        ring5,
        ..
    } = res;
    let arena_buf = &arena.buffer;

    // --- quantize dispatch (2 bindings) ---
    {
        let (pipeline, _) = cache
            .get(ctx, Kernel::QuantizeQ8_1, q_spec, q_push.len() as u32, 2)
            .map_err(|e| anyhow!("{name}: build q8_1_quantize pipeline: {e}"))?;
        let set = ring2
            .next_updated(&[
                (arena_buf, x_in_off, (ncols * 4) as u64),
                (arena_buf, quant_off, quant_bytes as u64),
            ])
            .map_err(|e| anyhow!("{name}: bind q8_1 ring set: {e}"))?;
        recorder.label_next("quant");
        recorder.dispatch_raw(pipeline, set, &q_push, q_groups);
    }

    // Barrier: GEMV reads the quantize's writes.
    recorder.barrier();

    // --- GEMV dispatch (5 bindings) ---
    {
        let (pipeline, _) = cache
            .get(ctx, gemv_kernel, g_spec, g_push.len() as u32, 5)
            .map_err(|e| anyhow!("{name}: build {gemv_kernel:?} GEMV pipeline: {e}"))?;
        let set = ring5
            .next_updated(&[
                // binding 0: weight sub-range (resident expert slice / whole tensor).
                (&weight.buffer, weight_offset, weight_len),
                // binding 1: q8_1_x4 activations.
                (arena_buf, quant_off, quant_bytes as u64),
                // binding 2: f32 dst rows.
                (arena_buf, dst_off, dst_bytes as u64),
                // bindings 3/4: fuse dummies (unread, fusion_flags=0).
                (arena_buf, fuse0.0, fuse0.1),
                (arena_buf, fuse1.0, fuse1.1),
            ])
            .map_err(|e| anyhow!("{name}: bind GEMV ring set: {e}"))?;
        recorder.label_next("gemv");
        recorder.dispatch_raw(pipeline, set, &g_push, g_groups);
    }
    Ok(())
}

/// Record (no submit) ONLY the Q8_0/K-quant GEMV against an ALREADY-quantized
/// q8_1_x4 activation at arena byte `quant_off`. This is the GEMV half of
/// [`record_quantize_gemv`] with the quantize + its barrier dropped: when several
/// projections share one input (q/k/v all read the normed input; gate+up both
/// read `mlp_in`), the caller quantizes that input ONCE via [`record_quantize`],
/// then issues each projection through this helper. That collapses the per-layer
/// `3×(quantize+barrier)` for q/k/v (and `2×` for gate/up) down to a single
/// quantize, matching the MoE fused path's "quantize the shared input once"
/// scheduling. The GEMV numeric contract (13-uint `gemv_params`, 5-buffer ABI)
/// is unchanged — only the redundant re-quantization is removed.
///
/// `ncols` MUST equal the width the activation at `quant_off` was quantized to
/// (the shared input width); `quant_off` MUST hold a fresh q8_1_x4 of that input
/// AND a barrier MUST already separate that quantize from this GEMV.
#[allow(clippy::too_many_arguments)]
fn record_gemv_only<'a>(
    ctx: &'a VulkanContext,
    res: &mut DecodeResources<'a>,
    weight: &DeviceTensor<'_>,
    name: &str,
    ncols: usize,
    nrows: usize,
    weight_offset: u64,
    weight_len: u64,
    quant_off: u64,
    dst_off: u64,
) -> Result<()> {
    let quant_bytes = q8_1_x4_bytes(ncols);
    let dst_bytes = nrows * std::mem::size_of::<f32>();
    let fuse0 = (res.arena.fuse0.offset, res.arena.fuse0.len as u64);
    let fuse1 = (res.arena.fuse1.offset, res.arena.fuse1.len as u64);

    let gemv_kernel = gemv_kernel_for(weight, name)?;
    let g_spec = gemv_kernel.specialization_u32();
    let g_push = gemv_params(ncols as u32, nrows as u32).to_le_bytes();
    let g_groups = {
        let d = gemv_dispatch(nrows as u32);
        [d.x, d.y, d.z]
    };

    let DecodeResources {
        cache,
        recorder,
        arena,
        ring5,
        ..
    } = res;
    let arena_buf = &arena.buffer;

    let (pipeline, _) = cache
        .get(ctx, gemv_kernel, g_spec, g_push.len() as u32, 5)
        .map_err(|e| anyhow!("{name}: build {gemv_kernel:?} GEMV pipeline: {e}"))?;
    let set = ring5
        .next_updated(&[
            (&weight.buffer, weight_offset, weight_len),
            (arena_buf, quant_off, quant_bytes as u64),
            (arena_buf, dst_off, dst_bytes as u64),
            (arena_buf, fuse0.0, fuse0.1),
            (arena_buf, fuse1.0, fuse1.1),
        ])
        .map_err(|e| anyhow!("{name}: bind GEMV-only ring set: {e}"))?;
    recorder.label_next("gemv");
    recorder.dispatch_raw(pipeline, set, &g_push, g_groups);
    Ok(())
}

/// `(ncols=in, nrows=out)` from a weight's recorded GGUF dims. GGUF stores
/// `dims = [ne0=in, ne1=out]` and the bytes are row-major `[out, in]`, which is
/// exactly the GEMV's `[nrows, ncols]` contract. The loader records these at
/// upload time ([`DeviceTensor::gemv_dims`]).
pub(crate) fn weight_dims(weight: &DeviceTensor<'_>, name: &str) -> Result<(usize, usize)> {
    weight
        .gemv_dims
        .ok_or_else(|| anyhow!("{name}: resident tensor has no GEMV dims recorded"))
}

/// The `mul_mat_vecq_*` GEMV kernel for a packed-quant weight's GGUF type.
/// All four share the q8_1_x4 activation + `gemv_params` layout; only the
/// weight-decode shader differs. Fails loud on a type with no registered GEMV
/// (e.g. the 122B's MXFP4 experts, which need a dedicated kernel).
fn gemv_kernel_for(weight: &DeviceTensor<'_>, name: &str) -> Result<Kernel> {
    let Residency::KeepQuant(ty) = weight.residency else {
        bail!(
            "{name}: GEMV expects a packed-quant weight, got {:?}",
            weight.residency
        );
    };
    Ok(match ty {
        GgmlType::Q4K => Kernel::GemvQ4K,
        GgmlType::Q5K => Kernel::GemvQ5K,
        GgmlType::Q6K => Kernel::GemvQ6K,
        GgmlType::Q8_0 => Kernel::GemvQ8_0,
        GgmlType::Mxfp4 => Kernel::GemvMxfp4,
        other => bail!("{name}: no registered GEMV kernel for packed type {other:?}"),
    })
}

/// The FUSED `mul_mat_vec_id` kernel for a packed-quant expert tensor's GGUF
/// type. Same activation/decode math as [`gemv_kernel_for`]; only the per-expert
/// id-offset push tail differs (the `MUL_MAT_ID` build of `mul_mat_vecq.comp`).
fn gemv_id_kernel_for(weight: &DeviceTensor<'_>, name: &str) -> Result<Kernel> {
    let Residency::KeepQuant(ty) = weight.residency else {
        bail!(
            "{name}: fused expert GEMV expects a packed-quant weight, got {:?}",
            weight.residency
        );
    };
    Ok(match ty {
        GgmlType::Q4K => Kernel::GemvIdQ4K,
        GgmlType::Q5K => Kernel::GemvIdQ5K,
        GgmlType::Q6K => Kernel::GemvIdQ6K,
        GgmlType::Q8_0 => Kernel::GemvIdQ8_0,
        GgmlType::Mxfp4 => Kernel::GemvIdMxfp4,
        other => bail!("{name}: no registered fused expert GEMV for packed type {other:?}"),
    })
}

/// Record (no submit) ONLY the q8_1 quantize of a `[ne]` f32 activation at arena
/// byte `in_off` into the arena slot at `quant_off`. The MoE fused path quantizes
/// the shared `mlp_in` once (gate+up read it) and the post-swiglu `act_all`
/// once (the down dispatch reads it per-expert) — separate from the GEMV, since
/// one quantize feeds two/eight expert dispatches.
fn record_quantize<'a>(
    ctx: &'a VulkanContext,
    res: &mut DecodeResources<'a>,
    name: &str,
    ne: usize,
    in_off: u64,
    quant_off: u64,
) -> Result<()> {
    let quant_bytes = q8_1_x4_bytes(ne);
    let q_spec = Kernel::QuantizeQ8_1.specialization_u32();
    let q_push = q8_1_quantize_params(ne as u32).to_le_bytes();
    let q_groups = {
        let d = q8_1_quantize_dispatch(ne as u32);
        [d.x, d.y, d.z]
    };
    let DecodeResources {
        cache,
        recorder,
        arena,
        ring2,
        ..
    } = res;
    let arena_buf = &arena.buffer;
    let (pipeline, _) = cache
        .get(ctx, Kernel::QuantizeQ8_1, q_spec, q_push.len() as u32, 2)
        .map_err(|e| anyhow!("{name}: build q8_1_quantize pipeline: {e}"))?;
    let set = ring2
        .next_updated(&[
            (arena_buf, in_off, (ne * 4) as u64),
            (arena_buf, quant_off, quant_bytes as u64),
        ])
        .map_err(|e| anyhow!("{name}: bind q8_1 ring set: {e}"))?;
    recorder.label_next("quant");
    recorder.dispatch_raw(pipeline, set, &q_push, q_groups);
    Ok(())
}

/// Record (no submit) ONE fused MoE expert GEMV (`mul_mat_vec_id`) into the OPEN
/// recorder: every selected expert's `[nrows]` projection of the per-expert q8_1
/// activation against its slice of the stacked `exps` weight, written back-to-back
/// into the arena `dst` slot (expert slot `i` → rows `[i*nrows ..]`). The caller
/// supplies the already-recorded q8_1 activation (`b_off`, with `n_experts` rows
/// of `ncols` each when each expert has its own activation — for gate/up there is
/// ONE shared row and `b_stride`=0-style via `ne11=1`; the params encode it).
///
/// `n_act_rows` is the number of `ncols`-wide q8_1 activation rows present at
/// `b_off` (1 when every expert shares one activation — gate/up; `n_experts`
/// when each expert has its own — down). It must equal `gemv_id_params`' `ne11`
/// for the per-expert `b_offset` math to be in range; the params are derived
/// here from it so they stay consistent.
///
/// `ids_off` is the arena byte offset of the i32 `[n_experts]` id list. Bindings
/// `[A exps weight, B q8_1 act, D f32 dst, Fuse0, Fuse1, IDS]`.
#[allow(clippy::too_many_arguments)]
fn record_gemv_id<'a>(
    ctx: &'a VulkanContext,
    res: &mut DecodeResources<'a>,
    weight: &DeviceTensor<'_>,
    name: &str,
    ncols: usize,
    nrows: usize,
    n_experts: usize,
    n_act_rows: usize,
    b_off: u64,
    dst_off: u64,
    ids_off: u64,
) -> Result<()> {
    let kernel = gemv_id_kernel_for(weight, name)?;
    let spec = kernel.specialization_u32();
    let mut push = gemv_id_params(ncols as u32, nrows as u32, n_experts as u32);
    // Override ne11 (word index 9) with the actual activation-row count so the
    // shader's `b_offset = (expert_i0 % ne11) * stride_b` indexes the right row:
    // shared activation (1 row) for gate/up, per-expert (n_experts rows) for down.
    let mut words = push.words().to_vec();
    words[9] = n_act_rows as u32;
    push = KernelParams::from_words(words);
    let push = push.to_le_bytes();
    let groups = {
        let d = gemv_id_dispatch(nrows as u32, n_experts as u32);
        [d.x, d.y, d.z]
    };
    let quant_bytes = q8_1_x4_bytes(n_act_rows * ncols);
    let dst_bytes = n_experts * nrows * std::mem::size_of::<f32>();
    let ids_bytes = (n_experts * std::mem::size_of::<i32>()) as u64;
    let fuse0 = (res.arena.fuse0.offset, res.arena.fuse0.len as u64);
    let fuse1 = (res.arena.fuse1.offset, res.arena.fuse1.len as u64);

    let DecodeResources {
        cache,
        recorder,
        arena,
        ring6,
        ..
    } = res;
    let arena_buf = &arena.buffer;
    let (pipeline, _) = cache
        .get(ctx, kernel, spec, push.len() as u32, 6)
        .map_err(|e| anyhow!("{name}: build {kernel:?} fused expert GEMV pipeline: {e}"))?;
    let set = ring6
        .next_updated(&[
            // binding 0: the WHOLE stacked expert weight (the shader slices by id).
            (&weight.buffer, 0, weight.buffer.len() as u64),
            // binding 1: q8_1 activation (1 or n_experts rows of ncols).
            (arena_buf, b_off, quant_bytes as u64),
            // binding 2: f32 dst (n_experts * nrows).
            (arena_buf, dst_off, dst_bytes as u64),
            // bindings 3/4: fuse dummies (unread, fusion_flags=0).
            (arena_buf, fuse0.0, fuse0.1),
            (arena_buf, fuse1.0, fuse1.1),
            // binding 5: the i32 expert-id list.
            (arena_buf, ids_off, ids_bytes),
        ])
        .map_err(|e| anyhow!("{name}: bind fused expert GEMV ring set: {e}"))?;
    recorder.label_next("gemv");
    recorder.dispatch_raw(pipeline, set, &push, groups);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// On-device elementwise / norm kernels (perf-parity Step 5b). RMSNorm, SwiGLU,
// and residual-Add now run on the GPU through the already-compiled `rms_norm`,
// `swiglu`, and `add` shaders, recorded through the persistent recorder + rings
// against arena slots. Each replaces its host f32 counterpart only after a
// device test confirms it matches the oracle within tolerance.
//
// Two layers: `record_*` (record into the OPEN recorder, no submit — for the
// fused FFN that chains several ops device-resident before one submit) and the
// standalone `*_device` (write host input -> begin -> record -> submit -> read
// back — the simple, oracle-gateable form used where the surrounding data still
// lives on host, e.g. the final norm and the MoE per-expert swiglu).
// ─────────────────────────────────────────────────────────────────────────────

/// Record (no submit) a plain RMSNorm `out[i] = in[i] * inv_rms * w[i]` over a
/// `[ncols]` row: input from arena byte `in_off`, weight from the device tensor
/// `w_buf` sub-range, output to arena byte `out_off`. Bindings 0=A(in), 1=B(w),
/// 2=D(out); spec `do_multiply=1`; push = [`rms_norm_params`]; one workgroup.
fn record_rms_norm<'a>(
    ctx: &'a VulkanContext,
    res: &mut DecodeResources<'a>,
    w_buf: &DeviceBuffer<'_>,
    name: &str,
    ncols: usize,
    eps: f32,
    in_off: u64,
    out_off: u64,
) -> Result<()> {
    let spec = Kernel::RmsNorm.specialization_u32();
    let push = rms_norm_params(ncols as u32, eps).to_le_bytes();
    let groups = {
        let d = rms_norm_dispatch();
        [d.x, d.y, d.z]
    };
    let row = (ncols * 4) as u64;
    let DecodeResources {
        cache,
        recorder,
        arena,
        ring3,
        ..
    } = res;
    let arena_buf = &arena.buffer;
    let (pipeline, _) = cache
        .get(ctx, Kernel::RmsNorm, spec, push.len() as u32, 3)
        .map_err(|e| anyhow!("{name}: build rms_norm pipeline: {e}"))?;
    let set = ring3
        .next_updated(&[
            (arena_buf, in_off, row),
            (w_buf, 0, w_buf.len() as u64),
            (arena_buf, out_off, row),
        ])
        .map_err(|e| anyhow!("{name}: bind rms_norm ring set: {e}"))?;
    recorder.label_next("norm");
    recorder.dispatch_raw(pipeline, set, &push, groups);
    Ok(())
}

/// Record (no submit) a SwiGLU `out[i] = silu(gate[i]) * up[i]` over `[n]`:
/// gate from arena byte `gate_off`, up from `up_off`, output to `out_off`.
/// Bindings 0=A(gate), 1=B(up), 2=D(out); mode=2 (split); push = [`swiglu_params`].
fn record_swiglu<'a>(
    ctx: &'a VulkanContext,
    res: &mut DecodeResources<'a>,
    name: &str,
    n: usize,
    gate_off: u64,
    up_off: u64,
    out_off: u64,
) -> Result<()> {
    let spec = Kernel::SwiGlu.specialization_u32();
    let push = swiglu_params(n as u32).to_le_bytes();
    let groups = {
        let d = swiglu_dispatch(n as u32);
        [d.x, d.y, d.z]
    };
    let row = (n * 4) as u64;
    let DecodeResources {
        cache,
        recorder,
        arena,
        ring3,
        ..
    } = res;
    let arena_buf = &arena.buffer;
    let (pipeline, _) = cache
        .get(ctx, Kernel::SwiGlu, spec, push.len() as u32, 3)
        .map_err(|e| anyhow!("{name}: build swiglu pipeline: {e}"))?;
    let set = ring3
        .next_updated(&[
            (arena_buf, gate_off, row),
            (arena_buf, up_off, row),
            (arena_buf, out_off, row),
        ])
        .map_err(|e| anyhow!("{name}: bind swiglu ring set: {e}"))?;
    recorder.label_next("swiglu");
    recorder.dispatch_raw(pipeline, set, &push, groups);
    Ok(())
}

/// Record (no submit) a residual Add `out[i] = a[i] + b[i]` over `[n]`: a from
/// arena byte `a_off`, b from `b_off`, output to `out_off`. Bindings 0=A, 1=B,
/// 2=D. The shader's binding-3 `PartialBuf` (the optional ADD_RMS reduction
/// target) is dead-code-eliminated by `glslc -O` when built with `ADD_RMS=0`, so
/// the pipeline has exactly 3 bindings — same `ring3` as rms_norm / swiglu.
fn record_add<'a>(
    ctx: &'a VulkanContext,
    res: &mut DecodeResources<'a>,
    name: &str,
    n: usize,
    a_off: u64,
    b_off: u64,
    out_off: u64,
) -> Result<()> {
    let spec = Kernel::Add.specialization_u32();
    let push = add_params(n as u32).to_le_bytes();
    let groups = {
        let d = add_dispatch(n as u32);
        [d.x, d.y, d.z]
    };
    let row = (n * 4) as u64;
    let DecodeResources {
        cache,
        recorder,
        arena,
        ring3,
        ..
    } = res;
    let arena_buf = &arena.buffer;
    let (pipeline, _) = cache
        .get(ctx, Kernel::Add, spec, push.len() as u32, 3)
        .map_err(|e| anyhow!("{name}: build add pipeline: {e}"))?;
    let set = ring3
        .next_updated(&[
            (arena_buf, a_off, row),
            (arena_buf, b_off, row),
            (arena_buf, out_off, row),
        ])
        .map_err(|e| anyhow!("{name}: bind add ring set: {e}"))?;
    recorder.label_next("add");
    recorder.dispatch_raw(pipeline, set, &push, groups);
    Ok(())
}

/// Record (no submit) a NeoX RoPE over ONE head vector `[head_dim]` at arena byte
/// `in_off`, writing to `out_off` (may alias `in_off` — the shader reads the pair
/// `(x[d], x[d+n_dims/2])` before writing both). `nrows=1`, single absolute `pos`.
/// Bindings `[0=X input f32, 1=Y pos int (dummy 1-int slot), 2=Z freq (dummy),
/// 3=D output f32, 4=I indices (dummy)]`. The `pos` lands in the dummy int slot.
/// Oracle-gated by `rope_neox_matches_host_oracle`.
#[allow(clippy::too_many_arguments)]
fn record_rope_neox<'a>(
    ctx: &'a VulkanContext,
    res: &mut DecodeResources<'a>,
    name: &str,
    head_dim: usize,
    rotary_dim: usize,
    pos: usize,
    theta: f32,
    in_off: u64,
    out_off: u64,
) -> Result<()> {
    let push = rope_neox_params(head_dim as u32, rotary_dim as u32, 1, theta).to_le_bytes();
    let groups = {
        let d = rope_neox_dispatch(rotary_dim as u32, 1);
        [d.x, d.y, d.z]
    };
    let row = (head_dim * 4) as u64;
    // The position buffer (binding 1) is read as `rope_data_pos[i2]` with i2=0
    // (ne02=1), so it needs a single i32 = pos. Stage it into the pos slot.
    let pos_off = res.arena.attn_pos.offset;
    res.arena
        .buffer
        .copy_from_host_at(pos_off, &(pos as i32).to_le_bytes())
        .map_err(|e| anyhow!("{name}: write rope pos: {e}"))?;
    let fuse0 = (res.arena.fuse0.offset, res.arena.fuse0.len as u64);
    let fuse1 = (res.arena.fuse1.offset, res.arena.fuse1.len as u64);
    let DecodeResources {
        cache,
        recorder,
        arena,
        ring5,
        ..
    } = res;
    let arena_buf = &arena.buffer;
    let (pipeline, _) = cache
        .get(
            ctx,
            Kernel::RopeNeox,
            Kernel::RopeNeox.specialization_u32(),
            push.len() as u32,
            5,
        )
        .map_err(|e| anyhow!("{name}: build rope_neox pipeline: {e}"))?;
    let _ = fuse1;
    let set = ring5
        .next_updated(&[
            (arena_buf, in_off, row),      // 0: X input
            (arena_buf, pos_off, 8),       // 1: Y pos (i32 at index 0)
            (arena_buf, fuse0.0, fuse0.1), // 2: Z freq (has_ff=0, unread)
            (arena_buf, out_off, row),     // 3: D output
            (arena_buf, pos_off, 8),       // 4: I indices uvec2 (set_rows_stride=0, unread)
        ])
        .map_err(|e| anyhow!("{name}: bind rope_neox ring set: {e}"))?;
    recorder.label_next("rope");
    recorder.dispatch_raw(pipeline, set, &push, groups);
    Ok(())
}

/// Record (no submit) a KV-cached flash-attention for ONE query head: Q `[hd]`
/// f32 at arena byte `q_off`, against the head's `[kv_len, hd]` f16 K/V planes
/// in the device KV cache (`k_plane`/`v_plane` byte offsets), writing the `[hsv]`
/// f32 output to arena byte `o_off`. Single head (gqa_ratio=1), no mask, no
/// ALiBi. Bindings `[0=Q f32, 1=K f16, 2=V f16, 3=M mask (dummy), 4=S sinks
/// (dummy), 5=O f32, 6=MO mask_opt (dummy)]`. The flash pipeline is pinned to a
/// 32-wide subgroup by `Kernel::required_subgroup_size`. Oracle-gated by
/// `flash_attn_matches_host_sdpa_oracle`.
#[allow(clippy::too_many_arguments)]
fn record_flash_attn<'a>(
    ctx: &'a VulkanContext,
    res: &mut DecodeResources<'a>,
    name: &str,
    head_dim: usize,
    kv_len: usize,
    scale: f32,
    q_off: u64,
    k_plane: u64,
    v_plane: u64,
    o_off: u64,
) -> Result<()> {
    let spec = FlashAttentionSpec::f32_f16(head_dim as u32);
    let push =
        flash_attn_params(head_dim as u32, head_dim as u32, kv_len as u32, scale).to_le_bytes();
    let groups = {
        let d = flash_attn_dispatch();
        [d.x, d.y, d.z]
    };
    let q_bytes = (head_dim * 4) as u64;
    let o_bytes = (head_dim * 4) as u64;
    // The cache plane is the whole `[max_seq, head_dim]` f16 region for this head;
    // flash reads only positions `< kv_len`, but bind through the cached length so
    // the descriptor range covers exactly what is read.
    let kv_bytes = (kv_len * head_dim * std::mem::size_of::<u16>()) as u64;
    let fuse0 = (res.arena.fuse0.offset, res.arena.fuse0.len as u64);
    let fuse1 = (res.arena.fuse1.offset, res.arena.fuse1.len as u64);
    let DecodeResources {
        cache,
        recorder,
        arena,
        ring7,
        kv_cache,
        ..
    } = res;
    let arena_buf = &arena.buffer;
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
            (arena_buf, q_off, q_bytes),          // 0: Q f32
            (kv_buf, k_plane, kv_bytes),          // 1: K f16
            (kv_buf, v_plane, kv_bytes),          // 2: V f16
            (arena_buf, fuse0.0, fuse0.1.max(8)), // 3: M mask (dummy)
            (arena_buf, fuse1.0, fuse1.1.max(8)), // 4: S sinks (dummy)
            (arena_buf, o_off, o_bytes),          // 5: O f32
            (arena_buf, fuse0.0, fuse0.1.max(8)), // 6: MO mask_opt (dummy)
        ])
        .map_err(|e| anyhow!("{name}: bind flash_attn ring set: {e}"))?;
    recorder.label_next("flash");
    recorder.dispatch_raw(pipeline, set, &push, groups);
    Ok(())
}

/// Record (no submit) a per-element sigmoid gate `out[i] = sigmoid(gate[i]) *
/// val[i]` over `[n]`: `gate` from arena byte `gate_off`, `val` from `val_off`,
/// output to `out_off` (may alias `val_off`). Bindings 0=A(gate), 1=B(val),
/// 2=D(out) — same `ring3` as add/rms_norm. Applies the full-attention per-head
/// sigmoid gate device-resident. Oracle-covered by the elementwise sigmoid path.
fn record_sigmoid_mul<'a>(
    ctx: &'a VulkanContext,
    res: &mut DecodeResources<'a>,
    name: &str,
    n: usize,
    gate_off: u64,
    val_off: u64,
    out_off: u64,
) -> Result<()> {
    let spec = Kernel::SigmoidMul.specialization_u32();
    let push = sigmoid_mul_params(n as u32).to_le_bytes();
    let groups = {
        let d = sigmoid_mul_dispatch(n as u32);
        [d.x, d.y, d.z]
    };
    let row = (n * 4) as u64;
    let DecodeResources {
        cache,
        recorder,
        arena,
        ring3,
        ..
    } = res;
    let arena_buf = &arena.buffer;
    let (pipeline, _) = cache
        .get(ctx, Kernel::SigmoidMul, spec, push.len() as u32, 3)
        .map_err(|e| anyhow!("{name}: build sigmoid_mul pipeline: {e}"))?;
    let set = ring3
        .next_updated(&[
            (arena_buf, gate_off, row),
            (arena_buf, val_off, row),
            (arena_buf, out_off, row),
        ])
        .map_err(|e| anyhow!("{name}: bind sigmoid_mul ring set: {e}"))?;
    recorder.label_next("sigmoid");
    recorder.dispatch_raw(pipeline, set, &push, groups);
    Ok(())
}

/// Record (no submit) a device f16 KV-pack of ONE head row: read `head_dim` f32
/// from arena byte `src_off`, write them as f16 into the device KV cache plane at
/// `(full_idx, kv_head, pos)` (`is_v` selects the V block). Bindings 0=A(f32 src),
/// 1=D(f16 dst) — a 2-binding layout, so it shares the decode `ring2` with the
/// q8_1 quantize. Replaces the host readback+convert+UMA-write so the whole
/// full-attention block records into ONE submit. Oracle:
/// `f16_kv_pack_matches_host_rne_oracle`.
#[allow(clippy::too_many_arguments)]
fn record_f16_kv_pack<'a>(
    ctx: &'a VulkanContext,
    res: &mut DecodeResources<'a>,
    name: &str,
    head_dim: usize,
    src_off: u64,
    full_idx: usize,
    kv_head: usize,
    pos: usize,
    is_v: bool,
) -> Result<()> {
    let push = f16_kv_pack_params(head_dim as u32).to_le_bytes();
    let groups = {
        let d = f16_kv_pack_dispatch(head_dim as u32);
        [d.x, d.y, d.z]
    };
    let src_bytes = (head_dim * std::mem::size_of::<f32>()) as u64;
    let DecodeResources {
        cache,
        recorder,
        arena,
        ring2,
        kv_cache,
        ..
    } = res;
    let (dst_off, dst_bytes) = kv_cache.row_dst(full_idx, kv_head, pos, is_v);
    let arena_buf = &arena.buffer;
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
        .next_updated(&[
            (arena_buf, src_off, src_bytes), // 0: f32 src head row
            (kv_buf, dst_off, dst_bytes),    // 1: f16 dst cache row
        ])
        .map_err(|e| anyhow!("{name}: bind f16_kv_pack ring set: {e}"))?;
    recorder.label_next("kvpack");
    recorder.dispatch_raw(pipeline, set, &push, groups);
    Ok(())
}

/// Fetch a per-layer F32-resident norm weight tensor (`attn_q_norm` /
/// `attn_k_norm`), erroring if missing or not F32-resident. The device rms_norm
/// binds its `.buffer` directly (head_dim-wide, broadcast across heads).
pub(crate) fn packed_or_f32_norm<'w>(
    weights: &'w ResidentWeights<'_>,
    layer: usize,
    suffix: &str,
) -> Result<&'w DeviceTensor<'w>> {
    let name = format!("blk.{layer}.{suffix}.weight");
    let w = weights
        .get(&name)
        .ok_or_else(|| anyhow!("missing norm weight {name}"))?;
    if !matches!(w.residency, Residency::DequantF32) {
        bail!(
            "{name}: full-attn norm weight must be F32-resident, got {:?}",
            w.residency
        );
    }
    Ok(w)
}

/// Fetch a per-layer packed-quant weight tensor, erroring if it is missing or
/// F32-resident (the dense FFN weights are always packed quant).
pub(crate) fn packed_layer_weight<'w>(
    weights: &'w ResidentWeights<'_>,
    layer: usize,
    suffix: &str,
) -> Result<&'w DeviceTensor<'w>> {
    let name = format!("blk.{layer}.{suffix}.weight");
    let w = weights
        .get(&name)
        .ok_or_else(|| anyhow!("missing weight {name}"))?;
    if !matches!(w.residency, Residency::KeepQuant(_)) {
        bail!(
            "{name}: dense FFN GEMV expects packed quant, got {:?}",
            w.residency
        );
    }
    Ok(w)
}

/// Record (NO begin/submit) the **fused dense FFN + post-attention norm + both
/// residual adds** device-resident into the caller's OPEN recorder (Step 3
/// residual-resident). Reads the residual stream from arena byte `hid_off` and the
/// attention output from `attn_off`, and writes the updated residual stream back
/// to `hid_off`. The sequence:
///
///   post_sum = hidden + attn_out                        (device Add)
///   mlp_in   = rms_norm(post_sum, post_attention_norm)  (device RMSNorm)
///   gate     = ffn_gate · mlp_in                        (device GEMV)
///   up       = ffn_up   · mlp_in                        (device GEMV)
///   act      = silu(gate) * up                          (device SwiGLU)
///   mlp_out  = ffn_down · act                           (device GEMV)
///   hidden'  = post_sum + mlp_out                        (device Add → hid_off)
///
/// All on-device, no host hop. The caller chains the next layer's input-norm into
/// the same residual-resident stream; only the FINAL `[vocab]` logits read back.
///
/// Arena work-slot plan (4 scratch slots; the live set peaks at {post_sum, mlp_in,
/// gate, up} after the up-proj). A barrier separates every recorded op (consecutive
/// GEMVs also share the single `quant` slot, so the barrier is required):
///   work0 = post_sum (lives the whole FFN, consumed by the final add)
///   work1 = mlp_in → (reused) act
///   work2 = gate   → (reused) mlp_out
///   work3 = up
fn record_fused_dense_ffn<'a>(
    ctx: &'a VulkanContext,
    config: &Qwen35Config,
    weights: &ResidentWeights<'_>,
    res: &mut DecodeResources<'a>,
    layer: usize,
    hid_off: u64,
    attn_off: u64,
) -> Result<()> {
    let h = config.hidden_size;
    let inter = config.intermediate_size;
    let eps = config.rms_norm_eps;
    if inter > res.arena.max_cols {
        bail!(
            "fused_dense_ffn[{layer}]: ffn intermediate {inter} exceeds arena ({})",
            res.arena.max_cols
        );
    }

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
            "fused_dense_ffn[{layer}]: weight dims gate[{gate_in},{gate_out}] \
             down[{down_in},{down_out}] disagree with hidden {h} / inter {inter}"
        );
    }

    let _ = (down_in, down_out); // validated above

    let w0 = res.arena.work[0].offset; // post_sum (persists)
    let w1 = res.arena.work[1].offset; // mlp_in -> act
    let w2 = res.arena.work[2].offset; // gate -> mlp_out
    let w3 = res.arena.work[3].offset; // up

    // post_sum = hidden(hid_off) + attn_out(attn_off) -> work0
    record_add(ctx, res, "ffn_post_add", h, hid_off, attn_off, w0)?;
    res.recorder.barrier();
    // mlp_in = rms_norm(post_sum=work0) -> work1
    record_rms_norm(
        ctx,
        res,
        &post_norm_w.buffer,
        "ffn_post_norm",
        h,
        eps,
        w0,
        w1,
    )?;
    res.recorder.barrier();
    // gate + up both project the SAME `mlp_in` (work1). Quantize it once, then run
    // gate/up as GEMV-only against the shared activation (disjoint dst slots w2/w3,
    // no inter-GEMV barrier) instead of re-quantizing the identical input twice.
    let ffn_quant_off = res.arena.quant.offset;
    record_quantize(ctx, res, "ffn_qin", gate_in, w1, ffn_quant_off)?;
    res.recorder.barrier();
    // gate = ffn_gate · mlp_in(work1) -> work2
    record_gemv_only(
        ctx,
        res,
        gate_w,
        "ffn_gate",
        gate_in,
        gate_out,
        0,
        gate_w.buffer.len() as u64,
        ffn_quant_off,
        w2,
    )?;
    // up = ffn_up · mlp_in(work1) -> work3
    record_gemv_only(
        ctx,
        res,
        up_w,
        "ffn_up",
        gate_in,
        gate_out,
        0,
        up_w.buffer.len() as u64,
        ffn_quant_off,
        w3,
    )?;
    res.recorder.barrier();
    // act = silu(gate=work2) * up(work3) -> work1 (mlp_in now dead)
    record_swiglu(ctx, res, "ffn_swiglu", inter, w2, w3, w1)?;
    res.recorder.barrier();
    // mlp_out = ffn_down · act(work1) -> work2 (gate now dead)
    record_quantize_gemv(
        ctx,
        res,
        down_w,
        "ffn_down",
        down_in,
        down_out,
        0,
        down_w.buffer.len() as u64,
        w1,
        w2,
    )?;
    res.recorder.barrier();
    // hidden' = post_sum(work0) + mlp_out(work2) -> hid_off (residual-resident)
    record_add(ctx, res, "ffn_mlp_add", h, w0, w2, hid_off)?;
    res.gemv_count += 3;
    Ok(())
}

/// Record (no submit) the F32 router / shared-gate GEMV `y[e] = Σ_c W[e,c]·x[c]`
/// (optional sigmoid) into the OPEN recorder. Binding 0 = input (arena `in_off`,
/// `hidden`-wide), 1 = F32 weight buffer `[n_out, hidden]`, 2 = output (arena
/// `out_off`, `n_out`-wide). 3-binding → `ring3`.
#[allow(clippy::too_many_arguments)]
fn record_router_gemv<'a>(
    ctx: &'a VulkanContext,
    res: &mut DecodeResources<'a>,
    w_buf: &DeviceBuffer<'_>,
    name: &str,
    n_out: usize,
    hidden: usize,
    apply_sigmoid: bool,
    in_off: u64,
    out_off: u64,
) -> Result<()> {
    let spec = Kernel::Qwen36RouterGemv.specialization_u32();
    let push = qwen36_router_gemv_params(n_out as u32, hidden as u32, apply_sigmoid).to_le_bytes();
    let groups = {
        let d = qwen36_router_gemv_dispatch(n_out as u32);
        [d.x, d.y, d.z]
    };
    let in_row = (hidden * 4) as u64;
    let out_row = (n_out * 4) as u64;
    let DecodeResources {
        cache,
        recorder,
        arena,
        ring3,
        ..
    } = res;
    let arena_buf = &arena.buffer;
    let (pipeline, _) = cache
        .get(ctx, Kernel::Qwen36RouterGemv, spec, push.len() as u32, 3)
        .map_err(|e| anyhow!("{name}: build router_gemv pipeline: {e}"))?;
    let set = ring3
        .next_updated(&[
            (arena_buf, in_off, in_row),
            (w_buf, 0, w_buf.len() as u64),
            (arena_buf, out_off, out_row),
        ])
        .map_err(|e| anyhow!("{name}: bind router_gemv ring set: {e}"))?;
    recorder.label_next("gemv");
    recorder.dispatch_raw(pipeline, set, &push, groups);
    Ok(())
}

/// Record (no submit) the router top-k: `[n_expert]` logits at `logits_off` →
/// `[top_k]` expert ids (i32) at `ids_off` + `[top_k]` weights (f32) at
/// `weights_off`. Single-thread kernel; all three buffers are arena. 3-binding →
/// `ring3`.
#[allow(clippy::too_many_arguments)]
fn record_router_topk<'a>(
    ctx: &'a VulkanContext,
    res: &mut DecodeResources<'a>,
    name: &str,
    n_expert: usize,
    top_k: usize,
    norm_topk: bool,
    logits_off: u64,
    ids_off: u64,
    weights_off: u64,
) -> Result<()> {
    let spec = Kernel::Qwen36RouterTopk.specialization_u32();
    let push = qwen36_router_topk_params(n_expert as u32, top_k as u32, norm_topk).to_le_bytes();
    let groups = {
        let d = qwen36_router_topk_dispatch();
        [d.x, d.y, d.z]
    };
    let logits_row = (n_expert * 4) as u64;
    let topk_row = (top_k * 4) as u64;
    let DecodeResources {
        cache,
        recorder,
        arena,
        ring3,
        ..
    } = res;
    let arena_buf = &arena.buffer;
    let (pipeline, _) = cache
        .get(ctx, Kernel::Qwen36RouterTopk, spec, push.len() as u32, 3)
        .map_err(|e| anyhow!("{name}: build router_topk pipeline: {e}"))?;
    let set = ring3
        .next_updated(&[
            (arena_buf, logits_off, logits_row),
            (arena_buf, ids_off, topk_row),
            (arena_buf, weights_off, topk_row),
        ])
        .map_err(|e| anyhow!("{name}: bind router_topk ring set: {e}"))?;
    recorder.label_next("topk");
    recorder.dispatch_raw(pipeline, set, &push, groups);
    Ok(())
}

/// Record (no submit) the device-weighted accumulate `acc[i] = (init?0:acc[i]) +
/// Σ_{e<count} weights[e]·src[e*hidden+i]`. `src` is `[count*hidden]` expert-major
/// at `src_off`, `weights` is `[count]` at `weights_off`, `acc` is `[hidden]` at
/// `acc_off`. All arena. 3-binding → `ring3`.
#[allow(clippy::too_many_arguments)]
fn record_weighted_accum<'a>(
    ctx: &'a VulkanContext,
    res: &mut DecodeResources<'a>,
    name: &str,
    hidden: usize,
    count: usize,
    init: bool,
    src_off: u64,
    weights_off: u64,
    acc_off: u64,
) -> Result<()> {
    let spec = Kernel::Qwen36MoeWeightedAccum.specialization_u32();
    let push = qwen36_moe_weighted_accum_params(hidden as u32, count as u32, init).to_le_bytes();
    let groups = {
        let d = qwen36_moe_weighted_accum_dispatch(hidden as u32);
        [d.x, d.y, d.z]
    };
    let src_row = (count * hidden * 4) as u64;
    let weights_row = (count * 4) as u64;
    let acc_row = (hidden * 4) as u64;
    let DecodeResources {
        cache,
        recorder,
        arena,
        ring3,
        ..
    } = res;
    let arena_buf = &arena.buffer;
    let (pipeline, _) = cache
        .get(
            ctx,
            Kernel::Qwen36MoeWeightedAccum,
            spec,
            push.len() as u32,
            3,
        )
        .map_err(|e| anyhow!("{name}: build weighted_accum pipeline: {e}"))?;
    let set = ring3
        .next_updated(&[
            (arena_buf, src_off, src_row),
            (arena_buf, weights_off, weights_row),
            (arena_buf, acc_off, acc_row),
        ])
        .map_err(|e| anyhow!("{name}: bind weighted_accum ring set: {e}"))?;
    recorder.label_next("accum");
    recorder.dispatch_raw(pipeline, set, &push, groups);
    Ok(())
}

/// Record (no submit) the WHOLE MoE FFN for one layer device-resident into the
/// open recorder, mirroring [`record_fused_dense_ffn`]'s residual-resident
/// contract (reads `hid_off` + `attn_off`, writes the residual sum back to
/// `hid_off`) — but routed on-device: router GEMV → top-k → fused expert gather →
/// device-weighted accumulate, plus the shared expert. Routing never returns to
/// host, so the MoE token collapses into the dense path's ONE submit/token.
///
///   post_sum  = hid + attn                              -> work0
///   mlp_in    = rms_norm(post_sum, post_attn_norm)      -> work1
///   logits    = router_gemv(ffn_gate_inp, mlp_in)       -> moe_logits
///   ids,w     = topk(softmax(logits))                   -> moe_ids, moe_weights
///   q_in      = quantize(mlp_in)                        -> moe_in_quant
///   gate/up   = ffn_*_exps[ids]·q_in  (fused id)        -> moe_gate, moe_up
///   act       = silu(gate)*up                           -> moe_gate
///   down      = ffn_down_exps[ids]·quantize(act)        -> moe_down [top_k,hidden]
///   acc       = Σ_e w_e·down[e]   (init)                -> work2
///   (shared)  acc += sigmoid(ffn_gate_inp_shexp·mlp_in)·shexp_swiglu(mlp_in)
///   hidden'   = post_sum + acc                          -> hid_off
fn record_fused_moe_ffn<'a>(
    ctx: &'a VulkanContext,
    config: &Qwen35Config,
    weights: &ResidentWeights<'_>,
    res: &mut DecodeResources<'a>,
    layer: usize,
    hid_off: u64,
    attn_off: u64,
) -> Result<()> {
    let h = config.hidden_size;
    let eps = config.rms_norm_eps;
    let n_expert = config.num_experts;
    let top_k = config.num_experts_per_tok;

    let post_norm_w = weights
        .get(&format!("blk.{layer}.post_attention_norm.weight"))
        .ok_or_else(|| anyhow!("missing blk.{layer}.post_attention_norm.weight"))?;
    let router_w = weights
        .get(&format!("blk.{layer}.ffn_gate_inp.weight"))
        .ok_or_else(|| anyhow!("missing blk.{layer}.ffn_gate_inp.weight"))?;
    let gate_exps = weights
        .get(&format!("blk.{layer}.ffn_gate_exps.weight"))
        .ok_or_else(|| anyhow!("missing blk.{layer}.ffn_gate_exps.weight"))?;
    let up_exps = weights
        .get(&format!("blk.{layer}.ffn_up_exps.weight"))
        .ok_or_else(|| anyhow!("missing blk.{layer}.ffn_up_exps.weight"))?;
    let down_exps = weights
        .get(&format!("blk.{layer}.ffn_down_exps.weight"))
        .ok_or_else(|| anyhow!("missing blk.{layer}.ffn_down_exps.weight"))?;
    let (moe_in, moe_inter) = weight_dims(gate_exps, "ffn_gate_exps")?;
    if moe_in != h {
        bail!("fused_moe_ffn[{layer}]: ffn_gate_exps in-dim {moe_in} != hidden {h}");
    }
    if top_k > res.arena.moe_top_k {
        bail!(
            "fused_moe_ffn[{layer}]: top_k {top_k} exceeds arena cap {}",
            res.arena.moe_top_k
        );
    }
    if moe_inter > res.arena.moe_inter_cap {
        bail!(
            "fused_moe_ffn[{layer}]: expert intermediate {moe_inter} exceeds arena cap {}",
            res.arena.moe_inter_cap
        );
    }
    if !moe_inter.is_multiple_of(Q8_1_X4_VALUES_PER_GROUP as usize) {
        bail!(
            "fused_moe_ffn[{layer}]: expert intermediate {moe_inter} not a multiple of {}",
            Q8_1_X4_VALUES_PER_GROUP
        );
    }

    let w0 = res.arena.work[0].offset; // post_sum (persists to the final add)
    let w1 = res.arena.work[1].offset; // mlp_in (router input + quantize source)
    let acc = res.arena.work[2].offset; // expert accumulator
    let in_quant = res.arena.moe_in_quant.offset;
    let gate_off = res.arena.moe_gate.offset;
    let up_off = res.arena.moe_up.offset;
    let act_quant = res.arena.moe_act_quant.offset;
    let down_off = res.arena.moe_down.offset;
    let ids_off = res.arena.moe_ids.offset;
    let logits_off = res.arena.moe_logits.offset;
    let weights_off = res.arena.moe_weights.offset;
    let shgate_off = res.arena.moe_shgate.offset;

    // post_sum = hid + attn -> work0
    record_add(ctx, res, "moe_post_add", h, hid_off, attn_off, w0)?;
    res.recorder.barrier();
    // mlp_in = rms_norm(post_sum) -> work1
    record_rms_norm(
        ctx,
        res,
        &post_norm_w.buffer,
        "moe_post_norm",
        h,
        eps,
        w0,
        w1,
    )?;
    res.recorder.barrier();
    // router logits = router_gemv(mlp_in) -> moe_logits
    record_router_gemv(
        ctx,
        res,
        &router_w.buffer,
        "moe_router_gemv",
        n_expert,
        h,
        false,
        w1,
        logits_off,
    )?;
    res.recorder.barrier();
    // ids, weights = topk(softmax(logits)) -> moe_ids, moe_weights
    record_router_topk(
        ctx,
        res,
        "moe_topk",
        n_expert,
        top_k,
        config.norm_topk_prob,
        logits_off,
        ids_off,
        weights_off,
    )?;
    res.recorder.barrier();
    // q_in = quantize(mlp_in) — shared by routed gate/up AND the shared expert.
    record_quantize(ctx, res, "moe_qin", h, w1, in_quant)?;
    res.recorder.barrier();
    // gate/up = ffn_*_exps[ids]·q_in (fused id, ne11=1) -> moe_gate / moe_up
    record_gemv_id(
        ctx,
        res,
        gate_exps,
        "ffn_gate_exps",
        h,
        moe_inter,
        top_k,
        1,
        in_quant,
        gate_off,
        ids_off,
    )?;
    record_gemv_id(
        ctx,
        res,
        up_exps,
        "ffn_up_exps",
        h,
        moe_inter,
        top_k,
        1,
        in_quant,
        up_off,
        ids_off,
    )?;
    res.recorder.barrier();
    // act = silu(gate)*up over [top_k*inter] -> moe_gate
    record_swiglu(
        ctx,
        res,
        "moe_swiglu",
        top_k * moe_inter,
        gate_off,
        up_off,
        gate_off,
    )?;
    res.recorder.barrier();
    // q_act = quantize(act) -> moe_act_quant
    record_quantize(ctx, res, "moe_qact", top_k * moe_inter, gate_off, act_quant)?;
    res.recorder.barrier();
    // down = ffn_down_exps[ids]·q_act (fused id, ne11=top_k) -> moe_down
    record_gemv_id(
        ctx,
        res,
        down_exps,
        "ffn_down_exps",
        moe_inter,
        h,
        top_k,
        top_k,
        act_quant,
        down_off,
        ids_off,
    )?;
    res.recorder.barrier();
    // acc = Σ_e weights[e]·down[e]  (init from 0) -> work2
    record_weighted_accum(
        ctx,
        res,
        "moe_routed_accum",
        h,
        top_k,
        true,
        down_off,
        weights_off,
        acc,
    )?;
    res.recorder.barrier();
    let mut gemv_dispatches = 4u64; // router + gate/up/down

    // ── Shared expert: acc += sigmoid(ffn_gate_inp_shexp·mlp_in)·shexp(mlp_in). ──
    if let Some(up_shexp) = weights.get(&format!("blk.{layer}.ffn_up_shexp.weight")) {
        let gate_shexp = weights
            .get(&format!("blk.{layer}.ffn_gate_shexp.weight"))
            .ok_or_else(|| anyhow!("missing blk.{layer}.ffn_gate_shexp.weight"))?;
        let down_shexp = weights
            .get(&format!("blk.{layer}.ffn_down_shexp.weight"))
            .ok_or_else(|| anyhow!("missing blk.{layer}.ffn_down_shexp.weight"))?;
        let (sh_in, sh_inter) = weight_dims(gate_shexp, "ffn_gate_shexp")?;
        if sh_in != h {
            bail!("fused_moe_ffn[{layer}]: ffn_gate_shexp in-dim {sh_in} != hidden {h}");
        }
        if sh_inter > res.arena.moe_inter_cap {
            bail!(
                "fused_moe_ffn[{layer}]: shared intermediate {sh_inter} exceeds arena cap {}",
                res.arena.moe_inter_cap
            );
        }
        // shgate = sigmoid(ffn_gate_inp_shexp · mlp_in) -> moe_shgate (device).
        let shrouter = weights
            .get(&format!("blk.{layer}.ffn_gate_inp_shexp.weight"))
            .ok_or_else(|| {
                anyhow!(
                    "fused_moe_ffn[{layer}]: shared expert present but ffn_gate_inp_shexp missing \
                     (resident MoE needs the on-device gate)"
                )
            })?;
        record_router_gemv(
            ctx,
            res,
            &shrouter.buffer,
            "moe_shgate_gemv",
            1,
            h,
            true,
            w1,
            shgate_off,
        )?;
        res.recorder.barrier();
        // shared gate/up read the SAME q8_1(mlp_in) -> moe_gate / moe_up
        record_gemv_only(
            ctx,
            res,
            gate_shexp,
            "ffn_gate_shexp",
            h,
            sh_inter,
            0,
            gate_shexp.buffer.len() as u64,
            in_quant,
            gate_off,
        )?;
        record_gemv_only(
            ctx,
            res,
            up_shexp,
            "ffn_up_shexp",
            h,
            sh_inter,
            0,
            up_shexp.buffer.len() as u64,
            in_quant,
            up_off,
        )?;
        res.recorder.barrier();
        record_swiglu(
            ctx,
            res,
            "moe_sh_swiglu",
            sh_inter,
            gate_off,
            up_off,
            gate_off,
        )?;
        res.recorder.barrier();
        // y_shared = ffn_down_shexp · quantize(act) -> moe_down[0..hidden]
        record_quantize_gemv(
            ctx,
            res,
            down_shexp,
            "ffn_down_shexp",
            sh_inter,
            h,
            0,
            down_shexp.buffer.len() as u64,
            gate_off,
            down_off,
        )?;
        res.recorder.barrier();
        // acc += shgate · y_shared  (count=1, accumulate into the routed acc)
        record_weighted_accum(
            ctx,
            res,
            "moe_shared_accum",
            h,
            1,
            false,
            down_off,
            shgate_off,
            acc,
        )?;
        res.recorder.barrier();
        gemv_dispatches += 4; // shgate + gate/up/down
    }

    // hidden' = post_sum(work0) + acc(work2) -> hid_off (residual-resident)
    record_add(ctx, res, "moe_mlp_add", h, w0, acc, hid_off)?;
    res.gemv_count += gemv_dispatches;
    Ok(())
}
