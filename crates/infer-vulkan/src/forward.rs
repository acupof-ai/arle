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
//! Perf-parity Steps 3+4 (see `docs/plans/amd-vulkan-perf-parity.md`): the GEMV
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
    BLOCK_Q8_1_BYTES, Kernel, KernelCache, Q8_1_X4_VALUES_PER_GROUP, add_dispatch, add_params,
    gemv_dispatch, gemv_params, q8_1_quantize_dispatch, q8_1_quantize_params, rms_norm_dispatch,
    rms_norm_params, scaled_add_dispatch, scaled_add_params, swiglu_dispatch, swiglu_params,
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

/// Bytes for the q8_1_x4 quantized form of an `ncols`-element activation vector
/// (the shader groups 128 values into one x4 super-block of 4×36 B).
fn q8_1_x4_bytes(ncols: usize) -> usize {
    let num_x4 = ncols.div_ceil(Q8_1_X4_VALUES_PER_GROUP as usize);
    num_x4 * 4 * BLOCK_Q8_1_BYTES
}

/// Round `n` up to the next multiple of `align` (a power-of-two device limit).
fn align_up(n: usize, align: usize) -> usize {
    if align == 0 {
        return n;
    }
    n.div_ceil(align) * align
}

// ─────────────────────────────────────────────────────────────────────────────
// DeviceArena — the per-GEMV scratch as named, offset-aligned sub-ranges of ONE
// wide UMA buffer, allocated once on the model (perf-parity Step 3).
// ─────────────────────────────────────────────────────────────────────────────

/// One `(offset, len)` named slot inside the arena's backing buffer.
#[derive(Clone, Copy)]
struct Slot {
    offset: u64,
    len: usize,
}

/// A single `DeviceLocal|HostVisible|HostCoherent` (UMA) buffer holding all the
/// per-GEMV scratch slots, each aligned to `minStorageBufferOffsetAlignment` so
/// every slot can be bound directly as a storage-buffer descriptor sub-range.
///
/// Allocated **once** on [`crate::model_qwen35::VulkanQwen35Model`] and sized to
/// the widest GEMV the forward will run. This replaces the deleted `GemvScratch`
/// and its per-call `DeviceBuffer::alloc`/`copy_to_host` churn: the host writes
/// the input activation into `x_in` (UMA, no staging) and reads the result back
/// from `dst`, while the quantize/GEMV dispatches read/write the slots in place.
///
/// Perf-parity Step 5b adds named **elementwise work slots** (`work0..work2`,
/// each `max_cols` f32 wide) so the device RMSNorm / SwiGLU / residual-Add can
/// read/write the arena directly. The dense FFN sub-sequence (norm → gate/up →
/// swiglu → down → residual add) chains through these slots **device-resident**:
/// only the FFN's host inputs land once and its single output reads back once,
/// killing the per-GEMV host round-trip that dominated decode.
pub struct DeviceArena<'a> {
    buffer: DeviceBuffer<'a>,
    /// f32 input activation (widest GEMV input cols).
    x_in: Slot,
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
    /// Pre-sized caps so a GEMV can bail loud rather than silently corrupt.
    max_cols: usize,
    max_rows: usize,
}

impl<'a> DeviceArena<'a> {
    /// Allocate the arena sized to the widest GEMV (`max_cols` input width,
    /// `max_rows` output rows), aligning every slot to the device's storage
    /// buffer offset granularity.
    pub fn new(ctx: &'a VulkanContext, max_cols: usize, max_rows: usize) -> Result<Self> {
        let align = ctx.min_storage_buffer_offset_alignment().max(1) as usize;
        let x_in_len = max_cols * std::mem::size_of::<f32>();
        let quant_len = q8_1_x4_bytes(max_cols);
        let dst_len = max_rows * std::mem::size_of::<f32>();
        let fuse_len = 4usize;
        // Elementwise work slots: the widest single-vector activation a fused FFN
        // op touches is `max_cols` f32 (the FFN intermediate). dst can also hold a
        // hidden-wide vector, but max_cols already covers hidden.
        let work_len = max_cols * std::mem::size_of::<f32>();

        let mut cursor = 0u64;
        let mut place = |len: usize| -> Slot {
            let offset = cursor;
            cursor += align_up(len, align) as u64;
            Slot { offset, len }
        };
        let x_in = place(x_in_len);
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
        let total = cursor as usize;

        let buffer = DeviceBuffer::alloc_uma(ctx, total)
            .map_err(|e| anyhow!("alloc GEMV device arena ({total} B): {e}"))?;
        Ok(Self {
            buffer,
            x_in,
            quant,
            dst,
            fuse0,
            fuse1,
            work,
            max_cols,
            max_rows,
        })
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

    /// Read `len` f32 back from work slot `i` (UMA, no staging).
    fn read_work(&self, i: usize, len: usize) -> Result<Vec<f32>> {
        let mut bytes = vec![0u8; len * std::mem::size_of::<f32>()];
        self.buffer
            .copy_to_host_at(self.work[i].offset, &mut bytes[..])
            .map_err(|e| anyhow!("read arena work[{i}]: {e}"))?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
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
    ring2: DescriptorSetRing<'a>,
    ring3: DescriptorSetRing<'a>,
    ring5: DescriptorSetRing<'a>,
    /// Lightweight per-call accumulators (nanoseconds) so a decode loop can
    /// attribute time between the GPU GEMV submits and the surrounding host
    /// prep/readback. Drained + printed via [`Self::take_profile`].
    gemv_submit_ns: u128,
    gemv_other_ns: u128,
    gemv_count: u64,
}

impl<'a> DecodeResources<'a> {
    pub fn new(ctx: &'a VulkanContext, config: &Qwen35Config) -> Result<Self> {
        let (max_cols, max_rows) = widest_gemv(config);
        let arena = DeviceArena::new(ctx, max_cols, max_rows)?;
        let cache = KernelCache::new();
        let recorder =
            CommandRecorder::new(ctx).map_err(|e| anyhow!("create decode CommandRecorder: {e}"))?;

        // One persistent layout + ring per decode binding count. A ring of N sets
        // permits N live dispatches between resets; each GEMV pair / FFN sub-step
        // submits + fence-waits before the next, so a modest ring (16) is ample
        // even with the fused FFN recording several dispatches into one submit.
        const RING_SIZE: usize = 16;
        let mk =
            |binding_count: usize| -> Result<(DescriptorSetLayout<'a>, DescriptorSetRing<'a>)> {
                let layout =
                    DescriptorSetLayout::storage_buffers(ctx, binding_count).map_err(|e| {
                        anyhow!("build descriptor layout ({binding_count} bindings): {e}")
                    })?;
                let ring = DescriptorSetRing::new(ctx, &layout, binding_count, RING_SIZE).map_err(
                    |e| anyhow!("build descriptor ring ({binding_count} bindings): {e}"),
                )?;
                Ok((layout, ring))
            };
        let (l2, ring2) = mk(2)?;
        let (l3, ring3) = mk(3)?;
        let (l5, ring5) = mk(5)?;

        Ok(Self {
            arena,
            cache,
            recorder,
            _layouts: vec![l2, l3, l5],
            ring2,
            ring3,
            ring5,
            gemv_submit_ns: 0,
            gemv_other_ns: 0,
            gemv_count: 0,
        })
    }

    /// Rewind every descriptor-set ring's round-robin cursor. Called once at the
    /// start of each token so its dispatches reuse the rings from slot 0 (the
    /// prior token's submissions have all fence-completed).
    pub fn reset_rings(&mut self) {
        self.ring2.reset();
        self.ring3.reset();
        self.ring5.reset();
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
        // input_layernorm: plain RMSNorm (GGUF weight applied directly).
        let attn_norm = norm_weight(weights, layer, "attn_norm")?;
        let normed = rms_norm_weight(&hidden, &attn_norm, eps);

        let attn_out = match layer_type {
            LayerType::FullAttention => {
                let out = full_attention(
                    ctx, config, weights, res, state, layer, full_idx, &normed, start_pos,
                )?;
                full_idx += 1;
                out
            }
            LayerType::LinearAttention => {
                let out =
                    linear_attention(ctx, config, weights, res, state, layer, linear_idx, &normed)?;
                linear_idx += 1;
                out
            }
        };

        // FFN. The DENSE swiglu MLP (qwen35) records post-attn add + post-norm +
        // gate/up/swiglu/down + the MLP residual add as ONE device-resident
        // barrier-chained submit (Step 5b: no host hop between the GEMVs). The
        // sparse MoE (qwen35moe) keeps the host post-add/norm + per-expert path
        // for now (its swiglu/add fusion is the next chunk).
        if config.is_moe_layer(layer) {
            let post_sum = add_vec(&hidden, &attn_out);
            let post_norm_w = norm_weight(weights, layer, "post_attention_norm")?;
            let mlp_in = rms_norm_weight(&post_sum, &post_norm_w, eps);
            let mlp_out = moe_ffn(ctx, config, weights, res, layer, &mlp_in)?;
            hidden = add_vec(&post_sum, &mlp_out);
        } else {
            hidden = fused_dense_ffn(ctx, config, weights, res, layer, &hidden, &attn_out)?;
        }
    }

    // Final norm (plain RMSNorm, on-device) + LM head GEMV → logits.
    let final_norm = weights
        .get("output_norm.weight")
        .ok_or_else(|| anyhow!("missing output_norm.weight"))?;
    let normed = rms_norm_device(ctx, res, &hidden, final_norm, eps, "final_norm")?;
    let logits = gemv_global(ctx, weights, res, "output.weight", &normed)?;

    state.seq_len += 1;
    Ok(logits)
}

// ─────────────────────────────────────────────────────────────────────────────
// MoE FFN (qwen35moe): softmax router → top-k routed experts + a sigmoid-gated
// shared expert. Transcribed from llama.cpp `build_moe_ffn` (SOFTMAX gating,
// norm_topk) + `llm_build_qwen35moe::build_layer_ffn` (shared expert mix).
// ─────────────────────────────────────────────────────────────────────────────

/// The sparse MoE FFN for one token. Returns the FFN output `[hidden]` to be
/// residual-added by the caller (matching the dense path's `mlp_out`).
///
/// Per llama.cpp:
///   router = ffn_gate_inp · mlp_in           → [n_expert] logits
///   probs  = softmax(router); top-k by prob; weights renormalized (norm_topk)
///   routed = Σ_e w_e · down_e( silu(gate_e·x) * up_e·x )      (e ∈ top-k)
///   shared = down_shexp( silu(gate_shexp·x) * up_shexp·x )
///            gated by sigmoid(ffn_gate_inp_shexp · x) if that router exists
///   ffn_out = routed + shared
///
/// Device-resident (perf-parity Step 5c): the router top-k stays on host (tiny —
/// a softmax over the readback router logits), but every expert's per-element op
/// — gate/up GEMV, SwiGLU, down GEMV, and the `acc += w_e · y_e` accumulate —
/// runs **on the GPU** through the Step-5b `record_swiglu` / `record_scaled_add`
/// helpers, threading arena work slots with **no host hop between the GEMVs**.
/// One submit per expert (and one for the shared expert) batches its ~8
/// barrier-chained dispatches; the accumulator lives in arena `work[0]`
/// device-resident across every submit and reads back **once** at the end.
///
/// Arena work-slot plan (4 slots, the arena's full set):
///   work[0] = acc     — Σ contributions, persists the whole FFN, read back once
///   work[1] = mlp_in  — the FFN input x, persists (every expert + shared reads it)
///   work[2] = gate → (reused) act      — per-expert transient
///   work[3] = up   → (reused) y        — per-expert transient
///
/// The router weight scale `w_e` folds into the accumulate (`record_scaled_add`),
/// not a separate scalar-multiply pass; the down GEMV's linearity makes
/// `acc += w_e · down_e(act)` exact whether the scale is applied before or after.
fn moe_ffn<'a>(
    ctx: &'a VulkanContext,
    config: &Qwen35Config,
    weights: &ResidentWeights<'_>,
    res: &mut DecodeResources<'a>,
    layer: usize,
    mlp_in: &[f32],
) -> Result<Vec<f32>> {
    use crate::model_qwen36::qwen36_topk_routes;

    let h = config.hidden_size;
    if mlp_in.len() != h {
        bail!(
            "moe_ffn[{layer}]: mlp_in width {} != hidden {h}",
            mlp_in.len()
        );
    }

    // ── Router: F32 GEMV [hidden → n_expert]. The router weight is F32-resident
    // (DequantF32), not packed, so the quantized GEMV path does not apply — do
    // the small matvec on the host (n_expert is tiny, e.g. 256) and keep the
    // softmax / top-k on host (the brief: router stays on host). ──
    let router = weights
        .get(&format!("blk.{layer}.ffn_gate_inp.weight"))
        .ok_or_else(|| anyhow!("missing blk.{layer}.ffn_gate_inp.weight"))?;
    let router_logits = gemv_f32_host(router, mlp_in, config.num_experts)?;

    let routes = qwen36_topk_routes(
        &router_logits,
        config.num_experts_per_tok,
        config.norm_topk_prob,
    );

    // Arena work-slot offsets (see the fn doc).
    let acc_off = res.arena.work[0].offset;
    let mlp_in_off = res.arena.work[1].offset;
    let gate_off = res.arena.work[2].offset; // gate -> act
    let up_off = res.arena.work[3].offset; // up   -> y

    // Land mlp_in into its slot once (read by every expert + the shared expert)
    // and zero the accumulator slot so the first scaled-add starts from 0.
    res.arena.write_work(1, mlp_in)?;
    res.arena.write_work(0, &vec![0.0f32; h])?;

    // ── Routed experts: slice each selected expert e out of the 3-D stacked
    // ffn_*_exps tensors by byte offset and run the swiglu MLP device-resident,
    // accumulating `acc += w_e · y_e` on-device. ──
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
        bail!("ffn_gate_exps[{layer}] in-dim {moe_in} != hidden {h}");
    }
    if moe_inter > res.arena.max_cols {
        bail!(
            "moe_ffn[{layer}]: expert intermediate {moe_inter} exceeds arena ({})",
            res.arena.max_cols
        );
    }

    for route in &routes {
        let e = route.expert;
        moe_expert_submit(
            ctx,
            res,
            gate_exps,
            up_exps,
            down_exps,
            e,
            h,
            moe_inter,
            route.weight,
            mlp_in_off,
            gate_off,
            up_off,
            acc_off,
        )?;
    }

    // ── Shared expert (present on qwen35moe): a dense swiglu MLP gated by a
    // per-token sigmoid scalar from ffn_gate_inp_shexp. Recorded device-resident
    // into one submit; the sigmoid scalar folds into its scaled-add. ──
    if let Some(up_shexp) = weights.get(&format!("blk.{layer}.ffn_up_shexp.weight")) {
        let gate_shexp = weights
            .get(&format!("blk.{layer}.ffn_gate_shexp.weight"))
            .ok_or_else(|| anyhow!("missing blk.{layer}.ffn_gate_shexp.weight"))?;
        let down_shexp = weights
            .get(&format!("blk.{layer}.ffn_down_shexp.weight"))
            .ok_or_else(|| anyhow!("missing blk.{layer}.ffn_down_shexp.weight"))?;
        let (sh_in, sh_inter) = weight_dims(gate_shexp, "ffn_gate_shexp")?;
        if sh_in != h {
            bail!("ffn_gate_shexp[{layer}] in-dim {sh_in} != hidden {h}");
        }
        if sh_inter > res.arena.max_cols {
            bail!(
                "moe_ffn[{layer}]: shared expert intermediate {sh_inter} exceeds arena ({})",
                res.arena.max_cols
            );
        }

        // Sigmoid gate: ffn_gate_inp_shexp is F32 [hidden → 1] (one scalar/token).
        // Compute it on host (one F32 dot) and fold the sigmoid into the shared
        // expert's scaled-add (`acc += s · y_shared`); default 1.0 if absent.
        let s = if let Some(shgate) = weights.get(&format!("blk.{layer}.ffn_gate_inp_shexp.weight"))
        {
            sigmoid(gemv_f32_host(shgate, mlp_in, 1)?[0])
        } else {
            1.0
        };

        moe_dense_expert_submit(
            ctx, res, gate_shexp, up_shexp, down_shexp, h, sh_inter, s, mlp_in_off, gate_off,
            up_off, acc_off,
        )?;
    }

    // Read the accumulator back once (the only host hop in the whole MoE FFN).
    res.arena.read_work(0, h)
}

/// Record one routed expert's `acc += w_e · y_e` device-resident into ONE submit:
/// gate/up GEMV (sliced from the stacked `ffn_*_exps` at expert `e`), SwiGLU,
/// down GEMV, then the weighted accumulate — all barrier-chained against arena
/// slots. `mlp_in_off`/`acc_off` are the persistent input/accumulator slots;
/// `gate_off`/`up_off` are reused for the transient act/y.
#[allow(clippy::too_many_arguments)]
fn moe_expert_submit<'a>(
    ctx: &'a VulkanContext,
    res: &mut DecodeResources<'a>,
    gate_exps: &DeviceTensor<'_>,
    up_exps: &DeviceTensor<'_>,
    down_exps: &DeviceTensor<'_>,
    e: usize,
    h: usize,
    inter: usize,
    weight: f32,
    mlp_in_off: u64,
    gate_off: u64,
    up_off: u64,
    acc_off: u64,
) -> Result<()> {
    let (g_off, g_len) = expert_slice(gate_exps, e, "ffn_gate_exps")?;
    let (u_off, u_len) = expert_slice(up_exps, e, "ffn_up_exps")?;
    let (d_off, d_len) = expert_slice(down_exps, e, "ffn_down_exps")?;

    let t_start = std::time::Instant::now();
    res.recorder
        .begin()
        .map_err(|er| anyhow!("ffn_exps[{e}]: recorder begin: {er}"))?;

    // gate = gate_e · mlp_in -> gate_off
    record_quantize_gemv(
        ctx,
        res,
        gate_exps,
        "ffn_gate_exps",
        h,
        inter,
        g_off,
        g_len,
        mlp_in_off,
        gate_off,
    )?;
    res.recorder.barrier();
    // up = up_e · mlp_in -> up_off
    record_quantize_gemv(
        ctx,
        res,
        up_exps,
        "ffn_up_exps",
        h,
        inter,
        u_off,
        u_len,
        mlp_in_off,
        up_off,
    )?;
    res.recorder.barrier();
    // act = silu(gate) * up -> gate_off (gate now dead)
    record_swiglu(
        ctx,
        res,
        "ffn_exps_swiglu",
        inter,
        gate_off,
        up_off,
        gate_off,
    )?;
    res.recorder.barrier();
    // y = down_e · act -> up_off (up now dead)
    record_quantize_gemv(
        ctx,
        res,
        down_exps,
        "ffn_down_exps",
        inter,
        h,
        d_off,
        d_len,
        gate_off,
        up_off,
    )?;
    res.recorder.barrier();
    // acc += w_e · y -> acc_off
    record_scaled_add(
        ctx,
        res,
        "ffn_exps_acc",
        h,
        weight,
        acc_off,
        up_off,
        acc_off,
    )?;

    let t_submit = std::time::Instant::now();
    res.recorder
        .submit_and_wait()
        .map_err(|er| anyhow!("ffn_exps[{e}]: submit: {er}"))?;
    let submit_ns = t_submit.elapsed().as_nanos();
    let total_ns = t_start.elapsed().as_nanos();
    res.gemv_submit_ns += submit_ns;
    res.gemv_other_ns += total_ns.saturating_sub(submit_ns);
    res.gemv_count += 3;
    Ok(())
}

/// Record the shared (dense) expert's `acc += s · y_shared` device-resident into
/// ONE submit: gate/up/swiglu/down over the whole `ffn_*_shexp` tensors, then the
/// sigmoid-gated accumulate. `s` is the per-token sigmoid scalar (1.0 if no
/// shared-router weight); it folds into the scaled-add.
#[allow(clippy::too_many_arguments)]
fn moe_dense_expert_submit<'a>(
    ctx: &'a VulkanContext,
    res: &mut DecodeResources<'a>,
    gate_w: &DeviceTensor<'_>,
    up_w: &DeviceTensor<'_>,
    down_w: &DeviceTensor<'_>,
    h: usize,
    inter: usize,
    s: f32,
    mlp_in_off: u64,
    gate_off: u64,
    up_off: u64,
    acc_off: u64,
) -> Result<()> {
    let g_len = gate_w.buffer.len() as u64;
    let u_len = up_w.buffer.len() as u64;
    let d_len = down_w.buffer.len() as u64;

    let t_start = std::time::Instant::now();
    res.recorder
        .begin()
        .map_err(|er| anyhow!("ffn_shexp: recorder begin: {er}"))?;

    record_quantize_gemv(
        ctx,
        res,
        gate_w,
        "ffn_gate_shexp",
        h,
        inter,
        0,
        g_len,
        mlp_in_off,
        gate_off,
    )?;
    res.recorder.barrier();
    record_quantize_gemv(
        ctx,
        res,
        up_w,
        "ffn_up_shexp",
        h,
        inter,
        0,
        u_len,
        mlp_in_off,
        up_off,
    )?;
    res.recorder.barrier();
    record_swiglu(
        ctx,
        res,
        "ffn_shexp_swiglu",
        inter,
        gate_off,
        up_off,
        gate_off,
    )?;
    res.recorder.barrier();
    record_quantize_gemv(
        ctx,
        res,
        down_w,
        "ffn_down_shexp",
        inter,
        h,
        0,
        d_len,
        gate_off,
        up_off,
    )?;
    res.recorder.barrier();
    record_scaled_add(ctx, res, "ffn_shexp_acc", h, s, acc_off, up_off, acc_off)?;

    let t_submit = std::time::Instant::now();
    res.recorder
        .submit_and_wait()
        .map_err(|er| anyhow!("ffn_shexp: submit: {er}"))?;
    let submit_ns = t_submit.elapsed().as_nanos();
    let total_ns = t_start.elapsed().as_nanos();
    res.gemv_submit_ns += submit_ns;
    res.gemv_other_ns += total_ns.saturating_sub(submit_ns);
    res.gemv_count += 3;
    Ok(())
}

/// Byte `(offset, len)` of expert slice `e` inside a 3-D stacked `ffn_*_exps`
/// tensor `[ne0=in, ne1=out, ne2=n_expert]` (row-major, ne0 fastest): expert `e`
/// is the 2-D `[out, in]` matrix at `e · out · row_bytes(in)`, length
/// `out · row_bytes(in)`. The stride is a multiple of the quant block bytes
/// (Q4_K 144 B / Q5_K 176 B / Q8_0 34 B per block), which clears the AMD storage-
/// buffer offset granularity. Shared with the (now device-resident) MoE recorder.
fn expert_slice(exps: &DeviceTensor<'_>, e: usize, name: &str) -> Result<(u64, u64)> {
    let (ncols, nrows) = weight_dims(exps, name)?;
    let ty = match exps.residency {
        Residency::KeepQuant(ty) => ty,
        other => bail!("{name}: expert GEMV expects packed quant, got {other:?}"),
    };
    let row_bytes = ty
        .row_bytes(ncols)
        .ok_or_else(|| anyhow!("{name}: {ty:?} row of {ncols} cols not block-aligned"))?
        as u64;
    let per_expert = nrows as u64 * row_bytes; // bytes for one [out, in] slice
    let offset = e as u64 * per_expert;
    if offset + per_expert > exps.buffer.len() as u64 {
        bail!(
            "{name}: expert {e} slice [{offset}, +{per_expert}) exceeds buffer {}",
            exps.buffer.len()
        );
    }
    Ok((offset, per_expert))
}

/// Tiny F32 matvec on the host: `y[r] = Σ_c W[r,c] · x[c]` for an F32-resident
/// weight `[ncols=in, nrows=out]` (GGUF row-major `[out, in]`). Used for the MoE
/// routers (`ffn_gate_inp` → n_expert, `ffn_gate_inp_shexp` → 1), which are F32
/// and too small to justify a device round-trip.
fn gemv_f32_host(weight: &DeviceTensor<'_>, x: &[f32], nrows: usize) -> Result<Vec<f32>> {
    if !matches!(weight.residency, Residency::DequantF32) {
        bail!(
            "{}: host F32 matvec expects F32-resident weight, got {:?}",
            weight.name,
            weight.residency
        );
    }
    let ncols = x.len();
    let w = dequant_f32(weight, nrows * ncols)?;
    let mut y = vec![0.0f32; nrows];
    for (r, yr) in y.iter_mut().enumerate() {
        let row = &w[r * ncols..r * ncols + ncols];
        let mut s = 0.0f32;
        for (&wv, &xv) in row.iter().zip(x) {
            s += wv * xv;
        }
        *yr = s;
    }
    Ok(y)
}

// ─────────────────────────────────────────────────────────────────────────────
// Full-attention layer (gated q_proj, per-head q/k norm + NeoX RoPE, causal SDPA)
// ─────────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn full_attention<'a>(
    ctx: &'a VulkanContext,
    config: &Qwen35Config,
    weights: &ResidentWeights<'_>,
    res: &mut DecodeResources<'a>,
    state: &mut Qwen35ForwardState,
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
    let q_full = gemv_layer(ctx, weights, res, layer, "attn_q", normed)?; // 2*q_dim
    let k_in = gemv_layer(ctx, weights, res, layer, "attn_k", normed)?; // kv_dim
    let v_in = gemv_layer(ctx, weights, res, layer, "attn_v", normed)?; // kv_dim

    let q_norm = norm_weight(weights, layer, "attn_q_norm")?; // [hd]
    let k_norm = norm_weight(weights, layer, "attn_k_norm")?; // [hd]

    // Build this token's query heads (q/k norm + RoPE) and the K/V rows.
    // Query layout in q_full per head h: [h*2*hd .. h*2*hd+hd] = query,
    // [h*2*hd+hd .. h*2*hd+2*hd] = gate (consumed after attention).
    let mut q = vec![0.0f32; q_dim];
    for hh in 0..nq {
        let src = &q_full[hh * 2 * hd..hh * 2 * hd + hd];
        let normed_head = rms_norm_weight(src, &q_norm, eps); // plain per-head
        let rotated = rope_neox(&normed_head, pos, half, hd, theta);
        q[hh * hd..hh * hd + hd].copy_from_slice(&rotated);
    }
    let mut k_row = vec![0.0f32; kv_dim];
    for hh in 0..nkv {
        let src = &k_in[hh * hd..hh * hd + hd];
        let normed_head = rms_norm_weight(src, &k_norm, eps);
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
    gemv_layer(ctx, weights, res, layer, "attn_output", &attn)
}

// ─────────────────────────────────────────────────────────────────────────────
// Linear (gated-delta) layer: in-proj → conv1d(SiLU) → recurrent delta → gated
// RMSNorm → out-proj. Transcribed from gated_delta_rule.cu / conv1d.cu.
// ─────────────────────────────────────────────────────────────────────────────

// Several inner loops index sibling arrays by the same loop variable (the
// depthwise conv taps, the gated-delta state passes) — a range loop mirrors the
// CUDA reference's index math one-to-one and is clearer than a zipped iterator.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn linear_attention<'a>(
    ctx: &'a VulkanContext,
    config: &Qwen35Config,
    weights: &ResidentWeights<'_>,
    res: &mut DecodeResources<'a>,
    state: &mut Qwen35ForwardState,
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
    let qkv = gemv_layer(ctx, weights, res, layer, "attn_qkv", normed)?; // [qkv_dim]
    let z = gemv_layer(ctx, weights, res, layer, "attn_gate", normed)?; // [v_dim_total]
    let a_proj = gemv_layer(ctx, weights, res, layer, "ssm_alpha", normed)?; // [nv]
    let b_proj = gemv_layer(ctx, weights, res, layer, "ssm_beta", normed)?; // [nv]

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
        // GQA key-head mapping. llama.cpp expands the 16 key heads to 48 value
        // heads with `ggml_repeat` (qwen35.cpp:326-327, delta-net-base.cpp:362),
        // which TILES rather than block-broadcasts: value head `vh` reads key
        // head `vh % nk`, NOT `vh * nk / nv`. See ggml_compute_forward_repeat_f32
        // (dst dim1 index = i1*ne01 + k1, so src head = dst_head % ne01). Using
        // `vh * nk / nv` scrambled every linear layer's q/k → degenerate output.
        let kh = vh % nk;
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
        // GGUF `ssm_a` already stores A = -exp(A_log) (converter pre-applied the
        // -exp; verified negative on-box). So the log-decay is A*softplus(dt),
        // NOT -exp(A)*softplus(dt). Matches llama.cpp delta-net-base.cpp:341
        // (`g = ggml_exp(g)` where g = softplus * ssm_a, qwen35.cpp:232).
        let exp_g = (a_log[vh] * softplus(a_val + dt_bias[vh])).exp();
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
    gemv_layer(ctx, weights, res, layer, "ssm_out", &normed_out)
}

// ─────────────────────────────────────────────────────────────────────────────
// On-device GEMV helpers (the proven q8_0 path), now recording the quantize+GEMV
// pair into the persistent recorder/cache against arena sub-buffers.
// ─────────────────────────────────────────────────────────────────────────────

/// Run one GEMV `y[out] = W[out,in] · x[in]` for a per-layer weight tensor
/// `blk.{layer}.{suffix}.weight`. Packed-quant weights run on the device (the
/// proven `mul_mat_vecq` path); F32-resident weights (e.g. the MoE 35B-A3B's
/// small `ssm_alpha`/`ssm_beta` `[hidden→32]` projections, which the converter
/// ships as F32 rather than Q8_0) run on the host — they are tiny and avoid a
/// device round-trip and a non-quantized GEMV pipeline.
fn gemv_layer<'a>(
    ctx: &'a VulkanContext,
    weights: &ResidentWeights<'_>,
    res: &mut DecodeResources<'a>,
    layer: usize,
    suffix: &str,
    x: &[f32],
) -> Result<Vec<f32>> {
    let name = format!("blk.{layer}.{suffix}.weight");
    let w = weights
        .get(&name)
        .ok_or_else(|| anyhow!("missing weight {name}"))?;
    if matches!(w.residency, Residency::DequantF32) {
        let nrows = weight_dims(w, &name)?.1;
        return gemv_f32_host(w, x, nrows);
    }
    gemv_device(ctx, w, res, x, &name)
}

/// Run a GEMV for a global (non-layer) weight tensor.
fn gemv_global<'a>(
    ctx: &'a VulkanContext,
    weights: &ResidentWeights<'_>,
    res: &mut DecodeResources<'a>,
    name: &str,
    x: &[f32],
) -> Result<Vec<f32>> {
    let w = weights
        .get(name)
        .ok_or_else(|| anyhow!("missing weight {name}"))?;
    gemv_device(ctx, w, res, x, name)
}

/// Core GEMV: write `x` into the arena's input slot, record the q8_1 quantize +
/// Q8_0 GEMV pair (with a barrier between) into the persistent `CommandRecorder`
/// from compile-once cached pipelines bound to the arena/weight sub-buffers,
/// submit the pair ONCE, then read back `[nrows]` f32 from the arena dst slot.
///
/// No per-call `DeviceBuffer::alloc`, no per-op `queue_wait_idle`, no
/// per-dispatch pipeline rebuild — the three perf-parity wins. `nrows`/`ncols`
/// come from the weight's GGUF `gemv_dims` (`[in, out]`).
fn gemv_device<'a>(
    ctx: &'a VulkanContext,
    weight: &DeviceTensor<'_>,
    res: &mut DecodeResources<'a>,
    x: &[f32],
    name: &str,
) -> Result<Vec<f32>> {
    let (ncols, nrows) = weight_dims(weight, name)?;
    // Whole 2-D weight: bind from byte 0 over the full buffer.
    gemv_device_at(
        ctx,
        weight,
        res,
        x,
        name,
        ncols,
        nrows,
        0,
        weight.buffer.len() as u64,
    )
}

/// GEMV against a sub-range of a (possibly 3-D, expert-stacked) packed-quant
/// weight buffer: the matrix is `[nrows=out, ncols=in]` starting at
/// `weight_offset` bytes (length `weight_len` bytes). For a whole 2-D tensor,
/// `weight_offset = 0` and the dims come from `gemv_dims`; for one expert slice
/// `e` of an `[in, out, n_expert]` GGUF tensor, `weight_offset = e * out *
/// row_bytes(in)`. `weight_offset` must be a multiple of the device's
/// `minStorageBufferOffsetAlignment` (caller-checked: every K-quant expert
/// stride is a multiple of its block bytes, which clears the AMD 16/32 B
/// granularity — see the MoE FFN caller).
#[allow(clippy::too_many_arguments)]
fn gemv_device_at<'a>(
    ctx: &'a VulkanContext,
    weight: &DeviceTensor<'_>,
    res: &mut DecodeResources<'a>,
    x: &[f32],
    name: &str,
    ncols: usize,
    nrows: usize,
    weight_offset: u64,
    weight_len: u64,
) -> Result<Vec<f32>> {
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
    if weight_offset + weight_len > weight.buffer.len() as u64 {
        bail!(
            "{name}: GEMV weight sub-range [{weight_offset}, +{weight_len}) exceeds buffer {}",
            weight.buffer.len()
        );
    }
    if ncols > res.arena.max_cols {
        bail!(
            "{name}: GEMV ncols {ncols} exceeds pre-sized arena ({})",
            res.arena.max_cols
        );
    }
    if nrows > res.arena.max_rows {
        bail!(
            "{name}: GEMV nrows {nrows} exceeds pre-sized arena ({})",
            res.arena.max_rows
        );
    }

    let t_start = std::time::Instant::now();

    // 1. Land the input activation into the arena's x_in slot (UMA, no alloc).
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let x_in_off = res.arena.x_in.offset;
    res.arena
        .buffer
        .copy_from_host_at(x_in_off, &x_bytes)
        .map_err(|e| anyhow!("{name}: write gemv x into arena: {e}"))?;

    // 2. Record quantize -> barrier -> GEMV into ONE submit, reading x from the
    //    x_in slot and writing the result into the dst slot.
    res.recorder
        .begin()
        .map_err(|e| anyhow!("{name}: recorder begin: {e}"))?;
    let dst_off = res.arena.dst.offset;
    record_quantize_gemv(
        ctx,
        res,
        weight,
        name,
        ncols,
        nrows,
        weight_offset,
        weight_len,
        x_in_off,
        dst_off,
    )?;

    // 3. ONE submit + ONE fence wait for the quantize+GEMV pair.
    let t_submit = std::time::Instant::now();
    res.recorder
        .submit_and_wait()
        .map_err(|e| anyhow!("{name}: submit gemv pair: {e}"))?;
    let submit_ns = t_submit.elapsed().as_nanos();

    // 4. Read back the f32 result rows from the arena dst slot.
    let dst_bytes = nrows * std::mem::size_of::<f32>();
    let mut out_bytes = vec![0u8; dst_bytes];
    res.arena
        .buffer
        .copy_to_host_at(res.arena.dst.offset, &mut out_bytes[..])
        .map_err(|e| anyhow!("{name}: read back gemv dst from arena: {e}"))?;
    let out: Vec<f32> = out_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let total_ns = t_start.elapsed().as_nanos();
    res.gemv_submit_ns += submit_ns;
    res.gemv_other_ns += total_ns.saturating_sub(submit_ns);
    res.gemv_count += 1;
    Ok(out)
}

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
        recorder.dispatch_raw(pipeline, set, &g_push, g_groups);
    }
    Ok(())
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
        other => bail!("{name}: no registered GEMV kernel for packed type {other:?}"),
    })
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
    recorder.dispatch_raw(pipeline, set, &push, groups);
    Ok(())
}

/// Record (no submit) a scaled residual Add `out[i] = a[i] + scale * b[i]` over
/// `[n]`: accumulator `a` from arena byte `a_off`, addend `b` from `b_off`,
/// output to `out_off` (may alias `a_off` — the shader reads both inputs at index
/// `i` before writing `out[i]`, so in-place accumulate is safe). Bindings
/// 0=A(acc), 1=B(addend), 2=D(out) — same `ring3` as add/rms_norm/swiglu.
///
/// Folds the MoE router weight `scale = w_e` into the per-expert accumulate
/// (`acc += w_e * y_e`) so the whole MoE accumulate stays device-resident.
fn record_scaled_add<'a>(
    ctx: &'a VulkanContext,
    res: &mut DecodeResources<'a>,
    name: &str,
    n: usize,
    scale: f32,
    a_off: u64,
    b_off: u64,
    out_off: u64,
) -> Result<()> {
    let spec = Kernel::ScaledAdd.specialization_u32();
    let push = scaled_add_params(n as u32, scale).to_le_bytes();
    let groups = {
        let d = scaled_add_dispatch(n as u32);
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
        .get(ctx, Kernel::ScaledAdd, spec, push.len() as u32, 3)
        .map_err(|e| anyhow!("{name}: build scaled_add pipeline: {e}"))?;
    let set = ring3
        .next_updated(&[
            (arena_buf, a_off, row),
            (arena_buf, b_off, row),
            (arena_buf, out_off, row),
        ])
        .map_err(|e| anyhow!("{name}: bind scaled_add ring set: {e}"))?;
    recorder.dispatch_raw(pipeline, set, &push, groups);
    Ok(())
}

/// Standalone device RMSNorm: write host `x` into a work slot, record + submit +
/// read back. Oracle-gated drop-in for [`rms_norm_weight`] where the surrounding
/// data lives on host (the final norm). `w` must be an F32-resident norm weight.
fn rms_norm_device<'a>(
    ctx: &'a VulkanContext,
    res: &mut DecodeResources<'a>,
    x: &[f32],
    w: &DeviceTensor<'_>,
    eps: f32,
    name: &str,
) -> Result<Vec<f32>> {
    let n = x.len();
    if !matches!(w.residency, Residency::DequantF32) {
        bail!(
            "{name}: rms_norm weight must be F32-resident, got {:?}",
            w.residency
        );
    }
    let in_off = res.arena.work[0].offset;
    let out_off = res.arena.work[1].offset;
    res.arena.write_work(0, x)?;
    res.recorder
        .begin()
        .map_err(|e| anyhow!("{name}: recorder begin: {e}"))?;
    record_rms_norm(ctx, res, &w.buffer, name, n, eps, in_off, out_off)?;
    res.recorder
        .submit_and_wait()
        .map_err(|e| anyhow!("{name}: submit rms_norm: {e}"))?;
    res.arena.read_work(1, n)
}

/// Fetch a per-layer packed-quant weight tensor, erroring if it is missing or
/// F32-resident (the dense FFN weights are always packed quant).
fn packed_layer_weight<'w>(
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

/// The **fused dense FFN + post-attention norm + both residual adds**, recorded
/// device-resident (perf-parity Step 5b). Given the layer's residual stream
/// `hidden` and the attention output `attn_out` (both host f32 from the
/// host-side attention this chunk does NOT port), this records — into ONE
/// submit, threading arena slots with barriers and **no host round-trip between
/// the GEMVs** — the sequence:
///
///   post_sum = hidden + attn_out                        (device Add)
///   mlp_in   = rms_norm(post_sum, post_attention_norm)  (device RMSNorm)
///   gate     = ffn_gate · mlp_in                        (device GEMV)
///   up       = ffn_up   · mlp_in                        (device GEMV)
///   act      = silu(gate) * up                          (device SwiGLU)
///   mlp_out  = ffn_down · act                           (device GEMV)
///   hidden'  = post_sum + mlp_out                        (device Add)
///
/// and reads back ONLY `hidden'` ([hidden] f32) at the end. This removes the
/// six device→host→device hops (two adds, one norm, swiglu, plus the per-GEMV
/// readbacks of gate/up/down) the host path forced per dense layer.
///
/// Arena work-slot plan (4 slots; the live set peaks at {post_sum, mlp_in, gate,
/// up} after the up-proj). Every slot is `max_cols`-wide so it holds a hidden-
/// or ffn_inter-wide vector. A barrier separates every recorded op (consecutive
/// GEMVs also share the single `quant` slot, so the barrier is required):
///   work0 = post_sum (lives the whole FFN, consumed by the final add)
///   work1 = mlp_in → (reused) act → (reused) hidden'
///   work2 = gate   → (reused) mlp_out
///   work3 = up
fn fused_dense_ffn<'a>(
    ctx: &'a VulkanContext,
    config: &Qwen35Config,
    weights: &ResidentWeights<'_>,
    res: &mut DecodeResources<'a>,
    layer: usize,
    hidden: &[f32],
    attn_out: &[f32],
) -> Result<Vec<f32>> {
    let h = config.hidden_size;
    let inter = config.intermediate_size;
    let eps = config.rms_norm_eps;
    if hidden.len() != h || attn_out.len() != h {
        bail!(
            "fused_dense_ffn[{layer}]: hidden {} / attn_out {} != hidden {h}",
            hidden.len(),
            attn_out.len()
        );
    }
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

    let t_start = std::time::Instant::now();

    let w0 = res.arena.work[0].offset; // post_sum (persists)
    let w1 = res.arena.work[1].offset; // mlp_in -> act -> hidden'
    let w2 = res.arena.work[2].offset; // gate -> mlp_out
    let w3 = res.arena.work[3].offset; // up

    // Land the two add inputs into scratch slots work1 (=hidden) / work2 (=attn).
    res.arena.write_work(1, hidden)?;
    res.arena.write_work(2, attn_out)?;

    res.recorder
        .begin()
        .map_err(|e| anyhow!("fused_dense_ffn[{layer}]: recorder begin: {e}"))?;

    // post_sum = hidden(work1) + attn_out(work2) -> work0
    record_add(ctx, res, "ffn_post_add", h, w1, w2, w0)?;
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
    // gate = ffn_gate · mlp_in(work1) -> work2
    record_quantize_gemv(
        ctx,
        res,
        gate_w,
        "ffn_gate",
        gate_in,
        gate_out,
        0,
        gate_w.buffer.len() as u64,
        w1,
        w2,
    )?;
    res.recorder.barrier();
    // up = ffn_up · mlp_in(work1) -> work3
    record_quantize_gemv(
        ctx,
        res,
        up_w,
        "ffn_up",
        gate_in,
        gate_out,
        0,
        up_w.buffer.len() as u64,
        w1,
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
    // hidden' = post_sum(work0) + mlp_out(work2) -> work1
    record_add(ctx, res, "ffn_mlp_add", h, w0, w2, w1)?;

    let t_submit = std::time::Instant::now();
    res.recorder
        .submit_and_wait()
        .map_err(|e| anyhow!("fused_dense_ffn[{layer}]: submit: {e}"))?;
    let submit_ns = t_submit.elapsed().as_nanos();

    let out = res.arena.read_work(1, h)?;

    // Attribute the three GEMVs' submit/host time into the same profile buckets
    // the standalone GEMV path uses, so the breakdown stays comparable.
    let total_ns = t_start.elapsed().as_nanos();
    res.gemv_submit_ns += submit_ns;
    res.gemv_other_ns += total_ns.saturating_sub(submit_ns);
    res.gemv_count += 3;
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Host f32 elementwise / norm primitives (transcribed from the CUDA reference).
// ─────────────────────────────────────────────────────────────────────────────

/// Plain RMSNorm: `out[i] = x[i] * inv_rms * w[i]`,
/// `inv_rms = 1/sqrt(mean(x^2) + eps)`. Qwen3.5 input / post / final / q-k norms.
///
/// NOTE: the HF/safetensors reference (`crates/infer-cuda/src/qwen35.rs` +
/// `norm.cu`'s `rms_norm_offset_kernel`) applies the **`(1+w)` offset** because
/// Qwen3-Next/3.5 store the norm scale zero-centered in HF. The **GGUF
/// converter folds the `+1` into the stored weight** (verified on the on-box
/// 27B: `attn_norm` ≈ 0.98, `output_norm` ≈ 1.96, q/k_norm ≈ 1.2 — centered on
/// 1, not 0), so the GGUF weights must be applied **plain**. This matches
/// llama.cpp's `build_norm(..., LLM_NORM_RMS)` (plain `x*inv_rms*w`) on the
/// same file, which decodes coherently. Applying `(1+w)` here roughly doubled
/// every norm scale AND scrambled the per-element ratios → garbage logits.
fn rms_norm_weight(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut sumsq = 0.0f32;
    for &v in x {
        sumsq += v * v;
    }
    let inv = 1.0 / (sumsq / n as f32 + eps).sqrt();
    (0..n).map(|i| x[i] * inv * w[i]).collect()
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
