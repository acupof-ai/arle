//! Batched decode for Qwen3.5: process multiple requests in one forward pass.
//!
//! Hybrid architecture: 8 full attention layers use HD256 paged decode
//! (FlashInfer for BF16 pools, custom quantized kernels otherwise), and
//! 24 linear attention layers use batched recurrent kernels (conv1d + GDR)
//! via pointer arrays.

use anyhow::Result;
use cudarc::driver::safe::CudaGraph;
use cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
use cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use log::info;

use super::forward::Qwen35State;
use super::weights::{
    FullAttentionLayer, LayerKind, LinearAttentionLayer, Qwen35Model, TransformerBlock35,
};
use crate::model::ModelForward;
use crate::model::kv_cache::KVFormat;
use crate::ops;
use cuda_kernels::kv_quant;
use cuda_kernels::kv_turboquant;
use cuda_kernels::prelude::{DeviceContext, FlashInferDecodeMetadata, HiddenStates, PagedKVPool};

// ── Sub-structs ─────────────────────────────────────────────────────────────

/// Buffers shared across all layer types: embedding, residuals, final norm.
pub(crate) struct CommonBufs {
    pub(super) hidden_out: HiddenStates,
    pub(super) normed: HiddenStates,
    pub(super) embedding_out: HiddenStates,
    pub(super) o_buf: HiddenStates,
    pub(super) attn_results: HiddenStates,
    pub(super) hidden_mid: HiddenStates,
}

// SAFETY: Exclusively accessed from the single scheduler inference thread.
unsafe impl Send for CommonBufs {}

impl CommonBufs {
    fn set_batch_size(&mut self, bs: usize) {
        self.hidden_out.seq_len = bs;
        self.normed.seq_len = bs;
        self.o_buf.seq_len = bs;
        self.attn_results.seq_len = bs;
        self.hidden_mid.seq_len = bs;
    }
}

/// Buffers for full attention layers (HD256, paged).
pub(crate) struct FullAttnBufs {
    pub(super) q_full_batch: HiddenStates,
    pub(super) q_batch: HiddenStates,
    pub(super) k_batch: HiddenStates,
    pub(super) v_batch: HiddenStates,
    pub(super) attn_output: HiddenStates,
    /// Rotated query buffer for TurboQuant fused attention [max_batch_size, q_dim].
    pub(super) q_rot: HiddenStates,
}

// SAFETY: Exclusively accessed from the single scheduler inference thread.
unsafe impl Send for FullAttnBufs {}

impl FullAttnBufs {
    fn set_batch_size(&mut self, bs: usize) {
        self.q_full_batch.seq_len = bs;
        self.q_batch.seq_len = bs;
        self.k_batch.seq_len = bs;
        self.v_batch.seq_len = bs;
        self.attn_output.seq_len = bs;
        self.q_rot.seq_len = bs;
    }
}

/// Buffers for linear attention layers (conv1d + GDR recurrent).
pub(crate) struct RecurrentBufs {
    pub(super) qkv_batch: HiddenStates,
    pub(super) z_batch: HiddenStates,
    pub(super) b_batch: HiddenStates,
    pub(super) a_batch: HiddenStates,
    /// Per-layer GPU pointer arrays for conv1d state.
    /// Pre-uploaded before decode body to enable future CUDA Graph capture.
    pub(super) conv_state_ptrs_per_layer: Vec<CudaSlice<u64>>,
    /// Per-layer GPU pointer arrays for GDR state.
    pub(super) gdr_state_ptrs_per_layer: Vec<CudaSlice<u64>>,
    /// Shared host staging buffer for pointer array uploads.
    pub(super) conv_state_ptrs_host: Vec<u64>,
    pub(super) gdr_state_ptrs_host: Vec<u64>,
    pub(super) qkv_conv_batch: HiddenStates,
    pub(super) gdr_out_batch: HiddenStates,
    pub(super) normed_gated: HiddenStates,
}

// SAFETY: Exclusively accessed from the single scheduler inference thread.
unsafe impl Send for RecurrentBufs {}

impl RecurrentBufs {
    fn set_batch_size(&mut self, bs: usize) {
        self.qkv_batch.seq_len = bs;
        self.z_batch.seq_len = bs;
        self.b_batch.seq_len = bs;
        self.a_batch.seq_len = bs;
        self.qkv_conv_batch.seq_len = bs;
        self.gdr_out_batch.seq_len = bs;
        self.normed_gated.seq_len = bs;
    }
}

/// Buffers for MLP (gate/up/down projections).
pub(crate) struct MlpBufs {
    pub(super) gate_out: HiddenStates,
    pub(super) up_out: HiddenStates,
    pub(super) act_out: HiddenStates,
}

// SAFETY: Exclusively accessed from the single scheduler inference thread.
unsafe impl Send for MlpBufs {}

impl MlpBufs {
    fn set_batch_size(&mut self, bs: usize) {
        self.gate_out.seq_len = bs;
        self.up_out.seq_len = bs;
        self.act_out.seq_len = bs;
    }
}

// ── Outer container ─────────────────────────────────────────────────────────

/// Pre-allocated buffers for batched decode, reused across steps.
pub struct BatchDecodeBuffers35 {
    pub(super) common: CommonBufs,
    pub(super) attn: FullAttnBufs,
    pub(super) recurrent: RecurrentBufs,
    pub(super) mlp: MlpBufs,

    // ── Logits + sampling ──
    pub(super) logits_batch: Option<HiddenStates>,
    pub(super) argmax_out: CudaSlice<i32>,
    pub(super) argmax_host: Vec<i32>,
    pub(super) logprobs_gpu: CudaSlice<f32>,
    pub(super) logprobs_host: Vec<f32>,

    // ── Token IDs ──
    token_ids_gpu: CudaSlice<i32>,
    token_ids_scratch: Vec<i32>,

    // ── FlashInfer metadata (for full attention layers) ──
    pub(crate) metadata: FlashInferDecodeMetadata,
    /// Packed page-aware metadata for quantized decode kernels:
    /// `[page_indptr..., last_page_len...]`.
    quantized_kv_meta: CudaSlice<i32>,

    /// Piecewise CUDA Graph cache for groups of consecutive linear layers.
    /// Indexed by [group_idx][batch_size - 1].
    /// Full attention layers run eagerly between groups.
    graph_cache: Vec<Vec<Option<CudaGraph>>>,
    /// One-shot eager decode override for verifier/correctness-sensitive paths.
    force_eager_once: bool,

    max_batch_size: usize,
}

// SAFETY: Exclusively accessed from the single scheduler inference thread.
unsafe impl Send for BatchDecodeBuffers35 {}

impl BatchDecodeBuffers35 {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &DeviceContext,
        hidden_dim: usize,
        q_proj_dim: usize, // num_q_heads * 256 * 2 (includes gate)
        q_dim: usize,      // num_q_heads * 256
        kv_dim: usize,     // num_kv_heads * 256
        inter_dim: usize,
        qkv_dim: usize, // linear attention QKV dim
        z_dim: usize,   // linear attention Z dim
        b_dim: usize,   // linear attention B dim (num_value_heads)
        max_batch_size: usize,
        num_qheads: usize,
        max_total_pages: usize,
        num_linear_layers: usize,
    ) -> Result<Self> {
        let common = CommonBufs {
            hidden_out: HiddenStates::zeros(ctx, hidden_dim, max_batch_size)?,
            normed: HiddenStates::zeros(ctx, hidden_dim, max_batch_size)?,
            embedding_out: HiddenStates::zeros(ctx, hidden_dim, max_batch_size)?,
            o_buf: HiddenStates::zeros(ctx, hidden_dim, max_batch_size)?,
            attn_results: HiddenStates::zeros(ctx, hidden_dim, max_batch_size)?,
            hidden_mid: HiddenStates::zeros(ctx, hidden_dim, max_batch_size)?,
        };

        let attn = FullAttnBufs {
            q_full_batch: HiddenStates::zeros(ctx, q_proj_dim, max_batch_size)?,
            q_batch: HiddenStates::zeros(ctx, q_dim, max_batch_size)?,
            k_batch: HiddenStates::zeros(ctx, kv_dim, max_batch_size)?,
            v_batch: HiddenStates::zeros(ctx, kv_dim, max_batch_size)?,
            attn_output: HiddenStates::zeros(ctx, q_dim, max_batch_size)?,
            q_rot: HiddenStates::zeros(ctx, q_dim, max_batch_size)?,
        };

        // Per-layer pointer arrays enable future CUDA Graph capture by moving
        // all H2D pointer uploads before the graph-capturable section.
        let mut conv_ptrs = Vec::with_capacity(num_linear_layers);
        let mut gdr_ptrs = Vec::with_capacity(num_linear_layers);
        for _ in 0..num_linear_layers {
            conv_ptrs.push(
                ctx.stream
                    .alloc_zeros::<u64>(max_batch_size)
                    .map_err(|e| anyhow::anyhow!("Alloc conv_state_ptrs_per_layer: {e}"))?,
            );
            gdr_ptrs.push(
                ctx.stream
                    .alloc_zeros::<u64>(max_batch_size)
                    .map_err(|e| anyhow::anyhow!("Alloc gdr_state_ptrs_per_layer: {e}"))?,
            );
        }

        let recurrent = RecurrentBufs {
            qkv_batch: HiddenStates::zeros(ctx, qkv_dim, max_batch_size)?,
            z_batch: HiddenStates::zeros(ctx, z_dim, max_batch_size)?,
            b_batch: HiddenStates::zeros(ctx, b_dim, max_batch_size)?,
            a_batch: HiddenStates::zeros(ctx, b_dim, max_batch_size)?,
            conv_state_ptrs_per_layer: conv_ptrs,
            gdr_state_ptrs_per_layer: gdr_ptrs,
            conv_state_ptrs_host: vec![0u64; max_batch_size],
            gdr_state_ptrs_host: vec![0u64; max_batch_size],
            qkv_conv_batch: HiddenStates::zeros(ctx, qkv_dim, max_batch_size)?,
            gdr_out_batch: HiddenStates::zeros(ctx, z_dim, max_batch_size)?,
            normed_gated: HiddenStates::zeros(ctx, z_dim, max_batch_size)?,
        };

        let mlp = MlpBufs {
            gate_out: HiddenStates::zeros(ctx, inter_dim, max_batch_size)?,
            up_out: HiddenStates::zeros(ctx, inter_dim, max_batch_size)?,
            act_out: HiddenStates::zeros(ctx, inter_dim, max_batch_size)?,
        };

        Ok(Self {
            common,
            attn,
            recurrent,
            mlp,

            logits_batch: None,
            argmax_out: ctx
                .stream
                .alloc_zeros(max_batch_size)
                .map_err(|e| anyhow::anyhow!("Alloc argmax_out: {e}"))?,
            argmax_host: vec![0i32; max_batch_size],
            logprobs_gpu: ctx
                .stream
                .alloc_zeros(max_batch_size)
                .map_err(|e| anyhow::anyhow!("Alloc logprobs_gpu: {e}"))?,
            logprobs_host: vec![0.0f32; max_batch_size],

            token_ids_gpu: ctx
                .stream
                .alloc_zeros(max_batch_size)
                .map_err(|e| anyhow::anyhow!("Alloc token_ids_gpu: {e}"))?,
            token_ids_scratch: Vec::with_capacity(max_batch_size),

            metadata: FlashInferDecodeMetadata::new(
                ctx,
                max_batch_size,
                max_total_pages,
                num_qheads,
            )?,
            quantized_kv_meta: ctx
                .stream
                .alloc_zeros(2 * max_batch_size + 1)
                .map_err(|e| anyhow::anyhow!("Alloc quantized_kv_meta: {e}"))?,

            // Piecewise graph cache: one entry per group of consecutive linear layers.
            // For Qwen3.5: full_attention_interval=4 → 8 groups of 3 linear layers.
            graph_cache: {
                let num_groups = if num_linear_layers > 0 {
                    // Groups of consecutive linear layers between full attention layers
                    // For interval=4: groups = num_hidden_layers / interval
                    num_linear_layers.div_ceil(3) // ceil(num_linear_layers / 3)
                } else {
                    0
                };
                (0..num_groups)
                    .map(|_| (0..max_batch_size).map(|_| None).collect())
                    .collect()
            },
            force_eager_once: false,

            max_batch_size,
        })
    }

    fn set_batch_size_inner(&mut self, bs: usize) {
        debug_assert!(bs <= self.max_batch_size);
        self.common.set_batch_size(bs);
        self.attn.set_batch_size(bs);
        self.recurrent.set_batch_size(bs);
        self.mlp.set_batch_size(bs);
    }
}

impl crate::model::DecodeContextOps for BatchDecodeBuffers35 {
    fn upload_token_ids(&mut self, ctx: &DeviceContext, tokens: &[u32]) -> Result<()> {
        self.token_ids_scratch.clear();
        self.token_ids_scratch
            .extend(tokens.iter().map(|&x| x as i32));
        ctx.stream
            .memcpy_htod(&self.token_ids_scratch, &mut self.token_ids_gpu)
            .map_err(|e| anyhow::anyhow!("H2D token_ids: {e}"))?;
        Ok(())
    }

    fn update_metadata(
        &mut self,
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        slot_indices: &[usize],
    ) -> Result<bool> {
        self.metadata.update(ctx, pool, slot_indices)
    }

    fn plan_attention(
        &mut self,
        ctx: &DeviceContext,
        batch_size: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        page_size: usize,
        head_dim: usize,
        kv_format: crate::model::kv_cache::KVFormat,
    ) -> Result<()> {
        // Only BF16 full-attention layers run through FlashInfer HD256.
        // FP8/INT8/TurboQuant decode uses quantized kernels instead.
        // FlashInfer plans once per forward; the TileLang HD256 decode kernel
        // is plan-less, so this gate must match the actual decode-kernel
        // selector. The broader `tilelang-attn` feature only covers HD128
        // prefill/TC-decode paths and still needs FlashInfer's HD256 plan.
        // Metadata uploads (positions, kv_indptr, kv_indices,
        // kv_last_page_len, qo_indptr) above in `update_metadata` are needed
        // by both paths and stay unconditional.
        #[cfg(not(feature = "tilelang-decode-hd256"))]
        if kv_format == KVFormat::BF16 {
            self.metadata.plan_hd256(
                ctx,
                batch_size,
                num_q_heads,
                num_kv_heads,
                page_size,
                head_dim,
            )?;
        }
        #[cfg(feature = "tilelang-decode-hd256")]
        {
            let _ = (
                ctx,
                batch_size,
                num_q_heads,
                num_kv_heads,
                page_size,
                head_dim,
                kv_format,
            );
        }
        Ok(())
    }

    fn set_batch_size(&mut self, bs: usize) {
        self.set_batch_size_inner(bs);
    }

    fn invalidate_graph_cache(&mut self, _batch_size: usize) {
        // Qwen3.5 uses piecewise graph cache (per linear-layer group).
        // No per-batch-size invalidation needed — reallocation doesn't
        // happen in the piecewise scheme.
    }

    fn force_eager_once(&mut self) {
        self.force_eager_once = true;
    }

    fn logprobs_host(&self) -> &[f32] {
        &self.logprobs_host
    }
}

impl Qwen35Model {
    pub(crate) fn prepare_decode_context(
        &self,
        tokens: &[u32],
        slot_indices: &[usize],
        paged_kv_pool: &PagedKVPool,
        bufs: &mut BatchDecodeBuffers35,
    ) -> Result<()> {
        use crate::model::DecodeContextOps;

        bufs.set_batch_size(tokens.len());
        bufs.upload_token_ids(&self.ctx, tokens)?;
        let reallocated = bufs.update_metadata(&self.ctx, paged_kv_pool, slot_indices)?;
        if reallocated {
            bufs.invalidate_graph_cache(tokens.len());
        }
        bufs.plan_attention(
            &self.ctx,
            tokens.len(),
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            paged_kv_pool.page_size,
            self.config.head_dim,
            paged_kv_pool.format,
        )?;
        Ok(())
    }

    /// Batched decode: process B tokens from B different requests in one pass.
    /// Falls back to sequential forward_decode() for non-paged path.
    pub fn decode_batch_contiguous(
        &self,
        tokens: &[u32],
        states: &mut [Qwen35State],
        slot_indices: &[usize],
    ) -> Result<()> {
        for (i, &token) in tokens.iter().enumerate() {
            self.forward_decode(token, &mut states[slot_indices[i]])?;
        }
        Ok(())
    }

    /// Batched decode with paged KV for full attention, per-request recurrent for linear.
    pub fn decode_batch(
        &self,
        tokens: &[u32],
        states: &mut [Qwen35State],
        slot_indices: &[usize],
        skip_logit_scatter: bool,
        paged_kv_pool: &mut PagedKVPool,
        bufs: &mut BatchDecodeBuffers35,
    ) -> Result<()> {
        let batch_size = tokens.len();
        debug_assert_eq!(batch_size, slot_indices.len());
        if batch_size == 0 {
            return Ok(());
        }
        debug_assert!(batch_size <= bufs.max_batch_size);

        // NOTE: set_batch_size, upload_token_ids, update_metadata, and
        // plan_attention are now called by the scheduler via DecodeContextOps
        // before this method is invoked.
        if matches!(paged_kv_pool.format, KVFormat::INT8 | KVFormat::FP8E4M3) {
            let packed = paged_kv_pool.build_quantized_decode_indptr(slot_indices);
            self.ctx
                .stream
                .memcpy_htod(&packed, &mut bufs.quantized_kv_meta)
                .map_err(|e| anyhow::anyhow!("H2D quantized_kv_meta: {e}"))?;
        }

        bufs.common.embedding_out.seq_len = batch_size;

        // Lazy-init logits buffer
        if bufs.logits_batch.is_none() {
            let vocab_size = self.embed_tokens.rows;
            bufs.logits_batch = Some(HiddenStates::zeros(
                &self.ctx,
                vocab_size,
                bufs.max_batch_size,
            )?);
        }

        // ── Pre-upload all recurrent state pointer arrays ──
        // Moving all H2D before the forward pass enables future CUDA Graph capture.
        {
            let mut linear_idx = 0usize;
            for layer in &self.layers {
                if matches!(layer.attn, LayerKind::LinearAttention(_)) {
                    for (b, &si) in slot_indices.iter().enumerate() {
                        let layer_state = &mut states[si].recurrent_state.layers[linear_idx];
                        let (conv_ptr, _) =
                            layer_state.conv_state.data.device_ptr_mut(&self.ctx.stream);
                        let (gdr_ptr, _) = layer_state.state.device_ptr_mut(&self.ctx.stream);
                        bufs.recurrent.conv_state_ptrs_host[b] = conv_ptr;
                        bufs.recurrent.gdr_state_ptrs_host[b] = gdr_ptr;
                    }
                    self.ctx
                        .stream
                        .memcpy_htod(
                            &bufs.recurrent.conv_state_ptrs_host[..batch_size],
                            &mut bufs.recurrent.conv_state_ptrs_per_layer[linear_idx],
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("H2D conv_state_ptrs layer {linear_idx}: {e}")
                        })?;
                    self.ctx
                        .stream
                        .memcpy_htod(
                            &bufs.recurrent.gdr_state_ptrs_host[..batch_size],
                            &mut bufs.recurrent.gdr_state_ptrs_per_layer[linear_idx],
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("H2D gdr_state_ptrs layer {linear_idx}: {e}")
                        })?;
                    linear_idx += 1;
                }
            }
        }

        // ── Forward pass ──
        self.decode_batch_body(bufs, states, slot_indices, paged_kv_pool, batch_size)?;

        // Scatter per-slot logits when needed (non-greedy fallback)
        if !skip_logit_scatter {
            let logits = bufs.logits_batch.as_ref().unwrap();
            for (b, &si) in slot_indices.iter().enumerate() {
                ops::extract_vec_into(
                    &self.ctx,
                    logits,
                    b,
                    &mut states[si].decode_bufs.logits_scratch,
                )?;
                states[si].decode_bufs.bind_logits_scratch(&self.ctx);
                states[si].base.prefill_logits = None;
            }
        }

        Ok(())
    }

    fn decode_batch_body(
        &self,
        bufs: &mut BatchDecodeBuffers35,
        _states: &mut [Qwen35State],
        _slot_indices: &[usize],
        kv_pool: &PagedKVPool,
        batch_size: usize,
    ) -> Result<()> {
        let c = &self.config;

        // Embedding (eager, before any graph)
        ops::embedding_batch(
            &self.ctx,
            &self.embed_tokens,
            &bufs.token_ids_gpu,
            &mut bufs.common.embedding_out,
        )?;

        let hidden_ptr = &raw mut bufs.common.embedding_out;

        // Process layers in groups: each group is consecutive linear layers
        // followed by one full attention layer. Linear groups are graph-captured.
        let force_eager = std::mem::take(&mut bufs.force_eager_once);
        let mut full_idx = 0usize;
        let mut linear_idx = 0usize;
        let mut group_idx = 0usize;
        let mut group_start: Option<usize> = None;

        for (layer_i, layer) in self.layers.iter().enumerate() {
            match &layer.attn {
                LayerKind::LinearAttention(_) => {
                    if group_start.is_none() {
                        group_start = Some(layer_i);
                    }
                    linear_idx += 1;
                }
                LayerKind::FullAttention(attn) => {
                    // Flush any pending linear group with graph capture
                    if let Some(start) = group_start.take() {
                        self.run_linear_group_graphed(
                            bufs,
                            start,
                            layer_i,
                            linear_idx,
                            group_idx,
                            batch_size,
                            force_eager,
                        )?;
                        group_idx += 1;
                    }

                    // Full attention: always eager (FlashInfer metadata changes)
                    let hidden = unsafe { &mut *hidden_ptr };
                    self.decode_batch_full_attn_layer(
                        layer, attn, hidden, bufs, kv_pool, full_idx, batch_size,
                    )?;
                    full_idx += 1;
                }
            }
        }
        // Flush final linear group if layers end with linear
        if let Some(start) = group_start.take() {
            self.run_linear_group_graphed(
                bufs,
                start,
                self.layers.len(),
                linear_idx,
                group_idx,
                batch_size,
                force_eager,
            )?;
        }

        // Final norm (offset variant) + logits GEMM (eager)
        let hidden = unsafe { &*hidden_ptr };
        ops::rms_norm_batch_offset_into(
            &self.ctx,
            hidden,
            &self.norm,
            c.rms_norm_eps,
            &mut bufs.common.normed,
        )?;
        let logits_buf = bufs.logits_batch.as_mut().unwrap();
        logits_buf.seq_len = batch_size;
        ops::gemm_into(
            &self.ctx,
            &self.embed_tokens,
            &bufs.common.normed,
            logits_buf,
        );

        Ok(())
    }

    /// Run a group of consecutive linear layers, using CUDA Graph capture/replay.
    fn run_linear_group_graphed(
        &self,
        bufs: &mut BatchDecodeBuffers35,
        layer_start: usize,
        layer_end: usize,
        linear_idx_end: usize,
        group_idx: usize,
        batch_size: usize,
        force_eager: bool,
    ) -> Result<()> {
        let linear_count = layer_end - layer_start;
        let linear_idx_start = linear_idx_end - linear_count;

        // Graph capture/replay for this group
        let use_graph = !force_eager
            && <Self as crate::model::ModelForward>::supports_cuda_graph_decode(self)
            && group_idx < bufs.graph_cache.len();
        if use_graph {
            if let Some(ref graph) = bufs.graph_cache[group_idx][batch_size - 1] {
                // Replay existing graph
                graph.launch().map_err(|e| {
                    anyhow::anyhow!("Graph replay (group={}, B={}): {e}", group_idx, batch_size)
                })?;
                return Ok(());
            }
        }

        // No graph cached — try to capture
        if use_graph {
            self.ctx
                .stream
                .begin_capture(CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .map_err(|e| anyhow::anyhow!("begin_capture: {e}"))?;
        }

        // Run the linear layers
        let hidden_ptr = &raw mut bufs.common.embedding_out;
        let mut li = linear_idx_start;
        for layer in &self.layers[layer_start..layer_end] {
            if let LayerKind::LinearAttention(attn) = &layer.attn {
                let hidden = unsafe { &mut *hidden_ptr };
                self.decode_batch_linear_attn_layer_graphable(
                    layer, attn, hidden, bufs, li, batch_size,
                )?;
                li += 1;
            }
        }

        // End capture
        if use_graph {
            let graph_opt = self
                .ctx
                .stream
                .end_capture(CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
                .map_err(|e| anyhow::anyhow!("end_capture: {e}"))?;

            if let Some(graph) = graph_opt {
                graph.launch().map_err(|e| {
                    anyhow::anyhow!(
                        "Graph first launch (group={}, B={}): {e}",
                        group_idx,
                        batch_size
                    )
                })?;
                info!(
                    "Piecewise CUDA Graph captured: group={}, layers={}-{}, B={}",
                    group_idx,
                    layer_start,
                    layer_end - 1,
                    batch_size
                );
                bufs.graph_cache[group_idx][batch_size - 1] = Some(graph);
            }
        }

        Ok(())
    }

    /// Linear attention layer for graph-capturable execution.
    /// No H2D, no state access — uses pre-uploaded per-layer pointer arrays.
    #[allow(clippy::too_many_arguments)]
    fn decode_batch_linear_attn_layer_graphable(
        &self,
        layer: &TransformerBlock35,
        attn: &LinearAttentionLayer,
        hidden: &mut HiddenStates,
        bufs: &mut BatchDecodeBuffers35,
        linear_idx: usize,
        batch_size: usize,
    ) -> Result<()> {
        let c = &self.config;
        let eps = c.rms_norm_eps;

        // 1. Input RMSNorm
        ops::rms_norm_batch_offset_into(
            &self.ctx,
            hidden,
            &layer.input_layernorm,
            eps,
            &mut bufs.common.normed,
        )?;

        // 2. Projections
        ops::gemm_into(
            &self.ctx,
            &attn.in_proj_qkv,
            &bufs.common.normed,
            &mut bufs.recurrent.qkv_batch,
        );
        ops::gemm_into(
            &self.ctx,
            &attn.in_proj_z,
            &bufs.common.normed,
            &mut bufs.recurrent.z_batch,
        );
        ops::gemm_into(
            &self.ctx,
            &attn.in_proj_b,
            &bufs.common.normed,
            &mut bufs.recurrent.b_batch,
        );
        ops::gemm_into(
            &self.ctx,
            &attn.in_proj_a,
            &bufs.common.normed,
            &mut bufs.recurrent.a_batch,
        );

        // 3. Conv1d + GDR using pre-uploaded per-layer pointer arrays
        ops::conv1d_decode_batch_into(
            &self.ctx,
            &bufs.recurrent.qkv_batch,
            &attn.conv1d_weight,
            &mut bufs.recurrent.conv_state_ptrs_per_layer[linear_idx],
            &mut bufs.recurrent.qkv_conv_batch,
            c.linear_conv_kernel_dim,
            batch_size,
        );
        ops::gdr_decode_batch_into(
            &self.ctx,
            &bufs.recurrent.qkv_conv_batch,
            &bufs.recurrent.b_batch,
            &bufs.recurrent.a_batch,
            &ops::GdrWeights {
                dt_bias: &attn.dt_bias,
                a_log: &attn.a_log,
            },
            &mut bufs.recurrent.gdr_state_ptrs_per_layer[linear_idx],
            &mut bufs.recurrent.gdr_out_batch,
            &ops::GdrHeadConfig {
                num_key_heads: c.linear_num_key_heads,
                num_value_heads: c.linear_num_value_heads,
                key_dim: c.linear_key_head_dim,
                val_dim: c.linear_value_head_dim,
            },
            batch_size,
        )?;

        // 4. Gated RMSNorm
        ops::rms_norm_gated_batch_into(
            &self.ctx,
            &bufs.recurrent.gdr_out_batch,
            &attn.norm_weight,
            &bufs.recurrent.z_batch,
            &mut bufs.recurrent.normed_gated,
            c.linear_num_value_heads,
            c.linear_value_head_dim,
            eps,
        );

        // 5. Out projection
        ops::gemm_into(
            &self.ctx,
            &attn.out_proj,
            &bufs.recurrent.normed_gated,
            &mut bufs.common.attn_results,
        );

        // 6. Residual + MLP
        self.decode_batch_mlp(layer, hidden, bufs, batch_size)?;

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_batch_full_attn_layer(
        &self,
        layer: &TransformerBlock35,
        attn: &FullAttentionLayer,
        hidden: &mut HiddenStates,
        bufs: &mut BatchDecodeBuffers35,
        kv_pool: &PagedKVPool,
        full_idx: usize,
        batch_size: usize,
    ) -> Result<()> {
        let c = &self.config;
        let eps = c.rms_norm_eps;
        let num_heads = c.num_attention_heads;
        let num_kv_heads = c.num_key_value_heads;
        let head_dim = c.head_dim;
        let page_size = kv_pool.page_size;

        // 1. Input RMSNorm (offset variant)
        ops::rms_norm_batch_offset_into(
            &self.ctx,
            hidden,
            &layer.input_layernorm,
            eps,
            &mut bufs.common.normed,
        )?;

        // 2. QKV projections (batched GEMM)
        ops::gemm_into(
            &self.ctx,
            &attn.q_proj,
            &bufs.common.normed,
            &mut bufs.attn.q_full_batch,
        );
        ops::gemm_into(
            &self.ctx,
            &attn.k_proj,
            &bufs.common.normed,
            &mut bufs.attn.k_batch,
        );
        ops::gemm_into(
            &self.ctx,
            &attn.v_proj,
            &bufs.common.normed,
            &mut bufs.attn.v_batch,
        );

        // 3. Decode prep: QK-norm (1+w) + partial RoPE + paged KV write
        let nrp = ops::NormRopeParams {
            q_norm: &attn.q_norm,
            k_norm: &attn.k_norm,
            cos_cache: &self.cos_cache,
            sin_cache: &self.sin_cache,
            rms_eps: eps,
        };
        let paged = ops::PagedKVMeta {
            kv_pool,
            layer_idx: full_idx,
            kv_indices: &bufs.metadata.kv_indices,
            kv_indptr: &bufs.metadata.kv_indptr,
            kv_last_page_len: &bufs.metadata.kv_last_page_len,
            page_size,
        };
        ops::decode_prep_paged_hd256(
            &self.ctx,
            &bufs.attn.q_full_batch,
            &mut bufs.attn.q_batch,
            &bufs.attn.k_batch,
            &bufs.attn.v_batch,
            &nrp,
            &bufs.metadata.positions,
            &paged,
            num_heads,
            num_kv_heads,
            c.rotary_dim,
        )?;

        // 4. Attention dispatch — format-aware (quantize new token + attention read)
        {
            let stream = &self.ctx.stream;

            match kv_pool.format {
                KVFormat::FP8E4M3 => {
                    kv_quant::quantize_paged_kv_fp8(
                        &self.ctx,
                        kv_pool.k_work_ptr(stream),
                        kv_pool.k_data_ptr(full_idx, stream),
                        kv_pool.k_scales_ptr(full_idx, stream),
                        &bufs.metadata.last_token_indices,
                        num_kv_heads,
                        head_dim,
                        kv_pool.kv_dim,
                        batch_size,
                    )?;
                    kv_quant::quantize_paged_kv_fp8(
                        &self.ctx,
                        kv_pool.v_work_ptr(stream),
                        kv_pool.v_data_ptr(full_idx, stream),
                        kv_pool.v_scales_ptr(full_idx, stream),
                        &bufs.metadata.last_token_indices,
                        num_kv_heads,
                        head_dim,
                        kv_pool.kv_dim,
                        batch_size,
                    )?;
                }
                KVFormat::INT8 => {
                    kv_quant::quantize_paged_kv_single(
                        &self.ctx,
                        kv_pool.k_work_ptr(stream),
                        kv_pool.k_data_ptr(full_idx, stream),
                        kv_pool.k_scales_ptr(full_idx, stream),
                        &bufs.metadata.last_token_indices,
                        num_kv_heads,
                        head_dim,
                        kv_pool.kv_dim,
                        batch_size,
                    )?;
                    kv_quant::quantize_paged_kv_single(
                        &self.ctx,
                        kv_pool.v_work_ptr(stream),
                        kv_pool.v_data_ptr(full_idx, stream),
                        kv_pool.v_scales_ptr(full_idx, stream),
                        &bufs.metadata.last_token_indices,
                        num_kv_heads,
                        head_dim,
                        kv_pool.kv_dim,
                        batch_size,
                    )?;
                }
                KVFormat::BF16 => {}
                KVFormat::TurboQuant { .. } => {
                    let tq_k = kv_pool.tq_k_state.as_ref().unwrap();
                    kv_turboquant::turboquant_quantize_paged_single(
                        &self.ctx,
                        kv_pool.k_work_ptr(stream),
                        kv_pool.k_data_slice(full_idx),
                        kv_pool.k_norms_slice(full_idx),
                        &bufs.metadata.last_token_indices,
                        tq_k,
                        full_idx,
                        num_kv_heads,
                        head_dim,
                        batch_size,
                    )?;
                    let tq_v = kv_pool.tq_v_state.as_ref().unwrap();
                    kv_turboquant::turboquant_quantize_paged_single(
                        &self.ctx,
                        kv_pool.v_work_ptr(stream),
                        kv_pool.v_data_slice(full_idx),
                        kv_pool.v_norms_slice(full_idx),
                        &bufs.metadata.last_token_indices,
                        tq_v,
                        full_idx,
                        num_kv_heads,
                        head_dim,
                        batch_size,
                    )?;
                }
            }

            match kv_pool.format {
                KVFormat::INT8 => {
                    let sm_scale = 1.0 / (head_dim as f32).sqrt();
                    kv_quant::decode_attention_int8(
                        &self.ctx,
                        &bufs.attn.q_batch,
                        kv_pool.k_data_ptr(full_idx, stream),
                        kv_pool.v_data_ptr(full_idx, stream),
                        kv_pool.k_scales_ptr(full_idx, stream),
                        kv_pool.v_scales_ptr(full_idx, stream),
                        &bufs.metadata.kv_indices,
                        &bufs.quantized_kv_meta,
                        &mut bufs.attn.attn_output,
                        batch_size,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        kv_pool.kv_dim,
                        sm_scale,
                        kv_pool.int8_attn_workspace.as_ref().unwrap(),
                        kv_pool.int8_attn_workspace_bytes,
                    )?;
                }
                KVFormat::FP8E4M3 => {
                    let sm_scale = 1.0 / (head_dim as f32).sqrt();
                    kv_quant::decode_attention_fp8(
                        &self.ctx,
                        &bufs.attn.q_batch,
                        kv_pool.k_data_ptr(full_idx, stream),
                        kv_pool.v_data_ptr(full_idx, stream),
                        kv_pool.k_scales_ptr(full_idx, stream),
                        kv_pool.v_scales_ptr(full_idx, stream),
                        &bufs.metadata.kv_indices,
                        &bufs.quantized_kv_meta,
                        &mut bufs.attn.attn_output,
                        batch_size,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        kv_pool.kv_dim,
                        sm_scale,
                        kv_pool.int8_attn_workspace.as_ref().unwrap(),
                        kv_pool.int8_attn_workspace_bytes,
                    )?;
                }
                KVFormat::BF16 => {
                    // qo_indptr / max_qlen / total_pages are TileLang-only
                    // (FlashInfer pulls them from plan_info), but we compute
                    // them unconditionally so the call shape is uniform across
                    // cfg arms. Decode = 1 Q row per request → max_qlen=1 and
                    // total_q_tokens=batch_size. Mirrors the T4 TC-decode
                    // pattern in qwen3/batch_decode.rs.
                    let max_qlen = bufs
                        .metadata
                        .qo_indptr_h
                        .windows(2)
                        .map(|w| w[1] - w[0])
                        .max()
                        .unwrap_or(0);
                    let total_pages = bufs.metadata.indptr_h.last().copied().unwrap_or(0);
                    ops::flashinfer_run_layer_hd256(
                        &self.ctx,
                        &bufs.attn.q_batch,
                        kv_pool,
                        full_idx,
                        &bufs.metadata.qo_indptr,
                        &bufs.metadata.kv_indptr,
                        &bufs.metadata.kv_indices,
                        &bufs.metadata.kv_last_page_len,
                        &mut bufs.attn.attn_output,
                        &mut bufs.metadata.flashinfer_ws,
                        &ops::FlashInferHeadConfig {
                            num_qo_heads: num_heads,
                            num_kv_heads,
                            page_size,
                            head_dim,
                        },
                        batch_size as i32,
                        max_qlen,
                        total_pages,
                    )?;
                }
                KVFormat::TurboQuant { .. } => {
                    // Fused TQ attention: rotate Q once, score from packed K centroids.
                    let tq_k = kv_pool.tq_k_state.as_ref().unwrap();
                    let tq_v = kv_pool.tq_v_state.as_ref().unwrap();
                    let sm_scale = 1.0 / (head_dim as f32).sqrt();

                    // Step 1: Rotate Q → Q_rot (sign flip + FWHT)
                    let q_ptr = {
                        let (p, _g) = bufs.attn.q_batch.data.device_ptr(stream);
                        p
                    };
                    let q_rot_ptr = {
                        let (p, _g) = bufs.attn.q_rot.data.device_ptr_mut(stream);
                        p
                    };
                    kv_turboquant::turboquant_rotate_query(
                        &self.ctx,
                        q_ptr,
                        q_rot_ptr,
                        tq_k,
                        full_idx,
                        batch_size * num_heads,
                        head_dim,
                    )?;

                    // Step 2: Fused attention
                    let attn_ptr = {
                        let (p, _g) = bufs.attn.attn_output.data.device_ptr_mut(stream);
                        p
                    };
                    kv_turboquant::turboquant_fused_decode_attention(
                        &self.ctx,
                        q_rot_ptr,
                        kv_pool.k_data_slice(full_idx),
                        kv_pool.k_norms_slice(full_idx),
                        kv_pool.v_data_slice(full_idx),
                        kv_pool.v_norms_slice(full_idx),
                        &bufs.metadata.kv_indices,
                        &bufs.metadata.kv_indptr,
                        attn_ptr,
                        tq_k,
                        tq_v,
                        batch_size,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        sm_scale,
                    )?;
                }
            }
        }

        // 5. Apply sigmoid gate
        ops::attention_gate_paged_hd256(
            &self.ctx,
            &bufs.attn.q_full_batch,
            &mut bufs.attn.attn_output,
            num_heads,
        );

        // 6. O projection
        ops::gemm_into(
            &self.ctx,
            &attn.o_proj,
            &bufs.attn.attn_output,
            &mut bufs.common.attn_results,
        );
        self.layer_communicator
            .post_attn_all_reduce_hidden_states(&mut bufs.common.attn_results)?;

        // 7. Residual + post-attention norm + MLP
        self.decode_batch_mlp(layer, hidden, bufs, batch_size)?;

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    fn decode_batch_linear_attn_layer(
        &self,
        layer: &TransformerBlock35,
        attn: &LinearAttentionLayer,
        hidden: &mut HiddenStates,
        bufs: &mut BatchDecodeBuffers35,
        _states: &mut [Qwen35State],
        _slot_indices: &[usize],
        linear_idx: usize,
        batch_size: usize,
    ) -> Result<()> {
        let c = &self.config;
        let eps = c.rms_norm_eps;

        // 1. Input RMSNorm (offset variant)
        ops::rms_norm_batch_offset_into(
            &self.ctx,
            hidden,
            &layer.input_layernorm,
            eps,
            &mut bufs.common.normed,
        )?;

        // 2. Batched projections (GEMM)
        ops::gemm_into(
            &self.ctx,
            &attn.in_proj_qkv,
            &bufs.common.normed,
            &mut bufs.recurrent.qkv_batch,
        );
        ops::gemm_into(
            &self.ctx,
            &attn.in_proj_z,
            &bufs.common.normed,
            &mut bufs.recurrent.z_batch,
        );
        ops::gemm_into(
            &self.ctx,
            &attn.in_proj_b,
            &bufs.common.normed,
            &mut bufs.recurrent.b_batch,
        );
        ops::gemm_into(
            &self.ctx,
            &attn.in_proj_a,
            &bufs.common.normed,
            &mut bufs.recurrent.a_batch,
        );

        // 3. Batched conv1d + GDR using pre-uploaded per-layer pointer arrays.
        // H2D uploads were done in decode_batch() before this body runs.
        {
            // Batched conv1d decode: one kernel launch for all B requests
            ops::conv1d_decode_batch_into(
                &self.ctx,
                &bufs.recurrent.qkv_batch,
                &attn.conv1d_weight,
                &mut bufs.recurrent.conv_state_ptrs_per_layer[linear_idx],
                &mut bufs.recurrent.qkv_conv_batch,
                c.linear_conv_kernel_dim,
                batch_size,
            );

            // Batched GDR decode: one kernel launch for all B requests
            ops::gdr_decode_batch_into(
                &self.ctx,
                &bufs.recurrent.qkv_conv_batch,
                &bufs.recurrent.b_batch,
                &bufs.recurrent.a_batch,
                &ops::GdrWeights {
                    dt_bias: &attn.dt_bias,
                    a_log: &attn.a_log,
                },
                &mut bufs.recurrent.gdr_state_ptrs_per_layer[linear_idx],
                &mut bufs.recurrent.gdr_out_batch,
                &ops::GdrHeadConfig {
                    num_key_heads: c.linear_num_key_heads,
                    num_value_heads: c.linear_num_value_heads,
                    key_dim: c.linear_key_head_dim,
                    val_dim: c.linear_value_head_dim,
                },
                batch_size,
            )?;
        }

        // 4. Batched gated RMSNorm
        ops::rms_norm_gated_batch_into(
            &self.ctx,
            &bufs.recurrent.gdr_out_batch,
            &attn.norm_weight,
            &bufs.recurrent.z_batch,
            &mut bufs.recurrent.normed_gated,
            c.linear_num_value_heads,
            c.linear_value_head_dim,
            eps,
        );

        // 5. Batched out projection
        ops::gemm_into(
            &self.ctx,
            &attn.out_proj,
            &bufs.recurrent.normed_gated,
            &mut bufs.common.attn_results,
        );
        self.layer_communicator
            .post_attn_all_reduce_hidden_states(&mut bufs.common.attn_results)?;

        // 6. Residual + post-attention norm + MLP
        self.decode_batch_mlp(layer, hidden, bufs, batch_size)?;

        Ok(())
    }

    /// Shared: residual add + post-attention norm + MLP + residual add.
    fn decode_batch_mlp(
        &self,
        layer: &TransformerBlock35,
        hidden: &mut HiddenStates,
        bufs: &mut BatchDecodeBuffers35,
        _batch_size: usize,
    ) -> Result<()> {
        let eps = self.config.rms_norm_eps;

        // Residual 1: hidden_mid = hidden + attn_results
        ops::add_batch_into(
            &self.ctx,
            hidden,
            &bufs.common.attn_results,
            &mut bufs.common.hidden_mid,
        )?;

        // Post-attention RMSNorm (offset variant)
        ops::rms_norm_batch_offset_into(
            &self.ctx,
            &bufs.common.hidden_mid,
            &layer.post_attention_layernorm,
            eps,
            &mut bufs.common.normed,
        )?;

        // MLP: gate + up → silu_mul → down
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.gate_proj,
            &bufs.common.normed,
            &mut bufs.mlp.gate_out,
        );
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.up_proj,
            &bufs.common.normed,
            &mut bufs.mlp.up_out,
        );
        ops::silu_mul_batch_into(
            &self.ctx,
            &bufs.mlp.gate_out,
            &bufs.mlp.up_out,
            &mut bufs.mlp.act_out,
        )?;
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.down_proj,
            &bufs.mlp.act_out,
            &mut bufs.common.o_buf,
        );
        self.layer_communicator
            .post_mlp_all_reduce_hidden_states(&mut bufs.common.o_buf)?;

        // Residual 2: hidden = hidden_mid + mlp_out
        ops::add_batch_into(
            &self.ctx,
            &bufs.common.hidden_mid,
            &bufs.common.o_buf,
            &mut bufs.common.hidden_out,
        )?;
        std::mem::swap(hidden, &mut bufs.common.hidden_out);

        Ok(())
    }
}
