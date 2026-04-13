//! Batched decode: process multiple requests' decode tokens in one forward pass.
//!
//! Uses GEMM (matrix multiply) for all linear projections (QKV, O, MLP),
//! batching B requests together. Attention uses FlashInfer with a shared
//! paged KV cache: QK-norm + RoPE + paged KV write are done in a prep kernel,
//! then FlashInfer's batch decode handles attention in a single launch.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use cudarc::driver::safe::CudaGraph;
use cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
use cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
use log::info;

use super::forward::Qwen3State;
use super::weights::{Qwen3Model, TransformerBlock};
use crate::backend::cuda::flashinfer::FlashInferDecodeMetadata;
use crate::model::ModelForward;
use crate::model::kv_cache::KVFormat;
use crate::ops;
use crate::ops::kv_quant;
use crate::ops::kv_turboquant;
use crate::backend::cuda::paged_kv::PagedKVPool;
use crate::backend::cuda::tensor::{DeviceContext, DeviceVec, HiddenStates};

/// Pre-allocated buffers for batched decode, reused across steps.
/// Allocated once for `max_batch_size`; smaller batches set `seq_len` on HiddenStates.
pub struct BatchDecodeBuffers {
    hidden_out: HiddenStates,
    normed: HiddenStates,
    q_batch: HiddenStates,
    k_batch: HiddenStates,
    v_batch: HiddenStates,
    /// Merged QKV output buffer [max_batch_size, q_dim + 2*kv_dim]
    qkv_batch: HiddenStates,
    attn_output: HiddenStates,
    /// Rotated query buffer for TurboQuant fused attention [max_batch_size, q_dim].
    q_rot: HiddenStates,
    o_buf: HiddenStates,
    gate_out: HiddenStates,
    up_out: HiddenStates,
    /// Merged gate+up output buffer [max_batch_size, 2*inter_dim]
    gate_up_out: HiddenStates,
    act_out: HiddenStates,

    /// Embedding output buffer [max_batch_size, hidden_dim] — avoids alloc in graph.
    embedding_out: HiddenStates,
    /// Batched logits buffer [max_batch_size, vocab_size] — avoids alloc in graph.
    pub(super) logits_batch: Option<HiddenStates>,
    /// Pre-allocated per-slot logits buffers (unused, kept for future non-greedy).
    logits_per_slot: Vec<DeviceVec>,
    /// Pre-allocated batch argmax output [max_batch_size] i32.
    pub(super) argmax_out: CudaSlice<i32>,
    /// Pre-allocated host buffer for batched argmax readback.
    pub(super) argmax_host: Vec<i32>,
    /// Pre-allocated batch logprob output [max_batch_size] f32.
    pub(super) logprobs_gpu: CudaSlice<f32>,
    /// Host readback for logprobs.
    pub logprobs_host: Vec<f32>,

    /// Pre-allocated token_ids buffer — avoids clone_htod alloc every step.
    token_ids_gpu: CudaSlice<i32>,

    /// Reusable host-side scratch vector to avoid per-step heap allocation.
    token_ids_scratch: Vec<i32>,

    /// FlashInfer paged attention metadata (positions, indptr, indices, workspace).
    pub(crate) metadata: FlashInferDecodeMetadata,

    /// Max batch size this buffer set was allocated for.
    max_batch_size: usize,

    /// CUDA Graph cache: index = batch_size - 1. Vec avoids HashMap overhead.
    graph_cache: Vec<Option<CudaGraph>>,
}

// SAFETY: BatchDecodeBuffers contains CudaGraph (CUgraphExec) which is !Send.
// Invariant: exclusively accessed from the single scheduler inference thread.
unsafe impl Send for BatchDecodeBuffers {}

impl BatchDecodeBuffers {
    /// Allocate buffers for up to `max_batch_size` requests.
    /// `max_total_pages` should be large enough for the worst-case total KV pages.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &DeviceContext,
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        inter_dim: usize,
        max_batch_size: usize,
        num_qheads: usize,
        max_total_pages: usize,
    ) -> Result<Self> {
        Ok(Self {
            hidden_out: HiddenStates::zeros(ctx, hidden_dim, max_batch_size)?,
            normed: HiddenStates::zeros(ctx, hidden_dim, max_batch_size)?,
            q_batch: HiddenStates::zeros(ctx, q_dim, max_batch_size)?,
            k_batch: HiddenStates::zeros(ctx, kv_dim, max_batch_size)?,
            v_batch: HiddenStates::zeros(ctx, kv_dim, max_batch_size)?,
            qkv_batch: HiddenStates::zeros(ctx, q_dim + 2 * kv_dim, max_batch_size)?,
            attn_output: HiddenStates::zeros(ctx, q_dim, max_batch_size)?,
            q_rot: HiddenStates::zeros(ctx, q_dim, max_batch_size)?,
            o_buf: HiddenStates::zeros(ctx, hidden_dim, max_batch_size)?,
            gate_out: HiddenStates::zeros(ctx, inter_dim, max_batch_size)?,
            up_out: HiddenStates::zeros(ctx, inter_dim, max_batch_size)?,
            gate_up_out: HiddenStates::zeros(ctx, 2 * inter_dim, max_batch_size)?,
            act_out: HiddenStates::zeros(ctx, inter_dim, max_batch_size)?,

            embedding_out: HiddenStates::zeros(ctx, hidden_dim, max_batch_size)?,
            logits_batch: None, // lazy-allocated on first use (needs vocab_size)
            logits_per_slot: Vec::new(),
            argmax_out: ctx
                .stream
                .alloc_zeros(max_batch_size)
                .map_err(|e| anyhow::anyhow!("Alloc argmax_out failed: {e}"))?,
            argmax_host: vec![0i32; max_batch_size],
            logprobs_gpu: ctx
                .stream
                .alloc_zeros(max_batch_size)
                .map_err(|e| anyhow::anyhow!("Alloc logprobs_gpu failed: {e}"))?,
            logprobs_host: vec![0.0f32; max_batch_size],

            token_ids_gpu: ctx
                .stream
                .alloc_zeros(max_batch_size)
                .map_err(|e| anyhow::anyhow!("Alloc token_ids_gpu failed: {e}"))?,

            token_ids_scratch: Vec::with_capacity(max_batch_size),

            metadata: FlashInferDecodeMetadata::new(
                ctx,
                max_batch_size,
                max_total_pages,
                num_qheads,
            )?,

            max_batch_size,
            graph_cache: (0..max_batch_size).map(|_| None).collect(),
        })
    }

    /// Set the actual batch size for this step (must be <= max_batch_size).
    fn set_batch_size_inner(&mut self, batch_size: usize) {
        debug_assert!(batch_size <= self.max_batch_size);
        self.hidden_out.seq_len = batch_size;
        self.normed.seq_len = batch_size;
        self.q_batch.seq_len = batch_size;
        self.k_batch.seq_len = batch_size;
        self.v_batch.seq_len = batch_size;
        self.qkv_batch.seq_len = batch_size;
        self.attn_output.seq_len = batch_size;
        self.o_buf.seq_len = batch_size;
        self.gate_out.seq_len = batch_size;
        self.up_out.seq_len = batch_size;
        self.gate_up_out.seq_len = batch_size;
        self.act_out.seq_len = batch_size;
    }
}

impl crate::model::DecodeContextOps for BatchDecodeBuffers {
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
        pool: &crate::backend::cuda::paged_kv::PagedKVPool,
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
        // Only BF16 uses FlashInfer plan. FP8/INT8 use our fused-dequant kernel
        // which doesn't need FlashInfer's work estimation.
        if kv_format == KVFormat::BF16 {
            self.metadata.plan(
                ctx,
                batch_size,
                num_q_heads,
                num_kv_heads,
                page_size,
                head_dim,
            )?;
        }
        Ok(())
    }

    fn set_batch_size(&mut self, bs: usize) {
        self.set_batch_size_inner(bs);
    }

    fn invalidate_graph_cache(&mut self, batch_size: usize) {
        if batch_size >= 1 && batch_size <= self.graph_cache.len() {
            self.graph_cache[batch_size - 1] = None;
        }
    }

    fn logprobs_host(&self) -> &[f32] {
        &self.logprobs_host
    }
}

impl Qwen3Model {
    /// Batched decode: process B tokens from B different requests in one pass.
    ///
    /// Batched decode using contiguous (per-slot) KV cache.
    /// Falls back to sequential forward_decode() calls — correct but not optimal.
    pub fn decode_batch_contiguous(
        &self,
        tokens: &[u32],
        states: &mut [Qwen3State],
        slot_indices: &[usize],
    ) -> Result<()> {
        for (i, &token) in tokens.iter().enumerate() {
            self.forward_decode(token, &mut states[slot_indices[i]])?;
        }
        Ok(())
    }

    /// `tokens[b]` is the next token for request `b`, whose state is
    /// `states[slot_indices[b]]`. All linear projections are batched via GEMM;
    /// attention uses FlashInfer with a shared paged KV cache.
    pub fn decode_batch(
        &self,
        tokens: &[u32],
        states: &mut [Qwen3State],
        slot_indices: &[usize],
        skip_logit_scatter: bool,
        paged_kv_pool: &mut PagedKVPool,
        bufs: &mut BatchDecodeBuffers,
    ) -> Result<()> {
        let batch_size = tokens.len();
        debug_assert_eq!(batch_size, slot_indices.len());
        debug_assert!(batch_size >= 1);
        debug_assert!(batch_size <= bufs.max_batch_size);

        // NOTE: set_batch_size, upload_token_ids, update_metadata, and
        // plan_attention are now called by the scheduler via DecodeContextOps
        // before this method is invoked.

        bufs.embedding_out.seq_len = batch_size;

        // ── Graph body: embedding + layers + final norm + logits GEMM ──
        // Embedding reads from token_ids_gpu (H2D done above, pointer is stable).
        // All use pre-allocated buffers with stable pointers.

        // Lazy-init logits buffer (allocation — must be before any graph capture)
        if bufs.logits_batch.is_none() {
            let vocab_size = self.output_projection().rows;
            bufs.logits_batch = Some(HiddenStates::zeros(
                &self.ctx,
                vocab_size,
                bufs.max_batch_size,
            )?);
        }

        // ── CUDA Graph: capture on first call per batch_size, replay on subsequent ──
        // plan() was called by the scheduler before this method (updates
        // int_workspace). graph_body only does kernel launches — no allocs, no
        // H2D, no CPU memcpy.
        if let Some(ref graph) = bufs.graph_cache[batch_size - 1] {
            graph
                .launch()
                .map_err(|e| anyhow::anyhow!("CUDA Graph replay (B={}): {e}", batch_size))?;
        } else {
            info!(
                "Capturing CUDA Graph for batched decode B={}...",
                batch_size
            );
            self.ctx
                .stream
                .begin_capture(CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .map_err(|e| anyhow::anyhow!("begin_capture: {e}"))?;

            self.decode_batch_graph_body(bufs, paged_kv_pool, batch_size)?;

            let graph_opt = self
                .ctx
                .stream
                .end_capture(CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
                .map_err(|e| anyhow::anyhow!("end_capture: {e}"))?;

            if let Some(graph) = graph_opt {
                graph
                    .launch()
                    .map_err(|e| anyhow::anyhow!("Graph first launch (B={}): {e}", batch_size))?;
                info!("CUDA Graph captured for batched decode B={}", batch_size);
                bufs.graph_cache[batch_size - 1] = Some(graph);
            } else {
                // Fallback: capture returned None (shouldn't happen)
                self.decode_batch_graph_body(bufs, paged_kv_pool, batch_size)?;
            }
        }

        // Scatter per-slot logits only when needed (non-greedy fallback).
        if !skip_logit_scatter {
            let logits = bufs.logits_batch.as_ref().unwrap();
            for (b, &si) in slot_indices.iter().enumerate() {
                ops::extract_vec_into(&self.ctx, logits, b, &mut states[si].decode_bufs.logits)?;
                states[si].base.prefill_logits = None;
            }
        }

        Ok(())
    }

    /// Graph body: embedding → layers → final norm → logits.
    /// All buffers are pre-allocated in `bufs`. No allocations, no H2D copies.
    /// Embedding reads from token_ids_gpu (H2D done before graph, pointer stable).
    fn decode_batch_graph_body(
        &self,
        bufs: &mut BatchDecodeBuffers,
        kv_pool: &PagedKVPool,
        batch_size: usize,
    ) -> Result<()> {
        let eps = self.config.rms_norm_eps;

        // Embedding (reads from pre-allocated token_ids_gpu, written by H2D before graph)
        ops::embedding_batch(
            &self.ctx,
            &self.embed_tokens,
            &bufs.token_ids_gpu,
            &mut bufs.embedding_out,
        )?;

        // Use embedding_out as the initial hidden state. The layer loop
        // ping-pongs between embedding_out and hidden_out via swap.
        // We use a raw pointer to avoid borrow conflicts with bufs.
        let hidden_ptr = &mut bufs.embedding_out as *mut HiddenStates;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // SAFETY: hidden_ptr points to bufs.embedding_out. The layer
            // function only accesses other fields of bufs (normed, q_batch, etc.)
            // and swaps hidden_ptr's target with bufs.hidden_out. No aliasing.
            let hidden = unsafe { &mut *hidden_ptr };
            self.decode_batch_layer_inner(layer_idx, layer, hidden, bufs, kv_pool)?;
        }

        // Final norm + logits. hidden is whichever buffer was last written.
        let hidden = unsafe { &*hidden_ptr };
        ops::rms_norm_batch_into(&self.ctx, hidden, &self.norm, eps, &mut bufs.normed);
        let logits_buf = bufs.logits_batch.as_mut().unwrap();
        logits_buf.seq_len = batch_size;
        ops::gemm_into(
            &self.ctx,
            self.output_projection(),
            &bufs.normed,
            logits_buf,
        );

        Ok(())
    }

    fn decode_batch_layer_inner(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        hidden: &mut HiddenStates,
        bufs: &mut BatchDecodeBuffers,
        kv_pool: &PagedKVPool,
    ) -> Result<()> {
        let eps = self.config.rms_norm_eps;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let page_size = 1; // token-level pool: page_size is always 1

        // 1. Batched RMSNorm → bufs.normed [B, hidden_dim]
        ops::rms_norm_batch_into(
            &self.ctx,
            hidden,
            &layer.input_layernorm,
            eps,
            &mut bufs.normed,
        );

        // 2. QKV projection
        if layer.attention.q_proj.is_quantized() {
            // Quantized: use 3 separate GEMVs (merged QKV concat doesn't work for quantized)
            ops::gemm_into(
                &self.ctx,
                &layer.attention.q_proj,
                &bufs.normed,
                &mut bufs.q_batch,
            );
            ops::gemm_into(
                &self.ctx,
                &layer.attention.k_proj,
                &bufs.normed,
                &mut bufs.k_batch,
            );
            ops::gemm_into(
                &self.ctx,
                &layer.attention.v_proj,
                &bufs.normed,
                &mut bufs.v_batch,
            );
        } else {
            // BF16: single merged GEMM + split (3x fewer kernel launches)
            ops::gemm_into(
                &self.ctx,
                &layer.attention.qkv_proj,
                &bufs.normed,
                &mut bufs.qkv_batch,
            );
            ops::split_qkv_batch(
                &self.ctx,
                &bufs.qkv_batch,
                &mut bufs.q_batch,
                &mut bufs.k_batch,
                &mut bufs.v_batch,
            )?;
        }

        // 3. Decode prep: QK-norm + RoPE (in-place on Q) + paged KV write
        let nrp = ops::NormRopeParams {
            q_norm: &layer.attention.q_norm,
            k_norm: &layer.attention.k_norm,
            cos_cache: &self.cos_cache,
            sin_cache: &self.sin_cache,
            rms_eps: eps,
        };
        let paged = ops::PagedKVMeta {
            kv_pool,
            layer_idx,
            kv_indices: &bufs.metadata.kv_indices,
            kv_indptr: &bufs.metadata.kv_indptr,
            kv_last_page_len: &bufs.metadata.kv_last_page_len,
            page_size,
        };
        ops::decode_prep_paged(
            &self.ctx,
            &mut bufs.q_batch,
            &bufs.k_batch,
            &bufs.v_batch,
            &nrp,
            &bufs.metadata.positions,
            &paged,
            num_heads,
            num_kv_heads,
        )?;

        // 4. Attention dispatch — format-aware
        //
        // FP8/INT8: quantize new token from bf16 working → pool, then attention
        //   reads directly from quantized pool (zero full-dequant).
        // BF16: FlashInfer reads bf16 pool directly (decode_prep already wrote there).
        {
            let batch_size = bufs.q_batch.seq_len;
            let stream = &self.ctx.stream;

            // Quantize new token into pool (FP8/INT8 only — bf16 wrote directly)
            match kv_pool.format {
                KVFormat::FP8E4M3 => {
                    kv_quant::quantize_paged_kv_fp8(
                        &self.ctx,
                        kv_pool.k_work_ptr(stream),
                        kv_pool.k_data_ptr(layer_idx, stream),
                        &bufs.metadata.last_token_indices,
                        num_kv_heads,
                        head_dim,
                        kv_pool.kv_dim,
                        batch_size,
                    )?;
                    kv_quant::quantize_paged_kv_fp8(
                        &self.ctx,
                        kv_pool.v_work_ptr(stream),
                        kv_pool.v_data_ptr(layer_idx, stream),
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
                        kv_pool.k_data_ptr(layer_idx, stream),
                        kv_pool.k_scales_ptr(layer_idx, stream),
                        &bufs.metadata.last_token_indices,
                        num_kv_heads,
                        head_dim,
                        kv_pool.kv_dim,
                        batch_size,
                    )?;
                    kv_quant::quantize_paged_kv_single(
                        &self.ctx,
                        kv_pool.v_work_ptr(stream),
                        kv_pool.v_data_ptr(layer_idx, stream),
                        kv_pool.v_scales_ptr(layer_idx, stream),
                        &bufs.metadata.last_token_indices,
                        num_kv_heads,
                        head_dim,
                        kv_pool.kv_dim,
                        batch_size,
                    )?;
                }
                KVFormat::BF16 => {} // decode_prep already wrote bf16 to pool
                KVFormat::TurboQuant { .. } => {
                    // Quantize new K token: bf16 working → TQ packed pool
                    let tq_k = kv_pool.tq_k_state.as_ref().unwrap();
                    kv_turboquant::turboquant_quantize_paged_single(
                        &self.ctx,
                        kv_pool.k_work_ptr(stream),
                        kv_pool.k_data_slice(layer_idx),
                        kv_pool.k_norms_slice(layer_idx),
                        &bufs.metadata.last_token_indices,
                        tq_k,
                        layer_idx,
                        num_kv_heads,
                        head_dim,
                        batch_size,
                    )?;
                    // Quantize new V token
                    let tq_v = kv_pool.tq_v_state.as_ref().unwrap();
                    kv_turboquant::turboquant_quantize_paged_single(
                        &self.ctx,
                        kv_pool.v_work_ptr(stream),
                        kv_pool.v_data_slice(layer_idx),
                        kv_pool.v_norms_slice(layer_idx),
                        &bufs.metadata.last_token_indices,
                        tq_v,
                        layer_idx,
                        num_kv_heads,
                        head_dim,
                        batch_size,
                    )?;
                }
            }

            // Attention: read from quantized pool
            match kv_pool.format {
                KVFormat::FP8E4M3 => {
                    // Fused-dequant FP8 — reads FP8 E4M3 from pool, casts in registers
                    let sm_scale = 1.0 / (head_dim as f32).sqrt();
                    kv_quant::decode_attention_fp8(
                        &self.ctx,
                        &bufs.q_batch,
                        kv_pool.k_data_ptr(layer_idx, stream),
                        kv_pool.v_data_ptr(layer_idx, stream),
                        &bufs.metadata.kv_indices,
                        &bufs.metadata.kv_indptr,
                        &mut bufs.attn_output,
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
                KVFormat::INT8 => {
                    // Fused-dequant decode attention — reads INT8+scale from pool directly
                    let sm_scale = 1.0 / (head_dim as f32).sqrt();
                    kv_quant::decode_attention_int8(
                        &self.ctx,
                        &bufs.q_batch,
                        kv_pool.k_data_ptr(layer_idx, stream),
                        kv_pool.v_data_ptr(layer_idx, stream),
                        kv_pool.k_scales_ptr(layer_idx, stream),
                        kv_pool.v_scales_ptr(layer_idx, stream),
                        &bufs.metadata.kv_indices,
                        &bufs.metadata.kv_indptr,
                        &mut bufs.attn_output,
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
                    ops::flashinfer_run_layer(
                        &self.ctx,
                        &bufs.q_batch,
                        kv_pool,
                        layer_idx,
                        &bufs.metadata.kv_indptr,
                        &bufs.metadata.kv_indices,
                        &bufs.metadata.kv_last_page_len,
                        &mut bufs.attn_output,
                        &mut bufs.metadata.flashinfer_ws,
                        num_heads,
                        num_kv_heads,
                        page_size,
                        head_dim,
                    )?;
                }
                KVFormat::TurboQuant { .. } => {
                    // Fused TQ attention: rotate Q once, score from packed K centroids.
                    // Avoids O(seq_len × D log D) full dequant per layer.
                    let tq_k = kv_pool.tq_k_state.as_ref().unwrap();
                    let tq_v = kv_pool.tq_v_state.as_ref().unwrap();
                    let sm_scale = 1.0 / (head_dim as f32).sqrt();

                    // Step 1: Rotate Q → Q_rot (sign flip + FWHT)
                    use cudarc::driver::{DevicePtr, DevicePtrMut};
                    let q_ptr = {
                        let (p, _g) = bufs.q_batch.data.device_ptr(stream);
                        p as u64
                    };
                    let q_rot_ptr = {
                        let (p, _g) = bufs.q_rot.data.device_ptr_mut(stream);
                        p as u64
                    };
                    kv_turboquant::turboquant_rotate_query(
                        &self.ctx,
                        q_ptr,
                        q_rot_ptr,
                        tq_k,
                        layer_idx,
                        batch_size * num_heads,
                        head_dim,
                    )?;

                    // Step 2: Fused attention: score from packed K, dequant V in-kernel
                    let attn_ptr = {
                        let (p, _g) = bufs.attn_output.data.device_ptr_mut(stream);
                        p as u64
                    };
                    kv_turboquant::turboquant_fused_decode_attention(
                        &self.ctx,
                        q_rot_ptr,
                        kv_pool.k_data_slice(layer_idx),
                        kv_pool.k_norms_slice(layer_idx),
                        kv_pool.v_data_slice(layer_idx),
                        kv_pool.v_norms_slice(layer_idx),
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

        // 5. Batched O projection → bufs.o_buf [B, hidden_dim]
        ops::gemm_into(
            &self.ctx,
            &layer.attention.o_proj,
            &bufs.attn_output,
            &mut bufs.o_buf,
        );

        // 6+7. Fused residual add + MLP RMSNorm:
        //   hidden += o_buf (in-place), normed = rms_norm(hidden, weight)
        //   Saves one global read of hidden vs separate add + swap + norm.
        ops::fused_add_rms_norm_batch_into(
            &self.ctx,
            hidden,
            &bufs.o_buf,
            &layer.post_attention_layernorm,
            eps,
            &mut bufs.normed,
        );

        // 8. Batched MLP: gate + up projections → fused silu_mul → down
        if layer.mlp.gate_proj.is_quantized() {
            // Quantized: separate gate + up GEMVs (merged concat doesn't work)
            ops::gemm_into(
                &self.ctx,
                &layer.mlp.gate_proj,
                &bufs.normed,
                &mut bufs.gate_out,
            );
            ops::gemm_into(
                &self.ctx,
                &layer.mlp.up_proj,
                &bufs.normed,
                &mut bufs.up_out,
            );
            ops::silu_mul_batch_into(&self.ctx, &bufs.gate_out, &bufs.up_out, &mut bufs.act_out)?;
        } else {
            // BF16: merged gate+up GEMM + fused silu_mul from merged buffer
            ops::gemm_into(
                &self.ctx,
                layer
                    .mlp
                    .gate_up_proj
                    .as_ref()
                    .expect("merged gate_up_proj required"),
                &bufs.normed,
                &mut bufs.gate_up_out,
            );
            ops::silu_mul_fused_batch_into(&self.ctx, &bufs.gate_up_out, &mut bufs.act_out)?;
        }
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.down_proj,
            &bufs.act_out,
            &mut bufs.o_buf,
        );

        // 9. Batched residual add
        ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        std::mem::swap(hidden, &mut bufs.hidden_out);

        Ok(())
    }
}
