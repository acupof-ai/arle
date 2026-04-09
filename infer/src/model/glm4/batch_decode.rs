//! Batched decode: process multiple requests' decode tokens in one forward pass.
//!
//! Uses GEMM for all linear projections, batching B requests together.
//! Attention uses FlashInfer with a shared paged KV cache.
//! GLM-4 specifics: QKV bias is added after merged QKV GEMM + split,
//! and no Q/K normalization is applied.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use cudarc::driver::safe::CudaGraph;
use cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
use cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
use log::info;

use super::config::Config;
use super::forward::GLM4State;
use super::weights::{GLM4Model, TransformerBlock};
use crate::flashinfer_metadata::FlashInferDecodeMetadata;
use crate::model::ModelForward;
use crate::model::kv_cache::KVFormat;
use crate::ops;
use crate::ops::kv_quant;
use crate::ops::kv_turboquant;
use crate::paged_kv::PagedKVPool;
use crate::tensor::{DeviceContext, DeviceVec, HiddenStates};

/// Pre-allocated buffers for batched decode, reused across steps.
pub struct BatchDecodeBuffers {
    hidden_out: HiddenStates,
    normed: HiddenStates,
    q_batch: HiddenStates,
    k_batch: HiddenStates,
    v_batch: HiddenStates,
    qkv_batch: HiddenStates,
    attn_output: HiddenStates,
    o_buf: HiddenStates,
    gate_out: HiddenStates,
    up_out: HiddenStates,
    gate_up_out: HiddenStates,
    act_out: HiddenStates,

    embedding_out: HiddenStates,
    pub(super) logits_batch: Option<HiddenStates>,
    logits_per_slot: Vec<DeviceVec>,
    pub(super) argmax_out: CudaSlice<i32>,
    pub(super) argmax_host: Vec<i32>,

    token_ids_gpu: CudaSlice<i32>,
    token_ids_scratch: Vec<i32>,

    pub(crate) metadata: FlashInferDecodeMetadata,

    /// Dummy Q/K norm weights (all ones) for decode_prep_paged.
    /// GLM-4 has no Q/K norm.
    dummy_q_norm: DeviceVec,
    dummy_k_norm: DeviceVec,

    max_batch_size: usize,
    graph_cache: Vec<Option<CudaGraph>>,
}

// SAFETY: BatchDecodeBuffers contains CudaGraph (CUgraphExec) which is !Send.
// Invariant: exclusively accessed from the single scheduler inference thread.
unsafe impl Send for BatchDecodeBuffers {}

impl BatchDecodeBuffers {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &DeviceContext,
        config: &Config,
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
            o_buf: HiddenStates::zeros(ctx, hidden_dim, max_batch_size)?,
            gate_out: HiddenStates::zeros(ctx, inter_dim, max_batch_size)?,
            up_out: HiddenStates::zeros(ctx, inter_dim, max_batch_size)?,
            gate_up_out: HiddenStates::zeros(ctx, 2 * inter_dim, max_batch_size)?,
            act_out: HiddenStates::zeros(ctx, inter_dim, max_batch_size)?,

            embedding_out: HiddenStates::zeros(ctx, hidden_dim, max_batch_size)?,
            logits_batch: None,
            logits_per_slot: Vec::new(),
            argmax_out: ctx
                .stream
                .alloc_zeros(max_batch_size)
                .map_err(|e| anyhow::anyhow!("Alloc argmax_out failed: {e}"))?,
            argmax_host: vec![0i32; max_batch_size],

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

            dummy_q_norm: DeviceVec::ones(ctx, config.head_dim())?,
            dummy_k_norm: DeviceVec::ones(ctx, config.head_dim())?,

            max_batch_size,
            graph_cache: (0..max_batch_size).map(|_| None).collect(),
        })
    }

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
        if kv_format == crate::model::kv_cache::KVFormat::BF16 {
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
        if batch_size > 0 && batch_size <= self.graph_cache.len() {
            self.graph_cache[batch_size - 1] = None;
        }
    }
}

impl GLM4Model {
    pub fn decode_batch_contiguous(
        &self,
        tokens: &[u32],
        states: &mut [GLM4State],
        slot_indices: &[usize],
    ) -> Result<()> {
        for (i, &token) in tokens.iter().enumerate() {
            self.forward_decode(token, &mut states[slot_indices[i]])?;
        }
        Ok(())
    }

    pub fn decode_batch(
        &self,
        tokens: &[u32],
        states: &mut [GLM4State],
        slot_indices: &[usize],
        skip_logit_scatter: bool,
        paged_kv_pool: &mut PagedKVPool,
        bufs: &mut BatchDecodeBuffers,
    ) -> Result<()> {
        let batch_size = tokens.len();
        debug_assert_eq!(batch_size, slot_indices.len());
        debug_assert!(batch_size >= 1);
        debug_assert!(batch_size <= bufs.max_batch_size);

        // Note: token IDs H2D, metadata update, and FlashInfer plan are now
        // handled by the scheduler via DecodeContextOps before this call.

        bufs.embedding_out.seq_len = batch_size;

        // Lazy-init logits buffer
        if bufs.logits_batch.is_none() {
            let vocab_size = self.output_projection().rows;
            bufs.logits_batch = Some(HiddenStates::zeros(
                &self.ctx,
                vocab_size,
                bufs.max_batch_size,
            )?);
        }

        // CUDA Graph capture/replay
        if let Some(ref graph) = bufs.graph_cache[batch_size - 1] {
            graph
                .launch()
                .map_err(|e| anyhow::anyhow!("CUDA Graph replay (B={}): {e}", batch_size))?;
        } else {
            info!(
                "Capturing CUDA Graph for GLM-4 batched decode B={}...",
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
                info!(
                    "CUDA Graph captured for GLM-4 batched decode B={}",
                    batch_size
                );
                bufs.graph_cache[batch_size - 1] = Some(graph);
            } else {
                self.decode_batch_graph_body(bufs, paged_kv_pool, batch_size)?;
            }
        }

        // Scatter per-slot logits only when needed
        if !skip_logit_scatter {
            let logits = bufs.logits_batch.as_ref().unwrap();
            for (b, &si) in slot_indices.iter().enumerate() {
                ops::extract_vec_into(&self.ctx, logits, b, &mut states[si].decode_bufs.logits)?;
                states[si].base.prefill_logits = None;
            }
        }

        Ok(())
    }

    fn decode_batch_graph_body(
        &self,
        bufs: &mut BatchDecodeBuffers,
        kv_pool: &PagedKVPool,
        batch_size: usize,
    ) -> Result<()> {
        let eps = self.config.layernorm_epsilon;

        ops::embedding_batch(
            &self.ctx,
            &self.embed_tokens,
            &bufs.token_ids_gpu,
            &mut bufs.embedding_out,
        )?;

        let hidden_ptr = &mut bufs.embedding_out as *mut HiddenStates;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let hidden = unsafe { &mut *hidden_ptr };
            self.decode_batch_layer_inner(layer_idx, layer, hidden, bufs, kv_pool)?;
        }

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
        let eps = self.config.layernorm_epsilon;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads();
        let head_dim = self.config.head_dim();
        let page_size = 1;

        // 1. Batched RMSNorm
        ops::rms_norm_batch_into(
            &self.ctx,
            hidden,
            &layer.input_layernorm,
            eps,
            &mut bufs.normed,
        );

        // 2. Merged QKV projection
        ops::gemm_into(
            &self.ctx,
            &layer.attention.qkv_proj,
            &bufs.normed,
            &mut bufs.qkv_batch,
        );
        // Add QKV bias (broadcast across batch)
        ops::add_bias_batch_into(&self.ctx, &mut bufs.qkv_batch, &layer.attention.qkv_bias)?;
        // Split into separate Q/K/V
        ops::split_qkv_batch(
            &self.ctx,
            &bufs.qkv_batch,
            &mut bufs.q_batch,
            &mut bufs.k_batch,
            &mut bufs.v_batch,
        )?;

        // 3. Decode prep: RoPE (no QK-norm) + paged KV write
        // GLM-4 has no Q/K norm — use dummy identity norms.
        let nrp = ops::NormRopeParams {
            q_norm: &bufs.dummy_q_norm,
            k_norm: &bufs.dummy_k_norm,
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
        {
            let batch_size = bufs.q_batch.seq_len;
            let stream = &self.ctx.stream;

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
                KVFormat::BF16 => {}
                KVFormat::TurboQuant { .. } => {
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

            match kv_pool.format {
                KVFormat::INT8 => {
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
                KVFormat::FP8E4M3 => {
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
                    let tq_k = kv_pool.tq_k_state.as_ref().unwrap();
                    kv_turboquant::turboquant_dequantize_inplace(
                        &self.ctx,
                        kv_pool.k_data_slice(layer_idx),
                        kv_pool.k_norms_slice(layer_idx),
                        kv_pool.k_work_ptr(stream),
                        &bufs.metadata.kv_indices,
                        tq_k,
                        layer_idx,
                        num_kv_heads,
                        head_dim,
                        bufs.metadata.kv_indices.len(),
                    )?;
                    let tq_v = kv_pool.tq_v_state.as_ref().unwrap();
                    kv_turboquant::turboquant_dequantize_inplace(
                        &self.ctx,
                        kv_pool.v_data_slice(layer_idx),
                        kv_pool.v_norms_slice(layer_idx),
                        kv_pool.v_work_ptr(stream),
                        &bufs.metadata.kv_indices,
                        tq_v,
                        layer_idx,
                        num_kv_heads,
                        head_dim,
                        bufs.metadata.kv_indices.len(),
                    )?;
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
            }
        }

        // 5. O projection
        ops::gemm_into(
            &self.ctx,
            &layer.attention.o_proj,
            &bufs.attn_output,
            &mut bufs.o_buf,
        );

        // 6+7. Fused residual add + MLP RMSNorm
        ops::fused_add_rms_norm_batch_into(
            &self.ctx,
            hidden,
            &bufs.o_buf,
            &layer.post_attention_layernorm,
            eps,
            &mut bufs.normed,
        );

        // 8. Batched MLP: merged gate+up GEMM -> fused silu_mul -> down
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
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.down_proj,
            &bufs.act_out,
            &mut bufs.o_buf,
        );

        // 9. Residual add
        ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        std::mem::swap(hidden, &mut bufs.hidden_out);

        Ok(())
    }
}
