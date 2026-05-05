//! Batched decode: process multiple requests' decode tokens in one forward pass.
//!
//! Uses GEMM (matrix multiply) for all linear projections (QKV, O, MLP),
//! batching B requests together. Attention uses FlashInfer with a shared
//! paged KV cache: QK-norm + RoPE + paged KV write are done in a prep kernel,
//! then FlashInfer's batch decode handles attention in a single launch.

use anyhow::Result;
use cudarc::driver::safe::CudaGraph;
use cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
use cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use log::info;

use super::forward::Qwen3State;
use super::weights::{Qwen3Model, TransformerBlock};
use crate::model::kv_cache::KVFormat;
use crate::model::{MixedBatchRequest, ModelForward};
use crate::ops;
use cuda_kernels::ffi;
use cuda_kernels::flashinfer::{DecodeMetaUpdate, FlashInferWorkspace};
use cuda_kernels::kv_quant;
use cuda_kernels::kv_turboquant;
use cuda_kernels::prelude::{
    DeviceContext, DeviceVec, FlashInferDecodeMetadata, HiddenStates, PagedKVPool,
};

const BF16_BYTES: usize = 2;

/// Mixed-batch FlashInfer float workspace.
///
/// Packed mixed batches can have thousands of QO rows. FlashInfer's
/// internal `batch_prefill_tmp_v` allocation grows with
/// `max_total_tokens × num_kv_heads × head_dim`; on Qwen3-4B at
/// max_total_tokens≈4112 and num_kv_heads=8, head_dim=128 it needs
/// ~270 MiB, so the 256 MiB DEFAULT_FLOAT_WORKSPACE_BYTES tier is too
/// tight even for HD128. Stay on the 640 MiB tier for all head_dims —
/// the activation-side wins from removing vocab_size and the
/// prefill_activation chunk-size fix already free more than this 384 MiB.
const MIXED_FLOAT_WORKSPACE_BYTES: usize = FlashInferWorkspace::HD256_FLOAT_WORKSPACE_BYTES;

fn bf16_matrix_bytes(rows: usize, cols: usize) -> usize {
    rows.saturating_mul(cols).saturating_mul(BF16_BYTES)
}

fn bytes_for<T>(count: usize) -> usize {
    count.saturating_mul(std::mem::size_of::<T>())
}

/// Pre-allocated buffers for batched decode, reused across steps.
/// Allocated once for `max_batch_size`; smaller batches set `seq_len` on HiddenStates.
pub struct BatchDecodeBuffers {
    hidden_out: HiddenStates,
    normed: HiddenStates,
    q_batch: HiddenStates,
    k_batch: HiddenStates,
    v_batch: HiddenStates,
    attn_output: HiddenStates,
    /// Rotated query buffer for TurboQuant fused attention [max_batch_size, q_dim].
    q_rot: HiddenStates,
    o_buf: HiddenStates,
    gate_out: HiddenStates,
    up_out: HiddenStates,
    act_out: HiddenStates,

    /// Embedding output buffer [max_batch_size, hidden_dim] — avoids alloc in graph.
    embedding_out: HiddenStates,
    /// Batched logits buffer [max_batch_size, vocab_size] — avoids alloc in graph.
    pub(super) logits_batch: Option<HiddenStates>,
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
    /// Packed page-aware metadata for quantized decode kernels:
    /// `[page_indptr..., last_page_len...]`.
    quantized_kv_meta: CudaSlice<i32>,
    /// One-shot marker for sparse decode calls whose quantized metadata was
    /// uploaded by `prepare_sparse_decode_context`.
    sparse_quantized_meta_once: bool,
    decode_meta_update: DecodeMetaUpdate,
    quantized_last_page_lens_scratch: Vec<i32>,
    quantized_indptr_scratch: Vec<i32>,

    /// Max batch size this buffer set was allocated for.
    max_batch_size: usize,
    max_total_pages: usize,

    /// CUDA Graph cache: index = batch_size - 1. Vec avoids HashMap overhead.
    graph_cache: Vec<Option<CudaGraph>>,
    /// One-shot eager decode override for verifier/correctness-sensitive paths.
    force_eager_once: bool,

    /// Lazily allocated eager mixed-batch workspace.
    mixed: Option<MixedBatchBuffers>,
}

// SAFETY: BatchDecodeBuffers contains CudaGraph (CUgraphExec) which is !Send.
// Invariant: exclusively accessed from the single scheduler inference thread.
unsafe impl Send for BatchDecodeBuffers {}

pub(crate) struct MixedBatchBuffers {
    embedding_out: HiddenStates,
    hidden_out: HiddenStates,
    normed: HiddenStates,
    q_batch: HiddenStates,
    k_batch: HiddenStates,
    v_batch: HiddenStates,
    attn_output: HiddenStates,
    o_buf: HiddenStates,
    gate_out: HiddenStates,
    up_out: HiddenStates,
    act_out: HiddenStates,
    /// Logits buffer sized for the *kept* output rows only:
    /// `max_logit_rows = max_batch_size`. Mixed-batch forward computes vocab
    /// projections only for decode rows (one per request) and the final token
    /// of each prefill chunk — never the intermediate prefill tokens. The
    /// intermediate rows are gathered out of `normed` before the output
    /// projection, so this stays at `vocab × max_batch_size` instead of
    /// `vocab × max_total_tokens`.
    logits: HiddenStates,
    token_ids_gpu: CudaSlice<i32>,
    quantized: Option<QuantizedMixedBatchBuffers>,
    metadata: FlashInferDecodeMetadata,
    max_tokens: usize,
    max_logit_rows: usize,
    max_total_pages: usize,
}

unsafe impl Send for MixedBatchBuffers {}

pub(crate) struct QuantizedMixedBatchBuffers {
    token_rows_gpu: CudaSlice<i32>,
    token_rows_host: Vec<i32>,
    attn_workspace: CudaSlice<u8>,
    attn_workspace_bytes: usize,
}

impl QuantizedMixedBatchBuffers {
    fn new(
        ctx: &DeviceContext,
        max_tokens: usize,
        num_qheads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let attn_workspace_bytes = kv_quant::decode_attention_varlen_fp8_workspace_bytes(
            max_tokens,
            num_qheads,
            head_dim,
            kv_quant::VARLEN_QUANTIZED_MAX_SPLITS,
        );
        Ok(Self {
            token_rows_gpu: ctx
                .stream
                .alloc_zeros(max_tokens)
                .map_err(|e| anyhow::anyhow!("Alloc mixed quantized token_rows_gpu failed: {e}"))?,
            token_rows_host: Vec::with_capacity(max_tokens),
            attn_workspace: ctx
                .stream
                .alloc_zeros(attn_workspace_bytes)
                .map_err(|e| anyhow::anyhow!("Alloc mixed quantized attn_workspace failed: {e}"))?,
            attn_workspace_bytes,
        })
    }
}

unsafe impl Send for QuantizedMixedBatchBuffers {}

impl MixedBatchBuffers {
    fn new(
        ctx: &DeviceContext,
        model: &Qwen3Model,
        kv_format: KVFormat,
        max_tokens: usize,
        max_logit_rows: usize,
        max_total_pages: usize,
    ) -> Result<Self> {
        let q_dim = model.config.num_attention_heads * model.config.head_dim;
        let kv_dim = model.config.num_key_value_heads * model.config.head_dim;
        let max_logit_rows = max_logit_rows.max(1);

        Ok(Self {
            embedding_out: HiddenStates::zeros(ctx, model.config.hidden_size, max_tokens)?,
            hidden_out: HiddenStates::zeros(ctx, model.config.hidden_size, max_tokens)?,
            normed: HiddenStates::zeros(ctx, model.config.hidden_size, max_tokens)?,
            q_batch: HiddenStates::zeros(ctx, q_dim, max_tokens)?,
            k_batch: HiddenStates::zeros(ctx, kv_dim, max_tokens)?,
            v_batch: HiddenStates::zeros(ctx, kv_dim, max_tokens)?,
            attn_output: HiddenStates::zeros(ctx, q_dim, max_tokens)?,
            o_buf: HiddenStates::zeros(ctx, model.config.hidden_size, max_tokens)?,
            gate_out: HiddenStates::zeros(ctx, model.config.intermediate_size, max_tokens)?,
            up_out: HiddenStates::zeros(ctx, model.config.intermediate_size, max_tokens)?,
            act_out: HiddenStates::zeros(ctx, model.config.intermediate_size, max_tokens)?,
            // Sized for kept output rows only — see field doc above.
            logits: HiddenStates::zeros(ctx, model.config.vocab_size, max_logit_rows)?,
            token_ids_gpu: ctx
                .stream
                .alloc_zeros(max_tokens)
                .map_err(|e| anyhow::anyhow!("Alloc mixed token_ids_gpu failed: {e}"))?,
            quantized: matches!(kv_format, KVFormat::FP8E4M3 | KVFormat::INT8)
                .then(|| {
                    QuantizedMixedBatchBuffers::new(
                        ctx,
                        max_tokens,
                        model.config.num_attention_heads,
                        model.config.head_dim,
                    )
                })
                .transpose()?,
            metadata: FlashInferDecodeMetadata::new_with_float_workspace_bytes(
                ctx,
                max_tokens,
                max_total_pages,
                model.config.num_attention_heads,
                MIXED_FLOAT_WORKSPACE_BYTES,
            )?,
            max_tokens,
            max_logit_rows,
            max_total_pages,
        })
    }

    /// Set the per-token sequence length for buffers shared by all rows
    /// in the mixed batch (decode + every prefill token). The `logits`
    /// buffer follows a different schedule — it covers only the *kept*
    /// output rows (decode + final token of each prefill) and is sized
    /// separately by the forward path.
    fn set_seq_len(&mut self, seq_len: usize) {
        self.embedding_out.seq_len = seq_len;
        self.hidden_out.seq_len = seq_len;
        self.normed.seq_len = seq_len;
        self.q_batch.seq_len = seq_len;
        self.k_batch.seq_len = seq_len;
        self.v_batch.seq_len = seq_len;
        self.attn_output.seq_len = seq_len;
        self.o_buf.seq_len = seq_len;
        self.gate_out.seq_len = seq_len;
        self.up_out.seq_len = seq_len;
        self.act_out.seq_len = seq_len;
    }
}

impl BatchDecodeBuffers {
    pub(crate) fn device_bytes(
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        inter_dim: usize,
        max_batch_size: usize,
        num_qheads: usize,
        max_total_pages: usize,
    ) -> usize {
        // Buffers in BatchDecodeBuffers::new:
        //   4×hidden_dim (hidden_out, normed, embedding_out, o_buf)
        //   3×q_dim      (q_batch, attn_output, q_rot)
        //   2×kv_dim     (k_batch, v_batch)
        //   3×inter_dim  (gate_out, up_out, act_out)
        let activation_dims = 4usize
            .saturating_mul(hidden_dim)
            .saturating_add(3usize.saturating_mul(q_dim))
            .saturating_add(2usize.saturating_mul(kv_dim))
            .saturating_add(3usize.saturating_mul(inter_dim));

        bf16_matrix_bytes(activation_dims, max_batch_size)
            .saturating_add(bytes_for::<i32>(max_batch_size)) // argmax_out
            .saturating_add(bytes_for::<f32>(max_batch_size)) // logprobs_gpu
            .saturating_add(bytes_for::<i32>(max_batch_size)) // token_ids_gpu
            .saturating_add(bytes_for::<i32>(2 * max_batch_size + 1)) // quantized_kv_meta
            .saturating_add(FlashInferDecodeMetadata::device_bytes(
                max_batch_size,
                max_total_pages,
                num_qheads,
            ))
    }

    pub(crate) fn logits_device_bytes(vocab_size: usize, max_batch_size: usize) -> usize {
        bf16_matrix_bytes(vocab_size, max_batch_size)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mixed_device_bytes(
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        inter_dim: usize,
        vocab_size: usize,
        kv_format: KVFormat,
        max_total_tokens: usize,
        max_logit_rows: usize,
        num_qheads: usize,
        max_total_pages: usize,
    ) -> usize {
        // Per-token activation buffers (every row in the mixed batch):
        //   4×hidden_dim (embedding_out, hidden_out, normed, o_buf)
        //   2×q_dim      (q_batch, attn_output)
        //   2×kv_dim     (k_batch, v_batch)
        //   3×inter_dim  (gate_out, up_out, act_out)
        let activation_dims = 4usize
            .saturating_mul(hidden_dim)
            .saturating_add(2usize.saturating_mul(q_dim))
            .saturating_add(2usize.saturating_mul(kv_dim))
            .saturating_add(3usize.saturating_mul(inter_dim));

        let mut total = bf16_matrix_bytes(activation_dims, max_total_tokens)
            // Logits buffer — sized for kept output rows only (decode rows +
            // one final-token row per prefill), not every prefill token.
            .saturating_add(bf16_matrix_bytes(vocab_size, max_logit_rows))
            .saturating_add(bytes_for::<i32>(max_total_tokens))
            .saturating_add(FlashInferDecodeMetadata::device_bytes_with_float_workspace(
                max_total_tokens,
                max_total_pages,
                num_qheads,
                MIXED_FLOAT_WORKSPACE_BYTES,
            ));
        if matches!(kv_format, KVFormat::FP8E4M3 | KVFormat::INT8) {
            total = total
                .saturating_add(bytes_for::<i32>(max_total_tokens))
                .saturating_add(kv_quant::decode_attention_varlen_fp8_workspace_bytes(
                    max_total_tokens,
                    num_qheads,
                    q_dim / num_qheads,
                    kv_quant::VARLEN_QUANTIZED_MAX_SPLITS,
                ));
        }
        total
    }

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
            attn_output: HiddenStates::zeros(ctx, q_dim, max_batch_size)?,
            q_rot: HiddenStates::zeros(ctx, q_dim, max_batch_size)?,
            o_buf: HiddenStates::zeros(ctx, hidden_dim, max_batch_size)?,
            gate_out: HiddenStates::zeros(ctx, inter_dim, max_batch_size)?,
            up_out: HiddenStates::zeros(ctx, inter_dim, max_batch_size)?,
            act_out: HiddenStates::zeros(ctx, inter_dim, max_batch_size)?,

            embedding_out: HiddenStates::zeros(ctx, hidden_dim, max_batch_size)?,
            logits_batch: None, // lazy-allocated on first use (needs vocab_size)
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
            quantized_kv_meta: ctx
                .stream
                .alloc_zeros(2 * max_batch_size + 1)
                .map_err(|e| anyhow::anyhow!("Alloc quantized_kv_meta failed: {e}"))?,
            sparse_quantized_meta_once: false,
            decode_meta_update: DecodeMetaUpdate::Full,
            quantized_last_page_lens_scratch: Vec::with_capacity(max_batch_size),
            quantized_indptr_scratch: Vec::with_capacity(max_batch_size + 1),

            max_batch_size,
            max_total_pages,
            graph_cache: (0..max_batch_size).map(|_| None).collect(),
            force_eager_once: false,
            mixed: None,
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
        self.attn_output.seq_len = batch_size;
        self.o_buf.seq_len = batch_size;
        self.gate_out.seq_len = batch_size;
        self.up_out.seq_len = batch_size;
        self.act_out.seq_len = batch_size;
    }

    fn upload_sparse_quantized_meta(
        &mut self,
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        slot_idx: usize,
        sparse_page_indices: &[u32],
    ) -> Result<()> {
        let slot_pages = pool.page_indices(slot_idx);
        let tail_page = slot_pages.last().copied().ok_or_else(|| {
            anyhow::anyhow!("sparse quantized decode slot {slot_idx} has no pages")
        })?;
        let last_page_len = if sparse_page_indices.last().copied() == Some(tail_page) {
            pool.build_last_page_lens(&[slot_idx])[0]
        } else {
            pool.page_size as i32
        };
        let packed = [0, sparse_page_indices.len() as i32, last_page_len];
        ctx.stream
            .memcpy_htod(&packed, &mut self.quantized_kv_meta)
            .map_err(|e| anyhow::anyhow!("H2D sparse quantized_kv_meta: {e}"))?;
        self.sparse_quantized_meta_once = true;
        Ok(())
    }

    fn upload_quantized_last_page_lens(
        &mut self,
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        slot_indices: &[usize],
    ) -> Result<()> {
        pool.fill_last_page_lens(slot_indices, &mut self.quantized_last_page_lens_scratch);
        let offset = slot_indices.len() + 1;
        let mut last_page_len_view = self
            .quantized_kv_meta
            .slice_mut(offset..offset + slot_indices.len());
        ctx.stream
            .memcpy_htod(
                &self.quantized_last_page_lens_scratch,
                &mut last_page_len_view,
            )
            .map_err(|e| anyhow::anyhow!("H2D quantized last_page_len: {e}"))?;
        Ok(())
    }

    fn upload_quantized_indptr_and_last_page_lens(
        &mut self,
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        slot_indices: &[usize],
    ) -> Result<()> {
        pool.fill_indptr(slot_indices, &mut self.quantized_indptr_scratch);
        let mut indptr_view = self
            .quantized_kv_meta
            .slice_mut(..self.quantized_indptr_scratch.len());
        ctx.stream
            .memcpy_htod(&self.quantized_indptr_scratch, &mut indptr_view)
            .map_err(|e| anyhow::anyhow!("H2D quantized indptr: {e}"))?;
        self.upload_quantized_last_page_lens(ctx, pool, slot_indices)
    }

    fn ensure_mixed_buffers(
        &mut self,
        model: &Qwen3Model,
        kv_format: KVFormat,
        min_total_tokens: usize,
    ) -> Result<&mut MixedBatchBuffers> {
        let needs_quantized = matches!(kv_format, KVFormat::FP8E4M3 | KVFormat::INT8);
        let needs_realloc = self.mixed.as_ref().is_none_or(|mixed| {
            mixed.max_tokens < min_total_tokens
                || mixed.max_logit_rows < self.max_batch_size
                || mixed.max_total_pages < self.max_total_pages
                || (needs_quantized && mixed.quantized.is_none())
        });
        if needs_realloc {
            self.mixed = Some(MixedBatchBuffers::new(
                &model.ctx,
                model,
                kv_format,
                min_total_tokens.max(self.max_batch_size),
                self.max_batch_size,
                self.max_total_pages,
            )?);
        }
        Ok(self.mixed.as_mut().expect("mixed buffers allocated"))
    }
}

fn upload_decode_quantized_meta(
    ctx: &DeviceContext,
    pool: &PagedKVPool,
    slot_indices: &[usize],
    bufs: &mut BatchDecodeBuffers,
) -> Result<()> {
    if std::mem::take(&mut bufs.sparse_quantized_meta_once) {
        return Ok(());
    }
    if pool.format == KVFormat::FP8E4M3 {
        match bufs.decode_meta_update {
            DecodeMetaUpdate::SamePages => {
                return bufs.upload_quantized_last_page_lens(ctx, pool, slot_indices);
            }
            DecodeMetaUpdate::AppendedPages => {
                return bufs.upload_quantized_indptr_and_last_page_lens(ctx, pool, slot_indices);
            }
            DecodeMetaUpdate::Full => {}
        }
    }
    let packed = pool.build_quantized_decode_indptr(slot_indices);
    ctx.stream
        .memcpy_htod(&packed, &mut bufs.quantized_kv_meta)
        .map_err(|e| anyhow::anyhow!("H2D quantized_kv_meta: {e}"))?;
    Ok(())
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
        self.sparse_quantized_meta_once = false;
        let (reallocated, mode) = self.metadata.update(ctx, pool, slot_indices)?;
        self.decode_meta_update = mode;
        Ok(reallocated)
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
        // Under `tilelang-attn`, the TC decode path is aliased onto the
        // TileLang prefill HD128 cubins, which are plan-less — skip the plan.
        #[cfg(not(feature = "tilelang-attn"))]
        if kv_format == KVFormat::BF16 {
            self.metadata.tc_plan(
                ctx,
                batch_size,
                num_q_heads,
                num_kv_heads,
                page_size,
                head_dim,
            )?;
        }
        #[cfg(feature = "tilelang-attn")]
        let _ = (
            ctx,
            batch_size,
            num_q_heads,
            num_kv_heads,
            page_size,
            head_dim,
            kv_format,
        );
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

    fn force_eager_once(&mut self) {
        self.force_eager_once = true;
    }

    fn logprobs_host(&self) -> &[f32] {
        &self.logprobs_host
    }
}

impl Qwen3Model {
    pub(crate) fn prepare_decode_context(
        &self,
        tokens: &[u32],
        slot_indices: &[usize],
        paged_kv_pool: &PagedKVPool,
        bufs: &mut BatchDecodeBuffers,
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

    pub(crate) fn prepare_sparse_decode_context(
        &self,
        token: u32,
        slot_idx: usize,
        sparse_page_indices: &[u32],
        paged_kv_pool: &PagedKVPool,
        bufs: &mut BatchDecodeBuffers,
    ) -> Result<()> {
        use crate::model::DecodeContextOps;

        bufs.set_batch_size(1);
        bufs.upload_token_ids(&self.ctx, &[token])?;
        let reallocated = bufs.metadata.update_sparse_single(
            &self.ctx,
            paged_kv_pool,
            slot_idx,
            sparse_page_indices,
        )?;
        if reallocated {
            bufs.invalidate_graph_cache(1);
        }
        if matches!(paged_kv_pool.format, KVFormat::INT8 | KVFormat::FP8E4M3) {
            bufs.upload_sparse_quantized_meta(
                &self.ctx,
                paged_kv_pool,
                slot_idx,
                sparse_page_indices,
            )?;
        }
        bufs.force_eager_once();
        bufs.plan_attention(
            &self.ctx,
            1,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            paged_kv_pool.page_size,
            self.config.head_dim,
            paged_kv_pool.format,
        )?;
        Ok(())
    }

    pub fn decode_batch_with_prefill(
        &self,
        batch: MixedBatchRequest<'_>,
        states: &mut [Qwen3State],
        paged_kv_pool: &mut PagedKVPool,
        bufs: &mut BatchDecodeBuffers,
    ) -> Result<bool> {
        if self.lora.is_some() {
            return Ok(false);
        }
        let kv_format = paged_kv_pool.format;
        if !matches!(
            kv_format,
            KVFormat::BF16 | KVFormat::FP8E4M3 | KVFormat::INT8
        ) {
            return Ok(false);
        }
        let b = batch.decode_tokens.len();
        let prefill_count = batch.prefills.len();
        if b == 0
            || b != batch.decode_slot_indices.len()
            || prefill_count == 0
            || prefill_count != batch.prefill_start_positions.len()
        {
            return Ok(false);
        }

        let mut prefill_slot_indices = Vec::with_capacity(prefill_count);
        let mut prefill_token_counts = Vec::with_capacity(prefill_count);
        let mut total_prefill_tokens = 0usize;
        for (prefill, &start_pos) in batch.prefills.iter().zip(batch.prefill_start_positions) {
            if prefill.tokens.is_empty()
                || batch.decode_slot_indices.contains(&prefill.slot_idx)
                || prefill_slot_indices.contains(&prefill.slot_idx)
                || paged_kv_pool.seq_len(prefill.slot_idx) != start_pos
            {
                return Ok(false);
            }
            prefill_slot_indices.push(prefill.slot_idx);
            prefill_token_counts.push(prefill.tokens.len());
            total_prefill_tokens += prefill.tokens.len();
        }

        if bufs.logits_batch.is_none() {
            bufs.logits_batch = Some(HiddenStates::zeros(
                &self.ctx,
                self.output_projection().rows,
                bufs.max_batch_size,
            )?);
        }
        let logits_batch_ptr = std::ptr::from_mut(
            bufs.logits_batch
                .as_mut()
                .expect("decode logits buffer initialized before mixed forward"),
        );

        let total_tokens = b + total_prefill_tokens;
        let mixed = bufs.ensure_mixed_buffers(self, kv_format, total_tokens)?;
        mixed.set_seq_len(total_tokens);

        for prefill in batch.prefills {
            paged_kv_pool.cow_tail_page_for_append(&self.ctx, prefill.slot_idx)?;
            paged_kv_pool.alloc_tokens(prefill.slot_idx, prefill.tokens.len())?;
        }

        let mut combined_tokens = Vec::with_capacity(total_tokens);
        combined_tokens.extend(batch.decode_tokens.iter().map(|&tok| tok as i32));
        for prefill in batch.prefills {
            combined_tokens.extend(prefill.tokens.iter().map(|&tok| tok as i32));
        }
        self.ctx
            .stream
            .memcpy_htod(&combined_tokens, &mut mixed.token_ids_gpu)
            .map_err(|e| anyhow::anyhow!("H2D mixed token_ids: {e}"))?;

        mixed.metadata.update_mixed_batch(
            &self.ctx,
            paged_kv_pool,
            batch.decode_slot_indices,
            &prefill_slot_indices,
            batch.prefill_start_positions,
            &prefill_token_counts,
        )?;
        let max_kv_len = batch
            .decode_slot_indices
            .iter()
            .chain(prefill_slot_indices.iter())
            .map(|&slot| paged_kv_pool.seq_len(slot))
            .max()
            .unwrap_or(0);
        if matches!(kv_format, KVFormat::FP8E4M3 | KVFormat::INT8) {
            let quantized = mixed
                .quantized
                .as_mut()
                .expect("quantized mixed buffers allocated");
            quantized.token_rows_host.clear();
            quantized
                .token_rows_host
                .extend(paged_kv_pool.build_last_indices(batch.decode_slot_indices));
            for ((&slot, &start_pos), &token_count) in prefill_slot_indices
                .iter()
                .zip(batch.prefill_start_positions.iter())
                .zip(prefill_token_counts.iter())
            {
                quantized.token_rows_host.extend(
                    paged_kv_pool
                        .token_rows_for_range(slot, start_pos, token_count)
                        .into_iter()
                        .map(|row| row as i32),
                );
            }
            debug_assert_eq!(quantized.token_rows_host.len(), total_tokens);
            self.ctx
                .stream
                .memcpy_htod(&quantized.token_rows_host, &mut quantized.token_rows_gpu)
                .map_err(|e| anyhow::anyhow!("H2D mixed quantized token rows: {e}"))?;
        }
        // TileLang TC decode alias is plan-less; skip FlashInfer planning.
        #[cfg(not(feature = "tilelang-attn"))]
        mixed.metadata.tc_plan(
            &self.ctx,
            b + prefill_count,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            paged_kv_pool.page_size,
            self.config.head_dim,
        )?;

        ops::embedding_batch(
            &self.ctx,
            &self.embed_tokens,
            &mixed.token_ids_gpu,
            &mut mixed.embedding_out,
        )?;

        let mut prefill_page_table_devs: Vec<CudaSlice<i32>> = Vec::with_capacity(prefill_count);
        for prefill in batch.prefills {
            let prefill_page_table_host: Vec<i32> = paged_kv_pool
                .page_indices(prefill.slot_idx)
                .iter()
                .map(|&idx| idx as i32)
                .collect();
            prefill_page_table_devs.push(
                self.ctx
                    .stream
                    .clone_htod(&prefill_page_table_host)
                    .map_err(|e| anyhow::anyhow!("H2D prefill_page_table: {e}"))?,
            );
        }

        let hidden_ptr = &raw mut mixed.embedding_out;
        let eps = self.config.rms_norm_eps;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let bf16_size = std::mem::size_of::<u16>();
        let page_size = paged_kv_pool.page_size;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let hidden = unsafe { &mut *hidden_ptr };
            let skip_input_norm = layer_idx > 0;
            let next_input_norm = self
                .layers
                .get(layer_idx + 1)
                .map(|next_layer| &next_layer.input_layernorm);

            if !skip_input_norm {
                ops::rms_norm_batch_into(
                    &self.ctx,
                    hidden,
                    &layer.input_layernorm,
                    eps,
                    &mut mixed.normed,
                );
            }

            ops::gemm_into(
                &self.ctx,
                &layer.attention.q_proj,
                &mixed.normed,
                &mut mixed.q_batch,
            );
            ops::gemm_into(
                &self.ctx,
                &layer.attention.k_proj,
                &mixed.normed,
                &mut mixed.k_batch,
            );
            ops::gemm_into(
                &self.ctx,
                &layer.attention.v_proj,
                &mixed.normed,
                &mut mixed.v_batch,
            );

            let nrp = ops::NormRopeParams {
                q_norm: &layer.attention.q_norm,
                k_norm: &layer.attention.k_norm,
                cos_cache: &self.cos_cache,
                sin_cache: &self.sin_cache,
                rms_eps: eps,
            };

            {
                let (q_ptr, _gq) = mixed.q_batch.data.device_ptr_mut(&self.ctx.stream);
                let (k_ptr, _gk) = mixed.k_batch.data.device_ptr(&self.ctx.stream);
                let (v_ptr, _gv) = mixed.v_batch.data.device_ptr(&self.ctx.stream);
                let (qn_ptr, _gqn) = nrp.q_norm.data.device_ptr(&self.ctx.stream);
                let (kn_ptr, _gkn) = nrp.k_norm.data.device_ptr(&self.ctx.stream);
                let (cos_ptr, _gcos) = nrp.cos_cache.data.device_ptr(&self.ctx.stream);
                let (sin_ptr, _gsin) = nrp.sin_cache.data.device_ptr(&self.ctx.stream);
                let (pos_ptr, _gpos) = mixed.metadata.positions.device_ptr(&self.ctx.stream);
                let (ind_ptr, _gind) = mixed.metadata.kv_indptr.device_ptr(&self.ctx.stream);
                let (idx_ptr, _gidx) = mixed.metadata.kv_indices.device_ptr(&self.ctx.stream);
                let (lp_ptr, _glp) = mixed.metadata.kv_last_page_len.device_ptr(&self.ctx.stream);
                let k_pool_ptr = paged_kv_pool.k_ptr(layer_idx, &self.ctx.stream);
                let v_pool_ptr = paged_kv_pool.v_ptr(layer_idx, &self.ctx.stream);

                unsafe {
                    ffi::decode_prep_paged_cuda(
                        q_ptr as *mut ffi::Half,
                        k_ptr as *const ffi::Half,
                        v_ptr as *const ffi::Half,
                        qn_ptr as *const ffi::Half,
                        kn_ptr as *const ffi::Half,
                        cos_ptr as *const ffi::Half,
                        sin_ptr as *const ffi::Half,
                        pos_ptr as *const i32,
                        k_pool_ptr as *mut ffi::Half,
                        v_pool_ptr as *mut ffi::Half,
                        idx_ptr as *const i32,
                        ind_ptr as *const i32,
                        lp_ptr as *const i32,
                        num_heads as i32,
                        num_kv_heads as i32,
                        page_size as i32,
                        (paged_kv_pool.kv_dim * page_size) as i32,
                        b as i32,
                        eps,
                        self.ctx.stream.cu_stream(),
                    )
                    .result()?;
                }

                let mut prefill_token_offset = 0usize;
                for (prefill_idx, prefill) in batch.prefills.iter().enumerate() {
                    let c = prefill.tokens.len();
                    let token_offset = b + prefill_token_offset;
                    let q_prefill_ptr =
                        (q_ptr as usize + token_offset * q_dim * bf16_size) as *mut ffi::Half;
                    let k_prefill_ptr =
                        (k_ptr as usize + token_offset * kv_dim * bf16_size) as *mut ffi::Half;
                    let v_prefill_ptr =
                        (v_ptr as usize + token_offset * kv_dim * bf16_size) as *const ffi::Half;
                    unsafe {
                        ffi::prefill_attention_paged_prep_cuda(
                            q_prefill_ptr,
                            k_prefill_ptr,
                            v_prefill_ptr,
                            qn_ptr as *const ffi::Half,
                            kn_ptr as *const ffi::Half,
                            cos_ptr as *const ffi::Half,
                            sin_ptr as *const ffi::Half,
                            {
                                let (ptr, _g) = prefill_page_table_devs[prefill_idx]
                                    .device_ptr(&self.ctx.stream);
                                ptr as *const i32
                            },
                            page_size as i32,
                            k_pool_ptr as *mut ffi::Half,
                            v_pool_ptr as *mut ffi::Half,
                            num_heads as i32,
                            num_kv_heads as i32,
                            head_dim as i32,
                            c as i32,
                            batch.prefill_start_positions[prefill_idx] as i32,
                            eps,
                            self.ctx.stream.cu_stream(),
                        )
                        .result()?;
                    }
                    prefill_token_offset += c;
                }
            }

            {
                let stream = &self.ctx.stream;
                match kv_format {
                    KVFormat::FP8E4M3 => {
                        let quantized = mixed
                            .quantized
                            .as_ref()
                            .expect("quantized mixed buffers allocated");
                        kv_quant::quantize_paged_kv_fp8(
                            &self.ctx,
                            paged_kv_pool.k_work_ptr(stream),
                            paged_kv_pool.k_data_ptr(layer_idx, stream),
                            paged_kv_pool.k_scales_ptr(layer_idx, stream),
                            &quantized.token_rows_gpu,
                            num_kv_heads,
                            head_dim,
                            paged_kv_pool.kv_dim,
                            total_tokens,
                        )?;
                        kv_quant::quantize_paged_kv_fp8(
                            &self.ctx,
                            paged_kv_pool.v_work_ptr(stream),
                            paged_kv_pool.v_data_ptr(layer_idx, stream),
                            paged_kv_pool.v_scales_ptr(layer_idx, stream),
                            &quantized.token_rows_gpu,
                            num_kv_heads,
                            head_dim,
                            paged_kv_pool.kv_dim,
                            total_tokens,
                        )?;
                    }
                    KVFormat::INT8 => {
                        let quantized = mixed
                            .quantized
                            .as_ref()
                            .expect("quantized mixed buffers allocated");
                        kv_quant::quantize_paged_kv_single(
                            &self.ctx,
                            paged_kv_pool.k_work_ptr(stream),
                            paged_kv_pool.k_data_ptr(layer_idx, stream),
                            paged_kv_pool.k_scales_ptr(layer_idx, stream),
                            &quantized.token_rows_gpu,
                            num_kv_heads,
                            head_dim,
                            paged_kv_pool.kv_dim,
                            total_tokens,
                        )?;
                        kv_quant::quantize_paged_kv_single(
                            &self.ctx,
                            paged_kv_pool.v_work_ptr(stream),
                            paged_kv_pool.v_data_ptr(layer_idx, stream),
                            paged_kv_pool.v_scales_ptr(layer_idx, stream),
                            &quantized.token_rows_gpu,
                            num_kv_heads,
                            head_dim,
                            paged_kv_pool.kv_dim,
                            total_tokens,
                        )?;
                    }
                    KVFormat::BF16 => {}
                    KVFormat::TurboQuant { .. } => unreachable!("TurboQuant does not enter mixed"),
                }

                match kv_format {
                    KVFormat::BF16 => {
                        let max_qlen = mixed
                            .metadata
                            .qo_indptr_h
                            .windows(2)
                            .map(|w| w[1] - w[0])
                            .max()
                            .unwrap_or(0);
                        let total_pages = mixed.metadata.indptr_h.last().copied().unwrap_or(0);
                        ops::flashinfer_tc_run_layer(
                            &self.ctx,
                            &mixed.q_batch,
                            &mixed.metadata.qo_indptr,
                            paged_kv_pool,
                            layer_idx,
                            &mixed.metadata.kv_indptr,
                            &mixed.metadata.kv_indices,
                            &mixed.metadata.kv_last_page_len,
                            &mut mixed.attn_output,
                            &mut mixed.metadata.flashinfer_ws,
                            &ops::FlashInferHeadConfig {
                                num_qo_heads: num_heads,
                                num_kv_heads,
                                page_size,
                                head_dim,
                            },
                            (b + prefill_count) as i32,
                            max_qlen,
                            total_pages,
                        )?;
                    }
                    KVFormat::FP8E4M3 | KVFormat::INT8 => {
                        let quantized = mixed
                            .quantized
                            .as_ref()
                            .expect("quantized mixed buffers allocated");
                        let sm_scale = 1.0 / (head_dim as f32).sqrt();
                        kv_quant::decode_attention_varlen_fp8(
                            &self.ctx,
                            &mixed.q_batch,
                            &mixed.metadata.qo_indptr,
                            paged_kv_pool.k_data_ptr(layer_idx, stream),
                            paged_kv_pool.v_data_ptr(layer_idx, stream),
                            Some(paged_kv_pool.k_scales_ptr(layer_idx, stream)),
                            Some(paged_kv_pool.v_scales_ptr(layer_idx, stream)),
                            &mixed.metadata.kv_indptr,
                            &mixed.metadata.kv_indices,
                            &mixed.metadata.kv_last_page_len,
                            &mut mixed.attn_output,
                            num_heads,
                            num_kv_heads,
                            page_size,
                            b + prefill_count,
                            total_tokens,
                            max_kv_len,
                            matches!(kv_format, KVFormat::INT8),
                            true,
                            sm_scale,
                            &quantized.attn_workspace,
                            quantized.attn_workspace_bytes,
                        )?;
                    }
                    KVFormat::TurboQuant { .. } => unreachable!("TurboQuant does not enter mixed"),
                }
            }

            ops::gemm_into(
                &self.ctx,
                &layer.attention.o_proj,
                &mixed.attn_output,
                &mut mixed.o_buf,
            );
            self.layer_communicator
                .post_attn_all_reduce_hidden_states(&mut mixed.o_buf)?;
            ops::fused_add_rms_norm_batch_into(
                &self.ctx,
                hidden,
                &mixed.o_buf,
                &layer.post_attention_layernorm,
                eps,
                &mut mixed.normed,
            );

            ops::gemm_into(
                &self.ctx,
                &layer.mlp.gate_proj,
                &mixed.normed,
                &mut mixed.gate_out,
            );
            ops::gemm_into(
                &self.ctx,
                &layer.mlp.up_proj,
                &mixed.normed,
                &mut mixed.up_out,
            );
            ops::silu_mul_batch_into(
                &self.ctx,
                &mixed.gate_out,
                &mixed.up_out,
                &mut mixed.act_out,
            )?;

            ops::gemm_into(
                &self.ctx,
                &layer.mlp.down_proj,
                &mixed.act_out,
                &mut mixed.o_buf,
            );
            self.layer_communicator
                .post_mlp_all_reduce_hidden_states(&mut mixed.o_buf)?;

            if let Some(next_input_norm) = next_input_norm {
                ops::fused_add_rms_norm_batch_into(
                    &self.ctx,
                    hidden,
                    &mixed.o_buf,
                    next_input_norm,
                    eps,
                    &mut mixed.normed,
                );
            } else {
                ops::add_batch_into(&self.ctx, hidden, &mixed.o_buf, &mut mixed.hidden_out)?;
                std::mem::swap(hidden, &mut mixed.hidden_out);
            }
        }

        for prefill in batch.prefills {
            states[prefill.slot_idx]
                .base
                .kv_cache
                .advance_seq_len(prefill.tokens.len());
        }

        let hidden = unsafe { &*hidden_ptr };
        ops::rms_norm_batch_into(&self.ctx, hidden, &self.norm, eps, &mut mixed.normed);

        // Gather only the rows we will need vocab logits for: every decode
        // row (rows 0..b are already in place) and the *last* token of each
        // prefill chunk (sources at b + Σ p_k - 1, destinations at b..b+P).
        // Compact in place: dst_i = b + i, src_i = b + Σ_{k≤i} p_k - 1.
        // Since p_k ≥ 1, src_i ≥ dst_i for every i, and src_i strictly
        // exceeds every earlier dst, so sequential forward copies on a
        // single stream cannot clobber any later source.
        //
        // We copy within the same `mixed.normed.data` buffer, so we go via
        // raw `cuMemcpyDtoDAsync_v2` to avoid Rust's aliasing rules on
        // overlapping immutable+mutable slices of the same allocation.
        let kept_rows = b + prefill_count;
        debug_assert!(kept_rows <= mixed.max_logit_rows);
        let hidden_dim = mixed.normed.hidden_dim;
        {
            use cudarc::driver::sys::{cuMemcpyDtoDAsync_v2, cudaError_enum::CUDA_SUCCESS};
            let bf16_size = std::mem::size_of::<u16>();
            let row_bytes = hidden_dim * bf16_size;
            let (base_ptr, _guard) = mixed.normed.data.device_ptr_mut(&self.ctx.stream);
            let mut prefill_token_offset = 0usize;
            for (i, prefill) in batch.prefills.iter().enumerate() {
                let src_row = b + prefill_token_offset + prefill.tokens.len() - 1;
                let dst_row = b + i;
                if src_row != dst_row {
                    let src_ptr = base_ptr + (src_row * row_bytes) as u64;
                    let dst_ptr = base_ptr + (dst_row * row_bytes) as u64;
                    let result = unsafe {
                        cuMemcpyDtoDAsync_v2(
                            dst_ptr,
                            src_ptr,
                            row_bytes,
                            self.ctx.stream.cu_stream(),
                        )
                    };
                    if result != CUDA_SUCCESS {
                        anyhow::bail!("D2D mixed normed gather failed: {result:?}");
                    }
                }
                prefill_token_offset += prefill.tokens.len();
            }
        }

        // Restrict the output projection to the kept rows; mixed.logits is
        // sized for max_batch_size only.
        mixed.normed.seq_len = kept_rows;
        mixed.logits.seq_len = kept_rows;
        ops::gemm_into(
            &self.ctx,
            self.output_projection(),
            &mixed.normed,
            &mut mixed.logits,
        );

        let decode_logits = unsafe { &mut *logits_batch_ptr };
        decode_logits.seq_len = b;
        if b > 0 {
            let src = mixed.logits.data.slice(0..b * decode_logits.hidden_dim);
            let mut dst = decode_logits
                .data
                .slice_mut(0..b * decode_logits.hidden_dim);
            self.ctx
                .stream
                .memcpy_dtod(&src, &mut dst)
                .map_err(|e| anyhow::anyhow!("D2D mixed decode logits: {e}"))?;
        }

        for (i, prefill) in batch.prefills.iter().enumerate() {
            let prefill_state = &mut states[prefill.slot_idx];
            if prefill_state.base.prefill_logits.is_none() {
                prefill_state.base.prefill_logits =
                    Some(DeviceVec::zeros(&self.ctx, self.output_projection().rows)?);
            }
            // After the in-place compaction above, the final-token row for
            // prefill i lives at row `b + i` in mixed.logits.
            ops::extract_vec_into(
                &self.ctx,
                &mixed.logits,
                b + i,
                prefill_state
                    .base
                    .prefill_logits
                    .as_mut()
                    .expect("prefill logits allocated"),
            )?;
        }

        Ok(true)
    }

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

        // LoRA path: keep the paged KV pool, but run eagerly (no graph
        // capture) with split QKV and split gate/up GEMMs so adapters can
        // be applied. `apply_lora_{gemv,gemm}_add` allocates small temp
        // DeviceVecs which CUDA Graph capture rejects.
        if self.lora.is_some() {
            if matches!(paged_kv_pool.format, KVFormat::INT8 | KVFormat::FP8E4M3) {
                upload_decode_quantized_meta(&self.ctx, paged_kv_pool, slot_indices, bufs)?;
            }
            bufs.embedding_out.seq_len = batch_size;
            if bufs.logits_batch.is_none() {
                let vocab_size = self.output_projection().rows;
                bufs.logits_batch = Some(HiddenStates::zeros(
                    &self.ctx,
                    vocab_size,
                    bufs.max_batch_size,
                )?);
            }
            self.decode_batch_lora_body(bufs, paged_kv_pool, batch_size)?;
            if !skip_logit_scatter {
                let logits = bufs.logits_batch.as_ref().unwrap();
                for (b, &si) in slot_indices.iter().enumerate() {
                    ops::extract_vec_into(
                        &self.ctx,
                        logits,
                        b,
                        &mut states[si].decode_bufs.logits,
                    )?;
                    states[si].base.prefill_logits = None;
                }
            }
            return Ok(());
        }

        // NOTE: set_batch_size, upload_token_ids, update_metadata, and
        // plan_attention are now called by the scheduler via DecodeContextOps
        // before this method is invoked.
        if matches!(paged_kv_pool.format, KVFormat::INT8 | KVFormat::FP8E4M3) {
            upload_decode_quantized_meta(&self.ctx, paged_kv_pool, slot_indices, bufs)?;
        }

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
        let force_eager = std::mem::take(&mut bufs.force_eager_once);
        if force_eager || !<Self as crate::model::ModelForward>::supports_cuda_graph_decode(self) {
            self.decode_batch_graph_body(bufs, paged_kv_pool, batch_size)?;
        } else if let Some(ref graph) = bufs.graph_cache[batch_size - 1] {
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

    /// LoRA-aware batched decode body. Runs eagerly (no CUDA graph capture)
    /// because `apply_lora_{gemv,gemm}_add` allocates per-call temps that
    /// stream capture rejects. Forces the split-QKV + split gate/up layout
    /// so per-projection LoRA adds can hit the right tensors.
    fn decode_batch_lora_body(
        &self,
        bufs: &mut BatchDecodeBuffers,
        kv_pool: &PagedKVPool,
        batch_size: usize,
    ) -> Result<()> {
        let eps = self.config.rms_norm_eps;

        ops::embedding_batch(
            &self.ctx,
            &self.embed_tokens,
            &bufs.token_ids_gpu,
            &mut bufs.embedding_out,
        )?;

        let hidden_ptr = &raw mut bufs.embedding_out;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let hidden = unsafe { &mut *hidden_ptr };
            let skip_input_norm = layer_idx > 0;
            let next_input_norm = self
                .layers
                .get(layer_idx + 1)
                .map(|next_layer| &next_layer.input_layernorm);
            self.decode_batch_layer_inner_lora(
                layer_idx,
                layer,
                hidden,
                bufs,
                kv_pool,
                skip_input_norm,
                next_input_norm,
            )?;
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

    /// LoRA-aware per-layer batched decode. Matches `decode_batch_layer_inner`
    /// but always uses the split-QKV path (separate q/k/v gemms +
    /// `decode_prep_paged`) and the split-gate/up MLP path so LoRA adds can
    /// be injected between base projections and downstream ops.
    #[allow(clippy::too_many_arguments)]
    fn decode_batch_layer_inner_lora(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        hidden: &mut HiddenStates,
        bufs: &mut BatchDecodeBuffers,
        kv_pool: &PagedKVPool,
        skip_input_norm: bool,
        next_input_norm: Option<&DeviceVec>,
    ) -> Result<()> {
        let eps = self.config.rms_norm_eps;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let page_size = kv_pool.page_size;

        if !skip_input_norm {
            ops::rms_norm_batch_into(
                &self.ctx,
                hidden,
                &layer.input_layernorm,
                eps,
                &mut bufs.normed,
            );
        }

        // Split QKV projections so LoRA adds can compose.
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
        if let Some(ll) = self.layer_lora(layer_idx) {
            if let Some(ad) = ll.q_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.normed, &mut bufs.q_batch)?;
            }
            if let Some(ad) = ll.k_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.normed, &mut bufs.k_batch)?;
            }
            if let Some(ad) = ll.v_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.normed, &mut bufs.v_batch)?;
            }
        }

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

        // Attention: reuse the non-LoRA attention dispatch by format.
        {
            let batch_size = bufs.q_batch.seq_len;
            let stream = &self.ctx.stream;
            match kv_pool.format {
                KVFormat::FP8E4M3 => {
                    kv_quant::quantize_paged_kv_fp8(
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
                    kv_quant::quantize_paged_kv_fp8(
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
                    let sm_scale = 1.0 / (head_dim as f32).sqrt();
                    kv_quant::decode_attention_fp8(
                        &self.ctx,
                        &bufs.q_batch,
                        kv_pool.k_data_ptr(layer_idx, stream),
                        kv_pool.v_data_ptr(layer_idx, stream),
                        kv_pool.k_scales_ptr(layer_idx, stream),
                        kv_pool.v_scales_ptr(layer_idx, stream),
                        &bufs.metadata.kv_indices,
                        &bufs.quantized_kv_meta,
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
                    let sm_scale = 1.0 / (head_dim as f32).sqrt();
                    kv_quant::decode_attention_int8(
                        &self.ctx,
                        &bufs.q_batch,
                        kv_pool.k_data_ptr(layer_idx, stream),
                        kv_pool.v_data_ptr(layer_idx, stream),
                        kv_pool.k_scales_ptr(layer_idx, stream),
                        kv_pool.v_scales_ptr(layer_idx, stream),
                        &bufs.metadata.kv_indices,
                        &bufs.quantized_kv_meta,
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
                    let max_qlen = bufs
                        .metadata
                        .qo_indptr_h
                        .windows(2)
                        .map(|w| w[1] - w[0])
                        .max()
                        .unwrap_or(0);
                    let total_pages = bufs.metadata.indptr_h.last().copied().unwrap_or(0);
                    ops::flashinfer_tc_run_layer(
                        &self.ctx,
                        &bufs.q_batch,
                        &bufs.metadata.qo_indptr,
                        kv_pool,
                        layer_idx,
                        &bufs.metadata.kv_indptr,
                        &bufs.metadata.kv_indices,
                        &bufs.metadata.kv_last_page_len,
                        &mut bufs.attn_output,
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
                    anyhow::bail!(
                        "LoRA + TurboQuant KV cache not supported — refuse earlier at load time"
                    );
                }
            }
        }

        // O projection + LoRA.
        ops::gemm_into(
            &self.ctx,
            &layer.attention.o_proj,
            &bufs.attn_output,
            &mut bufs.o_buf,
        );
        if let Some(ll) = self.layer_lora(layer_idx) {
            if let Some(ad) = ll.o_proj.as_ref() {
                ops::apply_lora_gemm_add(
                    &self.ctx,
                    &ad.a,
                    &ad.b,
                    &bufs.attn_output,
                    &mut bufs.o_buf,
                )?;
            }
        }
        self.layer_communicator
            .post_attn_all_reduce_hidden_states(&mut bufs.o_buf)?;

        ops::fused_add_rms_norm_batch_into(
            &self.ctx,
            hidden,
            &bufs.o_buf,
            &layer.post_attention_layernorm,
            eps,
            &mut bufs.normed,
        );

        // Split gate + up MLP + LoRA + silu_mul + down + LoRA.
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
        if let Some(ll) = self.layer_lora(layer_idx) {
            if let Some(ad) = ll.gate_proj.as_ref() {
                ops::apply_lora_gemm_add(
                    &self.ctx,
                    &ad.a,
                    &ad.b,
                    &bufs.normed,
                    &mut bufs.gate_out,
                )?;
            }
            if let Some(ad) = ll.up_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.normed, &mut bufs.up_out)?;
            }
        }
        ops::silu_mul_batch_into(&self.ctx, &bufs.gate_out, &bufs.up_out, &mut bufs.act_out)?;
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.down_proj,
            &bufs.act_out,
            &mut bufs.o_buf,
        );
        if let Some(ll) = self.layer_lora(layer_idx) {
            if let Some(ad) = ll.down_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.act_out, &mut bufs.o_buf)?;
            }
        }
        self.layer_communicator
            .post_mlp_all_reduce_hidden_states(&mut bufs.o_buf)?;

        if let Some(next_input_norm) = next_input_norm {
            ops::fused_add_rms_norm_batch_into(
                &self.ctx,
                hidden,
                &bufs.o_buf,
                next_input_norm,
                eps,
                &mut bufs.normed,
            );
        } else {
            ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
            std::mem::swap(hidden, &mut bufs.hidden_out);
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
        let hidden_ptr = &raw mut bufs.embedding_out;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // SAFETY: hidden_ptr points to bufs.embedding_out. The layer
            // function only accesses other fields of bufs (normed, q_batch, etc.)
            // and swaps hidden_ptr's target with bufs.hidden_out. No aliasing.
            let hidden = unsafe { &mut *hidden_ptr };
            let skip_input_norm = layer_idx > 0;
            let next_input_norm = self
                .layers
                .get(layer_idx + 1)
                .map(|next_layer| &next_layer.input_layernorm);
            self.decode_batch_layer_inner(
                layer_idx,
                layer,
                hidden,
                bufs,
                kv_pool,
                skip_input_norm,
                next_input_norm,
            )?;
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
        skip_input_norm: bool,
        next_input_norm: Option<&DeviceVec>,
    ) -> Result<()> {
        let eps = self.config.rms_norm_eps;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let page_size = kv_pool.page_size;

        // 1. Batched RMSNorm → bufs.normed [B, hidden_dim]
        if !skip_input_norm {
            ops::rms_norm_batch_into(
                &self.ctx,
                hidden,
                &layer.input_layernorm,
                eps,
                &mut bufs.normed,
            );
        }

        // 2. QKV projection
        // 3. Decode prep: QKV projection + QK-norm + RoPE + paged KV write
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

        // 3 separate Q/K/V GEMMs → decode_prep_paged (qk-norm + RoPE + paged write).
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
                        kv_pool.k_scales_ptr(layer_idx, stream),
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
                        kv_pool.v_scales_ptr(layer_idx, stream),
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
                        kv_pool.k_scales_ptr(layer_idx, stream),
                        kv_pool.v_scales_ptr(layer_idx, stream),
                        &bufs.metadata.kv_indices,
                        &bufs.quantized_kv_meta,
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
                        &bufs.quantized_kv_meta,
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
                    let max_qlen = bufs
                        .metadata
                        .qo_indptr_h
                        .windows(2)
                        .map(|w| w[1] - w[0])
                        .max()
                        .unwrap_or(0);
                    let total_pages = bufs.metadata.indptr_h.last().copied().unwrap_or(0);
                    ops::flashinfer_tc_run_layer(
                        &self.ctx,
                        &bufs.q_batch,
                        &bufs.metadata.qo_indptr,
                        kv_pool,
                        layer_idx,
                        &bufs.metadata.kv_indptr,
                        &bufs.metadata.kv_indices,
                        &bufs.metadata.kv_last_page_len,
                        &mut bufs.attn_output,
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
                    // Avoids O(seq_len × D log D) full dequant per layer.
                    let tq_k = kv_pool.tq_k_state.as_ref().unwrap();
                    let tq_v = kv_pool.tq_v_state.as_ref().unwrap();
                    let sm_scale = 1.0 / (head_dim as f32).sqrt();

                    // Step 1: Rotate Q → Q_rot (sign flip + FWHT)
                    let q_ptr = {
                        let (p, _g) = bufs.q_batch.data.device_ptr(stream);
                        p
                    };
                    let q_rot_ptr = {
                        let (p, _g) = bufs.q_rot.data.device_ptr_mut(stream);
                        p
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
                        p
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
        self.layer_communicator
            .post_attn_all_reduce_hidden_states(&mut bufs.o_buf)?;

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
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.down_proj,
            &bufs.act_out,
            &mut bufs.o_buf,
        );
        self.layer_communicator
            .post_mlp_all_reduce_hidden_states(&mut bufs.o_buf)?;

        // 9. Batched residual add, optionally fused with the next layer's input RMSNorm.
        if let Some(next_input_norm) = next_input_norm {
            ops::fused_add_rms_norm_batch_into(
                &self.ctx,
                hidden,
                &bufs.o_buf,
                next_input_norm,
                eps,
                &mut bufs.normed,
            );
        } else {
            ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
            std::mem::swap(hidden, &mut bufs.hidden_out);
        }

        Ok(())
    }
}
