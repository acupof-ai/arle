//! DeepSeek V4 batched-decode scratch buffers.
//!
//! Tracks scheduler batch capacity for DSv4 decode. The model batches the
//! stable projection/head pieces today; the cache/metadata attention core still
//! needs a batch FlashMLA path before full decode CUDA graph replay is viable.

use anyhow::{Result, ensure};

use super::state::DeepseekMhcRuntimeScratch;
use crate::model::DecodeContextOps;
use crate::model::kv_cache::KVFormat;
use cuda_kernels::prelude::{DeviceContext, DeviceVec, HiddenStates, PagedKVPool};
use cuda_kernels::tensor::CudaAllocTraceExt;
use cudarc::driver::safe::CudaGraph;
use cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
use cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
use cudarc::driver::{CudaSlice, DevicePtrMut};
use log::info;

/// MODEL1 FlashMLA sparse-FP8 KV block byte size — `page_block_size (64) ×
/// bytes_per_token (584)` = 37376 B per block per layer. Kept here (instead of
/// re-importing the private `weights.rs` constant) so the shared-pool sizing is
/// readable at its allocation site; both must stay in lock-step with the
/// `DSV4_FLASHMLA_MODEL1_*` constants in `weights.rs` (compile-time `const`s, no
/// drift across a single build).
const DSV4_FLASHMLA_MODEL1_BLOCK_BYTES: usize = 64 * 584;
/// `DecodingSchedMetaSize == 32 B == 8 × int32` in FlashMLA's params.h.
#[allow(dead_code)]
const DSV4_FLASHMLA_DECODING_SCHED_META_INTS: usize = 8;

/// Pre-allocated decode context.
///
/// Optionally owns the **shared persistent FP8 KV pool** that backs FlashMLA
/// sparse decode (Phase D-4) — but ONLY when `ARLE_DSV4_SHARED_KV_POOL=1`. When
/// the env knob is OFF (the default), `fp8_kv_pool` stays `None` and the model
/// uses the per-(slot, layer) lazy allocation (`ensure_dsv4_flashmla_fp8_kv_pool`
/// in `weights.rs`), byte-identical to the path shipped on `main`.
///
/// When ON, the pool is allocated once at `create_decode_context` sized for
/// `num_slots × layers × slot_blocks × block_bytes` so every concurrent
/// sequence gets a fixed, budgeted sub-range — replacing the per-state lazy
/// allocation that OOMed at c≥8 (each of `num_slots × layers` pools was a
/// separate unbudgeted `alloc_zeros`, and the compressed sub-pool was sized to
/// the full `max_position_embeddings` capacity).
///
/// Slot `s`, layer `l` owns the contiguous byte range
/// `[(s*layers + l) * slot_layer_bytes, (s*layers + l + 1) * slot_layer_bytes)`
/// where `slot_layer_bytes = slot_blocks * block_bytes`. The per-row pack/decode
/// logic is byte-identical to the per-state pool — only the base device pointer
/// differs (the slot's sub-range start instead of an owned buffer's byte 0).
///
/// Public so the `ModelForward::DecodeContext` associated type (a `pub` surface
/// on the trait) does not leak a private name.
pub struct DeepseekBatchDecodeBuffers {
    max_batch_size: usize,
    /// Shared owning FP8 KV pool, or `None` when `ARLE_DSV4_SHARED_KV_POOL` is
    /// off (the default — per-state lazy pool in use) or when the FlashMLA
    /// decode env knob is off / no layers are loaded.
    fp8_kv_pool: Option<CudaSlice<u8>>,
    /// Number of `num_slots` the pool was sized for.
    fp8_kv_slots: usize,
    /// Number of `layers` the pool was sized for.
    fp8_kv_layers: usize,
    /// `slot_blocks = sw_blocks + comp_blocks` per (slot, layer) sub-range.
    fp8_kv_slot_blocks: usize,
    /// Served `max_seq_len` the pool was sized for; the per-step binding reads
    /// it back so the layout matches the allocation.
    fp8_kv_max_seq_len: usize,
    #[allow(dead_code)]
    flashmla_decode_arena: Option<DeepseekFlashMlaDecodeBatchArena>,
    batched_scratch: Option<DeepseekBatchedDecodeScratch>,
    /// Piecewise graph cache for stable token-id input staging
    /// (embedding + initial HC stream), indexed by `batch_size - 1`.
    input_graph_cache: Vec<Option<CudaGraph>>,
    /// Piecewise graph cache for stable head/logits staging, indexed by
    /// `batch_size - 1`. Per-slot logits scatter stays outside the graph so
    /// captured destination pointers are scheduler-context stable.
    head_graph_cache: Vec<Option<CudaGraph>>,
    /// One-shot eager override used by correctness-sensitive verifier paths.
    force_eager_once: bool,
}

// SAFETY: `DeepseekBatchDecodeBuffers` owns CUDA graph handles, which must stay
// on the creating CUDA context/thread. The scheduler owns one decode context and
// uses it exclusively on the single inference thread.
unsafe impl Send for DeepseekBatchDecodeBuffers {}

/// Stable DSv4 batched-decode buffers.
///
/// The current DSv4 graph blocker is dynamic attention metadata, but the
/// batched decode body should still be allocation-free before graph capture is
/// considered. These buffers replace per-step embedding / stream / row / head /
/// logits allocations with stable pointers owned by the scheduler decode
/// context.
pub(super) struct DeepseekBatchedDecodeScratch {
    capacity_tokens: usize,
    hidden_size: usize,
    stream_hidden_dim: usize,
    attn_c_q_dim: usize,
    attn_local_width: usize,
    attn_head_dim: usize,
    attn_output_latent_dim: usize,
    head_mix_dim: usize,
    vocab_size: usize,
    token_ids_scratch: Vec<i32>,
    start_pos_scratch: Vec<i32>,
    token_ids_uploaded: usize,
    pub(super) token_ids_gpu: CudaSlice<i32>,
    pub(super) start_pos_gpu: CudaSlice<i32>,
    pub(super) embeddings: HiddenStates,
    pub(super) stream: HiddenStates,
    pub(super) attn_mhc: Option<DeepseekMhcRuntimeScratch>,
    pub(super) attn_pre: HiddenStates,
    pub(super) attn_normed: HiddenStates,
    pub(super) attn_normed_row: HiddenStates,
    pub(super) attn_c_q: HiddenStates,
    pub(super) attn_c_q_normed: HiddenStates,
    pub(super) attn_c_q_normed_row: HiddenStates,
    pub(super) attn_q_raw: HiddenStates,
    pub(super) attn_q_raw_row: HiddenStates,
    pub(super) attn_kv_raw: HiddenStates,
    pub(super) attn_kv_normed: HiddenStates,
    pub(super) attn_kv_normed_row: HiddenStates,
    pub(super) attn_local: HiddenStates,
    pub(super) attn_local_row: HiddenStates,
    pub(super) attn_output_latent: HiddenStates,
    pub(super) attn_out: HiddenStates,
    pub(super) attn_out_row: HiddenStates,
    pub(super) attn_stream: HiddenStates,
    pub(super) head_stream_row: HiddenStates,
    pub(super) head_mixes: HiddenStates,
    pub(super) head_hidden: DeviceVec,
    pub(super) head_normed: DeviceVec,
    pub(super) logits: DeviceVec,
    pub(super) logits_batch: HiddenStates,
}

/// Batch-shaped FlashMLA sparse-decode scratch.
///
/// This is deliberately owned by the scheduler decode context, not by a
/// per-slot attention cache: FlashMLA's decode ABI is batch-first (`b=N`), uses
/// one shared FP8 KV base, and sizes split/combine scratch as
/// `[b + num_sm_parts, ...]`.
#[allow(dead_code)]
pub(super) struct DeepseekFlashMlaDecodeBatchArena {
    capacity_tokens: usize,
    num_sm_parts_max: usize,
    topk_unified_max: usize,
    h_q: usize,
    d_v: usize,
    slot_layer_block_offsets_scratch: Vec<i32>,
    pub(super) slot_layer_block_offsets_gpu: CudaSlice<i32>,
    pub(super) indices: CudaSlice<i32>,
    pub(super) topk_length: CudaSlice<i32>,
    pub(super) num_splits: CudaSlice<i32>,
    pub(super) sched_meta: CudaSlice<i32>,
    pub(super) lse_accum: CudaSlice<f32>,
    pub(super) o_accum: CudaSlice<f32>,
    pub(super) lse_out: CudaSlice<f32>,
}

#[allow(dead_code)]
impl DeepseekFlashMlaDecodeBatchArena {
    fn new(
        ctx: &DeviceContext,
        capacity_tokens: usize,
        num_sm_parts_max: usize,
        topk_unified_max: usize,
        h_q: usize,
        d_v: usize,
    ) -> Result<Self> {
        ensure!(
            capacity_tokens > 0
                && num_sm_parts_max > 0
                && topk_unified_max > 0
                && h_q > 0
                && d_v > 0,
            "DSv4 batch FlashMLA arena requires positive sizing (capacity={}, num_sm_parts={}, topk={}, h_q={}, d_v={})",
            capacity_tokens,
            num_sm_parts_max,
            topk_unified_max,
            h_q,
            d_v
        );
        let split_axis = capacity_tokens + num_sm_parts_max;
        Ok(Self {
            capacity_tokens,
            num_sm_parts_max,
            topk_unified_max,
            h_q,
            d_v,
            slot_layer_block_offsets_scratch: Vec::with_capacity(capacity_tokens),
            slot_layer_block_offsets_gpu: ctx
                .stream
                .alloc_zeros_traced::<i32>(capacity_tokens)
                .map_err(|err| anyhow::anyhow!("DSv4 batch FlashMLA slot offsets alloc: {err}"))?,
            indices: ctx
                .stream
                .alloc_zeros_traced::<i32>(capacity_tokens * topk_unified_max)
                .map_err(|err| anyhow::anyhow!("DSv4 batch FlashMLA indices alloc: {err}"))?,
            topk_length: ctx
                .stream
                .alloc_zeros_traced::<i32>(capacity_tokens)
                .map_err(|err| anyhow::anyhow!("DSv4 batch FlashMLA topk_length alloc: {err}"))?,
            num_splits: ctx
                .stream
                .alloc_zeros_traced::<i32>(capacity_tokens + 1)
                .map_err(|err| anyhow::anyhow!("DSv4 batch FlashMLA num_splits alloc: {err}"))?,
            sched_meta: ctx
                .stream
                .alloc_zeros_traced::<i32>(
                    num_sm_parts_max * DSV4_FLASHMLA_DECODING_SCHED_META_INTS,
                )
                .map_err(|err| anyhow::anyhow!("DSv4 batch FlashMLA sched_meta alloc: {err}"))?,
            lse_accum: ctx
                .stream
                .alloc_zeros_traced::<f32>(split_axis * h_q)
                .map_err(|err| anyhow::anyhow!("DSv4 batch FlashMLA lse_accum alloc: {err}"))?,
            o_accum: ctx
                .stream
                .alloc_zeros_traced::<f32>(split_axis * h_q * d_v)
                .map_err(|err| anyhow::anyhow!("DSv4 batch FlashMLA o_accum alloc: {err}"))?,
            lse_out: ctx
                .stream
                .alloc_zeros_traced::<f32>(capacity_tokens * h_q)
                .map_err(|err| anyhow::anyhow!("DSv4 batch FlashMLA lse_out alloc: {err}"))?,
        })
    }

    fn has_capacity(
        &self,
        capacity_tokens: usize,
        num_sm_parts_max: usize,
        topk_unified_max: usize,
        h_q: usize,
        d_v: usize,
    ) -> bool {
        self.capacity_tokens >= capacity_tokens
            && self.num_sm_parts_max >= num_sm_parts_max
            && self.topk_unified_max >= topk_unified_max
            && self.h_q == h_q
            && self.d_v == d_v
    }

    pub(super) fn set_batch_size(&mut self, batch_size: usize) -> Result<()> {
        ensure!(
            batch_size <= self.capacity_tokens,
            "DSv4 batch FlashMLA arena batch {batch_size} exceeds capacity {}",
            self.capacity_tokens
        );
        Ok(())
    }

    fn validate_shape(
        &mut self,
        batch_size: usize,
        num_sm_parts_max: usize,
        topk_unified_max: usize,
        h_q: usize,
        d_v: usize,
    ) -> Result<()> {
        self.set_batch_size(batch_size)?;
        let split_axis = batch_size + num_sm_parts_max;
        ensure!(
            self.slot_layer_block_offsets_gpu.len() >= batch_size
                && self.indices.len() >= batch_size * topk_unified_max
                && self.topk_length.len() >= batch_size
                && self.num_splits.len() > batch_size
                && self.sched_meta.len()
                    >= num_sm_parts_max * DSV4_FLASHMLA_DECODING_SCHED_META_INTS
                && self.lse_accum.len() >= split_axis * h_q
                && self.o_accum.len() >= split_axis * h_q * d_v
                && self.lse_out.len() >= batch_size * h_q,
            "DSv4 batch FlashMLA arena allocation is smaller than requested shape"
        );
        Ok(())
    }

    fn upload_slot_layer_block_offsets(
        &mut self,
        ctx: &DeviceContext,
        slot_indices: &[usize],
        layer_idx: usize,
        fp8_kv_slots: usize,
        fp8_kv_layers: usize,
        fp8_kv_slot_blocks: usize,
    ) -> Result<u64> {
        self.set_batch_size(slot_indices.len())?;
        self.slot_layer_block_offsets_scratch.clear();
        for &slot_idx in slot_indices {
            let block_offset = fp8_kv_slot_layer_block_offset_for_layout(
                slot_idx,
                layer_idx,
                fp8_kv_slots,
                fp8_kv_layers,
                fp8_kv_slot_blocks,
            )?;
            ensure!(
                block_offset <= i32::MAX as usize,
                "DSv4 batch FlashMLA slot/layer block offset {block_offset} exceeds i32"
            );
            self.slot_layer_block_offsets_scratch
                .push(block_offset as i32);
        }
        let mut dst = self
            .slot_layer_block_offsets_gpu
            .slice_mut(0..slot_indices.len());
        ctx.stream
            .memcpy_htod(&self.slot_layer_block_offsets_scratch, &mut dst)
            .map_err(|err| anyhow::anyhow!("H2D DSv4 batch FlashMLA slot offsets: {err}"))?;
        let (ptr, guard) = self
            .slot_layer_block_offsets_gpu
            .device_ptr_mut(&ctx.stream);
        drop(guard);
        Ok(ptr)
    }
}

impl DeepseekBatchedDecodeScratch {
    fn new(
        ctx: &DeviceContext,
        capacity_tokens: usize,
        hidden_size: usize,
        stream_hidden_dim: usize,
        attn_c_q_dim: usize,
        attn_local_width: usize,
        attn_head_dim: usize,
        attn_output_latent_dim: usize,
        head_mix_dim: usize,
        vocab_size: usize,
    ) -> Result<Self> {
        let capacity_tokens = capacity_tokens.max(1);
        Ok(Self {
            capacity_tokens,
            hidden_size,
            stream_hidden_dim,
            attn_c_q_dim,
            attn_local_width,
            attn_head_dim,
            attn_output_latent_dim,
            head_mix_dim,
            vocab_size,
            token_ids_scratch: Vec::with_capacity(capacity_tokens),
            start_pos_scratch: Vec::with_capacity(capacity_tokens),
            token_ids_uploaded: 0,
            token_ids_gpu: ctx
                .stream
                .alloc_zeros(capacity_tokens)
                .map_err(|err| anyhow::anyhow!("Alloc DSv4 token_ids_gpu failed: {err}"))?,
            start_pos_gpu: ctx
                .stream
                .alloc_zeros(capacity_tokens)
                .map_err(|err| anyhow::anyhow!("Alloc DSv4 start_pos_gpu failed: {err}"))?,
            embeddings: HiddenStates::zeros(ctx, hidden_size, capacity_tokens)?,
            stream: HiddenStates::zeros(ctx, stream_hidden_dim, capacity_tokens)?,
            attn_mhc: None,
            attn_pre: HiddenStates::zeros(ctx, hidden_size, capacity_tokens)?,
            attn_normed: HiddenStates::zeros(ctx, hidden_size, capacity_tokens)?,
            attn_normed_row: HiddenStates::zeros(ctx, hidden_size, 1)?,
            attn_c_q: HiddenStates::zeros(ctx, attn_c_q_dim, capacity_tokens)?,
            attn_c_q_normed: HiddenStates::zeros(ctx, attn_c_q_dim, capacity_tokens)?,
            attn_c_q_normed_row: HiddenStates::zeros(ctx, attn_c_q_dim, 1)?,
            attn_q_raw: HiddenStates::zeros(ctx, attn_local_width, capacity_tokens)?,
            attn_q_raw_row: HiddenStates::zeros(ctx, attn_local_width, 1)?,
            attn_kv_raw: HiddenStates::zeros(ctx, attn_head_dim, capacity_tokens)?,
            attn_kv_normed: HiddenStates::zeros(ctx, attn_head_dim, capacity_tokens)?,
            attn_kv_normed_row: HiddenStates::zeros(ctx, attn_head_dim, 1)?,
            attn_local: HiddenStates::zeros(ctx, attn_local_width, capacity_tokens)?,
            attn_local_row: HiddenStates::zeros(ctx, attn_local_width, 1)?,
            attn_output_latent: HiddenStates::zeros(ctx, attn_output_latent_dim, capacity_tokens)?,
            attn_out: HiddenStates::zeros(ctx, hidden_size, capacity_tokens)?,
            attn_out_row: HiddenStates::zeros(ctx, hidden_size, 1)?,
            attn_stream: HiddenStates::zeros(ctx, stream_hidden_dim, capacity_tokens)?,
            head_stream_row: HiddenStates::zeros(ctx, stream_hidden_dim, 1)?,
            head_mixes: HiddenStates::zeros(ctx, head_mix_dim, 1)?,
            head_hidden: DeviceVec::zeros(ctx, hidden_size)?.with_label("dsv4_head_hidden"),
            head_normed: DeviceVec::zeros(ctx, hidden_size)?.with_label("dsv4_head_normed"),
            logits: DeviceVec::zeros(ctx, vocab_size)?.with_label("dsv4_batched_logits_scratch"),
            logits_batch: HiddenStates::zeros(ctx, vocab_size, capacity_tokens)?,
        })
    }

    pub(super) fn set_batch_size(&mut self, batch_size: usize) -> Result<()> {
        ensure!(
            batch_size <= self.capacity_tokens,
            "DSv4 batched scratch batch {batch_size} exceeds capacity {}",
            self.capacity_tokens
        );
        self.embeddings.seq_len = batch_size;
        self.stream.seq_len = batch_size;
        self.attn_pre.seq_len = batch_size;
        self.attn_normed.seq_len = batch_size;
        self.attn_normed_row.seq_len = 1;
        self.attn_c_q.seq_len = batch_size;
        self.attn_c_q_normed.seq_len = batch_size;
        self.attn_c_q_normed_row.seq_len = 1;
        self.attn_q_raw.seq_len = batch_size;
        self.attn_q_raw_row.seq_len = 1;
        self.attn_kv_raw.seq_len = batch_size;
        self.attn_kv_normed.seq_len = batch_size;
        self.attn_kv_normed_row.seq_len = 1;
        self.attn_local.seq_len = batch_size;
        self.attn_local_row.seq_len = 1;
        self.attn_output_latent.seq_len = batch_size;
        self.attn_out.seq_len = batch_size;
        self.attn_out_row.seq_len = 1;
        self.attn_stream.seq_len = batch_size;
        self.head_stream_row.seq_len = 1;
        self.head_mixes.seq_len = 1;
        self.logits_batch.seq_len = batch_size;
        Ok(())
    }

    pub(super) fn upload_token_ids(&mut self, ctx: &DeviceContext, tokens: &[u32]) -> Result<()> {
        self.set_batch_size(tokens.len())?;
        self.token_ids_scratch.clear();
        for &token in tokens {
            ensure!(
                token <= i32::MAX as u32,
                "DSv4 token id {token} exceeds i32 device metadata range"
            );
            self.token_ids_scratch.push(token as i32);
        }
        let mut dst = self.token_ids_gpu.slice_mut(0..tokens.len());
        ctx.stream
            .memcpy_htod(&self.token_ids_scratch, &mut dst)
            .map_err(|err| anyhow::anyhow!("H2D DSv4 token_ids: {err}"))?;
        self.token_ids_uploaded = tokens.len();
        Ok(())
    }

    pub(super) fn ensure_token_ids_uploaded(
        &mut self,
        ctx: &DeviceContext,
        tokens: &[u32],
    ) -> Result<()> {
        let already_uploaded = self.token_ids_uploaded == tokens.len()
            && self.token_ids_scratch.len() == tokens.len()
            && self
                .token_ids_scratch
                .iter()
                .zip(tokens.iter())
                .all(|(&cached, &token)| cached >= 0 && cached as u32 == token);
        if already_uploaded {
            self.set_batch_size(tokens.len())?;
            return Ok(());
        }
        self.upload_token_ids(ctx, tokens)
    }

    pub(super) fn upload_start_positions(
        &mut self,
        ctx: &DeviceContext,
        start_pos: &[usize],
    ) -> Result<()> {
        self.set_batch_size(start_pos.len())?;
        self.start_pos_scratch.clear();
        for &pos in start_pos {
            ensure!(
                pos <= i32::MAX as usize,
                "DSv4 start_pos {pos} exceeds i32 device metadata range"
            );
            self.start_pos_scratch.push(pos as i32);
        }
        let mut dst = self.start_pos_gpu.slice_mut(0..start_pos.len());
        ctx.stream
            .memcpy_htod(&self.start_pos_scratch, &mut dst)
            .map_err(|err| anyhow::anyhow!("H2D DSv4 start_pos: {err}"))?;
        Ok(())
    }
}

impl DeepseekBatchDecodeBuffers {
    /// Allocate the decode context. The current serial fallback still needs to
    /// validate scheduler batch capacity.
    pub(super) fn new(
        _ctx: &DeviceContext,
        max_batch_size: usize,
        _max_total_pages: usize,
    ) -> Result<Self> {
        ensure!(
            max_batch_size > 0,
            "DeepSeek V4 decode context needs batch capacity"
        );
        Ok(Self {
            max_batch_size,
            fp8_kv_pool: None,
            fp8_kv_slots: 0,
            fp8_kv_layers: 0,
            fp8_kv_slot_blocks: 0,
            fp8_kv_max_seq_len: 0,
            flashmla_decode_arena: None,
            batched_scratch: None,
            input_graph_cache: (0..max_batch_size).map(|_| None).collect(),
            head_graph_cache: (0..max_batch_size).map(|_| None).collect(),
            force_eager_once: false,
        })
    }

    pub(super) fn ensure_batched_scratch(
        &mut self,
        ctx: &DeviceContext,
        hidden_size: usize,
        stream_hidden_dim: usize,
        attn_c_q_dim: usize,
        attn_local_width: usize,
        attn_head_dim: usize,
        attn_output_latent_dim: usize,
        head_mix_dim: usize,
        vocab_size: usize,
        batch_size: usize,
    ) -> Result<&mut DeepseekBatchedDecodeScratch> {
        ensure!(
            batch_size <= self.max_batch_size,
            "DeepSeek V4 batched scratch batch {batch_size} exceeds context capacity {}",
            self.max_batch_size
        );
        let need_alloc = self
            .batched_scratch
            .as_ref()
            .map(|scratch| {
                scratch.capacity_tokens < self.max_batch_size
                    || scratch.hidden_size != hidden_size
                    || scratch.stream_hidden_dim != stream_hidden_dim
                    || scratch.attn_c_q_dim != attn_c_q_dim
                    || scratch.attn_local_width != attn_local_width
                    || scratch.attn_head_dim != attn_head_dim
                    || scratch.attn_output_latent_dim != attn_output_latent_dim
                    || scratch.head_mix_dim != head_mix_dim
                    || scratch.vocab_size != vocab_size
            })
            .unwrap_or(true);
        if need_alloc {
            self.batched_scratch = Some(DeepseekBatchedDecodeScratch::new(
                ctx,
                self.max_batch_size,
                hidden_size,
                stream_hidden_dim,
                attn_c_q_dim,
                attn_local_width,
                attn_head_dim,
                attn_output_latent_dim,
                head_mix_dim,
                vocab_size,
            )?);
        }
        let scratch = self
            .batched_scratch
            .as_mut()
            .expect("DSv4 batched scratch allocated");
        scratch.set_batch_size(batch_size)?;
        Ok(scratch)
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub(super) fn ensure_flashmla_decode_batch_arena(
        &mut self,
        ctx: &DeviceContext,
        batch_size: usize,
        num_sm_parts_max: usize,
        topk_unified_max: usize,
        h_q: usize,
        d_v: usize,
    ) -> Result<&mut DeepseekFlashMlaDecodeBatchArena> {
        ensure!(
            batch_size <= self.max_batch_size,
            "DSv4 batch FlashMLA arena batch {batch_size} exceeds context capacity {}",
            self.max_batch_size
        );
        let capacity_tokens = self.max_batch_size;
        let need_alloc = self.flashmla_decode_arena.as_ref().is_none_or(|arena| {
            !arena.has_capacity(
                capacity_tokens,
                num_sm_parts_max,
                topk_unified_max,
                h_q,
                d_v,
            )
        });
        if need_alloc {
            self.flashmla_decode_arena = Some(DeepseekFlashMlaDecodeBatchArena::new(
                ctx,
                capacity_tokens,
                num_sm_parts_max,
                topk_unified_max,
                h_q,
                d_v,
            )?);
        }
        let arena = self
            .flashmla_decode_arena
            .as_mut()
            .expect("DSv4 batch FlashMLA arena allocated");
        arena.validate_shape(batch_size, num_sm_parts_max, topk_unified_max, h_q, d_v)?;
        Ok(arena)
    }

    #[allow(dead_code)]
    pub(super) fn stage_flashmla_slot_layer_block_offsets(
        &mut self,
        ctx: &DeviceContext,
        slot_indices: &[usize],
        layer_idx: usize,
    ) -> Result<u64> {
        ensure!(
            slot_indices.len() <= self.max_batch_size,
            "DSv4 batch FlashMLA offset batch {} exceeds context capacity {}",
            slot_indices.len(),
            self.max_batch_size
        );
        ensure!(
            self.fp8_kv_pool.is_some(),
            "DSv4 batch FlashMLA slot offsets require shared FP8 KV pool"
        );
        let fp8_kv_slots = self.fp8_kv_slots;
        let fp8_kv_layers = self.fp8_kv_layers;
        let fp8_kv_slot_blocks = self.fp8_kv_slot_blocks;
        let arena = self
            .flashmla_decode_arena
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("DSv4 batch FlashMLA arena not allocated"))?;
        arena.upload_slot_layer_block_offsets(
            ctx,
            slot_indices,
            layer_idx,
            fp8_kv_slots,
            fp8_kv_layers,
            fp8_kv_slot_blocks,
        )
    }

    pub(super) fn take_force_eager_once(&mut self) -> bool {
        std::mem::take(&mut self.force_eager_once)
    }

    pub(super) fn run_input_piece_graph<F>(
        &mut self,
        ctx: &DeviceContext,
        batch_size: usize,
        force_eager: bool,
        body: F,
    ) -> Result<()>
    where
        F: FnMut(&mut DeepseekBatchedDecodeScratch) -> Result<()>,
    {
        self.run_piece_graph(
            ctx,
            batch_size,
            force_eager,
            DeepseekDecodeGraphPiece::Input,
            body,
        )
    }

    pub(super) fn run_head_piece_graph<F>(
        &mut self,
        ctx: &DeviceContext,
        batch_size: usize,
        force_eager: bool,
        body: F,
    ) -> Result<()>
    where
        F: FnMut(&mut DeepseekBatchedDecodeScratch) -> Result<()>,
    {
        self.run_piece_graph(
            ctx,
            batch_size,
            force_eager,
            DeepseekDecodeGraphPiece::Head,
            body,
        )
    }

    fn run_piece_graph<F>(
        &mut self,
        ctx: &DeviceContext,
        batch_size: usize,
        force_eager: bool,
        piece: DeepseekDecodeGraphPiece,
        mut body: F,
    ) -> Result<()>
    where
        F: FnMut(&mut DeepseekBatchedDecodeScratch) -> Result<()>,
    {
        ensure!(
            batch_size >= 1 && batch_size <= self.max_batch_size,
            "DeepSeek V4 graph piece batch {batch_size} exceeds context capacity {}",
            self.max_batch_size
        );
        let idx = batch_size - 1;
        let cache = match piece {
            DeepseekDecodeGraphPiece::Input => &mut self.input_graph_cache,
            DeepseekDecodeGraphPiece::Head => &mut self.head_graph_cache,
        };
        if !force_eager && let Some(graph) = cache[idx].as_ref() {
            graph.launch().map_err(|e| {
                anyhow::anyhow!(
                    "DSv4 {} graph replay (B={}): {e}",
                    piece.label(),
                    batch_size
                )
            })?;
            return Ok(());
        }
        if force_eager {
            let scratch = self
                .batched_scratch
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("DSv4 batched scratch not allocated"))?;
            return body(scratch);
        }

        info!(
            "Capturing DSv4 piecewise CUDA Graph: piece={} B={}",
            piece.label(),
            batch_size
        );
        ctx.stream
            .begin_capture(CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| {
                anyhow::anyhow!(
                    "DSv4 {} begin_capture (B={}): {e}",
                    piece.label(),
                    batch_size
                )
            })?;
        let body_result = {
            let scratch = self
                .batched_scratch
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("DSv4 batched scratch not allocated"))?;
            body(scratch)
        };
        if let Err(err) = body_result {
            let end_result = ctx
                .stream
                .end_capture(CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
            if let Err(end_err) = end_result {
                return Err(anyhow::anyhow!(
                    "DSv4 {} graph capture body failed (B={}): {err}; end_capture after failure also failed: {end_err}",
                    piece.label(),
                    batch_size,
                ));
            }
            return Err(err);
        }
        let graph_opt = ctx
            .stream
            .end_capture(CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .map_err(|e| {
                anyhow::anyhow!("DSv4 {} end_capture (B={}): {e}", piece.label(), batch_size)
            })?;
        if let Some(graph) = graph_opt {
            graph.launch().map_err(|e| {
                anyhow::anyhow!(
                    "DSv4 {} graph first launch (B={}): {e}",
                    piece.label(),
                    batch_size
                )
            })?;
            cache[idx] = Some(graph);
        } else {
            let scratch = self
                .batched_scratch
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("DSv4 batched scratch not allocated"))?;
            body(scratch)?;
        }
        Ok(())
    }

    /// Record the served `max_seq_len` the shared pool was sized for.
    pub(super) fn set_fp8_kv_max_seq_len(&mut self, max_seq_len: usize) {
        self.fp8_kv_max_seq_len = max_seq_len;
    }

    /// Served `max_seq_len` the shared pool was sized for, or `None` when the
    /// shared pool is not allocated (env knob off → per-state path in use).
    /// Used as the single "is the shared pool active?" predicate at the decode
    /// bind site, so OFF is a pure no-op.
    pub(super) fn fp8_kv_max_seq_len(&self) -> Option<usize> {
        if self.fp8_kv_pool.is_some() {
            Some(self.fp8_kv_max_seq_len)
        } else {
            None
        }
    }

    /// Byte size of the shared FP8 KV pool for the given shape. Used both by
    /// `ensure_fp8_kv_pool` and by the model's `scheduler_runtime_workspace_bytes`
    /// budget estimate so the static KV-pool sizing reserves headroom for it.
    pub(super) fn fp8_kv_pool_bytes(num_slots: usize, layers: usize, slot_blocks: usize) -> usize {
        num_slots
            .saturating_mul(layers)
            .saturating_mul(slot_blocks)
            .saturating_mul(DSV4_FLASHMLA_MODEL1_BLOCK_BYTES)
    }

    /// Allocate (or grow) the shared FP8 KV pool to cover
    /// `num_slots × layers × slot_blocks` blocks. Allocated once for the session
    /// shape; a later call with the same-or-smaller shape is a no-op (the buffer
    /// is monotonic, mirroring the per-state `ensure_*` it replaces). Only
    /// called when `ARLE_DSV4_SHARED_KV_POOL=1`.
    pub(super) fn ensure_fp8_kv_pool(
        &mut self,
        ctx: &DeviceContext,
        num_slots: usize,
        layers: usize,
        slot_blocks: usize,
    ) -> Result<()> {
        ensure!(
            num_slots > 0 && layers > 0 && slot_blocks > 0,
            "DSv4 shared FP8 KV pool sizing requires positive shape (num_slots={num_slots}, layers={layers}, slot_blocks={slot_blocks})"
        );
        let want_bytes = Self::fp8_kv_pool_bytes(num_slots, layers, slot_blocks);
        ensure!(
            want_bytes > 0,
            "DSv4 shared FP8 KV pool byte size overflow (num_slots={num_slots}, layers={layers}, slot_blocks={slot_blocks})"
        );
        let need_grow = self
            .fp8_kv_pool
            .as_ref()
            .is_none_or(|buf| buf.len() < want_bytes)
            || self.fp8_kv_slots != num_slots
            || self.fp8_kv_layers != layers
            || self.fp8_kv_slot_blocks != slot_blocks;
        if need_grow {
            self.fp8_kv_pool = Some(ctx.stream.alloc_zeros_traced::<u8>(want_bytes).map_err(
                |err| anyhow::anyhow!("DSv4 shared FlashMLA FP8 KV pool alloc failed: {err}"),
            )?);
            self.fp8_kv_slots = num_slots;
            self.fp8_kv_layers = layers;
            self.fp8_kv_slot_blocks = slot_blocks;
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub(super) fn fp8_kv_pool_base_ptr(&mut self, ctx: &DeviceContext) -> Result<u64> {
        let pool = self
            .fp8_kv_pool
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("DSv4 shared FP8 KV pool not allocated"))?;
        let (base_ptr, guard) = pool.device_ptr_mut(&ctx.stream);
        drop(guard);
        Ok(base_ptr)
    }

    #[allow(dead_code)]
    pub(super) fn fp8_kv_total_blocks(&self) -> Result<usize> {
        ensure!(
            self.fp8_kv_pool.is_some(),
            "DSv4 shared FP8 KV pool not allocated"
        );
        self.fp8_kv_slots
            .checked_mul(self.fp8_kv_layers)
            .and_then(|v| v.checked_mul(self.fp8_kv_slot_blocks))
            .ok_or_else(|| anyhow::anyhow!("DSv4 shared FP8 KV pool total block overflow"))
    }

    pub(super) fn fp8_kv_slot_layer_block_offset(
        &self,
        slot_idx: usize,
        layer_idx: usize,
    ) -> Result<usize> {
        fp8_kv_slot_layer_block_offset_for_layout(
            slot_idx,
            layer_idx,
            self.fp8_kv_slots,
            self.fp8_kv_layers,
            self.fp8_kv_slot_blocks,
        )
    }

    /// Device base pointer + byte length of the (slot, layer) sub-range inside
    /// the shared pool. The returned pointer is the start of the slot's
    /// `slot_blocks`-block window; the pack/decode kernels index block ids
    /// relative to it exactly as they did against a per-state buffer's byte 0.
    ///
    /// `slot_blocks` is the caller's current per-(slot, layer) requirement; it
    /// must be `<=` the pool's `fp8_kv_slot_blocks` (guaranteed because the
    /// caller sizes the pool from the same `(sw_blocks, comp_blocks)` formula
    /// just before binding).
    pub(super) fn fp8_kv_slot_layer_view(
        &mut self,
        ctx: &DeviceContext,
        slot_idx: usize,
        layer_idx: usize,
        slot_blocks: usize,
    ) -> Result<(u64, usize)> {
        ensure!(
            slot_idx < self.fp8_kv_slots,
            "DSv4 FP8 KV pool slot {slot_idx} out of range for {} slots",
            self.fp8_kv_slots
        );
        ensure!(
            layer_idx < self.fp8_kv_layers,
            "DSv4 FP8 KV pool layer {layer_idx} out of range for {} layers",
            self.fp8_kv_layers
        );
        ensure!(
            slot_blocks <= self.fp8_kv_slot_blocks,
            "DSv4 FP8 KV pool slot_blocks {slot_blocks} exceeds pool capacity {}",
            self.fp8_kv_slot_blocks
        );
        let block_offset = self.fp8_kv_slot_layer_block_offset(slot_idx, layer_idx)?;
        let byte_offset = block_offset * DSV4_FLASHMLA_MODEL1_BLOCK_BYTES;
        let view_bytes = slot_blocks * DSV4_FLASHMLA_MODEL1_BLOCK_BYTES;
        let pool = self
            .fp8_kv_pool
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("DSv4 shared FP8 KV pool not allocated"))?;
        ensure!(
            byte_offset + view_bytes <= pool.len(),
            "DSv4 FP8 KV pool sub-range [{byte_offset}, {}) exceeds pool len {}",
            byte_offset + view_bytes,
            pool.len()
        );
        let (base_ptr, _g) = pool.device_ptr_mut(&ctx.stream);
        Ok((base_ptr + byte_offset as u64, view_bytes))
    }
}

impl DecodeContextOps for DeepseekBatchDecodeBuffers {
    fn upload_token_ids(&mut self, ctx: &DeviceContext, tokens: &[u32]) -> Result<()> {
        ensure!(
            tokens.len() <= self.max_batch_size,
            "DeepSeek V4 Phase 2A.0 decode batch {} exceeds context capacity {}",
            tokens.len(),
            self.max_batch_size
        );
        if let Some(scratch) = self.batched_scratch.as_mut() {
            scratch.upload_token_ids(ctx, tokens)?;
        }
        Ok(())
    }

    fn update_metadata(
        &mut self,
        _ctx: &DeviceContext,
        _pool: &PagedKVPool,
        slot_indices: &[usize],
    ) -> Result<bool> {
        ensure!(
            slot_indices.len() <= self.max_batch_size,
            "DeepSeek V4 Phase 2A.0 slot batch {} exceeds context capacity {}",
            slot_indices.len(),
            self.max_batch_size
        );
        Ok(false)
    }

    fn plan_attention(
        &mut self,
        _ctx: &DeviceContext,
        batch_size: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _page_size: usize,
        _head_dim: usize,
        _kv_format: KVFormat,
    ) -> Result<()> {
        ensure!(
            batch_size <= self.max_batch_size,
            "DeepSeek V4 Phase 2A.0 attention batch {batch_size} exceeds context capacity {}",
            self.max_batch_size
        );
        Ok(())
    }

    fn set_batch_size(&mut self, _bs: usize) {}

    fn invalidate_graph_cache(&mut self, batch_size: usize) {
        if batch_size >= 1 && batch_size <= self.max_batch_size {
            self.input_graph_cache[batch_size - 1] = None;
            self.head_graph_cache[batch_size - 1] = None;
        }
    }

    fn force_eager_once(&mut self) {
        self.force_eager_once = true;
    }
}

fn fp8_kv_slot_layer_block_offset_for_layout(
    slot_idx: usize,
    layer_idx: usize,
    fp8_kv_slots: usize,
    fp8_kv_layers: usize,
    fp8_kv_slot_blocks: usize,
) -> Result<usize> {
    ensure!(
        slot_idx < fp8_kv_slots,
        "DSv4 FP8 KV pool slot {slot_idx} out of range for {fp8_kv_slots} slots"
    );
    ensure!(
        layer_idx < fp8_kv_layers,
        "DSv4 FP8 KV pool layer {layer_idx} out of range for {fp8_kv_layers} layers"
    );
    ensure!(
        fp8_kv_slot_blocks > 0,
        "DSv4 FP8 KV pool slot block count must be positive"
    );
    let slot_layer_idx = slot_idx
        .checked_mul(fp8_kv_layers)
        .and_then(|v| v.checked_add(layer_idx))
        .ok_or_else(|| anyhow::anyhow!("DSv4 FP8 KV pool slot/layer index overflow"))?;
    slot_layer_idx
        .checked_mul(fp8_kv_slot_blocks)
        .ok_or_else(|| anyhow::anyhow!("DSv4 FP8 KV pool block offset overflow"))
}

#[cfg(test)]
mod tests {
    use super::fp8_kv_slot_layer_block_offset_for_layout;

    #[test]
    fn fp8_kv_slot_layer_block_offset_is_slot_major() {
        assert_eq!(
            fp8_kv_slot_layer_block_offset_for_layout(0, 0, 4, 3, 11).unwrap(),
            0
        );
        assert_eq!(
            fp8_kv_slot_layer_block_offset_for_layout(0, 2, 4, 3, 11).unwrap(),
            22
        );
        assert_eq!(
            fp8_kv_slot_layer_block_offset_for_layout(1, 0, 4, 3, 11).unwrap(),
            33
        );
        assert_eq!(
            fp8_kv_slot_layer_block_offset_for_layout(3, 2, 4, 3, 11).unwrap(),
            121
        );
    }

    #[test]
    fn fp8_kv_slot_layer_block_offset_rejects_out_of_range() {
        assert!(fp8_kv_slot_layer_block_offset_for_layout(4, 0, 4, 3, 11).is_err());
        assert!(fp8_kv_slot_layer_block_offset_for_layout(0, 3, 4, 3, 11).is_err());
        assert!(fp8_kv_slot_layer_block_offset_for_layout(0, 0, 4, 3, 0).is_err());
    }
}

#[derive(Clone, Copy)]
enum DeepseekDecodeGraphPiece {
    Input,
    Head,
}

impl DeepseekDecodeGraphPiece {
    fn label(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Head => "head",
        }
    }
}
