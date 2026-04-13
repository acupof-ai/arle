//! FlashInfer decode metadata management.
//!
//! Owns the GPU-side paged KV metadata buffers (positions, indptr, indices,
//! last_page_len) and the FlashInfer workspace. Extracted from
//! `model::qwen3::batch_decode` so multiple model implementations can share it.

use anyhow::Result;
use cudarc::driver::CudaSlice;

use crate::ops;
use crate::ops::FlashInferWorkspace;
use super::paged_kv::PagedKVPool;
use super::tensor::DeviceContext;

/// GPU-side metadata buffers for FlashInfer batched decode attention.
///
/// Manages positions, KV indptr/indices/last_page_len arrays, the FlashInfer
/// workspace, and host-side scratch buffers for incremental index updates.
pub(crate) struct FlashInferDecodeMetadata {
    pub positions: CudaSlice<i32>,
    pub kv_indices: CudaSlice<i32>,
    pub kv_indptr: CudaSlice<i32>,
    pub kv_last_page_len: CudaSlice<i32>,
    /// Q indptr for tensor-core decode: [0, 1, 2, ..., B] (1 token per request).
    #[allow(dead_code)]
    pub q_indptr: CudaSlice<i32>,
    pub flashinfer_ws: FlashInferWorkspace,
    pub max_total_pages: usize,
    /// Total tokens in kv_indices from previous step (for incremental update).
    prev_total_tokens: usize,
    /// Previous batch slot indices (for detecting batch composition changes).
    prev_slot_indices: Vec<usize>,
    /// Scratch buffer for building positions on the host side.
    positions_scratch: Vec<i32>,
    /// Scratch buffer for incremental index H2D (avoids alloc).
    indices_scratch: Vec<i32>,
    /// Cached host-side indptr from last `update()`, reused by `plan()`.
    indptr_h: Vec<i32>,
    /// Host-side q_indptr for TC decode: [0, 1, 2, ..., max_batch_size].
    qo_indptr_h: Vec<i32>,
    /// Pool index of the most recently allocated token per request [max_batch_size].
    /// Used for INT8 quantize-after-write in batched decode.
    pub last_token_indices: CudaSlice<i32>,
    /// Host scratch for building last_token_indices.
    last_token_scratch: Vec<i32>,
    /// Total tokens across all requests from last update (sum of seq_lens).
    total_tokens: usize,
    /// Batch size from last successful plan (for plan caching).
    plan_batch_size: usize,
    /// Whether the plan needs to be re-run (batch composition changed).
    plan_dirty: bool,
}

impl FlashInferDecodeMetadata {
    /// Allocate metadata buffers for up to `max_batch_size` requests.
    /// `max_total_pages` should be large enough for the worst-case total KV pages.
    pub(crate) fn new(
        ctx: &DeviceContext,
        max_batch_size: usize,
        max_total_pages: usize,
        num_qheads: usize,
    ) -> Result<Self> {
        // Pre-build q_indptr for TC decode: [0, 1, 2, ..., max_batch_size]
        let qo_indptr_h: Vec<i32> = (0..=(max_batch_size as i32)).collect();
        let mut q_indptr: CudaSlice<i32> = ctx
            .stream
            .alloc_zeros(max_batch_size + 1)
            .map_err(|e| anyhow::anyhow!("Alloc q_indptr failed: {e}"))?;
        ctx.stream
            .memcpy_htod(&qo_indptr_h, &mut q_indptr)
            .map_err(|e| anyhow::anyhow!("H2D q_indptr: {e}"))?;

        Ok(Self {
            positions: ctx
                .stream
                .alloc_zeros(max_batch_size)
                .map_err(|e| anyhow::anyhow!("Alloc positions failed: {e}"))?,
            kv_indices: ctx
                .stream
                .alloc_zeros(max_total_pages.max(1))
                .map_err(|e| anyhow::anyhow!("Alloc kv_indices failed: {e}"))?,
            kv_indptr: ctx
                .stream
                .alloc_zeros(max_batch_size + 1)
                .map_err(|e| anyhow::anyhow!("Alloc kv_indptr failed: {e}"))?,
            kv_last_page_len: ctx
                .stream
                .alloc_zeros(max_batch_size)
                .map_err(|e| anyhow::anyhow!("Alloc kv_last_page_len failed: {e}"))?,
            q_indptr,
            flashinfer_ws: FlashInferWorkspace::new(ctx, max_batch_size, num_qheads)?,
            max_total_pages,
            prev_total_tokens: 0,
            prev_slot_indices: Vec::new(),
            positions_scratch: Vec::with_capacity(max_batch_size),
            indices_scratch: Vec::with_capacity(max_batch_size),
            indptr_h: Vec::with_capacity(max_batch_size + 1),
            qo_indptr_h,
            last_token_indices: ctx
                .stream
                .alloc_zeros(max_batch_size)
                .map_err(|e| anyhow::anyhow!("Alloc last_token_indices failed: {e}"))?,
            last_token_scratch: Vec::with_capacity(max_batch_size),
            total_tokens: 0,
            plan_batch_size: 0,
            plan_dirty: true,
        })
    }

    /// Upload metadata (positions, indptr, last_page_len, KV indices) to GPU.
    ///
    /// Uses incremental index updates when the batch composition is unchanged
    /// (same slot indices as previous step), falling back to a full rebuild
    /// when the batch changes or on the first call.
    ///
    /// Returns `true` if the kv_indices GPU buffer was reallocated (caller
    /// should invalidate any CUDA graph that captured the old pointer).
    pub(crate) fn update(
        &mut self,
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        slot_indices: &[usize],
    ) -> Result<bool> {
        self.indptr_h = pool.build_indptr(slot_indices);
        let last_page_lens_h = pool.build_last_page_lens(slot_indices);

        // Build positions (each request's current sequence position).
        self.positions_scratch.clear();
        self.positions_scratch
            .extend(slot_indices.iter().map(|&si| (pool.seq_len(si) - 1) as i32));

        // H2D copies -- positions, indptr, last_page_len (small, fixed size).
        ctx.stream
            .memcpy_htod(&self.positions_scratch, &mut self.positions)
            .map_err(|e| anyhow::anyhow!("H2D positions: {e}"))?;
        ctx.stream
            .memcpy_htod(&self.indptr_h, &mut self.kv_indptr)
            .map_err(|e| anyhow::anyhow!("H2D indptr: {e}"))?;
        ctx.stream
            .memcpy_htod(&last_page_lens_h, &mut self.kv_last_page_len)
            .map_err(|e| anyhow::anyhow!("H2D last_page_len: {e}"))?;

        // KV indices update: if same batch composition as last step, only do
        // an incremental update (H2D of full index array using scratch buffer).
        // Otherwise, full rebuild of the O(total_tokens) indices array.
        let same_batch = self.prev_slot_indices == slot_indices;
        let new_total = *self
            .indptr_h
            .last()
            .expect("invariant: indptr_h from build_indptr() always has at least one element")
            as usize;
        let mut reallocated = false;

        if same_batch && self.prev_total_tokens > 0 && new_total <= self.max_total_pages {
            // Truly incremental: each slot grew by exactly 1 token since last step.
            // Append only the new token indices to the existing scratch buffer,
            // then H2D the entire array. This avoids O(total_tokens) iteration.
            //
            // We must insert each slot's new token at the correct offset within
            // the flat indices array (after all that slot's existing tokens).
            // With page_size=1, indptr gives cumulative token counts per slot.
            for (i, &slot) in slot_indices.iter().enumerate() {
                let insert_pos = self.indptr_h[i + 1] as usize - 1; // last token of slot i
                let last_idx = *pool.token_indices(slot).last().ok_or_else(|| {
                    anyhow::anyhow!(
                        "FlashInfer incremental update: slot {} has no token indices",
                        slot
                    )
                })? as i32;
                if insert_pos < self.indices_scratch.len() {
                    self.indices_scratch[insert_pos] = last_idx;
                } else {
                    // Grow scratch to fit
                    self.indices_scratch.resize(insert_pos + 1, 0);
                    self.indices_scratch[insert_pos] = last_idx;
                }
            }
            // Truncate to exact size
            self.indices_scratch.truncate(new_total);
            ctx.stream
                .memcpy_htod(&self.indices_scratch, &mut self.kv_indices)
                .map_err(|e| anyhow::anyhow!("H2D indices: {e}"))?;
        } else {
            // Full rebuild (first call, or batch composition changed).
            // Populate indices_scratch so next incremental step can build on it.
            self.indices_scratch.clear();
            for &slot in slot_indices.iter() {
                for &idx in pool.token_indices(slot) {
                    self.indices_scratch.push(idx as i32);
                }
            }
            if self.indices_scratch.len() > self.max_total_pages {
                self.kv_indices = ctx
                    .stream
                    .alloc_zeros(self.indices_scratch.len())
                    .map_err(|e| anyhow::anyhow!("Realloc kv_indices: {e}"))?;
                self.max_total_pages = self.indices_scratch.len();
                reallocated = true;
            }
            ctx.stream
                .memcpy_htod(&self.indices_scratch, &mut self.kv_indices)
                .map_err(|e| anyhow::anyhow!("H2D indices: {e}"))?;
        }
        self.prev_total_tokens = new_total;
        self.total_tokens = new_total;

        // Upload last-token pool indices (for INT8 quantize-after-write).
        self.last_token_scratch.clear();
        self.last_token_scratch
            .extend(pool.build_last_indices(slot_indices));
        ctx.stream
            .memcpy_htod(&self.last_token_scratch, &mut self.last_token_indices)
            .map_err(|e| anyhow::anyhow!("H2D last_token_indices: {e}"))?;

        // Mark plan dirty if batch composition or size changed.
        if !same_batch || slot_indices.len() != self.plan_batch_size || reallocated {
            self.plan_dirty = true;
        }
        self.prev_slot_indices.clear();
        self.prev_slot_indices.extend_from_slice(slot_indices);

        Ok(reallocated)
    }

    /// Call FlashInfer's plan step (CPU-side scheduling).
    ///
    /// Must be called once per decode step after `update()`, before any layer
    /// runs. The plan result works for all layers since KV layout is the same.
    ///
    /// Skips re-planning when batch composition is unchanged (steady-state decode),
    /// since the work partitioning across CUDA blocks is stable when only KV lengths
    /// grow by 1 each step.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn plan(
        &mut self,
        ctx: &DeviceContext,
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        page_size: usize,
        head_dim: usize,
    ) -> Result<()> {
        if !self.plan_dirty && self.plan_batch_size == batch_size {
            return Ok(());
        }
        ops::flashinfer_plan(
            ctx,
            &self.indptr_h,
            &mut self.flashinfer_ws,
            batch_size,
            num_heads,
            num_kv_heads,
            page_size,
            head_dim,
        )?;
        self.plan_batch_size = batch_size;
        self.plan_dirty = false;
        Ok(())
    }

    /// HD256 decode plan (Qwen3.5 full attention, head_dim=256).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn plan_hd256(
        &mut self,
        ctx: &DeviceContext,
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        page_size: usize,
        head_dim: usize,
    ) -> Result<()> {
        if !self.plan_dirty && self.plan_batch_size == batch_size {
            return Ok(());
        }
        ops::flashinfer_plan_hd256(
            ctx,
            &self.indptr_h,
            &mut self.flashinfer_ws,
            batch_size,
            num_heads,
            num_kv_heads,
            page_size,
            head_dim,
        )?;
        self.plan_batch_size = batch_size;
        self.plan_dirty = false;
        Ok(())
    }

    /// Tensor-core decode plan (uses PrefillPlan for flat ITL at long contexts).
    ///
    /// For GQA group_size >= 4, this provides better performance than the standard
    /// decode kernel by tiling across KV chunks with tensor cores.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tc_plan(
        &mut self,
        ctx: &DeviceContext,
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        page_size: usize,
        head_dim: usize,
    ) -> Result<()> {
        // TC decode always re-plans (PrefillPlan depends on KV lengths for split-KV).
        let qo_indptr = &self.qo_indptr_h[..batch_size + 1];
        ops::flashinfer_tc_plan(
            ctx,
            qo_indptr,
            &self.indptr_h,
            &mut self.flashinfer_ws,
            batch_size,
            num_heads,
            num_kv_heads,
            page_size,
            head_dim,
        )?;
        self.plan_batch_size = batch_size;
        self.plan_dirty = false;
        Ok(())
    }

    /// Total tokens across all requests from the last `update()` call.
    pub(crate) fn total_tokens(&self) -> usize {
        self.total_tokens
    }
}
