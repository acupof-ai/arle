//! FlashInfer decode metadata management.
//!
//! Owns the GPU-side paged KV metadata buffers (positions, indptr, indices,
//! last_page_len) and the FlashInfer workspace. Extracted from
//! `model::qwen3::batch_decode` so multiple model implementations can share it.

use anyhow::Result;
use cudarc::driver::CudaSlice;

use super::paged_kv::PagedKVPool;
use super::tensor::DeviceContext;
use crate::ops;
use crate::ops::FlashInferWorkspace;

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
    /// Scratch buffer for building positions on the host side.
    positions_scratch: Vec<i32>,
    /// Scratch buffer for KV index H2D.
    indices_scratch: Vec<i32>,
    /// Cached host-side indptr from last `update()`, reused by `plan()`.
    indptr_h: Vec<i32>,
    /// Previous slot indices that produced `indices_scratch`.
    prev_slot_indices: Vec<usize>,
    /// Previous slot epochs; changes when a slot is recycled for a new request.
    prev_slot_epochs: Vec<u64>,
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
            positions_scratch: Vec::with_capacity(max_batch_size),
            indices_scratch: Vec::with_capacity(max_total_pages.max(1)),
            indptr_h: Vec::with_capacity(max_batch_size + 1),
            prev_slot_indices: Vec::with_capacity(max_batch_size),
            prev_slot_epochs: Vec::with_capacity(max_batch_size),
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
    /// Returns `true` if the kv_indices GPU buffer was reallocated (caller
    /// should invalidate any CUDA graph that captured the old pointer).
    pub(crate) fn update(
        &mut self,
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        slot_indices: &[usize],
    ) -> Result<bool> {
        let slot_epochs: Vec<u64> = slot_indices
            .iter()
            .map(|&slot| pool.slot_epoch(slot))
            .collect();
        let next_indptr_h = pool.build_indptr(slot_indices);
        let same_batch_identity =
            self.prev_slot_indices == slot_indices && self.prev_slot_epochs == slot_epochs;
        let can_incremental_update = same_batch_identity
            && can_append_decode_step(&self.indptr_h, &next_indptr_h)
            && self.indices_scratch.len() == self.total_tokens;
        let reuse_plan = same_batch_identity
            && self.plan_batch_size == slot_indices.len()
            && can_reuse_decode_plan(&self.indptr_h, &next_indptr_h);
        let last_page_lens_h = pool.build_last_page_lens(slot_indices);

        // Build positions (each request's current sequence position).
        self.positions_scratch.clear();
        self.positions_scratch
            .extend(slot_indices.iter().map(|&si| (pool.seq_len(si) - 1) as i32));
        let prev_indptr_h = std::mem::replace(&mut self.indptr_h, next_indptr_h);

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

        let new_total = *self
            .indptr_h
            .last()
            .expect("invariant: indptr_h from build_indptr() always has at least one element")
            as usize;
        let mut reallocated = false;

        self.last_token_scratch.clear();
        self.last_token_scratch
            .extend(pool.build_last_indices(slot_indices));

        if can_incremental_update {
            append_last_token_indices_in_place(
                &mut self.indices_scratch,
                &prev_indptr_h,
                &self.indptr_h,
                &self.last_token_scratch,
            );
        } else {
            self.indices_scratch.clear();
            self.indices_scratch.extend(flatten_token_indices(
                slot_indices.iter().map(|&slot| pool.token_indices(slot)),
            ));
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
        self.total_tokens = new_total;

        // Upload last-token pool indices (for INT8 quantize-after-write).
        ctx.stream
            .memcpy_htod(&self.last_token_scratch, &mut self.last_token_indices)
            .map_err(|e| anyhow::anyhow!("H2D last_token_indices: {e}"))?;

        // Re-plan when the batch size changed, lengths did not follow the
        // steady-state decode progression, or kv_indices had to move.
        if reallocated || !reuse_plan {
            self.plan_dirty = true;
        }
        self.prev_slot_indices.clear();
        self.prev_slot_indices.extend_from_slice(slot_indices);
        self.prev_slot_epochs.clear();
        self.prev_slot_epochs.extend_from_slice(&slot_epochs);

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

fn flatten_token_indices<'a>(token_groups: impl IntoIterator<Item = &'a [u32]>) -> Vec<i32> {
    let mut flat = Vec::new();
    for group in token_groups {
        flat.extend(group.iter().map(|&idx| idx as i32));
    }
    flat
}

fn can_append_decode_step(prev_indptr: &[i32], next_indptr: &[i32]) -> bool {
    if prev_indptr.len() != next_indptr.len() || prev_indptr.len() < 2 {
        return false;
    }

    prev_indptr
        .windows(2)
        .zip(next_indptr.windows(2))
        .all(|(prev, next)| (next[1] - next[0]) == (prev[1] - prev[0]) + 1)
}

fn append_last_token_indices_in_place(
    indices_scratch: &mut Vec<i32>,
    prev_indptr: &[i32],
    next_indptr: &[i32],
    last_token_indices: &[i32],
) {
    if next_indptr.is_empty() {
        indices_scratch.clear();
        return;
    }

    let new_total = *next_indptr
        .last()
        .expect("invariant: next_indptr always has at least one element")
        as usize;
    indices_scratch.resize(new_total, 0);

    for i in (0..last_token_indices.len()).rev() {
        let old_start = prev_indptr[i] as usize;
        let old_end = prev_indptr[i + 1] as usize;
        let new_start = next_indptr[i] as usize;
        let new_end = next_indptr[i + 1] as usize;
        indices_scratch.copy_within(old_start..old_end, new_start);
        indices_scratch[new_end - 1] = last_token_indices[i];
    }
}

fn can_reuse_decode_plan(prev_indptr: &[i32], next_indptr: &[i32]) -> bool {
    can_append_decode_step(prev_indptr, next_indptr)
}

#[cfg(test)]
mod tests {
    use super::{
        append_last_token_indices_in_place, can_append_decode_step, can_reuse_decode_plan,
        flatten_token_indices,
    };

    #[test]
    fn flatten_token_indices_preserves_per_slot_segments() {
        let flat =
            flatten_token_indices([&[10_u32, 11, 12][..], &[20_u32, 21, 22][..], &[30_u32][..]]);
        assert_eq!(flat, vec![10, 11, 12, 20, 21, 22, 30]);
    }

    #[test]
    fn decode_plan_reuse_allows_steady_state_growth() {
        assert!(can_reuse_decode_plan(&[0, 3, 7], &[0, 4, 9]));
    }

    #[test]
    fn decode_plan_reuse_rejects_batch_shape_changes() {
        assert!(!can_reuse_decode_plan(&[0, 3, 7], &[0, 3, 7]));
        assert!(!can_reuse_decode_plan(&[0, 3, 7], &[0, 5, 8]));
        assert!(!can_reuse_decode_plan(&[0, 3, 7], &[0, 3]));
        assert!(!can_reuse_decode_plan(&[0, 3, 7], &[0, 2, 4]));
    }

    #[test]
    fn incremental_append_updates_each_slot_segment_without_corruption() {
        let mut scratch = vec![10, 11, 20, 21, 22, 30];
        append_last_token_indices_in_place(
            &mut scratch,
            &[0, 2, 5, 6],
            &[0, 3, 7, 9],
            &[12, 23, 31],
        );
        assert_eq!(scratch, vec![10, 11, 12, 20, 21, 22, 23, 30, 31]);
    }

    #[test]
    fn append_decode_step_requires_every_slot_to_grow_by_one() {
        assert!(can_append_decode_step(&[0, 2, 5], &[0, 3, 7]));
        assert!(!can_append_decode_step(&[0, 2, 5], &[0, 2, 6]));
        assert!(!can_append_decode_step(&[0, 2, 5], &[0, 4, 6]));
    }
}
