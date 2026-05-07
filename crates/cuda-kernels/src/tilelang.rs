//! TileLang decode metadata management.
//!
//! Owns the GPU-side paged KV metadata buffers (positions, indptr, indices,
//! last_page_len). Extracted from
//! `model::qwen3::batch_decode` so multiple model implementations can share it.

use anyhow::{Result, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::ffi;
use super::paged_kv::PagedKVPool;
use super::tensor::DeviceContext;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeMetaUpdate {
    Full,
    SamePages,
    AppendedPages,
}

/// GPU-side metadata buffers for TileLang batched paged attention.
///
/// Manages positions, KV indptr/indices/last_page_len arrays and host-side
/// scratch buffers for incremental index updates.
pub struct TileLangDecodeMetadata {
    pub positions: CudaSlice<i32>,
    pub kv_indices: CudaSlice<i32>,
    pub kv_indptr: CudaSlice<i32>,
    pub kv_last_page_len: CudaSlice<i32>,
    prev_kv_indptr: CudaSlice<i32>,
    append_indptr: CudaSlice<i32>,
    appended_page_indices: CudaSlice<i32>,
    /// Q indptr for tensor-core paged attention.
    ///
    /// Decode-only steps use `[0, 1, 2, ..., B]` (1 Q row per request).
    /// Mixed decode+prefill steps append packed varlen prefill rows in request
    /// order, e.g. `[0, 1, 2, 2 + len(prefill_0), ...]`.
    pub qo_indptr: CudaSlice<i32>,
    pub tilelang_ws: TileLangWorkspace,
    pub max_total_pages: usize,
    /// Scratch buffer for building positions on the host side.
    positions_scratch: Vec<i32>,
    /// Scratch buffer for KV index H2D.
    indices_scratch: Vec<i32>,
    indptr_build_scratch: Vec<i32>,
    last_page_lens_scratch: Vec<i32>,
    append_indptr_h: Vec<i32>,
    appended_page_scratch: Vec<i32>,
    /// Cached host-side indptr from last `update()`, reused by `plan()`.
    /// Public so callers (e.g. TileLang TC decode dispatch) can read the
    /// total page count from the last entry without round-tripping through
    /// device memory.
    pub indptr_h: Vec<i32>,
    /// Previous slot indices that produced `indices_scratch`.
    prev_slot_indices: Vec<usize>,
    /// Previous slot epochs; changes when a slot is recycled for a new request.
    prev_slot_epochs: Vec<u64>,
    /// Host-side q_indptr scratch for TC paged attention.
    pub qo_indptr_h: Vec<i32>,
    /// Physical page id of the last logical page per request `[max_batch_size]`.
    /// Used only by the page_size=1 quantize-after-write fast paths.
    pub last_token_indices: CudaSlice<i32>,
    /// Host scratch for building last_token_indices.
    last_token_scratch: Vec<i32>,
    /// Total pages across all requests from last update.
    total_tokens: usize,
    /// Batch size from last successful plan (for plan caching).
    plan_batch_size: usize,
    /// Whether the plan needs to be re-run (batch composition changed).
    plan_dirty: bool,
    fast_page16_metadata: bool,
}

/// Workspace for TileLang attention phase kernels.
///
/// The HD128 BF16 split-KV decode path stores FlashDecoding partial outputs in
/// persistent scheduler-owned buffers so runtime launches never allocate. Mixed
/// and prefill metadata can still request a zero-sized workspace via
/// `new_with_float_bytes`.
pub struct TileLangWorkspace {
    partial_out: Option<CudaSlice<f32>>,
    partial_m: Option<CudaSlice<f32>>,
    partial_l: Option<CudaSlice<f32>>,
    max_batch_size: usize,
    num_qo_heads: usize,
    max_splits: usize,
    _extra_float_workspace: Option<CudaSlice<f32>>,
}

unsafe impl Send for TileLangWorkspace {}

fn decode_metadata_fast_page16_enabled() -> bool {
    matches!(
        std::env::var("INFER_DECODE_METADATA_FAST_PAGE16").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "on")
    )
}

impl TileLangWorkspace {
    pub const DEFAULT_FLOAT_WORKSPACE_BYTES: usize = 0;
    pub const HD256_FLOAT_WORKSPACE_BYTES: usize = 0;
    pub const HD128_DECODE_MAX_SPLITS: usize = 16;
    pub const HD128_DECODE_SPLIT_MIN_TOKENS: usize = 2048;

    pub fn new(ctx: &DeviceContext, max_batch_size: usize, num_qo_heads: usize) -> Result<Self> {
        Self::new_hd128_decode(ctx, max_batch_size, num_qo_heads)
    }

    pub fn device_bytes(
        _max_batch_size: usize,
        _num_qo_heads: usize,
        float_workspace_bytes: usize,
    ) -> usize {
        align_float_bytes(float_workspace_bytes)
    }

    pub fn default_device_bytes(max_batch_size: usize, num_qo_heads: usize) -> usize {
        Self::device_bytes(
            max_batch_size,
            num_qo_heads,
            Self::DEFAULT_FLOAT_WORKSPACE_BYTES,
        )
    }

    pub fn new_with_float_bytes(
        ctx: &DeviceContext,
        max_batch_size: usize,
        num_qo_heads: usize,
        float_workspace_bytes: usize,
    ) -> Result<Self> {
        let extra_elems = align_float_bytes(float_workspace_bytes) / std::mem::size_of::<f32>();
        let extra_float_workspace =
            if extra_elems > 0 {
                Some(ctx.stream.alloc_zeros(extra_elems).map_err(|e| {
                    anyhow::anyhow!("Alloc TileLang extra float workspace failed: {e}")
                })?)
            } else {
                None
            };
        Ok(Self {
            partial_out: None,
            partial_m: None,
            partial_l: None,
            max_batch_size,
            num_qo_heads,
            max_splits: 0,
            _extra_float_workspace: extra_float_workspace,
        })
    }

    pub fn hd128_decode_device_bytes(max_batch_size: usize, num_qo_heads: usize) -> usize {
        hd128_split_partial_out_elems(max_batch_size, num_qo_heads, Self::HD128_DECODE_MAX_SPLITS)
            .saturating_add(hd128_split_stat_elems(
                max_batch_size,
                num_qo_heads,
                Self::HD128_DECODE_MAX_SPLITS,
            ))
            .saturating_add(hd128_split_stat_elems(
                max_batch_size,
                num_qo_heads,
                Self::HD128_DECODE_MAX_SPLITS,
            ))
            .saturating_mul(std::mem::size_of::<f32>())
    }

    fn new_hd128_decode(
        ctx: &DeviceContext,
        max_batch_size: usize,
        num_qo_heads: usize,
    ) -> Result<Self> {
        let max_splits = Self::HD128_DECODE_MAX_SPLITS;
        let partial_out_elems =
            hd128_split_partial_out_elems(max_batch_size, num_qo_heads, max_splits);
        let stat_elems = hd128_split_stat_elems(max_batch_size, num_qo_heads, max_splits);
        Ok(Self {
            partial_out: Some(
                ctx.stream
                    .alloc_zeros(partial_out_elems.max(1))
                    .map_err(|e| anyhow::anyhow!("Alloc TileLang split partial_out failed: {e}"))?,
            ),
            partial_m: Some(
                ctx.stream
                    .alloc_zeros(stat_elems.max(1))
                    .map_err(|e| anyhow::anyhow!("Alloc TileLang split partial_m failed: {e}"))?,
            ),
            partial_l: Some(
                ctx.stream
                    .alloc_zeros(stat_elems.max(1))
                    .map_err(|e| anyhow::anyhow!("Alloc TileLang split partial_l failed: {e}"))?,
            ),
            max_batch_size,
            num_qo_heads,
            max_splits,
            _extra_float_workspace: None,
        })
    }

    pub fn hd128_decode_num_splits(&self) -> i32 {
        self.max_splits as i32
    }

    pub fn hd128_decode_split_workspace_mut(
        &mut self,
        batch_size: usize,
        num_qo_heads: usize,
    ) -> Option<(
        &mut CudaSlice<f32>,
        &mut CudaSlice<f32>,
        &mut CudaSlice<f32>,
    )> {
        if self.max_splits == 0
            || batch_size > self.max_batch_size
            || num_qo_heads != self.num_qo_heads
        {
            return None;
        }
        match (
            &mut self.partial_out,
            &mut self.partial_m,
            &mut self.partial_l,
        ) {
            (Some(partial_out), Some(partial_m), Some(partial_l)) => {
                Some((partial_out, partial_m, partial_l))
            }
            _ => None,
        }
    }
}

fn align_float_bytes(bytes: usize) -> usize {
    bytes
        .div_ceil(std::mem::size_of::<f32>())
        .saturating_mul(std::mem::size_of::<f32>())
}

fn hd128_split_partial_out_elems(
    max_batch_size: usize,
    num_qo_heads: usize,
    max_splits: usize,
) -> usize {
    max_splits
        .saturating_mul(max_batch_size)
        .saturating_mul(num_qo_heads)
        .saturating_mul(128)
}

fn hd128_split_stat_elems(max_batch_size: usize, num_qo_heads: usize, max_splits: usize) -> usize {
    max_splits
        .saturating_mul(max_batch_size)
        .saturating_mul(num_qo_heads)
}

impl TileLangDecodeMetadata {
    /// Allocate metadata buffers for up to `max_batch_size` requests.
    /// `max_total_pages` should be large enough for the worst-case total KV pages.
    pub fn new(
        ctx: &DeviceContext,
        max_batch_size: usize,
        max_total_pages: usize,
        num_qheads: usize,
    ) -> Result<Self> {
        Self::new_impl(ctx, max_batch_size, max_total_pages, num_qheads, false, 0)
    }

    pub fn device_bytes(max_batch_size: usize, max_total_pages: usize, num_qheads: usize) -> usize {
        Self::device_bytes_impl(max_batch_size, max_total_pages, num_qheads, false, 0)
    }

    pub fn new_with_hd128_decode_workspace(
        ctx: &DeviceContext,
        max_batch_size: usize,
        max_total_pages: usize,
        num_qheads: usize,
    ) -> Result<Self> {
        Self::new_impl(ctx, max_batch_size, max_total_pages, num_qheads, true, 0)
    }

    pub fn device_bytes_with_hd128_decode_workspace(
        max_batch_size: usize,
        max_total_pages: usize,
        num_qheads: usize,
    ) -> usize {
        Self::device_bytes_impl(max_batch_size, max_total_pages, num_qheads, true, 0)
    }

    pub fn device_bytes_with_float_workspace(
        max_batch_size: usize,
        max_total_pages: usize,
        num_qheads: usize,
        float_workspace_bytes: usize,
    ) -> usize {
        Self::device_bytes_impl(
            max_batch_size,
            max_total_pages,
            num_qheads,
            false,
            float_workspace_bytes,
        )
    }

    fn device_bytes_impl(
        max_batch_size: usize,
        max_total_pages: usize,
        num_qheads: usize,
        include_decode_workspace: bool,
        float_workspace_bytes: usize,
    ) -> usize {
        let i32_bytes = std::mem::size_of::<i32>();
        let workspace_bytes = if include_decode_workspace {
            TileLangWorkspace::hd128_decode_device_bytes(max_batch_size, num_qheads)
        } else {
            TileLangWorkspace::device_bytes(max_batch_size, num_qheads, float_workspace_bytes)
        };
        workspace_bytes
            .saturating_add(max_batch_size.saturating_mul(i32_bytes)) // positions
            .saturating_add(max_total_pages.max(1).saturating_mul(i32_bytes)) // kv_indices
            .saturating_add((max_batch_size + 1).saturating_mul(i32_bytes)) // kv_indptr
            .saturating_add(max_batch_size.saturating_mul(i32_bytes)) // kv_last_page_len
            .saturating_add((max_batch_size + 1).saturating_mul(i32_bytes)) // prev_kv_indptr
            .saturating_add((max_batch_size + 1).saturating_mul(i32_bytes)) // append_indptr
            .saturating_add(max_batch_size.max(1).saturating_mul(i32_bytes)) // appended_page_indices
            .saturating_add((max_batch_size + 1).saturating_mul(i32_bytes)) // qo_indptr
            .saturating_add(max_batch_size.saturating_mul(i32_bytes)) // last_token_indices
    }

    pub fn new_with_float_workspace_bytes(
        ctx: &DeviceContext,
        max_batch_size: usize,
        max_total_pages: usize,
        num_qheads: usize,
        float_workspace_bytes: usize,
    ) -> Result<Self> {
        Self::new_impl(
            ctx,
            max_batch_size,
            max_total_pages,
            num_qheads,
            false,
            float_workspace_bytes,
        )
    }

    fn new_impl(
        ctx: &DeviceContext,
        max_batch_size: usize,
        max_total_pages: usize,
        num_qheads: usize,
        include_decode_workspace: bool,
        float_workspace_bytes: usize,
    ) -> Result<Self> {
        let qo_indptr: CudaSlice<i32> = ctx
            .stream
            .alloc_zeros(max_batch_size + 1)
            .map_err(|e| anyhow::anyhow!("Alloc qo_indptr failed: {e}"))?;
        let tilelang_ws = if include_decode_workspace {
            TileLangWorkspace::new(ctx, max_batch_size, num_qheads)?
        } else {
            TileLangWorkspace::new_with_float_bytes(
                ctx,
                max_batch_size,
                num_qheads,
                float_workspace_bytes,
            )?
        };

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
            prev_kv_indptr: ctx
                .stream
                .alloc_zeros(max_batch_size + 1)
                .map_err(|e| anyhow::anyhow!("Alloc prev_kv_indptr failed: {e}"))?,
            append_indptr: ctx
                .stream
                .alloc_zeros(max_batch_size + 1)
                .map_err(|e| anyhow::anyhow!("Alloc append_indptr failed: {e}"))?,
            appended_page_indices: ctx
                .stream
                .alloc_zeros(max_batch_size.max(1))
                .map_err(|e| anyhow::anyhow!("Alloc appended_page_indices failed: {e}"))?,
            qo_indptr,
            tilelang_ws,
            max_total_pages,
            positions_scratch: Vec::with_capacity(max_batch_size),
            indices_scratch: Vec::with_capacity(max_total_pages.max(1)),
            indptr_build_scratch: Vec::with_capacity(max_batch_size + 1),
            last_page_lens_scratch: Vec::with_capacity(max_batch_size),
            append_indptr_h: Vec::with_capacity(max_batch_size + 1),
            appended_page_scratch: Vec::with_capacity(max_batch_size),
            indptr_h: Vec::with_capacity(max_batch_size + 1),
            prev_slot_indices: Vec::with_capacity(max_batch_size),
            prev_slot_epochs: Vec::with_capacity(max_batch_size),
            qo_indptr_h: vec![0i32; max_batch_size + 1],
            last_token_indices: ctx
                .stream
                .alloc_zeros(max_batch_size)
                .map_err(|e| anyhow::anyhow!("Alloc last_token_indices failed: {e}"))?,
            last_token_scratch: Vec::with_capacity(max_batch_size),
            total_tokens: 0,
            plan_batch_size: 0,
            plan_dirty: true,
            fast_page16_metadata: decode_metadata_fast_page16_enabled(),
        })
    }

    /// Upload metadata (positions, indptr, last_page_len, KV indices) to GPU.
    ///
    /// Returns whether the kv_indices GPU buffer was reallocated and which
    /// metadata path was used.
    pub fn update(
        &mut self,
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        slot_indices: &[usize],
    ) -> Result<(bool, DecodeMetaUpdate)> {
        let slot_epochs: Vec<u64> = slot_indices
            .iter()
            .map(|&slot| pool.slot_epoch(slot))
            .collect();
        let mut next_indptr_h = std::mem::take(&mut self.indptr_build_scratch);
        pool.fill_indptr(slot_indices, &mut next_indptr_h);
        let page_size = pool.page_size;
        let same_batch_identity =
            self.prev_slot_indices == slot_indices && self.prev_slot_epochs == slot_epochs;
        let can_incremental_update = page_size == 1
            && same_batch_identity
            && can_append_decode_step(&self.indptr_h, &next_indptr_h)
            && self.indices_scratch.len() == self.total_tokens;
        let reuse_plan = if page_size == 1 {
            same_batch_identity
                && self.plan_batch_size == slot_indices.len()
                && can_reuse_decode_plan(&self.indptr_h, &next_indptr_h)
        } else {
            same_batch_identity
                && self.plan_batch_size == slot_indices.len()
                && self.indptr_h == next_indptr_h
        };
        pool.fill_last_page_lens(slot_indices, &mut self.last_page_lens_scratch);

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
            .memcpy_htod(&self.last_page_lens_scratch, &mut self.kv_last_page_len)
            .map_err(|e| anyhow::anyhow!("H2D last_page_len: {e}"))?;
        self.qo_indptr_h.clear();
        self.qo_indptr_h.extend(0..=slot_indices.len() as i32);
        ctx.stream
            .memcpy_htod(&self.qo_indptr_h, &mut self.qo_indptr)
            .map_err(|e| anyhow::anyhow!("H2D qo_indptr: {e}"))?;

        let new_total = *self
            .indptr_h
            .last()
            .expect("invariant: indptr_h from build_indptr() always has at least one element")
            as usize;
        let mut reallocated = false;

        self.last_token_scratch.clear();
        self.last_token_scratch
            .extend(pool.build_last_indices(slot_indices));

        let mut mode = DecodeMetaUpdate::Full;
        if self.fast_page16_metadata
            && page_size == 16
            && same_batch_identity
            && self.indices_scratch.len() == self.total_tokens
            && prev_indptr_h.last().copied().unwrap_or_default() as usize == self.total_tokens
        {
            if self.indptr_h == prev_indptr_h
                && page_tables_match_cached(
                    pool,
                    slot_indices,
                    &prev_indptr_h,
                    &self.indices_scratch,
                )
            {
                mode = DecodeMetaUpdate::SamePages;
            } else if can_append_pages_step(&prev_indptr_h, &self.indptr_h)
                && collect_appended_page_indices(
                    pool,
                    slot_indices,
                    &prev_indptr_h,
                    &self.indptr_h,
                    &self.indices_scratch,
                    &mut self.append_indptr_h,
                    &mut self.appended_page_scratch,
                )
            {
                mode = DecodeMetaUpdate::AppendedPages;
            }
        }

        match mode {
            DecodeMetaUpdate::SamePages => {}
            DecodeMetaUpdate::AppendedPages => {
                append_new_page_indices_in_place(
                    &mut self.indices_scratch,
                    &prev_indptr_h,
                    &self.indptr_h,
                    &self.append_indptr_h,
                    &self.appended_page_scratch,
                );
            }
            DecodeMetaUpdate::Full => {
                if can_incremental_update {
                    append_last_token_indices_in_place(
                        &mut self.indices_scratch,
                        &prev_indptr_h,
                        &self.indptr_h,
                        &self.last_token_scratch,
                    );
                } else {
                    self.indices_scratch.clear();
                    self.indices_scratch.extend(flatten_page_indices(
                        slot_indices.iter().map(|&slot| pool.page_indices(slot)),
                    ));
                }
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

        // Upload last-token pool indices (for INT8 quantize-after-write).
        ctx.stream
            .memcpy_htod(&self.last_token_scratch, &mut self.last_token_indices)
            .map_err(|e| anyhow::anyhow!("H2D last_token_indices: {e}"))?;
        match mode {
            DecodeMetaUpdate::SamePages => {}
            DecodeMetaUpdate::AppendedPages if !reallocated => {
                append_new_page_indices_kernel(
                    ctx,
                    &mut self.kv_indices,
                    &mut self.prev_kv_indptr,
                    &self.kv_indptr,
                    &mut self.append_indptr,
                    &mut self.appended_page_indices,
                    &prev_indptr_h,
                    &self.append_indptr_h,
                    &self.appended_page_scratch,
                    slot_indices.len(),
                )?;
            }
            _ => {
                let use_gpu_incremental_update =
                    can_incremental_update && new_total <= self.max_total_pages;
                if use_gpu_incremental_update && !reallocated {
                    let (indices_ptr, _gidx) = self.kv_indices.device_ptr_mut(&ctx.stream);
                    let (indptr_ptr, _gindptr) = self.kv_indptr.device_ptr(&ctx.stream);
                    let (last_ptr, _glast) = self.last_token_indices.device_ptr(&ctx.stream);
                    unsafe {
                        ffi::paged_kv_append_last_token_indices_cuda(
                            indices_ptr as *mut i32,
                            indptr_ptr as *const i32,
                            last_ptr as *const i32,
                            slot_indices.len() as i32,
                            ctx.stream.cu_stream(),
                        )
                        .result()?;
                    }
                } else {
                    ctx.stream
                        .memcpy_htod(&self.indices_scratch, &mut self.kv_indices)
                        .map_err(|e| anyhow::anyhow!("H2D indices: {e}"))?;
                }
            }
        }
        self.total_tokens = new_total;

        // Re-plan when the batch size changed, lengths did not follow the
        // steady-state decode progression, or kv_indices had to move.
        if reallocated || !reuse_plan {
            self.plan_dirty = true;
        }
        self.prev_slot_indices.clear();
        self.prev_slot_indices.extend_from_slice(slot_indices);
        self.prev_slot_epochs.clear();
        self.prev_slot_epochs.extend_from_slice(&slot_epochs);
        self.indptr_build_scratch = prev_indptr_h;

        Ok((reallocated, mode))
    }

    /// Upload metadata for one sparse-KV decode row.
    ///
    /// `sparse_page_indices` must be in logical attention order. The final
    /// page length is taken from the real slot tail only when the sparse view
    /// ends with the slot's current tail page; otherwise all selected pages are
    /// treated as sealed full pages.
    pub fn update_sparse_single(
        &mut self,
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        slot: usize,
        sparse_page_indices: &[u32],
    ) -> Result<bool> {
        ensure!(
            !sparse_page_indices.is_empty(),
            "sparse decode metadata requires at least one page"
        );
        ensure!(
            sparse_page_indices.len() <= self.max_total_pages,
            "sparse decode metadata exceeded capacity: pages={} capacity={}",
            sparse_page_indices.len(),
            self.max_total_pages
        );

        let seq_len = pool.seq_len(slot);
        ensure!(seq_len > 0, "sparse decode slot {slot} has empty KV");

        self.positions_scratch.clear();
        self.positions_scratch.push((seq_len - 1) as i32);
        ctx.stream
            .memcpy_htod(&self.positions_scratch, &mut self.positions)
            .map_err(|e| anyhow::anyhow!("H2D sparse positions: {e}"))?;

        self.indptr_h.clear();
        self.indptr_h.push(0);
        self.indptr_h.push(sparse_page_indices.len() as i32);
        ctx.stream
            .memcpy_htod(&self.indptr_h, &mut self.kv_indptr)
            .map_err(|e| anyhow::anyhow!("H2D sparse indptr: {e}"))?;

        let slot_pages = pool.page_indices(slot);
        let tail_page = slot_pages
            .last()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("sparse decode slot {slot} has no pages"))?;
        let sparse_last_page_len = if sparse_page_indices.last().copied() == Some(tail_page) {
            pool.build_last_page_lens(&[slot])[0]
        } else {
            pool.page_size as i32
        };
        let last_page_lens_h = [sparse_last_page_len];
        ctx.stream
            .memcpy_htod(&last_page_lens_h, &mut self.kv_last_page_len)
            .map_err(|e| anyhow::anyhow!("H2D sparse last_page_len: {e}"))?;

        self.qo_indptr_h.clear();
        self.qo_indptr_h.extend([0, 1]);
        ctx.stream
            .memcpy_htod(&self.qo_indptr_h, &mut self.qo_indptr)
            .map_err(|e| anyhow::anyhow!("H2D sparse qo_indptr: {e}"))?;

        self.indices_scratch.clear();
        self.indices_scratch
            .extend(sparse_page_indices.iter().map(|&page| page as i32));
        ctx.stream
            .memcpy_htod(&self.indices_scratch, &mut self.kv_indices)
            .map_err(|e| anyhow::anyhow!("H2D sparse indices: {e}"))?;

        self.last_token_scratch.clear();
        self.last_token_scratch
            .extend(pool.build_last_indices(&[slot]));
        ctx.stream
            .memcpy_htod(&self.last_token_scratch, &mut self.last_token_indices)
            .map_err(|e| anyhow::anyhow!("H2D sparse last_token_indices: {e}"))?;

        self.total_tokens = sparse_page_indices.len();
        self.plan_batch_size = 0;
        self.plan_dirty = true;
        self.prev_slot_indices.clear();
        self.prev_slot_epochs.clear();
        Ok(false)
    }

    /// Upload metadata for a mixed decode + packed prefill batch.
    ///
    /// Layout:
    /// - decode requests: one Q row each
    /// - each prefill request: `prefill_token_counts[i]` consecutive Q rows
    /// - `qo_indptr`: `[0, 1, 2, ..., B, B + len(prefill_0), ...]`
    pub fn update_mixed_batch(
        &mut self,
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        decode_slot_indices: &[usize],
        prefill_slot_indices: &[usize],
        prefill_start_positions: &[usize],
        prefill_token_counts: &[usize],
    ) -> Result<bool> {
        ensure!(
            prefill_slot_indices.len() == prefill_start_positions.len(),
            "mixed metadata: prefill slot/start_pos len mismatch ({} vs {})",
            prefill_slot_indices.len(),
            prefill_start_positions.len()
        );
        ensure!(
            prefill_slot_indices.len() == prefill_token_counts.len(),
            "mixed metadata: prefill slot/token-count len mismatch ({} vs {})",
            prefill_slot_indices.len(),
            prefill_token_counts.len()
        );

        let total_prefill_rows: usize = prefill_token_counts.iter().sum();
        let request_count = decode_slot_indices.len() + prefill_slot_indices.len();
        let total_qo_rows = decode_slot_indices.len() + total_prefill_rows;
        if request_count == 0 || total_qo_rows == 0 {
            return Ok(false);
        }

        let mut all_slots = Vec::with_capacity(request_count);
        all_slots.extend_from_slice(decode_slot_indices);
        all_slots.extend_from_slice(prefill_slot_indices);

        self.positions_scratch.clear();
        self.positions_scratch.extend(
            decode_slot_indices
                .iter()
                .map(|&slot| (pool.seq_len(slot) - 1) as i32),
        );
        for (&start_pos, &token_count) in prefill_start_positions.iter().zip(prefill_token_counts) {
            ensure!(
                token_count > 0,
                "mixed metadata: prefill request at start_pos {} must not be empty",
                start_pos
            );
            self.positions_scratch
                .extend((start_pos..start_pos + token_count).map(|pos| pos as i32));
        }

        self.indptr_h = pool.build_indptr(&all_slots);

        let mut last_page_lens_h = pool.build_last_page_lens(decode_slot_indices);
        last_page_lens_h.extend(pool.build_last_page_lens(prefill_slot_indices));

        self.qo_indptr_h.clear();
        self.qo_indptr_h.push(0);
        let mut qo_rows = 0usize;
        for _ in decode_slot_indices {
            qo_rows += 1;
            self.qo_indptr_h.push(qo_rows as i32);
        }
        for &token_count in prefill_token_counts {
            qo_rows += token_count;
            self.qo_indptr_h.push(qo_rows as i32);
        }
        ensure!(
            qo_rows == total_qo_rows,
            "mixed metadata: qo_indptr terminal {} does not match total_qo_rows {}",
            qo_rows,
            total_qo_rows
        );

        self.indices_scratch.clear();
        self.indices_scratch.extend(flatten_page_indices(
            all_slots.iter().map(|&slot| pool.page_indices(slot)),
        ));

        let mut reallocated = false;
        if self.indices_scratch.len() > self.max_total_pages {
            self.kv_indices = ctx
                .stream
                .alloc_zeros(self.indices_scratch.len())
                .map_err(|e| anyhow::anyhow!("Realloc mixed kv_indices: {e}"))?;
            self.max_total_pages = self.indices_scratch.len();
            reallocated = true;
        }

        ctx.stream
            .memcpy_htod(&self.positions_scratch, &mut self.positions)
            .map_err(|e| anyhow::anyhow!("H2D mixed positions: {e}"))?;
        ctx.stream
            .memcpy_htod(&self.indptr_h, &mut self.kv_indptr)
            .map_err(|e| anyhow::anyhow!("H2D mixed indptr: {e}"))?;
        ctx.stream
            .memcpy_htod(&last_page_lens_h, &mut self.kv_last_page_len)
            .map_err(|e| anyhow::anyhow!("H2D mixed last_page_len: {e}"))?;
        ctx.stream
            .memcpy_htod(&self.qo_indptr_h, &mut self.qo_indptr)
            .map_err(|e| anyhow::anyhow!("H2D mixed qo_indptr: {e}"))?;
        ctx.stream
            .memcpy_htod(&self.indices_scratch, &mut self.kv_indices)
            .map_err(|e| anyhow::anyhow!("H2D mixed indices: {e}"))?;

        self.total_tokens = *self
            .indptr_h
            .last()
            .expect("mixed indptr must contain the terminal entry")
            as usize;
        self.plan_dirty = true;
        self.plan_batch_size = 0;
        self.prev_slot_indices.clear();
        self.prev_slot_epochs.clear();
        self.last_token_scratch.clear();

        Ok(reallocated)
    }

    /// TileLang attention is plan-less. This method is retained as a no-op
    /// while shared decode orchestration still calls a backend planning hook.
    #[allow(clippy::too_many_arguments)]
    pub fn plan(
        &mut self,
        ctx: &DeviceContext,
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        page_size: usize,
        head_dim: usize,
    ) -> Result<()> {
        let _ = (ctx, num_heads, num_kv_heads, page_size, head_dim);
        self.plan_batch_size = batch_size;
        self.plan_dirty = false;
        Ok(())
    }

    /// TileLang HD256 decode is plan-less.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_hd256(
        &mut self,
        ctx: &DeviceContext,
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        page_size: usize,
        head_dim: usize,
    ) -> Result<()> {
        let _ = (ctx, num_heads, num_kv_heads, page_size, head_dim);
        self.plan_batch_size = batch_size;
        self.plan_dirty = false;
        Ok(())
    }

    /// TileLang TC decode is plan-less.
    #[allow(clippy::too_many_arguments)]
    pub fn tc_plan(
        &mut self,
        ctx: &DeviceContext,
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        page_size: usize,
        head_dim: usize,
    ) -> Result<()> {
        let _ = (ctx, num_heads, num_kv_heads, page_size, head_dim);
        self.plan_batch_size = batch_size;
        self.plan_dirty = false;
        Ok(())
    }

    /// Total tokens across all requests from the last `update()` call.
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }
}

fn flatten_page_indices<'a>(page_groups: impl IntoIterator<Item = &'a [u32]>) -> Vec<i32> {
    let mut flat = Vec::new();
    for group in page_groups {
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

fn can_append_pages_step(prev_indptr: &[i32], next_indptr: &[i32]) -> bool {
    if prev_indptr.len() != next_indptr.len() || prev_indptr.len() < 2 {
        return false;
    }

    let mut appended_any = false;
    for (prev, next) in prev_indptr.windows(2).zip(next_indptr.windows(2)) {
        let prev_len = prev[1] - prev[0];
        let next_len = next[1] - next[0];
        if next_len < prev_len || next_len > prev_len + 1 {
            return false;
        }
        appended_any |= next_len > prev_len;
    }
    appended_any
}

fn page_tables_match_cached(
    pool: &PagedKVPool,
    slots: &[usize],
    indptr: &[i32],
    cached_indices: &[i32],
) -> bool {
    if indptr.len() != slots.len() + 1 {
        return false;
    }
    for (i, &slot) in slots.iter().enumerate() {
        let start = indptr[i] as usize;
        let end = indptr[i + 1] as usize;
        let pages = pool.page_indices(slot);
        if end < start || pages.len() != end - start || cached_indices.len() < end {
            return false;
        }
        if pages
            .iter()
            .zip(&cached_indices[start..end])
            .any(|(&page, &cached)| page as i32 != cached)
        {
            return false;
        }
    }
    true
}

fn collect_appended_page_indices(
    pool: &PagedKVPool,
    slots: &[usize],
    prev_indptr: &[i32],
    next_indptr: &[i32],
    cached_indices: &[i32],
    append_indptr: &mut Vec<i32>,
    appended_pages: &mut Vec<i32>,
) -> bool {
    if prev_indptr.len() != slots.len() + 1 || next_indptr.len() != slots.len() + 1 {
        return false;
    }

    append_indptr.clear();
    append_indptr.push(0);
    appended_pages.clear();
    for (i, &slot) in slots.iter().enumerate() {
        let old_start = prev_indptr[i] as usize;
        let old_end = prev_indptr[i + 1] as usize;
        let new_start = next_indptr[i] as usize;
        let new_end = next_indptr[i + 1] as usize;
        if old_end < old_start || new_end < new_start || cached_indices.len() < old_end {
            return false;
        }

        let old_len = old_end - old_start;
        let new_len = new_end - new_start;
        if new_len < old_len || new_len > old_len + 1 {
            return false;
        }

        let pages = pool.page_indices(slot);
        if pages.len() != new_len {
            return false;
        }
        if pages[..old_len]
            .iter()
            .zip(&cached_indices[old_start..old_end])
            .any(|(&page, &cached)| page as i32 != cached)
        {
            return false;
        }
        if new_len > old_len {
            appended_pages.push(pages[new_len - 1] as i32);
        }
        append_indptr.push(appended_pages.len() as i32);
    }
    !appended_pages.is_empty()
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

fn append_new_page_indices_in_place(
    indices_scratch: &mut Vec<i32>,
    prev_indptr: &[i32],
    next_indptr: &[i32],
    append_indptr: &[i32],
    appended_page_indices: &[i32],
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

    for i in (0..append_indptr.len().saturating_sub(1)).rev() {
        let old_start = prev_indptr[i] as usize;
        let old_end = prev_indptr[i + 1] as usize;
        let new_start = next_indptr[i] as usize;
        let new_end = next_indptr[i + 1] as usize;
        let append_start = append_indptr[i] as usize;
        let append_end = append_indptr[i + 1] as usize;
        let append_count = append_end - append_start;

        indices_scratch.copy_within(old_start..old_end, new_start);
        if append_count > 0 {
            let dst_start = new_end - append_count;
            indices_scratch[dst_start..new_end]
                .copy_from_slice(&appended_page_indices[append_start..append_end]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn append_new_page_indices_kernel(
    ctx: &DeviceContext,
    kv_indices: &mut CudaSlice<i32>,
    prev_kv_indptr: &mut CudaSlice<i32>,
    next_kv_indptr: &CudaSlice<i32>,
    append_indptr: &mut CudaSlice<i32>,
    appended_page_indices: &mut CudaSlice<i32>,
    prev_indptr_h: &[i32],
    append_indptr_h: &[i32],
    appended_pages_h: &[i32],
    batch_size: usize,
) -> Result<()> {
    ctx.stream
        .memcpy_htod(prev_indptr_h, prev_kv_indptr)
        .map_err(|e| anyhow::anyhow!("H2D prev_kv_indptr: {e}"))?;
    ctx.stream
        .memcpy_htod(append_indptr_h, append_indptr)
        .map_err(|e| anyhow::anyhow!("H2D append_indptr: {e}"))?;
    if !appended_pages_h.is_empty() {
        let mut appended_view = appended_page_indices.slice_mut(..appended_pages_h.len());
        ctx.stream
            .memcpy_htod(appended_pages_h, &mut appended_view)
            .map_err(|e| anyhow::anyhow!("H2D appended_page_indices: {e}"))?;
    }

    let (indices_ptr, _gidx) = kv_indices.device_ptr_mut(&ctx.stream);
    let (prev_indptr_ptr, _gprev) = prev_kv_indptr.device_ptr(&ctx.stream);
    let (next_indptr_ptr, _gnext) = next_kv_indptr.device_ptr(&ctx.stream);
    let (append_indptr_ptr, _gappend) = append_indptr.device_ptr(&ctx.stream);
    let (appended_ptr, _gappended) = appended_page_indices.device_ptr(&ctx.stream);
    unsafe {
        ffi::paged_kv_append_new_page_indices_cuda(
            indices_ptr as *mut i32,
            prev_indptr_ptr as *const i32,
            next_indptr_ptr as *const i32,
            append_indptr_ptr as *const i32,
            appended_ptr as *const i32,
            batch_size as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

fn can_reuse_decode_plan(prev_indptr: &[i32], next_indptr: &[i32]) -> bool {
    can_append_decode_step(prev_indptr, next_indptr)
}

#[cfg(test)]
mod tests {
    use super::{
        append_last_token_indices_in_place, append_new_page_indices_in_place,
        can_append_decode_step, can_append_pages_step, can_reuse_decode_plan, flatten_page_indices,
    };

    #[test]
    fn flatten_page_indices_preserves_per_slot_segments() {
        let flat =
            flatten_page_indices([&[10_u32, 11, 12][..], &[20_u32, 21, 22][..], &[30_u32][..]]);
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

    #[test]
    fn append_pages_step_allows_sparse_page_growth() {
        assert!(can_append_pages_step(&[0, 2, 5, 6], &[0, 2, 6, 7]));
        assert!(can_append_pages_step(&[0, 2, 5, 6], &[0, 3, 6, 7]));
        assert!(!can_append_pages_step(&[0, 2, 5, 6], &[0, 2, 5, 6]));
        assert!(!can_append_pages_step(&[0, 2, 5, 6], &[0, 4, 7, 8]));
    }

    #[test]
    fn appended_page_indices_shift_suffixes_in_place() {
        let mut scratch = vec![10, 11, 20, 21, 22, 30];
        append_new_page_indices_in_place(
            &mut scratch,
            &[0, 2, 5, 6],
            &[0, 2, 6, 8],
            &[0, 0, 1, 2],
            &[23, 31],
        );
        assert_eq!(scratch, vec![10, 11, 20, 21, 22, 23, 30, 31]);
    }
}
