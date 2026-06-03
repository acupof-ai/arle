//! Fixed device buffers for the B=1 captured decode path (CG-2).
//!
//! This is the additive companion to the eager [`crate::model::CudaModel::forward_tokens`]
//! path (which stays the numerically-verified correctness floor — Phase 0 closed
//! 2026-06-03 with exact HF-gold parity, see the R6 CUDA eager parity gate). The
//! eager path allocates the activation scratch (`HiddenStates::zeros`) and the
//! per-step paging metadata (`PageMeta::for_slot`) fresh on every call. A captured
//! CUDA graph cannot contain `cudaMalloc`, so the captured decode path instead
//! READS/WRITES the **fixed** buffers held here: the host overwrites their contents
//! (Stage 1) before each `graph.launch()` (Stage 2), exactly as the design doc
//! `docs/projects/2026-06-03-cuda-graph-design.md` §5 specifies.
//!
//! Scope (per design §9, the recommended first landing): **B=1 decode only**. The
//! batch-size buckets and pad-up logic the design describes for the server case are
//! deferred to R6b; this module sizes everything to a single decode row.
//!
//! # The page-table-length capture key (design §5 vs the TileLang scalar ABI)
//!
//! The clean R6 attention path issues the TileLang paged decode kernel with the
//! page-table length (`meta.num_pages`) as a **scalar launch argument** (see
//! `attention.rs` `run_tilelang_paged`, the `total_pages` arg). A captured graph
//! bakes that scalar in. The KV positions themselves are read from the fixed
//! `kv_indices`/`kv_indptr`/`kv_last_page_len` device buffers (overwritable), but
//! the page-*count* is frozen at capture. Because the decode sequence grows one
//! token per step, `num_pages` increments every `page_size` (16) tokens. So the
//! capture key is `(batch_size, num_pages)`: while the page count is unchanged the
//! graph replays; when a page boundary is crossed the executor recaptures for the
//! new page count. This is the design's "invalidate + recapture on shape change"
//! rule (§5/§8 `reallocated → invalidate_graph_cache`) applied to the page-table
//! length, and it keeps the captured kernel reading exactly the valid page span.

use anyhow::{Result, ensure};
use cuda_kernels::prelude::{DeviceContext, DeviceVec, HiddenStates, PagedKVPool};
use cudarc::driver::CudaSlice;

use crate::decode_graph_key::{DECODE_GRAPH_BATCH, DecodeGraphKey, decode_graph_key_for};
use crate::loader::PageMeta;

/// Fixed device buffers for one captured B=1 decode shape.
///
/// Holds the activation scratch (sized to `seq_len == 1`) and the paging metadata
/// buffers, all allocated once and reused across every replay at the same capture
/// key. The host overwrites the metadata contents each step via [`Self::stage1_write`]
/// before launching the graph; the buffer **addresses** never change, so the captured
/// kernels keep dereferencing valid pointers.
pub(crate) struct DecodeGraphContext {
    // -- activation scratch (was `HiddenStates::zeros` / `DeviceVec::zeros` per call) --
    pub(crate) hidden: HiddenStates,
    pub(crate) normed: HiddenStates,
    pub(crate) q_batch: HiddenStates,
    pub(crate) k_batch: HiddenStates,
    pub(crate) v_batch: HiddenStates,
    pub(crate) attn_output: HiddenStates,
    pub(crate) o_buf: HiddenStates,
    pub(crate) hidden_out: HiddenStates,
    pub(crate) gate_out: HiddenStates,
    pub(crate) up_out: HiddenStates,
    pub(crate) act_out: HiddenStates,
    pub(crate) last_hidden: DeviceVec,
    pub(crate) last_normed: DeviceVec,
    /// LM-head output. The captured graph ends here; sampling runs eager afterward
    /// (design §7 — sampling stays outside the graph).
    pub(crate) logits: DeviceVec,

    // -- fixed token id buffer (was `upload_i32(token_ids)` per call) --
    pub(crate) token_ids: CudaSlice<i32>,

    // -- fixed paging metadata (was `PageMeta::for_slot`, which uploads 7 fresh
    //    `CudaSlice<i32>` per call). This OWNS the fixed metadata buffers; Stage-1
    //    overwrites their CONTENTS in place each step (never reallocates), so the
    //    captured attention kernel keeps dereferencing the same stable addresses.
    //    `kv_indices` is sized to `max_pages` so it survives sequence growth; only
    //    the first `num_pages` entries are valid each step, and `meta.num_pages`
    //    (the captured TileLang `total_pages` scalar) bounds the kernel's walk. --
    pub(crate) meta: PageMeta,
    /// Max page-table entries `meta.kv_indices` can hold (sizing bound, never resized).
    max_pages: usize,

    /// Capture key this context was last written for: `(batch_size, num_pages)`.
    /// `None` until the first Stage-1 write. A change in `num_pages` (page-boundary
    /// crossing) means the captured graph's baked page-table length is stale.
    key: Option<DecodeGraphKey>,
}

impl DecodeGraphContext {
    /// Allocate the fixed buffers once for the model's config and a sequence-length
    /// budget. `max_seq_len` bounds the page-table buffer (`kv_indices`) so it is
    /// never reallocated mid-serve.
    pub(crate) fn new(
        ctx: &DeviceContext,
        hidden_size: usize,
        q_dim: usize,
        kv_dim: usize,
        inter: usize,
        vocab: usize,
        page_size: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        // B=1 decode: every activation row is one token.
        let seq_len = DECODE_GRAPH_BATCH;
        let max_pages = max_seq_len.div_ceil(page_size).max(1);
        Ok(Self {
            hidden: HiddenStates::zeros(ctx, hidden_size, seq_len)?,
            normed: HiddenStates::zeros(ctx, hidden_size, seq_len)?,
            q_batch: HiddenStates::zeros(ctx, q_dim, seq_len)?,
            k_batch: HiddenStates::zeros(ctx, kv_dim, seq_len)?,
            v_batch: HiddenStates::zeros(ctx, kv_dim, seq_len)?,
            attn_output: HiddenStates::zeros(ctx, q_dim, seq_len)?,
            o_buf: HiddenStates::zeros(ctx, hidden_size, seq_len)?,
            hidden_out: HiddenStates::zeros(ctx, hidden_size, seq_len)?,
            gate_out: HiddenStates::zeros(ctx, inter, seq_len)?,
            up_out: HiddenStates::zeros(ctx, inter, seq_len)?,
            act_out: HiddenStates::zeros(ctx, inter, seq_len)?,
            last_hidden: DeviceVec::zeros(ctx, hidden_size)?,
            last_normed: DeviceVec::zeros(ctx, hidden_size)?,
            logits: DeviceVec::zeros(ctx, vocab)?,
            token_ids: alloc_i32(ctx, seq_len)?,
            meta: PageMeta {
                q_indptr: alloc_i32(ctx, 2)?,
                kv_indptr: alloc_i32(ctx, 2)?,
                kv_indices: alloc_i32(ctx, max_pages)?,
                kv_last_page_len: alloc_i32(ctx, 1)?,
                page_table_offsets: alloc_i32(ctx, 1)?,
                start_positions: alloc_i32(ctx, 1)?,
                positions: alloc_i32(ctx, 1)?,
                seq_len,
                num_pages: 0,
            },
            max_pages,
            key: None,
        })
    }

    /// Stage 1: overwrite the fixed metadata buffers with this step's contents.
    ///
    /// Mirrors [`PageMeta::for_slot`] arithmetic but writes into the **existing**
    /// device allocations (in-place H2D) instead of allocating fresh ones. Returns
    /// the `(batch, num_pages)` key the captured graph must match to be replay-valid.
    ///
    /// `token` is this decode step's input token, `kv_seq_len` is the cache length
    /// BEFORE appending it (so the new position is `kv_seq_len`, and the new total
    /// length is `kv_seq_len + 1`). The pool must already have the appended token's
    /// page allocated (the executor calls `alloc_tokens(slot, 1)` before this).
    pub(crate) fn stage1_write(
        &mut self,
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        slot: usize,
        token: u32,
        kv_seq_len: usize,
    ) -> Result<DecodeGraphKey> {
        let total_len = kv_seq_len + 1;
        ensure!(
            pool.seq_len(slot) == total_len,
            "DecodeGraphContext: pool seq_len {} != materialized total_len {} for slot {}",
            pool.seq_len(slot),
            total_len,
            slot
        );
        let num_pages = total_len.div_ceil(pool.page_size);
        ensure!(
            num_pages <= self.max_pages,
            "DecodeGraphContext: num_pages {} exceeds fixed kv_indices budget {} (raise max_seq_len)",
            num_pages,
            self.max_pages
        );
        let pages = pool.page_indices(slot);
        ensure!(
            pages.len() >= num_pages,
            "DecodeGraphContext: slot {} has {} pages, expected at least {}",
            slot,
            pages.len(),
            num_pages
        );
        let last_page_len = total_len % pool.page_size;
        let last_page_len = if last_page_len == 0 {
            pool.page_size
        } else {
            last_page_len
        };

        // In-place overwrites of the fixed buffers — addresses never change.
        write_i32(ctx, &mut self.token_ids, &[token as i32])?;
        write_i32(
            ctx,
            &mut self.meta.q_indptr,
            &[0, DECODE_GRAPH_BATCH as i32],
        )?;
        write_i32(ctx, &mut self.meta.kv_indptr, &[0, num_pages as i32])?;
        let page_ids: Vec<i32> = pages[..num_pages].iter().map(|&p| p as i32).collect();
        // Write only the valid prefix; trailing entries stay as previously written
        // but are never walked because the kernel's page-table length is num_pages.
        write_i32_prefix(ctx, &mut self.meta.kv_indices, &page_ids)?;
        write_i32(
            ctx,
            &mut self.meta.kv_last_page_len,
            &[last_page_len as i32],
        )?;
        write_i32(ctx, &mut self.meta.page_table_offsets, &[0])?;
        write_i32(ctx, &mut self.meta.start_positions, &[kv_seq_len as i32])?;
        write_i32(ctx, &mut self.meta.positions, &[(total_len - 1) as i32])?;
        // The captured TileLang kernel reads num_pages as a scalar launch arg, so
        // it is part of the capture key (see module docs).
        self.meta.num_pages = num_pages;
        self.meta.seq_len = DECODE_GRAPH_BATCH;

        let key = decode_graph_key_for(pool.page_size, kv_seq_len);
        debug_assert_eq!(key.num_pages, num_pages);
        self.key = Some(key);
        Ok(key)
    }

    /// The capture key the buffers were last written for, if any.
    pub(crate) fn key(&self) -> Option<DecodeGraphKey> {
        self.key
    }
}

fn alloc_i32(ctx: &DeviceContext, len: usize) -> Result<CudaSlice<i32>> {
    ctx.stream
        .alloc_zeros::<i32>(len)
        .map_err(|e| anyhow::anyhow!("DecodeGraphContext i32 alloc ({len}) failed: {e}"))
}

/// In-place H2D write of a full fixed i32 buffer (length must match exactly).
fn write_i32(ctx: &DeviceContext, dst: &mut CudaSlice<i32>, values: &[i32]) -> Result<()> {
    ensure!(
        dst.len() == values.len(),
        "DecodeGraphContext fixed buffer len {} != write len {}",
        dst.len(),
        values.len()
    );
    ctx.stream
        .memcpy_htod(values, dst)
        .map_err(|e| anyhow::anyhow!("DecodeGraphContext H2D write failed: {e}"))
}

/// In-place H2D write of a prefix of a fixed i32 buffer (e.g. the valid page span
/// of the max-sized `kv_indices`). Trailing entries are left untouched.
fn write_i32_prefix(ctx: &DeviceContext, dst: &mut CudaSlice<i32>, values: &[i32]) -> Result<()> {
    ensure!(
        values.len() <= dst.len(),
        "DecodeGraphContext prefix write len {} exceeds buffer len {}",
        values.len(),
        dst.len()
    );
    if values.is_empty() {
        return Ok(());
    }
    let mut view = dst.slice_mut(0..values.len());
    ctx.stream
        .memcpy_htod(values, &mut view)
        .map_err(|e| anyhow::anyhow!("DecodeGraphContext kv_indices prefix H2D write failed: {e}"))
}
