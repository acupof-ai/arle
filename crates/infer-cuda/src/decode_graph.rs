//! Fixed device buffers for the B=1 captured decode path.
//!
//! A captured CUDA graph cannot contain `cudaMalloc`, so instead of the eager
//! path's per-call `HiddenStates::zeros` / `PageMeta::for_slot`, the captured
//! decode reads/writes the FIXED buffers here: the host overwrites their contents
//! (Stage 1) before each `graph.launch()` (Stage 2). B=1 decode only; batch
//! buckets are deferred.
//!
//! Capture key = `(batch_size, num_pages)`: the TileLang decode kernel takes the
//! page-table length (`meta.num_pages`) as a scalar launch arg, which the graph
//! bakes in. KV positions live in overwritable buffers, but the page count is
//! frozen at capture, so crossing a page boundary (every `page_size` tokens)
//! recaptures.

use anyhow::{Result, ensure};
use cuda_kernels::prelude::{DeviceContext, DeviceVec, HiddenStates, PagedKVPool};
use cudarc::driver::CudaSlice;

use crate::decode_graph_key::{DECODE_GRAPH_BATCH, DecodeGraphKey, decode_graph_key_for};
use crate::loader::PageMeta;

/// Fixed device buffers for one captured B=1 decode shape: activation scratch
/// (sized to `seq_len == 1`) + paging metadata, allocated once and reused across
/// replays. The host overwrites the metadata contents each step via
/// [`Self::stage1_write`]; buffer addresses never change.
pub(crate) struct DecodeGraphContext {
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
    /// LM-head output. The captured graph ends here; sampling runs eager after.
    pub(crate) logits: DeviceVec,

    pub(crate) token_ids: CudaSlice<i32>,

    /// Fixed paging metadata; Stage-1 overwrites contents in place each step.
    /// `kv_indices` is sized to `max_pages` so it survives sequence growth; only
    /// the first `num_pages` entries are valid and bound the kernel's walk.
    pub(crate) meta: PageMeta,
    /// Max page-table entries `meta.kv_indices` holds (never resized).
    max_pages: usize,

    /// Capture key last written for; `None` until the first Stage-1 write. A change
    /// in `num_pages` means the captured page-table length is stale.
    key: Option<DecodeGraphKey>,
}

impl DecodeGraphContext {
    /// Allocate the fixed buffers once. `max_seq_len` bounds `kv_indices` so it is
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
                kv_lens_dev: alloc_i32(ctx, 1)?,
                page_table_rect: alloc_i32(ctx, max_pages)?,
                page_table_stride: 0,
                kv_last_page_len: alloc_i32(ctx, 1)?,
                page_table_offsets: alloc_i32(ctx, 1)?,
                start_positions: alloc_i32(ctx, 1)?,
                positions: alloc_i32(ctx, 1)?,
                q_offsets: vec![0, seq_len],
                page_offsets: vec![0, 0],
                kv_lens: vec![0],
                seq_len,
                total_q: seq_len,
                num_pages: 0,
                // Captured graph is B=1 only (batch fixed at DECODE_GRAPH_BATCH).
                batch: 1,
                // Quant-KV metadata: the decode graph never runs under quant
                // formats (warmup hard-disables the graph for non-BF16 pools).
                start_pos: 0,
                new_token_rows: None,
                prefix_token_rows: None,
                quant_decode_meta: None,
                seqlen_k_capture: None,
                write_kv: 1,
            },
            max_pages,
            key: None,
        })
    }

    /// Stage 1: overwrite the fixed metadata in place (no alloc) with this step's
    /// contents; returns the `(batch, num_pages)` replay key. `kv_seq_len` is the
    /// cache length BEFORE appending `token`; the pool must already have the
    /// appended page allocated.
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

        // In-place overwrites; buffer addresses never change.
        write_i32(ctx, &mut self.token_ids, &[token as i32])?;
        write_i32(
            ctx,
            &mut self.meta.q_indptr,
            &[0, DECODE_GRAPH_BATCH as i32],
        )?;
        write_i32(ctx, &mut self.meta.kv_indptr, &[0, num_pages as i32])?;
        let page_ids: Vec<i32> = pages[..num_pages].iter().map(|&p| p as i32).collect();
        // Only the valid prefix is written; the kernel walks just num_pages.
        write_i32_prefix(ctx, &mut self.meta.kv_indices, &page_ids)?;
        // B=1, so the rectangular table is the same page list at stride
        // num_pages; the buffer addresses are baked into the capture.
        write_i32_prefix(ctx, &mut self.meta.page_table_rect, &page_ids)?;
        write_i32(ctx, &mut self.meta.kv_lens_dev, &[total_len as i32])?;
        self.meta.page_table_stride = num_pages;
        write_i32(
            ctx,
            &mut self.meta.kv_last_page_len,
            &[last_page_len as i32],
        )?;
        write_i32(ctx, &mut self.meta.page_table_offsets, &[0])?;
        write_i32(ctx, &mut self.meta.start_positions, &[kv_seq_len as i32])?;
        write_i32(ctx, &mut self.meta.positions, &[(total_len - 1) as i32])?;
        self.meta.num_pages = num_pages;
        self.meta.page_offsets[1] = num_pages;
        self.meta.kv_lens[0] = total_len;
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
