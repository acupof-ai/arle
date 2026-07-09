use super::*;
use crate::kv_tier::CudaKvTierStore;
use std::collections::HashMap;
pub(super) const DSV4_PREFILL_QUERY_CHUNK: usize = 4096;
pub(crate) struct Dsv4CompressorState {
    pub(super) pending_kv: CudaSlice<half::bf16>,
    pub(super) pending_score: CudaSlice<half::bf16>,
    pub(super) prev_overlap_kv: CudaSlice<half::bf16>,
    pub(super) prev_overlap_score: CudaSlice<half::bf16>,
    pub(super) compressed: HiddenStates,
    pub(super) compressed_capacity: usize,
    pub(super) ring_rows: usize,
}

/// Indexer key-staging ring depth (rows): 2 × the max per-forward query chunk.
pub(super) const DSV4_INDEXER_STAGING_RING_ROWS: usize = 2 * DSV4_PREFILL_QUERY_CHUNK;

impl Dsv4CompressorState {
    pub(super) fn new(
        ctx: &DeviceContext,
        head_dim: usize,
        ratio: usize,
        overlap: bool,
        max_seq_len: usize,
        staging_ring: bool,
    ) -> Result<Self> {
        let width = if overlap { 2 * head_dim } else { head_dim };
        let compressed_capacity = max_seq_len.div_ceil(ratio).max(1);
        let ring_rows = if staging_ring {
            DSV4_INDEXER_STAGING_RING_ROWS.min(compressed_capacity)
        } else {
            compressed_capacity
        };
        let compressed_rows = ring_rows;
        Ok(Self {
            pending_kv: ctx
                .stream
                .alloc_zeros::<half::bf16>(ratio * width)
                .map_err(|e| anyhow::anyhow!("DSv4 compressor pending kv alloc failed: {e}"))?,
            pending_score: ctx
                .stream
                .alloc_zeros::<half::bf16>(ratio * width)
                .map_err(|e| anyhow::anyhow!("DSv4 compressor pending score alloc failed: {e}"))?,
            prev_overlap_kv: ctx
                .stream
                .alloc_zeros::<half::bf16>(ratio * head_dim)
                .map_err(|e| anyhow::anyhow!("DSv4 compressor prev kv alloc failed: {e}"))?,
            prev_overlap_score: ctx
                .stream
                .alloc_zeros::<half::bf16>(ratio * head_dim)
                .map_err(|e| anyhow::anyhow!("DSv4 compressor prev score alloc failed: {e}"))?,
            compressed: HiddenStates::zeros(ctx, head_dim, compressed_rows)?,
            compressed_capacity,
            ring_rows,
        })
    }

    #[inline]
    pub(crate) fn compressed_capacity(&self) -> usize {
        self.compressed_capacity
    }

    pub(super) fn reset(&mut self, ctx: &DeviceContext) -> Result<()> {
        ctx.stream
            .memset_zeros(&mut self.pending_kv)
            .map_err(|e| anyhow::anyhow!("DSv4 compressor pending kv reset failed: {e}"))?;
        ctx.stream
            .memset_zeros(&mut self.pending_score)
            .map_err(|e| anyhow::anyhow!("DSv4 compressor pending score reset failed: {e}"))?;
        ctx.stream
            .memset_zeros(&mut self.prev_overlap_kv)
            .map_err(|e| anyhow::anyhow!("DSv4 compressor prev kv reset failed: {e}"))?;
        ctx.stream
            .memset_zeros(&mut self.prev_overlap_score)
            .map_err(|e| anyhow::anyhow!("DSv4 compressor prev score reset failed: {e}"))?;
        ctx.stream
            .memset_zeros(&mut self.compressed.data)
            .map_err(|e| anyhow::anyhow!("DSv4 compressor compressed reset failed: {e}"))?;
        self.compressed.seq_len = 0;
        Ok(())
    }

    /// Raw mutable device pointers (as `u64`) for gathering into the
    /// batched-compressor-update pointer arrays: `(pending_kv, pending_score,
    /// prev_overlap_kv, prev_overlap_score, compressed)`. `pending_kv`/
    /// `pending_score`/`compressed` are always THIS ROW's own per-slot buffers.
    /// `prev_overlap_kv`/`_score` are per-slot ONLY when `compress_pool` is
    /// `None`; when `Some`, they instead resolve to the SAME shared per-layer
    /// pool base for every row — the kernel addresses the actual page via
    /// `compressed_base`/`block`, not a per-row offset (see
    /// `Dsv4CompressStatePool`). Resolved on `ctx.stream`; the returned `u64`s
    /// stay valid because the buffers are not reallocated for the rest of the
    /// forward (same single-stream discipline the per-row path uses). The
    /// cudarc borrow guards are dropped before returning so the caller can
    /// re-borrow `state`/`compress_pool` later.
    pub(crate) fn batched_update_ptrs(
        &mut self,
        ctx: &DeviceContext,
        compress_pool: Option<&mut Dsv4CompressStatePool>,
    ) -> (u64, u64, u64, u64, u64) {
        let (pkv, g0) = self.pending_kv.device_ptr_mut(&ctx.stream);
        let (psc, g1) = self.pending_score.device_ptr_mut(&ctx.stream);
        let (prkv, prsc) = match compress_pool {
            Some(pool) => pool.base_ptrs_mut(ctx),
            None => {
                let (k, g2) = self.prev_overlap_kv.device_ptr_mut(&ctx.stream);
                let (s, g3) = self.prev_overlap_score.device_ptr_mut(&ctx.stream);
                drop(g2);
                drop(g3);
                (k, s)
            }
        };
        let (comp, g4) = self.compressed.data.device_ptr_mut(&ctx.stream);
        drop(g0);
        drop(g1);
        drop(g4);
        (pkv, psc, prkv, prsc, comp)
    }

    /// Exact requested device bytes owned by this compressor/indexer state:
    /// Σ over the four bf16 partial-row buffers + the `compressed` HiddenStates.
    #[allow(dead_code)]
    pub(crate) fn device_bytes(&self) -> usize {
        let bf16 = std::mem::size_of::<half::bf16>();
        self.pending_kv.len() * bf16
            + self.pending_score.len() * bf16
            + self.prev_overlap_kv.len() * bf16
            + self.prev_overlap_score.len() * bf16
            + self.compressed.device_bytes()
    }

    /// STATIC predictor of `device_bytes` from `new`'s dims — MUST mirror `new`
    /// (kept adjacent so drift is visible). Feeds `Dsv4Model::per_slot_device_bytes`
    /// (the KV budget runs before any slot exists, so it cannot measure one).
    pub(crate) fn device_bytes_for(
        head_dim: usize,
        ratio: usize,
        overlap: bool,
        max_seq_len: usize,
        staging_ring: bool,
    ) -> usize {
        let bf16 = std::mem::size_of::<half::bf16>();
        let width = if overlap { 2 * head_dim } else { head_dim };
        let compressed_capacity = max_seq_len.div_ceil(ratio.max(1)).max(1);
        let ring_rows = if staging_ring {
            DSV4_INDEXER_STAGING_RING_ROWS.min(compressed_capacity)
        } else {
            compressed_capacity
        };
        // pending_kv + pending_score (ratio*width each) + prev_overlap_kv +
        // prev_overlap_score (ratio*head_dim each) + compressed[head_dim, ring_rows].
        (2 * ratio * width + 2 * ratio * head_dim + head_dim * ring_rows) * bf16
    }
}

/// Page-addressable, cross-request-shared pool for ONE layer's compressor
/// `prev_overlap_kv`/`_score` carry-state, replacing the per-slot single
/// register in [`Dsv4CompressorState`] for `compress_ratio == 4` layers only
/// (other ratios keep the per-slot register). One page holds exactly one
/// compress-block's `ratio*head_dim` rows; `capacity_blocks` covers every
/// block position across `max_seq_len`, so a page is addressed by ABSOLUTE
/// block index (`start_pos / ratio`), not by which slot wrote it — two
/// different requests at the same prefix position read/write the same page,
/// which is the whole point (SGLang `CompressStatePool` precedent).
///
/// Only 2 of `Dsv4CompressorState`'s buffers move here: `pending_kv`/
/// `pending_score` stay per-slot (provably zero live rows at every
/// block-aligned restore boundary, `dsv4_attention.cu:1115`); `compressed` is
/// already covered by the FlashMLA page-addressable band.
///
/// `resident[b]` tracks whether block `b` has been written. Allocation still
/// sized to `capacity_blocks`, not shrunk: `dsv4_attention.cu` addresses
/// `prev_overlap_kv/score` by absolute block index (host-passed or
/// on-device-recomputed from `start_pos`), so a smaller ring needs a kernel
/// change (out of reach without nvcc — see the plan doc's step 6 note).
pub(crate) struct Dsv4CompressStatePool {
    pub(super) overlap_kv: CudaSlice<half::bf16>,
    pub(super) overlap_score: CudaSlice<half::bf16>,
    ratio: usize,
    head_dim: usize,
    resident: Vec<bool>,
}

impl Dsv4CompressStatePool {
    pub(super) fn new(
        ctx: &DeviceContext,
        head_dim: usize,
        ratio: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        let capacity_blocks = max_seq_len.div_ceil(ratio.max(1)).max(1);
        let elems = capacity_blocks * ratio * head_dim;
        Ok(Self {
            overlap_kv: ctx
                .stream
                .alloc_zeros::<half::bf16>(elems)
                .map_err(|e| anyhow!("DSv4 compress-state pool overlap kv alloc failed: {e}"))?,
            overlap_score: ctx
                .stream
                .alloc_zeros::<half::bf16>(elems)
                .map_err(|e| anyhow!("DSv4 compress-state pool overlap score alloc failed: {e}"))?,
            ratio,
            head_dim,
            resident: vec![false; capacity_blocks],
        })
    }

    /// Elements-per-page stride for the kernel's `overlap_page_stride` param
    /// (`ratio * head_dim`) — passed whenever this pool backs a call, `0`
    /// otherwise (collapses the kernel's indexing to the old single-register
    /// form; see `dsv4_compressor_update_body` in `dsv4_attention.cu`).
    pub(crate) fn page_stride(head_dim: usize, ratio: usize) -> usize {
        ratio * head_dim
    }

    /// Base device pointers `(overlap_kv, overlap_score)` — IDENTICAL for every
    /// row/slot sharing this layer's pool (the kernel resolves the actual page
    /// via `compressed_base`/`block`, not a per-row offset). Guards dropped
    /// immediately; the raw pointers stay valid under the same single-stream,
    /// no-realloc discipline `Dsv4CompressorState::batched_update_ptrs` uses.
    pub(crate) fn base_ptrs_mut(&mut self, ctx: &DeviceContext) -> (u64, u64) {
        let (kv, g0) = self.overlap_kv.device_ptr_mut(&ctx.stream);
        let (score, g1) = self.overlap_score.device_ptr_mut(&ctx.stream);
        drop(g0);
        drop(g1);
        (kv, score)
    }

    /// `false` covers both "never written" and "demoted to the tier store".
    pub(crate) fn is_resident(&self, block_index: usize) -> bool {
        self.resident.get(block_index).copied().unwrap_or(false)
    }

    pub(crate) fn mark_written(&mut self, block_index: usize) {
        if let Some(slot) = self.resident.get_mut(block_index) {
            *slot = true;
        }
    }

    pub(crate) fn mark_written_range(&mut self, first: usize, count: usize) {
        for b in first..first.saturating_add(count) {
            self.mark_written(b);
        }
    }

    fn block_range(&self, block_index: usize) -> std::ops::Range<usize> {
        let per_block = self.ratio * self.head_dim;
        let start = block_index * per_block;
        start..start + per_block
    }

    /// Not called yet: no memory win until the ring shrinks (see struct doc).
    #[allow(dead_code)]
    pub(crate) fn demote(
        &mut self,
        ctx: &DeviceContext,
        tier: &mut CudaKvTierStore,
        key: u64,
        block_index: usize,
    ) -> Result<bool> {
        ensure!(
            block_index < self.resident.len(),
            "DSv4 compress-state demote: block {block_index} outside capacity {}",
            self.resident.len()
        );
        if !self.is_resident(block_index) {
            return Ok(false);
        }
        let range = self.block_range(block_index);
        let kv_vals = ctx
            .stream
            .clone_dtoh(&self.overlap_kv.slice(range.clone()))
            .map_err(|e| anyhow!("DSv4 compress-state demote kv dtoh failed: {e}"))?;
        let score_vals = ctx
            .stream
            .clone_dtoh(&self.overlap_score.slice(range))
            .map_err(|e| anyhow!("DSv4 compress-state demote score dtoh failed: {e}"))?;
        let mut payload = Vec::with_capacity((kv_vals.len() + score_vals.len()) * 2);
        for v in &kv_vals {
            payload.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        for v in &score_vals {
            payload.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        if !tier.insert(key, payload) {
            return Ok(false);
        }
        self.resident[block_index] = false;
        Ok(true)
    }

    /// `Ok(false)` = absent from both device and tier (never written) —
    /// caller must fall back, never proceed with a stale/garbage read.
    pub(crate) fn ensure_resident(
        &mut self,
        ctx: &DeviceContext,
        tier: &mut CudaKvTierStore,
        key: u64,
        block_index: usize,
    ) -> Result<bool> {
        ensure!(
            block_index < self.resident.len(),
            "DSv4 compress-state promote: block {block_index} outside capacity {}",
            self.resident.len()
        );
        if self.is_resident(block_index) {
            return Ok(true);
        }
        let payload = match tier.read(key) {
            Ok(payload) => payload,
            Err(_) => return Ok(false),
        };
        let per_block = self.ratio * self.head_dim;
        ensure!(
            payload.len() == per_block * 2 * 2,
            "DSv4 compress-state tier payload size mismatch for block {block_index}: got {} \
             expected {}",
            payload.len(),
            per_block * 2 * 2
        );
        let (kv_bytes, score_bytes) = payload.split_at(per_block * 2);
        let to_bf16 = |bytes: &[u8]| -> Vec<half::bf16> {
            bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        };
        let kv_vals = to_bf16(kv_bytes);
        let score_vals = to_bf16(score_bytes);
        let range = self.block_range(block_index);
        ctx.stream
            .memcpy_htod(&kv_vals, &mut self.overlap_kv.slice_mut(range.clone()))
            .map_err(|e| anyhow!("DSv4 compress-state promote kv htod failed: {e}"))?;
        ctx.stream
            .memcpy_htod(&score_vals, &mut self.overlap_score.slice_mut(range))
            .map_err(|e| anyhow!("DSv4 compress-state promote score htod failed: {e}"))?;
        self.resident[block_index] = true;
        tier.remove(&[key]);
        Ok(true)
    }

    #[allow(dead_code)]
    pub(crate) fn device_bytes(&self) -> usize {
        let bf16 = std::mem::size_of::<half::bf16>();
        (self.overlap_kv.len() + self.overlap_score.len()) * bf16
    }

    /// STATIC predictor of `device_bytes` from `new`'s dims — MUST mirror `new`.
    /// Feeds `Dsv4Model::kv_budget_plan`'s summed-per-layer term.
    pub(crate) fn device_bytes_for(head_dim: usize, ratio: usize, max_seq_len: usize) -> usize {
        let bf16 = std::mem::size_of::<half::bf16>();
        let capacity_blocks = max_seq_len.div_ceil(ratio.max(1)).max(1);
        2 * capacity_blocks * ratio * head_dim * bf16
    }
}

/// Page-addressable, cross-request-shared FULL-HISTORY mirror of this layer's
/// DSA-official indexer key cache (`Dsv4LayerKvLayout::dsa_key_cache`), keyed
/// by absolute compressed row (`start_pos / index_ratio` — CSA: `compress_ratio`;
/// GLM SparseIndexed: `1`). Unlike [`Dsv4CompressStatePool`] this does NOT
/// replace the live per-slot band: `dsa_key_cache`'s slot-keyed physical layout
/// (`slot_idx * num_pages`) is baked into the CUDA-graph batched-decode
/// meta-build kernel (`dsv4_dsa_build_select_meta_cuda`), out of reach without
/// nvcc. Instead this pool is a write-mirrored shadow — every live cache-write
/// also copies its newly-packed rows here (append-only, so a straight D2D copy
/// keyed by the same row range), and a restore copies the matched row range
/// back OUT into the freshly-assigned slot's own band, leaving the slot-keyed
/// live read/write kernels untouched.
pub(crate) struct Dsv4DsaKeyCachePool {
    pub(super) key_cache: CudaSlice<u8>,
    row_bytes: usize,
    resident: Vec<bool>,
}

impl Dsv4DsaKeyCachePool {
    pub(super) fn new(
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        index_ratio: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        let bytes = dsv4_dsa_key_cache_bytes(config, index_ratio, max_seq_len)?;
        let capacity_rows = max_seq_len.div_ceil(index_ratio.max(1)).max(1);
        Ok(Self {
            key_cache: ctx
                .stream
                .alloc_zeros::<u8>(bytes)
                .map_err(|e| anyhow!("DSv4 DSA key-cache pool alloc failed: {e}"))?,
            row_bytes: config.index_head_dim + std::mem::size_of::<f32>(),
            resident: vec![false; capacity_rows],
        })
    }

    pub(crate) fn is_resident(&self, row: usize) -> bool {
        self.resident.get(row).copied().unwrap_or(false)
    }

    pub(crate) fn mark_written_range(&mut self, first: usize, count: usize) {
        for r in first..first.saturating_add(count) {
            if let Some(slot) = self.resident.get_mut(r) {
                *slot = true;
            }
        }
    }

    /// Mirror newly-written rows `[first, first+count)` from a slot's live
    /// `dsa_key_cache` band into this shared pool, right after the slot-keyed
    /// live cache-write kernel — keeps the pool durable across slot reuse
    /// without touching the live read/write kernels.
    pub(crate) fn mirror_from_slot_band(
        &mut self,
        ctx: &DeviceContext,
        slot_dsa_key_cache: &CudaSlice<u8>,
        slot_range: std::ops::Range<usize>,
        first_row: usize,
        row_count: usize,
    ) -> Result<()> {
        if row_count == 0 {
            return Ok(());
        }
        let rel = first_row * self.row_bytes..(first_row + row_count) * self.row_bytes;
        ensure!(
            slot_range.start + rel.end <= slot_range.end && rel.end <= self.key_cache.len(),
            "DSv4 DSA key-cache mirror: rows [{first_row},{}) exceed slot band or pool capacity",
            first_row + row_count
        );
        let src =
            slot_dsa_key_cache.slice(slot_range.start + rel.start..slot_range.start + rel.end);
        ctx.stream
            .memcpy_dtod(&src, &mut self.key_cache.slice_mut(rel.clone()))
            .map_err(|e| anyhow!("DSv4 DSA key-cache mirror D2D failed: {e}"))?;
        self.mark_written_range(first_row, row_count);
        Ok(())
    }

    /// Copy this pool's prefix `[0, row_count)` into a freshly-assigned slot's
    /// own `dsa_key_cache` band. Append-only + written in row order, so a
    /// resident boundary row (`row_count - 1`) implies every earlier row in
    /// the prefix is resident too — callers gate on the boundary alone.
    pub(crate) fn restore_prefix_into_slot_band(
        &self,
        ctx: &DeviceContext,
        slot_dsa_key_cache: &mut CudaSlice<u8>,
        slot_range: std::ops::Range<usize>,
        row_count: usize,
    ) -> Result<()> {
        let bytes = row_count * self.row_bytes;
        ensure!(
            bytes <= self.key_cache.len() && slot_range.start + bytes <= slot_range.end,
            "DSv4 DSA key-cache restore: {row_count} rows ({bytes}B) exceeds pool or slot band"
        );
        let mut dst = slot_dsa_key_cache.slice_mut(slot_range.start..slot_range.start + bytes);
        ctx.stream
            .memcpy_dtod(&self.key_cache.slice(0..bytes), &mut dst)
            .map_err(|e| anyhow!("DSv4 DSA key-cache restore D2D failed: {e}"))
    }

    /// Not called yet — same dormant-until-eviction posture as
    /// `Dsv4CompressStatePool::demote`.
    #[allow(dead_code)]
    pub(crate) fn demote(
        &mut self,
        ctx: &DeviceContext,
        tier: &mut CudaKvTierStore,
        key: u64,
        row: usize,
    ) -> Result<bool> {
        ensure!(
            row < self.resident.len(),
            "DSv4 DSA key-cache demote: row {row} outside capacity {}",
            self.resident.len()
        );
        if !self.is_resident(row) {
            return Ok(false);
        }
        let range = row * self.row_bytes..(row + 1) * self.row_bytes;
        let bytes = ctx
            .stream
            .clone_dtoh(&self.key_cache.slice(range))
            .map_err(|e| anyhow!("DSv4 DSA key-cache demote dtoh failed: {e}"))?;
        if !tier.insert(key, bytes) {
            return Ok(false);
        }
        self.resident[row] = false;
        Ok(true)
    }

    pub(crate) fn ensure_resident(
        &mut self,
        ctx: &DeviceContext,
        tier: &mut CudaKvTierStore,
        key: u64,
        row: usize,
    ) -> Result<bool> {
        ensure!(
            row < self.resident.len(),
            "DSv4 DSA key-cache promote: row {row} outside capacity {}",
            self.resident.len()
        );
        if self.is_resident(row) {
            return Ok(true);
        }
        let payload = match tier.read(key) {
            Ok(payload) => payload.into_owned(),
            Err(_) => return Ok(false),
        };
        ensure!(
            payload.len() == self.row_bytes,
            "DSv4 DSA key-cache tier payload size mismatch for row {row}: got {} expected {}",
            payload.len(),
            self.row_bytes
        );
        let range = row * self.row_bytes..(row + 1) * self.row_bytes;
        ctx.stream
            .memcpy_htod(&payload, &mut self.key_cache.slice_mut(range))
            .map_err(|e| anyhow!("DSv4 DSA key-cache promote htod failed: {e}"))?;
        self.resident[row] = true;
        tier.remove(&[key]);
        Ok(true)
    }

    #[allow(dead_code)]
    pub(crate) fn device_bytes(&self) -> usize {
        self.key_cache.len()
    }

    /// STATIC predictor — one slot-equivalent band, NOT `× num_slots` (this
    /// pool is shared). Feeds `Dsv4Model::kv_budget_plan` as a fixed term.
    pub(crate) fn device_bytes_for(
        config: &DeepSeekV4Config,
        index_ratio: usize,
        max_seq_len: usize,
    ) -> usize {
        dsv4_dsa_key_cache_bytes(config, index_ratio, max_seq_len).unwrap_or(0)
    }
}

/// Page-addressable, cross-request-shared periodic snapshot of this layer's
/// per-slot sliding-window ring (`Dsv4LayerAttentionState::sw_window_cache`,
/// `update_bf16_sw_window` in `attention.rs`). The ring is overwrite-in-place
/// (`pos % sliding_window`), so unlike the compressor/DSA pools it has no
/// natural append-only representation — instead, at every position where
/// `pos % sliding_window == 0`, the ENTIRE live ring is copied into a new
/// snapshot slot keyed by `block_index = pos / sliding_window`. Since
/// `sliding_window` IS the reuse granularity for this buffer, one full-ring
/// copy is exactly one reuse unit — no incremental-copy complexity needed.
/// Every layer has a ring (including the 3 pure-SlidingWindow layers with no
/// compressor), so this pool is built unconditionally whenever
/// `config.sliding_window > 0`.
pub(crate) struct Dsv4SwRingSnapshotPool {
    pub(super) snapshots: CudaSlice<half::bf16>,
    ring_elems: usize,
    resident: Vec<bool>,
}

impl Dsv4SwRingSnapshotPool {
    pub(super) fn new(
        ctx: &DeviceContext,
        head_dim: usize,
        sliding_window: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        let ring_elems = sliding_window * head_dim;
        let capacity_blocks = max_seq_len.div_ceil(sliding_window.max(1)).max(1);
        Ok(Self {
            snapshots: ctx
                .stream
                .alloc_zeros::<half::bf16>(capacity_blocks * ring_elems)
                .map_err(|e| anyhow!("DSv4 SW ring snapshot pool alloc failed: {e}"))?,
            ring_elems,
            resident: vec![false; capacity_blocks],
        })
    }

    pub(crate) fn is_resident(&self, block_index: usize) -> bool {
        self.resident.get(block_index).copied().unwrap_or(false)
    }

    /// Copy the whole live ring into snapshot slot `block_index` — called
    /// right after `update_bf16_sw_window` returns, only when the position it
    /// just advanced to is a `sliding_window` multiple.
    pub(crate) fn snapshot_from_ring(
        &mut self,
        ctx: &DeviceContext,
        ring: &CudaSlice<half::bf16>,
        block_index: usize,
    ) -> Result<()> {
        let range = self.block_range(block_index)?;
        ensure!(
            ring.len() == self.ring_elems,
            "DSv4 SW ring snapshot: live ring len {} != expected {}",
            ring.len(),
            self.ring_elems
        );
        ctx.stream
            .memcpy_dtod(ring, &mut self.snapshots.slice_mut(range))
            .map_err(|e| anyhow!("DSv4 SW ring snapshot D2D failed: {e}"))?;
        self.resident[block_index] = true;
        Ok(())
    }

    /// Restore snapshot `block_index` into a freshly-assigned slot's own ring.
    pub(crate) fn restore_into_ring(
        &self,
        ctx: &DeviceContext,
        ring: &mut CudaSlice<half::bf16>,
        block_index: usize,
    ) -> Result<()> {
        let range = self.block_range(block_index)?;
        ensure!(
            ring.len() == self.ring_elems,
            "DSv4 SW ring restore: live ring len {} != expected {}",
            ring.len(),
            self.ring_elems
        );
        ctx.stream
            .memcpy_dtod(&self.snapshots.slice(range), ring)
            .map_err(|e| anyhow!("DSv4 SW ring restore D2D failed: {e}"))
    }

    fn block_range(&self, block_index: usize) -> Result<std::ops::Range<usize>> {
        ensure!(
            block_index < self.resident.len(),
            "DSv4 SW ring snapshot: block {block_index} outside capacity {}",
            self.resident.len()
        );
        let start = block_index * self.ring_elems;
        Ok(start..start + self.ring_elems)
    }

    /// Not called yet — same dormant-until-eviction posture as
    /// `Dsv4CompressStatePool::demote`.
    #[allow(dead_code)]
    pub(crate) fn demote(
        &mut self,
        ctx: &DeviceContext,
        tier: &mut CudaKvTierStore,
        key: u64,
        block_index: usize,
    ) -> Result<bool> {
        if !self.is_resident(block_index) {
            return Ok(false);
        }
        let range = self.block_range(block_index)?;
        let vals = ctx
            .stream
            .clone_dtoh(&self.snapshots.slice(range))
            .map_err(|e| anyhow!("DSv4 SW ring demote dtoh failed: {e}"))?;
        let mut payload = Vec::with_capacity(vals.len() * 2);
        for v in &vals {
            payload.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        if !tier.insert(key, payload) {
            return Ok(false);
        }
        self.resident[block_index] = false;
        Ok(true)
    }

    pub(crate) fn ensure_resident(
        &mut self,
        ctx: &DeviceContext,
        tier: &mut CudaKvTierStore,
        key: u64,
        block_index: usize,
    ) -> Result<bool> {
        if self.is_resident(block_index) {
            return Ok(true);
        }
        let payload = match tier.read(key) {
            Ok(payload) => payload.into_owned(),
            Err(_) => return Ok(false),
        };
        ensure!(
            payload.len() == self.ring_elems * 2,
            "DSv4 SW ring tier payload size mismatch for block {block_index}: got {} expected {}",
            payload.len(),
            self.ring_elems * 2
        );
        let vals: Vec<half::bf16> = payload
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let range = self.block_range(block_index)?;
        ctx.stream
            .memcpy_htod(&vals, &mut self.snapshots.slice_mut(range))
            .map_err(|e| anyhow!("DSv4 SW ring promote htod failed: {e}"))?;
        self.resident[block_index] = true;
        tier.remove(&[key]);
        Ok(true)
    }

    #[allow(dead_code)]
    pub(crate) fn device_bytes(&self) -> usize {
        self.snapshots.len() * std::mem::size_of::<half::bf16>()
    }

    /// STATIC predictor — feeds `Dsv4Model::kv_budget_plan` as a fixed term.
    pub(crate) fn device_bytes_for(
        head_dim: usize,
        sliding_window: usize,
        max_seq_len: usize,
    ) -> usize {
        let bf16 = std::mem::size_of::<half::bf16>();
        let capacity_blocks = max_seq_len.div_ceil(sliding_window.max(1)).max(1);
        capacity_blocks * sliding_window * head_dim * bf16
    }
}

pub(crate) struct Dsv4KvAdapter {
    pub(super) layers: Vec<Dsv4LayerKvLayout>,
    pub(super) num_slots: usize,
    pub(super) slot_epochs: Vec<Option<u64>>,
    /// One shared official-DSA selector scratch for ALL CSA layers and slots
    /// (issue #67). `None` only when the model has no CSA/DSA indexer layer.
    pub(super) dsa_shared: Option<Dsv4DsaSharedScratch>,
    /// One shared MoE decode-graph scratch for ALL layers and slots (issue #60).
    /// `None` unless the DSv4 decode-graph path is enabled.
    pub(super) moe_decode_shared: Option<crate::moe::Dsv4MoeDecodeScratch>,
    /// One shared compact-FP8 decode-band MoE tail scratch for ALL layers and
    /// slots (launch-bound Step 1). Sized to the band ceiling (128 routes), so
    /// one instance serves every layer/step. `None` only when the model has no
    /// MoE layer; independent of the decode graph (the batched-stream decode
    /// path — the launch-bound target — never captures a graph).
    pub(super) moe_tail_scratch: Option<crate::moe::Dsv4MoeTailScratch>,
    /// One persistent B=1 MLA decode scratch per MODEL1 layer. This is not graph
    /// state; eager no-spec decode reuses it to avoid allocating q/kv/CSA/O-proj
    /// temporaries inside every per-layer MLA dispatch.
    pub(super) mla_decode: Vec<Option<Dsv4MlaDecodeGraphScratch>>,
    /// One shared-expert output for ALL layers and slots. Capacity covers both
    /// B=1 decode and the tiny multi-row MTP verify chunk; callers set
    /// `seq_len` before dispatch.
    pub(super) shared_expert_out: Option<HiddenStates>,
    /// One shared-expert FP8 scratch for ALL layers and slots. Replaces the
    /// per-layer `dsv4_shared_expert` temporary allocations on eager decode and
    /// scheduled MTP verify.
    pub(super) shared_expert_scratch: Option<crate::moe::Dsv4SharedDecodeScratch>,
    /// One shared batched (`b = N`) FlashMLA sparse-decode scratch for ALL
    /// FlashMLA layers and slots (#60). `None` when FlashMLA decode is disabled
    /// at the build/override level (no arena), or when the model has no FlashMLA
    /// layer. Sized for `max_batch = num_slots` rows. Engaged only on the n>1
    /// batched lane when `dsv4_flashmla_decode_batched_enabled()`.
    pub(super) flashmla_batch: Option<Dsv4FlashMlaDecodeBatchScratch>,
    /// One shared SINGLE-ROW (`s_q = 1`) FlashMLA decode SCRATCH for ALL FlashMLA
    /// layers and slots (#85 P3). Hoisted out of the per-(slot,layer)
    /// `Dsv4FlashMlaDecodeState` — its 13 per-forward accumulator/staging buffers
    /// (`o_accum` dominant at ~33.7 MB/layer) carry NO cross-call/cross-slot state
    /// (overwritten before read each layer step, serial on `ctx.stream`), so one
    /// instance sized for the worst-case layer shape serves every (slot, layer).
    /// Gated by the SAME predicate as the per-slot state alloc
    /// (`dsv4_flashmla_decode_alloc_enabled`); `None` ⇒ byte-identical to before.
    pub(super) flashmla_scratch: Option<Dsv4FlashMlaDecodeScratch>,
    /// One shared FP8 prefill DeepGEMM linear (quantize→GEMM) staging scratch for
    /// ALL layers and slots. Hoisted out of the per-slot `Dsv4LayerAttentionState`
    /// (it was the biggest single per-(slot,layer) offender, ~30 MB/layer): the
    /// scratch is M-chunk-bounded and fully overwritten from each chunk's
    /// activations before read, carrying NO cross-call/cross-slot state (see the
    /// SCRATCH verdict in `Dsv4LayerAttentionState::swap_out_image`). Sized by
    /// `config` + `max_seq_len` only (no layer-specific dimensions), so a single
    /// instance is valid for every layer and slot — DSv4 forward runs layers
    /// sequentially and the engine runs one forward at a time, so it is reused,
    /// never aliased concurrently (exactly like `dsa_shared`/`flashmla_batch`/
    /// decode-graph scratch). `None` when native DeepGEMM is unavailable.
    pub(super) prefill_linear: Option<Dsv4PrefillDeepGemmLinearScratch>,
    /// Maps host page IDs (from RadixCache/HostPagedKvPool) to per-layer
    /// FlashMLA physical page IDs. Index `[layer_idx]` gives that layer's page.
    /// Populated lazily in `prepare_kv_batch` when a slot first claims pages;
    /// reused by `mirror_slot_pages` / prefix attach so cross-request sharing
    /// maps the same host page to the same physical FlashMLA pages.
    pub(super) host_to_flashmla: HashMap<u32, Vec<u32>>,
    /// Model `head_dim` — MODEL1=512 (Stage B device-page-table routing,
    /// non-contiguous bands OK) vs V32/GLM=576 (band-base pack addressing,
    /// contiguous identity bands required).
    pub(super) head_dim: usize,
}

/// Index-layer map: the ONE `compressed-row → (slot-logical page, in-page row)`
/// function for a DSv4 FlashMLA band. The SW ring owns pages `[0, sw_blocks)`;
/// the compressed region follows. `page_size` is the FlashMLA MODEL1 block (64).
/// Replaces the inline `sw_blocks + r/64, r%64` scattered across the write paths
/// so drift between sites is structurally impossible (#146: above 2048 the
/// per-path maps diverged and a shared wrong-row write garbled both read lanes).
#[derive(Clone, Copy)]
pub(crate) struct Dsv4BlockMap {
    sw_blocks: usize,
    page_size: usize,
}

impl Dsv4BlockMap {
    pub(crate) fn new(sw_blocks: usize, page_size: usize) -> Self {
        Self {
            sw_blocks,
            page_size,
        }
    }

    /// Compressed row `r` → `(slot-logical page, in-page row)`.
    pub(crate) fn comp_row(&self, r: usize) -> (usize, usize) {
        (self.sw_blocks + r / self.page_size, r % self.page_size)
    }

    /// SW sub-pool block count — the compressed region's page base.
    pub(crate) fn sw_blocks(&self) -> usize {
        self.sw_blocks
    }

    /// FlashMLA MODEL1 page size (64). The ONE page-size source the pack/index
    /// kernel params draw from, so `Dsv4BlockMap` and those params cannot drift.
    pub(crate) fn page_size(&self) -> usize {
        self.page_size
    }
}

pub(crate) struct Dsv4LayerKvLayout {
    /// Shared FP8 MLA latent pool for this layer (#85 P2 Stage A): a
    /// `TokenKVPool` of opaque packed records (`KVFormat::PackedBytes`,
    /// 584 B/token), page = FlashMLA block = 64 tokens, single plane (records
    /// in the K plane, no V/scale buffers). Every slot's band is addressed
    /// ONLY through its block table ([`Self::flashmla_page_table`] /
    /// [`Self::flashmla_pages_byte_range`]) — never by `slot_idx × slot_bytes`
    /// arithmetic. Stage A allocates each slot's pages up-front in slot
    /// order, so tables are contiguous identity runs and the physical layout
    /// is byte-identical to the pre-paging band arena.
    pub(super) flashmla_kv_pool: Option<TokenKVPool>,
    pub(super) dsa_key_cache: Option<CudaSlice<u8>>,
    /// Slot-logical FlashMLA blocks per slot (`sw_blocks + comp_blocks` for
    /// this layer's shape) — every slot's block-table length.
    pub(super) flashmla_slot_pages: usize,
    /// Bytes of one pool page (`page_block_size × packed record bytes`).
    pub(super) flashmla_page_bytes: usize,
    pub(super) dsa_slot_bytes: usize,
    pub(super) num_slots: usize,
    /// Page-addressable, cross-request-shared `prev_overlap_kv`/`_score` pool
    /// for this layer's MAIN compressor. `Some` only for `compress_ratio == 4`
    /// layers with a compressor (`mode.has_compressor()`); `None` for every
    /// other ratio or a SlidingWindow layer with no compressor at all.
    pub(super) compress_state_pool: Option<Dsv4CompressStatePool>,
    /// Same pool species, for the CSA indexer's OWN key compressor
    /// sub-state (`index_head_dim` width, not `head_dim`). `Some` only for
    /// `compress_ratio == 4` layers with an indexer (`mode.has_indexer()`).
    pub(super) indexer_compress_state_pool: Option<Dsv4CompressStatePool>,
    /// Write-mirrored shared shadow of `dsa_key_cache`, keyed by absolute row.
    /// `Some` whenever `dsa_key_cache` itself is (`mode.has_indexer()` +
    /// DSA-official enabled) — every has_indexer layer, not ratio-gated.
    pub(super) dsa_key_cache_pool: Option<Dsv4DsaKeyCachePool>,
    /// Periodic full-ring snapshot pool for this layer's SW ring. `Some` for
    /// EVERY layer whenever `config.sliding_window > 0` (every layer has a
    /// ring, unlike the compressor/indexer pools above).
    pub(super) sw_ring_pool: Option<Dsv4SwRingSnapshotPool>,
}

impl Dsv4LayerKvLayout {
    /// `&mut` accessor for this layer's MAIN compressor compress-state pool
    /// (mirrors `dsa_shared`/`flashmla_batch`-style shared-scratch accessors). `None`
    /// for every layer except compress_ratio==4 with a compressor.
    pub(crate) fn compress_state_pool_mut(&mut self) -> Option<&mut Dsv4CompressStatePool> {
        self.compress_state_pool.as_mut()
    }

    /// `&mut` accessor for this layer's CSA INDEXER compress-state pool.
    /// `None` for every layer except compress_ratio==4 with an indexer.
    pub(crate) fn indexer_compress_state_pool_mut(&mut self) -> Option<&mut Dsv4CompressStatePool> {
        self.indexer_compress_state_pool.as_mut()
    }

    pub(crate) fn compress_state_pool(&self) -> Option<&Dsv4CompressStatePool> {
        self.compress_state_pool.as_ref()
    }

    pub(crate) fn indexer_compress_state_pool(&self) -> Option<&Dsv4CompressStatePool> {
        self.indexer_compress_state_pool.as_ref()
    }

    pub(crate) fn dsa_key_cache_pool_mut(&mut self) -> Option<&mut Dsv4DsaKeyCachePool> {
        self.dsa_key_cache_pool.as_mut()
    }

    pub(crate) fn dsa_key_cache_pool(&self) -> Option<&Dsv4DsaKeyCachePool> {
        self.dsa_key_cache_pool.as_ref()
    }

    pub(crate) fn sw_ring_pool_mut(&mut self) -> Option<&mut Dsv4SwRingSnapshotPool> {
        self.sw_ring_pool.as_mut()
    }

    pub(crate) fn sw_ring_pool(&self) -> Option<&Dsv4SwRingSnapshotPool> {
        self.sw_ring_pool.as_ref()
    }

    /// Copy `dsa_key_cache_pool`'s prefix `[0, row_count)` into `slot`'s own
    /// `dsa_key_cache` band. No-op without a pool (model has no indexer) or
    /// band (DSA-official disabled).
    pub(crate) fn restore_dsa_prefix_into_slot(
        &mut self,
        ctx: &DeviceContext,
        slot: usize,
        row_count: usize,
    ) -> Result<()> {
        let Some(dsa_pool) = self.dsa_key_cache_pool.as_ref() else {
            return Ok(());
        };
        let slot_range = self.dsa_slot_range(slot)?;
        let Some(slot_band) = self.dsa_key_cache.as_mut() else {
            return Ok(());
        };
        dsa_pool.restore_prefix_into_slot_band(ctx, slot_band, slot_range, row_count)
    }

    pub(crate) fn flashmla_total_pages(&self) -> usize {
        self.flashmla_kv_pool
            .as_ref()
            .map_or(0, |pool| pool.max_total_pages)
    }

    pub(crate) fn flashmla_slot_pages(&self) -> usize {
        self.flashmla_slot_pages
    }

    pub(crate) fn flashmla_page_size(&self) -> usize {
        self.flashmla_kv_pool
            .as_ref()
            .map_or(0, |pool| pool.page_size)
    }

    /// Advance the FlashMLA band cursor by `append_len` tokens.
    ///
    /// The band is NOT a sequential KV cache: the sliding-window ring lives at
    /// slot-logical pages `[0, sw_blocks)` and the compressed region at
    /// `[sw_blocks, total_blocks)`, both addressed by FIXED slot-logical block id
    /// (see `flashmla_pack_sw_ring` / `flashmla_pack_compressed_delta` /
    /// `fp8_sw_offsets`). So the readers/pack kernels need ALL `total_blocks`
    /// pages resident from the first token — a `ceil(seq_len/page_size)` draw
    /// would leave the SW ring / comp region unmapped. The band is adopted from
    /// the host slot page table by `prepare_kv_batch`; this method only advances
    /// the logical cursor, so the `seq_len == append_pos` invariant tracks
    /// position, not band size.
    pub(crate) fn flashmla_alloc_append(
        &mut self,
        slot_idx: usize,
        append_len: usize,
    ) -> Result<()> {
        if append_len == 0 || self.flashmla_slot_pages == 0 {
            return Ok(());
        }
        let band_pages = self.flashmla_slot_pages;
        let pool = self.flashmla_pool_mut()?;
        ensure!(
            pool.page_indices(slot_idx).len() == band_pages,
            "DSv4 FlashMLA slot {slot_idx} band not mirrored ({} pages, need {band_pages}) before cursor advance",
            pool.page_indices(slot_idx).len()
        );
        let new_cursor = pool.seq_len(slot_idx) + append_len;
        pool.set_band_cursor(slot_idx, new_cursor)
            .map_err(|e| anyhow!("DSv4 FlashMLA slot {slot_idx} cursor advance failed: {e}"))
    }

    pub(crate) fn flashmla_free_slot(&mut self, slot_idx: usize) -> Result<()> {
        if self.flashmla_slot_pages == 0 {
            return Ok(());
        }
        self.flashmla_pool_mut()?.free_slot(slot_idx);
        Ok(())
    }

    /// Set the FlashMLA band's logical cursor to `seq_len` (band stays resident).
    /// Used by the position-0 prefix-restore path: `restore_to` draws the band
    /// with cursor 0, then the slot-level swap-in sets it to the restored length
    /// so the tail prefill's `seq_len == append_pos` invariant holds.
    pub(crate) fn flashmla_set_band_cursor(
        &mut self,
        slot_idx: usize,
        seq_len: usize,
    ) -> Result<()> {
        if self.flashmla_slot_pages == 0 {
            return Ok(());
        }
        self.flashmla_pool_mut()?
            .set_band_cursor(slot_idx, seq_len)
            .map_err(|e| anyhow!("DSv4 FlashMLA slot {slot_idx} band cursor set failed: {e}"))
    }

    /// H2: clamp the FlashMLA pool's per-slot append cursor back to `new_len`
    /// tokens on an MTP reject — the pool advances 1 token/decode-tick but the
    /// MTP commit can roll back N drafted tokens, so the pool seq_len must shrink
    /// in lockstep with the compressor/indexer/DSA truncate or the next tick's
    /// `append_pos != pool seq_len` aborts the prepare invariant. Cursor-only:
    /// the fixed-layout band stays fully resident (never recycle band pages).
    pub(crate) fn flashmla_truncate_slot(&mut self, slot_idx: usize, new_len: usize) -> Result<()> {
        if self.flashmla_slot_pages == 0 {
            return Ok(());
        }
        self.flashmla_pool_mut()?
            .set_band_cursor(slot_idx, new_len)
            .map_err(|e| anyhow!("DSv4 FlashMLA slot {slot_idx} cursor truncate failed: {e}"))
    }

    pub(crate) fn flashmla_slot_first_block_or_zero(&self, slot_idx: usize) -> Result<usize> {
        Ok(self
            .flashmla_page_table(slot_idx)?
            .first()
            .copied()
            .unwrap_or(0) as usize)
    }

    pub(crate) fn flashmla_page_table_padded_i32(&self, slot_idx: usize) -> Result<Vec<i32>> {
        let table = self.flashmla_page_table(slot_idx)?;
        ensure!(
            table.len() <= self.flashmla_slot_pages,
            "DSv4 FlashMLA slot {slot_idx} page table len {} exceeds slot page budget {}",
            table.len(),
            self.flashmla_slot_pages
        );
        let mut out = table
            .iter()
            .map(|&page| {
                i32::try_from(page).map_err(|_| anyhow!("DSv4 FlashMLA page {page} exceeds i32"))
            })
            .collect::<Result<Vec<_>>>()?;
        let pad = out.last().copied().unwrap_or(0);
        out.resize(self.flashmla_slot_pages, pad);
        Ok(out)
    }

    pub(crate) fn flashmla_zero_band(
        &mut self,
        ctx: &DeviceContext,
        slot_idx: usize,
    ) -> Result<()> {
        if self.flashmla_slot_pages == 0 {
            return Ok(());
        }
        let table = self.flashmla_page_table(slot_idx)?.to_vec();
        let payload = vec![0u8; table.len() * self.flashmla_page_bytes];
        self.flashmla_pool_mut()?
            .copy_pages_from_host(ctx, &table, &payload)
            .map_err(|e| anyhow!("DSv4 shared FlashMLA slot zero failed: {e}"))
    }
}

pub(crate) trait ModelKvAdapter {
    type BatchView;

    fn prepare_kv_batch(&mut self, desc: &KvBatchDescriptor) -> Result<Self::BatchView>;
}

#[derive(Debug, Clone)]
pub(crate) struct Dsv4KvBatchView {
    pub(crate) rows: Vec<Dsv4KvBatchRowView>,
    pub(crate) flat_page_ids: Vec<u32>,
    pub(crate) flat_slot_page_ids: Vec<u32>,
}

#[derive(Debug, Clone)]
pub(crate) struct Dsv4KvBatchRowView {
    pub(crate) slot: usize,
    pub(crate) kind: KvBatchRowKind,
    pub(crate) seq_len: usize,
    pub(crate) append_pos: usize,
    pub(crate) append_len: usize,
    #[allow(dead_code)]
    pub(crate) slot_epoch: u64,
    pub(crate) page_range: std::ops::Range<usize>,
    pub(crate) slot_page_range: std::ops::Range<usize>,
}

impl Dsv4KvAdapter {
    pub(crate) fn new(
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        layer_specs: &[(DeepSeekV4AttentionMode, usize, usize)],
        max_seq_len: usize,
        kv_arena: &Dsv4MlaKvArena,
        tp_world: usize,
        num_slots: usize,
        mla_decode: Vec<Option<Dsv4MlaDecodeGraphScratch>>,
        moe_decode: Option<(
            &infer_moe::MoeConfig,
            &ExpertSplit,
            &crate::dsv4::Dsv4MoeLayer,
        )>,
        shared_expert_decode: Option<&crate::dsv4::Dsv4MoeLayer>,
        hidden_size: usize,
    ) -> Result<Self> {
        ensure!(num_slots > 0, "DSv4 attention pool needs at least one slot");
        ensure!(
            mla_decode.len() == layer_specs.len(),
            "DSv4 MLA decode scratch len {} != layer specs len {}",
            mla_decode.len(),
            layer_specs.len()
        );
        let layers = layer_specs
            .iter()
            .map(|&(mode, compress_ratio, local_heads)| {
                Dsv4LayerKvLayout::new(
                    ctx,
                    config,
                    mode,
                    compress_ratio,
                    max_seq_len,
                    kv_arena,
                    local_heads,
                    tp_world,
                    num_slots,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        // Build the ONE shared official-DSA scratch when any indexer layer
        // exists. Widened to has_indexer() so GLM SparseIndexed (every layer)
        // also builds the scratch. All indexer layers must agree on the indexer
        // ratio: CSA = compress_ratio (uniform cr=4 on DSv4-Flash); GLM
        // SparseIndexed maps to ratio=1 (full-sequence, every token a key). A
        // mixed CSA+SparseIndexed model trips the uniform-ratio assertion below
        // (out of scope: GLM is pure SparseIndexed, DSv4 pure CSA).
        let mut csa_ratios = layer_specs
            .iter()
            .filter(|(mode, _, _)| mode.has_indexer())
            .map(|&(mode, compress_ratio, _)| {
                if mode == DeepSeekV4AttentionMode::SparseIndexed {
                    1
                } else {
                    compress_ratio
                }
            });
        let dsa_shared = match csa_ratios.next() {
            Some(first_cr) if dsv4_dsa_official_enabled()? => {
                ensure!(
                    csa_ratios.all(|cr| cr == first_cr),
                    "DSv4 shared DSA scratch requires a uniform CSA compress_ratio"
                );
                Some(Dsv4DsaSharedScratch::new(
                    ctx,
                    config,
                    first_cr,
                    max_seq_len,
                    num_slots,
                )?)
            }
            _ => None,
        };
        let moe_decode_shared = moe_decode
            .map(|(cfg, split, layer)| {
                crate::moe::Dsv4MoeDecodeScratch::new(ctx, cfg, split, layer)
            })
            .transpose()?;
        // Compact-FP8 decode-band MoE tail scratch — allocated whenever the model
        // has a MoE layer (from the same `moe_decode` tuple), independent of the
        // decode graph, since the batched-stream decode path (launch-bound Step 1
        // target) is the primary consumer.
        let moe_tail_scratch = moe_decode
            .map(|(_cfg, split, layer)| {
                crate::moe::Dsv4MoeTailScratch::new(
                    ctx,
                    layer.hidden_dim,
                    layer.intermediate,
                    split.experts_per_rank,
                )
            })
            .transpose()?;
        // ALWAYS allocate the model-wide shared-expert output. Capacity is the
        // bounded MTP verify chunk (`MAX_SPEC_VERIFY_ROWS`); B=1 decode simply
        // sets `seq_len = 1` before dispatch. Keeping it pre-allocated avoids a
        // per-layer `uninit` on both the default decode and scheduled verify
        // paths.
        let shared_expert_out = Some(unsafe {
            HiddenStates::uninit(ctx, hidden_size, crate::dsv4::MAX_SPEC_VERIFY_ROWS)?
        });
        let shared_expert_scratch = shared_expert_decode
            .map(|layer| {
                crate::moe::Dsv4SharedDecodeScratch::new(
                    ctx,
                    layer.hidden_dim,
                    layer.shared_w2.cols,
                )
            })
            .transpose()?;
        // One model-wide batched (b=N) FlashMLA decode scratch (#60). Built for
        // canonical MODEL1 FlashMLA decode when the arena is live; V32/GLM keeps
        // the single-row path because the batched sparse-decode shim below is
        // MODEL1-only.
        // Build the per-layer FlashMLA decode shapes ONCE (shared by the batched
        // scratch and the new single-row shared scratch). Both gate on the same
        // `dsv4_flashmla_decode_alloc_enabled` predicate as the per-slot state.
        let (flashmla_batch, flashmla_scratch) = if dsv4_flashmla_decode_alloc_enabled()? {
            let layer_shapes = layer_specs
                .iter()
                .map(|&(mode, compress_ratio, local_heads)| {
                    Dsv4FlashMlaDecodeShape::new(
                        config,
                        mode,
                        compress_ratio,
                        max_seq_len,
                        kv_arena,
                        local_heads,
                        tp_world,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let batch = if config.head_dim == 512 {
                Some(Dsv4FlashMlaDecodeBatchScratch::new(
                    ctx,
                    config,
                    num_slots,
                    &layer_shapes,
                )?)
            } else {
                None
            };
            // ONE model-wide single-row decode scratch (#85 P3), hoisted from the
            // per-(slot,layer) state. Worst-case-sized across all FlashMLA layers.
            let single = Dsv4FlashMlaDecodeScratch::new(ctx, config, &layer_shapes)?;
            (batch, Some(single))
        } else {
            (None, None)
        };
        // ONE model-wide FP8 prefill DeepGEMM linear scratch (hoisted from the
        // per-slot `Dsv4LayerAttentionState`). Same gate the per-slot alloc used.
        // Sized by `config` + `max_seq_len` only (no layer-specific dims), so a
        // single instance serves every layer and slot; layers run sequentially
        // and the engine runs one forward at a time, so it is never aliased
        // concurrently. OFF by default → byte-identical to the disabled path.
        let prefill_linear = if dsv4_fp8_linear_deepgemm_enabled()? {
            Some(Dsv4PrefillDeepGemmLinearScratch::new(
                ctx,
                config,
                max_seq_len,
            )?)
        } else {
            None
        };
        Ok(Self {
            layers,
            num_slots,
            slot_epochs: vec![None; num_slots],
            dsa_shared,
            moe_decode_shared,
            moe_tail_scratch,
            mla_decode,
            shared_expert_out,
            shared_expert_scratch,
            flashmla_batch,
            flashmla_scratch,
            prefill_linear,
            host_to_flashmla: HashMap::new(),
            head_dim: config.head_dim,
        })
    }

    /// Exact requested device bytes owned by the model-wide KV adapter: the
    /// per-layer KV layouts (the FlashMLA FP8 latent pools + DSA key caches,
    /// sized for ALL slots — the dominant DSv4 KV byte sink) + the four
    /// model-wide shared scratches. `slot_epochs`/`num_slots` are host-only.
    #[allow(dead_code)]
    pub(crate) fn device_bytes(&self) -> usize {
        self.device_bytes_breakdown().iter().map(|(_, b)| *b).sum()
    }

    /// Per-component byte breakdown for the VRAM ledger log. `layers` collapses
    /// to one summed entry (per-layer detail would be 40+ rows); the four
    /// shared scratches are itemized.
    #[allow(dead_code)]
    pub(crate) fn device_bytes_breakdown(&self) -> Vec<(&'static str, usize)> {
        let layers_bytes: usize = self.layers.iter().map(|l| l.device_bytes()).sum();
        vec![
            ("layers(kv_pool+dsa_cache)", layers_bytes),
            (
                "dsa_shared",
                self.dsa_shared.as_ref().map_or(0, |s| s.device_bytes()),
            ),
            (
                "moe_decode_shared",
                self.moe_decode_shared
                    .as_ref()
                    .map_or(0, |s| s.device_bytes_live()),
            ),
            (
                "mla_decode",
                self.mla_decode
                    .iter()
                    .filter_map(Option::as_ref)
                    .map(Dsv4MlaDecodeGraphScratch::device_bytes)
                    .sum(),
            ),
            (
                "shared_expert_out",
                self.shared_expert_out
                    .as_ref()
                    .map_or(0, |s| s.device_bytes()),
            ),
            (
                "shared_expert_scratch",
                self.shared_expert_scratch
                    .as_ref()
                    .map_or(0, |s| s.device_bytes_live()),
            ),
            (
                "flashmla_batch",
                self.flashmla_batch.as_ref().map_or(0, |s| s.device_bytes()),
            ),
            (
                "flashmla_scratch",
                self.flashmla_scratch
                    .as_ref()
                    .map_or(0, |s| s.device_bytes()),
            ),
            (
                "prefill_linear",
                self.prefill_linear.as_ref().map_or(0, |s| s.device_bytes()),
            ),
        ]
    }

    /// Split-borrow accessor: one layer's KV layout, the model-wide shared DSA
    /// scratch, the model-wide single-row FlashMLA decode scratch, AND the
    /// model-wide shared FP8 prefill DeepGEMM linear scratch (all disjoint
    /// fields, so all can be `&mut` at once). The FlashMLA scratch is `None`
    /// when FlashMLA decode is disabled at the build/override level (same gate
    /// as the per-slot state); the prefill scratch is `None` when native DeepGEMM
    /// is unavailable.
    #[allow(clippy::type_complexity)]
    pub(crate) fn layer_and_dsa_shared_mut(
        &mut self,
        layer_idx: usize,
    ) -> Result<(
        &mut Dsv4LayerKvLayout,
        Option<&mut Dsv4DsaSharedScratch>,
        Option<&mut Dsv4FlashMlaDecodeScratch>,
        Option<&mut Dsv4PrefillDeepGemmLinearScratch>,
    )> {
        let len = self.layers.len();
        let layer = self
            .layers
            .get_mut(layer_idx)
            .ok_or_else(|| anyhow!("DSv4 attention pool layer {layer_idx} outside len {len}"))?;
        Ok((
            layer,
            self.dsa_shared.as_mut(),
            self.flashmla_scratch.as_mut(),
            self.prefill_linear.as_mut(),
        ))
    }

    /// Split-borrow accessor for the commit-fold path: one layer's KV layout +
    /// the model-wide single-row FlashMLA decode scratch (both disjoint fields).
    /// The fold's FP8 SW ring pack reuses the shared scratch's `sw_bulk_*`
    /// buffers. `None` for the scratch when FlashMLA decode is disabled.
    pub(crate) fn layer_and_flashmla_scratch_mut(
        &mut self,
        layer_idx: usize,
    ) -> Result<(
        &mut Dsv4LayerKvLayout,
        Option<&mut Dsv4FlashMlaDecodeScratch>,
    )> {
        let len = self.layers.len();
        let layer = self
            .layers
            .get_mut(layer_idx)
            .ok_or_else(|| anyhow!("DSv4 attention pool layer {layer_idx} outside len {len}"))?;
        Ok((layer, self.flashmla_scratch.as_mut()))
    }

    /// Split-borrow accessor for eager B=1 MODEL1 MLA decode: one layer's KV
    /// layout + shared DSA scratch + FlashMLA scratch + that layer's persistent
    /// MLA scratch.
    pub(crate) fn layer_flashmla_and_mla_decode_mut(
        &mut self,
        layer_idx: usize,
    ) -> Result<(
        &mut Dsv4LayerKvLayout,
        Option<&mut Dsv4DsaSharedScratch>,
        Option<&mut Dsv4FlashMlaDecodeScratch>,
        Option<&mut Dsv4MlaDecodeGraphScratch>,
    )> {
        let len = self.layers.len();
        let layer = self
            .layers
            .get_mut(layer_idx)
            .ok_or_else(|| anyhow!("DSv4 attention pool layer {layer_idx} outside len {len}"))?;
        let scratch = self
            .mla_decode
            .get_mut(layer_idx)
            .ok_or_else(|| anyhow!("DSv4 MLA decode scratch layer {layer_idx} outside len {len}"))?
            .as_mut();
        Ok((
            layer,
            self.dsa_shared.as_mut(),
            self.flashmla_scratch.as_mut(),
            scratch,
        ))
    }

    /// `&mut` accessor for the model-wide shared FP8 prefill DeepGEMM linear
    /// scratch alone (mirrors `dsa_shared`/`flashmla_batch` accessors). `None`
    /// when native DeepGEMM is unavailable.
    #[allow(dead_code)]
    pub(crate) fn prefill_linear_mut(&mut self) -> Option<&mut Dsv4PrefillDeepGemmLinearScratch> {
        self.prefill_linear.as_mut()
    }

    /// Split-borrow accessor for the batched (`b = N`) FlashMLA decode lane
    /// (#60): one layer's KV layout, the model-wide shared DSA scratch, the
    /// model-wide batched-decode scratch, AND the model-wide shared FP8 prefill
    /// DeepGEMM linear scratch (all disjoint fields). `None` for the batched
    /// scratch when the build/override disabled FlashMLA decode — the caller
    /// then takes the per-row lane; `None` for the prefill scratch when native
    /// DeepGEMM is unavailable.
    #[allow(clippy::type_complexity)]
    pub(crate) fn layer_dsa_and_flashmla_batch_mut(
        &mut self,
        layer_idx: usize,
    ) -> Result<(
        &mut Dsv4LayerKvLayout,
        Option<&mut Dsv4DsaSharedScratch>,
        Option<&mut Dsv4FlashMlaDecodeBatchScratch>,
        Option<&mut Dsv4FlashMlaDecodeScratch>,
        Option<&mut Dsv4PrefillDeepGemmLinearScratch>,
    )> {
        let len = self.layers.len();
        let layer = self
            .layers
            .get_mut(layer_idx)
            .ok_or_else(|| anyhow!("DSv4 attention pool layer {layer_idx} outside len {len}"))?;
        Ok((
            layer,
            self.dsa_shared.as_mut(),
            self.flashmla_batch.as_mut(),
            self.flashmla_scratch.as_mut(),
            self.prefill_linear.as_mut(),
        ))
    }

    /// Whether the model-wide batched FlashMLA decode scratch is allocated.
    pub(crate) fn has_flashmla_batch_scratch(&self) -> bool {
        self.flashmla_batch.is_some()
    }

    pub(crate) fn moe_decode_graph_scratch_mut(
        &mut self,
    ) -> Option<&mut crate::moe::Dsv4MoeDecodeScratch> {
        self.moe_decode_shared.as_mut()
    }

    pub(crate) fn moe_tail_scratch_mut(&mut self) -> Option<&mut crate::moe::Dsv4MoeTailScratch> {
        self.moe_tail_scratch.as_mut()
    }

    pub(crate) fn shared_expert_decode_mut(
        &mut self,
    ) -> (
        Option<&mut HiddenStates>,
        Option<&mut crate::moe::Dsv4SharedDecodeScratch>,
    ) {
        (
            self.shared_expert_out.as_mut(),
            self.shared_expert_scratch.as_mut(),
        )
    }

    pub(crate) fn layer_mut(&mut self, layer_idx: usize) -> Result<&mut Dsv4LayerKvLayout> {
        let len = self.layers.len();
        self.layers
            .get_mut(layer_idx)
            .ok_or_else(|| anyhow!("DSv4 attention pool layer {layer_idx} outside len {len}"))
    }

    pub(crate) fn layer(&self, layer_idx: usize) -> Result<&Dsv4LayerKvLayout> {
        let len = self.layers.len();
        self.layers
            .get(layer_idx)
            .ok_or_else(|| anyhow!("DSv4 attention pool layer {layer_idx} outside len {len}"))
    }

    pub(crate) fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Must track the same layer `flashmla_max_slot_pages` maximizes over —
    /// each layer's pool is sized to its OWN `flashmla_slot_pages` (per-layer
    /// budget), so `.first()` (often the cheapest layer) under-sized the host
    /// page-id space below `fixed_pages_per_slot` (#85; see
    /// docs/experience/errors/2026-07-08-dsv4-slot-abort-band-leak-crash.md).
    pub(crate) fn flashmla_total_pages(&self) -> Option<usize> {
        self.layers
            .iter()
            .max_by_key(|l| l.flashmla_slot_pages())
            .map(Dsv4LayerKvLayout::flashmla_total_pages)
    }

    pub(crate) fn flashmla_page_size(&self) -> Option<usize> {
        self.layers
            .first()
            .map(Dsv4LayerKvLayout::flashmla_page_size)
    }

    pub(crate) fn flashmla_max_slot_pages(&self) -> Option<usize> {
        let pages = self
            .layers
            .iter()
            .map(Dsv4LayerKvLayout::flashmla_slot_pages)
            .max()
            .unwrap_or(0);
        (pages > 0).then_some(pages)
    }

    /// MODEL1 (head_dim=512) uses Stage B device-page-table routing so bands
    /// may be non-contiguous. V32/GLM (head_dim=576) uses band-base pack
    /// addressing requiring contiguous identity-mapped bands — sharing disabled.
    pub(crate) fn supports_flashmla_page_sharing(&self) -> bool {
        self.head_dim != 576
    }

    /// Record host→FlashMLA mapping for a slot's fresh identity-mapped pages.
    fn record_host_mapping(&mut self, slot_pages: &[u32], slot: usize) {
        for (i, &host_page) in slot_pages.iter().enumerate() {
            self.host_to_flashmla.entry(host_page).or_insert_with(|| {
                (0..self.layers.len())
                    .map(|li| {
                        let lsp = self.layers[li].flashmla_slot_pages();
                        (slot * lsp + i.min(lsp.saturating_sub(1))) as u32
                    })
                    .collect()
            });
        }
    }

    /// Resolve host page IDs to one layer's FlashMLA physical pages.
    fn resolve_layer_pages(&self, host_pages: &[u32], layer_idx: usize) -> Vec<u32> {
        host_pages
            .iter()
            .filter_map(|&hp| {
                self.host_to_flashmla
                    .get(&hp)
                    .and_then(|v| v.get(layer_idx).copied())
            })
            .filter(|&p| p != u32::MAX)
            .collect()
    }

    pub(crate) fn mirror_slot_pages(
        &mut self,
        slot: usize,
        slot_pages: &[u32],
        seq_len: usize,
    ) -> Result<()> {
        ensure!(
            slot < self.num_slots,
            "DSv4 mirror slot {slot} outside adapter slots {}",
            self.num_slots
        );
        ensure!(
            !slot_pages.is_empty(),
            "DSv4 mirror slot {slot} has no pages"
        );
        let layer_count = self.layers.len();
        let per_layer_pages: Vec<Vec<u32>> = if self.supports_flashmla_page_sharing() {
            self.record_host_mapping(slot_pages, slot);
            (0..layer_count)
                .map(|li| self.resolve_layer_pages(slot_pages, li))
                .collect()
        } else {
            let pc = slot_pages.len();
            (0..layer_count)
                .map(|li| {
                    let lsp = self.layers[li].flashmla_slot_pages();
                    (0..pc)
                        .map(|i| (slot * lsp + i.min(lsp.saturating_sub(1))) as u32)
                        .collect()
                })
                .collect()
        };
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let layer_slot_pages = layer.flashmla_slot_pages();
            let n = layer_slot_pages.min(slot_pages.len());
            if n == 0 {
                continue;
            }
            let Some(pool) = layer.flashmla_kv_pool.as_mut() else {
                continue;
            };
            let local = &per_layer_pages[layer_idx][..n];
            pool.mirror_band(slot, local, seq_len)?;
        }
        Ok(())
    }

    pub(crate) fn zero_slot_band(&mut self, ctx: &DeviceContext, slot: usize) -> Result<()> {
        ensure!(
            slot < self.num_slots,
            "DSv4 zero slot {slot} outside adapter slots {}",
            self.num_slots
        );
        for layer in &mut self.layers {
            layer.flashmla_zero_band(ctx, slot)?;
        }
        Ok(())
    }

    /// Bump refcount on FlashMLA physical pages backing these host IDs (all layers).
    /// Called from `save_prefix_sidecar` when the engine publishes a prefix.
    pub(crate) fn retain_flashmla_pages(&mut self, host_pages: &[u32]) -> Result<()> {
        for layer_idx in 0..self.layers.len() {
            let Some(pool) = self.layers[layer_idx].flashmla_kv_pool.as_mut() else {
                continue;
            };
            let pages = self.resolve_layer_pages(host_pages, layer_idx);
            if !pages.is_empty() {
                pool.retain_pages(&pages)?;
            }
        }
        Ok(())
    }

    /// Drop refcount on FlashMLA physical pages backing these host IDs.
    /// Called from `release_prefix_pages` when the radix evicts a prefix.
    pub(crate) fn release_flashmla_pages(&mut self, host_pages: &[u32]) -> Result<()> {
        for layer_idx in 0..self.layers.len() {
            let Some(pool) = self.layers[layer_idx].flashmla_kv_pool.as_mut() else {
                continue;
            };
            let pages = self.resolve_layer_pages(host_pages, layer_idx);
            if !pages.is_empty() {
                pool.release_pages(&pages)?;
            }
        }
        for hp in host_pages {
            self.host_to_flashmla.remove(hp);
        }
        Ok(())
    }

    /// Attach a slot to existing FlashMLA pages via the host→FlashMLA mapping.
    /// No-op for V32/GLM (contiguous identity bands required).
    pub(crate) fn attach_slot_flashmla_pages(
        &mut self,
        slot: usize,
        host_pages: &[u32],
        token_count: usize,
    ) -> Result<()> {
        if !self.supports_flashmla_page_sharing() {
            return Ok(());
        }
        ensure!(
            slot < self.num_slots,
            "DSv4 attach slot {slot} outside adapter slots {}",
            self.num_slots
        );
        let layer_count = self.layers.len();
        let per_layer_pages: Vec<Vec<u32>> = (0..layer_count)
            .map(|li| self.resolve_layer_pages(host_pages, li))
            .collect();
        for (li, pages) in per_layer_pages.iter().enumerate() {
            ensure!(
                pages.len() == host_pages.len(),
                "DSv4 attach slot {slot}: layer {li} mapped {}/{} host pages — \
                 prefix not recorded (missing save_prefix_sidecar?)",
                pages.len(),
                host_pages.len()
            );
        }
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let Some(pool) = layer.flashmla_kv_pool.as_mut() else {
                continue;
            };
            let lsp = layer.flashmla_slot_pages();
            let n = lsp.min(host_pages.len());
            pool.mirror_band(slot, &per_layer_pages[layer_idx][..n], token_count)?;
        }
        Ok(())
    }
}

impl ModelKvAdapter for Dsv4KvAdapter {
    type BatchView = Dsv4KvBatchView;

    fn prepare_kv_batch(&mut self, desc: &KvBatchDescriptor) -> Result<Self::BatchView> {
        let mut rows = Vec::with_capacity(desc.rows.len());
        for (idx, row) in desc.rows.iter().enumerate() {
            ensure!(
                row.slot < self.num_slots,
                "DSv4 KV batch row {idx} slot {} outside adapter slots {}",
                row.slot,
                self.num_slots
            );
            ensure!(
                row.page_range.end <= desc.flat_page_ids.len(),
                "DSv4 KV batch row {idx} page range {:?} outside flat page len {}",
                row.page_range,
                desc.flat_page_ids.len()
            );
            ensure!(
                row.token_range.end <= desc.flat_token_ids.len(),
                "DSv4 KV batch row {idx} token range {:?} outside flat token len {}",
                row.token_range,
                desc.flat_token_ids.len()
            );
            ensure!(
                row.page_range.start < row.page_range.end,
                "DSv4 KV batch row {idx} has empty page range"
            );
            ensure!(
                row.slot_page_range.end <= desc.flat_slot_page_ids.len(),
                "DSv4 KV batch row {idx} slot page range {:?} outside flat slot page len {}",
                row.slot_page_range,
                desc.flat_slot_page_ids.len()
            );
            let slot_pages: Vec<u32> =
                desc.flat_slot_page_ids[row.slot_page_range.clone()].to_vec();
            ensure!(
                !slot_pages.is_empty(),
                "DSv4 KV batch row {idx} has empty slot page table"
            );
            let layer_count = self.layers.len();
            let per_layer_pages: Vec<Vec<u32>> = if self.supports_flashmla_page_sharing() {
                self.record_host_mapping(&slot_pages, row.slot);
                (0..layer_count)
                    .map(|li| self.resolve_layer_pages(&slot_pages, li))
                    .collect()
            } else {
                let pc = slot_pages.len();
                (0..layer_count)
                    .map(|li| {
                        let lsp = self.layers[li].flashmla_slot_pages();
                        (0..pc)
                            .map(|i| (row.slot * lsp + i.min(lsp.saturating_sub(1))) as u32)
                            .collect()
                    })
                    .collect()
            };
            for layer_idx in 0..layer_count {
                let layer = self.layer_mut(layer_idx)?;
                let layer_slot_pages = layer.flashmla_slot_pages();
                let n = layer_slot_pages.min(slot_pages.len());
                let layer_pages = &per_layer_pages[layer_idx][..n];
                match row.kind {
                    KvBatchRowKind::Prefill => {
                        if let Some(pool) = layer.flashmla_kv_pool.as_mut() {
                            pool.mirror_band(row.slot, layer_pages, row.append_pos)?;
                        }
                        ensure!(
                            layer
                                .flashmla_pool()
                                .map_or(true, |p| p.seq_len(row.slot) == row.append_pos),
                            "DSv4 prefill slot {} layer {layer_idx} pool seq_len {} != append_pos {}",
                            row.slot,
                            layer.flashmla_pool().map_or(0, |p| p.seq_len(row.slot)),
                            row.append_pos
                        );
                        layer.flashmla_alloc_append(row.slot, row.append_len)?;
                    }
                    KvBatchRowKind::Decode => {
                        if let Some(pool) = layer.flashmla_kv_pool.as_mut() {
                            pool.mirror_band(row.slot, layer_pages, row.append_pos)?;
                        }
                        ensure!(
                            layer
                                .flashmla_pool()
                                .map_or(true, |p| p.seq_len(row.slot) == row.append_pos),
                            "DSv4 decode slot {} layer {layer_idx} pool seq_len {} != append_pos {}",
                            row.slot,
                            layer.flashmla_pool().map_or(0, |p| p.seq_len(row.slot)),
                            row.append_pos
                        );
                        layer.flashmla_alloc_append(row.slot, row.append_len)?;
                    }
                }
            }
            self.slot_epochs[row.slot] = Some(row.slot_epoch);
            rows.push(Dsv4KvBatchRowView {
                slot: row.slot,
                kind: row.kind,
                seq_len: row.seq_len,
                append_pos: row.append_pos,
                append_len: row.append_len,
                slot_epoch: row.slot_epoch,
                page_range: row.page_range.clone(),
                slot_page_range: row.slot_page_range.clone(),
            });
        }

        Ok(Dsv4KvBatchView {
            rows,
            flat_page_ids: desc.flat_page_ids.clone(),
            flat_slot_page_ids: desc.flat_slot_page_ids.clone(),
        })
    }
}

impl Dsv4LayerKvLayout {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        mode: DeepSeekV4AttentionMode,
        compress_ratio: usize,
        max_seq_len: usize,
        kv_arena: &Dsv4MlaKvArena,
        local_heads: usize,
        tp_world: usize,
        num_slots: usize,
    ) -> Result<Self> {
        let flashmla_slot_pages = if dsv4_flashmla_decode_alloc_enabled()? {
            let shape = Dsv4FlashMlaDecodeShape::new(
                config,
                mode,
                compress_ratio,
                max_seq_len,
                kv_arena,
                local_heads,
                tp_world,
            )?;
            shape.total_blocks
        } else {
            0
        };
        let flashmla_page_bytes = kv_arena
            .page_block_size
            .checked_mul(kv_arena.bytes_per_token)
            .ok_or_else(|| anyhow!("DSv4 shared FlashMLA page byte size overflow"))?;
        let flashmla_kv_pool = if flashmla_slot_pages > 0 {
            // Shared packed MLA latent pool. DSv4 does not self-allocate a
            // sequential table here: `prepare_kv_batch` mirrors the
            // host-owned fixed band (`flat_slot_page_ids`) into this pool, so
            // engine/radix/tier page identity is the single source of truth.
            // The band is total_blocks pages (SW ring + comp region addressed
            // by fixed slot-logical block id, NOT ceil(seq_len)).
            let format = KVFormat::PackedBytes {
                bytes_per_token: kv_arena.bytes_per_token,
            };
            ensure!(
                format.default_page_size() == kv_arena.page_block_size,
                "DSv4 FlashMLA pool page size {} != arena block size {}",
                format.default_page_size(),
                kv_arena.page_block_size
            );
            // #85 Stage-B / KV-budget convergence: each layer is sized for its
            // OWN real need — `num_slots` whole bands of this layer's own
            // `flashmla_slot_pages` — not a uniform per-layer share of the
            // pool total. `kv_budget_plan`'s pool-affordable-slots re-clamp
            // (pod-verified 2026-07-06) already sizes `num_slots` against the
            // SUM of every layer's pages, so summing each layer's own exact
            // band here reconstructs that same total, per layer, exactly.
            let budget_bytes = num_slots
                .saturating_mul(flashmla_slot_pages)
                .saturating_mul(flashmla_page_bytes);
            let pool = TokenKVPool::with_format(
                ctx,
                1,
                1,
                config.head_dim,
                num_slots,
                budget_bytes,
                format,
            )
            .map_err(|e| anyhow!("DSv4 shared FlashMLA pool alloc failed: {e}"))?;
            // Sanity: the pool must cover every slot's band, not just one — catches
            // a `kv_budget_plan` clamp regression here instead of a mid-serve
            // `alloc_fixed_band` bail (pod-verified 2026-07-06: a 2-request
            // concurrent burst crashed the coordinator before this was in place).
            ensure!(
                pool.page_size == kv_arena.page_block_size
                    && pool.max_total_pages >= num_slots.saturating_mul(flashmla_slot_pages),
                "DSv4 FlashMLA pool page mismatch: page_size={} pages={} need page_size={} pages>={}",
                pool.page_size,
                pool.max_total_pages,
                kv_arena.page_block_size,
                num_slots.saturating_mul(flashmla_slot_pages)
            );
            Some(pool)
        } else {
            None
        };

        // GLM SparseIndexed: full-sequence indexer, every token a key (ratio=1,
        // no compressor). CompressedSparse keeps its real compress_ratio.
        let index_ratio = if mode == DeepSeekV4AttentionMode::SparseIndexed {
            1
        } else {
            compress_ratio
        };
        // Widen the per-slot DSA key-cache to has_indexer() (CSA + GLM
        // SparseIndexed); sized at index_ratio (=1 for GLM) so the div_ceil in
        // dsv4_dsa_key_cache_bytes never sees compress_ratio==0.
        let dsa_slot_bytes = if mode.has_indexer() && dsv4_dsa_official_enabled()? {
            dsv4_dsa_key_cache_bytes(config, index_ratio, max_seq_len)?
        } else {
            0
        };
        let dsa_key_cache = if dsa_slot_bytes > 0 {
            Some(
                ctx.stream
                    .alloc_zeros::<u8>(
                        dsa_slot_bytes
                            .checked_mul(num_slots)
                            .ok_or_else(|| anyhow!("DSv4 shared DSA key-cache total overflow"))?,
                    )
                    .map_err(|e| anyhow!("DSv4 shared DSA key-cache alloc failed: {e}"))?,
            )
        } else {
            None
        };
        // compress_ratio==4 only. Main pool needs a compressor
        // (`mode.has_compressor()`); indexer pool needs an indexer
        // (`mode.has_indexer()`) — a CSA layer at ratio==4 has both, a
        // SlidingWindow layer at ratio==0 has neither.
        let compress_state_pool = if compress_ratio == 4 && mode.has_compressor() {
            Some(Dsv4CompressStatePool::new(
                ctx,
                config.head_dim,
                compress_ratio,
                max_seq_len,
            )?)
        } else {
            None
        };
        let indexer_compress_state_pool = if compress_ratio == 4 && mode.has_indexer() {
            Some(Dsv4CompressStatePool::new(
                ctx,
                config.index_head_dim,
                compress_ratio,
                max_seq_len,
            )?)
        } else {
            None
        };
        // Shares `dsa_key_cache`'s own gate (every has_indexer layer, not
        // ratio-restricted like the compress-state pools above).
        let dsa_key_cache_pool = if dsa_slot_bytes > 0 {
            Some(Dsv4DsaKeyCachePool::new(
                ctx,
                config,
                index_ratio,
                max_seq_len,
            )?)
        } else {
            None
        };
        let sw_ring_pool = if config.sliding_window > 0 {
            Some(Dsv4SwRingSnapshotPool::new(
                ctx,
                config.head_dim,
                config.sliding_window,
                max_seq_len,
            )?)
        } else {
            None
        };
        Ok(Self {
            flashmla_kv_pool,
            dsa_key_cache,
            flashmla_slot_pages,
            flashmla_page_bytes,
            dsa_slot_bytes,
            num_slots,
            compress_state_pool,
            indexer_compress_state_pool,
            dsa_key_cache_pool,
            sw_ring_pool,
        })
    }

    /// Exact requested device bytes owned by this ONE layer's KV layout (sized
    /// for ALL slots): the shared FlashMLA FP8 latent `TokenKVPool` (the
    /// dominant DSv4 KV byte sink) + the shared DSA key-cache band + the
    /// compressor/indexer compress-state pools. The `flashmla_page_table` /
    /// scalar shape fields are host-only.
    #[allow(dead_code)]
    pub(crate) fn device_bytes(&self) -> usize {
        self.flashmla_kv_pool
            .as_ref()
            .map_or(0, |p| p.device_bytes())
            + self.dsa_key_cache.as_ref().map_or(0, |s| s.len())
            + self
                .compress_state_pool
                .as_ref()
                .map_or(0, |p| p.device_bytes())
            + self
                .indexer_compress_state_pool
                .as_ref()
                .map_or(0, |p| p.device_bytes())
            + self
                .dsa_key_cache_pool
                .as_ref()
                .map_or(0, |p| p.device_bytes())
            + self.sw_ring_pool.as_ref().map_or(0, |p| p.device_bytes())
    }

    pub(super) fn slot_range(
        slot_idx: usize,
        slot_bytes: usize,
        num_slots: usize,
    ) -> Result<std::ops::Range<usize>> {
        ensure!(
            slot_idx < num_slots,
            "DSv4 attention pool slot {slot_idx} outside num_slots {num_slots}"
        );
        let start = slot_idx
            .checked_mul(slot_bytes)
            .ok_or_else(|| anyhow!("DSv4 attention pool slot offset overflow"))?;
        let end = start
            .checked_add(slot_bytes)
            .ok_or_else(|| anyhow!("DSv4 attention pool slot end overflow"))?;
        Ok(start..end)
    }

    /// The layer's shared FlashMLA latent pool (present iff the decode-alloc
    /// gate is on and this layer has a non-empty FlashMLA shape).
    pub(super) fn flashmla_pool(&self) -> Result<&TokenKVPool> {
        self.flashmla_kv_pool
            .as_ref()
            .ok_or_else(|| anyhow!("DSv4 FlashMLA shared pool missing"))
    }

    pub(super) fn flashmla_pool_mut(&mut self) -> Result<&mut TokenKVPool> {
        self.flashmla_kv_pool
            .as_mut()
            .ok_or_else(|| anyhow!("DSv4 FlashMLA shared pool missing"))
    }

    /// Whole-pool data plane (packed records live in the K plane only).
    pub(super) fn flashmla_pool_data(&self) -> Result<&CudaSlice<u8>> {
        Ok(self.flashmla_pool()?.k_data_slice(0))
    }

    pub(super) fn flashmla_pool_data_mut(&mut self) -> Result<&mut CudaSlice<u8>> {
        Ok(self.flashmla_pool_mut()?.k_data_slice_mut(0))
    }

    /// Shared FlashMLA pool base device pointer (batched pack: one base, N rows
    /// write disjoint bands via their per-slot page tables). Uniform across rows.
    pub(crate) fn flashmla_pool_base_ptr(&mut self, ctx: &DeviceContext) -> Result<u64> {
        let pool_buf = self.flashmla_pool_data_mut()?;
        let (ptr, guard) = pool_buf.device_ptr_mut(&ctx.stream);
        drop(guard);
        Ok(ptr)
    }

    /// One slot's page table: slot-logical page → physical pool page
    /// (FlashMLA's FFI calls our 64-token page a "block"). The ONLY source of band addresses (#85 P2) — token counts and
    /// block counts come from the pool's table, never re-derived from
    /// `slot_idx` arithmetic.
    pub(super) fn flashmla_page_table(&self, slot_idx: usize) -> Result<&[u32]> {
        ensure!(
            slot_idx < self.num_slots,
            "DSv4 attention pool slot {slot_idx} outside num_slots {}",
            self.num_slots
        );
        Ok(self.flashmla_pool()?.page_indices(slot_idx))
    }

    /// Table-routed byte range of one slot's FlashMLA band.
    ///
    /// Invariant (#85 P2 Stage A): the range derives from the slot's PAGE
    /// TABLE, and the helper errors unless the table is a contiguous identity
    /// run — that contiguity is what licenses the band-base addressing the
    /// device-side pack/index kernels still use. Stage B must hand those
    /// kernels a device-resident table before tables may fragment.
    ///
    /// The expected page count is the slot's ACTUAL drawn pages (`table.len()`),
    /// not the per-slot reservation (`flashmla_slot_pages`): a slot with no band
    /// yet has an empty table (handled by `contiguous_page_table_byte_range`'s
    /// `expected_pages > 0` guard), and a live slot draws its WHOLE band on
    /// admission (`flashmla_alloc_append`), so for the single-request/sequential
    /// case `table.len() == flashmla_slot_pages`. The contiguity check stays —
    /// the draw remains contiguous per slot.
    pub(super) fn flashmla_pages_byte_range(
        &self,
        slot_idx: usize,
    ) -> Result<std::ops::Range<usize>> {
        let table = self.flashmla_page_table(slot_idx)?;
        let range = contiguous_page_table_byte_range(table, table.len(), self.flashmla_page_bytes)?;
        let pool_bytes = self.flashmla_pool_data()?.len();
        ensure!(
            range.end <= pool_bytes,
            "DSv4 FlashMLA table range {:?} outside pool bytes {}",
            range,
            pool_bytes
        );
        Ok(range)
    }

    pub(crate) fn dsa_slot_range(&self, slot_idx: usize) -> Result<std::ops::Range<usize>> {
        Self::slot_range(slot_idx, self.dsa_slot_bytes, self.num_slots)
    }

    pub(super) fn reset_dsa_slot(
        &mut self,
        ctx: &DeviceContext,
        dsa: &Dsv4DsaOfficialState,
    ) -> Result<()> {
        let range = self.dsa_slot_range(dsa.slot_idx)?;
        let pool = self
            .dsa_key_cache
            .as_mut()
            .ok_or_else(|| anyhow!("DSv4 DSA shared key-cache missing"))?;
        ensure!(
            range.end <= pool.len() && range.len() == dsa.key_cache_len,
            "DSv4 DSA shared key-cache range {:?} invalid for pool_len={} slot_len={}",
            range,
            pool.len(),
            dsa.key_cache_len
        );
        let mut view = pool.slice_mut(range);
        ctx.stream
            .memset_zeros(&mut view)
            .map_err(|e| anyhow!("DSv4 shared DSA key-cache reset failed: {e}"))?;
        Ok(())
    }
}
use crate::attention::DeviceContext;
use anyhow::Result;
