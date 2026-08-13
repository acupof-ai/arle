use super::kv_layout::{DSV4_INDEXER_STAGING_RING_ROWS, DSV4_PREFILL_QUERY_CHUNK};
use super::*;
/// Query-dimension tile for the official DSA indexer prefill. The logits scratch
/// is `TILE × compressed_capacity` (f32); tiling the query axis keeps it bounded
/// (e.g. compress_ratio=4 @ 900K ctx: 4096 × 225024 × 4B ≈ 3.7 GB instead of ~810 GB).
/// This is the only path — long prompts loop in tiles, never materialize full-N logits.
///
/// Per-layer (each CSA layer owns its own `Dsv4DsaOfficialState` and thus its own
/// `logits` scratch — no cross-layer sharing, which is unsafe under this codebase's
/// disabled event-tracking + forward-level keepalive). The tile is 1024 (not 4096) so
/// all ~43 per-layer `logits` buffers fit at 900K: cr=4 → 1024 × ~225024 × 4B ≈ 0.92 GB
/// per cr=4 layer × ~20 such layers ≈ 18.4 GB, within the ~31 GB free budget. The 4096
/// tile OOMs (4096 × ~225024 × 4B ≈ 3.7 GB/layer × ~20 ≈ 74 GB).
/// `csa_select_official` loops sub-chunks when a forward passes more than `query_tile`
/// query tokens, so correctness is unchanged — just more sub-iterations.
const DSV4_DSA_PREFILL_QUERY_TILE: usize = 1024;

/// Per-(slot, CSA-layer) STATEFUL half of the official DSA selector.
///
/// Only the pieces that carry cross-call state live here: the `packed_rows`
/// progress counter and the slot's key-cache band binding. `rotated_keys` is a
/// transient drain-immediate staging buffer (delta-sized, NOT full history).
/// Every per-forward scratch buffer and every constant table is shared across
/// slots AND layers in [`Dsv4DsaSharedScratch`] (issue #67 — the per-slot ×
/// per-layer copies of the `logits` tile alone made 256K boot impossible).
pub(super) struct Dsv4DsaOfficialState {
    pub(super) slot_idx: usize,
    pub(super) key_cache_len: usize,
    pub(super) rotated_keys: CudaSlice<half::bf16>,
    pub(super) packed_rows: usize,
}

impl Dsv4DsaOfficialState {
    pub(super) fn new(
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        compress_ratio: usize,
        max_seq_len: usize,
        slot_idx: usize,
        pool: &Dsv4LayerKvLayout,
    ) -> Result<Self> {
        let compressed_capacity = max_seq_len.div_ceil(compress_ratio.max(1)).max(1);
        let key_cache_bytes = dsv4_dsa_key_cache_bytes(config, compress_ratio, max_seq_len)?;
        let range = pool.dsa_slot_range(slot_idx)?;
        ensure!(
            pool.dsa_key_cache.is_some()
                && range.len() == key_cache_bytes
                && range.len() == pool.dsa_slot_bytes
                && !range.is_empty(),
            "DSv4 official DSA shared slot band missing/invalid for slot {slot_idx}"
        );
        Ok(Self {
            slot_idx,
            key_cache_len: key_cache_bytes,
            rotated_keys: ctx
                .stream
                .alloc_zeros::<half::bf16>(
                    dsv4_dsa_rotated_ring_rows(compressed_capacity) * config.index_head_dim,
                )
                .map_err(|e| anyhow!("DSv4 official DSA rotated key alloc failed: {e}"))?,
            packed_rows: 0,
        })
    }

    pub(super) fn reset(
        &mut self,
        ctx: &DeviceContext,
        pool: &mut Dsv4LayerKvLayout,
    ) -> Result<()> {
        self.packed_rows = 0;
        pool.reset_dsa_slot(ctx, self)?;
        Ok(())
    }

    /// Exact requested device bytes owned by this per-(slot,CSA-layer) DSA
    /// state: the transient `rotated_keys` staging buffer only. The slot's
    /// `dsa_key_cache` band is owned by [`Dsv4LayerKvLayout::dsa_key_cache`]
    /// (summed there once), not here.
    #[allow(dead_code)]
    pub(crate) fn device_bytes(&self) -> usize {
        self.rotated_keys.len() * std::mem::size_of::<half::bf16>()
    }

    pub(crate) fn swap_out(&self, ctx: &DeviceContext) -> Result<crate::attention::Dsv4DsaImage> {
        let rotated_keys = ctx
            .stream
            .clone_dtoh(&self.rotated_keys)
            .map_err(|e| anyhow!("DSv4 DSA swap rotated_keys D2H failed: {e}"))?;
        Ok(crate::attention::Dsv4DsaImage {
            key_cache_len: self.key_cache_len,
            rotated_keys,
            packed_rows: self.packed_rows,
        })
    }

    pub(crate) fn swap_in(
        &mut self,
        ctx: &DeviceContext,
        image: &crate::attention::Dsv4DsaImage,
    ) -> Result<()> {
        ctx.stream
            .memcpy_htod(&image.rotated_keys, &mut self.rotated_keys)
            .map_err(|e| anyhow!("DSv4 DSA swap rotated_keys H2D failed: {e}"))?;
        self.packed_rows = image.packed_rows;
        Ok(())
    }
}

/// Per-MODEL shared half of the official DSA selector — ONE instance per
/// executor, shared across every CSA layer and every slot (issue #67).
///
/// Sharing safety: every kernel that touches these buffers is enqueued on the
/// single `ctx.stream`, so a later (slot, layer) call's writes are
/// stream-ordered after the earlier call's reads — the overwrite-before-read
/// discipline that already held per-tile within one call holds across calls
/// for free. The disabled-event-tracking hazard is about FREEING device memory
/// while async kernels may still touch it; this scratch lives as long as the
/// KV adapter and is never dropped mid-forward, so no premature-reuse window
/// exists. Contents carry NO cross-call state:
///
/// - per-forward scratch, overwritten before every read: `logits`, `q_fp8`,
///   `weights`, `context_lens`, `positions`, `sched_meta`, `raw_indices`;
/// - constants of (config, compress_ratio, max_seq): `cache_locs`,
///   `page_table_identity`, `freqs_cis`.
///
/// The stateful per-(slot, layer) pieces stay in [`Dsv4DsaOfficialState`].
pub(crate) struct Dsv4DsaSharedScratch {
    pub(super) cache_locs: CudaSlice<i64>,
    pub(super) q_fp8: CudaSlice<u8>,
    pub(super) weights: CudaSlice<f32>,
    pub(super) context_lens: CudaSlice<i32>,
    pub(super) positions: CudaSlice<i32>,
    pub(super) page_table_identity: CudaSlice<i32>,
    pub(super) freqs_cis: CudaSlice<f32>,
    pub(super) sched_meta: CudaSlice<i32>,
    pub(super) logits: CudaSlice<f32>,
    pub(super) raw_indices: CudaSlice<i32>,
    // ── N-row BATCHED-DECODE scratch (#60). DISJOINT from the single-forward
    // query-tile buffers above: the batched decode select (`csa_select_official_batched`)
    // batches the READ side (logits + topk) of N decode rows into ONE
    // `batch_size=N` DeepGEMM paged-MQA call, instead of N `batch_size=1` calls.
    // Sized by `decode_max_batch == num_slots` (one decode row per slot). All are
    // overwritten-before-read each forward and stream-ordered; never aliased with
    // the single-row / prefill `q_fp8`/`weights`/`logits`/etc. buffers, so the
    // single-row path is byte-identical.
    pub(super) decode_max_batch: usize,
    pub(super) q_fp8_batch: CudaSlice<u8>,
    pub(super) weights_batch: CudaSlice<f32>,
    pub(super) context_lens_batch: CudaSlice<i32>,
    pub(super) positions_batch: CudaSlice<i32>,
    pub(super) logits_batch: CudaSlice<f32>,
    pub(super) raw_indices_batch: CudaSlice<i32>,
    pub(super) block_table_batch: CudaSlice<i32>,
    // N-row identity page-table for the batched topk. The topk launch validator
    // (dsv4_dsa_official.cu:621) rejects `page_table_stride <= 0`, so stride=0 is
    // illegal; we replicate the identity `[0..num_pages)` block `num_slots` times
    // and pass stride=num_pages. Each row then reads identity `[0..num_pages)` —
    // `page_to_slot(identity,i)=i` — byte-equivalent to the single-row path's
    // slot-relative mapping (which reads `page_table_identity[0..num_pages)` at bid=0).
    pub(super) page_table_identity_batch: CudaSlice<i32>,
    pub(super) max_tokens: usize,
    pub(super) query_tile: usize,
    pub(super) query_chunk: usize,
    pub(super) compressed_capacity: usize,
    pub(super) num_pages: usize,
    pub(super) num_heads: usize,
    pub(super) head_dim: usize,
    pub(super) logits_stride: usize,
    pub(super) num_sms: usize,
}

impl Dsv4DsaSharedScratch {
    pub(crate) fn new(
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        compress_ratio: usize,
        max_seq_len: usize,
        num_slots: usize,
    ) -> Result<Self> {
        ensure!(
            config.index_head_dim == 128,
            "Official DSv4 DSA indexer requires index_head_dim=128, got {}",
            config.index_head_dim
        );
        ensure!(
            config.index_n_heads == 32 || config.index_n_heads == 64,
            "Official DSv4 DSA indexer requires 32/64 heads, got {}",
            config.index_n_heads
        );
        let compressed_capacity = max_seq_len.div_ceil(compress_ratio.max(1)).max(1);
        let page_size = 64usize;
        let num_pages = compressed_capacity.div_ceil(page_size).max(1);
        let max_tokens = max_seq_len.max(1);
        // Query-dimension scratch is tiled: bounded by `query_tile` regardless of how
        // many query tokens a single call passes. When token_count <= query_tile the
        // compute loop runs a single iteration (offset 0), behavior-identical to the
        // pre-tiling code. Key-dimension / full-N buffers (cache_locs, freqs_cis)
        // stay sized by max_tokens/compressed_capacity; `raw_indices` (the
        // per-forward topk output) is chunk-sized — see below.
        let query_tile = DSV4_DSA_PREFILL_QUERY_TILE.min(max_tokens);
        // `raw_indices` is the topk OUTPUT: written per forward over the forward's
        // query tokens and read by the topk transform over those same <= chunk
        // queries — never the full max_tokens context. Size it by the
        // chunked-prefill query bound (`DSV4_PREFILL_QUERY_CHUNK`), not `max_tokens`,
        // which at 900K x topk would be ~1.9 GB/layer. `csa_select_official` asserts
        // `q_i.seq_len <= query_chunk` before the tile loop.
        let query_chunk = DSV4_PREFILL_QUERY_CHUNK.min(max_tokens);
        let q_elems = query_tile
            .checked_mul(config.index_n_heads)
            .and_then(|v| v.checked_mul(config.index_head_dim))
            .ok_or_else(|| anyhow!("DSv4 official DSA q scratch size overflow"))?;
        let logits_stride = compressed_capacity.div_ceil(256) * 256;
        let logits_elems = query_tile
            .checked_mul(logits_stride)
            .ok_or_else(|| anyhow!("DSv4 official DSA logits scratch size overflow"))?;
        let cache_locs_h: Vec<i64> = (0..compressed_capacity)
            .map(|v| i64::try_from(v).expect("compressed capacity fits i64"))
            .collect();
        let freqs_cis_h = dsv4_dsa_freqs_cis_real(config, compress_ratio, max_seq_len)?;
        let _ = query_tile
            .checked_mul(num_pages)
            .ok_or_else(|| anyhow!("DSv4 official DSA page table size overflow"))?;
        let page_table_h: Vec<i32> = (0..query_tile)
            .flat_map(|_| (0..num_pages).map(|v| i32::try_from(v).expect("page table fits i32")))
            .collect();
        let num_sms = crate::runtime_flags::dsv4_dsa_indexer_sms();
        // ── N-row batched-decode scratch sizing (#60). `decode_max_batch` is one
        // decode row per slot. All buffers are overwritten-before-read each forward.
        let decode_max_batch = num_slots.max(1);
        let q_fp8_batch_elems = decode_max_batch
            .checked_mul(config.index_n_heads)
            .and_then(|v| v.checked_mul(config.index_head_dim))
            .ok_or_else(|| anyhow!("DSv4 official DSA batched q scratch size overflow"))?;
        let logits_batch_elems = decode_max_batch
            .checked_mul(logits_stride)
            .ok_or_else(|| anyhow!("DSv4 official DSA batched logits scratch size overflow"))?;
        let block_table_batch_elems = decode_max_batch
            .checked_mul(num_pages)
            .ok_or_else(|| anyhow!("DSv4 official DSA batched block table size overflow"))?;
        // N-row identity page-table: the `[0..num_pages)` identity block replicated
        // `decode_max_batch` times (stride=num_pages), so every row's topk reads
        // the same identity block (stride>0 required by the launch validator).
        let page_table_batch_h: Vec<i32> = (0..decode_max_batch)
            .flat_map(|_| (0..num_pages).map(|v| i32::try_from(v).expect("page table fits i32")))
            .collect();
        Ok(Self {
            cache_locs: ctx
                .stream
                .clone_htod(&cache_locs_h)
                .map_err(|e| anyhow!("DSv4 official DSA cache loc upload failed: {e}"))?,
            q_fp8: ctx
                .stream
                .alloc_zeros::<u8>(q_elems)
                .map_err(|e| anyhow!("DSv4 official DSA q fp8 alloc failed: {e}"))?,
            weights: ctx
                .stream
                .alloc_zeros::<f32>(query_tile * config.index_n_heads)
                .map_err(|e| anyhow!("DSv4 official DSA weights alloc failed: {e}"))?,
            context_lens: ctx
                .stream
                .alloc_zeros::<i32>(query_tile)
                .map_err(|e| anyhow!("DSv4 official DSA context lens alloc failed: {e}"))?,
            positions: ctx
                .stream
                .alloc_zeros::<i32>(query_tile)
                .map_err(|e| anyhow!("DSv4 official DSA positions alloc failed: {e}"))?,
            page_table_identity: ctx
                .stream
                .clone_htod(&page_table_h)
                .map_err(|e| anyhow!("DSv4 official DSA page table upload failed: {e}"))?,
            freqs_cis: ctx
                .stream
                .clone_htod(&freqs_cis_h)
                .map_err(|e| anyhow!("DSv4 official DSA freqs_cis upload failed: {e}"))?,
            sched_meta: ctx
                .stream
                .alloc_zeros::<i32>((num_sms + 1) * 2)
                .map_err(|e| anyhow!("DSv4 official DSA sched meta alloc failed: {e}"))?,
            logits: ctx
                .stream
                .alloc_zeros::<f32>(logits_elems)
                .map_err(|e| anyhow!("DSv4 official DSA logits alloc failed: {e}"))?,
            raw_indices: ctx
                .stream
                .alloc_zeros::<i32>(query_chunk * config.index_topk)
                .map_err(|e| anyhow!("DSv4 official DSA raw indices alloc failed: {e}"))?,
            decode_max_batch,
            q_fp8_batch: ctx
                .stream
                .alloc_zeros::<u8>(q_fp8_batch_elems)
                .map_err(|e| anyhow!("DSv4 official DSA batched q fp8 alloc failed: {e}"))?,
            weights_batch: ctx
                .stream
                .alloc_zeros::<f32>(decode_max_batch * config.index_n_heads)
                .map_err(|e| anyhow!("DSv4 official DSA batched weights alloc failed: {e}"))?,
            context_lens_batch: ctx
                .stream
                .alloc_zeros::<i32>(decode_max_batch)
                .map_err(|e| anyhow!("DSv4 official DSA batched context lens alloc failed: {e}"))?,
            positions_batch: ctx
                .stream
                .alloc_zeros::<i32>(decode_max_batch)
                .map_err(|e| anyhow!("DSv4 official DSA batched positions alloc failed: {e}"))?,
            logits_batch: ctx
                .stream
                .alloc_zeros::<f32>(logits_batch_elems)
                .map_err(|e| anyhow!("DSv4 official DSA batched logits alloc failed: {e}"))?,
            raw_indices_batch: ctx
                .stream
                .alloc_zeros::<i32>(decode_max_batch * config.index_topk)
                .map_err(|e| anyhow!("DSv4 official DSA batched raw indices alloc failed: {e}"))?,
            block_table_batch: ctx
                .stream
                .alloc_zeros::<i32>(block_table_batch_elems)
                .map_err(|e| anyhow!("DSv4 official DSA batched block table alloc failed: {e}"))?,
            page_table_identity_batch: ctx.stream.clone_htod(&page_table_batch_h).map_err(|e| {
                anyhow!("DSv4 official DSA batched page table identity upload failed: {e}")
            })?,
            max_tokens,
            query_tile,
            query_chunk,
            compressed_capacity,
            num_pages,
            num_heads: config.index_n_heads,
            head_dim: config.index_head_dim,
            logits_stride,
            num_sms,
        })
    }

    /// Exact requested device bytes owned by this ONE model-wide shared DSA
    /// selector scratch: Σ over its `CudaSlice` fields (single-forward query-tile
    /// buffers + the N-row batched-decode buffers). (The standalone
    /// [`dsv4_dsa_shared_scratch_bytes`] predicts the batch-INDEPENDENT part for
    /// the KV budget; the batched part is folded into the per-slot budget term
    /// since it scales with `num_slots`.)
    #[allow(dead_code)]
    pub(crate) fn device_bytes(&self) -> usize {
        let i32_sz = std::mem::size_of::<i32>();
        let f32_sz = std::mem::size_of::<f32>();
        let i64_sz = std::mem::size_of::<i64>();
        self.cache_locs.len() * i64_sz
            + self.q_fp8.len() // u8
            + self.weights.len() * f32_sz
            + self.context_lens.len() * i32_sz
            + self.positions.len() * i32_sz
            + self.page_table_identity.len() * i32_sz
            + self.freqs_cis.len() * f32_sz
            + self.sched_meta.len() * i32_sz
            + self.logits.len() * f32_sz
            + self.raw_indices.len() * i32_sz
            + self.q_fp8_batch.len() // u8
            + self.weights_batch.len() * f32_sz
            + self.context_lens_batch.len() * i32_sz
            + self.positions_batch.len() * i32_sz
            + self.logits_batch.len() * f32_sz
            + self.raw_indices_batch.len() * i32_sz
            + self.block_table_batch.len() * i32_sz
            + self.page_table_identity_batch.len() * i32_sz
    }
}

/// Device bytes of the ONE [`Dsv4DsaSharedScratch`] (per model, NOT per slot).
/// MUST mirror [`Dsv4DsaSharedScratch::new`]'s allocations (kept adjacent so
/// drift is visible). Feeds `Dsv4Model::kv_budget_plan` as a one-off
/// subtraction from the budget.
pub(crate) fn dsv4_dsa_shared_scratch_bytes(
    config: &DeepSeekV4Config,
    compress_ratio: usize,
    max_seq_len: usize,
) -> usize {
    let cc = max_seq_len.div_ceil(compress_ratio.max(1)).max(1);
    let num_pages = cc.div_ceil(64).max(1);
    let query_tile = DSV4_DSA_PREFILL_QUERY_TILE.min(max_seq_len.max(1));
    let query_chunk = DSV4_PREFILL_QUERY_CHUNK.min(max_seq_len.max(1));
    let logits_stride = cc.div_ceil(256) * 256;
    let logits = query_tile.saturating_mul(logits_stride).saturating_mul(4);
    let cache_locs = cc.saturating_mul(8);
    let q_fp8 = query_tile
        .saturating_mul(config.index_n_heads)
        .saturating_mul(config.index_head_dim);
    let weights = query_tile
        .saturating_mul(config.index_n_heads)
        .saturating_mul(4);
    let lens_positions = query_tile.saturating_mul(8);
    let page_table = query_tile.saturating_mul(num_pages).saturating_mul(4);
    // freqs_cis covers max_tokens positions x rope dim (f32).
    let freqs_cis = max_seq_len
        .saturating_mul(config.qk_rope_head_dim)
        .saturating_mul(4);
    let raw_indices = query_chunk
        .saturating_mul(config.index_topk)
        .saturating_mul(4);
    logits
        .saturating_add(cache_locs)
        .saturating_add(q_fp8)
        .saturating_add(weights)
        .saturating_add(lens_positions)
        .saturating_add(page_table)
        .saturating_add(freqs_cis)
        .saturating_add(raw_indices)
}

/// Per-SLOT device bytes of the N-row batched-decode scratch inside the ONE
/// [`Dsv4DsaSharedScratch`] (the buffers sized by `decode_max_batch == num_slots`).
/// MUST mirror [`Dsv4DsaSharedScratch::new`]'s `*_batch` allocations. This is a
/// per-slot term (NOT a fixed subtraction): `decode_max_batch == num_slots`, so
/// the total batched-scratch cost is `num_slots * (this)`. It is fed into
/// `kv_budget_plan`'s per-slot budget rather than `dsv4_dsa_shared_scratch_bytes`
/// — putting it in the fixed subtraction would be CIRCULAR (the budget computes
/// `num_slots` FROM that subtraction).
pub(crate) fn dsv4_dsa_batched_scratch_bytes_per_slot(
    config: &DeepSeekV4Config,
    compress_ratio: usize,
    max_seq_len: usize,
) -> usize {
    let cc = max_seq_len.div_ceil(compress_ratio.max(1)).max(1);
    let num_pages = cc.div_ceil(64).max(1);
    let logits_stride = cc.div_ceil(256) * 256;
    // q_fp8_batch (u8) + weights_batch (f32) + context_lens_batch (i32) +
    // positions_batch (i32) + logits_batch (f32) + raw_indices_batch (i32) +
    // block_table_batch (i32) + page_table_identity_batch (i32), all per ONE row.
    let q_fp8 = config.index_n_heads.saturating_mul(config.index_head_dim);
    let weights = config.index_n_heads.saturating_mul(4);
    let context_lens = 4usize;
    let positions = 4usize;
    let logits = logits_stride.saturating_mul(4);
    let raw_indices = config.index_topk.saturating_mul(4);
    let block_table = num_pages.saturating_mul(4);
    let page_table_identity = num_pages.saturating_mul(4);
    q_fp8
        .saturating_add(weights)
        .saturating_add(context_lens)
        .saturating_add(positions)
        .saturating_add(logits)
        .saturating_add(raw_indices)
        .saturating_add(block_table)
        .saturating_add(page_table_identity)
}

/// Device bytes of ONE per-(slot, CSA-layer) [`Dsv4DsaOfficialState`] — the
/// transient `rotated_keys` staging buffer. Feeds the per-slot term of
/// `Dsv4Model::kv_budget_plan`.
pub(crate) fn dsv4_dsa_rotated_keys_bytes(
    config: &DeepSeekV4Config,
    compress_ratio: usize,
    max_seq_len: usize,
) -> usize {
    let cc = max_seq_len.div_ceil(compress_ratio.max(1)).max(1);
    dsv4_dsa_rotated_ring_rows(cc)
        .saturating_mul(config.index_head_dim)
        .saturating_mul(2)
}

/// Transient depth (rows) of the DSA Hadamard-rotated-key staging buffer.
/// Rotated keys are drain-immediate: Hadamard writes the per-forward delta and
/// `fused_store_index_k_cache` reads it into the FP8 `dsa_key_cache` in the SAME
/// forward, so only the live delta lives here — NOT the full history (that lives
/// in the FP8 cache the paged-MQA-logits kernel reads). Bounded by the indexer
/// staging ring, which already caps `newly_packed`.
pub(super) fn dsv4_dsa_rotated_ring_rows(compressed_capacity: usize) -> usize {
    DSV4_INDEXER_STAGING_RING_ROWS.min(compressed_capacity.max(1))
}

pub(crate) fn dsv4_dsa_key_cache_bytes(
    config: &DeepSeekV4Config,
    compress_ratio: usize,
    max_seq_len: usize,
) -> Result<usize> {
    let compressed_capacity = max_seq_len.div_ceil(compress_ratio).max(1);
    let page_size = 64usize;
    let num_pages = compressed_capacity.div_ceil(page_size).max(1);
    num_pages
        .checked_mul(page_size * (config.index_head_dim + std::mem::size_of::<f32>()))
        .ok_or_else(|| anyhow!("DSv4 official DSA key cache size overflow"))
}

pub(super) fn dsv4_dsa_freqs_cis_real(
    config: &DeepSeekV4Config,
    compress_ratio: usize,
    max_seq_len: usize,
) -> Result<Vec<f32>> {
    ensure!(
        config.qk_rope_head_dim.is_multiple_of(2),
        "DSv4 official DSA RoPE dim {} must be even",
        config.qk_rope_head_dim
    );
    let dim = config.qk_rope_head_dim;
    let half = dim / 2;
    let base = if compress_ratio > 0 {
        config.compress_rope_theta
    } else {
        config.rope_theta
    } as f64;
    let original_seq_len = if compress_ratio > 0 {
        config.rope_parameters.original_max_position_embeddings
    } else {
        0
    };
    let factor = config.rope_parameters.factor as f64;
    let beta_fast = config.rope_parameters.beta_fast as f64;
    let beta_slow = config.rope_parameters.beta_slow as f64;
    let inv_freq: Vec<f64> = (0..half)
        .map(|pair| {
            let mut freq = 1.0f64 / base.powf((2 * pair) as f64 / dim as f64);
            if original_seq_len > 0 {
                let find_correction_dim = |num_rotations: f64| -> f64 {
                    dim as f64
                        * ((original_seq_len as f64 / (num_rotations * 2.0 * std::f64::consts::PI))
                            .ln())
                        / (2.0 * base.ln())
                };
                let low = find_correction_dim(beta_fast).floor().max(0.0);
                let high = find_correction_dim(beta_slow).ceil().min((dim - 1) as f64);
                let mut high_adj = high;
                if (low - high_adj).abs() < f64::EPSILON {
                    high_adj += 0.001;
                }
                let ramp = ((pair as f64 - low) / (high_adj - low)).clamp(0.0, 1.0);
                let smooth = 1.0 - ramp;
                freq = freq / factor * (1.0 - smooth) + freq * smooth;
            }
            freq
        })
        .collect();

    let mut out = vec![0.0f32; max_seq_len * dim];
    for pos in 0..max_seq_len {
        for pair in 0..half {
            let theta = pos as f64 * inv_freq[pair];
            out[pos * dim + 2 * pair] = theta.cos() as f32;
            out[pos * dim + 2 * pair + 1] = theta.sin() as f32;
        }
    }
    Ok(out)
}

pub(crate) struct Dsv4LayerAttentionState {
    pub(super) sw_window_cache: CudaSlice<half::bf16>,
    pub(super) compressor: Option<Dsv4CompressorState>,
    pub(super) indexer: Option<Dsv4CompressorState>,
    pub(crate) flashmla: Option<Dsv4FlashMlaDecodeState>,
    pub(super) fused_wqkv: Option<Dsv4FusedWqkvDecodeScratch>,
    pub(super) dsa_official: Option<Dsv4DsaOfficialState>,
}

/// Per-layer one-slot snapshot of speculative-verify ring writes. The verify
/// forward can write BF16 SW-ring slots and FP8 FlashMLA ring bytes. Only the
/// boundary slot at `start_pos` must be protected; accepted rows are re-forwarded
/// and rejected rows are discarded after truncate. Buffers are allocated once at
/// slot construction and reused by D2D copy.
pub(crate) struct Dsv4SpecRingSnapshot {
    /// One BF16 SW ring slot.
    pub(super) sw_slots: CudaSlice<half::bf16>,
    /// One FP8 ring slot (data+scale); `None` when this layer has no FlashMLA/FP8 ring.
    pub(super) fp8_slots: Option<CudaSlice<u8>>,
    /// `flash.fp8_kv_comp_packed_rows` captured once pre-verify; restored on reject.
    pub(super) fp8_packed_rows_before: Option<usize>,
    /// `flash.fp8_kv_sw_bootstrapped` captured once pre-verify; restored on
    /// reject so the next decode re-bootstrap decision matches the restored
    /// bytes.
    pub(super) fp8_bootstrapped_before: Option<bool>,
    /// Layout metadata used by `fp8_sw_offsets`.
    pub(super) head_dim: usize,
    pub(super) sliding_window: usize,
    pub(super) fp8_page_block_size: usize,
    pub(super) fp8_token_data_bytes: usize,
    pub(super) fp8_scale_bytes: usize,
    pub(super) fp8_bytes_per_token: usize,
    /// Max draft depth this snapshot accepts for stale-window checks.
    pub(super) max_depth: usize,
    /// Capture-time `start_pos`/`depth`, asserted in restore so a stale snapshot
    /// can never be replayed against a different verify window.
    pub(super) captured_start_pos: usize,
    pub(super) captured_depth: usize,
}

impl Dsv4SpecRingSnapshot {
    /// Exact requested device bytes owned by this per-(slot,layer) spec-ring
    /// snapshot: the bf16 `sw_slots` + the optional `u8` `fp8_slots`.
    #[allow(dead_code)]
    pub(crate) fn device_bytes(&self) -> usize {
        self.sw_slots.len() * std::mem::size_of::<half::bf16>()
            + self.fp8_slots.as_ref().map_or(0, |s| s.len())
    }

    /// STATIC predictor of ONE layer's snapshot `device_bytes` — MUST mirror
    /// `alloc_spec_ring_snapshot`. `sw_slots` is `[head_dim]` bf16; the `[bytes_per_token]`
    /// u8 `fp8_slots` exists iff the layer has a FlashMLA decode state (the uniform
    /// `cuda_kernels::HAS_FLASHMLA` gate). Feeds `per_slot_device_bytes`.
    pub(crate) fn device_bytes_for(
        config: &DeepSeekV4Config,
        kv_arena: &Dsv4MlaKvArena,
    ) -> Result<usize> {
        let bf16 = std::mem::size_of::<half::bf16>();
        let fp8 = if cuda_kernels::HAS_FLASHMLA {
            kv_arena.bytes_per_token
        } else {
            0
        };
        Ok(config.head_dim * bf16 + fp8)
    }

    /// `(logical ring block, data offset in block, scale offset in block)` for
    /// one draft token's FP8 SW ring slot. The caller translates the logical
    /// block through the slot's page table. Returns `None` when this layer has no
    /// FP8 ring.
    pub(super) fn fp8_sw_offsets(&self, draft_abs_pos: usize) -> Option<(usize, usize, usize)> {
        self.fp8_slots.as_ref()?;
        let ring_idx = draft_abs_pos % self.sliding_window;
        let block_id = ring_idx / self.fp8_page_block_size;
        let row = ring_idx % self.fp8_page_block_size;
        let data_in_block = row * self.fp8_token_data_bytes;
        let scale_in_block =
            self.fp8_page_block_size * self.fp8_token_data_bytes + row * self.fp8_scale_bytes;
        Some((block_id, data_in_block, scale_in_block))
    }
}

/// Host image of one layer's attention state for whole-slot spill. Every
/// device buffer that carries cross-call state is captured byte-for-byte.
/// Scratch-only fields (`fused_wqkv`) are NOT captured — they're overwritten
/// before read each forward.
pub(crate) struct Dsv4LayerAttentionImage {
    pub(crate) sw_window_cache: Vec<half::bf16>,
    pub(crate) compressor: Option<Dsv4CompressorImage>,
    pub(crate) indexer: Option<Dsv4CompressorImage>,
    pub(crate) flashmla: Option<Dsv4FlashMlaImage>,
    pub(crate) dsa_official: Option<Dsv4DsaImage>,
}

#[derive(Default)]
pub(crate) struct Dsv4CompressorImage {
    pub(crate) pending_kv: Vec<half::bf16>,
    pub(crate) pending_score: Vec<half::bf16>,
    pub(crate) prev_overlap_kv: Vec<half::bf16>,
    pub(crate) prev_overlap_score: Vec<half::bf16>,
    pub(crate) compressed: Vec<half::bf16>,
    pub(crate) compressed_seq_len: usize,
    pub(crate) compressed_capacity: usize,
    pub(crate) ring_rows: usize,
    pub(crate) fp32_pending_kv: Vec<f32>,
    pub(crate) fp32_pending_score: Vec<f32>,
    pub(crate) fp32_prev_kv: Vec<f32>,
    pub(crate) fp32_prev_score: Vec<f32>,
    pub(crate) fp32_carry_stale: bool,
}

#[derive(Default)]
pub(crate) struct Dsv4FlashMlaImage {
    pub(crate) fp8_kv_pool_len: usize,
    pub(crate) sw_blocks: usize,
    pub(crate) comp_blocks: usize,
    pub(crate) max_compressed_keys: usize,
    pub(crate) topk_unified: usize,
    pub(crate) page_block_size: usize,
    pub(crate) fp8_kv_sw_bootstrapped: bool,
    pub(crate) fp8_kv_comp_packed_rows: usize,
    pub(crate) topk_length: Vec<i32>,
    pub(crate) sched_meta: Vec<i32>,
    pub(crate) num_splits: Vec<i32>,
    pub(crate) num_sm_parts: i32,
    pub(crate) fixed_overhead_num_blocks: i32,
    pub(crate) block_size_topk: i32,
    pub(crate) device_page_table: Vec<i32>,
}

#[derive(Default)]
pub(crate) struct Dsv4DsaImage {
    pub(crate) key_cache_len: usize,
    pub(crate) rotated_keys: Vec<half::bf16>,
    pub(crate) packed_rows: usize,
}

impl Dsv4LayerAttentionState {
    /// This layer-state's indexer compressed-key row count (`compressed.seq_len`),
    /// or `None` if the layer has no indexer. Used by the batched P1b cache-write
    /// pre-pass to read `indexer_rows_after` (P1a already advanced it).
    pub(crate) fn indexer_compressed_seq_len(&self) -> Option<usize> {
        self.indexer.as_ref().map(|s| s.compressed.seq_len)
    }

    /// Whether this layer's FP8 compressed band lags the committed compressed
    /// rows by MORE than the current step's delta — i.e. a post-restore (or
    /// post-prefill) bulk gap that only `flashmla_decode_pack_row`'s
    /// `[packed_rows, seq_len)` bulk rebuild can close. The batched decode
    /// pack (op "c") packs ONLY this step's row, so a gapped layer must run
    /// per-row this tick (codex R3; true batched gap-fill is future work).
    pub(crate) fn flashmla_comp_bulk_gap(&self) -> bool {
        match (&self.compressor, &self.flashmla) {
            (Some(c), Some(f)) => c.compressed.seq_len > f.fp8_kv_comp_packed_rows + 1,
            _ => false,
        }
    }

    /// This layer-state's official-DSA slot index, or `None` if there is no
    /// per-slot DSA state. Constant for the slot's lifetime — used to lazy-init
    /// the graph CSA select's persistent n=1 device-meta slot-id buffer.
    pub(crate) fn dsa_official_slot_idx(&self) -> Option<usize> {
        self.dsa_official.as_ref().map(|s| s.slot_idx)
    }

    /// This layer-state's indexer compressed-key CAPACITY (the constant the graph
    /// CSA select needs as `key_count` — the on-device `min(abs_pos/ratio,
    /// key_count)` recovers the live causal length). `None` if no indexer.
    pub(crate) fn indexer_compressed_capacity(&self) -> Option<usize> {
        self.indexer.as_ref().map(|s| s.compressed_capacity())
    }

    /// PHASE B (#60) split borrow for the batched decode pack loop: this slot's
    /// FlashMLA decode arena (`&mut`), the bf16 SW ring (`&`), and — for HCA —
    /// the compressed-key pool (`&`, disjoint field, `None` for SW). Errors if
    /// the FlashMLA arena is absent (the lane only runs with FlashMLA decode on).
    pub(crate) fn flashmla_pack_borrow(
        &mut self,
        want_compressed: bool,
    ) -> Result<(
        &mut Dsv4FlashMlaDecodeState,
        &CudaSlice<half::bf16>,
        Option<&HiddenStates>,
    )> {
        let compressed = if want_compressed {
            Some(
                &self
                    .compressor
                    .as_ref()
                    .ok_or_else(|| anyhow!("DSv4 batched HCA pack: compressor state missing"))?
                    .compressed,
            )
        } else {
            None
        };
        let flash = self
            .flashmla
            .as_mut()
            .ok_or_else(|| anyhow!("DSv4 batched decode pack: FlashMLA arena missing"))?;
        Ok((flash, &self.sw_window_cache, compressed))
    }

    /// PHASE B (#60) mutable borrow of the bf16 SW ring for the batched decode
    /// finish loop (the inverse-rope tail's window update writes it).
    pub(crate) fn sw_window_cache_mut(&mut self) -> &mut CudaSlice<half::bf16> {
        &mut self.sw_window_cache
    }

    pub(crate) fn new(
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        mode: DeepSeekV4AttentionMode,
        compress_ratio: usize,
        max_seq_len: usize,
        kv_arena: &Dsv4MlaKvArena,
        local_heads: usize,
        tp_world: usize,
        slot_idx: usize,
        pool: &Dsv4LayerKvLayout,
    ) -> Result<Self> {
        let sw_len = config.sliding_window * config.head_dim;
        ensure!(
            sw_len > 0,
            "DSv4 SW window cache len is zero (sliding_window={} head_dim={})",
            config.sliding_window,
            config.head_dim
        );
        let sw_window_cache = ctx
            .stream
            .alloc_zeros::<half::bf16>(sw_len)
            .map_err(|e| anyhow::anyhow!("DSv4 SW window cache alloc failed: {e}"))?;
        let overlap = compress_ratio < 16;
        // GLM SparseIndexed shares the indexer at ratio=1; the MAIN compressor
        // stays compressor-modes-only (CSA/HCA). SparseIndexed has no main key
        // compressor, so gating on has_compressor() yields None for GLM and
        // avoids Dsv4CompressorState::new(.., ratio=0, ..) (compress_ratio==0).
        let compressor = if mode.has_compressor() {
            Some(Dsv4CompressorState::new(
                ctx,
                config.head_dim,
                compress_ratio,
                overlap,
                max_seq_len,
                false,
            )?)
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
        let indexer = if mode.has_indexer() {
            Some(Dsv4CompressorState::new(
                ctx,
                config.index_head_dim,
                index_ratio,
                true,
                max_seq_len,
                mode == DeepSeekV4AttentionMode::SparseIndexed,
            )?)
        } else {
            None
        };
        let flashmla = if cuda_kernels::HAS_FLASHMLA {
            Some(Dsv4FlashMlaDecodeState::new(
                ctx,
                config,
                mode,
                compress_ratio,
                max_seq_len,
                kv_arena,
                local_heads,
                tp_world,
                slot_idx,
                pool,
            )?)
        } else {
            None
        };
        let fused_wqkv = if dsv4_fused_wqkv_decode_alloc_enabled()? {
            Some(Dsv4FusedWqkvDecodeScratch::new(ctx, config)?)
        } else {
            None
        };
        // GLM SparseIndexed shares the official DSA scratch at ratio=1; the
        // indexer gate widens to has_indexer(), compressor stays CSA/HCA-only.
        let dsa_official = if mode.has_indexer() {
            Some(Dsv4DsaOfficialState::new(
                ctx,
                config,
                index_ratio,
                max_seq_len,
                slot_idx,
                pool,
            )?)
        } else {
            None
        };
        Ok(Self {
            sw_window_cache,
            compressor,
            indexer,
            flashmla,
            fused_wqkv,
            dsa_official,
        })
    }

    /// Exact requested device bytes owned by this per-(slot,layer) attention
    /// state: `sw_window_cache` + each `Option` sub-struct's `device_bytes()`.
    /// `prefill_linear` was hoisted out to the adapter (#85) and is summed
    /// there; the FlashMLA FP8 KV pool pages live in `Dsv4LayerKvLayout`.
    #[allow(dead_code)]
    pub(crate) fn device_bytes(&self) -> usize {
        self.sw_window_cache.len() * std::mem::size_of::<half::bf16>()
            + self.compressor.as_ref().map_or(0, |s| s.device_bytes())
            + self.indexer.as_ref().map_or(0, |s| s.device_bytes())
            + self.flashmla.as_ref().map_or(0, |s| s.device_bytes())
            + self.fused_wqkv.as_ref().map_or(0, |s| s.device_bytes())
            + self.dsa_official.as_ref().map_or(0, |s| s.device_bytes())
    }

    /// STATIC predictor of ONE layer's `device_bytes` from config — MUST mirror
    /// `new` + `device_bytes` (kept adjacent so drift is visible). The `Option`
    /// sub-structs are gated exactly as `new` gates them (mode + the flashmla/
    /// fused-wqkv alloc flags). Feeds `Dsv4Model::per_slot_device_bytes` (the KV
    /// budget runs before any slot exists, so it cannot instantiate one).
    pub(crate) fn device_bytes_for(
        config: &DeepSeekV4Config,
        mode: DeepSeekV4AttentionMode,
        compress_ratio: usize,
        max_seq_len: usize,
    ) -> Result<usize> {
        let bf16 = std::mem::size_of::<half::bf16>();
        // sw_window_cache[sliding_window * head_dim] bf16 — always present.
        let mut total = config.sliding_window * config.head_dim * bf16;
        if mode.has_compressor() {
            total += Dsv4CompressorState::device_bytes_for(
                config.head_dim,
                compress_ratio,
                compress_ratio < 16,
                max_seq_len,
                false,
            );
        }
        let index_ratio = if mode == DeepSeekV4AttentionMode::SparseIndexed {
            1
        } else {
            compress_ratio
        };
        if mode.has_indexer() {
            total += Dsv4CompressorState::device_bytes_for(
                config.index_head_dim,
                index_ratio,
                true,
                max_seq_len,
                mode == DeepSeekV4AttentionMode::SparseIndexed,
            );
            // dsa_official: only the transient `rotated_keys` staging (its slot's
            // key-cache band lives in `Dsv4LayerKvLayout`, budgeted separately).
            total += dsv4_dsa_rotated_keys_bytes(config, index_ratio, max_seq_len);
        }
        if cuda_kernels::HAS_FLASHMLA {
            total += Dsv4FlashMlaDecodeState::device_bytes_estimate();
        }
        if super::dsv4_fused_wqkv_decode_alloc_enabled()? {
            total += Dsv4FusedWqkvDecodeScratch::device_bytes_for(config);
        }
        Ok(total)
    }

    /// Per-component byte breakdown for the VRAM ledger log.
    #[allow(dead_code)]
    pub(crate) fn device_bytes_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![
            (
                "sw_window_cache",
                self.sw_window_cache.len() * std::mem::size_of::<half::bf16>(),
            ),
            (
                "compressor",
                self.compressor.as_ref().map_or(0, |s| s.device_bytes()),
            ),
            (
                "indexer",
                self.indexer.as_ref().map_or(0, |s| s.device_bytes()),
            ),
            (
                "flashmla",
                self.flashmla.as_ref().map_or(0, |s| s.device_bytes()),
            ),
            (
                "fused_wqkv",
                self.fused_wqkv.as_ref().map_or(0, |s| s.device_bytes()),
            ),
            (
                "dsa_official",
                self.dsa_official.as_ref().map_or(0, |s| s.device_bytes()),
            ),
        ]
    }

    pub(crate) fn reset(
        &mut self,
        ctx: &DeviceContext,
        pool: &mut Dsv4LayerKvLayout,
    ) -> Result<()> {
        ctx.stream
            .memset_zeros(&mut self.sw_window_cache)
            .map_err(|e| anyhow::anyhow!("DSv4 SW window cache reset failed: {e}"))?;
        if let Some(compressor) = &mut self.compressor {
            compressor.reset(ctx)?;
        }
        if let Some(indexer) = &mut self.indexer {
            indexer.reset(ctx)?;
        }
        if let Some(flashmla) = &mut self.flashmla {
            flashmla.reset();
        }
        if let Some(dsa) = &mut self.dsa_official {
            dsa.reset(ctx, pool)?;
        }
        Ok(())
    }

    pub(crate) fn advance_decode_len(
        &mut self,
        mode: DeepSeekV4AttentionMode,
        ratio: usize,
        total_len: usize,
    ) {
        if mode == DeepSeekV4AttentionMode::SlidingWindow {
            return;
        }
        let compressed_rows = total_len / ratio;
        // Graph-replay ticks advance the bf16 carry with no compressor_forward
        // host call — mark the FP32 probe carry stale here (host bookkeeping
        // that runs every step); a redundant set on eager ticks is free.
        if let Some(compressor) = &mut self.compressor {
            compressor.compressed.seq_len = compressed_rows;
            compressor.fp32_carry_stale = true;
        }
        if let Some(indexer) = &mut self.indexer {
            indexer.compressed.seq_len = compressed_rows;
            indexer.fp32_carry_stale = true;
        }
    }

    /// This layer's FlashMLA pool slot index (the per-slot append cursor the
    /// H2 truncate must clamp), or `None` when this layer has no FlashMLA decode.
    pub(crate) fn flashmla_slot_idx(&self) -> Option<usize> {
        self.flashmla.as_ref().map(|f| f.slot_idx)
    }

    /// Re-sync this layer's persistent FlashMLA device page table from the
    /// host table. Dirty-bit driven (#154 Phase 0): called via
    /// `Dsv4SlotState::refresh_flashmla_device_page_tables` whenever
    /// `Dsv4KvAdapter::take_device_table_dirty` reports the slot's host band
    /// changed — never unconditionally per step (a CUDA-graph capture hazard,
    /// #8). No-op when this layer has no FlashMLA decode.
    pub(crate) fn refresh_flashmla_device_page_table(
        &mut self,
        ctx: &DeviceContext,
        pool: &Dsv4LayerKvLayout,
    ) -> Result<()> {
        if let Some(flash) = &mut self.flashmla {
            flash.refresh_device_page_table(ctx, pool)?;
        }
        Ok(())
    }

    pub(crate) fn truncate_decode_len(
        &mut self,
        mode: DeepSeekV4AttentionMode,
        ratio: usize,
        total_len: usize,
    ) {
        self.advance_decode_len(mode, ratio, total_len);
        // A rejected draft can advance the DSA packed-row counter past the
        // committed compressed-row count. Clamp it down so the next real decode
        // repacks the boundary row instead of reusing stale draft cache bytes.
        if let Some(dsa) = &mut self.dsa_official {
            let compressed_rows = total_len / ratio.max(1);
            dsa.packed_rows = dsa.packed_rows.min(compressed_rows);
        }
    }

    /// Allocate this layer's one-slot spec-ring snapshot at slot construction.
    /// It stores one BF16 SW-ring row and, when this layer has FlashMLA decode
    /// state, one FP8 ring row.
    pub(crate) fn alloc_spec_ring_snapshot(
        &self,
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        kv_arena: &Dsv4MlaKvArena,
        max_depth: usize,
    ) -> Result<Dsv4SpecRingSnapshot> {
        let slots = 1usize;
        // FP8 token data bytes = NoPE FP8 (1 B/dim) + RoPE bf16 (2 B/dim);
        // scale bytes = bytes_per_token - data bytes.
        let fp8_token_data_bytes = kv_arena
            .nope_dim
            .checked_add(kv_arena.rope_dim * std::mem::size_of::<half::bf16>())
            .ok_or_else(|| anyhow!("DSv4 spec-ring FP8 token data byte overflow"))?;
        ensure!(
            kv_arena.bytes_per_token >= fp8_token_data_bytes,
            "DSv4 spec-ring FP8 bytes/token {} smaller than token data bytes {}",
            kv_arena.bytes_per_token,
            fp8_token_data_bytes
        );
        let fp8_scale_bytes = kv_arena.bytes_per_token - fp8_token_data_bytes;
        Ok(Dsv4SpecRingSnapshot {
            sw_slots: ctx
                .stream
                .alloc_zeros::<half::bf16>(slots * config.head_dim)
                .map_err(|e| anyhow!("DSv4 spec-ring SW slots alloc failed: {e}"))?,
            fp8_slots: self
                .flashmla
                .as_ref()
                .map(|_| {
                    ctx.stream
                        .alloc_zeros::<u8>(slots * kv_arena.bytes_per_token)
                        .map_err(|e| anyhow!("DSv4 spec-ring FP8 slots alloc failed: {e}"))
                })
                .transpose()?,
            fp8_packed_rows_before: None,
            fp8_bootstrapped_before: None,
            head_dim: config.head_dim,
            sliding_window: config.sliding_window,
            fp8_page_block_size: kv_arena.page_block_size,
            fp8_token_data_bytes,
            fp8_scale_bytes,
            fp8_bytes_per_token: kv_arena.bytes_per_token,
            max_depth,
            captured_start_pos: 0,
            captured_depth: 0,
        })
    }

    /// Snapshot the one ring slot that can be observed after rollback. The
    /// accepted prefix is re-forwarded after truncate, so copying the whole
    /// depth window only burns commit time.
    pub(crate) fn capture_spec_rings(
        &self,
        ctx: &DeviceContext,
        pool: &mut Dsv4LayerKvLayout,
        snap: &mut Dsv4SpecRingSnapshot,
        start_pos: usize,
        depth: usize,
    ) -> Result<()> {
        ensure!(
            depth <= snap.max_depth,
            "DSv4 spec-ring capture depth {depth} exceeds snapshot max_depth {}",
            snap.max_depth
        );
        ensure!(
            snap.sliding_window > 0 && snap.head_dim > 0,
            "DSv4 spec-ring capture invalid shape sliding_window={} head_dim={}",
            snap.sliding_window,
            snap.head_dim
        );
        snap.fp8_packed_rows_before = self.flashmla.as_ref().map(|f| f.fp8_kv_comp_packed_rows);
        snap.fp8_bootstrapped_before = self.flashmla.as_ref().map(|f| f.fp8_kv_sw_bootstrapped);
        snap.capture_sw_slot(ctx, &self.sw_window_cache, 0, start_pos)?;
        if let Some(flash) = &self.flashmla {
            snap.capture_fp8_slot(ctx, pool, flash, 0, start_pos)?;
        }
        snap.captured_start_pos = start_pos;
        snap.captured_depth = depth;
        Ok(())
    }

    /// Restore the single protected boundary slot after truncate and before
    /// accepted-prefix re-forward. The FP8 slot id is read up front so page
    /// lookup does not hold a `flashmla` borrow across SW-ring restore.
    pub(crate) fn restore_spec_ring_tail(
        &mut self,
        ctx: &DeviceContext,
        pool: &mut Dsv4LayerKvLayout,
        snap: &Dsv4SpecRingSnapshot,
        start_pos: usize,
        accepted_n: usize,
        depth: usize,
    ) -> Result<()> {
        ensure!(
            snap.captured_start_pos == start_pos && snap.captured_depth == depth,
            "DSv4 spec-ring restore window mismatch captured=({},{}) restore=({start_pos},{depth})",
            snap.captured_start_pos,
            snap.captured_depth
        );
        ensure!(
            accepted_n <= depth,
            "DSv4 spec-ring restore accepted_n {accepted_n} exceeds depth {depth}"
        );
        // Read the FP8 slot id once up front so page lookup can run without
        // holding a `flashmla` borrow across the SW-ring restore.
        let fp8_slot_idx = self.flashmla.as_ref().map(|f| f.slot_idx);
        snap.restore_sw_slot(ctx, &mut self.sw_window_cache, 0, start_pos)?;
        if let Some(slot_idx) = fp8_slot_idx {
            snap.restore_fp8_slot(ctx, pool, slot_idx, 0, start_pos)?;
        }
        if let Some(flash) = &mut self.flashmla {
            if let Some(rows) = snap.fp8_packed_rows_before {
                flash.fp8_kv_comp_packed_rows = rows;
            }
            // Restore the bootstrap flag too; otherwise the next decode may skip
            // the repack that should overwrite restored stale FP8 bytes.
            if let Some(bootstrapped) = snap.fp8_bootstrapped_before {
                flash.fp8_kv_sw_bootstrapped = bootstrapped;
            }
        }
        Ok(())
    }

    pub(crate) fn swap_out(
        &mut self,
        ctx: &DeviceContext,
        pool: &Dsv4LayerKvLayout,
    ) -> Result<Dsv4LayerAttentionImage> {
        let sw_window_cache = ctx
            .stream
            .clone_dtoh(&self.sw_window_cache)
            .map_err(|e| anyhow!("DSv4 swap sw_window_cache D2H failed: {e}"))?;
        let compressor = self
            .compressor
            .as_ref()
            .map(|c| c.swap_out(ctx))
            .transpose()?;
        let indexer = self.indexer.as_ref().map(|c| c.swap_out(ctx)).transpose()?;
        let flashmla = self
            .flashmla
            .as_ref()
            .map(|f| f.swap_out(ctx, pool))
            .transpose()?;
        let dsa_official = self
            .dsa_official
            .as_ref()
            .map(|d| d.swap_out(ctx))
            .transpose()?;
        Ok(Dsv4LayerAttentionImage {
            sw_window_cache,
            compressor,
            indexer,
            flashmla,
            dsa_official,
        })
    }

    pub(crate) fn swap_in(
        &mut self,
        ctx: &DeviceContext,
        pool: &mut Dsv4LayerKvLayout,
        image: &Dsv4LayerAttentionImage,
    ) -> Result<()> {
        ctx.stream
            .memcpy_htod(&image.sw_window_cache, &mut self.sw_window_cache)
            .map_err(|e| anyhow!("DSv4 swap sw_window_cache H2D failed: {e}"))?;
        if let (Some(dst), Some(src)) = (&mut self.compressor, &image.compressor) {
            dst.swap_in(ctx, src)?;
        }
        if let (Some(dst), Some(src)) = (&mut self.indexer, &image.indexer) {
            dst.swap_in(ctx, src)?;
        }
        if let (Some(dst), Some(src)) = (&mut self.flashmla, &image.flashmla) {
            dst.swap_in(ctx, pool, src)?;
        }
        if let (Some(dst), Some(src)) = (&mut self.dsa_official, &image.dsa_official) {
            dst.swap_in(ctx, src)?;
        }
        Ok(())
    }
}

impl Dsv4SpecRingSnapshot {
    /// D2D the SW ring slot for `draft_abs_pos` into snapshot slot `i`.
    pub(super) fn capture_sw_slot(
        &mut self,
        ctx: &DeviceContext,
        sw_window_cache: &CudaSlice<half::bf16>,
        i: usize,
        draft_abs_pos: usize,
    ) -> Result<()> {
        ensure!(
            self.sliding_window > 0 && self.head_dim > 0,
            "DSv4 spec-ring SW snapshot has invalid shape sliding_window={} head_dim={}",
            self.sliding_window,
            self.head_dim
        );
        let ring_idx = draft_abs_pos % self.sliding_window;
        let src_offset = ring_idx * self.head_dim;
        let dst_offset = i * self.head_dim;
        ensure!(
            src_offset + self.head_dim <= sw_window_cache.len()
                && dst_offset + self.head_dim <= self.sw_slots.len(),
            "DSv4 spec-ring SW slot out of range src={} dst={} head_dim={} cache_len={} slots_len={}",
            src_offset,
            dst_offset,
            self.head_dim,
            sw_window_cache.len(),
            self.sw_slots.len()
        );
        let src = sw_window_cache.slice(src_offset..src_offset + self.head_dim);
        let mut dst = self
            .sw_slots
            .slice_mut(dst_offset..dst_offset + self.head_dim);
        ctx.stream
            .memcpy_dtod(&src, &mut dst)
            .map_err(|e| anyhow!("DSv4 spec-ring SW slot D2D snapshot failed: {e}"))?;
        Ok(())
    }

    /// Restore the SW ring slot for `draft_abs_pos` from snapshot slot `i`.
    pub(super) fn restore_sw_slot(
        &self,
        ctx: &DeviceContext,
        sw_window_cache: &mut CudaSlice<half::bf16>,
        i: usize,
        draft_abs_pos: usize,
    ) -> Result<()> {
        let ring_idx = draft_abs_pos % self.sliding_window;
        let dst_offset = ring_idx * self.head_dim;
        let src_offset = i * self.head_dim;
        ensure!(
            dst_offset + self.head_dim <= sw_window_cache.len()
                && src_offset + self.head_dim <= self.sw_slots.len(),
            "DSv4 spec-ring SW restore out of range src={} dst={} head_dim={} cache_len={} slots_len={}",
            src_offset,
            dst_offset,
            self.head_dim,
            sw_window_cache.len(),
            self.sw_slots.len()
        );
        let src = self.sw_slots.slice(src_offset..src_offset + self.head_dim);
        let mut dst = sw_window_cache.slice_mut(dst_offset..dst_offset + self.head_dim);
        ctx.stream
            .memcpy_dtod(&src, &mut dst)
            .map_err(|e| anyhow!("DSv4 spec-ring SW slot D2D restore failed: {e}"))?;
        Ok(())
    }

    /// D2D the FP8 ring data+scale bytes for `draft_abs_pos` into snapshot slot
    /// `i`. Early-returns when this layer has no FP8 ring.
    pub(super) fn capture_fp8_slot(
        &mut self,
        ctx: &DeviceContext,
        pool: &mut Dsv4LayerKvLayout,
        flash: &Dsv4FlashMlaDecodeState,
        i: usize,
        draft_abs_pos: usize,
    ) -> Result<()> {
        // `fp8_sw_offsets` returns `None` exactly when `fp8_slots` is `None`,
        // so this early-return also guards the `as_mut().ok_or_else` below.
        let Some((logical_page, data_in_block, scale_in_block)) =
            self.fp8_sw_offsets(draft_abs_pos)
        else {
            return Ok(());
        };
        // The ring block's byte base is its physical pool page.
        let page = physical_page(pool.flashmla_page_table(flash.slot_idx)?, logical_page)?;
        let block_base = page as usize * (self.fp8_page_block_size * self.fp8_bytes_per_token);
        let data_offset = block_base + data_in_block;
        let scale_offset = block_base + scale_in_block;
        let pool_buf = pool.flashmla_pool_data()?;
        let slot_base = i * self.fp8_bytes_per_token;
        let slots = self
            .fp8_slots
            .as_mut()
            .ok_or_else(|| anyhow!("DSv4 spec-ring FP8 slots missing during capture"))?;
        ensure!(
            data_offset + self.fp8_token_data_bytes <= pool_buf.len()
                && scale_offset + self.fp8_scale_bytes <= pool_buf.len()
                && slot_base + self.fp8_bytes_per_token <= slots.len(),
            "DSv4 spec-ring FP8 slot out of range data={} scale={} pool_len={} slot_base={} slots_len={}",
            data_offset,
            scale_offset,
            pool_buf.len(),
            slot_base,
            slots.len()
        );
        let src_data = pool_buf.slice(data_offset..data_offset + self.fp8_token_data_bytes);
        let mut dst_data = slots.slice_mut(slot_base..slot_base + self.fp8_token_data_bytes);
        ctx.stream
            .memcpy_dtod(&src_data, &mut dst_data)
            .map_err(|e| anyhow!("DSv4 spec-ring FP8 data snapshot failed: {e}"))?;
        let src_scale = pool_buf.slice(scale_offset..scale_offset + self.fp8_scale_bytes);
        let mut dst_scale = slots
            .slice_mut(slot_base + self.fp8_token_data_bytes..slot_base + self.fp8_bytes_per_token);
        ctx.stream
            .memcpy_dtod(&src_scale, &mut dst_scale)
            .map_err(|e| anyhow!("DSv4 spec-ring FP8 scale snapshot failed: {e}"))?;
        Ok(())
    }

    /// Restore the FP8 ring data+scale bytes for `draft_abs_pos` from snapshot
    /// slot `i`. Early-returns when this layer has no FP8 ring. The caller
    /// restores `fp8_kv_comp_packed_rows` separately.
    pub(super) fn restore_fp8_slot(
        &self,
        ctx: &DeviceContext,
        pool: &mut Dsv4LayerKvLayout,
        slot_idx: usize,
        i: usize,
        draft_abs_pos: usize,
    ) -> Result<()> {
        let Some((logical_page, data_in_block, scale_in_block)) =
            self.fp8_sw_offsets(draft_abs_pos)
        else {
            return Ok(());
        };
        let Some(slots) = &self.fp8_slots else {
            return Ok(());
        };
        // Same physical-page translation as capture; the table is re-resolved
        // from the caller-provided slot id.
        let page = physical_page(pool.flashmla_page_table(slot_idx)?, logical_page)?;
        let block_base = page as usize * (self.fp8_page_block_size * self.fp8_bytes_per_token);
        let data_offset = block_base + data_in_block;
        let scale_offset = block_base + scale_in_block;
        let pool_buf = pool.flashmla_pool_data_mut()?;
        let slot_base = i * self.fp8_bytes_per_token;
        ensure!(
            data_offset + self.fp8_token_data_bytes <= pool_buf.len()
                && scale_offset + self.fp8_scale_bytes <= pool_buf.len()
                && slot_base + self.fp8_bytes_per_token <= slots.len(),
            "DSv4 spec-ring FP8 restore out of range data={} scale={} pool_len={} slot_base={} slots_len={}",
            data_offset,
            scale_offset,
            pool_buf.len(),
            slot_base,
            slots.len()
        );
        let src_data = slots.slice(slot_base..slot_base + self.fp8_token_data_bytes);
        let mut dst_data = pool_buf.slice_mut(data_offset..data_offset + self.fp8_token_data_bytes);
        ctx.stream
            .memcpy_dtod(&src_data, &mut dst_data)
            .map_err(|e| anyhow!("DSv4 spec-ring FP8 data restore failed: {e}"))?;
        let src_scale = slots
            .slice(slot_base + self.fp8_token_data_bytes..slot_base + self.fp8_bytes_per_token);
        let mut dst_scale = pool_buf.slice_mut(scale_offset..scale_offset + self.fp8_scale_bytes);
        ctx.stream
            .memcpy_dtod(&src_scale, &mut dst_scale)
            .map_err(|e| anyhow!("DSv4 spec-ring FP8 scale restore failed: {e}"))?;
        Ok(())
    }
}
