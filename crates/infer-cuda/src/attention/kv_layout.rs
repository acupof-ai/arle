use super::*;

pub(crate) const DSV4_PREFILL_QUERY_CHUNK: usize = 4096;

/// Probe/pending row width of a compressor state: both halves when `overlap`.
/// Single source for `Dsv4CompressorState::{new, device_bytes_for}` and
/// `dsv4_compressor_fp32_max_width` — these MUST agree or the shared FP32
/// scratch mis-sizes against its consumers.
pub(crate) fn dsv4_compressor_width(head_dim: usize, overlap: bool) -> usize {
    if overlap { 2 * head_dim } else { head_dim }
}
pub(crate) struct Dsv4CompressorState {
    pub(super) pending_kv: CudaSlice<half::bf16>,
    pub(super) pending_score: CudaSlice<half::bf16>,
    pub(super) prev_overlap_kv: CudaSlice<half::bf16>,
    pub(super) prev_overlap_score: CudaSlice<half::bf16>,
    pub(super) compressed: HiddenStates,
    pub(super) compressed_capacity: usize,
    pub(super) ring_rows: usize,
    // FP32 probe carry (pending partial-group + prev-overlap rows) — genuine
    // persistent per-state carry; the big per-call kv/score GEMM scratch is
    // the model-wide [`Dsv4CompressorFp32Scratch`].
    pub(super) fp32_pending_kv: CudaSlice<f32>,
    pub(super) fp32_pending_score: CudaSlice<f32>,
    pub(super) fp32_prev_kv: CudaSlice<f32>,
    pub(super) fp32_prev_score: CudaSlice<f32>,
    /// The bf16 carry advanced past the FP32 carry (decode-lane update, prefix
    /// restore, reset) — the next `compressor_fp32_probe` must reseed FP32
    /// from bf16 before reading pending/prev. Every bf16-carry writer sets it;
    /// only the probe clears it.
    pub(super) fp32_carry_stale: bool,
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
        let width = dsv4_compressor_width(head_dim, overlap);
        let compressed_capacity = max_seq_len.div_ceil(ratio).max(1);
        let ring_rows = if staging_ring {
            DSV4_INDEXER_STAGING_RING_ROWS.min(compressed_capacity)
        } else {
            compressed_capacity
        };
        let fp32_pending_kv = ctx
            .stream
            .alloc_zeros::<f32>(ratio * width)
            .map_err(|e| anyhow::anyhow!("DSv4 compressor FP32 pending_kv alloc failed: {e}"))?;
        let fp32_pending_score = ctx
            .stream
            .alloc_zeros::<f32>(ratio * width)
            .map_err(|e| anyhow::anyhow!("DSv4 compressor FP32 pending_score alloc failed: {e}"))?;
        let fp32_prev_kv = ctx
            .stream
            .alloc_zeros::<f32>(ratio * head_dim)
            .map_err(|e| anyhow::anyhow!("DSv4 compressor FP32 prev_kv alloc failed: {e}"))?;
        let fp32_prev_score = ctx
            .stream
            .alloc_zeros::<f32>(ratio * head_dim)
            .map_err(|e| anyhow::anyhow!("DSv4 compressor FP32 prev_score alloc failed: {e}"))?;
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
            compressed: HiddenStates::zeros(ctx, head_dim, ring_rows)?,
            compressed_capacity,
            ring_rows,
            fp32_pending_kv,
            fp32_pending_score,
            fp32_prev_kv,
            fp32_prev_score,
            fp32_carry_stale: false,
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
        // The FP32 carry still holds the previous occupant; the next probe
        // reseeds it from the zeroed bf16 carry.
        self.fp32_carry_stale = true;
        Ok(())
    }

    /// Raw mutable device pointers (as `u64`) for gathering into the
    /// batched-compressor-update pointer arrays: `(pending_kv, pending_score,
    /// prev_overlap_kv, prev_overlap_score, compressed)` — all THIS ROW's own
    /// per-slot buffers (#154: the shared pool was deleted with Route A).
    /// Resolved on `ctx.stream`; the returned `u64`s stay valid because the
    /// buffers are not reallocated for the rest of the forward (same
    /// single-stream discipline the per-row path uses). The cudarc borrow
    /// guards are dropped before returning so the caller can re-borrow
    /// `state` later.
    pub(crate) fn batched_update_ptrs(&mut self, ctx: &DeviceContext) -> (u64, u64, u64, u64, u64) {
        let (pkv, g0) = self.pending_kv.device_ptr_mut(&ctx.stream);
        let (psc, g1) = self.pending_score.device_ptr_mut(&ctx.stream);
        let (prkv, g2) = self.prev_overlap_kv.device_ptr_mut(&ctx.stream);
        let (prsc, g3) = self.prev_overlap_score.device_ptr_mut(&ctx.stream);
        let (comp, g4) = self.compressed.data.device_ptr_mut(&ctx.stream);
        drop(g0);
        drop(g1);
        drop(g2);
        drop(g3);
        drop(g4);
        (pkv, psc, prkv, prsc, comp)
    }

    #[allow(dead_code)]
    pub(crate) fn device_bytes(&self) -> usize {
        let bf16 = std::mem::size_of::<half::bf16>();
        let f32 = std::mem::size_of::<f32>();
        self.pending_kv.len() * bf16
            + self.pending_score.len() * bf16
            + self.prev_overlap_kv.len() * bf16
            + self.prev_overlap_score.len() * bf16
            + self.compressed.device_bytes()
            + self.fp32_pending_kv.len() * f32
            + self.fp32_pending_score.len() * f32
            + self.fp32_prev_kv.len() * f32
            + self.fp32_prev_score.len() * f32
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
        let width = dsv4_compressor_width(head_dim, overlap);
        let compressed_capacity = max_seq_len.div_ceil(ratio.max(1)).max(1);
        let ring_rows = if staging_ring {
            DSV4_INDEXER_STAGING_RING_ROWS.min(compressed_capacity)
        } else {
            compressed_capacity
        };
        let bf16_bytes = (2 * ratio * width + 2 * ratio * head_dim + head_dim * ring_rows) * bf16;
        let f32 = std::mem::size_of::<f32>();
        let fp32_bytes = (2 * ratio * width + 2 * ratio * head_dim) * f32;
        bf16_bytes + fp32_bytes
    }

    pub(crate) fn swap_out(
        &self,
        ctx: &DeviceContext,
    ) -> Result<crate::attention::Dsv4CompressorImage> {
        let pending_kv = ctx
            .stream
            .clone_dtoh(&self.pending_kv)
            .map_err(|e| anyhow!("DSv4 compressor swap pending_kv D2H failed: {e}"))?;
        let pending_score = ctx
            .stream
            .clone_dtoh(&self.pending_score)
            .map_err(|e| anyhow!("DSv4 compressor swap pending_score D2H failed: {e}"))?;
        let prev_overlap_kv = ctx
            .stream
            .clone_dtoh(&self.prev_overlap_kv)
            .map_err(|e| anyhow!("DSv4 compressor swap prev_overlap_kv D2H failed: {e}"))?;
        let prev_overlap_score = ctx
            .stream
            .clone_dtoh(&self.prev_overlap_score)
            .map_err(|e| anyhow!("DSv4 compressor swap prev_overlap_score D2H failed: {e}"))?;
        let valid = self.compressed.seq_len * self.compressed.hidden_dim;
        let compressed = if valid > 0 {
            ctx.stream
                .clone_dtoh(&self.compressed.data.slice(..valid))
                .map_err(|e| anyhow!("DSv4 compressor swap compressed D2H failed: {e}"))?
        } else {
            Vec::new()
        };
        let (fp32_pending_kv, fp32_pending_score, fp32_prev_kv, fp32_prev_score) = if self
            .fp32_carry_stale
        {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        } else {
            let fp32_pending_kv = ctx
                .stream
                .clone_dtoh(&self.fp32_pending_kv)
                .map_err(|e| anyhow!("DSv4 compressor swap fp32_pending_kv D2H failed: {e}"))?;
            let fp32_pending_score = ctx
                .stream
                .clone_dtoh(&self.fp32_pending_score)
                .map_err(|e| anyhow!("DSv4 compressor swap fp32_pending_score D2H failed: {e}"))?;
            let fp32_prev_kv = ctx
                .stream
                .clone_dtoh(&self.fp32_prev_kv)
                .map_err(|e| anyhow!("DSv4 compressor swap fp32_prev_kv D2H failed: {e}"))?;
            let fp32_prev_score = ctx
                .stream
                .clone_dtoh(&self.fp32_prev_score)
                .map_err(|e| anyhow!("DSv4 compressor swap fp32_prev_score D2H failed: {e}"))?;
            (
                fp32_pending_kv,
                fp32_pending_score,
                fp32_prev_kv,
                fp32_prev_score,
            )
        };
        Ok(crate::attention::Dsv4CompressorImage {
            pending_kv,
            pending_score,
            prev_overlap_kv,
            prev_overlap_score,
            compressed,
            compressed_seq_len: self.compressed.seq_len,
            fp32_pending_kv,
            fp32_pending_score,
            fp32_prev_kv,
            fp32_prev_score,
            fp32_carry_stale: self.fp32_carry_stale,
        })
    }

    pub(crate) fn swap_in(
        &mut self,
        ctx: &DeviceContext,
        image: &crate::attention::Dsv4CompressorImage,
    ) -> Result<()> {
        ctx.stream
            .memcpy_htod(&image.pending_kv, &mut self.pending_kv)
            .map_err(|e| anyhow!("DSv4 compressor swap pending_kv H2D failed: {e}"))?;
        ctx.stream
            .memcpy_htod(&image.pending_score, &mut self.pending_score)
            .map_err(|e| anyhow!("DSv4 compressor swap pending_score H2D failed: {e}"))?;
        ctx.stream
            .memcpy_htod(&image.prev_overlap_kv, &mut self.prev_overlap_kv)
            .map_err(|e| anyhow!("DSv4 compressor swap prev_overlap_kv H2D failed: {e}"))?;
        ctx.stream
            .memcpy_htod(&image.prev_overlap_score, &mut self.prev_overlap_score)
            .map_err(|e| anyhow!("DSv4 compressor swap prev_overlap_score H2D failed: {e}"))?;
        if !image.compressed.is_empty() {
            ctx.stream
                .memcpy_htod(
                    &image.compressed,
                    &mut self.compressed.data.slice_mut(..image.compressed.len()),
                )
                .map_err(|e| anyhow!("DSv4 compressor swap compressed H2D failed: {e}"))?;
        }
        self.compressed.seq_len = image.compressed_seq_len;
        if image.fp32_carry_stale {
            self.fp32_carry_stale = true;
        } else {
            ctx.stream
                .memcpy_htod(&image.fp32_pending_kv, &mut self.fp32_pending_kv)
                .map_err(|e| anyhow!("DSv4 compressor swap fp32_pending_kv H2D failed: {e}"))?;
            ctx.stream
                .memcpy_htod(&image.fp32_pending_score, &mut self.fp32_pending_score)
                .map_err(|e| anyhow!("DSv4 compressor swap fp32_pending_score H2D failed: {e}"))?;
            ctx.stream
                .memcpy_htod(&image.fp32_prev_kv, &mut self.fp32_prev_kv)
                .map_err(|e| anyhow!("DSv4 compressor swap fp32_prev_kv H2D failed: {e}"))?;
            ctx.stream
                .memcpy_htod(&image.fp32_prev_score, &mut self.fp32_prev_score)
                .map_err(|e| anyhow!("DSv4 compressor swap fp32_prev_score H2D failed: {e}"))?;
            self.fp32_carry_stale = false;
        }
        Ok(())
    }
}

/// Model-wide (one instance, NOT per-slot/per-layer) FP32 compressor-probe GEMM
/// scratch, hoisted off the per-(slot,layer) [`Dsv4CompressorState`] (#85 P3
/// pattern, like [`Dsv4FlashMlaDecodeScratch`]): `kv_raw`/`score_raw` are written
/// by the probe's two GEMMs and consumed inside the SAME `compressor_fp32_probe`
/// call — layers run sequentially and the engine runs one forward at a time, so
/// one instance sized `max_width × max_seq_len` serves every (slot, layer). The
/// per-slot copies inflated the ledger by 2×width×max_seq_len×4 B per compressor
/// state ("per_slot 9922MB" clamped 256 slots to 1).
pub(crate) struct Dsv4CompressorFp32Scratch {
    pub(super) kv_raw: CudaSlice<f32>,
    pub(super) score_raw: CudaSlice<f32>,
}

impl Dsv4CompressorFp32Scratch {
    pub(super) fn new(ctx: &DeviceContext, max_width: usize, max_seq_len: usize) -> Result<Self> {
        let kv_raw = ctx
            .stream
            .alloc_zeros::<f32>(max_width * max_seq_len)
            .map_err(|e| anyhow::anyhow!("DSv4 shared compressor FP32 kv_raw alloc failed: {e}"))?;
        let score_raw = ctx
            .stream
            .alloc_zeros::<f32>(max_width * max_seq_len)
            .map_err(|e| {
                anyhow::anyhow!("DSv4 shared compressor FP32 score_raw alloc failed: {e}")
            })?;
        Ok(Self { kv_raw, score_raw })
    }

    /// STATIC predictor of `device_bytes` — MUST mirror `new` (feeds
    /// `Dsv4Model::kv_budget_plan`'s fixed subtraction before the adapter exists).
    pub(crate) fn device_bytes_for(max_width: usize, max_seq_len: usize) -> usize {
        2 * max_width * max_seq_len * std::mem::size_of::<f32>()
    }

    pub(crate) fn device_bytes(&self) -> usize {
        (self.kv_raw.len() + self.score_raw.len()) * std::mem::size_of::<f32>()
    }
}

/// Max FP32-probe row width over every compressor `compressor_fp32_probe` can
/// run — main compressor on `has_compressor()` (overlap ⇔ cr<16), indexer
/// compressor on CSA only. 0 ⇔ no probing layer (no scratch).
pub(crate) fn dsv4_compressor_fp32_max_width(
    config: &DeepSeekV4Config,
    layers: impl IntoIterator<Item = (DeepSeekV4AttentionMode, usize)>,
) -> usize {
    layers
        .into_iter()
        .map(|(mode, compress_ratio)| {
            let main = if mode.has_compressor() {
                dsv4_compressor_width(config.head_dim, compress_ratio < 16)
            } else {
                0
            };
            // Narrower than has_indexer() on purpose: SparseIndexed (GLM) index
            // keys go through sparse_indexed_index_key_forward and never probe.
            let indexer = if mode == DeepSeekV4AttentionMode::CompressedSparse {
                dsv4_compressor_width(config.index_head_dim, true)
            } else {
                0
            };
            main.max(indexer)
        })
        .max()
        .unwrap_or(0)
}

pub(crate) struct Dsv4KvAdapter {
    pub(super) layers: Vec<Dsv4LayerKvLayout>,
    pub(super) num_slots: usize,
    pub(super) slot_epochs: Vec<Option<u64>>,
    /// Stream handle for `prepare_kv_batch`-time demand-page zeroing (#154
    /// Phase 3b) — the `ModelKvAdapter` trait signature carries no ctx.
    pub(super) ctx: DeviceContext,
    /// Shared comp token capacity the demand-paged pools were sized for
    /// (`kv_budget_plan`); the engine host pool mirrors `pool_tokens / page`
    /// as its admission page count.
    pub(super) flashmla_pool_tokens: usize,
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
    /// batched lane when `dsv4_flashmla_decode_enabled()`.
    pub(super) flashmla_batch: Option<Dsv4FlashMlaDecodeBatchScratch>,
    /// One shared SINGLE-ROW (`s_q = 1`) FlashMLA decode SCRATCH for ALL FlashMLA
    /// layers and slots (#85 P3). Hoisted out of the per-(slot,layer)
    /// `Dsv4FlashMlaDecodeState` — its 13 per-forward accumulator/staging buffers
    /// (`o_accum` dominant at ~33.7 MB/layer) carry NO cross-call/cross-slot state
    /// (overwritten before read each layer step, serial on `ctx.stream`), so one
    /// instance sized for the worst-case layer shape serves every (slot, layer).
    /// Gated by the SAME predicate as the per-slot state alloc
    /// (`cuda_kernels::HAS_FLASHMLA`); `None` ⇒ byte-identical to before.
    pub(super) flashmla_scratch: Option<Dsv4FlashMlaDecodeScratch>,
    /// One shared FP8 prefill DeepGEMM linear (quantize→GEMM) staging scratch for
    /// ALL layers and slots. Hoisted out of the per-slot `Dsv4LayerAttentionState`
    /// (it was the biggest single per-(slot,layer) offender, ~30 MB/layer): the
    /// scratch is M-chunk-bounded and fully overwritten from each chunk's
    /// activations before read, carrying NO cross-call/cross-slot state (the
    /// 2026-06 whole-slot-swap SCRATCH verdict). Sized by
    /// `config` + `max_seq_len` only (no layer-specific dimensions), so a single
    /// instance is valid for every layer and slot — DSv4 forward runs layers
    /// sequentially and the engine runs one forward at a time, so it is reused,
    /// never aliased concurrently (exactly like `dsa_shared`/`flashmla_batch`/
    /// decode-graph scratch). `None` when native DeepGEMM is unavailable.
    pub(super) prefill_linear: Option<Dsv4PrefillDeepGemmLinearScratch>,
    /// One shared FP32 compressor-probe GEMM scratch for ALL compressor layers
    /// and slots, hoisted off the per-(slot,layer) `Dsv4CompressorState` (its
    /// two `width×max_seq_len` f32 buffers collapsed the slot budget). `None`
    /// only when the model has no compressor layer.
    pub(super) compressor_fp32: Option<Dsv4CompressorFp32Scratch>,
    /// Per-slot "host FlashMLA band changed since last device-table sync" bit
    /// (#154 Phase 0). Set by the mirror/attach sites; consumed once per
    /// forward via [`Self::take_device_table_dirty`] to drive the
    /// graph-referenced device-table refresh.
    pub(super) device_table_dirty: Vec<bool>,
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
    /// this layer's shape) — every slot's MAX block-table length.
    pub(super) flashmla_slot_pages: usize,
    pub(super) flashmla_page_bytes: usize,
    /// Demand-paged band (#154 Phase 3b, `dsv4_flashmla_demand_paged`): comp
    /// pages come from the pool free list as the sequence grows; false = the
    /// V32 identity full band (its pack lane needs band-base contiguity).
    pub(super) flashmla_demand_paged: bool,
    /// Ring blocks at the band's logical head (`[0, sw_blocks)`).
    pub(super) flashmla_sw_blocks: usize,
    /// Tokens one comp page covers (`page_block_size × compress_ratio`);
    /// 0 = no comp region (SlidingWindow-only layer).
    pub(super) flashmla_comp_tokens_per_page: usize,
    pub(super) dsa_slot_bytes: usize,
    pub(super) num_slots: usize,
}

impl Dsv4LayerKvLayout {
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
        // Host tables carry only real pages (#154 Phase 0): non-empty, ≤ band.
        ensure!(
            (1..=band_pages).contains(&pool.page_indices(slot_idx).len()),
            "DSv4 FlashMLA slot {slot_idx} band not mirrored ({} pages, band capacity {band_pages}) before cursor advance",
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

    /// Band pages a demand-paged slot needs to hold `tokens`: the ring blocks
    /// plus `ceil(tokens / (page × ratio))` comp pages (floored at one comp
    /// page — the shape's `compressed_rows.max(1)` capacity floor), capped at
    /// the full band.
    pub(super) fn flashmla_band_pages_for(&self, tokens: usize) -> usize {
        let comp = if self.flashmla_comp_tokens_per_page == 0 {
            0
        } else {
            tokens.div_ceil(self.flashmla_comp_tokens_per_page).max(1)
        };
        (self.flashmla_sw_blocks + comp).min(self.flashmla_slot_pages)
    }

    /// Grow `slot_idx`'s demand-paged band to cover `tokens` (#154 Phase 3b):
    /// missing pages come from the layer pool's free list and are ZEROED
    /// before use (a recycled page carries a prior occupant's bytes; the ring
    /// bootstrap and comp readers assume zeroed tails). Returns whether the
    /// table changed (drives the device-table dirty bit). No-op (false) on
    /// identity (V32) layers — `mirror_band` owns their tables — and when the
    /// table already covers `tokens`. Exhaustion here is a HARD error: the
    /// engine's device-budget plan gate (`kv_device_fit`, #160 — parks
    /// over-admitted rows before submit) must make it unreachable.
    pub(crate) fn flashmla_ensure_band(
        &mut self,
        ctx: &DeviceContext,
        slot_idx: usize,
        tokens: usize,
        zero: bool,
    ) -> Result<bool> {
        if !self.flashmla_demand_paged {
            return Ok(false);
        }
        let needed = self.flashmla_band_pages_for(tokens);
        let pool = self.flashmla_pool_mut()?;
        let have = pool.page_indices(slot_idx).len();
        if have >= needed {
            return Ok(false);
        }
        let new_pages = pool.band_extend(slot_idx, needed - have).map_err(|e| {
            anyhow!(
                "DSv4 FlashMLA layer pool exhausted growing slot {slot_idx} to {tokens} tokens \
                 ({have}->{needed} pages) — the device-budget plan gate (#160) should make this \
                 unreachable: {e}"
            )
        })?;
        if zero {
            self.flashmla_pool_mut()?
                .zero_pages(ctx, &new_pages)
                .map_err(|e| anyhow!("DSv4 FlashMLA demand-page zero failed: {e}"))?;
        }
        Ok(true)
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
        // Demand-paged bands are claim-zeroed page-by-page (`ensure_band`);
        // a fresh occupant's table only ever holds pages zeroed for THIS
        // occupancy (free_slot empties it at request end), so the occupancy
        // whole-band zero is both redundant and a blocking H2D.
        if self.flashmla_demand_paged {
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

pub(crate) type LayerFlashMlaAndMlaDecode<'a> = (
    &'a mut Dsv4LayerKvLayout,
    Option<&'a mut Dsv4DsaSharedScratch>,
    Option<&'a mut Dsv4FlashMlaDecodeScratch>,
    Option<&'a mut Dsv4MlaDecodeGraphScratch>,
);

impl Dsv4KvAdapter {
    pub(crate) fn slot_epoch(&self, slot: usize) -> Option<u64> {
        self.slot_epochs.get(slot).copied().flatten()
    }

    pub(crate) fn new(
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        layer_specs: &[(DeepSeekV4AttentionMode, usize, usize)],
        max_seq_len: usize,
        kv_arena: &Dsv4MlaKvArena,
        tp_world: usize,
        num_slots: usize,
        pool_tokens: usize,
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
                    pool_tokens,
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
            Some(first_cr) => {
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
        // SAFETY: uninit device scratch; fully written before first read.
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
        // both MODEL1 (head_dim=512) and V32/GLM (head_dim=576) when the arena
        // is live; the batched shim now handles both dim mappings.
        // Build the per-layer FlashMLA decode shapes ONCE (shared by the batched
        // scratch and the new single-row shared scratch). Both gate on the same
        // `cuda_kernels::HAS_FLASHMLA` predicate as the per-slot state.
        let (flashmla_batch, flashmla_scratch) = if cuda_kernels::HAS_FLASHMLA {
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
            let batch = Some(Dsv4FlashMlaDecodeBatchScratch::new(
                ctx,
                config,
                num_slots,
                &layer_shapes,
            )?);
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
        let prefill_linear = if dsv4_deepgemm_enabled() {
            Some(Dsv4PrefillDeepGemmLinearScratch::new(
                ctx,
                config,
                max_seq_len,
            )?)
        } else {
            None
        };
        // ONE model-wide FP32 compressor-probe scratch (hoisted off the
        // per-(slot,layer) compressor state). max_width 0 ⇔ no compressor layer.
        let fp32_width = dsv4_compressor_fp32_max_width(
            config,
            layer_specs.iter().map(|&(mode, cr, _)| (mode, cr)),
        );
        let compressor_fp32 = (fp32_width > 0)
            .then(|| Dsv4CompressorFp32Scratch::new(ctx, fp32_width, max_seq_len))
            .transpose()?;
        Ok(Self {
            layers,
            num_slots,
            slot_epochs: vec![None; num_slots],
            ctx: ctx.clone(),
            flashmla_pool_tokens: pool_tokens,
            dsa_shared,
            moe_decode_shared,
            moe_tail_scratch,
            mla_decode,
            shared_expert_out,
            shared_expert_scratch,
            flashmla_batch,
            flashmla_scratch,
            prefill_linear,
            compressor_fp32,
            device_table_dirty: vec![false; num_slots],
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
            (
                "compressor_fp32",
                self.compressor_fp32
                    .as_ref()
                    .map_or(0, |s| s.device_bytes()),
            ),
        ]
    }

    /// Split-borrow accessor: one layer's KV layout, the model-wide shared DSA
    /// scratch, the model-wide single-row FlashMLA decode scratch, the
    /// model-wide shared FP8 prefill DeepGEMM linear scratch, AND the model-wide
    /// FP32 compressor-probe scratch (all disjoint fields, so all can be `&mut`
    /// at once). The FlashMLA scratch is `None` when FlashMLA decode is disabled
    /// at the build/override level (same gate as the per-slot state); the
    /// prefill scratch is `None` when native DeepGEMM is unavailable; the FP32
    /// scratch is `None` when the model has no compressor layer.
    #[allow(clippy::type_complexity)]
    pub(crate) fn layer_and_dsa_shared_mut(
        &mut self,
        layer_idx: usize,
    ) -> Result<(
        &mut Dsv4LayerKvLayout,
        Option<&mut Dsv4DsaSharedScratch>,
        Option<&mut Dsv4FlashMlaDecodeScratch>,
        Option<&mut Dsv4PrefillDeepGemmLinearScratch>,
        Option<&mut Dsv4CompressorFp32Scratch>,
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
            self.compressor_fp32.as_mut(),
        ))
    }

    /// Split-borrow accessor for the commit-fold path: one layer's KV layout +
    /// the model-wide single-row FlashMLA decode scratch + the model-wide FP32
    /// compressor-probe scratch (all disjoint fields). The fold's FP8 SW ring
    /// pack reuses the shared scratch's `sw_bulk_*` buffers; its compressor
    /// re-ingestion runs the FP32 probe. `None` for the FlashMLA scratch when
    /// FlashMLA decode is disabled; `None` for the FP32 scratch when the model
    /// has no compressor layer.
    #[allow(clippy::type_complexity)]
    pub(crate) fn layer_and_flashmla_scratch_mut(
        &mut self,
        layer_idx: usize,
    ) -> Result<(
        &mut Dsv4LayerKvLayout,
        Option<&mut Dsv4FlashMlaDecodeScratch>,
        Option<&mut Dsv4CompressorFp32Scratch>,
    )> {
        let len = self.layers.len();
        let layer = self
            .layers
            .get_mut(layer_idx)
            .ok_or_else(|| anyhow!("DSv4 attention pool layer {layer_idx} outside len {len}"))?;
        Ok((
            layer,
            self.flashmla_scratch.as_mut(),
            self.compressor_fp32.as_mut(),
        ))
    }

    /// Split-borrow accessor for eager B=1 MODEL1 MLA decode: one layer's KV
    /// layout + shared DSA scratch + FlashMLA scratch + that layer's persistent
    /// MLA scratch.
    pub(crate) fn layer_flashmla_and_mla_decode_mut(
        &mut self,
        layer_idx: usize,
    ) -> Result<LayerFlashMlaAndMlaDecode<'_>> {
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

    /// Whether the model's FlashMLA bands are demand-paged (#154 Phase 3b) —
    /// uniform across layers (MODEL1 vs V32 is a model-wide shape).
    pub(crate) fn flashmla_demand_paged(&self) -> bool {
        self.layers.iter().any(|l| l.flashmla_demand_paged)
    }

    /// Per-row fit for the engine's device-budget gate (#160): every row
    /// whose band growth some demand-paged layer cannot serve is pushed
    /// unfit (ascending); fitting rows debit each layer's headroom
    /// cumulatively, unfit rows debit nothing so later smaller rows are
    /// still tested. Need is paired with headroom PER LAYER — a saturated
    /// SW-only layer is free=0 AND need=0, which a scalar min-free/max-need
    /// projection misreads as permanent exhaustion. Per row and layer the
    /// growth is the exact `needed - have` draw `flashmla_ensure_band` will
    /// make at submit, including `prepare_kv_batch`'s MTP verify margin;
    /// free counts are the host free-list lengths (plan slots are distinct,
    /// so `have` stays current). No device readback (CUDA-graph safe).
    pub(crate) fn flashmla_demand_fit(
        &self,
        rows: &[infer_seam::DeviceRowDemand],
        unfit: &mut Vec<usize>,
    ) {
        let layers: Vec<&Dsv4LayerKvLayout> = self
            .layers
            .iter()
            .filter(|l| l.flashmla_demand_paged)
            .collect();
        if layers.is_empty() {
            return;
        }
        let mut free: Vec<usize> = layers
            .iter()
            .map(|l| {
                l.flashmla_kv_pool
                    .as_ref()
                    .map_or(0, TokenKVPool::free_page_count)
            })
            .collect();
        let mut needs = vec![0usize; layers.len()];
        for (idx, row) in rows.iter().enumerate() {
            let ensure_tokens = row.target_tokens + crate::dsv4::MAX_SPEC_DRAFT_DEPTH + 1;
            for (need, l) in needs.iter_mut().zip(&layers) {
                let have = l
                    .flashmla_kv_pool
                    .as_ref()
                    .map_or(0, |p| p.page_indices(row.slot).len());
                *need = l
                    .flashmla_band_pages_for(ensure_tokens)
                    .saturating_sub(have);
            }
            if needs.iter().zip(&free).any(|(need, f)| need > f) {
                unfit.push(idx);
                continue;
            }
            for (f, need) in free.iter_mut().zip(&needs) {
                *f -= need;
            }
        }
    }

    /// Engine-facing admission page count. Demand-paged (#154 Phase 3b): the
    /// solved shared token capacity in engine 64-token pages — the host pool
    /// gates admission on token-projection availability against this.
    /// Identity (V32): the physical page count of the binding layer's pool,
    /// which must track the same layer `flashmla_max_slot_pages` maximizes
    /// over (#85; see
    /// docs/experience/errors/2026-07-08-dsv4-slot-abort-band-leak-crash.md).
    pub(crate) fn flashmla_total_pages(&self) -> Option<usize> {
        if self.flashmla_demand_paged() {
            let page = self.flashmla_page_size().unwrap_or(0).max(1);
            return Some(self.flashmla_pool_tokens / page);
        }
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

    /// Fixed-band admission (identity/V32 only): every slot draws its whole
    /// band up front. Demand-paged models return None — admission becomes
    /// token-projection page availability (#154 Phase 3b).
    pub(crate) fn flashmla_max_slot_pages(&self) -> Option<usize> {
        if self.flashmla_demand_paged() {
            return None;
        }
        let pages = self
            .layers
            .iter()
            .map(Dsv4LayerKvLayout::flashmla_slot_pages)
            .max()
            .unwrap_or(0);
        (pages > 0).then_some(pages)
    }

    /// Materialize `slot`'s band for a prefix restore at `seq_len` matched
    /// tokens: demand-paged layers allocate ring + comp pages for `seq_len`;
    /// identity (V32) layers mirror the full identity band (restored content
    /// is copied into the slot's own pages). Cursor set to `seq_len`.
    pub(crate) fn mirror_full_band(
        &mut self,
        ctx: &DeviceContext,
        slot: usize,
        seq_len: usize,
    ) -> Result<()> {
        ensure!(
            slot < self.num_slots,
            "DSv4 full-band mirror slot {slot} outside adapter slots {}",
            self.num_slots
        );
        let mut changed = false;
        let mut pages: Vec<BandPage> = Vec::new();
        for layer in &mut self.layers {
            let lsp = layer.flashmla_slot_pages();
            if lsp == 0 {
                continue;
            }
            if layer.flashmla_demand_paged {
                changed |= layer.flashmla_ensure_band(ctx, slot, seq_len, false)?;
                layer.flashmla_pool_mut()?.set_band_cursor(slot, seq_len)?;
                continue;
            }
            let Some(pool) = layer.flashmla_kv_pool.as_mut() else {
                continue;
            };
            pages.clear();
            pages.extend((0..lsp).map(|i| BandPage((slot * lsp + i) as u32)));
            changed |= pool.mirror_band(slot, &pages, seq_len)?;
        }
        self.device_table_dirty[slot] |= changed;
        Ok(())
    }

    /// Materialize `slot`'s band for a DIRECT forward that bypasses
    /// `prepare_kv_batch` (boot selftest / parity examples), cursor at 0:
    /// demand layers allocate ring + comp pages for `tokens` (claim-zeroed);
    /// identity layers mirror their full identity band. The caller must
    /// consume the dirty bit and refresh the device page tables before the
    /// forward — the codex-flagged selftest bug was running kernels against
    /// never-refreshed (all-zero) device tables.
    pub(crate) fn prepare_direct_forward(
        &mut self,
        ctx: &DeviceContext,
        slot: usize,
        tokens: usize,
    ) -> Result<()> {
        ensure!(
            slot < self.num_slots,
            "DSv4 direct-forward prep slot {slot} outside adapter slots {}",
            self.num_slots
        );
        let mut changed = false;
        let mut pages: Vec<BandPage> = Vec::new();
        for layer in &mut self.layers {
            let lsp = layer.flashmla_slot_pages();
            if lsp == 0 {
                continue;
            }
            if layer.flashmla_demand_paged {
                changed |= layer.flashmla_ensure_band(ctx, slot, tokens, true)?;
                continue;
            }
            let Some(pool) = layer.flashmla_kv_pool.as_mut() else {
                continue;
            };
            pages.clear();
            pages.extend((0..lsp).map(|i| BandPage((slot * lsp + i) as u32)));
            changed |= pool.mirror_band(slot, &pages, 0)?;
        }
        self.device_table_dirty[slot] |= changed;
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

    /// Consume the slot's "host band changed since last device-table sync" bit
    /// (#154 Phase 0). `true` ⇒ the caller must refresh the graph-referenced
    /// device page tables before the forward.
    pub(crate) fn take_device_table_dirty(&mut self, slot: usize) -> bool {
        std::mem::take(&mut self.device_table_dirty[slot])
    }

    pub(crate) fn flashmla_free_slot(&mut self, slot: usize) -> Result<()> {
        for layer in &mut self.layers {
            layer.flashmla_free_slot(slot)?;
        }
        Ok(())
    }
}

impl ModelKvAdapter for Dsv4KvAdapter {
    type BatchView = Dsv4KvBatchView;

    fn prepare_kv_batch(&mut self, desc: &KvBatchDescriptor) -> Result<Self::BatchView> {
        let mut rows = Vec::with_capacity(desc.rows.len());
        // Reused per (row, layer) — the identity band is rebuilt in place so a
        // decode tick doesn't heap-alloc per layer.
        let mut layer_pages: Vec<BandPage> = Vec::new();
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
            let slot_pages = &desc.flat_slot_page_ids[row.slot_page_range.clone()];
            ensure!(
                !slot_pages.is_empty(),
                "DSv4 KV batch row {idx} has empty slot page table"
            );
            let kind = match row.kind {
                KvBatchRowKind::Prefill => "prefill",
                KvBatchRowKind::Decode => "decode",
            };
            let layer_count = self.layers.len();
            let mut band_changed = false;
            // Demand bands reserve the row's KNOWN span (prefill: the whole
            // prompt at the FIRST chunk — one growth event + one device-table
            // refresh per request instead of one per 256-token chunk crossing;
            // the pages are needed regardless, so capacity cost is zero) plus
            // the MTP verify margin: a spec step writes ring/comp state for up
            // to depth+1 draft positions beyond the committed cursor, and the
            // commit fold advances by up to the same — all inside ONE tick,
            // after this (the only) alloc point. ≤ one extra comp page,
            // budgeted by `DSV4_COMP_SAFETY_PAGES_PER_SLOT`.
            let ensure_tokens = row.total_tokens.max(row.append_pos + row.append_len)
                + crate::dsv4::MAX_SPEC_DRAFT_DEPTH
                + 1;
            let ctx = self.ctx.clone();
            for layer_idx in 0..layer_count {
                let layer = self.layer_mut(layer_idx)?;
                let lsp = layer.flashmla_slot_pages();
                if lsp == 0 {
                    continue;
                }
                if layer.flashmla_demand_paged {
                    band_changed |=
                        layer.flashmla_ensure_band(&ctx, row.slot, ensure_tokens, true)?;
                } else {
                    // Identity band (#154): band page = slot*lsp + i. Engine
                    // logical ids never enter physical tables — V32 pack
                    // needs band contiguity; reuse crosses slots by copy.
                    layer_pages.clear();
                    layer_pages.extend(
                        (0..slot_pages.len().min(lsp))
                            .map(|i| BandPage((row.slot * lsp + i) as u32)),
                    );
                    if let Some(pool) = layer.flashmla_kv_pool.as_mut() {
                        band_changed |= pool.mirror_band(row.slot, &layer_pages, row.append_pos)?;
                    }
                }
                ensure!(
                    layer
                        .flashmla_pool()
                        .map_or(true, |p| p.seq_len(row.slot) == row.append_pos),
                    "DSv4 {kind} slot {} layer {layer_idx} pool seq_len {} != append_pos {}",
                    row.slot,
                    layer.flashmla_pool().map_or(0, |p| p.seq_len(row.slot)),
                    row.append_pos
                );
                layer.flashmla_alloc_append(row.slot, row.append_len)?;
            }
            self.device_table_dirty[row.slot] |= band_changed;
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
        pool_tokens: usize,
    ) -> Result<Self> {
        let (flashmla_slot_pages, flashmla_sw_blocks) = if cuda_kernels::HAS_FLASHMLA {
            let shape = Dsv4FlashMlaDecodeShape::new(
                config,
                mode,
                compress_ratio,
                max_seq_len,
                kv_arena,
                local_heads,
                tp_world,
            )?;
            (shape.total_blocks, shape.sw_blocks)
        } else {
            (0, 0)
        };
        let flashmla_demand_paged =
            flashmla_slot_pages > 0 && super::dsv4_flashmla_demand_paged(config);
        let flashmla_comp_tokens_per_page =
            if flashmla_demand_paged && mode != DeepSeekV4AttentionMode::SlidingWindow {
                kv_arena
                    .page_block_size
                    .saturating_mul(compress_ratio.max(1))
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
            // #154 Phase 3b: each layer is sized by the ONE shared formula
            // (`dsv4_flashmla_layer_pool_pages`) — identity layers get
            // `num_slots` whole bands; demand-paged layers get per-slot
            // ring/safety pages plus the SHARED comp capacity for
            // `pool_tokens` total tokens. `kv_budget_plan` solves
            // `pool_tokens` against the SAME formula, so budget and alloc
            // cannot drift.
            let pool_pages = super::dsv4_flashmla_layer_pool_pages(
                config,
                mode,
                compress_ratio,
                max_seq_len,
                kv_arena.page_block_size,
                num_slots,
                pool_tokens,
            )?;
            let budget_bytes = pool_pages.saturating_mul(flashmla_page_bytes);
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
            // Sanity: identity pools must cover every slot's whole band;
            // demand pools at least one full-length request's band on top of
            // the per-slot ring/safety reserve — catches a `kv_budget_plan`
            // regression here instead of a mid-serve exhaustion bail
            // (pod-verified 2026-07-06: the two gates disagreeing crashed
            // every worker rank).
            // Demand floor: every slot's ring + ONE full-length comp region
            // (`lsp − sw` — zero for ring-only SW layers, whose pool is
            // exactly `num_slots × sw_blocks`).
            let min_pages = if flashmla_demand_paged {
                num_slots
                    .saturating_mul(flashmla_sw_blocks)
                    .saturating_add(flashmla_slot_pages - flashmla_sw_blocks)
            } else {
                num_slots.saturating_mul(flashmla_slot_pages)
            };
            ensure!(
                pool.page_size == kv_arena.page_block_size && pool.max_total_pages >= min_pages,
                "DSv4 FlashMLA pool page mismatch: page_size={} pages={} need page_size={} pages>={min_pages}",
                pool.page_size,
                pool.max_total_pages,
                kv_arena.page_block_size,
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
        let dsa_slot_bytes = if mode.has_indexer() {
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
        Ok(Self {
            flashmla_kv_pool,
            dsa_key_cache,
            flashmla_slot_pages,
            flashmla_page_bytes,
            flashmla_demand_paged,
            flashmla_sw_blocks,
            flashmla_comp_tokens_per_page,
            dsa_slot_bytes,
            num_slots,
        })
    }

    /// Exact requested device bytes owned by this ONE layer's KV layout (sized
    /// for ALL slots): the shared FlashMLA FP8 latent `TokenKVPool` (the
    /// dominant DSv4 KV byte sink) + the shared DSA key-cache band. The
    /// `flashmla_page_table` / scalar shape fields are host-only.
    #[allow(dead_code)]
    pub(crate) fn device_bytes(&self) -> usize {
        self.flashmla_kv_pool
            .as_ref()
            .map_or(0, |p| p.device_bytes())
            + self.dsa_key_cache.as_ref().map_or(0, |s| s.len())
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
    pub(crate) fn flashmla_pool(&self) -> Result<&TokenKVPool> {
        self.flashmla_kv_pool
            .as_ref()
            .ok_or_else(|| anyhow!("DSv4 FlashMLA shared pool missing"))
    }

    pub(crate) fn flashmla_pool_mut(&mut self) -> Result<&mut TokenKVPool> {
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
    pub(crate) fn flashmla_page_table(&self, slot_idx: usize) -> Result<&[u32]> {
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
