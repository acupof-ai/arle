use super::*;
#[derive(Clone, Copy)]
pub(super) struct Dsv4FlashMlaDecodeShape {
    pub(super) sw_blocks: usize,
    pub(super) comp_blocks: usize,
    pub(super) max_compressed_keys: usize,
    pub(super) topk_unified: usize,
    pub(super) total_blocks: usize,
    pub(super) h_q: usize,
    /// FlashMLA MODEL1 page size (== `kv_arena.page_block_size`, asserted 64).
    /// Stored so the block→(page,row) map and the pack/index kernel params draw
    /// page size from ONE source (the arena), never a re-typed literal.
    pub(super) page_block_size: usize,
}

impl Dsv4FlashMlaDecodeShape {
    /// The ONE block→(page,row) map for this shape's band.
    pub(super) fn block_map(&self) -> Dsv4BlockMap {
        Dsv4BlockMap::new(self.sw_blocks, self.page_block_size)
    }
}

impl Dsv4FlashMlaDecodeShape {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        config: &DeepSeekV4Config,
        mode: DeepSeekV4AttentionMode,
        compress_ratio: usize,
        max_seq_len: usize,
        kv_arena: &Dsv4MlaKvArena,
        local_heads: usize,
        tp_world: usize,
    ) -> Result<Self> {
        // Two FlashMLA decode shapes are wired: MODEL1 (head_dim=512 / 584 B/tok,
        // DSv4 pre-absorbed) and V32 (head_dim=576 / 656 B/tok, GLM-5.2 latent =
        // kv_lora_rank(512)+rope(64)). Both ride the same sparse-FP8 decode kernel
        // with a model_type/byte-stride switch (see `try_flashmla_decode_attention`).
        ensure!(
            (config.head_dim == 512 && kv_arena.bytes_per_token == 584)
                || (config.head_dim == 576 && kv_arena.bytes_per_token == 656),
            "DSv4 FlashMLA decode only wires MODEL1 (head_dim=512 / 584 B/tok) or \
             V32 (head_dim=576 / 656 B/tok), got head_dim={} bytes/token={}",
            config.head_dim,
            kv_arena.bytes_per_token
        );
        ensure!(
            local_heads > 0 && tp_world > 0,
            "DSv4 FlashMLA decode requires non-zero local_heads and tp_world"
        );
        let h_q = local_heads
            .checked_mul(tp_world)
            .ok_or_else(|| anyhow!("DSv4 FlashMLA h_q overflow"))?;
        ensure!(
            matches!(h_q, 64 | 128),
            "DSv4 FlashMLA decode requires global h_q 64 or 128, got {h_q}"
        );
        ensure!(
            kv_arena.page_block_size == 64,
            "DSv4 FlashMLA decode requires page_block_size=64"
        );
        let sw_blocks = config.sliding_window.div_ceil(kv_arena.page_block_size);
        let compressed_rows = if mode == DeepSeekV4AttentionMode::SlidingWindow {
            0
        } else {
            // CSA/HCA genuinely compress — a zero ratio there is a config bug.
            // GLM SparseIndexed doesn't compress (ratio 0): its latent pool spans
            // the full sequence, so treat ratio as 1 to mirror the page budget in
            // `dsv4_flashmla_slot_pages` (keeps `total_blocks` == pool pages).
            ensure!(
                compress_ratio > 0 || mode == DeepSeekV4AttentionMode::SparseIndexed,
                "DSv4 FlashMLA compressed decode requires non-zero ratio"
            );
            max_seq_len.div_ceil(compress_ratio.max(1)).max(1)
        };
        let comp_blocks = compressed_rows.div_ceil(kv_arena.page_block_size);
        let max_compressed_keys = match mode {
            DeepSeekV4AttentionMode::SlidingWindow => 0,
            DeepSeekV4AttentionMode::CompressedSparse => config.index_topk,
            DeepSeekV4AttentionMode::HybridCompressed => compressed_rows.div_ceil(128) * 128,
            // GLM SparseIndexed: top-k cap over the full latent pool == CSA.
            // GLM topk_unified = sliding_window(0) + index_topk(2048) = 2048 (128-multiple).
            DeepSeekV4AttentionMode::SparseIndexed => config.index_topk,
        };
        let topk_unified = config
            .sliding_window
            .checked_add(max_compressed_keys)
            .ok_or_else(|| anyhow!("DSv4 FlashMLA topk_unified overflow"))?;
        ensure!(
            topk_unified.is_multiple_of(128),
            "DSv4 FlashMLA topk_unified {topk_unified} must be multiple of 128"
        );
        let total_blocks = sw_blocks
            .checked_add(comp_blocks)
            .ok_or_else(|| anyhow!("DSv4 FlashMLA total block overflow"))?;
        Ok(Self {
            sw_blocks,
            comp_blocks,
            max_compressed_keys,
            topk_unified,
            total_blocks,
            h_q,
            page_block_size: kv_arena.page_block_size,
        })
    }
}

pub(crate) struct Dsv4FlashMlaDecodeState {
    pub(crate) slot_idx: usize,
    pub(super) fp8_kv_pool_len: usize,
    pub(super) sw_blocks: usize,
    pub(super) comp_blocks: usize,
    pub(super) max_compressed_keys: usize,
    pub(super) topk_unified: usize,
    /// FlashMLA page size (== arena `page_block_size`, asserted 64) — the ONE
    /// page-size source for this slot's block map and pack/index kernel params.
    pub(super) page_block_size: usize,
    pub(super) fp8_kv_sw_bootstrapped: bool,
    pub(super) fp8_kv_comp_packed_rows: usize,
    // CONSTANT-after-init scheduler metadata + slot-shape constants. The 13
    // per-forward SCRATCH buffers (`sw_bulk_*`/`one_*`/`comp_*`/`indices`/
    // `lse_*`/`o_accum`/`tp_*`) were hoisted to the model-wide
    // [`Dsv4FlashMlaDecodeScratch`] (#85 P3) — they carry no cross-call/cross-slot
    // state, so one shared instance serves every (slot, layer).
    pub(super) topk_length: CudaSlice<i32>,
    pub(super) sched_meta: CudaSlice<i32>,
    pub(super) num_splits: CudaSlice<i32>,
    pub(super) num_sm_parts: i32,
    pub(super) fixed_overhead_num_blocks: i32,
    pub(super) block_size_topk: i32,
    /// Persistent device copy of the slot's page table (`flashmla_page_table`).
    /// Kept alive for the slot's lifetime so CUDA-graph-captured kernel args
    /// never reference a freed temporary (prefix-cache restore UAF, #8).
    pub(crate) device_page_table: CudaSlice<i32>,
}

impl Dsv4FlashMlaDecodeState {
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
        slot_idx: usize,
        pool: &Dsv4LayerKvLayout,
    ) -> Result<Self> {
        let shape = Dsv4FlashMlaDecodeShape::new(
            config,
            mode,
            compress_ratio,
            max_seq_len,
            kv_arena,
            local_heads,
            tp_world,
        )?;
        // #85 P2 Stage B: slots start EMPTY (the band is drawn on admission), so
        // the band length is derived from the SHAPE, not the table. The decode
        // shape must agree with the layout's per-slot block count (both derive
        // from the same plan-pinned construction params).
        ensure!(
            shape.total_blocks == pool.flashmla_slot_pages && shape.total_blocks > 0,
            "DSv4 FlashMLA shared slot band shape mismatch for slot {slot_idx} \
             (shape blocks {} vs layout blocks {})",
            shape.total_blocks,
            pool.flashmla_slot_pages
        );
        let fp8_kv_pool_len = shape
            .total_blocks
            .checked_mul(pool.flashmla_page_bytes)
            .ok_or_else(|| anyhow!("DSv4 FlashMLA slot band byte length overflow"))?;

        let mut num_sm_parts = 0_i32;
        let mut fixed_overhead_num_blocks = 0_i32;
        let mut block_size_topk = 0_i32;
        // SAFETY: ptrs from live device allocations sized to the dims passed.
        unsafe {
            ffi::arle_flashmla_sm90_sparse_decode_get_meta(
                shape.h_q as i32,
                DSV4_FLASHMLA_S_Q as i32,
                DSV4_FLASHMLA_MODEL1,
                &mut num_sm_parts,
                &mut fixed_overhead_num_blocks,
                &mut block_size_topk,
            )
            .result()
            .map_err(|e| anyhow!("DSv4 FlashMLA decode meta failed: {e}"))?;
        }
        let num_sm_parts_max = (num_sm_parts as usize).max(256);

        // SCRATCH buffers (`sw_bulk_*`/`one_*`/`comp_*`/`indices`/`lse_*`/`o_accum`/
        // `tp_*`) are not allocated here — they live in the model-wide
        // [`Dsv4FlashMlaDecodeScratch`] (#85). Only the slot-shape constants +
        // CONSTANT-after-init scheduler metadata remain per (slot, layer).
        let mut state = Self {
            slot_idx,
            fp8_kv_pool_len,
            sw_blocks: shape.sw_blocks,
            comp_blocks: shape.comp_blocks,
            max_compressed_keys: shape.max_compressed_keys,
            topk_unified: shape.topk_unified,
            page_block_size: shape.page_block_size,
            fp8_kv_sw_bootstrapped: false,
            fp8_kv_comp_packed_rows: 0,
            topk_length: ctx.stream.alloc_zeros::<i32>(1)?,
            sched_meta: ctx.stream.alloc_zeros::<i32>(num_sm_parts_max * 8)?,
            num_splits: ctx.stream.alloc_zeros::<i32>(2)?,
            num_sm_parts,
            fixed_overhead_num_blocks,
            block_size_topk,
            device_page_table: ctx.stream.alloc_zeros::<i32>(shape.total_blocks)?,
        };
        state.init_constant_sched_meta(ctx)?;
        // NOT refreshed here: at construction the host page table is
        // legitimately EMPTY (band drawn on first admission, #85 P2 Stage B).
        // The zeroed placeholder is safe — FlashMLA kernels never read this
        // buffer before the first band mirror marks the slot dirty and the
        // executor's dirty-driven refresh fills it.
        Ok(state)
    }

    /// Fill `topk_length` and the FlashMLA scheduler metadata ONCE: both
    /// depend only on slot constants (`topk_unified`, sm-part shape), so
    /// computing them per decode step was (a) wasted work (43 calls/token)
    /// and (b) a CUDA-graph capture hazard — the per-step
    /// `memcpy_htod(&[topk], ..)` recorded a memcpy node whose HOST source
    /// was a dead stack temporary, so replay read a dangling pointer
    /// (garbage topk → insane splits → the 2026-06-10 IMA).
    pub(super) fn init_constant_sched_meta(&mut self, ctx: &DeviceContext) -> Result<()> {
        let topk = i32::try_from(self.topk_unified)
            .map_err(|_| anyhow!("DSv4 FlashMLA topk {} overflows i32", self.topk_unified))?;
        ctx.stream
            .memcpy_htod(&[topk], &mut self.topk_length)
            .map_err(|e| anyhow!("DSv4 FlashMLA topk_length H2D failed: {e}"))?;
        let (topk_ptr, _tg) = self.topk_length.device_ptr(&ctx.stream);
        let (sched_ptr, _sg) = self.sched_meta.device_ptr_mut(&ctx.stream);
        let (splits_ptr, _pg) = self.num_splits.device_ptr_mut(&ctx.stream);
        // SAFETY: ptrs from live device allocations sized to the dims passed.
        unsafe {
            ffi::arle_flashmla_sm90_sparse_decode_sched_meta(
                1,
                1,
                self.block_size_topk,
                self.fixed_overhead_num_blocks,
                topk,
                0,
                topk_ptr as *const i32,
                std::ptr::null(),
                sched_ptr as *mut i32,
                splits_ptr as *mut i32,
                self.num_sm_parts,
                ctx.stream.cu_stream(),
            )
            .result()
            .map_err(|e| anyhow!("DSv4 FlashMLA sched_meta failed: {e}"))?;
        }
        Ok(())
    }

    /// Re-sync the persistent device page table from the host page table —
    /// the CUDA-graph-captured kernel arg points at this fixed buffer, so a
    /// stale copy reads garbage. Dirty-bit driven (#154 Phase 0:
    /// `Dsv4KvAdapter::take_device_table_dirty`).
    pub(super) fn refresh_device_page_table(
        &mut self,
        ctx: &DeviceContext,
        pool: &Dsv4LayerKvLayout,
    ) -> Result<()> {
        let mut table_i32: Vec<i32> = pool
            .flashmla_page_table(self.slot_idx)?
            .iter()
            .map(|&p| p as i32)
            .collect();
        ensure!(
            table_i32.len() <= self.device_page_table.len(),
            "DSv4 FlashMLA device page table size mismatch: host {} vs device {}",
            table_i32.len(),
            self.device_page_table.len()
        );
        // Host tables carry only real pages (#154 Phase 0); the graph-captured
        // device table is fixed-size, so pad HERE — the one device-format
        // padding boundary.
        let pad = table_i32.last().copied().unwrap_or(0);
        table_i32.resize(self.device_page_table.len(), pad);
        ctx.stream
            .memcpy_htod(&table_i32, &mut self.device_page_table)
            .map_err(|e| anyhow!("DSv4 FlashMLA device page table H2D failed: {e}"))?;
        Ok(())
    }

    pub(super) fn reset(&mut self) {
        self.fp8_kv_sw_bootstrapped = false;
        self.fp8_kv_comp_packed_rows = 0;
    }

    /// SW sub-pool block count (= `ceil(sliding_window/page_block_size)`, uniform
    /// across a layer's slots); the batched compressed-delta pack's `sw_blocks`.
    pub(crate) fn sw_blocks(&self) -> usize {
        self.sw_blocks
    }

    /// The ONE block→(page,row) map for this slot's band. The pack/index kernel
    /// `sw_blocks`/`page_block_size` params draw from this same source so they
    /// cannot drift from the map (#146 Index-layer single-source).
    pub(crate) fn block_map(&self) -> Dsv4BlockMap {
        Dsv4BlockMap::new(self.sw_blocks, self.page_block_size)
    }

    /// Exact requested device bytes owned by this per-(slot,layer) FlashMLA
    /// decode state. After the #85 P3 hoist this is only the CONSTANT-after-init
    /// scheduler metadata (`topk_length`/`sched_meta`/`num_splits`); the 13
    /// per-forward SCRATCH buffers live in the model-wide
    /// [`Dsv4FlashMlaDecodeScratch`]. The FP8 KV pool pages it reads through the
    /// slot block table are NOT owned here — they live in
    /// [`Dsv4LayerKvLayout::flashmla_kv_pool`] and are summed there once.
    #[allow(dead_code)]
    pub(crate) fn device_bytes(&self) -> usize {
        let i32_sz = std::mem::size_of::<i32>();
        self.topk_length.len() * i32_sz
            + self.sched_meta.len() * i32_sz
            + self.num_splits.len() * i32_sz
    }

    /// STATIC estimate of `device_bytes` for `Dsv4Model::per_slot_device_bytes`.
    /// `sched_meta`'s exact length uses `num_sm_parts` from a device-only FFI meta
    /// call; `new` floors it at `max(num_sm_parts, 256)`, so this static value uses
    /// the 256 floor. The whole term is ~8 KB/layer (<0.1% of a slot) — any
    /// FFI-driven excess is caught by the executor's per-slot drift guard.
    pub(crate) fn device_bytes_estimate() -> usize {
        let i32_sz = std::mem::size_of::<i32>();
        let num_sm_parts_max = 256usize;
        // topk_length[1] + sched_meta[num_sm_parts_max * 8] + num_splits[2].
        (1 + num_sm_parts_max * 8 + 2) * i32_sz
    }

    pub(crate) fn swap_out(
        &self,
        ctx: &DeviceContext,
        _pool: &Dsv4LayerKvLayout,
    ) -> Result<crate::attention::Dsv4FlashMlaImage> {
        let _ = ctx;
        Ok(crate::attention::Dsv4FlashMlaImage {
            fp8_kv_sw_bootstrapped: self.fp8_kv_sw_bootstrapped,
            fp8_kv_comp_packed_rows: self.fp8_kv_comp_packed_rows,
        })
    }

    pub(crate) fn swap_in(
        &mut self,
        ctx: &DeviceContext,
        _pool: &mut Dsv4LayerKvLayout,
        image: &crate::attention::Dsv4FlashMlaImage,
    ) -> Result<()> {
        let _ = ctx;
        self.fp8_kv_sw_bootstrapped = image.fp8_kv_sw_bootstrapped;
        self.fp8_kv_comp_packed_rows = image.fp8_kv_comp_packed_rows;
        Ok(())
    }
}

/// Model-wide (one instance, NOT per-slot/per-layer) per-forward SCRATCH for the
/// SINGLE-ROW (`s_q = 1`) FlashMLA sparse decode lane (#85 P3). These are the
/// per-step accumulators / staging buffers that the single-row pack + decode +
/// TP path overwrite-before-read every layer step; they carry NO cross-call or
/// cross-slot state (the 2026-06 whole-slot-swap SCRATCH verdict).
/// They were wrongly allocated PER (slot, layer) inside `Dsv4FlashMlaDecodeState`
/// — `o_accum` alone is `(num_sm_parts+1) × h_q × head_dim × f32 ≈ 33.7 MB/layer`,
/// so 43 FlashMLA layers cost ~1392 MB PER SLOT. DSv4 runs its layers
/// sequentially and the engine runs one forward at a time, so ONE shared instance
/// suffices (exactly like `dsa_shared`/`flashmla_batch`/`prefill_linear`): the
/// single-row decode loop overwrites the shared scratch before reading every
/// iteration, serial on `ctx.stream`. Hoisting drops per-slot from ~1466 MB to
/// ~74 MB.
///
/// Every buffer is sized for the WORST CASE across all FlashMLA layer shapes
/// (the per-mode `topk_unified` / compressed-row span differ by layer; `h_q` is
/// uniform 64/128). `num_sm_parts` depends only on `(h_q, s_q=1, model)` — the
/// same FFI meta the single-row state queried — so it is computed once.
pub(crate) struct Dsv4FlashMlaDecodeScratch {
    /// `[sliding_window]` i32 — SW bulk pack block-ids (ring-identity constants),
    /// written immediately before their single read in `flashmla_pack_sw_ring`.
    pub(super) sw_bulk_block_ids: CudaSlice<i32>,
    /// `[sliding_window]` i32 — SW bulk pack row offsets (same lifecycle).
    pub(super) sw_bulk_rows: CudaSlice<i32>,
    /// `[1]` i32 — one-token SW pack block-id, device-written from `start_pos`
    /// at the top of every `flashmla_pack_one_sw_token` before its single read.
    pub(super) one_block_id: CudaSlice<i32>,
    /// `[1]` i32 — one-token SW pack row offset (same lifecycle).
    pub(super) one_row: CudaSlice<i32>,
    /// `[max comp_slots]` i32 — compressed-delta bulk pack block-ids, H2D-written
    /// immediately before their single read in `flashmla_pack_compressed_delta`.
    /// Sized `max(comp_blocks × 64) over layers` (`.max(1)`) — a shape-derived
    /// upper bound on each layer's `comp_slots = max_seq_len.div_ceil(ratio)`.
    pub(super) comp_block_ids: CudaSlice<i32>,
    /// `[max comp_slots]` i32 — compressed-delta bulk pack row offsets (same).
    pub(super) comp_rows: CudaSlice<i32>,
    /// `[max topk_unified]` i32 — the unified sparse indices the build-indices
    /// kernel writes each decode step before the sparse-decode kernel reads them.
    pub(super) indices: CudaSlice<i32>,
    /// `[h_q]` f32 — decode-kernel LSE output, fully overwritten per launch.
    pub(super) lse_out: CudaSlice<f32>,
    /// `[(num_sm_parts+1) × h_q]` f32 — split-KV LSE accumulator (same lifecycle).
    pub(super) lse_accum: CudaSlice<f32>,
    /// `[(num_sm_parts+1) × h_q × head_dim]` f32 — split-KV O accumulator. The
    /// dominant byte sink; fully overwritten per launch before read.
    pub(super) o_accum: CudaSlice<f32>,
    /// `[h_q × head_dim]` bf16 — TP all-gather landing buffer for the global-head
    /// Q (TP path only), written each decode step before read.
    pub(super) tp_gathered_q: CudaSlice<half::bf16>,
    /// `[h_q × head_dim]` bf16 — repacked global-head Q (TP path), written each step.
    pub(super) tp_packed_q: CudaSlice<half::bf16>,
    /// `[h_q × head_dim]` bf16 — full global-head fwd output staging (TP path),
    /// written by the fwd then read by the out-slice each step.
    pub(super) tp_full_out: CudaSlice<half::bf16>,
}

impl Dsv4FlashMlaDecodeScratch {
    /// Allocate the ONE model-wide single-row FlashMLA decode scratch, each
    /// buffer sized for the worst case across `layer_shapes` (every FlashMLA
    /// layer's decode shape). All shapes must share `h_q`.
    pub(super) fn new(
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        layer_shapes: &[Dsv4FlashMlaDecodeShape],
    ) -> Result<Self> {
        ensure!(
            !layer_shapes.is_empty(),
            "DSv4 single-row FlashMLA decode scratch needs at least one layer shape"
        );
        let h_q = layer_shapes[0].h_q;
        ensure!(
            layer_shapes.iter().all(|s| s.h_q == h_q),
            "DSv4 single-row FlashMLA decode scratch requires a uniform h_q across layers"
        );
        let head_dim = config.head_dim;
        let h_q_d = h_q
            .checked_mul(head_dim)
            .ok_or_else(|| anyhow!("DSv4 FlashMLA scratch h_q*d overflow"))?;
        // `sw_*` are config-constant (`sliding_window`) like the per-slot alloc.
        let sw_slots = config.sliding_window;
        // Worst-case compressed-row span: `comp_blocks * page_block_size` is a
        // shape-derived upper bound on each layer's `comp_slots` (the per-slot
        // alloc used `max_seq_len.div_ceil(ratio)`, and `comp_blocks =
        // compressed_rows.div_ceil(64)` ⇒ `comp_blocks*64 ≥ comp_slots`). SW
        // layers have `comp_blocks=0`; the `.max(1)` matches the per-slot floor.
        let comp_slots = layer_shapes
            .iter()
            .map(|s| s.comp_blocks * 64)
            .max()
            .unwrap_or(0)
            .max(1);
        let max_topk_unified = layer_shapes
            .iter()
            .map(|s| s.topk_unified)
            .max()
            .unwrap_or(0);
        // num_sm_parts depends only on (h_q, s_q=1, model) — uniform across
        // layers; same call the single-row state + batch scratch use.
        let mut num_sm_parts = 0_i32;
        let mut fixed_overhead_num_blocks = 0_i32;
        let mut block_size_topk = 0_i32;
        // SAFETY: ptrs from live device allocations sized to the dims passed.
        unsafe {
            ffi::arle_flashmla_sm90_sparse_decode_get_meta(
                h_q as i32,
                DSV4_FLASHMLA_S_Q as i32,
                DSV4_FLASHMLA_MODEL1,
                &mut num_sm_parts,
                &mut fixed_overhead_num_blocks,
                &mut block_size_topk,
            )
            .result()
            .map_err(|e| anyhow!("DSv4 single-row FlashMLA scratch meta failed: {e}"))?;
        }
        let num_sm_parts_max = (num_sm_parts as usize).max(256);
        let accum_rows = num_sm_parts_max + 1;
        Ok(Self {
            sw_bulk_block_ids: ctx.stream.alloc_zeros::<i32>(sw_slots)?,
            sw_bulk_rows: ctx.stream.alloc_zeros::<i32>(sw_slots)?,
            one_block_id: ctx.stream.alloc_zeros::<i32>(1)?,
            one_row: ctx.stream.alloc_zeros::<i32>(1)?,
            comp_block_ids: ctx.stream.alloc_zeros::<i32>(comp_slots)?,
            comp_rows: ctx.stream.alloc_zeros::<i32>(comp_slots)?,
            indices: ctx.stream.alloc_zeros::<i32>(max_topk_unified)?,
            lse_out: ctx.stream.alloc_zeros::<f32>(h_q)?,
            lse_accum: ctx.stream.alloc_zeros::<f32>(accum_rows * h_q)?,
            o_accum: ctx.stream.alloc_zeros::<f32>(accum_rows * h_q_d)?,
            tp_gathered_q: ctx.stream.alloc_zeros::<half::bf16>(h_q_d)?,
            tp_packed_q: ctx.stream.alloc_zeros::<half::bf16>(h_q_d)?,
            tp_full_out: ctx.stream.alloc_zeros::<half::bf16>(h_q_d)?,
        })
    }

    /// Exact requested device bytes owned by this ONE model-wide single-row
    /// FlashMLA decode scratch.
    #[allow(dead_code)]
    pub(crate) fn device_bytes(&self) -> usize {
        let i32_sz = std::mem::size_of::<i32>();
        let f32_sz = std::mem::size_of::<f32>();
        let bf16 = std::mem::size_of::<half::bf16>();
        self.sw_bulk_block_ids.len() * i32_sz
            + self.sw_bulk_rows.len() * i32_sz
            + self.one_block_id.len() * i32_sz
            + self.one_row.len() * i32_sz
            + self.comp_block_ids.len() * i32_sz
            + self.comp_rows.len() * i32_sz
            + self.indices.len() * i32_sz
            + self.lse_out.len() * f32_sz
            + self.lse_accum.len() * f32_sz
            + self.o_accum.len() * f32_sz
            + self.tp_gathered_q.len() * bf16
            + self.tp_packed_q.len() * bf16
            + self.tp_full_out.len() * bf16
    }
}

/// Model-wide (one instance, NOT per-slot/per-layer) scratch for the batched
/// (`b = N`) FlashMLA sparse decode lane (#60). The single-row
/// [`Dsv4FlashMlaDecodeState`] is per-(slot, layer) and sized for `s_q = 1`;
/// the batched lane runs ONE `sparse_decode_fwd(b=N)` over N DIFFERENT slots
/// that all share one layer's KV pool, so its row-major buffers must live in a
/// single shared instance sized for `max_batch` rows. Mirrors the
/// allocate-once / reuse-every-forward discipline of the other model-wide
/// scratches ([`Dsv4DsaSharedScratch`], `Dsv4MoeDecodeScratch`).
///
/// All row-dimension buffers are `[max_batch, …]`; a forward over `n ≤ max_batch`
/// rows uses the `[0, n)` prefix. `max_topk_unified` is the MAX across every
/// FlashMLA layer mode (SW vs HCA vs CSA differ in `topk_unified`); `h_q` is
/// uniform (global heads 64 or 128). The split-KV accum scratch is sized
/// `[num_sm_parts + max_batch, …]` and SHARED across the batch. Pool addressing:
/// the batched indices builder emits POOL-ABSOLUTE block coords (it adds
/// `slot_block_offsets[row] * page_block_size`), so the single batched fwd reads
/// one pool base pointer — unlike the single-row path's per-slot `pool_view`.
pub(crate) struct Dsv4FlashMlaDecodeBatchScratch {
    /// `[max_batch, max_topk_unified]` i32 — per-row unified sparse indices in
    /// pool-absolute block coords. Written by the batched indices builder each
    /// forward (all `n` rows) before the fwd reads it.
    pub(super) indices: CudaSlice<i32>,
    /// `[max_batch]` i32 — per-row effective topk length (= each row's mode's
    /// `topk_unified`). Written by the batched indices builder.
    pub(super) topk_length: CudaSlice<i32>,
    /// `[max_batch]` i32 — per-row absolute decode position. Host→device each
    /// forward; feeds the batched indices builder's causal gate.
    pub(super) start_pos: CudaSlice<i32>,
    /// `[max_batch]` i32 — each row's slot's first FlashMLA pool block (the
    /// slot→pool translation the batched indices builder applies). Host→device
    /// each forward from the slot page tables.
    pub(super) slot_block_offsets: CudaSlice<i32>,
    /// `[max_batch, total_blocks]` i32 — padded per-row logical->physical page
    /// tables for the Stage-B batched indices build. Identity rows reproduce the
    /// old band path byte-for-byte; fragmented rows route directly to
    /// pool-absolute physical blocks and the kernel skips `slot_block_offsets`.
    pub(super) page_table_batched: CudaSlice<i32>,
    /// `[max_batch, h_q]` f32 — per-row LSE output of the sparse decode.
    pub(super) lse_out: CudaSlice<f32>,
    /// `[num_sm_parts + max_batch, h_q]` f32 — split-KV LSE accumulator, SHARED
    /// across the batch (split dim folds b in via `num_splits`). Strides match
    /// the single-row path (split stride = s_q*h_q).
    pub(super) lse_accum: CudaSlice<f32>,
    /// `[num_sm_parts + max_batch, h_q * d_v]` f32 — split-KV O accumulator,
    /// shared across the batch (see `lse_accum`). d_v=512 always.
    pub(super) o_accum: CudaSlice<f32>,
    /// `[num_sm_parts_max * 8]` i32 — tile-scheduler metadata. RECOMPUTED PER
    /// FORWARD via `sched_meta(b=n)` (the cached-constant pitfall: the b=1 cached
    /// meta is WRONG for n>1 — wrong split-KV merge).
    pub(super) sched_meta: CudaSlice<i32>,
    /// `[max_batch + 1]` i32 — split offsets. Written per forward by `sched_meta`.
    pub(super) num_splits: CudaSlice<i32>,
    /// `[max_batch, h_q * head_dim]` bf16 — gathered+repacked global-head Q
    /// (TP path) or the row-major batched Q (single-GPU). Fed as the fwd's q.
    /// PHASE B: each row r's prepared Q is written into `[r, ..]` by the per-row
    /// pack/gather before the ONE batched fwd reads the whole `[0, n)` prefix.
    pub(super) q_batched: CudaSlice<half::bf16>,
    /// `[max_batch, h_q * d_v]` bf16 — full global-head fwd output, read
    /// per-row by the finish step (TP out-slice on the TP path, direct copy on
    /// single-GPU). PHASE B. d_v=512 always.
    pub(super) out_batched: CudaSlice<half::bf16>,
    /// `[max_batch, tp_world * local_heads * head_dim]` bf16 — TP all-gather
    /// landing buffer (TP path only). PHASE B: the gather loop uses the `[0,
    /// tp_gather_cols)` slice per row (one row's allgather at a time), then
    /// repacks the global Q into `q_batched[r]`.
    pub(super) tp_gathered_q: CudaSlice<half::bf16>,
    /// `[max_batch, csa_topk]` i32 — CSA ONLY: the prepare loop gathers each
    /// row's indexer top-k `selected` (`[index_topk]`) into row r, then the ONE
    /// batched index build reads `selected + row * max_compressed_keys`
    /// (== `index_topk` for CSA). Disjoint from `indices`/`q_batched`/etc.
    /// `csa_topk == 0` (zero-len alloc, never read) when the model has no CSA
    /// layer; SW/HCA never touch this buffer (their `selected_ptr` stays 0).
    pub(super) selected_batched: CudaSlice<i32>,
    /// CSA per-row `selected` length (= `config.index_topk` = a CSA layer's
    /// `max_compressed_keys`); the row stride of `selected_batched`. 0 if the
    /// model has no CSA layer.
    pub(super) csa_topk: usize,
    /// Per-layer decode shape (`topk_unified`/`sw_blocks`/`comp_blocks`/… per
    /// mode), indexed by layer. The batched build_indices / fwd read the
    /// engaged layer's shape; stored here so the dsv4 loop needs only `layer_idx`.
    pub(super) layer_shapes: Vec<Dsv4FlashMlaDecodeShape>,
    pub(super) max_batch: usize,
    pub(super) max_topk_unified: usize,
    /// Global heads (64/128); PHASE B: row stride component of q_batched
    /// (`h_q*head_dim`) and out_batched (`h_q*d_v`).
    pub(super) h_q: usize,
    /// PHASE B: `h_q_d` row stride component.
    pub(super) head_dim: usize,
    /// Output latent dim (d_v=512 always; == head_dim for MODEL1, < head_dim
    /// for V32). Sizes out_batched/o_accum and the finish-step output slice.
    pub(super) d_v: usize,
    pub(super) num_sm_parts: i32,
    pub(super) fixed_overhead_num_blocks: i32,
    pub(super) block_size_topk: i32,
}

impl Dsv4FlashMlaDecodeBatchScratch {
    /// Allocate the batched-decode scratch sized for `max_batch` concurrent
    /// decode rows over the model's FlashMLA layers. `layer_shapes` are every
    /// FlashMLA layer's decode shape (so `max_topk_unified` / `accum_rows_max`
    /// cover the worst mode); all must share `h_q` and `head_dim`.
    pub(super) fn new(
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        max_batch: usize,
        layer_shapes: &[Dsv4FlashMlaDecodeShape],
    ) -> Result<Self> {
        ensure!(
            max_batch > 0,
            "DSv4 batched FlashMLA decode needs max_batch>0"
        );
        ensure!(
            !layer_shapes.is_empty(),
            "DSv4 batched FlashMLA decode needs at least one FlashMLA layer shape"
        );
        let h_q = layer_shapes[0].h_q;
        ensure!(
            layer_shapes.iter().all(|s| s.h_q == h_q),
            "DSv4 batched FlashMLA decode requires a uniform h_q across layers"
        );
        let head_dim = config.head_dim;
        let model_type_int = dsv4_flashmla_model_meta(config)?.model_type_int;
        let max_topk_unified = layer_shapes
            .iter()
            .map(|s| s.topk_unified)
            .max()
            .unwrap_or(0);
        // The scheduler tuning meta (num_sm_parts/fixed_overhead/block_size_topk)
        // depends only on (h_q, s_q=1, model). Same call the single-row state
        // uses; the resulting `num_sm_parts` sizes the accum + sched buffers.
        let mut num_sm_parts = 0_i32;
        let mut fixed_overhead_num_blocks = 0_i32;
        let mut block_size_topk = 0_i32;
        // SAFETY: ptrs from live device allocations sized to the dims passed.
        unsafe {
            ffi::arle_flashmla_sm90_sparse_decode_get_meta(
                h_q as i32,
                DSV4_FLASHMLA_S_Q as i32,
                model_type_int,
                &mut num_sm_parts,
                &mut fixed_overhead_num_blocks,
                &mut block_size_topk,
            )
            .result()
            .map_err(|e| anyhow!("DSv4 batched FlashMLA decode meta failed: {e}"))?;
        }
        let num_sm_parts_max = (num_sm_parts as usize).max(256);
        // Split-KV accum scratch is shared across the whole batch: the shim
        // documents lse_accum/o_accum as `[num_sm_parts + b, s_q, h_q(*d_v)]`
        // (arle_flashmla_decode_shim.cu:202-203) — the split dimension is
        // `num_sm_parts + b`, NOT `b * num_sm_parts`. Accum strides therefore
        // stay the single-row values (split stride = s_q*h_q etc.); b is folded
        // into the split index via `num_splits[b+1]`, not a separate batch axis.
        let accum_splits_max = num_sm_parts_max + max_batch;
        let h_q_d = h_q
            .checked_mul(head_dim)
            .ok_or_else(|| anyhow!("DSv4 batched FlashMLA h_q*d overflow"))?;
        // Output (out_batched/o_accum) is [b, h_q, d_v]: d_v=512 always (shim
        // hard-asserts). For MODEL1 d_v==head_dim; for V32 d_v=512 < head_dim=576.
        let d_v = if model_type_int == DSV4_FLASHMLA_V32 {
            512
        } else {
            head_dim
        };
        let h_q_d_v = h_q
            .checked_mul(d_v)
            .ok_or_else(|| anyhow!("DSv4 batched FlashMLA h_q*d_v overflow"))?;
        let tp_gather_cols = h_q_d; // tp_world * local_heads * head_dim == global h_q * head_dim
        // Size the per-row page table for the WIDEST layer's band (CSA total_blocks);
        // narrower SW-only layers use a row_width prefix of this shared buffer.
        let max_total_blocks = layer_shapes
            .iter()
            .map(|s| s.total_blocks)
            .max()
            .unwrap_or(0);
        Ok(Self {
            indices: ctx
                .stream
                .alloc_zeros::<i32>(max_batch * max_topk_unified)?,
            topk_length: ctx.stream.alloc_zeros::<i32>(max_batch)?,
            start_pos: ctx.stream.alloc_zeros::<i32>(max_batch)?,
            slot_block_offsets: ctx.stream.alloc_zeros::<i32>(max_batch)?,
            page_table_batched: ctx
                .stream
                .alloc_zeros::<i32>(max_batch * max_total_blocks)?,
            lse_out: ctx.stream.alloc_zeros::<f32>(max_batch * h_q)?,
            lse_accum: ctx.stream.alloc_zeros::<f32>(accum_splits_max * h_q)?,
            o_accum: ctx.stream.alloc_zeros::<f32>(accum_splits_max * h_q_d_v)?,
            sched_meta: ctx.stream.alloc_zeros::<i32>(num_sm_parts_max * 8)?,
            num_splits: ctx.stream.alloc_zeros::<i32>(max_batch + 1)?,
            q_batched: ctx.stream.alloc_zeros::<half::bf16>(max_batch * h_q_d)?,
            out_batched: ctx.stream.alloc_zeros::<half::bf16>(max_batch * h_q_d_v)?,
            tp_gathered_q: ctx
                .stream
                .alloc_zeros::<half::bf16>(max_batch * tp_gather_cols)?,
            // CSA per-row top-k gather buffer. `config.index_topk` is the single
            // config-level top-k width every CSA layer uses (== a CSA layer's
            // `max_compressed_keys`), so one fixed row stride serves all CSA
            // layers. Sized `max_batch * index_topk` (≈ slots × 512 × 4B, tiny).
            // For a model with no DSA/CSA layer index_topk is 0 → a zero-len alloc
            // that is never read (SW/HCA pass selected_ptr=0).
            selected_batched: ctx
                .stream
                .alloc_zeros::<i32>(max_batch * config.index_topk)?,
            csa_topk: config.index_topk,
            layer_shapes: layer_shapes.to_vec(),
            max_batch,
            max_topk_unified,
            h_q,
            head_dim,
            d_v,
            num_sm_parts,
            fixed_overhead_num_blocks,
            block_size_topk,
        })
    }

    /// Exact requested device bytes owned by this ONE model-wide batched
    /// FlashMLA decode scratch (`layer_shapes` is host-only metadata, excluded).
    #[allow(dead_code)]
    pub(crate) fn device_bytes(&self) -> usize {
        let i32_sz = std::mem::size_of::<i32>();
        let f32_sz = std::mem::size_of::<f32>();
        let bf16 = std::mem::size_of::<half::bf16>();
        self.indices.len() * i32_sz
            + self.topk_length.len() * i32_sz
            + self.start_pos.len() * i32_sz
            + self.slot_block_offsets.len() * i32_sz
            + self.page_table_batched.len() * i32_sz
            + self.lse_out.len() * f32_sz
            + self.lse_accum.len() * f32_sz
            + self.o_accum.len() * f32_sz
            + self.sched_meta.len() * i32_sz
            + self.num_splits.len() * i32_sz
            + self.q_batched.len() * bf16
            + self.out_batched.len() * bf16
            + self.tp_gathered_q.len() * bf16
            + self.selected_batched.len() * i32_sz
    }

    /// Upload the per-row decode positions + slot→pool block offsets + padded
    /// per-row page tables for an `n`-row forward. `slot_block_offsets[r]`
    /// remains for the legacy band path (`page_table == None`); Stage-B routes
    /// the batched kernel through `page_table_batched` and the offsets are
    /// ignored. Writes the `[0, n)` prefix of `self.start_pos` /
    /// `self.slot_block_offsets` and the `[0, n*row_width)` prefix of
    /// `self.page_table_batched`.
    pub(super) fn upload_row_meta(
        &mut self,
        ctx: &DeviceContext,
        start_positions: &[usize],
        slot_block_offsets: &[usize],
        page_tables: &[Vec<i32>],
    ) -> Result<()> {
        let n = start_positions.len();
        ensure!(
            n == slot_block_offsets.len(),
            "DSv4 batched decode start_pos/offsets length mismatch ({n} vs {})",
            slot_block_offsets.len()
        );
        ensure!(
            n == page_tables.len(),
            "DSv4 batched decode start_pos/page_tables length mismatch ({n} vs {})",
            page_tables.len()
        );
        ensure!(
            n <= self.max_batch,
            "DSv4 batched decode n={n} exceeds max_batch={}",
            self.max_batch
        );
        let start_host: Vec<i32> = start_positions
            .iter()
            .map(|&p| {
                i32::try_from(p)
                    .map_err(|_| anyhow!("DSv4 batched decode start_pos {p} overflows i32"))
            })
            .collect::<Result<_>>()?;
        let off_host: Vec<i32> = slot_block_offsets
            .iter()
            .map(|&b| {
                i32::try_from(b)
                    .map_err(|_| anyhow!("DSv4 batched decode block offset {b} overflows i32"))
            })
            .collect::<Result<_>>()?;
        let mut start_view = self.start_pos.slice_mut(0..n);
        ctx.stream
            .memcpy_htod(&start_host, &mut start_view)
            .map_err(|e| anyhow!("DSv4 batched decode start_pos H2D failed: {e}"))?;
        let mut off_view = self.slot_block_offsets.slice_mut(0..n);
        ctx.stream
            .memcpy_htod(&off_host, &mut off_view)
            .map_err(|e| anyhow!("DSv4 batched decode block-offset H2D failed: {e}"))?;
        if n > 0 {
            let row_width = page_tables[0].len();
            // Capacity check: the scratch is sized for the WIDEST layer's band
            // (max_total_blocks); narrower layers write a row_width prefix and the
            // kernel strides by this layer's shape.total_blocks (== row_width).
            ensure!(
                row_width * self.max_batch <= self.page_table_batched.len(),
                "DSv4 batched decode page-table width {} exceeds scratch len {} max_batch {}",
                row_width,
                self.page_table_batched.len(),
                self.max_batch
            );
            let mut host = vec![0_i32; n * row_width];
            for (r, table) in page_tables.iter().enumerate() {
                ensure!(
                    table.len() == row_width,
                    "DSv4 batched decode page-table row {} len {} != width {}",
                    r,
                    table.len(),
                    row_width
                );
                host[r * row_width..(r + 1) * row_width].copy_from_slice(table);
            }
            let mut table_view = self.page_table_batched.slice_mut(0..host.len());
            ctx.stream
                .memcpy_htod(&host, &mut table_view)
                .map_err(|e| anyhow!("DSv4 batched decode page-table H2D failed: {e}"))?;
        }
        Ok(())
    }

    /// CSA ONLY: gather row `r`'s indexer top-k `selected` into the contiguous
    /// `selected_batched[r * csa_topk ..]` band the batched index build reads
    /// (`selected + row * max_compressed_keys`, max_compressed_keys == csa_topk
    /// for CSA). `selected` is `[index_topk]` i32 (decode seq_len=1); its length
    /// must equal `csa_topk` (the single config top-k width). A single per-row
    /// `memcpy_dtod` on `ctx.stream` — stream-ordered before the index build that
    /// reads the buffer. Disjoint from `indices`/`q_batched`/etc.
    pub(crate) fn gather_selected_row(
        &mut self,
        ctx: &DeviceContext,
        selected: &CudaSlice<i32>,
        r: usize,
    ) -> Result<()> {
        ensure!(
            r < self.max_batch,
            "DSv4 batched CSA selected gather row {r} exceeds max_batch {}",
            self.max_batch
        );
        ensure!(
            self.csa_topk > 0,
            "DSv4 batched CSA selected gather: csa_topk is 0 (model has no CSA layer)"
        );
        ensure!(
            selected.len() == self.csa_topk,
            "DSv4 batched CSA selected gather: row {r} selected len {} != csa_topk {}",
            selected.len(),
            self.csa_topk
        );
        let mut dst = self
            .selected_batched
            .slice_mut(r * self.csa_topk..(r + 1) * self.csa_topk);
        ctx.stream
            .memcpy_dtod(selected, &mut dst)
            .map_err(|e| anyhow!("DSv4 batched CSA selected gather D2D failed: {e}"))?;
        Ok(())
    }

    /// Mutable handle to the `[max_batch, csa_topk]` CSA per-row top-k buffer the
    /// batched index build reads (`selected + row * csa_topk`). The batched CSA
    /// select (`csa_select_official_batched`, #60) writes the N rows here directly,
    /// replacing the per-row `gather_selected_row` copies. The buffer's row stride
    /// (`csa_topk`) equals `config.index_topk`, so the batched select's
    /// `out_selected` row stride matches.
    pub(crate) fn selected_batched_mut(&mut self) -> &mut CudaSlice<i32> {
        &mut self.selected_batched
    }

    /// Phase A convenience: build a layer's complete per-forward batched
    /// metadata — `upload_row_meta(n)` → `build_indices_batched(b=N)` →
    /// `sched_meta_for_batch(b=N)` — using the layer's STORED decode shape
    /// (`self.layer_shapes[layer_idx]`, the exact per-mode shape). SW/HCA pass
    /// `selected_ptr=0`; CSA sources the per-row top-k from `self.selected_batched`
    /// (which the prepare loop filled this forward via `gather_selected_row`).
    /// Leaves `self.indices`/`topk_length`/`sched_meta`/`num_splits` populated
    /// for this layer, ready for `sparse_decode_fwd_batched`.
    pub(crate) fn build_layer_batch_meta(
        &mut self,
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        layer_idx: usize,
        mode: DeepSeekV4AttentionMode,
        compress_ratio: usize,
        start_positions: &[usize],
        slot_block_offsets: &[usize],
        page_tables: &[Vec<i32>],
    ) -> Result<()> {
        let n = start_positions.len();
        let shape = *self.layer_shapes.get(layer_idx).ok_or_else(|| {
            anyhow!(
                "DSv4 batched decode layer {layer_idx} outside stored shapes {}",
                self.layer_shapes.len()
            )
        })?;
        self.upload_row_meta(ctx, start_positions, slot_block_offsets, page_tables)?;
        // Indexer modes (CSA + GLM SparseIndexed): feed the gathered per-row top-k
        // buffer; SW/HCA: selected_ptr=0 (byte-identical to the prior SW/HCA-only
        // batched index build). The kernel asserts `mode==CSA ⟹ selected != null`
        // (build_indices.cu:318); SparseIndexed shares mode_int=1, so its ptr must
        // be live too — `gather_selected_row` ran for every row in prepare.
        let selected_ptr = if mode.has_indexer() {
            ensure!(
                self.csa_topk == shape.max_compressed_keys,
                "DSv4 batched indexer: csa_topk {} != layer {layer_idx} max_compressed_keys {} \
                 (gather row stride would mismatch the kernel read stride)",
                self.csa_topk,
                shape.max_compressed_keys
            );
            let (ptr, guard) = self.selected_batched.device_ptr(&ctx.stream);
            drop(guard);
            ptr
        } else {
            0
        };
        self.build_indices_batched(ctx, n, mode, compress_ratio, config, &shape, selected_ptr)?;
        self.sched_meta_for_batch(ctx, n, shape.topk_unified)?;
        Ok(())
    }

    /// ONE batched indices build over all `n` rows (the orphaned
    /// `dsv4_flashmla_decode_build_indices_batched` kernel, #60). Emits
    /// `indices[n, topk_unified]` in pool-absolute block coords (slot→pool via
    /// `slot_block_offsets`) and `topk_length[n]`. `mode`/`compress_ratio` are
    /// uniform within a layer; `total_blocks` is this layer's `sw+comp` block
    /// count (bounds the offset translation). `selected` is the CSA per-row topk
    /// (`[n, max_compressed_keys]`) device ptr, or 0 for SW/HCA.
    ///
    /// Precondition: `upload_row_meta(n)` already ran this forward (reads
    /// `self.start_pos`/`self.slot_block_offsets` `[0,n)`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_indices_batched(
        &mut self,
        ctx: &DeviceContext,
        n: usize,
        mode: DeepSeekV4AttentionMode,
        compress_ratio: usize,
        config: &DeepSeekV4Config,
        shape: &Dsv4FlashMlaDecodeShape,
        selected_ptr: u64,
    ) -> Result<()> {
        ensure!(
            n > 0 && n <= self.max_batch,
            "DSv4 batched indices n={n} invalid"
        );
        ensure!(
            shape.topk_unified <= self.max_topk_unified,
            "DSv4 batched indices layer topk {} exceeds scratch {}",
            shape.topk_unified,
            self.max_topk_unified
        );
        let mode_int = flashmla_mode_int(mode);
        // Single-source sw_blocks / page_block_size for the batched index build.
        let bmap = shape.block_map();
        let (indices_ptr, indices_guard) = self.indices.device_ptr_mut(&ctx.stream);
        let (start_ptr, start_guard) = self.start_pos.device_ptr(&ctx.stream);
        let (off_ptr, off_guard) = self.slot_block_offsets.device_ptr(&ctx.stream);
        let (topk_ptr, topk_guard) = self.topk_length.device_ptr_mut(&ctx.stream);
        flash_kv::dsv4_flashmla_decode_build_indices_batched_raw(
            ctx,
            indices_ptr,
            start_ptr,
            off_ptr,
            selected_ptr,
            topk_ptr,
            n,
            bmap.sw_blocks(),
            config.sliding_window,
            shape.max_compressed_keys,
            // GLM SparseIndexed: full-sequence indexer, every token a key
            // (ratio=1); the kernel causality gate block_end = c*ratio + (ratio-1)
            // needs ratio=1. SW also passes 1; CSA/HCA keep compress_ratio.
            if mode == DeepSeekV4AttentionMode::SlidingWindow
                || mode == DeepSeekV4AttentionMode::SparseIndexed
            {
                1
            } else {
                compress_ratio
            },
            mode_int,
            bmap.page_size(),
            // POOL-absolute offset bound (Phase-B correctness fix). `slot_block_offsets[r]`
            // = slot_idx × shape.total_blocks is a POOL-absolute block offset (the pool
            // packs `num_slots == max_batch` contiguous per-slot bands), so the batched
            // kernel's `block_offset >= bound` guard must use the WHOLE-POOL block count,
            // not the per-slot `total_blocks` — the per-slot bound masked every index of
            // every row r≥1 to -1 (offset r*tb >= tb), yielding garbled decode (KILL
            // errors/2026-06-14-dsv4-batched-flashmla-decode-phaseB-correctness-kill.md).
            shape.total_blocks * self.max_batch,
            Some(&self.page_table_batched),
            // Fixed row stride the host writes at (upload_row_meta row_width =
            // page_table_batched.len()/max_batch = per-slot total_blocks),
            // independent of active n — the kernel reads page_table + row*width.
            shape.total_blocks,
        )?;
        drop(indices_guard);
        drop(start_guard);
        drop(off_guard);
        drop(topk_guard);
        Ok(())
    }

    /// Recompute the tile-scheduler metadata + split offsets for `b = n` THIS
    /// forward (the cached-constant pitfall, #60: the b=1 cached meta is wrong
    /// for n>1). Reads `self.topk_length[0..n]` (filled by `build_indices_batched`),
    /// writes `self.sched_meta` + `self.num_splits[0..=n]`. `layer_topk` is this
    /// layer's `topk_unified` (uniform per row in one layer).
    pub(super) fn sched_meta_for_batch(
        &mut self,
        ctx: &DeviceContext,
        n: usize,
        layer_topk: usize,
    ) -> Result<()> {
        ensure!(
            n > 0 && n <= self.max_batch,
            "DSv4 batched sched_meta n={n} invalid"
        );
        let topk = i32::try_from(layer_topk)
            .map_err(|_| anyhow!("DSv4 batched sched_meta topk {layer_topk} overflows i32"))?;
        let (topk_ptr, topk_guard) = self.topk_length.device_ptr(&ctx.stream);
        let (sched_ptr, sched_guard) = self.sched_meta.device_ptr_mut(&ctx.stream);
        let (splits_ptr, splits_guard) = self.num_splits.device_ptr_mut(&ctx.stream);
        // SAFETY: ptrs from live device allocations sized to the dims passed.
        unsafe {
            ffi::arle_flashmla_sm90_sparse_decode_sched_meta(
                n as i32,
                1,
                self.block_size_topk,
                self.fixed_overhead_num_blocks,
                topk,
                0,
                topk_ptr as *const i32,
                std::ptr::null(),
                sched_ptr as *mut i32,
                splits_ptr as *mut i32,
                self.num_sm_parts,
                ctx.stream.cu_stream(),
            )
            .result()
            .map_err(|e| anyhow!("DSv4 batched FlashMLA sched_meta failed: {e}"))?;
        }
        drop(topk_guard);
        drop(sched_guard);
        drop(splits_guard);
        Ok(())
    }

    /// ONE batched `sparse_decode_fwd(b=n)` over the whole shared KV pool base
    /// (#60 — the actual concurrency win, one launch over N rows replacing the
    /// per-row loop). `q_ptr` is `[n, h_q*head_dim]` bf16 (= `self.q_batched`
    /// prefix on the TP path, or the row-major global Q on single-GPU);
    /// `pool_ptr` is the layer's whole FP8 KV pool base (indices are
    /// pool-absolute); `out_ptr` is `[n, h_q*head_dim]` bf16.
    ///
    /// Precondition: `build_indices_batched(n)` + `sched_meta_for_batch(n)` ran
    /// this forward; `shape` is this layer's decode shape.
    ///
    /// PHASE B (#60): the actual batched kernel call — ONE launch over `n` rows.
    /// Wired by [`Self::decode_lane_fwd`] (which sources `pool_ptr`/`sink_ptr`
    /// and the `q_batched`/`out_batched` bases) after the per-row pack/gather
    /// loop fills `q_batched[0..n]` and `build_layer_batch_meta(n)` populated the
    /// indices/sched_meta. Stride mapping verified against the shim.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sparse_decode_fwd_batched(
        &mut self,
        ctx: &DeviceContext,
        n: usize,
        config: &DeepSeekV4Config,
        shape: &Dsv4FlashMlaDecodeShape,
        q_ptr: *const ffi::Half,
        pool_ptr: *const ffi::Half,
        sink_ptr: *const f32,
        out_ptr: *mut ffi::Half,
        sm_scale: f32,
    ) -> Result<()> {
        ensure!(
            n > 0 && n <= self.max_batch,
            "DSv4 batched fwd n={n} invalid"
        );
        // MODEL1 (head_dim=512, d_qk=d_v=512, 584 B/tok) and V32/GLM
        // (head_dim=576, d_qk=576, d_v=512 latent, 656 B/tok). Mirrors the
        // single-row decode path's dim mapping; the shim hard-asserts d_v==512.
        let meta = dsv4_flashmla_model_meta(config)?;
        let (model_type_int, bytes_per_token) = (meta.model_type_int, meta.bytes_per_token);
        let global_heads = shape.h_q;
        let head_dim = config.head_dim;
        let d_qk = head_dim as i32;
        let d_v = meta.d_v;
        let stride_kv_block_bytes = 64_i32 * bytes_per_token;
        let stride_q = (global_heads * head_dim) as i32; // s_q=1
        let stride_o = (global_heads as i32) * d_v;
        // Reader pitch MUST equal the writer pitch: the batched indices builder
        // (dsv4_flashmla_decode_build_indices.cu:210) writes row r at
        // `r * shape.topk_unified` (this layer's actual topk, mode-dependent), NOT
        // the scratch's allocated `max_topk_unified` (= max over ALL layers). Using
        // max here made row r≥1 read `r*max` while the writer put it at
        // `r*topk_unified` → garbled every row but row 0 (Phase-B correctness KILL,
        // numdiff FIRST-EXCEEDANCE L0 row1). SGLang's invariant: builder pitch ==
        // reader pitch by construction.
        let stride_indices = shape.topk_unified as i32; // writer pitch
        let stride_lse = global_heads as i32;
        let (indices_ptr, indices_guard) = self.indices.device_ptr(&ctx.stream);
        let (topk_ptr, topk_guard) = self.topk_length.device_ptr(&ctx.stream);
        let (sched_ptr, sched_guard) = self.sched_meta.device_ptr(&ctx.stream);
        let (splits_ptr, splits_guard) = self.num_splits.device_ptr(&ctx.stream);
        let (lse_out_ptr, lse_guard) = self.lse_out.device_ptr_mut(&ctx.stream);
        let (lse_accum_ptr, lse_accum_guard) = self.lse_accum.device_ptr_mut(&ctx.stream);
        let (o_accum_ptr, o_accum_guard) = self.o_accum.device_ptr_mut(&ctx.stream);
        // SAFETY: ptrs from live device allocations sized to the dims passed.
        unsafe {
            ffi::arle_flashmla_sm90_sparse_decode_fwd(
                q_ptr,
                pool_ptr,
                indices_ptr as *const i32,
                topk_ptr as *const i32,
                sink_ptr,
                out_ptr,
                lse_out_ptr as *mut f32,
                lse_accum_ptr as *mut f32,
                o_accum_ptr as *mut f32,
                sched_ptr as *const i32,
                splits_ptr as *const i32,
                n as i32,
                1,
                global_heads as i32,
                1,
                d_qk,
                d_v,
                (shape.sw_blocks + shape.comp_blocks) as i32,
                64,
                stride_indices,
                self.num_sm_parts,
                model_type_int,
                sm_scale,
                stride_q,
                stride_q,
                d_qk,
                stride_kv_block_bytes,
                bytes_per_token,
                stride_indices,
                stride_indices,
                stride_lse,
                1,
                stride_o,
                stride_o,
                d_v,
                // Split-KV accum strides: the accum buffers are `[num_sm_parts+b,
                // s_q=1, h_q(*d_v)]` (shim:202-203), so split stride = s_q*h_q etc.
                // — identical to the single-row path; b folds into the split index
                // via num_splits, NOT a stride.
                stride_lse, // stride_lse_accum_split = s_q*h_q = global_heads
                stride_lse, // stride_lse_accum_s_q  = h_q     = global_heads
                stride_o,   // stride_o_accum_split  = s_q*h_q*d_v
                stride_o,   // stride_o_accum_s_q    = h_q*d_v
                d_v,        // stride_o_accum_h_q = d_v
                ctx.stream.cu_stream(),
            )
            .result()
            .map_err(|e| anyhow!("DSv4 batched FlashMLA sparse decode failed: {e}"))?;
        }
        drop(indices_guard);
        drop(topk_guard);
        drop(sched_guard);
        drop(splits_guard);
        drop(lse_guard);
        drop(lse_accum_guard);
        drop(o_accum_guard);
        Ok(())
    }

    /// This layer's stored decode shape (PHASE B: the batched lane reads
    /// `topk_unified` / `sw_blocks` / `comp_blocks` / `h_q` per layer).
    pub(super) fn layer_shape(&self, layer_idx: usize) -> Result<Dsv4FlashMlaDecodeShape> {
        self.layer_shapes.get(layer_idx).copied().ok_or_else(|| {
            anyhow!(
                "DSv4 batched decode layer {layer_idx} outside stored shapes {}",
                self.layer_shapes.len()
            )
        })
    }

    pub(super) fn h_q_d(&self) -> usize {
        self.h_q * self.head_dim
    }

    /// Output row stride: `h_q * d_v` (d_v=512 always). For MODEL1 == h_q_d();
    /// for V32 (d_v=512 < head_dim=576) this is the out_batched/o_accum row pitch.
    pub(super) fn h_q_d_v(&self) -> usize {
        self.h_q * self.d_v
    }

    /// PHASE B: gather row `r`'s prepared local Q into `q_batched[r]` as
    /// global-head Q. Single-GPU (tp_world==1): a straight D2D copy of the
    /// `[local_heads*head_dim]` row (h_q==local_heads). TP: NCCL all-gather of
    /// this rank's Q into `tp_gathered_q`, then `dsv4_tp_q_repack_cuda` into
    /// `q_batched[r]` — identical to the single-row `try_flashmla_decode_attention`
    /// gather, run once per row instead of feeding a per-row fwd.
    pub(crate) fn gather_q_row(
        &mut self,
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        q_prepared: &HiddenStates,
        tp: &TpRuntime,
        r: usize,
        local_heads: usize,
    ) -> Result<()> {
        ensure!(
            r < self.max_batch,
            "DSv4 batched q row {r} >= max_batch {}",
            self.max_batch
        );
        let d = self.h_q_d();
        let tp_world = tp.config().world_size;
        let (q_ptr, q_guard) = q_prepared.data.device_ptr(&ctx.stream);
        if tp_world > 1 {
            let (gather_ptr, gather_guard) = self.tp_gathered_q.device_ptr_mut(&ctx.stream);
            {
                let _nvtx = crate::nvtx::range("dsv4/flashmla_q_allgather_batched");
                // SAFETY: per-rank Q is local_heads*head_dim bf16; the gather
                // landing buffer holds tp_world*local_heads*head_dim. Same
                // contract as the single-row path.
                unsafe {
                    tp.all_gather_bf16_raw(
                        ctx,
                        q_ptr as *const std::ffi::c_void,
                        local_heads * config.head_dim,
                        gather_ptr as *mut std::ffi::c_void,
                    )?;
                }
            }
            drop(gather_guard);
            let (gather_ptr, gather_guard) = self.tp_gathered_q.device_ptr(&ctx.stream);
            // `dst_view` holds the mutable borrow of `self.q_batched[r]` alive
            // across the FFI call; `dst_ptr` is its raw device ptr.
            let mut dst_view = self.q_batched.slice_mut(r * d..(r + 1) * d);
            let (dst_ptr, dst_guard) = dst_view.device_ptr_mut(&ctx.stream);
            drop(dst_guard);
            {
                let _nvtx = crate::nvtx::range("dsv4/flashmla_q_repack_batched");
                // SAFETY: repack tp_world×[local_heads,head_dim] gathered Q into
                // one global-head row (s_q=1); both buffers valid on ctx.stream.
                unsafe {
                    ffi::dsv4_tp_q_repack_cuda(
                        gather_ptr as *const ffi::Half,
                        dst_ptr as *mut ffi::Half,
                        tp_world as i32,
                        1,
                        local_heads as i32,
                        config.head_dim as i32,
                        ctx.stream.cu_stream(),
                    )
                    .result()
                    .map_err(|e| anyhow!("DSv4 batched FlashMLA TP Q repack failed: {e}"))?;
                }
            }
            drop(gather_guard);
            // dst_view drops at the end of this branch, ending the q_batched borrow.
        } else {
            // Single-GPU: global heads == local heads; copy the row verbatim.
            ensure!(
                q_prepared.hidden_dim == d && q_prepared.seq_len == 1,
                "DSv4 batched q row src {}x{} != [{d},1]",
                q_prepared.hidden_dim,
                q_prepared.seq_len
            );
            let mut dst = self.q_batched.slice_mut(r * d..(r + 1) * d);
            ctx.stream
                .memcpy_dtod(&q_prepared.data, &mut dst)
                .map_err(|e| anyhow!("DSv4 batched q row copy failed: {e}"))?;
        }
        drop(q_guard);
        Ok(())
    }

    /// PHASE B: read row `r`'s global-head fwd output from `out_batched` into the
    /// row's per-rank `local_attn`. Single-GPU: a straight D2D copy. TP:
    /// `dsv4_tp_out_slice_cuda` slices this rank's `[tp_rank*local_heads, +)`
    /// head block — identical to the single-row finish, run once per row.
    pub(crate) fn slice_out_row(
        &self,
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        tp: &TpRuntime,
        r: usize,
        local_heads: usize,
        local_attn: &mut HiddenStates,
    ) -> Result<()> {
        ensure!(
            r < self.max_batch,
            "DSv4 batched out row {r} >= max_batch {}",
            self.max_batch
        );
        let d = self.h_q_d_v();
        let tp_world = tp.config().world_size;
        let tp_rank = tp.config().rank;
        let local_width = local_heads * config.head_dim;
        ensure!(
            local_attn.hidden_dim == local_width && local_attn.seq_len == 1,
            "DSv4 batched out dst {}x{} != [{local_width},1]",
            local_attn.hidden_dim,
            local_attn.seq_len
        );
        let src_view = self.out_batched.slice(r * d..(r + 1) * d);
        if tp_world > 1 {
            let (src_ptr, src_guard) = src_view.device_ptr(&ctx.stream);
            let (dst_ptr, dst_guard) = local_attn.data.device_ptr_mut(&ctx.stream);
            {
                let _nvtx = crate::nvtx::range("dsv4/flashmla_out_slice_batched");
                // SAFETY: src is one global-head output row (h_q*d_v); dst this
                // rank's local block. Same args as the single-row out-slice.
                unsafe {
                    ffi::dsv4_tp_out_slice_cuda(
                        src_ptr as *const ffi::Half,
                        dst_ptr as *mut ffi::Half,
                        1,
                        self.h_q_d_v() as i32,
                        local_width as i32,
                        (tp_rank * local_width) as i32,
                        ctx.stream.cu_stream(),
                    )
                    .result()
                    .map_err(|e| anyhow!("DSv4 batched FlashMLA TP out slice failed: {e}"))?;
                }
            }
            drop(src_guard);
            drop(dst_guard);
        } else {
            ctx.stream
                .memcpy_dtod(&src_view, &mut local_attn.data)
                .map_err(|e| anyhow!("DSv4 batched out row copy failed: {e}"))?;
        }
        Ok(())
    }

    /// FINISH F1 batched: slice ALL `n` global-head rows of `out_batched` into a
    /// contiguous token-major `[n, local_width]` `local_attn_batched` in ONE
    /// launch. The n=1 result is byte-identical to a single `slice_out_row`:
    /// TP>1 calls the same `dsv4_tp_out_slice_cuda` with `s_q=n` (the kernel
    /// strides src by global width `h_q*d`, dst by `local_width` per row, slicing
    /// `[rank*local_width, +local_width]` — verified against the .cu); TP==1 is
    /// one D2D copy of the whole `[n, d]` block (local_width==global_width).
    pub(crate) fn slice_out_batched(
        &self,
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        tp: &TpRuntime,
        n: usize,
        local_heads: usize,
        local_attn_batched: &mut HiddenStates,
    ) -> Result<()> {
        ensure!(
            n > 0 && n <= self.max_batch,
            "DSv4 batched out-slice n={n} out of range (1..={})",
            self.max_batch
        );
        let d = self.h_q_d_v();
        let tp_world = tp.config().world_size;
        let tp_rank = tp.config().rank;
        let local_width = local_heads * config.head_dim;
        ensure!(
            local_attn_batched.hidden_dim == local_width && local_attn_batched.seq_len == n,
            "DSv4 batched out-slice dst {}x{} != [{local_width},{n}]",
            local_attn_batched.hidden_dim,
            local_attn_batched.seq_len
        );
        let src_view = self.out_batched.slice(0..n * d);
        if tp_world > 1 {
            let (src_ptr, src_guard) = src_view.device_ptr(&ctx.stream);
            let (dst_ptr, dst_guard) = local_attn_batched.data.device_ptr_mut(&ctx.stream);
            {
                let _nvtx = crate::nvtx::range("dsv4/flashmla_out_slice_batched");
                // SAFETY: src is `n` global-head output rows (stride h_q*d_v), dst is
                // n local rows (stride local_width); same per-row args as
                // `slice_out_row`, with s_q=n so the kernel loops rows internally.
                unsafe {
                    ffi::dsv4_tp_out_slice_cuda(
                        src_ptr as *const ffi::Half,
                        dst_ptr as *mut ffi::Half,
                        n as i32,
                        self.h_q_d_v() as i32,
                        local_width as i32,
                        (tp_rank * local_width) as i32,
                        ctx.stream.cu_stream(),
                    )
                    .result()
                    .map_err(|e| anyhow!("DSv4 batched FlashMLA TP out slice failed: {e}"))?;
                }
            }
            drop(src_guard);
            drop(dst_guard);
        } else {
            // TP==1: local_width == global_width == d, so the whole [n, d] block
            // copies contiguously into local_attn_batched.
            ctx.stream
                .memcpy_dtod(&src_view, &mut local_attn_batched.data)
                .map_err(|e| anyhow!("DSv4 batched out copy failed: {e}"))?;
        }
        Ok(())
    }

    /// PHASE B: ONE batched `sparse_decode_fwd(b=n)` for `layer_idx` over the
    /// shared FlashMLA pool base, reading the `q_batched[0..n]` prefix the
    /// per-row gather filled and writing global-head output into `out_batched`.
    /// Sources `pool_ptr` (the whole pool base — indices are pool-absolute),
    /// `sink_ptr` (the f32 sink base; the kernel computes ALL global heads so it
    /// needs the global sink, not a per-rank slice), and runs the kernel via
    /// [`Self::sparse_decode_fwd_batched`]. Precondition: `build_layer_batch_meta(n)`
    /// ran this forward for `layer_idx`.
    pub(crate) fn decode_lane_fwd(
        &mut self,
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        attention: &Dsv4Attention,
        pool: &mut Dsv4LayerKvLayout,
        layer_idx: usize,
        n: usize,
        sm_scale: f32,
    ) -> Result<()> {
        let shape = self.layer_shape(layer_idx)?;
        ensure!(
            attention
                .attn_sink_f32
                .as_ref()
                .expect("DSv4 attn_sink_f32")
                .len()
                >= shape.h_q,
            "DSv4 batched FlashMLA attn_sink_f32 len {} < global heads {}",
            attention
                .attn_sink_f32
                .as_ref()
                .expect("DSv4 attn_sink_f32")
                .len(),
            shape.h_q
        );
        // Resolve raw device pointers and release the cudarc borrow guards BEFORE
        // the `&mut self` fwd call (the SyncOnDrop guards would otherwise alias
        // `self.q_batched`/`self.out_batched`). The raw u64 ptrs stay valid —
        // the buffers are not reallocated; same single-stream discipline the
        // per-row path uses.
        let (sink_ptr, sink_guard) = attention
            .attn_sink_f32
            .as_ref()
            .expect("DSv4 attn_sink_f32")
            .device_ptr(&ctx.stream);
        let sink_ptr = sink_ptr as *const f32;
        drop(sink_guard);
        let (q_ptr, q_guard) = self.q_batched.device_ptr(&ctx.stream);
        let q_ptr = q_ptr as *const ffi::Half;
        drop(q_guard);
        let (out_ptr, out_guard) = self.out_batched.device_ptr_mut(&ctx.stream);
        let out_ptr = out_ptr as *mut ffi::Half;
        drop(out_guard);
        let pool_ptr = {
            let pool_buf = pool.flashmla_pool_data()?;
            let (pool_ptr, pool_guard) = pool_buf.device_ptr(&ctx.stream);
            let pool_ptr = pool_ptr as *const ffi::Half;
            drop(pool_guard);
            pool_ptr
        };
        self.sparse_decode_fwd_batched(
            ctx, n, config, &shape, q_ptr, pool_ptr, sink_ptr, out_ptr, sm_scale,
        )
    }
}

pub(super) struct Dsv4FusedWqkvDecodeScratch {
    pub(super) input_fp8: CudaSlice<u8>,
    pub(super) input_scales: CudaSlice<f32>,
    pub(super) qkv_raw: HiddenStates,
    pub(super) active_experts: CudaSlice<i32>,
    pub(super) active_offsets: CudaSlice<i32>,
    pub(super) active_counts: CudaSlice<i32>,
    pub(super) max_m: usize,
    pub(super) scale_stride_m: usize,
    pub(super) hidden_dim: usize,
    pub(super) q_lora_rank: usize,
    pub(super) head_dim: usize,
}

impl Dsv4FusedWqkvDecodeScratch {
    pub(super) fn new(ctx: &DeviceContext, config: &DeepSeekV4Config) -> Result<Self> {
        let max_m = 128;
        let scale_stride_m = 128;
        let hidden_dim = config.hidden_size;
        let q_lora_rank = config.q_lora_rank;
        let head_dim = config.head_dim;
        let scale_cols = hidden_dim.div_ceil(128);
        Ok(Self {
            input_fp8: ctx
                .stream
                .alloc_zeros::<u8>(max_m * hidden_dim)
                .map_err(|e| anyhow!("DSv4 fused wqkv input fp8 scratch alloc failed: {e}"))?,
            input_scales: ctx
                .stream
                .alloc_zeros::<f32>(scale_stride_m * scale_cols)
                .map_err(|e| anyhow!("DSv4 fused wqkv input scale scratch alloc failed: {e}"))?,
            // SAFETY: uninit device scratch; fully written before first read.
            qkv_raw: unsafe { HiddenStates::uninit(ctx, q_lora_rank + head_dim, 1)? },
            active_experts: ctx
                .stream
                .clone_htod(&[0_i32])
                .map_err(|e| anyhow!("DSv4 fused wqkv active_experts H2D failed: {e}"))?,
            active_offsets: ctx
                .stream
                .clone_htod(&[0_i32])
                .map_err(|e| anyhow!("DSv4 fused wqkv active_offsets H2D failed: {e}"))?,
            active_counts: ctx
                .stream
                .clone_htod(&[1_i32])
                .map_err(|e| anyhow!("DSv4 fused wqkv active_counts H2D failed: {e}"))?,
            max_m,
            scale_stride_m,
            hidden_dim,
            q_lora_rank,
            head_dim,
        })
    }

    /// Exact requested device bytes owned by this fused-wqkv decode scratch.
    #[allow(dead_code)]
    pub(crate) fn device_bytes(&self) -> usize {
        let i32_sz = std::mem::size_of::<i32>();
        let f32_sz = std::mem::size_of::<f32>();
        self.input_fp8.len() // u8
            + self.input_scales.len() * f32_sz
            + self.qkv_raw.device_bytes()
            + self.active_experts.len() * i32_sz
            + self.active_offsets.len() * i32_sz
            + self.active_counts.len() * i32_sz
    }

    /// STATIC predictor of `device_bytes` from config dims — MUST mirror `new`.
    /// Feeds `Dsv4Model::per_slot_device_bytes`.
    pub(crate) fn device_bytes_for(config: &DeepSeekV4Config) -> usize {
        let i32_sz = std::mem::size_of::<i32>();
        let f32_sz = std::mem::size_of::<f32>();
        let bf16 = std::mem::size_of::<half::bf16>();
        let max_m = 128usize;
        let scale_stride_m = 128usize;
        let hidden_dim = config.hidden_size;
        let scale_cols = hidden_dim.div_ceil(128);
        // input_fp8 (u8, max_m*hidden) + input_scales (f32, scale_stride_m*scale_cols)
        // + qkv_raw[q_lora_rank+head_dim, 1] (bf16) + active_experts/offsets/counts (1 i32 each).
        max_m * hidden_dim
            + scale_stride_m * scale_cols * f32_sz
            + (config.q_lora_rank + config.head_dim) * bf16
            + 3 * i32_sz
    }
}

pub(crate) struct Dsv4PrefillDeepGemmLinearScratch {
    pub(super) input_fp8: CudaSlice<u8>,
    pub(super) input_scales: CudaSlice<f32>,
    pub(super) qkv_raw: HiddenStates,
    pub(super) oproj_group_in: CudaSlice<half::bf16>,
    pub(super) oproj_group_out: CudaSlice<half::bf16>,
    pub(super) active_experts: CudaSlice<i32>,
    pub(super) active_offsets: CudaSlice<i32>,
    pub(super) active_counts: CudaSlice<i32>,
    pub(super) max_m: usize,
    pub(super) max_k: usize,
    pub(super) max_scale_stride_m: usize,
    pub(super) oproj_group_cols: usize,
    pub(super) oproj_group_rows: usize,
    pub(super) hidden_dim: usize,
    pub(super) q_lora_rank: usize,
    pub(super) head_dim: usize,
}

pub(super) fn dsv4_oproj_group_dims(config: &DeepSeekV4Config) -> Result<(usize, usize)> {
    ensure!(
        config.o_groups > 0 && config.num_attention_heads.is_multiple_of(config.o_groups),
        "DSv4 grouped wo_a scratch needs num_attention_heads {} divisible by o_groups {}",
        config.num_attention_heads,
        config.o_groups
    );
    let cols = (config.num_attention_heads / config.o_groups)
        .checked_mul(config.head_dim)
        .ok_or_else(|| anyhow!("DSv4 grouped wo_a cols overflow"))?;
    Ok((cols, config.o_lora_rank))
}

impl Dsv4PrefillDeepGemmLinearScratch {
    pub(super) fn new(
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        max_seq_len: usize,
    ) -> Result<Self> {
        // M (query/token) dimension is chunk-bounded: under chunked prefill the
        // layer forward processes at most `chunked_prefill_size` (<= chunk) query
        // tokens per call, so the M×K `input_fp8` / `input_scales` / `qkv_raw`
        // activation scratch is sized by `query_chunk`, not the slot's full
        // `max_seq_len` context (which would OOM at 900K). K = hidden_size stays
        // full. A debug assert at each forward call site (`prefill_proj_deepgemm`,
        // `run_fused_wqkv_prefill`) guards `seq_len <= max_m`.
        let query_chunk = DSV4_PREFILL_QUERY_CHUNK.min(max_seq_len.max(1));
        let max_m = query_chunk;
        let max_k = config.hidden_size;
        let q_lora_rank = config.q_lora_rank;
        let head_dim = config.head_dim;
        let (oproj_group_cols, oproj_group_rows) = dsv4_oproj_group_dims(config)?;
        let max_scale_stride_m = max_m.div_ceil(4) * 4;
        let scale_cols = max_k.div_ceil(128);
        Ok(Self {
            input_fp8: ctx
                .stream
                .alloc_zeros::<u8>(max_m.checked_mul(max_k).ok_or_else(|| {
                    anyhow!(
                        "DSv4 prefill DeepGEMM linear input scratch overflow: M={} K={}",
                        max_m,
                        max_k
                    )
                })?)
                .map_err(|e| anyhow!("DSv4 prefill DeepGEMM linear input alloc failed: {e}"))?,
            input_scales: ctx
                .stream
                .alloc_zeros::<f32>(max_scale_stride_m.checked_mul(scale_cols).ok_or_else(
                    || {
                        anyhow!(
                            "DSv4 prefill DeepGEMM linear scale scratch overflow: M={} K={}",
                            max_scale_stride_m,
                            max_k
                        )
                    },
                )?)
                .map_err(|e| anyhow!("DSv4 prefill DeepGEMM linear scales alloc failed: {e}"))?,
            // SAFETY: uninit device scratch; fully written before first read.
            qkv_raw: unsafe { HiddenStates::uninit(ctx, q_lora_rank + head_dim, max_m)? },
            oproj_group_in: ctx
                .stream
                .alloc_zeros::<half::bf16>(max_m.checked_mul(oproj_group_cols.max(1)).ok_or_else(
                    || {
                        anyhow!(
                            "DSv4 grouped wo_a input scratch overflow: M={} K={}",
                            max_m,
                            oproj_group_cols
                        )
                    },
                )?)
                .map_err(|e| anyhow!("DSv4 grouped wo_a input scratch alloc failed: {e}"))?,
            oproj_group_out: ctx
                .stream
                .alloc_zeros::<half::bf16>(max_m.checked_mul(oproj_group_rows.max(1)).ok_or_else(
                    || {
                        anyhow!(
                            "DSv4 grouped wo_a output scratch overflow: M={} N={}",
                            max_m,
                            oproj_group_rows
                        )
                    },
                )?)
                .map_err(|e| anyhow!("DSv4 grouped wo_a output scratch alloc failed: {e}"))?,
            active_experts: ctx
                .stream
                .clone_htod(&[0_i32])
                .map_err(|e| anyhow!("DSv4 prefill DeepGEMM active_experts H2D failed: {e}"))?,
            active_offsets: ctx
                .stream
                .clone_htod(&[0_i32])
                .map_err(|e| anyhow!("DSv4 prefill DeepGEMM active_offsets H2D failed: {e}"))?,
            active_counts: ctx
                .stream
                .clone_htod(&[0_i32])
                .map_err(|e| anyhow!("DSv4 prefill DeepGEMM active_counts H2D failed: {e}"))?,
            max_m,
            max_k,
            max_scale_stride_m,
            oproj_group_cols,
            oproj_group_rows,
            hidden_dim: config.hidden_size,
            q_lora_rank,
            head_dim,
        })
    }

    /// Exact requested device bytes owned by this ONE model-wide FP8 prefill
    /// DeepGEMM linear staging scratch.
    #[allow(dead_code)]
    pub(crate) fn device_bytes(&self) -> usize {
        let i32_sz = std::mem::size_of::<i32>();
        let f32_sz = std::mem::size_of::<f32>();
        self.input_fp8.len() // u8
            + self.input_scales.len() * f32_sz
            + self.qkv_raw.device_bytes()
            + self.oproj_group_in.len() * std::mem::size_of::<half::bf16>()
            + self.oproj_group_out.len() * std::mem::size_of::<half::bf16>()
            + self.active_experts.len() * i32_sz
            + self.active_offsets.len() * i32_sz
            + self.active_counts.len() * i32_sz
    }
}
