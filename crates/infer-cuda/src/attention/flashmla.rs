use super::*;
#[derive(Clone, Copy)]
pub(super) struct Dsv4FlashMlaDecodeShape {
    pub(super) sw_blocks: usize,
    pub(super) comp_blocks: usize,
    pub(super) max_compressed_keys: usize,
    pub(super) topk_unified: usize,
    pub(super) total_blocks: usize,
    pub(super) h_q: usize,
    /// Page size from the arena (asserted 64) — the one source for the block map
    /// and the pack/index kernel params.
    pub(super) page_block_size: usize,
}

impl Dsv4FlashMlaDecodeShape {
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
            // GLM SparseIndexed doesn't compress (ratio 0); treat it as 1 to mirror
            // the page budget in `dsv4_flashmla_slot_pages`.
            ensure!(
                compress_ratio > 0 || mode == DeepSeekV4AttentionMode::SparseIndexed,
                "DSv4 FlashMLA compressed decode requires non-zero ratio"
            );
            max_seq_len.div_ceil(indexer_stride(compress_ratio)).max(1)
        };
        let comp_blocks = compressed_rows.div_ceil(kv_arena.page_block_size);
        let max_compressed_keys = match mode {
            DeepSeekV4AttentionMode::SlidingWindow => 0,
            DeepSeekV4AttentionMode::CompressedSparse => config.index_topk,
            DeepSeekV4AttentionMode::HybridCompressed => compressed_rows.div_ceil(128) * 128,
            // GLM: sliding_window(0) + index_topk(2048) = 2048, a 128-multiple.
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
    /// Page size from the arena (asserted 64) — the one source for this slot's
    /// block map and pack/index kernel params.
    pub(super) page_block_size: usize,
    pub(super) fp8_kv_sw_bootstrapped: bool,
    pub(super) fp8_kv_comp_packed_rows: usize,
    pub(super) topk_length: CudaSlice<i32>,
    pub(super) sched_meta: CudaSlice<i32>,
    pub(super) num_splits: CudaSlice<i32>,
    pub(super) num_sm_parts: i32,
    pub(super) fixed_overhead_num_blocks: i32,
    pub(super) block_size_topk: i32,
    /// Lives as long as the slot so CUDA-graph-captured kernel args never
    /// reference a freed temporary (prefix-cache restore UAF, #8).
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

        let (num_sm_parts, fixed_overhead_num_blocks, block_size_topk) =
            flash_kv::flashmla_sm90_sparse_decode_get_meta(
                shape.h_q as i32,
                DSV4_FLASHMLA_S_Q as i32,
                DSV4_FLASHMLA_MODEL1,
            )
            .map_err(|e| anyhow!("DSv4 FlashMLA decode meta failed: {e}"))?;
        let num_sm_parts_max = (num_sm_parts as usize).max(256);

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
        // The zeroed page table is safe: kernels never read it before the first
        // band mirror marks the slot dirty and the refresh fills it.
        Ok(state)
    }

    /// Fill `topk_length` and the scheduler metadata ONCE: a per-step
    /// `memcpy_htod(&[topk], ..)` records a graph node whose host source is a dead
    /// stack temporary, so replay reads a dangling pointer.
    pub(super) fn init_constant_sched_meta(&mut self, ctx: &DeviceContext) -> Result<()> {
        let topk = i32::try_from(self.topk_unified)
            .map_err(|_| anyhow!("DSv4 FlashMLA topk {} overflows i32", self.topk_unified))?;
        ctx.stream
            .memcpy_htod(&[topk], &mut self.topk_length)
            .map_err(|e| anyhow!("DSv4 FlashMLA topk_length H2D failed: {e}"))?;
        let (topk_ptr, _tg) = self.topk_length.device_ptr(&ctx.stream);
        let (sched_ptr, _sg) = self.sched_meta.device_ptr_mut(&ctx.stream);
        let (splits_ptr, _pg) = self.num_splits.device_ptr_mut(&ctx.stream);
        {
            flash_kv::flashmla_sm90_sparse_decode_sched_meta_raw(
                &ctx.stream,
                1,
                1,
                self.block_size_topk,
                self.fixed_overhead_num_blocks,
                topk,
                0,
                topk_ptr,
                0,
                sched_ptr,
                splits_ptr,
                self.num_sm_parts,
            )
            .map_err(|e| anyhow!("DSv4 FlashMLA sched_meta failed: {e}"))?;
        }
        Ok(())
    }

    /// Re-sync the device page table from the host table — the graph-captured
    /// kernel arg points at this fixed buffer, so a stale copy reads garbage.
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
        // Host tables carry only real pages; the graph-captured device table is
        // fixed-size, so pad here.
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

    pub(crate) fn sw_blocks(&self) -> usize {
        self.sw_blocks
    }

    /// The one block→(page,row) map for this slot's band; the pack/index kernel
    /// params draw from it so they cannot drift.
    pub(crate) fn block_map(&self) -> Dsv4BlockMap {
        Dsv4BlockMap::new(self.sw_blocks, self.page_block_size)
    }

    /// Scheduler metadata only; the FP8 KV pool pages this slot reads are owned
    /// (and summed) by [`Dsv4LayerKvLayout::flashmla_kv_pool`].
    #[allow(dead_code)]
    pub(crate) fn device_bytes(&self) -> usize {
        let i32_sz = std::mem::size_of::<i32>();
        self.topk_length.len() * i32_sz
            + self.sched_meta.len() * i32_sz
            + self.num_splits.len() * i32_sz
    }

    /// Static estimate of `device_bytes`: `num_sm_parts` comes from a device-only
    /// FFI call, so use `new`'s 256 floor (~8 KB/layer).
    pub(crate) fn device_bytes_estimate() -> usize {
        let i32_sz = std::mem::size_of::<i32>();
        let num_sm_parts_max = 256usize;
        (1 + num_sm_parts_max * 8 + 2) * i32_sz
    }

    pub(crate) fn swap_out(
        &self,
        _ctx: &DeviceContext,
        _pool: &Dsv4LayerKvLayout,
    ) -> Result<crate::attention::Dsv4FlashMlaImage> {
        Ok(crate::attention::Dsv4FlashMlaImage {
            fp8_kv_sw_bootstrapped: self.fp8_kv_sw_bootstrapped,
            fp8_kv_comp_packed_rows: self.fp8_kv_comp_packed_rows,
        })
    }

    pub(crate) fn swap_in(
        &mut self,
        _ctx: &DeviceContext,
        _pool: &mut Dsv4LayerKvLayout,
        image: &crate::attention::Dsv4FlashMlaImage,
    ) -> Result<()> {
        self.fp8_kv_sw_bootstrapped = image.fp8_kv_sw_bootstrapped;
        self.fp8_kv_comp_packed_rows = image.fp8_kv_comp_packed_rows;
        Ok(())
    }
}

/// Model-wide per-forward scratch for the single-row (`s_q = 1`) FlashMLA sparse
/// decode lane. Buffers carry no cross-call or cross-slot state, so one shared
/// instance serves every (slot, layer): per-slot cost drops from ~1466 MB to
/// ~74 MB (`o_accum` alone is ~33.7 MB/layer × 43 layers). Every buffer is sized
/// for the worst case across FlashMLA layer shapes; `h_q` is uniform 64/128.
pub(crate) struct Dsv4FlashMlaDecodeScratch {
    /// `[sliding_window]` i32 — SW bulk pack block-ids.
    pub(super) sw_bulk_block_ids: CudaSlice<i32>,
    /// `[sliding_window]` i32 — SW bulk pack row offsets.
    pub(super) sw_bulk_rows: CudaSlice<i32>,
    /// `[1]` i32 — one-token SW pack block-id.
    pub(super) one_block_id: CudaSlice<i32>,
    /// `[1]` i32 — one-token SW pack row offset.
    pub(super) one_row: CudaSlice<i32>,
    /// `[max comp_slots]` i32 — compressed-delta bulk pack block-ids.
    pub(super) comp_block_ids: CudaSlice<i32>,
    /// `[max comp_slots]` i32 — compressed-delta bulk pack row offsets.
    pub(super) comp_rows: CudaSlice<i32>,
    /// `[max topk_unified]` i32 — unified sparse indices.
    pub(super) indices: CudaSlice<i32>,
    /// `[h_q]` f32 — decode-kernel LSE output.
    pub(super) lse_out: CudaSlice<f32>,
    /// `[(num_sm_parts+1) × h_q]` f32 — split-KV LSE accumulator.
    pub(super) lse_accum: CudaSlice<f32>,
    /// `[(num_sm_parts+1) × h_q × head_dim]` f32 — split-KV O accumulator.
    pub(super) o_accum: CudaSlice<f32>,
    /// `[h_q × head_dim]` bf16 — TP all-gather landing buffer for global-head Q.
    pub(super) tp_gathered_q: CudaSlice<half::bf16>,
    /// `[h_q × head_dim]` bf16 — repacked global-head Q (TP path).
    pub(super) tp_packed_q: CudaSlice<half::bf16>,
    /// `[h_q × head_dim]` bf16 — full global-head fwd output staging (TP path).
    pub(super) tp_full_out: CudaSlice<half::bf16>,
}

impl Dsv4FlashMlaDecodeScratch {
    /// `layer_shapes` is every FlashMLA layer's decode shape; all must share `h_q`.
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
        let sw_slots = config.sliding_window;
        // `comp_blocks * 64` upper-bounds each layer's `comp_slots`; SW layers have
        // `comp_blocks = 0`, hence the `.max(1)` floor.
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
        // num_sm_parts depends only on (h_q, s_q=1, model) — uniform across layers.
        let (num_sm_parts, _fixed_overhead_num_blocks, _block_size_topk) =
            flash_kv::flashmla_sm90_sparse_decode_get_meta(
                h_q as i32,
                DSV4_FLASHMLA_S_Q as i32,
                DSV4_FLASHMLA_MODEL1,
            )
            .map_err(|e| anyhow!("DSv4 single-row FlashMLA scratch meta failed: {e}"))?;
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

/// Model-wide scratch for the batched (`b = N`) FlashMLA sparse decode lane: one
/// `sparse_decode_fwd(b=N)` runs over N different slots sharing one layer's KV
/// pool, so the row-major buffers live in a single instance sized for
/// `max_batch`; a forward over `n ≤ max_batch` rows uses the `[0, n)` prefix.
/// The batched indices builder emits POOL-ABSOLUTE block coords, so the fwd
/// reads one pool base pointer.
pub(crate) struct Dsv4FlashMlaDecodeBatchScratch {
    /// `[max_batch, max_topk_unified]` i32 — per-row unified sparse indices in
    /// pool-absolute block coords.
    pub(super) indices: CudaSlice<i32>,
    /// `[max_batch]` i32 — per-row effective topk length.
    pub(super) topk_length: CudaSlice<i32>,
    /// `[max_batch]` i32 — per-row absolute decode position (causal gate input).
    pub(super) start_pos: CudaSlice<i32>,
    /// `[max_batch]` i32 — each row's slot's first FlashMLA pool block.
    pub(super) slot_block_offsets: CudaSlice<i32>,
    /// `[max_batch, total_blocks]` i32 — padded per-row logical→physical page
    /// tables; fragmented rows route to pool-absolute blocks and the kernel then
    /// skips `slot_block_offsets`.
    pub(super) page_table_batched: CudaSlice<i32>,
    /// `[max_batch, h_q]` f32 — per-row LSE output of the sparse decode.
    pub(super) lse_out: CudaSlice<f32>,
    /// `[num_sm_parts + max_batch, h_q]` f32 — split-KV LSE accumulator, shared
    /// across the batch (b folds into the split dim via `num_splits`).
    pub(super) lse_accum: CudaSlice<f32>,
    /// `[num_sm_parts + max_batch, h_q * d_v]` f32 — split-KV O accumulator.
    pub(super) o_accum: CudaSlice<f32>,
    /// `[num_sm_parts_max * 8]` i32 — tile-scheduler metadata, recomputed per
    /// forward: a cached b=1 meta merges split-KV wrongly for n>1.
    pub(super) sched_meta: CudaSlice<i32>,
    /// `[max_batch + 1]` i32 — split offsets, written per forward by `sched_meta`.
    pub(super) num_splits: CudaSlice<i32>,
    /// `[max_batch, h_q * head_dim]` bf16 — the fwd's q: gathered+repacked
    /// global-head Q (TP) or the row-major batched Q (single-GPU).
    pub(super) q_batched: CudaSlice<half::bf16>,
    /// `[max_batch, h_q * d_v]` bf16 — full global-head fwd output.
    pub(super) out_batched: CudaSlice<half::bf16>,
    /// `[max_batch, tp_world * local_heads * head_dim]` bf16 — TP all-gather
    /// landing buffer; the gather loop uses the `[0, h_q_d)` slice per row.
    pub(super) tp_gathered_q: CudaSlice<half::bf16>,
    /// `[max_batch, csa_topk]` i32 — indexer modes only; the batched index build
    /// reads `selected + row * max_compressed_keys`. Zero-len and never read when
    /// the model has no CSA layer (SW/HCA pass `selected_ptr = 0`).
    pub(super) selected_batched: CudaSlice<i32>,
    /// Row stride of `selected_batched` (= `config.index_topk`); 0 if the model
    /// has no CSA layer.
    pub(super) csa_topk: usize,
    /// Per-layer decode shape, indexed by layer, so the dsv4 loop needs only
    /// `layer_idx`.
    pub(super) layer_shapes: Vec<Dsv4FlashMlaDecodeShape>,
    pub(super) max_batch: usize,
    pub(super) max_topk_unified: usize,
    /// Global heads (64/128).
    pub(super) h_q: usize,
    pub(super) head_dim: usize,
    /// Output latent dim: 512 always; == head_dim for MODEL1, < head_dim for V32.
    pub(super) d_v: usize,
    pub(super) num_sm_parts: i32,
    pub(super) fixed_overhead_num_blocks: i32,
    pub(super) block_size_topk: i32,
}

impl Dsv4FlashMlaDecodeBatchScratch {
    /// `layer_shapes` is every FlashMLA layer's decode shape (so the buffers cover
    /// the worst mode); all must share `h_q` and `head_dim`.
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
        // Scheduler tuning meta depends only on (h_q, s_q=1, model).
        let (num_sm_parts, fixed_overhead_num_blocks, block_size_topk) =
            flash_kv::flashmla_sm90_sparse_decode_get_meta(
                h_q as i32,
                DSV4_FLASHMLA_S_Q as i32,
                model_type_int,
            )
            .map_err(|e| anyhow!("DSv4 batched FlashMLA decode meta failed: {e}"))?;
        let num_sm_parts_max = (num_sm_parts as usize).max(256);
        // The shim's accum split dimension is `num_sm_parts + b`, not
        // `b * num_sm_parts` (arle_flashmla_decode_shim.cu:202-203).
        let accum_splits_max = num_sm_parts_max + max_batch;
        let h_q_d = h_q
            .checked_mul(head_dim)
            .ok_or_else(|| anyhow!("DSv4 batched FlashMLA h_q*d overflow"))?;
        // d_v=512 always (the shim hard-asserts); == head_dim only for MODEL1.
        let d_v = if model_type_int == DSV4_FLASHMLA_V32 {
            512
        } else {
            head_dim
        };
        let h_q_d_v = h_q
            .checked_mul(d_v)
            .ok_or_else(|| anyhow!("DSv4 batched FlashMLA h_q*d_v overflow"))?;
        // Sized for the widest layer's band; narrower layers use a row_width prefix.
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
            tp_gathered_q: ctx.stream.alloc_zeros::<half::bf16>(max_batch * h_q_d)?,
            // `config.index_topk` is the one top-k width every CSA layer uses, so a
            // single row stride serves all of them; 0 → zero-len, never read.
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

    /// Upload the per-row decode positions, slot→pool block offsets and padded
    /// page tables for an `n`-row forward: writes the `[0, n)` prefix of
    /// `start_pos` / `slot_block_offsets` and `[0, n*row_width)` of
    /// `page_table_batched`.
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

    /// Gather row `r`'s indexer top-k into `selected_batched[r * csa_topk ..]`.
    /// `selected` is `[index_topk]` i32 and its length must equal `csa_topk`; the
    /// D2D is stream-ordered before the index build that reads the buffer.
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

    /// Handle for the batched CSA select to write all N rows directly; its
    /// `out_selected` row stride must be `csa_topk`.
    pub(crate) fn selected_batched_mut(&mut self) -> &mut CudaSlice<i32> {
        &mut self.selected_batched
    }

    /// Build a layer's complete per-forward batched metadata, leaving
    /// `indices`/`topk_length`/`sched_meta`/`num_splits` ready for
    /// `sparse_decode_fwd_batched`. Indexer modes require `selected_batched` to
    /// have been filled this forward.
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
        // The kernel asserts `mode==CSA ⟹ selected != null` (build_indices.cu:318);
        // SparseIndexed shares mode_int=1, so its ptr must be live too.
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

    /// Emits `indices[n, topk_unified]` in pool-absolute block coords and
    /// `topk_length[n]`. `selected_ptr` is the CSA per-row topk
    /// (`[n, max_compressed_keys]`) device ptr, or 0 for SW/HCA.
    ///
    /// Precondition: `upload_row_meta(n)` already ran this forward.
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
        // Indexer modes (CSA + GLM SparseIndexed) share mode_int=1; SW/HCA use 2.
        let mode_int = mode.flashmla_mode_int();
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
            // SparseIndexed is a full-sequence indexer, so the kernel's causality
            // gate `block_end = c*ratio + (ratio-1)` needs ratio=1, as does SW.
            if mode == DeepSeekV4AttentionMode::SlidingWindow
                || mode == DeepSeekV4AttentionMode::SparseIndexed
            {
                1
            } else {
                compress_ratio
            },
            mode_int,
            bmap.page_size(),
            // `slot_block_offsets[r]` is pool-absolute, so the kernel's
            // `block_offset >= bound` guard needs the whole-pool block count; the
            // per-slot count masks every index of every row r≥1 to -1.
            shape.total_blocks * self.max_batch,
            Some(&self.page_table_batched),
            // Fixed row stride the host writes at, independent of active n.
            shape.total_blocks,
        )?;
        drop(indices_guard);
        drop(start_guard);
        drop(off_guard);
        drop(topk_guard);
        Ok(())
    }

    /// Recompute the tile-scheduler metadata + split offsets for `b = n` this
    /// forward (a cached b=1 meta is wrong for n>1). Reads `topk_length[0..n]`,
    /// writes `sched_meta` + `num_splits[0..=n]`.
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
        {
            flash_kv::flashmla_sm90_sparse_decode_sched_meta_raw(
                &ctx.stream,
                n as i32,
                1,
                self.block_size_topk,
                self.fixed_overhead_num_blocks,
                topk,
                0,
                topk_ptr,
                0,
                sched_ptr,
                splits_ptr,
                self.num_sm_parts,
            )
            .map_err(|e| anyhow!("DSv4 batched FlashMLA sched_meta failed: {e}"))?;
        }
        drop(topk_guard);
        drop(sched_guard);
        drop(splits_guard);
        Ok(())
    }

    /// One batched `sparse_decode_fwd(b=n)` over the whole shared KV pool base.
    /// `q_ptr` is `[n, h_q*head_dim]` bf16, `pool_ptr` the layer's whole FP8 KV
    /// pool base (indices are pool-absolute), `out_ptr` `[n, h_q*head_dim]` bf16.
    ///
    /// Precondition: `build_indices_batched(n)` + `sched_meta_for_batch(n)` ran
    /// this forward; `shape` is this layer's decode shape.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sparse_decode_fwd_batched(
        &mut self,
        ctx: &DeviceContext,
        n: usize,
        config: &DeepSeekV4Config,
        shape: &Dsv4FlashMlaDecodeShape,
        q_ptr: u64,
        pool_ptr: u64,
        sink_ptr: u64,
        out_ptr: u64,
        sm_scale: f32,
    ) -> Result<()> {
        ensure!(
            n > 0 && n <= self.max_batch,
            "DSv4 batched fwd n={n} invalid"
        );
        let meta = dsv4_flashmla_model_meta(config)?;
        let (model_type_int, bytes_per_token) = (meta.model_type_int, meta.bytes_per_token);
        let global_heads = shape.h_q;
        let head_dim = config.head_dim;
        let d_qk = head_dim as i32;
        let d_v = meta.d_v;
        let stride_kv_block_bytes = 64_i32 * bytes_per_token;
        let stride_q = (global_heads * head_dim) as i32; // s_q=1
        let stride_o = (global_heads as i32) * d_v;
        // Reader pitch must equal the writer pitch: the builder writes row r at
        // `r * shape.topk_unified`, not the scratch's `max_topk_unified`.
        let stride_indices = shape.topk_unified as i32;
        let stride_lse = global_heads as i32;
        let (indices_ptr, indices_guard) = self.indices.device_ptr(&ctx.stream);
        let (topk_ptr, topk_guard) = self.topk_length.device_ptr(&ctx.stream);
        let (sched_ptr, sched_guard) = self.sched_meta.device_ptr(&ctx.stream);
        let (splits_ptr, splits_guard) = self.num_splits.device_ptr(&ctx.stream);
        let (lse_out_ptr, lse_guard) = self.lse_out.device_ptr_mut(&ctx.stream);
        let (lse_accum_ptr, lse_accum_guard) = self.lse_accum.device_ptr_mut(&ctx.stream);
        let (o_accum_ptr, o_accum_guard) = self.o_accum.device_ptr_mut(&ctx.stream);
        {
            flash_kv::flashmla_sm90_sparse_decode_fwd_raw(
                &ctx.stream,
                q_ptr,
                pool_ptr,
                indices_ptr,
                topk_ptr,
                sink_ptr,
                out_ptr,
                lse_out_ptr,
                lse_accum_ptr,
                o_accum_ptr,
                sched_ptr,
                splits_ptr,
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
                // Accum buffers are `[num_sm_parts+b, s_q=1, h_q(*d_v)]`, so b folds
                // into the split index via num_splits rather than into a stride.
                stride_lse, // stride_lse_accum_split = s_q*h_q = global_heads
                stride_lse, // stride_lse_accum_s_q  = h_q     = global_heads
                stride_o,   // stride_o_accum_split  = s_q*h_q*d_v
                stride_o,   // stride_o_accum_s_q    = h_q*d_v
                d_v,        // stride_o_accum_h_q = d_v
            )
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

    /// Output row pitch of out_batched/o_accum; `< h_q_d()` for V32 (d_v=512).
    pub(super) fn h_q_d_v(&self) -> usize {
        self.h_q * self.d_v
    }

    /// Gather row `r`'s prepared local Q into `q_batched[r]` as global-head Q:
    /// a D2D copy on single-GPU, all-gather + repack under TP.
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
                crate::profile::profile_op(ctx, "flashmla_q_allgather_batched", None, 1, || {
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
                    Ok(())
                })?;
            }
            drop(gather_guard);
            let (gather_ptr, gather_guard) = self.tp_gathered_q.device_ptr(&ctx.stream);
            // `dst_view` keeps the borrow of `q_batched[r]` alive across the FFI call.
            let mut dst_view = self.q_batched.slice_mut(r * d..(r + 1) * d);
            let (dst_ptr, dst_guard) = dst_view.device_ptr_mut(&ctx.stream);
            drop(dst_guard);
            {
                crate::profile::profile_op(ctx, "flashmla_q_repack_batched", None, 1, || {
                    // SAFETY: repack tp_world×[local_heads,head_dim] gathered Q into
                    // one global-head row (s_q=1); both buffers valid on ctx.stream.
                    {
                        flash_kv::dsv4_tp_q_repack_raw(
                            &ctx.stream,
                            gather_ptr,
                            dst_ptr,
                            tp_world as i32,
                            1,
                            local_heads as i32,
                            config.head_dim as i32,
                        )
                        .map_err(|e| anyhow!("DSv4 batched FlashMLA TP Q repack failed: {e}"))?;
                    }
                    Ok(())
                })?;
            }
            drop(gather_guard);
        } else {
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

    /// Read row `r`'s global-head fwd output into the row's per-rank `local_attn`:
    /// a D2D copy on single-GPU, this rank's head-block slice under TP.
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
                crate::profile::profile_op(ctx, "flashmla_out_slice_batched", None, 1, || {
                    // SAFETY: src is one global-head output row (h_q*d_v); dst this
                    // rank's local block. Same args as the single-row out-slice.
                    {
                        flash_kv::dsv4_tp_out_slice_raw(
                            &ctx.stream,
                            src_ptr,
                            dst_ptr,
                            1,
                            self.h_q_d_v() as i32,
                            local_width as i32,
                            (tp_rank * local_width) as i32,
                        )
                        .map_err(|e| anyhow!("DSv4 batched FlashMLA TP out slice failed: {e}"))?;
                    }
                    Ok(())
                })?;
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

    /// Slice all `n` global-head rows of `out_batched` into a contiguous
    /// token-major `[n, local_width]` `local_attn_batched` in one launch.
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
                crate::profile::profile_op(ctx, "flashmla_out_slice_batched", None, n, || {
                    // SAFETY: src is `n` global-head output rows (stride h_q*d_v), dst
                    // is
                    // n local rows (stride local_width); same per-row args as
                    // `slice_out_row`, with s_q=n so the kernel loops rows internally.
                    {
                        flash_kv::dsv4_tp_out_slice_raw(
                            &ctx.stream,
                            src_ptr,
                            dst_ptr,
                            n as i32,
                            self.h_q_d_v() as i32,
                            local_width as i32,
                            (tp_rank * local_width) as i32,
                        )
                        .map_err(|e| anyhow!("DSv4 batched FlashMLA TP out slice failed: {e}"))?;
                    }
                    Ok(())
                })?;
            }
            drop(src_guard);
            drop(dst_guard);
        } else {
            // TP==1: local_width == global_width, so the whole [n, d] block copies
            // contiguously.
            ctx.stream
                .memcpy_dtod(&src_view, &mut local_attn_batched.data)
                .map_err(|e| anyhow!("DSv4 batched out copy failed: {e}"))?;
        }
        Ok(())
    }

    /// Run `layer_idx`'s batched fwd over the `q_batched[0..n]` prefix into
    /// `out_batched`. The sink is the global f32 base, not a per-rank slice — the
    /// kernel computes all global heads. Precondition: `build_layer_batch_meta(n)`
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
        // Release the cudarc borrow guards before the `&mut self` fwd call, which
        // they would otherwise alias; the raw ptrs stay valid (no reallocation).
        let (sink_ptr, sink_guard) = attention
            .attn_sink_f32
            .as_ref()
            .expect("DSv4 attn_sink_f32")
            .device_ptr(&ctx.stream);
        drop(sink_guard);
        let (q_ptr, q_guard) = self.q_batched.device_ptr(&ctx.stream);
        drop(q_guard);
        let (out_ptr, out_guard) = self.out_batched.device_ptr_mut(&ctx.stream);
        drop(out_guard);
        let pool_ptr = {
            let pool_buf = pool.flashmla_pool_data()?;
            let (pool_ptr, pool_guard) = pool_buf.device_ptr(&ctx.stream);
            drop(pool_guard);
            pool_ptr
        };
        self.sparse_decode_fwd_batched(
            ctx, n, config, &shape, q_ptr, pool_ptr, sink_ptr, out_ptr, sm_scale,
        )
    }

    // TEMPORARY (#228/#229): remove after root-cause.
    /// D2H dump num_splits/topk_length + first indices per row for GPU diagnosis.
    pub(crate) fn debug_dump_sched_state(
        &self,
        ctx: &DeviceContext,
        n: usize,
        layer_idx: usize,
    ) -> Result<()> {
        let layer_topk = self.layer_shapes[layer_idx].topk_unified;
        let mut splits = vec![0i32; n + 1];
        ctx.stream
            .memcpy_dtoh(&self.num_splits, &mut splits)
            .map_err(|e| anyhow!("debug D2H num_splits failed: {e}"))?;
        let mut topk = vec![0i32; n];
        ctx.stream
            .memcpy_dtoh(&self.topk_length, &mut topk)
            .map_err(|e| anyhow!("debug D2H topk_length failed: {e}"))?;
        let copy_len = n * layer_topk;
        let indices_view = self.indices.slice(0..copy_len);
        let mut idx_host = vec![0i32; copy_len];
        ctx.stream
            .memcpy_dtoh(&indices_view, &mut idx_host)
            .map_err(|e| anyhow!("debug D2H indices failed: {e}"))?;
        let dump_len = layer_topk.min(8);
        eprintln!(
            "[batch-debug] layer={} n={} topk={:?} splits={:?}",
            layer_idx,
            n,
            &topk[..n],
            &splits[..=n]
        );
        for r in 0..n {
            let start = r * layer_topk;
            eprintln!(
                "[batch-debug] layer={} row={} idx[..{}]={:?}",
                layer_idx,
                r,
                dump_len,
                &idx_host[start..start + dump_len]
            );
        }
        Ok(())
    }

    // TEMPORARY (#228/#229): remove after root-cause.
    /// D2H dump per-row output checksums for GPU diagnosis.
    pub(crate) fn debug_dump_output(
        &self,
        ctx: &DeviceContext,
        n: usize,
        layer_idx: usize,
    ) -> Result<()> {
        let row_len = self.h_q * self.d_v;
        let mut host = vec![half::bf16::from_f32(0.0); n * row_len];
        ctx.stream
            .memcpy_dtoh(&self.out_batched, &mut host)
            .map_err(|e| anyhow!("debug D2H out_batched failed: {e}"))?;
        for r in 0..n {
            let row = &host[r * row_len..(r + 1) * row_len];
            let (sum_abs, max_abs, non_zero) =
                row.iter().fold((0.0f32, 0.0f32, 0usize), |(s, m, nz), &x| {
                    let v = x.to_f32();
                    (s + v.abs(), m.max(v.abs()), nz + usize::from(v != 0.0))
                });
            eprintln!(
                "[batch-debug] layer={} row={} sum_abs={:.4} max_abs={:.4} nz={}/{}",
                layer_idx, r, sum_abs, max_abs, non_zero, row_len
            );
        }
        Ok(())
    }
}

pub(super) struct Dsv4FusedWqkvDecodeScratch {
    pub(super) input_fp8: CudaSlice<u8>,
    pub(super) input_scales: CudaSlice<f32>,
    pub(super) qkv_raw: HiddenStates,
    pub(super) active_experts: CudaSlice<i32>,
    pub(super) active_offsets: CudaSlice<i32>,
    pub(super) active_counts: CudaSlice<i32>,
    /// Grouped O-LoRA gather/scatter staging for the one-row decode lane;
    /// fixed at construction so a captured decode graph keeps stable addresses.
    /// `Option` so the decode lane can `take()` them for the GEMM borrow and
    /// put them back: the placeholder buffers a `mem::replace` needed recorded
    /// two alloc nodes per layer into the capture.
    pub(super) oproj_in: Option<HiddenStates>,
    pub(super) oproj_out: Option<HiddenStates>,
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
        let (oproj_cols, oproj_rows) = dsv4_oproj_group_dims(config)?;
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
            // SAFETY: uninit device scratch; fully written before first read.
            oproj_in: Some(unsafe { HiddenStates::uninit(ctx, oproj_cols, 1)? }),
            // SAFETY: uninit device scratch; fully written before first read.
            oproj_out: Some(unsafe { HiddenStates::uninit(ctx, oproj_rows, 1)? }),
            max_m,
            scale_stride_m,
            hidden_dim,
            q_lora_rank,
            head_dim,
        })
    }

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
            + self.oproj_in.as_ref().map_or(0, HiddenStates::device_bytes)
            + self.oproj_out.as_ref().map_or(0, HiddenStates::device_bytes)
    }

    /// Static predictor of `device_bytes` from config dims — must mirror `new`.
    pub(crate) fn device_bytes_for(config: &DeepSeekV4Config) -> usize {
        let i32_sz = std::mem::size_of::<i32>();
        let f32_sz = std::mem::size_of::<f32>();
        let bf16 = std::mem::size_of::<half::bf16>();
        let max_m = 128usize;
        let scale_stride_m = 128usize;
        let hidden_dim = config.hidden_size;
        let scale_cols = hidden_dim.div_ceil(128);
        let (oproj_cols, oproj_rows) = dsv4_oproj_group_dims(config).unwrap_or((0, 0));
        max_m * hidden_dim
            + scale_stride_m * scale_cols * f32_sz
            + (config.q_lora_rank + config.head_dim) * bf16
            + 3 * i32_sz
            + (oproj_cols + oproj_rows) * bf16
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
        // M is chunk-bounded: sizing the activation scratch by the slot's full
        // `max_seq_len` instead would OOM at 900K. Call sites debug-assert
        // `seq_len <= max_m`.
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
