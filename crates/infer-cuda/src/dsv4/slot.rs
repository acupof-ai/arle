use super::*;

pub(crate) struct Dsv4SlotState {
    pub(super) attention: Vec<crate::attention::Dsv4LayerAttentionState>,
    /// Per-attention-layer one-slot snapshot of speculative-verify ring writes.
    /// `Some` only when `model.spec_decode_on`; index-aligned with `attention`.
    pub(super) spec_rings: Option<Vec<crate::attention::Dsv4SpecRingSnapshot>>,
    /// Per-layer attn-normed verify rows (`[hidden, MAX_SPEC_VERIFY_ROWS]`),
    /// persisted by the verify lane so the commit can re-ingest the accepted
    /// prefix without a second full forward. `Some` only when spec decode is on.
    pub(super) spec_normed: Option<Vec<HiddenStates>>,
    /// Scheduled MTP verify forward workspace: row-major layer temporaries kept
    /// resident so the tiny `depth + 1` row forward avoids the prefill allocator.
    pub(super) spec_verify: Option<Dsv4SpecVerifyScratch>,
    pub(crate) start_pos_device: CudaSlice<i32>,
    /// Pre-allocated NVSHMEM low-latency MoE scratch, overwritten in place each
    /// `dsv4_moe_forward_deepep_ll` call. One per slot suffices — layers run
    /// sequentially. `Some` only when the `deepep_ll` transport is booted.
    #[cfg(feature = "deepep")]
    pub(super) deepep_ll_scratch: Option<crate::deepep::DeepEpLlScratch>,
    /// Wide HC residual stream captured at each `config.dspark_target_layer_ids`
    /// layer OUTPUT, index-aligned with the id list. Empty unless DSpark.
    pub(super) dspark_taps: Vec<DeviceVec>,
    /// Transient multi-row prompt taps captured during a PREFILL forward, consumed
    /// once by the DSpark prefix seed. Prefill-scoped and bounded by the chunked-
    /// prefill chunk size — deliberately EXCLUDED from the per-slot VRAM ledger.
    pub(super) dspark_prompt_taps: Option<Dsv4DsparkPromptTaps>,
    pub(crate) seq_len: usize,
    pub(super) max_seq_len: usize,
}

/// Full-prompt-chunk tap capture for the DSpark prefix seed. Each buffer holds
/// one target layer's wide HC stream `[stream_dim * rows]` token-major (column
/// `j*hc_mult + r` at `(j*hc_mult + r) * hidden`), index-aligned with
/// `dspark_target_layer_ids`.
pub(crate) struct Dsv4DsparkPromptTaps {
    pub(crate) bufs: Vec<DeviceVec>,
    pub(crate) rows: usize,
}

impl Dsv4SlotState {
    pub(super) fn new(
        model: &Dsv4Model,
        max_seq_len: usize,
        slot_idx: usize,
        kv_adapter: &crate::attention::Dsv4KvAdapter,
    ) -> Result<Self> {
        ensure!(max_seq_len > 0, "DSv4 slot max_seq_len must be positive");
        let attention: Vec<_> = model
            .layers
            .iter()
            .enumerate()
            .map(|(layer_idx, layer)| {
                let local_width = layer.attention.wq_b.rows;
                ensure!(
                    local_width.is_multiple_of(model.config.head_dim),
                    "DSv4 slot attention local width {local_width} is not a multiple of head_dim {}",
                    model.config.head_dim
                );
                let pool = kv_adapter.layer(layer_idx)?;
                crate::attention::Dsv4LayerAttentionState::new(
                    &model.ctx,
                    &model.config,
                    layer.mode,
                    layer.compress_ratio,
                    max_seq_len,
                    &model.kv_arena,
                    local_width / model.config.head_dim,
                    model.tp.config().world_size,
                    slot_idx,
                    pool,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let spec_rings = model
            .spec_decode_on
            .then(|| {
                attention
                    .iter()
                    .map(|state| {
                        state.alloc_spec_ring_snapshot(
                            &model.ctx,
                            &model.config,
                            &model.kv_arena,
                            MAX_SPEC_DRAFT_DEPTH,
                        )
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        let spec_normed = model
            .spec_decode_on
            .then(|| {
                (0..attention.len())
                    .map(|_| {
                        // SAFETY: rows are written by the verify lane before any read.
                        unsafe {
                            HiddenStates::uninit(
                                &model.ctx,
                                model.config.hidden_size,
                                model.spec_verify_rows(),
                            )
                        }
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        let spec_verify = if model.spec_decode_on {
            Some(Dsv4SpecVerifyScratch::new(model)?)
        } else {
            None
        };
        let start_pos_device = model
            .ctx
            .stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| anyhow!("DSv4 slot start_pos device scalar alloc failed: {e}"))?;
        // `intermediate` is uniform across layers, so layer 0's value sizes it.
        #[cfg(feature = "deepep")]
        let deepep_ll_scratch = match model.deepep.as_ref() {
            Some(transport) if transport.is_low_latency() => {
                let intermediate = model
                    .layers
                    .first()
                    .map(|layer| layer.moe.as_ref().expect("DSv4 layer.moe").intermediate)
                    .ok_or_else(|| anyhow!("DSv4 deepep_ll: model has no layers"))?;
                Some(transport.alloc_ll_scratch(&model.ctx, intermediate)?)
            }
            _ => None,
        };
        let dspark_taps = if model.config.is_dspark() {
            let stream_dim = model.config.hidden_size * model.config.hc_mult;
            let rows = model.config.dspark_block_size + 1;
            model
                .config
                .dspark_target_layer_ids
                .iter()
                .map(|_| DeviceVec::zeros(&model.ctx, stream_dim * rows))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        Ok(Self {
            attention,
            spec_rings,
            spec_normed,
            spec_verify,
            start_pos_device,
            #[cfg(feature = "deepep")]
            deepep_ll_scratch,
            dspark_taps,
            dspark_prompt_taps: None,
            seq_len: 0,
            max_seq_len,
        })
    }

    /// Exact requested device bytes owned by this ONE slot. EXCLUDES
    /// `deepep_ll_scratch`: only `Some` when the NVSHMEM LL transport is booted,
    /// and sized off-band.
    pub(crate) fn device_bytes(&self) -> usize {
        self.device_bytes_breakdown().iter().map(|(_, b)| *b).sum()
    }

    #[allow(dead_code)]
    pub(crate) fn device_bytes_breakdown(&self) -> Vec<(&'static str, usize)> {
        let attention_bytes: usize = self.attention.iter().map(|s| s.device_bytes()).sum();
        let spec_rings_bytes: usize = self
            .spec_rings
            .as_ref()
            .map_or(0, |rings| rings.iter().map(|r| r.device_bytes()).sum());
        let spec_normed_bytes: usize = self
            .spec_normed
            .as_ref()
            .map_or(0, |cache| cache.iter().map(|h| h.device_bytes()).sum());
        let spec_verify_bytes = self
            .spec_verify
            .as_ref()
            .map_or(0, Dsv4SpecVerifyScratch::device_bytes);
        let dspark_taps_bytes: usize = self
            .dspark_taps
            .iter()
            .map(|v| v.len * std::mem::size_of::<half::bf16>())
            .sum();
        vec![
            ("attention(per-layer)", attention_bytes),
            ("spec_rings", spec_rings_bytes),
            ("spec_normed", spec_normed_bytes),
            ("spec_verify", spec_verify_bytes),
            (
                "start_pos_device",
                self.start_pos_device.len() * std::mem::size_of::<i32>(),
            ),
            ("dspark_taps", dspark_taps_bytes),
        ]
    }

    #[allow(dead_code)]
    pub(crate) fn dspark_taps(&self) -> &[DeviceVec] {
        &self.dspark_taps
    }

    #[allow(dead_code)]
    pub(crate) fn take_dspark_prompt_taps(&mut self) -> Option<Dsv4DsparkPromptTaps> {
        self.dspark_prompt_taps.take()
    }

    #[allow(dead_code)]
    pub(crate) fn attention_breakdown_total(&self) -> Vec<(&'static str, usize)> {
        let mut totals: std::collections::BTreeMap<&'static str, usize> =
            std::collections::BTreeMap::new();
        for layer in &self.attention {
            for (name, bytes) in layer.device_bytes_breakdown() {
                *totals.entry(name).or_insert(0) += bytes;
            }
        }
        totals.into_iter().collect()
    }

    pub(crate) fn capture_spec_rings(
        &mut self,
        ctx: &DeviceContext,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        start_pos: usize,
        depth: usize,
    ) -> Result<()> {
        let Some(rings) = self.spec_rings.as_mut() else {
            return Ok(());
        };
        ensure!(
            self.attention.len() == rings.len(),
            "DSv4 spec-ring layer count {} != attention states {}",
            rings.len(),
            self.attention.len()
        );
        for (layer_idx, (state, snap)) in self.attention.iter().zip(rings).enumerate() {
            let pool = kv_adapter.layer_mut(layer_idx)?;
            state.capture_spec_rings(ctx, pool, snap, start_pos, depth)?;
        }
        Ok(())
    }

    /// Restore the speculative boundary ring slot across all attention layers
    /// AFTER the commit truncate and BEFORE the accepted-prefix fold. No-op
    /// when spec decode is off.
    pub(crate) fn restore_spec_ring_tail(
        &mut self,
        ctx: &DeviceContext,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        start_pos: usize,
        accepted_n: usize,
        depth: usize,
    ) -> Result<()> {
        let Some(rings) = self.spec_rings.as_ref() else {
            return Ok(());
        };
        ensure!(
            self.attention.len() == rings.len(),
            "DSv4 spec-ring restore layer count {} != attention states {}",
            rings.len(),
            self.attention.len()
        );
        for (layer_idx, (state, snap)) in self.attention.iter_mut().zip(rings).enumerate() {
            let pool = kv_adapter.layer_mut(layer_idx)?;
            state.restore_spec_ring_tail(ctx, pool, snap, start_pos, accepted_n, depth)?;
        }
        Ok(())
    }

    pub(crate) fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub(crate) fn reset(
        &mut self,
        ctx: &DeviceContext,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
    ) -> Result<()> {
        self.seq_len = 0;
        self.dspark_prompt_taps = None;
        ctx.stream
            .memset_zeros(&mut self.start_pos_device)
            .map_err(|e| anyhow!("DSv4 slot start_pos reset failed: {e}"))?;
        for (layer_idx, layer) in self.attention.iter_mut().enumerate() {
            let pool = kv_adapter.layer_mut(layer_idx)?;
            layer.reset(ctx, pool)?;
        }
        Ok(())
    }

    /// Re-sync every layer's FlashMLA device page table from the host pool.
    /// Dirty-bit driven (prefill AND decode); host tables carry only real pages,
    /// padding to the fixed device size happens per layer.
    pub(crate) fn refresh_flashmla_device_page_tables(
        &mut self,
        ctx: &DeviceContext,
        kv_adapter: &crate::attention::Dsv4KvAdapter,
    ) -> Result<()> {
        for (layer_idx, layer) in self.attention.iter_mut().enumerate() {
            let pool = kv_adapter.layer(layer_idx)?;
            layer.refresh_flashmla_device_page_table(ctx, pool)?;
        }
        Ok(())
    }

    /// D2H one completed host page's state across every layer into a
    /// content-keyed pool entry. `boundary` = the forward ended exactly at this
    /// page's end (the only moment the page-end overlap registers + ring are
    /// observable). The D2H clones are stream-ordered; the CALLER syncs once
    /// after all pages of the tick, not per page.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn capture_prefix_page(
        &self,
        ctx: &DeviceContext,
        layers: &[Dsv4Layer],
        kv_adapter: &crate::attention::Dsv4KvAdapter,
        index_head_dim: usize,
        page_tokens: usize,
        page_index: usize,
        boundary: bool,
    ) -> Result<crate::attention::Dsv4PrefixPageEntry> {
        ensure!(
            layers.len() == self.attention.len(),
            "DSv4 prefix capture layer count {} != attention states {}",
            layers.len(),
            self.attention.len()
        );
        let states = self
            .attention
            .iter()
            .enumerate()
            .map(|(idx, state)| {
                let pool = kv_adapter.layer(idx)?;
                state.capture_prefix_page(
                    ctx,
                    pool,
                    layers[idx].mode,
                    layers[idx].compress_ratio,
                    index_head_dim,
                    page_tokens,
                    page_index,
                    boundary,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(crate::attention::Dsv4PrefixPageEntry {
            page_index: u32::try_from(page_index)
                .map_err(|_| anyhow!("DSv4 prefix page index {page_index} exceeds u32"))?,
            boundary,
            layers: states,
        })
    }

    /// Capture the finish frontier page: the frontier page's own content + carry
    /// PLUS the sub-page tail the radix match can't cover (`[matched_len,
    /// finish_len)`). The whole carry reflects `finish_len` because capture runs
    /// after the finish forward.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn capture_frontier_page(
        &self,
        ctx: &DeviceContext,
        layers: &[Dsv4Layer],
        kv_adapter: &crate::attention::Dsv4KvAdapter,
        index_head_dim: usize,
        page_tokens: usize,
        page_index: usize,
        matched_len: usize,
        finish_len: usize,
    ) -> Result<crate::attention::Dsv4PrefixPageEntry> {
        ensure!(
            layers.len() == self.attention.len(),
            "DSv4 frontier capture layer count {} != attention states {}",
            layers.len(),
            self.attention.len()
        );
        let states = self
            .attention
            .iter()
            .enumerate()
            .map(|(idx, state)| {
                let pool = kv_adapter.layer(idx)?;
                let mut s = state.capture_prefix_page(
                    ctx,
                    pool,
                    layers[idx].mode,
                    layers[idx].compress_ratio,
                    index_head_dim,
                    page_tokens,
                    page_index,
                    true,
                )?;
                state.capture_frontier_tail(
                    ctx,
                    pool,
                    layers[idx].mode,
                    layers[idx].compress_ratio,
                    index_head_dim,
                    matched_len,
                    finish_len,
                    &mut s,
                )?;
                Ok(s)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(crate::attention::Dsv4PrefixPageEntry {
            page_index: u32::try_from(page_index)
                .map_err(|_| anyhow!("DSv4 frontier page index {page_index} exceeds u32"))?,
            boundary: true,
            layers: states,
        })
    }

    /// Restore a cross-request prefix into this slot from content-keyed pool
    /// entries. `entries[k]` is host page k's state; the last must carry the
    /// boundary sections. The caller has already mirrored the identity band.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn restore_prefix_state(
        &mut self,
        ctx: &DeviceContext,
        layers: &[Dsv4Layer],
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        index_head_dim: usize,
        entries: &[crate::attention::Dsv4PrefixPageEntry],
        matched_len: usize,
        finish_len: usize,
        page_tokens: usize,
    ) -> Result<()> {
        ensure!(
            layers.len() == self.attention.len(),
            "DSv4 prefix restore layer count {} != attention states {}",
            layers.len(),
            self.attention.len()
        );
        ensure!(
            page_tokens > 0 && matched_len == entries.len() * page_tokens,
            "DSv4 prefix restore matched_len {matched_len} != {} pages × {page_tokens}",
            entries.len()
        );
        // finish_len is the exact frontier: matched_len (aligned) plus the last
        // entry's sub-page tail. The tail is < page_tokens by construction.
        ensure!(
            (matched_len..matched_len + page_tokens).contains(&finish_len),
            "DSv4 prefix restore finish_len {finish_len} not in [matched_len {matched_len}, +{page_tokens})"
        );
        ensure!(
            finish_len <= self.max_seq_len,
            "DSv4 prefix restore finish_len {finish_len} exceeds slot max_seq_len {}",
            self.max_seq_len
        );
        ensure!(
            entries.last().is_some_and(|e| e.boundary),
            "DSv4 prefix restore: final matched page lacks boundary sections"
        );
        for (k, entry) in entries.iter().enumerate() {
            ensure!(
                entry.page_index as usize == k,
                "DSv4 prefix restore: entry captured at page {} restored at {k} \
                 (recycled host page id?)",
                entry.page_index
            );
            ensure!(
                entry.layers.len() == self.attention.len(),
                "DSv4 prefix restore: entry layer count {} != attention states {}",
                entry.layers.len(),
                self.attention.len()
            );
            let boundary = k + 1 == entries.len();
            for (idx, (state, layer_state)) in
                self.attention.iter_mut().zip(&entry.layers).enumerate()
            {
                let pool = kv_adapter.layer_mut(idx)?;
                state.restore_prefix_page(
                    ctx,
                    pool,
                    layers[idx].mode,
                    layers[idx].compress_ratio,
                    index_head_dim,
                    page_tokens,
                    k,
                    layer_state,
                    boundary,
                )?;
            }
        }
        // Frontier tail: the sub-page leftover the last matched page's own rows
        // don't cover. A no-op when finish_len == matched_len.
        if let Some(frontier) = entries.last() {
            for (idx, (state, layer_state)) in
                self.attention.iter_mut().zip(&frontier.layers).enumerate()
            {
                let pool = kv_adapter.layer_mut(idx)?;
                state.restore_frontier_tail(
                    ctx,
                    pool,
                    layers[idx].mode,
                    layers[idx].compress_ratio,
                    index_head_dim,
                    matched_len,
                    finish_len,
                    layer_state,
                )?;
            }
        }
        for (idx, state) in self.attention.iter_mut().enumerate() {
            state.restore_prefix_counters(layers[idx].mode, layers[idx].compress_ratio, finish_len);
        }
        self.seq_len = finish_len;
        ctx.sync()?;
        Ok(())
    }

    pub(crate) fn truncate(
        &mut self,
        layers: &[Dsv4Layer],
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        new_len: usize,
    ) -> Result<()> {
        ensure!(
            new_len <= self.seq_len,
            "DSv4 slot truncate cannot grow from {} to {new_len}",
            self.seq_len
        );
        ensure!(
            layers.len() == self.attention.len(),
            "DSv4 slot truncate layer count {} != attention states {}",
            layers.len(),
            self.attention.len()
        );
        self.seq_len = new_len;
        for (layer_idx, (layer, state)) in layers.iter().zip(&mut self.attention).enumerate() {
            state.truncate_decode_len(layer.mode, layer.compress_ratio, new_len);
            if let Some(slot_idx) = state.flashmla_slot_idx() {
                kv_adapter
                    .layer_mut(layer_idx)?
                    .flashmla_truncate_slot(slot_idx, new_len)?;
            }
        }
        Ok(())
    }

    pub(crate) fn swap_out_image(
        &mut self,
        ctx: &DeviceContext,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        slot_idx: usize,
    ) -> Result<Dsv4SlotImage> {
        let num_layers = self.attention.len();
        let mut layers = Vec::with_capacity(num_layers);
        let mut kv_pages = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let pool = kv_adapter.layer(layer_idx)?;
            let layer_image = self.attention[layer_idx].swap_out(ctx, pool)?;
            let pages = if let Some(flash) = &self.attention[layer_idx].flashmla {
                debug_assert_eq!(flash.slot_idx, slot_idx);
                let table = pool.flashmla_page_table(slot_idx)?;
                if table.is_empty() {
                    Vec::new()
                } else {
                    pool.flashmla_pool()?
                        .copy_pages_to_host_no_sync(ctx, table)?
                }
            } else {
                Vec::new()
            };
            layers.push(layer_image);
            kv_pages.push(pages);
        }
        ctx.sync()?;
        Ok(Dsv4SlotImage {
            seq_len: self.seq_len,
            layers,
            kv_pages,
        })
    }

    /// Call only after the tier has accepted the image on EVERY rank.
    pub(crate) fn release_swapped_out(
        &mut self,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
    ) -> Result<()> {
        for layer_idx in 0..self.attention.len() {
            if let Some(slot_idx_in_layer) = self.attention[layer_idx].flashmla_slot_idx() {
                kv_adapter
                    .layer_mut(layer_idx)?
                    .flashmla_free_slot(slot_idx_in_layer)?;
            }
        }
        self.seq_len = 0;
        Ok(())
    }

    pub(crate) fn swap_in_image(
        &mut self,
        ctx: &DeviceContext,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        slot_idx: usize,
        image: &Dsv4SlotImage,
    ) -> Result<()> {
        ensure!(
            image.layers.len() == self.attention.len(),
            "DSv4 swap-in layer count {} != attention states {}",
            image.layers.len(),
            self.attention.len()
        );
        kv_adapter.mirror_full_band(ctx, slot_idx, image.seq_len)?;
        for layer_idx in 0..self.attention.len() {
            if let Some(flash) = &self.attention[layer_idx].flashmla {
                debug_assert_eq!(flash.slot_idx, slot_idx);
                if !image.kv_pages[layer_idx].is_empty() {
                    let table: Vec<u32> = kv_adapter
                        .layer(layer_idx)?
                        .flashmla_page_table(slot_idx)?
                        .to_vec();
                    kv_adapter
                        .layer_mut(layer_idx)?
                        .flashmla_pool_mut()?
                        .copy_pages_from_host(ctx, &table, &image.kv_pages[layer_idx])?;
                }
            }
            let pool = kv_adapter.layer_mut(layer_idx)?;
            self.attention[layer_idx].swap_in(ctx, pool, &image.layers[layer_idx])?;
            self.attention[layer_idx]
                .refresh_flashmla_device_page_table(ctx, kv_adapter.layer(layer_idx)?)?;
        }
        self.seq_len = image.seq_len;
        ctx.sync()?;
        Ok(())
    }
}

impl Dsv4Model {
    pub(crate) fn truncate_slot(
        &self,
        slot: &mut Dsv4SlotState,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        new_len: usize,
    ) -> Result<()> {
        slot.truncate(&self.layers, kv_adapter, new_len)
    }

    pub(crate) fn capture_spec_rings(
        &self,
        slot: &mut Dsv4SlotState,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        start_pos: usize,
        depth: usize,
    ) -> Result<()> {
        slot.capture_spec_rings(&self.ctx, kv_adapter, start_pos, depth)
    }

    /// Restore the speculative boundary ring slot after truncate and before
    /// accepted-prefix fold. No-op when spec decode is off.
    pub(crate) fn restore_spec_ring_tail(
        &self,
        slot: &mut Dsv4SlotState,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        start_pos: usize,
        accepted_n: usize,
        depth: usize,
    ) -> Result<()> {
        slot.restore_spec_ring_tail(&self.ctx, kv_adapter, start_pos, accepted_n, depth)
    }

    /// Commit the accepted prefix (`accepted_rows` = verify row indices in
    /// schedule order, root first) from the persisted verify rows, then advance
    /// the slot length. Caller order: truncate → rejected-tail restore → THIS.
    pub(crate) fn commit_accepted_fold<I>(
        &self,
        slot: &mut Dsv4SlotState,
        kv_adapter: &mut crate::attention::Dsv4KvAdapter,
        accepted_rows: I,
        start_pos: usize,
    ) -> Result<()>
    where
        I: Clone + ExactSizeIterator<Item = usize>,
    {
        let m = accepted_rows.len();
        ensure!(m > 0, "DSv4 commit fold needs at least the pending row");
        let hidden_size = self.config.hidden_size;
        let mut keepalive = Dsv4ForwardKeepalive::new(false);
        // SAFETY: uninit device scratch; fully written before first read.
        let mut gathered = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, m)? };
        keepalive.keep_hidden(&gathered);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            {
                let cache = slot
                    .spec_normed
                    .as_ref()
                    .ok_or_else(|| anyhow!("DSv4 commit fold without persisted verify rows"))?;
                for (i, row) in accepted_rows.clone().enumerate() {
                    let src = cache[layer_idx]
                        .data
                        .slice(row * hidden_size..(row + 1) * hidden_size);
                    let mut dst = gathered
                        .data
                        .slice_mut(i * hidden_size..(i + 1) * hidden_size);
                    self.ctx
                        .stream
                        .memcpy_dtod(&src, &mut dst)
                        .map_err(|e| anyhow!("DSv4 commit fold gather failed: {e}"))?;
                }
            }
            let (layer_pool, flashmla_scratch, fp32_scratch) =
                kv_adapter.layer_and_flashmla_scratch_mut(layer_idx)?;
            crate::attention::commit_layer_fold(
                &self.ctx,
                &self.config,
                &layer.attention,
                layer.mode,
                layer.compress_ratio,
                &mut slot.attention[layer_idx],
                flashmla_scratch,
                fp32_scratch,
                layer_pool,
                &gathered,
                start_pos,
                &mut keepalive,
            )?;
        }
        std::hint::black_box(keepalive.len());
        drop(keepalive);
        // Advance the FlashMLA pool cursor by the committed token count so
        // `pool.seq_len(slot) == start_pos + m`: the next decode tick validates
        // it against `append_pos`. commit_layer_fold draws no pool tokens.
        for (layer_idx, state) in slot.attention.iter().enumerate() {
            if let Some(slot_idx) = state.flashmla_slot_idx() {
                kv_adapter
                    .layer_mut(layer_idx)?
                    .flashmla_alloc_append(slot_idx, m)?;
            }
        }
        slot.seq_len = start_pos + m;
        Ok(())
    }
}
