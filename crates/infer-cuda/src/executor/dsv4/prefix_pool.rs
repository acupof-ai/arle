use super::*;
use infer_kvspace::{
    MAX_PENDING_PREFIX_CAPTURES, PendingPrefixPage, PrefixPoolIndex, capture_epoch_matches,
    rekey_target_conflicts, reusable_prefix_blocks, reusable_prefix_blocks_for_prompt,
};

/// Real [`PrefixPoolIndex`] over the executor's host pool.
struct PrefixStateIndex<'a> {
    page_block_size: usize,
    state: &'a crate::attention::Dsv4PrefixStatePool,
}

impl PrefixPoolIndex for PrefixStateIndex<'_> {
    fn page_block_size(&self) -> usize {
        self.page_block_size
    }
    fn page_boundary(&self, page: u32) -> Option<bool> {
        self.state.page_meta(page).map(|meta| meta.boundary)
    }
    fn frontier_tail_tokens(&self, page: u32) -> Option<&[u32]> {
        self.state.frontier_tail_tokens(page)
    }
}

pub(super) struct PendingPrefixCapture {
    slot: usize,
    slot_epoch: u64,
    slot_pages: Vec<u32>,
    fence: CudaPipelineFence,
    pages: Vec<PendingPrefixPage>,
}

impl Dsv4CudaExecutor {
    /// Best-effort: a publish failure only forfeits a future reuse, never the
    /// forward.
    fn enqueue_prefix_capture(
        &mut self,
        slot: usize,
        slot_pages: &[u32],
        pages: Vec<PendingPrefixPage>,
    ) -> Result<()> {
        if pages.is_empty() {
            return Ok(());
        }
        let slot_epoch = self.kv_adapter.slot_epoch(slot).unwrap_or(u64::MAX);
        // A fence-record failure DROPS the capture: a fence-less entry at the
        // front would never poll Ready, pinning the queue full forever.
        let fence = self
            .model
            .ctx
            .record_pipeline_fence(CudaPipelineStreamKind::Compute)?;
        self.pending_prefix_captures
            .push_back(PendingPrefixCapture {
                slot,
                slot_epoch,
                slot_pages: slot_pages.to_vec(),
                fence,
                pages,
            });
        Ok(())
    }

    fn prefix_capture_queue_full(&self) -> bool {
        self.pending_prefix_captures.len() >= MAX_PENDING_PREFIX_CAPTURES
    }

    pub(crate) fn poll_prefix_captures(&mut self) {
        loop {
            let status = match self.pending_prefix_captures.front() {
                Some(capture) => capture.fence.query(),
                None => return,
            };
            match status {
                Ok(CudaPipelineFenceStatus::NotReady) => return,
                Err(err) => {
                    // Query errors are sticky: drop the capture so the queue
                    // drains instead of stalling behind a stuck front.
                    warn!("DSv4 prefix capture fence failed; dropping capture: {err:#}");
                    self.pending_prefix_captures.pop_front();
                    continue;
                }
                Ok(CudaPipelineFenceStatus::Ready) => {}
            }
            let capture = self
                .pending_prefix_captures
                .pop_front()
                .expect("front observed");
            let epoch_matches =
                capture_epoch_matches(capture.slot_epoch, self.kv_adapter.slot_epoch(capture.slot));
            let protected_pages = if epoch_matches {
                capture.slot_pages
            } else {
                capture
                    .pages
                    .iter()
                    .filter(|page| page.is_confirmed() && !page.is_cancelled())
                    .map(|page| page.target_page())
                    .collect()
            };
            for page in capture.pages {
                let target_exists = self.prefix_state.page_meta(page.target_page()).is_some();
                if page.is_cancelled()
                    || (!page.is_confirmed() && !epoch_matches)
                    || rekey_target_conflicts(page.source_page(), page.target_page(), target_exists)
                {
                    continue;
                }
                if !self
                    .prefix_state
                    .publish(page.target_page(), page.entry(), &protected_pages)
                {
                    continue;
                }
                if page.is_confirmed() {
                    self.prefix_state.confirm_pages(&[page.target_page()]);
                }
                if let Some(tokens) = page.frontier_tail() {
                    self.prefix_state
                        .set_frontier_tail(page.target_page(), tokens.to_vec());
                }
            }
        }
    }

    pub(super) fn publish_completed_prefix_pages(
        &mut self,
        slot: usize,
        slot_pages: &[u32],
        start_pos: usize,
        end_pos: usize,
    ) {
        if self.prefix_state.is_inactive() {
            return;
        }
        self.poll_prefix_captures();
        if self.prefix_capture_queue_full() {
            return;
        }
        let page_tokens = self.model.kv_arena.page_block_size;
        if page_tokens == 0 || end_pos <= start_pos {
            return;
        }
        let align = self.model.config.sliding_window.max(1);
        let mut captured = Vec::new();
        let mut capture_failed = false;
        for page_index in (start_pos / page_tokens)..(end_pos / page_tokens) {
            let page_end = (page_index + 1) * page_tokens;
            let Some(&page_id) = slot_pages.get(page_index) else {
                warn!(
                    "DSv4 prefix publish: slot {slot} page {page_index} outside host table ({} pages)",
                    slot_pages.len()
                );
                capture_failed = true;
                break;
            };
            // Boundary sections exist only when the forward ended exactly here,
            // and restore commits only at `align` multiples. A commit that
            // CROSSES the page end is unrecoverable — the SW ring advances every
            // token, so the boundary-instant ring is already gone.
            let boundary = page_end == end_pos && page_end.is_multiple_of(align);
            match self.slots[slot].capture_prefix_page(
                &self.model.ctx,
                &self.model.layers,
                &self.kv_adapter,
                self.model.config.index_head_dim,
                page_tokens,
                page_index,
                boundary,
            ) {
                Ok(entry) => captured.push(PendingPrefixPage::new(page_id, page_id, entry, None)),
                Err(err) => {
                    warn!("DSv4 prefix publish failed for slot {slot} page {page_index}: {err:#}");
                    capture_failed = true;
                    break;
                }
            }
        }
        if captured.is_empty() {
            return;
        }
        if capture_failed {
            captured.iter_mut().for_each(PendingPrefixPage::cancel);
        }
        if let Err(err) = self.enqueue_prefix_capture(slot, slot_pages, captured) {
            warn!("DSv4 prefix publish fence failed for slot {slot}: {err:#}");
        }
    }

    /// Publish pool entries for the finished slot's sealed region + frontier
    /// tail, so a later turn restores to the EXACT finish position instead of
    /// flooring at a page boundary. Entries land PROVISIONAL on the slot's own
    /// page ids; the finish's `save_prefix_sidecar` reconciles them to the
    /// radix's canonical ids. Best-effort: a failure only forfeits a reuse.
    pub(crate) fn capture_finish_frontier(
        &mut self,
        slot: usize,
        tokens: &[u32],
        slot_pages: &[u32],
    ) -> Result<()> {
        if self.prefix_state.is_inactive() {
            return Ok(());
        }
        self.poll_prefix_captures();
        if self.prefix_capture_queue_full() {
            return Ok(());
        }
        let page_tokens = self.model.kv_arena.page_block_size;
        if page_tokens == 0 {
            return Ok(());
        }
        let finish_len = tokens.len().min(self.slots[slot].seq_len());
        let sealed_pages = finish_len / page_tokens;
        if sealed_pages == 0 {
            return Ok(()); // < one page: no radix anchor to reuse
        }
        let matched_len = sealed_pages * page_tokens;
        let index_head_dim = self.model.config.index_head_dim;
        let frontier = sealed_pages - 1;
        // A page the prefill publish already stored is skipped; the frontier is
        // always recaptured to attach its tail + carry at finish_len.
        let mut captured = Vec::new();
        let mut capture_failed = false;
        for page_index in 0..sealed_pages {
            let Some(&page_id) = slot_pages.get(page_index) else {
                warn!(
                    "DSv4 finish frontier: slot {slot} page {page_index} outside host table ({} pages)",
                    slot_pages.len()
                );
                capture_failed = true;
                break;
            };
            let is_frontier = page_index == frontier;
            if !is_frontier && self.prefix_state.page_meta(page_id).is_some() {
                continue;
            }
            // A tail-less boundary entry licenses ANY continuation; a tail-gated
            // recapture would veto every diverging suffix (#166).
            if is_frontier
                && self
                    .prefix_state
                    .page_meta(page_id)
                    .is_some_and(|m| m.boundary)
                && self.prefix_state.frontier_tail_tokens(page_id).is_none()
            {
                continue;
            }
            let entry = if is_frontier {
                self.slots[slot].capture_frontier_page(
                    &self.model.ctx,
                    &self.model.layers,
                    &self.kv_adapter,
                    index_head_dim,
                    page_tokens,
                    page_index,
                    matched_len,
                    finish_len,
                )
            } else {
                self.slots[slot].capture_prefix_page(
                    &self.model.ctx,
                    &self.model.layers,
                    &self.kv_adapter,
                    index_head_dim,
                    page_tokens,
                    page_index,
                    false,
                )
            };
            match entry {
                Ok(entry) => captured.push(PendingPrefixPage::new(
                    page_id,
                    page_id,
                    entry,
                    (is_frontier && finish_len > matched_len)
                        .then(|| tokens[matched_len..finish_len].to_vec()),
                )),
                Err(err) => {
                    warn!(
                        "DSv4 finish frontier capture failed for slot {slot} page {page_index}: {err:#}"
                    );
                    capture_failed = true;
                    break;
                }
            }
        }
        if captured.is_empty() {
            return Ok(());
        }
        if capture_failed {
            captured.iter_mut().for_each(PendingPrefixPage::cancel);
        }
        self.enqueue_prefix_capture(slot, slot_pages, captured)?;
        Ok(())
    }

    /// Radix evicted these host pages: drop their pool entries — the pool's
    /// lifetime rides the radix, it has no independent one.
    pub(crate) fn release_prefix_pages(&mut self, pages: &[u32]) {
        for capture in &mut self.pending_prefix_captures {
            for page in &mut capture.pages {
                if pages.contains(&page.target_page()) {
                    page.cancel();
                }
            }
        }
        self.prefix_state.remove_pages(pages);
    }

    /// Slot free/abort returned these pages: drop provisional entries only.
    pub(crate) fn release_provisional_prefix_pages(&mut self, pages: &[u32]) {
        for capture in &mut self.pending_prefix_captures {
            for page in &mut capture.pages {
                page.cancel_provisional(pages);
            }
        }
        self.prefix_state.remove_provisional_pages(pages);
    }

    pub(crate) fn confirm_prefix_pages(&mut self, pages: &[u32]) {
        for capture in &mut self.pending_prefix_captures {
            for page in &mut capture.pages {
                page.confirm(pages);
            }
        }
        self.prefix_state.confirm_pages(pages);
    }

    /// `canonical` is the radix chain the sidecar rides; `slot_pages` is the
    /// finishing slot's own chain at the same positions. Where dedup diverged
    /// them and the canonical entry is missing, adopt the slot's provisional
    /// entry under the canonical id.
    pub(crate) fn repair_prefix_pool_chain(&mut self, canonical: &[u32], slot_pages: &[u32]) {
        for (&canon, &own) in canonical.iter().zip(slot_pages) {
            if canon != own {
                let canonical_exists = self.prefix_state.page_meta(canon).is_some();
                for capture in &mut self.pending_prefix_captures {
                    for page in &mut capture.pages {
                        page.repair(canon, own, canonical_exists);
                    }
                }
                self.prefix_state.adopt_canonical(canon, own);
            }
        }
        self.poll_prefix_captures();
    }

    /// Reuse license: a leading page is attachable only while every page up to
    /// and including it has a pool entry; committing additionally requires the
    /// page to carry the boundary sections. Pool presence covers host DRAM and
    /// disk alike — licensing never does capacity math.
    pub(crate) fn reusable_prefix_blocks(&self, blocks: &[PrefixBlock]) -> usize {
        let index = PrefixStateIndex {
            page_block_size: self.model.kv_arena.page_block_size,
            state: &self.prefix_state,
        };
        reusable_prefix_blocks(&index, blocks)
    }

    /// Like [`Self::reusable_prefix_blocks`], but a frontier page carrying a
    /// sub-page tail commits ONLY when `tokens` is a verified continuation
    /// through the finish position: the radix proves identity to the page
    /// boundary only, so a divergent prompt would otherwise restore a different
    /// request's KV (or over-restore into `seq_len > append_pos`).
    pub(crate) fn reusable_prefix_blocks_for_prompt(
        &self,
        blocks: &[PrefixBlock],
        tokens: &[u32],
    ) -> usize {
        let index = PrefixStateIndex {
            page_block_size: self.model.kv_arena.page_block_size,
            state: &self.prefix_state,
        };
        reusable_prefix_blocks_for_prompt(&index, blocks, tokens)
    }

    /// Boundary sections exist only at a forward's own end position, so without
    /// this a single-call prefill never visits an earlier boundary.
    pub(crate) fn prefill_restore_boundary_alignment(&self) -> usize {
        self.model.config.sliding_window.max(1)
    }

    /// Restore a radix-matched prefix into `slot`; call AFTER attaching the
    /// host pages. Every entry decodes to owned host state first, so a
    /// missing/undecodable page aborts before any device byte moves — never a
    /// partial restore. Returns the EXTRA tokens restored beyond `matched_len`
    /// (`0` = restored exactly the page-aligned match).
    pub(crate) fn restore_prefix_state(
        &mut self,
        slot: usize,
        tokens: &[u32],
        matched_len: usize,
        prefix_pages: &[u32],
    ) -> Result<usize> {
        self.poll_prefix_captures();
        ensure!(
            slot < self.num_slots,
            "DSv4 prefix restore slot {slot} outside executor slots {}",
            self.num_slots
        );
        let page_tokens = self.model.kv_arena.page_block_size;
        ensure!(
            page_tokens > 0 && matched_len > 0,
            "DSv4 prefix restore needs a non-empty match (matched_len {matched_len})"
        );
        // An unaligned match means the engine trim/clamp ordering regressed —
        // fail loud, never silently skip the ring restore.
        ensure!(
            matched_len.is_multiple_of(page_tokens),
            "DSv4 prefix restore matched_len {matched_len} not aligned to page {page_tokens}"
        );
        ensure!(
            prefix_pages.len() == matched_len / page_tokens,
            "DSv4 prefix restore {} pages != matched_len {matched_len} / page {page_tokens}",
            prefix_pages.len()
        );
        let entries = prefix_pages
            .iter()
            .map(|&page_id| self.prefix_state.read_entry(page_id))
            .collect::<Result<Vec<_>>>()?;
        // The frontier's sub-page tail has no radix key, so reuse it only when
        // the prompt actually contains those exact tokens; otherwise fall back
        // to the page-aligned match.
        let frontier = *prefix_pages.last().expect("matched_len > 0 ⇒ ≥1 page");
        let tail_len = match self.prefix_state.frontier_tail_tokens(frontier) {
            Some(tail)
                if tokens.len() >= matched_len + tail.len()
                    && tokens[matched_len..matched_len + tail.len()] == *tail =>
            {
                tail.len()
            }
            _ => 0,
        };
        let finish_len = matched_len + tail_len;
        self.kv_adapter
            .mirror_full_band(&self.model.ctx, slot, finish_len)?;
        // A restored occupant enters Decoding without a tail warm step, so any
        // spec state here belongs to the prior occupant; rebase the DSpark
        // latent cache to the restored frontier for the same reason.
        self.spec_slots[slot] = Dsv4SpecSlotState::default();
        self.reset_dspark_slot(slot, finish_len);
        // The restored occupant has a new page band and ring/compressor state;
        // the prior occupant's capture would replay over it unwarmed.
        self.decode_graphs[slot] = None;
        self.slots[slot].restore_prefix_state(
            &self.model.ctx,
            &self.model.layers,
            &mut self.kv_adapter,
            self.model.config.index_head_dim,
            &entries,
            matched_len,
            finish_len,
            page_tokens,
        )?;
        Ok(tail_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(page: u32, boundary: bool) -> infer_kvspace::Dsv4PrefixPageEntry {
        infer_kvspace::Dsv4PrefixPageEntry {
            page_index: page,
            boundary,
            layers: vec![],
        }
    }

    /// Contract test against the REAL adapter: the policy reads the host
    /// pool's meta and frontier tails exactly as the executor does. Runs on a
    /// CUDA host only — the executor module is cuda-gated, so the test binary
    /// links CUDA FFI; the fake-side contract tests in infer-kvspace cover the
    /// same policy branches on Mac.
    #[test]
    fn real_adapter_reads_pool_meta_and_frontier_tails() {
        let mut pool = crate::attention::Dsv4PrefixStatePool::new(1 << 20, 4096);
        assert!(pool.publish(10, &entry(10, true), &[]));
        assert!(pool.publish(11, &entry(11, true), &[]));
        pool.set_frontier_tail(11, vec![1, 2, 3]);
        let index = PrefixStateIndex {
            page_block_size: 16,
            state: &pool,
        };
        let blocks = [PrefixBlock::ResidentPage(10), PrefixBlock::ResidentPage(11)];
        assert_eq!(reusable_prefix_blocks(&index, &blocks), 2);
        // Matching continuation through the real pool's tail (page 11 ends
        // at token 32; the tail occupies [32, 35)).
        let mut tokens = vec![0u32; 35];
        tokens[32..35].copy_from_slice(&[1, 2, 3]);
        assert_eq!(
            reusable_prefix_blocks_for_prompt(&index, &blocks, &tokens),
            2
        );
        // Divergent prompt: the tail page must not commit.
        tokens[34] ^= 0xFFFF;
        assert_eq!(
            reusable_prefix_blocks_for_prompt(&index, &blocks, &tokens),
            1
        );
    }
}
