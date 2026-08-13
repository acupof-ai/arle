//! Radix prefix-cache choreography for [`Engine`].
//!
//! `impl Engine` methods orchestrating `self.radix` (the trie in `radix.rs`) with
//! the `KvPrefixStore`/`KvAllocator` ops on `self.kv`: attach a matched prefix,
//! publish sealed blocks on finish, release reused pages, and reclaim via LRU
//! eviction when allocation would fail.

use std::time::Instant;

use anyhow::{Result, anyhow};
use infer_seam::{BackendExecutor, KvPool, KvTierLocation, PrefixBlock};

use crate::{BlockId, Engine, PrefixMatch, RequestPhase, RequestState};

impl<E: BackendExecutor, K: KvPool> Engine<E, K> {
    pub(crate) fn request_pages_needed_after_prefix(
        &self,
        request: &RequestState,
        matched_tokens: usize,
    ) -> usize {
        let page_size = self.kv.page_size().max(1);
        let tokens = request
            .prompt_tokens
            .len()
            .saturating_sub(matched_tokens)
            .saturating_add(request.max_tokens);
        let token_pages = tokens.div_ceil(page_size);

        // Fixed-band pools (DSv4): each slot consumes exactly
        // `fixed_pages_per_slot` physical pages regardless of token count.
        // A prefix-matched slot still needs the full band (top-up from free).
        if let Some(fixed) = self.kv.fixed_pages_per_slot() {
            let matched_pages = matched_tokens / page_size;
            return fixed
                .saturating_sub(matched_pages)
                .max(token_pages.min(fixed));
        }

        token_pages
    }

    /// Clamp a radix prefix match to leading pages that are complete backend
    /// restore boundaries. The host radix can cache every page boundary, while
    /// a backend may only be able to restore KV plus side state at boundaries
    /// it explicitly snapshotted. Trim the match to the executor-reported
    /// reusable page count and re-prefill the tail.
    pub(crate) fn clamp_prefix_to_backend(
        &self,
        mut prefix_match: PrefixMatch,
        tokens: &[u32],
    ) -> PrefixMatch {
        let blocks: Vec<_> = prefix_match
            .block_ids
            .iter()
            .copied()
            .map(PrefixBlock::ResidentPage)
            .collect();
        // `_for_prompt`: a finish-write-through frontier is reusable only when
        // `tokens` continues through the cached tail (the tail has no radix key).
        let serveable = self
            .executor
            .reusable_prefix_blocks_for_prompt(&blocks, tokens)
            .min(prefix_match.block_ids.len());
        if serveable < prefix_match.block_ids.len() {
            prefix_match.block_ids.truncate(serveable);
            prefix_match.matched_len = serveable.saturating_mul(self.radix.block_size());
        }
        prefix_match
    }

    /// `tokens` is the request's committed stream (prompt + generated, #156) —
    /// callers already hold it for the radix lookup.
    pub(crate) fn attach_prefix_to_request(
        &mut self,
        slot: usize,
        request: &mut RequestState,
        tokens: &[u32],
        mut prefix_match: PrefixMatch,
    ) -> Result<()> {
        if self.config.enable_prefix_cache {
            self.prefix_cache_stats.lookups = self.prefix_cache_stats.lookups.saturating_add(1);
        }
        let target = tokens.len();
        // The last committed token must PREFILL, never restore: entering
        // `Decoding` with it already materialized makes the decode seed re-feed
        // it, duplicating its KV (see docs/experience/errors/
        // 2026-07-06-dsv4-concurrent-decode-digit-corruption-unresolved.md).
        // Both the full-match trim and the sidecar-restore clamp bound to this.
        let attach_cap = target.saturating_sub(1);

        // Trim the last matched block so the tail re-prefills through the
        // standard chunked path, which samples the first token from real
        // logits. Runs BEFORE `clamp_prefix_to_backend` (#154 D4): the clamp's
        // alignment floor must see the post-trim length, or the pop un-aligns
        // `matched_len` and the backend's restore predicate rejects it.
        if !prefix_match.is_empty() && prefix_match.matched_len > attach_cap {
            prefix_match.block_ids.pop();
            prefix_match.matched_len = prefix_match.block_ids.len() * self.radix.block_size();
        }
        let prefix_match = self.clamp_prefix_to_backend(prefix_match, tokens);
        if prefix_match.is_empty() {
            self.record_prefix_restore_metrics(0);
            request.prefill_start_pos = 0;
            request.phase = RequestPhase::Prefilling { progress: 0 };
            return Ok(());
        }

        // Pin matched pages first (page_refs++ / radix ref_count++) so the
        // pre-eviction below can't reclaim pages we're about to attach.
        self.kv.retain_pages(&prefix_match.block_ids);
        self.radix.retain_blocks(&prefix_match.block_ids);

        // Fixed-band pools (DSv4): attach_pages tops up to `fixed_pages_per_slot`
        // from the free pool. Pre-evict prefix cache if free is short, so the
        // attach doesn't bail mid-flight (the admission guard's token-based
        // estimate undercounts band consumption — see request_pages_needed_after_prefix).
        if let Some(fixed) = self.kv.fixed_pages_per_slot() {
            let top_up = fixed.saturating_sub(prefix_match.block_ids.len());
            let free = self.kv.free_pages();
            if top_up > free {
                self.evict_prefix_cache_for_pages(top_up.saturating_sub(free));
            }
        }

        // An admission-time alloc failure must never propagate — the caller
        // sits on the step path and the error is fatal to the whole TP group
        // (#164). Fixed-band top-up exhaustion degrades to full recompute;
        // the prefill then allocates through the shed-able plan path.
        if let Err(err) =
            self.kv
                .attach_pages(slot, &prefix_match.block_ids, prefix_match.matched_len)
        {
            log::warn!("prefix attach failed for slot {slot}: {err:#}; full recompute fallback");
            self.kv_system_metrics.fallback_recompute =
                self.kv_system_metrics.fallback_recompute.saturating_add(1);
            self.record_prefix_restore_metrics(0);
            self.radix.release_blocks(&prefix_match.block_ids);
            self.kv.release_pages(&prefix_match.block_ids);
            request.phase = RequestPhase::Prefilling { progress: 0 };
            return Ok(());
        }

        // Restore the recurrent sidecar for hybrid models (Qwen3.5/3.6). No-op for
        // full-attention-only backends. On miss, release the attached pages and fall
        // back to full recompute — a zeroed linear-attn state with non-zero full-attn
        // KV causes a cross-type mismatch that corrupts model output.
        let restored = match self.executor.restore_prefix_sidecar(
            slot,
            tokens,
            prefix_match.matched_len,
            &prefix_match.block_ids,
        ) {
            Ok(restored) => restored,
            Err(err) => {
                log::warn!(
                    "recurrent sidecar restore failed for slot {slot}: {err:#}; \
                     full recompute fallback"
                );
                self.kv_system_metrics.fallback_recompute =
                    self.kv_system_metrics.fallback_recompute.saturating_add(1);
                self.record_prefix_restore_metrics(0);
                // Undo retain_pages + attach_pages; executor already reset full_attn_kv.
                self.free_slot_pages(slot);
                self.radix.release_blocks(&prefix_match.block_ids);
                self.kv.release_pages(&prefix_match.block_ids);
                // request.prefill_start_pos stays at 0 (the pre-attach default).
                request.phase = RequestPhase::Prefilling { progress: 0 };
                return Ok(());
            }
        };

        // `restored` is the ABSOLUTE length the backend materialized. Three cases:
        // - `== matched_len`: today's page-aligned restore (all backends' common path).
        // - `> matched_len`: DSv4 finish-write-through restored PAST the 64-block
        //   radix `matched_len` into the finished turn's sub-page tail (< page).
        //   Grow the slot into its OWN partial page (never radix-published, so no
        //   retain/release) so only the leftover suffix prefills.
        // - `< matched_len`: Qwen3.5/3.6 hybrid restored a PERIODIC snapshot
        //   boundary `B` because no sidecar existed exactly at `matched_len` (a
        //   cross-conversation prefix hit). The device pool is at `B`; truncate the
        //   over-attached host pages [B..matched_len] back to `B` (radix-retained
        //   pages survive the reclaim; they stay in `reused_prefix_pages` and are
        //   released on finish), so the tail prefill covers [B..prompt].
        let restored_len = restored.min(attach_cap);
        if restored_len > prefix_match.matched_len {
            // Same degrade contract as the sidecar-miss arm above: an
            // admission alloc failure must not propagate into the fatal step
            // path (#164) — undo the attach and full-recompute instead.
            if let Err(err) = self.kv.alloc(slot, restored_len - prefix_match.matched_len) {
                log::warn!(
                    "prefix-restore grow alloc failed for slot {slot}: {err:#}; \
                     full recompute fallback"
                );
                self.kv_system_metrics.fallback_recompute =
                    self.kv_system_metrics.fallback_recompute.saturating_add(1);
                self.record_prefix_restore_metrics(0);
                self.free_slot_pages(slot);
                self.radix.release_blocks(&prefix_match.block_ids);
                self.kv.release_pages(&prefix_match.block_ids);
                request.phase = RequestPhase::Prefilling { progress: 0 };
                return Ok(());
            }
        } else if restored_len < prefix_match.matched_len {
            self.kv.truncate_slot(slot, restored_len)?;
        }

        log::info!(
            "prefix-attach: slot={slot} matched={} restored={restored_len} committed={target}",
            prefix_match.matched_len,
        );
        self.record_prefix_restore_metrics(restored_len);

        request.prefill_start_pos = restored_len;
        request.reused_prefix_pages = prefix_match.block_ids;
        request.phase = if request.prefill_start_pos == target {
            RequestPhase::Decoding
        } else {
            RequestPhase::Prefilling {
                progress: request.prefill_start_pos,
            }
        };
        Ok(())
    }

    // record_prefix_tier_hits moved into materialize_prefix_blocks.

    /// Grow `slot` to at least `target`. A speculative executor already grew it
    /// to draft its chain, so this no-ops instead of double-appending.
    pub(crate) fn alloc_to_len_with_prefix_reclaim(
        &mut self,
        slot: usize,
        target: usize,
    ) -> Result<()> {
        match target.checked_sub(self.kv.seq_len(slot)) {
            Some(0) | None => Ok(()),
            Some(missing) => self.alloc_with_prefix_reclaim(slot, missing),
        }
    }

    pub(crate) fn alloc_with_prefix_reclaim(&mut self, slot: usize, tokens: usize) -> Result<()> {
        let needed = self.kv.append_pages_needed(slot, tokens);
        if needed > self.kv.free_pages() {
            self.evict_prefix_cache_for_pages(needed - self.kv.free_pages());
        }

        match self.kv.alloc(slot, tokens) {
            Ok(()) => Ok(()),
            Err(first_err) => {
                let needed = self.kv.append_pages_needed(slot, tokens);
                let freed = self.evict_prefix_cache_for_pages(needed);
                if freed == 0 {
                    return Err(first_err);
                }
                self.kv.alloc(slot, tokens).map_err(|retry_err| {
                    anyhow!(
                        "KV alloc retry failed after freeing {freed} pages: first error: {first_err}; retry error: {retry_err}"
                    )
                })
            }
        }
    }

    /// Seal the leading `tokens` blocks into the radix. Callers choose the
    /// boundary: prompt-only at prefill time (planner), prompt+generated at
    /// finish — the same full-sequence boundary the recurrent sidecar
    /// captures, so an agentic follow-up turn radix-matches THROUGH the
    /// previous turn's generated tokens instead of re-prefilling them.
    /// Returns the newly cached pages (already cache-retained), in order.
    pub(crate) fn publish_prefix_blocks(&mut self, slot: usize, tokens: &[u32]) -> Vec<BlockId> {
        if !self.kv.is_active() || !self.config.enable_prefix_cache {
            return Vec::new();
        }

        let block_size = self.radix.block_size().max(1);
        let publishable_tokens = tokens.len().min(self.kv.seq_len(slot));
        let sealed_blocks = publishable_tokens / block_size;
        if sealed_blocks == 0 {
            return Vec::new();
        }

        let sealed_tokens = sealed_blocks * block_size;
        let pages = self
            .kv
            .page_indices_for_token_range(slot, 0, sealed_tokens)
            .to_vec();
        let mut publish_blocks = sealed_blocks.min(pages.len());
        // A recall-evicted prompt page leaves an EVICTED_PAGE sentinel; a cached
        // prefix must be contiguous-resident, so truncate at the first hole — never
        // publish a sentinel as a ResidentPage (it would later mirror u32::MAX →
        // -1 in kv_indices and corrupt attention).
        if let Some(hole) = pages
            .iter()
            .take(publish_blocks)
            .position(|&p| p == infer_seam::EVICTED_PAGE)
        {
            publish_blocks = hole;
        }
        if publish_blocks == 0 {
            return Vec::new();
        }
        let blocks: Vec<_> = pages
            .iter()
            .take(publish_blocks)
            .copied()
            .map(PrefixBlock::ResidentPage)
            .collect();
        publish_blocks = self
            .executor
            .reusable_prefix_blocks(&blocks)
            .min(publish_blocks);
        if publish_blocks == 0 {
            return Vec::new();
        }

        let token_len = publish_blocks * block_size;
        let newly_cached = self
            .radix
            .insert(&tokens[..token_len], &pages[..publish_blocks]);
        if !newly_cached.is_empty() {
            self.prefix_cache_stats.published_pages = self
                .prefix_cache_stats
                .published_pages
                .saturating_add(newly_cached.len() as u64);
            self.kv.retain_pages(&newly_cached);
        }
        // Only `newly_cached` pages are radix-retained; a deduped page's id
        // will be freed and recycled, so backends must not key durable state
        // (e.g. DSv4's pool-entry confirm) to the full page list.
        // Capture the recurrent sidecar at the SAME boundary and pages just
        // sealed into the radix, so a hybrid backend (Qwen3.5/3.6) restores the
        // recurrent state on a future prefix hit instead of full-recomputing, and
        // the sidecar's lifetime rides these radix blocks (dropped via
        // `release_prefix_pages` when they evict). No-op for full-attention-only
        // backends. Runs on EVERY publish — including a restore-derived finish —
        // so a repeat turn's sidecar is always re-published. Best-effort: a save
        // failure only forfeits the next reuse, never blocks the publish.
        //
        // #155: a deduped block keeps its ORIGINAL radix page, so on any dedupe
        // (restore-miss → full-recompute resend) the slot's pages are not the
        // chain the sidecar rides — key it to the radix's own pages, else the
        // eviction key is a finish-freed slot page: the blob leaks, then drops
        // on page-id reuse while the cached prefix still expects it.
        let radix_pages = (newly_cached.len() != publish_blocks).then(|| {
            self.radix
                .peek_longest_prefix_match(&tokens[..token_len])
                .block_ids
        });
        let sidecar_pages = radix_pages.as_deref().unwrap_or(&pages[..publish_blocks]);
        if let Err(err) = self.executor.save_prefix_sidecar(
            slot,
            tokens,
            token_len,
            sidecar_pages,
            &pages[..publish_blocks],
            &newly_cached,
        ) {
            log::debug!("recurrent sidecar save failed for slot {slot}: {err:#}");
        }
        // Publishing over a demoted node revives it with the re-prefilled
        // page; the superseded tier entries surface on the drain.
        self.drain_dropped_tier_keys();
        newly_cached
    }

    /// Swap-style preemption support: demote exactly these just-published
    /// victim pages into the host tier (deepest block first so each parent
    /// becomes an evictable leaf in turn), severing on store refusal. Either
    /// way every page ends up free, so retraction releases the same device
    /// pages as a plain `free_slot` — only the contents' fate differs.
    pub(crate) fn demote_published_pages(&mut self, pages: &[BlockId]) {
        let pages: Vec<_> = pages.iter().rev().copied().collect();
        let mut offset = 0usize;
        while offset < pages.len() {
            let demoted = self.try_demote_pages(&pages[offset..]);
            for &page in &pages[offset..offset + demoted] {
                self.kv.release_pages(&[page]);
                self.executor.release_prefix_pages(&[page]);
            }
            offset += demoted;
            if offset >= pages.len() {
                break;
            }
            if demoted > 0 {
                continue;
            }

            let page = pages[offset];
            if !self.radix.evict_page(page) {
                // Defensive: a just-published page must be an idle leaf by
                // construction; if not, leave it cache-resident rather than
                // corrupt accounting.
                offset += 1;
                continue;
            }
            self.kv.release_pages(&[page]);
            self.executor.release_prefix_pages(&[page]);
            offset += 1;
        }
        self.drain_dropped_tier_keys();
    }

    pub(crate) fn release_reused_prefix(&mut self, pages: &[BlockId]) {
        if pages.is_empty() {
            return;
        }
        self.radix.release_blocks(pages);
        self.kv.release_pages(pages);
    }

    /// Evict LRU prefix-cache pages until `pages_needed` pages have actually
    /// been RETURNED to the free pool, or no further progress is possible.
    /// Victims are filtered by `KvQuery::page_is_evictable` — the exact
    /// predicate `resident_evictable_pages` counts, one source — so a cached
    /// page still attached to a live slot (or a ref-drifted orphan) is
    /// skipped and the loop continues past it into the deeper LRU (#164
    /// residual: the old count-of-severed-nodes "reclaimed 2 pages" while
    /// freeing 0, then gave up with freeable pages still resident).
    /// Returns the number of pages actually freed.
    pub(crate) fn evict_prefix_cache_for_pages(&mut self, pages_needed: usize) -> usize {
        if pages_needed == 0 {
            return 0;
        }
        let free_before = self.kv.free_pages();
        let tiered = self.kv_tier_capacity() > 0;
        loop {
            let freed = self.kv.free_pages().saturating_sub(free_before);
            if freed >= pages_needed {
                break;
            }
            // Snapshot the whole evictable frontier in LRU order (severing a
            // leaf exposes its parent, so re-query each round) and keep only
            // victims whose release genuinely frees a page.
            let victims: Vec<BlockId> = self
                .radix
                .lru_evictable_pages(usize::MAX)
                .into_iter()
                .filter(|&page| self.kv.page_is_evictable(page))
                .take(pages_needed - freed)
                .collect();
            if victims.is_empty() {
                break;
            }
            // Tier path: demote into the backend host store (contents stay
            // promotable); sever only when the store refuses everything.
            let demoted = if tiered {
                self.try_demote_pages(&victims)
            } else {
                0
            };
            for &page in &victims[..demoted] {
                self.kv.release_pages(&[page]);
                self.executor.release_prefix_pages(&[page]);
            }
            if demoted > 0 {
                continue;
            }
            let mut severed = 0usize;
            for &page in &victims {
                if !self.radix.evict_page(page) {
                    break;
                }
                self.kv.release_pages(&[page]);
                self.executor.release_prefix_pages(&[page]);
                severed += 1;
            }
            if severed == 0 {
                break;
            }
        }
        self.drain_dropped_tier_keys();
        self.kv.free_pages().saturating_sub(free_before)
    }

    /// Invalidate the whole prefix cache after a resident-weight change (OPD
    /// live LoRA re-merge / SOPD inline adapter update).
    ///
    /// Once `q_proj`/`v_proj` change, every cached block's `V = v_proj(x)` was
    /// computed under the prior adapter epoch and is stale, so no cached block
    /// — resident *or* host-tier-demoted — may serve a post-update request.
    /// Unlike [`Self::evict_prefix_cache_for_pages`], which *demotes* victims to
    /// the host tier (keeping them promotable), this **drops** everything:
    /// resident pages return to the KV pool, demoted blocks are severed and
    /// their tier keys forwarded to the backend tier store. Demoting would only
    /// move stale-epoch KV to host, where a later prefix match could promote it
    /// back — exactly the contamination we are removing.
    ///
    /// Precondition (caller-proven): no in-flight request pins a prefix page
    /// (`ref_count > 0`). The OPD inline-update loop calls this between rollouts
    /// on a quiesced engine, so every cached page is idle and is dropped. A page
    /// pinned by a concurrent in-flight request is **skipped** here (never freed
    /// under a live reader) and would keep serving stale-epoch KV until that
    /// request finishes — concurrent serving + live update needs per-request
    /// epoch tagging, out of scope for the Phase-0 keystone.
    pub fn invalidate_prefix_cache(&mut self) {
        // 1. Drop every idle resident cached page. `evict_lru` re-scans the
        //    evictable frontier each step, so one call bounded by the resident
        //    count drains the whole idle trie (severing a leaf exposes its
        //    parent as the next evictable leaf). Pinned pages stay in place.
        let resident = self.radix.cached_page_count();
        if resident > 0 {
            let pages = self.radix.evict_lru(resident);
            if !pages.is_empty() {
                self.kv.release_pages(&pages);
                self.executor.release_prefix_pages(&pages);
            }
        }
        // 2. Drop every idle host-tier demoted block — also stale-epoch KV that
        //    must not be promotable back into a post-update rollout. After step
        //    1 no demoted node has a resident descendant, so each is severable;
        //    the `false` guard is defensive against a pinned-subtree corner.
        while let Some(key) = self.radix.lru_demoted_key() {
            if !self.radix.drop_demoted(key) {
                break;
            }
        }
        // 3. Forward every severed tier key to the backend tier store.
        self.drain_dropped_tier_keys();
    }

    /// Host-tier capacity in pages; `0` disables every tier path. Tier use is
    /// gated on the prefix cache because demoted blocks are only reachable
    /// through radix prefix matches.
    pub(crate) fn kv_tier_capacity(&self) -> usize {
        if self.config.enable_prefix_cache {
            self.executor.kv_tier_capacity_pages()
        } else {
            0
        }
    }

    /// Copy pages into the backend host tier and mark accepted radix nodes
    /// demoted. Makes room by severing cold demoted blocks before the batch.
    fn try_demote_pages(&mut self, pages: &[BlockId]) -> usize {
        let capacity = self.executor.kv_tier_capacity_pages();
        if capacity == 0 || pages.is_empty() {
            return 0;
        }

        let mut entries = Vec::with_capacity(pages.len());
        for &page in pages {
            while self
                .radix
                .demoted_block_count()
                .saturating_add(entries.len())
                >= capacity
            {
                let Some(coldest) = self.radix.lru_demoted_key() else {
                    break;
                };
                self.radix.drop_demoted(coldest);
                // Drain immediately so the store slot is reusable for this
                // mset batch, not only after the eviction batch.
                self.drain_dropped_tier_keys();
            }
            if self
                .radix
                .demoted_block_count()
                .saturating_add(entries.len())
                >= capacity
            {
                break;
            }
            let key = self.next_tier_key;
            self.next_tier_key = self.next_tier_key.wrapping_add(1);
            entries.push((page, key));
        }
        if entries.is_empty() {
            return 0;
        }

        let started = Instant::now();
        let result = self.executor.demote_prefix_pages(&entries);
        let elapsed_ms = elapsed_ms(started);
        let charge_copy = !self.executor.kv_tier_transfer_is_zero_copy();
        self.kv_system_metrics.demote_mset_count =
            self.kv_system_metrics.demote_mset_count.saturating_add(1);
        if charge_copy {
            self.kv_system_metrics.demote_mset_copy_ms = self
                .kv_system_metrics
                .demote_mset_copy_ms
                .saturating_add(elapsed_ms);
        }
        let accepted = match result {
            Ok(accepted) => accepted.min(entries.len()),
            Err(err) => {
                log::warn!("KV tier demote failed for {} pages: {err:#}", entries.len());
                return 0;
            }
        };
        if charge_copy {
            let bytes = (accepted as u64).saturating_mul(self.executor.kv_tier_page_bytes() as u64);
            self.kv_system_metrics.demote_mset_copy_bytes = self
                .kv_system_metrics
                .demote_mset_copy_bytes
                .saturating_add(bytes);
        }

        let mut demoted = 0usize;
        for (idx, &(page, key)) in entries.iter().take(accepted).enumerate() {
            if self.radix.demote_block(page, key) {
                self.kv_tier_stats.demoted_pages =
                    self.kv_tier_stats.demoted_pages.saturating_add(1);
                demoted += 1;
            } else {
                // The radix refused (page is not an idle cached leaf);
                // accepted store copies from this point are unreachable.
                let stale: Vec<_> = entries[idx..accepted]
                    .iter()
                    .map(|&(_, stale_key)| stale_key)
                    .collect();
                self.executor.drop_kv_tier_entries(&stale);
                break;
            }
        }
        demoted
    }

    /// Prefix lookup used at slot attach. With a host tier, demoted blocks in
    /// the matched prefix are promoted back into freshly allocated pages so
    /// the existing resident-only attach path applies unchanged; a promote
    /// failure truncates the match there and the tail re-prefills.
    pub(crate) fn lookup_prefix_for_attach(&mut self, tokens: &[u32]) -> PrefixMatch {
        if self.kv_tier_capacity() == 0 {
            let matched = self.radix.longest_prefix_match(tokens);
            let raw = matched.block_ids.len();
            let clamped = self.clamp_prefix_to_backend(matched, tokens);
            self.record_prefix_match_metrics(raw, clamped.block_ids.len());
            log::info!(
                "prefix-lookup: prompt={} raw_blocks={raw} licensed_blocks={}",
                tokens.len(),
                clamped.block_ids.len()
            );
            return clamped;
        }
        let blocks_all = self.radix.tiered_longest_prefix_match(tokens).blocks;
        let resident = blocks_all
            .iter()
            .take_while(|b| matches!(b, PrefixBlock::ResidentPage(_)))
            .count();
        let mut blocks = blocks_all;
        let reusable = self
            .executor
            .reusable_prefix_blocks_for_prompt(&blocks, tokens)
            .min(blocks.len());
        self.record_prefix_match_metrics(blocks.len(), reusable);
        log::info!(
            "prefix-lookup(tier): prompt={} raw_blocks={} resident_run={resident} licensed_blocks={reusable}",
            tokens.len(),
            blocks.len()
        );
        blocks.truncate(reusable);
        let block_ids = self.materialize_prefix_blocks(&blocks);
        // record_prefix_tier_hits now lives inside materialize_prefix_blocks
        // so block locations are captured before promotion removes entries.
        self.drain_dropped_tier_keys();
        PrefixMatch {
            matched_len: block_ids.len() * self.radix.block_size(),
            block_ids,
        }
    }

    /// Materialize a backend-approved leading prefix into resident page ids.
    ///
    /// Resident blocks pass through. Demoted blocks are restored by one mget
    /// batch before any attach; if the batch fails, only the leading resident
    /// run before the first demoted block is returned.
    fn materialize_prefix_blocks(&mut self, blocks: &[PrefixBlock]) -> Vec<BlockId> {
        let demoted = blocks
            .iter()
            .filter_map(|block| match *block {
                PrefixBlock::ResidentPage(_) => None,
                PrefixBlock::DemotedKey(key) => Some(key),
            })
            .collect::<Vec<_>>();
        if demoted.is_empty() {
            let ids: Vec<_> = blocks
                .iter()
                .filter_map(|block| match *block {
                    PrefixBlock::ResidentPage(page) => Some(page),
                    PrefixBlock::DemotedKey(_) => None,
                })
                .collect();
            // All-resident: count every block as a resident hit.
            for _ in &ids {
                self.kv_system_metrics.reuse_hit_resident =
                    self.kv_system_metrics.reuse_hit_resident.saturating_add(1);
            }
            return ids;
        }

        // Snapshot per-block location BEFORE promotion (the promote path
        // removes entries from the tier store, so a post-promote query
        // would lose the disk/host attribution).
        for block in blocks {
            match *block {
                PrefixBlock::ResidentPage(_) => {
                    self.kv_system_metrics.reuse_hit_resident =
                        self.kv_system_metrics.reuse_hit_resident.saturating_add(1);
                }
                PrefixBlock::DemotedKey(key) => match self.executor.kv_tier_location(key) {
                    Some(KvTierLocation::HostDemoted) | None => {
                        self.kv_system_metrics.reuse_hit_host_demoted = self
                            .kv_system_metrics
                            .reuse_hit_host_demoted
                            .saturating_add(1);
                    }
                    Some(KvTierLocation::Disk) => {
                        self.kv_system_metrics.reuse_hit_disk =
                            self.kv_system_metrics.reuse_hit_disk.saturating_add(1);
                    }
                },
            }
        }

        let promoted_pages = match self.kv.alloc_detached_pages(demoted.len()) {
            Ok(pages) => pages,
            Err(err) => {
                log::warn!(
                    "KV tier promote skipped: could not allocate {} pages: {err:#}",
                    demoted.len()
                );
                self.kv_system_metrics.fallback_recompute =
                    self.kv_system_metrics.fallback_recompute.saturating_add(1);
                return leading_resident_pages(blocks);
            }
        };

        let promote_entries: Vec<_> = blocks
            .iter()
            .filter_map(|block| match *block {
                PrefixBlock::DemotedKey(key) => Some(key),
                _ => None,
            })
            .zip(promoted_pages.iter().copied())
            .collect();

        let started = Instant::now();
        let result = self.executor.promote_prefix_pages(&promote_entries);
        let elapsed_ms = elapsed_ms(started);
        let charge_copy = !self.executor.kv_tier_transfer_is_zero_copy();
        self.kv_system_metrics.promote_mget_count =
            self.kv_system_metrics.promote_mget_count.saturating_add(1);
        if charge_copy {
            self.kv_system_metrics.promote_mget_copy_ms = self
                .kv_system_metrics
                .promote_mget_copy_ms
                .saturating_add(elapsed_ms);
            self.kv_system_metrics.fetch_wait_ms = self
                .kv_system_metrics
                .fetch_wait_ms
                .saturating_add(elapsed_ms);
        }
        if let Err(err) = result {
            log::warn!(
                "KV tier promote failed for {} pages: {err:#}; recomputing tail",
                promote_entries.len()
            );
            self.executor.release_prefix_pages(&promoted_pages);
            self.kv.free_detached_pages(&promoted_pages);
            for &(key, _) in &promote_entries {
                self.radix.drop_demoted(key);
            }
            self.kv_tier_stats.promote_failures = self
                .kv_tier_stats
                .promote_failures
                .saturating_add(promote_entries.len() as u64);
            self.kv_system_metrics.fallback_recompute =
                self.kv_system_metrics.fallback_recompute.saturating_add(1);
            return leading_resident_pages(blocks);
        }
        if charge_copy {
            let bytes = (promote_entries.len() as u64)
                .saturating_mul(self.executor.kv_tier_page_bytes() as u64);
            self.kv_system_metrics.promote_mget_copy_bytes = self
                .kv_system_metrics
                .promote_mget_copy_bytes
                .saturating_add(bytes);
        }

        let mut block_ids = Vec::with_capacity(blocks.len());
        let mut promoted_idx = 0usize;
        for block in blocks {
            match *block {
                PrefixBlock::ResidentPage(page) => block_ids.push(page),
                PrefixBlock::DemotedKey(key) => {
                    let page = promote_entries[promoted_idx].1;
                    promoted_idx += 1;
                    if self.radix.promote_block(key, page) {
                        self.kv.retain_pages(&[page]);
                        self.kv_tier_stats.promoted_pages =
                            self.kv_tier_stats.promoted_pages.saturating_add(1);
                        block_ids.push(page);
                    } else {
                        self.kv_system_metrics.fallback_recompute =
                            self.kv_system_metrics.fallback_recompute.saturating_add(1);
                        self.executor.release_prefix_pages(&[page]);
                        self.kv.free_detached_pages(&[page]);
                        self.executor.drop_kv_tier_entries(&[key]);
                        let tail_pages: Vec<_> = promote_entries[promoted_idx..]
                            .iter()
                            .map(|&(_, tail_page)| tail_page)
                            .collect();
                        self.executor.release_prefix_pages(&tail_pages);
                        self.kv.free_detached_pages(&tail_pages);
                        break;
                    }
                }
            }
        }
        block_ids
    }

    /// Forward tier keys invalidated by radix mutations (sever, revive,
    /// promote) to the backend store. Called after every mutation batch so no
    /// path can leak a store entry.
    pub(crate) fn drain_dropped_tier_keys(&mut self) {
        let keys = self.radix.take_dropped_tier_keys();
        if !keys.is_empty() {
            self.executor.drop_kv_tier_entries(&keys);
        }
    }
}

fn leading_resident_pages(blocks: &[PrefixBlock]) -> Vec<BlockId> {
    blocks
        .iter()
        .take_while(|block| matches!(block, PrefixBlock::ResidentPage(_)))
        .filter_map(|block| match *block {
            PrefixBlock::ResidentPage(page) => Some(page),
            PrefixBlock::DemotedKey(_) => None,
        })
        .collect()
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64
}
