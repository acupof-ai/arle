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

// `RequestPhase` is used by both `attach_prefix_to_request` and
// `attach_cached_prefix`.

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
        tokens.div_ceil(page_size)
    }

    /// Clamp a radix prefix match to leading pages that are complete backend
    /// restore boundaries. The host radix can cache every page boundary, while
    /// a backend may only be able to restore KV plus side state at boundaries
    /// it explicitly snapshotted. Trim the match to the executor-reported
    /// reusable page count and re-prefill the tail.
    pub(crate) fn clamp_prefix_to_backend(&self, mut prefix_match: PrefixMatch) -> PrefixMatch {
        let blocks: Vec<_> = prefix_match
            .block_ids
            .iter()
            .copied()
            .map(PrefixBlock::ResidentPage)
            .collect();
        let serveable = self
            .executor
            .reusable_prefix_blocks(&blocks)
            .min(prefix_match.block_ids.len());
        if serveable < prefix_match.block_ids.len() {
            prefix_match.block_ids.truncate(serveable);
            prefix_match.matched_len = serveable.saturating_mul(self.radix.block_size());
        }
        prefix_match
    }

    pub(crate) fn attach_prefix_to_request(
        &mut self,
        slot: usize,
        request: &mut RequestState,
        prefix_match: PrefixMatch,
    ) -> Result<()> {
        if self.config.enable_prefix_cache {
            self.prefix_cache_stats.lookups = self.prefix_cache_stats.lookups.saturating_add(1);
        }

        let mut prefix_match = self.clamp_prefix_to_backend(prefix_match);
        // A full-prompt match must still run one genuine forward+sample step —
        // jumping straight to `Decoding` leaves `generated_tokens` empty, and the
        // planner's decode-seed `.or_else` fallback then silently re-feeds the
        // prompt's own last token as the seed, duplicating it into KV (see
        // docs/experience/errors/2026-07-06-dsv4-concurrent-decode-digit-corruption-unresolved.md,
        // "Layer-0-15 residual bisection" section). Trim the last matched block
        // so the tail always re-prefills through the standard chunked-prefill
        // path, which samples the first token from real logits.
        if !prefix_match.is_empty() && prefix_match.matched_len == request.prompt_len() {
            prefix_match.block_ids.pop();
            prefix_match.matched_len = prefix_match.block_ids.len() * self.radix.block_size();
        }
        if prefix_match.is_empty() {
            request.prefill_start_pos = 0;
            request.phase = RequestPhase::Prefilling { progress: 0 };
            request.waiting_hint.immediate_reuse_tokens = 0;
            request.waiting_hint.total_reuse_tokens = 0;
            return Ok(());
        }

        self.kv.retain_pages(&prefix_match.block_ids);
        self.radix.retain_blocks(&prefix_match.block_ids);
        if let Err(err) =
            self.kv
                .attach_pages(slot, &prefix_match.block_ids, prefix_match.matched_len)
        {
            self.radix.release_blocks(&prefix_match.block_ids);
            self.kv.release_pages(&prefix_match.block_ids);
            return Err(err);
        }

        // Restore the recurrent sidecar for hybrid models (Qwen3.5/3.6). No-op for
        // full-attention-only backends. On miss, release the attached pages and fall
        // back to full recompute — a zeroed linear-attn state with non-zero full-attn
        // KV causes a cross-type mismatch that corrupts model output.
        if let Err(err) = self.executor.restore_prefix_sidecar(
            slot,
            &request.prompt_tokens,
            prefix_match.matched_len,
            &prefix_match.block_ids,
        ) {
            log::warn!(
                "recurrent sidecar restore failed for slot {slot}: {err:#}; \
                 full recompute fallback"
            );
            // Undo retain_pages + attach_pages; executor already reset full_attn_kv.
            self.kv.free_slot(slot);
            self.radix.release_blocks(&prefix_match.block_ids);
            self.kv.release_pages(&prefix_match.block_ids);
            // request.prefill_start_pos stays at 0 (the pre-attach default).
            request.phase = RequestPhase::Prefilling { progress: 0 };
            request.waiting_hint.immediate_reuse_tokens = 0;
            request.waiting_hint.total_reuse_tokens = 0;
            return Ok(());
        }

        self.prefix_cache_stats.hits = self.prefix_cache_stats.hits.saturating_add(1);
        self.prefix_cache_stats.hit_tokens = self
            .prefix_cache_stats
            .hit_tokens
            .saturating_add(prefix_match.matched_len as u64);
        self.prefix_cache_stats.hit_pages = self
            .prefix_cache_stats
            .hit_pages
            .saturating_add(prefix_match.block_ids.len() as u64);

        request.prefill_start_pos = prefix_match.matched_len.min(request.prompt_len());
        request.reused_prefix_pages = prefix_match.block_ids;
        request.used_prefix_restore = true;
        request.waiting_hint.immediate_reuse_tokens = request.prefill_start_pos;
        request.waiting_hint.total_reuse_tokens = request.prefill_start_pos;
        request.phase = if request.prefill_start_pos == request.prompt_len() {
            RequestPhase::Decoding
        } else {
            RequestPhase::Prefilling {
                progress: request.prefill_start_pos,
            }
        };
        Ok(())
    }

    /// Cross-request position-0 prefix reuse for backends whose KV cannot be
    /// page-reattached at arbitrary positions (DSv4). The host radix page route
    /// returns no match for these backends (`reusable_prefix_blocks == 0`), so
    /// the engine asks the executor whether it holds a whole-slot image captured
    /// at absolute position 0 whose tokens are a leading prefix of this prompt.
    /// On a hit it allocates the prefix KV pages on `slot`, restores the image
    /// (KV lands at the same positions it was captured at), and sets
    /// `prefill_start_pos = matched_len` so the existing tail-prefill flow runs.
    ///
    /// Returns the matched length (0 ⇒ no reuse; caller falls through to the
    /// page-radix attach, which is the no-op `PrefixMatch::empty()` for DSv4).
    /// On any restore failure the slot's KV pages are freed and 0 is returned so
    /// the caller re-prefills the whole prompt from a clean slot.
    pub(crate) fn attach_cached_prefix(
        &mut self,
        slot: usize,
        request: &mut RequestState,
    ) -> Result<usize> {
        if !self.config.enable_prefix_cache {
            return Ok(0);
        }
        let matched_len = self
            .executor
            .cached_prefix_match_len(&request.prompt_tokens)?
            .min(request.prompt_len());
        if matched_len == 0 {
            return Ok(0);
        }
        // Never restore the full prompt: the last token must still go through
        // a genuine forward+sample (matching fresh-prefill's own final-chunk
        // behavior), never a direct jump to `Decoding` with empty
        // `generated_tokens` — see
        // docs/experience/errors/2026-07-06-dsv4-concurrent-decode-digit-corruption-unresolved.md,
        // "Layer-0-15 residual bisection" section. This route allocates a
        // fresh, private slot (no radix page sharing), so shaving one raw
        // token off is safe without any block-alignment concern.
        let matched_len = if matched_len == request.prompt_len() {
            matched_len - 1
        } else {
            matched_len
        };
        if matched_len == 0 {
            return Ok(0);
        }
        // Allocate the prefix KV pages (sets the host pool's slot seq_len to
        // matched_len), then restore the device image into them.
        if let Err(err) = self.alloc_with_prefix_reclaim(slot, matched_len) {
            log::warn!(
                "position-0 prefix reuse skipped for request {}: KV alloc failed: {err:#}",
                request.handle.id()
            );
            return Ok(0);
        }
        let slot_pages = self.kv.page_indices(slot).to_vec();
        if let Err(err) = self.executor.restore_cached_prefix(
            slot,
            &request.prompt_tokens,
            matched_len,
            &slot_pages,
        ) {
            log::warn!(
                "position-0 prefix restore failed for request {}: {err:#}; recomputing",
                request.handle.id()
            );
            self.kv.free_slot(slot);
            self.kv_system_metrics.fallback_recompute =
                self.kv_system_metrics.fallback_recompute.saturating_add(1);
            return Ok(0);
        }

        request.prefill_start_pos = matched_len;
        request.used_prefix_restore = true;
        request.waiting_hint.immediate_reuse_tokens = matched_len;
        request.waiting_hint.total_reuse_tokens = matched_len;
        request.phase = if matched_len == request.prompt_len() {
            RequestPhase::Decoding
        } else {
            RequestPhase::Prefilling {
                progress: matched_len,
            }
        };
        // Placement-neutral counters only: the engine cannot see whether the
        // whole-slot blob was served from host RAM or NVMe, so the page-route
        // `reuse_hit_{resident,host_demoted,disk}` buckets stay untouched
        // (they would lie under `--kv-dram 0`). Reuse truth for this route =
        // prefix_cache hits/hit_tokens + prefix_match_full_blocks.
        let block_size = self.radix.block_size().max(1);
        let pages = (matched_len / block_size) as u64;
        if pages > 0 {
            self.prefix_cache_stats.hits = self.prefix_cache_stats.hits.saturating_add(1);
            self.prefix_cache_stats.hit_tokens = self
                .prefix_cache_stats
                .hit_tokens
                .saturating_add(matched_len as u64);
            self.kv_system_metrics.prefix_match_full_blocks = self
                .kv_system_metrics
                .prefix_match_full_blocks
                .saturating_add(pages);
        }
        Ok(matched_len)
    }

    // record_prefix_tier_hits moved into materialize_prefix_blocks.

    pub(crate) fn alloc_with_prefix_reclaim(&mut self, slot: usize, tokens: usize) -> Result<()> {
        let needed = self.kv.append_pages_needed(slot, tokens);
        if needed > self.kv.free_pages() {
            self.evict_prefix_cache_for_pages(needed - self.kv.free_pages());
        }

        match self.kv.alloc(slot, tokens) {
            Ok(()) => Ok(()),
            Err(first_err) => {
                let needed = self.kv.append_pages_needed(slot, tokens);
                let reclaimed = self.evict_prefix_cache_for_pages(needed);
                if reclaimed == 0 {
                    return Err(first_err);
                }
                self.kv.alloc(slot, tokens).map_err(|retry_err| {
                    anyhow!(
                        "KV alloc retry failed after reclaiming {reclaimed} pages: first error: {first_err}; retry error: {retry_err}"
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
        if !self.kv.is_active() {
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

    pub(crate) fn evict_prefix_cache_for_pages(&mut self, pages_needed: usize) -> usize {
        if pages_needed == 0 {
            return 0;
        }
        if self.kv_tier_capacity() == 0 {
            let pages = self.radix.evict_lru(pages_needed);
            let reclaimed = pages.len();
            if reclaimed > 0 {
                self.kv.release_pages(&pages);
                self.executor.release_prefix_pages(&pages);
            }
            self.drain_dropped_tier_keys();
            return reclaimed;
        }

        // Tier path: demote each LRU page into the backend host store instead
        // of dropping it; fall back to plain eviction when the store refuses.
        let mut reclaimed = 0usize;
        while reclaimed < pages_needed {
            let pages = self.radix.lru_evictable_pages(pages_needed - reclaimed);
            if pages.is_empty() {
                break;
            };
            let demoted = self.try_demote_pages(&pages);
            for &page in &pages[..demoted] {
                self.kv.release_pages(&[page]);
                self.executor.release_prefix_pages(&[page]);
                reclaimed += 1;
            }
            if demoted > 0 {
                continue;
            }
            let mut blocked = false;
            for &page in &pages[demoted..] {
                if !self.radix.evict_page(page) {
                    // Neither demotable nor severable — stop instead of spinning.
                    blocked = true;
                    break;
                }
                self.kv.release_pages(&[page]);
                self.executor.release_prefix_pages(&[page]);
                reclaimed += 1;
                if reclaimed >= pages_needed {
                    break;
                }
            }
            if blocked {
                break;
            }
        }
        self.drain_dropped_tier_keys();
        reclaimed
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
            return self.clamp_prefix_to_backend(matched);
        }
        let mut blocks = self.radix.tiered_longest_prefix_match(tokens).blocks;
        let reusable = self
            .executor
            .reusable_prefix_blocks(&blocks)
            .min(blocks.len());
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
