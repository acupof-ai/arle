//! Radix prefix-cache choreography for [`Engine`].
//!
//! `impl Engine` methods orchestrating `self.radix` (the trie in `radix.rs`) with
//! the `KvPrefixStore`/`KvAllocator` ops on `self.kv`: attach a matched prefix,
//! publish sealed blocks on finish, release reused pages, and reclaim via LRU
//! eviction when allocation would fail.

use anyhow::{Result, anyhow};
use infer_seam::{BackendExecutor, KvPool};

use crate::radix::TierBlock;
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
        tokens.div_ceil(page_size)
    }

    /// Clamp a radix prefix match to the leading pages the backend can actually
    /// attach. The host radix caches a block at every page boundary, but a
    /// backend whose layers carry prefix-wide recurrent state (Metal GDR /
    /// linear attention) only snapshots that state at the boundaries a forward
    /// pass landed on. Chunked prefill skips interior boundaries, so the radix
    /// can offer a prefix the executor cannot serve; attaching it errors in the
    /// executor and kills the engine thread. Trim the match to the reusable
    /// page count and re-prefill the unsnapshotted tail.
    pub(crate) fn clamp_prefix_to_backend(&self, mut prefix_match: PrefixMatch) -> PrefixMatch {
        let serveable = self.executor.reusable_prefix_pages(&prefix_match.block_ids);
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

        let prefix_match = self.clamp_prefix_to_backend(prefix_match);
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

    pub(crate) fn publish_prefix_blocks(&mut self, slot: usize, request: &RequestState) {
        if !self.kv.is_active() {
            return;
        }

        let block_size = self.radix.block_size().max(1);
        let publishable_tokens = request.prompt_len().min(self.kv.seq_len(slot));
        let sealed_blocks = publishable_tokens / block_size;
        if sealed_blocks == 0 {
            return;
        }

        let sealed_tokens = sealed_blocks * block_size;
        let pages = self
            .kv
            .page_indices_for_token_range(slot, 0, sealed_tokens)
            .to_vec();
        let publish_blocks = sealed_blocks.min(pages.len());
        if publish_blocks == 0 {
            return;
        }

        let token_len = publish_blocks * block_size;
        let newly_cached = self.radix.insert(
            &request.prompt_tokens[..token_len],
            &pages[..publish_blocks],
        );
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
            let Some(page) = self.radix.lru_evictable_page() else {
                break;
            };
            if !self.try_demote_page(page) && !self.radix.evict_page(page) {
                // Neither demotable nor severable — stop instead of spinning.
                break;
            }
            self.kv.release_pages(&[page]);
            self.executor.release_prefix_pages(&[page]);
            reclaimed += 1;
        }
        self.drain_dropped_tier_keys();
        reclaimed
    }

    /// Host-tier capacity in pages; `0` disables every tier path. Tier use is
    /// gated on the prefix cache because demoted blocks are only reachable
    /// through radix prefix matches.
    fn kv_tier_capacity(&self) -> usize {
        if self.config.enable_prefix_cache {
            self.executor.kv_tier_capacity_pages()
        } else {
            0
        }
    }

    /// Copy `page` into the backend host tier and mark its radix node demoted.
    /// Makes room by severing the coldest demoted block when the tier is full.
    fn try_demote_page(&mut self, page: BlockId) -> bool {
        let capacity = self.executor.kv_tier_capacity_pages();
        if self.radix.demoted_block_count() >= capacity {
            match self.radix.lru_demoted_key() {
                Some(coldest) => {
                    self.radix.drop_demoted(coldest);
                    // Drain immediately so the store slot is reusable for the
                    // demote below, not only after the eviction batch.
                    self.drain_dropped_tier_keys();
                }
                None => return false,
            }
        }
        let key = self.next_tier_key;
        self.next_tier_key = self.next_tier_key.wrapping_add(1);
        match self.executor.demote_prefix_pages(&[(page, key)]) {
            Ok(accepted) if accepted >= 1 => {
                if self.radix.demote_block(page, key) {
                    self.kv_tier_stats.demoted_pages =
                        self.kv_tier_stats.demoted_pages.saturating_add(1);
                    true
                } else {
                    // The radix refused (page is not an idle cached leaf);
                    // the store copy is unreachable — drop it.
                    self.executor.drop_kv_tier_entries(&[key]);
                    false
                }
            }
            Ok(_) => false,
            Err(err) => {
                log::warn!("KV tier demote failed for page {page}: {err:#}");
                false
            }
        }
    }

    /// Prefix lookup used at slot attach. With a host tier, demoted blocks in
    /// the matched prefix are promoted back into freshly allocated pages so
    /// the existing resident-only attach path applies unchanged; a promote
    /// failure truncates the match there and the tail re-prefills.
    pub(crate) fn lookup_prefix_for_attach(&mut self, tokens: &[u32]) -> PrefixMatch {
        if self.kv_tier_capacity() == 0 {
            return self.radix.longest_prefix_match(tokens);
        }
        let tiered = self.radix.tiered_longest_prefix_match(tokens);
        let mut block_ids = Vec::with_capacity(tiered.blocks.len());
        for block in tiered.blocks {
            let page = match block {
                TierBlock::Resident(page) => Some(page),
                TierBlock::Demoted(key) => self.promote_demoted_block(key),
            };
            let Some(page) = page else { break };
            block_ids.push(page);
        }
        self.drain_dropped_tier_keys();
        PrefixMatch {
            matched_len: block_ids.len() * self.radix.block_size(),
            block_ids,
        }
    }

    /// Promote one demoted block into a fresh device page and restore its
    /// radix node to residency (cache-owned, like a published page).
    fn promote_demoted_block(&mut self, key: u64) -> Option<BlockId> {
        let page = match self.kv.alloc_detached_pages(1) {
            Ok(mut pages) => pages.pop()?,
            Err(_) => return None,
        };
        match self.executor.promote_prefix_pages(&[(key, page)]) {
            Ok(()) if self.radix.promote_block(key, page) => {
                self.kv.retain_pages(&[page]);
                self.kv_tier_stats.promoted_pages =
                    self.kv_tier_stats.promoted_pages.saturating_add(1);
                Some(page)
            }
            Ok(()) => {
                // Unknown key (should not happen for a key the match returned).
                self.kv.free_detached_pages(&[page]);
                self.executor.drop_kv_tier_entries(&[key]);
                None
            }
            Err(err) => {
                log::warn!("KV tier promote failed for key {key}: {err:#}");
                self.kv.free_detached_pages(&[page]);
                self.radix.drop_demoted(key);
                self.kv_tier_stats.promote_failures =
                    self.kv_tier_stats.promote_failures.saturating_add(1);
                None
            }
        }
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
