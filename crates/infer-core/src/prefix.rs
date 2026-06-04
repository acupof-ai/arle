//! Radix prefix-cache choreography for [`Engine`].
//!
//! `impl Engine` methods orchestrating `self.radix` (the trie in `radix.rs`) with
//! the `KvPrefixStore`/`KvAllocator` ops on `self.kv`: attach a matched prefix,
//! publish sealed blocks on finish, release reused pages, and reclaim via LRU
//! eviction when allocation would fail.

use anyhow::{Result, anyhow};
use infer_seam::{BackendExecutor, KvPool};

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

    pub(crate) fn attach_prefix_to_request(
        &mut self,
        slot: usize,
        request: &mut RequestState,
        prefix_match: PrefixMatch,
    ) -> Result<()> {
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
            self.kv.retain_pages(&newly_cached);
        }
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
        let pages = self.radix.evict_lru(pages_needed);
        let reclaimed = pages.len();
        if reclaimed > 0 {
            self.kv.release_pages(&pages);
        }
        reclaimed
    }
}
