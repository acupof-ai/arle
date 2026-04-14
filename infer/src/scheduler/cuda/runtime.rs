use super::*;

fn raw_prefix_len(cached: &[u32], prompt_tokens: &[u32]) -> usize {
    cached
        .iter()
        .zip(prompt_tokens.iter())
        .take_while(|(a, b)| a == b)
        .count()
}

fn effective_prefix_reuse_len(cached: &[u32], prompt_tokens: &[u32]) -> usize {
    let prefix_len = raw_prefix_len(cached, prompt_tokens);
    if prefix_len > 0 && prefix_len == prompt_tokens.len() && prefix_len < cached.len() {
        0
    } else {
        prefix_len
    }
}

fn best_prefix_slot_for_cached_prompts(
    cached_prompts: &[Vec<u32>],
    free_slots: &[usize],
    prompt_tokens: &[u32],
) -> usize {
    let mut best_slot = free_slots[0];
    let mut best_match = 0usize;

    for &slot in free_slots {
        let match_len = effective_prefix_reuse_len(&cached_prompts[slot], prompt_tokens);
        if match_len > best_match {
            best_match = match_len;
            best_slot = slot;
        }
    }

    best_slot
}

impl<M: ModelForward> Scheduler<M> {
    /// Run the scheduler loop. Blocks until all handles are dropped.
    pub fn run(mut self) {
        self.warmup_cuda_graphs();
        info!("Scheduler run loop started");
        loop {
            while let Ok(req) = self.request_rx.try_recv() {
                self.waiting_count.fetch_sub(1, Ordering::Relaxed);
                self.waiting.push_back(req);
            }

            if self.active.is_empty() && self.waiting.is_empty() {
                match self.request_rx.blocking_recv() {
                    Some(req) => {
                        self.waiting_count.fetch_sub(1, Ordering::Relaxed);
                        self.waiting.push_back(req);
                    }
                    None => {
                        info!("Scheduler shutting down: all handles dropped");
                        break;
                    }
                }
            }

            let step_start = std::time::Instant::now();
            self.assign_slots();
            let assign_us = step_start.elapsed().as_micros();

            let step_t = std::time::Instant::now();
            self.step();
            let step_us = step_t.elapsed().as_micros();

            let clean_t = std::time::Instant::now();
            self.cleanup();
            let clean_us = clean_t.elapsed().as_micros();

            let total_us = step_start.elapsed().as_micros();
            if total_us > 50_000 {
                // Log slow iterations (>50ms)
                info!(
                    "Scheduler step: assign={}us step={}us cleanup={}us total={}us active={}",
                    assign_us,
                    step_us,
                    clean_us,
                    total_us,
                    self.active.len()
                );
            }
        }
    }

    fn assign_slots(&mut self) {
        while !self.waiting.is_empty() {
            let free_slots = self.free_slots();
            if free_slots.is_empty() {
                break;
            }

            let incoming = self.waiting.pop_front().expect("checked non-empty above");
            let prompt_tokens = match self.tokenizer.encode(&incoming.prompt) {
                Ok(tokens) if !tokens.is_empty() => tokens,
                Ok(_) => {
                    error!("Empty prompt after tokenization, skipping");
                    continue;
                }
                Err(e) => {
                    error!("Tokenization error: {}", e);
                    continue;
                }
            };

            // Pick the free slot with the best prefix match to maximize KV reuse.
            let slot_idx = self.best_prefix_slot(&free_slots, &prompt_tokens);

            // M1 shadow observer: ask the global RadixCache what the best
            // achievable prefix hit length would be across ALL previously
            // completed requests, not just the per-slot cached_prompts
            // entries. This is logging only — the actual slot selection
            // still goes through `best_prefix_slot` above because the
            // radix does not own paged-pool page references yet (M2).
            //
            // The lookup bumps `ref_count` on matched nodes internally;
            // release immediately so we do not leak refs into the eviction
            // barrier while the radix is still observational.
            let (radix_hit_len, radix_blocks) = self.prefix_cache.lookup(&prompt_tokens);
            if radix_hit_len > 0 {
                info!(
                    "radix shadow: best cross-slot prefix hit = {}/{} tokens ({} blocks, slot selection unchanged)",
                    radix_hit_len,
                    prompt_tokens.len(),
                    radix_blocks.len(),
                );
            }
            self.prefix_cache.release(&radix_blocks);

            let id = self.next_id;
            self.next_id += 1;

            let prefix_len =
                effective_prefix_reuse_len(&self.cached_prompts[slot_idx], &prompt_tokens);
            if prefix_len > 0 {
                info!(
                    "Request {} → slot {} (prompt={} tokens, prefix_reuse={}, queue={})",
                    id,
                    slot_idx,
                    prompt_tokens.len(),
                    prefix_len,
                    self.waiting.len()
                );
            } else {
                info!(
                    "Request {} → slot {} (prompt={} tokens, queue={})",
                    id,
                    slot_idx,
                    prompt_tokens.len(),
                    self.waiting.len()
                );
            }

            self.active.push(ActiveRequest {
                id,
                slot_idx,
                prompt: incoming.prompt,
                prompt_tokens,
                generated_tokens: Vec::new(),
                max_tokens: incoming.max_tokens,
                sampling: incoming.sampling,
                stop: incoming.stop,
                session_id: incoming.session_id,
                delta_tx: incoming.delta_tx,
                full_decoded: String::new(),
                decoded_token_count: 0,
                sent_len: 0,
                phase: Phase::New,
                cacheable_prompt_len: 0,
                prefix_byte_len: 0,
                latest_logprob: None,
            });
        }
    }

    /// Find all free slot indices.
    fn free_slots(&self) -> Vec<usize> {
        let in_use: Vec<usize> = self.active.iter().map(|a| a.slot_idx).collect();
        (0..self.states.len())
            .filter(|i| !in_use.contains(i))
            .collect()
    }

    /// Pick the free slot whose cached prompt best matches the given tokens.
    /// Falls back to the first free slot if no prefix matches.
    fn best_prefix_slot(&self, free_slots: &[usize], prompt_tokens: &[u32]) -> usize {
        best_prefix_slot_for_cached_prompts(&self.cached_prompts, free_slots, prompt_tokens)
    }

    fn cleanup(&mut self) {
        let mut i = 0;
        while i < self.active.len() {
            if matches!(self.active[i].phase, Phase::Finished) {
                let req = self.active.remove(i);
                let gen_tokens = req.generated_tokens.len() as u64;

                // M2a cleanup ordering: publish to the radix + retain
                // pool pages BEFORE `free_slot`. `publish_to_prefix_cache`
                // reads `token_indices(slot)` and calls `retain_pages`
                // on the contiguous span that backs the prompt, so
                // the subsequent `free_slot` leaves those pages in
                // limbo instead of pushing them back to the primary
                // free list. Pages beyond the last full block (i.e.
                // the `prompt.len() % block_size` tail plus any
                // generated tokens) are not retained and go back to
                // the free pool as before.
                if let Some(prompt_tokens) = req.cached_prompt_to_publish() {
                    let prompt_vec = prompt_tokens.to_vec();
                    // Per-slot cached prompt stays as the legacy
                    // authoritative backing for the linear
                    // `best_prefix_slot` scan. M2b will retire it.
                    self.cached_prompts[req.slot_idx] = prompt_vec.clone();
                    self.publish_to_prefix_cache(req.slot_idx, &prompt_vec);
                } else {
                    self.cached_prompts[req.slot_idx].clear();
                }
                self.paged_kv_pool.free_slot(req.slot_idx);
                let _ = self.states[req.slot_idx].offload_kv_if_needed();

                self.total_completed += 1;
                self.total_generated_tokens += gen_tokens;

                info!(
                    "Request {} done: {} tokens (active={}, waiting={})",
                    req.id,
                    gen_tokens,
                    self.active.len(),
                    self.waiting.len()
                );

                if self.total_completed % STATS_LOG_INTERVAL == 0 {
                    info!(
                        "Scheduler stats: completed={}, generated_tokens={}, active={}, waiting={}",
                        self.total_completed,
                        self.total_generated_tokens,
                        self.active.len(),
                        self.waiting.len()
                    );
                }

                if self.last_served >= self.active.len() && !self.active.is_empty() {
                    self.last_served = 0;
                }
            } else {
                i += 1;
            }
        }

        // M2a: amortised LRU eviction for the prefix cache.
        // Runs after the per-request free_slot loop so the pool's
        // retained fraction is fresh. No-op unless retained pages
        // crossed `PREFIX_CACHE_HIGH_WATER`; then evicts down to
        // `PREFIX_CACHE_LOW_WATER`. See
        // `core::Scheduler::evict_prefix_cache_if_pressured`.
        let _reclaimed = self.evict_prefix_cache_if_pressured();
    }
}

#[cfg(test)]
mod tests {
    use super::{best_prefix_slot_for_cached_prompts, effective_prefix_reuse_len, raw_prefix_len};

    #[test]
    fn effective_prefix_reuse_ignores_prompt_that_is_prefix_of_longer_cache() {
        let cached = vec![1, 2, 3, 4];
        let prompt = vec![1, 2, 3];
        assert_eq!(raw_prefix_len(&cached, &prompt), 3);
        assert_eq!(effective_prefix_reuse_len(&cached, &prompt), 0);
    }

    #[test]
    fn effective_prefix_reuse_keeps_exact_and_extendable_hits() {
        assert_eq!(effective_prefix_reuse_len(&[1, 2, 3], &[1, 2, 3]), 3);
        assert_eq!(effective_prefix_reuse_len(&[1, 2, 3], &[1, 2, 3, 4]), 3);
        assert_eq!(effective_prefix_reuse_len(&[1, 9, 3], &[1, 2, 3, 4]), 1);
    }

    #[test]
    fn best_prefix_slot_prefers_effectively_reusable_prefixes() {
        let cached_prompts = vec![vec![1, 2, 3, 4], vec![1, 2], vec![1]];
        let free_slots = vec![0, 1, 2];
        let prompt = vec![1, 2, 3];

        assert_eq!(
            best_prefix_slot_for_cached_prompts(&cached_prompts, &free_slots, &prompt),
            1
        );
    }
}
