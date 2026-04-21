use super::{ModelForward, Ordering, Phase, STATS_LOG_INTERVAL, Scheduler, error, info, warn};
use crate::prefix_cache::BlockMetadataUpdate;

pub(super) fn best_reusable_slot_for_radix_hit(
    matched_blocks: &[crate::prefix_cache::BlockId],
    free_slots: &[usize],
    block_owner_slots: &std::collections::HashMap<crate::prefix_cache::BlockId, usize>,
    slot_materialized_prompt_lens: &[usize],
    block_size: usize,
) -> Option<(usize, usize, usize)> {
    for (idx, &bid) in matched_blocks.iter().enumerate().rev() {
        let Some(&slot_idx) = block_owner_slots.get(&bid) else {
            continue;
        };
        if !free_slots.contains(&slot_idx) {
            continue;
        }
        let reusable_prefix_len = (idx + 1) * block_size;
        let cached_prompt_len = slot_materialized_prompt_lens
            .get(slot_idx)
            .copied()
            .unwrap_or_default();
        if cached_prompt_len >= reusable_prefix_len && reusable_prefix_len > 0 {
            return Some((slot_idx, reusable_prefix_len, cached_prompt_len));
        }
    }
    None
}

pub(super) fn lookup_blocks_ready_on_gpu(blocks: &[crate::kv_tier::LookupBlock]) -> bool {
    blocks
        .iter()
        .filter(|block| !matches!(block.hit_kind, crate::kv_tier::HitKind::Miss))
        .all(|block| matches!(block.hit_kind, crate::kv_tier::HitKind::ReadyOnGpu))
}

pub(super) fn sort_waiting_queue_by_priority(
    waiting: &mut std::collections::VecDeque<super::IncomingRequest>,
) {
    let waiting = waiting.make_contiguous();
    waiting.sort_by_key(|req| std::cmp::Reverse(req.priority));
}

impl<M: ModelForward> Scheduler<M> {
    fn drain_coordinator_events(&mut self) {
        loop {
            match self.coordinator_events.try_recv() {
                Ok(
                    crate::kv_tier::CoordinatorEvent::CommandQueued(_)
                    | crate::kv_tier::CoordinatorEvent::SpillQueued { .. },
                ) => {}
                Err(crossbeam_channel::TryRecvError::Empty) => break,
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    error!("Coordinator event channel disconnected");
                    break;
                }
                Ok(crate::kv_tier::CoordinatorEvent::SpillCompleted { ticket, locations }) => {
                    if let Some((block_id, region)) = self.spill_waiting.remove(&ticket) {
                        for (completed_block, location) in locations {
                            if completed_block != block_id {
                                continue;
                            }
                            let _ = self.prefix_cache.update_block_metadata(
                                block_id,
                                BlockMetadataUpdate {
                                    location: Some(crate::kv_tier::BlockLocation::Disk {
                                        fingerprint: location.fingerprint,
                                        payload_len: location.payload_len,
                                    }),
                                    ..BlockMetadataUpdate::default()
                                },
                            );
                            self.release_host_region(region);
                        }
                    }
                }
                Ok(crate::kv_tier::CoordinatorEvent::SpillFailed {
                    ticket,
                    failed_block,
                    reason,
                }) => {
                    warn!(
                        "Spill failed for ticket {} on block {:?}: {}",
                        ticket.0, failed_block, reason
                    );
                    self.spill_waiting.remove(&ticket);
                }
            }
        }
    }

    /// Run the scheduler loop. Blocks until all handles are dropped.
    pub fn run(mut self) {
        self.warmup_cuda_graphs();
        info!("Scheduler run loop started");
        loop {
            while let Ok(req) = self.request_rx.try_recv() {
                self.waiting_count.fetch_sub(1, Ordering::Relaxed);
                self.waiting.push_back(req);
            }

            self.drain_coordinator_events();

            if self.active_len() == 0 && self.waiting.is_empty() {
                if let Some(req) = self.request_rx.blocking_recv() {
                    self.waiting_count.fetch_sub(1, Ordering::Relaxed);
                    self.waiting.push_back(req);
                } else {
                    info!("Scheduler shutting down: all handles dropped");
                    break;
                }
            }

            let step_start = std::time::Instant::now();
            // FUTURE WORK (GPU/CPU overlap): `self.step()` already overlaps
            // decode with `emit_delta`, but batched decode itself is still
            // serial because `step_decode_batch()` runs
            // `forward_decode_batch(...)` and then immediately
            // `sample_batch_greedy(...)`, whose fast path launches argmax,
            // `ctx.sync()`s, and reads tokens/logprobs back. Real overlap
            // needs a `step_launch()` / `step_readback()` split at that
            // boundary; loop reordering alone does not create it.
            let step_t = std::time::Instant::now();
            self.step();
            let step_us = step_t.elapsed().as_micros();

            let clean_t = std::time::Instant::now();
            self.cleanup();
            let clean_us = clean_t.elapsed().as_micros();
            self.metrics.set_active(self.active_len() as u64);
            self.metrics.set_waiting(self.waiting.len() as u64);
            if self.paged_kv_pool.is_active() {
                // Both in token units so kv_util = (total-free)/total is correct.
                let total =
                    (self.paged_kv_pool.max_total_pages * self.paged_kv_pool.page_size) as u64;
                let free = self.paged_kv_pool.free_count() as u64;
                self.metrics.set_kv_gpu_blocks(free, total);
            }
            // Throttled GPU memory query — at most once per second.
            if self.last_mem_query.elapsed().as_secs() >= 1 {
                self.last_mem_query = std::time::Instant::now();
                if let Ok((free, total)) =
                    crate::backend::cuda::tensor::DeviceContext::gpu_memory_info()
                {
                    let active = (total - free) as u64;
                    self.peak_mem_bytes = self.peak_mem_bytes.max(active);
                    self.metrics
                        .set_memory_bytes(active, self.peak_mem_bytes, 0);
                }
            }

            let total_us = step_start.elapsed().as_micros();
            if total_us > 50_000 {
                // Log slow iterations (>50ms)
                info!(
                    "Scheduler step: step={}us cleanup={}us total={}us active={}",
                    step_us,
                    clean_us,
                    total_us,
                    self.active_len()
                );
            }
        }
    }

    /// Find all free slot indices.
    pub(super) fn free_slots(&self) -> Vec<usize> {
        self.active
            .iter()
            .enumerate()
            .filter_map(|(slot_idx, req)| req.is_none().then_some(slot_idx))
            .collect()
    }

    fn cleanup(&mut self) {
        for slot_idx in 0..self.active.len() {
            let finished = matches!(
                self.request(slot_idx).map(|req| &req.phase),
                Some(Phase::Finished)
            );
            if finished {
                let req = self.active[slot_idx]
                    .take()
                    .expect("finished slot must hold a request");
                let gen_tokens = req.generated_tokens.len() as u64;
                self.release_attached_prefix_blocks(&req.attached_prefix_blocks);
                self.dequeue_prefill(slot_idx);
                self.dequeue_running(slot_idx);
                self.clear_slot_prefix_ownership(slot_idx);

                if let Some(prompt_tokens) = req.cached_prompt_to_publish() {
                    let prompt_vec = prompt_tokens.to_vec();
                    self.slot_materialized_prompt_lens[slot_idx] = prompt_vec.len();
                    self.publish_to_prefix_cache(slot_idx, &prompt_vec, req.session_id.as_ref());
                } else {
                    self.slot_materialized_prompt_lens[slot_idx] = 0;
                }
                self.paged_kv_pool.free_slot(slot_idx);

                self.total_completed += 1;
                self.total_generated_tokens += gen_tokens;
                let e2e_s = req.admitted_at.elapsed().as_secs_f64();
                let ttft_s = req
                    .first_token_at
                    .map_or(e2e_s, |t| t.duration_since(req.admitted_at).as_secs_f64());
                let tpot_s = if gen_tokens > 1 {
                    (e2e_s - ttft_s).max(0.0) / (gen_tokens - 1) as f64
                } else {
                    0.0
                };
                self.metrics.record_request_completed(
                    req.prompt_tokens.len() as u64,
                    gen_tokens,
                    ttft_s,
                    tpot_s,
                    e2e_s,
                );

                info!(
                    "Request {} done: {} tokens (active={}, waiting={})",
                    req.id,
                    gen_tokens,
                    self.active_len(),
                    self.waiting.len()
                );

                if self.total_completed.is_multiple_of(STATS_LOG_INTERVAL) {
                    info!(
                        "Scheduler stats: completed={}, generated_tokens={}, active={}, waiting={}",
                        self.total_completed,
                        self.total_generated_tokens,
                        self.active_len(),
                        self.waiting.len()
                    );
                }
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
    use super::best_reusable_slot_for_radix_hit;
    use crate::prefix_cache::BlockId;
    use std::collections::HashMap;

    #[test]
    fn best_reusable_slot_prefers_deepest_free_owned_block() {
        let matched_blocks = vec![BlockId(10), BlockId(20), BlockId(30)];
        let free_slots = vec![1, 2];
        let mut owners = HashMap::new();
        owners.insert(BlockId(10), 0);
        owners.insert(BlockId(20), 1);
        owners.insert(BlockId(30), 2);

        let reusable = best_reusable_slot_for_radix_hit(
            &matched_blocks,
            &free_slots,
            &owners,
            &[0, 32, 48],
            16,
        );
        assert_eq!(reusable, Some((2, 48, 48)));
    }

    #[test]
    fn best_reusable_slot_skips_busy_or_stale_slots() {
        let matched_blocks = vec![BlockId(10), BlockId(20)];
        let free_slots = vec![1];
        let mut owners = HashMap::new();
        owners.insert(BlockId(10), 1);
        owners.insert(BlockId(20), 0);

        let reusable =
            best_reusable_slot_for_radix_hit(&matched_blocks, &free_slots, &owners, &[0, 8], 16);
        assert_eq!(reusable, None);
    }
}
