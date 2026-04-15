use super::core::STAGE_WAIT_KEEPALIVE_TICKS;
use super::*;
use crate::prefix_cache::BlockMetadataUpdate;

fn best_reusable_slot_for_radix_hit(
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

fn lookup_blocks_ready_on_gpu(blocks: &[crate::kv_tier::LookupBlock]) -> bool {
    blocks
        .iter()
        .filter(|block| !matches!(block.hit_kind, crate::kv_tier::HitKind::Miss))
        .all(|block| matches!(block.hit_kind, crate::kv_tier::HitKind::ReadyOnGpu))
}

impl<M: ModelForward> Scheduler<M> {
    fn drain_coordinator_events(&mut self) {
        loop {
            match self.coordinator_events.try_recv() {
                Ok(crate::kv_tier::CoordinatorEvent::CommandQueued(_))
                | Ok(crate::kv_tier::CoordinatorEvent::StagingQueued { .. }) => {}
                Ok(crate::kv_tier::CoordinatorEvent::StagingCompleted { ticket }) => {
                    let Some(staged) = self.stage_waiting.remove(&ticket) else {
                        log::debug!(
                            "Dropping staging completion for unknown ticket {}",
                            ticket.0
                        );
                        continue;
                    };
                    let waited_ticks = self
                        .prefix_cache
                        .logical_clock()
                        .saturating_sub(staged.enqueued_at_clock);
                    for &block_id in &staged.block_ids {
                        let _ = self.prefix_cache.update_block_metadata(
                            block_id,
                            BlockMetadataUpdate {
                                // A4 stub: ready-but-not-yet-owned-by-slot sentinel; real slot
                                // assigned in publish_to_prefix_cache.
                                location: Some(crate::kv_tier::BlockLocation::Gpu {
                                    slot: u32::MAX,
                                }),
                                ..BlockMetadataUpdate::default()
                            },
                        );
                    }
                    self.prefix_cache.release(&staged.block_ids);
                    for &block_id in &staged.block_ids {
                        let _ = self.prefix_cache.update_block_metadata(
                            block_id,
                            BlockMetadataUpdate {
                                soft_pin_until: Some(None),
                                ..BlockMetadataUpdate::default()
                            },
                        );
                    }
                    log::debug!(
                        "Staging completed for ticket={} (prompt={} tokens, staged_blocks={}, waited_ticks={})",
                        ticket.0,
                        staged.prompt_tokens.len(),
                        staged.block_ids.len(),
                        waited_ticks
                    );
                    self.waiting.push_front(staged.request);
                    self.waiting_count.fetch_add(1, Ordering::Relaxed);
                }
                Err(crossbeam_channel::TryRecvError::Empty) => break,
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    error!("Coordinator event channel disconnected");
                    self.coordinator_unavailable = true;
                    break;
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

            if self.active.is_empty() && self.waiting.is_empty() && self.stage_waiting.is_empty() {
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
            let _ = self.evict_prefix_cache_if_pressured();
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

            let planner = if self.coordinator_unavailable {
                None
            } else {
                Some(&self.coordinator_handle as &dyn crate::kv_tier::StagePlanner)
            };
            let lookup = self.prefix_cache.lookup_or_stage(
                &prompt_tokens,
                crate::kv_tier::LookupHeuristics::default(),
                planner,
            );
            let radix_blocks: Vec<_> = lookup
                .blocks
                .iter()
                .filter_map(|block| block.block_id)
                .collect();
            if let Some(ticket) = lookup.staging_ticket {
                if lookup.recompute_advised {
                    self.prefix_cache.release(&radix_blocks);
                } else {
                    let stage_deadline = self
                        .prefix_cache
                        .logical_clock()
                        .saturating_add(STAGE_WAIT_KEEPALIVE_TICKS);
                    for &block_id in &radix_blocks {
                        let _ = self.prefix_cache.update_block_metadata(
                            block_id,
                            BlockMetadataUpdate {
                                soft_pin_until: Some(Some(stage_deadline)),
                                ..BlockMetadataUpdate::default()
                            },
                        );
                    }
                    log::info!(
                        "Request {} staged on ticket={} (prompt={} tokens, staged_blocks={})",
                        self.next_id,
                        ticket.0,
                        prompt_tokens.len(),
                        radix_blocks.len()
                    );
                    self.stage_waiting.insert(
                        ticket,
                        super::core::StagedAdmission {
                            request: incoming,
                            prompt_tokens,
                            block_ids: radix_blocks,
                            enqueued_at_clock: self.prefix_cache.logical_clock(),
                        },
                    );
                    continue;
                }
            }

            let ready_on_gpu = lookup_blocks_ready_on_gpu(&lookup.blocks);
            let radix_hit_len = if ready_on_gpu && !lookup.recompute_advised {
                lookup.matched_len
            } else {
                0
            };
            let gpu_ready_prefix_blocks: Vec<_> = lookup
                .blocks
                .iter()
                .take_while(|block| matches!(block.hit_kind, crate::kv_tier::HitKind::ReadyOnGpu))
                .filter_map(|block| block.block_id)
                .collect();
            let reusable_gpu_prefix = best_reusable_slot_for_radix_hit(
                &gpu_ready_prefix_blocks,
                &free_slots,
                &self.block_owner_slots,
                &self.slot_materialized_prompt_lens,
                self.prefix_cache.block_size(),
            );
            let reusable = if ready_on_gpu && !lookup.recompute_advised {
                reusable_gpu_prefix
            } else {
                None
            };
            let (slot_idx, reusable_prefix_len, reusable_cached_prompt_len) =
                reusable.unwrap_or((free_slots[0], 0, 0));
            self.prefix_cache.release(&radix_blocks);

            let id = self.next_id;
            self.next_id += 1;

            if reusable_prefix_len > 0 {
                info!(
                    "Request {} → slot {} (prompt={} tokens, radix_hit={}, reusable_prefix={}, cached_len={}, queue={})",
                    id,
                    slot_idx,
                    prompt_tokens.len(),
                    radix_hit_len,
                    reusable_prefix_len,
                    reusable_cached_prompt_len,
                    self.waiting.len()
                );
            } else {
                let bytes_not_on_gpu =
                    lookup.matched_len > 0 && (!ready_on_gpu || lookup.recompute_advised);
                let no_reusable_free_slot = lookup.matched_len > 0
                    && !gpu_ready_prefix_blocks.is_empty()
                    && reusable_gpu_prefix.is_none();
                // A radix match is only reusable when the bytes already live in
                // T0 and a free slot still materializes that prefix. When
                // either precondition fails we degrade to cold prefill, but we
                // keep both blockers in the log so admission debugging does not
                // lose the "no reusable free slot" signal behind a staging miss.
                if bytes_not_on_gpu || no_reusable_free_slot {
                    info!(
                        "Request {} → slot {} (prompt={} tokens, radix_hit={} not reusable: bytes_not_on_gpu={}, no_free_slot={}, queue={})",
                        id,
                        slot_idx,
                        prompt_tokens.len(),
                        lookup.matched_len,
                        bytes_not_on_gpu,
                        no_reusable_free_slot,
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
                reusable_prefix_len,
                reusable_cached_prompt_len,
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

    fn cleanup(&mut self) {
        let mut i = 0;
        while i < self.active.len() {
            if matches!(self.active[i].phase, Phase::Finished) {
                let req = self.active.remove(i);
                let gen_tokens = req.generated_tokens.len() as u64;
                self.clear_slot_prefix_ownership(req.slot_idx);

                if let Some(prompt_tokens) = req.cached_prompt_to_publish() {
                    let prompt_vec = prompt_tokens.to_vec();
                    self.slot_materialized_prompt_lens[req.slot_idx] = prompt_vec.len();
                    self.publish_to_prefix_cache(req.slot_idx, &prompt_vec, req.session_id.clone());
                } else {
                    self.slot_materialized_prompt_lens[req.slot_idx] = 0;
                }
                self.paged_kv_pool.free_slot(req.slot_idx);

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
