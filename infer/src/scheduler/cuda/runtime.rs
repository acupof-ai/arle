use super::{
    ActiveRequest, CompletionStreamDelta, FinishReason, IncomingRequest, ModelForward, Ordering,
    Phase, QueuedRequest, STATS_LOG_INTERVAL, Scheduler, TokenUsage, error, info, warn,
};
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
    waiting: &mut std::collections::VecDeque<QueuedRequest>,
) {
    waiting
        .make_contiguous()
        .sort_by_key(|request| std::cmp::Reverse(request.priority));
}

impl<M: ModelForward> Scheduler<M> {
    fn abort_staged_admission(&mut self, staged: super::core::StagedAdmission) {
        for block in &staged.staged_blocks {
            if block.release_on_abort {
                self.release_host_region(block.host_region);
            }
        }
        self.prefix_cache.release(&staged.block_ids);
        self.waiting.push_front(staged.request);
    }

    fn drain_coordinator_events(&mut self) {
        loop {
            match self.coordinator_events.try_recv() {
                Ok(
                    crate::kv_tier::CoordinatorEvent::CommandQueued(_)
                    | crate::kv_tier::CoordinatorEvent::StagingQueued { .. }
                    | crate::kv_tier::CoordinatorEvent::SpillQueued { .. },
                ) => {}
                Ok(crate::kv_tier::CoordinatorEvent::StagingCompleted { ticket }) => {
                    let Some(staged) = self.stage_waiting.remove(&ticket) else {
                        log::debug!(
                            "Dropping staging completion for unknown ticket {}",
                            ticket.0
                        );
                        continue;
                    };
                    if let Err(err) = self.promote_staged_blocks(&staged) {
                        warn!(
                            "Staging promotion failed for ticket {}: {}. Falling back to cold requeue.",
                            ticket.0, err
                        );
                        self.abort_staged_admission(staged);
                        continue;
                    }
                    let waited_ticks = self
                        .prefix_cache
                        .logical_clock()
                        .saturating_sub(staged.enqueued_at_clock);
                    // Release the stage-era refs but **deliberately leave
                    // `soft_pin_until` at its stage-wait deadline**. The
                    // soft pin keeps eviction off these blocks until the
                    // re-admission's next `lookup_or_stage` refreshes it
                    // down to the normal keepalive_ticks. Without this,
                    // the window between `release` and the second lookup
                    // is unprotected — the next waiting-queue admission pass runs
                    // `evict_prefix_cache_if_pressured` before picking
                    // the parked request back up and could reclaim the
                    // just-staged blocks.
                    self.prefix_cache.release(&staged.block_ids);
                    log::debug!(
                        "Staging completed for ticket={} (prompt={} tokens, staged_blocks={}, waited_ticks={})",
                        ticket.0,
                        staged.request.prompt_tokens.len(),
                        staged.block_ids.len(),
                        waited_ticks
                    );
                    // `waiting_count` tracks the `request_rx` channel
                    // depth (decremented at intake, incremented only at
                    // `SchedulerHandle::submit`). `self.waiting` is the
                    // scheduler-internal VecDeque; moving a request
                    // into its front here is NOT a new submission, so
                    // we MUST NOT `fetch_add(1)` — doing so would
                    // permanently inflate the counter, break the idle
                    // exit guard, and saturate backpressure.
                    self.waiting.push_front(staged.request);
                }
                Err(crossbeam_channel::TryRecvError::Empty) => break,
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    // Coordinator thread gone — we cannot complete any
                    // future StageTicket. Cold-requeue every parked
                    // admission so the idle-exit guard in `run()` can
                    // fire and waiting requests are not leaked. Release
                    // stage-era refs but leave soft_pin_until untouched
                    // (same reasoning as the completion path above —
                    // the re-admission might still hit the prefix).
                    let pending = std::mem::take(&mut self.stage_waiting);
                    if pending.is_empty() {
                        error!("Coordinator event channel disconnected");
                    } else {
                        error!(
                            "Coordinator event channel disconnected; cold-requeuing {} staged admissions",
                            pending.len()
                        );
                    }
                    // Same `waiting_count` semantics as the completion
                    // path: re-queuing into `self.waiting.push_front`
                    // is not a new submission, so the counter stays
                    // untouched here.
                    for (ticket, staged) in pending {
                        warn!(
                            "Cold-requeuing ticket {} ({} prompt tokens, {} staged blocks dropped)",
                            ticket.0,
                            staged.request.prompt_tokens.len(),
                            staged.block_ids.len()
                        );
                        self.abort_staged_admission(staged);
                    }
                    self.coordinator_unavailable = true;
                    break;
                }
                Ok(crate::kv_tier::CoordinatorEvent::StagingFailed {
                    ticket,
                    failed_block,
                    reason,
                }) => {
                    warn!(
                        "Staging failed for ticket {} on block {:?}: {}",
                        ticket.0, failed_block, reason
                    );
                    if let Some(staged) = self.stage_waiting.remove(&ticket) {
                        self.abort_staged_admission(staged);
                    }
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
                if let Some(queued) = self.normalize_request(req) {
                    self.waiting.push_back(queued);
                }
            }

            self.drain_coordinator_events();

            if self.active_len() == 0 && self.waiting.is_empty() && self.stage_waiting.is_empty() {
                if let Some(req) = self.request_rx.blocking_recv() {
                    self.waiting_count.fetch_sub(1, Ordering::Relaxed);
                    if let Some(queued) = self.normalize_request(req) {
                        self.waiting.push_back(queued);
                    }
                } else {
                    info!("Scheduler shutting down: all handles dropped");
                    break;
                }
            }

            let step_start = std::time::Instant::now();
            let step_t = std::time::Instant::now();
            // FUTURE WORK (GPU/CPU overlap): `self.step()` already overlaps
            // decode with `emit_delta`, but batched decode itself is still
            // serial because `step_decode_batch()` runs
            // `forward_decode_batch(...)` and then immediately
            // `sample_batch_greedy(...)`, whose fast path launches argmax,
            // `ctx.sync()`s, and reads tokens/logprobs back. Real overlap
            // needs a `step_launch()` / `step_readback()` split at that
            // boundary; loop reordering alone does not create it.
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

    fn normalize_request(&mut self, incoming: IncomingRequest) -> Option<QueuedRequest> {
        let prompt_tokens = match self.tokenizer.encode(&incoming.prompt) {
            Ok(tokens) if !tokens.is_empty() => tokens,
            Ok(_) => {
                error!("Empty prompt after tokenization, skipping");
                return None;
            }
            Err(e) => {
                error!("Tokenization error: {}", e);
                return None;
            }
        };
        if !self
            .request_length_contract
            .admits_prompt_len(prompt_tokens.len())
        {
            warn!(
                "Rejecting request {}: prompt length {} exceeds max_req_input_len {}",
                self.next_id,
                prompt_tokens.len(),
                self.request_length_contract.max_request_input_len(),
            );
            let _ = incoming.delta_tx.send(CompletionStreamDelta {
                text_delta: String::new(),
                finish_reason: Some(FinishReason::Length),
                usage: Some(TokenUsage {
                    prompt_tokens: prompt_tokens.len(),
                    completion_tokens: 0,
                    total_tokens: prompt_tokens.len(),
                }),
                logprob: None,
            });
            return None;
        }
        let admitted_max_tokens = self
            .request_length_contract
            .clamp_max_tokens(prompt_tokens.len(), incoming.max_tokens);
        if admitted_max_tokens == 0 {
            warn!(
                "Rejecting request {}: prompt length {} leaves no room for completion under max_req_len {}",
                self.next_id,
                prompt_tokens.len(),
                self.request_length_contract.max_request_len(),
            );
            let _ = incoming.delta_tx.send(CompletionStreamDelta {
                text_delta: String::new(),
                finish_reason: Some(FinishReason::Length),
                usage: Some(TokenUsage {
                    prompt_tokens: prompt_tokens.len(),
                    completion_tokens: 0,
                    total_tokens: prompt_tokens.len(),
                }),
                logprob: None,
            });
            return None;
        }
        if admitted_max_tokens != incoming.max_tokens {
            info!(
                "Request {}: clamping max_tokens {} -> {} to fit max_req_len {}",
                self.next_id,
                incoming.max_tokens,
                admitted_max_tokens,
                self.request_length_contract.max_request_len(),
            );
        }
        Some(QueuedRequest {
            prompt: incoming.prompt,
            prompt_tokens,
            max_tokens: admitted_max_tokens,
            sampling: incoming.sampling,
            stop: incoming.stop,
            priority: incoming.priority,
            session_id: incoming.session_id,
            delta_tx: incoming.delta_tx,
        })
    }

    pub(super) fn materialize_queued_request(
        &mut self,
        slot_idx: usize,
        queued: QueuedRequest,
        reusable_prefix_len: usize,
        reusable_cached_prompt_len: usize,
        radix_hit_len: usize,
        bytes_not_on_gpu: bool,
        no_reusable_free_slot: bool,
    ) {
        let id = self.next_id;
        self.next_id += 1;

        if reusable_prefix_len > 0 {
            info!(
                "Request {} → slot {} (prompt={} tokens, radix_hit={}, reusable_prefix={}, cached_len={}, queue={})",
                id,
                slot_idx,
                queued.prompt_tokens.len(),
                radix_hit_len,
                reusable_prefix_len,
                reusable_cached_prompt_len,
                self.waiting.len()
            );
        } else if bytes_not_on_gpu || no_reusable_free_slot {
            info!(
                "Request {} → slot {} (prompt={} tokens, radix_hit={} not reusable: bytes_not_on_gpu={}, no_free_slot={}, queue={})",
                id,
                slot_idx,
                queued.prompt_tokens.len(),
                radix_hit_len,
                bytes_not_on_gpu,
                no_reusable_free_slot,
                self.waiting.len()
            );
        } else {
            info!(
                "Request {} → slot {} (prompt={} tokens, queue={})",
                id,
                slot_idx,
                queued.prompt_tokens.len(),
                self.waiting.len()
            );
        }

        self.active[slot_idx] = Some(ActiveRequest {
            id,
            admitted_at: std::time::Instant::now(),
            first_token_at: None,
            prompt: queued.prompt,
            prompt_tokens: queued.prompt_tokens,
            generated_tokens: Vec::new(),
            priority: queued.priority,
            max_tokens: queued.max_tokens,
            sampling: queued.sampling,
            stop: queued.stop,
            session_id: queued.session_id,
            delta_tx: queued.delta_tx,
            full_decoded: String::new(),
            decoded_token_count: 0,
            sent_len: 0,
            phase: Phase::Prefilling {
                effective_tokens: Vec::new(),
                progress: 0,
            },
            cacheable_prompt_len: 0,
            prefix_byte_len: 0,
            latest_logprob: None,
            reusable_prefix_len,
            reusable_cached_prompt_len,
        });
        self.step_new(slot_idx);
        if matches!(
            self.request(slot_idx).map(|req| &req.phase),
            Some(Phase::Prefilling { .. })
        ) {
            self.queue_prefill(slot_idx);
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
mod priority_tests {
    use super::sort_waiting_queue_by_priority;
    use crate::sampler::SamplingParams;
    use crate::scheduler::RequestPriority;
    use crate::scheduler::cuda::QueuedRequest;
    use tokio::sync::mpsc;

    fn waiting_request(label: &str, priority: RequestPriority) -> QueuedRequest {
        let (delta_tx, _delta_rx) = mpsc::unbounded_channel();
        QueuedRequest {
            prompt: label.to_string(),
            prompt_tokens: vec![1, 2, 3],
            max_tokens: 8,
            sampling: SamplingParams::default(),
            stop: None,
            priority,
            session_id: None,
            delta_tx,
        }
    }

    #[test]
    fn waiting_queue_sort_prefers_higher_priority_and_keeps_fifo_within_ties() {
        let mut waiting = std::collections::VecDeque::from(vec![
            waiting_request("normal-0", RequestPriority::Normal),
            waiting_request("high-0", RequestPriority::High),
            waiting_request("low-0", RequestPriority::Low),
            waiting_request("high-1", RequestPriority::High),
            waiting_request("normal-1", RequestPriority::Normal),
        ]);

        sort_waiting_queue_by_priority(&mut waiting);

        let ordered: Vec<_> = waiting.into_iter().map(|req| req.prompt).collect();
        assert_eq!(
            ordered,
            vec!["high-0", "high-1", "normal-0", "normal-1", "low-0"]
        );
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
