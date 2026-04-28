//! Scheduler main loop: wakeup wait, run, assign_slots, cleanup.
//!
//! Split out of `runtime.rs` (pure structural refactor — no behavior change).
//! Contains: the `pub fn run` driver loop, slot assignment, slot cleanup +
//! prefix-cache eviction tick, wakeup-channel orchestration, and `free_slots`.

use super::super::budget::{PageBudget, estimated_request_target};
use super::super::{ModelForward, Phase, STATS_LOG_INTERVAL, Scheduler, error, info};
use super::helpers::{
    DeferredWaitingRequest, WaitingInsertBias, WaitingRequestHint, insert_deferred_waiting_request,
};

impl<M: ModelForward> Scheduler<M> {
    pub(super) fn wait_for_coordinator_or_request(&mut self) -> bool {
        if !self.wakeup_live {
            crossbeam_channel::select! {
                recv(self.coordinator_events) -> event => {
                    match event {
                        Ok(event) => self.handle_coordinator_event(event),
                        Err(_) => error!("Coordinator event channel disconnected"),
                    }
                }
                recv(self.emit_events) -> event => {
                    match event {
                        Ok(event) => self.handle_emit_event(event),
                        Err(err) => panic!("emit event channel disconnected: {err}"),
                    }
                }
            }
            return true;
        }

        crossbeam_channel::select! {
            recv(self.coordinator_events) -> event => {
                match event {
                    Ok(event) => self.handle_coordinator_event(event),
                    Err(_) => error!("Coordinator event channel disconnected"),
                }
                true
            }
            recv(self.emit_events) -> event => {
                match event {
                    Ok(event) => self.handle_emit_event(event),
                    Err(err) => panic!("emit event channel disconnected: {err}"),
                }
                true
            }
            recv(self.wakeup_rx) -> wakeup => {
                match wakeup {
                    Ok(()) => {
                        self.drain_request_rx();
                        self.drain_wakeup_rx();
                    }
                    Err(_) => self.handle_wakeup_disconnect(),
                }
                true
            }
        }
    }

    pub(super) fn wait_for_wakeup(&mut self) -> bool {
        if self.is_fetch_wait_bound() {
            return self.wait_for_coordinator_or_request();
        }

        if self.waiting.is_empty()
            && self.prefill_queue.is_empty()
            && !self.has_pending_gpu_work()
            && !self.emit_gate_waiting.is_empty()
            && !self.has_runnable_decode_work()
        {
            return self.wait_for_coordinator_or_request();
        }

        if self.active_len() == 0 && self.waiting.is_empty() && !self.has_pending_gpu_work() {
            if self.trigger_background_store_drain() {
                return self.wait_for_coordinator_or_request();
            }
            if !self.wakeup_live {
                info!("Scheduler shutting down: all handles dropped");
                return false;
            }
            if let Ok(()) = self.wakeup_rx.recv() {
                self.drain_request_rx();
                self.drain_wakeup_rx();
                return true;
            }
            self.handle_wakeup_disconnect();
            info!("Scheduler shutting down: all handles dropped");
            return false;
        }

        true
    }

    /// Run the scheduler loop. Blocks until all handles are dropped.
    pub fn run(mut self) {
        self.warmup_cuda_graphs();
        info!("Scheduler run loop started");
        loop {
            self.drain_request_rx();
            self.drain_coordinator_events();
            self.drain_emit_events();
            if !self.wait_for_wakeup() {
                break;
            }

            let step_start = std::time::Instant::now();
            self.assign_slots();
            let assign_us = step_start.elapsed().as_micros();

            let step_t = std::time::Instant::now();
            // `step()` keeps decode/prefill readback pending across loop turns
            // so this iteration's intake/admission work can overlap the
            // previous iteration's GPU compute. The sync points live in the
            // corresponding readback/completion calls.
            self.step();
            let step_us = step_t.elapsed().as_micros();

            let clean_t = std::time::Instant::now();
            self.cleanup();
            let clean_us = clean_t.elapsed().as_micros();
            self.metrics.set_active(self.active_len() as u64);
            self.metrics.set_waiting(self.waiting.len() as u64);
            self.metrics.set_scheduler_occupancy(
                self.running_batch.len() as u64,
                self.prefill_queue.len() as u64,
            );
            let coordinator_stats = self.coordinator_queue_stats();
            self.metrics.set_kv_coordinator(
                coordinator_stats.queue_capacity() as u64,
                coordinator_stats.fetch_queue_depth() as u64,
                coordinator_stats.fetch_waiters as u64,
                coordinator_stats.store_queue_depth() as u64,
                coordinator_stats.fetch_backpressured(),
                coordinator_stats.store_backpressured(),
                coordinator_stats.store.submitted,
                coordinator_stats.store.completed,
                coordinator_stats.store.failed,
                coordinator_stats.store.rejected,
            );
            let (fetch_wait_s, store_wait_s) = self.current_tier_wait_seconds();
            self.metrics
                .set_tier_wait_seconds(fetch_wait_s, store_wait_s);
            if self.paged_kv_pool.is_active() {
                // Both in token units so kv_util = (total-free)/total is correct.
                let total =
                    (self.paged_kv_pool.max_total_pages * self.paged_kv_pool.page_size) as u64;
                let free = self.paged_kv_pool.free_count() as u64;
                self.metrics.set_kv_gpu_blocks(free, total);
            }
            // Throttled GPU memory query — at most once per second.
            if self.stats.last_mem_query.elapsed().as_secs() >= 1 {
                self.stats.last_mem_query = std::time::Instant::now();
                if let Ok((free, total)) =
                    crate::backend::cuda::tensor::DeviceContext::gpu_memory_info()
                {
                    let active = (total - free) as u64;
                    self.stats.peak_mem_bytes = self.stats.peak_mem_bytes.max(active);
                    self.metrics
                        .set_memory_bytes(active, self.stats.peak_mem_bytes, 0);
                }
            }

            let total_us = step_start.elapsed().as_micros();
            if total_us > 50_000 {
                // Log slow iterations (>50ms)
                info!(
                    "Scheduler step: assign={}us step={}us cleanup={}us total={}us active={}",
                    assign_us,
                    step_us,
                    clean_us,
                    total_us,
                    self.active_len()
                );
            }
        }
    }

    pub(super) fn assign_slots(&mut self) {
        if self.waiting.is_empty() {
            return;
        }
        let _ = self.evict_prefix_cache_if_pressured();
        let mut available_free_slots = self.free_slots();

        let candidates = self.collect_admission_candidates(&available_free_slots);
        let mut deferred_waiting = std::collections::VecDeque::new();
        let mut admission_budget = PageBudget::from_scheduler(self, self.paged_kv_pool.is_active());
        for (slot_idx, req) in self.active.iter().enumerate() {
            let Some(req) = req.as_ref() else {
                continue;
            };
            if req.delta_tx.is_closed() || matches!(req.phase, Phase::Finished) {
                continue;
            }
            admission_budget.reserve_target(estimated_request_target(
                slot_idx,
                req.prompt_tokens.len(),
                req.max_tokens,
                req.reusable_prefix_len,
            ));
        }
        for candidate in candidates {
            let Some((slot_idx, reusable_prefix_len, reusable_cached_prompt_len)) =
                Self::choose_admission_slot(&candidate.plan, &available_free_slots)
            else {
                self.maybe_prefetch_staged_prefix(&candidate.plan);
                let hint = WaitingRequestHint::from_plan(&candidate.plan, 0);
                self.release_admission_plan(&candidate.plan);
                insert_deferred_waiting_request(
                    &mut deferred_waiting,
                    DeferredWaitingRequest {
                        incoming: candidate.incoming,
                        prompt_tokens: candidate.prompt_tokens,
                        hint,
                    },
                    WaitingInsertBias::BeforeEqual,
                );
                continue;
            };
            let reserved_prefix_tokens =
                Self::full_isl_reserved_tokens(&candidate.plan, reusable_prefix_len);
            if !Self::can_reserve_full_isl(
                &admission_budget,
                slot_idx,
                candidate.prompt_tokens.len(),
                candidate.incoming.max_tokens,
                reserved_prefix_tokens,
            ) {
                self.maybe_prefetch_staged_prefix(&candidate.plan);
                let hint = WaitingRequestHint::from_plan(&candidate.plan, reusable_prefix_len);
                self.release_admission_plan(&candidate.plan);
                insert_deferred_waiting_request(
                    &mut deferred_waiting,
                    DeferredWaitingRequest {
                        incoming: candidate.incoming,
                        prompt_tokens: candidate.prompt_tokens,
                        hint,
                    },
                    WaitingInsertBias::BeforeEqual,
                );
                continue;
            }
            if candidate.plan.attached_prefix_blocks.is_empty()
                && candidate.plan.staged_prefix_plan.is_none()
            {
                self.release_admission_plan(&candidate.plan);
            }
            admission_budget.reserve_target(estimated_request_target(
                slot_idx,
                candidate.prompt_tokens.len(),
                candidate.incoming.max_tokens,
                reserved_prefix_tokens,
            ));
            if let Some(pos) = available_free_slots
                .iter()
                .position(|&slot| slot == slot_idx)
            {
                available_free_slots.remove(pos);
            }
            self.admit_waiting_candidate(
                candidate.incoming,
                candidate.prompt_tokens,
                candidate.plan,
                slot_idx,
                reusable_prefix_len,
                reusable_cached_prompt_len,
            );
        }
        self.restore_deferred_waiting_requests(deferred_waiting);
    }

    /// Find all free slot indices.
    pub(super) fn free_slots(&self) -> Vec<usize> {
        self.active
            .iter()
            .enumerate()
            .filter_map(|(slot_idx, req)| req.is_none().then_some(slot_idx))
            .collect()
    }

    pub(super) fn cleanup(&mut self) {
        for slot_idx in 0..self.active.len() {
            if self.slot_has_pending_gpu_work(slot_idx) {
                continue;
            }
            let finished = matches!(
                self.request(slot_idx).map(|req| &req.phase),
                Some(Phase::Finished)
            );
            if finished {
                let req = self.active[slot_idx]
                    .take()
                    .expect("finished slot must hold a request");
                let gen_tokens = req.generated_tokens.len() as u64;
                self.release_attached_prefix_blocks(&req.held_prefix_blocks());
                self.clear_fetch_waiting_for_slot(slot_idx, req.id);
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

                self.stats.total_completed += 1;
                self.stats.total_generated_tokens += gen_tokens;
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

                if self
                    .stats
                    .total_completed
                    .is_multiple_of(STATS_LOG_INTERVAL)
                {
                    info!(
                        "Scheduler stats: completed={}, generated_tokens={}, active={}, waiting={}",
                        self.stats.total_completed,
                        self.stats.total_generated_tokens,
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
