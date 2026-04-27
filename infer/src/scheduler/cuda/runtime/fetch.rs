//! Fetch-completion + coordinator/event drain methods.
//!
//! Split out of `runtime.rs` (pure structural refactor — no behavior change).
//! Contains: staged-prefix promotion adopt path, fetch waiter collection,
//! coordinator/emit/wakeup event drains, and request intake normalization.

use super::super::{ModelForward, Ordering, Phase, Scheduler, error, info, warn};
use super::helpers::{FetchWaiter, WaitingInsertBias, staged_prefix_direct_host_blocks};
use crate::scheduler::types::RequestLengthContract;

impl<M: ModelForward> Scheduler<M> {
    pub(super) fn adopt_promoted_prefix(&mut self, waiter: &FetchWaiter) -> anyhow::Result<bool> {
        let slot_idx = waiter.slot_idx;
        let request_id = waiter.request_id;
        let prompt_tokens = &waiter.prompt_tokens;
        let staged_prefix = waiter.staged_prefix.clone();
        if staged_prefix.blocks.is_empty() {
            return Ok(false);
        }
        let final_plan =
            self.gpu_ready_staged_prefix_plan(request_id, prompt_tokens, &staged_prefix)?;
        let attached_prefix_blocks = final_plan.block_ids();
        if let Some(req) = self.request_mut(slot_idx) {
            if req.id != request_id {
                return Ok(false);
            }
            if let Some(plan) = req.staged_prefix.as_mut() {
                plan.mark_ready();
                plan.mark_consumed();
            }
            req.staged_prefix = None;
            req.attached_prefix_blocks = attached_prefix_blocks;
            req.reusable_prefix_len = staged_prefix.matched_len;
            req.reusable_cached_prompt_len = staged_prefix.matched_len;
            req.phase = Phase::Prefilling {
                effective_tokens: Vec::new(),
                progress: 0,
            };
        }
        Ok(true)
    }

    pub(super) fn collect_fetch_waiters(&mut self, waiters: Vec<(usize, u64)>) -> Vec<FetchWaiter> {
        let mut ready = Vec::new();
        for (slot_idx, request_id) in waiters {
            let Some(req) = self.request(slot_idx) else {
                continue;
            };
            if req.id != request_id {
                continue;
            }
            if req.delta_tx.is_closed() {
                self.finish_slot(slot_idx);
                continue;
            }
            if !matches!(req.phase, Phase::WaitingFetch) {
                continue;
            }
            let Some(staged_prefix) = req.staged_prefix.clone() else {
                continue;
            };
            ready.push(FetchWaiter {
                slot_idx,
                request_id,
                prompt_tokens: req.prompt_tokens.clone(),
                session_id: req.session_id.clone(),
                staged_prefix,
            });
        }
        ready
    }

    pub(super) fn complete_ready_fetch_waiters(
        &mut self,
        ready_waiters: Vec<FetchWaiter>,
        blocks: &[crate::kv_tier::FetchedBlock],
        readmit_started_at: Option<std::time::Instant>,
    ) {
        if ready_waiters.is_empty() {
            self.release_unclaimed_fetch_regions(blocks);
            return;
        }
        for waiter in &ready_waiters {
            self.prefix_cache.release(&waiter.staged_prefix.block_ids());
        }
        if let Err(err) = self.promote_fetched_prefix(&ready_waiters[0], blocks) {
            warn!(
                "Request {}: staged prefix fetch failed, falling back to cold prefill: {}",
                ready_waiters[0].request_id, err
            );
            for waiter in ready_waiters {
                self.fallback_to_cold_prefill_without_release(waiter.slot_idx);
            }
            self.release_unclaimed_fetch_regions(blocks);
            return;
        }
        let (host_blocks, disk_blocks, remote_blocks) =
            ready_waiters[0].staged_prefix.source_counts();
        if let Some(started_at) = readmit_started_at {
            info!(
                "Request {}: staged prefix ready in {:.1}ms src=h:{}/d:{}/r:{} waiters={}",
                ready_waiters[0].request_id,
                started_at.elapsed().as_secs_f64() * 1000.0,
                host_blocks,
                disk_blocks,
                remote_blocks,
                ready_waiters.len()
            );
        }
        for waiter in ready_waiters {
            match self.adopt_promoted_prefix(&waiter) {
                Ok(true) => {
                    self.step_new(waiter.slot_idx);
                    if matches!(
                        self.request(waiter.slot_idx).map(|req| &req.phase),
                        Some(Phase::Prefilling { .. })
                    ) {
                        self.queue_prefill(waiter.slot_idx);
                    }
                }
                Ok(false) => {}
                Err(err) => {
                    warn!(
                        "Request {}: staged prefix adopt failed, falling back to cold prefill: {}",
                        waiter.request_id, err
                    );
                    self.fallback_to_cold_prefill_without_release(waiter.slot_idx);
                }
            }
        }
        self.release_unclaimed_fetch_regions(blocks);
    }

    pub(super) fn try_complete_direct_host_staged_prefix(&mut self, slot_idx: usize) -> bool {
        let Some(req) = self.request(slot_idx) else {
            return false;
        };
        let Some(staged_prefix) = req.staged_prefix.as_ref() else {
            return false;
        };
        let Some(fetched_blocks) = staged_prefix_direct_host_blocks(staged_prefix) else {
            return false;
        };
        let waiter = FetchWaiter {
            slot_idx,
            request_id: req.id,
            prompt_tokens: req.prompt_tokens.clone(),
            session_id: req.session_id.clone(),
            staged_prefix: staged_prefix.clone(),
        };
        let start = std::time::Instant::now();
        self.complete_ready_fetch_waiters(vec![waiter], &fetched_blocks, Some(start));
        true
    }

    pub(super) fn handle_coordinator_event(&mut self, event: crate::kv_tier::CoordinatorEvent) {
        // The runtime consumes coordinator results in one place so the request
        // state machine stays linear:
        // - `Store*` mutates prefix-cache store state and releases T1 regions
        // - `FetchCompleted` promotes staged bytes into T0, then re-enters the
        //   normal prefill path
        // - `FetchFailed` always falls back to cold prefill
        match event {
            crate::kv_tier::CoordinatorEvent::FetchQueued { .. }
            | crate::kv_tier::CoordinatorEvent::PlanQueued { .. }
            | crate::kv_tier::CoordinatorEvent::PlanCompleted { .. }
            | crate::kv_tier::CoordinatorEvent::PlanFailed { .. } => {
                // The scheduler does not act on plan-* events: planning is
                // currently a coordinator-internal concern (the M3
                // future-orchestrator integration was scoped down). Listen-only
                // so the unified event channel stays exhaustive.
            }
            crate::kv_tier::CoordinatorEvent::StoreQueued { ticket, .. } => {
                if let Some(waiters) = self.store_waiting.get(&ticket) {
                    for (block_id, _) in waiters {
                        let _ = self.prefix_cache.mark_block_storing(*block_id);
                    }
                }
            }
            crate::kv_tier::CoordinatorEvent::StoreCompleted { ticket, locations } => {
                self.store_ticket_started_at.remove(&ticket);
                if let Some(key) = self.store_ticket_keys.remove(&ticket) {
                    self.store_dedupe.remove(&key);
                }
                if let Some(waiters) = self.store_waiting.remove(&ticket) {
                    let canonical_location =
                        locations.first().map(|(_, location)| location.clone());
                    for (block_id, region) in waiters {
                        if let Some(location) = canonical_location.clone() {
                            let _ = self
                                .prefix_cache
                                .mark_block_stored(block_id, Some(location));
                        } else {
                            let _ = self.prefix_cache.mark_block_store_failed(block_id);
                        }
                        self.release_host_region(region);
                    }
                }
            }
            crate::kv_tier::CoordinatorEvent::StoreFailed {
                ticket,
                failed_block,
                class,
                reason,
            } => {
                // Typed class lets us downgrade cooperative cancellation to
                // info — only true store failures need the warn level.
                if matches!(class, crate::kv_tier::FailureClass::Cancelled) {
                    info!(
                        "Store cancelled for ticket {} on block {:?}: {}",
                        ticket.0, failed_block, reason
                    );
                } else {
                    warn!(
                        "Store failed for ticket {} on block {:?}: {}",
                        ticket.0, failed_block, reason
                    );
                }
                self.store_ticket_started_at.remove(&ticket);
                if let Some(key) = self.store_ticket_keys.remove(&ticket) {
                    self.store_dedupe.remove(&key);
                }
                if let Some(waiters) = self.store_waiting.remove(&ticket) {
                    for (block_id, region) in waiters {
                        let _ = self.prefix_cache.mark_block_store_failed(block_id);
                        self.release_host_region(region);
                    }
                } else {
                    let _ = self.prefix_cache.mark_block_store_failed(failed_block);
                }
            }
            crate::kv_tier::CoordinatorEvent::FetchCompleted { ticket, blocks } => {
                let fetch_started_at = self.fetch_ticket_started_at.remove(&ticket);
                let prefetch_state = self.prefetch_fetching.remove(&ticket);
                let waiters = self.fetch_waiting.remove(&ticket).unwrap_or_default();
                if let Some(key) = self.fetch_ticket_keys.remove(&ticket) {
                    self.fetch_dedupe.remove(&key);
                }
                if waiters.is_empty() {
                    if let Some(prefetch_state) = prefetch_state {
                        let materialized = self.materialize_prefetched_host_blocks(&blocks);
                        info!(
                            "Prefetch {} ready: {:.1}ms materialized={} src=h:{}/d:{}/r:{}",
                            ticket.0,
                            fetch_started_at
                                .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                                .unwrap_or_default(),
                            materialized,
                            prefetch_state.host_blocks,
                            prefetch_state.disk_blocks,
                            prefetch_state.remote_blocks
                        );
                    } else {
                        self.release_unclaimed_fetch_regions(&blocks);
                    }
                    return;
                }
                let ready_waiters = self.collect_fetch_waiters(waiters);
                self.complete_ready_fetch_waiters(ready_waiters, &blocks, fetch_started_at);
            }
            crate::kv_tier::CoordinatorEvent::FetchFailed {
                ticket,
                failed_block,
                class,
                reason,
            } => {
                if matches!(class, crate::kv_tier::FailureClass::Cancelled) {
                    info!(
                        "Fetch cancelled for ticket {} on block {:?}: {}",
                        ticket.0, failed_block, reason
                    );
                } else {
                    warn!(
                        "Fetch failed for ticket {} on block {:?}: {}",
                        ticket.0, failed_block, reason
                    );
                }
                self.fetch_ticket_started_at.remove(&ticket);
                self.prefetch_fetching.remove(&ticket);
                let waiters = self.fetch_waiting.remove(&ticket).unwrap_or_default();
                if let Some(key) = self.fetch_ticket_keys.remove(&ticket) {
                    self.fetch_dedupe.remove(&key);
                }
                for (slot_idx, request_id) in waiters {
                    if self
                        .request(slot_idx)
                        .is_some_and(|req| req.id == request_id)
                    {
                        self.fallback_to_cold_prefill(slot_idx);
                    }
                }
            }
        }
    }

    pub(super) fn drain_coordinator_events(&mut self) {
        loop {
            match self.coordinator_events.try_recv() {
                Ok(event) => self.handle_coordinator_event(event),
                Err(crossbeam_channel::TryRecvError::Empty) => break,
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    error!("Coordinator event channel disconnected");
                    break;
                }
            }
        }
    }

    pub(super) fn handle_emit_event(&mut self, event: crate::scheduler::cuda::core::EmitEvent) {
        match event {
            crate::scheduler::cuda::core::EmitEvent::GateReady {
                request_id,
                finished,
            } => {
                let Some(slot_idx) = self.emit_gate_waiting.remove(&request_id) else {
                    return;
                };
                if self
                    .request(slot_idx)
                    .is_none_or(|req| req.id != request_id)
                {
                    return;
                }
                if finished {
                    if let Some(req) = self.request_mut(slot_idx) {
                        req.pending_finish_reason = None;
                        req.phase = Phase::Finished;
                    }
                    self.finish_slot(slot_idx);
                } else if let Some(reason) = self
                    .request_mut(slot_idx)
                    .and_then(|req| req.pending_finish_reason.take())
                {
                    self.finish_request(slot_idx, reason);
                }
            }
        }
    }

    pub(super) fn drain_emit_events(&mut self) {
        loop {
            match self.emit_events.try_recv() {
                Ok(event) => self.handle_emit_event(event),
                Err(crossbeam_channel::TryRecvError::Empty) => break,
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    panic!("emit event channel disconnected")
                }
            }
        }
    }

    pub(super) fn drain_request_rx(&mut self) {
        let length_contract = RequestLengthContract::derive(
            self.paged_kv_pool.max_total_tokens,
            self.effective_max_seq_len,
        );
        while let Ok(req) = self.request_rx.try_recv() {
            self.waiting_count.fetch_sub(1, Ordering::Relaxed);
            let Some((mut incoming, prompt_tokens)) =
                self.normalize_waiting_request(req, length_contract)
            else {
                continue;
            };
            incoming.prompt_tokens = Some(prompt_tokens);
            self.enqueue_waiting_request(incoming, WaitingInsertBias::AfterEqual);
        }
    }

    pub(super) fn drain_wakeup_rx(&mut self) {
        while self.wakeup_rx.try_recv().is_ok() {}
    }

    pub(super) fn handle_wakeup_disconnect(&mut self) {
        self.wakeup_live = false;
        self.drain_request_rx();
    }
}
