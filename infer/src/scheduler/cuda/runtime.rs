use super::{
    ActiveRequest, CompletionStreamDelta, FinishReason, ModelForward, Ordering, Phase,
    STATS_LOG_INTERVAL, Scheduler, TokenUsage, error, info, warn,
};
use crate::kv_tier::{ReadmissionSource, RequestChunkState};
use crate::prefix_cache::BlockMetadataUpdate;
use crate::scheduler::types::RequestLengthContract;

#[derive(Clone)]
struct FetchWaiter {
    slot_idx: usize,
    request_id: u64,
    prompt_tokens: Vec<u32>,
    session_id: Option<crate::types::SessionId>,
    staged_prefix: crate::kv_tier::ReadmissionPlan,
}

#[derive(Clone)]
struct PrefixAdmissionPlan {
    radix_blocks: Vec<crate::prefix_cache::BlockId>,
    lookup: crate::kv_tier::LookupOutcome,
    reusable: Option<(usize, usize, usize)>,
    direct_gpu_attach: bool,
    gpu_ready_prefix_blocks: Vec<crate::prefix_cache::BlockId>,
    attached_prefix_blocks: Vec<crate::prefix_cache::BlockId>,
    staged_prefix_plan: Option<crate::kv_tier::ReadmissionPlan>,
}

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
    waiting.sort_by(|lhs, rhs| rhs.priority.cmp(&lhs.priority));
}

fn finish_rejected_request(
    delta_tx: &tokio::sync::mpsc::UnboundedSender<CompletionStreamDelta>,
    reason: FinishReason,
    prompt_tokens: usize,
) {
    let _ = delta_tx.send(CompletionStreamDelta {
        text_delta: String::new(),
        finish_reason: Some(reason),
        usage: Some(TokenUsage {
            prompt_tokens,
            completion_tokens: 0,
            total_tokens: prompt_tokens,
        }),
        logprob: None,
    });
}

impl<M: ModelForward> Scheduler<M> {
    /// Canonical admission decision for one incoming prompt.
    ///
    /// Order matters and is intentionally centralized here so the runtime and
    /// docs stay in sync:
    ///
    /// 1. `lookup_or_stage()` classifies each matched radix block as
    ///    `ReadyOnGpu`, `StagingFromHost`, `StagingFromDisk`, or `Miss`.
    /// 2. If every matched block is already runnable on T0 and the model uses
    ///    the paged pool, prefer direct GPU page attachment.
    /// 3. Otherwise, if the model uses the paged pool and some matched blocks
    ///    live below T0, build a staged readmission plan.
    /// 4. Otherwise, fall back to the older same-slot contiguous reuse path if
    ///    a free slot still materializes the radix-owned prefix.
    /// 5. Any staged / non-runnable hit that cannot progress immediately
    ///    degrades to cold prefill rather than leaving a second parked path.
    fn build_prefix_admission_plan(
        &mut self,
        prompt_tokens: &[u32],
        free_slots: &[usize],
    ) -> PrefixAdmissionPlan {
        let lookup = self
            .prefix_cache
            .lookup_or_stage(prompt_tokens, crate::kv_tier::LookupHeuristics::default());
        let radix_blocks: Vec<_> = lookup
            .blocks
            .iter()
            .filter_map(|block| block.block_id)
            .collect();
        let ready_on_gpu = lookup_blocks_ready_on_gpu(&lookup.blocks);
        let gpu_ready_prefix_blocks: Vec<_> = lookup
            .blocks
            .iter()
            .take_while(|block| matches!(block.hit_kind, crate::kv_tier::HitKind::ReadyOnGpu))
            .filter_map(|block| block.block_id)
            .collect();
        let staged_prefix_plan = if self.model.prefill_uses_paged_pool()
            && !lookup.recompute_advised
            && !ready_on_gpu
            && lookup.blocks.iter().any(|block| {
                matches!(
                    block.hit_kind,
                    crate::kv_tier::HitKind::StagingFromHost
                        | crate::kv_tier::HitKind::StagingFromDisk
                )
            }) {
            self.build_staged_prefix_plan(&lookup)
        } else {
            None
        };
        let direct_gpu_attach = self.model.prefill_uses_paged_pool()
            && !lookup.recompute_advised
            && !gpu_ready_prefix_blocks.is_empty()
            && ready_on_gpu
            && staged_prefix_plan.is_none();
        let reusable_gpu_prefix = if direct_gpu_attach || staged_prefix_plan.is_some() {
            None
        } else if ready_on_gpu && !lookup.recompute_advised {
            best_reusable_slot_for_radix_hit(
                &gpu_ready_prefix_blocks,
                free_slots,
                &self.block_owner_slots,
                &self.slot_materialized_prompt_lens,
                self.prefix_cache.block_size(),
            )
        } else {
            None
        };

        PrefixAdmissionPlan {
            radix_blocks,
            lookup,
            reusable: reusable_gpu_prefix,
            direct_gpu_attach,
            gpu_ready_prefix_blocks: gpu_ready_prefix_blocks.clone(),
            attached_prefix_blocks: if direct_gpu_attach {
                gpu_ready_prefix_blocks
            } else {
                Vec::new()
            },
            staged_prefix_plan,
        }
    }

    fn fallback_to_cold_prefill(&mut self, slot_idx: usize) {
        self.fallback_to_cold_prefill_inner(slot_idx, true);
    }

    fn fallback_to_cold_prefill_without_release(&mut self, slot_idx: usize) {
        self.fallback_to_cold_prefill_inner(slot_idx, false);
    }

    fn fallback_to_cold_prefill_inner(&mut self, slot_idx: usize, release_held_blocks: bool) {
        let held_blocks = self
            .request(slot_idx)
            .and_then(|req| req.staged_prefix.as_ref().map(|plan| plan.block_ids()))
            .unwrap_or_default();
        if release_held_blocks && !held_blocks.is_empty() {
            self.prefix_cache.release(&held_blocks);
        }
        if let Some(req) = self.request_mut(slot_idx) {
            req.staged_prefix = None;
            req.reusable_prefix_len = 0;
            req.reusable_cached_prompt_len = 0;
            req.attached_prefix_blocks.clear();
            req.phase = Phase::Prefilling {
                materialized_prefix_len: 0,
                effective_tokens: Vec::new(),
                progress: 0,
            };
        }
        self.step_new(slot_idx);
        if matches!(
            self.request(slot_idx).map(|req| &req.phase),
            Some(Phase::Prefilling { .. })
        ) {
            self.queue_prefill(slot_idx);
        }
    }

    fn release_unclaimed_fetch_regions(&self, blocks: &[crate::kv_tier::FetchedBlock]) {
        for block in blocks {
            if block.release_after_promote {
                self.release_host_region(block.host_region);
            }
        }
    }

    fn promote_fetched_prefix(
        &mut self,
        waiter: &FetchWaiter,
        fetched_blocks: &[crate::kv_tier::FetchedBlock],
    ) -> anyhow::Result<()> {
        let slot_idx = waiter.slot_idx;
        let request_id = waiter.request_id;
        let prompt_tokens = &waiter.prompt_tokens;
        let session_id = waiter.session_id.clone();
        let staged_prefix = waiter.staged_prefix.clone();
        if staged_prefix.blocks.is_empty() {
            return Ok(());
        }
        let fetched_by_id: std::collections::HashMap<
            crate::prefix_cache::BlockId,
            &crate::kv_tier::FetchedBlock,
        > = fetched_blocks
            .iter()
            .map(|block| (block.block_id, block))
            .collect();

        let pages_per_block = self
            .prefix_cache
            .block_size()
            .div_ceil(self.paged_kv_pool.page_size)
            .max(1);
        let mut promoted_pages: Vec<(
            crate::prefix_cache::BlockId,
            crate::prefix_cache::BlockId,
            Vec<u32>,
        )> = Vec::new();
        let mut final_block_ids = Vec::with_capacity(staged_prefix.blocks.len());
        let mut fingerprints = Vec::with_capacity(staged_prefix.blocks.len());

        for block in &staged_prefix.blocks {
            fingerprints.push(block.fingerprint);
            match &block.source {
                None => final_block_ids.push(block.block_id),
                Some(ReadmissionSource::HostPinned { .. })
                | Some(ReadmissionSource::Disk { .. })
                | Some(ReadmissionSource::Remote { .. }) => {
                    let fetched = fetched_by_id.get(&block.block_id).copied().ok_or_else(|| {
                        anyhow::anyhow!(
                            "missing fetched host staging for staged prefix block {:?}",
                            block.block_id
                        )
                    })?;
                    let payload = self.host_pinned_pool.read_region(fetched.host_region)?;
                    let pages = self.paged_kv_pool.alloc_detached_pages(pages_per_block)?;
                    if let Err(err) = self.paged_kv_pool.copy_pages_from_host(
                        self.model.device_context(),
                        &pages,
                        &payload,
                    ) {
                        let _ = self.paged_kv_pool.release_pages(&pages);
                        return Err(err);
                    }
                    let new_block_id = crate::prefix_cache::BlockId(
                        *pages
                            .first()
                            .expect("detached promoted block must allocate pages"),
                    );
                    promoted_pages.push((block.block_id, new_block_id, pages));
                    final_block_ids.push(new_block_id);
                }
            }
        }

        if staged_prefix.matched_len == 0
            || staged_prefix.matched_len > prompt_tokens.len()
            || staged_prefix.matched_len
                != staged_prefix.blocks.len() * self.prefix_cache.block_size()
        {
            for (_, _, pages) in promoted_pages {
                let _ = self.paged_kv_pool.release_pages(&pages);
            }
            return Err(anyhow::anyhow!(
                "invalid staged prefix shape for request {} (matched={} blocks={} prompt={})",
                request_id,
                staged_prefix.matched_len,
                staged_prefix.blocks.len(),
                prompt_tokens.len()
            ));
        }

        let prefix_tokens = &prompt_tokens[..staged_prefix.matched_len];
        let inserted = self.prefix_cache.insert_with_fingerprints(
            prefix_tokens,
            &final_block_ids,
            &fingerprints,
        );
        if inserted != prefix_tokens.len() {
            warn!(
                "Request {}: staged prefix remap inserted {} / {} prefix tokens",
                request_id,
                inserted,
                prefix_tokens.len()
            );
            for (_, _, pages) in promoted_pages {
                let _ = self.paged_kv_pool.release_pages(&pages);
            }
            return Err(anyhow::anyhow!(
                "staged prefix remap inserted {} / {} tokens for request {}",
                inserted,
                prefix_tokens.len(),
                request_id
            ));
        }

        let block_byte_len = self
            .paged_kv_pool
            .storage_bytes_for_tokens(self.prefix_cache.block_size())
            .min(u32::MAX as usize) as u32;
        let keepalive_deadline = session_id.as_ref().map(|_| {
            self.prefix_cache
                .logical_clock()
                .saturating_add(self.config.prefix_cache_keepalive_ticks)
        });
        for (old_block_id, new_block_id, pages) in promoted_pages {
            self.block_owner_slots.remove(&old_block_id);
            self.block_to_pages.insert(new_block_id, pages);
            let _ = self.prefix_cache.update_block_metadata(
                new_block_id,
                BlockMetadataUpdate {
                    location: Some(crate::kv_tier::BlockLocation::Gpu {
                        slot: slot_idx as u32,
                    }),
                    byte_len: Some(block_byte_len),
                    session_id: Some(session_id.clone()),
                    soft_pin_until: Some(keepalive_deadline),
                    ..BlockMetadataUpdate::default()
                },
            );
        }

        info!(
            "Request {}: staged prefix ready, promoted {}/{} tokens into T0",
            request_id,
            staged_prefix.matched_len,
            prompt_tokens.len()
        );
        Ok(())
    }

    fn adopt_promoted_prefix(&mut self, waiter: &FetchWaiter) -> anyhow::Result<bool> {
        let slot_idx = waiter.slot_idx;
        let request_id = waiter.request_id;
        let prompt_tokens = &waiter.prompt_tokens;
        let staged_prefix = waiter.staged_prefix.clone();
        if staged_prefix.blocks.is_empty() {
            return Ok(false);
        }

        let final_lookup = self.prefix_cache.lookup_or_stage(
            &prompt_tokens[..staged_prefix.matched_len],
            crate::kv_tier::LookupHeuristics::default(),
        );
        if final_lookup.matched_len != staged_prefix.matched_len
            || !lookup_blocks_ready_on_gpu(&final_lookup.blocks)
        {
            return Err(anyhow::anyhow!(
                "staged prefix promotion did not become GPU-runnable (matched={} expected={})",
                final_lookup.matched_len,
                staged_prefix.matched_len
            ));
        }

        let attached_prefix_blocks: Vec<_> = final_lookup
            .blocks
            .iter()
            .filter_map(|block| block.block_id)
            .collect();
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
                materialized_prefix_len: 0,
                effective_tokens: Vec::new(),
                progress: 0,
            };
        }
        Ok(true)
    }

    fn collect_fetch_waiters(&mut self, waiters: Vec<(usize, u64)>) -> Vec<FetchWaiter> {
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

    fn drain_coordinator_events(&mut self) {
        // The runtime consumes coordinator results in one place so the request
        // state machine stays linear:
        // - `Store*` mutates prefix-cache store state and releases T1 regions
        // - `FetchCompleted` promotes staged bytes into T0, then re-enters the
        //   normal prefill path
        // - `FetchFailed` always falls back to cold prefill
        loop {
            match self.coordinator_events.try_recv() {
                Ok(crate::kv_tier::CoordinatorEvent::CommandQueued(_))
                | Ok(crate::kv_tier::CoordinatorEvent::SpillQueued { .. })
                | Ok(crate::kv_tier::CoordinatorEvent::FetchQueued { .. }) => {}
                Ok(crate::kv_tier::CoordinatorEvent::StoreQueued { ticket, .. }) => {
                    if let Some((block_id, _)) = self.store_waiting.get(&ticket).copied() {
                        let _ = self.prefix_cache.mark_block_storing(block_id);
                    }
                }
                Err(crossbeam_channel::TryRecvError::Empty) => break,
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    error!("Coordinator event channel disconnected");
                    break;
                }
                Ok(crate::kv_tier::CoordinatorEvent::SpillCompleted { ticket, locations }) => {
                    if let Some((block_id, region)) = self.store_waiting.remove(&ticket) {
                        for (completed_block, location) in locations {
                            if completed_block != block_id {
                                continue;
                            }
                            let _ = self.prefix_cache.mark_block_stored(
                                block_id,
                                Some(crate::kv_tier::BlockLocation::Disk {
                                    fingerprint: location.fingerprint,
                                    payload_len: location.payload_len,
                                }),
                            );
                            self.release_host_region(region);
                        }
                    }
                }
                Ok(crate::kv_tier::CoordinatorEvent::StoreCompleted { ticket, locations }) => {
                    if let Some((block_id, region)) = self.store_waiting.remove(&ticket) {
                        for (completed_block, location) in locations {
                            if completed_block != block_id {
                                continue;
                            }
                            let _ = self
                                .prefix_cache
                                .mark_block_stored(block_id, Some(location));
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
                    self.store_waiting.remove(&ticket);
                    let _ = self.prefix_cache.mark_block_store_failed(failed_block);
                }
                Ok(crate::kv_tier::CoordinatorEvent::StoreFailed {
                    ticket,
                    failed_block,
                    reason,
                }) => {
                    warn!(
                        "Store failed for ticket {} on block {:?}: {}",
                        ticket.0, failed_block, reason
                    );
                    self.store_waiting.remove(&ticket);
                    let _ = self.prefix_cache.mark_block_store_failed(failed_block);
                }
                Ok(crate::kv_tier::CoordinatorEvent::FetchCompleted { ticket, blocks }) => {
                    let Some(waiters) = self.fetch_waiting.remove(&ticket) else {
                        self.release_unclaimed_fetch_regions(&blocks);
                        continue;
                    };
                    if let Some(key) = self.fetch_ticket_keys.remove(&ticket) {
                        self.fetch_dedupe.remove(&key);
                    }
                    let ready_waiters = self.collect_fetch_waiters(waiters);
                    if ready_waiters.is_empty() {
                        self.release_unclaimed_fetch_regions(&blocks);
                        continue;
                    }
                    for waiter in &ready_waiters {
                        self.prefix_cache.release(&waiter.staged_prefix.block_ids());
                    }
                    if let Err(err) = self.promote_fetched_prefix(&ready_waiters[0], &blocks) {
                        warn!(
                            "Request {}: staged prefix fetch failed, falling back to cold prefill: {}",
                            ready_waiters[0].request_id, err
                        );
                        for waiter in ready_waiters {
                            self.fallback_to_cold_prefill_without_release(waiter.slot_idx);
                        }
                        self.release_unclaimed_fetch_regions(&blocks);
                        continue;
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
                    self.release_unclaimed_fetch_regions(&blocks);
                }
                Ok(crate::kv_tier::CoordinatorEvent::FetchFailed {
                    ticket,
                    failed_block,
                    reason,
                }) => {
                    warn!(
                        "Fetch failed for ticket {} on block {:?}: {}",
                        ticket.0, failed_block, reason
                    );
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
            self.assign_slots();
            let assign_us = step_start.elapsed().as_micros();

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
            let coordinator_stats = self.coordinator_queue_stats();
            self.metrics.set_kv_coordinator(
                coordinator_stats.queue_capacity() as u64,
                coordinator_stats.fetch_queue_depth() as u64,
                coordinator_stats.fetch_waiters as u64,
                coordinator_stats.store_queue_depth() as u64,
                coordinator_stats.fetch_backpressured(),
                coordinator_stats.store_backpressured(),
            );
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

    fn assign_slots(&mut self) {
        sort_waiting_queue_by_priority(&mut self.waiting);
        let length_contract = RequestLengthContract::derive(
            self.paged_kv_pool.max_total_tokens,
            self.effective_max_seq_len,
        );
        while !self.waiting.is_empty() {
            let _ = self.evict_prefix_cache_if_pressured();
            let free_slots = self.free_slots();
            if free_slots.is_empty() {
                break;
            }

            let mut incoming = self.waiting.pop_front().expect("checked non-empty above");
            if incoming.delta_tx.is_closed() {
                continue;
            }
            let prompt_tokens = match self.tokenizer.encode(&incoming.prompt) {
                Ok(tokens) if !tokens.is_empty() => tokens,
                Ok(_) => {
                    error!("Empty prompt after tokenization, skipping");
                    finish_rejected_request(&incoming.delta_tx, FinishReason::Length, 0);
                    continue;
                }
                Err(e) => {
                    error!("Tokenization error: {}", e);
                    finish_rejected_request(&incoming.delta_tx, FinishReason::Length, 0);
                    continue;
                }
            };
            if !length_contract.admits_prompt_len(prompt_tokens.len()) {
                warn!(
                    "Rejecting prompt with {} tokens: scheduler max_input={} max_request={}",
                    prompt_tokens.len(),
                    length_contract.max_request_input_len(),
                    length_contract.max_request_len(),
                );
                finish_rejected_request(
                    &incoming.delta_tx,
                    FinishReason::Length,
                    prompt_tokens.len(),
                );
                continue;
            }
            incoming.max_tokens =
                length_contract.clamp_max_tokens(prompt_tokens.len(), incoming.max_tokens);

            let plan = self.build_prefix_admission_plan(&prompt_tokens, &free_slots);
            let ready_on_gpu = lookup_blocks_ready_on_gpu(&plan.lookup.blocks);
            let radix_hit_len = if ready_on_gpu && !plan.lookup.recompute_advised {
                plan.lookup.matched_len
            } else {
                0
            };
            let (slot_idx, reusable_prefix_len, reusable_cached_prompt_len) =
                plan.reusable.unwrap_or((free_slots[0], 0, 0));
            if plan.attached_prefix_blocks.is_empty() && plan.staged_prefix_plan.is_none() {
                self.prefix_cache.release(&plan.radix_blocks);
            }

            let id = self.next_id;
            self.next_id += 1;

            if let Some(staged) = plan.staged_prefix_plan.as_ref() {
                info!(
                    "Request {} → slot {} (prompt={} tokens, staged_prefix={}, queue={})",
                    id,
                    slot_idx,
                    prompt_tokens.len(),
                    staged.matched_len,
                    self.waiting.len()
                );
            } else if plan.direct_gpu_attach {
                info!(
                    "Request {} → slot {} (prompt={} tokens, radix_gpu_attach={}, queue={})",
                    id,
                    slot_idx,
                    prompt_tokens.len(),
                    plan.lookup.matched_len,
                    self.waiting.len()
                );
            } else if reusable_prefix_len > 0 {
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
                    plan.lookup.matched_len > 0 && (!ready_on_gpu || plan.lookup.recompute_advised);
                let no_reusable_free_slot = plan.lookup.matched_len > 0
                    && !plan.gpu_ready_prefix_blocks.is_empty()
                    && plan.reusable.is_none();
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
                        plan.lookup.matched_len,
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

            self.active[slot_idx] = Some(ActiveRequest {
                id,
                admitted_at: std::time::Instant::now(),
                first_token_at: None,
                prompt: incoming.prompt,
                prompt_tokens,
                generated_tokens: Vec::new(),
                priority: incoming.priority,
                max_tokens: incoming.max_tokens,
                sampling: incoming.sampling,
                stop: incoming.stop,
                session_id: incoming.session_id,
                delta_tx: incoming.delta_tx,
                full_decoded: String::new(),
                decoded_token_count: 0,
                sent_len: 0,
                phase: if plan.staged_prefix_plan.is_some() {
                    // WaitingFetch is the only parked state left in the CUDA
                    // scheduler. It exists solely for staged KV readmission and
                    // always resolves to either:
                    // 1. `FetchCompleted` -> promote into T0 -> Prefilling
                    // 2. `FetchFailed` / queue saturation / invalid plan ->
                    //    cold Prefilling fallback
                    Phase::WaitingFetch
                } else {
                    Phase::Prefilling {
                        materialized_prefix_len: 0,
                        effective_tokens: Vec::new(),
                        progress: 0,
                    }
                },
                cacheable_prompt_len: 0,
                prefix_byte_len: 0,
                latest_logprob: None,
                reusable_prefix_len: if plan.direct_gpu_attach {
                    plan.lookup.matched_len
                } else {
                    reusable_prefix_len
                },
                reusable_cached_prompt_len,
                attached_prefix_blocks: plan.attached_prefix_blocks.clone(),
                staged_prefix: plan.staged_prefix_plan.clone(),
            });
            if incoming.max_tokens == 0 {
                self.finish_request(slot_idx, crate::server_engine::FinishReason::Length);
                continue;
            }
            if matches!(
                self.request(slot_idx).map(|req| &req.phase),
                Some(Phase::WaitingFetch)
            ) {
                let Some(staged_prefix) = self
                    .request(slot_idx)
                    .and_then(|req| req.staged_prefix.as_ref())
                    .cloned()
                else {
                    self.fallback_to_cold_prefill(slot_idx);
                    continue;
                };

                match staged_prefix.fetch_key() {
                    None => {
                        warn!(
                            "Request {}: invalid staged prefix plan, falling back to cold prefill",
                            id
                        );
                        self.fallback_to_cold_prefill(slot_idx);
                    }
                    Some(fetch_key) => {
                        if let Some(ticket) = self.fetch_dedupe.get(&fetch_key).copied() {
                            if let Some(req) = self.request_mut(slot_idx) {
                                if let Some(plan) = req.staged_prefix.as_mut() {
                                    plan.mark_fetching();
                                }
                            }
                            self.fetch_waiting
                                .entry(ticket)
                                .or_default()
                                .push((slot_idx, id));
                        } else if self.coordinator_queue_stats().fetch_backpressured() {
                            let coordinator_stats = self.coordinator_queue_stats();
                            warn!(
                                "Request {}: staged fetch backpressured (fetch_q={}/{} waiters={}), falling back to cold prefill",
                                id,
                                coordinator_stats.fetch_queue_depth(),
                                coordinator_stats.queue_capacity(),
                                coordinator_stats.fetch_waiters,
                            );
                            self.fallback_to_cold_prefill(slot_idx);
                        } else if let Some(fetch_requests) =
                            staged_prefix.fetch_requests(&self.host_pinned_pool)
                        {
                            if let Some(req) = self.request_mut(slot_idx) {
                                if let Some(plan) = req.staged_prefix.as_mut() {
                                    debug_assert_eq!(plan.state, RequestChunkState::Planned);
                                    plan.mark_fetching();
                                }
                            }
                            if let Some(ticket) =
                                self.coordinator_handle.submit_fetch(fetch_requests)
                            {
                                self.fetch_dedupe.insert(fetch_key.clone(), ticket);
                                self.fetch_ticket_keys.insert(ticket, fetch_key);
                                self.fetch_waiting.insert(ticket, vec![(slot_idx, id)]);
                            } else {
                                let coordinator_stats = self.coordinator_queue_stats();
                                warn!(
                                    "Request {}: fetch queue full after submit attempt (fetch_q={}/{} waiters={}), falling back to cold prefill",
                                    id,
                                    coordinator_stats.fetch_queue_depth(),
                                    coordinator_stats.queue_capacity(),
                                    coordinator_stats.fetch_waiters,
                                );
                                self.fallback_to_cold_prefill(slot_idx);
                            }
                        } else {
                            warn!(
                                "Request {}: invalid staged prefix fetch request, falling back to cold prefill",
                                id
                            );
                            self.fallback_to_cold_prefill(slot_idx);
                        }
                    }
                }
            } else {
                self.step_new(slot_idx);
                if matches!(
                    self.request(slot_idx).map(|req| &req.phase),
                    Some(Phase::Prefilling { .. })
                ) {
                    self.queue_prefill(slot_idx);
                }
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
                self.release_attached_prefix_blocks(&req.held_prefix_blocks());
                self.clear_fetch_waiting_for_slot(slot_idx, req.id);
                self.dequeue_prefill(slot_idx);
                self.dequeue_running(slot_idx);
                self.clear_slot_prefix_ownership(slot_idx);

                if let Some(prompt_tokens) = req.cached_prompt_to_publish() {
                    let prompt_vec = prompt_tokens.to_vec();
                    self.slot_materialized_prompt_lens[slot_idx] = prompt_vec.len();
                    self.publish_to_prefix_cache(slot_idx, &prompt_vec, req.session_id.clone());
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
    use super::{best_reusable_slot_for_radix_hit, finish_rejected_request};
    use crate::prefix_cache::BlockId;
    use crate::server_engine::FinishReason;
    use std::collections::HashMap;
    use tokio::sync::mpsc;

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

    #[test]
    fn rejected_request_emits_terminal_delta() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        finish_rejected_request(&tx, FinishReason::Length, 17);
        let delta = rx.try_recv().expect("terminal delta");
        assert_eq!(delta.text_delta, "");
        assert_eq!(delta.finish_reason, Some(FinishReason::Length));
        let usage = delta.usage.expect("usage");
        assert_eq!(usage.prompt_tokens, 17);
        assert_eq!(usage.completion_tokens, 0);
        assert_eq!(usage.total_tokens, 17);
    }
}
