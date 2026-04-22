use super::budget::{PageBudget, full_request_target};
use super::core::{is_full_sealed_prefix, sealed_block_token_count};
use super::{
    ActiveRequest, CompletionStreamDelta, FinishReason, ModelForward, Ordering, Phase,
    STATS_LOG_INTERVAL, Scheduler, TokenUsage, error, info, warn,
};
use crate::kv_tier::{ReadmissionSource, RequestChunkState};
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
    attached_prefix_blocks: Vec<crate::prefix_cache::BlockId>,
    staged_prefix_plan: Option<crate::kv_tier::ReadmissionPlan>,
}

struct QueuedAdmissionCandidate {
    incoming: super::IncomingRequest,
    prompt_tokens: Vec<u32>,
    plan: PrefixAdmissionPlan,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct WaitingRequestHint {
    immediate_reuse_tokens: usize,
    total_reuse_tokens: usize,
}

impl WaitingRequestHint {
    fn from_plan(plan: &PrefixAdmissionPlan, reusable_prefix_len: usize) -> Self {
        let immediate_reuse_tokens = if plan.direct_gpu_attach {
            plan.lookup.matched_len
        } else {
            reusable_prefix_len
        };
        let total_reuse_tokens = if immediate_reuse_tokens > 0 {
            immediate_reuse_tokens
        } else if let Some(staged_prefix_plan) = plan.staged_prefix_plan.as_ref() {
            staged_prefix_plan.matched_len
        } else if !plan.lookup.recompute_advised {
            plan.lookup.matched_len
        } else {
            0
        };
        Self {
            immediate_reuse_tokens,
            total_reuse_tokens,
        }
    }
}

struct DeferredWaitingRequest {
    incoming: super::IncomingRequest,
    prompt_tokens: Vec<u32>,
    hint: WaitingRequestHint,
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

fn matched_sealed_lookup_blocks(blocks: &[crate::kv_tier::LookupBlock]) -> usize {
    blocks
        .iter()
        .filter(|block| !matches!(block.hit_kind, crate::kv_tier::HitKind::Miss))
        .count()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum WaitingInsertBias {
    BeforeEqual,
    AfterEqual,
}

fn waiting_request_precedes(
    incoming_priority: super::RequestPriority,
    incoming_hint: WaitingRequestHint,
    queued_priority: super::RequestPriority,
    queued_hint: WaitingRequestHint,
    bias: WaitingInsertBias,
) -> bool {
    match incoming_priority.cmp(&queued_priority) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        std::cmp::Ordering::Equal => match (
            incoming_hint.immediate_reuse_tokens,
            incoming_hint.total_reuse_tokens,
        )
            .cmp(&(
                queued_hint.immediate_reuse_tokens,
                queued_hint.total_reuse_tokens,
            )) {
            std::cmp::Ordering::Greater => true,
            std::cmp::Ordering::Less => false,
            std::cmp::Ordering::Equal => matches!(bias, WaitingInsertBias::BeforeEqual),
        },
    }
}

fn waiting_insert_position<T>(
    waiting: &std::collections::VecDeque<T>,
    incoming_priority: super::RequestPriority,
    incoming_hint: WaitingRequestHint,
    bias: WaitingInsertBias,
    queued_key: impl Fn(&T) -> (super::RequestPriority, WaitingRequestHint),
) -> usize {
    waiting
        .iter()
        .position(|queued| {
            let (queued_priority, queued_hint) = queued_key(queued);
            waiting_request_precedes(
                incoming_priority,
                incoming_hint,
                queued_priority,
                queued_hint,
                bias,
            )
        })
        .unwrap_or(waiting.len())
}

fn insert_waiting_request_by_priority(
    waiting: &mut std::collections::VecDeque<super::IncomingRequest>,
    incoming: super::IncomingRequest,
    bias: WaitingInsertBias,
) {
    let insert_at = waiting_insert_position(
        waiting,
        incoming.priority,
        WaitingRequestHint::default(),
        bias,
        |queued| (queued.priority, WaitingRequestHint::default()),
    );
    waiting.insert(insert_at, incoming);
}

fn insert_deferred_waiting_request(
    waiting: &mut std::collections::VecDeque<DeferredWaitingRequest>,
    incoming: DeferredWaitingRequest,
    bias: WaitingInsertBias,
) {
    let insert_at = waiting_insert_position(
        waiting,
        incoming.incoming.priority,
        incoming.hint,
        bias,
        |queued| (queued.incoming.priority, queued.hint),
    );
    waiting.insert(insert_at, incoming);
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
    pub(super) fn enqueue_waiting_request(
        &mut self,
        incoming: super::IncomingRequest,
        bias: WaitingInsertBias,
    ) {
        insert_waiting_request_by_priority(&mut self.waiting, incoming, bias);
    }

    fn full_isl_reserved_tokens(plan: &PrefixAdmissionPlan, reusable_prefix_len: usize) -> usize {
        if plan.direct_gpu_attach {
            plan.lookup.matched_len
        } else if reusable_prefix_len > 0 {
            reusable_prefix_len
        } else {
            0
        }
    }

    fn can_reserve_full_isl(
        budget: &PageBudget,
        slot_idx: usize,
        prompt_tokens: usize,
        max_tokens: usize,
        reserved_prefix_tokens: usize,
    ) -> bool {
        budget.can_fit_target(full_request_target(
            slot_idx,
            prompt_tokens,
            max_tokens,
            reserved_prefix_tokens,
        ))
    }

    fn normalize_waiting_request(
        &mut self,
        mut incoming: super::IncomingRequest,
        length_contract: RequestLengthContract,
    ) -> Option<(super::IncomingRequest, Vec<u32>)> {
        if incoming.delta_tx.is_closed() {
            return None;
        }

        let prompt_tokens = match incoming.prompt_tokens.take() {
            Some(tokens) if !tokens.is_empty() => tokens,
            Some(_) => {
                error!("Empty cached prompt tokenization, skipping");
                finish_rejected_request(&incoming.delta_tx, FinishReason::Length, 0);
                return None;
            }
            None => match self.tokenizer.encode(&incoming.prompt) {
                Ok(tokens) if !tokens.is_empty() => tokens,
                Ok(_) => {
                    error!("Empty prompt after tokenization, skipping");
                    finish_rejected_request(&incoming.delta_tx, FinishReason::Length, 0);
                    return None;
                }
                Err(e) => {
                    error!("Tokenization error: {}", e);
                    finish_rejected_request(&incoming.delta_tx, FinishReason::Length, 0);
                    return None;
                }
            },
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
            return None;
        }

        incoming.max_tokens =
            length_contract.clamp_max_tokens(prompt_tokens.len(), incoming.max_tokens);
        Some((incoming, prompt_tokens))
    }

    fn choose_admission_slot(
        plan: &PrefixAdmissionPlan,
        free_slots: &[usize],
    ) -> Option<(usize, usize, usize)> {
        if free_slots.is_empty() {
            return None;
        }
        if let Some((slot_idx, reusable_prefix_len, reusable_cached_prompt_len)) = plan.reusable
            && free_slots.contains(&slot_idx)
        {
            return Some((slot_idx, reusable_prefix_len, reusable_cached_prompt_len));
        }
        Some((free_slots[0], 0, 0))
    }

    fn restore_deferred_waiting_requests(
        &mut self,
        mut deferred_waiting: std::collections::VecDeque<DeferredWaitingRequest>,
    ) {
        while let Some(mut deferred) = deferred_waiting.pop_front() {
            deferred.incoming.prompt_tokens = Some(deferred.prompt_tokens);
            self.waiting.push_back(deferred.incoming);
        }
    }

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
        let block_size = self.prefix_cache.block_size();
        let lookup = self
            .prefix_cache
            .lookup_or_stage(prompt_tokens, crate::kv_tier::LookupHeuristics::default());
        let radix_blocks: Vec<_> = lookup
            .blocks
            .iter()
            .filter_map(|block| block.block_id)
            .collect();
        let matched_sealed_block_count = matched_sealed_lookup_blocks(&lookup.blocks);
        let lookup_is_full_sealed = lookup.matched_len == 0
            || is_full_sealed_prefix(lookup.matched_len, block_size, matched_sealed_block_count);
        debug_assert!(
            lookup_is_full_sealed,
            "lookup_or_stage must classify sealed full blocks only: matched={} blocks={} block_size={}",
            lookup.matched_len, matched_sealed_block_count, block_size,
        );
        let ready_on_gpu = lookup_is_full_sealed && lookup_blocks_ready_on_gpu(&lookup.blocks);
        let gpu_ready_sealed_blocks: Vec<_> = lookup
            .blocks
            .iter()
            .take_while(|block| matches!(block.hit_kind, crate::kv_tier::HitKind::ReadyOnGpu))
            .filter_map(|block| block.block_id)
            .collect();
        let gpu_ready_sealed_tokens =
            sealed_block_token_count(block_size, gpu_ready_sealed_blocks.len());
        let fully_addressable_gpu_hit =
            ready_on_gpu && gpu_ready_sealed_tokens == lookup.matched_len;
        let supports_cross_slot_prefix_attach = self.model.supports_cross_slot_prefix_attach();
        let staged_prefix_plan = if supports_cross_slot_prefix_attach
            && lookup_is_full_sealed
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
        let direct_gpu_attach = supports_cross_slot_prefix_attach
            && lookup_is_full_sealed
            && !lookup.recompute_advised
            && !gpu_ready_sealed_blocks.is_empty()
            && fully_addressable_gpu_hit
            && staged_prefix_plan.is_none();
        let reusable_gpu_prefix = if direct_gpu_attach || staged_prefix_plan.is_some() {
            None
        } else if fully_addressable_gpu_hit && !lookup.recompute_advised {
            best_reusable_slot_for_radix_hit(
                &gpu_ready_sealed_blocks,
                free_slots,
                &self.block_owner_slots,
                &self.slot_materialized_prompt_lens,
                block_size,
            )
        } else {
            None
        };

        PrefixAdmissionPlan {
            radix_blocks,
            lookup,
            reusable: reusable_gpu_prefix,
            direct_gpu_attach,
            attached_prefix_blocks: if direct_gpu_attach {
                gpu_ready_sealed_blocks
            } else {
                Vec::new()
            },
            staged_prefix_plan,
        }
    }

    fn collect_admission_candidates(
        &mut self,
        free_slots: &[usize],
        length_contract: RequestLengthContract,
    ) -> Vec<QueuedAdmissionCandidate> {
        let mut candidates = Vec::new();
        let scan_len = self.waiting.len();
        for _ in 0..scan_len {
            let Some(incoming) = self.waiting.pop_front() else {
                break;
            };
            let Some((incoming, prompt_tokens)) =
                self.normalize_waiting_request(incoming, length_contract)
            else {
                continue;
            };
            let plan = self.build_prefix_admission_plan(&prompt_tokens, free_slots);
            candidates.push(QueuedAdmissionCandidate {
                incoming,
                prompt_tokens,
                plan,
            });
        }
        candidates
    }

    fn release_admission_plan(&mut self, plan: &PrefixAdmissionPlan) {
        self.prefix_cache.release(&plan.radix_blocks);
    }

    fn admit_waiting_candidate(
        &mut self,
        incoming: super::IncomingRequest,
        prompt_tokens: Vec<u32>,
        plan: PrefixAdmissionPlan,
        slot_idx: usize,
        reusable_prefix_len: usize,
        reusable_cached_prompt_len: usize,
    ) {
        let PrefixAdmissionPlan {
            lookup,
            direct_gpu_attach,
            attached_prefix_blocks,
            staged_prefix_plan,
            ..
        } = plan;
        let waiting_fetch = staged_prefix_plan.is_some();
        let ready_on_gpu = lookup_blocks_ready_on_gpu(&lookup.blocks);
        let radix_hit_len = if ready_on_gpu && !lookup.recompute_advised {
            lookup.matched_len
        } else {
            0
        };
        let id = self.next_id;
        self.next_id += 1;

        if let Some(staged) = staged_prefix_plan.as_ref() {
            info!(
                "Request {} → slot {} (prompt={} tokens, staged_prefix={}, queue={})",
                id,
                slot_idx,
                prompt_tokens.len(),
                staged.matched_len,
                self.waiting.len()
            );
        } else if direct_gpu_attach {
            info!(
                "Request {} → slot {} (prompt={} tokens, radix_gpu_attach={}, queue={})",
                id,
                slot_idx,
                prompt_tokens.len(),
                lookup.matched_len,
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
                lookup.matched_len > 0 && (!ready_on_gpu || lookup.recompute_advised);
            let no_reusable_free_slot = lookup.matched_len > 0 && !ready_on_gpu;
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
            trace_context: incoming.trace_context,
            delta_tx: incoming.delta_tx,
            emit_cursor: super::request::EmitCursor::default(),
            phase: if waiting_fetch {
                Phase::WaitingFetch
            } else {
                Phase::Prefilling {
                    effective_tokens: Vec::new(),
                    progress: 0,
                }
            },
            cacheable_prompt_len: 0,
            latest_logprob: None,
            pending_finish_reason: None,
            reusable_prefix_len: if direct_gpu_attach {
                lookup.matched_len
            } else {
                reusable_prefix_len
            },
            reusable_cached_prompt_len,
            attached_prefix_blocks,
            staged_prefix: staged_prefix_plan,
        });
        if incoming.max_tokens == 0 {
            self.finish_request(slot_idx, crate::server_engine::FinishReason::Length);
            return;
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
                return;
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
                    let (host_blocks, disk_blocks, remote_blocks) = staged_prefix.source_counts();
                    self.metrics
                        .record_tier_fetch_plan(host_blocks, disk_blocks, remote_blocks);
                    if let Some(ticket) = self.fetch_dedupe.get(&fetch_key).copied() {
                        if let Some(req) = self.request_mut(slot_idx)
                            && let Some(plan) = req.staged_prefix.as_mut()
                        {
                            plan.mark_fetching();
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
                        if let Some(req) = self.request_mut(slot_idx)
                            && let Some(plan) = req.staged_prefix.as_mut()
                        {
                            debug_assert_eq!(plan.state, RequestChunkState::Planned);
                            plan.mark_fetching();
                        }
                        if let Some(ticket) = self.coordinator_handle.submit_fetch(fetch_requests) {
                            self.fetch_dedupe.insert(fetch_key.clone(), ticket);
                            self.fetch_ticket_keys.insert(ticket, fetch_key);
                            self.fetch_ticket_started_at
                                .insert(ticket, std::time::Instant::now());
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

    fn fallback_to_cold_prefill(&mut self, slot_idx: usize) {
        self.fallback_to_cold_prefill_inner(slot_idx, true);
    }

    fn fallback_to_cold_prefill_without_release(&mut self, slot_idx: usize) {
        self.fallback_to_cold_prefill_inner(slot_idx, false);
    }

    fn fallback_to_cold_prefill_inner(&mut self, slot_idx: usize, release_held_blocks: bool) {
        if let Some((host_blocks, disk_blocks, remote_blocks)) =
            self.request(slot_idx).and_then(|req| {
                req.staged_prefix
                    .as_ref()
                    .map(crate::kv_tier::ReadmissionPlan::source_counts)
            })
            && host_blocks + disk_blocks + remote_blocks > 0
        {
            self.metrics.record_tier_fetch_fallback();
        }
        let held_blocks = self
            .request(slot_idx)
            .and_then(|req| {
                req.staged_prefix
                    .as_ref()
                    .map(crate::kv_tier::ReadmissionPlan::block_ids)
            })
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

    fn validate_staged_sealed_prefix(
        &self,
        request_id: u64,
        prompt_tokens: &[u32],
        staged_prefix: &crate::kv_tier::ReadmissionPlan,
    ) -> anyhow::Result<()> {
        if staged_prefix.blocks.is_empty() {
            return Ok(());
        }
        let block_size = self.prefix_cache.block_size();
        let sealed_tokens = sealed_block_token_count(block_size, staged_prefix.blocks.len());
        if staged_prefix.matched_len > prompt_tokens.len()
            || !is_full_sealed_prefix(
                staged_prefix.matched_len,
                block_size,
                staged_prefix.blocks.len(),
            )
        {
            return Err(anyhow::anyhow!(
                "invalid staged sealed prefix shape for request {} (matched={} blocks={} prompt={})",
                request_id,
                staged_prefix.matched_len,
                staged_prefix.blocks.len(),
                prompt_tokens.len()
            ));
        }
        debug_assert_eq!(
            staged_prefix.matched_len, sealed_tokens,
            "staged readmission plans must only cover full sealed radix blocks"
        );
        Ok(())
    }

    fn gpu_ready_staged_prefix_plan(
        &mut self,
        request_id: u64,
        prompt_tokens: &[u32],
        staged_prefix: &crate::kv_tier::ReadmissionPlan,
    ) -> anyhow::Result<crate::kv_tier::ReadmissionPlan> {
        self.validate_staged_sealed_prefix(request_id, prompt_tokens, staged_prefix)?;
        let final_lookup = self.prefix_cache.lookup_or_stage(
            &prompt_tokens[..staged_prefix.matched_len],
            crate::kv_tier::LookupHeuristics::default(),
        );
        if !lookup_blocks_ready_on_gpu(&final_lookup.blocks) {
            return Err(anyhow::anyhow!(
                "staged sealed prefix promotion did not become GPU-runnable (matched={} expected={})",
                final_lookup.matched_len,
                staged_prefix.matched_len
            ));
        }
        let final_plan = self
            .build_staged_prefix_plan(&final_lookup)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "staged sealed prefix promotion lost full-block shape (matched={} expected={})",
                    final_lookup.matched_len,
                    staged_prefix.matched_len
                )
            })?;
        if final_plan.matched_len != staged_prefix.matched_len {
            return Err(anyhow::anyhow!(
                "staged sealed prefix promotion matched {} tokens, expected {}",
                final_plan.matched_len,
                staged_prefix.matched_len
            ));
        }
        debug_assert!(
            final_plan.blocks.iter().all(|block| block.source.is_none()),
            "GPU-ready staged prefix plans should not retain staging sources"
        );
        Ok(final_plan)
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
        let block_size = self.prefix_cache.block_size();
        self.validate_staged_sealed_prefix(request_id, prompt_tokens, &staged_prefix)?;
        let fetched_by_id: std::collections::HashMap<
            crate::prefix_cache::BlockId,
            &crate::kv_tier::FetchedBlock,
        > = fetched_blocks
            .iter()
            .map(|block| (block.block_id, block))
            .collect();

        let pages_per_block = block_size.div_ceil(self.paged_kv_pool.page_size).max(1);
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
                Some(
                    ReadmissionSource::HostPinned { .. }
                    | ReadmissionSource::Disk { .. }
                    | ReadmissionSource::Remote { .. },
                ) => {
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

        let promoted_blocks = promoted_pages
            .into_iter()
            .map(|(old_block_id, new_block_id, pages)| {
                self.block_owner_slots.remove(&old_block_id);
                (new_block_id, pages)
            })
            .collect::<Vec<_>>();
        self.record_sealed_gpu_blocks(
            slot_idx,
            promoted_blocks,
            session_id.as_ref(),
            self.config.prefix_cache_keepalive_ticks,
            false,
        );
        let (host_blocks, disk_blocks, remote_blocks) = staged_prefix.source_counts();
        self.metrics
            .record_tier_fetch_promoted(host_blocks + disk_blocks + remote_blocks);

        info!(
            "Request {}: staged sealed prefix ready, promoted {}/{} tokens into T0",
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

    fn handle_coordinator_event(&mut self, event: crate::kv_tier::CoordinatorEvent) {
        // The runtime consumes coordinator results in one place so the request
        // state machine stays linear:
        // - `Store*` mutates prefix-cache store state and releases T1 regions
        // - `FetchCompleted` promotes staged bytes into T0, then re-enters the
        //   normal prefill path
        // - `FetchFailed` always falls back to cold prefill
        match event {
            crate::kv_tier::CoordinatorEvent::CommandQueued(_)
            | crate::kv_tier::CoordinatorEvent::FetchQueued { .. } => {}
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
                reason,
            } => {
                warn!(
                    "Store failed for ticket {} on block {:?}: {}",
                    ticket.0, failed_block, reason
                );
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
                self.fetch_ticket_started_at.remove(&ticket);
                let Some(waiters) = self.fetch_waiting.remove(&ticket) else {
                    self.release_unclaimed_fetch_regions(&blocks);
                    return;
                };
                if let Some(key) = self.fetch_ticket_keys.remove(&ticket) {
                    self.fetch_dedupe.remove(&key);
                }
                let ready_waiters = self.collect_fetch_waiters(waiters);
                if ready_waiters.is_empty() {
                    self.release_unclaimed_fetch_regions(&blocks);
                    return;
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
                    return;
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
            crate::kv_tier::CoordinatorEvent::FetchFailed {
                ticket,
                failed_block,
                reason,
            } => {
                warn!(
                    "Fetch failed for ticket {} on block {:?}: {}",
                    ticket.0, failed_block, reason
                );
                self.fetch_ticket_started_at.remove(&ticket);
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

    fn drain_coordinator_events(&mut self) {
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

    fn handle_emit_event(&mut self, event: crate::scheduler::cuda::core::EmitEvent) {
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

    fn drain_emit_events(&mut self) {
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

    fn drain_request_rx(&mut self) {
        while let Ok(req) = self.request_rx.try_recv() {
            self.waiting_count.fetch_sub(1, Ordering::Relaxed);
            self.enqueue_waiting_request(req, WaitingInsertBias::AfterEqual);
        }
    }

    fn drain_wakeup_rx(&mut self) {
        while self.wakeup_rx.try_recv().is_ok() {}
    }

    fn handle_wakeup_disconnect(&mut self) {
        self.wakeup_live = false;
        self.drain_request_rx();
    }

    pub(super) fn wait_for_emit_gates(&mut self) -> u128 {
        if self.emit_gate_waiting.is_empty() {
            return 0;
        }

        let wait_t = std::time::Instant::now();
        while !self.emit_gate_waiting.is_empty() {
            crossbeam_channel::select! {
                recv(self.emit_events) -> event => {
                    match event {
                        Ok(event) => self.handle_emit_event(event),
                        Err(err) => panic!("emit event channel disconnected: {err}"),
                    }
                }
                recv(self.coordinator_events) -> event => {
                    match event {
                        Ok(event) => self.handle_coordinator_event(event),
                        Err(_) => error!("Coordinator event channel disconnected"),
                    }
                }
                recv(self.wakeup_rx) -> wakeup => {
                    match wakeup {
                        Ok(()) => {
                            self.drain_request_rx();
                            self.drain_wakeup_rx();
                        }
                        Err(_) => self.handle_wakeup_disconnect(),
                    }
                }
            }
        }
        wait_t.elapsed().as_micros()
    }

    fn wait_for_coordinator_or_request(&mut self) -> bool {
        if !self.wakeup_live {
            if let Ok(event) = self.coordinator_events.recv() {
                self.handle_coordinator_event(event);
                return true;
            }
            error!("Coordinator event channel disconnected");
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

    fn wait_for_wakeup(&mut self) -> bool {
        if self.is_fetch_wait_bound() {
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
        let length_contract = RequestLengthContract::derive(
            self.paged_kv_pool.max_total_tokens,
            self.effective_max_seq_len,
        );
        let _ = self.evict_prefix_cache_if_pressured();
        let mut available_free_slots = self.free_slots();
        if available_free_slots.is_empty() {
            return;
        }

        let candidates = self.collect_admission_candidates(&available_free_slots, length_contract);
        let mut deferred_waiting = std::collections::VecDeque::new();
        let mut admission_budget = PageBudget::from_scheduler(self, self.paged_kv_pool.is_active());
        for (slot_idx, req) in self.active.iter().enumerate() {
            let Some(req) = req.as_ref() else {
                continue;
            };
            if req.delta_tx.is_closed() || matches!(req.phase, Phase::Finished) {
                continue;
            }
            admission_budget.reserve_target(full_request_target(
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
            admission_budget.reserve_target(full_request_target(
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

    fn cleanup(&mut self) {
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
    use super::{
        DeferredWaitingRequest, PrefixAdmissionPlan, WaitingInsertBias, WaitingRequestHint,
        best_reusable_slot_for_radix_hit, finish_rejected_request, insert_deferred_waiting_request,
        insert_waiting_request_by_priority, lookup_blocks_ready_on_gpu,
        matched_sealed_lookup_blocks,
    };
    use crate::prefix_cache::BlockId;
    use crate::scheduler::cuda::budget::{PageBudget, full_request_target};
    use crate::scheduler::cuda::core::is_full_sealed_prefix;
    use crate::scheduler::{IncomingRequest, RequestPriority};
    use crate::server_engine::FinishReason;
    use std::collections::{HashMap, VecDeque};
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

    fn queued_request(label: &str, priority: RequestPriority) -> IncomingRequest {
        let (tx, _rx) = mpsc::unbounded_channel();
        IncomingRequest {
            prompt: label.to_string(),
            prompt_tokens: None,
            max_tokens: 1,
            sampling: crate::sampler::SamplingParams::default(),
            stop: None,
            priority,
            session_id: None,
            delta_tx: tx,
            trace_context: None,
        }
    }

    #[test]
    fn waiting_insert_after_equal_preserves_fifo_for_same_priority() {
        let mut waiting = VecDeque::from(vec![
            queued_request("high", RequestPriority::High),
            queued_request("first-normal", RequestPriority::Normal),
        ]);

        insert_waiting_request_by_priority(
            &mut waiting,
            queued_request("second-normal", RequestPriority::Normal),
            WaitingInsertBias::AfterEqual,
        );

        assert_eq!(
            waiting
                .into_iter()
                .map(|req| req.prompt)
                .collect::<Vec<_>>(),
            vec!["high", "first-normal", "second-normal"]
        );
    }

    #[test]
    fn waiting_insert_before_equal_gives_requeues_equal_priority_preference() {
        let mut waiting = VecDeque::from(vec![
            queued_request("high", RequestPriority::High),
            queued_request("queued-normal", RequestPriority::Normal),
        ]);

        insert_waiting_request_by_priority(
            &mut waiting,
            queued_request("requeued-normal", RequestPriority::Normal),
            WaitingInsertBias::BeforeEqual,
        );

        assert_eq!(
            waiting
                .into_iter()
                .map(|req| req.prompt)
                .collect::<Vec<_>>(),
            vec!["high", "requeued-normal", "queued-normal"]
        );
    }

    fn deferred_request(
        label: &str,
        priority: RequestPriority,
        hint: WaitingRequestHint,
    ) -> DeferredWaitingRequest {
        DeferredWaitingRequest {
            incoming: queued_request(label, priority),
            prompt_tokens: vec![1, 2, 3],
            hint,
        }
    }

    fn hint_plan(
        matched_len: usize,
        reusable_prefix_len: usize,
        direct_gpu_attach: bool,
        staged_prefix_len: Option<usize>,
        recompute_advised: bool,
    ) -> WaitingRequestHint {
        let staged_prefix_plan = staged_prefix_len
            .map(|matched_len| crate::kv_tier::ReadmissionPlan::new(matched_len, Vec::new()));
        WaitingRequestHint::from_plan(
            &PrefixAdmissionPlan {
                radix_blocks: Vec::new(),
                lookup: crate::kv_tier::LookupOutcome::new(
                    matched_len,
                    Vec::new(),
                    recompute_advised,
                ),
                reusable: None,
                direct_gpu_attach,
                attached_prefix_blocks: Vec::new(),
                staged_prefix_plan,
            },
            reusable_prefix_len,
        )
    }

    #[test]
    fn deferred_waiting_keeps_priority_primary_over_prefix_hint() {
        let mut waiting = std::collections::VecDeque::from(vec![deferred_request(
            "high-cold",
            RequestPriority::High,
            WaitingRequestHint::default(),
        )]);

        insert_deferred_waiting_request(
            &mut waiting,
            deferred_request(
                "normal-gpu-ready",
                RequestPriority::Normal,
                hint_plan(48, 48, false, None, false),
            ),
            WaitingInsertBias::BeforeEqual,
        );

        assert_eq!(
            waiting
                .into_iter()
                .map(|req| req.incoming.prompt)
                .collect::<Vec<_>>(),
            vec!["high-cold", "normal-gpu-ready"]
        );
    }

    #[test]
    fn deferred_waiting_prefers_gpu_ready_then_larger_prefix_within_same_priority() {
        let mut waiting = std::collections::VecDeque::from(vec![
            deferred_request(
                "queued-staged",
                RequestPriority::Normal,
                hint_plan(64, 0, false, Some(64), false),
            ),
            deferred_request(
                "queued-cold",
                RequestPriority::Normal,
                WaitingRequestHint::default(),
            ),
        ]);

        insert_deferred_waiting_request(
            &mut waiting,
            deferred_request(
                "gpu-ready",
                RequestPriority::Normal,
                hint_plan(16, 16, false, None, false),
            ),
            WaitingInsertBias::BeforeEqual,
        );

        assert_eq!(
            waiting
                .into_iter()
                .map(|req| req.incoming.prompt)
                .collect::<Vec<_>>(),
            vec!["gpu-ready", "queued-staged", "queued-cold"]
        );
    }

    #[test]
    fn deferred_waiting_before_equal_still_precedes_same_hint_peer() {
        let mut waiting = std::collections::VecDeque::from(vec![deferred_request(
            "queued",
            RequestPriority::Normal,
            hint_plan(32, 32, false, None, false),
        )]);

        insert_deferred_waiting_request(
            &mut waiting,
            deferred_request(
                "requeued",
                RequestPriority::Normal,
                hint_plan(32, 32, false, None, false),
            ),
            WaitingInsertBias::BeforeEqual,
        );

        assert_eq!(
            waiting
                .into_iter()
                .map(|req| req.incoming.prompt)
                .collect::<Vec<_>>(),
            vec!["requeued", "queued"]
        );
    }

    #[test]
    fn admission_budget_accounts_for_prior_reservations_in_same_pass() {
        let mut budget = PageBudget::new(3, vec![0, 0], 4, true);

        assert!(budget.can_fit_target(full_request_target(0, 8, 4, 0)));
        budget.reserve_target(full_request_target(0, 8, 4, 0));

        assert!(
            !budget.can_fit_target(full_request_target(1, 8, 8, 0)),
            "later admissions must see pages reserved by earlier ones in the same assign pass",
        );
    }

    #[test]
    fn admission_budget_honors_attached_prefix_as_existing_seq() {
        let mut budget = PageBudget::new(1, vec![0], 4, true);

        assert!(budget.can_fit_target(full_request_target(0, 4, 4, 4)));
        budget.reserve_target(full_request_target(0, 4, 4, 4));
        assert_eq!(budget.remaining_free_pages(), 0);
        assert_eq!(budget.planned_seq_len(0), 8);
    }

    #[test]
    fn admission_budget_keeps_full_request_headroom_for_active_slots() {
        let mut budget = PageBudget::new(2, vec![4, 0], 4, true);

        // An already-admitted slot with a 4-token prompt and 4-token decode
        // tail must keep the extra decode page reserved across scheduler
        // iterations; new admissions cannot borrow it away.
        budget.reserve_target(full_request_target(0, 4, 4, 0));
        assert_eq!(budget.remaining_free_pages(), 1);

        assert!(
            !budget.can_fit_target(full_request_target(1, 4, 4, 0)),
            "later admissions must respect full-request headroom held by active slots",
        );
        assert!(budget.can_fit_target(full_request_target(1, 4, 0, 0)));
    }

    #[test]
    fn matched_sealed_lookup_blocks_ignore_trailing_tombstone() {
        let blocks = vec![
            crate::kv_tier::LookupBlock {
                block_id: Some(crate::prefix_cache::BlockId(10)),
                hit_kind: crate::kv_tier::HitKind::ReadyOnGpu,
            },
            crate::kv_tier::LookupBlock {
                block_id: Some(crate::prefix_cache::BlockId(20)),
                hit_kind: crate::kv_tier::HitKind::ReadyOnGpu,
            },
            crate::kv_tier::LookupBlock {
                block_id: None,
                hit_kind: crate::kv_tier::HitKind::Miss,
            },
        ];

        assert_eq!(matched_sealed_lookup_blocks(&blocks), 2);
        assert!(lookup_blocks_ready_on_gpu(&blocks));
        assert!(is_full_sealed_prefix(
            8,
            4,
            matched_sealed_lookup_blocks(&blocks)
        ));
    }
}
