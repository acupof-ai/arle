//! Admission-side scheduler runtime methods.
//!
//! Split out of `runtime.rs` (pure structural refactor — no behavior change).
//! Contains: waiting-queue normalization, prefix admission planning, slot
//! materialization, cold-prefill fallback, and the staged-prefix promotion path.

use super::super::budget::{PageBudget, estimated_request_target};
use super::super::core::{is_full_sealed_prefix, sealed_block_token_count};
use super::super::{ActiveRequest, ModelForward, Phase, Scheduler, error, info, warn};
use super::helpers::{
    DeferredWaitingRequest, FetchWaiter, PrefixAdmissionPlan, QueuedAdmissionCandidate,
    WaitingInsertBias, best_reusable_slot_for_radix_hit, finish_rejected_request,
    insert_waiting_request_by_priority, lookup_blocks_ready_on_gpu, matched_sealed_lookup_blocks,
    session_affinity_tokens_for_plan, staged_prefix_prefetch_state,
};
use crate::kv_tier::{LookupHeuristics, LookupOutcome, ReadmissionSource, RequestChunkState};
use crate::scheduler::types::RequestLengthContract;
use crate::server_engine::FinishReason;
use crate::types::SessionId;

impl<M: ModelForward> Scheduler<M> {
    fn cold_prefix_admission_plan(&self) -> PrefixAdmissionPlan {
        PrefixAdmissionPlan {
            radix_blocks: Vec::new(),
            lookup: crate::kv_tier::LookupOutcome::new(0, Vec::new(), false),
            session_resume_tokens: 0,
            reusable: None,
            direct_gpu_attach: false,
            attached_prefix_blocks: Vec::new(),
            staged_prefix_plan: None,
            session_slot_hold: None,
        }
    }

    pub(in crate::scheduler::cuda) fn enqueue_waiting_request(
        &mut self,
        incoming: super::super::IncomingRequest,
        bias: WaitingInsertBias,
    ) {
        insert_waiting_request_by_priority(&mut self.waiting, incoming, bias);
    }

    pub(super) fn full_isl_reserved_tokens(
        plan: &PrefixAdmissionPlan,
        reusable_prefix_len: usize,
    ) -> usize {
        if plan.direct_gpu_attach {
            plan.lookup.matched_len
        } else if reusable_prefix_len > 0 {
            reusable_prefix_len
        } else {
            0
        }
    }

    pub(super) fn can_reserve_full_isl(
        budget: &PageBudget,
        slot_idx: usize,
        prompt_tokens: usize,
        max_tokens: usize,
        reserved_prefix_tokens: usize,
    ) -> bool {
        budget.can_fit_target(estimated_request_target(
            slot_idx,
            prompt_tokens,
            max_tokens,
            reserved_prefix_tokens,
        ))
    }

    pub(super) fn normalize_waiting_request(
        &mut self,
        mut incoming: super::super::IncomingRequest,
        length_contract: RequestLengthContract,
    ) -> Option<(super::super::IncomingRequest, Vec<u32>)> {
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

    pub(super) fn choose_admission_slot(
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

    pub(super) fn restore_deferred_waiting_requests(
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
    pub(super) fn build_prefix_admission_plan(
        &mut self,
        prompt_tokens: &[u32],
        session_id: Option<&SessionId>,
        free_slots: &[usize],
    ) -> PrefixAdmissionPlan {
        if !self.config.prefix_cache_enabled {
            return self.cold_prefix_admission_plan();
        }
        if self.config.short_prompt_bypass_tokens > 0
            && prompt_tokens.len() <= self.config.short_prompt_bypass_tokens
        {
            return self.cold_prefix_admission_plan();
        }

        let block_size = self.prefix_cache.block_size();
        let heuristics = LookupHeuristics::default();
        let mut session_slot_hold = None;
        let mut lookup = if let Some(session_id) = session_id
            && let Some(session_lookup) =
                self.lookup_session_slot_or_stage(session_id, prompt_tokens.len(), heuristics)
        {
            session_slot_hold = Some(session_lookup.hold);
            session_lookup.lookup
        } else {
            self.prefix_cache.lookup_or_stage(prompt_tokens, heuristics)
        };
        let mut session_resume_tokens = 0;
        if session_slot_hold.is_some() {
            session_resume_tokens = lookup.matched_len;
        } else if let Some(session_id) = session_id {
            let session_lookup = self.prefix_cache.lookup_session_prefix_or_stage(
                session_id,
                prompt_tokens,
                heuristics,
            );
            if session_lookup.matched_len > 0 {
                session_resume_tokens = session_lookup.matched_len;
            }
            if Self::session_lookup_preferred(&session_lookup, &lookup) {
                self.release_lookup_blocks(&lookup);
                lookup = session_lookup;
            } else {
                self.release_lookup_blocks(&session_lookup);
            }
        }
        let radix_blocks: Vec<_> = if session_slot_hold.is_some() {
            Vec::new()
        } else {
            lookup
                .blocks
                .iter()
                .filter_map(|block| block.block_id)
                .collect()
        };
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
            session_resume_tokens,
            reusable: reusable_gpu_prefix,
            direct_gpu_attach,
            attached_prefix_blocks: if direct_gpu_attach {
                gpu_ready_sealed_blocks
            } else {
                Vec::new()
            },
            staged_prefix_plan,
            session_slot_hold,
        }
    }

    pub(super) fn collect_admission_candidates(
        &mut self,
        free_slots: &[usize],
    ) -> Vec<QueuedAdmissionCandidate> {
        let mut candidates = Vec::new();
        let scan_len = self.waiting.len();
        for _ in 0..scan_len {
            let Some(mut incoming) = self.waiting.pop_front() else {
                break;
            };
            let Some(prompt_tokens) = incoming.prompt_tokens.take() else {
                error!("Waiting request missing normalized prompt tokens, rejecting");
                finish_rejected_request(&incoming.delta_tx, FinishReason::Length, 0);
                continue;
            };
            if prompt_tokens.is_empty() {
                error!("Waiting request has empty normalized prompt tokens, rejecting");
                finish_rejected_request(&incoming.delta_tx, FinishReason::Length, 0);
                continue;
            }
            let plan = self.build_prefix_admission_plan(
                &prompt_tokens,
                incoming.session_id.as_ref(),
                free_slots,
            );
            let reusable_prefix_len = plan
                .reusable
                .map(|(_, reusable_prefix_len, _)| reusable_prefix_len)
                .unwrap_or_default();
            let session_affinity_tokens = session_affinity_tokens_for_plan(
                &plan,
                incoming.session_id.as_ref(),
                self.prefix_cache.block_size(),
                |block_id| {
                    self.prefix_cache
                        .block_metadata(block_id)
                        .and_then(|metadata| metadata.session_id)
                },
            );
            let hint = super::helpers::WaitingRequestHint::from_plan(&plan, reusable_prefix_len)
                .with_session_affinity_tokens(session_affinity_tokens);
            candidates.push(QueuedAdmissionCandidate {
                incoming,
                prompt_tokens,
                plan,
                hint,
            });
        }
        candidates
    }

    pub(super) fn release_admission_plan(&mut self, plan: &PrefixAdmissionPlan) {
        self.prefix_cache.release(&plan.radix_blocks);
        self.release_session_slot_hold(plan.session_slot_hold.as_ref());
    }

    fn release_lookup_blocks(&mut self, lookup: &LookupOutcome) {
        let blocks = lookup
            .blocks
            .iter()
            .filter_map(|block| block.block_id)
            .collect::<Vec<_>>();
        self.prefix_cache.release(&blocks);
    }

    fn session_lookup_preferred(
        session_lookup: &LookupOutcome,
        token_lookup: &LookupOutcome,
    ) -> bool {
        session_lookup.matched_len > token_lookup.matched_len
            || (session_lookup.matched_len > 0
                && session_lookup.matched_len == token_lookup.matched_len
                && token_lookup.recompute_advised
                && !session_lookup.recompute_advised)
    }

    pub(super) fn admit_waiting_candidate(
        &mut self,
        incoming: super::super::IncomingRequest,
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
            session_slot_hold,
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
            speculative: incoming.speculative,
            spec_acceptance_tracker: self
                .config
                .spec_enabled
                .then(crate::speculative::AcceptanceTracker::default_window),
            spec_decode_disabled: false,
            session_id: incoming.session_id,
            trace_context: incoming.trace_context,
            delta_tx: incoming.delta_tx,
            emit_cursor: super::super::request::EmitCursor::default(),
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
            session_slot_hold,
        });
        if incoming.max_tokens == 0 {
            self.finish_request(slot_idx, crate::server_engine::FinishReason::Length);
            return;
        }
        if matches!(
            self.request(slot_idx).map(|req| &req.phase),
            Some(Phase::WaitingFetch)
        ) {
            if self.try_complete_direct_host_staged_prefix(slot_idx) {
                return;
            }
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

    pub(super) fn fallback_to_cold_prefill(&mut self, slot_idx: usize) {
        self.fallback_to_cold_prefill_inner(slot_idx, true);
    }

    pub(super) fn fallback_to_cold_prefill_without_release(&mut self, slot_idx: usize) {
        self.fallback_to_cold_prefill_inner(slot_idx, false);
    }

    pub(super) fn fallback_to_cold_prefill_inner(
        &mut self,
        slot_idx: usize,
        release_held_blocks: bool,
    ) {
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
                if req.session_slot_hold.is_some() {
                    None
                } else {
                    req.staged_prefix
                        .as_ref()
                        .map(crate::kv_tier::ReadmissionPlan::block_ids)
                }
            })
            .unwrap_or_default();
        if release_held_blocks && !held_blocks.is_empty() {
            self.prefix_cache.release(&held_blocks);
        }
        let session_slot_hold = self
            .request(slot_idx)
            .and_then(|req| req.session_slot_hold.clone());
        if release_held_blocks {
            self.release_session_slot_hold(session_slot_hold.as_ref());
        }
        if let Some(req) = self.request_mut(slot_idx) {
            req.staged_prefix = None;
            req.reusable_prefix_len = 0;
            req.reusable_cached_prompt_len = 0;
            req.attached_prefix_blocks.clear();
            req.session_slot_hold = None;
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

    pub(super) fn release_unclaimed_fetch_regions(&self, blocks: &[crate::kv_tier::FetchedBlock]) {
        for block in blocks {
            if block.release_after_promote {
                self.release_host_region(block.host_region);
            }
        }
    }

    pub(super) fn maybe_prefetch_staged_prefix(&mut self, plan: &PrefixAdmissionPlan) {
        let Some(staged_prefix) = plan.staged_prefix_plan.as_ref() else {
            return;
        };
        let Some(prefetch_state) = staged_prefix_prefetch_state(staged_prefix) else {
            return;
        };
        if !self.tier_policy.allow_prefetch(
            self.coordinator_handle
                .queue_stats(crate::kv_tier::QueueKind::Fetch),
        ) {
            return;
        }
        let Some(fetch_key) = staged_prefix.fetch_key() else {
            return;
        };
        if self.fetch_dedupe.contains_key(&fetch_key) {
            return;
        }
        let Some(fetch_requests) = staged_prefix.fetch_requests(&self.host_pinned_pool) else {
            return;
        };
        let Some(ticket) = self.coordinator_handle.submit_fetch(fetch_requests) else {
            return;
        };
        self.fetch_dedupe.insert(fetch_key.clone(), ticket);
        self.fetch_ticket_keys.insert(ticket, fetch_key);
        self.fetch_ticket_started_at
            .insert(ticket, std::time::Instant::now());
        self.prefetch_fetching.insert(ticket, prefetch_state);
        info!(
            "Prefetch {} queued: matched={} src=h:{}/d:{}/r:{}",
            ticket.0,
            staged_prefix.matched_len,
            prefetch_state.host_blocks,
            prefetch_state.disk_blocks,
            prefetch_state.remote_blocks
        );
    }

    pub(super) fn validate_staged_sealed_prefix(
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

    pub(super) fn gpu_ready_staged_prefix_plan(
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

    pub(super) fn promote_fetched_prefix(
        &mut self,
        waiter: &FetchWaiter,
        fetched_blocks: &[crate::kv_tier::FetchedBlock],
    ) -> anyhow::Result<()> {
        let slot_idx = waiter.slot_idx;
        let request_id = waiter.request_id;
        let prompt_tokens = &waiter.prompt_tokens;
        let session_id = waiter.session_id.clone();
        let staged_prefix = waiter.staged_prefix.clone();
        let session_slot_hold = waiter.session_slot_hold.clone();
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
        let staged_block_count = staged_prefix
            .blocks
            .iter()
            .filter(|block| block.source.is_some())
            .count();
        let required_promote_pages = staged_block_count.saturating_mul(pages_per_block);
        if required_promote_pages > self.pool_free_pages() {
            self.evict_prefix_cache_for_allocation(required_promote_pages);
        }
        let mut promoted_pages: Vec<(
            crate::prefix_cache::BlockId,
            crate::prefix_cache::BlockId,
            Vec<u32>,
        )> = Vec::new();
        let mut final_block_ids = Vec::with_capacity(staged_prefix.blocks.len());
        let mut fingerprints = Vec::with_capacity(staged_prefix.blocks.len());
        let mut consumed_host_regions = Vec::new();

        for block in &staged_prefix.blocks {
            fingerprints.push(block.fingerprint);
            match &block.source {
                None => final_block_ids.push(block.block_id),
                Some(source) => {
                    let fetched = fetched_by_id.get(&block.block_id).copied().ok_or_else(|| {
                        anyhow::anyhow!(
                            "missing fetched host staging for staged prefix block {:?}",
                            block.block_id
                        )
                    })?;
                    let pages = self.paged_kv_pool.alloc_detached_pages(pages_per_block)?;
                    let copy_result =
                        self.host_pinned_pool
                            .with_region_slice(fetched.host_region, |payload| {
                                self.paged_kv_pool.copy_pages_from_host(
                                    self.model.device_context(),
                                    &pages,
                                    payload,
                                )
                            });
                    let copy_result = match copy_result {
                        Ok(inner) => inner,
                        Err(err) => {
                            let _ = self.paged_kv_pool.release_pages(&pages);
                            return Err(err);
                        }
                    };
                    if let Err(err) = copy_result {
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
                    if let ReadmissionSource::HostPinned { region } = source {
                        consumed_host_regions.push(*region);
                    }
                }
            }
        }

        if session_slot_hold.is_some() {
            for (old_block_id, new_block_id, _) in &promoted_pages {
                if !self.prefix_cache.retag_block(*old_block_id, *new_block_id) {
                    for (_, _, pages) in promoted_pages {
                        let _ = self.paged_kv_pool.release_pages(&pages);
                    }
                    return Err(anyhow::anyhow!(
                        "session-slot staged prefix retag failed for {:?} -> {:?}",
                        old_block_id,
                        new_block_id
                    ));
                }
                self.retag_session_slot_block(*old_block_id, *new_block_id);
            }
        } else {
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
            session_id.is_some()
                && prompt_tokens.len() >= self.config.t1_host_pinned_min_prompt_tokens,
        );
        for region in consumed_host_regions {
            self.release_host_region(region);
        }
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
}
