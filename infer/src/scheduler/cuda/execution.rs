use super::budget::{PageBudget, PageGrowth, StepTokenBudget, clipped_max_new_tokens_estimate};
use super::nvtx_scopes::nvtx_scope;
use super::spec_path::SpecPath;
use super::{ModelForward, Phase, Scheduler, info};
use crate::metrics::SchedulerPlanLabel;
use crate::scheduler::{LogicalDecodeRow, LogicalPrefillRow, LogicalServePlan};
use crate::types::RequestId;

fn logical_scheduler_shadow_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("INFER_LOGICAL_SCHEDULER_SHADOW")
                .as_deref()
                .map(str::to_ascii_lowercase)
                .as_deref(),
            Ok("1" | "true" | "yes" | "on")
        )
    })
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PrefillReservation {
    pub prefill_tokens: usize,
    pub page_growth: PageGrowth,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PrefillCandidate {
    pub slot_idx: usize,
    pub reservation: PrefillReservation,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PrefillCandidateScore {
    queue_rank: usize,
}

#[derive(Clone, Copy, Debug)]
struct ScoredPrefillCandidate {
    candidate: PrefillCandidate,
    score: PrefillCandidateScore,
}

#[derive(Clone, Debug)]
enum StepPlan {
    Idle,
    Decode,
    SpecDecode,
    Prefill(Vec<PrefillCandidate>),
    Split(Vec<PrefillCandidate>),
    Mixed(Vec<PrefillCandidate>),
}

impl StepPlan {
    fn label(&self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Decode | Self::SpecDecode => "decode",
            Self::Prefill(_) => "prefill",
            Self::Split(_) => "split",
            Self::Mixed(_) => "mixed",
        }
    }

    fn metrics_label(&self) -> SchedulerPlanLabel {
        match self {
            Self::Idle => SchedulerPlanLabel::Idle,
            Self::Decode | Self::SpecDecode => SchedulerPlanLabel::Decode,
            Self::Prefill(_) => SchedulerPlanLabel::Prefill,
            Self::Split(_) => SchedulerPlanLabel::Split,
            Self::Mixed(_) => SchedulerPlanLabel::Mixed,
        }
    }

    fn is_idle(&self) -> bool {
        matches!(self, Self::Idle)
    }

    fn expects_decode(&self) -> bool {
        matches!(
            self,
            Self::Decode | Self::SpecDecode | Self::Split(_) | Self::Mixed(_)
        )
    }

    fn scheduled_prefill_rows(&self) -> u64 {
        match self {
            Self::Prefill(candidates) | Self::Split(candidates) | Self::Mixed(candidates) => {
                candidates.len() as u64
            }
            Self::Idle | Self::Decode | Self::SpecDecode => 0,
        }
    }

    fn scheduled_prefill_tokens(&self) -> u64 {
        match self {
            Self::Prefill(candidates) | Self::Split(candidates) | Self::Mixed(candidates) => {
                candidates
                    .iter()
                    .map(|candidate| candidate.reservation.prefill_tokens as u64)
                    .sum()
            }
            Self::Idle | Self::Decode | Self::SpecDecode => 0,
        }
    }
}

fn route_spec_plan(spec_enabled: bool, _spec_draft_k: usize, plan: StepPlan) -> StepPlan {
    if spec_enabled && matches!(plan, StepPlan::Decode) {
        StepPlan::SpecDecode
    } else {
        plan
    }
}

#[derive(Debug)]
struct PrefillBudget {
    token_budget: StepTokenBudget,
    long_prefill_token_threshold: usize,
    decode_active: bool,
    page_budget: PageBudget,
}

impl PrefillBudget {
    fn from_scheduler<M: ModelForward>(scheduler: &Scheduler<M>) -> Self {
        let decode_slots: Vec<usize> = scheduler
            .running_batch
            .iter()
            .filter(|&&slot_idx| scheduler.slot_is_runnable_decode(slot_idx))
            .copied()
            .collect();
        Self::from_scheduler_for_decode_slots(scheduler, &decode_slots)
    }

    fn from_scheduler_for_decode_slots<M: ModelForward>(
        scheduler: &Scheduler<M>,
        decode_slots: &[usize],
    ) -> Self {
        let mut budget = Self {
            token_budget: StepTokenBudget::for_prefill(
                scheduler.config.max_num_batched_tokens,
                scheduler.config.max_prefill_tokens,
                decode_slots.len(),
                scheduler
                    .config
                    .prefill_max_requests
                    .unwrap_or(usize::MAX)
                    .min(
                        scheduler
                            .model
                            .max_concurrent_prefill_requests()
                            .unwrap_or(usize::MAX),
                    ),
            ),
            long_prefill_token_threshold: scheduler.config.long_prefill_token_threshold,
            decode_active: !decode_slots.is_empty(),
            page_budget: PageBudget::from_scheduler(scheduler, true),
        };
        for &slot_idx in &scheduler.running_batch {
            let remaining = scheduler.remaining_decode_reservation_tokens(slot_idx);
            if remaining > 0 {
                budget.page_budget.reserve_growth(PageGrowth {
                    slot_idx,
                    tokens: remaining,
                });
            }
        }
        budget
    }

    fn can_schedule(&self, reservation: PrefillReservation) -> bool {
        if !self.token_budget.can_fit(reservation.prefill_tokens) {
            return false;
        }
        self.page_budget.can_fit_growth(reservation.page_growth)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn reserve(&mut self, reservation: PrefillReservation) {
        debug_assert!(self.can_schedule(reservation));
        self.token_budget.reserve(reservation.prefill_tokens);
        self.page_budget.reserve_growth(reservation.page_growth);
    }
}

fn score_prefill_candidates(
    queued_candidates: Vec<PrefillCandidate>,
) -> Vec<ScoredPrefillCandidate> {
    let mut scored = Vec::with_capacity(queued_candidates.len());
    for (queue_rank, candidate) in queued_candidates.into_iter().enumerate() {
        scored.push(ScoredPrefillCandidate {
            candidate,
            score: PrefillCandidateScore { queue_rank },
        });
    }
    scored
}

fn select_prefill_candidates(
    budget: &mut PrefillBudget,
    mut scored_candidates: Vec<ScoredPrefillCandidate>,
) -> Vec<PrefillCandidate> {
    scored_candidates.sort_by_key(|scored| scored.score);
    let mut selected = Vec::with_capacity(scored_candidates.len());
    for scored in scored_candidates {
        let candidate = scored.candidate;
        if !budget.can_schedule(candidate.reservation) {
            continue;
        }
        budget.reserve(candidate.reservation);
        selected.push(candidate);
    }
    selected
}

fn cap_prefill_candidates_by_tokens(
    candidates: Vec<PrefillCandidate>,
    token_budget: usize,
) -> Vec<PrefillCandidate> {
    let mut remaining = token_budget;
    let mut capped = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if candidate.reservation.prefill_tokens > remaining {
            continue;
        }
        remaining -= candidate.reservation.prefill_tokens;
        capped.push(candidate);
    }
    capped
}

impl<M: ModelForward> Scheduler<M> {
    fn dispatch_decode_emits(&mut self) -> u128 {
        nvtx_scope!("step_dispatch_emits");
        let emit_t = std::time::Instant::now();
        let decode_slots: Vec<usize> = self.running_batch.iter().copied().collect();
        for slot_idx in decode_slots {
            let has_pending_emit = self
                .request(slot_idx)
                .is_some_and(super::request::ActiveRequest::has_pending_emit);
            if has_pending_emit {
                self.dispatch_emit(slot_idx);
            }
        }
        emit_t.elapsed().as_micros()
    }

    fn remaining_decode_reservation_tokens(&self, slot_idx: usize) -> usize {
        self.request(slot_idx).map_or(0, |req| {
            clipped_max_new_tokens_estimate(
                req.max_tokens.saturating_sub(req.generated_tokens.len()),
            )
        })
    }

    pub(super) fn runnable_decode_reservation_slots(&self) -> Vec<usize> {
        self.running_batch
            .iter()
            .filter(|&&slot_idx| self.slot_is_runnable_decode(slot_idx))
            .copied()
            .collect()
    }

    fn capped_prefill_reservation(
        &self,
        slot_idx: usize,
        prefill_token_cap: usize,
    ) -> Option<PrefillReservation> {
        if prefill_token_cap == 0 {
            return None;
        }
        let req = self.request(slot_idx)?;
        if req.delta_tx.is_closed() {
            return None;
        }
        let Phase::Prefilling {
            effective_tokens,
            progress,
        } = &req.phase
        else {
            return None;
        };

        let remaining_tokens = effective_tokens.len().saturating_sub(*progress);
        if remaining_tokens == 0 {
            return None;
        }

        let prefill_tokens = remaining_tokens.min(prefill_token_cap);
        let first_decode_token = if prefill_tokens >= remaining_tokens {
            usize::from(self.remaining_decode_reservation_tokens(slot_idx) > 0)
        } else {
            0
        };
        Some(PrefillReservation {
            prefill_tokens,
            page_growth: PageGrowth {
                slot_idx,
                tokens: prefill_tokens.saturating_add(first_decode_token),
            },
        })
    }

    fn prefill_reservation(
        &self,
        slot_idx: usize,
        decode_active: bool,
        long_prefill_token_threshold: usize,
    ) -> Option<PrefillReservation> {
        let per_request_cap = if decode_active {
            self.prefill_chunk_size().min(long_prefill_token_threshold)
        } else {
            self.prefill_chunk_size()
        };
        self.capped_prefill_reservation(slot_idx, per_request_cap.max(1))
    }

    pub(super) fn select_launch_prefill_candidates(
        &self,
        candidates: &[PrefillCandidate],
        decode_slots: &[usize],
    ) -> Vec<PrefillCandidate> {
        let mut budget = PrefillBudget::from_scheduler_for_decode_slots(self, decode_slots);
        let mut scored_candidates = Vec::with_capacity(candidates.len());
        for (queue_rank, candidate) in candidates.iter().enumerate() {
            let Some(reservation) = self.capped_prefill_reservation(
                candidate.slot_idx,
                candidate.reservation.prefill_tokens,
            ) else {
                continue;
            };
            scored_candidates.push(ScoredPrefillCandidate {
                candidate: PrefillCandidate {
                    slot_idx: candidate.slot_idx,
                    reservation,
                },
                score: PrefillCandidateScore { queue_rank },
            });
        }
        select_prefill_candidates(&mut budget, scored_candidates)
    }

    pub(super) fn select_mixed_launch_prefill_candidates(
        &self,
        candidates: &[PrefillCandidate],
        decode_slots: &[usize],
    ) -> Vec<PrefillCandidate> {
        cap_prefill_candidates_by_tokens(
            self.select_launch_prefill_candidates(candidates, decode_slots),
            self.config.mixed_prefill_token_budget(),
        )
    }

    fn collect_prefill_candidates(
        &mut self,
        budget: &PrefillBudget,
    ) -> Vec<ScoredPrefillCandidate> {
        let mut queued_candidates = Vec::new();
        let queued_slots: Vec<usize> = self.prefill_queue.iter().copied().collect();
        for slot_idx in queued_slots {
            if self
                .request(slot_idx)
                .is_some_and(|req| req.delta_tx.is_closed())
            {
                self.finish_slot(slot_idx);
                continue;
            }
            let Some(reservation) = self.prefill_reservation(
                slot_idx,
                budget.decode_active,
                budget.long_prefill_token_threshold,
            ) else {
                self.dequeue_prefill(slot_idx);
                continue;
            };
            queued_candidates.push(PrefillCandidate {
                slot_idx,
                reservation,
            });
        }
        score_prefill_candidates(queued_candidates)
    }

    fn plan_step(&mut self) -> StepPlan {
        let has_decode = self
            .running_batch
            .iter()
            .any(|&slot_idx| self.slot_is_runnable_decode(slot_idx));
        let mut budget = PrefillBudget::from_scheduler(self);
        let scored_candidates = self.collect_prefill_candidates(&budget);
        let candidates = select_prefill_candidates(&mut budget, scored_candidates);
        if has_decode {
            let plan = if candidates.is_empty() {
                StepPlan::Decode
            } else if self.model.supports_mixed_batch(self.paged_kv_pool.format) {
                StepPlan::Mixed(candidates)
            } else if self.config.short_prompt_bypass_tokens > 0
                && candidates.iter().all(|candidate| {
                    candidate.reservation.prefill_tokens <= self.config.short_prompt_bypass_tokens
                })
            {
                // Short prompts get their first sampled token from prefill
                // completion itself. Avoid the legacy decode+prefill split
                // launch for these requests; the small prefill runs as the
                // fused first-token path and decode rows resume next tick.
                StepPlan::Prefill(candidates)
            } else {
                // Keep the legacy split launches for models that do not have a
                // real single-launch mixed lowering yet.
                StepPlan::Split(candidates)
            };
            return route_spec_plan(self.config.spec_enabled, self.config.spec_draft_k, plan);
        }
        let plan = if candidates.is_empty() {
            StepPlan::Idle
        } else {
            StepPlan::Prefill(candidates)
        };
        route_spec_plan(self.config.spec_enabled, self.config.spec_draft_k, plan)
    }

    fn logical_decode_rows_for_current_batch(&self) -> Vec<LogicalDecodeRow> {
        let mut rows = Vec::with_capacity(self.running_batch.len());
        for &slot_idx in &self.running_batch {
            if !self.slot_is_runnable_decode(slot_idx) {
                continue;
            }
            let Some(req) = self.request(slot_idx) else {
                continue;
            };
            let Some(&input_token) = req.generated_tokens.last() else {
                continue;
            };
            let logical_kv_offset = req
                .prompt_tokens
                .len()
                .saturating_add(req.generated_tokens.len().saturating_sub(1));
            rows.push(LogicalDecodeRow::new(
                rows.len(),
                RequestId(req.id),
                input_token,
                logical_kv_offset,
            ));
        }
        rows
    }

    fn logical_prefill_rows_for_candidates(
        &self,
        candidates: &[PrefillCandidate],
        row_index_start: usize,
    ) -> Vec<LogicalPrefillRow> {
        let mut rows = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let Some(req) = self.request(candidate.slot_idx) else {
                continue;
            };
            let Phase::Prefilling {
                effective_tokens,
                progress,
            } = &req.phase
            else {
                continue;
            };
            let total_effective_tokens = effective_tokens.len();
            let chunk_end =
                (*progress + candidate.reservation.prefill_tokens).min(total_effective_tokens);
            if chunk_end <= *progress {
                continue;
            }

            let prompt_base = req
                .prompt_tokens
                .len()
                .saturating_sub(total_effective_tokens);
            let prompt_start = prompt_base + *progress;
            let prompt_end = prompt_base + chunk_end;
            rows.push(LogicalPrefillRow::with_logical_kv_span(
                row_index_start + rows.len(),
                RequestId(req.id),
                effective_tokens[*progress..chunk_end].to_vec(),
                prompt_start,
                prompt_end,
                req.prompt_tokens.len(),
                prompt_start,
                prompt_end,
            ));
        }
        rows
    }

    fn logical_shadow_plan_from_step_plan(&self, plan: &StepPlan) -> Option<LogicalServePlan> {
        match plan {
            StepPlan::SpecDecode => None,
            StepPlan::Idle => Some(LogicalServePlan::idle()),
            StepPlan::Decode => Some(LogicalServePlan::new(
                self.logical_decode_rows_for_current_batch(),
                Vec::new(),
            )),
            StepPlan::Prefill(candidates) => Some(LogicalServePlan::new(
                Vec::new(),
                self.logical_prefill_rows_for_candidates(candidates, 0),
            )),
            StepPlan::Split(candidates) | StepPlan::Mixed(candidates) => {
                let decode_rows = self.logical_decode_rows_for_current_batch();
                let prefill_rows =
                    self.logical_prefill_rows_for_candidates(candidates, decode_rows.len());
                Some(LogicalServePlan::new(decode_rows, prefill_rows))
            }
        }
    }

    fn maybe_log_logical_shadow_plan(&self, plan: &StepPlan) {
        if !logical_scheduler_shadow_enabled() {
            return;
        }
        let Some(logical_plan) = self.logical_shadow_plan_from_step_plan(plan) else {
            log::debug!(
                "cuda logical scheduler shadow skipped: plan={} spec_enabled={} sparse_kv={}",
                plan.label(),
                self.config.spec_enabled,
                self.config.spec_sparse_kv_enabled
            );
            return;
        };

        let decode_req_ids: Vec<RequestId> = logical_plan
            .decode_rows
            .iter()
            .map(|row| row.req_id)
            .collect();
        let prefill_req_ids: Vec<RequestId> = logical_plan
            .prefill_rows
            .iter()
            .map(|row| row.req_id)
            .collect();
        log::debug!(
            "cuda logical scheduler shadow: plan={} shape={:?} decode_req_ids={:?} prefill_req_ids={:?}",
            plan.label(),
            logical_plan.batch_shape,
            decode_req_ids,
            prefill_req_ids
        );
    }

    fn slot_for_request_id(&self, req_id: RequestId) -> Option<usize> {
        self.active.iter().enumerate().find_map(|(slot_idx, req)| {
            req.as_ref()
                .is_some_and(|active| RequestId(active.id) == req_id)
                .then_some(slot_idx)
        })
    }

    fn prefill_candidates_from_logical_plan(
        &self,
        logical_plan: &LogicalServePlan,
    ) -> Vec<PrefillCandidate> {
        let mut candidates = Vec::with_capacity(logical_plan.prefill_rows.len());
        for row in &logical_plan.prefill_rows {
            let Some(slot_idx) = self.slot_for_request_id(row.req_id) else {
                log::debug!(
                    "unified scheduler lowering skipped prefill row for missing request {:?}",
                    row.req_id
                );
                continue;
            };
            let Some(reservation) =
                self.capped_prefill_reservation(slot_idx, row.input_tokens.len())
            else {
                log::debug!(
                    "unified scheduler lowering skipped unschedulable prefill row for request {:?}",
                    row.req_id
                );
                continue;
            };
            candidates.push(PrefillCandidate {
                slot_idx,
                reservation,
            });
        }
        candidates
    }

    fn launch_legacy_step_plan(&mut self, plan: &StepPlan) -> (u128, u128) {
        match plan {
            StepPlan::Idle => (0, 0),
            StepPlan::Decode => {
                let t = std::time::Instant::now();
                self.step_decode_launch();
                (0, t.elapsed().as_micros())
            }
            StepPlan::SpecDecode => {
                let t = std::time::Instant::now();
                SpecPath::draft_then_verify(self, None);
                (0, t.elapsed().as_micros())
            }
            StepPlan::Prefill(candidates) => {
                let t = std::time::Instant::now();
                self.step_prefill_batch(candidates);
                (t.elapsed().as_micros(), 0)
            }
            StepPlan::Split(candidates) => {
                let prefill_t = std::time::Instant::now();
                self.step_prefill_batch(candidates);
                let prefill_us = prefill_t.elapsed().as_micros();
                let decode_t = std::time::Instant::now();
                self.step_decode_launch();
                (prefill_us, decode_t.elapsed().as_micros())
            }
            StepPlan::Mixed(candidates) => {
                let t = std::time::Instant::now();
                self.step_mixed_launch(candidates);
                (0, t.elapsed().as_micros())
            }
        }
    }

    fn launch_logical_serve_plan(&mut self, logical_plan: &LogicalServePlan) -> (u128, u128) {
        let has_decode = !logical_plan.decode_rows.is_empty();
        let prefill_candidates = self.prefill_candidates_from_logical_plan(logical_plan);
        match (has_decode, prefill_candidates.is_empty()) {
            (false, true) => (0, 0),
            (true, true) => {
                let t = std::time::Instant::now();
                self.step_decode_launch();
                (0, t.elapsed().as_micros())
            }
            (false, false) => {
                let t = std::time::Instant::now();
                self.step_prefill_batch(&prefill_candidates);
                (t.elapsed().as_micros(), 0)
            }
            (true, false) if self.model.supports_mixed_batch(self.paged_kv_pool.format) => {
                let t = std::time::Instant::now();
                self.step_mixed_launch(&prefill_candidates);
                (0, t.elapsed().as_micros())
            }
            (true, false) => {
                let prefill_t = std::time::Instant::now();
                self.step_prefill_batch(&prefill_candidates);
                let prefill_us = prefill_t.elapsed().as_micros();
                let decode_t = std::time::Instant::now();
                self.step_decode_launch();
                (prefill_us, decode_t.elapsed().as_micros())
            }
        }
    }

    fn launch_planned_step(&mut self, plan: &StepPlan) -> (u128, u128) {
        if cfg!(feature = "unified_scheduler")
            && let Some(logical_plan) = self.logical_shadow_plan_from_step_plan(plan)
        {
            if plan.expects_decode() && logical_plan.decode_rows.is_empty() {
                return self.launch_legacy_step_plan(plan);
            }
            self.launch_logical_serve_plan(&logical_plan)
        } else {
            self.launch_legacy_step_plan(plan)
        }
    }

    pub(super) fn step(&mut self, assign_us: u128) {
        let num = self.active_len();
        if num == 0 && self.waiting.is_empty() && !self.has_pending_gpu_work() {
            self.metrics.set_scheduler_step(0, 0, 0, 0, 0, 0);
            return;
        }

        let prefill_readback_us = if self.pending_prefill.is_some() {
            let t = std::time::Instant::now();
            self.step_prefill_readback();
            t.elapsed().as_micros()
        } else {
            0
        };

        // Read back the previous iteration's in-flight GPU work first.
        // `pending_prefill` / `pending_decode` live across loop turns so
        // `run()` can overlap the next round of intake/admission work with GPU
        // compute instead of launching and synchronizing in the same iteration.
        let readback_us = if self.pending_decode.is_some() {
            let t = std::time::Instant::now();
            self.step_decode_readback();
            t.elapsed().as_micros()
        } else {
            0
        };

        let (plan, plan_us) = {
            nvtx_scope!("step_plan");
            let plan_t = std::time::Instant::now();
            let plan = self.plan_step();
            (plan, plan_t.elapsed().as_micros())
        };
        let admission_us = assign_us + plan_us;
        let scheduled_prefill_rows = plan.scheduled_prefill_rows();
        let scheduled_prefill_tokens = plan.scheduled_prefill_tokens();
        self.metrics.record_scheduler_plan(plan.metrics_label());
        self.maybe_log_logical_shadow_plan(&plan);

        assert!(
            self.pending_decode.is_none(),
            "pending decode must be cleared before the next launch"
        );
        assert!(
            self.pending_prefill.is_none(),
            "pending prefill must be cleared before the next launch"
        );

        let (mut prefill_us, mut decode_launch_us) = self.launch_planned_step(&plan);
        let scheduled_decode_rows = self
            .pending_decode
            .as_ref()
            .map_or(0, |pending| pending.decode_indices.len() as u64);
        let mixed_prefill_rows = self
            .pending_decode
            .as_ref()
            .and_then(|pending| pending.mixed_prefill.as_ref())
            .map_or(0, |pending| pending.rows.len());
        if mixed_prefill_rows > 0 {
            prefill_us += decode_launch_us;
            decode_launch_us = 0;
        }
        let emit_us = self.dispatch_decode_emits();
        let decode_us = decode_launch_us + readback_us;
        let scheduled_rows = scheduled_decode_rows + scheduled_prefill_rows;

        let total_us = decode_us + emit_us + admission_us + prefill_readback_us + prefill_us;
        self.metrics.set_scheduler_step(
            scheduled_rows,
            scheduled_decode_rows,
            scheduled_prefill_rows,
            scheduled_decode_rows,
            scheduled_prefill_tokens,
            scheduled_rows,
        );
        self.metrics
            .observe_scheduler_step(total_us as f64 / 1_000_000.0);
        let update_ema = |ema: &mut f64, val: u128| {
            const ALPHA: f64 = 0.1;
            let v = val as f64;
            if *ema == 0.0 {
                *ema = v;
            } else {
                *ema = ALPHA * v + (1.0 - ALPHA) * *ema;
            }
        };
        update_ema(&mut self.stats.step_timing_admission_us, admission_us);
        update_ema(&mut self.stats.step_timing_decode_us, decode_us);
        update_ema(&mut self.stats.step_timing_emit_us, emit_us);
        update_ema(
            &mut self.stats.step_timing_prefill_us,
            prefill_readback_us + prefill_us,
        );
        update_ema(&mut self.stats.step_timing_total_us, total_us);
        self.metrics.set_scheduler_step_phase_us(
            self.stats.step_timing_admission_us,
            self.stats.step_timing_prefill_us,
            self.stats.step_timing_decode_us,
            self.stats.step_timing_emit_us,
            self.stats.step_timing_total_us,
        );

        if total_us > 100_000 && !plan.is_idle() {
            info!(
                "step breakdown: plan={} admission={}us decode={}us emit={}us prefill={}us total={}us batch={}",
                plan.label(),
                admission_us,
                decode_us,
                emit_us,
                prefill_readback_us + prefill_us,
                total_us,
                num
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PrefillBudget, PrefillCandidate, PrefillCandidateScore, PrefillReservation,
        ScoredPrefillCandidate, StepPlan, cap_prefill_candidates_by_tokens, route_spec_plan,
        score_prefill_candidates, select_prefill_candidates,
    };
    use crate::scheduler::cuda::budget::{PageBudget, PageGrowth, StepTokenBudget};

    fn collect_schedulable_indices(
        mut budget: PrefillBudget,
        reservations: &[(usize, PrefillReservation)],
    ) -> Vec<usize> {
        let candidates = reservations
            .iter()
            .map(|&(idx, reservation)| PrefillCandidate {
                slot_idx: idx,
                reservation,
            })
            .collect();
        let scored_candidates = score_prefill_candidates(candidates);
        select_prefill_candidates(&mut budget, scored_candidates)
            .into_iter()
            .map(|candidate| candidate.slot_idx)
            .collect()
    }

    #[test]
    fn budget_skips_first_token_budget_miss() {
        let budget = PrefillBudget {
            token_budget: StepTokenBudget::new(6, usize::MAX),
            long_prefill_token_threshold: usize::MAX,
            decode_active: false,
            page_budget: PageBudget::new(usize::MAX, vec![0, 0, 0], 16, true),
        };
        let reservations = vec![
            (
                0,
                PrefillReservation {
                    prefill_tokens: 8,
                    page_growth: PageGrowth {
                        slot_idx: 0,
                        tokens: 8,
                    },
                },
            ),
            (
                1,
                PrefillReservation {
                    prefill_tokens: 4,
                    page_growth: PageGrowth {
                        slot_idx: 1,
                        tokens: 4,
                    },
                },
            ),
            (
                2,
                PrefillReservation {
                    prefill_tokens: 2,
                    page_growth: PageGrowth {
                        slot_idx: 2,
                        tokens: 2,
                    },
                },
            ),
        ];

        assert_eq!(
            collect_schedulable_indices(budget, &reservations),
            vec![1, 2]
        );
    }

    #[test]
    fn budget_skips_first_page_budget_miss() {
        let budget = PrefillBudget {
            token_budget: StepTokenBudget::new(8, usize::MAX),
            long_prefill_token_threshold: usize::MAX,
            decode_active: false,
            page_budget: PageBudget::new(1, vec![0, 3], 4, true),
        };
        let reservations = vec![
            (
                0,
                PrefillReservation {
                    prefill_tokens: 5,
                    page_growth: PageGrowth {
                        slot_idx: 0,
                        tokens: 5,
                    },
                },
            ),
            (
                1,
                PrefillReservation {
                    prefill_tokens: 1,
                    page_growth: PageGrowth {
                        slot_idx: 1,
                        tokens: 1,
                    },
                },
            ),
        ];

        assert_eq!(collect_schedulable_indices(budget, &reservations), vec![1]);
    }

    #[test]
    fn budget_reserves_decode_headroom_for_prefill_completion() {
        let budget = PrefillBudget {
            token_budget: StepTokenBudget::new(4, 1),
            long_prefill_token_threshold: usize::MAX,
            decode_active: false,
            page_budget: PageBudget::new(1, vec![4], 4, true),
        };
        let reservation = PrefillReservation {
            prefill_tokens: 4,
            page_growth: PageGrowth {
                slot_idx: 0,
                tokens: 5,
            },
        };

        assert!(!budget.can_schedule(reservation));
    }

    #[test]
    fn page_budget_reserves_running_decode_headroom_before_prefill() {
        let mut page_budget = PageBudget::new(1, vec![4, 0], 4, true);
        page_budget.reserve_growth(PageGrowth {
            slot_idx: 0,
            tokens: 1,
        });

        assert!(!page_budget.can_fit_growth(PageGrowth {
            slot_idx: 1,
            tokens: 4,
        }));
    }

    #[test]
    fn budget_honors_whole_step_token_cap_before_prefill_cap() {
        let budget = PrefillBudget {
            token_budget: StepTokenBudget::new(3, usize::MAX),
            long_prefill_token_threshold: usize::MAX,
            decode_active: true,
            page_budget: PageBudget::new(usize::MAX, vec![0, 0], 16, true),
        };
        let reservations = vec![
            (
                0,
                PrefillReservation {
                    prefill_tokens: 4,
                    page_growth: PageGrowth {
                        slot_idx: 0,
                        tokens: 4,
                    },
                },
            ),
            (
                1,
                PrefillReservation {
                    prefill_tokens: 2,
                    page_growth: PageGrowth {
                        slot_idx: 1,
                        tokens: 2,
                    },
                },
            ),
        ];

        assert_eq!(collect_schedulable_indices(budget, &reservations), vec![1]);
    }

    #[test]
    fn score_prefill_candidates_preserves_queue_order_as_canonical_rank() {
        let reservations = vec![
            PrefillCandidate {
                slot_idx: 9,
                reservation: PrefillReservation {
                    prefill_tokens: 1,
                    page_growth: PageGrowth {
                        slot_idx: 9,
                        tokens: 1,
                    },
                },
            },
            PrefillCandidate {
                slot_idx: 3,
                reservation: PrefillReservation {
                    prefill_tokens: 2,
                    page_growth: PageGrowth {
                        slot_idx: 3,
                        tokens: 2,
                    },
                },
            },
        ];

        let scored = score_prefill_candidates(reservations);
        assert_eq!(
            scored
                .iter()
                .map(|candidate| candidate.score)
                .collect::<Vec<_>>(),
            vec![
                PrefillCandidateScore { queue_rank: 0 },
                PrefillCandidateScore { queue_rank: 1 }
            ]
        );
    }

    #[test]
    fn mixed_prefill_cap_preserves_queue_order_and_skips_oversized_rows() {
        let candidates = vec![
            PrefillCandidate {
                slot_idx: 0,
                reservation: PrefillReservation {
                    prefill_tokens: 3,
                    page_growth: PageGrowth {
                        slot_idx: 0,
                        tokens: 3,
                    },
                },
            },
            PrefillCandidate {
                slot_idx: 1,
                reservation: PrefillReservation {
                    prefill_tokens: 4,
                    page_growth: PageGrowth {
                        slot_idx: 1,
                        tokens: 4,
                    },
                },
            },
            PrefillCandidate {
                slot_idx: 2,
                reservation: PrefillReservation {
                    prefill_tokens: 2,
                    page_growth: PageGrowth {
                        slot_idx: 2,
                        tokens: 2,
                    },
                },
            },
        ];

        let selected = cap_prefill_candidates_by_tokens(candidates, 5);

        assert_eq!(
            selected
                .into_iter()
                .map(|candidate| candidate.slot_idx)
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
    }

    #[test]
    fn select_prefill_candidates_orders_by_score_before_fit() {
        let mut budget = PrefillBudget {
            token_budget: StepTokenBudget::new(3, usize::MAX),
            long_prefill_token_threshold: usize::MAX,
            decode_active: false,
            page_budget: PageBudget::new(usize::MAX, vec![0, 0], 16, true),
        };
        let selected = select_prefill_candidates(
            &mut budget,
            vec![
                ScoredPrefillCandidate {
                    candidate: PrefillCandidate {
                        slot_idx: 1,
                        reservation: PrefillReservation {
                            prefill_tokens: 2,
                            page_growth: PageGrowth {
                                slot_idx: 1,
                                tokens: 2,
                            },
                        },
                    },
                    score: PrefillCandidateScore { queue_rank: 1 },
                },
                ScoredPrefillCandidate {
                    candidate: PrefillCandidate {
                        slot_idx: 0,
                        reservation: PrefillReservation {
                            prefill_tokens: 2,
                            page_growth: PageGrowth {
                                slot_idx: 0,
                                tokens: 2,
                            },
                        },
                    },
                    score: PrefillCandidateScore { queue_rank: 0 },
                },
            ],
        );

        assert_eq!(
            selected
                .into_iter()
                .map(|candidate| candidate.slot_idx)
                .collect::<Vec<_>>(),
            vec![0]
        );
    }

    #[test]
    fn spec_disabled_route_returns_existing_step_plan() {
        let plan = route_spec_plan(false, 1, StepPlan::Decode);
        assert!(matches!(plan, StepPlan::Decode));
    }

    #[test]
    fn spec_enabled_single_token_route_uses_verifier_step_plan() {
        let plan = route_spec_plan(true, 1, StepPlan::Decode);
        assert!(matches!(plan, StepPlan::SpecDecode));
    }

    #[test]
    fn multi_token_spec_route_waits_for_real_verifier() {
        let plan = route_spec_plan(true, 5, StepPlan::Decode);
        assert!(matches!(plan, StepPlan::SpecDecode));
    }
}
