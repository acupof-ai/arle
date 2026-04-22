use super::{ModelForward, Phase, Scheduler, info};

#[derive(Clone, Copy, Debug)]
pub(super) struct PrefillReservation {
    pub prefill_tokens: usize,
    pub page_reserve: SlotPageReserve,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct SlotPageReserve {
    pub slot_idx: usize,
    pub prefill_pool_tokens: usize,
    pub decode_headroom_tokens: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PrefillCandidate {
    pub slot_idx: usize,
    pub reservation: PrefillReservation,
}

#[derive(Clone, Debug, Default)]
struct StepPlan {
    launch_decode: bool,
    prefill: Vec<PrefillCandidate>,
}

impl StepPlan {
    fn label(&self) -> &'static str {
        match (self.launch_decode, self.prefill.is_empty()) {
            (false, true) => "idle",
            (true, true) => "decode",
            (false, false) => "prefill",
            (true, false) => "decode+prefill",
        }
    }

    fn is_idle(&self) -> bool {
        !self.launch_decode && self.prefill.is_empty()
    }
}

#[derive(Debug)]
struct PrefillBudget {
    remaining_step_tokens: usize,
    remaining_prefill_tokens: usize,
    remaining_requests: usize,
    long_prefill_token_threshold: usize,
    decode_active: bool,
    page_budget: PrefillPageBudget,
}

#[derive(Debug)]
struct PrefillPageBudget {
    remaining_free_pages: usize,
    planned_seq_lens: Vec<usize>,
    page_size: usize,
}

impl PrefillPageBudget {
    fn new(remaining_free_pages: usize, planned_seq_lens: Vec<usize>, page_size: usize) -> Self {
        Self {
            remaining_free_pages,
            planned_seq_lens,
            page_size,
        }
    }

    fn can_fit(&self, reserve: SlotPageReserve) -> bool {
        self.additional_pages_needed(reserve) <= self.remaining_free_pages
    }

    fn reserve_running_decode_headroom(&mut self, slot_idx: usize, decode_tokens: usize) {
        let reserve = SlotPageReserve {
            slot_idx,
            prefill_pool_tokens: 0,
            decode_headroom_tokens: decode_tokens,
        };
        let new_pages = self.additional_pages_needed(reserve);
        debug_assert!(new_pages <= self.remaining_free_pages);
        self.remaining_free_pages = self.remaining_free_pages.saturating_sub(new_pages);
        self.planned_seq_lens[slot_idx] =
            self.planned_seq_lens[slot_idx].saturating_add(decode_tokens);
    }

    fn additional_pages_needed(&self, reserve: SlotPageReserve) -> usize {
        let total_tokens = reserve
            .prefill_pool_tokens
            .saturating_add(reserve.decode_headroom_tokens);
        if total_tokens == 0 {
            return 0;
        }
        let current_pages = self.planned_seq_lens[reserve.slot_idx].div_ceil(self.page_size);
        let future_pages = self.planned_seq_lens[reserve.slot_idx]
            .saturating_add(total_tokens)
            .div_ceil(self.page_size);
        future_pages.saturating_sub(current_pages)
    }
}

impl PrefillBudget {
    fn from_scheduler<M: ModelForward>(scheduler: &Scheduler<M>) -> Self {
        let running_decode_tokens = scheduler
            .running_batch
            .iter()
            .filter(|&&slot_idx| {
                scheduler.request(slot_idx).is_some_and(|req| {
                    matches!(req.phase, Phase::Decoding) && !req.delta_tx.is_closed()
                })
            })
            .count();
        let remaining_step_tokens = scheduler
            .config
            .max_num_batched_tokens
            .saturating_sub(running_decode_tokens);
        let mut budget = Self {
            remaining_step_tokens,
            remaining_prefill_tokens: scheduler
                .config
                .max_prefill_tokens
                .min(remaining_step_tokens),
            remaining_requests: scheduler.config.prefill_max_requests.unwrap_or(usize::MAX),
            long_prefill_token_threshold: scheduler.config.long_prefill_token_threshold,
            decode_active: running_decode_tokens > 0,
            page_budget: PrefillPageBudget::new(
                scheduler.pool_free_pages(),
                (0..scheduler.states.len())
                    .map(|slot_idx| scheduler.paged_kv_pool.seq_len(slot_idx))
                    .collect(),
                scheduler.paged_kv_pool.page_size.max(1),
            ),
        };
        for &slot_idx in &scheduler.running_batch {
            let remaining = scheduler.remaining_decode_tokens(slot_idx);
            if remaining > 0 {
                budget
                    .page_budget
                    .reserve_running_decode_headroom(slot_idx, remaining);
            }
        }
        budget
    }

    fn can_schedule(&self, reservation: PrefillReservation) -> bool {
        if reservation.prefill_tokens == 0
            || reservation.prefill_tokens > self.remaining_step_tokens
            || reservation.prefill_tokens > self.remaining_prefill_tokens
            || self.remaining_requests == 0
        {
            return false;
        }
        self.page_budget.can_fit(reservation.page_reserve)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn reserve(&mut self, reservation: PrefillReservation) {
        debug_assert!(self.can_schedule(reservation));
        self.remaining_step_tokens = self
            .remaining_step_tokens
            .saturating_sub(reservation.prefill_tokens);
        self.remaining_prefill_tokens = self
            .remaining_prefill_tokens
            .saturating_sub(reservation.prefill_tokens);
        self.remaining_requests = self.remaining_requests.saturating_sub(1);
        let new_pages = self
            .page_budget
            .additional_pages_needed(reservation.page_reserve);
        self.page_budget.remaining_free_pages = self
            .page_budget
            .remaining_free_pages
            .saturating_sub(new_pages);
        self.page_budget.planned_seq_lens[reservation.page_reserve.slot_idx] =
            self.page_budget.planned_seq_lens[reservation.page_reserve.slot_idx]
                .saturating_add(reservation.page_reserve.prefill_pool_tokens)
                .saturating_add(reservation.page_reserve.decode_headroom_tokens);
    }
}

fn select_prefill_candidates(
    budget: &mut PrefillBudget,
    queued_candidates: Vec<PrefillCandidate>,
) -> Vec<PrefillCandidate> {
    let mut selected = Vec::with_capacity(queued_candidates.len());
    for candidate in queued_candidates {
        if !budget.can_schedule(candidate.reservation) {
            continue;
        }
        budget.reserve(candidate.reservation);
        selected.push(candidate);
    }
    selected
}

impl<M: ModelForward> Scheduler<M> {
    fn remaining_decode_tokens(&self, slot_idx: usize) -> usize {
        self.request(slot_idx)
            .map(|req| req.max_tokens.saturating_sub(req.generated_tokens.len()))
            .unwrap_or(0)
    }

    fn prefill_reservation(
        &self,
        slot_idx: usize,
        decode_active: bool,
        long_prefill_token_threshold: usize,
    ) -> Option<PrefillReservation> {
        let req = self.request(slot_idx)?;
        if req.delta_tx.is_closed() {
            return None;
        }
        let Phase::Prefilling {
            materialized_prefix_len: _,
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

        let per_request_cap = if decode_active {
            self.prefill_chunk_size().min(long_prefill_token_threshold)
        } else {
            self.prefill_chunk_size()
        };
        let prefill_tokens = remaining_tokens.min(per_request_cap.max(1));
        Some(PrefillReservation {
            prefill_tokens,
            page_reserve: SlotPageReserve {
                slot_idx,
                prefill_pool_tokens: prefill_tokens,
                decode_headroom_tokens: req.max_tokens.saturating_sub(req.generated_tokens.len()),
            },
        })
    }

    fn collect_prefill_candidates(&mut self, budget: &mut PrefillBudget) -> Vec<PrefillCandidate> {
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
        select_prefill_candidates(budget, queued_candidates)
    }

    fn plan_step(&mut self) -> StepPlan {
        let has_decode = self.running_batch.iter().any(|&slot_idx| {
            self.request(slot_idx).is_some_and(|req| {
                matches!(req.phase, Phase::Decoding) && !req.delta_tx.is_closed()
            })
        });
        let mut budget = PrefillBudget::from_scheduler(self);
        let candidates = self.collect_prefill_candidates(&mut budget);
        StepPlan {
            launch_decode: has_decode,
            prefill: candidates,
        }
    }

    pub(super) fn step(&mut self) {
        let num = self.active_len();
        if num == 0 && self.waiting.is_empty() && self.pending_decode.is_none() {
            return;
        }

        // Read back the previous iteration's decode first. This keeps
        // `pending_decode` live across loop turns so `run()` can overlap the
        // next round of intake/admission work with GPU compute instead of
        // launching and synchronizing in the same iteration.
        let readback_us = if self.pending_decode.is_some() {
            let t = std::time::Instant::now();
            self.step_decode_readback();
            t.elapsed().as_micros()
        } else {
            0
        };

        let emit_t = std::time::Instant::now();
        let decode_slots: Vec<usize> = self.running_batch.iter().copied().collect();
        {
            let Self {
                active, tokenizer, ..
            } = self;
            for slot_idx in decode_slots {
                let Some(req) = active[slot_idx].as_mut() else {
                    continue;
                };
                if matches!(req.phase, Phase::Decoding)
                    && req.decoded_token_count < req.generated_tokens.len()
                {
                    req.emit_delta(tokenizer);
                }
            }
        }
        let emit_us = emit_t.elapsed().as_micros();

        let plan_t = std::time::Instant::now();
        let plan = self.plan_step();
        let admission_us = plan_t.elapsed().as_micros();

        assert!(
            self.pending_decode.is_none(),
            "pending decode must be cleared before the next launch"
        );

        let prefill_t = std::time::Instant::now();
        if !plan.prefill.is_empty() {
            self.step_prefill_batch(&plan.prefill);
        }
        let prefill_us = prefill_t.elapsed().as_micros();

        let decode_launch_us = if plan.launch_decode {
            let t = std::time::Instant::now();
            self.step_decode_launch();
            t.elapsed().as_micros()
        } else {
            0
        };
        let decode_us = decode_launch_us + readback_us;

        let total_us = decode_us + emit_us + admission_us + prefill_us;
        let update_ema = |ema: &mut f64, val: u128| {
            const ALPHA: f64 = 0.1;
            let v = val as f64;
            if *ema == 0.0 {
                *ema = v;
            } else {
                *ema = ALPHA * v + (1.0 - ALPHA) * *ema;
            }
        };
        update_ema(&mut self.step_timing_decode_us, decode_us);
        update_ema(&mut self.step_timing_emit_us, emit_us);
        update_ema(&mut self.step_timing_prefill_us, prefill_us);
        update_ema(&mut self.step_timing_total_us, total_us);

        if total_us > 100_000 && !plan.is_idle() {
            info!(
                "step breakdown: plan={} admission={}us decode={}us emit={}us prefill={}us total={}us batch={}",
                plan.label(),
                admission_us,
                decode_us,
                emit_us,
                prefill_us,
                total_us,
                num
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PrefillBudget, PrefillCandidate, PrefillPageBudget, PrefillReservation, SlotPageReserve,
        select_prefill_candidates,
    };

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
        select_prefill_candidates(&mut budget, candidates)
            .into_iter()
            .map(|candidate| candidate.slot_idx)
            .collect()
    }

    #[test]
    fn budget_skips_first_token_budget_miss() {
        let budget = PrefillBudget {
            remaining_step_tokens: 6,
            remaining_prefill_tokens: 6,
            remaining_requests: usize::MAX,
            long_prefill_token_threshold: usize::MAX,
            decode_active: false,
            page_budget: PrefillPageBudget::new(usize::MAX, vec![0, 0, 0], 16),
        };
        let reservations = vec![
            (
                0,
                PrefillReservation {
                    slot_idx: 0,
                    prefill_tokens: 8,
                    page_reserve: SlotPageReserve {
                        slot_idx: 0,
                        prefill_pool_tokens: 8,
                        decode_headroom_tokens: 0,
                    },
                },
            ),
            (
                1,
                PrefillReservation {
                    slot_idx: 1,
                    prefill_tokens: 4,
                    page_reserve: SlotPageReserve {
                        slot_idx: 1,
                        prefill_pool_tokens: 4,
                        decode_headroom_tokens: 0,
                    },
                },
            ),
            (
                2,
                PrefillReservation {
                    slot_idx: 2,
                    prefill_tokens: 2,
                    page_reserve: SlotPageReserve {
                        slot_idx: 2,
                        prefill_pool_tokens: 2,
                        decode_headroom_tokens: 0,
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
            remaining_step_tokens: 8,
            remaining_prefill_tokens: 8,
            remaining_requests: usize::MAX,
            long_prefill_token_threshold: usize::MAX,
            decode_active: false,
            page_budget: PrefillPageBudget::new(1, vec![0, 3], 4),
        };
        let reservations = vec![
            (
                0,
                PrefillReservation {
                    slot_idx: 0,
                    prefill_tokens: 5,
                    page_reserve: SlotPageReserve {
                        slot_idx: 0,
                        prefill_pool_tokens: 5,
                        decode_headroom_tokens: 0,
                    },
                },
            ),
            (
                1,
                PrefillReservation {
                    slot_idx: 1,
                    prefill_tokens: 1,
                    page_reserve: SlotPageReserve {
                        slot_idx: 1,
                        prefill_pool_tokens: 1,
                        decode_headroom_tokens: 0,
                    },
                },
            ),
        ];

        assert_eq!(collect_schedulable_indices(budget, &reservations), vec![1]);
    }

    #[test]
    fn budget_reserves_decode_headroom_for_prefill_completion() {
        let budget = PrefillBudget {
            remaining_step_tokens: 4,
            remaining_prefill_tokens: 4,
            remaining_requests: 1,
            long_prefill_token_threshold: usize::MAX,
            decode_active: false,
            page_budget: PrefillPageBudget::new(1, vec![4], 4),
        };
        let reservation = PrefillReservation {
            slot_idx: 0,
            prefill_tokens: 4,
            page_reserve: SlotPageReserve {
                slot_idx: 0,
                prefill_pool_tokens: 4,
                decode_headroom_tokens: 1,
            },
        };

        assert!(!budget.can_schedule(reservation));
    }

    #[test]
    fn page_budget_reserves_running_decode_headroom_before_prefill() {
        let mut page_budget = PrefillPageBudget::new(1, vec![4, 0], 4);
        page_budget.reserve_running_decode_headroom(0, 1);

        assert!(!page_budget.can_fit(SlotPageReserve {
            slot_idx: 1,
            prefill_pool_tokens: 4,
            decode_headroom_tokens: 0,
        }));
    }

    #[test]
    fn budget_honors_whole_step_token_cap_before_prefill_cap() {
        let budget = PrefillBudget {
            remaining_step_tokens: 3,
            remaining_prefill_tokens: 8,
            remaining_requests: usize::MAX,
            long_prefill_token_threshold: usize::MAX,
            decode_active: true,
            page_budget: PrefillPageBudget::new(usize::MAX, vec![0, 0], 16),
        };
        let reservations = vec![
            (
                0,
                PrefillReservation {
                    slot_idx: 0,
                    prefill_tokens: 4,
                    page_reserve: SlotPageReserve {
                        slot_idx: 0,
                        prefill_pool_tokens: 4,
                        decode_headroom_tokens: 0,
                    },
                },
            ),
            (
                1,
                PrefillReservation {
                    slot_idx: 1,
                    prefill_tokens: 2,
                    page_reserve: SlotPageReserve {
                        slot_idx: 1,
                        prefill_pool_tokens: 2,
                        decode_headroom_tokens: 0,
                    },
                },
            ),
        ];

        assert_eq!(collect_schedulable_indices(budget, &reservations), vec![1]);
    }
}
