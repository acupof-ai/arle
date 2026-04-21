use super::{ModelForward, Phase, Scheduler, info};

#[derive(Clone, Copy, Debug)]
struct PrefillReservation {
    slot_idx: usize,
    prefill_tokens: usize,
    pool_tokens: usize,
    decode_headroom_tokens: usize,
}

#[derive(Debug)]
struct PrefillBudget {
    remaining_prefill_tokens: usize,
    remaining_requests: usize,
    remaining_free_pages: usize,
    planned_seq_lens: Vec<usize>,
    page_size: usize,
}

impl PrefillBudget {
    fn from_scheduler<M: ModelForward>(scheduler: &Scheduler<M>) -> Self {
        let mut budget = Self {
            remaining_prefill_tokens: scheduler.config.max_prefill_tokens,
            remaining_requests: scheduler.config.prefill_max_requests.unwrap_or(usize::MAX),
            remaining_free_pages: scheduler.pool_free_pages(),
            planned_seq_lens: (0..scheduler.states.len())
                .map(|slot_idx| scheduler.paged_kv_pool.seq_len(slot_idx))
                .collect(),
            page_size: scheduler.paged_kv_pool.page_size.max(1),
        };
        for &slot_idx in &scheduler.running_batch {
            if scheduler
                .request(slot_idx)
                .is_some_and(|req| req.generated_tokens.len() < req.max_tokens)
            {
                budget.reserve_decode_headroom(slot_idx, 1);
            }
        }
        budget
    }

    fn can_schedule(&self, reservation: PrefillReservation) -> bool {
        if reservation.prefill_tokens == 0
            || reservation.prefill_tokens > self.remaining_prefill_tokens
            || self.remaining_requests == 0
        {
            return false;
        }
        self.additional_pages_needed(
            reservation.slot_idx,
            reservation.pool_tokens,
            reservation.decode_headroom_tokens,
        ) <= self.remaining_free_pages
    }

    fn reserve(&mut self, reservation: PrefillReservation) {
        debug_assert!(self.can_schedule(reservation));
        self.remaining_prefill_tokens = self
            .remaining_prefill_tokens
            .saturating_sub(reservation.prefill_tokens);
        self.remaining_requests = self.remaining_requests.saturating_sub(1);
        self.reserve_tokens(
            reservation.slot_idx,
            reservation.pool_tokens,
            reservation.decode_headroom_tokens,
        );
    }

    fn reserve_mixed_prefill(&mut self, prefill_tokens: usize) {
        if prefill_tokens == 0 {
            return;
        }
        self.remaining_prefill_tokens =
            self.remaining_prefill_tokens.saturating_sub(prefill_tokens);
        self.remaining_requests = self.remaining_requests.saturating_sub(1);
    }

    fn reserve_decode_headroom(&mut self, slot_idx: usize, decode_tokens: usize) {
        if decode_tokens == 0 {
            return;
        }
        let new_pages = self.additional_pages_needed(slot_idx, 0, decode_tokens);
        debug_assert!(new_pages <= self.remaining_free_pages);
        self.remaining_free_pages = self.remaining_free_pages.saturating_sub(new_pages);
        self.planned_seq_lens[slot_idx] =
            self.planned_seq_lens[slot_idx].saturating_add(decode_tokens);
    }

    fn reserve_tokens(
        &mut self,
        slot_idx: usize,
        pool_tokens: usize,
        decode_headroom_tokens: usize,
    ) {
        let new_pages = self.additional_pages_needed(slot_idx, pool_tokens, decode_headroom_tokens);
        debug_assert!(new_pages <= self.remaining_free_pages);
        self.remaining_free_pages = self.remaining_free_pages.saturating_sub(new_pages);
        self.planned_seq_lens[slot_idx] = self.planned_seq_lens[slot_idx]
            .saturating_add(pool_tokens)
            .saturating_add(decode_headroom_tokens);
    }

    fn additional_pages_needed(
        &self,
        slot_idx: usize,
        pool_tokens: usize,
        decode_headroom_tokens: usize,
    ) -> usize {
        let total_tokens = pool_tokens.saturating_add(decode_headroom_tokens);
        if total_tokens == 0 {
            return 0;
        }
        let current_pages = self.planned_seq_lens[slot_idx].div_ceil(self.page_size);
        let future_pages = self.planned_seq_lens[slot_idx]
            .saturating_add(total_tokens)
            .div_ceil(self.page_size);
        future_pages.saturating_sub(current_pages)
    }
}

impl<M: ModelForward> Scheduler<M> {
    fn prefill_reservation(&self, slot_idx: usize) -> Option<PrefillReservation> {
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

        let prefill_tokens = remaining_tokens.min(self.prefill_chunk_size());
        let pool_tokens = if self.model.prefill_uses_paged_pool() && self.paged_kv_pool.is_active()
        {
            prefill_tokens
        } else {
            0
        };
        Some(PrefillReservation {
            slot_idx,
            prefill_tokens,
            pool_tokens,
            decode_headroom_tokens: usize::from(req.generated_tokens.len() < req.max_tokens),
        })
    }

    pub(super) fn step(&mut self) {
        let num = self.active_len();
        if num == 0 {
            return;
        }

        // Phase 1: Batched decode (GPU forward pass + sampling)
        //
        // Launched BEFORE emit_delta so that GPU compute overlaps with
        // CPU tokenizer work in Phase 2. emit_delta only needs tokens
        // from the PREVIOUS step (already in generated_tokens), so
        // there's no data dependency with the current forward pass.
        let has_decode = !self.running_batch.is_empty();
        assert!(
            self.pending_decode.is_none(),
            "pending decode must be cleared before scheduler step"
        );
        self.pending_mixed_prefill_idx = None;
        let decode_launch_us = if has_decode {
            let t = std::time::Instant::now();
            let mixed_prefill_idx = if self.config.enable_mixed_chunk {
                self.prefill_queue.iter().copied().find(|&slot_idx| {
                    self.request(slot_idx)
                        .is_some_and(|req| !req.delta_tx.is_closed())
                })
            } else {
                None
            };
            if let Some(pi) = mixed_prefill_idx {
                self.step_decode_launch_mixed(pi);
            } else {
                self.step_decode_launch();
            }
            t.elapsed().as_micros()
        } else {
            0
        };

        // Phase 2: Overlap window — GPU decode forward is in flight.
        //
        // Queue CPU work + prefill on the same stream while decode runs:
        //   a) emit_delta (CPU tokenizer, no GPU dependency)
        //   b) admit new requests (CPU)
        //   c) prefill chunks — GPU kernels queued AFTER decode on same stream,
        //      eliminating the dead time that existed when prefill ran after readback.
        //      When step_prefill_chunk internally syncs (for sampling), that sync
        //      catches both decode argmax AND prefill GPU work. The subsequent
        //      decode readback sync becomes a no-op.
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

        // Admissions happen in `assign_slots`; the runtime no longer
        // carries a transient "new request" phase through `step()`.
        let new_us = 0;

        // Phase 2c: Prefill chunks — queued on the same GPU stream after decode.
        // Decode always launches first; prefill work is then bounded by explicit
        // SGLang-style token/request budgets instead of a decode-active hard cap.
        let prefill_t = std::time::Instant::now();
        let already_mixed = self.pending_mixed_prefill_idx.take();
        let mixed_prefill_tokens = self
            .pending_decode
            .as_ref()
            .map_or(0, |pending| pending.mixed_prefill_tokens);
        let mut budget = PrefillBudget::from_scheduler(self);
        budget.reserve_mixed_prefill(mixed_prefill_tokens);
        let prefill_indices: Vec<usize> = self
            .prefill_queue
            .iter()
            .copied()
            .filter(|&slot_idx| Some(slot_idx) != already_mixed)
            .collect();
        for slot_idx in prefill_indices {
            let Some(reservation) = self.prefill_reservation(slot_idx) else {
                continue;
            };
            if !budget.can_schedule(reservation) {
                break;
            }
            let scheduled = self.step_prefill_chunk(slot_idx, reservation.prefill_tokens);
            if scheduled == 0 {
                break;
            }
            budget.reserve(PrefillReservation {
                prefill_tokens: scheduled,
                pool_tokens: if self.model.prefill_uses_paged_pool()
                    && self.paged_kv_pool.is_active()
                {
                    scheduled
                } else {
                    0
                },
                ..reservation
            });
        }
        let prefill_us = prefill_t.elapsed().as_micros();

        // Phase 3: Readback decode results (sync + D2H + token mutation).
        // If prefill already synced the stream, this sync is a no-op.
        let readback_us = if has_decode && self.pending_decode.is_some() {
            let t = std::time::Instant::now();
            self.step_decode_readback();
            t.elapsed().as_micros()
        } else {
            0
        };
        let decode_us = decode_launch_us + readback_us;

        // Step timing — always tracked for /v1/stats and profiling.
        // EMA (exponential moving average) with α=0.1 smooths noise
        // while responding quickly to sustained changes.
        let total_us = decode_us + emit_us + new_us + prefill_us;
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

        if total_us > 100_000 {
            info!(
                "step breakdown: decode={}us emit={}us new={}us prefill={}us total={}us batch={}",
                decode_us, emit_us, new_us, prefill_us, total_us, num
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{PrefillBudget, PrefillReservation};

    fn collect_schedulable_indices(
        mut budget: PrefillBudget,
        reservations: &[(usize, PrefillReservation)],
    ) -> Vec<usize> {
        let mut scheduled = Vec::new();
        for &(idx, reservation) in reservations {
            if !budget.can_schedule(reservation) {
                break;
            }
            budget.reserve(reservation);
            scheduled.push(idx);
        }
        scheduled
    }

    #[test]
    fn budget_stops_on_first_token_budget_miss() {
        let budget = PrefillBudget {
            remaining_prefill_tokens: 6,
            remaining_requests: usize::MAX,
            remaining_free_pages: usize::MAX,
            planned_seq_lens: vec![0, 0, 0],
            page_size: 16,
        };
        let reservations = vec![
            (
                0,
                PrefillReservation {
                    slot_idx: 0,
                    prefill_tokens: 8,
                    pool_tokens: 8,
                    decode_headroom_tokens: 0,
                },
            ),
            (
                1,
                PrefillReservation {
                    slot_idx: 1,
                    prefill_tokens: 4,
                    pool_tokens: 4,
                    decode_headroom_tokens: 0,
                },
            ),
            (
                2,
                PrefillReservation {
                    slot_idx: 2,
                    prefill_tokens: 2,
                    pool_tokens: 2,
                    decode_headroom_tokens: 0,
                },
            ),
        ];

        assert_eq!(
            collect_schedulable_indices(budget, &reservations),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn budget_stops_on_first_page_budget_miss() {
        let budget = PrefillBudget {
            remaining_prefill_tokens: 8,
            remaining_requests: usize::MAX,
            remaining_free_pages: 1,
            planned_seq_lens: vec![0, 3],
            page_size: 4,
        };
        let reservations = vec![
            (
                0,
                PrefillReservation {
                    slot_idx: 0,
                    prefill_tokens: 5,
                    pool_tokens: 5,
                    decode_headroom_tokens: 0,
                },
            ),
            (
                1,
                PrefillReservation {
                    slot_idx: 1,
                    prefill_tokens: 1,
                    pool_tokens: 1,
                    decode_headroom_tokens: 0,
                },
            ),
        ];

        assert_eq!(
            collect_schedulable_indices(budget, &reservations),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn budget_reserves_decode_headroom_for_prefill_completion() {
        let budget = PrefillBudget {
            remaining_prefill_tokens: 4,
            remaining_requests: 1,
            remaining_free_pages: 1,
            planned_seq_lens: vec![4],
            page_size: 4,
        };
        let reservation = PrefillReservation {
            slot_idx: 0,
            prefill_tokens: 4,
            pool_tokens: 4,
            decode_headroom_tokens: 1,
        };

        assert!(!budget.can_schedule(reservation));
    }
}
