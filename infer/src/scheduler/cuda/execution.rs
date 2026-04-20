use super::{ModelForward, Phase, Scheduler, info};

#[derive(Clone, Copy, Debug)]
struct PrefillReservation {
    slot_idx: usize,
    prefill_tokens: usize,
    pool_tokens: usize,
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
        let page_size = scheduler.paged_kv_pool.page_size.max(1);
        let partial_capacity = (0..scheduler.states.len())
            .map(|slot_idx| {
                let used = scheduler.paged_kv_pool.seq_len(slot_idx) % page_size;
                if used == 0 { 0 } else { page_size - used }
            })
            .sum::<usize>();
        let free_page_tokens = scheduler
            .paged_kv_pool
            .free_count()
            .saturating_sub(partial_capacity);

        Self {
            remaining_prefill_tokens: scheduler.config.max_prefill_tokens,
            remaining_requests: scheduler.config.prefill_max_requests.unwrap_or(usize::MAX),
            remaining_free_pages: free_page_tokens / page_size,
            planned_seq_lens: (0..scheduler.states.len())
                .map(|slot_idx| scheduler.paged_kv_pool.seq_len(slot_idx))
                .collect(),
            page_size,
        }
    }

    fn can_schedule(&self, reservation: PrefillReservation) -> bool {
        if reservation.prefill_tokens == 0
            || reservation.prefill_tokens > self.remaining_prefill_tokens
            || self.remaining_requests == 0
        {
            return false;
        }
        self.additional_pages_needed(reservation.slot_idx, reservation.pool_tokens)
            <= self.remaining_free_pages
    }

    fn reserve(&mut self, reservation: PrefillReservation) {
        debug_assert!(self.can_schedule(reservation));
        self.remaining_prefill_tokens = self
            .remaining_prefill_tokens
            .saturating_sub(reservation.prefill_tokens);
        self.remaining_requests = self.remaining_requests.saturating_sub(1);
        self.reserve_pool_tokens(reservation.slot_idx, reservation.pool_tokens);
    }

    fn reserve_mixed_prefill(&mut self, prefill_tokens: usize) {
        if prefill_tokens == 0 {
            return;
        }
        self.remaining_prefill_tokens = self.remaining_prefill_tokens.saturating_sub(prefill_tokens);
        self.remaining_requests = self.remaining_requests.saturating_sub(1);
    }

    fn reserve_pool_tokens(&mut self, slot_idx: usize, pool_tokens: usize) {
        let new_pages = self.additional_pages_needed(slot_idx, pool_tokens);
        debug_assert!(new_pages <= self.remaining_free_pages);
        self.remaining_free_pages = self.remaining_free_pages.saturating_sub(new_pages);
        self.planned_seq_lens[slot_idx] =
            self.planned_seq_lens[slot_idx].saturating_add(pool_tokens);
    }

    fn additional_pages_needed(&self, slot_idx: usize, pool_tokens: usize) -> usize {
        if pool_tokens == 0 {
            return 0;
        }
        let used_in_last_page = self.planned_seq_lens[slot_idx] % self.page_size;
        let available_in_last_page = if used_in_last_page == 0 {
            0
        } else {
            self.page_size - used_in_last_page
        };
        pool_tokens
            .saturating_sub(available_in_last_page)
            .div_ceil(self.page_size)
    }
}

impl<M: ModelForward> Scheduler<M> {
    fn prefill_reservation(&self, idx: usize) -> Option<PrefillReservation> {
        if self.active[idx].delta_tx.is_closed() {
            return None;
        }
        let Phase::Prefilling {
            effective_tokens,
            progress,
        } = &self.active[idx].phase
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
            slot_idx: self.active[idx].slot_idx,
            prefill_tokens,
            pool_tokens,
        })
    }

    pub(super) fn step(&mut self) {
        let num = self.active.len();
        if num == 0 {
            return;
        }

        // Phase 1: Batched decode (GPU forward pass + sampling)
        //
        // Launched BEFORE emit_delta so that GPU compute overlaps with
        // CPU tokenizer work in Phase 2. emit_delta only needs tokens
        // from the PREVIOUS step (already in generated_tokens), so
        // there's no data dependency with the current forward pass.
        let has_decode = self
            .active
            .iter()
            .any(|r| matches!(r.phase, Phase::Decoding));
        assert!(
            self.pending_decode.is_none(),
            "pending decode must be cleared before scheduler step"
        );
        self.pending_mixed_prefill_idx = None;
        let decode_launch_us = if has_decode {
            let t = std::time::Instant::now();
            let mixed_prefill_idx = if self.config.enable_mixed_chunk {
                self.active
                    .iter()
                    .enumerate()
                    .find(|(_, r)| {
                        matches!(r.phase, Phase::Prefilling { .. }) && !r.delta_tx.is_closed()
                    })
                    .map(|(i, _)| i)
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
        {
            let Self {
                active, tokenizer, ..
            } = self;
            for req in active.iter_mut() {
                if matches!(req.phase, Phase::Decoding)
                    && req.decoded_token_count < req.generated_tokens.len()
                {
                    req.emit_delta(tokenizer);
                }
            }
        }
        let emit_us = emit_t.elapsed().as_micros();

        // Phase 2b: Accept new requests (CPU-only, no GPU)
        let new_t = std::time::Instant::now();
        loop {
            let new_idx =
                (0..self.active.len()).find(|&i| matches!(self.active[i].phase, Phase::New));
            match new_idx {
                Some(idx) => self.step_new(idx),
                None => break,
            }
        }
        let new_us = new_t.elapsed().as_micros();

        // Phase 2c: Prefill chunks — queued on the same GPU stream after decode.
        // Decode always launches first; prefill work is then bounded by explicit
        // SGLang-style token/request budgets instead of a decode-active hard cap.
        let prefill_t = std::time::Instant::now();
        let already_mixed = self.pending_mixed_prefill_idx.take();
        let mixed_prefill_tokens = self
            .pending_decode
            .as_ref()
            .map(|pending| pending.mixed_prefill_tokens)
            .unwrap_or(0);
        let mut budget = PrefillBudget::from_scheduler(self);
        budget.reserve_mixed_prefill(mixed_prefill_tokens);
        let prefill_indices: Vec<usize> = (0..self.active.len())
            .filter(|&i| matches!(self.active[i].phase, Phase::Prefilling { .. }))
            .filter(|&i| Some(i) != already_mixed)
            .collect();
        for idx in prefill_indices {
            let Some(reservation) = self.prefill_reservation(idx) else {
                continue;
            };
            if !budget.can_schedule(reservation) {
                break;
            }
            let scheduled = self.step_prefill_chunk(idx, reservation.prefill_tokens);
            if scheduled == 0 {
                if matches!(self.active[idx].phase, Phase::Prefilling { .. }) {
                    break;
                }
                continue;
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
