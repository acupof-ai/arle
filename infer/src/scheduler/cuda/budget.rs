use super::{ModelForward, Scheduler};

pub(super) const CLIPPED_MAX_NEW_TOKENS_ESTIMATE: usize = 4_096;

pub(super) fn normalized_page_size(page_size: usize) -> usize {
    page_size.max(1)
}

pub(super) fn page_count(tokens: usize, page_size: usize) -> usize {
    tokens.div_ceil(normalized_page_size(page_size))
}

pub(super) fn clipped_max_new_tokens_estimate(max_tokens: usize) -> usize {
    max_tokens.min(CLIPPED_MAX_NEW_TOKENS_ESTIMATE)
}

/// Pages a request needs *to be admitted*, in tokens.
///
/// SGLang-style admission: charge prefill cost only (prompt + 1 page tail
/// for the just-generated token) and let decode pages allocate lazily during
/// the decode loop. The mid-step OOM safety net is
/// [`crate::scheduler::cuda::decode::Scheduler::retract_decode_to_fit`]
/// (called from both `step_decode_launch` and `step_mixed_launch`), which
/// preempts the least-progressed decode and re-queues it (Recompute mode).
///
/// Already-running decodes still reserve their remaining `max_tokens` against
/// the page budget when planning prefill admission — that path goes through
/// [`crate::scheduler::cuda::execution::Scheduler::remaining_decode_reservation_tokens`],
/// which keeps using `clipped_max_new_tokens_estimate`. Only the *new*
/// admission path drops the upfront max_tokens reservation.
pub(super) fn estimated_request_target_tokens(prompt_tokens: usize, _max_tokens: usize) -> usize {
    prompt_tokens.saturating_add(1)
}

pub(super) fn estimated_request_pages(
    prompt_tokens: usize,
    max_tokens: usize,
    page_size: usize,
) -> usize {
    page_count(
        estimated_request_target_tokens(prompt_tokens, max_tokens),
        page_size,
    )
}

pub(super) fn additional_pages_needed(
    seq_len: usize,
    additional_tokens: usize,
    page_size: usize,
) -> usize {
    if additional_tokens == 0 {
        return 0;
    }
    let page_size = normalized_page_size(page_size);
    let current_pages = seq_len.div_ceil(page_size);
    let future_pages = seq_len
        .saturating_add(additional_tokens)
        .div_ceil(page_size);
    future_pages.saturating_sub(current_pages)
}

pub(super) fn partial_tail_capacity<I>(seq_lens: I, page_size: usize) -> usize
where
    I: IntoIterator<Item = usize>,
{
    let page_size = normalized_page_size(page_size);
    seq_lens
        .into_iter()
        .map(|seq_len| {
            let used = seq_len % page_size;
            if used == 0 { 0 } else { page_size - used }
        })
        .sum()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PageGrowth {
    pub slot_idx: usize,
    pub tokens: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PageTarget {
    pub slot_idx: usize,
    pub seq_floor: usize,
    pub target_seq: usize,
}

pub(super) fn estimated_request_target(
    slot_idx: usize,
    prompt_tokens: usize,
    max_tokens: usize,
    reserved_prefix_tokens: usize,
) -> PageTarget {
    PageTarget {
        slot_idx,
        seq_floor: reserved_prefix_tokens,
        target_seq: estimated_request_target_tokens(prompt_tokens, max_tokens),
    }
}

#[derive(Clone, Debug)]
pub(super) struct PageBudget {
    remaining_free_pages: usize,
    planned_seq_lens: Vec<usize>,
    append_cow_pages: Vec<usize>,
    page_size: usize,
    active: bool,
}

impl PageBudget {
    pub(super) fn new(
        remaining_free_pages: usize,
        planned_seq_lens: Vec<usize>,
        page_size: usize,
        active: bool,
    ) -> Self {
        Self {
            remaining_free_pages,
            planned_seq_lens,
            append_cow_pages: Vec::new(),
            page_size: normalized_page_size(page_size),
            active,
        }
    }

    pub(super) fn from_scheduler<M: ModelForward>(scheduler: &Scheduler<M>, active: bool) -> Self {
        let mut budget = Self::new(
            scheduler.effective_pool_free_pages(),
            (0..scheduler.states.len())
                .map(|slot_idx| scheduler.paged_kv_pool.seq_len(slot_idx))
                .collect(),
            scheduler.paged_kv_pool.page_size,
            active,
        );
        budget.append_cow_pages = (0..scheduler.states.len())
            .map(|slot_idx| scheduler.paged_kv_pool.append_cow_pages_needed(slot_idx))
            .collect();
        budget
    }

    #[cfg(test)]
    pub(super) fn remaining_free_pages(&self) -> usize {
        self.remaining_free_pages
    }

    #[cfg(test)]
    pub(super) fn planned_seq_len(&self, slot_idx: usize) -> usize {
        self.planned_seq_lens[slot_idx]
    }

    pub(super) fn can_fit_growth(&self, growth: PageGrowth) -> bool {
        !self.active || self.additional_pages_for_growth(growth) <= self.remaining_free_pages
    }

    pub(super) fn reserve_growth(&mut self, growth: PageGrowth) {
        if !self.active {
            return;
        }
        let new_pages = self.additional_pages_for_growth(growth);
        self.remaining_free_pages = self.remaining_free_pages.saturating_sub(new_pages);
        self.planned_seq_lens[growth.slot_idx] =
            self.planned_seq_lens[growth.slot_idx].saturating_add(growth.tokens);
        if let Some(cow_pages) = self.append_cow_pages.get_mut(growth.slot_idx)
            && growth.tokens > 0
        {
            *cow_pages = 0;
        }
    }

    pub(super) fn can_fit_target(&self, target: PageTarget) -> bool {
        !self.active || self.additional_pages_for_target(target) <= self.remaining_free_pages
    }

    pub(super) fn reserve_target(&mut self, target: PageTarget) {
        if !self.active {
            return;
        }
        let current_seq = self.effective_seq_len(target.slot_idx, target.seq_floor);
        let target_seq = target.target_seq.max(target.seq_floor);
        let new_pages = self.additional_pages_for_target(target);
        self.remaining_free_pages = self.remaining_free_pages.saturating_sub(new_pages);
        self.planned_seq_lens[target.slot_idx] = target_seq;
        if let Some(cow_pages) = self.append_cow_pages.get_mut(target.slot_idx)
            && target_seq > current_seq
        {
            *cow_pages = 0;
        }
    }

    fn additional_pages_for_growth(&self, growth: PageGrowth) -> usize {
        additional_pages_needed(
            self.planned_seq_lens[growth.slot_idx],
            growth.tokens,
            self.page_size,
        ) + if growth.tokens > 0 {
            self.append_cow_pages
                .get(growth.slot_idx)
                .copied()
                .unwrap_or(0)
        } else {
            0
        }
    }

    fn additional_pages_for_target(&self, target: PageTarget) -> usize {
        let current_seq = self.effective_seq_len(target.slot_idx, target.seq_floor);
        let target_seq = target.target_seq.max(target.seq_floor);
        additional_pages_needed(
            current_seq,
            target_seq.saturating_sub(current_seq),
            self.page_size,
        ) + if target_seq > current_seq {
            self.append_cow_pages
                .get(target.slot_idx)
                .copied()
                .unwrap_or(0)
        } else {
            0
        }
    }

    fn effective_seq_len(&self, slot_idx: usize, seq_floor: usize) -> usize {
        self.planned_seq_lens[slot_idx].max(seq_floor)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct StepTokenBudget {
    remaining_tokens: usize,
    remaining_requests: usize,
}

impl StepTokenBudget {
    pub(super) fn new(remaining_tokens: usize, remaining_requests: usize) -> Self {
        Self {
            remaining_tokens,
            remaining_requests,
        }
    }

    pub(super) fn for_prefill(
        max_num_batched_tokens: usize,
        max_prefill_tokens: usize,
        running_decode_tokens: usize,
        remaining_requests: usize,
    ) -> Self {
        Self::new(
            max_num_batched_tokens
                .saturating_sub(running_decode_tokens)
                .min(max_prefill_tokens),
            remaining_requests,
        )
    }

    pub(super) fn can_fit(&self, tokens: usize) -> bool {
        tokens > 0 && tokens <= self.remaining_tokens && self.remaining_requests > 0
    }

    pub(super) fn reserve(&mut self, tokens: usize) {
        debug_assert!(self.can_fit(tokens));
        self.remaining_tokens = self.remaining_tokens.saturating_sub(tokens);
        self.remaining_requests = self.remaining_requests.saturating_sub(1);
    }
}

pub(super) fn waiting_admission_shortage_pages<I>(
    free_pages: usize,
    page_size: usize,
    free_slots: usize,
    waiting_requests: usize,
    waiting: I,
) -> usize
where
    I: IntoIterator<Item = (Option<usize>, usize)>,
{
    if free_slots == 0 && waiting_requests == 0 {
        return 0;
    }
    let mut requested_pages = 0usize;
    let mut considered = 0usize;
    let admission_window = free_slots.max(waiting_requests);
    for (prompt_tokens, max_tokens) in waiting {
        let Some(prompt_tokens) = prompt_tokens else {
            continue;
        };
        requested_pages = requested_pages.saturating_add(estimated_request_pages(
            prompt_tokens,
            max_tokens,
            page_size,
        ));
        considered += 1;
        if considered >= admission_window {
            break;
        }
    }
    requested_pages.saturating_sub(free_pages)
}

pub(super) fn prefix_cache_reclaim_goal_pages(
    retained_pages: usize,
    high_water_pages: usize,
    low_water_pages: usize,
    waiting_shortage_pages: usize,
) -> usize {
    let watermark_goal = if retained_pages > high_water_pages {
        retained_pages.saturating_sub(low_water_pages)
    } else {
        0
    };
    watermark_goal
        .max(waiting_shortage_pages)
        .min(retained_pages)
}

pub(super) fn coordinator_submit_headroom(capacity: usize, active: usize) -> usize {
    capacity.saturating_sub(active.saturating_add(1))
}

#[cfg(test)]
mod tests {
    use super::{
        PageBudget, PageGrowth, PageTarget, StepTokenBudget, additional_pages_needed,
        clipped_max_new_tokens_estimate, coordinator_submit_headroom, estimated_request_pages,
        estimated_request_target, page_count, partial_tail_capacity,
        prefix_cache_reclaim_goal_pages, waiting_admission_shortage_pages,
    };

    #[test]
    fn additional_pages_needed_respects_partial_pages() {
        assert_eq!(additional_pages_needed(0, 0, 4), 0);
        assert_eq!(additional_pages_needed(0, 4, 4), 1);
        assert_eq!(additional_pages_needed(3, 1, 4), 0);
        assert_eq!(additional_pages_needed(4, 1, 4), 1);
        assert_eq!(additional_pages_needed(5, 2, 4), 0);
    }

    #[test]
    fn estimated_request_pages_charges_prompt_only_plus_one() {
        // SGLang-style admission: prompt + 1 token for the first generated
        // sample, no upfront reservation for the full max_tokens decode tail.
        // 4_097 + 1 = 4_098 tokens → ceil(4_098/16) = 257 pages.
        assert_eq!(estimated_request_pages(4_097, 256, 16), 257);
    }

    #[test]
    fn estimated_request_pages_ignores_max_tokens_at_admission() {
        // Decode-tail size no longer affects admission; the retract path
        // handles mid-step OOM. 2_048 + 1 = 2_049 → ceil(2_049/16) = 129.
        assert_eq!(estimated_request_pages(2_048, 8_192, 16), 129);
        // The clipping helper itself is still used by execution.rs for
        // *running* decode reservations, so its semantics are unchanged.
        assert_eq!(clipped_max_new_tokens_estimate(8_192), 4_096);
    }

    #[test]
    fn partial_tail_capacity_counts_unfilled_page_tails() {
        assert_eq!(partial_tail_capacity([0, 4, 7, 9], 4), 4);
    }

    #[test]
    fn page_budget_growth_reserves_decode_headroom() {
        let mut budget = PageBudget::new(1, vec![4, 0], 4, true);
        budget.reserve_growth(PageGrowth {
            slot_idx: 0,
            tokens: 1,
        });

        assert!(!budget.can_fit_growth(PageGrowth {
            slot_idx: 1,
            tokens: 4,
        }));
    }

    #[test]
    fn page_budget_evictable_pages_expand_admission_capacity() {
        let old_budget = PageBudget::new(0, vec![4], 4, true);
        let growth = PageGrowth {
            slot_idx: 0,
            tokens: 4,
        };
        assert!(!old_budget.can_fit_growth(growth));

        let mut budget_with_evictable = PageBudget::new(1, vec![4], 4, true);
        assert!(budget_with_evictable.can_fit_growth(growth));
        budget_with_evictable.reserve_growth(growth);
        assert_eq!(budget_with_evictable.remaining_free_pages(), 0);
    }

    #[test]
    fn page_budget_growth_counts_shared_tail_cow_once() {
        let mut budget = PageBudget::new(1, vec![3], 4, true);
        budget.append_cow_pages = vec![1];

        assert!(budget.can_fit_growth(PageGrowth {
            slot_idx: 0,
            tokens: 1,
        }));
        budget.reserve_growth(PageGrowth {
            slot_idx: 0,
            tokens: 1,
        });
        assert_eq!(budget.remaining_free_pages(), 0);
        assert_eq!(budget.planned_seq_len(0), 4);
        assert!(!budget.can_fit_growth(PageGrowth {
            slot_idx: 0,
            tokens: 1,
        }));
        assert!(budget.can_fit_growth(PageGrowth {
            slot_idx: 0,
            tokens: 0,
        }));
    }

    #[test]
    fn page_budget_target_counts_shared_tail_cow_once() {
        let mut budget = PageBudget::new(1, vec![3], 4, true);
        budget.append_cow_pages = vec![1];
        let target = PageTarget {
            slot_idx: 0,
            seq_floor: 3,
            target_seq: 4,
        };

        assert!(budget.can_fit_target(target));
        budget.reserve_target(target);
        assert_eq!(budget.remaining_free_pages(), 0);
        assert_eq!(budget.planned_seq_len(0), 4);
        assert!(budget.can_fit_target(PageTarget {
            slot_idx: 0,
            seq_floor: 4,
            target_seq: 4,
        }));
    }

    #[test]
    fn page_budget_target_honors_attached_prefix_floor() {
        let mut budget = PageBudget::new(1, vec![0], 4, true);
        let target = PageTarget {
            slot_idx: 0,
            seq_floor: 4,
            target_seq: 8,
        };

        assert!(budget.can_fit_target(target));
        budget.reserve_target(target);
        assert_eq!(budget.remaining_free_pages(), 0);
        assert_eq!(budget.planned_seq_len(0), 8);
    }

    #[test]
    fn inactive_page_budget_is_noop() {
        let mut budget = PageBudget::new(0, vec![0], 4, false);
        budget.reserve_target(PageTarget {
            slot_idx: 0,
            seq_floor: 4,
            target_seq: 8,
        });
        budget.reserve_growth(PageGrowth {
            slot_idx: 0,
            tokens: 4,
        });

        assert_eq!(budget.remaining_free_pages(), 0);
        assert_eq!(budget.planned_seq_len(0), 0);
    }

    #[test]
    fn step_token_budget_clamps_whole_step_before_prefill_cap() {
        let mut budget = StepTokenBudget::for_prefill(3, 10, 0, usize::MAX);

        assert!(!budget.can_fit(4));
        assert!(budget.can_fit(2));
        budget.reserve(2);
        assert!(budget.can_fit(1));
    }

    #[test]
    fn step_token_budget_respects_prefill_request_cap() {
        let mut budget = StepTokenBudget::for_prefill(16, 16, 0, 1);

        assert!(budget.can_fit(2));
        budget.reserve(2);
        assert!(!budget.can_fit(1));
    }

    #[test]
    fn page_count_rounds_up_by_page_size() {
        assert_eq!(page_count(0, 16), 0);
        assert_eq!(page_count(1, 16), 1);
        assert_eq!(page_count(16, 16), 1);
        assert_eq!(page_count(17, 16), 2);
    }

    #[test]
    fn estimated_request_target_sets_prefix_floor_and_goal() {
        // target_seq = prompt_tokens + 1 (prompt-only admission); seq_floor
        // is the reserved prefix length, kept untouched.
        assert_eq!(
            estimated_request_target(3, 128, 64, 96),
            PageTarget {
                slot_idx: 3,
                seq_floor: 96,
                target_seq: 129,
            }
        );
    }

    #[test]
    fn waiting_shortage_expands_beyond_free_slots_for_backlog() {
        // Per-request page need under prompt-only admission:
        //   ceil((4_097 + 1) / 16) = 257 pages.
        // Backlog of 11 requests → 11 * 257 = 2_827 pages.
        // Free pages = 1_385 → shortage = 2_827 - 1_385 = 1_442.
        let waiting = std::iter::repeat_n((Some(4_097usize), 256usize), 11);
        assert_eq!(
            waiting_admission_shortage_pages(1_385, 16, 5, 11, waiting),
            1_442
        );
    }

    #[test]
    fn reclaim_goal_can_trigger_below_watermark_when_waiting_needs_headroom() {
        assert_eq!(prefix_cache_reclaim_goal_pages(1_384, 2_076, 1_384, 0), 0);
        assert_eq!(
            prefix_cache_reclaim_goal_pages(1_384, 2_076, 1_384, 1_345),
            1_345
        );
        assert_eq!(
            prefix_cache_reclaim_goal_pages(2_408, 2_076, 1_384, 1_345),
            1_345
        );
        assert_eq!(
            prefix_cache_reclaim_goal_pages(2_408, 2_076, 1_384, 2_369),
            2_369
        );
    }

    #[test]
    fn coordinator_submit_headroom_reserves_one_slot() {
        assert_eq!(coordinator_submit_headroom(16, 14), 1);
        assert_eq!(coordinator_submit_headroom(16, 15), 0);
        assert_eq!(coordinator_submit_headroom(16, 16), 0);
        assert_eq!(coordinator_submit_headroom(0, 0), 0);
    }
}
