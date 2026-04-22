use super::{ModelForward, Scheduler};

pub(super) fn normalized_page_size(page_size: usize) -> usize {
    page_size.max(1)
}

pub(super) fn page_count(tokens: usize, page_size: usize) -> usize {
    tokens.div_ceil(normalized_page_size(page_size))
}

pub(super) fn full_request_pages(
    prompt_tokens: usize,
    max_tokens: usize,
    page_size: usize,
) -> usize {
    page_count(prompt_tokens.saturating_add(max_tokens), page_size)
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

pub(super) fn full_request_target(
    slot_idx: usize,
    prompt_tokens: usize,
    max_tokens: usize,
    reserved_prefix_tokens: usize,
) -> PageTarget {
    PageTarget {
        slot_idx,
        seq_floor: reserved_prefix_tokens,
        target_seq: prompt_tokens.saturating_add(max_tokens),
    }
}

#[derive(Clone, Debug)]
pub(super) struct PageBudget {
    remaining_free_pages: usize,
    planned_seq_lens: Vec<usize>,
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
            page_size: normalized_page_size(page_size),
            active,
        }
    }

    pub(super) fn from_scheduler<M: ModelForward>(scheduler: &Scheduler<M>, active: bool) -> Self {
        Self::new(
            scheduler.pool_free_pages(),
            (0..scheduler.states.len())
                .map(|slot_idx| scheduler.paged_kv_pool.seq_len(slot_idx))
                .collect(),
            scheduler.paged_kv_pool.page_size,
            active,
        )
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
        debug_assert!(new_pages <= self.remaining_free_pages);
        self.remaining_free_pages = self.remaining_free_pages.saturating_sub(new_pages);
        self.planned_seq_lens[growth.slot_idx] =
            self.planned_seq_lens[growth.slot_idx].saturating_add(growth.tokens);
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
        let new_pages = additional_pages_needed(
            current_seq,
            target_seq.saturating_sub(current_seq),
            self.page_size,
        );
        debug_assert!(new_pages <= self.remaining_free_pages);
        self.remaining_free_pages = self.remaining_free_pages.saturating_sub(new_pages);
        self.planned_seq_lens[target.slot_idx] = target_seq;
    }

    fn additional_pages_for_growth(&self, growth: PageGrowth) -> usize {
        additional_pages_needed(
            self.planned_seq_lens[growth.slot_idx],
            growth.tokens,
            self.page_size,
        )
    }

    fn additional_pages_for_target(&self, target: PageTarget) -> usize {
        let current_seq = self.effective_seq_len(target.slot_idx, target.seq_floor);
        let target_seq = target.target_seq.max(target.seq_floor);
        additional_pages_needed(
            current_seq,
            target_seq.saturating_sub(current_seq),
            self.page_size,
        )
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
        requested_pages = requested_pages.saturating_add(full_request_pages(
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
        coordinator_submit_headroom, full_request_pages, full_request_target, page_count,
        partial_tail_capacity, prefix_cache_reclaim_goal_pages, waiting_admission_shortage_pages,
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
    fn full_request_pages_rounds_prompt_plus_decode_up() {
        assert_eq!(full_request_pages(4_097, 256, 16), 273);
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
    fn page_count_rounds_up_by_page_size() {
        assert_eq!(page_count(0, 16), 0);
        assert_eq!(page_count(1, 16), 1);
        assert_eq!(page_count(16, 16), 1);
        assert_eq!(page_count(17, 16), 2);
    }

    #[test]
    fn full_request_target_sets_prefix_floor_and_goal() {
        assert_eq!(
            full_request_target(3, 128, 64, 96),
            PageTarget {
                slot_idx: 3,
                seq_floor: 96,
                target_seq: 192,
            }
        );
    }

    #[test]
    fn waiting_shortage_expands_beyond_free_slots_for_backlog() {
        let waiting = std::iter::repeat_n((Some(4_097usize), 256usize), 11);
        assert_eq!(
            waiting_admission_shortage_pages(1_385, 16, 5, 11, waiting),
            1_618
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
