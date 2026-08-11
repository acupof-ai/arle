//! Session KV-recall planning (device-neutral core of the infinite-memory feature).
//!
//! Given a session's resident token count and per-block relevance scores, decide
//! the **working set** the decode actually attends: `sink ∪ recalled-blocks ∪
//! local-window`. The output is a set of ascending, merged token ranges consumed
//! by the executor page-gather primitive (`infer-metal` `gather_kv_ranges`, #4).
//!
//! This module is pure arithmetic — no device types, no session id — so it is
//! fully unit-testable. The score computation (`query · mean-key`) is device-side;
//! the offload/promote of evicted blocks reuses `prefix.rs` + the kv tier. The
//! budget (sink/local/recall split) is carved out of `resource.rs`'s
//! `kv_capacity_tokens` by the caller (`SessionMemory`, #2).

/// Fixed-region budget for a session's GPU working set, in tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecallConfig {
    /// Attention-sink prefix kept resident (system/task anchors).
    pub n_init: usize,
    /// Local window: the most-recent tokens, always resident.
    pub n_local: usize,
    /// Tokens per recall block (the granularity of offload + recall).
    pub l_bs: usize,
    /// How many middle blocks to recall into the working set ("多召回些" → larger).
    pub top_k: usize,
}

impl RecallConfig {
    #[must_use]
    pub fn working_set_tokens(&self) -> usize {
        self.n_init + self.n_local + self.top_k * self.l_bs
    }
}

/// The planned decode working set for one step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecallPlan {
    /// Ascending, non-overlapping token ranges `[start, end)` to gather. When the
    /// session fits the budget this is the single full range `[(0, cache_len)]`
    /// (identical to today's contiguous read).
    pub ranges: Vec<(usize, usize)>,
    /// Middle-block indices recalled this step (for the executor + promote
    /// bookkeeping). Empty when no recall happened (everything resident).
    pub recalled_blocks: Vec<usize>,
}

impl RecallPlan {
    #[must_use]
    pub fn token_count(&self) -> usize {
        self.ranges.iter().map(|&(s, e)| e - s).sum()
    }

    fn all_resident(cache_len: usize) -> Self {
        Self {
            ranges: if cache_len == 0 {
                Vec::new()
            } else {
                vec![(0, cache_len)]
            },
            recalled_blocks: Vec::new(),
        }
    }
}

/// Plan the working set for a session with `cache_len` resident tokens.
///
/// `block_scores[i]` is the relevance of middle block `i` (e.g. `query · mean-key`);
/// higher = more relevant. The middle is `[n_init, n_init + nb·l_bs)` where `nb` is
/// the number of whole `l_bs` blocks that fit before the local window; any partial
/// tail is absorbed into the local window so the plan covers `[0, cache_len)` with
/// no gap. Returns the contiguous full range when the session fits the budget or
/// has no evictable middle (so the default path is byte-identical to today).
#[must_use]
pub fn plan_recall(cache_len: usize, block_scores: &[f32], cfg: &RecallConfig) -> RecallPlan {
    if cfg.l_bs == 0 || cache_len <= cfg.n_init + cfg.n_local + cfg.l_bs {
        return RecallPlan::all_resident(cache_len);
    }
    let mid_lo = cfg.n_init;
    let mid_span = cache_len - cfg.n_init - cfg.n_local;
    let nb = mid_span / cfg.l_bs; // whole middle blocks
    if nb == 0 || cfg.top_k >= nb {
        return RecallPlan::all_resident(cache_len);
    }
    let local_start = mid_lo + nb * cfg.l_bs; // tail (n_local + partial block) is local

    // Select the top_k middle blocks by score (stable: higher score, then lower
    // index), then sort ascending to keep temporal order.
    debug_assert!(
        block_scores.len() >= nb,
        "plan_recall needs a score per middle block ({nb}), got {}",
        block_scores.len()
    );
    let mut idx: Vec<usize> = (0..nb).collect();
    idx.sort_by(|&a, &b| {
        let sa = block_scores.get(a).copied().unwrap_or(f32::NEG_INFINITY);
        let sb = block_scores.get(b).copied().unwrap_or(f32::NEG_INFINITY);
        sb.partial_cmp(&sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(cfg.top_k);
    idx.sort_unstable();

    // Assemble sink + selected blocks + local, then merge adjacent ranges so the
    // gather does the fewest slices.
    let ranges: Vec<(usize, usize)> = (cfg.n_init > 0)
        .then_some((0, cfg.n_init))
        .into_iter()
        .chain(idx.iter().map(|&b| {
            let s = mid_lo + b * cfg.l_bs;
            (s, s + cfg.l_bs)
        }))
        .chain(std::iter::once((local_start, cache_len)))
        .collect();
    RecallPlan {
        ranges: merge_adjacent(ranges),
        recalled_blocks: idx,
    }
}

/// Merge touching/overlapping ascending ranges (`ranges` already sorted by start
/// because sink < blocks < local by construction).
fn merge_adjacent(ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (s, e) in ranges {
        if s >= e {
            continue;
        }
        match out.last_mut() {
            Some(last) if s <= last.1 => last.1 = last.1.max(e),
            _ => out.push((s, e)),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(n_init: usize, n_local: usize, l_bs: usize, top_k: usize) -> RecallConfig {
        RecallConfig {
            n_init,
            n_local,
            l_bs,
            top_k,
        }
    }

    #[test]
    fn short_session_stays_fully_resident() {
        // cache_len within sink+local+one block → no recall, single contiguous range.
        let p = plan_recall(40, &[], &cfg(8, 16, 8, 4));
        assert_eq!(p.ranges, vec![(0, 40)]);
        assert!(p.recalled_blocks.is_empty());
        assert_eq!(p.token_count(), 40);
    }

    #[test]
    fn empty_cache_is_empty_plan() {
        assert_eq!(plan_recall(0, &[], &cfg(8, 16, 8, 4)).ranges, Vec::new());
    }

    #[test]
    fn recall_selects_top_k_blocks_plus_sink_and_local() {
        // cache_len 128, sink 8, local 16 → middle [8,112) = 104 tokens, l_bs 8 → 13
        // blocks. top_k=2 picks the two highest-scored, kept in temporal order.
        let c = cfg(8, 16, 8, 2);
        // 13 block scores; make blocks 5 and 10 the hottest.
        let mut scores = vec![0.0_f32; 13];
        scores[5] = 9.0;
        scores[10] = 8.0;
        let p = plan_recall(128, &scores, &c);
        // working set = sink(8) + 2 blocks(16) + local. local_start = 8 + 13*8 = 112.
        let block5 = (8 + 5 * 8, 8 + 6 * 8); // (48, 56)
        let block10 = (8 + 10 * 8, 8 + 11 * 8); // (88, 96)
        assert_eq!(
            p.ranges,
            vec![(0, 8), block5, block10, (112, 128)],
            "sink + 2 hottest blocks (temporal order) + local"
        );
        assert_eq!(p.recalled_blocks, vec![5, 10]);
        assert_eq!(p.token_count(), 8 + 8 + 8 + 16);
    }

    #[test]
    fn topk_covering_all_middle_blocks_stays_resident() {
        // top_k >= number of middle blocks → nothing to evict → contiguous.
        let p = plan_recall(96, &[1.0; 9], &cfg(8, 16, 8, 100));
        assert_eq!(p.ranges, vec![(0, 96)]);
    }

    #[test]
    fn adjacent_selected_blocks_merge() {
        // sink touches block 0, and blocks 0 and 1 are adjacent → merge.
        let c = cfg(8, 8, 8, 3);
        let mut scores = vec![0.0_f32; 12];
        scores[0] = 3.0; // block 0 = [8,16), adjacent to sink [0,8)
        scores[1] = 2.0; // block 1 = [16,24), adjacent to block 0
        scores[5] = 1.0;
        let p = plan_recall(112, &scores, &c);
        // sink+block0+block1 merge into [0,24); block5 = [48,56); local_start=8+12*8=104.
        assert_eq!(p.ranges, vec![(0, 24), (48, 56), (104, 112)]);
        assert_eq!(p.recalled_blocks, vec![0, 1, 5]);
    }

    #[test]
    fn working_set_is_budget_bounded_regardless_of_history() {
        // The whole point: token_count stays ~constant as cache_len grows.
        let c = cfg(8, 16, 8, 4);
        // enough scores for the largest session's middle blocks (1M/8 ≈ 125K).
        let scores = vec![0.0_f32; 200_000];
        let small = plan_recall(1_000, &scores, &c).token_count();
        let huge = plan_recall(1_000_000, &scores, &c).token_count();
        assert_eq!(
            small, huge,
            "recall working set is constant vs history length"
        );
        assert_eq!(huge, c.working_set_tokens()); // 8 + 16 + 4*8 = 56 tokens
    }
}
