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
}

impl RecallPlan {
    fn all_resident(cache_len: usize) -> Self {
        Self {
            ranges: if cache_len == 0 {
                Vec::new()
            } else {
                vec![(0, cache_len)]
            },
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
    // A dead scorer is indistinguishable from a working one downstream: all-equal
    // scores fall through the tie-break to blocks 0..top_k and still produce a
    // plausible plan. Say so rather than silently keeping the prompt's head.
    if block_scores
        .get(..nb)
        .is_some_and(|s| s.iter().all(|v| *v == s[0]))
    {
        log::warn!("plan_recall: zero-variance scores over {nb} blocks — the scorer is dead");
    }
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
