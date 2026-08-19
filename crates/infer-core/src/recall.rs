//! Session KV-recall planning (device-neutral core of the infinite-memory feature).
//!
//! Given a session's resident token count and per-block relevance scores, decide
//! the **working set** the decode actually attends: `sink ∪ recalled-blocks ∪
//! local-window`. The output is a set of ascending, merged token ranges consumed
//! by the executor page-gather primitive (`infer-metal` `gather_kv_ranges`, #4).
//!
//! This module is pure arithmetic — no device types, no session id — so it is
//! fully unit-testable. It owns the block representation ([`fold_key`],
//! [`score_block`]) as well as the plan, so both backends rank blocks the same
//! way; only fetching the keys is device-side. The offload/promote of evicted
//! blocks reuses `prefix.rs` + the kv tier. The budget (sink/local/recall split)
//! is carved out of `resource.rs`'s `kv_capacity_tokens` by the caller
//! (`SessionMemory`, #2).

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
    /// The budget validated by the offline Qwen3.6 quality gate
    /// (`wins/2026-06-23-kv-recall-arle-core-e2e.md`): working set
    /// `32 + 256 + 8·32 = 544` tokens, 9.6% of the KV in that run. One copy —
    /// every backend reads this, so retuning cannot silently diverge them.
    ///
    /// Recall only restricts attention once `cache_len` exceeds the budget;
    /// below it [`plan_recall`] returns the full contiguous range.
    pub const VALIDATED: Self = Self {
        n_init: 32,
        n_local: 256,
        l_bs: 32,
        top_k: 8,
    };

    #[must_use]
    pub fn working_set_tokens(&self) -> usize {
        self.n_init + self.n_local + self.top_k * self.l_bs
    }

    /// Every region boundary must land on a page edge, or a block spans a
    /// partial page and the per-page readback folds foreign tokens into it.
    #[must_use]
    pub fn is_page_aligned(&self, page_size: usize) -> bool {
        page_size > 0
            && self.n_init.is_multiple_of(page_size)
            && self.n_local.is_multiple_of(page_size)
            && self.l_bs.is_multiple_of(page_size)
    }
}

/// Widen a block's per-channel key envelope `[lo, hi]` to contain `key`.
///
/// An interval, not a mean: K is cached post-RoPE, so averaging over a block's
/// consecutive positions rotates each key by a different angle and the
/// high-frequency channels cancel to zero — measured 2026-08-18, a mean-key
/// scorer retrieved 0/16 at every context length, flat in how much of the middle
/// it kept. The envelope survives rotation and bounds `max q·k`.
///
/// `key` is one head's `head_dim` channels; `lo`/`hi` are that head's slice of
/// the envelope. Seed them with `f32::INFINITY` / `f32::NEG_INFINITY` — those are
/// the identities, so an untouched block stays neutral under a cross-shard widen.
pub fn fold_key(lo: &mut [f32], hi: &mut [f32], key: impl IntoIterator<Item = f32>) {
    for ((l, h), v) in lo.iter_mut().zip(hi.iter_mut()).zip(key) {
        *l = l.min(v);
        *h = h.max(v);
    }
}

/// Upper bound on a block's true `max q·k`, from its key envelope.
///
/// Per channel take the better end of the interval. Being a bound is what makes
/// a top-k selection admissible — a mean gives none.
#[must_use]
pub fn score_block(q: &[f32], lo: &[f32], hi: &[f32]) -> f32 {
    q.iter()
        .zip(lo)
        .zip(hi)
        .map(|((&q, &l), &h)| (q * l).max(q * h))
        .sum()
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

/// How many middle blocks [`plan_recall`] will score for `cache_len`; `0` when
/// the session fits the budget and the plan is the full contiguous range.
///
/// A function of the config and the length only, so every rank agrees on it. A
/// cross-shard reduction must size its payload from this rather than from
/// however many block reps a rank happens to hold — an early stop on one rank
/// would otherwise mismatch the collective and hang the group.
#[must_use]
pub fn recall_block_count(cache_len: usize, cfg: &RecallConfig) -> usize {
    if cfg.l_bs == 0 || cache_len <= cfg.n_init + cfg.n_local + cfg.l_bs {
        return 0;
    }
    let nb = (cache_len - cfg.n_init - cfg.n_local) / cfg.l_bs;
    if cfg.top_k >= nb { 0 } else { nb }
}

/// Plan the working set for a session with `cache_len` resident tokens.
///
/// `block_scores[i]` is the relevance of middle block `i`; higher = more
/// relevant. The middle is `[n_init, n_init + nb·l_bs)` where `nb` is
/// [`recall_block_count`]; any partial tail is absorbed into the local window so
/// the plan covers `[0, cache_len)` with no gap. Returns the contiguous full
/// range when the session fits the budget or has no evictable middle (so the
/// default path is byte-identical to today).
#[must_use]
pub fn plan_recall(cache_len: usize, block_scores: &[f32], cfg: &RecallConfig) -> RecallPlan {
    let nb = recall_block_count(cache_len, cfg);
    if nb == 0 {
        return RecallPlan::all_resident(cache_len);
    }
    let mid_lo = cfg.n_init;
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

#[cfg(test)]
mod envelope_tests {
    use super::*;

    /// The property the selector rests on: the envelope score is an upper bound
    /// on the block's true `max q·k`, and a mean is not. Without the bound, a
    /// top-k selection can drop the block that actually matters.
    #[test]
    fn envelope_bounds_max_dot_where_mean_does_not() {
        let hd = 4;
        // Two keys that cancel channel-wise, as post-RoPE keys in one block do.
        let keys = [vec![1.0, -1.0, 0.5, -0.5], vec![-1.0, 1.0, -0.5, 0.5]];
        let mut lo = vec![f32::INFINITY; hd];
        let mut hi = vec![f32::NEG_INFINITY; hd];
        let mut mean = vec![0.0_f32; hd];
        for k in &keys {
            fold_key(&mut lo, &mut hi, k.iter().copied());
            for (m, v) in mean.iter_mut().zip(k) {
                *m += v / keys.len() as f32;
            }
        }
        let q = vec![1.0, 0.0, 2.0, 0.0];
        let truth = keys
            .iter()
            .map(|k| q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>())
            .fold(f32::NEG_INFINITY, f32::max);
        let mean_score: f32 = q.iter().zip(&mean).map(|(a, b)| a * b).sum();
        assert!(truth > 0.0, "the block IS relevant: {truth}");
        assert!(score_block(&q, &lo, &hi) >= truth, "envelope must bound it");
        assert_eq!(mean_score, 0.0, "the mean cancels this block to zero");
    }

    #[test]
    fn identity_seed_is_neutral_under_widen() {
        let (lo_a, hi_a) = (vec![f32::INFINITY; 2], vec![f32::NEG_INFINITY; 2]);
        let mut lo_b = vec![f32::INFINITY; 2];
        let mut hi_b = vec![f32::NEG_INFINITY; 2];
        fold_key(&mut lo_b, &mut hi_b, [3.0_f32, -1.0]);
        let widened_lo: Vec<f32> = lo_a.iter().zip(&lo_b).map(|(a, b)| a.min(*b)).collect();
        let widened_hi: Vec<f32> = hi_a.iter().zip(&hi_b).map(|(a, b)| a.max(*b)).collect();
        assert_eq!(widened_lo, lo_b);
        assert_eq!(widened_hi, hi_b);
    }
}
