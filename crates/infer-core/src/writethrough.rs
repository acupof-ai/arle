//! Write-through tiered KV memory policy (device-neutral core).
//!
//! The two host-side decisions of the write-through model, as pure functions
//! so they are fully unit-testable with no device types:
//!
//! 1. [`prefetch_blocks`] — at a turn's prefill, score every non-resident
//!    historical block by `prefetch query (R3) · block rep` and pick the top-k to
//!    pull tier->HBM (the recall). Reuses [`crate::plan_recall`]'s selection so the
//!    working set stays budget-bounded regardless of history length.
//! 2. [`evict_drop_pages`] — when the resident working set exceeds the HBM page
//!    budget, choose which pages to evict-drop: the coldest unpinned pages, never
//!    the pinned sink (first `n_init`) or local (last `n_local`) window. No
//!    write-back — the page was already write-through'd.
//!
//! The reps are computed at write-through time (R6) and kept resident in a capped
//! pool (R1); the rep pool cap lives here as [`cap_rep_pool`].

use crate::RecallConfig;

/// Build the **prefetch query** (R3): the mean of the last `m` prompt tokens'
/// query vectors — the "what am I about to generate" signal — instead of the
/// whole prompt (which dilutes the relevance signal). `queries` is the per-token
/// query laid out `[num_tokens, dim]` row-major; the returned vector is `[dim]`.
///
/// `m` is clamped to the number of available tokens. An empty prompt or `dim == 0`
/// yields an empty query (the caller then keeps everything resident — no recall).
#[must_use]
pub fn prefetch_query(queries: &[f32], num_tokens: usize, m: usize) -> Vec<f32> {
    if num_tokens == 0 || queries.is_empty() {
        return Vec::new();
    }
    let dim = queries.len() / num_tokens;
    if dim == 0 {
        return Vec::new();
    }
    let m = m.clamp(1, num_tokens);
    let first = num_tokens - m;
    let mut q = vec![0.0_f32; dim];
    for t in first..num_tokens {
        let base = t * dim;
        for d in 0..dim {
            q[d] += queries[base + d];
        }
    }
    let inv = 1.0_f32 / m as f32;
    for v in &mut q {
        *v *= inv;
    }
    q
}

/// Score every historical block against the prefetch query and select the
/// top-`top_k` to pull into HBM at prefill.
///
/// `reps[i]` is block `i`'s resident mean-key representative (`[dim]`, computed at
/// write-through time, R6). `query` is [`prefetch_query`]'s output (`[dim]`).
/// Returns the selected block indices in ascending temporal order (so the
/// prefetch reads pages in cache order). An empty `query` or no reps selects
/// nothing (the working set is sink + local only).
///
/// The selection mirrors [`plan_recall`]: higher score wins, ties break to the
/// lower (older) index, then the chosen set is sorted ascending. `top_k >= reps`
/// selects all of them.
#[must_use]
pub fn prefetch_blocks(reps: &[Vec<f32>], query: &[f32], top_k: usize) -> Vec<usize> {
    if query.is_empty() || reps.is_empty() || top_k == 0 {
        return Vec::new();
    }
    let scores: Vec<f32> = reps.iter().map(|rep| dot(rep, query)).collect();
    let mut idx: Vec<usize> = (0..reps.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(top_k);
    idx.sort_unstable();
    idx
}

/// Choose which resident device pages to **evict-drop** to fit the HBM page
/// budget, honoring the sink/local pins.
///
/// - `resident_pages`: the slot's device pages in temporal (page-index) order.
/// - `last_access`: per-page recency stamp (higher = more recently used); same
///   length as `resident_pages`. Pinned pages' stamps are ignored.
/// - `page_size`: tokens per page.
/// - `cache_len`: resident token count (so the local window's page span is known).
/// - `budget_pages`: max resident pages allowed.
/// - `cfg`: the recall budget; `n_init` tokens pin the leading sink pages and
///   `n_local` tokens pin the trailing local pages — those NEVER evict.
///
/// Returns the page ids to drop (coldest unpinned first), at most
/// `resident_pages.len() - budget_pages`. Returns empty when already within
/// budget or when every over-budget page is pinned (the caller then keeps them —
/// a correctness-over-budget choice, never silent corruption).
#[must_use]
pub fn evict_drop_pages(
    resident_pages: &[u32],
    last_access: &[u64],
    page_size: usize,
    cache_len: usize,
    budget_pages: usize,
    cfg: &RecallConfig,
) -> Vec<u32> {
    let n = resident_pages.len();
    if n <= budget_pages || page_size == 0 {
        return Vec::new();
    }
    let to_drop = n - budget_pages;

    let sink_pages = cfg.n_init.div_ceil(page_size);
    let local_start_token = cache_len.saturating_sub(cfg.n_local);
    let first_local_page = local_start_token / page_size;

    let mut candidates: Vec<usize> = (0..n)
        .filter(|&i| i >= sink_pages && i < first_local_page)
        .collect();
    candidates.sort_by(|&a, &b| {
        let sa = last_access.get(a).copied().unwrap_or(0);
        let sb = last_access.get(b).copied().unwrap_or(0);
        sa.cmp(&sb).then(a.cmp(&b))
    });
    candidates.truncate(to_drop);
    candidates.sort_unstable();
    candidates.iter().map(|&i| resident_pages[i]).collect()
}

/// Cap the resident rep pool (R1): one `[dim]` f32 rep per recall block does not
/// scale free, so when the pool exceeds `cap` blocks, drop the reps for the
/// COLDEST blocks (lowest `last_access`). Those blocks become prefix-only
/// recallable (graceful horizon, not a cliff) — they keep their tier copy but can
/// no longer be relevance-scored.
///
/// Returns the block indices whose reps were dropped (ascending). Mutates neither
/// argument; the caller removes the returned indices from its rep map. `cap == 0`
/// disables the cap (unbounded, the small-session default).
#[must_use]
pub fn cap_rep_pool(block_last_access: &[(usize, u64)], cap: usize) -> Vec<usize> {
    if cap == 0 || block_last_access.len() <= cap {
        return Vec::new();
    }
    let evict = block_last_access.len() - cap;
    let mut by_recency: Vec<(usize, u64)> = block_last_access.to_vec();
    by_recency.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
    by_recency.truncate(evict);
    let mut dropped: Vec<usize> = by_recency.into_iter().map(|(b, _)| b).collect();
    dropped.sort_unstable();
    dropped
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc = 0.0_f32;
    for k in 0..n {
        acc += a[k] * b[k];
    }
    acc
}
