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
    fn prefetch_query_means_last_m_tokens() {
        // 4 tokens, dim 2. Last 2 tokens are [3,3] and [5,5] -> mean [4,4].
        let queries = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 5.0, 5.0];
        let q = prefetch_query(&queries, 4, 2);
        assert_eq!(q, vec![4.0, 4.0]);
    }

    #[test]
    fn prefetch_query_clamps_m_to_available_and_handles_empty() {
        // m larger than tokens -> mean of all tokens.
        let queries = vec![2.0, 4.0]; // 1 token, dim 2
        assert_eq!(prefetch_query(&queries, 1, 16), vec![2.0, 4.0]);
        // empty inputs -> empty query (keep everything resident).
        assert!(prefetch_query(&[], 0, 16).is_empty());
        assert!(prefetch_query(&queries, 0, 16).is_empty());
    }

    #[test]
    fn prefetch_selects_top_k_in_temporal_order() {
        // 4 blocks, dim 1. query=[1]; reps pick blocks 3 and 1 as hottest.
        let reps = vec![vec![0.1], vec![0.9], vec![0.2], vec![0.8]];
        let sel = prefetch_blocks(&reps, &[1.0], 2);
        assert_eq!(sel, vec![1, 3], "two hottest, ascending temporal order");
    }

    #[test]
    fn prefetch_empty_query_or_reps_selects_nothing() {
        let reps = vec![vec![1.0]];
        assert!(prefetch_blocks(&reps, &[], 4).is_empty());
        assert!(prefetch_blocks(&[], &[1.0], 4).is_empty());
        assert!(prefetch_blocks(&reps, &[1.0], 0).is_empty());
    }

    #[test]
    fn prefetch_topk_covering_all_selects_all() {
        let reps = vec![vec![1.0], vec![2.0], vec![3.0]];
        assert_eq!(prefetch_blocks(&reps, &[1.0], 100), vec![0, 1, 2]);
    }

    #[test]
    fn evict_drops_coldest_unpinned_respecting_pins() {
        // page_size 16, cache_len 160 -> 10 pages (0..9).
        // sink n_init=16 -> page 0 pinned. local n_local=32 -> local_start=128,
        // first_local_page=8 -> pages 8,9 pinned. Candidates = pages 1..=7.
        let pages: Vec<u32> = (0..10).collect();
        // Make page 3 the coldest, then 5, then 1.
        let mut access = vec![100u64; 10];
        access[3] = 1;
        access[5] = 2;
        access[1] = 3;
        let c = cfg(16, 32, 32, 4);
        // budget 7 -> drop 3 pages: the 3 coldest unpinned = pages 3,5,1 -> ids.
        let drop = evict_drop_pages(&pages, &access, 16, 160, 7, &c);
        assert_eq!(drop, vec![1, 3, 5], "coldest unpinned, returned ascending");
    }

    #[test]
    fn evict_within_budget_drops_nothing() {
        let pages: Vec<u32> = (0..5).collect();
        let access = vec![0u64; 5];
        let c = cfg(16, 32, 32, 4);
        assert!(evict_drop_pages(&pages, &access, 16, 80, 5, &c).is_empty());
        assert!(evict_drop_pages(&pages, &access, 16, 80, 10, &c).is_empty());
    }

    #[test]
    fn evict_never_touches_pinned_even_over_budget() {
        // All middle pages pinned away: tiny cache where sink+local cover everything.
        // page_size 16, cache_len 64 -> 4 pages. n_init=32 -> sink pages 0,1.
        // n_local=32 -> local_start=32, first_local_page=2 -> pages 2,3 pinned.
        // No unpinned candidates -> nothing can be dropped even asking for budget 1.
        let pages: Vec<u32> = (0..4).collect();
        let access = vec![0u64; 4];
        let c = cfg(32, 32, 32, 2);
        assert!(
            evict_drop_pages(&pages, &access, 16, 64, 1, &c).is_empty(),
            "all over-budget pages pinned -> keep them (correct over budget, never corrupt)"
        );
    }

    #[test]
    fn rep_pool_cap_drops_coldest_blocks() {
        // 5 blocks; cap 3 -> drop the 2 coldest by access.
        let access = vec![(0, 50u64), (1, 10), (2, 99), (3, 20), (4, 80)];
        let dropped = cap_rep_pool(&access, 3);
        assert_eq!(dropped, vec![1, 3], "blocks 1 (10) and 3 (20) are coldest");
    }

    #[test]
    fn rep_pool_cap_zero_or_under_cap_drops_nothing() {
        let access = vec![(0, 1u64), (1, 2)];
        assert!(cap_rep_pool(&access, 0).is_empty(), "cap 0 = unbounded");
        assert!(cap_rep_pool(&access, 5).is_empty(), "under cap");
    }

    #[test]
    fn evict_drop_keeps_resident_page_set_bounded_as_session_grows() {
        // The flat-VRAM invariant expressed at the policy level: simulate a slot
        // whose physical page set is the working set, growing the logical length
        // one page at a time and RELEASING (not just un-attending) the coldest
        // unpinned page whenever the resident set exceeds the page budget. The
        // number of physically-resident pages must stay bounded, even as the
        // session grows to hundreds of pages.
        let page_size = 16usize;
        let c = cfg(16, 64, 16, 4); // sink 1 page, local 4 pages, k=4 -> budget ~9 pages
        let budget_pages = c.working_set_tokens().div_ceil(page_size); // 9
        // The slot's physically-resident pages, in temporal order. Each entry is a
        // physical page id; we model a release by removing the entry entirely
        // (the page went back to the pool) — NOT by leaving it resident.
        let mut resident: Vec<u32> = Vec::new();
        let mut access: Vec<u64> = Vec::new();
        let mut high_water = 0usize;
        for step in 0..400usize {
            // A new page rolls every `page_size` tokens; here one per step. The
            // physical page id is the step number (each step mints a fresh page).
            resident.push(step as u32);
            access.push(step as u64);
            let cache_len = (step + 1) * page_size;
            high_water = high_water.max(resident.len());
            let drop = evict_drop_pages(&resident, &access, page_size, cache_len, budget_pages, &c);
            // Actually RELEASE the dropped pages from the resident set (the win):
            // remove them so the resident vector shrinks, proving they are freed,
            // not merely excluded from attention.
            if !drop.is_empty() {
                let drop_set: std::collections::HashSet<u32> = drop.iter().copied().collect();
                (resident, access) = resident
                    .iter()
                    .copied()
                    .zip(access.iter().copied())
                    .filter(|(p, _)| !drop_set.contains(p))
                    .unzip();
            }
            assert!(
                resident.len() <= budget_pages,
                "step {step}: resident {} exceeded budget {budget_pages} (page not freed)",
                resident.len()
            );
        }
        // Over 400 pages of history the resident set never grew past the budget.
        assert_eq!(
            resident.len(),
            budget_pages,
            "resident set converged to exactly the page budget"
        );
        assert!(
            high_water <= budget_pages + 1,
            "resident high-water {high_water} stayed at the budget (+1 transient) despite a 400-page session"
        );
    }

    #[test]
    fn plan_working_set_is_budget_bounded() {
        // Bridge stays consistent with plan_recall: the working set stays constant
        // vs history (the whole point). The exact token_count can exceed
        // working_set_tokens() by at most one local partial block depending on
        // where the local window falls relative to the l_bs grid, but it does NOT
        // grow with cache_len — that is the invariant that matters.
        let c = cfg(16, 256, 32, 8);
        let scores = vec![0.0_f32; 200_000];
        let a = crate::plan_recall(64_000, &scores, &c).token_count();
        let b = crate::plan_recall(1_000_000, &scores, &c).token_count();
        assert_eq!(a, b, "working set is constant regardless of history length");
        assert!(
            b <= c.working_set_tokens() + c.l_bs,
            "working set {b} stays bounded near {} (+ at most one local block)",
            c.working_set_tokens()
        );
    }
}
