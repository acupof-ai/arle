//! Tier-movement policies — write-through / write-back and prefetch modes.
//!
//! These enums govern when blocks travel between tiers. They live in
//! `kv_tier/` (not in `scheduler/`) because the scheduler and coordinator
//! share tier-movement policy vocabulary even though T0↔T1 and T1↔T2/T3
//! byte movement are owned by different boundaries.
//!
//! Names align with the SGLang HiCache convention (see
//! `docs/research/2026-05-04-sglang-hicache-guide.md` Part VI §6.4
//! and §6.5) so cross-system reasoning is one vocabulary.

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PrefetchPolicy {
    /// Issue a prefetch when there is queue headroom; otherwise skip
    /// and let the prefill batch start without that segment (the
    /// missing tail is recomputed on GPU).
    BestEffort,
    /// Always issue prefetch; always wait for it to complete before
    /// admitting the prefill batch. Maximizes hit rate at the cost
    /// of head-of-line blocking on a slow tier.
    WaitComplete,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum WritePolicy {
    /// Every newly produced KV block is persisted to the next tier
    /// immediately. Highest hit-rate, highest write traffic.
    WriteThrough,
    /// Persist only blocks whose hit count has crossed a threshold —
    /// i.e. blocks that earned the right to live in slower tiers by
    /// being reused. Recommended production default.
    WriteThroughSelective,
}
