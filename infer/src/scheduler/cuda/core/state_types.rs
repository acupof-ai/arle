//! Pending decode/prefill bookkeeping types + tier prefetch + store-dedup keys.
//!
//! Split out of `core.rs` (pure structural refactor — no behavior change).
//! These structs travel together as the scheduler's per-step "in-flight"
//! state carried across loop turns.

use crate::types::BlockFingerprint;
use fastrace::Span;

pub(in crate::scheduler::cuda) struct PendingDecode {
    pub decode_indices: Vec<usize>,
    pub slot_indices: Vec<usize>,
    /// True only when `sample_batch_greedy_launch` actually fired the argmax kernel.
    pub greedy_launched: bool,
    pub decode_spans: Vec<(usize, Span)>,
    pub mixed_prefill: Option<PendingMixedPrefill>,
}

pub(in crate::scheduler::cuda) struct PendingPrefillRow {
    pub slot_idx: usize,
    pub total_tokens: usize,
    pub next_progress: usize,
}

pub(in crate::scheduler::cuda) struct PendingPrefill {
    pub rows: Vec<PendingPrefillRow>,
    pub uses_paged: bool,
    pub prefill_spans: Vec<(usize, Span)>,
}

pub(in crate::scheduler::cuda) struct PendingMixedPrefill {
    pub rows: Vec<PendingPrefillRow>,
    pub uses_paged: bool,
    pub prefill_spans: Vec<(usize, Span)>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(in crate::scheduler::cuda) struct PrefetchTicketState {
    pub host_blocks: usize,
    pub disk_blocks: usize,
    pub remote_blocks: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(in crate::scheduler::cuda) struct StoreDedupKey {
    pub fingerprint: BlockFingerprint,
    pub target: crate::kv_tier::StoreTarget,
}
