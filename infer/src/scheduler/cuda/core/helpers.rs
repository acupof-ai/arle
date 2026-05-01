//! Top-level helper functions and constants for the CUDA scheduler core.
//!
//! Split out of `core.rs` (pure structural refactor — no behavior change).
//! Pure functions only: watermark/spill math, sealed-prefix predicates,
//! and the radix block-size constant.

use std::collections::HashSet;

use crate::prefix_cache::{BlockId, BlockSelectionIntent};

/// Block size (in tokens) for the global `RadixCache` prefix observer.
/// Chosen to match the M0.3 target paged-pool page size so that when
/// M2 dual residency wires the radix directly onto the pool, the block
/// boundaries already agree.
pub(in crate::scheduler::cuda) const PREFIX_CACHE_BLOCK_SIZE: usize = 16;

/// Contiguous KV working buffer per slot (tokens). Only prefill uses it;
/// decode writes directly to the paged pool via `decode_prep_paged`.
/// Prefill chunk size is capped to this value to prevent buffer overflow.
pub(in crate::scheduler::cuda) const CONTIGUOUS_KV_TOKENS: usize = 512;

// Prefix-cache and T1 watermark / keepalive tunables live on
// `crate::scheduler::types::SchedulerConfig`. Per the project env-var
// policy (`docs/environment.md` §0), these are **not** env-driven —
// callers assign directly to `SchedulerConfig` fields.

pub(in crate::scheduler::cuda) fn prefix_cache_retain_hard_cap_pages(
    total_pages: usize,
    cap_fraction: f64,
) -> usize {
    (total_pages as f64 * cap_fraction) as usize
}

fn watermark_bytes(capacity_bytes: usize, fraction: f64) -> usize {
    (capacity_bytes as f64 * fraction) as usize
}

pub(in crate::scheduler::cuda) fn host_spill_target_bytes(
    reserved_bytes: usize,
    capacity_bytes: usize,
    high_water: f64,
    low_water: f64,
    intent: BlockSelectionIntent,
) -> usize {
    let low_water_bytes = watermark_bytes(capacity_bytes, low_water);
    match intent {
        BlockSelectionIntent::Evict => 0,
        BlockSelectionIntent::Spill => {
            let high_water_bytes = watermark_bytes(capacity_bytes, high_water);
            if reserved_bytes < high_water_bytes {
                0
            } else {
                reserved_bytes.saturating_sub(low_water_bytes)
            }
        }
        BlockSelectionIntent::Drain => reserved_bytes.saturating_sub(low_water_bytes),
    }
}

pub(in crate::scheduler::cuda) fn can_publish_prefix_pages(
    retained_pages: usize,
    total_pages: usize,
    new_pages: usize,
    cap_fraction: f64,
) -> bool {
    retained_pages.saturating_add(new_pages)
        <= prefix_cache_retain_hard_cap_pages(total_pages, cap_fraction)
}

pub(in crate::scheduler::cuda) fn sealed_block_token_count(
    block_size: usize,
    block_count: usize,
) -> usize {
    block_size.saturating_mul(block_count)
}

pub(in crate::scheduler::cuda) fn is_full_sealed_prefix(
    matched_len: usize,
    block_size: usize,
    block_count: usize,
) -> bool {
    block_count > 0 && matched_len == sealed_block_token_count(block_size, block_count)
}

pub(in crate::scheduler::cuda) fn select_sparse_pages_from_slot_pages(
    slot_pages: &[u32],
    page_size: usize,
    seq_len: usize,
    recent_tokens: usize,
    top_k: usize,
) -> Vec<BlockId> {
    if slot_pages.is_empty() || seq_len == 0 || (recent_tokens == 0 && top_k == 0) {
        return Vec::new();
    }

    let live_pages = seq_len.div_ceil(page_size.max(1)).min(slot_pages.len());
    let slot_pages = &slot_pages[..live_pages];
    let recent_pages = recent_tokens
        .div_ceil(page_size.max(1))
        .min(slot_pages.len());
    let recent_start = slot_pages.len().saturating_sub(recent_pages);

    let mut selected = Vec::new();
    let mut seen = HashSet::new();

    for &page in slot_pages.iter().take(top_k) {
        let block = BlockId(page);
        if seen.insert(block) {
            selected.push(block);
        }
    }

    for &page in &slot_pages[recent_start..] {
        let block = BlockId(page);
        if seen.insert(block) {
            selected.push(block);
        }
    }

    selected
}
