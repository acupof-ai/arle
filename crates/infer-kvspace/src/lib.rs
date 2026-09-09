//! Backend-neutral KV host logic: capacity accounting, content indexing
//! (prefix match, radix, sidecar lifecycle), and layout description. The
//! kernel-operand role — the page table layout the attention kernel reads —
//! stays in `infer-cuda`, because it tracks the kernel and needs a GPU to
//! verify.
//!
//! Crate boundaries follow pipeline stages; modules inside may follow model
//! families. The DSv4 byte codec lives in the `dsv4` module.

mod dsv4;
mod tier;

pub use dsv4::{
    Dsv4LayerPageState, Dsv4PrefixPageEntry, MAX_PENDING_PREFIX_CAPTURES, PendingPrefixPage,
    PrefixPoolIndex, capture_epoch_matches, rekey_target_conflicts, reusable_prefix_blocks,
    reusable_prefix_blocks_for_prompt,
};
pub use tier::{
    KvSlotTier, NS_SIDECAR, NS_SIDECAR_CHUNK, NS_SLOT, NS_SLOT_CHUNK, SidecarSnapshot,
    hash_prefix_tokens, sidecar_candidates, sidecar_lstar, sidecar_mat_len,
    sidecar_periodic_savable, tier_io_stats,
};
