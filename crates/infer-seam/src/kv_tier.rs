//! Write-through tiered KV memory contract (device-neutral, cross-backend).
//!
//! This is the **write-through cache** model, NOT swap. HBM is a bounded
//! write-through cache of a session's KV; the host tier (DRAM L2 -> NVMe L3) is
//! the source of truth holding everything. The four timing rules are encoded as
//! three verbs so neither backend has to invent the semantics:
//!
//! | rule | verb | when |
//! |---|---|---|
//! | **write** | [`KvTier::write_through`] | a KV page fills (prefill/decode) -> async mirror device->tier |
//! | **HBM full** | [`KvTier::evict_drop`] | working set over budget -> drop the coldest unpinned page (no write-back) |
//! | **prefill** | [`KvTier::prefetch`] | a turn's prefill -> alloc + load tier->device for the recalled history |
//! | **decode** | (no verb) | append + attend resident; never a synchronous tier read |
//!
//! Why this is not the swap (`demote_block`/`promote_block`-as-swap) API: that
//! model freed-then-reloaded the SAME page under decode-time memory pressure,
//! which contends with the single-allocator `KvPool`. Write-through dissolves
//! that: eviction is a free with NO write-back (the page was already mirrored),
//! prefetch happens only at the one batched prefill sync point, and decode never
//! touches the tier. See `docs/plans/2026-06-23-writethrough-tiered-kv-memory.md`.
//!
//! ## Relationship to the existing `BackendExecutor` tier methods
//!
//! The CUDA backend already physically implements the device<->tier copies as
//! [`crate::BackendExecutor::demote_prefix_pages`] (D2H into the host store) and
//! [`crate::BackendExecutor::promote_prefix_pages`] (H2D back). Those ARE the
//! transport for this contract — there is exactly ONE session-keyed store (R5),
//! not a parallel one. `KvTier` re-expresses them with the write-through verb
//! names plus the relevance-prefetch entry point, mapping:
//!
//! - `write_through` == demote-after-an-async-mirror (the page stays resident; the
//!   tier just gains a durable copy, so a later `evict_drop` is free).
//! - `evict_drop` == free the device page (the host `KvAllocator` owns the free);
//!   no tier write because `write_through` already copied it.
//! - `prefetch` == promote (alloc a device page + H2D from the tier) at prefill.
//!
//! The trait is intentionally minimal and host-only: it names device pages by
//! `u32` page id, blocks by [`TierBlockKey`], and never exposes a device tensor.
//! Both the CUDA paged pool (page == write-through grain, natural) and the Metal
//! windowed session KV satisfy it.

use crate::KvTierLocation;

/// Session-scoped key for one recall block's KV in the tier store.
///
/// The tier is the source of truth for ALL sessions, so a block is addressed by
/// `(session, block_index)` — multi-tenant isolation (the "session A never
/// prefetches into session B" gate) is structural: a `prefetch` for session A
/// can only name session-A keys. The backend flattens this to its own opaque
/// `u64` store key; the host never sees the flattening.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TierBlockKey {
    /// Stable per-session id (tenant isolation namespace). The host assigns it
    /// from the request's `session_id`; two requests with the same `session_id`
    /// share a tier namespace, distinct ones never collide.
    pub session: u64,
    /// Index of the recall block within the session's history (token base =
    /// `n_init + block * l_bs`, the coarse recall-block grain of R1, decoupled
    /// from the device page size).
    pub block: u64,
}

impl TierBlockKey {
    /// Construct a key for `block` within `session`.
    #[must_use]
    pub const fn new(session: u64, block: u64) -> Self {
        Self { session, block }
    }
}

/// Device-neutral write-through tier contract.
///
/// All three verbs are host-orchestrated: they take page ids and block keys, copy
/// device<->tier internally, and never cross the seam with a device tensor.
/// Implementations MUST keep `write_through` off the decode critical path (a side
/// stream / pinned-host ring on CUDA; a cheap buffer handoff on Metal's unified
/// memory) so decode never stalls on tier I/O (R4).
///
/// The default methods make the trait a zero-cost opt-in: a backend with no tier
/// reports `tier_capacity_pages() == 0` and the host never calls the verbs, so the
/// baseline decode path stays byte-for-byte unchanged (CUDA is the Stable
/// backend). A backend opts in by overriding the verbs (the CUDA dense-Qwen3 paged
/// pool does, delegating to its `CudaKvTierStore`).
pub trait KvTier {
    /// Pages the tier store can hold (host DRAM + NVMe). `0` (default) disables
    /// every write-through verb — the host never mirrors, evicts, or prefetches.
    fn tier_capacity_pages(&self) -> usize {
        0
    }

    /// Bytes one tier page occupies (for budget accounting). `0` when no tier.
    fn tier_page_bytes(&self) -> usize {
        0
    }

    /// Where a block currently lives, or `None` if the tier has no copy.
    fn tier_location(&self, _key: TierBlockKey) -> Option<KvTierLocation> {
        None
    }

    /// **write** — mirror a just-filled device page into the tier, off the decode
    /// critical path. After this the tier holds a durable copy of the page, so a
    /// later [`Self::evict_drop`] of that page is free (no write-back). The mirror
    /// itself need not be complete on return (it is async by design), but a
    /// subsequent `prefetch` of the same key MUST observe the write — the
    /// implementation drains the write queue at the next sync point if needed.
    ///
    /// Returns `true` if the tier accepted the page (had room or made room);
    /// `false` means the page was not mirrored and MUST NOT be evict-dropped (it
    /// would be lost). Default: reject (no tier).
    fn write_through(&mut self, _key: TierBlockKey, _page: u32) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// **HBM full** — the page is already mirrored ([`Self::write_through`]
    /// returned `true`), so dropping it from HBM only needs the device page freed
    /// by the host allocator; the tier keeps the source of truth. This hook lets
    /// the backend drop any device-side mirror/sidecar it holds for `page`; the
    /// host `KvAllocator` performs the actual page free. NO tier write happens
    /// here — that is the whole point of write-through. Default: no-op.
    fn evict_drop(&mut self, _page: u32) {}

    /// **prefill** — load a block's KV from the tier into the freshly allocated
    /// device `page` (`(key, page)` pairs). The copy MUST be complete before
    /// return: the engine writes these pages into the prefill page table right
    /// after and the next forward reads them. This is the ONE place a tier read
    /// happens, at the single batched prefill sync point — never during decode.
    /// Default: error (no tier).
    fn prefetch(&mut self, _entries: &[(TierBlockKey, u32)]) -> anyhow::Result<()> {
        anyhow::bail!("backend has no write-through KV tier store")
    }

    /// Drop tier entries for a finished/abandoned session so its DRAM/NVMe is
    /// reclaimed (multi-tenant hygiene). Default: no-op.
    fn drop_tier_session(&mut self, _session: u64) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The no-tier default is a strict no-op so the baseline stays byte-identical.
    struct NoTier;
    impl KvTier for NoTier {}

    #[test]
    fn default_tier_is_disabled() {
        let mut t = NoTier;
        assert_eq!(t.tier_capacity_pages(), 0);
        assert_eq!(t.tier_page_bytes(), 0);
        assert_eq!(t.tier_location(TierBlockKey::new(1, 2)), None);
        assert!(!t.write_through(TierBlockKey::new(1, 2), 7).unwrap());
        // evict_drop / drop_tier_session are infallible no-ops.
        t.evict_drop(7);
        t.drop_tier_session(1);
        // prefetch must refuse loudly when there is no store.
        assert!(t.prefetch(&[(TierBlockKey::new(1, 2), 7)]).is_err());
    }

    #[test]
    fn block_key_namespaces_by_session() {
        // Same block index in two sessions never collides (tenant isolation).
        assert_ne!(TierBlockKey::new(1, 5), TierBlockKey::new(2, 5));
        assert_eq!(TierBlockKey::new(7, 3), TierBlockKey::new(7, 3));
    }
}
