//! Block identity: [`BlockId`] and [`BlockHashCtx`].
//!
//! See `crate::kv_tier` for the module-level design notes.

/// Content-addressable KV block identifier.
///
/// Stable across processes and across nodes once [`BlockId::derive`] is
/// implemented in P5. Phase 1–4 use caller-assigned values (simple integer
/// counters or hashes of convenience); the type remains opaque so callers
/// can swap implementations without touching downstream code.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct BlockId(pub u64);

/// Context for deterministic [`BlockId`] derivation. Locked in at P5; kept
/// public here so P1/P2/P3 call sites can already thread the values through
/// even while [`BlockId::derive`] is a stub.
#[derive(Debug, Clone, Copy)]
pub struct BlockHashCtx {
    /// Fingerprint of the model (architecture + weight digest + numeric
    /// profile). Two different models must produce different hashes even
    /// for the same tokens.
    pub model_fingerprint: u64,
    /// Layer index in the model stack.
    pub layer_idx: u16,
    /// KV dtype selector — so bf16 and int8 variants of the same layer
    /// hash to different ids.
    pub kv_format_tag: u8,
    /// Hash of the parent block along the radix path. Chains the tree
    /// walk into the content so divergence at block granularity is
    /// detectable without walking up the tree.
    pub parent_hash: u64,
}

impl BlockId {
    /// Deterministic derivation used by [`super::directory::TierDirectory`] insertion.
    ///
    /// **Phase gate**: lands in P5 (content-addressable Tiered KV Cache).
    /// Until then, callers pass caller-assigned ids directly.
    pub fn derive(_ctx: &BlockHashCtx, _tokens: &[u32]) -> Self {
        todo!("P5: blake3 of (model_fingerprint, layer_idx, kv_format_tag, parent_hash, tokens)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_id_is_copy_and_ordered() {
        // Pins the Copy + Ord derives that the directory relies on.
        let a = BlockId(1);
        let b = a;
        assert_eq!(a, b);
        assert!(BlockId(1) < BlockId(2));
    }
}
