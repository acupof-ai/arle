//! Core domain types shared across infer workspace crates.

use std::sync::Arc;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Opaque identifier for a KV block that is currently resident in some tier.
///
/// Scope: lives only as long as the block is in memory or on disk; **not**
/// stable across restarts, **not** stable across nodes. Used by the radix
/// tree, the paged KV pool, the block manager, and the tier-aware metadata
/// on `RadixCache::Node`.
///
/// u32 is sufficient for the block counts we expect in any single process:
/// worst-case `page_size=16` + 80 GB T0 HBM + 1 TB T1 host pinned on
/// DeepSeek-V3 (64 layers × 8 KV heads × 128 head_dim × bf16 ≈ 256 KB/block)
/// is roughly 4.5 M blocks — well below 2³². vLLM and SGLang also both use
/// 32-bit block ids. Keeping `BlockId` at `u32` makes radix-tree nodes
/// cache-line friendly.
///
/// **Do not confuse with [`BlockFingerprint`]**, which is the *content*
/// hash used for persistence and cross-node reuse. `BlockId` is a pool
/// slot id, not a content identifier.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct BlockId(pub u32);

/// Content-addressable fingerprint for a KV block's semantic identity.
///
/// The stable hash binds token content to the model identity, KV pool format,
/// and radix-path parent chain so persisted blocks do not collide across
/// mismatched engines or reload contexts. `compute_from_tokens` remains as a
/// compatibility shim for local tests that only care about deterministic token
/// sensitivity.
///
/// Radix-tree nodes carry `Option<BlockFingerprint>`; `None` means a
/// transient in-memory block that never went through the publish path.
///
/// Non-token inputs to `BlockFingerprint::compute`. These bind a
/// fingerprint to the specific (model, numeric format, parent
/// block) it was produced under, so two blocks with the same token
/// content but different parent chains or different KV formats
/// hash to different fingerprints. Required for M4 session
/// save/load: a reloaded session under the wrong model or wrong
/// format must not collide with a saved one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvContentContext<'a> {
    /// Stable per-engine identifier for the model weights.
    pub model_fingerprint: &'a [u8],
    /// Pool numeric format wire-level u8 discriminant (stable numeric id).
    pub kv_format_tag: u8,
    /// Parent block's fingerprint, or None for the first block in a radix path.
    pub parent: Option<BlockFingerprint>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct BlockFingerprint(pub [u8; 16]);

impl BlockFingerprint {
    /// Compute a stable 16-byte content fingerprint over the full block
    /// identity chain.
    pub fn compute(ctx: KvContentContext<'_>, tokens: &[u32]) -> Self {
        Self::compute_with_multimodal_hashes(ctx, tokens, &[])
    }

    /// Compute a block fingerprint that also binds non-text media identities.
    ///
    /// `mm_hashes` must be in the same order as the synthetic multimodal
    /// pad-token sequences inserted into `tokens`. Keeping these hashes outside
    /// the token stream preserves persisted KV identity even when tokenizer or
    /// packing conventions change.
    pub fn compute_with_multimodal_hashes(
        ctx: KvContentContext<'_>,
        tokens: &[u32],
        mm_hashes: &[[u8; 16]],
    ) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(b"infer-kv-v2\x00");
        h.update(b"model\x00");
        h.update(&(ctx.model_fingerprint.len() as u64).to_le_bytes());
        h.update(ctx.model_fingerprint);
        h.update(b"fmt\x00");
        h.update(&[ctx.kv_format_tag]);
        h.update(b"parent\x00");
        match ctx.parent {
            Some(fp) => {
                h.update(&[1u8]);
                h.update(&fp.0);
            }
            None => {
                h.update(&[0u8]);
            }
        }
        h.update(b"tokens\x00");
        h.update(&(tokens.len() as u64).to_le_bytes());
        for &t in tokens {
            h.update(&t.to_le_bytes());
        }
        if !mm_hashes.is_empty() {
            h.update(b"mm_hashes\x00");
            h.update(&(mm_hashes.len() as u64).to_le_bytes());
            for mm_hash in mm_hashes {
                h.update(mm_hash);
            }
        }
        let full = h.finalize();
        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(&full.as_bytes()[..16]);
        BlockFingerprint(bytes)
    }

    #[doc(hidden)]
    pub fn compute_from_tokens(tokens: &[u32]) -> Self {
        Self::compute(
            KvContentContext {
                model_fingerprint: b"",
                kv_format_tag: 0,
                parent: None,
            },
            tokens,
        )
    }
}

/// Stable request identifier across scheduler/runtime boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RequestId(pub u64);

/// Client-supplied session/conversation identifier used by the scheduler to
/// route subsequent turns of the same agent session to the slot or radix
/// subtree that already holds their KV prefix.
///
/// The value is opaque to the engine. Callers should treat it as a stable
/// per-conversation key (typically a UUID or hash of the conversation array).
/// It is the protocol-level plumbing for
/// `docs/projects/agent-first-architecture.md::A2` (session-sticky routing);
/// admission logic consumes it once `A1` wires the RadixCache into the
/// CUDA/Metal schedulers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionId(Arc<str>);

impl SessionId {
    /// Wrap an arbitrary string as a session id. Empty strings are rejected by
    /// callers; this type does not enforce non-emptiness because higher layers
    /// decide what to do with invalid input.
    pub fn new(id: impl Into<Arc<str>>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for SessionId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for SessionId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Serialize for SessionId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for SessionId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(Self::from(value))
    }
}

/// Canonical inference mode used by control/data-plane orchestration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceMode {
    Prefill,
    Decode,
}

/// Lifecycle action emitted by scheduler-like components.
///
/// These events are intentionally action-oriented instead of state-oriented so
/// consumers can distinguish initial admission from preemption/requeue, and
/// lifecycle transitions from chunked work units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestEventKind {
    Enqueued,
    Requeued,
    PrefillStarted,
    DecodeStep,
    Evicted,
    Completed,
    Cancelled,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_is_deterministic_and_sensitive() {
        let a = BlockFingerprint::compute_from_tokens(&[1, 2, 3, 4]);
        let b = BlockFingerprint::compute_from_tokens(&[1, 2, 3, 4]);
        let c = BlockFingerprint::compute_from_tokens(&[1, 2, 3, 5]);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn fingerprint_compute_is_stable_across_equivalent_context() {
        let ctx = KvContentContext {
            model_fingerprint: b"qwen3-4b",
            kv_format_tag: 2,
            parent: Some(BlockFingerprint([0x11; 16])),
        };
        let a = BlockFingerprint::compute(ctx, &[1, 2, 3, 4]);
        let b = BlockFingerprint::compute(ctx, &[1, 2, 3, 4]);

        assert_eq!(a, b);
    }

    #[test]
    fn fingerprint_compute_differs_on_different_model() {
        let tokens = [1, 2, 3, 4];
        let a = BlockFingerprint::compute(
            KvContentContext {
                model_fingerprint: b"qwen3-4b",
                kv_format_tag: 1,
                parent: None,
            },
            &tokens,
        );
        let b = BlockFingerprint::compute(
            KvContentContext {
                model_fingerprint: b"qwen35-4b",
                kv_format_tag: 1,
                parent: None,
            },
            &tokens,
        );

        assert_ne!(a, b);
    }

    #[test]
    fn fingerprint_compute_differs_on_different_parent() {
        let tokens = [1, 2, 3, 4];
        let a = BlockFingerprint::compute(
            KvContentContext {
                model_fingerprint: b"qwen3-4b",
                kv_format_tag: 1,
                parent: Some(BlockFingerprint([0x11; 16])),
            },
            &tokens,
        );
        let b = BlockFingerprint::compute(
            KvContentContext {
                model_fingerprint: b"qwen3-4b",
                kv_format_tag: 1,
                parent: Some(BlockFingerprint([0x22; 16])),
            },
            &tokens,
        );

        assert_ne!(a, b);
    }

    #[test]
    fn fingerprint_compute_empty_tokens_is_non_trivial() {
        let fp = BlockFingerprint::compute(
            KvContentContext {
                model_fingerprint: b"",
                kv_format_tag: 0,
                parent: None,
            },
            &[],
        );

        assert_ne!(fp, BlockFingerprint([0; 16]));
    }

    #[test]
    fn multimodal_fingerprint_empty_hashes_match_text_path() {
        let ctx = KvContentContext {
            model_fingerprint: b"qwen3-vl",
            kv_format_tag: 1,
            parent: Some(BlockFingerprint([0x11; 16])),
        };
        let tokens = [1, 2, 3, 4];

        assert_eq!(
            BlockFingerprint::compute(ctx, &tokens),
            BlockFingerprint::compute_with_multimodal_hashes(ctx, &tokens, &[])
        );
    }

    #[test]
    fn multimodal_fingerprint_binds_media_hashes() {
        let ctx = KvContentContext {
            model_fingerprint: b"qwen3-vl",
            kv_format_tag: 1,
            parent: None,
        };
        let tokens = [1, 0x8000_0001, 2];
        let image_a = [[0xAA; 16]];
        let image_b = [[0xBB; 16]];

        let text_only = BlockFingerprint::compute(ctx, &tokens);
        let with_a = BlockFingerprint::compute_with_multimodal_hashes(ctx, &tokens, &image_a);
        let with_b = BlockFingerprint::compute_with_multimodal_hashes(ctx, &tokens, &image_b);

        assert_ne!(text_only, with_a);
        assert_ne!(with_a, with_b);
    }

    #[test]
    fn session_id_round_trips_from_str_and_string() {
        let from_str = SessionId::from("abc-123");
        let from_string = SessionId::from(String::from("abc-123"));
        assert_eq!(from_str, from_string);
        assert_eq!(from_str.as_str(), "abc-123");
        assert_eq!(from_str.to_string(), "abc-123");
    }
}
