//! Multimodal input identity and encoder-cache primitives.
//!
//! The runtime still routes text-only requests through the existing prefix
//! cache path. These helpers provide the stable content identities needed by
//! future VLM callers to bind image/audio inputs into the same KV fingerprint
//! chain without leaking protocol-specific JSON into the scheduler.

pub mod encoder_cache;
pub mod hash;

pub use encoder_cache::{EncoderCache, EncoderCacheError};
pub use hash::{
    ImageIdentity, MmHash, VisionPadTokens, hash_canonical_media, hash_media_sequence,
    pad_tokens_from_hash,
};
