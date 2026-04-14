//! Block identity — re-export of the canonical [`crate::types::BlockId`].
//!
//! After the 2026-04-15 tiered-kv-cache revision (see
//! `docs/projects/tiered-kv-cache.md` §5.1), this module no longer defines
//! its own `BlockId` type. There is exactly one canonical `BlockId(u32)`
//! in `crate::types`, and all three former locations (`kv_tier::id`,
//! `prefix_cache`, `block_manager`) re-export it.
//!
//! The old `BlockId(u64)` + `BlockHashCtx` content-hash sketch that lived
//! in this file before the revision has been deleted. Content hashing
//! moves to [`crate::types::BlockFingerprint`] and is only constructed
//! at the M4 persistence path or the M5 cross-node reuse path.

pub use crate::types::BlockId;
