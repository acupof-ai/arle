//! Radix-tree prefix cache for KV cache reuse across requests.
//!
//! # Overview
//!
//! Multiple requests often share a common prefix: the same system prompt,
//! few-shot examples, or a conversation history. Re-computing the KV cache for
//! that prefix on every request wastes GPU time. This module provides a
//! content-addressable cache that maps token sequences → block IDs, so the
//! scheduler can skip prefill for already-cached prefixes.
//!
//! # Data model
//!
//! ```text
//! Root
//! ├── [tok0, tok1, tok2] → BlockId(0)
//! │   ├── [tok3, tok4]   → BlockId(1)   ← "sequence A"
//! │   └── [tok3, tok5]   → BlockId(2)   ← "sequence B"
//! └── [tok6]             → BlockId(3)   ← "sequence C"
//! ```
//!
//! Each node stores:
//! - A **token edge**: the slice of tokens from the parent to this node.
//! - An optional **block ID**: the GPU KV block that caches those tokens.
//! - A **ref count**: how many in-flight requests are using this node's blocks.
//! - A **last access time**: for LRU eviction.
//!
//! # Block granularity
//!
//! Tokens are grouped into fixed-size blocks (e.g. 16 tokens). A node may carry
//! a block only when the cumulative path length to that node ends on a
//! `block_size` boundary. In a compressed radix that node's own edge may be
//! shorter than `block_size` after a split. Fractional trailing blocks are not
//! cached — the final partial block must be re-prefilled.
//!
//! # Eviction
//!
//! When `evict(n)` is called, the cache removes the `n` least-recently-used
//! leaf nodes that have `ref_count == 0`, freeing their block IDs back to the
//! caller.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::kv_tier::{
    BlockLocation, HitKind, IndexEntryState, LookupBlock, LookupHeuristics, LookupOutcome,
    StoreState, Tier,
};
use crate::scheduler::policy::{EvictionCandidate, EvictionPolicy, SchedulerSignals};
use crate::types::{BlockFingerprint, SessionId};

#[allow(dead_code)]
#[derive(Clone, Copy)]
struct SparseDraftCandidate {
    block_id: BlockId,
    path_len: usize,
    last_access: u64,
}

// ============================================================================
// Types
// ============================================================================

/// Opaque GPU KV cache block identifier. Assigned by the block allocator
/// (see `block_manager.rs`) and stored in the cache node.
///
/// Re-exported from [`crate::types::BlockId`] — the canonical single
/// `u32` type after the 2026-04-15 tiered-kv-cache revision. See
/// `docs/projects/tiered-kv-cache.md` §5.1 for the rationale.
pub use crate::types::BlockId;

/// Coalesced metadata update for one cached block.
///
/// `session_id` / `soft_pin_until` / `host_spill_pin_until` use
/// `Option<Option<T>>` so callers can
/// distinguish "leave untouched" from "explicitly clear".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BlockMetadataUpdate {
    pub location: Option<BlockLocation>,
    pub byte_len: Option<u32>,
    pub session_id: Option<Option<SessionId>>,
    pub host_swap_eligible: Option<bool>,
    pub soft_pin_until: Option<Option<u64>>,
    pub host_spill_pin_until: Option<Option<u64>>,
    pub entry_state: Option<IndexEntryState>,
}

/// Read-only snapshot of one cached block's current metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockMetadata {
    pub location: Option<BlockLocation>,
    pub byte_len: u32,
    pub session_id: Option<SessionId>,
    pub host_swap_eligible: bool,
    pub fingerprint: Option<BlockFingerprint>,
    pub soft_pin_until: Option<u64>,
    pub host_spill_pin_until: Option<u64>,
    pub entry_state: IndexEntryState,
    pub store_state: StoreState,
    pub hit_count: u32,
    pub ref_count: u32,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BlockSelectionIntent {
    Evict,
    Spill,
    Drain,
}

/// Summary of a fingerprint reconciliation pass after deserializing a snapshot.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReconcileReport {
    pub remapped: usize,
    pub tombstoned: usize,
    pub orphans_cleared: usize,
}

/// A node in the radix tree.
#[derive(Serialize, Deserialize)]
struct Node {
    /// Token sequence stored on this edge (from parent to this node).
    tokens: Vec<u32>,
    /// GPU block ID cached for the sealed block ending at this node's path.
    /// `None` if the node is off-boundary or has not been committed yet.
    block_id: Option<BlockId>,
    /// Number of in-flight requests currently pinning this node.
    /// Runtime-only: in-flight pins do not survive a snapshot boundary.
    #[serde(default, skip)]
    ref_count: u32,
    /// Monotonically increasing access clock (set on insert/lookup).
    /// Runtime-only: restored caches restart their LRU epoch from zero.
    #[serde(default, skip)]
    last_access: u64,
    /// How many times this node's block participated in a successful lookup.
    #[serde(default)]
    hit_count: u32,
    /// Cross-tier location of this node's bytes once the coordinator is wired.
    #[serde(default)]
    tier_location: Option<BlockLocation>,
    /// Serialized session affinity hint for future policy / persistence work.
    #[serde(default)]
    session_id: Option<SessionId>,
    /// Runtime retention hint: true when the owning request was long enough
    /// that T1 host-pinned demotion should preserve this block instead of
    /// dropping it under GPU pressure.
    #[serde(default, skip)]
    host_swap_eligible: bool,
    /// Optional content fingerprint used by the M4/M5 persistence paths.
    #[serde(default)]
    fingerprint: Option<BlockFingerprint>,
    /// Byte length of the block payload in its current tier.
    #[serde(default)]
    byte_len: u32,
    /// Logical soft pin deadline (scheduler tick / epoch), if any.
    /// Runtime-only: the deadline is relative to the process-local clock.
    #[serde(default, skip)]
    soft_pin_until: Option<u64>,
    /// One-shot anti-thrash pin for freshly demoted host blocks.
    ///
    /// Unlike `soft_pin_until`, lookup hits never refresh this deadline.
    #[serde(default, skip)]
    host_spill_pin_until: Option<u64>,
    /// Control-plane readiness state. Prevents readers from treating partially
    /// staged or evicting entries as canonical ready bytes.
    #[serde(default)]
    entry_state: IndexEntryState,
    /// Write-side persistence state for background store work.
    #[serde(default)]
    store_state: StoreState,
    /// Runtime-only marker for P2.B sparse-KV drafting. A sparse draft view can
    /// exclude a block from the draft attention window while the authoritative
    /// full-KV verifier still owns the physical pages.
    #[serde(default, skip)]
    sparse_dropped: bool,
    /// Children, indexed by the first token of their edge.
    children: HashMap<u32, usize>,
}

impl Node {
    fn new(tokens: Vec<u32>, block_id: Option<BlockId>, now: u64) -> Self {
        Self {
            tokens,
            block_id,
            ref_count: 0,
            last_access: now,
            hit_count: 0,
            tier_location: None,
            session_id: None,
            host_swap_eligible: false,
            fingerprint: None,
            byte_len: 0,
            soft_pin_until: None,
            host_spill_pin_until: None,
            entry_state: IndexEntryState::Ready,
            store_state: StoreState::Idle,
            sparse_dropped: false,
            children: HashMap::new(),
        }
    }

    fn is_tombstone(&self) -> bool {
        self.block_id.is_none()
            && (self.tier_location.is_some()
                || self.session_id.is_some()
                || self.fingerprint.is_some()
                || self.byte_len != 0
                || self.soft_pin_until.is_some()
                || self.host_spill_pin_until.is_some())
    }
}

// ============================================================================
// RadixCache
// ============================================================================

/// Radix tree for content-addressable KV prefix caching.
///
/// The `Serialize` / `Deserialize` derives exist so the full cache state
/// can be snapshotted to disk for session persistence (Tiered KV Cache
/// §P3). The format is a straightforward serde representation of the
/// private fields; callers are expected to pair every deserialize with
/// a separate reconciliation pass if block IDs need to be mapped onto a
/// fresh allocator (the canonical use case is restoring a snapshot on a
/// new process where the KV pool layout has changed).
#[derive(Serialize, Deserialize)]
pub struct RadixCache {
    /// All nodes, indexed by stable integer IDs.
    nodes: Vec<Node>,
    /// Reclaimed node slots available for reuse after tombstone GC.
    #[serde(default)]
    free_nodes: Vec<usize>,
    /// Index 0 is always the virtual root (empty token sequence).
    // root index is always 0
    /// Block size: a node gets a block_id only when its cumulative path length
    /// ends on a `block_size` boundary. The edge itself may be shorter after a
    /// compressed-radix split. Must match the paged KV block size.
    block_size: usize,
    /// Monotonically increasing clock for LRU tracking.
    /// Runtime-only: snapshots keep structure/metadata, then restart the
    /// logical access epoch from zero after restore.
    #[serde(default, skip)]
    clock: u64,
    /// Optional lookup-time keepalive extension for already-soft-pinned blocks.
    #[serde(default)]
    soft_pin_keepalive_ticks: Option<u64>,
    /// O(1) reverse index: `block_id → self.nodes` slot. Always kept in sync
    /// with insert / evict / tombstone paths so `find_block_node_mut` doesn't
    /// have to scan `self.nodes`. Skipped by serde — rebuild via
    /// [`Self::rebuild_block_index`] after deserialization.
    #[serde(default, skip)]
    block_index: HashMap<BlockId, usize>,
}

impl RadixCache {
    /// Create a new empty radix cache.
    ///
    /// `block_size` must match the KV block size used by the block manager.
    pub fn new(block_size: usize) -> Self {
        Self::new_with_soft_pin_keepalive(block_size, None)
    }

    /// Create a radix cache that refreshes already-soft-pinned blocks on every
    /// successful lookup / lookup_or_stage hit.
    pub fn with_soft_pin_keepalive(block_size: usize, soft_pin_keepalive_ticks: u64) -> Self {
        Self::new_with_soft_pin_keepalive(block_size, Some(soft_pin_keepalive_ticks))
    }

    fn new_with_soft_pin_keepalive(
        block_size: usize,
        soft_pin_keepalive_ticks: Option<u64>,
    ) -> Self {
        assert!(block_size > 0, "block_size must be > 0");
        let root = Node::new(vec![], None, 0);
        Self {
            nodes: vec![root],
            free_nodes: Vec::new(),
            block_size,
            clock: 0,
            soft_pin_keepalive_ticks,
            block_index: HashMap::new(),
        }
    }

    /// Rebuild `block_index` from the current `nodes` array.
    ///
    /// Callers that deserialize a snapshot (which does not persist the index)
    /// or that bulk-mutate `block_id`s via [`Self::reconcile`] should
    /// call this once to restore O(1) `find_block_node_mut` behavior.
    pub fn rebuild_block_index(&mut self) {
        self.block_index.clear();
        for (idx, node) in self.nodes.iter().enumerate() {
            if let Some(bid) = node.block_id {
                self.block_index.insert(bid, idx);
            }
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn fingerprints_for_session(&self, session_id: &str) -> Vec<BlockFingerprint> {
        self.nodes
            .iter()
            .filter_map(|node| match (node.session_id.as_ref(), node.fingerprint) {
                (Some(node_session), Some(fingerprint)) if node_session.as_str() == session_id => {
                    Some(fingerprint)
                }
                _ => None,
            })
            .collect()
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn block_id_for_fingerprint(
        &self,
        fingerprint: BlockFingerprint,
    ) -> Option<BlockId> {
        self.nodes.iter().find_map(|node| {
            (node.fingerprint == Some(fingerprint))
                .then_some(node.block_id)
                .flatten()
        })
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// Returns the root node index (always 0).
    fn root() -> usize {
        0
    }

    /// Allocate a node slot, reusing a reclaimed tombstone if possible.
    fn alloc_node(&mut self, node: Node) -> usize {
        if let Some(idx) = self.free_nodes.pop() {
            self.nodes[idx] = node;
            idx
        } else {
            let idx = self.nodes.len();
            self.nodes.push(node);
            idx
        }
    }

    fn maybe_refresh_soft_pin(node: &mut Node, now: u64, soft_pin_keepalive_ticks: Option<u64>) {
        if let (Some(_), Some(ticks)) = (node.soft_pin_until, soft_pin_keepalive_ticks) {
            node.soft_pin_until = Some(now.saturating_add(ticks));
        }
    }

    fn edge_match_len(edge: &[u32], remaining: &[u32]) -> usize {
        edge.iter()
            .zip(remaining.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }

    fn child_edge_match(&self, child_idx: usize, remaining: &[u32]) -> (usize, usize) {
        let edge = &self.nodes[child_idx].tokens;
        (Self::edge_match_len(edge, remaining), edge.len())
    }

    fn next_block_boundary(&self, block_idx: usize) -> usize {
        (block_idx + 1) * self.block_size
    }

    fn full_block_end(&self, block_idx: usize, total_tokens: usize) -> Option<usize> {
        let boundary = self.next_block_boundary(block_idx);
        (boundary <= total_tokens).then_some(boundary)
    }

    fn truncated_block_end(&self, block_idx: usize, total_tokens: usize) -> usize {
        self.next_block_boundary(block_idx).min(total_tokens)
    }

    fn path_is_block_aligned(&self, path_len: usize) -> bool {
        path_len != 0 && path_len.is_multiple_of(self.block_size)
    }

    fn rounded_prefix_len(&self, walked_tokens: usize, matched_blocks: usize) -> usize {
        let matched_len = (matched_blocks * self.block_size).min(walked_tokens);
        (matched_len / self.block_size) * self.block_size
    }

    fn find_block_node_mut(&mut self, block: BlockId) -> Option<&mut Node> {
        let idx = *self.block_index.get(&block)?;
        self.nodes.get_mut(idx)
    }

    /// Retag one cached block while preserving its radix position and
    /// metadata. T1/T2 blocks use logical ids that must not collide with T0
    /// page ids after the original GPU pages are released.
    pub fn retag_block(&mut self, old: BlockId, new: BlockId) -> bool {
        if old == new {
            return self.block_index.contains_key(&old);
        }
        if self.block_index.contains_key(&new) {
            return false;
        }
        let Some(idx) = self.block_index.remove(&old) else {
            return false;
        };
        let Some(node) = self.nodes.get_mut(idx) else {
            return false;
        };
        if node.block_id != Some(old) {
            self.block_index.insert(old, idx);
            return false;
        }
        node.block_id = Some(new);
        self.block_index.insert(new, idx);
        true
    }

    // -------------------------------------------------------------------------
    // Lookup
    // -------------------------------------------------------------------------

    /// Find the longest cached prefix of `tokens`.
    ///
    /// Returns `(matched_len, matched_blocks)` where:
    /// - `matched_len` is the number of tokens in the matched prefix,
    ///   rounded down to a block boundary.
    /// - `matched_blocks` is the ordered list of block IDs covering those tokens.
    ///
    /// The matched nodes' `ref_count` is incremented — call `release(blocks)`
    /// when the request is done with them.
    pub fn lookup(&mut self, tokens: &[u32]) -> (usize, Vec<BlockId>) {
        let now = self.tick();
        let soft_pin_keepalive_ticks = self.soft_pin_keepalive_ticks;
        let mut node_idx = Self::root();
        let mut pos = 0;
        let mut matched_blocks = Vec::new();

        loop {
            // Update access time on each visited node.
            self.nodes[node_idx].last_access = now;

            if pos >= tokens.len() {
                break;
            }

            let next_token = tokens[pos];
            let Some(child_idx) = self.nodes[node_idx].children.get(&next_token).copied() else {
                break;
            };

            let remaining = &tokens[pos..];
            let (match_len, edge_len) = self.child_edge_match(child_idx, remaining);

            if match_len < edge_len {
                // Partial match — stop here. We cannot use a partially-matched node.
                break;
            }

            // Full edge match.
            pos += match_len;
            self.nodes[child_idx].last_access = now;

            if let Some(bid) = self.nodes[child_idx].block_id {
                self.nodes[child_idx].ref_count += 1;
                self.nodes[child_idx].hit_count = self.nodes[child_idx].hit_count.saturating_add(1);
                Self::maybe_refresh_soft_pin(
                    &mut self.nodes[child_idx],
                    now,
                    soft_pin_keepalive_ticks,
                );
                matched_blocks.push(bid);
            }

            node_idx = child_idx;
        }

        // Round down matched_len to block boundary.
        let rounded = self.rounded_prefix_len(pos, matched_blocks.len());
        // Trim blocks if we exceeded rounded boundary (shouldn't happen, but be safe).
        let block_count = rounded / self.block_size;
        if matched_blocks.len() > block_count {
            // Release extra refs.
            let extra: Vec<_> = matched_blocks.drain(block_count..).collect();
            self.dec_refs_by_block_id(&extra);
        }

        (rounded, matched_blocks)
    }

    /// Lookup cached blocks and classify whether they are ready in T0, need
    /// staging from another tier, or have fallen back to an index-only miss.
    pub fn lookup_or_stage(
        &mut self,
        tokens: &[u32],
        heuristics: LookupHeuristics,
    ) -> LookupOutcome {
        let now = self.tick();
        let soft_pin_keepalive_ticks = self.soft_pin_keepalive_ticks;
        let mut node_idx = Self::root();
        let mut pos = 0;
        let mut blocks = Vec::new();
        let mut recompute_advised = false;

        loop {
            self.nodes[node_idx].last_access = now;

            if pos >= tokens.len() {
                break;
            }

            let next_token = tokens[pos];
            let Some(child_idx) = self.nodes[node_idx].children.get(&next_token).copied() else {
                break;
            };

            let remaining = &tokens[pos..];
            let (match_len, edge_len) = self.child_edge_match(child_idx, remaining);

            if match_len < edge_len {
                break;
            }

            pos += match_len;

            let child = &mut self.nodes[child_idx];
            child.last_access = now;

            if let Some(block_id) = child.block_id {
                child.ref_count += 1;
                child.hit_count = child.hit_count.saturating_add(1);
                Self::maybe_refresh_soft_pin(child, now, soft_pin_keepalive_ticks);

                let byte_len = child.byte_len.max(self.block_size as u32);
                let hit_kind = match child.tier_location {
                    Some(BlockLocation::HostPinned { .. }) => HitKind::StagingFromHost,
                    Some(BlockLocation::Disk { .. } | BlockLocation::Remote { .. }) => {
                        HitKind::StagingFromDisk
                    }
                    Some(BlockLocation::Gpu { .. }) | None => HitKind::ReadyOnGpu,
                };
                let hit_kind = if child.entry_state == IndexEntryState::Ready {
                    hit_kind
                } else {
                    HitKind::Miss
                };

                if matches!(
                    hit_kind,
                    HitKind::StagingFromHost | HitKind::StagingFromDisk
                ) {
                    recompute_advised |=
                        heuristics.advise_recompute(hit_kind, self.block_size, byte_len as u64);
                }

                blocks.push(LookupBlock {
                    block_id: Some(block_id),
                    hit_kind,
                });
            } else if child.is_tombstone() {
                blocks.push(LookupBlock {
                    block_id: None,
                    hit_kind: HitKind::Miss,
                });
                break;
            }

            node_idx = child_idx;
        }

        let matched_blocks = blocks
            .iter()
            .filter(|block| !matches!(block.hit_kind, HitKind::Miss))
            .count();
        let matched_len = self.rounded_prefix_len(pos, matched_blocks);
        LookupOutcome::new(matched_len, blocks, recompute_advised)
    }

    /// Find the longest token-prefix path that is proven to belong to
    /// `session_id` by at least one block on that path.
    ///
    /// Shared ancestor blocks may have their `session_id` metadata overwritten by
    /// another session that published the same bytes later. The semantic session
    /// proof is therefore the deepest matching block on the path, not the first
    /// block. Returned blocks still cover the complete prefix up to that point so
    /// the scheduler can attach a contiguous KV prefix.
    pub fn lookup_session_prefix_or_stage(
        &mut self,
        session_id: &SessionId,
        tokens: &[u32],
        heuristics: LookupHeuristics,
    ) -> LookupOutcome {
        let mut node_idx = Self::root();
        let mut pos = 0;
        let mut matched_nodes = Vec::new();
        let mut last_session_block_count = 0;
        let mut last_session_pos = 0;

        loop {
            if pos >= tokens.len() {
                break;
            }

            let next_token = tokens[pos];
            let Some(child_idx) = self.nodes[node_idx].children.get(&next_token).copied() else {
                break;
            };

            let remaining = &tokens[pos..];
            let (match_len, edge_len) = self.child_edge_match(child_idx, remaining);
            if match_len < edge_len {
                break;
            }

            pos += match_len;
            let child = &self.nodes[child_idx];
            if child.block_id.is_some() {
                matched_nodes.push((child_idx, pos));
                if child.session_id.as_ref() == Some(session_id) {
                    last_session_block_count = matched_nodes.len();
                    last_session_pos = pos;
                }
            } else if child.is_tombstone() {
                break;
            }

            node_idx = child_idx;
        }

        if last_session_block_count == 0 {
            return LookupOutcome::new(0, Vec::new(), false);
        }

        let matched_len = self.rounded_prefix_len(last_session_pos, last_session_block_count);
        let block_count = matched_len / self.block_size;
        if block_count == 0 {
            return LookupOutcome::new(0, Vec::new(), false);
        }

        let now = self.tick();
        let block_size = self.block_size;
        let soft_pin_keepalive_ticks = self.soft_pin_keepalive_ticks;
        let mut blocks = Vec::with_capacity(block_count);
        let mut recompute_advised = false;

        for (node_idx, _) in matched_nodes.into_iter().take(block_count) {
            let child = &mut self.nodes[node_idx];
            child.last_access = now;
            let Some(block_id) = child.block_id else {
                blocks.push(LookupBlock {
                    block_id: None,
                    hit_kind: HitKind::Miss,
                });
                break;
            };

            child.ref_count += 1;
            child.hit_count = child.hit_count.saturating_add(1);
            Self::maybe_refresh_soft_pin(child, now, soft_pin_keepalive_ticks);

            let byte_len = child.byte_len.max(block_size as u32);
            let hit_kind = match child.tier_location {
                Some(BlockLocation::HostPinned { .. }) => HitKind::StagingFromHost,
                Some(BlockLocation::Disk { .. } | BlockLocation::Remote { .. }) => {
                    HitKind::StagingFromDisk
                }
                Some(BlockLocation::Gpu { .. }) | None => HitKind::ReadyOnGpu,
            };
            let hit_kind = if child.entry_state == IndexEntryState::Ready {
                hit_kind
            } else {
                HitKind::Miss
            };

            if matches!(
                hit_kind,
                HitKind::StagingFromHost | HitKind::StagingFromDisk
            ) {
                recompute_advised |=
                    heuristics.advise_recompute(hit_kind, block_size, byte_len as u64);
            }

            blocks.push(LookupBlock {
                block_id: Some(block_id),
                hit_kind,
            });
        }

        let matched_blocks = blocks
            .iter()
            .filter(|block| !matches!(block.hit_kind, HitKind::Miss))
            .count();
        let matched_len = self.rounded_prefix_len(matched_len, matched_blocks);
        LookupOutcome::new(matched_len, blocks, recompute_advised)
    }

    // -------------------------------------------------------------------------
    // Insert
    // -------------------------------------------------------------------------

    /// Insert (or update) a prefix of `tokens` with the corresponding block IDs.
    ///
    /// `blocks[i]` covers `tokens[i*block_size .. (i+1)*block_size]`.
    /// Trailing tokens (beyond `blocks.len() * block_size`) are not inserted.
    ///
    /// Returns the number of tokens actually inserted (may be less than
    /// `tokens.len()` if `tokens` is not a whole number of blocks).
    pub fn insert(&mut self, tokens: &[u32], blocks: &[BlockId]) -> usize {
        let fingerprints = vec![BlockFingerprint([0; 16]); blocks.len()];
        self.insert_with_fingerprints(tokens, blocks, &fingerprints)
    }

    /// Insert (or update) a prefix of `tokens` with block IDs and their
    /// per-block content fingerprints.
    pub fn insert_with_fingerprints(
        &mut self,
        tokens: &[u32],
        blocks: &[BlockId],
        fps: &[BlockFingerprint],
    ) -> usize {
        assert_eq!(blocks.len(), fps.len(), "blocks/fps length mismatch");
        if blocks.is_empty() || tokens.is_empty() {
            return 0;
        }

        let total_tokens = blocks.len() * self.block_size;
        let tokens = &tokens[..tokens.len().min(total_tokens)];
        let now = self.tick();

        let mut node_idx = Self::root();
        let mut pos = 0;
        let mut block_idx = 0;

        while pos < tokens.len() && block_idx < blocks.len() {
            let next_token = tokens[pos];

            if let Some(&child_idx) = self.nodes[node_idx].children.get(&next_token) {
                let remaining = &tokens[pos..];
                let (match_len, edge_len) = self.child_edge_match(child_idx, remaining);

                if match_len == edge_len {
                    // Full edge match — descend.
                    self.nodes[child_idx].last_access = now;
                    pos += edge_len;

                    // Advance block_idx when walking through a block-bearing node.
                    //
                    // The invariant is on cumulative path length, not edge
                    // length: a compressed-radix split may leave this node with
                    // a short edge while its path still ends exactly on the next
                    // sealed block boundary. So block progress is keyed off
                    // `block_id` presence plus the walked path, not `edge_len`.
                    if self.nodes[child_idx].block_id.is_some() && block_idx < blocks.len() {
                        debug_assert!(
                            self.path_is_block_aligned(pos),
                            "block-bearing node must end on a block boundary: child_idx={child_idx}, pos={pos}, block_size={}",
                            self.block_size,
                        );
                        let new_bid = blocks[block_idx];
                        if let Some(old_bid) = self.nodes[child_idx].block_id
                            && old_bid != new_bid
                        {
                            self.block_index.remove(&old_bid);
                        }
                        self.nodes[child_idx].block_id = Some(new_bid);
                        self.nodes[child_idx].fingerprint = Some(fps[block_idx]);
                        self.block_index.insert(new_bid, child_idx);
                        block_idx += 1;
                    }

                    node_idx = child_idx;
                } else {
                    // Partial match — split the existing edge.
                    let split_point = match_len;
                    let old_block = self.nodes[child_idx].block_id;
                    let old_ref_count = self.nodes[child_idx].ref_count;
                    let old_children: HashMap<u32, usize> =
                        std::mem::take(&mut self.nodes[child_idx].children);
                    let mut shared_tokens = std::mem::take(&mut self.nodes[child_idx].tokens);
                    let old_suffix = shared_tokens.split_off(split_point);

                    // Create an intermediate node for the shared prefix.
                    //
                    // Inherit the splitting child's `ref_count`. SGLang does
                    // the same in `_split_node` (see `radix_cache.py`,
                    // `new_node.lock_ref = child.lock_ref`): a request that
                    // was holding a lock on the deeper node is, by the
                    // path-lock invariant, also locking every ancestor on
                    // its matched path. Without this inheritance, a future
                    // garbage-collection pass that prunes orphaned non-block
                    // intermediates could remove a shared parent that is
                    // structurally part of an in-flight request.
                    let mut shared_node = Node::new(shared_tokens, None, now);
                    shared_node.ref_count = old_ref_count;
                    let shared_idx = self.alloc_node(shared_node);

                    // Rewire the original child to become a child of the shared node.
                    // `block_index[old_block]` already points at child_idx (if
                    // old_block was Some) — the slot itself hasn't moved, only
                    // its edge label, so no index fix-up is needed here.
                    self.nodes[child_idx].tokens = old_suffix;
                    self.nodes[child_idx].block_id = old_block;
                    self.nodes[child_idx].children = old_children;

                    let first_old = self.nodes[child_idx].tokens[0];
                    self.nodes[shared_idx].children.insert(first_old, child_idx);

                    // Replace the original child pointer with the shared node.
                    self.nodes[node_idx].children.insert(next_token, shared_idx);

                    // Place the caller's current block as a NEW sibling under
                    // shared_idx, completing the current sealed block window
                    // on the path. The edge ends at the next block boundary
                    // tracked by `block_idx`, not at `pos + block_size`: after
                    // descending through non-block shared intermediates `pos`
                    // can already be mid-block, but the aligned path boundary
                    // still lives at `next_block_boundary(block_idx)`.
                    if let Some(block_end) = self.full_block_end(block_idx, tokens.len()) {
                        let new_block_tokens = tokens[pos + match_len..block_end].to_vec();
                        let new_bid = blocks[block_idx];
                        let new_fp = fps[block_idx];
                        let mut new_node = Node::new(new_block_tokens, Some(new_bid), now);
                        new_node.fingerprint = Some(new_fp);
                        let new_idx = self.alloc_node(new_node);
                        self.block_index.insert(new_bid, new_idx);
                        let first_new = self.nodes[new_idx].tokens[0];
                        self.nodes[shared_idx].children.insert(first_new, new_idx);
                        pos = block_end;
                        block_idx += 1;
                        node_idx = new_idx;
                        continue;
                    }
                    // No room for an aligned block (partial tail or out of
                    // block_ids) — leave shared intermediate as the terminal
                    // insertion point and stop.
                    break;
                }
            } else {
                // No matching child — insert remaining tokens as a new subtree.
                // Edge windows are computed from the next block boundary
                // ((block_idx+1)*block_size), not pos+block_size. When the
                // walk landed here via a shared non-block-bearing parent, pos
                // is mid-block and the first new edge needs to complete the
                // current block (len < block_size, path-from-last-boundary
                // still sums to block_size so it takes the caller's block_id).
                while pos < tokens.len() && block_idx < blocks.len() {
                    let next_boundary = self.next_block_boundary(block_idx);
                    let end = self.truncated_block_end(block_idx, tokens.len());
                    let edge_tokens = tokens[pos..end].to_vec();
                    let is_full_block = end == next_boundary;
                    let (block_id, fingerprint) = if is_full_block {
                        let bid = blocks[block_idx];
                        let fp = fps[block_idx];
                        block_idx += 1;
                        (Some(bid), Some(fp))
                    } else {
                        (None, None)
                    };

                    let first_tok = edge_tokens[0];
                    let mut new_node = Node::new(edge_tokens, block_id, now);
                    new_node.fingerprint = fingerprint;
                    let new_idx = self.alloc_node(new_node);
                    if let Some(bid) = block_id {
                        debug_assert!(
                            self.path_is_block_aligned(end),
                            "new block-bearing node must end on a block boundary: end={end}, block_size={}",
                            self.block_size,
                        );
                        self.block_index.insert(bid, new_idx);
                    }

                    self.nodes[node_idx].children.insert(first_tok, new_idx);

                    pos = end;
                    node_idx = new_idx;

                    if !is_full_block {
                        break;
                    }
                }
                break;
            }
        }

        block_idx * self.block_size
    }

    /// Post-deserialization pass: reconcile saved fingerprints against a fresh
    /// pool mapping and rebuild the O(1) reverse block index.
    ///
    /// `BlockId` is only stable inside one allocator instance, so any
    /// save/load path that restores a `RadixCache` onto a new pool must call
    /// this before treating `block_id` as live. Nodes without fingerprints
    /// cannot be reconciled and are downgraded to tombstones.
    pub fn reconcile(
        &mut self,
        known: &std::collections::HashMap<BlockFingerprint, BlockId>,
    ) -> ReconcileReport {
        let mut report = ReconcileReport::default();
        for node in &mut self.nodes {
            if let Some(fp) = node.fingerprint {
                if let Some(&block_id) = known.get(&fp) {
                    node.block_id = Some(block_id);
                    report.remapped += 1;
                } else {
                    node.block_id = None;
                    report.tombstoned += 1;
                }
            } else {
                node.block_id = None;
                report.orphans_cleared += 1;
            }
        }
        self.rebuild_block_index();
        report
    }

    // -------------------------------------------------------------------------
    // Reference counting
    // -------------------------------------------------------------------------

    /// Release all blocks in `blocks`. Decrements the ref_count of nodes
    /// matching those blocks.
    pub fn release(&mut self, blocks: &[BlockId]) {
        self.dec_refs_by_block_id(blocks);
    }

    fn dec_refs_by_block_id(&mut self, blocks: &[BlockId]) {
        let block_set: std::collections::HashSet<BlockId> = blocks.iter().copied().collect();
        for node in &mut self.nodes {
            if let Some(bid) = node.block_id {
                if block_set.contains(&bid) && node.ref_count > 0 {
                    node.ref_count -= 1;
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Eviction
    // -------------------------------------------------------------------------

    /// Evict up to `n` least-recently-used leaf nodes with `ref_count == 0`.
    ///
    /// Returns the freed `BlockId`s so the block allocator can reclaim them.
    ///
    /// Evictions cascade through orphaned parents in a single call: when the
    /// LRU leaf is freed, its parent may become an active leaf itself (all
    /// its children are now virtually evicted). The loop picks the next LRU
    /// candidate from the updated structure until it has freed `n` blocks or
    /// run out of evictable candidates. Matches the shape of SGLang's
    /// iterative `evict()` (`radix_cache.py`: after `_delete_leaf`, re-push
    /// the now-childless parent onto the candidate heap).
    ///
    /// Intermediate non-block nodes (shared prefixes left behind by a split)
    /// still block cascading at their own level because the candidate filter
    /// requires `block_id.is_some()`. That leaves a structural residue but
    /// no correctness issue; a dedicated GC pass can prune them later.
    pub fn evict(&mut self, n: usize) -> Vec<BlockId> {
        if n == 0 {
            return vec![];
        }

        let mut freed: Vec<BlockId> = Vec::with_capacity(n);
        let mut evicted_set: std::collections::HashSet<usize> =
            std::collections::HashSet::with_capacity(n);

        while freed.len() < n {
            // Scan for the LRU evictable candidate: not root, not already
            // evicted, has a block_id, ref_count == 0, and is currently an
            // active leaf (all of its children are in `evicted_set`).
            let mut best: Option<(u64, usize)> = None;
            for (idx, node) in self.nodes.iter().enumerate() {
                if idx == Self::root() || evicted_set.contains(&idx) {
                    continue;
                }
                if node.ref_count != 0 || node.block_id.is_none() {
                    continue;
                }
                let active_leaf = self.node_is_active_leaf_for_tier(idx, None, &evicted_set);
                if !active_leaf {
                    continue;
                }
                match best {
                    Some((best_access, _)) if best_access <= node.last_access => {}
                    _ => best = Some((node.last_access, idx)),
                }
            }

            let Some((_, victim_idx)) = best else {
                break;
            };

            evicted_set.insert(victim_idx);
            if let Some(bid) = self.nodes[victim_idx].block_id.take() {
                self.block_index.remove(&bid);
                freed.push(bid);
            }
        }

        // Physically remove evicted children from their parents' child maps.
        // Unreachable nodes stay in `self.nodes` — their indices remain stable
        // for any in-flight lookup that may still reference them, and the
        // next eviction pass will not see them again because their children
        // have been rewired away.
        for node in &mut self.nodes {
            node.children
                .retain(|_, child_idx| !evicted_set.contains(child_idx));
        }

        self.gc_orphan_tombstones();

        freed
    }

    /// Evict up to `n` blocks using the provided policy.
    ///
    /// This is the M3b convergence path that lets production callers move off
    /// the hard-coded LRU loop and onto the shared [`EvictionPolicy`] trait.
    /// The implementation keeps the same iterative active-leaf semantics as
    /// [`Self::evict`]: after each victim is chosen, its parent can become an
    /// evictable active leaf in the same call.
    pub fn evict_with_policy(
        &mut self,
        policy: &dyn EvictionPolicy,
        signals: SchedulerSignals,
        n: usize,
        tier_filter: Option<Tier>,
    ) -> Vec<BlockId> {
        self.evict_with_policy_for_intent(
            policy,
            signals,
            n,
            tier_filter,
            BlockSelectionIntent::Evict,
        )
    }

    /// Evict up to `n` blocks using the provided policy and selection intent.
    ///
    /// `BlockSelectionIntent::Drain` is used by T1 demotion headroom: it still
    /// keeps the active-leaf and ref-count invariants, but does not let a
    /// lookup soft-pin on an old host leaf force the scheduler to drop a newer
    /// GPU prefix block instead.
    pub fn evict_with_policy_for_intent(
        &mut self,
        policy: &dyn EvictionPolicy,
        signals: SchedulerSignals,
        n: usize,
        tier_filter: Option<Tier>,
        intent: BlockSelectionIntent,
    ) -> Vec<BlockId> {
        if n == 0 {
            return vec![];
        }

        let mut freed: Vec<BlockId> = Vec::with_capacity(n);
        let mut evicted_set: std::collections::HashSet<usize> =
            std::collections::HashSet::with_capacity(n);
        let now = self.clock;

        while freed.len() < n {
            let mut best: Option<(f32, usize)> = None;

            for (idx, node) in self.nodes.iter().enumerate() {
                if idx == Self::root() || evicted_set.contains(&idx) {
                    continue;
                }

                if node.block_id.is_none() {
                    continue;
                }
                if node.ref_count != 0 {
                    continue;
                }
                if let Some(tier) = tier_filter
                    && node.tier_location.as_ref().map(BlockLocation::tier) != Some(tier)
                {
                    continue;
                }
                if Self::selection_pin_active(node, tier_filter, intent, now) {
                    continue;
                }
                if node.entry_state != IndexEntryState::Ready {
                    continue;
                }
                if matches!(
                    intent,
                    BlockSelectionIntent::Spill | BlockSelectionIntent::Drain
                ) && !matches!(node.store_state, StoreState::Idle | StoreState::Failed)
                {
                    continue;
                }

                let active_leaf = self.node_is_active_leaf_for_tier(idx, tier_filter, &evicted_set);
                if !active_leaf {
                    continue;
                }

                let slot = match node.tier_location {
                    Some(BlockLocation::Gpu { slot }) => slot,
                    _ => 0,
                };
                let candidate = EvictionCandidate {
                    slot,
                    tokens: self.block_size as u32,
                    last_access_step: node.last_access,
                    hit_count: node.hit_count,
                    prefix_depth: node.tokens.len() as u32,
                    pinned: false,
                };
                let score = policy.score(candidate, signals);
                if !score.is_finite() {
                    continue;
                }

                let keep_existing_best = match best {
                    Some((best_score, best_idx)) => {
                        let ordering = best_score.total_cmp(&score);
                        ordering.is_lt() || (ordering.is_eq() && best_idx <= idx)
                    }
                    None => false,
                };
                if !keep_existing_best {
                    best = Some((score, idx));
                }
            }

            let Some((_, victim_idx)) = best else {
                break;
            };

            evicted_set.insert(victim_idx);
            if let Some(bid) = self.nodes[victim_idx].block_id.take() {
                self.block_index.remove(&bid);
                freed.push(bid);
            }
        }

        for node in &mut self.nodes {
            node.children
                .retain(|_, child_idx| !evicted_set.contains(child_idx));
        }

        self.gc_orphan_tombstones();

        freed
    }

    /// Select up to `n` evictable blocks using the provided policy without
    /// mutating the radix tree.
    ///
    /// This is the scheduler's demote/spill picker: it wants the same scoring
    /// as eviction, but it needs to move blocks across tiers without deleting
    /// their index entries.
    pub fn select_blocks_with_policy(
        &self,
        policy: &dyn EvictionPolicy,
        signals: SchedulerSignals,
        n: usize,
        tier_filter: Option<Tier>,
        intent: BlockSelectionIntent,
        require_host_swap_eligible: bool,
    ) -> Vec<BlockId> {
        if n == 0 {
            return vec![];
        }

        let now = self.clock;
        let mut selected = Vec::with_capacity(n);
        let mut selected_nodes = std::collections::HashSet::with_capacity(n);

        while selected.len() < n {
            let mut best: Option<(f32, usize, BlockId)> = None;
            for idx in 1..self.nodes.len() {
                if selected_nodes.contains(&idx) {
                    continue;
                }
                let node = &self.nodes[idx];
                let Some(block_id) = node.block_id else {
                    continue;
                };
                if node.ref_count != 0 {
                    continue;
                }
                if require_host_swap_eligible && !node.host_swap_eligible {
                    continue;
                }
                if Self::selection_pin_active(node, tier_filter, intent, now) {
                    continue;
                }
                if node.entry_state != IndexEntryState::Ready {
                    continue;
                }
                if matches!(
                    intent,
                    BlockSelectionIntent::Spill | BlockSelectionIntent::Drain
                ) && !matches!(node.store_state, StoreState::Idle | StoreState::Failed)
                {
                    continue;
                }
                if let Some(tier) = tier_filter
                    && node.tier_location.as_ref().map(BlockLocation::tier) != Some(tier)
                {
                    continue;
                }
                if !self.node_is_active_leaf_for_tier(idx, tier_filter, &selected_nodes) {
                    continue;
                }

                let slot = match node.tier_location {
                    Some(BlockLocation::Gpu { slot }) => slot,
                    _ => 0,
                };
                let candidate = EvictionCandidate {
                    slot,
                    tokens: self.block_size as u32,
                    last_access_step: node.last_access,
                    hit_count: node.hit_count,
                    prefix_depth: node.tokens.len() as u32,
                    pinned: false,
                };
                let score = policy.score(candidate, signals);
                if !score.is_finite() {
                    continue;
                }
                let keep_existing_best = match best {
                    Some((best_score, best_idx, best_bid)) => {
                        let ordering = best_score.total_cmp(&score);
                        ordering.is_lt()
                            || (ordering.is_eq() && (best_bid.0, best_idx) <= (block_id.0, idx))
                    }
                    None => false,
                };
                if !keep_existing_best {
                    best = Some((score, idx, block_id));
                }
            }
            let Some((_, idx, block_id)) = best else {
                break;
            };
            selected_nodes.insert(idx);
            selected.push(block_id);
        }

        selected
    }

    /// Select a read-only sparse KV page view for MagicDec-style drafting.
    ///
    /// This slot-wide selector is only a fallback for callers that do not have
    /// the active request tokens. Scheduler paths should prefer
    /// [`Self::select_sparse_pages_for_draft_tokens`] so divergent same-slot
    /// radix branches cannot enter one request's sparse view.
    ///
    /// This does **not** mutate radix refs, LRU clocks, locations, or eviction
    /// state. The returned block IDs are GPU-resident blocks for `slot_idx`.
    /// Selection is StreamingLLM-shaped: include up to `top_k` hottest blocks
    /// by radix LRU recency, then always include the most recent
    /// `recent_tokens` worth of blocks by path depth. The recent window may
    /// make the result larger than `top_k`.
    #[allow(dead_code)]
    pub(crate) fn select_sparse_pages_for_draft(
        &self,
        slot_idx: usize,
        recent_tokens: usize,
        top_k: usize,
    ) -> Vec<BlockId> {
        fn collect_slot(
            cache: &RadixCache,
            node_idx: usize,
            parent_path_len: usize,
            slot_idx: u32,
            out: &mut Vec<SparseDraftCandidate>,
        ) {
            let node = &cache.nodes[node_idx];
            let path_len = parent_path_len + node.tokens.len();
            if let (Some(block_id), Some(BlockLocation::Gpu { slot })) =
                (node.block_id, node.tier_location.as_ref())
                && *slot == slot_idx
            {
                out.push(SparseDraftCandidate {
                    block_id,
                    path_len,
                    last_access: node.last_access,
                });
            }
            for &child_idx in node.children.values() {
                collect_slot(cache, child_idx, path_len, slot_idx, out);
            }
        }

        let Ok(slot_idx) = u32::try_from(slot_idx) else {
            return Vec::new();
        };
        let mut candidates = Vec::new();
        collect_slot(self, Self::root(), 0, slot_idx, &mut candidates);
        self.select_sparse_pages_from_candidates(candidates, recent_tokens, top_k)
    }

    /// Select a sparse draft view along one request token path.
    ///
    /// Prefer this over [`Self::select_sparse_pages_for_draft`] when active
    /// request tokens are available: it avoids pulling same-slot blocks from
    /// stale divergent radix branches.
    #[allow(dead_code)]
    pub(crate) fn select_sparse_pages_for_draft_tokens(
        &self,
        slot_idx: usize,
        tokens: &[u32],
        recent_tokens: usize,
        top_k: usize,
    ) -> Vec<BlockId> {
        self.select_sparse_pages_for_draft_tokens_with_attached(
            slot_idx,
            tokens,
            recent_tokens,
            top_k,
            &[],
        )
    }

    /// Select a sparse draft view along one request token path, treating
    /// directly attached prefix blocks as active for `slot_idx` even though
    /// their radix metadata still points at the source slot.
    #[allow(dead_code)]
    pub(crate) fn select_sparse_pages_for_draft_tokens_with_attached(
        &self,
        slot_idx: usize,
        tokens: &[u32],
        recent_tokens: usize,
        top_k: usize,
        attached_blocks: &[BlockId],
    ) -> Vec<BlockId> {
        let Ok(slot_idx) = u32::try_from(slot_idx) else {
            return Vec::new();
        };
        if tokens.is_empty() {
            return self.select_sparse_pages_for_draft(slot_idx as usize, recent_tokens, top_k);
        }

        let attached_blocks: std::collections::HashSet<BlockId> =
            attached_blocks.iter().copied().collect();
        let mut candidates = Vec::new();
        let mut node_idx = Self::root();
        let mut pos = 0;
        loop {
            if pos >= tokens.len() {
                break;
            }
            let Some(child_idx) = self.nodes[node_idx].children.get(&tokens[pos]).copied() else {
                break;
            };
            let remaining = &tokens[pos..];
            let (match_len, edge_len) = self.child_edge_match(child_idx, remaining);
            if match_len < edge_len {
                break;
            }
            pos += match_len;
            let node = &self.nodes[child_idx];
            if let (Some(block_id), Some(BlockLocation::Gpu { slot })) =
                (node.block_id, node.tier_location.as_ref())
                && (*slot == slot_idx || attached_blocks.contains(&block_id))
            {
                candidates.push(SparseDraftCandidate {
                    block_id,
                    path_len: pos,
                    last_access: node.last_access,
                });
            }
            node_idx = child_idx;
        }
        self.select_sparse_pages_from_candidates(candidates, recent_tokens, top_k)
    }

    #[allow(dead_code)]
    fn select_sparse_pages_from_candidates(
        &self,
        mut candidates: Vec<SparseDraftCandidate>,
        recent_tokens: usize,
        top_k: usize,
    ) -> Vec<BlockId> {
        if candidates.is_empty() || (recent_tokens == 0 && top_k == 0) {
            return Vec::new();
        }
        let mut selected = Vec::new();
        let mut selected_set = std::collections::HashSet::new();

        candidates.sort_by(|a, b| {
            b.last_access
                .cmp(&a.last_access)
                .then_with(|| a.block_id.0.cmp(&b.block_id.0))
        });
        for candidate in candidates.iter().take(top_k) {
            if selected_set.insert(candidate.block_id) {
                selected.push(candidate.block_id);
            }
        }

        let recent_blocks = recent_tokens
            .div_ceil(self.block_size)
            .min(candidates.len());
        candidates.sort_by(|a, b| {
            b.path_len
                .cmp(&a.path_len)
                .then_with(|| a.block_id.0.cmp(&b.block_id.0))
        });
        for candidate in candidates.iter().take(recent_blocks) {
            if selected_set.insert(candidate.block_id) {
                selected.push(candidate.block_id);
            }
        }

        selected
    }

    /// Mark non-selected GPU blocks for a sparse draft view.
    ///
    /// P2.B.2 is deliberately metadata-only: this does not remove radix nodes,
    /// decrement refs, or release physical paged-KV pages. It records that the
    /// draft view may ignore these blocks while the full verifier keeps the
    /// canonical KV intact. Only unpinned ready blocks for `slot_idx` can add
    /// eviction headroom. Explicitly attached cross-slot blocks may be marked
    /// even while ref-counted so the sparse draft can omit them, but the normal
    /// `node_is_reclaimable_now` guard keeps them out of evictable accounting.
    #[allow(dead_code)]
    pub(crate) fn drop_pages_for_sparse_view(
        &mut self,
        slot_idx: usize,
        sparse_page_ids: &[BlockId],
        attached_page_ids: &[BlockId],
    ) -> Vec<BlockId> {
        let Ok(slot_idx) = u32::try_from(slot_idx) else {
            return Vec::new();
        };
        let keep: std::collections::HashSet<BlockId> = sparse_page_ids.iter().copied().collect();
        let attached: std::collections::HashSet<BlockId> =
            attached_page_ids.iter().copied().collect();
        let now = self.clock;
        let mut dropped = Vec::new();

        for node in &mut self.nodes {
            let Some(block_id) = node.block_id else {
                continue;
            };
            if keep.contains(&block_id) {
                node.sparse_dropped = false;
                continue;
            }
            let active_for_slot = matches!(
                node.tier_location.as_ref(),
                Some(BlockLocation::Gpu { slot }) if *slot == slot_idx
            ) || attached.contains(&block_id);
            if !active_for_slot {
                continue;
            }
            let attached_for_slot = attached.contains(&block_id);
            let ref_ok = node.ref_count == 0 || attached_for_slot;
            if ref_ok
                && node.entry_state == IndexEntryState::Ready
                && node.soft_pin_until.is_none_or(|deadline| deadline <= now)
            {
                node.sparse_dropped = true;
                dropped.push(block_id);
            } else {
                node.sparse_dropped = false;
            }
        }

        dropped.sort_by_key(|block| block.0);
        dropped
    }

    fn selection_pin_active(
        node: &Node,
        tier_filter: Option<Tier>,
        intent: BlockSelectionIntent,
        now: u64,
    ) -> bool {
        match intent {
            BlockSelectionIntent::Evict => {
                node.soft_pin_until.is_some_and(|deadline| deadline > now)
            }
            BlockSelectionIntent::Spill => match tier_filter {
                Some(Tier::HostPinned) => node
                    .host_spill_pin_until
                    .is_some_and(|deadline| deadline > now),
                _ => node.soft_pin_until.is_some_and(|deadline| deadline > now),
            },
            BlockSelectionIntent::Drain => false,
        }
    }

    fn node_matches_tier_filter(&self, idx: usize, tier_filter: Option<Tier>) -> bool {
        let Some(node) = self.nodes.get(idx) else {
            return false;
        };
        node.block_id.is_some()
            && tier_filter.is_none_or(|tier| {
                node.tier_location.as_ref().map(BlockLocation::tier) == Some(tier)
            })
    }

    fn subtree_has_unselected_block_in_tier(
        &self,
        idx: usize,
        tier_filter: Option<Tier>,
        selected_nodes: &std::collections::HashSet<usize>,
    ) -> bool {
        if selected_nodes.contains(&idx) {
            return false;
        }
        if self.node_matches_tier_filter(idx, tier_filter) {
            return true;
        }
        self.nodes.get(idx).is_some_and(|node| {
            node.children.values().any(|&child_idx| {
                self.subtree_has_unselected_block_in_tier(child_idx, tier_filter, selected_nodes)
            })
        })
    }

    fn node_is_active_leaf_for_tier(
        &self,
        idx: usize,
        tier_filter: Option<Tier>,
        selected_nodes: &std::collections::HashSet<usize>,
    ) -> bool {
        self.nodes.get(idx).is_some_and(|node| {
            node.children.values().all(|&child_idx| {
                !self.subtree_has_unselected_block_in_tier(child_idx, tier_filter, selected_nodes)
            })
        })
    }

    /// Reclaim orphaned tombstones after eviction.
    ///
    /// A node is reclaimable when it is:
    /// - not the root,
    /// - blockless (`block_id == None`),
    /// - unpinned (`ref_count == 0`), and
    /// - childless once already-reclaimable descendants are ignored.
    ///
    /// Reclaimed slots are pushed onto `free_nodes` so later inserts can reuse
    /// them instead of letting the backing `Vec<Node>` grow without bound.
    fn gc_orphan_tombstones(&mut self) {
        use std::collections::HashSet;

        let mut reclaimable: HashSet<usize> = HashSet::new();

        loop {
            let mut progressed = false;

            for idx in 1..self.nodes.len() {
                if reclaimable.contains(&idx) {
                    continue;
                }

                let node = &self.nodes[idx];
                if node.block_id.is_some() || node.ref_count != 0 {
                    continue;
                }

                if node
                    .children
                    .iter()
                    .all(|(_, child_idx)| reclaimable.contains(child_idx))
                {
                    reclaimable.insert(idx);
                    progressed = true;
                }
            }

            if !progressed {
                break;
            }
        }

        if reclaimable.is_empty() {
            return;
        }

        for node in &mut self.nodes {
            node.children
                .retain(|_, child_idx| !reclaimable.contains(child_idx));
        }

        let mut reclaimed: Vec<usize> = reclaimable.into_iter().collect();
        reclaimed.sort_unstable();
        for idx in reclaimed {
            let node = &mut self.nodes[idx];
            node.tokens.clear();
            node.block_id = None;
            node.ref_count = 0;
            node.last_access = 0;
            node.entry_state = IndexEntryState::Ready;
            node.store_state = StoreState::Idle;
            node.children.clear();
            self.free_nodes.push(idx);
        }
    }

    // -------------------------------------------------------------------------
    // Stats
    // -------------------------------------------------------------------------

    /// Block size the cache was constructed with (in tokens).
    ///
    /// Exposed so callers that generate block ids at insert time can
    /// mint exactly the right number of ids per prompt without
    /// duplicating the constant.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Total number of nodes in the tree (including root).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of reclaimed node slots waiting to be reused.
    pub fn free_node_count(&self) -> usize {
        self.free_nodes.len()
    }

    /// Number of cached blocks (nodes with a `block_id`).
    pub fn cached_block_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.block_id.is_some()).count()
    }

    fn node_is_reclaimable_now(&self, node: &Node, tier_filter: Option<Tier>) -> bool {
        let now = self.clock;
        node.block_id.is_some()
            && node.ref_count == 0
            && node.entry_state == IndexEntryState::Ready
            && node.soft_pin_until.is_none_or(|deadline| deadline <= now)
            && tier_filter.is_none_or(|tier| {
                node.tier_location.as_ref().map(BlockLocation::tier) == Some(tier)
            })
    }

    fn node_counts_as_sparse_dropped(&self, node: &Node, tier_filter: Option<Tier>) -> bool {
        node.sparse_dropped && self.node_is_reclaimable_now(node, tier_filter)
    }

    fn node_is_cascade_evictable_now(&self, idx: usize, tier_filter: Option<Tier>) -> bool {
        let Some(node) = self.nodes.get(idx) else {
            return false;
        };
        (self.node_counts_as_sparse_dropped(node, tier_filter)
            || self.node_is_reclaimable_now(node, tier_filter))
            && node
                .children
                .values()
                .all(|&child_idx| self.node_is_cascade_evictable_now(child_idx, tier_filter))
    }

    fn collect_cascade_evictable_blocks(
        &self,
        idx: usize,
        tier_filter: Option<Tier>,
        out: &mut Vec<BlockId>,
    ) -> bool {
        let Some(node) = self.nodes.get(idx) else {
            return false;
        };
        let mut children_evictable = true;
        for &child_idx in node.children.values() {
            children_evictable &=
                self.collect_cascade_evictable_blocks(child_idx, tier_filter, out);
        }
        let evictable = (self.node_counts_as_sparse_dropped(node, tier_filter)
            || self.node_is_reclaimable_now(node, tier_filter))
            && children_evictable;
        if evictable && let Some(block_id) = node.block_id {
            out.push(block_id);
        }
        evictable
    }

    /// Cached blocks that a single cascade eviction pass can reclaim now.
    pub fn cascade_evictable_blocks(&self, tier_filter: Option<Tier>) -> Vec<BlockId> {
        let mut blocks = Vec::new();
        for &child_idx in self.nodes[Self::root()].children.values() {
            self.collect_cascade_evictable_blocks(child_idx, tier_filter, &mut blocks);
        }
        blocks
    }

    /// True when a cached block can be reclaimed from the requested tier now.
    pub fn is_block_evictable(&self, block: BlockId, tier_filter: Option<Tier>) -> bool {
        let Some(idx) = self.block_index.get(&block).copied() else {
            return false;
        };
        self.node_is_cascade_evictable_now(idx, tier_filter)
    }

    /// Number of cached block tokens that the eviction path can reclaim now.
    ///
    /// This mirrors the lock-ref discipline used by eviction: only ready,
    /// block-bearing nodes with `ref_count == 0` and no active soft pin are
    /// counted. Each block-bearing node represents one sealed cache block, even
    /// if compressed-radix splitting made its local edge shorter than
    /// `block_size`.
    pub fn evictable_tokens(&self) -> usize {
        self.cascade_evictable_blocks(None)
            .len()
            .saturating_mul(self.block_size)
    }

    /// Number of KV pool pages represented by currently evictable cache blocks.
    pub fn evictable_pages(&self, page_size: usize) -> usize {
        self.evictable_tokens().div_ceil(page_size.max(1))
    }

    /// Stamp the physical location for a cached block.
    pub fn set_block_location(&mut self, block: BlockId, location: BlockLocation) -> bool {
        self.find_block_node_mut(block)
            .map(|node| {
                node.tier_location = Some(location);
                if !matches!(node.tier_location, Some(BlockLocation::HostPinned { .. })) {
                    node.host_spill_pin_until = None;
                }
            })
            .is_some()
    }

    /// Read back the current metadata snapshot for a cached block.
    pub fn block_metadata(&self, block: BlockId) -> Option<BlockMetadata> {
        let idx = *self.block_index.get(&block)?;
        let node = self.nodes.get(idx)?;
        Some(BlockMetadata {
            location: node.tier_location.clone(),
            byte_len: node.byte_len,
            session_id: node.session_id.clone(),
            host_swap_eligible: node.host_swap_eligible,
            fingerprint: node.fingerprint,
            soft_pin_until: node.soft_pin_until,
            host_spill_pin_until: node.host_spill_pin_until,
            entry_state: node.entry_state,
            store_state: node.store_state,
            hit_count: node.hit_count,
            ref_count: node.ref_count,
        })
    }

    /// Stamp the byte length metadata for a cached block.
    pub fn set_block_byte_len(&mut self, block: BlockId, byte_len: u32) -> bool {
        self.find_block_node_mut(block)
            .map(|node| {
                node.byte_len = byte_len;
            })
            .is_some()
    }

    /// Stamp the session affinity metadata for a cached block.
    pub fn set_block_session_id(&mut self, block: BlockId, session_id: Option<SessionId>) -> bool {
        self.find_block_node_mut(block)
            .map(|node| {
                node.session_id = session_id;
            })
            .is_some()
    }

    /// Stamp or clear the logical soft-pin deadline for a cached block.
    pub fn set_block_soft_pin_until(
        &mut self,
        block: BlockId,
        soft_pin_until: Option<u64>,
    ) -> bool {
        self.find_block_node_mut(block)
            .map(|node| {
                node.soft_pin_until = soft_pin_until;
            })
            .is_some()
    }

    pub fn set_block_host_spill_pin_until(
        &mut self,
        block: BlockId,
        host_spill_pin_until: Option<u64>,
    ) -> bool {
        self.find_block_node_mut(block)
            .map(|node| {
                node.host_spill_pin_until = host_spill_pin_until;
            })
            .is_some()
    }

    /// Update multiple metadata fields for one cached block with a single node
    /// lookup in the radix backing store.
    pub fn update_block_metadata(&mut self, block: BlockId, update: BlockMetadataUpdate) -> bool {
        self.find_block_node_mut(block)
            .map(|node| {
                if let Some(location) = update.location {
                    node.tier_location = Some(location);
                    if !matches!(node.tier_location, Some(BlockLocation::HostPinned { .. })) {
                        node.host_spill_pin_until = None;
                    }
                }
                if let Some(byte_len) = update.byte_len {
                    node.byte_len = byte_len;
                }
                if let Some(session_id) = update.session_id {
                    node.session_id = session_id;
                }
                if let Some(host_swap_eligible) = update.host_swap_eligible {
                    node.host_swap_eligible = host_swap_eligible;
                }
                if let Some(soft_pin_until) = update.soft_pin_until {
                    node.soft_pin_until = soft_pin_until;
                }
                if let Some(host_spill_pin_until) = update.host_spill_pin_until {
                    node.host_spill_pin_until = host_spill_pin_until;
                }
                if let Some(entry_state) = update.entry_state {
                    node.entry_state = entry_state;
                }
            })
            .is_some()
    }

    pub fn set_block_index_state(&mut self, block: BlockId, entry_state: IndexEntryState) -> bool {
        self.find_block_node_mut(block)
            .map(|node| {
                node.entry_state = entry_state;
            })
            .is_some()
    }

    pub fn set_block_store_state(&mut self, block: BlockId, store_state: StoreState) -> bool {
        self.find_block_node_mut(block)
            .map(|node| {
                node.store_state = store_state;
            })
            .is_some()
    }

    pub fn insert_pending(&mut self, block: BlockId) -> bool {
        self.set_block_index_state(block, IndexEntryState::Pending)
    }

    pub fn mark_block_evicting(&mut self, block: BlockId) -> bool {
        self.set_block_index_state(block, IndexEntryState::Evicting)
    }

    pub fn commit_ready(&mut self, block: BlockId, location: Option<BlockLocation>) -> bool {
        self.update_block_metadata(
            block,
            BlockMetadataUpdate {
                location,
                entry_state: Some(IndexEntryState::Ready),
                ..BlockMetadataUpdate::default()
            },
        )
    }

    pub fn mark_block_store_pending(&mut self, block: BlockId) -> bool {
        self.set_block_store_state(block, StoreState::Pending)
    }

    pub fn mark_block_storing(&mut self, block: BlockId) -> bool {
        self.set_block_store_state(block, StoreState::Storing)
    }

    pub fn mark_block_stored(&mut self, block: BlockId, location: Option<BlockLocation>) -> bool {
        let updated = if let Some(location) = location {
            self.set_block_location(block, location)
        } else {
            self.block_index.contains_key(&block)
        };
        updated && self.set_block_store_state(block, StoreState::Stored)
    }

    pub fn mark_block_store_failed(&mut self, block: BlockId) -> bool {
        self.set_block_store_state(block, StoreState::Failed)
    }

    pub fn batch_lookup_or_stage<'a, I>(
        &mut self,
        prefixes: I,
        heuristics: LookupHeuristics,
    ) -> Vec<LookupOutcome>
    where
        I: IntoIterator<Item = &'a [u32]>,
    {
        prefixes
            .into_iter()
            .map(|prefix| self.lookup_or_stage(prefix, heuristics))
            .collect()
    }

    /// Extend the deadline of an already-soft-pinned block relative to the
    /// cache's current logical clock.
    ///
    /// Returns `true` only when the block exists and already had a
    /// `soft_pin_until` value to refresh.
    pub fn bump_soft_pin(&mut self, block: BlockId, ticks_ahead: u64) -> bool {
        let deadline = self.clock.saturating_add(ticks_ahead);
        match self.find_block_node_mut(block) {
            Some(node) if node.soft_pin_until.is_some() => {
                node.soft_pin_until = Some(deadline);
                true
            }
            _ => false,
        }
    }

    /// Current logical clock value used by lookup/insert/eviction bookkeeping.
    pub fn logical_clock(&self) -> u64 {
        self.clock
    }
}

#[cfg(test)]
#[path = "prefix_cache/tests.rs"]
mod tests;
