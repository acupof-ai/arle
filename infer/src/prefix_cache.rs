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
//! Tokens are grouped into fixed-size blocks (e.g. 16 tokens). A node's block
//! is only valid when the node holds exactly `block_size` tokens. Fractional
//! trailing blocks are not cached — the final partial block must be re-prefilled.
//!
//! # Eviction
//!
//! When `evict(n)` is called, the cache removes the `n` least-recently-used
//! leaf nodes that have `ref_count == 0`, freeing their block IDs back to the
//! caller.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::kv_tier::{
    BlockLocation, HitKind, LookupBlock, LookupHeuristics, LookupOutcome, StagePlanner,
    StageRequest,
};
use crate::scheduler::policy::{EvictionCandidate, EvictionPolicy, SchedulerSignals};
use crate::types::{BlockFingerprint, SessionId};

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
/// `session_id` / `soft_pin_until` use `Option<Option<T>>` so callers can
/// distinguish "leave untouched" from "explicitly clear".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BlockMetadataUpdate {
    pub location: Option<BlockLocation>,
    pub byte_len: Option<u32>,
    pub session_id: Option<Option<SessionId>>,
    pub soft_pin_until: Option<Option<u64>>,
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
    /// GPU block ID cached for this node. `None` if tokens < block_size or
    /// if this node has not yet been committed to the GPU.
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
            fingerprint: None,
            byte_len: 0,
            soft_pin_until: None,
            children: HashMap::new(),
        }
    }

    fn is_tombstone(&self) -> bool {
        self.block_id.is_none()
            && (self.tier_location.is_some()
                || self.session_id.is_some()
                || self.fingerprint.is_some()
                || self.byte_len != 0
                || self.soft_pin_until.is_some())
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
    /// Block size: a node gets a block_id only when it holds exactly
    /// `block_size` tokens. Must match the paged KV block size.
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

    /// Rebuild [`Self::block_index`] from the current `nodes` array.
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

    fn find_block_node_mut(&mut self, block: BlockId) -> Option<&mut Node> {
        let idx = *self.block_index.get(&block)?;
        self.nodes.get_mut(idx)
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

            // Check how many tokens of the child's edge match.
            let child_tokens = self.nodes[child_idx].tokens.clone();
            let remaining = &tokens[pos..];
            let match_len = child_tokens
                .iter()
                .zip(remaining.iter())
                .take_while(|(a, b)| a == b)
                .count();

            if match_len < child_tokens.len() {
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
        let matched_len = (matched_blocks.len() * self.block_size).min(pos);
        let rounded = (matched_len / self.block_size) * self.block_size;
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
    ///
    /// This is the local M3b contract surface. It does **not** wire the
    /// scheduler to the coordinator yet; it only makes the lookup result
    /// explicit and testable on the no-cuda / metal lanes.
    pub fn lookup_or_stage(
        &mut self,
        tokens: &[u32],
        heuristics: LookupHeuristics,
        planner: Option<&dyn StagePlanner>,
    ) -> LookupOutcome {
        let now = self.tick();
        let soft_pin_keepalive_ticks = self.soft_pin_keepalive_ticks;
        let mut node_idx = Self::root();
        let mut pos = 0;
        let mut blocks = Vec::new();
        let mut stage_requests = Vec::new();

        loop {
            self.nodes[node_idx].last_access = now;

            if pos >= tokens.len() {
                break;
            }

            let next_token = tokens[pos];
            let Some(child_idx) = self.nodes[node_idx].children.get(&next_token).copied() else {
                break;
            };

            let child_tokens = self.nodes[child_idx].tokens.clone();
            let remaining = &tokens[pos..];
            let match_len = child_tokens
                .iter()
                .zip(remaining.iter())
                .take_while(|(a, b)| a == b)
                .count();

            if match_len < child_tokens.len() {
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
                let (hit_kind, request) = match child.tier_location.clone() {
                    Some(BlockLocation::HostPinned { .. }) => (
                        HitKind::StagingFromHost,
                        child.tier_location.clone().map(|from| StageRequest {
                            block_id,
                            from,
                            byte_len,
                        }),
                    ),
                    Some(BlockLocation::Disk { .. } | BlockLocation::Remote { .. }) => (
                        HitKind::StagingFromDisk,
                        child.tier_location.clone().map(|from| StageRequest {
                            block_id,
                            from,
                            byte_len,
                        }),
                    ),
                    Some(BlockLocation::Gpu { .. }) | None => (HitKind::ReadyOnGpu, None),
                };

                if let Some(request) = request {
                    stage_requests.push(request);
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
        let matched_len = (matched_blocks * self.block_size).min(pos);

        let recompute_advised = stage_requests.iter().any(|request| {
            let hit_kind = match request.from {
                BlockLocation::HostPinned { .. } => HitKind::StagingFromHost,
                BlockLocation::Disk { .. } | BlockLocation::Remote { .. } => {
                    HitKind::StagingFromDisk
                }
                BlockLocation::Gpu { .. } => HitKind::ReadyOnGpu,
            };
            heuristics.advise_recompute(hit_kind, self.block_size, request.byte_len as u64)
        });

        let staging_ticket = if stage_requests.is_empty() || recompute_advised {
            None
        } else {
            planner.and_then(|planner| planner.stage(&stage_requests))
        };

        LookupOutcome::new(matched_len, blocks, staging_ticket, recompute_advised)
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
                let child_tokens = self.nodes[child_idx].tokens.clone();
                let remaining = &tokens[pos..];
                let match_len = child_tokens
                    .iter()
                    .zip(remaining.iter())
                    .take_while(|(a, b)| a == b)
                    .count();

                if match_len == child_tokens.len() {
                    // Full edge match — descend.
                    self.nodes[child_idx].last_access = now;
                    pos += match_len;

                    // Update block_id if child has the right size.
                    if child_tokens.len() == self.block_size && block_idx < blocks.len() {
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
                    let old_suffix = child_tokens[split_point..].to_vec();
                    let old_block = self.nodes[child_idx].block_id;
                    let old_ref_count = self.nodes[child_idx].ref_count;
                    let old_children: HashMap<u32, usize> =
                        std::mem::take(&mut self.nodes[child_idx].children);

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
                    let shared_tokens = child_tokens[..split_point].to_vec();
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
                    // shared_idx, covering tokens[pos+match_len .. pos+block_size].
                    // Path-from-root to this new node is match_len (shared) +
                    // (block_size - match_len) (new) = block_size, so block_id
                    // alignment holds. This turns the previous `break` — which
                    // returned 0 tokens inserted when the split hit in the first
                    // block, or N-1 blocks when it hit later — into a full insert.
                    // The new sibling's first token differs from `first_old` by
                    // construction (that's why match_len stopped short).
                    let rest_end = pos + self.block_size;
                    if block_idx < blocks.len() && rest_end <= tokens.len() {
                        let new_block_tokens = tokens[pos + match_len..rest_end].to_vec();
                        let new_bid = blocks[block_idx];
                        let new_fp = fps[block_idx];
                        let mut new_node = Node::new(new_block_tokens, Some(new_bid), now);
                        new_node.fingerprint = Some(new_fp);
                        let new_idx = self.alloc_node(new_node);
                        self.block_index.insert(new_bid, new_idx);
                        let first_new = self.nodes[new_idx].tokens[0];
                        self.nodes[shared_idx].children.insert(first_new, new_idx);
                        pos = rest_end;
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
                while pos < tokens.len() && block_idx < blocks.len() {
                    let end = (pos + self.block_size).min(tokens.len());
                    let edge_tokens = tokens[pos..end].to_vec();
                    let (block_id, fingerprint) = if edge_tokens.len() == self.block_size {
                        let bid = blocks[block_idx];
                        let fp = fps[block_idx];
                        block_idx += 1;
                        (Some(bid), Some(fp))
                    } else {
                        (None, None)
                    };

                    let mut new_node = Node::new(edge_tokens.clone(), block_id, now);
                    new_node.fingerprint = fingerprint;
                    let new_idx = self.alloc_node(new_node);
                    if let Some(bid) = block_id {
                        self.block_index.insert(bid, new_idx);
                    }

                    let first_tok = edge_tokens[0];
                    self.nodes[node_idx].children.insert(first_tok, new_idx);

                    pos = end;
                    node_idx = new_idx;

                    if edge_tokens.len() < self.block_size {
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
                let active_leaf = node
                    .children
                    .iter()
                    .all(|(_, child_idx)| evicted_set.contains(child_idx));
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

                let active_leaf = node
                    .children
                    .iter()
                    .all(|(_, child_idx)| evicted_set.contains(child_idx));
                if !active_leaf {
                    continue;
                }

                let slot = match node.tier_location {
                    Some(BlockLocation::Gpu { slot }) => slot,
                    _ => 0,
                };
                let soft_pinned = node.soft_pin_until.is_some_and(|deadline| deadline > now);
                let candidate = EvictionCandidate {
                    slot,
                    tokens: self.block_size as u32,
                    last_access_step: node.last_access,
                    hit_count: node.hit_count,
                    prefix_depth: node.tokens.len() as u32,
                    pinned: node.ref_count != 0 || soft_pinned,
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

    /// Stamp the physical location for a cached block.
    pub fn set_block_location(&mut self, block: BlockId, location: BlockLocation) -> bool {
        self.find_block_node_mut(block)
            .map(|node| {
                node.tier_location = Some(location);
            })
            .is_some()
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

    /// Update multiple metadata fields for one cached block with a single node
    /// lookup in the radix backing store.
    pub fn update_block_metadata(&mut self, block: BlockId, update: BlockMetadataUpdate) -> bool {
        self.find_block_node_mut(block)
            .map(|node| {
                if let Some(location) = update.location {
                    node.tier_location = Some(location);
                }
                if let Some(byte_len) = update.byte_len {
                    node.byte_len = byte_len;
                }
                if let Some(session_id) = update.session_id {
                    node.session_id = session_id;
                }
                if let Some(soft_pin_until) = update.soft_pin_until {
                    node.soft_pin_until = soft_pin_until;
                }
            })
            .is_some()
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Mutex};

    use super::*;
    use crate::kv_tier::{LookupHeuristics, StagePlanner, StageRequest, StageTicket};

    fn bids(ids: &[u32]) -> Vec<BlockId> {
        ids.iter().map(|&id| BlockId(id)).collect()
    }

    struct RecordingPlanner {
        requests: Mutex<Vec<StageRequest>>,
        ticket: StageTicket,
    }

    impl RecordingPlanner {
        fn new(ticket: u64) -> Self {
            Self {
                requests: Mutex::new(Vec::new()),
                ticket: StageTicket(ticket),
            }
        }
    }

    impl StagePlanner for RecordingPlanner {
        fn stage(&self, requests: &[StageRequest]) -> Option<StageTicket> {
            self.requests.lock().unwrap().extend_from_slice(requests);
            Some(self.ticket)
        }
    }

    #[test]
    fn empty_cache_returns_no_match() {
        let mut cache = RadixCache::new(4);
        let (len, blocks) = cache.lookup(&[1, 2, 3, 4]);
        assert_eq!(len, 0);
        assert!(blocks.is_empty());
    }

    #[test]
    fn insert_and_lookup_exact_one_block() {
        let mut cache = RadixCache::new(4);
        let tokens = vec![10, 20, 30, 40];
        cache.insert(&tokens, &bids(&[100]));

        let (len, blocks) = cache.lookup(&tokens);
        assert_eq!(len, 4);
        assert_eq!(blocks, bids(&[100]));
    }

    #[test]
    fn lookup_prefix_longer_than_cached() {
        let mut cache = RadixCache::new(4);
        let tokens = vec![1, 2, 3, 4];
        cache.insert(&tokens, &bids(&[10]));

        // Query with 8 tokens — only first 4 are cached.
        let query = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let (len, blocks) = cache.lookup(&query);
        assert_eq!(len, 4);
        assert_eq!(blocks, bids(&[10]));
    }

    #[test]
    fn two_requests_share_prefix() {
        let mut cache = RadixCache::new(4);
        let shared = vec![1, 2, 3, 4];
        cache.insert(&shared, &bids(&[10]));

        // Two requests diverge after the shared prefix.
        let req_a: Vec<u32> = shared.iter().copied().chain([5, 6, 7, 8]).collect();
        let req_b: Vec<u32> = shared.iter().copied().chain([9, 10, 11, 12]).collect();

        cache.insert(&req_a, &bids(&[10, 20]));
        cache.insert(&req_b, &bids(&[10, 30]));

        let (len_a, blocks_a) = cache.lookup(&req_a);
        let (len_b, blocks_b) = cache.lookup(&req_b);

        assert_eq!(len_a, 8);
        assert_eq!(blocks_a, bids(&[10, 20]));
        assert_eq!(len_b, 8);
        assert_eq!(blocks_b, bids(&[10, 30]));
    }

    #[test]
    fn lookup_mismatch_returns_zero() {
        let mut cache = RadixCache::new(4);
        cache.insert(&[1, 2, 3, 4], &bids(&[10]));

        let (len, blocks) = cache.lookup(&[5, 6, 7, 8]);
        assert_eq!(len, 0);
        assert!(blocks.is_empty());
    }

    #[test]
    fn evict_lru_leaf() {
        let mut cache = RadixCache::new(4);
        cache.insert(&[1, 2, 3, 4], &bids(&[10]));
        cache.insert(&[5, 6, 7, 8], &bids(&[20]));

        // Access the second sequence to make it more recently used.
        let _ = cache.lookup(&[5, 6, 7, 8]);
        let _ = cache.release(&bids(&[20])); // release so ref_count drops

        // After insert, ref_count is 0 by default.
        // First lookup incremented ref on block 10. Release it.
        cache.release(&bids(&[10]));
        cache.release(&bids(&[20]));

        let freed = cache.evict(1);
        // Should evict block 10 (older access) but not block 20.
        assert_eq!(freed.len(), 1);
        assert_eq!(freed[0], BlockId(10));
    }

    #[test]
    fn evict_respects_ref_count() {
        let mut cache = RadixCache::new(4);
        cache.insert(&[1, 2, 3, 4], &bids(&[10]));

        // Look up and hold a reference (don't release).
        let (_, _blocks) = cache.lookup(&[1, 2, 3, 4]);
        // ref_count is now > 0.

        let freed = cache.evict(1);
        // Should NOT evict because ref_count > 0.
        assert!(freed.is_empty());
    }

    #[test]
    fn multiple_blocks_insert_lookup() {
        let mut cache = RadixCache::new(2);
        // 6 tokens = 3 blocks of size 2
        let tokens: Vec<u32> = (1..=6).collect();
        cache.insert(&tokens, &bids(&[10, 20, 30]));

        let (len, blocks) = cache.lookup(&tokens);
        assert_eq!(len, 6);
        assert_eq!(blocks, bids(&[10, 20, 30]));
    }

    #[test]
    fn partial_block_not_cached() {
        let mut cache = RadixCache::new(4);
        // Only 3 tokens — less than one block.
        let tokens = vec![1, 2, 3];
        let inserted = cache.insert(&tokens, &bids(&[10])); // blocks slice is checked
        // Actually inserting with an empty blocks slice (no complete blocks).
        let inserted2 = cache.insert(&tokens, &[]);
        assert_eq!(inserted2, 0);
        _ = inserted; // silence unused warning

        let (len, _) = cache.lookup(&tokens);
        assert_eq!(len, 0); // nothing was cached
    }

    #[test]
    fn insert_idempotent() {
        let mut cache = RadixCache::new(4);
        let tokens = vec![1, 2, 3, 4];
        cache.insert(&tokens, &bids(&[10]));
        cache.insert(&tokens, &bids(&[10]));

        let (len, blocks) = cache.lookup(&tokens);
        assert_eq!(len, 4);
        assert_eq!(blocks, bids(&[10]));
    }

    #[test]
    fn block_index_tracks_inserts_and_evictions() {
        let mut cache = RadixCache::new(4);
        cache.insert(&[1, 2, 3, 4], &bids(&[10]));
        cache.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &bids(&[10, 20]));
        cache.insert(&[9, 10, 11, 12], &bids(&[30]));

        // O(1) lookup via set_block_location: returns true when the index
        // knows about the block and the underlying node is reachable.
        assert!(cache.set_block_location(BlockId(10), BlockLocation::Gpu { slot: 0 }));
        assert!(cache.set_block_location(BlockId(20), BlockLocation::Gpu { slot: 0 }));
        assert!(cache.set_block_location(BlockId(30), BlockLocation::Gpu { slot: 0 }));

        let freed = cache.evict(1);
        assert_eq!(freed.len(), 1);
        let evicted_bid = freed[0];
        assert!(!cache.set_block_location(evicted_bid, BlockLocation::Gpu { slot: 0 }));

        // Re-insert a fresh block; index must now find it O(1).
        cache.insert(&[100, 101, 102, 103], &bids(&[40]));
        assert!(cache.set_block_location(BlockId(40), BlockLocation::Gpu { slot: 1 }));
    }

    #[test]
    fn rebuild_block_index_round_trips_after_serde() {
        let mut cache = RadixCache::new(4);
        cache.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &bids(&[10, 20]));

        let snapshot = serde_json::to_string(&cache).expect("serialize");
        let mut restored: RadixCache = serde_json::from_str(&snapshot).expect("deserialize");

        // Right after deserialize, the index is empty (skip+default).
        assert!(!restored.set_block_location(BlockId(10), BlockLocation::Gpu { slot: 0 }));

        restored.rebuild_block_index();
        assert!(restored.set_block_location(BlockId(10), BlockLocation::Gpu { slot: 0 }));
        assert!(restored.set_block_location(BlockId(20), BlockLocation::Gpu { slot: 0 }));
    }

    #[test]
    fn insert_stamps_fingerprint_on_cached_nodes() {
        let mut cache = RadixCache::new(4);
        cache.insert_with_fingerprints(
            &[1, 2, 3, 4],
            &[BlockId(10)],
            &[BlockFingerprint([0x11; 16])],
        );

        let idx = cache
            .nodes
            .iter()
            .position(|n| n.block_id == Some(BlockId(10)))
            .unwrap();
        assert_eq!(
            cache.nodes[idx].fingerprint,
            Some(BlockFingerprint([0x11; 16]))
        );
    }

    #[test]
    fn cached_block_count() {
        let mut cache = RadixCache::new(4);
        assert_eq!(cache.cached_block_count(), 0);

        cache.insert(&[1, 2, 3, 4], &bids(&[10]));
        assert_eq!(cache.cached_block_count(), 1);

        cache.insert(&[5, 6, 7, 8], &bids(&[20]));
        assert_eq!(cache.cached_block_count(), 2);
    }

    // ──────────────────────────────────────────────────────────────────
    // Regression tests for SGLang-parity fixes (2026-04-13):
    //   - Bug 1: split must inherit ref_count from the old child
    //   - Bug 3: evict() must cascade through orphaned parents in one call
    //   - Sanity: lookup() already path-bumps every block-bearing node on
    //     the matched path (this is NOT the B1-research "Bug 2" — the
    //     current code already handles this correctly; document it so a
    //     future refactor does not regress).
    // ──────────────────────────────────────────────────────────────────

    /// Build a 3-deep block chain `[100, 200, 300]` covering tokens
    /// `[1..=12]` at `block_size=4`. Each node holds one block.
    fn chain_tree() -> RadixCache {
        let mut cache = RadixCache::new(4);
        cache.insert(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            &bids(&[100, 200, 300]),
        );
        cache
    }

    #[test]
    fn lookup_bumps_every_block_bearing_node_on_path() {
        // Sanity test documenting that `lookup()` bumps ref_count on ALL
        // block-bearing nodes along the matched path (not only the leaf).
        // This was the behavior before the 2026-04-13 fix series; keep a
        // test so it stays that way.
        let mut cache = chain_tree();
        let (_, blocks) = cache.lookup(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        assert_eq!(blocks, bids(&[100, 200, 300]));

        // Find each block-bearing node and check ref_count == 1.
        let ref_counts: Vec<u32> = cache
            .nodes
            .iter()
            .filter(|n| n.block_id.is_some())
            .map(|n| n.ref_count)
            .collect();
        assert_eq!(ref_counts.len(), 3, "expected 3 block-bearing nodes");
        for (i, rc) in ref_counts.iter().enumerate() {
            assert_eq!(*rc, 1, "node {i} ref_count should be 1 after lookup");
        }
    }

    #[test]
    fn split_node_inherits_ref_count_from_child() {
        // Build a single-prefix tree, lock it with a lookup, then trigger
        // a split by inserting a diverging suffix. The shared intermediate
        // node (block_id=None) must inherit the old child's ref_count so a
        // future GC pass cannot prune it out from under the in-flight lock.
        let mut cache = RadixCache::new(4);
        cache.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &bids(&[10, 20]));

        // Lock the deep child by looking it up.
        let (_, blocks) = cache.lookup(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(blocks, bids(&[10, 20]));

        // After lookup: the [5,6,7,8] leaf has ref_count == 1.
        // Insert a diverging suffix that forces a split at token 4.
        // Before the split the existing [1..=8] edge is already a chain of
        // two full-block nodes: [1,2,3,4] (block 10) and [5,6,7,8] (block
        // 20). So inserting [1,2,3,4,5,6,9,10] matches the first block
        // fully and splits the SECOND node at its 3rd token (5,6 shared,
        // then diverges 7,8 vs 9,10).
        cache.insert(&[1, 2, 3, 4, 5, 6, 9, 10], &bids(&[10, 99]));

        // Walk the tree from the root to the shared intermediate node and
        // verify it inherited `ref_count == 1`. Structurally: root → [1..4]
        // (block 10, ref 1) → shared [5,6] (no block, ref 1 via inherit)
        // → [7,8] (block 20, ref 1) and → [9,10] (block 99, ref 0).
        let root_child = *cache.nodes[0].children.values().next().unwrap();
        let first_block_node = &cache.nodes[root_child];
        assert_eq!(first_block_node.block_id, Some(BlockId(10)));
        assert_eq!(first_block_node.ref_count, 1);

        // Descend to the shared split node.
        let shared_idx = *first_block_node.children.values().next().unwrap();
        let shared_node = &cache.nodes[shared_idx];
        assert!(
            shared_node.block_id.is_none(),
            "shared split node should not carry a block_id"
        );
        assert_eq!(
            shared_node.ref_count, 1,
            "shared split node must inherit ref_count from the locked child"
        );
    }

    // Before 2026-04-19, insert_with_fingerprints bailed with `break` on any
    // partial-edge-split, returning 0 tokens when the split hit in block 1 and
    // N-1 blocks worth when it hit later. The caller (scheduler/cuda/core.rs)
    // then logged `prefix_cache.insert: expected X, got Y` and refused to pin
    // any pages. The fix creates a new sibling under the shared intermediate
    // covering the caller's current block, preserving block-id alignment.

    #[test]
    fn split_on_first_block_inserts_remaining_blocks_as_sibling() {
        // "got 0" regression: before the fix, this insert returned 0.
        // block_size=4, existing tree has root → [1,2,3,4] (block 10).
        // New request shares only the first token [1], then diverges.
        let mut cache = RadixCache::new(4);
        cache.insert(&[1, 2, 3, 4], &bids(&[10]));

        let inserted = cache.insert(&[1, 5, 6, 7, 8, 9, 10, 11], &bids(&[100, 200]));
        assert_eq!(
            inserted, 8,
            "after split on first token, the full 2 blocks should be registered \
             (would return 0 before the fix)",
        );

        // Both requests remain reachable.
        let (len_a, blocks_a) = cache.lookup(&[1, 2, 3, 4]);
        assert_eq!(len_a, 4);
        assert_eq!(blocks_a, bids(&[10]));

        let (len_b, blocks_b) = cache.lookup(&[1, 5, 6, 7, 8, 9, 10, 11]);
        assert_eq!(len_b, 8);
        assert_eq!(blocks_b, bids(&[100, 200]));
    }

    #[test]
    fn split_on_later_block_inserts_divergent_block_under_shared() {
        // "got 4080" regression at scale 2-blocks: split fires on block 2, the
        // prior code returned 1 * block_size (4) instead of 8.
        // block_size=4, existing tree has root → [1,2,3,4] (block 10) →
        // [5,6,7,8] (block 20). New request shares block 1 fully, then
        // diverges mid-block-2.
        let mut cache = RadixCache::new(4);
        cache.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &bids(&[10, 20]));

        let inserted = cache.insert(&[1, 2, 3, 4, 5, 6, 9, 10], &bids(&[10, 99]));
        assert_eq!(
            inserted, 8,
            "after split on block 2, both blocks should be registered \
             (would return 4 before the fix)",
        );

        // Original request still reachable.
        let (len_orig, blocks_orig) = cache.lookup(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(len_orig, 8);
        assert_eq!(blocks_orig, bids(&[10, 20]));

        // New divergent request also reachable, with its new block_id=99.
        let (len_new, blocks_new) = cache.lookup(&[1, 2, 3, 4, 5, 6, 9, 10]);
        assert_eq!(len_new, 8);
        assert_eq!(blocks_new, bids(&[10, 99]));
    }

    #[test]
    fn split_with_short_tail_still_registers_aligned_block() {
        // Edge case: caller's tokens end exactly at block_size * n, with the
        // split on block n. The new sibling must still be created.
        let mut cache = RadixCache::new(4);
        cache.insert(&[1, 2, 3, 4], &bids(&[10]));

        // Shares only token [1], diverges, has exactly one block of tokens.
        let inserted = cache.insert(&[1, 5, 6, 7], &bids(&[100]));
        assert_eq!(inserted, 4);

        let (len, blocks) = cache.lookup(&[1, 5, 6, 7]);
        assert_eq!(len, 4);
        assert_eq!(blocks, bids(&[100]));
    }

    #[test]
    fn evict_cascades_through_orphaned_parent_chain() {
        // Insert a 3-deep chain with no branches. Evict all three blocks in
        // ONE call — the prior implementation would stop after evicting the
        // deepest leaf because the parents were not re-examined.
        let mut cache = chain_tree();
        assert_eq!(cache.cached_block_count(), 3);

        let freed = cache.evict(3);
        assert_eq!(
            freed.len(),
            3,
            "evict(3) on a 3-deep chain must cascade through parents in one call"
        );
        assert_eq!(cache.cached_block_count(), 0);

        // The freed blocks arrive in LRU order: 300 (deepest, oldest by tie)
        // then 200 then 100. The exact ordering depends on `last_access`
        // bookkeeping — just make sure the set matches.
        let mut freed_sorted: Vec<u32> = freed.iter().map(|b| b.0).collect();
        freed_sorted.sort_unstable();
        assert_eq!(freed_sorted, vec![100, 200, 300]);
    }

    #[test]
    fn evict_cascade_respects_limit_n() {
        // Same chain, but only evict 2 — the oldest leaf and the
        // newly-orphaned middle node.
        let mut cache = chain_tree();
        let freed = cache.evict(2);
        assert_eq!(freed.len(), 2);
        assert_eq!(cache.cached_block_count(), 1, "one block should remain");
    }

    #[test]
    fn evict_cascade_respects_ref_count() {
        // Chain with a locked middle node: evict should stop when it
        // reaches a node with ref_count > 0.
        let mut cache = chain_tree();

        // Lookup the middle prefix to bump [1..4] and [5..8]. The deepest
        // node [9..12] is NOT in the lookup range → ref_count = 0.
        let _ = cache.lookup(&[1, 2, 3, 4, 5, 6, 7, 8]);

        let freed = cache.evict(3);
        // Only the deepest leaf (block 300) can be evicted; the middle
        // (200) and top (100) are pinned by the outstanding lookup.
        assert_eq!(
            freed,
            bids(&[300]),
            "only the unlocked deepest leaf should be evicted"
        );
        assert_eq!(cache.cached_block_count(), 2);
    }

    #[test]
    fn evict_with_policy_matches_lru_when_signals_are_cold() {
        let mut cache = RadixCache::new(4);
        cache.insert(&[1, 2, 3, 4], &bids(&[100]));
        cache.insert(&[5, 6, 7, 8], &bids(&[200]));

        let _ = cache.lookup(&[5, 6, 7, 8]);
        cache.release(&bids(&[200]));

        let freed = cache.evict_with_policy(
            &crate::scheduler::policy::LruEviction,
            SchedulerSignals::default(),
            1,
        );
        assert_eq!(freed, bids(&[100]));
    }

    #[test]
    fn evict_with_policy_respects_session_affinity_slot() {
        let mut cache = RadixCache::new(4);
        cache.insert(&[1, 2, 3, 4], &bids(&[100]));
        cache.insert(&[5, 6, 7, 8], &bids(&[200]));

        let idx_a = cache
            .nodes
            .iter()
            .position(|node| node.block_id == Some(BlockId(100)))
            .unwrap();
        let idx_b = cache
            .nodes
            .iter()
            .position(|node| node.block_id == Some(BlockId(200)))
            .unwrap();
        cache.nodes[idx_a].tier_location = Some(BlockLocation::Gpu { slot: 1 });
        cache.nodes[idx_b].tier_location = Some(BlockLocation::Gpu { slot: 7 });
        cache.nodes[idx_a].last_access = 10;
        cache.nodes[idx_b].last_access = 1;

        let freed = cache.evict_with_policy(
            &crate::scheduler::policy::SessionBiasedLru::default(),
            SchedulerSignals {
                session_affinity_slot: Some(7),
                ..SchedulerSignals::default()
            },
            1,
        );
        assert_eq!(freed, bids(&[100]));
    }

    #[test]
    fn evict_with_policy_skips_soft_pinned_nodes() {
        let mut cache = RadixCache::new(4);
        cache.insert(&[1, 2, 3, 4], &bids(&[100]));
        cache.insert(&[5, 6, 7, 8], &bids(&[200]));

        let idx_a = cache
            .nodes
            .iter()
            .position(|node| node.block_id == Some(BlockId(100)))
            .unwrap();
        let idx_b = cache
            .nodes
            .iter()
            .position(|node| node.block_id == Some(BlockId(200)))
            .unwrap();
        cache.clock = 100;
        cache.nodes[idx_a].soft_pin_until = Some(200);
        cache.nodes[idx_a].last_access = 0;
        cache.nodes[idx_b].last_access = 1;

        let freed = cache.evict_with_policy(
            &crate::scheduler::policy::LruEviction,
            SchedulerSignals::default(),
            1,
        );
        assert_eq!(freed, bids(&[200]));
    }

    // ──────────────────────────────────────────────────────────────────
    // Serde round-trip — Tiered KV Cache §P3 session persistence gate.
    // The on-disk format is a straightforward serde representation; this
    // test pins the expectation that a full insert → lookup state can be
    // JSON-roundtripped without loss and that lookups on the restored
    // cache still find every previously-cached block.
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn radix_cache_serde_roundtrip_preserves_lookups() {
        let mut cache = RadixCache::with_soft_pin_keepalive(4, 5);

        // Populate with three sequences sharing a common prefix so the
        // tree has a non-trivial shape with a split.
        cache.insert(&[1, 2, 3, 4], &bids(&[10]));
        cache.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &bids(&[10, 20]));
        cache.insert(&[1, 2, 3, 4, 9, 10, 11, 12], &bids(&[10, 30]));
        let _ = cache.lookup(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(cache.set_block_soft_pin_until(BlockId(20), Some(123)));

        let before_node_count = cache.node_count();
        let before_cached = cache.cached_block_count();

        let json = serde_json::to_string(&cache).expect("serialize RadixCache");
        // Exercise pretty-print too — the disk format is allowed to use
        // either shape.
        let pretty = serde_json::to_string_pretty(&cache).expect("serialize RadixCache (pretty)");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&json).unwrap(),
            serde_json::from_str::<serde_json::Value>(&pretty).unwrap(),
            "compact + pretty JSON must decode to the same value tree",
        );

        let mut restored: RadixCache = serde_json::from_str(&json).expect("deserialize RadixCache");

        assert_eq!(restored.node_count(), before_node_count);
        assert_eq!(restored.cached_block_count(), before_cached);
        assert_eq!(restored.logical_clock(), 0);
        assert!(restored.nodes.iter().all(|node| node.ref_count == 0));
        assert!(restored.nodes.iter().all(|node| node.last_access == 0));
        assert!(
            restored
                .nodes
                .iter()
                .all(|node| node.soft_pin_until.is_none())
        );

        // Every prefix the original cache had should still resolve after
        // the round trip. Note: `lookup` bumps ref_count — that is
        // expected, the restored cache is a clean slate.
        let (len_a, blocks_a) = restored.lookup(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let (len_b, blocks_b) = restored.lookup(&[1, 2, 3, 4, 9, 10, 11, 12]);
        assert_eq!(len_a, 8);
        assert_eq!(blocks_a, bids(&[10, 20]));
        assert_eq!(len_b, 8);
        assert_eq!(blocks_b, bids(&[10, 30]));
    }

    #[test]
    fn reconcile_remaps_known_fingerprints_into_new_pool() {
        let mut cache = RadixCache::new(4);
        let fp_a = BlockFingerprint([0x11; 16]);
        let fp_b = BlockFingerprint([0x22; 16]);
        cache.insert_with_fingerprints(
            &[1, 2, 3, 4, 5, 6, 7, 8],
            &[BlockId(10), BlockId(20)],
            &[fp_a, fp_b],
        );

        let json = serde_json::to_string(&cache).expect("serialize RadixCache");
        let mut restored: RadixCache = serde_json::from_str(&json).expect("deserialize RadixCache");
        let known = HashMap::from([(fp_a, BlockId(100)), (fp_b, BlockId(200))]);

        let report = restored.reconcile(&known);

        assert_eq!(report.remapped, 2);
        assert_eq!(report.tombstoned, 0);
        assert!(restored.find_block_node_mut(BlockId(100)).is_some());
        assert!(restored.find_block_node_mut(BlockId(200)).is_some());
        assert!(restored.find_block_node_mut(BlockId(10)).is_none());
    }

    #[test]
    fn reconcile_tombstones_fingerprints_missing_in_new_pool() {
        let mut cache = RadixCache::new(4);
        let fp_a = BlockFingerprint([0x11; 16]);
        let fp_b = BlockFingerprint([0x22; 16]);
        cache.insert_with_fingerprints(
            &[1, 2, 3, 4, 5, 6, 7, 8],
            &[BlockId(10), BlockId(20)],
            &[fp_a, fp_b],
        );

        let json = serde_json::to_string(&cache).expect("serialize RadixCache");
        let mut restored: RadixCache = serde_json::from_str(&json).expect("deserialize RadixCache");

        let report = restored.reconcile(&HashMap::from([(fp_a, BlockId(100))]));

        assert_eq!(report.remapped, 1);
        assert_eq!(report.tombstoned, 1);
        let tombstoned = restored
            .nodes
            .iter()
            .find(|node| node.fingerprint == Some(fp_b))
            .expect("fingerprinted node should still exist structurally");
        assert_eq!(tombstoned.block_id, None);
    }

    #[test]
    fn reconcile_clears_orphan_nodes_with_no_fingerprint() {
        let mut cache = RadixCache::new(4);
        let fingerprint = BlockFingerprint([0x11; 16]);
        cache.insert_with_fingerprints(&[1, 2, 3, 4], &[BlockId(10)], &[fingerprint]);

        let orphan_idx = cache
            .nodes
            .iter()
            .position(|node| node.block_id == Some(BlockId(10)))
            .expect("block node should exist");
        cache.nodes[orphan_idx].fingerprint = None;

        let report = cache.reconcile(&HashMap::new());

        assert!(report.orphans_cleared >= 1);
        assert_eq!(cache.nodes[orphan_idx].block_id, None);
    }

    #[test]
    fn serde_round_trip_then_reconcile_end_to_end() {
        let mut cache = RadixCache::with_soft_pin_keepalive(4, 7);
        let fp_a = BlockFingerprint([0x11; 16]);
        let fp_b = BlockFingerprint([0x22; 16]);
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        cache.insert_with_fingerprints(&tokens, &[BlockId(10), BlockId(20)], &[fp_a, fp_b]);
        assert!(cache.set_block_soft_pin_until(BlockId(20), Some(88)));

        let expected = cache.lookup_or_stage(&tokens, LookupHeuristics::default(), None);
        let expected_hit_kinds: Vec<HitKind> =
            expected.blocks.iter().map(|block| block.hit_kind).collect();

        let json = serde_json::to_string(&cache).expect("serialize RadixCache");
        drop(cache);

        let mut restored: RadixCache = serde_json::from_str(&json).expect("deserialize RadixCache");
        assert_eq!(restored.logical_clock(), 0);
        assert!(restored.nodes.iter().all(|node| node.ref_count == 0));
        assert!(restored.nodes.iter().all(|node| node.last_access == 0));
        assert!(
            restored
                .nodes
                .iter()
                .all(|node| node.soft_pin_until.is_none())
        );

        let report =
            restored.reconcile(&HashMap::from([(fp_a, BlockId(100)), (fp_b, BlockId(200))]));
        assert_eq!(report.remapped, 2);
        assert_eq!(report.tombstoned, 0);

        let outcome = restored.lookup_or_stage(&tokens, LookupHeuristics::default(), None);
        let block_ids: Vec<Option<BlockId>> =
            outcome.blocks.iter().map(|block| block.block_id).collect();
        let hit_kinds: Vec<HitKind> = outcome.blocks.iter().map(|block| block.hit_kind).collect();

        assert_eq!(outcome.matched_len, expected.matched_len);
        assert_eq!(hit_kinds, expected_hit_kinds);
        assert_eq!(block_ids, vec![Some(BlockId(100)), Some(BlockId(200))]);
        assert_eq!(outcome.staging_ticket, expected.staging_ticket);
        assert_eq!(outcome.recompute_advised, expected.recompute_advised);
    }

    #[test]
    fn block_id_serde_roundtrip() {
        // BlockId is serialized as a newtype around u32 — keep this
        // explicit so the wire format is stable.
        let id = BlockId(42);
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "42", "BlockId(N) must serialize as the bare N");
        let back: BlockId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, id);
    }

    // ------------------------------------------------------------------
    // Scheduler integration pattern — validates the exact usage
    // shape `scheduler::cuda::core::Scheduler::publish_to_prefix_cache`
    // + the admission lookup path rely on. Kept here (rather
    // than in `scheduler/cuda/`) because the scheduler itself needs a
    // full `Model` + CUDA context to construct and so cannot host a
    // pure-Rust test.
    // ------------------------------------------------------------------

    #[test]
    fn scheduler_shadow_observer_roundtrip_releases_all_refs() {
        // Mirrors the M1 scheduler wiring: fresh cache, block_size=16
        // (matches the `PREFIX_CACHE_BLOCK_SIZE` constant on the
        // scheduler), synthetic block ids minted from a monotonic
        // counter, insert on request-complete, lookup + release on
        // next admission.
        const BLOCK_SIZE: usize = 16;
        let mut cache = RadixCache::new(BLOCK_SIZE);
        let mut next_id: u32 = 0;

        // Prompt A: 32 tokens = 2 blocks. Scheduler inserts on cleanup.
        let prompt_a: Vec<u32> = (100..132).collect();
        let blocks_a = {
            let n = prompt_a.len() / BLOCK_SIZE;
            let v: Vec<BlockId> = (0..n)
                .map(|_| {
                    let id = BlockId(next_id);
                    next_id += 1;
                    id
                })
                .collect();
            assert_eq!(v.len(), 2, "32 tokens → exactly 2 blocks");
            v
        };
        let inserted = cache.insert(&prompt_a, &blocks_a);
        assert_eq!(inserted, 32);
        assert_eq!(next_id, 2);
        assert_eq!(cache.cached_block_count(), 2);

        // New request arrives with the same prompt_a — scheduler runs
        // `radix.lookup` on admission. The refs it bumps must be
        // released before the assigner proceeds so the radix does not
        // pin evictable blocks forever.
        let (hit_len, hit_blocks) = cache.lookup(&prompt_a);
        assert_eq!(hit_len, 32, "full prompt should hit both blocks");
        assert_eq!(hit_blocks, blocks_a);
        cache.release(&hit_blocks);

        // After release, the matched nodes must evict cleanly (nothing
        // is holding them). This is the invariant that keeps the
        // admission path safe now that the radix is load-bearing for
        // slot selection.
        let freed = cache.evict(cache.cached_block_count());
        assert_eq!(freed.len(), 2, "both blocks must be evictable");
        assert_eq!(cache.cached_block_count(), 0);

        // Fresh insert with a new id stream must succeed on the same
        // cache (no residual state). This matches the "many requests
        // arrive over the lifetime of a scheduler" pattern.
        let prompt_b: Vec<u32> = (200..232).collect();
        let blocks_b: Vec<BlockId> = (0..2)
            .map(|_| {
                let id = BlockId(next_id);
                next_id += 1;
                id
            })
            .collect();
        assert_eq!(cache.insert(&prompt_b, &blocks_b), 32);
        let (hit_len_b, hit_blocks_b) = cache.lookup(&prompt_b);
        assert_eq!(hit_len_b, 32);
        assert_eq!(hit_blocks_b, blocks_b);
        cache.release(&hit_blocks_b);
    }

    #[test]
    fn scheduler_shadow_observer_partial_block_is_dropped() {
        // Mirrors publish_to_prefix_cache's early-return path: a
        // prompt shorter than one block (or whose tail is <
        // block_size) does not produce any cached entry.
        const BLOCK_SIZE: usize = 16;
        let mut cache = RadixCache::new(BLOCK_SIZE);

        // 10-token prompt — zero full blocks.
        let short: Vec<u32> = (0..10).collect();
        let num_blocks = short.len() / BLOCK_SIZE;
        assert_eq!(num_blocks, 0);
        // publish_to_prefix_cache short-circuits before calling insert
        // in this case, but even if it reached insert() with an empty
        // blocks slice the semantics are the same: nothing inserted.
        let inserted = cache.insert(&short, &[]);
        assert_eq!(inserted, 0);
        assert_eq!(cache.cached_block_count(), 0);

        // 17-token prompt — exactly 1 full block, 1-token tail dropped.
        let mixed: Vec<u32> = (0..17).collect();
        let blocks = bids(&[999]);
        assert_eq!(cache.insert(&mixed, &blocks), 16);
        let (hit_len, hit_blocks) = cache.lookup(&mixed);
        assert_eq!(hit_len, 16, "tail token must not round up");
        assert_eq!(hit_blocks, blocks);
        cache.release(&hit_blocks);
    }

    #[test]
    fn evict_prunes_orphan_tombstones_into_free_list() {
        let mut cache = RadixCache::new(4);

        cache.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &bids(&[10, 20]));
        // Split at block 2: after the fix (2026-04-19), both divergent
        // branches are registered under a shared [5,6] tombstone:
        //   root → [1..4](10) → shared[5,6] → {[7,8](20), [9,10](99)}
        cache.insert(&[1, 2, 3, 4, 5, 6, 9, 10], &bids(&[10, 99]));

        assert_eq!(cache.cached_block_count(), 3);
        let before_nodes = cache.node_count();

        // Evict both divergent leaves so shared becomes a blockless orphan
        // tombstone and cascades into the free list.
        let freed = cache.evict(2);
        let mut freed_ids: Vec<u32> = freed.iter().map(|b| b.0).collect();
        freed_ids.sort_unstable();
        assert_eq!(freed_ids, vec![20, 99]);
        assert_eq!(cache.cached_block_count(), 1);
        assert_eq!(
            cache.free_node_count(),
            3,
            "two leaves and shared tombstone should be reclaimed"
        );
        assert_eq!(
            cache.node_count(),
            before_nodes,
            "reclaiming tombstones should not grow the backing node array"
        );
    }

    #[test]
    fn insert_reuses_reclaimed_tombstone_slots() {
        let mut cache = RadixCache::new(4);

        cache.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &bids(&[10, 20]));
        cache.insert(&[1, 2, 3, 4, 5, 6, 9, 10], &bids(&[10, 99]));
        // Evict both divergent leaves → shared cascades → 3 free slots.
        let _ = cache.evict(2);

        let before_nodes = cache.node_count();
        assert_eq!(cache.free_node_count(), 3);

        cache.insert(&[1, 2, 3, 4, 11, 12, 13, 14], &bids(&[10, 77]));

        assert_eq!(
            cache.node_count(),
            before_nodes,
            "insert should reuse a reclaimed slot instead of appending"
        );
        assert_eq!(
            cache.free_node_count(),
            2,
            "one reclaimed slot should be removed from the free list"
        );
    }

    #[test]
    fn lookup_or_stage_defaults_to_ready_on_gpu() {
        let mut cache = RadixCache::new(4);
        cache.insert(&[1, 2, 3, 4], &bids(&[10]));

        let outcome = cache.lookup_or_stage(&[1, 2, 3, 4], LookupHeuristics::default(), None);

        assert_eq!(outcome.matched_len, 4);
        assert_eq!(
            outcome.blocks,
            vec![LookupBlock {
                block_id: Some(BlockId(10)),
                hit_kind: HitKind::ReadyOnGpu,
            }]
        );
        assert_eq!(outcome.staging_ticket, None);
        assert!(!outcome.recompute_advised);
    }

    #[test]
    fn lookup_or_stage_queues_host_stage() {
        let mut cache = RadixCache::new(4);
        cache.insert(&[1, 2, 3, 4], &bids(&[10]));
        assert!(cache.set_block_location(BlockId(10), BlockLocation::HostPinned { offset: 4096 }));
        assert!(cache.set_block_byte_len(BlockId(10), 8192));

        let planner = RecordingPlanner::new(41);
        let outcome =
            cache.lookup_or_stage(&[1, 2, 3, 4], LookupHeuristics::default(), Some(&planner));

        assert_eq!(outcome.matched_len, 4);
        assert_eq!(outcome.staging_ticket, Some(StageTicket(41)));
        assert_eq!(
            outcome.blocks,
            vec![LookupBlock {
                block_id: Some(BlockId(10)),
                hit_kind: HitKind::StagingFromHost,
            }]
        );
        assert!(!outcome.recompute_advised);

        let recorded = planner.requests.lock().unwrap().clone();
        assert_eq!(
            recorded,
            vec![StageRequest {
                block_id: BlockId(10),
                from: BlockLocation::HostPinned { offset: 4096 },
                byte_len: 8192,
            }]
        );
    }

    #[test]
    fn lookup_or_stage_advises_recompute_for_small_disk_hit() {
        let mut cache = RadixCache::new(16);
        let tokens: Vec<u32> = (0..16).collect();
        cache.insert(&tokens, &bids(&[10]));
        assert!(cache.set_block_location(
            BlockId(10),
            BlockLocation::Disk {
                file_id: 1,
                offset: 0,
            }
        ));
        assert!(cache.set_block_byte_len(BlockId(10), 2 * 1024 * 1024));

        let planner = RecordingPlanner::new(7);
        let outcome = cache.lookup_or_stage(
            &tokens,
            LookupHeuristics {
                prefill_tokens_per_sec: 200_000.0,
                host_bandwidth_bytes_per_sec: 25.0 * 1024.0 * 1024.0 * 1024.0,
                disk_bandwidth_bytes_per_sec: 50.0 * 1024.0 * 1024.0,
            },
            Some(&planner),
        );

        assert_eq!(outcome.matched_len, 16);
        assert_eq!(outcome.staging_ticket, None);
        assert!(outcome.recompute_advised);
        assert_eq!(
            outcome.blocks,
            vec![LookupBlock {
                block_id: Some(BlockId(10)),
                hit_kind: HitKind::StagingFromDisk,
            }]
        );
    }

    #[test]
    fn lookup_or_stage_surfaces_tombstone_miss() {
        let mut cache = RadixCache::new(4);
        cache.insert(&[1, 2, 3, 4], &bids(&[10]));

        let idx = cache
            .nodes
            .iter()
            .position(|node| node.block_id == Some(BlockId(10)))
            .unwrap();
        cache.nodes[idx].block_id = None;
        cache.nodes[idx].tier_location = Some(BlockLocation::Disk {
            file_id: 9,
            offset: 0,
        });
        cache.nodes[idx].byte_len = 4096;

        let outcome = cache.lookup_or_stage(&[1, 2, 3, 4], LookupHeuristics::default(), None);

        assert_eq!(outcome.matched_len, 0);
        assert_eq!(outcome.staging_ticket, None);
        assert!(!outcome.recompute_advised);
        assert_eq!(
            outcome.blocks,
            vec![LookupBlock {
                block_id: None,
                hit_kind: HitKind::Miss,
            }]
        );
    }

    #[test]
    fn lookup_or_stage_ignores_trailing_tombstone_for_gpu_ready_prefix() {
        let mut cache = RadixCache::new(4);
        let tokens: Vec<u32> = (1..=12).collect();
        cache.insert(&tokens, &bids(&[10, 20, 30]));

        let idx = cache
            .nodes
            .iter()
            .position(|node| node.block_id == Some(BlockId(30)))
            .unwrap();
        cache.nodes[idx].block_id = None;
        cache.nodes[idx].tier_location = Some(BlockLocation::Disk {
            file_id: 9,
            offset: 0,
        });
        cache.nodes[idx].byte_len = 4096;

        let outcome = cache.lookup_or_stage(&tokens, LookupHeuristics::default(), None);
        let ready_on_gpu = outcome
            .blocks
            .iter()
            .filter(|block| !matches!(block.hit_kind, HitKind::Miss))
            .all(|block| matches!(block.hit_kind, HitKind::ReadyOnGpu));

        assert_eq!(outcome.matched_len, 8);
        assert_eq!(
            outcome.blocks,
            vec![
                LookupBlock {
                    block_id: Some(BlockId(10)),
                    hit_kind: HitKind::ReadyOnGpu,
                },
                LookupBlock {
                    block_id: Some(BlockId(20)),
                    hit_kind: HitKind::ReadyOnGpu,
                },
                LookupBlock {
                    block_id: None,
                    hit_kind: HitKind::Miss,
                },
            ]
        );
        assert!(
            ready_on_gpu,
            "scheduler admission should ignore trailing tombstones when checking T0 readiness"
        );
    }

    #[test]
    fn metadata_mutators_stamp_session_and_soft_pin() {
        let mut cache = RadixCache::new(4);
        cache.insert(&[1, 2, 3, 4], &bids(&[10]));

        assert!(cache.set_block_session_id(BlockId(10), Some(SessionId::from("session-1"))));
        assert!(cache.set_block_soft_pin_until(BlockId(10), Some(42)));

        let idx = cache
            .nodes
            .iter()
            .position(|node| node.block_id == Some(BlockId(10)))
            .unwrap();
        assert_eq!(
            cache.nodes[idx].session_id,
            Some(SessionId::from("session-1"))
        );
        assert_eq!(cache.nodes[idx].soft_pin_until, Some(42));

        assert!(cache.set_block_session_id(BlockId(10), None));
        assert!(cache.set_block_soft_pin_until(BlockId(10), None));
        assert_eq!(cache.nodes[idx].session_id, None);
        assert_eq!(cache.nodes[idx].soft_pin_until, None);
    }

    #[test]
    fn update_block_metadata_coalesces_session_and_soft_pin_updates() {
        let mut cache = RadixCache::new(4);
        cache.insert(&[1, 2, 3, 4], &bids(&[10]));

        assert!(cache.update_block_metadata(
            BlockId(10),
            BlockMetadataUpdate {
                location: Some(BlockLocation::Gpu { slot: 7 }),
                byte_len: Some(4096),
                session_id: Some(Some(SessionId::from("session-1"))),
                soft_pin_until: Some(Some(42)),
            }
        ));

        let idx = cache
            .nodes
            .iter()
            .position(|node| node.block_id == Some(BlockId(10)))
            .unwrap();
        assert_eq!(
            cache.nodes[idx].tier_location,
            Some(BlockLocation::Gpu { slot: 7 })
        );
        assert_eq!(cache.nodes[idx].byte_len, 4096);
        assert_eq!(
            cache.nodes[idx].session_id,
            Some(SessionId::from("session-1"))
        );
        assert_eq!(cache.nodes[idx].soft_pin_until, Some(42));

        assert!(cache.update_block_metadata(
            BlockId(10),
            BlockMetadataUpdate {
                location: None,
                byte_len: None,
                session_id: Some(None),
                soft_pin_until: Some(None),
            }
        ));
        assert_eq!(
            cache.nodes[idx].tier_location,
            Some(BlockLocation::Gpu { slot: 7 })
        );
        assert_eq!(cache.nodes[idx].byte_len, 4096);
        assert_eq!(cache.nodes[idx].session_id, None);
        assert_eq!(cache.nodes[idx].soft_pin_until, None);
    }

    #[test]
    fn lookup_refreshes_existing_soft_pin_without_starting_new_pins() {
        let mut cache = RadixCache::with_soft_pin_keepalive(4, 64);
        cache.insert(&[1, 2, 3, 4], &bids(&[10]));

        let idx = cache
            .nodes
            .iter()
            .position(|node| node.block_id == Some(BlockId(10)))
            .unwrap();
        assert_eq!(cache.nodes[idx].soft_pin_until, None);

        let (_, blocks) = cache.lookup(&[1, 2, 3, 4]);
        assert_eq!(blocks, bids(&[10]));
        assert_eq!(
            cache.nodes[idx].soft_pin_until, None,
            "lookup should not start pinning cold blocks"
        );
        cache.release(&blocks);

        assert!(cache.set_block_soft_pin_until(BlockId(10), Some(20)));
        cache.clock = 10;

        let (_, blocks) = cache.lookup(&[1, 2, 3, 4]);
        assert_eq!(blocks, bids(&[10]));
        assert_eq!(cache.nodes[idx].soft_pin_until, Some(75));
        cache.release(&blocks);

        let outcome = cache.lookup_or_stage(&[1, 2, 3, 4], LookupHeuristics::default(), None);
        assert_eq!(outcome.matched_len, 4);
        assert_eq!(cache.nodes[idx].soft_pin_until, Some(76));
        cache.release(&[BlockId(10)]);
    }
}
