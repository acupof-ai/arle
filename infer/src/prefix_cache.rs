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

// ============================================================================
// Types
// ============================================================================

/// Opaque GPU KV cache block identifier. Assigned by the block allocator
/// (see `block_manager.rs`) and stored in the cache node.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct BlockId(pub u32);

/// A node in the radix tree.
#[derive(Serialize, Deserialize)]
struct Node {
    /// Token sequence stored on this edge (from parent to this node).
    tokens: Vec<u32>,
    /// GPU block ID cached for this node. `None` if tokens < block_size or
    /// if this node has not yet been committed to the GPU.
    block_id: Option<BlockId>,
    /// Number of in-flight requests currently pinning this node.
    ref_count: u32,
    /// Monotonically increasing access clock (set on insert/lookup).
    last_access: u64,
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
            children: HashMap::new(),
        }
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
    /// Index 0 is always the virtual root (empty token sequence).
    // root index is always 0
    /// Block size: a node gets a block_id only when it holds exactly
    /// `block_size` tokens. Must match the paged KV block size.
    block_size: usize,
    /// Monotonically increasing clock for LRU tracking.
    clock: u64,
}

impl RadixCache {
    /// Create a new empty radix cache.
    ///
    /// `block_size` must match the KV block size used by the block manager.
    pub fn new(block_size: usize) -> Self {
        assert!(block_size > 0, "block_size must be > 0");
        let root = Node::new(vec![], None, 0);
        Self {
            nodes: vec![root],
            block_size,
            clock: 0,
        }
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// Returns the root node index (always 0).
    fn root() -> usize {
        0
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
                        self.nodes[child_idx].block_id = Some(blocks[block_idx]);
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
                    let shared_idx = self.nodes.len();
                    self.nodes.push(shared_node);

                    // Rewire the original child to become a child of the shared node.
                    self.nodes[child_idx].tokens = old_suffix;
                    self.nodes[child_idx].block_id = old_block;
                    self.nodes[child_idx].children = old_children;

                    let first_old = self.nodes[child_idx].tokens[0];
                    self.nodes[shared_idx].children.insert(first_old, child_idx);

                    // Replace the original child pointer with the shared node.
                    self.nodes[node_idx].children.insert(next_token, shared_idx);

                    // Don't advance block_idx — the shared node has < block_size tokens.
                    break;
                }
            } else {
                // No matching child — insert remaining tokens as a new subtree.
                while pos < tokens.len() && block_idx < blocks.len() {
                    let end = (pos + self.block_size).min(tokens.len());
                    let edge_tokens = tokens[pos..end].to_vec();
                    let block_id = if edge_tokens.len() == self.block_size {
                        let bid = blocks[block_idx];
                        block_idx += 1;
                        Some(bid)
                    } else {
                        None
                    };

                    let new_node = Node::new(edge_tokens.clone(), block_id, now);
                    let new_idx = self.nodes.len();
                    self.nodes.push(new_node);

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

        freed
    }

    // -------------------------------------------------------------------------
    // Stats
    // -------------------------------------------------------------------------

    /// Total number of nodes in the tree (including root).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of cached blocks (nodes with a `block_id`).
    pub fn cached_block_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.block_id.is_some()).count()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn bids(ids: &[u32]) -> Vec<BlockId> {
        ids.iter().map(|&id| BlockId(id)).collect()
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

    // ──────────────────────────────────────────────────────────────────
    // Serde round-trip — Tiered KV Cache §P3 session persistence gate.
    // The on-disk format is a straightforward serde representation; this
    // test pins the expectation that a full insert → lookup state can be
    // JSON-roundtripped without loss and that lookups on the restored
    // cache still find every previously-cached block.
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn radix_cache_serde_roundtrip_preserves_lookups() {
        let mut cache = RadixCache::new(4);

        // Populate with three sequences sharing a common prefix so the
        // tree has a non-trivial shape with a split.
        cache.insert(&[1, 2, 3, 4], &bids(&[10]));
        cache.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &bids(&[10, 20]));
        cache.insert(&[1, 2, 3, 4, 9, 10, 11, 12], &bids(&[10, 30]));

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
    fn block_id_serde_roundtrip() {
        // BlockId is serialized as a newtype around u32 — keep this
        // explicit so the wire format is stable.
        let id = BlockId(42);
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "42", "BlockId(N) must serialize as the bare N");
        let back: BlockId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, id);
    }
}
