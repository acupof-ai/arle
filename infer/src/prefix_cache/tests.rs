use std::collections::HashMap;

use super::*;
use crate::kv_tier::LookupHeuristics;
use crate::scheduler::policy::LruEviction;

fn bids(ids: &[u32]) -> Vec<BlockId> {
    ids.iter().map(|&id| BlockId(id)).collect()
}

fn assert_block_path_alignment(cache: &RadixCache) {
    fn walk(cache: &RadixCache, node_idx: usize, parent_path_len: usize) {
        let node = &cache.nodes[node_idx];
        let path_len = parent_path_len + node.tokens.len();
        if let Some(block_id) = node.block_id {
            assert!(
                cache.path_is_block_aligned(path_len),
                "block-bearing node {block_id:?} ended at non-boundary path len {path_len}",
            );
        }
        for &child_idx in node.children.values() {
            walk(cache, child_idx, path_len);
        }
    }

    for &child_idx in cache.nodes[RadixCache::root()].children.values() {
        walk(cache, child_idx, 0);
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
    let () = cache.release(&bids(&[20])); // release so ref_count drops

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

#[test]
fn evictable_pages_counts_only_ready_unpinned_blocks() {
    let mut cache = RadixCache::new(4);
    cache.insert(&[1, 2, 3, 4], &bids(&[10]));
    cache.insert(&[5, 6, 7, 8], &bids(&[20]));
    cache.insert(&[9, 10, 11, 12], &bids(&[30]));

    let _held = cache.lookup(&[1, 2, 3, 4]);
    cache.clock = 10;
    assert!(cache.set_block_soft_pin_until(BlockId(20), Some(20)));
    assert!(cache.update_block_metadata(
        BlockId(30),
        BlockMetadataUpdate {
            entry_state: Some(IndexEntryState::Pending),
            ..BlockMetadataUpdate::default()
        }
    ));

    assert_eq!(cache.cached_block_count(), 3);
    assert_eq!(cache.evictable_tokens(), 0);
    assert_eq!(cache.evictable_pages(4), 0);

    cache.release(&bids(&[10]));
    assert!(cache.set_block_soft_pin_until(BlockId(20), None));
    assert!(cache.update_block_metadata(
        BlockId(30),
        BlockMetadataUpdate {
            entry_state: Some(IndexEntryState::Ready),
            ..BlockMetadataUpdate::default()
        }
    ));

    assert_eq!(cache.evictable_tokens(), 12);
    assert_eq!(cache.evictable_pages(4), 3);
    assert_eq!(cache.evictable_pages(8), 2);
}

#[test]
fn block_evictable_respects_tier_location() {
    let mut cache = RadixCache::new(4);
    cache.insert(&[1, 2, 3, 4], &bids(&[10]));
    cache.insert(&[5, 6, 7, 8], &bids(&[20]));
    cache.insert(&[9, 10, 11, 12], &bids(&[30]));

    assert!(cache.set_block_location(BlockId(10), BlockLocation::Gpu { slot: 0 }));
    assert!(cache.set_block_location(BlockId(20), BlockLocation::HostPinned { offset: 4096 }));
    assert!(cache.set_block_location(
        BlockId(30),
        BlockLocation::Disk {
            fingerprint: crate::types::BlockFingerprint([0xAB; 16]),
            payload_len: 1024,
        }
    ));

    assert!(cache.is_block_evictable(BlockId(10), None));
    assert!(cache.is_block_evictable(BlockId(10), Some(Tier::Gpu)));
    assert!(!cache.is_block_evictable(BlockId(20), Some(Tier::Gpu)));
    assert!(!cache.is_block_evictable(BlockId(30), Some(Tier::Gpu)));

    let _held = cache.lookup(&[1, 2, 3, 4]);
    assert!(!cache.is_block_evictable(BlockId(10), Some(Tier::Gpu)));
    cache.release(&bids(&[10]));
    assert!(cache.is_block_evictable(BlockId(10), Some(Tier::Gpu)));
}

#[test]
fn block_evictable_counts_cascade_parents() {
    let mut cache = RadixCache::new(4);
    cache.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &bids(&[10, 20]));
    assert!(cache.set_block_location(BlockId(10), BlockLocation::Gpu { slot: 0 }));
    assert!(cache.set_block_location(BlockId(20), BlockLocation::Gpu { slot: 0 }));

    assert!(cache.is_block_evictable(BlockId(10), Some(Tier::Gpu)));
    assert!(cache.is_block_evictable(BlockId(20), Some(Tier::Gpu)));
    assert_eq!(cache.evictable_pages(4), 2);

    let freed = cache.evict_with_policy(
        &crate::scheduler::policy::LruEviction,
        SchedulerSignals::default(),
        2,
        Some(Tier::Gpu),
    );
    assert_eq!(freed, bids(&[20, 10]));
}

#[test]
fn block_evictable_rejects_parent_with_pinned_child() {
    let mut cache = RadixCache::new(4);
    cache.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &bids(&[10, 20]));
    assert!(cache.set_block_location(BlockId(10), BlockLocation::Gpu { slot: 0 }));
    assert!(cache.set_block_location(BlockId(20), BlockLocation::Gpu { slot: 0 }));
    assert!(cache.set_block_soft_pin_until(BlockId(20), Some(100)));

    assert!(!cache.is_block_evictable(BlockId(10), Some(Tier::Gpu)));
    assert!(!cache.is_block_evictable(BlockId(20), Some(Tier::Gpu)));
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
    assert_block_path_alignment(&cache);

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
    assert_block_path_alignment(&cache);

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
fn reinsert_after_split_does_not_reuse_first_block_id() {
    // Codex-caught P1 regression (2026-04-19): the initial split fix
    // created short-edge block-bearing nodes, and the full-match branch
    // only advanced block_idx when child.tokens.len() == block_size,
    // causing subsequent walks through the short edge to reuse
    // blocks[0] for the second block. Lookup then returned [100, 100]
    // instead of [100, 300].
    let mut cache = RadixCache::new(4);
    cache.insert(&[1, 2, 3, 4], &bids(&[10]));
    cache.insert(&[1, 5, 6, 7, 8, 9, 10, 11], &bids(&[100, 200]));

    let inserted = cache.insert(&[1, 5, 6, 7, 12, 13, 14, 15], &bids(&[100, 300]));
    assert_eq!(inserted, 8);
    assert_block_path_alignment(&cache);

    let (len, blocks) = cache.lookup(&[1, 5, 6, 7, 12, 13, 14, 15]);
    assert_eq!(len, 8);
    assert_eq!(blocks, bids(&[100, 300]));
}

#[test]
fn split_can_leave_short_edge_block_bearing_node_when_path_is_aligned() {
    let mut cache = RadixCache::new(4);
    cache.insert(&[1, 2, 3, 4], &bids(&[10]));
    cache.insert(&[1, 5, 6, 7, 8, 9, 10, 11], &bids(&[100, 200]));

    assert_block_path_alignment(&cache);

    let shared_idx = *cache.nodes[RadixCache::root()]
        .children
        .get(&1)
        .expect("root should point at the shared split node");
    let shared = &cache.nodes[shared_idx];
    assert_eq!(shared.tokens, vec![1]);
    assert!(shared.block_id.is_none());

    let short_edge_idx = *shared
        .children
        .get(&5)
        .expect("shared split node should have the divergent block child");
    let short_edge = &cache.nodes[short_edge_idx];
    assert_eq!(short_edge.tokens, vec![5, 6, 7]);
    assert_eq!(short_edge.block_id, Some(BlockId(100)));
    assert!(
        short_edge.tokens.len() < cache.block_size(),
        "edge itself should stay compressed after the split",
    );
}

#[test]
fn split_mid_block_via_shared_intermediate_registers_tail_block() {
    // Codex-caught P1 regression round 2 (2026-04-19): after the walk
    // passed through a non-block-bearing shared intermediate, `pos` was
    // mid-block but the split branch computed `rest_end = pos + block_size`,
    // which overshot the block boundary and failed the bounds check,
    // dropping the tail block silently. Fix: compute rest_end from
    // (block_idx+1)*block_size.
    let mut cache = RadixCache::new(4);
    cache.insert(&[1, 2, 3, 4], &bids(&[10]));
    cache.insert(&[1, 5, 6, 7, 8, 9, 10, 11], &bids(&[100, 200]));
    cache.insert(&[1, 5, 6, 7, 8, 12, 13, 14], &bids(&[100, 300]));

    // Walk: @1→shared_a (no block), @5→blk100 (advance), @8→shared_b
    // (no block — now pos=5 mid-block), @9→blk200 matches only [9],
    // split at match_len=1. rest_end must be 8, not 9 (8+1+4).
    let inserted = cache.insert(&[1, 5, 6, 7, 8, 9, 20, 21], &bids(&[100, 400]));
    assert_eq!(inserted, 8);
    assert_block_path_alignment(&cache);

    let (len, blocks) = cache.lookup(&[1, 5, 6, 7, 8, 9, 20, 21]);
    assert_eq!(len, 8);
    assert_eq!(blocks, bids(&[100, 400]));
}

#[test]
fn else_branch_mid_block_via_shared_intermediate_registers_tail_block() {
    // Codex-caught P1 regression round 2 (2026-04-19): symmetric to the
    // split case but lands in the else (no-matching-child) branch. Walk
    // lands mid-block (pos=5, block_idx=1), else branch's edge window
    // used `pos+block_size` so the first new edge was len 3 < block_size
    // and was registered without a block_id — dropping the caller's
    // final block. Fix: compute edge end from (block_idx+1)*block_size,
    // assign block_id when end lands on the next boundary.
    let mut cache = RadixCache::new(4);
    cache.insert(&[1, 2, 3, 4], &bids(&[10]));
    cache.insert(&[1, 5, 6, 7, 8, 9, 10, 11], &bids(&[100, 200]));
    cache.insert(&[1, 5, 6, 7, 8, 12, 13, 14], &bids(&[100, 300]));

    // Walk: @1→shared_a (no block), @5→blk100 (advance), @8→shared_b
    // (no block — pos=5 mid-block). shared_b has children at 9 and 12;
    // token 16 hits no match → else branch. First edge must complete
    // block 2 with tokens[5..8] and take blocks[1].
    let inserted = cache.insert(&[1, 5, 6, 7, 8, 16, 17, 18], &bids(&[100, 500]));
    assert_eq!(inserted, 8);
    assert_block_path_alignment(&cache);

    let (len, blocks) = cache.lookup(&[1, 5, 6, 7, 8, 16, 17, 18]);
    assert_eq!(len, 8);
    assert_eq!(blocks, bids(&[100, 500]));
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
    assert_block_path_alignment(&cache);

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
        None,
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
        None,
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
        None,
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

    let expected = cache.lookup_or_stage(&tokens, LookupHeuristics::default());
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

    let report = restored.reconcile(&HashMap::from([(fp_a, BlockId(100)), (fp_b, BlockId(200))]));
    assert_eq!(report.remapped, 2);
    assert_eq!(report.tombstoned, 0);

    let outcome = restored.lookup_or_stage(&tokens, LookupHeuristics::default());
    let block_ids: Vec<Option<BlockId>> =
        outcome.blocks.iter().map(|block| block.block_id).collect();
    let hit_kinds: Vec<HitKind> = outcome.blocks.iter().map(|block| block.hit_kind).collect();

    assert_eq!(outcome.matched_len, expected.matched_len);
    assert_eq!(hit_kinds, expected_hit_kinds);
    assert_eq!(block_ids, vec![Some(BlockId(100)), Some(BlockId(200))]);
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

    let outcome = cache.lookup_or_stage(&[1, 2, 3, 4], LookupHeuristics::default());

    assert_eq!(outcome.matched_len, 4);
    assert_eq!(
        outcome.blocks,
        vec![LookupBlock {
            block_id: Some(BlockId(10)),
            hit_kind: HitKind::ReadyOnGpu,
        }]
    );
    assert!(!outcome.recompute_advised);
}

#[test]
fn lookup_or_stage_marks_host_stage() {
    let mut cache = RadixCache::new(4);
    cache.insert(&[1, 2, 3, 4], &bids(&[10]));
    assert!(cache.set_block_location(BlockId(10), BlockLocation::HostPinned { offset: 4096 }));
    assert!(cache.set_block_byte_len(BlockId(10), 8192));

    let outcome = cache.lookup_or_stage(&[1, 2, 3, 4], LookupHeuristics::default());

    assert_eq!(outcome.matched_len, 4);
    assert_eq!(
        outcome.blocks,
        vec![LookupBlock {
            block_id: Some(BlockId(10)),
            hit_kind: HitKind::StagingFromHost,
        }]
    );
    assert!(!outcome.recompute_advised);
}

#[test]
fn lookup_or_stage_preserves_mixed_gpu_and_host_hits() {
    let mut cache = RadixCache::new(4);
    cache.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &bids(&[10, 20]));
    assert!(cache.set_block_location(BlockId(10), BlockLocation::Gpu { slot: 1 }));
    assert!(cache.set_block_byte_len(BlockId(10), 4096));
    assert!(cache.set_block_location(BlockId(20), BlockLocation::HostPinned { offset: 8192 }));
    assert!(cache.set_block_byte_len(BlockId(20), 4096));

    let outcome = cache.lookup_or_stage(&[1, 2, 3, 4, 5, 6, 7, 8], LookupHeuristics::default());

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
                hit_kind: HitKind::StagingFromHost,
            },
        ]
    );
    assert!(!outcome.recompute_advised);
}

#[test]
fn release_before_remap_keeps_shared_prefix_reusable() {
    let mut cache = RadixCache::new(4);
    let tokens: Vec<u32> = (1..=8).collect();
    let old_fingerprints = [BlockFingerprint([0x11; 16]), BlockFingerprint([0x22; 16])];
    let new_fingerprints = [BlockFingerprint([0x33; 16]), BlockFingerprint([0x44; 16])];

    assert_eq!(
        cache.insert_with_fingerprints(&tokens, &bids(&[10, 20]), &old_fingerprints),
        8
    );
    assert!(cache.set_block_location(BlockId(10), BlockLocation::Gpu { slot: 1 }));
    assert!(cache.set_block_byte_len(BlockId(10), 4096));
    assert!(cache.set_block_location(BlockId(20), BlockLocation::HostPinned { offset: 8192 }));
    assert!(cache.set_block_byte_len(BlockId(20), 4096));

    let first = cache.lookup_or_stage(&tokens, LookupHeuristics::default());
    let second = cache.lookup_or_stage(&tokens, LookupHeuristics::default());
    assert_eq!(first.matched_len, 8);
    assert_eq!(second.matched_len, 8);
    assert_eq!(cache.block_metadata(BlockId(10)).unwrap().ref_count, 2);
    assert_eq!(cache.block_metadata(BlockId(20)).unwrap().ref_count, 2);

    cache.release(&[BlockId(10), BlockId(20)]);
    cache.release(&[BlockId(10), BlockId(20)]);
    assert_eq!(cache.block_metadata(BlockId(10)).unwrap().ref_count, 0);
    assert_eq!(cache.block_metadata(BlockId(20)).unwrap().ref_count, 0);

    assert_eq!(
        cache.insert_with_fingerprints(&tokens, &bids(&[30, 40]), &new_fingerprints),
        8
    );
    assert!(cache.set_block_location(BlockId(30), BlockLocation::Gpu { slot: 3 }));
    assert!(cache.set_block_byte_len(BlockId(30), 4096));
    assert!(cache.set_block_location(BlockId(40), BlockLocation::Gpu { slot: 3 }));
    assert!(cache.set_block_byte_len(BlockId(40), 4096));
    assert!(cache.block_metadata(BlockId(10)).is_none());
    assert!(cache.block_metadata(BlockId(20)).is_none());

    let remapped = cache.lookup_or_stage(&tokens, LookupHeuristics::default());
    assert_eq!(remapped.matched_len, 8);
    assert_eq!(
        remapped.blocks,
        vec![
            LookupBlock {
                block_id: Some(BlockId(30)),
                hit_kind: HitKind::ReadyOnGpu,
            },
            LookupBlock {
                block_id: Some(BlockId(40)),
                hit_kind: HitKind::ReadyOnGpu,
            },
        ]
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
            fingerprint: BlockFingerprint([0x11; 16]),
            payload_len: 2 * 1024 * 1024,
        }
    ));
    assert!(cache.set_block_byte_len(BlockId(10), 2 * 1024 * 1024));

    let outcome = cache.lookup_or_stage(
        &tokens,
        LookupHeuristics {
            prefill_tokens_per_sec: 200_000.0,
            host_bandwidth_bytes_per_sec: 25.0 * 1024.0 * 1024.0 * 1024.0,
            disk_bandwidth_bytes_per_sec: 50.0 * 1024.0 * 1024.0,
        },
    );

    assert_eq!(outcome.matched_len, 16);
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
        fingerprint: BlockFingerprint([0x22; 16]),
        payload_len: 4096,
    });
    cache.nodes[idx].byte_len = 4096;

    let outcome = cache.lookup_or_stage(&[1, 2, 3, 4], LookupHeuristics::default());

    assert_eq!(outcome.matched_len, 0);
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
        fingerprint: BlockFingerprint([0x33; 16]),
        payload_len: 4096,
    });
    cache.nodes[idx].byte_len = 4096;

    let outcome = cache.lookup_or_stage(&tokens, LookupHeuristics::default());
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
            host_spill_pin_until: Some(Some(84)),
            entry_state: Some(IndexEntryState::Pending),
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
    assert_eq!(cache.nodes[idx].host_spill_pin_until, Some(84));
    assert_eq!(cache.nodes[idx].entry_state, IndexEntryState::Pending);
    assert_eq!(cache.nodes[idx].store_state, StoreState::Idle);

    assert!(cache.update_block_metadata(
        BlockId(10),
        BlockMetadataUpdate {
            location: None,
            byte_len: None,
            session_id: Some(None),
            soft_pin_until: Some(None),
            host_spill_pin_until: Some(None),
            entry_state: Some(IndexEntryState::Ready),
        }
    ));
    assert_eq!(
        cache.nodes[idx].tier_location,
        Some(BlockLocation::Gpu { slot: 7 })
    );
    assert_eq!(cache.nodes[idx].byte_len, 4096);
    assert_eq!(cache.nodes[idx].session_id, None);
    assert_eq!(cache.nodes[idx].soft_pin_until, None);
    assert_eq!(cache.nodes[idx].host_spill_pin_until, None);
    assert_eq!(cache.nodes[idx].entry_state, IndexEntryState::Ready);
    assert_eq!(cache.nodes[idx].store_state, StoreState::Idle);
}

#[test]
fn spill_selection_ignores_lookup_soft_pin_for_host_blocks() {
    let mut cache = RadixCache::new(4);
    cache.insert(&[1, 2, 3, 4], &bids(&[10]));
    assert!(cache.set_block_location(BlockId(10), BlockLocation::HostPinned { offset: 4096 }));
    assert!(cache.set_block_soft_pin_until(BlockId(10), Some(32)));
    cache.clock = 16;

    let selected = cache.select_blocks_with_policy(
        &LruEviction,
        SchedulerSignals::default(),
        1,
        Some(Tier::HostPinned),
        BlockSelectionIntent::Spill,
    );

    assert_eq!(selected, vec![BlockId(10)]);
}

#[test]
fn spill_selection_respects_host_spill_pin() {
    let mut cache = RadixCache::new(4);
    cache.insert(&[1, 2, 3, 4], &bids(&[10]));
    assert!(cache.set_block_location(BlockId(10), BlockLocation::HostPinned { offset: 4096 }));
    assert!(cache.set_block_host_spill_pin_until(BlockId(10), Some(32)));
    cache.clock = 16;

    let pinned = cache.select_blocks_with_policy(
        &LruEviction,
        SchedulerSignals::default(),
        1,
        Some(Tier::HostPinned),
        BlockSelectionIntent::Spill,
    );
    assert!(pinned.is_empty());

    cache.clock = 32;
    let released = cache.select_blocks_with_policy(
        &LruEviction,
        SchedulerSignals::default(),
        1,
        Some(Tier::HostPinned),
        BlockSelectionIntent::Spill,
    );
    assert_eq!(released, vec![BlockId(10)]);
}

#[test]
fn drain_selection_ignores_host_spill_pin() {
    let mut cache = RadixCache::new(4);
    cache.insert(&[1, 2, 3, 4], &bids(&[10]));
    assert!(cache.set_block_location(BlockId(10), BlockLocation::HostPinned { offset: 4096 }));
    assert!(cache.set_block_host_spill_pin_until(BlockId(10), Some(32)));
    cache.clock = 16;

    let selected = cache.select_blocks_with_policy(
        &LruEviction,
        SchedulerSignals::default(),
        1,
        Some(Tier::HostPinned),
        BlockSelectionIntent::Drain,
    );

    assert_eq!(selected, vec![BlockId(10)]);
}

#[test]
fn spill_selection_skips_non_idle_store_states() {
    let mut cache = RadixCache::new(4);
    cache.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &bids(&[10, 20]));
    assert!(cache.set_block_location(BlockId(10), BlockLocation::HostPinned { offset: 4096 }));
    assert!(cache.set_block_location(BlockId(20), BlockLocation::HostPinned { offset: 8192 }));
    assert!(cache.set_block_store_state(BlockId(10), StoreState::Storing));

    let selected = cache.select_blocks_with_policy(
        &LruEviction,
        SchedulerSignals::default(),
        2,
        Some(Tier::HostPinned),
        BlockSelectionIntent::Spill,
    );

    assert_eq!(selected, vec![BlockId(20)]);
}

#[test]
fn selection_sorts_lowest_score_first() {
    let mut cache = RadixCache::new(4);
    cache.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &bids(&[10, 20]));
    assert!(cache.set_block_location(BlockId(10), BlockLocation::Gpu { slot: 0 }));
    assert!(cache.set_block_location(BlockId(20), BlockLocation::Gpu { slot: 1 }));

    let idx10 = cache
        .nodes
        .iter()
        .position(|node| node.block_id == Some(BlockId(10)))
        .unwrap();
    let idx20 = cache
        .nodes
        .iter()
        .position(|node| node.block_id == Some(BlockId(20)))
        .unwrap();
    cache.nodes[idx10].last_access = 5;
    cache.nodes[idx20].last_access = 10;

    let selected = cache.select_blocks_with_policy(
        &LruEviction,
        SchedulerSignals::default(),
        1,
        Some(Tier::Gpu),
        BlockSelectionIntent::Evict,
    );

    assert_eq!(selected, vec![BlockId(10)]);
}

#[test]
fn block_state_helpers_cover_read_and_store_lifecycle() {
    let mut cache = RadixCache::new(4);
    cache.insert(&[1, 2, 3, 4], &bids(&[10]));

    assert!(cache.insert_pending(BlockId(10)));
    assert_eq!(
        cache.block_metadata(BlockId(10)).unwrap().entry_state,
        IndexEntryState::Pending
    );

    assert!(cache.mark_block_evicting(BlockId(10)));
    assert_eq!(
        cache.block_metadata(BlockId(10)).unwrap().entry_state,
        IndexEntryState::Evicting
    );

    assert!(cache.commit_ready(BlockId(10), Some(BlockLocation::Gpu { slot: 3 })));
    let metadata = cache.block_metadata(BlockId(10)).unwrap();
    assert_eq!(metadata.entry_state, IndexEntryState::Ready);
    assert_eq!(metadata.location, Some(BlockLocation::Gpu { slot: 3 }));

    assert!(cache.mark_block_store_pending(BlockId(10)));
    assert_eq!(
        cache.block_metadata(BlockId(10)).unwrap().store_state,
        StoreState::Pending
    );

    assert!(cache.mark_block_storing(BlockId(10)));
    assert_eq!(
        cache.block_metadata(BlockId(10)).unwrap().store_state,
        StoreState::Storing
    );

    let disk_location = BlockLocation::Disk {
        fingerprint: BlockFingerprint([0x44; 16]),
        payload_len: 4096,
    };
    assert!(cache.mark_block_stored(BlockId(10), Some(disk_location.clone())));
    let metadata = cache.block_metadata(BlockId(10)).unwrap();
    assert_eq!(metadata.store_state, StoreState::Stored);
    assert_eq!(metadata.location, Some(disk_location));

    assert!(cache.mark_block_store_failed(BlockId(10)));
    assert_eq!(
        cache.block_metadata(BlockId(10)).unwrap().store_state,
        StoreState::Failed
    );
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

    let outcome = cache.lookup_or_stage(&[1, 2, 3, 4], LookupHeuristics::default());
    assert_eq!(outcome.matched_len, 4);
    assert_eq!(cache.nodes[idx].soft_pin_until, Some(76));
    cache.release(&[BlockId(10)]);
}
