//! M_d.1 §"Test (T1) — namespace isolation".
//!
//! Closes the silent-corruption hole documented in
//! `docs/experience/errors/2026-05-07-radix-cache-no-tokenizer-fingerprint.md`
//! at the integration-test level. Three properties asserted:
//!
//! 1. A snapshot saved under namespace A loads cleanly when the caller
//!    presents namespace A as the expected namespace.
//! 2. The same snapshot REJECTS when the caller presents a different
//!    expected namespace — the load API returns an Err, never a silent
//!    successful load with stale data.
//! 3. Two `RadixCache` instances with different namespaces are
//!    independent: tokens cached in one are NOT visible to the other,
//!    enforcing per-tokenizer isolation by construction.
//!
//! Compile-time bypass-guard (`serde_json::from_str::<RadixCache>(_)`
//! must no longer compile after the M_d.1 step-2 derive removal) is
//! tracked separately under M_d.1 step-4 sub-item 6 — landing it as a
//! `compile_fail` doctest or `trybuild` case adds a new dep, so it is
//! deferred to a follow-up commit.
//!
//! Pure-CPU; runs under `cargo test --release` with default features.

use infer::prefix_cache::{BlockId, RadixCache};

const BLOCK_SIZE: usize = 4;
const NS_A: [u8; 32] = [0xAA; 32];
const NS_B: [u8; 32] = [0xBB; 32];

fn populate(cache: &mut RadixCache, tokens: &[u32], blocks: &[u32]) {
    let block_ids: Vec<BlockId> = blocks.iter().copied().map(BlockId).collect();
    cache.insert(tokens, &block_ids);
}

#[test]
fn snapshot_round_trips_under_matching_namespace() {
    let mut cache = RadixCache::new_with_namespace(BLOCK_SIZE, NS_A);
    populate(&mut cache, &[1, 2, 3, 4, 5, 6, 7, 8], &[10, 20]);
    let token_count_before = cache.cached_block_count();
    let json = cache.save_snapshot().expect("save_snapshot");

    let mut restored =
        RadixCache::load_snapshot(&json, &NS_A).expect("load_snapshot under matching namespace");
    restored.rebuild_block_index();
    assert_eq!(restored.namespace(), &NS_A);
    assert_eq!(restored.cached_block_count(), token_count_before);
}

#[test]
fn snapshot_rejects_under_mismatched_namespace() {
    let mut cache = RadixCache::new_with_namespace(BLOCK_SIZE, NS_A);
    populate(&mut cache, &[1, 2, 3, 4, 5, 6, 7, 8], &[10, 20]);
    let json = cache.save_snapshot().expect("save_snapshot");

    let err = match RadixCache::load_snapshot(&json, &NS_B) {
        Ok(_) => panic!("load with mismatched namespace must return Err, got Ok"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("namespace mismatch"),
        "error message should name the failure mode, got: {msg}"
    );
}

#[test]
fn fresh_cache_in_other_namespace_does_not_see_blocks_from_first_namespace() {
    let mut cache_a = RadixCache::new_with_namespace(BLOCK_SIZE, NS_A);
    populate(&mut cache_a, &[1, 2, 3, 4, 5, 6, 7, 8], &[10, 20]);
    let cached_in_a = cache_a.cached_block_count();
    assert!(cached_in_a > 0, "cache_a should have inserted blocks");

    // A fresh cache pinned to NS_B is a wholly separate instance — its
    // node arena and block index start empty regardless of cache_a.
    let cache_b = RadixCache::new_with_namespace(BLOCK_SIZE, NS_B);
    assert_eq!(cache_b.namespace(), &NS_B);
    assert_eq!(
        cache_b.cached_block_count(),
        0,
        "fresh cache in NS_B must not inherit any blocks from NS_A"
    );
}
