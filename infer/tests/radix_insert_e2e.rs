//! G6: lock the RadixCache insert -> next lookup contract.
//!
//! The CUDA scheduler publishes completed prompts through `RadixCache::insert`
//! and reads the next same-prompt admission through `lookup_or_stage`. This
//! CPU-only test covers that shared cache contract without constructing a CUDA
//! scheduler or touching GPU state.

use infer::kv_tier::{HitKind, LookupHeuristics};
use infer::prefix_cache::{BlockId, RadixCache};

const BLOCK_SIZE: usize = 16;

fn block_aligned_prompt() -> Vec<u32> {
    (0..(BLOCK_SIZE * 2)).map(|token| token as u32).collect()
}

#[test]
fn same_prompt_second_lookup_reuses_full_inserted_prefix() {
    let prompt = block_aligned_prompt();
    let blocks = [BlockId(101), BlockId(117)];
    let mut cache = RadixCache::new(BLOCK_SIZE);

    let inserted = cache.insert(&prompt, &blocks);
    assert_eq!(
        inserted,
        prompt.len(),
        "block-aligned prompt should publish every prompt token"
    );
    assert_eq!(cache.cached_block_count(), blocks.len());

    let second_lookup = cache.lookup_or_stage(&prompt, LookupHeuristics::default());
    assert_eq!(
        second_lookup.matched_len,
        prompt.len(),
        "same prompt should hit the full inserted prefix on the next lookup"
    );
    assert!(
        !second_lookup.recompute_advised,
        "fresh T0 blocks should not be recompute-advised"
    );
    assert_eq!(second_lookup.blocks.len(), blocks.len());

    let matched_blocks = second_lookup
        .blocks
        .iter()
        .map(|block| (block.block_id, block.hit_kind))
        .collect::<Vec<_>>();
    assert_eq!(
        matched_blocks,
        vec![
            (Some(BlockId(101)), HitKind::ReadyOnGpu),
            (Some(BlockId(117)), HitKind::ReadyOnGpu),
        ],
        "freshly inserted radix blocks should be immediately runnable from T0"
    );
}
