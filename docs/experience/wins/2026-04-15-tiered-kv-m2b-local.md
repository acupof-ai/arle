# 2026-04-15 · Tiered KV M2b local landing

## Context

M2a had already made radix-held T0 pages survive `free_slot`, but CUDA
scheduler admission still selected slots through the legacy
`cached_prompts` scan. That meant the retained pages existed physically but
were not yet load-bearing for real cross-request reuse. The 2026-04-15
batch had to make radix-driven reuse real without introducing unsafe
cross-slot page aliasing.

## What Worked

- Switched CUDA scheduler admission from the legacy
  `best_prefix_slot_for_cached_prompts` path to a radix-driven selector
  keyed by `block_owner_slots` and `slot_materialized_prompt_lens`.
- Replaced scheduler-owned `cached_prompts: Vec<Vec<u32>>` with reusable
  prefix metadata stored on `ActiveRequest`
  (`reusable_prefix_len`, `reusable_cached_prompt_len`).
- Made `step_new()` consume that metadata and reuse the matched prefix via
  safe same-slot resurrection. Exact-hit replay, partial-hit truncation,
  and prompt-prefix-of-cached fallback all remain model-aware.
- Added `GenerationState::prefetch_kv_to_gpu()` and wired it for Qwen3
  and Qwen3.5 so any CPU-offloaded contiguous KV is back on GPU
  before prefix reuse touches it.
- Aligned the deprecated single-request engine in `server_engine.rs` with
  the same correctness rules: prefetch before reuse, save a prompt snapshot
  after prefill, and restore/truncate before reusing an already-cached prompt.
- Added `alloc_pool_tokens_with_retry()` so prefix migration, prefill
  migration, and decode allocation can force one synchronous radix eviction
  before giving up on pool OOM.
- Added a retain hard cap (`0.90`) so `publish_to_prefix_cache()` fails
  open under pressure instead of pinning the pool into starvation.
- Added radix tombstone GC (`free_nodes`, `alloc_node`,
  `gc_orphan_tombstones()`) so repeated evict/insert cycles reclaim
  blockless structure instead of growing the node array monotonically.

## Rule

When prefix reuse reads contiguous KV after request cleanup, **prefetch
first**. Also, do **not** alias paged-pool pages across slots until the pool
has explicit alias-safe ownership semantics; safe same-slot resurrection is
the correct M2b boundary.
