# Prefix cache eviction prefers strict leaves

## Goal

Make KV tier pressure reclaim the least destructive prefix-cache victim first:
strict resident leaves before resident parents that only have demoted children.

## Hypothesis

The existing relaxed frontier is correctness-safe because a resident parent
whose descendants are all demoted can be severed together with those demoted
descendants. It is not always the best first victim: severing a strict leaf
drops one resident page, while severing a demoted-only parent also drops prefix
shape and invalidates tier keys below it. A two-pass LRU should preserve the
safe fallback while preferring strict leaves whenever any exist.

## Params

- `RadixCache::least_recent_evictable_leaf` now scans strict resident leaves
  first (`children.is_empty()`), ordered by `last_access`.
- If no strict leaf exists, it falls back to the existing relaxed frontier:
  idle resident nodes with no resident descendants.
- Tier demotion remains deepest-first; a parent can still demote or evict after
  its child has been demoted.

## Env

Local macOS Rust release checks. This is backend-neutral `infer-core` scheduler
logic; no CUDA or Metal kernel path changed.

## Results

Verification:

- `cargo fmt --check`: passed.
- `cargo test -p infer-core --release lru_prefers_strict_leaf_before_demoted_only_parent -- --nocapture`: 1 passed.
- `cargo test -p infer-core --release radix -- --nocapture`: 7 passed.
- `cargo test -p infer-core --release -- --nocapture`: 51 passed, 1 ignored.
- `cargo clippy -p infer-core --release -- -D warnings`: passed.

New regression coverage:

- `lru_prefers_strict_leaf_before_demoted_only_parent` builds a cache where an
  older relaxed parent and a newer strict leaf are both reclaimable. The old
  mixed-LRU policy would choose the parent first; the new policy must evict the
  strict leaf first, then fall back to the parent.
- `tier_promote_failure_truncates_match_and_reprefills` now asserts the failed
  tier key is dropped precisely instead of requiring the whole mock store to be
  empty. Leaf-first pressure can legitimately demote another cold page while
  the tail is recomputed.

## Problems

This is a policy/correctness-shape fix, not a throughput claim. No guideLLM
sweep was run because the change only affects which cold prefix block is chosen
under cache pressure; service-level latency impact depends on workload reuse
shape and should be measured in a separate pressure benchmark.

## Learnings

For a two-tier prefix cache, "safe to evict" and "least destructive to evict"
are different predicates. Keep the relaxed frontier for progress, but make the
default victim order preserve reachable prefix shape first.
