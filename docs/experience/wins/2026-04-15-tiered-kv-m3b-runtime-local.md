# 2026-04-15 · Tiered KV M3b local runtime wire

## Context

After the M3b contract/state-machine tranche landed, the tree still had one
awkward gap: `lookup_or_stage`, `StageTicket`, `session_id`, and
`soft_pin_until` all existed in the type layer, but the CUDA scheduler still
admitted requests through the old `lookup()` path and evicted retained blocks
with static default signals.

This local batch moved the scheduler onto the smallest safe runtime wire
without pretending that staged bytes can already complete back onto GPU.

## What Worked

- Added a proactive prefix-cache pressure hook at admission so the scheduler
  can trim cold retained pages before slot assignment starts.
- Replaced admission's plain radix lookup with plannerless
  `lookup_or_stage(..., None)` classification:
  - `ReadyOnGpu` hits still drive same-slot resurrection
  - staged / recompute-advised hits are downgraded to cold prefill
  - no fake `StageTicket` completion path was introduced
- Plumbed queue/decode state into the prefix-cache eviction call site so
  future pressure-sensitive policies have a live signal to consume. Scoring
  output is currently bit-identical — no `EvictionPolicy` impl reads queue
  pressure yet.
- Stamped published radix blocks with truthful runtime metadata:
  `BlockLocation::Gpu { slot }`, `byte_len`, `session_id`, and a conservative
  soft-pin keepalive deadline.
- Session-scoped `ReadyOnGpu` hits now refresh that keepalive metadata
  without inventing a fake staging-complete path.
- Verified locally with:
  - `cargo test -p infer --no-default-features --features no-cuda prefix_cache`
  - `cargo check -p infer --tests --no-default-features --features cuda,no-cuda`
  - `cargo check -p infer --no-default-features --features metal`
  - `cargo fmt --all -- --check`

## Rule

When a staged-lookup design has no real completion path yet, the only safe
runtime wire is **classifier-first**: consume the contract for
`ReadyOnGpu` hits, downgrade every staged hit to recompute, and leave the
coordinator untouched until completion semantics exist.
