# 2026-04-15 · Tiered KV M3b local contract tranche

## Context

The 2026-04-15 M2b + M0.3 + M3a local batches had already landed the safe
same-slot resurrection path, BF16 `page_size=16`, and the first host-tier
scaffolding. The next local-only step for M3b was to land the **contract**
for staged lookups and the **pure page lifecycle state machine** without
pretending that CUDA runtime behavior was already wired.

## What Worked

- Kept the tranche local-only and structural: `lookup_or_stage`,
  `LookupOutcome`, `HitKind`, `LookupHeuristics`, `StageTicket`, and
  `StagePlanner` all landed in always-on Rust code.
- Let `RadixCache` classify GPU / host / disk / tombstone hits directly
  from node metadata, with small metadata stamp helpers so tests stop
  mutating internals ad hoc.
- Added ticketed staging commands/events on the coordinator skeleton,
  which keeps the scheduler transport-agnostic before real CUDA staging is
  wired.
- Added a pure `Free | Resident | Demoting` page-lifecycle state machine
  with explicit invalid-transition errors and cancel-on-hit coverage.
- Verified locally with:
  - `cargo test -p infer --no-default-features --features no-cuda prefix_cache`
  - `cargo test -p infer --no-default-features --features no-cuda kv_tier`
  - `cargo check -p infer --tests --no-default-features --features cuda,no-cuda`

## Rule

When a tiered-KV milestone is blocked on remote CUDA behavior, keep the
local tranche to **contracts + pure state machines** and make the boundary
explicit. Do not fake runtime wiring locally just to make the milestone
look "complete".
