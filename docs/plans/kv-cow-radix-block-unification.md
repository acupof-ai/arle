# KV COW / Radix / Block Unification Plan

## Context

The current CUDA tiered-KV path is already structurally correct in three key
ways:

1. `prefix_cache.rs` is the single metadata index for reusable prefixes.
2. `paged_kv.rs` supports direct GPU page attachment for reusable T0 prefixes.
3. decode already protects shared prefixes with tail-page copy-on-write before
   append.

What is still too implicit is the contract across those layers.

Today the implementation works, but several rules are encoded only by local
comments and branch structure:

- which prefix pages are allowed to be shared
- when a cached block is considered sealed vs still mutable
- what radix shape is valid after a mid-block split
- which objects are canonical for live T0 blocks vs staged T1/T2 bytes

This plan makes those rules explicit and then tightens the code so the runtime,
tests, and documentation all follow the same model.

## Problem Statement

Three issues are coupled:

1. **COW is correct but narrow.**
   `TokenKVPool::cow_tail_page_for_append()` protects the shared-tail append
   case, but the higher-level runtime contract is still implicit. Readers have
   to infer that only the hot tail is mutable and that radix-published blocks
   are sealed.

2. **Compressed-radix block semantics are hard to read.**
   `RadixCache::insert_with_fingerprints()` correctly handles partial-edge
   splits, but it relies on subtle `pos` / `block_idx` arithmetic. In a
   compressed radix, a block-bearing node may legitimately have a short edge
   after a mid-block split. That is not a bug, but the invariant is currently
   under-specified and easy to regress.

3. **Block-level runtime ownership is spread across multiple helpers.**
   `prefix_cache`, `block_to_pages`, `block_owner_slots`, and paged-KV page
   refcounts together implement the block lifecycle. The current code is
   correct enough, but it should expose one cleaner story for:
   - publish full blocks into the radix
   - attach shared prefix blocks to a fresh slot
   - promote staged blocks back into T0
   - mutate only the live tail via COW

## Goals

1. Make the COW contract explicit: sealed shared blocks are read-only; only the
   hot tail may become writable, and only through one COW boundary.
2. Keep the compressed radix, but make the block-boundary invariant explicit
   and centralized.
3. Reduce branch-specific block math and token-vector cloning in radix hot
   paths.
4. Clean up block-level scheduler helpers so publish / attach / promote follow
   one canonical full-block story.
5. Add tests that lock the shape and lifecycle in place.

## Non-Goals

- No new tier type or transport abstraction.
- No rewrite of `BlockId` semantics. `BlockId` remains a live pool-slot / first
  page identifier, not durable identity.
- No replacement of `BlockFingerprint`; persistence and reconciliation still
  key off fingerprints.
- No speculative “partial block persistence” path.
- No new parallel source of truth for tier location outside `RadixCache`.

## Current Truth

### Shared-prefix path

- `runtime.rs::build_prefix_admission_plan()` classifies prefix hits via
  `lookup_or_stage()`.
- Fully T0-ready paged-prefill requests attach radix-backed pages directly into
  an empty slot.
- Non-T0 staged hits go through `ReadmissionPlan -> WaitingFetch ->
  promote_fetched_prefix()`.

### Publish path

- `core.rs::publish_to_prefix_cache()` publishes only full blocks.
- Published blocks pin every physical page that backs the block.
- The scheduler records `block_to_pages` and `block_owner_slots` for live T0
  blocks.

### Mutation path

- `paged_kv.rs` clones the partially-filled tail page only when appending into
  a shared prefix page.
- Full blocks are immutable from the perspective of the radix / tiering path.
- Partial tails are not published, fingerprinted, staged, or demoted.

## Hard Invariants

### 1. Sealed block vs hot tail

- A **sealed block** is a full `block_size` token window whose backing pages
  have been published into the radix or promoted back into T0.
- A **hot tail** is the final partially-filled page of an active slot.
- Sealed blocks are shared read-only.
- The hot tail is the only writable frontier.

### 2. One data-plane COW boundary

- Shared-prefix mutation is allowed only at append time.
- The only data-plane COW operation is “clone the shared tail page before
  append”.
- There is no block-level COW at publish time, eviction time, or staged
  promotion time.

### 3. Compressed radix, boundary-on-path invariant

The radix remains edge-compressed.

That means a block-bearing node may legitimately have a short edge after a
mid-block split. The real invariant is:

- a node may carry `block_id` only when the cumulative path length to that node
  ends on a block boundary
- the node’s own edge length does **not** need to equal `block_size`

This distinction is load-bearing and must be encoded directly in helper
functions and tests.

### 4. Full blocks only across tier boundaries

- publish: full blocks only
- T0 -> T1/T2 demote: full blocks only
- T1/T2 -> T0 promote: full blocks only
- fingerprint computation: full blocks only

The decode tail stays request-private until it becomes full.

### 5. Live T0 block identity

For live T0 blocks:

- `BlockId` = first page id of the block’s current T0 allocation
- `block_to_pages[BlockId]` = canonical T0 page list for that block
- `RadixCache` metadata = canonical cross-tier location / state / fingerprint

No additional block registry should be introduced.

## Target Design

### A. Radix shape

`RadixCache` keeps compressed edges, but block-boundary operations move through
one explicit notion of progress:

- token progress along the request path
- block progress along sealed block windows
- next block boundary derived from `block_idx`, not inferred ad hoc from edge
  length

Insertion and split handling should route through shared helper logic instead
of re-encoding boundary math in multiple branches.

### B. COW contract

At the paged-KV layer, the contract becomes:

1. attached shared prefix pages are readable immediately
2. full shared pages stay shared
3. a partial shared tail page must be detached by COW before append
4. once the tail fills, it becomes an ordinary sealed block eligible for
   publish / demote

### C. Block lifecycle

For one block:

1. request computes or attaches a full block in T0
2. scheduler publishes it to the radix and pins its pages
3. later requests may attach those pages directly
4. if a request needs to append into a shared partial tail, paged-KV clones the
   tail page first
5. under pressure, sealed blocks may demote to T1 / T2
6. staged sealed blocks may promote back into fresh T0 pages and re-enter the
   same publish / attach contract

## Implementation Tranches

### Tranche 1: Radix shape cleanup

Files:

- `infer/src/prefix_cache.rs`

Changes:

- centralize block-boundary math used by split and no-match insertion paths
- remove avoidable `tokens.clone()` usage in lookup / lookup_or_stage / insert
  hot paths
- make the “boundary-on-path, not edge-len” rule explicit in code comments and
  tests
- add shape-validation tests for compressed-radix block-bearing nodes

Acceptance:

- no change to external `RadixCache` API
- existing split regressions remain green
- new tests assert the boundary-on-path invariant

### Tranche 2: Paged-KV COW cleanup

Files:

- `crates/cuda-kernels/src/paged_kv.rs`

Changes:

- make the shared-tail detection / detach path explicit and self-contained
- encode the sealed-block vs hot-tail boundary in helper naming and comments
- keep the tail-page COW path as the only mutable shared-prefix operation
- strengthen tests around attach + shared-tail append + full-page transition

Acceptance:

- direct attach remains unchanged for full shared prefixes
- tail-page COW still happens only for the shared partial-tail case
- no new write path for sealed blocks

### Tranche 3: Scheduler block-path cleanup

Files:

- `infer/src/scheduler/cuda/core.rs`
- `infer/src/scheduler/cuda/runtime.rs`

Changes:

- make publish / attach / promote helpers speak in terms of full sealed blocks
- keep hot-tail state out of radix publish / staged promote paths
- reduce duplicated block metadata update logic where possible
- keep the admission order unchanged

Acceptance:

- direct attach path still uses radix-owned T0 blocks
- staged promote still rebuilds T0-resident full blocks before adopt
- cleanup / spill paths still operate on sealed blocks only

## Failure Modes To Guard Against

1. Publishing a partial tail into the radix.
2. Treating “short edge with block_id” as invalid even though the path is
   block-aligned.
3. Regressing split insertion so `insert_with_fingerprints()` drops tail blocks
   after a shared intermediate.
4. Mutating a shared sealed block without passing through the tail-page COW
   boundary.
5. Introducing a second canonical T0 block registry beyond
   `block_to_pages + RadixCache`.

## Verification

### Local correctness

- `cargo test -p infer --release --no-default-features --features no-cuda prefix_cache -- --nocapture`
- targeted `infer` scheduler tests covering staged-prefix adopt / attach paths
- targeted `cuda-kernels` paged-KV tests on the no-CUDA lane where possible
- `cargo check -p infer --no-default-features --features cuda,no-cuda`

### Required regression tests

- compressed-radix split tests remain green
- direct attach + tail-page COW tests remain green
- publishable-full-block boundary tests remain green

## Benchmark Plan

This change is runtime-visible and therefore requires a bench entry.

Local Mac verification can prove:

- radix correctness
- no-CUDA typecheck
- local unit-test coverage

Remote CUDA still needs the canonical regression run:

```bash
scripts/bench_guidellm.sh kv-cow-radix-block-unification \
  --target http://<remote-host>:8000 \
  --model Qwen/Qwen3-4B \
  --processor models/Qwen3-4B
```

Required workload slices:

1. warm-prefix reuse
2. spill-pressure / staged-readmission pressure

The wins entry should compare against the latest tiered-KV CUDA baseline and
record whether TTFT / ITL stay within noise while preserving the new block/COW
contract.

## Why This Plan Is The Right Scope

This plan intentionally does **not** rewrite the tiered-KV architecture.

It tightens the live design where the current tree already has the right shape:

- compressed radix
- direct T0 attach
- one COW boundary at the shared tail
- staged readmission for lower tiers

The work is therefore:

- architectural enough to remove ambiguity
- local enough to ship without reopening the whole project
- testable on the local lane
- benchmarkable on the remote CUDA lane
