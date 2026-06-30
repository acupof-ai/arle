# DSv4 fixed-band page-attn and TP-safe slot snapshot store

## Context

DSv4 could not treat the KV tier as a serialized whole-slot snapshot forever. The
engine/radix/tier layer owns page identity, while FlashMLA and DSA read their own
fixed-band device layouts. The missing connection was a backend-owned page
metadata path that lets DSv4 lower the host slot page table into FlashMLA without
pretending DSv4 is a sequential paged cache.

## What Worked

- `HostPagedKvPool` now has fixed-band allocation semantics: a DSv4 slot draws the
  complete FlashMLA logical band once, and `truncate_slot` only rewinds the logical
  cursor. It does not release tail band pages after MTP reject or prefix restore.
- `KvBatchDescriptor` carries both live prefix pages (`flat_page_ids`) and the
  complete slot page table (`flat_slot_page_ids`). Existing models keep using the
  live prefix table; DSv4 consumes the full slot table.
- `Dsv4KvAdapter::prepare_kv_batch` mirrors the host slot page table into each
  layer's `TokenKVPool` with `mirror_band`, then advances the FlashMLA cursor.
- Whole-slot and position-0 prefix restore receive the host slot page table from
  `infer-core`, mirror it first, then restore `Dsv4SlotSnapshot` payloads into those
  physical pages.
- TP is supported by deterministic rank-local storage: every rank stores/restores
  its own shard under the same engine key. Hit length, demote room, snapshot fit,
  insert, read/parse/restore success all go through TP min-reduce, so any rank miss
  or failure makes every rank take the same branch.

## Verification

Local gates:

```
cargo test -p infer-seam --release --lib
cargo test -p infer-core --release --lib
CUDARC_CUDA_VERSION=12090 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib
```

Results:

- `infer-seam`: 34 passed.
- `infer-core`: 83 passed, 1 ignored benchmark.
- `infer-api` CUDA/no-cuda typecheck: passed.

`cargo test -p infer-cuda --features cuda,no-cuda --lib ...` reaches the linker
on macOS and fails on missing CUDA symbols, which is the existing local no-cuda
test limitation. The CUDA crate remains covered locally by the `infer-api`
typecheck; runtime validation must run in the H20 pod.

## Pending Remote

Run TP=4/EP=4 on the sglang container with L1/L2/L3 enabled and record:

- repeated identical prompt to prove DSv4 position-0 prefix hit across all ranks,
- slot oversubscription or forced preemption to prove whole-slot demote/promote,
- `/v1/stats` prefix/tier counters,
- decode phase timing before/after page-table routing.

## Rule

For DSv4 TP, never let a single rank decide prefix or slot-tier reuse. Store bytes
rank-locally, but reduce the decision to a scalar consensus before the scheduler
branches. Page identity must come from the host slot table, and DSA remains a
fixed-band sidecar until it is page-addressable at arbitrary radix boundaries.
