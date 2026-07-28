# The qwen35 sidecar's KV round-trip is load-bearing — qwen35 runs its own device allocator

## Context

Warm TTFT on `bench-agent-32k-16x8` was 14.2 s at c=1 while the prefix cache
reported an 87% hit rate. A probe that resends one ~33k prompt to a warm server
isolated the cost: `prefill_tokens` advanced by **9** and TTFT was **8.72 s**,
reproducing to 0.01 s. Sweeping the prefix length gave 1.36 / 2.44 / 4.67 /
8.82 s at 4k / 8k / 16k / 33k — strictly linear, 0.27 ms/token. At 64 KB of
full-attention KV per token that is 245 MB/s against a 20+ GB/s H2D link, so
the cost was host-side copies, not the transfer.

The source was obvious: `Qwen35RecurrentSnapshot` carries the whole matched
prefix's full-attention KV. Save does a full D2H plus serialize at every turn
end; restore does a tier read, a deserialize, `free_slot`, `alloc_tokens` and
an H2D at every turn start.

## Root Cause

I concluded the KV was redundant because `prefix.rs:129` calls
`attach_pages(slot, block_ids, matched_len)` before `restore_prefix_sidecar`,
so the matched pages looked already-attached. Replacing the free/alloc/H2D with
`truncate_slot(slot, boundary)` should then have been a pure deletion.

It measured **worse**: every request full-recomputed
(`prefill_computed = 33065` on every resend) and 33k TTFT went 9.39 → 19.5 s.
The log named it — `cannot grow slot 0 via truncate (33056 > 0)`: the device
pool's slot was empty.

`attach_pages` runs on the engine's **host** pool
(`infer-seam/src/host_paged_kv_pool.rs`), which owns "only logical
slot/page/token accounting". Whether that implies anything about device HBM
depends on the executor:

- **`executor/qwen.rs`** (Qwen-dense) does not run the device pool's allocator
  at all. The host pool is the single allocator and each row's page list is
  lowered via `mirror_slot`, with host page ids indexing device storage rows
  1:1 — so a fresh slot mirroring a published prefix's ids reads that prefix's
  KV straight out of HBM. Zero copies (`paged_kv.rs:868`).
- **`executor/qwen35.rs`** runs the device pool's own `alloc_tokens` /
  `free_slot`. Its page ids are unrelated to the host pool's, and
  `mirror_slot`'s contract forbids mixing the two modes on one pool.

So on qwen35 the radix hit is only a *token* match; the KV bytes behind it are
not reachable by page identity, and the sidecar round-trip is the mechanism
that makes prefix reuse work at all. It is a workaround for a second allocator,
not dead weight.

## Fix

Reverted (`247461a90`). The 8.7 s stands as measured and remains the largest
single term in warm TTFT — c=1 on the long-agent dataset spends 974 s of its
3161 s wall there — but removing it means converging qwen35 onto the
host-allocator + `mirror_slot` model that `qwen.rs` already uses, not deleting
the blob under the current architecture.

## Rule

**"Already attached" is a claim about one specific pool.** Before deleting a
copy because the data "is already there", name the allocator that owns the
destination and prove the source and destination share an id space. Two pools
with the same page-id type and the same `attach_pages` call above them can
still be unrelated address spaces — here one was logical bookkeeping and the
other was device storage, and only one of the two executors bridges them.
