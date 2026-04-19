# Paged-prefill lifecycle audit — why five kernel patches didn't close the bench crash

**Date:** 2026-04-18
**Status:** Active — root-cause investigation, not a patch plan
**Supersedes:** none; complements
[`docs/plans/p99-unified-mixed-batch.md`](p99-unified-mixed-batch.md) §Phase 1C.

## Scope

Today's work stacked five independent kernel-level fixes onto the paged-KV
prefill path (commits `190baf4` / `f08d265` / `927c390` / `7e198a9` /
`96b1a9f`) and each new guidellm sweep against Qwen3-4B with
`prefill_uses_paged_pool() → true` still crashed the scheduler thread
with `gemm_cuda: CUDA_ERROR_UNKNOWN`. The user's feedback was blunt:
stop patching kernels; find the structural bug.

This doc captures what we know and names the invariants the paged path
assumes vs. what the scheduler actually enforces.

## What each of the five fixes addressed

1. **`190baf4` — HD256 `float_workspace` 256→512 MiB + try/catch.**
   FlashInfer's `PrefillPlan` for HD256 at `cta_tile_q=64, padded_batch_size=256`
   carves `tmp_v = 256 MiB` before `tmp_s` is allocated. Real bug; ship.

2. **`f08d265` — HD256 `total_num_rows` device pointer.** When
   `PrefillPlan(...enable_cuda_graph=true)`, the kernel reads
   `params.total_num_rows` as a device pointer into
   `int_workspace[plan_info.total_num_rows_offset]`. We weren't wiring
   it; kernel did OOB Q reads. Real bug; ship.

3. **`927c390` — HD128 same two fixes + workspace sized by call site.**
   Mechanical copy of the HD128 version of both fixes. Real bug; ship.

4. **`7e198a9` — `enable_cuda_graph=false` in PrefillPlan.** Found via
   research: sglang only passes `use_cuda_graph=True` during actual
   CUDA-graph capture. We don't graph-capture prefill, so the flag
   should be false. Keeping it true inflated `padded_batch_size` from
   real-workload (`new_batch_size`, ~64 at 4096 rows) to
   `max(max_batch_size_if_split, total_num_tiles_q)` which ate the
   workspace. Real bug; ship. After this fix, HD128 can go back to
   the 256 MiB default workspace.

5. **`96b1a9f` — `slot_page_indices` trimmed to `num_pages_needed`.**
   `pool.page_indices(slot)` returned the slot's full allocated page
   list, but on a reused slot the tail past `num_pages_needed` held
   stale page IDs now owned by another slot. The HD128/HD256 prep
   kernel's `page_table[pos / page_size]` would read those stale
   entries for some `pos` values. Real bug; ship.

**All five fixes are correct.** They close specific defects in the
kernel path. None of them, alone or together, closes the concurrent
guidellm sweep crash.

## The remaining crash signature

```
thread '<unnamed>' panicked at infer/src/ops/linear.rs:513:14:
gemm_cuda failed: DriverError(CUDA_ERROR_UNKNOWN, "unknown error")
WARN Coordinator thread shutdown failed: coordinator event send failed
```

**Observed repro on `96b1a9f`:**

- Qwen3-4B, `num_slots=10`, `max_seq_len=5120`, `mem_fraction_static=0.88`.
- Server VRAM: 20138 MiB (well under 88% of 23034 MiB budget).
- `RUST_LOG=info` log immediately before panic:
  ```
  Request 9 done: 32 tokens (active=0, waiting=0)
  Scheduler stats: completed=10, generated_tokens=320, active=0, waiting=0
  prefix cache eviction: released 287 pool pages back to free list
    (1004 evicted blocks; retained now 2296)
  [33 seconds idle]
  Request 10 → slot 0 (prompt=4104 tokens, queue=0)
  Request 10: prefix MISS
  Request 10: chunked prefill starting (4104 effective tokens, chunk_size=4096)
  Request 10: prefill chunk 4096/4104 tokens
  [~0 ms later: panic]
  ```
- Direct 10-way concurrent curl with an **identical 4601-token prompt**
  completes cleanly — because the radix cache collapses requests 1-9
  onto request 0's prefix, only one request goes through prefill.
  guidellm sends distinct prompts, so all 10 go through prefill.

So the failure mode is chunk 2 of a `4096 + small_tail` split
(here 8 tokens) **after prior requests 0..9 completed and their
pages were evicted**.

## The architectural invariants that should hold

For the paged prefill path to be correct, these must all be true at
every kernel launch site:

1. **Page-table validity.** `slot_page_indices` on the GPU holds
   exactly `ceil((start_pos + seq_len) / page_size)` valid page IDs,
   where each ID was allocated by `alloc_tokens` for **this slot** and
   has not been released since. `96b1a9f` handles the truncation half
   (too many pages); the allocation side is assumed correct by
   `alloc_pool_tokens_with_retry`.

2. **Page exclusivity at write time.** A page ID in
   `slot_page_indices[i]` must not be simultaneously written by a
   different slot's prefill kernel. The scheduler is single-threaded
   and `ctx.stream` serialises within one stream, so this holds for
   **same-stream** kernel launches — but does every paged kernel
   launch use the same stream? (Open question, see §2 below.)

3. **Post-eviction fence.** When `evict_prefix_cache_for_allocation`
   releases pages back to the free list, no pending kernel on any
   stream may still be writing to those pages. The contract is
   "eviction only touches pages with refcount 0, which means no live
   slot owns them and no in-flight kernel is writing them." Whether
   the scheduler enforces a GPU sync between scheduler steps needs
   verifying.

4. **Chunk-to-chunk invariant.** For a single request's chunk N+1,
   the pool's `seq_len(slot)` is the sum of chunk 0..N token counts
   plus chunk N+1's count (after `alloc_tokens`). `start_pos` passed
   to the forward is `pool.seq_len(slot) - seq_len(chunk N+1)`. All
   K/V writes for chunk N+1 go to pages at positions
   `[start_pos, start_pos + seq_len)`. Pages `[0, start_pos)` are
   **written already** by prior chunks and must not be touched.

5. **Workspace exclusivity.** `BatchPrefillPagedPlan` owns one
   FlashInfer workspace; it's wrapped in `Mutex<Option<...>>` at the
   model level. One paged-prefill call at a time on a given model
   instance. That's fine for a single-threaded scheduler.

## What we haven't yet verified

### (a) Does `ctx.stream` stay the same across scheduler steps?

All model forwards go through `ctx.stream`. If cudarc's
`DeviceContext::stream` is the one-and-only stream, then kernel
launches across scheduler steps serialise naturally. If it's a stream
per call (unlikely given cudarc API) or the pool uses a different
stream for page bookkeeping, §2 / §3 break.

**Verify:** `grep "stream" infer/src/backend/cuda/tensor.rs` and
`crates/cuda-kernels/src/paged_kv.rs` — is there exactly one
stream?

### (b) Does `evict_prefix_cache_for_allocation` ever release a page currently in a slot's `page_indices`?

The eviction path decrements radix refcounts on pages that were
"in limbo" (live in pool but not owned by a slot). The invariant is
that a page ID can be in *either* a slot's `page_indices` *or* the
radix retained set, not both. Is that invariant actually enforced in
`paged_kv.rs` / `radix_cache.rs`?

**Verify:** walk through `free_slot` → `release_pages` → eviction —
does any path put a page back on the free list while a slot still
references it?

### (c) What happens to the paged pool when chunk 1 completes but chunk 2 hasn't started?

Between chunks, the scheduler state is:
- pool.seq_len(slot) = chunk_1_tokens (say 4096)
- slot's `phase = Prefilling { progress: 4096, total: 4104 }`
- Request isn't in waiting; it's still active.

Could another concurrent admission run in this window? A new request
arrives, scheduler calls `alloc_pool_tokens_with_retry(new_slot, ...)`,
pool is full, eviction runs. Does the eviction look at "retained
pages" or at "pages in the free list"? If it walks retained pages and
those include pages belonging to the SLOT THAT IS MID-PREFILL... that
would poison chunk 2.

**Verify:** inspect
`Scheduler::evict_prefix_cache_for_allocation` and
`RadixCache::evict` — are mid-prefill slots protected?

### (d) Chunk 2 launches read/write the same pool region as chunk 1

For a 4096+8 split, chunk 1 writes pages 0-255 (all of them full),
chunk 2 writes page 256 (tokens 4096-4103, last_page_len=8). The
paged-prefill FlashInfer call for chunk 2 reads the entire 257-page
KV span for attention. If any of pages 0-255 got reused between
chunks for another slot's allocation, that's the corruption.

**Verify:** does `alloc_tokens` for slot B ever return a page that's
currently listed in slot A's `page_indices`? The pool's free list
should prevent this, but trace the actual code path.

## Hypothesized root cause (to be proven by reading, not patching)

**The eviction path runs when the pool is full, and at that moment
may release a retained page that happens to ALSO be listed in an
active slot's `page_indices`.** This can happen if:

- Slot X holds pages `P1..P256` for request R.
- R completes; scheduler calls `retain_pages(&P1..P256)` to publish
  the prefix to radix, bumping refcount on each.
- R's `free_slot` decrements refcount on pages with radix retention
  back to whatever radix holds (if radix held 1 ref and slot held 1
  ref, both go to 0 for pages with only one radix holder).
- A new request comes in, allocates pages from the free list — may
  get some of `P1..P256` back.
- Meanwhile slot X is still listed with `page_indices = [P1..P256]`
  because `free_slot` didn't clear the slot's indices array for the
  "retained" case.

This is a hypothesis about the bookkeeping in `paged_kv.rs`
`free_slot` semantics. It needs to be checked against the actual
code — the point of this doc is to frame the check, not jump to
a patch.

## Plan

1. **Leave Qwen3 + Qwen3.5 `prefill_uses_paged_pool() = false`** —
   revert the Phase 3a flag flip. (Done in this commit.) Keep the
   five kernel-level fixes because they are correct on their own.

2. **Read `paged_kv.rs` end-to-end** (max ~400 lines) and write
   down the precise contract for when a page ID moves between:
   - a slot's `page_indices`
   - the free list
   - retained (radix-held, refcount>0, not in any slot)

3. **Read `scheduler/cuda/core.rs::evict_prefix_cache_for_allocation`
   and `RadixCache::evict`** and check whether any path violates that
   contract.

4. Once the contract violation is named (or proven absent), fix the
   structural issue. Then retest the full guidellm sweep. Kernel
   patches stop; scheduler fix drops in.

## Rule

Five kernel-level patches in a row that don't close the bug mean the
bug isn't in the kernels. Audit the control flow. This doc is the
switch-to-audit deliverable; no further code changes on the paged
path until the audit proves a specific contract violation.
