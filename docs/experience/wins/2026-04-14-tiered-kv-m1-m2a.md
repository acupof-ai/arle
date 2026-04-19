# 2026-04-14 · Tiered KV Cache M1 + M2a landed — radix is now the pool's pinning authority

## Context

Tiered KV Cache (project doc
[`../../projects/tiered-kv-cache.md`](../../projects/tiered-kv-cache.md),
task doc
[`../../plans/tiered-kv-cache-tasks.md`](../../plans/tiered-kv-cache-tasks.md))
carves the global KV storage hierarchy into milestones M0–M5. The 2026-04-15
revision §0.5 retired the old P0–P5 labels and made M0.1 / M0.2 / M1 / M2
first-class prerequisites for the real cross-slot prefix reuse win.

Going into this session, M0.1 and M0.2 had already landed upstream; M1 and
M2 were still pending, with M0.3 (the `page_size = 16` lift) blocked on an
in-flight Codex CUDA kernel extraction stabilising and its own rescoped P0
blocker in [`../errors/2026-04-14-p0-page16-blocker.md`](../errors/2026-04-14-p0-page16-blocker.md).

This session pushed **M1a → M1b → M2a** to `main` as three atomic commits.
M0.3 stayed blocked; M2b through M5 remain ahead.

## What landed

| Commit | Milestone | Scope | Delta |
|---|---|---|---|
| `08718ad` | M1a | Delete `infer/src/kv_tier/directory.rs` (322 lines) + update `kv_tier.rs` module docs + retire the one `transport/disk.rs` doc comment referencing it. Zero production callers — grep for `TierDirectory` / `BlockDescriptor` / `DirectoryError` outside the file itself returned only module docs and the 7 `#[test]` functions that lived in the deleted file. | −346 / +28 |
| `323aee0` | M1b | Wire `RadixCache` into `Scheduler<M>` as a **shadow observer**. New fields `prefix_cache: RadixCache` (block_size=16, owned outright since the scheduler runs on a single `std::thread`) + `next_prefix_block_id: u32` (monotonic synthetic id counter). `cleanup` inserts the completed prompt into the radix with fresh synthetic `BlockId`s; `assign_slots` runs `radix.lookup` before the legacy linear scan and logs `"radix shadow: best cross-slot prefix hit = X/Y tokens"` before releasing the refs. Two new prefix_cache tests mirror the exact usage shape (insert → lookup → release → evict roundtrip + partial-block-drop edge case). Behavior unchanged — `cached_prompts` still drives actual KV reuse. | +214 / −1 |
| `4402ab0` | M2a | Upgrade M1b from synthetic ids to **real physical pool page indices** and teach `TokenKVPool` to honor an external refcount on free. `TokenKVPool` gains `page_ref_count: Vec<u32>` + `retain_pages` / `release_pages` / `retained_count` + a refcount-aware `free_slot` that leaves pinned pages in limbo. `Scheduler::publish_to_prefix_cache` now takes a `slot_idx`, snapshots `token_indices(slot_idx)`, inserts real page ids into the radix, and calls `retain_pages` across the full block-wide span. New side map `block_to_pages: HashMap<BlockId, Vec<u32>>` records each block's complete (non-contiguous after a few alloc/free cycles) page span. New `evict_prefix_cache_if_pressured` helper with `PREFIX_CACHE_HIGH_WATER = 0.75` / `LOW_WATER = 0.50` watermarks runs at the end of `cleanup` and releases radix-held pages back to the pool when retained fraction crosses the high mark. Four new refcount unit tests via a `MockPool` mirror cover retain/release/free_slot/multi-refcount. | +507 / −54 |

The three commits compose cleanly:

1. **M1a** removes dead code so the next commit does not accumulate a
   "retire unused module" diff on top of real behavior changes.
2. **M1b** installs the infrastructure (radix exists, is populated, is
   queried) in a behavior-neutral way so regressions are easy to attribute
   to later behavior flips.
3. **M2a** is the **first behavior change**: pool pages now survive
   `free_slot` when the radix retains them, and the radix carries real
   pool data instead of synthetic ids.

## What is actually wired end-to-end

- **Happy path**: a request finishes → `cleanup` reads
  `paged_kv_pool.token_indices(slot)` → inserts the prompt into
  `prefix_cache` with those page ids as `BlockId`s → calls
  `paged_kv_pool.retain_pages` across the full `num_blocks × block_size`
  span → `paged_kv_pool.free_slot(slot)` runs but pages with refcount > 0
  stay in limbo (not in any slot, not in `free_slots`, still live in HBM).
- **Watermark eviction**: at the end of `cleanup`, if
  `retained_count / max_total_tokens > 0.75`,
  `evict_prefix_cache_if_pressured` calls
  `RadixCache::evict(blocks_to_evict)`, looks up each evicted `BlockId` in
  `block_to_pages`, removes the entry, and calls
  `paged_kv_pool.release_pages` on the full span. Pages whose refcount
  drops to zero rejoin the primary free list; pages still referenced by
  another radix block stay pinned. Hysteresis between high and low marks
  (0.75 → 0.50) prevents thrash.
- **Shadow lookup**: `assign_slots` still runs `radix.lookup` for every
  admission and logs the best cross-slot match length, then releases the
  refs immediately. Slot selection still goes through the M1-era
  `best_prefix_slot_for_cached_prompts` linear scan over
  `cached_prompts`. M2a changes the pool's bookkeeping but not the
  selector.

## What is still **not** wired

M2a is explicitly the "data model + refcount discipline" half of M2. The
**scheduler side of dual residency** — actually using the radix's pinned
pages to short-circuit prefill on cross-slot prefix matches — is M2b and
did not land this session. Concrete gaps:

1. `assign_slots` does not yet prefer slots whose radix hit is non-empty.
   The radix lookup runs but only the log line sees the result.
2. `step_new` does not yet have a "resurrect matched prefix" path. The
   pieces are there: `block_to_pages` maps each matched `BlockId` to its
   full page span, `paged_kv_pool.retain_pages` lets that span live
   across slot boundaries. But the actual read path (copy matched pages
   into the new slot's `token_indices` *or* share them directly across
   slots via an aliased entry, then advance the per-slot generation
   state to `matched_len` without running `forward_prefill` over the
   prefix) is unimplemented. This is the change that will produce a
   real TTFT / decode tok/s improvement on repeated-prompt workloads;
   M2a alone is net-neutral on single-request benches.
3. `alloc_tokens` still returns `Err` on OOM with no retry-after-evict.
   The amortised cleanup-time eviction holds under steady-state load
   but an adversarial burst of long-prompt admissions can drain the
   free list between two `cleanup` calls and bounce requests. M2b adds
   the synchronous retry.
4. `cached_prompts: Vec<Vec<u32>>` still lives on `Scheduler<M>` as the
   legacy ground truth. It is only retired once the radix owns slot
   selection (M2b).

## Why the shadow observer still matters

Even before M2b lands, the M1b+M2a combo gives operators three things
they did not have before:

1. **Observability**: scheduler logs now carry a `"radix shadow: best
   cross-slot prefix hit = X/Y tokens"` line on every admission. This
   turns into a "could-have-saved" metric that quantifies exactly how
   much prefill work the legacy per-slot selector is leaving on the
   table. When M2b flips the selector, that metric becomes "did-save"
   with the same formula.
2. **Bounded growth**: the watermark eviction + refcount-aware
   `free_slot` mean the radix cannot grow unboundedly and cannot leak
   pool pages. Under M1b's synthetic-id regime, eviction was
   academic — no pool resource was actually held. Under M2a, eviction
   directly controls T0 HBM occupancy.
3. **M2b blast radius**: because M2a already set `block_to_pages` +
   real page ids + `retain_pages`/`release_pages`, M2b reduces to
   "teach the selector + resurrect" — two focused touches, not a
   simultaneous data-model + behavior change.

## Verification

All three commits built clean on the L4 box (`CUDA 13.0`,
`driver 580.82.07`, `SM 89`) and passed their respective test gates:

- `cargo test -p cuda-kernels --features cuda --lib` → 31 passed,
  0 failed (27 pre-existing + 4 new `MockPool`-based refcount tests:
  `retain_then_free_slot_keeps_page_in_limbo`,
  `release_after_free_slot_reclaims_limbo_pages`,
  `retain_release_without_free_slot_does_not_move_pages`,
  `double_retain_needs_double_release_to_free`).
- `cargo test -p infer --release --lib` → 240 passed, 0 failed, 11
  ignored. All 18 `prefix_cache::tests::*` including the two new
  M1b `scheduler_shadow_observer_*` tests stay green after the
  synthetic-id → real-id swap.
- `cargo build -p infer --release` → clean 17.70s incremental build
  on M2a (one failed attempt first with E0502 — borrow conflict on
  `slot_pages` held immutable across the `retain_pages` mut call;
  fixed by cloning the required span into an owned `Vec` up-front).
- No e2e or bench re-runs this session — M2a is a pure-book-keeping
  change with no hot-path impact on the single-request path that
  `bench_serving request` exercises. The regression gate is
  specifically the four refcount unit tests plus the existing
  scheduler_shadow_observer tests. An e2e re-run will happen with M2b
  when the selector flip makes behavior observable.

## Rule

**When the data model change and the behavior flip are separable,
land the data model first as an atomic commit, even if it looks
purely structural and has no measurable impact on its own.** M1b and
M2a each look "neutral" in isolation but together they reduce M2b's
scope from "rewrite the scheduler's slot selection + invent a side
map + add refcount + audit every free path + survive the pool OOM
retry" down to "change three calls in `assign_slots` and add one
path in `step_new`". Bundling those steps would have been a 12-file
diff with no atomic rollback point; splitting gives two atomic
rollback points and keeps the blast radius of the behavior flip
small.

**Corollary**: when the data model already has well-defined
invariants (here: `page_ref_count > 0 ⟺ in a slot or in limbo, never
in free_slots`), the unit tests covering those invariants
(`MockPool` suite) become the regression gate for everything that
follows. M2b can be validated against the same tests — no
scheduler mock required.

## Next steps

Ordered by dependency + cost:

1. **M2b — scheduler flip** (next target): replace
   `best_prefix_slot_for_cached_prompts` with a radix-aware selector
   + add the resurrection read path in `step_new` that copies matched
   pages' state into the new slot and skips prefill for the matched
   prefix. Also wire `alloc_tokens` OOM → `evict_prefix_cache_if_pressured`
   retry. Estimated 5-7 files, builds on M2a's side map + refcount
   API exactly as written. Validation: run
   `bench_serving matrix --concurrency 4 --prompt-lens 128,512
   --output-lens 64` and expect measurable TTFT win on repeated
   prompts.
2. **M0.3 — `page_size = 16`** (parallel track, blocked on Codex):
   still gated on the Codex CUDA kernel wave stabilising + the rescoped
   P0 list in
   [`../errors/2026-04-14-p0-page16-blocker.md`](../errors/2026-04-14-p0-page16-blocker.md)
   (new BF16 HND range kernel, `migrate_kv_range_to_paged` API change,
   `stride_page * page_size` fix at both `decode_prep_paged*` call
   sites — last one already landed in upstream `96b1489`).
3. **M3 — T1 host pinned tier + coordinator**: waits on M2b. Needs the
   OS-thread coordinator owning two CUDA streams + layer-granular
   `cudaEvent` sync, per task doc §3 course corrections.
4. **M4 — T2 disk tier + session save/load + MLX wired memory**: mostly
   local-Mac work, waits on M3 for the coordinator shape.
5. **M5 — NIXL**: stub already shipped upstream; real RDMA deferred
   until a trigger (prefill/decode disaggregation, cross-node session
   roaming, or a second kernel-crate consumer) fires.
