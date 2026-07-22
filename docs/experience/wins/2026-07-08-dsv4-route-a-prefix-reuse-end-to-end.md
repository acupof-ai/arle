# DSv4 Route A prefix reuse — end-to-end verified, 69% latency win on cache hit

**Date:** 2026-07-08/09. **Backend:** CUDA, DeepSeek-V4-Flash-FP8, TP=4, 4×H20
(GPUs 4-7). **Commits:** `6a78a490d`..`0198c3ba7` (full chain below).
**Scope:** `crates/infer-cuda/{attention.rs,executor.rs,dsv4.rs,attention/
{kv_layout.rs,dsa.rs},kv_tier.rs}`, `crates/infer-core/{planner.rs,lib.rs}`,
`crates/infer-seam/{lib.rs,host_paged_kv_pool.rs}`.

> Superseded 2026-07-10: the Route A compressor/ring/page-sharing mechanism measured here was deleted after a FlashMLA-lane correctness regression. Preserve the 69% historical measurement; do not treat the listed pools, restore path, or file sites as current architecture. Current reuse uses the prefix-state/finish-frontier path.

## Goal

Ship Route A (page-granular DSv4 prefix reuse,
`docs/plans/2026-07-08-dsv4-route-a-page-granular-prefix-reuse.md`) steps
4-6 and get cross-request prefix reuse working end-to-end for the first
time — DSv4 had zero reuse since Route B's deletion earlier the same day.

## What shipped, in order (each fix exposed the next)

1. `3ebc763f9` — FlashMLA per-layer KV budget: sum real per-layer need, not
   `.max()` + uniform-divide (prerequisite, step 2).
2. `6481d9c2d` — delete Route B (whole-slot snapshot/restore), scope
   corrected mid-execution (step 3).
3. `6a78a490d` — Step 4: `Dsv4CompressStatePool`, page-addressable
   compressor state for `compress_ratio==4` layers, new `overlap_page_stride`
   kernel param.
4. `95a2fab94` — Step 5: `dsa_official` write-mirrored shadow pool +
   `Dsv4SwRingSnapshotPool` (periodic full-ring snapshot at
   `sliding_window`-boundaries).
5. `d2482c7c6` — Step 6: `CudaKvTierStore` wiring for the compressor pool,
   `reusable_prefix_blocks` made real for DSv4 (was hardcoded `0`).
6. `c042a47fb` — fix: `HostPagedKvPool::attach_pages` tops up a fixed-band
   slot to full length instead of leaving it `matched_len`-short (crash #1).
7. `f317a7e27` — fix: force DSv4 prefill chunks onto `sliding_window`
   boundaries (LCM of page_size and a new `prefill_restore_boundary_alignment`
   seam method) — the ring only snapshots at a call's own end position, so a
   single-call whole-prompt prefill never visited an earlier boundary.
8. `4e9acf47f` — fix: the 2 pure-SlidingWindow layers never called
   `snapshot_sw_ring_at_boundary` in any of their 3 forward sub-paths,
   permanently vetoing the AND-gate over all 43 layers (crash #2 exposed by
   #7 actually clearing the scheduling gap).
9. `fc743af44`/`55a74d870` — fix: `Dsv4SlotState.seq_len` never updated on a
   partial-prefix-hit slot resume, crashing the tail prefill's contiguity
   check (crash #3, exposed by #8 actually reaching a slot resume for the
   first time); `55a74d870` fixed an unrelated build break from a `git add`
   race against a concurrent session sharing the tree.
10. `0198c3ba7` — fix: the same stale-counter bug one layer deeper — per-layer
    `Dsv4CompressorState.compressed.seq_len` (main + indexer) and
    `Dsv4DsaOfficialState.packed_rows` never reset on restore, crashing
    `csa_select_official`'s monotonicity invariant (crash #4).

Five bugs, each real, each exposed only once the previous fix cleared the
path far enough to reach it — the code path had never been exercised since
Route B's deletion, so none of these were previously reachable to find.

## Results — round 5 (`0198c3ba7`), TP=4 GPUs 4-7

| Check | Result |
|---|---|
| Build | PASS, `BUILD_EXIT=0` |
| Crash repro (both accidental + engineered overlap shapes) | PASS, zero panics |
| `needle_gate.py` 512/2048/3800×3 | PASS, 9/9 needle retrieved, zero misses (first clean completion all day) |
| Reuse fires (`/metrics`) | PASS — `hit_tokens_total` +1152, `hit_pages_total` +18 on a repeat request |
| Intermediate boundary granularity | PASS — partial match (`hit_tokens_delta=128`) at the first shared boundary, not 0 or full |
| **Net perf: cache-hit vs cache-miss, same 1587-token prompt** | **hit_mean=0.636s, miss_mean=2.048s — 69.0% speedup** |
| Stress loop, 10 mixed shared/disjoint requests | PASS, 10/10 OK, server alive |

Cumulative session `/metrics`: `prefix_cache_lookups_total=52, hits_total=33,
hit_tokens_total=29184, hit_pages_total=456` — reuse fired continuously and
correctly across the whole stress run.

## Problems

- **Ratio=1 (SparseIndexed/GLM-5.2) path untested** — `dsa_index_ratio()`'s
  ratio=1 branch is exclusively a GLM-5.2 `per_layer_attention_mode`
  override; no GLM-5.2 checkpoint is present on the verification pod. Flagged
  as an explicit gap, not fabricated as a pass. Needs its own verification
  pass once a GLM-5.2 checkpoint is available.
- Real GPU-memory eviction for the compress-state pool (shrinking below
  `capacity_blocks`) still needs a `.cu` kernel change (device-side
  page-table gather or new kernel parameter) — not built; the pool is
  GPU-resident-only today, tier-store wiring provides bookkeeping +
  restart-survival groundwork but no HBM footprint reduction yet.
- Multi-rank (TP≥2) consistency for the compressor pool was separately
  confirmed via rank-diff MD5 hashing (bit-identical across 4 ranks,
  `compressor_forward`'s hidden-state input is fully replicated, no
  TP-sharding) — no relay/broadcast mechanism needed. Not yet re-confirmed
  for the ring/dsa_official pools specifically, though the same replication
  argument should apply.

## Learnings

- **A code path unreachable since a prior deletion can hide an arbitrary
  number of stale-state bugs, discovered one at a time as each fix clears
  the next gate.** Don't be alarmed by a chain of "fix → verify → find the
  next bug" — that's the expected shape of hardening a genuinely
  never-exercised path, not scope creep. Budget for it.
- **"No crash" is not a pass.** Every round in this chain that stopped at
  "no crash observed" without checking `/metrics` counters missed that reuse
  had not actually fired yet (rounds where `hits_total` stayed at 0 despite
  a clean-looking response). The actual pass/fail signal for a reuse feature
  is the reuse counter, not the absence of an error.
- **A coarse per-request counter and its finer per-layer siblings are
  separate bugs, not one bug.** `fc743af44` fixed `Dsv4SlotState.seq_len`;
  fixing it did not fix the analogous `Dsv4CompressorState.compressed.
  seq_len`/`Dsv4DsaOfficialState.packed_rows` one layer down — every
  "restore physical bytes on resume" code path needs its own enumeration of
  which position counters it must also reset, not just the most visible one.
- **A shared dev tree with a concurrent session can silently reintroduce
  scope creep or race a `git add`** — confirmed twice today (a subagent's
  diff picking up unrelated Qwen3.5 comment-stripping mid-edit; a `git add`
  capturing a concurrent session's half-written struct field). Diff-review
  before every commit in a shared tree, not just before delegating.
