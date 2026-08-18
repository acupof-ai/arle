# --kv-recall loses mid-context content — TP=2 CP=2, 2026-08-18

## Context

Verifying that L2 (host DRAM) and L3 (SSD) KV tiers work under TP=2 CP=2 with
`--kv-recall`, on ThinkingCap-Qwen3.6-27B-FP8, build `cpkv7` (`2ae47a16`; the CP
recall sources are byte-identical to HEAD `97433f9e8`).

The tier plumbing works. What does not work is retrieval.

## Phenomenon

Needle position decides the outcome, at 16K prompt tokens, 16 concurrent rows,
a distinct needle per row:

| Needle depth | Region | Recall ON | Recall OFF |
|---|---|---|---|
| ~0% | sink (`n_init` = 32 tok) | 48/48 exact | 48/48 exact |
| 50% | middle | **0/48 exact** | 48/48 exact |
| 99% | local window (`n_local` = 256 tok) | 4/4 exact | — |

Same binary, same `--mem-fraction-static`, same `--chunked-prefill-size`; the
only difference between the arms is the `--kv-recall` flag. The recall arm
answers with invented codes (`1024`, `1234567890`), never another row's needle —
this is content loss, not cross-row contamination.

The 50% failure reproduces on both configurations tested:

- L1 = 52885 pages (`mem_fraction_static` 0.9), no chunked prefill, L2 only
- L1 = 4112 pages (`mem_fraction_static` 0.25), chunk 2048, L2 1 GiB/rank + L3

So it is independent of L1 pressure, of chunked prefill, and of L3.

## Cause

`prefill_row_recall` is "the ONLY place the whole recall cycle runs: decode never
recalls, prefetch happens only here". The working set is therefore chosen once,
at the last prefill row's tail, from the layer-0 query of the final prompt token,
scored against per-block mean-key reps. Everything outside sink ∪ top-k ∪ local
is evict-dropped, and decode never brings any of it back.

Two observations confirm nothing is ever restored:

- `arle_kv_tier_io_useful_read_bytes_total` = 0 and
  `arle_kv_system_reuse_hit_{host_demoted,disk}_total` = 0 after every run,
  while `useful_write_bytes` reached 5.4 GB. Writes only, no reads.
- Adding `--chunked-prefill-size 2048` (8 recall cycles per 16K request instead
  of 1) did not produce a single read either.

Under CP there is a second, unseparated defect in the same scoring path.
`CudaRecallState::block_reps` is documented as "under CP, each rep covers only
this shard's local tokens (partial mean)", and each rank runs `plan_recall`
independently. Ranks can therefore select different top-k blocks, leaving every
selected block half-resident — rank 0 holding one block's even pages while rank 1
holds another block's odd pages.

**Not yet separated**: whether depth-50 fails identically at CP=1/TP=1. Without
that control, the weak layer-0 retriever and the CP partial-mean divergence are
both live candidates, and the CP rule below is stated as a design fact, not as
the measured cause.

## What did work

Tier mechanics under CP are correct and measured:

- L3 stores are created per rank —
  `arle-kv-recall-st-<epoch>-format-1-world-4-rank-{0,1,2,3}-page-524288` — so
  local page indices cannot collide across ranks.
- Spill reached 1.2 GB on disk per rank × 4 ranks; the four ranks wrote
  near-identical volumes, consistent with block-cyclic sharding.
- L1 residency during decode: **18 local pages held for a 34,926-token context**,
  against 1092 for the full context. The 544-token budget is 17 local pages under
  CP=2; measured 18. Eviction engages exactly as designed.
- Front-needle `needle_concurrent.py` passes 48/48 under recall at 16 concurrent
  × 16K, including with L1 shrunk to 2× under the offered working set.

## Rule

A front-of-prompt needle cannot gate `--kv-recall`. Position 0 lands inside the
pinned sink window, so the gate passes without the recall cycle ever retrieving
anything — the existing `needle_concurrent.py` gave 48/48 on a configuration
that loses 100% of mid-context content. Gate recall with the needle in the
middle: `needle_concurrent.py <port> <conc> <tokens> <rounds> 50`.

Second: a tier that only ever writes is not a tier. Read counters
(`useful_read_bytes`, `reuse_hit_*`) at exactly 0 while write counters climb is
the signature of a one-way evict path, and it is visible without any
correctness test.

## Follow-up

1. CP=1/TP=1 depth-50 control — separates retriever quality from CP shard
   divergence. One GPU, ~5 min.
2. If CP-specific: make block scoring shard-collective (all-reduce the partial
   mean-key reps before `plan_recall`) so every rank selects the same top-k.
3. If generic: the working set must be re-planned during decode, or selection
   must use something stronger than a layer-0 mean-key dot product.

Until (1) lands, `--kv-recall` should not be treated as usable for retrieval
workloads on any parallelism, CP included.

## Method note

`/metrics` under multiproc returns a zeroed snapshot while the engine thread is
busy — `kv_free_pages` and `active_requests` both read 0 mid-request, and the
scrape itself blocks ~2.3 s. Sampled gauges are only trustworthy from the arm
whose decode steps are fast enough to keep the relay fed; a 0 is not a reading.
Counters read before/after a run are unaffected.
