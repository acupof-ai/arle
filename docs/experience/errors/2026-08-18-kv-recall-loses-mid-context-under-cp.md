# --kv-recall: mid-context loss under CP, FA3 abort without it — 2026-08-18

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

The CP=1/TP=1 control cannot answer the content question, because recall aborts
the server there — see the second bug below. So the weak layer-0 retriever and
the CP partial-mean divergence both remain live candidates for the content loss,
and the CP paragraph above is a design fact, not a measured cause.

## Second bug: FA3 combine abort without CP

At TP=1 CP=1, `--kv-recall` aborts the process on the first decode step after a
19,312-token prefill: `CUTLASS error
(flash-attention/hopper/flash_fwd_combine_launch_template.h:52): Error
Internal`. Deterministic at concurrency 2 and 8. The recall-OFF control on the
same GPU and binary passes depth-50 at both concurrencies, so it is
recall-specific, not TP=1-specific.

`PageMeta::for_recall_decode` had a branch on `shard.size`:

```rust
let (kv_lens, last_page_len, write_kv) = if shard.size > 1 {
    (local_token_count, local_last_fill, i32::from(owns_last))
} else {
    let g = total_len % pool.page_size;
    (total_len, if g == 0 { pool.page_size } else { g }, 1)  // full context
};
```

Without CP it passed `total_len` as `seqlen_k` while handing FA3 a page table
holding only the recall working set (34 pages / 544 tokens against 19,312).
FA3 sizes its split-KV work from `seqlen_k`, so the combine kernel indexed
splits with no pages behind them. The CP branch already derived the length from
the table — which is exactly why CP=2 survives and TP=1 aborts.

Fix (`1bf969aa9`): a 1-wide shard owns every page, so both branches collapse to
the CP arithmetic. Byte-identical whenever the table is the full contiguous page
list (`num_pages == global_pages`), i.e. every non-recall decode. **Verification
build pending-remote** — the pod builder was occupied by concurrent W4AFP8 work.

Result matrix, depth-50 needle, 16K prompt, same `cpkv7` binary throughout:

| config | recall | result |
|---|---|---|
| TP=1 CP=1 | off | PASS 4/4 and 16/16 |
| TP=1 CP=1 | on | abort, `flash_fwd_combine` |
| TP=2 CP=2 | off | PASS 48/48 |
| TP=2 CP=2 | on | 0/48 exact, no abort |

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

1. Build `1bf969aa9` on the pod and re-run depth-50 at TP=1 CP=1. With the abort
   gone, that run separates retriever quality from CP shard divergence: a TP=1
   pass means the content loss is CP-specific; a TP=1 fail means the selector
   itself is too weak.
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
