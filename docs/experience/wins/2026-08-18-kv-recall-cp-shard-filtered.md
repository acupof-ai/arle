# KV-recall × CP — shard-filtered recall under 2D parallelism — CUDA, 2026-08-18

> Status: Shipped

## Goal

Enable `--kv-recall` under TP=2 CP=2 (2D parallelism) on
ThinkingCap-Qwen3.6-27B-FP8. The recall mutex (`attn_cp>1` → reject) blocked
the tiered KV cache in the long-context regime where it matters most.

## Hypothesis

Block-cyclic CP sharding breaks the recall algorithm's 1:1 global-to-local
page mapping. Mapping every page-table access through
`ShardSpec::local_index()` / `global_page()` makes recall rank-local: each
rank scores/evicts/prefetches only its own shard's pages. Partial block reps
(mean over local tokens only) are sufficient for dot-product scoring.

## Parameters

```bash
TEMPLATE=qwen3_nonthink RAW=1 PORT=18189 \
  python3 scripts/needle_gate.py 115,300,446,2000,8000,8192,16384 3
```

- Baseline: `cpkv5` build (pre-`kv_lens` fix), TP=2 CP=2 — CUTLASS crash
- Treatment: `a29d24f5b` (fix), TP=2 CP=2
- Needle: `738291`, depth=0.0 (position 0)
- Trials: 3 per length, 7 lengths = 21 requests

## Environment

- Host / GPU: H20 pod, 8× H20 (sm_90, 97 GB each)
- Driver / CUDA: CUDA 12.x
- Model / dtype: ThinkingCap-Qwen3.6-27B-FP8, BF16 KV pool
- TP / CP: TP=2 × CP=2 = world 4, GPUs 0,1,2,3
- Server flags: `--tensor-parallel-size 2 --context-parallel-size 2 --kv-recall --port 18189`

## Results

| len | baseline (cpkv5) | treatment (a29d24f5b) |
|---:|---|---|
| 115 | server crash | 3/3 exact |
| 300 | — | 3/3 exact |
| 446 | — | 3/3 exact |
| 2000 | — | 3/3 exact |
| 8000 | — | 3/3 exact |
| 8192 | — | 3/3 exact |
| 16384 | — | 3/3 exact |

Treatment: 21/21 exact, zero errors, zero lockstep stalls.
Wall-clock per request: len=2000 0.8s, len=8000 2.9s, len=16384 5.9s.

Recall did not engage under this workload: L1 is 8192 pages × 16 = ~131K
tokens/rank, the largest prompt is 16.7K tokens, and requests are sequential
(each frees its slots before the next arrives). L1 never fills, so no
eviction to L2 occurs. The needle ladder verifies correctness of the
CP-aware recall code paths (page-table sharding, `kv_lens` sizing, `write_kv`
ownership, cross-cp merge) but does not exercise the eviction/prefetch cycle.

Raw artifacts: `/tmp/needle-cpkv6.log` on the pod.

## Problems

**CUTLASS crash on first build (cpkv5).** `for_recall_decode` sets
`kv_lens` to the global sequence length; under 2D the FA3 kernel sizes from
`seqlen_k` and reads past the local shard's page table, tripping the
cross-cp combine kernel (`flash_fwd_combine_launch_template.h:52`).
Fix: override `kv_lens` / `kv_lens_dev` with `local_token_count`
(`(local_pages-1)*page_size + local_last_fill`), matching
`sharded_decode_meta`.

**Build source-digest mismatch.** A running `arle serve` on the pod writes
metrics to `observe-data/observe-*.jsonl` every second; `source_digest`
includes untracked non-gitignored files, so the digest changed between sync
and build. Fix: gitignore `observe-data/` (runtime metrics, not source).

## Learnings

PASS. The 2D decode path's contract is "every page-table field counts the
LOCAL shard, not the global sequence": `kv_lens`, `kv_last_page_len`,
`write_kv`, and the page table itself. `for_recall_decode` was built for the
CP=1 case where local == global; the CP override must touch all four fields.

The recall algorithm's rank-local independence (each rank scores/evicts/
prefetches its own pages) is the right decomposition for block-cyclic
sharding — no cross-rank coordination needed, no collective schedule
divergence.
