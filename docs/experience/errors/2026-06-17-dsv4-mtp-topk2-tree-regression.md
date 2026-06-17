# DSv4 MTP top-k tree verifier is correct but slower than chain

## Context

Implemented opt-in DSv4 MTP tree verification behind `--mtp-draft-topk K`.
`K=1` keeps the existing chain verifier; `K>1` drafts a complete flattened
top-k tree and verifies every row once. Remote gate used node 61,
`/data01/arle-build`, clean `HEAD=49657c61a8cc198ceb2f97c9a16d66da33287b23`
plus only the top-k/tree patch.

Build and reachability:
- `scripts/dsv4_fast_build.sh` with `FEATURES=cuda,nccl PROFILE=release-fast BIN=arle`
  passed via vendored crates and dsv4 prebuilt CUDA artifacts.
- `strings target/release-fast/arle` contained `dsv4-mtp-tree` and
  `mtp_draft_topk`.
- `target/release-fast/arle serve --help` exposed `--mtp-draft-topk <K>`.

Correctness:
- D2/T1 (`--mtp-draft-tokens 2 --mtp-draft-topk 1`):
  `scripts/dsv4_batched_decode_validate.py 18321` passed
  (`BYTE_PARITY_PASS`, `ANSWER_PASS`, c8 `errs=0`).
- D2/T2 (`--mtp-draft-tokens 2 --mtp-draft-topk 2`):
  `scripts/dsv4_batched_decode_validate.py 18322` passed
  (`BYTE_PARITY_PASS`, `ANSWER_PASS`, c8 `errs=0`).
- Tree path was actually exercised:
  `[dsv4-mtp-tree] depth=2 topk=2 verify_rows=7 draft_nodes=6 ...`.

Perf workload:
- Model: `/data01/models/DeepSeek-V4-Flash`, 8xH20, TP=8.
- `scripts/bench_guidellm.sh`, OpenAI completions, fixed concurrent profile.
- Data: 512 prompt tokens / 128 output tokens.
- Concurrency: `1,2,4,8`; max seconds `30`; warmup `5`.
- Raw artifacts:
  `/data01/arle-build/bench-output/2026-06-17-dsv4-mtp-d2t1-topk1-512x128/`
  and
  `/data01/arle-build/bench-output/2026-06-17-dsv4-mtp-d2t2-topk2-512x128/`.

| c | topk=1 out tok/s | topk=2 out tok/s | delta | topk=1 ITL p50 | topk=2 ITL p50 | completed topk1->topk2 | incomplete topk1->topk2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23.67 | 13.57 | -42.7% | 41.3 ms | 71.7 ms | 6 -> 4 | 0 -> 0 |
| 2 | 23.43 | 14.38 | -38.6% | 86.3 ms | 135.2 ms | 5 -> 3 | 1 -> 1 |
| 4 | 29.73 | 14.22 | -52.2% | 126.6 ms | 221.2 ms | 5 -> 1 | 3 -> 4 |
| 8 | 35.49 | 13.44 | -62.1% | 212.1 ms | 555.3 ms | 8 -> 1 | 6 -> 7 |

Both arms had `errored=0`; high-concurrency rows are completion-count weak
because guidellm's 30s window leaves in-flight requests incomplete, but the
c1/c2 rows are enough to kill the topk2 performance claim.

## Root Cause

The complete D2/T2 tree verifies seven rows per speculative step
(`root + 2 + 4`) versus the chain's three rows (`root + 2`). The extra
candidate coverage can accept longer paths, but the current DSv4 verify row
cost dominates that gain. On this workload, topk2 increases ITL by 57-162%
and lowers output throughput by 39-62%.

## Fix

Keep the feature opt-in only. The default remains `topk=1`, which preserves the
validated chain path and cross-slot batched MTP verifier. Do not flip tree
width on by default unless a future drafter or cheaper verifier reverses this
wall-clock result.

## Rule

Top-k speculative trees need a wall-clock license, not just reachability or
acceptance logs. A correct tree verifier with more accepted paths is still a
regression if verify-row cost dominates end-to-end ITL and output tok/s.
