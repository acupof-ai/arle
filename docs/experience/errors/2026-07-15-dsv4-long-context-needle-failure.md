# DSv4 long-context needle retrieval failure

> Status: Active

## Goal

Verify long-context retrieval before measuring TP=4 throughput.

## Hypothesis

The built-in greedy decode path should retrieve a six-digit needle throughout
the supported 8,448-token prompt window.

## Parameters

- Endpoint: `/v1/chat/completions`, greedy, needle `738291`.
- Approximate target lengths: 3,000, 6,000, and 8,000 tokens; needle depth 50%.
- Three same-config runs per length; output caps 512 and 1,024.
- Additional 8,000-token depth probes at 0% and 90%.
- L2/L3 and speculative decode off, so every request performs full prefill.

## Environment

- Commit `5d53f40e7`; binary SHA256
  `07aba935be745c52d95b51d5c3235ba200061af07a03c9d37943488869af24c8`.
- DeepSeek-V4-Flash-FP8, 4x H20, TP=4 on GPUs 2/3/4/5, driver 535.161.08.
- NCCL allreduce; `max_prompt_tokens=8448`, `max_total_tokens=9216`.
- Local NVMe checkpoint; `ARLE_LOADER_PREFETCH=0`; `--kv-dram off`.

## Results

| actual prompt tokens | depth | output cap | exact | observed failure |
|---:|---:|---:|---:|---|
| 2,720 | 50% | 512 | 2/3 | one empty content |
| 5,424 | 50% | 512 | 2/3 | `738292` |
| 7,222 | 50% | 1,024 | 0/3 | `738` in reasoning and content, all `stop` |
| 7,222 | 0% | 1,024 | 0/1 | `738102` |
| 7,222 | 90% | 1,024 | 1/1 | `738291` |

Service counters after 33 requests: 177,124 prefill tokens, 8,597 generated
tokens, 33 completed, zero prefix hits, zero active or queued requests, and no
HTTP, OOM, NCCL, or fallback error. The 7,222-token 50% cases completed normally
in 23.0-23.7 seconds; this is retrieval failure, not truncation.

## Problems

The original 32/128/256-token output caps were invalid for this thinking model:
reasoning exhausted the budget before content. Those runs do not count. The
first server launch was also discarded because an OPD service concurrently
occupied 48.6 GiB on GPU0; the clean run used GPUs 2/3/4/5.

## Learnings

BLOCK long-context throughput claims. Retrieval is already nondeterministic at
5,424 actual tokens and fails deterministically at 7,222 tokens for a middle
needle while a 90%-depth needle passes. Investigate DSA long-range position
selection before performance work.

## Artifacts

- `/host/arle-megamoe-t1/logs/longctx-final.log`
- `/host/arle-megamoe-t1/logs/longctx-needle-3k-8k-512.log`
- `/host/arle-megamoe-t1/logs/longctx-8k-raw-1024.log`
- `/host/arle-megamoe-t1/logs/longctx-8k-depth-ab.log`
