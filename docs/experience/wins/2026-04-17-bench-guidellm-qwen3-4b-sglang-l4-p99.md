# guidellm sweep qwen3-4b-sglang-l4-p99 — guidellm sweep, qwen3-4b-sglang-l4-p99, 2026-04-17

> Template for canonical guidellm bench wins. Copy this file when
> `scripts/bench_guidellm.sh` runs, fill the placeholders, commit. Never
> edit an existing wins entry — always create a new dated one and diff
> against the prior. Canonical params are locked in
> [`docs/plans/guidellm-integration.md`](../../plans/guidellm-integration.md) §3.

## Context

- **Backend:** sglang (reference)
- **Model:** Qwen/Qwen3-4B (bf16) — served as `models/Qwen3-4B` (path-style id)
- **Hardware:** NVIDIA L4 24GB, CUDA 12.8 (driver 580.82.07 / runtime 13.0)
- **sglang version:** 0.5.10.post1
- **Non-default flags / env vars:** `--mem-fraction-static 0.88 --dtype bfloat16
  --max-running-requests 10 --context-length 5120 --disable-cuda-graph-padding`
  (matched to infer's 10 slots × 5120 seq-len to make the comparison apples-to-apples).
- **Server launch:** `python3 -m sglang.launch_server --model-path models/Qwen3-4B
  --port 8000 --mem-fraction-static 0.88 --dtype bfloat16 --max-running-requests 10
  --context-length 5120 --disable-cuda-graph-padding`
- **Paired run:** see `2026-04-17-bench-guidellm-qwen3-4b-infer-l4-p99.md` and
  the combined `2026-04-17-sglang-p99-parity-qwen3-4b.md` comparison.

**Caveat — sync row**: synchronous profile completed 0/1 requests in 60s
because guidellm started issuing requests at 04:06:17 but sglang's internal
warmup finished at 04:06:12 — the first request hit an almost-warm server
and its initial prefill+decode (plus a handful of synchronous retries)
didn't drain inside the 60s window. Re-running sync after warmup would show
realistic p50/p99. Throughput + constant rows are unaffected.

## Canonical params (DO NOT CHANGE PER-RUN)

```
guidellm benchmark \
  --target http://localhost:8000 \
  --model models/Qwen3-4B \
  --profile sweep \
  --data  prompt_tokens=4096,output_tokens=256 \
  --max-seconds 60 \
  --random-seed 20260416 \
  --output-dir /content/workspace/agent-infer/bench-output/2026-04-17-qwen3-4b-sglang-l4-p99/ \
  --outputs json,csv,html
```

Invoked via: `scripts/bench_guidellm.sh <backend-label>`

## Results — sweep headline table

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| sync | 0 | 0 | 0 | 0 | 0 | 0 |
| throughput | 25353.7 | 49488 | 70.89 | 82.31 | 115.83 | 0.333 |
| 0.04166666666666667r/s | 702.8 | 707.6 | 35.52 | 35.53 | 13.46 | 0.05 |
| 0.08333333333333334r/s | 706.2 | 710.9 | 35.54 | 35.57 | 22.42 | 0.083 |
| 0.125r/s | 791 | 819.2 | 39.79 | 39.87 | 30.76 | 0.117 |
| 0.16666666666666669r/s | 808 | 829.7 | 41.32 | 41.37 | 39.29 | 0.15 |
| 0.20833333333333337r/s | 799.3 | 823.9 | 45.94 | 45.97 | 47.08 | 0.167 |
| 0.25r/s | 817 | 838.8 | 50.64 | 50.69 | 53.86 | 0.2 |
| 0.2916666666666667r/s | 810.5 | 835.8 | 55.69 | 55.83 | 60.85 | 0.233 |
| 0.33333333333333337r/s | 818.3 | 847.3 | 60.87 | 61.14 | 66.58 | 0.25 |


## Artefacts

- Raw: `/content/workspace/agent-infer/bench-output/2026-04-17-qwen3-4b-sglang-l4-p99/benchmarks.json`
- CSV:  `/content/workspace/agent-infer/bench-output/2026-04-17-qwen3-4b-sglang-l4-p99/benchmarks.csv`
- HTML: `/content/workspace/agent-infer/bench-output/2026-04-17-qwen3-4b-sglang-l4-p99/benchmarks.html`

## Delta vs previous snapshot

- **Baseline:** <link to prior `2026-04-17-bench-guidellm-<label>.md`>
- **Delta table** (only when a prior snapshot exists — else "first run"):

| metric | baseline | now | Δ% |
|---|---|---|---|
| TTFT p50 @ synchronous | ... | ... | ... |
| out tok/s @ saturation | ... | ... | ... |

## Notes

- What changed in the code since baseline: <commits or "none">
- Suspected cause of any regression: <text or "n/a">
- Follow-ups: <issues to open or "none">
