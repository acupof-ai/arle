# guidellm sweep qwen35-4b-sglang-l4-p99 — guidellm sweep, qwen35-4b-sglang-l4-p99, 2026-04-17

> Template for canonical guidellm bench wins. Copy this file when
> `scripts/bench_guidellm.sh` runs, fill the placeholders, commit. Never
> edit an existing wins entry — always create a new dated one and diff
> against the prior. Canonical params are locked in
> [`docs/plans/guidellm-integration.md`](../../plans/guidellm-integration.md) §3.

## Context

- **Backend:** sglang (reference) — `qwen3_5.py` path (`Qwen3_5ForConditionalGeneration`)
- **Model:** Qwen/Qwen3.5-4B bf16 — served as `models/Qwen3.5-4B`
- **Hardware:** NVIDIA L4 24GB, CUDA 12.8 (driver 580.82.07)
- **sglang version:** 0.5.10.post1
- **Non-default flags / env vars:** `--mem-fraction-static 0.88 --dtype bfloat16
  --max-running-requests 10 --context-length 5120 --disable-cuda-graph-padding`
- **Server launch:** `python3 -m sglang.launch_server --model-path models/Qwen3.5-4B
  --port 8000 --mem-fraction-static 0.88 --dtype bfloat16 --max-running-requests 10
  --context-length 5120 --disable-cuda-graph-padding`
- **Paired run:** `2026-04-17-bench-guidellm-qwen35-4b-infer-l4-p99.md` and combined
  comparison `2026-04-17-sglang-p99-parity-qwen35-4b.md`.
- **sync TTFT p99 = 2356ms** reflects cold-start (first request landed ~4s after
  `fired up`, still finishing warmup). Constant-rate rows unaffected.

## Canonical params (DO NOT CHANGE PER-RUN)

```
guidellm benchmark \
  --target http://localhost:8000 \
  --model models/Qwen3.5-4B \
  --profile sweep \
  --data  prompt_tokens=4096,output_tokens=256 \
  --max-seconds 60 \
  --random-seed 20260416 \
  --output-dir /content/workspace/agent-infer/bench-output/2026-04-17-qwen35-4b-sglang-l4-p99/ \
  --outputs json,csv,html
```

Invoked via: `scripts/bench_guidellm.sh <backend-label>`

## Results — sweep headline table

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| sync | 767 | 2355.9 | 37.48 | 37.57 | 25.08 | 0.083 |
| throughput | 24248.9 | 46212.9 | 60.91 | 72.49 | 134.01 | 0.5 |
| 0.13541666666666666r/s | 831 | 963.9 | 41.79 | 42.33 | 32.82 | 0.117 |
| 0.1875r/s | 833.7 | 851.9 | 45.79 | 45.89 | 42.87 | 0.15 |
| 0.23958333333333331r/s | 848.5 | 862.6 | 49.7 | 49.8 | 52.37 | 0.2 |
| 0.29166666666666663r/s | 853.3 | 879.5 | 54.04 | 54.21 | 61.34 | 0.233 |
| 0.34374999999999994r/s | 837.5 | 858.9 | 57.88 | 57.99 | 70.08 | 0.267 |
| 0.3958333333333333r/s | 845.3 | 879.7 | 62.61 | 62.74 | 78.14 | 0.3 |
| 0.44791666666666663r/s | 846.8 | 873.5 | 70.2 | 70.27 | 83.63 | 0.317 |
| 0.49999999999999994r/s | 829.7 | 857.8 | 74.45 | 74.85 | 90.9 | 0.35 |


## Artefacts

- Raw: `/content/workspace/agent-infer/bench-output/2026-04-17-qwen35-4b-sglang-l4-p99/benchmarks.json`
- CSV:  `/content/workspace/agent-infer/bench-output/2026-04-17-qwen35-4b-sglang-l4-p99/benchmarks.csv`
- HTML: `/content/workspace/agent-infer/bench-output/2026-04-17-qwen35-4b-sglang-l4-p99/benchmarks.html`

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
