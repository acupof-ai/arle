# guidellm sweep cuda-l4-kv-baseline-v2 — guidellm sweep, cuda-l4-kv-baseline-v2, 2026-04-16

> Template for canonical guidellm bench wins. Copy this file when
> `scripts/bench_guidellm.sh` runs, fill the placeholders, commit. Never
> edit an existing wins entry — always create a new dated one and diff
> against the prior. Canonical params are locked in
> [`docs/plans/guidellm-integration.md`](../../plans/guidellm-integration.md) §3.

## Context

- **Backend:** <cuda | metal>
- **Model:** Qwen/Qwen3-4B
- **Hardware:** <GPU model / SoC, VRAM, CUDA or Metal version>
- **Commit:** aef22a7
- **Feature set:** `cargo build --release <features>`
- **Non-default flags / env vars:** <list or "none">
- **Server launch:** `scripts/start_infer.sh <model> <port>` (or equivalent)

## Canonical params (DO NOT CHANGE PER-RUN)

```
guidellm benchmark \
  --target http://localhost:8000 \
  --model <model> \
  --profile sweep \
  --data  prompt_tokens=1024,output_tokens=256 \
  --max-seconds 60 \
  --random-seed 20260416 \
  --output-dir /content/workspace/agent-infer/bench-output/2026-04-16-cuda-l4-kv-baseline-v2/ \
  --outputs json,csv,html
```

Invoked via: `scripts/bench_guidellm.sh <backend-label>`

## Results — sweep headline table

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| sync | 766.3 | 784.3 | 35.46 | 35.53 | 26.43 | 0.1 |
| throughput | 21290.7 | 51425.3 | 53.71 | 69.94 | 89.39 | 0.35 |
| 0.13125r/s | 1244.8 | 1304.9 | 40.86 | 41.06 | 31.66 | 0.117 |
| 0.1625r/s | 1238.6 | 1280.8 | 41.63 | 42.12 | 38.15 | 0.133 |
| 0.19374999999999998r/s | 1246.9 | 1279.5 | 42.45 | 43.11 | 39.91 | 0.167 |
| 0.22499999999999998r/s | 1255 | 1310.7 | 45.5 | 45.71 | 41.23 | 0.2 |
| 0.25625r/s | 1261.3 | 1387.9 | 46.82 | 48.77 | 53.33 | 0.2 |
| 0.2875r/s | 1259.3 | 1290.9 | 46.79 | 48.52 | 62.16 | 0.233 |
| 0.31875r/s | 1242 | 1484.4 | 50.17 | 52.83 | 67.21 | 0.25 |
| 0.35r/s | 1243.1 | 1316 | 50.55 | 54.23 | 73.67 | 0.283 |


## Artefacts

- Raw: `/content/workspace/agent-infer/bench-output/2026-04-16-cuda-l4-kv-baseline-v2/benchmarks.json`
- CSV:  `/content/workspace/agent-infer/bench-output/2026-04-16-cuda-l4-kv-baseline-v2/benchmarks.csv`
- HTML: `/content/workspace/agent-infer/bench-output/2026-04-16-cuda-l4-kv-baseline-v2/benchmarks.html`

## Delta vs previous snapshot

- **Baseline:** <link to prior `2026-04-16-bench-guidellm-<label>.md`>
- **Delta table** (only when a prior snapshot exists — else "first run"):

| metric | baseline | now | Δ% |
|---|---|---|---|
| TTFT p50 @ synchronous | ... | ... | ... |
| out tok/s @ saturation | ... | ... | ... |

## Notes

- What changed in the code since baseline: <commits or "none">
- Suspected cause of any regression: <text or "n/a">
- Follow-ups: <issues to open or "none">
