# guidellm sweep cuda-l4-autotune — guidellm sweep, cuda-l4-autotune, 2026-04-16

> Template for canonical guidellm bench wins. Copy this file when
> `scripts/bench_guidellm.sh` runs, fill the placeholders, commit. Never
> edit an existing wins entry — always create a new dated one and diff
> against the prior. Canonical params are locked in
> [`docs/plans/guidellm-integration.md`](../../plans/guidellm-integration.md) §3.

## Context

- **Backend:** <cuda | metal>
- **Model:** Qwen/Qwen3-4B
- **Hardware:** <GPU model / SoC, VRAM, CUDA or Metal version>
- **Commit:** cc81b65
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
  --output-dir /content/workspace/agent-infer/bench-output/2026-04-16-cuda-l4-autotune/ \
  --outputs json,csv,html
```

Invoked via: `scripts/bench_guidellm.sh <backend-label>`

## Results — sweep headline table

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| sync | 871 | 883.7 | 35.36 | 35.44 | 26.26 | 0.1 |
| throughput | 9742.3 | 33217.5 | 94.27 | 110.81 | 88.09 | 0.3 |
| 0.125r/s | 1219.1 | 1245.1 | 41.25 | 41.36 | 30.45 | 0.117 |
| 0.15r/s | 1217.4 | 1234.4 | 42.75 | 42.85 | 35.32 | 0.133 |
| 0.175r/s | 1250.8 | 1278.6 | 47.2 | 47.25 | 39.62 | 0.15 |
| 0.2r/s | 1269.7 | 1289.2 | 47.7 | 47.74 | 44.43 | 0.167 |
| 0.22499999999999998r/s | 1277.6 | 1293.8 | 48.76 | 49.61 | 49.93 | 0.183 |
| 0.25r/s | 1255.2 | 1296.5 | 50.67 | 51.16 | 51.8 | 0.2 |
| 0.275r/s | 1272.4 | 1295.8 | 54.26 | 54.37 | 51.39 | 0.217 |
| 0.3r/s | 1243.1 | 1426.8 | 54.34 | 55 | 60.17 | 0.233 |


## Artefacts

- Raw: `/content/workspace/agent-infer/bench-output/2026-04-16-cuda-l4-autotune/benchmarks.json`
- CSV:  `/content/workspace/agent-infer/bench-output/2026-04-16-cuda-l4-autotune/benchmarks.csv`
- HTML: `/content/workspace/agent-infer/bench-output/2026-04-16-cuda-l4-autotune/benchmarks.html`

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
