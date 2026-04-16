# guidellm sweep cuda-l4-kv-baseline — guidellm sweep, cuda-l4-kv-baseline, 2026-04-16

> Template for canonical guidellm bench wins. Copy this file when
> `scripts/bench_guidellm.sh` runs, fill the placeholders, commit. Never
> edit an existing wins entry — always create a new dated one and diff
> against the prior. Canonical params are locked in
> [`docs/plans/guidellm-integration.md`](../../plans/guidellm-integration.md) §3.

## Context

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B
- **Hardware:** NVIDIA L4 24GB, CUDA 12.8, driver 580.82.07
- **Commit:** c1956da (feat: wire ServerMetrics + optimize bench scripts)
- **Feature set:** `cargo build --release -p infer` (default CUDA features)
- **Non-default flags / env vars:** `PEGAINFER_CUDA_SM=89`
- **Server launch:** `./target/release/infer --model-path models/Qwen3-4B --port 8000`
- **Baseline:** first canonical run — no prior snapshot to diff against

## Canonical params (DO NOT CHANGE PER-RUN)

```
guidellm benchmark \
  --target http://localhost:8000 \
  --model <model> \
  --profile sweep \
  --data  prompt_tokens=1024,output_tokens=256 \
  --max-seconds 60 \
  --random-seed 20260416 \
  --output-dir /content/workspace/agent-infer/bench-output/2026-04-16-cuda-l4-kv-baseline/ \
  --outputs json,csv,html
```

Invoked via: `scripts/bench_guidellm.sh <backend-label>`

## Results — sweep headline table

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| sync | 805 | 836.3 | 35.56 | 35.61 | 26.26 | 0.1 |
| throughput | 21347.8 | 51390.6 | 53.77 | 68.25 | 89.56 | 0.35 |
| 0.13124999999999998r/s | 1270.2 | 1306.4 | 40.8 | 40.97 | 31.72 | 0.117 |
| 0.16249999999999998r/s | 1261.7 | 1283.1 | 41.16 | 42.18 | 31.88 | 0.15 |
| 0.19374999999999998r/s | 1240 | 1280.5 | 42.47 | 43.1 | 31.85 | 0.167 |
| 0.22499999999999998r/s | 1239.7 | 1352.7 | 43.08 | 48.02 | 49.78 | 0.183 |
| 0.25625r/s | 1273.6 | 1296.5 | 46.26 | 48.4 | 56.29 | 0.217 |
| 0.2875r/s | 1243 | 1307.4 | 46.83 | 48.34 | 62.12 | 0.233 |
| 0.31875r/s | 1250.6 | 1287.2 | 50.07 | 52.02 | 67.17 | 0.25 |
| 0.35r/s | 1253.5 | 1290.2 | 50.52 | 54.63 | 73.64 | 0.283 |


## Artefacts

- Raw: `/content/workspace/agent-infer/bench-output/2026-04-16-cuda-l4-kv-baseline/benchmarks.json`
- CSV:  `/content/workspace/agent-infer/bench-output/2026-04-16-cuda-l4-kv-baseline/benchmarks.csv`
- HTML: `/content/workspace/agent-infer/bench-output/2026-04-16-cuda-l4-kv-baseline/benchmarks.html`

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
