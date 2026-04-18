# guidellm sweep qwen3-4b-infer-l4-p99 — guidellm sweep, qwen3-4b-infer-l4-p99, 2026-04-17

> Template for canonical guidellm bench wins. Copy this file when
> `scripts/bench_guidellm.sh` runs, fill the placeholders, commit. Never
> edit an existing wins entry — always create a new dated one and diff
> against the prior. Canonical params are locked in
> [`docs/plans/guidellm-integration.md`](../../plans/guidellm-integration.md) §3.

## Context

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B (bf16)
- **Hardware:** NVIDIA L4 24GB, CUDA 12.8 (driver 580.82.07 / runtime 13.0)
- **Commit:** cae7d38 (+ local build.rs fix for flashinfer include path)
- **Feature set:** `cargo build --release -p infer --features cuda`
- **Non-default flags / env vars:** `--num-slots 10 --max-seq-len 5120`
  (auto-sized defaults give max_seq_len=2816 which is < 4096+256 bench prompt+output)
- **Server launch:** `./target/release/infer --model-path models/Qwen3-4B --port 8000
  --num-slots 10 --max-seq-len 5120`
- **Paired run:** see `2026-04-17-bench-guidellm-qwen3-4b-sglang-l4-p99.md` and
  the combined `2026-04-17-sglang-p99-parity-qwen3-4b.md` comparison.

## Canonical params (DO NOT CHANGE PER-RUN)

```
guidellm benchmark \
  --target http://localhost:8000 \
  --model Qwen/Qwen3-4B \
  --profile sweep \
  --data  prompt_tokens=4096,output_tokens=256 \
  --max-seconds 60 \
  --random-seed 20260416 \
  --output-dir /content/workspace/agent-infer/bench-output/2026-04-17-qwen3-4b-infer-l4-p99/ \
  --outputs json,csv,html
```

Invoked via: `scripts/bench_guidellm.sh <backend-label>`

## Results — sweep headline table

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| sync | 850.1 | 871.4 | 35.3 | 35.32 | 26.3 | 0.1 |
| throughput | 13232.2 | 37772.9 | 78.2 | 101.81 | 97.91 | 0.383 |
| 0.13541666666666666r/s | 1205.5 | 1233.5 | 40.86 | 40.94 | 32.62 | 0.117 |
| 0.17083333333333334r/s | 1198.7 | 1236 | 44.03 | 44.46 | 39.42 | 0.15 |
| 0.20625r/s | 1227.1 | 1251.4 | 47.1 | 47.24 | 46.23 | 0.167 |
| 0.2416666666666667r/s | 1242.6 | 1266.8 | 51.99 | 52.08 | 51.91 | 0.183 |
| 0.27708333333333335r/s | 1258.4 | 1289.4 | 57.13 | 57.24 | 57.06 | 0.217 |
| 0.3125r/s | 1264.4 | 1301.6 | 62.51 | 62.95 | 61.67 | 0.233 |
| 0.3479166666666667r/s | 1290 | 1325.3 | 68.41 | 68.99 | 65.85 | 0.25 |
| 0.38333333333333336r/s | 1292.6 | 1349.5 | 74.03 | 75.48 | 69.57 | 0.267 |


## Artefacts

- Raw: `/content/workspace/agent-infer/bench-output/2026-04-17-qwen3-4b-infer-l4-p99/benchmarks.json`
- CSV:  `/content/workspace/agent-infer/bench-output/2026-04-17-qwen3-4b-infer-l4-p99/benchmarks.csv`
- HTML: `/content/workspace/agent-infer/bench-output/2026-04-17-qwen3-4b-infer-l4-p99/benchmarks.html`

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
