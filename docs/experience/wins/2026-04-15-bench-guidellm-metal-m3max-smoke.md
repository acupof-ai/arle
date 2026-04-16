# guidellm smoke metal-m3max — guidellm sweep, metal-m3max-smoke, 2026-04-15

> Template for canonical guidellm bench wins. Copy this file when
> `scripts/bench_guidellm.sh` runs, fill the placeholders, commit. Never
> edit an existing wins entry — always create a new dated one and diff
> against the prior. Canonical params are locked in
> [`docs/plans/guidellm-integration.md`](../../plans/guidellm-integration.md) §3.

## Context

- **Backend:** <cuda | metal>
- **Model:** Qwen/Qwen3-0.6B
- **Hardware:** <GPU model / SoC, VRAM, CUDA or Metal version>
- **Commit:** 27299c2
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
  --output-dir bench-output/smoke-sync/ \
  --outputs json,csv,html
```

Invoked via: `scripts/bench_guidellm.sh <backend-label>`

## Results — sweep headline table

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| sync | 3473.5 | 3503.8 | 7.58 | 7.62 | 34.3 | 0.226 |


## Artefacts

- Raw: `bench-output/smoke-sync/benchmarks.json`
- CSV:  `bench-output/smoke-sync/benchmarks.csv`
- HTML: `bench-output/smoke-sync/benchmarks.html`

## Delta vs previous snapshot

- **Baseline:** <link to prior `2026-04-15-bench-guidellm-<label>.md`>
- **Delta table** (only when a prior snapshot exists — else "first run"):

| metric | baseline | now | Δ% |
|---|---|---|---|
| TTFT p50 @ synchronous | ... | ... | ... |
| out tok/s @ saturation | ... | ... | ... |

## Notes

- What changed in the code since baseline: <commits or "none">
- Suspected cause of any regression: <text or "n/a">
- Follow-ups: <issues to open or "none">
