# guidellm sweep metal-dflash-scheduler — guidellm sweep, metal-dflash-scheduler, 2026-04-16

> Template for canonical guidellm bench wins. Copy this file when
> `scripts/bench_guidellm.sh` runs, fill the placeholders, commit. Never
> edit an existing wins entry — always create a new dated one and diff
> against the prior. Canonical params are locked in
> [`docs/plans/guidellm-integration.md`](../../plans/guidellm-integration.md) §3.

## Context

- **Backend:** <cuda | metal>
- **Model:** mlx-community/Qwen3-4B-bf16
- **Hardware:** <GPU model / SoC, VRAM, CUDA or Metal version>
- **Commit:** 3b75e5e
- **Feature set:** `cargo build --release <features>`
- **Non-default flags / env vars:** <list or "none">
- **Server launch:** `scripts/start_pegainfer.sh <model> <port>` (or equivalent)

## Canonical params (DO NOT CHANGE PER-RUN)

```
guidellm benchmark \
  --target http://127.0.0.1:8014 \
  --model <model> \
  --profile sweep \
  --data  prompt_tokens=1024,output_tokens=256 \
  --max-seconds 60 \
  --random-seed 20260416 \
  --output-dir /path/to/code/agent-infer/bench-output/2026-04-16-metal-dflash-scheduler/ \
  --outputs json,csv,html
```

Invoked via: `scripts/bench_guidellm.sh <backend-label>`

## Results — sweep headline table

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| sync | 0 | 0 | 0 | 0 | 50.67 | 0.167 |
| throughput | 0 | 0 | 0 | 0 | 45.58 | 0.15 |
| 0.16458333333333333r/s | 0 | 0 | 0 | 0 | 46.86 | 0.15 |
| 0.16249999999999998r/s | 0 | 0 | 0 | 0 | 46.51 | 0.15 |
| 0.16041666666666665r/s | 0 | 0 | 0 | 0 | 45.52 | 0.15 |
| 0.15833333333333333r/s | 0 | 0 | 0 | 0 | 45.07 | 0.15 |
| 0.15625r/s | 0 | 0 | 0 | 0 | 44.44 | 0.15 |
| 0.15416666666666667r/s | 0 | 0 | 0 | 0 | 43.92 | 0.15 |
| 0.15208333333333332r/s | 0 | 0 | 0 | 0 | 43.27 | 0.15 |
| 0.15r/s | 0 | 0 | 0 | 0 | 43.22 | 0.15 |


## Artefacts

- Raw: `/path/to/code/agent-infer/bench-output/2026-04-16-metal-dflash-scheduler/benchmarks.json`
- CSV:  `/path/to/code/agent-infer/bench-output/2026-04-16-metal-dflash-scheduler/benchmarks.csv`
- HTML: `/path/to/code/agent-infer/bench-output/2026-04-16-metal-dflash-scheduler/benchmarks.html`

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
