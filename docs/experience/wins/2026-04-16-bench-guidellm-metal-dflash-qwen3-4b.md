# guidellm sweep metal-dflash-qwen3-4b — guidellm sweep, metal-dflash-qwen3-4b, 2026-04-16

> Template for canonical guidellm bench wins. Copy this file when
> `scripts/bench_guidellm.sh` runs, fill the placeholders, commit. Never
> edit an existing wins entry — always create a new dated one and diff
> against the prior. Canonical params are locked in
> [`docs/plans/guidellm-integration.md`](../../plans/guidellm-integration.md) §3.

## Context

- **Backend:** <cuda | metal>
- **Model:** mlx-community/Qwen3-4B-bf16
- **Hardware:** <GPU model / SoC, VRAM, CUDA or Metal version>
- **Commit:** d83ee99
- **Feature set:** `cargo build --release <features>`
- **Non-default flags / env vars:** <list or "none">
- **Server launch:** `scripts/start_infer.sh <model> <port>` (or equivalent)

## Canonical params (DO NOT CHANGE PER-RUN)

```
guidellm benchmark \
  --target http://127.0.0.1:8014 \
  --model <model> \
  --profile sweep \
  --data  prompt_tokens=1024,output_tokens=256 \
  --max-seconds 60 \
  --random-seed 20260416 \
  --output-dir /path/to/code/agent-infer/bench-output/2026-04-16-metal-dflash-qwen3-4b-run2/ \
  --outputs json,csv,html
```

Invoked via: `scripts/bench_guidellm.sh <backend-label>`

## Results — sweep headline table

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| sync | 5828.5 | 5875.2 | 85.05 | 85.55 | 10.37 | 0.033 |
| throughput | 5799.8 | 33030.6 | 53.8 | 82.33 | 12.5 | 0.033 |
| 0.03333333333333333r/s | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.03333333333333333r/s | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.03333333333333333r/s | 5740.5 | 9068.3 | 58.94 | 61.71 | 12.07 | 0.033 |
| 0.033333333333333326r/s | 5883.7 | 6097.8 | 73.48 | 79.66 | 10.46 | 0.033 |
| 0.033333333333333326r/s | 5903.3 | 5944.9 | 84.48 | 93.11 | 9.53 | 0.033 |
| 0.033333333333333326r/s | 5738.2 | 5738.2 | 77.02 | 77.02 | 13.03 | 0.017 |
| 0.033333333333333326r/s | 5737.8 | 6001.9 | 70.24 | 83.52 | 10.63 | 0.033 |
| 0.033333333333333326r/s | 5589.2 | 6105.8 | 77.57 | 83.53 | 10.18 | 0.033 |


## Artefacts

- Raw: `/path/to/code/agent-infer/bench-output/2026-04-16-metal-dflash-qwen3-4b-run2/benchmarks.json`
- CSV:  `/path/to/code/agent-infer/bench-output/2026-04-16-metal-dflash-qwen3-4b-run2/benchmarks.csv`
- HTML: `/path/to/code/agent-infer/bench-output/2026-04-16-metal-dflash-qwen3-4b-run2/benchmarks.html`

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
