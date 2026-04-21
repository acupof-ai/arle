# guidellm sweep cuda-l4-c16-prefill-pack — guidellm sweep, cuda-l4-c16-prefill-pack, 2026-04-21

> Template for canonical guidellm bench wins. Copy this file when
> `scripts/bench_guidellm.sh` runs, fill the placeholders, commit. Never
> edit an existing wins entry — always create a new dated one and diff
> against the prior. `scripts/bench_throughput.py` is legacy only.
> Canonical params are locked in
> [`docs/plans/guidellm-integration.md`](../../plans/guidellm-integration.md) §3.

## Goal

- <one sentence describing the benchmark goal and goal type>

## Hypothesis

- <expected outcome before the run>

## Command

```bash
scripts/bench_guidellm.sh <backend-label> \
  [--target http://localhost:8000] \
  [--model Qwen/Qwen3-4B] \
  [--processor models/Qwen3-4B]
```

Invoked via: `scripts/bench_guidellm.sh <backend-label> [--target URL] [--model NAME] [--processor PATH]`

## Environment

- **Backend:** <cuda | metal>
- **Model:** Qwen/Qwen3-4B
- **Hardware:** <GPU model / SoC, VRAM, CUDA or Metal version>
- **Commit:** 435225d
- **Feature set:** `cargo build --release <features>`
- **Non-default flags / env vars:** <list or "none">
- **Server launch:** `scripts/start_infer.sh <model> <port>` (or equivalent)

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json,csv,html`
- Wrapper: `scripts/bench_guidellm.sh <backend-label>`

## Results — sweep headline table

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| sync | 761.2 | 772.7 | 35.36 | 35.4 | 26.47 | 0.1 |
| throughput | 31537.2 | 54736.1 | 62.45 | 319.01 | 46.33 | 0.55 |
| 0.15625r/s | 860.8 | 870 | 40.67 | 40.9 | 32.37 | 0.133 |
| 0.21250000000000002r/s | 874.5 | 7000 | 43.48 | 47.49 | 40.77 | 0.167 |
| 0.26875r/s | 3908 | 12668.5 | 43.28 | 129.09 | 31.24 | 0.183 |
| 0.325r/s | 6042.7 | 19181.8 | 40.5 | 66.29 | 37.12 | 0.133 |
| 0.38125000000000003r/s | 7234.6 | 26543.2 | 38.68 | 66.27 | 37.64 | 0.133 |
| 0.4375r/s | 7667.1 | 23217.5 | 40.81 | 66.25 | 37.33 | 0.133 |
| 0.49375r/s | 9829.6 | 31585.1 | 38.67 | 66.35 | 37.51 | 0.133 |
| 0.55r/s | 8616.2 | 25620.5 | 40.98 | 66.27 | 37.61 | 0.133 |


## Problems

- <anything that degraded, crashed, or deviated from the watch-list>

## Learnings

- <generalizable rule or tuning takeaway>

## Δ vs baseline

- **Baseline:** <link to prior `2026-04-21-bench-guidellm-<label>.md`>
- **Delta table** (only when a prior snapshot exists — else "first run"):

| metric | baseline | now | Δ% |
|---|---|---|---|
| TTFT p50 @ synchronous | ... | ... | ... |
| out tok/s @ saturation | ... | ... | ... |

## Artefacts

- Raw: `/content/workspace/agent-infer/bench-output/2026-04-21-cuda-l4-c16-prefill-pack/benchmarks.json`
- CSV:  `/content/workspace/agent-infer/bench-output/2026-04-21-cuda-l4-c16-prefill-pack/benchmarks.csv`
- HTML: `/content/workspace/agent-infer/bench-output/2026-04-21-cuda-l4-c16-prefill-pack/benchmarks.html`

## Notes

- What changed in the code since baseline: <commits or "none">
- Suspected cause of any regression: <text or "n/a">
- Follow-ups: <issues to open or "none">
