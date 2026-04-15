# <SHORT TITLE> — guidellm sweep, <BACKEND-LABEL>, <YYYY-MM-DD>

> Template for canonical guidellm bench wins. Copy this file when
> `scripts/bench_guidellm.sh` runs, fill the placeholders, commit. Never
> edit an existing wins entry — always create a new dated one and diff
> against the prior. Canonical params are locked in
> [`docs/plans/guidellm-integration.md`](../../plans/guidellm-integration.md) §3.

## Context

- **Backend:** <cuda | metal>
- **Model:** <Qwen/Qwen3-4B | Qwen/Qwen3.5-4B | THUDM/GLM-4 | ...>
- **Hardware:** <GPU model / SoC, VRAM, CUDA or Metal version>
- **Commit:** <short sha>
- **Feature set:** `cargo build --release <features>`
- **Non-default flags / env vars:** <list or "none">
- **Server launch:** `scripts/start_pegainfer.sh <model> <port>` (or equivalent)

## Canonical params (DO NOT CHANGE PER-RUN)

```
guidellm benchmark \
  --target http://localhost:8000 \
  --model <model> \
  --profile sweep \
  --data  prompt_tokens=1024,output_tokens=256 \
  --max-seconds 60 \
  --random-seed 20260416 \
  --output-dir bench-output/<date>-<label>/ \
  --outputs json,csv,html
```

Invoked via: `scripts/bench_guidellm.sh <backend-label>`

## Results — sweep headline table

| rate (req/s) | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| synchronous | ... | ... | ... | ... | ... | ... |
| ... (sweep auto-steps) ... |
| saturation | ... | ... | ... | ... | ... | ... |

## Artefacts

- Raw: `bench-output/<date>-<label>/benchmarks.json`
- CSV:  `bench-output/<date>-<label>/benchmarks.csv`
- HTML: `bench-output/<date>-<label>/benchmarks.html`

## Delta vs previous snapshot

- **Baseline:** <link to prior `<YYYY-MM-DD>-bench-guidellm-<label>.md`>
- **Delta table** (only when a prior snapshot exists — else "first run"):

| metric | baseline | now | Δ% |
|---|---|---|---|
| TTFT p50 @ synchronous | ... | ... | ... |
| out tok/s @ saturation | ... | ... | ... |

## Notes

- What changed in the code since baseline: <commits or "none">
- Suspected cause of any regression: <text or "n/a">
- Follow-ups: <issues to open or "none">
