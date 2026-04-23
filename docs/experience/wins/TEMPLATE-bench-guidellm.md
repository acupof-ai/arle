# <SHORT TITLE> — guidellm sweep, <BACKEND-LABEL>, <YYYY-MM-DD>

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
  [--processor models/Qwen3-4B] \
  [--trace-interval-ms 1000]
```

Invoked via: `scripts/bench_guidellm.sh <backend-label> [--target URL] [--model NAME] [--processor PATH] [--trace-interval-ms N]`

## Environment

- **Backend:** <cuda | metal>
- **Model:** <Qwen/Qwen3-4B | Qwen/Qwen3.5-4B | THUDM/GLM-4 | ...>
- **Hardware:** <GPU model / SoC, VRAM, CUDA or Metal version>
- **Commit:** <short sha>
- **Feature set:** `cargo build --release <features>`
- **Non-default flags / env vars:** <list or "none">
- **Server launch:** `scripts/start_infer.sh <model> <port>` (or equivalent)

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh <backend-label>`

## Results — sweep headline table

| rate (req/s) | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| synchronous | ... | ... | ... | ... | ... | ... |
| ... (sweep auto-steps) ... |
| saturation | ... | ... | ... | ... | ... | ... |

## Results — service-side KV / scheduler metrics

| metric | value |
|---|---:|
| peak active | ... |
| peak waiting | ... |
| peak prefill_queue | ... |
| peak kv_util | ... |
| `prefix_hit_rate` | ... |
| `prefix_skip_rate` | ... |
| `kv_fetch_q` | ... |
| `kv_fetch_waiters` | ... |
| `kv_store_q` | ... |
| `kv_store` | ... |
| `kv_bp` | ... |
| `tier_recall` | ... / n/a |
| `tier_src` | ... / n/a |
| `tier_promoted` | ... / n/a |
| `tier_fallback` | ... / n/a |

## Results — request accounting

| metric | value |
|---|---:|
| completed input tokens | ... |
| incomplete input tokens | ... |
| completed output tokens | ... |
| incomplete output tokens | ... |

## Problems

- <anything that degraded, crashed, or deviated from the watch-list>

## Learnings

- <generalizable rule or tuning takeaway>

## Δ vs baseline

- **Baseline:** <link to prior `<YYYY-MM-DD>-bench-guidellm-<label>.md`>
- **Delta table** (only when a prior snapshot exists — else "first run"):

| metric | baseline | now | Δ% |
|---|---|---|---|
| TTFT p50 @ synchronous | ... | ... | ... |
| out tok/s @ saturation | ... | ... | ... |

## Artefacts

- Raw: `bench-output/<date>-<label>/benchmarks.json`
- CSV:  `bench-output/<date>-<label>/benchmarks.csv`
- HTML: `bench-output/<date>-<label>/benchmarks.html`
- Service trace (before): `bench-output/<date>-<label>/service_stats_before.txt`
- Service trace (during): `bench-output/<date>-<label>/service_stats_trace.jsonl`
- Service trace (after):  `bench-output/<date>-<label>/service_stats_after.txt`
- Service trace (summary): `bench-output/<date>-<label>/service_stats_trace_summary.md`

## Notes

- What changed in the code since baseline: <commits or "none">
- Suspected cause of any regression: <text or "n/a">
- Follow-ups: <issues to open or "none">
