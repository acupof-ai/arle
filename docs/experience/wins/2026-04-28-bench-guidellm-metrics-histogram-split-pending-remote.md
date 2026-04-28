# Metrics Histogram Split — guidellm sweep, pending-remote, 2026-04-28

## Goal

- Regression-check a behavior-preserving extraction of `infer/src/metrics.rs` histogram code before continuing the metrics cleanup.

## Hypothesis

- No serving metric, latency, or throughput movement is expected because only `Histogram`, `HistogramSet`, latency buckets, and seconds/microseconds conversion helpers moved behind the same `crate::metrics` facade.

## Command

```bash
scripts/bench_guidellm.sh cuda-l4-metrics-histogram-split
```

Invoked via: not run locally; status `pending-remote`.

## Environment

- **Backend:** pending CUDA remote
- **Model:** pending, use current CUDA L4 Qwen3 baseline model
- **Hardware:** pending CUDA L4 benchmark host
- **Commit:** pending post-commit SHA
- **Feature set:** `CUDA_HOME=/usr/local/cuda cargo build --release`
- **Non-default flags / env vars:** none expected
- **Server launch:** pending, use existing `scripts/bench_guidellm.sh` wrapper defaults

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
| pending-remote | pending | pending | pending | pending | pending | pending |

## Results — service-side KV / scheduler metrics

| metric | value |
|---|---:|
| peak active | pending |
| peak waiting | pending |
| peak prefill_queue | pending |
| peak kv_util | pending |
| `prefix_hit_rate` | pending |
| `prefix_skip_rate` | pending |
| `kv_fetch_q` | pending |
| `kv_fetch_waiters` | pending |
| `kv_store_q` | pending |
| `kv_store` | pending |
| `kv_bp` | pending |
| `tier_recall` | pending |
| `tier_src` | pending |
| `tier_promoted` | pending |
| `tier_fallback` | pending |

## Results — request accounting

| metric | value |
|---|---:|
| completed input tokens | pending |
| incomplete input tokens | pending |
| completed output tokens | pending |
| incomplete output tokens | pending |

## Problems

- Local workspace is not the CUDA benchmark host, so the canonical guidellm run is deferred.
- Full workspace no-cuda tests remain blocked by the existing train SFT data assertion noted in the previous tranche; metrics-specific infer tests passed.

## Learnings

- Keeping histogram mechanics behind a small module preserves `crate::metrics` API while separating reusable latency math from the large server counter/rendering surface.

## Δ vs baseline

- **Baseline:** pending, compare against the latest CUDA L4 guidellm entry after the remote run.

| metric | baseline | now | Δ% |
|---|---|---|---|
| TTFT p50 @ synchronous | pending | pending | pending |
| out tok/s @ saturation | pending | pending | pending |

## Artefacts

- Raw: pending
- CSV: pending
- HTML: pending
- Service trace (before): pending
- Service trace (during): pending
- Service trace (after): pending
- Service trace (summary): pending

## Notes

- What changed in the code since baseline: extracted histogram data structures, bucket constants, and conversion helpers from `metrics.rs` into `metrics/histogram.rs`.
- Suspected cause of any regression: n/a; any movement should be treated as noise or an accidental rendering/measurement change.
- Follow-ups: run the pending remote guidellm sweep before claiming this cleanup has no performance effect.
