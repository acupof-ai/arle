# CUDA L4 per-leg rerun on `f15f87c` stays valid through `c4`, then goes invalid at `c8/c16`

## Context

- Pulled `origin/main` to `f15f87c` and rebuilt `infer`.
- The first serial `c1,c2,c4,c8,c16` sweep stalled once all 16 slots were filled; see [`2026-04-22-cuda-l4-c16-pull-rerun-stalls-after-slot-fill.md`](./2026-04-22-cuda-l4-c16-pull-rerun-stalls-after-slot-fill.md).
- To finish the comparison, reran `infer` one concurrency leg at a time on fresh server instances:
  - `c1`: `bench-output/2026-04-22-infer-qwen3-4b-l4-c1-only-f15f87c`
  - `c2`: `bench-output/2026-04-22-infer-qwen3-4b-l4-c2-only-f15f87c`
  - `c4`: `bench-output/2026-04-22-infer-qwen3-4b-l4-c4-only-f15f87c`
  - `c8`: `bench-output/2026-04-22-infer-qwen3-4b-l4-c8-only-f15f87c`
  - `c16`: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-only-f15f87c`
- Comparison baseline for `sglang` reused the same-day same-host same-model run already captured at `bench-output/2026-04-22-sglang-qwen3-4b-l4-c1-c16-serial`.

## Root Cause

- `infer` on `f15f87c` remains benchmark-valid at `c1/c2/c4`.
- At `c8` and `c16`, the server no longer produces trustworthy benchmark outputs:
  - `guidellm` reports `successful requests ... every sampled output was empty`
  - `TTFT p50 = 0`
  - `ITL p50 = 0`
- Server logs show the invalidity is real, not a `guidellm` formatting glitch:
  - high-concurrency requests frequently end as `Request N done: 0 tokens`
  - `c16` also surfaces repeated `CUDA_ERROR_OUT_OF_MEMORY` on prefill batch allocation
- The resulting `out tok/s` figures at `c8/c16` are therefore not comparable to either the pre-pull run or `sglang`.

## Fix

- Treat `c8/c16` on `f15f87c` as a serving regression until the runtime can complete requests with non-empty streamed text and non-zero TTFT/ITL.
- Keep the valid `c1/c2/c4` data; do not average or smooth in the invalid `c8/c16` numbers.
- Use the trace plan in [`../../plans/2026-04-22-cuda-end-to-end-trace.md`](../../plans/2026-04-22-cuda-end-to-end-trace.md) to instrument:
  - request admission
  - batch launch/readback
  - zero-token completion reasons
  - OOM path on prefill allocation

## Comparison

| conc | `infer` `f15f87c` out tok/s | `infer` `f15f87c` TTFT p50 (ms) | `infer` `f15f87c` ITL p50 (ms) | status | pre-pull `infer` `f98ca92` out tok/s | `sglang` out tok/s |
|---|---:|---:|---:|---|---:|---:|
| 1 | 26.53 | 744.7 | 35.32 | valid | 26.59 | 26.46 |
| 2 | 41.69 | 1524.5 | 38.83 | valid | 41.59 | 45.81 |
| 4 | 58.50 | 3029.1 | 43.53 | valid | 36.70 | 74.05 |
| 8 | 1025.84 | 0.0 | 0.00 | invalid | 57.71 | 107.79 |
| 16 | 4170.53 | 0.0 | 0.00 | invalid | 45.08 | 137.07 |

Valid-leg deltas against `sglang`:

- `c1`: essentially tied (`+0.3%` throughput).
- `c2`: still behind (`-9.0%` throughput).
- `c4`: materially improved versus pre-pull (`+59.4%` throughput) but still trails `sglang` (`-21.0%`).

## Rule

- When a pull turns a previously slow but valid high-concurrency leg into zero-token or empty-output completions, classify the run as **invalid** rather than reporting the inflated throughput numbers.
- For scheduler hot-path changes, a finished benchmark process is not enough; the output must also preserve non-empty streamed text and non-zero TTFT/ITL at the target concurrency.
