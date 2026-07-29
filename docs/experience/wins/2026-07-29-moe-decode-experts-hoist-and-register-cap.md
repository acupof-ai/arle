# MoE decode experts: hoist the shared activation load, cap the registers

## Context

After batched FA3, `dsv4_fp8_grouped_{down,swiglu}_decode` were 53.9% of GPU
time at c=16 — 1,025 µs/layer against an ~89 µs weight-read roofline. ncu said
they were neither bandwidth- nor compute-bound: DRAM 5-12%, SM 20-25%, L2 5-9%,
achieved occupancy 20-27% against a 25-37.5% theoretical ceiling set by 72
(swiglu) / 86 (down) registers. Latency-bound, with too few resident warps to
hide it.

## What Worked

Two inner-loop fixes, no restructuring.

1. **The activation vector was re-loaded per weight row.** `fp8d_dot16` took a
   pointer and did its own two 16B loads, but the activations are shared across
   the rows a warp owns — down loaded them 4× (`ROW_TILE`), swiglu 2× (gate and
   up). Hoisted out of the row loop; `fp8d_dot16_x` takes them by value.
2. **`__launch_bounds__(256, 4)`** caps registers at `65536/(256*4) = 64` and
   takes theoretical occupancy to 50%.

## Measurement

Matched A/B, same GPU, `Qwen3.6-35B-A3B-FP8`, 1×H20, 48 req/point.

| c | TPOT before | after | ITL p50 before | after |
|---|---:|---:|---:|---:|
| 1 | 16.04 ms | 15.49 ms | 15.77 ms | 15.29 ms |
| 8 | 42.61 ms | **37.53 ms** | 37.21 ms | **31.85 ms** (1.17×) |
| 16 | 71.24 ms | **61.02 ms** | 59.51 ms | **48.75 ms** (1.22×) |

total tok/s at c=16: 32,374 → **37,302** (+15.2%). Step model
`14.9 + 2.79·B` → `14.9 + 2.11·B` — the per-row marginal falls **1.32×** with a
flat intercept. Gate exact=3 DET at 512/4k/16k/32k.

ncu confirms each mechanism:

| | before | after |
|---|---:|---:|
| registers (swiglu / down) | 72 / 86 | **63 / 64** |
| theoretical occupancy | 37.5% / 25% | **50% / 50%** |
| achieved occupancy | 26.6% / 20.1% | **34.0% / 34.1%** |
| active warps/SM | 17.0 / 12.9 | **21.7 / 21.9** |
| waves/SM | 70 / 105 | **52.5 / 52.5** |
| L2 hit (down) | 9.41% | **57.88%** |
| DRAM throughput | 12.5% / 5.3% | 14.5% / 8.0% |

The down kernel's L2 hit rate is the tell: dropping the 4× redundant activation
reads leaves streaming weights plus an activation footprint small enough to sit
in L2.

## Still open

DRAM is 8-14%, so the kernel is still latency-bound — the remaining fix is
`cp.async`/TMA multi-stage pipelining, which is the sm90 idiom this kernel has
never had. Occupancy alone cannot cover a load latency nothing is prefetching.

## Rule

**Look for the redundant load before reaching for the exotic fix.** The sm90
answer here was going to be TMA and warp specialization; two thirds of the
available win was a vector being re-read four times inside an unrolled loop,
visible in fifteen lines of source. Profile first, but read the inner loop
before designing a pipeline.
