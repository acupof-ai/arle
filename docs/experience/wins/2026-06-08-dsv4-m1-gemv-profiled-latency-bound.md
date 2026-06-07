# DSv4 M=1 FP8 GEMV profiled: memory-LATENCY-bound (long_scoreboard), not bandwidth/compute

## Context

The decode-6ms forward lever is the M=1 FP8 GEMV (projections + MoE), which runs ~12-18×
above its bandwidth floor. ncu on the TP=8 job fails (NCCL + kernel-replay), so I built a
single-GPU microbench (`/tmp/gemv_microbench.cu` → links `quantized_gemv.cu`) to isolate
`dsv4_fp8_gemv_batch_kernel` for clean profiling.

## Measured (single-GPU H20, M=1)

| shape | us/call | bw floor | ratio |
|---|---:|---:|---:|
| 4096×4096 | 60.4 | 5.0 | 12.1× |
| 1536×4096 | 26.2 | 1.9 | 14× |
| 4096×512 | 11.4 | 0.6 | 18× |

ncu SpeedOfLight (4096×4096): **sm 62.6%, dram 5.7%, occupancy 83%** → not bandwidth-bound.
ncu stall reasons (per issue active): **long_scoreboard = 16.6 (DOMINANT)**, wait 2.5,
barrier 0.67, short_scoreboard 0.53, throttles ~0. **→ memory-LATENCY-bound**: warps stall
on global-load latency that 83% occupancy doesn't hide (too few independent loads in flight
relative to the latency).

## Fixes tried (license-or-kill, both REVERTED — neither won)

1. **Scale-hoist + drop per-element division** (decode e8m0 once per 128-block, not per
   element): sm 62.6→51.2% (real compute saving) but **time flat** (58 vs 60µs) — the
   compute wasn't the wall; latency is.
2. **4× unroll + 4 accumulators + prefetched loads** (more memory-level parallelism):
   **WORSE, 58→69µs** — register pressure beat the compiler's own scheduling; occupancy
   held at 83% but extra instructions didn't buy latency hiding.

## Rule / next

- The M=1 GEMV is memory-LATENCY-bound (long_scoreboard), confirmed by ncu — NOT compute
  (the hardware `__nv_fp8_e4m3` dequant is fine) nor bandwidth (dram 5%). Obvious
  compute/ILP fixes don't win; this matches the prior "MMA GEMV killed" (tensor-core was
  the wrong shape, and the latency wall is the real difficulty).
- The remaining real lever to decode-6ms is hiding this load latency: iterative tiling/
  config tuning (GEMV_ROWS / threads_per_row to raise concurrent loads-in-flight; an
  async-copy `cp.async` double-buffer of the FP8 weight row; or a layout that lets the
  memory system pipeline more) — a focused experimental kernel pass, not a one-shot edit.
  Microbench harness is in place (`/tmp/gemv_microbench.cu`) for fast single-GPU ncu A/B.
- DSv4-Flash B=1 decode stands at ~15ms (MTP +71%, landed); this GEMV latency wall is the
  gate to ~6ms, now profiled to the exact stall (long_scoreboard) for the next pass.
