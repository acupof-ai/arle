# guidellm sweep qwen3-4b-infer-l4-mixed256 — **REGRESSION** MIXED_PREFILL_CAP=256 experiment

## TL;DR — negative result

Raising `MIXED_PREFILL_CAP` from 64 to 256 regresses peak throughput by
**-14%** and worsens TTFT p99 at the lowest rate by **+31%**, with no
meaningful TTFT improvement at mid-range rates. Reverted on commit
after this snapshot. Retaining as the first pass of experimental
evidence that the post-overlap residual TTFT gap vs sglang is NOT
caused by mixed-prefill chunk size.

## Context

- **Backend:** cuda
- **Model:** Qwen3-4B (bf16)
- **Hardware:** NVIDIA L4 24GB, CUDA 12.8 (driver 580.82.07 / runtime 13.0)
- **Commit:** 7257983 (+ local `MIXED_PREFILL_CAP = 64 → 256` in
  `infer/src/scheduler/cuda/decode.rs` and `infer/src/model/qwen3/batch_decode.rs`,
  **reverted after this bench**)
- **Feature set:** `cargo build --release -p infer --features cuda`
- **Non-default flags:** `--num-slots 10 --max-seq-len 5120 --mem-fraction-static 0.88`

## Results — sweep headline table

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| sync | 836 | 847 | 35.24 | 35.27 | 26.37 | 0.1 |
| throughput | 26735.9 | 55106.8 | 75.59 | 85.47 | **84.1** | 0.367 |
| 0.1333r/s | 1201.6 | **1614.4** | 40.85 | 40.93 | 32.56 | 0.117 |
| 0.1667r/s | 1192.7 | 1217.4 | 42.14 | 42.23 | 38.97 | 0.15 |
| 0.2000r/s | 1222.2 | 1240.7 | 46.93 | 46.96 | 44.63 | 0.167 |
| 0.2333r/s | 1224.0 | 1252.0 | 51.42 | 51.48 | 49.94 | 0.183 |
| 0.2667r/s | 1243.3 | 1258.3 | 53.08 | 53.16 | 56.45 | 0.217 |
| 0.3000r/s | 1256.5 | 1283.2 | 59.13 | 59.46 | 60.81 | 0.233 |
| 0.3333r/s | 1287.0 | 1323.4 | 64.65 | 64.98 | 65.02 | 0.25 |
| 0.3667r/s | 1287.0 | 1336.7 | 69.80 | 72.74 | 67.13 | 0.25 |

## Delta vs baseline (mixed=64, `2026-04-17-bench-guidellm-qwen3-4b-infer-l4-p99.md`)

| metric | baseline (64) | now (256) | Δ |
|---|---:|---:|---:|
| sync TTFT p50 (ms) | 850.1 | 836.0 | -14 (-1.7%) |
| sync TTFT p99 (ms) | 871.4 | 847.0 | -24 (-2.8%) |
| sync ITL p50 (ms) | 35.3 | 35.2 | -0.1 |
| sync out tok/s | 26.3 | 26.4 | +0.4% |
| **throughput out tok/s** | **97.91** | **84.1** | **-14%  REGRESSION** |
| throughput ITL p99 (ms) | 101.81 | 85.47 | -16% (noise; rate differs) |
| **@0.135r/s TTFT p99 (ms)** | **1233.5** | **1614.4** | **+31%  REGRESSION** |
| @0.135 ITL p99 (ms) | 40.94 | 40.93 | same |
| @0.205 TTFT p99 (ms) | 1251.4 | 1240.7 | -0.9% |
| @0.277 TTFT p99 (ms) | 1289.4 | 1258.3 | -2.4% |
| @0.313 TTFT p99 (ms) | 1301.6 | 1283.2 | -1.4% |
| @0.348 TTFT p99 (ms) | 1325.3 | 1323.4 | -0.1% |
| @0.383 TTFT p99 (ms) | 1349.5 | 1336.7 | -0.9% |

## Interpretation

### Peak-throughput regression (-14%)

Qwen3's batched decode allocates buffers sized
`max_tokens = max_batch_size + MIXED_PREFILL_CAP`. At 64 that's
10+64=74 tokens. At 256 it's 10+256=266 tokens. **But CUDA Graphs are
only captured for batch_size ∈ [1,10]** with max_tokens ≤ 74. A mixed
forward that packs 10 decode + 256 prefill = 266 tokens cannot replay
from a captured graph and runs live. Live forward is slower than
replayed graph → throughput drops.

### Low-rate TTFT p99 regression (+31%)

Single large mixed-prefill step = ~150ms (observed in scheduler step
breakdown logs). When decode batch size is 1 and a new request arrives,
the first 256-token prefill chunk stalls decode for ~150ms before the
next decode step can land. At 0.135 r/s that's disproportionately
visible in the long tail.

### Mid-rate TTFT (no change)

Above ~0.2 r/s, decode batch saturation dominates scheduler cadence,
and whether each mixed step packs 64 or 256 prefill tokens matters
less for p99 TTFT. Marginal improvement (-1 to -2%) is noise.

### What this rules out

The 420ms post-overlap TTFT p99 gap vs sglang is **not** driven by
mixed-prefill chunk size. Increasing the chunk doesn't buy us TTFT
at the rates where the gap exists, and costs us throughput. The
gap has to be kernel-level — consistent with the 100ms single-request
4096-token prefill delta that `2026-04-17-ttft-scaling-infer-vs-sglang-l4.md`
attributes to sglang's `BatchPrefill*` wrapper vs our `SinglePrefill*`.

## Follow-ups

- **Kept at 64** as the safe default.
- **Next experiment:** FlashInfer planned-prefill wrapper
  (`docs/plans/flashinfer-planned-prefill.md`). Target the actual
  kernel-level gap instead of scheduler-chunk tuning.
- **Deferred:** if FlashInfer planned-prefill lands and MIXED_PREFILL_CAP
  retuning matters after, revisit with also-bumping CUDA Graph
  capture's max_tokens coverage to avoid the non-replay penalty.

## Rule

Before changing a scheduler constant, **first check the downstream
buffer sizing it feeds into**. `MIXED_PREFILL_CAP` feeds
`max_tokens = max_batch_size + MIXED_PREFILL_CAP` which in turn feeds
CUDA Graph capture shapes. Bumping the cap without also expanding
graph capture silently falls off the replay path and costs throughput.

## Artefacts

- Raw: `bench-output/2026-04-17-qwen3-4b-infer-l4-mixed256/benchmarks.json`
- HTML: `bench-output/2026-04-17-qwen3-4b-infer-l4-mixed256/benchmarks.html`
