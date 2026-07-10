# Qwen FP8 small-M dense GEMM — DeepGEMM from M=2 (measured crossover), M=1 attributed

> Status: Licensed 2026-07-10. Routing fix landed; M=1 GEMV kernel variants killed with attribution.

## Context

Task: dense_ffn ~26 ms/step profiled at M=16 vs ~3.2 ms "weight-read floor";
plain decode 23 ms/tok vs "~7 ms roofline" (27 GB FP8 / 4 TB/s H20). Plan:
[dspark plan §Next wall](../../plans/2026-07-09-dspark-dflash-spec-decode-qwen36.md).
Harness: `crates/infer-cuda/examples/fp8_smallm_gemm_probe.rs` (CUDA events,
200 iters warmed, deterministic non-NaN fills) + one nsys pass; H20, GPU-idle,
Qwen3.6-27B dense shapes.

## Ceilings first (the "4 TB/s roofline" premise was wrong)

| measurement | rate |
|---|---|
| HBM theoretical (2619 MHz x 6144-bit) | 4.02 TB/s |
| torch 2 GiB read-sum (achievable read) | **3.50 TB/s** |
| DtoD copy (bytes moved) | 2.0-2.1 TB/s (copies are not the read ceiling) |
| wread probe: GEMV pattern, x-work removed | 2.9-3.7 TB/s |
| wread probe: + fp8->f32 decode | 2.8-3.0 TB/s |

So the honest per-token decode floor is ~27 GB / 3.5 TB/s ≈ 7.7 ms of pure
GEMM, and the dense_ffn @M=16 floor is ~4.9 ms/step (64 x 267 MB / 3.5), not 3.2.

## Micro-bench: GEMV vs DeepGEMM per M (cuda_us, 200 iters)

ffn_gate_up = 17408x5120, ffn_down = 5120x17408, attn_sq = 5120x5120.
dg = DeepGEMM dense kernel; +pack = activation quantize (3.1-9.0 us, add it).

| M | gate_up gemv | gate_up dg | down gemv | down dg | attn gemv | attn dg |
|---|---|---|---|---|---|---|
| 1 | **50.1** | 47.6 (+3.2 pack) | **49.8** | 58.7 | **16.6** | 18.8 |
| 2 | 80.6 | **47.5** | 87.8 | **58.0** | 23.8 | **19.1** |
| 4 | 117.5 | **47.5** | 138.6 | **58.1** | 35.8 | **18.6** |
| 8 | 215.2 | **47.4** | 262.1 | **57.7** | 64.1 | **18.6** |
| 16 | 424.7 | **47.5** | 504.2 | **57.8** | 121.1 | **18.6** |
| 17 | 507.6 | **47.5** | 590.0 | **57.8** | 141.7 | **18.6** |
| 32 | 838.7 | **47.5** | 966.2 | **57.5** | 237.3 | **18.6** |

- DeepGEMM dense is FLAT in M (one weight pass, 1.4-1.9 TB/s); the tiled GEMV
  scales ~linearly (M=2 already 1.6-1.8x a single decode).
- **Crossover = M=2** -> `QWEN_FP8_DEEPGEMM_DENSE_MIN_M` 16 -> 2
  (`quant_linear.rs`). M=1 stays on the GEMV (wins 2 of 3 shapes incl. pack).
- Pre-Hopper dequant->cuBLAS fallback keeps its own floor 16
  (`QWEN_FP8_DEQUANT_GEMM_MIN_M`): full-weight dequant never pays at tiny M.

## Host-overhead (memoize) hypothesis — KILLED by measurement

Per-call DeepGEMM bridge host cost (config search + codegen string + digest +
2x fs-exists): host_us = **10.4** vs device 47.5 us; nsys: cuLaunchKernelEx
3.7 us avg, kernels flat (45.9/56.5/17.8 us, stddev <0.5%). Device-bound;
memoizing `(m,n,k)`->runtime saves ~7 us/call of already-overlapped host time.
Not implemented.

## M=1 (plain decode lever) — variants killed, gap attributed

Legacy B=1 GEMV runs 1.78 TB/s; identical pattern with x-work removed runs
2.9 TB/s -> the per-row x load+convert tail is the whole gap. Two informed
fixes both measured SLOWER and were killed:
- smem-staged x, bank-conflict-free padded layout: 89.2 vs 49.8 us (ffn_down).
  An unpadded float4 layout was 4x worse (32-way conflicts). LDS wavefronts
  cost the same as L1 hits; staging adds work without removing wavefronts.
- x-in-registers 4-row tile (x converted once per 4 rows, scale hoisted):
  62.2 vs 49.9 us.

Residual, reported honestly: M=1 GEMV sits at ~51% of achievable read BW;
plain decode stays ~23 ms/tok. Next credible lever is a Marlin-style
tensor-core W8A16 FP8 kernel port (vLLM `marlin_fp8`), not GEMV tuning —
recorded in the plan. Diagnostic kept: `fp8_wread_probe` (probe-only).

## E2E (H20 GPU-idle, 500-tok csv/rust prompts, fresh nonce)

Matched single-variable A/B, same tree + same day, only MIN_M differs:

| lane | MIN_M=16 control | MIN_M=2 | delta |
|---|---|---|---|
| dspark greedy csv (3 runs) | 153.5-160.1 tok/s | 158.6-174.6 | **+5-9%** |
| dspark greedy rust (3 runs) | 101.4-104.4 | 103.4-110.0 | **+2-5%** |
| plain greedy csv/rust | n/a (M=1 path untouched) | 40.4-41.8 wall (~42.5 net) | ~= 42.6-43.6 anchor |
| dspark sampled csv / rust | — | 72.1-147.0 / 79.9-80.8 | within 64-106 anchor band+ |

Note: the 07-10 wins anchors (104-108 dspark greedy) predate other landed
work (partial-ctx drafting etc.); the matched control (~155 csv) is the valid
baseline — cross-day anchor deltas would have over-credited this change.

## Gates

- Plain lane: needle 738291 x3 **exact**, DET (NEEDLE_MAX_TOKENS=768 per the
  think-preamble trap; the default 16 false-fails 3/3).
- DSpark lane: needle 738291 x3 **exact** DET, run twice (same-config-twice
  both exact/DET).
- Probe numeric check: deterministic non-NaN fills, both routes finite.

## Rule

- A "weight-read floor" must name its measured achievable BW, not the spec
  sheet: 4.02 theoretical vs 3.50 read vs 2.0 DtoD on the same H20.
- GEMV-vs-GEMM crossovers are measured per shape, not assumed: the amortizing
  tiled GEMV never beats one flat DeepGEMM weight pass beyond M=1.
- Attribute before optimizing: the wread probe (x-work removed) killed two
  plausible kernel fixes for the price of one diagnostic.
