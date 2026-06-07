# DSv4 decode CLEAN critical-path profile: ARLE glue is ~half the GPU time; mhc_params is #1

## Context
First UN-confounded decode kernel profile (nsys rank-0, LOAD separated from DECODE). Prior
profiles misled: synced stage_profile inflates overlapped stages; whole-trace gpukernsum was
LOAD-dominated; post-load window had harness-idle. This is the clean per-forward breakdown.

## Per-forward decode GPU time (40 forwards, flashmla_fused_wqkv, B=1, 8×H20 TP=8)
LOAD (one-time, EXCLUDED): block_scaled_to_fp8_cache scales+values = 590ms total (weight build).

DECODE kernels (total ms ÷ 40 forwards):
| kernel | ms/fwd | inst/fwd | owner |
|---|---:|---:|---|
| **dsv4_mhc_params** | **3.05** | 86 | **ARLE** (#1 — 35.5µs each, single-block thread0-serial tail) |
| ncclAllReduce | 1.30 | 86 | vendored (NCCL) |
| nvjet (cublas) | 0.89+0.42 | 167+86 | **ARLE** (un-fused MLA proj, task #36) |
| deep_gemm sm90_fp8 (4 variants) | ~2.5 | 43 ea | vendored |
| dsv4_deepgemm_pack_quantize | 0.83 | **258** | **ARLE** |
| get_mla_metadata | 0.67 | 42 | vendored |
| flash_fwd_mla (FlashMLA) | 0.64 | 42 | vendored |
| ncclAllGather | 0.55 | 43 | vendored |
| splitKreduce (cublas) | 0.48 | **274** | **ARLE** (un-fused MLA proj) |
| rms_norm_batched | 0.44 | 172 | **ARLE** |

## Rule / the evidence-grounded 6ms roadmap
- **ARLE-owned glue ≈ half the decode GPU time**: mhc_params 3.05 + cublas-MLA ~1.8 +
  pack_quantize 0.83 + rms 0.44 ≈ **6ms/forward**. Vendored (DeepGEMM+FlashMLA+AR) ≈ 4.4ms.
  The optimization target is the ARLE glue (not the vendored floor).
- **#1 = dsv4_mhc_params (3.05ms, 86/forward).** Single-block (num_tokens=1 → 1 block),
  with `if (threadIdx.x != 0) return;` → the Sinkhorn + pre/post/comb tail runs on THREAD 0
  ALONE. 35.5µs for a tiny 4×4 HC-routing matrix is wildly inefficient — structural.
- **~440 of ~860 kernels/forward are cublas (nvjet 167+86, splitKreduce 274)** — the 3/4
  un-fused MLA projections. Porting to DeepGEMM (task #36) cuts BOTH ~1.8ms AND ~440 kernels
  (→ less GPU-side per-kernel latency on the serial chain).
- This supersedes "6ms = vendored frontier only": HALF the decode GPU time is ARLE glue with
  clear inefficiencies (single-block thread0-serial mhc_params; cublas M=1 MLA projections).
  6ms is achievable (active-weight floor ~0.3ms; 26ms forward is ~87× it). Levers, in order:
  mhc_params kernel rewrite, cublas-MLA→DeepGEMM, pack_quantize fusion — then vendored+MTP.
