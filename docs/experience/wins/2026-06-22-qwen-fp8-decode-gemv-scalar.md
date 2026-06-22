# Qwen3.6-27B-FP8 decode 5.5× — the MMA GEMV was occupancy-starved at B=1

**Commit:** 7669ff69 (`perf(qwen35): default FP8 decode GEMV to coalesced scalar kernel`).
**Measured:** Qwen3.6-27B-FP8, 1×H20, `arle serve` (no-think prompt, max_tokens=200, ignore_eos).

## Context

ckl flagged 27B-FP8 decode as "比 Mac 还差". Measured single-stream **7.5 tok/s**
— ~19× off the H20 memory roofline (~140 tok/s for a 27B-FP8 weight read at
~4 TB/s). §0 profile (`ARLE_QWEN35_QUANT_PROFILE`) attributed the entire step to
the FP8 block-scaled GEMVs.

## Root cause (measured, not inferred)

The decode GEMV ran the **`mma.m16n8k16` tensor-core path**
(`gemv_fp8_block_scaled_batch_mma`), which was built for **DSv4 batched decode
(B≤16)** and is the wrong tool at **B=1**:

- **Occupancy-starved:** `grid = (N/32, B/16)` → at B=1 only `N/32` blocks
  (160–544) on the H20's 132 SMs (~25% occupancy). Bandwidth scaled *linearly*
  with block count — the smoking gun:

  | proj | N | blocks | GB/s | % of 4 TB/s |
  |------|---|--------|------|-------------|
  | gate/up | 17408 | 544 | 341 | 9% |
  | down | 5120 | 160 | 126 | 3% |

  gate and down read the **same 89 MB** but down (160 blocks) is 2.7× slower —
  fully explained by block count, not bytes.
- **Uncoalesced:** the MMA col-major B-operand makes each warp read 8 different
  N-rows (stride K), scattering HBM transactions.

Both come from forcing a batched-GEMM MMA onto a B=1 GEMV. Graph (`ARLE_QWEN35_DECODE_GRAPH=1`)
and launch overhead were **confirmed washes** (identical 7.5 tok/s) — it is
GPU-bound on the kernel, not host.

## Fix

The repo already had a **coalesced scalar warp-per-row** FP8 GEMV
(`dsv4_fp8_gemv_kernel` / `dsv4_fp8_gemv_batch_kernel`, `quantized_gemv.cu`):
`grid = N/GEMV_ROWS` (~16× more blocks) and threads stride K reading consecutive
bytes (coalesced) + warp-reduce. Flipped the Qwen FP8 decode dispatch
(`quant_linear.rs`) to default to it; MMA is now opt-in (`ARLE_QWEN35_FP8_GEMV_MMA=1`).

## Measured

| | MMA (old default) | scalar (new default) | speedup |
|---|---|---|---|
| single-stream | 7.5 tok/s | **41.3 tok/s** | **5.5×** |
| conc=8 | 17.6 tok/s | **50.4 tok/s** | **2.9×** |

Scalar GEMV bandwidth jumped from 3–9% to **32–42%** (down-proj 0.705→0.053ms =
13×; gate 0.262→0.054ms = 4.9×). (The A/B's 27 tok/s was 3-serve contention;
isolated default is 41.3.)

Correctness: the scalar kernel **is the reference** the MMA was built to match
(cosine 1.0 / max_abs 0); decoded output coherent (17×23 = 391). Parity-safe flip.

## Rule

A "fast" tensor-core kernel for batched decode (B≤16) can be a **5× regression**
at B=1 — the MMA's batch tile + col-major weight layout starve occupancy and
coalescing. For a memory-bound B=1 GEMV the right kernel is coalesced
warp-per-row with `N/rows` blocks. Always profile the **actual decode shape**;
the block-count↔bandwidth correlation is the occupancy fingerprint.

## Follow-ups (in progress — multi-front, ckl "多线开花")

- **GEMV vectorization** 41%→60-80% BW (1-byte→uint32/uint4 loads).
- **linear-attn `in_proj`** 0.176ms/layer outlier — likely off the scalar path.
- **lm_head** 1.37ms/step (vocab 248320) — FP8/vectorize.
- **norm/RoPE/residual** fusion to cut B=1 launches.
- **Deletion-refactor**: remove the dead MMA Qwen path (collapse to one scalar path).
