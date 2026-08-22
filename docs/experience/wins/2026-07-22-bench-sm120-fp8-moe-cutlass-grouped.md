# sm_120 FP8 MoE prefill — CUTLASS grouped GEMM replaces hand-GEMV (G2) — CUDA (RTX PRO 6000), 2026-07-22

> Status: Shipped (G2 — the sm_120 MoE grouped FP8 prefill lever)

## Goal

Replace the pathological hand-grouped GEMV fallback (~99.5% of sm_120 FP8 MoE
prefill; baseline ~35 tok/s) with the CUTLASS 4.3.5 sm_120a grouped
blockwise-scaling FP8 tensor-core collective, on `Qwen/Qwen3.6-35B-A3B-FP8`.

## What shipped

The sm_120 replacement for the Hopper-only DeepGEMM
`m_grouped_fp8_gemm_nt_contiguous`:
- `crates/cuda-kernels/csrc/gemm/fp8_moe_grouped_cutlass_sm120.cu` — extern-C
  grouped GEMM wrapping the CUTLASS `Sm120BlockwiseScaleConfig<1,128,128>`
  collective (type stack from example 87c), persistent device scratch,
  per-group geometry from device `group_offsets`/`group_counts`.
- build.rs gates the CUTLASS instantiation + `compute_120a,sm_120a` gencode behind
  an sm_120 target; other builds link a `cudaErrorNotSupported` stub.
- Loader builds the contiguous grouped FP8 caches on sm_120 (bypassing the
  Hopper-only DeepGEMM preflight) and transposes weight scales to CUTLASS's
  N-contiguous SFB layout at load. Dispatch routes both expert GEMMs to the
  CUTLASS wrapper; DeepGEMM warm-prefill skipped on sm_120.

## Scale-layout resolution (the one real integration risk)

**Matched by construction — no runtime repack debugging.** Source-verified:
- SFA (activations): DeepGEMM's per-token packing `k_block*scale_stride_m + row`
  is consumed directly by building CUTLASS `LayoutSFA` with the K-block stride =
  `scale_stride_m` (not Mg) and `ptr_SFA[g] = sfa + group_offset[g]`. No repack.
- SFB (weights): checkpoint `weight_scale_inv` is `[n_blocks,k_blocks]` K-contiguous;
  CUTLASS `majorSFB=MN` wants N-contiguous (`n_block + k_block*n_blocks`) — a
  per-expert transpose done once at load (weights static).
- Weights B: checkpoint row-major `[N,K]` matches the collective's InternalStrideB
  (`Stride<int64_t,_1,_0>`, leading=K) directly — no repack.

## Correctness gate — PASS

`scripts/needle_gate.py` (RAW=1 TEMPLATE=qwen3_nonthink), lengths 115..8000, ×3
same-config repeats, on the real checkpoint through the CUTLASS route:

```
SUMMARY len=115  exact=3 partial=0 miss=0 DET
SUMMARY len=241  exact=3 partial=0 miss=0 DET   (spans the historical boundary)
SUMMARY len=1000 exact=3 partial=0 miss=0 DET
SUMMARY len=2000 exact=3 partial=0 miss=0 DET
SUMMARY len=4000 exact=3 partial=0 miss=0 DET
SUMMARY len=8000 exact=3 partial=0 miss=0 DET
```

Exact needle recall, deterministic at every length — the CUTLASS grouped output's
own autoregressive generation is coherent and correct. The scale layout is right.

## Perf — bench vs baseline

Baseline: `2026-07-22-bench-sm120-fp8-moe-baseline.md`, cold prefill ~85 s / 3013
tok = **~35 tok/s** on the hand-grouped GEMV fallback; c=16 collapsed (0/16).

Canonical `bench_throughput.py`, same params/model/box, 64 prompts (p50 ~3 k tok),
120 s/concurrency, max_tokens 256, seed 20260416. `error=0`, `correctness_failed=0`
at every point. Raw: `bench-output/2026-07-22-sm120-fp8-moe-cutlass/bench.csv`.

| c | arm | complete | TTFT p50 ms | TTFT mean ms |
| --: | ----- | ---------: | ------------: | -------------: |
| 1 | baseline (GEMV) | 3 | **84,634** | 84,634 |
| 1 | **CUTLASS grouped** | 45/46 | **760** | 751 |
| 4 | baseline | 6 | 119,293 | — |
| 4 | **CUTLASS** | 98/100 | 4,306 | 3,731 |
| 8 | baseline | 8 | 175,404 | — |
| 8 | **CUTLASS** | 151/154 | 6,708 | 5,600 |
| 16 | baseline | **0/16 (collapsed)** | n/a | n/a |
| 16 | **CUTLASS** | 202/206 | 9,749 | 8,935 |

**Δ prefill (the lever): c=1 cold TTFT 84,634 → 760 ms = 111× faster** (~3 k-tok
prefill ~35 → ~3960 tok/s). c=16 went from full collapse to 202/206 complete.
Decode ITL stays healthy
(c=1 p50 11.3 ms). The GEMV→tensor-core-grouped change is the entire win.

## Environment

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, sm_120, 96 GB, CUDA 12.8.
- Model: `Qwen/Qwen3.6-35B-A3B-FP8`, world=1, BF16 KV.
- Build: `TORCH_CUDA_ARCH_LIST=12.0 cargo build --release --features cuda`.

## Rule

The sm_120 FP8 MoE grouped GEMM is the CUTLASS sm_120a blockwise collective, not
DeepGEMM (Hopper-only) and not the hand GEMV (prefill-pathological). Scale layouts
are matched to the collective by source analysis (SFA custom stride, SFB load-time
transpose), verified by the needle gate on the real checkpoint — not by hoping the
DeepGEMM and CUTLASS packings coincide.
