# DSv4 decode real per-kernel breakdown (nsys, no stage-profile sync)

## Context

Profiling DSv4 P/D to answer "decode/prefill 时间花在哪". First pass used the
in-tree `stage_profile` (CUDA-event per stage). It reported decode = **48-51
ms/token** with a **~16ms "unaccounted/host gap"** and `mla_attn` (12.4ms) as
the dominant stage. Two of those conclusions were **measurement artifacts** —
see [`../errors/2026-06-07-stage-profile-per-stage-sync-inflates-decode.md`](../errors/2026-06-07-stage-profile-per-stage-sync-inflates-decode.md).
This is the clean re-measure with `nsys --capture-range=cudaProfilerApi` on a
run with `stage_profile` OFF (no per-stage `stop.synchronize()`).

**Config (code-verified):** infer-cuda rewrite tree @`1e2ae52d` (official DSA +
FlashMLA decode/prefill + FP8 DeepGEMM default-on), built `release-fast`
`cuda,nccl` arch 9.0, 8×H20 TP=8/EP=8, DeepSeek-V4-Flash (**43 layers**, FP8/FP4,
149GB). Driver: `dsv4_resident_ab --example`, variant `flashmla_fused_wqkv`
(fastest decode: flashmla + fused_wqkv ON, masked MoE), MoE transport =
allreduce (infer-cuda default). 4000-token prompt + 80 decode, 16 warmup, **63
steady steps captured**, rank0 under nsys `-t cuda,nvtx` (no osrt).
**Caveat:** synthetic prompt (`(i*7919+13)%120000+100`) → degenerate output
(`oracle16=FAIL`); timing valid, MoE expert-routing distribution differs from
real text (affects MoE/route shares only).

## What Worked

**Real decode = 29.0 ms/token** (`steady_tok_s=34.4`, 1829ms/63). nsys adds
~%-level overhead; the non-profiled warmup steps and the independent
2026-06-07 win both read ~26ms, so the clean decode is **~26-29 ms/token** — NOT
the 48-51ms the stage profiler reported.

**~66% GPU-busy, ~34% host-launch gap.** nsys `cuda_gpu_kern_sum` total over the
window ≈ **1202ms = 19.1 ms/token GPU**; window wall = 1829ms → **~10 ms/token
(34%) is GPU-idle between kernels** (host launch latency; comm is NOT overlapped
by default so kern_sum ≈ GPU-busy-wall). Batched decode amortizes this ~10ms (not
the 16ms the artifact suggested).

**Real per-kernel decode GPU time (nsys `cuda_gpu_kern_sum`, ms/token = Total ÷ 63):**

| kernel | ms/tok | % GPU | instances | note |
|---|---:|---:|---:|---|
| `dsv4_fp8_gemv_batch_kernel` | **3.62** | 18.9 | 9450 | attn/dense projection GEMV (scalar, <1% tensor pipe) — #1 |
| `dsv4_mhc_params_kernel` | **3.06** | 16.0 | 5418 | hyper-connection Sinkhorn `<<<T,256>>>` thread0-serial — #2 |
| `deepgemm sm90_fp8_gemm` ×5 | **~2.74** | ~14 | 2709 ea | routed MoE + shared expert FP8 grouped GEMM |
| `ncclDevKernel_AllReduce` | 1.29 | 6.8 | 5418 | TP comm |
| `nvjet`/`cublasLt splitK` (BF16 dense) | ~1.9 | ~10 | — | router gate / dense parts |
| `flash_fwd_splitkv_mla_fp8_sparse` + `get_mla_metadata` + `mla_combine` | **~1.85** | ~10 | 2709 | **the actual sparse-attention math — small!** |
| `gemv_handwritten_kernel` | 0.41 | 2.1 | 63 | lm_head vocab GEMV (406µs ×1/step) |
| `ncclDevKernel_AllGather` | 0.55 | 2.9 | 2709 | TP Q all-gather |
| `dsv4_deepgemm_pack_quantize` + `swiglu_quantize` | ~0.70 | ~3.7 | 8127/5418 | MoE FP8 companion |
| `rms_norm_batched` | 0.44 | 2.3 | 10836 | |
| `dsv4_route` + compressor + kv_pack + glue | ~1.3 | ~7 | 2709 ea | route/counts/pack/scatter/combine, compressor, fp8_kv_pack |

**Grouped (ms/token):** projection GEMV **4.0** · mHC family **3.4** · routed-MoE
DeepGEMM (incl quant+glue) **~4.3** · dense BF16 GEMM **~1.9** · comm **1.84** ·
FlashMLA attention math **~2.3** · norm/compressor/kv **~1.2**. Sum ≈ 19 ms/token
GPU + ~10 ms/token launch gap = ~29 ms/token wall.

**Prefill @4000 warm = 3508 ms** (stage-profile composite, sync negligible at
ms-scale stages, matches the 2026-06-07 win 3.48s): `mla_attn` **82%**, routed-MoE
**12%**, HC/shared/comm/norm **6%**. Per-kernel sub-split of prefill `mla_attn`
not yet nsys-traced (decode-only nsys this pass).

## Rule

For DSv4 **decode** wall-clock + per-op attribution, use `nsys
--capture-range=cudaProfilerApi` on a **`stage_profile`-OFF** run, not the
in-tree per-stage CUDA-event profiler (its `stop.synchronize()` ×hundreds/token
inflates the B=1 wall ~1.7×). The real decode is **GPU-bound ~66% / launch-gap
~34%**; the dominant GPU kernels are the **scalar projection GEMV
(`dsv4_fp8_gemv_batch`)** and the **mHC Sinkhorn (`dsv4_mhc_params`)**, NOT the
FlashMLA attention math (only ~1.85ms). Optimization order: (1) projection
GEMV → tensor-core/DeepGEMM, (2) mHC fuse (SGLang `mhc_pre` TileLang), (3) batched
decode (摊薄 ~10ms launch gap), (4) comm-overlap (1.8ms, secondary).
