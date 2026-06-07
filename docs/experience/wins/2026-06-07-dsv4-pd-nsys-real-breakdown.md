# DSv4 prefill+decode real per-kernel breakdown (nsys, no stage-profile sync)

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

**Prefill @4000 warm — nsys per-kernel (the `mla_attn` 82% decomposed).**
Captured the warm (rep1) prefill forward via a per-rep `cudaProfilerStart/Stop`
gate (`INFER_DSV4_AB_PROFILE_PREFILL`, fires on the 2nd prefill). nsys cuda-trace
inflates the Instant wall ~5× (17481ms traced vs ~3508ms real), but
`cuda_gpu_kern_sum` measures true kernel GPU time: total ≈ **3340ms ≈ the real
3508ms wall → ~95% GPU-busy (prefill is GPU-compute-bound, no launch gap, the
opposite of decode's 34%).**

| kernel | ms | % | note |
|---|---:|---:|---|
| `dsv4_fp8_gemv_batch_tiled` | **2077** | **62.2** | **attention/dense projection GEMV (self-written, CUDA-core not tensor-core) — the prefill bottleneck** |
| `dsv4_compressor_update` | **552** | **16.5** | DSA KV-block compressor |
| `deepgemm sm90_fp8_gemm` ×N | 236 | 7.1 | routed MoE + the one `wq_a\|wkv` projection that IS on DeepGEMM |
| MoE `pack_quantize`+`swiglu_quantize` | 114 | 3.4 | FP8 companion |
| mHC (`post`+`params`+`pre`) | 110 | 3.3 | hyper-connection |
| **`sparse_attn_fwd`+`swa_attention`** | **110** | **3.3** | **the actual FlashMLA sparse-attention math — small, vendored, already efficient** |
| comm (allgather+allreduce) | 52 | 1.6 | |
| DSA select + glue (prepare_q/k, tp_q_repack, build_indices, paged_mqa, rope) | ~70 | 2.1 | |

**Verdict: `mla_attn` 82% = projections (62%) + DSA compressor (16.5%), NOT the
attention math (3.3%).** `FP8_LINEAR_DEEPGEMM` default-on routes only the
`wq_a|wkv` fusion through DeepGEMM; the rest of the projections (wq_b, wo, indexer
wq_b, …) still run the self-written `fp8_gemv_batch_tiled`, so the bulk of
projection FLOPs stay on CUDA-core GEMV → 62%. Matches H20 being compute-starved:
the inefficiency is running GEMM-shaped work (M=4000) as a tiled GEMV.

**Cross-cutting:** `dsv4_fp8_gemv_batch[_tiled]` is the #1 self-written cost in
BOTH prefill (62%) and decode (3.6ms, #1 GPU). The FlashMLA/DeepGEMM vendored
kernels are already efficient in both. One lever — route ALL projections through
DeepGEMM/tensor-core — hits both P and D.

## Rule

For DSv4 P/D wall-clock + per-op attribution, use `nsys
--capture-range=cudaProfilerApi` on a **`stage_profile`-OFF** run (decode: profiler
fires after warmup; prefill: per-rep gate on the warm forward), NOT the in-tree
per-stage CUDA-event profiler — its `stop.synchronize()` inflates the B=1 decode
wall ~1.7× and its `mla_attn` stage is a composite that hides the real split. Read
GPU time from `cuda_gpu_kern_sum`.

The real bottleneck in BOTH P and D is the **self-written projection GEMV
`dsv4_fp8_gemv_batch[_tiled]`** (prefill 62%, decode #1 GPU 3.6ms), plus the
**mHC Sinkhorn `dsv4_mhc_params`** (decode #2, 3.1ms) and the **DSA compressor**
(prefill 16.5%). The vendored **FlashMLA attention math is NOT a bottleneck**
(prefill 3.3%, decode ~1.85ms) — don't optimize it. Prefill is GPU-compute-bound
(~95% busy, H20 compute-starved); decode is ~66% GPU / ~34% B=1 launch-gap.
Optimization order: (1) **route ALL projections off `fp8_gemv` → DeepGEMM/tensor-core**
(hits P and D; `FP8_LINEAR_DEEPGEMM` today only covers `wq_a|wkv`), (2) mHC fuse
(SGLang `mhc_pre` TileLang — decode), (3) DSA compressor (prefill), (4) batched
decode (摊薄 ~10ms decode launch gap), (5) comm-overlap (secondary).
