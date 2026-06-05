# Per-projection DeepGEMM dense for DSv4 FP8 attention linear — no wall-clock win

## Context

ncu showed the DSv4 attention-side FP8 block-scaled linear
(`dsv4_fp8_gemv_batch`, the wq/wkv/wo + compressor + HC projections) runs as a
**scalar** CUDA-core kernel — tensor pipe <1%, ~10% HBM BW (roadmap §9). The
SGLang H20 A/B (15.89 ms no-spec vs ARLE 39.5 ms) confirmed a real 2.5× kernel
gap, and SGLang's FP8 dense GEMM (4.94 ms/token) uses `deep_gemm.fp8_gemm_nt`
(`sm90_fp8_gemm_1d2d`, WGMMA). The obvious move: replace the scalar kernel with
DeepGEMM dense, per projection.

## Root cause / result

Wired DeepGEMM dense (`dsv4_deepgemm_fp8_gemm_nt`, native bridge) behind each MLA
linear, gated by `ARLE_DSV4_FP8_LINEAR_DEEPGEMM`. Verified on 8×H20, DeepSeek-V4-Flash:

- **Numerically correct** — same `clean_tokens` as scalar (per-projection max_abs
  to 0.07–0.13 on wo_a/wo_b, bf16/quant noise, no token drift).
- **No wall-clock win** — 512 prefill **4.5% slower** (5224 vs 4972 ms), 5-tok
  prefill 9079 vs 6761 ms (slower), 2048 only 0.8% faster, decode ~flat.

The cause is the **call form, not the kernel**: a single prefill issues ~344
DeepGEMM calls (5 projections × 43 layers + compressor/HC), each paying its own
launch overhead **and** its own BF16→FP8 `pack_quantize` of the activation. That
per-call overhead eats the WGMMA tensor-core speedup. SGLang's 4.94 ms is the
*fused/scheduled* form (qkv-fused, activation quantized once and reused, batched
across the projection set) — not 344 separate `fp8_gemm_nt` calls.

## Fix / direction

Killed the per-projection wiring (not committed — restored the `seq_len<=1` skip,
no half-state). The real lever is the fused call organization: fuse same-input
projections into fewer/larger GEMMs and quantize the shared activation once. That
is a restructure of the MLA-linear forward, not a kernel swap — deferred behind
the higher-ROI **FlashMLA decode** lever (ARLE hybrid attention ~11.8 ms vs SGLang
2.02 ms is the bigger ~10 ms/token decode gap, and a well-defined kernel port vs
an uncertain fusion rewrite).

## Rule

A scalar kernel that ncu shows at <1% tensor / ~10% BW is a **kernel-licensed**
target, but swapping it for a tensor-core library kernel **per existing call site**
is not pipeline-licensed: if the op is invoked hundreds of times per forward (per
projection × per layer), per-call launch + per-call activation quant dominates and
erases the tensor-core win. The upstream (SGLang) advantage is the *fused/batched
call form*, not the kernel in isolation. Before swapping, count the call sites per
forward and check whether the win requires fusing them — otherwise a numerically
correct swap ships ~0% (or a regression). Same family as the launch-overlap and
NVTX-sync framing traps: the narrow per-kernel metric (tensor-pipe %) is not the
wall-clock lever until the call structure matches the reference.
