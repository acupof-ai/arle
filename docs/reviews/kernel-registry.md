# Kernel Registry — live hot-path operator variants

> **Govern gate** for [`../plans/gpu-dispatch-governance.md`](../plans/gpu-dispatch-governance.md)
> Phase 4. This table is the **data behind the operator library's `plan()`
> selection** (forward cross-link: [`../plans/backend-operator-library.md`](../plans/backend-operator-library.md),
> not yet written — this registry seeds its variant catalog). One row per
> *live* hot-path operator variant: the thing a dispatcher actually launches,
> not every `.cu` in the tree.
>
> **Required verify-phase update.** A runtime diff that adds / removes / re-wires
> a `LinearKernelPlan` variant, a `KVFormat` attention arm, an MoE-prep kernel,
> or a Metal MLX primitive is **not done until its registry row lands**
> (root `AGENTS.md` §Benchmarks verify exit). The Phase-4 license-or-kill is
> "the registry is killed if it drifts stale" — so a stale row is a CI-visible
> omission, not silent drift. Seeded from the `LinearKernelPlan` variants
> ([`../../infer/src/ops/linear.rs:64-101`](../../infer/src/ops/linear.rs)),
> the attention `KVFormat` paths, the six-principles heat map
> ([`2026-04-14-cuda-kernel-six-principles-review.md`](2026-04-14-cuda-kernel-six-principles-review.md)),
> and the kernel-vs-SOTA audit
> ([`../research/2026-05-28-arle-kernel-vs-sota-audit.md`](../research/2026-05-28-arle-kernel-vs-sota-audit.md)).

## Column contract

| Column | Meaning |
|---|---|
| **op family** | the logical operator (linear / attention / KV-quant / MoE-route / norm / …) |
| **variant** | the `LinearKernelPlan` arm or attention/MoE kernel symbol the dispatcher resolves |
| **impl type** | hand-rolled CUDA · TileLang AOT · Marlin · DeepGEMM · cuBLASLt · FlashMLA · MLX-primitive |
| **SKU / shape class** | the `(SM tier, batch class, quant, head-dim)` cell this variant serves |
| **tuned?** | hardcoded launch cfg vs autotuned. Today **only `Bf16Gemv` autotunes, and it is opt-in** (`INFER_GEMM_AUTOTUNE=1`) — every other variant is hardcoded (`../plans/gpu-dispatch-governance.md` §2.4) |
| **roofline position** | measured frac-of-peak / % of decode GPU time, with source. *hypothesis* where projected, not measured |
| **best-known alt + why not wired** | the SOTA replacement and the precise license state (`wire-it` / `needs-paired-A/B` / `killed-already` / `roofline-share-too-low`) |
| **owner** | who holds the gap |

Verdict vocabulary in the "why not wired" column matches the gap doc
([`../research/2026-05-29-oplib-sota-kernel-gap.md`](../research/2026-05-29-oplib-sota-kernel-gap.md)).

---

## Linear / GEMM — `LinearKernelPlan` (30 variants, `linear.rs:64-101`)

| op family | variant | impl type | SKU / shape class | tuned? | roofline position | best-known alt + why not wired | owner |
|---|---|---|---|---|---|---|---|
| linear | `Bf16Gemv` | hand-rolled CUDA (`gemm/gemv.cu` `gemv_handwritten_kernel`) | any SM · decode M<16 · BF16 | **autotuned (opt-in `INFER_GEMM_AUTOTUNE=1`, default OFF)** | bandwidth-bound; PASS calibrator (audit PASS-1). Vectorized bf16x4/x8 | — at SOTA; cuBLASLt sub-optimal at M<16, this is the right path | — |
| linear | `Bf16GraphsafeGemm` | cuBLASLt (`gemm/gemv.cu` `gemm_graphsafe_cuda`) | any SM · batch N=1 · BF16 · graph-capture | hardcoded (cuBLASLt algo cache, gen-counter protected) | PASS-1; identical to TRT-LLM/vLLM cuBLAS path, `CUBLAS_TENSOR_OP_MATH` | — at SOTA | — |
| linear | `Bf16CublasGemm` | cuBLASLt (`gemm/gemv.cu` `gemm_cuda`) | any SM · batch N>1 · BF16 prefill | hardcoded (cuBLASLt algo cache) | PASS-1 | — at SOTA | — |
| linear | `MarlinW4Gemm` | Marlin (`gemm/marlin_kernel.cu`, IST-DASLab port) | SM_80+ · W4A16 · all batch≥2 + decode-hybrid | hardcoded (Marlin thread_k/thread_n heuristic) | tensor-core; PASS (audit #7). vLLM/SGLang ship identical file | — at SOTA. **`MarlinW4Hybrid` small-batch fallback KILLED** (see below) | — |
| linear | `MarlinW4A8Gemm` | Marlin W4A8 (`gemm/marlin_w4a8_kernel.cu`) | SM_80+ · W4A8 · all batch | hardcoded | tensor-core; PASS (audit #7) | — at SOTA | — |
| linear | `MarlinW4Hybrid` | Marlin W4+W4A8 prefill fuse (`gemm/marlin_w4a8_kernel.cu`) | SM_80+ · W4A16 · **prefill batch>1 only**, opt-in `INFER_HYBRID_W4A8_PREFILL` | hardcoded; alignment fallback demotes via `log::trace!` (`linear.rs:139`, invisible at default log) | tensor-core prefill | — at SOTA for prefill; default OFF, env-gated | — |
| linear | `MarlinW4FP8Prefill` | Marlin W4+FP8 (`gemm/marlin_w4_fp8_kernel.cu` + `marlin_pf8/`) | SM_80+ · W4A16 · **prefill batch>1 only**, opt-in `INFER_MARLIN_W4_FP8_PREFILL` | hardcoded; alignment fallback `log::trace!` (`linear.rs:169`) | tensor-core; PF8.4 path. Decode kept on W4A8 deliberately (FP8 mma wrong lever for HBM-bound decode) | — at SOTA for prefill; default OFF, env-gated | — |
| linear | `Dsv4Fp8Gemv` / `Dsv4Fp8BatchGemv` | hand-rolled CUDA scalar FFMA (`gemm/quantized_gemv.cu` `dsv4_fp8_gemv_batch_tiled_kernel`) | SM_89/SM_90 · DSv4 FP8 block-scaled · decode B≤16 | hardcoded | **`frac_peak=0.043` (B=1,4), `0.011` (B=16)** of HBM3 on H20 — compute/issue-bound, 96-99% BW unused. L3 ≈ **16% of decode GPU time** (FP8 batch 8.6% + FP4 4.5% + FP8 tiled 1.6% + FP4 tiled 1.3%) per 2026-05-14 trace. **GAP-A (P0)** | **`mma.m16n8k16` BF16×BF16→FP32 — IMPLEMENTED in `gemm/quantized_gemv_mma.cu`, C-side dispatch wired at `quantized_gemv.cu:2636` behind `ARLE_DSV4_FP8_GEMV_MMA` (default OFF), parity-tested (`ops/tests.rs:297`, relerr≤1e-3). `needs-paired-A/B` — pod TPOT A/B (env OFF vs ON, ≥4% decode gate) still pending-remote** | GAP-A subagent |
| linear | `Dsv4Fp4Gemv` / `Dsv4Fp4BatchGemv` | hand-rolled CUDA scalar (`gemm/quantized_gemv.cu`, E2M1 LUT) | SM_89/SM_90 · DSv4 FP4 block-scaled · decode B≤16 | hardcoded | same scalar pattern as Dsv4Fp8; part of GAP-A's ~16% | same MMA port as Dsv4Fp8 once it lands. `needs-paired-A/B` (FP4 LUT dequant → BF16 then same MMA) | GAP-A subagent |
| linear | (DSv4 grouped, SM_89 fallback) `dsv4_fp8_grouped_gemm_batch_kernel` | hand-rolled CUDA scalar (`gemm/dsv4_grouped_gemm.cu`) | **SM_89 only** · DSv4 FP8 grouped MoE · prefill | hardcoded | same scalar FP32-accum as GAP-A + per-expert grid-Z. On DSv4 prefill SLO-kill path. **GAP-D** | mechanical port of GAP-A MMA + expert-indexed weight ptr. `needs-paired-A/B`; SM_89 not a DSv4 deploy target so **roofline-share medium**. On SM_90 superseded by DeepGEMM | GAP-A subagent (after GAP-A) |
| linear | (DSv4 grouped, SM_90) `dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked` | **DeepGEMM v3** wrapper (`gemm/deepgemm_native.cu`) | SM_90 · DSv4 FP8 grouped MoE | hardcoded (DeepGEMM cubin, cluster launch) | PASS-3; `cute::UMMA` + TMA-swizzle + 232KB smem. **This *is* SOTA** | — at SOTA. **Caveat: was built+cached but never branched-to** in `forward_native_deepep_routed_gpu` (`errors/2026-05-27-b33-deepgemm-not-wired-on-native-deepep.md`) — a *reachability* bug, not an optimality bug; Phase-1 counters close it | — |
| linear | `W4A16BatchGemv` (+ W2/W8 siblings) | hand-rolled CUDA (`gemm/quantized_gemv.cu`) | SM_80+ · INT W{2,4,8}A16 · decode batch 2..8 | hardcoded | CUDA-core (no mma); single-launch | Marlin (already the default for batch≥2). **`killed-already`: hybrid dispatch preferring this at batch=4 gave +60.7% ITL regression — Marlin tensor-core dominates launch overhead** (`errors/2026-05-08-r4-hybrid-dispatch-killed-batch4-decode-regression.md`) | — (closed) |
| linear | `W4A16Gemv` (+ W2/W8 siblings) | hand-rolled CUDA (`gemm/quantized_gemv.cu`) | SM_80+ · INT W{2,4,8}A16 · decode B=1 | hardcoded | CUDA-core GEMV; M=1 tensor-core advantage tiny | Marlin at M=1 untested in r4-hybrid; B=1 may keep GEMV. `roofline-share-too-low` to revisit without new evidence | — |
| linear | `Q4KGemv` / `Q4KBatchGemv` / `Q4KDequantCublasGemm` (+ Q3/Q5/Q6 K siblings) | hand-rolled CUDA dequant + cuBLASLt (`gemm/quantized_gemv.cu`) | any SM · GGUF Q{3,4,5,6}_K · B=1 / 2..8 / >8 | hardcoded | parallel-bitmask nibble extract (llama.cpp/vLLM pattern); bandwidth-bound dequant | — at SOTA for GGUF K-quant; no faster OSS path | — |
| linear | `TurboQuantGemv` / `TurboQuantDequantCublasGemm` | hand-rolled CUDA (`gemm/turboquant_weight_gemv.cu`, `quant/turboquant_fast.cu`) | any SM · TurboQuant rotated W2/W4 · decode / prefill | hardcoded | FWHT O(D log D) rotation (right algorithm). Decode-path, smallish | research-grade (QuaRot/TurboQuant); no vLLM/SGLang equivalent. **`roofline-share-too-low`** (audit GAP-minor #8) | — |

---

## Attention — `KVFormat` → kernel (no resolver; `match` duplicated across `batch_decode.rs:1220-1520`, `prefill.rs:463`, `forward.rs:461`)

| op family | variant | impl type | SKU / shape class | tuned? | roofline position | best-known alt + why not wired | owner |
|---|---|---|---|---|---|---|---|
| attn-prefill | `batch_prefill_paged_hd{128,256,64}` | **TileLang AOT** (`tools/tilelang/`) | SM_80/89 (+planned SM_90) · paged BF16 prefill · head-dim 64/128/256 | hardcoded (TileLang planner; pinned 0.1.9 for sm_89 bug) | PASS-5; lowers to `cute::SM80::mma`. **Beats SGLang TTFT? No — TTFT is the loss axis, but scheduler-bound not kernel-bound** (head-to-head wins) | — at SOTA. **Hard-fails at runtime on `(qo,kv)` pairs not AOT-compiled — only `(8,2),(16,2),(16,4)`** baked (`attention.rs:1477-1486`); no SM-tier resolver to recover | — |
| attn-prefill | `prefill_attention{,_hd256}` / `nonpaged_prefill_attention` | hand-rolled CUDA scalar warp-tile softmax (`attention/`) | any SM · **non-paged fallback** · BF16 | hardcoded | PASS-with-caveat (audit #16); fallback only, TileLang is primary | — at SOTA for a fallback; FA-v3 template would help but path is cold | — |
| attn-decode | `batch_decode_paged_hd{128,256,64}` | **TileLang AOT** (`tools/tilelang/`) | SM_80/89 · paged BF16 decode FlashDecoding split-KV | hardcoded | PASS-with-watch (audit #33). **Beats SGLang ITL +8.6% (c=4) → +21.8% (c=16)** on L4 | — at/above SOTA on SM_89 | — |
| attn-decode | `batch_decode_paged_hd128_fp8` | **TileLang AOT** | SM_80/89 · paged FP8 KV decode | hardcoded | PASS-with-watch; only fp8 KV TileLang decode path | — at SOTA | — |
| attn-decode | `decode_attention_int8_per_channel_k_partial` | hand-rolled CUDA + `cp.async` (`attention/decode_attention_quantized.cu`) | SM_80+ · INT8 KIVI per-channel-K KV · decode | hardcoded | scalar warp-shuffle softmax, **no mma**, but cp.async double-buffered (since `8afecffe`) | FlashInfer `BatchDecodeWithPagedKVCache` mma softmax. `needs-paired-A/B` (GAP-C-medium); cp.async half already done | GAP-C subagent |
| attn-decode | `decode_attention_fp8_per_channel_k_partial` | hand-rolled CUDA + `cp.async` (`decode_attention_quantized.cu`) | SM_80+ · FP8 E4M3 per-channel-K KV · decode (Qwen3.5 high-conc) | hardcoded | scalar softmax, **no mma**. **GAP-C (P0)**: cp.async-cheap-fix LANDED (`ab850f7a`, ~111 LoC mirror of INT8 sibling) | **GAP-C-cheap (cp.async) `needs-paired-A/B`** — committed, pod TPOT A/B `pending-remote` (c=16 ≥1% gate). **GAP-C-medium (`mma.m16n8k16` QK/PV softmax)** is the separate followup, not yet started | GAP-C subagent |
| attn-decode | `decode_attention_varlen_fp8` | hand-rolled CUDA (`attention/decode_attention_varlen_fp8.cu`) | SM_80+ · FP8/INT8 paged KV · mixed decode+prefill HD128 | hardcoded | FlashDecoding split-KV+merge; varlen Q | same FA-v3/FlashInfer target as GAP-C; folded into GAP-C-medium scope | GAP-C subagent |
| attn-decode | `decode_attention_turboquant` | hand-rolled CUDA (`attention/decode_attention_turboquant.cu`) | any SM · TurboQuant KV · decode | hardcoded | research-grade | no OSS equivalent. `roofline-share-too-low` | — |
| attn-decode-prep | `decode_prep_paged{,_hd256}` | hand-rolled CUDA (`attention/`) | any SM · per-head QK RMSNorm+RoPE → paged HND write | hardcoded | PASS (audit #17); bandwidth-bound, vLLM `_paged_attention_v2_prep` pattern | — at SOTA | — |
| attn-decode | `fused_gqa_attention_single_token` | hand-rolled CUDA online-softmax (`attention/fused_attention.cu`) | any SM · GQA single-token, RMSNorm+RoPE fused, **HEAD_DIM=128 hardcoded** | hardcoded | GAP-minor (audit #15); shared-mem online-softmax, no tensor cores | FA-v3/FlashInfer template on head-dim + mma. `roofline-share-too-low` on L4, matters on H100 | — |
| attn-MLA | `mla_decode` dispatch shim → TileLang / FlashMLA | dispatch table (`attention/mla_decode.cu`) → TileLang AOT or FlashMLA wrapper | SM_90 · DSv4 MLA decode | hardcoded | PASS (audit #18); real kernels are FlashMLA (out of registry scope — separate subagent) | — at SOTA (FlashMLA) | FlashMLA subagent |

---

## MoE routing / prep — `moe/dsv4_route.cu` (+ 28 supporting kernels)

| op family | variant | impl type | SKU / shape class | tuned? | roofline position | best-known alt + why not wired | owner |
|---|---|---|---|---|---|---|---|
| moe-route | `dsv4_route_kernel` | hand-rolled CUDA, **body gated on `threadIdx.x==0`** (`moe/dsv4_route.cu:328,640,…`) | any SM · DSv4 learned-bias router · per-token | hardcoded `DSV4_ROUTE_BLOCK=256` | **255/256 threads idle** — serial expert scan + selection-sort top-k. L3 ≈ **4.4% of decode GPU time** (2026-05-14 trace). **GAP-B (P0)** | **SGLang `topk_softmax.cu` (1 warp/token, warp-parallel softmax + bitonic/tournament top-k); FlashInfer `top_k_renorm_probs` (data-parallel). `needs-paired-A/B` at production E256/top8 — the warp-parallel rewrite is *untried*.** ⚠ Note: prior route kills (`errors/2026-05-16-p3-3-dsv4-route-launch-shape-kill.md`, `…-persistent-kernel-deferred.md`) tested a **different axis** (block-size tune + persistent kernel) on the **local 1B E16/top2 shape**, not this fix on production shape — do not read them as killing GAP-B | GAP-B subagent (unstarted) |
| moe-mask | `deepseek_mask_indices_by_ep` | hand-rolled CUDA (`moe/deepseek_mask_indices_by_ep.cu`) + TileLang sister | any SM · EP/TP index filter | hardcoded | PASS (audit #10); DeepSeek TileKernels algorithm | — at SOTA | — |

---

## MoE compute / MHC — `misc/`

| op family | variant | impl type | SKU / shape class | tuned? | roofline position | best-known alt + why not wired | owner |
|---|---|---|---|---|---|---|---|
| moe-mhc | `dsv4_mhc_{params,pre,post,expand,head_pre}` | hand-rolled CUDA (`misc/dsv4_mhc.cu`) | any SM · DSv4 multi-head compressor | hardcoded | GAP-minor (audit #20); per-token row/col softmax serialized over row axis (`for row in 0..n`) | one-warp-per-row parallel. `roofline-share-too-low` | — |
| moe-attn | `dsv4_swa_attention` / `dsv4_compressor_update` / `dsv4_prepare_qk_fused` (+window-update) | hand-rolled CUDA (`misc/dsv4_attention.cu`) | SM_90 · DSv4 hybrid SWA+compressor prep | hardcoded | GAP (audit #19/E); A2.0 fused window-update (−9504 launches), A2.1 fused QK-prep landed; compressor-update still separate launch | SGLang V4 `flashmla_hybrid_attention` fuses SWA+compressed. Blocked: `needs-paired-A/B` after FlashMLA subagent lands | FlashMLA subagent (tail) |

---

## Elementwise / norm / KV-quant / sampling — `misc/`, `kv/`, `quant/`

| op family | variant | impl type | SKU / shape class | tuned? | roofline position | best-known alt + why not wired | owner |
|---|---|---|---|---|---|---|---|
| norm | `rms_norm{,_batched}` / `fused_add_rms_norm{,_offset}` / `rms_norm_gated` | hand-rolled CUDA (`misc/norm.cu`) | any SM · RMSNorm + residual-add | hardcoded | PASS-4; bf16x4 + FP32-accum + bf16-rounded 2nd pass = vLLM/FlashInfer pattern, HF-exact | — at SOTA | — |
| activation | `silu_mul_native` / `dsv4_swiglu_clamped` / `add_scaled_row` / embedding | hand-rolled CUDA (`misc/elementwise_basic.cu`) | any SM · SiLU·up, scaled-add, gather | hardcoded | PASS (audit #23); bandwidth-bound, bf16x4 vectorized | — at SOTA. **SwiGLU fusion KILLED**: fusing silu+mul into one kernel moved Qwen3-0.6B OPD step +0.006% (noise); bandwidth-bound surface (`errors/2026-05-21-arle-cuda-opd-swiglu-fused-kill.md`) | — (closed) |
| mlp | `fused_mlp_intermediate` / `fused_mlp_output` | hand-rolled CUDA (`misc/fused_mlp.cu`) | small-MLP shapes (Qwen3.5 prod bypasses via cuBLASLt) | hardcoded | GAP-minor (audit #24); cuBLASLt faster at large intermediate; mostly dead in DSv4 | `roofline-share-too-low`; candidate for removal | — |
| recurrent | `gated_delta_rule_decode` / `gdr_*_batch` / `gdr_prefill_solve` | hand-rolled CUDA + TileLang AOT (`misc/gated_delta_rule.cu`, `tools/tilelang/gated_delta_rule.py`) | any SM · Qwen3.5 linear-attn layer | hardcoded (4-slice J partition, 16 warps/block) | PASS (audit #25/26); **no SOTA baseline exists** (gated-delta-rule is Qwen3.5-specific). Six-principles fixed cross-slice global re-read | — ARLE-original at SOTA | — |
| conv | `conv1d{,_decode_batch,_prefill_batch}` | hand-rolled CUDA (`misc/conv1d*.cu`) | any SM · depthwise causal Conv1d | hardcoded (kernel-size 2/3/4 template-specialized) | PASS (audit #27); Mamba2/vLLM `causal_conv1d` pattern. Six-principles removed runtime branch | — at SOTA | — |
| kv-quant | `quantize_paged_kv_fp8` / `int8` / `int4` | hand-rolled CUDA (`kv/kv_quant.cu`) | any SM · BF16→FP8/INT8/INT4 paged write, per-channel K / per-token V | hardcoded (warp-per-token-per-head) | PASS-mostly (audit #11); FlashInfer/vLLM kv-quant recipe | — at SOTA. **K/V two-call fusion `roofline-share-too-low`**: paired component bench showed no two-launch penalty under runtime sync framing (`errors/2026-05-12-fp8-kv-pair-quantize-fusion-no-license.md`) | — (closed) |
| kv-copy | `paged_kv_append` / `scatter_kv` / `kv_cache_to_paged` / `paged_kv_metadata` | hand-rolled CUDA (`kv/`) | any SM · per-decode append / prefill scatter / pool convert | hardcoded | PASS (audit #12/13); FlashInfer `append_paged_kv_cache` pattern, bandwidth-bound | — at SOTA | — |
| kv-tier | `transfer_kv_pages_layer_table` | hand-rolled CUDA (`kvcacheio/transfer.cu`, **SGLang port**) | any SM · DRAM↔HBM page copy | hardcoded | PASS-2; line-by-line SGLang port, `ld.global.nc.b64`/`st.global.cg.b64` non-temporal. At HW BW limit | — at SOTA | — |
| sampling | `argmax_kernel_fast` / `argmax_batch` / `gpu_sample` (top-k+top-p) | hand-rolled CUDA (`misc/sampling.cu`) | any SM · greedy / top-k+top-p sample | hardcoded | GAP-minor (audit #22); single-block, serial across batch. At c=16 vocab=152k binary search ~5% decode wall-clock *(hypothesis)* | FlashInfer `sampling.cu` multi-block argmax. `needs-paired-A/B` (low priority) | — |
| quant-util | `dtype_convert` / `turboquant{,_fast}` / `split_qkv` | hand-rolled CUDA (`quant/`, `misc/split_qkv.cu`) | any SM · dtype convert / rotated quant / QKV slice | hardcoded | PASS (audit #28/29/30); memcpy-equivalent / research-grade FWHT | — at SOTA | — |

---

## Metal — MLX primitives (below-FFI kernel choice is a black box per gov non-goal)

The Metal backend dispatches to **MLX library primitives**; the actual Metal
kernel inside `mx::eval` is opaque (no hook to report which kernel fired —
`../plans/gpu-dispatch-governance.md` §2.1, non-goal §"Not building MLX kernel
introspection"). Registry tracks the *primitive chosen*, not the GPU kernel.

| op family | variant | impl type | SKU / shape class | tuned? | roofline position | best-known alt + why not wired | owner |
|---|---|---|---|---|---|---|---|
| linear (dense) | `quantized_matmul` | MLX-primitive (`mlx_qwen35_model.cpp:528`) | Apple Silicon · 4-bit dense linear | MLX-internal | n/a (MLX owns); 0.3GB vs 1.2GB/step read (`:1593`) | — MLX is the canonical Metal path | — |
| moe-experts | `gather_qmm` ×3 + SiLU (SwitchGLU) | MLX-primitive (`mlx_qwen35_moe_block.cpp:164-182`) | Apple Silicon · Qwen3.5/3.6 MoE experts | MLX-internal (sorted-indices fast path threshold 64) | n/a; mlx-lm SwitchGLU pattern | — MLX is canonical | — |
| attn | `fast::scaled_dot_product_attention` | MLX-primitive (`mlx_qwen35_model.cpp`, `mlx_dflash_draft_model.cpp:183`) | Apple Silicon · GQA attention | MLX-internal | n/a | — MLX is canonical | — |
| pos-enc | `fast::rope` | MLX-primitive | Apple Silicon · RoPE | MLX-internal | n/a; layout `[B,heads,seq,d]` (T=2nd-last axis) | — MLX is canonical | — |
| norm | `fast::rms_norm` | MLX-primitive | Apple Silicon · RMSNorm | MLX-internal | n/a | — MLX is canonical | — |

---

## P0 gaps — concrete "best-known alternative + why not wired"

The three P0 gaps the governance §2.4 names. Verdicts are **current as of
2026-05-29** (the rows above carry the same state; this is the narrative):

| Gap | Best-known alternative | Wire state | Verdict |
|---|---|---|---|
| **GAP-A** quantized GEMV scalar-FMA (~16% decode) | `mma.m16n8k16` BF16×BF16→FP32 tensor-core variant (CUTLASS/FlashInfer-class) | **Already implemented** (`gemm/quantized_gemv_mma.cu`), C-dispatch wired (`quantized_gemv.cu:2636`) behind `ARLE_DSV4_FP8_GEMV_MMA` (default OFF), bit-parity test PASS (relerr≤1e-3, `ops/tests.rs:297`). Phase-2 license-or-kill PASS on H20 (scalar `frac_peak=0.04`, 96% BW unused) | **`needs-paired-A/B`** — pod TPOT A/B (env OFF vs ON, ≥4% decode gate) is `pending-remote`. Not yet a default flip. |
| **GAP-B** MoE routing 255/256 threads idle (~4.4% decode) | SGLang `topk_softmax.cu` (1 warp/token, warp-parallel softmax + bitonic top-k); FlashInfer `top_k_renorm_probs` | **Not started.** The warp-parallel rewrite has *not* been written | **`needs-paired-A/B`** — and the production E256/top8 shape, not the 1B E16/top2 microbench. The two prior route kills were a *different axis* (block-size + persistent kernel) and do **not** kill this fix. |
| **GAP-C** FP8 fused-decode attention scalar softmax | FA-v3 / FlashInfer `BatchDecodeWithPagedKVCache` (cp.async pipeline + `mma.m16n8k16` softmax) | **cp.async-cheap-half LANDED** (`ab850f7a`, FP8 kernel now mirrors INT8 cp.async double-buffer). MMA-softmax medium-half **not started** | cp.async half: **`needs-paired-A/B`** (committed, pod TPOT `pending-remote`, c=16 ≥1% gate). MMA-softmax half: separate licensed followup, possibly reuses the FlashMLA-decode template. |

> **None of the three P0 gaps is a default flip yet.** GAP-A and GAP-C-cheap
> are code-complete but blocked on the same pod-TPOT-A/B gate the governance
> Phase-3 `--expect-kernel` + `dispatch_fallback_total==0` machinery exists to
> make mechanical; GAP-B is unwritten. This registry makes "best kernel exists
> but isn't wired" a tracked, owned item instead of a silent default.

## Cross-refs

- Govern gate plan: [`../plans/gpu-dispatch-governance.md`](../plans/gpu-dispatch-governance.md) §Phase 4
- Root-cause analysis: [`2026-05-29-gpu-dispatch-governance-analysis.md`](2026-05-29-gpu-dispatch-governance-analysis.md) §2.4
- SOTA replacement gap (companion): [`../research/2026-05-29-oplib-sota-kernel-gap.md`](../research/2026-05-29-oplib-sota-kernel-gap.md)
- Kernel-vs-SOTA audit (richest source): [`../research/2026-05-28-arle-kernel-vs-sota-audit.md`](../research/2026-05-28-arle-kernel-vs-sota-audit.md)
- Six-principles heat map: [`2026-04-14-cuda-kernel-six-principles-review.md`](2026-04-14-cuda-kernel-six-principles-review.md)
- Operator library design (forward link, not yet written): [`../plans/backend-operator-library.md`](../plans/backend-operator-library.md)
