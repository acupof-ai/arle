# ARLE CUDA kernels vs industry SOTA — audit

Date: 2026-05-28
Scope: every kernel under `crates/cuda-kernels/csrc/` and
`crates/cuda-kernels/tools/tilelang/` that is on the Qwen3.5 or DSv4 hot path,
EXCEPT the four files already being worked by the FlashMLA-decode / DSv4-prefill
multi-stream subagents (`vendor/flashmla/csrc/sm90/*`, `arle_flashmla_shim.cu`,
`arle_flashmla_csa_prep.cu`, `dsv4_tp_attention_repack.cu`, and the
`weights.rs` decode/prefill branches).

This is a read-only audit; no `.cu` edits in this pass. Each entry pairs the
ARLE kernel with the canonical industry-SOTA equivalent, scores PASS / GAP, and
proposes the work axis (A1–A8 from
[`docs/projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md`](../projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md))
or `new-axis` for any GAP.

Performance anchors used:
- L4 head-to-head bench:
  [`docs/experience/wins/2026-05-25-bench-guidellm-cuda-l4-arle-vs-sglang-headtohead.md`](../experience/wins/2026-05-25-bench-guidellm-cuda-l4-arle-vs-sglang-headtohead.md)
- DSv4 decode binding-constraint table (L1 alloc / L2 memset / L3 in-kernel /
  L4 attention / L5 DtoH / L6 NCCL):
  [`docs/research/2026-05-15-dsv4-decode-memaccess-binding-constraints.md`](2026-05-15-dsv4-decode-memaccess-binding-constraints.md)
- DSv4 FlashMLA v2 22× prefill win:
  [`docs/experience/wins/2026-05-27-dsv4-flashmla-v2-22x-prefill-22x-pre-crash.md`](../experience/wins/2026-05-27-dsv4-flashmla-v2-22x-prefill-22x-pre-crash.md)

## Headline ranking by expected wall-clock impact

1. **GAP-A** — Quantized batch-GEMV scalar FP8/FP4 decode at M ≥ 16 (no MMA,
   no tensor cores). L3 = 14.6 % GPU time today; +N-tile + WGMMA / mma.sync
   would close ~half. Belongs to A4 (multi-stream overlap is a separate axis;
   this is intrinsic kernel optimization — new sub-axis under L3).
2. **GAP-B** — `dsv4_route_kernel` runs entirely on `threadIdx.x == 0` per
   token (255 / 256 threads idle). L3 = 4.4 %. Trivially fixed; should be a
   one-day win.
3. **GAP-C** — Fused-decode-attention quantized path (`fused_attention.cu` +
   `decode_attention_quantized.cu`) is scalar warp-shuffle softmax with no
   cp.async pipeline for FP8 path, no tensor cores, and `HEAD_DIM` baked in
   as `#define`. FlashInfer / FlashAttention-v3 / FlashMLA do CUTLASS-MMA +
   software pipeline. Qwen3.5 decode at high concurrency rides here.

Three PASS calibrators: `gemv.cu` (cuBLASLt routed BF16 GEMM), `transfer.cu`
(SGLang-ported KV transfer), `deepgemm_native.cu` (official DeepSeek DeepGEMM
v3 SM_90 wrapper).

---

## Per-kernel table

| # | File / kernel | Op (shape) | Verdict | Industry-SOTA reference | Action axis |
|---|---|---|---|---|---|
| 1 | `gemm/quantized_gemv.cu` — `dsv4_fp8_gemv_batch_tiled_kernel` and the FP4 / route / grouped variants | `[B≤32, K=7168] · [N, K] fp8 → [B, N] bf16` per expert | **GAP** | DeepGEMM `m_grouped_gemm_nt_masked` (TMA + WGMMA, FP8E4M3); FlashInfer `bgmv_*` | L3 / new-axis (under L3) |
| 2 | `gemm/gemv.cu` — `gemm_cuda` / `gemm_graphsafe_cuda` | BF16 × BF16 row-major GEMM via cuBLASLt | **PASS** | cuBLASLt TensorOp algo cache; identical to TRT-LLM cublasLt path | — |
| 3 | `gemm/dsv4_grouped_gemm.cu` — `dsv4_fp8_grouped_gemm_batch_kernel` / `pair_batch_kernel` | Grouped FP8E4M3 GEMM with M-tile=32 weight reuse across experts | **GAP** | DeepGEMM `m_grouped_gemm_nt_contiguous`; same shape, but TMA + WGMMA on SM_90 | L1 (Mega-MoE) / new sub-axis on SM_89 |
| 4 | `gemm/deepgemm_native.cu` (SM_90 only) | Wrapper that loads DeepSeek DeepGEMM v3 cubins for `m_grouped_gemm_nt_masked`, etc. | **PASS** | This *is* DeepGEMM; uses `cute::UMMA`, TMA-swizzle, 232 KB smem stages | — |
| 5 | `gemm/dsv4_deepgemm_ops.cu` — pack+quantize, swiglu+quantize, unpad | BF16→FP8E4M3 row pack + per-block scale; SwiGLU prefuse | **PASS** | DeepGEMM expects exactly this layout; per-block `granK=128` scale matches DeepSeek's spec | — |
| 6 | `gemm/dsv4_fp8_cache.cu` — `dsv4_block_scaled_to_fp8_cache_kernel` | Convert weight cache to DeepGEMM block-scaled FP8 layout | **PASS** | Build-time data conversion, not on hot path | — |
| 7 | `gemm/marlin_kernel.cu` / `marlin_w4a8_kernel.cu` / `marlin_w4_fp8_kernel.cu` | INT4 weight × FP16/BF16/FP8 act GEMM with mma.sync | **PASS (cite)** | Direct IST-DASLab Marlin port; vLLM ships the same file; SGLang `sgl-kernel/csrc/gemm/marlin/` is identical | — |
| 8 | `gemm/turboquant_weight_gemv.cu` | Rotated-quant W2/W4 GEMV with FWHT-rotated activations | **GAP-minor** | Recent QuaRot / TurboQuant papers; not standard in vLLM/SGLang yet. Decode path only, smallish impact | new-axis (low priority) |
| 9 | `moe/dsv4_route.cu` — `dsv4_route_kernel` (and the 28 supporting permute / pack / scan / scatter kernels) | `[T, E] bf16 logits → top-k indices + weights` | **GAP** | SGLang `sgl-kernel/csrc/moe/topk_softmax.cu` uses 1 warp per token, parallel reductions; FlashInfer `topk_renorm_probs` is fully data-parallel | A4 / new sub-axis under L3 |
| 10 | `moe/deepseek_mask_indices_by_ep.cu` | Filter top-k indices by EP/TP rank, compact gaps | **PASS** | Adapted from DeepSeek TileKernels `mask_indices_by_tp_kernel.py`; same algorithm as upstream | — |
| 11 | `kv/kv_quant.cu` — `quantize_paged_kv_fp8_kernel` / `int8` / `int4`, per-channel K and per-token V variants | bf16 → INT8 / FP8E4M3 / INT4 quantize-then-write to paged pool, HND layout | **PASS (mostly)** | FlashInfer `quantize_kv_cache.cu` uses identical per-token symmetric INT8 + per-channel K scheme; vLLM kv quant follows same recipe; one warp-per-token-per-head | — |
| 12 | `kv/kv_cache_to_paged.cu` | Contiguous KV → paged HND pool conversion | **PASS** | One-shot conversion at session start, not hot; warp-per-row copy is bandwidth-bound | — |
| 13 | `kv/paged_kv_append.cu` / `scatter_kv.cu` / `paged_kv_metadata.cu` | Per-decode-token append; per-prefill scatter; metadata pack | **PASS** | Same pattern as FlashInfer `append_paged_kv_cache.cu`; warp-per-token-per-head BF16 copy | — |
| 14 | `attention/decode_attention_quantized.cu` (FP8 / INT8 / INT4 per-channel) | Flash-decoding split-KV partial + log-sum-exp merge; one warp per K-row | **GAP** | FlashInfer `BatchDecodeWithPagedKVCache` (cuBLASDx + cp.async pipeline); FlashAttention-v3 quantized path. ARLE: `cp.async` only on INT8 path, FP8 path is sync loads (see comment at line 416) | A2 / new-axis |
| 15 | `attention/fused_attention.cu` — `fused_gqa_attention_single_token_kernel` and `decode_*` variants | Online-softmax tile attention with RMSNorm + RoPE fused into the same kernel; HEAD_DIM=128 hard-coded | **GAP-minor** | FlashAttention-v3 / FlashInfer template on HEAD_DIM; cp.async pipeline; MMA inside QK / PV. ARLE is shared-mem online-softmax, no tensor cores | A2 / new-axis (low priority on L4, important on H100) |
| 16 | `attention/prefill_attention.cu` / `prefill_attention_hd256.cu` / `nonpaged_prefill_attention.cu` | Causal prefill attention, BF16, scalar warp-tile softmax | **PASS (with caveat)** | These are fallback paths — TileLang AOT (`batch_prefill_paged_hd128.py`, `hd256.py`) is the primary prefill; this file is the no-paged fallback. TileLang uses cute::SM80 mma | — |
| 17 | `attention/prefill_attention_paged_prep.cu` / `decode_prep_paged.cu` / `decode_prep_paged_hd256.cu` | Per-head QK RMSNorm + RoPE, then write KV into the paged HND pool | **PASS** | Bandwidth-bound prep step; one block per (head, token); identical pattern to vLLM `_paged_attention_v2_prep`. RoPE half-split convention matches HF | — |
| 18 | `attention/mla_decode.cu` | MLA dispatch shim (delegates to TileLang AOT / FlashMLA wrapper) | **PASS** | Just a dispatch table; the actual kernels are TileLang or FlashMLA (out of scope) | — |
| 19 | `misc/dsv4_attention.cu` — `dsv4_swa_attention_kernel`, `dsv4_compressor_update_kernel`, `dsv4_prepare_qk_fused_kernel`, `dsv4_update_window_cache_kernel` (the **non-FlashMLA** DSv4 prep paths) | SWA + compressor update for DSv4 hybrid attention; QK norm+RoPE fuse | **GAP** | SGLang V4's `flashmla_hybrid_attention` fuses SWA + compressed attention into one kernel call (A2). ARLE has fused QK-prep (A2.1) but SWA + compressor + window-update are still separate launches. dsv4_csa_select_kernel (in this file) and dsv4_hybrid_attention_kernel are being replaced by the FlashMLA subagent — leave alone | A2 |
| 20 | `misc/dsv4_mhc.cu` — `dsv4_mhc_params_kernel`, `dsv4_mhc_pre/post/expand/head_pre_kernel` | Multi-head compressor (MHC) parameter computation, softmax/row/column normalize | **GAP-minor** | Block-per-token with full row/col softmax in one block; this is fine for `topk ≤ 8` but the per-token softmax is serialized over the row axis (`for row in 0..n`). Could be one warp per row in parallel | new-axis (low priority) |
| 21 | `misc/norm.cu` — `rms_norm_*`, `fused_add_rms_norm_*`, `rms_norm_batched_*`, `rms_norm_gated_kernel` | RMSNorm + residual-add fused, vectorized bf16x4 loads, FP32 accumulators | **PASS** | Same vectorized + 2-pass-fused pattern as vLLM `csrc/layernorm_kernels.cu` and FlashInfer `rmsnorm.cuh`; FP32-accum + bf16-rounded second-pass mid-states match HF | — |
| 22 | `misc/sampling.cu` — `argmax_kernel_fast`, `argmax_batch_kernel`, `gpu_sample_kernel` (top-k + top-p) | Single-block sampler with binary-search top-k + scan top-p | **GAP-minor** | FlashInfer `sampling.cu` does multi-block argmax for very large vocab; `gpu_sample_kernel` is single-block and serial across batch (one call per request). For batched decode at c=16 with vocab=152k the binary search dominates ~5 % of decode wall-clock | new-axis (low priority) |
| 23 | `misc/elementwise_basic.cu` — `silu_mul_native_kernel`, `dsv4_swiglu_clamped_kernel`, `add_scaled_row_kernel`, `embedding_*_kernel` | Element-wise SiLU·up, scaled add, embedding gather, bf16x4 vectorized | **PASS** | Bandwidth-bound element-wise ops; same pattern as vLLM `csrc/activation_kernels.cu` | — |
| 24 | `misc/fused_mlp.cu` — `fused_mlp_intermediate_kernel` (gate+up interleaved) / `fused_mlp_output_kernel` | Fused gate+up dot in one pass over x (read x once) | **GAP-minor** | This is a per-block dot-product fallback for the small-MLP shapes. cuBLASLt GEMM is faster at large intermediate_size; the kernel is bypassed by `gemv.cu` for Qwen3.5 prod path. Mostly dead code in DSv4 path | new-axis (low; consider removal) |
| 25 | `misc/gated_delta_rule.cu` — `gated_delta_rule_decode_kernel` (and `gdr_*_batch.cu` variants) | Recurrent linear-attention decode step (Qwen3.5 linear-attn layer); per-block per-value-head, 512 threads with 4 J-slices | **PASS** | This is the ARLE-original lowering of FLA's chunk-wise gated delta rule — there is no canonical SOTA for this in vLLM/SGLang (gated delta rule is Qwen3.5-specific). The 4-slice J partition + 16 warps/block + smem reduction is well-tuned for L4/A100 occupancy | — |
| 26 | `misc/gdr_prefill_solve.cu` | Strict-lower triangular solve for chunk-wise gated delta rule prefill | **PASS** | Likewise unique to Qwen3.5 linear-attn; warp-per-chunk-row inversion is appropriate | — |
| 27 | `misc/conv1d.cu` / `conv1d_decode_batch.cu` / `conv1d_prefill_batch.cu` | Depthwise causal Conv1d for gated delta net | **PASS** | Channel-per-thread depthwise conv is the standard pattern (Mamba2 / vLLM `csrc/mamba/causal_conv1d_*`) | — |
| 28 | `misc/split_qkv.cu` | Slice fused QKV tensor into Q / K / V | **PASS** | One-element-per-thread copy; bandwidth-bound | — |
| 29 | `quant/turboquant.cu` / `turboquant_fast.cu` | TurboQuant rotated-quant kernels (random Hadamard transform + symmetric INT8) | **PASS (research-grade)** | TurboQuant is a research method ARLE adopted; no vLLM/SGLang equivalent. `turboquant_fast.cu` uses FWHT (O(D log D)) vs the naive O(D²) rotation matmul — this is the right algorithm | — |
| 30 | `quant/dtype_convert.cu` | bf16↔fp32 / int8↔bf16 element-wise conversions | **PASS** | Memcpy-equivalent kernels | — |
| 31 | `kvcacheio/transfer.cu` — `transfer_kv_pages_layer_table_kernel` | KV-tier page copy between pools (DRAM ↔ HBM) | **PASS** | Direct port of SGLang `sgl-kernel/csrc/kvcacheio/transfer.cu` with the same `ld.global.nc.b64` / `st.global.cg.b64` non-temporal copy strategy | — |
| 32 | TileLang AOT `batch_prefill_paged_hd128.py` / `hd256.py` / `hd64.py` | Paged-KV prefill attention generated via TileLang | **PASS** | TileLang lowers to `cute::SM80::mma` (SM_80 / SM_89) and the planned SM_90 path. ARLE pinned TileLang 0.1.9 for sm_89 planner bug fix — matches SGLang's pinned version | — |
| 33 | TileLang AOT `batch_decode_paged_hd128.py` / `hd128_fp8.py` / `hd256.py` / `hd64.py` | Paged-KV decode attention with FlashDecoding split-KV | **PASS-with-watch** | Same TileLang generator; `hd128_fp8` is the only fp8 KV decode path. Vs FlashInfer `BatchDecodeWithPagedKVCache` this is competitive at c≤4 (head-to-head shows ARLE ITL beats SGLang +8.6 % at c=4 to +21.8 % at c=16). Keep | — |
| 34 | TileLang AOT `gated_delta_rule.py` | Chunk-wise gated delta rule prefill stages | **PASS** | Same as #25; no vLLM/SGLang baseline | — |
| 35 | TileLang AOT `deepseek_moe_mask_indices_by_ep.py` | MoE index masking | **PASS** | Same algorithm as `moe/deepseek_mask_indices_by_ep.cu` (sister TileLang impl); kept for ABI parity | — |

---

## GAP entries — detail and fix plan

### GAP-A · Quantized batch-GEMV (FP8 / FP4) — scalar accumulation, no tensor cores

**ARLE today**: `dsv4_fp8_gemv_batch_tiled_kernel` and the FP4 sibling
process a `DSV4_BATCH_TILE=32` tile of tokens against shared weight. Each
thread does scalar `__bfloat162float(x[batch * K + k]) * fp8_decode(W[row, k])`
in FP32 registers, then a warp-shuffle + shared-mem reduction. Inner loop
issues **one FMA per K element per token per row** — no `mma.sync`, no
`mma.m16n8k16`, no WGMMA. With `DSV4_BATCH_TILE=32` the effective compute
intensity is `32 × 7168 FMA / 7168 FP8 byte load = 32 FLOP/B`, just barely
above the L4 roofline tipping point. We are bandwidth-bound on weights but
compute-bound on the scalar reduction because every K element triggers
`tile_batches` (up to 32) extra adds and ALU mul, serializing 32-token
inner-loop on the same warp.

**Wall-clock anchor**: L3 in-kernel access budget = `FP8 batch GEMV 8.6 %`
+ `FP4 batch GEMV 4.5 %` + `FP8 tiled 1.6 %` + `FP4 tiled 1.3 %` =
**16.0 % of decode GPU time** (2026-05-14 trace,
[`docs/research/2026-05-15-...md`](2026-05-15-dsv4-decode-memaccess-binding-constraints.md)
§Binding constraints table). Note: this is decode-shape with batch ≤ 16;
prefill shape (B = 29795) hits the 67×-off-SLO catastrophe documented in
`2026-05-27-dsv4-tp-allreduce-slo-prefill-kill.md`, which is exactly why
`dsv4_grouped_gemm.cu` was added with M-tile reuse — but that file is the
**same scalar pattern**, just with multi-expert dispatch fused. Even tiled,
no WGMMA.

**SOTA (cite)**:
- DeepGEMM v3 `m_grouped_gemm_nt_masked`
  ([`vendor/deepgemm`](https://github.com/deepseek-ai/DeepGEMM)) uses
  `cute::UMMA::Major::K` + TMA-multicast + `wgmma.fence_async` — the
  reference DSv4-style FP8 GEMM, already on SM_90. ARLE wraps this for the
  full-tile case but the GEMV-shape DSv4 batch decode (B = 1..16, decode
  step) still falls back to scalar `quantized_gemv.cu`.
- FlashInfer `bgmv_*` (`flashinfer/python/csrc/bgmv/`) — batched GEMV with
  `mma.m16n8k16` for INT8 / FP8E4M3, available SM_80+.
- SGLang `sgl-kernel/csrc/gemm/fp8_blockwise_scaled_grouped_mm.cu` uses
  CUTLASS-MMA grouped GEMM for batches `M ≤ 16`. Matches our DSv4 decode
  shape exactly.

**Fix**: porting CUTLASS or FlashInfer's `mma.m16n8k16`-based path for
M ≤ 16. SM_89 fallback to `mma.m16n8k8`. Expect ~2× kernel-local speedup
on the `_batch_tiled_kernel`, which is half of L3 — wall-clock impact
~4–6 % of decode (per binding-constraints table). On L4 (SM_89) this is
the largest accessible non-attention lever; on H20 (SM_90) DeepGEMM
already wraps the WGMMA path so the gap is narrower.

**Axis**: L3 = in-kernel optimization. Could become a new "A9 — quantized
GEMV tensorcore" axis under the SGLang-V4 framing in
[`docs/projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md`](../projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md).
**Don't piggyback on A4** — A4 is scheduler / multi-stream overlap, this
is intrinsic kernel work.

---

### GAP-B · `dsv4_route_kernel` runs on `threadIdx.x == 0` per token

**ARLE today**:
[`crates/cuda-kernels/csrc/moe/dsv4_route.cu:206-318`](../../crates/cuda-kernels/csrc/moe/dsv4_route.cu).
Launches with `<<<num_tokens, DSV4_ROUTE_BLOCK=256, …>>>` but the entire
kernel body is gated by `if (threadIdx.x == 0) { … }`. Inside that branch
it serially iterates over all `n_experts` (up to 512), computes per-expert
score, then runs a serial selection-sort to extract top-k. **255 threads
per block sit idle**.

Code review evidence (route_kind == 1, scoring_kind != 0 branch):
```cpp
if (threadIdx.x == 0) {
  for (int expert = 0; expert < n_experts; ++expert)
    scores[expert] = dsv4_route_score(...);
  // ...
  for (int expert = 0; expert < n_experts; ++expert) {
    float top_score = scores[expert] + bias[expert];
    for (int k = 0; k < topk; ++k) {  // serial insertion-sort
      if (better) shift_and_insert();
    }
  }
}
```

**Wall-clock anchor**: L3 = `dsv4_route_kernel 4.4 % GPU time` (2026-05-14
trace). Per-token call shape: `<<<num_tokens, 256>>>` with effective
parallelism of 1. With 256 threads doing work in parallel this would be
~10×-50× faster on the per-token cost; in wall-clock-decode-window terms
the math caps at the 4.4 % bound, so ~3–4 % decode wall-clock improvement
is the realistic ceiling.

**SOTA (cite)**:
- SGLang `sgl-kernel/csrc/moe/topk_softmax.cu` — **one warp per token**;
  warp-parallel softmax + warp-shuffle top-k via bitonic / max-tournament.
- FlashInfer `top_k_renorm_probs` (`flashinfer/python/csrc/sampling.cu`)
  is fully data-parallel along the expert axis.
- vLLM `vllm/model_executor/layers/fused_moe/fused_moe.py` uses Triton
  `topk_softmax` which compiles to one block per token, 256 threads
  cooperating on softmax via tree reduction.

**Fix**:
1. Parallelize softmax across the block (warp-stride reduction; already
   used elsewhere in the file).
2. Replace serial selection-sort with bitonic top-k (`n_experts ≤ 512`,
   `topk ≤ 16` — small constant), or a warp-tournament top-k like
   FlashInfer's. With `n_experts=256, topk=8` the bitonic top-k is well
   under a microsecond.
3. Same fix applies to the **other 28 supporting kernels** in
   [`dsv4_route.cu`](../../crates/cuda-kernels/csrc/moe/dsv4_route.cu)
   that follow the same `threadIdx.x == 0` single-thread pattern — they
   are L1-cheap individually but the launch churn (28 kernels × per-token
   calls) is part of why DSv4 decode shows 16k launches per wave.

**Axis**: This is **the single highest-leverage easy win** identified in
this audit (small diff, no architecture change, clearly wrong). Belongs
to a new "A9 — MoE prep kernels" sub-axis under L3 / launch-churn (A3 is
the in-graph metadata axis, which is complementary not duplicate).

---

### GAP-C · Decode-attention quantized path — FP8 sync-load, no MMA softmax

**ARLE today**: [`attention/decode_attention_quantized.cu`](../../crates/cuda-kernels/csrc/attention/decode_attention_quantized.cu).
The INT8 partial kernel uses `cp.async` (lines 23, 410) for K/V tile prefetch.
**The FP8 sibling explicitly does NOT** — quote line 416:
*"cp.async pipelining (which the INT8 sibling uses but the FP8 sibling
[does not] yet)"*. The FP8 path issues blocking `__nv_fp8x4_e4m3` loads
inside the QK inner loop. Both paths use scalar warp-shuffle softmax with
FP32 accumulators in registers — no `mma.sync` for QK or PV.

**Wall-clock anchor**:
- Quwn3.5 c=1 decode ITL p50 = 36.14 ms (matches SGLang within 2.5 %, so
  not a *bottleneck* at c=1).
- c=16 ITL = 71.85 ms and we beat SGLang by 21.8 % — but SGLang at c=16
  is **also** scalar in its INT8 decode kernel. The right comparison is
  vs FlashInfer's `BatchDecodeWithPagedKVCache` FP8 path (uses
  `mma.m16n8k32` for QK), which we currently can't run against (no FlashInfer
  build in our stack). DeepSeek's FlashMLA decode path (also out of scope
  for this audit) is the wall-clock evidence here: 22× prefill speedup
  came partly from FP8 KV + WGMMA QK.
- L4 = `dsv4_hybrid_attention 6.4 % + csa_select 3.9 % = 10.3 %` GPU time
  in the DSv4 binding-constraints table; the quantized-decode path lights
  up in Qwen3.5 (not DSv4) traces — different model, both same
  authoring style.

**SOTA (cite)**:
- FlashAttention-v3 (Dao-AILab) `flash_attn_with_kvcache` — FP8 KV + WGMMA
  on SM_90, mma.m16n8k16 on SM_89:
  [`flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h`](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h).
- FlashInfer `BatchDecodeWithPagedKVCache` — split-KV + cp.async pipeline +
  cuBLASDx-driven MMA softmax:
  [`flashinfer/include/flashinfer/attention/decode.cuh`](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/decode.cuh).
- TRT-LLM `paged_attention_v2_quant` for the INT8 / FP8 split-K path.

**Fix**:
1. **Cheap**: add `cp.async` pipeline to the FP8 partial kernel (the INT8
   sibling shows it's a ~50 LoC delta; the comment at line 416 is an
   explicit invitation).
2. **Medium**: hoist `q_reg`/`k_scale_reg` register layout into a CUTLASS
   `MainloopFP8` template; replace scalar QK FMA with `mma.m16n8k16`.
   Requires touching the merge kernel layout (FP32 partial accumulators).
3. **Hard**: full FlashAttention-v3 port — but that's exactly what the
   FlashMLA-decode subagent is doing for the DSv4 family, so wait for
   that to land first and see if the Qwen3.5 family can reuse the same
   template.

**Axis**: A2 for FlashMLA-style fusion (DSv4 family is being done by
another agent; Qwen3.5 family is the gap here). The `cp.async` cheap-fix
is **independent** and could ship as a new "A10 — Qwen3.5 quantized
decode pipeline" sub-axis under L3.

---

### GAP-D · `dsv4_grouped_gemm.cu` — SM_89 fallback for DSv4 grouped GEMM

**ARLE today**: When SM_90 / DeepGEMM is unavailable, ARLE falls back to
`dsv4_fp8_grouped_gemm_batch_kernel` (lines 60–192 of
`dsv4_grouped_gemm.cu`) which is **the same scalar FP32-accum pattern as
GAP-A**, just with one more grid axis (Z = num_experts). All the GAP-A
caveats apply — no tensor cores, M-tile-32 reuse only via register
spilling.

**Wall-clock anchor**: This kernel is on the DSv4 prefill SLO-kill path
([`2026-05-27-dsv4-tp-allreduce-slo-prefill-kill.md`](../experience/errors/2026-05-27-dsv4-tp-allreduce-slo-prefill-kill.md)).
The 22× prefill speedup landed via FlashMLA, not via this kernel — so on
H20 the situation is OK (DeepGEMM handles it). On L4 (no SM_90) this
kernel is decisive for DSv4 prefill but L4 is not a deployment target for
DSv4, so the priority is **medium**.

**SOTA**: Same as GAP-A. The SGLang grouped CUTLASS path
(`sgl-kernel/csrc/gemm/fp8_blockwise_scaled_grouped_mm.cu`) handles
exactly this shape.

**Fix**: Once GAP-A's CUTLASS-MMA path lands, port it here with one extra
grid Z axis and an `expert_indices`-indexed weight pointer. Should be
mechanical given GAP-A is done.

**Axis**: L3 / A1 (sub-axis under Mega-MoE — A1 is the bigger
architecture answer with symmetric memory; this is the SM_89 fallback
quality lever).

---

### GAP-E · `misc/dsv4_attention.cu` non-FlashMLA prep kernels — fragmentation

**ARLE today**: This file has ~13 kernels (lines 105–1028). The FlashMLA
subagent is replacing `dsv4_hybrid_attention_kernel` and
`dsv4_csa_select_kernel` (the last two). The **other** 11 kernels —
`dsv4_prepare_q_kernel`, `dsv4_prepare_k_kernel`,
`dsv4_prepare_qk_fused_kernel` (the 2026-05-26 A2.1 fusion landing),
`dsv4_update_window_cache_kernel`, `dsv4_compressor_update_kernel`,
`dsv4_swa_attention_kernel`, plus various scatter / pack helpers — remain.

Per the project log
[`2026-05-25-dsv4-next-axis-from-sglang-v4.md`](../projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md)
A2.0 already fused window-cache update into SWA tails (saved 9504
standalone launches); A2.1 fused QK prep (`prepare_q + prepare_k` → one).
What remains:

- `dsv4_compressor_update_kernel` is still a separate launch per layer
  per token.
- `dsv4_swa_attention_kernel` is separate from `dsv4_hybrid_attention_kernel`
  (the latter is being replaced by FlashMLA; SWA could collapse into the
  FlashMLA fused-hybrid call once that lands — confirm with the other
  agent).

**Wall-clock anchor**: launch-churn axis. Per-launch overhead at the
~16k-launch / decode-wave count contributes to L1 / L5. Not a big
arithmetic win, but a real per-token launch overhead saving.

**Fix**: Wait for the FlashMLA-decode subagent to land, then audit which
of these 11 kernels are still actually called. The compressor update
fusion is its own story.

**Axis**: A2 (FlashMLA fusion) tail / A3 (in-graph metadata) — let the
other subagent finish before touching.

---

## PASS calibrators — where ARLE is at or above SOTA

For sanity, the kernels below are **at parity or better** than what vLLM /
SGLang / FlashInfer ship today. Listed here so the GAP list isn't taken as
"ARLE kernels are uniformly behind".

### PASS-1 · `gemm/gemv.cu` (cuBLASLt-routed BF16 GEMM)

ARLE routes BF16 × BF16 GEMM through `cublasLtMatmul` with a generation-
counter-protected `cublasLtMatmulAlgo_t` cache (lines 200–250).
`CUBLAS_TENSOR_OP_MATH` is set on every handle. This is identical to the
TRT-LLM and vLLM cuBLAS path — including the same pre-allocated workspace
pattern. The bf16 handwritten path (`gemv_handwritten_kernel`, lines
45–130) only runs when M < 16, where cuBLAS is known to be sub-optimal.
Both branches use `bf16x4` / `bf16x8` vectorized loads.

### PASS-2 · `kvcacheio/transfer.cu` (KV-tier page transfer)

This is a **direct line-by-line port of SGLang's**
`sgl-kernel/csrc/kvcacheio/transfer.cu` (acknowledged in the file header).
Uses `ld.global.nc.b64` + `st.global.cg.b64` non-temporal copies with a
warp-per-item layout, which is the same recipe TRT-LLM uses for chunked-
KV moves. Bandwidth-bound and already at the L4 / H100 hardware limit per
SGLang's own benchmarks.

### PASS-3 · `gemm/deepgemm_native.cu` (DSv4 FP8 grouped GEMM, SM_90)

This is the official DeepSeek **DeepGEMM v3** wrapper — `cute::UMMA::Major::K`,
TMA-swizzle (line 555), per-block scale `granK = 128` (line 41), 232 KB
smem stages (`kSm90SmemCapacity = 232448`, line 42), `cuLaunchKernelEx`
launch (line 132/395) with cluster shape. There is no faster FP8 grouped
GEMM on SM_90 in any open-source stack; this *is* the SOTA. ARLE pulls
the cubins from `vendor/deepgemm/deep_gemm/` and dispatches through
`dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked_cuda` (line 908).

### PASS-4 · `misc/norm.cu` RMSNorm family

`fused_add_rms_norm_kernel` and friends use the canonical vLLM /
FlashInfer pattern: bf16x4 vectorized loads, FP32 sum-of-squares
accumulator, second-pass bf16-rounded mid-state to match HF numerics
exactly. The `_offset_kernel` variant (lines 726+) handles per-request
offsets in the prefill batched path. Identical algorithm to vLLM
`csrc/layernorm_kernels.cu::fused_add_rms_norm_kernel`.

### PASS-5 · TileLang AOT decode/prefill kernels

The L4 head-to-head bench shows ARLE decode ITL **beats SGLang
+8.6–21.8 %** at c=4..16 with prefill TTFT being the *only* loss axis
(scheduler-bound, not kernel-bound — that was confirmed in the same wins
entry). The TileLang HD128 / HD256 paged decode + prefill is therefore
**ahead of** the SGLang FlashInfer path on the L4 SM_89 target. Don't
touch.

---

## Recommended ordering for downstream work

Drop-in / cheap wins (do first, no architecture license needed):

1. **GAP-B fix** — parallelize `dsv4_route_kernel` across the block.
   Estimated diff: ~200 LoC. Expected wall-clock: ~3–4 % decode.
2. **GAP-C cheap-half** — add `cp.async` pipeline to the FP8 decode
   partial kernel; mirror the INT8 sibling. Estimated diff: ~80 LoC.
   Expected wall-clock: ~2–3 % decode for Qwen3.5 quantized KV paths.

Medium effort, single-axis (architecture license recommended):

3. **GAP-A** — CUTLASS-MMA quantized batch GEMV (FP8 / FP4). Largest
   pure-kernel lever on the SM_89 stack. Estimated diff: ~600 LoC (port
   from CUTLASS or FlashInfer). Expected wall-clock: ~4–6 % decode.
4. **GAP-D** — Port GAP-A to the grouped-expert shape. Mechanical once
   GAP-A is done. Same ballpark wall-clock win on DSv4 prefill on L4
   (not H20 — H20 already uses DeepGEMM).

Wait-for-other-subagents:

5. **GAP-E** — non-FlashMLA prep kernels in `misc/dsv4_attention.cu`.
   Re-audit after the FlashMLA-decode subagent lands.

---

## Cross-refs

- Source-of-truth axis backlog:
  [`docs/projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md`](../projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md)
- DSv4 binding-constraints (L1–L6):
  [`docs/research/2026-05-15-dsv4-decode-memaccess-binding-constraints.md`](2026-05-15-dsv4-decode-memaccess-binding-constraints.md)
- L4 ARLE vs SGLang head-to-head:
  [`docs/experience/wins/2026-05-25-bench-guidellm-cuda-l4-arle-vs-sglang-headtohead.md`](../experience/wins/2026-05-25-bench-guidellm-cuda-l4-arle-vs-sglang-headtohead.md)
- DSv4 FlashMLA v2 22× prefill:
  [`docs/experience/wins/2026-05-27-dsv4-flashmla-v2-22x-prefill-22x-pre-crash.md`](../experience/wins/2026-05-27-dsv4-flashmla-v2-22x-prefill-22x-pre-crash.md)
- SGLang V4 blog (industry SOTA framing):
  https://www.lmsys.org/blog/2026-04-25-deepseek-v4/
- Industry-SOTA repos cited:
  - vLLM kernels: https://github.com/vllm-project/vllm/tree/main/csrc
  - SGLang sgl-kernel: https://github.com/sgl-project/sglang/tree/main/sgl-kernel
  - FlashAttention-v3: https://github.com/Dao-AILab/flash-attention
  - FlashInfer: https://github.com/flashinfer-ai/flashinfer
  - DeepGEMM (DeepSeek): https://github.com/deepseek-ai/DeepGEMM
  - Marlin (IST-DASLab): https://github.com/IST-DASLab/marlin
  - DeepSeek TileKernels: https://github.com/deepseek-ai/TileKernels
  - Megatron-LM: https://github.com/NVIDIA/Megatron-LM

---

## Self-check (CLAUDE.md §0 SOLID gate)

- **Evidence vs hypothesis**: GPU-time percentages are SOLID (2026-05-14
  nsys trace, cited directly). Wall-clock impact projections are flagged
  Hypothesis where derived (each fix estimate has a "~%" qualifier).
- **Confounders**: each GAP is a single-kernel axis. The SOTA citations
  are file-level so the comparison reasoning is reviewable.
- **Framing trap**: percentages are per-decode-window not per-wall-clock.
  Per CLAUDE.md §0 the per-request wall-clock is ground truth, so the
  "Expected wall-clock" for each fix is **capped** by the per-window %
  (cannot exceed it). I am not claiming any fix dominates beyond its
  kernel-budget share.
- **Gaps deferred**: no detailed nsys re-trace for current-main was run
  (would require GPU, out of audit scope). The 2026-05-14 binding-
  constraints table is the canonical source; current-main may have
  shifted as documented in
  [`2026-05-15-...md`](2026-05-15-dsv4-decode-memaccess-binding-constraints.md)
  §Cross-check. Caller-count instrumentation is the right next step
  before landing any of these fixes.
