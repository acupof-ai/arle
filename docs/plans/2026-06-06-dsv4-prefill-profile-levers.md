# DSv4 prefill 调研 — clean §0 profile + lever ranking (implementation-level)

**Date:** 2026-06-06. **Evidence:** `nsys cuda_gpu_kern_sum` on the 4096-tok prefill
(`dsv4_prefill4096_default_nsys`), TP=8/EP=8, 8×H20. Prefill is GPU-compute-bound
(4096 tokens), so kernel-time % ≈ wall-clock % (unlike decode — §0 framing holds).

## The profile (85% in 3 kernels)

| kernel | %prefill GPU | what |
|---|---|---|
| `dsv4_hybrid_attention_kernel` | **31.3%** | CSA/HCA attention over SW + compressed blocks |
| `dsv4_fp8_gemv_batch_tiled_kernel` | **30.2%** | MLA-LoRA projections (wq_a/wq_b/wkv/wo) as SCALAR FP8 GEMV — 1888 instances |
| `dsv4_csa_select_kernel` | **23.6%** | top-512 bitonic block selection — 168 instances |
| `dsv4_compressor_update_kernel` | 5.6% | compressor |
| `ncclAllReduce` | 3.5% | TP comm |
| deep_gemm (MoE) + mhc + rest | ~5.8% | already efficient (MoE DeepGEMM ~1.3% + 0.7%) |

## Lever #1 (PROVEN, lowest-risk) — MLA-LoRA projections → FP8 DeepGEMM (30.2%)

The MLA-LoRA matmuls (`wq_a`/`wq_b`/`wkv`/`wo`) run as the **scalar**
`dsv4_fp8_gemv_batch_tiled_kernel` (30.2%, 1888 instances). For prefill the M
dimension is the token count (4096) → a **tensor-core FP8 DeepGEMM** is far more
efficient: the SAME `deep_gemm::sm90_fp8_gemm` already runs the MoE matmuls at only
~1.3%+0.7% for comparable work. Route the MLA projections through DeepGEMM instead
of the scalar GEMV. This is broader than (and subsumes) the earlier E1 `wq_a|wkv`
fusion design ([`2026-06-06-dsv4-prefill-fused-wqkv-extend.md`](2026-06-06-dsv4-prefill-fused-wqkv-extend.md)).

**Implementation (§0.1 granularity):**
- Sites: `mla_linear` (`attention.rs:~1420`) / the per-projection `dsv4_linear`
  calls in the prefill attention path dispatch `WeightFormat::Dsv4Fp8BlockScaled`
  → `dsv4_fp8_gemv_batch_cuda`. For `seq_len > 1` (prefill), dispatch the FP8
  DeepGEMM grouped path (`dsv4_deepgemm_*`, already wired for MoE) instead.
- Reuse the decode fused-wqkv scratch shape, extended to `max_tokens` (the prefill
  chunk size); the act-quantize → grouped-GEMM pattern is the MoE one.
- Quant note: the weights are already FP8 block-scaled (E8M0 scales); the DeepGEMM
  consumes that layout directly. Activation is quantized BF16→FP8 per the existing
  `dsv4_deepgemm_pack_quantize_bf16_to_fp8` (already in the profile at 0.6%).
- Gate: needle (FP8 DeepGEMM vs scalar GEMV float order differs on near-ties — gate
  on needle, not byte-identity) + **prefill_ms A/B** (4096-tok, same-binary env
  flip). The 30.2% scalar bucket should collapse toward the DeepGEMM tensor-core
  cost (~few %). License on wall-clock TTFT.

## Lever #2 — csa_select cross-layer reuse / skip_topk (23.6%)

`dsv4_csa_select` (top-512 bitonic) is 23.6% (168 instances). SGLang's DSA reuses
the sparse top-k selection ACROSS LAYERS (`deepseek_v2.py:1556` `skip_topk`: "when
True this layer skips computation and reuses previous layer's topk indices") and
skips it entirely when `kv_len ≤ index_topk` (`dsa_indexer.py:1359`). Consecutive
DSv4 layers select similar compressed blocks → reuse the selection every N layers
(or share within a layer group). Gate: needle + prefill_ms; the 23.6% bucket
shrinks by ~the reuse factor. Higher-risk than #1 (correctness of the reuse) —
do it second, validate the selection-reuse doesn't degrade the needle.

## Lever #3 (HARD, deferred) — hybrid_attention 31.3%

The CSA/HCA attention math. FlashMLA-prefill was KILLED (+36% — the prepare-chain
overhead exceeds the attention-math savings). The kernel itself may be inefficient
(scalar-ish vs a tiled/fused version) but rewriting it is high-risk. Defer until
#1+#2 land and re-profile (the bucket re-ranks once the GEMV/select shrink).

## Sequence (按顺序并行推)

#1 (DeepGEMM projections) is the proven, lowest-risk, biggest single bankable
prefill win — execute first (after frozen-KV frees the pod + attention.rs). #2
(csa_select reuse) second. #3 deferred. Each: needle + prefill_ms A/B, license on
wall-clock TTFT. Claude owns the spec; Codex executes; Claude reviews + licenses.
