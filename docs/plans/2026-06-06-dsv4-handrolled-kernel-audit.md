# Hand-rolled CUDA kernel audit — 删除所有自己写的算子 (adopt official/vendored)

**Date:** 2026-06-06. **Driver (ckl):** "用官方的或者开源优化好的替换。先处理好 dsv4 的,不要
用任何自研算子。自研算子只能在比开源的更好的时候才用,先不要全部删除。" **Principle:**
[[../../memory/feedback_no_closed_door_solutions]].

**Operative rule (refined):** **DSv4 first.** Bar = **zero self-developed compute operators
UNLESS proven better (same-binary A/B) than the best official/open-source kernel.** Replacement
source = official (FlashMLA / DeepGEMM / DSA) OR well-optimized OSS (FlashInfer / FLA / Marlin /
cutlass). **Do NOT bulk-delete** — it's a **per-operator license-or-kill**: A/B each hand-rolled
compute kernel vs the best OSS, adopt OSS + delete the hand-roll if OSS ≥, KEEP the hand-roll
only if it WINS with evidence. (csa_select clearly loses — 1/78 SMs — so it goes.)

**Scope of "算子":** the hand-rolled COMPUTE kernels that duplicate an official/vendored kernel.
NOT in scope (genuine, irreducible — deleting them breaks the engine, they have no upstream
drop-in): ARLE-specific orchestration glue (EP routing scatter/scan, paged-KV layout, TP repack,
dtype convert), and env-gated diagnostics. **Classification is first-pass from op names + the
vendored API; each DELETE row needs per-kernel verification of the official equivalent's call
shape before deleting (§0 — hypothesis until the wire-up + needle/A-B passes).**

Inventory: ~280 `__global__` across `crates/cuda-kernels/csrc/{attention,gemm,kv,quant,misc,moe}`.

## A. DELETE + ADOPT — hand-rolled compute that duplicates a vendored/official kernel

| hand-rolled | file | official replacement | vendored? | priority |
|---|---|---|---|---|
| `dsv4_csa_select` (DSA indexer scoring + bitonic top-k) | misc/dsv4_attention.cu | **`fp8_paged_mqa_logits` + `clean_logits`/topk** (DeepSeek DSA indexer, = SGLang) | ✅ vendor/deepgemm | **#39 IN PROGRESS** (decode 74.9%) |
| `dsv4_hybrid_attention` (CSA/HCA prefill attn) | misc/dsv4_attention.cu | **FlashMLA `sparse_fwd`** (prefill sparse MLA) | ✅ vendor/flashmla/sm90/prefill/sparse | P1 (prefill 31%; "killed +36%" was the hand-roll framing — re-judge the official kernel) |
| `dsv4_swa_attention` (sliding-window attn) | misc/dsv4_attention.cu | FlashMLA / FlashInfer SWA, or the SW path of FlashMLA sparse | ⚠ verify | P2 |
| decode attn (already done) | — | FlashMLA `sparse_decode` | ✅ wired | ✅ DONE |
| scalar FP8 GEMV `dsv4_fp8_gemv_batch_*` | gemm/quantized_gemv*.cu | **DeepGEMM fp8_gemm** (where M warrants; #36 found prefill overlap-protected, decode M=1 is GEMV-bound — adopt where DeepGEMM wins) | ✅ vendor/deepgemm | P2 (decode 14.4% bucket, partial) |
| generic `gemv` / `turboquant_weight_gemv` | gemm/gemv.cu, turboquant_weight_gemv.cu | DeepGEMM / cutlass / Marlin | ✅/⚠ | P3 |
| Qwen quant-attention (`decode_attention_quantized/turboquant/varlen_fp8`, `prefill_attention*`, `fused_attention`, `nonpaged_prefill`) | attention/*.cu | **FlashInfer** (paged quant attn) or FlashMLA; TileLang-gen ones are already an adopted lib | ❌ needs vendoring | P2 (Qwen track) |
| `dsv4_mhc_*` (HC/Sinkhorn mix) | misc/dsv4_mhc.cu | SGLang fused `mhc_pre_big_fuse` (TileLang) | ⚠ adopt SGLang's | P3 (decode 12.2%) |
| `conv1d*`, `gated_delta_rule*` (Qwen3-Next GDN) | misc/conv1d*.cu, gated_delta_rule*.cu, gdr_* | **FLA** (flash-linear-attention) library | ❌ needs vendoring | P3 (Qwen3-Next) |

## B. KEEP — already-adopted (shims wrapping vendored official kernels)

- gemm/`deepgemm_native.cu`, `deepgemm_bridge_stub.cu`, `dsv4_deepgemm_ops.cu`, `dsv4_fp8_cache.cu` → wrap vendored **DeepGEMM**. ✅
- gemm/`marlin_*.cu` (`marlin_gemm`, `gemm_w4_fp8_marlin`, `gemm_w4a8_marlin`, repack, preprocess) → **Marlin** is an adopted upstream W4A8/W4A16 kernel. ✅
- misc/`arle_flashmla_shim.cu`, `arle_flashmla_decode_shim.cu`, `arle_flashmla_csa_prep.cu` → wrap vendored **FlashMLA**. ✅ (csa_prep is the index-format glue for the FlashMLA sparse path.)

## C. KEEP — genuine ARLE glue (no upstream drop-in; deleting breaks the engine)

- moe/`dsv4_route.cu` (31 ops: scan/fill/scatter/pack/count/dispatch-payload) → **EP routing orchestration** around DeepEP+DeepGEMM. The expert GEMM is DeepGEMM (adopted); this is the dispatch/combine plumbing. (Some scatter/combine could move to DeepEP kernels — verify case-by-case, but most is irreducible orchestration.)
- kv/`kv_cache_to_paged`, `scatter_kv`, `paged_kv_append`, `paged_kv_metadata`, kv/`kv_quant` → **paged-KV layout + KV-quant plumbing**. ARLE paged-pool specific. (KV-quant numeric kernels — see TurboQuant below.)
- misc/`split_qkv`, `dsv4_tp_attention_repack`, `arle_dtype_convert`, `fused_mlp`, `elementwise_basic` (add/silu_mul/embedding) → small glue/fusions. (silu_mul/embedding have FlashInfer equivalents but are trivial — low ROI.)
- misc/`norm.cu` (RMSNorm), `sampling.cu` → small ops; FlashInfer has official versions (norm, sampling) — **low priority adopt** (correctness-trivial, not bottlenecks per the trace). Borderline B/C.
- attention/`dsv4_fp8_kv_pack`, `dsv4_flashmla_decode_build_indices`, `decode_prep_paged*`, `prefill_attention_paged_prep` → index/KV-format prep for the FlashMLA path (glue feeding the adopted kernels).
- `arle_dsv4_output_inverse_rope`, `dsv4_prepare_qk*`, `dsv4_update_window_cache`, `dsv4_compressor_update` → DSv4 MLA RoPE/compressor glue (verify dsv4_compressor vs the official DSA compressor — possible B-tier adopt).
- quant/`turboquant*` (Lloyd-Max + Hadamard KV quant) → ARLE KV-quant (QuaRot-family); no clean vendored drop-in. KEEP unless a KV-quant lib is adopted.

## Sequence
1. **#39** csa_select → fp8_paged_mqa_logits (in progress — proves the DeepGEMM-attention wire-up pattern).
2. **#40 P1:** dsv4_hybrid_attention → FlashMLA sparse_fwd (prefill 31%; vendored). Re-judge the "+36% kill" on the OFFICIAL kernel, not the hand-roll.
3. **P2/P3** as ranked, each: wire official → delete hand-rolled → gate needle + same-twice + A/B.
Verify each official call-shape before deleting (the DELETE rows are hypotheses until the wire-up passes). Glue (C) stays.
