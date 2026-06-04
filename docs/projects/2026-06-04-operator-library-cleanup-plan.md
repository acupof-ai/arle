# cuda-kernels operator-library cleanup + optimization plan

Date 2026-06-04. Analysis = grep/source (hypothesis-grade). **Execution gated on: DSv4
port landed (cuda-kernels free) + re-verify zero-callers INCLUDING the new DSv4 code**
(some "dead" MLA/dsv4 symbols may get wired by the port). Do NOT execute concurrently
with the in-flight DSv4 work or while the build references the files being moved.

## Tier 1 — safe, high-ROI (after DSv4 port; re-verify first)
1. **Delete ~16 zero-caller FFI symbols** (re-grep AFTER DSv4 lands): argmax_batch/logprob/batch_logprob, split_qkv, mla_decode_paged_bf16(+_cuda), gdr_prefill_batch/solve, paged_kv_append, kv_cache_to_paged(+_int8), turboquant_{dequantize_kv,dequantize_paged,fast_dequantize_kv,fast_quantize_kv,quantize_kv}, decode_prep_paged_fused_qkv, dsv4_add_local_expert. Files: csrc/misc/gdr_prefill_{batch,solve}.cu, sampling/KV/turboquant legacy. ~500 LOC. RISK low — but **dsv4_*/mla_* must be re-checked vs the DSv4 port** (it may need them).
2. **Reorganize** (pure move + build.rs paths): csrc/misc/{dsv4_attention.cu(1710),dsv4_mhc.cu(434),dsv4_tp_attention_repack.cu} → csrc/attention/; csrc/gemm/{w4_fp8_activation_quant,w4a8_activation_quant,dsv4_deepgemm_ops,dsv4_fp8_cache}.cu → csrc/quant/; clarify the 3 overlapping FlashMLA shim files. RISK low (build.rs already selects per-SM).

## Tier 2 — A/B-gated (fold into perf phase #16)
3. **Custom M=32 dsv4_grouped_gemm.cu (408 LOC) vs native DeepGEMM** — A/B on Qwen3-MoE/DSv4 decode+prefill; if DeepGEMM wins, delete custom + retire ARLE_DSV4_GROUPED_GEMM_M_THRESHOLD (SGLang survey #4).
4. **GEMV template consolidation** — quantized_gemv.cu (3241 LOC) → parameterized (bits 2/4/8) ~800 LOC. RISK med (numerical validation). Keep Q3K/Q4K/Q5K/Q6K per-format.
5. **Fuse SiLU+mul+requant** between gate/up & down GEMM (SGLang #5) — only if A/B >5%.

## Tier 3 — later
Marlin W4A8+W4FP8 merge (~90% overlap, ~300 LOC); LL-vs-normal DeepEP dispatch (#16 scheduling, not kernel).

## No cruft found
build.rs: rerun-if-changed IS recursive (✓), SM-tier logic sound, no dead feature gates.

## Top-5 to schedule (post-DSv4): (1) delete dead [re-verify], (2) reorganize files, (3) DeepGEMM A/B [#16], (4) GEMV template, (5) Marlin merge.
