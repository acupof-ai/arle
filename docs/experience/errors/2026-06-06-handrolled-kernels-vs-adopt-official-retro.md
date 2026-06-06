# Retrospective: hand-rolling kernels instead of adopting the official/vendored ones — the expensive default-inversion

## Context

DSv4-Flash perf campaign (decode 6ms goal). After weeks of per-kernel work, ckl forced a
reframe: "别自己写算子,先抄业界最好的…用官方的或者开源优化好的替换…像素级学习 sglang."
Pixel-level study of SGLang's DSv4/Qwen3.5/Qwen3.6/Gemma4 backends (the full source at
/tmp/sglang-full + sgl-kernel) + an end-to-end wall-clock trace exposed that ARLE's
hand-rolled kernels were the bottleneck — and the official replacements were mostly
**already vendored, just unwired**.

## Root Cause (the failure pattern, ordered by cost)

1. **闭门造车 with the official kernel already in `vendor/` unwired.** ARLE hand-rolled
   degraded duplicates of kernels that were sitting in `vendor/flashmla` + `vendor/deepgemm`:
   - `dsv4_csa_select` (1 CTA → **1/78 SMs** at B=1 decode, 101ms/token = 74.9% of decode)
     duplicates the official DeepGEMM `fp8_paged_mqa_logits` (DSA "lightning indexer").
   - the scalar FP8 GEMVs, `moe_grouped_gemm` (self-labeled "no tensor-core, sm_70-safe")
     duplicate DeepGEMM grouped FP8 GEMM. FlashMLA's MLA kernel was unwired for weeks
     (prior instance, [[feedback_no_closed_door_solutions]]).
2. **An "ours" mental frame manufactured the anti-pattern.** Treating SW/CSA/HCA as
   "ARLE-original on top of MLA" produced 3 scalar kernels where SGLang runs ONE fused kernel.
3. **Narrow / smoke-shape profiling chased the wrong bottleneck.** An 8-token decode profile
   said "comm 32.4% / GEMV"; the real 4096-SLO bottleneck (csa_select 75%) only surfaced via an
   end-to-end **wall-clock** trace. `cuda_gpu_kern_sum` % overstates wall (stream overlap) —
   effort went to overlap-protected GEMV + a +9% per-projection DeepGEMM kill.
4. **The rewrite (R6) dropped continuous batching.** Single-row executor; c>1 decode ERRORS.
   A self-built throughput ceiling vs SGLang's paged-KV continuous batching.
5. **Vendoring gaps left the heaviest axes hand-rolled.** GDN linear-attn + GQA attention have
   no vendored equivalent → hand-rolled (slow; GDN WGMMA hang). Should vendor FLA / FlashInfer /
   causal_conv1d.
6. **Quant not wired into the clean paths.** Marlin + DeepGEMM vendored but Qwen3.5's BF16 path
   carries quant only as TODOs.
7. **Effort misallocation.** Time spent hand-optimizing (per-projection DeepGEMM, a nearly-landed
   2-kernel csa split) instead of wiring the vendored official kernel. The 2-kernel split was
   interrupted by ckl mid-dispatch.

**One-line root cause: the default was inverted — "self-develop/optimize" was the default and
"adopt official" was the fallback.**

## Fix

Invert the default. For each compute operator: **adopt the official/open-source kernel first**
(FlashMLA / DeepGEMM `fp8_paged_mqa_logits` / FlashInfer / FLA / Marlin — check `vendor/` first,
it may be present-but-unwired), **mirror SGLang's integration pixel-level** (the exact dtype /
scale / layout / metadata / topk-transform — e.g. DSA needs Hadamard-rotate→FP8→paged→logits→
topk_transform, not a BF16 hand-roll), and **self-develop only when an A/B proves the hand-roll
beats OSS** (defer that A/B; correctness-gate the adoption now). Keep irreducible orchestration
glue (EP routing, paged-KV layout, TP repack). Map: `docs/plans/2026-06-06-dsv4-handrolled-kernel-audit.md`
+ the per-model specs `2026-06-06-{qwen35,qwen36,gemma4}-official-adoption-spec.md`.

## Rule

**先抄业界最好的(官方/开源,多半已 vendored),再考虑自研;自研只在 A/B 证明更好时才保留。**
Profile on the SLO shape with wall-clock framing (not smoke-shape kernel-sum %). Mirror the
upstream integration posture exactly — a guessed dtype/scale/layout is silent garbage.
See [[feedback_no_closed_door_solutions]] for the operative procedure.
