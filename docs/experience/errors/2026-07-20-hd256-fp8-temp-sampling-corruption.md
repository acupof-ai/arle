# hd256/FP8 temp>0 sampling corruption silently degraded every OPD rollout

> Status: root fix in progress (plan:
> [2026-07-20-hd256-fp8-temp-sampling-corruption](../../plans/2026-07-20-hd256-fp8-temp-sampling-corruption.md));
> temp=0.3 workaround shipped (2394a2ab0).

## Context

Adopting a concise-reasoning student (ThinkingCap-Qwen3.6-27B-FP8) for the
agent-OPD lane. Serve-side sampling looked broken: temp=1.0 requests returned
multilingual token-salad. First read: my new `SamplingDefaults` fix wasn't
threading nucleus to the CUDA sampler.

## Root Cause

Two layers, only the second is the real bug:

1. **Not the sampler.** A control A/B on the same binary — hd128 Qwen3-4B
   coherent at temp=1.0+nucleus, hd256 model salad — proved the nucleus DOES
   reach the CUDA sampler. The `SamplingDefaults` fix is fine.
2. **hd256/FP8 temp>0 distribution corruption.** The hd256 (Qwen3.6-27B) FP8
   path produces a distribution whose **argmax is correct** (greedy coherent —
   `b4b293f0c` fixed the q/k RMSNorm OFFSET→STANDARD convention) but whose
   **tail is mis-scaled/noisy**. Temperature is the "trust the tail" knob: at
   0.3 the sharpened distribution suppresses the residual error (coherent), at
   1.0 the flat distribution samples it (salad). Temperature-graded (clean
   ≤0.3, onset ~0.6, salad 1.0), NOT length-driven (0.3 holds over 24K chars).
   Decoded on BOTH the production student and ThinkingCap → generic hd256/FP8,
   not checkpoint-specific.

**The silent damage:** `--rollout-temperature` defaulted to 1.0 (F.6, for
non-empty behavior logprobs). Every agent-OPD rollout on the hd256/FP8 student
was therefore sampling the corrupted distribution — degradation we had been
attributing to thinking-chain length / timeouts.

**Confounder resolved — it is the hd256 COMPUTE residual, FP8-independent.**
Root cause pinned & patched (`9851ced6b` + `bf66a3854`): the per-layer
`input_layernorm` / `post_attention_layernorm` were loaded raw, but they ship in
STANDARD format (~1-centered) and the `rms_norm_offset` trunk kernel applies
`(1 + weight)` → a ~2× multiplier per layer, compounding across 64 layers.
`b4b293f0c` had fixed only the q/k norms. Fix: load these norms as `(w − 1)` too
(`load_final_norm_offset`). Still owes an empirical temp=1.0 gate on a rebuilt
binary (the pod binary predates the fix) — a 2×/layer error "should" have broken
greedy too, so the mechanism magnitude is measured, not assumed.

## Fix

- **Workaround (shipped, 2394a2ab0):** `--rollout-temperature` default 1.0→0.3
  (still >0, F.6 logprobs intact) + rubric-lane nucleus hygiene (top_p 0.95 /
  top_k 20). Verified by decoded-case coherence A/B + a long thinking-on
  nonascii=0.0000 gate on the SLO shape.
- **Root fix (pending, #55):** branch on Phase 0 — FP8 quant/dequant precision,
  or a hd256 compute-convention residual. Then restore temp=1.0.

## Rule

**Gate a served/quantized model on the actual SAMPLED generation at the
production temperature, never on greedy alone.** Greedy (argmax) survives a
mis-scaled distribution tail that temp>0 sampling turns into garbage — a
"greedy coherent" smoke test passes a model that is broken for every real
rollout. When temp>0 output degrades, A/B the ONE variable (head_dim, quant,
temperature) before blaming the nearest recent change: the salad here looked
like a sampler-plumbing regression but was an orthogonal kernel-numerics bug two
lanes away. See also [[feedback_validate_comparison_inputs_before_bug]] (decode
the cases) and the greedy-decode-the-generation lesson
(2026-05-26-fp8-kv-catastrophic-was-test-artifact).
