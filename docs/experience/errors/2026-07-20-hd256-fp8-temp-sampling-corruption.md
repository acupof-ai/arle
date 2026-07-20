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

**A wrong "fix" (`9851ced6b`) sent us on a norm detour — reverted.** It loaded
the per-layer `input_layernorm`/`post_attention_layernorm` as `w−1`, assuming
STANDARD format. But these norms ship OFFSET (HF convention, `mean|w| ≈ 0.24 <
0.75`) and the CUDA `(1+w)` kernel already matches that raw — confirmed by the
Metal reference, which detects offset (`mean|w| < 0.75`) and converts with `w+1`,
identical to the CUDA kernel. `9851ced6b` double-offset input/post →
`(1+(w−1)) = w ≈ 0.24` → 1/5 scale/layer → greedy SALAD (decoded regression,
2 clean builds). Reverted (`485eefe0d`). The final norm (`mean|w| 3.3 > 0.75`,
direct) correctly stays `w−1`. **The norm handling was never the temp>0 bug.**

**The real bug — MoE routing flip from a broken FP8 export (NOT our runtime,
NOT generic FP8 noise).** Localized by config diff + same-binary A/B:

| bf16-kept | base Qwen3.6-27B-FP8 | ThinkingCap-FP8 |
|---|---|---|
| `mlp.gate` (router) | 65/65 layers | **1/65** |
| `shared_expert_gate` | 65/65 | **0/65** |

Same reverted binary, temp=1.0: **base (bf16 router) coherent (nonascii 0);
ThinkingCap (FP8 router) salad (nonascii 0.37)**. The only structural difference
is router quantization — identical FP8 expert/attention GEMMs, identical bf16
lm_head/norms. FP8-quantizing the MoE router perturbs its gate logits →
per-token top-k expert selection flips → wrong experts corrupt the residual tail
→ greedy top-1 survives, temp>0 samples the scrambled tail → salad.

**Two corrections to earlier reads:** (1) it is NOT smooth FP8 accumulation — if
it were, the base (same expert GEMMs) would salad too; it doesn't. (2) "The whole
OPD lane silently degraded" was a phantom — the base FP8 student is fine at
temp=1.0; the earlier "current student salad" was on the norm-regression binary /
conflated with ThinkingCap.

Router weights are stored lossy-FP8 in ThinkingCap's checkpoint → a loader
force-bf16 can't recover the lost precision (dequant keeps the loss). **Fix
(chosen): re-export ThinkingCap-FP8 with `modules_to_not_convert` covering all
`mlp.gate` + `shared_expert_gate` (match the base), from the bf16 ThinkingCap we
already have.** experts/attention stay FP8. Then restore rollout temp=1.0.

Revert A/B (greedy, temp=0.0): reverted binary → both FP8 models coherent
("Paris"/"64", nonascii 0); `00224faa0` (9851ced6b) → salad. Revert confirmed.

## Fix

- **Workaround (shipped, 2394a2ab0):** `--rollout-temperature` default 1.0→0.3
  (still >0, F.6 logprobs intact) + rubric-lane nucleus hygiene (top_p 0.95 /
  top_k 20). Verified by decoded-case coherence A/B + a long thinking-on
  nonascii=0.0000 gate on the SLO shape.
- **Root fix (pending, #55):** branch on Phase 0 — FP8 quant/dequant precision,
  or a hd256 compute-convention residual. Then restore temp=1.0.

## Rule

**Never FP8-quantize a MoE router.** The gate is tiny and routing is
discrete — a ~1-2% weight perturbation flips top-k expert selection, and greedy
hides it (top-1 survives) while temp>0 exposes it as salad. Correctly-exported
FP8 checkpoints keep `mlp.gate`/`shared_expert_gate` bf16 (vLLM/SGLang standard);
**verify a third-party quant export's `modules_to_not_convert` before trusting
it** — a config diff against a known-good sibling checkpoint localizes a broken
export in 2 minutes, no rebuild.

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
