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

**The real bug — isolated to FP8 (dtype), confirmed by A/B.** Same reverted
binary, same prompt/sampling (temp=1.0 top_k20 top_p0.95), hd256 27B:
- **bf16** → `The capital of France is Paris` coherent, **nonascii 0.000**
- **FP8** → `Paris パリ Παρίσιμο Париж…` salad, **nonascii 0.409**

bf16 clean ⇒ NOT a hd256 compute residual (that would corrupt bf16 too). The
temp>0 salad is **FP8 logit-tail noise**: FP8 quantization perturbs the logit
tail; greedy argmax survives, temp>0 samples it. Fix lives in the FP8
quant/dequant or FP8 compute precision, not the hd256 kernels. Localization
(per-layer bf16↔FP8 divergence + MoE routing agreement) in progress. temp=0.3
workaround holds until fixed.

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
