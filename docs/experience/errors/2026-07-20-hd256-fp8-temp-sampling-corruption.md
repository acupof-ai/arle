# temp=1.0 long-gen degeneration on Qwen3.6-27B — a five-hypothesis false-root-cause chain

> Status: RESOLVED. The temp>0 "salad" is **temperature=1.0 + long generation
> degeneration**, uniform across ALL Qwen3.6-27B variants (base & ThinkingCap,
> FP8 & bf16) on the clean binary `fea8e1fd0`. NOT FP8, NOT the MoE router (there
> is none), NOT ThinkingCap's weights, NOT norm handling, NOT config. Five static
> hypotheses + the driving premise were each killed by measurement. Fix already
> shipped: rollout `--rollout-temperature 0.3` (`2394a2ab0`) is the correct
> operating point. One cheap sampler spot-check open (garbage-token nucleus leak
> vs model tail).

## Context

Adopting a concise-reasoning student (ThinkingCap-Qwen3.6-27B-FP8) for the
agent-OPD lane. temp=1.0 rollouts looked like multilingual token-salad; a prior
session decoded "base FP8 coherent / ThinkingCap FP8 salad, nonascii 0.37" and
built a root cause on it. Every layer of that root cause turned out wrong.

## Root Cause (measured, clean binary `fea8e1fd0`)

Controlled A/B — 3 models × {greedy, temp=1.0 top_p0.95 top_k20} × length:

| | 400 tok | 2000 tok greedy | 2000 tok temp=1.0 |
|---|---|---|---|
| ThinkingCap-FP8 | COHERENT | coherent (loops, no answer) | **SALAD** na0.004 |
| **base**-FP8 | COHERENT | **COHERENT** (code, finish=stop) | **SALAD** ttr0.10 |
| ThinkingCap-**bf16** | COHERENT | COHERENT | **SALAD** na0.018 |

Control: temp=1.0 with **top_k=1** → base & bf16 both COHERENT, finish=stop.
**Temperature is the flipped variable, uniform across weights and quant.** These
are thinking models with no `repetition_penalty`; long unconstrained temp=1.0
sampling degenerates into loops + occasional garbage. Greedy/top_k=1 are always
coherent. Short (≤400 tok) is always coherent — the failure only shows at length.

The prior "nonascii 0.37 multilingual salad, base coherent / ThinkingCap salad"
**does not reproduce on the clean binary** — it was an artifact of a
pre-norm-revert binary (`9851ced6b` era) and/or long unseeded generation.

## The false chain — each killed by measurement (CPU forensics, then GPU A/B)

1. **"MoE router was FP8-quantized" → DEAD.** The model has *no routers*: it is a
   **hybrid linear-attention** (Mamba-style) arch — 48/64 layers are
   `linear_attn.*`, 16 are softmax `self_attn`. The "router 65/65 vs 1/65" table
   was a loose grep matching `.mlp.gate_proj` (dense FFN) as `.mlp.gate` (router).
2. **"dense hd256" → WRONG.** Never checked against `layer_types`; it is hybrid
   linear-attn, head_dim 256, `full_attention_interval` 4.
3. **"FP8 scales broken" → DEAD.** base & ThinkingCap `weight_scale_inv` are
   **bit-identical** (1.48M values, ratio 1.0000, zero 0/inf/nan). The export
   reused base's scale grid — which is fine, see (4).
4. **"FP8 value clipping against frozen scales" → DEAD.** Dequant(ThinkingCap-FP8)
   vs ThinkingCap-bf16 = flat **2.65%** relative error (intrinsic e4m3 block-128
   noise), saturation 0.01%, forced-clip 0.003%. A correct-requant control (scale
   recomputed from bf16 truth) lands on the *same* 0.0265. FP8 is faithful.
5. **"sampling/rope/template/eos config mismatch" → DEAD.** ThinkingCap's
   `generation_config` (temp 1.0/top_p 0.95/top_k 20), rope, `chat_template.jinja`
   (byte-identical sha256), and eos set are all identical to base.
6. **The premise "base coherent / ThinkingCap salad at temp=1.0" → ARTIFACT.**
   Does not reproduce on the clean binary; base & ThinkingCap are indistinguishable.

Separately killed earlier: the norm mis-fix `9851ced6b` (loaded input/post
layernorm as `w−1`, double-offsetting HF's offset norms → greedy salad), reverted
`485eefe0d`/`d50e4782f`. The Metal reference (`mean|w|<0.75 → w+1`) settled that
the CUDA `(1+w)` offset kernel was already correct.

## Fix

- **Shipped (`2394a2ab0`), now re-attributed as correct, not a workaround:**
  `--rollout-temperature` 1.0→0.3 (+ rubric nucleus hygiene top_p0.95/top_k20).
  temp=0.3 (or top_k=1) is coherent across all variants; this is the right
  operating point for these no-rep-penalty thinking models at length, NOT a patch
  for a quant/weights bug. Keep it.
- **Voided:** #55 router bf16 re-export (no routers → no-op); any FP8 requant (FP8
  is faithful); any bf16 swap (bf16 salads identically).
- **Sampler exonerated (closed).** The host sampler (`infer-plan/src/sample.rs:56-109`)
  truncates to top_k *then* cuts top_p at the first cum≥0.95 — by construction the
  drawn token is always within top-20 and its predecessors sum <0.95, so neither
  nucleus-leak condition is possible; the device sampler
  (`csrc/sampling/sampling.cu:327`) does the same. Control confirms the filter is
  live: temp=1.0 top_k=1 COHERENT, top_k=20 garbage. So the occasional non-ASCII
  token at temp=1.0 is genuinely inside the model's own top-20 tail — model
  behavior, not a leak. Fix stays serving-config (temp=0.3). (arle's OpenAI
  endpoint does not implement `logprobs`, so the per-step table couldn't be pulled
  from the API — the source + control settle it.)

## Rule

- **Reproduce the premise on the CURRENT clean binary BEFORE building a
  root-cause chain.** The entire five-probe forensic tower stood on a decoded A/B
  from a prior session whose *sibling* inferences (the router table, "dense") were
  themselves artifacts. Once two claims from that session fell, the premise itself
  owed re-verification first — not more probes built on top. One clean-binary
  A/B at the end overturned everything the CPU forensics had "narrowed."
- **Tensor-name / architecture ground truth, not loose grep.** `.mlp.gate` matched
  `.mlp.gate_proj`; "dense" was never checked against `layer_types`. Read the
  index keys and the arch config before naming a mechanism. See
  [[feedback_validate_comparison_inputs_before_bug]],
  [[feedback_dont_file_hypothesis_as_root_cause]].
- **Temp-graded + length-dependent symptom ⇒ characterize the sampling path
  first.** A/B the sampling variables (temperature, top_k, length) and a reference
  weight/quant BEFORE blaming weights or quant. Here greedy/top_k=1 coherence +
  temp=1.0 salad across base *and* ThinkingCap, FP8 *and* bf16, localized it to
  temperature-at-length in one controlled run. Gate a served model on the actual
  SAMPLED generation at production temperature *and length*, never greedy or short
  alone (2026-05-26-fp8-kv-catastrophic-was-test-artifact).
