# Eval — M_e.8 Tier-1 deterministic regression for INFER_MOE_TOP_K — 2026-05-07

## Goal

Implement and run M_e.8 Tier-1 (deterministic regression) to put data
behind the question: "Is `INFER_MOE_TOP_K=6` safe to flip default-on
for chat workloads?"

Per the research subagent design at
[`docs/plans/M_e8-moe-quality-eval-scaffold.md`](../../plans/M_e8-moe-quality-eval-scaffold.md),
Tier-1 boots `metal_serve` once per top_k value, runs N fixed prompts
at temperature=0 max_tokens=N, diffs token-for-token. Pass criterion:
match_rate ≥ 0.90 AND mean_first_divergence ≥ 32 tokens.

## What was built

`scripts/eval_topk_regression.py` (~270 lines) — server-launch loop +
per-prompt POST + char-level divergence metric. Uses `httpx`, sets
`INFER_MOE_TOP_K` per server boot, supports any number of top_k
arms, writes a JSON results dump for follow-up Tier-2.

CLI:
```bash
./scripts/eval_topk_regression.py \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --top-ks 8,6,4 --max-tokens 256
```

50 default prompts span chat (10) + code (10) + reasoning (10) +
instruction-following (10) + ML-domain (10).

## Tier-1 result on Qwen3.6 35B-A3B-4bit (15 prompts, 128 tok)

```
Baseline: top_k=8

vs top_k=6:
  exact match rate  : 6.67% (1/15)
  mean divergence   : 85.3 chars (~21 tokens)
  median divergence : 13 chars (~3 tokens)
  Tier-1 pass-proxy : FAIL
```

**Hard FAIL on the original Tier-1 pass criterion (90% exact match).**

But the failure is itself a finding, not a regression. Eyeball
inspection of 3 representative diverging pairs shows both arms produce
**coherent, topical, correct** outputs — the divergence is sampling-
trajectory-level, not quality-level:

- **Prompt "Explain why sky appears blue"**:
  - top_k=8 starts with "Here's a thinking process: 1. **Analyze User
    Input**" (meta-reasoning preamble).
  - top_k=6 starts with "The sky appears blue due to Rayleigh
    scattering, which occurs when sunlight interacts with Earth's
    atmosphere. Sunlight, though appearing white, is composed of all
    colors of the [spectrum]" (direct answer).
  - **top_k=6's answer is arguably better** — same model, just
    different sampling.
- **Prompt "List 5 benefits of regular exercise"**:
  - Both arms identical for first ~150 chars, then diverge at "lower
    blood pressure" (k=8) vs "regulate blood pressure" (k=6).
  - Both medically accurate.
- **Prompt "Write a Python Fibonacci"**:
  - Identical outputs through at least 200 chars.

## Diagnosis: exact-match is the wrong gate for top_k changes

The plan's pass criterion (90% exact match) was modeled on Apple's
Recurrent Drafter (https://arxiv.org/abs/2403.09919) where spec decode
is **provably lossless**: target greedy = spec greedy. Top_k=6 vs 8
is **not lossless** — it computes a different function, so exact
match is not even theoretically achievable. The 6.67% match rate
ACTUALLY confirms the env var is doing what it says (changing the
function).

The right interpretation:

| Metric | top_k change | Spec decode |
|---|---|---|
| Exact match rate | "Did the function change at all?" | "Is decode lossless?" |
| Coherence/topicality | The real question | (irrelevant; exact match implies it) |
| Numerical accuracy on benchmarks (HumanEval, GSM8K) | The real gate | (irrelevant for lossless) |

For top_k changes, **Tier-1 is a "function-still-changes" sanity
check, not a quality gate.** A 100% match would actually mean the
env var WASN'T routing through MoE differently. The 6.67% rate (1
identical, 14 different) is consistent with sub-50%-of-tokens being
expert-routing-sensitive (most tokens have a clear top-1 expert
that's the same in both top_k=6 and top_k=8 routing windows).

## Decision

1. **Tier-1 ships as a per-commit "did the knob still behave"
   regression check**, not a default-flip gate.
2. **Tier-2 (HumanEval pass@1 + GSM8K accuracy) becomes THE
   default-flip gate.** Run before any commit changing top_k default.
3. **Update the plan doc to reflect this** — "match_rate ≥ 0.90"
   was wrong; better is "match_rate stays stable across commits at
   the same top_k". Actually relevant gate: regression catches a
   commit that *unintentionally* broke top_k=6 by causing massive
   re-routing.

## Tier-1's correct value proposition

For a per-commit CI-style check, the right pattern is:

- Record baseline at first commit: e.g. top_k=6 produces N specific
  outputs on the 50 fixed prompts.
- On every subsequent commit, re-run; assert N matches stay > some
  threshold (e.g. 95% of last-commit's outputs).
- Catch: a commit that breaks MoE routing in a way that makes top_k=6
  produce DIFFERENT outputs than yesterday.

Versus original "matches top_k=8" which can never pass.

## What worked

- **Subagent-designed Tier-1 script delivered first try.** The
  server-launch loop pattern from `eval_ppl.py` ported cleanly.
  ~270 lines including 50-prompt set, eval runner, divergence
  metric, JSON output.
- **Eyeball inspection of 3 diverging pairs** was enough to see
  both arms produce coherent answers. Quick gut-check before
  drawing conclusions.
- **Server-boot loop is reliable** — both arms came up in <2 min
  with the auto-wired-limit.

## What didn't work as planned

- **Pass criterion was wrong.** "match_rate ≥ 0.90" can't be met
  by definition for top_k changes. Need to revise the plan.
- **15 prompts, 128 max_tokens is slim** — would want 50/256 for
  publishable numbers. Initial fast run was right call to validate
  the script; full run is a follow-up.

## Rule

When porting a literature pass criterion from one domain (spec
decode = lossless) to another (MoE top_k = quality knob), check
whether the underlying assumption holds. Spec decode → exact match
is THE gate. Quality knob → exact match is meaningless; numerical
benchmark accuracy (HumanEval pass@1, GSM8K accuracy) is the gate.

Generalizing: for any optimization that's mathematically
equivalent (oMLX-C v3, M_e.4 SwiGLU compile), exact match IS the
right Tier-1 gate. For any optimization that's deliberately
quality-trading (top_k reduction, KV quantization, lower-bit weights),
exact match is wrong; numerical accuracy is right.

## Next

- **Tier-2 (HumanEval pass@1, GSM8K accuracy)** — extends
  `scripts/eval_ppl.py` per the M_e.8 plan. Run at top_k ∈ {4, 6, 8}.
  Decide default flip if Δ < 3% on either benchmark.
- **Update M_e.8 plan** — clarify Tier-1 is a "is MoE behaving
  consistently" gate, not a "did we beat the baseline" gate.
- **Don't flip top_k default yet.** Stays opt-in until Tier-2.

## References

- Plan source:
  [`docs/plans/M_e8-moe-quality-eval-scaffold.md`](../../plans/M_e8-moe-quality-eval-scaffold.md)
- Predecessor (the knob):
  [`2026-05-07-bench-qwen36-topk-sweep.md`](2026-05-07-bench-qwen36-topk-sweep.md)
- Apple Recurrent Drafter (where the exact-match-criterion comes
  from): https://arxiv.org/abs/2403.09919
- Implementation: [`scripts/eval_topk_regression.py`](../../../scripts/eval_topk_regression.py)
