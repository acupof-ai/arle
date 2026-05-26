# FP8 KV "catastrophic divergence" on Qwen3-4B was a test-methodology bug

## Context

Across 2026-05-02 → 2026-05-26 the FP8 KV path on Qwen3-4B (and
Qwen3-dense generally) was repeatedly characterized as "catastrophic
step-1 divergence" with `mean_match=0.0156` vs BF16 in
`infer/tests/kv_precision_parity.rs`. The Qwen3.5 hybrid (8 full-
attention layers) was characterized as "milder drift". Multiple
investigations attempted structural fixes:

- 2026-05-02: tier-1 FP8 KV numerical fail
- 2026-05-05: tier-1 FP8 KV still failing
- 2026-05-12: FP8 kill series
- 2026-05-26 (f50dd674): multi-layer dump → "precision-floor
  compounding across 36 dense layers" hypothesis, FP8 routed off
  `auto`-default
- 2026-05-26 (8c6d92db, this session): KIVI per-channel K
  implementation (V1)
- 2026-05-26 (049b2fc0, this session): KIVI killed as "insufficient
  fix" based on bit-identical `mean_match=0.0156` after KIVI engaged

All of these treated the metric as a quality signal. **The metric was
not a quality signal.**

## Root cause

Direct token decode of the audit sequences on A100 sm_80
(Qwen3-4B base, greedy, prompts from `DEFAULT_PROMPTS`, prompt 0,
first 8 generated tokens):

| Precision | Token IDs                                            | Decoded                              |
|-----------|------------------------------------------------------|--------------------------------------|
| bf16 ref  | `[0, 0, 0, 0, 0, 0, 0, 0]`                            | `"!!!!!!!!"` (token 0 = `!`)         |
| int8      | `[0, 0, 0, 0, 0, 0, 0, 0]`                            | `"!!!!!!!!"`                         |
| fp8       | `[0, 1124, 42469, 1671, 323, 5144, 7894, 198]`        | `"! \\ntenists and/or_element\n"`    |
| tq4       | `[4710, 108843, 13, 33222, 279, 116, 13, 55778]`      | divergent fragments                  |

**The BF16 reference itself is degenerate**: Qwen3-4B *base* (not
instruct) under greedy decoding on long technical-system-design
prompts collapses to a single-token repetition loop (token `!`). This
is well-known behavior for base LMs on out-of-distribution greedy
decode — there is no instruction-following objective forcing the
generation into a coherent trajectory.

INT8's `mean_match=1.0000` is therefore not "INT8 has perfect parity
with BF16 quality" — it is "INT8's quant noise is small enough to
faithfully reproduce the degenerate `!`-loop output". FP8's
`mean_match=0.0156` is not "FP8 is catastrophically broken" — it is
"FP8's slightly larger quant noise *breaks out* of the `!`-loop and
generates real text fragments that do not match the BF16 nonsense
baseline".

`mean_match` measures token-trajectory match against a reference. It
does *not* measure generation quality. When the reference itself is
junk, the metric becomes "do you faithfully reproduce the junk?", and
the *most* numerically faithful path (INT8) scores highest. A real
quality metric (perplexity on a held-out corpus, lm-eval-harness
benchmarks, BLEU/ROUGE on standardized tasks) would have inverted the
ranking — likely showing FP8 close to or even slightly better than
INT8 in coherent-text generation.

## Decision

1. **Retract** the 2026-05-26 KIVI kill entry
   (`2026-05-26-kivi-per-channel-k-insufficient-for-qwen3-4b-fp8.md`)
   — its conclusion is invalid because it relied on `mean_match` to
   judge KIVI's effectiveness, and `mean_match` is not a quality
   metric under a degenerate reference.

2. **Keep** the KIVI implementation
   (`8c6d92db`/`73a72615`/`25c7d409`/`0ef57994`) — it is correct
   per unit tests, the normalize fix (`73a72615`) was a real
   pre-existing bug in the partial-kernel `o_s` write, and the floor
   fix (`25c7d409`) was a real pre-existing 1e-6-clip bug in the
   FP8 quant scale. Both fixes apply to legacy FP8 too; both improve
   numerical correctness regardless of the test methodology.

3. **Fix the test methodology**: add a degenerate-baseline detector
   (refuse to draw conclusions if the BF16 reference is a single
   repeated token), and switch (or extend) the audit to use a
   prompt/model combination that produces real coherent generation
   under greedy decode — e.g., short instruction-style prompts on
   the same base model, or the instruct variant if available.

4. **Re-audit FP8** under the corrected methodology before any new
   kill/license decision on FP8 KV's production readiness.

## Rule

**A quality conclusion is only as valid as its reference.** Before
treating a `mean_match`-style trajectory metric as evidence of
quality, decode the reference itself and verify it produces sensible
output. Greedy + base LM + long prompts is a known degenerate path —
any test built on that combination is measuring noise-fidelity, not
quality.

When two precisions tie at `mean_match=1.0` on a long-token horizon,
that is *itself* a yellow flag: either the model is extraordinarily
robust to quant noise (rare), or both precisions are reproducing a
degenerate reference (much more common). Investigate by **dumping
the actual generated tokens** before drawing conclusions about
quality.

This rule cost ~3 weeks of "FP8 KV is broken" investigation
(2026-05-02 → 2026-05-26, five docs, three implementations, one
kill, one retract) on a problem that was a test-framework artifact.
The cheap diagnostic (decode and print 8 tokens of each sequence)
was a single eprintln away the entire time.

## Related

- `docs/experience/errors/2026-05-26-kivi-per-channel-k-insufficient-for-qwen3-4b-fp8.md`
  — the retracted kill entry; conclusion invalid for the reasons
  documented here.
- `docs/experience/errors/2026-05-26-fp8-kv-step1-divergence-known-deferred.md`
  — f50dd674; "precision-floor compounding" hypothesis was layered
  on top of the bad metric, also needs re-evaluation under a sane
  reference.
- Commits 8c6d92db, 73a72615, 25c7d409, 0ef57994 — KIVI infra +
  two legacy-FP8 correctness fixes; all kept.
