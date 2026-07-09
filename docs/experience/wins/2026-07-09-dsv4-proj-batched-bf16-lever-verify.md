# DSv4 `ARLE_DSV4_PROJ_BATCHED_BF16` lever — wired + zero-cost verified; correctness effect masked by same-day Route A regression

## Context

#150's recommended mitigation landed as an opt-in flag (`a9beff4d1`,
`crates/infer-cuda/src/attention.rs::proj_batched`): force the bf16 cublasLt
path for the batched decode compressor/indexer projections at m>1, skipping
the FP8-repack DeepGEMM lane. Pod verification 2026-07-09 (TP=4, GPUs
4/5/6/7, `DeepSeek-V4-Flash-FP8`, rebuilt `concurrent_needle_v3.py`,
len=500, `--max-total-tokens 2048`).

## What Worked

- **Engagement proven, zero-code probe**: `ARLE_DSV4_PROJ_BATCHED_BF16=bogus`
  boots clean, serves n=1, and the FIRST n≥2 decode step fails loudly
  (`unsupported ARLE_DSV4_PROJ_BATCHED_BF16 'bogus'`) — the flag is read on
  exactly the batched lane and nowhere else. `strings` probe on the binary
  confirmed the symbol (=1).
- **Zero perf cost**: n=2 mean per-request wall 1.624s (off) vs 1.601s (on),
  Δ −1.4%; per-trial wall Δ −1.8% — noise-level, the lever is free when the
  DeepGEMM lane isn't the binding constraint.
- **Correctness effect NOT re-measurable at HEAD**: both arms sat at ~82%
  n=2 miss because HEAD carried the same-day Route A FlashMLA-lane
  regression
  ([errors/2026-07-09-dsv4-route-a-flashmla-needle-regression-bisected.md](../errors/2026-07-09-dsv4-route-a-flashmla-needle-regression-bisected.md)),
  which swamps the FP8-gate signal (108/108 misses were the regression's
  truncation class, zero digit-substitution). The lever's license basis
  remains Experiment B (2026-07-07, pre-window: n=2 miss 57.1%→30.0%, its
  truncation class eliminated). Re-run the A/B after the regression fix
  lands before any default-flip consideration.

## Rule

A correctness lever's A/B needs a clean baseline — verify the baseline
reproduces its own historical floor FIRST (one solo sweep) before spending
two boots on the lever arms; here the baseline check itself surfaced a P0
regression worth more than the A/B.
