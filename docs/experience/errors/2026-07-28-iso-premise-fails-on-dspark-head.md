# ISO near-isospectral premise fails on the DSpark head — no objective separates drift, 2026-07-28

> Verdict: **negative premise.** ISO stays non-default; do NOT propagate to
> Agent-RFT (#32 stays gated). Resolves the 2026-07-27 INCONCLUSIVE sweep — both
> confounds fixed, premise still does not hold.

## Context

Phase 5's gate: license ISO only if the pure-PG arm shows the paper's
near-isospectral behavior AND the positive control (dense supervision) moves the
spectrum hard. The paper (arXiv:2607.19331) reports κ_spec 1.02–1.35 for RLVR vs
89–1364 for dense SFT — a ~100× separation the drift instrument must reproduce
before ISO's "put moments on the frames, freeze Σ" is justified.

The 2026-07-27 sweep was inconclusive on two confounds: a cold head (w2≈0 gated
w1's gradient) and a numerically inert α=1 arm (PM loss underflowed f32). Both
are now fixed — nonzero warm-started head, PM loss normalized per-token
(2026-07-28 win). This is the clean re-run.

## The measurement

Qwen3.6-27B-FP8 single-GPU, ISO-off, warm head (w1 L2=113, w2 L2=0.71 — both
factors nonzero, both get real gradient), PM-fixed, w1 axis:

| α | w1 spectrum_drift | w2 drift | loss | accept_ema |
|---|---|---|---|---|
| 0 (pure PG) | 2.6e-6 (2.18→3.11e-6) | 0.17→1.26 | ~−0.4 | 0.398→0.329 |
| 0.5 | 3.67e-6 (range to 1.72e-5) | 0.12→0.93 | ~−0.05 | 0.398→0.317 |
| 1 (pure PM) | **4.21e-6** | 0.078 | **~0.35** (live) | 0.398→0.376 |

Raw α=1 line, captured directly off `alpha_1/serve.log`:
`pm_alpha=1.00 spectrum_drift=[4.21e-6, 7.77e-2]`. All w2 drifts are real
O(0.1–1) now (not the cold run's 1e13 divide-by-zero), and the α=1 loss is ~0.35
— a **live** dense-PM gradient, not the pre-fix null 0.0000. So the head genuinely
moved under every objective; the instrument works.

## Root cause of the negative

**All three arms drift ~1e-6 regardless of the objective.** The α=1 arm was the
positive control: pure probability-matching is dense self-distillation, the exact
SFT regime the paper says moves the spectrum ~100×. It moved w1's spectrum by
4.21e-6 — ~1.6× pure PG's 2.6e-6, not the 100× the premise requires.

So the near-isospectral behavior is **a property of this low-rank DSpark head,
not evidence that RLVR uniquely preserves the spectrum.** With no separation
between the RLVR arm and the dense-supervision arm, the drift instrument cannot
distinguish "PG preserves the spectrum" (the premise) from "this head's spectrum
barely moves under anything" (a structural artifact). The rank-256 head over a
248,320 vocab has σ² concentrated in a few directions that a small proximal step
cannot rotate far — the paper's premise was measured on full transformer weight
matrices, a different regime.

## Impact

- **ISO on the DSpark head is not licensed.** No premise → no A/B → ISO stays
  behind `--dspark-train-iso`, off by default. The implementation is correct
  (local invariant/chain-rule/resume gates green); it is the *premise* that fails,
  not the code.
- **#32 (Agent-RFT ISO) stays gated and cannot inherit a license from here.**
  Two independent reasons now: (1) this premise is negative on the one RLVR-shaped
  head we had; (2) DSpark itself was repriced net-negative for concurrent serving
  after FA3 (2026-07-27), so there is no acceptance win to A/B against regardless.
  Agent-RFT ISO, if ever revisited, must re-establish the premise on its *own*
  full-weight target modules — LoRA B=0 and separate A/B spectra are gauge-
  dependent and do not equal the full weight spectrum.

## Rule

**A fixed-spectrum optimizer needs a positive control that actually moves the
spectrum before you trust the "it barely moves" arm.** If dense supervision and
RLVR produce the same tiny drift, the tiny drift is the parameter set's property,
not the optimizer's opportunity — freezing Σ buys nothing because nothing was
spending on Σ. Check the separation, not just the RLVR floor. The paper's
premise is regime-specific (full weight matrices); a low-rank head is not that
regime.
