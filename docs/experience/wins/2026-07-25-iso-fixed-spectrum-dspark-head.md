# ISO fixed-spectrum optimization on the DSpark acceptance head

> Status: implemented + local invariant gates green, **effect unmeasured** — no
> acceptance-rate A/B has run. Default off (`--dspark-train-iso`).

## Context

ISO (*An RLVR-Native Optimization Stack*, arXiv:2607.19331) reports that an RLVR
update barely moves a weight's singular spectrum — κ_spec 1.02–1.35, i.e.
indistinguishable from isotropic noise — while dense SFT supervision moves it
hard, κ_spec 89–1364. The behaviour change is a rotation of the singular
*frames*, so an optimizer inherited from pretraining spends updates on the one
thing RLVR does not need. Constraining the iterate to the fixed-spectrum family
ℱ(W₀) = {U Σ₀ Vᵀ} bought the paper ~2.2× fewer steps to equal accuracy.

**Placement was the whole decision, and it is not the obvious one.** The
tempting target is the OPD trainer — it is the repo's main training path. That
would be wrong: OPD is dense per-token distillation, structurally the SFT side
of the paper's own control condition, the regime where the spectrum moves by two
orders of magnitude. The DSpark acceptance head is the one RLVR-shaped trainable
we have: sparse outcome-level verifiable reward (`accepted / block_size`), seeded
from a base checkpoint, small proximal steps. So ISO went there and nowhere else.

## What Worked

The retraction, for a factor `W [n, k]`:

```text
S = Wᵀ W        M = S^{-1/2} S₀^{1/2}        W ← W M
```

`W M = polar(W) · S₀^{1/2}`, and an isometry cannot change singular values, so
`σ(W M) = σ(W₀)` exactly — ISO's step-4 projection with no eigensolver and no new
dependency. Matrix roots come from coupled Newton–Schulz on the `k × k` Gram in
f64 (the paper likewise uses FP64 for its polar step); the only large work is two
`n·k²` GEMMs through the store's own tuned kernel.

- **Composes with any base optimizer**, as ISO intends: AdamW steps, then the
  projection. `FixedSpectrum` knows nothing about AdamW.
- **Cadence came from a measurement, not the paper.** At the real head shape
  (151936 × 256, both factors, M4 Pro single-thread) one retraction costs
  **5.0 s**, against a ~1 s train step. Per-step retraction would therefore cost
  more wall-clock than a 2.2× step reduction buys back — the paper's 7% figure
  does not transfer to a CPU sidecar. Retraction runs on the `swap_every`
  cadence instead: exact whenever it runs, and measured inter-retraction drift is
  ~1e-3 relative, so the iterate stays in a tight neighbourhood of ℱ(W₀).
  Sharing the publish cadence also makes "the head handed to the engine is on the
  manifold" true by construction rather than by luck; the off-cadence final
  publish retracts first.
- **The premise is logged, not assumed.** Each step prints `iso_drift` =
  `‖W − W M‖_F / ‖W‖_F`. A run *without* `--dspark-train-iso` therefore measures
  whether this head's updates were already frame rotations — the paper's Day-1
  experiment, on our head, for free.
- Gates: `retraction_restores_the_base_spectrum` (unit) and
  `iso_fixed_spectrum_pins_the_spectrum_without_freezing_the_head` (end-to-end
  through the real trainer). Both compare Gram spectra — same statement as
  `‖σ(W) − σ(W₀)‖` without an eigensolver. The second asserts three things: ISO
  pins the spectrum, the unconstrained arm actually moves it (else the test
  proves nothing), and the head still moves (a frozen head would pass a spectrum
  check while learning nothing).

## Rule

- **Check which regime a paper's observation was made in before porting it.**
  ISO's fixed spectrum is an RLVR finding whose own control condition is dense
  supervision. Wiring it into the dense-supervision path would have been a
  faithful implementation of the method in the exact place the paper says it
  fails.
- **A paper's overhead figure is a measurement of its setup, not of yours.** 7%
  on their GPU became 500% on our CPU sidecar. Time the step at the real shape
  before adopting the paper's cadence.
- **Log the premise a method rests on.** The constraint is only free if updates
  were already frame rotations; one printed number per step turns that from an
  assumption into a result.

## The placement argument is only half true

The loop is self-RL — experiences are the serve's own drafts, the reward is its
own acceptance, no external data anywhere. But the *objective* is a 50/50 hybrid:
`prob_match_alpha` defaults to 0.5, so half the gradient is dense
probability-matching against the trunk's full target distribution. That half is
self-distillation, i.e. **dense per-token supervision — exactly the regime the
paper's own control condition says moves the spectrum by 100×.** Only the policy-
gradient half is the RLVR regime ISO's observation was made in.

So the head is not cleanly RLVR-shaped, and this sharpens the experiment into a
falsifiable prediction rather than a hope: **`iso_drift` should scale with
`prob_match_alpha`.** At `prob_match_alpha = 0` (pure acceptance PG) drift should
sit near the isotropic floor and the fixed-spectrum constraint should be nearly
free; at 1.0 (pure self-distillation) drift should be large and ISO should hurt.
The step line prints both numbers together so a log is self-describing.

If drift turns out large at the 0.5 default, the useful conclusion is not "ISO
fails here" — it is that the objective is half dense supervision, and the
fixed-spectrum constraint and the probability-matching term are pulling against
each other.

## Open

The effect on acceptance rate is unmeasured. The run that decides it is a
`prob_match_alpha` sweep (0 / 0.5 / 1.0) with `--dspark-train-iso` off, reading
`iso_drift` — that measures the premise before spending anything on the
constraint. Only then is the on/off A/B worth running, at whichever alpha the
drift says is in ISO's regime.
