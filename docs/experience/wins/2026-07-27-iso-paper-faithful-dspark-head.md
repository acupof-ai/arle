# Paper-faithful ISO on the DSpark acceptance head, 2026-07-27

> Status: correctness complete; local invariant + chain-rule + resume gates
> green; **acceptance effect unmeasured** — H20 A/B pending-remote. Default off
> (`--dspark-train-iso`).
>
> Supersedes the 2026-07-25 projection prototype (AdamW-on-dense-`W` + periodic
> `project(W)`). That was *a* fixed-spectrum retraction, not the paper optimizer;
> this entry replaces it. Do not call the old form ISO-Optimizer.

## Context

ISO (*An RLVR-Native Optimization Stack*, arXiv:2607.19331) reports that an RLVR
update barely moves a weight's singular spectrum — κ_spec 1.02–1.35,
indistinguishable from isotropic noise — while dense SFT supervision moves it
hard, κ_spec 89–1364. The behaviour change is a rotation of the singular
*frames*, so an optimizer inherited from pretraining spends Adam moments on the
one thing RLVR does not need. ISO removes Σ from the coordinates: factor
`W₀ = U₀ diag(Σ₀) V₀ᵀ`, freeze Σ₀, optimize the orthonormal frames `U,V`. That
bought the paper ~2.2× fewer steps to equal accuracy.

**Placement is the whole decision and not the obvious one.** OPD is dense
per-token distillation — the SFT side of the paper's own control condition, where
the spectrum moves by 100×. The DSpark acceptance head is the one RLVR-shaped
trainable we have (sparse outcome reward `accepted / block_size`, seeded from a
base checkpoint, small proximal steps). ISO went there and nowhere else.

## What changed (prototype → paper-faithful)

The prototype kept Adam moments on dense `W` and snapped `W ← W·S^{-1/2}·S₀^{1/2}`
periodically. Between projections the optimizer still accumulated moment mass on
spectrum-changing directions — the exact waste ISO diagnoses. Paper-faithful ISO
never gives the optimizer those coordinates:

- **Capture** (once, from the seeded head): thin SVD via Jacobi eig of the
  `[k,k]` Gram → `U [n,k]`, `V [k,k]` trainable leaves, `Σ₀ [k]` frozen. A cold
  head (zero Gram trace) is rejected — ISO adapts a head, it cannot grow one.
- **Reconstruct** (per forward): `W = U diag(Σ₀) Vᵀ` on the autograd tape, so a
  backward through the head yields `G_U = C·V·diag(Σ₀)`, `G_V = Cᵀ·U·diag(Σ₀)`
  by the chain rule — **no manual gradient** (unit-verified against the analytic
  form). AdamW keys state by `TensorId`, so the frames get independent moment
  state for free — no optimizer change.
- **Retract** (per cadence): `U ← polar(U)`, `V ← polar(V)` via Newton–Schulz on
  the `[k,k]` Gram in f64. With `UᵀU = VᵀV = I`, `σ(U diag(Σ₀) Vᵀ) = Σ₀` exactly.

Scope stayed at 4 files (`iso_spectrum.rs`, `dspark_train.rs`, its tests, this
entry): the serve/reload contract is untouched because `save_weights` persists
only the *materialized* `w1,w2`, and retract-before-publish means σ(W_saved)=Σ₀ —
so a resumed trainer re-captures the same Σ₀ from the reloaded head, with no Σ₀
file, no optimizer-state serialization, no `optim.rs`/`checkpoint.rs` change.

## Results

Local:

```text
iso_spectrum::tests: 4 passed
  - capture reconstructs W₀ and UᵀU = VᵀV = I
  - retraction restores the σ(W₀) MULTISET after a frame kick
  - backward matches the analytic G_U = C·V·diag(σ), G_V = Cᵀ·U·diag(σ)
  - cold (zero-spectrum) factor rejected
test_dspark_train: 7 passed (incl. ISO pins spectrum / rotates frames;
                              resume recovers the same Σ₀)
train lib: 179 passed ; clippy clean ; CUDA/no-CUDA typecheck: passed
```

The invariant is the singular-value **multiset**, not the Gram matrix: ISO
rotates the frames freely, so the Gram moves and only σ² is fixed. The old
prototype's full-Gram check was a stronger-but-wrong-for-ISO statement; the gates
now compare sorted Gram eigenvalues.

## Cadence (measurement, not the paper)

At the real head shape (151936 × 256, both frames, M4 Pro single-thread) one
retraction costs ~5.0 s against a ~1 s train step. Per-step retraction would cost
more wall-clock than a 2.2× step reduction buys back — the paper's 7% figure is a
GPU measurement, not a CPU-sidecar one. Retraction runs on the `swap_every`
publish cadence: exact whenever it runs, ~1e-3 inter-retraction drift, and the
head handed to the engine is on ℱ(W₀) by construction (off-cadence final publish
retracts first).

## The placement argument is only half true

The objective is a 50/50 hybrid: `prob_match_alpha` defaults to 0.5, so half the
gradient is dense probability-matching against the trunk — self-distillation,
the regime the paper says moves the spectrum 100×. Only the PG half is RLVR. So
the premise is a falsifiable prediction, logged not assumed: **`iso_drift` should
scale with `prob_match_alpha`** — near the isotropic floor at 0 (constraint ~free),
large at 1.0 (constraint hurts). Each step prints drift so a log is
self-describing.

## H20 license (pending-remote)

Unchanged from the plan's Phase-5 gate: (1) ISO-off drift sweep at
`prob_match_alpha = 0 / 0.5 / 1`; (2) proceed only if the pure-PG arm shows the
near-isospectral premise; (3) matched ISO-AdamW vs AdamW on identical
experience/seed/head; (4) primary metric is steps and wall-clock to equal
speculative acceptance, not loss; (5) record retraction wall, step wall, spectrum
error, accepted tokens/target-step. A failed premise or A/B keeps ISO non-default
and records an `errors/` verdict — it does not propagate into Agent-RFT.

### H20 sweep run (2026-07-27) — INCONCLUSIVE, do not green-light the A/B

Qwen3.6-27B-FP8 single-GPU, cold head, ISO-off, α ∈ {0, 0.5, 1}, w1 axis:

| α | w1 spectrum_drift | train loss | verdict |
|---|---|---|---|
| 0 (pure PG) | 2.07e-6 | ≈−0.4 (active) | near-isospectral, but see confound 2 |
| 0.5 | 1.7–2.6e-6 | ≈−0.07 | PM half contributes ~0 |
| 1 (pure PM) | 4e-10 | **0.0000** | null arm — no gradient |

Two confounds make this neither a confirm nor a clean falsification:

1. **α=1 is a null discriminator.** The prob-match loss
   `mean_{v∈vocab}(softmax(draft)−softmax(target))²` divides a TV-style surrogate
   by 248,320 classes; on a head where the draft already tracks the trunk it
   underflows f32 print precision → loss 0.0000, `accept_ema` drifts *down* (no
   learning). So α=1's tiny drift is "head didn't move," not "dense supervision
   preserves the spectrum" — the arm meant to show LARGE drift produced no signal.
2. **Cold head confounds the α=0 arm too.** `bias = w2·w1[cond]`, so
   `∂bias/∂w1 = w2 ≈ 0` at a cold (w2=0) start — w1's gradient is gated by the
   near-zero w2, so 2.07e-6 is partly "w1 barely trained," not cleanly "w1
   trained hard yet stayed isospectral." The plan specified a **nonzero
   pretrained** head for exactly this reason; using a cold head to dodge the
   missing seeded-Qwen head was too clever.

To measure the premise cleanly: a **nonzero pretrained Qwen3.6 DSpark head**
(both factors non-zero, so each gets real gradient) + a PM regime that actually
moves the head (higher `--dspark-train-lr`, or harder traffic with real
draft↔target divergence, or per-active-class PM normalization instead of
`/vocab` — see the 7b finding). Until then the ISO premise is unmeasured; #32
stays gated.

## Rule

- **A "fixed-spectrum retraction" is not the ISO optimizer.** Preserving σ(W₀)
  after each step (the prototype) still lets Adam chase the spectrum between
  projections. Paper ISO puts the moments on the frames so there are no spectrum
  coordinates to chase — that is the mechanism behind the 2.2×, and the
  distinction is the whole point of the rewrite.
- **Reconstruct on the tape instead of hand-coding the chain rule.** `W = U Σ₀ Vᵀ`
  as ops gives `G_U/G_V` for free and unit-checkable against the closed form.
- **Check which regime a paper's observation was made in, and log the premise.**
  One printed `iso_drift` per step turns "updates are frame rotations" from an
  assumption into a per-run result.
