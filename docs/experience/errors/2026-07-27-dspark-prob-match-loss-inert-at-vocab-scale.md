# DSpark probability-matching loss is numerically inert at vocab scale, 2026-07-27

## Context

The DSpark head objective is a hybrid: `loss = (1−α)·PG + α·prob_match`, `α =
prob_match_alpha` (default 0.5). The PG half is acceptance-weighted policy
gradient; the prob-match half is meant to be a dense self-distillation term
pulling the draft distribution toward the trunk's. The H20 ISO premise sweep
(Qwen3.6-27B-FP8, cold head, α ∈ {0, 0.5, 1}) exposed that the prob-match half
contributes ~0.

## Root cause

`dspark_train.rs:527-532`:

```rust
let prob_match_loss_id = ops::mul_scalar(
    ops::sum(weighted_sq_id, ...)?,          // Σ over rows × vocab of (Δsoftmax)²
    1.0 / (weight_sum * vocab_size as f32),  // ÷ (rows × 248320)
    ...
)?;
```

The loss is `mean_{row, v∈vocab}(softmax(draft)[v] − softmax(target)[v])²` — a
squared-difference surrogate averaged over **all 248,320 vocab classes**. On a
head where the draft already tracks the trunk (natural prompts keep the DFlash
draft close), the per-class squared diff is tiny and dividing by 248,320 pushes
the whole loss under f32 print precision. Measured across the α-sweep:

| α | PM contribution | loss |
|---|---|---|
| 0 (pure PG) | none | ≈ −0.4 (active) |
| 0.5 | ~half of α=0 → **PM half ≈ 0** | ≈ −0.07 |
| 1 (pure PM) | the whole loss | **0.0000, accept_ema drifts down (no learning)** |

So at the default `α = 0.5`, the "hybrid" objective is **effectively pure PG** in
production — the dense term is numerically inert. `α = 1` trains nothing.

## Impact

- The ISO premise sweep's α=1 discriminator is a null arm — it can't show the
  "dense supervision moves the spectrum hard" contrast the premise needs (see the
  ISO win entry's H20 section).
- Phase 7b ("a loss is licensed only if speculative acceptance improves"): this
  loss cannot improve acceptance because it produces no gradient at scale. The
  probability-matching term is disproven as currently normalized.

## Fix direction (not applied — needs its own acceptance gate)

The `/vocab` averaging is the defect: a distribution-matching loss should weight
the classes that carry mass, not dilute over the whole vocabulary. Options:
- Normalize per **active/top-k class** rather than `/vocab` (the tail is ~0 in
  both draft and target and only adds denominator).
- Use a proper divergence (KL(target‖draft) or cross-entropy on the trunk's
  top-k) instead of a `/vocab`-averaged squared diff.
- If the term stays MSE, at minimum drop the `1/vocab_size` factor (keep only
  `1/weight_sum`) so gradient magnitude survives.

Any of these changes the objective and must clear a matched acceptance A/B (7b),
so it is not a drive-by patch.

## Rule

**A per-class loss averaged over a 250k vocabulary is inert by construction —
check the gradient magnitude survives the normalizer before trusting the knob.**
`prob_match_alpha` looked like a live 50/50 mix; it was a dead term hidden behind
a `/vocab` that drove it under f32 precision. A config knob that silently
contributes nothing is the same failure class as a no-op CLI flag — measure the
term's actual gradient norm, don't assume the weight in the loss expression is
the weight in the update.
