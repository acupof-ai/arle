# OPD 51.03% MMLU cross-base was inside noise — methodological retraction

## Context

`docs/experience/wins/2026-05-27-opd-rollout128-sota-mmlu-cross-base.md`
claimed v4 step_020 = MMLU **51.03%** (+1.37pp over re-measured base
0.8B at 49.66%) as "first OPD cross-base." The wins entry also called
out a GSM8K **regression** from base 32.5% to v3/v4/v5/v6's 28-30%
as "structural at this scale ... distillation on next-token KL trades
reasoning chain quality for token distribution match."

2026-05-28 multi-seed verification at matched n=200 (5 seeds, same
ckpt, same harness) replaces both claims with a null effect:

| task | det. baseline | 5-seed mean | 5-seed σ | min | max |
|---|---|---|---|---|---|
| MMLU  | 0.5103 | **0.5053** | 0.0509 | 0.4397 | 0.5772 |
| GSM8K | 0.3000 | **0.3360** | 0.0370 | 0.2950 | 0.3800 |

The MMLU σ overshoots the kill threshold (1.5pp from
`docs/research/2026-05-28-opd-effect-axis-next.md` Gap 1) by 3.4×.
The GSM8K mean **direction-reverses** the wins claim — multi-seed mean
(33.6%) is *higher* than base 32.5%, not lower.

Raw seeds (commit `7c592054` driver):
`runs/2026-05-26-rollout128-v4-diverse1k-train-60/capability_seeds/seed_{0..4}/`.

## Cross-run confirmation (added 2026-05-28 tick 5)

Re-computing per-run mean ± σ across all 7 saved ckpts (deterministic
n=200 eval, *same* 145 MMLU + 200 GSM8K questions) for every v2-v6 run:

| run | MMLU mean | MMLU σ | GSM8K mean | GSM8K σ |
|---|---|---|---|---|
| v2 (sanitizer band-aid only)    | 0.4777 | 0.0051 | 0.0108 | 0.0045 |
| v3 (NaN real fix + KL mask)     | 0.4904 | 0.0119 | 0.2907 | 0.0217 |
| v4 (+1k diverse corpus) ⭐      | 0.4962 | 0.0106 | 0.2957 | 0.0127 |
| v5 (lr 1e-5)                    | 0.4937 | 0.0088 | 0.3014 | 0.0075 |
| v6 (lr 4e-5)                    | 0.5000 | 0.0056 | 0.2850 | 0.0135 |
| base 0.8B                       | 0.4966 |   —    | 0.3250 |   —    |

Two patterns the wins doc missed:

1. **Every post-NaN-fix run has MMLU mean ≈ base** (within ±1pp of
   49.66). The MMLU "best ckpt" peaks the wins doc highlighted —
   v3 step_030=50.68, v4 step_020=51.03, v5 step_020=50.68, v6
   step_010=50.68 — are *all* 1.2-1.5σ upper-tail draws from each
   run's own ckpt distribution (σ_run ≈ 0.5-1.2pp across ckpts).
   The "lr sweep U-curve around 2e-5" pattern was max-across-ckpts
   selection bias on a flat true distribution.
2. **GSM8K regression is real on the deterministic subset but not
   universal.** Every run lands ~2.5-4pp below base 32.5% on the
   deterministic 200 questions (v3=29.07, v4=29.57, v5=30.14,
   v6=28.50). But v4 step_020 multi-seed across seeds 0-4 gives
   GSM8K mean=33.60% > base 32.50%. **The regression is
   subset-specific** — the first-200 questions happen to be ones
   where the OPD-shifted reasoning style hurts. This invalidates
   the wins-doc "structural at this scale" generalization without
   ruling out the actual subset-level effect.

v2 stands out: the sanitizer band-aid only era genuinely broke the
model (GSM8K 1.08% — model produced gibberish, not low-quality
answers). The "532k → 0 NaN per step" mechanism-level wins from the
v3 fixes are real engineering progress; what they bought was *not
breaking the model*, not net capability gain.

## Root cause

1. **n was too small.** `scripts/arle_capability_eval.py --n-samples 200`
   produces ~145 scored MMLU questions after the invalid-extraction
   drop. Binomial 1σ at n=145, p≈0.5 ≈ √(0.25/145) ≈ 4.15pp. The wins
   doc's "±2-3% noise floor" line **underestimated** the true noise floor
   by ~30%.

2. **Single-eval interpretation.** The 51.03% point estimate was one
   draw from a 5pp-σ distribution. The 5-run lr sweep table in the
   wins entry (v3/v4/v5/v6 all landing 50.68–51.03% on best ckpt) was
   not 5 independent confirmations — it was 5 different ckpts on the
   *same* deterministic sample set, all visiting the same upper tail.
   Multi-seed at the v4 ckpt reproduces neither the absolute level nor
   the cluster.

3. **Asymmetric framing.** Picking the *best* ckpt per run (step_020 for
   v4 lr=2e-5) compounds the upward bias: across 6 ckpts per run (step
   10/20/30/40/50/60 + final) with σ=5pp each, the expected max is
   ~0.5–1σ above mean. So "v4 step_020 = 51.03%" is mean+~1σ+~1σ of
   ckpt-selection bias — a 2σ overestimate of the per-run capability.

4. **Missing variance gate.** No multi-seed run was done before the
   wins entry shipped. `feedback_3sample_too_noisy_for_10pct_effects.md`
   was already in memory; the rule didn't get applied to this
   1.37pp / ~2.7%-relative claim.

## Fix

- This errors entry stands as the post-mortem.
- `docs/experience/wins/2026-05-27-opd-rollout128-sota-mmlu-cross-base.md`
  amended with a top-banner pointing here. The train-side engineering
  wins in that entry (NaN root cause via linear-attention backward
  decay stabilization + state-division avoidance, chunked KL bwd, KV
  in-place append, mid-rollout retain pruning, completion-only KL mask)
  **remain valid**: those fixes were verified by sanitizer NaN count
  going from 532k → 0 per step, which is mechanism-level evidence, not
  capability-level.
- Pending: base 0.8B multi-seed at matched seeds 0..4 (running at
  `runs/2026-05-28-base-multiseed-eval/`), then a paired per-seed
  delta computation to nail down the actual OPD effect (or null) at
  this n. Errors-entry numerical update will follow.

## Rule

- **OPD effect claims with magnitude < 5pp on a small-n eval (n_samples ≤ 200,
  i.e. ≤ ~145 scored MMLU) MUST run multi-seed (≥5 seeds) and report
  mean ± σ + Wilson 95% CI BEFORE the wins entry ships.** The Verify
  exit condition in `CLAUDE.md §Execution phases` already requires a
  bench entry; for capability-axis claims it now also requires this
  variance proof.
- **"Cross-base" claims require explicit base multi-seed comparison.**
  Single-deterministic base = 49.66% is itself one σ-noisy draw.
- **Picking the best ckpt across a save-every-10 sweep is a
  conditional-on-best estimator with positive bias.** When reporting
  per-run capability, take the *mean across last-3 ckpts* or
  *multi-seed at one ckpt*, not max-across-ckpts at one seed.
- **The "GSM8K regression is structural" hypothesis was unsupported
  by data.** Any future reasoning-corpus work must license the
  hypothesis on a multi-seed delta first, not on a single-eval
  observation.
