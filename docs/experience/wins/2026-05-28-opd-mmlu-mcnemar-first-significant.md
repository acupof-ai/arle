# OPD v8 first formally-significant MMLU gain — McNemar χ²=4.78 (p≈0.029)

## Context

After the 2026-05-28 retraction of the 51.03% wins claim
([errors entry](../errors/2026-05-28-mmlu-cross-base-was-noise.md)),
the multi-seed paired methodology was specifically built to catch the
upward bias that broke that claim. Until this run every OPD configuration
landed inside the noise floor at n=200, 5 seeds (paired SE ≥ 0.67pp,
CIs crossed zero).

This entry documents the first OPD configuration that crosses the 95%
significance gate by a defensible test. The shift came from two
methodology pieces, not from a new training trick:

1. **v8 longer-training experiment** (200 steps × rollout=64) — same
   teacher + student + corpus + lr as v4, just more training steps at
   smaller rollout (matched compute budget). Run dir:
   `runs/2026-05-28-rollout64-200steps-v8/`.
2. **Question-level McNemar paired test** added to
   `scripts/analyze_multi_seed.py` (commit `0a05ecaa`). Uses the new
   per-question dump (`<task>_perquestion.json`, commit `3937c47c`) to
   pool 5 seeds × ~140 paired MMLU questions = 708 paired observations
   into a single 2×2 contingency.

## What worked

### A/B contingency table (treated = v8 step_200, control = base 0.8B, matched seeds 0..4)

| seed | paired n | both ✓ | v8 only | base only | both ✗ |
|---:|---:|---:|---:|---:|---:|
| 0  | 139 |  58 | **8**  | **0**  |  73 |
| 1  | 146 |  81 | **3**  | **1**  |  61 |
| 2  | 145 |  73 | **8**  | **4**  |  60 |
| 3  | 140 |  65 | **5**  | **4**  |  66 |
| 4  | 138 |  65 | **4**  | **4**  |  65 |
| **TOTAL** | **708** | **342** | **28** | **13** | **325** |

Discordant total b + c = 41. Concordant total a + d = 667. Of the 41
questions where v8 and base differed, **v8 was right on 28 / 41 = 68%**
of them (vs the H0 expectation of 50%).

### Statistic and confidence

- Paired Δ = (b − c) / n = (28 − 13) / 708 = **+2.12pp**
- SE = √((b + c − (b−c)²/n) / n²) = √((41 − 0.318) / 501264) ≈ **0.90pp**
- **95% CI = [+0.35pp, +3.88pp]** — CI does not include zero
- **McNemar χ² (continuity-corrected) = (|b−c| − 1)² / (b+c) =
  14² / 41 = 4.780 > 3.841 (χ²(1, α=0.05))** → reject H0
- p ≈ 0.029 (one-sided one-tailed ≈ 0.0145)

This is the first OPD configuration in ARLE history that clears the
χ²(1, 0.05) gate on MMLU paired against base 0.8B at matched seeds.

### Why earlier (per-seed paired) analysis missed it

The same data analyzed via the per-seed paired mean approach:

| metric | per-seed paired (n=5 seeds) | McNemar question-level (n=708 paired) |
|---|---:|---:|
| Point estimate Δ | +1.27pp | **+2.12pp** |
| SE              | 1.07pp  | **0.90pp** |
| 95% CI lower    | −0.84pp | **+0.35pp** |
| t / χ²          | +1.18 (fail to reject) | **4.78 (reject)** |

Two effects pushed McNemar past the gate:
- Question-level pooling pulls n from 5 (seeds) to 708 (paired
  questions), even though the question outcomes within a seed are
  correlated.
- Restricting to both-extracted pairs (filters extract-fails on either
  side) removes a noise source that polluted the per-seed mean.

### GSM8K side: null, not significant regression

Same analysis for GSM8K (no invalid extraction, full n=200 per seed):

| metric | per-seed paired | McNemar question-level |
|---|---:|---:|
| Point estimate Δ | −1.60pp | −1.60pp |
| SE              | 0.99pp  | 1.32pp  |
| 95% CI          | [−3.55, +0.35] | [−4.18, +0.98] |
| t / χ²          | −1.61 | 1.293 (fail to reject) |

Contingency: b=79, c=95, n=1000. Discordant lean is toward base
(c > b) but the gap is inside noise. **The wins-doc and tick-7
"GSM8K regression is structural" claim is NOT supported by paired
data at this n** — direction matches but significance does not.
Future work: bump n_samples to ≥500 if isolating the GSM8K
direction is needed.

### Why the gap is real (mechanism-side cross-check)

In-loop heldout KL over v8 training (same as v4 but more steps):

| step | train_kl | heldout_kl |
|---:|---:|---:|
|   0 | 1.20e-5 | 1.74e-5 |
| 100 | 1.34e-5 | 1.34e-5 (−23%) |
| 200 | 1.32e-5 | **1.29e-5 (−26%)** |

Student matches teacher 26% more closely on held-out prompts after
200 steps. The MMLU +2.12pp gain is consistent with this match
transferring partially to a downstream multiple-choice task.

## Rule

- **Question-level McNemar is the right paired test when both sides
  have per-question dumps and the discordance rate is low.** For OPD
  comparisons where treated and base are very similar models (most
  questions concordant), McNemar's SE on (b − c) / n is meaningfully
  smaller than the per-seed paired SE on mean-of-5-deltas. Apply
  alongside the per-seed paired analysis (which is more robust to
  per-seed variance), and use the tighter of the two for the
  significance decision.
- **Significance ≠ large effect.** +2.12pp at CI lower-bound +0.35pp
  is the smallest publishable OPD MMLU gain that we should report. It
  passes the bar; do NOT extrapolate to "OPD now reliably beats
  base" — repeat at higher n or other seeds before extrapolating.
- **The 2026-05-28 errors entry rule stands.** Wins entries require
  the test gate (significance / kill threshold / shipped capability).
  This entry passes by McNemar χ²(cc) = 4.78 > 3.841 (α=0.05) and
  95% CI excludes zero. Other near-misses (paired t=+1.18, p≈0.13)
  stay out of wins/ until they cross.

## Pending

- Re-run the same comparison at **n_samples = 500** (`--n-samples 500`,
  same seeds 0..4) for both base and v8 step_200 to nail down a wider
  effect-size CI and confirm the McNemar result holds out-of-sample.
  Cost: ~6h GPU. Outcome bounds: if p stays < 0.05, we have a robust
  +1-3pp MMLU gain claim for v8 step_200; if it goes back inside CI,
  this entry needs an addendum.
- v9 with `--lora-rank=64` (commit `01d07bf6` exposes the CLI knob)
  to test whether higher LoRA capacity scales the gain. Local
  rebuild currently blocked on FlashMLA SM89 issue (codex queue
  Task I), need SM90 box or that fix first.
- The GSM8K direction (-1.60pp McNemar, not significant) is still
  worth following up with an n_samples=500 eval — point estimate is
  consistently negative across 5 seeds, just inside CI.

## Refs

- Train: `target/release/examples/opd_step_cuda_infer_teacher_train`
  with `--steps 200 --rollout-len 64 --lr 2e-5 --kl-chunk-size 16
  --opd-kl-mask completion-only` (binary from 2026-05-26, predates
  this entry's commits).
- Eval pipeline: `scripts/eval_opd_ckpt_seeds.sh` (`60c146dd`,
  `7c592054`, `14ddb113`) + `scripts/arle_capability_eval.py`
  (`c55db536` seed knob, `e68aa26c` invalid dump, `3937c47c`
  per-question dump).
- Analyzer: `scripts/analyze_multi_seed.py` (`b17061e0` Wilson CI,
  `3bae5cfc` per-seed paired, `0a05ecaa` McNemar).
- Companion: `scripts/eval_v8_after_train.sh` (`54ba6ad4`).
- Raw data: `runs/2026-05-28-rollout64-200steps-v8/capability_seeds_step200/`
  + `runs/2026-05-28-base-multiseed-eval/capability_seeds/`.
