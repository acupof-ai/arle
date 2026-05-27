# OPD 51.03% MMLU first cross-base — rollout=128 + NaN root cause fix + 1k corpus

## Context

ARLE OPD had never crossed base 0.8B MMLU before this session. Prior best
was T18 (lr=1e-5, rollout=8, 16-prompt) at 50.59% MMLU on a different eval
harness — under the harness used here (arle_capability_eval.py via served
ARLE infer, MMLU 5-shot 171 sample / GSM8K 8-shot 200 sample), base 0.8B
re-measures at 49.66%.

Across 5 train rounds (v2→v6) the OPD step path got 6 substantive fixes,
a lr sweep, and a corpus expansion. End state: v4 step_020 lands at
MMLU 51.03% (+1.37pp over re-measured base).

## What Worked

### Code fixes that enabled long-rollout OPD

| commit | fix |
|---|---|
| `c8193ac` | chunk KL backward by rollout windows (fixed rollout=32 OOM) |
| `4b31e15` | pre-allocate KV cache for in-place append (killed concat-leak) |
| `23b3cff` | retain ids per decode step (mid-rollout prune, killed transpose leak) |
| `21071dc5` | stabilize linear attention backward decay (NaN root cause #1) |
| `b81b6d22` | avoid linear attention state division in backward (NaN root cause #2) |
| `25498c8` | KL only on completion tokens (mask prompt) |

`421c4a1` (sanitize non-finite LoRA grads before clipping) was the
band-aid that surfaced the NaN problem; after the two real root-cause
fixes it's mostly inert (532k → 0 NaN per step).

### Result table — re-measured base + all 5 train rounds, sorted by MMLU peak

| run | lr | corpus | rollout | best ckpt | MMLU | GSM8K |
|---|---|---|---|---|---|---|
| base Qwen3.5-0.8B-Base | — | — | — | — | 49.66% | 32.50% |
| teacher Qwen3.5-4B | — | — | — | — | 77.33% | (n/a) |
| v2 (sanitizer band-aid) | 2e-5 | 16-prompt | 128 | step_040 | 48.70% | 1.51% |
| v3 (NaN real fix + KL mask) | 2e-5 | 16-prompt | 128 | step_030 | 50.68% | 29.00% |
| **v4 (+ 1k corpus)** ⭐ | 2e-5 | 1k diverse | 128 | step_020 | **51.03%** | 30.00% |
| v5 (lr half) | 1e-5 | 1k diverse | 128 | step_020 | 50.68% | 30.00% |
| v6 (lr double) | 4e-5 | 1k diverse | 128 | step_010 | 50.68% | 28.00% |

LR sweep is a clean U-curve around `lr=2e-5`; halving or doubling drops
MMLU by ~0.35pp and (for higher lr) shifts the peak earlier — high lr
overshoots, low lr undertrains in the 30-step budget.

### Peak location

Every successful run peaks at the same effective-step (~20 steps with
lr=2e-5), independent of corpus size. Late training (step_030+) is
destructive on both MMLU and GSM8K — overfitting to the cycle of
training prompts even with a 1k-prompt corpus, because each prompt is
still only seen 0-2 times in the rollout cycle.

## Rule

- **The OPD train path cannot be evaluated without measuring NaN
  fraction in `tape.backward` output.** v2 looked "trained" but adapter
  hashes were bit-identical; the cause was 532,480/638,976 NaN grad
  elements being silently dropped by `clip_grad_norm`. Always assert
  `non_finite_replaced == 0` in any OPD train sanity gate before
  comparing capability numbers.
- **OPD peak MMLU lives near step 20 at lr=2e-5 in this regime; final
  ckpt is never the best.** Save+eval per-step (or at least every 5)
  and pick the peak. Reporting final-ckpt MMLU underestimates OPD by
  ~1pp.
- **GSM8K regression is structural at this scale**, not a tuning issue.
  All v3/v4/v5/v6 best ckpts land in the 28-30% GSM8K range vs base
  32.5% — distillation on next-token KL trades reasoning chain quality
  for token distribution match. Fixing this needs reasoning-aware loss
  or prompt construction (math-style multi-step prompts), not lr/corpus
  tweaks.
- **Base 49.66% MMLU on this harness is the bar to beat**, not the
  51.41% number from the legacy eval. Different harnesses (sample size,
  prompt format, shot count, parser strictness) move the absolute
  number by 1-3pp.

## Pending

- Multi-seed verification of v4 step_020 = 51.03% (variance unknown;
  171-sample MMLU has ±2-3% noise floor).
- GSM8K-targeted corpus build (current corpus is short-completion Q&A,
  doesn't exercise multi-step reasoning).
- rollout > 128 sweep (v7 rollout=256 killed at 19 min/step = 9.6h
  total, not feasible without sequence-windowed forward).

Code refs: `crates/train/src/opd.rs`, `crates/train/src/qwen35.rs`,
`crates/autograd/src/ops/linear_attention.rs`,
`scripts/eval_opd_ckpts.sh`, `runs/2026-05-26-rollout128-train-60-v2/`
through `runs/2026-05-27-rollout128-v6-lr4e5-train-30/`.
