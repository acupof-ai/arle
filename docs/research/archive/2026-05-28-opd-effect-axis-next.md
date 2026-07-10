# OPD effect axis — next steps after 51.03% MMLU first cross-base

**Date**: 2026-05-28 (Claude /loop tick 1)
**Predecessor**: [`docs/experience/wins/2026-05-27-opd-rollout128-sota-mmlu-cross-base.md`](../experience/wins/2026-05-27-opd-rollout128-sota-mmlu-cross-base.md)

## TL;DR

Two unrelated effect-axis gaps surface from the 51.03% wins entry. Both
need to be closed before we can claim further OPD effect progress.

1. **The 51.03% cross-base result is statistically unverified.** At n=171
   MMLU, the binomial 1σ ≈ 3.8pp. The 1.37pp gap over re-measured base
   sits well inside 1σ noise. The wins entry already lists "multi-seed
   verification" as pending; current eval harness has no `--seed` knob,
   so multi-seed is mechanically blocked.
2. **The GSM8K regression is corpus-distribution-structural.** All three
   existing OPD corpora (`sample-prompts.jsonl`, `opd-diverse-1k.jsonl`,
   `sft-anchor-mmlu-gsm8k.jsonl`) have either short factual answers
   (≤16 tok) or single-letter MCQ. None teaches multi-step
   chain-of-thought, which is exactly what GSM8K eval requires.
   Distilling on short completions necessarily degrades the reasoning
   skill the student inherited from base pretraining.

This doc scopes (1) and (2) with kill criteria. No code change yet —
that's the next step.

---

## Gap 1 — 51.03% is unverified at current eval n

### Evidence

`scripts/arle_capability_eval.py:269-276`:

```python
# Sample n_samples evenly across subjects for speed.
subjects = sorted({ex["subject"] for ex in ds_test})
n_per_subject = max(1, n_samples // len(subjects))
pool: list[dict] = []
for subj in subjects:
    pool.extend([ex for ex in ds_test if ex["subject"] == subj][:n_per_subject])
pool = pool[:n_samples]
```

- 57 MMLU subjects, `n_samples=200`, `n_per_subject = 200 // 57 = 3`.
- Pool size = 3 × 57 = 171 (matches wins entry).
- Selection is `[:n_per_subject]` — **deterministic order**, no shuffle,
  no `seed` parameter exposed.

GSM8K similarly deterministic: `ds_test.select(range(min(n_samples, len(ds_test))))`
at line 395.

### Noise floor math (SOLID)

Binomial standard error for a single accuracy estimate:

```
σ_acc = sqrt(p · (1 - p) / n)
```

At p ≈ 0.5, n = 171: σ_acc ≈ √(0.25 / 171) ≈ 0.0382 = **3.8 pp**.

The wins entry's "±2-3% noise floor" was an underestimate (closer to 4pp 1σ).

| run | MMLU | gap vs base 49.66% | gap in 1σ units |
|---|---|---:|---:|
| v4 step_020 (best ckpt) | 51.03% | +1.37pp | **0.36 σ** |
| v3 step_030 | 50.68% | +1.02pp | 0.27 σ |
| v5 step_020 | 50.68% | +1.02pp | 0.27 σ |

**Conclusion**: the v4 cross-base claim is statistically indistinguishable
from base at n=171. The cluster of three runs sitting 1pp above base
(v3 step_030, v4 step_020, v5 step_020) is consistent with a true OPD
effect — but no individual point passes a 1σ gate.

### License-or-kill — multi-seed eval patch

**Hypothesis**: Adding a `--seed` knob to `arle_capability_eval.py` that
shuffles MMLU pool before subject-balanced subsampling, then running v4
step_020 eval at seeds {0, 1, 2}, will give us a 3-point sample mean
+ sample σ. If mean ≥ 50.5% with σ ≤ 1.5pp, the cross-base claim
survives. If mean < 50.5% or σ > 2pp, the claim is noise.

**Implementation cost**: ~30 lines patch (add `--seed` flag, do
`random.Random(seed).shuffle(pool_per_subject)` before slicing).

**Run cost**: ~30-60 min × 3 seeds = 1.5-3h GPU at ~3-4 GB peak.
Single serve instance, safe on 16 GB SKU.

**Kill criterion** (explicit numerical threshold per
`feedback_license_or_kill_with_explicit_threshold.md`):

- Pass:  mean(MMLU@v4 step_020, seeds 0-2) ≥ 50.5%  AND  σ ≤ 1.5pp
- Kill:  mean < 50.5%  OR  σ > 2pp
- Action on kill: retract the "first cross-base" wins claim; investigate
  whether OPD is statistically distinguishable from base at this scale
  on any harness, before chasing further corpus/loss tweaks.

**Alternative considered — larger n**: bumping `--n-samples` from 200
to 1000 drops σ from 3.8pp to 1.7pp. Could combine: 3 seeds × n=500
gives 1500 effective samples, σ ≈ 1.3pp. But cost ≈ 6× current eval
time per ckpt. Multi-seed at original n is the cheaper variance check.

---

## Gap 2 — GSM8K regression is corpus-structural

### Evidence

Corpora inspection (2026-05-28):

| file | n | example completion | reasoning content |
|---|---|---|---|
| `examples/opd/sample-prompts.jsonl` | 20 | "They learn from teacher feedback..." (16 tok) | none |
| `examples/opd/opd-diverse-1k.jsonl` | 1000 | "oaks" (2 tok); "the woods" (2 tok) | none |
| `examples/opd/sft-anchor-mmlu-gsm8k.jsonl` | 56 | "B. Because ... is the correct choice." (32 tok) | MCQ letter only |

None of the three contains an explicit "Let's think step by step.
First, ... Then, ... So the answer is N." chain-of-thought.

GSM8K eval at `scripts/arle_capability_eval.py:355` uses an 8-shot
prompt with full numeric reasoning chains in each shot. The model is
asked to produce a multi-step solution and a `####` final answer.

OPD with current corpora distills the student toward the teacher's
short-completion logit distribution. The distillation signal never
exercises the multi-step generation path. Whatever reasoning skill the
base 0.8B retained from pretraining is being washed out by the strong
KL signal on short, factual completions.

This is the same structural pattern that hits GSM8K hard:
- base 0.8B: 32.5%
- v3 step_030 / v4 step_020 / v5 step_020: 28-30%
- v6 step_010 (high lr, fastest decay): 28% (steepest drop)

The decay rate correlates with effective OPD intensity, supporting the
corpus-distribution hypothesis (not lr-noise, not random-eval).

### License-or-kill — reasoning corpus path

**Hypothesis**: Building an OPD prompts file `opd-gsm8k-reasoning.jsonl`
of ~200 math word problems with multi-step `completion_max_tokens=128-256`
completions (sourced from GSM8K train split, with teacher rollouts as
ground truth), and adding it as a 20-30% mix into the v4 corpus,
will recover GSM8K toward base 32.5% without sacrificing MMLU.

**Implementation cost**:
- corpus build script: ~50 lines Python, calls teacher to generate
  completions for ~200 GSM8K-train prompts. One-time GPU cost ~10 min
  (200 prompts × 128 tok @ ~5 tok/s).
- mix into training: trivial CLI change (multi-file `--prompts-file`
  is the natural extension; currently single file).

**Run cost**: one v7-with-reasoning train at rollout=128, 60 steps =
matches v4 wall clock (~5 h GPU). Single train at a time per the
hardware budget rule. GPU mem peak ~11 GB at rollout=128 (already
verified on v4 path).

**Kill criterion**:
- Pass:  v7-reasoning step_020 GSM8K ≥ 32.0%  AND  MMLU ≥ 50.0%
- Kill:  GSM8K < 30.0%  OR  MMLU < 49.0%
- Action on kill: corpus mix is not the right knob — investigate
  reasoning-aware loss (loss weighting by completion length, or
  per-step KL only on reasoning-chain tokens) instead.

**Risk**: longer completions (256 vs 16 tok) blow up rollout=128 step
time. v4 step_020 ran at 305 s/step with mostly short completions; a
mixed corpus pushes mean completion length up, possibly to 60-80 tok
average. Worst case +3× step time → ~16 h for 60 steps. Need to
re-budget against the rollout=256 kill from the v7 attempt
(`runs/2026-05-26-rollout256-perffix-v3-*-dryrun` at 19 min/step).

---

## Sequencing

Both gaps are independent. Suggested order (lowest cost first):

1. **Multi-seed patch** (low GPU, mechanical code). If it kills 51.03%,
   we have a different problem (find a real cross-base lever); if it
   confirms, we have firm baseline for the GSM8K work.
2. **Reasoning corpus build** (low GPU, code-heavy — codex track).
   Can be done while seed eval is running.
3. **v7 train + multi-seed eval of v7 ckpts**. Compare against
   multi-seed v4 baseline.

Step 1 unblocks the SOLID interpretation of step 3. Doing step 3 first
without step 1 risks chasing 1pp differences inside a 4pp noise floor.

## Not in scope here

- Reasoning-aware loss (chain-of-thought weighted KL). License only if
  the corpus-mix hypothesis is killed.
- Rollout > 128 via sequence-windowed forward. Perf-axis investment;
  separate research thread.
- kv_tier multi-layer behavior. Codex track this cycle.
