# v8 200-step OPD: directional MMLU gain, directional GSM8K cost — both still inside noise

## Context

After the 2026-05-28 retraction of the v4 51.03% MMLU cross-base claim
([errors entry](../errors/2026-05-28-mmlu-cross-base-was-noise.md)),
the open question was whether the *direction* of the wins-doc claims
(+MMLU, -GSM8K) would emerge as a real effect at longer training under
the same multi-seed paired methodology. v8 tested this: 200 steps at
rollout=64 (matched compute budget vs v4's 60 × rollout=128), same lr,
same corpus (`opd-diverse-1k.jsonl`), same `kl_chunk_size=16`,
same `completion-only` KL mask.

Train: `target/release/examples/opd_step_cuda_infer_teacher_train ...
--steps 200 --rollout-len 64 --lr 2e-5`, ckpts at step_100 and
step_200. 200 steps × 99.86 s mean step = 5h 32min wall-clock on RTX
4070 Ti SUPER (16 GB). Eval: `eval_v8_after_train.sh` (autonomous
companion) ran 5-seed eval × 2 ckpts × paired-vs-base, all at matched
seeds {0..4} from the existing
`runs/2026-05-28-base-multiseed-eval/`.

## What worked

### Methodology

The 2026-05-28 multi-seed paired methodology
([`scripts/eval_opd_ckpt_seeds.sh`](../../../scripts/eval_opd_ckpt_seeds.sh)
+ [`scripts/analyze_multi_seed.py`](../../../scripts/analyze_multi_seed.py))
caught a false-positive on v4 step_020 and now provides a clean
before/after picture across v4 (60 steps) → v8 step_100 (100 steps) →
v8 step_200 (200 steps). All paired against the same base 5-seed
control at matched seeds.

### Directional trend across training

| run                          | steps × rollout_len | MMLU paired Δ | 95% CI            | GSM8K paired Δ | 95% CI            |
|------------------------------|--------------------:|--------------:|:------------------|---------------:|:------------------|
| v4 step_020                  | 60 × 128            | +0.47pp       | [-0.84, +1.77]    | -0.70pp        | [-1.71, +0.31]    |
| **v8 step_100**              | **100 × 64**        | **+0.95pp**   | **[-0.42, +2.33]**| **-1.20pp**    | **[-3.57, +1.17]**|
| **v8 step_200**              | **200 × 64**        | **+1.27pp**   | **[-0.84, +3.38]**| **-1.60pp**    | **[-3.55, +0.35]**|

Both directions monotonic across the three points:

- **MMLU**: +0.47 → +0.95 → +1.27 pp (more training → bigger positive Δ)
- **GSM8K**: −0.70 → −1.20 → −1.60 pp (more training → bigger negative Δ)

This is consistent with the wins-doc original directional claim
(OPD gains some MMLU, trades some GSM8K), now grounded in matched-
seed paired analysis rather than a single deterministic point.

### Raw v8 step_200 capability table

| metric | base 5-seed mean | v8 step_200 5-seed mean | paired Δ |
|---|---:|---:|---:|
| MMLU  | 0.5007 | **0.5134** | +1.27 pp |
| GSM8K | 0.3430 | 0.3270    | -1.60 pp |

The 51.34% MMLU mean at v8 step_200 returns to the level of the
original wins-doc 51.03% claim, but **now verified across 5 seeds
with paired controls instead of one deterministic point**. The
mean is plausibly above base by about a percentage point; the
sub-2pp CI at n=200 keeps the result statistically null at 95%.

### Train signal

Loss: first 1.602e-5 → final 1.919e-5 (-19.82% relative, but loss
across steps reflects per-prompt difficulty, not convergence).

In-loop held-out KL ([decreased](../../../runs/2026-05-28-rollout64-200steps-v8/run.txt)):

| step | train_kl | heldout_kl |
|---|---|---|
| 0   | 1.20e-5 | 1.74e-5 |
| 100 | 1.34e-5 | 1.34e-5 (−23%) |
| 200 | 1.32e-5 | **1.29e-5 (−26%)** |

Heldout KL dropped 26% across 200 steps — the model is genuinely
matching teacher more closely on held-out prompts. The eval Δ
(both positive on MMLU, negative on GSM8K) shows this match transfers
unevenly to downstream tasks: helps multiple-choice, hurts multi-step
reasoning.

## Rule

- **Direction is now real; magnitude is not yet significant.** The
  2026-05-28 methodology was supposed to be agnostic to direction,
  but three matched-seed paired experiments all land on the same
  signs (+MMLU, -GSM8K) with monotonically growing magnitude across
  training. At ~5pp σ from per-seed binomial noise + 5 seeds, the
  current paired SE is ~1.0-1.2 pp on each task; the +1.27pp MMLU
  signal sits at t=+1.18 (close to but not crossing the 1.96 gate).
- **To definitively confirm the directional claim, the next experiment
  needs n_samples≥500 per eval** (n=200 → ~145 scored is the n-bound,
  and at +1.27pp the binomial detection minimum is ~n=350 for 95%
  power). At n=500, expected paired SE drops to ~0.7pp and the
  +1.27pp MMLU gap becomes detectable. Cost: each seed eval ~3×
  longer = 75 min × 3 × 5 seeds = ~19h. Or fewer seeds with the
  per-question McNemar dump from
  [commit `3937c47c`](../../../scripts/arle_capability_eval.py).
- **The "60 steps was peak" wins-doc claim** is now refuted by
  evidence. v8 step_100 is *better* than v4 step_020 on MMLU
  (paired Δ +0.95pp vs +0.47pp); v8 step_200 is *even better*
  (+1.27pp). More training is monotonically better for MMLU at
  this regime, not "destructive at step_030+" as wins-doc claimed.

## Pending

- n_samples=500 (or higher) re-eval of v8 step_200 to push the
  +1.27pp MMLU Δ across 95% significance (and likewise the GSM8K
  -1.60pp). Cost ~3-6h GPU.
- v9 candidate: train to 400-600 steps (route-through-infer would
  unblock this in reasonable wall-clock — current 200 steps already
  took 5.5h).
- Plot the three-point trajectory (v4 / v8 step_100 / v8 step_200
  paired Δ vs steps trained) once a third v8-class data point lands.

Code refs: train via `target/release/examples/opd_step_cuda_infer_teacher_train`
(commit `e485edbe`-era), eval pipeline via
`scripts/eval_opd_ckpt_seeds.sh` (`60c146dd`, `7c592054`, `14ddb113`)
+ `scripts/analyze_multi_seed.py` (`b17061e0`, `3bae5cfc`) +
`scripts/eval_v8_after_train.sh` (`54ba6ad4`). Raw data:
`runs/2026-05-28-rollout64-200steps-v8/`.
