# OPD Capability A/B Matrix

Date: 2026-06-18
Scope: local prep only. Do not run on H20/tmux2 while DSv4 owns the box.

## Objective

Prepare the OPD capability A/B surface for the 4B bring-up lane:

- Teacher: `Qwen3.6-35B-A3B-FP8`
- Student: `Qwen3.5-4B`
- Dataset for capability verdict: MATH-500
- Train prompt lane: existing MATH R1 question-only launcher shape
- Loss memory policy: fused/windowed loss path enabled by `--logits-window-size 32`
- Eval policy: greedy generation, `max_tokens >= 4096`, full `n=500`.
  Deterministic MATH-500 uses one eval per checkpoint with Wilson 95% CI.
  Variance comes from at least 3 OPD training seeds.

No training or evaluation was run while preparing this matrix.

## Prepared Launchers

All training launchers reuse `examples/opd/run-math-r1-35b-to-4b.sh`. The shared
runner now accepts optional `KL_BETA`; unset preserves the old behavior.

| Cell | Launcher | Single variable vs baseline | Required flags |
|---|---|---:|---|
| baseline | `examples/opd/run-math-r1-35b-to-4b-ab-baseline.sh` | none | `--kl-direction forward`, greedy rollout |
| arm2-control | `examples/opd/run-math-r1-35b-to-4b-ab-dense-forward-kl.sh` | fused loss off | `--no-fused-distill`, `--kl-direction forward`, greedy rollout |
| arm1 | `examples/opd/run-math-r1-35b-to-4b-ab-reverse-kl.sh` | KL direction | `--kl-direction reverse`, greedy rollout |
| arm2a | `KL_BETA=0.3 examples/opd/run-math-r1-35b-to-4b-ab-beta-jsd.sh` | beta-JSD beta | `--kl-beta 0.3`, greedy rollout; Δ vs dense control |
| arm2b | `KL_BETA=0.5 examples/opd/run-math-r1-35b-to-4b-ab-beta-jsd.sh` | beta-JSD beta | `--kl-beta 0.5`, greedy rollout; Δ vs dense control |
| arm3 | `examples/opd/run-math-r1-35b-to-4b-ab-stochastic-rollout.sh` | rollout sampling | `--rollout-temperature 0.9`, `--rollout-top-k 0`, `--rollout-top-p 1.0` |

Shared controlled variables:

- `TEACHER_MODEL=/data01/models/Qwen3.6-35B-A3B-FP8`
- `STUDENT_MODEL=/data01/modelscope-cache/Qwen/Qwen3___5-4B`
- `PROMPTS_FILE=/data01/arle-opd-runs/math-train-question-only.jsonl`
- `GKD_LAMBDA=0.0`
- `LOGITS_WINDOW_SIZE=32`
- `KL_MASK=completion`
- `KL_TEMPERATURE=1.0`
- `ROLLOUT_LEN=2048`
- `PROMPT_MAX_TOKENS=2048`
- `STEPS=250`, `SAVE_EVERY=50`
- `LR=2e-5`, `LR_SCHEDULE=cosine`, `LR_WARMUP_STEPS=8`, `GRAD_CLIP=1.0`
- `LORA_TARGET_SET=all-linear`, `LORA_RANK=32`, `LORA_ALPHA=64`
- `TRAIN_BACKEND=cuda`, `TEACHER_RUNTIME=infer`, `ENGINE_OFFLOAD=teacher`

## Fused-Loss Gate

All cells force the windowed path with `LOGITS_WINDOW_SIZE=32`.

Current caveat: beta-JSD is prepared but not yet fused-linear clean. The present
Route B branch sends `kl_beta.is_some()` through `student_logits_window_start`
and `kl_distill_loss_for_config`, not `fused_linear_distill_loss`
(`crates/train/src/opd.rs:2194`). Because this task explicitly does not touch
`fused_linear_distill.rs`, do not run arm2a/arm2b as comparable fused-loss cells
until the beta/JSD fused path is verified. Baseline, reverse-KL, and stochastic
forward-KL stay on the fused-linear KL branch when `KL_BETA` is unset. To
isolate beta itself, arm2a/arm2b report Δ against `dense-forward-kl-control`,
not against the fused baseline.

Run-time check for non-beta cells:

```bash
grep -q "fused_linear_distill_start" "$RUN_ROOT/logs/train.log"
```

For beta cells, the gate is stricter: the train log must not show
`student_logits_window_start kl_beta=true` as the loss path for the measured run.

## Eval Harness

Prepared launcher:

```bash
examples/opd/eval-math500-35b-to-4b-ab-curve.sh
```

It reuses:

- `scripts/opd_capability_curve.py` for checkpoint fan-out and baseline deltas
- `scripts/arle_capability_eval.py` for MATH-500 greedy scoring
- `scripts/analyze_multi_seed.py` for Wilson CI, mean, sample sigma, and paired deltas

Default eval contract:

```bash
TASKS=math500
N_SAMPLES=500
TRAINING_SEEDS=0,1,2
MATH_MAX_TOKENS=4096
REQUEST_TIMEOUT=1800
DRY_RUN=1
```

To run later, set one checkpoint path per arm per OPD training seed, then flip
`DRY_RUN=0`:

```bash
BASELINE_MODEL_PATH_S0=/path/to/baseline-s0 \
DENSE_FORWARD_KL_MODEL_PATH_S0=/path/to/dense-control-s0 \
REVERSE_KL_MODEL_PATH_S0=/path/to/reverse-s0 \
BETA_JSD_03_MODEL_PATH_S0=/path/to/beta03-s0 \
BETA_JSD_05_MODEL_PATH_S0=/path/to/beta05-s0 \
STOCHASTIC_MODEL_PATH_S0=/path/to/stochastic-s0 \
# repeat the same variables for _S1 and _S2
DRY_RUN=0 \
examples/opd/eval-math500-35b-to-4b-ab-curve.sh
```

Expected outputs:

- `curve.json`: per checkpoint metric and delta table
- `*/capability/math500.json`: deterministic 500-question raw counts and
  Wilson 95% CI
- `*/capability/math500_perquestion.json`: per-question records
- curve stdout: across-training-seed mean, sample σ, and paired Δ mean/σ by arm

## Run Order

1. Build one release binary from the exact commit to be tested.
2. For each training seed, train baseline first.
3. Train dense-forward-KL control for the same seed.
4. Train and gate arm1 and arm3 for the same seed.
5. Only after fused beta/JSD loss is verified, train arm2a and arm2b.
6. For every checkpoint, pass the needle gate before capability scoring.
7. Run one deterministic MATH-500 eval per checkpoint.
8. Report arm1/arm3 Δ vs fused baseline; report arm2a/arm2b Δ vs dense control.

Reason for ordering: same training seed gives paired OPD runs; full MATH-500
greedy eval is deterministic, so repeated eval seeds only change ordering and
do not estimate model variance.

## Correctness Gate Before Capability

Do not score capability before the checkpoint passes a greedy correctness gate.
For each checkpoint:

```bash
GATE_PROFILE=generic \
MODEL=/path/to/checkpoint \
LENGTHS=115,300,446,2000,8000 \
RUNS=3 \
scripts/lever_gate.sh <label>
```

Pass criteria:

- No serve crash.
- Needle exact/partial/miss distribution is within the baseline envelope.
- No new degenerate loop or garbage-class output.
- Same-config repeats establish the non-determinism floor before comparing arms.

If the correctness gate fails, mark that arm KILL for capability and do not run
MATH-500.

## Capability Acceptance

For every arm, report against `baseline-forward-kl-greedy`:

- Per-checkpoint MATH-500 accuracy with Wilson 95% CI over 500 binomial trials.
- Across-training-seed `mean±sigma`.
- Paired training-seed delta in percentage points.
- Invalid/extractor-fail rate.

Control pairing:

- `reverse-kl` and `stochastic-rollout`: Δ vs `baseline`.
- `beta-jsd-0.3` and `beta-jsd-0.5`: Δ vs `dense-forward-kl-control`.
- `dense-forward-kl-control`: diagnostic Δ vs `baseline`, not a recipe arm.

Verdicts:

- PASS: needle gate passes, invalid rate is not worse than baseline by more than
  1 pp, and paired training-seed Δ mean is positive with no seed showing a large
  regression.
- KILL: needle gate fails, or paired training-seed Δ is non-positive in all seeds.
- DEFER: needle gate passes but the ≥3 training-seed sample is mixed/noisy.

No default recipe flip is licensed by this matrix alone. A default flip still
needs the broader H20 multi-shape A/B called out in the recipe-gap research.
