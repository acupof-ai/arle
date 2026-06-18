# OPD infer rollout student pre-teacher offload

## Context

35B FP8 OPD smoke on `.62` reached train-student load and infer-rollout-student
load, then failed while loading the infer teacher:

- Command shape: Qwen3.6-35B-A3B-FP8, CUDA, `--teacher-runtime infer`,
  `--rollout-len 8`, LoRA rank 1, attention-qv.
- Observed failure: teacher load failed during FP8 block-scaled tensor upload
  after the train student and infer rollout student were already resident.
- Log: `/tmp/arle_opd35_smoke.log` on `iv-ye8is8fbi8s6iplibbg7`.

## What Worked

The CLI now honors `ARLE_OPD_ENGINE_OFFLOAD=student/all` before the initial
infer-teacher load: after loading the rollout infer student, it fences the train
backend and offloads the rollout engine weights to host RAM before constructing
the teacher engine. The steady-state step path already reloads the rollout
student before LoRA sync and decode.

Default behavior is unchanged when `ARLE_OPD_ENGINE_OFFLOAD` is unset or set to
`teacher`.

## Verification

Pending remote 35B smoke rerun after this code lands:

- Gate: Qwen3.6-35B-A3B-FP8 OPD, `ARLE_OPD_INFER_ROLLOUT=1`,
  `ARLE_OPD_ENGINE_OFFLOAD=student`, `--steps 1`, `--rollout-len 8`.
- Expected reachability signal: `infer_rollout_generate_start` followed by a
  completed OPD step, or a later measured blocker unrelated to the initial
  teacher-load residency failure.

## Rule

OPD infer-rollout validation on 35B must include the startup residency ordering,
not only the per-step offload/reload path. A time-share mode is incomplete if it
only frees memory after the first step.
