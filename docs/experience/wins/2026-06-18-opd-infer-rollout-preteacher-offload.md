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

The first 35B FP8 rerun reached that new path and exposed the next blocker:
Qwen3.6 MoE infer-engine weights were dense-only in the OPD offload layer. The
MoE offload now snapshots and reloads the full `MoeLayerWeights` structure:
per-expert BF16/FP8 matrices, BF16 grouped caches, FP8 grouped caches, router
gate, shared expert weights, and rebuilt routed pointer/scale tables.

Default behavior is unchanged when `ARLE_OPD_ENGINE_OFFLOAD` is unset or set to
`teacher`.

## Verification

Local gates:

- `rustfmt --edition 2024 --check crates/cli/src/train_cli.rs`
- `git diff --check`
- `CUDARC_CUDA_VERSION=12090 cargo check -p cli --release --no-default-features --features cpu,no-cuda --lib`
- `CUDARC_CUDA_VERSION=12090 cargo check -p train --release --no-default-features --features cuda,no-cuda --lib`
- `CUDARC_CUDA_VERSION=12090 cargo check -p cli --release --no-default-features --features cuda,no-cuda --lib`
- `CUDARC_CUDA_VERSION=12090 cargo check -p infer-cuda --release --no-default-features --features cuda,no-cuda --lib`
- `CUDARC_CUDA_VERSION=12090 cargo clippy -p infer-cuda --release --no-default-features --features cuda,no-cuda --lib -- -D warnings`
- `CUDARC_CUDA_VERSION=12090 cargo clippy -p cli --release --no-default-features --features cpu,no-cuda --lib -- -D warnings`

Remote evidence:

- Pre-MoE-offload build `ab22f727` on `.62` reached
  `offload infer rollout student before infer teacher load`, then failed with
  `Qwen3.6 MoE weight offload is not supported (OPD teacher time-share is dense-only)`.

Pending remote 35B smoke rerun after the MoE offload support lands:

- Gate: Qwen3.6-35B-A3B-FP8 OPD, `ARLE_OPD_INFER_ROLLOUT=1`,
  `ARLE_OPD_ENGINE_OFFLOAD=student`, `--steps 1`, `--rollout-len 8`.
- Expected reachability signal: `infer_rollout_generate_start` followed by a
  completed OPD step, or a later measured blocker unrelated to initial teacher
  load residency / MoE offload coverage.

## Rule

OPD infer-rollout validation on 35B must include the startup residency ordering,
not only the per-step offload/reload path. A time-share mode is incomplete if it
only frees memory after the first step, and it must cover the actual Qwen3.6 MoE
weight container, not only dense MLP layers.
