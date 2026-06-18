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

The second rerun then proved the capacity ordering problem: the rollout student
offloaded 34.1 GiB and the teacher initial load succeeded, but the first step
failed while reloading the rollout student with the teacher still resident. The
CLI now also honors `ARLE_OPD_ENGINE_OFFLOAD=all/teacher` immediately after the
initial infer-teacher load, so `all` starts each step with both idle infer
engines offloaded. The existing windowed KL path reloads teacher only after
rollout and after offloading the rollout student.

Default behavior is unchanged when `ARLE_OPD_ENGINE_OFFLOAD` is unset or set to
`student`.

## Verification

Local gates:

- `rustfmt --edition 2024 --check crates/cli/src/train_cli.rs`
- `git diff --check`
- `CUDARC_CUDA_VERSION=12090 cargo check -p cli --release --no-default-features --features cpu,no-cuda --lib`
- `CUDARC_CUDA_VERSION=12090 cargo check -p train --release --no-default-features --features cuda,no-cuda --lib`
- `CUDARC_CUDA_VERSION=12090 cargo check -p cli --release --no-default-features --features cuda,no-cuda --lib`
- `CUDARC_CUDA_VERSION=12090 cargo check -p infer-cuda --release --no-default-features --features cuda,no-cuda --lib`
- `CUDARC_CUDA_VERSION=12090 cargo check -p cli --release --no-default-features --features cuda,no-cuda --lib`
- `CUDARC_CUDA_VERSION=12090 cargo clippy -p infer-cuda --release --no-default-features --features cuda,no-cuda --lib -- -D warnings`
- `CUDARC_CUDA_VERSION=12090 cargo clippy -p cli --release --no-default-features --features cpu,no-cuda --lib -- -D warnings`

Remote evidence:

- Pre-MoE-offload build `ab22f727` on `.62` reached
  `offload infer rollout student before infer teacher load`, then failed with
  `Qwen3.6 MoE weight offload is not supported (OPD teacher time-share is dense-only)`.
- MoE-offload build `bd1f2d27` on `.62` reached
  `student_pre_teacher_offloaded freed_mib=34057.8`, loaded the infer teacher,
  then failed at first-step rollout-student reload while the teacher was still
  resident (`infer student reload failed: reload moe.gate[35]`).
- Teacher-pre-step-offload build `1f104267` on `.62` passed the 35B FP8 OPD
  smoke with `ARLE_OPD_ENGINE_OFFLOAD=all`:
  - `student_pre_teacher_offloaded freed_mib=34057.8`
  - `teacher_pre_step_offloaded freed_mib=34057.8`
  - `infer_rollout_generate_start` -> `infer_rollout_generate_done`
    (`actual_rollout_len=11`, prompt 3 + rollout 8)
  - `teacher_reload_done` 5.190 s, `teacher_full_forward_done` 0.085 s
  - `student_hidden_forward_done` 2.295 s
  - `optimizer_step_done` at 26.547 s
  - JSON result: `losses=[4.843664646148682]`, `steps=1`, `teacher_runtime=Infer`
  - Exit: `EXIT:0`
- Rollout-256 target smoke on `.62` also passed with the same binary shape and
  `ARLE_OPD_ENGINE_OFFLOAD=all`:
  - Log start: `START 2026-06-18T06:21:29+00:00
    host=iv-ye8is8fbi8s6iplibbg7 ... model=/data01/models/Qwen3.6-35B-A3B-FP8 gpu=1`
  - `student_pre_teacher_offloaded freed_mib=34057.8`
  - `teacher_pre_step_offloaded freed_mib=34057.8`
  - `infer_rollout_generate_start` at 5.009 s ->
    `infer_rollout_generate_done` at 7.896 s (`actual_rollout_len=259`), so the
    256-token rollout generation itself took 2.886 s with infer KV-cache decode.
  - `teacher_reload_done` 4.959 s, `teacher_full_forward_done` 12.337 s for
    shape `[1, 259, 248320]`.
  - `student_hidden_forward_done` 66.712 s and `base_backward_done` 96.948 s:
    the remaining wall time is now the train/autograd forward-backward path, not
    rollout generation.
  - `optimizer_step_done` at 201.159 s.
  - JSON result: `losses=[0.42180460691452026]`, `steps=1`,
    `rollout_len=256`, `teacher_runtime=Infer`.
  - Exit: `EXIT:0`.

Remote command:

```bash
CUDA_VISIBLE_DEVICES=1 INFER_CUDA_DEVICES=0 INFER_TP_SIZE=1 \
ARLE_OPD_INFER_ROLLOUT=1 ARLE_OPD_ENGINE_OFFLOAD=all \
ARLE_OPD_STEP_TRACE=1 ARLE_OPD_STEP_PROFILE=1 \
/data01/arle-verify-ab22f727-target/release-fast/arle train opd \
  --backend cuda \
  --student-model /data01/models/Qwen3.6-35B-A3B-FP8 \
  --teacher-model /data01/models/Qwen3.6-35B-A3B-FP8 \
  --teacher-runtime infer \
  --steps 1 \
  --rollout-len 8 \
  --prompt-max-tokens 64 \
  --prompt-ids 1,3,8 \
  --logits-window-size 8 \
  --lora-rank 1 \
  --lora-alpha 2 \
  --lora-target-set attention-qv \
  --grad-clip 1.0 \
  --json
```

Remote log: `/tmp/arle_opd35_student_offload_smoke.log` on
`iv-ye8is8fbi8s6iplibbg7`.

Rollout-256 rerun used the same command with `--rollout-len 256`; remote log:
`/tmp/arle_opd35_r256_smoke.log` on `iv-ye8is8fbi8s6iplibbg7`.

## Rule

OPD infer-rollout validation on 35B must include the startup residency ordering,
not only the per-step offload/reload path. A time-share mode is incomplete if it
only frees memory after the first step, and it must cover the actual Qwen3.6 MoE
weight container, not only dense MLP layers.
