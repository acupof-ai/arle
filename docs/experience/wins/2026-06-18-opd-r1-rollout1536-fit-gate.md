# OPD R1 rollout-1536 fit gate

## Context

R1 math rollout needs long completions so CoT is not truncated. The load-bearing
gate was a single OPD step at `ROLLOUT_LEN=1536`, the length that previously
hit the `[seq,524288]` CUDA H2D OOM during frozen-base MoE backward.

This records the fit verdict for the current R1 launcher stack, not a
single-commit attribution to `f3a690dc` alone. The passing configuration requires
the current launcher defaults: teacher engine offload plus gradient
checkpointing.

## What Worked

The decisive pass used the same shape as `examples/opd/run-math-r1-35b-to-4b.sh`
with `--steps 1`, `--rollout-len 1536`, `ARLE_OPD_ENGINE_OFFLOAD=teacher`, and
`ARLE_OPD_GRADIENT_CHECKPOINTING=1`.

Pod evidence:

```text
run=/data01/arle-opd-runs/r1-fit-f3-1536-teacher-20260618-081932
gpu=2
mode=teacher
gradient_checkpointing=1
binary=/data01/arle-verify-ab22f727-target/release-fast/arle
exit_code=0
peak_gpu2_mib=55789
old_524288_htod_present=False
oom_present=False
```

Step trace:

```text
opd_step_trace event=start elapsed_seconds=0.000003 prompt_len=137 rollout_len=1536
opd_step_trace event=infer_rollout_generate_done elapsed_seconds=21.176336 actual_rollout_len=1673
opd_step_trace event=windowed_backward_start elapsed_seconds=21.176400
opd_window_trace kind=kl event=base_backward_start index=0 elapsed_seconds=0.000000 seed_grad_id=1019
opd_window_trace kind=kl event=base_backward_done index=0 elapsed_seconds=258.225108
opd_step_trace event=windowed_backward_done elapsed_seconds=481.514715
opd_step_trace event=optimizer_step_done elapsed_seconds=482.052675
```

Control notes from the same pod:

- `ARLE_OPD_ENGINE_OFFLOAD=student` failed before rollout/backward at infer
  student reload: teacher and student weights were co-resident.
- `ARLE_OPD_ENGINE_OFFLOAD=all` without gradient checkpointing reproduced the
  old frozen-MoE full-sequence H2D pressure as flattened
  `shape=[877133824]`, which is `1673 * 524288`.

## Rule

For R1 long-rollout OPD, fit is a stack property: frozen-MoE resident input-grad,
checkpoint inputs restricted to trainable LoRA tensors, teacher offload, and
gradient checkpointing must stay enabled together. Do not judge the long-rollout
gate from a hand-written command that omits launcher env defaults.
