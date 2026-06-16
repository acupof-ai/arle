# OPD Student Rollout Uses Infer KV Decode

## Context

The corrected OPD arm needs rollout-256. The autograd rollout path recomputed a
full forward for every sampled token, making one training step take roughly tens
of minutes and blocking a clean capability verdict.

## What Worked

`train opd` now loads an in-process infer student for the rollout phase and uses
the serving scheduler token-generation path for prompt-plus-rollout generation.
That keeps a KV slot alive and performs incremental decode. Teacher scoring and
student train forward/backward still use the OPD autograd path.

The infer rollout path is guarded to `--lora-target-set attention-qv` for now.
The current infer-side LoRA remerge only syncs full-attention q/v adapters, so
running `all-linear` would desynchronize the rollout policy from the trained
student. This entry licenses the rollout-speed seam, not the final all-linear
capability verdict.

Measured on .62 GPU1 with Qwen3.5-4B teacher, Qwen3.5-0.8B student, GSM8K
question-only prompts, rollout_len=256, temperature=1.0:

```text
run_dir=/data01/arle-opd-runs/opd-infer-rollout-probe-6604da1-20260616-221402
step 1/1 loss 0.000003 rollout_len 328
opd_step_profile step=1 total_seconds=88.446974 student_rollout_seconds=2.017471 teacher_forward_seconds=34.323858 student_forward_seconds=8.577225 kl_loss_seconds=8.690974 backward_seconds=43.317963 optimizer_step_seconds=0.055234
real 175.23
```

Local verification:

```text
cargo fmt -p cli -p train -p infer-api --check
cargo check -p cli --release --no-default-features --features cpu,no-cuda
cargo test -p train --release --no-default-features --features no-cuda
CUDARC_CUDA_VERSION=12090 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib
CUDARC_CUDA_VERSION=12090 cargo check -p cli --release --no-default-features --features cuda,no-cuda
CUDARC_CUDA_VERSION=12090 cargo clippy -p cli -p train -p infer-api --release --no-default-features --features cuda,no-cuda -- -D warnings
```

## Rule

Do not run OPD capability verdicts through the autograd rollout bottleneck, and
do not claim an all-linear verdict until the infer rollout engine syncs the same
adapter target set as the train student.
