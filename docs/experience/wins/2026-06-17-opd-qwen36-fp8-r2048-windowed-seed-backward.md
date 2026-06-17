# OPD Qwen3.6 FP8 r2048 windowed seed-backward gate

## Goal

Optimization / enablement: make one real Qwen3.6-35B-A3B-FP8 LoRA OPD step
complete at rollout length 2048 without OOM, using infer-engine rollout and
windowed KL scoring.

## Hypothesis

The remaining OOM was not rollout generation. It was the full `[seq, vocab]`
student/teacher logits and per-window backward retention. Keeping one cached
student hidden graph, scoring logits in 256-token windows, accumulating a
device-resident hidden gradient, and then running one seeded base backward
should reduce the peak enough for a full optimizer step.

## Command

35B command, remote `.62`, isolated tmux socket to avoid the default tmux
server being reaped:

```text
cd /data01/arle-r4-49657c61-window
tmux -L r4opd new-session -d -s r4_35b_r2048_final_w256_isolated \
  'set -o pipefail;
   export CUDA_VISIBLE_DEVICES=5;
   export CUDARC_CUDA_VERSION=12090;
   export INFER_TILELANG_PYTHON=/root/tl-venv/bin/python;
   export ARLE_CUDA_DISABLE_FLASHMLA=1;
   export ARLE_OPD_STEP_PROFILE=1;
   export ARLE_OPD_STEP_TRACE=1;
   export ARLE_OPD_GRADIENT_CHECKPOINTING=1;
   ./target/release/examples/opd_step_cuda_realckpt_lora_bench \
     --teacher-model /data01/models/Qwen3.5-0.8B \
     --student-model /data01/models/Qwen3.6-35B-A3B-FP8 \
     --steps 1 \
     --rollout-len 2048 \
     --logits-window-size 256 \
     --eval-steps none \
     --prompt-set 8 \
     --prompt-max-tokens 16 \
     --safety-first-step-max-seconds 999999 \
     2>&1 | tee /data01/r4_35b_r2048_final_w256_isolated.log;
   echo EXIT:${PIPESTATUS[0]} | tee -a /data01/r4_35b_r2048_final_w256_isolated.log'
```

Source: base `68e4756d` plus the OPD/autograd pathspec diff in this commit,
copied to `/data01/arle-r4-49657c61-window` and rebuilt with CUDA 12.9.

0.8B fit command, same code and host:

```text
cd /data01/arle-r4-49657c61-window
tmux -L r4opd new-session -d -s r4_08b_r2048_w256 \
  'set -o pipefail;
   export CUDA_VISIBLE_DEVICES=6;
   export CUDARC_CUDA_VERSION=12090;
   export INFER_TILELANG_PYTHON=/root/tl-venv/bin/python;
   export ARLE_CUDA_DISABLE_FLASHMLA=1;
   export ARLE_OPD_STEP_PROFILE=1;
   export ARLE_OPD_STEP_TRACE=1;
   export ARLE_OPD_GRADIENT_CHECKPOINTING=1;
   ./target/release/examples/opd_step_cuda_realckpt_lora_bench \
     --teacher-model /data01/models/Qwen3.5-0.8B \
     --student-model /data01/models/Qwen3.5-0.8B \
     --steps 1 \
     --rollout-len 2048 \
     --logits-window-size 256 \
     --eval-steps none \
     --prompt-set 8 \
     --prompt-max-tokens 16 \
     --safety-first-step-max-seconds 999999 \
     2>&1 | tee /data01/r4_08b_r2048_w256.log;
   echo EXIT:${PIPESTATUS[0]} | tee -a /data01/r4_08b_r2048_w256.log'
```

## Environment

- GPU: H20, 97 GB VRAM. 35B ran on GPU5; 0.8B fit arm ran on GPU6.
- 35B student: `/data01/models/Qwen3.6-35B-A3B-FP8`.
- 35B teacher: `/data01/models/Qwen3.5-0.8B`.
- 0.8B fit arm: teacher and student both `/data01/models/Qwen3.5-0.8B`.
- Student mode: LoRA rank 16, alpha 32, target set `attention-qv`.
- Rollout: infer-engine generation, prompt-set 8, first prompt selected.
- Loss path: pure KL, completion-only mask, logits window size 256.

## Results

The full 35B r2048 train step completed, including backward and optimizer:

```text
infer_rollout_generate_done elapsed_seconds=42.626864 actual_rollout_len=2052
student_hidden_forward_done elapsed_seconds=520.220266
window 8 done elapsed_seconds=51.034219 loss_accum=2.150222778320e0
base_backward_done elapsed_seconds=1333.129506
windowed_backward_done elapsed_seconds=2126.495680
optimizer_step_done elapsed_seconds=2126.497290
train_step step=1 loss=2.150222778320e0 rollout_len=2052
  step_seconds=2126.499271
  student_rollout_seconds=42.626093
  teacher_forward_seconds=229.138003
  student_forward_seconds=520.228546
  kl_loss_seconds=0.811757
  backward_seconds=1333.641275
training_summary total_steps=1 total_wall_seconds=2126.499285
EXIT:0
```

Peak observed GPU5 memory during the window/base-backward run was about
88.7 GiB, below the 97 GiB H20 limit. Earlier r2048 attempts failed before this
gate:

| route | result |
|---|---|
| `--logits-window-size 1024` | OOM in window KL backward |
| `--logits-window-size 512` | OOM around window 4 |
| `--logits-window-size 256`, default tmux | reached `base_backward_start`, then the default tmux session disappeared without `EXIT` |
| `--logits-window-size 256`, isolated tmux | PASS, `EXIT:0` |

The 0.8B r2048 fit gate also completed on the same windowed path:

```text
infer_rollout_generate_done elapsed_seconds=14.008227 actual_rollout_len=2052
student_hidden_forward_done elapsed_seconds=52.565500
base_backward_done elapsed_seconds=55.725616
windowed_backward_done elapsed_seconds=364.607221
optimizer_step_done elapsed_seconds=364.608430
train_step step=1 loss=1.924777403474e-2 rollout_len=2052
  step_seconds=364.608523
  student_rollout_seconds=14.008206
  teacher_forward_seconds=240.475094
  student_forward_seconds=52.566393
  kl_loss_seconds=0.898821
  backward_seconds=56.392314
training_summary total_steps=1 total_wall_seconds=364.608534
EXIT:0
```

Peak observed GPU6 memory in the 0.8B fit run was about 30 GiB. Window trace
lines showed per-window logits, not full-sequence logits:

```text
teacher_forward_done index=1 shape=[1, 256, 248320]
...
teacher_forward_done index=8 shape=[1, 256, 248320]
```

Numerical-semantics gates:

```text
cargo test -p train --release --test test_opd_step \
  opd_step_windowed_pure_kl_matches_full_loss_and_grads -- --nocapture
PASS

cargo test -p train --release --test test_opd_step \
  opd_step_windowed_pure_kl_matches_chunked_loss_and_grads -- --nocapture
PASS
```

The first test compares the new windowed path against the original full
logits/non-windowed OPD path on the same forced rollout. The second compares it
against the existing chunked-KL route.

Local gates for the pathspec diff:

```text
cargo fmt -p autograd -p train --check
rustfmt --edition 2024 --check \
  crates/autograd/src/tape.rs \
  crates/train/examples/opd_step_cuda_realckpt_train.rs \
  crates/train/src/loss.rs \
  crates/train/src/opd.rs \
  crates/train/src/qwen35.rs \
  crates/train/tests/test_opd_step.rs
cargo test -p autograd --release backward_from_seed_accumulate_targets_uses_explicit_output_grad -- --nocapture
cargo test -p train --release --test test_opd_step -- --nocapture
cargo test -p train --release --test test_opd_step opd_step_windowed_pure_kl_matches_full_loss_and_grads -- --nocapture
cargo test -p train --release --test test_opd_step opd_step_windowed_pure_kl_matches_chunked_loss_and_grads -- --nocapture
cargo clippy -p autograd --release -- -D warnings
cargo clippy -p train --release --tests -- -D warnings
CUDARC_CUDA_VERSION=12090 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib
```

All passed.

## Problems

This is not a speed win yet. The step is runnable but still slow:

- total step wall time: 2126.5 s, about 35.4 min;
- base backward alone: 1333.1 s, about 22.2 min;
- student hidden forward: 520.2 s;
- teacher window forward: 229.1 s.

`cargo fmt --check` for the whole workspace was not used as this tranche's gate
because the current HEAD has an unrelated `crates/cli/src/train_cli.rs` import
ordering diff under this rustfmt. The scoped package/file format checks for
`autograd` and `train` passed, and `train_cli.rs` was not touched.

## Learnings

For 35B OPD, the rollout engine path and logits windowing solve different
walls:

- infer-engine rollout makes generation cheap enough to stop being the first
  blocker;
- windowed logits plus seeded hidden backward removes the r2048 logits OOM;
- the next wall is the base student backward, not logits memory.

Do not substitute a small dense model for this gate. The accepted evidence is a
real Qwen3.6-35B-A3B-FP8 r2048 forward+backward+optimizer step.

## Delta vs baseline

Baseline for this gate was "no valid 35B r2048 optimizer step":

| arm | r2048 step | failure / wall |
|---|---:|---|
| pre-window/large-window | fail | OOM in logits/window backward |
| 0.8B window 256 | pass | 364.6 s total, peak observed memory about 30 GiB |
| window 256 + isolated tmux | pass | 2126.5 s total, base backward is next wall |
