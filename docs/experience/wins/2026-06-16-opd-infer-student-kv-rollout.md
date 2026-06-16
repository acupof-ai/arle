# OPD student rollout uses infer KV decode

## Context

OPD rollout generation was the clean-verdict blocker: rollout-256 through the
autograd student loop re-forwarded the whole growing sequence for every sampled
token, making a step take operator-reported tens of minutes. Teacher scoring
already used the infer API; the training forward/backward path still needs
autograd for gradients.

## Goal

Route only the OPD student rollout generation through the inference engine so
one request owns a KV slot and decodes incrementally, while leaving teacher
scoring and student backward unchanged.

## What Worked

- Added a programmatic token-id generation surface:
  `LoadedInferenceEngine::generate_token_ids` -> `ServeInferenceEngine::generate_token_ids`.
- Changed `InferStudent::generate_rollout` to submit one exact-length request
  to infer-core instead of calling `forward_token_logits` once per generated
  token.
- OPD step still syncs LoRA before rollout via `InferStudent::sync_lora_from_store`,
  then uses the infer request path for rollout and returns to autograd for the
  KL/backward pass.
- Preserved exact rollout length by forcing `ignore_eos=true` and clearing stop
  token ids on the per-rollout sampling copy, matching the old unconditional
  token loop.
- Extended the infer-side LoRA update contract from full-attention q/v to the
  dense Qwen3.5 all-linear target set: full-attention q/k/v/o, linear-attention
  qkv/z/b/a/out, and dense MLP gate/up/down projections. The merge path still
  fails loud for TP>1 and MoE MLP adapters.

## Environment

- Host: `.62` pod `sglang-eic-test`, GPU3 avoided.
- Binary: `/data01/arle-opd-runs/target-opd-infer-rollout/release/arle`
  built from `/data01/arle-opd-runs/agent-infer-a4239598-opd`.
- CUDA: `CUDARC_CUDA_VERSION=12090`, `/usr/local/cuda-12.9`.
- Build mode: `cargo build --release --no-default-features --features cli/cuda,cli/no-cuda --bin arle`
  with `ARLE_CUDA_KERNELS_PREBUILT_DIR=/data01/prebuilt-kernels`.
- Teacher: `/data01/modelscope-cache/Qwen/Qwen3___5-4B`.
- Student: `/data01/modelscope-cache/Qwen/Qwen3___5-0___8B-Base`.
- Prompt corpus:
  `/data01/arle-opd-runs/h20-opd-corrected-gsm8k-4b0p8b-r256-alllinear-r32-queued-20260616-145508/gsm8k-question-only-train.jsonl`.

## Verification

Local checks before the remote run:

```bash
cargo fmt --check
CUDARC_CUDA_VERSION=12090 cargo check -p train --release --no-default-features --features cuda,no-cuda --lib
CUDARC_CUDA_VERSION=12090 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib
CUDARC_CUDA_VERSION=12090 cargo clippy -p train --release --no-default-features --features cuda,no-cuda --lib -- -D warnings
CUDARC_CUDA_VERSION=12090 cargo clippy -p infer-api --release --no-default-features --features cuda,no-cuda --lib -- -D warnings
cargo test -p train --release --no-default-features --features no-cuda --lib opd::tests:: -- --nocapture
CUDARC_CUDA_VERSION=12090 cargo check -p train --release --no-default-features --features cuda,no-cuda --examples
CUDARC_CUDA_VERSION=12090 cargo check -p cli --release --no-default-features --features cuda,no-cuda --lib
CUDARC_CUDA_VERSION=12090 cargo check --release --no-default-features --features cli/cuda,cli/no-cuda --bin arle
CUDARC_CUDA_VERSION=12090 cargo clippy -p cli --release --no-default-features --features cuda,no-cuda --lib -- -D warnings
CUDARC_CUDA_VERSION=12090 cargo clippy --release --no-default-features --features cli/cuda,cli/no-cuda --bin arle -- -D warnings
```

Follow-up local checks for the dense all-linear LoRA remerge extension:

```bash
CUDARC_CUDA_VERSION=12090 cargo clippy -p train --release --no-default-features --features cuda,no-cuda --lib -- -D warnings
CUDARC_CUDA_VERSION=12090 cargo clippy -p infer-api --release --no-default-features --features cuda,no-cuda --lib -- -D warnings
CUDARC_CUDA_VERSION=12090 cargo test -p train --release --no-default-features --features cuda,no-cuda --lib infer_student::tests::parse_adapter_name_covers_all_linear_targets -- --nocapture
```

Remote one-step rollout-256 gate:

```text
step 1/1 loss 0.000003 rollout_len 328
opd_step_profile step=1 total_seconds=79.923760
  student_rollout_seconds=1.865626
  teacher_forward_seconds=33.505374
  student_forward_seconds=8.580048
  kl_loss_seconds=8.698071
  backward_seconds=35.829011
```

The rollout bottleneck moved from full autograd re-forward to infer KV decode:
rollout-256 is now 1.87 s inside the OPD step. The remaining wall time is
teacher/student training forward plus backward.

Inference coherence smoke on the same binary/student model:

```text
simple completion: ' 5Question: What is the sum of 2 and 3? Answer: 5Question: What'
needle completion: ' BLUE-73-MANGO ... Answer: BLUE-73-MANGO ...'
needle_hit=true
```

Persistent 250-step attention-qv run started on the already-built rollout
binary:

```text
tmux:  opd_infer_rollout_r256_attnqv_250
run:   /data01/arle-opd-runs/opd-infer-rollout-r256-attnqv-250step-20260616-142404
log:   /data01/arle-opd-runs/opd-infer-rollout-r256-attnqv-250step-20260616-142404/driver.log
ckpt:  /data01/arle-opd-runs/opd-infer-rollout-r256-attnqv-250step-20260616-142404/checkpoints
early: step 1 rollout=1.844086s, step 2 rollout=1.773307s,
       step 3 rollout=1.748096s
```

Clean follow-up all-linear runtime gate on `.62` used a fresh `3ef96951` source
tree plus only the all-linear sync patch, excluding unrelated local `loss.rs`
and loader mmap dirty hunks:

```text
source=/data01/arle-opd-runs/agent-infer-alllinear-sync-3ef96951-work
binary=/data01/arle-opd-runs/target-opd-alllinear-sync-3ef96951/release/arle
gpu=1
build=CUDA_HOME=/usr/local/cuda CUDARC_CUDA_VERSION=12090 cargo build --release --features cuda -p agent-infer
build_result=pass wall=4m00s
```

All-linear rollout-256 probes:

```text
run=/data01/arle-opd-runs/opd-alllinear-sync-r256-probe-3ef96951-20260616-223907
step 1/1 loss 0.000003 rollout_len 328
opd_step_profile step=1 total_seconds=86.239966 student_rollout_seconds=1.300164 teacher_forward_seconds=32.421836 student_forward_seconds=8.586466 kl_loss_seconds=8.702474 backward_seconds=43.722809

run=/data01/arle-opd-runs/opd-alllinear-sync-r256-probe2-3ef96951-20260616-224715
step 1/2 loss 0.000003 rollout_len 328
opd_step_profile step=1 total_seconds=85.071850 student_rollout_seconds=1.307183 teacher_forward_seconds=32.165360 student_forward_seconds=8.513436 kl_loss_seconds=8.630292 backward_seconds=42.875463
step 2/2 loss 0.000012 rollout_len 308
opd_step_profile step=2 total_seconds=65.282445 student_rollout_seconds=1.197390 teacher_forward_seconds=20.080001 student_forward_seconds=7.669110 kl_loss_seconds=7.750132 backward_seconds=36.179129
```

Current full-recipe all-linear 250-step run (same corrected GSM8K prompt
corpus, rollout-256, all-linear r32/alpha64) was started on `.62` GPU2 from a
fresh `86678c24` source snapshot. The older libssl1.1 `target-opd-*` binaries
were no longer runnable in the current pod image (`libssl.so.1.1` missing), so
the run uses a freshly built target-opd binary linked to the pod's OpenSSL 3
runtime rather than `/data01/arle-clean`.

```text
tmux=opd_track1_alllinear_r256_250_86678c24
run=/data01/arle-opd-runs/opd-track1-alllinear-r256-250-86678c24-20260616-160115
binary=/data01/arle-target-opd-track1-example/release/arle
source=/data01/arle-opd-runs/agent-infer-track1-953182a8-example
step 1/250 loss 2.742942 rollout_len 328
opd_step_profile step=1 total_seconds=88.492180 student_rollout_seconds=1.187512 teacher_forward_seconds=33.623830 student_forward_seconds=8.567812 kl_loss_seconds=8.696979 backward_seconds=44.882772
step 2/250 loss 3.042609 rollout_len 308
opd_step_profile step=2 total_seconds=66.556390 student_rollout_seconds=1.108200 teacher_forward_seconds=19.841534 student_forward_seconds=7.655881 kl_loss_seconds=7.738039 backward_seconds=37.787993
```

## Delta

| Metric | Before | After | Verdict |
|---|---:|---:|---|
| rollout-256 generation path | 256 autograd full-sequence forwards | one infer-core request with KV decode | fixed |
| rollout-256 measured time | operator-reported ~30 min/step blocker | 1.865626 s rollout phase | pass |
| total OPD step | blocked by rollout | 79.923760 s | now bounded by training/scoring |
| exact needle on same infer path | not measured in this tranche | `BLUE-73-MANGO` hit | pass |

## Problems

- Dense Qwen3.5 all-linear LoRA remerge is now covered by the clean `.62`
  all-linear 1-step and 2-step probes above. This is still an integration/speed
  gate, not a capability verdict.
- MoE student adapter remerge is not covered by this path. The projection
  resolver fails loud when a dense MLP projection is requested on a MoE layer.
- The old `target-opd-*` binaries on `.62` still depend on `libssl.so.1.1`.
  The verified binary is a fresh target-opd build that links system OpenSSL 3,
  not the unrelated `/data01/arle-clean` binary.

## Rule

Rollout generation is an inference problem, not a training-gradient problem.
Use the serving scheduler/KV cache for sampled trajectories, then use autograd
only for the differentiable student pass over the sampled tokens.
