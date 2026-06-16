# OPD rollout-256 current-head infer rollout gate

## Context

Track 1 needed a clean verdict that OPD rollout generation no longer uses the
autograd full-sequence loop for every generated token. The intended path is:
sync the live LoRA adapter into `InferStudent`, submit one infer-core request,
let the backend own a KV slot, then return to autograd only for the
teacher/student scoring and backward pass.

The implementation already exists on `main`:

- `InferStudent::generate_rollout` calls
  `LoadedInferenceEngine::generate_token_ids`.
- `LoadedInferenceEngine::generate_token_ids` submits one request through the
  serving scheduler/KV path.
- `opd_step` calls `sync_lora_from_store` before rollout when
  `InferRolloutCtx` is present.
- `arle train opd` loads that infer student by default; `ARLE_OPD_INFER_ROLLOUT=0`
  is the explicit A/B fallback to the train-crate rollout.

## Environment

- Local source: `378255fa` (`bench(train): record A1 attention scale`).
- Remote: `.62` / `arle`, GPU4 (`CUDA_VISIBLE_DEVICES=4`; GPU3 avoided).
- Source sync: clean `git archive HEAD`, excluding local unrelated
  `crates/infer-cuda/src/loader.rs` dirty work.
- Build dir:
  `/data01/arle-opd-runs/agent-infer-opd-rollout-head-378255fa`.
- Binary:
  `/data01/arle-target-opd-rollout-head-378255fa-noprebuilt/release/arle`.
- Build env:
  `CUDA_HOME=/usr/local/cuda CUDARC_CUDA_VERSION=12090
  ARLE_CUDA_DISABLE_FLASHMLA=1 INFER_TILELANG_PYTHON=/root/tl-venv/bin/python`.
- Note: `/data01/prebuilt-kernels` was stale for current HEAD and missed three
  DSv4 batched symbols, so this validation used a non-prebuilt CUDA-kernel
  build. That build passed in 4m00s.

## Verification

Local no-CUDA gates:

```text
cargo check -p train --release --no-default-features --features no-cuda --lib
PASS

cargo test -p train --release --no-default-features --features no-cuda --lib opd::tests:: -- --nocapture
PASS: 29 passed
```

Remote OPD 1-step gate:

```text
run=/data01/arle-opd-runs/opd-rollout-head-378255fa-r256-1step-20260617-003907
teacher=/data01/modelscope-cache/Qwen/Qwen3___5-4B
student=/data01/modelscope-cache/Qwen/Qwen3___5-0___8B-Base
prompts=/data01/arle-opd-runs/h20-opd-corrected-gsm8k-4b0p8b-r256-alllinear-r32-20260616-144316/gsm8k-question-only-train.jsonl
rollout_len=256
rollout_temperature=1.0
lora_target_set=all-linear
lora_rank=32
lora_alpha=64
kl_mask=completion
```

Result:

```text
step 1/1 loss 0.756020 rollout_len 328
opd_step_profile step=1 total_seconds=85.973270
  student_rollout_seconds=1.294490
  teacher_forward_seconds=32.142428
  student_forward_seconds=8.425907
  kl_loss_seconds=8.546977
  backward_seconds=43.891580
ARLE train opd: ran 1 step(s) on Qwen3.x (vocab=248320, hidden=1024, layers=24, full_attn_gated=true, backend=cuda:0)
```

The rollout phase is seconds, not the old 256 full-sequence autograd forwards.
The step wall is now dominated by teacher forward plus autograd student
forward/backward, which is the expected remaining work.

Coherence / needle smoke on the same binary and GPU:

```text
prompt="Context: The secret code is BLUE-73-MANGO. Question: What is 2 + 3? Also repeat the secret code exactly. Answer:"
output="2 + 3 = 5. The secret code is BLUE-73-MANGO."
```

## Ongoing Long Runs

Existing `.62` runs also show the same verdict:

- `opd_infer_rollout_r256_attnqv_250`: step 57-116
  `student_rollout_seconds` stays around 1.7-1.8s.
- `opd_verdict_33475048_gpu1` all-linear batchmean run reached step 68 and
  saved step 50; all-linear rollout phase is around 7.7-8.3s after step 3,
  which includes full all-linear LoRA sync plus KV decode. It is still seconds,
  not the old autograd rollout loop.

## Rule

OPD rollout generation is an inference-engine responsibility. Keep the sampled
rollout on the serving scheduler/KV path, then recompute under autograd only for
the differentiable KL/backward pass.
