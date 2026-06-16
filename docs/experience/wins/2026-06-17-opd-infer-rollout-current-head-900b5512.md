# OPD rollout-256 current-head infer rollout refresh

## Context

Track 1 needed a clean current-head verdict for the OPD rollout path. The
desired behavior is narrow: rollout generation uses the inference engine's KV
decode path, while teacher scoring and student KL/backward remain in autograd.

No source change was needed in this tranche. Current `main` already has the
requested wiring:

- `InferStudent::generate_rollout` submits one request through
  `LoadedInferenceEngine::generate_token_ids`.
- `opd_step` syncs the live LoRA adapter into the infer student with
  `sync_lora_from_store` before sampling.
- `arle train opd` constructs `InferRolloutCtx` by default when CUDA is built;
  `ARLE_OPD_INFER_ROLLOUT=0` is the explicit fallback to the train-crate rollout.

`train self-opd` is not included in this verdict; it still passes no infer
rollout context and remains a separate path.

## Environment

- Local source: `900b5512` (`feat(train): group qwen moe backward`).
- Remote: `.62` / `iv-ye8is8fbi8s6iplibbg7`.
- Source sync: `git archive HEAD` to
  `/data01/arle-opd-runs/agent-infer-rollout-900b5512`, excluding local
  unrelated dirty work in `crates/infer-cuda/src/loader.rs`.
- Binary:
  `/data01/arle-target-opd-rollout-900b5512/release/arle`.
- Build env:
  `CUDA_HOME=/usr/local/cuda CUDARC_CUDA_VERSION=12090
  ARLE_CUDA_DISABLE_FLASHMLA=1 INFER_TILELANG_PYTHON=/usr/bin/python3`.
- Build result: release CUDA build passed in 4m33s.
- Dynamic link reality on the current pod image: this fresh target-opd build
  links `libssl.so.3`. Older target-opd binaries that link `libssl.so.1.1`
  are not runnable on this pod because `libssl.so.1.1` is absent.

## Verification

Local no-CUDA gates:

```text
cargo test -p train --release --no-default-features --features no-cuda --lib opd::tests:: -- --nocapture
PASS: 29 passed

cargo test -p cli --release --no-default-features --features no-cuda train_cli -- --nocapture
PASS: 10 passed
```

Remote OPD one-step gate:

```text
CUDA_VISIBLE_DEVICES=5
ARLE_OPD_STEP_PROFILE=1
ARLE_OPD_INFER_ROLLOUT=1

arle train opd \
  --backend cuda \
  --student-model /data01/modelscope-cache/Qwen/Qwen3___5-0___8B-Base \
  --teacher-model /data01/modelscope-cache/Qwen/Qwen3___5-4B \
  --prompts-file /data01/arle-opd-runs/h20-opd-corrected-gsm8k-4b0p8b-r256-alllinear-r32-20260616-144316/gsm8k-question-only-train.jsonl \
  --prompt-seed 0 \
  --steps 1 \
  --rollout-len 256 \
  --rollout-temperature 1.0 \
  --lora-target-set all-linear \
  --lora-rank 32 \
  --lora-alpha 64 \
  --kl-mask completion \
  --lr 1e-7
```

Result:

```text
step 1/1 loss 2.742942 rollout_len 328
opd_step_profile step=1 total_seconds=86.841409
  student_rollout_seconds=1.197110
  teacher_forward_seconds=33.125404
  student_forward_seconds=8.433203
  kl_loss_seconds=8.551920
  backward_seconds=43.867436
  optimizer_zero_grad_seconds=0.000008
  grad_clip_seconds=0.022096
  optimizer_step_seconds=0.051822
  post_step_cleanup_seconds=0.000260
ARLE train opd: ran 1 step(s) on Qwen3.x (vocab=248320, hidden=1024, layers=24, full_attn_gated=true, backend=cuda:0)
EXIT:0
```

Coherence / needle smoke on the same binary:

```text
CUDA_VISIBLE_DEVICES=6
arle --model-path /data01/modelscope-cache/Qwen/Qwen3___5-0___8B-Base \
  --max-tokens 64 \
  --temperature 0 \
  run --no-tools \
  --prompt "Context: The secret code is BLUE-73-MANGO. Question: What is 2 + 3? Also repeat the secret code exactly. Answer:"

loaded Qwen3___5-0___8B-Base (cuda) in 1.0s
2 + 3 = 5. The secret code is BLUE-73-MANGO.
```

## Verdict

The OPD rollout-256 generation phase is no longer the old 256x full-sequence
autograd loop. On current HEAD, the sampled rollout takes 1.197s and the full
step takes 86.84s. The remaining wall time is teacher forward plus differentiable
student forward/backward, which is outside the rollout-generation fix.

## Rule

Keep OPD sampled rollout generation on the infer-core scheduler/KV path. Use
autograd only where the tensors must be differentiable for KL and backward.
