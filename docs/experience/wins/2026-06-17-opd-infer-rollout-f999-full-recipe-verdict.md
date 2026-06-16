# OPD infer rollout current-head full-recipe verdict

## Context

Track 1 needed a clean verdict for the OPD rollout-generation bottleneck. The
target behavior is narrow: sampled rollout generation should use the inference
engine's KV-cache decode path, while teacher scoring and student KL/backward
stay in autograd because those tensors are differentiable.

No source change was needed in this tranche. Current `main` at `f9992164`
already has the requested wiring:

- `InferStudent::generate_rollout` submits one exact-length request through
  `LoadedInferenceEngine::generate_token_ids`.
- `opd_step` syncs the live LoRA adapter into the infer student with
  `sync_lora_from_store` once per step before sampling.
- `arle train opd` constructs `InferRolloutCtx` by default for CUDA; set
  `ARLE_OPD_INFER_ROLLOUT=0` only for the train-crate rollout fallback.

## Environment

- Local source: `f9992164` (`feat(train): add fp8 frozen-base qlora gate`).
- Remote: `.62` / `iv-ye8is8fbi8s6iplibbg7`.
- Source sync: `git archive HEAD` to
  `/data01/arle-opd-runs/agent-infer-opd-rollout-f9992164`, excluding the local
  unrelated dirty file `crates/infer-cuda/src/loader.rs`.
- Binary: `/data01/arle-target-opd-rollout-f9992164/release/arle`.
- Build env:
  `CUDA_HOME=/usr/local/cuda CUDARC_CUDA_VERSION=12090
  ARLE_CUDA_DISABLE_FLASHMLA=1 INFER_TILELANG_PYTHON=/root/tl-venv/bin/python`.
- Build result: release CUDA build passed in 3m59s.
- Dynamic link check: the binary links the pod-compatible `libssl.so.1.1` and
  `libcrypto.so.1.1`.

## Verification

Local no-CUDA gates:

```text
cargo test -p train --release --no-default-features --features no-cuda --lib opd::tests:: -- --nocapture
PASS: 29 passed

cargo test -p cli --release --no-default-features --features no-cuda train_cli -- --nocapture
PASS: 10 passed

cargo check -p train --release --no-default-features --features no-cuda --lib
PASS

cargo fmt --check
PASS

cargo clippy -p train --release --no-default-features --features no-cuda --lib -- -D warnings
PASS
```

Remote rollout-256 one-step gate:

```text
CUDA_VISIBLE_DEVICES=4
ARLE_OPD_STEP_PROFILE=1
ARLE_OPD_INFER_ROLLOUT=1

/data01/arle-target-opd-rollout-f9992164/release/arle train opd \
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
step 1/1 loss 2.054664 rollout_len 328
opd_step_profile step=1 total_seconds=85.606894
  student_rollout_seconds=1.315513
  teacher_forward_seconds=31.754497
  student_forward_seconds=8.469826
  kl_loss_seconds=8.585197
  backward_seconds=43.849376
  optimizer_zero_grad_seconds=0.000005
  grad_clip_seconds=0.021859
  optimizer_step_seconds=0.055160
  post_step_cleanup_seconds=0.000254
ARLE train opd: ran 1 step(s) on Qwen3.x (vocab=248320, hidden=1024, layers=24, full_attn_gated=true, backend=cuda:0)
EXIT:0
```

Coherence / needle smoke on the same binary:

```text
loaded Qwen3___5-0___8B-Base (cuda) in 1.0s

2 + 3 = 5. The secret code is BLUE-73-MANGO.
EXIT:0
```

Full-recipe 250-step run:

```text
run_dir=/data01/arle-opd-runs/opd-verdict-4b0p8b-gsm8k-r256-alllinear-r32-batchmean-33475048-20260616-230256
binary=/data01/arle-opd-runs/target-opd-verdict-33475048/release/arle
CUDA_VISIBLE_DEVICES=1
ARLE_OPD_STEP_PROFILE=1
rollout_len=256
lora_target_set=all-linear
lora_rank=32
steps=250
```

Completion evidence:

```text
step 250/250 loss 1.199268 rollout_len 357
opd_step_profile step=250 total_seconds=84.805799
  student_rollout_seconds=7.800164
  teacher_forward_seconds=23.635739
  student_forward_seconds=9.028645
  kl_loss_seconds=9.158448
  backward_seconds=44.040567
checkpoint_saved kind=full_materialized mode=opd step=250 dir=/data01/arle-opd-runs/opd-verdict-4b0p8b-gsm8k-r256-alllinear-r32-batchmean-33475048-20260616-230256/checkpoints/step_000250 seconds=13.817617
ARLE train opd: ran 250 step(s) on Qwen3.x (vocab=248320, hidden=1024, layers=24, full_attn_gated=true, backend=cuda:0)
```

Step-250 checkpoint needle smoke:

```text
SUMMARY len=115 depth=0.00 exact=1 partial=0 miss=0 DET
```

Capability curve, n=100 seed=0:

| point | GSM8K | MMLU |
|---|---:|---:|
| base_0p8b | 39/100 = 0.39 | 25/47 = 0.5319, invalid=10 |
| step50 | 23/100 = 0.23 | 31/57 = 0.5439 |
| step100 | 30/100 = 0.30 | 29/57 = 0.5088 |
| step150 | 35/100 = 0.35 | 32/57 = 0.5614 |
| step200 | 29/100 = 0.29 | 30/57 = 0.5263 |
| step250 | 38/100 = 0.38 | 29/57 = 0.5088 |

## Verdict

The rollout-generation bottleneck is fixed on current HEAD. Rollout-256
generation is 1.315s in the one-step gate instead of the old autograd
full-recompute path that made a step take about 30 minutes. The full OPD step is
now dominated by teacher forward plus differentiable student forward/backward,
which are outside the rollout-generation change.

The 250-step full recipe completes, and the step-250 checkpoint passes the
needle smoke. Capability direction is not licensed yet: GSM8K n=100 is flat to
slightly negative at step250 (38/100) versus base (39/100). That is a
recipe/objective verdict, not a rollout-runtime blocker.

## Rule

Separate the runtime unlock from the learning verdict. OPD sampled rollout
generation belongs on infer-core KV decode; autograd remains only for
differentiable KL/backward. A completed 250-step run proves the runtime path is
usable, but capability claims still need a positive, statistically defensible
eval curve.
