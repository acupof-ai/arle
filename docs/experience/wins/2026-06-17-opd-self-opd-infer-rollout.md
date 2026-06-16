# OPD self-opd rollout uses infer KV decode

## Context

`arle train opd` already routed sampled student rollout through
`InferStudent::generate_rollout`, but `arle train self-opd` still passed no
infer rollout context. That left the self-opd path on the train-crate rollout
fallback instead of the serving scheduler/KV-cache decode path.

This tranche wires only rollout generation. Teacher scoring and the
differentiable student KL/backward pass remain in autograd.

## What Worked

- `run_self_opd_from_dir` now loads the CUDA infer student once and passes an
  `InferRolloutCtx` into each OPD step.
- The step syncs the current LoRA tensors into the infer student before
  rollout, then uses one infer-core generation request for prompt plus rollout.
- `InferStudent::sync_lora_from_store` now fails loud on unsupported adapter
  tensor names instead of silently skipping them.
- Qwen3.6 MoE adapter names are parsed for router/shared/expert projections.
  Dense BF16 resident MoE weights can be remerged in place; grouped/FP8 expert
  weights still fail loud rather than pretending to sync.
- CUDA LoRA remerge now uploads into the existing `DeviceMatrix` allocation so
  resident pointer tables and graph-captured addresses stay stable.

## Environment

- Host: `.62` / `iv-ye8is8fbi8s6iplibbg7`.
- GPU: H20 GPU5, GPU3 avoided.
- Source: `/data01/arle-track1-selfopd-infer-20260617054105`, synced from the
  local dirty tree with only the Track 1 rollout files.
- Binary:
  `/data01/arle-target-track1-selfopd-infer-cuda-20260617054305/release/arle`.
- Build env:
  `CUDA_HOME=/usr/local/cuda CUDARC_CUDA_VERSION=12090
  ARLE_CUDA_DISABLE_FLASHMLA=1 INFER_TILELANG_PYTHON=/root/tl-venv/bin/python`.
- Build note: `/data01/prebuilt-kernels` was stale for current DSv4 symbols, so
  the verified binary used source-built CUDA kernels. Build passed in 3m59s.
- Model: `/data01/modelscope-cache/Qwen/Qwen3___5-0___8B-Base`.

## Verification

Local gates:

```text
cargo fmt --check
PASS

cargo check -p cli --release --no-default-features --features no-cuda --lib
PASS

cargo test -p cli --release --no-default-features --features no-cuda train_cli -- --nocapture
PASS: 10 passed

CUDARC_CUDA_VERSION=12090 cargo check -p cli --release --no-default-features --features cuda,no-cuda --lib
PASS

cargo clippy -p cli --release --no-default-features --features no-cuda --lib -- -D warnings
PASS

cargo clippy -p train --release --no-default-features --features no-cuda --lib -- -D warnings
PASS

cargo test -p train --release --no-default-features --features no-cuda --lib qwen36_moe_lora -- --nocapture
PASS: finite-diff rel_err=5.862e-3 on experts.0.up_proj.lora_b

CUDARC_CUDA_VERSION=12090 cargo test -p train --release --no-default-features --features cuda,no-cuda --lib infer_student::tests:: -- --nocapture
PASS
```

Known unrelated CUDA clippy blocker:

```text
CUDARC_CUDA_VERSION=12090 cargo clippy -p train --release --no-default-features --features cuda,no-cuda --lib -- -D warnings
FAIL: pre-existing crates/infer-cuda/src/dsv4.rs needless_option_as_deref
```

Remote self-opd reachability smoke:

```text
CUDA_VISIBLE_DEVICES=5
ARLE_OPD_STEP_PROFILE=1
ARLE_OPD_INFER_ROLLOUT=1

arle train self-opd --backend cuda \
  --student-model /data01/modelscope-cache/Qwen/Qwen3___5-0___8B-Base \
  --steps 1 --rollout-len 32 --rollout-temperature 0 \
  --gkd-lambda 0.5 --lora-target-set attention-qv \
  --lora-rank 8 --lora-alpha 16 \
  --prompt-ids 1,3,8 --eval-ids 1,3,8,5 --lr 1e-7
```

Result:

```text
[arle train opd] loading infer rollout student from /data01/modelscope-cache/Qwen/Qwen3___5-0___8B-Base (max_seq_len=128)
step 1/1 loss 1.731703 grad_norm 0.635417 rollout_len 35
opd_step_profile step=1 total_seconds=6.173308 student_rollout_seconds=0.216084 teacher_forward_seconds=0.679249 student_forward_seconds=0.649780 kl_loss_seconds=0.012109 backward_seconds=4.608377 optimizer_step_seconds=0.001215
RUN_EXIT:0
```

Remote rollout-256 gate:

```text
CUDA_VISIBLE_DEVICES=5
ARLE_OPD_STEP_PROFILE=1
ARLE_OPD_INFER_ROLLOUT=1

arle train self-opd --backend cuda \
  --student-model /data01/modelscope-cache/Qwen/Qwen3___5-0___8B-Base \
  --steps 1 --rollout-len 256 --rollout-temperature 0 \
  --gkd-lambda 0.5 --lora-target-set attention-qv \
  --lora-rank 8 --lora-alpha 16 \
  --prompt-ids 1,3,8 --eval-ids 1,3,8,5 --lr 1e-7
```

Result:

```text
[arle train opd] loading infer rollout student from /data01/modelscope-cache/Qwen/Qwen3___5-0___8B-Base (max_seq_len=291)
step 1/1 loss 0.275275 grad_norm 0.114481 rollout_len 259
opd_step_profile step=1 total_seconds=44.052214 student_rollout_seconds=1.008975 teacher_forward_seconds=6.717663 student_forward_seconds=6.641446 kl_loss_seconds=0.121596 backward_seconds=29.543104 optimizer_step_seconds=0.000722
RUN_EXIT:0
```

Current-binary coherence / needle smoke:

```text
loaded Qwen3___5-0___8B-Base (cuda) in 1.3s

2 + 3 = 5. The secret code is BLUE-73-MANGO.
```

## Delta

| Metric | Before | After | Verdict |
|---|---:|---:|---|
| self-opd rollout path | train-crate fallback | infer-core request with KV decode | fixed |
| rollout-256 generation phase | operator-reported ~30 min/step blocker on autograd full-recompute path | 1.008975 s | pass |
| rollout-256 total step | blocked by rollout generation | 44.052214 s | bounded by autograd train/scoring |
| current binary needle | not measured in this tranche | `BLUE-73-MANGO` exact | pass |

## Problems

- Full Qwen3.6 FP8 grouped-MoE LoRA overlay is not implemented in this tranche.
  The code now fails loud for grouped/FP8 expert weights instead of silently
  skipping unsupported adapter tensors.
- CUDA clippy for `train` is still blocked by an unrelated pre-existing
  `dsv4.rs` lint. No DSv4 files were changed here.

## Rule

Rollout sampling is an inference/KV-cache job. Keep it on the infer-core
scheduler and reserve autograd for the differentiable KL/backward pass.
