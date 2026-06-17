# OPD infer rollout current HEAD recheck

## Context

Track 1 asked to unlock the clean OPD verdict by moving rollout generation off
the autograd full-recompute loop and onto the inference engine. Current `main`
already has the required code path:

- `opd_step_with_teacher_forward_profiled_gkd_anchor` accepts
  `InferRolloutCtx`.
- Each step syncs live train LoRA tensors with `sync_lora_from_store`.
- Rollout generation calls `InferStudent::generate_rollout`, which submits one
  inference-engine request and decodes through the KV-cache path.
- Teacher scoring and student KL/backward remain in autograd.

This entry records the current-HEAD recheck after `37ffbd56`.

## Environment

- Host: `.62` / `iv-ye8is8fbi8s6iplibbg7`.
- Source: clean `git archive HEAD` from `37ffbd56`, excluding the local
  unrelated dirty `crates/infer-cuda/src/loader.rs`.
- Remote source: `/data01/arle-opd-runs/agent-infer-opd-rollout-37ffbd56`.
- Target: `/data01/arle-target-opd-rollout-37ffbd56`.
- Build env: `CUDA_HOME=/usr/local/cuda CUDARC_CUDA_VERSION=12090
  ARLE_CUDA_DISABLE_FLASHMLA=1 INFER_TILELANG_PYTHON=/root/tl-venv/bin/python`.
- Build result: `cargo build --release --features cuda` passed in 4m00s.
- Dynamic link check: `release/arle` links `libssl.so.1.1` and
  `libcrypto.so.1.1`.
- GPUs: GPU1 for Qwen3.6 FP8 rollout smoke, GPU2 for production CLI OPD gate;
  GPU3 was not used.

## Verification

### Current HEAD recheck after A16

Rechecked after `7a0f8d49` because the OPD rollout path is the gate for the
35B Path A pane, and the working tree had unrelated CUDA-loader dirt that must
not be part of the verdict.

Clean remote snapshot:

```text
local HEAD=7a0f8d49
remote source=/data01/arle-opd-runs/agent-infer-opd-rollout-7a0f8d49
target=/data01/arle-target-opd-rollout-7a0f8d49
build=CUDA_HOME=/usr/local/cuda CUDARC_CUDA_VERSION=12090 \
  ARLE_CUDA_DISABLE_FLASHMLA=1 INFER_TILELANG_PYTHON=/root/tl-venv/bin/python \
  cargo build --release --features cuda --bin arle
build result=PASS, 3m58s
ldd=libssl.so.1.1 + libcrypto.so.1.1
```

Production `arle train opd` gate, GPU0, `ARLE_OPD_INFER_ROLLOUT=1`:

```text
student=/data01/modelscope-cache/Qwen/Qwen3___5-0___8B-Base
teacher=/data01/modelscope-cache/Qwen/Qwen3___5-4B
prompts=/data01/arle-opd-runs/h20-opd-corrected-gsm8k-4b0p8b-r256-alllinear-r32-20260616-144316/gsm8k-question-only-train.jsonl
steps=1 rollout_len=256 target_set=all-linear rank=32 alpha=64 lr=1e-7
log=/data01/arle-opd-runs/agent-infer-opd-rollout-7a0f8d49/opd_rollout256_gpu0_20260617_112838.log

step 1/1 loss 2.054664 rollout_len 328
opd_step_profile step=1 total_seconds=86.374485
  student_rollout_seconds=1.307595
  teacher_forward_seconds=32.074022
  student_forward_seconds=8.421153
  kl_loss_seconds=8.546261
  backward_seconds=44.347757
```

Qwen3.6-35B-A3B-FP8 LoRA `InferStudent` needle smoke, GPU1,
`ARLE_QWEN35_DEEPGEMM=0` to avoid the known local DeepGEMM JIT toolchain
confounder:

```text
log=/data01/arle-opd-runs/agent-infer-opd-rollout-7a0f8d49/qwen36_fp8_rollout_smoke_gpu1_20260617_113624.log

qwen36_fp8_lora_load_gate_result load_seconds=13.718792
  used_delta_mib=34080.0 live_host_mib=2514.1
  hidden=2048 layers=40 vocab=248320 experts=256 topk=8
  target_set=all-linear adapters=62220

qwen36_fp8_lora_sync_gate_result
  infer_load_seconds=12.863261
  sync_seconds=1.704885

qwen36_fp8_lora_rollout_smoke_result
  prompt_tokens=25 generated_tokens=128 smoke_seconds=1.295202
  expect=Some("BLUE-73-MANGO") contains_expect=true
```

Decoded output was coherent and retained the exact needle:

```text
Math question: 2 + 3 = 5
Secret code to repeat exactly: BLUE-73-MANGO

<think>
Here's a thinking process:
...
The main instruction is: "Secret code to repeat exactly: BLUE-73-MANGO"
```

### Qwen3.6 35B FP8 LoRA rollout smoke

The first attempt without `ARLE_QWEN35_DEEPGEMM=0` reached train-side FP8 LoRA
load and perturb, then failed during infer engine load because `.62`'s host
compiler cannot JIT the native DeepGEMM C++20 path. That is the known
DeepGEMM-JIT environment blocker, not an OPD rollout wiring failure.

The isolated rollout gate was rerun with `ARLE_QWEN35_DEEPGEMM=0` to test the
OPD rollout path without the confounding JIT dependency:

```text
qwen36_fp8_lora_load_gate_result load_seconds=13.796522
  used_delta_mib=34080.0 live_host_mib=2514.1
  hidden=2048 layers=40 vocab=248320 experts=256 topk=8
  target_set=all-linear adapters=62220

qwen36_fp8_lora_sync_gate_result
  infer_load_seconds=12.817161
  sync_seconds=1.654809

qwen36_fp8_lora_rollout_smoke_result
  prompt_tokens=32 generated_tokens=128 smoke_seconds=1.327704
  expect=Some("BLUE-73-MANGO") contains_expect=true
```

Decoded output was coherent and repeated the exact needle:

```text
Math question: 2 + 3 = 5
Secret code to repeat exactly: BLUE-73-MANGO
```

### Production `arle train opd` rollout-256 gate

Current HEAD binary:

```text
/data01/arle-target-opd-rollout-37ffbd56/release/arle
libssl.so.1.1 => /lib/x86_64-linux-gnu/libssl.so.1.1
libcrypto.so.1.1 => /lib/x86_64-linux-gnu/libcrypto.so.1.1
```

Command shape:

```text
CUDA_VISIBLE_DEVICES=2
ARLE_OPD_STEP_PROFILE=1
ARLE_OPD_INFER_ROLLOUT=1

arle train opd --backend cuda
  --student-model /data01/modelscope-cache/Qwen/Qwen3___5-0___8B-Base
  --teacher-model /data01/modelscope-cache/Qwen/Qwen3___5-4B
  --prompts-file .../gsm8k-question-only-train.jsonl
  --prompt-seed 0 --steps 1 --rollout-len 256
  --rollout-temperature 1.0
  --lora-target-set all-linear --lora-rank 32 --lora-alpha 64
  --kl-mask completion --lr 1e-7
```

Reachability evidence:

```text
[arle train opd] loading infer rollout student from ...Qwen3___5-0___8B-Base
```

Result:

```text
step 1/1 loss 2.054664 rollout_len 328
opd_step_profile step=1 total_seconds=87.649836
  student_rollout_seconds=1.312957
  teacher_forward_seconds=32.357307
  student_forward_seconds=8.467933
  kl_loss_seconds=8.582613
  backward_seconds=45.292827
  optimizer_zero_grad_seconds=0.000005
  grad_clip_seconds=0.022725
  optimizer_step_seconds=0.055993
  post_step_cleanup_seconds=0.000259
ARLE train opd: ran 1 step(s) on Qwen3.x
EXIT:0
```

## Verdict

Rollout generation is no longer the OPD blocker on current HEAD. The rollout
phase is 1.31s for rollout-256 in the production CLI gate, and 1.33s for a
128-token Qwen3.6-35B-A3B-FP8 LoRA `InferStudent` smoke. The remaining
rollout-256 step time is dominated by differentiable teacher/student forward
and autograd backward, which this tranche intentionally does not move to the
inference engine.

The 35B FP8 direct rollout path is coherent and needle-positive. A full 35B OPD
step remains gated by the train-side 35B autograd/backward track and memory
layout, not by rollout generation.

## Rule

For OPD, sampled rollout generation belongs on infer-core KV decode. Autograd
should re-enter only where gradients are needed: teacher/student scoring,
loss, and student backward.
