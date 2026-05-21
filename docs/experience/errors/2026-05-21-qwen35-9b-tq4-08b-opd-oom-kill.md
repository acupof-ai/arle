# Qwen3.5-9B-TQ4 -> Qwen3.5-0.8B OPD OOM KILL

## Context

After the untied `lm_head.weight` fix, Qwen3.5-9B-TQ4 full-model logits parity
improved from top-64 dominant relerr `1.17` to `0.180`. The user relaxed the
functional gate to `<=0.20` and moved the final license decision to a 100-step
OPD bench: no OOM, no NaN, and held-out KL monotonically decreasing at
`0/25/50/100`.

Bench target:

- Teacher: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B-TQ4` via infer
- Student: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base`
- Student mode: LoRA rank 16, `attention-qv`, lr `1e-5`
- Prompts: `examples/opd/sample-prompts.jsonl`
- Rollout length: `8`

## Evidence

Raw artifacts:

```text
bench-output/2026-05-21-qwen35-9b-tq4-08b-opd-infer-teacher/
```

Default infer CUDA Graph attempt:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
CARGO_BUILD_JOBS=1 \
cargo run -p train --example opd_step_cuda_infer_teacher_train --release --features cuda -- \
  --teacher-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B-TQ4 \
  --student-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base \
  --prompts-file examples/opd/sample-prompts.jsonl \
  --steps 100 \
  --rollout-len 8 \
  --lr 1e-5 \
  --eval-steps 0,25,50,100 \
  --prompt-max-tokens 16 \
  --max-step-seconds 90
```

It loaded both models and completed step-0 eval:

| Metric | Value |
| --- | ---: |
| Student load | `9.998135 s` |
| Infer teacher load | `120.854407 s` |
| Step-0 train KL | `1.499280006101e-5` |
| Step-0 held-out KL | `1.821073738029e-5` |
| Peak GPU used | `15814 MiB / 16376 MiB` |

Then the first training step failed before a `train_step` row:

```text
Error: InvalidInput("OPD student rollout Qwen3.5 forward autograd error: cuda alloc_zeros failed. Hint: verify the checkpoint tensor shapes match config.json, that teacher and student use compatible Qwen3.5-family layouts, and include this stage name in the OPD loader/model follow-up report.")
```

`--no-cuda-graph` control:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
CARGO_BUILD_JOBS=1 \
cargo run -p train --example opd_step_cuda_infer_teacher_train --release --features cuda -- \
  --teacher-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B-TQ4 \
  --student-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base \
  --prompts-file examples/opd/sample-prompts.jsonl \
  --steps 1 \
  --rollout-len 8 \
  --lr 1e-5 \
  --eval-steps 0,1 \
  --prompt-max-tokens 16 \
  --max-step-seconds 120 \
  --no-cuda-graph
```

This also loaded both models and completed step-0 eval, then failed at the same
student rollout allocation:

| Metric | Value |
| --- | ---: |
| Student load | `8.847640 s` |
| Infer teacher load | `109.766954 s` |
| Step-0 train KL | `1.414202961314e-5` |
| Step-0 held-out KL | `1.632275893826e-5` |
| Peak GPU used | `15872 MiB / 16376 MiB` |

## Root Cause

The 9B-TQ4 teacher is now numerically plausible enough to try OPD, but it does
not fit the current cross-runtime OPD memory plan on this 16 GiB card once the
correct untied dense `lm_head.weight` is loaded.

The failure occurs after both teacher and student load and after step-0 eval,
but before the first training step can complete. That places the immediate
pressure in student rollout/autograd activation allocation, not in teacher
startup or the D2D bridge. Disabling infer CUDA Graph does not recover enough
headroom, so graph capture is not the dominant memory source.

The practical memory budget is tighter than the earlier estimate because the
fixed 9B-TQ4 path includes the separate dense `lm_head.weight` plus packed
TurboQuant weights, dense fallback tensors, the 0.8B BF16 LoRA student, and
rollout/backward activations. The monitor peaked within roughly `500-560 MiB`
of the device limit before the first train step.

## Fix

Killed at the OPD bench no-OOM gate. Do not switch README, README.zh-CN, web
content, usage manual, or comparison PNG headline docs to 9B-TQ4.

Next memory axis should be explicit and single-variable. The most direct
candidate is reducing teacher memory, especially the now-correct but large
untied output projection path: quantize or otherwise memory-optimize
`lm_head.weight`, then rerun dense `lm_head` parity, full-model logits parity,
and the same OPD no-OOM/KL gate.

## Rule

Relaxing a numerical parity gate does not relax the runtime memory gate. A
teacher can be functionally plausible and still fail the OPD headline path if
the first train step cannot allocate rollout/autograd tensors on the target
card.
