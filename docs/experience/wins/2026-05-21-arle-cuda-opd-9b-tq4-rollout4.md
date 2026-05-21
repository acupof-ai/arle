# Qwen3.5-9B-TQ4 -> Qwen3.5-0.8B OPD Rollout-4 Bench

## Context

The first 9B-TQ4 -> 0.8B OPD attempt used `rollout_len=8` and failed the 16
GiB runtime memory gate after loading the validated dense BF16 `lm_head.weight`.
Quantizing `lm_head.weight` to TQ4 was killed because the output projection
itself incurred about 10% tensor-local error. This run keeps the validated
original Qwen3.5-9B-TQ4 teacher and instead halves rollout length from 8 to 4
to reduce rollout/autograd activation footprint.

## Command

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
  --rollout-len 4 \
  --lr 1e-5 \
  --eval-steps 0,25,50,100 \
  --prompt-max-tokens 16 \
  --max-step-seconds 90
```

Raw artifacts:

```text
bench-output/2026-05-21-qwen35-9b-tq4-08b-opd-rollout4/
```

## Environment

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GiB class
- Teacher: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B-TQ4`
- Teacher lm head: dense BF16, previously validated at `0.00176%`
  RMSE/ref-RMS module parity
- Student: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base`
- Student mode: LoRA rank 16, alpha 32, target set `attention-qv`
- Prompts: `examples/opd/sample-prompts.jsonl`
- Prompt split: 16 train, 4 held-out
- Rollout length: 4 generated tokens
- LR: `1e-5`
- CUDA graph: enabled for infer teacher runtime

## Results

The run completed without OOM or NaN.

| Metric | Value |
|---|---:|
| Steps | 100 |
| Total training wall-clock | 490.775 s |
| Mean step seconds | 4.302815 |
| Median step seconds | 4.408215 |
| Peak GPU used | 15878 MiB |
| Idle/non-bench baseline used | 955 MiB |
| Net peak over baseline | 14923 MiB |
| First sampled loss | 1.524423714727e-5 |
| Final sampled loss | 1.127386985900e-5 |
| Sampled loss reduction | -26.045% |

KL trajectory:

| Step | Train KL | Held-out KL |
|---:|---:|---:|
| 0 | 1.499280006101e-5 | 1.821073738029e-5 |
| 25 | 1.496206778029e-5 | 1.816574831537e-5 |
| 50 | 1.493208799275e-5 | 1.812112896005e-5 |
| 100 | 1.487306491299e-5 | 1.802543692975e-5 |

Reduction at step 100:

| Metric | Delta |
|---|---:|
| Train KL | -0.799% |
| Held-out KL | -1.017% |

## What Worked

The 9B-TQ4 teacher path is licensed under the user's functional gate: no OOM,
no NaN, and held-out KL decreases monotonically at every eval point. The
previous `rollout_len=8` OOM was an activation-footprint problem, not a blocker
for the validated original 9B-TQ4 teacher when rollout is reduced to 4.

The memory result is tight but workable on this 16 GiB card: peak was
`15878 / 16376 MiB`, about 498 MiB below the device limit and about 64 MiB above
the previous rollout-8 pre-step peak. The difference is that rollout-4 leaves
enough allocation headroom to complete all 100 train steps.

## Problems

This is a directionality and fit gate, not a quality headline by itself. The
100-step held-out KL improvement is modest (-1.017%). The 9B teacher now fits
and trains on the 16 GiB card, but a longer run or broader prompt set is still
needed for task-quality claims.

Step time is slower than the 4B BF16 teacher run per token because the teacher
is a larger infer runtime and synchronization dominates the teacher phase.
Across sampled steps, `infer_sync` is about 1.0-1.4 s while the D2D bridge stays
sub-millisecond. The next perf axis is teacher-runtime synchronization, not the
bridge import.

## Rule

When a memory gate fails after a numerically valid teacher loads, try reducing
activation footprint before weakening quantization fidelity. For this setup,
`rollout_len=4` is the smallest change that keeps the original validated 9B-TQ4
teacher and licenses the OPD path on a 16 GiB card.
