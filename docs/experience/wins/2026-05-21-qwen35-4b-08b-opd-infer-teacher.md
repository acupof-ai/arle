# Qwen3.5-4B -> Qwen3.5-0.8B InferTeacher OPD Bench

## Context

Path B replaces the in-train teacher with the infer runtime teacher path:
teacher logits come from `LoadedInferenceEngine::forward_token_logits`, then a
BF16 device-to-device bridge imports the logits into the train autograd store.
This run is the first real cross-size bench through that architectural path.

The user relaxed the earlier Qwen3.5-0.8B self-teach gate to 5 s/step. Track A
then measured the 0.8B self-teach step at 2.151 s and showed the bridge itself
was not the bottleneck. This entry records Track B: Qwen3.5-4B teacher via
infer to Qwen3.5-0.8B-Base LoRA student via train.

## Command

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
CARGO_BUILD_JOBS=1 \
cargo run -p train --example opd_step_cuda_infer_teacher_train --release --features cuda -- \
  --teacher-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-4B \
  --student-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base \
  --prompts-file examples/opd/sample-prompts.jsonl \
  --steps 200 \
  --rollout-len 8 \
  --lr 1e-5 \
  --eval-steps 0,50,100,200 \
  --prompt-max-tokens 16 \
  --max-step-seconds 30 \
  | tee bench-output/2026-05-21-qwen35-4b-08b-opd-infer-teacher/run.txt
```

Raw artefacts:

- `bench-output/2026-05-21-qwen35-4b-08b-opd-infer-teacher/run.txt`
- `bench-output/2026-05-21-qwen35-4b-08b-opd-infer-teacher/nvidia-smi-before.txt`
- `bench-output/2026-05-21-qwen35-4b-08b-opd-infer-teacher/nvidia-smi-monitor.csv`
- `bench-output/2026-05-21-qwen35-4b-08b-opd-infer-teacher/nvidia-smi-after.txt`

## Environment

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GiB class
- Teacher: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-4B`
- Student: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base`
- Student mode: LoRA, rank 16, alpha 32, target set `attention-qv`
- Prompts: `examples/opd/sample-prompts.jsonl`
- Prompt split: 16 train, 4 held-out
- Rollout length: 8 generated tokens
- LR: `1e-5`
- CUDA graph: enabled for infer teacher runtime

## Results

The run completed without OOM or NaN.

| Metric | Value |
|---|---:|
| Steps | 200 |
| Total training wall-clock | 1155.176 s |
| Mean step seconds | 5.658789 |
| Median step seconds | 5.728152 |
| Step sigma | 0.650019 s |
| Step sigma pct | 11.487% |
| Peak GPU used | 14758 MiB |
| Idle/non-bench baseline used | 955 MiB |
| Net peak over baseline | 13803 MiB |

The high step-time sigma is expected for this run because the prompt lengths
are intentionally varied. `rollout_len=8` means total scored length ranges from
15 to 22 tokens across the prompt set.

Average phase attribution over 200 steps:

| Phase | Avg seconds | Share |
|---|---:|---:|
| Student rollout | 2.566167 | 45.35% |
| Backward | 2.281542 | 40.32% |
| Teacher forward total | 0.463884 | 8.20% |
| Student KL forward | 0.323720 | 5.72% |
| Post-step cleanup | 0.012850 | 0.23% |
| KL loss compute + readback | 0.009388 | 0.17% |
| Optimizer step | 0.000596 | 0.01% |
| Grad clip | 0.000534 | 0.01% |
| D2D bridge import | 0.000413 | 0.01% |

KL trajectory:

| Step | Train KL | Held-out KL |
|---:|---:|---:|
| 0 | 1.509985423809e-5 | 1.738248056427e-5 |
| 50 | 1.503763178334e-5 | 1.730247004161e-5 |
| 100 | 1.497842333720e-5 | 1.722338788568e-5 |
| 200 | 1.484889571657e-5 | 1.702545500848e-5 |

Reduction at step 200:

| Metric | Delta |
|---|---:|
| Train KL | -1.662% |
| Held-out KL | -2.054% |
| Sampled train-step loss | -22.058% |

## What Worked

The architectural Path B bench is licensed under the user's Track B criterion:
it completed on the 16 GiB card and both train and held-out KL decreased
monotonically at every eval point.

The D2D bridge is not the current limiter. At the 4B -> 0.8B shape it averages
0.413 ms per step, while rollout plus backward account for 85.7% of wall-clock.
The infer teacher itself averages 464 ms per step, which is visible but not the
dominant constraint.

## Problems

This is not a convergence claim. The 200-step KL improvement is small
(-2.054% held-out). It is enough to validate the runnable cross-size path and
directionality, but not enough to claim task-quality improvement. A longer run
or a stronger prompt/eval setup is required before using this as a quality
headline.

The 14.8 GiB peak leaves limited headroom on this 16 GiB card. This is workable
for Qwen3.5-4B BF16 teacher plus 0.8B LoRA student, but the 9B teacher still
needs the quantized infer-teacher path or another memory reduction.

## Rule

For cross-runtime OPD, measure the teacher bridge separately from student
rollout/backward. The bridge can look architecturally suspicious, but this run
shows it is a sub-millisecond phase; optimizing it before rollout/backward
would be the wrong next axis.
