# BF16 Frozen-Base Substrate Cuts Real 4B OPD Peak Memory

## Goal

Verify that the BF16 frozen-base substrate saves memory inside the real OPD
pipeline, not just in isolated autograd unit coverage.

Matched-control baseline:
[`2026-05-21-qwen35-4b-08b-opd-infer-teacher.md`](2026-05-21-qwen35-4b-08b-opd-infer-teacher.md).

License threshold:

- PASS: same 4B-teacher OPD shape peaks at <= 13.5 GiB and KL remains
  monotonic.
- KILL: memory saving < 0.5 GiB or KL stops decreasing.

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
  --steps 200 --rollout-len 8 --lr 1e-5 \
  --eval-steps 0,50,100,200 \
  --prompt-max-tokens 16 --max-step-seconds 30 \
  | tee bench-output/2026-05-22-qwen35-4b-08b-opd-bf16-frozen-base/run.txt
```

Raw artefacts:

- `bench-output/2026-05-22-qwen35-4b-08b-opd-bf16-frozen-base/run.txt`
- `bench-output/2026-05-22-qwen35-4b-08b-opd-bf16-frozen-base/nvidia-smi-before.txt`
- `bench-output/2026-05-22-qwen35-4b-08b-opd-bf16-frozen-base/nvidia-smi-monitor.csv`
- `bench-output/2026-05-22-qwen35-4b-08b-opd-bf16-frozen-base/nvidia-smi-after.txt`

## Environment

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GiB class
- Driver/CUDA from `nvidia-smi`: 595.71.05 / 13.2
- Teacher: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-4B`
- Student: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base`
- Student mode: LoRA, rank 16, alpha 32, target set `attention-qv`
- Prompts: `examples/opd/sample-prompts.jsonl`
- Prompt split: 16 train, 4 held-out
- Rollout length: 8 generated tokens
- LR: `1e-5`
- CUDA graph: enabled for infer teacher runtime

## Results

The run completed without OOM or NaN. The BF16 frozen-base substrate is
licensed for the real 4B -> 0.8B OPD pipeline: peak GPU memory fell below the
13.5 GiB threshold and both train and held-out KL decreased monotonically.

| Metric | 2026-05-21 baseline | 2026-05-22 BF16 frozen-base | Delta |
|---|---:|---:|---:|
| Steps | 200 | 200 | matched |
| Mean step seconds | 5.658789 | 5.439205 | -3.88% |
| Median step seconds | 5.728152 | 5.562066 | -2.90% |
| Step sigma pct | 11.487% | 10.301% | -1.186 pp |
| Peak GPU used | 14758 MiB | 13447 MiB | -1311 MiB (-1.28 GiB) |
| Idle/non-bench baseline used | 955 MiB | 1082 MiB | +127 MiB |
| Net peak over idle | 13803 MiB | 12365 MiB | -1438 MiB (-1.40 GiB) |

Acceptance threshold: 13.5 GiB = 13824 MiB. Measured peak was 13447 MiB, so
the run passed with 377 MiB headroom under the threshold.

KL trajectory:

| Step | Train KL | Held-out KL |
|---:|---:|---:|
| 0 | 1.510384544190e-5 | 1.739055323924e-5 |
| 50 | 1.503710529960e-5 | 1.731266047500e-5 |
| 100 | 1.497718710652e-5 | 1.723095101624e-5 |
| 200 | 1.484045571942e-5 | 1.702685312921e-5 |

Reduction at step 200:

| Metric | Delta |
|---|---:|
| Train KL | -1.744% |
| Held-out KL | -2.091% |
| Sampled train-step loss | -21.993% |

Average phase attribution over 200 steps:

| Phase | Avg seconds | Share |
|---|---:|---:|
| Student rollout | 2.341925 | 43.06% |
| Backward | 2.271327 | 41.76% |
| Teacher forward total | 0.482336 | 8.87% |
| Student KL forward | 0.320954 | 5.90% |
| Post-step cleanup | 0.013213 | 0.24% |
| KL loss compute + readback | 0.008153 | 0.15% |
| Optimizer step | 0.000577 | 0.01% |
| Grad clip | 0.000526 | 0.01% |
| D2D bridge import | 0.000203 | 0.00% |

## What Worked

The BF16 frozen-base substrate saves real pipeline memory, not only isolated
operator memory. The matched OPD run saved 1.28 GiB absolute peak GPU memory
and 1.40 GiB net-over-idle memory versus the 2026-05-21 baseline while keeping
the same OPD quality direction.

The phase table also shows the optimization did not move the bottleneck into a
new host or bridge path. Rollout plus backward remain the dominant wall-clock
share at 84.8% of the step.

## Problems

This run intentionally used the current working tree because the benchmark
example already had parallel dirty edits in the workspace. No code was edited
or staged for this result; only new bench artefacts and this evidence entry are
part of the tranche.

The memory result licenses the substrate but does not make the 9B same-card
pipeline fit by itself. The 4B teacher case now has enough headroom for a
follow-up single-variable stretch such as `rollout_len=16` or
`prompt_max_tokens=32`, but those must be separate matched-control runs.

## Rule

Substrate-level memory changes need a pipeline-level memory gate before they
become durable claims. Here the real OPD pipeline confirmed the BF16 frozen-base
substrate as a memory win: lower peak memory, monotonic KL, and no new dominant
overhead path.
