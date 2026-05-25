# ARLE OPD End-to-End Trace

Related:
`docs/projects/2026-05-24-opd-mainline-task-backlog.md` T2,
`docs/bench-and-trace-spec.md`, and
`docs/experience/wins/2026-05-25-p5-pure-opd-5k-capability-sweep.md`.

## Goal

Measure the P5 pure-OPD shape end to end and rank the real wall-clock
bottlenecks before licensing another optimization axis.

## Hypothesis

Before the run, the likely bottleneck was student-side train work, not KL or
checkpoint save. Teacher scoring was suspected to be material but capped
because P5 already used the in-process infer teacher path.

## Command

Wall-clock anchor:

```bash
OUT=bench-output/2026-05-25-t2-opd-end-to-end-trace
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
NVCC_CCBIN=g++-14 \
target/release/examples/opd_step_cuda_infer_teacher_train \
  --teacher-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-4B \
  --student-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base \
  --prompts-file examples/opd/sample-prompts.jsonl \
  --steps 5 --rollout-len 8 --lr 2e-5 \
  --eval-steps 0 --prompt-max-tokens 16 --max-step-seconds 60 \
  --save-student-checkpoint "$OUT/checkpoints-anchor" --save-every 5
```

Nsight Systems pass, same shape:

```bash
OUT=bench-output/2026-05-25-t2-opd-end-to-end-trace
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
NVCC_CCBIN=g++-14 \
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --stats=true --force-overwrite=true --output "$OUT/nsys-p5-steps5" \
  target/release/examples/opd_step_cuda_infer_teacher_train \
    --teacher-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-4B \
    --student-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base \
    --prompts-file examples/opd/sample-prompts.jsonl \
    --steps 5 --rollout-len 8 --lr 2e-5 \
    --eval-steps 0 --prompt-max-tokens 16 --max-step-seconds 120 \
    --save-student-checkpoint "$OUT/checkpoints-nsys" --save-every 5
```

## Environment

- Commit: `57817fd` clean before T2 run.
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GB.
- CUDA trace tool: Nsight Systems 2025.6.3.
- Feature set: train example built with `--features cuda`.
- Models:
  - teacher: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-4B`
  - student: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base`
- Prompt corpus: `examples/opd/sample-prompts.jsonl`, `prompt_max_tokens=16`.
- GPU before/after each run: 1093 MiB used, 0% utilization.

## Results

Anchor run:

| metric | value |
| --- | ---: |
| eval step 0 | 7.756407 s |
| train total, 5 steps | 25.299457 s |
| mean step | 5.054796 s |
| median step | 5.151358 s |
| checkpoint step_000005 | 0.023730 s |
| checkpoint final | 0.001643 s |

Per-phase wall-clock, summed over the 5 anchor train steps:

| rank | phase | total seconds | mean per step | share of train wall |
| ---: | --- | ---: | ---: | ---: |
| 1 | backward | 10.823877 | 2.164775 | 42.83% |
| 2 | student_rollout | 10.591366 | 2.118273 | 41.91% |
| 3 | teacher_forward_total | 2.144397 | 0.428879 | 8.48% |
| 4 | student_forward | 1.599517 | 0.319903 | 6.33% |
| 5 | post_step_cleanup | 0.067115 | 0.013423 | 0.27% |
| 6 | kl_loss | 0.041829 | 0.008366 | 0.17% |
| 7 | grad_clip | 0.002639 | 0.000528 | 0.01% |
| 8 | optimizer_step | 0.002510 | 0.000502 | 0.01% |
| 9 | optimizer_zero_grad | 0.000104 | 0.000021 | 0.00% |

Teacher sub-phase split:

| sub-phase | total seconds | share of teacher | share of train wall |
| --- | ---: | ---: | ---: |
| infer_sync | 2.119283 | 98.83% | 8.39% |
| infer_forward_token_logits | 0.024247 | 1.13% | 0.10% |
| d2d_bridge_import | 0.000846 | 0.04% | 0.00% |

Nsight Systems same-shape run:

| metric | value |
| --- | ---: |
| nsys train total, 5 steps | 27.726662 s |
| nsys mean step | 5.540299 s |
| nsys CUDA API total | 11.080542 s |
| nsys GPU kernel total | 1.728842 s |
| nsys HtoD memcpy time | 1.688286 s |
| nsys DtoH memcpy time | 0.369085 s |
| nsys CUDA memset time | 0.177725 s |

Top CUDA API rows:

| rank | API | time | calls | share |
| ---: | --- | ---: | ---: | ---: |
| 1 | `cuStreamSynchronize` | 7.114805 s | 23362 | 64.2% |
| 2 | `cuMemcpyHtoDAsync_v2` | 1.695660 s | 22592 | 15.3% |
| 3 | `cuMemcpyDtoHAsync_v2` | 0.611997 s | 12116 | 5.5% |
| 4 | `cuMemsetD8Async` | 0.247275 s | 63712 | 2.2% |
| 5 | `cuStreamDestroy_v2` | 0.218832 s | 6 | 2.0% |
| 6 | `cuLaunchKernel` | 0.216417 s | 58721 | 2.0% |

Top GPU kernels are BF16 CUTLASS GEMMs:

| rank | kernel family | time | instances | share |
| ---: | --- | ---: | ---: | ---: |
| 1 | `cutlass_80_wmma_tensorop_bf16_16x16_128x2_tn` | 0.464093 s | 6540 | 26.8% |
| 2 | `cutlass_80_wmma_tensorop_bf16_16x16_128x1_tn` | 0.402405 s | 2640 | 23.3% |
| 3 | `cutlass_80_wmma_tensorop_bf16_32x32_128x2_tn` | 0.301656 s | 2700 | 17.4% |
| 4 | `cutlass_80_tensorop_bf16_s16816gemm_relu_256x128_32x3_tn` | 0.188830 s | 17 | 10.9% |
| 5 | `cutlass_80_wmma_tensorop_bf16_32x32_32x1_tn` | 0.061057 s | 144 | 3.5% |

## Bottleneck Rank

The ground-truth framing is anchor wall-clock, not the nsys kernel-only window.

1. **Student train loop dominates**: `backward + student_rollout` is
   84.74% of train-step wall-clock.
2. **Teacher scoring is secondary**: teacher forward is 8.48% of train wall,
   and 98.83% of that bucket is `infer_sync`.
3. **KL, optimizer, grad clip, and checkpoint are not P5-shape bottlenecks**:
   KL is 0.17%, optimizer+grad clip are about 0.02%, and checkpoint save is
   about 25 ms outside the step profile.

Nsight Systems agrees that raw GPU kernel time is not the wall-clock ceiling:
GPU kernels sum to 1.73 s over a 27.73 s profiled train window, while CUDA API
sync/memcpy time sums to 11.08 s. Optimizing one isolated kernel family cannot
move the P5 wall-clock unless it also reduces the surrounding synchronized
launch/copy pattern.

## License-Or-Kill Thresholds

Top-3 licensed follow-up gates:

| axis | license threshold | kill condition |
| --- | --- | --- |
| Student rollout/backward loop | PASS only if a same-shape 5-step anchor improves mean step by >=10% or reduces `student_rollout + backward` by >=15% without worse loss logging. | KILL if nsys shows only kernel-window improvement but wall-clock mean step changes <5%. |
| Teacher sync path | PASS only if teacher_forward_total drops by >=30% and total mean step drops by >=3%. | KILL if the improvement is only inside `infer_forward_token_logits`; that sub-phase is 0.10% of train wall. |
| KL/checkpoint/optimizer micro-optimizations | PASS only for a different shape where that phase exceeds 10% wall-clock, e.g. 512-token real-corpus acceptance. | KILL for P5 shape: current KL 0.17%, checkpoint about 25 ms per save, optimizer+grad clip about 0.02%. |

## Problems

- The direct OPD train example is not the HTTP serving scheduler path. CPU
  scheduling/admission/prefill/decode queues are therefore not meaningful T2
  buckets for P5; the relevant measured buckets are the train example's
  `phase_summary` fields plus nsys CUDA/API counters.
- NVTX ranges in this trace are scheduler-internal and tiny
  (`step_total` 0.782 ms over 25 instances). They do not represent the OPD
  train phases. The in-process `phase_summary` is the correct wall-clock source
  for this task.
- Nsight overhead raised mean step from 5.05 s to 5.54 s (+9.6%), so nsys
  percentages are diagnosis context only. License decisions use the anchor run.

## Learnings

- For P5-shape OPD, target student rollout/backward first. That is where the
  next meaningful wall-clock gain must show up.
- Teacher-forward micro work is capped at about 8.5% total-step share unless
  the recipe changes shape or routes through a slower external teacher.
- KL chunking is not a performance win for P5 shape; its value is memory
  eligibility for longer prompts, and T5b showed loss-only chunking is
  insufficient for 512-token real-corpus GKD.

## Artifacts

```text
bench-output/2026-05-25-t2-opd-end-to-end-trace/anchor-run.txt
bench-output/2026-05-25-t2-opd-end-to-end-trace/nsys-run.txt
bench-output/2026-05-25-t2-opd-end-to-end-trace/nsys-p5-steps5.nsys-rep
bench-output/2026-05-25-t2-opd-end-to-end-trace/nsys-p5-steps5.sqlite
bench-output/2026-05-25-t2-opd-end-to-end-trace/nsys-stats_cuda_api_sum.csv
bench-output/2026-05-25-t2-opd-end-to-end-trace/nsys-stats_cuda_gpu_kern_sum.csv
bench-output/2026-05-25-t2-opd-end-to-end-trace/nsys-stats_cuda_gpu_mem_time_sum.csv
bench-output/2026-05-25-t2-opd-end-to-end-trace/nsys-stats_cuda_gpu_mem_size_sum.csv
```

## Verdict

PASS for T2 measurement. The next optimization should not be another KL,
checkpoint, or isolated teacher-logits kernel pass for P5. It should attack
student rollout/backward wall-clock or explicitly switch to a different shape
whose phase table justifies a different bottleneck.
