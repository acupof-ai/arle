# 9B-TQ4 OPD Headline Switch

## Context

The licensed 9B-TQ4 -> 0.8B LoRA rollout-4 bench completed on the RTX 4070 Ti
SUPER 16 GB card:

- Teacher: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B-TQ4`
- Student: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base`
- LoRA: rank 16, `attention-qv`
- Rollout: 4 generated tokens
- Steps: 100
- Gate: no OOM, no NaN, held-out KL monotonically decreasing at
  `0/25/50/100`

The run is documented in
[`2026-05-21-arle-cuda-opd-9b-tq4-rollout4.md`](2026-05-21-arle-cuda-opd-9b-tq4-rollout4.md).

## What Changed

User-facing surfaces now use the 9B-TQ4 teacher run as the CUDA OPD headline:

- `README.md`
- `README.zh-CN.md`
- `web/src/data/content.ts`
- `docs/projects/2026-05-21-arle-opd-cuda-usage-manual.md`
- `examples/opd/run-distillation.sh`
- `docs/projects/img/2026-05-21-arle-vs-pytorch-opd-comparison.png`

The 4B BF16 -> 0.8B run remains a smaller scaling reference. The Qwen3-0.6B
LoRA and TRL GKD rows remain in the comparison table because that is the matched
PyTorch/HuggingFace baseline. No 9B TRL baseline is claimed.

## Headline Numbers

| Run | Mean step | Peak GPU memory | Held-out KL |
|---|---:|---:|---:|
| Qwen3.5-9B-TQ4 -> 0.8B LoRA r=16 | 4.302815 s | 15.9 GiB | -1.017% @ 100 steps |
| Qwen3.5-4B BF16 -> 0.8B LoRA r=16 | 5.66 s | 14.8 GiB | -2.05% @ 200 steps |
| Qwen3-0.6B LoRA r=16 | 0.140 s | 3.93 GiB | -36.4% @ 500 steps |
| TRL GKD matched Qwen3-0.6B | 0.408 s | 12.6 GiB | -5.5% @ 500 steps |

## Problems

The 9B-TQ4 result is a fit and directionality gate, not a task-quality
headline. Its 100-step held-out KL improvement is modest. Future claims about
instruction quality still need longer 9B-TQ4 runs and an external eval such as
IFEval.

The comparison image is intentionally labeled as licensed distillation runs,
not as a 9B-vs-TRL head-to-head. The only matched TRL comparison in this table
is still the Qwen3-0.6B GKD baseline.

## Rule

For user-facing OPD positioning, lead with the largest teacher that has passed
the functional bench gate on the target consumer GPU. Keep unmatched PyTorch
comparisons visibly scoped so the headline remains bench-honest.
