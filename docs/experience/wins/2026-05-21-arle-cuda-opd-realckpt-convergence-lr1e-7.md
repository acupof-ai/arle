# ARLE CUDA OPD Real-Checkpoint Convergence at lr=1e-7

## Goal

Repeat the 500-step Qwen3-0.6B real-checkpoint OPD convergence run with the
same setup as the failed `lr=5e-5` run, changing only the AdamW learning rate
to `1e-7`.

Verdict: **CONVERGES** for this matched prompt set.

All pre-licensed acceptance gates passed:

- training-prompt decode overlap at step 200: `81.25%` >= `60%`;
- held-out decode overlap at step 500: `64.06%` >= `30%`;
- training forward KL reduction at step 200: `41.27%` >= `30%`.

This licenses the diagnosis from
`docs/research/2026-05-21-arle-cuda-opd-convergence-h1-vs-h2.md`: the earlier
catastrophic run was caused by an over-aggressive `5e-5` learning rate, not by
an OPD algorithm or CUDA substrate bug observed in this test.

## Command

```bash
mkdir -p bench-output/2026-05-21-arle-cuda-opd-realckpt-train-lr1e-7
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu --format=csv \
  > bench-output/2026-05-21-arle-cuda-opd-realckpt-train-lr1e-7/nvidia-smi-before.txt

NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_realckpt_train --release --features cuda -- --lr 1e-7 \
  | tee bench-output/2026-05-21-arle-cuda-opd-realckpt-train-lr1e-7/run.txt

nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu --format=csv \
  > bench-output/2026-05-21-arle-cuda-opd-realckpt-train-lr1e-7/nvidia-smi-after.txt
```

## Environment

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GiB
- Feature set: `--features cuda`
- Model: `/home/ckl/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B`
- Shape: hidden=1024, intermediate=3072, layers=28, vocab=151936,
  heads=16, kv_heads=8, head_dim=128, tied embeddings
- Teacher: frozen Qwen3-0.6B checkpoint
- Student: same checkpoint, all trainable params perturbed by uniform
  `[-1e-3, 1e-3]`
- Optimizer: AdamW lr=`1e-7`, betas=(0.9, 0.999), eps=1e-8, wd=0
- Rollout: `rollout_len=8`
- Decode eval: greedy 16-token suffix

GPU memory snapshots before and after the process both reported `955 MiB`
used, because snapshots were outside the benchmark process lifetime.

## Prompts

Training prompts:

```text
0 [1, 872, 198, 3456]
1 [1, 198, 1512, 429]
2 [1, 770, 3186, 25, 220]
3 [1, 644, 374, 279, 1887]
4 [1, 3838, 374, 264, 2077, 13]
5 [1, 785, 594, 287, 374, 1690]
6 [1, 3347, 11, 358, 1052, 429]
7 [1, 2610, 527, 1139, 304, 279, 1670]
```

Held-out prompts:

```text
0 [1, 4438, 374, 279, 2768]
1 [1, 1516, 374, 264, 1296, 4339]
2 [1, 785, 1401, 315, 279, 1967]
3 [1, 3198, 279, 1296, 25, 220]
```

## Results

First-step safety passed: `0.260222 s`, under the `0.5 s` stop threshold.

Training wall-clock:

| Metric | Value |
|---|---:|
| total steps | 500 |
| total training wall seconds | 176.848619 |
| mean step seconds | 0.296080 |
| median step seconds | 0.296389 |
| first sampled OPD loss | 1.788745e-5 |
| step 200 sampled OPD loss | 3.987005e-5 |
| final sampled OPD loss | 3.929620e-5 |

Eval summary:

| Step | Train overlap % | Held-out overlap % | Train KL | Held-out KL |
|---:|---:|---:|---:|---:|
| 0 | 74.218750 | 50.000000 | 1.433391e-2 | 2.172812e-2 |
| 50 | 75.000000 | 50.000000 | 1.141542e-2 | 2.124996e-2 |
| 100 | 75.000000 | 64.062500 | 9.946998e-3 | 2.086548e-2 |
| 200 | 81.250000 | 64.062500 | 8.418775e-3 | 2.025527e-2 |
| 500 | 78.906250 | 64.062500 | 6.851923e-3 | 1.938898e-2 |

Acceptance status:

| Criterion | Result | Status |
|---|---:|---|
| Training-prompt decode overlap >= 60% at step 200 | 81.250000% | pass |
| Held-out decode overlap >= 30% at step 500 | 64.062500% | pass |
| Training KL loss reduction >= 30% by step 200 | 41.266725% | pass |

## Delta vs lr=5e-5

| Metric | lr=5e-5 run | lr=1e-7 run | Interpretation |
|---|---:|---:|---|
| step 200 train overlap | 63.281250% | 81.250000% | +17.97 pp |
| step 200 train KL | 1.672269e-1 | 8.418775e-3 | lower is better |
| step 200 train KL change vs step 0 | -1066.65% | +41.27% | divergence -> convergence |
| step 500 held-out overlap | 14.062500% | 64.062500% | +50.00 pp |
| step 500 held-out KL | 3.456596e-1 | 1.938898e-2 | lower is better |

## Interpretation

The useful alignment signal is the teacher-conditioned forward KL trajectory,
not the sampled rollout loss scalar. The sampled OPD loss still rises from the
first measured rollout sample, but full-sequence eval KL falls monotonically
at every eval point:

- train KL drops from `1.433e-2` to `8.419e-3` by step 200 and `6.852e-3` by
  step 500;
- held-out KL drops from `2.173e-2` to `1.939e-2` by step 500;
- decode overlap improves or holds on both train and held-out prompts.

This is a real-checkpoint OPD improvement under the exact matched control that
previously failed, with only LR changed.

## Problems

The held-out set is still small: 4 hand-picked token-id prompts and 16 decoded
tokens each. The verdict is therefore scoped to this convergence smoke, not a
general benchmark of model quality.

Prompt 4 in the training set remains fragile: decode overlap fell to `0%` at
step 500 even though aggregate train KL improved. The next recipe sweep should
track per-prompt regressions, not only mean overlap and mean KL.

## Next Axis

Run a single-variable LR sweep around the licensed region with the same
prompts, perturb seed, and 500-step budget:

- `lr = 3e-8, 1e-7, 3e-7, 1e-6`
- license the largest LR that keeps step 200 train KL down by >= 30%, step 500
  held-out KL no worse than step 0, and no training prompt below 25% overlap at
  step 500
- kill any LR whose train KL exceeds 2x step 0 by step 50

## Artefacts

- `bench-output/2026-05-21-arle-cuda-opd-realckpt-train-lr1e-7/run.txt`
- `bench-output/2026-05-21-arle-cuda-opd-realckpt-train-lr1e-7/nvidia-smi-before.txt`
- `bench-output/2026-05-21-arle-cuda-opd-realckpt-train-lr1e-7/nvidia-smi-after.txt`
