# ARLE CUDA OPD Real-Checkpoint Convergence 2k Steps

## Goal

Extend the matched `lr=1e-7` real-checkpoint convergence run from `5939cc7`
from 500 steps to 2000 steps. The question was whether the 500-step win keeps
improving, plateaus, or regresses.

Verdict: **PLATEAU on held-out decode overlap; continues improving on KL**.

The 2000-step run completed without crash or NaN. Training overlap improves
past step 500, and both train and held-out KL continue falling. Held-out exact
token overlap, however, reaches `64.062500%` by step 100 and remains there
through step 2000. For this prompt set, the next eval axis should change
supervision breadth, not merely run longer.

## Command

```bash
mkdir -p bench-output/2026-05-21-arle-cuda-opd-realckpt-train-lr1e-7-2k
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu --format=csv \
  > bench-output/2026-05-21-arle-cuda-opd-realckpt-train-lr1e-7-2k/nvidia-smi-before.txt

NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_realckpt_train --release --features cuda -- \
  --lr 1e-7 --steps 2000 2>&1 \
  | tee bench-output/2026-05-21-arle-cuda-opd-realckpt-train-lr1e-7-2k/run.txt

nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu --format=csv \
  > bench-output/2026-05-21-arle-cuda-opd-realckpt-train-lr1e-7-2k/nvidia-smi-after.txt
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
- Prompt set: identical to
  [`2026-05-21-arle-cuda-opd-realckpt-convergence-lr1e-7.md`](2026-05-21-arle-cuda-opd-realckpt-convergence-lr1e-7.md)

GPU memory snapshots before and after the process both reported `955 MiB`
used, because snapshots were outside the benchmark process lifetime.

## Harness Change

`crates/train/examples/opd_step_cuda_realckpt_train.rs` now uses the 2k eval
cadence requested for this run:

```text
0, 100, 250, 500, 1000, 2000
```

It also prints steps 1-5 explicitly so future long convergence runs can do the
safety comparison without relying only on step 1 and step 10.

## Safety Check

The first-step and step-10 losses match the prior `5939cc7` 500-step artefact:

| Step | 5939cc7 loss | 2k run loss | Status |
|---:|---:|---:|---|
| 1 | 1.788745430531e-5 | 1.788745430531e-5 | match |
| 10 | 3.805938104051e-5 | 3.805938104051e-5 | match |

New first-five trace captured for future comparisons:

| Step | Loss | Rollout len | Step seconds |
|---:|---:|---:|---:|
| 1 | 1.788745430531e-5 | 12 | 0.167987 |
| 2 | 3.806550739682e-5 | 12 | 0.199352 |
| 3 | 3.408079282963e-5 | 13 | 0.197843 |
| 4 | 3.934769483749e-5 | 13 | 0.198156 |
| 5 | 2.925589251390e-5 | 14 | 0.200035 |

## Results

Training wall-clock:

| Metric | Value |
|---|---:|
| total steps | 2000 |
| total loop wall seconds | 421.727588 |
| mean OPD step seconds | 0.200187 |
| median OPD step seconds | 0.200263 |
| first sampled OPD loss | 1.788745e-5 |
| step 250 sampled OPD loss | 3.800779e-5 |
| final sampled OPD loss | 3.982442e-5 |

Eval trajectory:

| Step | Train overlap % | Held-out overlap % | Train KL | Held-out KL |
|---:|---:|---:|---:|---:|
| 0 | 74.218750 | 50.000000 | 1.433391e-2 | 2.172812e-2 |
| 100 | 75.000000 | 64.062500 | 9.946998e-3 | 2.086548e-2 |
| 250 | 81.250000 | 64.062500 | 7.980873e-3 | 2.003209e-2 |
| 500 | 78.906250 | 64.062500 | 6.851923e-3 | 1.938898e-2 |
| 1000 | 89.843750 | 64.062500 | 5.546355e-3 | 1.867448e-2 |
| 2000 | 89.843750 | 64.062500 | 4.304279e-3 | 1.757938e-2 |

Derived deltas:

| Metric | Step 0 | Step 500 | Step 2000 | Interpretation |
|---|---:|---:|---:|---|
| train overlap | 74.218750% | 78.906250% | 89.843750% | improves past 500 |
| held-out overlap | 50.000000% | 64.062500% | 64.062500% | plateaus after early jump |
| train KL | 1.433391e-2 | 6.851923e-3 | 4.304279e-3 | keeps improving |
| held-out KL | 2.172812e-2 | 1.938898e-2 | 1.757938e-2 | keeps improving |

KL reductions from step 0:

| Point | Train KL reduction | Held-out KL reduction |
|---:|---:|---:|
| 250 | 44.32% | 7.81% |
| 500 | 52.19% | 10.77% |
| 1000 | 61.31% | 14.06% |
| 2000 | 69.97% | 19.09% |

## Interpretation

The "training keeps improving" hypothesis is partly true:

- train KL improves monotonically through step 2000;
- held-out KL also improves monotonically through step 2000;
- train exact-token overlap improves after step 500, reaching `89.84%`;
- held-out exact-token overlap does **not** improve after step 100/500.

This is a plateau, not a regression. The held-out overlap ceiling is likely a
supervision and eval-resolution limit for the current tiny prompt set:

- only 8 training prompts and 4 held-out prompts;
- greedy exact-token overlap is coarse at 16 tokens per prompt;
- the same fragile prompt behavior observed at 500 steps remains visible.

Longer training at the same prompt mix continues to reduce KL, but it does not
unlock new held-out exact-token matches by 2000 steps.

## Next Axis

Change supervision before running longer:

- increase training prompts from 8 to at least 64 token-id prompts, with the
  same 4 held-out prompts plus a larger held-out split;
- keep `lr=1e-7`, `perturb=1e-3`, `rollout_len=8` fixed for the first matched
  control;
- evaluate at 0, 500, 1000, 2000;
- license if held-out exact overlap exceeds `70%` or held-out KL falls at least
  `30%` by step 2000;
- kill if held-out overlap remains flat and held-out KL reduction stays below
  `20%`.

## Verification

- `cargo check -p train --example opd_step_cuda_realckpt_train --features cuda`:
  passed.
- 2000-step CUDA run: passed, no crash / no NaN, six eval points present.

## Artefacts

- `bench-output/2026-05-21-arle-cuda-opd-realckpt-train-lr1e-7-2k/run.txt`
- `bench-output/2026-05-21-arle-cuda-opd-realckpt-train-lr1e-7-2k/nvidia-smi-before.txt`
- `bench-output/2026-05-21-arle-cuda-opd-realckpt-train-lr1e-7-2k/nvidia-smi-after.txt`
