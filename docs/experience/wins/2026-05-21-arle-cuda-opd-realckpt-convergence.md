# ARLE CUDA OPD Real-Checkpoint Convergence Eval

## Goal

Run the now-viable CUDA OPD substrate on real Qwen3-0.6B weights and verify
whether a lightly perturbed full-finetune student converges back toward a
frozen teacher.

Verdict: **WEAK**, not licensed as `CONVERGES`.

The run proved the 500-step real-checkpoint experiment is operational at
sub-0.3 s/step, but it did not satisfy the eval-improvement acceptance bar.
Decode overlap recovered on training prompts, while held-out overlap missed the
target and distribution KL worsened versus the already-close step 0 baseline.

## Command

```bash
mkdir -p bench-output/2026-05-21-arle-cuda-opd-realckpt-train
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu --format=csv \
  > bench-output/2026-05-21-arle-cuda-opd-realckpt-train/nvidia-smi-before.txt

NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_realckpt_train --release --features cuda \
  | tee bench-output/2026-05-21-arle-cuda-opd-realckpt-train/run.txt

nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu --format=csv \
  > bench-output/2026-05-21-arle-cuda-opd-realckpt-train/nvidia-smi-after.txt
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
- Optimizer: AdamW lr=5e-5, betas=(0.9, 0.999), eps=1e-8, wd=0
- Rollout: `rollout_len=8`
- Decode eval: greedy 16-token suffix

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

First-step safety passed: `0.260722 s`, under the `0.5 s` stop threshold.

Training wall-clock:

| Metric | Value |
|---|---:|
| total steps | 500 |
| total training wall seconds | 177.526524 |
| mean step seconds | 0.297141 |
| median step seconds | 0.297430 |
| first sampled OPD loss | 1.788745e-5 |
| step 200 sampled OPD loss | 4.003397e-5 |
| final sampled OPD loss | 3.929456e-5 |

Eval summary:

| Step | Train overlap % | Held-out overlap % | Train KL | Held-out KL |
|---:|---:|---:|---:|---:|
| 0 | 74.218750 | 50.000000 | 1.433391e-2 | 2.172812e-2 |
| 50 | 25.781250 | 7.812500 | 5.507875e-1 | 5.110196e-1 |
| 100 | 50.000000 | 10.937500 | 2.465412e-1 | 4.856422e-1 |
| 200 | 63.281250 | 23.437500 | 1.672269e-1 | 4.346739e-1 |
| 500 | 63.281250 | 14.062500 | 8.822937e-2 | 3.456596e-1 |

Acceptance status:

| Criterion | Result | Status |
|---|---:|---|
| Training-prompt decode overlap >= 40% at step 200 | 63.281250% | pass |
| Held-out decode overlap >= 20% at step 500 | 14.062500% | fail |
| Training KL loss reduction >= 30% by step 200 | -1066.652602% | fail |

## Interpretation

This is not a usable convergence win. The student starts very close to the
teacher because both are loaded from the same checkpoint and the perturbation
is only `1e-3`. Initial decode overlap is already high, especially held-out
overlap at 50%. OPD then damages distribution alignment sharply before partial
training-prompt overlap recovery:

- training KL worsens from `1.433e-2` to `1.672e-1` by step 200;
- held-out KL worsens from `2.173e-2` to `3.457e-1` by step 500;
- held-out decode overlap falls from 50.0% at step 0 to 14.1% at step 500.

The sampled OPD loss reported by the training step is also not decreasing from
the first measured step. It rises from `1.789e-5` to `4.003e-5` at step 200.

## Problems

The acceptance frame exposed a setup issue rather than a CUDA correctness
issue. Starting from the same real checkpoint plus `1e-3` perturbation makes
step 0 a strong baseline. Full-finetune AdamW at `5e-5` over single-prompt
on-policy rollouts appears to overfit or perturb global behavior faster than
it improves the evaluated teacher-conditioned distributions.

Decode overlap alone is not SOLID here: the initial overlap is already high,
and several prompts are repetitive. The true forward-KL table is the stricter
signal and says the distribution moved away from the teacher.

## Next Axis

Run a matched LR/perturbation sweep before changing kernels or model logic:

- `lr = 1e-6, 5e-6, 1e-5, 5e-5`
- keep the same 8 train prompts, 4 held-out prompts, perturb seed, and
  500-step budget
- license only if step 200 train KL drops by >= 30% from step 0 and step 500
  held-out KL is not worse than step 0 while held-out overlap is >= 20%

Kill the axis if no LR improves held-out KL versus step 0; the next likely
root cause would be objective setup, not optimizer speed.

## Artefacts

- `bench-output/2026-05-21-arle-cuda-opd-realckpt-train/run.txt`
- `bench-output/2026-05-21-arle-cuda-opd-realckpt-train/nvidia-smi-before.txt`
- `bench-output/2026-05-21-arle-cuda-opd-realckpt-train/nvidia-smi-after.txt`
