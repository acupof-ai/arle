# ARLE CUDA OPD Convergence H1 vs H2 Diagnostic

## Goal

Distinguish whether the 2026-05-21 real-checkpoint OPD convergence failure was
caused by aggressive hyperparameters or an algorithm bug in the OPD KL /
AdamW / KV-cache rollout path.

Verdict: **H1 confirmed, lr is the binding culprit. H2 is not supported by
this experiment.**

## Hypotheses

- H1: hyperparameters are too aggressive. Gentler settings should stop the
  catastrophic KL blow-up.
- H2: OPD has an algorithm bug. Even tiny perturbation or tiny lr should
  diverge.

## Command

```bash
mkdir -p bench-output/2026-05-21-arle-cuda-opd-convergence-h1-vs-h2
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu --format=csv \
  > bench-output/2026-05-21-arle-cuda-opd-convergence-h1-vs-h2/nvidia-smi-before.txt

NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_realckpt_diag --release --features cuda \
  | tee bench-output/2026-05-21-arle-cuda-opd-convergence-h1-vs-h2/run.txt

nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu --format=csv \
  > bench-output/2026-05-21-arle-cuda-opd-convergence-h1-vs-h2/nvidia-smi-after.txt
```

## Environment

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GiB
- Feature set: `--features cuda`
- Model: `/home/ckl/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B`
- Teacher: frozen Qwen3-0.6B checkpoint
- Student: same checkpoint, full-finetune trainable params
- Optimizer: AdamW betas=(0.9, 0.999), eps=1e-8, wd=0
- Rollout: `rollout_len=8`
- Steps per config: 100
- Eval steps: 0, 25, 50, 100
- Decode eval: greedy 16-token suffix

GPU memory snapshots before and after the process both reported `955 MiB`
used, because the snapshots were outside the process lifetime.

## Configs

| Config | Perturb | LR | Purpose |
|---|---:|---:|---|
| A | 1e-5 | 5e-5 | isolate perturbation: same lr, 100x smaller perturb |
| B | 1e-3 | 1e-7 | isolate lr: same perturb, 500x smaller lr |
| C | 1e-5 | 1e-7 | minimum-everything stability baseline |

Prompt sets are identical to
`docs/experience/wins/2026-05-21-arle-cuda-opd-realckpt-convergence.md`.

## Results

### Config A: smaller perturb, original lr

| Step | Train overlap % | Train KL | Held-out overlap % | Held-out KL |
|---:|---:|---:|---:|---:|
| 0 | 91.406250 | 1.446806e-6 | 100.000000 | 2.472795e-6 |
| 25 | 18.750000 | 4.296175e-1 | 31.250000 | 5.313993e-1 |
| 50 | 9.375000 | 4.078324e-1 | 28.125000 | 4.029515e-1 |
| 100 | 39.843750 | 4.170165e-1 | 18.750000 | 4.698898e-1 |

Summary: train KL ratio `288232.53x`, train overlap delta `-51.56 pp`.
This diverges despite 100x smaller perturbation.

### Config B: original perturb, smaller lr

| Step | Train overlap % | Train KL | Held-out overlap % | Held-out KL |
|---:|---:|---:|---:|---:|
| 0 | 74.218750 | 1.433391e-2 | 50.000000 | 2.172812e-2 |
| 25 | 73.437500 | 1.230013e-2 | 50.000000 | 2.146773e-2 |
| 50 | 75.000000 | 1.141542e-2 | 50.000000 | 2.124996e-2 |
| 100 | 75.000000 | 9.946998e-3 | 64.062500 | 2.086548e-2 |

Summary: train KL ratio `0.693949x`, train KL reduction `30.61%`, held-out
KL ratio `0.960298x`, held-out overlap delta `+14.06 pp`.
This is stable and mildly improves the measured teacher-alignment metrics.

### Config C: smaller perturb, smaller lr

| Step | Train overlap % | Train KL | Held-out overlap % | Held-out KL |
|---:|---:|---:|---:|---:|
| 0 | 91.406250 | 1.446806e-6 | 100.000000 | 2.472795e-6 |
| 25 | 91.406250 | 1.089337e-6 | 100.000000 | 2.295764e-6 |
| 50 | 91.406250 | 9.498839e-7 | 100.000000 | 2.239104e-6 |
| 100 | 91.406250 | 8.128582e-7 | 100.000000 | 2.162430e-6 |

Summary: train KL ratio `0.561830x`, train KL reduction `43.82%`, held-out
KL ratio `0.874488x`. This is stable, but it starts almost identical to the
teacher, so it is a stability check rather than a useful training recipe by
itself.

## Diagnosis Matrix

| Case | Result | Interpretation |
|---|---|---|
| A converges + B diverges | no | perturbation is not the primary culprit |
| A diverges + B converges | yes | lr is the primary culprit |
| A and B both diverge | no | H2 algorithm-bug hypothesis not supported |
| C converges + A diverges | yes | lowering lr is required; tiny perturb alone is insufficient |

The SOLID root-cause claim is narrow: the previous catastrophic failure is
licensed to **lr too aggressive at 5e-5**, not to a proven algorithm bug.
The OPD sampled loss still rises in B/C, so the training-step loss scalar is
not a reliable alignment diagnostic for this setup; the teacher-conditioned
forward KL trajectory is the licensing signal.

## Recommendation

Next action: tune lr, not kernels and not OPD semantics.

Run a single-variable lr sweep with `perturb=1e-3`, `rollout_len=8`, same
prompts, and 500 steps:

- `lr = 1e-7, 3e-7, 1e-6, 3e-6, 1e-5`
- license a recipe only if step 200 train KL drops by >= 30%, step 500
  held-out KL is not worse than step 0, and held-out overlap is >= 20%
- kill any lr whose train KL exceeds 2x step 0 by step 25

Do not patch KL direction, AdamW ordering, or KV-cache rollout based on this
tranche. Those remain fallback investigations only if the lr sweep fails.

## Artefacts

- `bench-output/2026-05-21-arle-cuda-opd-convergence-h1-vs-h2/run.txt`
- `bench-output/2026-05-21-arle-cuda-opd-convergence-h1-vs-h2/nvidia-smi-before.txt`
- `bench-output/2026-05-21-arle-cuda-opd-convergence-h1-vs-h2/nvidia-smi-after.txt`
