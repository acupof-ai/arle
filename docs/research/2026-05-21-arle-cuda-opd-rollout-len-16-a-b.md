# ARLE CUDA OPD Rollout Length 16 A/B

## Goal

Test whether the held-out exact-token overlap plateau from the `lr=1e-7`
real-checkpoint OPD run is caused by insufficient rollout supervision. The
matched control is the `rollout_len=8` run from
[`../experience/wins/2026-05-21-arle-cuda-opd-realckpt-convergence-2k-steps.md`](../experience/wins/2026-05-21-arle-cuda-opd-realckpt-convergence-2k-steps.md).

Verdict: **HELD-OUT FLAT at 64.062500%**.

Increasing rollout length from 8 to 16 improves KL slightly at the same 500
step budget, but it does not break the held-out exact-overlap plateau. For
this prompt set, rollout length is not the binding constraint for held-out
decode-exact overlap; prompt-set widening is the next axis.

## Command

```bash
mkdir -p bench-output/2026-05-21-arle-cuda-opd-rollout-len-16-a-b
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu --format=csv \
  > bench-output/2026-05-21-arle-cuda-opd-rollout-len-16-a-b/nvidia-smi-before.txt

NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_realckpt_train --release --features cuda -- \
  --lr 1e-7 --steps 500 --rollout-len 16 --eval-steps 0,50,100,250,500 2>&1 \
  | tee bench-output/2026-05-21-arle-cuda-opd-rollout-len-16-a-b/run.txt

nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu --format=csv \
  > bench-output/2026-05-21-arle-cuda-opd-rollout-len-16-a-b/nvidia-smi-after.txt
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
- Rollout A/B variable: `rollout_len=16` vs prior `rollout_len=8`
- Decode eval: greedy 16-token suffix
- Prompt set: same 8 training prompts and 4 held-out prompts as the prior
  `lr=1e-7` runs

GPU memory snapshots before and after the process both reported `955 MiB`
used, because snapshots were outside the benchmark process lifetime.

## Harness Change

`crates/train/examples/opd_step_cuda_realckpt_train.rs` now accepts:

- `--rollout-len VALUE`
- `--eval-steps CSV`

Defaults remain compatible with the previous 2k run. This tranche used the
explicit `--rollout-len 16 --eval-steps 0,50,100,250,500` flags so the A/B
changes only the rollout length.

## Safety

The run completed without OOM, crash, or NaN.

| Metric | Result |
|---|---:|
| first step seconds | 0.234477 |
| safety ceiling | 0.600000 |
| mean OPD step seconds | 0.265632 |
| median OPD step seconds | 0.265965 |
| total loop wall seconds | 149.836579 |

First-five trace:

| Step | Loss | Rollout len | Step seconds |
|---:|---:|---:|---:|
| 1 | 1.467865604354e-5 | 20 | 0.234477 |
| 2 | 2.990225402755e-5 | 20 | 0.269820 |
| 3 | 2.505216252757e-5 | 21 | 0.265732 |
| 4 | 3.993470454589e-5 | 21 | 0.265921 |
| 5 | 2.288054565724e-5 | 22 | 0.266108 |

## Results

Rollout length 16 trajectory:

| Step | Train overlap % | Held-out overlap % | Train KL | Held-out KL |
|---:|---:|---:|---:|---:|
| 0 | 74.218750 | 50.000000 | 1.433391e-2 | 2.172812e-2 |
| 50 | 73.437500 | 50.000000 | 1.040553e-2 | 2.116957e-2 |
| 100 | 73.437500 | 64.062500 | 8.877711e-3 | 2.067949e-2 |
| 250 | 71.093750 | 64.062500 | 6.928254e-3 | 1.974962e-2 |
| 500 | 78.906250 | 64.062500 | 5.631429e-3 | 1.913682e-2 |

Matched A/B against rollout length 8:

| Step | Metric | Rollout 8 | Rollout 16 | Delta |
|---:|---|---:|---:|---:|
| 50 | train overlap % | 75.000000 | 73.437500 | -1.562500 pp |
| 50 | held-out overlap % | 50.000000 | 50.000000 | 0.000000 pp |
| 50 | train KL | 1.141542e-2 | 1.040553e-2 | -8.85% |
| 50 | held-out KL | 2.124996e-2 | 2.116957e-2 | -0.38% |
| 100 | train overlap % | 75.000000 | 73.437500 | -1.562500 pp |
| 100 | held-out overlap % | 64.062500 | 64.062500 | 0.000000 pp |
| 100 | train KL | 9.946998e-3 | 8.877711e-3 | -10.75% |
| 100 | held-out KL | 2.086548e-2 | 2.067949e-2 | -0.89% |
| 250 | train overlap % | 81.250000 | 71.093750 | -10.156250 pp |
| 250 | held-out overlap % | 64.062500 | 64.062500 | 0.000000 pp |
| 250 | train KL | 7.980873e-3 | 6.928254e-3 | -13.19% |
| 250 | held-out KL | 2.003209e-2 | 1.974962e-2 | -1.41% |
| 500 | train overlap % | 78.906250 | 78.906250 | 0.000000 pp |
| 500 | held-out overlap % | 64.062500 | 64.062500 | 0.000000 pp |
| 500 | train KL | 6.851923e-3 | 5.631429e-3 | -17.81% |
| 500 | held-out KL | 1.938898e-2 | 1.913682e-2 | -1.30% |

Rollout length 8 comparison points come from:

- step 50/100/500:
  [`../experience/wins/2026-05-21-arle-cuda-opd-realckpt-convergence-lr1e-7.md`](../experience/wins/2026-05-21-arle-cuda-opd-realckpt-convergence-lr1e-7.md)
- step 250:
  [`../experience/wins/2026-05-21-arle-cuda-opd-realckpt-convergence-2k-steps.md`](../experience/wins/2026-05-21-arle-cuda-opd-realckpt-convergence-2k-steps.md)

## Interpretation

Rollout length 16 gives a stronger per-step KL signal:

- train KL is lower at every shared eval point;
- held-out KL is also lower at every shared eval point, though only by about
  0.4-1.4%;
- sampled rollout loss is not comparable across rollout lengths because the
  sampled sequence length and rollout tokens differ.

It does **not** improve the target decode-exact metric:

- held-out overlap reaches the same `64.062500%` ceiling by step 100;
- it stays there at step 250 and step 500;
- training exact overlap is worse at step 250 and tied by step 500.

The held-out plateau is therefore not explained by too-short rollout under
this 8-prompt setup. Longer rollout may help KL, but it is not enough to
unlock new held-out greedy tokens.

## Next Axis

Widen the prompt set before increasing rollout length further:

- keep `lr=1e-7`, `perturb=1e-3`, AdamW settings, and model checkpoint fixed;
- use at least 64 training token-id prompts and a larger held-out split;
- start with `rollout_len=8` for cost control, then repeat `rollout_len=16`
  only if the wider prompt set moves held-out overlap;
- license if held-out exact overlap exceeds `70%` by step 500 or held-out KL
  drops at least `20%` by step 500;
- kill if held-out overlap remains `64.062500%` and held-out KL improves less
  than `5%` versus the current 8-prompt control.

## Verification

- `cargo check -p train --example opd_step_cuda_realckpt_train --features cuda`:
  passed after adding the flags.
- 500-step CUDA run with `--rollout-len 16`: passed, no OOM / no NaN.

## Artefacts

- `bench-output/2026-05-21-arle-cuda-opd-rollout-len-16-a-b/run.txt`
- `bench-output/2026-05-21-arle-cuda-opd-rollout-len-16-a-b/nvidia-smi-before.txt`
- `bench-output/2026-05-21-arle-cuda-opd-rollout-len-16-a-b/nvidia-smi-after.txt`
