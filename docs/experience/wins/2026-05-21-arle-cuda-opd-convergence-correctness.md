# ARLE CUDA OPD Convergence Correctness — component bench, 2026-05-21

## Goal

- **Diagnosis / correctness.** Verify that the CUDA OPD step is not only CPU-loss equivalent, but also deterministic, cross-backend rollout aligned, and capable of reducing student/teacher KL on a perturbed student.

## Hypothesis

- CUDA repeat runs with the same prompt, seed, and learning rate should be bit-identical for the exercised moderate shape.
- CPU and CUDA should produce the same greedy rollout tokens for the first 10 OPD steps; any divergence would be documented as CUDA math nondeterminism.
- A perturbed student should reduce KL over 50-500 steps, but exact greedy-decode overlap may remain weak with `rollout_len=2`, one training prompt, and scratch random weights.

## Command

```bash
mkdir -p bench-output/2026-05-21-arle-cuda-opd-convergence

NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_convergence_bench --release --features cuda \
  | tee bench-output/2026-05-21-arle-cuda-opd-convergence/run.txt

cargo test -p train --test test_opd_determinism --release \
  | tee bench-output/2026-05-21-arle-cuda-opd-convergence/test_opd_determinism_cpu.txt

NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test -p train --test test_opd_determinism --release --features cuda \
  | tee bench-output/2026-05-21-arle-cuda-opd-convergence/test_opd_determinism_cuda_feature.txt

cargo test -p train --test test_qwen35_loader --release -- --nocapture \
  | tee bench-output/2026-05-21-arle-cuda-opd-convergence/test_qwen35_loader.txt
```

## Environment

- **Backend:** CUDA component bench through `autograd::backend_cuda::CudaBackend`.
- **Model:** moderate Qwen3.5-like scratch config, `hidden=512`, `intermediate=1536`, `layers=12`, `vocab=32768`, `heads=8`, `kv_heads=4`, `head_dim=64`.
- **Hardware:** NVIDIA GeForce RTX 4070 Ti SUPER, 16,376 MiB VRAM, 14,989 MiB free before run.
- **Driver / CUDA:** NVIDIA driver `595.71.05`; PyTorch reference env reports `torch 2.11.0+cu130`, CUDA `13.0`.
- **Commit:** parent `e7ca73d` plus the new convergence bench and this report.
- **Feature set:** `cargo run -p train --example opd_step_cuda_convergence_bench --release --features cuda`.
- **Non-default env:** `NVCC_CCBIN=/usr/bin/g++-14`, `INFER_TILELANG_PYTHON=$PWD/.venv/bin/python`, `CUDARC_CUDA_VERSION=13010`, `TORCH_CUDA_ARCH_LIST=8.9`.

## Results

### Determinism And Cross-Backend Correctness

| Check | Result |
|---|---:|
| Existing CPU determinism test | pass |
| Existing determinism test under `--features cuda` | pass |
| CUDA repeat, 10 steps, loss bit-identical | true |
| CUDA repeat, 10 steps, rollout identical | true |
| CUDA repeat max abs loss diff | `0.000000000000e0` |
| CPU vs CUDA rollout match, 10 steps | `10/10` |
| CPU vs CUDA max loss relerr, 10 steps | `2.154206928271e-6` |

### 500-Step CUDA KL Trajectory

| Step | Loss | Δ vs step 1 |
|---:|---:|---:|
| 1 | `3.270824381616e-4` | baseline |
| 50 | `3.158988838550e-4` | `-3.4207%` |
| 100 | `3.152190183755e-4` | `-3.6270%` |
| 500 | `3.146462549921e-4` | `-3.8022%` |

The 500-step run took `49.335591s`, or `0.098671182s/step`, consistent with the prior CUDA moderate-step wall-clock baseline.

### Greedy Decode Teacher Overlap

The decode probe uses three fixed prompts and 8 generated tokens per prompt.

| Step | Mean exact-position overlap |
|---:|---:|
| 0 | `0.000%` |
| 50 | `0.000%` |
| 100 | `0.000%` |
| 500 | `8.333%` |

Only the training prompt showed visible exact-token convergence by step 500: prompt 0 reached `25.000%` overlap (`[30806, 4126, ...]` matched the teacher's first two generated tokens). The two held-out prompts stayed at `0.000%`.

### Real Checkpoint Probe

The local ModelScope `Qwen/Qwen3-0.6B` checkpoint is present and the existing loader smoke passes:

| Check | Result |
|---|---:|
| Path | `/home/ckl/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B` |
| `load_qwen35_from_hf_dir` smoke | pass |
| Loaded parameter ids | `312` |
| Forward logits finite | true |

This run did not train a full Qwen3-0.6B student. The moderate shape remains the memory-safe exercised substrate; full trainable 0.6B OPD needs a separate memory-budgeted run because it adds two FP32 model copies, gradients, AdamW moments, and temporary activations on a 16 GB GPU.

## Problems

- This is a **correctness/convergence confirmation**, not a real model quality win. KL falls, but only `3.80%` after 500 steps and exact greedy decode overlap remains weak.
- Eval improvement is not confirmed for a real checkpoint. The real Qwen3-0.6B loader works, but full trainable OPD was not run in this tranche to avoid mixing correctness verification with a higher-memory experiment.
- `rollout_len=2` plus one repeated training prompt is likely too little supervision to show robust held-out decode convergence. The held-out prompt overlap staying at `0%` is the key limitation.

## Learnings

- The CUDA OPD substrate is numerically stable for this exercised path: deterministic repeat, CPU/CUDA rollout alignment, and CPU/CUDA loss relerr all pass.
- Loss improvement is real but shallow under the current smoke-style task; decode behavior needs longer rollout, more prompts, and probably a LoRA-only or checkpoint-backed student run before calling OPD useful for model quality.
- Future OPD eval should separate two axes: keep the moderate scratch bench as a fast correctness guard, then run a memory-budgeted real-checkpoint eval with a fixed prompt set and token-overlap/rank metrics.

## Delta Vs Baseline

- **Baseline:** `docs/experience/wins/2026-05-20-arle-cuda-opd-moderate-first-run.md` / commit `e7ca73d` established `99.18 ms/step` CUDA moderate OPD performance and CPU loss relerr `1.276e-6`.

| Metric | Baseline | Now | Δ |
|---|---:|---:|---:|
| CUDA moderate step wall-clock | `99.18 ms` | `98.67 ms` | `-0.51%` |
| CPU/CUDA max loss relerr | `1.276e-6` | `2.154e-6` | still `<5e-5` |
| CUDA repeat determinism | not measured | bit-identical | new guard |
| CPU/CUDA rollout match | not measured | `10/10` | new guard |
| KL loss over long run | not measured | `-3.80%` over 500 steps | new evidence |
| Mean decode overlap | not measured | `8.33%` at step 500 | weak / insufficient |

## Artefacts

- Raw convergence output: `bench-output/2026-05-21-arle-cuda-opd-convergence/run.txt`
- CPU determinism output: `bench-output/2026-05-21-arle-cuda-opd-convergence/test_opd_determinism_cpu.txt`
- CUDA-feature determinism output: `bench-output/2026-05-21-arle-cuda-opd-convergence/test_opd_determinism_cuda_feature.txt`
- Real checkpoint loader smoke: `bench-output/2026-05-21-arle-cuda-opd-convergence/test_qwen35_loader.txt`

## Notes

- What changed in code: added `crates/train/examples/opd_step_cuda_convergence_bench.rs`.
- Bench scope: component-level OPD correctness/eval probe, not a guidellm serving run.
- Follow-up: promote real Qwen3-0.6B OPD eval only after a memory budget is explicit; prefer LoRA-only student or longer rollout/multi-prompt setup to avoid full AdamW state pressure.
