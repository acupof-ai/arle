# ARLE CUDA OPD Fused Grad Clip

## Goal

Reduce the Qwen3-0.6B CUDA OPD `grad_clip` phase. The pre-license target was
`<= 0.225 s/step`; the previous stable profile measured `0.253540 s/step`
with `grad_clip = 0.036084 s` / 14.2% of the step.

## Hypothesis

The old clip path computed norm and scaling through per-tensor host-visible
work. A backend CUDA path that reduces all device-resident gradients through
one batched pointer-array kernel, then scales all gradients through one batched
kernel, should cut clip time below 12 ms and move total step into the
`0.225-0.245 s` license-with-investigation band.

## Command

Real-checkpoint profile, repeated three times:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_realckpt_profile --release --features cuda \
  | tee bench-output/2026-05-21-arle-cuda-opd-fused-grad-clip/realckpt-profile-run.txt
```

Moderate non-regression:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_moderate_bench --release --features cuda \
  | tee bench-output/2026-05-21-arle-cuda-opd-fused-grad-clip/moderate-run.txt
```

50-step convergence non-regression:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_realckpt_train --release --features cuda -- \
  --lr 1e-7 --steps 50 \
  | tee bench-output/2026-05-21-arle-cuda-opd-fused-grad-clip/convergence-50-lr1e-7.txt
```

## Environment

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GiB
- Driver/runtime: CUDA 13.x path, `--features cuda`
- Model: `/home/ckl/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B`
- Shape: hidden=1024, intermediate=3072, layers=28, vocab=151936,
  heads=16, kv_heads=8, head_dim=128, tied embeddings
- Real profile prompt: `[1, 872, 198, 3456]`
- Rollout length: 8, final rollout length 12
- LR: 5e-5 for profile, 1e-7 for convergence non-regression
- AdamW betas=(0.9, 0.999), eps=1e-8, wd=0

## Results

Qwen3-0.6B profile, n=3:

| Run | Step seconds | Grad clip seconds |
|---|---:|---:|
| 1 | 0.231168 | 0.013385 |
| 2 | 0.232474 | 0.013525 |
| 3 | 0.231453 | 0.013397 |
| mean | 0.231698 | 0.013436 |
| median | 0.231453 | 0.013397 |
| sigma / mean | 0.242 % | 0.477 % |

Delta vs `2026-05-21-arle-cuda-opd-rollout-kvcache.md`:

| Metric | Before | After mean | Delta |
|---|---:|---:|---:|
| total step seconds | 0.253540 | 0.231698 | -8.62 % |
| grad clip seconds | 0.036084 | 0.013436 | -62.76 % |
| grad clip share | 14.232 % | 5.798 % | -8.434 pp |
| speedup | 1.00x | 1.09x | +1.09x |

Moderate OPD CUDA non-regression:

```text
summary mean_steps_per_sec=18.542534 median_steps_per_sec=18.506113 sigma_steps_per_sec=0.127407 sigma_pct=0.687 mean_step_seconds=0.053930 median_step_seconds=0.054037 max_loss_relative_error_vs_cpu=0.000001276
```

This passes the `<= 60 ms` moderate ceiling and the `<= 1e-4` CPU/CUDA loss
relative-error gate.

50-step convergence non-regression at `lr=1e-7`:

| Step | Train overlap % | Held-out overlap % | Train KL | Held-out KL |
|---:|---:|---:|---:|---:|
| 0 | 74.218750 | 50.000000 | 1.433391042756e-2 | 2.172812433186e-2 |
| 50 | 75.000000 | 50.000000 | 1.141541907359e-2 | 2.124995536075e-2 |

This matches the prior 5939cc7 step 0-50 trajectory: train overlap remains
stable and train KL improves by 20.36%.

## Problems

This is a license-with-investigation result, not a full license. The mean
`0.231698 s/step` lands inside the pre-licensed `0.225-0.245 s` band, but it
does not meet the stronger `<= 0.225 s` target. The residual wall-clock is now
rollout forward, backward, optimizer, cleanup, and the two final full forwards;
grad clip is no longer a top-3 phase.

The implementation still performs one host readback of per-chunk partial
`f64` sums to compute the global scale. This keeps the scalar decision simple
and preserves CPU fallback semantics, but a fully device-side scalar path could
trim another small sync if it becomes visible in nsys.

## Learnings

For OPD's 600M-param full-finetune case, global grad clipping was memory and
dispatch bound enough that a batched all-gradient reduction plus batched
all-gradient scaling cut the phase by 62.76%. Once reduced to 13.4 ms, further
grad-clip work is not the next binding axis.

Next single-variable axis: reduce `rollout_student_forward` launch count or
post-step cleanup. Pre-license for the next axis: Qwen3-0.6B step `<= 0.205 s`
with n=3 and sigma `< 5%`; kill if the axis stays above `0.225 s` wall-clock
or regresses moderate above `60 ms`.

## Gates

- `cargo check --workspace`: passed
- `cargo test -p train --test test_opd_determinism --release`: passed
- `cargo test -p autograd --test test_cuda_lazy_ops --release --features cuda`: passed
- `cargo test -p train --test test_grad_clip --release --features cuda`: passed
- `cargo run -p train --example opd_step_cuda_moderate_bench --release --features cuda`: passed,
  mean `0.053930 s/step`, max CPU/CUDA loss relerr `1.276e-6`
- `cargo run -p train --example opd_step_cuda_realckpt_profile --release --features cuda`: passed,
  mean `0.231698 s/step`
- `cargo run -p train --example opd_step_cuda_realckpt_train --release --features cuda -- --lr 1e-7 --steps 50`: passed,
  step50 train overlap `75.000000%`, train KL `1.141541907359e-2`

## Artefacts

- Real-checkpoint profile:
  `bench-output/2026-05-21-arle-cuda-opd-fused-grad-clip/realckpt-profile-run.txt`,
  `bench-output/2026-05-21-arle-cuda-opd-fused-grad-clip/realckpt-profile-run-2.txt`,
  `bench-output/2026-05-21-arle-cuda-opd-fused-grad-clip/realckpt-profile-run-3.txt`
- Moderate CUDA OPD:
  `bench-output/2026-05-21-arle-cuda-opd-fused-grad-clip/moderate-run.txt`
- Convergence non-regression:
  `bench-output/2026-05-21-arle-cuda-opd-fused-grad-clip/convergence-50-lr1e-7.txt`
- GPU before/after:
  `bench-output/2026-05-21-arle-cuda-opd-fused-grad-clip/nvidia-smi-before.txt`,
  `bench-output/2026-05-21-arle-cuda-opd-fused-grad-clip/nvidia-smi-after.txt`
