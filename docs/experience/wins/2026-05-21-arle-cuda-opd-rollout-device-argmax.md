# ARLE CUDA OPD Rollout Device Argmax

## Goal

Remove the per-token host logits readback from Qwen3-0.6B OPD rollout.
The pre-license target was `<= 0.198 s/step`; the prior stable profile
after the post-step cleanup KILL measured `0.209 s/step` with
`rollout_argmax_readback = 0.013 s` / 6.1% of the step.

## Hypothesis

For `rollout_len=8` and `vocab=151936`, downloading one logits row and doing
host argmax eight times serializes the decode loop. A CUDA argmax kernel plus
device-token embedding should keep greedy tokens device-resident through the
rollout and leave only one final tiny readback of the generated token buffer.

## Command

Real-checkpoint profile, repeated three times:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_realckpt_profile --release --features cuda \
  | tee bench-output/2026-05-21-arle-cuda-opd-rollout-device-argmax/run.txt
```

Moderate non-regression:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_moderate_bench --release --features cuda \
  | tee bench-output/2026-05-21-arle-cuda-opd-rollout-device-argmax/moderate.txt
```

50-step convergence non-regression:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_realckpt_train --release --features cuda -- \
  --lr 1e-7 --steps 50 \
  | tee bench-output/2026-05-21-arle-cuda-opd-rollout-device-argmax/convergence-50.txt
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

| Run | Step seconds | Rollout forward seconds | Final token readback seconds |
|---|---:|---:|---:|
| 1 | 0.196337 | 0.080741 | 0.001075 |
| 2 | 0.196970 | 0.081115 | 0.001073 |
| 3 | 0.194528 | 0.079995 | 0.001076 |
| mean | 0.195945 | 0.080617 | 0.001075 |
| median | 0.196337 | 0.080741 | 0.001075 |
| sigma / mean | 0.528 % | 0.578 % | 0.116 % |

Delta vs the pre-axis state:

| Metric | Before | After mean | Delta |
|---|---:|---:|---:|
| total step seconds | 0.209000 | 0.195945 | -6.25 % |
| rollout argmax/readback path | 0.013000 | 0.001172 | -90.98 % |
| speedup | 1.00x | 1.07x | +1.07x |

Rollout equivalence probe from the profile harness:

```text
rollout_equivalence host=[1, 872, 198, 3456, 888, 536, 4697, 972, 262, 584, 1099, 737] device=[1, 872, 198, 3456, 888, 536, 4697, 972, 262, 584, 1099, 737] match=true
```

Moderate OPD CUDA non-regression:

```text
summary mean_steps_per_sec=19.510985 median_steps_per_sec=19.499602 sigma_steps_per_sec=0.021206 sigma_pct=0.109 mean_step_seconds=0.051253 median_step_seconds=0.051283 max_loss_relative_error_vs_cpu=0.000001276
```

50-step convergence non-regression at `lr=1e-7`:

| Step | Train overlap % | Held-out overlap % | Train KL | Held-out KL |
|---:|---:|---:|---:|---:|
| 0 | 74.218750 | 50.000000 | 1.433391042756e-2 | 2.172812433186e-2 |
| 50 | 75.000000 | 50.000000 | 1.141541907359e-2 | 2.124995536075e-2 |

This matches the expected step 0-50 behavior from the prior lr=1e-7 run:
train overlap remains stable and train KL improves by 20.36%.

## Problems

The first unconditional implementation regressed the moderate shape to
`0.060241 s/step`. At `rollout_len=2` and `vocab=32768`, the old host
readback is only a small 128 KiB row, while the new path adds argmax,
token-write, final-readback, and device-token embedding launches. The fix is a
matched-control guard: CUDA uses device-resident rollout argmax for the
real target regime (`rollout_len >= 4` or `vocab >= 65536`) and preserves the
legacy host argmax path for tiny moderate-shape control runs.

The device token buffer uses f32 ids because `DeviceHandle` is currently
f32-only. This is exact for current vocab sizes (`151936 << 2^24`) but should
be replaced by an integer device handle if rollout token plumbing expands.

## Learnings

The root cause was real: per-iteration host argmax/readback was a 13 ms
wall-clock sync in the Qwen3-0.6B profile, and moving it device-side cut that
path by about 91%. The win is regime-dependent; for tiny vocab/rollout control
shapes, launch count can dominate over readback bytes.

Next single-variable axis: `post_step_cleanup` device-free churn or rollout
forward launch count. Pre-license for the next axis: Qwen3-0.6B step
`<= 0.18 s` with n=3 and sigma `< 5%`; kill if total stays above `0.195 s`
or if moderate regresses above `56 ms`.

## Gates

- `cargo check -p train --example opd_step_cuda_realckpt_profile --features cuda`: passed
- `cargo check --workspace`: passed
- `cargo build --workspace --release`: passed
- `cargo test -p train --release`: passed
- `cargo test -p autograd --test test_cuda_lazy_ops --release --features cuda -- --nocapture`: passed,
  31/31 tests
- `cargo test -p train --test test_opd_determinism --release`: passed
- `cargo run -p train --example opd_step_cuda_moderate_bench --release --features cuda`: passed,
  mean `0.051253 s/step`, max CPU/CUDA loss relerr `1.276e-6`
- `cargo run -p train --example opd_step_cuda_realckpt_profile --release --features cuda`: passed,
  mean `0.195945 s/step`, sigma `0.528%`
- `cargo run -p train --example opd_step_cuda_realckpt_train --release --features cuda -- --lr 1e-7 --steps 50`: passed,
  step50 train overlap `75.000000%`, train KL `1.141541907359e-2`

## Artefacts

- Real-checkpoint profile:
  `bench-output/2026-05-21-arle-cuda-opd-rollout-device-argmax/run.txt`,
  `bench-output/2026-05-21-arle-cuda-opd-rollout-device-argmax/run-2.txt`,
  `bench-output/2026-05-21-arle-cuda-opd-rollout-device-argmax/run-3.txt`
- Moderate CUDA OPD:
  `bench-output/2026-05-21-arle-cuda-opd-rollout-device-argmax/moderate.txt`
- Convergence non-regression:
  `bench-output/2026-05-21-arle-cuda-opd-rollout-device-argmax/convergence-50.txt`
