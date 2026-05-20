# ARLE CUDA Fused AdamW on Qwen3-0.6B OPD

## Goal

Cut the Qwen3-0.6B CUDA OPD `optimizer_step` phase, which dominated the
post-host-mirror profile at 1.036148 s / 79.4 % of step wall-clock.

## Hypothesis

The existing CUDA AdamW was already device-resident, but it still allocated
fresh `param`, `m`, and `v` buffers per tensor and seeded them with three
full device-to-device copies before launching the update kernel. On the first
optimizer step it also initialized moments via host zero vectors, uploading
two full moment tensors per parameter. Updating `param/m/v` in place and
allocating zero moments on device should move optimizer time below 200 ms.

## Command

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_realckpt_profile --release --features cuda \
  2>&1 | tee bench-output/2026-05-21-arle-cuda-fused-adamw-qwen3-06b/run-after-zeros.txt
```

Moderate non-regression:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_moderate_bench --release --features cuda \
  2>&1 | tee bench-output/2026-05-21-arle-cuda-fused-adamw-qwen3-06b/moderate-after-zeros.txt
```

## Environment

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GiB
- Driver: CUDA runtime reported by `nvidia-smi`: 13.2
- Feature set: `--features cuda`
- Model: `/home/ckl/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B`
- Shape: hidden=1024, intermediate=3072, layers=28, vocab=151936,
  heads=16, kv_heads=8, head_dim=128, tied embedding/lm head
- Prompt: `[1, 872, 198, 3456]`
- Rollout length: 8, full rollout length 12
- LR: 5e-5, AdamW betas=(0.9, 0.999), eps=1e-8, wd=0
- Perturbation: uniform 1e-3 on trainable student params

## Results

Final Qwen3-0.6B profile, n=3:

| Run | Step seconds | Optimizer seconds |
|---|---:|---:|
| 1 | 0.291443 | 0.025380 |
| 2 | 0.295275 | 0.025379 |
| 3 | 0.296245 | 0.025465 |
| mean | 0.294321 | 0.025408 |
| median | 0.295275 | 0.025380 |
| sigma / mean | 0.704 % | 0.159 % |

Delta vs `4a631c0` host-mirror fix profile:

| Metric | Before | After mean | Delta |
|---|---:|---:|---:|
| total step seconds | 1.304788 | 0.294321 | -77.44 % |
| optimizer step seconds | 1.036148 | 0.025408 | -97.55 % |
| optimizer share | 79.411 % | 8.634 % | -70.777 pp |
| speedup | 1.00x | 4.43x | +4.43x |

Final phase table from run 1:

| Phase | Seconds | % step |
|---|---:|---:|
| rollout student forward | 0.125247 | 42.975 |
| grad clip | 0.035801 | 12.284 |
| backward | 0.027373 | 9.392 |
| optimizer step | 0.025380 | 8.708 |
| rollout argmax readback | 0.021119 | 7.247 |
| post step cleanup | 0.019152 | 6.572 |
| student forward | 0.017569 | 6.028 |
| teacher forward | 0.016081 | 5.518 |
| KL distill loss | 0.003662 | 1.257 |

Moderate OPD CUDA non-regression:

```text
summary mean_steps_per_sec=14.509078 median_steps_per_sec=14.520130 sigma_steps_per_sec=0.095642 sigma_pct=0.659 mean_step_seconds=0.068925 median_step_seconds=0.068870 max_loss_relative_error_vs_cpu=0.000001276
```

This improves the moderate step from the prior `82.7 ms` reference to
`68.925 ms`, so it stays under the `87 ms` non-regression ceiling.

## Problems

The first in-place-only attempt measured `1.151817 s/step` with
`optimizer_step=0.882442 s`. That missed the acceptance threshold and
identified a second optimizer-local confounder: first-step moments were
created by allocating host zero vectors and uploading them through
`Backend::upload`. The licensed result includes the paired device-zero
allocation fix. This is still one optimizer axis: no OPD step logic,
forward, backward, or loss code changed.

This is not a single all-parameter kernel. It remains one AdamW kernel per
parameter tensor, but each launch now mutates device-resident `param/m/v` in
place and first-step moments are allocated zeroed on device. Since optimizer
time is now 25 ms / 8.7 % of step, pointer-array multi-tensor fusion is no
longer the next binding axis.

## Learnings

For CUDA AdamW, "device-resident" was not enough. Returning fresh handles
forced three full DtoD seed copies per tensor, and first-step moment
initialization hid another two full-tensor host uploads. At Qwen3-0.6B,
removing those optimizer-local memory paths was worth a 4.43x end-to-end
step speedup even without reducing launch count.

The next visible single-variable axis is rollout student forward
(`~0.126 s`, 43 % of the fixed step). A reasonable pre-license target is
step <= 0.22 s by reducing repeated rollout forward work; kill if the axis
stays above 0.27 s with matched n=3 runs.

## Gates

- `cargo check --workspace`: passed
- `cargo test -p autograd --release`: passed
- `cargo test -p train --test test_opd_determinism --release`: passed
- `cargo test -p autograd --test test_cuda_adamw_step --release --features cuda`: passed
- `cargo test -p autograd --test test_cuda_lazy_ops --release --features cuda`: passed
- `cargo run -p train --example opd_step_cuda_moderate_bench --release --features cuda`: passed, mean `0.068925 s/step`, max CPU/CUDA loss relerr `1.276e-6`
- `cargo run -p train --example opd_step_cuda_realckpt_profile --release --features cuda`: passed, mean `0.294321 s/step`

## Artefacts

- Real-checkpoint profile:
  `bench-output/2026-05-21-arle-cuda-fused-adamw-qwen3-06b/run-after-zeros.txt`
- Real-checkpoint repeats:
  `bench-output/2026-05-21-arle-cuda-fused-adamw-qwen3-06b/run-repeat-1.txt`,
  `bench-output/2026-05-21-arle-cuda-fused-adamw-qwen3-06b/run-repeat-2.txt`
- Moderate CUDA OPD:
  `bench-output/2026-05-21-arle-cuda-fused-adamw-qwen3-06b/moderate-after-zeros.txt`
- In-place-only failed intermediate:
  `bench-output/2026-05-21-arle-cuda-fused-adamw-qwen3-06b/run.txt`,
  `bench-output/2026-05-21-arle-cuda-fused-adamw-qwen3-06b/moderate.txt`
- GPU before/after:
  `bench-output/2026-05-21-arle-cuda-fused-adamw-qwen3-06b/nvidia-smi-before-final.txt`,
  `bench-output/2026-05-21-arle-cuda-fused-adamw-qwen3-06b/nvidia-smi-after-final.txt`
