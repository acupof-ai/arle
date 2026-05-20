# ARLE CUDA Host Mirror Fix on Qwen3-0.6B OPD

## Goal

Fix the `TensorStore::ensure_device` host-mirror retention bug that made the
real-checkpoint CUDA OPD profile spend wall-clock time cloning host buffers on
the device-resident fast path.

## Hypothesis

The 10.411479 s Qwen3-0.6B profile was not compute-bound. The previous
host-mirror control cleared 4547 MiB of host mirrors and dropped the same
profile to 1.266622 s, so moving the `ensure_device` early return before the
host `Vec<f32>` clone should reproduce that result without changing OPD math.

## Command

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_realckpt_profile --release --features cuda \
  2>&1 | tee bench-output/2026-05-21-arle-cuda-opd-host-mirror-fix/run.txt
```

GPU memory sample run:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
target/release/examples/opd_step_cuda_realckpt_profile \
  2>&1 | tee bench-output/2026-05-21-arle-cuda-opd-host-mirror-fix/run-sampled.txt
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

Headline run:

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| total step seconds | 10.411479 | 1.304788 | -87.47 % |
| speedup | 1.00x | 7.98x | +7.98x |
| rollout student forward | 6.521386 | 0.126434 | -98.06 % |
| teacher forward | 0.802192 | 0.016064 | -98.00 % |
| student forward | 0.790930 | 0.017712 | -97.76 % |
| optimizer step | 1.696729 | 1.036148 | -38.94 % |

Sampled confirmation run:

| Metric | Value |
|---|---:|
| total step seconds | 1.274769 |
| speedup vs 10.411479 s baseline | 8.17x |
| gap vs 1.266622 s host-mirror control | +0.64 % |
| GPU memory used min | 955 MiB |
| GPU memory used max | 15070 MiB |
| GPU memory used delta | 14115 MiB |
| after-process GPU memory used | 955 MiB |

Phase attribution after the fix:

| Phase | Seconds | % step |
|---|---:|---:|
| optimizer step | 1.036148 | 79.411 |
| rollout student forward | 0.126434 | 9.690 |
| grad clip | 0.035785 | 2.743 |
| backward | 0.027340 | 2.095 |
| rollout argmax readback | 0.021852 | 1.675 |
| post step cleanup | 0.019621 | 1.504 |
| student forward | 0.017712 | 1.357 |
| teacher forward | 0.016064 | 1.231 |
| KL distill loss | 0.003773 | 0.289 |

Moderate CUDA OPD also improves because it no longer clones clean host
mirrors before every device use:

```text
summary mean_steps_per_sec=12.092292 median_steps_per_sec=12.104868 sigma_steps_per_sec=0.029048 sigma_pct=0.240 mean_step_seconds=0.082698 median_step_seconds=0.082611 max_loss_relative_error_vs_cpu=0.000001276
```

## Problems

This fix preserves `Dirty::Both` host mirrors intentionally. Some CPU fallback
backward paths still read host data directly after tensors have been uploaded,
so clearing every host mirror on upload would be a semantics change. The
licensed single-variable fix is narrower: if a tensor already has a usable
device handle and is not `Dirty::Host`, `ensure_device` now returns before
cloning its host buffer.

The requested full real-checkpoint CPU-vs-CUDA relerr was not rerun in this
tranche because a full CPU real-checkpoint OPD step is not part of the
existing profile harness. The CUDA real-checkpoint loss stayed exactly at the
pre-fix CPU-validated value (`1.788745430531e-5`), and the moderate OPD
CPU/CUDA correctness gate stayed at `1.276e-6` max relerr. Treat the unchanged
real-checkpoint loss as a consistency check, not a fresh CPU/CUDA measurement.

## Learnings

`Dirty::Both` is not itself the bug. It is a valid state for CPU fallback
semantics, but every clean-device fast path must test residency before touching
the host mirror. At production vocab size, a discarded host `Vec` clone can
dominate wall-clock even when the CUDA kernels are already device-resident.

The next bottleneck is now optimizer step, not attention or backward:
`optimizer_step` is 79.4 % of the fixed real-checkpoint step. The next
single-variable axis should target AdamW parameter update wall-clock, with a
license threshold of real-checkpoint step <= 0.90 s and a kill threshold of
> 1.20 s.

## Gates

- `cargo test -p autograd --release`: passed
- `cargo test -p train --test test_opd_determinism --release`: passed
- `cargo test -p autograd --test test_cuda_lazy_ops --release --features cuda`: passed
- `cargo run -p train --example opd_step_cuda_realckpt_profile --release --features cuda`: passed, 1.304788 s
- `cargo run -p train --example opd_step_cuda_moderate_bench --release --features cuda`: passed, max CPU/CUDA loss relerr `1.276e-6`

## Artefacts

- Real-checkpoint profile:
  `bench-output/2026-05-21-arle-cuda-opd-host-mirror-fix/run.txt`
- Sampled confirmation profile:
  `bench-output/2026-05-21-arle-cuda-opd-host-mirror-fix/run-sampled.txt`
- Moderate correctness/bench:
  `bench-output/2026-05-21-arle-cuda-opd-host-mirror-fix/moderate-correctness.txt`
- GPU before/after:
  `bench-output/2026-05-21-arle-cuda-opd-host-mirror-fix/nvidia-smi-before.txt`,
  `bench-output/2026-05-21-arle-cuda-opd-host-mirror-fix/nvidia-smi-after.txt`
- GPU memory samples:
  `bench-output/2026-05-21-arle-cuda-opd-host-mirror-fix/nvidia-smi-samples.csv`,
  `bench-output/2026-05-21-arle-cuda-opd-host-mirror-fix/nvidia-smi-peak.txt`
