# ARLE CUDA OPD Moderate Step First Run

## Goal

Port the ARLE OPD moderate-step harness onto CUDA and set the first
runtime-native target against the PyTorch CUDA baseline from `b1c53cc`.

## Hypothesis

The missing CUDA paths for projection-with-transposed-weight, RMSNorm,
attention softmax backward, layout ops, embedding, and device-backed AdamW
were forcing host round-trips. Keeping the OPD step device-resident should
move the moderate shape under the `200 ms/step` target.

## Params

- Backend: ARLE autograd CUDA
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER
- Shape: hidden=512, intermediate=1536, layers=12, vocab=32768
- Attention: heads=8, kv_heads=4, head_dim=64, gated q_proj, GQA, RoPE
- Prompt: `[1, 3, 8]`
- Rollout length: 2
- Optimizer: AdamW, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0
- Runs: 1 warmup, 3 measured, 10 OPD steps per measured run
- Env: `NVCC_CCBIN=/usr/bin/g++-14`,
  `INFER_TILELANG_PYTHON=$PWD/.venv/bin/python`,
  `CUDARC_CUDA_VERSION=13010`, `TORCH_CUDA_ARCH_LIST=8.9`

Command:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_moderate_bench --release --features cuda \
  | tee bench-output/2026-05-20-arle-cuda-opd-moderate/run.txt
```

## Results

```text
run=1 wall_seconds=0.973134 per_step_seconds=0.097313 steps_per_sec=10.276076 first_loss=0.000314202 last_loss=0.000315440
run=2 wall_seconds=1.024238 per_step_seconds=0.102424 steps_per_sec=9.763353 first_loss=0.000314202 last_loss=0.000315440
run=3 wall_seconds=0.978071 per_step_seconds=0.097807 steps_per_sec=10.224208 first_loss=0.000314202 last_loss=0.000315440
summary mean_steps_per_sec=10.087879 median_steps_per_sec=10.224208 sigma_steps_per_sec=0.230449 sigma_pct=2.284 mean_step_seconds=0.099181 median_step_seconds=0.097807 max_loss_relative_error_vs_cpu=0.000001276
```

| Metric | Value |
|---|---:|
| mean step seconds | 0.099181 |
| median step seconds | 0.097807 |
| sigma / mean | 2.284% |
| max relative loss error vs CPU, 3 steps | 0.000001276 |
| ratio vs PyTorch CUDA 0.083179s/step | 1.19x |
| speedup vs ARLE reference CPU 0.83s/step | 8.37x |
| speedup vs local CPU gate 0.972428s/step | 9.80x |

## Problems

The first runnable CUDA path was not licensed: it measured `0.704146s/step`,
above the `500 ms/step` stop-and-diagnose threshold. `nsys` attributed the
failure to host/device copies rather than GEMM time: host AdamW, device
AdamW `zero_grad`, attention `softmax_backward`, and initial embedding
dispatch each left large gradients or weights on host. The committed path
removes the AdamW host fallback, adds CUDA softmax backward, and makes
CUDA embedding device-resident from the first step. A review pass also found
that host-only gradient clipping would silently skip `Dirty::Device` grads;
the final run includes device sumsq reduction and device grad scaling.

The current `causal_sdpa` remains the temporary matmul-decomposed path:
reshape, transpose, matmul, scale, causal mask add, softmax, matmul. This
gets the OPD step runnable and below target, but a fused causal-SDPA CUDA
kernel remains deferred.

The local CPU non-regression run was `0.972428s/step`, not the historical
`0.83s/step` comparison point. The determinism test stayed bit-identical
and the touched CPU dispatch is CUDA-gated, so this is recorded as host
load / current-machine framing rather than a CPU optimization claim.

## Learnings

For this OPD shape, CUDA becomes competitive as soon as the full backward
and optimizer chain stays device-resident. Small host-only attention
softmax backward looked harmless by tensor size, but it demoted every
upstream projection grad and dominated wall-clock through AdamW copies.
Wall-clock plus residency checks were the right framing; per-kernel math
was not the blocker.

## Gates

- `cargo test -p autograd --release --features cuda`: passed
- `cargo test -p train --test test_opd_determinism --release`: passed
- `cargo run -p train --example opd_step_cpu_moderate_bench --release`:
  passed functionally, current mean `0.972428s/step`
- `cargo run -p train --example opd_step_cuda_moderate_bench --release --features cuda`:
  passed, mean `0.099181s/step`

## Artefacts

- Raw run: `bench-output/2026-05-20-arle-cuda-opd-moderate/run.txt`
- JSON: `bench-output/2026-05-20-arle-cuda-opd-moderate/results.json`
- GPU env: `bench-output/2026-05-20-arle-cuda-opd-moderate/nvidia-smi.txt`
- nsys diagnosis summary:
  `bench-output/2026-05-20-arle-cuda-opd-moderate/nsys-diagnosis-summary.txt`
