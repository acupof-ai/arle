# P3.4 DSv4 MHC Params Persistent Kernel KILL

## Context

Kernel: `dsv4_mhc_params_kernel` in
`crates/cuda-kernels/csrc/misc/dsv4_mhc.cu`.

Scope: P3.4 A6 evaluated whether launch reduction or a persistent-kernel shape
is justified for the local MHC params path.

## Root Cause

Hypothesis: because t1 and t64 component latency are both about 42 us, launch
overhead might be a separable binding cost.

nsys falsified that as the primary local root cause. The kernel body itself is
the dominant median cost; launch is measurable but not binding enough to justify
persistent-worker complexity for this tranche.

## Evidence

Steady nsys command:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --force-overwrite=true \
  -o /tmp/p3_4_a6_dsv4_mhc_params_t1_steady \
  target/release/deps/ops_bench-d80e79bd3e0cee50 \
  --bench ops_cuda/dsv4_mhc_params/dsv4_mini_decode_t1_h4096_m24_hc4 \
  --exact --sample-size 10 --noplot --discard-baseline
```

Summary:

| Metric | Value |
|---|---:|
| Kernel launches | `22436` |
| Criterion under nsys point | `43.571 us` |
| `cudaLaunchKernel` avg / median | `3.3625 us` / `3.2700 us` |
| `cuStreamSynchronize` calls | `44918` |
| `cuStreamSynchronize` avg / median | `24.3234 us` / `3.9950 us` |
| Kernel avg / median | `45.7936 us` / `36.1630 us` |
| Kernel launch+queue+kernel avg / median | `50.6728 us` / `40.8690 us` |

Kernel median is about 11x launch median. Persistent launch reduction is not
the first-order fix for this kernel.

## Fix

No runtime patch was made. Keep A6 closed for P3.4.

## Rule

Do not open a persistent-worker path for MHC params from the current local
evidence. Optimize the kernel body first; launch reduction would be a separate
runtime/harness design, not an in-kernel memory-access axis.
