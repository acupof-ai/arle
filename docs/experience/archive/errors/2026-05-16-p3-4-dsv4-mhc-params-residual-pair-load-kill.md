# P3.4 DSv4 MHC Params Residual Pair Load KILL

## Context

Kernel: `dsv4_mhc_params_kernel` in
`crates/cuda-kernels/csrc/misc/dsv4_mhc.cu`.

Scope: P3.4 A2 tested vectorized residual loads for the RMS sumsq loop. The
local 1B shape uses residual dim 4096, mix dim 24, `hc_mult=4`, and 20 Sinkhorn
iterations.

## Root Cause

Hypothesis: loading two BF16 residual values via `uint32_t` could reduce load
loop instructions enough to move the MHC params kernel.

Formula check predicted limited upside: each token reads only 4096 BF16 values
(8 KiB), which is about 0.012 us at 672 GB/s before cache effects. The 42 us
observed kernel time is therefore unlikely to be HBM-load dominated.

## Evidence

Baseline:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_mhc_params --save-baseline p3_4_a2_before
```

Treatment used `uint32_t` pair-loads for even `residual_hidden_dim` and kept a
scalar fallback for odd dims.

Results:

| Shape | Baseline | Treatment | Change | p | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| decode t1/h4096/m24/hc4 | 42.278 us | 41.446 us | -1.9223% | 0.00 | KILL |
| batch t64/h4096/m24/hc4 | 42.276 us | 41.576 us | -1.6454% | 0.00 | KILL |

The effect is real but below the >=3% license gate and below the 2-3% review
band.

## Fix

Treatment reverted. Keep the scalar residual load loop.

## Rule

Do not land residual pair-load for MHC params on the local 1B shape. It is a
sub-2% improvement with extra odd-dim fallback complexity and does not meet the
Phase 3 license threshold.
