# P3.4 DSv4 MHC Params Launch Shape REVIEW

## Context

Kernel: `dsv4_mhc_params_kernel` in
`crates/cuda-kernels/csrc/misc/dsv4_mhc.cu`.

Scope: P3.4 A5 swept `DSV4_MHC_BLOCK`, which controls the residual RMS
reduction block size for MHC params. The default is 256.

## Root Cause

Hypothesis: changing block size may improve the residual sumsq reduction for
the local 1B h4096/m24/hc4 shape.

Formula prediction was mixed:

- Smaller blocks reduce CTA resources but increase per-thread residual work.
- Larger blocks reduce per-thread residual work but increase CTA resources and
  affect all MHC kernels using the shared `DSV4_MHC_BLOCK` macro.
- Only the RMS reduction is affected; the thread0 hc4 Sinkhorn loop is not.

## Evidence

Baseline:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_mhc_params --save-baseline p3_4_a5_before
```

Results:

| Shape | Baseline | block=128 | block=128 change | block=512 | block=512 change | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| decode t1/h4096/m24/hc4 | 42.278 us | 44.208 us | +4.5321%, p=0.00 | 41.378 us | -2.1474%, p=0.00 | REVIEW |
| batch t64/h4096/m24/hc4 | 42.333 us | 44.218 us | +4.4156%, p=0.00 | 41.436 us | -2.1211%, p=0.00 | REVIEW |

Block=128 is a clear regression. Block=512 is statistically significant but
below the >=3% license gate, so it is not shipped. It sits in the 2-3% REVIEW
band.

## Fix

Treatment reverted. Keep `DSV4_MHC_BLOCK == 256`.

## Tradeoff

- LOC complexity: low if changing the macro globally, higher if splitting a
  params-only block macro.
- SM89 specificity: evidence is local to RTX 4070 Ti SUPER / SM89.
- Shared blast radius: the global macro also affects MHC expand/pre/post
  kernels, which were not part of this A5 bench.
- Generality: evidence covers the local 1B h4096/m24/hc4 params path only.
- Correctness: not evaluated because no treatment was licensed.

## Rule

Do not ship the MHC block=512 variant alone. It is a marginal 2.1% local win
with shared-kernel blast radius. Reconsider only if multiple marginal MHC axes
combine coherently and the combined treatment gets full correctness and
multi-kernel bench coverage.
