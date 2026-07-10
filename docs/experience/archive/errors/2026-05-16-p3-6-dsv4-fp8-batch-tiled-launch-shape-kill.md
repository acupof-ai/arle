# P3.6 DSv4 FP8 batch tiled launch shape kill

## Context

Phase 3 P3.6 A5 tested launch-shape variants for
`dsv4_fp8_gemv_batch_tiled_kernel` after the A1 scale-column hoist landed in
`a47f723`.

The current shared constants are:

```c
#define GEMV_THREADS 256
#define GEMV_ROWS 4
```

The experiment temporarily changed these shared macros to test the tiled B=4
bench only. No launch-shape runtime change was shipped.

## Formula Prediction

Hypothesis:

- `256x8` halves CTA count but halves per-row thread parallelism.
- `512x4` doubles per-row thread parallelism but may reduce block residency and
  increase scheduling cost.
- `512x8` keeps per-row thread parallelism at 64 while halving CTA count, but
  uses 512-thread blocks and larger per-CTA reduction state.

Because these macros are shared by multiple GEMV kernels, any winning treatment
would need to be reimplemented as tiled-kernel-specific constants before
shipping. The global macro change itself is only an experimental probe.

## Evidence

Baseline:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --bench ops_bench --features cuda -- \
  ops_cuda/dsv4_fp8_gemv_batch/ --save-baseline p3_6_a5_before
```

Baseline results:

| Shape | Time |
|---|---:|
| `dsv4_mini_hidden_1024x1024` | `21.687 us` |
| `dsv4_mini_moe_512x1024` | `15.595 us` |

Treatment command for each launch shape:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --bench ops_bench --features cuda -- \
  ops_cuda/dsv4_fp8_gemv_batch/ --baseline p3_6_a5_before
```

Treatment results:

| Variant | Hidden time | Hidden change | MoE time | MoE change | Decision |
|---|---:|---:|---:|---:|---|
| `256x8` | `22.796 us` | `+5.0554%` | `21.019 us` | `+34.873%` | KILL |
| `512x4` | `26.483 us` | `+22.139%` | `16.959 us` | `+8.7570%` | KILL |
| `512x8` | `22.584 us` | `+4.1299%` | `15.721 us` | `+0.8508%` | KILL |

All p-values were `0.00 < 0.05`; the measured direction is regression or
noise-regression for every variant.

## Fix

Treatment reverted. Keep the shared launch constants at `256x4`.

## Rule

Do not change the shared GEMV launch-shape macros for the DSv4 FP8 B=4 tiled
path on local SM89. The A1 scale-column loop wants the existing 64 threads per
row and 4 rows per CTA balance; reducing CTAs or increasing block size regresses
at least one measured shape.
