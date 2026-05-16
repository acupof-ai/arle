# P3.6 DSv4 FP8 batch tiled smem input STOP

## Context

Phase 3 P3.6 A3 attempted to evaluate shared-memory activation staging for
`dsv4_fp8_gemv_batch_tiled_kernel` in
`crates/cuda-kernels/csrc/gemm/quantized_gemv.cu`.

A1 scale-column hoist from `a47f723` was the baseline. The intended treatment
was to stage the current batch tile's BF16 activations into dynamic shared
memory so the four row groups in a CTA could reuse input loads.

## Formula Prediction

Hypothesis: unlike the B=1 raw path, the B=4 tiled path has CTA-local input
reuse across rows and batch slots. Staging up to `DSV4_BATCH_TILE * K` BF16
values could reduce redundant global input loads.

Risk: the treatment adds a pre-load loop, dynamic shared memory, and an extra
`__syncthreads()`. It also needs a safe production fallback because
`DSV4_BATCH_TILE * K * sizeof(bf16)` can exceed the per-block shared-memory
limit for larger K.

## Root Cause

This axis did not produce valid performance evidence. Two treatment attempts
failed at nvcc compile time because broad patch anchors inserted the staging
block into the wrong sibling kernel:

- first into `dsv4_fp8_gemv_kernel`, where `B` and `batch_base` are undefined
- then into `dsv4_fp8_gemv_batch_kernel`, where `batch_base` is undefined

Because no treatment binary was produced, there is no SOLID performance
conclusion for shared-memory staging on the tiled FP8 path. The correct next
attempt should use a dedicated helper or a separate tiled-smem kernel rather
than broad context matching across near-identical GEMV kernels.

## Evidence

Baseline before treatment:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --bench ops_bench --features cuda -- \
  ops_cuda/dsv4_fp8_gemv_batch/ --save-baseline p3_6_a3_before
```

Baseline results:

| Shape | Time |
|---|---:|
| `dsv4_mini_hidden_1024x1024` | `21.658 us` |
| `dsv4_mini_moe_512x1024` | `15.606 us` |

Failed treatment command:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --bench ops_bench --features cuda -- \
  ops_cuda/dsv4_fp8_gemv_batch/ --baseline p3_6_a3_before
```

Compile failures:

```text
csrc/gemm/quantized_gemv.cu(269): error: identifier "B" is undefined
csrc/gemm/quantized_gemv.cu(269): error: identifier "batch_base" is undefined
```

and then:

```text
csrc/gemm/quantized_gemv.cu(364): error: identifier "batch_base" is undefined
csrc/gemm/quantized_gemv.cu(442): error: identifier "tile_batches" is undefined
csrc/gemm/quantized_gemv.cu(443): error: identifier "x_smem" is undefined
```

## Fix

The treatment was fully reverted. No runtime patch was shipped.

## Rule

Do not continue A3 by broad-patching shared snippets in
`quantized_gemv.cu`. If this axis is reopened, create a separate
`dsv4_fp8_gemv_batch_tiled_smem_kernel` or use exact function-scoped editing,
and add a host-side shared-memory-size guard before any winning treatment can
ship.
