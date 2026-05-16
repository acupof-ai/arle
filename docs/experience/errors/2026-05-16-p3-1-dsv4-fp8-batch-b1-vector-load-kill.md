# P3.1 DSv4 FP8 batch B1 vector weight load kill

## Context

Phase 3 P3.1 A2 tested `uint32_t` vectorized weight loads in
`dsv4_fp8_gemv_batch_kernel`, the B=1 raw path behind
`dsv4_fp8_gemv_batch_cuda`.

## Formula Prediction

Hypothesis before edit:

- SM89 constants: 64K registers/SM, 100KB shared memory/SM, 1536 threads/SM,
  672 GB/s HBM.
- Workload constants: `GEMV_THREADS=256`, `GEMV_ROWS=4`,
  `threads_per_row=64`, `K=1024`, `B=1`.
- Baseline has 1024 scalar `uint8_t` weight loads per output row. The
  treatment changes the K-multiple-of-4 path to 256 `uint32_t` loads per row
  and decodes four FP8 bytes from each register.
- HBM bytes are not expected to fall by 75% because the scalar baseline is
  already coalesced across warp lanes. The expected win is reduced load
  instruction count and less loop overhead, while input BF16 loads, scale
  loads, FP8 decode, launch overhead, and reductions remain.

Predicted point delta was -3% to -5% for `1024x1024` and -2% to -5% for
`512x1024`.

## Root Cause

The result was positive but below the Phase 3 license threshold. This falsifies
the assumption that scalar byte loads are a material standalone bottleneck for
this B=1 raw path. The baseline scalar loads are coalesced enough that replacing
them with `uint32_t` loads does not buy a shippable local component win.

## Evidence

Baseline command:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_fp8_gemv_batch_b1 --save-baseline p3_1_a2_before
```

Treatment command:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_fp8_gemv_batch_b1 --baseline p3_1_a2_before
```

| Shape | Baseline point | Treatment point | Criterion change | p-value | Verdict |
|---|---:|---:|---:|---:|---|
| `dsv4_mini_hidden_1024x1024` | `9.8116 us` | `9.7623 us` | `-0.5871%` | `0.00 < 0.05` | KILL |
| `dsv4_mini_moe_512x1024` | `8.2380 us` | `8.0951 us` | `-1.5234%` | `0.00 < 0.05` | KILL |

The MoE shape improved, but the point estimate is below 2%; the hidden shape
was also marked "Change within noise threshold" by Criterion. This does not
meet the `>=3%` license gate or the `2-3%` review bucket.

## Fix

Reverted the A2 runtime change. Keep scalar `uint8_t` weight loads in
`dsv4_fp8_gemv_batch_kernel`.

## Tradeoff

- LOC complexity: treatment added a K-alignment branch and four-way decode
  expansion.
- SM89 specificity: no explicit SM-specific code, but the decision is local to
  SM89 measurements.
- Shared memory budget: unchanged.
- Register budget: worse due four decoded weights and four scale-column
  temporaries per loop.
- CUDA Graph compatibility: unchanged.
- Generality across batch sizes: B=1 only; B>1 tiled path was not touched.
- Numerical correctness margin: treatment changed reduction grouping, so a
  correctness test would have been required if licensed.

## Rule

Do not ship standalone `uint32_t` vectorized weight loads for
`dsv4_fp8_gemv_batch_kernel` B=1 on the current local shapes. The measured
benefit is real but too small for the complexity and below the Phase 3 gate.
