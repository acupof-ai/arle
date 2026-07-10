# P3.2 DSv4 FP4 batch B1 scale decode reuse kill

## Context

Phase 3 P3.2 A4 tested E8M0 scale decode reuse inside
`dsv4_fp4_gemv_batch_kernel`, the B=1 raw path behind
`dsv4_fp4_gemv_batch_cuda`, after the A2 pair-load optimization landed.

## Formula Prediction

Hypothesis before edit:

- SM89 constants: 64K registers/SM, 100KB shared memory/SM, 1536 threads/SM,
  672 GB/s HBM.
- Workload constants: `GEMV_THREADS=256`, `GEMV_ROWS=4`,
  `threads_per_row=64`, `K=1024`, `scale_cols=8`, `block_w=128`, `B=1`.
- After A2, each thread iteration handles one FP4 packed byte and therefore
  two adjacent logical K values (`k0`, `k1`).
- For the local shapes, `block_w=128`, so `k0` and `k1` are in the same scale
  column for every packed byte. Reusing the decoded scale should remove one
  E8M0 decode per pair.

Predicted point delta was -3% to -7% for `1024x1024` and -2% to -6% for
`512x1024`.

## Root Cause

The hypothesis was falsified. Reusing the decoded scale for adjacent nibbles
is below measurement noise. Scale decode is not a standalone binding cost after
A2 pair-load; the extra compare/branch also leaves no measurable gain.

## Evidence

Baseline command:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_fp4_gemv_batch_b1 --save-baseline p3_2_a4_before
```

Treatment command:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_fp4_gemv_batch_b1 --baseline p3_2_a4_before
```

| Shape | Baseline point | Treatment point | Criterion change | p-value | Verdict |
|---|---:|---:|---:|---:|---|
| `dsv4_mini_hidden_1024x1024` | `9.5383 us` | `9.5372 us` | `-0.1276%` | `0.22 > 0.05` | KILL |
| `dsv4_mini_moe_512x1024` | `7.9854 us` | `7.9976 us` | `+0.0459%` | `0.60 > 0.05` | KILL |

Criterion reported "No change in performance detected" for both shapes.

## Fix

Reverted the A4 runtime change. Keep separate E8M0 scale decode expressions
for the low and high FP4 nibbles.

## Tradeoff

- LOC complexity: treatment added scale temporaries and an `sc1 == sc0` branch.
- SM89 specificity: measured locally on RTX 4070 Ti SUPER / SM89.
- Shared memory budget: unchanged.
- Register budget: slightly worse due extra live scale values.
- CUDA Graph compatibility: unchanged.
- Generality across batch sizes: B=1 only; B>1 tiled path was not touched.
- Generality across shape: no significant change on either shape.
- Numerical correctness margin: equivalent for same-scale pairs, but no
  correctness gate was needed because performance failed.

## Rule

Do not ship standalone adjacent-nibble E8M0 scale decode reuse for
`dsv4_fp4_gemv_batch_kernel` B=1. It is below noise after A2 pair-load.
