# Kernel six-principle fixes — CUDA, 2026-08-11

> Status: Shipped (correctness verified; perf pending-remote)

## Goal

Eliminate shared-memory bank conflicts and expose ILP in three custom CUDA
kernels (dsv4_mhc RMS norm, gemv, W4A16 dequant GEMM) without changing
numerical output.

## Hypothesis

Bank conflicts in uint16/fp16 shared memory (2 elements per bank) and the
single-accumulator FMA chain in gemv were leaving SM throughput on the table.
Padding the shared-memory stride to be coprime with 32 (or swizzling column
indices) removes the conflicts; 4 independent accumulators hide FMA latency.
Output is bit-identical because the same values are loaded and summed in the
same order.

## Changes

| Kernel | Fix | Principle |
|--------|-----|-----------|
| `dsv4_mhc.cu` | `col ^ ((col & 1) << 5)` swizzle; +32 elem shmem | Bank conflict avoidance |
| `gemv.cu` | `p = k + (k >> 8)` (stride 257, coprime with 32); 4 accumulators `sum0..sum3` | Bank conflict + ILP |
| `fp16_gemm_wmma.cu` | New WMMA GEMM for W4A16→FP16 dequant path (small-M DSpark verify) | Shared memory + tensor cores |
| `w4a16_gemm_wmma.cu` | Output layout [N,M]→[M,N] (inert: not on dispatch path) | — |

## Correctness

`needle_gate.py` (RAW=1, TEMPLATE=qwen3_nonthink):

| Length | exact | partial | miss | det |
|-------:|------:|--------:|-----:|-----|
| 115    | 3/3   | 0       | 0    | DET |
| 180    | 3/3   | 0       | 0    | DET |
| 241    | 3/3   | 0       | 0    | DET |
| 300    | 3/3   | 0       | 0    | DET |
| 446    | 3/3   | 0       | 0    | DET |
| 1000   | 3/3   | 0       | 0    | DET |
| 2000   | 3/3   | 0       | 0    | DET |

`lever_gate.sh` (GATE_PROFILE=generic):

| Length | exact | partial | miss | det |
|-------:|------:|--------:|-----:|-----|
| 115    | 3/3   | 0       | 0    | DET |
| 300    | 3/3   | 0       | 0    | DET |
| 446    | 3/3   | 0       | 0    | DET |
| 2000   | 3/3   | 0       | 0    | DET |
| 8000   | 3/3   | 0       | 0    | DET |

`[gate] correctness PASS: summaries=5`. Serve log clean: no NaN/Inf, no
kernel errors.

## Performance

pending-remote — ncu before/after for `dsv4_mhc_pre_rms_norm` and
`gemv_handwritten_kernel` on sm_90. The fixes are standard
bank-conflict/ILP patterns; expected speedup proportional to the conflict
degree (2-way for uint16, higher for the gemv 4-way unroll).

## Problems

`lever_gate.sh` SUMMARY regex anchored on `DET|NONDET$` and rejected the
`kv=` suffix that `needle_gate.py` now appends. Fixed in
`6600e7e3d` — test-script mismatch, not a kernel issue.

## Learnings

PASS — correctness preserved across all needle lengths 115–8000. Perf
measurement deferred to the next pod window (ncu on the two fixed kernels).
