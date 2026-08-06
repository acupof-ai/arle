# Native FP8 forward halved the GEMM cluster and moved the step wall 1%

**Date:** 2026-08-06 · **Verdict:** REJECT `--fp8-native-gemm` · **Commits:**
`cafda607c` (feature, on main), stream-guard fix on main (`3c021aead`)

## Context

The FlashQLA default flip killed the old 71% recurrent-GDN-backward row. A fresh
nsys profile of one flag-on 80K step ranked the new bottleneck as GEMM precision:
nvjet cuBLAS **bf16** GEMM ×4 = 41% of the step, plus `fp8_block_scaled_to_bf16`
dequant 5.1% (16026 launches — the frozen FP8 weight re-dequantized every GEMM).
Serve runs the identical FP8 weights through native DeepGEMM at ~67% of FP8 peak;
training left FP8 on the table. The plan copied serve's proven
`try_fp8_deepgemm_dense_batch` call into the training forward (`matmul_bt`),
flag-gated, forward-only (the backward is bf16 straight-through — frozen weights
need no weight-grad).

## Result

Matched A/B, SEQ=81920, cp-size 2, GPU 4,5, same seed/model, `da51f303d`:

| Metric | bf16 baseline | fp8-native | Δ |
|---|---|---|---|
| step wall | 363.6 s | 359.8 s | **−1.0%** |
| forward_hidden_states | 63.3 s | 57.0 s | −10.4% |
| loss | 4.537510 | 4.532657 | in-envelope (MoE non-det) |
| nvjet bf16 GEMM cluster | 45.1% | 15.2% | −30pp |
| `fp8_block_scaled_to_bf16` dequant | 5.1% / 16026 launches | 1.5% / 4052 | −75% launches |

The mechanism did exactly what it was designed to: native `sm90_fp8_gemm` (~19%
of the ARM-B step) replaced the bf16 forward-projection GEMMs and cut the per-GEMM
dequant by 75%. Forward-value parity held (Stage-1 base-point loss matched to
4.14e-4). This is a confirmed, correct optimization of the forward.

## Root cause of the null result

The forward is only ~17% of the step; the backward is ~84% and is bf16
straight-through, which the flag does not touch by design. During the backward,
GPUs 4,5 ran at 0–11% utilization — the wall is host-orchestrated chunked-scan CP
+ nccl SendRecv + slice ops, not GEMM. Halving forward GEMM time saves ~6.6 s, but
that is a small slice of a backward-bound, host-orchestration-bound step, so the
net wall moved 1%. The predicted ~300 s was unreachable by a forward-only flag.

## Fix

`--fp8-native-gemm` stays **off** (opt-in, no default flip). The feature and the
stream-guard fix (`3c021aead`: the dense DeepGEMM entry now accepts CUDA's legal
null default stream, which autograd runs on) remain in the tree — the stream fix
is a genuine correctness bug repair, and the fp8 forward is correct and available
for any future backward-bound-elsewhere regime. No revert.

## Rule

Before optimizing a profiled cluster, confirm the cluster is on the step's
**critical path**, not just large in kernel-time. A 30pp GPU-kernel-time shift in
a phase that is 17% of the wall — and whose sibling phase runs the GPU at
0–11% — cannot move the wall more than that phase's share. Rank optimizations by
`phase_share × achievable_speedup`, and check GPU utilization per phase first: a
backward that idles the GPU is host-bound, and no kernel-precision change touches
a host-bound wall. Copying serve's SOTA was correct execution; the miss was not
re-checking that the forward was the binding constraint after FlashQLA moved it.
