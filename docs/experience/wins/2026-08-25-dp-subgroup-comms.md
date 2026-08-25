# DP subgroup comms via CommAxis::Dp — autograd/train, 2026-08-25

> Status: pending-remote

## Goal

Replace the host zero-masking on the World comm with a real DP subgroup axis
(#224): under DP×CP, the count reduce (`inv_n`) sums over the DP subgroup
(ncclCommSplit) instead of zeroing non-cp-rank-0 contributions on World.

## What landed

- `CommAxis::Dp` variant; `CudaBackend::nccl_dp` field populated by
  `new_with_mesh(dp_group = (cp.rank, dp.size, dp.rank))` — the same
  ncclCommSplit pattern as the existing Seq subgroup.
- `comm()`: `Dp => nccl_dp.or(nccl)` — World fallback when no CP (DP subgroup
  == World), so DP-only deployments are unchanged.
- `dp_group_sum_scalar/count` take an explicit `CommAxis` param.
- writeback: count reduce passes `CommAxis::Dp` and drops the `cp.rank == 0`
  zero-masking (every rank contributes; the DP subgroup holds one CP rank per
  replica). Loss reduce stays on `CommAxis::World` (the world sum of partials
  is the global mean).
- Drive-by: remove a pre-existing `mut` lint in math_opd.rs (39c43d5d2).

world==1 identity preserved (cpu-lane `all_reduce_sum_device` is identity):
`cargo test -p autograd -p train` green. Mac CUDA clippy lint green
(cuda,no-cuda,nccl,deepep -D warnings).

## Parameters

```bash
# pending-remote: cp=2 x dp=2 OPD 1-step on H20
# - numerical agreement vs World-comm baseline (host zero-masking) within rtol 1e-5
# - no divergence across the step
```

- Baseline: `50f8183f3` (zero-masking on World)
- Treatment: this commit (CommAxis::Dp subgroup)
- Trials: pending-remote

## Environment

- Host / GPU: H20 pod, cp=2 × dp=2 (pending-remote)

## Rule

A subgroup axis with a World fallback is the migration seam: callers opt in
per-reduce, and a composed mesh populates the split comm while an uncomposed
one keeps the identity fallback.
