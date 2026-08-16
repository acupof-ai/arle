# CP T1: ring-attention core extracted into cuda-kernels, gates green

**Date:** 2026-08-16 · **Commit:** 083e2e89a · **Plan:** docs/plans/2026-08-16-cp-ideal-state.md (T1)

## Context

CP ideal-state tranche T1: the training-only autograd crate held the entire
ring-attention core (flash-2 merge math, FA3 pair route, per-block launches),
blocking any engine CP consumer. Behavior-neutral extraction into
`cuda-kernels/src/ring_attention.rs` (tape-free, engine-callable).

## What worked

- Extraction seam: the FA3-layer fns already operated on `CudaSlice<u16>/<f32>`;
  only the outer wrappers touched `CudaBackend`. Re-parameterized to
  `&Arc<CudaStream>`; autograd keeps DeviceHandle translation, f32↔bf16
  staging, scalar fallback kernels, and the tape op
  (`backend_cuda/ring_attn.rs` 728→309 lines).
- Host merge math compiles without the `cuda` feature, so autograd's CPU gates
  keep running; `cuda-kernels` became a mandatory default-features-off dep
  (featureless build stays nvcc-free).

## Result (pod, 8×H20, GPUs 3,4)

- `cp_ring_transport_parity`: PASS, worst_rank_max_diff 2.98e-8 (tol 5e-3).
- `cp_hidden_parity`: PASS, ce_cp_vs_cpu 3.90e-4; cp_vs_cpu_f32 3.16e-2 vs
  single_vs_cpu_f32 3.49e-2 — CP tracks f32 truth at least as well as
  single-card.
- autograd cuda,nccl tests: 51 pass; `cuda_trim_memory_pool_releases_unused_pages`
  fails only in-suite (shared-pool test-parallelism artifact; passes solo).
- Local: arle cuda,no-cuda typecheck clean; autograd CPU tests 29/29 identical
  before/after.

## Rule

`scripts/pod-build-env.sh` exports `CUDA_VISIBLE_DEVICES=""` (build-time GPU
hiding); any pod-side RUN script sourcing it must `unset CUDA_VISIBLE_DEVICES`
before launching, or every rank fails `CudaContext::new`.
