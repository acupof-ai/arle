# Mempool hoard column in the OPD VRAM ledger — CUDA, 2026-07-26

> Status: Shipped (diagnostic, behind `ARLE_OPD_VRAM_TRACE`). On-GPU hoard
> readout at seq=40960 pending-remote.

## Goal

`cuMemGetInfo` reports pool pages that are freed-but-retained as "used", so its
`used` can't be split from live tensors. That gap is exactly what forced this
session's four-arm A/B to localize the 40960 backward OOM to fragmentation
([[2026-07-26-per-replay-trim-fixes-40960-writeback]]). Surface the split
directly.

## What changed

Read the async mempool's `RESERVED/USED_MEM_CURRENT` attributes; `reserved -
used` is the hoard (retained-but-free pages). Thread it into the existing
`--writeback`/`ARLE_OPD_VRAM_TRACE` ledger.

- `crates/cuda-kernels/src/tensor.rs`: `CudaContext::mem_pool_stats() -> Option<(reserved, used)>`.
- `crates/autograd/src/backend.rs`: `Backend::mem_pool_stats()` (default `None`).
- `crates/autograd/src/backend_cuda.rs`: CUDA impl via `cuMemPoolGetAttribute`
  (cfg'd `not(no-cuda)`; no-cuda falls to the `None` default).
- `crates/train/src/opd.rs`: `VramSample` gains `pool` + `hoarded_mib()`; the
  per-milestone line and the ledger summary print `hoarded_fwd/bwd/clean_mib`.

## Rule

A rising hoard across forward→backward milestones with flat live-used is the
fragmentation signature — now one ledger line instead of a four-arm sweep.
`reserved - used` is the number `cuMemGetInfo` hides.

## Pending-remote

Run seq=40960 writeback with `ARLE_OPD_VRAM_TRACE=1`; confirm the pre-concat
hoard spike the trim reclaims is visible in `hoarded_bwd_mib`.
