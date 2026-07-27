# Mempool hoard column in the OPD VRAM ledger — CUDA, 2026-07-26

> Status: Shipped (diagnostic, behind `ARLE_OPD_VRAM_TRACE`). On-GPU hoard
> readout at seq=40960 verified 2026-07-27 (H20, GPU 6, `5fbf38e4e`).

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

## Verified (2026-07-27, H20 GPU 6, `5fbf38e4e`)

`--synthetic-writeback-seq 40960` with `ARLE_OPD_VRAM_TRACE=1`, rc=0, `DONE
loss=8.685793` (default-path re-run bit-identical). Ledger:

```
hoarded_fwd/bwd/clean_mib = 39030 / 3167 / 6413
```

The spike is at the FORWARD checkpoint, not post-backward: the forward
accumulates a 39 GB hoard across the 64 checkpoint groups — the pre-concat
fragmentation `cuMemGetInfo` hides — and the per-replay trim reclaims it,
collapsing the retained hoard to 3167 MiB by post-backward (−35.9 GB) and
6413 MiB post-cleanup. One ledger line now shows the whole envelope the
four-arm sweep once had to localize.
