# The cp=2 wall at global 163,840 is one checkpoint group's replay, and the peak model undercounts it — 2026-08-20

Instrumented on `c7c9ec4c4` (`fix(train): surface the driver error when a zeros
allocation fails`).

## Context

Target is global sequence 262,144 on 2 GPUs. The ceiling has sat at 131,072
since `28a1a79ef`. Global 163,840 (local 81,920) fails in the backward on
`zeros [1, 81920, 5120]`, 1.6 GB, and the peak model reads
`actual=77,026 MiB` of 97,508 — apparently 20 GB of headroom.

## Phenomenon

`map_err(|_| ...)` on `CudaBackend::zeros` discarded the driver error, so the
failure carried no evidence. With `cuda_alloc_failed_rich` wired in:

```
alloc zeros failed: bytes=1677721600 err=DriverError(CUDA_ERROR_OUT_OF_MEMORY)
free_total=Some((10027008, 102245335040))
```

Free at the moment of failure is **9.56 MiB**, not 20 GB.

`ARLE_OPD_OP_MEM_CHECKPOINT_FN=60` arms a per-op ledger inside the last
checkpoint group (`layers=63..64`, one layer). It records `scope_enter` and
then **nothing** — the replay dies on its first allocation:

```
scope_enter pool_reserved_mib=50784 pool_used_current_mib=48265 live_tensors=990
```

Against the driver reading two tape entries earlier (`index=61 op=RMSNorm`,
`used_mib=52,539`), non-pool memory is about **4,300 MiB** — CUDA context,
module images, NCCL buffers. The rollout engine's 29,415 MiB of FP8 weights
(30 GB on disk, one copy, shared with training) sits inside the pool, in
`pool_used_current`.

So the layout at `scope_enter` is roughly 48.3 GB in the pool (weights plus
live training state), 4.3 GB outside it, and 45 GB free. Replaying **one
layer** then consumes that 45 GB and dies, against a modeled `layer` term of
27,130 MiB.

## Root cause of the wrong readings

Three wrong conclusions this session, all the same mistake: **subtracting two
numbers sampled at different moments.**

1. `ckpt-peak actual` is sampled at checkpoint boundaries. The peak is set
   inside one group's replay — `checkpoint_backward` (`tape.rs:986`) replays
   the group forward on an **enabled** inner tape, so that layer's whole
   intermediate set goes live at once, between two samples. The model reports
   77,026 while the run is at 97,498.

2. The per-op backward profiler's last line before the failure reads
   `free_mib=44,969`, which invited "a 1.6 GB allocation is failing with 44 GB
   free — an allocator anomaly." It is not: those 45 GB are consumed between
   that line and the allocation, inside the next op's backward, which never
   reaches its own `after_op`.

3. Subtracting `pool_reserved` at `scope_enter` from driver `used` at the
   failure produced a phantom 17,300 MiB of unattributed non-pool memory. The
   pool grew by ~45 GB between those two samples. Non-pool is 4,300 MiB.

## Fix

`CudaBackend::zeros` now reports through `cuda_alloc_failed_rich`, which
already existed for this case — its comment reads "tell fragmentation from a
sticky async fault (fails with GB free)". Only `zeros` was still on the lossy
path; `ring_attn.rs` has five more sites that still discard the driver error.

## Where the remaining capacity is

One place: the group replay. `checkpoint_sequential` groups one layer, and
`checkpoint_backward` replays that layer with the tape enabled, so every
intermediate in the layer is live simultaneously. Nothing else on the device is
a lever — the weights are one shared copy and are required by every matmul in
the replay, and non-pool is 4.3 GB.

## Rule

Read allocator state from the failing allocation, never from a log line before
it, and never subtract two samples taken at different moments — the quantity
between them is exactly what you are trying to measure. Never write
`map_err(|_| ...)` on an allocation: the driver code plus live free/total is
the whole diagnosis.
