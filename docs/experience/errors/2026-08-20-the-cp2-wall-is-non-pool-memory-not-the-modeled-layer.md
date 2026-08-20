# The cp=2 wall at global 163,840 is memory outside the allocator pool, and the peak model does not see it — 2026-08-20

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

Free at the moment of failure is **9.56 MiB**, not 20 GB. The accounting, taken
from `ARLE_OPD_OP_MEM_CHECKPOINT_FN=60` at `scope_enter` — the last thing
recorded before the replay dies on its first allocation:

| Item | MiB |
|---|---:|
| Total | 97,508 |
| Pool reserved | 50,784 |
| Pool in use | 48,265 |
| Free inside the pool | 2,519 |
| Allocated outside the pool | ~46,714 |
| Driver free | 0.01 |

`cuMemAllocAsync` cannot grow the pool — the driver has nothing left — and the
2,519 MiB free inside it will not yield a 1.6 GB block.

Of the ~46,714 MiB outside the pool, the rollout engine accounts for 29,415
(free drops 97,508 → 68,093 at engine load). The remaining ~17,300 MiB is
unattributed; cause unknown.

## Root cause of the wrong reading

Two separate mistakes, both of reading rather than measurement.

`ckpt-peak actual` is sampled at checkpoint boundaries. The peak is set inside
one group's replay: `checkpoint_backward` (`tape.rs:986`) replays the group's
forward on an **enabled** inner tape, so that layer's whole intermediate set
becomes live at once, between two samples. The model reports 77,026 while the
run is at 97,498.

The per-op backward profiler then misled the same way at finer grain. Its last
line before the failure reads `free_mib=44,969`, which invited the conclusion
that a 1.6 GB allocation was failing with 44 GB free — an allocator anomaly.
It was not: ~45 GB is consumed between that line and the allocation, inside the
next op's backward, which never reaches its own `after_op`.

**A number logged before the failure is not the state at the failure.** The
allocation site is the only place that knows, which is exactly why throwing the
driver error away was expensive.

## Fix

`CudaBackend::zeros` now reports through `cuda_alloc_failed_rich`, which already
existed for this case — its comment reads "tell fragmentation from a sticky
async fault (fails with GB free)". Only `zeros` was still on the lossy path;
`ring_attn.rs` has five more sites that still discard the driver error.

## Where the remaining capacity is

Not in the modeled layer term. The levers that fit inside the pool are worth
single-digit GB each and the pool is not the binding side. The two sized
targets are the ~17,300 MiB of unattributed non-pool memory, and the
group-replay peak itself — one layer's full intermediate set going live at
once is what `layer=27,130 MiB` describes, and splitting that replay is the
only thing that moves it.

## Rule

Read allocator state from the failing allocation, never from the last log line
before it. And never write `map_err(|_| ...)` on an allocation: the driver code
plus live free/total is the whole diagnosis, and discarding it turns a
five-minute answer into a day of inference.
