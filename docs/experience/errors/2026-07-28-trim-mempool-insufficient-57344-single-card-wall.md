# Pool trim did not fix the 57,344 backward peak

> Historical result for binary `3ad11830`. Later checkpoint replay and in-place
> gradient fixes supersede this capability ceiling.

## Context

Hypothesis (from the baseline trace): 57344 OOMs because forward
gradient-checkpointing hoards ~42.8 GB of allocator free-list pages the pool never
returns; backward starts on top of that hoard with only 16.5 GB free and OOMs on
its first `add_into_device` (2.82 GB fp32 q_proj grad). Fix tried: one line
`store.backend().trim_memory_pool()` at the forward→backward boundary.

## What was measured (H20 GPU6, seq=57344 chunk=4096, binary 3ad11830)

**Still OOMs — byte-identical failure signature:**
`add_into_device bytes=2818572288 free_total=(26804224, ...)` — 2.82 GB alloc,
25.56 MiB free, same as baseline.

**But the trim demonstrably fired.** Direct `nvidia-smi -i 6` while the log sat at
`fused_ce`: GPU memory dropped **89 → 63 → 43 GB** (the ~40 GB hoard returned to
the driver), then climbed back to **91 GB** as backward ran, before failing on a
*later* `add_into_device`. Baseline OOMed on the *first* backward alloc
immediately; this run ran ~4-5 min into backward before hitting the same wall.

**Conclusion for this binary:** the trim moved the OOM deeper into backward but
did not close the gap. Reclaiming the 42.8 GB hoard was not enough.

## Root cause correction

The baseline trace's "42.8 GB hoard is the load" read was half right: the hoard
WAS reclaimable and the trim DID return it. But the flagged 2.82 GB alloc is not
merely a straw sitting on a reclaimable cache — after the cache is gone, backward's
live peak still overflows the card. The real constraint at 57344 is backward
live-activation memory, which the trim cannot touch.

## Rule

Watching `pool_used_current` plateau while driver-used is far higher tells you the
hoard is reclaimable — but reclaiming it only helps if the post-reclaim live peak
fits. Verify the terminal outcome (completed vs OOM), never call a fix from an
interim `nvidia-smi` dip: a mid-backward 43 GB trough is the trim working, not the
workload fitting. When backward live peak exceeds one card after the hoard is
freed, the answer is live-peak reduction (activation chunking) or context
parallelism — not more trimming.

## Disposition of the trim line

Necessary-but-insufficient: it returns the forward hoard but does not reduce live
backward tensors. The single-GPU path must reduce those tensors and remove index
walls; context parallelism is a separate milestone.
