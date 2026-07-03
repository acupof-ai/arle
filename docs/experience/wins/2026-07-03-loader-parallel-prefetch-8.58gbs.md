# DSv4 cold load 31min → ~34s: parallel shard prefetch hits 8.58 GB/s

## Context
Cold DSv4 boot loaded the 274 GB FP8 checkpoint in ~31 min (~150 MB/s). The
early assumption was disk-bandwidth-bound (dd QD1 = 197 MB/s → "physical floor").

## What Worked
WRONG assumption: the mmap page-fault path is synchronous single-stream; the
virtio disk scales HUGELY with parallelism. `prefetch_shards` (loader.rs,
`cef391b7`) parallel-reads every shard (work-stealing, up-to-16 threads) into the
page cache before the per-tensor mmap loads. Pod-measured:

    loader prefetch: 294.0 GB across 46 shards in 34.3s (8.58 GB/s, 16 threads)

8.58 GB/s vs ~150 MB/s single-stream = **~57×**. Cold boot: ~34s prefetch + a
warm-cache mmap load, versus 31 min. The tensor loads then hit page cache
(~55 GB/s, round-6).

## Rule
"Disk-bandwidth-bound" from a dd QD1 number is a FALSE floor for an mmap-fault
loader — QD1 ≠ the disk's parallel ceiling. Measure the parallel rate (a
prefetch with a GB/s log line) before calling a load disk-bound; a virtio disk
here went 197 MB/s (QD1) → 8.58 GB/s (16 threads).
