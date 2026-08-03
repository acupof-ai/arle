# Model load is single-core bound at ~200 MB/s — the disk was never involved, 2026-08-04

> Status: **Measured, not yet fixed.** Cold boot of the 27B W8A16 checkpoint
> takes ~122 s for 29 GB. Two wrong causes were proposed and killed by
> measurement before the real one was found.

## The measurements

| probe | result |
|---|---|
| `/host/nvme0` device | `/dev/nvme0n1`, ext4 — a real NVMe, not overlay or network |
| `dd iflag=direct` on a model shard | **2.7 GB/s** |
| weight load, cold | 29 GB in ~122 s = **238 MB/s** |
| weight load, page-cache warm | same rate |
| loader process `read_bytes` during load | **0** |
| loader CPU over a 20 s window | **2004 ticks = 100.2% — exactly one core** |
| GPU memory growth over the same window | 4064 MiB = **203 MB/s** |

`read_bytes = 0` with the cache warm, at the *same* throughput as the cold
run, is the whole proof: the loader is not waiting on storage in either case.
One core is saturated and weight bytes arrive at ~200 MB/s.

## Two wrong causes, both killed by measurement

**"The pod's disk is slow."** It reads at 2.7 GB/s. The 0.2 GB/s figure in
[[reference_h20_pod_storage_is_02gbs_cold_boot_25min]] is about `/host`
(`/dev/vda2`), not `/host/nvme0` — the checkpoint lives on the NVMe.

**"mmap has no readahead."** `SafetensorLoader` does map shards with a bare
`mmap(PROT_READ, MAP_PRIVATE)` and calls `madvise` nowhere, which is a real
gap — but it cannot be *this* gap, because a warm-cache load takes zero disk
reads and still runs at 200 MB/s. Adding `MADV_WILLNEED` would change nothing
until the CPU bound is lifted.

Both were plausible, and I stated the first one to the user as fact before
measuring it. The correction cost more than the measurement would have.

## What is actually unknown

The single-core work has **not** been localized to a function. Ruled out by
reading: the Marlin repack runs on the GPU (`gptq_marlin_repack_cuda`), so it
is not the host cost. Remaining candidates are the per-tensor host path
between the mmap'd bytes and the device — staging copies, dtype handling, and
pageable-memory H2D (the driver bounce-buffers pageable source with the CPU).

**Next step is a stack sample of the loading process, not more code reading.**
`perf` is absent on this pod; `gdb -p` sampling is the established fallback.

## Rule

**A throughput number is not a bottleneck until you have measured the
resource you are blaming.** "29 GB took 122 s" is compatible with a slow disk,
a slow mmap, and a busy core; picking one by plausibility picked wrong twice.
`read_bytes` and a CPU-tick delta over a fixed window cost one command each
and are decisive.

Related: [[feedback_no_ungrounded_estimates]],
[[feedback_dont_file_hypothesis_as_root_cause]].
