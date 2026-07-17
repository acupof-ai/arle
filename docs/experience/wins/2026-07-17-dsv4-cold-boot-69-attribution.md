# DSv4 cold-boot #69 — code symptoms already fixed; residual cold cost is virtio disk bandwidth

## Context

#69 (filed 2026-06-10, pre-loader-rewrite) named two code symptoms — rank-0
serializes ~3.5 min after the workers' parallel build, and repeat boots not
faster despite 1.9 TB RAM — plus an 8× read-amplification suspect. Re-measured
2026-07-17 on current main (8×H20, DSv4-Flash-FP8 294 GB, TP=4,
`ARLE_CUDA_STARTUP_PROFILE=1`; logs `/root/boot-profile-69{,-cold}.log`).

## What Worked

Warm boot (page-cache resident): **33.0 s** launch → HTTP. All 4 ranks build
in 29.97 ± 0.01 s, fully concurrent — no rank-0 serialization; rank-0
prefetch 294 GB in 5.2 s (56.6 GB/s). Both filed symptoms are gone, fixed by
the intervening loader work (mmap zero-copy 2026-06-16/18, parallel prefetch
2026-07-03). Read amplification is also gone in wall-clock terms: rank-0
prefetches the checkpoint once into the shared page cache
(`loader.rs:761-835`), the other ranks hold at `broadcast_rank0_i32` (5.2 s
warm) and then demand-page warm slices.

Cold boot (model pages evicted via `posix_fadvise(DONTNEED)`, buff/cache
−274 GB): **1588.5 s (26.5 min)**, of which the prefetch read is 1557.3 s =
**98.0%** (294 GB @ 0.19 GB/s); everything after prefetch-done is 28.2 s.
dd attribution probes: cold aggregate is ~190–195 MB/s at 1, 4, and 16
parallel streams alike — parallelism splits the same cap — while a warm
re-read runs 3.8 GB/s. `/host` is a single virtio block device (`vda`, ext4,
`rotational=1` cloud volume). **0.19 GB/s is the device limit, not a
read-pattern artifact**; the single sequential prefetch pass already
saturates it.

Issue levers now moot: a pre-sliced per-rank shard cache still totals the
same ~294 GB of cold bytes at TP=4 (no cold-read reduction, only added disk
state); phase-timing logs already exist (`ARLE_CUDA_STARTUP_PROFILE`); JIT
warmup is a DSv4 no-op (`executor.rs:307`).

## Rule

Cold boot is storage-bandwidth-bound: floor ≈ checkpoint_bytes / device_GBps
(here 294 GB / 0.19 GB/s ≈ 26 min). The only remaining lever is faster
storage (local NVMe / RAM-backed model tier) — an infra ticket, not runtime
code. Warm boots are 33 s; keep the page cache warm between runs and never
attribute a slow boot to the loader without checking `free -g` first.
