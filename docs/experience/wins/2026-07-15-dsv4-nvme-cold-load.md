# DSv4 cold startup on local NVMe

> Status: Shipped

## Goal

Reduce TP=4 process-start-to-ready time for the 294 GB DSv4 checkpoint.

## Hypothesis

The virtual system disk, not loader CPU work, limits cold startup; local NVMe
plus lazy rank loading should remove the full-checkpoint prefetch.

## Parameters

- Baseline path: `/root/DeepSeek-V4-Flash-FP8` on `/dev/vda2`.
- Treatment path: `/data00/models/DeepSeek-V4-Flash-FP8` on `/dev/nvme0n1`.
- Treatment: `ARLE_LOADER_PREFETCH=0`; TP ranks load directly.
- Cache state: target checkpoint residency forced from 34,706,395,136 bytes to
  zero with per-file `POSIX_FADV_DONTNEED` before startup.

## Environment

- Host: 8x H20; driver 535.161.08; CUDA 12.9; TP=4 on GPUs 0,2,3,4.
- Model: DeepSeek-V4-Flash-FP8, 56 files, 294,055,072,457 bytes.
- Server: allreduce, 16 running requests, L2 off during the load measurement.
- Binary SHA256: `c3097824ca96244f3c7680fa14d500820e9bd297eae563cea7477ff62586f979`.

## Results

| measurement | system disk | local NVMe |
|---|---:|---:|
| direct read | 0.18 GB/s observed during prefetch | 5.6 GB/s `O_DIRECT` |
| full prefetch | 1,675.0 s | disabled |
| process start to HTTP ready | baseline failed after prefetch/load pressure | **80.95 s** |

The treatment's entire cold startup was 20.7x shorter than the baseline's
prefetch stage alone. The one-time copy took 33:17 at 140.38 MB/s; `rsync -ani`
reported zero differences and source/target byte counts and index hashes match.

Raw log: `/host/arle-megamoe-t1/logs/mega-compare-nvme-l0.log`.

## Problems

Two existing OPD services each held 840-870 GB anonymous L2 memory. The
treatment used `--kv-dram off`; no existing service was stopped.

## Learnings

PASS. Local checkpoints should use lazy rank loading. Full page-cache prefetch
is for high-latency storage only and must remain explicitly disableable.
