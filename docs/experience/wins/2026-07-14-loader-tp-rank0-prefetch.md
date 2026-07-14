# TP rank-zero checkpoint prefetch

> Status: Shipped

## Goal

Remove redundant TP checkpoint reads that made DSv4 cold load take about 50 minutes.

## Hypothesis

Every TP rank prefetching the full base and draft checkpoints multiplied storage traffic while the host lacked enough page cache to retain the base checkpoint.

## Params

- Model: DeepSeek-V4-Flash + DSpark draft
- GPU: 4× H20, TP=4
- Base checkpoint: 294 GB
- Draft checkpoint: 19.9 GB
- Prefetch cache headroom: 64 GiB
- Server: GPUs 3–6, port 8799

## Env

- Deployment pod, CUDA/NCCL release-fast build
- Host `MemAvailable` at the slow start: about 156 GiB
- Post-fix measurements were hot-cache starts; no second destructive cold cycle was run

## Results

| Variant | Prefetch readers | Observed checkpoint traffic | Ready time |
|---|---:|---:|---:|
| Before, cold | 4 ranks | 4×294 GB base + 4×19.9 GB draft; about 1.65 TB including load faults | about 50 min |
| After, hot cache | rank 0 only | 294 GB in 6.3–9.6 s; 19.9 GB in 1.2 s | 41–43 s |

The fix permits prefetch only on rank zero when `MemAvailable >= checkpoint bytes + 64 GiB`, then synchronizes all ranks through the existing collective. Otherwise rank zero logs the capacity skip and loading proceeds normally. A targeted test passed the rank and capacity decision matrix; the real TP=4 log contained no rank 1–3 prefetch.

## Problems

- The before/after ready times are not a controlled cold-cache A/B, so they do not license a speedup ratio.
- The previous prefetch optimization assumed the entire checkpoint remained cached; that premise was false on this host.
- Canonical GuideLLM is not informative for loader-only startup behavior; the measured gate is process-ready wall time and physical I/O.

## Learnings

Checkpoint prefetch is useful only when the page cache can retain the data. In TP, one reader is sufficient because all ranks share the host page cache.
