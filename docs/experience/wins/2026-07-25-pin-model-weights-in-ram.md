# Pinning the weights in RAM kills the 25-min cold boot (#181)

> Measured 2026-07-25 on the 8×H20 pod. 294 GB `VmLck`, survives
> `drop_caches=3`, post-drop full re-read **38 s at 7.7 GB/s** vs 0.19 GB/s
> from disk.

## Context

The pod's `/host` (ext4 on virtio `/dev/vda2`) reads at ~0.2 GB/s and does
**not** scale with concurrency (`dd iflag=direct`: 1 / 4 / 16 streams all land
0.19-0.23 GB/s, and the rate is high-variance, 0.10-0.23 across repeats). So a
cold DeepSeek-V4-Flash boot spends 1557 s in `loader prefetch` — 25 min before
engine-ready — while a warm page cache makes the same prefetch 5.3 s. Every cold
start burned 25 min before any verification could begin.

Nothing in code can fix that: the device is the floor, one rank already
saturates it, and reading per-rank in parallel would only 4× the volume against
the same ceiling (~100 min). See the closed #181 for the measurement that
overturned the first "serialized on rank 0" hypothesis.

## What Worked

The page cache was already the right cache — it just wasn't *guaranteed*.
`scripts/pin_model_cache.py` mmaps every shard `MAP_SHARED | MAP_POPULATE` and
`mlock`s it, then holds. ~90 lines, stdlib + ctypes, no new dependency and no
filesystem layer.

Rejected on the way: a FUSE cache. It would reimplement the page cache in
userspace (warm reads drop from 55 GB/s to userspace-round-trip speed), and it
cannot touch the cold path at all — there is no faster tier to serve from, since
`/host` and the overlay are the same `vda2` and the only faster medium is the RAM
the page cache already uses. Residency was the *only* thing missing, and `mlock`
buys exactly that.

Evidence chain, in order:

1. Mechanics on one 1.1 GB shard: `VmLck` 1.03 GB, global `Mlocked` +1.07 GB.
2. Full run: 46 shards, `VmLck` 287147388 kB (274 GiB), `MemAvailable`
   1954 → 1666 GiB — 274 GiB of 1929 GiB total, 14%.
3. **The pin holds under eviction**: `drop_caches=3` cut `Cached` by 204 GB
   (497.4 → 292.8) while `Mlocked` and `VmLck` did not move a byte; the residual
   `Cached` is the pinned model.
4. **End-to-end**: `cat *.safetensors > /dev/null` immediately after that drop
   read 294 GB in 38.4 s (7.7 GB/s, single-threaded). From disk the same read is
   25 min.

Guards worth keeping: mlocked pages are unreclaimable, so the script refuses to
pin more than `--max-fraction` (default 0.5) of MemTotal, and it raises
`RLIMIT_MEMLOCK` itself (the pod's default `ulimit -l` is 64 KB; root has
CAP_IPC_LOCK).

## Rule

- Before building a cache, check whether the kernel already is one. Here the
  page cache was doing the job and the gap was residency, not caching — one
  `mlock` call, not a filesystem.
- A residency claim is only proven by eviction. `VmLck` shows intent;
  `drop_caches` + a timed re-read shows it survived. Measure the second one.
- The pin dies with the container. Re-run it after any pod restart — it is the
  first command of a session, before sync and build, so the one-time cost
  overlaps setup.
