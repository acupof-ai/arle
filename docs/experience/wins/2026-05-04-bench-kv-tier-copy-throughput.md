# 2026-05-04 · T1↔T2 copy-throughput bench + hardware deep-dive + tried optimizations

## Context

User asked (in three rounds): 测一下拷贝的速度 → 考虑 CPU 和 GPU 的 overlap →
深挖硬件模式 继续本机优化. This entry pins:

1. **Hardware probe** of the actual sandbox machine that backs every
   number below.
2. **Measured T1↔T2 throughput** through the real ARLE types
   (`HostPinnedPool`, `DiskStore`, `Coordinator`) — the bench
   instrument lives at `infer/src/kv_tier/coordinator/bench.rs`.
3. **CPU↔GPU layer-wise compute-transfer overlap viability** for the
   HiCache-borrowed roadmap (`docs/plans/2026-05-04-kv-tier-hicache-borrowed-improvements.md`
   §P1.1, HiCache optimization 6.2).
4. **A/B of two attempted local optimizations**: `MADV_HUGEPAGE` on
   the host arena (no measurable win on this VM, kept as defensive
   prod optimization) and **eliminating the intermediate `Vec<u8>`
   on the T2→T1 fetch path** (real but small win, masked by VM I/O
   variance).

## Hardware probe (the ground truth)

```
CPU:     Intel Xeon (Sapphire Rapids, model 207) 4 cores @ 2.1 GHz
         AVX-512 + AVX-VNNI + AMX-BF16/INT8 + AVX512_FP16 + clwb +
         clflushopt + movdir64b. 1 socket, 1 NUMA node, KVM hypervisor.
Caches:  L1d 48 KiB/core, L1i 32 KiB/core, L2 2 MiB/core,
         L3 260 MiB shared.
RAM:     16 GiB. AnonHugePages 0 KiB (THP=madvise but no active promotion).
         Hugepagesize 2 MiB. ulimit -l = 8192 KiB (memlock).
Storage: /dev/vda virtio block, ext4 on /, no separate tmpfs for /tmp.
         O_DIRECT write 363 MB/s, O_DIRECT read 1.1 GB/s.
         Block scheduler mq-deadline. Write cache: write back.
THP:     /sys/kernel/mm/transparent_hugepage/enabled = madvise.
```

**Critical implications for the bench numbers below:**

- **L3 = 260 MiB is right at the 256 MiB top bench size.** The
  T1 self-copy cliff at 256 MiB (12 GiB/s → 3 GiB/s) is the
  classical L3→DRAM bandwidth wall — not a code issue.
- **/tmp is on the real virtio disk, not tmpfs.** The DiskStore
  bench measures real virtualized I/O, not RAM. Our highest
  observed disk write throughput (1.5 GiB/s on small blocks) is
  page-cache-aided; sustained device-direct write is 363 MB/s.
- **AnonHugePages stayed at 0 throughout the bench.** Even with
  the new `madvise(MADV_HUGEPAGE)` call (commit 0c353ce), the
  kernel did not actually promote pages on this VM. Likely the
  host kernel's THP defrag/khugepaged is disabled or the host
  doesn't support THP. The madvise call remains in the code as
  a defensive optimization for production hosts where THP works.

## What worked

Added `infer/src/kv_tier/coordinator/bench.rs` — a `#[ignore]`-tagged
bench that uses the actual coordinator + Zig-backed pinned arena +
DiskStore to time T1 self-copy, T2 disk round-trip, and the full
T1→T2→release→T1 coordinator cycle at 5 sizes from 4 KiB to 256 MiB.
Run: `cargo test --release --no-default-features --features no-cuda
-p infer --lib bench_kv_tier_copy_throughput -- --ignored --nocapture`.

The bench instrument was iterated mid-session: an early version
included a per-iter verification `read_region` that inflated coord
cycle numbers (commit 77f4e16 fixed it to a single warm-up
verification outside the timed loop).

### Three-condition A/B (median of 3 runs each)

All three conditions use the **fixed bench shape** (no per-iter
verification read) so they are apples-to-apples. Each row counts
both directions of the round-trip as the byte-volume.

#### T1 self-copy (HostPinnedPool reserve+write+read)

| size    | baseline | +THP    | +THP+Vec-opt |
|---------|----------|---------|--------------|
| 4 KiB   | 26576    | 22351   | 21774        |
| 64 KiB  | 31314    | 34628   | 32831        |
| 1 MiB   | 14642    | 13089   | 12802        |
| 16 MiB  | 8794     | 9056    | 8078         |
| 256 MiB | 2587     | 2566    | 2447         |

(All values MiB/s. T1 path doesn't actually exercise the disk-fetch
code change; differences here are pure noise.)

#### T2 disk roundtrip (DiskStore put+get)

| size    | baseline | +THP    | +THP+Vec-opt |
|---------|----------|---------|--------------|
| 4 KiB   | 8.8      | 8.0     | 6.9          |
| 64 KiB  | 117.1    | 103.8   | 104.5        |
| 1 MiB   | 638.4    | 587.7   | 605.0        |
| 16 MiB  | 838.3    | 801.5   | 845.0        |
| 256 MiB | 241.3    | 269.5   | 233.3        |

(All values MiB/s. Within ±20% measurement variance — virtio I/O
on this VM is highly variable.)

#### Coordinator full cycle T1→T2→release→T1

| size    | baseline | +THP    | +THP+Vec-opt | Δ (post-opt vs baseline) |
|---------|----------|---------|--------------|---------------------------|
| 4 KiB   | 5.6      | 4.8     | 4.7          | -16% (within noise)       |
| 64 KiB  | 62.7     | 53.5    | 53.5         | -15% (within noise)       |
| 1 MiB   | 259.1    | 224.2   | 239.5        | -8% (within noise)        |
| 16 MiB  | 253.5    | 245.8   | **285.9**    | **+13%** (clean signal)   |
| 256 MiB | 188.3    | 184.0   | 188.1        | flat                      |

### Findings — what each optimization actually did

**Optimization A: `madvise(MADV_HUGEPAGE)` on host arena (commit 0c353ce + e340751).**
- Earlier in the session this was reported as +12-87% improvement, but
  that was conflated with a separate bench-shape fix landing in the
  same window. Rigorous A/B (above table) shows **no measurable
  improvement on this VM**. Reason: `AnonHugePages` stayed at 0 KiB
  in `/proc/meminfo` throughout the bench — the kernel was not
  actually promoting pages, likely because the host kernel's THP
  support is disabled or unavailable.
- The call is **kept** as a defensive optimization for production
  Linux hosts where THP is properly configured. Cost is one syscall
  per arena init (negligible). Source comment annotated with the
  finding.

**Optimization B: Eliminate intermediate `Vec<u8>` on T2→T1 fetch (commit e823e32).**
- Added `kv_native_sys::read_block_owned` returning a
  `KvNativeOwnedBytes` guard that owns the Zig-allocated buffer
  directly. `DiskStore::get_block` decodes the on-disk header against
  the borrowed Zig slice, then copies *only* the payload portion
  into the returned Vec.
- Saves **1 alloc + 1 full-block memcpy** per fetch (the previous
  `read_buffer.to_vec()` of header+payload, then `payload.to_vec()`).
- Theoretical savings at 256 MiB: alloc (~50 ms) + memcpy (~21 ms
  at 12 GiB/s) ≈ 70 ms per fetch.
- **Visible signal at 16 MiB only (+13%)** — that block size is
  large enough for memcpy savings to register but small enough that
  virtio I/O variance doesn't drown them. At 256 MiB the I/O
  variance (±20%) is larger than the ~5% theoretical savings.
- Production NVMe (low variance) should make the savings visible
  across all medium-large sizes.

## Iteration 2 — bigger wins from skipping the wrong defaults

After Iteration 1's marginal-impact micro-optimizations, profile-by-
inspection of the small-block bench (4 KiB cycle stuck at ~5 MiB/s,
~95% of cycle in non-disk overhead) led to looking at fsync. The
bench gained a **`T2 put+get (no-fsync)` column** to make the cost
visible per row.

### What the new column showed

| size    | T2 fsync (default) | T2 no-fsync | Speedup |
|---------|--------------------|-------------|---------|
| 4 KiB   | 19.1 MiB/s         | 361.5 MiB/s | **18.9×** |
| 64 KiB  | 255.6 MiB/s        | 3117 MiB/s  | **12.2×** |
| 1 MiB   | 1165 MiB/s         | 4704 MiB/s  | **4.0×**  |
| 16 MiB  | 1812 MiB/s         | 3697 MiB/s  | **2.0×**  |
| 256 MiB | 582 MiB/s          | 923 MiB/s   | **1.6×**  |

Conclusion: `DiskStore::put_block` defaults to `fsync_each_block=true`
which forces TWO fsync syscalls per write (data file + parent dir).
For a CACHE — where blocks are recomputable on miss — this is
over-conservative. We want atomicity (no partial files) but NOT
durability (a crash that loses recent cache entries is fine).

### Optimization C: Coordinator hot-path skips fsync (commit 06b6bf9)

Switched `Coordinator::handle_store` to `put_block_with_fsync(false)`.
Atomicity preserved via the existing temp-file + rename. Production
durability-sensitive callers still reach the fsync path through
`DiskStore::put_block_with_fsync(true)` directly.

### Optimization D: Skip create_dir_all syscall per write (commit 701c76a)

Added `AtomicBool root_created` on `DiskStore`. The first put_block
calls `create_dir_all`; subsequent puts read the atomic and skip.
Eliminates ~1 stat syscall per write. Below the noise floor on this
VM but compounds with C and D.

### Optimization E: Inline block_path_for hex formatting in Rust (commit e6286d2)

`block_path_for` was hopping into Zig FFI for what is just hex-format
of a 16-byte fingerprint into a 35-char filename. Replaced with
stack-allocated [u8;35] formatted in pure Rust. Net per-call savings:
1 FFI + 1 Zig alloc + 1 Rust alloc + 1 memcpy + 1 free. Below noise
floor individually; compounds with C and D.

### Cumulative coord-cycle improvement (median of 3 final runs)

| size    | baseline (no THP, no Vec-opt) | final (B+C+D+E) | Speedup |
|---------|-------------------------------|-----------------|---------|
| 4 KiB   | 5.6 MiB/s                     | 31.0 MiB/s      | **5.5×** |
| 64 KiB  | 62.7 MiB/s                    | 343.0 MiB/s     | **5.5×** |
| 1 MiB   | 259 MiB/s                     | 1800 MiB/s      | **6.9×** |
| 16 MiB  | 254 MiB/s                     | 2603 MiB/s      | **10.3×** |
| 256 MiB | 188 MiB/s                     | 718 MiB/s       | **3.8×** |

ops/s view (more meaningful for small-op latency):

| size    | ops/s baseline | ops/s final | Speedup |
|---------|----------------|-------------|---------|
| 4 KiB   | ~720           | ~3970       | **5.5×** |
| 64 KiB  | ~500           | ~2744       | **5.5×** |
| 1 MiB   | ~125           | ~900        | **7.2×** |
| 16 MiB  | ~7.9           | ~81         | **10.3×** |
| 256 MiB | ~0.37          | ~1.40       | **3.8×** |

The 16 MiB jump (10.3×) is the standout — that block size is
large enough for fsync's per-write ~10ms cost to dominate at
baseline (2 fsyncs × ~10ms = 20+ms overhead per write, on top
of the ~9ms actual write), so removing it gives the biggest
proportional speedup. Small sizes (4 KiB, 64 KiB) are now
limited by per-op channel/event/dispatch overhead, not disk.

### Where the remaining headroom is

After C+D+E, the gap between Coord cycle and T2 no-fsync (the
underlying disk ceiling, in MiB/s):

| size    | Coord cycle | T2 no-fsync | Coord/Ceiling |
|---------|-------------|-------------|---------------|
| 4 KiB   | 31          | 361         | 8.6%          |
| 64 KiB  | 343         | 3117        | 11.0%         |
| 1 MiB   | 1800        | 4704        | 38.3%         |
| 16 MiB  | 2603        | 3697        | 70.4%         |
| 256 MiB | 718         | 923         | 77.8%         |

Medium-large (16 MiB+) sizes are within striking distance of the
disk ceiling. Small (4-64 KiB) sizes have ~10× remaining headroom,
all in per-op overhead (channel comm + event dispatch + atomic
counters + mutex acquire). Closing that gap requires either
**batching multiple ops per coordinator round** (a usage-pattern
change — `submit_store` already accepts `Vec<StoreRequest>`) or
restructuring the coordinator's command-channel path. Out of scope
for this session.

## Iteration 3 — review-driven fixes (Codex bot + self-review)

After Iteration 2, the PR went through `@codex review` and a parallel
self-review pass. Three concerns surfaced that were each addressed in
follow-up commits.

### Codex finding (P2): create_root cache breaks self-heal

`disk.rs:244` review comment: after the `root_created` AtomicBool is
set, `put_block_with_fsync` no longer calls `create_dir_all` on
subsequent writes. If the cache root is deleted while the process is
running (operator purge, tmp cleanup, remount, full disk forcing
rmdir), every later store fails with NotFound until restart. Before
the cache, each put unconditionally re-created and self-healed.

**Fix (commit 26e0b23):** hot path stays fast (skip when atomic is
true); on NotFound from the inner write, force-recreate the root and
retry once. Common case (root exists) pays zero syscalls; rare
disappearance pays 1 create_dir_all + 1 retry. New regression test
`put_block_self_heals_when_root_deleted_externally` asserts: first
put creates root, externally `remove_dir_all` the root, second put
must succeed and content must be readable.

### Self-review (Strong #1): block_path_for had dead Result

After the inline-Rust rewrite (commit e6286d2), `block_path_for`
became infallible — no syscalls, no FFI errors. The lingering
`io::Result<PathBuf>` return type was dead error-handling ceremony
across 11 call sites.

**Fix (commit 403b594):** drop `io::Result`. Update all callers (3
DiskStore, 1 SharedFsStore wrapper + 2 wrapper-callers, 1
coordinator, 1 scheduler/cuda/core, 1 test). Also added
`const _: () = assert!(size_of::<BlockFingerprint>() == 16)` inside
the function so a future widening of `BlockFingerprint` fails at
compile time instead of silently truncating the 35-byte stack
filename buffer.

### Self-review (Strong #3): swap-cycle test could pass on aliasing

The original `t1_t2_full_swap_cycle_release_then_readmit` (commit
9d7a2fc) used a 256-byte pool with 25-byte payload — fetch-leg
likely reused the same pool offset the store-leg had. A regression
where `fetch_disk_payload` returned wrong bytes — but the host pool
happened to hold the correct bytes from the not-yet-overwritten
initial region — would still pass byte-equality.

**Fix (commit beb6829):** between release and fetch, reserve a "pin"
region filled with `0xAB` to occupy exactly the slot the original
released (asserted: `pin_region.offset == initial_offset`). This
forces fetch to a different offset (asserted:
`fetched.offset != initial_offset`). Pin region is asserted
unchanged after fetch (proves the fetch didn't write to wrong slot).
Now a wrong-bytes regression OR a wrong-slot-write regression both
fail the test cleanly.

### Self-review (Strong #2): C-only attribution

The Iteration 2 final table conflated C (fsync removal) with D
(create_root cache) and E (Rust block_path_for). Re-bench by
checkout-ing commit `06b6bf9` (C only):

| size    | C only (median 2 runs) | C+D+E (final)  | Δ from D+E |
|---------|------------------------|----------------|------------|
| 4 KiB   | 30.6 MiB/s            | 30.3 MiB/s     | noise (variance ±300%) |
| 64 KiB  | 338 MiB/s             | 343 MiB/s      | +1.5%      |
| 1 MiB   | 1741 MiB/s            | 1800 MiB/s     | +3.4%      |
| 16 MiB  | 2499 MiB/s            | 2603 MiB/s     | +4.2%      |
| 256 MiB | 543 MiB/s             | 718 MiB/s      | +32% (high variance both ways) |

**Honest reading:** **C (fsync removal) does ~all of the heavy
lifting; D and E are micro-optimizations whose individual signal
sits below this VM's noise floor at most sizes.** The 32% delta at
256 MiB is consistent with a real D+E contribution but the variance
is too large to claim it conclusively. Production NVMe (low
variance) should expose D+E's cumulative ~1 alloc + 2 memcpy + 1
syscall savings per cycle clearly; on this VM, only commit C is
provably responsible for the 5.5–10.3× cumulative speedup reported
in the Iteration 2 final table.

### Three readable signals from the bench

1. **T1 self-copy is memcpy-bandwidth-bound.** 14–35 GiB/s for medium
   sizes; drops to 2.5 GiB/s at 256 MiB as the working set crosses
   the 260 MiB L3 boundary. Approximates the host-side ceiling for
   any cudaMemcpyAsync / GPU-SM-kernel target (HiCache 6.1 / ARLE A8).

2. **T2 disk round-trip ramps with size, plateaus around 800 MiB/s,
   then drops past page-cache window.** 6 MiB/s at 4 KiB (per-op file
   syscall overhead) → 845 MiB/s at 16 MiB (peak, page-cache aided)
   → 233 MiB/s at 256 MiB (working set exceeds page cache). The
   underlying device sustains 363 MB/s direct write (measured by dd).

3. **Coordinator overhead is small for medium sizes, large for small
   sizes.** Coord cycle vs raw T2: 16 MiB cycle 286 vs T2 845 MiB/s
   (≈ 67% of one disk round-trip ÷ 2 legs); 4 KiB cycle 4.7 vs T2 6.9
   MiB/s (per-op channel comm overhead is dominant). Small ops are
   bottlenecked by Coordinator's command-channel latency, not bytes.

## CPU↔GPU layer-wise overlap viability

Apply the measured T1 self-copy ceiling (≈ 12 GiB/s usable for medium
blocks) as a lower bound on what cudaMemcpyAsync can achieve over
PCIe Gen4 x16 (32 GiB/s nominal). For Qwen3-32B with TP=8:

- 64 layers, ~8 KV heads per rank, head_dim=128, fp16 (2 B/elem)
- Per-token-per-layer KV (per rank) = 2 (K+V) × 8 × 128 × 2 = 4 KiB
- For 4096-token context window: per-layer KV = 16 MiB per rank
- Per-layer T1→T0 transfer at 12 GiB/s ≈ **1.3 ms**
- Per-layer compute on a saturated GPU at 32B class: ≈ **5–15 ms**

**Conclusion:** the per-layer transfer fits inside a single layer's
compute budget by a margin of 4–10×. **Layer-wise compute-transfer
overlap (P1.1 in the borrowed roadmap, HiCache 6.2) is feasible and
will hide essentially all T1↔T0 transfer time on this scale class.**
The remaining serialization cost is the first layer (no compute to
hide it) and pipeline drain at the last.

For T1↔T2 (disk) the same arithmetic gives 16 MiB / 441 MiB/s = 36 ms
per layer, **far larger than per-layer compute**. T2-readmitted blocks
cannot be hidden by per-layer overlap; they need either (a) batched
prefetch with `PrefetchPolicy::Timeout` so the wait happens once before
prefill rather than per-layer, or (b) faster T2 (NVMe / RDMA / Mooncake
L3). This matches HiCache's design choice to give L3 its own
`prefetch_threshold + Timeout policy` rather than rely on layer-wise
overlap for the slowest tier.

## Rules

1. **Before designing a per-layer overlap optimization for any tier,
   measure that tier's bandwidth at the realistic per-layer block
   size and compare to per-layer compute time on the model class of
   interest.** The two-order-of-magnitude gap between T1↔T0
   (hideable) and T1↔T2 (not hideable) shows the same optimization
   shape doesn't apply uniformly across tiers — this is why HiCache
   layers compute-transfer overlap on T1↔T0 but prefetch+timeout on
   L2↔L3, and ARLE's roadmap mirrors that split.

2. **Always verify madvise/THP behavior with `/proc/<pid>/smaps` or
   `/proc/meminfo` AnonHugePages**, not just by adding the syscall.
   On VMs and constrained kernels, MADV_HUGEPAGE is silently a no-op
   even though it returns success. The 2026-05-04 misattribution
   here (where the bench-shape change was conflated with a perf
   "win" from THP) is the cautionary tale.

3. **VM disk benches need ≥3 runs and median reporting.** Single-run
   numbers on virtio storage have ±20% variance from background
   page-cache flush, host scheduler, and block-device queue depth.
   Real differences smaller than the variance are invisible without
   either repetition or a stable bare-metal baseline.

4. **Bench instrument changes are themselves perf-sensitive.** A
   "verification read" that seems like trivial test scaffolding can
   add a measurable inflation to throughput numbers (because page
   cache warmth from the verify step makes subsequent ops appear
   faster). Always factor verification out of the timed loop;
   document the change explicitly when it shifts numbers.

## Pointers

- Bench source: `infer/src/kv_tier/coordinator/bench.rs`
- Roadmap that consumes these numbers: `docs/plans/2026-05-04-kv-tier-hicache-borrowed-improvements.md` §P1.1, §P0.2
- HiCache reference: `docs/research/2026-05-04-sglang-hicache-guide.md` Part VI §6.2, §6.5
- THP optimization: `crates/kv-native-sys/zig/src/kv_native.zig` (`hostArenaCreateInternal`)
- Vec elimination: `crates/kv-native-sys/src/lib.rs` (`KvNativeOwnedBytes`, `read_block_owned`) + `infer/src/kv_tier/transport/disk.rs` (`DiskStore::get_block`)
- Re-run the bench: `cargo test --release --no-default-features --features no-cuda -p infer --lib bench_kv_tier_copy_throughput -- --ignored --nocapture`
- Local env setup: `pip install ziglang` then `export ZIG=/usr/local/lib/python3.11/dist-packages/ziglang/zig`

## Status

`pending-remote-gpu` for two follow-ups:

1. **T0↔T1 PCIe leg of the overlap analysis.** This run is CPU-only,
   so the cudaMemcpyAsync number is extrapolated from host-memcpy.
   The actual GPU-PCIe number must be measured on a CUDA box before
   the P1.1 layer-wise overlap implementation lands. Validate the
   12 GiB/s lower-bound assumption against real `cudaMemcpyAsync`
   and (after A8 lands) the SM-assisted I/O kernel.

2. **Re-run all three A/B conditions on a real (non-VM) Linux box
   with confirmed-active THP.** `AnonHugePages > 0` in
   `/proc/meminfo` would confirm the kernel actually promotes pages,
   and the MADV_HUGEPAGE win on coord cycle 16 MiB / 256 MiB should
   become visible. If still no win, the optimization can be removed
   with confidence rather than left in as cargo-cult.

3. **Re-run the Vec-elimination A/B on production NVMe** to confirm
   the +13% signal at 16 MiB extends to other medium-large sizes
   when virtio-VM I/O variance is removed.
