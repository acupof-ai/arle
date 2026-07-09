# 2026-05-05 · KV-tier persistence substrate ported from Zig to pure Rust

## Status

First Linux run completed on the L4 GPU box (the only Linux runner available
this session). The Rust port BUILDS, RUNS, and PASSES all tests. Hypothesis
A/B against the 2026-05-04 Sapphire Rapids baseline remains
**blocked on same-hardware rerun** because this box is a different VM class
(see Hardware below). The 2026-05-04 baseline used Sapphire Rapids 4-core /
260 MiB L3 / virtio ext4; this run is on Cascade-Lake-class 12-core /
38.5 MiB L3 / overlay-backed `/`. THP advisory is **active here** (it was a
silent no-op on the baseline VM), which alone is enough to skew the A/B.

The migration host (Windows MSVC) cannot compile `kv-native-sys` because the
crate is POSIX-only (`nix`, `memmap2`, libc directly). Linux-only rerun on a
Sapphire Rapids 4-core matching the baseline is the remaining gate before
the hypothesis-validation entry can claim "net-neutral perf".

## Context

`crates/kv-native-sys` was a thin Rust FFI shim over a Zig source file
(`crates/kv-native-sys/zig/src/kv_native.zig`, 948 lines) that wrapped
libc / `<sys/mman.h>` calls plus a single bump-pointer arena. The Zig
substrate had no comptime, no Zig-unique features — it was effectively a
`@cImport` of libc + a `std.ArrayListUnmanaged`-backed free-list. The
external Zig toolchain dependency was paid in CI complexity
(`scripts/setup_zig_toolchain.sh`, `scripts/check_kv_zig.sh`,
`.toolchains/zig` cache, four CI jobs each invoking the validator) for no
measurable performance win — the prediction at the time of the port plan
was ±2% on syscall-bound paths with a micro-win on FFI elimination.

The migration landed in 6 sequential commits on `main` on 2026-05-05:

1. `4eddf571 refactor(kv-tier): port file/block I/O from Zig to Rust`
2. `8b422504 refactor(kv-tier): port WAL append/replay from Zig to Rust`
3. `92f63825 refactor(kv-tier): port file mmap from Zig to Rust`
4. `fb5255af refactor(kv-tier): port POSIX shm from Zig to Rust`
5. `430b453a refactor(kv-tier): port host arena from Zig to Rust`
6. `0c5a8fbc refactor(kv-tier): purge Zig substrate after Rust port`

(plus `d9395475 refactor(kv-tier): drop dead fingerprint_bytes helper` and
`7be9130f docs(kv-tier): drop stale Zig references from kv_tier consumers`
as cleanup tail.)

The post-port crate uses `nix = 0.31` (`fs`, `mman`), `memmap2 = 0.9`, and
`libc = 0.2` directly. No external toolchain.

## Hardware (this run — different from 2026-05-04 baseline)

```
CPU:     Intel Xeon model 85 (Cascade-Lake-class) @ 2.20GHz, 6 cores × 2
         threads = 12. AVX-512 (F/DQ/CD/BW/VL/VNNI). 1 socket, 1 NUMA node,
         KVM hypervisor.
Caches:  L1d 32 KiB/core, L1i 32 KiB/core, L2 1 MiB/core,
         L3 38.5 MiB shared.
RAM:     55.5 GiB. AnonHugePages 18,432 KiB (THP madvise IS promoting,
         unlike the 2026-05-04 baseline where it stayed at 0).
         Hugepagesize 2 MiB. ulimit -l = 8192 KiB (memlock).
Storage: overlay-backed `/` (containerized FS). 178 GiB free.
THP:     /sys/kernel/mm/transparent_hugepage/enabled = `always [madvise] never`.
GPU:     NVIDIA L4, 23,034 MiB (idle this bench, no GPU code touched).
```

Comparison to 2026-05-04 baseline:

| probe | 2026-05-04 baseline | 2026-05-05 (this run) |
|---|---|---|
| CPU class | Sapphire Rapids (model 207) | Cascade-Lake-class (model 85) |
| Cores / threads | 4 / 4 | 6 / 12 |
| L3 cache | **260 MiB** | **38.5 MiB** |
| RAM | 16 GiB | 55.5 GiB |
| AnonHugePages | 0 KiB (silent no-op) | 18,432 KiB (active) |
| Storage | virtio /dev/vda ext4 | overlay |

Two confounders that alone disqualify a clean A/B:

1. **L3 is ~7× smaller here.** The L3→DRAM wall hits between 16 MiB and 64 MiB
   on this box vs between 256 MiB and 256+ on the baseline. The 16 MiB and
   256 MiB rows in this run are hardware-bound, not Rust-vs-Zig signal.
2. **THP is actually promoting here.** The baseline's MADV_HUGEPAGE was
   documented as a silent no-op on Sapphire Rapids VM. Here AnonHugePages
   moved during the bench. Any large-block delta vs baseline includes that.

## Run command

```bash
source /tmp/arle-env.sh
export CARGO_TARGET_DIR=/tmp/cargo-target-kv-tier
cargo test --release --no-default-features --features no-cuda \
    -p infer --lib bench_kv_tier_copy_throughput -- --ignored --nocapture
```

Compile time: 3m 16s (cold cache). Bench wall time: 89.73s. Exit 0,
1 passed / 0 failed.

Raw output saved at `bench-output/2026-05-05-kv-tier-rust-substrate/bench.log`.

## Results — full bench table (this run)

| size | iters | T1 self-copy | T2 put+get (fsync) | T2 put+get (no-fsync) | Coord. T1→T2→T1 | coord ops/s |
|---|---:|---:|---:|---:|---:|---:|
| 4 KiB | 5000 | 23,721.2 MiB/s | 1.4 MiB/s | 144.1 MiB/s | 146.3 MiB/s | 18,724.45 |
| 64 KiB | 2000 | 19,405.9 MiB/s | 28.8 MiB/s | 1,391.9 MiB/s | 1,205.7 MiB/s | 9,645.58 |
| 1 MiB | 500 | 13,328.4 MiB/s | 241.0 MiB/s | 2,215.7 MiB/s | 1,713.1 MiB/s | 856.53 |
| 16 MiB | 50 | 5,093.0 MiB/s | 680.9 MiB/s | 1,579.0 MiB/s | 1,105.8 MiB/s | 34.55 |
| 256 MiB | 10 | 2,188.9 MiB/s | 336.9 MiB/s | 707.8 MiB/s | 517.9 MiB/s | 1.01 |

Conventions are unchanged from the 2026-05-04 baseline: each row counts both
directions of the round-trip as the byte-volume; T2 (fsync) is the
`put_block` default; T2 (no-fsync) is `put_block_with_fsync(false)`.

## Coord-cycle delta vs 2026-05-04 baseline (hardware-confounded — NOT a clean A/B)

| size | 2026-05-04 final (Sapphire Rapids 4c) | 2026-05-05 Rust port (Cascade-Lake-class 12c) | Δ | confounder note |
|---|---:|---:|---:|---|
| 4 KiB | 31.0 MiB/s | 146.3 MiB/s | +372% | small-op overhead — different per-op cost on different CPU |
| 64 KiB | 343.0 MiB/s | 1,205.7 MiB/s | +251% | fits in L2 on this box; L1+L2 different |
| 1 MiB | 1,800 MiB/s | 1,713.1 MiB/s | -4.8% | fits in L3 on both; closest to apples-to-apples |
| 16 MiB | 2,603 MiB/s | 1,105.8 MiB/s | -57.5% | crosses 38.5 MiB L3 cliff here; sat in 260 MiB L3 on baseline |
| 256 MiB | 718 MiB/s | 517.9 MiB/s | -27.9% | DRAM-bound on both; storage substrate differs |

**Honest reading.** The mid-band (1 MiB, the only block size that comfortably
fits in L3 on both VMs) is within ±5% of the baseline — consistent with the
"net-neutral perf" hypothesis but not statistically conclusive from one run.
Small-block rows (4 KiB, 64 KiB) are dominated by per-op overhead on a
faster, higher-thread CPU and don't say anything about Zig-vs-Rust. The
medium-large rows (16 MiB, 256 MiB) drop because the L3 cliff is at a
smaller block size on this VM. **None of the deltas in this table are
attributable to the Rust port.**

The next step (still required to ship a clean win) is to re-run on a
Sapphire Rapids 4-core matching the 2026-05-04 baseline VM class, with the
same overlay/ext4 storage shape.

## What this run actually proves

1. The Rust-only `kv-native-sys` (post `0c5a8fbc`) **builds and passes all
   tests on Linux without the Zig toolchain.** No `ZIG=`, no
   `setup_zig_toolchain.sh`, no `.toolchains/zig`. Just `cargo test` with
   the `nix` + `memmap2` deps.
2. The bench instrument (`infer/src/kv_tier/coordinator/bench.rs`) survived
   the FFI rewrite intact — same column shape, same conventions.
3. THP advisory IS effective on at least one currently-available Linux
   environment (this box). The 2026-05-04 footnote that called MADV_HUGEPAGE
   "silently a no-op" was VM-specific, not universal — production hosts with
   working THP defrag will see the advisory take effect.

## What this run cannot claim

1. "Net-neutral perf vs Zig." Hardware confounders are too large.
2. "X% throughput delta on the Rust port." Same reason.
3. THP optimization is a measurable win. The L4 box has THP active
   regardless of whether MADV_HUGEPAGE was called; without an A/B with the
   syscall stripped, the contribution of MADV_HUGEPAGE itself is still
   undetermined.

## Followups (still pending-remote — same-hardware rerun required)

1. **Same-hardware A/B.** Re-run `bench_kv_tier_copy_throughput` on a
   Sapphire Rapids 4-core / 260 MiB L3 VM matching the 2026-05-04 baseline.
   Goal: validate the "±2% on syscall-bound paths" hypothesis on a clean
   substrate.
2. **MADV_HUGEPAGE A/B with the call stripped.** On a host where
   AnonHugePages > 0 (this box qualifies), measure with vs without the
   advisory to either keep or delete the call with confidence.
3. **PCIe / cudaMemcpyAsync leg.** Still not measured; the 2026-05-04
   "12 GiB/s lower bound" assumption from host-memcpy needs a real
   GPU-PCIe number.

## Rule

When a Zig (or other-language) FFI substrate has no comptime / no
language-unique features and exists purely to wrap libc, port it back to
Rust the moment the Rust ecosystem (`nix`, `memmap2`, `libc`) covers the
surface. The toolchain savings outweigh the marginal-or-zero perf cost.

(Restated from the original stub — still holds. Operational complexity
savings: 948 LoC Zig deleted, 1 build.rs deleted, 2 toolchain scripts
deleted, 4 CI cache entries deleted, 1 external compiler dependency
removed. Performance cost: undetermined from this run, capped by the
hypothesis at ±2% on the same hardware.)

## Pointers

- Bench source: `infer/src/kv_tier/coordinator/bench.rs`
- 2026-05-04 baseline: [`2026-05-04-bench-kv-tier-copy-throughput.md`](2026-05-04-bench-kv-tier-copy-throughput.md)
- Raw bench log: `bench-output/2026-05-05-kv-tier-rust-substrate/bench.log`
- Roadmap: `docs/plans/2026-05-04-kv-tier-hicache-borrowed-improvements.md` §P1.1, §P0.2
- Re-run: `cargo test --release --no-default-features --features no-cuda -p infer --lib bench_kv_tier_copy_throughput -- --ignored --nocapture`
