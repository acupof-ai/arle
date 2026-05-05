# 2026-05-05 · KV-tier persistence substrate ported from Zig to pure Rust

## Status: pending-remote

This bench entry is a **stub**. The `kv-native-sys` crate is POSIX-only and
the migration host (Windows MSVC) cannot compile it. The bench rerun will
land once a Linux or macOS runner picks the work up.

## Context

`crates/kv-native-sys` was a thin Rust FFI shim over a Zig source file
(`crates/kv-native-sys/zig/src/kv_native.zig`, 948 lines) that wrapped
libc / `<sys/mman.h>` calls plus a single bump-pointer arena. The Zig
substrate had no comptime, no Zig-unique features — it was effectively
a `@cImport` of libc + a `std.ArrayListUnmanaged`-backed free-list. The
external Zig toolchain dependency was paid in CI complexity
(`scripts/setup_zig_toolchain.sh`, `scripts/check_kv_zig.sh`,
`.toolchains/zig` cache, four CI jobs each invoking the validator) for
no measurable performance win — the prediction at the time of the port
plan was ±2% on syscall-bound paths with a micro-win on FFI elimination.

The migration landed in 6 sequential commits on `main` on 2026-05-05:

1. `refactor(kv-tier): port file/block I/O from Zig to Rust`
2. `refactor(kv-tier): port WAL append/replay from Zig to Rust`
3. `refactor(kv-tier): port file mmap from Zig to Rust`
4. `refactor(kv-tier): port POSIX shm from Zig to Rust`
5. `refactor(kv-tier): port host arena from Zig to Rust`
6. `refactor(kv-tier): purge Zig substrate after Rust port`

The post-port crate uses `nix = 0.31` (`fs`, `mman`), `memmap2 = 0.9`,
and `libc = 0.2` directly. No external toolchain.

## Bench plan (to be executed remotely)

A/B against the [2026-05-04 baseline](2026-05-04-bench-kv-tier-copy-throughput.md):

```bash
cargo test --release --no-default-features --features no-cuda \
    -p infer --lib bench_kv_tier_copy_throughput -- --ignored --nocapture
```

Same command, same instrument
(`infer/src/kv_tier/coordinator/bench.rs`), same hardware envelope
(Sapphire Rapids 4-core VM or equivalent Linux runner). Compare:

- **T1→T2 spill** throughput (host arena → `DiskStore`).
- **T2→T1 readmission** throughput (`DiskStore::get_block` →
  `HostPinnedPool::reserve` + memcpy).
- **`madvise(MADV_HUGEPAGE)` no-op assertion** still holds on the same
  VM (the 2026-05-04 baseline showed THP did not promote on Sapphire
  Rapids hosts where THP is set to `madvise`; the Rust port replicates
  the same advisory call with the same silent-fallback semantics).

## Expected numbers

±2% on syscall-bound paths; no regression on the L3→DRAM path that
dominated the 2026-05-04 baseline. The FFI-elimination win
(`read_block_owned` no longer crosses an FFI boundary, the buffer is a
`Box<[u8]>` directly) is below the VM's variance floor — count it as
an architectural-cleanup win, not a perf win, unless the bench shows
otherwise.

## Hypothesis

Migrating to pure Rust is **net neutral on perf** and **strictly
positive on operational complexity**: drops 948 lines of Zig, the
Zig 0.16.0 toolchain pin, two scripts, four CI cache entries, and one
build.rs that spawned an external compiler.

## Rule

When a Zig (or other-language) FFI substrate has no comptime / no
language-unique features and exists purely to wrap libc, port it back
to Rust the moment the Rust ecosystem (`nix`, `memmap2`, `libc`) covers
the surface. The toolchain savings outweigh the marginal-or-zero perf
cost.
