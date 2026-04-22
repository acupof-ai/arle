# KV Native Host Arena O(1) Reserved-Bytes Accounting

## Goal

- Optimization: remove host-arena bookkeeping overhead on the Zig-backed T1
  host pool by making `host_arena_reserved_bytes()` O(1) and by collapsing
  reverse-order tail releases back into the allocation frontier.

## Hypothesis

- `host_arena_reserved_bytes()` is currently dominated by a linear scan of the
  fragmented free list, so maintaining `reserved_bytes` on the arena should cut
  the query to near-constant-time noise.
- Rewinding the frontier on tail releases should also keep the free list
  smaller under reverse-order lifetimes without changing allocation stability.

## Command

Baseline before the optimization:

```bash
cargo test -p kv-native-sys --release \
  tests::bench_host_arena_reserved_bytes_fragmented \
  -- --ignored --exact --nocapture
```

Validation after the optimization:

```bash
cargo test -p kv-native-sys host_arena --release
```

```bash
cargo test -p kv-native-sys --release
```

```bash
cargo test -p kv-native-sys host_arena_bench --release -- --ignored --nocapture
```

## Environment

- Host: `Apple M4 Pro`, `48 GB`
- OS: `macOS 26.3.1 (a)`
- Arch: `Darwin arm64`
- Zig: `0.16.0`
- Workspace base: `7d458a9`
- Dirty scope under test: `crates/kv-native-sys/{zig/src/kv_native.zig,src/lib.rs}`
- Build mode: `cargo test --release`

## Results

Microbench workload:

- arena capacity: `8192 * 64 = 524288` bytes
- live holes after fragmentation: `4096`
- timed queries: `5000000`
- warmup queries: `100000`
- expected reserved bytes: `262144`

Raw timings:

| run | code state | total ns | ns/query |
| --- | --- | ---: | ---: |
| 1 | pre-change baseline | `218733709` | `1093.67` |
| 2 | post-change | `16062083` | `3.21` |
| 3 | post-change rerun | `5916709` | `1.18` |
| 4 | post-change rerun | `4972083` | `0.99` |

Correctness / regression checks:

- `cargo test -p kv-native-sys host_arena --release`: `3 passed, 0 failed, 1 ignored`
- `cargo test -p kv-native-sys --release`: `9 passed, 0 failed, 1 ignored`
- reverse-release regression now verifies `reserved_bytes()` rewinds
  `192 -> 128 -> 64 -> 0` before a full-capacity reserve from offset `0`

## Problems

- This is a local microbench on the native host arena only. It proves the
  substrate bookkeeping win, not end-to-end spill/readmission throughput.
- The baseline was captured on the same workstation before the local patch and
  is not tied to a prior committed wins entry.
- The pre-change baseline came from the earlier shorter-loop version of the
  same microbench (`200000` timed queries). The comparison is still meaningful
  because the reported unit is `ns/query`, but the raw total-ns rows are not a
  same-loop apples-to-apples wall-time comparison.
- After the optimization, one query is so cheap that `ns/query` now sits close
  to timer / scheduler noise. The stable conclusion is the order-of-magnitude
  drop into the low-single-digit-nanosecond range, not the exact last decimal.

## Learnings

- A management-path free-list scan can dominate T1 bookkeeping even when the
  allocator itself is otherwise cheap. Keep counters on the arena when the
  scheduler polls them frequently.
- Tail-release rewind is worth doing even without a more complex free-list
  structure because it shrinks both allocation pressure and query overhead on
  common LIFO lifetimes.
- For this substrate, the clean cut is not a smarter global allocator; it is
  constant-time accounting plus opportunistic tail collapse.

## Δ vs baseline

- Prior wins entry: first local host-arena microbench for this path

| metric | baseline | now | Δ% |
| --- | ---: | ---: | ---: |
| `host_arena_reserved_bytes` ns/query | `1093.67` | `1.18` median of post runs | `-99.89%` |
| total timed query wall time | `218733709 ns` | `5916709 ns` median-like post run | `-97.29%` |

## Notes

- Code changes in this tranche:
  - `KvHostArena` now stores `reserved_bytes`
  - `host_arena_reserved_bytes()` reads that field directly
  - `host_arena_release()` rewinds `next_offset` on tail releases and keeps
    collapsing free-list regions that newly touch the tail
  - Rust-side tests now expose a dedicated `host_arena_bench` filter and a
    small RAII test harness for cleaner local validation
