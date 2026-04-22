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

Historical baseline reference captured before the optimization:

```bash
cargo test -p kv-native-sys --release \
  tests::host_arena_bench_reserved_bytes_fragmented \
  -- --ignored --exact --nocapture
```

Focused validation reproduced after the optimization:

```bash
cargo test -p kv-native-sys --release host_arena -- --nocapture
```

```bash
cargo test -p kv-native-sys --release \
  tests::host_arena_bench_reserved_bytes_fragmented \
  -- --ignored --exact --nocapture
```

Package-level validation reproduced on the same workstation:

```bash
cargo test -p kv-native-sys --release
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

Baseline reference workload before the optimization:

- arena capacity: `8192 * 64 = 524288` bytes
- live holes after fragmentation: `4096`
- timed queries: `200000`
- warmup queries: `20000`
- expected reserved bytes: `262144`

Post-change workload reproduced in this entry:

- arena capacity: `8192 * 64 = 524288` bytes
- live holes after fragmentation: `4096`
- timed queries: `5000000`
- warmup queries: `100000`
- expected reserved bytes: `262144`

Raw timings:

| run | code state | workload | ns/query |
| --- | --- | --- | ---: |
| 1 | historical pre-change reference | `200000` timed queries | `1093.67` |
| 2 | post-change | `5000000` timed queries | `3.21` |
| 3 | post-change rerun | `5000000` timed queries | `1.18` |
| 4 | post-change rerun | `5000000` timed queries | `0.99` |
| 5 | post-change rerun | `5000000` timed queries | `1.09` |

Correctness / regression checks:

- `cargo test -p kv-native-sys --release host_arena -- --nocapture`: `4 passed, 0 failed, 1 ignored`
- `cargo test -p kv-native-sys --release`: `10 passed, 0 failed, 1 ignored`
- reverse-release regression verifies `reserved_bytes()` rewinds
  `192 -> 128 -> 64 -> 0` before a full-capacity reserve from offset `0`
- tail-collapse regression now verifies a free-list hole chain (`64..128`,
  `128..192`) is folded into the tail when the adjacent tail block is
  released, allowing a single `192`-byte reserve from offset `64`

## Problems

- This is a local microbench on the native host arena only. It proves the
  substrate bookkeeping win, not end-to-end spill/readmission throughput.
- The baseline was captured on the same workstation before the local patch and
  is not tied to a prior committed wins entry.
- The pre-change baseline came from the earlier shorter-loop version of the
  same microbench (`200000` timed queries). The comparison is still meaningful
  as a directional order-of-magnitude reference because the reported unit is
  `ns/query`, but it is not a same-workload A/B.
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

## Notes

- Code changes in this tranche:
  - `KvHostArena` now stores `reserved_bytes`
  - `host_arena_reserved_bytes()` reads that field directly
  - `host_arena_release()` rewinds `next_offset` on tail releases and keeps
    collapsing free-list regions that newly touch the tail
  - Rust-side tests now expose a dedicated `host_arena_bench` filter and a
    small RAII test harness for cleaner local validation
