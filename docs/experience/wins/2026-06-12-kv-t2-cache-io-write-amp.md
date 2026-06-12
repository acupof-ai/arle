# KV T2 cache I/O write amplification reduction — pending remote A/B

## Goal

Reduce KV T2 cache write amplification and avoid repeated read allocations in
the CUDA page-tier store, without changing prefix-cache or scheduler semantics.

## Hypothesis

T2 block files are rebuildable cache payloads, not WAL records. The cache path
can keep temp-write + rename replacement while skipping `sync_data` and parent
directory fsync, and repeated T2 reads can reuse one store-owned scratch buffer
before H2D promotion.

## Params

- `kv-native-sys`: added `write_block_cache` / `write_file_atomic_cache`
  (rename atomic, no durability fsync) and `read_block_into` / `read_file_into`
  (caller-provided buffer).
- `infer-cuda::CudaKvTierStore`: T2 writes now use cache writes; T2 reads fill
  a reusable `read_scratch`; T1 coldest selection uses a `(stamp,key)` LRU index
  instead of scanning all T1 entries at spill time.
- Durable APIs (`write_block_atomic`, WAL append, mmap/shm paths) are unchanged.

## Env

Local Apple Silicon host, no CUDA execution. T2 disk spill is opt-in and the
device wall-clock gate must run remotely.

## Results

- `cargo test -p kv-native-sys --release -- --nocapture`: 14 passed, 1 ignored.
- `cargo test -p infer-cuda --release --no-default-features --features no-cuda kv_tier -- --nocapture`: 9 passed.
- `cargo fmt --check`: passed.
- `cargo clippy -p kv-native-sys --release -- -D warnings`: passed.
- `cargo clippy -p infer-cuda --release --no-default-features --features no-cuda -- -D warnings`: passed.
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib`: passed.
- `CUDARC_CUDA_VERSION=12080 cargo clippy -p infer-api --release --no-default-features --features cuda,no-cuda --lib -- -D warnings`: passed.
- `cargo test -p kv-native-sys cache_block_io_bench --release -- --ignored --nocapture`: passed.

Structural delta:

| Path | Before | After |
| --- | --- | --- |
| T2 cache write | temp write + `sync_data` + rename + parent dir fsync | temp write + rename |
| T2 cache read | allocate a fresh `Vec<u8>` per read | reuse store scratch buffer |
| T1 spill victim | scan all T1 entries by stamp | O(log n) LRU index |

Local Rust substrate microbench (Apple Silicon tempdir, release build):

| payload | durable atomic write | cache write | read fresh `Vec` | read into scratch |
| --- | ---: | ---: | ---: | ---: |
| 4 KiB x128 | 0.4 MiB/s | 5.3 MiB/s | 132.4 MiB/s | 234.7 MiB/s |
| 256 KiB x32 | 29.2 MiB/s | 139.2 MiB/s | 1729.2 MiB/s | 3849.2 MiB/s |
| 4 MiB x4 | 370.5 MiB/s | 1988.3 MiB/s | 3021.5 MiB/s | 7966.0 MiB/s |

This verifies the pure storage substrate directly. It does not prove GPU
wall-clock serving impact; the HBM↔host copy and scheduler interaction still
need CUDA validation.

## Problems

No CUDA wall-clock A/B was run locally. Required remote gate: same-binary
long-prefix replay with `--kv-ssd-path`, comparing tier off vs T1+T2 on, and
recording TTFT, demote/promote counts, T2 bytes, and end-to-end correctness.

## Learnings

Cache replacement and durable persistence need separate substrate APIs. Reusing
the durable atomic writer for a rebuildable KV cache made every spill pay WAL-like
fsync cost; splitting the API keeps the durable default intact while letting the
cache path choose lower write amplification explicitly.
