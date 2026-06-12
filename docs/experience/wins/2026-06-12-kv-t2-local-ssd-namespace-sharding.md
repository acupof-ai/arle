# KV T2 local SSD namespace + sharded layout

## Goal

Make the opt-in local SSD KV tier safer and more scalable as a process-local
cache before adding async readmission, GDS, or remote/shared transports.

## Hypothesis

The current T2 key is engine-local (`u64`), so writing directly into the
operator-provided `--kv-ssd-path` risks collisions when two serve processes use
the same root. A large flat directory of `.kv` files also becomes an avoidable
filesystem hot spot. A process-owned namespace plus two-level sharding should
make local SSD use safer without changing scheduler or KV semantics.

## Params

- `kv-native-sys`: added sharded cache block helpers:
  `block_path_sharded`, `write_block_cache_sharded`,
  `read_block_into_sharded`, and `remove_block_sharded`.
- `infer-cuda::CudaKvTierStore`: `set_disk` now creates a unique
  `arle-kv-tier-<pid>-<time>-<counter>` namespace under the configured root and
  returns `false` if that namespace cannot be created.
- T2 block files now live under `<namespace>/<hex[0..2]>/<hex[2..4]>/<hex>.kv`.
- `DiskTier::drop` removes only its process-owned namespace. The
  operator-provided root survives.
- `RealCudaExecutor::set_kv_tier_disk` now propagates Qwen namespace creation
  failure instead of always returning `true`.

## Env

Local Apple Silicon host. No CUDA device wall-clock run in this tranche.

## Results

- `cargo fmt --check`: passed.
- `cargo test -p kv-native-sys --release -- --nocapture`: 15 passed, 2 ignored.
- `cargo test -p infer-cuda --release --no-default-features --features no-cuda kv_tier -- --nocapture`: 10 passed.
- `cargo clippy -p kv-native-sys --release -- -D warnings`: passed.
- `cargo clippy -p infer-cuda --release --no-default-features --features no-cuda -- -D warnings`: passed.
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib`: passed.

New regression coverage:

- `cache_block_sharded_roundtrips_and_removes`: verifies sharded cache writes,
  scratch reads, and deletion.
- `disk_namespace_shards_and_cleans_up_process_owned_cache`: verifies the disk
  tier does not write into the shared root directly, writes under the shard
  path, cleans up the namespace on drop, and leaves the operator root intact.

## Problems

This is still process-local SSD spill, not durable cross-restart reuse. The
engine-local key format is deliberately isolated instead of advertised as a
persistent cache identity. CUDA wall-clock A/B remains required before claiming
TTFT or throughput impact.

## Learnings

Local SSD should first be treated as a rebuildable, process-owned cache. That
keeps the low-write-amplification cache writer from
`2026-06-12-kv-t2-cache-io-write-amp.md`, avoids unsafe cross-process key
collisions, and prevents flat-directory scaling issues. Durable/shared reuse
needs a content fingerprint and scheduler-visible cache identity, not reuse of
the engine-local tier key.
