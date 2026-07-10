# Metal KV tier convergence — retire substrate S3, share the CUDA tier store

> Status: Shipped 2026-07-11 — runtime demote/promote smoke PASS ([wins](../experience/wins/2026-07-11-metal-kv-tier-convergence-smoke.md)).

**Verdict**: Metal's disk tier (`MetalSsdTier`, `infer-metal/src/kv_ssd.rs`) is a
full parallel reimplementation — own store, LRU, budget accounting,
serialization (`AMT2KV1`), fingerprint, and keying — sitting on the sharded
per-block-file substrate (S3) that CUDA abandoned for `KvMmapStore` (~4ms/page
→ mmap zero-copy). Converge: Metal instantiates the same two-level store as
its L3-only degenerate case (`host_capacity_pages = 0` — unified memory makes
a DRAM L2 a no-op), and substrate S3 gets deleted repo-wide.

## Steps

1. **Move the store backend-neutral.** `CudaKvTierStore`
   (`infer-cuda/src/kv_tier.rs`) is host-only (BTreeMap/LRU + kv-native-sys;
   no CUDA types). Move it to `kv-native-sys` as `KvTierStore` (kv-native-sys
   gains a dep on `infer-seam` for `KvTierLocation`/budget types — both are
   leaves, seam is trait-only). `infer-cuda` re-exports; call sites keep
   working (`pub use` alias `CudaKvTierStore` during the move is a half-state
   — rename call sites in the same commit).

2. **Metal adopts it.** `MetalPageStore.ssd: Option<MetalSsdTier>` →
   `tier: Option<KvTierStore>` constructed disk-only (`host` budget 0):
   - Keys: `tier_key(NS_*, logical_id)`; prefix snapshots (keyed today by
     `Vec<u64>`) get a hashed key + the chunked-blob API
     (`insert_chunked`/`read_chunked`) for variable-size MLX payloads —
     same pattern as DSv4 slot images.
   - Payload codec (MLX array bytes) stays in `kv_ssd.rs` — the store is
     payload-opaque.
   - Delete: `MetalSsdTier`, `encode/decode_metal_t2_record`, `MetalT2Cursor`,
     `put_u32/u64/i32`, `metal_t2_fingerprint`, `metal_t2_namespace`,
     `AMT2KV1` constants, the private budget/LRU/eviction block
     (kv_ssd.rs:62-277, 292-472).
   - Keep: `MetalPageStore` resident side (`pages`/`prefixes`,
     `publish_slot`, `materialize_slot_from_prefix`,
     `reusable_prefix_blocks`) — the MLX-specific live-slot↔tier bridge.

3. **Honesty fixes.** `kv_tier_transfer_is_zero_copy` on Metal returned
   `true` while S3 copied into `read_scratch`; on `KvMmapStore` reads are
   `Cow::Borrowed` and the claim becomes true for real. `kv_tier_location`
   may now also report `HostDemoted` if a nonzero L2 is ever configured —
   Metal pins L2 to 0 and documents why (unified memory: demoting device→host
   frees nothing; OS pressure already spills via SSD).

4. **Delete substrate S3.** `write_block_cache_sharded` /
   `read_block_into_sharded` / `remove_block_sharded`
   (`kv-native-sys/src/lib.rs:117-167`) — Metal was the last caller. Also the
   dead `DiskTier::mmap_path` + `KvMmapStore::flush` if the 3a sweep hasn't
   already taken them.

## Gates (all local — Apple Silicon box)

- `cargo test -p kv-native-sys --profile release-fast`
- `cargo test -p cli --release --no-default-features --features metal,no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo clippy -p infer-cuda --release --no-default-features --features cuda,no-cuda` (0/0 — the move must not break CUDA)
- Serve smoke with the tier live: `arle serve --backend metal --model-path
  mlx-community/Qwen3.6-35B-A3B-4bit --kv-disk`, demote/promote exercised
  (grep tier counters in `/v1/stats`), needle exact after a promote.
- Bench entry: one `scripts/bench_guidellm.sh` c=1 vs latest Metal baseline
  (tier off-path — expect wash; Δ% row per the bench mandate).

## Non-goals

- No L2 for Metal (unified memory — documented above).
- No durable/recall tier for Metal (CUDA's `--kv-recall` durability stays
  CUDA-only until a Metal use case exists).
- Whole-slot park stays disabled on Metal (trait defaults).
