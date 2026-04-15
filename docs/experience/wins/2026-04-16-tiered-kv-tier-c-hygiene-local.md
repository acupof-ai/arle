# 2026-04-16 · Tiered KV Tier C hygiene local

## Context

Tier C tightens the prefix-cache hot path and promotes the five M3b
shipped constants into `SchedulerConfig` so operators can tune them
without recompiling. No architectural moves — pure hygiene.

## What Worked

- `RadixCache` grew a private `block_index: HashMap<BlockId, usize>`
  kept in sync with every insert / eviction / tombstone / fingerprint
  rebuild path. `find_block_node_mut` is O(1). `publish_to_prefix_cache`
  no longer pays 4×N linear scans per block; it now pays one hash
  lookup per `update_block_metadata` call.
- `rebuild_block_index()` helper restores the map after
  `serde` round-trips (the index is `#[serde(default, skip)]`) and
  after `rebuild_from_fingerprints` drops stale entries.
- `SchedulerConfig` absorbed the five M3b constants as real fields:
  `prefix_cache_high_water`, `prefix_cache_low_water`,
  `prefix_cache_retain_hard_cap`, `prefix_cache_keepalive_ticks`,
  `stage_wait_keepalive_ticks`. Defaults match the previous constants
  exactly; the old `pub(super) const`s are deleted.
- `SchedulerConfig::validate` gained watermark ordering /
  retain-cap monotonicity / keepalive ≥ 1 / stage_wait ≥ keepalive
  checks. Misconfigured configs refuse to construct a scheduler.
- Callers tune by explicit field assignment on a `SchedulerConfig`;
  there is deliberately **no env-var escape hatch**. The only
  `PEGAINFER_*` env vars that exist in the tree are debug-only
  diagnostic overrides (SM override, Triton python path, test model
  path, API disable flag).
- Verified locally with:
  - `cargo test -p infer --release --no-default-features --features no-cuda`
    (17 test result groups, 41 `prefix_cache` tests including two new
    block_index tests, 5 new `scheduler::types` tests)
  - `cargo check -p infer --tests --no-default-features --features cuda,no-cuda`
  - `cargo check -p infer --no-default-features --features metal`
  - `cargo fmt --all -- --check`

## Rule

When a constant is read from >1 site in the hot path, promote it to
`SchedulerConfig` with a `validate()` guard rather than letting it
drift. Env-var overrides are reserved for debug / diagnostic knobs —
tuning knobs belong on the config struct where callers can set them
explicitly and tests can sweep them.
