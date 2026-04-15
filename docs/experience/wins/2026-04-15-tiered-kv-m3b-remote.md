# 2026-04-15 · Tiered KV Cache M3b — remote CUDA acceptance

## Context

M3b is the contract/state-machine tranche of the T1 host-pinned tier
work: `RadixCache::lookup_or_stage(...)`, `HitKind`, `LookupOutcome`,
`LookupHeuristics`, `StageTicket`, `StagePlanner`, the pure
`Free | Resident | Demoting` page-lifecycle state machine, and
`RadixCache::evict_with_policy(...)` as the live eviction path. Local
win notes:
[`2026-04-15-tiered-kv-m3b-local.md`](2026-04-15-tiered-kv-m3b-local.md)
and
[`2026-04-15-tiered-kv-m3b-runtime-local.md`](2026-04-15-tiered-kv-m3b-runtime-local.md)
for the subsequent runtime-wire landing.

This note records the remote acceptance run against
[`../../plans/tiered-kv-cache-m3b-remote-acceptance.md`](../../plans/tiered-kv-cache-m3b-remote-acceptance.md).

## Environment

- GPU: NVIDIA L4 24 GB (driver 580.82.07, CUDA 13.0, SM 89)
- Model: Qwen3-4B BF16, `Qwen/Qwen3-4B` HF Instruct variant
- Commits evaluated: `58cd8de` (first pass against M3b contract
  batch) → `85bc85b` (post-pull re-run, now includes
  `66072e8 feat(scheduler): wire local lookup classification and
  keepalive metadata` — the runtime-wire tranche that extends M3b).
- Cargo env: `CARGO_HOME=/tmp/cargo-home-local`

## Static sanity (§2)

Ran the 4 greps from §2 of the M3b remote acceptance doc:

- `rg lookup_or_stage|set_block_location|set_block_byte_len infer/src/prefix_cache.rs`
  → `prefix_cache.rs:268` (public `fn lookup_or_stage`),
  `:780` (`set_block_location`), `:791` (`set_block_byte_len`), plus
  5 test functions
  (`lookup_or_stage_defaults_to_ready_on_gpu`, `..._queues_host_stage`,
  `..._advises_recompute_for_small_disk_hit`,
  `..._surfaces_tombstone_miss`, `evict_with_policy_*`). ✅
- `rg HitKind|LookupOutcome|LookupHeuristics|StageTicket|StagePlanner|PageLifecycleState`
  in `infer/src/kv_tier/` + `infer/src/kv_tier.rs` →
  `kv_tier/lookup.rs` carries `HitKind`, `LookupOutcome`,
  `LookupHeuristics`, `StageTicket`, `StagePlanner`, `StageRequest`;
  `kv_tier/coordinator.rs` carries `PageLifecycleState` +
  `PageLifecycleError` + `CoordinatorHandle: StagePlanner` impl;
  `kv_tier.rs:126` re-exports them all. ✅
- `rg lookup_or_stage|StageTicket|PageLifecycleState infer/src/scheduler/cuda`
  → **no matches** on commit `58cd8de` (M3b contract-only tranche,
  scheduler not yet wired), **matches** on commit `85bc85b` after
  `66072e8` landed the runtime wire. Both are expected outcomes at
  the commit they were measured at. ✅
- `rg evict_with_policy infer/src/prefix_cache.rs infer/src/scheduler/cuda/core.rs`
  → `prefix_cache.rs:606` (the live `fn evict_with_policy`) + 3 test
  functions, and `scheduler/cuda/core.rs:490,536` (two call sites
  inside the scheduler cleanup / allocation path). ✅

## Build / test gates (§3)

```
cargo build -p infer --release                                     # 4m 11s cold, 12s incremental
cargo test --workspace --exclude mlx-sys --release --lib           # 349 tests pass (274 infer + 35 cuda-kernels + 40 others)
cargo test -p infer --release --test e2e                           # Phase 1-4 all green (after replay-drift fix)
cargo fmt --all -- --check                                         # clean
```

Pre-existing failures (same list as M2b, not in scope): `e2e_qwen35`
baseline drift, `greedy_consistency` B=3 decode, clippy unused import
in infer-tools.

The Phase 3 e2e regression that surfaced here was addressed inline —
see
[`../errors/2026-04-15-e2e-phase3-replay-drift.md`](../errors/2026-04-15-e2e-phase3-replay-drift.md)
for root cause and the `ReplayFinalToken` → full-recompute fix in
`server_engine.rs`. That fix is in the single-request engine only;
it does not touch any of M3b's contract types.

## Focused M3b smoke (§4)

```
cargo test -p infer --release prefix_cache      # 34 pass (1 added post-pull in 66072e8)
cargo test -p infer --release kv_tier           # 20 pass
```

Both green. The new tests covered by this run include:

- `lookup_or_stage_defaults_to_ready_on_gpu` — GPU hit classification
- `lookup_or_stage_queues_host_stage` — `HostPinned` hit enters
  `StagingFromHost`
- `lookup_or_stage_advises_recompute_for_small_disk_hit` —
  `LookupHeuristics::should_recompute_instead_of_stage` triggers for
  a too-small disk stage
- `lookup_or_stage_surfaces_tombstone_miss` — tombstoned node reports
  `HitKind::Miss`
- `evict_with_policy_*` — three cases covering LRU fallback, session
  affinity, soft-pin respect

§4 optional stub grep:

```
rg "GPU required: LocalCudaTransport poll is a structural stub" \
   infer/src/kv_tier/transport/local_cuda.rs
```

Still present — real `cudaMemcpyAsync` behaviour is correctly deferred
to later M3 runtime patches. ✅

## Sign-off

- [x] Static sanity checks passed (both the pre-runtime-wire and
      post-runtime-wire variants of the scheduler grep).
- [x] Build/test gate passed on CUDA after the replay-drift fix.
- [x] Focused `prefix_cache` / `kv_tier` smoke passed — all new
      `lookup_or_stage_*` and `PageLifecycleState` tests green.
- [x] `LocalCudaTransport` structural stub markers still in place.
- [x] This win note exists with commands and test counts.

**M3b contract tranche accepted on the 2026-04-15 L4 host.** The
runtime-wire follow-up (`66072e8`) is carried into the post-pull
re-run and does not regress any of the contract tests.

## Rule

When a big contract tranche lands (M3b: `lookup_or_stage`,
`StagePlanner`, `PageLifecycleState`, `evict_with_policy`), the
remote acceptance test matrix must cover **both** the pre-wire state
(contract-only, scheduler grep should be empty) and the post-wire
state (scheduler consumes the contract, same greps should now
match). This is cheap: just run the same doc twice at two commits.
It catches the case where the contract compiles cleanly but the
runtime wire silently misses a classification branch.
