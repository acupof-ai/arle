# M2 - Metal KV-Tier Adapter

## Context

Backend-unification M2 wires the existing backend-agnostic tier vocabulary into
Metal without importing CUDA or Metal concrete types into cross-backend modules.
The milestone keeps CUDA behavior unchanged and makes Metal's Qwen3.5 SSD prefix
snapshot path the first T2 disk persistence path; Metal still skips T1
HostPinned because unified memory makes that tier non-actionable.

## What Worked

- Added `KvTierAdapter` in `infer/src/kv_tier.rs` with backend-neutral
  `BlockId`/`Tier` signatures.
- Wrapped CUDA `TieredKvPolicy` with a no-op adapter implementation so existing
  CUDA coordinator flow remains the only mover of CUDA KV blocks.
- Added `MetalTierAdapter` in `infer/src/backend/metal/runtime.rs`, routing the
  Qwen3.5 SSD prefix snapshot read/write/delete path through the adapter and
  rejecting T1/T3 requests explicitly.
- Exposed `MetalKVPool::paged_pool_pressure()` and refreshed the adapter pressure
  from active Metal request `kv_pool_usage()` during scheduler metric refresh.
- Added Metal adapter smoke coverage for T1 rejection and disk snapshot
  round-trip, plus CPU-testable pressure coverage for the Metal slot ledger.

## Bench Status

`pending-remote`: this Linux runner has CUDA only. `cargo check` validates the
Metal Rust surface in non-macOS stub mode, but Metal tests and the GuideLLM
long-context run need the Apple Silicon Metal runner.

Pending command:

```bash
scripts/bench_guidellm.sh metal-m3max --workload longctx-32k
```

Acceptance target: with Metal disk persistence enabled, a second run should show
cross-run prefix-cache hit rate >= 50%.

## Verification

| Check | Result |
|---|---|
| `cargo fmt --all --check` | pass |
| `git diff --check` | pass |
| `cargo check -p infer --features cuda` | pass |
| `cargo check -p infer --no-default-features --features cuda,no-cuda` | pass |
| `cargo check -p infer --no-default-features --features metal,no-cuda` | pass, non-macOS MLX bridge stub |
| `cargo clippy -p infer --features cuda -- -D warnings` | pass |
| `cargo clippy -p infer --no-default-features --features no-cuda -- -D warnings` | pass |
| `cargo test --release -p infer --no-default-features --features no-cuda kv_tier::coordinator` | pass |
| `cargo test --release -p infer --no-default-features --features no-cuda paged_pool_pressure_tracks_occupancy` | pass |
| `cargo test --release -p infer --features cuda --test e2e` | pass |
| `cargo test --release -p infer --features cuda --test greedy_consistency` | pass |

Metal test execution on this host fails at link time because `mlx-sys` skips the
MLX/Metal bridge on non-macOS, leaving expected symbols such as
`mlx_array_free`, `mlx_zeros`, and `qwen35_compiled_free` undefined. The Rust
typecheck path above still validates cfg isolation for this change.

## Rule

Tier adapter traits stay value-typed and backend-neutral. Backend-specific
storage handles live behind the adapter implementation, and Metal T2 persistence
should reuse the existing disk transport while T1 remains an explicit rejection
instead of a silent fake tier.
