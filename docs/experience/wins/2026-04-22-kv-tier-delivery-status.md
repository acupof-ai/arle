# KV Tier Delivery Status — 2026-04-22

## Context

This note closes the current `kv-tier` integration line after the staged
readmission / shared-fs / cluster-backend / store-surface cleanup work was
rebased onto current `origin/main` and revalidated in a clean temporary
worktree.

## Landed Scope

The current tree now ships one unified local-to-cluster-shaped `kv-tier` path:

- `RadixCache` remains the metadata/control-plane source of truth
- paged-prefill models can direct-attach GPU-resident prefix pages
- staged readmission is live through `ReadmissionPlan + WaitingFetch + FetchTicket + promote_fetched_prefix`
- slower-tier writes now use one queue/event vocabulary: `StoreRequest / StoreTicket / CoordinatorEvent::Store*`
- local disk and shared-fs remote storage both run through the same coordinator fetch/store path
- store submissions are deduped per `(fingerprint, target)`
- scheduler admission now follows one explicit order: direct GPU attach -> staged readmission -> same-slot reuse -> cold prefill fallback
- `rdma-nixl` and `rdma-nixl-real` are now real, distinct Cargo surfaces instead of two names for the same stub dependency shape

## Verification Completed

Validated locally from the clean integration worktree using the shared target dir:

- `cargo check -p infer --release --no-default-features --features no-cuda`
- `cargo test -p infer --release --no-default-features --features no-cuda kv_tier::coordinator -- --nocapture`
- `cargo test -p infer --release --no-default-features --features no-cuda scheduler -- --nocapture`
- `cargo clippy -p infer --release --no-default-features --features no-cuda -- -D warnings`
- `cargo check -p infer --release --no-default-features --features cuda,no-cuda`
- `cargo check -p infer --release --no-default-features --features no-cuda,rdma-nixl`
- `cargo check -p infer --release --no-default-features --features no-cuda,rdma-nixl-real`
- `cargo check -p infer --release --no-default-features --features metal,no-cuda`

## Known Limits

These are the remaining limits after the current tranche, not hidden TODOs:

1. Remote CUDA / guidellm acceptance is still required for the scheduler/runtime changes under `infer/src/scheduler/cuda/*` and `infer/src/kv_tier/*`. The repo already carries `pending-remote` wins stubs for those runtime tranches.
2. `NixlTransport` remains non-functional at runtime. The feature surface is now honest and split correctly, but the transport itself is still a control-plane stub until the real NIXL/RDMA implementation lands.
3. On macOS, `cargo test --features no-cuda,rdma-nixl` is blocked by the external `nixl-sys` crate linking `-lstdc++` into test binaries. This is tracked in `docs/experience/errors/2026-04-22-nixl-sys-macos-tests-link-stdcpp.md`.
4. The minimal cluster-shared backend in-tree is `shared-fs`. RDMA/NIXL/Mooncake data-plane implementations are not yet present.

## Rule

For this line, “done locally” now means: rebased onto current `origin/main`, no-cuda scheduler/kv-tier tests green, `cuda,no-cuda` typecheck green, metal/no-cuda typecheck green, and every remaining gap is either a remote CUDA validation item or an external dependency blocker written down explicitly.
