# HTTP serving identity snapshot cleanup — guidellm pending-remote, 2026-04-20

**Status:** `pending-remote`  
**Commissioned by:** [`docs/plans/2026-04-20-project-constitution-and-refactor-plan.md`](../../plans/2026-04-20-project-constitution-and-refactor-plan.md)

## Goal

- Regression check: prove the HTTP boundary cleanup does not move canonical `guidellm` TTFT / ITL / throughput outside noise.

## Hypothesis

- The change is control-plane only: the served model identity and DFlash init metadata are snapshotted at boot, while request submission still flows through `RequestHandle`.
- Expected perf impact is within noise because no scheduler, kernel, or batch-shape path changed.

## Command

```bash
scripts/bench_guidellm.sh http-serving-identity-snapshot
```

- Canonical params remain the locked `scripts/bench_guidellm.sh` defaults.
- This command was **not** executed locally in this turn; this file exists to satisfy the required `pending-remote` stub flow for a runtime-affecting HTTP diff.

## Environment

- Backend: HTTP serving surface
- Hardware: pending remote / canonical bench host for the affected backend
- Commit: pending-this-commit
- Feature set: `cargo test --release --manifest-path infer/Cargo.toml --no-default-features --features no-cuda --lib -- --test-threads 1`
- Non-default flags / env vars: `RUSTFLAGS='-D warnings'`

## Results — sweep headline table

- Pending remote execution.

## Problems

- No canonical `guidellm` run was executed locally in this turn.
- Local verification was limited to focused HTTP tests plus `cargo check` with warnings denied.

## Learnings

- HTTP boundary cleanups still need an explicit performance stub, even when the change is expected to be perf-neutral.

## Δ vs baseline

- Pending remote execution against the most recent relevant baseline.

## Artefacts

- Local verification only:
  - `RUSTFLAGS='-D warnings' cargo check --release --manifest-path infer/Cargo.toml --no-default-features --features no-cuda --lib`
  - `RUSTFLAGS='-D warnings' cargo test --release --manifest-path infer/Cargo.toml --no-default-features --features no-cuda --lib http_server::tests::serving_identity_is_snapshotted_once_and_reused_by_http_handlers -- --exact --test-threads 1`
- Remote canonical artefacts: pending.
