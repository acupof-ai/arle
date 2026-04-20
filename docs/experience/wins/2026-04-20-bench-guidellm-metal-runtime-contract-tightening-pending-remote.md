# metal runtime contract tightening — guidellm pending-remote, 2026-04-20

**Status:** `pending-remote`
**Commissioned by:** [`docs/plans/2026-04-15-metal-backend-acceptance-plan.md`](../../plans/2026-04-15-metal-backend-acceptance-plan.md)

## Goal

- Regression check: prove the Metal scheduler/runtime contract tightening does not move TTFT / ITL / throughput outside normal noise.

## Hypothesis

- The change is control-plane only: finished requests stop leaking into scheduler snapshots, and admitted-request snapshot invariants fail fast instead of degrading silently.
- Expected perf impact is within noise because no MLX kernel path or batch shape math changed.

## Command

```bash
scripts/bench_guidellm.sh metal-runtime-contract-tightening
```

- Canonical params remain the locked `scripts/bench_guidellm.sh` defaults.
- This command was **not** executed locally in this turn; this file exists to satisfy the required `pending-remote` stub flow for a runtime-affecting Metal diff.

## Environment

- Backend: Metal
- Hardware: pending remote / dedicated Apple Silicon bench host
- Commit: pending-this-commit
- Feature set: `cargo test --release --manifest-path infer/Cargo.toml --no-default-features --features metal,no-cuda --lib -- --test-threads 1`
- Non-default flags / env vars: `RUSTFLAGS='-D warnings'`

## Results — sweep headline table

- Pending remote execution.

## Problems

- No canonical `guidellm` run was executed locally in this turn.
- Local verification was limited to `cargo check` + full `metal,no-cuda` lib tests on the development Mac.

## Learnings

- Scheduler/runtime contract fixes still need an explicit performance stub, even when the change is expected to be perf-neutral.

## Δ vs baseline

- Pending remote execution against the most recent Metal baseline.

## Artefacts

- Local verification only:
  - `RUSTFLAGS='-D warnings' cargo check --release --manifest-path infer/Cargo.toml --no-default-features --features metal,no-cuda --lib`
  - `RUSTFLAGS='-D warnings' cargo test --release --manifest-path infer/Cargo.toml --no-default-features --features metal,no-cuda --lib -- --test-threads 1`
- Remote canonical artefacts: pending.
