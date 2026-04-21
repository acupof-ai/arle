# KV Tier Zig Uniformity Fixes — guidellm sweep, cuda, 2026-04-21

## Goal

- Pending remote regression check for the final simplicity/uniformity fixes to the Zig KV substrate:
  - deterministic fresh SHM generations without time-based reseeding
  - uniform non-blocking coordinator queue submission for `stage`, `spill`, and `rehydrate`

## Hypothesis

- This change should remain runtime-neutral for steady-state serving because it only tightens descriptor freshness and queue submission semantics around the coordinator boundary.

## Command

```bash
scripts/bench_guidellm.sh kv-tier-zig-uniformity-fixes
```

Invoked via: `pending-remote`

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B
- **Hardware:** pending-remote
- **Commit:** `pending-next-push`
- **Feature set:** `cargo build --release`
- **Non-default flags / env vars:** pending-remote
- **Server launch:** pending-remote

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json,csv,html`
- Wrapper: `scripts/bench_guidellm.sh kv-tier-zig-uniformity-fixes`

## Results — sweep headline table

Pending remote CUDA run.

## Problems

- Local dev box has no CUDA lane for the required canonical regression run.
- The fix was validated locally with focused `cargo test/check/clippy`, but the guidellm sweep still needs a remote Linux/NVIDIA host.

## Learnings

- A plain process-local monotonic generation counter is simpler and more reliable than rebuilding pseudo-random freshness tokens from wall-clock state.
- `CoordinatorHandle` is easier to reason about when all best-effort submission APIs share the same non-blocking queue semantics.

## Δ vs baseline

- **Baseline:** [2026-04-21-bench-guidellm-kv-tier-zig-review-fixes.md](/path/to/code/agent-infer/docs/experience/wins/2026-04-21-bench-guidellm-kv-tier-zig-review-fixes.md:1)
- Delta table: pending-remote

## Artefacts

- Pending remote run output.

## Notes

- Local validation completed:
  - `cargo test -p kv-native-sys`
  - `cargo test -p infer --no-default-features --features no-cuda kv_tier::coordinator -- --nocapture`
  - `cargo clippy -p kv-native-sys -- -D warnings`
  - `cargo clippy -p infer --no-default-features --features no-cuda -- -D warnings`
  - `cargo check -p infer --no-default-features --features metal`
- What changed in the code since baseline:
  - `nextShmGeneration()` now uses a monotonic counter instead of time-seeded PRNG state
  - `StagePlanner for CoordinatorHandle` now uses the same `try_send` path as spill/rehydrate
  - added a full-queue `stage()` regression test
- Suspected cause of any regression: none expected; changes remain off the token-generation hot path.
