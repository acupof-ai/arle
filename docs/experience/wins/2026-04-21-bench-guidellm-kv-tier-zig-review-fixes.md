# KV Tier Zig Review Fixes — guidellm sweep, cuda, 2026-04-21

## Goal

- Pending remote regression check for the Zig substrate review fixes:
  - durable atomic block writes on Darwin/Linux
  - non-blocking coordinator spill / rehydrate submit semantics
  - generation-checked shared-memory descriptors
  - Darwin `zig build-lib` archive fix via `ranlib`

## Hypothesis

- This change should remain runtime-neutral for steady-state serving because it only touches the disk / shared-memory substrate, coordinator queue semantics, and build-tooling correctness.

## Command

```bash
scripts/bench_guidellm.sh kv-tier-zig-review-fixes
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
- Wrapper: `scripts/bench_guidellm.sh kv-tier-zig-review-fixes`

## Results — sweep headline table

Pending remote CUDA run.

## Problems

- Local dev box has no CUDA lane for the required regression run.
- These fixes were validated locally with `cargo test/clippy/check`, but the canonical runtime regression check still needs a remote Linux/NVIDIA host.

## Learnings

- The Darwin Zig toolchain path needed a real root-cause fix (`ranlib` over the archive produced by `zig build-lib`) rather than a packaging detour.
- Shared-memory descriptors need freshness identity, not just a POSIX name, if they are going to behave like exportable handles.

## Δ vs baseline

- **Baseline:** [2026-04-21-bench-guidellm-kv-tier-zig-phase1.md](/path/to/code/agent-infer/docs/experience/wins/2026-04-21-bench-guidellm-kv-tier-zig-phase1.md:1)
- Delta table: pending-remote

## Artefacts

- Pending remote run output.

## Notes

- Local validation completed:
  - `cargo test -p kv-native-sys`
  - `cargo clippy -p kv-native-sys -- -D warnings`
  - `cargo test -p infer --no-default-features --features no-cuda kv_tier::coordinator -- --nocapture`
  - `cargo test -p infer --no-default-features --features no-cuda kv_tier::transport::disk -- --nocapture`
  - `cargo clippy -p infer --no-default-features --features no-cuda -- -D warnings`
  - `cargo check -p infer --no-default-features --features metal`
- Suspected cause of any regression: none expected; changes remain off the token-generation hot path.
