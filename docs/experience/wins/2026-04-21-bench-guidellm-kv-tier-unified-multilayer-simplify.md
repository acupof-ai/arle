# KV Tier Unified Multilayer Simplify — guidellm sweep, cuda, 2026-04-21

## Goal

- Pending remote regression check for the follow-up refactor that deletes leftover multilayer-KV side abstractions and keeps one simpler scheduler/coordinator path.

## Hypothesis

- The refactor should be runtime-neutral or slightly positive because it removes duplicate scheduler state and a dead planner abstraction without adding new hot-path work.

## Command

```bash
scripts/bench_guidellm.sh kv-tier-unified-multilayer-simplify
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
- Wrapper: `scripts/bench_guidellm.sh kv-tier-unified-multilayer-simplify`

## Results — sweep headline table

Pending remote CUDA run.

## Problems

- Local dev box cannot run the required CUDA guidellm sweep.
- Verification is currently local compile/test/clippy only.

## Learnings

- `lookup_or_stage` is simpler as a pure lookup surface; background stage submission belongs on the concrete coordinator handle, not inside the radix API.
- `prefix_cache` metadata already contains enough information to reconstruct host regions, so a second scheduler-side `block_to_host_regions` map only adds drift.
- The batch `BlockManager` does not need dead COW scaffolding when the production multilayer path no longer depends on it.

## Δ vs baseline

- **Baseline:** [2026-04-21-bench-guidellm-kv-tier-unified-multilayer-path.md](/path/to/code/agent-infer/docs/experience/wins/2026-04-21-bench-guidellm-kv-tier-unified-multilayer-path.md:1)
- Delta table: pending-remote

## Artefacts

- Pending remote run output.

## Notes

- Local validation completed:
  - `cargo test -p infer --no-default-features --features no-cuda prefix_cache -- --nocapture`
  - `cargo test -p infer --no-default-features --features no-cuda kv_tier::coordinator -- --nocapture`
  - `cargo test -p infer --no-default-features --features no-cuda block_manager -- --nocapture`
  - `cargo check -p infer --no-default-features --features no-cuda`
  - `cargo check -p infer --no-default-features --features metal`
  - `cargo check -p infer --no-default-features --features cuda,no-cuda`
  - `cargo clippy -p infer --no-default-features --features no-cuda -- -D warnings`
- What changed in the code since baseline:
  - removed `StagePlanner` and `LookupOutcome::staging_ticket`
  - made `CoordinatorHandle::stage` the only stage submission API
  - removed the duplicate scheduler `block_to_host_regions` side map
  - removed unused `BlockManager` copy-on-write helpers
- Suspected cause of any regression: if any regression appears, the likely source is changed host-region reconstruction during T1 stage/spill, not the deleted API surface itself.
