# KV Tier Unified Multilayer Path — guidellm sweep, cuda, 2026-04-21

## Goal

- Pending remote regression check for the refactor that collapses local multilayer KV onto one path: `prefix_cache + scheduler + coordinator + DiskStore`, with `stage + spill` as the only live coordinator commands.

## Hypothesis

- The refactor should be runtime-neutral or slightly positive because it deletes dead state machines and unifies tier transitions without adding hot-path work.

## Command

```bash
scripts/bench_guidellm.sh kv-tier-unified-multilayer-path
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
- Wrapper: `scripts/bench_guidellm.sh kv-tier-unified-multilayer-path`

## Results — sweep headline table

Pending remote CUDA run.

## Problems

- Local dev box cannot run the required CUDA guidellm sweep.
- Verification is currently local compile/test/clippy only.

## Learnings

- One live coordinator restore path is enough: disk-backed `Stage` replaces a separate `Rehydrate` command without losing behavior.
- `RadixCache` metadata plus scheduler-owned residency maps are sufficient; extra lifecycle sidecars only add drift.

## Δ vs baseline

- **Baseline:** [2026-04-21-bench-guidellm-kv-tier-zig-uniformity-fixes.md](/path/to/code/agent-infer/docs/experience/wins/2026-04-21-bench-guidellm-kv-tier-zig-uniformity-fixes.md:1)
- Delta table: pending-remote

## Artefacts

- Pending remote run output.

## Notes

- Local validation completed:
  - `./scripts/check_kv_zig.sh`
  - `cargo check -p infer --no-default-features --features no-cuda`
  - `cargo check -p infer --no-default-features --features cuda,no-cuda`
  - `cargo test -p infer --no-default-features --features no-cuda prefix_cache -- --nocapture`
  - `cargo test -p infer --no-default-features --features no-cuda kv_tier::coordinator -- --nocapture`
  - `cargo test -p infer --no-default-features --features no-cuda kv_tier::transport::local_cuda -- --nocapture`
  - `cargo check -p infer --no-default-features --features metal`
- What changed in the code since baseline:
  - removed the separate coordinator `Rehydrate` command/event path in favor of disk-backed `Stage`
  - removed unused `Demote` / `Promote` coordinator commands
  - removed the unused `PageLifecycle` side state machine
  - de-scoped `block_manager` back to the batch scheduler helper instead of the production multilayer story
- Suspected cause of any regression: if any regression appears, the likely source is changed cleanup/admission ordering around T1 host staging rather than the deleted bookkeeping.
