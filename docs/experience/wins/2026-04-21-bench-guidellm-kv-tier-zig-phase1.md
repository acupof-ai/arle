# KV Tier Zig Phase 1 — guidellm sweep, cuda, 2026-04-21

## Goal

- Pending remote regression check for the Phase 1 `kv-native-sys` integration that moved `DiskStore` low-level file I/O under a Zig-backed substrate.

## Hypothesis

- This change should be runtime-neutral for serving because it only affects the local disk/session persistence path and preserves the existing `DiskStore` API.

## Command

```bash
scripts/bench_guidellm.sh kv-tier-zig-phase1
```

Invoked via: `pending-remote`

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B
- **Hardware:** pending-remote
- **Commit:** `1eae964`
- **Feature set:** `cargo build --release`
- **Non-default flags / env vars:** pending-remote
- **Server launch:** pending-remote

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json,csv,html`
- Wrapper: `scripts/bench_guidellm.sh kv-tier-zig-phase1`

## Results — sweep headline table

Pending remote CUDA run.

## Problems

- Local dev box has no CUDA lane for the required regression run.
- Zig toolchain installation is still being finalized before local compile verification.

## Learnings

- Even storage-path refactors under `infer/src/` need a bench stub immediately so the remote follow-up is explicit instead of implied.

## Δ vs baseline

- **Baseline:** pending selection after the first remote run
- Delta table: pending-remote

## Artefacts

- Pending remote run output.

## Notes

- What changed in the code since baseline: introduced `crates/kv-native-sys` and routed `infer/src/kv_tier/transport/disk.rs` through it.
- Suspected cause of any regression: none expected; change is off hot-path for steady-state decode.
- Follow-ups: run the canonical CUDA regression check once Zig-backed build verification is complete.
