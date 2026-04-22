# Host-Pool Native Error Surfacing — guidellm sweep, cuda, 2026-04-22

**Status:** `pending-remote`  
**Change scope:** `infer/src/kv_tier/host_pool.rs`, `infer/src/kv_tier/coordinator.rs`, `infer/src/scheduler/cuda/core.rs`

## Goal

- Regression / stability-tightening: confirm that turning host-pool native
  query / reserve failures into explicit control-plane errors does not regress
  CUDA serving latency.

## Hypothesis

- TTFT / ITL should stay within noise because this change only replaces panic
  paths with explicit `Result` handling and conservative scheduler fallbacks.
- If a regression appears, it will most likely come from extra host-pool
  bookkeeping in the demote / staged-fetch control path.

## Command

```bash
scripts/bench_guidellm.sh host-pool-native-error-surfacing \
  --target http://<remote-host>:8000 \
  --model Qwen/Qwen3-4B \
  --processor models/Qwen3-4B \
  --trace-interval-ms 1000
```

Invoked via: `scripts/bench_guidellm.sh host-pool-native-error-surfacing [--target URL] [--model NAME] [--processor PATH] [--trace-interval-ms N]`

## Environment

- **Backend:** `cuda`
- **Model:** `Qwen/Qwen3-4B`
- **Hardware:** `pending-remote`
- **Commit:** `pending-remote`
- **Feature set:** `cargo build -p infer --release --bin infer`
- **Non-default flags / env vars:** `pending-remote`
- **Server launch:** `pending-remote`

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh host-pool-native-error-surfacing`

## Results — sweep headline table

Pending remote CUDA run.

## Problems

- This workstation cannot run the canonical CUDA serving bench for the live T1
- local validation only covers no-CUDA typecheck and control-plane tests

## Learnings

- Host-pool native failures should surface as ordinary Rust errors, not
  implicit panics hidden behind `expect()`.
- Scheduler accounting should degrade to conservative defaults when host-pool
  usage cannot be queried.

## Δ vs baseline

- **Baseline:** [2026-04-22-bench-guidellm-host-pool-stability-guards-pending-remote.md](./2026-04-22-bench-guidellm-host-pool-stability-guards-pending-remote.md)
- Delta table: `pending-remote`

## Artefacts

- Raw: `pending-remote`
- CSV: `pending-remote`
- HTML: `pending-remote`
- Service trace (before): `pending-remote`
- Service trace (during): `pending-remote`
- Service trace (after): `pending-remote`
- Service trace (summary): `pending-remote`

## Notes

- Local validation for this tranche:
  - `cargo test -p infer --release --no-default-features --features no-cuda kv_tier::host_pool -- --nocapture`
  - `cargo test -p infer --release --no-default-features --features no-cuda kv_tier::coordinator -- --nocapture`
  - `cargo check -p infer --release --no-default-features --features cuda,no-cuda`
- What changed in the code since baseline:
  - `HostPinnedPool::reserve`, `reserved_bytes`, and `remaining_bytes` now return explicit `Result`
  - staged fetch and demote paths now surface host-pool native errors instead of panicking
  - scheduler accounting falls back conservatively when host-pool usage queries fail
