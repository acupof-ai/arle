# Host-Pool Stability Guards — guidellm sweep, cuda, 2026-04-22

**Status:** `pending-remote`  
**Change scope:** `infer/src/kv_tier/host_pool.rs`, `infer/src/kv_tier/coordinator.rs`, `infer/src/scheduler/cuda/core.rs`

## Goal

- Regression / stability-tightening: confirm that making T1 host-pool region
  release failures explicit does not perturb steady-state CUDA serving latency.

## Hypothesis

- This change should be neutral for TTFT / ITL because it only tightens
  control-plane validation around host-pinned region lifetime tracking.
- If a regression appears, it will most likely be from extra bookkeeping in
  host-pool reserve / release rather than the CUDA decode path itself.

## Command

```bash
scripts/bench_guidellm.sh host-pool-stability-guards \
  --target http://<remote-host>:8000 \
  --model Qwen/Qwen3-4B \
  --processor models/Qwen3-4B \
  --trace-interval-ms 1000
```

Invoked via: `scripts/bench_guidellm.sh host-pool-stability-guards [--target URL] [--model NAME] [--processor PATH] [--trace-interval-ms N]`

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
- Wrapper: `scripts/bench_guidellm.sh host-pool-stability-guards`

## Results — sweep headline table

Pending remote CUDA run.

## Problems

- This workstation cannot run the canonical CUDA serving bench for the live T1
  host-pool path.
- Local validation can prove host-pool correctness and no-CUDA typecheck
  coverage, but not service-level TTFT / ITL on the CUDA lane.

## Learnings

- A safe Rust wrapper over the Zig host arena should reject stale / double
  region releases before they reach native state.
- Coordinator and scheduler cleanup paths must not silently swallow host-pool
  release failures; they need explicit warning surfaces.

## Δ vs baseline

- **Baseline:** [2026-04-22-bench-kv-native-host-arena-accounting.md](./2026-04-22-bench-kv-native-host-arena-accounting.md)
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
  - `cargo test -p infer --release --no-default-features --features no-cuda kv_tier::host_pool`
  - `cargo test -p infer --release --no-default-features --features no-cuda kv_tier::coordinator`
- What changed in the code since baseline:
  - `HostPinnedPool` now tracks live regions and rejects stale / double release
  - scheduler and coordinator host-region cleanup paths now emit warnings on
    release failure instead of silently ignoring them
