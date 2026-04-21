# KV Tier COW + Zig Host Pool Bench Stub

## Context

Runtime change on the CUDA tiered-KV path:
- paged-prefill models now direct-attach radix-backed GPU pages
- `paged_kv` clones a shared tail page before append (decode-time COW)
- `HostPinnedPool` moved from Rust-managed storage to the Zig host arena substrate

Local Mac verification covered typecheck, no-cuda tests, no-cuda clippy, and
Metal typecheck. Remote CUDA throughput / latency validation is still pending.

Status: `pending-remote`

## Goal

Confirm that the new shared-page attach + tail-page COW path preserves TTFT/ITL
within noise for warm-prefix traffic, and that moving T1 host storage under Zig
does not regress spill-path latency.

## Hypothesis

- warm-prefix requests on paged-prefill models should avoid recompute and keep
  TTFT at or below the pre-change same-host baseline
- append-time COW should only affect the shared-tail case and should not add a
  visible ITL regression on steady decode
- Zig-backed T1 storage should be within noise of the previous Rust-managed host
  pool on spill-heavy traces

## Params

- Canonical tool: `scripts/bench_guidellm.sh kv-tier-cow-zig-host-pool`
- Backend: CUDA
- Compare against the most recent tiered-KV CUDA baseline before this change
- Include at least one warm-prefix scenario and one spill-pressure scenario

## Env

- Remote CUDA machine required
- Record GPU model, CUDA version, driver version, model, feature flags, and any
  non-default scheduler/tier config

## Results

Pending remote CUDA run.

## Problems

- Mac local lane cannot execute CUDA runtime benches
- `cargo test -p cuda-kernels --no-default-features --features cuda,no-cuda --lib`
  still links against CUDA symbols on macOS, so the pure-Rust `paged_kv` unit
  tests are compile-validated locally but not executable without a CUDA host

## Learnings

- The local refactor is now truthful about the hot path: direct GPU attach and
  decode-time COW are implemented, while T1/T2 live readmission remains a
  separate follow-on
