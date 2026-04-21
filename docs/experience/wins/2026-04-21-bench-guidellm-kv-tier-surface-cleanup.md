# Bench Stub — KV Tier Surface Cleanup

## Goal

Status: `pending-remote`

Record the post-cleanup CUDA benchmark for the deletion-first KV-tier/runtime
surface cleanup:
- single waiting-queue admission path in `scheduler/cuda/runtime.rs`
- spill-only coordinator surface in `infer/src/kv_tier/coordinator.rs`
- removed `--max-gpu-kv` operator shim and deleted `scripts/bench_offload_memory.py`

## Hypothesis

The cleanup should be behavior-neutral or slightly positive:
- no throughput regression from removing duplicate waiting admission logic
- no TTFT regression from restoring prompt-length gating and priority ordering
- no operator-visible change beyond the removed stale CLI flag

## Params

Planned remote run:
- tool: `scripts/bench_guidellm.sh kv-tier-surface-cleanup`
- backend: CUDA
- compare against the most recent tiered-KV local multilayer baseline

## Env

Remote CUDA host required. Local Mac validation only covered compile/tests/clippy.

## Results

Pending remote benchmark.

## Problems

No local CUDA device was available in this workspace, so only Rust-side checks
ran locally.
- `cargo test -p infer --no-default-features --features cuda,no-cuda ...`
  still cannot be used as a local CUDA acceptance surrogate on this Mac,
  because the test/binary link step reaches unresolved CUDA symbols without a
  real CUDA toolchain/runtime. Local CUDA coverage for this batch therefore
  stops at `cargo check`.

## Learnings

Deletion-first cleanup is still a runtime change and must keep a benchmark
receipt even when the intended outcome is “no numerical or throughput change.”
