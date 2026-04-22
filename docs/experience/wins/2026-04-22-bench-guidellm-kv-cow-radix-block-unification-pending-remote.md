# KV COW / Radix / Block Unification — guidellm sweep, cuda, 2026-04-22

**Status:** `pending-remote`  
**Plan anchor:** [`docs/plans/kv-cow-radix-block-unification.md`](../../plans/kv-cow-radix-block-unification.md)  
**Change scope:** `infer/src/prefix_cache.rs`, `crates/cuda-kernels/src/paged_kv.rs`, `infer/src/scheduler/cuda/{core,runtime}.rs`

## Goal

- Regression / contract-tightening: confirm that tightening the live CUDA
  tiered-KV contract around compressed-radix block boundaries, sealed full
  blocks, and tail-page COW preserves warm-prefix TTFT / ITL within noise.

## Hypothesis

- Warm-prefix requests should keep the current direct-attach behavior and avoid
  recompute.
- Making the sealed-block / hot-tail boundary explicit should not introduce a
  visible steady-state decode regression.
- The staged promote path should remain flat relative to the latest local
  tiered-KV baseline because the change is a contract cleanup, not a new data
  movement path.

## Command

```bash
scripts/bench_guidellm.sh kv-cow-radix-block-unification \
  --target http://<remote-host>:8000 \
  --model Qwen/Qwen3-4B \
  --processor models/Qwen3-4B \
  --trace-interval-ms 1000
```

Invoked via: `scripts/bench_guidellm.sh kv-cow-radix-block-unification [--target URL] [--model NAME] [--processor PATH] [--trace-interval-ms N]`

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
- Wrapper: `scripts/bench_guidellm.sh kv-cow-radix-block-unification`

## Results — sweep headline table

Pending remote CUDA run.

## Problems

- This workstation is not the final CUDA runtime bench host.
- Local validation can prove radix correctness, scheduler contract coherence,
  and no-CUDA typecheck coverage, but not serving-level TTFT / ITL on the live
  CUDA path.

## Learnings

- In a compressed radix, a short-edge block-bearing node is valid when the
  cumulative path ends on a block boundary; the runtime contract must encode
  boundary-on-path rather than `edge_len == block_size`.
- The only shared-prefix write path should remain tail-page COW before append;
  sealed blocks stay read-only across publish / attach / promote.

## Δ vs baseline

- **Baseline:** [2026-04-21-bench-guidellm-kv-tier-cow-zig-host-pool.md](./2026-04-21-bench-guidellm-kv-tier-cow-zig-host-pool.md)
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

- What changed in the code since baseline:
  - tightened the compressed-radix block-boundary contract in `prefix_cache`
  - made the sealed-block vs hot-tail COW boundary explicit in `paged_kv`
  - cleaned up scheduler publish / attach / promote helpers to speak in terms
    of sealed full blocks
- Suspected cause of any regression: if one appears, the likely source is extra
  scheduler bookkeeping or a missed fast-path in direct attach / staged promote,
  not a new transport path
- Follow-ups:
  - run the canonical remote CUDA sweep
  - compare warm-prefix and spill-pressure scenarios against the latest
    tiered-KV baseline
