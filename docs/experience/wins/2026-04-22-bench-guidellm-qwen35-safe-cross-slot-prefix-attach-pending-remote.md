# Qwen3.5 safe cross-slot prefix attach gating

## Context

`Qwen3.5` now ships through packed paged-prefill on CUDA, but its hybrid
recurrent state is still not reconstructible from shared paged KV pages alone.
The scheduler previously treated any paged-prefill model as eligible for
cross-slot GPU prefix attach / staged readmission, which is correct for pure
attention models and unsafe for `Qwen3.5`.

This tranche tightens the runtime contract:

- `ModelForward` now exposes `supports_cross_slot_prefix_attach()`
- `Qwen3.5` returns `false`
- CUDA admission therefore uses same-slot stateful reuse or cold prefill for
  `Qwen3.5` instead of pretending that shared pages are enough to resume the
  hybrid state on a fresh slot

## Goal

Confirm that the safer `Qwen3.5` admission contract does not regress baseline
CUDA throughput/TTFT unexpectedly and does not reintroduce the earlier paged
prefix corruption class.

## Status

`pending-remote`

## Planned validation

- `scripts/bench_guidellm.sh qwen35-safe-cross-slot-prefix-attach`
- Compare against the latest `Qwen3.5` CUDA baseline and the previous
  packed-paged-prefill tranche
- Watch for:
  - TTFT regression from reduced prefix reuse
  - `c4/c8/c16` throughput shape
  - any correctness regressions on repeated-prefix traces

## Notes

This is a correctness-first contract fix. A later tranche can re-enable richer
cross-slot reuse only after `Qwen3.5` has a bounded-memory recurrent-state
restore story at reused prefix lengths.
