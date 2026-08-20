# Quant-linear storage states validated at load — 2026-08-20

> Status: **pending-remote.** Host-only route/storage test passes locally on
> `cargo check --tests`; the Mac `cuda,no-cuda` test binary cannot link the CUDA
> C symbols, so execution is a pod gate (plan Tranche 2, dispatch-consolidation).

## Context

`DeviceMatrix` stores each representation in independent `Option` fields, and
the repacks free their sources inline. An invalid combination — half a Marlin
pair, `qweight_u8` without `scale_f32`, NVFP4 whose repack silently no-opped on
an incomplete source triplet — surfaced only as a serve-time missing-buffer
error. The W8A16 untied `lm_head` defect (2026-08-20-qwen-spec-budget-and-
w8-lm-head.md) was this class: source freed, one lane with no consumer.

## Root Cause

No final-state check after repack/release. Routing reconstructed the storage
contract from field presence at GEMM time, per lane.

## Fix

- `quant_linear::validate_storage` (entry via `ops::validate_quant_linear_storage`),
  called once at the end of `loader.rs::marlin_repack_dense` — the choke point
  for all four dense quant load paths (plain, untied lm_head, row-fused, TP-sharded).
- Predicates live with their route owners: `fp8_missing_representation`,
  `fp4_missing_representation`, `int_missing_representation` (pure, host-testable);
  Dsv4 source-pair check inline. Errors carry tensor name, format, shape, group
  size, and the missing representation.
- Path audit: LoRA already hard-errors on `quant_source_freed()`
  (`qwen35_lora.rs:178,619`); offload/reload round-trips every Option field and
  the `fp8_deepgemm_prefill` / `fp4_marlin_scale_lift_inv` plain fields, so the
  restored route equals the pre-offload route; fuse repacks once post-fuse then
  validates; a repack-declined TP shard keeps its source pair and validates.
- `storage_states` table test added beside the route tests (host-only).

## Remote gate

1. An incompatible checkpoint (e.g. stripped `scale_f32`) must fail at load
   with the tensor-named error, never at first token.
2. Offload → reload must restore the same route counters per implementation ID.

## Rule

Validate the retained-representation set once, where the load releases sources —
every reachable M must have a resident consumer before the matrix publishes.
