# Metal MLX Array Byte Export Pending Bench

## Context

P0-1A SSD KV persistence needs a bridge primitive that can serialize and
restore MLX arrays without losing dtype or shape metadata. This tranche adds
raw byte export/import wrappers for MLX arrays and tests them independently.

## What Worked

- `crates/mlx-sys` now exposes array byte count and contiguous byte export.
- `infer::backend::metal::mlx::MlxArray` validates shape/dtype byte lengths
  before importing raw bytes.
- The API is not wired into the scheduler hot path yet; it only materializes
  arrays when a caller explicitly asks to export bytes.

## Benchmark

Status: `pending-remote`

No guidellm throughput run was executed for this tranche because the new API is
not used by request execution or prefix publish yet. The first runtime-impacting
bench is attached to the follow-up snapshot persistence tranche that calls this
API from `MetalQwen35PrefixRuntime`.

## Rule

Keep byte export as an explicit persistence-boundary API. Do not call it from
per-token decode or normal prefill scheduling.
