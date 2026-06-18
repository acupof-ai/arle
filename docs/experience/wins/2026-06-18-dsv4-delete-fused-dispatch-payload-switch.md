# DSv4 delete obsolete fused dispatch payload switch

## Context

The fused dispatch payload control had become a stale configuration surface:
current Rust hot paths no longer called the dispatch-payload pack/unpack FFI,
but scripts and docs still exported/described the old switch. That made bench
configs look more meaningful than they were.

## What Worked

- Removed the obsolete script and documentation references.
- Removed the unused pack/unpack FFI declarations.
- Removed the unused CUDA pack/unpack kernels.
- Changed DSv4 profiling controls from presence-based to truthy-only, so
  `0` / `false` no longer accidentally enable synchronizing profilers.

## Verification

- `rg` over the repository returns no remaining obsolete fused-dispatch switch
  or dispatch-payload pack/unpack symbol hits.
- Mac no-CUDA typecheck pending in this change set; no serving benchmark was
  run because the deleted pack/unpack FFI had no Rust call sites at HEAD.

## Rule

Delete stale bench knobs when the implementation path they imply no longer
exists. A dead env var in scripts is enough to contaminate performance analysis.
