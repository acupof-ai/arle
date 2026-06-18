# CUDA Path-Check Dedupe — pending remote bench, 2026-06-18

## Context

Review found two duplicated path-check sites:

- DSv4 chain verify accepted `is_chain()` plus `has_prefix_ancestors()` but did
  not bind the absolute `start_pos` in the same invariant.
- Qwen MoE offload/reload rebuilt routed expert pointer tables with a copied
  load-time branch tree.

## What Worked

- Replaced DSv4 split checks with `SpecVerifySchedule::is_prefix_chain_at`.
- Replaced duplicated MoE pointer-table branch trees with
  `build_moe_layer_pointer_tables`, shared by load and reload.
- The shared MoE helper rejects partial BF16/FP8 grouped caches explicitly.

## Verification

```bash
cargo fmt --check
git diff --check -- crates/infer-cuda/src/loader.rs crates/infer-cuda/src/dsv4.rs crates/infer-cuda/src/executor/spec_decode.rs
CUDARC_CUDA_VERSION=12090 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib
CUDARC_CUDA_VERSION=12090 cargo test -p infer-cuda --release --no-default-features --features cuda,no-cuda spec_decode --lib
CUDARC_CUDA_VERSION=12090 cargo clippy -p infer-api --release --no-default-features --features cuda,no-cuda --lib -- -D warnings
```

All passed locally. CUDA kernels / runtime bench remain `pending-remote`: this
host uses the no-cuda gate, `nvcc` is unavailable locally, and the user has
forbidden using nodes 61 and 62.

## Bench Status

- **Backend:** CUDA
- **Model:** DSv4 / Qwen3.6 MoE affected code paths
- **Result:** pending remote CUDA bench
- **Reason:** no allowed CUDA pod in this turn

## Rule

Path-shape checks should live in one helper that names the full invariant,
including absolute start position or concrete cache layout, not as repeated
local boolean fragments.
