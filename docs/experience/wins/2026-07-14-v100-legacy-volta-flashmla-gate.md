# Legacy Volta (sm_70, V100) build excludes SM90-only FlashMLA

**Date:** 2026-07-14. **Backend:** CUDA, legacy Volta (sm_70, V100).
**Scope:** `crates/cuda-kernels/build.rs` — FlashMLA enablement gate.
**Status: pending-remote** — fix compiles clean (build.rs edition-2024 parse
verified on Mac); full `cargo build --features cuda` + `arle --doctor` / serve
verify to run on the V100 box (sm_70 test lane).

## Context

The V100 (sm_70) new-user flow failed to build: `cargo build --features cuda`
with `TORCH_CUDA_ARCH_LIST=7.0` errored in `vendor/flashmla` with
`__nv_fp8_e8m0` undefined and sm80 asm-operand mismatches. FlashMLA is
SM90-only sparse-FP8 (prefill + decode); sm_70 has no FP8, so the SM90
instantiations cannot compile there.

`enable_flashmla` (build.rs:2230) only checked the vendored-tree presence and
`ARLE_CUDA_DISABLE_FLASHMLA` — it did not exclude legacy Volta. DeepGEMM native
(build.rs:2390) and FA3 (opt-in, default off) were already correctly gated;
FlashMLA was the remaining gap.

## What Worked

Added `&& !legacy_volta_build` to the `enable_flashmla` predicate, where
`legacy_volta_build = has_legacy_volta(&sm_targets)`. The binding was hoisted
to right after `sm_targets` is computed (build.rs:2182) so both the FlashMLA
gate and the existing Marlin-W4-FP8 gate (build.rs:2405) share it. When
disabled, the existing `if !enable_flashmla` branch drops the SM90-coupled
shims and compiles the `cudaErrorNotSupported` stub — the same path an
explicit opt-out uses. The runtime FlashMLA gates default OFF, so the stub is
never actually called.

BF16-only on sm_70 remains the rule: FP8 KV and DSv4 HD64 wrappers return
`cudaErrorNotSupported` (documented in environment.md Legacy Volta tier).

## Rule

- **Per-backend-SM build gates must enumerate every SM90-only path.** A
  vendored kernel tree that compiles SM90/FP8 instantiations (FlashMLA) breaks
  a legacy Volta build unless explicitly excluded — same as the already-gated
  DeepGEMM native bridge.
- **SM-pinned legacy tiers can't mix targets.** `has_legacy_volta` +
  `sm_targets.len() != 1` already panics (build.rs:97); the FlashMLA gate
  relies on the same single-sm legacy set.

Verify (V100 box, pending):
`RUSTC_WRAPPER= CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=7.0 cargo build --release --features cuda --bin arle`.
