# DSv4 DeepGEMM preflight makes the fast path explicit

## Context

A DSv4 performance run passed `ARLE_DSV4_EXPERT_BACKEND=deepgemm-auto`, but the
first request showed `DeepGEMM native bridge failed` and then fell back to the
native grouped expert path. The startup logs only said the FP8 expert cache was
built, which is not proof that the DeepGEMM JIT kernel can compile or launch.

## What Worked

- Added a native bridge preflight FFI that checks the exact runtime JIT inputs:
  `ARLE_DEEPGEMM_LIBRARY_ROOT`, CUTLASS `cutlass/arch/barrier.h`, `nvcc`,
  `cuobjdump`, CUDA home, and the JIT cache root.
- Changed `deepgemm-auto` to run that preflight during model load before
  building the resident FP8 expert cache. If the bridge is unavailable, the
  fallback reason is logged before the first request.
- Kept `deepgemm` as required mode: a missing native bridge dependency fails
  fast instead of silently serving on the native expert backend.
- Updated the DSv4 build/toolchain helpers to resolve and print
  `ARLE_DEEPGEMM_CUTLASS_INCLUDE`, falling back to FlashMLA's vendored CUTLASS
  when the DeepGEMM submodule is not populated.

This is a high-performance path guard, not a fallback optimization. The intended
fast path is DeepGEMM; the change prevents a native fallback from looking like a
valid DeepGEMM run.

## Verification

- `cargo fmt --check`
- `bash -n scripts/dsv4_fast_build.sh scripts/dsv4_toolchain.sh`
- `git diff --check`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,no-cuda`
- `cargo check -p infer --no-default-features --features no-cuda`

Remote DSv4 pod build and startup-log verification are pending for the follow-up
entry/update.

## Rule

Do not treat a DSv4 FP8 expert cache as proof that DeepGEMM is active. The
runtime must prove the native DeepGEMM JIT dependencies before cache build and
before any performance run.
