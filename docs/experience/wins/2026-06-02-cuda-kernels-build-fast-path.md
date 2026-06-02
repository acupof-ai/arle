# CUDA kernel build fast path

## Context

Small Rust-side DSv4 changes were triggering a full `cuda-kernels` rebuild on
the pod. That rebuild recompiles native CUDA C, FlashMLA instantiations, and
all TileLang AOT families. The observed failure was unrelated to the Rust
change: the current TileLang package failed on
`tilelang_batch_prefill_paged_hd128_q16_kv8_run` with a pipeline-stage
planning mismatch.

## What Worked

- Added `ARLE_CUDA_KERNELS_PREBUILT_DIR`: when it points at a directory with
  `libkernels_cuda.a` and `libtilelang_kernels_aot.a`, `build.rs` links those
  archives and skips `nvcc` and TileLang AOT. If the same directory contains
  `arle_deepep_sidecar`, the sidecar path is baked in too.
- Added `ARLE_DEEPEP_SIDECAR_PREBUILT`: reuse only the ARLE DeepEP sidecar
  binary without rebuilding it.
- Added `ARLE_NVCC_WRAPPER`: main CUDA `.cu` compilation can now run through a
  wrapper such as `sccache`; follow-up coverage includes the ARLE DeepEP
  sidecar and `deepep-sys` native build.
- Added `ARLE_NVCC_SPLIT_COMPILE`: optional bounded `nvcc --split-compile=N`
  for full rebuilds; follow-up coverage includes the ARLE DeepEP sidecar and
  `deepep-sys` native build.
- Added `ARLE_CUDA_KERNEL_SET=dsv4_flash`: keeps native CUDA C + FlashMLA but
  replaces non-DSv4 TileLang AOT symbols with `CUDA_ERROR_NOT_SUPPORTED` stubs,
  avoiding Qwen/GDR TileLang AOT cost for DSv4-Flash validation binaries.
- Added `scripts/dsv4_fast_build.sh`: one command that prefers `sccache`, uses
  the `release-fast` profile, defaults to `kernel-set=dsv4_flash`, reuses a
  harvested CUDA artifact directory, and harvests artifacts after the first
  full build.
- Added `release-fast`: an iteration profile with no LTO, more codegen units,
  and incremental enabled. Final SLO/perf numbers still use the existing
  `--release` profile.

## Verification

- `cargo fmt --check`
- `git diff --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`
- `CUDARC_CUDA_VERSION=12080 ARLE_CUDA_KERNELS_PREBUILT_DIR=/tmp/arle-prebuilt-test cargo check -p cuda-kernels --no-default-features --features cuda`
- `bash -n scripts/dsv4_fast_build.sh`

The last check used empty placeholder archives and verified the prebuilt branch
is selected before any `nvcc` or TileLang probe. A real release build still
needs real archives produced under the matching CUDA/SM/feature/source hash.

## Rule

Separate CUDA artifact production from Rust binary iteration. Rust-only DSv4
changes should link a validated kernel artifact set; only CUDA source or AOT
spec changes should rebuild native CUDA and TileLang kernels.
