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
  archives and skips `nvcc`, TileLang AOT, and DeepEP sidecar compilation.
- Added `ARLE_NVCC_WRAPPER`: main CUDA `.cu` compilation can now run through a
  wrapper such as `sccache`.
- Added `ARLE_NVCC_SPLIT_COMPILE`: optional bounded `nvcc --split-compile=N`
  for full rebuilds.
- Added `ARLE_CUDA_KERNEL_SET=dsv4_flash`: keeps native CUDA C + FlashMLA but
  replaces non-DSv4 TileLang AOT symbols with `CUDA_ERROR_NOT_SUPPORTED` stubs,
  avoiding Qwen/GDR TileLang AOT cost for DSv4-Flash validation binaries.

## Verification

- `cargo fmt --check`
- `git diff --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`
- `CUDARC_CUDA_VERSION=12080 ARLE_CUDA_KERNELS_PREBUILT_DIR=/tmp/arle-prebuilt-test cargo check -p cuda-kernels --no-default-features --features cuda`

The last check used empty placeholder archives and verified the prebuilt branch
is selected before any `nvcc` or TileLang probe. A real release build still
needs real archives produced under the matching CUDA/SM/feature/source hash.

## Rule

Separate CUDA artifact production from Rust binary iteration. Rust-only DSv4
changes should link a validated kernel artifact set; only CUDA source or AOT
spec changes should rebuild native CUDA and TileLang kernels.
