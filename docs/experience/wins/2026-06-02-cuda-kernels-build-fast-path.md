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
- Split FlashMLA sparse prefill from sparse-FP8 decode at build time:
  prefill remains enabled with FlashMLA, while decode real kernels are opt-in
  via `ARLE_CUDA_ENABLE_FLASHMLA_DECODE=1`. CUDA 12.5 H20 builds otherwise fail
  on `__nv_fp8_e8m0`, and the runtime decode gate is default-off anyway.
- Added `scripts/dsv4_fast_build.sh`: one command that prefers `sccache`, uses
  the `release-fast` profile, defaults to `kernel-set=dsv4_flash`, reuses a
  harvested CUDA artifact directory, and harvests artifacts after the first
  full build. The script writes `arle-cuda-kernels.manifest` and refuses stale
  caches whose CUDA source tree / SM list / CUDA version / build-env key no
  longer matches. It also prefers the newest `/usr/local/cuda-*` toolkit when
  `CUDA_HOME` is unset, instead of blindly using the generic `/usr/local/cuda`
  symlink.
- Follow-up fix: `scripts/dsv4_fast_build.sh` now defaults
  `ARLE_CUDA_ENABLE_FLASHMLA_DECODE=1` when `ARLE_CUDA_KERNEL_SET=dsv4_flash`.
  Without this, a real DSv4-Flash prebuilt artifact could be skipped because
  the manifest saw `enable_flashmla_decode=` while the intended artifact was
  built with `enable_flashmla_decode=1`.
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
- Remote H20 probe, isolated checkout `/data01/build/arle-build-fast-check`:
  stale archives without a manifest were rejected by the script instead of
  being linked into a mismatched binary. Directly forcing those old archives
  reproduced undefined symbols, confirming the manifest guard is required.
- Remote H20 first full `dsv4_flash` build found the next compile blocker:
  vendored FlashMLA sparse-FP8 decode fails on CUDA 12.5 with undefined
  `__nv_fp8_e8m0`. `build.rs` still stubs decode by default and compiles it
  only when `ARLE_CUDA_ENABLE_FLASHMLA_DECODE=1`; the DSv4 fast-build helper
  opts into real decode for the H20/CUDA-12.8 validation lane.
- The same pod also had `/usr/local/cuda-12.9`; the first failing run used the
  generic `/usr/local/cuda -> 12.5` symlink. The fast-build script now prefers
  the newest explicit toolkit directory.
- Remote DSv4 pod `/data01/build/arle @ 17050ba4`: after rebuilding once with
  `enable_flashmla_decode=1`, the immediate follow-up
  `bash scripts/dsv4_fast_build.sh` printed
  `using CUDA prebuilt artifacts` and
  `Using prebuilt CUDA kernel artifacts ... skipping nvcc and TileLang AOT`.
  Wall time was 4.92 s.

The last check used empty placeholder archives and verified the prebuilt branch
is selected before any `nvcc` or TileLang probe. A real release build still
needs real archives produced under the matching CUDA/SM/feature/source hash.

## Rule

Separate CUDA artifact production from Rust binary iteration. Rust-only DSv4
changes should link a validated kernel artifact set; only CUDA source or AOT
spec changes should rebuild native CUDA and TileLang kernels.
