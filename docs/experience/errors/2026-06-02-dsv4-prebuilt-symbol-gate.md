# DSv4 Prebuilt Archive Symbol Gate

## Context

While validating the DSv4 SGLang-profile routing tranche on the remote pod, the
`scripts/dsv4_fast_build.sh` prebuilt path produced a fast release binary, but a
CUDA/NCCL lib-test link failed with missing DSv4 FFI symbols:

- `dsv4_deepgemm_native_preflight_cuda`
- `arle_dsv4_fp8_kv_fill_one_sw_slot_from_start_pos_cuda`
- `arle_dsv4_flashmla_decode_build_indices_start_pos_ptr_cuda`
- `arle_dsv4_output_inverse_rope_cuda`
- `dsv4_update_window_cache_start_pos_ptr_cuda`

The source files and archive members existed, but `nm -g` showed the accepted
prebuilt archive did not export the new C ABI symbols.

## Root Cause

`dsv4_fast_build.sh` accepted a prebuilt archive by manifest, then still ran
`harvest_prebuilt` after the build. In a prebuilt build, `cuda-kernels/build.rs`
skips nvcc and does not emit a fresh `OUT_DIR/libkernels_cuda.a`, so harvesting
could copy an older `target/release/build/.../out` archive and stamp it with a
current manifest. That created a false-positive cache: tree hash looked current,
but the archive was stale.

The crate build script also trusted `ARLE_CUDA_KERNELS_PREBUILT_DIR` without
checking the DSv4 symbols it was about to link.

## Fix

`scripts/dsv4_fast_build.sh` now:

- validates required DSv4 symbols before accepting a prebuilt archive;
- validates the same symbols before harvesting an nvcc-built archive;
- skips harvesting entirely when the build used the prebuilt fast path.

`crates/cuda-kernels/build.rs` now validates required DSv4 symbols before
linking a caller-provided prebuilt archive.

## Rule

For DSv4, a prebuilt CUDA archive is not valid because its manifest matches.
It is valid only if `nm -g libkernels_cuda.a` proves the current DSv4 FFI symbol
set is exported. Build fast paths must fail closed before runtime fallback.
