# TileLang AOT Integration

Build-time AOT for CUDA kernels generated from TileLang. The CUDA feature
uses TileLang as the only AOT compiler surface for paged attention and Qwen3.5
chunk-wise GDR.

## What this covers

- TileLang attention kernels: `batch_prefill_paged_hd128.py`,
  `batch_prefill_paged_hd256.py`, and
  `batch_decode_paged_hd256.py`.
- AOT-specialized per Qwen head config and target SM. Build emits one cubin + C
  wrapper per `(config, SM)` and runtime dispatches by model shape plus the active
  device SM. Add a size by updating `kernels.toml`, `build.rs`, and the matching
  FFI/runtime call sites.
- TileLang GDR scaffold: `gated_delta_rule.py` mirrors the Qwen3.5 chunk-wise
  stages that TileLang 0.1.9 can lower on sm_89; the strict-lower triangular
  solve symbol is native CUDA C in `csrc/misc/gdr_prefill_solve.cu`.
- Build-time CUBIN generation under `OUT_DIR/tilelang_aot/<artifact>/`.
- Generated C wrappers compiled into `libtilelang_kernels_aot.a` and
  linked with the native CUDA C kernels.
- Compile-time dispatch: `cuda` enables the complete TileLang CUDA backend.

## Prerequisites

```bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Bootstrap the pinned, patched repo-local toolchain from the repo root:

```bash
scripts/pod-tilelang-env.sh
export INFER_TILELANG_PYTHON=$PWD/crates/cuda-kernels/tools/tilelang/.venv/bin/python
```

The build uses the exact `INFER_TILELANG_PYTHON` path; it does not resolve venv
symlinks to a system interpreter. It requires `import tilelang`, the package's
bundled `lib/`, `src/`, and CUTLASS trees, plus the installed `tvm_ffi` package.
A separate top-level `tvm` package is not required: TileLang loads its bundled
TVM during import. These runtime/codegen trees are hashed into the persistent
kernel-cache identity.

The build also probes `crates/cuda-kernels/tools/tilelang/.venv/bin/python`
and `.venv/bin/python` before falling back to `python3` / `python`.

If `nvidia-smi` is unavailable where you build, set the target SM manually
via the standard PyTorch env var:

```bash
export TORCH_CUDA_ARCH_LIST="9.0"               # H100 only
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"   # T1 fat binary
```

## Build

Build through the workspace root when you want the `arle`/`cli` binaries:

```bash
cargo build --release --features cuda
```

Build only the kernel producer crate:

```bash
cargo build --release -p cuda-kernels --features cuda
```

A T1 candidate is generated and packed once on an sm_90 build.
Cold consumers fetch those exact bytes and build with
`INFER_TILELANG_PYTHON=/usr/bin/false`; they cannot regenerate the candidate.
GPU qualification emits fragments bound to the candidate archive SHA-256,
bundle ID, kernel build ID, product binary SHA-256, tested SM, workload profile,
and exercised capabilities. Aggregation requires the complete policy set.
Qualification publishes the original archive and checksum unchanged, adding only
a `.qualification.json` sidecar.

The generated per-SM C++ wrapper exports its CUDA entry point with C linkage;
the C dispatch wrapper preserves the same ABI for Rust `extern "C"` callers.
Artifacts land under `target/release/build/cuda-kernels-*/out/tilelang_aot/`.
The generated C wrapper embeds the cubin bytes via `cuModuleLoadData`, so
the produced binary is self-contained and survives `cargo clean` /
relocation.

## Current status

- TileLang version pinned during the H100 spike; see
  `docs/experience/wins/2026-04-26-bench-guidellm-cuda-tilelang-prefill-hd128-pending-remote.md`.
- TileLang paged prefill HD128/HD256, HD256 decode, and the AOT-compatible
  Qwen3.5 GDR stages are linked under `--features cuda`.
- The old external AOT and wrapper surfaces have been removed from the
  CUDA runtime. New attention/GDR kernels should be added through
  `tools/tilelang/` or native CUDA C only.

## macOS Metal dev checkout

For local ARLE development against an upstream TileLang Metal branch, use the
repo-level wrapper:

```bash
ARLE_TILELANG_REPO=/tmp/tilelang-metal-pr \
ARLE_TILELANG_PYTHON=/tmp/arle-tilelang-mac-venv/bin/python \
  scripts/tilelang_metal_dev_backend.sh smoke
```

The smoke imports TileLang from that checkout, lowers ARLE's in-tree
`batch_prefill_paged_hd128.py` attention kernel to Metal, and executes a
TileLang Metal `T.gemm` kernel on MPS. This is a development-only compiler gate;
the production Metal executor remains `crates/mlx-sys`.

## Risk gates

If `tilelang.compile(...)` cannot AOT-export for a target SM, or if the prefill
kernel cannot express paged-KV BatchPrefill in the version pinned, the
generator exits non-zero and the build fails loudly.
