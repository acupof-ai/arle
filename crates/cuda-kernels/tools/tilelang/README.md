# TileLang AOT Integration

Build-time AOT for CUDA kernels generated from TileLang. The CUDA feature
uses TileLang as the only AOT compiler surface for paged attention and Qwen3.5
chunk-wise GDR. See `docs/plans/tilelang-integration.md` and
`docs/plans/2026-05-05-cuda-kernel-tilelang-unification.md` for the full plan.

## What this covers

- TileLang attention kernels: `batch_prefill_paged_hd128.py`,
  `batch_prefill_paged_hd256.py`, and
  `batch_decode_paged_hd256.py`.
- AOT-specialized per Qwen head config. Build emits one cubin + C wrapper per
  config; Rust dispatches by `(num_q_heads, num_kv_heads)`. Add a new size by
  extending the lockstep lists in the kernel module, `build.rs`,
  `ffi/attention.rs`, and `infer/src/ops/attention.rs`.
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

Bootstrap a repo-local TileLang Python:

```bash
uv venv crates/cuda-kernels/tools/tilelang/.venv
uv pip install -p crates/cuda-kernels/tools/tilelang/.venv/bin/python tilelang
```

Or, from the repo root: `pip install -e ".[tilelang]"`.

Point the build at that interpreter explicitly:

```bash
export INFER_TILELANG_PYTHON=$PWD/crates/cuda-kernels/tools/tilelang/.venv/bin/python
```

The build also probes `crates/cuda-kernels/tools/tilelang/.venv/bin/python`
and `.venv/bin/python` before falling back to `python3` / `python`.

If `nvidia-smi` is unavailable where you build, set the target SM manually
via the standard PyTorch env var:

```bash
export TORCH_CUDA_ARCH_LIST="9.0"               # H100 only
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"   # T1 fat binary
```

See [`docs/plans/sm-coverage.md`](../../../../docs/plans/sm-coverage.md) for tier policy.

## Build

Build through the workspace root when you want the `arle`/`cli` binaries:

```bash
cargo build --release --features cuda
```

Build the runtime crate directly when you only need `infer`:

```bash
cargo build --release -p infer --features cuda
```

For scripted server launches, set `INFER_FEATURES=cuda` before calling
`scripts/start_infer.sh`.

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
TileLang Metal `T.gemm` kernel on MPS. For a full local server/bench loop:

```bash
scripts/tilelang_metal_dev_backend.sh bench models/Qwen3-0.6B 8765
```

This is a Metal dev gate for the local TileLang checkout. The production
ARLE Metal inference path still runs through `metal_serve` +
`crates/mlx-sys`; replacing inference ops with TileLang-generated Metal
kernels requires a separate runtime integration.

## Risk gates

If `tilelang.compile(...)` cannot AOT-export for a target SM, or if the prefill
kernel cannot express paged-KV BatchPrefill in the version pinned, the
generator exits non-zero and the build fails loudly. See
`docs/plans/tilelang-integration.md` §5 for the recorded error path.
