# TileLang AOT Integration

Build-time AOT for CUDA kernels generated from TileLang. The CUDA feature
enables the TileLang paged-attention path through `tilelang-attn`; Qwen3.5
chunk-wise GDR remains on the validated Triton AOT path until the TileLang GDR
scaffold has a CUDA parity bench. See `docs/plans/tilelang-integration.md` and
`docs/plans/2026-05-05-cuda-kernel-tilelang-unification.md` for the full plan.

## What this covers

- TileLang attention kernels: `batch_prefill_paged_hd128.py`,
  `batch_prefill_paged_hd256.py`, and optional
  `batch_decode_paged_hd256.py`.
- AOT-specialized per Qwen head config. Build emits one cubin + C wrapper per
  config; Rust dispatches by `(num_q_heads, num_kv_heads)`. Add a new size by
  extending the lockstep lists in the kernel module, `build.rs`,
  `ffi/attention.rs`, and `infer/src/ops/attention.rs`.
- TileLang GDR scaffold: `gated_delta_rule.py` mirrors the seven Qwen3.5
  chunk-wise stages, but `build.rs` does not link it into the runtime ABI yet.
  Production GDR symbols are still generated from `tools/triton/`.
- Build-time CUBIN generation under `OUT_DIR/tilelang_aot/<artifact>/`.
- Generated C wrappers compiled into `libtilelang_kernels_aot.a` and
  linked alongside the Triton AOT artifacts.
- Compile-time dispatch: `cuda` enables `tilelang-attn` by default;
  `tilelang-decode-hd256` opts into the experimental HD256 decode tranche.

## Prerequisites

```bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Bootstrap a repo-local TileLang Python (separate from the Triton venv —
TileLang and Triton may want different Python envs):

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
relocation. Compare against the Triton AOT track which links the cubin
through Triton's own runtime.

## Current status

- TileLang version pinned during the H100 spike; see
  `docs/experience/wins/2026-04-26-bench-guidellm-cuda-tilelang-prefill-hd128-pending-remote.md`.
- Triton AOT remains the production owner for Qwen3.5 GDR. Removing it before
  a GPU-validated TileLang swap leaves unresolved GDR symbols or silently
  changes an unbenchmarked hot path.
- TileLang GDR is kept as source-level scaffold only; the runtime ABI swap is
  a separate Phase 2b change with e2e_qwen35 + guidellm gates.

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

If `tilelang.compile(...)` cannot AOT-export for `sm_90`, or if the prefill
kernel cannot express paged-KV BatchPrefill in the version pinned, the
generator exits non-zero and the build fails loudly. See
`docs/plans/tilelang-integration.md` §5 for the recorded error path.
