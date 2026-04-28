# TileLang AOT Integration (Phase 0)

Build-time AOT for the prefill HD128 paged attention kernel, gated behind
`--features tilelang-attn`. Mirrors the Triton AOT track in
`tools/triton/`. See `docs/plans/tilelang-integration.md` for the full plan.

## What this covers

- TileLang kernel: `batch_prefill_paged_hd128.py` (BF16, causal, page_size=16).
- AOT-specialized per Qwen3 head config in `SUPPORTED_HEADS`. Today:
  `(16,8)` 0.6B/1.7B, `(32,8)` 4B/8B, `(40,8)` 14B, `(64,8)` 32B.
  Build emits one cubin + C wrapper per config; Rust dispatches by
  `(num_q_heads, num_kv_heads)`. Add a new size by extending the lockstep
  lists in this kernel module, `build.rs`, `ffi/attention.rs`, and
  `infer/src/ops/attention.rs`.
- Build-time CUBIN generation under `OUT_DIR/tilelang_aot/<artifact>/`.
- Generated C wrappers compiled into `libtilelang_kernels_aot.a` and
  linked alongside the Triton AOT artifacts.
- Compile-time dispatch: `--features tilelang-attn` swaps the FlashInfer
  `_run` call for the TileLang one. Default builds are unchanged.

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
cargo build --release --features cuda,tilelang-attn
```

Build the runtime crate directly when you only need `infer`:

```bash
cargo build --release -p infer --features cuda,tilelang-attn
```

For scripted server launches, set `INFER_FEATURES=cuda,tilelang-attn` before
calling `scripts/start_infer.sh`.

Artifacts land under `target/release/build/cuda-kernels-*/out/tilelang_aot/`.
The generated C wrapper embeds the cubin bytes via `cuModuleLoadData`, so
the produced binary is self-contained and survives `cargo clean` /
relocation. Compare against the Triton AOT track which links the cubin
through Triton's own runtime.

## Phase 0 status

- TileLang version pinned during the H100 spike; see
  `docs/experience/wins/2026-04-26-bench-guidellm-cuda-tilelang-prefill-hd128-pending-remote.md`.
- Default build (no `tilelang-attn`) is byte-identical to before this track
  landed.

## Risk gates

If `tilelang.compile(...)` cannot AOT-export for `sm_90`, or if the prefill
kernel cannot express paged-KV BatchPrefill in the version pinned, the
generator exits non-zero and the build fails loudly. See
`docs/plans/tilelang-integration.md` §5 for the recorded error path.
