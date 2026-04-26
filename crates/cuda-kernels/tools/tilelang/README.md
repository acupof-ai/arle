# TileLang AOT Integration (Phase 0)

Build-time AOT for the prefill HD128 paged attention kernel, gated behind
`--features tilelang-attn`. Mirrors the Triton AOT track in
`tools/triton/`. See `docs/plans/tilelang-integration.md` for the full plan.

## What this covers

- One TileLang kernel: `batch_prefill_paged_hd128.py` (BF16, causal, page_size=16).
- Build-time CUBIN generation under `OUT_DIR/tilelang_aot/<artifact>/`.
- Generated C wrapper compiled into `libtilelang_kernels_aot.a` and linked
  alongside the Triton AOT artifacts.
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

If `nvidia-smi` is unavailable where you build, set the target SM manually:

```bash
export INFER_CUDA_SM=90
```

## Build

```bash
cargo build --release --features cuda,tilelang-attn
```

Artifacts land under `target/release/build/cuda-kernels-*/out/tilelang_aot/`.

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
