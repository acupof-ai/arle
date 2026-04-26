# TileLang AOT Integration (Phase 0)

Build-time AOT for the prefill HD128 paged attention kernel, gated behind
`--features tilelang-attn`. Mirrors the Triton AOT track in
`tools/triton/`. See `docs/plans/tilelang-integration.md` for the full plan.

## What this covers

- TileLang kernel: `batch_prefill_paged_hd128.py` (BF16, causal, page_size=16).
- AOT-specialized per Qwen3 head config in `SUPPORTED_HEADS`. Today:
  `(16,8)` 0.6B/1.7B, `(32,8)` 4B/8B, `(40,8)` 14B, `(64,8)` 32B.
  Build emits one cubin + C wrapper per config; Rust dispatches by
  `(num_q_heads, num_kv_heads)`. Add a new size by extending three
  in-lockstep lists (see comments in `build.rs` and `ffi/attention.rs`).
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

If `nvidia-smi` is unavailable where you build, set the target SM manually:

```bash
export INFER_CUDA_SM=90
```

## Build

The `tilelang-attn` feature is declared on `infer` and `cuda-kernels`, not on
the workspace root binary. Build the runtime crate directly:

```bash
cargo build --release -p infer --features cuda,tilelang-attn
```

For binaries that must be built through the workspace root (`arle`, `cli`)
the feature has to be forwarded there first; that is intentionally deferred
until Phase 1 because Phase 0 is evaluated by running the `infer` server
binary plus `scripts/bench_guidellm.sh`.

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
