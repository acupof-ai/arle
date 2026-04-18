# `infer::backend::metal` — Agent Guide

Apple Silicon Metal backend via `crates/mlx-sys`. Serial single-request today,
with an accounting-only scheduler wired for a future hot-path upgrade. Load
before touching any Metal code.

## Module map

```
metal.rs                — MetalBackend, InferenceBackend + StreamingInferenceBackend impl
metal/config.rs         — MetalModelConfig, quant config parsing (serde)
metal/loader.rs         — safetensors → MLX unified memory
metal/weights.rs        — MetalWeights, projection fusion, tensor merging
metal/forward.rs        — rust_transformer_layer (Qwen3 path)
metal/qwen35.rs         — Qwen3.5 path (delegates to mlx-sys C++ step model)
metal/ops.rs            — MLX-backed linear, extend_kv_cache, clear_metal_cache
metal/mlx.rs            — thin mlx-sys wrappers (MlxArray, slice, take_axis, eval, rms_norm, …)
metal/generate.rs       — top-level generate loop, KV_CACHE_CHUNK, MetalGenerateOutput
metal/sampling.rs       — gpu_sample_token
metal/dflash.rs         — Metal DFlash speculative draft runtime
metal/kv_pool.rs        — KV pool accounting (not yet on the hot path)
metal/prefix_cache.rs   — Metal prefix cache accounting
metal/gdr.rs            — Metal draft runtime glue
metal/scheduler.rs      — MetalScheduler (CPU accounting skeleton, decode-priority + chunked prefill)
```

## Feature gating

- **Build:** `--no-default-features --features metal`. Never combine with
  `cuda`; use `no-cuda` alongside for pure Mac dev.
- **Most files are `#[cfg(feature = "metal")]`.** `config.rs`, `kv_pool.rs`,
  `prefix_cache.rs`, `scheduler.rs`, and `gdr.rs` are **always-on** so
  `cargo check --features no-cuda` still validates the scheduler skeleton.
- The `mlx` submodule + FFI imports are strictly gated — anything in
  `metal.rs` that calls `mlx::*` or `linear(...)` must be inside a
  `#[cfg(feature = "metal")]` block.

## Invariants (violating these is the #1 source of Metal bugs)

1. **No repo-local `.metal` shaders.** Actual Metal kernels live inside MLX,
   built by `crates/mlx-sys` via cmake FetchContent. If you need a new
   kernel, add it to `crates/mlx-sys/src/mlx_bridge.cpp` (C++ side) —
   never into `infer/src/`.
2. **MLX unified memory = self-memcpy on Apple Silicon.** There is no host
   pinned tier (T1 skipped — see `kv_tier/AGENTS.md`). Don't add PCIe-style
   prefetch paths; they're no-ops at best.
3. **`fast::rope` layout is `[B, heads, seq, d]`**, not `[B, seq, heads, d]`.
   Transpose **before** calling rope so `T = seq`, not `T = heads`.
   (Auto-memory: `feedback_mlx_rope_layout.md`, `feedback_mlx_rope_axis.md`;
   incident: `docs/experience/errors/2026-04-02-rope-axis-bug.md`.)
4. **Metal scheduler is on the hot path** as of `M0.2b` (2026-04-15).
   `run_metal_scheduler_runtime` (`runtime.rs:639`) drives
   `MetalScheduler::step()` → `PrefillChunk` / `DecodeBatch` / `Mixed` /
   `Idle` and dispatches each decision through `execute_prefill_chunk` /
   `execute_decode_batch`. Qwen3.5 same-length batched decode runs through
   `CachedQwen35DecodeBatch` (`runtime.rs:1057`) with `retain_rows` (shrink)
   + `admit_rows` (prefix-preserving grow via `admit_row_indices`). Qwen3
   batched decode goes through `MetalRequestState::decode_batch` →
   `decode_qwen3_batch` and still requires same-length. **Variable-length
   decode batching is scaffolded but not yet enabled** — see the
   left-padding pattern in invariant #8 and the open blocker in
   [`docs/experience/errors/2026-04-16-metal-varlen-rope-blocker.md`](../../../docs/experience/errors/2026-04-16-metal-varlen-rope-blocker.md).
5. **Qwen3 and Qwen3.5 take different paths.** Qwen3 runs through the Rust
   `rust_transformer_layer` path in `forward.rs`. Qwen3.5 delegates to the
   dedicated C++ step model in `qwen35.rs` + `mlx-sys/src/mlx_qwen35_model.cpp`
   — don't mix them.
6. **DFlash (speculative decode) is experimental and optional.** Guarded by
   `MetalDflashOptions`; empty draft model = feature off. See
   `docs/resources/metal-dflash.md` for user-facing flags. DFlash dispatches
   from the scheduler runtime via `execute_decode_single`
   (`runtime.rs:1051-1056`) — one row at a time. When a multi-row tick has
   `open.len() >= 2`, the scheduler permanently disables DFlash on every row
   so they all join the packed batch (`runtime.rs:1040-1045`); this is a
   policy choice, not a wiring gap, and the Layer 2 verify-batch plan
   (`docs/plans/metal-dflash-qwen35-verify-batch.md`) lifts it once packed
   speculative verify lands.
7. **Variable-length decode uses left-padding + additive mask + per-row
   RoPE offsets** (mlx-lm `BatchKVCache` pattern).
   `Qwen35PackedDecodeBatch` carries a shared `batch_cache_len` cursor
   and a per-row `left_padding: Vec<i32>`. Every batched-decode step
   passes two supplementary arrays through the C++ bridge:
   - `attn_mask` from `build_varlen_decode_mask` (`mlx.rs`) — additive
     `[B, 1, 1, key_len]` with `-inf` at columns `[0, left_padding[b])`
     — only materialized when at least one row is padded.
   - `rope_offsets`, an int32[B] vector of per-row logical positions
     (`batch_cache_len - left_padding[row]`) — **always** materialized
     for batched decode. Both same-length AND varlen batches take the
     array-offset RoPE path because MLX 0.31.1's scalar-offset
     `fast::rope` silently drops batch rows > 0 on `[B, H, S=1, D]`
     input (see the tripwire tests in `backend::metal::mlx::tests` and
     `docs/experience/errors/2026-04-16-metal-varlen-rope-blocker.md`).
   `left_pad_kv_cache_row` + `strip_left_padding_from_packed_row`
   (`request_state.rs`) convert between the per-request zero-aligned
   layout and the batch-shared left-aligned layout. `admit_rows` appends
   new shorter rows into an active packed batch without a full rebuild;
   the runtime admit pre-check is `cache_len <= batch_cursor`.
8. **The hot-path runtime is `run_metal_scheduler_runtime`** in
   `backend/metal/runtime.rs`, NOT the legacy `backend/runtime.rs`.
   The backend exposes `InferenceBackend::generate` and
   `StreamingInferenceBackend::generate_stream`; `metal_serve` routes
   traffic through the scheduler runtime. Concurrency, admission,
   cancellation, prefix reuse, and KV-pool lifecycle all live on the
   scheduler side. The legacy serial runtime is retained only for DFlash
   (see invariant #6).

## Build requirements

- Xcode + Command Line Tools (Apple Silicon host only).
- `cmake` for MLX (`crates/mlx-sys/build.rs` uses `cmake::Config`).
- First build downloads + compiles MLX 0.31.1 via FetchContent — slow,
  cached under `target/.../build/mlx-sys-*/out/`.

## Pointers

- `crates/mlx-sys/AGENTS.md` — the bridge layer below this.
- `docs/plans/2026-04-15-metal-backend-execution-checklist.md` — active
  prioritized backlog.
- `docs/plans/2026-04-15-metal-backend-acceptance-plan.md` — acceptance gates
  for turning Metal from beta into production.
- `docs/resources/metal-dflash-params.md` — DFlash CLI parameter reference.
- `docs/experience/errors/2026-04-09-metal-optimization-pitfalls.md` — Metal-specific
  optimization gotchas collected from earlier waves.
- `docs/experience/errors/2026-04-16-metal-varlen-rope-blocker.md` —
  why variable-length decode batching is scaffolded but not yet enabled,
  and what Phase 2 must solve (per-row RoPE via `fast::rope(..., array
  offset)`).
- `docs/experience/wins/2026-04-14-metal-dflash-qwen3.md` — reference win
  (5.9× decode on M4 Pro).
