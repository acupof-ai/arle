# `infer::backend::metal` — Agent Guide

Apple Silicon Metal backend via `crates/mlx-sys`. The scheduler runtime is the
live hot path: decode-first continuous batching, chunked prefill, Qwen3.5
packed decode, and optional DFlash all execute through `runtime.rs`. Load
before touching any Metal code.

## Refactor posture

- Keep the Metal path simple and uniform. Prefer deletion-style refactors:
  remove stale fallback/policy layers, collapse duplicate scheduler/runtime
  flows, and keep one canonical hot path instead of stacking special cases.

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
metal/scheduler.rs      — MetalScheduler policy (decode-first step + optional prefill chunk)
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
   `run_metal_scheduler_runtime` drives one `MetalScheduleStep` per tick:
   `decode` first, then an optional `prefill` chunk. Qwen3.5 packed decode runs through
   `CachedQwen35DecodeBatch` (`runtime.rs`) with `retain_rows` (shrink) +
   `admit_rows` (prefix-preserving grow via `admit_row_indices`) and
   supports variable-length rows via a shared `batch_cache_len` cursor plus
   per-row `left_padding`. Qwen3 batched decode still goes through
   `MetalRequestState::decode_batch` → `decode_qwen3_batch` and still
   requires same-length rows.
5. **Qwen3 and Qwen3.5 take different paths.** Qwen3 runs through the Rust
   `rust_transformer_layer` path in `forward.rs`. Qwen3.5 delegates to the
   dedicated C++ step model in `qwen35.rs` + `mlx-sys/src/mlx_qwen35_model.cpp`
   — don't mix them.
6. **DFlash (speculative decode) is experimental and optional.** Guarded by
   `MetalDflashOptions`; empty draft model = feature off. See
   `docs/resources/metal-dflash.md` for user-facing flags. DFlash dispatches
   from the scheduler runtime as part of the single decode path. A lone
   DFlash row falls through to `execute_decode_single`; two or more DFlash
   rows use `execute_qwen35_dflash_packed_batch`. Plain rows still batch
   through `execute_qwen35_packed_decode_batch`.
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
   scheduler side.

## Metal vs CUDA: mental model differences

- **Metal is not "CUDA with different syntax".** This backend rides on
  MLX lazy graphs plus the `mlx-sys` C++ bridge; CUDA uses explicit
  kernels (`cudarc` + FlashInfer + Triton AOT). Porting an optimization
  from CUDA to Metal verbatim is usually wrong.
- **In Metal, `.item()` / `eval()` / `async_eval()` are scheduling
  boundaries.** A stray scalar materialization can turn an overlapped
  graph into a fully synchronous step and blow up TTFT. Treat every
  eager boundary as hot-path API design, not harmless plumbing.
- **Unified memory changes the failure mode.** Apple Silicon does not
  have the PCIe copy boundaries or host-pinned staging patterns that
  CUDA code expects. Do not add "prefetch" or host/device mirror logic
  unless the bridge actually needs it.
- **Count objects, not just bytes.** MLX/Metal can fail on allocator
  resource count (`MTLBuffer` count) before resident memory looks scary.
  Many small temporaries can be worse than one large buffer. Reuse-first
  beats allocator-cache optimism.
- **Batching strategy is different.** CUDA/FlashInfer already has a
  strong varlen story; Metal often needs explicit left-padding,
  additive masks, and per-row RoPE offsets to make packed decode
  correct. Same-length assumptions are not portable.
- **Prefill scalarization is fatal on Metal.** If a scheduler chunk of
  512 prompt tokens degenerates into 512 one-token graphs, you will pay
  twice: TTFT collapses and allocator churn spikes. Prefer chunk-batched
  prefill whenever the model path supports it.
- **Do not reason from "no memcpy observed" to "cheap".** Unified memory
  removes explicit copies, but it does not remove page churn, lazy graph
  materialization cost, or MLX buffer lifetime pressure.
- **CUDA-style optimization instincts still help, but at a different
  layer.** On CUDA we usually chase kernel fusion, occupancy, stream
  overlap, and memory layout. On Metal the first questions are: where is
  the lazy graph materialized, how many arrays are created per step, and
  whether the scheduler is forcing unnecessary sync points.
- **Cross-backend parity must be measured, not assumed.** Sampling,
  quantization, and batching implementations differ. After changing a
  Metal hot path, rerun the Metal baseline instead of trusting CUDA-era
  intuition or numerical equivalence by inspection.

## Build requirements

- Xcode + Command Line Tools (Apple Silicon host only).
- `cmake` for MLX (`crates/mlx-sys/build.rs` uses `cmake::Config`).
- First build downloads + compiles MLX 0.31.1 via FetchContent — slow,
  cached under `target/.../build/mlx-sys-*/out/`.

## Pointers

- `crates/mlx-sys/AGENTS.md` — the bridge layer below this.
- `docs/projects/mlx-backend-roadmap.md` — current Metal backend project,
  including the prioritized backlog and acceptance gates.
- `docs/resources/metal-dflash-params.md` — DFlash CLI parameter reference.
- `docs/resources/metal-dflash.md` — DFlash usage runbook.
- `docs/experience/errors/2026-04-09-metal-optimization-pitfalls.md` — Metal-specific
  optimization gotchas collected from earlier waves.
- `docs/experience/errors/2026-04-16-metal-varlen-rope-blocker.md` —
  retrospective on the MLX scalar-RoPE bug for `B > 1, S = 1` and why
  batched decode must always use per-row array offsets even when every row
  has the same logical position.
- `docs/experience/wins/2026-04-14-metal-dflash-qwen3.md` — reference win
  (5.9× decode on M4 Pro).
