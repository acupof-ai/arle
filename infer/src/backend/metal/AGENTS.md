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
4. **Metal scheduler is currently accounting-only.** It validates policy
   decisions without touching the real execution path. Wiring it in is
   tracked under `docs/plans/2026-04-15-metal-backend-acceptance-plan.md`.
   Don't assume `MetalScheduler::admit` runs on the hot path yet.
5. **Qwen3 and Qwen3.5 take different paths.** Qwen3 runs through the Rust
   `rust_transformer_layer` path in `forward.rs`. Qwen3.5 delegates to the
   dedicated C++ step model in `qwen35.rs` + `mlx-sys/src/mlx_qwen35_model.cpp`
   — don't mix them.
6. **DFlash (speculative decode) is experimental and optional.** Guarded by
   `MetalDflashOptions`; empty draft model = feature off. See
   `docs/resources/metal-dflash.md` for user-facing flags.
7. **`serial runtime`** (`backend/runtime.rs`) is what handles request
   queuing for the Metal backend today. The backend itself only exposes
   `InferenceBackend::generate` — concurrency is the runtime's problem.

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
- `docs/experience/wins/2026-04-14-metal-dflash-qwen3.md` — reference win
  (5.9× decode on M4 Pro).
