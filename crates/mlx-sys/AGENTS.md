# `mlx-sys` — Agent Guide

**Single source of truth** for the Metal bridge. Builds MLX from source,
compiles the C++ bridge, exposes `extern "C"` FFI consumed by `infer-metal`.
Load this file before touching the Metal path from either side.

## Refactor posture

- Keep the bridge simple and uniform. Prefer deletion-style refactors:
  remove redundant wrappers, collapse duplicate Rust/C++ glue paths, and keep
  one canonical bridge contract instead of stacked compatibility layers.

## What lives here

```
crates/mlx-sys/
├── Cargo.toml           — build-deps: cmake, cc
├── build.rs             — vendored MLX cmake build → C++ bridge cc build → link chain
├── vendor/              — pinned MLX / metal-cpp / fmt / json / gguflib source snapshots
└── src/
    ├── lib.rs           — extern "C" declarations (no mlx-c intermediate)
    ├── mlx_bridge.cpp   — C++ wrappers for mlx::core API
    ├── mlx_qwen35_model.cpp     — dedicated C++ Qwen3.5 step model (per-layer hot path)
    ├── mlx_qwen35_moe_block.cpp — Qwen3.5 / Qwen3.6 SparseMoeBlock forward composed in C++; wired from `mlx_qwen35_model.cpp`
    ├── mlx_dflash_draft_model.cpp — Metal DFlash draft-model step (the C++ side of the speculative-draft path consumed by the `dflash` module of `infer-metal`)
    ├── mlx_metal_capture.mm     — env-gated `MTLCaptureManager` hook around `qwen35_compiled_step_session` (default OFF; see Debugging hooks)
    └── mlx_common.h             — shared C header (dtype constants, struct layouts)
```

All eight `.cpp`/`.mm` translation units (7 C++ + 1 Objective-C++) are
explicitly listed in `build.rs` (`cc::Build::new().file(...)`) and
registered with
`cargo:rerun-if-changed`. Adding a new C++ file requires updating both
lists — there is no glob.

## Invariants (violating these breaks the Metal path)

1. **No mlx-c shim.** `mlx_array` is an **opaque pointer to `mlx::core::array*`**,
   reinterpret_cast in the bridge. Do not add a wrapper struct. Do not
   import `mlx-c` crates.
2. **No `.metal` shader files in this repo.** Metal kernels live inside MLX
   itself (fetched by CMake). Any "new Metal kernel" is either an MLX PR
   or a C++ change in `mlx_bridge.cpp` that composes existing MLX ops.
3. **Dtype constants must match `mlx::core::Dtype::Val` and `mlx_common.h`.**
   If you add a dtype, update all three sites (`lib.rs`, `mlx_common.h`, and
   the bridge). The CI / type-check on Linux will not catch a drift — it's
   Apple-only.
4. **`mlx_last_error()` is thread-local.** Every C++→C boundary that can
   throw must catch and set it. Rust callers must check for null return
   and read `mlx_last_error()` immediately afterwards.
5. **Single source of truth for the Metal bridge.** Only Metal-facing runtime
   code should consume this crate directly: `infer-metal` and `autograd`'s
   Metal backend. Nothing else (no scheduler, no model registry,
   no generic train logic) should link `mlx-sys` directly. If you find
   yourself wiring mlx-sys into a non-Metal module, you're recreating the
   bridge. Callers that serialize MLX access must use `mlx_sys::mlx_guard()`
   so process-global MLX state has one Rust synchronization boundary.
6. **The Qwen3.5 step model is a separate C++ file** (`mlx_qwen35_model.cpp`),
   not a generic MLX composition. It exists because Qwen3.5 hybrid attention
   benefits from a fused C++ step path — keep this dedicated, don't fold it
   into the generic Rust `rust_transformer_layer` fallback without a bench
   snapshot.
7. **Specialized C++ helpers for Qwen3.5 sub-layers compose the C++ side
   of the bridge.** `mlx_qwen35_moe_block.cpp` is the canonical SparseMoE
   forward; `mlx_dflash_draft_model.cpp` is the canonical draft-model
   step. Both are reachable from Rust via dedicated FFI entry points and
   are also called from `mlx_qwen35_model.cpp`'s per-layer dispatch.
   Adding a new fused sub-layer goes here, not into `mlx_bridge.cpp`.

## Build chain (`build.rs`)

1. **cmake** builds MLX directly from `vendor/mlx`, with every
   `FetchContent` dependency overridden to a pinned local source tree under
   `vendor/` and the build running with `FETCHCONTENT_FULLY_DISCONNECTED=ON`.
   Flags: `MLX_BUILD_METAL=ON`, `MLX_BUILD_ACCELERATE=ON`, tests/examples/
   benchmarks/python OFF, `BUILD_SHARED_LIBS=OFF`, `CMAKE_CXX_STANDARD=17`.
2. **cc** compiles all bridge translation units (`mlx_bridge.cpp`,
   `mlx_qwen35_model.cpp`, `mlx_qwen35_moe_block.cpp`,
   `mlx_dflash_draft_model.cpp`, `mlx_metal_capture.mm`) as
   `libmlx_ffi.a` with `-std=c++17 -Wno-deprecated-copy
   -Wno-unused-parameter -Wno-sign-compare`.
3. **Link order (strict):**
   - `static=mlx_ffi` (our bridge)
   - `static=mlx` (the fetched library)
   - macOS frameworks: `Metal`, `Foundation`, `Accelerate`, `MetalPerformanceShaders`
   - `c++` (C++ stdlib)
4. `cargo:rerun-if-changed` covers every bridge translation unit (8 files) + `mlx/CMakeLists.txt`.
   Touching MLX headers transitively does not trigger rebuild — if you
   edit an MLX header in a fork, also bump `mlx/CMakeLists.txt`.

## First-build cost

Fetching + compiling MLX from source takes 5–15 minutes on an M-series Mac.
Cached under `target/.../build/mlx-sys-*/out/build/_deps/mlx-src/`. A
`cargo clean -p mlx-sys` is expensive — avoid unless the MLX version bumps.

## FFI patterns (when adding bridge functions)

- **Every function returning `*mut mlx_array` must set `mlx_last_error()`
  and return `nullptr` on exception.** The Rust wrapper in
  `crates/infer-metal/src/mlx.rs` relies on this contract.
- **`mlx_array_clone` bumps the shared_ptr refcount**; `mlx_array_free`
  decrements it. Always pair them. Rust wrappers already do this — don't
  double-free when writing new bridge functions.
- **Shape/dtype data crosses the boundary as `*const i32` + `i32 ndim`**,
  never `std::vector`. Don't introduce `std::string` or STL containers in
  the public bridge API.

## Common mistakes

- Importing `mlx_sys::*` from a scheduler or model module. **Wrong.**
  All MLX types go through the thin wrapper in `infer-metal`
  (`crates/infer-metal/src/mlx.rs`).
- Adding a second C++ model file without wiring `build.rs`. `cc::Build::new()`
  must explicitly `.file(...)` each `.cpp`; there's no glob.
- Forgetting to add new frameworks to the link line. Rare — MLX's own
  dependencies cover most things — but a new MPS call may require more.

## Debugging hooks

### Qwen3.5 GPU trace capture (`mlx_metal_capture.mm`)

Env-gated `MTLCaptureManager` hook around `qwen35_compiled_step_session`.
Default OFF — the only hot-path cost when disabled is one relaxed atomic load
inside `maybe_capture_qwen35_step_begin`.

```bash
MTL_CAPTURE_ENABLED=1 \
INFER_CAPTURE_STEP=5 \
  ./target/release/arle serve --backend metal --model-path <path>
```

- `INFER_CAPTURE_STEP=N` — **0-indexed count of `qwen35_compiled_step_session`
  calls since process start**, across every request and caller. The counter is
  process-global and NOT reset between requests. With `arle serve`, each
  request's prefill + decode advances it, so compute N from the requests
  issued before the one to capture. Unset = disabled.
- `INFER_CAPTURE_PATH=…` — optional override; default
  `/tmp/qwen35_step_<unix_ts>.gputrace`.
- The hook issues `eval(outputs)` **before** swapping session state so an
  eval failure cleanly rolls back — the caller sees `-1` with no partial
  cache advance and no leaked output handle.

Open the resulting `.gputrace` in Xcode for inspection.

## Active priority — P3 Metal serving-grade closure

This crate is the bridge layer beneath P3. The current Qwen3.5 step
model (Rust path 305.5 tok/s on M4 Pro for `1024/256`) and the DFlash
draft path (5.9× decode reference win) both depend on the dedicated
C++ files staying separate from `mlx_bridge.cpp`. New Metal-only fused
ops should land here, not as Rust compositions in `infer-metal`.

## State-mutating change — enumerate every buffer (事无巨细)

Any change mutating MLX or
bridge-cache state (DFlash draft KV, `BatchKVCache`, per-slot scratch, a rollback):
**enumerate EVERY buffer it writes, prove each reverted / self-heals (with the
exact precondition) / snapshotted** — never assume self-heal. Pre-allocate once
and reuse (no per-step alloc on the encode path — `mx::async_eval` encodes on the
caller thread, so a buffer's lifetime spans the async eval; a freed-too-early
input corrupts silently). Save/restore at minimum granularity (a ring touched at
one slot → that one slot). Keep it inside the opt-in path so the baseline step is
byte-for-byte untouched, and gate correctness on **correct inference** (needle +
same-config-twice floor), not byte-identity to a reference run.

## Distilled lessons (recurring ≥2 entries)

- **mlx.metallib must be colocated with the binary on every macOS distribution path.** build.rs,
  package script, install.sh, brew formula — all must ship `mlx.metallib` next to the binary
  or runtime fails to load the Metal shaders
  (`feedback_mlx_metallib_must_be_colocated.md`).
- **`mx::compile` needs cache-as-input for position-dependent graphs.** `item()` bakes scalars
  into the compiled graph; runtime `cache_pos` must enter as an active-prefix input tensor,
  not as `(cache, position)` separately — else compile re-runs every step
  (`feedback_mx_compile_cache_as_input.md`).
- **`mx::async_eval` encodes on the *caller* thread.** No worker thread will steal the encode
  work — falsified multi-stream encode pipelining (5–13% Qwen3.6 regression). Don't propose
  encode-side pipelining (`feedback_mlx_async_eval_is_caller_thread.md`).
- **Adding a second C++ translation unit requires `cc::Build::new().file(...)` in `build.rs`
  AND `cargo:rerun-if-changed`.** There is no glob — silently-missing files compile-skip then
  link-error opaquely.
- **Use `mlx_sys::mlx_guard()` for any cross-crate MLX serialization.** MLX state is
  process-global; autograd's Metal backend and `infer-metal` must share one Rust
  synchronization boundary, not local mutexes.

## Pointers

- `crates/infer-metal/src/mlx.rs` — the thin wrapper that turns this
  FFI into safe-ish Rust. `crates/infer-metal/` is the Rust consumer side
  (no consumer-side AGENTS.md).
  FFI into safe-ish Rust.
