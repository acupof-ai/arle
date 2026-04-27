# `mlx-sys` — Agent Guide

**Single source of truth** for the Metal bridge. Builds MLX from source,
compiles the C++ bridge, exposes `extern "C"` FFI consumed by
`infer::backend::metal`. Load this file before touching the Metal path from
either side.

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
    ├── mlx_qwen35_model.cpp — dedicated C++ Qwen3.5 step model
    └── mlx_common.h     — shared C header (dtype constants, struct layouts)
```

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
   code should consume this crate directly: `infer::backend::metal` and
   `autograd`'s Metal backend. Nothing else (no scheduler, no model registry,
   no generic train logic) should link `mlx-sys` directly. If you find
   yourself wiring mlx-sys into a non-Metal module, you're recreating the
   bridge. Callers that serialize MLX access must use `mlx_sys::mlx_guard()`
   so process-global MLX state has one Rust synchronization boundary.
6. **The Qwen3.5 step model is a separate C++ file** (`mlx_qwen35_model.cpp`),
   not a generic MLX composition. It exists because Qwen3.5 hybrid attention
   benefits from a fused C++ step path; Qwen3 still goes through the Rust
   `rust_transformer_layer`. Don't merge the two without a bench snapshot.

## Build chain (`build.rs`)

1. **cmake** builds MLX directly from `vendor/mlx`, with every
   `FetchContent` dependency overridden to a pinned local source tree under
   `vendor/` and the build running with `FETCHCONTENT_FULLY_DISCONNECTED=ON`.
   Flags: `MLX_BUILD_METAL=ON`, `MLX_BUILD_ACCELERATE=ON`, tests/examples/
   benchmarks/python OFF, `BUILD_SHARED_LIBS=OFF`, `CMAKE_CXX_STANDARD=17`.
2. **cc** compiles `mlx_bridge.cpp` + `mlx_qwen35_model.cpp` as `libmlx_ffi.a`
   with `-std=c++17 -Wno-deprecated-copy -Wno-unused-parameter -Wno-sign-compare`.
3. **Link order (strict):**
   - `static=mlx_ffi` (our bridge)
   - `static=mlx` (the fetched library)
   - macOS frameworks: `Metal`, `Foundation`, `Accelerate`, `MetalPerformanceShaders`
   - `c++` (C++ stdlib)
4. `cargo:rerun-if-changed` covers the three bridge files + `mlx/CMakeLists.txt`.
   Touching MLX headers transitively does not trigger rebuild — if you
   edit an MLX header in a fork, also bump `mlx/CMakeLists.txt`.

## First-build cost

Fetching + compiling MLX from source takes 5–15 minutes on an M-series Mac.
Cached under `target/.../build/mlx-sys-*/out/build/_deps/mlx-src/`. A
`cargo clean -p mlx-sys` is expensive — avoid unless the MLX version bumps.

## FFI patterns (when adding bridge functions)

- **Every function returning `*mut mlx_array` must set `mlx_last_error()`
  and return `nullptr` on exception.** The Rust wrapper in
  `infer/src/backend/metal/mlx.rs` relies on this contract.
- **`mlx_array_clone` bumps the shared_ptr refcount**; `mlx_array_free`
  decrements it. Always pair them. Rust wrappers already do this — don't
  double-free when writing new bridge functions.
- **Shape/dtype data crosses the boundary as `*const i32` + `i32 ndim`**,
  never `std::vector`. Don't introduce `std::string` or STL containers in
  the public bridge API.

## Common mistakes

- Importing `mlx_sys::*` from `infer::scheduler` or `infer::model`. **Wrong.**
  All MLX types are behind `infer::backend::metal::mlx::*` (the thin wrapper).
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
  ./target/release/metal_bench --model <path> --use-step-driver \
      --prompt-tokens 32 --generation-tokens 10 --warmup 3 --runs 1
```

- `INFER_CAPTURE_STEP=N` — **0-indexed count of `qwen35_compiled_step_session`
  calls since process start**, across all warmup runs, timed runs, and any
  other callers. The counter is process-global and NOT reset between runs.
  For `metal_bench --warmup W --runs R --generation-tokens G --use-step-driver`,
  the first post-warmup decode step is `W × G` (e.g. `--warmup 3
  --generation-tokens 10` → use `INFER_CAPTURE_STEP=30` to capture the 1st
  timed-run decode; `=31` for the 2nd; etc.). Unset = disabled.
- `INFER_CAPTURE_PATH=…` — optional override; default
  `/tmp/qwen35_step_<unix_ts>.gputrace`.
- The hook issues `eval(outputs)` **before** swapping session state so an
  eval failure cleanly rolls back — the caller sees `-1` with no partial
  cache advance and no leaked output handle.

Open the resulting `.gputrace` in Xcode for inspection.

## Pointers

- `infer/src/backend/metal/AGENTS.md` — the Rust consumer side.
- `infer/src/backend/metal/mlx.rs` — the thin wrapper that turns this
  FFI into safe-ish Rust.
- `docs/projects/mlx-backend-roadmap.md` — current Metal backend project,
  including continuous-batching / batched-decode milestones.
