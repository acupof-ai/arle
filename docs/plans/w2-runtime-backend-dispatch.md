# W2 PR2 — Runtime backend dispatch: design

## Goal

Replace compile-time `--features` backend selection with runtime `--backend`
dispatch. Crates above the seam (infer-core, infer-seam, infer-server,
infer-api, cli, arle) end with zero backend features. Which backend crates
are in the link stays a leaf decision.

## Approved 3-PR split

1. **PR1 (DONE, #252):** seam object-safety + de-genericize Engine.
   `PollResult` non-generic, `BackendExecutor: 'static` with boxed
   submit/poll + `name()` + `as_any_mut()`. Engine/ServeHandle/
   ServeInferenceEngine drop `<E, K>` generics.
2. **PR2 (this design):** runtime registry + `--backend` dispatch + remove
   features above the seam.
3. **PR3 (later):** OPD surface downcast + `no-cuda` cleanup.

## Architecture

### Registry (in infer-api/src/loaded.rs)

```rust
/// Engine + KV pool constructor, registered by the leaf binary.
pub type BackendBuilderFn = fn(
    model_path: &str,
    config: &EngineLoadConfig,
) -> anyhow::Result<(Box<dyn infer_seam::BackendExecutor>, Box<dyn infer_seam::KvPool>)>;

struct BackendEntry {
    name: &'static str,
    build: BackendBuilderFn,
}

static REGISTRY: std::sync::OnceLock<std::sync::RwLock<Vec<BackendEntry>>> =
    std::sync::OnceLock::new();

pub fn register_backend(name: &'static str, build: BackendBuilderFn) { ... }
pub(crate) fn lookup_backend(name: &str) -> Option<BackendBuilderFn> { ... }
```

The builder returns `(Box<dyn BackendExecutor>, Box<dyn KvPool>)` — seam
types. infer-api's `load_with_config` calls the builder, then constructs
`Engine::with_config(executor, kv, scheduler_config)` + `ServeHandle::spawn`
+ `ServeInferenceEngine::new`. All backend-neutral.

### LoadedInferenceEngine collapse

Enum → struct:

```rust
pub struct LoadedInferenceEngine {
    engine: ServeInferenceEngine,
    backend: &'static str,
}
```

`backend_name()` returns `self.backend`. OPD methods (`forward_token_logits`,
`remerge_student_lora`, `frozen_base_*_pointers`) use `with_cuda_executor`
downcast (already in place from PR1) — they stay in infer-api for now,
move to train in PR3.

### load_with_config dispatch

```rust
pub fn load_with_config(model_path: &str, config: EngineLoadConfig) -> Result<Self> {
    let backend = config.backend.clone().unwrap_or_else(detect_default_backend);
    let build = lookup_backend(&backend)
        .ok_or_else(|| anyhow::anyhow!("backend '{backend}' not registered"))?;
    // ... set sampling defaults, resolve path, load tokenizer ...
    let (executor, kv) = build(&resolved, &config)?;
    let serve = ServeHandle::spawn(executor, kv, config.scheduler_config())?;
    Ok(Self { engine: ServeInferenceEngine::new(model_id, tokenizer, serve), backend })
}
```

`--backend` flag already exists (`ServeBackendArg` in cli/args.rs). The
`Auto` variant resolves to the first registered backend.

### Where the backend-specific code goes

Two viable options:

**Option A (chosen): move to arle `src/backends/`.**
- arle adds `infer-api` + backend crates as deps.
- `src/backends/cuda.rs`: `build_cuda_engine` + `cuda_serve_handle` +
  `classify_cuda_model` + `tp_world_size` + `TpEnvGuard` (~700 lines).
- `src/backends/metal.rs`: `metal_serve_handle` +
  `metal_deepseek_ocr_serve_handle` (~350 lines).
- `src/backends/hip.rs`, `vulkan.rs`, `cpu.rs` (~100 lines each).
- `src/backends/mod.rs`: `pub fn register_all()` calling
  `infer_api::register_backend("cuda", cuda_builder)` etc.
- arle `main()`: `backends::register_all(); cli::run()`.

**Option B: new crate `infer-backends`.** Same content, separate crate.
Cleaner dependency graph but more workspace churn.

### Cargo.toml changes

| Crate | Change |
|-------|--------|
| infer-api | Remove `cuda/metal/hip/vulkan/cpu/no-cuda/nccl/deepep` features + optional backend deps. Keep `grammar` etc. |
| cli | Remove `cuda/metal/hip/vulkan/cpu/no-cuda/nccl/deepep` features. Keep non-backend features. |
| arle (root) | Add `infer-api`, `infer-cuda`, `infer-metal`, `infer-hip`, `infer-vulkan`, `cuda-kernels` as deps. Features: `cuda = ["dep:infer-cuda", "infer-cuda/cuda", "dep:cuda-kernels", "cuda-kernels/cuda"]`, `metal = ["dep:infer-metal", "infer-metal/metal"]`, etc. No forwarding to cli. |
| train | Keep `cuda` feature (train is NOT "above the seam"). OPD methods move here in PR3. |

### `no-cuda` localization

`no-cuda` becomes a `cuda-kernels`-local build.rs switch (auto-detect nvcc
absence, skip AOT). Not a workspace feature. cuda-kernels' build.rs already
handles this via the `no-cuda` feature — the change is that no other crate
forwards it.

### `router_for_backend`

Same registry dispatch. The router variant calls the builder, spawns the
serve handle, builds the axum router via `infer_server::openai_router`.
Currently has separate `router_metal`/`router_cuda`/etc. functions — these
collapse into one backend-neutral path.

### OPD surface (PR3 preview)

`forward_token_logits`, `forward_training_taps`, `remerge_student_lora`,
`frozen_base_*_pointers`, `update_dspark_markov_weights` stay on
LoadedInferenceEngine in PR2 (they use `with_cuda_executor` downcast from
PR1). In PR3 they move to `train` as extension functions, since train is
the only caller and can have a `cuda` feature.

## Execution order

1. Registry + `register_backend` in infer-api (compiles, empty registry).
2. Collapse `LoadedInferenceEngine` enum → struct. Update all methods.
3. Change `load_with_config` + `router_for_backend` to registry dispatch.
4. Create `src/backends/` in arle, move code from loaded.rs.
5. arle `main()`: `backends::register_all()`.
6. Update Cargo.toml files (infer-api, cli, arle).
7. Update callers (cli, train, examples).
8. Verify: `cargo check` all feature combos, clippy, pre-push hook.
9. Fingerprint check: one `.fingerprint` per target for seam/core/server.

## Risks

- **train examples** call `load_with_config` directly. They need
  `infer_api::register_backend(...)` before the call. Add a `register_all()`
  call in each example's `main()`.
- **`router_for_backend`** has backend-specific router arms (CUDA has
  additional routes). Verify the backend-neutral `openai_router` covers all
  cases, or move the extra routes to a post-construction hook.
- **Multiproc serve** (`serve_multiproc`) constructs engines on worker
  ranks. The registry must be populated on each rank. The worker binary
  is the same arle binary, so `register_all()` in `main()` covers it.
