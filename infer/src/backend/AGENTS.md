# `infer::backend` — Agent Guide

Backend abstraction layer: **cross-backend contract + per-backend dispatch**.
Load this file before editing anything under `infer/src/backend/`.

## What lives here

| File | Role |
|------|------|
| `backend.rs` | `InferenceBackend` trait + `GenerateResult` + `StreamingInferenceBackend` (the default blanket impl forwards `generate_stream` → `generate`). Single-request, `Send` (not `Sync`) — runtime owns threading. |
| `backend/cuda.rs` | **Thin re-export shim** for `infer_cuda_kernels::{ffi, flashinfer, paged_kv, prelude, tensor, ...}`. ~15 lines. Historical `crate::backend::cuda::...` paths still resolve through this. Only `bootstrap.rs` is a real module, and it only exists with `#[cfg(feature = "cuda")]`. |
| `backend/metal.rs` + `metal/` | Real Metal backend. See [`metal/AGENTS.md`](metal/AGENTS.md). |
| `backend/cpu.rs` | Dev-only synthetic backend (feature `cpu`, ~309 lines, generates fake tokens). Intentionally unextracted — zero independence benefit. |
| `backend/runtime.rs` | `BackendRuntimeHandle`: serial-runtime `RequestHandle` shared by Metal + CPU paths. CAS-loop admission for `max_waiting`; drops waiting count on channel-send failure. |

## Invariants

1. **`InferenceBackend` is `Send`, not `Sync`.** If you need parallelism,
   do it in the runtime on top — don't share one backend across threads.
2. **`backend/cuda.rs` must stay a re-export shim.** New CUDA types go in
   `crates/infer-cuda-kernels/`; new glue goes in `cuda/bootstrap.rs`.
   Adding types directly here re-creates the bootstrap straddle that
   Route-A reverted (`docs/experience/wins/2026-04-15-route-a-cuda-internal-hygiene.md`).
3. **Cross-backend code cannot mention CUDA/Metal types directly.** Use the
   trait, or an enum at the `server_engine::LoadedInferenceEngine` layer.
4. **`cfg(feature = "cuda")` and `cfg(feature = "metal")` are mutually
   compatible at type-check time** — do not write `cfg(not(feature = "metal"))`
   on CUDA paths or vice versa. Check with
   `cargo check -p infer --no-default-features --features cuda,no-cuda`.

## Where NOT to put things

- Model-specific CUDA glue → `infer/src/backend/cuda/bootstrap.rs`, not here.
- Metal kernel launches → `crates/mlx-sys/src/mlx_bridge.cpp` (the C++ side).
- Serial scheduling policy → `backend/runtime.rs`; **not** the CUDA scheduler.

## Extension pattern (adding a new backend)

1. Implement `InferenceBackend` + optionally `StreamingInferenceBackend`.
2. Add a new arm to `server_engine::LoadedInferenceEngine`.
3. Gate everything behind a new cargo feature; wire into `backend.rs` mod decls.
4. If it's single-request, reuse `BackendRuntimeHandle`; if multi-request,
   write a scheduler analogous to `scheduler/cuda/`, not a special case of it.

## Pointers

- `docs/architecture.md` — the Option-A/Option-B story, why CUDA is still
  in-tree, the trip wires for final kernel-crate extraction.
- `docs/experience/wins/2026-04-15-route-a-cuda-internal-hygiene.md` — why
  the 4-shell split was reverted; do not re-litigate.
- `crates/infer-cuda-kernels/src/prelude.rs` — the proto-API contract this
  module re-exports.
