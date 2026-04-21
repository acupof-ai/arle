# `cuda-kernels` — Agent Guide

Extracted CUDA kernel crate: CUDA C kernels + Triton AOT + FFI + the seven
tensor/pool/metadata types that `infer` proper consumes. **This is the
proto-public API for the eventual Option-B split.** Load this file before
touching anything under `crates/cuda-kernels/`.

## Refactor posture

- Keep kernel-crate code simple and uniform. Prefer deletion-style refactors:
  remove stale shims, collapse duplicate FFI/kernel entry paths, and keep one
  canonical ownership boundary between `infer` and `cuda-kernels`.

## Why this crate exists

See `docs/architecture.md` + `docs/plans/cuda-kernel-crate-extraction.md`.
Short version: the 2026-04-15 Route-A revert turned the old four-shell
split into one kernel crate. `infer/src/backend/cuda.rs` is now a ~15-line
`pub use` shim over this crate, so the 60+ existing `crate::backend::cuda::…`
call sites still resolve while we wait for the final extraction trigger
(FA-3 H100, MLA, NCCL, FP8 GEMM, spec-decode GPU, or a second consumer).

**Invariant:** the dependency edge is `infer → cuda-kernels`, **never
the reverse**. Nothing in this crate may depend on `infer` — no tokenizer,
no scheduler, no model-specific weight struct, no `EngineOptions`.

## Crate layout

```
crates/cuda-kernels/
├── Cargo.toml           — features: `cuda` (enables cudarc), `no-cuda` (compile-without-nvcc)
├── build.rs             — SM auto-detection, Triton AOT, CUDA C compile, FlashInfer link
├── csrc/                — CUDA C sources, grouped by concern
│   ├── common.cuh       — shared header (include with `#include "common.cuh"`)
│   ├── attention/       — FlashInfer prefill/decode, Triton decode, turboquant decode
│   ├── gemm/            — gemv, Marlin W4, quantized gemv, turboquant weight gemv
│   ├── kv/              — kv_cache_to_paged, kv_quant, paged_kv_append, scatter_kv
│   ├── quant/           — weight quant kernels
│   └── misc/            — everything else
├── src/
│   ├── lib.rs           — pub module declarations, feature gating
│   ├── prelude.rs       — **the proto-API contract** (7 types; see Prelude discipline)
│   ├── ffi.rs + ffi/    — extern "C" declarations, grouped by domain
│   ├── flashinfer.rs    — FlashInferWorkspace + metadata staging
│   ├── paged_kv.rs      — PagedKVPool, TokenKVPool
│   ├── tensor.rs        — DeviceContext, DeviceVec, DeviceMatrix, HiddenStates, RawDevicePtr
│   ├── graph_pool.rs    — CUDA graph capture/replay pool
│   ├── kv_quant.rs      — KV quant state/dispatch
│   ├── kv_turboquant.rs — TurboQuant-specific KV state
│   ├── kv_types.rs      — KVCacheDtype, KVFormat (always-on enum)
│   └── turboquant_state.rs — TurboQuant calibration state
└── tools/triton/        — Triton Python kernels (AOT compiled by build.rs)
```

## Prelude discipline (enforce strictly — this is the public surface)

`src/prelude.rs` currently exports exactly 7 symbols:

```rust
FlashInferDecodeMetadata
PagedKVPool
DeviceContext
DeviceMatrix
DeviceVec
HiddenStates
RawDevicePtr
```

**Adding a symbol requires three justifications in writing on the PR:**

1. **Consumed by ≥3 files outside `backend/cuda/`.** Two-file helpers stay
   on direct module paths. Example: `TokenKVPool` has exactly 3 callers
   and **does not** belong in the prelude — it lives at
   `infer_cuda_kernels::TokenKVPool` (re-exported at crate root).
2. **Stable.** Name, layout, and method signatures will not change in the
   next 6 months. Internal types in active design must not be in the prelude.
3. **Removing it would not break the kernel-crate extraction plan.** If
   exporting a symbol forces some currently-private `infer` type to become
   `pub` cross-crate, the symbol does not belong here — it belongs in
   `infer` proper.

**What the prelude deliberately does NOT contain:**

- Anything from `ffi::*` — consumers that need `extern "C"` symbols use
  `infer_cuda_kernels::ffi::xxx` directly.
- `EngineOptions` / runtime configs — owned by `infer::server_engine`.
- Model-specific state (`Qwen3Model`, etc.) — application types, stay in `infer::model::*`.

Removing a symbol is **encouraged** if it stops meeting the three criteria.

## `build.rs` rules

- **SM auto-detection order:** `INFER_CUDA_SM` → `CUDA_SM` → `nvidia-smi`
  → fallback `sm_80` (A100). Always emit a `cargo:warning` on fallback.
- **Triton AOT** is driven by `find_triton_python()` — order: `INFER_TRITON_PYTHON`
  → `tools/triton/.venv/bin/python` → `./.venv/bin/python` → `python3` → `python`.
  Generated artifacts land under `OUT_DIR/triton/...`.
- **Recursive `.cu` walk under `csrc/`.** nvcc is invoked with `-I csrc/` so
  `#include "common.cuh"` works from any subdir. Don't hand-list files — the
  walk is the rerun-if-changed contract.
- **`no-cuda` feature** means `build.rs` skips nvcc entirely, and every
  cudarc-using module is gated. This is what makes
  `cargo check --features cuda,no-cuda` work on a Mac. Never add unconditional
  cudarc imports.
- **FlashInfer link** comes from the vendored tree; don't add system `-lflashinfer`.

## csrc conventions

- All CUDA C files end in `.cu`; headers in `.cuh`. One canonical header
  (`common.cuh`) at `csrc/common.cuh`, included by every subdir.
- Group new kernels by the closest existing subdir (`attention/`, `gemm/`,
  `kv/`, `quant/`, `misc/`). Don't create a new subdir for fewer than 3 files.
- `csrc/attention/flashinfer_prefill_paged.cu` is the HD128 paged-prefill
  FlashInfer wrapper (`PrefillPlan` + `BatchPrefillWithPagedKVCacheDispatched`).
- `csrc/attention/flashinfer_prefill_paged_hd256.cu` is the HD256 paged-prefill
  FlashInfer wrapper used for Qwen3.5 full-attention parity.
- `csrc/attention/prefill_attention_paged_prep.cu` holds the paged-only
  prefill prep kernels that do QK norm + RoPE and write K/V directly into HND pages.
- When optimizing, check the heat map in
  `docs/reviews/2026-04-14-cuda-kernel-six-principles-review.md` first.

## `no-cuda` gotchas

With `--features cuda,no-cuda`:

- `lib.rs` still declares every `#[cfg(feature = "cuda")]` module so rustc
  type-checks them, but `build.rs` skips nvcc. This is **not** a release
  configuration — ops will fail at runtime. It's only for refactor validation.
- Code that uses `cudarc::driver::*` types is fine; linking will fail if
  you actually try to build a binary, but `cargo check` is happy.

## Pointers

- `src/prelude.rs` — the full discipline rule, in-code comments.
- `docs/architecture.md` §Future Evolution — Option A → Option B story.
- `docs/plans/cuda-kernel-crate-extraction.md` — full extraction blueprint.
- `docs/reviews/2026-04-14-cuda-kernel-six-principles-review.md` — kernel
  optimization heat map.
- `docs/experience/wins/2026-04-15-route-a-cuda-internal-hygiene.md` —
  what the ffi split + prelude landed, and why.
