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
│   ├── attention/       — FlashInfer prefill/decode, turboquant decode, varlen FP8 split-KV (P0)
│   ├── gemm/            — gemv, Marlin W4, quantized gemv, turboquant weight gemv
│   ├── kv/              — kv_cache_to_paged, kv_quant, paged_kv_append, scatter_kv
│   ├── quant/           — weight quant kernels
│   └── misc/            — everything else
├── src/
│   ├── lib.rs           — pub module declarations, feature gating
│   ├── prelude.rs       — **the proto-API contract** (7 types; see Prelude discipline)
│   ├── ffi.rs + ffi/    — extern "C" declarations, grouped by domain (see FFI domain layout below)
│   ├── flashinfer.rs    — FlashInferWorkspace + metadata staging
│   ├── paged_kv.rs      — PagedKVPool, TokenKVPool
│   ├── tensor.rs        — DeviceContext, DeviceVec, DeviceMatrix, HiddenStates, RawDevicePtr
│   ├── collective.rs    — `CollectiveBackend` trait + `NcclBackend` skeleton (F0 multi-GPU). F7 adds CustomAR / mscclpp / quick_ar / symm_mem behind the same trait. Method set is taken from actual F1+ callers (LayerCommunicator AR, PP send/recv, MoE all-to-all via group_start/end).
│   ├── kv_quant.rs      — KV quant state/dispatch
│   ├── kv_turboquant.rs — TurboQuant-specific KV state
│   ├── kv_types.rs      — KVCacheDtype, KVFormat (always-on enum)
│   └── turboquant_state.rs — TurboQuant calibration state
└── tools/triton/        — Triton Python kernels (AOT compiled by build.rs)
```

### FFI domain layout

`src/ffi/` splits extern "C" declarations into one file per concern. Add new
declarations to the closest existing domain; do not create a domain for fewer
than ~3 functions.

| File | Domain |
|------|--------|
| `attention.rs` | FlashInfer + custom decode/prefill kernels |
| `elementwise.rs` | add/silu_mul/extract_vec/etc. batched scalars |
| `embedding.rs` | embedding_batch / embedding_decode |
| `gemm.rs` | gemv, gemm, fused_mlp, Marlin W4 |
| `kv.rs` | scatter_kv, kv_cache_to_paged, paged_kv_append |
| `mla.rs` | DeepSeek V4 MLA decode/prep (P0'', design-ready, partial wiring) |
| `misc.rs` | catch-all |
| `nccl.rs` | NCCL collective primitives consumed by `collective.rs` |
| `norm.rs` | rms_norm, fused_add_rms_norm |
| `quant.rs` | weight + activation quant kernels |
| `recurrent.rs` | gated_delta_rule prefill/decode (Qwen3.5 hybrid) |
| `sampling.rs` | argmax / argmax_with_logprob / gpu_sample |

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
  `cuda_kernels::ffi::xxx` directly.
- `EngineOptions` / runtime configs — owned by `infer::server_engine`.
- Model-specific state (`Qwen3Model`, etc.) — application types, stay in `infer::model::*`.
- `CollectiveBackend` / `NcclBackend` — multi-GPU collective trait lives at
  `cuda_kernels::collective::*`. It will graduate to the prelude only once
  more than two callers exist outside the F0–F2 distributed scaffold.

Removing a symbol is **encouraged** if it stops meeting the three criteria.

## `build.rs` rules

- **SM auto-detection order:** `TORCH_CUDA_ARCH_LIST` → `CMAKE_CUDA_ARCHITECTURES`
  → `nvidia-smi --query-gpu=compute_cap` → T1 default set `{80, 86, 89, 90}`.
  Always emit a `cargo:warning` on the T1-default fallback.
- **Tier policy** (canonical: [`docs/plans/sm-coverage.md`](../../docs/plans/sm-coverage.md)):
  T1 `{80, 86, 89, 90}` default-built; T2 `{100, 120}` opt-in via env var;
  T3 `< 80` panics at build time. Adding a new SM = update `T1_SMS`/`T2_SMS`
  in `build.rs` and the GPU/SM row in `docs/support-matrix.md`.
- **AOT failure policy:** any (SM, kernel) combination that fails to emit
  cubin → `panic!`. No warn-skip. Error message must suggest a
  `TORCH_CUDA_ARCH_LIST=...` value that excludes the failing SM.
- **Multi-cubin AOT + runtime SM dispatch.** Both Triton AOT
  (`build_triton_kernel`) and TileLang AOT (`build_tilelang_kernel`) loop
  over `sm_targets` and emit one cubin per (kernel, SM) tuple. Per-SM
  symbol uniqueness comes from a `_sm{sm}` suffix on `out_name` (Triton:
  the public C symbol the generator writes) and on **both** `kernel_name`
  and `out_name` for TileLang (TileLang's gen script appends `_cuda` to
  `kernel_name`, so varying it gives a unique exported symbol per SM).
  A C dispatch wrapper (`format_dispatch_wrapper`) extern-declares every
  per-SM symbol, caches `compute_capability_major*10 + minor` in a
  `static __thread` slot via `cuCtxGetDevice` + `cuDeviceGetAttribute`,
  and `switch`es to the matching cubin. Public FFI entry names (e.g.
  `silu_mul_triton_aot_cuda`, `tilelang_*_run_cuda`) are unchanged from
  the single-SM era — only the wrapper internals dispatch. **TLS, not
  pthread_once + global static**: multi-GPU runtimes may bind different
  threads to different devices, and a process-global cache would race
  + silently dispatch the wrong cubin.
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

## Active priorities touching this crate

- **P0 long-context.** Varlen FP8 split-KV decode kernels live in
  `csrc/attention/`; FFI surface is `ffi/attention.rs`. Phase 2 spec-decode
  K+1 packed verifier work also lands here once the design closes.
- **P0' multi-GPU F0–F4.** `collective.rs` + `ffi/nccl.rs` are the
  multi-GPU primitive surface. F2 production NCCL forward collectives
  block both P0' (TP=2 throughput bench) and P0'' (DeepSeek V4 DS5
  collectives in forward).
- **P0'' DeepSeek V4.** `ffi/mla.rs` carries the planned MLA decode/prep
  declarations. The kernel implementation under `csrc/attention/` is
  design-ready
  ([`docs/plans/2026-05-01-mla-kernel-design.md`](../../docs/plans/2026-05-01-mla-kernel-design.md))
  but waits on DS2 block-FP8 layout and DS3 cache-format wiring before
  shipping.

## Pointers

- `src/prelude.rs` — the full discipline rule, in-code comments.
- `docs/architecture.md` §Future Evolution — Option A → Option B story.
- `docs/plans/cuda-kernel-crate-extraction.md` — full extraction blueprint.
- `docs/plans/2026-04-28-single-node-multi-gpu.md` — `CollectiveBackend`
  method-set rationale + F0–F4 scaffold roadmap.
- `docs/plans/2026-05-01-mla-kernel-design.md` — MLA kernel family
  layout (P0'' future).
- `docs/reviews/2026-04-14-cuda-kernel-six-principles-review.md` — kernel
  optimization heat map.
- `docs/experience/wins/2026-04-15-route-a-cuda-internal-hygiene.md` —
  what the ffi split + prelude landed, and why.
