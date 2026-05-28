# `cuda-kernels` — Agent Guide

Extracted CUDA kernel crate: CUDA C kernels + TileLang AOT + FFI + the seven
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
├── build.rs             — SM auto-detection, TileLang AOT, CUDA C compile
├── csrc/                — CUDA C sources, grouped by concern
│   ├── common.cuh       — shared header (include with `#include "common.cuh"`)
│   ├── attention/       — TileLang prep/dispatch helpers, turboquant decode, varlen FP8 split-KV (P0)
│   ├── gemm/            — gemv, Marlin W4, quantized gemv, turboquant weight gemv
│   ├── kv/              — kv_cache_to_paged, kv_quant, paged_kv_append, scatter_kv
│   ├── quant/           — weight quant kernels
│   └── misc/            — everything else
├── src/
│   ├── lib.rs           — pub module declarations, feature gating
│   ├── prelude.rs       — **the proto-API contract** (7 types; see Prelude discipline)
│   ├── ffi.rs + ffi/    — extern "C" declarations, grouped by domain (see FFI domain layout below)
│   ├── tilelang.rs      — TileLang metadata staging
│   ├── paged_kv.rs      — PagedKVPool, TokenKVPool
│   ├── tensor.rs        — DeviceContext, DeviceVec, DeviceMatrix, HiddenStates, RawDevicePtr
│   ├── collective.rs    — `CollectiveBackend` trait + `NcclBackend` skeleton (F0 multi-GPU). F7 adds CustomAR / mscclpp / quick_ar / symm_mem behind the same trait. Method set is taken from actual F1+ callers (LayerCommunicator AR, PP send/recv, MoE all-to-all via group_start/end).
│   ├── kv_quant.rs      — KV quant state/dispatch
│   ├── kv_turboquant.rs — TurboQuant-specific KV state
│   ├── kv_types.rs      — KVCacheDtype, KVFormat (always-on enum)
│   └── turboquant_state.rs — TurboQuant calibration state
└── tools/tilelang/      — TileLang Python kernels (AOT compiled by build.rs)
```

### FFI domain layout

`src/ffi/` splits extern "C" declarations into one file per concern. Add new
declarations to the closest existing domain; do not create a domain for fewer
than ~3 functions.

| File | Domain |
|------|--------|
| `attention.rs` | TileLang + custom decode/prefill kernels |
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
TileLangDecodeMetadata
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
- Model-specific state (`Qwen35Model`, etc.) — application types, stay in `infer::model::*`.
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
  legacy Volta `{70}` is SM-pinned for the V100 Qwen3.5 BF16 attention + GDR
  lane; other T3 `< 80` SMs panic at build time. Adding a new SM = update the
  tier lists in `build.rs` and the GPU/SM row in `docs/support-matrix.md`.
- **AOT failure policy:** any (SM, kernel) combination that fails to emit
  cubin → `panic!`. No warn-skip. Error message must suggest a
  `TORCH_CUDA_ARCH_LIST=...` value that excludes the failing SM.
- **Multi-cubin AOT + runtime SM dispatch.** TileLang AOT
  (`build_tilelang_kernel`) loops over `sm_targets` and emits one cubin per
  (kernel, SM) tuple. Per-SM symbol uniqueness comes from a `_sm{sm}` suffix
  on **both** `kernel_name` and `out_name` (TileLang's gen script appends
  `_cuda` to `kernel_name`, so varying it gives a unique exported symbol per SM).
  A C dispatch wrapper (`format_dispatch_wrapper`) extern-declares every
  per-SM symbol, caches `compute_capability_major*10 + minor` in a
  `static __thread` slot via `cuCtxGetDevice` + `cuDeviceGetAttribute`,
  and `switch`es to the matching cubin. Public TileLang FFI entry names
  (`tilelang_*_run_cuda`) remain stable — only the wrapper internals dispatch. **TLS, not
  pthread_once + global static**: multi-GPU runtimes may bind different
  threads to different devices, and a process-global cache would race
  + silently dispatch the wrong cubin.
- **TileLang AOT** is driven by `find_tilelang_python()` — order:
  `INFER_TILELANG_PYTHON` -> `tools/tilelang/.venv/bin/python` ->
  `./.venv/bin/python` -> `python3` -> `python`.
  Generated artifacts land under `OUT_DIR/tilelang_aot/...`.
- **Recursive `.cu` walk under `csrc/`.** nvcc is invoked with `-I csrc/` so
  `#include "common.cuh"` works from any subdir. Don't hand-list files — the
  walk is the rerun-if-changed contract.
- **`no-cuda` feature** means `build.rs` skips nvcc entirely, and every
  cudarc-using module is gated. This is what makes
  `cargo check --features cuda,no-cuda` work on a Mac. Never add unconditional
  cudarc imports.
- **No legacy external-kernel link path.** New attention/GDR work goes through
  TileLang AOT or native CUDA C.

## csrc conventions

- All CUDA C files end in `.cu`; headers in `.cuh`. One canonical header
  (`common.cuh`) at `csrc/common.cuh`, included by every subdir.
- Group new kernels by the closest existing subdir (`attention/`, `gemm/`,
  `kv/`, `quant/`, `misc/`). Don't create a new subdir for fewer than 3 files.
- TileLang paged prefill/decode kernels are generated from
  `tools/tilelang/` and linked by `build.rs`.
- Historical external attention wrappers are removed; do not recreate them.
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
- **P0'' DeepSeek V4.** `ffi/mla.rs` carries the legacy `mla_decode_paged_bf16`
  ABI scaffold. New MLA attention should use TileLang AOT, cute-DSL, or a
  hand CUDA kernel; do not reintroduce external attention wrappers. The DSV4
  small-substrate SKUs in
  [`docs/plans/2026-05-05-deepseek-v4-small-substrate.md`](../../docs/plans/2026-05-05-deepseek-v4-small-substrate.md)
  §6.1.1 use smaller dims and need a different kernel (cute-DSL or hand-port);
  tracked as future work in that plan.

## Distilled lessons (recurring ≥2 entries)

- **Micro-kernel knobs (launch-shape sweep, vector-load, scale/row-pointer hoist, persistent
  kernel) are sub-noise on tiny operators.** ~25 P3 DSv4 micro-opt kills (`errors/2026-05-16-p3-*`)
  share the same lesson: a 0.5–2% kernel-level delta on a < 50 µs kernel doesn't survive the
  worst-shape gate. License-or-kill via the kernel's wall-clock fraction first
  (`errors/2026-05-28-dsv4-a4-multi-stream-overlap-kill.md`).
- **Don't mechanically transfer optimizations across KV-quant formats.** INT8→FP8 stage prefetch,
  pair-fusion, scale-broadcast all kill on same kernel shape because the bottleneck differs.
  Require a Qwen3.5-shaped component A/B with tight separated intervals
  (`errors/2026-05-12-fp8-kv-decode-shared-prefetch-kill.md`,
  `errors/2026-05-12-fp8-kv-hnd-refill-scale-broadcast-kill.md`,
  `errors/2026-05-12-fp8-kv-pair-quantize-fusion-no-license.md`,
  `errors/2026-05-12-fp8-kv-quantize-thread-grouping-kill.md`).
- **Memory-capacity math is not a kernel license.** FP4's 2× smaller KV bytes does NOT auto-beat FP8
  on sm_89 — unpack/dequant overhead can dominate. Require full-attention Phase 0 wall-clock
  evidence, not operator scan results, before runtime integration
  (`errors/2026-05-12-kv-w4a8-fp4-sm89-scan-kill.md`).
- **Math-intrinsic swaps (e.g. `expf` → fast variant) on small elementwise kernels are noise.** Use
  ncu/SASS first to confirm the intrinsic is actually the bottleneck; if it is, prefer a structural
  rewrite (polynomial sigmoid, fused split-gate path) over scalar swaps
  (`errors/2026-05-12-silu-mul-fast-exp-kill.md`).
- **Quantized loaders need a 3-stage parity gate before claiming load works.**
  (1) tensor-local dequant slice vs reference at relerr ≤ 1e-3,
  (2) layer-local matmul parity,
  (3) full-model logits parity. "Model loads and decodes one token" is only a smoke gate
  (`errors/2026-05-21-arle-infer-awq-zero-point-relerr-kill.md`).
- **`cargo:rerun-if-changed=<dir>` is NOT recursive.** Subdirectory edits ship stale cubins
  silently; the build.rs walk must recurse and emit per-file directives
  (`feedback_cargo_rerun_dir_not_recursive.md`).
- **DeepEP kernel API has inverted `num_tokens` / `num_recv_tokens` naming.** `combine`'s
  `num_tokens` is the *input* size, `num_recv_tokens` is the *output* — derived from
  `send_head.size(0)` (= original dispatch source count). Mirror Python wrapper call sites
  exactly or hit silent deadlocks (`feedback_deepep_kernel_api_inverted_naming.md`,
  `wins/2026-05-26-dsv4-deepep-cpp-full-dispatch.md`).
- **DeepEP `combine` `channel_prefix_matrix` parameter is the dispatch *output* exclusive prefix,
  NOT the `notify_dispatch` inclusive prefix.** Use `recv_channel_prefix_matrix`, not the
  same-named earlier one — different tensor, silent kernel deadlock
  (`feedback_deepep_combine_uses_recv_channel_prefix.md`).
- **NVCC parser errors in TileLang AOT compile are usually CUDA-version × tilelang-version
  drift, not `-std=c++17` workarounds.** On CUDA 12.2, pin `tilelang>=0.1,<0.1.10` (newer
  ships cute-cutlass C++20 requirements nvcc 12.2 rejects)
  (`errors/2026-05-27-tilelang-0110-cuda122-cutlass-incompat.md`).
- **Per-warp NaN patterns from a TileLang kernel → inspect generated CUDA.** AOT output sits
  under `target/release/build/cuda-kernels-*/out/tilelang_aot/<kernel>/device_kernel.cu`;
  mma-fragment layout bugs aren't visible from the Python kernel definition
  (`errors/2026-05-27-tilelang-0110-fullrow-warp23-nan-sm80.md`).
- **For pool-lifecycle APIs, returned "freed/reclaimed" vectors must reflect actual allocator
  transitions, not refcount transitions.** Refcount-zero isn't enough when a page can still
  be attached to a live slot (`errors/2026-05-12-paged-kv-release-attached-return.md`).
- **GAP-style top-k tournaments must `__syncthreads()` after broadcast reads and bake the
  tie-break sentinel into both per-thread merge AND every warp-tournament round.** Mask
  sentinel races are the #1 silent corruptor of multi-block top-k reductions
  (`wins/2026-05-28-cuda-gap-b-dsv4-route-block-parallel.md`).
- **Wrapper-class porting (e.g. DeepEP torch wrapper → torch-free C++) is usually 80% smaller
  than first sizing.** Grep the upstream kernel layer for framework coupling
  (`grep -rln "torch::\|at::\|c10::" csrc/kernels/`) before estimating a "multi-week rewrite"
  (`wins/2026-05-26-dsv4-deepep-cpp-kernels-torch-free.md`).
- **Vendored-kernel wrapper bugs hide in "non-mandatory" tensors.** Hand-check the Rust wrapper
  against upstream's reference invocation (typically a Python C++ extension that lays out all
  params via `torch::empty`) — even tensors the kernel "doesn't require" may be unconditionally
  written (`wins/2026-05-28-dsv4-v2-4-flashmla-root-cause-fix.md`).

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
