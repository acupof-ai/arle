# `infer::ops` — Agent Guide

Tensor ops over `DeviceVec` / `HiddenStates`. Thin Rust wrappers around
`infer_cuda_kernels::ffi::*` with batched/fused variants for scheduler use.
Load before adding or modifying any op.

## Refactor posture

- Keep ops simple and uniform. Prefer deletion-style refactors: remove stale
  wrappers, collapse duplicate batched/scalar helpers when one path can carry
  both, and keep one canonical entry point per operation family.

## Module layout

```
ops.rs                — module root, submodule #[path] decls, pub surface
ops/attention.rs      — TileLang paged prefill/decode + quantized custom CUDA decode
ops/linear.rs         — gemm, gemv, fused_mlp
ops/norm.rs           — rms_norm, fused_add_rms_norm (batched + offset variants)
ops/embedding.rs      — embedding_batch, embedding_decode_into
ops/elementwise.rs    — add_batch, silu_mul_batch, extract_vec
ops/kv_ops.rs         — scatter_write_kv
ops/recurrent.rs      — gated_delta_rule_prefill_chunkwise + decode (Qwen3.5 linear attention)
ops/sampling.rs       — argmax, argmax_with_logprob, gpu_sample
ops/tests.rs          — unit tests (CPU can run with `no-cuda`)
```

## Visibility policy

- **`pub`** — ops used outside this crate's `model/` + `backend/` (today:
  `tilelang_tc_run_layer`, `gemm`, `gemv`, `fused_mlp_into`, the non-batched
  `rms_norm_into`, sampling entry points, `scatter_write_kv`).
- **`pub(crate)`** — batched `_batched_into` variants, anything with a
  `_batch` suffix, and model-internal fused paths (`nonpaged_prefill_hd256_into`,
  `attention_gate_paged_hd256`, `fused_add_rms_norm_batch_into`).

Keep the `pub` surface small: the `_batched_into` + scheduler paths are
*not* stable API; they evolve with the scheduler. New op? Start `pub(crate)`;
promote only when a cross-crate consumer appears.

## Naming conventions (enforced)

| Suffix | Meaning |
|--------|---------|
| `_into` | Writes into a caller-provided output buffer. No allocation. Always prefer. |
| `_batch` / `_batched` | Single launch over B requests. Arg slices in scheduler batch order. |
| `_offset` | Reads/writes at a scheduler-supplied row offset inside a pre-allocated buffer. |
| `_fused` | One kernel launch combining two conceptual ops (e.g. `fused_add_rms_norm`). |
| `_gated` | Has an extra gate/residual path — typical in Qwen3.5 hybrid layers. |

**Compose, don't duplicate.** If you need an allocating op, call the `_into`
version with a scratch buffer; don't add a second non-`_into` variant unless
there's an existing caller that can't hold the buffer.

## Hot-path rules

1. **No per-call GPU allocations.** Scratch, KV metadata, and logit buffers
   must be owned by the `DecodeContext` on the scheduler side.
2. **Every call takes `&DeviceContext`.** The stream/cublas handle is
   threaded explicitly; global statics are forbidden.
3. **Batched ops read their inputs in scheduler batch order.** If you index
   with `slot_idx`, you're probably wrong — the scheduler passes compact
   batch positions, not slot indices.
4. **Parameter structs for high-arity ops.** See `attention.rs::NormRopeParams`,
   `HeadConfig`, `PagedKVMeta`. ≥4 related params → make a struct.
5. **Attention decode has three paged paths** (`attention.rs`): BF16 via
   TileLang paged attention, INT8 via custom split-KV kernel with fused
   dequant, FP8 via custom split-KV kernel with FP32 cast. The selector is `KVFormat`, not
   the model — adding a fourth format means a fourth path.
6. **Prefill dispatches on head dim:** HD128 -> TileLang paged prefill HD128,
   HD256 -> TileLang paged prefill HD256. Qwen3 = HD128;
   Qwen3.5 full-attention = HD256.
7. **Single-token BF16 decode** uses the TileLang paged decode path. Don't
   split or add a second BF16 path without a bench snapshot.

## `OpsBackend` trait (M4 unification surface)

`infer/src/ops.rs` defines `OpsBackend` + `Tensor` (M4 of
`docs/plans/backend-unification.md`). The trait is the cross-backend
contract for the 5 hot ops (norm + linear + sampling + elementwise +
embedding); attention / recurrent / kv_ops are deliberately NOT in the
trait yet (M5+ scope).

**When adding a method:**
- Mirror the existing `_into` convention (caller-supplied output).
- Pass `&DeviceContext` even when the backend doesn't need it — the
  trait stays backend-neutral.
- CUDA impl in `CudaOpsBackend` (`infer/src/ops.rs`) — must be a thin
  forward to the existing free fn, **not** a re-implementation. The
  free fn stays the canonical kernel entry; the trait method just
  routes.
- Metal impl in `MetalOpsBackend` (`infer/src/backend/metal/ops.rs`)
  goes through MLX. Lazy-graph `eval()` is the impl's responsibility,
  not the caller's.
- New methods land `pub(crate)` first; promote only when a cross-crate
  consumer appears.

**When migrating a callsite:**
- Model code (`infer/src/model/qwen3/*.rs`) takes `&dyn OpsBackend` (or
  generics) and calls the method instead of the free fn directly.
- The CUDA-graph capture path may stay on the free fn for now —
  raw-pointer launches with cached pointers are still on the legacy
  free-fn ABI per `5f209d2`'s commit body.
- Migrate every callsite of one op family in one commit; do NOT
  partial-migrate (`961ac13` / `477b761` / `c70ad34` / `5f209d2` are
  the precedent — one op family per commit).

**Don't:**
- Don't dual-route a callsite (one branch calls the free fn, another
  calls the trait method) — pick one.
- Don't add backend-specific methods (`fn cuda_only_xxx`) to the trait;
  put those on a backend extension trait or keep them on the free fn.

## Pointers

- `crates/cuda-kernels/src/prelude.rs` — the types you're allowed to
  take as arguments across the crate boundary.
- `crates/cuda-kernels/csrc/` — the underlying CUDA C source.
- `crates/cuda-kernels/tools/tilelang/` — TileLang AOT kernel definitions.
- `docs/reviews/2026-04-14-cuda-kernel-six-principles-review.md` — the
  audited kernel heat map; don't add a new op without checking where it
  lands on this list.
- `docs/experience/wins/2026-04-14-bench-single-token-kernel-port.md` —
  historical context for the single-token CUDA C decode port.
