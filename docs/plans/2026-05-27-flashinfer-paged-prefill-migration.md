# FlashInfer drop-in for paged prefill+decode (Path F)

## Why

Two unrelated TileLang 0.1.10 regressions landed in one week
(2026-05-27):

1. CUDA 12.2 + cutlass C++20 build break
   ([`2026-05-27-tilelang-0110-cuda122-cutlass-incompat.md`](../experience/errors/2026-05-27-tilelang-0110-cuda122-cutlass-incompat.md))
2. sm_80 FullRow + warps 2/3 runtime NaN
   ([`2026-05-27-tilelang-0110-fullrow-warp23-nan-sm80.md`](../experience/errors/2026-05-27-tilelang-0110-fullrow-warp23-nan-sm80.md))

Both pin to 0.1.9 as the short-term fix. Both signal that we are
the **only mid-tier user** of TileLang's varlen+paged+GQA+bf16 prefill
on sm_80 — every 0.1.x release surfaces a regression we have to
diagnose and patch ourselves.

Our paged prefill ABI (`qo_indptr` / `kv_indptr` / `kv_indices` /
`kv_last_page_len`) already **mirrors FlashInfer's C++ template
signature verbatim**, and the existing kernel's docstring is full of
"`Mirrors FlashInfer's mask_iteration in prefill.cuh`"-style
attributions. We are reimplementing FlashInfer in TileLang. The bug
budget for this strategy is exhausted.

vLLM, SGLang, TensorRT-LLM, MLC all ship FlashInfer as the production
paged-attention backend on sm_80+. Moving to FlashInfer:

- Removes the TileLang AOT codegen + cubin caching + sm70 patch +
  version-pinning maintenance burden for this surface.
- Inherits upstream fixes and perf tuning we currently fork-and-track
  by hand.
- Unifies prefill and decode under one ABI (today TileLang covers both
  but with separate kernel files and separate bug surfaces).
- Aligns with industry — community contributors and downstream users
  see a familiar primitive.

## What we keep

- **TileLang stays for everything FlashInfer doesn't cover**:
  Qwen3.5 GDR / linear-attention chunked recompute, DSv4 MoE-route
  scratch, decode FP8 quant decode variants, prefill HD64/HD256
  (covered by FlashInfer but only after we validate parity).
- **TileLang stays the build-time AOT path** for those remaining
  kernels — we don't rip the whole TileLang integration out.
- **Custom CUDA C** in `crates/cuda-kernels/csrc/{attention,kv,quant}`
  stays. FlashInfer covers the attention-kernel surface; quant kernels
  (FP8/INT8/TQ pack/unpack, KIVI per-channel calibration) and KV pool
  management stay ours.

## FlashInfer API surface to integrate

FlashInfer's primary entry point for our pattern is
`BatchPrefillWithPagedKVCacheWrapper` (header-only C++ at
[flashinfer/prefill.cuh](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/prefill.cuh)):

```cpp
template <typename DTypeQ, typename DTypeKV, typename DTypeO,
          typename IdType>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(
    DTypeQ* q,                          // [total_qo_tokens, num_qo_heads * head_dim]
    IdType* qo_indptr,                  // [batch + 1]
    paged_kv_t<DTypeKV, IdType> paged_kv, // wraps page_table, page_indices, etc.
    DTypeO* o,                          // [total_qo_tokens, num_qo_heads * head_dim]
    float* lse,                         // optional, [total_qo_tokens, num_qo_heads]
    uint32_t num_qo_heads, uint32_t num_kv_heads,
    QKVLayout kv_layout, /* causal mask, sm_scale, rope settings */ ...,
    cudaStream_t stream);
```

This is **exactly** what our current `prefill_attention_paged_batch`
in `infer/src/ops/attention.rs:589` produces (modulo the
TileLang-vs-FlashInfer kernel binding).

For decode: `BatchDecodeWithPagedKVCacheWrapper` (same paged-kv struct,
different m=1 path).

Both come from one header tree, one set of CUDA templates, one set of
dtype enums. dtypes: `__nv_bfloat16` / `half` / `__nv_fp8_e4m3` /
`__nv_fp8_e5m2`. INT8 KV is not first-class in FlashInfer mainline —
TBD whether we keep our INT8 quantize+TileLang path or also migrate.

## Build integration

FlashInfer is **header-only C++ + CUDA**, built via JIT or AOT.
For ARLE's ahead-of-time policy, AOT is the only option:

```
crates/cuda-kernels/
├── build.rs                    ← extend to call flashinfer AOT codegen
├── tools/
│   ├── tilelang/               ← stays (Qwen3.5 GDR / DSv4 / decode FP8)
│   └── flashinfer/             ← new
│       ├── gen_flashinfer_aot.py    ← invokes FlashInfer's
│       │                               aot_build_utils.py for each
│       │                               (dtype, head_dim, page_size,
│       │                                num_qo_heads, num_kv_heads)
│       │                               specialization in our matrix
│       └── README.md
└── csrc/
    └── attention/
        └── flashinfer_dispatch.cu  ← thin wrapper exposing C ABI
                                        callable from Rust FFI
```

FlashInfer's `aot_build_utils.py` (upstream) takes a JIT spec list
and emits `.cu` files + a CMake target. We mirror our TileLang AOT
pattern: enumerate `(num_q_heads, num_kv_heads, head_dim, page_size,
kv_dtype)` per supported SM tier, codegen at build time, compile
into static `libflashinfer_kernels_aot.a`, link into `cuda-kernels`.

Same SM-tier policy as TileLang (see
[`docs/plans/sm-coverage.md`](sm-coverage.md)): T1 = sm_80 (every
kernel must emit cubin). FlashInfer does not support sm_70 → V100
takes the existing CUDA C `nonpaged_prefill_attention.cu` contig
path (already works there per
`docs/experience/wins/2026-05-25-v100-sm70-p1-build-pass.md`).

## Rust FFI

Mirror the existing `crates/cuda-kernels/src/ffi/attention.rs` pattern:

```rust
// New: crates/cuda-kernels/src/ffi/flashinfer.rs
unsafe extern "C" {
    pub fn flashinfer_batch_prefill_paged_run(
        head_dim: c_int,
        page_size: c_int,
        num_qo_heads: c_int,
        num_kv_heads: c_int,
        kv_dtype: c_int,          // 0=BF16, 1=FP8E4M3
        q_ptr: *const c_void,
        qo_indptr_ptr: *const c_int,
        kv_indices_ptr: *const c_int,
        kv_indptr_ptr: *const c_int,
        kv_last_page_len_ptr: *const c_int,
        k_pool_ptr: *const c_void,
        v_pool_ptr: *const c_void,
        out_ptr: *mut c_void,
        batch_size: c_int,
        total_qo_tokens: c_int,
        sm_scale: c_float,
        stream: cudaStream_t,
    ) -> cudaError_t;
}
```

Dispatch table in Rust selects the AOT'd specialization based on
runtime shape (mirrors what `infer/src/ops/attention.rs:589`
`PagedPrefillForward::new_hd128` does today for the TileLang HD128
variants).

## Phased migration

Each phase is independently shippable + revertable behind a feature
flag `cfg(feature = "flashinfer-prefill")`.

**Phase 1 — Smoke (3-5 days):**
- Vendor FlashInfer at a pinned commit under
  `crates/cuda-kernels/3rdparty/flashinfer/` (git submodule or
  vendored dir, same pattern as `vendor/deepgemm/` from 66a76819).
- Stand up `gen_flashinfer_aot.py` with **one** specialization:
  `(num_q_heads=32, num_kv_heads=8, head_dim=128, page_size=16, BF16)`.
- Build with `cargo build -p cuda-kernels --features flashinfer`,
  cubin succeeds.
- Direct `forward_token_logits`-style smoke: feed a single 14-token
  ChatML prompt through the new path, verify argmax matches contig
  reference (151667 for Qwen3-4B).

**Phase 2 — Parity (3-5 days):**
- Plug into `infer/src/ops/attention.rs::prefill_attention_paged_batch`
  behind feature flag.
- Run `kv_precision_parity` and `kv_fp8_prefill_logit_parity` —
  same shapes as today's TileLang path.
- Expand specializations to all Qwen3/Qwen3.5 head configs
  ((16,8), (32,8), (40,8), (64,8)).
- Add FP8 KV specialization.

**Phase 3 — Decode (3-5 days):**
- Add `flashinfer_batch_decode_paged_run` mirroring TileLang's
  `batch_decode_paged_hd128` family.
- Plug into the decode path.
- Validate parity vs the existing TileLang decode kernels.

**Phase 4 — Default flip (1 day, after parity gates pass):**
- Flip `default = ["flashinfer-prefill"]` in
  `crates/cuda-kernels/Cargo.toml`.
- TileLang prefill HD128 stays buildable behind a non-default flag
  for ~1 release as a rollback safety net.

**Phase 5 — Decommission TileLang prefill HD128 (1 release later):**
- Delete `crates/cuda-kernels/tools/tilelang/batch_prefill_paged_hd128.py`
  + dispatch table + AOT codegen for the prefill HD128 family.
- TileLang stays for HD64/HD256 prefill (FlashInfer also covers,
  migrate opportunistically), decode (separate decision), Qwen3.5 GDR,
  DSv4 MoE.

## Validation gate

Each phase must pass before next phase starts:

| Phase | Gate | How to verify |
|---|---|---|
| 1 | Smoke prefill produces argmax=151667 on 14-token ChatML | `kv_fp8_prefill_logit_parity` adapted to call flashinfer path |
| 2 | `kv_precision_parity` mean_match for BF16 paged ≥ 1.0 (matches contig ref) | Compare per-precision first 8 tokens against contig BF16 — full match expected |
| 3 | `kv_precision_parity` mean_match for BF16 paged ≥ 1.0 with max_tokens=64 | Decode path also exercised |
| 4 | TTFT / ITL Δ ≤ +5% vs TileLang baseline (or better) | `scripts/bench_guidellm.sh flashinfer-prefill-default` vs prior bench snapshot |
| 5 | No regressions on 30-day post-flip bench cycle | Existing wins/ comparison |

## V100 / sm_70 story

FlashInfer requires sm_80+ tensor-core paths. V100 (sm_70) can't run
the FlashInfer kernels. Concrete handling:

- `crates/cuda-kernels/build.rs` already gates per-SM cubin emission
  via `TORCH_CUDA_ARCH_LIST` (see `docs/plans/sm-coverage.md`).
  FlashInfer cubins emit only for sm_80+ tiers.
- For sm_70 (V100): `infer/src/ops/attention.rs` dispatches to the
  existing `nonpaged_prefill_attention.cu` contig path, which already
  works on V100 with the `scripts/sm70_tilelang.patch` FMA fallback
  (currently used by the V100 audit).
- Paged-pool memory efficiency loss on V100 is acceptable —
  V100 is not a production T1 SKU (it's a P2 legacy-validation tier
  per `sm-coverage.md`).

## Out of scope (this plan)

- Decode-side migration is Phase 3, not Phase 1-2.
- INT8 KV: stays on the current quantize-then-attention path until
  we decide whether FlashInfer's INT8 support is mature enough to
  migrate (separate decision after Phase 2).
- Qwen3.5 GDR / DSv4 MoE / weight-quant Marlin paths: untouched.
- ROCm / Vulkan multi-backend (per
  `docs/plans/2026-05-05-multi-backend-tilelang-rocm-vulkan.md`):
  FlashInfer CUDA-only; ROCm uses Composable Kernel via TileLang
  for now.

## Trip wires

Kill the migration and stay on TileLang 0.1.9 pin if:

- Phase 1 smoke fails — FlashInfer paged prefill produces wrong
  argmax (= FlashInfer has the same bug — unlikely given vLLM/SGLang
  production use, but possible if our paged-KV layout differs from
  FlashInfer's expectations).
- Phase 2 parity fails for any KV format — investigate before
  flipping.
- Phase 4 bench shows >10% TTFT regression — defer flip, profile,
  or settle on hybrid (flashinfer for FP8, TileLang 0.1.9 for BF16).

## Effort

~3-4 weeks of focused work, ~1 dev. Bulk of risk is in build
integration (FlashInfer's nvcc invocations, header dependency tree).
Once build is clean, kernel parity + perf is the well-trodden vLLM /
SGLang path.

## Related

- [`docs/experience/errors/2026-05-27-tilelang-0110-fullrow-warp23-nan-sm80.md`](../experience/errors/2026-05-27-tilelang-0110-fullrow-warp23-nan-sm80.md)
  — the bug this plan removes.
- [`docs/experience/errors/2026-05-27-tilelang-0110-cuda122-cutlass-incompat.md`](../experience/errors/2026-05-27-tilelang-0110-cuda122-cutlass-incompat.md)
  — second TileLang 0.1.10 regression, also pinned to 0.1.9.
- [`docs/research/2026-05-07-sglang-prefill-stack-survey.md`](../research/2026-05-07-sglang-prefill-stack-survey.md)
  — confirms FlashInfer is SGLang's attention backend.
- [`docs/plans/tilelang-integration.md`](tilelang-integration.md) —
  the existing TileLang integration; this plan does NOT remove it,
  only carves out the paged-prefill HD128 surface.
- [`docs/plans/sm-coverage.md`](sm-coverage.md) — SM tier policy
  applies identically.
