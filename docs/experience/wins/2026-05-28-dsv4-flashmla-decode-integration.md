# DSv4 FlashMLA decode integration — vendor + shim landed; runtime wire-up killed at D-3 (FP8 KV contract not satisfied by ARLE's bf16 pool)

## SLO-shape probed? — N (pre-flight blocked on KV layout mismatch; no end-to-end TPOT delta measured)

## TL;DR

Vendored upstream FlashMLA SM90 sparse-FP8 decode (`sm90::decode::sparse_fp8`
+ `smxx::decode::{combine,get_decoding_sched_meta}` at pin `df022eb`),
extended `build.rs` to compile the 4 instantiations + combine + sched-meta
kernel against ARLE's existing FlashMLA include path, added the
`arle_flashmla_decode_shim.cu` (4 entry points: `get_meta` /
`bytes_per_token` / `sched_meta` / `decode_fwd`) with full pre-flight
+ exception-safety, and declared the FFI in
`crates/cuda-kernels/src/ffi/misc.rs`.

**Phase D-3 killed before runtime wire-up.** The design doc's KV-adapter
approach (build per-token block-paged indices on top of ARLE's existing
bf16 sliding-window + compressed pool) is falsified by the upstream kernel
source: `sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel`
reinterprets `params.kv` as `fp8*` and asserts
`stride_kv_row == BYTES_PER_TOKEN` where the byte layout is model-specific
and packs FP8 NoPE + bf16 RoPE + fp8_e8m0 scales contiguously per token.
ARLE's KV pool is unquantized bf16 and cannot satisfy that contract.

## Roofline check

| Op | Achieved | Peak (8×H20 BF16) | % | Verdict |
|---|---:|---:|---:|---|
| DSv4 decode TPOT (legacy `dsv4_hybrid_attention_cuda`, no change) | ~26 ms/tok | ~5–10 ms/tok target | — | unchanged — runtime dispatch still routes through the legacy kernel; FlashMLA decode path stayed gated OFF |

No bench delta this session — the win is the **honest kill** at the
license-or-kill gate rather than a silent half-state. Per CLAUDE.md §0,
"推断 ≠ SOLID"; the design doc's KV adapter sketch was a hypothesis, and
the upstream source is evidence falsifying it.

## What landed

1. **Vendor** (`5d18b624` — chore(cuda)): mirrored 14 upstream files into
   `crates/cuda-kernels/vendor/flashmla/csrc/` (sparse_fp8 dir + combine
   + sched_meta + the splitkv_mla header) at the existing prefill pin.

2. **Build + shim + FFI** (`3f7923a3` — feat(cuda)): `build.rs` now adds
   the 6 decode `.cu` files to `cu_files` and extends the
   `is_flashmla_kernel || stem == arle_flashmla_shim` gate to include the
   new `arle_flashmla_decode_shim` stem. The shim implements four entry
   points (see commit body); FFI declarations in `src/ffi/misc.rs` mirror
   them.

3. **Runtime dispatch — not modified.** The `weights.rs::finish_attention_gpu`
   `token_count == 1` branch still calls `dsv4_hybrid_attention_cuda`.
   The env knob `ARLE_DSV4_FLASHMLA_DECODE` is **not introduced** because
   wiring it up before D-3's blocker is resolved would create exactly the
   parallel-old-new-paths half-state CLAUDE.md forbids.

## Root-cause finding — why D-3a is dead as designed

The design doc assumes upstream's decode kernel will consume a block-paged
bf16 KV buffer at `[num_blocks, page_block_size=64, d_qk]`. Re-reading
the kernel source shows that's only the declared C type, not the actual
contract:

| Field | Declared | Kernel actually does |
|---|---|---|
| `SparseAttnDecodeParams::kv` | `cutlass::bfloat16_t* __restrict__ kv` ([`csrc/params.h:72`](https://github.com/sgl-project/FlashMLA/blob/df022eb/csrc/params.h)) | `fp8* k_ptr = (fp8*)params.kv` ([`splitkv_mla.cuh:491+`](https://github.com/sgl-project/FlashMLA/blob/df022eb/csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh)) — reinterprets as fp8 and runs `cvt_fp8x8_bf16x8` dequant inside the kernel. |
| `stride_kv_row` | `int` ([`csrc/params.h:86`](https://github.com/sgl-project/FlashMLA/blob/df022eb/csrc/params.h)) | `KU_ASSERT(params.stride_kv_row == BYTES_PER_TOKEN)` for MODEL1, `== 656` for V32 ([`splitkv_mla.cuh:695,702`](https://github.com/sgl-project/FlashMLA/blob/df022eb/csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh)) |
| Per-token byte layout | unspecified in struct | MODEL1: 448 fp8 NoPE + 128 bf16 RoPE + 7 fp8_e8m0 scales + 1 pad = **584 bytes/token** (`config.h:NUM_SCALES = HEAD_DIM_NOPE/QUANT_TILE_SIZE`) |

The upstream torch wrapper enforces this in its preflight too —
`KU_CHECK_DTYPE(kv, torch::kFloat8_e4m3fn || torch::kInt8 || torch::kUInt8)`
([`csrc/api/sparse_decode.h:260`](https://github.com/sgl-project/FlashMLA/blob/df022eb/csrc/api/sparse_decode.h)).
There is no bf16 input path; the kernel requires the model-specific
fp8-packed contract.

ARLE's DSv4 KV cache today:
- **Sliding-window cache**: `bf16 [sliding_window * head_dim]` per layer per
  request, ring-buffered by `(key_pos % sliding_window)` —
  `dsv4_attention.cu:347-349`.
- **Compressed pool**: `bf16 [compressed_count * head_dim]` per layer per
  request — fed into `dsv4_hybrid_attention_cuda` as the `compressed` arg.

Both are unquantized bf16. The FlashMLA decode kernel cannot consume
them without an intermediate FP8-packing step that:
1. Quantizes the bf16 NoPE portion to fp8 e4m3 at quant_tile_size=64
   (MODEL1) or 128 (V32), producing fp8_e8m0 scales.
2. Copies the bf16 RoPE tail through.
3. Lays the result out at the exact per-token byte stride the kernel
   asserts.

That FP8-packing kernel does not exist in ARLE today, and writing it
isn't "the KV adapter" the design doc describes (D-3a says "build per-token
indices in FlashMLA's block coords on-the-fly" — that's the index map, not
the dtype conversion). The work item is materially different:

- **Design doc D-3a estimate**: 4–6 hours (index map only).
- **Real work**: a per-step `bf16 → fp8 + scales + rope_tail` packing
  kernel that runs every decode step before the FlashMLA dispatch, plus
  an allocator for the per-layer per-request FP8 KV pool. Estimate ~1–2
  weeks including correctness vs the bf16 reference, and the per-step
  packing overhead may eat most of the FlashMLA speedup unless we keep
  a persistent FP8 pool that's written-through at compressor-update time.

## SOLID-aligned decision

Per CLAUDE.md §0:
- "推断 ≠ SOLID" — the design doc's D-3a sketch was a hypothesis falsified
  by `splitkv_mla.cuh:491` (the `(fp8*)params.kv` reinterpret + the
  `BYTES_PER_TOKEN` assertion).
- "Framing 多角度交叉" — wall-clock ground truth: ~16 ms/token of the
  current 26 ms decode is the legacy kernel itself; the rest is per-step
  setup (compressor update, CSA select, window cache update). Swapping
  the kernel alone gives at most ~16 → ~6 ms = 10 ms saved, but only IF
  the FP8 pack/write is amortized into the compressor update (free) and
  IF the index-map overhead is sub-ms.
- "License-or-kill 决策必须用 wall-clock framing": 5–10 ms TPOT savings
  is the upside, ~1–2 weeks of kernel work is the cost. Not the right
  bet against ARLE's current next-axis priorities (A4 multi-stream
  overlap, expected ~24K chunked-prefill wash recovery).

**Decision**: vendor + shim + FFI land (so a future session can wire up
quickly), but the runtime dispatch does not change in this session.
`ARLE_DSV4_FLASHMLA_DECODE` env knob is not introduced — adding a gated
runtime branch with no test coverage and a known-blocked path would be
the "parallel old + new paths" half-state CLAUDE.md forbids.

## What's still on the table

1. **FP8 KV pool design**. The honest next axis here is "do we want to
   take DSv4 decode KV to FP8 e4m3 with fp8_e8m0 scales?" — that's a
   model-quality question (numerical impact of FP8 KV at decode time),
   not a kernel-integration question. If yes, the packing kernel + pool
   rework would be ~1–2 weeks and would also unlock the upstream decode
   kernel as a free downstream.
2. **A4 multi-stream overlap (prefill side)** — referenced as the next
   axis in the V2.4 wins entry. Still the right priority over decode
   FP8 if 24K chunked-prefill is the binding SLO.
3. **Light alternative**: profile whether the legacy
   `dsv4_hybrid_attention_cuda` decode kernel itself can be tuned (e.g.
   block-size, vectorization) for the ~26 → ~15 ms band without crossing
   the FP8 quant boundary. Cheap to try, may surface a wash-or-win
   without the FP8 rework.

## Rule

**Before promising a kernel-integration win, read the kernel — not the
struct.** The design-doc miss here was reading `cutlass::bfloat16_t* kv`
in `params.h` and not the `(fp8*)params.kv` + `BYTES_PER_TOKEN` assert in
`splitkv_mla.cuh`. The torch wrapper's `KU_CHECK_DTYPE` is the API-side
mirror of that, and `sparse_decode.h` was already accessible from the
existing vendor tree. A 10-minute grep for `(fp8*)params.kv` would have
caught this in the planning phase.

## Refs

- Project plan: [`docs/plans/2026-05-28-dsv4-flashmla-decode-integration.md`](../../plans/2026-05-28-dsv4-flashmla-decode-integration.md)
- V2.4 prefill wins entry (sibling integration): [`2026-05-28-dsv4-v2-4-flashmla-root-cause-fix.md`](2026-05-28-dsv4-v2-4-flashmla-root-cause-fix.md)
- Vendor pin: sgl-project/FlashMLA @ df022eb (same pin as existing prefill)
- Upstream kernel evidence:
  - `csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh:491,576-597,695` — `(fp8*)params.kv` + `BYTES_PER_TOKEN` assert + `cvt_fp8x8_bf16x8` dequant
  - `csrc/sm90/decode/sparse_fp8/components/config.h` — per-token byte layout breakdown
  - `csrc/api/sparse_decode.h:260,289-304` — torch-side preflight enforcing the FP8 dtype + byte layout
- Commits this session:
  - `5d18b624` — vendor decode sources (D-1).
  - `3f7923a3` — build.rs + shim + FFI (D-2 + FFI).

## Pending — local bench

`pending-remote` per CLAUDE.md (no SLO-shape bench run because the
runtime hot path was deliberately not changed; legacy decode kernel
still in place at 26 ms/tok baseline).
