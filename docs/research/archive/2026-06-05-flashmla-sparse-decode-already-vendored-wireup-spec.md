# DSv4-Flash MLA decode — the fused kernel is already vendored; the work is wire-up

**Date:** 2026-06-05. Code-level read of SGLang's `dsv4` attention backend +
FlashMLA (`vendor/flashmla` `63a1a061`). **Supersedes the "weeks-long FlashMLA
port" framing and corrects `2026-06-05-dsv4-fp8-kernel-upstream-scan.md` §Q3.**

## Headline

ARLE's decode attention is 3 **separate scalar** bf16 kernels per layer (SW +
compressed-sparse + hyper-compressed) — the anti-pattern. SGLang's `dsv4` backend
(`DeepseekV4AttnBackend`) launches **exactly one fused kernel** per layer:
`flash_mla_with_kvcache` → `sparse_decode_fwd` → FlashMLA
`sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel`
(`vendor/flashmla/csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh:686`). SW (`indices`)
+ the layer's c4/c128 compressed stream (`extra_*`) are walked in **one** producer
loop into the **same** K smem buffers — one attention over one merged KV stream.

**And ARLE already has every CUDA piece vendored/written — only the runtime
wire-up in `attention.rs` is missing.** This is "delete 3 scalar kernels, route
decode through the kernel already in the tree," not a kernel-authoring project.

## Why it fixes the SM-1-3% occupancy bottleneck (the ncu finding)

Two stacked structural mechanisms — grid/tile, not precision:
1. **MQA-absorb** (`h_kv==1` asserted): the 64–128 q-heads become the `BLOCK_M=64`
   rows of the QK WGMMA. One decode token → a full 64-row tensor-core tile (not a
   1-row scalar reduction). Latent KV read from HBM once, shared across heads.
2. **Split-KV persistent grid** sized to the device:
   `num_sm_parts = max(num_sms / s_q / (h_q/64), 1)`, `grid = (h_q/64, s_q,
   num_sm_parts)`. **B=1 on H20 (132 SMs): `132/1/2 = 66` → grid `(2,1,66) = 132
   CTAs = one per SM`** — a single decode token fills the whole device. The topk
   KV (SW + compressed) is partitioned across the CTAs by `get_decoding_sched_meta`;
   a `combine` kernel merges partials via log-sum-exp. This is precisely the
   1-3% → full-occupancy lever ncu pointed at.

## What's already in ARLE's tree (reuse, do NOT rewrite)

| Component | Location |
|---|---|
| Fused sparse decode kernel | `vendor/flashmla/csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh` (+ combine + sched_meta) |
| Shim + FFI | `arle_flashmla_sm90_sparse_decode_fwd` / `_get_meta` / `_sched_meta` / `_bytes_per_token` (`csrc/misc/arle_flashmla_decode_shim.cu`, `ffi/misc.rs:506-583`) |
| FP8-KV pack | `csrc/attention/dsv4_fp8_kv_pack.cu` (523 LOC, MODEL1 584-byte AoS + e8m0) |
| CSA top-k selector + index build | `dsv4_csa_select_cuda` + `arle_flashmla_{csa,hca}_build_indices` + `dsv4_flashmla_decode_build_indices` |

**Missing:** `crates/infer-cuda/src/attention.rs` decode dispatch (~1130-1410)
still calls scalar `dsv4_swa_attention_splitkv_cuda` / `dsv4_hybrid_attention_splitkv_cuda`;
the FlashMLA path is bail-gated behind `Dsv4MlaKvArena::alloc_fp8_arena`.

## Wire-up spec

1. **Un-gate `alloc_fp8_arena`** + route KV store through `dsv4_fp8_kv_pack_kernel`
   (584-byte FP8 MODEL1 layout) — the sparse kernel exists **only** in FP8 form
   (`is_fp8_kvcache` asserted, `stride_kv_row==584`), so FP8-KV is a hard
   prerequisite to *running* it (orthogonal to *why* it's fast).
2. **Rewrite the decode dispatch** to one `arle_flashmla_sm90_sparse_decode_fwd`
   per layer: SW → `indices`/`topk_length`; the layer's c4/c128 →
   `extra_k_cache`/`extra_indices`/`extra_topk_length` (existing builders produce
   both).
3. **Delete the 3 scalar attention kernels** *after* parity (keep as bf16 reference
   until the FP8 path passes KV-parity).

**Hard asserts to honor:** sm90a; `h_kv==1`; `h_q∈{64,128}`; `d_qk==512` MODEL1 /
`d_v==512`; `topk%64==0` + 64-aligned indices; `page_block_size==64`. ARLE's
`index_topk=512` + 448-NoPE/64-RoPE=512 already match MODEL1.

## Gate

KV-precision-parity (fused FP8 vs the bf16 scalar reference being deleted) + nsys/ncu
before/after on the B=1 SLO shape showing the SM-occupancy jump, as a `wins/` entry.
**Verify the pod built the new symbol (`strings | grep arle_flashmla_sm90_sparse_decode`)
before trusting parity** — the `errors/2026-05-28-dsv4-flashmla-decode-parity-precond-fail`
trap.

## Correction to the prior scan

`2026-06-05-dsv4-fp8-kernel-upstream-scan.md` §Q3 said "SW/CSA/HCA is ARLE-original,
FlashMLA gives only the dense base." **Wrong for decode** — that scan read the
*dense* `decode/dense/` path; the **sparse** `splitkv_mla.cuh` natively fuses SW
(`indices`) + compressed (`extra_*`), and SGLang runs exactly it for DSv4-Flash.
ARLE's SW/CSA/HCA map one-to-one onto a single upstream kernel's args; the "ours"
framing is what produced the 3-scalar-kernel anti-pattern.

**Lesson:** before grinding a hand-rolled kernel toward an upstream number, check
whether the upstream kernel is *already vendored* and just unwired — the
kernel-registry's "library-present but unwired" rows (`arle_flashmla_*`) were the
tell. Reading the reference's actual call path beats incrementally rediscovering it.
