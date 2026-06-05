# DSv4-Flash decode: contiguous active-row MoE layout → +12.78% (MoE slice −49.8%)

**Date:** 2026-06-05. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Status:** landed, **gated** (`ARLE_DSV4_MOE_CONTIG_DECODE=1`, masked default),
correctness = resident oracle16 + 80-tok no-bail (full KV-parity gate pending —
see Honest read). Endgame MoE lever; supersedes the grouped-GEMM-kernel-swap
hypothesis.

## Context

The lever order was corrected three times by §0 profiling. The fresh stage
profile put the **MoE expert path (~14.6 ms/token)** as the biggest non-attention
slice — but a detail probe ([[2026-06-05-dsv4-resident-ab-flashmla-decode-18pct]]
closeout method) pinned the cost on the **padded layout, not the kernel**:
inside `moe_deepgemm_grouped` (11.5 ms), `dg_unpad 4.5 + dg_pack_quant 3.7 +
dg_swiglu_quant 2.0 ≈ 10.2 ms` was pack/unpad/materialize of a `32 groups × 128
padded rows` masked layout — while the actual w13+w2 WGMMA GEMMs cost only
**2.99 ms**. At B=1/topk=6 most of the 32 experts have count=0, so the padded
layout does ~10 ms of wasted materialization. The kernel was never the problem.

## What Worked

Adopt SGLang's **contiguous** decode layout (`先用最好的` —
[[feedback_no_closed_door_solutions]]): SGLang's `deepep_normal → deep_gemm` path
runs `ep_scatter → m_indices` with `use_masked_gemm=False`, materializing only the
active rows. ARLE now mirrors it for seq_len==1 decode (gated):

- **`deepgemm_native.cu` + FFI**: new **DeepGEMM `MGroupedContiguous`** native
  bridge (`dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous`) — the vendored
  contiguous WGMMA entry that was unexposed.
- **`dsv4_route.cu` + FFI**: new pack kernel writing the active route tile
  (`route_capacity × 128`, padding `-1`), producing `m_indices` (per-row local
  expert id) for the contiguous GEMM.
- **`infer-cuda/src/moe.rs`**: gated `use_contiguous_decode_moe()` path
  (`Dsv4GroupedContiguousDecodeScratch`) — packs only `num_tokens×topk` active
  rows, skips the 32×128 padded pack/unpad entirely. Masked path unchanged as the
  default fallback.

## Results (resident same-load A/B, B=1, steady decode, both orders × 3)

| variant | steady tok/s | moe_deepgemm_grouped |
|---|---:|---:|
| flashmla + fused wqkv (baseline) | 29.243 ± 0.022 | 11.542 ms/token |
| **+ contiguous MoE** | **32.979 ± 1.049** | **5.794 ms/token** |
| Δ | **+12.78%** | **−49.8%** |

Both `oracle16=PASS`, 80-token run no-bail. **Cumulative decode arc: scalar 23.7 →
FlashMLA 28.0 → fused 29.4 → contiguous MoE 33.0 tok/s (+39% over scalar; ~1.9×
off SGLang no-spec 62.9).**

## Honest read

- **Correctness is oracle16 + 80-tok no-bail, NOT the full KV-precision-parity
  gate** — that audit is documented legacy-`infer/`-only, not yet re-ported to
  `infer-cuda` DSv4. The full parity gate is a **precondition for any default
  flip**; gated-off landing is fine on the current evidence.
- **The contig scratch is allocated unconditionally** (not alloc-gated like the
  FlashMLA/fused FP8 arenas). Cost is small at B=1 (`route_capacity×128` rows,
  ~MBs), but for consistency it should become alloc-gated in a follow-up.
- The −49.8% on the MoE slice confirms the §0 verdict: **the cost was the
  self-written padded layout, not the DeepGEMM kernel** — exactly the
  [[2026-06-05-self-written-op-perf-inventory]] meta-pattern (b).

## Rule

When a "grouped-GEMM is slow" profile points at the MoE expert path, **break the
slice into kernel vs layout before touching the kernel**: here the WGMMA GEMM was
3 ms and the self-written padded pack/unpad was 10 ms. The lever was adopting the
upstream *contiguous active-row layout* (`ep_scatter → m_indices`), which halved
the slice — the kernel needed no change.
