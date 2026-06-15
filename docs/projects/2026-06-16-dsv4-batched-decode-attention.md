# DSv4 batched decode attention (#60) — the high-concurrency throughput lever

**Status:** scoped, not started · **Prereq:** VRAM per-slot 40× DONE
([wins](../experience/wins/2026-06-16-dsv4-vram-per-slot-40x-state-scratch-split.md),
commit `964a010b`) · **SKU:** 8×H20 TP=8/EP=8.

## Problem (adversarially root-caused 2026-06-16)
After the VRAM fix unlocked 128 concurrent slots, aggregate decode throughput
**plateaus at ~100 tok/s** (c=8→64: 85→96→105, i.e. 1.2× for 8× concurrency;
per-req collapses 11.7→2.0 tok/s). That is **12–25× below SGLang's same-model
4×H20-3e = 1297 tok/s @ c≈97** (sgl-project/sglang#23896).

**Root cause (measured + code, not inferred): the decode attention is per-row.**
Decode step-time grows ~linearly with batch — measured **85 ms @ 8 rows → 610 ms
@ 64 rows** (≈9.5 ms/row). A properly batched decode would run N rows in ≈ one
row's time (~50 ms) → ~1280 tok/s — exactly SGLang's order.

What is per-row (`crates/infer-cuda/src/dsv4.rs forward_decode_batch_stream_impl`
+ `attention.rs`):
- **CSA layers (cr=4, ~21 of 43 layers) are forced per-row** — `attention.rs:79`
  `batched_attn_lane && layer.mode != CompressedSparse`. CSA's per-row top-k
  `selected` (index_topk=512) keeps the whole layer on the per-row `mla_attention`
  loop.
- **SW layers (cr=128, ~20 layers)** batch only the core sparse-decode kernel
  (`dsv4_flashmla_decode_batched_enabled`, default ON since 2026-06-15); their
  **prepare (wq/wkv+RoPE, HCA compressor) / pack-KV / finish / o-proj stay
  per-row** (the `for r in 0..n` loops).
- `compress_ratios = [0,0, 4,128,4,128,…,4,0]` (43 layers): ~21 CSA + ~20 SW + 3
  dense (cr=0).

## What is NOT the cause (killed)
- **MTP** — spec-on 85 > spec-off 66 tok/s @c=8 (+29%); MTP helps, keep it. The
  `[dsv4-mtp-batched] accepted=0` per-step lines are noise (cumulative accept ~56%).
- **VRAM** — fixed; 128 slots boot with 44.5 GB free.
- **DP-attention** — NOT required first. Continuous batching + chunked prefill
  (`chunked_prefill_size=64`) are already on; the gap is the attention KERNEL, not
  the batching framework. DP-attn (#89) only removes the per-token TP collectives —
  a later, smaller, orthogonal piece.

## Fix (phased)
1. **Batch the CSA decode attention** (the ~21 cr=4 layers, the biggest per-row
   chunk): batched indexer top-k over N rows + block-diagonal / per-row-selected
   sparse attention across N rows in one kernel. Hardest + highest value.
2. **Batch the SW per-row tails** (prepare/pack/finish/o-proj over [N] for the ~20
   cr=128 layers; the core is already batched).
3. Re-measure c-sweep each phase; license-or-kill on aggregate tok/s + needle.

Plan detail: SGLang `/sgl-workspace/sglang` batched-MLA-decode is the reference.
Existing assets: `Dsv4FlashMlaDecodeBatchScratch` (batched core scratch),
`Dsv4FlashMlaDecodeScratch` (shared single-row scratch, this round).

## Acceptance
c=64 aggregate decode tok/s ≥ ~5× the current ~105 (license-or-kill per phase),
needle exact at c≥32, no per-slot VRAM regression (ledger residual stays ~0).

## Validation lane
node 192.168.12.61 (8×H20, isolated from eic-test): build on .62 (1.95.0
toolchain), nc-transfer binary to .61, serve + `scripts/dsv4_concurrent_probe.py`
c-sweep + needle. Stage attribution: the per-step-vs-batch slope is the metric
(`ARLE_DSV4_MTP_STEP_PROFILE` verify-ms-vs-n, or nsys rank-0 short window).
