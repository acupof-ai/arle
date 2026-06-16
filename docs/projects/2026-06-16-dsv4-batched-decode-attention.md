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

## R1 measured (2026-06-16, decode-phase probe, spec-off, short-prompt+256-decode, c=64→n=22)
| stage | n=5 | n=22 | scaling | verdict |
|---|---|---|---|---|
| csa_attn (else per-row `mla_attention`, ~21 CSA layers) | 28ms | **119.6ms** | ~linear 5.5ms/row | **#1 — batch this** |
| sw_attn (batched lane prepare+fwd+finish, ~20 SW layers) | 17ms | 67ms | ~linear 3ms/row | #2 (prepare/finish per-row) |
| moe | 30ms | 47ms | **sub-linear** (1.57×/4.4×) | OK, amortizes — not the target |
Attention = ~80% of step, ~linear in n; MoE amortizes. Also: decode batch capped ~n=22 at c=64 (scheduler/stagger; secondary).

## R2 plan (CSA → existing batched lane, NOT a from-scratch kernel)
The batched-index infra for CSA already exists but is orphaned: `build_indices_batched(selected_ptr)` (attention.rs) takes the CSA per-row topk `selected`; `decode_lane_fwd` is the batched sparse attention. The per-row `mla_attention_prepare` already produces `Dsv4MlaPrepared.selected` for CSA. Increment: drop the `layer.mode != CompressedSparse` gate (dsv4.rs:1994), feed each row's `selected` into `build_indices_batched`, route CSA's compressed-cache pack/read through the batched lane. Per-row prepare (indexer topk) + finish stay per-row for now; the sparse-attention KERNEL batches → expect csa_attn ~119→~70ms (matches SW). R3 = batch the per-row prepare projection GEMMs (m=1→m=N) for all layers.

## R2 done + license-or-kill (2026-06-16): CORRECT but +7%, hypothesis corrected
Wired CSA into the batched lane (gate + want_compressed + per-row `selected`→`build_indices_batched`; SW/HCA byte-identical; `decode_lane_fwd`/pack were CSA-ready — pure wiring, no new kernel). **needle 8/8 exact** (correct). But phase timing: csa_attn 119.6→0, sw_attn 67→**173** (absorbed CSA) → total attention 186.6→173.4ms = **−7% only**; c=8 throughput 66.3→65.7 (flat).
**Adversarial conclusion: the attention KERNEL was NOT the bottleneck — the per-row PREPARE is.** The 173ms is per-row `mla_attention_prepare` (wq_b/wkv_b projection GEMMs @ m=1 × 43 layers × N rows + indexer/compressor) + per-row finish, which R2 didn't touch.
**Also: R2 only batches the NON-spec path (`forward_decode_batch_stream_impl`); production is spec-on (MTP verify path) where CSA is still per-row** → R2 is production-inert until the verify path is batched too.

## Remaining levers (measured, multi-week — each a build+pod+needle cycle)
1. **Batch the per-row prepare projection GEMMs** (wq_b/wkv_b m=1→m=N, weight-read amortized ×N — the canonical decode win, like the lm_head batch). The dominant 173ms. Applies to ALL layers + both paths.
2. **Apply the batched-attention + batched-prepare to the MTP verify path** (production is spec-on).
3. **Lift the decode-batch cap** (n≈22 at c=64 — scheduler/stagger; needed to reach n~97 for SGLang parity).
4. (later) batch the per-slot compressor/indexer scatter; per-row finish.
R2 (CSA in batched lane) is the correct foundation for #1 (CSA prepare batches with the rest). SGLang 1297@c97 needs all of 1-3.

## R3 + verified bottleneck (2026-06-16): it's the DSA indexer/compressor, NOT GEMMs
Adversarial loop corrected the hypothesis FOUR times (each measured): MTP (helps, R1) → attention kernel (+7%, R2) → projection GEMMs (R3). R3 batched the prepare projections (wqkv_a+wq_b) two ways: hand-GEMV (prep 137→111) and DeepGEMM via the shared prefill_linear (prep 111, IDENTICAL). DeepGEMM source (`sm90_fp8_gemm_1d2d.cuh`) confirms M≤BLOCK_M ⇒ weight read once (vs hand-GEMV's N× reads), so DeepGEMM is the correct primitive — but it didn't move prep because the projections aren't the cost.
**Bit-exact split @ n=22: prep=103ms = [proj=4.0ms (DeepGEMM batched, negligible) + compidx=98.9ms], fwd=3.2, finish=43, moe=47.** The 137→111 drop was the RoPE batch, not the GEMM.
**THE bottleneck = per-row compressor + CSA lightning-indexer top-k (`mla_attention_prepare_compressed_only`) = 99ms = 66% of step.** Per-slot DSA state (each slot's compressed cache + index_topk=512 selection), the hardest to batch.

## True lever (deepest #60 core, multi-week)
Batch the DSA across N slots: (a) batched compressor (compress N rows' KV → N per-slot caches in one grouped op), (b) batched lightning indexer (indexer scores for N rows + batched top-k over per-slot compressed caches). SGLang's DSA decode has this; we have the per-row version. This is the genuine throughput unlock + the hardest correctness (per-slot DSA state, EAGLE-rollback-class).
Foundations landed: R2 (CSA in batched lane, attention kernel batched), R3 (batched DeepGEMM projection pre-pass + RoPE batch, +~12% from RoPE) — both correct (needle 8/8), the prepass structure is where the batched compressor/indexer plugs in. Also still: finish (43ms) batch + the MTP verify path + n≈22 cap.
