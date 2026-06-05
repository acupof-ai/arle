# ARLE self-written operator perf inventory

**Date:** 2026-06-05. **Provenance:** 5-agent read-only survey (attention / MoE /
GEMM+quant / norm+sample+misc+registry) → synthesis, cross-referencing the gated
stage profiler, ncu, the 2026-06-05 DSv4 wins/errors, and
[`kernel-registry.md`](kernel-registry.md). **Status:** evidence-disciplined — every
"slow" claim is tagged **measured** (a real ms/token / ncu SM% / matched A-B) or
**hypothesis** (source-survey smell, NOT yet profiled). Load-bearing file:symbol
refs spot-checked against source and confirmed real (dg_unpad/pack/swiglu at
`dsv4_deepgemm_ops.cu:62/117/173`, `dsv4_mhc_params_kernel<<<num_tokens,…>>>`, the
scalar `dsv4_swa/hybrid_attention` + `<<<1>>>` `dsv4_compressor_update`,
`dsv4_fp8_gemv_batch_kernel`). Don't assert as fact without re-checking source
([[feedback_docs_are_not_truth]]); §0: a "slow op" needs a number, not a launch-
count guess ([[feedback_measure_batching_before_ceiling]]).

---

## 1. Verdict

Three meta-patterns account for every measured-slow self-written op in ARLE. **(a) Tiny-grid scalar kernels at B=1** — the DSv4 MLA attention family (`dsv4_swa_attention`, `dsv4_hybrid_attention`) and the FP8 attention-linear GEMV (`dsv4_fp8_gemv_batch`) launch `<<<num_tokens*heads, BLOCK>>>` or `<<<1, BLOCK>>>` grids that ncu confirms run at 1-3% SM and <1% tensor pipe; the work belongs on WGMMA tiles. **(b) Self-written layout/dispatch glue around a *fine* vendored kernel** — the MoE DeepGEMM companion ops (`dg_unpad`/`dg_pack_quant`/`dg_swiglu_quant`, ~10.2 ms/tok) are slow not because their arithmetic is wrong but because they materialize a padded `32 groups × 128 rows` layout at B=1; the vendored WGMMA GEMMs they wrap cost only 2.99 ms. **(c) ~300 tiny per-layer launches** — the hyper-connection family (`dsv4_mhc_*`, 4.92 ms/tok) fires 17 call sites × 61 layers of single-CTA kernels. The honest split: **measured-slow = 6 op families** (3 of them already replaced/in-progress); **everything else is HYPOTHESIS** — route/scatter/combine, norm, sampling, elementwise, KV-quant — either confirmed cheap-by-construction, dormant with 0 callers, or never broken out of a stage slice. lm_head/sampling's apparent 9.4 ms is a confirmed NVTX-sync phantom, not a target.

## 2. Measured low-perf table (real number, ranked by cost)

| Op (file:symbol) | Measured cost | Evidence source | Vendored / best-practice alternative | Status |
|---|---|---|---|---|
| **MoE `dg_unpad`** `csrc/gemm/dsv4_deepgemm_ops.cu:173` `dsv4_deepgemm_unpad_grouped_bf16_kernel` | **4.50 ms/token** | plan §Progress 2026-06-05 stage profile (`stage_profile::profile`, moe.rs:1394) | SGLang contiguous decode layout (`ep_scatter→m_indices`, `use_masked_gemm=False`) — **eliminates** unpad | in-progress / gated-off (`ARLE_DSV4_MOE_CONTIG_DECODE`, default-OFF) |
| **HC / hyper-connection family** `csrc/misc/dsv4_mhc.cu` `dsv4_mhc_{params,pre,post,head_pre,expand}` | **4.92 ms/token (~9.2% decode)** | stage profiler `dsv4/stage/*_hc_*` + `shared_hc` 5.4%; `wins/2026-06-05-dsv4-decode-scratch-pool-5x.md:56` | none (DSv4-specific) — fuse 61-layer params into one launch / into decode graph | open (cost SOLID, root cause hypothesis) |
| **MoE `dg_pack_quant`** `csrc/gemm/dsv4_deepgemm_ops.cu:62` `dsv4_deepgemm_pack_quantize_bf16_to_fp8_kernel` | **3.72 ms/token** | plan §Progress stage profile | contiguous layout — quantize only `num_tokens×topk` active rows | in-progress / gated-off (same gate) |
| **MoE `dg_swiglu_quant`** `csrc/gemm/dsv4_deepgemm_ops.cu:117` `dsv4_deepgemm_swiglu_quantize_w13_kernel` | **2.03 ms/token** | plan §Progress stage profile | contiguous layout — operate only on active rows | in-progress / gated-off (same gate) |
| **Scalar MLA attn** `csrc/misc/dsv4_attention.cu:539/646/1138/1297` `dsv4_swa_attention_cuda` + `dsv4_hybrid_attention_cuda` | part of `mla_attn` **11.8 ms/tok (15.9% decode)**; ncu **SM 1-3%**; replacement = **+18.03% decode** A/B | `wins/2026-06-05-dsv4-resident-ab-flashmla-decode-18pct.md` (23.713→27.988 tok/s); `wins/2026-06-05-dsv4-flashmla-decode-wireup.md` ncu | **vendored FlashMLA** `arle_flashmla_sm90_sparse_decode_fwd` (fused SW+compressed, 64-head WGMMA, 132-CTA persistent) | in-progress / gated-off (`ARLE_DSV4_FLASHMLA_DECODE`; default-flip blocked on occupancy-SOL ncu precond-fail ×2) |
| **Scalar FP8 attention-linear GEMV** `csrc/gemm/quantized_gemv.cu:374` `dsv4_fp8_gemv_batch_kernel` | ncu **scalar CUDA-core, <1% tensor, ~10% HBM BW**; ~344 launches/forward; routing around it = **+18% decode**, fusing = **+5.07%** | `errors/2026-06-05-fp8-linear-per-projection-deepgemm-no-win.md`; `wins/2026-06-05-dsv4-fp8-fused-wqkv-a-decode-5pct.md` | vendored **DeepGEMM `fp8_gemm_nt` in fused call form** (per-projection swap KILLED 0.8%; `wq_a+wkv→wqkv_a` fusion WON +5.07%) | in-progress (fused gated `ARLE_DSV4_FUSED_WQKV_DECODE=1`; scalar still default for prefill + non-fused projections) |
| **TP Q all-gather** `attention.rs:1424` `tp.all_gather_bf16_raw` | **~0.61 ms/token** (NVTX) | `wins/2026-06-05` resident-ab closeout `dsv4_flashmla_wrapper_nvtx_stats` | SGLang DP-attention (`--enable-dp-attention`) removes attn all-gather (endgame lever #4, unwired) | in-progress (FlashMLA-path glue; trim deferred) |
| **FP8-KV pack** `csrc/attention/dsv4_fp8_kv_pack.cu` `arle_dsv4_fp8_kv_pack_*` | **~0.56 ms/token** (NVTX) | same resident-ab closeout | none — hard prerequisite for FlashMLA's FP8-only path | in-progress (required, not removable; cost of admission) |
| **TP Q repack** `csrc/misc/dsv4_tp_attention_repack.cu:30` `dsv4_tp_q_repack_cuda` | **~0.23 ms/token** (NVTX) | same closeout | removed if DP-attn lands (no gather → no repack) | in-progress (glue; trim deferred) |
| **Scalar MLA attn (prefill)** `csrc/misc/dsv4_attention.cu` (per-token scalar grid) | **15–22× slower** than FlashMLA prefill (282s→13s @29K) | `wins/2026-05-27-dsv4-flashmla-v2-22x-prefill...md` | vendored `arle_flashmla_sm90_sparse_prefill_fwd` | gated-off (FlashMLA prefill crashes >24K TMA-OOB; SLO-default-on KILLED `errors/2026-05-31-dsv4-flashmla-default-slo-kill.md`) |
| **Scalar grouped MoE GEMM** `csrc/gemm/moe_grouped_gemm.cu:50/157` `moe_bf16_grouped_gemm_batch_kernel` | prefill **+13% over M-blind baseline but 0.13% of roofline** (818/624000 tok/s); deepgemm-vs-scalar: scalar −9.5% @545tok but deepgemm **times out @1089** | `errors/2026-05-27-dsv4-grouped-gemm-marginal.md`; `wins/2026-05-31-bench-dsv4-deepgemm-vs-scalar-prefill.md` | DeepGEMM masked grouped (worse scaling cliff as-is) — real lever is layout, not kernel | open (correct + helpful, NOT the dominant blocker; ~60× SLO gap remains) |

## 3. Hypothesis table (source-survey smells — NOT yet proven slow)

| Op (file:symbol) | Smell | Why suspect | Cheapest experiment |
|---|---|---|---|
| **Compressor KV update** `csrc/misc/dsv4_attention.cu:818/995` `dsv4_compressor_update_cuda` | grid `<<<1, BLOCK>>>` — literally one block | ugliest tiny-grid in the attention lane; one CTA does the whole compressor stream update at B=1 | ncu occupancy on one launch + isolate from `mla_attn` via sync-before probe |
| **CSA top-k selector** `csrc/misc/dsv4_attention.cu:1546/1687` `dsv4_csa_select_cuda` | grid `<<<num_tokens, BLOCK>>>` = 1 block at B=1 | scalar block-score reduction; feeds FlashMLA indices (kept either way) | stage-profile slice within `mla_attn` |
| **RoPE Q/K prep** `csrc/misc/dsv4_attention.cu:367-538` `dsv4_prepare_qk_cuda` (+`_fused`) | per-token RoPE prep, padded 576 layout | live in both paths; only ever launch-count-fused, never profiled | sync-before probe / ncu duration |
| **MHC root cause** `csrc/misc/dsv4_mhc.cu` `dsv4_mhc_params_kernel` `<<<1,256>>>` thread0-only Sinkhorn | single-CTA + thread0 serial loop + ~300 launches/tok | cost SOLID (4.92 ms) but cause undetermined: launch-overhead vs occupancy vs f32 materialization | ncu occupancy on `mhc_params`/`head_pre` + per-call-site launch-count probe (DSV4_MHC_BLOCK=512 already KILLED) |
| **MoE route** `csrc/moe/dsv4_route.cu:329` `dsv4_route_cuda` | top-k reduction over `[1, n_experts]` at B=1 | `moe_route` stage instrumented but per-stage ms never cited | break out `moe_route` stage-profile slice |
| **MoE pack scans/casts** `csrc/moe/dsv4_route.cu:502/539/1590` `count_local_experts`/`exclusive_scan`/`cast_i32_i64` | per-token launch of trivial scans/casts | classic tiny-grid B=1 launch-overhead smell; folded into `moe_pack` total | break out `moe_pack` sub-launches in stage profile |
| **MoE scatter/combine** `csrc/moe/dsv4_route.cu:1318/1392` `scatter_all_route_slots`/`combine_route_slot_outputs` | per-row scatter + topk sum at B=1 | inside `moe_combine_scatter`; individual ms never cited | break out `moe_combine_scatter` slice |
| **RMSNorm** `csrc/misc/norm.cu` `rms_norm_cuda` (decode `<<<1,256>>>`) | single-CTA at B=1 | inherent to op shape (not a bug); well-tuned warp-reduce | unlikely worth measuring — only if a per-layer overhead hunt needs it |
| **BF16 GEMV** `csrc/gemm/gemv.cu:45` `gemv_handwritten_kernel` | classic self-written B=1 GEMV vs cuBLAS smell | dense-GEMM sibling proved cuBLAS near-optimal; GEMV may be bandwidth-bound (cuBLAS may not beat it) | A/B vs `gemm_cublaslt_impl` gemv at M=1 |

These are **hypothesis — unmeasured**. None has an isolated ms/token or ncu number; they are source-survey smells only.

## 4. Already handled (measured-slow AND replaced/in-progress)

- **3 scalar MLA attn kernels → vendored FlashMLA decode** — +18.03% same-load A/B proven; gated behind `ARLE_DSV4_FLASHMLA_DECODE`, default-flip blocked only on occupancy-SOL ncu (precond-fail ×2) and FP8-vs-bf16 KV-parity gate (DIFF@122).
- **Padded MoE decode layout → SGLang contiguous layout** — `dsv4_pack_local_experts_with_slots_and_indices` + `dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous` materialize only active rows and eliminate `dg_unpad`; gated `ARLE_DSV4_MOE_CONTIG_DECODE`, default-OFF, no matched A/B landed yet.
- **Per-projection scalar FP8 GEMV → fused wqkv_a DeepGEMM call** — quantize-once + fused `wq_a+wkv→wqkv_a` won +5.07%; gated `ARLE_DSV4_FUSED_WQKV_DECODE=1`. (Note: per-projection DeepGEMM swap was KILLED at 0.8% — fusion was the lever, not the kernel swap.)

## 5. Genuine gaps (no vendored/best-practice alternative — custom op justified)

- ~~**Hyper-connection family** — DSv4-specific; no upstream HC kernel exists.~~
  **CORRECTED 2026-06-05: this was WRONG — HC is NOT a gap, it's adopt-able.**
  SGLang's DeepSeek-V4 fuses the entire `mhc_pre` (RMSNorm + Sinkhorn-Knopp +
  residual-mix) into ONE TileLang kernel — `mhc_pre_big_fuse_tilelang`, PDL
  enabled — so it has no launch storm. ARLE's **86 launches/token + single-CTA
  thread0-Sinkhorn** (`dsv4_mhc_params_kernel<<<num_tokens,256>>>`) is the
  ARLE-specific anti-pattern (meta-pattern (a)+(c)), not an inherent HC cost.
  Root cause (nsys, 2026-06-05): `dsv4_mhc_params` 86 launch/tok = 3.05 ms +
  `mix_fn` scalar GEMV ~2.16 ms; NOT f32-materialize. **Lever (in-progress):
  adopt SGLang's fused `mhc_pre` TileLang structure** (ARLE already runs TileLang
  AOT), Sinkhorn parallelized across warps/CTAs, optionally PDL. (mHC-lite,
  [arXiv 2601.05732](https://arxiv.org/pdf/2601.05732), cuts Sinkhorn iters —
  numeric change, parity-gated, deferred.)
- **Compressor KV update** (`dsv4_compressor_update_cuda`, `<<<1,BLOCK>>>`) — SGLang's compressed-KV stream maintenance has no drop-in replacement; the `<<<1>>>` grid is the worst smell in the attention lane, but cost is unproven — measure before licensing a rewrite.
- **FP8-KV pack + output inverse-RoPE + FlashMLA build-indices** — required by the FlashMLA FP8-only ABI; genuine glue gaps (~0.56 ms + smaller), removable only by changing the kernel ABI, not by adopting an upstream op.
- **MHC params / Sinkhorn** — DSv4-specific normalization; no vendored equivalent.

**Next measurement priority:** (1) ~~ncu-isolate the HC Sinkhorn~~ **DONE
2026-06-05** — root-caused to 86 launch/tok + single-CTA (not f32); now adopting
SGLang's fused `mhc_pre_big_fuse_tilelang` (in-progress); (2) break out
`moe_route`/`moe_pack`/`moe_combine_scatter` stage slices to confirm or dismiss
the route-family tiny-grid hypotheses; (3) sync-before probe on
`dsv4_compressor_update` (`<<<1>>>`) before any rewrite license.

---

Excluded from "slow self-written" (vendored or library-backed — do not attribute their cost to ARLE code): dense BF16 GEMM `gemm_cuda` (**cuBLAS/cublasLt**, confirmed near-optimal `wins/2026-05-07-m_pf-gemm-phase0-kill`); FP8 `fp8_gemm_nt` + masked grouped GEMM (**vendored DeepGEMM** WGMMA, only 2.99 ms/tok); FlashMLA fwd shim (thin wrapper over vendored kernel, the +18% *win* path); ported upstream Marlin W4/W4A8. Dead/dormant (prune candidates, not perf targets, all 0 live callers per `docs/reviews/kernel-registry.md`): `csrc/attention/{mla_decode,fused_attention,prefill_attention,decode_attention_quantized,decode_attention_varlen_fp8,decode_attention_turboquant}.cu`, `csrc/kv/kv_quant.cu` family, `csrc/misc/{fused_mlp,split_qkv,conv1d_*,gdr_*}.cu`, TurboQuant family, FP4 batch GEMV. (Registry drift caught: `arle_bf16_to_f32_cuda` has 1 live caller, not dead.)