# DSv4 decode → 6ms: remaining levers (post +27%) — precise implementation plan

## Superseded by later evidence

**The lever ranking in this doc is a SMOKE-SHAPE artifact and was overturned the
same day.** The "comm 32.4% (AllReduce 16.4% + AllGather 16.0%)" / "GEMV 14.4%" /
"mhc 12.2%" ranking comes from an **8-token decode window** (short prompt + 64
steps, prefill diluted), where the context-scaling sparse selector is trivial. The
end-to-end **wall-clock** trace at the 4096 SLO shape found the real #1 bottleneck
is `dsv4_csa_select` (74.9% of decode at 4096; flat-vs-scaling 124ms → 26ms after
the fix) — comm is only ~4% at the SLO shape. See
[`2026-06-06-dsv4-pd-systematic-analysis.md`](2026-06-06-dsv4-pd-systematic-analysis.md)
§3 (the corrected wall-clock conclusion) and the retro
[`../experience/errors/2026-06-06-handrolled-kernels-vs-adopt-official-retro.md`](../experience/errors/2026-06-06-handrolled-kernels-vs-adopt-official-retro.md).

Correct conclusions that replaced the levers here:
- **csa_select → official DeepSeek DSA indexer** (`fp8_paged_mqa_logits` +
  `deepseek_v4_topk_transform_512`), not a hand-rolled parallel kernel. Default-on,
  decode flat ~26ms @4096:
  [`../experience/wins/2026-06-07-dsv4-official-dsa-default-on.md`](../experience/wins/2026-06-07-dsv4-official-dsa-default-on.md).
- The **H20 reference baseline** (base decode ~20-35ms; 6ms needs spec):
  [`2026-06-06-dsv4-h20-reference-baseline.md`](2026-06-06-dsv4-h20-reference-baseline.md).
- The forward-looking program is the **unified batched-decode/paged-KV abstraction**,
  not per-kernel B=1 levers:
  [`2026-06-07-unified-batched-kvpool-abstraction.md`](2026-06-07-unified-batched-kvpool-abstraction.md).

Lever 2 (residual `wo` GEMV) and lever 3 (mhc-fuse) below are subsumed by the
"adopt the official kernel" posture (the fp8 GEMVs go to DeepGEMM where it wins;
mhc-fuse was blocked on TileLang f32-mma — see
[`../experience/errors/2026-06-06-dsv4-mhc-fuse-tilelang-f32-mma-blocked.md`](../experience/errors/2026-06-06-dsv4-mhc-fuse-tilelang-f32-mma-blocked.md)).
Lever 4 (EAGLE/MTP via 6ms) is correct in principle but the "1.9× now" framing was
overturned — MTP is parked at the draft-quality wall (39% accept vs SGLang 68%):
[`../experience/errors/2026-06-06-dsv4-mtp-perf-acceptance-workload-blockers.md`](../experience/errors/2026-06-06-dsv4-mtp-perf-acceptance-workload-blockers.md).
Kept for history (the profiling method + the comm-overlap decomposition are valid
process records).

---

**Date:** 2026-06-06. **State:** decode **29.5 → 37.6 tok/s (~26.6 ms/token)** this
session via env-A/B-licensed flips (gpu-router on-device, FlashMLA-decode +24%,
fused-wqkv +18.4%) + D2D/memset cleanups. The cheap env-A/B wins are **exhausted** —
every remaining lever is substantial implementation. Levers are ranked by the **clean
decode-window profile** (8-tok prompt + 64 steps, prefill diluted; fused-wqkv ON):

| # | Kernel bucket | %decode | Lever |
|---|---|---|---|
| 1 | `ncclAllReduce` 16.4% + `ncclAllGather` 16.0% | **32.4%** | comm overlap / custom all-reduce |
| 2 | `dsv4_fp8_gemv_batch_kernel` | 14.4% | fuse `wo`/compressor (residual scalar GEMV) |
| 3 | `dsv4_mhc_params` | 12.2% | mhc-fuse (parked f32-mma) |
| — | FlashMLA fwd, KV pack | small | already optimal (vendored) |

## Lever 1 decomposition (UPDATE 2026-06-06 — source-verified, §0 confounder)

The "comm 32.4%" is **not one lever** — and it is profiled on the **allreduce MoE
backend** (`dsv4_use_deepep_transport` defaults to `allreduce`, dsv4.rs:1898; the
only debug-runnable lane). Breakdown of where the two NCCL kernels come from:

| NCCL kernel | %decode | source | nature |
|---|---|---|---|
| `ncclAllReduce` | 16.4% | MoE all-reduce (`needs_moe_allreduce = !use_deepep`, dsv4.rs:1125) **+** TP attention all-reduce | the MoE half is a **backend artifact** — native-DeepEP replaces it with dispatch/combine (different cost). Re-profile on `ARLE_DSV4_MOE_TRANSPORT=deepep` before sizing this. |
| `ncclAllGather` | 16.0% | **FlashMLA decode Q all-gather** (attention.rs:2572-2607, `dsv4/flashmla_q_allgather` + `dsv4_tp_q_repack`) | ARLE shards Q heads across TP but MLA's KV latent is replicated, so it all-gathers local Q → full Q for FlashMLA. **Open: does SGLang's MLA-TP / DP-attention avoid this?** Concrete, NVTX-marked, possibly-avoidable 16%. |

**Consequence for prioritization:** do NOT license a generic "RING→one-shot
all-reduce" lever off the 32.4% number. First (a) re-profile decode on native-DeepEP
(`ARLE_DSV4_MOE_TRANSPORT=deepep`) to get the production comm cost — the MoE-AR half
likely shrinks; then (b) the **FlashMLA Q all-gather (16%)** is the largest single
*concrete* comm cost and is an operator-layout question vs SGLang (DP-attention?),
investigate it on its own. The §5.1 multi-stream overlap below still applies to the
attention-prepare chain regardless.

## Lever 1 — comm overlap (32.4%, the dominant lever) — DELICATE, focused effort

**Why it's not a quick edit:** `dsv4_shared_expert_forward` (`moe.rs:1725`) is
hard-coded to `ctx.stream` in 5+ launches (`moe.rs:1902/2034/2161/2265/2421` + clone/
memcpy). Infra exists but is **unused in the forward**: `comm_stream`
(`tensor.rs:183`, "collectives that can overlap independent compute") + `CudaPipelineFence`
(`tensor.rs:216`).

**Plan (two commits, token-exact gate each):**
1. **Stream-parameterize** `dsv4_shared_expert_forward` + its callees: add a
   `stream: &CudaStream` param, thread it to every `cu_stream()`/clone/memcpy. Default
   the caller to `ctx.stream` → **zero behavior change** (token-exact must hold). Do NOT
   commit this alone (half-state) — land it *with* step 2.
2. **Fence-orchestrate** in `dsv4.rs` (~`928` moe AR / `939` shared expert): record a
   fence on the compute stream after `normed` is produced; have `comm_stream` wait it;
   run `dsv4_shared_expert_forward(..., comm_stream)`; record a fence on `comm_stream`
   after `shared`; run the moe `all_reduce_sum` on the compute stream **concurrently**;
   before `add_batch` (`dsv4.rs:958`), make the compute stream wait the shared-fence.
   The shared expert reads `normed` (not the AR output) so it is provably dep-free.
   `feedback_private_stream_needs_stream_wait` — the comm_stream MUST wait the compute
   stream's `normed` producer or it reads stale data.
**Risk:** a wrong fence = an intermittent race that token-exact can pass by luck →
verify with a **needle + multi-run** at the deterministic shape, not one token-exact.
**Alt (lever 1b):** replace NCCL RING all-reduce with a custom one-shot/two-shot
graph-capturable all-reduce for the small B=1 decode tensors (SGLang's GroupCoordinator
approach; decode-perf-roadmap step 4) — kills the AllReduce 16.4% AND unblocks the full
decode graph.

## Lever 2 — residual GEMV fusion (14.4%)

fused-wqkv (`#9`) fused `wq_a|wkv_a`. The residual `dsv4_fp8_gemv_batch_kernel` is the
`wo` o-projection + the compressor `wkv`/`wgate` GEMVs (still scalar). Mirror the
fused-wqkv DeepGEMM path for `wo` (and the compressor if shapes allow). Gate +
env-A/B-license like #9. Lower risk than lever 1 (same pattern, proven).

## Lever 3 — HC mhc-fuse (12.2%)

The parked `mhc_pre_big_fuse` TileLang kernel — now evidence-backed (12.2%, not
"uncertain"). The blocker fix is known (`errors/2026-06-06-dsv4-mhc-fuse-tilelang-f32-mma-blocked.md`):
route the `norm_fn_kernel.py:107` `T.gemm(f32,f32,f32)` through the **bf16 `x_smem_16`**
(already allocated) or tf32, instead of the unsupported f32 mma. Then wire FFI + the 3
call-sites. Needs a pod CUDA build + needle precision check on the HC mix.

## Lever 4 — EAGLE/MTP Phase 2 (~1.9× amortization — the multiplier)

Phase 1 landed (`2e0cde16`, MTP head loads + drafts). Phase 2 = the verify loop +
KV rollback (`truncate_slot` exists, `lib.rs:258`) + the 2 control experiments, spec'd
in `2026-06-06-dsv4-wholesale-kernel-adopt.md` §3. Independent of the per-kernel work —
amortizes the whole forward. The single biggest step toward 6ms-effective
(26.6 / 1.9 ≈ 14 ms; with lever 1 → ~10 ms).

## Prefill axis (separate) — clean profile 2026-06-06 (4096-tok, GPU activity)

Prefill is **attention+prepare bound, NOT MoE-bound** (DeepGEMM ~2.3%). Breakdown:

| Stage | %prefill GPU | Lever |
|---|---|---|
| scalar `dsv4_hybrid_attention_kernel` (CSA/HCA math) | 23.6% | **hard** — FlashMLA-prefill killed (+36%; its prepare overhead > the attention-math savings) |
| FP8 GEMV projections | 22.8% | **#1: extend fused-wqkv (DeepGEMM) to prefill** — the proven +18.4% decode pattern, lifted to multi-token (review #9 "extend to prefill") |
| `dsv4_csa_select_kernel` (bitonic top-512) | 17.7% | #2: fused top-k (SGLang Indexer) / overlap |
| compressor update | 4.2% | prepare-chain overlap (only if it hides csa_select+compressor+GEMV behind attention) |

**Plan:**
1. **Extend fused-wqkv to prefill** (#1, 22.8%, lowest-risk — mirrors the proven decode
   fused path): remove the `token_count==1` restriction in `run_fused_wqkv_decode`
   (attention.rs:~2296), make the fused `wqkv_a` DeepGEMM handle multi-token shapes.
   Gate + token-exact + prefill_ms A/B like the decode flip.
2. **CSA-select** (#2, 17.7%): the scalar bitonic `dsv4_csa_bitonic_sort_desc` →
   adopt a fused top-k; SGLang also reuses the index cross-layer (`skip_topk`).
3. **prepare-chain overlap** (compressor/indexer on `comm_stream` behind the attention/
   GEMV) — same fence infra as decode lever 1; licensed *only* if it overlaps
   csa_select+compressor+GEMV with the attention path (the prepare chain is serial today).
- NVTX wall shows `dsv4/attn_allreduce` 17.9% but that range absorbs async backlog/skew —
  use the GPU-activity column for attribution, not the NVTX wall (§0 framing).

## Method (proven this session — do not regress)
path-probe to kill profile confounders → **clean** decode-window profile (short prompt
+ many steps) → attack the true #1 kernel → **same-binary env-A/B** to license → flip.
Verify wall-clock on the same harness; stage_profile/linear_profile OFF (they sync per
stage); gpu-router ON. B=1 decode is GPU/comm-bound — overhead-removal (alloc/launch) is
wash ([[feedback_b1_decode_gpu_bound_overhead_removal_wash]]).
