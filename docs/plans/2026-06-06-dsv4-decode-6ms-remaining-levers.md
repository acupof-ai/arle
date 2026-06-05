# DSv4 decode → 6ms: remaining levers (post +27%) — precise implementation plan

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
