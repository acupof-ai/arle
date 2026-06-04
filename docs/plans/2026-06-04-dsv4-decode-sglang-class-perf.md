# DSv4 decode → SGLang-class (5–6 ms/token) perf roadmap

**Date:** 2026-06-04. **Target:** DSv4-Flash TP=8/EP=8 decode at SGLang-class
latency (~5–6 ms/token) via full CUDA-graph capture + on-device routing, on 8×H20.
**Status:** roadmap; gated behind the incremental-decode correctness fix (#23).

## 1. Why we're far from 5–6 ms today

DSv4 decode is currently (a) **broken** (incremental-decode `start_pos>0` diverges from
prefill — #23) and (b) architecturally **launch-bound + host-round-trip-bound + eager**:

| Gap | Evidence (our code) | Cost |
|---|---|---|
| Static expert weights rebuilt every layer/step | ~~`moe.rs` `build_grouped_cache` per call~~ **FIXED** `d5115de5` | was ~529 ms/token |
| MoE routing on the **host** every MoE layer every step | `moe.rs:181/512/708` `clone_dtoh(&logits.data)` → `infer_moe::route`; comment `moe.rs:20`: *"a device router is a perf follow-up"* | per-layer H2D/D2H round-trip + host compute, **un-capturable** |
| TP all-reduce stays **eager** under TP | `collective.rs:406` `supports_graph_capture()` → `false`; `executor.rs:408` "NCCL not graph-capturable" | ~250–400 `cuLaunchKernel`/token, no graph under TP |
| No full decode-graph for DSv4 | `executor.rs:563` "decode graph disabled (MLA host-routing per step)" | per-kernel launch overhead |

## 2. SGLang / DeepEP precedents (how 5–6 ms is reached)

- **NCCL itself need not be graph-capturable.** SGLang captures the full TP decode via
  either (a) a **custom graph-capturable all-reduce** (one-shot/two-shot single kernel,
  used for the small decode tensors) in its `GroupCoordinator`, or (b) **"breakable CUDA
  graph"** — capture the graphable segments between all-reduces, re-invoke the collective
  (with the same tensor refs) at the break points. So our `supports_graph_capture=false`
  is a self-imposed limit, not hardware.
  ([SGLang breakable cuda graph](https://sgl-project.github.io/advanced_features/breakable_cuda_graph.html),
  [SGLang cuda-graph](https://sgl-project-sglang-93.mintlify.app/optimization/cuda-graph))
- **DeepEP low-latency decode mode** — pure-RDMA GPU dispatch/combine, **hook-based
  comm/compute overlap that occupies 0 SMs**, ~20 SMs to saturate, GPU-native group-limited
  gating (DeepSeek-V3 routing on-device), FP8 supported. This is the "GPU instructions
  reduce H2D/D2H": routing + token-shuffle run on-device, no host round-trip.
  ([DeepEP README](https://github.com/deepseek-ai/DeepEP/blob/main/README.md))

## 3. Roadmap (sequenced — correctness before perf)

0. **[DONE `d5115de5`]** Grouped-cache prebuild — static expert weights built once at load
   (~529 ms/token host rebuild eliminated). Plus the wq_b TP-shard correctness fix.
1. **[#23, Codex — PREREQ]** Fix DSv4 incremental decode (`start_pos>0`). Decode must match
   prefill before optimizing it. Hypothesis: SW ring-cache carry + RoPE position at
   `start_pos>0`. Nothing below is safe until decode is numerically correct.
2. **On-device MoE routing.** Compute top-k / group-limited gating on-GPU (the "device
   router" follow-up at `moe.rs:20`); drop the `clone_dtoh(&logits)`→`infer_moe::route`
   round-trip at `moe.rs:181/512/708`. Kills the per-layer H2D/D2H **and** makes the MoE
   step capturable. License: decode-step D2H count → 0 (nsys), no correctness change.
3. **DeepEP low-latency decode mode.** Route the EP all-to-all through DeepEP's
   decode-optimized kernels (GPU dispatch/combine, hook overlap, ~20 SM) instead of the
   host-routed `allreduce` path. Pairs with (2). License: decode tok/s A/B vs the
   allreduce path at the production shape.
4. **Graph-capturable all-reduce.** Add a custom one-shot/two-shot all-reduce kernel
   (capturable) for the decode-sized TP tensors, OR adopt breakable-graph; flip
   `supports_graph_capture`. License: captured == eager logits (greedy bit-exact).
5. **Full decode-graph capture.** Capture the whole DSv4 decode step (MLA + on-device-routed
   MoE + capturable all-reduce) into one graph; remove the `executor.rs:408` TP gate for the
   capturable path. License: ~250–400 launches/token → 1; decode tok/s A/B.

**Endpoint:** launch-overhead-free + host-round-trip-free + comm/compute-overlapped decode
→ SGLang-class 5–6 ms/token. Each step is its own commit + bench entry (per the bench
mandate); steps 2–5 are individually license-or-kill at the production decode shape (not a
smoke shape — distilled lesson: SLO verdict from the SLO workload).

## 4. Note

Steps 2–5 are a multi-commit architecture arc, not a single fix. The hard prerequisite is
#23 (decode correctness); the biggest single launch-reduction is step 5 (full graph), and
the biggest host-overhead kill is step 2 (on-device routing). Sources are upstream
precedent (hypothesis-grade per the upstream-scan skill); each lands only on a local
H20 A/B at the production shape.

## 5. Measured 卡点 — host-route baseline decode profile (2026-06-05)

Full-chain nsys profile, per-stage NVTX (`infer-cuda/src/nvtx.rs`), DSv4-Flash FP8
TP=8/EP=8 on 8×H20, decode steady-state (16/16 correct, `clean_tokens` ==
oracle), **236.8 ms/token** (4.223 tok/s under nsys). This is the ground-truth
that re-prioritizes the levers above (replaces the earlier API-level estimate).

| Stage (NVTX) | ms/token | wall % | Dominated by (API) |
|---|---|---|---|
| `moe_route` | 98.1 | **41.4%** | `cuStreamSynchronize` 93.7 (39.6%) — host-route D2H+sync → **#24** |
| `deepgemm_grouped` | 92.0 | **38.8%** | `cuMemAllocAsync` 87.4 (36.9%) — per-step grouped-GEMM allocs → **#29** |
| `mla_attn` | 11.9 | 5.0% | real kernel (FlashMLA path) |
| `shared_hc` | 9.3 | 3.9% | real kernel (shared expert + hyperconnections) |
| `moe_allreduce` | 1.1 | 0.5% | NCCL |
| `combine_scatter` | 1.1 | 0.5% | DeepEP combine |
| `attn_allreduce` | 0.8 | 0.4% | NCCL |
| `lm_head_sample` | 0.8 | 0.3% | |

API cross-check (per token): `cuStreamSynchronize` 93.7 ms (39.6%) +
`cuMemAllocAsync` 87.4 ms (36.9%) = **~76% host overhead**; `cudaLaunchKernel`
7.8 ms (3.3%); `cuMemcpyDtoHAsync` 1.2 ms (0.5%). Real kernel floor (MLA + HC +
GEMM compute + launches) ≈ 15–20%.

**Re-prioritized levers (quantified):**
- **#24 on-device routing** kills `moe_route`'s 41.4% (the `cuStreamSynchronize`).
  Route kernel already licensed (0-diff); async wiring blocked on #29.
- **#29 persistent scratch pool** kills `deepgemm_grouped`'s 38.8% (the
  `cuMemAllocAsync`) — *and* is the async-router correctness fix: the garbage
  under sync-free was localized (post-MoE keepalive / keepalive-all / same-stream
  event all FAIL) to a `cudaMallocAsync`/free/raw-pointer **reuse-boundary**, not
  buffer lifetime. Fixed device addresses remove both the alloc cost and the
  race. Scoped to decode/B=1 (prefill stays on-demand — avoid 512/2048 resident
  workspace peak).
- #24 + #29 together attack **~80% of decode wall-clock** → 236.8 ms/token should
  drop toward the ~40–50 ms kernel floor. **#25 full decode graph** then attacks
  the residual launch overhead. Whether 5–6 ms is physically reachable on H20
  (vs H100/H800) is a kernel-floor question answered *after* host+launch overhead
  is removed — license-or-kill on wall-clock, not asserted.

Artifacts (pod): `dsv4_hostroute_decode2.{nsys-rep,log}`. The NVTX instrumentation
commits with the #29 patch; the before/after profile pair lands as the #29 wins
entry.

## 6. After #24+#29 — decode 4.365 → 21.615 tok/s (~5×), re-profiled (2026-06-05)

#24 (on-device router) + #29 (persistent `Dsv4MoeDecodeScratch`) landed
(`4d62062a`, wins
[`2026-06-05-dsv4-decode-scratch-pool-5x.md`](../experience/wins/2026-06-05-dsv4-decode-scratch-pool-5x.md)).
Decode **21.615 tok/s** (non-nsys) / 19.724 (nsys), async sync-free, 16/16. Host
overhead gone: `moe_route` 41.4%→3.7%, `deepgemm_grouped` 38.8%→4.3%,
`cuMemAllocAsync` 87.4→2.81 ms/token, route D2H → 0.018 ms/token.

**Clean #29 wall-clock breakdown (nsys, 50.70 ms/token):**

| Bucket | ms/token | % wall | Lever |
|---|---|---|---|
| `cuMemcpyDtoD` (D2D copies) | 9.03 | 17.8% | **next: buffer reuse / in-place** |
| `mla_attn` kernel | 11.79 | 23.3% | kernel floor — FlashMLA (align SGLang), last |
| launch API (`cudaLaunchKernel` 8.82 + `cuLaunchKernelEx` 2.56) | 11.39 | 22.5% | **#25 full decode graph** |
| `cuMemset` (zeroing) | 5.29 | 10.4% | **next: skip unnecessary zeroing** |
| alloc/free residual (non-MoE per-layer temps) | 5.02 | 9.9% | **next: extend scratch arena** |
| `lm_head_sample` (GEMV + argmax + D2H) | ~0.6 | ~1.2% | **ruled out** (see below) |

**Evidence-based lever ranking (replaces assumption):**
1. **[DONE `7bdb3819`] Buffer/D2D/zeroing/alloc.** Was ~19.3 ms/token (38% wall).
   Root cause: the #23 keepalive's `CudaSlice::clone()` is a device-to-device copy
   — redundant once #29's scratch pool gives stable addresses → default-off.
   D2D −86%, memset −88%, alloc+free −65%. Decode 21.615 → **25.129 tok/s**,
   wall 50.7 → **41.2 ms/token**. Wins
   [`2026-06-05-dsv4-decode-buffer-d2d-reduction.md`](../experience/wins/2026-06-05-dsv4-decode-buffer-d2d-reduction.md).
2. **[DONE `7104813e`, gated] #25 breakable decode graph.** Full 43-layer
   breakable graph (attn seg → eager TP all-reduce → MoE seg → eager all-reduce →
   tail). Launch API 13.36 → **1.68 ms/token (−87%)** — but **wall flat**
   (25.129 → 25.517 tok/s). The launch time was CPU-side overlapped with GPU
   execution, *not* critical-path (§0 framing trap). `cuStreamSynchronize` now
   waits on GPU backlog. Wins
   [`2026-06-05-dsv4-decode-breakable-graph-launch-overlapped.md`](../experience/wins/2026-06-05-dsv4-decode-breakable-graph-launch-overlapped.md).
3. **MLA attn ~11.79 ms/token + FP8 GEMV + DeepGEMM pack/unpad + MHC** — the real
   kernel floor; deepest, align SGLang FlashMLA decode.

## 7. Verdict — host+launch axis complete; decode is GPU-kernel-bound (2026-06-05)

Cumulative decode: 4.365 → **25.5 tok/s** (**~6×**), 236.8 → **~39.5 ms/token**.
#21 (grouped-cache) + #24 (on-device router) + #29 (scratch pool) + buffer/D2D
lever + #25 (graph) eliminated **all** host-route sync, per-step alloc, D2D
copies, zeroing, and launch overhead. The remaining ~39.5 ms/token is the **real
GPU kernel floor** — hybrid MLA attention (~11.8 ms), FP8 GEMV, DeepGEMM
pack/unpad, MHC params.

**The 5–6 ms/token target is an H100/H800 number** (where SGLang measures it). On
H20 (substantially lower FLOPs/bandwidth) the realistic floor for this 671B-class
FP8 MoE is **~25–40 ms/token** without deep per-kernel rewrites. Reaching the
floor further needs kernel-level work (FlashMLA decode tuning, DeepGEMM, FP8
GEMV) with diminishing returns, and would still not hit 6 ms on H20. The
structural-overhead arc is done; further decode gains are a kernel-floor /
hardware question, not a scheduling/buffer/launch one.

**`lm_head_sample` ruled out (§0 framing-trap, falsified):** its 9.4 ms/token
NVTX range *included* the step-end `sample_cuda_token → ctx.sync()`, so it was
timing the whole step's GPU backlog, not the LM head. A `sync_before_lm_head`
probe drops it 9.42→0.565 ms/token with decode tok/s unchanged
(19.724→19.534); the LM-head GEMV is ~0.4 ms, argmax 0.009 ms, D2H 0.018 ms.
(`ARLE_DSV4_FP8_GEMV_MMA=1` is also a kill — corrupts token1, slower.) **Lesson:
an NVTX range that ends in a sync absorbs the upstream backlog and reads as a
phantom 卡点 — always cross-check with a sync-before probe before chasing it.**

Artifacts (pod): `dsv4_scratch_pool_clean_profile_{api,nvtx,gpu}.txt`,
`dsv4_lmhead_presync_profile_{api,nvtx,gpu}.txt`.

## 8. Prefill profile (#28) — host-route already gone, now allreduce-wait + kernel floor (2026-06-05)

The decode work transferred: **DSv4 prefill 512 dropped 14.66s → 1.204s** (the
early "93% GPU idle from host-route" is gone — `moe_route` is now 6–11 ms total,
not the bottleneck). Profiled under `ARLE_DSV4_GPU_ROUTER=1` + NVTX, no-nsys wall
(nsys perturbs 512 heavily, used only for attribution):

| shape | wall (no-nsys) | rank0 GPU kernels | result |
|---|---|---|---|
| 512 | 1.204 s | 0.893 s | clean `[294]` |
| 2048 | 10.917 s | 3.974 s | clean `[671]` |

Per-stage NVTX (512 / 2048): MLA attn 211 / 846 ms · MoE route 5.9 / 10.8 ms
(host-route non-issue) · DeepGEMM grouped 843 / 2199 ms (mostly pack/quant/unpad
wrapper, not WGMMA) · combine 47 / 193 ms · **attn+moe allreduce 607 / 4963 ms**
(2048 dominated by a single ~4.96 s *first* attn-allreduce wait) · shared+HC
539 / 548 ms · lm_head_sample 311 / 1367 ms (end-sync framing trap, not LM head).

**Kernel-floor top (512 / 2048), shared with decode:** FP8 block-scaled GEMV
365 / 1464 ms (#1) · hybrid attention 189 / 1190 ms (#2) · DeepGEMM unpad
106 / 425 ms · DeepGEMM pack/quant 82.5 / 330 ms · compressor update 55 / 228 ms.

**Verdict + next:** the 2048 prefill bottleneck is (a) a ~4.96 s first
attn_allreduce wait — **pending-proof**: per-rank arrival instrumentation must
show whether it's true cross-rank skew, a one-time NCCL-warmup cost, or a
profiling artifact (don't chase until proven — the 4th potential framing trap);
and (b) the real kernel floor in **FP8 block-scaled GEMV + hybrid attention** —
the *same* kernels that floor decode, so per-kernel work there is a prefill+decode
double-win (align SGLang FlashMLA). Host routing is no longer a prefill lever.
