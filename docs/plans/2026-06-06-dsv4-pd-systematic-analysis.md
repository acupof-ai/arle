# DSv4 PD systematic analysis — operators 接入好用好 → end-to-end trace → PD problems → throughput

**Date:** 2026-06-06. **Driver (ckl):** "把算子先接入好用好,然后端到端 trace nsys 发现真实
问题和瓶颈,系统化优化吞吐 … PD 分别的问题是什么解决好 … 系统上限远远没达到呢."

**Why this doc:** prior DSv4 perf work profiled narrow per-kernel `cuda_gpu_kern_sum` on
isolated runs — which **overstates wall-clock** (kernels overlap on streams: 81 s summed /
19.9 s wall ≈ 4×) and produced a false "near-ceiling" read. The ceiling is far off. This
re-bases on (1) operator integration truth, (2) the actual PD pipeline, (3) the questions a
proper **wall-clock end-to-end trace** must answer.

## 1. Operator integration state (算子接入好用好 — the audit)

| Operator | Flag | Default | "好用"? |
|---|---|---|---|
| FlashMLA decode | `FLASHMLA_DECODE` | **ON** | ⚠ **BROKEN in rebuild** — `CUDA_ERROR_NOT_SUPPORTED` at `arle_flashmla_sm90_sparse_decode_get_meta` (sm_90 vs sm_90a / stale obj / stub). Prod binary works. **Fix = #1 prerequisite.** |
| Fused-wqkv decode | `FUSED_WQKV_DECODE` | **ON** | ✓ (+18.4%) |
| GPU router (on-device) | `GPU_ROUTER` | **ON** (moe.rs `!="0"`) | ⚠ naming wart: `dsv4.rs:813` reuses the same name `.is_some()` = default-OFF for the *pooled-decode scratch* (a different thing). Confusing; #35. |
| Fused wq_a\|wkv prefill (DeepGEMM) | `FP8_LINEAR_DEEPGEMM` | **OFF** | ⚠ licensed −5%, never default-flipped. opt-in only. |
| MoE transport | `MOE_TRANSPORT` | **allreduce** | ⚠ **production path = native-DeepEP is OFF** (gated behind perf profile). Default runs the slower comm path (redundant MoE all-reduce). |
| Comm overlap | `COMM_OVERLAP` | **OFF** | infra exists (`comm_stream`, fences) but **not wired into the forward**. The 32.4%-comm lever. |
| Decode CUDA graph | `DECODE_GRAPH` | **OFF** | landed (#25); B=1 launch-overlap was ~+1.5% (wash) — only matters paired with one-shot comm. |
| MoE contig decode | `MOE_CONTIG_DECODE` | **OFF** | slower at B=1 (28.4 vs 37.6) — correctly off. |
| Spec decode (MTP) | `SPEC_DECODE` | **OFF** | parked; depth-1 head 96.9%, needs frozen-prepare-chain verify for 6ms. |
| FlashMLA prefill | `FLASHMLA_PREFILL` | **OFF** | killed +36% (prepare-chain > attention-math savings). |

**接入好用好 gaps to close before trusting any trace:** (a) FlashMLA decode build fixed;
(b) native-DeepEP runnable + traced (production comm); (c) FP8_LINEAR_DEEPGEMM on. Flags →
CLI `--` (#35).

## 2. The PD pipeline (NVTX stages — the actual op sequence)

**Per layer ×61** (`dsv4/layer_NN`):
`embed` → [ `mla_attn` (proj wq_a/wq_b/wkv/wo + compressor + indexer + csa_select +
FlashMLA/hybrid attn) → **`attn_allreduce`** (TP) → `moe_forward` (router + grouped experts,
DeepGEMM) → `shared_hc` (shared expert + HC mix) → **`moe_allreduce`** (TP) ] → final RMSNorm
→ `lm_head_project` → `sample`.

**→ 2 NCCL all-reduces per layer = 122 per forward step.** Prefill runs this once over N
tokens (compute-bound, batched); decode runs it per token (B=1, latency-bound).

## 3. PD PROBLEM STATEMENTS — end-to-end wall-clock trace RESULT (2026-06-06)

**Trace:** `dsv4_e2e_4096x64.nsys-rep`, working binary (FlashMLA decode ON, confirmed
`flash_fwd_splitkv_mla_fp8_sparse_kernel` present), 4096 prompt + 64 decode, FP8_LINEAR_DEEPGEMM=1
+ GPU_ROUTER=1 + **allreduce** (native-DeepEP NOT runnable in parity: "not enough CUDA devices").
Method: windowed (prefill / cold-first-decode / steady-decode), **exclusive interval coverage**
(priority-ordered, overlapped kernels not double-counted → wall-clock truth, fixes the
summed-kernel trap).

### ⭐ HEADLINE: `dsv4_csa_select` (sparse top-k selection) is the #1 cost at the 4096 SLO shape — for BOTH P and D.

This OVERTURNS the earlier "decode = comm 32.4%" picture, which was measured on an **8-token
smoke shape** where csa_select is trivial. csa_select scales with context; at 4096 it dominates.
(SLO-vs-smoke, [[../experience/errors/2026-05-27-dsv4-tp-allreduce-slo-prefill-kill]].) The
GEMV/comm levers chased earlier were smoke-shape artifacts.

### PREFILL (4096, cold first request — 13.81s wall, exclusive):
| bucket | wall | % |
|---|---|---|
| attention / sparse-prepare (csa_select + compressor + indexer + CSA/HCA math) | 6.13s | **44.4%** |
| **host-gap / no GPU active** | 4.52s | **32.7%** |
| dense MLA linear (the GEMV bucket) | 2.59s | 18.7% |
| NCCL | 0.18s | 1.3% |
| MoE | 0.17s | 1.2% |

**Statement:** prefill is **sparse-attention-prepare-bound (44.4%) + host-gap-bound (32.7%)**,
NOT MoE/NCCL. The 32.7% host-gap is **largely cold-start** (lazy weight-load / H2D /
DeepGEMM-JIT — cuMemcpyHtoDAsync 62.5s summed across ranks, a 2.36s no-GPU gap in layer_00) →
a warm prefill is faster; **separate cold vs warm** (re-measure a 2nd request). Levers, ranked:
(1) **csa_select reduction** (reuse / skip-when-kv≤topk / faster kernel), (2) **pre-warm the
cold-start H2D/JIT chain**, (3) dense linear (overlap-protected per #36 — low ROI).

### DECODE (4096 steady, 62 steps — 137.6ms/token wall **osrt-INFLATED**, exclusive proportions valid; host_gap 0.3% ⇒ GPU-bound):
| bucket | per-token | % |
|---|---|---|
| attention / sparse-prepare — **`dsv4_csa_select` dominant** | 103.0ms | **74.9%** |
| MoE expert path | 12.4ms | 9.0% |
| dense linear | 6.0ms | 4.4% |
| NCCL | 5.5ms | 4.0% |
| HC | 3.9ms | 2.9% |
| FlashMLA math itself (`flash_fwd_splitkv_mla_fp8`) | ~5.5ms (small) | — |

**Statement:** decode steady-state is **GPU-bound, csa-sparse-selection-bound (74.9%)** — NOT
comm (4%), NOT FlashMLA math, NOT host. The critical path is **per-layer CSA selection / index
prep**. Lever #1: **reduce/amortize csa_select** (cross-layer reuse = SGLang `skip_topk`;
skip-when-kv≤index_topk; or a faster selection kernel). Cold first-decode step is 94% host-gap
(one-time VMM/shareable-handle setup — `cuMemImportFromShareableHandle` etc.) → pre-warm or
exclude. **CAVEAT:** absolute 137ms is osrt-inflated (real ≈ 26-50ms; re-measure clean
`--trace=cuda,nvtx` at 4096); proportions hold (host_gap 0.3%).

### Unified conclusion
**csa_select is the single biggest single-request lever (P 44.4%-bucket + D 74.9%), and it
scales with context.** Cross-layer reuse (E2) — previously down-ranked as lossy/risky — is now
the top priority; the cheap read-only overlap diagnostic is licensed. Throughput ceiling (#38,
single-row executor) is the separate big axis. Comm (4%), GEMV (#36, overlap-protected), and
mhc are all secondary at the SLO shape.

## 3.9 (superseded) Earlier open questions the trace answered

Not kernel-sum %. The wall-clock critical path, with gaps/idle, **prefill and decode separately**:

**PREFILL (open questions):**
- **Does the 16.9s include model load?** (Codex: "the profile wraps model load.") If load is
  ~Ns, real compute-prefill is 16.9−N — could change the whole problem. EXCLUDE load (NVTX
  prefill range / first-token ts).
- Of real compute-prefill wall: GPU-compute vs comm (122 AR) vs host-gaps vs memory. Which
  dominates? Is the prepare-chain (compressor/indexer/csa_select) serial on the critical path,
  or overlapped behind attention?
- csa_select recomputed per CSA layer — wall-clock cost (not summed-kernel)?

**DECODE (open questions):**
- Steady-state per-step wall (26.6 ms): how much is comm (122 AR + the FlashMLA Q all-gather),
  MLA attention, MoE, shared_hc, sample, host-gaps?
- Is the step GPU-bound, comm-bound, or host-bound? (B=1 was called GPU/comm-bound — confirm
  on the wall-clock timeline, cross-check NVTX-vs-GPU-activity for the sync-phantom trap.)
- Does native-DeepEP shrink the comm vs allreduce? Is the FlashMLA Q all-gather (16% kern-sum)
  real wall-clock?

## 3.5 THROUGHPUT CEILING (found by source analysis — the "系统上限远远没达到" answer)

**The R6 CUDA executor is SINGLE-ROW ONLY.** `Dsv4Executor::submit` bails
`ensure!(rows == 1, "DSv4 CUDA forward is single-row only")` (executor.rs:783; same for
the generic R6 path executor.rs:325, Qwen executor.rs:1036). Every forward processes ONE
row — prefill OR decode.

But the **planner builds MULTI-ROW plans with NO cap** (`build_forward_plan` pushes a
`DecodeRow` for EVERY active Decoding request, planner.rs:21-38, no limit;
`retract_decode_to_fit` trims only for *memory*, not to 1) and the engine calls
`submit(&plan)` **once** with no per-row split (lib.rs:404-432). So at c=N the plan has N
decode rows → `submit` hits `ensure!(rows == 1)` → **DSv4 c>1 LIKELY ERRORS** (not merely
serializes). **No true batched decode exists in the rewrite, and concurrent decode may be
outright broken.** The prior throughput-sweep c=8 numbers
([[../experience/wins/2026-06-06-dsv4-first-throughput-sweep-scaling-gap]]) may be
stale/pre-cutover — **Codex must confirm empirically: does DSv4 c=2 decode error at the
single-row bail?** (The pre-rewrite `infer/src/model/deepseek/` HAD an FFN-batched decode —
[[../experience/wins/2026-05-29-dsv4-true-batched-decode]] — it was NOT ported; R6 was a
clean single-row skeleton, batching deferred, #5.)

**Consequence:** at c=N, weight-load + 122 all-reduces + FFN are NOT amortized across rows
(each row pays them fully) → throughput barely scales (c=8 = 1.63×, per-request 33→7,
[[../experience/wins/2026-06-06-dsv4-first-throughput-sweep-scaling-gap]]). SGLang batches
all concurrent decode rows into ONE forward → near-linear. **This is the single biggest
untapped axis.**

**The fix (the throughput program):** implement batched decode in the R6 executor — consume
the planner's N-row decode plan in one forward: batched FFN/MoE/comm over `[N, hidden]`
(row-independent / sum-reduce, the pre-rewrite proved byte-identity) + attention. The
attention is the hard part: the FlashMLA decode FFI is hard-`b=1` (single-row indices
builder / pack / `sched_meta(b=1)`), so either loop attention per-row inside the batched
layer (the pre-rewrite approach — bounded win, FFN/comm amortize) OR build a true batched
FlashMLA decode (`b=N`, shared cross-sequence KV pool — a CUDA kernel tranche, the full win).
**Open (verify on pod): does DSv4 c>1 currently error at the single-row bail, or does the
serving path serialize per-row?** Codex to confirm.

## 4. Plan (ckl's sequence)

1. **算子接入好用好** — FlashMLA-decode build fixed; native-DeepEP runnable; licensed operators
   ON. (Codex, in progress.)
2. **端到端 trace** — one full request (4096 prefill + ≥64 decode), wall-clock timeline,
   PD-separated, load-excluded. (Codex analysis focus.)
3. **PD problem statements** — two crisp, evidence-backed bottleneck statements (§3 answered).
4. **系统化优化吞吐** — attack the true bottlenecks; throughput is the big untapped axis
   (c=8 only 1.63×, c=32 OOM at static `total_pages=8192` — #37 dynamic VRAM).

**Method discipline:** wall-clock is ground truth (not kern-sum %, not NVTX-wall-ending-in-sync).
One variable per A/B. License-or-kill on the SLO shape. Operators integrated before measured.
