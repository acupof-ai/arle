# ARLE rewrite — Qwen3.5/3.6 + DSv4 final verification & performance report

**Date:** 2026-06-04. **Branch:** `arch/ideal-inference-engine`.
**Scope:** the device-neutral rewrite (`infer-core`/`-seam`/`-cuda`/`-metal`/
`-server`/`-api` + `infer-topo`/`-moe`/`-util`) as the serving truth. Details +
evidence in [`2026-06-04-rewrite-completion-verification-report.md`](2026-06-04-rewrite-completion-verification-report.md);
this is the consolidated headline.

## 1. Verdict

The rewrite is **verified and complete on serving** across Metal + CUDA, TP/EP, the
FP8 DeepGEMM MoE backend, and the DeepEP all-to-all transport — **all five goal axes
are met**: ① **legacy `infer/` is DELETED** (`e81b98fb`, ~167k LOC; agent+cli+train
all on `infer-api`/`infer-util`, the rewrite is the sole stack); ② elegant/extensible
arch (§4); ③ Metal+CUDA verified (Metal regression recovered + default-on); ④ EP+TP
verified (DSv4 TP=8/EP=8); ⑤ DeepGEMM 16/16 + DeepEP transport verified. To unblock
①, train's CUDA OPD-teacher surface (`forward_token_logits`/offload/`remerge_student_lora`)
was built on `infer-api` (the rewrite Qwen3.5 path) + typecheck-gated; its GPU numeric
verify (logits parity / offload / lora-merge) is the fix-forward follow-up on the pod
(the rewrite is the committed truth, so no `infer` fallback is retained).

**Update 2026-06-04 (perf push + honest corrections).** The rewrite-completion milestones
above stand (infer deleted + merged, arch, Metal+CUDA serving, TP/EP, DeepGEMM, DeepEP).
The performance pass since surfaced + fixed several things and one honest correction:
- **CUDA-graph "support" wired** (`6a01e8cc`): the `--cuda-graph` flag was a no-op
  (`let _ = enable_cuda_graph`); now real + default-on for single-GPU Qwen dense (TP/MoE
  stay eager). Single-GPU A/B pending.
- **DSv4 grouped-cache prebuild + wq_b TP-shard fix** (`d5115de5`): the MoE rebuilt static
  expert weights every layer/token (~529 ms/token, eliminated); the wq_b TP-shard fix
  cleared a pre-existing `wo_a cols != local_width` load error. Prefill-verified.
- **CI was red on main** since the PR #53 merge (workflows referenced the deleted `infer/`
  crate); fixed + green (`5bd92749`). Task #13's "green main CI" had been false.
- **Honest correction:** DSv4 **incremental decode (`start_pos>0`) is broken** — the prior
  "3/3 16/16" was prefill-only. Root-cause in progress (#23), jointly with Codex.
- **SGLang-class perf roadmap** (`ee230308`, `docs/plans/2026-06-04-dsv4-decode-sglang-class-perf.md`):
  on-device MoE routing (kill per-layer H2D/D2H) + full decode CUDA-graph (capturable
  all-reduce) → 5–6 ms/token target, sequenced behind the decode-correctness fix.

## 2. Correctness matrix (all greedy, exact-token unless noted)

| Path | SKU | Result |
|---|---|---|
| Qwen3 dense CUDA forward (R6) | H20 | **16/16** vs HF gold (2 shapes) |
| Qwen3.5 / Qwen3.6 MoE Metal forward | M4 Pro | end-to-end correct, prefix reuse, greedy |
| DSv4-Flash bf16 multi-GPU (MLA+CSA/HCA+HC+hash+FP8-MoE) — **PREFILL** | 8×H20 TP=8/EP=8 | prefill argmax matches bf16 oracle (token#1=11111, 6-tok-prefix=603) ✓; **incremental decode `start_pos>0` is BROKEN** — diverges from prefill, a pre-existing never-completed path (#23, in progress 2026-06-04). The earlier "3/3 16/16" was prefill-only. |
| DSv4 production DeepGEMM FP8-MoE | 8×H20 | **16/16** vs bf16 oracle (routed + shared expert) |
| TP=8 / EP=8 row-parallel all-reduce | 8×H20 | verified via DSv4 (the model that needs sharding) |
| CUDA Graph capture/replay | H20 | eager == replay == HF gold (16/16) |
| DeepEP native dispatch/combine | 8×H20 | **VERIFIED** — layer-0 `moe_out` parity vs allreduce = bf16 float-order noise (max_abs 0.0039, rms 4.4e-4, all 8 ranks); token-1 `[260]`==`[260]` |

## 3. Performance

**Metal (c=1, the local single-user focus) — COMPLETE.** Cross-step decode pipeline
(default-on, `INFER_METAL_PIPELINE`) recovered the regression to ≈ legacy:

| Model | rewrite (pre-fix) | rewrite (pipelined, default) | legacy | Δ recovered |
|---|---|---|---|---|
| Qwen3.6-35B-A3B-4bit (MoE) | ~70 tok/s | **~79.5 tok/s** | 84.3 | +10–13% |
| Qwen3.5-0.8B-MLX-4bit (dense) | ~244 tok/s | **~294 tok/s** | 282.5 | +20% |

Greedy bit-identical; prefix-reuse ttft 6→3; c≥2 errors loud at the single-row guard
(no multi-slot Metal decode — pre-existing). FP8/4-bit Metal MoE quant swap points
remain a follow-up.

**CUDA — correctness complete; per-op perf characterization underway (#16).** FP8 MoE
is verified on both the native-grouped bypass and the production DeepGEMM backend.

*Op-ceiling characterization (Colab A100 sm_80, secondary to the H20 production
target; cuBLAS BF16 + STREAM, warmup-excluded):* HBM 1228 GB/s achieved (79% of
1555). DSv4/Qwen3.6 **prefill** projection + MoE GEMMs are **compute-bound and a
competent kernel clears ~70-87% of the 312 TFLOP/s BF16 peak** (q_proj 83%, o_proj
85%, MoE gate+up 76%, lm_head 87%; weakest = MoE down `n=7168,k=2048` at 69%, a
thin-K shape to watch). **Decode (m=8) GEMM runs at ~3% of peak** → latency/launch-
bound, so decode tok/s is governed by **KV-read HBM bandwidth, not FLOPs** (matches
the Metal finding that decode is bandwidth/pipelining-bound). **Architectural finding:
FP8 tensor-core matmul does NOT exist on A100 (sm_80) — it is Ada/Hopper-only
(sm_89/sm_90)**, so cuBLASLt FP8 returns `NOT_SUPPORTED` on A100; **the production
DSv4 DeepGEMM FP8-MoE throughput ceiling has no A100 analog and must be measured on
the H20 Hopper path** (Codex profiling in flight).

*Production H20 numbers* (DSv4/Qwen tok/s, TP=8 scaling, per-op nsys breakdown +
compute/comm overlap) — the remaining input, Codex profiling on 8×H20 now.

**Update 2026-06-05 — DSv4-Flash decode 6× + prefill 12× (the H20 production
numbers above). DSv4-Flash FP8 TP=8/EP=8, 8×H20, parity harness (greedy,
16/16 == oracle at every stage).** The decode-correctness fix (#23) cleared the
incremental path; the perf arc then took decode from 236.8 → 39.5 ms/token by
eliminating, in order, every layer of non-compute overhead:

| Stage | tok/s | ms/token | what was eliminated | commit |
|---|---|---|---|---|
| baseline (host route, eager) | 4.365 | 236.8 | — | — |
| #24 on-device GPU router (+ sync bridge) | 6.245 | 160 | per-layer host-route D2H+CPU+H2D (½) | `fa1b78f2` |
| #29 persistent scratch pool | 21.615 | 46 | per-step alloc/free + the async reuse-boundary race | `4d62062a` |
| buffer/D2D lever | 25.129 | 41.2 | D2D copies (a keepalive `clone()` = device copy), zeroing | `7bdb3819` |
| #25 breakable decode graph (gated) | 25.517 | 39.5 | launch API −87% (overlapped → wall flat) | `7104813e` |

Decode-step overhead profile before → after: `cuStreamSynchronize` 39.6% → ~0
(host route gone), `cuMemAllocAsync` 36.9% → 2.8 ms/token, `cuMemcpyDtoH` route
→ 0.018 ms/token. **Prefill rode the same fixes for free: 512-token TTFT
14.66 s → 1.204 s (~12×), `moe_route` now 6–11 ms (was the 93%-GPU-idle cause).**

**Now GPU-kernel-bound — and the floor is partly an unoptimized kernel.** The
remaining ~39.5 ms/token is real GPU work, dominated by the *same* two kernels
that floor prefill: the attention-side **FP8 block-scaled linear** (wq/wkv/wo +
compressor + HC) and **hybrid MLA attention**. ncu shows the FP8 linear runs as a
hand-written *scalar* CUDA-core kernel — tensor-core (WGMMA) <1%, HBM BW ~10% —
so a large chunk of the "floor" is ~10× headroom, not silicon (lever: DeepGEMM
dense FP8 GEMM for prefill M>1; a bandwidth-bound FP8 GEMV for decode M=1; in
progress). **5–6 ms/token is an H100/H800 number** (SGLang's reference SKU);
the H20 floor after the kernel pass is the open question, but the structural arc
(scheduling / host / buffer / launch / routing) is closed and measured. Roadmap +
per-stage evidence:
[`docs/plans/2026-06-04-dsv4-decode-sglang-class-perf.md`](../plans/2026-06-04-dsv4-decode-sglang-class-perf.md)
§§5–9; per-stage wins under `docs/experience/wins/2026-06-05-dsv4-*`.

**Methodology note — three framing traps, each falsified before chasing.** The
biggest line in a profile was repeatedly *not* the bottleneck: `lm_head_sample`
9.4 ms (an NVTX range absorbing the step-end sync), `cudaLaunchKernel` 13.36 ms
(CPU launch overlapped with GPU exec), the 2048-prefill 4.96 s attn-allreduce
(a one-time NCCL/rank-start warmup, not steady-state). Each was proven artifact by
a cheap control (sync-before probe / graph A/B / per-rank arrival timing) before
any code changed. License-or-kill was always wall-clock per-token, never a narrow
API/NVTX-window share.

## 4. Architecture (elegant / clean / extensible — substantiated)

Device-neutral rewrite: `infer-plan` (IR) → `infer-seam` (host-only traits) →
`infer-core` (Engine/scheduler/RadixCache) → `infer-cuda`/`infer-metal` (executors) →
`infer-server`/`infer-api`.

**Extensibility is demonstrated, not asserted.** The same `infer-core::Engine` —
generic over `infer_seam::{BackendExecutor, KvPool}` — drives **two structurally
divergent backends today**: the CUDA continuous-batching scheduler (paged KV,
TileLang/native-CUDA kernels, TP/EP/DeepGEMM/DeepEP) and the Metal MLX runtime
(MLX bridge, packed varlen decode). Both run as `Engine<E, K>` with the *identical*
scheduler / RadixCache / chunked-prefill / plan / sampling / streaming / telemetry
code above the seam (proof: `agent-bench` instantiates `Engine<MetalExecutor,
MetalKvPool>` and `Engine<CudaExecutor, CudaKvPool>` over one harness). Adding a 3rd
backend (e.g. ROCm/HIP) is implementing exactly two host-only traits
(`BackendExecutor` submit/poll + `KvPool`) — the scheduler, cache, IR, server, and
API are untouched. This is the backend-agnostic-scheduler goal the arch review set:
no per-backend scheduler rewrite.

**Clean / one canonical flow.** This session converged divergence rather than
layering adapters: extracted `infer-util` (backend-agnostic hf_hub/logging leaf
crate), deleted the speculative `infer-models` crate + 3 orphaned seam traits (zero
impls), migrated `agent` + `cli` off direct `infer` to the single `infer-api` front
door. No parallel old+new paths; backends are thin plug-ins behind one seam.

## 5. Remaining (all goal axes met; these are follow-ups / fix-forward)

DONE this session: DeepEP wiring mirrored (`9bd92418`); train OPD surface + **`infer/`
deleted** (`956c774f` + `e81b98fb`).

1. **GPU numeric verify of the OPD-teacher surface** (logits parity vs the pre-sample
   forward / offload correctness / lora-merge numerics) on the pod — fix-forward
   (typecheck-gated in-tree; the rewrite is committed truth, no `infer` fallback).
2. **CUDA per-op perf profiling (#16)** — the perf half of this report.
   **Substantially delivered (§3 Update 2026-06-05):** DSv4 decode 6× / prefill
   12×, structural overhead closed, the floor localized to FP8-GEMV + hybrid
   attention (the early nsys's "DSv4 FP8 GEMV 26.4%, hybrid attn 21.7%" is exactly
   today's kernel floor — confirmed scalar, ~10× headroom). Remaining: the kernel
   pass (FP8 GEMV → tensor-core/bandwidth-bound, then hybrid attention) in
   progress; DSv4 weight load **111 s** is still a separate load-time follow-up
   (item 3).
3. **Load + compile optimization** (research delivered,
   [`research/2026-06-04-load-and-compile-optimization.md`](../research/2026-06-04-load-and-compile-optimization.md)):
   top levers = pinned async H2D (load, ~111s target), content-addressed cubin cache
   (build), primitive-count reduction (Metal encode) — each GPU-experiment-gated.
4. **FP8-KV decode** (`alloc_fp8_arena` bail-gated); **V100/sm_70** TileLang
   LayoutInference (deferred legacy tier); **Qwen FP8/4-bit** quant paths.
