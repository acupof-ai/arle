# ARLE rewrite — Qwen3.5/3.6 + DSv4 final verification & performance report

**Date:** 2026-06-04. **Branch:** `arch/ideal-inference-engine`.
**Scope:** the device-neutral rewrite (`infer-core`/`-seam`/`-cuda`/`-metal`/
`-server`/`-api` + `infer-topo`/`-moe`/`-util`) as the serving truth. Details +
evidence in [`2026-06-04-rewrite-completion-verification-report.md`](2026-06-04-rewrite-completion-verification-report.md);
this is the consolidated headline.

## 1. Verdict

The rewrite is **verified and complete on serving** across Metal + CUDA, TP/EP, and
the FP8 DeepGEMM MoE path. The Metal decode regression flagged this session was
root-caused and **recovered to ≈ legacy parity**, now on by default. The one
deferred item is deleting legacy `infer/`, gated solely on porting train's CUDA
OPD-teacher surface (scoped, ~3-4 wk — see the deletion-gate doc); `infer/` is
retained only as a train-only OPD dependency, off every serving path.

## 2. Correctness matrix (all greedy, exact-token unless noted)

| Path | SKU | Result |
|---|---|---|
| Qwen3 dense CUDA forward (R6) | H20 | **16/16** vs HF gold (2 shapes) |
| Qwen3.5 / Qwen3.6 MoE Metal forward | M4 Pro | end-to-end correct, prefix reuse, greedy |
| DSv4-Flash bf16 multi-GPU (MLA+CSA/HCA+HC+hash+FP8-MoE) | 8×H20 TP=8/EP=8 | **3/3 prompts 16/16** vs bf16 oracle |
| DSv4 production DeepGEMM FP8-MoE | 8×H20 | **16/16** vs bf16 oracle (routed + shared expert) |
| TP=8 / EP=8 row-parallel all-reduce | 8×H20 | verified via DSv4 (the model that needs sharding) |
| CUDA Graph capture/replay | H20 | eager == replay == HF gold (16/16) |
| DeepEP native dispatch/combine | 8×H20 | **wiring complete; parity run in final verification** |

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

**CUDA — correctness complete; per-op perf characterization is the remaining work
(#16).** Throughput/latency sweeps (DSv4 tok/s, TP=8 scaling curve, Qwen CUDA tok/s)
need dedicated pod benches not yet run — the GPU time this session went to closing
correctness (sink-offset, DeepGEMM small-m, DeepEP wiring). FP8 MoE is verified on
both the native-grouped bypass and the production DeepGEMM backend.

## 4. Architecture

Device-neutral rewrite: `infer-plan` (IR) → `infer-seam` (host-only traits) →
`infer-core` (Engine/scheduler/RadixCache) → `infer-cuda`/`infer-metal` (executors) →
`infer-server`/`infer-api`. This session: extracted `infer-util` (hf_hub/logging leaf
crate), deleted the speculative `infer-models` crate + 3 orphaned seam traits,
migrated `agent` + `cli` off direct `infer`. One canonical engine contract; backends
are thin plug-ins behind the seam.

## 5. Remaining

1. **DeepEP parity** — final verification in progress (host-sync bring-up run).
2. **CUDA per-op perf profiling (#16)** — DSv4 / Qwen CUDA throughput + TP scaling
   + compute/comm overlap (SGLang-ref); the perf half of this report.
3. **`infer/` deletion** — train's CUDA OPD-teacher surface (~3-4 wk, roadmap in
   [`2026-06-04-train-opd-surface-deletion-gate.md`](2026-06-04-train-opd-surface-deletion-gate.md)).
4. **FP8-KV decode** (`alloc_fp8_arena` bail-gated); **V100/sm_70** TileLang
   LayoutInference (deferred legacy tier); **Qwen FP8/4-bit** quant paths.
