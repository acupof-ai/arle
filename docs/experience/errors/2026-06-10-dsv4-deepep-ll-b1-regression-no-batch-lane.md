# DSv4 deepep_ll is −55% at B=1 and the batched lane that would justify it doesn't exist

**Date:** 2026-06-10. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Commit:** `d8b675e1` (deepep_ll integration, gated `ARLE_DSV4_MOE_BACKEND=deepep`, default stays `allreduce`).

## Context

Multi-day effort built a complete, correct, NVSHMEM-proven token-owned DeepEP
low-latency MoE path for DSv4 (`dsv4_moe_forward_deepep_ll`): deepep-sys LL FFI
(`internode_ll` + NVSHMEM bootstrap), masked silu_mul_quant, masked grouped DeepGEMM,
slice→LL-dispatch→masked-GEMM→masked-silu→masked-GEMM→LL-combine→all-gather. All NVSHMEM
unknowns were de-risked (build+link, 8-rank `nvshmem_init`, IBGDA env). Correctness
PASSES: needle 7391 retrieved + byte-stable, GSM8K=72, coherent.

The plan's premise: DeepEP wins at the **batched throughput lane** (SGLang max-tput),
not B=1. So the license gate was a batched A/B.

## The A/B (same binary, env-flip `ARLE_DSV4_MOE_BACKEND`, serial)

| backend | B=1 p50 tok/s | c=8 / c=16 / c=32 |
|---|---|---|
| allreduce | **39.88** | crash |
| deepep_ll | **17.97** | unsupported |
| **Δ (deepep vs allreduce)** | **−54.9%** | — |

deepep_ll booted clean (LL buffer on all 8 ranks, baked IBGDA env working, no hang/assert),
decoded correctly. It is simply **2.2× slower at B=1** — the NVSHMEM LL dispatch/combine
round-trip moves one token with 7 ranks idle (~7.1 s/128 tok vs allreduce ~3.2 s). Pure
overhead at B=1, by construction.

## Root cause — the batched lane does not exist

The c=8/16/32 lanes crash the engine on **both** arms:
```
DSv4 CUDA prefill/mixed forward is single-row only, got 1 prefill + 1 decode rows
```
The DSv4 CUDA forward is **single-row only** — when continuous batching co-schedules one
request's prefill with another's decode (any concurrency ≥2), the forward refuses → engine
thread dies. This is a **serving-stack limitation independent of the MoE transport**
(reproduces on allreduce). Separately the binary refuses deepep batched decode outright
(`DSv4 batched decode does not yet support the DeepEP MoE transport`).

So deepep_ll's only possible winning lane (many tokens fanned across ranks) is unreachable.
The A/B can neither validate nor exonerate it — the lane it needs doesn't run.

## Rule

- **Verify the SLO *lane exists and runs* before building the optimization that targets it.**
  The entire deepep_ll plan assumed a batched throughput lane. The DSv4 CUDA forward can't
  batch at all (single-row only) — a prerequisite that was never checked. A correct,
  fully-de-risked implementation still produced a −55% regression because it optimizes a
  lane the serving stack cannot run (§0: "SLO verdict from the SLO workload, not a smoke
  shape"; here, the SLO *shape* doesn't execute).
- **deepep_ll stays gated/opt-in, NOT default** (`d8b675e1`). It is correct + NVSHMEM-proven
  foundation; it is a B=1 regression with no batched lane to redeem it today.
- **The real lever is NOT more deepep work — it's DSv4 batched decode** (kill the single-row
  forward ceiling so a concurrent throughput lane exists). Only then is a deepep_ll-vs-allreduce
  batched A/B meaningful. Until then, allreduce remains the best runnable path.

## Refs
- Plan: `docs/plans/2026-06-09-dsv4-deepep-best-path-rewrite.md`
- Bench scripts (pod): `/data01/build/arle_serve_{allreduce,deepep}.sh`, `dsv4_ab_bench.py`
