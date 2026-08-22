# Batched DSpark over quantized KV loses from c=8 — cause unknown, verify kernel ruled out

> Status: Rejected twice on measurement; gate restored (`1f36fea61`). Runtime `2aa569adc` (gate
> lifted) vs `7d58850dc`, Qwen3.6-27B-FP8 + `Qwen3.6-27B-DFlash`, fp8 KV,
> 1×H20, 32K agent chain, 32 req/point, greedy, `--dspark-block-size 6`.

## Context

With FP8/INT8 KV the batched DSpark draft was gated to BF16 pools, so DSpark
speculated at c=1 only. The draft reads its own ctx ring and the verify is
the trunk's paged attention, so lifting the gate looked free.

## Phenomenon

Per-request decode tok/s, same binary otherwise:

| c | no-spec | DSpark, per-row (gate on) | DSpark, batched (gate off) |
|---:|---:|---:|---:|
| 1 | 55.8 | 84.3 | 84.2 |
| 4 | 32.2 | 32.3 | 32.5 |
| 8 | 21.5 | 21.8 | **19.4 (−10 %)** |
| 16 | 13.5 | 13.9 | **11.4 (−16 %)** |

Needle ladder ×3 under a concurrent 32K stream (batched path engaged): 12/12
exact, DET — parity holds; throughput does not.

## Root Cause

Cause unknown. The first hypothesis — the verify step (block + 1 query rows)
running the FA3 dequant shim instead of the tensor-core quantized-pool kernel
— was tested and refuted (below).

## Fix

Gate restored (`paged_kv_bf16()`), with the numbers in the comment.

## Follow-up: verify-shape MMA kernel does not close the gap

`paged_attention_quantized_fa3.cu` now takes up to 8 query tokens per row
(`c177cda5b`, 16-row tile of (token, head) pairs, in-block causal mask), so
verify rows and plain rows share one kernel. Gate lifted again (`6f8d7da6c`),
same setup, needle ladder ×3 12/12 DET:

| c | no-spec | DSpark, per-row | DSpark, batched + MMA verify |
|---:|---:|---:|---:|
| 1 | 55.8 | 84.3 | 82.9 |
| 4 | 32.2 | 32.3 | 34.2 (+6 %) |
| 8 | 21.5 | 21.8 | **19.2 (−12 %)** |
| 16 | 13.5 | 13.9 | **11.2 (−19 %)** |

The c≥8 loss is unchanged, so it does not sit in verify attention. The
kernel extension stays (one attention path for every decode shape; c=1 is a
wash); the gate lift is reverted (`1f36fea61`). Remaining candidates are the
batched draft forward and rejected-token waste at ≈1.9 accepted/step — an
`ARLE_CUDA_PROFILE=1` op_timing split of the batched step is the next probe.

## Rule

A spec-decode gate on KV format is a statement about the verify kernel, not
about the draft. Before lifting it, compare the verify path's attention
kernel with the plain decode path's — if they differ, the batched form can
lose even with parity intact.
