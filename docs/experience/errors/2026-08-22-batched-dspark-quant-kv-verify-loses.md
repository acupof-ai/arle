# Batched DSpark over quantized KV loses from c=8 — the verify path is the slow one

> Status: Rejected on measurement; gate restored. Runtime `2aa569adc` (gate
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

At c≥4 on this chain decode is ≈80 % attention. Plain decode rows run the
tensor-core quantized-pool kernel (`paged_attention_quantized_fa3.cu`); the
verify step (block + 1 query rows) runs the FA3 dequant shim — a bf16 temp of
the row's pages plus the bf16 FA3 kernel — which costs several plain steps
per verify. At ≈1.9 accepted tokens per step that is a loss from c=8. On BF16
pools verify and decode share the FA3 kernel, which is why the same batched
path paid +8.5 % at c=16 there.

## Fix

Gate restored (`paged_kv_bf16()`), with the numbers in the comment. The
lever is a quantized-pool attention kernel that takes the few-row verify
shape: extend the 16-row MMA tile from (6 heads × 1 token) to (token, head)
pairs with the in-block causal mask. One kernel unlocks batched DSpark on
quantized KV and the batched MTP verify plan.

## Rule

A spec-decode gate on KV format is a statement about the verify kernel, not
about the draft. Before lifting it, compare the verify path's attention
kernel with the plain decode path's — if they differ, the batched form can
lose even with parity intact.
