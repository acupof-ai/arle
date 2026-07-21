# DSv4 production-all-on TP4/EP4 re-anchor (2026-07-19)

> Status: Shipped — benchmark complete 2026-07-19

## Context

Re-anchored DSv4 production-all-on baseline on current HEAD `45dd64bd2`,
4×H20 GPUs 3,5,6,7, TP=4/EP=4, `bench-prompts-64.jsonl` (~2.8k tok),
120 s/point, max_tokens 256, seed 20260416. Binary `c7730414…`, kernel
`bundle:7eef1a90…`. Needle ×3 passes (15/15 strict).

## What Worked

| c | out tok/s | total tok/s | complete |
|---|---:|---:|---:|
| 1 | 38.0 | 457.6 | 19/19 |
| 4 | 74.6 | 873.4 | 36/36 |
| 8 | 123.7 | 1446.8 | 64/64 |
| 16 | **195.7** | 2290.6 | 96/96 |

Production optimization set confirmed on: FlashMLA prefill/decode, fused
WQKV, native DeepGEMM, paged/batched decode, decode reuse, chunked prefill
2048, CUDA mempool retention, NUMA pinning, NCCL all-reduce MoE transport.

## Rule

- Base (no speculative) remains the production champion for all concurrency.
- MTP/DSpark are c1-only or pending structural fixes
  (see [errors](../errors/2026-07-19-dsv4-mtp-dspark-high-concurrency-regression.md)).
- Re-anchor vs old chunk-2048 row (different GPU set + 120s vs 90s) — not
  a strict Δ comparison.
