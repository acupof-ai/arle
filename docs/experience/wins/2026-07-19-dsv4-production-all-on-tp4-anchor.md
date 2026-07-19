# DSv4 production-all-on TP4 anchor — CUDA, 2026-07-19

> Status: Shipped

## Goal

Measure DSv4 serving throughput with every current production optimization enabled on four free H20 GPUs.

## Hypothesis

The current production defaults remain correct and establish a clean TP=4/EP=4 anchor on the available GPU set.

## Parameters

```bash
python3 scripts/bench_throughput.py \
  --url http://127.0.0.1:8000 \
  --model /host/DeepSeek-V4-Flash-FP8 \
  --prompts-jsonl bench-prompts-64.jsonl \
  --concurrency-grid 1,4,8,16 \
  --seconds-per-concurrency 120 \
  --max-tokens 256 \
  --seed 20260416 \
  --output bench
```

- Build: `0c0d148dfd28649eb3760d16959d5bd23a3aecaa`, binary SHA256 `ae025e24259b33b5d459103c21c5b8f7b39b87f7e5e28beccfc3ad347a7fbf3c`.
- Kernel ID: `bundle:3d61bd71eb8e552c2895fb87431867dc2daba64bab83e6a1bf848607a75f1e86`.
- Serve flags: `--comm-backend nccl --max-running-requests 32`.
- Workload: 64 unique documents, SHA256 `3543ac33dd23e349eadf08fcf08073a9c6fe2098f3ea2b36d64c75c1e695b835`.
- Completion tokens: max 256; temperature 0.
- Trials: one 120-second run per concurrency.

## Environment

- GPU: 4× NVIDIA H20, physical GPUs `3,5,6,7`; driver `535.161.08`; CUDA `12.9.86`.
- Model: `/host/DeepSeek-V4-Flash-FP8`.
- TP/EP: 4/4; 32 slots; BF16 KV.
- Production defaults: FlashMLA decode/prefill, fused WQKV, native DeepGEMM, decode reuse, 2048-token prefill chunks, CUDA mempool retention, NUMA pinning, and all-reduce MoE transport enabled.
- Disabled by design: contiguous MoE, speculative decode, decode graph, DSA device metadata, and other experimental paths.

## Results

The strict `115,300,446,2000,8000 × 3` needle gate passed 15/15 exact. The 2000-token slice varied only in Markdown emphasis. A separate non-degenerate smoke completed normally.

| concurrency | completed | errors | output tok/s | total tok/s | TTFT p50/p99 ms | ITL p50/p99 ms |
|---:|---:|---:|---:|---:|---|---|
| 1 | 19 | 0 | 37.98 | 455.00 | 1083.78 / 1127.35 | 22.08 / 41.02 |
| 4 | 36 | 0 | 74.50 | 872.52 | 1261.36 / 4197.51 | 44.31 / 89.77 |
| 8 | 64 | 0 | 123.26 | 1442.07 | 1259.23 / 7363.76 | 47.84 / 96.31 |
| 16 | 97 | 0 | 192.20 | 2258.07 | 2177.70 / 2261.04 | 71.68 / 122.04 |

Raw artifacts: `/host/arle-evidence/dsv4-allon-0c0d148d-20260719T0633Z/`; `SHA256SUMS` passes.

## Problems

This fingerprint differs from the stored champion: GPU set `3,5,6,7` instead of `0–3`, 120 instead of 90 seconds, a new source/binary, and no historical dataset SHA. It is not a valid delta. The receipt helper also attempted to create its remote default state directory during a local-only `source-digest`; that side effect is removed in the accompanying fix.

## Learnings

PASS. Current production-all-on DSv4 is correct and reaches 192.20 output tok/s at c=16 on this fingerprint. Treat this as a re-anchor candidate; performance attribution requires a matched archived-binary A/B on the same GPU set and workload bytes.
