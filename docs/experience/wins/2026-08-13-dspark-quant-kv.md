# DSpark spec decode with quantized KV — CUDA, 2026-08-13

> Status: Shipped

## Goal

Enable DSpark speculative decode (`--spec-type dspark`) with FP8/INT8 KV pools on qwen35, and quantify the speedup over no-spec baseline.

## Hypothesis

The `paged_kv_bf16()` gate on batched DSpark draft was conservative ("unvalidated with quant-KV pools"). The draft forward uses the draft model's own KV (ctx ring), not the trunk's KV pool — so quant KV should not affect it. The verify step uses the trunk's paged attention, which already supports quant KV. Removing the gate should be safe.

## Parameters

- Model: ThinkingCap-Qwen3.6-27B-FP8
- DSpark draft: Qwen3.6-27B-DFlash (5-layer, block_size=16, taps=[1,16,31,46,61])
- KV formats: fp8, int8
- Needle ladder: 115, 300, 446, 2000, 8000 ×3, greedy
- Concurrent: 4 × 2000-token requests
- Accept rate: 30 requests × 64 max tokens
- Throughput: synthetic prompts, c=1/4, 8 req/concurrency, 100 max tokens

## Environment

- H20 pod, single GPU, CUDA 12.8
- Binary: commit c04c700a7 (gate removal)
- Server flags: `--spec-type dspark --mtp-draft-model <DFlash> --mtp-draft-tokens 16 --kv-cache-dtype fp8|int8`

## Results

| Metric | DSpark FP8 | DSpark INT8 | Baseline (no spec) FP8 |
|--------|-----------|-------------|----------------------|
| Needle 115-8000 ×3 | 15/15 exact | 15/15 exact | 15/15 exact |
| Concurrent c=4 | 4/4 passed | 4/4 passed | 4/4 passed |
| Accept rate | 31.7% | 31.9% | — |
| Throughput c=1 | 98.3 tok/s | 106.0 tok/s | 50.0 tok/s |
| Throughput c=4 | 171.2 tok/s | 165.3 tok/s | 127.4 tok/s |
| **Speedup c=1** | **1.97×** | **2.12×** | 1.0× |
| **Speedup c=4** | **1.34×** | **1.30×** | 1.0× |
| Errors | 0 | 0 | 0 |

L2 tier demote/promote verified separately: with `--mem-fraction-static 0.7` (46K-token pool), 30 concurrent 2000-token requests oversubscribed the pool. 2 promotes observed (~300 ms per 2416-token slot), 1 promote failed (host pool full → correct recompute fallback). 30/30 needle passed, post-tier needle 15/15.

## Problems

- Accept rate ~32% is the DFlash draft model's ceiling, not a quant-KV issue (same rate with BF16 KV).
- c=4 INT8 is 3.4% behind FP8 — within noise, no matched A/B run.
- **c=4 ITL p99 = 265-272 ms (10.9× mean)**: root cause is prefill-decode interleaving in continuous batching. When a new request starts, its prefill is batched with the decode of existing requests, blocking decode for that step. Evidence: TTFT mean = 270 ms for 63-token prompts (decode is busy), ITL p99 ≈ TTFT mean, no warnings/errors in log, DSpark c=4 has the same p99 (290 ms). Not a bug or regression — a known continuous-batching tradeoff. The fix is disaggregated prefill/decode, a much larger change.

## L3 (NVMe) tier verified

With `--mem-fraction-static 0.7 --kv-dram 1GiB --kv-disk /host/nvme0/kv-ssd --kv-disk-limit 4GiB`:
- L1 = 46K tokens, L2 = 1 GiB, L3 = 4 GiB NVMe
- 40 concurrent × 2000-token requests (80K total) oversubscribed L1+L2
- 3.8 GB spilled to L3 (`kv.mmap` on NVMe)
- 6 promote failures (L2 full → correct recompute fallback)
- 40/40 needle passed, post-L3 needle 15/15 exact

## Learnings

DSpark + quant KV is correct and delivers ~2× decode speedup at c=1. The gate removal was safe: the draft path doesn't touch the trunk KV pool. The accept rate is draft-model-bound, not KV-precision-bound.
