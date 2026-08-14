# FA3 quantized KV paths (A: dequant+FA3, B: native quant kernel) — CUDA, 2026-08-13

> Status: Shipped

## Goal

Enable FA3-class paged attention for FP8/INT8 KV pools on qwen35, so quantized KV is not locked to the older split-KV varlen kernel.

## Hypothesis

Two complementary paths:
- **Path A**: dequantize active pages to a BF16 temp, call the existing FA3 BF16 kernel. Enables CUDA graph for quant pools.
- **Path B**: self-contained persistent split-KV kernel reading FP8/INT8 + scales directly, dequantizing on-the-fly. No temp buffer.

Path B should match or beat the varlen kernel; Path A trades temp VRAM for FA3's scheduling efficiency.

## Parameters

- Model: ThinkingCap-Qwen3.6-27B-FP8 (head_dim=256, 16 full-attn layers, 4 KV heads)
- KV formats: fp8, int8
- Needle ladder: 115, 300, 446, 2000, 8000 tokens ×3 runs, greedy
- Concurrent: 4 × 2000-token requests
- Throughput: synthetic prompts, c=1/4/8, 8 req/concurrency, 100 max tokens

## Environment

- H20 pod, single GPU, CUDA 12.8
- Binary: commit a3a769db1
- Server flags: `--kv-cache-dtype fp8|int8`

## Results

| Test | FP8 | INT8 |
|------|-----|------|
| Needle 115-8000 ×3 | 15/15 exact | 15/15 exact |
| Concurrent needle c=4 | 4/4 passed | 4/4 passed |
| Throughput c=8 | 180.3 tok/s | — |
| Errors | 0 | 0 |

Reference (varlen kernel, same model, FP8): c=8 185.9 tok/s. Path A/B at 180.3 is −3.0% — the dequant overhead or the native kernel's less mature scheduling eats the FA3 advantage at c=8. At c=1 the difference is within noise.

## Problems

- Path A's full-pool dequant allocates a BF16 temp the size of the active working set. With `--mem-fraction-static 0.9` this is ~40 GB — too much. The compact-page-table variant (active pages only) is the shipping path.
- The native kernel (Path B) lacks FA3's PackGQA and TMA; it is a straightforward persistent split-KV with online softmax. Closing the gap to FA3 BF16 needs TMA loads and warp-specialized scheduling.

## Learnings

Both paths are correct and shipping. The varlen kernel remains the default for quant KV at c=8 (3% faster); Path A/B are the fallback for CUDA graph capture and future optimization. The −3% is a dequant/scheduling tax, not a correctness issue.
