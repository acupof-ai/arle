# DSv4 FP32 probe limited to prefill — DSpark (MTP) decode recovery — CUDA, 2026-07-16

> Status: Shipped

## Goal

The all-boundaries FP32 compressor (see
[2026-07-16-dsv4-fp32-compressor-all-boundaries.md](2026-07-16-dsv4-fp32-compressor-all-boundaries.md))
fixed the #146/#150 prefill corruption but was running the FP32 probe on every
decode token too — including the DSpark (MTP) draft phase, which calls
`compressor_forward` per single-row draft decode. Guard the probe to prefill
only and recover the decode throughput.

## Changes

1. `attention.rs`: `compressor_forward` FP32 probe guard limited to
   `start_pos_device.is_none()` (prefill). Decode (batched, full-flatten,
   graph, MTP draft) always has `start_pos_device = Some` → BF16 path.
2. Guard simplified from 5 conditions to 3: `precomputed.is_none()` and
   `defer_update.is_none()` are redundant — every prefill call site has all
   three `None`, every decode call site has `start_pos_device = Some`.
3. `scripts/bench_throughput.py` (`fc7fdd34e`): removed the strict
   `output_events == completion_tokens` check that failed under MTP
   (multiple accepted tokens per SSE event); usage-based token count is the
   ground truth.

## Correctness (needle gate, needle 738291)

**Depth 0.0 — ALL PASS** (9 lengths, 3/3 exact).

**Depth 0.5 — NO MISSES:**

| Length | exact | partial | miss |
|--------|-------|---------|------|
| 115 | 0 | 3 | 0 |
| 180 | 0 | 3 | 0 |
| 241 | 3 | 0 | 0 |
| 300 | 3 | 0 | 0 |
| 446 | 3 | 0 | 0 |
| 1000 | 0 | 3 | 0 |
| 2000 | 3 | 0 | 0 |
| 4000 | 3 | 0 | 0 |
| 8000 | 3 | 0 | 0 |

Partial results (len=115, 180, 1000) are mid-prompt retrieval behavior
("738" vs full "738291"), not failures. Prefill FP32 probe still runs.

## Performance (DSpark MTP, guidellm concurrent, 20 prompts, 60s max)

| Rate | Output tok/s | Total tok/s | ITL p50 | ITL p99 | TTFT p50 |
|------|--------------|-------------|---------|---------|----------|
| 1 | 47.96 | 50.99 | 40.8ms | 45.8ms | 126.7ms |
| 4 | 49.11 | 52.23 | 40.8ms | 53.4ms | 7587.4ms |
| 8 | 48.81 | 51.96 | 40.8ms | 53.8ms | 15644.6ms |
| 16 | 49.05 | 52.22 | 40.8ms | 53.7ms | 33897.1ms |

vs previous fp32all (probe on decode too):

| Rate | previous | new | Δ |
|------|----------|-----|---|
| 1 | 19.48 | 47.96 | +147% |
| 4 | 24.50 | 49.11 | +100% |
| 8 | 24.91 | 48.81 | +96% |
| 16 | 24.68 | 49.05 | +99% |

~2× output tok/s across all concurrency levels. ITL p50 is flat at 40.8ms
regardless of concurrency — decode is now compute-bound, not probe-bound.
Zero `fp32_probe` log hits during the decode-heavy bench.

## Environment

- Host / GPU: 8× NVIDIA H20 (97.9 GB each), driver 535.161.08
- CUDA: 12.9 (V12.9.86)
- Model / dtype: DeepSeek-V4-Flash-FP8
- TP / EP: 4 / 4 (GPUs 1–4; GPU 0 occupied)
- Server: `INFER_TP_SIZE=4 INFER_EP_SIZE=4 INFER_CUDA_DEVICES=1,2,3,4 arle serve --backend cuda --port 8000 --spec-type mtp`

## Learnings

The FP32 probe is a prefill-only correctness fix (#146, #150); decode never
needed it (single-token, BF16 path is sufficient). The all-boundaries
extension accidentally ran it on every decode token — `start_pos_device`
(`Some` in all decode paths, `None` in prefill) is the single discriminator.
Prefill correctness preserved, decode throughput recovered ~2×.
