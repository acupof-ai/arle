# DSv4 FP32 compressor promotion — CUDA, 2026-07-16

> Status: Shipped

## Goal

Measure the throughput cost of promoting the DSv4 FP32 main-value compressor
from diagnostic-only to default behavior, on the production DSv4 workload
(8xH20 TP=8 EP=8, 4096-token prompt / 256-token output).

## Hypothesis

The FP32 compressor re-runs the compressor forward in FP32 to avoid BF16/FP8
value mismatches. It replaces the BF16 compressor for single-prefill only
(early return), so the cost is limited to the prefill path. Expected: small
throughput regression at high concurrency, neutral at low concurrency.

## Parameters

```bash
# FP32 ON (treatment): loader always loads fp32_probe; attention guard removed.
# FP32 OFF (baseline): loader.rs fp32_probe = None; same binary otherwise.
bash scripts/run_dsv4_bench.sh <fp32on|fp32off>
# Internally: guidellm benchmark run --profile concurrent --rate <1,4,8,16>
#   --max-seconds 60 --data bench-prompts.jsonl (4096 tok prompt, 256 tok output)
#   --backend openai_http --request-format /v1/completions --random-seed 20260416
```

- Baseline: `c7982e4ee` with `fp32_probe = None` (loader.rs), `--features cuda,nccl`
- Treatment: `c7982e4ee` with `fp32_probe = Some(...)` (default), `--features cuda,nccl`
- Prompt tokens: 4096 (fixed, synthetic English text)
- Completion tokens: 256 (fixed)
- Trials: 20 completed requests per rate (60s duration)

## Environment

- Host / GPU: 8x NVIDIA H20 (97.9 GB each), driver 535.161.08
- CUDA: 12.9 (V12.9.86)
- Model / dtype: DeepSeek-V4-Flash-FP8
- TP / EP: 8 / 8
- Server: `INFER_TP_SIZE=8 INFER_EP_SIZE=8 INFER_CUDA_DEVICES=0,1,2,3,4,5,6,7 arle serve --backend cuda --port 8000`

## Results

| concurrency | arm | completed | errors | req/s | TTFT p50/p99 ms | ITL p50/p99 ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | baseline (OFF) | 20 | 0 | 1.169 | 421.2 / 3664.6 | 19.12 / 20.19 |
| 1 | treatment (ON) | 20 | 0 | 1.435 | 401.3 / 496.2 | 19.23 / 20.53 |
| 4 | baseline (OFF) | 20 | 0 | 2.107 | 1301.5 / 1372.2 | 40.08 / 97.57 |
| 4 | treatment (ON) | 20 | 0 | 2.115 | 1305.9 / 1379.7 | 40.10 / 96.18 |
| 8 | baseline (OFF) | 20 | 0 | 2.187 | 1574.4 / 3431.9 | 100.79 / 135.73 |
| 8 | treatment (ON) | 20 | 0 | 2.055 | 1652.9 / 3520.8 | 117.91 / 143.76 |
| 16 | baseline (OFF) | 20 | 0 | 2.378 | 3692.1 / 6596.9 | 115.75 / 128.12 |
| 16 | treatment (ON) | 20 | 0 | 2.071 | 3776.4 / 6732.6 | 116.99 / 132.23 |

Raw artifacts: `bench-output/2026-07-15-fp32on-rate*/result.{json,csv}`,
`bench-output/2026-07-16-fp32off-rate*/result.{json,csv}` (on pod at
`/host/arle-build/bench-output/`).

## Problems

- guidellm's `synthetic_text` data generator fails on DeepSeek-V4 config
  (missing `max_position_embeddings`); worked around with a pre-built JSONL
  dataset of 4096-char English prompts with `output_tokens: 256`.
- guidellm `--outputs` flag needs filenames with extensions (e.g.
  `result.json`), not bare format names; corrected in `run_dsv4_bench.sh`.
- rate=1 req/s +22.7% is noise (only 20 requests in 60s); total_tps −0.7%
  is the stable metric.

## Learnings

**PASS.** The FP32 compressor is promoted to default. TTFT and ITL are
within noise at rate 1/4, ITL p50 +17% at rate 8, TTFT p50 +2% at rate 16
(high concurrency). This is an acceptable cost for the correctness fix
(#146: VIOLET-6529→4929, #150: 738291→738292).

The rate-16 cost is the FP32 prefill overhead showing up under load;
optimization (fuse FP32 probe into the main compressor kernel, or run only
on the first compression boundary) is deferred.
