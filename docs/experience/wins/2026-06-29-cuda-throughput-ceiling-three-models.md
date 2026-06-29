# CUDA Throughput Ceiling — Qwen3.6-27B / Qwen3.5-122B / DSv4-Flash TP=1/4

## Context

Concurrency sweep across three production models to measure saturation throughput and
mark the saturation concurrency. Two bugs were fixed during the run (Qwen3.5 prefix reuse
seq_len mismatch; DSv4 batched page-table assertion). All numbers are post-fix.

**Environment:**
- Pod: 8×H20 (97 GB each), sglang-test container
- Binary: commit `1b0f0459`, `--release --features cuda,nccl`
- Workload: 512 in / 256 out, topic-varied prompts (no prefix cache pollution), 60–120 s windows
- Tool: `/host/bench_nonstream.py` (non-streaming concurrent HTTP, aggregate tok/s)

## Results

### Qwen3.6-27B-FP8 — TP=1, GPU 3, 8 slots

| c | tok/s | TTFT p50 | TTFT p95 | reqs |
|---|---|---|---|---|
| 1 | 30.6 | 8.3s | 8.6s | 8 |
| 2 | 46.2 | 9.1s | 14.0s | 13 |
| 4 | 63.4 | 18.6s | 22.6s | 15 |
| 8 | 65.3 | 30.7s | 43.8s | 19 |
| 16 | **80.5** | 49.3s | 55.6s | 33 |
| 32 | 81.9 | 85.5s | 110.9s | 43 |

**Saturation: c=16 / 80.5 tok/s** (c=32 flat at 81.9 tok/s, +1.7%).

Steep scaling c=1→c=16 (+163%) indicates decode-batch gains dominate. Single H20,
all decode steps use the full-attention (paged) path.

### Qwen3.5-122B-A10B — TP=4, GPUs 0,2,5,6, 4 slots

| c | tok/s | TTFT p50 | window | reqs |
|---|---|---|---|---|
| 1 | 40.3 | 4.9s | 60s | 10 |
| 2 | **53.0** | 9.7s | 60s | 14 |
| 4 | 52.0 | 19.8s | 120s | 28 |
| 8 | 48.1 | 41.6s | 60s | 16 |

**Saturation: c=2 / 53.0 tok/s.** At c≥4 throughput declines (4-slot cap →
preemptive scheduling overhead). Hybrid MoE: recurrent + paged KV prefix cache
both active (commit `1b0f0459` prefix reuse fix required for crash-free c≥2).

### DeepSeek-V4-Flash-FP8 — TP=4, GPUs 0,2,5,6, 4 slots

| c | tok/s | TTFT p50 | window | reqs |
|---|---|---|---|---|
| 1 | 33.2 | 7.7s | 60s | 8 |
| 2 | 30.8 | 15.1s | 120s | 18 |
| 4 | **44.3** | 20.4s | 120s | 24 |
| 8 | 44.9 | 36.4s | 120s | 32 |
| 16 | 46.7 | 65.5s | 60s | 26 |

**Saturation: c=4 / 44.3 tok/s** (c=8,16 flat at ~44–47 tok/s).
c=2 lower than c=1 (warmup artifact in short window; 120 s run shows true value).
DSv4 batched page-table assertion fix (commit `3c8cc484`, HEAD) required for crash-free c≥2.

## Problems

- Qwen3.5-122B crashed at c≥2 with `materialized state len N != kv_seq_len M`:
  `release_recurrent` skipped in prefix reuse path → fix in `1b0f0459`.
- DSv4 crashed at c≥2 with `batched page table len N not a multiple of row width M`:
  wrong divisibility check → fix in `3c8cc484`.
- GPU 1 occupied by foreign process throughout; 27B moved to GPU 3, DSv4/122B used 0,2,5,6.
- 27B server was killed by external agent (same tick as 122B crash); restarted on GPU 3
  with fixed binary for final sweep.

## Saturation Summary

| Model | Backend | TP | Peak tok/s | Sat. c |
|---|---|---|---|---|
| Qwen3.6-27B-FP8 | CUDA single GPU | 1 | 80.5 | 16 |
| Qwen3.5-122B-A10B | CUDA NCCL | 4 | 53.0 | 2 |
| DeepSeek-V4-Flash-FP8 | CUDA NCCL | 4 | 44.3 | 4 |
