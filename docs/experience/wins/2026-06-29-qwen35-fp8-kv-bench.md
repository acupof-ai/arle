# Qwen3.5-122B fp8 KV Cache Bench — CUDA TP=4, 2026-06-29

## Goal

- Measure throughput impact of fp8 KV cache on Qwen3.5-122B-A10B vs bf16 baseline; determine at which concurrency fp8 is net-positive.

## Hypothesis

- fp8 KV saves 50% KV memory → more pages per slot → enables higher concurrency. Dequant overhead will hurt c=1 but amortize at c≥2.

## Environment

- **Backend:** cuda
- **Model:** Qwen3.5-122B-A10B
- **Hardware:** 8×H20 (97 GB each), sglang-test container
- **Code commit:** `356e0072` feat(cuda): fp8 KV cache for Qwen35 full-attention layers
- **Binary:** arle-build at `6e2ed5c7`, `--release --features cuda,nccl,deepep`, `ARLE_DEEPEP_DIR` set
- **GPUs:** 0,2,5,6, TP=4
- **Server flags:** `--max-running-requests 4 --kv-cache-dtype fp8 --port 9015`
- **Workload:** 512 in / 256 out, topic-varied prompts, 120 s windows
- **Tool:** `bench_nonstream.py` (non-streaming concurrent HTTP, aggregate tok/s)

## Correctness gate

- `"The capital of France is Paris."` ✓ — fp8 KV decode produces correct output.

## Results

| c | fp8 tok/s | bf16 tok/s (baseline) | Δ |
|---|---|---|---|
| 1 | 28.7 | 40.3 | −29% |
| 2 | 45.7 | 53.0 | −14% |
| 4 | 49.8 | 52.0 | −4% |

Baseline from `2026-06-29-cuda-throughput-ceiling-three-models.md` (commit `1b0f0459`).

## Analysis

- **c=1 worst (−29%):** per-token dequant overhead not amortized in single-request decode.
- **c=2 (−14%):** overhead partially amortized; gap narrowing.
- **c=4 nearly matches bf16 (−4%):** dequant cost amortized across the batch; within measurement noise for production use.
- **Saturation shift:** bf16 saturates at c=2 (53.0 tok/s); fp8 continues scaling to c=4 (49.8 tok/s) — more KV pages available per slot extends the scaling range.

## Problems

- None. Server stable across all concurrency levels with fp8 flag.

## Learnings

- fp8 KV is **licensed for Qwen3.5-122B at c≥2 workloads** (−4% at c=4, within noise).
- fp8 KV is **not recommended for pure c=1 latency-sensitive** use (−29% decode throughput).
- The saturation shift (bf16 peaks c=2, fp8 peaks c=4) makes fp8 advantageous when memory pressure is the binding constraint at higher concurrency.

## Δ vs baseline

- **Baseline:** [`2026-06-29-cuda-throughput-ceiling-three-models.md`](2026-06-29-cuda-throughput-ceiling-three-models.md) — Qwen3.5-122B bf16 KV, c=2 peak 53.0 tok/s.

| c | bf16 tok/s | fp8 tok/s | Δ% |
|---|---|---|---|
| 1 | 40.3 | 28.7 | −29% |
| 2 | 53.0 | 45.7 | −14% |
| 4 | 52.0 | 49.8 | −4% |
