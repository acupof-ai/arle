# DSv4 Decode — Throughput Profile & Optimization Plan

> Status: Active
> Date: 2026-07-07
> Env: 4×H20, TP4/EP4, DeepSeek-V4-Flash-FP8, FlashMLA on, NCCL (no DeepEP)

## 1. Measured Throughput

| c | agg tok/s | per_req tok/s | step ms |
|---|-----------|---------------|---------|
| 1 | 20.7 | 26.3 | 38 |
| 2 | 30.2 | 15.2 | 66 |
| 4 | 47.3 | 11.9 | 84 |
| 8 | 61.6 | 7.7 | 130 |
| 16 | 87.8 | 6.7 | 149 |

c=32 OOM. 105 slots/GPU cap at max_seq_len=128K.

Parallel efficiency at c=16: 21%. Sub-linear because per-expert tokens ≪ 1.

## 2. The Structural Constraint

EP4 → 64 local experts/rank. top_k=6. Avg tokens/expert = 6B/64 = **0.094B**.

| c | tokens/expert | regime |
|---|---------------|--------|
| 1 | 0.09 | pure GEMV |
| 16 | 1.5 | GEMV |
| 43 | 4.0 | amortization starts |
| 128 | 12.0 | GEMM |
| 171 | 16.0 | DeepGEMM threshold (1024 routes) |

Weight amortization needs ≥4 tokens/expert = ≥43 concurrent requests.
Current cap: c=16. **The bottleneck is slot count, not kernel speed.**

HBM floor at B=1: ~0.07ms. Actual: 38ms. 540× gap = GEMV latency, not bandwidth.

## 3. Decode Time Breakdown (B=1, code-path estimate)

| Category | ms | % |
|----------|-----|---|
| Dense proj FP8 GEMV (wqkv_a, wq_b, wo_b, comp) | 10-12 | 28-32% |
| wo_a BF16 cuBLAS (per-group loop, 8 groups) | 3-4 | 9-11% |
| MoE expert (grouped SwiGLU + down) | 8 | 21% |
| NCCL (allreduce + allgather) | 7 | 18% |
| FP8 KV write (block_scaled_to_fp8) | 2.6 | 7% |
| FlashMLA attention | 0.6 | 1.5% |
| Router BF16 cuBLAS | 1 | 3% |
| DeepGEMM (wq_b, wo_b — FP8 with cache) | 1-2 | 3-5% |
| Launch + misc | 3 | 8% |

**nsys `gemv_handwritten_kernel` 21.9% = prefill contamination.**
Decode-only: `ops::gemv` BF16 called only by lm_head (once/token).

## 4. Key Facts (confirmed)

- **KV already FP8 packed**: 584 bytes/token. NoPE 448d FP8 + RoPE 64d BF16.
- **wo_a is BF16** in this checkpoint (`/host/DeepSeek-V4-Flash-FP8` serializes the tiny low-rank as dense BF16). No DeepGEMM cache → cuBLAS per-group loop.
- **wq_b, wo_b are FP8** with DeepGEMM caches. Used in decode.
- **Router is BF16**: `gemm_batch` → cuBLAS. Explicitly not FP8.
- **FlashMLA = 1.5%**: attention compute is not the bottleneck.

## 5. KV Slot Budget

Each slot: `sw_blocks × 64 × 584 × 43 layers`. `sw_blocks = ceil(max_seq_len/128)`.

| max_seq_len | per-slot KV | max slots | safe c | tokens/expert |
|-------------|-------------|-----------|--------|---------------|
| 128K | 1.65 GB | 105 | 16 | 1.5 (GEMV) |
| 32K | 0.41 GB | ~420 | 128 | **12 (GEMM)** |
| 16K | 0.21 GB | ~840 | 256 | 24 (GEMM) |

## 6. Optimization Priority

| # | What | Gain | Effort |
|---|------|------|--------|
| 1 | **`--max-seq-len 32768 --num-slots 256`** | 4-6× agg tok/s | S (flag) |
| 2 | **wo_a BF16→FP8 quant → DeepGEMM** | 1.3-1.5× dense | M (loader+kernel) |
| 3 | **DeepEP → NCCL 18%→5%** | 1.15× | M (feature) |
| 4 | **RoPE BF16→FP8 in KV** | 1.12× slots | M (pack+FlashMLA) |
| 5 | Router BF16→FP8 | 1.03× | S |
| 6 | FP4 NoPE latent | 1.46× slots | XL, high risk |

#1 is zero-risk zero-effort. Run it first to measure actual c scaling before kernel investment.

## 7. KV Quantization Detail

### RoPE BF16→FP8 (11% raw savings → ~520 bytes/token)
- Risk: RoPE position precision at 128K. Mitigate: per-channel scale, needle gate at 128K.
- Files: `kv_layout.rs` pack/unpack, `block_scaled_to_fp8` kernel, FlashMLA dequant.

### FP4 NoPE latent (38% raw savings → DEFER)
- Risk: latent distribution may not survive FP4. Must dump + simulate first.
- Only if max_seq_len=128K is mandatory.

## 8. Next Steps

1. **Phase 0**: `arle serve --max-seq-len 32768 --num-slots 256` → c=128 throughput
2. **Decode-only nsys**: 30s steady-state c=8, no prefill in window
3. **wo_a FP8 quant**: loader + DeepGEMM cache build
4. **DeepEP enable**: `cargo build --features cuda,nccl,deepep`
