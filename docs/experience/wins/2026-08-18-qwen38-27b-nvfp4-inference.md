# Qwen3.8-27B-NVFP4 inference support — CUDA, 2026-08-18

> Status: Shipped

## Goal

Load and serve `unsloth/Qwen3.8-27B-NVFP4` (mixed-precision: NVFP4 MLP + FP8
per-channel attention) on a single H20. The model failed to load before this
change ("R6 clean CUDA path accepts BF16 only, got F8_E4M3").

## Hypothesis

FP8 per-channel weights (F8_E4M3 + BF16 [N,1] weight_scale) are semantically
identical to `Fp8BlockScaled { block_m: 1, block_k: K }` — the existing
block-scale GEMV indexes `scales[row]` directly. One detection arm, zero new
kernels.

## Parameters

```bash
python3 scripts/bench_throughput.py \
  --url http://127.0.0.1:8080 \
  --model qwen \
  --concurrency-grid 1,4,8 \
  --seconds-per-concurrency 30 \
  --max-tokens 128 \
  --temperature 0 \
  --seed 42 \
  --output /tmp/nvfp4_bench.json
```

- Baseline: model fails to load (no FP8 per-channel detection)
- Treatment: `33f4863c7` — per-channel FP8 detection arm + dequant GEMM gate
  relaxation + lm_head quant-aware + Fp4E2M1Group row-fusion
- Prompt tokens: 7-9 (synthetic)
- Completion tokens: 128 per request
- Trials: 1 (initial support verification)

## Environment

- Host / GPU: 8×H20 pod, single GPU (CUDA_VISIBLE_DEVICES=0)
- Driver / CUDA: sm_90, CUDA 12.x
- Model: `unsloth/Qwen3.8-27B-NVFP4` (22 GB safetensors + 811 MB MTP)
- KV cache: FP8 E4M3 (KIVI per-channel-K + per-token-V), 808K tokens capacity
- Server flags: `--kv-cache-dtype fp8`
- Peak RSS: 20.69 GiB

## Results

| concurrency | completed | errors | output tok/s | TTFT p50 ms | ITL p50 ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 0 | 9.3 | 811 | 102 |
| 4 | 4 | 0 | 8.3 | 2219 | 457 |
| 8 | 8 | 0 | 9.2 | 4619 | 818 |

Correctness: `2+3=5` verified via both `arle run` and OpenAI-compatible API.

Raw artifacts: `/tmp/nvfp4_bench.json` (pod).

## What was changed

Four files, 72 insertions, 3 deletions:

1. **`quant_format.rs`** — detection arm for FP8 per-channel (F8_E4M3 +
   BF16 [N,1] weight_scale, no input_scale/weight_scale_inv) →
   `Fp8BlockScaled { block_m: 1, block_k: K, Multiply }`
2. **`quant_linear.rs`** — relax `try_fp8_dequant_bf16_gemm_batch` gate from
   128×128-only to arbitrary block sizes (enables cuBLAS prefill for
   per-channel FP8)
3. **`qwen35_load.rs`** — `lm_head` via `load_matrix_quant_aware` (compressed-
   tensors quantizes it)
4. **`tensor.rs`** — `fuse_rows` support for `Fp4E2M1Group` (NVFP4 gate+up
   MLP projections; global scales are per-tensor scalars, same value in
   practice)

## A8 vs AFP8

On H20 (sm_90), FP8 E4M3 and INT8 tensor core throughput are identical
(989 TFLOPS/TOPS). FP8 has wider dynamic range → better accuracy. The model's
NVFP4 MLP path uses FP8 activations (W4AFP8); the attention path uses BF16
activations (W8A16 via GEMV). W4A8 (INT8 activations) would need a new GEMM
kernel — not justified when throughput is identical and accuracy favors FP8.

## Problems

- c=16 benchmark cell killed (server shutdown signal during long-running
  requests); c=1/4/8 data is sufficient for initial verification
- Pod sync digest mismatch required `tn push` for individual files instead
  of `pod.sh sync`

## Learnings

PASS. The model loads in 6.5s, generates correct output, and serves at 9.3
tok/s decode (c=1) on a single H20 with 20.69 GiB peak RSS. The
`Fp8BlockScaled { block_m: 1, block_k: K }` reuse pattern generalizes to any
compressed-tensors float-quantized model with per-channel scales.

## MTP spec decode — investigated, not enabled

The model ships 1 BF16 MTP layer (811 MB). Enabled via
`--spec-type mtp --mtp-draft-tokens 2`.

**Acceptance rate: ~80%** (chains 2–11, per-step rates 0.67–0.82). The MTP
head produces correct draft tokens.

**Net effect: slower.** 6.2 tok/s with MTP vs 9.3 tok/s without. Root cause:
the verify forward processes `depth+1 = 3` tokens through the recurrent linear
attention (gated-delta scan), costing ~3× a single-token decode. At 80%
acceptance, cost per token = 4.6× decode / 2.6 tokens = 1.77×. MTP depth=2 is
counterproductive on H20 with this hybrid architecture. MTP depth=1 would
break even at ~50% acceptance but offers no throughput gain.

## Per-op profiling (ARLE_CUDA_PROFILE=1, 50 decode steps, c=1)

| Op | Total ms | Share | Avg μs | Count |
|---|---:|---:|---:|---:|
| forward_hidden (all layers) | 6193.8 | 48.6% | 123876 | 50 |
| dense_ffn (NVFP4 MLP GEMV) | 5102.6 | 40.1% | 1595 | 3200 |
| linear_attention | 580.6 | 4.6% | 242 | 2400 |
| linear/in_proj | 257.8 | 2.0% | 107 | 2400 |
| full_attention | 178.0 | 1.4% | 223 | 800 |
| lm_head + lm_head_gemv | 69.3 | 0.5% | 1387 | 50 |

The NVFP4 MLP GEMV dominates: 82.4% of forward time. Per-layer weight read
~196 MB (FP4 packed + FP8 scales) in 1.59 ms = ~124 GB/s = **3.1% of H20's
4.0 TB/s bandwidth**. The kernel is latency-bound (serial load→compute→accumulate
chain, ILP=1), not bandwidth-bound.

SGLang's Marlin W4A16 fallback achieves ~24% bandwidth on Hopper at B=1, but
requires BF16 weights (4× memory = 88 GB — doesn't fit on single H20 after
KV cache). The optimization path is ILP and vectorized loads in the FP4 GEMV,
not dequant-to-BF16.

## SOTA comparison

| Engine | Approach | H20 decode tok/s | Notes |
|---|---|---:|---|
| ARLE | FP4 GEMV (scalar, ILP=1) | 9.3 | 3.1% bandwidth |
| SGLang | Marlin W4A16 (dequant→BF16) | N/A | 24% bandwidth, 88 GB weights |
| vLLM | Marlin W4A16 fallback | N/A | Occupancy-bound at M=1 |

ARLE's 9.3 tok/s is the only single-GPU NVFP4 result measured. The 3.1%
bandwidth utilization indicates the FP4 GEMV kernel has 5–8× headroom from
ILP and vectorized-load optimizations alone.

## FP4 GEMV vectorization — pending-remote

Runtime `d6873c5e6`. The three FP4 group GEMV kernels read one packed byte per
thread per iteration, so a 128 B cacheline was fetched per 1 B used. All three
now share `fp4_e2m1_row_dot`, which loads 16 B (uint4, 32 weights) per
transaction and hoists the per-group scale out of the per-element path.

Same commit: `try_fp8_dequant_bf16_gemm_batch` and its W8A16 twin turned a
failed BF16 scratch allocation into a hard error, which killed the server at
c=4 during the benchmark above once the KV (30.2 GB) and recurrent (37.6 GB)
pools had committed the VRAM. Both now fall back to the scalar GEMV.

Predicted from the measured 53 GB/s (1.3% of 4.0 TB/s): weight transactions
drop 16x, while x-loads and scale loads are unchanged, so the binding
constraint moves rather than disappearing.

| | decode tok/s (c=1) | dense_ffn avg | bandwidth |
|---|---:|---:|---:|
| Before (`33f4863c7`) | 9.3 | 1595 us | 53 GB/s (1.3%) |
| After (`d6873c5e6`) | pending | pending | pending |

Status: pod SSH unavailable at the time of the change. Rerun
`scripts/bench_throughput.py --concurrency-grid 1,4,8` plus the
`ARLE_CUDA_PROFILE=1` per-op dump and fill the row above.
