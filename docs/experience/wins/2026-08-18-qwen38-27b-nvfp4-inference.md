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

The NVFP4 MLP GEMV dominates: 82.4% of forward time. Per-layer weight read is
84.9 MB (gate_up 50.3 MB + down 25.2 MB packed FP4, plus 9.4 MB of FP8 group
scales) in 1.595 ms = 53 GB/s = **1.3% of H20's 4.0 TB/s bandwidth**. The
kernel loads one packed byte per thread per iteration, so a 128 B cacheline is
fetched per 1 B used — transaction-bound, not bandwidth-bound.

SGLang's Marlin W4A16 fallback achieves ~24% bandwidth on Hopper at B=1, but
requires BF16 weights (4× memory = 88 GB — doesn't fit on single H20 after
KV cache). The optimization path is vectorized loads in the FP4 GEMV, not
dequant-to-BF16.

## SOTA comparison

| Engine | Approach | H20 decode tok/s | Notes |
|---|---|---:|---|
| ARLE | FP4 GEMV (byte loads) | 9.3 | 1.3% bandwidth |
| SGLang | Marlin W4A16 (dequant→BF16) | N/A | 24% bandwidth, 88 GB weights |
| vLLM | Marlin W4A16 fallback | N/A | Occupancy-bound at M=1 |

ARLE's 9.3 tok/s is the only single-GPU NVFP4 result measured. At 1.3% of
bandwidth the FP4 GEMV is bound by transaction count, which vectorized loads
address directly.

## FP4 GEMV vectorization + dequant prefill path — measured

Runtime `2a3a2164f` (kernel `d6873c5e6`, dequant path `a23539905`), same pod,
same H20, `--kv-cache-dtype fp8`.

The three FP4 group GEMV kernels read one packed byte per thread per iteration,
so a 128 B cacheline was fetched per 1 B used. All three now share
`fp4_e2m1_row_dot`, which loads 16 B (uint4, 32 weights) per transaction and
hoists the per-group scale out of the per-element path. Separately, NVFP4
prefill gained a dequant-to-BF16 + cuBLAS arm
(`dequantize_fp4_e2m1_group_to_bf16_cuda`), matching what FP8 / W8A16 / W4A16
already had.

Also in this range: `try_fp8_dequant_bf16_gemm_batch` and its W8A16 twin turned
a failed BF16 scratch allocation into a hard error, which killed the server at
c=4 in the run above once the KV (30.2 GB) and recurrent (37.6 GB) pools had
committed the VRAM. Both now fall back to the scalar GEMV.

### GPU-side, c=1 decode, ARLE_CUDA_PROFILE=1 (400 steps vs the baseline's 50)

| | before | after | Δ |
|---|---:|---:|---:|
| `dense_ffn` per step | 102.1 ms | 86.2 ms | **−15.5%** |
| `forward_hidden` per step | 123.9 ms | 106.7 ms | **−13.9%** |
| `dense_ffn` bandwidth | 53 GB/s (1.33%) | 63 GB/s (1.58%) | +19% |
| `dense_ffn` share of forward | 82.4% | 80.8% | — |

The two deltas track: the GEMV is 81% of the forward, so a 15.5% kernel win
lands as a 13.9% forward win.

### End to end

| c | before | after |
|---:|---:|---:|
| 1 | 9.3 tok/s | 9.3 tok/s |
| 4 | 8.3 tok/s | 14.1 tok/s |
| 8 | 9.2 tok/s | 26.5 tok/s |

c=1 is unchanged in the harness number even though the forward is 13.9% faster:
at one concurrent request the reported figure includes TTFT and scheduling, so
the GPU win does not surface. c=4 and c=8 move a lot, but they are not a clean
comparison — those cells previously died on the dequant-scratch OOM, so the
"before" figures are a crashing server, not a slower one.

Correctness: needle ladder 512 / 4096 × 3 = **6/6 exact, deterministic**
(`NEEDLE_MAX_TOKENS=512` — the gate's default of 16 is below this reasoning
model's thinking budget and yields empty completions at any code revision).
This is the gate that matters here, since the prefill path is the one that
changed.

Raw artifacts on the pod: `/tmp/nvfp4_v5_bench.json`, `/tmp/decode_only.json`.

### What is still open

At 1.58% of bandwidth the GEMV is still far from memory-bound. The uint4 load
removed the transaction-count problem but the kernel is now limited by
something else — per-element LUT decode and the scalar FMA chain are the
candidates. A prior ILP=4 attempt (four independent accumulators, no change to
transaction count) measured as a wash, which is what pointed at transactions in
the first place.
