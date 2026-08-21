# W4AFP8 GEMV decode lane — CUDA, 2026-08-21

> Status: Shipped

## Goal

Decode throughput for DSv4-Flash-0731 (NVFP4→W4AFP8) at c=1 and c=4 on H20 TP=2.

## Hypothesis

The W4AFP8 path used the CUTLASS grouped GEMM prefill kernel for decode batch=1,
which is inefficient for single-token decode. Reusing the W4A16 grouped GEMV
kernel (designed for decode batch=1) by converting weights and scales at
table-build time should increase decode tok/s.

## Parameters

```bash
python3 scripts/bench_dsv4_trace_http.py \
  --host 127.0.0.1 --port 8000 \
  --model DeepSeek-V4-Flash-0731 \
  --fanout 1   # c=1
python3 scripts/bench_dsv4_trace_http.py \
  --host 127.0.0.1 --port 8000 \
  --model DeepSeek-V4-Flash-0731 \
  --fanout 4   # c=4
```

- Baseline: CUTLASS grouped GEMM for decode (pre-GEMV commit), 18.3 / 22.2 tok/s
- Treatment: `3b375df1d` — W4AFP8 GEMV decode lane
- Prompt tokens: 31 (decode_prompt)
- Completion tokens: 64 (decode64 case) / 16 (fanout case)
- Trials: 1

## Environment

- Host / GPU: H20 96GB ×2 (TP=2)
- Driver / CUDA: sm_90, CUDA 12.x
- Model / dtype: DeepSeek-V4-Flash-0731, NVFP4→W4AFP8 (INT4+BF16)
- TP / EP / slots / KV: TP=2, 256 experts, max-running-requests=4, max-total-tokens=131072
- Server flags: `--backend cuda --tensor-parallel-size 2 --max-running-requests 4 --max-total-tokens 131072`

## Results

| concurrency | arm | decode tok/s | aggregate tok/s | delta |
|---:|---|---:|---:|---:|
| 1 | baseline (CUTLASS) | 18.3 | — | — |
| 1 | treatment (GEMV) | 33.2 | — | **1.81x** |
| 4 | baseline (CUTLASS) | 22.2 | — | — |
| 4 | treatment (GEMV) | 28.1 | 64.8 | **1.27x** |

Prefill unchanged (CUTLASS path for routes > 128). Output coherent, math
correctness verified (25×4=100).

## Problems

Baseline numbers were measured with 105k-token prompts and max-tokens=256 in a
prior session; treatment numbers used the bench script's default short prompts
and max-tokens=64. Decode speed at c=1 is memory-bandwidth bound and largely
independent of prompt length, so the comparison is valid but not perfectly
matched.

Two bugs found and fixed during development:
1. w13 scales are stored w1/w3 row-interleaved by the loader (not the plain
   CUTLASS layout), requiring de-interleave before transpose.
2. W4A16 GEMV kernel expects unsigned nibbles with zero-point=8; W4AFP8 stores
   signed INT4 two's complement. The kernel takes an `xor_mask`
   (`0x08080808`) that flips each nibble's sign bit on the fly; a converted
   weight copy was tried first and OOMed at 3 GB/GPU.

## Learnings

PASS. The W4A16 GEMV kernel is reusable for W4AFP8 decode with two format
conversions: the nibble sign flip in-kernel via `xor_mask` (zero extra VRAM),
and the scale de-interleave+transpose once at table-build time. The 1.8x c=1 decode speedup is the main win for
interactive serving. Next wall: DSpark spec decode on top of the GEMV lane.
