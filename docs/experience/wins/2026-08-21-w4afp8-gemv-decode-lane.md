# W4AFP8 GEMV decode lane — CUDA, 2026-08-21

> Status: Shipped

## Goal

Decode throughput for DSv4-Flash-0731 (NVFP4→W4AFP8) at c=1/4/8/16 on H20,
TP=2 and TP=4.

## Implementation

Reuses the W4A16 grouped GEMV kernel for W4AFP8 decode. Two format conversions:

1. **Nibble sign flip in-kernel** — W4A16 GEMV expects unsigned nibbles with
   zero-point=8 (`value = (nibble - 8) * scale`); W4AFP8 stores signed INT4
   two's complement. The kernel takes an `xor_mask` parameter (`0x08080808`
   for W4AFP8, `0` for W4A16) that flips each nibble's sign bit on the fly.
   Zero extra VRAM. A converted weight copy was tried first and OOMed at
   3 GB/GPU.
2. **Scale de-interleave + transpose** at table-build time (one-time, lazy).
   w13 scales are stored w1/w3 row-interleaved by the loader; w2 scales are
   not interleaved. Both are transposed from `[K//512, N*4]` CUTLASS layout
   to `[N, K//128]` row-major for the GEMV kernel.

Dispatch: W4AFP8 checkpoint, routes ≤ 128 → GEMV; routes > 128 → CUTLASS
(prefill band). No env var, no feature flag — single path.

## Parameters

```bash
python3 scripts/bench_throughput.py \
  --url http://localhost:8000 \
  --concurrency-grid 1,4,8,16 \
  --requests-per-concurrency 16 \
  --max-tokens 128 \
  --synthetic-prompts 8
```

- Baseline: CUTLASS grouped GEMM for decode (pre-GEMV), 18.3 tok/s at c=1
- Treatment: `521db0bed` — xor_mask GEMV (final, zero-VRAM)
- Prompt tokens: 8 (synthetic)
- Completion tokens: 128
- Trials: 16 requests per concurrency level

## Environment

- Host / GPU: H20 96GB ×8
- Driver / CUDA: sm_90, CUDA 12.x
- Model / dtype: DeepSeek-V4-Flash-0731, NVFP4→W4AFP8 (INT4+BF16)
- TP=2: `--tensor-parallel-size 2 --max-running-requests 8 --max-total-tokens 131072`, 88 GB/GPU
- TP=4: `--tensor-parallel-size 4 --max-running-requests 16 --max-total-tokens 131072`, 67 GB/GPU

## Results

### TP=2

| concurrency | decode tok/s (per-req) | aggregate tok/s |
|---:|---:|---:|
| 1 | 31.0 | 30.1 |
| 4 | 16.6 | 63.2 |
| 8 | 12.1 | 90.2 |
| 16 | 11.7 | 90.5 |

c=1 speedup vs baseline (18.3 tok/s): **1.69x**. Saturates at c=8
(max-running-requests=8); c=16 queues half the requests.

### TP=4

| concurrency | decode tok/s (per-req) | aggregate tok/s |
|---:|---:|---:|
| 1 | 41.1 | 39.8 |
| 4 | 22.5 | 85.8 |
| 8 | 17.0 | 127.2 |
| 16 | 11.0 | 161.7 |

TP=4 c=1 is 1.33x TP=2 (memory-bound GEMV; less work per GPU offset by NCCL
all-reduce). TP=4 still scaling at c=16: 161.7 tok/s aggregate, 1.79x the
TP=2 ceiling.

## Problems

Two bugs found and fixed during development:
1. w13 scales are stored w1/w3 row-interleaved by the loader (not the plain
   CUTLASS layout), requiring de-interleave before transpose.
2. W4A16 GEMV kernel expects unsigned nibbles with zero-point=8; W4AFP8 stores
   signed INT4 two's complement. Fixed with the `xor_mask` kernel parameter
   (zero VRAM overhead).

An earlier weight-copy conversion approach (3 GB/GPU) OOMed at TP=2 with
max-running-requests=16. The xor_mask approach has zero VRAM overhead.

## Learnings

PASS. The W4A16 GEMV kernel is reusable for W4AFP8 decode with two format
conversions: the nibble sign flip in-kernel via `xor_mask` (zero extra VRAM),
and the scale de-interleave+transpose once at table-build time. The 1.69x c=1
decode speedup is the main win for interactive serving. TP=4 scales to c=16
with 161.7 tok/s aggregate. Next wall: DSpark spec decode on top of the GEMV
lane.
