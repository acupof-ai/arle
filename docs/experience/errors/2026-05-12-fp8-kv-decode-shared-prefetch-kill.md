# FP8 KV decode shared-prefetch kill

## Context

User focus: FP8 / FP4 quantization operators and KV quantization operators.
After the Qwen3.5 FP8 KV HND refill pairwise-store win, the next FP8 KV
decode hypothesis was:

> `decode_attention_fp8_partial_kernel` still reads FP8 K/V directly from
> global memory, while the INT8 variant preloads 16-token pages into shared
> memory with `cp.async`. Porting the same tiled prefetch structure to FP8 may
> reduce global-load latency for Qwen3.5 `head_dim=256` decode attention.

A Qwen3.5-shaped component bench was added first:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/decode_attention_fp8_qwen35 --quiet
```

Bench shape:

| Param | Value |
|---|---:|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER |
| SM | 89 |
| batch_size | 4 |
| seq_len | 4096 |
| num_q_heads | 16 |
| num_kv_heads | 4 |
| head_dim | 256 |
| page_size | 16 |
| KV format | FP8 E4M3 + f32 per-token/head scales |

## Root Cause

The treatment changed exactly one runtime variable: FP8 decode K/V reads used
the same 16-token shared-memory double buffer and `__pipeline_memcpy_async`
structure as the INT8 kernel. It was then reverted.

Measured component A/B:

| Arm | Criterion time | Throughput |
|---|---:|---:|
| baseline direct global FP8 reads | `100.40-100.56 us`, point `100.51 us` | `667.69 Gelem/s` |
| treatment shared-tile prefetch | `134.17-134.47 us`, point `134.35 us` | `499.50 Gelem/s` |

Delta: **+33.7% latency regression**. The intervals are tight and fully
separated, so this is a clear kill, not noise.

The likely mechanism is that the FP8 path's per-lane direct byte loads are
already cheap enough for this shape, while the shared-memory path adds
pipeline, synchronization, and shared-memory traffic overhead without the
INT8 path's same bottleneck profile. This mechanism is still hypothesis; the
measured regression is the decision evidence.

## Fix

Killed and reverted the FP8 shared-prefetch runtime change. Keep the existing
direct-global-load FP8 decode kernel.

The new `ops_cuda/decode_attention_fp8_qwen35` microbench remains as a
measurement harness for future FP8 KV decode experiments. Any next FP8 decode
attempt should use this bench for baseline/treatment before touching serving
paths.

## Rule

Do not mechanically transfer a memory-staging optimization from INT8 KV to FP8
KV. Same kernel shape does not mean same bottleneck. For FP8 decode attention,
require a Qwen3.5-shaped component A/B with tight separated intervals before
landing any load-path or staging change.
