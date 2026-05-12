# FP8 KV quantize thread grouping kill

## Context

User focus: FP8 / FP4 quantization operators and KV quantization operators.
After the Qwen3.5 FP8 KV refill and decode-load wins, the next live FP8 KV
candidate was the write path:

```cpp
quantize_paged_kv_fp8_kernel
```

This kernel quantizes BF16 K/V from the paged HND work buffer into the durable
FP8 E4M3 NHD paged pool with one FP32 scale per token/head. Qwen3.5 serving
calls it for K and V when `--kv-cache-dtype fp8` is enabled.

A component bench was added first:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/quantize_paged_kv_fp8_qwen35 --quiet
```

Bench shape:

| Param | Value |
|---|---:|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER |
| SM | 89 |
| batch_size | 8 |
| num_kv_heads | 4 |
| head_dim | 256 |
| kv_dim | 1024 |
| token rows | 8 |
| KV format | BF16 work buffer -> FP8 E4M3 + f32 scale |

## Root Cause

The hypothesis was that `head_dim=256` uses one thread per dim, so reducing
threads per token/head and letting each thread process 2 or 4 dim values might
reduce reduction overhead.

Only the thread-to-dim grouping changed in each treatment; the FP8 format,
scale formula, pool layout, and Rust dispatch remained unchanged. Both runtime
treatments were reverted after measurement.

Measured component A/B:

| Arm | Criterion time | Throughput | Verdict |
|---|---:|---:|---|
| baseline 1 dim/thread | `6.6840-6.7050 us`, point `6.6920 us` | `1.2241 Gelem/s` | baseline |
| 2 dims/thread | `6.6804-6.6857 us`, point `6.6827 us` | `1.2259 Gelem/s` | kill: tiny/overlapping |
| 4 dims/thread | `6.8662-6.8917 us`, point `6.8783 us` | `1.1910 Gelem/s` | kill: regression |

Final rerun after reverting the runtime kernel: `6.6931-6.7064 us`, point
`6.7008 us`, throughput `1.2225 Gelem/s`.

The 2-dim treatment is only `-0.14%` by point estimate and overlaps the
baseline interval. The 4-dim treatment is a clear `+2.78%` regression. This
does not license a hot-path quantization rewrite.

The likely mechanism is that this small per-step write kernel is dominated by
launch and conversion/reduction fixed costs, while reducing active lanes cuts
parallelism enough to erase any shared-reduction savings. This mechanism is a
hypothesis; the measured overlap/regression is the decision evidence.

## Fix

Killed and reverted the runtime thread-grouping changes. Keep the current
one-dim-per-thread FP8 KV quantize kernel.

The new `ops_cuda/quantize_paged_kv_fp8_qwen35` bench remains as a measurement
harness for future FP8 KV write-path experiments.

## Rule

Do not optimize FP8 KV quantize by reducing threads per head_dim without a
clear component win. For Qwen3.5 `head_dim=256`, the measured 2-dim grouping is
noise and 4-dim grouping regresses; future attempts need a different mechanism
such as fewer launches, fused K/V writes, or a serving-level trace proving this
kernel is material in wall-clock.
