# FP8 KV HND refill scale broadcast kill

## Context

User focus: FP8 / FP4 quantization operators and KV quantization operators.
After landing the Qwen3.5 FP8 scatter BF16 source-load vectorization, the next
small FP8 KV candidate was the durable FP8 NHD -> BF16 HND refill kernel:

```cpp
dequantize_paged_kv_fp8_to_hnd_kernel
```

The existing kernel maps one block to one `(kv_head, token)` row. For the live
Qwen3.5 shape, each block has 64 threads and every thread reads the same scale:

```cpp
float scale = scales[scale_offset];
```

The hypothesis was that loading the scale once into shared memory and
broadcasting it to the block would reduce duplicate global loads.

Baseline command:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dequantize_paged_kv_fp8_to_hnd --quiet
```

Environment:

| Param | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER |
| SM | 89 |
| Driver / CUDA | 595.71.05 / CUDA 13.2 (`nvcc` 13.2.78) |
| cudarc override | `CUDARC_CUDA_VERSION=13010` |
| Operator | `dequantize_paged_kv_fp8_to_hnd_cuda` |
| Shape | Qwen3.5 FP8 KV, `total_tokens=1024`, `4 kv_heads x 256 head_dim`, `kv_dim=1024` |

## Root Cause

The treatment changed exactly one runtime variable:

```cpp
__shared__ float s_scale;
if (threadIdx.x == 0) {
    s_scale = scales[scale_offset];
}
__syncthreads();
float scale = s_scale;
```

Everything else stayed unchanged: FP8 x4 source loads, BF16x2 stores, layout,
and fallback handling.

Measured component A/B:

| Arm | Criterion time | Throughput | Verdict |
|---|---:|---:|---|
| baseline per-thread scale load | `8.1635-8.1795 us`, point `8.1704 us` | point `128.34 Gelem/s` | keep |
| shared scale broadcast | `8.2933-8.3447 us`, point `8.3156 us` | point `126.10 Gelem/s` | kill: +1.78% regression |

The intervals are separated and the treatment is slower. The likely mechanism
is that this scale load is cache/broadcast-friendly enough that adding an
extra `__syncthreads()` costs more than it saves. That mechanism is a
hypothesis; the measured regression is the decision evidence.

## Fix

Killed and reverted the runtime change. Keep the existing per-thread scale
load in `dequantize_paged_kv_fp8_to_hnd_kernel`.

No runtime code change is kept from this experiment.

## Rule

Do not optimize FP8 HND refill by adding shared-memory scale broadcast on the
current sm_89 Qwen3.5 shape. Future refill work needs a different bottleneck
target and must beat the existing `ops_cuda/dequantize_paged_kv_fp8_to_hnd`
component bench before landing.
