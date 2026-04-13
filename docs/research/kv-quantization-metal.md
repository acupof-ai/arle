# KV Cache Quantization on Metal / MLX Backend

_Date: 2026-04-14_

## Current State

### CUDA Backend (Already Done)
`infer/src/backend/cuda/paged_kv.rs` already has:
- `migrate_from_contiguous_fp8()` (line 743): FP16 → FP8 E4M3 KV conversion
- `ops::kv_quant::quantize_scatter_kv_fp8()`: scatter to paged KV pool
- `ops::kv_quant::decode_attention_int8_workspace_bytes()`: INT8 workspace
- FlashInfer native FP8 paged KV decode attention
- ROADMAP: "KV quantization FP8/INT8/TurboQuant 2-4 bit + fused-dequant attention" ✅

### Metal Backend Status
`infer/src/backend/metal/generate.rs`:
- `kv_dtype` is read from `weights.layers[0].attention_inputs.kv_dtype()` (line 144)
- `kv_dtype()` returns the projection weight dtype (BFloat16 or Float16)
- No explicit quantization: KV stored in same dtype as projection weights

## MLX Dtype Support Analysis

From `infer/mlx-sys/src/lib.rs` Dtype constants:

```
MLX_BOOL = 0, MLX_UINT8 = 1, MLX_UINT16 = 2, MLX_UINT32 = 3
MLX_INT8 = 5, MLX_INT16 = 6, MLX_INT32 = 7, MLX_INT64 = 8
MLX_FLOAT16 = 9, MLX_FLOAT32 = 10, MLX_BFLOAT16 = 12, MLX_COMPLEX64 = 13
```

**FP8 (E4M3/E5M2) is NOT in MLX's Dtype enum.** As of MLX 0.26 (April 2026), MLX
does not support FP8 operations. Apple Silicon hardware (M3/M4) does not have
native FP8 tensor cores — FP8 would be emulated in software with no performance benefit.

## Feasibility Assessment

| Format | MLX Support | Notes |
|--------|-------------|-------|
| FP8 E4M3 | ❌ No | Not in Dtype enum; hardware has no FP8 cores |
| INT8 | ✅ MLX_INT8 exists | But MLX quantize ops target weights, not activations |
| INT4 | ✅ via mlx.quantize | Per-group packed INT4, weight quantization only |
| BFloat16 | ✅ Default | Current KV dtype for Metal backend |

### Why MLX KV Quantization Is Harder than CUDA

On CUDA, FlashInfer handles quantized KV natively (FP8 paged attention kernels).
On Metal, MLX's attention (`mlx.core.fast.scaled_dot_product_attention`) has no
quantized KV input path — it always expects float16/bfloat16 input.

To quantize KV on Metal, we would need to:
1. Quantize K/V to INT8 on insert (custom Metal kernel or MLX ops)
2. Dequantize K/V to BFloat16 before attention (adds bandwidth + compute)
3. The net benefit depends on memory bandwidth vs. dequant overhead

On M3 Max (400 GB/s unified memory bandwidth), the KV cache rarely dominates
at typical inference concurrency (C=1–4). BFloat16 KV is generally fine.

## Recommendation

**FP8 KV on Metal: Not feasible** — hardware and MLX framework don't support it.

**INT8 KV on Metal: Low priority** — M3 Max has 400 GB/s bandwidth; KV memory
pressure only matters at C≥8 with long contexts. Current 52 tok/s baseline
is memory-bandwidth-limited at the weight GEMM level, not KV read.

**Defer Metal KV quantization** until:
1. MLX adds FP8 support, OR
2. Concurrency target exceeds C=8 with context > 8K tokens

**Alternative**: Use model weight quantization (INT4 via MLX quantize) to reduce
total memory footprint, freeing more bandwidth for unquantized KV.
