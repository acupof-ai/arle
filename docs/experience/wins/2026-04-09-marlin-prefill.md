# 2026-04-09 · Marlin W4A16 GEMM — 25x Prefill Speedup

## Context

Integrated IST-DASLab Marlin W4A16 GEMM kernel for prefill (seq_len > 1).
Weight repack done offline via `scripts/marlin_repack.py` using the exact
vLLM/IST-DASLab tile layout permutation.

L4 GPU, Qwen3-4B W4-g128, num_slots=1.

## Results

| Prompt Length | GEMV Batch (ms) | Marlin (ms) | Speedup |
|--------------|-----------------|-------------|---------|
| 5 tokens | 76 | 63 | 1.2x |
| 50 tokens | 368 | 67 | **5.5x** |
| 200 tokens | 1389 | 84 | **16.5x** |
| 500 tokens | 3447 | 137 | **25.2x** |

Decode speed unchanged at ~70 tok/s (Marlin not used for seq_len=1).

## What Worked

1. **Offline Python repack** (`scripts/marlin_repack.py`): Converts our [N, K/2] packed
   INT4 → GPTQ [K/8, N] int32 → Marlin tile layout using IST-DASLab's `_get_perms()`
   permutation tables. Also transposes + permutes scales.

2. **Pre-computed Marlin weights in safetensors**: Stored as `.marlin_qweight` and
   `.marlin_scales` alongside original weights. Loaded at inference time with zero
   runtime repack overhead.

3. **BF16↔FP16 conversion**: Lightweight kernels (~1μs each) bridge our BF16 engine
   with Marlin's FP16 interface.

## Limitations

- **Decode quality degradation**: First token is correct but subsequent decode tokens
  show repetition artifacts. Root cause: FP16↔BF16 precision boundary between
  Marlin prefill and native W4 GEMV decode. Needs investigation.

- **Extra storage**: Marlin model has ~2.5GB additional Marlin-packed weights alongside
  the original INT4 weights (needed for decode GEMV). Total ~5GB vs 2.5GB.

- **Offline repack required**: `scripts/marlin_repack.py` must be run before using Marlin.
  Only models with `.marlin_qweight` tensors use the Marlin path.

## Root Cause of Earlier Bug

Our original simplified `marlin_repack.cu` stored weights sequentially within tiles.
The real Marlin kernel expects a specific `ldmatrix`-optimized layout defined by the
`_get_perms()` permutation table from IST-DASLab. The fix: use the correct permutation
(ported to Python from the original Marlin repo).

## Rule

For Marlin integration: always use the reference permutation tables from IST-DASLab/marlin.
Do NOT write simplified repack kernels — the tile layout is critical for tensor core MMA.
