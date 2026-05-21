# ARLE TurboQuant Weight FWHT Fix

## Goal

Fix the Qwen3.5-9B TurboQuant 4-bit tensor-local dequant failure attributed in
[`docs/research/2026-05-21-arle-turboquant-9b-tensor-local-parity.md`](../../research/2026-05-21-arle-turboquant-9b-tensor-local-parity.md).

The license gate was tensor-local, not full-model: on
`model.language_model.layers.1.mlp.gate_proj` rows `0..8`, ARLE CUDA dequant
had to drop from `140%` RMSE/source-RMS to `<=15%`, matching the Python
faithful decoder plus BF16 rounding noise.

## What Worked

`scripts/turboquant_weights.py` defines the saved format:

- signs are applied before FWHT during quantization: `rotated = weight * signs`
- FWHT butterfly is lower=`a+b`, upper=`a-b`
- normalization happens once at the end: divide by `sqrt(n)`

The OPD teacher path for TurboQuant model weights uses
`crates/cuda-kernels/csrc/gemm/turboquant_weight_gemv.cu`, not the KV
TurboQuant kernels. Its shared `fwht_warp_optimized` helper feeds both fused
decode GEMV and bulk-dequant prefill. The bug was the upper butterfly lane:
CUDA used `upper - lower`; Python uses `lower - upper`.

The fix aligns both shuffle and shared-memory stages with the Python encoder.

## Results

Artifact directory:

```text
bench-output/2026-05-21-qwen35-9b-tq4-fwht-fix/
```

Commands:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
CARGO_BUILD_JOBS=1 \
cargo run -p infer --example turboquant_weight_dequant_dump --release --features cuda -- \
  --model-path /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B-TQ4 \
  --tensor-base model.language_model.layers.1.mlp.gate_proj \
  --row-start 0 \
  --row-count 8 \
  --output bench-output/2026-05-21-qwen35-9b-tq4-fwht-fix/arle-cuda-dequant.json

.venv/bin/python \
  bench-output/2026-05-21-qwen35-9b-tq4-fwht-fix/turboquant_tensor_local_parity.py \
  --source-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B \
  --tq-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B-TQ4 \
  --tensor-base model.language_model.layers.1.mlp.gate_proj \
  --row-start 0 \
  --row-count 8 \
  --cuda-json bench-output/2026-05-21-qwen35-9b-tq4-fwht-fix/arle-cuda-dequant.json \
  --summary bench-output/2026-05-21-qwen35-9b-tq4-fwht-fix/summary.json \
  --top-errors bench-output/2026-05-21-qwen35-9b-tq4-fwht-fix/top-cuda-python-errors.csv
```

| Comparison | Before RMSE/source-RMS | After RMSE/source-RMS |
| --- | ---: | ---: |
| Python faithful dequant vs BF16 source | `9.61%` | `9.61%` |
| ARLE CUDA dequant vs BF16 source | `140.67%` | `9.62%` |
| ARLE CUDA dequant vs Python faithful | `140.93%` | `0.166%` |

The gate passes: ARLE CUDA dequant is now within the `<=15%` source-RMS
threshold and matches the Python faithful dequant after BF16 rounding.

## Problems

This only licenses tensor-local dequant parity. It does not yet license
Qwen3.5-9B-TQ4 as an OPD teacher. The next gates remain:

1. full-model raw-logits parity against Qwen3.5-9B BF16, top-64 dominant relerr
   `<=0.10`;
2. 100-step Qwen3.5-9B-TQ4 -> Qwen3.5-0.8B LoRA OPD bench with held-out KL
   monotonically decreasing at `0/25/50/100`.

## Rule

For quantized model weights, the Python offline encoder owns the saved tensor
format. CUDA decode/GEMV kernels must match its transform exactly before any
serving or OPD benchmark can be licensed.
