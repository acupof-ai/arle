# ARLE TurboQuant 9B Tensor-Local Parity

## Goal

Replace the too-coarse full-model logits gate with a tensor-local attribution
gate for Qwen3.5-9B TurboQuant 4-bit. The tested tensor is:

```text
model.language_model.layers.1.mlp.gate_proj
rows 0..8, full K=4096
```

This is the first stage in the quantized-teacher ladder:

1. tensor-local dequant parity,
2. layer-local matmul parity,
3. layer-0 forward parity,
4. full-model logits parity,
5. OPD bench.

## Hypothesis

The 9B-TQ4 full-model logits KILL could come from either:

- the TurboQuant 4-bit format being fundamentally too lossy for this teacher, or
- ARLE's CUDA weight dequant path not matching the Python quantizer's transform.

Tensor-local dequant separates those cases.

## Params

- Source BF16 model:
  `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B`
- Quantized model:
  `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B-TQ4`
- Tensor:
  `model.language_model.layers.1.mlp.gate_proj`
- Slice: first 8 rows, all 4096 columns
- Quantization: TurboQuant 4-bit, group size 128
- GPU: RTX 4070 Ti SUPER
- CUDA env:
  `NVCC_CCBIN=/usr/bin/g++-14`,
  `INFER_TILELANG_PYTHON=$PWD/.venv/bin/python`,
  `CUDARC_CUDA_VERSION=13010`, `TORCH_CUDA_ARCH_LIST=8.9`

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
  --output bench-output/2026-05-21-qwen35-9b-tq4-tensor-local-parity/arle-cuda-dequant.json

.venv/bin/python \
  bench-output/2026-05-21-qwen35-9b-tq4-tensor-local-parity/turboquant_tensor_local_parity.py \
  --source-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B \
  --tq-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B-TQ4 \
  --tensor-base model.language_model.layers.1.mlp.gate_proj \
  --row-start 0 \
  --row-count 8 \
  --cuda-json bench-output/2026-05-21-qwen35-9b-tq4-tensor-local-parity/arle-cuda-dequant.json \
  --summary bench-output/2026-05-21-qwen35-9b-tq4-tensor-local-parity/summary.json \
  --top-errors bench-output/2026-05-21-qwen35-9b-tq4-tensor-local-parity/top-cuda-python-errors.csv
```

Raw artifacts:

- `bench-output/2026-05-21-qwen35-9b-tq4-tensor-local-parity/arle-cuda-dequant-run.txt`
- `bench-output/2026-05-21-qwen35-9b-tq4-tensor-local-parity/arle-cuda-dequant.json`
- `bench-output/2026-05-21-qwen35-9b-tq4-tensor-local-parity/turboquant_tensor_local_parity.py`
- `bench-output/2026-05-21-qwen35-9b-tq4-tensor-local-parity/summary.json`
- `bench-output/2026-05-21-qwen35-9b-tq4-tensor-local-parity/top-cuda-python-errors.csv`

## Results

The Python quantizer centroids and ARLE C-side Lloyd-Max centroids match to
float noise:

| Comparison | Max abs | Mean abs |
| --- | ---: | ---: |
| config centroids vs CUDA centroids | `1.49e-8` | `3.38e-9` |

Tensor-local dequant metrics:

| Comparison | Max abs | Mean abs | RMSE | RMSE / source RMS | Median rel | P99 rel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Python faithful dequant vs BF16 source | `0.005408` | `0.000854` | `0.001073` | `0.0961` | `0.0959` | `5.039` |
| ARLE CUDA dequant vs BF16 source | `0.097412` | `0.009253` | `0.015700` | `1.4067` | `1.4106` | `5.584` |
| ARLE CUDA dequant vs Python faithful | `0.097181` | `0.008820` | `0.015669` | `1.4093` | `0.1891` | `2.003` |

The full-model TQ4 failure is therefore not explained by 4-bit precision alone.
The faithful Python dequant is lossy, but it is an order of magnitude closer to
the BF16 source than the ARLE CUDA dequant path.

## Root Cause

The CUDA weight FWHT sign convention is wrong relative to the Python quantizer.

In `crates/cuda-kernels/csrc/gemm/turboquant_weight_gemv.cu`, the shared helper
used by both fused GEMV and bulk dequant computes the upper butterfly lane as
`upper - lower`:

```text
line 38: float diff = val - other;
line 39: val = (tid & stride) ? diff : sum;
line 57: val = (tid < pair) ? (a + b) : (a - b);
```

The Python quantizer in `scripts/turboquant_weights.py` uses the standard FWHT
update for each pair:

```text
lower = lower + upper
upper = lower_old - upper_old
```

The cheap control emulated the current CUDA sign convention in Python and then
rounded to BF16. It matches the ARLE CUDA output:

| Control | Max abs vs ARLE CUDA | RMSE vs ARLE CUDA |
| --- | ---: | ---: |
| Faithful Python FWHT | `0.097181` | `0.015669` |
| CUDA sign-bug emulation | `0.000120` | `1.84e-5` |
| CUDA sign-bug emulation + BF16 rounding | `1.53e-5` | `9.42e-8` |

This licenses the root cause: the current ARLE CUDA weight dequant/GEMV path is
performing a different transform than the offline TurboQuant quantizer.

## Decision

Do not run the 9B-TQ4 -> 0.8B OPD bench yet.

Next axis should be a one-line-kernel-family fix plus the same staged gates:

1. Fix `fwht_warp_optimized` so the upper lane receives `lower - upper` for
   both shuffle and shared-memory stages.
2. Rerun this tensor-local parity harness.
3. Add a layer-local matmul parity harness for the same tensor and a fixed
   hidden vector.
4. Rerun full-model raw-logits parity with top-64 dominant relerr `<= 5e-2`.
5. Only after full logits pass, rerun the 9B-TQ4 InferTeacher OPD bench.

## Problems

This report does not license that TurboQuant 4-bit is sufficient for OPD
teacher quality after the kernel fix. The faithful Python path still has
`9.61%` RMSE/source-RMS on this small slice, and relative error is unstable near
zero-valued weights. The next full-model logits gate is still required.

## Learnings

For quantized weight formats, tensor-local dequant parity must precede every
serving or OPD benchmark. Full-model logits failure only says the teacher is
wrong; it cannot attribute format loss versus dequant/kernel mismatch.
