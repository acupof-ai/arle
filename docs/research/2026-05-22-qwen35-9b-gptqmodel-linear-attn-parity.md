# Qwen3.5-9B GPTQModel Layer-0 Linear-Attention Parity

## Context

`DavidWen2025/Qwen3.5-9B-GPTQ-4bit` can load behind
`INFER_EXPERIMENTAL_GPTQMODEL_W4=1`, but multi-token generation collapsed into
repeated punctuation. Prior attribution ruled out the sampled GPTQ W4 GEMV path
and then ruled out embedding, final RMSNorm, and untied `lm_head` as the direct
culprit. The next dense-path suspect was the hybrid layer-0 `linear_attention`
block.

This tranche extends `infer/examples/qwen35_linear_attn_parity.rs` so ARLE can
load the GPTQModel checkpoint while PyTorch loads the original BF16 source
checkpoint as reference.

## Command

```bash
INFER_EXPERIMENTAL_GPTQMODEL_W4=1 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
CARGO_BUILD_JOBS=1 \
cargo run -p infer --example qwen35_linear_attn_parity --release --features cuda -- \
  --model-path /home/ckl/.cache/modelscope/hub/DavidWen2025/Qwen3___5-9B-GPTQ-4bit \
  --reference-model-path /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B \
  --python /home/ckl/projects/arle/.venv/bin/python \
  --python-device cpu \
  --token-id 9419 \
  --output bench-output/2026-05-22-qwen35-9b-gptqmodel-linear-attn-parity/layer0-linear-attn.json
```

Artifacts:

```text
bench-output/2026-05-22-qwen35-9b-gptqmodel-linear-attn-parity/
```

## Results

| Metric | Value |
| --- | ---: |
| Layer type | `linear_attention` |
| Output length | `4096` |
| Finite ARLE/PyTorch pairs | `0` |
| ARLE non-finite values | `4096` |
| PyTorch non-finite values | `0` |
| PyTorch reference RMS | `1.02829233e-1` |
| RMSE/reference-RMS | `NaN` |
| Gate | FAIL |

First 8 values:

| Index | ARLE | PyTorch |
| ---: | ---: | ---: |
| 0 | `NaN` | `5.87463379e-4` |
| 1 | `NaN` | `-7.87353516e-3` |
| 2 | `NaN` | `-8.97216797e-3` |
| 3 | `NaN` | `-3.34472656e-2` |
| 4 | `NaN` | `-6.83593750e-2` |
| 5 | `NaN` | `-6.28662109e-3` |
| 6 | `NaN` | `-6.50024414e-3` |
| 7 | `NaN` | `1.97753906e-2` |

## Attribution

The 9B GPTQModel generation failure now has a narrow failing module-local gate:
ARLE's layer-0 `linear_attention` output is entirely non-finite for token
`9419`, while the PyTorch BF16 reference is finite.

This means the current failure should not be attributed to:

- GPTQ W4 projection GEMV packing or kernel execution: sampled projections
  already matched the faithful GPTQ reference within `0.25%` RMSE/reference-RMS.
- Embedding, final RMSNorm, or untied `lm_head`: those module outputs passed in
  the dense-module continue scan.
- Generic Qwen3.5 linear attention at small scale: the 0.8B BF16 layer-0
  linear-attention harness passed at `0.4814%` RMSE/reference-RMS.

The new suspect is the 9B GPTQModel hybrid linear-attention path specifically:
its dense `A_log` / `norm.weight` handling, conv/state update, scale/clamp
behavior, or interaction with quantized adjacent projections.

## Decision

Do not run OPD or switch headline docs for this 9B GPTQModel checkpoint yet.
The next axis should split layer-0 `linear_attention` into sub-op checkpoints
and find the first NaN-producing operation.

Suggested single-variable order:

1. Compare layer-0 normalized input hidden state before `linear_attention`.
2. Compare `linear_attn.in_proj_qkv`, `in_proj_z`, and `out_proj` outputs under
   the real hidden input instead of a random GEMV vector.
3. Compare `A_log`, `norm.weight`, conv/state update, and gating intermediates.
4. Only after all layer-0 intermediates are finite and within gate, rerun
   multi-token generation and the OPD bench.

Kill criterion for the next tranche: if any sub-op still produces NaNs after
matching input tensors are verified finite, hold the 9B GPTQModel path and fix
that sub-op before attempting full-model or OPD validation.

