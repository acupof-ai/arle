# ARLE Qwen3.5 Layer-0 Linear-Attention Parity

## Goal

After TurboQuant projection-local GEMV checks passed for Qwen3.5-9B-TQ4,
the remaining full-model logits drift moved to dense-path suspects. This
tranche checks whether ARLE infer's Qwen3.5 layer-0 `linear_attention` forward
matches the PyTorch transformers BF16 reference.

## Params

- ARLE model: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base`
- PyTorch reference: same checkpoint through `transformers`
- Layer: 0
- Layer type: `linear_attention`
- Input: token id `9419`; comparison point is the layer-0 linear-attention
  output after embedding + layer-0 input RMSNorm.
- ARLE runtime: CUDA infer path
- PyTorch runtime: CPU BF16 fallback implementation
- Gate: `RMSE/reference-RMS <= 5%`
- GPU env:
  `NVCC_CCBIN=/usr/bin/g++-14`,
  `INFER_TILELANG_PYTHON=$PWD/.venv/bin/python`,
  `CUDARC_CUDA_VERSION=13010`, `TORCH_CUDA_ARCH_LIST=8.9`,
  `CARGO_BUILD_JOBS=1`

Command:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
CARGO_BUILD_JOBS=1 \
cargo run -p infer --example qwen35_linear_attn_parity --release --features cuda -- \
  --model-path /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base \
  --python .venv/bin/python \
  --python-device cpu \
  --token-id 9419 \
  --output bench-output/2026-05-21-qwen35-linear-attn-parity/qwen35-08b-layer0-linear-attn.json
```

Raw artifacts:

- `bench-output/2026-05-21-qwen35-linear-attn-parity/qwen35-08b-layer0-linear-attn.run.txt`
- `bench-output/2026-05-21-qwen35-linear-attn-parity/qwen35-08b-layer0-linear-attn.json`

## Results

| Metric | Value |
| --- | ---: |
| Output length | `1024` |
| Max abs | `7.32421875e-4` |
| Max rel | `4.91304350` |
| Mean abs | `1.54260691e-4` |
| RMSE | `2.12224928e-4` |
| Reference RMS | `4.40851748e-2` |
| RMSE/reference-RMS | `0.4814%` |
| Gate | PASS |

`max_rel` is dominated by near-zero reference entries. The scale-stable metric
passes comfortably.

First 8 output entries:

| Index | ARLE | PyTorch | Abs err | Rel err |
| ---: | ---: | ---: | ---: | ---: |
| 0 | `-6.67968750e-1` | `-6.67968750e-1` | `0.00000000e0` | `0.00000000e0` |
| 1 | `1.98974609e-2` | `1.98974609e-2` | `0.00000000e0` | `0.00000000e0` |
| 2 | `1.20849609e-2` | `1.20239258e-2` | `6.10351562e-5` | `5.07614203e-3` |
| 3 | `-3.17382812e-2` | `-3.14941406e-2` | `2.44140625e-4` | `7.75193796e-3` |
| 4 | `-4.54101562e-2` | `-4.51660156e-2` | `2.44140625e-4` | `5.40540554e-3` |
| 5 | `-1.20239258e-2` | `-1.16577148e-2` | `3.66210938e-4` | `3.14136110e-2` |
| 6 | `1.35742188e-1` | `1.35742188e-1` | `0.00000000e0` | `0.00000000e0` |
| 7 | `-3.58886719e-2` | `-3.61328125e-2` | `2.44140625e-4` | `6.75675692e-3` |

## Decision

Layer-0 linear attention agrees on Qwen3.5-0.8B-Base under the BF16-realistic
gate. This does not support "ARLE Qwen3.5 dense linear-attention forward is the
root cause" for the 9B-TQ4 full-model logits drift.

Do not run the 9B-TQ4 OPD bench yet. The next attribution should move to the
remaining dense suspects: embedding, final RMSNorm, and LM head. A direct 9B
linear-attn recheck is deferred until the harness supports separate ARLE model
and PyTorch BF16 reference paths, because PyTorch cannot load the ARLE TQ4
checkpoint format directly.

## Problems

- This validates the 0.8B checkpoint first, as requested for iteration speed.
- It does not prove every 9B linear-attention layer is clean.
- The current harness uses one model path for ARLE and PyTorch; 9B-TQ4 needs a
  split path: ARLE TQ4 checkpoint plus original 9B BF16 checkpoint.

## Learnings

Dense-path attribution should proceed module by module. With projection GEMV
and 0.8B layer-0 linear attention passing, broad full-logit parity failure
should not be attributed to TurboQuant GEMV or linear attention without a
new, narrower failing gate.
