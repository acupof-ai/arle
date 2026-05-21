# Qwen3.5-9B TurboQuant Logits KILL After FWHT Fix

## Context

`b8864cf` fixed the tensor-local TurboQuant weight FWHT mismatch. On
`model.language_model.layers.1.mlp.gate_proj` rows `0..8`, ARLE CUDA dequant
now matches the Python faithful decoder and passes the tensor-local license
gate:

```text
ARLE CUDA vs BF16 source RMSE/source-RMS: 140.67% -> 9.62%
ARLE CUDA vs Python faithful RMSE/Python-RMS: 140.93% -> 0.166%
```

The next gate was full-model raw-logits parity for Qwen3.5-9B-TQ4 against the
BF16 ModelScope checkpoint on fixed input token `[9419]`. The user gate for
this retry was top-64 dominant relerr `<=0.10`.

## Evidence

Raw artifacts:

```text
bench-output/2026-05-21-qwen35-9b-tq4-fwht-fix-logits/
```

ARLE logits dump:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
CARGO_BUILD_JOBS=1 \
cargo run -p infer --example raw_token_logits_dump --release --features cuda -- \
  --model-path /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B-TQ4 \
  --input-ids 9419 \
  --output bench-output/2026-05-21-qwen35-9b-tq4-fwht-fix-logits/arle-tq4-logits.json \
  --cuda-graph false
```

Result:

```text
seq_len=1
vocab_size=248320
load_seconds=114.752035
forward_seconds=2.982240
host_readback_seconds=0.060123
```

PyTorch BF16 comparison:

```bash
.venv/bin/python \
  bench-output/2026-05-21-qwen35-9b-tq4-08b-opd-infer-teacher/compare_tq4_vs_pytorch_bf16.py \
  --bf16-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B \
  --arle-logits bench-output/2026-05-21-qwen35-9b-tq4-fwht-fix-logits/arle-tq4-logits.json \
  --input-ids 9419 \
  --summary bench-output/2026-05-21-qwen35-9b-tq4-fwht-fix-logits/parity-summary.json \
  --top64 bench-output/2026-05-21-qwen35-9b-tq4-fwht-fix-logits/parity-top64.csv \
  --pytorch-logits bench-output/2026-05-21-qwen35-9b-tq4-fwht-fix-logits/pytorch-bf16-logits.json
```

Full-model logits still fail:

| Metric | Value |
| --- | ---: |
| `top64_max_abs` | `14.953125` |
| `top64_mean_abs` | `9.844091` |
| `top64_max_rel` | `1.172468` |
| `top64_mean_rel` | `0.919028` |
| `top64_ref_mean_abs` | `10.701172` |
| user gate | `<=0.10` |

The script's persisted summary still records the older `0.05` threshold from
the first 9B-TQ4 attempt, but this retry fails the relaxed `0.10` gate by more
than `11x`.

## Root Cause

The FWHT sign mismatch was real and fixed, but it was not sufficient to license
the complete 9B-TQ4 model as an OPD teacher. Full-model logits remain dominated
by quantization or a deeper model-weight path mismatch.

This full-model test is intentionally not enough to attribute the remaining
drift. The next SOLID attribution level is layer-local and path-specific:

1. Compare fused TurboQuant GEMV against bulk dequant + cuBLAS for the same
   projection and hidden vector.
2. Run tensor-local dequant parity across representative projection families
   (`q/k/v/o/gate/up/down`, lm-head if quantized, and dense fallbacks).
3. Run layer-0 forward parity before retrying full-model logits.

## Fix

Killed at the full-model logits gate. Do not run the 100-step
Qwen3.5-9B-TQ4 -> Qwen3.5-0.8B LoRA OPD bench and do not switch README or web
headline docs to 9B-TQ4.

The licensed deliverable from this axis is only the tensor-local CUDA FWHT fix
in `b8864cf`; the 9B-TQ4 teacher remains blocked.

## Rule

A tensor-local quantized-kernel fix licenses exactly that tensor-local gate.
For quantized teachers, full-model logits parity is still a separate gate, and
failure there must route back to layer-local attribution before any OPD bench.
