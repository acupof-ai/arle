# Qwen3.5-9B TurboQuant Logits KILL After LM Head Fix

## Context

The Qwen3.5 untied-lm-head fix corrected the dense output projection path:
`lm_head` RMSE/ref-RMS dropped from `130.502%` to `0.00176%` on
Qwen3.5-9B-TQ4. The next license gate was full-model raw-logits parity against
the Qwen3.5-9B BF16 ModelScope checkpoint on fixed token `[9419]`.

User gate for this retry: top-64 dominant relerr `<=0.10`.

## Evidence

Raw artifacts:

```text
bench-output/2026-05-21-qwen35-9b-tq4-lm-head-fix/
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
  --output bench-output/2026-05-21-qwen35-9b-tq4-lm-head-fix/arle-tq4-logits.json \
  --cuda-graph false
```

Result:

```text
seq_len=1
vocab_size=248320
load_seconds=117.897762
forward_seconds=1.948254
host_readback_seconds=0.060079
```

PyTorch BF16 comparison:

```bash
.venv/bin/python \
  bench-output/2026-05-21-qwen35-9b-tq4-08b-opd-infer-teacher/compare_tq4_vs_pytorch_bf16.py \
  --bf16-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B \
  --arle-logits bench-output/2026-05-21-qwen35-9b-tq4-lm-head-fix/arle-tq4-logits.json \
  --input-ids 9419 \
  --summary bench-output/2026-05-21-qwen35-9b-tq4-lm-head-fix/parity-summary.json \
  --top64 bench-output/2026-05-21-qwen35-9b-tq4-lm-head-fix/parity-top64.csv \
  --pytorch-logits bench-output/2026-05-21-qwen35-9b-tq4-lm-head-fix/pytorch-bf16-logits.json
```

Full-model logits still fail:

| Metric | Value |
| --- | ---: |
| `top64_max_abs` | `1.937500` |
| `top64_mean_abs` | `0.549805` |
| `top64_max_rel` | `0.180233` |
| `top64_mean_rel` | `0.051946` |
| `top64_ref_mean_abs` | `10.701172` |
| user gate | `<=0.10` |

The old comparison script still persists a stricter `0.05` gate in JSON, but
this retry also fails the user-relaxed `0.10` gate.

## Root Cause

The previous lm-head attribution was real, and the dense-module gate is fixed.
It was not the only source of full-model drift. With `lm_head.weight` now
correctly loaded, the remaining top-64 relerr is much smaller than the prior
`1.17`, but still too high to license Qwen3.5-9B-TQ4 as an OPD teacher.

This result points away from the dense copied tensors and toward accumulated
model-path drift after the already-clean gates:

- TurboQuant projection GEMV parity passed for sampled q/k/v/o/gate/up/down
  projections.
- Dense embedding/final_norm/lm_head module parity now passes.
- Full-model logits still fail, so the next attribution level needs a
  layer-by-layer forward scan after layer 0, especially hybrid linear-attention
  recurrence and residual accumulation across many layers.

## Fix

Killed at the full-model logits gate. Do not run the 100-step
Qwen3.5-9B-TQ4 -> Qwen3.5-0.8B LoRA OPD bench and do not switch README, web,
usage manual, or comparison PNG headline docs to 9B-TQ4.

## Rule

Fixing a known dense-module mismatch does not license a quantized teacher.
After every localized parity fix, full-model logits remain an independent gate;
if it fails, route back to deeper layer-local attribution before any OPD bench.
