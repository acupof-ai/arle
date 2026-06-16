# A9 Qwen3.6 FP8 Real-Checkpoint MoE LoRA Finite-Diff Gate

## Context

Path A needs ARLE autograd-native Qwen3.6-35B-A3B FP8 LoRA training for OPD
35B. A7 licensed the real FP8 loader/memory gate and A8 licensed train-to-infer
LoRA sync plus post-sync generation quality. The remaining gradient evidence
needed to move past synthetic A0/A5 was a real-checkpoint finite-diff gate on
the 35B FP8 MoE weights.

The first full-model finite-diff attempt was intentionally killed: after more
than three minutes it had GPU7 resident at 34.5GiB but 0% GPU util, while the
main process ran at 100% CPU. That localized the full-model gate to the
train-side Qwen3.6 `linear_attention_core` host fallback, not to FP8 LoRA
gradient math. This tranche therefore validates the real FP8 MoE layer directly
with a diagnostic MLP-only gate, leaving full-model finite-diff blocked on
linear-attention CUDA coverage.

## What Worked

- Added `Qwen35Model::forward_mlp_for_diagnostics`, a hidden diagnostic wrapper
  that runs one loaded layer's MLP on a synthetic hidden state.
- Added `qwen36_fp8_lora_fd_gate`, a CUDA-only real-checkpoint finite-diff
  harness. Default mode is `mlp-layer`, so it avoids the known full-model
  linear-attention host fallback.
- The harness supports:
  - explicit adapter targets such as
    `model.language_model.layers.0.mlp.shared_expert.up_proj.weight.lora_b`;
  - `--target-adapter auto:routed-up`, which scans the chosen layer's routed
    expert `up_proj.lora_b` gradients and finite-diffs the largest non-zero
    element.

## Environment

- Host: `.62` / `iv-ye8is8fbi8s6iplibbg7`.
- GPU: H20 GPU7 via `CUDA_VISIBLE_DEVICES=7`; GPU3 avoided.
- Model: `/data01/models/Qwen3.6-35B-A3B-FP8`.
- Source:
  `/data01/arle-track1-opd-rollout-infer-202606170646`, overlaid with only
  `crates/train/src/qwen35.rs` and
  `crates/train/examples/qwen36_fp8_lora_fd_gate.rs`.
- Target:
  `/data01/arle-target-track1-opd-rollout-infer-202606170646`.
- Build env:
  `CUDA_HOME=/usr/local/cuda CUDARC_CUDA_VERSION=12090
  ARLE_CUDA_DISABLE_FLASHMLA=1 INFER_TILELANG_PYTHON=/root/tl-venv/bin/python`.
- Remote build: `cargo build -p train --example qwen36_fp8_lora_fd_gate
  --release --no-default-features --features cuda` passed in 7.65s after the
  target was warm; after the final error-path cleanup sync, the same command
  passed again in 29.43s.

## Verification

Local gates:

```text
cargo fmt --check
PASS

cargo check -p train --release --no-default-features --features no-cuda --example qwen36_fp8_lora_fd_gate
PASS

CUDARC_CUDA_VERSION=12090 cargo check -p train --release --no-default-features --features cuda,no-cuda --example qwen36_fp8_lora_fd_gate
PASS
```

Remote shared-expert finite diff:

```text
CUDA_VISIBLE_DEVICES=7 qwen36_fp8_lora_fd_gate \
  --model /data01/models/Qwen3.6-35B-A3B-FP8 \
  --target-set all-linear --rank 8 --alpha 16 \
  --eps 1e-3 --tokens 1,3,8 --mode mlp-layer --layer 0
```

Result:

```text
qwen36_fp8_lora_fd_gate_result load_seconds=17.039255 analytic_seconds=7.700530 plus_seconds=0.577362 minus_seconds=0.576140 live_host_mib=5586.2 mode=mlp-layer layer=0 target=model.language_model.layers.0.mlp.shared_expert.up_proj.weight.lora_b index=2940 eps=1.0e-3 loss_base=1.941415121e-6 loss_minus=1.939566118e-6 loss_plus=1.943267534e-6 analytic=1.851400043e-6 numeric=1.850707918e-6 rel_err=3.738e-4
qwen36_fp8_lora_fd_gate PASS
```

Remote routed-expert finite diff:

```text
CUDA_VISIBLE_DEVICES=7 qwen36_fp8_lora_fd_gate \
  --model /data01/models/Qwen3.6-35B-A3B-FP8 \
  --target-set all-linear --rank 8 --alpha 16 \
  --target-adapter auto:routed-up \
  --eps 1e-3 --tokens 1,3,8 --mode mlp-layer --layer 0
```

Result:

```text
qwen36_fp8_lora_fd_gate_result load_seconds=13.820885 analytic_seconds=7.770421 plus_seconds=0.595916 minus_seconds=0.574951 live_host_mib=5586.2 mode=mlp-layer layer=0 target=model.language_model.layers.0.mlp.experts.210.up_proj.weight.lora_b index=186 eps=1.0e-3 loss_base=1.941415121e-6 loss_minus=1.941672963e-6 loss_plus=1.941155006e-6 analytic=-2.581575984e-7 numeric=-2.589786163e-7 rel_err=3.170e-3
qwen36_fp8_lora_fd_gate PASS
```

## Delta

| Gate | Before | After | Verdict |
|---|---:|---:|---|
| Synthetic FP8 QLoRA linear finite diff | A5 rel_err <= 1e-2 | unchanged | pass |
| Synthetic Qwen3.6 MoE LoRA finite diff | A6 rel_err=5.862e-3 | unchanged | pass |
| Real 35B FP8 shared-expert up_proj LoRA-B finite diff | not licensed | rel_err=3.738e-4 | pass |
| Real 35B FP8 routed expert up_proj LoRA-B finite diff | not licensed | rel_err=3.170e-3 | pass |
| Full-model 35B FP8 finite diff | attempted | blocked by linear-attention host fallback | defer |

## Remaining Wall

The real FP8 MoE/QLoRA gradient path is licensed at the loaded-layer level. The
next Path A wall is full-model Qwen3.6 training forward/backward: the
`linear_attention_core` path still forces host materialization and made the
full-model finite-diff run CPU-bound. Full OPD 35B training requires either
CUDA coverage for Qwen3.6 linear attention or a training recipe that avoids
backpropagating through those layers.

## Rule

When a full-model finite-diff gate is dominated by an unrelated host fallback,
do not pretend it is a gradient verdict. Split the gate at the layer boundary:
license the real FP8 MoE gradient path directly, then track the full-model
blocker separately.
