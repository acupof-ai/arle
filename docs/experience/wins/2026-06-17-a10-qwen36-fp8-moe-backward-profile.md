# A10 Qwen3.6 FP8 MoE Backward Profile

## Context

A9 licensed the real-checkpoint Qwen3.6-35B-A3B FP8 MoE LoRA gradient with a
finite-diff gate, but the single-layer analytic backward still took about 7-8s.
Before writing the CUDA MoE backward kernel, this tranche adds an opt-in
backward profile to the same gate and measures where that time actually goes.

This is a measurement tranche, not a correctness change: the finite-diff gate
and selected routed-expert element stay identical.

## What Worked

- `qwen36_fp8_lora_fd_gate` now accepts `--profile-backward`.
- With the flag enabled, the gate uses `Tape::backward_profiled` and prints
  total backward time, op totals, and site totals before the finite-diff result.
- Default behavior is unchanged unless the flag is passed.

## Environment

- Host: `.62` / `iv-ye8is8fbi8s6iplibbg7`.
- GPU: H20 GPU7 via `CUDA_VISIBLE_DEVICES=7`; GPU3 avoided.
- Model: `/data01/models/Qwen3.6-35B-A3B-FP8`.
- Source:
  `/data01/arle-track1-opd-rollout-infer-202606170646`, overlaid with only
  `crates/train/examples/qwen36_fp8_lora_fd_gate.rs`.
- Target:
  `/data01/arle-target-track1-opd-rollout-infer-202606170646`.
- Build env:
  `CUDA_HOME=/usr/local/cuda CUDARC_CUDA_VERSION=12090
  ARLE_CUDA_DISABLE_FLASHMLA=1 INFER_TILELANG_PYTHON=/root/tl-venv/bin/python`.

## Verification

Local gates:

```text
cargo fmt --check
PASS

cargo check -p train --release --no-default-features --features no-cuda --example qwen36_fp8_lora_fd_gate
PASS

CUDARC_CUDA_VERSION=12090 cargo check -p train --release --no-default-features --features cuda,no-cuda --example qwen36_fp8_lora_fd_gate
PASS

cargo clippy -p train --release --no-default-features --features no-cuda --example qwen36_fp8_lora_fd_gate -- -D warnings
PASS
```

Remote build:

```text
cargo build -p train --example qwen36_fp8_lora_fd_gate --release --no-default-features --features cuda
Finished `release` profile [optimized] target(s) in 7.91s
```

Remote routed-expert finite diff with backward profile:

```text
CUDA_VISIBLE_DEVICES=7 qwen36_fp8_lora_fd_gate \
  --model /data01/models/Qwen3.6-35B-A3B-FP8 \
  --device 0 \
  --target-set all-linear \
  --target-adapter auto:routed-up \
  --mode mlp-layer \
  --layer 0 \
  --eps 1e-3 \
  --profile-backward
```

Result:

```text
qwen36_fp8_lora_fd_backward_profile total_seconds=7.156945 op_seconds=7.107308 prelude_seconds=0.000146 merge_grad_seconds=0.049313 op_kinds=12 site_kinds=11
qwen36_fp8_lora_fd_backward_profile_op rank=1 op=MoeGroupedLinear count=3 seconds=7.103443 pct_total=99.252
qwen36_fp8_lora_fd_backward_profile_op rank=2 op=Mul count=4 seconds=0.001596 pct_total=0.022
qwen36_fp8_lora_fd_backward_profile_op rank=3 op=Silu count=2 seconds=0.001057 pct_total=0.015
qwen36_fp8_lora_fd_backward_profile_op rank=4 op=MatmulBT count=11 seconds=0.000671 pct_total=0.009
qwen36_fp8_lora_fd_backward_profile_op rank=5 op=MoeGroupedWeightedScatter count=1 seconds=0.000326 pct_total=0.005
qwen36_fp8_lora_fd_gate_result load_seconds=13.631451 analytic_seconds=7.737183 plus_seconds=0.575294 minus_seconds=0.575430 live_host_mib=5586.2 mode=mlp-layer layer=0 target=model.language_model.layers.0.mlp.experts.210.up_proj.weight.lora_b index=186 eps=1.0e-3 loss_base=1.941415121e-6 loss_minus=1.941672963e-6 loss_plus=1.941155006e-6 analytic=-2.581575984e-7 numeric=-2.589786163e-7 rel_err=3.170e-3
qwen36_fp8_lora_fd_gate PASS
```

## Delta

| Gate | Before | After | Verdict |
|---|---:|---:|---|
| Real 35B routed expert finite diff | rel_err=3.170e-3 | rel_err=3.170e-3 | unchanged pass |
| Backward attribution | only aggregate analytic_seconds=7.77s | `MoeGroupedLinear` 7.103s / 99.25% of profiled backward | localized |
| Merge-grad overhead | unknown | 0.049s | not the wall |
| Non-MoE backward ops | unknown | <0.004s total | not the wall |

## Next Wall

The dominant wall is the `MoeGroupedLinear` backward op itself, not gradient
merge, router/scatter, or ordinary LoRA `MatmulBT` sites. The current autograd
implementation still packs expert-major tensors on host and calls device
matmul/readback helpers inside the grouped backward path. Full 35B OPD training
needs a device-resident CUDA grouped MoE backward for the gate/up/down trio
before full-model step time can be judged.

The earlier A9 note that the full-model gate was blocked by linear-attention
host fallback is now only a partial hypothesis. This A10 measurement proves a
single real Qwen3.6 MoE layer already spends about 7.1s in grouped-linear
backward, so MoE backward is the next licensed optimization wall.

## Rule

Profile the licensed micro-gate before optimizing the full model. If one layer
already spends 99% of backward in a single op, fix that op before using a
full-model run to infer the next wall.
