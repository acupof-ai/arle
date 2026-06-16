# A12 Qwen3.6 MoE Backward Active-Expert Compact

## Context

A11 removed no-grad base work from the Qwen3.6 FP8 MoE backward gate, dropping
`MoeGroupedLinear` from 7.103s to 2.080s. The remaining waste was structural:
the backward math still packed and ran grouped matmuls for all 256 routed
experts even though the layer-0 finite-diff gate has one token and top-8
routing, so only 8 experts are active.

This tranche keeps the public tensor/output shape unchanged but compacts the
internal backward work to active experts only, then scatters gradients back to
their original expert ids.

## What Worked

- `moe_grouped_linear_backward` now builds an active-expert map from the saved
  routes.
- Backward packs input, upstream gradient, base weights, and LoRA A/B only for
  active experts.
- Gradients for weights and LoRA adapters scatter back into the original
  expert-indexed tensors, preserving the existing `GradPairs` contract.
- Packed-input gradients for the down projection scatter back into the full
  `[experts, max_rows, dim]` shape so upstream `silu`/`mul` semantics are
  unchanged.

## Environment

- Host: `.62` / `iv-ye8is8fbi8s6iplibbg7`.
- GPU: H20 GPU7 via `CUDA_VISIBLE_DEVICES=7`; GPU3 avoided.
- Model: `/data01/models/Qwen3.6-35B-A3B-FP8`.
- Source:
  `/data01/arle-track1-opd-rollout-infer-202606170646`, overlaid with only
  `crates/autograd/src/ops/moe.rs`.
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

cargo check -p autograd --release --no-default-features --features no-cuda
PASS

CUDARC_CUDA_VERSION=12090 cargo check -p autograd --release --no-default-features --features cuda,no-cuda
PASS

cargo test -p train --release --test test_moe_a0 -- --nocapture
PASS, max_rel=3.407372e-3, tiny_abs_failures=0

cargo test -p autograd --release --lib
PASS, 15/15

cargo clippy -p autograd --release --no-default-features --features no-cuda -- -D warnings
PASS

cargo check -p train --release --no-default-features --features no-cuda --example qwen36_fp8_lora_fd_gate
PASS

CUDARC_CUDA_VERSION=12090 cargo check -p train --release --no-default-features --features cuda,no-cuda --example qwen36_fp8_lora_fd_gate
PASS
```

Remote build:

```text
cargo build -p train --example qwen36_fp8_lora_fd_gate --release --no-default-features --features cuda
Finished `release` profile [optimized] target(s) in 31.89s
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
qwen36_fp8_lora_fd_backward_profile total_seconds=0.253390 op_seconds=0.202411 prelude_seconds=0.000137 merge_grad_seconds=0.050670 op_kinds=12 site_kinds=11
qwen36_fp8_lora_fd_backward_profile_op rank=1 op=MoeGroupedLinear count=3 seconds=0.198487 pct_total=78.333
qwen36_fp8_lora_fd_backward_profile_op rank=2 op=Mul count=4 seconds=0.001643 pct_total=0.648
qwen36_fp8_lora_fd_backward_profile_op rank=3 op=Silu count=2 seconds=0.001079 pct_total=0.426
qwen36_fp8_lora_fd_backward_profile_op rank=4 op=MatmulBT count=11 seconds=0.000667 pct_total=0.263
qwen36_fp8_lora_fd_backward_profile_op rank=5 op=MoeGroupedWeightedScatter count=1 seconds=0.000326 pct_total=0.129
qwen36_fp8_lora_fd_gate_result load_seconds=13.679419 analytic_seconds=0.826345 plus_seconds=0.577678 minus_seconds=0.578015 live_host_mib=5586.2 mode=mlp-layer layer=0 target=model.language_model.layers.0.mlp.experts.210.up_proj.weight.lora_b index=186 eps=1.0e-3 loss_base=1.941415121e-6 loss_minus=1.941672963e-6 loss_plus=1.941155006e-6 analytic=-2.581575700e-7 numeric=-2.589786163e-7 rel_err=3.170e-3
qwen36_fp8_lora_fd_gate PASS
```

## Delta

| Metric | A10 baseline | A11 no-grad | A12 active-compact | Delta vs A10 |
|---|---:|---:|---:|---:|
| Routed-expert finite-diff rel_err | 3.170e-3 | 3.170e-3 | 3.170e-3 | unchanged pass |
| Profiled backward total | 7.156945s | 2.107237s | 0.253390s | -96.46% |
| `MoeGroupedLinear` total | 7.103443s | 2.080250s | 0.198487s | -97.21% |
| Analytic phase wall time | 7.737183s | 2.682385s | 0.826345s | -89.32% |
| Merge-grad | 0.049313s | 0.022778s | 0.050670s | now visible |

## Next Wall

For this single-layer real-checkpoint gate, the 256-expert waste is gone. The
remaining profile is no longer dominated by GB-scale dead pack/upload: the
useful grouped-linear work is about 0.20s and merge-grad is about 0.05s. Full
35B OPD still needs the device-resident/CUDA grouped MoE backward path for
larger rollout/scoring shapes, but it should be developed from this active-only
baseline rather than the earlier full-expert padded path.

## Rule

MoE training kernels must operate on active routes, not nominal expert count.
Keeping the public expert-major tensor shape is fine; padding 248 inactive
experts into every backward matmul is not.
