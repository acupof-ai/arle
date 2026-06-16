# A11 Qwen3.6 MoE Backward No-Grad Fast Path

## Context

A10 localized the real-checkpoint Qwen3.6-35B-A3B FP8 LoRA MLP-layer
finite-diff gate to `MoeGroupedLinear`: 7.103s of the 7.157s profiled backward.
Inspection showed a concrete waste: for the routed gate/up grouped-linears, the
base weights are frozen and the synthetic input does not require gradients, but
the backward path still packed and uploaded the full base `packed_weight_t`
before calling a matmul backward that returned no gradients.

This tranche removes only that no-grad work. It does not claim the CUDA grouped
MoE backward kernel is complete; it licenses the next step on a cleaner
baseline.

## What Worked

- `moe_grouped_linear_backward` now skips base-weight transpose packing unless
  the base path needs either input grad or weight grad.
- `grouped_matmul_backward` returns immediately when both requested gradients
  are false, avoiding accidental upload work at future call sites.
- The LoRA finite-diff target and route selection are unchanged.

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

cargo clippy -p autograd --release --no-default-features --features no-cuda -- -D warnings
PASS

cargo test -p autograd --release --lib
PASS, 15/15

cargo check -p train --release --no-default-features --features no-cuda --example qwen36_fp8_lora_fd_gate
PASS

CUDARC_CUDA_VERSION=12090 cargo check -p train --release --no-default-features --features cuda,no-cuda --example qwen36_fp8_lora_fd_gate
PASS
```

Remote build:

```text
cargo build -p train --example qwen36_fp8_lora_fd_gate --release --no-default-features --features cuda
Finished `release` profile [optimized] target(s) in 31.94s
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
qwen36_fp8_lora_fd_backward_profile total_seconds=2.107237 op_seconds=2.084139 prelude_seconds=0.000147 merge_grad_seconds=0.022778 op_kinds=12 site_kinds=11
qwen36_fp8_lora_fd_backward_profile_op rank=1 op=MoeGroupedLinear count=3 seconds=2.080250 pct_total=98.719
qwen36_fp8_lora_fd_backward_profile_op rank=2 op=Mul count=4 seconds=0.001649 pct_total=0.078
qwen36_fp8_lora_fd_backward_profile_op rank=3 op=Silu count=2 seconds=0.001043 pct_total=0.050
qwen36_fp8_lora_fd_backward_profile_op rank=4 op=MatmulBT count=11 seconds=0.000682 pct_total=0.032
qwen36_fp8_lora_fd_backward_profile_op rank=5 op=MoeGroupedWeightedScatter count=1 seconds=0.000301 pct_total=0.014
qwen36_fp8_lora_fd_gate_result load_seconds=13.616863 analytic_seconds=2.682385 plus_seconds=0.573247 minus_seconds=0.571671 live_host_mib=5586.2 mode=mlp-layer layer=0 target=model.language_model.layers.0.mlp.experts.210.up_proj.weight.lora_b index=186 eps=1.0e-3 loss_base=1.941415121e-6 loss_minus=1.941672963e-6 loss_plus=1.941155006e-6 analytic=-2.581575984e-7 numeric=-2.589786163e-7 rel_err=3.170e-3
qwen36_fp8_lora_fd_gate PASS
```

## Delta

| Metric | A10 before | A11 after | Delta |
|---|---:|---:|---:|
| Routed-expert finite-diff rel_err | 3.170e-3 | 3.170e-3 | unchanged pass |
| Profiled backward total | 7.156945s | 2.107237s | -70.56% |
| `MoeGroupedLinear` total | 7.103443s | 2.080250s | -70.72% |
| Analytic phase wall time | 7.737183s | 2.682385s | -65.33% |
| Merge-grad | 0.049313s | 0.022778s | not the wall |

## Next Wall

The no-grad waste is gone, but `MoeGroupedLinear` still owns 98.7% of the
profiled backward. The remaining work is the real CUDA grouped MoE backward
kernel/device-resident path for the useful gate/up/down gradients, especially
the down projection's input-gradient path and the LoRA A/B grouped matmuls.

## Rule

Before writing a new kernel, kill proven no-op work on the exact licensed
micro-gate. A frozen base path with no input gradient must not pack or upload
the frozen base weights just to return no gradients.
