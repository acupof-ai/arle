# A4 Qwen3.5 MoE Grouped Backward

## Context

Path A needs Qwen3.5/Qwen3.6-style MoE LoRA backward to stop issuing one tape
linear per selected expert/projection. The A0 MoE finite-diff gate was already
licensed; this tranche replaces the expert gate/up/down matmul loop with three
grouped expert-linear tape entries while keeping router/top-k and final weighted
scatter semantics unchanged.

## What Worked

- Added `MoeGroupedLinear` and `MoeGroupedWeightedScatter` autograd ops.
- `MoeWithLora::forward` now builds one expert-major route table and runs:
  `grouped gate`, `grouped up`, packed `silu*up`, `grouped down`, then one
  grouped weighted scatter.
- Backward uses batched GEMM form for the heavy linear math:
  base `X @ W^T`, LoRA `X @ A^T`, and LoRA `low @ B^T`; gradients are then
  unpacked back to the original expert weight / LoRA A / LoRA B tensors.
- Structure gate: `cpu_moe_uses_grouped_expert_linear_entries` verifies the
  MoE unit has `MoeGroupedLinear=3`, `MoeGroupedWeightedScatter=1`, and
  `MatmulBT=1` (router only). The old expert linear loop is no longer on
  `MatmulBT`.

## Results

| Gate | Env | Result |
|---|---|---|
| CPU finite diff | local Mac, `cargo test -p train --release --test test_moe_a0 -- --nocapture` | PASS, `max_rel=3.407e-3`, tiny failures 0 |
| CUDA finite diff | `.62` GPU4, `CUDARC_CUDA_VERSION=12090`, `ARLE_CUDA_KERNEL_SET=dsv4_flash`, `CUDA_VISIBLE_DEVICES=4`, `cargo test -p train --release --no-default-features --features cuda --test test_moe_a0 -- --nocapture` | PASS, `max_rel=2.487e-3`, tiny failures 0 |
| Local train lib | `cargo test -p train --release --no-default-features --features no-cuda --lib` | PASS, 102/102 |
| Local autograd lib | `cargo test -p autograd --release --lib` | PASS, 14/14 |
| Clippy | no-cuda + cuda/no-cuda typecheck profile | PASS |

## Caveat

This tranche eliminates the expert matmul tape loop and routes grouped backward
through batched GEMM math, but pack/unpack and route scatter are still host-side.
That is the next wall before claiming full 35B step-time speedup.

## Rule

For training-kernel milestones, require both a numerical gate and a structural
reachability gate. Here the finite-diff gates prove gradient correctness, and
the profile-count test proves the old per-expert `MatmulBT` loop is not still
silently running.
