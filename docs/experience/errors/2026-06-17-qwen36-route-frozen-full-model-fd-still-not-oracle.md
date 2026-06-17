# Qwen3.6 route-frozen full-model FD is still not a clean oracle

## Context

The previous full-model Qwen3.6 FP8 LoRA finite-diff gate crossed dynamic MoE
top-k route boundaries. This tranche added a route-frozen diagnostic:

- `moe_topk_softmax_with_indices` recomputes router weights over caller-provided
  expert ids and records the same ids in the backward context.
- `Qwen35Model::forward_with_frozen_moe_routes_for_diagnostics` reuses base
  route signatures for analytic, plus, and minus arms.
- `qwen36_fp8_lora_fd_gate` accepts `--freeze-base-routes` and
  `--sparse-logit-probe`.

The intent was to test the same local piecewise-smooth function in all FD arms.

## Evidence

Remote `.62` / `iv-ye8is8fbi8s6iplibbg7`, GPU3 avoided, model
`/data01/models/Qwen3.6-35B-A3B-FP8`.

Source:
`/data01/arle-track1-route-frozen-fd-fast-20260617095440`.

Target:
`/data01/arle-target-track1-route-frozen-fd`.

Build:

```text
CUDA_HOME=/usr/local/cuda CUDARC_CUDA_VERSION=12090
ARLE_CUDA_DISABLE_FLASHMLA=1 INFER_TILELANG_PYTHON=/root/tl-venv/bin/python
CARGO_TARGET_DIR=/data01/arle-target-track1-route-frozen-fd
cargo build -p train --example qwen36_fp8_lora_fd_gate --release --no-default-features --features cuda
PASS
```

The route-freeze precondition was reached on the real checkpoint:

```text
qwen36_fp8_lora_route_frozen base_layers=40 total_slots=960 tokens=3 top_k=8 experts=256
```

But the full-model FD still failed:

| Probe | Layer / target | eps | analytic | numeric | rel_err |
|---|---:|---:|---:|---:|---:|
| full logits | layer0 routed-up | 1e-3 | 3.636742830e-1 | -2.017020988e1 | 1.018e0 |
| full logits | layer0 routed-up | 1e-4 | 3.636742830e-1 | 4.806518555e1 | 9.924e-1 |
| full logits | layer39 routed-up | 1e-3 | -1.796592951e0 | -3.294944525e0 | 4.547e-1 |
| full logits | layer39 shared-up | 1e-3 | -8.056034088e0 | -7.674216747e0 | 4.740e-2 |
| sparse 1-logit mask | layer39 routed-up | 1e-3 | 2.477753311e-1 | 0.0 | 1.000e0 |
| sparse 1-logit mask | layer39 routed-up | 1e-1 | 2.477753311e-1 | 2.451598644e-1 | 1.056e-2 |
| sparse 64-logit/row mask | layer39 routed-up | 1e-3 | -1.678510904e0 | 1.347160244e1 | 1.125e0 |
| sparse 64-logit/row mask | layer39 routed-up | 1e-2 | -1.678510904e0 | -1.984786987e0 | 1.543e-1 |

The sparse 1-logit `eps=1e-1` reachability run nearly matched, which shows the
analytic gradient is not simply fake. But at useful smaller eps the selected
logit deltas quantize to zero; broader sparse probes reintroduce noisy
full-model loss behavior.

Local gates for the diagnostic code:

```text
cargo fmt --check
PASS

cargo test -p autograd --release moe_topk_softmax_with_indices_freezes_forward_and_backward
PASS

cargo check -p train --release --no-default-features --features no-cuda --example qwen36_fp8_lora_fd_gate
PASS

CUDARC_CUDA_VERSION=12090 cargo check -p train --release --no-default-features --features cuda,no-cuda --example qwen36_fp8_lora_fd_gate
PASS

cargo clippy -p train --release --no-default-features --features no-cuda --example qwen36_fp8_lora_fd_gate -- -D warnings
PASS

cargo clippy -p autograd --release --no-default-features --features no-cuda -- -D warnings
PASS
```

## Root Cause

Dynamic route boundaries were a real confounder, but not the whole cause of
the full-model FD failure. After freezing all MoE top-k ids, the full-model
scalar-loss FD remains unstable because the loss oracle is still too noisy for
35B FP8 full-logit central differences:

- all-logit random probes sum about 747k logits and amplify f32 accumulation /
  output noise;
- very sparse probes avoid that sum but individual logit deltas are below the
  useful output quantization window at eps near 1e-3;
- increasing eps can recover reachability but is not a clean 1% finite-diff
  license for the full model.

The A9/A12/A14 MLP-layer real-checkpoint FD remains the clean gradient license
for routed expert LoRA. This full-model diagnostic does not supersede it.

## Fix

Keep the route-frozen diagnostic code because it removes one known confounder
and makes the failure reproducible. Do not treat `--freeze-base-routes` as a
passing full-model gradient gate yet.

Next valid routes:

1. Build a narrower full-model oracle that avoids full-vocab f32 reductions and
   avoids per-logit quantization underflow, then rerun route-frozen FD.
2. Treat downstream full-model MatmulBT / MoE backward walls as performance
   work while continuing to use MLP-layer FD for correctness.
3. If a true full-model scalar oracle is required, compute the scalar in a
   higher-precision host path over a small deterministic logit subset and prove
   same-function reachability before using it as a license.

## Rule

Freezing MoE routes is necessary but not sufficient for a 35B full-model FD
oracle. The scalar loss itself must have a measured numeric window; otherwise
the FD result is another confounded verdict.
