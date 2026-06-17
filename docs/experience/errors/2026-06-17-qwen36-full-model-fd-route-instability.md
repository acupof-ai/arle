# Qwen3.6 full-model finite diff crosses MoE route boundaries

## Context

A14 removed the sparse-MLP forward padding wall, but the full-model Qwen3.6
FP8 LoRA finite-diff gate still failed by orders of magnitude. The open
hypothesis was whether the central-diff perturbation crossed discrete MoE
router boundaries, making the scalar full-logit loss non-smooth, or whether the
failure was primarily a backward ownership bug.

This tranche added a diagnostic route-stability probe to
`qwen36_fp8_lora_fd_gate`: in `--mode full-model`, `--check-route-stability`
records every MoE layer's top-k expert ids for the base, plus, and minus
forward passes and prints changed layers/slots before the finite-diff verdict.

## Evidence

Remote `.62`, model `/data01/models/Qwen3.6-35B-A3B-FP8`, GPU3 avoided.
Source:
`/data01/arle-track1-opd-rollout-infer-202606170646`, target:
`/data01/arle-target-track1-opd-rollout-infer-202606170646`.

Build:

```text
CUDA_HOME=/usr/local/cuda CUDARC_CUDA_VERSION=12090
ARLE_CUDA_DISABLE_FLASHMLA=1 INFER_TILELANG_PYTHON=/root/tl-venv/bin/python
cargo build -p train --example qwen36_fp8_lora_fd_gate --release --no-default-features --features cuda
PASS: Finished release profile in 3m52s
```

The original full-model FD shape with `eps=1e-3` crosses many routes:

```text
qwen36_fp8_lora_route_stability base_layers=40 plus_layers=40 minus_layers=40
plus_changed_layers=4 plus_changed_slots=10 plus_total_slots=960
minus_changed_layers=27 minus_changed_slots=119 minus_total_slots=960

qwen36_fp8_lora_fd_gate_result eps=1.0e-3
analytic=3.636742830e-1 numeric=-2.372965698e2 rel_err=1.002e0
```

Smaller eps values do not produce a usable smooth central-diff window:

| eps | plus changed slots | minus changed slots | numeric | verdict |
|---:|---:|---:|---:|---|
| 1e-3 | 10 / 960 | 119 / 960 | -2.372965698e2 | route-confounded |
| 1e-4 | 96 / 960 | 96 / 960 | -1.083087921e2 | route-confounded |
| 1e-5 | 13 / 960 | 18 / 960 | -1.767244336e4 | route-confounded |
| 1e-6 | 53 / 960 | 53 / 960 | -1.930570703e4 | route-confounded |
| 1e-8 | 0 / 960 | 0 / 960 | 0.0 | below numeric resolution |

The near-zero control is important: route collection itself is deterministic
when the perturbation is negligible (`0 / 960` route changes), but the loss
difference is then quantized to zero. The finite-diff failure is therefore not
a random replay artifact; it is a dynamic-router smoothness problem at useful
epsilon.

Local gates for the diagnostic code:

```text
cargo fmt --check
PASS

cargo check -p train --release --no-default-features --features no-cuda --example qwen36_fp8_lora_fd_gate
PASS

CUDARC_CUDA_VERSION=12090 cargo check -p train --release --no-default-features --features cuda,no-cuda --example qwen36_fp8_lora_fd_gate
PASS

cargo clippy -p train --release --no-default-features --features no-cuda --example qwen36_fp8_lora_fd_gate -- -D warnings
PASS

cargo check -p train --release --no-default-features --features no-cuda --lib
PASS

CUDARC_CUDA_VERSION=12090 cargo check -p train --release --no-default-features --features cuda,no-cuda --lib
PASS
```

## Root Cause

The full-model scalar-loss central finite difference is crossing MoE top-k
route boundaries in later layers. At eps values large enough to produce a
measurable loss delta, the plus/minus arms no longer evaluate the same
piecewise-smooth function as the analytic backward. At eps small enough to
preserve routes, the full-model loss delta is below the useful numeric
resolution of the current gate.

This does not invalidate the A9/A12/A14 real-checkpoint MLP-layer gradient
license. It invalidates this full-model dynamic-routing finite-diff gate as a
gradient oracle.

## Fix

Do not use unfrozen-route full-model central diff as the 35B gradient license.
The next valid options are:

1. Add a route-frozen full-model diagnostic that reuses the base top-k routes
   for plus/minus/analytic, so all arms evaluate the same local function.
2. Keep the licensed MLP-layer finite-diff gate for routed expert gradients and
   separately optimize the measured full-model backward walls.
3. Audit the remaining full-model `MatmulBT` wall as a performance issue, not
   as an explanation for the finite-diff correctness failure.

## Rule

For routed MoE models, full-model finite difference must prove route stability
or freeze routes. A central-diff mismatch across different top-k route sets is
not a gradient bug verdict.
