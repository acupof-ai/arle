# A0 MoE LoRA Finite-Diff Gate

## Context

A0 is the de-risk gate for native MoE training: a small routed MoE layer with
LoRA adapters must produce clean gradients before larger CUDA MoE backward work
is licensed.

## What Worked

- Replaced the prior absolute-tolerance-only finite-diff verdict with a central
  difference gate at `eps=1e-3`.
- Numeric derivative now computes `dot(out_plus - out_minus, probe) / (2 eps)`
  directly, avoiding scalar-loss cancellation.
- Non-tiny gradients (`>=2e-4`) use relative tolerance `1e-2`; smaller values
  are constrained by a strict `3e-6` absolute noise floor.
- Added a CUDA backend variant of the same A0 gate.

## Evidence

Local CPU:

```text
cargo test -p train --release --test test_moe_a0 -- --nocapture
a0_moe_finite_diff backend=cpu eps=1.0e-3 checked_values=4864 relative_values=1037 tiny_values=3827 max_rel=6.956220e-3 max_tiny_abs=2.136566e-6 tiny_abs_failures=0
test result: ok. 1 passed
```

Remote GPU0 CUDA:

```text
CUDA_VISIBLE_DEVICES=0 CUDARC_CUDA_VERSION=12090 CUDA_HOME=/usr/local/cuda ARLE_CUDA_TEST_DEVICE=0 cargo test -p train --release --features cuda --test test_moe_a0 -- --nocapture
a0_moe_finite_diff backend=cpu  eps=1.0e-3 checked_values=4864 relative_values=1037 tiny_values=3827 max_rel=6.916606e-3 tiny_abs_failures=0
a0_moe_finite_diff backend=cuda eps=1.0e-3 checked_values=4864 relative_values=1037 tiny_values=3827 max_rel=7.155292e-3 tiny_abs_failures=0
test result: ok. 2 passed
```

## Rule

Do not license CUDA MoE backward from an absolute-tolerance finite-diff pass.
For small gradients, separate estimator noise from meaningful gradients; for
meaningful gradients, require relative agreement.
