# A1 Attention Recompute Finite-Diff Gate

## Context

A1 needs a clean attention gradient path before OPD can move from rollout-only
speedups to native 35B training. The previous full-attention training path was a
composite `matmul -> softmax -> matmul` tape, so backward kept the full
attention graph and its intermediate scores/probs.

## What Worked

- Added a single `CausalSdpaRecompute` autograd op for causal full attention.
- Forward reuses the existing device-capable SDPA implementation with the inner
  tape disabled, so the outer tape records only `q/k/v`.
- Backward recomputes causal scores and softmax in f32 from `q/k/v` and returns
  direct `dq/dk/dv` gradients.
- Switched Qwen full-attention training to the recompute op; rollout decode and
  KV-cache paths remain unchanged.
- Added an HD256 finite-diff gate matching the Qwen3.6 head dimension.

## Evidence

Local CPU, exhaustive `q/k/v` check:

```text
cargo test -p train --release --test test_attention_a1 -- --nocapture
a1_attention_finite_diff backend=cpu eps=1.0e-3 max_per_tensor=2048 checked_values=6144 relative_values=2044 tiny_values=4100 max_abs_at_worst_rel=2.810158e-7 max_rel=1.193557e-3 worst=v[379] analytic=-2.354439e-4 numeric=-2.351629e-4 max_tiny_abs=1.835457e-5 tiny_abs_failures=0
test result: ok. 1 passed
```

Local type/lint:

```text
CUDARC_CUDA_VERSION=12090 cargo check -p train --release --no-default-features --features cuda,no-cuda --tests
cargo clippy -p train --release --test test_attention_a1 -- -D warnings
```

Remote H20 GPU1 CUDA, clean temporary tree under `/data01/arle-a1-verify`:

```text
CUDA_VISIBLE_DEVICES=1 ARLE_CUDA_TEST_DEVICE=0 CUDARC_CUDA_VERSION=12090 CARGO_TARGET_DIR=/data01/arle-target-a1 cargo test -p train --release --no-default-features --features cuda --test test_attention_a1 -- --nocapture
a1_attention_finite_diff backend=cpu eps=1.0e-3 max_per_tensor=2048 checked_values=6144 relative_values=2044 tiny_values=4100 max_abs_at_worst_rel=2.810158e-7 max_rel=1.193557e-3 worst=v[379] analytic=-2.354439e-4 numeric=-2.351629e-4 max_tiny_abs=1.835457e-5 tiny_abs_failures=0
a1_attention_finite_diff backend=cuda eps=1.0e-3 max_per_tensor=128 checked_values=387 relative_values=129 tiny_values=258 max_abs_at_worst_rel=2.959155e-7 max_rel=9.878366e-5 worst=v[16] analytic=-2.995295e-3 numeric=-2.995591e-3 max_tiny_abs=1.332444e-5 tiny_abs_failures=0
test result: ok. 2 passed
```

## Limits

This licenses the A1 correctness shape and recompute-in-backward tape semantics.
It does not yet license a native CUDA flash-attention backward path; the current
backward implementation is host f32 recompute after readback, so the next A1
tranche is device-resident/flash backward performance.

## Rule

For attention gradients, validate the production head dimension with a
central-diff `dot(out_plus - out_minus, probe) / (2 eps)` gate. Use relative
tolerance on meaningful gradients and a separate f32 estimator-noise bound for
tiny gradients.
