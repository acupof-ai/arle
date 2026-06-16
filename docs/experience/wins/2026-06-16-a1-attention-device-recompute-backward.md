# A1 Attention Device Recompute Backward

## Context

The first A1 tranche licensed the causal SDPA recompute math, but the backward
implementation still called `tensor_host(q/k/v/upstream)` and broke the CUDA
device-resident gradient chain. That was a correctness gate, not a usable 35B
training path.

## What Worked

- Moved the causal SDPA recompute CPU reference into `backend.rs` so host
  fallback and tests share one formula.
- Added a device-resident backward path for `CausalSdpaRecompute`:
  recompute `probs` under a disabled inner tape, then use existing device
  `matmul`, `softmax_backward`, `mul_scalar`, and layout ops to produce
  `dq/dk/dv`.
- Ensured the saved original `q/k/v` tensor ids have device handles before the
  forward records the single outer tape entry.
- Added `sum_backward_device`, because the A1 test loss used `sum` and the
  scalar seed was otherwise expanded into a host gradient before attention.
- Extended the CUDA finite-diff gate with a residency assertion: `dq/dk/dv` must
  be device handles before the test reads them back for numeric comparison.

## Evidence

Local CPU:

```text
cargo test -p train --release --test test_attention_a1 -- --nocapture
a1_attention_finite_diff backend=cpu eps=1.0e-3 max_per_tensor=2048 checked_values=6144 relative_values=2044 tiny_values=4100 max_abs_at_worst_rel=2.810158e-7 max_rel=1.193557e-3 worst=v[379] analytic=-2.354439e-4 numeric=-2.351629e-4 max_tiny_abs=1.835457e-5 tiny_abs_failures=0
test result: ok. 1 passed
```

Local scoped type/lint:

```text
CUDARC_CUDA_VERSION=12090 cargo check -p autograd --release --no-default-features --features cuda,no-cuda --lib
CUDARC_CUDA_VERSION=12090 cargo clippy -p autograd --release --no-default-features --features cuda,no-cuda --lib -- -D warnings
cargo clippy -p train --release --test test_attention_a1 -- -D warnings
```

Remote H20 GPU2 CUDA, clean temporary tree under
`/data01/arle-a1-device-verify`:

```text
CUDA_VISIBLE_DEVICES=2 ARLE_CUDA_TEST_DEVICE=0 CUDARC_CUDA_VERSION=12090 CARGO_TARGET_DIR=/data01/arle-target-a1 cargo test -p train --release --no-default-features --features cuda --test test_attention_a1 -- --nocapture
a1_attention_finite_diff backend=cuda eps=1.0e-3 max_per_tensor=128 checked_values=387 relative_values=129 tiny_values=258 max_abs_at_worst_rel=2.956826e-7 max_rel=9.870593e-5 worst=v[16] analytic=-2.995296e-3 numeric=-2.995591e-3 max_tiny_abs=1.332444e-5 tiny_abs_failures=0
a1_attention_finite_diff backend=cpu eps=1.0e-3 max_per_tensor=2048 checked_values=6144 relative_values=2044 tiny_values=4100 max_abs_at_worst_rel=2.810158e-7 max_rel=1.193557e-3 worst=v[379] analytic=-2.354439e-4 numeric=-2.351629e-4 max_tiny_abs=1.835457e-5 tiny_abs_failures=0
test result: ok. 2 passed
```

## Limits

This removes the CUDA host-readback break in A1 attention backward and keeps
`dq/dk/dv` device-resident. It is still a recompute composition over existing
device primitives, not a native FlashAttention backward kernel. The next A1
performance tranche is to replace the recompute composition with an adopted or
native flash backward path and measure full OPD step wall-clock.

## Rule

Gradient numeric parity is not enough for a training-runtime milestone. The
gate also has to prove residency at the exact op boundary that used to read
back, otherwise a "correct" gradient can still be unusable at 35B scale.
