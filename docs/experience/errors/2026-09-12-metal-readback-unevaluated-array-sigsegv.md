# Metal SIGSEGV: readback of unevaluated MLX arrays in device-matmul fallbacks

## Context

Two Metal tests crashed with `SIGSEGV (KERN_INVALID_ADDRESS at 0x0)` on main:

- `autograd` integration test `metal_add_broadcast_forward_stays_lazy`
  (`crates/autograd/tests/test_device_handle.rs:1386`).
- `train` integration test `qwen35_hybrid_forward_matches_cpu_on_metal_backend`
  (`crates/train/tests/test_qwen35_hybrid_forward_metal.rs:14`).

Crash reports (`~/Library/Logs/DiagnosticReports/*.ips`) both fault in the same
C++ frame:

```
mlx::core::allocator::Buffer::raw_ptr()
mlx::core::array::data<float>()
mlx_array_data_float32
autograd::backend_metal::MetalBackend::readback
```

## Root Cause

`MetalBackend::readback` called `mlx_array_data_float32` directly, with the
precondition "the caller has evaluated the array". The `Backend` trait-default
device fallbacks do not honor that precondition:

- `matmul_bt` (`crates/autograd/src/backend.rs:1213`) and
  `matmul_backward_device` (`:1272`) call `self.readback(a/b/grad)` on handles
  that are lazy unevaluated MLX nodes.
- `write_slice_device` (`:2404`) and the other host-fallback defaults do the
  same.

`MetalBackend` overrode only `matmul`/`matmul_forward`/`matmul_backward`; the
`*_device`/`matmul_bt` methods fell through to the host-fallback defaults.
MLX represents an unrealized array with a null data buffer, so
`array::data<float>()->raw_ptr()` dereferences address 0. The pre-backward
batch flush (`tape.rs:659`, `prefers_pre_backward_flush`) realizes tape
*outputs*, but the backward GEMMs also read saved *operands* (weights,
intermediate lazy nodes) that were never flushed, so the precondition still
failed.

The two tests reached two different defaults: the autograd test faulted in
`matmul_backward_device`; after that was routed on-device, the train test
faulted one frame later in `write_slice_device` — same defect class.

## Fix

1. `MetalBackend::readback` now realizes the array through the existing
   `eval_and_readback` tail instead of touching the buffer directly
   (`crates/autograd/src/backend_metal.rs`). A new bridge binding
   `mlx_array_is_available` (MLX `array::is_available()`) gates the eval:
   already-realized arrays (the batch-flush path) skip the eval boundary, so
   `eval_count` accounting and the test's `<= 4` bound stay honest.
2. `MetalBackend` gains lazy on-device overrides for the four matmul entry
   points the hot backward path uses — `matmul_bt`,
   `matmul_backward_device`, `matmul_bt_backward_device`,
   `matmul_bt_input_grad_device` — built from `mlx_transpose(_axes)` +
   `mlx_matmul`, returning unevaluated nodes, matching the CUDA override set.
   This keeps the OPD backward GEMMs on-device instead of relying on the
   readback safety net (which would host-round-trip the large weight-grad
   GEMMs; see `ops/matmul.rs:143`).

## Rule

An FFI "data pointer" accessor on a lazy array backend needs no precondition —
the readback wrapper owns realization. A trait-default fallback that assumes a
realized buffer is a latent SIGSEGV on every backend whose arrays start lazy;
either override it for that backend or make readback realize. One crash frame
moving to the next default after a fix means the whole fallback class is
unverified, not that the fix was wrong.

## Verification

Private target dir (`CARGO_TARGET_DIR=/tmp/metal-sigsegv-target`, deleted
afterward):

- Before: both tests exit 139/101 with the `raw_ptr` frame.
- After: `test_device_handle` 23/23 pass; all autograd `--features metal`
  integration tests pass (82 total); `test_qwen35_hybrid_forward_metal`
  passes.
