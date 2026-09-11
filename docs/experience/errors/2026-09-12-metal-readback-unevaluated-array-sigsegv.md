# Metal SIGSEGV: readback of unevaluated MLX arrays

## Context

Two Metal tests crashed with `SIGSEGV (KERN_INVALID_ADDRESS at 0x0)` on main:

- `autograd` integration test `metal_add_broadcast_forward_stays_lazy`
  (`crates/autograd/tests/test_device_handle.rs:1386`).
- `train` integration test `qwen35_hybrid_forward_matches_cpu_on_metal_backend`
  (`crates/train/tests/test_qwen35_hybrid_forward_metal.rs:14`).

Both crash reports (`~/Library/Logs/DiagnosticReports/*.ips`) fault in the
same C++ frame:

```
mlx::core::allocator::Buffer::raw_ptr()
mlx::core::array::data<float>()
mlx_array_data_float32
autograd::backend_metal::MetalBackend::readback
```

## Root Cause

`MetalBackend::readback` called `mlx_array_data_float32` directly, on the
precondition that the caller had already evaluated the array. The `Backend`
trait-default device fallbacks do not honor that precondition:

- `matmul_bt` (`crates/autograd/src/backend.rs:1213`) and
  `matmul_backward_device` (`:1272`) call `self.readback(...)` on handles
  that are lazy, unevaluated MLX nodes.
- `write_slice_device` (`:2404`) and the other host-fallback defaults do the
  same.

An unrealized MLX array has a null data buffer, so
`array::data<float>()->raw_ptr()` dereferences address 0. The pre-backward
batch flush (`tape.rs`, `prefers_pre_backward_flush`) realizes tape
*outputs*, but the backward GEMMs also read saved *operands* (weights and
intermediate lazy nodes) that were never flushed, so the precondition still
failed. The two tests faulted in two different defaults (`matmul_backward_device`,
then one frame later `write_slice_device`) — one defect class.

## Fix

`MetalBackend::readback` realizes the array before touching its buffer when
MLX reports it is not yet available, and copies directly when it is. The
gate is a new bridge binding `mlx_array_is_available` (MLX
`array::is_available()`): already-realized arrays — the batched pre-backward
flush path — skip the eval boundary entirely, so `eval_count` accounting and
the `metal_add_broadcast…` test's `<= 4` bound are unchanged.

This is a correctness fix only; the Metal backward still runs through the
trait-default host fallbacks. Moving the backward GEMMs to on-device Metal
overrides is a separate runtime-path change with its own tests and
before/after measurement.

## Rule

An FFI data-pointer accessor on a lazy-array backend carries no "caller
evaluated" precondition — the readback wrapper owns realization. A
trait-default fallback that assumes a realized buffer is a latent SIGSEGV on
every backend whose arrays start lazy.

## Verification

Private target dir (`CARGO_TARGET_DIR` under `/tmp`, deleted afterward),
features `--no-default-features --features metal`:

- Before: both tests exit 101/139 with the `raw_ptr` frame.
- After: `test_device_handle` 23/23 pass (incl.
  `metal_add_broadcast_forward_stays_lazy`); all `autograd --features metal`
  integration tests pass (80 total); `test_qwen35_hybrid_forward_metal`
  passes.
- `cargo clippy -p autograd --features metal --lib -- -D warnings` clean;
  `--no-default-features --all-targets` clean.
