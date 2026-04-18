# MLX matmul SIGSEGV under cargo test parallelism

## Context

Full-workspace Metal regression pass
(`cargo test --release --no-default-features --features metal -p train -p autograd`)
blew up with `signal: 11, SIGSEGV` inside `tests/test_backend.rs` after the
first test passed. Rerunning with `--test-threads=1` ran all four tests green.

The test file only exercises `autograd::MetalBackend::matmul_forward`, which
wraps `mlx_array_from_data` → `mlx_matmul` → `mlx_eval` → copy-out →
`mlx_array_free`. Each test is self-contained; there is no shared state
between them on the Rust side.

## Root Cause

MLX's default stream / device / allocator are process-global. `mlx_matmul`
dispatched concurrently from two threads on the default stream hits a race
inside MLX's C++ allocator or command-buffer dispatch, and the MLX process
segfaults. MLX does not document the default-stream dispatch path as
thread-safe, and the mlx-sys bridge doesn't add any cross-thread guard
(the crate AGENTS.md mentions `mlx_last_error()` is thread-local but says
nothing about the compute path).

cargo test's default parallelism (`N = num_cpus`) triggers this the moment
two `#[test]` functions both pass a live matmul into MLX at once.

## Fix

`crates/autograd/src/backend_metal.rs` now holds a `static MLX_GUARD: Mutex<()>`
and locks it for the duration of each `matmul_forward` FFI round-trip.
Training is single-threaded, so this mutex is effectively free on the happy
path; cargo test parallelism gets serialized across Metal calls.

## Rule

Any new MLX FFI call from `autograd`'s Metal backend must take `MLX_GUARD`
before `mlx_array_from_data` and hold it until the last `mlx_array_free`.
Don't assume per-backend-instance mutexes are enough — MLX's state is
process-global, so the guard must be `static`.

If we later need concurrent Metal dispatch (e.g. overlapping prefill + decode
from the scheduler), that's an MLX-side fix (explicit per-thread streams +
per-op stream argument), not a Rust-side workaround.
