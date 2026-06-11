# Vulkan backend bring-up pending remote

## Context

P0 adds the feature-gated Vulkan runtime substrate for the AIPC lane:
`crates/vulkan-sys` dynamically loads Vulkan through `ash` and stays stubbed
by default so Mac/off-box builds do not require a Vulkan SDK or loader.
P1 adds `crates/vulkan-kernels`, a `glslc` build wrapper around the vendored
llama.cpp Vulkan shader corpus plus raw-buffer launch wrappers for the generic
operator set.

The backend is not reachable from `arle serve` yet. CLI and model wiring are
scheduled for P7, so there is no production hot path or benchmarkable endpoint
on this Mac.

## Results

Off-box checks:

```text
cargo check -p vulkan-sys
cargo check -p vulkan-sys --features vulkan
cargo test -p vulkan-sys
cargo test -p vulkan-sys --features vulkan
cargo check
cargo check --features vulkan
cargo test
cargo clippy -p vulkan-sys --all-features -- -D warnings
cargo check -p vulkan-kernels
cargo check -p vulkan-kernels --features vulkan
cargo test -p vulkan-kernels
cargo test -p vulkan-kernels --features vulkan
cargo clippy -p vulkan-kernels --all-features -- -D warnings
```

All passed on 2026-06-11. `cargo check --features vulkan` also passed after
P1.

On-box validation is pending: SPIR-V compile path, Vulkan loader/device
enumeration on the AIPC target, numeric correctness, and throughput.

## Problems

The exact root `cargo clippy --all-features -- -D warnings` gate is blocked
before P0 completion because all-features enables unrelated pre-existing CUDA
and autograd code paths on this Mac. The visible blockers include protected
dirty `crates/infer-cuda/src/moe.rs` errors and an `autograd` CUDA match gap.

P1 has a second pending item: this Mac has `glslc`, and several vendored
llama.cpp shader variants warn-and-skip because ARLE has not yet replicated
llama.cpp's full specialization matrix for compile-time defines and push
constant layouts. The Rust launchers fail loud with `ShaderMissing` for those
variants until the AIPC/on-box compile matrix is fixed.

## Learnings

Keep Vulkan off by default and use dynamic loader mode (`ash` `loaded`) so
off-box typecheck does not depend on system Vulkan libraries. The bench entry
stays `pending-remote` until the AIPC box can run the real Vulkan device path.
