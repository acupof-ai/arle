# Vulkan backend bring-up pending remote

## Context

P0 adds the feature-gated Vulkan runtime substrate for the AIPC lane:
`crates/vulkan-sys` dynamically loads Vulkan through `ash` and stays stubbed
by default so Mac/off-box builds do not require a Vulkan SDK or loader.
P1 adds `crates/vulkan-kernels`, a `glslc` build wrapper around the vendored
llama.cpp Vulkan shader corpus plus raw-buffer launch wrappers for the generic
operator set.
P2 adds `crates/infer-vulkan`, a seam-correct backend skeleton that reuses
`infer_hip::{gguf,dequant,config}` and pins the dense Qwen3 forward sequence
against the CUDA implementation.
P3 adds the DSv4 fallback-forward contract: non-FlashMLA operator order,
per-layer RoPE theta switch, `enable_prefix_cache=false`, and the exact
mutated slot-buffer enumeration.
P4 adds the Qwen3.5 hybrid contract: 3:1 linear/full attention structure,
gated-delta recurrent state, conv4 ring state, hd256 full attention, and MLP
order.
P5 adds the Qwen3.6 MoE contract on top of Qwen3.5 hybrid attention: router
GEMV, top-k, routed expert gate/up/down, shared expert, and weighted expert
mix.
P6 adds `crates/gemma-spec` and the Gemma4 Vulkan text contract: nested
`text_config` parsing, sliding/global layer typing with final-global
validation, PLE fields, global KV/p-RoPE fields, QK-norm, GeGLU, and the
Gemma RMSNorm(+1) convention.

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
cargo check -p infer-vulkan
cargo check -p infer-vulkan --features vulkan
cargo test -p infer-vulkan
cargo test -p infer-vulkan --features vulkan
cargo clippy -p infer-vulkan --all-features -- -D warnings
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

P3 numeric DSv4 execution is blocked independently: CSA/compressor/hybrid
attention have no vendored standalone Vulkan GLSL equivalent. They need either
new `.comp` ports from `dsv4_attention.cu` semantics or a measured composition
from primitives before DSv4 can run on Vulkan.

P4 numeric Qwen3.5 execution is blocked independently: conv4 and recurrent
gated-delta kernels do not exist in the vendored Vulkan corpus and need Vulkan
ports before hybrid layers can execute.

P5 numeric Qwen3.6 execution is blocked independently: MoE router/top-k and
expert-mix launch integration is not implemented for Vulkan yet.

P6 numeric Gemma4 execution is blocked independently: p-RoPE/global-KV sharing
and Gemma RMSNorm(+1) kernels are not validated on Vulkan yet.

## Learnings

Keep Vulkan off by default and use dynamic loader mode (`ash` `loaded`) so
off-box typecheck does not depend on system Vulkan libraries. The bench entry
stays `pending-remote` until the AIPC box can run the real Vulkan device path.
