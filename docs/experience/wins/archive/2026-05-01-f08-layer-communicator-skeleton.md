# F0.8 LayerCommunicator skeleton

## Context

Single-node multi-GPU F0.8 needs a model-level communication surface before F2
wires actual TP/DP/CP forward paths. This tranche intentionally does not touch
Qwen forward call sites or scheduler execution.

## What Worked

- Added `infer/src/model/layer_communicator.rs` with a `LayerCommunicator`
  skeleton for post-attention all-reduce, post-MLP all-reduce, DP-attention
  gather/scatter, CP split/gather, and future fused residual/RMSNorm collectives.
- Default single-rank behavior is exact pass-through/no-op.
- Multi-rank calls reject early until F1/F2 wires real group coordinators and
  CUDA collective buffers.

Validation:

```bash
cargo fmt
ZIG=/tmp/zig14/zig cargo test -p infer model::layer_communicator --features cuda
ZIG=/tmp/zig14/zig cargo check -p infer --no-default-features --features cuda,no-cuda
```

`ZIG=/tmp/zig14/zig cargo clippy -p infer --features cuda -- -D warnings`
remains blocked by existing crate-wide lint debt beginning at
`infer/src/model/common.rs`, `infer/src/model/generation_state.rs`, and
`infer/src/model/qwen3/lora.rs`; the new `LayerCommunicator` lint issue found
during validation was fixed before landing.

No GuideLLM bench was run because this is an inert model-surface skeleton and no
runtime call site changes.

## Rule

Land model communication surfaces as single-rank no-op first, with tests that
prove buffers are unchanged, then wire real collectives only in the TP forward
tranche.
