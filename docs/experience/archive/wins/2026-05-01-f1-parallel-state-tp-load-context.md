# F1 parallel state and TP load-context foundation

## Context

Single-node multi-GPU F1 software-layer tranche after the F0 NCCL smoke:
parallel-state accessors, group-coordinator surface, env rendezvous parsing,
and shard-aware BF16 weight-load helpers. This is a foundation commit, not a
throughput claim, so no GuideLLM row is attached.

## What Worked

- Added SGLang-style parallel-state metadata for world, TP, PP, EP,
  attention-TP/DP/CP, and MoE-TP/EP/DP groups.
- Added a `GroupCoordinator` surface that no-ops for single-rank groups and can
  wrap the existing NCCL smoke group for f32 all-reduce, all-gather, and
  broadcast.
- Extended env bootstrap parsing for `MASTER_ADDR`, `MASTER_PORT`, `RANK`,
  `WORLD_SIZE`, `LOCAL_RANK`, and `INFER_CUDA_DEVICE`.
- Added `TpLoadContext` constructors for row/column/head sharding over the
  existing `tensor_parallel.rs` math.
- Added shard-aware BF16 matrix slicing for weight loading. Mock TP=2 tests
  verify row and column shards have expected shapes and preserve the full
  matrix sum across ranks.
- Codex review caught two F1 surface issues before landing: column-parallel
  linear weights must shard output rows for existing `[out_dim, in_dim]`
  matrices, and NCCL broadcast needs an explicit count so non-root ranks do not
  derive collective length from placeholder input.
- A second review pass caught full-tensor host materialization in the sharded
  loader; the helper now converts only the selected byte ranges for the local TP
  shard.
- A third review pass caught NCCL all-gather's fixed sendcount constraint; the
  coordinator now requires an explicit uniform per-rank count and rejects
  uneven inputs instead of allowing a collective hang.

Validation passed:

```bash
ZIG=/tmp/zig14/zig cargo test -p infer distributed --features cuda
ZIG=/tmp/zig14/zig cargo test -p infer tp --features cuda
ZIG=/tmp/zig14/zig cargo test -p infer tensor_parallel --features cuda
ZIG=/tmp/zig14/zig cargo test -p infer weight_loader::tests::tp --features cuda
ZIG=/tmp/zig14/zig cargo check -p infer --features nccl
ZIG=/tmp/zig14/zig cargo check -p infer --no-default-features --features cuda,no-cuda
```

Full-suite notes:

- `ZIG=/tmp/zig14/zig cargo test -p infer --features cuda` currently fails in
  unrelated existing areas: HostPinnedPool tests hit `CUDA_ERROR_NOT_INITIALIZED`,
  Qwen3.5 tests require a missing `infer/models/Qwen3.5-4B`, several ops tests
  hit `CUDA_ERROR_MISALIGNED_ADDRESS`, and two scheduler budget assertions were
  already outside this F1 surface.
- `ZIG=/tmp/zig14/zig cargo clippy -p infer --features cuda -- -D warnings`
  remains blocked by existing crate-wide lint debt beginning at
  `infer/src/model/common.rs`, `infer/src/model/generation_state.rs`,
  `infer/src/model/qwen3/lora.rs`, and `infer/src/model/qwen35/*`.

## Rule

For F1 foundation work, keep the default single-rank path inert and verify the
new surfaces with metadata and shard-shape tests before wiring Qwen forward or
scheduler execution.
