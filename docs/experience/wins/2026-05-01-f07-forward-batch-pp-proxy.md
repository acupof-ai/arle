# F0.7 ForwardBatch PP proxy type surface

## Context

Single-node multi-GPU F0.7 needs a scheduler-visible slot for future
pipeline-parallel intermediate tensors. This tranche is type-only and does not
change CUDA execution, model forward signatures, or request scheduling.

## What Worked

- Added `infer/src/scheduler/forward_batch.rs` with `ForwardBatch`,
  `ForwardBatchKind`, `IntermediateTensors`, and `IntermediateTensorMeta`.
- Added `ForwardBatch::pp_proxy: Option<IntermediateTensors>` with default
  `None`.
- Added unit tests covering `ForwardBatch::new`, decode construction, and
  attach/clear behavior for the PP proxy slot.

Validation:

```bash
cargo fmt
ZIG=/tmp/zig14/zig cargo test -p infer scheduler::forward_batch --features cuda
ZIG=/tmp/zig14/zig cargo check -p infer --no-default-features --features cuda,no-cuda
```

No GuideLLM bench was run because this is an inert type surface with no runtime
call-site changes.

## Rule

Reserve PP proxy metadata early, but do not consume it in model or scheduler
execution until the F2 pipeline-parallel wiring tranche.
