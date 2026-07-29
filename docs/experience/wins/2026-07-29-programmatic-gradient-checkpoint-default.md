# Programmatic OPD keeps gradient checkpointing enabled

## Context

`TrainRuntimeFlags::default()` enabled gradient checkpointing, but the backing
atomic disabled it until the CLI applied flags. Direct library callers therefore
used a different, memory-unsafe default.

## What Worked

Initialize the atomic to the public default. CLI behavior is unchanged.

Validation: release `train` clippy and `autograd` tests pass. Long-context GPU
validation is covered by the single-GPU 64K→256K ladder.

## Rule

Runtime storage must start with the same value exposed by its public default.
