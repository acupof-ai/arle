# 64K reaches backward, then checkpoint replay rebuilds the full MLP

## Context

Candidate `a9bb1e49` plus autograd diff `ac9fd401` passed 192 CUDA autograd
tests. A single-H20 `agent-opd --synthetic-writeback-seq 65536` run completed
all 64 forward groups in 734.963 seconds and fused CE in 2.936 seconds.

## Root Cause

Backward's first 4.56 GiB allocation is exactly
`65536 * intermediate_size(17408) * sizeof(f32)`: the dense MLP
`mul(silu(gate), up)` inside checkpoint replay.

`checkpoint_seq_chunked` chunked backward replay but still ran its replay
forward once at full sequence length. The run also discarded the result of
`trim_memory_pool`, leaving phase-boundary reclamation unverifiable.

This clears the earlier index wall but does not license 64K: forward completion
is not an optimizer step.

## Fix

Chunk the position-wise replay in forward and backward. Move the required trim
into the shared OPD backward entry, bind the CUDA context, and propagate errors.
Remove the caller-specific best-effort trim.

Remote validation is pending.

## Rule

Chunk both halves of a recompute contract. Phase-boundary reclamation is
required, not best effort.
