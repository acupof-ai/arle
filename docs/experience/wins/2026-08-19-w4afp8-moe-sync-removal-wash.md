# W4AFP8 MoE path: per-GEMM sync removal — measured wash — 2026-08-19

> Status: Verified on GPU (H20, W4AFP8 MoE decode). Shipped in `c4b922d5b`.

## Context

The W4AFP8 grouped-GEMM wrapper called `cudaStreamSynchronize` after every GEMM
— 86 syncs per step at 43 layers — serializing the CPU-GPU pipeline. The same
forward path also made 43 `mem_get_info` driver calls per step for logging, and
allocated the 32 MB CUTLASS workspace with `alloc_zeros` though CUTLASS only
writes it.

## What Worked

Removing the per-GEMM syncs, the hot-path `mem_get_info` logging, and switching
the workspace to `alloc`:

- Decode: 37 tok/s, unchanged — the GPU is the bottleneck, not the CPU.
- Prefill and correctness: unchanged.

The stream already orders the kernels; async errors surface at the next sync
point (output readback or step boundary), so the per-GEMM syncs bought no
observability that the step boundary does not already provide.

## Rule

A per-kernel sync in a stream-ordered pipeline is serialization, not error
handling — the next natural sync point (readback, step boundary) surfaces the
same errors. On a GPU-bound decode path the removal is a free wash; measure to
prove the wash rather than assuming the syncs were load-bearing.
