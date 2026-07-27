# CUDA broadcast_copy_f32 drops the device zeros carrier — repeat_kv, 2026-07-27

> Status: shipped, byte-identical verified on H20. Memory win structural
> (an alloc + memset removed per call); wall-clock A/B pending (sub-ms per call,
> below the step-timer floor).

## Goal

`repeat_kv`'s GQA-expand (`broadcast_expand`) must not allocate a full-size
zeroed carrier on CUDA.

## Context

`a61d44579` replaced `repeat_kv`'s `add_broadcast(zeros, x)` with a
`broadcast_expand` op to drop the full-size zeros *tape tensor*. But the CUDA
backend implemented it by reusing `add_broadcast_f32` as `out = 0 +
src_broadcast` — allocating **two** target-size device buffers per call: a
`d_zero` all-zeros operand (the carrier, back at the device level) and a
`d_out` that is `alloc_zeros` yet fully overwritten by the kernel. At seq=40960
the expanded K/V is 640 MiB, so each `repeat_kv` call burned 2×640 MiB of alloc
plus two memsets, 12.8 GiB of it (`d_zero`) carrying no information. 4-angle
/simplify review (reuse+simplification+efficiency+altitude) all converged here.

## What changed

New `broadcast_copy_f32` kernel: `out[i] = src[broadcast_offset(i)]`, same
right-aligned stride convention as `add_broadcast_f32`'s `b`, no `a` operand.
`cuda_broadcast_expand_device` now allocs the output **uninitialized**
(`stream.alloc`, not `alloc_zeros`) and launches the copy — one buffer, no
memset, no carrier. Host `broadcast_expand_forward` likewise drops its
`vec![0.0; …]` for a direct `broadcast_offset` copy.

- `crates/autograd/src/backend_cuda/kernels/add_broadcast.cu`: `broadcast_copy_f32`.
- `crates/autograd/src/backend_cuda.rs`: `cuda_broadcast_expand_device` uninit + copy kernel.
- `crates/autograd/src/backend.rs`: host forward direct-copy.

## Verified (H20, GPU 5, binary `52b0b460`)

`--synthetic-writeback-seq 40960` masked-CE writeback: `RUN_EXIT=0`, `DONE
loss=8.685793` — bit-identical to the pre-change baseline (same loss the
`add_broadcast` carrier path produced). grad_check `broadcast_expand` passes on CPU.

## Rule

Removing a tape-visible buffer doesn't remove the allocation if the backend
re-materializes it as scratch. `out = 0 + x` reusing an add kernel keeps the
zeros carrier at the device level plus a dead output memset — a dedicated copy
kernel with uninit output is the actual win. Check the backend impl, not just
the op signature, when a refactor claims a memory reduction.
