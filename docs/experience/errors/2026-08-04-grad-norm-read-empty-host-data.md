# The grad-norm reader routed on backend identity, not on where the grad lives — and reported 0.0

**Date:** 2026-08-04 · **Commits:** e3ff7c368 (gate guard) + this fix · **Found by:** the CP grad-parity gate's f32 anchor

## Context

The new CP grad-parity gate compares post-backward global grad norms three ways
(CPU f32 / single-card CUDA / CP). On the pod the f32 arm returned
**exactly 0.000000000** with `layer_types=[FullAttention, FullAttention]`, but
73.51 with `[LinearAttention, FullAttention]`. Loss was finite and correct in
both. LoRA targets were `AllLinear`, so an exactly-zero global norm should have
been impossible even if attention were detached — the residual stream still
feeds the MLP adapters.

## Root cause

`grad_clip.rs` decided how to read a gradient from **which backend is
installed** instead of **where that gradient actually lives**:

```rust
if store.backend().device() != Device::Cpu && grad.dirty != Dirty::Host && ...
```

On the CPU backend the first conjunct is false, so it always fell through to
summing `grad.data`. But a CPU-backend gradient can be device-resident:
`ChunkSum::add` allocates its accumulator with `alloc_device_tensor`
(`ops/chunk_accum.rs:112`), which leaves `data` **empty** and the value in the
handle. Every grad produced by `seq_chunked_recompute_backward` is like this —
that covers the MLP (`qwen35.rs:626`) and full attention (`qwen35.rs:1792`).
Linear attention runs outside the chunked wrapper, so its grads stay host-side.

Hence the arithmetic: full-attn-only → nothing host-resident → 0.0. Hybrid →
only the GDN layer counted → 73.51, and the true norm was 73.57. The 0.09% gap
to the CUDA value looked like agreement purely because
`in_proj_qkv.lora_b` carries 73.49 of the 73.57 on its own.

`clip_grad_norm` had the identical guard, so **gradient clipping was a silent
no-op on the CPU backend** for those grads (it scaled an empty vec). It never
bit because the gate passes `max_norm=0.0`.

## Fix

Delete the backend check at both sites and route on `dirty`/`device_handle`
alone. `Backend::sum_squares` and `mul_scalar` both have backend-agnostic
default impls (readback → compute → upload), so the CPU backend needs no
special case. Regression test: a CPU-backend param with a device-resident grad
must count toward the norm and must be scaled by clipping — verified to fail
with `got 0` when the guard is reinstated.

Blast radius, checked rather than assumed: CUDA/Metal were always correct (the
guard's first conjunct is true there), and `AdamW::step` peeks at grad
residency itself (`optim.rs:249-257`), so **the CPU path was still learning** —
only the observable was wrong. The damage was a dead gate anchor: on the real
27B the f32 reference would have been missing the 16 full-attn layers and all
64 MLPs.

## Rule

Route on the property you actually depend on. "Which backend am I" is a proxy
for "is this tensor host-resident"; the proxy was false the moment any op
allocated device-side on CPU. When a reader and a writer disagree about where
data lives, the reader returns a plausible number, not an error — so an
instrument that reads 0.0 should be suspected before the thing it measures.
