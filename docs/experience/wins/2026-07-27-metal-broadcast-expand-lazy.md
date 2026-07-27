# Metal broadcast_expand stays lazy — repeat_kv backward, 2026-07-27

> Status: correctness restored + Metal build/test green; wall-clock A/B pending
> a local Metal capture (direction is structurally certain — a readback removed).

## Goal

`repeat_kv`'s GQA-expand must not force a device→host→device round-trip on Metal.

## Context

`a61d44579` (`repeat_kv via broadcast_expand`) replaced the old
`add_broadcast(zeros, x)` lazy path with `broadcast_expand`. But
`broadcast_expand` had no MetalBackend override, so it fell to the default impl:
`readback(src) → cpu_add_broadcast_forward → upload`. `repeat_kv` runs in the
non-GQA recompute backward (attention.rs:41), so every affected attention layer
took a full MLX sync + host transfer + re-upload per step — exactly the flush the
lazy MLX graph exists to avoid. Codex review P1.

## What changed

- `backend_metal.rs`: native `broadcast_expand` via `mlx_broadcast_to` — MLX
  broadcasts size-1 dims to `target_shape` in-graph, no eval boundary, mirroring
  the existing lazy `add_broadcast`. Import added; `_src_shape` unused (MLX reads
  shapes off the array).

## Results

```text
autograd metal build: Finished (clean)
m1_layout::broadcast_expand_grad_matches_central_difference: passed (metal)
clippy + fmt: clean
```

The forward/backward values are unchanged — MLX `broadcast_to` is semantically
the host expand, and the CPU central-difference gradient gate already pins it.
The fix is purely the lazy-vs-readback path.

## Problems

Wall-clock before/after needs a local Metal training capture (Xcode / MLX
instruments) on `repeat_kv`'s backward — not run here (no model loaded this
session). Direction is not in question: the change deletes a per-layer,
per-step sync + 2 transfers, restoring the pre-`a61d44579` on-device path.

## Learnings

**A new op needs a device override on every backend it runs on, or it silently
takes the readback default.** `broadcast_expand` had a CUDA override and a host
fallback; Metal fell through to the fallback, turning a lazy graph op into a
per-step sync. When adding an op used on the hot path, check each backend's
`impl` — the default trait method is a correctness floor, not a performance one.
