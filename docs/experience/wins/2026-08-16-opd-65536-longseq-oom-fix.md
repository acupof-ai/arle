# OPD Long-Sequence OOM Fix (65536 + 131072)

**Date:** 2026-08-16
**Scope:** `crates/train/src/opd/`, `crates/train/src/qwen35/`, `crates/train/src/teacher_infer.rs`, `crates/cli/src/train_cli/`
**Commits:** `cd9784f6c`, `a52761816`, `e96ee6a43`

## Context

OPD training at 65536 prompt tokens failed with CUDA OOM despite 92 GB free VRAM.
The error surfaced as `cuda alloc_zeros failed (la v)` in the linear-attention
forward, with the actual allocation being 134 M elements (~256 MB) — far from
the 256 KB expected for a 32-token window.

## Root Cause

Two compounding issues:

1. **Growing-prefix re-computation**: The windowed KL path called
   `teacher.forward_logits_window_device(&rollout[..window.end], ...)` for each
   window. This re-ran the full teacher forward on a growing prefix: window 1
   processed 32 tokens, window 2 processed 64, ..., window N processed the
   entire 65 543-token sequence. The last window's linear-attention `v`
   allocation alone was 256 MB; across 18 linear-attention layers the
   per-layer scratch totalled ~60 GB.

2. **TensorStore accumulation**: Even with the tape disabled, the
   `TensorStore` held every intermediate tensor from every layer until
   `cleanup_after_backward` — which runs after the entire backward loop, not
   between layers. At 65 K tokens the store's peak exceeded available VRAM.

## Fix

- **One full-seq teacher forward**: `forward_hidden_device` runs the teacher
  forward once on the full sequence, returning hidden states. Per-window
  logits are computed from the cached hidden via
  `logits_from_hidden_window_device` (slice + final norm + lm_head).

- **Per-layer scratch pruning**: `forward_hidden_freeing_intermediates` calls
  `store.retain_ids()` after each layer, keeping only model params, cos/sin,
  the current hidden, and a caller-provided retain set (student params, LoRA,
  optimizer state). This bounds the store's peak to one layer's scratch.

- **`device_synchronize` before trim**: `cuCtxSynchronize` drains all streams
  (not just the train backend's) before `trim_memory_pool`, ensuring the
  infer engine's pending frees are visible to the pool.

- **`max_seq_len` 8/7 compensation**: `scheduler_config()` reserves 1/8 of
  `per_req_cap` for generation, clamping `max_prompt_tokens` to 7/8. The OPD
  driver now sets `engine_seq = (seq * 8/7 + 15) / 16 * 16` so the engine
  accepts the full prompt.

- **Diagnostic error messages**: Linear-attention allocation failures now
  include the actual `DriverError`, allocation length, and free/total VRAM.

## Result

65536:
```
step 1/1 loss 4.334862 rollout_len 65544
```
Peak VRAM ~26 GB. ~21 min on H20 GPU 5.

131072 (after O(n) student forward refactor, `e96ee6a43`):
```
step 1/1 loss 0.151320 rollout_len 131080
```
Peak VRAM ~44 GB. ~90 min on H20 GPU 5. Gradient checkpointing engaged
(`[ckpt-gate] engage=true seq=131080`). The O(n²) growing-prefix approach
was killed at 48 min without completing; the O(n) single-forward approach
completes.

## Rule

When a windowed/distillation path re-runs a forward per window, the forward
must process only the window's tokens — not a growing prefix. Cache the
full-seq hidden once and derive per-window logits from it. The student
forward follows the same pattern: one full-seq forward with gradient
checkpointing, then per-window logits from the cached hidden, with a single
backward for the accumulated loss.
