# REPL/OCR single-slot load — Metal, 2026-08-06

> Status: Shipped

## Goal

Let the interactive REPL and `arle ocr` load Qwen3.5-9B-class models on a
48 GB Apple Silicon box without hitting the Metal resource guard.

## Problem

`LoadedInferenceEngine::load` used `EngineLoadConfig::default()`, which sets
`num_slots = 256`. Qwen3.5's GDR (linear-attention) layers carry per-slot
recurrent state, so the resource guard's `static_state = gdr_bytes_per_slot ×
num_slots` came out to **12.5 GiB** for the 9B model. Combined with weights
(5 GiB) and runtime headroom (4 GiB), the fixed requirement was 21 GiB —
above the ~15 GiB anti-swap budget on a box with other apps open.

`arle serve` already had `--max-running-requests 1` to collapse slots, but
the REPL and OCR paths had no equivalent knob.

## Fix

`load()` now sets `max_running_requests: Some(1)`. The serve path is
untouched — it goes through `load_with_config` with the serve-derived slot
budget.

## Result

- Qwen3.5-9B-MLX-4bit loads in the REPL: `static_state` 12576 MiB → 49 MiB,
  peak RSS 4.8 GiB.
- Agent tools (bash, python) verified working on the 9B model.
- `arle ocr` unchanged in behavior (was already single-slot in practice).

## Command

```bash
arle --model-path mlx-community/Qwen3.5-9B-MLX-4bit run --prompt "use python to compute 17*23"
```
