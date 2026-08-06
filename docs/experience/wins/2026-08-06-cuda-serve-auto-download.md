# CUDA serve auto-downloads HF models — CUDA, 2026-08-06

> Status: Shipped

## Goal

Make `arle serve --model-path Qwen/Qwen3.5-4B` work on CUDA without a
preliminary `arle model download`, matching the Metal behavior.

## Problem

Metal's `metal_serve_handle` called `infer_metal::resolve_model_path`, which
downloads the checkpoint from HuggingFace when absent. CUDA's
`cuda_serve_handle` passed the raw `model_path` straight through, so a HF id
failed with "config.json not found" instead of downloading.

## Fix

`cuda_serve_handle` now calls `infer_util::hf_hub::resolve_model_path` (the
same backend-neutral resolver `arle model download` uses) before loading the
tokenizer / engine. Added `infer-util` to `infer-api`'s dependencies.

## Result

- `arle serve --backend cuda --model-path Qwen/Qwen3.5-4B` downloads the
  checkpoint on first use.
- Local paths still short-circuit (no network).
- Metal path unchanged.
