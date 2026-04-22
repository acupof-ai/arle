# Bench Stub — Qwen3.6 DFlash Capture Helper Unification

## Context

This tranche was a deletion-style refactor on the Metal Qwen3.5/Qwen3.6 DFlash
path:

- removed duplicated `qwen35_set_capture_layers(...); ...; reset` sequences from
  both the single-request and scheduler prefill paths
- centralized C++ hidden-capture gating in
  `infer/src/backend/metal/qwen35.rs::with_qwen35_capture_layers`
- updated `docs/resources/metal-dflash.md` so the runtime map points at the
  shared helper rather than two handwritten call patterns

## What Worked

- `cargo fmt --all` passed after the refactor.
- `cargo test -p infer --release --no-default-features --features metal,no-cuda --bin metal_bench tests::baseline_compare_default_draft_qwen36_a3b -- --exact` passed.
- `cargo run -p infer --release --no-default-features --features metal,no-cuda --bin metal_request -- --model /path/to/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46 --prompt hi --raw-prompt --max-new-tokens 1 --dflash-draft-model z-lab/Qwen3.6-35B-A3B-DFlash` completed locally with `exit 0`.
- The refactor removed one borrow/lifetime bug introduced during the first
  extraction pass by cloning the scheduler path's `target_layer_ids` before
  entering the capture helper closure.

## Rule

Status: `pending-remote`

Local regression checks passed for the Qwen3.6 DFlash lane after the helper
unification. A canonical remote guidellm run is still required because this is
runtime-facing Metal code under `infer/src/backend/metal/*`.
