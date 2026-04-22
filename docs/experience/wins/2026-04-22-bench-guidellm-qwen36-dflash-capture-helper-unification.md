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
- The refactor removed one borrow/lifetime bug introduced during the first
  extraction pass by cloning the scheduler path's `target_layer_ids` before
  entering the capture helper closure.

## Rule

Status: `pending-local-blocked`

Runtime-facing Metal verification is currently blocked by unrelated compile
errors already present in `infer/src/kv_tier.rs` and
`infer/src/kv_tier/coordinator.rs` (`SpillRequest` / `SpillTicket` /
`CoordinatorCommand::Spill` / `handle_spill` / `DiskBlockLocation`). Re-run the
canonical Qwen3.6 DFlash unit test and local smoke after that broken kv-tier
line is repaired in-tree.
