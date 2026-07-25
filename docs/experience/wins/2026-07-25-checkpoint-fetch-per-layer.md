# checkpoint fetch-per-layer: re-offload after backward replay — CUDA, 2026-07-25

> Status: pending-remote

## Goal

Cut the masked-writeback backward VRAM slope by keeping only one grad-checkpoint
device-resident at a time instead of all N accumulating. Target: 2.72 → ~1.43
MiB/tok (single-card seq wall 25K → 48K), no numerics change.

## Hypothesis

Forward already offloads saved checkpoints to host (`offload_checkpoint_to_host`).
Backward fetches each per-layer on replay (`ensure_checkpoint_device`, tape.rs:796)
but never re-offloads it — so residency accumulates monotonically and all N end up
co-resident (doc `2026-07-25-writeback-vram-theory-vs-measured.md` waste ①). Re-offload
the replayed hidden after `free_new_except` so device holds one checkpoint, host holds
N-1. Symmetric with forward → host peak unchanged from forward's, zero added peak.

## Parameters

```bash
# OPD writeback replay, offload on (default off → this path is a no-op):
#   --checkpoint-offload true, ThinkingCap-27B-FP8, seq sweep 5120..28672
[opd-vram-ledger] base_used_mib post_forward_used_mib post_backward_used_mib
# before/after: compare post_backward peak slope (MiB/tok) at matched seq.
```

- Baseline: pre-change, fetch-per-layer but no re-offload (backward peak = all N resident).
- Treatment: re-offload replayed hidden (tape.rs:824), `offload_checkpoints` on.
- Correctness gate: `scripts/needle_gate.py` ×3 + `cargo test -p autograd checkpoint`.

## Environment

- Host / GPU: H20 sm_90 (OPD writeback lane) + G4 sm_120.
- Model / dtype: ThinkingCap-27B-FP8, LoRA OPD, 64 grad-checkpoint groups.

## Expected result

- VRAM: backward peak slope 2.72 → ~1.43 MiB/tok; the doc's offload A/B (+15.4 GB
  backward regression) flips to a reduction. Host peak = forward host peak (no new host cost).
- needle: byte-identical (re-offload is a residency move, `ensure_device` re-fetches).

## Rule

Backward release of a fetched checkpoint uses `offload_checkpoint_to_host` (readback →
host copy survives), never `drop_device_residency` (no readback — only safe under the
linear ownership of `checkpoint_sequential`, unsafe for the general `checkpoint()` API
where a hidden may be shared/reused). Symmetric offload needs no upper-abstraction change.

## Note

codex review flagged the initial `drop_device_residency` version (P2): a shared/reused
hidden would hit empty host data on the next `ensure_device`. Fixed at root by the
symmetric readback offload above, not by scoping the drop to `checkpoint_sequential`.
