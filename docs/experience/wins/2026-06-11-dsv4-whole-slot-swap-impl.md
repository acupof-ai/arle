# DSv4 Whole-Slot Swap — Executor Serializer (#84/#85 Route B) — pending-remote

## Goal

Implement the DSv4 half of the whole-slot KV swap: serialize a slot's
COMPLETE device state (MLA FP8 band + compressor/indexer streams + SW ring +
spec chain + scalars) into a host image on `demote_slot`, restore it exactly
on `promote_slot`, so a preempted DSv4 request resumes decode at the demoted
position instead of recomputing. Engine half landed in `f8fb0421`; design:
[`docs/plans/2026-06-11-dsv4-whole-slot-kv-swap.md`](../../plans/2026-06-11-dsv4-whole-slot-kv-swap.md).

## Hypothesis

Full-allocation D2H copies (extent-proof by construction) over the §0.1
buffer enumeration — with the FP8 pool band via the existing
`flashmla_slot_range` accessor — capture everything a resumed decode reads;
scratch verdicts hold because every scratch buffer is written from step
inputs before its sole same-call read.

## Params

- `Dsv4LayerImage` (+ compressor/FlashMLA/DSA sub-images) in `attention.rs`;
  `Dsv4SlotImage` + `Dsv4SlotState::{swap_out_image,swap_in_image}` in
  `dsv4.rs` (slot-agnostic: pool bands re-resolved from the TARGET slot, so
  promote into a different slot index works); store + hooks + dispatch in
  `executor.rs`/`lib.rs`. 664 insertions, implemented by a delegated
  general-purpose agent against the line-level spec, reviewed hunk-by-hunk.
- Snapshot set: SW ring (full), compressor/indexer pending/prev_overlap +
  `compressed.data` (full — the rollback snapshot skips data only because
  rollback shrinks a live slot; a swapped slot gets REUSED), FP8 pool band
  (full band, written-extent perf TODO), DSA packed_rows/rotated_keys/key
  band, FlashMLA host scalars, `seq_len`, MTP `spec_pending`/`spec_hidden`.
- Scratch (no snapshot, write-before-read verified per buffer in comments):
  FlashMLA index slices + split accumulators, fused_wqkv/prefill_linear
  staging, MoE/deepep scratch, `start_pos_device`, `spec_rollback`;
  `decode_graph` re-arms one eager warm pass on swap-in (request-boundary
  discipline, same as `reset`).
- Gate: `kv_slot_tier_enabled` only at `world_size == 1`; multi-rank logs
  once and reports disabled (lockstep SwapOut/SwapIn relay envelopes are the
  tracked follow-up). Store cap: `2 × num_slots` images (≈2× device arena in
  host RAM; churn beyond that = thrash, recompute is the better fallback).
- Deviation from the plan doc: MTP spec chain is SNAPSHOTTED, not reset —
  `forward_decode_tokens` hard-errors on a missing `spec.pending` under
  `--spec-type mtp`, so a reset would break every resumed decode.

## Env

Local Apple Silicon; CUDA code typechecked
(`CUDARC_CUDA_VERSION=12060`, `cuda,no-cuda`), no GPU execution.

## Results

- `cargo check -p infer-api … cuda,no-cuda --lib` — clean.
- `cargo test -p infer-cuda` — 59 passed; `cargo test -p infer-core` — 50
  passed (engine swap semantics covered by the f8fb0421 mock tests).
- clippy: zero new warnings (two pre-existing `manual_checked_ops` in
  kv_tier.rs fixed alongside).

**pending-remote** — the pod gate owes (single 8×H20 window, shared with
#82/#83):
1. Forced swap mid-generation, needle ladder ×3 + same-config-twice, with
   FlashMLA decode on AND `--spec-type mtp` on (the spec-chain snapshot and
   the split-accumulator scratch verdicts are source-verified only).
2. Promote-into-different-slot case if the gate can force it.
3. Wall-clock: swap-restore vs recompute TTFT at the SLO shape.

## Problems

DSv4 preemption is currently unreachable in practice: `retract_decode_to_fit`
fires on host-pool page pressure, and the DSv4 admission pool is sized to
never run out (`cuda_admission_total_pages` covers every slot at max len).
The swap machinery is therefore complete-but-dormant until an
oversubscription/priority-preemption policy lands — that policy is the real
unlock for "KV stays hot" on DSv4 and belongs to the pod-phase scope of #84.

## Learnings

Full-allocation copies turn the hardest extent questions (ring cursors,
packed-row counts) into non-questions — the only inferred extent in the
whole serializer is zero. The 2026-06-06 EAGLE enumeration did the
discovery; the swap image is that enumeration plus the one thing rollback
never needed (compressed.data), which is exactly the kind of gap §0.1
enumeration exists to expose.
