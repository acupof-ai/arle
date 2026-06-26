# agent-OPD masked-CE writeback: forward-once fix removes the logits-tile OOM but exposes a forward-activation wall

## Context

The agent-OPD masked-CE writeback (`crates/train/src/opd.rs:masked_writeback_ce_step`)
persistently OOM'd (`cuda alloc_zeros failed`) on the writeback step after a
passing rollout. The old design looped `sequence_windows` and per window called
`forward_logits_window([0..window.end])`, re-forwarding the growing prefix
(O(N²)) and materializing a dense `[1, window, vocab=248320]` f32 logits tile
(~2 GB at window=2048). At a 63.6 GB-resident shared-base engine
(`--share-frozen-base`, 256 borrowed FP8 projections) on a 97.8 GB H20, the tile
alloc failed on the fragmented caching pool.

Fix shipped (correct, verified): forward the trajectory ONCE
(`forward_hidden_states`, gradient-checkpointed) → hidden, then
`fused_linear_ce_loss_indexed(hidden, lm_head, masked_positions, targets,
chunk_rows)` — a new chunked fused cross-entropy in
`crates/autograd/src/ops/fused_linear_distill.rs` mirroring
`fused_linear_distill_loss_sparse` but with hard-target CE instead of teacher-KL,
taking the masked predicting positions as an explicit non-contiguous index list.
It chunks over positions, computing `logits_chunk = hidden_chunk @ lm_headᵀ` + CE
+ gradient per chunk and freeing each, so peak transient is `chunk * vocab * 4`
(~0.25 GB at chunk=512), never the full `[seq, vocab]` tile. `d_weight` only
flows if the lm_head is trainable (frozen under `--share-frozen-base`).
Equivalence unit test (`crates/autograd/tests/test_fused_linear_ce.rs`) gates
fused loss + `d_hidden` vs a materialize-then-`cross_entropy_loss` reference
within 1e-3 on dense and sparse positions — passes.

## Root Cause

The fix is correct but **not sufficient**: it traded the logits-tile OOM for a
forward-activation OOM that was always latent. Pod run (Qwen3.6-27B-FP8, GPU 5,
H20 97.8 GB, `--share-frozen-base`, grad-checkpointing on, seq_len=15858,
total_targets=2303, chunk_rows=512):

```
pre-writeback  GPU5 = 63659 MiB   (shared-base engine resident)
[masked-writeback] seq_len=15858 total_targets=2303 chunk_rows=512   ← new path live
post-writeback GPU5 = 97487 MiB   (full 97871 MiB cap)
error: ... "OPD masked writeback student hidden ... cuda alloc_zeros failed"
```

The failure stage **moved off the logits tile** (proof the fix works) onto
`student hidden` — the single `forward_hidden_states` over 15858 tokens. Memory
climbed 63.6 → 97.5 GB during the forward, before the chunked CE ran at all.

Why it was latent, not new: `forward_logits_window` forwards
`input_ids[..window.end]`. The OLD loop's EARLY windows forwarded a short prefix
(window `[512,1024)` → 1024 tokens) and OOM'd on that window's *logits tile*
first — but its LAST window `[15360,15858)` would have forwarded the full ~15858
tokens, hitting the identical forward wall. The logits-tile OOM simply fired
earlier and masked the forward wall. A ~15858-token gradient-checkpointed forward
+ a 63.6 GB-resident engine needs > `97.8 − 63.6 = 34.2 GB` of forward
activations/attention scratch and does not fit. Shrinking `chunk_rows` cannot
help — the chunked CE never executes; the forward OOMs first.

## Fix

Landed (kept, correct): forward-once + chunked fused-CE removes the O(N²)
re-forward and the `[seq, vocab]` logits-tile OOM. NOT committed to the branch
because the end-to-end loop does not close at this trajectory length / resident
budget — the commit gate required loop-closure.

Next wall (separate task, not yet licensed): bound the
`forward_hidden_states` peak for ~16K-token trajectories on top of a 63.6 GB
engine. Candidate levers, each needs a measured single-variable A/B before
landing — do NOT stack:
1. `--lora-layer-start` (suffix-detach): skip backward/checkpoints for the
   frozen prefix layers — the CLI comment already names this as a co-requisite
   of grad-checkpointing for the 27B; the verify run did NOT pass it.
2. Chunk the forward itself over sequence (recompute hidden per CE chunk from a
   layer-boundary checkpoint) instead of one full-length forward — trades
   compute for a bounded forward transient.
3. Attention-scratch audit: confirm whether the 34 GB is layer-boundary
   checkpoints (should be offloaded by `set_offload_checkpoints(true)`) or
   un-fused attention scores at seq=15858 — nsys/`ARLE_VRAM_TRACE` the forward
   to attribute before picking a lever.
4. Lower resident floor: the 63.6 GB shared-base engine is the other half of the
   budget; releasing more inference scratch pre-writeback widens headroom.

## Rule

A memory fix scoped to one transient (the logits tile) must enumerate **every**
peak transient on the path before claiming loop-closure — the forward's own
activation peak is a co-equal term. Decode the failure stage from the actual
error string (`student hidden` ≠ logits): the OOM **moving** to a new stage is
evidence the targeted fix worked AND that a second wall was masked behind it.
Verify loop-closure on the **production trajectory length** (here ~16K tokens),
not a smoke shape — a windowed path's early-window OOM hides the full-length
forward wall its last window would hit.
