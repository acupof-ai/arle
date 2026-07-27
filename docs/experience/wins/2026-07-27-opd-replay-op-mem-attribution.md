# Per-op checkpoint-replay memory attribution — CUDA, 2026-07-27

> Status: Shipped (diagnostic, `ARLE_OPD_OP_MEM_CHECKPOINT_FN=<fn>`). Verified on
> H20 GPU4, binary `6da8e866`, seq=40960, rc=0.

## Goal

Decompose the OPD masked-writeback single-layer checkpoint replay by op, to
settle whether MLP intermediates (11.4 GB, the old doc's "biggest movable
block") actually dominate — using `pool_used_current`, not driver-used.

## What changed

`Tape::checkpoint_backward` arms a one-shot scope when `function_id` matches
`ARLE_OPD_OP_MEM_CHECKPOINT_FN`; records `pool_used_current` + live-tensor count
at each inner backward op (pre/post/merge) and at 7 replay-forward stage anchors
in `Qwen35Layer::forward`, buffered and flushed once after the scope. No added
sync — reads the mempool `USED_MEM_CURRENT` attribute only. `checkpoint_fn=N`
maps to layers via a `[checkpoint-map]` line (frozen layers 0–2 shift the id
below the layer number, so layer 63 = fn=60).

## Result — layer 63 (full-attention, historical OOM layer), seq=40960

Single replay + inner backward, `pool_used_current` deltas from the layer floor
(37.9 GiB):

| stage | pool_used | Δ |
|---|---:|---:|
| post_input_norm | 38.7 | +0.8 |
| **post_attention** | **62.1** | **+23.4** ← biggest |
| post_mlp | 75.3 | +11.4 |
| post_replay (full layer materialized) | 76.1 | +38.2 total |
| **inner-backward peak** | **85.9** | +9.8 grad |
| scope_exit | 37.9 | freed |

Peak pool_used 85.9 GiB / reserved 89.4 GiB, device 97.5 GiB → **40960 fits with
~8 GiB headroom, rc=0, loss=8.685793.** forward pool_used stays flat at 34.7 GiB
while driver-used climbs to 75.9 GiB (39 GiB hoard, trimmed to 3.2 GiB in
backward).

## Rule

**Attention forward recompute (23.4 GiB) is the largest single-layer block, not
MLP (11.4 GiB) — the old MLP-first recompute plan targeted the second-biggest
term.** And 40960 has no wall to move: the historical OOM was allocator hoard,
not live bytes. Decompose by `pool_used` stage delta before naming a recompute
target; a plateau in driver-used is hoard, not activation. The real wall is a
`pool_used` length-ladder question (65536→131072), where growth becomes live
bytes. See [[2026-07-26-mempool-hoard-ledger]].
