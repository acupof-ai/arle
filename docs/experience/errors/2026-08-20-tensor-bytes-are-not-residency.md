# A lever sized by tensor bytes bought 0 MiB — tensor bytes are not residency — 2026-08-20

Reverted: `bdf19e8a3` + `bf7c8f032` (`perf(train): rebuild the gated-delta conv
pair in the backward`), reverted by `edcc653b7` + `9466ffdff`.

## Context

Chasing global sequence 262,144 on 2 GPUs. The bf16 audit
(`2026-08-20-cp2-ceiling-114688-to-131072.md`) named two remaining backward-side
levers and sized both: fusing the three `slice_backward` scatters (8,960 MiB at
local 131,072) and recomputing `preact` in the conv backward (4,480 MiB).

The first shipped and measured **−6,210 MiB** at local 65,536, matched arms.

The second was implemented as the stronger version: `preact` (f32) and
`qkv_conv` (bf16) are both outputs of one depthwise causal conv over `qkv`, so
one recompute in the backward drops both from the tape — 6,720 MiB of tensors at
local 131,072. The host backward already recomputes the whole forward and never
read them, so only the CUDA backward held them.

## Phenomenon

| local | `actual` before | `actual` after |
|---|---:|---:|
| 65,536 | 69,665 MiB | 69,665 MiB |
| 81,920 | 77,026 MiB | 77,026 MiB |

Bit-identical at both rungs. Global 163,840 still fails on the same
`zeros [1, 81920, 5120]`. Backward wall-clock 202.95/203.01 s → 203.51 s.

**0 MiB, for one extra kernel launch per layer per backward.**

## Root cause

The arm was not dead. The taped fields were deleted from `SavedContext`
outright, so no path could have read a stale `preact`; had the recompute not
run, the loss would have been wrong. It was `3.036179` — bit-identical to
baseline, so the recompute ran and was exact.

The tensors simply were not live at the moment the peak is set. Under gradient
checkpointing the whole layer forward runs on a disabled tape, and `preact` /
`qkv_conv` are allocated only inside the backward's replay — they already lived
for one short stretch of one sub-group. Moving that stretch from "the sub-group"
to "one kernel sequence" does not move a high-water mark set elsewhere.

The audit sized the lever by **how large the tensors are**. What sets a peak is
**how long they are resident, and whether that overlaps the peak**. Those are
different quantities, and only the second one matters.

The slice-scatter lever worked for the opposite reason: it did not shrink a
tensor, it removed *allocations that happen at the peak moment* — two
full-input zero buffers that existed only to be summed.

## Fix

Reverted. The change is numerically correct and slightly simpler as a tape, but
it buys nothing measured and costs a launch, which is speculative debt.

## Rule

Size a memory lever by residency across the peak, not by tensor bytes. Before
implementing one, name the moment the peak is set and show the tensor is alive
then. Under gradient checkpointing most forward intermediates are already dead
at that moment, and an audit that lists tensor sizes will keep proposing them.
