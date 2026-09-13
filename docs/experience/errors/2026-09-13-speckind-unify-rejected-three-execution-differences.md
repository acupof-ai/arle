# One draft head cannot carry both DSpark and MTP: the difference is execution, not naming

Date: 2026-09-13. Investigation closed 2026-09-09 as not mergeable; recorded
here because the conclusion existed only as an agenda verdict.

## Context

`tileRL` loads Qwen NextN and DSpark through one `DraftHead` class plus a
seven-entry parameter-name map (`spec.py _DRAFT_TOP`), 279 lines in total.
Our equivalent is 4,567 lines across seven files. The obvious reading is that
we have duplicated a loader seven ways and a name map would collapse it, and
the task was opened on that reading.

## Root Cause

The reading is wrong, and the line-count comparison is what makes it look
right. The two heads do not execute the same graph:

1. **Norm placement differs.** DSpark normalizes after summing the tap
   projections. MTP normalizes the embedding and the hidden state separately,
   before a single `fc`. These are different forward graphs, not the same
   graph with different weight names.
2. **KV dtype is pinned differently.** DSpark's batched path is fixed to BF16
   KV. The MTP lever's target configuration is FP8 KV. A merged path either
   inherits DSpark's BF16 gate for both heads, or splits the gate per head,
   which is the duplication the merge was supposed to remove.
3. **State shapes have no counterpart.** DSpark keeps a persistent context
   ring and a multi-tap `fc`; MTP has neither. MTP allocates fresh KV per
   block; DSpark has no equivalent.

`tileRL`'s single class carries both norm conventions in its body. That is a
union of two implementations behind one name, and it is smaller than ours
because it is 279 lines of Python calling into a framework, not because it
resolved the three differences above.

## Fix

None applied; the task was closed as not mergeable. If it is reopened, the
thing to attack is (2) — the BF16 pin on DSpark's batched path — because it
is the only one of the three that is a constraint rather than a shape
difference. (1) and (3) would still forbid a merge afterwards.

## Rule

A line-count ratio between two codebases is not evidence that the larger one
is duplicated. Before treating a name map as the missing abstraction, check
whether the two things being unified run the same graph: compare norm order,
dtype pins, and per-step state shapes. If any of the three differ, a shared
class is a union with a dispatch inside it, and it will be larger than the
two implementations once every difference is expressed.
