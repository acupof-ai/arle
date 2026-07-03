# #138 decode-side fixed: zero the reused TP FlashMLA output scratch

## Context
DSv4-Flash eager decode NaN'd from absolute context length 129 (sliding_window
128 + 1) — argmax(NaN)=token 0 → empty output. compute-sanitizer initcheck
pinned 100 uninitialized global reads to ONE kernel, `dsv4_tp_out_slice_kernel`
reading `full_out`.

## What Worked
The decode TP output scratch (`scratch.tp_full_out`, `alloc_zeros` ONCE then
reused every decode step) is left partly unwritten when the SM90 sparse decode
kernel's top-k selection is empty for a (query,head); the rank-slice copied the
PRIOR step's stale value (a propagated NaN) into the residual. Fix (`3f2a9b32`):
`memset_zeros` the scratch before each FlashMLA decode write — an unwritten
slice reads 0 (the correct empty-attention output), not stale.

Pod verify (TP=4/EP=4, GPUs 0-3, --spec-type none, greedy): count task, prompt
21 tok → completion 156 tok `6,7,8,…,59` monotonic and CORRECT through absolute
ctx 177 (48 past the 129 wall), finish_reason=stop. Pre-fix: token-0/empty from
129. The count being exactly right proves a genuine fix, not NaN→wrong-finite.
Control (<128 ctx) coherent, no regression. 0 NaN/inf warnings.

## Still open
Prefill-crossing (a 165-token prompt whose prefill crosses 128) is STILL empty —
a DISTINCT mechanism. The prefill `full_out` is a fresh per-call `alloc_zeros`
(no reuse, no stale), so its NaN is a WRITTEN value (FlashMLA sparse-prefill
producing NaN for a boundary position, likely an empty-selection 0/0), not the
stale-read the decode memset fixed. Tracked in #138; needs a -lineinfo sanitizer
run (OOMs TP=4, needs the full box) or a code-level empty-selection audit.

## Rule
A reused device scratch that a kernel writes only partially must be re-zeroed
before each use, not just at alloc — the unwritten slices otherwise carry the
prior iteration's value. initcheck ("uninitialized read") is the exact tool;
run it before guessing at a race.
