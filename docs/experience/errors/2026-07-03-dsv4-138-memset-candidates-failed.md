# #138: two memset candidates FAILED — it's a FlashMLA sparse write-NaN, not a stale read

## Context
DSv4-Flash NaNs from context length 129 (sliding_window 128 + 1), on BOTH the
eager decode lane (generation crossing 128) and prefill (prompt already >128).
compute-sanitizer initcheck flagged uninitialized reads in
`dsv4_tp_out_slice_kernel` (the TP FlashMLA output slice).

## What failed
Two candidates, both reverted (`0c6ddfc9`, `9165dc93`):
1. `3f2a9b32` — memset the REUSED decode output scratch before write. An early
   verify (digit-count `text`) read as "decode fixed", but a detok-safe re-check
   (the non-stream `text` field mangles space-less multi-digit tokens —
   `131132` renders as `131 99 100`) showed decode STILL plateaus at ~pos 127
   then emits an empty tail. The "fix" was a detok artifact.
2. `a610060f` — explicit on-stream zero of the fresh prefill full_out. No effect;
   prefill-crossing stayed empty.

## Root cause (revised)
Zeroing the output buffer cannot help because the sparse-attention kernel
OVERWRITES those slices with a NaN/garbage value for `(query_pos >= 128, head)`
whose top-k selection is empty — a WRITTEN NaN, not an unwritten/stale read. The
manifestation is content/position-dependent (empty for most, doubled-garbage for
some, occasionally correct within one process; cross-process non-deterministic at
T=0) — the signature of an empty-selection slice getting a bad write.

## Next
`-lineinfo` full-box compute-sanitizer run (initcheck OOMs TP=4 → needs TP=8 on
all 8 GPUs) to localize the exact `(pos>=128, head)` write in the FlashMLA
sparse-prefill kernel AND the DSv4 decode attention path. Fix the empty-selection
normalization (a sink / a guarded 0-out) at the write, not the read.

## Rule
An "uninitialized read" from initcheck can be a RED HERRING for a downstream
WRITE-NaN: the slice reads uninit only because the producer wrote garbage there.
Verify the fix with a detok-SAFE signal (empty-vs-nonempty, space-separated
digits, a word cycle) — never the raw `text` field on space-less token streams.
