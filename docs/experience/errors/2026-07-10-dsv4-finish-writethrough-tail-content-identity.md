# DSv4 finish-write-through: the reuse tail has no radix content identity

## Context

Finish-write-through decode reuse (`--dsv4-decode-reuse`) pod-verified: the
mechanism WORKS (multi-turn turn-2 match 640→704 tokens, +1 page to the exact
finished length, deterministic ×5). But the ON path crashes the whole TP serve
deterministically under mixed-length reuse:

```
DSv4 decode slot 0 layer 0 pool seq_len 494 != append_pos 485
```

A shorter request (prompt 485) matched a prior finished turn's write-through
entry (finished at 494); restore set the pool `seq_len = 494` (the stored
finish length) while the request's matched prefix is `append_pos = 485`.

## Root Cause (a DESIGN miss, not an impl bug)

The radix guarantees content identity ONLY up to `matched_len` (its page-granular
match). The write-through entry stores a sub-page **tail** `[matched_len,
finish_len)` that is NOT part of any radix key — it is the *previous* turn's
continuation tokens. Two different requests can share the `[0, matched_len)`
prefix yet diverge in the tail. The restore blindly applied the stored tail's
length/content:
- **crash**: `seq_len = finish_len (494) > append_pos (485)` because the new
  request's prompt is shorter than the stored finish length;
- **latent worse-than-crash**: even when lengths fit, restoring the stored tail
  KV injects the PRIOR turn's tokens into a request that doesn't have them.

The plan's per-buffer disposition table proved every DEVICE buffer's range but
never asked "is the tail CONTENT the same tokens as this request's prompt?" —
the one invariant the radix does not cover.

## Fix

Guard the tail on content identity (store it, verify it), + two smaller wires:
1. **Tail token ids**: the pool entry stores the tail token ids `[matched_len,
   finish_len)`.
2. **Restore + admission guard**: reuse the tail ONLY if `prompt_len >=
   finish_len` AND `prompt[matched_len..finish_len] == entry.tail_tokens`. Full
   match → restore to `finish_len`, return `extra = tail_len`. Otherwise the
   frontier page is not committed: `committed` drops to the last verified
   boundary at/below `matched_len` (the last prompt-side boundary — NOT
   necessarily `matched_len` itself, since the intra-decode-region pages carry
   `boundary=false`), `extra = 0`, and the slot restores to that page-aligned
   length ⇒ `seq_len <= matched_len = append_pos`. Never over-restores → the
   crash (`seq_len > append_pos`) is unreachable.
3. **CLI flag** `--dsv4-decode-reuse true` engages it (CLI-only per
   §runtime-config-CLI-flags; the env var is intentionally NOT wired).

## Rule

**Any reuse of content BEYOND the radix match needs its own token verification
against the requesting prompt.** The radix key proves identity only to
`matched_len`; a stored tail, sidecar, or carry that extends past it is
unverified content — compare tokens before applying it, or a prefix collision
injects a different request's KV. (Restore must also clamp its restored length
to the engine's, never to a value stored from a different request.)
