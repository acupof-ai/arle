# DSpark draft parity failure on Metal — verify path drifts over long generations, 2026-09-09

> Status: Open. The gate caught the bug; the runtime fix is a separate work item.

## Context

`scripts/spec_parity.py` (the speculative-decoding parity gate) run on the
default Metal pairing — target `models/Qwen3.5-0.8B-MLX-4bit`, draft
`r3lax/Qwen3.5-0.8B-DSpark` — fails zero-mismatch: 7 of 8 prompts diverge,
first divergence at token 13-50, never at token 0. The negative control
(baseline at temperature 0.3) goes red as designed, so the gate machinery is
working; this is a real correctness failure in the draft arm, not a gate bug.

Facts established:

- The baseline arm is deterministic (two identical 96-token greedy runs).
- The draft arm is deterministic (two identical 96-token greedy runs).
- Token 0 always agrees. The drift accumulates over decode steps.
- The divergence is identical in pattern with the Markov head enabled and
  with `DSPARK_NO_MARKOV=1` (7/8 mismatches, tokens 13-50, both). The draft
  token source is irrelevant.

The last point is structural: in `qwen35_compiled_verify_block_summary`
(`crates/mlx-sys/src/mlx_qwen35_model.cpp:2079`) every emitted token is a
target argmax — matched drafts equal the target's argmax by the acceptance
test, and `next_token = sampled[matched]` is the target's argmax at the first
rejected position. The draft head only affects which tokens are *proposed*,
never which are *emitted*. So the divergence cannot come from the draft head;
it comes from the target's verify-forward logits differing from the baseline
decode logits for the same input prefix.

## Root cause

Cause not fully confirmed. The leading candidate is the GDR recurrent state
in the hybrid target.

Qwen3.5-0.8B is a hybrid model: 6 full-attention layers and 18 GDR (linear
attention) layers. The full-attention KV cache is exact — the rollback
(`rollback_kv_to_accepted`, `crates/infer-metal/src/dflash.rs:1536`) trims the
KV arrays to the accepted prefix, and K/V values are the same whether
computed in a batched verify forward or a per-token decode forward. The GDR
state is different: it is a compressed recurrent state updated in-place
during the forward. The verify path processes `block_size` tokens in one
batched call; the decode path processes one token per call. If the batched
GDR recurrence uses a different reduction order or kernel than the per-token
recurrence, the recurrent state differs slightly, and because it is recurrent
the difference compounds over steps — matching the observed drift starting
at token 13-50 rather than token 0.

The GDR rollback (`rollback_gdr_to_accepted`, `dflash.rs:1574`) restores the
pre-verify snapshot and tape-replays the accepted prefix onto it. If the tape
replay is exact, the post-rollback state equals the verify forward's state at
the accepted position — which still reflects the verify forward's batched
numerics, not the decode path's. So rollback correctness does not save parity
when the two forward paths disagree.

Not yet ruled out: a full-attention control. No full-attention model with a
DSpark draft is available locally, so the GDR-specific hypothesis is not yet
isolated from a shared KV/verify-kernel cause. The drift pattern (accumulates
over steps, absent at token 0) is consistent with recurrent-state drift but
not proof of it.

## Fix

Not fixed in this lane. The gate is the deliverable; it caught a real bug.
The runtime fix is one of:

1. Make the batched verify GDR recurrence numerically identical to the
   per-token decode recurrence (same kernel, same reduction order).
2. Recompute the GDR state from the accepted prefix after rollback using the
   decode-path recurrence. This re-runs the target over the accepted prefix
   for GDR layers only, losing the spec-decode speedup for those layers but
   restoring parity.
3. Gate the Metal DSpark path behind a parity-tested pairing list and refuse
   pairings that have not passed `spec_parity.py`.

Option 3 is the immediate mitigation; options 1-2 are the real fix.

## Rule

A draft pairing that has never run a parity gate is an unvalidated pairing,
however much code surrounds it. The Metal DSpark path shipped with a Markov
head that had never produced a verified token on Metal, and the first parity
gate run found it was not correctness-preserving. The gate is now the
admission test: `scripts/spec_parity.py` must pass zero-mismatch before a
Metal draft pairing is used in a benchmark or a default.
