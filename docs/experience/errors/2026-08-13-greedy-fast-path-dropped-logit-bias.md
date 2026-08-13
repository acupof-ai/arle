# The greedy fast path re-derived "no sampling needed" and got it wrong — 2026-08-13

> Status: Fixed. Two divergences between the device fast path and the host
> sampler, both licensed by the same unsound equivalence claim.

## Context

`sample_cuda_token` skips the host sampler when the request is greedy:

```rust
if params.is_greedy() && params.grammar_bitmask.is_none() {
    return Ok(argmax(ctx, logits)?);   // device argmax over raw logits
}
```

`SamplingParams::is_greedy()` is `self.temperature <= 0.0`.

## Problem

**`logit_bias` was dropped.** The host sampler adds the bias *before* taking its
argmax (`sample.rs`, "Apply logit_bias before any temperature/filtering"). The
device fast path argmaxes the raw logits. A request with `temperature = 0` and a
non-empty `logit_bias` therefore ignored the bias entirely. `grammar_bitmask`
had an explicit guard; `logit_bias` did not — the same hazard was handled once
and missed once.

**Ties resolved differently.** `sampling.cu`'s `warp_reduce_argmax` breaks ties
to the LOWEST index (`other_val == val && other_idx < idx`); the host
`argmax_logit` used `max_by`, which returns the LAST maximum. So even with no
bias and no mask, a tied top logit produced a different token depending on which
path ran. bf16 logits over a ~150k vocab tie often enough for this to be real,
not theoretical. The divergence was already documented in
`merge_vocab_shard_argmax`'s doc comment and left standing.

## Root cause

The fast path re-derived the sampler's own precondition instead of asking for
it. "Can I skip `sample_token`?" was answered by a predicate that only knows
about temperature, sitting in a different crate from the code whose behavior it
claims to reproduce. Any new logits-rewriting knob would silently inherit the
same bug.

## Fix

`SamplingParams::is_raw_argmax()` in `infer-plan`, next to `sample_token`, is
now the single predicate licensing the fast path. It is greedy AND no
`grammar_bitmask` AND empty `logit_bias`. Its body destructures `Self` with
every field named, so **adding a field to `SamplingParams` is a compile error
here** — the author has to state whether the new knob rewrites logits. Verified
by adding a probe field and observing `E0027: pattern does not mention field`.

Ten fast-path gates now read `is_raw_argmax()`: both CUDA samplers, DSv4's
`fast_head`, the Qwen3.6 batched decode and spec emission paths, and Metal's
`sample_inflight` plus its two pipeline gates. Sites that ask about the sampling
POLICY (spec-decode accept rules, batched-DSpark routing) keep `is_greedy()`.

`argmax_logit` now breaks ties to the lowest index, matching the device kernel.
This changes the host path's output on exact ties — CPU backend, and greedy
requests carrying a bias or grammar mask.

`crates/infer-plan/tests/raw_argmax_gate.rs` fails if a rewriting knob stops
vetoing the fast path, or if the fast path and the sampler disagree.

## Rule

**A fast path must ask its slow path for permission, not re-derive it.** The
predicate belongs next to the code whose behavior it claims to reproduce, and
must be the only thing the shortcut consults.

**When a predicate licenses skipping work, enumerate the inputs exhaustively and
make the compiler enforce it.** A destructure that breaks on a new field turns
"someone will remember to update this" into a build failure.

**A documented divergence is still a bug.** `merge_vocab_shard_argmax` spelled
out that host and device tie-break differently, and that comment stood while the
fast path assumed they agreed. Writing the discrepancy down is not the same as
resolving it.
