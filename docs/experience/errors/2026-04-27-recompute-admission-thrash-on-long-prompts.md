# Recompute admission preemption — structural thrash on long prompts

## Context

Phase 1 of the
[`active-kv-swap-out-unification`](../../projects/active-kv-swap-out-unification.md)
plan wired `PreemptionMode::Recompute` through the CUDA admission path:
when a new prompt's full ISL doesn't fit in the page budget (and
mixed-batch refactors couldn't free enough), pick an active victim,
call `requeue_preempted_decode` (existing helper — frees pages, drops
prefix retention, re-enqueues with cached prompt), retry admission.

Verified on the matched 2026-04-26 SGLang config (`--num-slots 16
--max-seq-len 4608 --mem-fraction-static 0.94 --chunked-prefill-size
4096`):

- First bench (no grace): 502 server-side requests in ~11 min for what
  guidellm should have driven as ~150 unique prompts. Server log full of
  `Request N: preempting (recompute) — 1 generated tokens` — newly
  admitted slots being kicked out before they could amortize prefill.
  Recompute storm exactly as `R1` of the design doc described.
- Second bench (grace = 8 generated tokens before preemption-eligible,
  cap = `running_batch.len()` per tick): still 457 preempts in 60 s
  c=16 leg. `tpot_p50=300 ms` (vs healthy ~50 ms). Still thrashing,
  bench never converged.

Killed both benches.

## Root Cause

Recompute admission preemption is structurally unsuitable for the
canonical 4096-in / 256-out workload — the math doesn't work:

- 1 chunked prefill of 4096 tokens ≈ **1 second of GPU work** at L4
  HBM bandwidth.
- Each preempted slot loses its decode progress and re-prefills. With
  256 max output tokens, the *entire* useful output value of one
  request is ~256 tokens × 50 ms = **12.8 seconds of decode**.
- A request preempted once spends `1 s prefill + 1 s re-prefill +
  decode = ~14 s` to produce 256 tokens. Two preempts: ~15 s. Three:
  ~16 s. Each preempt strictly inflates total cost.
- With the pool admitting only ~6 concurrent at the matched flags,
  any c=16 client load forces aggressive preemption. The system
  livelocks: every admission kicks a partially-decoded request,
  which re-enqueues and demands another admission, which kicks
  another partially-decoded request…

For c=16 / 4096-token prompts, **Recompute is worse than the
"hard wait" admission baseline** — at least the baseline lets some
requests complete while others queue.

## Fix

The implementation was prototyped end-to-end (decode-side helper +
admission `'preempt_retry` loop in `runtime.rs::assign_slots`),
benched twice, and **reverted in full before committing** — both
attempts thrashed (502 / 457 server-side preempts in a 60 s c=16 leg
that should have driven ~30 unique requests). What remains in the
tree from this investigation:

- This errors entry — the discovery record.
- The original Phase 1 design — see
  [`projects/active-kv-swap-out-unification.md`](../../projects/active-kv-swap-out-unification.md)
  Phase 1 still describes the same wiring; the lesson here updates
  it with the per-prompt-length amortization rule from §Rule below.
- No code changes — the runtime is unchanged from `12787f5c`.

The *strategy* — Recompute mode — is wrong for long-prompt workloads.
The next attempt is **Phase 2: Swap mode** in the same project doc.
Swap doesn't re-prefill — it copies KV pages T0 → T1 (~23 ms for 16
slots' worth) and back, preserving generated-token state. For
4096-token prompts, swap cost (≤50 ms round-trip) is 20× cheaper
than the ~1 s recompute cost. The math finally works.

Phase 1 may still ship later as the **PreemptionMode::Recompute
arm** of the unified path, gated for short-prompt workloads (chat
agents with ≤256-token prompts), where the amortization rule in
§Rule below holds in the right direction.

## Rule

**Validate preemption strategy against per-prompt amortization before
shipping it.** Recompute is sensible *iff* per-request prefill cost
is small relative to per-request decode value. The decision rule:

```
recompute_amortizes ≡ prefill_cost_per_token × prompt_tokens
                   ≪ decode_cost_per_token × max_output_tokens
                   × expected_completion_probability_per_admission
```

For Qwen3-4B / L4 / 4096-in / 256-out, prefill cost ≈ decode cost
on per-token basis, and prompt is 16× the output → recompute is
fundamentally upside down.

For short-prompt workloads (e.g., chat agents with 256-in / 1024-out),
recompute IS the right answer because re-prefilling 256 tokens is
cheap relative to the 1024 tokens of decode value preserved.

The Phase 1 wiring is **not** in the tree (reverted, see §Fix).
The eventual unified-path implementation should ship Recompute as
one arm of `PreemptionMode`, with the grace-period guard, gated for
short-prompt regimes by either explicit operator config or by an
amortization-aware autoswitch. Phase 2 (Swap) is the default for
the long-prompt regime that exposed this issue.

## Cross-references

- Project plan:
  [`projects/active-kv-swap-out-unification.md`](../../projects/active-kv-swap-out-unification.md)
- SGLang head-to-head exposing the gap:
  [`wins/2026-04-26-bench-guidellm-cuda-l4-vs-sglang-c1-c16.md`](../wins/2026-04-26-bench-guidellm-cuda-l4-vs-sglang-c1-c16.md)
- (Future) the grace-period guard the eventual Recompute arm will
  ship should reference this entry in its constant's comment so the
  next maintainer doesn't reset `MIN_GEN_TOKENS_TO_PREEMPT` to `0`
  thinking it's pessimistic.
