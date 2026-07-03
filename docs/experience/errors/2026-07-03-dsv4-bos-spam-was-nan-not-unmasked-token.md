# DSv4 tool-calling BOS/`<image>` spam was NaN logits, not an unmasked token — a symptom-patch reverted

## Context

Agentic tool-calling re-test (DSv4-Flash, TP=4): every long request degenerated
into a repeated special token — BOS (id 0) at temp 0, `<｜image｜>` (129279) at
temp 0.6 — `finish_reason=length`, empty content, no `tool_calls`. Short
requests (2+2, a 10-tok control) were coherent and stopped cleanly.

## Root Cause

**The decode logits are NaN**, not a flat distribution over an unmasked
special token. The probe lens at the first bad decode step (pos 373, 372-tok
context) is decisive:

```
layer 33-36: entropy 5.2→7.3, nll finite, top1=104937   (finite)
layer 37-42: entropy=NaN, nll=NaN, top1=129279          (NaN)
```

The residual stream goes NaN between decode layers 36 and 37. `argmax` over
all-NaN logits returns index 0 → **BOS** (greedy); softmax over NaN lands on
**129279** (temp 0.6) — exactly the two attractors. This is the **known,
pre-existing DSv4 long-context (>128-token, past the sliding-window boundary)
compressed/DSA-attention decode NaN** ([[reference_dsv4_longctx_decode_broken_and_deepgemm_skew]],
2026-06-07: 37-tok clean, 122/333/446-tok garbage; and the
`errors/2026-07-02-dsv4-tp4-nan-from-pos4` correction). The tool prompt (372
tok) merely crosses the 128-tok boundary and triggers it. Not tools, not the
sampler, not TP=4.

The mistaken fix — commit `0c2819a1`, masking input-only special tokens to
`-inf` before argmax — **cannot work**: `-inf` never beats NaN (NaN comparisons
are false, so the argmax running-max stays at the initial `-inf`@0 → returns 0).
The mask helps in zero scenarios (a working DSv4 never argmaxes an input-only
token; a NaN forward can't be masked), so it was reverted here.

## Fix

Reverted `0c2819a1` (mask + its wins stub). The real bug is the long-context
compressed-attention decode NaN — a separate, known-hard CUDA numerical
investigation (which op at the compressed/DSA path first emits NaN once context
> sliding_window); out of scope for the tools-API work. The OpenAI tools render
(byte-verified vs the official `encoding_dsv4.py`) and the thinking-on default
are unaffected and still correct — they need a **short (<128-tok) tool prompt**
to validate end-to-end, independent of this NaN.

## Rule

When a decode metric looks catastrophic (empty output, special-token spam),
**decode the generation's own entropy/nll FIRST** — `entropy=NaN` names a NaN
forward, `entropy=ln(V)` names a flat distribution; they look identical at the
token level but have opposite fixes. Here the probe that already recorded
`nll:NaN` was built in-house and not consulted before writing the patch —
symptom pattern-matched ("input-only token wins argmax") to a plausible-but-wrong
mechanism. A root-cause hypothesis gets license-or-kill too: the cheap verify
(read the degenerate step's entropy) was one grep away.
