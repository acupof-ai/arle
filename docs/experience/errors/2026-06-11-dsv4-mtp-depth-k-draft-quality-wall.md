# DSv4 MTP depth-K KILLED — draft-quality wall, not a depth bug

## Context

DSv4-Flash FP8 TP=8/EP=8, 8×H20 pod, B=1 decode. Depth-1 MTP speculative
decode had just landed as a validated win (+39% 问答 / +47% 客服, accept
86/93%, needle-gate clean — see
`wins/2026-06-11-dsv4-mtp-dsa-rollback-selfheal-fix.md`). The depth-1 verify
amortizes ~1.4 tokens/pass (44.04 → ~60 tok/s ≈ 16.7 ms/tok).

To push toward the 6 ms-path target (167 tok/s) the next amortization lever is
**depth-K**: one draft pass predicts K tokens, verify K+1 in one batched
forward, accept the longest correct prefix. Built it (`96ca8b55`): chain K×
`mtp_forward` off the checkpoint's 1-layer `mtp.0` nextn head, longest-prefix
accept, multi-position rollback. K from `--mtp-draft-tokens N`. Drove K=4 on
the pod, flag-only (no env), correctness-gated.

## Root Cause

**The 1-layer checkpoint MTP head cannot draft a coherent chain.** It is trained
to predict token *t+1* given the target's hidden at *t*; chained on its OWN draft
hidden it has no signal and collapses into a 2-cycle.

Pod e2e, depth-4 on 问答 (flag-driven, no env):
- Generation is garbage: `光合作用，简称"什么是"…然而，然而，这是一个"什么"的问题。然而，然而…`
  (depth-1 control on the same prompt is coherent: `光合作用是一个过程，通过它植物、藻…`).
- Drafts loop: `drafts=[223, 4489, 223, 4489]`, `[223, 2619, 223, 4489]` — a
  223↔4489 2-cycle, independent of context.
- Accept is pinned at **1/4** every step: `accepted=1/4` repeated; per-position
  `matches=[true, false, false, false]`. Only draft 0 (conditioned on the real
  target hidden) ever lands; drafts 1-3 (conditioned on draft hiddens) never do.

Accept 1/4 means depth-4 amortizes the SAME ~1 token/pass as depth-1 while paying
4× the draft cost + rollback — strictly worse. A residual K>1 verify-rollback
correctness bug exists on top, but it is moot: even a correct depth-K driver has
no token-acceptance headroom to harvest with this draft head.

This is the #62-stated STOP condition reached empirically. It also confirms the
earlier `25df9494` finding ("depth-2 sequential chain ~33% kill, EAGLE-tree is
the 6 ms path") at K=4 with the productionized flag.

## Fix

Clamp the effective depth to 1 (`executor.rs:1103`, commit `37986aeb`); keep
`--mtp-draft-tokens` accepted and warning when N>1, so the plumbing is ready for
a future EAGLE-tree draft head without re-exposing the broken path. Default
(no flag) is unchanged: validated depth-1 batched verify.

## Rule

**Speculative-decode depth is gated on draft-head quality, not driver
correctness.** A single-layer checkpoint nextn head is a depth-1 instrument —
chaining it past its training signal produces a context-free token cycle whose
acceptance pins at 1/K, so depth-K is strictly worse than depth-1 regardless of
how correct the rollback logic is. Measure per-position acceptance (`matches=`)
BEFORE crediting any depth-K speedup; the win only exists if drafts 1..K-1 land.
The path from 16 ms → 6 ms is a stronger multi-token draft (EAGLE-style trained
head), not a deeper chain off the depth-1 head — and per the B=1 decode washes
(`a222ab64`: per-kernel GPU-time optimization DEAD, mHC overlapped → wall-neutral;
`daccf20a`: decode-graph wash −5%) amortization is the *only* lever that moves
the wall.
