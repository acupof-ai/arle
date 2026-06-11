# DSv4 MTP linear depth-K KILLED — LINEAR-chain wall (first thought draft-quality; corrected: EAGLE-tree off the SAME head reaches 2.44 tokens)

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

## CORRECTION (2026-06-11, same day) — this was MIS-ATTRIBUTED

Research into DeepSeek's own MTP (`docs/research/2026-06-11-dsv4-mtp-eagle-and-decode-operators.md`)
overturns the "head too weak" root cause. **DeepSeek-V4-Flash ships exactly one
NextN module (`num_nextn_predict_layers=1`) — the same head we have — and SGLang
chains it 4 steps via EAGLE to a measured 2.44 accepted tokens / step** (2.18 at
3-token; 1.8× decode at bs=1, LMSYS). The single head provably CAN draft ~2.44
tokens. Our 1/4 is **3.4× worse than the identical-architecture head**, so the
collapse is our **linear chain**, not the head:

- We draft topk=1 **linearly**; SGLang drafts a **tree** (eagle-topk>1) and
  verifies all paths — a linear chain forfeits the whole tail on the first wrong
  draft (the 1/4 plateau).
- We feed the single MTP layer's own stream back as `h_prev` (off-distribution);
  the tree hedges this for SGLang.
- Our chain attends the **shared** frozen target KV; the draft needs its **own**
  scratch KV so draft *i* sees draft *i−1* without corrupting the target (the
  2-cycle `[223,4489]` smells like draft *i* blind to the chain — confirm by
  logging draft attention length per step).

The depth-1 **clamp stays correct as a safety** (our *broken* linear depth-K IS
worse than depth-1), but the wall is a **chain bug**, not draft-head quality and
not a missing tree. SGLang's *default* DeepSeek config is a **linear** chain
(`--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-
tokens 4`) reaching 2.18–2.44 tok — `topk=1` = linear, no tree. So the fix is to
**debug our linear chain to SGLang parity** (no kernel build); the chain already
writes+reads its own KV (Explore confirmed — "shared-KV blindness" falsified), so
the 2-cycle is off-distribution feedback / position-RoPE / feeding the wrong
hidden, to be isolated on the pod one variable at a time. Tree (topk>1) is a
later optional gain and our kernels can't do it yet (FlashMLA `s_q=1`, no mask).

## Rule

**Before declaring a spec-decode depth wall, compare acceptance against the
reference impl on the SAME architecture.** Our single NextN head is identical to
DeepSeek's; SGLang gets 2.44 tokens from it, we got 1 — that 3.4× gap is the
evidence that the *driver* (linear chain, shared KV, off-dist feedback), not the
*head*, is the limit. Measuring per-position acceptance (`matches=`) was right; the
error was concluding "head too weak" without the reference-impl cross-check. A
linear chain off a single head plateaus at ~1 accepted; an EAGLE tree off the
SAME head reaches ~2.44. "Depth-K is dead" → "linear depth-K is dead; tree depth-K
is the open lever."
