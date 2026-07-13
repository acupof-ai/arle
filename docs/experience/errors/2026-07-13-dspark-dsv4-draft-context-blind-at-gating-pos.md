# DSpark-for-DSv4 draft is context-blind at the gating position — accept≈0, "geometry solved" overturned

## Context

DSpark speculative decode for DeepSeek-V4-Flash, TP=4, GPUs 3-6. A prior win
([geometry-solved-absolute-rope](../wins/2026-07-13-dspark-dsv4-geometry-solved-absolute-rope.md))
declared the draft geometry SOLVED at `accept_rate 0.143` and named the
per-committed-token **context window** as the next lever to reach DeepSeek's
60-85%. Both claims are **wrong**. This entry supersedes that win.

## Root cause (measured, token-level)

Two clean pod probes on the b350b0f90 baseline (`INFER_DSPARK_DEBUG=1`):

1. **Window is not the lever.** A 200-token generation grows the committed
   context window from ~1 to ~200 tokens. `accept_rate` stayed **flat at 0.0** —
   deepening the window contributed nothing.

2. **The draft is a context-blind function of the anchor.** With
   `--dspark-conf-threshold 0` (full block always drafted): 60 chains, 300
   drafted, **0 accepted**. The decoded tokens show the mechanism:
   - Same anchor → **identical early drafts**, every time, at different positions
     with different accumulated context: `15363 → [22974,118277,2535,…]` (×4);
     `128822 → [105553,6187,…]` (first two identical ×4); `16 → [11274,6189,…]`.
   - The **target** for the same anchor **varies** with context
     (`15363 → target[0] = 16, 16, 1531, 16…`). The draft predicts a fixed token
     per anchor; the true next token is context-dependent. `draft[0]` never
     tracks `target[0]` → accept≈0.

The attention path IS live and context-sensitive — proof: for a fixed anchor the
drafts **diverge at positions ≥2** across occurrences, which only context-varying
base logits can cause (the Markov chain is deterministic given the anchor). But
at the **gating position (draft[0])** the context-blind prior
(`markov_w2 · markov_w1[anchor]`) dominates the attention-derived base logits, so
the anchor-position prediction is a bigram, not a context-aware forward.

The earlier `0.143` was **noise**: it came from ONE France case whose single
match was token 11111 (a repeated-digit token the Markov prior happened to hit).
On a real prompt the same geometry scores 0/300.

## Rule

**One aligned match (accept≥1) does NOT prove draft correctness** — it proves one
token aligned. Attribute over a MULTI-PROMPT slice with full blocks
(`--dspark-conf-threshold 0`) before declaring geometry solved; a bigram-degenerate
draft passes the single-match test on any repeated/common token. The §0
case-as-fact bar: decode ≥1 real prompt, not a smoke shape, and check whether the
draft's **gating-position** prediction tracks the context-dependent target — if
`draft[0]` is constant per anchor while `target[0]` varies, the draft is
context-blind regardless of accept magnitude on a lucky case.

## Refinement (measured — `base_argmax` probe)

Added a diagnostic field: `base_argmax` = the draft's pure-forward greedy pick per
block row BEFORE the Markov bias. One full-block run (Roman-Empire prompt, 60
chains / 300 drafted / **0 accepted**) splits the hypotheses:

- **"Markov masks a working forward" — REJECTED.** With the Markov bias removed,
  `base_argmax[0]` STILL never matches `target[0]` (anchor 7726 → base 53316 vs
  target 4608; anchor 223 → base 4649 vs target 29658). The forward itself is
  wrong, not merely biased.
- **"Fully context-blind" — REJECTED.** `base_argmax[0]` DOES vary with context
  for a fixed anchor (7726 → 53316 then 38946; 223 → 4649/97267/70610/117626).
  Attention is doing something.

The real failure: **the 3-stage draft forward is numerically degenerate** —
context-sensitive but producing a wrong distribution that collapses toward ~7
attractor tokens (53316, 125095, 113530, 117626, 127279, 70610, 36613) across all
anchors/positions, often the same token repeated across a whole block
(`base_argmax=[53316,53316,53316,53316,53316]`). Never target-aligned.

## Open (next, unstarted)

This is a correctness bug in the draft forward, NOT geometry/window/Markov.
Attractor-collapse across all contexts smells like a systematic
weight-mapping/layout or norm-scale error. Cheapest first: (a) audit the draft
weight loading / tensor-name mapping for the DSv4-specific pieces the shipped
Qwen3.6 DSpark doesn't have (MLA `wq_b`/`wkv`/`wo`, `main_proj` HC-fuse, DSv4 MoE,
exit `hc_head`→tied `lm_head`); (b) if source-clean, dump `block_hidden` norm/stats
to locate where the attractor-collapse enters (embed→HC→attn→MoE→exit). The Qwen
B track works with the same skeleton, so the bug is in a DSv4-only substitution.
