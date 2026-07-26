# The DSpark draft is a good ranker and a bad argmax — top-2 rescues 47% of rejections

## Context

`k_med = 2` out of depth 15 caps everything else in the DSpark tick: shrinking
the block (the previous entry) is damage control, and every remaining
optimization divides a smaller and smaller number. The question that decides
whether candidate trees are worth building is not "how often is the draft
wrong" but **how wrong** — if the token that breaks the chain sits at rank 5000
in the draft's own ranking, no tree width saves it.

`ARLE_DSPARK_RANK` answers it without building anything: at the position that
broke the chain, D2H the draft's logits row and report where the trunk's token
ranked. ThinkingCap-Qwen3.6-27B-FP8 + Qwen3.6-27B-DFlash, 1×H20 GPU 0,
`--dspark-block-size 8 --spec-max-batch 16 --max-running-requests 16`, c=8,
48 requests, greedy — 540 rejections.

## What Worked

| candidate width | trunk token inside draft top-w |
|---|---:|
| 1 | 3.3% |
| **2** | **47.0%** |
| 3 | 63.0% |
| 4 | 73.3% |
| 8 | 87.8% |
| 16 | 95.9% |
| 64 | 99.1% |

Rank median 2, mean 5, p90 10. `k_mean = 2.19`.

**One alternative candidate per position rescues 47% of the rejections that
currently end the chain.** The draft head is not guessing badly — it is ranking
well and losing on the argmax tie-break.

What that is worth, if per-position survival is treated as geometric:
current `p = k/(k+1) = 0.69`; with width 2, `p₂ = p + (1-p)·0.47 = 0.835`, so
`E[k] = p₂/(1-p₂) ≈ 5.1` against today's 2.19 — **2.3× the accepted tokens per
chain**, before paying for the extra verify rows.

The draft side of that is nearly free here, and that is specific to DSpark's
design: the block is a single non-causal forward over mask tokens, so **row r's
logits do not depend on which token was selected at rows < r**
(`dspark_draft_block`, `qwen35/dspark.rs`). Top-2 at every position comes out of
the forward already computed. Only the verify pays.

## Problems

- **Rank 1 at 3.3% is bf16 ties.** The probe counts strictly-greater entries, so
  an exact tie reads as rank 0 even though the accept scan rejected. On a 150k
  vocab with an 8-bit mantissa this is expected; it makes the width-2 number
  slightly optimistic at the margin and does not change the shape.
- The rank is read off the draft's base logits. Valid here because this
  checkpoint has no markov head (the batched-argmax path is taken, which is
  gated on `markov.is_none()`); with one active the proposal ranking would
  differ from the probed ranking.
- Geometric survival is a model, not a measurement. The 2.3× is an estimate; the
  measured quantities are the rank distribution and `k_mean`.
- Nothing was built. Cashing this needs tree attention in the verify path, which
  is an architectural change and is outlined, not started.

## Rule

Before building width, measure rank. "The speculator is wrong 70% of the time"
and "the speculator ranks the right token second 47% of the time" are the same
accept rate and completely different engineering problems — the first says train
a better draft, the second says stop throwing away the ranking you already
computed. One whole-vocab D2H behind an env gate separates them.
