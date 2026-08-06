# A field's doc comment decided the geometry, and I shipped it

## Context

Auditing the offline DSpark draft trainer, I filed a P0: "training's visible
context is one token behind the serve's." It was wrong in the one direction
that matters, and it survived a seven-agent audit, an adversarial refute pass,
and my own source read. `c4e7a2286` shipped it; `1fdf2817c` reverted it.

The fix I shipped moved the training context's upper bound from `anchor` to
`anchor + 1`, putting `taps[anchor]` in reach.

## Root cause

I read `anchors[i] = decode_rows[i].last_token` with `starts[i] = kv_seq_len`
and concluded the anchor sits at `start - 1`. Both fields are declared four
lines apart in `crates/infer-plan/src/lib.rs:44-47`:

> `last_token`: Last token produced for this slot, **used as the next decode input**
> `kv_seq_len`: Logical KV sequence length **already present**

"Produced" is past tense and "already present" is a length, so the anchor is
*not yet* in KV: it occupies `start`. The write sites say the same thing.
`executor/qwen35.rs:1893-1897` forwards `last_token` at `kv_seq_len` and
appends its tap at that index; `:2183` appends the anchor's own tap only after
the verify. At draft time the ring is `[ctx_base, start)`, strictly below the
anchor.

I never read the field docs or either write site. I read `ensure!(df.ctx_end
== start)` and reasoned about what `start` *should* mean.

## Why nothing caught it

- **Relative RoPE geometry is invariant to the error.** Shifting queries and
  keys together by +1 leaves every offset unchanged. No shape, index, or
  length assertion can fire. Only one key's *content* differs.
- **I wrote the tests from the same premise.** All three new gates hardcoded
  `let start = anchor + 1`, then re-derived the span from it. A test that
  transcribes the hypothesis cannot refute it, however faithfully it copies
  the serve's formula.
- **The refute pass inherited the premise.** It was given the claim to attack,
  not the question to answer, so it checked the derivation rather than the
  reading it rests on.

The defect was worse than a mismatch. `trainer.rs` projects
`last_hidden[anchor]` through `lm_head` for row 0's target; `taps[anchor]` is
that same position's residual at layers 40-42. The change put a near-linear
shortcut to the answer on the key nearest row 0's query. Its signature is a
loss that falls faster with acceptance flat — the exact false signal the
tranche existed to remove.

## Fix

Upper bound back to `anchor`. The other half of the finding was real and
stands: the serve's span narrows one key per row, training gave every row the
same width, so row `t` saw `1 + t` keys inference never supplies.
`draft_positions` is `anchor + t` again and is also the supervising trunk
hidden, so `target_hidden_positions` is deleted rather than kept as a second
name for one quantity. The three gates now assert the opposite, and
`a_block_is_blind_from_its_anchor_onward` carries a `taps[anchor-1]` positive
control so "blind to everything" cannot pass it.

## Rule

**An index convention is settled by the field's declaration and its write
site, not by the name at the read site.** `last_token` and `kv_seq_len` are
descriptive names whose tense and units carry the whole answer; I inferred a
convention from how they were *used* one layer up.

**When queries and keys are indexed by the same offset, an off-by-one in the
base is unobservable by construction.** Nothing but content can distinguish
it, so the gate must be a content perturbation with a positive control — bump
`taps[a-1]` and require movement, bump `taps[a..]` and require none. A test
that reproduces the formula under test cannot discriminate, no matter which
source it was transcribed from.

**Give a refuter the question, not the claim.** "Is this finding real?"
inherits the finding's premise. "Where does the anchor sit, and what proves
it?" does not.
