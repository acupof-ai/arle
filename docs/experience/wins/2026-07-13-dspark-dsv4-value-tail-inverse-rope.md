# DSpark-for-DSv4 value-tail inverse-RoPE — accept 0 → 6.15% (root cause #1 fixed, #2 open)

## Context

DSpark draft for DeepSeek-V4-Flash scored `accept_rate 0/300` with the draft's
tied-head argmax collapsing to ~7 attractor tokens regardless of context
([errors](../errors/2026-07-13-dspark-dsv4-draft-context-blind-at-gating-pos.md)).
A `base_argmax` probe (pure forward, pre-Markov) rejected both "Markov masks a
working forward" and "fully context-blind": the forward was context-sensitive but
numerically degenerate. A source audit vs the working Qwen3.6 DSpark (B track) and
the main DSv4 attention found the DSv4-MLA-only bug.

## What Worked

**The draft attention returned the MLA value with its RoPE tail still
forward-rotated.** MLA has no separate V — K == V == the single compressed latent
(`head_dim` = NoPE + `rope_dim`). The draft cached the latent with a partial
forward-RoPE on its `rope_dim` tail (correct for scoring: q·k relative position),
then summed the FULL latent as the value and returned it raw. The trailing
`rope_dim` dims stayed position-rotated → a position-correlated near-constant
flowed through wo → HC residual → 3 stages → exit → tied `lm_head`, collapsing the
argmax to attractor tokens. The main cr==0 path (`dsv4_swa.cu:86-101`) un-rotates
this tail (inverse-RoPE, sign −1, at the query's absolute position) before o-proj;
the draft kernel omitted it. Qwen's draft has no MLA value tail, so B track works.

Fix (mirror `dsv4_swa.cu` verbatim): in
`dsv4_dspark_draft_attention.cu`, accumulate the value into a `__shared__ out_vec`,
then inverse-RoPE `out_vec[head_dim-rope_dim .. head_dim]` at
`abs_pos = base_start_pos + token` with the SAME rope params the forward q/latent
prep used (`rope_theta`, `original_seq_len=0`, YaRN `factor`/`beta_*`; cr==0, no
YaRN). Threaded `block_abs` + rope params through the FFI signature
(`ffi/attention.rs`) and the call site (`dspark.rs`).

### Result (GPUs 3-6, TP=4, `--dspark-conf-threshold 0`, Roman-Empire prompt)

`spec_decode`: **accepted 16, drafted 260, accept_rate 0.0615** (was 0/300/0.0).
Drafts now vary per anchor and sometimes match the target's leading tokens.

## Open — root cause #2: the forward still copy-degenerates

`base_argmax` (pure forward, pre-Markov) still does NOT track the target — it
collapses to the ANCHOR token (`19738 → [19738×5]`) or a residual attractor
(112434). block_hidden decodes back to the INPUT token, not the next token: the
input-embedding HC residual dominates and the attention/context contribution is
too weak. The 6.15% accepts ride on the Markov head, not the forward. Next: audit
the context-fusion (`main_proj` hc_mult tap mapping, audit suspect #2) and the
attention→HC-residual weighting; confirm vs the DeepSeek reference whether
base_logits are meant to come from block_hidden at all. NOTE: this probe's
generation LOOPED (target repeats 19738/26964/10395) — re-measure on a
non-degenerate decode before trusting the 6.15% magnitude (§0.1: a looping decode
is not a valid test case).

## Rule

MLA spec-decode draft attention that caches K==V (the full compressed latent as
the value) MUST inverse-RoPE the value's `rope_dim` tail at the query position
before o-proj — the same step the validated main-model kernel does. Omitting it
runs cleanly (shapes consistent over the full `head_dim`) but leaks a
position-encoded near-constant into the hidden → attractor-collapse that reads
like a weight/training bug but is pure MLA value geometry. A `(void)rope_dim;` in
an MLA value path is a red flag.
