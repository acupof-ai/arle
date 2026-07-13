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

### Result (GPUs 3-6, TP=4, `--dspark-conf-threshold 0`)

- On the Roman-Empire prompt: accepted 16 / drafted 260 (0.0615) vs 0/300 pre-fix.
  **But that prompt's greedy generation LOOPED** (target repeats 19738/26964/10395)
  — §0.1: a looping decode is not a valid test case. The "gain" was the Markov
  head coincidentally hitting the repeated target tokens.
- **On STABLE targets (raw `/v1/completions`: "The capital of France is"→" Paris.",
  "Water is made of hydrogen and"→" oxygen atoms."): accepted 0 / drafted 170 —
  accept ≈ 0.** The inverse-RoPE fix is correct (a real MLA geometry bug, mirrors
  the validated main path) but its benefit is MASKED by a dominant second bug.

## Open — root cause #2 (dominant): block_hidden is a CONSTANT

`base_argmax` (pure forward, pre-Markov) on stable targets is pegged at token
**24132 for nearly every anchor AND every block position** (`[24132×5]`),
independent of the input — even at position 0, whose input is the varying anchor
embedding. So the draft forward outputs a **constant block_hidden** carrying zero
information; the tied head always emits 24132. All apparent accepts ride on the
Markov bigram, never the forward.

### Bisect results + eliminated suspects (2026-07-13)

`[dspark-stat]` L2/spread bisect (embed → context → stage0/1/2 → exit) shows the
stages DO vary with input (stage0-2 L2 in the 100s–1000s, cross-row spread 4–8);
the RMSNorms confound a clean "spread→0" localization (each resets magnitude).
`base_argmax` is directionally degenerate — attractor-collapse to a SET
({9722, 112434, 24132, 35119, 18942, …}) with some input variation, not a single
constant. My earlier "pegged at 24132" was one state; the general failure is
attractor-collapse, not a frozen constant.

**Eliminated (verified correct, do NOT re-check):**
- **Exit head** — structurally mirrors `head_normed_rows` (`mtp.2.norm(mtp.2.hc_head(·))`).
- **`main_proj` tap-fuse interleave** — the HC stream is LANE-MAJOR
  (`dsv4_mhc.cu:140` `col = idx % hidden_size`, lane `l` at `[l*hidden..(l+1)*hidden]`);
  the fuse slices `tap[r*hidden..(r+1)*hidden]` — matches. Correct.
- **inverse-RoPE value tail** — fixed here.

- **Output head** — the draft correctly decodes via the base model's separate
  `head.weight` (`model.lm_head`, `tie_word_embeddings=false`, verified in the
  checkpoint index; `dsv4.rs:392-396`). "Tie to embed" is a known false positive.

**Remaining locus (next):** `dspark_stage_forward` (`dsv4/dspark.rs:135-426`)
REIMPLEMENTS the DSv4 block inline (HC pre-norm → wq_a/q_norm/wq_b, wkv/kv_norm,
prepare_qk, draft attention, oproj, hc_post, MoE + shared expert, hc_post) rather
than reusing the validated `mtp_forward_level` / main layer forward. The bug is
almost certainly a subtle divergence here — an fp8 weight+scale pairing, an HC
combine/residual detail, or a MoE routing detail — since every OTHER piece is now
verified. Next: line-by-line diff of `dspark_stage_forward` against the validated
`mtp_forward_level` + the main DSv4 layer forward, one divergence at a time.
Estimated multi-cycle. NOT the window, NOT geometry, NOT exit/fuse/head.

## Rule

MLA spec-decode draft attention that caches K==V (the full compressed latent as
the value) MUST inverse-RoPE the value's `rope_dim` tail at the query position
before o-proj — the same step the validated main-model kernel does. Omitting it
runs cleanly (shapes consistent over the full `head_dim`) but leaks a
position-encoded near-constant into the hidden → attractor-collapse that reads
like a weight/training bug but is pure MLA value geometry. A `(void)rope_dim;` in
an MLA value path is a red flag.
