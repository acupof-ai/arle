# DSv4 prefill indexer wq_b → DeepGEMM: −95.5% kernel (default-OFF pending a planted-answer needle)

## Context

After wq_b/wo went to DeepGEMM, the DSA indexer query projection (`csa_select` →
`indexer.wq_b`) became the #1 remaining projection: 134.9ms = 67% of the post-wq_b/wo
linear time at M=1024 (clean rank-0 linear-profile). Wired it through the same
`prefill_proj_deepgemm` helper: added `wq_b_deepgemm` to `Dsv4Indexer` + a loader
cache + threaded the prefill scratch into `csa_select`.

## What worked (perf) / what's unresolved (correctness)

Clean per-stage A/B (8×H20 TP=8, M=1024): indexer_wq_b **134.9 → 6.05 ms (−95.5%,
22×)** — consistent with wq_b/wo. The full projection family (wq_a|wkv fused, wq_b, wo,
indexer) is now ~542 → ~30 ms when enabled.

**Correctness NOT yet established** — kept default-OFF (`ARLE_DSV4_PREFILL_INDEXER_DEEPGEMM`,
separate from the wq_b/wo flag). Unlike the residual wq_b/wo (byte-identical needle),
the indexer feeds the **top-k block selector**, so an FP8 numeric flip could change
*which blocks* are attended. The 37-tok needle is < sliding_window (128) so the indexer
never runs for it; an ad-hoc long-context needle (passcode + 150 filler + query)
diverged DG=0 vs DG=1 (`[436,260,295,…]` vs `[436,86,2358,…]`) but has **no planted
answer** to gate on — uninformative (could be selection diff or long-context
non-determinism).

## Rule

- A projection that feeds a SELECTOR (top-k, argmax-routing) needs a planted-answer
  correctness gate, not a residual-stream byte-identity check — the non-linearity
  amplifies tiny FP8 diffs. Default-OFF until a dsv4_parity-style planted-answer
  long-context needle (CSA active, answer far above the top-k boundary) confirms
  flag-on retrieves it. Same discipline as `feedback_correct_inference_not_baseline_identity`.
- The residual wq_b/wo (no selector downstream) are safely default-ON; the indexer is
  the exception precisely because of the selector.
