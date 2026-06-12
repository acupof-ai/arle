# DSv4 MTP spec-decode perf KILL — even the re-forward-free ceiling is noise

## Context

frozen-KV made DSv4 MTP spec-decode *correct* at long context (needle exact ×12,
[wins](../wins/2026-06-11-dsv4-mtp-frozen-kv-p1-longctx-fix.md)), but Option A
always-re-forwards the accepted prefix to commit the frozen compressor, which is
perf-negative. The proposed fix A1 — "compressor-only commit": retain the verify's
per-layer hiddens and commit the compressor without the full re-forward — is
invasive (forward-path hidden retention + commit-path compressor). Before building
it, a cheap diagnostic (`ARLE_DSV4_MTP_SKIP_REFORWARD`, off by default) measured the
A1 *ceiling* by skipping the re-forward entirely (correctness-broken; tok/s only).

## The A/B (same binary `2407ef92`, same prompt, max_tokens=256, depth-1, ×3)

| config | tok/s | vs no-spec |
|--------|-------|------------|
| no-spec baseline (sed-stripped `--spec-type` from the *same* serve) | **33.4** | — |
| spec + re-forward (current Option A) | 20.9 | **−37%** |
| spec + A1 ceiling (`SKIP_REFORWARD=1`, no re-forward) | 34.6 | **+3.6%** |

The 8×H20 no-spec baseline is **33.4 tok/s**, not the "~44" the prior summary
carried — that was a cross-day ghost that didn't survive a same-binary measurement
([feedback: matched A/B](../../../README.md)).

## Root cause (why MTP loses on DSv4, structurally)

- The A1 *ceiling* — no re-forward AND no compressor commit, physically un-shippable —
  is only **+3.6%** over no-spec, inside the matched-A/B noise floor. Real A1 must
  still commit the compressor for the accepted prefix, so it sits *below* the ceiling
  → break-even or negative.
- DSv4's compressed/sparse attention makes the spec **verify** expensive per token,
  and the 1-layer NextN draft head caps accept at ~50-70% (depth-1). The amortization
  a draft buys is eaten by the compressed-attention verify cost. depth-K is worse
  (accept 1/4, draft-quality wall). So even a perfect runtime is ~break-even.

## Decision

**KILL A1 and the MTP-perf pursuit on DSv4-Flash.** Not licensed: the invasive
hidden-retention buys ≤ +3.6% (noise) at the impossible ceiling, less in reality.
The batched-verify (s_q=K) kernel work has the same ceiling problem for depth-1
(K=1 is already minimally batched). The levers that *would* win — a multi-layer
EAGLE draft head, cheaper target attention — are **model/checkpoint** changes, not
runtime. frozen-KV correctness stays (spec-decode is opt-in and now non-corrupting).

## Rule

- **License a runtime optimization against its perf *ceiling*, measured cheap, before
  building the invasive version.** A throwaway "skip the expensive step"
  diagnostic (off by default, correctness-broken-when-on) gave the A1 ceiling in one
  serve restart and killed a multi-day hidden-retention build. The ceiling, not the
  current number, is the license gate.
- **The no-spec baseline must be the *same serve minus spec*, measured same-binary
  same-prompt.** A cross-day "44" inverted the verdict; the real 33.4 made spec-decode
  a clear loss. See [[feedback_bench_delta_vs_baseline_not_raw]],
  [[feedback_matched_ab_for_small_bench_effects]].
- **Speculative decode is not free on models with expensive per-token attention.** The
  verify re-pays the target attention; if that attention is heavy (compressed/sparse
  MLA), the draft amortization can't cover it. Check the verify cost against the
  accept rate before committing to a spec-decode perf line.
