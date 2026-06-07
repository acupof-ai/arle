# DSv4 MTP batched verify LANDS: +61/+70% decode, byte-identical (col1 "bug" was a selftest artifact)

## Context

MTP/EAGLE spec decode at B=1 was net-SLOWER (−11%) because `forward_tokens_verify`
looped two single-token forwards (read the 149GB weights 2×). The batched verify (one
2-token forward) was +63% but appeared to break correctness ("col1 bug"). This entry
resolves both: the batched verify is correct + the win lands.

## What worked

Two fixes made the batched MTP correct AND fast:
1. **Executor batched-reject** (`forward_decode_tokens`): when the batched verify is on,
   reject = `truncate_slot(start_pos)` + `restore_spec_rollback(start_pos+1)` +
   re-forward pending (M=1). The earlier divergence was the per-token reject mis-applied
   to the batched verify, corrupting the rollback.
2. **Retired the col1 byte-identity gate** — it was a SELFTEST ARTIFACT: it checked the
   2-token verify's col1 (bonus) on a FORCED-WRONG draft (wrong_b=token_b+2), which is a
   REJECT case whose bonus real decode DISCARDS (rejects emit only base_next). Any
   byte-identity check there is also confounded by the M=2-vs-M=1 FP8 kernel path
   (SWA-prefill + prefill-DeepGEMM vs FlashMLA-decode + decode-DeepGEMM). On the ACCEPT
   path (real tokens) the argmax is robust to the FP8 delta, so the FULL decode is exact.

Validation (8×H20 TP=8, same binary, SPEC=0 ref vs SPEC=1+batched):

| prompt | ref tok/s | batched MTP tok/s | Δ | output |
|---|---:|---:|---:|---|
| needle | 39.9 | **64.2** | **+61%** | **byte-identical** (accept 15/1) |
| capital | 38.2 | **65.0** | **+70%** | **byte-identical** (accept 16/0) |

Decode **~27 → ~16ms/token**, lossless. `ARLE_DSV4_MTP_BATCHED_VERIFY` flipped default-ON.

## Rule

- A spec-decode verify's bonus is only valid/used on ACCEPT (draft==base_next). NEVER
  gate on the reject-path bonus (discarded) or on byte-identity of an M>1 verify vs the
  M=1 non-spec path (confounded by the different FP8 kernels). The correct gate is
  FULL-DECODE byte-identity vs non-spec on ≥2 prompts (here exact on needle+capital) —
  per `feedback_correct_inference_not_baseline_identity`. I burned many cycles chasing
  the col1 artifact; the full-decode A/B was the truth all along.
- `ARLE_DSV4_SPEC_DECODE` stays opt-in (licensed for B=1: +61%, byte-identical). Default
  flip is deferred: SPEC on routes c>1 to per-row spec (disables batched decode, line
  ~1133) — untested throughput; needs c>1 verification first (multi-shape default-flip
  rule). For the decode-6ms B=1 target, enable ARLE_DSV4_SPEC_DECODE → ~16ms; the
  remaining gap to 6ms is the FlashMLA-decode graph (−10ms launch gap).
