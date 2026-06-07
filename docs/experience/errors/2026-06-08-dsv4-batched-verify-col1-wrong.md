# DSv4 batched MTP verify: +63% speed but col1 (draft position) computed wrong

## Context

MTP at B=1 is net-slower because `forward_tokens_verify` loops two single-token
forwards (reads the 149GB weights 2×). Implemented an opt-in batched path
(`ARLE_DSV4_MTP_BATCHED_VERIFY`, default off): ONE
`forward_tokens_stream_impl(&[pending, draft], start_pos)` + sample col0=base_next /
col1=bonus, with `capture_spec_rollback(start_pos+1)` before the batch and a reject
path of `truncate_slot(start_pos)` + restore + re-forward pending.

## What the A/B showed (8×H20, two prompts, vs spec-off ref)

- **Speed: the amortization works.** Batched verify hit **62.9 tok/s on needle
  (+63% vs spec-off 38.5, +87% vs per-token 33.7)** and 50.9 on capital (+28%). A
  2-token forward ≈ a 1-token forward (memory-bound) — confirmed.
- **Correctness: FAIL.** Batched output diverges from spec-off EARLY on the
  non-degenerate capital prompt: off/pertok both `[11111,14,778,…]`, batched
  `[11111,14,260,…]` (position 2). Per-token spec is byte-identical to spec-off, so
  this is a real bug, not MoE non-determinism.

## Root cause (diagnosed, not yet fixed)

col0 (base_next, the pending position) is **correct** (14 matches); col1 (bonus, the
draft position) is **wrong** (260 vs 778). So the **second token of the 2-token
forward is computed incorrectly** — the reject rollback is not even involved (position
2 is an accept). The prefill driver's multi-token path at `start_pos > 0` (a 2-token
"chunk" attending to KV history + intra-chunk) does not reproduce what two sequential
single-token forwards produce for DSv4's compressed attention — likely the DSA
compressor / sliding-window / head-HC sees the draft (token 2) without token 1's
sequential update. This is the chunked-prefill-at-start_pos>0 path for compressed
attention.

## Root cause localized further (2026-06-08)

`head_hidden_from_stream(stream, token_idx)` extracts the row correctly
(`copy_row_to_hidden(stream, token_idx)`), so the output-extraction is NOT the bug —
**the 2-token forward computes the draft's stream row (col1) wrong**. The smoking gun:
`forward_tokens_stream_impl(seq_len=2, start_pos>0)` is a combination the normal flow
**never exercises** — prefill is always a single chunk at start_pos=0; decode is
seq_len=1 at start_pos>0. The batched verify is the FIRST caller of seq_len>1 AT
start_pos>0 (a "decode-position chunk"), so it hits a latent bug in the
sliding-window/compressed attention's handling of a multi-token chunk attending to KV
history at start_pos>0 (token 0 / col0 is correct; token 1 / col1 — which must attend
to both the history AND token 0 — is wrong). The fix is in the chunked-prefill SW/CSA
attention path, not the verify wiring. Re-validate with a planted-answer needle.

## Rule

- A batched K+1 verify is the right MTP lever (speed proven +63%), but it requires the
  multi-token forward at `start_pos>0` to be **bit-equivalent to sequential single-token
  forwards** for compressed attention — verify that first (dump col1 hidden vs a
  sequential draft forward) before wiring the verify. Gate on **greedy-identity on a
  NON-degenerate prompt** (capital), not the degenerate needle loop (confounded by MoE
  non-determinism per `feedback_correct_inference_not_baseline_identity`).
- Reverted the code (default-off but incorrect = a half-state); re-implement after the
  chunked-prefill col-N correctness is established. Design is in
  `reference_dsv4_decode_6ms_path_state`.
