# DSv4 batched FlashMLA decode (#228) — fix verified, issue closed

## Context

#228 (2026-06-14): batched FlashMLA decode (`sparse_decode_fwd(b=N)`) produced
garbled output at c≥4. Root-caused and fixed in `b4fec44b` (2026-06-15): the
batched indices reader used `max_topk_unified` (640) as stride while the writer
used per-layer `topk_unified` (128/256/640), so rows ≥1 read wrong offsets on
non-max layers. The fix shipped but never had a bench entry, and the issue
stayed open.

This round (2026-08-19) verified the fix on the pod with the `dsv4_parity`
example's batch-decode validation (c=1 reference + batched run, byte-parity
check).

## What worked

- `ARLE_DSV4_BATCHED_DEBUG=1` dumps per-layer `num_splits`, `topk_length`,
  first-8 indices per row, and per-row output checksums — confirmed indices
  row offsets, split counts, and output magnitudes all correct.
- Batch validation: batch=4 `byte_parity=true` (all 4 rows match c=1
  reference). Batch=8: 5/6 runs pass; the 1 failure is #229 (a separate,
  non-FlashMLA, non-deterministic bug — reproduces on the scalar eager kernel
  too).

## Numbers

| Metric | Before (2026-06-14) | After (2026-08-19) |
|--------|---------------------|---------------------|
| c=4 decode | garbled | `byte_parity=true` |
| numdiff maxdiff (row1) | 5.75 (~115× thresh) | 0 |
| batch=8 pass rate | N/A | 5/6 (1 failure = #229) |

Pod: 4×H20, TP=4, `DeepSeek-V4-Flash-0731`, 250-token prompt, max_new=8.

## Rule

A fix commit without a bench entry leaves the issue unverifiable — the
`b4fec44b` fix sat unproven for 2 months because the bench entry was deferred
and never written. The `dsv4_parity` batch-validation harness
(`INFER_DSV4_BATCH_DECODE_VALIDATE`) is the reusable gate: c=1 reference +
batched byte-parity, no HTTP/tokenizer/scheduler in the loop.
