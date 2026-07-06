# OPD KL batchmean scale centralized + gradient regression guard

`bench-exempt` — correctness guard on the KL reduction scale, no hot-path change.

## Context
`errors/2026-06-16-opd-kl-vocab-reduction-lr-collapse.md`: a `mean`-over-all-logits
KL reduction left a `1/vocab` rescale that pushed AdamW's `sqrt(v_hat)` below
`eps=1e-8`, collapsing the effective LR ~vocab×. The fix (×vocab in three sites)
sat as a bare `shape.vocab as f32` at each — nothing stopped a re-drop.

## What Worked
- One helper `kl_batchmean_scale(vocab)` (`loss.rs`); forward/reverse/chunked KL
  all route through it.
- Regression test `kl_distill_gradient_is_batchmean_scaled_not_vocab_collapsed`:
  the analytic student-logit grad is `(softmax(s)_j − t_j)/positions`; the test
  fails by exactly `vocab×` if the scale drops. Green on the CPU autograd backend
  (`cargo test -p train --features no-cuda`, full KL suite green).

## Rule
Guard a load-bearing loss scale structurally: one named helper + an analytic
gradient test whose failure mode is the exact `vocab×` discrepancy.
