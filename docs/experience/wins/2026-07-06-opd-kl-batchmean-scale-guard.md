# OPD KL batchmean scale centralized + analytic gradient regression guard

`bench-exempt` — this is a correctness/regression guard on the KL reduction
scale, not a hot-path perf change. The three KL reductions already multiplied by
`vocab`; this refactor routes them through one named helper and adds a test that
fails by exactly `vocab×` if the scale regresses. No kernel/latency delta. The
capability-A/B that would move a bench number is the 2026-06-16 fix itself,
already landed.

## Context

`errors/2026-06-16-opd-kl-vocab-reduction-lr-collapse.md` traced a silent
effective-LR collapse to a `mean`-over-all-logits (`positions × vocab`) KL
reduction: the constant `1/vocab` (≈1/152k) rescale pushed the per-parameter
gradient second moment `sqrt(v_hat)` below AdamW `eps=1e-8`, degenerating
adaptive normalization into `eps`-dominated scaled-SGD. The fix multiplied the
`mean` result by `vocab` to recover `batchmean` (`sum_v / positions`) in three
places (`kl_distill_loss` forward + reverse, `kl_distill_loss_chunked`), but the
scale lived as a bare `shape.vocab as f32` at each site — nothing structural
prevented a future edit from dropping it again.

## What Worked

- **Single source of truth**: `kl_batchmean_scale(vocab)` in
  `crates/train/src/loss.rs` with `debug_assert!(vocab > 0)` and a load-bearing
  doc comment (scale is NOT optimizer-invariant; blended GKD CE anchor must use
  the same face value, never re-divide by `vocab`). All three reductions route
  through it, so the scale cannot drift between forward/reverse/chunked.
- **Analytic gradient regression test**
  (`test_opd.rs::kl_distill_gradient_is_batchmean_scaled_not_vocab_collapsed`):
  the forward-KL student-logit gradient is analytically
  `(softmax(s)_j − t_j) / positions` under batchmean, vs
  `… / (positions·vocab)` under the buggy reduction. The test computes the
  student grad on the CPU autograd backend and asserts it matches the
  `1/positions` analytic value AND that `max|grad| > 1e-3` (the collapsed regime
  is ~`1/vocab` of this, far below any grad-clip threshold). It fails by exactly
  `vocab×` if `kl_batchmean_scale` is ever dropped.
- `cargo test -p train --profile release-fast --no-default-features --features
  no-cuda`: new test + full existing KL suite green (11/11 KL tests, incl. the
  finite-difference and stable-reference grad checks).

## Rule

A constant loss rescale that "the optimizer should absorb" is a landmine under
AdamW — guard it structurally, not with a comment. Centralize the scale in one
named helper and pin it with an analytic-gradient test whose failure mode is the
exact `vocab×` discrepancy, so a future edit can't silently re-collapse the LR.
