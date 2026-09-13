# The prefill index negative tooth could not move the output: one weak key swap washed out

Date: 2026-09-13. Found on the first GPU execution of the
`--negative-control` arm of `flashmla_prefill_parity`, right after #433
fixed the HCA index pitch.

## Context

The index-family negative tooth (main `9a19d0f01`) corrupts the CSA
selection at the last token — two visible compressed keys swap — reruns the
production index builder on the corrupted selection, runs the sparse fwd on
the kernel-built indices, and requires output/LSE/maxlogit to leave the
clean-run envelope. On its first real run the index check fired but the
numeric checks did not:

```
output rel 0.00542 vs bound 0.05
lse      1.1e-5  vs bound 0.10
maxlogit 0.0     vs bound 0.05
```

The gate exited 1 with "output negative control did not fire".

## Root cause

The corruption was in a region the attention weights ignore:

- compressed rows are built at magnitude ±0.02 (`comp` latent,
  example old :213-218);
- Q is aligned to the current-chunk keys, which are not in the sparse
  index set at all;
- the swap exchanges one compressed row for another among ~640 rows in
  the softmax.

A one-key substitution of two equally weak rows changes the weighted sum
by ~1/640 of a small number. The tooth exercised the builder (its indices
were provably honored) but could not exercise the fwd: no perturbation
reaches the output through a near-zero weight. This is the dominant-key
washout class f4 measured independently on `flashmla_hca_decode_parity`
and `flashmla_sparse_decode_parity`.

## Fix

Two index regimes, both driven through the real kernel builder:

1. **Strong (asserted).** The negative run builds a pool with one
   designated dominant compressed row — the last token's Q direction at Q
   magnitude, so its pre-softmax score (~7.6) beats every background row
   (~0.4) outright, mirroring the decode gates' K=Q dominant row. The
   corruption drops that one entry from the last token's selection. The
   kernel must produce the shortened row (indices tooth) and the fwd must
   lose a near-total-weight key. A clean-selection forward on the same
   dominant pool is checked against its own f64 reference first, so a
   broken dominant pool cannot masquerade as a fired tooth.
2. **Weak (kept, reported, not asserted numerically).** The original
   visible↔visible swap stays, because that is the regime production runs.
   Its index tooth is still asserted; out/lse/maxlogit are printed with the
   washout label instead of being required to fire. Asserting them would
   have forced either a tolerance move or a data change that hid the
   production regime — both wrong.

Card result, H20 sm90 card 1 (release, kernel AOT bundle hash printed by the run), safe-form exit capture:
positive `POS_EXIT=0` (18/18, unchanged metrics); negative
`NEG_EXIT=0` with `strong out=true(0.07640) lse=true(2.50874)
maxlogit=true(7.20724)` and `weak out=false(0.00538) lse=false(0.00001)
maxlogit=false(0.00004)` as designed. Bounds 0.05/0.10/0.05 unchanged.

Also folded in: the index-compare positive control added in #433 now runs
only when `idx_len_ok && s_q >= 2`, so an `s_q=1` case or a short buffer
cannot trip the control itself.

## Rule

A negative control that swaps one weighted input proves nothing unless the
swapped input carries observable weight: construct the regime's dominant
term, then corrupt it. Keep the weak production regime in the output as a
measured washout row so the gate documents both regimes instead of
replacing the real one with a synthetic one. The same construction
(dominant key aligned to Q, dropped via a real builder path) applies to
all three sparse gates — prefill, HCA decode, sparse decode — and is now
consistent across them.
