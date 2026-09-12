# Top-k filter kept k+1 tokens: threshold bisection used the wrong comparison — CUDA, 2026-09-13

## Context

`dspark_sampler_parity` reported probability-family failures on the P1 GPU
batch: `filter probs` 61/258, `draft q row` 61/258, `draft token id` 5/258,
with relL2 0.12–0.37, while the `chain` rejection family passed 252/252.
Failures spread across all eight batch lanes and the top-k / combined filter
cases, so it was not one bad row.

The first hypothesis — and the natural reading of a 24% failure rate at topk8 —
was tie-breaking: top-k over a 16384-vocab distribution has near-ties at the
boundary, and the kernel versus oracle might order ties differently. That
hypothesis was wrong. The failing rows have no tie at the k-th boundary.

## Root Cause

Both filter kernels find the top-k keep threshold with a 32-step bisection over
the cumulative "count of probs ≥ mid", then zero out everything below the
converged threshold. The bisection predicate was

```c
if (count > top_k) lo = mid; else hi = mid;
```

`lo` therefore converges to the threshold at which the kept count is *greater
than* k — rank **k+1** — not rank k. The keep step `probs[i] >= lo` then keeps
k+1 elements (k+2 when a tie sits near the boundary). A genuine top-k keeps k.
The extra element is a distinct, non-tied neighbor carrying several percent of
the probability mass; after renormalization that shifts every surviving
probability by far more than f32 rounding, which is what the relL2 0.12–0.37
measured. On the cited row (`topk8 vocab=16384 B=8 lane=7`) the 8th-largest
probability is 0.003299 and the 9th is 0.003099 (0.94×, not tied); the kernel
kept the 9th.

The discrete `chain` family passed because it consumes sampled token ids and
accepted lengths, which the probability redistribution did not change — a
filter can be materially wrong while the draws it drives stay the same.

The comparison `>` fails top-k by definition: it returns a rank-(k+1)
threshold. Genuine ties at the boundary must still over-keep (every standard
implementation keeps all tokens tied at the k-th probability), and the oracle
models that with `p >= kth-largest`. The correct predicate converges to the
k-th rank:

```c
if (count >= top_k) lo = mid; else hi = mid;
```

## Fix

One operator changed at two sites in `crates/cuda-kernels/csrc/sampling/sampling.cu`:

- `dspark_filter_row` top-k bisection (the dspark verify filter, drives both
  `filter probs` and `draft q row`).
- `gpu_sample_kernel` top-k bisection (the legacy single-block sampler),
  same predicate, same defect. Fixed by class even though it has no live
  caller found (see finding below): the cost of the one-character change is
  zero and a kernel that gains a caller later must not silently keep k+1.

A CPU reproduction using the gate's own xorshift RNG, bf16 rounding, f32
exp/sum, and the 32-step bisection matched the device failure bit-for-bit in
magnitude (relL2 0.24359735 vs the gate log's 2.4360e-1) and, with `>=`,
drives the relL2>2% count across the whole filter/draft sweep from 93 rows to
0. Rows with a genuine k-th-boundary tie still over-keep, and the fixed kernel
then keeps exactly the set the oracle keeps.

The on-card before/after is the contract: positive family counts
61/61/5/0 must become 0/0/0/0, with `chain` unchanged. If `chain` moved, the
discrete path was depending on the extra element and the fix would be wrong.

## Rule

When a bisection converges to an order statistic, the predicate's rank is the
contract. "Count exceeds k" finds rank k+1; "count reaches k" finds rank k.
Verify which rank the downstream threshold test (`>=` keep) implies, because a
one-rank-low threshold silently enlarges the selected set rather than erroring.
A high failure rate at a small k over a large vocab reads first as a tie
problem; check whether the boundary values are actually equal before believing
that.

Two follow-ups, not in this lane:

- `gpu_sample_cuda` / `gpu_sample_kernel` may have no live caller. The FFI is
  declared in `crates/cuda-kernels/src/ffi/sampling.rs:31` and mirrored in
  `crates/hip-kernels/src/lib.rs:157`, but no Rust code calls it and there is
  no safe wrapper; the C symbol is exported so an external consumer is
  possible. Deletion needs its own audit and decision.
- The parity gate's negative control is oracle-side (it corrupts the expected
  value at lane 0; it never perturbs uploaded logits), so it cannot catch a
  device-side rank error like this one. A device-input tooth — perturb the
  logits and require the kept set to change — belongs in the gate-conversion
  queue, not a two-line kernel fix.
