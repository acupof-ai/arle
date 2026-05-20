# OPD merge_grad shared-first reverted after wall-clock kill

## Context

`e53654a` changed `crates/autograd/src/tape.rs::merge_grad` to share the
first cloned gradient tensor between the returned gradient map and
`Tensor.grad`. The target metric looked good in the original note:
`merge_grad_seconds` improved, and the local short A/B also showed a full-step
win.

Claude's follow-up in `b50ee90` reran the axis with larger matched controls
and flipped the decision: the target metric still improved, but OPD
step-level wall-clock regressed by 2.6%. Per ARLE SOLID rules, wall-clock is
the licensing frame.

## Root Cause

The original license over-weighted the narrow `merge_grad` counter. Sharing the
first accumulated gradient changed object lifetime and storage aliasing inside
the tape. That can improve the isolated accumulation path while still losing
in the full OPD step once allocator, cache, and downstream access effects are
included.

The root mistake was not the shape assertion or the returned gradient value,
which remained correct. The mistake was accepting a target-metric improvement
before it survived a step-level matched-control check.

## Fix

Reverted the shared-first `.grad` alias path in `merge_grad` and restored the
simple flow:

- keep cloning the first gradient into the returned `grads` map;
- call `TensorStore::accumulate_grad` for tensors that require gradients;
- keep the test assertion that returned gradients and stored tensor gradients
  have matching host values.

No public API changed.

## Evidence

Code-side revert validation:

| run | total_step_s | backward_s | merge_grad_s |
|---|---:|---:|---:|
| after-revert-r1 | 14.707145 | 4.229562 | 1.657045 |
| after-revert-r2 | 14.542637 | 4.203392 | 1.639159 |
| after-revert-r3 | 14.740363 | 4.244210 | 1.662472 |

| metric | after-revert mean |
|---|---:|
| `total_step_seconds` | 14.663382 |
| `backward total_seconds` | 4.225721 |
| `merge_grad_seconds` | 1.652892 |

The revert returns `merge_grad_seconds` to the pre-e53654a range documented in
`2026-05-20-opd-merge-grad-shared-first.md` while honoring the later
wall-clock KILL from `b50ee90`.

Raw outputs:

- `bench-output/2026-05-20-merge-grad-shared-first-revert/after-revert-r1.txt`
- `bench-output/2026-05-20-merge-grad-shared-first-revert/after-revert-r2.txt`
- `bench-output/2026-05-20-merge-grad-shared-first-revert/after-revert-r3.txt`

## Rule

An OPD optimization that wins a local phase counter but loses full-step
wall-clock is dead. Target metrics are only diagnostic; the license decision
must use matched end-to-end wall-clock, with the narrower counter used only to
explain the result.
