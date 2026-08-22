# The group layout was the wrong oracle for a shared NVFP4 base

## Context

`--share-frozen-base` extended to NVFP4: the student borrows the serving
engine's Marlin-repacked bytes instead of uploading its own copy. The gate was
`cuda_marlin_fp4_dequant_matches_group_layout` -- dequantize the same weight
through both layouts, require exact equality. It passed on GPU, twice.

An end-to-end A/B on `ThinkingCap-Qwen3.6-27B-NVFP4` then disagreed: greedy
rollouts, fixed seed, same 20 prompts, same 2 trained records, and the two arms
came out at `mean_loss=0.3363` (shared) against `0.6414` (private upload).

## Root Cause

Two separate things, and only the second was a defect in the code.

The test was not a defect but it could not have caught one. Its scales are
powers of two, chosen so nothing is lost -- and `repack_for_marlin_fp4` has
exactly one lossy step, flushing lifted values below 2.0 to zero. On a real
checkpoint that step fires, so the engine's Marlin bytes are NOT the
checkpoint's group bytes, and "the two layouts agree" is a claim about a case
the feature never operates in. The oracle was co-selected with the input that
makes it hold.

The feature's actual contract is not layout equality. It is that the student
reads what the engine serves. The right oracle is `marlin_fp4_gemm` -- the
kernel the engine runs -- over the same packed buffer, with scales that do
flush. Under that oracle the borrowed view is correct.

The loss gap is therefore expected: the shared arm trains against the engine's
post-flush weights, the private arm against the checkpoint's pre-flush weights.
The shared arm scoring lower is consistent with its student matching the engine
that generated the rollouts, which is the point of the feature. The magnitude
was not quantified -- the flush rate on the real checkpoint was never measured.

A separate check confirmed the borrow mechanism itself: the same A/B on the FP8
checkpoint, which has no lossy repack, gives `0.1220` on both arms.

## Fix

`cuda_marlin_fp4_dequant_matches_the_serving_gemm`: compare the borrowed view
against `marlin_fp4_gemm` on non-power-of-two scales spanning E4M3 exponents
2^-9..2^6, including the ones the repack flushes.

## Rule

When a path has a lossy step, a test whose inputs avoid that step is testing a
regime the code never runs in. Pick the oracle from what the feature promises
-- here, "the student sees what the engine serves" -- not from whichever
reference is easiest to compare against.
