# An over-allocation from one bug hid an out-of-range row index from another

## Context

`deepgemm_grouped_prefill_parity` compares the grouped fp8 GEMM against an f64
reference. Two independent defects sat in the harness at once, and the first
kept the second from ever being observed.

## Root Cause

`build_band` was declared `(rng, valid, k, n, zero_weights)` while both call
sites passed `(rng, c, n, k, c == 0)`, so the parameter named `k` received the
output width and the parameter named `n` received the reduction depth. Every
use inside the body then read the caller's values under the wrong names.

Separately, the "flat" case of `sweep_distributions` divided the remaining
token count by the constant slot count instead of by the slots still to fill:

    let take = left.div_ceil(active).min(left).min(M_CAP);

With 512 tokens over 8 slots the takes are 64, 56, 49, 43, 38, 33, 29, 25,
summing to 337, and the trailing `if left > 0 { flat[0] += left; }` placed the
remaining 175 on band 0. Band 0 therefore held 239 rows against a per-band
capacity of `M_CAP` = 128.

The two interact. With the arguments swapped, `a_q` was allocated
`M_CAP * 4096` = 524288 bytes rather than the correct `M_CAP * 2048` = 262144.
Row 238 reads `a_f[238 * 2048]` = 487424, which is out of range for the correct
allocation and in range for the doubled one. The oversized band read from the
wrong rows and produced a number instead of a panic. Correcting the argument
order shrank the buffer to its intended size, and the same access became an
out-of-bounds panic in `band_ref`.

## Fix

Order the parameters to match the call sites. Divide by `active - i` so the
eight takes are 64 each and sum to the token count exactly, and replace the
silent remainder assignment with `assert_eq!(left, 0, ...)`, matching the
one-hot arm. Add `assert!(valid <= M_CAP)` at the top of `build_band`: every
distribution in the file assumes a band fits in one output tile, and nothing
enforced it.

## Rule

A buffer sized by a bug is not a bound. When one fix makes a second failure
appear, the second defect was already present and unobserved — do not attribute
it to the fix. State the capacity a routine assumes as an assertion at the
point of construction, so a violation names itself instead of surfacing as an
index far away.
