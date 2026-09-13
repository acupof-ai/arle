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

## A third defect, exposed the same way

With both panics gone the harness reached the kernel and every comparison
failed: inf or NaN on the 16-group distributions, finite but wrong on
full-256, and several bands at exactly `rel_l2` 1.000.

The activation scale buffer was 128 times too small. The contract is stated at
`crates/cuda-kernels/src/moe.rs:1601` — `sfa_aligned_m` is the TMA-aligned
leading dimension of the activation scale matrix — and the address arithmetic
at `crates/cuda-kernels/csrc/gemm/dsv4_deepgemm_ops.cu:53` spells it out:
`expert * scale_stride_m * scale_k_blocks + k_block * scale_stride_m + row`.
Each group therefore holds `sfa_aligned_m * k/128` floats, one per row per
k-block, because DeepGEMM scales activations per token. The harness allocated
one scale row per band and passed `M_CAP` as the leading dimension, so the
kernel read past the end of every group.

The three symptoms are one fault. Reads that land on unrelated memory give inf
and NaN; reads that land on the allocation's zeros give an output of zero, and
a zero output is exactly `rel_l2` 1.000 because the metric is the difference
norm over the reference norm.

The oracle carried the same wrong model — `a_scale[bk]`, one scale for every
row of the band — so oracle and harness agreed with each other and both
disagreed with the kernel. Correcting only the upload would have moved the
failure rather than removed it.

## Rule

A buffer sized by a bug is not a bound. When one fix makes a second failure
appear, the second defect was already present and unobserved — do not attribute
it to the fix. This file produced three in a row, each hidden by the one before
it, so treat "the fix revealed a new failure" as the expected outcome in a
harness that has never once run to completion, not as evidence the fix was
wrong. State the capacity a routine assumes as an assertion at the
point of construction, so a violation names itself instead of surfacing as an
index far away.
