# How to audit a negative control — parity gates, 2026-09-12

## Problem

A parity gate runs positive (`got` must match an oracle within a bound) and
again under `--negative-control` (something is corrupted and the same
comparator must then fail). The batch harness only checks that the negative
run exits 0 after printing `NEGATIVE CONTROL OK`. Printing that marker proves
the code path ran, not that the control can fail. A negative control that
mutates something the compared output does not depend on, or that fails by
construction on a host-side edit, prints the marker forever and gates nothing.

Auditing one gate answers three questions. All three must be "yes" for the
control to have teeth.

1. **Is the perturbation on the path the positive arm measures?** Trace what
   the negative branch mutates, then confirm the comparator reads it. A bias
   added to the reference expectation that the comparator diffs against is on
   path. A byte flipped in an input the kernel ignores, or a constant the
   output does not depend on, is not. The failure must be caused by the
   comparison, not asserted independently of it.
2. **Are the comparator families distinct?** A gate that names N families but
   perturbs one shared thing N times has one tooth, not N. Either the
   perturbations hit different outputs/kernel stages, or — stronger — each
   sabotage trips its own family while the others stay green ("specificity":
   not only does mine fail, theirs does not). Reusing a perturbation technique
   (e.g. "scale the expectation 3x") across different kernel ABIs is fine;
   renaming one perturbation is not.
3. **Is the control's bound the same bound the positive arm must stay under?**
   A control that only has to move the output by *less* than the pass
   threshold is vacuous. The corruption must cross the identical floor/slope /
   relative-L2 / exact-equality line the clean arm is held to. A per-element
   arm that requires zero violations is especially robust: corrupting one
   element to exceed its band fails regardless of tensor size. A global
   fraction threshold is vulnerable to dilution — one corrupted row averaged
   over all checked rows and heads can fall under the cap if the case grows.
   Judge the tooth on the corrupted row's own violation, not a global
   aggregate.

## Two shapes of negative control

Both are legitimate, but they prove different things. Do not mistake one for
the other.

- **End-to-end kernel sabotage.** A real input or index set is corrupted and
  fed *into* the kernel; the kernel output is compared against the clean
  oracle. This proves that a kernel bug of the sabotaged kind (wrong gathered
  index, wrong quant scale, wrong weight pointer) would change an output the
  comparator sees and be caught. The strongest gates additionally rebuild the
  oracle under the corruption and require the kernel output to match *that*
  (self-consistency) while differing from the clean reference — which
  distinguishes "the kernel honored the bad input correctly" from "the output
  moved for an unrelated reason." Example: the FlashMLA decode gates feed a
  bad start position / masked selected index into `run_fwd`, then assert the
  output differs from clean AND stays consistent with the corrupted oracle.

- **Host-side comparator-liveness check.** After the device output has been
  fetched, a host copy of an already-produced value (the reference, or the
  fetched output) is mutated and the verdict is recomputed. This proves the
  comparator itself is capable of failing — that its bound is not loose enough
  to pass everything. It does NOT prove a kernel defect producing that output
  would be caught; the clean positive run is what covers the kernel. This
  shape is a cheap unit test of the tooth, not an end-to-end corruption. It is
  acceptable when labelled as such and paired with a green positive run; it is
  a fraud when presented as a kernel control.

The dead-tooth failure mode specific to this shape: edit a host copy and then
assert the edited copy differs from the clean reference, with no recomputation
and no kernel rerun. That comparison is true the instant the assignment runs
and cannot fail. A tell in the code is a comment like "rerun is not needed;
the family flag is whether [the edited array] differs." The honest test of an
index/selection output is to feed the tampered selection through the builder
or kernel and compare what it actually produces, as the sibling gates do.

## Worked example: documented but unproven individually — moe_routing

`moe_routing_parity` defines eight comparator families (route indices,
counts, offsets, totals, packed slots, m_indices, weights, combine). Its
negative control explicitly sabotages only six. The `totals` and
`m_indices` families have no dedicated corruption; they are expected to trip
*transitively* if counts/offsets/pack are wrong, because they share the
counts/offsets arithmetic. The gate's own header states this rather than
claiming eight independent teeth.

This is honest and useful, but the two families are not proven individually:
there is no corruption that fails totals while leaving its transitive inputs
green, so a bug isolated to the totals reduction or the m_indices fill could
escape the negative run and still print `NEGATIVE CONTROL OK`. When auditing,
record this distinction separately from a fully-dead tooth — "covered
transitively, no direct tooth" — and add a direct sabotage if that stage is
important enough to warrant an independent guarantee.

## Device/launcher SKIP is not coverage

A gate that prints `SKIP: requires sm_70, device is sm_90` (or that needs a
multi-rank model launcher the batch does not provide) executes neither arm on
that machine. A batch summary of `pass=N fail=0 skip=M` must name the skipped
gates and the device/launcher each needs; otherwise a green run is read as
full coverage for operators that were never exercised. A genuine SKIP is far
better than a fake pass, but it is zero evidence on the device that skipped.

## Rule

For every gate under `--negative-control`, verify in order: the corrupted
value reaches the comparator the positive arm uses; each named family is a
distinct perturbation (preferably with cross-family specificity); and the
corruption crosses the positive arm's own bound, measured on the corrupted
element or row rather than a dilutable global fraction. Prefer an
end-to-end kernel sabotage with corrupted-oracle self-consistency; accept a
host-side liveness check only as a labelled comparator unit test. A missing
tooth stated plainly in the gate header beats a tooth that looks covered but
cannot fail.
