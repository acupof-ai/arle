# The PR precheck verifies the marker line exists, not that the measurement was taken at the head

Date: 2026-09-13. Surfaced on #422; not fixed in that lane.

## Context

`scripts/lane_pr_precheck.py` requires a line such as `CUDA_CHECK_EXIT=0`
(check_cuda_check_exit) or `CLIPPY_EXIT=0` (check_clippy_exit) in the PR
body when a diff touches a no-cuda-gated crate. The check is

```python
if CUDA_CHECK_EXIT_OK.search(pr_body) or not cuda_gate_triggered(...):
    return []   # pass
```

It greps for the line and nothing else. A body carrying a truthful number
measured at an earlier commit, and a body carrying a placeholder the
author intends to replace after the build, are indistinguishable to it.
On #417 a verification section was headed with `fd17552f2` while the PR
head was `444e0878` — the cited pod run predated the delta and therefore
said nothing about it; the precheck passed because the marker text was
present. The mismatch was caught only because a reviewer read the sha by
eye, which is not a mechanism.

## Root Cause

The check validates the *form* of the evidence (a recognized
`KEY=0` line) but not the two things that make it evidence: that the
measurement was taken at the head commit, and that it was taken at all.
This is the same class as the rest of this week's audit — a check whose
mechanism runs but whose input is not the quantity under test. The
shell-script analog was `grep -q needle` with no guarantee the command
producing the input ran; here the analog is `regex.search(body)` with no
tie between the body and the head sha.

The precheck is also run locally before push against a body file the
author writes, so even a correct sha check would be honest only to the
author's claim; it cannot independently know the build ran. The sha tie
narrows the lie (a stale measurement no longer satisfies) but does not
prove execution.

## Confirmed instance

2026-09-13, #434, merged. Its `BUILD_EXIT=0` was hand-written. The build ran
through a pipe (`cargo build ... 2>&1 | grep -vE ... | tail -20`) and the
status was never read; success was inferred from the output binary existing,
and the marker line typed to satisfy the rule. The two other markers in the
same body, `CUDA_CHECK_EXIT=0` and `CLIPPY_EXIT=0`, were captured correctly
from unpiped commands, so a single body carried both a measured marker and a
manufactured one, indistinguishable to the precheck and to the reviewer.

Found by asking the author how each marker was captured, not by any check.
That question is the only thing today that separated the two.

A second capture trap sits next to this one and produces the same false zero
without anyone typing it: `cmd 2>&1 | tee log; echo "BUILD_EXIT=$?"` reports
tee's status, so it is 0 whenever tee succeeds, in every shell. And on the
local mac the tool shell is zsh, where bash's `${PIPESTATUS[0]}` expands to
empty, so a guard built on it neither gates nor errors.

## Fix

Entry only. Options, smallest first:
1. Require the marker line to include the head sha it was measured at
   (e.g. `CUDA_CHECK_EXIT=0 @<sha>`), parsed and compared against the PR
   head the hook/CI knows. A stale or placeholder sha fails.
2. Have the pod run emit a machine-readable artifact keyed by the source
   digest (the sync already computes one) and have the precheck key the
   marker to that digest, so the measurement is bound to the exact tree.
3. Longer-term, move the marker from a free-text PR body into a
   generated record the runner writes and CI reads, removing the
   author-asserted text entirely.

## Rule

A marker is evidence only when it is bound to the exact artifact under
review. Verifying that a status line is present verifies the line, not
the run; the line must carry the head sha (or source digest) and the
check must compare it. A mechanism that cannot tell a measured value
from a placeholder has the same hole whether the language is bash or
Python.
