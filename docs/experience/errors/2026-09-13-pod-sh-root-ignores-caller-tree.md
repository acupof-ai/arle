# pod.sh derives ROOT from its own path, so a lane calling the main copy syncs the main checkout

Date: 2026-09-13. Surfaced firsthand on the gauge-presence-gates lane; entry
only, not fixed in that lane.

## Context

The pod workflow is one source tree per lane; `scripts/pod.sh sync` ships the
caller's current lane to a matching remote tree. Invoked from a lane, I ran the
script via an absolute path pointing at the main checkout's copy:

```sh
<main-checkout>/scripts/pod.sh sync --full
```

The sync reported success and printed a head. The synced tree was main, not the
lane: it reported the current main-tip head, while the lane I intended to ship
was a different head. The remote build then compiled the wrong tree. No error
was raised; the command did what its resolved ROOT pointed at.

## Root Cause

`pod.sh:18` computes the source tree from the script's own location:

```sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
```

It ignores the caller's working directory. Calling the main checkout's copy
from a lane therefore makes `ROOT` the main checkout regardless of `$PWD`.

This is the same defect that was fixed in `lane.sh` (#411), where the identical
line had caused physical damage. Fixing it there did not surface the second
copy in `pod.sh` — a same-defect-second-instance miss, not a new bug.

## Fix direction (deliberately not the lane.sh fix)

`lane.sh` was corrected to derive ROOT from git's common directory
(`--git-common-dir`), which always resolves to the main checkout. That is right
for lane management, because lanes hang off the main checkout. Copying it into
`pod.sh` would cement this bug: every pod invocation from a lane would sync the
main checkout by design.

`pod.sh` needs the tree the caller is in. Preferred shape: refuse when the
script's resolved tree and the caller's tree differ — print both and exit,
rather than silently syncing one of them. A silent sync of the wrong tree cost a
build cycle and nearly a marker claimed against the wrong source; an upfront
error costs nothing. Deriving ROOT from `$PWD`/`git rev-parse --show-toplevel`
is the alternative, but the explicit mismatch refusal is safer because it
catches every confused invocation instead of guessing.

## Rule

A script that acts on "the current tree" must bind that tree to the caller's
location, not the script's; and when the two plausibly disagree it must fail
loudly. When a same-shape bug is fixed in one of two sibling scripts, grep for
the second instance before closing it.
