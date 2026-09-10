# Freshness assertion false-positive on ANSI-decorated cargo output

Date: 2026-09-10. Lane: `lane/step1b` (Step 1b, host KV pool single writer).

## Context

PR #275 added per-step freshness assertions to `scripts/pre_push_checks.sh`:
a cargo step that reports a changed crate `Fresh` fails the push, because a
shared target dir lets one lane's build mask another's. Pushing the Step 1b
branch (which changes `infer-plan`, `infer-core`, `infer-seam`) failed with:

```
[pre-push] cargo test group 2 reported infer-plan Fresh while this push changes it
```

The printed remedy (`CARGO_TARGET_DIR=... cargo clean -p infer-plan`) did not
help: after a clean, test group 1 recompiles the crate and group 2 reports it
`Fresh` again. Two clean-and-retry cycles failed the same way, and a full
`cargo clean` (9.2 GiB) of the hook target failed the same way.

## Root Cause

The hook exports `CARGO_TERM_COLOR=always` (line 67) so its streamed logs keep
their color. Cargo then decorates every status line:

```
^[[1m^[[92m   Compiling^[[0m infer-plan v0.5.8 (...)
```

`assert_step_rebuilt` greps the step output for
`(Checking|Compiling) ${crate}( |$)`. The ANSI reset sits between
`Compiling` and the crate name, so the pattern never matches: a crate the
step actually rebuilt reads as `Fresh`, and every push touching a test-group
crate fails. The CUDA lint step alone set `CARGO_TERM_COLOR=never` for its
capture, which is why only the test-group path was broken and why the lint
path's red/green test stayed green.

The assertion's test (`test_hook_cuda_lint_freshness.sh`) mocked cargo with
plain, undecorated output, so the test suite never exercised color and the
false positive shipped.

## Fix

PR #282: `assert_step_rebuilt` strips ANSI codes from the captured output
before grepping. The test gains a color world — the mock emits
ANSI-decorated lines and the push must still pass — plus the reverse control
(the unpatched hook rejects that world, reproducing the failure).

## Rule

When a hook greps cargo's human-readable output, strip ANSI first or set
`CARGO_TERM_COLOR=never` for that step; an export of `CARGO_TERM_COLOR=always`
and an undecorated grep cannot coexist. A mock for such a test must emit the
same decoration the real binary does — a plain-output mock proves nothing
about a colored world.

Both reverse controls in this incident chain were written by the gate's
author, and both holes were found by someone else constructing a shape the
author did not imagine: `Option::take()` on the hygiene gate, ANSI
decoration on this one. The control's world is best constructed by someone
other than the gate's author; an author-built control replays the author's
blind spots.
