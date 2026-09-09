# A step recorder added to the pre-push hook made every push fail

## Context

The hook's fast checks run in a background block, so a failure surfaced only as
the exit status of `wait` and git printed `failed to push some refs`. Two lanes
read that as a transport error the same day. The fix was to record the step in
flight and name it on failure:

    FAST_STEP="${STAGE_ROOT}/fast-step"
    run_fast() { printf '%s\n' "$*" > "${FAST_STEP}"; run "$@"; }

## Root cause

`STAGE_ROOT` is the staging directory for the HEAD archive. It does not survive
the snapshot refresh, so by the time the background block ran, the path was
gone. The `printf` failed, and under `set -e` that killed the whole fast-checks
block — every push, on every lane, including pushes whose checks would all have
passed. The reported failure was `parallel fast-checks FAILED at: unknown step`,
which is the recorder reporting that it could not record.

Two properties combined. The recorder wrote to a location whose lifetime was
shorter than the thing it was recording. And its write was on the critical
path: nothing separated "the diagnostic failed" from "the check failed".

## Fix

Its own temp file, removed in `cleanup`, and a write that cannot fail the run:

    FAST_STEP="$(mktemp "${TMPDIR:-/tmp}/arle-pre-push-faststep.XXXXXX")"
    run_fast() { printf '%s\n' "$*" > "${FAST_STEP}" 2>/dev/null || true; run "$@"; }

Both arms exercised before landing: an injected failing step is named
(`FAILED at: bash -c exit 3`), and an unwritable `FAST_STEP` still lets the
block pass.

## Rule

A diagnostic must not be able to fail what it diagnoses. Its writes are
non-fatal, and its storage outlives the thing it observes. Test the arm where
the diagnostic itself is broken, not only the arm where it reports correctly —
otherwise the first thing the new code proves is that it can take down the gate.
