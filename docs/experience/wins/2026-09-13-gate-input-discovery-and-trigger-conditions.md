# A gate needs both the right input set and a trigger that fires — CI shell tests + docs hygiene

> Status: Shipped

## Context

Two related gates were correct in mechanism and input set but never executed on
the changes they guarded.

1. The CI "Shell Contracts" step hand-listed eight of the nineteen
   `scripts/tests/test_*.sh`. The pre-push hook hand-listed nineteen of
   nineteen. A new test landed in the hook's list only when someone remembered
   to add it there; CI's list had no owner. #423 added a case to
   `test_parity_gpu_batch.sh`; CI reported Shell Contracts SUCCESS without
   running it. The green was truthful about the eight tests and silent about
   the other eleven. The test about skipping shell tests
   (`test_hook_skips_shell_tests.sh`) was one of the enforced eight.
2. `check_repo_hygiene.py` — the markdown link gate widened to every tracked
   markdown file the same day — ran inside `ci.yml`, whose trigger carries
   `paths-ignore: docs/**, **/*.md, memory/**`. A documentation-only pull
   request started no job in that workflow: no link check, no marker bans, no
   experience-entry caps, no ledger checks. #424 (one added errors entry) ran
   only GitGuardian; hygiene passed when run by hand, but nothing would have
   reported a failure.

The input-set audit that found the first gap missed the second because it
audited what each gate reads, not under which event it runs. A correct input
set behind a trigger filter that excludes the relevant events is the same
defect in a different place.

## What Worked

One owner of the test batch. `scripts/run_shell_tests.sh` discovers
`scripts/tests/test_*.sh` with a filesystem glob (not `git ls-files`: the hook
runs fast checks inside a `git archive` snapshot that has no `.git`) and is the
only place the list exists. The pre-push hook and the CI step both call it.

Discovery alone is not enough; three safeguards close the adjacent holes:

- **Per-file skip directives.** A test that cannot run on a platform carries
  `# TEST-SKIP: linux: <reason>` in its own first lines. The skip prints the
  file and the reason, an unknown platform token fails the runner, and an
  unannotated failure is always a failure. The reason lives in the excluded
  file, not in a list elsewhere. Zero skips today; the mechanism is auditable.
- **Relevance is shared, not duplicated.** The "which changes are worth the
  batch" regex lives once in the runner; the hook sources it and passes its
  push-range file list in via `SHELL_TEST_CHANGED_FILES`, CI derives the list
  from its PR/push base. An empty or unresolvable list runs everything —
  only a list successfully computed with no relevant path may skip.
- **Count line and retained logs.** Every run ends with
  `N tests, M ran, K skipped, J failed`, so a growing skip set is visible.
  Each test's full output lands in a per-run directory under
  `${TMPDIR}/arle-shell-tests/` (symlinked from `latest/`); the hook
  interleaves batch output with cargo, so after a failure the test's own log
  must remain readable.

Hygiene moved to its own unfiltered workflow. `.github/workflows/hygiene.yml`
runs hygiene plus `--selftest` on every push and pull request, no path filter.
The job was deleted from `ci.yml`; the compile lanes keep their
`paths-ignore`, so a docs change pays for the seconds-long hygiene gate and not
for 10-15 minutes of compile CI. The inverse-filter alternative was rejected:
duplicating hygiene across two workflows is a seam, and a PR touching both code
and docs would run it twice.

Linux compatibility was measured, not assumed. Every test was run on a real
`ubuntu-latest` runner (throwaway probe workflow, deleted after): 17 of 19
passed, two failed for one test-side reason — single-quoted EREs containing
`\t`. Local `grep` on this box is a shell function around ugrep, which
interprets `\t` as a tab; GNU grep in CI treats it as a literal `t`. The mock
TSVs were correct; the quoting was not. Converted to ANSI-C `$'…'` quoting
(which two lines already used). The general form: local verification with a
shell-function grep differs from CI in regex semantics and in the file set,
because ugrep's default flags silently exclude ignored and hidden files.

Hardcoded test-server ports were fixed in the same owner. Four tests bound
fixed loopback ports (19400, 19941, 19931-34, 19951-52). Sessions sharing this
box run the hook concurrently; overlapping binds collided and surfaced as test
failures. All four now bind port 0 and read the assigned port back.

## Rule

Audit a gate on two axes: the set of inputs it reads, and the set of events
on which it runs. A hand-maintained list of either drifts to exclude the
failures. Discover the input; make exclusion a visible, in-file, reason-bearing
act; and when an unresolvable boundary (diff base, platform, dependency)
threatens a skip, run the full set — fail toward coverage, never toward
silence.

Two findings left deliberately unfixed in this lane:

- Four test traps use `kill "${SRV_PID:-0}"`; an unset pid becomes `kill 0`,
  signaling the whole process group. It only fires when setup fails early
  (e.g. a missing Python dependency), which is exactly when it produces the
  most confusing signal. The runner's `setsid` containment is the structural
  mitigation; the traps should take a pid-or-noop form in a later lane.
- Branch protection on `main` configures no required status checks. Every gate
  here passes or fails only because someone reads the check page; nothing
  enforces a green merge. A gate's existence is not a gate's authority.

The 0-gap hand lists from the input-set audit — CI's four named
`scripts/tests/*.py` pytest invocations — were confirmed as drift channels and
left out of this lane's scope.

Related: [2026-09-13-link-gate-input-whitelist-excluded-the-broken-links](2026-09-13-link-gate-input-whitelist-excluded-the-broken-links.md)
(input-set half of the same class, earlier the same day).
