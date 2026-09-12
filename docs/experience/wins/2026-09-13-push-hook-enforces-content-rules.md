# Pre-push hook now enforces the body-less content rules over the pushed range

Date: 2026-09-13. Follow-up to
`2026-09-13-lane-precheck-bypassed-by-direct-main.md`. Closes the enforcement
gap; no device involvement (shell + python only).

## Context

The five body-less content rules (bench-entry, comment PR/SHA refs, machine
paths, parity-gate registration, dead docs shas) ran only in
`lane_pr_precheck.py` from `lane.sh pr`, so a direct-to-main push bypassed all
of them. The pre-push hook invoked the checker only as `--selftest`. This lane
moves those five into the hook while keeping the three PR-body marker rules
(`BUILD/CUDA/CLIPPY_EXIT`) PR-only — a push has no body to carry a marker.

## What changed

- `lane_pr_precheck.py` exposes one range-parameterized content path:
  `added_lines(repo, base, head)` / `changed_files(repo, base, head)` compute
  the range once; `run_content(repo, base, head)` runs the five body-less
  rules; `run_pr` calls it with merge-base..HEAD plus the marker rules; the new
  CLI mode `--push-content BASE..HEAD` calls `run_push` over the exact pushed
  endpoints. The PR path and the push path share one range function with
  different endpoints — they do not each compute a base.
- `pre_push_checks.sh` already resolves a push range from pre-push stdin. The
  loop now processes **every** ref update (`continue`, not `break`): a push like
  `git push origin :old new` lists a deletion first, and the old unconditional
  `break` stopped after it, silently passing the bytes-carrying branch. Each
  non-deletion ref is gated on its own `remote..local` (new branch → merge-base)
  endpoints before the snapshot/build work, so a docs-only violation fails in
  milliseconds without the cargo lock and cannot hide behind another ref. The
  cargo/shell decisions use the union of every ref's changed files. `#430`'s
  `run_shell_tests.sh` and the shared relevance filter are reused unchanged.
- Named exemptions, by exact path only; these four are the complete set, and a
  fifth means the rule needs rethinking rather than another row:
  - abs-path exempts `docs/agenda.jsonl` and
    `docs/experience/prereg.jsonl` (run-provenance records whose purpose is the
    location), `scripts/lane.sh` and `scripts/check_repo_hygiene.py` (they
    print/test for the literal strings).
  - comment-ref exempts `.github/` (upstream YAML PR references).
  - CHANGELOG prose is explicitly NOT exempt: a ledger paragraph is exactly
    the class the rule must catch.

## Red proof

`scripts/tests/test_prepush_content_rules.sh` drives the real hook in a
nested-fixture repo (docs-only, cargo skipped) with nine cases, each asserting
the rule's own rejection/acceptance text:

- five pushes rejected, one per rule: machine path in a CHANGELOG line
  (`abs-path: CHANGELOG.md`), `#123` in a Rust comment (`comment-ref:`), a
  backticked unresolvable sha in docs (`dead-sha:`), an unregistered
  `infer-cuda/examples/*_parity.rs` taking `--negative-control`
  (`gate-registry:`), and a runtime change with no experience entry
  (`bench-entry:`);
- the exemption pair: the same machine path in `docs/agenda.jsonl` is accepted
  while the CHANGELOG line is rejected, and an upstream PR reference under
  `.github/` is accepted;
- multi-ref: when the first stdin line is a branch deletion and the second
  carries a bad branch, the second is still rejected (the old `break` would
  have masked it);
- deletion-only and empty stdin: the shell-test runner (which defines
  `RELEVANCE_RE`) is sourced **once before the ref loop**, because under
  `set -u` a loop whose body never runs would otherwise read an unbound
  variable and abort. A deletion-only push and a no-stdin manual run both
  reach the skip decision cleanly.

The same fixture run against the pre-lane hook failed at the first case with
"hook ACCEPTED a push it must reject" — the old hook had no content gate. The
new test is auto-discovered by `run_shell_tests.sh`'s `test_*.sh` glob. The
checker's own selftest and the existing nested-snapshot / shell-test-skip
fixtures still pass.

## Rule

A rule that must also protect direct-to-main commits belongs on the push,
gated on the exact bytes being pushed; rules whose evidence is a PR body stay
on the PR path. Two enforcement entry points may run the same checks (a
duplicate that agrees is cheap), but they must share one range computation so
they cannot disagree about the base.
