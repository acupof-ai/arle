# PR marker rules now run in CI and bind to the head sha — marker form, 2026-09-13

## Context

The three PR-body marker rules (`BUILD_EXIT`, `CUDA_CHECK_EXIT`, `CLIPPY_EXIT`)
ran from exactly one place: `lane.sh pr`. A pull request opened with `gh pr
create` or the web UI never ran them — #431, #433 and #436 were all merged in
that state. The companion entry `2026-09-13-pr-marker-not-bound-to-head.md`
records the second half: even on the one path that ran, `regex.search(body)`
verified that a `KEY=0` line existed, not that the number was measured at the
reviewed head or measured at all.

## Fix

`scripts/lane_pr_precheck.py`:

- New `--ci GITHUB_EVENT_PATH` mode. A `pr-markers` job in
  `.github/workflows/hygiene.yml` (no path filter, pull_request only) runs it
  with the checkout fetched to depth 0. Head sha, base sha and the PR body come
  from the event payload, so it works for fork PRs (the body is not in local
  refs) and binds to the exact head GitHub reviews, not the synthetic merge
  commit a pull_request checkout makes.
- Markers bind to the head: `KEY=0 @<7+ hex prefix>`. The prefix is compared
  against the one head the check knows; seven hex is unambiguous there and is
  the form `gh` prints, so a full 40 is not required.
- Trailing text is admitted on the marker line (`BUILD_EXIT=0 cargo build …`,
  or `BUILD_EXIT=0 @<sha> cargo build …`). The old anchored regex forced the
  value onto a line by itself, which punished recording the command that
  produced the number — the one fact that separates a measured marker from a
  manufactured one.
- A missing marker and a present-but-malformed one give different messages.
  The stale/attribution messages name the sha found and the head wanted and say
  to re-measure after the last push rather than rebuild.
- A diff endpoint the clone does not have (a behind-local remote tip after a
  forge-side merge) triggers one quiet `git fetch origin` and a retry; if the
  object is still absent the check refuses cleanly, naming the range and the
  fetch recovery, instead of dumping a raw `CalledProcessError` traceback that
  read like a content-rule failure. Endpoints that are present but whose diff
  genuinely fails are reported as a real git error, the other branch of the
  same handler. A range that cannot be read refuses; it does not pass.

Local `lane.sh pr` now also binds to its HEAD when `origin/main` is present,
catching a stale marker before push; CI is the binding gate regardless.

## What the sha does and does not prove

The prefix makes a stale measurement fail: a number taken before the final push
no longer satisfies. It does **not** prove the command ran, or that it was the
prescribed command. #436 carried an honest, correct-sha marker measured from
`-p infer-api --lib` with `no-cuda` — the wrong command for a rule that names a
pod run without `no-cuda`. Closing that needs the marker keyed to the source
digest the pod sync computes (option 2 in the companion entry); the CI
enforcement here is the necessary precondition, not that proof.

Rule coverage note preserved: rule 7 (`clippy --workspace --all-targets`)
covers an examples-only diff; rule 5 (`cargo check --features cuda,nccl`) does
not compile examples without `--examples`, so a `CUDA_CHECK_EXIT` green on an
examples-only diff reports on untouched code.

## Rule

A check that verifies a status line exists verifies the line, not the run, and
a check that runs on only an author-chosen path is opted out of by choosing a
different client. Bind evidence to the reviewed artifact and run the check from
the forge (CI), not from a helper the author can bypass. When relaxing a
pattern to admit useful annotation, keep a positive control that the
near-miss value still refuses (`=1`, `=01`, `=0-ish`); a boundary check that
only tests the exact good value silently admits the suffix it meant to reject.
