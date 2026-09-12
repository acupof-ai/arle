# The lane precheck rule-set runs exactly once and never on a direct-to-main commit

Date: 2026-09-13. Surfaced from the `/host` pod-path that reached main in a
peer's errors entry; quantified over two weeks of history. Entry only; the fix
is separate lanes. Read-only audit, no code changed here.

## Context

`scripts/lane_pr_precheck.py` carries the repo's author-facing rule-set: bench
entries, comment style, machine paths, dead shas, build/CUDA/clippy markers,
parity-gate registration. It is invoked from exactly one place —
`scripts/lane.sh pr`, against `merge-base origin/main...HEAD` with the PR body.

The pre-push hook does not run it against the diff:
`scripts/pre_push_checks.sh:230` runs `lane_pr_precheck.py --selftest`, which
is the checker's own unit tests, not the checker over the pushed range. CI
(`.github/workflows/ci.yml:55`) runs `check_repo_hygiene.py`, which covers
none of these rules except a repo-wide `<home>` ban, and `paths-ignore`
skips the whole workflow for docs-only commits.

The ledger convention routes CHANGELOG paragraphs, agenda entries, doc indexes
and experience entries straight to main because a PR would make the record lag.
That reasoning is sound; what was not intended is that the whole rule-set goes
with them.

## Measurement — two weeks of first-parent history

Window: `git log --first-parent` since 2026-08-30. Non-merge first-parent
commits total **177**, split by whether the rule-set could have run:

- **19 squash-merged PRs** — subject ends in `(#NNN)`, merged through
  `lane.sh pr`; the precheck ran on these.
- **158 true direct-to-main commits** — no PR trailer; the precheck never saw
  them, regardless of what files they touched.

The numbers below count only the 158. Each was tested by importing the
checker and running its own functions over that commit's added lines (parent
→ commit), not by grepping for the banned strings. To avoid this entry itself
emitting the banned literals, write the path roots as placeholders:
`<home>` = the macOS user-home prefix, `<root>`/`<host>` = the two container
trees, `<data>` = the data volume root, `<mnt>` = the mount root.

Positive control: the checker's abs-path regex matches the known offending
line (a parenthesized reference to `<host>/p1-results/results.tsv`) verbatim
on the host segment, so a count of zero would have been a failed measurement,
not evidence of cleanliness.

## Rule table and direct-to-main violations

| Rule | Scope | Other enforcement on a direct commit | Distinct direct commits that would fail |
|---|---|---|---|
| bench-entry | runtime path (`crates/*/`, `scripts/bench_*/`, `src/`) per commit needs an experience entry or "Bench-entry exemption" in the body | none — hygiene caps/seals entries but never ties one to a runtime commit | 2 |
| comment-ref | added comment lines in `.rs/.cu/.h/.cc/.cpp` and `.py/.sh/.toml/.yml`; no `#NNN`, no 7+ hex SHA; dated experience entries exempt | none | 1 (a false positive, below) |
| abs-path | added lines in any file; bans the `<home>` `<root>` `<host>` `<data>` `<mnt>` prefixes; only carve-out is `${VAR:-…}` in non-comment `.sh` | hygiene bans only the `<home>` prefix repo-wide | 26 verbatim; **4 genuinely new prose leaks** |
| build-exit | an added `examples/`/`benches/` file needs `BUILD_EXIT=0` in the PR body | none; the body does not exist on a push | 0 |
| cuda-check / clippy-exit | changed path in a no-cuda-gated crate or under `cuda-kernels/` needs pod real-nvcc markers in the PR body | the hook runs Mac no-cuda clippy, not the pod gate | 4 structurally-unsatisfiable |
| gate-registry | a new `infer-cuda/examples/*_parity.rs` taking `--negative-control` must be in `registry.toml` and print `NEGATIVE CONTROL OK` | none | 0 |
| dead-sha | an added backticked 7-12 hex sha in `docs/**.md` must resolve on main; upstream-project cue exempt | none | 7 |

The 26 verbatim abs-path hits are not 26 leaks. Categorized:

- 4 distinct commits add genuinely **new** machine/container paths in committed
  `docs/**.md` prose — the true class: the peer `p1-results` entry (since
  removed), plus new entries citing `<host>` build and data paths (a container
  build tree, a `<mnt>/<data>` model dir, an agent corpus file, a shared build
  root), and one docs-cleanup commit that introduced three new such lines while
  editing.
- 19 commits are `docs/agenda.jsonl` / `docs/experience/prereg.jsonl`
  run-provenance records (`<data>`-volume model paths) — the path is the point
  of the record.
- 3 are machinery: `lane.sh` help text naming the `<host>` tree, and the
  hygiene selftest fixture that intentionally writes a `<home>/…` string.
- 1 CHANGELOG prose line, 1 committed benchmark snapshot JSON with a
  `<home>/.../models/…` path.

The 1 comment-ref hit is the dependabot bump editing `.github/` YAML whose
comments reference upstream PR numbers — a false positive, not a violation.
The 4 cuda/clippy-marker commits (`bf4a81ba8`, `0e75276c2`, `232ce41e7`,
`bb0d623c1`) include a `perf(qwen35)` kernel change with no bench entry and
no possible pod marker, because a direct push cannot carry a PR body. That one
is a policy/escalation question, not a tooling defect.

## Root cause

Two enforcement mechanisms with disjoint coverage. The lane path enforces the
rule-set but only on code that takes a PR; the direct-to-main path — used on
purpose for the ledger — runs only a hook whose fast block invokes the
checker as `--selftest`, so the mechanism appears in a green push ("the
precheck ran") while the actual rules never execute against the bytes. This is
the same shape as the rest of the week: a check whose machinery runs but whose
input is not the thing under test, and a documented rule believed enforced by
nobody.

## Fix direction (separate lanes)

Split the rules by whether they need a PR body:

- The five **body-less content rules** (bench-entry, comment-ref, abs-path,
  gate-registry, dead-sha) move into the pre-push hook run against the **actual
  pushed range** (`remote_sha..local_sha`; new branch → merge-base), which the
  hook already parses. A main push is a push to `refs/heads/main`, so this
  closes the hole without touching the ledger's direct-commit workflow beyond
  the rules that genuinely apply.
- The three **body-marker rules** (`BUILD/CUDA/CLIPPY_EXIT`) stay in the PR
  precheck. There is no body on a push; forcing one there would block ledger
  work and provide no signal. Whether runtime code may land direct to main is
  branch protection, not a hook.

The range computation must be one shared function both callers use with their
own endpoints — a duplicate that agrees is cheap (seconds), two implementations
that pick different bases is the bug being prevented. The pushed-range base is
right for a gate blocking bytes; merge-base is right for a pre-merge summary,
so the two contexts must not share one formula.

Exemptions, narrow and by name:

- `docs/agenda.jsonl` and `docs/experience/prereg.jsonl` exempt from abs-path
  only — they are run-provenance records whose purpose is the location. Not a
  `docs/**` glob. This exemption does NOT cover CHANGELOG prose, which must
  still be blocked.
- `.github/**` exempt from comment-ref (upstream YAML PR references).
- `scripts/lane.sh` and `scripts/check_repo_hygiene.py` exempt from abs-path by
  name (help text and the selftest fixture). If that whitelist ever needs a
  fourth named entry, the rule is wrong and needs rethinking, not another name.

## Rule

A rule that runs only on the PR path does not protect a branch that accepts
direct commits; enforce body-less content rules on the push itself, against the
exact bytes being pushed. Do not infer "checked" from a hook that runs the
checker's self-tests — selftests prove the checker can fail, they do not check
the diff. When quantifying a gap, run the real checker per commit and prove with
a positive control that it would have caught the known instance; otherwise a
zero is a measurement failure.
