# Repo Hygiene Guardrails for Public Docs and Contribution Flow

## Context

- Goal: close the infrastructure gap between "the governance docs exist" and
  "the repository actually enforces them".
- Problems found during the audit:
  - public and governance docs could still drift into private/local-path links
  - PR and issue templates did not require support / stability /
    compatibility / migration context
  - local pre-push and CI did not have a dedicated repository-hygiene gate
  - README and CONTRIBUTING did not expose one obvious command for doc/template
    health

## What Worked

- Added `scripts/check_repo_hygiene.py` as a small, dependency-free guardrail
  for:
  - required public/governance docs
  - public local-link validity
  - private/local path markers in public docs
  - minimum PR / issue template fields
  - tracked junk-file checks
- Wired the check into:
  - `make hygiene`
  - `scripts/pre_push_checks.sh`
  - `.github/workflows/ci.yml`
- Tightened `.github/PULL_REQUEST_TEMPLATE.md` so stability tier, support-matrix
  impact, compatibility, docs, validation, benchmark evidence, and migration
  notes become explicit instead of optional maintainer memory.
- Tightened bug / feature issue templates so reports carry backend, command
  surface, environment, and evidence by default.
- Replaced a public doc reference to a private `.claude/...` memory path in
  `docs/bench-and-trace-spec.md` with self-contained guidance.
- Surfaced the new operator/contributor commands in `README.md`,
  `README.zh-CN.md`, and `CONTRIBUTING.md`.

## Rule

- If a repo-level policy matters, it should be enforced by a small automated
  check, not only described in prose.
- Public docs must never depend on maintainer-local paths or private memory
  files.
- Governance-sensitive PR context should be required by template, not inferred
  during review.
- Repo hygiene / docs / CI template work is bench-exempt when it does not touch
  runtime hot paths; say so explicitly instead of silently skipping benchmarks.
