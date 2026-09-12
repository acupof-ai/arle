# The link gate passed while its whitelist excluded the broken links — coverage widening, 2026-09-13

Date: 2026-09-13. Found while auditing this year's bulk-mechanical commits for
collateral; fixed with the widened link gate in this lane.

## Context

A mechanical doc commit (`631ffd407`, strip commit-hash references) left broken
*sentences* across 34 files, repaired earlier the same day. That damage had no
mechanical detector. Chasing whether other bulk commits had collateral, a
hand-rolled markdown-link scan found 15 broken links. The repo **already had**
a link checker — `check_markdown_links` in `scripts/check_repo_hygiene.py:169`,
whose logic is correct: it parses each markdown link, resolves it relative to
the containing file, and reports a missing target. It had been passing the whole
time.

The one call site passed it `PUBLIC_CHECK_FILES`, a hand-maintained union of
three lists (`PUBLIC_DOCS + GOVERNANCE_DOCS + TEMPLATE_DOCS`). Every broken
link lived in a directory those lists omitted: `docs/experience/**`,
`docs/design/**`, `benchmarks/**`, `examples/**`. The green result was truthful
about the 15 files it measured and silent about the other ~600.

## What changed

- The link gate now runs over every tracked `.md`/`.markdown` instead of the
  curated list (`list_tracked_markdown_docs`), feeding the unchanged
  `check_markdown_links`. No second checker was added beside the working one.
- Exclusions are by path, not by link text: `*/vendor/*` (upstream changelogs
  whose links do not resolve in this checkout), `.claude/projects/*` (agent
  session-memory snapshot, not repo docs), `docs/experience/archived/*` (the
  sealed-archive gate forbids modifying those hash-frozen entries), and
  machine-local home targets (a leading `~/`, or an absolute path under the
  current user's home directory) which can never resolve in-repo.
- A `~/` target is not skipped: it resolves only on its author's machine, so it
  is reported as a `machine-local home link` (the cause is not a missing
  target). The absolute-home form is skipped because the literal is already
  banned tree-wide by `REPO_WIDE_DISALLOWED_MARKERS`; the tilde form matched no
  other rule and previously passed silently, so two such links in
  `crates/autograd/AGENTS.md` were converted to plain backtick references.

The archive exclusion knowingly hides one broken link: the sealed
`docs/experience/archived/wins/2026-05-27-int4-kv-kivi-poc.md` carries a
one-`../`-short `quantization.md` reference, but fixing it would break the
manifest hash seal. That is a deliberate trade of a real defect for the
archive's immutability, stated here rather than left as an unremarked gap in a
green gate.
- The first widened run was red with 20 broken repo-internal links, all then
  fixed: the 11 missing-`../` links in `docs/design/what-breaks.md` plus its
  corpus pointer and one errors→wins path, two `iso_merger.py` 404s (script
  deleted 2026-09-11; links dropped, names kept in backticks), four
  `benchmarks/README.md` and one `examples/opd/README.md` links to entries the
  2026-07-22 corpus purge deleted (same treatment).

## Rule — the recurring shape

This is the third instance in one day of one defect:

1. The gate registry validated that a gate *file exists* but never that
   anything invokes it.
2. The pre-push hook ran a hardcoded list of shell tests, so a new test file
   never executed until someone remembered to add it.
3. The link checker's input was a curated file whitelist.

In all three the **mechanism was sound and the input set was a whitelist that
excluded the failures.** A check's green result means only "clean over the set I
looked at"; when that set is hand-maintained, the directories most likely to
accumulate damage (the large, churny `experience/` corpus) are exactly the ones
left off. Prefer deriving the input set from the tracked-file tree and
excluding the genuinely non-applicable paths explicitly, over enumerating the
paths that should pass.

## Two classes of invisible doc damage — only one is detectable

The broken *sentences* from the hash cleanup (dangling "Commit.", orphaned
clauses) are not broken links and would **not** have been caught by this gate,
before or after widening. A link checker detects a target that does not exist;
it cannot detect prose whose referent was removed. Mechanical bulk edits that
rewrite text still need the source-commit hunk-pair audit (read what was
removed, check whether surviving text still makes sense); there is no grep that
substitutes for reading the paragraph.

## Reproducible history finding (bounded)

Claim: of the 55 non-merge commits since 2026-01-01 that touched ≥40 files
(`git log --since=2026-01-01 --no-merges --shortstat`, filtered on file count),
only two damaged surviving text: the commit-hash cleanup (`631ffd407`, broken
sentences) and the script deletion (`900cc2aa5`, the two `iso_merger.py` 404s).
Method: classify the 55 by signature — huge insertions = vendoring, pure
deletions = corpus pruning, near-equal ± = rename/refactor, small delta =
prose/comment sweep — then for every modify-in-place candidate extract each
removed line that carried the swept token and check whether the surviving
context still reads at HEAD. The other high-count classes (vendored trees,
pegainfer→infer renames, the metric drop, doc-citation purge, comment and
inline-test sweeps, corpus prunes) were clean at their sampled sites. This is a
bounded negative finding, not proof of zero damage outside the ≥40-file set;
it is reproducible by re-running the two commands above.
