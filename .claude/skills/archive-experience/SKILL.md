---
name: archive-experience
description: Use this skill when archiving docs/experience/wins or docs/experience/errors entries — when the entry count approaches the hygiene cap (wins 790, errors 296), during the weekly hygiene sweep, or when an entry is fully superseded. It enforces oldest-first zero-inbound-reference selection, the frozen-archive seal via scripts/archive_experience.py, and the rule that sealed entries never change.
version: 1.0.0
---

# archive-experience

Calibrated workflow for retiring `docs/experience/{wins,errors}/` entries into
the frozen archive. Modeled on deepseek-harness's `dsh-archive-agent-notes`
skill: archival is a seal, not a deletion, and the selection rule is explicit
so the cap never becomes a panic.

## When to run

- The weekly hygiene sweep (see CHANGELOG section of AGENTS.md).
- `scripts/check_repo_hygiene.py` warns that wins/errors approach the cap.
- An entry is fully superseded by a newer one on the same topic.

## Selection rule

1. List entries oldest first: `git ls-files docs/experience/wins '*.md' | sort`.
2. Skip `TEMPLATE-bench.md` and any entry with inbound references:
   `git grep -l -F '<path>' -- .` — the seal script rewrites inbound links
   automatically, but referenced entries are usually still load-bearing;
   archive zero-reference entries first.
3. Archive in batches of 5–20 until the count is comfortably below the cap.

## Seal

```bash
python3 scripts/archive_experience.py --write <entry.md>...
python3 scripts/check_repo_hygiene.py
```

The script moves each entry to `docs/experience/archived/<wins|errors>/`,
rewrites inbound links to the archived path, and appends the entry's sha256 to
`docs/experience/archived/manifest.json`. Default mode is a dry-run plan.

## Freeze rules

- Sealed entries are frozen forever: no edit, no reformat, no move, no delete.
  The hygiene checker fails on any manifest hash drift or missing sealed file.
- Superseded entries are archived, not deleted — the archive is the record of
  what was tried. Consolidation (merging two entries into one) happens while
  both are still in the live tree, before sealing.
- Corrections to a sealed entry land as a new live entry, never as an edit to
  the sealed one.

## Commit

Seal + link rewrites + manifest update are one commit:
`chore(docs): archive N experience entries`.
