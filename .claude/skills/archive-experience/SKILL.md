---
name: archive-experience
description: Use this skill when archiving docs/experience/wins or docs/experience/errors entries — when the entry count approaches the hygiene cap (wins 790, errors 296), during the weekly hygiene sweep, or when an entry is fully superseded. It enforces future-value classification (not age or quota), oldest-first zero-inbound-reference discovery, the frozen-archive seal via scripts/archive_experience.py, and the rule that sealed entries never change.
version: 1.1.0
---

# archive-experience

Calibrated workflow for retiring `docs/experience/{wins,errors}/` entries into
the frozen archive. Modeled on deepseek-harness's `dsh-archive-agent-notes`
skill: archival is a seal, not a deletion, and the decision criterion is
future decision value, not age or word count.

## When to run

- The weekly hygiene sweep (see CHANGELOG section of AGENTS.md).
- `scripts/check_repo_hygiene.py` warns that wins/errors approach the cap.
  The cap is a hard gate, so when it approaches, archival is forced — but
  the *selection* is still future-value-ranked, not blind oldest-first.
- An entry is fully superseded by a newer one on the same topic.

## Classification (the decision)

Judge every candidate semantically. Age and zero inbound references are
discovery aids, never the criterion.

- **Archive** when the shipped decision is complete and the body is unlikely
  to guide future work: one-off bench snapshots whose numbers are superseded,
  a narrow workaround for a since-fixed bug, process history whose current
  behavior is obvious from the code, an entry fully superseded by a newer
  one on the same axis.
- **Keep** when the entry still carries future leverage: a binding constraint
  or roofline number still quoted in planning, a negative result that
  prevents re-litigating a tempting path, a reintroduction condition, a
  measured bound a future optimization must beat.
- **Supersession check when adding an entry**: every new wins/errors entry
  triggers a scoped look at existing entries on the same decision or axis.
  Fully superseded entries get sealed in the same change; partial
  supersessions stay live and cross-linked.

## Selection procedure

1. List entries oldest first: `git ls-files 'docs/experience/wins/*.md' | sort`.
2. Check inbound references: `git grep -l -F '<path>' -- .` — referenced
   entries are usually still load-bearing; start with zero-reference ones.
3. Classify each candidate by the rules above. Seal in batches of 5–20 when
   the cap forces it; otherwise seal entries as they become superseded.

## Seal

```bash
python3 scripts/archive_experience.py --write <entry.md>...
python3 scripts/check_repo_hygiene.py
```

The script moves each entry to `docs/experience/archived/<wins|errors>/`,
rewrites inbound links to the archived path, and appends the entry's sha256
to `docs/experience/archived/manifest.json`. Default mode is a dry-run plan.
The entry's content is never edited — sealed means byte-identical.

## Freeze rules

- Sealed entries are frozen forever: no edit, no reformat, no move, no delete.
  The hygiene checker fails on any manifest hash drift or missing sealed file.
- Corrections to a sealed entry land as a new live entry, never as an edit to
  the sealed one.
- Do not verify or repair links *out of* a sealed entry; it is a historical
  snapshot, not current authority.

## Commit

Seal + link rewrites + manifest update are one commit:
`chore(docs): archive N experience entries`.
Report: entries sealed, entries kept as still-load-bearing, borderline cases
with the reason each went the way it did.
