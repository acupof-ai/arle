# Repo cleaning machinery adopted from deepseek-harness — 2026-08-24

> Status: Shipped

## Context

The wins/errors cap (790/296) was only a tripwire: archival was a manual
weekly chore with no frozen-history guarantee, the tree had no residue
sweeper, CI actions ran on floating tags, and nothing detected unused
dependencies. deepseek-harness (a 7900-file TypeScript monorepo) runs a
mature cleaning regime; this entry records what was ported and what was
rejected.

## What Worked

Ported (fit):

- **Frozen archive** — `scripts/archive_experience.py` seals wins/errors
  entries into `docs/experience/archived/`, rewrites inbound links, and
  appends sha256 to `archived/manifest.json`. `check_repo_hygiene.py`
  fails on any hash drift, missing sealed file, or unsealed file in the
  archive. Sealed entries never change; corrections land as new live
  entries. The script's `--delete --retarget-to` mode handles the dsh
  consolidation rule: a fully-superseded entry has its unique facts merged
  into the owner, its inbound links retargeted, and is then deleted rather
  than frozen.
- **Residue sweeper** — `scripts/clean_repo.py`, plan-then-delete modeled
  on dsh `clean.ts`: dry-run default, `--apply` to delete, tracked-file
  assertion, realpath boundary, protected roots (`target/`, `models/`,
  `.venv/`, `vendor/`).
- **Archive skill** — `.claude/skills/archive-experience/SKILL.md`:
  oldest-first zero-inbound-reference selection, seal command, freeze
  rules.
- **SHA-pinned actions** — all 22 workflow action refs pinned to
  full-length SHAs with tag comments; the existing monthly+grouped
  dependabot keeps them fresh.
- **cargo-machete** — nightly unused-dependency sweep in
  `fmt-nightly.yml` (renamed to "Nightly Drift Sweep"). First run found
  4 dead deps, all removed: `deepep-sys/libc`, `vulkan-kernels/ash` (the
  vulkan feature goes through vulkan-sys), `infer-cuda/infer-core`,
  `infer-metal/infer-core` (backends depending on the engine was
  architecturally backwards).
- **Loud gate skips** — `lever_gate.sh` skip arms now echo their skip;
  an env-set skip can no longer pass silently.

Rejected (with reason):

- Dependabot — already present (monthly + grouped, post-2026-04-26
  reset; weekly ungrouped once produced 23 open PRs). Kept as-is.
- Issue/PR label policy + ProjectV2 lifecycle — team-scale governance;
  solo repo with no label taxonomy.
- i18n triplet pairing — repo docs are English-only by policy.
- jscpd duplication gate — Rust `-D warnings` dead_code covers the axis.
- run-gates DAG / lefthook staged hooks — linear pre-push is sufficient;
  the pre-push hook was already disabled for slowness.
- "golden" filename ban, sandbox confinement CI — no corresponding
  surface here.

## Rule

Adopt external machinery by mechanism, not by surface: dsh's `clean.ts`
is TypeScript-project-graph-shaped; the port is an allowlist sweeper
because cargo owns `target/`. The durable parts are the invariants —
plan-then-delete, frozen sealed history with a manifest gate, loud
skips, pinned supply chain — not the file layout.
