---
name: find-simplifications
description: Use when asked to find dead code, unused flags, speculative generality, or simplification candidates in ARLE — "what can we delete", "simplification sweep", "flag deletion wave", or auditing a surface for over-built machinery. Turns a broad "find things to simplify" request into evidence-backed deletions or wins/errors entries. Prefer a few well-proven candidates over a pile of thin guesses.
version: 1.0.0
---

# find-simplifications

Evidence-backed simplification sweeps for ARLE. The project deletes in waves
(flag deletion waves 1–3, the 2026-05-18 pivot that removed 167k LOC) — this
skill is the method those waves applied, written down.

## What counts as a strong candidate

A strong candidate removes, folds, or demotes something real, with evidence
that the current design costs more than it buys:

- A flag, config knob, public method, trait method, registry row, or module
  has no production consumer — `crates/*/src`, `src/`, serve paths, and
  loader/config paths are the production corpus.
- Tests, benches, and docs are the only consumers, and the behavior they pin
  is not load-bearing.
- Two mechanisms mirror the same fact (two knobs for one axis, two code paths
  for one shape).
- A feature implements speculative generality with no product owner:
  multi-session, background rosters, mid-turn steering, "for later" seams.
- Hand-rolled code reimplements what a dependency or stdlib already provides,
  and the swap deletes the implementation plus its tests.
- An invariant, rollback path, or special case exists only to protect an
  unused API.

Thin candidates: one dead symbol, a typo, "this looks complex" without
call-site proof. `cargo machete` (nightly) finds unused dependencies; it is
not a substitute for reading call sites.

## Survey

- Start with the largest production-code deltas. Obvious unused symbols are
  the floor, not the sweep — duplicated lifecycle and defensive machinery
  carry the real cost.
- For flags: `crates/cli/src/args.rs` is the registry; trace each flag to
  its serve mapping and its consumption site. A flag whose consumption is
  a doc comment or a test is dead.
- For trait/impl surface: every method every implementation must support but
  no caller uses is a candidate, not just every unused method.
- Use `rg` first: exact symbol, both `.name(` and `name(`, config key, wire
  string. Then read the call sites.

## Prove or reject each candidate

Classify consumers before writing anything:

- **Production** — `crates/*/src`, `src/`, serve/loader/config paths.
- **Non-production** — tests, benches, docs, wins/errors entries, comments.
- **Ambiguous** — examples and scripts that may be product smoke paths.

Reject or downgrade when:

- A production caller exists — the removal is a feature decision, not a
  cleanup, and needs its own proposal.
- The behavior is justified by a wins/errors entry or a hard-won defensive
  pattern, and the new evidence does not beat that reason.
- The removal forces unrelated churn without reducing public surface.
- The idea is correct but tiny — add a targeted `TODO(name):` instead.

## Record the outcome

- Deletions land as a commit with the full chain removed (flag → mapping →
  doc → test), per the no-half-states rule.
- A rejected simplification that was tempting enough to recur lands as a
  wins/errors entry recording why it loses, so it is not re-litigated.
- Inline `TODO(name):` / `FIXME(name):` only for small, local cleanups with
  an actionable next step. No speculative TODOs.

## Validation

- `cargo check` on the lanes the diff touches (cuda,no-cuda typecheck is
  Mac-runnable; metal lane on Apple Silicon).
- `python3 scripts/check_repo_hygiene.py`.
- Correctness gates when the deletion touches inference behavior:
  `scripts/lever_gate.sh`, not just compile-green.
- Report: candidates deleted, candidates rejected with reason, what was
  intentionally excluded.
