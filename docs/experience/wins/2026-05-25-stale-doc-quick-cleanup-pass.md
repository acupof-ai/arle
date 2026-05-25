# Stale-Doc Quick Cleanup Pass

Related: `docs/projects/2026-05-24-opd-mainline-task-backlog.md` T19 and the
user directive 2026-05-25 "删除所有的陈旧的文档和无用的代码 需要的时候再加".

## Context

After 24h of execution since the 2026-05-18 OPD-only pivot, several plan and
project docs still carried claims like "OPD substrate landing next milestone"
or "OPD will host this when substrate lands". Substrate actually landed on
2026-05-24 (`14c3be9`, `crates/cli/src/train_cli.rs::run_opd_from_dirs`), so
those claims were factually stale.

The T19 backlog item licenses codex to do a full scan-classify-prune pass once
T18 finishes. This entry covers the immediate doc-only fixes done by Claude in
parallel — the easy SAFE_EDIT items.

## What Was Edited (not deleted — these docs still have historical value)

| Doc | Stale claim | Fix |
|---|---|---|
| `docs/projects/2026-05-18-opd-only-pivot.md` (commit `2a291e1`) | "substrate landing next milestone"; CLI dispatch keeps `arle train test` stub | Banner pointing at actual landing (`14c3be9`, `run_opd_from_dirs`); T3 prune killed `train test` (`81842cc`) |
| `docs/plans/train-eval-infer-dx-v1.md` (commit `c3e0f2a`) | "...will host the OPD command's DX when the substrate lands" | Substrate landed 2026-05-24; cross-link P5 5k cycle wins entry |
| `docs/plans/train-observability-v1.md` (commit `c3e0f2a`) | "reused by `arle train opd --serve` once the OPD substrate lands" | OPD CLI shipped as one-shot runner; `--serve` mode is separate scope not yet licensed (cross-link support-matrix §5a) |
| `docs/plans/2026-05-05-deepseek-v4-small-substrate.md` (commit `1f81b71`) | Whole plan still readable as if pretrain-dsv4 is active | Top banner: §0a, §0b, §1-5 RETIRED with OPD-pivot + T17 feasibility KILL; §6 (Runtime Adaptation kernels) remains active ROADMAP P0 |
| `docs/support-matrix.md` (commits `b2c0348`, `e3a0ce0`, earlier in session) | "OPD substrate landing"; "`arle train pretrain-dsv4` seeds from..."; "OPD wiring pending" | OPD = Supported (Beta), `pretrain-dsv4` killed in OPD pivot, `--serve` scope clarified |

## What Was NOT Touched

Per `docs/bench-and-trace-spec.md` §9 + project discipline:

- All `docs/experience/wins/` and `docs/experience/errors/` entries are immutable
  historical record. Even stale ones describe past state correctly.
- `bench-output/` 2026-05-19/2026-05-20 CPU OPD experiments are historical
  bench artifacts. Not stale; they document what was true then.
- `runs/` is local training/checkpoint output. Per the T8 audit, it's
  user-owned data (not Claude's territory to prune).
- `dsv4-small-repro.md` already has its own 2026-05-18 retirement banner;
  no new edit needed.
- `bench-matrix-design-2026-04-29.md` and `cutlass-sm89-fp8-template-found...`
  have generic "will land as ..." phrasing but in research/planning context;
  not actionable.

## What Codex T19 Should Pick Up After T18

The harder, more code-side passes deferred to T19:

- Dead code (`cargo clippy --workspace -- -D warnings` with no `#[allow]` —
  unused `pub` items, dead branches that survived T3 prune).
- `TODO`/`FIXME` comments without actionable item; either action or delete.
- Stale `#[cfg(feature = "x")]` gates with no callers.
- `bench-output/` artifacts that no current wins/errors entry references AND
  are older than 30 days — codex evaluates removability.
- Docs with subjects fully superseded by newer plans (codex's scan may surface
  more than my quick pass).

## Rule

When a plan or project doc's "next milestone" claim becomes false, **edit the
claim in place** — don't delete the doc. The historical context (why we chose
this path, what we tried) is still valuable. Add a dated status banner pointing
at the actual landing commit + any cross-links. Preserve original prose so the
"plan vs reality" delta is readable.

## Bench Status

Pure doc edits across 5 files, no runtime/code/script touched. Bench-exempt
per CLAUDE.md §Benchmarks.
