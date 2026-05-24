# T3 Non-Mainline Prune — Retired Train Sample Corpus

Related: `docs/projects/2026-05-24-opd-mainline-task-backlog.md` T3 and
`docs/projects/2026-05-18-opd-only-pivot.md`.

## Context

- T3 asks for non-mainline/dead-code deletion with zero-usage evidence.
- `crates/train/data/sample.txt` was a generic toy corpus left behind after
  the OPD-only pivot deleted the `pretrain-dsv4` and generic pretrain paths.
- Current OPD entrypoints use prompt JSONL files or model directories; they do
  not read this corpus file.

## What Worked

- Deleted `crates/train/data/sample.txt`.
- Left historical `docs/experience/*` references untouched; those entries are
  past evidence, not live callers.

## Zero-Usage Evidence

```bash
rg -n "crates/train/data/sample.txt|data/sample.txt|sample.txt" \
  crates src infer README.md docs/codebase-map.md docs/support-matrix.md \
  docs/projects docs/plans docs/research -g '!docs/experience/**'
```

- Exit code 1; no live code, active docs, plans, or research hits remain.

## Verification

```bash
cargo check -p train --no-default-features
```

- Exit 0.
- This was CPU-only and did not touch P5 PID 28950, GPU runtime,
  `infer/src/kv_tier/`, or `infer/src/scheduler/`.

## Rule

- Do not keep generic pretrain fixture data after the corresponding pretrain
  command surface is retired. OPD prompt fixtures should live under explicit
  OPD paths with task-specific semantics.
