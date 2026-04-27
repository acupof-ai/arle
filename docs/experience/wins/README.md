# `wins/` — bench / runtime snapshot log

This directory is the canonical log of dated benchmark and runtime snapshots
that ARLE keeps as evidence for performance and correctness claims. Each file
follows the [`TEMPLATE-bench-guidellm.md`](TEMPLATE-bench-guidellm.md) skeleton
or a slim ad-hoc variant; nothing here is overwritten — after-snapshots cite
before-snapshots with deltas instead.

Process and discipline rules live in
[`docs/bench-and-trace-spec.md`](../../bench-and-trace-spec.md). The
"every runtime change requires a bench entry" rule is in
[`AGENTS.md`](../../../AGENTS.md) and mirrored in
[`CLAUDE.md`](../../../CLAUDE.md).

---

## Filename conventions

| Pattern | Meaning |
| --- | --- |
| `YYYY-MM-DD-<slug>.md` | **Validated win.** Numbers in the file were measured locally or on the remote machine that owns the bench, and the entry has landed with verified results. |
| `YYYY-MM-DD-<slug>-pending-remote.md` | **Staged stub awaiting remote validation.** The change is in scope for a bench but the local machine cannot run it (typically: a CUDA change on a Mac development machine). The file is opened up-front per [`AGENTS.md` §Benchmarks](../../../AGENTS.md), mirrored in [`CLAUDE.md`](../../../CLAUDE.md), so the bench obligation is tracked, but the actual measurements are pending. **Do not read the numbers in these files as published claims.** |

When a `pending-remote` stub is validated on the target machine, the
process is to **rename the file** to drop the `-pending-remote` suffix in
the same commit that adds the real measurements (or in a follow-up commit
that links back to it). Renaming preserves git history; deleting the stub
and writing a fresh file would lose the trail.

## Current validated anchors

| Area | Latest validated entry |
| --- | --- |
| Metal Qwen3.5 GGUF decode | [`2026-04-27-bench-metal-qwen35-0p8b-gguf-q5-q8-q6qmv.md`](2026-04-27-bench-metal-qwen35-0p8b-gguf-q5-q8-q6qmv.md) — Qwen3.5-0.8B GGUF Q4_K_M, 512/1024 decode, 211.7 tok/s on M4 Pro. |
| Metal Qwen3.6 DFlash check | [`2026-04-27-bench-metal-qwen36-a3b-dflash-quick-check.md`](2026-04-27-bench-metal-qwen36-a3b-dflash-quick-check.md) — Qwen3.6-35B-A3B short load/execute diagnostic; not DFlash performance acceptance. |

---

## What lives here vs. elsewhere

- **Benchmark snapshots that succeeded** → `wins/`
- **Benchmark or runtime regressions, or recurring bugs** → [`../errors/`](../errors/)
- **Forward-looking plans / project docs that haven't shipped yet** → [`../../plans/`](../../plans/) and [`../../projects/`](../../projects/)

---

## For external readers

If you are evaluating ARLE's performance claims, please:

1. Skim filenames first — anything matching `*pending-remote.md` is a stub,
   not a published number.
2. Cross-reference against [`docs/support-matrix.md`](../../support-matrix.md),
   which is the authoritative status surface — wins/ entries are evidence,
   not the source of truth.
3. The canonical bench tool is [`scripts/bench_guidellm.sh`](../../../scripts/bench_guidellm.sh);
   you can reproduce any non-`pending-remote` entry against your own
   hardware using the parameters cited inside the file.
