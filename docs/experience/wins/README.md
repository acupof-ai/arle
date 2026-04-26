# `wins/` — bench / runtime snapshot log

This directory is the canonical log of dated benchmark and runtime snapshots
that ARLE keeps as evidence for performance and correctness claims. Each file
follows the [`TEMPLATE-bench-guidellm.md`](TEMPLATE-bench-guidellm.md) skeleton
or a slim ad-hoc variant; nothing here is overwritten — after-snapshots cite
before-snapshots with deltas instead.

Process and discipline rules live in
[`docs/bench-and-trace-spec.md`](../../bench-and-trace-spec.md). The
"every runtime change requires a bench entry" rule is in
[`CLAUDE.md`](../../../CLAUDE.md).

---

## Filename conventions

| Pattern | Meaning |
| --- | --- |
| `YYYY-MM-DD-<slug>.md` | **Validated win.** Numbers in the file were measured locally or on the remote machine that owns the bench, and the entry has landed with verified results. |
| `YYYY-MM-DD-<slug>-pending-remote.md` | **Staged stub awaiting remote validation.** The change is in scope for a bench but the local machine cannot run it (typically: a CUDA change on a Mac development machine). The file is opened up-front per [`CLAUDE.md` §Benchmarks](../../../CLAUDE.md) so the bench obligation is tracked, but the actual measurements are pending. **Do not read the numbers in these files as published claims.** |

When a `pending-remote` stub is validated on the target machine, the
process is to **rename the file** to drop the `-pending-remote` suffix in
the same commit that adds the real measurements (or in a follow-up commit
that links back to it). Renaming preserves git history; deleting the stub
and writing a fresh file would lose the trail.

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
