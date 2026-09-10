# Pre-push snapshot rsync restored commit-time mtimes, making changed crates report Fresh

## Context

The pre-push hook builds every push in a snapshot of `HEAD` so it never
compiles a dirty worktree:

```sh
git archive HEAD | tar -x -C "$STAGE"
rsync -a --delete --checksum "$STAGE/" "$SNAPSHOT/"
```

Over 2026-09-09..10 several lanes hit "Fresh while this push changes it" gate
failures and, worse, pushes that compiled the wrong artifact (a missing
`KvAllocator::alloc`, fake E0050/E0063). The working theory was cross-lane
poisoning of the shared `target/pre-push-quick`: lane A builds a crate, lane B
links lane A's rmeta. That is real (see the 09-09 entry) but it is a *symptom*.
A same-HEAD retry could still be told Fresh, with no concurrent peer involved.

## Root Cause

`git archive` writes each entry with the **commit** timestamp. The hook then
rsynced with `-a`, which includes `-t` (preserve mtimes). So a source file
landed in the snapshot with its commit time, not the refresh time.

Cargo fingerprints are mtime+size: a crate rebuilds when a source mtime is
newer than the artifact. For a commit made *before* the previous build, the
restored source mtime is **older** than the output artifact, so cargo reports
the crate Fresh even when its content changed. One lane, no concurrency,
reproduces it.

Controlled experiment (real `git archive | tar` + `rsync -a --checksum`, two
commits timestamped in the past, one shared target):

| step | snapshot | content vs prior | cargo with `-a` |
|---|---|---|---|
| 1 | B (cold) | — | **REBUILD** |
| 1b | B again | identical | FRESH (warm, correct) |
| 2 | A | changed (v=2 → v=1) | **FRESH — wrong** |
| 3 | B | changed (v=1 → v=2) | **FRESH — wrong** |

Steps 2/3 both changed `src/lib.rs` content yet skipped compilation.

Two amplifiers:
- The snapshot was **per-worktree** (`…/snapshot-<hash>`) but the target was
  shared, so one target held artifacts built from many source roots.
- The freshness gate's advice (`cargo clean -p`) does not hold: cleaning the
  leaf crate does not invalidate its *dependents'* fingerprints, and a peer hook
  rebuilds it from its own snapshot before the retry.

## Fix

1. Rsync without `-t`: `rsync -rlpD --delete --checksum`. Transferred (changed)
   files take mtime=now, so a content change always rebuilds; `--checksum`
   leaves content-identical files' mtimes, so an unchanged re-refresh keeps the
   warm cache (verified: FRESH).
2. Collapse the snapshot back to ONE shared dir and hold a machine lock around
   the refresh+cargo window **for runs that compile** (any `.rs` change or
   Metal enabled). The lock keeps its historical path
   `arle-pre-push-cargo.lock` so old- and new-hook lanes take the same lock
   during rollout instead of compiling concurrently; it now spans refresh, not
   just cargo. With one source root and serialized refresh+build, artifacts in
   the shared target always come from the snapshot on disk; an interrupted
   rsync is repaired by the next `--checksum --delete`. A **cargo-free** push
   (no `.rs`, Metal off) never touches the shared target, so it builds from a
   private mktemp snapshot and takes no lock — otherwise waiting behind a
   peer's cargo run held its already-open SSH connection idle until the remote
   dropped it (push exit 141; that batching half is #298).
3. Two orphan sweeps: legacy per-worktree `arle-pre-push-snapshot-*` dirs after
   7 days (the suffix-less shared dir never matches; 7 days covers un-rebased
   rollout lanes), and SIGKILL-orphaned private `arle-pre-push-nocargo.*` dirs
   after 2 hours.
4. The freshness assertion stays as a detector; its message now says this points
   to a real bug to report, not stale cache to `cargo clean`.

Fixture `test_prepush_snapshot_mtime.sh` pins rebuild-on-change / Fresh-when-identical with the real cargo; the lock worlds in
`test_hook_cuda_lint_freshness.sh` pin that a docs-only push takes the private
snapshot without waiting while a `.rs` push waits on the shared lock.

## Rule

Never let a build input carry a historical mtime into a fresh build tree.
Archiving VCS content restores commit times; a copy that feeds a fingerprinting
build must either re-stamp mtimes to now or key on content hashes. "Same crate
name printed a Checking line" is necessary but not sufficient — the gate cannot
read which source root produced an artifact, so there must be only one source
root.

Cross-lane symptom, same shared-state family:
`2026-09-09-shared-target-dir-mtime-poisoning.md`; disk-cost sibling:
`2026-09-09-pre-push-target-dir-per-worktree.md`. Related: #298.
