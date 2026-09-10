# Nested fixture hooks rsync-deleted the shared snapshot mid peer build

## Context

The pre-push hook serializes `rsync → cargo` into one shared target with a
machine lock, building from a persistent shared snapshot
(`${TMPDIR}/arle-pre-push-snapshot`) synced with `rsync --delete`. Shell tests
drive the REAL hook as a nested process with `ARLE_PRE_PUSH_NESTED=1`, which
skips taking that lock (the test already runs inside an outer hook holding
it). They set a private `ARLE_PREPUSH_LOCK_DIR`; most also set a private
`ARLE_PREPUSH_SNAPSHOT_ROOT`.

`scripts/tests/test_hook_skips_shell_tests.sh` set NESTED but NOT
`ARLE_PREPUSH_SNAPSHOT_ROOT`. Its mock-cargo fixture push contains `.rs`
files, so `CARGO_RUNS=1` and the hook selected the shared-snapshot branch:
it `rsync --delete`ed the tiny fixture tree over the REAL shared snapshot
while holding no lock.

## Root Cause

Snapshot selection trusted the nested flag for the LOCK but not for the
SNAPSHOT: `ARLE_PRE_PUSH_NESTED=1` skipped serialization, yet an unset
`ARLE_PREPUSH_SNAPSHOT_ROOT` silently defaulted to the shared root. Lock-free
`--delete` against a peer's live build directory is a data race. Observed
2026-09-11 (fd's push): a peer hook compiling mlx-sys lost
`crates/mlx-sys/vendor/mlx/mlx/array.cpp` mid-compilation ("No such file or
directory"). Every scripts/-touching push since #298 could trigger it.

## Fix

Structural, in `pre_push_checks.sh`: a nested run (`ARLE_PRE_PUSH_NESTED=1`)
with no explicit `ARLE_PREPUSH_SNAPSHOT_ROOT` is forced onto a private mktemp
snapshot (the existing `SNAPSHOT_PRIVATE` cleanup path), regardless of
`CARGO_RUNS`. An explicitly set root is still honored — that is the
deliberate test-owned seeding fixture tests rely on. A log line names the
private root.

`test_prepush_nested_snapshot.sh` (runs under a scratch TMPDIR) pre-seeds the
shared root with top-level and deep sentinel files, drives a nested cargo
fixture run with no override, and asserts both sentinels survive and the hook
logged a private mktemp path; an explicit-override control confirms owned
roots still sync and the default shared root stays untouched.

## Rule

A process that skips a lock protecting a shared mutable directory must also
be denied the default path to that directory. "Skip the lock" and "use the
shared root" cannot be independent knobs when the operation deletes.
Fixture/test harnesses opt into a specific shared root explicitly, never by
default.
