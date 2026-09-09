# Shared target dir poisons builds across worktrees via mtime fingerprints

## Context

All lane worktrees compile into one shared `target/` (the lane setup's
deliberate trade: cargo takes one lock per target dir, so lanes serialize
instead of multiplying 8.5 GB of artifacts). On 2026-09-09 a `lane/c` build
failed with errors that did not match the source in the tree:

- `infer-server`: `Engine<E, K>` — "struct takes 0 generic arguments but 2
  were supplied", while `infer-core/src/lib.rs` plainly defines
  `pub struct Engine<E: BackendExecutor, K: KvPool>`.
- after cleaning `infer-core`: `E::Inflight` — "associated type `Inflight`
  not found", while `infer-seam` plainly defines `type Inflight`.

The code compiled on CI and in a clean tree.

## Root Cause

Cargo fingerprints are mtime+size, not content hashes, and a shared target
dir has one fingerprint set per crate/feature regardless of which checkout
wrote it. A build in an older checkout records that checkout's file mtimes;
a different checkout (newer content, but files checked out *earlier in wall
clock*) then compares its mtimes against fingerprints that say "built from
even newer files" and treats the stale rlibs as fresh. The build links
artifacts from another tree's source, and the errors describe APIs that
exist nowhere in the tree you are reading.

The lane decision counted the lock's cost (serialized builds) and not this
one: any checkout can poison every other checkout, and the symptom points
at the victim's code, not at the shared state.

## Fix

`cargo clean` (full; a partial `-p` clean helps only if you know which
crates were poisoned — the stale set is unbounded because trait changes
ripple to every implementor). Cold build afterwards compiled and passed.

## Rule

When compile errors reference APIs that do not match the source in front
of you — a struct's generic arity, an associated type, a function's
signature — suspect the shared `target/` before the code. Clean and rebuild
once. The same shape as the other 2026-09-09 shared-state bugs (`.git/config`,
pre-push snapshot, pod build tree): one mutable copy, multiple writers, no
identity; the symptom always lands on the writer whose turn it is to read.

Sibling failure of the same shared target:
`docs/experience/errors/2026-09-09-pre-push-target-dir-per-worktree.md` — the
pre-push hook's `CARGO_TARGET_DIR` overrode the shared `target-dir` config,
so each lane grew its own full target tree (18.6 GB leaked). That one is an
env-var bypass (disk); this one is fingerprint invalidation (compile errors).
