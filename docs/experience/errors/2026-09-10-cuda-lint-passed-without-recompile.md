# CUDA lint gate passed green without recompiling infer-cuda — pre-push hook, 2026-09-10

> Status: Confirmed

## Context

A push changing `crates/infer-cuda/src/executor/qwen35.rs` ran the hook's CUDA
lint (`cargo clippy -p infer-api --features cuda,no-cuda`) and it passed in
6.64s, having not recompiled `infer-cuda`. The change carried a real compile
error (`agree_abort` early-unwrapped an image, leaving a residual `image?` on
the next line); the hook's `arle cpu,no-cuda` lane caught it instead. This
clippy is the only automated gate for the CUDA-Rust surface — no GPU CI, no
self-hosted runner — so a silent no-op there lets CUDA type breakage reach
main and only fail on the pod build.

## Root Cause

The hook's shared `CARGO_TARGET_DIR` (`pre_push_checks.sh:71`, one
`target/pre-push-quick` for every lane and session on the machine) had no
interlock: a `rm -rf` or `cargo clean` by one session while another's hook
was building deletes that build's artifacts mid-flight and leaves
self-inconsistent state — old rmeta next to new rlib, a half-populated deps
dir. Two sessions independently reported doing exactly this on 2026-09-10
(deleting a peer's in-flight build), and the failure is self-sustaining:
whoever hits the weird errors reaches for `rm -rf` too and cuts the next
person's build. (The snapshot dirs are per-worktree and cargo stores
artifacts under content-hashed names, so the target itself does not
cross-contaminate between lanes; the deletion is the mechanism.)

The specific `E0050`/`E0063` chain — how a half-deleted build produced
"submit has 4 parameters but the trait declares 3" against this tree's
4-param source — is **cause unknown**. The phenomena are recorded below;
no unverified explanation is filled in.

## Fix

`scripts/pre_push_checks.sh` now runs the CUDA clippy with `-v` when the pushed
range touches `crates/infer-cuda/` or `crates/cuda-kernels/`, and fails unless
the output shows `Checking`/`Compiling` for each changed crate. The condition
is the changed crates only: an unchanged crate is legitimately Fresh, and a
gate that false-positives gets disabled. `scripts/tests/test_hook_cuda_lint_freshness.sh`
constructs the stale-artifact world (mock cargo reports Fresh) and proves the
gate goes red; the test goes red itself when the assertion is removed.

The cargo section now also runs under a machine-global lock
(`$TMPDIR/arle-pre-push-cargo.lock`, mkdir-based — no flock on macOS) so two
hooks never interleave, and an in-flight build is visible to anyone about to
clean the shared target. Stale locks (dead holder PID, or older than an hour)
are reclaimed. The hook sweeps orphaned per-worktree snapshot dirs older than
a week (20 dirs / 1.9 GB had accumulated from deleted lanes).

## Follow-up (same day): the test steps had the same gap

A push later that day failed the hook with `E0050` (`submit` has 4 parameters
but the trait declares 3) and `E0063` (`StepOutput` missing `kv_actual`).
**The first instinct on seeing that E0050 is to edit `submit`'s signature to
match — that would break the real code. The instinct is doubly dangerous now:
e2's #276 changes that exact signature for real, so the fake error and the
real migration look identical.** A fake API-mismatch error in the hook
(E0050/E0063 naming APIs that match main) means the shared target was damaged
by a concurrent deletion. First rule out the ANSI false positive (Follow-up
2): a *stable* `Fresh while this push changes it` on every push touching a
crate is the color bug, not contamination — cleaning does not help and sends
you in circles. Otherwise clean before touching code:

```
CARGO_TARGET_DIR=<main-checkout>/target/pre-push-quick cargo clean
```

The freshness assertion now covers the three cargo test steps too
(`assert_step_rebuilt`, same criterion, same negative-control world in the
same test file) — a half-deleted build that links foreign artifacts fails
loudly at the step that consumed them instead of surfacing as type errors.

Secondary phenomena observed while clearing it, all symptoms of the same
unlocked shared target:

- A concurrent lane's hook deleted artifacts mid-build — `tokio` failed with
  `extern location for signal_hook_registry does not exist` (its rmeta was
  gone). Not a tokio problem; retry once the other lane finishes.
- The cold rebuild that follows a full clean OOM-killed the push on a Mac
  under memory pressure. `CARGO_BUILD_JOBS=4` on the `lane.sh pr` invocation
  caps it; the push then completes.

## Follow-up 2 (same day): the test-step assertion's ANSI false positive

`assert_step_rebuilt` shipped with a stable false positive — every push
touching a group-2 crate blocked, and the clean remedy did nothing. Full
account in `2026-09-10-hook-freshness-ansi-false-positive.md`; fixed in #282.
The lesson for this entry's remedy: a *stable* Fresh failure on every push
is the ANSI bug, not contamination — clean first only after ruling it out.

## Rule

A gate whose failure mode is "checks nothing and exits 0" must assert it did
the work, not just that the command succeeded — and the assertion needs a
negative control that constructs the no-op world. A gate that shares state
with unrelated builds (one target dir per machine) must serialize them, and
must assert it for every step that compiles, not just the first one that
failed.
