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

One target directory shared by lanes whose trait signatures disagree
(`feedback_shared_target_mtime_poisoning`): all lanes compile into the main
checkout's `target/` (and the hook into `target/pre-push-quick`), and cargo
judges freshness by mtime across worktrees. The poisoning runs both
directions — another lane's artifact can be *older* (a stale crate judged
Fresh) or *newer* (a crate built from a concurrent lane's incompatible
source judged Fresh for this tree). Either way the step checks nothing or
links foreign artifacts, and exits 0 or fails with errors that describe
someone else's tree. The hook had no assertion that the crates under change
were actually processed.

## Fix

`scripts/pre_push_checks.sh` now runs the CUDA clippy with `-v` when the pushed
range touches `crates/infer-cuda/` or `crates/cuda-kernels/`, and fails unless
the output shows `Checking`/`Compiling` for each changed crate. The condition
is the changed crates only: an unchanged crate is legitimately Fresh, and a
gate that false-positives gets disabled. `scripts/tests/test_hook_cuda_lint_freshness.sh`
constructs the stale-artifact world (mock cargo reports Fresh) and proves the
gate goes red; the test goes red itself when the assertion is removed.

## Follow-up (same day): the test steps had the same gap

A push later that day failed the hook with `E0050` (`submit` has 4 parameters
but the trait declares 3) and `E0063` (`StepOutput` missing `kv_actual`). Both
errors were fake: this tree's source had the 4-param `submit`, and the
3-param trait artifact came from a concurrent lane (Step 1b) built into the
shared `pre-push-quick` target. **The first instinct on seeing that E0050 is
to edit `submit`'s signature to match — that would break the real code.** A
fake API-mismatch error in the hook (E0050/E0063 naming APIs that match main)
means the shared target is cross-contaminated; clean before touching code:

```
CARGO_TARGET_DIR=<main-checkout>/target/pre-push-quick cargo clean
```

The freshness assertion now covers the three cargo test steps too
(`assert_step_rebuilt`, same criterion, same negative-control world in the
same test file).

Secondary phenomena observed while clearing it, all symptoms of the same
shared target:

- A concurrent lane's hook deleted artifacts mid-build — `tokio` failed with
  `extern location for signal_hook_registry does not exist` (its rmeta was
  gone). Not a tokio problem; retry once the other lane finishes.
- The cold rebuild that follows a full clean OOM-killed the push on a Mac
  under memory pressure. `CARGO_BUILD_JOBS=4` on the `lane.sh pr` invocation
  caps it; the push then completes.

## Rule

A gate whose failure mode is "checks nothing and exits 0" must assert it did
the work, not just that the command succeeded — and the assertion needs a
negative control that constructs the no-op world. A gate that shares state
with unrelated builds (one target dir per machine) must assert it for every
step that compiles, not just the first one that failed.
