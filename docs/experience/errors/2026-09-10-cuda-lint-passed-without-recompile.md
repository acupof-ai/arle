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

Shared target dir, mtime fingerprints (`feedback_shared_target_mtime_poisoning`):
all lanes compile into the main checkout's `target/`, and cargo judges
freshness by mtime across worktrees. Another lane's newer artifact made this
run report `infer-cuda` Fresh, so the lint checked nothing and exited 0. The
hook had no assertion that the crates under change were actually processed.

## Fix

`scripts/pre_push_checks.sh` now runs the CUDA clippy with `-v` when the pushed
range touches `crates/infer-cuda/` or `crates/cuda-kernels/`, and fails unless
the output shows `Checking`/`Compiling` for each changed crate. The condition
is the changed crates only: an unchanged crate is legitimately Fresh, and a
gate that false-positives gets disabled. `scripts/tests/test_hook_cuda_lint_freshness.sh`
constructs the stale-artifact world (mock cargo reports Fresh) and proves the
gate goes red; the test goes red itself when the assertion is removed.

## Rule

A gate whose failure mode is "checks nothing and exits 0" must assert it did
the work, not just that the command succeeded — and the assertion needs a
negative control that constructs the no-op world.
