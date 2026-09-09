# Pre-push snapshot was shared across lanes; a mixed tree failed the kernel bundle id test

## Context

`scripts/pre_push_checks.sh` exports HEAD into a snapshot dir and runs the
fast checks there. The snapshot path was a fixed global
`${TMPDIR}/arle-pre-push-snapshot`, shared by the main checkout and every
lane worktree.

On 2026-09-09 a `lane/c` push failed in
`scripts/tests/test_kernel_artifact_qualification.sh` with
`bundle identity mismatch: expected=bb57d49e… actual=1e3b0fee…`. The test
passed when run standalone, in a fresh `git archive` export, and on CI.

## Root Cause

Two lanes' hooks ran against the same snapshot dir. Each hook rsyncs its own
HEAD into it; an interrupted rsync (killed hook — six stale stage dirs left
behind) leaves a tree mixed from two HEADs. The failing test hashes
working-tree content for the kernel bundle identity (its `git ls-files`
path is inert outside a repo), and a concurrent rsync changed
`crates/cuda-kernels/build.rs` and `src/quant_linear.rs` between the test's
two identity computations, so the recomputed id no longer matched the one
baked into the candidate bundle at test start.

## Fix

The snapshot path now includes a hash of the worktree root, so each checkout
has its own snapshot and keeps its warm incremental cache.

## Rule

A shared build/test cache dir must be namespaced per worktree. A test that
fails only inside the hook and passes standalone means the hook's
environment — shared dirs, parallel steps — is part of the system under
test.
