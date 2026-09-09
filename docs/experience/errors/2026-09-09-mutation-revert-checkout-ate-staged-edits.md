# Mutation-test revert with `git checkout --` ate the file's staged edits

## Context

During the `infer-quant` extraction, the negative-control test mutated
`crates/infer-quant/src/lib.rs` with `sed` (broke one format constraint),
confirmed the test went red, then reverted the mutation with
`git checkout -- crates/infer-quant/src/lib.rs`. The file had entered the
index via `git mv` at its original content, and the move's required
visibility edits (`pub(crate)` → `pub`) had been made with `sed` but never
staged. The checkout restored the index version — the pre-edit original —
silently wiping the visibility edits. The Mac CUDA clippy gate caught it as
14 `dead_code` errors; the file looked fine in review because it still
compiled standalone.

## Root Cause

`git checkout -- <file>` copies the index version over the working tree; it
does not revert one hunk. Any unstaged edit in the file is destroyed along
with the mutation. This is the second occurrence of a `git checkout` data-loss
shape (the first involved a shared dirty file); the new shape is that the
revert of a deliberate mutation ate *unrelated, unstaged* edits in the same
file.

## Fix

Re-applied the visibility edits, re-ran the tests, amended the commit.

## Rule

In a mutation test, never revert with `git checkout --` on a file that has
unstaged edits. Either `git add` the file before mutating (so checkout
restores your edited version), copy the original text and revert with the
inverse `sed`, or `git stash push -- <file>` and pop it after the red
assertion. Re-run the test after any revert.
