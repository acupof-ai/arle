# String-filtered hunk staging swept half of a concurrent session's change

## Context

Committing #182 from a shared working tree that also carried another session's
in-flight #197 work in the same file (`executor/qwen35.rs`). Staged my hunks by
filtering the diff for the string `spec_row_tokens` and dropping matches.

## Root Cause

The filter keyed on one identifier, but the #197 change had a second,
lexically unrelated part: the `grow_host_slot_to` -> `set_host_slot_to`
rewrite (5 hunks, none containing the filter string). Those landed in
`c5802bc9b` while their engine-side counterpart (planner/core/seam pre-budget)
stayed uncommitted — a cross-crate change committed by half. The binary
truncated host slots nobody had pre-budgeted; the first request on the pod
died with `materialized state len 29 != DecodeRow.kv_seq_len 34`
(run sgserve3, RUN_EXIT=75).

## Fix

Reverse-applied the five hunks to the index only (`git apply --cached -R`), so
HEAD returned to the pre-#197 semantics while the owner's full change stayed
intact in the working tree (`8ad726e1c`). Verified the repaired commit in an
isolated worktree (`git worktree add <tmp> HEAD` + the Mac CUDA lane) before
rebuilding.

## Rule

Ownership of hunks in a shared file is semantic, not lexical — a string filter
bounds a concurrent change from below, never from above. Before committing a
co-edited file, diff the staged result against the merge-base and attribute
every hunk by reading it; and compile the COMMIT (isolated worktree), not the
working tree, because the tree still contains the other session's half.
