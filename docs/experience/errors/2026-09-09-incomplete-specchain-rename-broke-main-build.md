# Incomplete SpecChain rename broke main's CUDA lint lane — 2026-09-09

## Context

Commit (quantized decode split ceiling 16 -> 64) also executed a
planned rename from the batched-mtp-verify plan: struct `DsparkChain` ->
`SpecChain` in `crates/infer-cuda/src/executor/qwen35.rs`. The definition was
renamed; three use sites (lines 1775, 1906, 2094) kept the old name. The
commit landed on main, and every cargo build compiling infer-cuda failed with
`cannot find type DsparkChain in this scope` (E0425/E0422) — including the
pre-push hook's `clippy -p infer-api --features cuda,no-cuda` lane, which
blocked every push whose range contained a `.rs` file.

## Root Cause

Two stacked faults:

1. The rename was incomplete. The old identifier was never grepped after the
   edit; a `grep -rn DsparkChain crates/` would have shown the three remaining
   sites immediately.
2. The commit message claimed the Mac CUDA lint lane was clean. The lint had
   actually failed, but its output was piped to `tail`, and `tail` exited 0,
   masking clippy's exit code. The same masking happened twice in that
   session and was noticed only when a later command failed for an unrelated
   reason.

## Fix

 renames the three use sites. Both lint lanes were re-run without
a pipe and exit 0:

- `clippy -p infer-api --no-default-features --features cuda,no-cuda --lib
  -- -D warnings` (debug, the hook's exact lane)
- `clippy -p infer-api --release --no-default-features --features
  cuda,no-cuda,nccl,deepep --lib -- -D warnings` (the CLAUDE.md lane)

The batched-MTP plan doc's struct table was updated to the new name (plan deleted 2026-09-11; the batched MTP feature shipped #248/#258).

## Rule

A rename is complete only when the old identifier has zero hits in the tree —
grep it after committing. A verification command's output never pipes its
exit code away: check `$pipestatus[1]` (zsh) / `${PIPESTATUS[0]}` (bash), or
run without the pipe. A green tail is not a green build.
