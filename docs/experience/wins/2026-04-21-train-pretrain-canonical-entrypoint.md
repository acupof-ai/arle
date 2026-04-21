# Canonical `pretrain` entrypoint now fronts the generic Qwen-family trainer

## Context

The scratch-pretrain binary had already stopped being Qwen3-only, but the
public surface still looked legacy: the cargo bin was named
`pretrain_qwen3`, docs kept treating that as the active entrypoint, and the
CLI still used the older `--resume` spelling while the rest of the train
stack had standardized on `--resume-from`.

## What Worked

- Added a canonical `pretrain` cargo bin that points at the generic
  Qwen-family pretrain implementation.
- Kept `pretrain_qwen3` as a compatibility alias instead of forking the code
  path.
- Made `--resume-from` the canonical accepted flag on the pretrain binary
  while preserving `--resume` as a compatibility alias.
- Updated the pretrain binary's user-facing log/error prefixes to use the
  generic `pretrain` name.
- Refreshed active train docs to describe `pretrain` as the live scratch
  pretrain entrypoint and pushed `pretrain_qwen3` down to compatibility-only
  wording.

## Verification

- `cargo test -p train --release --bin pretrain -- --nocapture`
- `cargo test -p train --release --bin pretrain_qwen3 -- --nocapture`
- `cargo test -p train --release`
- `cargo clippy -p train --all-targets -- -D warnings`

## Rule

Once a binary stops being architecture-specific, the canonical CLI/docs
surface must stop encoding the old architecture name. Keep the old name only
as a compatibility alias until every active caller moves.
