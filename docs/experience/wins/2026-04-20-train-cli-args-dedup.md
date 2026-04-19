# train: dedup argv helpers into `train::cli_args`

## Context

Five `crates/train/src/bin/*.rs` binaries (`pretrain`, `pretrain_qwen3`,
`train_grpo`, `train_multi_turn`, `train_sft`) each carried byte-identical
`fn next_value` + `fn parse_value` helpers plus a `CliError` enum whose
first three variants (`UnknownFlag`, `MissingValue`, `InvalidValue`) were
also byte-identical. Only the binary-specific wrappers (`Io`, `Qwen3`,
`Autograd`, `Json`, `Custom`) genuinely differed.

## What Worked

- New `crates/train/src/cli_args.rs` (38 lines) owns `pub fn next_value`,
  `pub fn parse_value<T: FromStr>`, and a narrow `pub enum ArgError` with
  the three shared variants.
- Each binary keeps its own `CliError`, with the three duplicated variants
  replaced by `#[error(transparent)] Arg(#[from] ArgError)`. Binary-specific
  variants (`Custom`, `Io`, `Qwen3`, `Json`) stay local.
- Mechanical change: delete local helpers, import
  `train::cli_args::{ArgError, next_value, parse_value}`, rewrite
  `CliError::{UnknownFlag,MissingValue,InvalidValue}(...)` sites as
  `CliError::Arg(ArgError::...)`. `?` still works via `From<ArgError>`.

## Acceptance

- `cargo build --release -p train`: green.
- `cargo build --release --no-default-features --features metal -p train`: green.
- `cargo test --release -p train --lib`: 10/10 pass.
- `cargo clippy -p train --no-default-features -- -D warnings`: clean.
- Sanity: `train_sft --model` still prints `missing value for flag --model`
  verbatim; `pretrain --dataset bytes --steps notanumber` still yields an
  `InvalidValue` with matching flag+value. `#[error("...")]` Display strings
  match pre-refactor wording.

## LOC

- Before (sum of 5 bins): 2733.
- After (5 bins + cli_args.rs): 2686. Net: -47 LOC.

## Rule

Shared helper duplication with differing error-variant surfaces doesn't
need a megaenum: extract the helpers + a narrow error type, then let each
binary wrap it with `#[error(transparent)] Arg(#[from] ArgError)`. You keep
per-binary error surfaces, delete the duplication, and `?` still threads.
