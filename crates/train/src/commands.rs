//! In-process CLI command implementations dispatched by `arle train ...`
//! and `arle data ...` (see `crates/cli/src/train_cli.rs`). Each submodule
//! exposes `pub fn dispatch_from_args` consumed by the CLI front door.

#[path = "commands/convert_dataset.rs"]
pub mod convert_dataset;
#[path = "commands/download_dataset.rs"]
pub mod download_dataset;
#[path = "commands/eval_lm.rs"]
pub mod eval_lm;
