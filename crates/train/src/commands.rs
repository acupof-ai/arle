//! In-process CLI command implementations dispatched by `arle train ...`
//! and `arle data ...` (see `crates/cli/src/train_cli.rs`). Each submodule
//! exposes `pub fn dispatch_from_args` consumed by the CLI front door.

#[path = "commands/convert_dataset.rs"]
pub mod convert_dataset;
#[path = "commands/download_dataset.rs"]
pub mod download_dataset;
#[path = "commands/eval_lm.rs"]
pub mod eval_lm;
#[path = "commands/pretrain.rs"]
pub mod pretrain;
#[path = "commands/pretrain_dsv4.rs"]
pub mod pretrain_dsv4;
#[path = "commands/train_grpo.rs"]
pub mod train_grpo;
#[path = "commands/train_multi_turn.rs"]
pub mod train_multi_turn;
#[path = "commands/train_sft.rs"]
pub mod train_sft;
