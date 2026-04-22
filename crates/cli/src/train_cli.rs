use std::process::ExitCode;

use anyhow::{Result, bail};

use crate::args::{CliCommand, DataArgs, DataCommand, ForwardedArgs, TrainArgs, TrainCommand};

#[allow(dead_code)]
#[path = "../../train/src/bin/convert_dataset.rs"]
mod convert_dataset_entry;
#[allow(dead_code)]
#[path = "../../train/src/bin/download_dataset.rs"]
mod download_dataset_entry;
#[allow(dead_code)]
#[path = "../../train/src/bin/eval_lm.rs"]
mod eval_lm_entry;
#[allow(dead_code)]
#[path = "../../train/src/bin/pretrain.rs"]
mod pretrain_entry;
#[allow(dead_code)]
#[path = "../../train/src/bin/train_grpo.rs"]
mod train_grpo_entry;
#[allow(dead_code)]
#[path = "../../train/src/bin/train_multi_turn.rs"]
mod train_multi_turn_entry;
#[allow(dead_code)]
#[path = "../../train/src/bin/train_sft.rs"]
mod train_sft_entry;

const PRETRAIN_HELP: &str = r#"usage: agent-infer train pretrain [args...]

Scratch-pretrain a Qwen-family checkpoint from random init.

Common flags:
  --corpus <txt>        Plain-text training corpus
  --tokenizer <json>    tokenizer.json path
  --out <dir>           Output checkpoint directory
  --steps <n>           Training steps
  --seq <n>             Sequence length
  --backend <cpu|metal|cuda>

Example:
  agent-infer train pretrain --corpus corpus.txt --tokenizer tokenizer.json --out out/pretrain --steps 1000 --seq 512 --backend metal
"#;

const SFT_HELP: &str = r#"usage: agent-infer train sft [args...]

Supervised fine-tuning on canonical chat JSONL.

Common flags:
  --model <dir>         Base checkpoint directory
  --data <jsonl>        Chat-style dataset
  --out <dir>           Output checkpoint directory
  --steps <n>           Training steps
  --seq-len <n>         Sequence length
  --backend <cpu|metal|cuda>

Example:
  agent-infer train sft --model checkpoints/base --data train.chat.jsonl --out out/sft --steps 500 --seq-len 512 --backend metal
"#;

const GRPO_HELP: &str = r#"usage: agent-infer train grpo [args...]

Run GRPO on the synthetic prompt/reward loop.

Common flags:
  --model-family <qwen35|qwen3>
  --grpo-iters <n>
  --batch-prompts <n>
  --group-size <n>
  --seq <n>
  --backend <cpu|metal|cuda>

Example:
  agent-infer train grpo --model-family qwen35 --grpo-iters 20 --batch-prompts 4 --group-size 8 --seq 64 --backend metal
"#;

const MULTI_TURN_HELP: &str = r#"usage: agent-infer train multi-turn [args...]

Run multi-turn RL training with stepwise GRPO or sequence GSPO.

Common flags:
  --iters <n>
  --group-size <n>
  --prompt-len <n>
  --agent-tokens <n>
  --obs-tokens <n>
  --turns <n>
  --backend <cpu|metal|cuda>

Example:
  agent-infer train multi-turn --iters 20 --group-size 8 --prompt-len 16 --agent-tokens 16 --obs-tokens 16 --turns 2 --backend metal
"#;

const EVAL_HELP: &str = r#"usage: agent-infer train eval [args...]

Evaluate a checkpoint on tokenized or chat JSONL.

Common flags:
  --model-path <dir>    Checkpoint directory
  --data <jsonl|txt>    Evaluation dataset
  --seq-len <n>         Sequence length
  --backend <cpu|metal|cuda>

Example:
  agent-infer train eval --model-path checkpoints/base --data eval.chat.jsonl --seq-len 512 --backend metal
"#;

const DOWNLOAD_HELP: &str = r#"usage: agent-infer data download [args...]

Download one file from a Hugging Face dataset repository.

Required flags:
  --repo <hf-dataset-id>
  --file <path-in-repo>

Example:
  agent-infer data download --repo allenai/tulu-3-sft-mixture --file data/train.jsonl
"#;

const CONVERT_HELP: &str = r#"usage: agent-infer data convert [args...]

Convert instruction-tuning JSONL into canonical chat JSONL.

Required flags:
  --input <jsonl>
  --format <chat|dolly|alpaca|sharegpt>
  --output <jsonl>

Example:
  agent-infer data convert --input raw.jsonl --format dolly --output train.chat.jsonl
"#;

pub(crate) fn run(command: CliCommand) -> Result<()> {
    match command {
        CliCommand::Train(train) => run_train(train),
        CliCommand::Data(data) => run_data(data),
    }
}

fn run_train(train: TrainArgs) -> Result<()> {
    match train.command {
        TrainCommand::Pretrain(args) => {
            run_train_entry(args, PRETRAIN_HELP, pretrain_entry::dispatch_from_args)
        }
        TrainCommand::Sft(args) => {
            run_train_entry(args, SFT_HELP, train_sft_entry::dispatch_from_args)
        }
        TrainCommand::Grpo(args) => {
            run_train_entry(args, GRPO_HELP, train_grpo_entry::dispatch_from_args)
        }
        TrainCommand::MultiTurn(args) => run_train_entry(
            args,
            MULTI_TURN_HELP,
            train_multi_turn_entry::dispatch_from_args,
        ),
        TrainCommand::Eval(args) => {
            run_train_entry(args, EVAL_HELP, eval_lm_entry::dispatch_from_args)
        }
    }
}

fn run_data(data: DataArgs) -> Result<()> {
    match data.command {
        DataCommand::Download(args) => run_data_entry(
            args,
            DOWNLOAD_HELP,
            download_dataset_entry::dispatch_from_args,
        ),
        DataCommand::Convert(args) => run_data_entry(
            args,
            CONVERT_HELP,
            convert_dataset_entry::dispatch_from_args,
        ),
    }
}

fn run_train_entry<F>(args: ForwardedArgs, help: &str, run: F) -> Result<()>
where
    F: FnOnce(Vec<String>) -> std::result::Result<(), String>,
{
    if should_print_help(&args.args) {
        print!("{help}");
        return Ok(());
    }
    run(args.args).map_err(|err| anyhow::anyhow!(err))
}

fn run_data_entry<F>(args: ForwardedArgs, help: &str, run: F) -> Result<()>
where
    F: FnOnce(Vec<String>) -> ExitCode,
{
    if should_print_help(&args.args) {
        print!("{help}");
        return Ok(());
    }
    let code = run(args.args);
    if code == ExitCode::SUCCESS {
        Ok(())
    } else {
        bail!("dataset command failed with exit code {:?}", code);
    }
}

fn should_print_help(args: &[String]) -> bool {
    args.is_empty() || args.iter().any(|arg| arg == "-h" || arg == "--help")
}
