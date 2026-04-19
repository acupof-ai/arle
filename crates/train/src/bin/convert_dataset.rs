//! CLI: convert an instruction-tuning dataset JSONL from a common public
//! schema into the canonical `{"messages": [...]}` format consumed by
//! `train_sft`. See `train::data_adapter` for supported formats.
//!
//! Usage:
//! ```bash
//! convert_dataset --input  /path/to/dolly.jsonl \
//!                 --format dolly \
//!                 --output /path/to/dolly.chat.jsonl
//! ```

use std::{path::PathBuf, process::ExitCode};

use train::data_adapter::{InputFormat, convert_file};

fn main() -> ExitCode {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut format: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => input = args.next().map(PathBuf::from),
            "--output" => output = args.next().map(PathBuf::from),
            "--format" => format = args.next(),
            "--help" | "-h" => {
                eprintln!(
                    "usage: convert_dataset --input <jsonl> --format <chat|dolly|alpaca|sharegpt> --output <jsonl>"
                );
                return ExitCode::from(0);
            }
            other => {
                eprintln!("unknown flag: {other}");
                return ExitCode::from(2);
            }
        }
    }

    let Some(input) = input else {
        eprintln!("error: --input is required");
        return ExitCode::from(2);
    };
    let Some(output) = output else {
        eprintln!("error: --output is required");
        return ExitCode::from(2);
    };
    let Some(format) = format else {
        eprintln!("error: --format is required (chat|dolly|alpaca|sharegpt)");
        return ExitCode::from(2);
    };

    let format = match InputFormat::parse(&format) {
        Ok(f) => f,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::from(2);
        }
    };

    eprintln!(
        "[convert_dataset] {} ({:?}) → {}",
        input.display(),
        format,
        output.display()
    );
    match convert_file(&input, &output, format) {
        Ok(stats) => {
            eprintln!(
                "[convert_dataset] {} lines · {} written · {} skipped",
                stats.total_lines, stats.written, stats.skipped
            );
            if stats.written == 0 {
                eprintln!(
                    "[convert_dataset] error: produced 0 records (wrong --format, or input didn't match the expected schema)"
                );
                return ExitCode::from(1);
            }
            ExitCode::from(0)
        }
        Err(err) => {
            eprintln!("[convert_dataset] error: {err:#}");
            ExitCode::from(1)
        }
    }
}
