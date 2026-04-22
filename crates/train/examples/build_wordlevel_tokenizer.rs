//! Build a generic whitespace word-level tokenizer from a plain-text corpus.
//!
//! Usage:
//!   cargo run -p train --release --example build_wordlevel_tokenizer -- \
//!     --corpus /path/to/corpus.txt \
//!     --out /path/to/tokenizer.json \
//!     --vocab-size 10000 \
//!     --min-count 2
//!
//! Extra framework-specific sentinel tokens can be added with repeated
//! `--special-token ...` flags. This stays generic on purpose: the example
//! builds a reusable tokenizer file, while the training binaries stay free of
//! corpus-specific vocab heuristics.

use std::{
    collections::HashMap,
    env,
    fs::{self, File},
    io::{BufRead, BufReader},
    path::PathBuf,
};

use anyhow::{Result, bail};
use train::{
    cli_args::{ArgError, next_value, parse_value},
    tokenizer::write_wordlevel_tokenizer,
};

#[derive(Debug, Clone)]
struct CliArgs {
    corpus: PathBuf,
    out: PathBuf,
    vocab_size: usize,
    min_count: usize,
    special_tokens: Vec<String>,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            corpus: PathBuf::new(),
            out: PathBuf::new(),
            vocab_size: 10_000,
            min_count: 1,
            special_tokens: Vec::new(),
        }
    }
}

fn main() -> Result<()> {
    let Some(args) = parse_args()? else {
        return Ok(());
    };
    validate_args(&args)?;

    let corpus = File::open(&args.corpus)?;
    let reader = BufReader::new(corpus);
    let mut counts = HashMap::<String, usize>::new();
    let mut total_tokens = 0usize;

    for line in reader.lines() {
        let line = line?;
        for token in line.split_whitespace() {
            total_tokens += 1;
            *counts.entry(token.to_string()).or_insert(0) += 1;
        }
    }

    let unique_tokens = counts.len();
    let mut ranked = counts
        .into_iter()
        .filter(|(_, count)| *count >= args.min_count)
        .collect::<Vec<_>>();
    ranked.sort_unstable_by(|(left_token, left_count), (right_token, right_count)| {
        right_count
            .cmp(left_count)
            .then_with(|| left_token.cmp(right_token))
    });

    let vocab = ranked
        .into_iter()
        .take(args.vocab_size)
        .map(|(token, _)| token)
        .collect::<Vec<_>>();

    if let Some(parent) = args.out.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    write_wordlevel_tokenizer(
        &args.out,
        vocab.iter().cloned(),
        args.special_tokens.iter().cloned(),
    )?;

    println!(
        "corpus={} total_tokens={} unique_tokens={} kept_tokens={} special_tokens={} out={}",
        args.corpus.display(),
        total_tokens,
        unique_tokens,
        vocab.len(),
        args.special_tokens.len(),
        args.out.display(),
    );

    Ok(())
}

fn parse_args() -> Result<Option<CliArgs>> {
    parse_args_from(env::args().skip(1))
}

fn parse_args_from<I>(mut iter: I) -> Result<Option<CliArgs>>
where
    I: Iterator<Item = String>,
{
    let mut args = CliArgs::default();
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--corpus" => args.corpus = PathBuf::from(next_value(&mut iter, &flag)?),
            "--out" => args.out = PathBuf::from(next_value(&mut iter, &flag)?),
            "--vocab-size" => {
                args.vocab_size = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--min-count" => {
                args.min_count = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--special-token" => args.special_tokens.push(next_value(&mut iter, &flag)?),
            "--help" | "-h" => {
                print_help();
                return Ok(None);
            }
            _ => return Err(ArgError::UnknownFlag(flag).into()),
        }
    }
    Ok(Some(args))
}

fn validate_args(args: &CliArgs) -> Result<()> {
    if args.corpus.as_os_str().is_empty() {
        bail!("--corpus is required");
    }
    if args.out.as_os_str().is_empty() {
        bail!("--out is required");
    }
    if args.vocab_size == 0 {
        bail!("--vocab-size must be >= 1");
    }
    if args.min_count == 0 {
        bail!("--min-count must be >= 1");
    }
    Ok(())
}

fn print_help() {
    println!(
        "build_wordlevel_tokenizer\n\
         \n\
         Flags:\n\
           --corpus <path>          Plain-text corpus to scan\n\
           --out <path>             tokenizer.json output path\n\
           --vocab-size <n>         Keep the top-N whitespace tokens (default: 10000)\n\
           --min-count <n>          Drop tokens seen fewer than N times (default: 1)\n\
           --special-token <text>   Add a special token; repeatable\n\
           --help, -h               Show this help\n"
    );
}
