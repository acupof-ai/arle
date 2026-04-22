//! Build a generic byte-level BPE tokenizer from one or more plain-text corpora.
//!
//! Usage:
//!   cargo run -p train --release --example build_bpe_tokenizer -- \
//!     --corpus /path/to/corpus_a.txt \
//!     --corpus /path/to/corpus_b.txt \
//!     --out /path/to/tokenizer.json \
//!     --vocab-size 4096 \
//!     --min-frequency 2 \
//!     --special-token "<|endoftext|>"
//!
//! This stays framework-agnostic on purpose. The output is a reusable
//! `tokenizer.json`; training binaries keep tokenizer policy outside the core
//! train loop.

use std::{collections::HashSet, env, fs, path::PathBuf};

use anyhow::{Result, bail};
use tokenizers::{
    AddedToken, DecoderWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper,
    TokenizerBuilder,
    decoders::{byte_fallback::ByteFallback, sequence::Sequence as DecoderSequence},
    models::bpe::{BPE, BpeTrainerBuilder},
    normalizers::{strip::Strip, unicode::NFC, utils::Sequence as NormalizerSequence},
    pre_tokenizers::byte_level::ByteLevel,
};
use train::cli_args::{ArgError, next_value, parse_value};

#[derive(Debug, Clone)]
struct CliArgs {
    corpora: Vec<PathBuf>,
    out: PathBuf,
    vocab_size: usize,
    min_frequency: u64,
    special_tokens: Vec<String>,
    unk_token: String,
    add_prefix_space: bool,
    byte_fallback: bool,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            corpora: Vec::new(),
            out: PathBuf::new(),
            vocab_size: 4_096,
            min_frequency: 2,
            special_tokens: Vec::new(),
            unk_token: "[UNK]".to_string(),
            add_prefix_space: false,
            byte_fallback: false,
        }
    }
}

fn main() -> Result<()> {
    let Some(args) = parse_args()? else {
        return Ok(());
    };
    validate_args(&args)?;

    let byte_level = ByteLevel::default().add_prefix_space(args.add_prefix_space);
    let special_tokens = dedup_special_tokens(args.special_tokens.iter().cloned(), &args.unk_token);
    let added_special_tokens = special_tokens
        .iter()
        .cloned()
        .map(|token| AddedToken::from(token, true).special(true))
        .collect::<Vec<_>>();
    let mut trainer = BpeTrainerBuilder::new()
        .show_progress(true)
        .vocab_size(args.vocab_size)
        .min_frequency(args.min_frequency)
        .special_tokens(added_special_tokens)
        .build();
    let decoder = if args.byte_fallback {
        DecoderSequence::new(vec![ByteFallback::default().into(), byte_level.into()]).into()
    } else {
        byte_level.into()
    };
    let mut tokenizer = TokenizerBuilder::<
        BPE,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >::default()
    .with_model(
        BPE::builder()
            .unk_token(args.unk_token.clone())
            .byte_fallback(args.byte_fallback)
            .build()
            .map_err(|err| anyhow::anyhow!("build BPE model: {err}"))?,
    )
    .with_normalizer(Some(NormalizerWrapper::Sequence(NormalizerSequence::new(
        vec![Strip::new(true, true).into(), NFC.into()],
    ))))
    .with_pre_tokenizer(Some(PreTokenizerWrapper::ByteLevel(byte_level)))
    .with_post_processor(Some(PostProcessorWrapper::ByteLevel(byte_level)))
    .with_decoder(Some(decoder))
    .build()
    .map_err(|err| anyhow::anyhow!("build tokenizer: {err}"))?;

    let corpus_paths = args
        .corpora
        .iter()
        .map(|path| path.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    tokenizer
        .train_from_files(&mut trainer, corpus_paths)
        .map_err(|err| anyhow::anyhow!("train tokenizer: {err}"))?;

    if let Some(parent) = args.out.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    tokenizer
        .save(&args.out, false)
        .map_err(|err| anyhow::anyhow!("save tokenizer: {err}"))?;

    println!(
        "corpora={} vocab_size={} unk_token={} special_tokens={} add_prefix_space={} byte_fallback={} out={}",
        args.corpora.len(),
        tokenizer.get_vocab_size(true),
        args.unk_token,
        special_tokens.len(),
        args.add_prefix_space,
        args.byte_fallback,
        args.out.display(),
    );

    Ok(())
}

fn dedup_special_tokens(tokens: impl IntoIterator<Item = String>, unk_token: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();
    for token in std::iter::once(unk_token.to_string()).chain(tokens) {
        if seen.insert(token.clone()) {
            deduped.push(token);
        }
    }
    deduped
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
            "--corpus" => args
                .corpora
                .push(PathBuf::from(next_value(&mut iter, &flag)?)),
            "--out" => args.out = PathBuf::from(next_value(&mut iter, &flag)?),
            "--vocab-size" => {
                args.vocab_size = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--min-frequency" => {
                args.min_frequency = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--special-token" => args.special_tokens.push(next_value(&mut iter, &flag)?),
            "--unk-token" => args.unk_token = next_value(&mut iter, &flag)?,
            "--add-prefix-space" => args.add_prefix_space = true,
            "--byte-fallback" => args.byte_fallback = true,
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
    if args.corpora.is_empty() {
        bail!("at least one --corpus is required");
    }
    if args.out.as_os_str().is_empty() {
        bail!("--out is required");
    }
    if args.vocab_size == 0 {
        bail!("--vocab-size must be >= 1");
    }
    if args.min_frequency == 0 {
        bail!("--min-frequency must be >= 1");
    }
    if args.unk_token.is_empty() {
        bail!("--unk-token must not be empty");
    }
    Ok(())
}

fn print_help() {
    println!(
        "build_bpe_tokenizer\n\
         \n\
         Flags:\n\
           --corpus <path>          Plain-text corpus path; repeatable\n\
           --out <path>             tokenizer.json output path\n\
           --vocab-size <n>         Target vocab size (default: 4096)\n\
           --min-frequency <n>      Drop pairs below this frequency (default: 2)\n\
           --special-token <text>   Add a special token; repeatable\n\
           --unk-token <text>       Unknown token string (default: [UNK])\n\
           --add-prefix-space       Enable ByteLevel add_prefix_space\n\
           --byte-fallback          Enable BPE byte_fallback\n\
           --help, -h               Show this help\n"
    );
}
