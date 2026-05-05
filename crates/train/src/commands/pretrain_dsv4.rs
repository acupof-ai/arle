// DSV4 pretrain driver — scaffold fork of `pretrain.rs`.
//
// Per docs/plans/2026-05-05-deepseek-v4-small-substrate.md the runtime
// scaffold + spec config are landing now; the autograd-side `DeepseekModel`
// (mirroring `train::qwen3::Qwen3Model`) is still pending. This driver
// exists so the CLI surface, argument validation, and config wiring are
// already proven once that train-side model lands — at which point this
// file's `dispatch_from_args` will be filled in with a real `Trainer<O, C, S>`
// loop, mirroring `pretrain.rs::run_with_family`.
//
// Mirrors `commands/pretrain.rs::dispatch_from_args` — the entry point CLI
// front door (`crates/cli/src/train_cli.rs`) calls.

use std::path::PathBuf;

use deepseek_spec::DeepSeekConfig;
use thiserror::Error;

/// DSV4 SKU selector. Only `nano` is wired today; SKU-A and SKU-B follow
/// once their `DeepSeekConfig::tiny_dense()` / `mini_moe()` constructors
/// land in `crates/deepseek-spec/`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepseekSku {
    Nano,
}

impl DeepseekSku {
    pub fn from_str(input: &str) -> Result<Self, DsV4PretrainError> {
        match input {
            "nano" => Ok(Self::Nano),
            other => Err(DsV4PretrainError::UnknownSku(other.to_string())),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Nano => "nano",
        }
    }

    pub fn build_config(self) -> DeepSeekConfig {
        match self {
            Self::Nano => DeepSeekConfig::nano(),
        }
    }
}

/// Parsed CLI for `arle train pretrain-dsv4`. Mirrors a thin subset of
/// `pretrain.rs::CliArgs` — only the arguments the scaffold actually
/// consumes today. The full superset (LR schedule, eval cadence, resume,
/// etc.) wires in alongside the autograd model.
#[derive(Debug, Clone)]
pub struct CliArgs {
    pub sku: DeepseekSku,
    pub corpus: PathBuf,
    pub tokenizer: PathBuf,
    pub out: PathBuf,
    pub seed: u64,
}

#[derive(Debug, Error)]
pub enum DsV4PretrainError {
    #[error("missing required argument: {0}")]
    MissingArg(&'static str),
    #[error("unknown DSV4 SKU `{0}` — only `nano` is wired today")]
    UnknownSku(String),
    #[error("argument `{flag}` requires a value")]
    MissingValue { flag: String },
    #[error("argument `{flag}` value `{value}` is not a valid {kind}")]
    InvalidValue {
        flag: String,
        value: String,
        kind: &'static str,
    },
    #[error(
        "DSV4 pretrain driver: train-side `DeepseekModel` (autograd) is not yet wired. \
         The runtime scaffold + spec config landed under \
         docs/plans/2026-05-05-deepseek-v4-small-substrate.md §6; the train-side autograd \
         model + checkpoint adapter follow once the MLA prefill + decode kernels stabilize. \
         See substrate plan §4 (Pretrain Stack) for the cold-path policy."
    )]
    AutogradModelPending,
}

pub fn parse_args_from<I>(args: I) -> Result<CliArgs, DsV4PretrainError>
where
    I: IntoIterator<Item = String>,
{
    let mut iter = args.into_iter();
    let mut sku: Option<DeepseekSku> = None;
    let mut corpus: Option<PathBuf> = None;
    let mut tokenizer: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut seed: u64 = 0xC0FFEE;

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--deepseek-config" => {
                let value = iter.next().ok_or_else(|| DsV4PretrainError::MissingValue {
                    flag: arg.to_string(),
                })?;
                sku = Some(DeepseekSku::from_str(&value)?);
            }
            "--corpus" => {
                corpus = Some(PathBuf::from(iter.next().ok_or_else(|| {
                    DsV4PretrainError::MissingValue {
                        flag: arg.to_string(),
                    }
                })?));
            }
            "--tokenizer" => {
                tokenizer = Some(PathBuf::from(iter.next().ok_or_else(|| {
                    DsV4PretrainError::MissingValue {
                        flag: arg.to_string(),
                    }
                })?));
            }
            "--out" => {
                out = Some(PathBuf::from(iter.next().ok_or_else(|| {
                    DsV4PretrainError::MissingValue {
                        flag: arg.to_string(),
                    }
                })?));
            }
            "--seed" => {
                let value = iter.next().ok_or_else(|| DsV4PretrainError::MissingValue {
                    flag: arg.to_string(),
                })?;
                seed = value
                    .parse::<u64>()
                    .map_err(|_| DsV4PretrainError::InvalidValue {
                        flag: arg.to_string(),
                        value,
                        kind: "u64",
                    })?;
            }
            // Unknown flags are kept silent so the CLI front door can pass
            // through extra arguments destined for the future full driver.
            _ => {}
        }
    }

    Ok(CliArgs {
        sku: sku.unwrap_or(DeepseekSku::Nano),
        corpus: corpus.ok_or(DsV4PretrainError::MissingArg("--corpus"))?,
        tokenizer: tokenizer.ok_or(DsV4PretrainError::MissingArg("--tokenizer"))?,
        out: out.ok_or(DsV4PretrainError::MissingArg("--out"))?,
        seed,
    })
}

/// Public CLI entry. Same shape as `pretrain::dispatch_from_args` so
/// `train_cli.rs::run_train_command` can route through the same harness.
pub fn dispatch_from_args<I>(args: I) -> Result<(), String>
where
    I: IntoIterator<Item = String>,
{
    let parsed = parse_args_from(args).map_err(|err| err.to_string())?;
    run(&parsed).map_err(|err| err.to_string())
}

fn run(args: &CliArgs) -> Result<(), DsV4PretrainError> {
    let cfg = args.sku.build_config();
    println!(
        "[pretrain-dsv4] sku={} hidden={} layers={} kv_lora_rank={} qk_rope_head_dim={} \
         qk_nope_head_dim={} v_head_dim={} vocab={}",
        args.sku.as_str(),
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.kv_lora_rank,
        cfg.qk_rope_head_dim,
        cfg.qk_nope_head_dim,
        cfg.v_head_dim,
        cfg.vocab_size,
    );
    println!(
        "[pretrain-dsv4] corpus={} tokenizer={} out={} seed={}",
        args.corpus.display(),
        args.tokenizer.display(),
        args.out.display(),
        args.seed,
    );

    // The training loop body (autograd model + Trainer<O, C, S>) is not
    // yet wired — see `DsV4PretrainError::AutogradModelPending` for the
    // full rationale and substrate-plan pointer.
    Err(DsV4PretrainError::AutogradModelPending)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parses_minimal_invocation() {
        let parsed = parse_args_from(vec_of(&[
            "--deepseek-config",
            "nano",
            "--corpus",
            "corpus.txt",
            "--tokenizer",
            "tokenizer.json",
            "--out",
            "/tmp/dsv4-nano",
        ]))
        .unwrap();

        assert_eq!(parsed.sku, DeepseekSku::Nano);
        assert_eq!(parsed.corpus, PathBuf::from("corpus.txt"));
        assert_eq!(parsed.tokenizer, PathBuf::from("tokenizer.json"));
        assert_eq!(parsed.out, PathBuf::from("/tmp/dsv4-nano"));
    }

    #[test]
    fn defaults_sku_to_nano() {
        let parsed = parse_args_from(vec_of(&[
            "--corpus",
            "corpus.txt",
            "--tokenizer",
            "tokenizer.json",
            "--out",
            "/tmp/out",
        ]))
        .unwrap();
        assert_eq!(parsed.sku, DeepseekSku::Nano);
    }

    #[test]
    fn rejects_unknown_sku() {
        let err = parse_args_from(vec_of(&[
            "--deepseek-config",
            "tiny-dense",
            "--corpus",
            "c",
            "--tokenizer",
            "t",
            "--out",
            "o",
        ]))
        .unwrap_err();
        match err {
            DsV4PretrainError::UnknownSku(name) => assert_eq!(name, "tiny-dense"),
            other => panic!("expected UnknownSku, got {other:?}"),
        }
    }

    #[test]
    fn requires_corpus() {
        let err = parse_args_from(vec_of(&["--tokenizer", "t", "--out", "o"])).unwrap_err();
        match err {
            DsV4PretrainError::MissingArg(name) => assert_eq!(name, "--corpus"),
            other => panic!("expected MissingArg(--corpus), got {other:?}"),
        }
    }

    #[test]
    fn dispatch_surfaces_pending_error() {
        let err = dispatch_from_args(vec_of(&[
            "--deepseek-config",
            "nano",
            "--corpus",
            "corpus.txt",
            "--tokenizer",
            "tokenizer.json",
            "--out",
            "/tmp/out",
        ]))
        .unwrap_err();
        // Surface should clearly cite the substrate plan so callers know
        // why the dispatch errors out today.
        assert!(err.contains("autograd"));
        assert!(err.contains("substrate-plan") || err.contains("substrate plan"));
    }
}
