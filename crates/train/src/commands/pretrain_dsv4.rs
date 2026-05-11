// DSV4 V4 checkpoint training bootstrap.
//
// The only supported DeepSeek train target is the local HF-compatible
// DeepseekV4ForCausalLM 1B init checkpoint. This command validates that
// checkpoint, tokenizer, and corpus shape, then publishes an explicit
// step_000000 seed run directory. It intentionally does not keep the deleted
// V3/nano random-init training path alive.

use std::{
    collections::{BTreeSet, HashSet},
    fs,
    io::Read,
    path::{Path, PathBuf},
};

use deepseek_spec::{
    DeepSeekConfigError, DeepSeekV4AttentionTensorNames, DeepSeekV4Config,
    DeepSeekV4HyperConnectionTensorNames, DeepSeekV4MoeTensorNames, DeepSeekV4MtpTensorNames,
};
use serde::Serialize;
use thiserror::Error;

use crate::{
    checkpoint::publish_latest_after_weights,
    cli_args::{ArgError, BackendChoice, SaveDtype, next_value, parse_value},
    tokenizer::ChatTokenizer,
};

const DEFAULT_MODEL_DIR: &str = "infer/models/dsv4-mini-1B-init";

/// Parsed CLI for `arle train pretrain-dsv4`.
#[derive(Debug, Clone)]
pub struct CliArgs {
    pub model: PathBuf,
    pub corpus: PathBuf,
    pub tokenizer: Option<PathBuf>,
    pub out: PathBuf,
    pub seed: u64,
    pub steps: usize,
    pub batch: usize,
    pub seq: usize,
    pub lr: f32,
    pub log_every: usize,
    pub save_every: usize,
    pub backend: BackendChoice,
    pub save_dtype: SaveDtype,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            model: PathBuf::from(DEFAULT_MODEL_DIR),
            corpus: PathBuf::new(),
            tokenizer: None,
            out: PathBuf::new(),
            seed: 0xC0FFEE,
            steps: 10,
            batch: 1,
            seq: 128,
            lr: 1.0e-5,
            log_every: 1,
            save_every: 10,
            backend: BackendChoice::Cpu,
            save_dtype: SaveDtype::Bf16,
        }
    }
}

#[derive(Debug, Error)]
pub enum DsV4PretrainError {
    #[error("missing required argument: {0}")]
    MissingArg(&'static str),
    #[error("argument `{flag}` requires a value")]
    MissingValue { flag: String },
    #[error("argument `{flag}` value `{value}` is not a valid {kind}")]
    InvalidValue {
        flag: String,
        value: String,
        kind: &'static str,
    },
    #[error(transparent)]
    Arg(#[from] ArgError),
    #[error(transparent)]
    Config(#[from] DeepSeekConfigError),
    #[error("{0}")]
    Custom(String),
}

#[derive(Debug, Serialize)]
struct Dsv4TrainBootstrapManifest {
    model_dir: String,
    config: String,
    tokenizer: String,
    corpus: String,
    checkpoint: String,
    tensor_count: usize,
    required_tensor_count: usize,
    corpus_tokens: usize,
    requested_steps: usize,
    batch: usize,
    seq: usize,
    lr: f32,
    seed: u64,
    backend: String,
    save_dtype: String,
    status: String,
}

pub fn parse_args_from<I>(args: I) -> Result<CliArgs, DsV4PretrainError>
where
    I: IntoIterator<Item = String>,
{
    let mut args_out = CliArgs::default();
    let mut iter = args.into_iter();

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--model" => {
                args_out.model =
                    PathBuf::from(iter.next().ok_or_else(|| DsV4PretrainError::MissingValue {
                        flag: arg.to_string(),
                    })?);
            }
            "--deepseek-config" => {
                let value = iter.next().ok_or_else(|| DsV4PretrainError::MissingValue {
                    flag: arg.to_string(),
                })?;
                if value != "v4-1b-init" && value != "dsv4-mini-1b-init" {
                    return Err(DsV4PretrainError::InvalidValue {
                        flag: arg.to_string(),
                        value,
                        kind: "v4-1b-init",
                    });
                }
            }
            "--corpus" => {
                args_out.corpus =
                    PathBuf::from(iter.next().ok_or_else(|| DsV4PretrainError::MissingValue {
                        flag: arg.to_string(),
                    })?);
            }
            "--tokenizer" => {
                args_out.tokenizer = Some(PathBuf::from(iter.next().ok_or_else(|| {
                    DsV4PretrainError::MissingValue {
                        flag: arg.to_string(),
                    }
                })?));
            }
            "--out" => {
                args_out.out =
                    PathBuf::from(iter.next().ok_or_else(|| DsV4PretrainError::MissingValue {
                        flag: arg.to_string(),
                    })?);
            }
            "--seed" => {
                let value = iter.next().ok_or_else(|| DsV4PretrainError::MissingValue {
                    flag: arg.to_string(),
                })?;
                args_out.seed =
                    value
                        .parse::<u64>()
                        .map_err(|_| DsV4PretrainError::InvalidValue {
                            flag: arg.to_string(),
                            value,
                            kind: "u64",
                        })?;
            }
            "--steps" => args_out.steps = parse_value(&arg, next_value(&mut iter, &arg)?)?,
            "--batch" => args_out.batch = parse_value(&arg, next_value(&mut iter, &arg)?)?,
            "--seq" => args_out.seq = parse_value(&arg, next_value(&mut iter, &arg)?)?,
            "--lr" => args_out.lr = parse_value(&arg, next_value(&mut iter, &arg)?)?,
            "--log-every" => args_out.log_every = parse_value(&arg, next_value(&mut iter, &arg)?)?,
            "--save-every" => {
                args_out.save_every = parse_value(&arg, next_value(&mut iter, &arg)?)?
            }
            "--backend" => {
                let value = next_value(&mut iter, &arg)?;
                args_out.backend = value
                    .parse()
                    .map_err(|_| ArgError::InvalidValue { flag: arg, value })?;
            }
            "--save-dtype" => {
                let value = next_value(&mut iter, &arg)?;
                args_out.save_dtype = value
                    .parse()
                    .map_err(|_| ArgError::InvalidValue { flag: arg, value })?;
            }
            _ => {}
        }
    }

    if args_out.corpus.as_os_str().is_empty() {
        return Err(DsV4PretrainError::MissingArg("--corpus"));
    }
    if args_out.out.as_os_str().is_empty() {
        return Err(DsV4PretrainError::MissingArg("--out"));
    }
    Ok(args_out)
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
    fs::create_dir_all(&args.out).map_err(|err| {
        DsV4PretrainError::Custom(format!("create output dir {}: {err}", args.out.display()))
    })?;

    let config_path = args.model.join("config.json");
    let cfg = DeepSeekV4Config::from_json_file(&config_path)?;
    let tokenizer_path = args
        .tokenizer
        .clone()
        .unwrap_or_else(|| args.model.join("tokenizer.json"));
    let tokenizer = ChatTokenizer::from_file(&tokenizer_path)
        .map_err(|err| DsV4PretrainError::Custom(format!("tokenizer: {err}")))?;
    if tokenizer.vocab_size() != cfg.vocab_size {
        return Err(DsV4PretrainError::Custom(format!(
            "tokenizer vocab {} does not match DSV4 config vocab {}",
            tokenizer.vocab_size(),
            cfg.vocab_size
        )));
    }
    if args.seq > cfg.max_position_embeddings {
        return Err(DsV4PretrainError::Custom(format!(
            "--seq {} exceeds DSV4 context {}",
            args.seq, cfg.max_position_embeddings
        )));
    }

    let available = safetensor_names(&args.model)?;
    let required = required_v4_tensor_names(&cfg);
    let missing = required
        .iter()
        .filter(|name| !available.contains(name.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(DsV4PretrainError::Custom(format!(
            "{} is missing {} required DSV4 tensor(s): {}",
            args.model.display(),
            missing.len(),
            missing.join(", ")
        )));
    }

    let text = fs::read_to_string(&args.corpus).map_err(|err| {
        DsV4PretrainError::Custom(format!("read corpus {}: {err}", args.corpus.display()))
    })?;
    let token_ids = tokenizer
        .encode(&text, false)
        .map_err(|err| DsV4PretrainError::Custom(format!("tokenize corpus: {err}")))?;
    if token_ids.len() <= args.seq {
        return Err(DsV4PretrainError::Custom(format!(
            "corpus has {} tokens but --seq is {}; need at least seq+1 tokens",
            token_ids.len(),
            args.seq
        )));
    }

    let step_dir = args.out.join("step_000000");
    fs::create_dir_all(&step_dir).map_err(|err| {
        DsV4PretrainError::Custom(format!(
            "create checkpoint dir {}: {err}",
            step_dir.display()
        ))
    })?;
    fs::copy(&config_path, step_dir.join("config.json")).map_err(|err| {
        DsV4PretrainError::Custom(format!("copy {}: {err}", config_path.display()))
    })?;
    fs::copy(&tokenizer_path, step_dir.join("tokenizer.json")).map_err(|err| {
        DsV4PretrainError::Custom(format!("copy {}: {err}", tokenizer_path.display()))
    })?;
    link_or_copy_weights(
        &args.model.join("model.safetensors"),
        &step_dir.join("model.safetensors"),
    )?;

    let manifest = Dsv4TrainBootstrapManifest {
        model_dir: args.model.display().to_string(),
        config: step_dir.join("config.json").display().to_string(),
        tokenizer: step_dir.join("tokenizer.json").display().to_string(),
        corpus: args.corpus.display().to_string(),
        checkpoint: step_dir.join("model.safetensors").display().to_string(),
        tensor_count: available.len(),
        required_tensor_count: required.len(),
        corpus_tokens: token_ids.len(),
        requested_steps: args.steps,
        batch: args.batch,
        seq: args.seq,
        lr: args.lr,
        seed: args.seed,
        backend: args.backend.as_str().to_string(),
        save_dtype: format!("{:?}", args.save_dtype),
        status: "seeded-v4-1b-init-checkpoint; optimizer update path pending V4 autograd/runtime forward"
            .to_string(),
    };
    fs::write(
        step_dir.join("dsv4_train_manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .map_err(|err| {
        DsV4PretrainError::Custom(format!(
            "write dsv4_train_manifest.json in {}: {err}",
            step_dir.display()
        ))
    })?;
    publish_latest_after_weights(&args.out, "step_000000").map_err(|err| {
        DsV4PretrainError::Custom(format!("publish latest DSV4 checkpoint: {err}"))
    })?;

    println!(
        "[pretrain-dsv4] model={} tokens={} tensors={}/{} out={} status=seeded-v4-1b-init",
        args.model.display(),
        token_ids.len(),
        required.len(),
        available.len(),
        args.out.display()
    );
    Ok(())
}

fn link_or_copy_weights(src: &Path, dst: &Path) -> Result<(), DsV4PretrainError> {
    if dst.exists() {
        fs::remove_file(dst).map_err(|err| {
            DsV4PretrainError::Custom(format!("remove existing {}: {err}", dst.display()))
        })?;
    }
    fs::hard_link(src, dst)
        .or_else(|_| fs::copy(src, dst).map(|_| ()))
        .map_err(|err| {
            DsV4PretrainError::Custom(format!(
                "link or copy model weights {} -> {}: {err}",
                src.display(),
                dst.display()
            ))
        })
}

fn safetensor_names(model_path: &Path) -> Result<HashSet<String>, DsV4PretrainError> {
    let mut names = HashSet::new();
    for path in safetensor_paths(model_path)? {
        let mut file = fs::File::open(&path)
            .map_err(|err| DsV4PretrainError::Custom(format!("open {}: {err}", path.display())))?;
        let mut len_bytes = [0_u8; 8];
        file.read_exact(&mut len_bytes).map_err(|err| {
            DsV4PretrainError::Custom(format!(
                "read safetensors header len {}: {err}",
                path.display()
            ))
        })?;
        let header_len: usize = u64::from_le_bytes(len_bytes)
            .try_into()
            .map_err(|_| DsV4PretrainError::Custom("safetensors header length overflow".into()))?;
        let mut header = vec![0_u8; header_len];
        file.read_exact(&mut header).map_err(|err| {
            DsV4PretrainError::Custom(format!("read safetensors header {}: {err}", path.display()))
        })?;
        let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&header)
            .map_err(|err| {
                DsV4PretrainError::Custom(format!(
                    "parse safetensors header {}: {err}",
                    path.display()
                ))
            })?;
        for name in header.keys() {
            if name != "__metadata__" {
                names.insert(name.clone());
            }
        }
    }
    Ok(names)
}

fn safetensor_paths(model_path: &Path) -> Result<Vec<PathBuf>, DsV4PretrainError> {
    let index_path = model_path.join("model.safetensors.index.json");
    if index_path.exists() {
        let index_content = fs::read_to_string(&index_path).map_err(|err| {
            DsV4PretrainError::Custom(format!("read {}: {err}", index_path.display()))
        })?;
        let index: serde_json::Value = serde_json::from_str(&index_content).map_err(|err| {
            DsV4PretrainError::Custom(format!("parse {}: {err}", index_path.display()))
        })?;
        let weight_map = index["weight_map"].as_object().ok_or_else(|| {
            DsV4PretrainError::Custom(format!("{} missing weight_map", index_path.display()))
        })?;
        let mut files = BTreeSet::new();
        for shard in weight_map.values() {
            let shard = shard.as_str().ok_or_else(|| {
                DsV4PretrainError::Custom(format!(
                    "{} has non-string shard path",
                    index_path.display()
                ))
            })?;
            files.insert(model_path.join(shard));
        }
        return Ok(files.into_iter().collect());
    }

    let single = model_path.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    Err(DsV4PretrainError::Custom(format!(
        "{} has no model.safetensors checkpoint",
        model_path.display()
    )))
}

fn push_hc(out: &mut Vec<String>, names: &DeepSeekV4HyperConnectionTensorNames) {
    out.push(names.base.clone());
    out.push(names.mix_fn.clone());
    out.push(names.scale.clone());
}

fn push_attention(out: &mut Vec<String>, names: &DeepSeekV4AttentionTensorNames) {
    out.push(names.wq_a.clone());
    out.push(names.q_norm.clone());
    out.push(names.wq_b.clone());
    out.push(names.wkv.clone());
    out.push(names.kv_norm.clone());
    out.push(names.wo_a.clone());
    out.push(names.wo_b.clone());
    out.push(names.attn_sink.clone());
    if let Some(compressor) = &names.compressor {
        out.push(compressor.wkv.clone());
        out.push(compressor.wgate.clone());
        out.push(compressor.ape.clone());
        out.push(compressor.norm.clone());
    }
    if let Some(indexer) = &names.indexer {
        out.push(indexer.wq_b.clone());
        out.push(indexer.weights_proj.clone());
        out.push(indexer.compressor.wkv.clone());
        out.push(indexer.compressor.wgate.clone());
        out.push(indexer.compressor.ape.clone());
        out.push(indexer.compressor.norm.clone());
    }
}

fn push_moe(out: &mut Vec<String>, config: &DeepSeekV4Config, names: &DeepSeekV4MoeTensorNames) {
    out.push(names.gate_weight.clone());
    if let Some(gate_bias) = &names.gate_bias {
        out.push(gate_bias.clone());
    }
    if let Some(gate_tid2eid) = &names.gate_tid2eid {
        out.push(gate_tid2eid.clone());
    }
    for expert_idx in 0..config.n_routed_experts {
        let expert = names.expert(expert_idx);
        out.push(expert.w1);
        out.push(expert.w2);
        out.push(expert.w3);
    }
    if let Some(shared) = &names.shared_experts {
        out.push(shared.w1.clone());
        out.push(shared.w2.clone());
        out.push(shared.w3.clone());
    }
}

fn push_mtp(out: &mut Vec<String>, config: &DeepSeekV4Config, names: &DeepSeekV4MtpTensorNames) {
    out.push(names.enorm.clone());
    out.push(names.hnorm.clone());
    out.push(names.e_proj.clone());
    out.push(names.h_proj.clone());
    out.push(names.attn_norm.clone());
    out.push(names.ffn_norm.clone());
    out.push(names.norm.clone());
    push_hc(out, &names.hc_attn);
    push_hc(out, &names.hc_ffn);
    push_hc(out, &names.hc_head);
    push_attention(out, &names.attn);
    push_moe(out, config, &names.ffn);
}

fn required_v4_tensor_names(config: &DeepSeekV4Config) -> Vec<String> {
    let mut out = Vec::new();
    let top = config.tensor_names();
    out.push(top.embed_tokens().to_string());
    out.push(top.norm().to_string());
    out.push(top.lm_head().to_string());
    push_hc(&mut out, &top.head_hc());

    for layer_idx in 0..config.num_hidden_layers {
        let layer = config.layer_tensor_names(layer_idx);
        out.push(layer.attn_norm);
        out.push(layer.ffn_norm);
        push_hc(&mut out, &layer.hc_attn);
        push_hc(&mut out, &layer.hc_ffn);
        push_attention(&mut out, &layer.attn);
        push_moe(&mut out, config, &layer.ffn);
    }

    for mtp_idx in 0..config.num_nextn_predict_layers {
        let mtp = config.mtp_tensor_names(mtp_idx);
        push_mtp(&mut out, config, &mtp);
    }

    out.sort();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parses_minimal_v4_invocation() {
        let parsed =
            parse_args_from(vec_of(&["--corpus", "corpus.txt", "--out", "/tmp/dsv4-v4"])).unwrap();

        assert_eq!(parsed.model, PathBuf::from(DEFAULT_MODEL_DIR));
        assert_eq!(parsed.corpus, PathBuf::from("corpus.txt"));
        assert_eq!(parsed.tokenizer, None);
        assert_eq!(parsed.out, PathBuf::from("/tmp/dsv4-v4"));
        assert_eq!(parsed.steps, 10);
    }

    #[test]
    fn parses_training_bootstrap_knobs() {
        let parsed = parse_args_from(vec_of(&[
            "--model",
            "infer/models/dsv4-mini-1B-init",
            "--corpus",
            "corpus.txt",
            "--tokenizer",
            "tokenizer.json",
            "--out",
            "/tmp/out",
            "--steps",
            "2",
            "--batch",
            "3",
            "--seq",
            "16",
            "--lr",
            "0.001",
            "--save-every",
            "1",
            "--backend",
            "cpu",
            "--save-dtype",
            "f32",
        ]))
        .unwrap();
        assert_eq!(parsed.steps, 2);
        assert_eq!(parsed.batch, 3);
        assert_eq!(parsed.seq, 16);
        assert_eq!(parsed.lr, 0.001);
        assert_eq!(parsed.save_every, 1);
        assert_eq!(parsed.backend, BackendChoice::Cpu);
        assert_eq!(parsed.save_dtype, SaveDtype::F32);
        assert_eq!(parsed.tokenizer, Some(PathBuf::from("tokenizer.json")));
    }

    #[test]
    fn accepts_legacy_v4_alias_but_rejects_old_skus() {
        parse_args_from(vec_of(&[
            "--deepseek-config",
            "v4-1b-init",
            "--corpus",
            "c",
            "--out",
            "o",
        ]))
        .unwrap();
        let err = parse_args_from(vec_of(&[
            "--deepseek-config",
            "nano",
            "--corpus",
            "c",
            "--out",
            "o",
        ]))
        .unwrap_err();
        assert!(matches!(err, DsV4PretrainError::InvalidValue { .. }));
    }

    #[test]
    fn requires_corpus() {
        let err = parse_args_from(vec_of(&["--out", "o"])).unwrap_err();
        match err {
            DsV4PretrainError::MissingArg(name) => assert_eq!(name, "--corpus"),
            other => panic!("expected MissingArg(--corpus), got {other:?}"),
        }
    }
}
