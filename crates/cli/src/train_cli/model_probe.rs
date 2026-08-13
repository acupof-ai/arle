use std::{
    fs,
    path::{Path, PathBuf},
    process::ExitCode,
};

use anyhow::{Context, Result, bail};
use deepseek_spec::{DeepSeekV4AttentionMode, DeepSeekV4Config};
use qwen35_spec::{LayerType, Qwen35Config};
use serde::Serialize;
use train::model_family::{ModelFamily, resolve_model_family};

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use crate::args::{ModelArgs, ModelCommand, ModelDownloadArgs, ModelSourceArg};

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
pub(crate) fn run_model(model: ModelArgs) -> ExitCode {
    match model.command {
        ModelCommand::Download(args) => run_model_download(args),
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn run_model_download(args: ModelDownloadArgs) -> ExitCode {
    let source_label = match args.source {
        ModelSourceArg::Hf => "hf",
        ModelSourceArg::Modelscope => "modelscope",
    };
    if args.render.dry_run {
        if args.render.json {
            println!(
                "{}",
                serde_json::json!({
                    "command": "model download",
                    "argv": [args.model_id],
                    "source": source_label,
                })
            );
        } else {
            println!("command model download");
            println!("argv {}", args.model_id);
            println!("source {source_label}");
        }
        return ExitCode::SUCCESS;
    }
    let result = match args.source {
        ModelSourceArg::Hf => crate::download::download_model_with_progress(&args.model_id),
        ModelSourceArg::Modelscope => {
            crate::modelscope::download_model_from_modelscope_with_progress(&args.model_id)
        }
    };
    match result {
        Ok(path) => {
            eprintln!(
                "[ARLE model download] downloaded ({source_label}) to: {}",
                path.display()
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("[ARLE model download] error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

pub(super) fn inspect_model_source(source: &Path, allow_download: bool) -> Result<ModelInspection> {
    let raw_source = source.display().to_string();
    let resolved_dir = if allow_download {
        Some(resolve_model_dir_allow_download(source)?)
    } else {
        resolve_model_dir_local_only(source)
    };
    let notes: Vec<String> = (!allow_download && resolved_dir.is_none())
        .then(|| "model source is not local/cached; dry-run skipped remote resolution".to_string())
        .into_iter()
        .collect();
    let summary = resolved_dir
        .as_deref()
        .map(inspect_resolved_model_dir)
        .transpose()?;

    Ok(ModelInspection {
        source: raw_source,
        resolved_dir: resolved_dir.as_ref().map(|path| path.display().to_string()),
        config_path: summary.as_ref().map(|s| s.config_path.clone()),
        tokenizer_path: summary.as_ref().and_then(|s| s.tokenizer_path.clone()),
        generation_config_path: summary
            .as_ref()
            .and_then(|s| s.generation_config_path.clone()),
        family: summary.as_ref().map(|s| s.family.clone()),
        notes,
    })
}

pub(super) fn inspect_resolved_model_dir(model_dir: &Path) -> Result<ModelDirSummary> {
    let config_path = model_dir.join("config.json");
    let config_value: serde_json::Value = serde_json::from_str(&fs::read_to_string(&config_path)?)
        .with_context(|| {
            format!(
                "reading model inspection config from {}",
                config_path.display()
            )
        })?;
    let is_deepseek_v4 = config_value
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|model_type| model_type == "deepseek_v4")
        || config_value
            .get("architectures")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|architectures| {
                architectures
                    .iter()
                    .any(|arch| arch.as_str() == Some("DeepseekV4ForCausalLM"))
            });
    if is_deepseek_v4 {
        let cfg = DeepSeekV4Config::from_json_value(&config_value)?;
        return Ok(ModelDirSummary {
            family: "deepseek-v4".to_string(),
            config: ResolvedModelConfig::DeepSeekV4,
            config_path: config_path.display().to_string(),
            tokenizer_path: existing_display_path(model_dir.join("tokenizer.json")),
            generation_config_path: existing_display_path(model_dir.join("generation_config.json")),
            vocab_size: cfg.vocab_size,
            hidden_size: cfg.hidden_size,
            param_count: deepseek_v4_param_count(&cfg),
        });
    }

    let family = match resolve_model_family(&config_path, ModelFamily::Auto)? {
        ModelFamily::Qwen35 => "qwen35",
        ModelFamily::Auto => unreachable!("auto must resolve to a concrete family"),
    };
    match family {
        "qwen35" => {
            let cfg = Qwen35Config::from_json_file(&config_path)?;
            Ok(ModelDirSummary {
                family: "qwen35".to_string(),
                config: ResolvedModelConfig::Qwen35(Box::new(cfg.clone())),
                config_path: config_path.display().to_string(),
                tokenizer_path: existing_display_path(model_dir.join("tokenizer.json")),
                generation_config_path: existing_display_path(
                    model_dir.join("generation_config.json"),
                ),
                vocab_size: cfg.vocab_size,
                hidden_size: cfg.hidden_size,
                param_count: qwen35_param_count(&cfg),
            })
        }
        _ => unreachable!("family resolver returned an unknown family"),
    }
}

fn resolve_model_dir_allow_download(source: &Path) -> Result<PathBuf> {
    let source_text = source.display().to_string();
    infer_util::hf_hub::resolve_model_path(&source_text)
        .with_context(|| format!("resolving model source {source_text}"))
}

fn resolve_model_dir_local_only(source: &Path) -> Option<PathBuf> {
    let source_text = source.display().to_string();
    infer_util::hf_hub::resolve_local_model_path(&source_text)
}

pub(crate) fn resolve_local_tokenizer_path(source: &Path) -> Result<PathBuf> {
    if source.is_file() {
        return Ok(source.to_path_buf());
    }
    if source.is_dir() {
        let candidate = source.join("tokenizer.json");
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    let source_text = source.display().to_string();
    if let Some(model_dir) = infer_util::hf_hub::resolve_local_model_path(&source_text) {
        let candidate = model_dir.join("tokenizer.json");
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    bail!(
        "tokenizer source {} must be tokenizer.json or a local model dir containing tokenizer.json",
        source.display()
    );
}

pub(super) fn qwen35_param_count(cfg: &Qwen35Config) -> u64 {
    let embed = mul_u64(cfg.vocab_size, cfg.hidden_size);
    let lm_head = if cfg.tie_word_embeddings { 0 } else { embed };
    let common = mul_u64(2, cfg.hidden_size)
        + mul_u64(cfg.hidden_size, cfg.intermediate_size) * 2
        + mul_u64(cfg.intermediate_size, cfg.hidden_size);
    let attention = cfg
        .layer_types
        .iter()
        .map(|layer_type| match layer_type {
            LayerType::FullAttention => {
                mul_u64(cfg.hidden_size, cfg.full_attn_q_proj_dim())
                    + mul_u64(cfg.hidden_size, cfg.full_attn_kv_dim()) * 2
                    + mul_u64(cfg.full_attn_q_dim(), cfg.hidden_size)
                    + mul_u64(2, cfg.head_dim)
            }
            LayerType::LinearAttention => {
                mul_u64(cfg.hidden_size, cfg.linear_attn_qkv_dim())
                    + mul_u64(cfg.hidden_size, cfg.linear_attn_z_dim())
                    + mul_u64(cfg.hidden_size, cfg.linear_num_value_heads) * 2
                    + mul_u64(cfg.linear_attn_qkv_dim(), cfg.linear_conv_kernel_dim)
                    + mul_u64(2, cfg.linear_num_value_heads)
                    + cfg.linear_value_head_dim as u64
                    + mul_u64(cfg.linear_attn_z_dim(), cfg.hidden_size)
            }
        })
        .sum::<u64>();
    embed
        + lm_head
        + (cfg.num_hidden_layers as u64).saturating_mul(common)
        + attention
        + cfg.hidden_size as u64
}

fn deepseek_v4_param_count(cfg: &DeepSeekV4Config) -> u64 {
    let embed = mul_u64(cfg.vocab_size, cfg.hidden_size);
    let lm_head = if cfg.tie_word_embeddings { 0 } else { embed };
    let hc_mix = (2 + cfg.hc_mult) * cfg.hc_mult;
    let hc_flat = cfg.hc_mult * cfg.hidden_size;
    let head_hc = mul_u64(cfg.hc_mult, hc_flat) + cfg.hc_mult as u64 + 1;
    let per_hc = mul_u64(hc_mix, hc_flat) + hc_mix as u64 + 3;
    let heads_per_group = cfg.num_attention_heads / cfg.o_groups;
    let base_attn = mul_u64(cfg.q_lora_rank, cfg.hidden_size)
        + cfg.q_lora_rank as u64
        + mul_u64(cfg.num_attention_heads * cfg.head_dim, cfg.q_lora_rank)
        + mul_u64(cfg.head_dim, cfg.hidden_size)
        + cfg.head_dim as u64
        + mul_u64(
            cfg.o_groups * cfg.o_lora_rank,
            heads_per_group * cfg.head_dim,
        )
        + mul_u64(cfg.hidden_size, cfg.o_groups * cfg.o_lora_rank)
        + cfg.num_attention_heads as u64;
    let expert = mul_u64(cfg.moe_intermediate_size, cfg.hidden_size) * 2
        + mul_u64(cfg.hidden_size, cfg.moe_intermediate_size);
    let routed_experts = (cfg.n_routed_experts as u64).saturating_mul(expert);
    let shared_experts = if cfg.n_shared_experts == 0 {
        0
    } else {
        let shared_intermediate = cfg.moe_intermediate_size * cfg.n_shared_experts;
        mul_u64(shared_intermediate, cfg.hidden_size) * 2
            + mul_u64(cfg.hidden_size, shared_intermediate)
    };
    let gate_bias_or_hash = cfg
        .n_routed_experts
        .max(cfg.vocab_size * cfg.num_experts_per_tok);
    let moe = mul_u64(cfg.n_routed_experts, cfg.hidden_size)
        + gate_bias_or_hash as u64
        + routed_experts
        + shared_experts;

    let layers = cfg
        .compress_ratios
        .iter()
        .copied()
        .map(|compress_ratio| {
            let compressor = cfg
                .compressor_shape(compress_ratio)
                .map(|shape| {
                    mul_u64(shape.wkv_rows, shape.wkv_cols)
                        + mul_u64(shape.wgate_rows, shape.wgate_cols)
                        + mul_u64(shape.ape_rows, shape.ape_cols)
                        + shape.norm_len as u64
                })
                .unwrap_or(0);
            let indexer = if cfg.attention_mode_for_compress_ratio(compress_ratio)
                == DeepSeekV4AttentionMode::CompressedSparse
            {
                let shape = cfg
                    .indexer_shape(compress_ratio)
                    .expect("CSA layer has indexer shape");
                let compressor = shape
                    .compressor
                    .as_ref()
                    .expect("CSA indexer always has a compressor");
                mul_u64(shape.wq_b_rows, shape.wq_b_cols)
                    + mul_u64(shape.weights_proj_rows, shape.weights_proj_cols)
                    + mul_u64(compressor.wkv_rows, compressor.wkv_cols)
                    + mul_u64(compressor.wgate_rows, compressor.wgate_cols)
                    + mul_u64(compressor.ape_rows, compressor.ape_cols)
                    + compressor.norm_len as u64
            } else {
                0
            };
            mul_u64(2, cfg.hidden_size) + per_hc * 2 + base_attn + compressor + indexer + moe
        })
        .sum::<u64>();

    let mtp = (cfg.num_nextn_predict_layers as u64).saturating_mul(
        mul_u64(7, cfg.hidden_size)
            + mul_u64(2, cfg.hidden_size * cfg.hidden_size)
            + per_hc * 2
            + head_hc
            + base_attn
            + moe,
    );

    embed + lm_head + cfg.hidden_size as u64 + head_hc + layers + mtp
}

pub(super) fn mul_u64(lhs: usize, rhs: usize) -> u64 {
    (lhs as u64).saturating_mul(rhs as u64)
}

fn existing_display_path(path: PathBuf) -> Option<String> {
    path.is_file().then(|| path.display().to_string())
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct ModelInspection {
    source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    resolved_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    config_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokenizer_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    family: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    notes: Vec<String>,
}

impl ModelInspection {
    pub(super) fn local_dir_path(&self) -> Option<PathBuf> {
        self.resolved_dir.as_ref().map(PathBuf::from)
    }
}

#[derive(Debug)]
pub(super) struct ModelDirSummary {
    pub(super) family: String,
    pub(super) config: ResolvedModelConfig,
    config_path: String,
    pub(super) tokenizer_path: Option<String>,
    generation_config_path: Option<String>,
    pub(super) vocab_size: usize,
    pub(super) hidden_size: usize,
    pub(super) param_count: u64,
}

#[derive(Debug)]
pub(super) enum ResolvedModelConfig {
    Qwen35(Box<Qwen35Config>),
    DeepSeekV4,
}
