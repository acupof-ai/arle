use std::{
    fs, io,
    path::{Path, PathBuf},
};

use autograd::AutogradError;
use qwen35_spec::{LayerType, Qwen35Config};
use serde_json::json;
use thiserror::Error;
use tokenizers::{Tokenizer, models::wordlevel::WordLevel};

use crate::checkpoint::publish_latest_after_weights;

#[derive(Debug, Error)]
pub enum Qwen35CheckpointError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Autograd(#[from] AutogradError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Custom(String),
}

pub enum ConfigJsonSource<'a> {
    CopyFrom(&'a Path),
    Synthesize {
        cfg: &'a Qwen35Config,
        torch_dtype: &'static str,
    },
}

pub enum GenerationConfigSource<'a> {
    CopyFrom(&'a Path),
    Synthesize {
        bos_token_id: Option<u32>,
        eos_token_id: u32,
    },
    CopyOrSynthesize {
        source_path: &'a Path,
        fallback_config_path: &'a Path,
    },
}

pub struct Qwen35StepCheckpoint<'a> {
    pub out_dir: &'a Path,
    pub step: usize,
    pub tokenizer_path: Option<&'a Path>,
    pub config_json: ConfigJsonSource<'a>,
    pub generation_config: GenerationConfigSource<'a>,
}

pub fn save_step_checkpoint<F>(
    spec: Qwen35StepCheckpoint<'_>,
    save_weights: F,
) -> Result<PathBuf, Qwen35CheckpointError>
where
    F: FnOnce(&Path) -> Result<(), Qwen35CheckpointError>,
{
    let synth_tokenizer_cfg = match &spec.config_json {
        ConfigJsonSource::Synthesize { cfg, .. } => Some(*cfg),
        ConfigJsonSource::CopyFrom(_) => None,
    };
    let step_basename = format!("step_{:06}", spec.step);
    let step_dir = spec.out_dir.join(&step_basename);
    fs::create_dir_all(&step_dir)?;

    write_config_json(step_dir.join("config.json"), spec.config_json)?;
    let tokenizer_out = step_dir.join("tokenizer.json");
    if let Some(tokenizer_path) = spec.tokenizer_path {
        fs::copy(tokenizer_path, &tokenizer_out)?;
    } else if let Some(cfg) = synth_tokenizer_cfg {
        write_synth_tokenizer(&tokenizer_out, cfg)?;
    }
    write_generation_config(
        step_dir.join("generation_config.json"),
        spec.generation_config,
    )?;

    let weights_path = step_dir.join("model.safetensors");
    save_weights(&weights_path)?;
    publish_latest_after_weights(spec.out_dir, &step_basename)?;
    Ok(step_dir)
}

fn write_config_json(
    target_path: PathBuf,
    source: ConfigJsonSource<'_>,
) -> Result<(), Qwen35CheckpointError> {
    match source {
        ConfigJsonSource::CopyFrom(source_path) => {
            fs::copy(source_path, target_path)?;
        }
        ConfigJsonSource::Synthesize { cfg, torch_dtype } => {
            let text_config = json!({
                "hidden_size": cfg.hidden_size,
                "intermediate_size": cfg.intermediate_size,
                "num_hidden_layers": cfg.num_hidden_layers,
                "num_attention_heads": cfg.num_attention_heads,
                "num_key_value_heads": cfg.num_key_value_heads,
                "head_dim": cfg.head_dim,
                "vocab_size": cfg.vocab_size,
                "rms_norm_eps": cfg.rms_norm_eps,
                "layer_types": cfg
                    .layer_types
                    .iter()
                    .map(layer_type_name)
                    .collect::<Vec<_>>(),
                "linear_conv_kernel_dim": cfg.linear_conv_kernel_dim,
                "linear_key_head_dim": cfg.linear_key_head_dim,
                "linear_num_key_heads": cfg.linear_num_key_heads,
                "linear_num_value_heads": cfg.linear_num_value_heads,
                "linear_value_head_dim": cfg.linear_value_head_dim,
                "rope_parameters": {
                    "rope_theta": cfg.rope_theta,
                    "partial_rotary_factor": cfg.partial_rotary_factor,
                },
                "eos_token_id": cfg.eos_token_id,
                "bos_token_id": cfg.bos_token_id,
                "tie_word_embeddings": cfg.tie_word_embeddings,
                "max_position_embeddings": cfg.rope_cache_len_hint,
            });
            let config_json = json!({
                "architectures": ["Qwen2ForCausalLM"],
                "text_config": text_config,
                "torch_dtype": torch_dtype,
            });
            fs::write(target_path, serde_json::to_string_pretty(&config_json)?)?;
        }
    }
    Ok(())
}

fn write_generation_config(
    target_path: PathBuf,
    source: GenerationConfigSource<'_>,
) -> Result<(), Qwen35CheckpointError> {
    match source {
        GenerationConfigSource::CopyFrom(source_path) => {
            fs::copy(source_path, target_path)?;
        }
        GenerationConfigSource::Synthesize {
            bos_token_id,
            eos_token_id,
        } => {
            write_generation_config_json(target_path, bos_token_id, eos_token_id)?;
        }
        GenerationConfigSource::CopyOrSynthesize {
            source_path,
            fallback_config_path,
        } => {
            if source_path.is_file() {
                fs::copy(source_path, target_path)?;
            } else {
                let config: serde_json::Value = serde_json::from_str(&fs::read_to_string(
                    fallback_config_path,
                )?)
                .map_err(|err| {
                    Qwen35CheckpointError::Custom(format!(
                        "save checkpoint config parse error: {err}"
                    ))
                })?;
                let text = config.get("text_config").unwrap_or(&config);
                let bos_token_id = text
                    .get("bos_token_id")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|value| u32::try_from(value).ok());
                let eos_token_id = read_token_id(text, "eos_token_id", fallback_config_path)?;
                write_generation_config_json(target_path, bos_token_id, eos_token_id)?;
            }
        }
    }
    Ok(())
}

fn write_generation_config_json(
    target_path: PathBuf,
    bos_token_id: Option<u32>,
    eos_token_id: u32,
) -> Result<(), Qwen35CheckpointError> {
    let mut eos_token_ids = vec![eos_token_id];
    if let Some(bos_token_id) = bos_token_id.filter(|bos| *bos != eos_token_id) {
        eos_token_ids.push(bos_token_id);
    }
    fs::write(
        target_path,
        serde_json::to_string_pretty(&json!({
            "eos_token_id": eos_token_ids,
        }))?,
    )?;
    Ok(())
}

fn write_synth_tokenizer(
    target_path: &Path,
    cfg: &Qwen35Config,
) -> Result<(), Qwen35CheckpointError> {
    let reserved_ids = [
        cfg.bos_token_id.and_then(|id| usize::try_from(id).ok()),
        usize::try_from(cfg.eos_token_id).ok(),
    ];
    let unk_id = (0..cfg.vocab_size)
        .find(|id| {
            !reserved_ids
                .into_iter()
                .flatten()
                .any(|reserved| reserved == *id)
        })
        .unwrap_or(0);
    let unk_token = synth_token_for_id(unk_id, cfg, unk_id);
    let vocab = (0..cfg.vocab_size)
        .map(|id| {
            (
                synth_token_for_id(id, cfg, unk_id),
                u32::try_from(id).expect("vocab id fits in u32"),
            )
        })
        .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token(unk_token)
        .build()
        .map_err(|err| Qwen35CheckpointError::Custom(format!("build synth tokenizer: {err}")))?;
    let tokenizer = Tokenizer::new(model);
    tokenizer
        .save(target_path, false)
        .map_err(|err| Qwen35CheckpointError::Custom(format!("save synth tokenizer: {err}")))?;
    Ok(())
}

fn synth_token_for_id(id: usize, cfg: &Qwen35Config, unk_id: usize) -> String {
    if id == unk_id {
        return "[UNK]".into();
    }
    if cfg
        .bos_token_id
        .and_then(|token_id| usize::try_from(token_id).ok())
        == Some(id)
    {
        return "<s>".into();
    }
    if usize::try_from(cfg.eos_token_id).ok() == Some(id) {
        return "</s>".into();
    }
    format!("<tok_{id}>")
}

fn read_token_id(
    config: &serde_json::Value,
    key: &str,
    fallback_config_path: &Path,
) -> Result<u32, Qwen35CheckpointError> {
    let value = config
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            Qwen35CheckpointError::Custom(format!(
                "source config {} is missing {key}",
                fallback_config_path.display()
            ))
        })?;
    u32::try_from(value).map_err(|_| {
        Qwen35CheckpointError::Custom(format!(
            "source config {} has out-of-range {key}: {value}",
            fallback_config_path.display()
        ))
    })
}

fn layer_type_name(layer_type: &LayerType) -> &'static str {
    match layer_type {
        LayerType::FullAttention => "full_attention",
        LayerType::LinearAttention => "linear_attention",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use tokenizers::Tokenizer;

    fn dense_qwen35_config() -> Qwen35Config {
        Qwen35Config {
            hidden_size: 128,
            intermediate_size: 256,
            num_hidden_layers: 2,
            vocab_size: 1024,
            rms_norm_eps: 1.0e-6,
            stop_token_ids: vec![7],
            bos_token_id: Some(3),
            eos_token_id: 7,
            tie_word_embeddings: true,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            head_dim: 32,
            linear_num_key_heads: 4,
            linear_key_head_dim: 32,
            linear_num_value_heads: 4,
            linear_value_head_dim: 32,
            linear_conv_kernel_dim: 4,
            rope_theta: 1_000_000.0,
            partial_rotary_factor: 1.0,
            rotary_dim: 32,
            rope_cache_len_hint: Some(512),
            layer_types: vec![LayerType::FullAttention, LayerType::FullAttention],
            num_experts: 0,
            num_experts_per_tok: 0,
            decoder_sparse_step: 1,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            norm_topk_prob: true,
            mlp_only_layers: Vec::new(),
        }
    }

    #[test]
    fn save_step_checkpoint_synthesizes_qwen35_dense_config() {
        let tmp = tempdir().expect("tempdir");
        let cfg = dense_qwen35_config();

        let step_dir = save_step_checkpoint(
            Qwen35StepCheckpoint {
                out_dir: tmp.path(),
                step: 3,
                tokenizer_path: None,
                config_json: ConfigJsonSource::Synthesize {
                    cfg: &cfg,
                    torch_dtype: "float32",
                },
                generation_config: GenerationConfigSource::Synthesize {
                    bos_token_id: cfg.bos_token_id,
                    eos_token_id: cfg.eos_token_id,
                },
            },
            |weights_path| {
                fs::write(weights_path, b"weights").expect("write weights");
                Ok(())
            },
        )
        .expect("save checkpoint");

        let config_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(step_dir.join("config.json")).unwrap())
                .expect("parse config");
        assert_eq!(
            config_value["text_config"]["layer_types"],
            json!(["full_attention", "full_attention"])
        );
        assert_eq!(
            config_value["text_config"]["rope_parameters"]["partial_rotary_factor"],
            json!(1.0)
        );

        let generation_value: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(step_dir.join("generation_config.json")).unwrap(),
        )
        .expect("parse generation config");
        assert_eq!(generation_value["eos_token_id"], json!([7, 3]));

        let tokenizer =
            Tokenizer::from_file(step_dir.join("tokenizer.json")).expect("load tokenizer");
        assert_eq!(tokenizer.get_vocab_size(false), cfg.vocab_size);
    }
}
