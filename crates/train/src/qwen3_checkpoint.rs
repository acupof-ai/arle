use std::{
    fs, io,
    path::{Path, PathBuf},
};

use autograd::AutogradError;
use qwen3_spec::Qwen3Config;
use serde_json::{Value, json};
use thiserror::Error;

use crate::checkpoint::publish_latest_after_weights;

#[derive(Debug, Error)]
pub enum Qwen3CheckpointError {
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
        cfg: &'a Qwen3Config,
        bos_token_id: u32,
        eos_token_id: u32,
        torch_dtype: &'static str,
    },
}

pub enum GenerationConfigSource<'a> {
    CopyFrom(&'a Path),
    Synthesize {
        bos_token_id: u32,
        eos_token_id: u32,
    },
    CopyOrSynthesize {
        source_path: &'a Path,
        fallback_config_path: &'a Path,
    },
}

pub struct Qwen3StepCheckpoint<'a> {
    pub out_dir: &'a Path,
    pub step: usize,
    pub tokenizer_path: Option<&'a Path>,
    pub config_json: ConfigJsonSource<'a>,
    pub generation_config: GenerationConfigSource<'a>,
}

pub fn save_step_checkpoint<F>(
    spec: Qwen3StepCheckpoint<'_>,
    save_weights: F,
) -> Result<PathBuf, Qwen3CheckpointError>
where
    F: FnOnce(&Path) -> Result<(), Qwen3CheckpointError>,
{
    let step_basename = format!("step_{:06}", spec.step);
    let step_dir = spec.out_dir.join(&step_basename);
    fs::create_dir_all(&step_dir)?;

    write_config_json(step_dir.join("config.json"), spec.config_json)?;
    if let Some(tokenizer_path) = spec.tokenizer_path {
        fs::copy(tokenizer_path, step_dir.join("tokenizer.json"))?;
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
) -> Result<(), Qwen3CheckpointError> {
    match source {
        ConfigJsonSource::CopyFrom(source_path) => {
            fs::copy(source_path, target_path)?;
        }
        ConfigJsonSource::Synthesize {
            cfg,
            bos_token_id,
            eos_token_id,
            torch_dtype,
        } => {
            let config_json = json!({
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "hidden_size": cfg.hidden_size,
                "intermediate_size": cfg.intermediate_size,
                "num_hidden_layers": cfg.num_hidden_layers,
                "num_attention_heads": cfg.num_attention_heads,
                "num_key_value_heads": cfg.num_key_value_heads,
                "head_dim": cfg.head_dim,
                "vocab_size": cfg.vocab_size,
                "rms_norm_eps": cfg.rms_norm_eps,
                "rope_theta": cfg.rope_theta,
                "bos_token_id": bos_token_id,
                "eos_token_id": eos_token_id,
                "tie_word_embeddings": cfg.tie_word_embeddings,
                "max_position_embeddings": cfg.max_position_embeddings,
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
) -> Result<(), Qwen3CheckpointError> {
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
                let config: Value = serde_json::from_str(&fs::read_to_string(
                    fallback_config_path,
                )?)
                .map_err(|err| {
                    Qwen3CheckpointError::Custom(format!(
                        "save checkpoint config parse error: {err}"
                    ))
                })?;
                let bos_token_id = read_token_id(&config, "bos_token_id", fallback_config_path)?;
                let eos_token_id = read_token_id(&config, "eos_token_id", fallback_config_path)?;
                write_generation_config_json(target_path, bos_token_id, eos_token_id)?;
            }
        }
    }
    Ok(())
}

fn write_generation_config_json(
    target_path: PathBuf,
    bos_token_id: u32,
    eos_token_id: u32,
) -> Result<(), Qwen3CheckpointError> {
    let mut eos_token_ids = vec![eos_token_id];
    if bos_token_id != eos_token_id {
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

fn read_token_id(
    config: &Value,
    key: &str,
    fallback_config_path: &Path,
) -> Result<u32, Qwen3CheckpointError> {
    let value = config.get(key).and_then(Value::as_u64).ok_or_else(|| {
        Qwen3CheckpointError::Custom(format!(
            "source config {} is missing {key}",
            fallback_config_path.display()
        ))
    })?;
    u32::try_from(value).map_err(|_| {
        Qwen3CheckpointError::Custom(format!(
            "source config {} has out-of-range {key}: {value}",
            fallback_config_path.display()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn save_step_checkpoint_synthesizes_generation_config_with_stop_tokens() {
        let tmp = tempdir().expect("tempdir");
        let tokenizer_path = tmp.path().join("tokenizer.json");
        fs::write(&tokenizer_path, "{}").expect("write tokenizer");

        let cfg = Qwen3Config {
            hidden_size: 256,
            intermediate_size: 512,
            num_hidden_layers: 4,
            num_attention_heads: 8,
            num_key_value_heads: 4,
            head_dim: 32,
            vocab_size: 1024,
            rms_norm_eps: 1.0e-6,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: true,
            max_position_embeddings: 2048,
        };

        let step_dir = save_step_checkpoint(
            Qwen3StepCheckpoint {
                out_dir: tmp.path(),
                step: 7,
                tokenizer_path: Some(&tokenizer_path),
                config_json: ConfigJsonSource::Synthesize {
                    cfg: &cfg,
                    bos_token_id: 151_643,
                    eos_token_id: 151_645,
                    torch_dtype: "bfloat16",
                },
                generation_config: GenerationConfigSource::Synthesize {
                    bos_token_id: 151_643,
                    eos_token_id: 151_645,
                },
            },
            |weights_path| {
                fs::write(weights_path, b"weights").expect("write weights");
                Ok(())
            },
        )
        .expect("save checkpoint");

        let value: Value = serde_json::from_str(
            &fs::read_to_string(step_dir.join("generation_config.json"))
                .expect("read generation config"),
        )
        .expect("parse generation config");
        assert_eq!(value["eos_token_id"], json!([151_645, 151_643]));
        assert_eq!(
            fs::read_link(tmp.path().join("latest")).expect("read latest"),
            PathBuf::from("step_000007")
        );
    }

    #[test]
    fn save_step_checkpoint_copies_generation_config_when_present() {
        let tmp = tempdir().expect("tempdir");
        let model_dir = tmp.path().join("model");
        fs::create_dir_all(&model_dir).expect("create model dir");
        fs::write(
            model_dir.join("config.json"),
            r#"{"bos_token_id":3,"eos_token_id":7}"#,
        )
        .expect("write config");
        fs::write(
            model_dir.join("generation_config.json"),
            r#"{"eos_token_id":[7,3]}"#,
        )
        .expect("write generation config");
        let tokenizer_path = tmp.path().join("tokenizer.json");
        fs::write(&tokenizer_path, "{}").expect("write tokenizer");

        let step_dir = save_step_checkpoint(
            Qwen3StepCheckpoint {
                out_dir: tmp.path(),
                step: 3,
                tokenizer_path: Some(&tokenizer_path),
                config_json: ConfigJsonSource::CopyFrom(&model_dir.join("config.json")),
                generation_config: GenerationConfigSource::CopyOrSynthesize {
                    source_path: &model_dir.join("generation_config.json"),
                    fallback_config_path: &model_dir.join("config.json"),
                },
            },
            |weights_path| {
                fs::write(weights_path, b"weights").expect("write weights");
                Ok(())
            },
        )
        .expect("save checkpoint");

        let copied: Value = serde_json::from_str(
            &fs::read_to_string(step_dir.join("generation_config.json"))
                .expect("read generation config"),
        )
        .expect("parse generation config");
        assert_eq!(copied["eos_token_id"], json!([7, 3]));
    }

    #[test]
    fn save_step_checkpoint_falls_back_to_config_tokens_when_generation_config_missing() {
        let tmp = tempdir().expect("tempdir");
        let model_dir = tmp.path().join("model");
        fs::create_dir_all(&model_dir).expect("create model dir");
        fs::write(
            model_dir.join("config.json"),
            r#"{"bos_token_id":3,"eos_token_id":7}"#,
        )
        .expect("write config");
        let tokenizer_path = tmp.path().join("tokenizer.json");
        fs::write(&tokenizer_path, "{}").expect("write tokenizer");

        let step_dir = save_step_checkpoint(
            Qwen3StepCheckpoint {
                out_dir: tmp.path(),
                step: 4,
                tokenizer_path: Some(&tokenizer_path),
                config_json: ConfigJsonSource::CopyFrom(&model_dir.join("config.json")),
                generation_config: GenerationConfigSource::CopyOrSynthesize {
                    source_path: &model_dir.join("generation_config.json"),
                    fallback_config_path: &model_dir.join("config.json"),
                },
            },
            |weights_path| {
                fs::write(weights_path, b"weights").expect("write weights");
                Ok(())
            },
        )
        .expect("save checkpoint");

        let generated: Value = serde_json::from_str(
            &fs::read_to_string(step_dir.join("generation_config.json"))
                .expect("read generation config"),
        )
        .expect("parse generation config");
        assert_eq!(generated["eos_token_id"], json!([7, 3]));
    }

    #[test]
    fn save_step_checkpoint_does_not_publish_latest_before_weights_land() {
        let tmp = tempdir().expect("tempdir");
        let tokenizer_path = tmp.path().join("tokenizer.json");
        fs::write(&tokenizer_path, "{}").expect("write tokenizer");

        let cfg = Qwen3Config {
            hidden_size: 256,
            intermediate_size: 512,
            num_hidden_layers: 4,
            num_attention_heads: 8,
            num_key_value_heads: 4,
            head_dim: 32,
            vocab_size: 1024,
            rms_norm_eps: 1.0e-6,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: true,
            max_position_embeddings: 2048,
        };

        let err = save_step_checkpoint(
            Qwen3StepCheckpoint {
                out_dir: tmp.path(),
                step: 8,
                tokenizer_path: Some(&tokenizer_path),
                config_json: ConfigJsonSource::Synthesize {
                    cfg: &cfg,
                    bos_token_id: 151_643,
                    eos_token_id: 151_645,
                    torch_dtype: "bfloat16",
                },
                generation_config: GenerationConfigSource::Synthesize {
                    bos_token_id: 151_643,
                    eos_token_id: 151_645,
                },
            },
            |_weights_path| Err(Qwen3CheckpointError::Custom("weights failed".into())),
        )
        .expect_err("weights failure should abort publish");

        assert!(
            err.to_string().contains("weights failed"),
            "unexpected error: {err}"
        );
        assert!(
            !tmp.path().join("latest").exists(),
            "latest should not be published on failed weight write"
        );
    }

    #[test]
    fn save_step_checkpoint_skips_tokenizer_when_unspecified() {
        let tmp = tempdir().expect("tempdir");
        let cfg = Qwen3Config {
            hidden_size: 128,
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 32,
            vocab_size: 512,
            rms_norm_eps: 1.0e-6,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: true,
            max_position_embeddings: 1024,
        };

        let step_dir = save_step_checkpoint(
            Qwen3StepCheckpoint {
                out_dir: tmp.path(),
                step: 9,
                tokenizer_path: None,
                config_json: ConfigJsonSource::Synthesize {
                    cfg: &cfg,
                    bos_token_id: 11,
                    eos_token_id: 12,
                    torch_dtype: "float32",
                },
                generation_config: GenerationConfigSource::Synthesize {
                    bos_token_id: 11,
                    eos_token_id: 12,
                },
            },
            |weights_path| {
                fs::write(weights_path, b"weights").expect("write weights");
                Ok(())
            },
        )
        .expect("save checkpoint");

        assert!(!step_dir.join("tokenizer.json").exists());
        assert!(step_dir.join("model.safetensors").exists());
    }
}
