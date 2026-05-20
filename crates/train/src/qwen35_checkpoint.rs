use std::{
    fs, io,
    path::{Path, PathBuf},
};

use autograd::{AutogradError, SafetensorsRegistry, Tape, TensorId, TensorStore};
use qwen35_spec::{LayerType, Qwen35Config};
use serde_json::json;
use thiserror::Error;
use tokenizers::{Tokenizer, models::wordlevel::WordLevel};

use crate::{
    causal_lm::{live_tensor_ids, save_materialized_registry},
    checkpoint::write_latest_symlink,
    lora::LoraAdapterConfig,
    qwen35::Qwen35Model,
};

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

#[derive(Debug, Clone, Copy)]
pub enum Qwen35StudentWeights<'a> {
    /// Save a full model checkpoint. For LoRA students this materializes
    /// base+adapter weights under the base HF tensor names.
    FullMaterialized { bf16: bool },
    /// Save only LoRA adapter tensors as a PEFT adapter directory:
    /// `adapter_config.json` plus `adapter_model.safetensors`.
    AdapterOnly {
        bf16: bool,
        adapter_config: &'a LoraAdapterConfig,
    },
}

const MODEL_WEIGHTS_FILENAME: &str = "model.safetensors";
const ADAPTER_WEIGHTS_FILENAME: &str = "adapter_model.safetensors";
const ADAPTER_CONFIG_FILENAME: &str = "adapter_config.json";

pub fn save_step_checkpoint<F>(
    spec: Qwen35StepCheckpoint<'_>,
    save_weights: F,
) -> Result<PathBuf, Qwen35CheckpointError>
where
    F: FnOnce(&Path) -> Result<(), Qwen35CheckpointError>,
{
    save_step_checkpoint_with_artifact(spec, MODEL_WEIGHTS_FILENAME, save_weights)
}

fn save_step_checkpoint_with_artifact<F>(
    spec: Qwen35StepCheckpoint<'_>,
    artifact_filename: &'static str,
    save_artifact: F,
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
    let created_step_dir = !step_dir.exists();
    fs::create_dir_all(&step_dir)?;

    let result = (|| {
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

        let artifact_path = step_dir.join(artifact_filename);
        save_artifact(&artifact_path)?;
        publish_latest_after_artifact(spec.out_dir, &step_basename, artifact_filename)?;
        Ok(step_dir.clone())
    })();

    if result.is_err() && created_step_dir {
        let _ = fs::remove_dir_all(&step_dir);
    }
    result
}

pub fn save_qwen35_student_checkpoint<'a>(
    spec: Qwen35StepCheckpoint<'_>,
    student: &Qwen35Model,
    store: &mut TensorStore,
    tape: &mut Tape,
    weights: Qwen35StudentWeights<'a>,
) -> Result<PathBuf, Qwen35CheckpointError> {
    match weights {
        Qwen35StudentWeights::FullMaterialized { bf16 } => {
            save_step_checkpoint(spec, |weights_path| {
                save_full_materialized_weights(student, store, tape, weights_path, bf16)
            })
        }
        Qwen35StudentWeights::AdapterOnly {
            bf16,
            adapter_config,
        } => save_step_checkpoint_with_artifact(spec, ADAPTER_WEIGHTS_FILENAME, |weights_path| {
            save_adapter_only_weights(student, store, weights_path, bf16, adapter_config)
        }),
    }
}

fn publish_latest_after_artifact(
    parent: &Path,
    target_basename: &str,
    artifact_filename: &str,
) -> io::Result<()> {
    let artifact = parent.join(target_basename).join(artifact_filename);
    if !artifact.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "publish_latest_after_artifact: {} missing — refusing to publish \
                 `latest` before the final artifact lands (publish-last contract)",
                artifact.display()
            ),
        ));
    }
    write_latest_symlink(parent, target_basename)
}

fn save_full_materialized_weights(
    student: &Qwen35Model,
    store: &mut TensorStore,
    tape: &mut Tape,
    weights_path: &Path,
    bf16: bool,
) -> Result<(), Qwen35CheckpointError> {
    let keep = live_tensor_ids(store);
    let result = save_materialized_registry(student, store, tape, weights_path, bf16)
        .map_err(Qwen35CheckpointError::Autograd);
    cleanup_materialized_temps(store, &keep);
    result
}

fn cleanup_materialized_temps(store: &mut TensorStore, keep: &std::collections::HashSet<TensorId>) {
    for tensor_id in 0..store.tensors.len() {
        if keep.contains(&tensor_id) || store.get(tensor_id).is_none() {
            continue;
        }
        let _ = store.free(tensor_id);
    }
}

fn save_adapter_only_weights(
    student: &Qwen35Model,
    store: &mut TensorStore,
    weights_path: &Path,
    bf16: bool,
    adapter_config: &LoraAdapterConfig,
) -> Result<(), Qwen35CheckpointError> {
    let registry = build_peft_adapter_registry(student)?;
    if registry.is_empty() {
        return Err(Qwen35CheckpointError::Custom(
            "Qwen3.5 adapter-only checkpoint requested, but the student has no \
             LoRA adapter tensors. Hint: build the student with \
             Qwen35Model::new_with_lora(..., Some(LoraConfig { .. }), ...) or \
             save a FullMaterialized checkpoint."
                .to_owned(),
        ));
    }
    let step_dir = weights_path.parent().ok_or_else(|| {
        Qwen35CheckpointError::Custom(format!(
            "adapter checkpoint path {} has no parent directory",
            weights_path.display()
        ))
    })?;
    fs::write(
        step_dir.join(ADAPTER_CONFIG_FILENAME),
        serde_json::to_string_pretty(adapter_config)?,
    )?;
    if bf16 {
        registry.save_from_bf16(store, weights_path)?;
    } else {
        registry.save_from(store, weights_path)?;
    }
    Ok(())
}

fn build_peft_adapter_registry(
    student: &Qwen35Model,
) -> Result<SafetensorsRegistry, Qwen35CheckpointError> {
    let mut registry = SafetensorsRegistry::new();
    for (internal_name, tensor_id) in student.adapter_name_map() {
        registry.insert(peft_adapter_name(internal_name)?, tensor_id);
    }
    Ok(registry)
}

fn peft_adapter_name(internal_name: &str) -> Result<String, Qwen35CheckpointError> {
    let (base_name, peft_suffix) = if let Some(base_name) = internal_name.strip_suffix(".lora_a") {
        (base_name, "lora_A")
    } else if let Some(base_name) = internal_name.strip_suffix(".lora_b") {
        (base_name, "lora_B")
    } else {
        return Err(Qwen35CheckpointError::Custom(format!(
            "Qwen3.5 adapter tensor name {internal_name:?} does not end in .lora_a or .lora_b"
        )));
    };
    let Some(base_name) = base_name.strip_suffix(".weight") else {
        return Err(Qwen35CheckpointError::Custom(format!(
            "Qwen3.5 adapter tensor base name {base_name:?} does not end in .weight"
        )));
    };
    Ok(format!("base_model.model.{base_name}.{peft_suffix}.weight"))
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
                let bos_token_id =
                    read_optional_token_id(text, &config, "bos_token_id", fallback_config_path)?;
                let eos_token_id =
                    read_token_id(text, &config, "eos_token_id", fallback_config_path)?;
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
    text_config: &serde_json::Value,
    root_config: &serde_json::Value,
    key: &str,
    fallback_config_path: &Path,
) -> Result<u32, Qwen35CheckpointError> {
    read_optional_token_id(text_config, root_config, key, fallback_config_path)?.ok_or_else(|| {
        Qwen35CheckpointError::Custom(format!(
            "source config {} is missing {key} in text_config or root object",
            fallback_config_path.display()
        ))
    })
}

fn read_optional_token_id(
    text_config: &serde_json::Value,
    root_config: &serde_json::Value,
    key: &str,
    fallback_config_path: &Path,
) -> Result<Option<u32>, Qwen35CheckpointError> {
    let Some(raw) = text_config.get(key).or_else(|| root_config.get(key)) else {
        return Ok(None);
    };
    let value = raw.as_u64().ok_or_else(|| {
        Qwen35CheckpointError::Custom(format!(
            "source config {} has non-integer {key}: {raw}",
            fallback_config_path.display()
        ))
    })?;
    u32::try_from(value)
        .map_err(|_| {
            Qwen35CheckpointError::Custom(format!(
                "source config {} has out-of-range {key}: {value}",
                fallback_config_path.display()
            ))
        })
        .map(Some)
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
    use crate::{
        lora::{LoraAdapterConfig, LoraConfig},
        qwen35::Qwen35Model,
        qwen35_loader::load_qwen35_from_hf_dir,
    };
    use autograd::{Tape, TensorStore};
    use safetensors::SafeTensors;
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
            rope_scaling: None,
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
            full_attn_gated: true,
        }
    }

    fn tiny_qwen35_config() -> Qwen35Config {
        let cfg = Qwen35Config {
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            vocab_size: 32,
            rms_norm_eps: 1.0e-6,
            stop_token_ids: vec![31],
            bos_token_id: Some(1),
            eos_token_id: 31,
            tie_word_embeddings: false,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 8,
            linear_num_key_heads: 2,
            linear_key_head_dim: 8,
            linear_num_value_heads: 2,
            linear_value_head_dim: 8,
            linear_conv_kernel_dim: 4,
            rope_theta: 10_000.0,
            rope_scaling: None,
            partial_rotary_factor: 1.0,
            rotary_dim: 8,
            rope_cache_len_hint: Some(16),
            layer_types: vec![LayerType::FullAttention],
            num_experts: 0,
            num_experts_per_tok: 0,
            decoder_sparse_step: 1,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            norm_topk_prob: true,
            mlp_only_layers: Vec::new(),
            full_attn_gated: true,
        };
        cfg.validate_train_scratch_contract()
            .expect("tiny checkpoint config should satisfy scratch contract");
        cfg.validate_train_lora_or_frozen_contract()
            .expect("tiny checkpoint config should satisfy LoRA contract");
        cfg
    }

    fn safetensor_names(path: &Path) -> Vec<String> {
        let bytes = fs::read(path).expect("read safetensors");
        let tensors = SafeTensors::deserialize(&bytes).expect("deserialize safetensors");
        let mut names = tensors
            .iter()
            .map(|(name, _)| name.to_string())
            .collect::<Vec<_>>();
        names.sort();
        names
    }

    fn test_lora_config() -> LoraConfig {
        LoraConfig {
            rank: 2,
            alpha: 4.0,
        }
    }

    fn test_adapter_config(lora: LoraConfig) -> LoraAdapterConfig {
        LoraAdapterConfig::new("base-qwen35-test", "qwen35", lora)
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

    #[test]
    fn save_step_checkpoint_cleans_new_step_dir_when_weight_write_fails() {
        let tmp = tempdir().expect("tempdir");
        let cfg = dense_qwen35_config();

        let err = save_step_checkpoint(
            Qwen35StepCheckpoint {
                out_dir: tmp.path(),
                step: 4,
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
            |_weights_path| Err(Qwen35CheckpointError::Custom("weight write failed".into())),
        )
        .expect_err("failing weight writer should fail checkpoint save");

        assert!(err.to_string().contains("weight write failed"));
        assert!(
            !tmp.path().join("step_000004").exists(),
            "failed weight save must remove the newly-created partial step dir"
        );
        assert!(
            !tmp.path().join("latest").exists(),
            "failed weight save must not publish latest"
        );
    }

    #[test]
    fn copy_or_synthesize_generation_config_reads_root_token_ids() {
        let tmp = tempdir().expect("tempdir");
        let source_config = tmp.path().join("source_config.json");
        let missing_generation_config = tmp.path().join("missing_generation_config.json");
        fs::write(
            &source_config,
            serde_json::to_string_pretty(&json!({
                "eos_token_id": 9,
                "bos_token_id": 2,
                "text_config": {
                    "hidden_size": 16
                }
            }))
            .expect("serialize source config"),
        )
        .expect("write source config");

        let step_dir = save_step_checkpoint(
            Qwen35StepCheckpoint {
                out_dir: tmp.path(),
                step: 5,
                tokenizer_path: None,
                config_json: ConfigJsonSource::CopyFrom(&source_config),
                generation_config: GenerationConfigSource::CopyOrSynthesize {
                    source_path: &missing_generation_config,
                    fallback_config_path: &source_config,
                },
            },
            |weights_path| {
                fs::write(weights_path, b"weights").expect("write weights");
                Ok(())
            },
        )
        .expect("save checkpoint");

        let generation_value: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(step_dir.join("generation_config.json")).unwrap(),
        )
        .expect("parse generation config");
        assert_eq!(generation_value["eos_token_id"], json!([9, 2]));
    }

    #[test]
    fn save_qwen35_student_checkpoint_writes_adapter_only_safetensors() {
        let tmp = tempdir().expect("tempdir");
        let cfg = tiny_qwen35_config();
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let lora = test_lora_config();
        let adapter_config = test_adapter_config(lora);
        let student =
            Qwen35Model::new_with_lora(&cfg, Some(lora), &mut store).expect("lora student");
        let adapter_count = student.adapter_name_map().len();

        let step_dir = save_qwen35_student_checkpoint(
            Qwen35StepCheckpoint {
                out_dir: tmp.path(),
                step: 8,
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
            &student,
            &mut store,
            &mut tape,
            Qwen35StudentWeights::AdapterOnly {
                bf16: false,
                adapter_config: &adapter_config,
            },
        )
        .expect("save adapter checkpoint");

        let names = safetensor_names(&step_dir.join("adapter_model.safetensors"));
        assert_eq!(names.len(), adapter_count);
        assert!(
            names
                .iter()
                .all(|name| name.contains(".lora_A.weight") || name.contains(".lora_B.weight")),
            "adapter-only checkpoint must use PEFT LoRA tensor keys: {names:?}"
        );
        assert!(
            names.iter().all(|name| !name.contains(".weight.lora_")),
            "adapter-only checkpoint must not expose internal train-side keys: {names:?}"
        );
        let adapter_config_value: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(step_dir.join("adapter_config.json")).unwrap(),
        )
        .expect("parse adapter config");
        assert_eq!(adapter_config_value["r"], json!(2));
        assert_eq!(adapter_config_value["lora_alpha"], json!(4.0));
        assert!(
            !step_dir.join("model.safetensors").exists(),
            "adapter-only checkpoint must not masquerade as a full model"
        );
        assert!(
            tmp.path()
                .join("latest")
                .join("adapter_model.safetensors")
                .is_file(),
            "latest should publish only after adapter_model.safetensors lands"
        );
    }

    #[test]
    fn save_qwen35_student_checkpoint_full_materialized_cleans_lora_temps() {
        let tmp = tempdir().expect("tempdir");
        let cfg = tiny_qwen35_config();
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let student = Qwen35Model::new_with_lora(&cfg, Some(test_lora_config()), &mut store)
            .expect("lora student");
        let live_before = live_tensor_ids(&store).len();

        let step_dir = save_qwen35_student_checkpoint(
            Qwen35StepCheckpoint {
                out_dir: tmp.path(),
                step: 9,
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
            &student,
            &mut store,
            &mut tape,
            Qwen35StudentWeights::FullMaterialized { bf16: false },
        )
        .expect("save full checkpoint");

        let live_after = live_tensor_ids(&store).len();
        assert_eq!(
            live_after, live_before,
            "full materialized save must not retain merged LoRA temporary tensors"
        );
        let names = safetensor_names(&step_dir.join("model.safetensors"));
        assert!(
            names
                .iter()
                .any(|name| name == cfg.embed_tokens_tensor_name())
        );
        assert!(
            names.iter().all(|name| !name.contains(".lora_")),
            "full materialized checkpoint should use base HF tensor names: {names:?}"
        );
    }

    #[test]
    fn save_qwen35_student_checkpoint_full_materialized_loads_from_hf_dir() {
        let tmp = tempdir().expect("tempdir");
        let cfg = tiny_qwen35_config();
        let mut source_store = TensorStore::default();
        let mut tape = Tape::new();
        let student = Qwen35Model::new(&cfg, &mut source_store).expect("scratch student");
        let source_param_map = student.param_name_map();

        let step_dir = save_qwen35_student_checkpoint(
            Qwen35StepCheckpoint {
                out_dir: tmp.path(),
                step: 10,
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
            &student,
            &mut source_store,
            &mut tape,
            Qwen35StudentWeights::FullMaterialized { bf16: false },
        )
        .expect("save full checkpoint");

        let mut loaded_store = TensorStore::default();
        let loaded =
            load_qwen35_from_hf_dir(&step_dir, &mut loaded_store).expect("reload saved student");
        let loaded_param_map = loaded.param_name_map();
        assert_eq!(loaded_param_map.len(), source_param_map.len());

        let mut names = source_param_map.keys().copied().collect::<Vec<_>>();
        names.sort();
        for name in names {
            let source_id = source_param_map[name];
            let loaded_id = loaded_param_map[name];
            let source = source_store.to_host(source_id).expect("source host");
            let loaded = loaded_store.to_host(loaded_id).expect("loaded host");
            assert_eq!(loaded.len(), source.len(), "tensor {name} length mismatch");
            for (idx, (a, b)) in source.iter().zip(&loaded).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "tensor {name}[{idx}] changed across save/load"
                );
            }
        }
    }

    #[test]
    fn save_qwen35_student_checkpoint_rejects_adapter_only_without_lora() {
        let tmp = tempdir().expect("tempdir");
        let cfg = tiny_qwen35_config();
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let student = Qwen35Model::new(&cfg, &mut store).expect("scratch student");
        let adapter_config = test_adapter_config(test_lora_config());

        let err = save_qwen35_student_checkpoint(
            Qwen35StepCheckpoint {
                out_dir: tmp.path(),
                step: 11,
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
            &student,
            &mut store,
            &mut tape,
            Qwen35StudentWeights::AdapterOnly {
                bf16: false,
                adapter_config: &adapter_config,
            },
        )
        .expect_err("adapter-only save should reject a non-LoRA student");

        let message = err.to_string();
        assert!(message.contains("adapter-only checkpoint requested"));
        assert!(message.contains("Qwen35Model::new_with_lora"));
        assert!(
            !tmp.path().join("step_000011").exists(),
            "failed adapter-only save must remove partial step dir"
        );
        assert!(
            !tmp.path().join("latest").exists(),
            "failed adapter-only save must not publish latest"
        );
    }
}
