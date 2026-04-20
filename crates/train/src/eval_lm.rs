use std::{
    collections::HashSet,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use autograd::{
    AutogradError, Tape, TensorId, TensorStore,
    ops::{gather_last_dim, log_softmax, mul, mul_scalar, sum},
};
use serde::Deserialize;

use crate::{
    CausalLm,
    policy::GrpoPolicyConfig,
    sft_data::{SftExample, tokenize_example},
    tokenizer::ChatTokenizer,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalExample {
    pub input_ids: Vec<u32>,
    pub labels: Vec<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvalSummary {
    pub loss: f64,
    pub token_count: u64,
}

impl EvalSummary {
    pub fn ppl(self) -> f64 {
        self.loss.exp()
    }
}

#[derive(Debug, Deserialize)]
struct TokenizedJsonlRecord {
    input_ids: Vec<u32>,
    #[serde(default)]
    labels: Option<Vec<i32>>,
}

#[derive(Debug, thiserror::Error)]
pub enum EvalLmError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Autograd(#[from] AutogradError),
    #[error("{0}")]
    Custom(String),
}

pub fn load_eval_examples(
    data_path: &Path,
    tokenizer_path: Option<&Path>,
    seq_len: usize,
) -> Result<Vec<EvalExample>, EvalLmError> {
    if seq_len == 0 {
        return Err(EvalLmError::Custom(
            "--seq-len must be greater than zero".into(),
        ));
    }

    let file = File::open(data_path)?;
    let reader = BufReader::new(file);
    let mut examples = Vec::new();
    let mut tokenizer: Option<ChatTokenizer> = None;

    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(trimmed).map_err(|err| {
            EvalLmError::Custom(format!(
                "failed to parse eval JSONL line {} from {}: {err}",
                line_index + 1,
                data_path.display()
            ))
        })?;

        let example = if value.get("messages").is_some() {
            if tokenizer.is_none() {
                let tokenizer_path = tokenizer_path.ok_or_else(|| {
                    EvalLmError::Custom(
                        "tokenizer path is required when eval data contains chat messages".into(),
                    )
                })?;
                tokenizer = Some(ChatTokenizer::from_file(tokenizer_path)?);
            }
            let tokenizer = tokenizer
                .as_ref()
                .ok_or_else(|| EvalLmError::Custom("tokenizer load invariant violated".into()))?;
            let sft: SftExample = serde_json::from_value(value).map_err(|err| {
                EvalLmError::Custom(format!(
                    "failed to parse eval chat JSONL line {} from {}: {err}",
                    line_index + 1,
                    data_path.display()
                ))
            })?;
            let tokenized = tokenize_example(&sft, tokenizer, seq_len)?;
            EvalExample {
                input_ids: tokenized.input_ids,
                labels: tokenized.labels,
            }
        } else if value.get("input_ids").is_some() {
            let record: TokenizedJsonlRecord = serde_json::from_value(value).map_err(|err| {
                EvalLmError::Custom(format!(
                    "failed to parse tokenized eval JSONL line {} from {}: {err}",
                    line_index + 1,
                    data_path.display()
                ))
            })?;
            let mut input_ids = record.input_ids;
            let mut labels = record
                .labels
                .unwrap_or_else(|| input_ids.iter().map(|&id| id as i32).collect());
            if input_ids.len() != labels.len() {
                return Err(EvalLmError::Custom(format!(
                    "tokenized eval JSONL line {} from {} has mismatched input_ids/labels lengths: {} != {}",
                    line_index + 1,
                    data_path.display(),
                    input_ids.len(),
                    labels.len()
                )));
            }
            if input_ids.len() > seq_len {
                input_ids.truncate(seq_len);
                labels.truncate(seq_len);
            }
            EvalExample { input_ids, labels }
        } else {
            return Err(EvalLmError::Custom(format!(
                "unsupported eval JSONL line {} from {}: expected either messages or input_ids",
                line_index + 1,
                data_path.display()
            )));
        };

        if example.input_ids.len() < 2 {
            continue;
        }
        if !example.labels.iter().skip(1).any(|&label| label >= 0) {
            continue;
        }
        examples.push(example);
    }

    if examples.is_empty() {
        return Err(EvalLmError::Custom(format!(
            "no usable eval examples remained after loading {}",
            data_path.display()
        )));
    }

    Ok(examples)
}

pub fn evaluate_examples<M: CausalLm>(
    model: &M,
    examples: &[EvalExample],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<EvalSummary, EvalLmError> {
    let vocab_size = model.config().vocab_size();
    let keep_ids: HashSet<TensorId> = model.all_parameter_ids().into_iter().collect();
    let mut total_loss = 0.0f64;
    let mut total_tokens = 0u64;

    for example in examples {
        let input_len = example.input_ids.len() - 1;
        let input_ids = example.input_ids[..input_len]
            .iter()
            .map(|&id| id as usize)
            .collect::<Vec<_>>();
        tape.entries.clear();
        tape.set_enabled(true);
        let logits = model.forward_batch_tokens(&input_ids, 1, input_len, store, tape)?;
        let (loss_id, token_count) =
            masked_causal_loss(logits, &example.labels[1..], store, tape, vocab_size)?;
        let loss_value = store.to_host(loss_id).map_err(EvalLmError::Autograd)?[0] as f64;
        total_loss += loss_value * token_count as f64;
        total_tokens += token_count as u64;
        tape.entries.clear();
        store.retain_ids(&keep_ids);
    }

    if total_tokens == 0 {
        return Err(EvalLmError::Custom(
            "no supervised tokens remained after evaluation".into(),
        ));
    }

    Ok(EvalSummary {
        loss: total_loss / total_tokens as f64,
        token_count: total_tokens,
    })
}

fn masked_causal_loss(
    logits: TensorId,
    labels: &[i32],
    store: &mut TensorStore,
    tape: &mut Tape,
    vocab_size: usize,
) -> Result<(TensorId, usize), EvalLmError> {
    let logits_shape = store
        .get(logits)
        .ok_or(AutogradError::InvalidTensorId(logits))?
        .shape
        .clone();
    let target_count = logits_shape
        .iter()
        .take(logits_shape.len() - 1)
        .product::<usize>();
    if labels.len() != target_count {
        return Err(EvalLmError::Custom(format!(
            "eval labels len {} does not match logits prefix size {}",
            labels.len(),
            target_count
        )));
    }

    let valid_count = labels.iter().filter(|&&label| label >= 0).count();
    if valid_count == 0 {
        return Err(EvalLmError::Custom(
            "eval example has no supervised tokens".into(),
        ));
    }

    let gather_indices = labels
        .iter()
        .map(|&label| {
            let index = if label >= 0 { label as usize } else { 0 };
            if index >= vocab_size {
                return Err(EvalLmError::Custom(format!(
                    "eval label {index} is outside vocab size {vocab_size}"
                )));
            }
            Ok(index)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mask_values = labels
        .iter()
        .map(|&label| if label >= 0 { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();

    let log_probs = log_softmax(logits, store, tape)?;
    let target_log_probs = gather_last_dim(log_probs, &gather_indices, store, tape)?;
    let target_shape = store
        .get(target_log_probs)
        .ok_or(AutogradError::InvalidTensorId(target_log_probs))?
        .shape
        .clone();
    let mask = store.from_slice(&mask_values, &target_shape)?;
    let masked = mul(target_log_probs, mask, store, tape)?;
    let total = sum(masked, store, tape)?;
    let loss = mul_scalar(total, -1.0 / valid_count as f32, store, tape)?;
    Ok((loss, valid_count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    use tempfile::tempdir;

    use crate::{
        causal_lm::build_registry,
        qwen3::{Qwen3Config, Qwen3Model},
        qwen3_checkpoint::{
            ConfigJsonSource as Qwen3ConfigJsonSource,
            GenerationConfigSource as Qwen3GenerationConfigSource, Qwen3StepCheckpoint,
            save_step_checkpoint as save_qwen3_step_checkpoint,
        },
        qwen35::{LayerType, Qwen35Config, Qwen35Model},
        qwen35_checkpoint::{
            ConfigJsonSource as Qwen35ConfigJsonSource,
            GenerationConfigSource as Qwen35GenerationConfigSource, Qwen35StepCheckpoint,
            save_step_checkpoint as save_qwen35_step_checkpoint,
        },
    };

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn tiny_qwen3_config() -> Qwen3Config {
        Qwen3Config {
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            vocab_size: 16,
            rms_norm_eps: 1.0e-6,
            tie_word_embeddings: false,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 8,
            max_position_embeddings: 8,
            rope_theta: 10_000.0,
        }
    }

    fn tiny_qwen35_config() -> Qwen35Config {
        Qwen35Config {
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            vocab_size: 16,
            rms_norm_eps: 1.0e-6,
            stop_token_ids: vec![15],
            bos_token_id: Some(1),
            eos_token_id: 15,
            tie_word_embeddings: false,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 8,
            linear_num_key_heads: 2,
            linear_key_head_dim: 8,
            linear_num_value_heads: 2,
            linear_value_head_dim: 8,
            linear_conv_kernel_dim: 4,
            rope_theta: 10_000.0,
            partial_rotary_factor: 1.0,
            rotary_dim: 8,
            rope_cache_len_hint: Some(8),
            layer_types: vec![LayerType::FullAttention; 2],
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
    fn eval_qwen3_checkpoint_on_tokenized_jsonl() -> TestResult {
        let tmp = tempdir()?;
        let model_dir = tmp.path().join("qwen3-model");
        fs::create_dir_all(&model_dir)?;
        let tokenizer_path = tmp.path().join("tokenizer.json");
        fs::write(&tokenizer_path, "{}")?;

        let cfg = tiny_qwen3_config();
        let mut store = TensorStore::default();
        let model = Qwen3Model::new(&cfg, &mut store)?;
        let registry = build_registry(&model);
        let step_dir = save_qwen3_step_checkpoint(
            Qwen3StepCheckpoint {
                out_dir: &model_dir,
                step: 1,
                tokenizer_path: Some(&tokenizer_path),
                config_json: Qwen3ConfigJsonSource::Synthesize {
                    cfg: &cfg,
                    bos_token_id: 1,
                    eos_token_id: 15,
                    torch_dtype: "float32",
                },
                generation_config: Qwen3GenerationConfigSource::Synthesize {
                    bos_token_id: 1,
                    eos_token_id: 15,
                },
            },
            |weights_path| {
                registry.save_from(&mut store, weights_path)?;
                Ok(())
            },
        )?;

        let data_path = tmp.path().join("eval.jsonl");
        fs::write(
            &data_path,
            r#"{"input_ids":[1,2,3,4],"labels":[-100,2,3,4]}"#,
        )?;
        let examples = load_eval_examples(&data_path, None, 8)?;

        let mut load_store = TensorStore::default();
        let load_model = Qwen3Model::new(&cfg, &mut load_store)?;
        let mut load_registry = build_registry(&load_model);
        load_registry.load_into(&mut load_store, &step_dir.join("model.safetensors"))?;
        let mut tape = Tape::new();
        let summary = evaluate_examples(&load_model, &examples, &mut load_store, &mut tape)?;

        assert!(summary.loss.is_finite());
        assert!(summary.token_count > 0);
        assert!(summary.ppl().is_finite());
        Ok(())
    }

    #[test]
    fn eval_qwen35_checkpoint_on_tokenized_jsonl() -> TestResult {
        let tmp = tempdir()?;
        let model_dir = tmp.path().join("qwen35-model");
        fs::create_dir_all(&model_dir)?;

        let cfg = tiny_qwen35_config();
        let mut store = TensorStore::default();
        let model = Qwen35Model::new(&cfg, &mut store)?;
        let registry = build_registry(&model);
        let step_dir = save_qwen35_step_checkpoint(
            Qwen35StepCheckpoint {
                out_dir: &model_dir,
                step: 1,
                tokenizer_path: None,
                config_json: Qwen35ConfigJsonSource::Synthesize {
                    cfg: &cfg,
                    torch_dtype: "float32",
                },
                generation_config: Qwen35GenerationConfigSource::Synthesize {
                    bos_token_id: cfg.bos_token_id,
                    eos_token_id: cfg.eos_token_id,
                },
            },
            |weights_path| {
                registry.save_from(&mut store, weights_path)?;
                Ok(())
            },
        )?;

        let data_path = tmp.path().join("eval.jsonl");
        fs::write(&data_path, r#"{"input_ids":[1,2,3,4],"labels":[1,2,3,4]}"#)?;
        let examples = load_eval_examples(&data_path, None, 8)?;

        let mut load_store = TensorStore::default();
        let load_model = Qwen35Model::new(&cfg, &mut load_store)?;
        let mut load_registry = build_registry(&load_model);
        load_registry.load_into(&mut load_store, &step_dir.join("model.safetensors"))?;
        let mut tape = Tape::new();
        let summary = evaluate_examples(&load_model, &examples, &mut load_store, &mut tape)?;

        assert!(summary.loss.is_finite());
        assert!(summary.token_count > 0);
        assert!(summary.ppl().is_finite());
        Ok(())
    }
}
