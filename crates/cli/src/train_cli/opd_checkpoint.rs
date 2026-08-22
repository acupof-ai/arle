use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use autograd::{Tape, TensorStore};

struct OpdCheckpointSources {
    config_path: PathBuf,
    tokenizer_path: PathBuf,
    generation_config_path: PathBuf,
}

fn checkpoint_sources(model_dir: &Path) -> Result<OpdCheckpointSources> {
    let config_path = model_dir.join("config.json");
    if !config_path.is_file() {
        bail!(
            "cannot save OPD checkpoint: source config.json is missing at {}",
            config_path.display()
        );
    }
    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.is_file() {
        bail!(
            "cannot save OPD checkpoint: source tokenizer.json is missing at {}",
            tokenizer_path.display()
        );
    }
    Ok(OpdCheckpointSources {
        config_path,
        tokenizer_path,
        generation_config_path: model_dir.join("generation_config.json"),
    })
}

pub(super) fn should_save_step_checkpoint(
    step: usize,
    total_steps: usize,
    save_every: usize,
) -> bool {
    step == total_steps || (save_every > 0 && step.is_multiple_of(save_every))
}

pub(super) fn maybe_save_full_student_checkpoint(
    label: &str,
    save_checkpoint: Option<&Path>,
    save_every: usize,
    step: usize,
    total_steps: usize,
    model_dir: &Path,
    student: &train::qwen35::Qwen35Model,
    store: &mut TensorStore,
    tape: &mut autograd::Tape,
) -> Result<Option<PathBuf>> {
    let Some(out_dir) = save_checkpoint else {
        return Ok(None);
    };
    if !should_save_step_checkpoint(step, total_steps, save_every) {
        return Ok(None);
    }

    use train::qwen35_checkpoint::{
        ConfigJsonSource, GenerationConfigSource, Qwen35StepCheckpoint, Qwen35StudentWeights,
        save_qwen35_student_checkpoint,
    };

    fs::create_dir_all(out_dir)
        .with_context(|| format!("create OPD checkpoint root {}", out_dir.display()))?;
    let started = Instant::now();
    let sources = checkpoint_sources(model_dir)?;
    let saved_dir = save_qwen35_student_checkpoint(
        Qwen35StepCheckpoint {
            out_dir,
            step,
            tokenizer_path: Some(&sources.tokenizer_path),
            config_json: ConfigJsonSource::CopyFrom(&sources.config_path),
            generation_config: GenerationConfigSource::CopyOrSynthesize {
                source_path: &sources.generation_config_path,
                fallback_config_path: &sources.config_path,
            },
        },
        student,
        store,
        tape,
        Qwen35StudentWeights::FullMaterialized { bf16: true },
    )
    .with_context(|| format!("save {label} full-materialized checkpoint at step {step}"))?;
    println!(
        "checkpoint_saved kind=full_materialized mode={label} step={step} dir={} seconds={:.6}",
        saved_dir.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(Some(saved_dir))
}

#[cfg(feature = "cuda")]
pub(super) fn agent_opd_adapter_config(
    student_dir: &Path,
    target_set: train::lora::LoraTargetSet,
    lora: train::lora::LoraConfig,
) -> train::lora::LoraAdapterConfig {
    use train::lora::{LoraAdapterConfig, LoraTargetSet};

    let mut config = LoraAdapterConfig::new(student_dir.display().to_string(), "qwen35", lora);
    config.target_modules = match target_set {
        LoraTargetSet::AttentionQv => vec!["q_proj".to_owned(), "v_proj".to_owned()],
        LoraTargetSet::AttentionFull => vec![
            "q_proj".to_owned(),
            "k_proj".to_owned(),
            "v_proj".to_owned(),
            "o_proj".to_owned(),
            "in_proj_qkv".to_owned(),
            "out_proj".to_owned(),
        ],
        LoraTargetSet::AllLinear => vec!["all-linear".to_owned()],
    };
    config
}

/// Adapter-only save; avoids the full-materialize host-loop hang. Loadable by
/// HF PEFT / vLLM / SGLang.
#[cfg(feature = "cuda")]
pub(super) fn save_agent_opd_adapters(
    adapter_dir: &Path,
    dirname: &str,
    step: usize,
    student_dir: &Path,
    student: &train::qwen35::Qwen35Model,
    store: &mut TensorStore,
    adapter_config: &train::lora::LoraAdapterConfig,
) -> Result<()> {
    use train::qwen35_checkpoint::{
        ConfigJsonSource, GenerationConfigSource, Qwen35NamedCheckpoint, Qwen35StudentWeights,
        save_named_qwen35_student_checkpoint,
    };

    fs::create_dir_all(adapter_dir)
        .with_context(|| format!("create LoRA adapter dir {}", adapter_dir.display()))?;
    let sources = checkpoint_sources(student_dir)?;
    let started = Instant::now();
    let mut adapter_tape = Tape::new();
    let saved_dir = save_named_qwen35_student_checkpoint(
        Qwen35NamedCheckpoint {
            out_dir: adapter_dir,
            dirname,
            tokenizer_path: Some(&sources.tokenizer_path),
            config_json: ConfigJsonSource::CopyFrom(&sources.config_path),
            generation_config: GenerationConfigSource::CopyOrSynthesize {
                source_path: &sources.generation_config_path,
                fallback_config_path: &sources.config_path,
            },
        },
        student,
        store,
        &mut adapter_tape,
        Qwen35StudentWeights::AdapterOnly {
            bf16: true,
            adapter_config,
        },
    )
    .with_context(|| format!("save LoRA PEFT adapter dir {dirname}"))?;
    println!(
        "checkpoint_saved kind=peft_adapter mode=agent-opd step={step} dir={} seconds={:.6}",
        saved_dir.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

pub(super) fn save_w2s_adapter(
    student: &train::qwen35::Qwen35Model,
    store: &mut TensorStore,
    adapter_dir: &std::path::Path,
    student_model: &std::path::Path,
    lora: &train::lora::LoraConfig,
    target_set: train::lora::LoraTargetSet,
) -> Result<()> {
    use train::lora::{LoraAdapterConfig, LoraTargetSet};
    use train::qwen35_checkpoint::{
        ConfigJsonSource, GenerationConfigSource, Qwen35NamedCheckpoint, Qwen35StudentWeights,
        save_named_qwen35_student_checkpoint,
    };

    // The adapter dir may have been created (empty) by an earlier setup step;
    // the checkpoint writer refuses to merge into an existing directory.
    if adapter_dir.exists() {
        fs::remove_dir_all(adapter_dir)
            .with_context(|| format!("remove existing adapter dir {}", adapter_dir.display()))?;
    }

    let sources = checkpoint_sources(student_model)?;
    let mut adapter_config =
        LoraAdapterConfig::new(student_model.display().to_string(), "qwen35", *lora);
    adapter_config.target_modules = match target_set {
        LoraTargetSet::AttentionQv => vec!["q_proj".to_owned(), "v_proj".to_owned()],
        LoraTargetSet::AttentionFull => vec![
            "q_proj".to_owned(),
            "k_proj".to_owned(),
            "v_proj".to_owned(),
            "o_proj".to_owned(),
            "in_proj_qkv".to_owned(),
            "out_proj".to_owned(),
        ],
        LoraTargetSet::AllLinear => vec!["all-linear".to_owned()],
    };
    let mut adapter_tape = Tape::new();
    save_named_qwen35_student_checkpoint(
        Qwen35NamedCheckpoint {
            out_dir: adapter_dir.parent().unwrap_or(std::path::Path::new(".")),
            dirname: adapter_dir
                .file_name()
                .unwrap_or_default()
                .to_str()
                .unwrap_or("adapter"),
            tokenizer_path: Some(&sources.tokenizer_path),
            config_json: ConfigJsonSource::CopyFrom(&sources.config_path),
            generation_config: GenerationConfigSource::CopyOrSynthesize {
                source_path: &sources.generation_config_path,
                fallback_config_path: &sources.config_path,
            },
        },
        student,
        store,
        &mut adapter_tape,
        Qwen35StudentWeights::AdapterOnly {
            bf16: true,
            adapter_config: &adapter_config,
        },
    )
    .with_context(|| format!("save w2s adapter to {}", adapter_dir.display()))?;
    Ok(())
}
