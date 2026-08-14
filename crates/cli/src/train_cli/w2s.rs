use anyhow::{Context, Result, anyhow, bail};
use autograd::TensorStore;

use super::{
    opd_checkpoint::{save_w2s_adapter, should_save_step_checkpoint},
    opd_runtime::{build_opd_store, parse_lora_target_set, trainable_param_ids},
};
use crate::args::TrainW2sArgs;

fn vram_gb(store: &TensorStore) -> Option<(f64, f64)> {
    store.backend().device_mem_info().map(|(free, total)| {
        let gb = |b: usize| b as f64 / (1u64 << 30) as f64;
        (gb(total - free), gb(free))
    })
}

/// `arle train w2s` — weak-to-strong online distillation.
///
/// Loads the student (LoRA), two auxiliary models (each with pre-RL + post-RL
/// checkpoints), and runs the w2s step: ΔT → proxy teacher → reverse KL +
/// local/global KL regularization.
pub(super) fn run_w2s(args: TrainW2sArgs) -> Result<()> {
    use crate::args::W2sAuxBackendArg;
    use autograd::{Tape, optim::AdamW};
    use train::{
        lora::LoraConfig,
        qwen35::Qwen35Model,
        qwen35_loader::load_qwen35_from_hf_dir,
        w2s::{W2sAuxModel, W2sConfig, sync_lora_adapters, w2s_step},
    };

    let target_set = parse_lora_target_set(&args.lora_target_set)?;
    let lora = LoraConfig {
        rank: args.lora_rank,
        alpha: args.lora_alpha,
    };

    let (mut store, train_backend, backend_label) = build_opd_store(args.backend)?;
    #[cfg(not(feature = "cuda"))]
    let _ = train_backend;
    let mut tape = Tape::new();

    eprintln!("[arle train w2s] backend={backend_label}");

    fn vram(store: &TensorStore, phase: &str) {
        if let Some((used, free)) = vram_gb(store) {
            eprintln!("[arle train w2s] vram after {phase}: used={used:.1} GB free={free:.1} GB");
        }
    }

    macro_rules! load_model {
        ($label:expr, $path:expr, $store:expr) => {{
            eprintln!(
                "[arle train w2s] loading {} from {}",
                $label,
                $path.display()
            );
            load_qwen35_from_hf_dir($path, $store)
                .with_context(|| format!("load {} from {}", $label, $path.display()))?
        }};
    }

    // π_base for the global KL regularizer.
    let base = load_model!("base", &args.student_model, &mut store);
    #[cfg(feature = "cuda")]
    let vocab_size = base.config().vocab_size;

    // Student (shadow adapter) — the model being trained.
    eprintln!("[arle train w2s] creating student (shadow adapter)");
    let student = Qwen35Model::new_lora_from_base(&base, lora, target_set, &mut store)
        .context("create student shadow adapter")?;
    vram(&store, "base+student load");

    // Serving adapter — π_old for the local KL regularizer. Created from the
    // student (not the base) so `new_lora_from_base`'s retain_ids keeps the
    // student's adapter params alive alongside the serving adapter's.
    eprintln!("[arle train w2s] creating serving adapter");
    let serving = Qwen35Model::new_lora_from_base(&student, lora, target_set, &mut store)
        .context("create serving adapter")?;

    // Initialize serving = shadow (both start from the same LoRA init).
    sync_lora_adapters(&student, &serving, &mut store)?;

    let aux1 = match args.aux_backend {
        W2sAuxBackendArg::InProcess => {
            // Omitted --aux1-pre reuses the base: ΔT = post_RL − base measures
            // how far the teacher sits above the student's starting point.
            let aux1_pre = match &args.aux1_pre {
                Some(path) => load_model!("aux1 pre-RL", path, &mut store),
                None => {
                    eprintln!("[arle train w2s] aux1 pre-RL = student base model (reused)");
                    base.clone()
                }
            };
            let aux1_post = load_model!("aux1 post-RL", &args.aux1_post, &mut store);
            W2sAuxModel::new_in_process(aux1_pre, aux1_post)
        }
        #[cfg(feature = "cuda")]
        W2sAuxBackendArg::Infer => {
            use infer_api::{EngineLoadConfig, LoadedInferenceEngine};
            use std::sync::{Arc, Mutex};
            use train::teacher_infer::{InferTeacher, TeacherForward};

            // Aux engines run forward passes over the full prompt+completion
            // (chain-of-thought KL), so the sequence budget must cover GSM8K
            // solutions (~500 tokens). `mem_fraction_static` keeps the KV pool
            // small so the student's activations + aux weights fit on one GPU.
            let aux_cfg = EngineLoadConfig {
                mem_fraction_static: 0.1,
                tp_size: None,
                ..EngineLoadConfig::single_sequence(2048)
        };
            // Each engine offloads right after load so only one aux is resident;
            // the w2s step reloads it just-in-time for the delta forward.
            let load_teacher = |label: &str, path: &std::path::Path| -> Result<InferTeacher> {
                eprintln!(
                    "[arle train w2s] loading aux1 {label} (infer) from {}",
                    path.display()
                );
                let engine = LoadedInferenceEngine::load_with_config(
                    path.to_str()
                        .ok_or_else(|| anyhow!("aux1 {} path not UTF-8", label))?,
                    true,
                    aux_cfg.clone(),
                )
                .with_context(|| format!("load aux1 {label} infer from {}", path.display()))?;
                let teacher = InferTeacher::new(
                    Arc::new(Mutex::new(engine)),
                    train_backend.clone(),
                    vocab_size,
                );
                let freed = teacher
                    .offload_engine_weights()
                    .with_context(|| format!("offload aux1 {label} engine weights"))?;
                eprintln!("[arle train w2s] aux1 {label} offload freed {freed} bytes");
                Ok(teacher)
            };
            let pre_teacher = load_teacher(
                "pre-RL",
                args.aux1_pre.as_ref().unwrap_or(&args.student_model),
            )?;
            let post_teacher = load_teacher("post-RL", &args.aux1_post)?;
            W2sAuxModel::new_infer(pre_teacher, post_teacher)
        }
        #[cfg(not(feature = "cuda"))]
        W2sAuxBackendArg::Infer => {
            bail!("--aux-backend infer requires the cuda feature");
        }
    };

    let aux2 = match args.aux_backend {
        W2sAuxBackendArg::InProcess => {
            match (&args.aux2_pre, &args.aux2_post) {
                (Some(pre_path), Some(post_path)) => {
                    let aux2_pre = load_model!("aux2 pre-RL", pre_path, &mut store);
                    let aux2_post = load_model!("aux2 post-RL", post_path, &mut store);
                    (W2sAuxModel::new_in_process(aux2_pre, aux2_post), false)
                }
                _ => {
                    // Sharing aux1 makes the consistency gate a no-op; the rest
                    // of the pipeline still runs.
                    eprintln!("[arle train w2s] no aux2 checkpoints — sharing aux1");
                    (aux1.clone(), true)
                }
            }
        }
        #[cfg(feature = "cuda")]
        W2sAuxBackendArg::Infer => {
            // Independent aux2 engines are not wired for the infer backend.
            eprintln!("[arle train w2s] sharing aux1 engines for aux2");
            (aux1.clone(), true)
        }
        #[cfg(not(feature = "cuda"))]
        W2sAuxBackendArg::Infer => {
            bail!("--aux-backend infer requires the cuda feature");
        }
    };
    let (aux2, share_aux) = aux2;

    vram(&store, "aux load");

    let trainable_params = trainable_param_ids(&student.all_parameter_ids(), &store);
    eprintln!(
        "[arle train w2s] trainable params: {}",
        trainable_params.len()
    );

    let cfg = W2sConfig {
        alpha: args.alpha,
        temperature: args.temperature,
        confidence_threshold: args.confidence_threshold,
        consistency_threshold: args.consistency_threshold,
        beta_local: args.beta_local,
        beta_global: args.beta_global,
        grad_clip: args.grad_clip,
    };

    // Build the list of training prompts. If `--train-data` is set, load the
    // JSONL and tokenize each row; otherwise fall back to the single
    // `--prompt-ids` sequence (or the default BOS smoke-test prompt).
    let prompts: Vec<Vec<u32>> = match &args.train_data {
        Some(path) => {
            use std::io::{BufRead, BufReader};
            let file = std::fs::File::open(path)
                .with_context(|| format!("open train data {}", path.display()))?;
            let reader = BufReader::new(file);
            let mut out = Vec::new();
            for (i, line) in reader.lines().enumerate() {
                let line = line.with_context(|| format!("read train data line {i}"))?;
                if line.trim().is_empty() {
                    continue;
                }
                let row: serde_json::Value = serde_json::from_str(&line)
                    .with_context(|| format!("parse train data line {i}"))?;
                if let Some(ids) = row.get("prompt_ids").and_then(|v| v.as_array()) {
                    let ids: Vec<u32> = ids
                        .iter()
                        .filter_map(|v| v.as_u64().map(|n| n as u32))
                        .collect();
                    if !ids.is_empty() {
                        out.push(ids);
                    }
                } else if let Some(text) = row.get("text").and_then(|v| v.as_str()) {
                    // Tokenize with the student's tokenizer. If a `completion`
                    // field is present, append it so the KL covers the full
                    // chain-of-thought, not just the first answer token.
                    let mut full_text = text.to_string();
                    if let Some(completion) = row.get("completion").and_then(|v| v.as_str()) {
                        full_text.push_str(completion);
                    }
                    let ids = tokenize_text(&full_text, &args.student_model)?;
                    if !ids.is_empty() {
                        out.push(ids);
                    }
                }
            }
            if out.is_empty() {
                bail!("no valid prompts found in {}", path.display());
            }
            eprintln!(
                "[arle train w2s] loaded {} training prompts from {}",
                out.len(),
                path.display()
            );
            out
        }
        None => {
            let ids = match &args.prompt_ids {
                Some(s) => s
                    .split(',')
                    .map(|t| t.trim().parse::<u32>().context("parse prompt token id"))
                    .collect::<Result<Vec<_>>>()?,
                None => vec![1, 3, 8],
            };
            vec![ids]
        }
    };

    let mut optimizer = AdamW::new(args.lr, (0.9, 0.999), 1e-8, 0.0);

    for step in 0..args.steps {
        let prompt_ids = &prompts[step % prompts.len()];
        let outcome = w2s_step(
            &student,
            &serving,
            &base,
            &aux1,
            &aux2,
            prompt_ids,
            &cfg,
            &trainable_params,
            &mut optimizer,
            &mut store,
            &mut tape,
            share_aux,
        )?;

        let stages = outcome
            .stages
            .iter()
            .map(|(label, secs)| format!("{label}={secs:.3}"))
            .collect::<Vec<_>>()
            .join(" ");
        let total: f64 = outcome.stages.iter().map(|(_, s)| s).sum();
        // Per-step, not per-phase: an OOM mid-run is only attributable to us or
        // to a co-tenant if the trend is on every line.
        let vram_used = match vram_gb(&store) {
            Some((used, free)) => format!(" vram_used={used:.1} vram_free={free:.1}"),
            None => String::new(),
        };
        if outcome.skipped {
            eprintln!(
                "[arle train w2s] step={step} skipped reason={:?} max_prob={:.4} consistency={:.4} total={total:.3} {stages}{vram_used}",
                outcome.skip_reason, outcome.max_prob, outcome.consistency
            );
        } else {
            eprintln!(
                "[arle train w2s] step={step} loss={:.6} max_prob={:.4} consistency={:.4} total={total:.3} {stages}{vram_used}",
                outcome.loss, outcome.max_prob, outcome.consistency
            );
            // Shadow → serving: the local KL regularizer now anchors against
            // the just-updated adapter. In the full online flow this only
            // happens after a validation-set eval passes; here we sync every
            // step so the local KL tracks the latest shadow state.
            sync_lora_adapters(&student, &serving, &mut store)?;
        }

        if let Some(adapter_dir) = &args.save_adapter
            && should_save_step_checkpoint(step + 1, args.steps, args.save_every)
        {
            save_w2s_adapter(
                &student,
                &mut store,
                adapter_dir,
                &args.student_model,
                &lora,
                target_set,
            )?;
            eprintln!(
                "[arle train w2s] adapter saved at step={} to {}",
                step + 1,
                adapter_dir.display()
            );
        }
    }

    Ok(())
}

/// Tokenize a text string using the model's tokenizer.json.
fn tokenize_text(text: &str, model_dir: &std::path::Path) -> Result<Vec<u32>> {
    use tokenizers::Tokenizer;
    let tok_path = model_dir.join("tokenizer.json");
    let tok = Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow!("load tokenizer from {}: {e}", tok_path.display()))?;
    let ids = tok
        .encode(text, false)
        .map_err(|e| anyhow!("tokenize: {e}"))?;
    Ok(ids.get_ids().to_vec())
}
