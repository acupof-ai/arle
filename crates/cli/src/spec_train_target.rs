//! `arle train spec-draft` — offline DSpark draft training against a resident
//! trunk.
//!
//! The engine adapter lives here rather than in `spec-train` because
//! `infer-api` already depends on that crate for the Markov-head artifact;
//! this is the one direction left open.

use anyhow::{Context, Result, bail};
use std::path::Path;

use crate::args::TrainSpecDraftArgs;

/// The serving engine as the trainer's trunk. One prefill per sample on a
/// private transient slot, so the serving slots are untouched.
pub(crate) struct EngineTarget<'a> {
    pub(crate) engine: &'a infer_api::LoadedInferenceEngine,
    pub(crate) target_layer_ids: Vec<i64>,
}

impl spec_train::trainer::Target for EngineTarget<'_> {
    fn forward(&mut self, input_ids: &[u32]) -> Result<(Vec<f32>, Vec<f32>)> {
        self.engine
            .forward_training_taps(input_ids, &self.target_layer_ids)
    }
}

pub(crate) fn run_spec_draft(args: TrainSpecDraftArgs) -> Result<()> {
    use autograd::{Tape, TensorStore};
    use infer_api::{EngineLoadConfig, LoadedInferenceEngine};
    use qwen35_spec::{DsparkConfig, Qwen35Config};
    use spec_train::{backbone, trainer};

    let model_path = args
        .model_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("model path is not valid UTF-8"))?;
    let cfg = DsparkConfig::from_dir(&args.draft)
        .with_context(|| format!("read draft config from {}", args.draft.display()))?;
    let trunk = Qwen35Config::from_model_dir(&args.model_path)
        .with_context(|| format!("read trunk config from {}", args.model_path.display()))?;

    let samples = load_samples(&args.data, &args.model_path, args.max_len)?;
    if samples.is_empty() {
        bail!("{} produced no trainable samples", args.data.display());
    }
    println!(
        "{} samples, {} draft layers",
        samples.len(),
        cfg.num_hidden_layers
    );

    // The draft's parameters live on the same device the trunk forward runs on.
    let backend: std::sync::Arc<dyn autograd::Backend> =
        std::sync::Arc::new(autograd::backend_cuda::CudaBackend::new(0)?);
    let mut store = TensorStore::with_backend(backend);
    let embed = backbone::load_frozen(
        &args.model_path,
        trunk.embed_tokens_tensor_name(),
        &mut store,
    )
    .context("load trunk embeddings")?;
    // Tie is a config fact, not something to infer from a failed load — an
    // untied model whose lm_head is missing or unreadable must stop, not train
    // a draft against the wrong output distribution.
    let lm_head = if trunk.tie_word_embeddings {
        embed
    } else {
        backbone::load_frozen(&args.model_path, trunk.lm_head_tensor_name(), &mut store)
            .context("load trunk lm_head")?
    };

    let draft = if backbone::has_weights(&args.draft)? {
        println!("warm-starting from {}", args.draft.display());
        backbone::load(&args.draft, embed, lm_head, &mut store)?
    } else {
        println!("training from scratch");
        backbone::init(
            cfg,
            trunk.vocab_size,
            args.markov_rank,
            embed,
            lm_head,
            &mut store,
        )?
    };
    let target_layer_ids = draft.cfg.target_layer_ids.clone();

    let engine = LoadedInferenceEngine::load_with_config(
        model_path,
        /*cuda_graph=*/ false,
        EngineLoadConfig::single_sequence(args.max_len),
    )
    .with_context(|| format!("load engine from {model_path}"))?;
    let mut target = EngineTarget {
        engine: &engine,
        target_layer_ids,
    };

    let train_cfg = trainer::Config {
        lr: args.lr,
        num_anchors: args.num_anchors,
        blocks_per_backward: args.blocks_per_backward,
        ..trainer::Config::default()
    };
    println!(
        "peak activation {:.1} GiB/backward",
        trainer::peak_activation_bytes(
            &draft.cfg,
            args.blocks_per_backward,
            args.max_len,
            trunk.vocab_size,
        ) as f64
            / (1u64 << 30) as f64
    );
    let mut trainer = trainer::Trainer::new(draft, train_cfg, args.steps);
    let mut tape = Tape::new();

    for step in 0..args.steps {
        let batch: Vec<_> = (0..args.batch)
            .map(|i| {
                let s = &samples[(step * args.batch + i) % samples.len()];
                trainer::Sample {
                    input_ids: s.input_ids.clone(),
                    loss_mask: s.loss_mask.clone(),
                }
            })
            .collect();
        let loss = trainer
            .train_step(&batch, &mut target, &mut store, &mut tape)
            .with_context(|| format!("step {step}"))?;
        if step.is_multiple_of(10) {
            println!("step {step} loss {loss:.6}");
        }
        if (step + 1).is_multiple_of(args.save_every) || step + 1 == args.steps {
            backbone::save(&trainer.draft, &args.out, &mut store)?;
            println!("saved {}", args.out.display());
        }
    }
    Ok(())
}

fn load_samples(
    data: &Path,
    model_dir: &Path,
    max_len: usize,
) -> Result<Vec<spec_train::trainer::Sample>> {
    let tokenizer_path = crate::train_cli::resolve_local_tokenizer_path(model_dir)?;
    spec_train::data::load_samples(data, &tokenizer_path, max_len)
}
