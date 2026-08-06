//! `arle train spec-draft` — offline DSpark draft training against a resident
//! trunk.
//!
//! The engine adapter lives here rather than in `spec-train` because
//! `infer-api` already depends on that crate for the Markov-head artifact;
//! this is the one direction left open.

use anyhow::{Context, Result, ensure};
use std::path::Path;
use std::time::Instant;

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
    use spec_train::{backbone, block, trainer};

    let model_path = args
        .model_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("model path is not valid UTF-8"))?;
    let cfg = DsparkConfig::from_dir(&args.draft)
        .with_context(|| format!("read draft config from {}", args.draft.display()))?;
    // Block construction is unconditionally next-token: row `t` is supervised
    // with `anchor+1+t`, which a `next_token_heads: false` serve reads one
    // position off.
    ensure!(
        cfg.next_token_heads,
        "{} has next_token_heads = false (same-position DFlash); training only \
         produces next-token drafts, so the serve would read every row one \
         position off",
        args.draft.display()
    );
    let trunk = Qwen35Config::from_model_dir(&args.model_path)
        .with_context(|| format!("read trunk config from {}", args.model_path.display()))?;

    let samples = load_samples(
        &args.data,
        &args.model_path,
        spec_train::data::Limits {
            vocab_size: trunk.vocab_size,
            mask_token_id: cfg.mask_token_id,
            max_len: args.max_len,
        },
    )?;
    let tokens: usize = samples.iter().map(|s| s.input_ids.len()).sum();
    let masked: usize = samples
        .iter()
        .map(|s| s.loss_mask.iter().filter(|&&m| m).count())
        .sum();
    let candidates: Vec<usize> = samples
        .iter()
        .map(|s| block::anchor_candidates(&s.loss_mask).len())
        .collect();
    println!(
        "{} samples, {tokens} tokens, {:.1}% loss-masked, {} anchor candidates in sample 0, {} draft layers",
        samples.len(),
        100.0 * masked as f64 / tokens as f64,
        candidates[0],
        cfg.num_hidden_layers
    );

    // The draft's parameters live on the same device the trunk forward runs on.
    let backend = std::sync::Arc::new(autograd::backend_cuda::CudaBackend::new(0)?);
    let mut store = TensorStore::with_backend(backend.clone());
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

    // `single_sequence` sets `total_pages` as a FLOOR; the pool is still sized
    // from free VRAM by `mem_fraction_static`, whose serving default of 0.9
    // leaves the draft nothing (measured: 87.7 of 95.6 GiB for one 512-token
    // sequence).
    let engine = LoadedInferenceEngine::load_with_config(
        model_path,
        /*cuda_graph=*/ false,
        EngineLoadConfig {
            mem_fraction_static: args.trunk_mem_fraction,
            ..EngineLoadConfig::single_sequence(args.max_len)
        },
    )
    .with_context(|| format!("load engine from {model_path}"))?;
    let mut target = EngineTarget {
        engine: &engine,
        target_layer_ids,
    };

    let train_cfg = trainer::Config {
        lr: args.lr,
        warmup_ratio: args.warmup_ratio,
        weight_decay: args.weight_decay,
        max_grad_norm: args.max_grad_norm,
        num_anchors: args.num_anchors,
        blocks_per_backward: args.blocks_per_backward,
        loss_decay_gamma: args.loss_decay_gamma,
        weights: spec_train::loss::Weights {
            ce: args.loss_ce,
            tv: args.loss_tv,
            confidence: args.loss_confidence,
        },
        seed: args.seed,
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
    let mut trainer = trainer::Trainer::new(draft, train_cfg, args.steps, backend);
    let mut tape = Tape::new();

    let mut order = epoch_order(samples.len(), args.seed, 0);
    let mut cursor = 0usize;
    let mut epoch = 0u64;
    for step in 0..args.steps {
        let mut batch = Vec::with_capacity(args.batch);
        for _ in 0..args.batch {
            if cursor == order.len() {
                epoch += 1;
                order = epoch_order(samples.len(), args.seed, epoch);
                cursor = 0;
            }
            let s = &samples[order[cursor]];
            cursor += 1;
            batch.push(trainer::Sample {
                input_ids: s.input_ids.clone(),
                loss_mask: s.loss_mask.clone(),
            });
        }
        let started = Instant::now();
        let st = trainer
            .train_step(&batch, &mut target, &mut store, &mut tape)
            .with_context(|| format!("step {step}"))?;
        if step.is_multiple_of(args.log_every) || step + 1 == args.steps {
            println!(
                "step {step} loss {:.6} ce {:.4} tv {:.4} conf {:.4} accept {:.4} \
                 gnorm {:.4} lr {:.3e} {:.2}s/step counted {}/{} chunks {}",
                st.loss,
                st.ce,
                st.tv,
                st.conf,
                st.mean_accept,
                st.grad_norm,
                st.lr,
                started.elapsed().as_secs_f64(),
                st.counted,
                batch.len(),
                st.chunks
            );
        }
        if (step + 1).is_multiple_of(args.save_every) || step + 1 == args.steps {
            backbone::save(&trainer.draft, &args.out, &store)?;
            println!("saved {}", args.out.display());
        }
    }
    Ok(())
}

/// Sample order for one epoch: the keyed shuffle `block::sample_anchors` uses,
/// so `--seed` alone reproduces both the anchor draw and the batch composition.
fn epoch_order(n: usize, seed: u64, epoch: u64) -> Vec<usize> {
    let mut keyed: Vec<(u64, usize)> = (0..n)
        .map(|i| {
            let key = seed
                ^ epoch.wrapping_mul(0xD1B5_4A32_D192_ED03)
                ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            (splitmix64(key), i)
        })
        .collect();
    keyed.sort_unstable();
    keyed.into_iter().map(|(_, i)| i).collect()
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn load_samples(
    data: &Path,
    model_dir: &Path,
    limits: spec_train::data::Limits,
) -> Result<Vec<spec_train::trainer::Sample>> {
    let tokenizer_path = crate::train_cli::resolve_local_tokenizer_path(model_dir)?;
    spec_train::data::load_samples(data, &tokenizer_path, limits)
}
