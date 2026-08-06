//! `arle train spec-draft` — offline DSpark draft training against a resident
//! trunk.
//!
//! The engine adapter lives here rather than in `spec-train` because
//! `infer-api` already depends on that crate for the Markov-head artifact;
//! this is the one direction left open.

use anyhow::{Context, Result, bail, ensure};
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

    let (samples, tokenizer_vocab) = load_samples(
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
    println!(
        "{} samples, {tokens} tokens, {:.1}% loss-masked, {} anchor candidates in sample 0, \
         {} draft layers, vocab tokenizer {tokenizer_vocab} / trunk {}",
        samples.len(),
        100.0 * masked as f64 / tokens as f64,
        block::anchor_candidates(&samples[0].loss_mask).len(),
        cfg.num_hidden_layers,
        trunk.vocab_size,
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

    if args.probe {
        return probe(&samples[0], &draft, &mut target, &mut store);
    }

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

/// The samples plus the tokenizer's vocab size, printed beside the trunk's: a
/// tokenizer wider than the logit row indexes off the end of every target.
fn load_samples(
    data: &Path,
    model_dir: &Path,
    limits: spec_train::data::Limits,
) -> Result<(Vec<spec_train::trainer::Sample>, usize)> {
    let tokenizer_path = crate::train_cli::resolve_local_tokenizer_path(model_dir)?;
    let vocab = train::tokenizer::ChatTokenizer::from_file(&tokenizer_path)
        .with_context(|| format!("load tokenizer {}", tokenizer_path.display()))?
        .vocab_size();
    Ok((
        spec_train::data::load_samples(data, &tokenizer_path, limits)?,
        vocab,
    ))
}

/// Positions sampled across the sequence for the target-logits oracle.
const PROBE_POSITIONS: usize = 16;
const PROBE_TOPK: usize = 64;
// The engine's head is FP8/Marlin and the reconstruction is f32 over bf16
// weights, so these sit well under clean agreement and far over a wrong tap
// layer or a pre/post-norm slip, which put all three near zero.
const MIN_ARGMAX_AGREE: f64 = 0.75;
const MIN_TOPK_OVERLAP: f64 = 0.50;
const MAX_LOGPROB_DELTA: f64 = 1.0;
/// Two taps this aligned are one trunk layer captured twice, which no shape
/// check sees.
const MAX_TAP_COSINE: f64 = 0.99;

/// `--probe`: one trunk forward over the first sample through the trainer's own
/// `Target`, then the checks that catch a supervision signal that is wrong and
/// still trains.
fn probe(
    sample: &spec_train::trainer::Sample,
    draft: &spec_train::backbone::Draft,
    target: &mut EngineTarget<'_>,
    store: &mut autograd::TensorStore,
) -> Result<()> {
    use spec_train::trainer::Target as _;

    let seq = sample.input_ids.len();
    ensure!(seq >= 2, "probe needs 2+ tokens, sample 0 has {seq}");
    let hidden = draft.cfg.hidden_size;
    let layer_ids = &draft.cfg.target_layer_ids;
    let ntaps = layer_ids.len();

    let started = Instant::now();
    let (taps, last_hidden) = target.forward(&sample.input_ids)?;
    ensure!(
        taps.len() == seq * ntaps * hidden && last_hidden.len() == seq * hidden,
        "trunk returned {} tap and {} hidden values for {seq} tokens × {ntaps} taps × {hidden} hidden",
        taps.len(),
        last_hidden.len()
    );
    println!(
        "trunk forward: {seq} tokens, {ntaps} taps at {layer_ids:?}, {:.2}s",
        started.elapsed().as_secs_f64()
    );

    let mut failures = Vec::new();
    probe_taps(&taps, seq, hidden, layer_ids, &mut failures);
    probe_logits(
        sample,
        &last_hidden,
        draft,
        target.engine,
        store,
        &mut failures,
    )?;

    if failures.is_empty() {
        println!("PROBE PASS");
        return Ok(());
    }
    bail!("PROBE FAIL: {}", failures.join("; "));
}

/// Per-tap norm, range, non-finite fraction, and the pairwise cosine matrix.
fn probe_taps(
    taps: &[f32],
    seq: usize,
    hidden: usize,
    layer_ids: &[i64],
    failures: &mut Vec<String>,
) {
    let ntaps = layer_ids.len();
    let mut dot = vec![0.0f64; ntaps * ntaps];
    let mut lo = vec![f32::INFINITY; ntaps];
    let mut hi = vec![f32::NEG_INFINITY; ntaps];
    let mut nonfinite = vec![0usize; ntaps];
    for r in 0..seq {
        let row = &taps[r * ntaps * hidden..][..ntaps * hidden];
        for t in 0..ntaps {
            let a = &row[t * hidden..][..hidden];
            for &x in a {
                if x.is_finite() {
                    lo[t] = lo[t].min(x);
                    hi[t] = hi[t].max(x);
                } else {
                    nonfinite[t] += 1;
                }
            }
            for u in t..ntaps {
                let b = &row[u * hidden..][..hidden];
                dot[t * ntaps + u] += a
                    .iter()
                    .zip(b)
                    .map(|(&x, &y)| f64::from(x) * f64::from(y))
                    .sum::<f64>();
            }
        }
    }

    let values = (seq * hidden) as f64;
    for t in 0..ntaps {
        let norm = dot[t * ntaps + t].sqrt();
        println!(
            "tap[{t}] layer {:>4}  l2 {norm:.4e}  min {:+.4e}  max {:+.4e}  nonfinite {:.3}%",
            layer_ids[t],
            lo[t],
            hi[t],
            100.0 * nonfinite[t] as f64 / values,
        );
        if norm.is_nan() || norm <= 0.0 {
            failures.push(format!("tap[{t}] (layer {}) l2 {norm}", layer_ids[t]));
        }
        if nonfinite[t] > 0 {
            failures.push(format!(
                "tap[{t}] (layer {}) has {} non-finite values",
                layer_ids[t], nonfinite[t]
            ));
        }
    }

    println!("tap cosine:");
    for t in 0..ntaps {
        print!("  tap[{t}]");
        for u in 0..ntaps {
            let (a, b) = (t.min(u), t.max(u));
            let cos = dot[a * ntaps + b] / (dot[t * ntaps + t].sqrt() * dot[u * ntaps + u].sqrt());
            print!(" {cos:>9.5}");
            if u > t && (cos.is_nan() || cos >= MAX_TAP_COSINE) {
                failures.push(format!(
                    "tap[{t}] (layer {}) and tap[{u}] (layer {}) cosine {cos:.5} >= {MAX_TAP_COSINE}",
                    layer_ids[t], layer_ids[u]
                ));
            }
        }
        println!();
    }
}

/// The trainer's `matmul_bt(last_hidden, lm_head)` against the engine's own
/// lm-head output at the same positions. Ranking agreement only, since the two
/// heads run at different precisions by design.
fn probe_logits(
    sample: &spec_train::trainer::Sample,
    last_hidden: &[f32],
    draft: &spec_train::backbone::Draft,
    engine: &infer_api::LoadedInferenceEngine,
    store: &mut autograd::TensorStore,
    failures: &mut Vec<String>,
) -> Result<()> {
    use autograd::{Tape, ops};

    let seq = sample.input_ids.len();
    let hidden = draft.cfg.hidden_size;
    // The last token has no next token, so it carries no target log-probability.
    let count = PROBE_POSITIONS.min(seq - 1);
    let positions: Vec<usize> = (0..count)
        .map(|i| i * (seq - 2) / (count - 1).max(1))
        .collect();

    let rows: Vec<f32> = positions
        .iter()
        .flat_map(|&p| last_hidden[p * hidden..][..hidden].to_vec())
        .collect();
    let rows = store.from_slice(&rows, &[count, hidden])?;
    let mut frozen = Tape::new();
    frozen.set_enabled(false);
    let ours = ops::matmul_bt(rows, draft.lm_head, store, &mut frozen)?;
    let ours = store.to_host(ours)?;
    ensure!(
        ours.len() % count == 0,
        "reconstruction is {} values over {count} positions",
        ours.len()
    );
    let our_vocab = ours.len() / count;

    println!(
        "oracle forward: {seq}×{our_vocab} logits, {:.1} GiB",
        (seq * our_vocab * 4) as f64 / (1u64 << 30) as f64
    );
    let oracle_positions: Vec<u32> = (0..seq as u32).collect();
    let raw = engine
        .forward_token_logits(&sample.input_ids, &oracle_positions)
        .context("engine oracle forward")?;
    ensure!(
        raw.seq_len() == seq,
        "engine returned {} logit rows for {seq} tokens",
        raw.seq_len()
    );
    let vocab = raw.vocab_size();
    ensure!(
        our_vocab == vocab,
        "reconstruction is {our_vocab} wide, the engine's logits {vocab}"
    );
    ensure!(
        positions
            .iter()
            .all(|&p| (sample.input_ids[p + 1] as usize) < vocab),
        "a sampled next token indexes past the {vocab}-wide logit row"
    );
    let reference = raw.to_host_f32()?;
    drop(raw);

    let mut argmax_hits = 0usize;
    let mut overlap = 0.0f64;
    let mut delta = 0.0f64;
    for (i, &p) in positions.iter().enumerate() {
        let ours = &ours[i * vocab..][..vocab];
        let reference = &reference[p * vocab..][..vocab];
        if argmax(ours) == argmax(reference) {
            argmax_hits += 1;
        }
        overlap += topk_overlap(ours, reference, PROBE_TOPK);
        let next = sample.input_ids[p + 1] as usize;
        delta += (log_softmax_at(ours, next) - log_softmax_at(reference, next)).abs();
    }
    let n = count as f64;
    let (agree, overlap, delta) = (argmax_hits as f64 / n, overlap / n, delta / n);
    println!(
        "target logits vs engine over {count} positions {}..{}: argmax {agree:.3}  \
         top-{PROBE_TOPK} overlap {overlap:.3}  mean |Δ log p(next)| {delta:.4}",
        positions[0],
        positions[count - 1],
    );

    if agree < MIN_ARGMAX_AGREE {
        failures.push(format!("argmax agreement {agree:.3} < {MIN_ARGMAX_AGREE}"));
    }
    if overlap < MIN_TOPK_OVERLAP {
        failures.push(format!(
            "top-{PROBE_TOPK} overlap {overlap:.3} < {MIN_TOPK_OVERLAP}"
        ));
    }
    if delta.is_nan() || delta > MAX_LOGPROB_DELTA {
        failures.push(format!(
            "mean |Δ log p(next)| {delta:.4} > {MAX_LOGPROB_DELTA}"
        ));
    }
    Ok(())
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}

/// `|top-k(a) ∩ top-k(b)| / k`.
fn topk_overlap(a: &[f32], b: &[f32], k: usize) -> f64 {
    let k = k.clamp(1, a.len());
    let top = |row: &[f32]| -> std::collections::HashSet<usize> {
        let mut idx: Vec<usize> = (0..row.len()).collect();
        idx.select_nth_unstable_by(k - 1, |&i, &j| row[j].total_cmp(&row[i]));
        idx[..k].iter().copied().collect()
    };
    top(a).intersection(&top(b)).count() as f64 / k as f64
}

/// `log_softmax(row)[i]`, max-subtracted.
fn log_softmax_at(row: &[f32], i: usize) -> f64 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = row.iter().map(|&x| f64::from(x - max).exp()).sum();
    f64::from(row[i] - max) - sum.ln()
}
