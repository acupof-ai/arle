// Pretrain a Qwen3-layout model from random init on a plain-text corpus,
// saving checkpoints in the safetensors + config.json + tokenizer.json
// layout that infer/ can load. Mirrors train_sft.rs's save pipeline but
// starts from scratch (no source model dir) and drives a packed 1D forward
// over random corpus windows.

use std::{
    collections::HashSet,
    env, fs,
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
    time::Instant,
};

use autograd::{
    AutogradError, Backend, CpuBackend, SafetensorsRegistry, Tape, TensorId, TensorStore,
    ops::mul_scalar, optim::AdamW,
};
use serde_json::json;
use thiserror::Error;
use train::{
    cli_args::{ArgError, next_value, parse_value},
    dataset::LcgRng,
    qwen3::{Qwen3Config, Qwen3Error, Qwen3Model},
    tokenizer::ChatTokenizer,
    trainer::{clip_grad_norm, cross_entropy_loss},
};

// Qwen3 tokenizer defaults — infer/Config requires both fields.
const DEFAULT_BOS_TOKEN_ID: u32 = 151_643;
const DEFAULT_EOS_TOKEN_ID: u32 = 151_645;

const DEFAULT_BETAS: (f32, f32) = (0.9, 0.999);
const DEFAULT_EPS: f32 = 1.0e-8;
const DEFAULT_WEIGHT_DECAY: f32 = 0.01;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendChoice {
    Cpu,
    Metal,
    Cuda,
}

impl FromStr for BackendChoice {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "cuda" => Ok(Self::Cuda),
            _ => Err(format!("unknown backend: {value}")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SaveDtype {
    F32,
    Bf16,
}

impl FromStr for SaveDtype {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "f32" => Ok(Self::F32),
            "bf16" => Ok(Self::Bf16),
            _ => Err(format!("unknown save dtype: {value}")),
        }
    }
}

#[derive(Debug, Clone)]
struct CliArgs {
    corpus: PathBuf,
    tokenizer: PathBuf,
    out: PathBuf,
    steps: usize,
    batch: usize,
    seq: usize,
    lr: f32,
    log_every: usize,
    save_every: usize,
    eval_every: usize,
    eval_windows: usize,
    eval_frac: f32,
    resume: Option<PathBuf>,
    seed: u64,
    grad_clip: Option<f32>,
    backend: BackendChoice,
    save_dtype: SaveDtype,
    // Model hyperparams — vocab_size defaults to tokenizer.vocab_size().
    vocab_size: Option<usize>,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    tie_word_embeddings: bool,
    bos_token_id: u32,
    eos_token_id: u32,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            corpus: PathBuf::new(),
            tokenizer: PathBuf::new(),
            out: PathBuf::new(),
            steps: 100,
            batch: 1,
            seq: 128,
            lr: 3.0e-4,
            log_every: 5,
            save_every: 50,
            eval_every: 0,
            eval_windows: 8,
            eval_frac: 0.1,
            resume: None,
            seed: 0xC0FFEE,
            grad_clip: Some(1.0),
            backend: BackendChoice::Cpu,
            save_dtype: SaveDtype::Bf16,
            vocab_size: None,
            hidden_size: 256,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            intermediate_size: 512,
            max_position_embeddings: 512,
            rms_norm_eps: 1.0e-6,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: true,
            bos_token_id: DEFAULT_BOS_TOKEN_ID,
            eos_token_id: DEFAULT_EOS_TOKEN_ID,
        }
    }
}

#[derive(Debug, Error)]
enum CliError {
    #[error(transparent)]
    Autograd(#[from] AutogradError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Qwen3(#[from] Qwen3Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Arg(#[from] ArgError),
    #[error("{0}")]
    Custom(String),
}

fn main() -> Result<(), CliError> {
    let args = parse_args()?;
    validate_args(&args)?;
    fs::create_dir_all(&args.out)?;

    let tokenizer = ChatTokenizer::from_file(&args.tokenizer)?;
    let vocab_size = args.vocab_size.unwrap_or_else(|| tokenizer.vocab_size());

    let cfg = Qwen3Config {
        vocab_size,
        hidden_size: args.hidden_size,
        num_hidden_layers: args.num_hidden_layers,
        num_attention_heads: args.num_attention_heads,
        num_kv_heads: args.num_kv_heads,
        head_dim: args.head_dim,
        intermediate_size: args.intermediate_size,
        max_position_embeddings: args.max_position_embeddings,
        rms_norm_eps: args.rms_norm_eps,
        rope_theta: args.rope_theta,
        tie_word_embeddings: args.tie_word_embeddings,
    };
    if args.seq > cfg.max_position_embeddings {
        return Err(CliError::Custom(format!(
            "--seq {} exceeds --max-pos {}",
            args.seq, cfg.max_position_embeddings
        )));
    }

    let backend = build_backend(args.backend)?;
    println!("backend: {:?}", backend.device());
    println!(
        "config: vocab={} hidden={} layers={} heads={} kv_heads={} head_dim={} ffn={} max_pos={} tie_embed={}",
        cfg.vocab_size,
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_kv_heads,
        cfg.head_dim,
        cfg.intermediate_size,
        cfg.max_position_embeddings,
        cfg.tie_word_embeddings,
    );

    let mut store = TensorStore::with_backend(backend);
    let mut tape = Tape::new();
    let model = Qwen3Model::new(&cfg, &mut store)?;
    let mut registry = build_registry(&model);

    // --resume loads weights from a prior checkpoint (step_N dir). Optimizer
    // state (AdamW m/v) is NOT persisted yet — the optimizer reinitializes
    // with zeroed moments, so you may want a brief LR warmup after resume.
    let start_step = if let Some(resume_dir) = &args.resume {
        resume_from_checkpoint(resume_dir, &mut registry, &mut store, &cfg)?
    } else {
        0
    };
    if start_step > 0 {
        println!(
            "[pretrain_qwen3] resumed from {} at step {} (optimizer state reset)",
            args.resume.as_ref().unwrap().display(),
            start_step
        );
    }

    // Tokenize corpus once and validate IDs against configured vocab.
    let text = fs::read_to_string(&args.corpus).map_err(|e| {
        CliError::Custom(format!(
            "failed to read corpus {}: {e}",
            args.corpus.display()
        ))
    })?;
    let token_ids = tokenizer.encode(&text, false)?;
    if token_ids.len() <= args.seq {
        return Err(CliError::Custom(format!(
            "corpus has {} tokens but --seq is {}; need more tokens",
            token_ids.len(),
            args.seq
        )));
    }
    for &id in &token_ids {
        if (id as usize) >= cfg.vocab_size {
            return Err(CliError::Custom(format!(
                "token id {id} exceeds configured vocab_size {}",
                cfg.vocab_size
            )));
        }
    }
    // Hold out the last `eval_frac` slice for eval. The boundary is
    // deterministic from corpus length + eval_frac, so eval and train stay
    // reproducible across resumes of the same corpus.
    let eval_len = ((token_ids.len() as f32) * args.eval_frac).floor() as usize;
    let (train_tokens, eval_tokens) = if args.eval_every > 0 && eval_len > args.seq {
        let split = token_ids.len() - eval_len;
        let (train, eval) = token_ids.split_at(split);
        (train.to_vec(), eval.to_vec())
    } else {
        (token_ids.clone(), Vec::new())
    };
    if train_tokens.len() <= args.seq {
        return Err(CliError::Custom(format!(
            "train slice has {} tokens after eval split but --seq is {}; reduce --eval-frac or grow corpus",
            train_tokens.len(),
            args.seq
        )));
    }
    println!(
        "corpus: {} tokens from {} (train={} eval={})",
        token_ids.len(),
        args.corpus.display(),
        train_tokens.len(),
        eval_tokens.len(),
    );

    let model_ids = live_tensor_ids(&store);
    let params = trainable_params(&model, &store);
    if params.is_empty() {
        return Err(CliError::Custom(
            "qwen3 model exposed no trainable parameters".into(),
        ));
    }
    let param_count: usize = params
        .iter()
        .map(|&id| store.get(id).map_or(0, |t| t.size))
        .sum();
    println!(
        "params: {param_count} ({:.2}M)",
        param_count as f64 / 1_000_000.0
    );

    let mut optimizer = AdamW::new(args.lr, DEFAULT_BETAS, DEFAULT_EPS, DEFAULT_WEIGHT_DECAY);
    let mut rng = LcgRng::seed(args.seed ^ start_step as u64);
    let mut eval_rng = LcgRng::seed(args.seed ^ 0x4556_414C_5F50_5245);
    let window = args.seq + 1;
    let upper = train_tokens.len().saturating_sub(window) + 1;

    for step_offset in 0..args.steps {
        let step = start_step + step_offset;
        let step_start = Instant::now();
        optimizer.zero_grad(&params, &mut store);

        let mut step_loss = 0.0_f32;
        let loss_scale = 1.0_f32 / args.batch.max(1) as f32;

        for _micro in 0..args.batch {
            let start = (rng.next_u64() % upper as u64) as usize;
            let slice = &train_tokens[start..start + window];
            let input_ids: Vec<u32> = slice[..args.seq].to_vec();
            let targets: Vec<usize> = slice[1..].iter().map(|&t| t as usize).collect();
            let position_ids: Vec<u32> = (0..args.seq as u32).collect();

            tape.entries.clear();
            tape.set_enabled(true);

            let logits = model.forward(&mut store, &mut tape, &input_ids, &position_ids)?;
            let loss_id = cross_entropy_loss(logits, &targets, &mut store, &mut tape)?;
            let loss_value = store.to_host(loss_id)?[0];
            step_loss += loss_value;

            let backward_id = if (loss_scale - 1.0).abs() > f32::EPSILON {
                mul_scalar(loss_id, loss_scale, &mut store, &mut tape)?
            } else {
                loss_id
            };
            tape.backward(backward_id, &mut store)?;

            tape.entries.clear();
            tape.set_enabled(true);
            let keep = retained_ids(&model_ids, &params, &store);
            store.retain_ids(&keep);
        }

        if let Some(max_norm) = args.grad_clip {
            clip_grad_norm(&params, max_norm, &mut store);
        }
        optimizer.step(&params, &mut store);

        tape.entries.clear();
        tape.set_enabled(true);
        let keep = retained_ids(&model_ids, &params, &store);
        store.retain_ids(&keep);

        let mean_loss = step_loss / args.batch as f32;
        if step_offset % args.log_every == 0 || step_offset + 1 == args.steps {
            println!(
                "step {} loss {:.4} ms {:.2}",
                step,
                mean_loss,
                step_start.elapsed().as_secs_f64() * 1000.0
            );
        }

        if args.eval_every > 0
            && !eval_tokens.is_empty()
            && ((step_offset + 1) % args.eval_every == 0 || step_offset + 1 == args.steps)
        {
            let eval_loss = run_eval(
                &model,
                &mut store,
                &mut tape,
                &eval_tokens,
                args.seq,
                args.eval_windows,
                &mut eval_rng,
                &model_ids,
                &params,
            )?;
            println!(
                "eval @ step {} loss {:.4} windows {}",
                step, eval_loss, args.eval_windows
            );
        }

        if (step_offset + 1) % args.save_every == 0 || step_offset + 1 == args.steps {
            save_checkpoint(
                &args.out,
                step + 1,
                &registry,
                &mut store,
                &cfg,
                &args.tokenizer,
                args.bos_token_id,
                args.eos_token_id,
                args.save_dtype,
            )?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_eval(
    model: &Qwen3Model,
    store: &mut TensorStore,
    tape: &mut Tape,
    eval_tokens: &[u32],
    seq: usize,
    windows: usize,
    rng: &mut LcgRng,
    model_ids: &HashSet<TensorId>,
    params: &[TensorId],
) -> Result<f32, CliError> {
    let window = seq + 1;
    let upper = eval_tokens.len().saturating_sub(window) + 1;
    if upper == 0 {
        return Ok(f32::NAN);
    }
    let position_ids: Vec<u32> = (0..seq as u32).collect();
    let mut sum = 0.0_f32;
    let mut count = 0;
    for _ in 0..windows {
        let start = (rng.next_u64() % upper as u64) as usize;
        let slice = &eval_tokens[start..start + window];
        let input_ids: Vec<u32> = slice[..seq].to_vec();
        let targets: Vec<usize> = slice[1..].iter().map(|&t| t as usize).collect();

        tape.entries.clear();
        tape.set_enabled(false);
        let logits = model.forward(store, tape, &input_ids, &position_ids)?;
        let loss_id = cross_entropy_loss(logits, &targets, store, tape)?;
        sum += store.to_host(loss_id)?[0];
        count += 1;

        tape.entries.clear();
        let keep = retained_ids(model_ids, params, store);
        store.retain_ids(&keep);
    }
    tape.set_enabled(true);
    Ok(sum / count.max(1) as f32)
}

fn resume_from_checkpoint(
    resume_dir: &Path,
    registry: &mut SafetensorsRegistry,
    store: &mut TensorStore,
    cfg: &Qwen3Config,
) -> Result<usize, CliError> {
    let weights = resume_dir.join("model.safetensors");
    if !weights.exists() {
        return Err(CliError::Custom(format!(
            "resume path {} has no model.safetensors",
            resume_dir.display()
        )));
    }

    // Verify the checkpoint's config.json matches the requested config so a
    // silent shape mismatch doesn't surface as a mid-training crash.
    let cfg_path = resume_dir.join("config.json");
    if cfg_path.exists() {
        let file_cfg: serde_json::Value = serde_json::from_str(&fs::read_to_string(&cfg_path)?)?;
        let mismatches: Vec<String> = [
            ("hidden_size", cfg.hidden_size as i64),
            ("intermediate_size", cfg.intermediate_size as i64),
            ("num_hidden_layers", cfg.num_hidden_layers as i64),
            ("num_attention_heads", cfg.num_attention_heads as i64),
            ("num_key_value_heads", cfg.num_kv_heads as i64),
            ("head_dim", cfg.head_dim as i64),
            ("vocab_size", cfg.vocab_size as i64),
        ]
        .iter()
        .filter_map(|(k, v)| match file_cfg.get(*k).and_then(|x| x.as_i64()) {
            Some(seen) if seen != *v => Some(format!("{k}: ckpt={seen} cli={v}")),
            _ => None,
        })
        .collect();
        if !mismatches.is_empty() {
            return Err(CliError::Custom(format!(
                "resume config mismatch: {}",
                mismatches.join(", ")
            )));
        }
    }

    registry.load_into(store, &weights)?;

    // Derive absolute step from the dir name `step_<N>` if present; otherwise 0.
    let start_step = resume_dir
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|s| s.strip_prefix("step_"))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    Ok(start_step)
}

fn parse_args() -> Result<CliArgs, CliError> {
    let mut args = CliArgs::default();
    let mut iter = env::args().skip(1);
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--corpus" => args.corpus = PathBuf::from(next_value(&mut iter, &flag)?),
            "--tokenizer" => args.tokenizer = PathBuf::from(next_value(&mut iter, &flag)?),
            "--out" => args.out = PathBuf::from(next_value(&mut iter, &flag)?),
            "--steps" => args.steps = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--batch" => args.batch = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--seq" => args.seq = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--lr" => args.lr = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--log-every" => args.log_every = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--save-every" => args.save_every = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--eval-every" => args.eval_every = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--eval-windows" => {
                args.eval_windows = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--eval-frac" => args.eval_frac = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--resume" => args.resume = Some(PathBuf::from(next_value(&mut iter, &flag)?)),
            "--seed" => args.seed = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--grad-clip" => {
                args.grad_clip = Some(parse_value(&flag, next_value(&mut iter, &flag)?)?);
            }
            "--no-grad-clip" => args.grad_clip = None,
            "--backend" => {
                args.backend = next_value(&mut iter, &flag)?.parse().map_err(|value| {
                    CliError::Arg(ArgError::InvalidValue {
                        flag: flag.clone(),
                        value,
                    })
                })?;
            }
            "--save-dtype" => {
                args.save_dtype = next_value(&mut iter, &flag)?.parse().map_err(|value| {
                    CliError::Arg(ArgError::InvalidValue {
                        flag: flag.clone(),
                        value,
                    })
                })?;
            }
            "--vocab-size" => {
                args.vocab_size = Some(parse_value(&flag, next_value(&mut iter, &flag)?)?);
            }
            "--hidden" => args.hidden_size = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--layers" => {
                args.num_hidden_layers = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--heads" => {
                args.num_attention_heads = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--kv-heads" => args.num_kv_heads = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--head-dim" => args.head_dim = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--intermediate" => {
                args.intermediate_size = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--max-pos" => {
                args.max_position_embeddings = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--rms-eps" => args.rms_norm_eps = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--rope-theta" => args.rope_theta = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--no-tie-embed" => args.tie_word_embeddings = false,
            "--bos-token-id" => {
                args.bos_token_id = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--eos-token-id" => {
                args.eos_token_id = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            _ => return Err(CliError::Arg(ArgError::UnknownFlag(flag))),
        }
    }
    Ok(args)
}

fn validate_args(args: &CliArgs) -> Result<(), CliError> {
    if args.corpus.as_os_str().is_empty() {
        return Err(CliError::Custom("--corpus is required".into()));
    }
    if args.tokenizer.as_os_str().is_empty() {
        return Err(CliError::Custom("--tokenizer is required".into()));
    }
    if args.out.as_os_str().is_empty() {
        return Err(CliError::Custom("--out is required".into()));
    }
    for (flag, value) in [
        ("--steps", args.steps),
        ("--batch", args.batch),
        ("--seq", args.seq),
        ("--log-every", args.log_every),
        ("--save-every", args.save_every),
        ("--hidden", args.hidden_size),
        ("--layers", args.num_hidden_layers),
        ("--heads", args.num_attention_heads),
        ("--kv-heads", args.num_kv_heads),
        ("--head-dim", args.head_dim),
        ("--intermediate", args.intermediate_size),
        ("--max-pos", args.max_position_embeddings),
    ] {
        if value == 0 {
            return Err(CliError::Arg(ArgError::InvalidValue {
                flag: flag.into(),
                value: "0".into(),
            }));
        }
    }
    if !(0.0..1.0).contains(&args.eval_frac) {
        return Err(CliError::Arg(ArgError::InvalidValue {
            flag: "--eval-frac".into(),
            value: args.eval_frac.to_string(),
        }));
    }
    if args.eval_every > 0 && args.eval_windows == 0 {
        return Err(CliError::Arg(ArgError::InvalidValue {
            flag: "--eval-windows".into(),
            value: "0".into(),
        }));
    }
    Ok(())
}

fn build_backend(choice: BackendChoice) -> Result<Arc<dyn Backend>, CliError> {
    match choice {
        BackendChoice::Cpu => Ok(Arc::new(CpuBackend)),
        #[cfg(feature = "metal")]
        BackendChoice::Metal => Ok(Arc::new(autograd::backend_metal::MetalBackend)),
        #[cfg(not(feature = "metal"))]
        BackendChoice::Metal => {
            eprintln!(
                "[pretrain_qwen3] warning: metal backend requested without --features metal; falling back to cpu"
            );
            Ok(Arc::new(CpuBackend))
        }
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => Ok(Arc::new(autograd::backend_cuda::CudaBackend::new(0)?)),
        #[cfg(not(feature = "cuda"))]
        BackendChoice::Cuda => {
            eprintln!(
                "[pretrain_qwen3] warning: cuda backend requested without --features cuda; falling back to cpu"
            );
            Ok(Arc::new(CpuBackend))
        }
    }
}

fn build_registry(model: &Qwen3Model) -> SafetensorsRegistry {
    let mut registry = SafetensorsRegistry::new();
    for (name, tensor_id) in model.param_name_map() {
        registry.insert(name, tensor_id);
    }
    registry
}

fn trainable_params(model: &Qwen3Model, store: &TensorStore) -> Vec<TensorId> {
    let mut params = model
        .param_name_map()
        .into_values()
        .collect::<HashSet<_>>()
        .into_iter()
        .filter(|id| store.get(*id).is_some_and(|t| t.requires_grad))
        .collect::<Vec<_>>();
    params.sort_unstable();
    params
}

fn live_tensor_ids(store: &TensorStore) -> HashSet<TensorId> {
    store
        .tensors
        .iter()
        .enumerate()
        .filter_map(|(id, slot)| slot.as_ref().map(|_| id))
        .collect()
}

fn retained_ids(
    model_ids: &HashSet<TensorId>,
    params: &[TensorId],
    store: &TensorStore,
) -> HashSet<TensorId> {
    let mut keep = model_ids.clone();
    for &param_id in params {
        keep.insert(param_id);
        if let Some(grad_id) = store.get(param_id).and_then(|t| t.grad) {
            keep.insert(grad_id);
        }
    }
    keep
}

#[allow(clippy::too_many_arguments)]
fn save_checkpoint(
    out_dir: &Path,
    step: usize,
    registry: &SafetensorsRegistry,
    store: &mut TensorStore,
    cfg: &Qwen3Config,
    tokenizer_path: &Path,
    bos_token_id: u32,
    eos_token_id: u32,
    save_dtype: SaveDtype,
) -> Result<(), CliError> {
    let step_dir = out_dir.join(format!("step_{step}"));
    fs::create_dir_all(&step_dir)?;

    let config_json = json!({
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_kv_heads,
        "head_dim": cfg.head_dim,
        "vocab_size": cfg.vocab_size,
        "rms_norm_eps": cfg.rms_norm_eps,
        "rope_theta": cfg.rope_theta,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "tie_word_embeddings": cfg.tie_word_embeddings,
        "max_position_embeddings": cfg.max_position_embeddings,
        "torch_dtype": match save_dtype { SaveDtype::F32 => "float32", SaveDtype::Bf16 => "bfloat16" },
    });
    fs::write(
        step_dir.join("config.json"),
        serde_json::to_string_pretty(&config_json)?,
    )?;
    fs::copy(tokenizer_path, step_dir.join("tokenizer.json"))?;

    let weights_path = step_dir.join("model.safetensors");
    match save_dtype {
        SaveDtype::F32 => registry.save_from(store, &weights_path)?,
        SaveDtype::Bf16 => registry.save_from_bf16(store, &weights_path)?,
    }
    println!(
        "[pretrain_qwen3] saved step {} to {} (dtype: {:?})",
        step,
        step_dir.display(),
        save_dtype
    );
    Ok(())
}
