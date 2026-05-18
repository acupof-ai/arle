use std::{
    fs,
    path::{Path, PathBuf},
    process::ExitCode,
};

use anyhow::{Context, Result, anyhow, bail};
use deepseek_spec::{DeepSeekV4AttentionMode, DeepSeekV4Config};
use qwen35_spec::{LayerType, Qwen35Config};
use serde::Serialize;
use train::{
    model_family::{ModelFamily, resolve_model_family},
    tokenizer::ChatTokenizer,
};

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use crate::args::{ModelArgs, ModelCommand, ModelDownloadArgs, ModelSourceArg};
use crate::{
    args::{
        ModelFamilyArg, PretrainPresetArg, SaveDtypeArg, TrainArgs, TrainCommand, TrainEnvArgs,
        TrainEstimateMemoryArgs, TrainTestArgs,
    },
    hardware, hub_discovery,
};

const TRAIN_ENV_COMMANDS: &[&str] = &[
    "train env",
    "train test",
    "train estimate-memory",
    "train opd",
];

pub(crate) fn run_train(train: TrainArgs) -> ExitCode {
    match train.command {
        TrainCommand::Env(args) => exit_from_result(run_train_env(args)),
        TrainCommand::Test(args) => run_train_test(args),
        TrainCommand::EstimateMemory(args) => exit_from_result(run_train_estimate_memory(args)),
        TrainCommand::Opd(_) => run_opd(),
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
pub(crate) fn run_model(model: ModelArgs) -> ExitCode {
    match model.command {
        ModelCommand::Download(args) => run_model_download(args),
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn run_model_download(args: ModelDownloadArgs) -> ExitCode {
    let source_label = match args.source {
        ModelSourceArg::Hf => "hf",
        ModelSourceArg::Modelscope => "modelscope",
    };
    if args.render.dry_run {
        if args.render.json {
            println!(
                "{}",
                serde_json::json!({
                    "command": "model download",
                    "argv": [args.model_id],
                    "source": source_label,
                })
            );
        } else {
            println!("command model download");
            println!("argv {}", args.model_id);
            println!("source {source_label}");
        }
        return ExitCode::SUCCESS;
    }
    let result = match args.source {
        ModelSourceArg::Hf => crate::download::download_model_with_progress(&args.model_id),
        ModelSourceArg::Modelscope => {
            crate::modelscope::download_model_from_modelscope_with_progress(&args.model_id)
        }
    };
    match result {
        Ok(path) => {
            eprintln!(
                "[ARLE model download] downloaded ({source_label}) to: {}",
                path.display()
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("[ARLE model download] error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn run_train_env(args: TrainEnvArgs) -> Result<()> {
    let info = hardware::detect_system();
    let report = TrainEnvReport {
        version: env!("CARGO_PKG_VERSION"),
        train_default_backend: default_train_backend(),
        compiled_infer_backend: info.compiled_backend.name(),
        supports_inference: info.compiled_backend.supports_inference(),
        cpu: info.cpu_name,
        cpu_cores: info.cpu_cores,
        total_ram_gb: info.total_ram_gb,
        available_ram_gb: info.available_ram_gb,
        gpu: gpu_label(&info.gpu),
        hf_cache_root: hub_discovery::hub_cache_root()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "<unavailable>".to_string()),
        cwd: std::env::current_dir()
            .ok()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "<unknown>".to_string()),
        commands: TRAIN_ENV_COMMANDS,
    };
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    println!("ARLE train env");
    println!("version {}", report.version);
    println!("train default backend {}", report.train_default_backend);
    println!("compiled infer backend {}", report.compiled_infer_backend);
    println!("cpu {} · {} cores", report.cpu, report.cpu_cores);
    println!(
        "ram {:.1} GB total · {:.1} GB free",
        report.total_ram_gb, report.available_ram_gb
    );
    println!("gpu {}", report.gpu);
    println!("hf cache {}", report.hf_cache_root);
    println!("cwd {}", report.cwd);
    println!("commands {}", report.commands.join(", "));
    Ok(())
}

fn run_train_test(_args: TrainTestArgs) -> ExitCode {
    eprintln!(
        "[arle train test] OPD smoke fixture pending — the legacy \
         convert→pretrain→sft→eval pipeline was retired in the \
         2026-05-18 OPD-only pivot. Re-implementation lands with \
         the OPD substrate. See docs/projects/2026-05-18-opd-only-pivot.md."
    );
    ExitCode::from(0)
}

fn run_train_estimate_memory(args: TrainEstimateMemoryArgs) -> Result<()> {
    let report = if let Some(model_source) = args.model.as_deref() {
        estimate_from_model_dir(model_source, &args)?
    } else {
        estimate_from_scratch(&args)?
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    println!("ARLE train estimate-memory");
    println!("mode {}", report.mode);
    println!("family {}", report.family);
    if let Some(model_dir) = &report.model_dir {
        println!("model {}", model_dir);
    }
    if let Some(tokenizer_path) = &report.tokenizer_path {
        println!("tokenizer {}", tokenizer_path);
    }
    println!("params {}", format_count(report.param_count));
    println!(
        "trainable params {}",
        format_count(report.trainable_param_count)
    );
    println!("weights fp32 {}", format_bytes(report.weight_bytes_fp32));
    println!("grads fp32 {}", format_bytes(report.gradient_bytes_fp32));
    println!(
        "adam states fp32 {}",
        format_bytes(report.adam_state_bytes_fp32)
    );
    println!(
        "checkpoint {} {}",
        report.save_dtype,
        format_bytes(report.checkpoint_bytes)
    );
    if let Some(adapter_bytes) = report.adapter_checkpoint_bytes {
        println!("adapter checkpoint {}", format_bytes(adapter_bytes));
    }
    println!(
        "activation floor (batch={} seq={}) {}",
        report.batch,
        report.seq,
        format_bytes(report.activation_floor_bytes)
    );
    if let Some(vocab_size) = report.vocab_size {
        println!("vocab {}", vocab_size);
    }
    Ok(())
}

fn run_opd() -> ExitCode {
    eprintln!(
        "[ARLE train opd] error: OPD substrate landing next milestone; see \
         docs/projects/2026-05-18-opd-only-pivot.md"
    );
    ExitCode::FAILURE
}

fn estimate_from_model_dir(
    model_source: &Path,
    args: &TrainEstimateMemoryArgs,
) -> Result<EstimateMemoryReport> {
    let model = inspect_model_source(model_source, false)?;
    let local_dir = model.local_dir_path().ok_or_else(|| {
        anyhow!("estimate-memory requires a local model dir or cached HF model id")
    })?;
    let summary = inspect_resolved_model_dir(&local_dir)?;
    let trainable_params = lora_param_count(&summary.config, args.lora_rank);
    let checkpoint_bytes = bytes_for_params(summary.param_count, args.save_dtype.bytes_per_param());
    let adapter_checkpoint_bytes =
        bytes_for_params(trainable_params, args.save_dtype.bytes_per_param());
    Ok(EstimateMemoryReport {
        mode: "sft-lora".to_string(),
        family: summary.family.clone(),
        model_dir: Some(local_dir.display().to_string()),
        tokenizer_path: summary.tokenizer_path.clone(),
        vocab_size: Some(summary.vocab_size),
        batch: args.batch,
        seq: args.seq,
        param_count: summary.param_count,
        trainable_param_count: trainable_params,
        weight_bytes_fp32: bytes_for_params(summary.param_count, 4),
        gradient_bytes_fp32: bytes_for_params(trainable_params, 4),
        adam_state_bytes_fp32: bytes_for_params(trainable_params, 8),
        checkpoint_bytes,
        adapter_checkpoint_bytes: Some(adapter_checkpoint_bytes),
        activation_floor_bytes: activation_floor_bytes(summary.hidden_size, args.batch, args.seq),
        save_dtype: args.save_dtype.as_train_dtype().to_string(),
    })
}

fn estimate_from_scratch(args: &TrainEstimateMemoryArgs) -> Result<EstimateMemoryReport> {
    let tokenizer_source = args
        .tokenizer
        .as_deref()
        .ok_or_else(|| anyhow!("estimate-memory requires either --model or --tokenizer"))?;
    let tokenizer_path = resolve_local_tokenizer_path(tokenizer_source)?;
    let tokenizer = ChatTokenizer::from_file(&tokenizer_path)?;
    let mut shape = ScratchShape::default();
    if let Some(preset) = args.preset {
        shape.apply_preset(preset);
    }
    shape.apply_overrides(
        args.hidden,
        args.layers,
        args.heads,
        args.kv_heads,
        args.head_dim,
        args.intermediate,
        args.max_pos,
        args.linear_attn_every,
    );
    let vocab_size = args.vocab_size.unwrap_or_else(|| tokenizer.vocab_size());
    let family = args
        .model_family
        .unwrap_or(ModelFamilyArg::Qwen35)
        .as_train_family()
        .to_string();
    let param_count = qwen35_param_count(&shape.qwen35_config(vocab_size));
    let hidden_size = shape.hidden;
    Ok(EstimateMemoryReport {
        mode: "scratch-pretrain".to_string(),
        family,
        model_dir: None,
        tokenizer_path: Some(tokenizer_path.display().to_string()),
        vocab_size: Some(vocab_size),
        batch: args.batch,
        seq: args.seq,
        param_count,
        trainable_param_count: param_count,
        weight_bytes_fp32: bytes_for_params(param_count, 4),
        gradient_bytes_fp32: bytes_for_params(param_count, 4),
        adam_state_bytes_fp32: bytes_for_params(param_count, 8),
        checkpoint_bytes: bytes_for_params(param_count, args.save_dtype.bytes_per_param()),
        adapter_checkpoint_bytes: None,
        activation_floor_bytes: activation_floor_bytes(hidden_size, args.batch, args.seq),
        save_dtype: args.save_dtype.as_train_dtype().to_string(),
    })
}

fn inspect_model_source(source: &Path, allow_download: bool) -> Result<ModelInspection> {
    let raw_source = source.display().to_string();
    let resolved_dir = if allow_download {
        Some(resolve_model_dir_allow_download(source)?)
    } else {
        resolve_model_dir_local_only(source)
    };
    let mut notes = Vec::new();
    if !allow_download && resolved_dir.is_none() {
        notes.push(
            "model source is not local/cached; dry-run skipped remote resolution".to_string(),
        );
    }
    let summary = resolved_dir
        .as_deref()
        .map(inspect_resolved_model_dir)
        .transpose()?;

    Ok(ModelInspection {
        source: raw_source,
        resolved_dir: resolved_dir.as_ref().map(|path| path.display().to_string()),
        config_path: summary.as_ref().map(|s| s.config_path.clone()),
        tokenizer_path: summary.as_ref().and_then(|s| s.tokenizer_path.clone()),
        generation_config_path: summary
            .as_ref()
            .and_then(|s| s.generation_config_path.clone()),
        family: summary.as_ref().map(|s| s.family.clone()),
        notes,
    })
}

fn inspect_resolved_model_dir(model_dir: &Path) -> Result<ModelDirSummary> {
    let config_path = model_dir.join("config.json");
    let config_value: serde_json::Value = serde_json::from_str(&fs::read_to_string(&config_path)?)
        .with_context(|| {
            format!(
                "reading model inspection config from {}",
                config_path.display()
            )
        })?;
    let is_deepseek_v4 = config_value
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|model_type| model_type == "deepseek_v4")
        || config_value
            .get("architectures")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|architectures| {
                architectures
                    .iter()
                    .any(|arch| arch.as_str() == Some("DeepseekV4ForCausalLM"))
            });
    if is_deepseek_v4 {
        let cfg = DeepSeekV4Config::from_json_value(&config_value)?;
        return Ok(ModelDirSummary {
            family: "deepseek-v4".to_string(),
            config: ResolvedModelConfig::DeepSeekV4,
            config_path: config_path.display().to_string(),
            tokenizer_path: existing_display_path(model_dir.join("tokenizer.json")),
            generation_config_path: existing_display_path(model_dir.join("generation_config.json")),
            vocab_size: cfg.vocab_size,
            hidden_size: cfg.hidden_size,
            param_count: deepseek_v4_param_count(&cfg),
        });
    }

    let family = match resolve_model_family(&config_path, ModelFamily::Auto)? {
        ModelFamily::Qwen35 => "qwen35",
        ModelFamily::Auto => unreachable!("auto must resolve to a concrete family"),
    };
    match family {
        "qwen35" => {
            let cfg = Qwen35Config::from_json_file(&config_path)?;
            Ok(ModelDirSummary {
                family: "qwen35".to_string(),
                config: ResolvedModelConfig::Qwen35(Box::new(cfg.clone())),
                config_path: config_path.display().to_string(),
                tokenizer_path: existing_display_path(model_dir.join("tokenizer.json")),
                generation_config_path: existing_display_path(
                    model_dir.join("generation_config.json"),
                ),
                vocab_size: cfg.vocab_size,
                hidden_size: cfg.hidden_size,
                param_count: qwen35_param_count(&cfg),
            })
        }
        _ => unreachable!("family resolver returned an unknown family"),
    }
}

fn resolve_model_dir_allow_download(source: &Path) -> Result<PathBuf> {
    let source_text = source.display().to_string();
    infer::hf_hub::resolve_model_path(&source_text)
        .with_context(|| format!("resolving model source {source_text}"))
}

fn resolve_model_dir_local_only(source: &Path) -> Option<PathBuf> {
    let source_text = source.display().to_string();
    infer::hf_hub::resolve_local_model_path(&source_text)
}

fn resolve_local_tokenizer_path(source: &Path) -> Result<PathBuf> {
    if source.is_file() {
        return Ok(source.to_path_buf());
    }
    if source.is_dir() {
        let candidate = source.join("tokenizer.json");
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    let source_text = source.display().to_string();
    if let Some(model_dir) = infer::hf_hub::resolve_local_model_path(&source_text) {
        let candidate = model_dir.join("tokenizer.json");
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    bail!(
        "tokenizer source {} must be tokenizer.json or a local model dir containing tokenizer.json",
        source.display()
    );
}

fn qwen35_param_count(cfg: &Qwen35Config) -> u64 {
    let embed = mul_u64(cfg.vocab_size, cfg.hidden_size);
    let lm_head = if cfg.tie_word_embeddings { 0 } else { embed };
    let common = mul_u64(2, cfg.hidden_size)
        + mul_u64(cfg.hidden_size, cfg.intermediate_size) * 2
        + mul_u64(cfg.intermediate_size, cfg.hidden_size);
    let attention = cfg
        .layer_types
        .iter()
        .map(|layer_type| match layer_type {
            LayerType::FullAttention => {
                mul_u64(cfg.hidden_size, cfg.full_attn_q_proj_dim())
                    + mul_u64(cfg.hidden_size, cfg.full_attn_kv_dim()) * 2
                    + mul_u64(cfg.full_attn_q_dim(), cfg.hidden_size)
                    + mul_u64(2, cfg.head_dim)
            }
            LayerType::LinearAttention => {
                mul_u64(cfg.hidden_size, cfg.linear_attn_qkv_dim())
                    + mul_u64(cfg.hidden_size, cfg.linear_attn_z_dim())
                    + mul_u64(cfg.hidden_size, cfg.linear_num_value_heads) * 2
                    + mul_u64(cfg.linear_attn_qkv_dim(), cfg.linear_conv_kernel_dim)
                    + mul_u64(2, cfg.linear_num_value_heads)
                    + cfg.linear_value_head_dim as u64
                    + mul_u64(cfg.linear_attn_z_dim(), cfg.hidden_size)
            }
        })
        .sum::<u64>();
    embed
        + lm_head
        + (cfg.num_hidden_layers as u64).saturating_mul(common)
        + attention
        + cfg.hidden_size as u64
}

fn deepseek_v4_param_count(cfg: &DeepSeekV4Config) -> u64 {
    let embed = mul_u64(cfg.vocab_size, cfg.hidden_size);
    let lm_head = if cfg.tie_word_embeddings { 0 } else { embed };
    let hc_mix = (2 + cfg.hc_mult) * cfg.hc_mult;
    let hc_flat = cfg.hc_mult * cfg.hidden_size;
    let head_hc = mul_u64(cfg.hc_mult, hc_flat) + cfg.hc_mult as u64 + 1;
    let per_hc = mul_u64(hc_mix, hc_flat) + hc_mix as u64 + 3;
    let heads_per_group = cfg.num_attention_heads / cfg.o_groups;
    let base_attn = mul_u64(cfg.q_lora_rank, cfg.hidden_size)
        + cfg.q_lora_rank as u64
        + mul_u64(cfg.num_attention_heads * cfg.head_dim, cfg.q_lora_rank)
        + mul_u64(cfg.head_dim, cfg.hidden_size)
        + cfg.head_dim as u64
        + mul_u64(
            cfg.o_groups * cfg.o_lora_rank,
            heads_per_group * cfg.head_dim,
        )
        + mul_u64(cfg.hidden_size, cfg.o_groups * cfg.o_lora_rank)
        + cfg.num_attention_heads as u64;
    let expert = mul_u64(cfg.moe_intermediate_size, cfg.hidden_size) * 2
        + mul_u64(cfg.hidden_size, cfg.moe_intermediate_size);
    let routed_experts = (cfg.n_routed_experts as u64).saturating_mul(expert);
    let shared_experts = if cfg.n_shared_experts == 0 {
        0
    } else {
        let shared_intermediate = cfg.moe_intermediate_size * cfg.n_shared_experts;
        mul_u64(shared_intermediate, cfg.hidden_size) * 2
            + mul_u64(cfg.hidden_size, shared_intermediate)
    };
    let gate_bias_or_hash = cfg
        .n_routed_experts
        .max(cfg.vocab_size * cfg.num_experts_per_tok);
    let moe = mul_u64(cfg.n_routed_experts, cfg.hidden_size)
        + gate_bias_or_hash as u64
        + routed_experts
        + shared_experts;

    let layers = cfg
        .compress_ratios
        .iter()
        .copied()
        .map(|compress_ratio| {
            let compressor = cfg
                .compressor_shape(compress_ratio)
                .map(|shape| {
                    mul_u64(shape.wkv_rows, shape.wkv_cols)
                        + mul_u64(shape.wgate_rows, shape.wgate_cols)
                        + mul_u64(shape.ape_rows, shape.ape_cols)
                        + shape.norm_len as u64
                })
                .unwrap_or(0);
            let indexer = if cfg.attention_mode_for_compress_ratio(compress_ratio)
                == DeepSeekV4AttentionMode::CompressedSparse
            {
                let shape = cfg
                    .indexer_shape(compress_ratio)
                    .expect("CSA layer has indexer shape");
                mul_u64(shape.wq_b_rows, shape.wq_b_cols)
                    + mul_u64(shape.weights_proj_rows, shape.weights_proj_cols)
                    + mul_u64(shape.compressor.wkv_rows, shape.compressor.wkv_cols)
                    + mul_u64(shape.compressor.wgate_rows, shape.compressor.wgate_cols)
                    + mul_u64(shape.compressor.ape_rows, shape.compressor.ape_cols)
                    + shape.compressor.norm_len as u64
            } else {
                0
            };
            mul_u64(2, cfg.hidden_size) + per_hc * 2 + base_attn + compressor + indexer + moe
        })
        .sum::<u64>();

    let mtp = (cfg.num_nextn_predict_layers as u64).saturating_mul(
        mul_u64(7, cfg.hidden_size)
            + mul_u64(2, cfg.hidden_size * cfg.hidden_size)
            + per_hc * 2
            + head_hc
            + base_attn
            + moe,
    );

    embed + lm_head + cfg.hidden_size as u64 + head_hc + layers + mtp
}

fn lora_param_count(config: &ResolvedModelConfig, rank: usize) -> u64 {
    match config {
        ResolvedModelConfig::Qwen35(cfg) => {
            let common = lora_linear(cfg.hidden_size, cfg.intermediate_size, rank) * 2
                + lora_linear(cfg.intermediate_size, cfg.hidden_size, rank);
            let attention = cfg
                .layer_types
                .iter()
                .map(|layer_type| match layer_type {
                    LayerType::FullAttention => {
                        lora_linear(cfg.hidden_size, cfg.full_attn_q_proj_dim(), rank)
                            + lora_linear(cfg.hidden_size, cfg.full_attn_kv_dim(), rank) * 2
                            + lora_linear(cfg.full_attn_q_dim(), cfg.hidden_size, rank)
                    }
                    LayerType::LinearAttention => {
                        lora_linear(cfg.hidden_size, cfg.linear_attn_qkv_dim(), rank)
                            + lora_linear(cfg.hidden_size, cfg.linear_attn_z_dim(), rank)
                            + lora_linear(cfg.hidden_size, cfg.linear_num_value_heads, rank) * 2
                            + lora_linear(cfg.linear_attn_z_dim(), cfg.hidden_size, rank)
                    }
                })
                .sum::<u64>();
            (cfg.num_hidden_layers as u64).saturating_mul(common) + attention
        }
        ResolvedModelConfig::DeepSeekV4 => 0,
    }
}

fn activation_floor_bytes(hidden_size: usize, batch: usize, seq: usize) -> u64 {
    mul_u64(hidden_size, batch * seq * 4)
}

fn bytes_for_params(param_count: u64, bytes_per_param: u64) -> u64 {
    param_count.saturating_mul(bytes_per_param)
}

fn lora_linear(in_features: usize, out_features: usize, rank: usize) -> u64 {
    mul_u64(rank, in_features + out_features)
}

fn mul_u64(lhs: usize, rhs: usize) -> u64 {
    (lhs as u64).saturating_mul(rhs as u64)
}

fn existing_display_path(path: PathBuf) -> Option<String> {
    path.is_file().then(|| path.display().to_string())
}

fn format_count(count: u64) -> String {
    if count >= 1_000_000_000 {
        format!("{:.2}B", count as f64 / 1_000_000_000.0)
    } else if count >= 1_000_000 {
        format!("{:.2}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.2}K", count as f64 / 1_000.0)
    } else {
        count.to_string()
    }
}

fn format_bytes(bytes: u64) -> String {
    let kib = 1024.0;
    let mib = kib * 1024.0;
    let gib = mib * 1024.0;
    let bytes = bytes as f64;
    if bytes >= gib {
        format!("{:.2} GiB", bytes / gib)
    } else if bytes >= mib {
        format!("{:.2} MiB", bytes / mib)
    } else if bytes >= kib {
        format!("{:.2} KiB", bytes / kib)
    } else {
        format!("{} B", bytes as u64)
    }
}

fn exit_from_result(result: Result<()>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("[ARLE train] error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn gpu_label(info: &hardware::GpuInfo) -> String {
    match info {
        hardware::GpuInfo::Cuda { name, vram_gb } => format!("{name} ({vram_gb:.1} GB VRAM)"),
        hardware::GpuInfo::Metal {
            chip,
            unified_memory_gb,
        } => format!("{chip} ({unified_memory_gb:.1} GB unified)"),
        hardware::GpuInfo::None => "none".to_string(),
    }
}

fn default_train_backend() -> &'static str {
    #[cfg(feature = "cuda")]
    {
        return "cuda";
    }
    #[cfg(all(not(feature = "cuda"), feature = "metal"))]
    {
        return "metal";
    }
    #[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
    {
        "cpu"
    }
}

#[derive(Debug, Clone)]
struct ScratchShape {
    hidden: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    intermediate: usize,
    max_pos: usize,
    linear_attn_every: usize,
}

impl Default for ScratchShape {
    fn default() -> Self {
        Self {
            hidden: 256,
            layers: 4,
            heads: 4,
            kv_heads: 2,
            head_dim: 64,
            intermediate: 512,
            max_pos: 512,
            linear_attn_every: 0,
        }
    }
}

impl ScratchShape {
    fn apply_preset(&mut self, preset: PretrainPresetArg) {
        match preset {
            PretrainPresetArg::Tiny3m => {
                self.hidden = 96;
                self.layers = 2;
                self.heads = 3;
                self.kv_heads = 3;
                self.head_dim = 32;
                self.intermediate = 192;
                self.max_pos = 256;
                self.linear_attn_every = 0;
            }
            PretrainPresetArg::Small25m => {
                self.hidden = 160;
                self.layers = 2;
                self.heads = 5;
                self.kv_heads = 5;
                self.head_dim = 32;
                self.intermediate = 320;
                self.max_pos = 512;
                self.linear_attn_every = 0;
            }
            PretrainPresetArg::Small30m => {
                self.hidden = 192;
                self.layers = 2;
                self.heads = 6;
                self.kv_heads = 3;
                self.head_dim = 32;
                self.intermediate = 384;
                self.max_pos = 512;
                self.linear_attn_every = 0;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_overrides(
        &mut self,
        hidden: Option<usize>,
        layers: Option<usize>,
        heads: Option<usize>,
        kv_heads: Option<usize>,
        head_dim: Option<usize>,
        intermediate: Option<usize>,
        max_pos: Option<usize>,
        linear_attn_every: Option<usize>,
    ) {
        if let Some(hidden) = hidden {
            self.hidden = hidden;
        }
        if let Some(layers) = layers {
            self.layers = layers;
        }
        if let Some(heads) = heads {
            self.heads = heads;
        }
        if let Some(kv_heads) = kv_heads {
            self.kv_heads = kv_heads;
        }
        if let Some(head_dim) = head_dim {
            self.head_dim = head_dim;
        }
        if let Some(intermediate) = intermediate {
            self.intermediate = intermediate;
        }
        if let Some(max_pos) = max_pos {
            self.max_pos = max_pos;
        }
        if let Some(linear_attn_every) = linear_attn_every {
            self.linear_attn_every = linear_attn_every;
        }
    }

    fn qwen35_config(&self, vocab_size: usize) -> Qwen35Config {
        let mut layer_types = vec![LayerType::FullAttention; self.layers];
        if self.linear_attn_every > 0 {
            for (layer_idx, layer_type) in layer_types.iter_mut().enumerate().take(self.layers) {
                if (layer_idx + 1) % self.linear_attn_every == 0 {
                    *layer_type = LayerType::LinearAttention;
                }
            }
        }
        Qwen35Config {
            hidden_size: self.hidden,
            intermediate_size: self.intermediate,
            num_hidden_layers: self.layers,
            vocab_size,
            rms_norm_eps: 1.0e-6,
            stop_token_ids: vec![vocab_size.saturating_sub(1) as u32],
            bos_token_id: Some(1),
            eos_token_id: vocab_size.saturating_sub(1) as u32,
            tie_word_embeddings: true,
            num_attention_heads: self.heads,
            num_key_value_heads: self.kv_heads,
            head_dim: self.head_dim,
            linear_num_key_heads: self.heads,
            linear_key_head_dim: self.head_dim,
            linear_num_value_heads: self.heads,
            linear_value_head_dim: self.head_dim,
            linear_conv_kernel_dim: 4,
            rope_theta: 1_000_000.0,
            rope_scaling: None,
            partial_rotary_factor: 1.0,
            rotary_dim: self.head_dim,
            rope_cache_len_hint: Some(self.max_pos),
            layer_types,
            num_experts: 0,
            num_experts_per_tok: 0,
            decoder_sparse_step: 1,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            norm_topk_prob: true,
            mlp_only_layers: Vec::new(),
        }
    }
}

#[derive(Debug, Serialize)]
struct TrainEnvReport {
    version: &'static str,
    train_default_backend: &'static str,
    compiled_infer_backend: &'static str,
    supports_inference: bool,
    cpu: String,
    cpu_cores: usize,
    total_ram_gb: f64,
    available_ram_gb: f64,
    gpu: String,
    hf_cache_root: String,
    cwd: String,
    commands: &'static [&'static str],
}

#[derive(Debug, Clone, Serialize)]
struct ModelInspection {
    source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    resolved_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    config_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokenizer_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    family: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    notes: Vec<String>,
}

impl ModelInspection {
    fn local_dir_path(&self) -> Option<PathBuf> {
        self.resolved_dir.as_ref().map(PathBuf::from)
    }
}

#[derive(Debug)]
struct ModelDirSummary {
    family: String,
    config: ResolvedModelConfig,
    config_path: String,
    tokenizer_path: Option<String>,
    generation_config_path: Option<String>,
    vocab_size: usize,
    hidden_size: usize,
    param_count: u64,
}

#[derive(Debug)]
enum ResolvedModelConfig {
    Qwen35(Box<Qwen35Config>),
    DeepSeekV4,
}

#[derive(Debug, Serialize)]
struct EstimateMemoryReport {
    mode: String,
    family: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokenizer_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vocab_size: Option<usize>,
    batch: usize,
    seq: usize,
    param_count: u64,
    trainable_param_count: u64,
    weight_bytes_fp32: u64,
    gradient_bytes_fp32: u64,
    adam_state_bytes_fp32: u64,
    checkpoint_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    adapter_checkpoint_bytes: Option<u64>,
    activation_floor_bytes: u64,
    save_dtype: String,
}

impl SaveDtypeArg {
    fn bytes_per_param(self) -> u64 {
        match self {
            SaveDtypeArg::F32 => 4,
            SaveDtypeArg::Bf16 => 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{PretrainPresetArg, ScratchShape};

    #[test]
    fn small_30m_preset_applies_expected_shape() {
        let mut shape = ScratchShape::default();
        shape.apply_preset(PretrainPresetArg::Small30m);
        assert_eq!(shape.hidden, 192);
        assert_eq!(shape.layers, 2);
        assert_eq!(shape.heads, 6);
        assert_eq!(shape.kv_heads, 3);
        assert_eq!(shape.head_dim, 32);
    }
}
