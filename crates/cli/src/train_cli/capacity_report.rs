use std::path::Path;

use anyhow::{Result, anyhow};
use qwen35_spec::LayerType;
use serde::Serialize;

use super::model_probe::{
    ResolvedModelConfig, inspect_model_source, inspect_resolved_model_dir, mul_u64,
};
use crate::{
    args::{SaveDtypeArg, TrainEnvArgs, TrainEstimateMemoryArgs},
    hardware, hub_discovery,
};

const TRAIN_ENV_COMMANDS: &[&str] = &["train env", "train estimate-memory", "train opd"];

pub(super) fn run_train_env(args: TrainEnvArgs) -> Result<()> {
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

pub(super) fn run_train_estimate_memory(args: TrainEstimateMemoryArgs) -> Result<()> {
    let model_source = args
        .model
        .as_deref()
        .ok_or_else(|| anyhow!("estimate-memory requires --model"))?;
    let report = estimate_from_model_dir(model_source, &args)?;

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

fn gpu_label(info: &hardware::GpuInfo) -> String {
    match info {
        hardware::GpuInfo::Cuda { name, vram_gb } => format!("{name} ({vram_gb:.1} GB VRAM)"),
        hardware::GpuInfo::Metal {
            chip,
            unified_memory_gb,
            recommended_working_set_gb,
        } => {
            if let Some(working_set) = recommended_working_set_gb {
                format!(
                    "{chip} ({unified_memory_gb:.1} GB unified, {working_set:.1} GB working set)"
                )
            } else {
                format!("{chip} ({unified_memory_gb:.1} GB unified)")
            }
        }
        hardware::GpuInfo::None => "none".to_string(),
    }
}

// cfg arms are additive: in single-backend builds only one `return` is live, so
// clippy sees it as needless; the `return`s are required so the multi-backend
// (`cuda` + `metal`) build still compiles. Matches `CompiledBackend::detect`.
#[allow(clippy::needless_return)]
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
