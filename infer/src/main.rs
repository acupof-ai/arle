use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use infer::backend::cuda::bootstrap::{
    InferenceEngineOptions, ServerRuntimeConfig, detect_model_type,
    spawn_scheduler_handle_from_path,
};
use infer::http_server::build_app;
use infer::logging;
use infer::model::{KVCacheDtype, KVFormat};
use infer::scheduler::SchedulerConfig;
use infer::trace_reporter::FileReporter;
use log::info;

const DEFAULT_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");
const VALID_KV_CACHE_MODES: &str = "'bf16', 'fp8', 'int8', 'tq2', 'tq3', or 'tq4'";

#[derive(Parser)]
#[command(name = "infer", about = "Qwen3/3.5 GPU inference server")]
struct Args {
    /// Model directory containing config, tokenizer, and safetensor shards
    #[arg(long, default_value = DEFAULT_MODEL_PATH)]
    model_path: PathBuf,

    /// Port to listen on
    #[arg(long, default_value_t = 8000)]
    port: u16,

    /// Enable CUDA Graph capture/replay on decode path (`--cuda-graph=false` to disable)
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    cuda_graph: bool,

    /// Enable request tracing and write trace JSON files to this directory
    #[arg(long)]
    trace_output_path: Option<PathBuf>,

    /// Number of concurrent request slots (each gets its own KV cache).
    /// If unset, auto-computed from available GPU memory.
    #[arg(long)]
    num_slots: Option<usize>,

    /// Maximum sequence length (tokens) per KV cache slot. If unset, auto-computed
    /// from available GPU memory to fit all slots without OOM.
    #[arg(long)]
    max_seq_len: Option<usize>,

    /// Prefill chunk cap (tokens) when decode requests are active.
    /// Lower values reduce decode latency at the cost of prefill throughput.
    #[arg(long, default_value_t = 512)]
    decode_prefill_cap: usize,

    /// GPU memory (MB) reserved as headroom when auto-sizing KV cache.
    #[arg(long, default_value_t = 512)]
    gpu_reserved_mb: usize,

    /// KV pool headroom (MB) reserved during pool initialization.
    /// Covers BatchDecodeBuffers (~750MB), CUDA Graph captures (~500MB), and workspace.
    #[arg(long, default_value_t = 4096)]
    kv_pool_headroom_mb: usize,

    /// Minimum sequence length per slot when auto-sizing KV cache.
    #[arg(long, default_value_t = 256)]
    min_seq_len: usize,

    /// Fallback KV pool budget (MB) when GPU memory query fails.
    #[arg(long, default_value_t = 4096)]
    kv_pool_fallback_mb: usize,

    /// KV cache mode: "bf16" (default), "fp8", "int8", or TurboQuant pool
    /// modes "tq2"/"tq3"/"tq4". FP8 and TurboQuant keep the contiguous prefill
    /// cache in BF16 and quantize when migrating into the paged token pool.
    #[arg(long, default_value = "bf16")]
    kv_cache_dtype: String,
}

#[tokio::main]
async fn main() {
    logging::init_default();

    let args = Args::parse();

    if let Some(ref trace_path) = args.trace_output_path {
        std::fs::create_dir_all(trace_path).expect("Failed to create trace output directory");
        fastrace::set_reporter(
            FileReporter::new(trace_path.clone()),
            fastrace::collector::Config::default(),
        );
        info!("Tracing enabled: output_dir={}", trace_path.display());
    }

    let model_path = args
        .model_path
        .to_str()
        .expect("Model path must be valid UTF-8");
    let model_type = detect_model_type(model_path).expect("Failed to detect model type");
    info!("=== Infer Server - {} (GPU) ===", model_type);
    info!("Loading model...");
    let start = Instant::now();
    let num_slots = args
        .num_slots
        .unwrap_or_else(|| auto_num_slots(model_path, args.max_seq_len));

    info!(
        "Config: model_path={}, cuda_graph={}, num_slots={} ({}), kv_cache_mode={}",
        args.model_path.display(),
        args.cuda_graph,
        num_slots,
        if args.num_slots.is_some() {
            "explicit"
        } else {
            "auto"
        },
        args.kv_cache_dtype,
    );

    let (kv_cache_dtype, kv_pool_format) =
        parse_kv_cache_mode(&args.kv_cache_dtype).unwrap_or_else(|err| panic!("{err}"));
    info!("KV cache layout: contiguous={kv_cache_dtype:?}, paged_pool={kv_pool_format:?}");

    let runtime = ServerRuntimeConfig {
        engine: InferenceEngineOptions {
            enable_cuda_graph: args.cuda_graph,
        },
        scheduler: SchedulerConfig {
            decode_active_prefill_cap: args.decode_prefill_cap,
            gpu_reserved_bytes: args.gpu_reserved_mb.saturating_mul(1024 * 1024),
            kv_pool_headroom_bytes: args.kv_pool_headroom_mb.saturating_mul(1024 * 1024),
            min_seq_len: args.min_seq_len,
            kv_pool_fallback_bytes: args.kv_pool_fallback_mb.saturating_mul(1024 * 1024),
            ..SchedulerConfig::runtime_defaults(num_slots)
        },
        seed: 42,
        max_seq_len: args.max_seq_len,
        kv_cache_dtype,
        kv_pool_format,
    };

    let handle =
        spawn_scheduler_handle_from_path(model_path, runtime).expect("Failed to create scheduler");

    info!(
        "Model loaded: elapsed_ms={}, model_id={}",
        start.elapsed().as_millis(),
        handle.model_id()
    );

    let app = build_app(handle);

    let addr = format!("0.0.0.0:{}", args.port);
    info!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind to {addr}: {e}"));
    axum::serve(
        axum::serve::ListenerExt::tap_io(listener, |tcp_stream| {
            let _ = tcp_stream.set_nodelay(true);
        }),
        app,
    )
    .with_graceful_shutdown(shutdown_signal())
    .await
    .expect("Server error");

    if args.trace_output_path.is_some() {
        info!("Flushing pending traces...");
        fastrace::flush();
    }
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install CTRL+C handler");
    info!("Shutdown signal received");
}

fn parse_kv_cache_mode(mode: &str) -> std::result::Result<(KVCacheDtype, KVFormat), String> {
    let normalized = mode.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "bf16" => Ok((KVCacheDtype::BF16, KVFormat::BF16)),
        "fp8" => Ok((KVCacheDtype::BF16, KVFormat::FP8E4M3)),
        "int8" => Ok((KVCacheDtype::INT8, KVFormat::INT8)),
        "tq2" => Ok((
            KVCacheDtype::BF16,
            KVFormat::TurboQuant {
                key_bits: 2,
                val_bits: 2,
            },
        )),
        "tq3" => Ok((
            KVCacheDtype::BF16,
            KVFormat::TurboQuant {
                key_bits: 3,
                val_bits: 3,
            },
        )),
        "tq4" => Ok((
            KVCacheDtype::BF16,
            KVFormat::TurboQuant {
                key_bits: 4,
                val_bits: 4,
            },
        )),
        _ => Err(format!(
            "Invalid --kv-cache-dtype '{mode}': expected {VALID_KV_CACHE_MODES}"
        )),
    }
}

/// Auto-calculate num_slots from GPU memory and model config.
///
/// Strategy: estimate model weight size from safetensor files, subtract from GPU free
/// memory, divide remaining by per-slot cost (KV cache + recurrent state at target
/// sequence length). Clamp to [4, 128].
fn auto_num_slots(model_path: &str, max_seq_len: Option<usize>) -> usize {
    use infer::backend::cuda::tensor::DeviceContext;
    use std::path::Path;

    const DEFAULT_SEQ_LEN: usize = 4096;
    // Reserve memory for: BatchDecodeBuffers (~500MB), FlashInfer workspace (256MB),
    // CUDA Graph captures (~200MB), paged KV pool overhead, and safety margin.
    const RESERVED_BYTES: usize = 6 * 1024 * 1024 * 1024; // 6 GB headroom
    const MIN_SLOTS: usize = 4;
    const MAX_SLOTS: usize = 128;

    let seq_len = max_seq_len.unwrap_or(DEFAULT_SEQ_LEN);

    // Estimate model weight size from safetensor files on disk
    let weight_bytes: u64 = std::fs::read_dir(Path::new(model_path))
        .ok()
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
                .filter_map(|e| e.metadata().ok().map(|m| m.len()))
                .sum()
        })
        .unwrap_or(0);

    // Ensure CUDA context is initialized before querying memory.
    // DeviceContext::new() creates the cudarc CudaDevice which inits CUDA.
    let _ctx = match DeviceContext::new() {
        Ok(ctx) => ctx,
        Err(_) => {
            info!("auto_num_slots: CUDA init failed, using default 8 slots");
            return 8;
        }
    };

    let (free_bytes, _total) = match DeviceContext::gpu_memory_info() {
        Ok(info) => info,
        Err(_) => {
            info!("auto_num_slots: GPU memory query failed, using default 8 slots");
            return 8;
        }
    };

    // Available = GPU free - weights (already loaded by now? no, not yet) - reserved
    // Since we're called before model load, we estimate available after load
    let available = free_bytes
        .saturating_sub(weight_bytes as usize)
        .saturating_sub(RESERVED_BYTES);

    // Per-slot cost estimate: KV cache at seq_len tokens
    // Conservative estimate: 2 (K+V) * layers * heads * head_dim * 2 (bf16) * seq_len
    // Read from config.json for accuracy
    let per_slot_bytes = estimate_per_slot_bytes(model_path, seq_len);

    let slots = if per_slot_bytes > 0 {
        (available / per_slot_bytes).clamp(MIN_SLOTS, MAX_SLOTS)
    } else {
        8
    };

    info!(
        "auto_num_slots: gpu_free={:.1}GB, weights={:.1}GB, reserved={:.1}GB, \
         available={:.1}GB, per_slot={:.1}MB (seq_len={}), slots={}",
        free_bytes as f64 / 1e9,
        weight_bytes as f64 / 1e9,
        RESERVED_BYTES as f64 / 1e9,
        available as f64 / 1e9,
        per_slot_bytes as f64 / 1e6,
        seq_len,
        slots,
    );

    slots
}

/// Estimate per-slot memory cost from model config.json.
fn estimate_per_slot_bytes(model_path: &str, seq_len: usize) -> usize {
    use std::path::Path;

    let config_path = Path::new(model_path).join("config.json");
    let config_str = match std::fs::read_to_string(&config_path) {
        Ok(s) => s,
        Err(_) => return 0,
    };
    let config: serde_json::Value = match serde_json::from_str(&config_str) {
        Ok(v) => v,
        Err(_) => return 0,
    };

    let num_layers = config["num_hidden_layers"].as_u64().unwrap_or(32) as usize;
    let num_kv_heads = config["num_key_value_heads"].as_u64().unwrap_or(4) as usize;
    let head_dim = config["head_dim"].as_u64().unwrap_or(128) as usize;

    // Check if hybrid model (Qwen3.5): only full-attention layers use KV cache
    let num_full_attn = config["num_full_attention_layers"]
        .as_u64()
        .unwrap_or(num_layers as u64) as usize;
    let kv_layers = num_full_attn.min(num_layers);

    // Per-slot contiguous KV: 2 (K+V) * kv_layers * num_kv_heads * head_dim * 2 (bf16) * seq_len
    // This is pre-allocated by the scheduler for each slot.
    let kv_bytes = 2 * kv_layers * num_kv_heads * head_dim * 2 * seq_len;
    // Also account for paged KV pool share (roughly 1x contiguous cost per slot)
    let kv_bytes = kv_bytes * 2;

    // Recurrent state (if hybrid): per linear layer, fixed size independent of seq_len
    let num_linear_layers = num_layers.saturating_sub(kv_layers);
    let linear_key_dim = config["linear_key_head_dim"].as_u64().unwrap_or(128) as usize;
    let linear_val_dim = config["linear_value_head_dim"].as_u64().unwrap_or(128) as usize;
    let linear_val_heads = config["linear_num_value_heads"].as_u64().unwrap_or(32) as usize;
    let recurrent_bytes =
        num_linear_layers * linear_val_heads * linear_key_dim * linear_val_dim * 4; // f32

    kv_bytes + recurrent_bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_kv_cache_mode_supports_all_quantized_pool_modes() {
        assert_eq!(
            parse_kv_cache_mode("bf16").unwrap(),
            (KVCacheDtype::BF16, KVFormat::BF16)
        );
        assert_eq!(
            parse_kv_cache_mode("fp8").unwrap(),
            (KVCacheDtype::BF16, KVFormat::FP8E4M3)
        );
        assert_eq!(
            parse_kv_cache_mode("int8").unwrap(),
            (KVCacheDtype::INT8, KVFormat::INT8)
        );
        assert_eq!(
            parse_kv_cache_mode("tq2").unwrap(),
            (
                KVCacheDtype::BF16,
                KVFormat::TurboQuant {
                    key_bits: 2,
                    val_bits: 2
                }
            )
        );
        assert_eq!(
            parse_kv_cache_mode("tq3").unwrap(),
            (
                KVCacheDtype::BF16,
                KVFormat::TurboQuant {
                    key_bits: 3,
                    val_bits: 3
                }
            )
        );
        assert_eq!(
            parse_kv_cache_mode("tq4").unwrap(),
            (
                KVCacheDtype::BF16,
                KVFormat::TurboQuant {
                    key_bits: 4,
                    val_bits: 4
                }
            )
        );
    }

    #[test]
    fn parse_kv_cache_mode_is_case_insensitive() {
        assert_eq!(
            parse_kv_cache_mode("FP8").unwrap(),
            (KVCacheDtype::BF16, KVFormat::FP8E4M3)
        );
        assert_eq!(
            parse_kv_cache_mode("INT8").unwrap(),
            (KVCacheDtype::INT8, KVFormat::INT8)
        );
    }

    #[test]
    fn parse_kv_cache_mode_rejects_unknown_values() {
        let err = parse_kv_cache_mode("fp4").unwrap_err();
        assert!(err.contains("fp4"));
        assert!(err.contains(VALID_KV_CACHE_MODES));
    }
}
