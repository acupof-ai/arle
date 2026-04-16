use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use infer::backend::cuda::bootstrap::{
    InferenceEngineOptions, ServerRuntimeConfig, detect_model_type,
    spawn_scheduler_handle_from_path,
};
use infer::http_server::build_app_with_metrics;
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

    /// Fraction of total GPU memory for weights + KV cache (SGLang-compatible).
    /// The remaining (1 - fraction) is headroom for activations, CUDA graphs,
    /// FlashInfer workspace, and OS. Default 0.88 matches SGLang's auto-detect.
    /// Increase to 0.92 on dedicated inference boxes; decrease to 0.80 if sharing GPU.
    #[arg(long, default_value_t = 0.88)]
    mem_fraction_static: f64,

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
    let (kv_cache_dtype, kv_pool_format) =
        parse_kv_cache_mode(&args.kv_cache_dtype).unwrap_or_else(|err| panic!("{err}"));

    let num_slots = args.num_slots.unwrap_or_else(|| {
        auto_num_slots(
            model_path,
            args.max_seq_len,
            kv_pool_format,
            args.mem_fraction_static,
        )
    });

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
    info!("KV cache layout: contiguous={kv_cache_dtype:?}, paged_pool={kv_pool_format:?}");

    let runtime = ServerRuntimeConfig {
        engine: InferenceEngineOptions {
            enable_cuda_graph: args.cuda_graph,
        },
        scheduler: SchedulerConfig {
            decode_active_prefill_cap: args.decode_prefill_cap,
            mem_fraction_static: args.mem_fraction_static,
            min_seq_len: args.min_seq_len,
            kv_pool_fallback_bytes: args.kv_pool_fallback_mb.saturating_mul(1024 * 1024),
            ..SchedulerConfig::runtime_defaults(num_slots)
        },
        seed: 42,
        max_seq_len: args.max_seq_len,
        kv_cache_dtype,
        kv_pool_format,
    };

    let metrics = infer::metrics::ServerMetrics::new(model_path);
    let handle = spawn_scheduler_handle_from_path(model_path, runtime, metrics.clone())
        .expect("Failed to create scheduler");

    info!(
        "Model loaded: elapsed_ms={}, model_id={}",
        start.elapsed().as_millis(),
        handle.model_id()
    );

    let app = build_app_with_metrics(handle, metrics);

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
/// memory, take out the pool-side reserves the user passed via CLI flags, then
/// divide the remainder by the per-slot KV-cache cost at the requested dtype.
/// Clamp to [4, 128].
///
/// **Dtype awareness** (2026-04-15): the per-slot estimate now respects
/// `kv_pool_format`, so INT8 / FP8 quant pools auto-size to roughly twice the
/// number of slots BF16 picks at the same `max_seq_len`. Without this, the
/// auto-sizer was bf16-blind and quant KV silently lost its capacity benefit
/// at default flags. See
/// `docs/experience/wins/2026-04-15-bench-hbm-peak-throughput.md` for the
/// HBM inventory that surfaced this.
///
/// SGLang-compatible memory budget: `total_budget = gpu_total × mem_fraction_static`.
/// KV budget = total_budget − weight_size. Single knob, no multi-parameter tuning.
fn auto_num_slots(
    model_path: &str,
    max_seq_len: Option<usize>,
    kv_pool_format: KVFormat,
    mem_fraction_static: f64,
) -> usize {
    use infer::backend::cuda::tensor::DeviceContext;
    use std::path::Path;

    const DEFAULT_SEQ_LEN: usize = 4096;
    const CONTIGUOUS_CHUNK_SIZE: usize = 512;
    const MIN_SLOTS: usize = 4;
    const MAX_SLOTS: usize = 128;

    let seq_len = max_seq_len.unwrap_or(DEFAULT_SEQ_LEN);

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

    let _ctx = match DeviceContext::new() {
        Ok(ctx) => ctx,
        Err(_) => {
            info!("auto_num_slots: CUDA init failed, using default 8 slots");
            return 8;
        }
    };

    let (_free_bytes, total_bytes) = match DeviceContext::gpu_memory_info() {
        Ok(info) => info,
        Err(_) => {
            info!("auto_num_slots: GPU memory query failed, using default 8 slots");
            return 8;
        }
    };

    // SGLang formula: total_budget = gpu_total × fraction, kv_budget = total_budget − weights
    let total_budget = (total_bytes as f64 * mem_fraction_static) as usize;
    let kv_budget = total_budget.saturating_sub(weight_bytes as usize);

    let per_slot_bytes =
        estimate_per_slot_bytes(model_path, seq_len, CONTIGUOUS_CHUNK_SIZE, kv_pool_format);

    let slots = if per_slot_bytes > 0 {
        (kv_budget / per_slot_bytes).clamp(MIN_SLOTS, MAX_SLOTS)
    } else {
        8
    };

    let headroom_gb = (total_bytes as f64 * (1.0 - mem_fraction_static)) / 1e9;
    info!(
        "auto_num_slots: gpu_total={:.1}GB, weights={:.1}GB, fraction={:.0}%, \
         headroom={:.1}GB, kv_budget={:.1}GB, per_slot={:.1}MB, slots={}",
        total_bytes as f64 / 1e9,
        weight_bytes as f64 / 1e9,
        mem_fraction_static * 100.0,
        headroom_gb,
        kv_budget as f64 / 1e9,
        per_slot_bytes as f64 / 1e6,
        slots,
    );

    slots
}

/// Estimate per-slot memory cost from model config.json.
///
/// `kv_pool_format` is consulted for the contiguous KV byte width so INT8 and
/// FP8 quant pools auto-size to the smaller per-token footprint instead of
/// being charged as bf16. The recurrent state (Qwen3.5 hybrid models) is
/// always f32 regardless of the KV format choice.
fn estimate_per_slot_bytes(
    model_path: &str,
    seq_len: usize,
    chunk_size: usize,
    kv_pool_format: KVFormat,
) -> usize {
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

    // Per-slot contiguous KV bytes, dtype-aware via
    // KVFormat::pool_bytes_per_kv_head (BF16=2*head_dim, INT8=head_dim+4
    // including per-token f32 scale, FP8=head_dim, TurboQuant=packed+norms).
    let bytes_per_kv_head_side = kv_pool_format.pool_bytes_per_kv_head(head_dim);
    // Per-slot cost = contiguous working buffer (chunk_size) + paged pool share (full seq_len).
    // Contiguous is the small prefill chunk; paged covers the full sequence.
    let bytes_per_token_kv = 2 * kv_layers * num_kv_heads * bytes_per_kv_head_side;
    let kv_bytes = bytes_per_token_kv * chunk_size + bytes_per_token_kv * seq_len;

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
