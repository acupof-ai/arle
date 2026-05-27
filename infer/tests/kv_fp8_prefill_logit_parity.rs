#![cfg(feature = "cuda")]

//! FP8 vs BF16 prefill-logit parity isolation diagnostic.
//!
//! The 2026-05-26 cross-precision parity audit
//! (`infer/tests/kv_precision_parity.rs`) reproduces the 2026-05-02 FP8 KV
//! token-1 catastrophic divergence end-to-end. The two FP8 quantize kernels
//! (`quantize_scatter_kv_fp8_range`, `quantize_paged_kv_fp8`) test clean in
//! isolation at production Qwen3-4B layout. The divergence must therefore
//! come from the dispatch/wiring around those kernels.
//!
//! This test isolates whether the prefill path is broken. Both BF16 and FP8
//! KV modes call `forward_token_logits`, a one-shot prefill-style compute,
//! on the same `input_ids`. If the per-position logits agree numerically
//! (within FP8 quantization noise — see the kernel diagnostics that bound
//! the per-(token, head) error around max_abs ≈ 0.11), the prefill path is
//! clean and the audit's step-1 divergence is downstream in the per-decode
//! quantize + `decode_attention_fp8` reads. If they diverge, the bug is in
//! the paged prefill kernel's interaction with the FP8 finalize.
//!
//! Skipped if the model dir is missing; run with `cargo test --release -p
//! infer --features cuda --test kv_fp8_prefill_logit_parity -- --nocapture
//! --test-threads=1`.

use std::path::Path;

use anyhow::{Context, Result};
use log::info;

use infer::backend::cuda::bootstrap::ServerRuntimeConfig;
use infer::model::{KVCacheDtype, KVFormat};
use infer::server_engine::LoadedInferenceEngine;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");
const INPUT_IDS: &[u32] = &[
    151644, 872, 198, 785, 468, 3092, 301, 21938, 374, 151645, 198, 151644, 77091, 198,
];

fn get_model_path() -> String {
    std::env::var("INFER_TEST_MODEL_PATH").unwrap_or_else(|_| MODEL_PATH.to_string())
}

fn init_logging() {
    use std::sync::OnceLock;
    static LOGGER: OnceLock<()> = OnceLock::new();
    LOGGER.get_or_init(|| {
        infer::logging::init_stderr("info");
    });
}

fn run_one(
    model_path: &str,
    kv_cache_dtype: KVCacheDtype,
    kv_pool_format: KVFormat,
    label: &str,
) -> Result<Vec<f32>> {
    info!(
        "kv_fp8_prefill_logit_parity: booting {label} (dtype={kv_cache_dtype:?} format={kv_pool_format:?})"
    );
    let runtime = ServerRuntimeConfig {
        kv_cache_dtype,
        kv_pool_format,
        ..Default::default()
    };
    let engine = LoadedInferenceEngine::load_with_runtime_config(model_path, runtime)
        .with_context(|| format!("load {label}"))?;

    let positions: Vec<u32> = (0..INPUT_IDS.len() as u32).collect();
    let logits = engine
        .forward_token_logits(INPUT_IDS, &positions)
        .with_context(|| format!("{label} forward_token_logits"))?;

    // Pull host BF16 of the final-position logit vector and convert to f32.
    let host = logits.to_host_f32().context("to_host_f32")?;
    let vocab = logits.vocab_size();
    assert_eq!(host.len(), INPUT_IDS.len() * vocab);
    let last_start = (INPUT_IDS.len() - 1) * vocab;
    Ok(host[last_start..last_start + vocab].to_vec())
}

#[test]
fn fp8_vs_bf16_prefill_logits_parity() -> Result<()> {
    init_logging();
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!(
            "skipping fp8_vs_bf16_prefill_logits_parity: model path missing ({})",
            model_path
        );
        return Ok(());
    }

    let bf16 = run_one(&model_path, KVCacheDtype::BF16, KVFormat::BF16, "bf16")?;
    let fp8 = run_one(&model_path, KVCacheDtype::BF16, KVFormat::FP8E4M3, "fp8")?;
    assert_eq!(bf16.len(), fp8.len(), "vocab size must match");

    // Per-element delta stats over the final-position logit vector.
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut max_rel = 0.0f32;
    let mut top1_bf16_idx = 0usize;
    let mut top1_fp8_idx = 0usize;
    let mut top1_bf16_val = f32::MIN;
    let mut top1_fp8_val = f32::MIN;
    for (i, (&b, &f)) in bf16.iter().zip(fp8.iter()).enumerate() {
        let d = (b - f).abs();
        let r = if b.abs() > 1.0e-6 { d / b.abs() } else { 0.0 };
        if d > max_abs {
            max_abs = d;
        }
        if r > max_rel {
            max_rel = r;
        }
        sum_abs += d as f64;
        if b > top1_bf16_val {
            top1_bf16_val = b;
            top1_bf16_idx = i;
        }
        if f > top1_fp8_val {
            top1_fp8_val = f;
            top1_fp8_idx = i;
        }
    }
    let mean_abs = sum_abs / bf16.len() as f64;
    let argmax_match = top1_bf16_idx == top1_fp8_idx;

    eprintln!(
        "fp8_vs_bf16_prefill_logits_parity: vocab={} max_abs={max_abs:.6} \
         mean_abs={mean_abs:.6} max_rel={max_rel:.6} argmax_bf16={top1_bf16_idx} \
         argmax_fp8={top1_fp8_idx} argmax_match={argmax_match} \
         top1_bf16_val={top1_bf16_val:.4} top1_fp8_val={top1_fp8_val:.4}",
        bf16.len()
    );

    // Diagnostic floor: if `argmax_match=false`, the FP8 prefill path
    // produces a different best-next-token than BF16 even though the
    // prefill is supposed to use BF16 K/V values during the attention
    // compute. That isolates the bug to FP8's prefill-time wiring (most
    // likely `finalize_paged_prefill_kv_layer` running BEFORE the prefill
    // attention reads, or scale/index plumbing that contaminates the
    // attention compute). If `argmax_match=true` and `max_abs` is small,
    // the prefill is bit-clean and the parity audit's step-1 divergence
    // lives strictly in the per-decode-step quantize + `decode_attention_fp8`
    // reads. We do not gate the test on a specific delta; we want the
    // numbers in the bench output regardless.
    assert!(
        bf16.iter().all(|v| v.is_finite()),
        "BF16 logits contain NaN/Inf"
    );
    assert!(
        fp8.iter().all(|v| v.is_finite()),
        "FP8 logits contain NaN/Inf"
    );

    Ok(())
}
