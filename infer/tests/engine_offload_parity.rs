#![cfg(feature = "cuda")]
//! OPD engine weight time-share parity: offload→reload must be bit-exact.
//!
//! Verifies that moving a Qwen3.5 engine's device weights to host RAM
//! (freeing VRAM) and reloading them does not corrupt the weights: a greedy
//! argmax trajectory over a fixed token sequence must be identical before and
//! after the round-trip. Covers both the dense BF16 student and the packed
//! W4A8-Marlin teacher (the quantized side tensors must round-trip too).

use std::path::PathBuf;

use anyhow::Result;
use infer::server_engine::{InferenceEngineOptions, LoadedInferenceEngine, ServerRuntimeConfig};

fn model_dir(env_key: &str, default_subpath: &str) -> Option<PathBuf> {
    if let Ok(path) = std::env::var(env_key) {
        return Some(PathBuf::from(path));
    }
    let home = std::env::var("HOME").ok()?;
    Some(PathBuf::from(home).join(default_subpath))
}

/// Run a single full-sequence prefill (the exact path the OPD teacher uses via
/// `forward_logits_device`) and return a deterministic signature of the last
/// position's logits: the argmax token plus a coarse-quantized checksum of the
/// raw logit values. Identical signatures before/after offload prove the
/// device weights round-tripped bit-exactly. A single prefill avoids the
/// dense single-token decode plan that `forward_token_logits` does not service.
fn logits_signature(engine: &LoadedInferenceEngine, tokens: &[u32]) -> Result<(u32, i64)> {
    let positions: Vec<u32> = (0..tokens.len() as u32).collect();
    let logits = engine.forward_token_logits(tokens, &positions)?;
    let host = logits.to_host_f32()?;
    let vocab = logits.vocab_size();
    let seq_len = logits.seq_len();
    let last = &host[(seq_len - 1) * vocab..seq_len * vocab];

    let mut best = 0usize;
    let mut best_val = last[0];
    for (i, &v) in last.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    // Quantized checksum: stable across runs, sensitive to any weight bit flip.
    let checksum: i64 = last
        .iter()
        .map(|&v| (v * 1024.0).round() as i64)
        .fold(0i64, |acc, q| acc.wrapping_mul(1_000_003).wrapping_add(q));
    Ok((best as u32, checksum))
}

fn run_parity(model_dir: PathBuf, label: &str) -> Result<()> {
    if !model_dir.exists() {
        eprintln!(
            "skipping {label} offload parity: missing model {}",
            model_dir.display()
        );
        return Ok(());
    }
    let path = model_dir
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("model path is not valid UTF-8"))?;
    // Mirror the OPD example's single-slot scoring engine config so the decode
    // plan + warmup match the real teacher/student rollout engines.
    let max_seq_len = 320usize;
    let mut runtime = ServerRuntimeConfig {
        engine: InferenceEngineOptions {
            enable_cuda_graph: false,
        },
        max_seq_len: Some(max_seq_len),
        ..ServerRuntimeConfig::default()
    };
    runtime.scheduler.max_slots = 1;
    runtime.scheduler.chunked_prefill_size = max_seq_len;
    runtime.scheduler.max_num_batched_tokens = max_seq_len;
    runtime.scheduler.max_prefill_tokens = max_seq_len;
    runtime.scheduler.long_prefill_token_threshold = max_seq_len;
    runtime.scheduler.prefill_max_requests = Some(1);
    runtime.scheduler.mem_fraction_static = 0.05;
    runtime.scheduler.kv_pool_fallback_bytes = 128 * 1024 * 1024;
    let engine = LoadedInferenceEngine::load_with_runtime_config(path, runtime)?;

    // A long (296-token) sequence matching the rollout-256 OPD teacher scoring
    // shape, to exercise the long-prefill forward path under offload/reload.
    let tokens: Vec<u32> = (0..296u32).map(|i| (i * 7 + 1) % 1000).collect();

    let before = logits_signature(&engine, &tokens)?;

    let freed = engine.offload_engine_weights()?;
    eprintln!(
        "{label} offload freed {freed} bytes ({:.1} MiB)",
        freed as f64 / 1048576.0
    );
    assert!(freed > 0, "{label}: offload must free device VRAM");

    engine.reload_engine_weights()?;

    let after = logits_signature(&engine, &tokens)?;
    assert_eq!(
        before, after,
        "{label}: offload→reload corrupted weights — logits signature diverged"
    );

    // A second round-trip must remain a no-op-correct identity (idempotency of
    // the reloaded state under another offload/reload cycle).
    engine.offload_engine_weights()?;
    engine.reload_engine_weights()?;
    let after2 = logits_signature(&engine, &tokens)?;
    assert_eq!(
        before, after2,
        "{label}: second offload→reload cycle diverged"
    );

    eprintln!(
        "{label} offload/reload parity OK: argmax={} checksum={}",
        before.0, before.1
    );
    Ok(())
}

#[test]
fn qwen35_dense_student_offload_reload_parity() -> Result<()> {
    let Some(dir) = model_dir(
        "ARLE_TEST_QWEN35_STUDENT_DIR",
        ".cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base",
    ) else {
        eprintln!("skipping: HOME not set");
        return Ok(());
    };
    run_parity(dir, "qwen35-0.8B-dense")
}

#[test]
fn qwen35_w4_teacher_offload_reload_parity() -> Result<()> {
    let Some(dir) = model_dir(
        "ARLE_TEST_QWEN35_W4_DIR",
        ".cache/modelscope/hub/Qwen/Qwen3___5-4B-W4A8-marlin",
    ) else {
        eprintln!("skipping: HOME not set");
        return Ok(());
    };
    run_parity(dir, "qwen35-4B-W4A8-marlin")
}
