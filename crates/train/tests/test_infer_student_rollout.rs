#![cfg(all(feature = "cuda", not(feature = "no-cuda")))]

//! OPD Phase P1 gating-unknown validation: zero-LoRA (step-0 == base) greedy
//! rollout through the infer engine. Measures per-token latency at growing
//! context and peak VRAM, then states the license/kill verdict vs the
//! train-crate anchor (208 s / ~130 tokens = 1.6-2.88 s/tok, O(n^2)).
//!
//! Run: `cargo test --release -p train --features cuda \
//!   --test test_infer_student_rollout -- --ignored --nocapture`

use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex},
    time::Instant,
};

use autograd::{Backend, backend_cuda::CudaBackend};
use infer::server_engine::{InferenceEngineOptions, LoadedInferenceEngine, ServerRuntimeConfig};
use train::{infer_student::InferStudent, qwen35_loader::load_qwen35_from_hf_dir};

const DEFAULT_QWEN35_08B_DIR: &str = "/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base";
const ROLLOUT_TOKENS: usize = 128;
const MAX_SEQ_LEN: usize = 256;

type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

fn resolve_qwen35_08b_dir() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("ARLE_PARITY_QWEN35_08B_DIR") {
        let path = PathBuf::from(explicit);
        if path.is_dir() {
            return Some(path);
        }
    }
    let path = PathBuf::from(DEFAULT_QWEN35_08B_DIR);
    if path.is_dir() && path.join("config.json").is_file() {
        return Some(path);
    }
    None
}

fn vram_used_mib() -> Option<u64> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines().next()?.trim().parse::<u64>().ok()
}

#[test]
#[ignore = "GPU + Qwen3.5-0.8B weights; run explicitly with --ignored"]
fn infer_student_zero_lora_rollout_latency() -> TestResult {
    let Some(model_dir) = resolve_qwen35_08b_dir() else {
        eprintln!(
            "infer_student_zero_lora_rollout_latency: skipping; \
             set ARLE_PARITY_QWEN35_08B_DIR or populate {DEFAULT_QWEN35_08B_DIR}"
        );
        return Ok(());
    };

    let vram_before = vram_used_mib();

    // Train-side backend + model load (mirrors InferTeacher construction). We
    // load the train Qwen35Model only to read its vocab_size; this is also the
    // store the future KL path uses, so we keep it in the footprint.
    let backend: Arc<dyn Backend> = Arc::new(CudaBackend::new(0)?);
    let mut store = autograd::TensorStore::with_backend(backend.clone());
    let train_model = load_qwen35_from_hf_dir(&model_dir, &mut store)?;
    let vocab_size = train_model.config().vocab_size;

    let infer_engine = load_infer_engine(&model_dir)?;
    let student = InferStudent::new(Arc::new(Mutex::new(infer_engine)), backend, vocab_size);

    let vram_after_load = vram_used_mib();

    // Fixed ~16-token prompt (arbitrary valid token ids in range).
    let prompt: Vec<u32> = vec![
        9419, 374, 264, 1273, 9934, 369, 279, 4128, 1614, 13, 5651, 752, 911, 432, 25, 220,
    ];
    assert!(prompt.iter().all(|&t| (t as usize) < vocab_size));

    let mut sequence = prompt.clone();
    let mut per_token_ms: Vec<f64> = Vec::with_capacity(ROLLOUT_TOKENS);

    let rollout_start = Instant::now();
    for _ in 0..ROLLOUT_TOKENS {
        let positions: Vec<u32> = (0..sequence.len() as u32).collect();
        let step_start = Instant::now();
        let next = student.decode_next_token(&sequence, &positions)?;
        let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
        per_token_ms.push(step_ms);
        sequence.push(next);
    }
    let total_s = rollout_start.elapsed().as_secs_f64();

    let vram_peak = vram_used_mib();

    let at = |i: usize| {
        per_token_ms
            .get(i.saturating_sub(1))
            .copied()
            .unwrap_or(f64::NAN)
    };
    eprintln!("=== InferStudent zero-LoRA rollout (P1) ===");
    eprintln!(
        "prompt_len={} rollout_tokens={ROLLOUT_TOKENS}",
        prompt.len()
    );
    eprintln!("per-token latency (ms):");
    eprintln!("  t=1   : {:.2}", at(1));
    eprintln!("  t=32  : {:.2}", at(32));
    eprintln!("  t=64  : {:.2}", at(64));
    eprintln!("  t=128 : {:.2}", at(128));
    let mean: f64 = per_token_ms.iter().sum::<f64>() / per_token_ms.len() as f64;
    eprintln!("  mean  : {mean:.2}");
    eprintln!("total rollout wall-clock: {total_s:.3} s for {ROLLOUT_TOKENS} tokens");
    eprintln!(
        "VRAM (MiB): before={:?} after_load={:?} peak={:?}",
        vram_before, vram_after_load, vram_peak
    );
    eprintln!(
        "generated tail tokens: {:?}",
        &sequence[sequence.len().saturating_sub(8)..]
    );

    // Sanity: the engine must produce in-vocab tokens and not stall.
    assert!(sequence.iter().all(|&t| (t as usize) < vocab_size));
    assert_eq!(sequence.len(), prompt.len() + ROLLOUT_TOKENS);

    Ok(())
}

fn load_infer_engine(model_dir: &Path) -> anyhow::Result<LoadedInferenceEngine> {
    let mut runtime = ServerRuntimeConfig {
        engine: InferenceEngineOptions {
            enable_cuda_graph: false,
        },
        max_seq_len: Some(MAX_SEQ_LEN),
        ..ServerRuntimeConfig::default()
    };
    runtime.scheduler.max_slots = 1;
    runtime.scheduler.chunked_prefill_size = MAX_SEQ_LEN;
    runtime.scheduler.max_num_batched_tokens = MAX_SEQ_LEN;
    runtime.scheduler.max_prefill_tokens = MAX_SEQ_LEN;
    runtime.scheduler.long_prefill_token_threshold = MAX_SEQ_LEN;
    runtime.scheduler.prefill_max_requests = Some(1);
    runtime.scheduler.mem_fraction_static = 0.05;
    runtime.scheduler.kv_pool_fallback_bytes = 128 * 1024 * 1024;
    LoadedInferenceEngine::load_with_runtime_config(
        model_dir
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("model path is not valid UTF-8"))?,
        runtime,
    )
}
