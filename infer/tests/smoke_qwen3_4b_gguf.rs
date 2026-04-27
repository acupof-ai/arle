//! Smoke test Qwen3-4B (NOT Qwen3.5) from GGUF via LoadedInferenceEngine auto-detect.
//! Qwen3 has no linear attention — if this works, linear-attention is the
//! Qwen3.5 failure site; if it also fails, the bug is in the shared GGUF path.
//!
//!   INFER_QWEN3_PATH=/abs/path/to/models/Qwen3-4B-GGUF \
//!       cargo test --release --test smoke_qwen3_4b_gguf -- --nocapture --ignored

#![cfg(feature = "cuda")]

use std::time::Instant;

use infer::sampler::SamplingParams;
use infer::server_engine::{InferenceEngine, InferenceEngineOptions, LoadedInferenceEngine};

fn path() -> String {
    std::env::var("INFER_QWEN3_PATH").unwrap_or_else(|_| "models/Qwen3-4B-GGUF".to_string())
}

#[test]
#[ignore = "requires Qwen3-4B GGUF weights + CUDA GPU"]
fn qwen3_4b_gguf_generate() {
    infer::logging::init_stderr("info");
    let p = path();
    println!("loading {p}");
    let t0 = Instant::now();
    let mut engine = LoadedInferenceEngine::load_with_options(
        &p,
        42,
        InferenceEngineOptions {
            enable_cuda_graph: false,
        },
    )
    .expect("load failed");
    println!(
        "loaded in {:.1}s, model_type={:?}",
        t0.elapsed().as_secs_f32(),
        engine.model_type()
    );

    // Build a long prompt (~1.5k tokens) to stress-test long-context prefill.
    let long_body = "The quick brown fox jumps over the lazy dog. ".repeat(256);
    let long_prompt =
        format!("{long_body}\nBased on the passage above, the animal that jumps is the");
    let prompts: Vec<String> = vec![
        "The capital of France is".to_string(),
        "1 + 1 = ".to_string(),
        long_prompt,
    ];
    for prompt in &prompts {
        let req = infer::server_engine::CompletionRequest {
            prompt: prompt.clone(),
            max_tokens: 16,
            sampling: SamplingParams::default(),
            stop: None,
            logprobs: false,
            session_id: None,
            trace_context: None,
        };
        // Use the sync complete() so we can see the token_ids directly.
        let out = match &mut engine {
            LoadedInferenceEngine::Qwen3(e) => e.complete(req).unwrap(),
            LoadedInferenceEngine::Qwen35(e) => e.complete(req).unwrap(),
            LoadedInferenceEngine::Qwen35Moe(e) => e.complete(req).unwrap(),
        };
        let shown: String = prompt.chars().take(60).collect();
        println!("prompt_len={} prompt_head={shown:?}", prompt.len());
        println!("  text={:?}", out.text);
        println!("  finish={:?}", out.finish_reason);
    }
}
