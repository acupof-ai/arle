//! Smoke test Qwen3-4B (NOT Qwen3.5) from GGUF via LoadedServerEngine auto-detect.
//! Qwen3 has no linear attention — if this works, linear-attention is the
//! Qwen3.5 failure site; if it also fails, the bug is in the shared GGUF path.
//!
//!   PEGAINFER_QWEN3_PATH=/abs/path/to/models/Qwen3-4B-GGUF \
//!       cargo test --release --test smoke_qwen3_4b_gguf -- --nocapture --ignored

#![cfg(feature = "cuda")]

use std::time::Instant;

use infer::sampler::SamplingParams;
use infer::server_engine::{
    CompleteRequest, EngineOptions, LoadedServerEngine, ServerEngine, StreamDelta,
};
use tokio::sync::mpsc;

fn path() -> String {
    std::env::var("PEGAINFER_QWEN3_PATH").unwrap_or_else(|_| "models/Qwen3-4B-GGUF".to_string())
}

#[test]
#[ignore]
fn qwen3_4b_gguf_generate() {
    infer::logging::init_stderr("info");
    let p = path();
    println!("loading {p}");
    let t0 = Instant::now();
    let mut engine = LoadedServerEngine::load_with_options(
        &p,
        42,
        EngineOptions {
            enable_cuda_graph: false,
        },
    )
    .expect("load failed");
    println!(
        "loaded in {:.1}s, model_type={:?}",
        t0.elapsed().as_secs_f32(),
        engine.model_type()
    );

    for prompt in ["The capital of France is", "1 + 1 = "] {
        let req = infer::server_engine::CompleteRequest {
            prompt: prompt.to_string(),
            max_tokens: 8,
            sampling: SamplingParams::default(),
            stop: None,
            logprobs: false,
        };
        // Use the sync complete() so we can see the token_ids directly.
        let out = match &mut engine {
            LoadedServerEngine::Qwen3(e) => e.complete(req).unwrap(),
            LoadedServerEngine::Qwen35(e) => e.complete(req).unwrap(),
            LoadedServerEngine::GLM4(e) => e.complete(req).unwrap(),
        };
        println!("prompt={prompt:?}");
        println!("  text={:?}", out.text);
        println!("  finish={:?}", out.finish_reason);
    }
}
