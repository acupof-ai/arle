#![cfg(feature = "cuda")]
use infer::sampler::SamplingParams;
use infer::server_engine::{
    CompletionRequest, InferenceEngine, InferenceEngineOptions, Qwen35InferenceEngine,
};

#[test]
#[ignore]
fn qwen35_gguf_generate() {
    infer::logging::init_stderr("info");
    let p = std::env::var("INFER_Q35_PATH")
        .unwrap_or_else(|_| "models/Qwen3.5-4B-GGUF-Q6_K".to_string());
    println!("loading {p}");
    let mut engine = Qwen35InferenceEngine::load_with_options(
        &p,
        42,
        InferenceEngineOptions {
            enable_cuda_graph: false,
        },
    )
    .expect("load");
    for prompt in ["The capital of France is", "1 + 1 = "] {
        let req = CompletionRequest {
            prompt: prompt.to_string(),
            max_tokens: 16,
            sampling: SamplingParams::default(),
            stop: None,
            logprobs: false,
        };
        let out = engine.complete(req).unwrap();
        println!(
            "prompt={prompt:?}\n  text={:?}\n  finish={:?}",
            out.text, out.finish_reason
        );
    }
}
