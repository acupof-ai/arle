#![cfg(feature = "cuda")]
use infer::sampler::SamplingParams;
use infer::server_engine::{CompleteRequest, EngineOptions, Qwen35ServerEngine, ServerEngine};

#[test]
#[ignore]
fn qwen35_gguf_generate() {
    infer::logging::init_stderr("info");
    let p = std::env::var("PEGAINFER_Q35_PATH")
        .unwrap_or_else(|_| "models/Qwen3.5-4B-GGUF-Q6_K".to_string());
    println!("loading {p}");
    let mut engine = Qwen35ServerEngine::load_with_options(
        &p,
        42,
        EngineOptions {
            enable_cuda_graph: false,
        },
    )
    .expect("load");
    for prompt in ["The capital of France is", "1 + 1 = "] {
        let req = CompleteRequest {
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
