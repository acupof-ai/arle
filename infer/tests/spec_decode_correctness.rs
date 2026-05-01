#![cfg(feature = "cuda")]

use std::path::Path;

use tokio::sync::mpsc;

use infer::metrics::ServerMetrics;
use infer::model::kv_cache::{KVCacheDtype, KVFormat};
use infer::model::{ModelRuntimeConfig, Qwen3Model};
use infer::sampler::SamplingParams;
use infer::scheduler::{DraftMode, IncomingRequest, RequestPriority, Scheduler, SchedulerConfig};
use infer::server_engine::CompletionStreamDelta;
use infer::tokenizer::Tokenizer;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");

fn model_path() -> String {
    std::env::var("INFER_TEST_MODEL_PATH").unwrap_or_else(|_| MODEL_PATH.to_string())
}

fn collect_output(rx: &mut mpsc::UnboundedReceiver<CompletionStreamDelta>) -> String {
    let mut text = String::new();
    while let Some(delta) = rx.blocking_recv() {
        text.push_str(&delta.text_delta);
        if delta.finish_reason.is_some() {
            break;
        }
    }
    text
}

fn make_request(
    prompt: &str,
    max_tokens: usize,
) -> (
    IncomingRequest,
    mpsc::UnboundedReceiver<CompletionStreamDelta>,
) {
    let (tx, rx) = mpsc::unbounded_channel();
    let req = IncomingRequest {
        prompt: prompt.to_string(),
        prompt_tokens: None,
        max_tokens,
        sampling: SamplingParams::default(),
        stop: None,
        speculative: None,
        priority: RequestPriority::default(),
        session_id: None,
        delta_tx: tx,
        trace_context: None,
    };
    (req, rx)
}

fn run_prompt(path: &str, prompt: &str, spec_enabled: bool) -> (String, ServerMetrics) {
    let model = Qwen3Model::from_safetensors_with_runtime(
        path,
        ModelRuntimeConfig {
            enable_cuda_graph: true,
        },
    )
    .expect("load model");
    let tokenizer = Tokenizer::from_file(path).expect("load tokenizer");
    let metrics = ServerMetrics::new("spec-test");
    let mut config = SchedulerConfig::runtime_defaults(2);
    config.spec_enabled = spec_enabled;
    config.spec_draft_k = 5;
    if spec_enabled {
        config.spec_draft_model = DraftMode::SelfSpec;
    }

    let (scheduler, handle) = Scheduler::with_config(
        model,
        tokenizer,
        "spec-test",
        42,
        metrics.clone(),
        config,
        Some(512),
        KVCacheDtype::BF16,
        KVFormat::BF16,
    )
    .expect("create scheduler");

    let scheduler_thread = std::thread::spawn(move || scheduler.run());
    let (req, mut rx) = make_request(prompt, 12);
    handle.submit(req).expect("submit");
    let output = collect_output(&mut rx);
    drop(handle);
    scheduler_thread.join().expect("scheduler join");
    (output, metrics)
}

#[test]
fn spec_decode_greedy_is_bit_identical_for_three_prompts() {
    infer::logging::init_stderr("info");
    let path = model_path();
    if !Path::new(&path).exists() {
        eprintln!("Skipping test: model not found at {path}");
        return;
    }

    for prompt in [
        "Explain attention in one sentence.",
        "What is 7 plus 5?",
        "Write a tiny Rust function name.",
    ] {
        let (plain, _) = run_prompt(&path, prompt, false);
        let (spec, metrics) = run_prompt(&path, prompt, true);
        assert_eq!(
            plain, spec,
            "spec decode changed greedy output for {prompt:?}"
        );
        assert!(
            metrics.spec_acceptance_rate() >= 0.3,
            "expected spec acceptance >= 0.3, got {}",
            metrics.spec_acceptance_rate()
        );
    }
}
