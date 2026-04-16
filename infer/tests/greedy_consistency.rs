#![cfg(feature = "cuda")]

//! Greedy consistency test: verifies that greedy decode output is identical
//! whether a request runs solo (batch_size=1) or alongside concurrent requests
//! (batch_size=2+). Regression test for the Triton/FlashInfer divergence bug.

use std::path::Path;
use std::time::Instant;

use log::info;
use tokio::sync::mpsc;

use infer::model::{ModelRuntimeConfig, Qwen3Model};
use infer::sampler::SamplingParams;
use infer::scheduler::{IncomingRequest, RequestPriority, Scheduler};
use infer::server_engine::CompletionStreamDelta;
use infer::tokenizer::Tokenizer;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");

fn get_model_path() -> String {
    std::env::var("INFER_TEST_MODEL_PATH").unwrap_or_else(|_| MODEL_PATH.to_string())
}

fn init_logging() {
    infer::logging::init_stderr("info");
}

/// Collect the full text output from a stream of deltas.
fn collect_output(rx: &mut mpsc::UnboundedReceiver<CompletionStreamDelta>) -> String {
    let mut text = String::new();
    loop {
        match rx.blocking_recv() {
            Some(delta) => {
                text.push_str(&delta.text_delta);
                if delta.finish_reason.is_some() {
                    break;
                }
            }
            None => break,
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
        max_tokens,
        sampling: SamplingParams::default(), // greedy (temperature=0)
        stop: None,
        priority: RequestPriority::default(),
        session_id: None,
        delta_tx: tx,
    };
    (req, rx)
}

/// Run a single request through the scheduler (solo = batch_size=1 during decode).
fn run_solo(prompt: &str, max_tokens: usize, model_path: &str) -> String {
    let model = Qwen3Model::from_safetensors_with_runtime(
        model_path,
        ModelRuntimeConfig {
            enable_cuda_graph: true,
        },
    )
    .expect("Failed to load model");
    let tokenizer = Tokenizer::from_file(model_path).expect("Failed to load tokenizer");

    let (scheduler, handle) =
        Scheduler::with_max_seq_len(model, tokenizer, "test", 4, 42, Some(512))
            .expect("Failed to create scheduler");

    let scheduler_thread = std::thread::spawn(move || scheduler.run());

    let (req, mut rx) = make_request(prompt, max_tokens);
    handle.submit(req).expect("submit failed");
    let output = collect_output(&mut rx);

    drop(handle);
    scheduler_thread.join().expect("scheduler thread panicked");

    output
}

/// Run the target request alongside filler requests (concurrent = batch_size>1 during decode).
fn run_concurrent(
    prompt: &str,
    max_tokens: usize,
    filler_prompts: &[&str],
    model_path: &str,
) -> String {
    let model = Qwen3Model::from_safetensors_with_runtime(
        model_path,
        ModelRuntimeConfig {
            enable_cuda_graph: true,
        },
    )
    .expect("Failed to load model");
    let tokenizer = Tokenizer::from_file(model_path).expect("Failed to load tokenizer");

    let num_slots = 1 + filler_prompts.len();
    let (scheduler, handle) =
        Scheduler::with_max_seq_len(model, tokenizer, "test", num_slots, 42, Some(512))
            .expect("Failed to create scheduler");

    let scheduler_thread = std::thread::spawn(move || scheduler.run());

    // Submit filler requests first so they enter decode before the target.
    let mut filler_rxs = Vec::new();
    for &fp in filler_prompts {
        let (req, rx) = make_request(fp, max_tokens);
        handle.submit(req).expect("submit filler failed");
        filler_rxs.push(rx);
    }

    // Submit target request.
    let (req, mut target_rx) = make_request(prompt, max_tokens);
    handle.submit(req).expect("submit target failed");

    // Drain all outputs.
    let target_output = collect_output(&mut target_rx);
    for rx in &mut filler_rxs {
        collect_output(rx);
    }

    drop(handle);
    scheduler_thread.join().expect("scheduler thread panicked");

    target_output
}

#[test]
fn test_greedy_solo_vs_concurrent() {
    init_logging();
    let model_path = get_model_path();

    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {}", model_path);
        return;
    }

    let prompt = "Tell me a story";
    let max_tokens = 30;

    info!("=== Solo run (B=1 decode) ===");
    let t0 = Instant::now();
    let solo_output = run_solo(prompt, max_tokens, &model_path);
    info!("Solo output ({:.1?}): {:?}", t0.elapsed(), solo_output);

    info!("=== Concurrent run (B=3 decode) ===");
    let t0 = Instant::now();
    let concurrent_output = run_concurrent(
        prompt,
        max_tokens,
        &["My name is", "What is 2 + 2?"],
        &model_path,
    );
    info!(
        "Concurrent output ({:.1?}): {:?}",
        t0.elapsed(),
        concurrent_output
    );

    assert_eq!(
        solo_output, concurrent_output,
        "Greedy output diverged!\n  solo:       {:?}\n  concurrent: {:?}",
        solo_output, concurrent_output
    );
    info!("PASS: greedy output is consistent across batch compositions");
}
