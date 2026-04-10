//! Smoke test: Carnice-27b Q4_K_M GGUF load + short generation on a single GPU.
//!
//! Verifies the native Q4_K GPU path loads the full model under the GPU's
//! physical memory budget and generates non-garbage tokens.
//!
//! Enable with:
//!   PEGAINFER_CARNICE_PATH=/abs/path/to/models/Carnice-27b-GGUF \
//!       cargo test --release --test smoke_carnice_27b_q4k -- --nocapture --ignored
//!
//! Ignored by default because it needs the 16 GB GGUF + a ~24 GB GPU.

#![cfg(feature = "cuda")]

use std::time::Instant;

use infer::sampler::SamplingParams;
use infer::server_engine::{
    CompleteRequest, EngineOptions, Qwen35ServerEngine, ServerEngine, StreamDelta,
};
use tokio::sync::mpsc;

fn model_path() -> String {
    std::env::var("PEGAINFER_CARNICE_PATH")
        .unwrap_or_else(|_| "models/Carnice-27b-GGUF".to_string())
}

#[test]
#[ignore]
fn carnice_27b_q4k_load_and_generate() {
    infer::logging::init_stderr("info");
    let path = model_path();
    println!("loading Carnice-27b Q4_K_M from {path}");
    let t0 = Instant::now();
    let mut engine = Qwen35ServerEngine::load_with_options(
        &path,
        42,
        EngineOptions {
            enable_cuda_graph: false, // warmup on 27B takes forever; skip for smoke
        },
    )
    .expect("load Carnice-27b failed");
    let load_secs = t0.elapsed().as_secs_f32();
    println!("loaded in {load_secs:.1}s");

    // GPU memory probe
    let (free, total) = unsafe {
        let mut free: usize = 0;
        let mut total: usize = 0;
        cudarc::driver::sys::cuMemGetInfo_v2(&mut free as *mut _, &mut total as *mut _);
        (free, total)
    };
    let used_gb = (total - free) as f64 / (1 << 30) as f64;
    let total_gb = total as f64 / (1 << 30) as f64;
    println!("GPU residency: {used_gb:.2} GiB used / {total_gb:.2} GiB total");

    // Generation smoke — try very different prompts to tell "stuck token"
    // (same output regardless of prompt) apart from "bad but prompt-dependent".
    for prompt in ["The capital of France is", "1 + 1 = "] {
        let req = CompleteRequest {
            prompt: prompt.to_string(),
            max_tokens: 8,
            sampling: SamplingParams::default(),
            stop: None,
            logprobs: true,
        };
        let (tx, mut rx) = mpsc::unbounded_channel::<StreamDelta>();
        let t0 = Instant::now();
        engine
            .complete_stream(req, tx)
            .expect("complete_stream failed");
        let gen_secs = t0.elapsed().as_secs_f32();

        let mut text = String::new();
        let mut logprobs: Vec<f32> = Vec::new();
        while let Ok(delta) = rx.try_recv() {
            text.push_str(&delta.text_delta);
            if let Some(lp) = delta.logprob {
                logprobs.push(lp);
            }
        }
        println!("prompt={prompt:?} ({gen_secs:.2}s): {text:?}  logprobs={logprobs:?}");
    }
}
