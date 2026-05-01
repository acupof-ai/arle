#![cfg(feature = "cuda")]

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use tokio::sync::mpsc;

use infer::metrics::ServerMetrics;
use infer::model::{KVCacheDtype, KVFormat, ModelRuntimeConfig, Qwen3Model};
use infer::sampler::SamplingParams;
use infer::scheduler::{DraftMode, IncomingRequest, RequestPriority, Scheduler, SchedulerConfig};
use infer::server_engine::CompletionStreamDelta;
use infer::speculative::DraftEngine;
use infer::tokenizer::Tokenizer;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");
const DRAFT_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-0.6B");

fn model_path() -> String {
    std::env::var("INFER_TEST_MODEL_PATH").unwrap_or_else(|_| MODEL_PATH.to_string())
}

fn gpu_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
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

#[derive(Clone, Copy, Debug, Default)]
struct SpecMetricSnapshot {
    draft_tokens: u64,
    verified_tokens: u64,
    accepted_tokens: u64,
}

impl SpecMetricSnapshot {
    fn capture(metrics: &ServerMetrics) -> Self {
        Self {
            draft_tokens: metrics.spec_draft_tokens_total(),
            verified_tokens: metrics.spec_verified_tokens_total(),
            accepted_tokens: metrics.spec_accepted_tokens_total(),
        }
    }
}

fn run_prompt(
    path: &str,
    prompt: &str,
    spec_enabled: bool,
    draft_mode: DraftMode,
    draft_k: usize,
) -> (String, ServerMetrics) {
    let (output, metrics, _) =
        run_prompt_with_sparse(path, prompt, spec_enabled, draft_mode, draft_k, false);
    (output, metrics)
}

fn run_prompt_with_sparse(
    path: &str,
    prompt: &str,
    spec_enabled: bool,
    draft_mode: DraftMode,
    draft_k: usize,
    sparse_kv: bool,
) -> (String, ServerMetrics, SpecMetricSnapshot) {
    let model = Qwen3Model::from_safetensors_with_runtime(
        path,
        ModelRuntimeConfig {
            enable_cuda_graph: false,
            ..ModelRuntimeConfig::default()
        },
    )
    .expect("load model");
    let tokenizer = Tokenizer::from_file(path).expect("load tokenizer");
    let metrics = ServerMetrics::new("spec-test");
    let mut config = SchedulerConfig::runtime_defaults(2);
    config.spec_enabled = spec_enabled;
    config.spec_draft_k = draft_k;
    config.spec_acceptance_threshold = 0.3;
    if spec_enabled {
        config.spec_draft_model = draft_mode;
    }
    config.spec_sparse_kv_enabled = sparse_kv;
    if sparse_kv {
        config.short_prompt_bypass_tokens = 0;
        config.spec_sparse_recent_tokens = 64;
        config.spec_sparse_top_k_pages = 1;
    }

    let (scheduler, handle) = Scheduler::with_config(
        model,
        tokenizer,
        "spec-test",
        42,
        metrics.clone(),
        config,
        Some(2048),
        KVCacheDtype::BF16,
        KVFormat::BF16,
    )
    .expect("create scheduler");

    let scheduler_thread = std::thread::spawn(move || scheduler.run());
    if sparse_kv {
        let (warmup_req, mut warmup_rx) = make_request(prompt, 1);
        handle.submit(warmup_req).expect("submit sparse warmup");
        let _ = collect_output(&mut warmup_rx);
    }
    let metric_baseline = SpecMetricSnapshot::capture(&metrics);
    let (req, mut rx) = make_request(prompt, 12);
    handle.submit(req).expect("submit");
    let output = collect_output(&mut rx);
    drop(handle);
    scheduler_thread.join().expect("scheduler join");
    (output, metrics, metric_baseline)
}

fn first_token_divergence(tokenizer: &Tokenizer, plain: &str, spec: &str) -> String {
    let plain_ids = tokenizer.encode(plain).unwrap_or_default();
    let spec_ids = tokenizer.encode(spec).unwrap_or_default();
    let max_len = plain_ids.len().max(spec_ids.len());
    let idx = (0..max_len)
        .find(|&idx| plain_ids.get(idx) != spec_ids.get(idx))
        .unwrap_or(max_len);
    format!("first_token_divergence={idx}, plain_ids={plain_ids:?}, spec_ids={spec_ids:?}")
}

#[test]
fn spec_decode_greedy_is_bit_identical_for_three_prompts() {
    let _guard = gpu_test_lock();
    infer::logging::init_stderr("info");
    let path = model_path();
    if !Path::new(&path).exists() {
        eprintln!("Skipping test: model not found at {path}");
        return;
    }
    let tokenizer = Tokenizer::from_file(&path).expect("load tokenizer for diagnostics");

    for prompt in [
        "Explain attention in one sentence.",
        "What is 7 plus 5?",
        "Write a tiny Rust function name.",
    ] {
        let (plain, _) = run_prompt(&path, prompt, false, DraftMode::None, 1);
        let (spec, metrics) = run_prompt(&path, prompt, true, DraftMode::SelfSpec, 1);
        assert_eq!(
            plain,
            spec,
            "spec decode changed greedy output for {prompt:?}: {}",
            first_token_divergence(&tokenizer, &plain, &spec)
        );
        assert!(
            metrics.spec_acceptance_rate() >= 0.3,
            "expected spec acceptance >= 0.3, got {}",
            metrics.spec_acceptance_rate()
        );
    }
}

#[test]
fn external_spec_decode_greedy_is_bit_identical_for_three_prompts() {
    let _guard = gpu_test_lock();
    infer::logging::init_stderr("info");
    let path = model_path();
    let draft_path = std::env::var("INFER_TEST_DRAFT_MODEL_PATH")
        .unwrap_or_else(|_| DRAFT_MODEL_PATH.to_string());
    if !Path::new(&path).exists() {
        eprintln!("Skipping test: model not found at {path}");
        return;
    }
    if !Path::new(&draft_path).exists() {
        eprintln!("Skipping test: draft model not found at {draft_path}");
        return;
    }
    let tokenizer = Tokenizer::from_file(&path).expect("load tokenizer for diagnostics");

    for prompt in [
        "Explain attention in one sentence.",
        "What is 7 plus 5?",
        "Write a tiny Rust function name.",
    ] {
        let (plain, _) = run_prompt(&path, prompt, false, DraftMode::None, 1);
        let (spec, metrics) = run_prompt(
            &path,
            prompt,
            true,
            DraftMode::External(draft_path.clone().into()),
            5,
        );
        assert_eq!(
            plain,
            spec,
            "external spec decode changed greedy output for {prompt:?}: {}",
            first_token_divergence(&tokenizer, &plain, &spec)
        );
        assert!(
            metrics.spec_verified_tokens_total() > 0,
            "expected external verifier to process draft tokens"
        );
    }
}

#[test]
fn sparse_self_spec_decode_greedy_is_bit_identical_and_updates_verifier_metrics() {
    let _guard = gpu_test_lock();
    infer::logging::init_stderr("info");
    let path = model_path();
    if !Path::new(&path).exists() {
        eprintln!("Skipping test: model not found at {path}");
        return;
    }
    let tokenizer = Tokenizer::from_file(&path).expect("load tokenizer for diagnostics");
    let prompt = [
        "Sparse KV verifier correctness prompt.",
        "Repeat enough context so radix sealed blocks exist before decode.",
        "The verifier must use full KV while the draft path may use a sparse view.",
    ]
    .repeat(48)
    .join(" ");

    let (plain, _) = run_prompt(&path, &prompt, false, DraftMode::None, 1);
    let (spec, metrics, metric_baseline) =
        run_prompt_with_sparse(&path, &prompt, true, DraftMode::SelfSpec, 5, true);
    assert_eq!(
        plain,
        spec,
        "sparse self-spec changed greedy output: {}",
        first_token_divergence(&tokenizer, &plain, &spec)
    );
    assert!(
        metrics.spec_draft_tokens_total() > metric_baseline.draft_tokens,
        "expected sparse self-spec to draft tokens"
    );
    assert!(
        metrics.spec_verified_tokens_total() > metric_baseline.verified_tokens,
        "expected full-KV verifier to check sparse draft tokens"
    );
    assert!(
        metrics.spec_accepted_tokens_total() >= metric_baseline.accepted_tokens,
        "expected sparse self-spec accepted-token counter to remain monotonic"
    );
}

#[test]
fn external_draft_state_persists_across_steps() {
    let _guard = gpu_test_lock();
    infer::logging::init_stderr("info");
    let draft_path = std::env::var("INFER_TEST_DRAFT_MODEL_PATH")
        .unwrap_or_else(|_| DRAFT_MODEL_PATH.to_string());
    if !Path::new(&draft_path).exists() {
        eprintln!("Skipping test: draft model not found at {draft_path}");
        return;
    }

    let tokenizer = Tokenizer::from_file(&draft_path).expect("load draft tokenizer");
    let prefix = tokenizer
        .encode("Persistent draft KV test prompt.")
        .expect("encode draft prefix");
    assert!(!prefix.is_empty(), "draft prefix must not be empty");

    let engine = DraftEngine::load_qwen3(&draft_path).expect("load draft engine");
    let request_id = 20260501;
    let draft_max_seq_len = prefix.len() + 32;
    engine
        .create_request_state(request_id, &prefix, draft_max_seq_len)
        .expect("create persistent draft state");
    assert_eq!(engine.request_position(request_id), Some(prefix.len()));

    let first = engine
        .draft_for_request(request_id, 2)
        .expect("draft first step");
    let after_first = engine
        .request_position(request_id)
        .expect("position after first draft");
    assert_eq!(after_first, prefix.len() + first.tokens.len());

    let second = engine
        .draft_for_request(request_id, 3)
        .expect("draft second step");
    let after_second = engine
        .request_position(request_id)
        .expect("position after second draft");
    assert_eq!(after_second, after_first + second.tokens.len());

    engine.release_request_state(request_id);
    assert!(!engine.has_request_state(request_id));
}
