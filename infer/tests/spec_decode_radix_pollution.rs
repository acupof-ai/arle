//! M_d Q1 — spec-decode RadixCache pollution repro test.
//!
//! Hypothesis (from `docs/plans/M_d-tier-kv-spec-decode-coordination.md` §1.2):
//! when a spec-decode request rejects draft tokens midway, the K/V for the
//! rejected draft positions has already been written to the paged KV pool by
//! the prefill kernel. If those pages get published to the RadixCache as part
//! of the request's "committed prefix" without distinguishing
//! `spec-tentative` vs `committed`, a subsequent request sharing the prefix
//! would do a prefix-cache hit on a block whose last K/V slot holds a
//! draft-rejected value instead of the target's real K/V, diverging from a
//! clean-baseline run.
//!
//! This file owns the test contract for that hypothesis. The first scenario
//! (`same_scheduler_back_to_back`) is the most direct repro: drive a single
//! scheduler with two requests that share a prefix, request A self-spec-decode,
//! request B vanilla-decode, and assert request B's tokens match a vanilla
//! baseline run on a fresh scheduler.
//!
//! Skip behaviour matches the rest of the spec-decode test suite: if
//! `INFER_TEST_MODEL_PATH` (or the default `models/Qwen3-4B`) is missing, the
//! test prints a skip message and returns. CUDA-feature gated.
//!
//! NOTE: this is the M_d Q1 *gating* test. The plan deliberately scoped Q1
//! to confirm-or-rule-out the pollution before investing in scratch-page
//! infrastructure (M_d Q2). If this test passes today against the existing
//! prefix-cache publish path, M_d Q2-Q5 are unnecessary; if it fails, the
//! body of the failure pinpoints which slot/block is contaminated.

#![cfg(feature = "cuda")]

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use tokio::sync::mpsc;

use infer::metrics::ServerMetrics;
use infer::model::{KVCacheDtype, KVFormat, ModelRuntimeConfig, Qwen3Model};
use infer::sampler::SamplingParams;
use infer::scheduler::{
    DraftMode, IncomingRequest, RequestPriority, RequestSpecConfig, Scheduler, SchedulerConfig,
};
use infer::server_engine::CompletionStreamDelta;
use infer::tokenizer::Tokenizer;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");

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

fn build_scheduler(
    path: &str,
    spec_enabled: bool,
    draft_k: usize,
    metrics: ServerMetrics,
) -> (Scheduler<Qwen3Model>, infer::scheduler::SchedulerHandle) {
    let model = Qwen3Model::from_safetensors_with_runtime(
        path,
        ModelRuntimeConfig {
            enable_cuda_graph: false,
            ..ModelRuntimeConfig::default()
        },
    )
    .expect("load model");
    let tokenizer = Tokenizer::from_file(path).expect("load tokenizer");
    let mut config = SchedulerConfig::runtime_defaults(4);
    config.spec_enabled = spec_enabled;
    config.spec_draft_k = draft_k;
    config.spec_acceptance_threshold = 0.0; // never auto-disable; we want the test to exercise the path
    if spec_enabled {
        config.spec_draft_model = DraftMode::SelfSpec;
        config.spec_sparse_kv_enabled = true;
    }
    config.prefix_cache_enabled = true;
    config.short_prompt_bypass_tokens = 0;
    Scheduler::with_config(
        model,
        tokenizer,
        "radix-pollution-test",
        42,
        metrics,
        config,
        Some(2048),
        KVCacheDtype::BF16,
        KVFormat::BF16,
    )
    .expect("create scheduler")
}

/// Run a single prompt on a fresh scheduler. Used to capture the
/// "clean baseline" output that the contamination test compares against.
fn run_clean(path: &str, prompt: &str, max_tokens: usize) -> String {
    let metrics = ServerMetrics::new("radix-pollution-clean");
    let (scheduler, handle) = build_scheduler(path, /*spec_enabled=*/ false, 1, metrics);
    let scheduler_thread = std::thread::spawn(move || scheduler.run());
    let (req, mut rx) = make_request(prompt, max_tokens);
    handle.submit(req).expect("submit clean");
    let output = collect_output(&mut rx);
    drop(handle);
    scheduler_thread.join().expect("scheduler join");
    output
}

/// Run prompt A with self-spec, then prompt B with vanilla on the same
/// scheduler. Returns B's output. The test compares this against
/// `run_clean(path, prompt_b, max_tokens_b)` to detect any RadixCache
/// pollution leaking from A's spec-rejected drafts into B's prefix
/// lookup.
fn run_pair_spec_then_vanilla(
    path: &str,
    prompt_a: &str,
    max_tokens_a: usize,
    prompt_b: &str,
    max_tokens_b: usize,
) -> String {
    let metrics = ServerMetrics::new("radix-pollution-pair");
    let (scheduler, handle) = build_scheduler(path, /*spec_enabled=*/ true, 4, metrics);
    let scheduler_thread = std::thread::spawn(move || scheduler.run());

    // Request A — self-spec, allowed to reject; we don't assert on its output.
    let (req_a, mut rx_a) = make_request(prompt_a, max_tokens_a);
    handle.submit(req_a).expect("submit A");
    let _ = collect_output(&mut rx_a);

    // Request B — vanilla, sharing some prompt prefix with A so the
    // RadixCache lookup hits whatever A's prefill (and any spec-tentative
    // pages that may have been mistakenly published) left behind.
    let (mut req_b, mut rx_b) = make_request(prompt_b, max_tokens_b);
    req_b.speculative = Some(RequestSpecConfig {
        enabled: Some(false),
        ..RequestSpecConfig::default()
    });
    handle.submit(req_b).expect("submit B");
    let output_b = collect_output(&mut rx_b);

    drop(handle);
    scheduler_thread.join().expect("scheduler join");
    output_b
}

#[test]
fn spec_decode_does_not_pollute_radix_for_subsequent_request_with_shared_prefix() {
    let _guard = gpu_test_lock();
    infer::logging::init_stderr("info");
    let path = model_path();
    if !Path::new(&path).exists() {
        eprintln!("Skipping test: model not found at {path}");
        return;
    }

    // A and B share more than one 16-token cache block. The divergent suffix
    // forces B to attach to blocks published by A's prefill.
    let shared_prefix = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega repeated stable scheduler prefix ";
    let prompt_a = format!("{shared_prefix}branch A asks for a lazy draft continuation.");
    let prompt_b = format!("{shared_prefix}branch B asks for a slow baseline continuation.");
    let max_tokens = 16;

    let baseline_b = run_clean(&path, &prompt_b, max_tokens);
    let after_spec_b =
        run_pair_spec_then_vanilla(&path, &prompt_a, max_tokens, &prompt_b, max_tokens);

    assert_eq!(
        baseline_b, after_spec_b,
        "RadixCache pollution detected: prompt B output diverged after a \
         self-spec-decode request A on the same scheduler.\n  \
         baseline_b = {baseline_b:?}\n  after_spec_b = {after_spec_b:?}\n\n\
         If this assertion ever trips, M_d Q2 (scratch-page commit barrier) \
         is required — see docs/plans/M_d-tier-kv-spec-decode-coordination.md \
         §1.2 for the design and follow-up tasks Q3-Q5."
    );
}

/// Self-back-to-back: same prompt run twice on the same scheduler with
/// spec on. Catches the simpler regression where a single spec-decode
/// pass leaks state into the next iteration of itself (e.g. through
/// the slot's last_committed_token bookkeeping).
#[test]
fn self_spec_decode_back_to_back_same_prompt_is_idempotent() {
    let _guard = gpu_test_lock();
    infer::logging::init_stderr("info");
    let path = model_path();
    if !Path::new(&path).exists() {
        eprintln!("Skipping test: model not found at {path}");
        return;
    }

    let prompt = "Explain attention in one sentence.";
    let max_tokens = 16;

    let metrics = ServerMetrics::new("radix-pollution-idempotent");
    let (scheduler, handle) = build_scheduler(&path, /*spec_enabled=*/ true, 4, metrics);
    let scheduler_thread = std::thread::spawn(move || scheduler.run());

    let (req1, mut rx1) = make_request(prompt, max_tokens);
    handle.submit(req1).expect("submit 1");
    let out1 = collect_output(&mut rx1);

    let (req2, mut rx2) = make_request(prompt, max_tokens);
    handle.submit(req2).expect("submit 2");
    let out2 = collect_output(&mut rx2);

    drop(handle);
    scheduler_thread.join().expect("scheduler join");

    assert_eq!(
        out1, out2,
        "self-spec-decode is not idempotent across two back-to-back same-prompt \
         requests: out1 = {out1:?}, out2 = {out2:?}"
    );
}
