//! Adapter wiring test: drive `InferenceEngine` through a mock backend.
//!
//! Proves the `tokenize -> submit -> collect -> detokenize` plumbing in
//! [`infer_api::ServeInferenceEngine`] without loading a real model. The mock
//! backend is `infer_server::EchoExecutor` (echoes `last_token + 1`) over the
//! real backend-neutral host paged KV pool. No MLX, no CUDA.
//!
//! Follow-up (not run here): a real Metal e2e against
//! `LoadedInferenceEngine::Metal` would load the 19 GB Qwen3.6 model and is
//! left as a local-heavy follow-up.

use std::path::Path;

use anyhow::Result;
use infer_api::{
    CompletionRequest, FinishReason, InferenceEngine, SamplingParams, ServeInferenceEngine,
};
use infer_core::SchedulerConfig;
use infer_seam::HostPagedKvPool;
use infer_server::{EchoExecutor, OpenAiTokenizer, ServeHandle};
use tokenizers::{
    Tokenizer as HfTokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
};

/// Build a tiny WordLevel tokenizer in `dir` so `OpenAiTokenizer` can load it.
///
/// Vocab is chosen so the EchoExecutor's `+1` token rule maps generated ids
/// back onto real words: prompt "hello" (id 5) -> decode emits 6, 7, 8, ...
/// which decode to "w1", "w2", "w3". This makes the round-trip assertion exact.
fn write_tiny_tokenizer(dir: &Path) -> Result<()> {
    let vocab = [
        ("<unk>".to_string(), 0u32),
        ("a".to_string(), 1u32),
        ("b".to_string(), 2u32),
        ("c".to_string(), 3u32),
        ("d".to_string(), 4u32),
        ("hello".to_string(), 5u32),
        ("w1".to_string(), 6u32),
        ("w2".to_string(), 7u32),
        ("w3".to_string(), 8u32),
        ("w4".to_string(), 9u32),
        ("w5".to_string(), 10u32),
    ]
    .into_iter()
    .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("<unk>".to_string())
        .build()
        .map_err(|err| anyhow::anyhow!("wordlevel build: {err}"))?;
    let mut hf = HfTokenizer::new(model);
    hf.with_pre_tokenizer(Some(Whitespace));
    hf.save(dir.join("tokenizer.json"), false)
        .map_err(|err| anyhow::anyhow!("save tokenizer: {err}"))?;
    Ok(())
}

fn build_engine() -> Result<ServeInferenceEngine<EchoExecutor, HostPagedKvPool>> {
    let dir = tempfile::tempdir()?;
    write_tiny_tokenizer(dir.path())?;
    let tokenizer = OpenAiTokenizer::from_model_dir(dir.path())?;

    let config = SchedulerConfig {
        num_slots: 2,
        max_prompt_tokens: 4096,
        max_total_tokens: 8192,
        ..SchedulerConfig::default()
    };
    // Feature-free placeholder executor + real host KV pool. EchoExecutor gives
    // a deterministic `+1` token rule so token plumbing is exactly assertable.
    let kv = HostPagedKvPool::new(2, 256, 16);
    let serve = ServeHandle::spawn(EchoExecutor, kv, config);

    // Keep the tempdir alive for the lifetime of the engine: the tokenizer is
    // already loaded into memory, so leaking the dir handle is fine for a test.
    std::mem::forget(dir);

    Ok(ServeInferenceEngine::new(
        "tiny-mock".to_string(),
        tokenizer,
        serve,
    ))
}

#[test]
fn complete_runs_tokenize_submit_collect_detokenize() -> Result<()> {
    let mut engine = build_engine()?;

    assert_eq!(engine.model_id(), "tiny-mock");

    // "hello" -> [5]. EchoExecutor: prefill commits last_prompt+1 = 6 ("w1"),
    // then decode emits 7 ("w2"), 8 ("w3") for max_tokens=3.
    let output = engine.complete(CompletionRequest {
        prompt: "hello".to_string(),
        max_tokens: 3,
        sampling: SamplingParams::default(),
        stop: None,
        logprobs: false,
        session_id: None,
        trace_context: None,
        cancel: None,
    })?;

    // tokenize wired: the prompt the engine saw is [5].
    assert_eq!(output.prompt_token_ids, vec![5]);
    // submit -> collect wired: exactly max_tokens generated, echo `+1` rule.
    assert_eq!(output.response_token_ids, vec![6, 7, 8]);
    // detokenize wired: ids 6/7/8 decode back to their words.
    assert_eq!(output.text, "w1 w2 w3");
    // No stop token (id 0) hit -> length-bound finish.
    assert_eq!(output.finish_reason, FinishReason::Length);
    // usage accounting maps CompletedRequest lengths.
    assert_eq!(output.usage.prompt_tokens, 1);
    assert_eq!(output.usage.completion_tokens, 3);
    assert_eq!(output.usage.total_tokens, 4);
    // logprobs gap: always empty.
    assert!(output.token_logprobs.is_empty());

    Ok(())
}

#[test]
fn tokenize_uses_backend_tokenizer() -> Result<()> {
    let engine = build_engine()?;
    assert_eq!(engine.tokenize("hello")?, vec![5]);
    assert_eq!(engine.tokenize("a b c")?, vec![1, 2, 3]);
    Ok(())
}

#[test]
fn complete_stream_emits_text_then_finish_delta() -> Result<()> {
    let mut engine = build_engine()?;
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

    engine.complete_stream(
        CompletionRequest {
            prompt: "hello".to_string(),
            max_tokens: 2,
            sampling: SamplingParams::default(),
            stop: None,
            logprobs: false,
            session_id: None,
            trace_context: None,
            cancel: None,
        },
        tx,
    )?;

    let mut text = String::new();
    let mut finish = None;
    let mut usage = None;
    while let Ok(delta) = rx.try_recv() {
        assert!(
            delta.error.is_none(),
            "no error expected: {:?}",
            delta.error
        );
        text.push_str(&delta.text_delta);
        if delta.finish_reason.is_some() {
            finish = delta.finish_reason;
            usage = delta.usage;
        }
    }
    // "hello" -> [5]; max_tokens=2 -> 6 ("w1"), 7 ("w2").
    assert_eq!(text, "w1 w2");
    assert_eq!(finish, Some(FinishReason::Length));
    assert_eq!(usage.map(|u| u.completion_tokens), Some(2));
    Ok(())
}

#[test]
fn complete_applies_stop_string_truncation() -> Result<()> {
    let mut engine = build_engine()?;

    // max_tokens=3 -> "w1 w2 w3"; stop on "w3" truncates to "w1 w2 " and forces
    // a Stop finish reason even though the engine itself was length-bound.
    let output = engine.complete(CompletionRequest {
        prompt: "hello".to_string(),
        max_tokens: 3,
        sampling: SamplingParams::default(),
        stop: Some(vec!["w3".to_string()]),
        logprobs: false,
        session_id: None,
        trace_context: None,
        cancel: None,
    })?;

    assert_eq!(output.text, "w1 w2 ");
    assert_eq!(output.finish_reason, FinishReason::Stop);
    Ok(())
}
