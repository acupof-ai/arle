#[cfg(feature = "cuda")]
pub use crate::backend::cuda::bootstrap::{
    InferenceEngineOptions, ModelType, ServerRuntimeConfig, detect_model_type,
};

#[cfg(any(feature = "metal", feature = "cpu"))]
#[path = "server_engine/backend_engine.rs"]
mod backend_engine;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
#[path = "server_engine/loaded.rs"]
mod loaded;
#[path = "server_engine/pool.rs"]
mod pool;
#[path = "server_engine/request_handle_engine.rs"]
mod request_handle_engine;
#[path = "server_engine/stream.rs"]
mod stream;
#[path = "server_engine/types.rs"]
mod types;

#[cfg(any(feature = "metal", feature = "cpu"))]
pub use backend_engine::BackendInferenceEngine;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
pub use loaded::LoadedInferenceEngine;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
pub use pool::LoadedEnginePool;
pub use pool::{
    EngineLease, EnginePool, EnginePoolConfig, EnginePoolModelInfo, EnginePoolModelSpec,
    EnginePoolModelType,
};
pub use request_handle_engine::RequestHandleInferenceEngine;
pub use types::{
    CompletionOutput, CompletionRequest, CompletionStreamDelta, FinishReason, InferenceEngine,
    TokenUsage,
};

#[cfg(test)]
fn truncate_at_first_stop(text: &str, stops: &[String]) -> Option<String> {
    stream::truncate_at_first_stop(text, stops)
}

#[cfg(test)]
fn model_id_from_path(model_path: &str) -> String {
    stream::model_id_from_path(model_path)
}

#[cfg(test)]
fn parse_finish_reason(finish_reason: &str) -> FinishReason {
    stream::parse_finish_reason(finish_reason)
}

#[cfg(test)]
mod tests {
    use super::{FinishReason, model_id_from_path, parse_finish_reason, truncate_at_first_stop};

    #[test]
    fn test_truncate_at_first_stop() {
        let stops: Vec<String> = vec!["\n\n".into(), "END".into()];
        assert_eq!(
            truncate_at_first_stop("4\n\nand more", &stops),
            Some("4".to_string())
        );
        assert_eq!(
            truncate_at_first_stop("helloEND", &stops),
            Some("hello".to_string())
        );
        assert_eq!(truncate_at_first_stop("hello", &stops), None);
        assert_eq!(truncate_at_first_stop("", &stops), None);
        assert_eq!(
            truncate_at_first_stop("a\n\nbEND", &stops),
            Some("a".to_string())
        );
        let stops_nl: Vec<String> = vec!["\n".into()];
        assert_eq!(
            truncate_at_first_stop("hello\nworld", &stops_nl),
            Some("hello".to_string())
        );
        assert_eq!(
            truncate_at_first_stop("ab", &["ab".to_string()]),
            Some(String::new())
        );
    }

    /// Regression test for codex review 0da212f/97c1a95 High: before this
    /// fix, `BackendInferenceEngine<Metal|Cpu>::complete_stream` called
    /// `self.complete(req)?` — a blocking full generation — and only
    /// touched `tx` at the end. Dropping `rx` mid-generation had no
    /// effect: the worker thread blocked until completion. The REPL's
    /// Ctrl-C path at `crates/cli/src/repl.rs:646` (which relies on
    /// `tx.send` failing when `rx` is dropped) was a lie on Metal + CPU.
    ///
    /// This test drops the receiver after zero chunks read, then
    /// asserts the mock backend's chunk counter stops at 1 (not 10) —
    /// i.e. the `on_chunk` callback propagated the `rx-disconnected`
    /// error back through `generate_stream`, short-circuiting the loop.
    #[cfg(any(feature = "metal", feature = "cpu"))]
    #[test]
    fn backend_complete_stream_short_circuits_when_rx_dropped() {
        use super::{BackendInferenceEngine, CompletionRequest, InferenceEngine};
        use crate::backend::{GenerateResult, InferenceBackend, StreamingInferenceBackend};
        use crate::sampler::SamplingParams;
        use std::path::Path;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::sync::mpsc;

        #[derive(Clone)]
        struct CountingMock {
            chunks_attempted: Arc<AtomicUsize>,
        }

        impl InferenceBackend for CountingMock {
            fn load(&mut self, _p: &Path) -> anyhow::Result<()> {
                Ok(())
            }
            fn generate(
                &self,
                _prompt: &str,
                _params: &SamplingParams,
            ) -> anyhow::Result<GenerateResult> {
                unreachable!("test exercises streaming path only")
            }
            fn name(&self) -> &'static str {
                "counting-mock"
            }
        }

        impl StreamingInferenceBackend for CountingMock {
            fn generate_stream<F>(
                &self,
                _prompt: &str,
                _params: &SamplingParams,
                mut on_chunk: F,
            ) -> anyhow::Result<GenerateResult>
            where
                F: FnMut(&str) -> anyhow::Result<()>,
            {
                for _ in 0..10 {
                    self.chunks_attempted.fetch_add(1, Ordering::Relaxed);
                    on_chunk("x")?; // returns Err the instant tx fails
                }
                Ok(GenerateResult {
                    text: "xxxxxxxxxx".into(),
                    prompt_tokens: 1,
                    completion_tokens: 10,
                    finish_reason: "length".into(),
                    ttft_ms: 0.0,
                    prompt_tps: 0.0,
                    generation_tps: 0.0,
                    total_time_ms: 0.0,
                })
            }
        }

        let counter = Arc::new(AtomicUsize::new(0));
        let mut engine = BackendInferenceEngine {
            model_id: "counting-mock".into(),
            backend: CountingMock {
                chunks_attempted: counter.clone(),
            },
        };

        let (tx, rx) = mpsc::unbounded_channel();
        drop(rx); // simulate REPL cancel before the first chunk.

        let res = engine.complete_stream(
            CompletionRequest {
                prompt: "hi".into(),
                max_tokens: 10,
                sampling: SamplingParams::default(),
                stop: None,
                logprobs: false,
                session_id: None,
                trace_context: None,
            },
            tx,
        );

        assert!(res.is_ok(), "consumer-dropped is not an error");
        assert_eq!(
            counter.load(Ordering::Relaxed),
            1,
            "generate_stream must exit after the first failed tx.send, \
             not keep looping through all 10 chunks"
        );
    }

    /// Normal completion: no stop sequences, reader intact → backend
    /// runs to completion, each chunk flows through, and the final
    /// delta carries `finish_reason` + `usage`.
    #[cfg(any(feature = "metal", feature = "cpu"))]
    #[tokio::test]
    async fn backend_complete_stream_emits_all_chunks_and_finish_marker() {
        use super::{BackendInferenceEngine, CompletionRequest, InferenceEngine};
        use crate::backend::{GenerateResult, InferenceBackend, StreamingInferenceBackend};
        use crate::sampler::SamplingParams;
        use std::path::Path;
        use tokio::sync::mpsc;

        struct FullRunMock;
        impl InferenceBackend for FullRunMock {
            fn load(&mut self, _p: &Path) -> anyhow::Result<()> {
                Ok(())
            }
            fn generate(
                &self,
                _prompt: &str,
                _params: &SamplingParams,
            ) -> anyhow::Result<GenerateResult> {
                unreachable!()
            }
            fn name(&self) -> &'static str {
                "full-run-mock"
            }
        }
        impl StreamingInferenceBackend for FullRunMock {
            fn generate_stream<F>(
                &self,
                _prompt: &str,
                _params: &SamplingParams,
                mut on_chunk: F,
            ) -> anyhow::Result<GenerateResult>
            where
                F: FnMut(&str) -> anyhow::Result<()>,
            {
                on_chunk("hel")?;
                on_chunk("lo")?;
                Ok(GenerateResult {
                    text: "hello".into(),
                    prompt_tokens: 1,
                    completion_tokens: 2,
                    finish_reason: "length".into(),
                    ttft_ms: 0.0,
                    prompt_tps: 0.0,
                    generation_tps: 0.0,
                    total_time_ms: 0.0,
                })
            }
        }

        let mut engine = BackendInferenceEngine {
            model_id: "full-run-mock".into(),
            backend: FullRunMock,
        };

        let (tx, mut rx) = mpsc::unbounded_channel();
        let res = engine.complete_stream(
            CompletionRequest {
                prompt: "p".into(),
                max_tokens: 8,
                sampling: SamplingParams::default(),
                stop: None,
                logprobs: false,
                session_id: None,
                trace_context: None,
            },
            tx,
        );
        assert!(res.is_ok());

        let mut text_parts: Vec<String> = Vec::new();
        let mut finish: Option<FinishReason> = None;
        while let Ok(chunk) = rx.try_recv() {
            if chunk.finish_reason.is_some() {
                finish = chunk.finish_reason;
            }
            if !chunk.text_delta.is_empty() {
                text_parts.push(chunk.text_delta);
            }
        }
        assert_eq!(text_parts.concat(), "hello");
        assert_eq!(finish, Some(FinishReason::Length));
    }

    #[cfg(any(feature = "metal", test))]
    #[test]
    fn request_handle_engine_complete_collects_streamed_deltas() {
        use super::{
            CompletionOutput, CompletionRequest, CompletionStreamDelta, InferenceEngine,
            RequestHandleInferenceEngine, TokenUsage,
        };
        use crate::request_handle::{RequestHandle, SubmitError};
        use crate::sampler::SamplingParams;
        use crate::scheduler::IncomingRequest;

        struct MockHandle;

        impl RequestHandle for MockHandle {
            fn submit(&self, req: IncomingRequest) -> Result<(), SubmitError> {
                let _ = req.delta_tx.send(CompletionStreamDelta {
                    text_delta: "hel".into(),
                    finish_reason: None,
                    usage: None,
                    logprob: None,
                    token_ids: Vec::new(),
                });
                let _ = req.delta_tx.send(CompletionStreamDelta {
                    text_delta: "lo".into(),
                    finish_reason: None,
                    usage: None,
                    logprob: None,
                    token_ids: Vec::new(),
                });
                let _ = req.delta_tx.send(CompletionStreamDelta {
                    text_delta: String::new(),
                    finish_reason: Some(FinishReason::Stop),
                    usage: Some(TokenUsage {
                        prompt_tokens: 3,
                        completion_tokens: 2,
                        total_tokens: 5,
                    }),
                    logprob: None,
                    token_ids: Vec::new(),
                });
                Ok(())
            }

            fn model_id(&self) -> &str {
                "mock-handle"
            }
        }

        let mut engine = RequestHandleInferenceEngine {
            model_id: "mock-handle".into(),
            handle: MockHandle,
        };
        let output = engine
            .complete(CompletionRequest {
                prompt: "hi".into(),
                max_tokens: 2,
                sampling: SamplingParams::default(),
                stop: None,
                logprobs: false,
                session_id: None,
                trace_context: None,
            })
            .expect("complete");

        let CompletionOutput {
            text,
            finish_reason,
            usage,
            token_logprobs,
            prompt_token_ids,
            response_token_ids,
        } = output;
        assert_eq!(text, "hello");
        assert_eq!(finish_reason, FinishReason::Stop);
        assert_eq!(
            usage,
            TokenUsage {
                prompt_tokens: 3,
                completion_tokens: 2,
                total_tokens: 5,
            }
        );
        assert!(token_logprobs.is_empty());
        // MockHandle has no tokenizer attached, so tokenize() errors and
        // both ID vectors fall back to empty.
        assert!(prompt_token_ids.is_empty());
        assert!(response_token_ids.is_empty());
    }

    /// Regression for codex review 70e2776 High #1 — stop *inside* a
    /// single chunk. The default `StreamingInferenceBackend` impl sends
    /// the whole completion as one chunk; the old end-of-buffer check
    /// would only fire when the chunk *ended* with the stop, so a stop
    /// mid-chunk leaked the raw marker + trailing bytes. With
    /// `StopChunkProcessor::push_chunk` scanning the unsent suffix,
    /// everything after the stop is withheld.
    #[cfg(any(feature = "metal", feature = "cpu"))]
    #[tokio::test]
    async fn backend_complete_stream_stop_inside_single_chunk() {
        use super::{BackendInferenceEngine, CompletionRequest, InferenceEngine, TokenUsage};
        use crate::backend::{GenerateResult, InferenceBackend, StreamingInferenceBackend};
        use crate::sampler::SamplingParams;
        use std::path::Path;
        use tokio::sync::mpsc;

        struct SingleChunkMock;
        impl InferenceBackend for SingleChunkMock {
            fn load(&mut self, _p: &Path) -> anyhow::Result<()> {
                Ok(())
            }
            fn generate(
                &self,
                _prompt: &str,
                _params: &SamplingParams,
            ) -> anyhow::Result<GenerateResult> {
                unreachable!()
            }
            fn name(&self) -> &'static str {
                "single-chunk-mock"
            }
        }
        impl StreamingInferenceBackend for SingleChunkMock {
            fn generate_stream<F>(
                &self,
                _prompt: &str,
                _params: &SamplingParams,
                mut on_chunk: F,
            ) -> anyhow::Result<GenerateResult>
            where
                F: FnMut(&str) -> anyhow::Result<()>,
            {
                match on_chunk("hello<|im_end|>trailing") {
                    Ok(()) => {}
                    Err(err)
                        if err
                            .downcast_ref::<crate::backend::StreamStopMatched>()
                            .is_some() => {}
                    Err(err) => return Err(err),
                }
                Ok(GenerateResult {
                    text: "hello<|im_end|>trailing".into(),
                    prompt_tokens: 3,
                    completion_tokens: 7,
                    finish_reason: "stop".into(),
                    ttft_ms: 0.0,
                    prompt_tps: 0.0,
                    generation_tps: 0.0,
                    total_time_ms: 0.0,
                })
            }
        }

        let mut engine = BackendInferenceEngine {
            model_id: "single-chunk-mock".into(),
            backend: SingleChunkMock,
        };

        let (tx, mut rx) = mpsc::unbounded_channel();
        engine
            .complete_stream(
                CompletionRequest {
                    prompt: "p".into(),
                    max_tokens: 32,
                    sampling: SamplingParams::default(),
                    stop: Some(vec!["<|im_end|>".into()]),
                    logprobs: false,
                    session_id: None,
                    trace_context: None,
                },
                tx,
            )
            .unwrap();

        let mut text_parts: Vec<String> = Vec::new();
        let mut finish: Option<FinishReason> = None;
        let mut usage: Option<TokenUsage> = None;
        while let Ok(chunk) = rx.try_recv() {
            if chunk.finish_reason.is_some() {
                finish = chunk.finish_reason;
            }
            if chunk.usage.is_some() {
                usage = chunk.usage;
            }
            if !chunk.text_delta.is_empty() {
                text_parts.push(chunk.text_delta);
            }
        }
        let joined = text_parts.concat();
        assert_eq!(joined, "hello", "stop marker + trailing must be withheld");
        assert!(
            !joined.contains("<|im_end|>"),
            "raw stop marker must never reach the consumer"
        );
        assert_eq!(finish, Some(FinishReason::Stop));
        assert_eq!(
            usage,
            Some(TokenUsage {
                prompt_tokens: 3,
                completion_tokens: 7,
                total_tokens: 10,
            })
        );
    }

    /// Regression for codex review 70e2776 High #2 — stop *spanning*
    /// chunk boundaries. Before the fix, the first chunk's bytes were
    /// forwarded immediately and the stop was only detected on the
    /// chunk that completed it — by then the prefix had already been
    /// leaked. `StopChunkProcessor` withholds the last `max_stop_len-1`
    /// bytes of each chunk until the next one arrives.
    #[cfg(any(feature = "metal", feature = "cpu"))]
    #[tokio::test]
    async fn backend_complete_stream_stop_spanning_chunks() {
        use super::{BackendInferenceEngine, CompletionRequest, InferenceEngine, TokenUsage};
        use crate::backend::{GenerateResult, InferenceBackend, StreamingInferenceBackend};
        use crate::sampler::SamplingParams;
        use std::path::Path;
        use tokio::sync::mpsc;

        struct SplitChunkMock;
        impl InferenceBackend for SplitChunkMock {
            fn load(&mut self, _p: &Path) -> anyhow::Result<()> {
                Ok(())
            }
            fn generate(
                &self,
                _prompt: &str,
                _params: &SamplingParams,
            ) -> anyhow::Result<GenerateResult> {
                unreachable!()
            }
            fn name(&self) -> &'static str {
                "split-chunk-mock"
            }
        }
        impl StreamingInferenceBackend for SplitChunkMock {
            fn generate_stream<F>(
                &self,
                _prompt: &str,
                _params: &SamplingParams,
                mut on_chunk: F,
            ) -> anyhow::Result<GenerateResult>
            where
                F: FnMut(&str) -> anyhow::Result<()>,
            {
                on_chunk("hello<|im_")?;
                match on_chunk("end|>trail") {
                    Ok(()) => {}
                    Err(err)
                        if err
                            .downcast_ref::<crate::backend::StreamStopMatched>()
                            .is_some() => {}
                    Err(err) => return Err(err),
                }
                Ok(GenerateResult {
                    text: "hello<|im_end|>trail".into(),
                    prompt_tokens: 2,
                    completion_tokens: 5,
                    finish_reason: "stop".into(),
                    ttft_ms: 0.0,
                    prompt_tps: 0.0,
                    generation_tps: 0.0,
                    total_time_ms: 0.0,
                })
            }
        }

        let mut engine = BackendInferenceEngine {
            model_id: "split-chunk-mock".into(),
            backend: SplitChunkMock,
        };

        let (tx, mut rx) = mpsc::unbounded_channel();
        engine
            .complete_stream(
                CompletionRequest {
                    prompt: "p".into(),
                    max_tokens: 32,
                    sampling: SamplingParams::default(),
                    stop: Some(vec!["<|im_end|>".into()]),
                    logprobs: false,
                    session_id: None,
                    trace_context: None,
                },
                tx,
            )
            .unwrap();

        let mut text_parts: Vec<String> = Vec::new();
        let mut finish: Option<FinishReason> = None;
        let mut usage: Option<TokenUsage> = None;
        while let Ok(chunk) = rx.try_recv() {
            if chunk.finish_reason.is_some() {
                finish = chunk.finish_reason;
            }
            if chunk.usage.is_some() {
                usage = chunk.usage;
            }
            if !chunk.text_delta.is_empty() {
                text_parts.push(chunk.text_delta);
            }
        }
        let joined = text_parts.concat();
        assert_eq!(
            joined, "hello",
            "stop split across chunks must still strip the marker"
        );
        assert!(
            !joined.contains("<|im_") && !joined.contains("im_end") && !joined.contains("|>"),
            "no partial stop-marker bytes may leak (got {joined:?})",
        );
        assert_eq!(finish, Some(FinishReason::Stop));
        assert_eq!(
            usage,
            Some(TokenUsage {
                prompt_tokens: 2,
                completion_tokens: 5,
                total_tokens: 7,
            })
        );
    }

    /// Regression for codex review 2026-04-20 P1 — once a streamed text
    /// stop is matched, the consumer must stop seeing bytes, but final
    /// usage must still come from the backend's real completion result.
    #[cfg(any(feature = "metal", feature = "cpu"))]
    #[tokio::test]
    async fn backend_complete_stream_text_stop_keeps_real_usage() {
        use super::{BackendInferenceEngine, CompletionRequest, InferenceEngine, TokenUsage};
        use crate::backend::{GenerateResult, InferenceBackend, StreamingInferenceBackend};
        use crate::sampler::SamplingParams;
        use std::path::Path;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::sync::mpsc;

        #[derive(Clone)]
        struct CountingStopMock {
            chunks_attempted: Arc<AtomicUsize>,
        }

        impl InferenceBackend for CountingStopMock {
            fn load(&mut self, _p: &Path) -> anyhow::Result<()> {
                Ok(())
            }
            fn generate(
                &self,
                _prompt: &str,
                _params: &SamplingParams,
            ) -> anyhow::Result<GenerateResult> {
                unreachable!()
            }
            fn name(&self) -> &'static str {
                "counting-stop-mock"
            }
        }

        impl StreamingInferenceBackend for CountingStopMock {
            fn generate_stream<F>(
                &self,
                _prompt: &str,
                _params: &SamplingParams,
                mut on_chunk: F,
            ) -> anyhow::Result<GenerateResult>
            where
                F: FnMut(&str) -> anyhow::Result<()>,
            {
                for chunk in ["hello<|im_end|>waste", "never-sent"] {
                    self.chunks_attempted.fetch_add(1, Ordering::Relaxed);
                    if let Err(err) = on_chunk(chunk) {
                        if err
                            .downcast_ref::<crate::backend::StreamStopMatched>()
                            .is_some()
                        {
                            return Ok(GenerateResult {
                                text: "hello<|im_end|>waste".into(),
                                prompt_tokens: 4,
                                completion_tokens: 5,
                                finish_reason: "stop".into(),
                                ttft_ms: 0.0,
                                prompt_tps: 0.0,
                                generation_tps: 0.0,
                                total_time_ms: 0.0,
                            });
                        }
                        return Err(err);
                    }
                }
                Ok(GenerateResult {
                    text: "hello<|im_end|>waste never-sent".into(),
                    prompt_tokens: 4,
                    completion_tokens: 9,
                    finish_reason: "length".into(),
                    ttft_ms: 0.0,
                    prompt_tps: 0.0,
                    generation_tps: 0.0,
                    total_time_ms: 0.0,
                })
            }
        }

        let chunks_attempted = Arc::new(AtomicUsize::new(0));
        let mut engine = BackendInferenceEngine {
            model_id: "counting-stop-mock".into(),
            backend: CountingStopMock {
                chunks_attempted: Arc::clone(&chunks_attempted),
            },
        };

        let (tx, mut rx) = mpsc::unbounded_channel();
        engine
            .complete_stream(
                CompletionRequest {
                    prompt: "p".into(),
                    max_tokens: 32,
                    sampling: SamplingParams::default(),
                    stop: Some(vec!["<|im_end|>".into()]),
                    logprobs: false,
                    session_id: None,
                    trace_context: None,
                },
                tx,
            )
            .unwrap();

        let mut text = String::new();
        let mut finish = None;
        let mut usage = None;
        while let Ok(chunk) = rx.try_recv() {
            text.push_str(&chunk.text_delta);
            if chunk.finish_reason.is_some() {
                finish = chunk.finish_reason;
                usage = chunk.usage;
                break;
            }
        }

        assert_eq!(text, "hello");
        assert_eq!(finish, Some(FinishReason::Stop));
        assert_eq!(
            usage,
            Some(TokenUsage {
                prompt_tokens: 4,
                completion_tokens: 5,
                total_tokens: 9,
            })
        );
        assert_eq!(chunks_attempted.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn model_id_uses_final_path_segment() {
        assert_eq!(
            model_id_from_path("mlx-community/Qwen3-0.6B-4bit"),
            "Qwen3-0.6B-4bit"
        );
        assert_eq!(model_id_from_path("/tmp/models/Qwen3-4B"), "Qwen3-4B");
    }

    #[test]
    fn parse_finish_reason_defaults_to_stop() {
        assert_eq!(parse_finish_reason("length"), FinishReason::Length);
        assert_eq!(parse_finish_reason("stop"), FinishReason::Stop);
        assert_eq!(parse_finish_reason("tool_calls"), FinishReason::Stop);
    }
}
