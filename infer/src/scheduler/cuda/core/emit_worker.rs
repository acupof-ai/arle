//! Emit worker thread: streams completion deltas + finish events to clients.
//!
//! Split out of `core.rs` (pure structural refactor — no behavior change).
//! Owns the `EmitCommand` / `EmitEvent` channel pair plus the worker thread
//! that consumes commands and produces gate-ready events for the scheduler.

use std::collections::HashMap;

use fastrace::Span;
use fastrace::collector::SpanContext;
use tokio::sync::mpsc;

use crate::scheduler::cuda::request::StreamDecodeState;
use crate::server_engine::{CompletionStreamDelta, FinishReason};
use crate::tokenizer::Tokenizer;

pub(in crate::scheduler::cuda) enum EmitCommand {
    Append {
        request_id: u64,
        prompt_tokens: usize,
        tokens: Vec<u32>,
        latest_logprob: Option<f32>,
        delta_tx: mpsc::UnboundedSender<CompletionStreamDelta>,
        stops: Option<Vec<String>>,
        gated: bool,
        trace_context: Option<SpanContext>,
    },
    Finish {
        request_id: u64,
        prompt_tokens: usize,
        completion_tokens: usize,
        generated_tokens: Vec<u32>,
        reason: FinishReason,
        delta_tx: mpsc::UnboundedSender<CompletionStreamDelta>,
        stops: Option<Vec<String>>,
        trace_context: Option<SpanContext>,
    },
    Abort {
        request_id: u64,
    },
}

pub(in crate::scheduler::cuda) enum EmitEvent {
    GateReady { request_id: u64, finished: bool },
}

struct EmitWorkerRequest {
    prompt_tokens: usize,
    generated_tokens: Vec<u32>,
    latest_logprob: Option<f32>,
    delta_tx: mpsc::UnboundedSender<CompletionStreamDelta>,
    stream: StreamDecodeState,
    stops: Option<Vec<String>>,
    stream_flush_span: Option<Span>,
}

pub(in crate::scheduler::cuda) fn spawn_emit_worker(
    tokenizer: Tokenizer,
    stream_interval: usize,
) -> (
    crossbeam_channel::Sender<EmitCommand>,
    crossbeam_channel::Receiver<EmitEvent>,
    std::thread::JoinHandle<()>,
) {
    let (tx, rx) = crossbeam_channel::unbounded();
    let (event_tx, event_rx) = crossbeam_channel::unbounded();
    let stream_interval = stream_interval.max(1);
    let thread = std::thread::Builder::new()
        .name("infer-cuda-emit".to_string())
        .spawn(move || {
            let mut active = HashMap::<u64, EmitWorkerRequest>::new();
            while let Ok(command) = rx.recv() {
                match command {
                    EmitCommand::Append {
                        request_id,
                        prompt_tokens,
                        tokens,
                        latest_logprob,
                        delta_tx,
                        stops,
                        gated,
                        trace_context,
                    } => {
                        let state = active
                            .entry(request_id)
                            .or_insert_with(|| EmitWorkerRequest {
                                prompt_tokens,
                                generated_tokens: Vec::new(),
                                latest_logprob: None,
                                delta_tx,
                                stream: StreamDecodeState::default(),
                                stops,
                                stream_flush_span: trace_context.map(|parent| {
                                    Span::root("stream_flush", parent).with_properties(|| {
                                        [
                                            ("request_id", request_id.to_string()),
                                            ("prompt_tokens", prompt_tokens.to_string()),
                                        ]
                                    })
                                }),
                            });
                        state.generated_tokens.extend(tokens);
                        state.latest_logprob = latest_logprob;
                        let should_flush = state
                            .generated_tokens
                            .len()
                            .saturating_sub(state.stream.decoded_token_count)
                            >= stream_interval;
                        let outcome = if should_flush || gated {
                            state.stream.emit_delta(
                                &state.generated_tokens,
                                &tokenizer,
                                &state.delta_tx,
                                state.latest_logprob,
                                state.stops.as_deref(),
                                state.prompt_tokens,
                            )
                        } else {
                            crate::scheduler::cuda::request::EmitOutcome::Continue
                        };
                        if gated {
                            let finished = matches!(
                                outcome,
                                crate::scheduler::cuda::request::EmitOutcome::Finished
                            );
                            if finished {
                                let finish_parent = state
                                    .stream_flush_span
                                    .as_ref()
                                    .and_then(SpanContext::from_span)
                                    .or(trace_context);
                                let _finish_span = finish_parent.map(|parent| {
                                    Span::root("finish", parent).with_properties(|| {
                                        [("request_id", request_id.to_string())]
                                    })
                                });
                            }
                            let _ = event_tx.send(EmitEvent::GateReady {
                                request_id,
                                finished,
                            });
                            if finished {
                                active.remove(&request_id);
                            }
                        }
                    }
                    EmitCommand::Finish {
                        request_id,
                        prompt_tokens,
                        completion_tokens,
                        generated_tokens,
                        reason,
                        delta_tx,
                        stops,
                        trace_context,
                    } => {
                        if completion_tokens != generated_tokens.len() {
                            log::warn!(
                                "Request {request_id}: finish token count mismatch \
                                 completion_tokens={completion_tokens} generated_token_ids={}",
                                generated_tokens.len()
                            );
                        }
                        if let Some(mut state) = active.remove(&request_id) {
                            let finish_parent = state
                                .stream_flush_span
                                .as_ref()
                                .and_then(SpanContext::from_span)
                                .or(trace_context);
                            let _finish_span = finish_parent.map(|parent| {
                                Span::root("finish", parent)
                                    .with_properties(|| [("request_id", request_id.to_string())])
                            });
                            let finish_tokens =
                                if generated_tokens.len() > state.generated_tokens.len() {
                                    log::debug!(
                                        "Request {request_id}: finish carried {} scheduler tokens \
                                     while emit worker had {}; using scheduler copy",
                                        generated_tokens.len(),
                                        state.generated_tokens.len()
                                    );
                                    &generated_tokens
                                } else {
                                    &state.generated_tokens
                                };
                            if !finish_tokens.is_empty()
                                && tokenizer
                                    .decode(finish_tokens)
                                    .map(|text| text.is_empty())
                                    .unwrap_or(false)
                            {
                                log::warn!(
                                    "Request {request_id}: finish has {} generated token ids \
                                     but decodes to empty visible text",
                                    finish_tokens.len()
                                );
                            }
                            state.stream.finish(
                                finish_tokens,
                                &tokenizer,
                                &state.delta_tx,
                                state.prompt_tokens,
                                reason,
                                state.stops.as_deref(),
                            );
                        } else {
                            let _finish_span = trace_context.map(|parent| {
                                Span::root("finish", parent)
                                    .with_properties(|| [("request_id", request_id.to_string())])
                            });
                            if completion_tokens > 0 {
                                log::warn!(
                                    "Request {request_id}: finish arrived without emit worker \
                                     state; recovering from {} scheduler token ids",
                                    generated_tokens.len()
                                );
                            }
                            StreamDecodeState::default().finish(
                                &generated_tokens,
                                &tokenizer,
                                &delta_tx,
                                prompt_tokens,
                                reason,
                                stops.as_deref(),
                            );
                        }
                    }
                    EmitCommand::Abort { request_id } => {
                        active.remove(&request_id);
                    }
                }
            }
        })
        .expect("emit worker thread");
    (tx, event_rx, thread)
}
