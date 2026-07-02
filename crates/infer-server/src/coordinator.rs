//! SPMD multiproc coordinator: the thin control plane (B split).
//!
//! The parent owns NO TP rank — all `world_size` ranks are symmetric child
//! workers — so it joins no collective and can never be the missing rank at a
//! barrier. It bridges async HTTP (tokio) to the sync relay (blocking TCP): a
//! dedicated-thread lockstep loop broadcasts one `TickAdmissions` per step to
//! every worker, and per-rank completion readers route rank-0's deltas into the
//! per-request sinks the async handlers await. TP=1 never reaches here.
//! See `docs/plans/2026-06-24-multiproc-control-data-plane-redesign.md`.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver as SyncReceiver, RecvTimeoutError, Sender as SyncSender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::{DefaultBodyLimit, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use infer_plan::{FinishReason, MultimodalKind, SamplingParams};
use uuid::Uuid;

use crate::multiproc_relay::{
    CompletionSinks, RelayCompletionDelta, RelayCoordinator, RelayEnvelope, WireRequest,
};
use crate::schema::{
    ApiError, ChatCompletionRequest, ChatCompletionResponse, CompletionRequest, CompletionResponse,
    ModelsResponse, StatsResponse,
};
use crate::sse_util::{
    StreamingReasoningSplitter, chat_stream_chunk, completion_stream_chunk, finish_reason,
    unix_time_secs,
};
use crate::tokenizer::OpenAiTokenizer;

/// Idle park on the submit channel; matches the in-process engine loop.
const IDLE_PARK: Duration = Duration::from_millis(2);

/// Cap on ticks broadcast beyond the slowest rank's [`RelayEnvelope::TickAck`].
/// Pacing the tick stream to engine speed bounds the worker FIFO depth, so a
/// mid-decode submission (or `StatsQuery`, which rides the same FIFO) waits
/// ≤ window × step time instead of queue-depth × step time (measured pre-fix:
/// ~608k queued ticks for ~600 engine steps).
const TICK_WINDOW: u64 = 4;

/// Block until `seq` is within [`TICK_WINDOW`] of the slowest rank's tick ack.
/// Never aborts: a wedged worker already hangs its NCCL peers today, and a long
/// prefill chunk legitimately holds acks for seconds — warn (rate-limited) and
/// keep waiting.
fn wait_for_ack_window(relay: &Arc<Mutex<RelayCoordinator>>, seq: u64) {
    let started = Instant::now();
    let mut next_warn = Duration::from_secs(10);
    loop {
        let min_acked = relay
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .min_acked_ticks();
        if seq < min_acked + TICK_WINDOW {
            return;
        }
        if started.elapsed() >= next_warn {
            log::warn!(
                "[coordinator] lockstep stalled: tick #{seq} awaiting acks (min_acked={min_acked})"
            );
            next_warn += Duration::from_secs(10);
        }
        std::thread::sleep(Duration::from_micros(500));
    }
}

struct CoordSubmission {
    request: WireRequest,
}

/// The coordinator's HTTP-facing state (lives behind `Arc` in the axum router).
pub struct CoordinatorHandle {
    model: String,
    tokenizer: Mutex<OpenAiTokenizer>,
    max_thinking_tokens: usize,
    next_request_id: AtomicU64,
    /// Submitted but not yet terminally completed; gates decode-only ticks.
    in_flight: Arc<AtomicUsize>,
    submit_tx: SyncSender<CoordSubmission>,
    /// Registered/unregistered directly (not via the `RelayCoordinator` mutex) so
    /// HTTP (un)register never contends with the lockstep loop's blocking broadcast.
    sinks: CompletionSinks,
    /// Relay handle for stats queries to rank-0.
    relay: Arc<Mutex<RelayCoordinator>>,
    /// Monotonic counter for stats query request IDs (separate from request IDs to
    /// avoid collisions with completion sink IDs).
    stats_request_id: AtomicU64,
    /// In-process channel to the multimodal driver thread (VLM backends only).
    multimodal_tx: Option<crate::LocalMultimodalTx>,
    /// Multimodal kind for the current backend (VLM backends only).
    multimodal_kind: Option<MultimodalKind>,
}

impl CoordinatorHandle {
    fn alloc_request_id(&self) -> u64 {
        self.next_request_id.fetch_add(1, Ordering::Relaxed)
    }
}

/// Build the coordinator router + spawn the lockstep loop thread. `relay` is the
/// accepted [`RelayCoordinator`] (all N ranks connected via `accept_symmetric`).
/// The lockstep loop runs for the process lifetime. Pass `multimodal` for VLM
/// backends; text-only backends pass `None`.
#[allow(private_interfaces)]
pub fn coordinator_router(
    relay: RelayCoordinator,
    tokenizer: OpenAiTokenizer,
    model: impl Into<String>,
    max_thinking_tokens: usize,
    multimodal: Option<(crate::LocalMultimodalTx, MultimodalKind)>,
) -> Router {
    let sinks = relay.completion_sinks();
    let relay = Arc::new(Mutex::new(relay));
    let in_flight = Arc::new(AtomicUsize::new(0));
    let (submit_tx, submit_rx) = std::sync::mpsc::channel::<CoordSubmission>();

    {
        let relay = Arc::clone(&relay);
        let in_flight = Arc::clone(&in_flight);
        let sinks = sinks.clone();
        std::thread::Builder::new()
            .name("arle-coordinator-lockstep".to_string())
            .spawn(move || lockstep_loop(relay, in_flight, sinks, submit_rx))
            .expect("spawn coordinator lockstep thread");
    }

    let (multimodal_tx, multimodal_kind) = multimodal.unzip();

    let state = Arc::new(CoordinatorHandle {
        model: model.into(),
        tokenizer: Mutex::new(tokenizer),
        max_thinking_tokens,
        next_request_id: AtomicU64::new(1),
        in_flight,
        submit_tx,
        sinks,
        relay,
        stats_request_id: AtomicU64::new(0),
        multimodal_tx,
        multimodal_kind,
    });

    Router::new()
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/v1/stats", get(stats))
        .route("/metrics", get(metrics))
        .layer(DefaultBodyLimit::max(256 * 1024 * 1024))
        .with_state(state)
}

/// Broadcasts exactly one `TickAdmissions` per step to every worker (carrying
/// that step's drained submissions, empty on pure-decode ticks), stepping while
/// `in_flight` remains and parking when idle. Per-rank tick acks pace the busy
/// path to engine speed ([`wait_for_ack_window`]), so ticks never flood the
/// worker FIFOs. A broadcast failure breaks the lockstep group, so it logs at
/// error and stops (workers then EOF and exit).
fn lockstep_loop(
    relay: Arc<Mutex<RelayCoordinator>>,
    in_flight: Arc<AtomicUsize>,
    sinks: CompletionSinks,
    submit_rx: SyncReceiver<CoordSubmission>,
) {
    let mut seq: u64 = 0;
    let mut submit_open = true;
    loop {
        // LOCKSTEP INVARIANT (load-bearing): `in_flight > 0` must strictly contain
        // every rank's non-idle window — increment BEFORE submit, decrement only
        // AFTER the terminal delta. Then the coordinator can only OVER-send
        // decode-only ticks (idle workers skip them), never UNDER-send one a worker
        // needs for an NCCL collective. Reordering the inc/dec can deadlock NCCL.
        let busy = in_flight.load(Ordering::Acquire) > 0;
        if busy {
            // Flow control: wait for the ack window BEFORE the drain, so a
            // submission landing during the wait coalesces into the very next
            // allowed tick instead of queueing behind unacked ones.
            wait_for_ack_window(&relay, seq);
        }

        // Drain queued submissions without blocking.
        let mut drained: Vec<WireRequest> = Vec::new();
        loop {
            match submit_rx.try_recv() {
                Ok(sub) => drained.push(sub.request),
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    submit_open = false;
                    break;
                }
            }
        }

        if !drained.is_empty() || busy {
            let requests = drained;
            let send = {
                let mut coord = relay
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                coord.broadcast(&RelayEnvelope::TickAdmissions { seq, requests })
            };
            if let Err(err) = send {
                log::error!(
                    "[coordinator] tick #{seq} admission broadcast failed; stopping lockstep \
                     loop (workers will EOF and exit): {err:#}"
                );
                // Worker group broken: fail every in-flight sink so awaiting
                // handlers return an error (and drop their guard) instead of hanging.
                sinks.fail_all("multiproc worker group failed (admission broadcast error)");
                return;
            }
            seq += 1;
            continue;
        }

        // Fully idle: exit if the frontend is gone, else park for the next submit.
        if !submit_open {
            return;
        }
        match submit_rx.recv_timeout(IDLE_PARK) {
            Ok(sub) => {
                let send = {
                    let mut coord = relay
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    coord.broadcast(&RelayEnvelope::TickAdmissions {
                        seq,
                        requests: vec![sub.request],
                    })
                };
                if let Err(err) = send {
                    log::error!(
                        "[coordinator] tick #{seq} admission broadcast failed; stopping lockstep \
                         loop: {err:#}"
                    );
                    sinks.fail_all("multiproc worker group failed (admission broadcast error)");
                    return;
                }
                seq += 1;
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => submit_open = false,
        }
    }
}

/// One coordinator request's result, reconstructed from relay completion deltas.
struct CollectedGeneration {
    prompt_tokens: usize,
    generated_tokens: Vec<u32>,
    finish: Option<FinishReason>,
}

/// RAII guard for one in-flight request: `Drop` does the `in_flight` decrement +
/// sink unregister, so a cancelled HTTP task still releases both (a leaked
/// `in_flight` would pin the lockstep loop into a forever busy-spin). Armed after
/// sink registration and before the submit await to cover the whole window.
struct InFlightGuard {
    in_flight: Arc<AtomicUsize>,
    sinks: CompletionSinks,
    request_id: u64,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.in_flight.fetch_sub(1, Ordering::AcqRel);
        self.sinks.unregister(self.request_id);
    }
}

/// Set up the relay sink + in_flight guard + submit for a streaming request.
/// Returns the per-token delta receiver and the guard (caller must keep alive
/// until streaming ends so `in_flight` is decremented only after the last delta).
fn streaming_submit(
    state: &Arc<CoordinatorHandle>,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    sampling: SamplingParams,
) -> Result<
    (
        tokio::sync::mpsc::UnboundedReceiver<RelayCompletionDelta>,
        InFlightGuard,
    ),
    ApiError,
> {
    let request_id = state.alloc_request_id();
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<RelayCompletionDelta>();
    state
        .sinks
        .register(request_id, tx)
        .map_err(ApiError::from)?;
    state.in_flight.fetch_add(1, Ordering::AcqRel);
    let guard = InFlightGuard {
        in_flight: Arc::clone(&state.in_flight),
        sinks: state.sinks.clone(),
        request_id,
    };
    state
        .submit_tx
        .send(CoordSubmission {
            request: WireRequest {
                request_id,
                prompt_tokens,
                max_tokens,
                sampling,
            },
        })
        .map_err(|_| ApiError::internal("coordinator lockstep loop closed; cannot submit"))?;
    Ok((rx, guard))
}

async fn submit_and_collect(
    state: &Arc<CoordinatorHandle>,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    sampling: SamplingParams,
) -> Result<CollectedGeneration, ApiError> {
    let prompt_len = prompt_tokens.len();
    let (mut rx, _guard) = streaming_submit(state, prompt_tokens, max_tokens, sampling)?;
    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut finish: Option<FinishReason> = None;
    let mut error: Option<String> = None;
    while let Some(delta) = rx.recv().await {
        generated_tokens.extend_from_slice(&delta.token_ids);
        let done = delta.is_done();
        error = error.or(delta.error);
        if done {
            finish = delta.finish_reason;
            break;
        }
    }
    if let Some(message) = error {
        return Err(ApiError::internal(message));
    }
    Ok(CollectedGeneration {
        prompt_tokens: prompt_len,
        generated_tokens,
        finish,
    })
}

fn encode(state: &Arc<CoordinatorHandle>, text: &str) -> Result<Vec<u32>, ApiError> {
    let tokenizer = state
        .tokenizer
        .lock()
        .map_err(|_| ApiError::internal("tokenizer lock poisoned"))?;
    Ok(tokenizer.encode(text)?)
}

fn decode(state: &Arc<CoordinatorHandle>, tokens: &[u32]) -> Result<String, ApiError> {
    let tokenizer = state
        .tokenizer
        .lock()
        .map_err(|_| ApiError::internal("tokenizer lock poisoned"))?;
    Ok(tokenizer.decode(tokens)?)
}

async fn completions(
    State(state): State<Arc<CoordinatorHandle>>,
    Json(request): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    request.validate()?;
    let sampling = request.sampling_params();
    let max_tokens = sampling.max_new_tokens.unwrap_or(16);
    let prompt_tokens = encode(&state, &request.prompt)?;

    if request.stream.unwrap_or(false) {
        let (mut rx, guard) = streaming_submit(&state, prompt_tokens, max_tokens, sampling)?;
        let id = format!("cmpl-{}", Uuid::new_v4().simple());
        let created = unix_time_secs();
        let model = state.model.clone();
        let state_clone = Arc::clone(&state);
        // Bounded channel: backpressure keeps the task from racing too far ahead.
        let (chunk_tx, chunk_rx) =
            tokio::sync::mpsc::channel::<Result<Vec<u8>, std::convert::Infallible>>(64);
        tokio::spawn(async move {
            // `guard` dropped when this task exits, decrementing in_flight + unregistering sink.
            let _guard = guard;
            while let Some(delta) = rx.recv().await {
                if let Some(err) = delta.error {
                    let chunk = serde_json::to_string(&completion_stream_chunk(
                        &id,
                        created,
                        &model,
                        err,
                        Some("error"),
                        None,
                    ))
                    .unwrap_or_default();
                    let _ = chunk_tx
                        .send(Ok(format!("data: {chunk}\n\ndata: [DONE]\n\n").into_bytes()))
                        .await;
                    break;
                }
                if !delta.token_ids.is_empty() {
                    let text = {
                        let tok = state_clone
                            .tokenizer
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        tok.decode(&delta.token_ids).unwrap_or_default()
                    };
                    let chunk = serde_json::to_string(&completion_stream_chunk(
                        &id, created, &model, text, None, None,
                    ))
                    .unwrap_or_default();
                    if chunk_tx
                        .send(Ok(format!("data: {chunk}\n\n").into_bytes()))
                        .await
                        .is_err()
                    {
                        break; // Client disconnected; guard drops, releasing in_flight.
                    }
                }
                if delta.finish {
                    let fr = finish_reason(delta.finish_reason.as_ref());
                    let final_chunk = serde_json::to_string(&completion_stream_chunk(
                        &id,
                        created,
                        &model,
                        String::new(),
                        Some(fr),
                        None,
                    ))
                    .unwrap_or_default();
                    let _ = chunk_tx
                        .send(Ok(
                            format!("data: {final_chunk}\n\ndata: [DONE]\n\n").into_bytes()
                        ))
                        .await;
                    break;
                }
            }
        });
        let body =
            axum::body::Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(chunk_rx));
        return Ok((
            [
                (header::CONTENT_TYPE, "text/event-stream"),
                (header::CACHE_CONTROL, "no-cache"),
                (header::HeaderName::from_static("x-accel-buffering"), "no"),
            ],
            body,
        )
            .into_response());
    }

    let outcome = submit_and_collect(&state, prompt_tokens, max_tokens, sampling).await?;
    let text = decode(&state, &outcome.generated_tokens)?;
    Ok(Json(CompletionResponse::from_parts(
        state.model.clone(),
        text,
        outcome.prompt_tokens,
        outcome.generated_tokens.len(),
        outcome.finish.as_ref(),
        request
            .return_token_ids
            .unwrap_or(false)
            .then_some(outcome.generated_tokens),
    ))
    .into_response())
}

async fn chat_completions(
    State(state): State<Arc<CoordinatorHandle>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    request.validate()?;
    let sampling = request.sampling_params();
    let stream = request.stream.unwrap_or(false);
    // A configured thinking budget also flips the server default to thinking-on
    // (so terminus/litellm, which can't set the kwarg, still gets the split +
    // budget); `0` keeps it off and byte-identical. Mirrors the in-process path.
    let thinking = request.enable_thinking(state.max_thinking_tokens > 0);
    let mut max_tokens = sampling.max_new_tokens.unwrap_or_else(|| {
        if thinking && state.max_thinking_tokens > 0 {
            state.max_thinking_tokens
        } else {
            16
        }
    });
    if state.max_thinking_tokens > 0 && thinking {
        max_tokens = max_tokens.min(state.max_thinking_tokens);
    }

    // Multimodal dispatch: if the backend is a VLM and the request has images,
    // route through the in-process multimodal channel instead of the relay.
    if let Some(kind) = state.multimodal_kind {
        let images = crate::multimodal::extract_images(&request.messages, Some(kind))?;
        if !images.is_empty() {
            if stream {
                return Err(ApiError::bad_request(
                    "stream=true is not supported for multimodal chat yet",
                ));
            }
            let prompt = if kind == MultimodalKind::DeepseekOcr {
                crate::multimodal::build_deepseek_ocr_prompt(&request.messages)
            } else {
                let tokenizer = state
                    .tokenizer
                    .lock()
                    .map_err(|_| ApiError::internal("tokenizer lock poisoned"))?;
                tokenizer.render_chat_with_kwargs(
                    &request.messages,
                    request.chat_template_kwargs.as_ref(),
                )?
            };
            let prompt = crate::multimodal::expand_image_markers(&prompt, &images, Some(kind))?;
            let prompt_tokens = encode(&state, &prompt)?;
            let prompt_token_count = prompt_tokens.len();
            let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
            state
                .multimodal_tx
                .as_ref()
                .ok_or_else(|| ApiError::internal("multimodal channel not initialized"))?
                .send(crate::LocalMultimodalRequest {
                    prompt_tokens,
                    images,
                    max_tokens,
                    sampling,
                    response_tx: resp_tx,
                })
                .map_err(|_| ApiError::internal("multimodal channel closed"))?;
            let delta = tokio::time::timeout(std::time::Duration::from_secs(300), resp_rx)
                .await
                .map_err(|_| ApiError::internal("multimodal request timed out"))?
                .map_err(|_| ApiError::internal("multimodal response channel closed"))?;
            if let Some(err) = delta.error {
                return Err(ApiError::internal(err));
            }
            let content = decode(&state, &delta.token_ids)?;
            return Ok(Json(ChatCompletionResponse::from_parts(
                state.model.clone(),
                content,
                prompt_token_count,
                delta.token_ids.len(),
                delta.finish_reason.as_ref(),
                thinking,
            ))
            .into_response());
        }
    }

    let prompt = {
        let tokenizer = state
            .tokenizer
            .lock()
            .map_err(|_| ApiError::internal("tokenizer lock poisoned"))?;
        tokenizer
            .render_chat_with_kwargs(&request.messages, request.chat_template_kwargs.as_ref())?
    };
    let prompt_tokens = encode(&state, &prompt)?;

    if stream {
        let (mut rx, guard) = streaming_submit(&state, prompt_tokens, max_tokens, sampling)?;
        let id = format!("chatcmpl-{}", Uuid::new_v4().simple());
        let created = unix_time_secs();
        let model = state.model.clone();
        let state_clone = Arc::clone(&state);
        // Bounded channel: backpressure keeps the task from racing too far ahead.
        let (chunk_tx, chunk_rx) =
            tokio::sync::mpsc::channel::<Result<Vec<u8>, std::convert::Infallible>>(64);
        tokio::spawn(async move {
            // `guard` dropped when this task exits, decrementing in_flight + unregistering sink.
            let _guard = guard;
            let mut splitter = StreamingReasoningSplitter::new(thinking);
            // OpenAI convention: the first emitted chunk's delta carries `role`.
            let mut role_sent = false;
            let mut with_role = move |mut delta: serde_json::Value| {
                if !std::mem::replace(&mut role_sent, true) {
                    delta["role"] = serde_json::json!("assistant");
                }
                delta
            };
            'stream: while let Some(delta) = rx.recv().await {
                if let Some(err) = delta.error {
                    let chunk = serde_json::to_string(&chat_stream_chunk(
                        &id,
                        created,
                        &model,
                        with_role(serde_json::json!({"content": err})),
                        Some("error"),
                    ))
                    .unwrap_or_default();
                    let _ = chunk_tx
                        .send(Ok(format!("data: {chunk}\n\ndata: [DONE]\n\n").into_bytes()))
                        .await;
                    break;
                }
                if !delta.token_ids.is_empty() {
                    let text = {
                        let tok = state_clone
                            .tokenizer
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        tok.decode(&delta.token_ids).unwrap_or_default()
                    };
                    for piece in splitter.push(&text) {
                        let chunk = serde_json::to_string(&chat_stream_chunk(
                            &id,
                            created,
                            &model,
                            with_role(piece.into_delta()),
                            None,
                        ))
                        .unwrap_or_default();
                        if chunk_tx
                            .send(Ok(format!("data: {chunk}\n\n").into_bytes()))
                            .await
                            .is_err()
                        {
                            break 'stream; // Client disconnected; guard drops, releasing in_flight.
                        }
                    }
                }
                if delta.finish {
                    // Truncated thinking: flush the held-back partial closer as reasoning.
                    if let Some(piece) = splitter.finish() {
                        let chunk = serde_json::to_string(&chat_stream_chunk(
                            &id,
                            created,
                            &model,
                            with_role(piece.into_delta()),
                            None,
                        ))
                        .unwrap_or_default();
                        if chunk_tx
                            .send(Ok(format!("data: {chunk}\n\n").into_bytes()))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    let fr = finish_reason(delta.finish_reason.as_ref());
                    let final_chunk = serde_json::to_string(&chat_stream_chunk(
                        &id,
                        created,
                        &model,
                        with_role(serde_json::json!({})),
                        Some(fr),
                    ))
                    .unwrap_or_default();
                    let _ = chunk_tx
                        .send(Ok(
                            format!("data: {final_chunk}\n\ndata: [DONE]\n\n").into_bytes()
                        ))
                        .await;
                    break;
                }
            }
        });
        let body =
            axum::body::Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(chunk_rx));
        return Ok((
            [
                (header::CONTENT_TYPE, "text/event-stream"),
                (header::CACHE_CONTROL, "no-cache"),
                (header::HeaderName::from_static("x-accel-buffering"), "no"),
            ],
            body,
        )
            .into_response());
    }

    let outcome = submit_and_collect(&state, prompt_tokens, max_tokens, sampling).await?;
    let content = decode(&state, &outcome.generated_tokens)?;
    Ok(Json(ChatCompletionResponse::from_parts(
        state.model.clone(),
        content,
        outcome.prompt_tokens,
        outcome.generated_tokens.len(),
        outcome.finish.as_ref(),
        thinking,
    ))
    .into_response())
}

async fn list_models(State(state): State<Arc<CoordinatorHandle>>) -> Json<ModelsResponse> {
    Json(ModelsResponse::single(state.model.clone()))
}

async fn metrics(
    State(state): State<Arc<CoordinatorHandle>>,
) -> ([(header::HeaderName, &'static str); 1], String) {
    let request_id = state.stats_request_id.fetch_add(1, Ordering::Relaxed);
    let (rx, query_ok) = {
        let mut relay = state
            .relay
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        (
            relay.register_stats_awaiter(request_id),
            relay.send_stats_query(request_id).is_ok(),
        )
    };
    let counters = if query_ok {
        tokio::time::timeout(std::time::Duration::from_secs(2), rx)
            .await
            .ok()
            .and_then(|r| r.ok())
            .map(|w| w.into_counter_snapshot())
            .unwrap_or_default()
    } else {
        crate::execution::CounterSnapshot::default()
    };
    (
        [(
            header::CONTENT_TYPE,
            crate::metrics::PROMETHEUS_CONTENT_TYPE,
        )],
        crate::metrics::render_prometheus(&counters, &state.model),
    )
}

async fn stats(
    State(state): State<Arc<CoordinatorHandle>>,
) -> Result<Json<StatsResponse>, ApiError> {
    let request_id = state.stats_request_id.fetch_add(1, Ordering::Relaxed);
    // Register awaiter BEFORE sending to avoid a race with the reader thread.
    let (rx, send_result) = {
        let mut relay = state
            .relay
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let rx = relay.register_stats_awaiter(request_id);
        let result = relay
            .send_stats_query(request_id)
            .map_err(|e| ApiError::internal(e.to_string()));
        (rx, result)
    };
    send_result?;
    let wire = tokio::time::timeout(std::time::Duration::from_secs(5), rx)
        .await
        .map_err(|_| ApiError::internal("stats query timed out"))?
        .map_err(|_| ApiError::internal("stats oneshot closed"))?;
    Ok(Json(StatsResponse::from_wire(wire)))
}
