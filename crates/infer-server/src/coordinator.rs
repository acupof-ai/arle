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
use std::time::Duration;

use axum::extract::{DefaultBodyLimit, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use infer_plan::{FinishReason, SamplingParams};

use crate::multiproc_relay::{
    CompletionSinks, RelayCompletionDelta, RelayCoordinator, RelayEnvelope, WireRequest,
};
use crate::schema::{
    ApiError, ChatCompletionRequest, ChatCompletionResponse, CompletionRequest, CompletionResponse,
    ModelsResponse,
};
use crate::tokenizer::OpenAiTokenizer;

/// Idle park on the submit channel; matches the in-process engine loop.
const IDLE_PARK: Duration = Duration::from_millis(2);

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
}

impl CoordinatorHandle {
    fn alloc_request_id(&self) -> u64 {
        self.next_request_id.fetch_add(1, Ordering::Relaxed)
    }
}

/// Build the coordinator router + spawn the lockstep loop thread. `relay` is the
/// accepted [`RelayCoordinator`] (all N ranks connected via `accept_symmetric`).
/// The lockstep loop runs for the process lifetime.
pub fn coordinator_router(
    relay: RelayCoordinator,
    tokenizer: OpenAiTokenizer,
    model: impl Into<String>,
    max_thinking_tokens: usize,
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

    let state = Arc::new(CoordinatorHandle {
        model: model.into(),
        tokenizer: Mutex::new(tokenizer),
        max_thinking_tokens,
        next_request_id: AtomicU64::new(1),
        in_flight,
        submit_tx,
        sinks,
    });

    Router::new()
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/metrics", get(metrics))
        .layer(DefaultBodyLimit::max(256 * 1024 * 1024))
        .with_state(state)
}

/// Broadcasts exactly one `TickAdmissions` per step to every worker (carrying
/// that step's drained submissions, empty on pure-decode ticks), stepping while
/// `in_flight` remains and parking when idle. A broadcast failure breaks the
/// lockstep group, so it logs at error and stops (workers then EOF and exit).
fn lockstep_loop(
    relay: Arc<Mutex<RelayCoordinator>>,
    in_flight: Arc<AtomicUsize>,
    sinks: CompletionSinks,
    submit_rx: SyncReceiver<CoordSubmission>,
) {
    let mut seq: u64 = 0;
    let mut submit_open = true;
    loop {
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

        // LOCKSTEP INVARIANT (load-bearing): `in_flight > 0` must strictly contain
        // every rank's non-idle window — increment BEFORE submit, decrement only
        // AFTER the terminal delta. Then the coordinator can only OVER-send
        // decode-only ticks (idle workers skip them), never UNDER-send one a worker
        // needs for an NCCL collective. Reordering the inc/dec can deadlock NCCL.
        let busy = in_flight.load(Ordering::Acquire) > 0;
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

/// Submit one request and collect its full generation by draining relay deltas.
/// `in_flight` brackets the request so the lockstep loop keeps stepping until it
/// terminally completes; cancellation/early return is handled by [`InFlightGuard`].
async fn submit_and_collect(
    state: &Arc<CoordinatorHandle>,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    sampling: SamplingParams,
) -> Result<CollectedGeneration, ApiError> {
    let request_id = state.alloc_request_id();
    let prompt_len = prompt_tokens.len();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<RelayCompletionDelta>();

    // Register the sink BEFORE submitting so no delta can race ahead of it.
    state
        .sinks
        .register(request_id, tx)
        .map_err(ApiError::from)?;

    // Increment in_flight + arm the guard BEFORE submit: the increment-before-submit
    // ordering is the load-bearing lockstep invariant (see `lockstep_loop`), and the
    // guard releases both decrement + unregister on every early-return path.
    state.in_flight.fetch_add(1, Ordering::AcqRel);
    let _guard = InFlightGuard {
        in_flight: Arc::clone(&state.in_flight),
        sinks: state.sinks.clone(),
        request_id,
    };

    let submit_result = state.submit_tx.send(CoordSubmission {
        request: WireRequest {
            request_id,
            prompt_tokens,
            max_tokens,
            sampling,
        },
    });
    if submit_result.is_err() {
        return Err(ApiError::internal(
            "coordinator lockstep loop closed; cannot submit",
        ));
    }

    // Accumulate deltas until terminal. `recv()` also yields `None` when
    // `fail_all` dropped the sender on worker-group failure (handled below).
    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut finish: Option<FinishReason> = None;
    let mut error: Option<String> = None;
    while let Some(delta) = rx.recv().await {
        generated_tokens.extend_from_slice(&delta.token_ids);
        let done = delta.is_done();
        if delta.error.is_some() {
            error = delta.error;
        }
        if done {
            finish = delta.finish_reason;
            break;
        }
    }

    // `_guard` drops here (fetch_sub + unregister), as on any early return above.
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
        // Streaming is not wired on the coordinator path (blocking-only); fail closed.
        return Err(ApiError::bad_request(
            "streaming is not supported on the multiproc coordinator path; use stream=false",
        ));
    }
    let outcome = submit_and_collect(&state, prompt_tokens, max_tokens, sampling).await?;
    let text = decode(&state, &outcome.generated_tokens)?;
    Ok(Json(CompletionResponse::from_parts(
        state.model.clone(),
        text,
        outcome.prompt_tokens,
        outcome.generated_tokens.len(),
        outcome.finish.as_ref(),
    ))
    .into_response())
}

async fn chat_completions(
    State(state): State<Arc<CoordinatorHandle>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, ApiError> {
    request.validate()?;
    let sampling = request.sampling_params();
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
    let prompt = {
        let tokenizer = state
            .tokenizer
            .lock()
            .map_err(|_| ApiError::internal("tokenizer lock poisoned"))?;
        tokenizer
            .render_chat_with_kwargs(&request.messages, request.chat_template_kwargs.as_ref())?
    };
    let prompt_tokens = encode(&state, &prompt)?;
    let outcome = submit_and_collect(&state, prompt_tokens, max_tokens, sampling).await?;
    let content = decode(&state, &outcome.generated_tokens)?;
    Ok(Json(ChatCompletionResponse::from_parts(
        state.model.clone(),
        content,
        outcome.prompt_tokens,
        outcome.generated_tokens.len(),
        outcome.finish.as_ref(),
        thinking,
    )))
}

async fn list_models(State(state): State<Arc<CoordinatorHandle>>) -> Json<ModelsResponse> {
    Json(ModelsResponse::single(state.model.clone()))
}

async fn metrics(
    State(_state): State<Arc<CoordinatorHandle>>,
) -> ([(header::HeaderName, &'static str); 1], String) {
    // The coordinator owns no engine counters (the per-rank engines hold them);
    // expose a minimal scrape surface so monitoring probes do not 404.
    (
        [(
            header::CONTENT_TYPE,
            crate::metrics::PROMETHEUS_CONTENT_TYPE,
        )],
        String::from("# arle multiproc coordinator: per-rank engine counters not aggregated\n"),
    )
}
