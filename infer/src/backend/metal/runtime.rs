use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use anyhow::{Context, Result, anyhow, bail};
use log::{error, info, warn};
use tokio::sync::mpsc;

use super::request_state::{MetalRequestPhase as RuntimePhase, MetalRequestState};
use super::scheduler::{
    MetalRequestPriority, MetalScheduleDecision, MetalScheduler, MetalSchedulerConfig,
};
use super::{MetalBackend, MetalBackendOptions};
use crate::backend::InferenceBackend;
use crate::scheduler::{IncomingRequest, RequestPriority, SchedulerHandle};
use crate::server_engine::{CompletionStreamDelta, FinishReason, TokenUsage};
use crate::tokenizer::{IncrementalDecoder, Tokenizer};
use crate::types::RequestId;

struct ActiveMetalRequest {
    delta_tx: mpsc::UnboundedSender<CompletionStreamDelta>,
    request_state: MetalRequestState<'static>,
    decoder: IncrementalDecoder<'static>,
    stop_processor: StopChunkProcessor,
    prompt_tokens: usize,
}

impl ActiveMetalRequest {
    fn from_incoming(
        backend: &'static MetalBackend,
        tokenizer: &'static Tokenizer,
        incoming: IncomingRequest,
    ) -> Result<(Vec<u32>, usize, Self)> {
        let prompt_tokens = tokenizer.encode(&incoming.prompt)?;
        let max_tokens = incoming.max_tokens;
        let request_state = backend.create_request_state(&prompt_tokens, &incoming.sampling)?;
        Ok((
            prompt_tokens.clone(),
            max_tokens,
            Self {
                delta_tx: incoming.delta_tx,
                request_state,
                decoder: tokenizer.incremental_decoder(),
                stop_processor: StopChunkProcessor::new(incoming.stop.unwrap_or_default()),
                prompt_tokens: prompt_tokens.len(),
            },
        ))
    }

    fn delta_closed(&self) -> bool {
        self.delta_tx.is_closed()
    }

    fn phase(&self) -> RuntimePhase {
        self.request_state.phase()
    }

    fn stop_hit(&self) -> bool {
        self.stop_processor.hit_stop()
    }

    fn prefill_chunk(&mut self, budget: usize) -> Result<Option<u32>> {
        let result = self.request_state.prefill_chunk(budget)?;
        if let Some(token) = result.emitted_token {
            self.process_token(token)?;
            Ok(Some(token))
        } else {
            Ok(None)
        }
    }

    fn decode_step(&mut self) -> Result<u32> {
        let token = self
            .request_state
            .decode_step()?
            .context("decode_step did not emit a token")?;
        self.process_token(token)?;
        Ok(token)
    }

    fn cancel(&mut self) -> Result<()> {
        self.request_state.cancel()
    }

    fn send_final_delta(&mut self) -> Result<()> {
        if let Some(tail) = self.decoder.finish()? {
            self.push_text_chunk(&tail)?;
        }
        if let Some(final_delta) = self.stop_processor.finish() {
            send_text_delta(&self.delta_tx, final_delta)?;
        }

        let finish_reason = if self.stop_processor.hit_stop() {
            FinishReason::Stop
        } else {
            map_finish_reason(self.request_state.finish_reason())
        };
        let completion_tokens = self.request_state.generated_tokens();
        let usage = TokenUsage {
            prompt_tokens: self.prompt_tokens,
            completion_tokens,
            total_tokens: self.prompt_tokens + completion_tokens,
        };

        let _ = self.delta_tx.send(CompletionStreamDelta {
            text_delta: String::new(),
            finish_reason: Some(finish_reason),
            usage: Some(usage),
            logprob: None,
        });
        Ok(())
    }

    fn process_token(&mut self, token_id: u32) -> Result<()> {
        if let Some(chunk) = self.decoder.step(token_id)? {
            self.push_text_chunk(&chunk)?;
        }
        Ok(())
    }

    fn push_text_chunk(&mut self, chunk: &str) -> Result<()> {
        if let Some(delta) = self.stop_processor.push_chunk(chunk) {
            send_text_delta(&self.delta_tx, delta)?;
        }
        Ok(())
    }
}

/// Spawn the first live Metal scheduler runtime.
///
/// This runtime uses the request-state API to interleave chunked prefill and
/// decode scheduling. It does not yet implement cross-request batched GPU
/// decode; decode steps are still executed request-by-request inside the
/// scheduler loop.
pub fn spawn_metal_scheduler_handle_from_path_with_options(
    model_path: &str,
    options: MetalBackendOptions,
    max_waiting: usize,
) -> Result<SchedulerHandle> {
    if options.dflash.is_some() {
        bail!(
            "the live Metal scheduler runtime does not support DFlash yet; use the serial runtime path instead"
        );
    }

    let mut backend = MetalBackend::with_options(options);
    backend.load(Path::new(model_path))?;

    let model_id = Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(model_path)
        .to_string();

    let (tx, rx) = mpsc::unbounded_channel();
    let waiting_count = Arc::new(AtomicUsize::new(0));
    let handle =
        SchedulerHandle::with_shared_waiting_count(tx, &model_id, max_waiting, waiting_count);

    let runtime_handle = handle.clone();
    std::thread::spawn(move || {
        // The runtime owns one backend instance for the process lifetime. The
        // request-state API currently borrows backend internals, so keep the
        // loaded backend stable inside the worker thread until the server exits.
        let backend: &'static MetalBackend = Box::leak(Box::new(backend));
        let tokenizer = match backend.tokenizer.as_ref() {
            Some(tokenizer) => tokenizer,
            None => {
                error!("Metal scheduler runtime failed: model tokenizer not loaded");
                return;
            }
        };
        let tokenizer: &'static Tokenizer = tokenizer;

        let result = catch_unwind(AssertUnwindSafe(|| {
            run_metal_scheduler_runtime(
                backend,
                tokenizer,
                rx,
                runtime_handle,
                MetalSchedulerConfig {
                    max_waiting_requests: 0,
                    ..MetalSchedulerConfig::default()
                },
            )
        }));

        match result {
            Ok(Ok(())) => {}
            Ok(Err(err)) => error!("Metal scheduler runtime failed: {err:#}"),
            Err(panic) => error!(
                "Metal scheduler runtime panicked: {}",
                super::panic_message(panic)
            ),
        }
    });

    Ok(handle)
}

pub fn spawn_metal_scheduler_handle_from_path(
    model_path: &str,
    max_waiting: usize,
) -> Result<SchedulerHandle> {
    spawn_metal_scheduler_handle_from_path_with_options(
        model_path,
        MetalBackendOptions::default(),
        max_waiting,
    )
}

fn run_metal_scheduler_runtime(
    backend: &'static MetalBackend,
    tokenizer: &'static Tokenizer,
    mut request_rx: mpsc::UnboundedReceiver<IncomingRequest>,
    handle: SchedulerHandle,
    config: MetalSchedulerConfig,
) -> Result<()> {
    let mut scheduler = MetalScheduler::new(config)?;
    let mut active = HashMap::<RequestId, ActiveMetalRequest>::new();
    let mut request_rx_closed = false;

    info!("Metal scheduler runtime started");

    loop {
        drain_incoming_requests(
            backend,
            tokenizer,
            &handle,
            &mut request_rx,
            &mut request_rx_closed,
            &mut scheduler,
            &mut active,
        );
        reap_closed_clients(&mut scheduler, &mut active);

        if request_rx_closed && active.is_empty() && scheduler.waiting_len() == 0 {
            info!("Metal scheduler runtime shutting down: all handles dropped");
            break;
        }

        if active.is_empty() && scheduler.waiting_len() == 0 {
            match request_rx.blocking_recv() {
                Some(incoming) => {
                    handle.consume_one();
                    admit_request(backend, tokenizer, incoming, &mut scheduler, &mut active);
                }
                None => {
                    request_rx_closed = true;
                    continue;
                }
            }
        }

        match scheduler.step() {
            MetalScheduleDecision::Idle => continue,
            MetalScheduleDecision::PrefillChunk(prefill) => execute_prefill_chunk(
                prefill.req_id,
                prefill.input_tokens.len(),
                &mut scheduler,
                &mut active,
            ),
            MetalScheduleDecision::DecodeBatch(batch) => {
                execute_decode_batch(batch.req_ids, &mut scheduler, &mut active);
            }
            MetalScheduleDecision::Mixed { decode, prefill } => {
                execute_decode_batch(decode.req_ids, &mut scheduler, &mut active);
                execute_prefill_chunk(
                    prefill.req_id,
                    prefill.input_tokens.len(),
                    &mut scheduler,
                    &mut active,
                );
            }
        }
    }

    Ok(())
}

fn drain_incoming_requests(
    backend: &'static MetalBackend,
    tokenizer: &'static Tokenizer,
    handle: &SchedulerHandle,
    request_rx: &mut mpsc::UnboundedReceiver<IncomingRequest>,
    request_rx_closed: &mut bool,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    loop {
        match request_rx.try_recv() {
            Ok(incoming) => {
                handle.consume_one();
                admit_request(backend, tokenizer, incoming, scheduler, active);
            }
            Err(mpsc::error::TryRecvError::Empty) => break,
            Err(mpsc::error::TryRecvError::Disconnected) => {
                *request_rx_closed = true;
                break;
            }
        }
    }
}

fn admit_request(
    backend: &'static MetalBackend,
    tokenizer: &'static Tokenizer,
    incoming: IncomingRequest,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    if incoming.delta_tx.is_closed() {
        return;
    }

    let priority = map_request_priority(incoming.priority);
    let (prompt_tokens, max_tokens, request) =
        match ActiveMetalRequest::from_incoming(backend, tokenizer, incoming) {
            Ok(request) => request,
            Err(err) => {
                error!("Metal scheduler request init failed: {err:#}");
                return;
            }
        };

    let req_id = match scheduler.submit(prompt_tokens, max_tokens, priority) {
        Ok(req_id) => req_id,
        Err(err) => {
            error!("Metal scheduler submit failed: {err}");
            return;
        }
    };

    if active.insert(req_id, request).is_some() {
        warn!("Metal scheduler request id collision for {:?}", req_id);
    }
}

fn execute_prefill_chunk(
    req_id: RequestId,
    budget: usize,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    enum Outcome {
        Progress {
            emitted_token: Option<u32>,
            runtime_finished: bool,
            stop_hit: bool,
        },
        ClientDropped,
        Failed(anyhow::Error),
    }

    let outcome = {
        let Some(request) = active.get_mut(&req_id) else {
            warn!(
                "Metal prefill chunk referenced missing request {:?}",
                req_id
            );
            scheduler.finish_request(req_id);
            return;
        };

        if request.delta_closed() {
            Outcome::ClientDropped
        } else {
            match request.prefill_chunk(budget) {
                Ok(emitted_token) => Outcome::Progress {
                    emitted_token,
                    runtime_finished: request.phase() == RuntimePhase::Finished,
                    stop_hit: request.stop_hit(),
                },
                Err(err) => {
                    if request.delta_closed() {
                        Outcome::ClientDropped
                    } else {
                        Outcome::Failed(err)
                    }
                }
            }
        }
    };

    match outcome {
        Outcome::Progress {
            emitted_token,
            runtime_finished,
            stop_hit,
        } => {
            let mut scheduler_finished = false;
            if let Some(token) = emitted_token {
                match scheduler.complete_prefill(req_id, token) {
                    Ok(done) => scheduler_finished = done,
                    Err(err) => {
                        error!("Metal complete_prefill failed for {:?}: {err}", req_id);
                        cancel_request(req_id, scheduler, active);
                        return;
                    }
                }
            }

            if runtime_finished || stop_hit || scheduler_finished {
                finalize_request(req_id, scheduler, active);
            }
        }
        Outcome::ClientDropped => cancel_request(req_id, scheduler, active),
        Outcome::Failed(err) => {
            error!("Metal prefill chunk failed for {:?}: {err:#}", req_id);
            cancel_request(req_id, scheduler, active);
        }
    }
}

fn execute_decode_batch(
    req_ids: Vec<RequestId>,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    enum Outcome {
        Progress {
            sampled_token: u32,
            runtime_finished: bool,
            stop_hit: bool,
        },
        ClientDropped,
        Failed(anyhow::Error),
    }

    for req_id in req_ids {
        let outcome = {
            let Some(request) = active.get_mut(&req_id) else {
                warn!("Metal decode batch referenced missing request {:?}", req_id);
                scheduler.finish_request(req_id);
                continue;
            };

            if request.delta_closed() {
                Outcome::ClientDropped
            } else {
                match request.decode_step() {
                    Ok(sampled_token) => Outcome::Progress {
                        sampled_token,
                        runtime_finished: request.phase() == RuntimePhase::Finished,
                        stop_hit: request.stop_hit(),
                    },
                    Err(err) => {
                        if request.delta_closed() {
                            Outcome::ClientDropped
                        } else {
                            Outcome::Failed(err)
                        }
                    }
                }
            }
        };

        match outcome {
            Outcome::Progress {
                sampled_token,
                runtime_finished,
                stop_hit,
            } => {
                let scheduler_finished = match scheduler.advance_decode(req_id, sampled_token) {
                    Ok(done) => done,
                    Err(err) => {
                        error!("Metal advance_decode failed for {:?}: {err}", req_id);
                        cancel_request(req_id, scheduler, active);
                        continue;
                    }
                };

                if runtime_finished || stop_hit || scheduler_finished {
                    finalize_request(req_id, scheduler, active);
                }
            }
            Outcome::ClientDropped => cancel_request(req_id, scheduler, active),
            Outcome::Failed(err) => {
                error!("Metal decode step failed for {:?}: {err:#}", req_id);
                cancel_request(req_id, scheduler, active);
            }
        }
    }
}

fn reap_closed_clients(
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    let closed: Vec<_> = active
        .iter()
        .filter_map(|(req_id, request)| request.delta_closed().then_some(*req_id))
        .collect();

    for req_id in closed {
        cancel_request(req_id, scheduler, active);
    }
}

fn cancel_request(
    req_id: RequestId,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    scheduler.finish_request(req_id);
    if let Some(mut request) = active.remove(&req_id)
        && let Err(err) = request.cancel()
    {
        warn!("Metal request cancel failed for {:?}: {err:#}", req_id);
    }
}

fn finalize_request(
    req_id: RequestId,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    scheduler.finish_request(req_id);
    let Some(mut request) = active.remove(&req_id) else {
        return;
    };
    if let Err(err) = request.cancel() {
        warn!("Metal request cleanup failed for {:?}: {err:#}", req_id);
    }
    if let Err(err) = request.send_final_delta()
        && !request.delta_closed()
    {
        warn!("Metal request final delta failed for {:?}: {err:#}", req_id);
    }
}

fn map_request_priority(priority: RequestPriority) -> MetalRequestPriority {
    match priority {
        RequestPriority::Low => MetalRequestPriority::Low,
        RequestPriority::Normal => MetalRequestPriority::Normal,
        RequestPriority::High => MetalRequestPriority::High,
    }
}

fn map_finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("length") => FinishReason::Length,
        _ => FinishReason::Stop,
    }
}

fn send_text_delta(
    delta_tx: &mpsc::UnboundedSender<CompletionStreamDelta>,
    text_delta: String,
) -> Result<()> {
    if text_delta.is_empty() {
        return Ok(());
    }

    delta_tx
        .send(CompletionStreamDelta {
            text_delta,
            finish_reason: None,
            usage: None,
            logprob: None,
        })
        .map_err(|_| anyhow!("stream consumer dropped"))
}

struct StopChunkProcessor {
    accumulated: String,
    sent_len: usize,
    stops: Vec<String>,
    max_stop_len: usize,
    hit_stop: bool,
}

impl StopChunkProcessor {
    fn new(stops: Vec<String>) -> Self {
        let max_stop_len = stops.iter().map(String::len).max().unwrap_or(0);
        Self {
            accumulated: String::new(),
            sent_len: 0,
            stops,
            max_stop_len,
            hit_stop: false,
        }
    }

    fn push_chunk(&mut self, chunk: &str) -> Option<String> {
        if self.hit_stop {
            return None;
        }

        self.accumulated.push_str(chunk);

        if let Some(stop_pos) = self.find_earliest_stop() {
            let delta = self.accumulated[self.sent_len..stop_pos].to_string();
            self.sent_len = stop_pos;
            self.hit_stop = true;
            return Some(delta);
        }

        if self.max_stop_len <= 1 {
            return self.flush_ready(self.accumulated.len());
        }

        let safe_end = self
            .accumulated
            .len()
            .saturating_sub(self.max_stop_len.saturating_sub(1));
        let safe_end = clamp_char_boundary(&self.accumulated, safe_end);
        self.flush_ready(safe_end)
    }

    fn finish(&mut self) -> Option<String> {
        if self.hit_stop {
            return None;
        }
        self.flush_ready(self.accumulated.len())
    }

    fn hit_stop(&self) -> bool {
        self.hit_stop
    }

    fn flush_ready(&mut self, end: usize) -> Option<String> {
        if end <= self.sent_len {
            return None;
        }
        let delta = self.accumulated[self.sent_len..end].to_string();
        self.sent_len = end;
        Some(delta)
    }

    fn find_earliest_stop(&self) -> Option<usize> {
        let unsent = &self.accumulated[self.sent_len..];
        let mut earliest = None::<usize>;

        for stop in &self.stops {
            if stop.is_empty() {
                continue;
            }
            if let Some(pos) = unsent.find(stop) {
                let absolute = self.sent_len + pos;
                earliest = Some(match earliest {
                    None => absolute,
                    Some(existing) => existing.min(absolute),
                });
            }
        }

        earliest
    }
}

fn clamp_char_boundary(text: &str, mut idx: usize) -> usize {
    idx = idx.min(text.len());
    while idx > 0 && !text.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}
