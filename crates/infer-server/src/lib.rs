//! Frontend serve layer for [`infer_core::Engine`].
//!
//! This crate implements the vLLM-V1 / ideal-architecture **frontend-||-engine**
//! split: CPU frontend work (request ingress, completion fan-out) is decoupled
//! from the engine loop, which owns the [`Engine`] on a dedicated background
//! thread. The frontend talks to the engine thread only through channels, so a
//! caller never blocks the engine while preparing or collecting a request.
//!
//! The API is intentionally `std`-only (threads + `mpsc`), no async runtime:
//!
//! - [`ServeHandle::spawn`] starts the engine thread owning an `Engine<E, K>`.
//! - [`ServeHandle::submit`] hands a prompt to the engine and returns a
//!   [`RequestTicket`] carrying the engine-assigned [`RequestHandle`] plus a
//!   private back-channel for that request's completion.
//! - [`ServeHandle::collect`] / [`RequestTicket::collect`] blocks until the
//!   request finishes and returns its [`CompletedRequest`] (generated tokens,
//!   finish reason).
//!
//! The engine thread is an OS good citizen: it only spins while there is work,
//! and parks on the submit channel (`recv_timeout`) when fully idle instead of
//! busy-looping.

use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::{Result, anyhow};
use infer_core::{CompletedRequest, Engine, RequestHandle, SchedulerConfig};
use infer_plan::{ForwardPlan, SlotToken, StepOutput};
use infer_seam::{BackendExecutor, KvPool, PollResult};

mod openai;
pub use openai::{
    ChatCompletionRequest, ChatCompletionResponse, CompletionRequest, CompletionResponse,
    OpenAiTokenizer, openai_router,
};

#[cfg(feature = "metal")]
pub use openai::metal_openai_router_from_model_path;

/// How long the engine thread parks on the submit channel when fully idle.
///
/// Short enough that a freshly-submitted request is picked up promptly, long
/// enough that an idle engine does not busy-spin and steal CPU from the OS.
const IDLE_PARK: Duration = Duration::from_millis(2);

/// One unit of frontend->engine work: a prompt plus a place to send back the
/// engine-assigned handle and (later) the completion.
struct Submission {
    prompt: Vec<u32>,
    max_tokens: usize,
    /// Carries the engine-assigned handle back to the submitting caller.
    handle_tx: Sender<RequestHandle>,
    /// Carries the request's single completion back to the submitting caller.
    completion_tx: Sender<CompletedRequest>,
}

/// Handle to a running engine thread.
///
/// Owns the engine thread's `JoinHandle` and the submit channel. Dropping the
/// handle closes the submit channel; the engine thread then drains any
/// in-flight work, exits its loop, and joins. The generic parameters mirror
/// [`Engine`]; both must be `Send + 'static` because the engine lives on a
/// separate thread.
pub struct ServeHandle<E: BackendExecutor, K: KvPool> {
    submit_tx: Option<Sender<Submission>>,
    join: Option<JoinHandle<()>>,
    _backend: std::marker::PhantomData<fn() -> (E, K)>,
}

/// Receipt for a submitted request.
///
/// Holds the engine-assigned [`RequestHandle`] and the private back-channel on
/// which that request's [`CompletedRequest`] will arrive. Collect the result
/// via [`RequestTicket::collect`] or [`ServeHandle::collect`].
pub struct RequestTicket {
    handle: RequestHandle,
    completion_rx: Receiver<CompletedRequest>,
}

impl RequestTicket {
    /// Return the engine-assigned handle for this request.
    #[must_use]
    pub fn handle(&self) -> RequestHandle {
        self.handle
    }

    /// Block until this request completes and return its result.
    ///
    /// Returns an error if the engine thread exited before delivering the
    /// completion (e.g. the engine panicked or was shut down).
    pub fn collect(self) -> Result<CompletedRequest> {
        self.completion_rx.recv().map_err(|_| {
            anyhow!(
                "engine thread closed before request {} completed",
                self.handle.id()
            )
        })
    }

    /// Block up to `timeout` for this request to complete.
    ///
    /// On timeout the ticket is returned in the `Err` so the caller can retry.
    pub fn collect_timeout(
        self,
        timeout: Duration,
    ) -> std::result::Result<CompletedRequest, CollectTimeout> {
        match self.completion_rx.recv_timeout(timeout) {
            Ok(completed) => Ok(completed),
            Err(RecvTimeoutError::Timeout) => Err(CollectTimeout::Pending(self)),
            Err(RecvTimeoutError::Disconnected) => Err(CollectTimeout::Closed(self.handle)),
        }
    }
}

/// Outcome of a [`RequestTicket::collect_timeout`] that did not deliver a result.
pub enum CollectTimeout {
    /// The request is still running; the ticket is returned to retry.
    Pending(RequestTicket),
    /// The engine thread closed before delivering the completion.
    Closed(RequestHandle),
}

impl<E, K> ServeHandle<E, K>
where
    E: BackendExecutor + Send + 'static,
    K: KvPool + Send + 'static,
{
    /// Spawn an engine thread owning `Engine::with_config(executor, kv, config)`.
    ///
    /// The returned handle is the only way to reach the engine: submit requests
    /// with [`ServeHandle::submit`] and collect results with
    /// [`ServeHandle::collect`].
    pub fn spawn(executor: E, kv: K, config: SchedulerConfig) -> Self {
        let (submit_tx, submit_rx) = mpsc::channel::<Submission>();
        let join = thread::Builder::new()
            .name("infer-engine".to_string())
            .spawn(move || engine_loop(Engine::with_config(executor, kv, config), submit_rx))
            .expect("spawn infer-engine thread");
        Self {
            submit_tx: Some(submit_tx),
            join: Some(join),
            _backend: std::marker::PhantomData,
        }
    }
}

impl<E, K> ServeHandle<E, K>
where
    E: BackendExecutor + 'static,
    K: KvPool + 'static,
{
    /// Spawn an engine thread and build the engine inside that thread.
    ///
    /// This is for backends whose executor owns thread-affine handles. The
    /// builder itself must be sendable, but the constructed `E` and `K` never
    /// cross a thread boundary; they are created and consumed by the engine
    /// thread in place.
    pub fn spawn_with_engine_builder<B>(builder: B) -> Result<Self>
    where
        B: FnOnce() -> Result<Engine<E, K>> + Send + 'static,
    {
        let (submit_tx, submit_rx) = mpsc::channel::<Submission>();
        let (ready_tx, ready_rx) = mpsc::sync_channel::<std::result::Result<(), String>>(1);
        let join = thread::Builder::new()
            .name("infer-engine".to_string())
            .spawn(move || match builder() {
                Ok(engine) => {
                    let _ = ready_tx.send(Ok(()));
                    engine_loop(engine, submit_rx);
                }
                Err(err) => {
                    let _ = ready_tx.send(Err(err.to_string()));
                }
            })
            .expect("spawn infer-engine thread");

        match ready_rx
            .recv()
            .map_err(|_| anyhow!("engine thread exited before signalling readiness"))?
        {
            Ok(()) => Ok(Self {
                submit_tx: Some(submit_tx),
                join: Some(join),
                _backend: std::marker::PhantomData,
            }),
            Err(err) => {
                let _ = join.join();
                Err(anyhow!("engine build failed: {err}"))
            }
        }
    }

    /// Submit a prompt for generation and return a [`RequestTicket`].
    ///
    /// Blocks only until the engine thread assigns a [`RequestHandle`] (a single
    /// channel round-trip), not until the request finishes. Use the returned
    /// ticket to [`collect`](RequestTicket::collect) the result.
    pub fn submit(&self, prompt: Vec<u32>, max_tokens: usize) -> Result<RequestTicket> {
        let submit_tx = self
            .submit_tx
            .as_ref()
            .ok_or_else(|| anyhow!("ServeHandle already shut down"))?;
        let (handle_tx, handle_rx) = mpsc::channel::<RequestHandle>();
        let (completion_tx, completion_rx) = mpsc::channel::<CompletedRequest>();
        submit_tx
            .send(Submission {
                prompt,
                max_tokens,
                handle_tx,
                completion_tx,
            })
            .map_err(|_| anyhow!("engine thread closed; cannot submit"))?;
        let handle = handle_rx
            .recv()
            .map_err(|_| anyhow!("engine thread closed before assigning a request handle"))?;
        Ok(RequestTicket {
            handle,
            completion_rx,
        })
    }

    /// Block until the request behind `ticket` completes and return its result.
    ///
    /// Convenience wrapper over [`RequestTicket::collect`].
    pub fn collect(&self, ticket: RequestTicket) -> Result<CompletedRequest> {
        ticket.collect()
    }

    /// Close the submit channel and join the engine thread.
    ///
    /// The engine drains its in-flight and waiting work to completion before the
    /// thread exits, so any tickets already submitted still resolve. Called
    /// automatically on drop; exposed so callers can observe a panic in the
    /// engine thread.
    pub fn shutdown(mut self) -> thread::Result<()> {
        self.submit_tx.take();
        match self.join.take() {
            Some(join) => join.join(),
            None => Ok(()),
        }
    }
}

impl<E: BackendExecutor, K: KvPool> Drop for ServeHandle<E, K> {
    fn drop(&mut self) {
        // Close the submit channel so the engine loop can observe shutdown,
        // then join so the engine thread fully drains before we return.
        self.submit_tx.take();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

/// The engine thread body: own the engine, drain submits, step, deliver.
fn engine_loop<E, K>(mut engine: Engine<E, K>, submit_rx: Receiver<Submission>)
where
    E: BackendExecutor,
    K: KvPool,
{
    // Per-request completion back-channels, keyed by the engine-assigned handle.
    // Entries are removed as their completion is delivered.
    let mut pending: std::collections::HashMap<RequestHandle, Sender<CompletedRequest>> =
        std::collections::HashMap::new();
    let mut submit_open = true;

    loop {
        // 1. Drain every queued submission without blocking. Each one is handed
        //    to the engine and registered for completion delivery.
        loop {
            match submit_rx.try_recv() {
                Ok(submission) => admit_submission(&mut engine, &mut pending, submission),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    submit_open = false;
                    break;
                }
            }
        }

        // 2. If there is engine work, run one tick and deliver any completions.
        //    Looping here (rather than one tick per outer pass) keeps latency low
        //    without re-checking the submit channel between every micro-step.
        if !engine.is_idle() {
            if let Err(err) = engine.step() {
                log::error!("infer-server engine step failed: {err}");
                // Drop all back-channels so collectors observe the failure as a
                // closed channel rather than hanging forever.
                pending.clear();
                return;
            }
            deliver_completions(&engine, &mut pending);
            continue;
        }

        // 3. Fully idle. If the frontend is gone and nothing remains, exit.
        if !submit_open {
            // Flush any straggler completions before leaving.
            deliver_completions(&engine, &mut pending);
            return;
        }

        // 4. Idle but still serving: park on the submit channel instead of
        //    busy-spinning, so the engine thread is an OS good citizen.
        match submit_rx.recv_timeout(IDLE_PARK) {
            Ok(submission) => admit_submission(&mut engine, &mut pending, submission),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => submit_open = false,
        }
    }
}

/// Submit one request to the engine and register its back-channels.
fn admit_submission<E, K>(
    engine: &mut Engine<E, K>,
    pending: &mut std::collections::HashMap<RequestHandle, Sender<CompletedRequest>>,
    submission: Submission,
) where
    E: BackendExecutor,
    K: KvPool,
{
    let handle = engine.submit_request(submission.prompt, submission.max_tokens);
    // If the submitter dropped its ticket before we replied, just don't track it.
    let _ = submission.handle_tx.send(handle);
    pending.insert(handle, submission.completion_tx);

    // The engine may complete a request synchronously at ingress (empty/too-long
    // prompt, or zero effective max_tokens). Deliver those immediately so the
    // caller's `collect()` does not block waiting for a step that never runs.
    if let Some(completed) = engine.completed(handle) {
        if let Some(tx) = pending.remove(&handle) {
            let _ = tx.send(completed.clone());
        }
    }
}

/// Deliver any newly-completed requests to their waiting collectors.
fn deliver_completions<E, K>(
    engine: &Engine<E, K>,
    pending: &mut std::collections::HashMap<RequestHandle, Sender<CompletedRequest>>,
) where
    E: BackendExecutor,
    K: KvPool,
{
    if pending.is_empty() {
        return;
    }
    let ready: Vec<RequestHandle> = pending
        .keys()
        .copied()
        .filter(|handle| engine.completed(*handle).is_some())
        .collect();
    for handle in ready {
        if let Some(completed) = engine.completed(handle) {
            if let Some(tx) = pending.remove(&handle) {
                // The collector may have dropped its ticket; ignore send errors.
                let _ = tx.send(completed.clone());
            }
        }
    }
}

/// A tiny in-crate executor that echoes each row's input forward by `+1`.
///
/// Mirrors the engine-core test mock's token rule so the serve layer can be
/// exercised end-to-end without a real backend (no MLX, no CUDA): a decode row
/// emits `last_token + 1`, and a prefill chunk's committed token (on the final
/// chunk) is `last_prompt_token + 1`. Completion is therefore length-bound by
/// `max_tokens`, which makes token-count assertions deterministic.
#[derive(Debug, Clone, Copy, Default)]
pub struct EchoExecutor;

impl BackendExecutor for EchoExecutor {
    type Inflight = StepOutput;

    fn submit(&mut self, plan: &ForwardPlan, _kv: &mut dyn KvPool) -> Result<Self::Inflight> {
        let mut tokens = Vec::with_capacity(plan.prefill_rows.len() + plan.decode_rows.len());
        for row in &plan.prefill_rows {
            let token = row.tokens.last().copied().map_or(1, |last| last + 1);
            tokens.push(SlotToken {
                slot: row.slot,
                token,
                logprob: None,
                finish: None,
            });
        }
        for row in &plan.decode_rows {
            tokens.push(SlotToken {
                slot: row.slot,
                token: row.last_token + 1,
                logprob: None,
                finish: None,
            });
        }
        Ok(StepOutput { tokens })
    }

    fn poll(&mut self, inflight: Self::Inflight) -> Result<PollResult<Self::Inflight>> {
        Ok(PollResult::Ready(inflight))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infer_metal::MetalKvPool;

    /// End-to-end frontend-||-engine smoke test with no GPU/MLX dependency.
    ///
    /// Spawns the engine thread on an `EchoExecutor` + real `MetalKvPool`,
    /// submits two requests, collects both off their private back-channels, and
    /// asserts the generated token counts match each request's `max_tokens`
    /// (the echo executor never emits a stop token, so finish is length-bound).
    #[test]
    fn submit_two_requests_and_collect_both() -> Result<()> {
        let config = SchedulerConfig {
            num_slots: 2,
            max_prompt_tokens: 4096,
            max_total_tokens: 8192,
            ..SchedulerConfig::default()
        };
        // Real host-side pool: 2 slots, ample pages, page_size 16.
        let kv = MetalKvPool::new(2, 256, 16);
        let serve = ServeHandle::spawn(EchoExecutor, kv, config);

        let first = serve.submit(vec![10, 11, 12], 5)?;
        let second = serve.submit(vec![100, 101], 3)?;

        // Handles are distinct and ordered by submission.
        assert_ne!(first.handle(), second.handle());

        let first_done = first.collect()?;
        let second_done = second.collect()?;

        assert_eq!(
            first_done.generated_tokens.len(),
            5,
            "first request should generate exactly max_tokens=5 tokens"
        );
        assert_eq!(
            second_done.generated_tokens.len(),
            3,
            "second request should generate exactly max_tokens=3 tokens"
        );
        // Echo rule: final prefill chunk commits last_prompt + 1, then +1/decode.
        assert_eq!(first_done.generated_tokens, vec![13, 14, 15, 16, 17]);
        assert_eq!(second_done.generated_tokens, vec![102, 103, 104]);

        serve.shutdown().expect("engine thread joins cleanly");
        Ok(())
    }

    /// A request the engine completes synchronously at ingress (prompt too long)
    /// must still deliver its completion to the collector without hanging.
    #[test]
    fn rejected_request_completes_without_hanging() -> Result<()> {
        let config = SchedulerConfig {
            num_slots: 1,
            max_prompt_tokens: 2,
            max_total_tokens: 16,
            ..SchedulerConfig::default()
        };
        let kv = MetalKvPool::new(1, 64, 16);
        let serve = ServeHandle::spawn(EchoExecutor, kv, config);

        let ticket = serve.submit(vec![1, 2, 3], 4)?;
        let done = ticket
            .collect_timeout(Duration::from_secs(5))
            .map_err(|_| anyhow!("rejected request did not complete in time"))?;
        assert!(done.generated_tokens.is_empty());
        assert!(done.finish.is_some());

        serve.shutdown().expect("engine thread joins cleanly");
        Ok(())
    }
}
