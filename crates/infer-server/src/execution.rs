//! The engine thread body: own the [`Engine`], drain submits, step, deliver.
//!
//! This is the frontend-||-engine hot loop. [`crate::ServeHandle`] spawns a
//! thread running [`engine_loop`], which owns the [`Engine`] and talks to the
//! frontend only through the submit channel ([`Submission`]) and the per-request
//! completion back-channels. The loop only spins while there is work and parks
//! on the submit channel (`recv_timeout`) when fully idle.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use infer_core::{CompletedRequest, Engine, RequestHandle, RequestOptions};
use infer_plan::SamplingParams;
use infer_seam::{BackendExecutor, KvPool};

/// How long the engine thread parks on the submit channel when fully idle.
///
/// Short enough that a freshly-submitted request is picked up promptly, long
/// enough that an idle engine does not busy-spin and steal CPU from the OS.
const IDLE_PARK: Duration = Duration::from_millis(2);

/// One live event on a request's streaming back-channel: each token as it
/// commits, then exactly one terminal `Done` carrying the same completion the
/// blocking `collect()` path delivers. Streaming is additive; blocking is intact.
pub enum StreamItem {
    Token { token: u32, logprob: Option<f32> },
    Done(CompletedRequest),
}

/// Per-handle streaming senders, shared between the token observer (which runs
/// inside `engine.step()`) and the loop body. The engine loop is single-threaded,
/// so `Rc<RefCell<_>>` is sound; every borrow is short and never spans a step.
type Streamers = Rc<RefCell<HashMap<RequestHandle, Sender<StreamItem>>>>;
type PendingCompletions = HashMap<RequestHandle, Sender<CompletedRequest>>;

/// Live scheduler counters the engine loop publishes each tick for the frontend
/// (`ServeHandle::counters`). Only counters the engine already tracks — no
/// fabricated latency metrics.
#[derive(Debug, Clone, Copy, Default)]
pub struct CounterSnapshot {
    pub active_requests: usize,
    pub queue_depth: usize,
    pub kv_free_pages: usize,
}

/// Cross-thread handle to the latest [`CounterSnapshot`]: the engine loop writes
/// it each tick, the frontend reads it.
type CounterHandle = Arc<Mutex<CounterSnapshot>>;

/// Publish the engine's current counters to the shared snapshot.
fn publish_counters<E: BackendExecutor, K: KvPool>(
    engine: &Engine<E, K>,
    counters: &CounterHandle,
) {
    if let Ok(mut snap) = counters.lock() {
        snap.active_requests = engine.active_count();
        snap.queue_depth = engine.waiting_count();
        snap.kv_free_pages = engine.kv_free_pages();
    }
}

/// An out-of-band control closure run against the engine-thread-owned executor.
///
/// The OPD control surface (raw-logits forward, weight offload/reload, student
/// LoRA re-merge) reaches the thread-owned `&mut E` through these between steps,
/// off the request hot path. The closure carries its own response channel, so the
/// loop body only has to invoke it.
pub(crate) type ControlMessage<E> = Box<dyn FnOnce(&mut E) + Send>;

/// One unit of frontend->engine work: a prompt plus a place to send back the
/// engine-assigned handle and (later) the completion.
pub(crate) struct Submission {
    pub(crate) prompt: Vec<u32>,
    pub(crate) max_tokens: usize,
    /// Per-request sampling/stop parameters parsed at the HTTP boundary.
    pub(crate) sampling: SamplingParams,
    /// Carries the engine-assigned handle back to the submitting caller.
    pub(crate) handle_tx: Sender<RequestHandle>,
    /// Carries the request's single completion back to the submitting caller.
    pub(crate) completion_tx: Sender<CompletedRequest>,
    /// Live token stream for this request: `None` for blocking `submit`, `Some`
    /// for `submit_streaming`.
    pub(crate) stream_tx: Option<Sender<StreamItem>>,
}

/// The engine thread body: own the engine, drain submits, step, deliver.
pub(crate) fn engine_loop<E, K>(
    mut engine: Engine<E, K>,
    submit_rx: Receiver<Submission>,
    control_rx: Receiver<ControlMessage<E>>,
    counters: CounterHandle,
) where
    E: BackendExecutor,
    K: KvPool,
{
    // Per-request completion back-channels, keyed by the engine-assigned handle.
    // Entries are removed as their completion is delivered.
    let mut pending: PendingCompletions = std::collections::HashMap::new();
    // Per-request live token streams (streaming submissions only), shared with the
    // observer installed below; entries are removed when their `Done` is emitted.
    let streamers: Streamers = Rc::new(RefCell::new(HashMap::new()));
    let mut submit_open = true;

    // Forward each committed token to its request's live stream (if any). Runs on
    // the engine thread inside `engine.step()`, so it shares `streamers` via the
    // single-threaded `Rc<RefCell<_>>`. The blocking `pending` path is untouched.
    {
        let streamers = Rc::clone(&streamers);
        engine.set_token_observer(Box::new(move |handle, token| {
            if let Some(tx) = streamers.borrow().get(&handle) {
                let _ = tx.send(StreamItem::Token {
                    token: token.token,
                    logprob: token.logprob,
                });
            }
        }));
    }

    loop {
        // 0. Run any queued out-of-band control closures against the executor
        //    (OPD raw-logits forward, weight offload/reload, LoRA re-merge). These
        //    run between steps with no request in flight, so `&mut E` access is
        //    exclusive. A disconnected control channel is benign (frontend gone).
        drain_control(&mut engine, &control_rx);

        // 1. Drain every queued submission without blocking. Each one is handed
        //    to the engine and registered for completion delivery.
        loop {
            match submit_rx.try_recv() {
                Ok(submission) => {
                    admit_submission(&mut engine, &mut pending, &streamers, submission)
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    submit_open = false;
                    break;
                }
            }
        }
        publish_counters(&engine, &counters);

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
            deliver_completions(&engine, &mut pending, &streamers);
            publish_counters(&engine, &counters);
            continue;
        }

        // 3. Fully idle. If the frontend is gone and nothing remains, exit.
        if !submit_open {
            // Flush any straggler completions before leaving.
            deliver_completions(&engine, &mut pending, &streamers);
            return;
        }

        // 4. Idle but still serving: park on the submit channel instead of
        //    busy-spinning, so the engine thread is an OS good citizen.
        match submit_rx.recv_timeout(IDLE_PARK) {
            Ok(submission) => admit_submission(&mut engine, &mut pending, &streamers, submission),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => submit_open = false,
        }
    }
}

/// Run every queued control closure against the engine's executor without
/// blocking. Each closure carries its own response channel, so the loop only
/// invokes it. A disconnected channel is benign (the frontend dropped its
/// `ServeHandle`); the loop's normal shutdown path still runs.
fn drain_control<E, K>(engine: &mut Engine<E, K>, control_rx: &Receiver<ControlMessage<E>>)
where
    E: BackendExecutor,
    K: KvPool,
{
    while let Ok(closure) = control_rx.try_recv() {
        closure(engine.executor_mut());
    }
}

/// Submit one request to the engine and register its back-channels.
fn admit_submission<E, K>(
    engine: &mut Engine<E, K>,
    pending: &mut PendingCompletions,
    streamers: &Streamers,
    submission: Submission,
) where
    E: BackendExecutor,
    K: KvPool,
{
    // Multiproc lockstep (Stage 2): broadcast this request to worker ranks 1..N-1
    // BEFORE submitting it to the local rank-0 Engine, mirroring the old
    // `DistributedSchedulerGroup::submit` ordering (`e81b98fb^:infer/src/
    // request_handle.rs:909` broadcast `Request2 { wire }` ahead of `permit.submit`).
    // This site is the single deterministic admission point on the engine thread,
    // so every rank admits the same requests in the same FIFO order and their
    // deterministic planners build identical per-step batches. On single-process
    // serves no broadcaster is installed, so this is a cheap no-op load.
    crate::broadcast_admission(
        &submission.prompt,
        submission.max_tokens,
        &submission.sampling,
    );
    let handle = engine.submit_request_with_options(
        submission.prompt,
        submission.max_tokens,
        RequestOptions {
            sampling: submission.sampling,
            ..RequestOptions::default()
        },
    );
    // If the submitter dropped its ticket before we replied, just don't track it.
    let _ = submission.handle_tx.send(handle);
    pending.insert(handle, submission.completion_tx);
    if let Some(stream_tx) = submission.stream_tx {
        streamers.borrow_mut().insert(handle, stream_tx);
    }

    // The engine may complete a request synchronously at ingress (empty/too-long
    // prompt, or zero effective max_tokens). Deliver those immediately so the
    // caller's `collect()` does not block waiting for a step that never runs.
    if let Some(completed) = engine.completed(handle) {
        finish_handle(handle, completed.clone(), pending, streamers);
    }
}

/// Deliver any newly-completed requests to their waiting collectors.
fn deliver_completions<E, K>(
    engine: &Engine<E, K>,
    pending: &mut PendingCompletions,
    streamers: &Streamers,
) where
    E: BackendExecutor,
    K: KvPool,
{
    if pending.is_empty() {
        return;
    }
    let ready: Vec<(RequestHandle, CompletedRequest)> = pending
        .keys()
        .copied()
        .filter_map(|handle| engine.completed(handle).map(|c| (handle, c.clone())))
        .collect();
    for (handle, completed) in ready {
        finish_handle(handle, completed, pending, streamers);
    }
}

/// Deliver one request's completion: send it to the blocking collector and, if
/// the request was streaming, emit the terminal `Done`. Both channels are then
/// dropped (collector + streamer may have already hung up; ignore send errors).
fn finish_handle(
    handle: RequestHandle,
    completed: CompletedRequest,
    pending: &mut PendingCompletions,
    streamers: &Streamers,
) {
    if let Some(stream_tx) = streamers.borrow_mut().remove(&handle) {
        let _ = stream_tx.send(StreamItem::Done(completed.clone()));
    }
    if let Some(tx) = pending.remove(&handle) {
        let _ = tx.send(completed);
    }
}
