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
pub(crate) fn engine_loop<E, K>(mut engine: Engine<E, K>, submit_rx: Receiver<Submission>)
where
    E: BackendExecutor,
    K: KvPool,
{
    // Per-request completion back-channels, keyed by the engine-assigned handle.
    // Entries are removed as their completion is delivered.
    let mut pending: std::collections::HashMap<RequestHandle, Sender<CompletedRequest>> =
        std::collections::HashMap::new();
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

/// Submit one request to the engine and register its back-channels.
fn admit_submission<E, K>(
    engine: &mut Engine<E, K>,
    pending: &mut std::collections::HashMap<RequestHandle, Sender<CompletedRequest>>,
    streamers: &Streamers,
    submission: Submission,
) where
    E: BackendExecutor,
    K: KvPool,
{
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
    pending: &mut std::collections::HashMap<RequestHandle, Sender<CompletedRequest>>,
    streamers: &Streamers,
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
            finish_handle(handle, completed.clone(), pending, streamers);
        }
    }
}

/// Deliver one request's completion: send it to the blocking collector and, if
/// the request was streaming, emit the terminal `Done`. Both channels are then
/// dropped (collector + streamer may have already hung up; ignore send errors).
fn finish_handle(
    handle: RequestHandle,
    completed: CompletedRequest,
    pending: &mut std::collections::HashMap<RequestHandle, Sender<CompletedRequest>>,
    streamers: &Streamers,
) {
    if let Some(stream_tx) = streamers.borrow_mut().remove(&handle) {
        let _ = stream_tx.send(StreamItem::Done(completed.clone()));
    }
    if let Some(tx) = pending.remove(&handle) {
        let _ = tx.send(completed);
    }
}
