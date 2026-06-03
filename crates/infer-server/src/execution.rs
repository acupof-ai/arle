//! The engine thread body: own the [`Engine`], drain submits, step, deliver.
//!
//! This is the frontend-||-engine hot loop. [`crate::ServeHandle`] spawns a
//! thread running [`engine_loop`], which owns the [`Engine`] and talks to the
//! frontend only through the submit channel ([`Submission`]) and the per-request
//! completion back-channels. The loop only spins while there is work and parks
//! on the submit channel (`recv_timeout`) when fully idle.

use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::time::Duration;

use infer_core::{CompletedRequest, Engine, RequestHandle};
use infer_seam::{BackendExecutor, KvPool};

/// How long the engine thread parks on the submit channel when fully idle.
///
/// Short enough that a freshly-submitted request is picked up promptly, long
/// enough that an idle engine does not busy-spin and steal CPU from the OS.
const IDLE_PARK: Duration = Duration::from_millis(2);

/// One unit of frontend->engine work: a prompt plus a place to send back the
/// engine-assigned handle and (later) the completion.
pub(crate) struct Submission {
    pub(crate) prompt: Vec<u32>,
    pub(crate) max_tokens: usize,
    /// Carries the engine-assigned handle back to the submitting caller.
    pub(crate) handle_tx: Sender<RequestHandle>,
    /// Carries the request's single completion back to the submitting caller.
    pub(crate) completion_tx: Sender<CompletedRequest>,
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
