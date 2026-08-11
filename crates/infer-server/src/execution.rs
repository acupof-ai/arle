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
use std::time::{Duration, Instant};

use infer_core::{
    CompletedRequest, Engine, KvSystemMetrics, KvTierStats, PrefixCacheStats, RequestHandle,
    RequestOptions, ThroughputStats,
};
use infer_plan::SamplingParams;
use infer_seam::{BackendExecutor, KvPool};

use crate::ServeShutdown;

type TickBroadcast<'a> =
    Option<&'a dyn Fn(u64, Vec<crate::multiproc_relay::WireRequest>) -> anyhow::Result<()>>;

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
#[derive(Debug, Clone, Default)]
pub struct CounterSnapshot {
    pub active_requests: usize,
    pub queue_depth: usize,
    pub kv_free_pages: usize,
    pub prefix_cache: PrefixCacheStats,
    pub throughput: ThroughputStats,
    pub kv_tier: KvTierStats,
    pub kv_system: KvSystemMetrics,
    pub spec_decode: infer_seam::SpecDecodeStats,
    pub op_timing: infer_seam::OpTimingStats,
}

/// Cross-thread handle to the latest [`CounterSnapshot`]: the engine loop writes
/// it each tick, the frontend reads it.
type CounterHandle = Arc<Mutex<CounterSnapshot>>;

fn publish_counters<E: BackendExecutor, K: KvPool>(
    engine: &Engine<E, K>,
    counters: &CounterHandle,
) {
    if let Ok(mut snap) = counters.lock() {
        snap.active_requests = engine.active_count();
        snap.queue_depth = engine.waiting_count();
        snap.kv_free_pages = engine.kv_free_pages();
        snap.prefix_cache = engine.prefix_cache_stats();
        snap.throughput = engine.throughput_stats();
        snap.kv_tier = engine.kv_tier_stats();
        snap.kv_system = engine.kv_system_metrics();
        snap.spec_decode = engine.spec_decode_stats();
        snap.op_timing = engine.op_timing_stats();
    }
}

/// An out-of-band control closure run against the engine-thread-owned `Engine`.
///
/// The OPD control surface (raw-logits forward, weight offload/reload, student
/// LoRA re-merge + prefix-cache invalidation) reaches the thread-owned
/// `&mut Engine<E, K>` through these between steps, off the request hot path. The
/// closure gets the whole engine (scheduler + RadixCache + executor), so a
/// resident-weight re-merge can drop the now-stale prefix cache atomically in one
/// message. The closure carries its own response channel, so the loop body only
/// has to invoke it.
pub(crate) type ControlMessage<E, K> = Box<dyn FnOnce(&mut Engine<E, K>) + Send>;

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
    pub(crate) grammar: Option<infer_core::GrammarHook>,
}

pub(crate) fn engine_loop<E, K>(
    engine: Engine<E, K>,
    submit_rx: Receiver<Submission>,
    control_rx: Receiver<ControlMessage<E, K>>,
    counters: CounterHandle,
    shutdown: ServeShutdown,
) where
    E: BackendExecutor,
    K: KvPool,
{
    let broadcast_tick = |seq, requests| crate::multiproc_relay::broadcast_tick(seq, requests);
    let tick_broadcast: TickBroadcast<'_> = if crate::multiproc_relay::tick_broadcaster_installed()
    {
        Some(&broadcast_tick)
    } else {
        None
    };
    engine_loop_with_tick_broadcaster(
        engine,
        submit_rx,
        control_rx,
        counters,
        shutdown,
        tick_broadcast,
    );
}

fn engine_loop_with_tick_broadcaster<E, K>(
    mut engine: Engine<E, K>,
    submit_rx: Receiver<Submission>,
    control_rx: Receiver<ControlMessage<E, K>>,
    counters: CounterHandle,
    shutdown: ServeShutdown,
    tick_broadcast: TickBroadcast<'_>,
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
    let trace_submit = crate::submit_trace_enabled();

    // Streaming requests whose receiver hung up (client gone): recorded by the
    // observer inside `engine.step()`, cancelled by the loop body after the step
    // (the observer can't touch the engine reentrantly).
    let dropped_streams: Rc<RefCell<Vec<RequestHandle>>> = Rc::default();

    // Forward each committed token to its request's live stream (if any). Runs on
    // the engine thread inside `engine.step()`, so it shares `streamers` via the
    // single-threaded `Rc<RefCell<_>>`. The blocking `pending` path is untouched.
    {
        let streamers = Rc::clone(&streamers);
        let dropped_streams = Rc::clone(&dropped_streams);
        engine.set_token_observer(Box::new(move |handle, token| {
            if let Some(tx) = streamers.borrow().get(&handle)
                && tx
                    .send(StreamItem::Token {
                        token: token.token,
                        logprob: token.logprob,
                    })
                    .is_err()
            {
                dropped_streams.borrow_mut().push(handle);
            }
        }));
    }

    // Multiproc lockstep state (rank-0 coordinator only; resolved once — the
    // coordinator installs the broadcaster at boot, before this engine spawns).
    // `tick_seq` pairs exactly one `TickAdmissions` with every engine step so
    // worker ranks admit the same requests at the same step index; see
    // `multiproc_relay::set_tick_broadcaster` for the desync proof this closes.
    let lockstep = tick_broadcast.is_some();
    let mut tick_seq: u64 = 0;
    let mut next_request_id: u64 = 1;
    // Submission picked up by the idle park below, admitted on the next pass so
    // it flows through the same per-tick broadcast as the drained batch.
    let mut carry: Vec<Submission> = Vec::new();

    loop {
        if shutdown.is_requested() {
            abort_pending(&mut pending, &streamers);
            publish_counters(&engine, &counters);
            return;
        }

        // 0. Snapshot queued submissions before controls run. A Serving→Quiesced
        //    transition aborts this pre-existing batch; later submissions defer.
        let mut drained = std::mem::take(&mut carry);
        loop {
            match submit_rx.try_recv() {
                Ok(submission) => drained.push(submission),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    submit_open = false;
                    break;
                }
            }
        }

        // 1. Run queued out-of-band controls between steps. A closure may cancel
        //    requests; deliver those completions even if the engine stays idle.
        let was_quiesced = engine.is_quiesced();
        if drain_control(&mut engine, &control_rx) > 0 {
            deliver_completions(&engine, &mut pending, &streamers);
        }
        if !was_quiesced && engine.is_quiesced() {
            drained.clear();
        }

        // OPD writeback bracket: the KV pool is released while quiesced, so no
        // submission may be admitted (it would prefill onto the dropped pool) and
        // none may become a waiter (a non-idle engine busy-spins the loop and
        // explodes the lockstep tick). Defer every drained submission into `carry`
        // — admitted verbatim once `resume_serving` clears the flag — and park on
        // the submit channel so `drain_control` (step 0) still observes resume.
        if engine.is_quiesced() {
            carry = drained;
            // Keep telemetry live: `quiesce_serve` polls `active_requests == 0`
            // after `quiesce()` cancelled everything — without this publish the
            // counter stays stale and that poll never clears (60s timeout).
            publish_counters(&engine, &counters);
            // Frontend gone (submit channel closed) → exit like the idle path,
            // so engine teardown never deadlocks on a quiesced engine.
            if !submit_open {
                deliver_completions(&engine, &mut pending, &streamers);
                return;
            }
            match submit_rx.recv_timeout(IDLE_PARK) {
                Ok(submission) => carry.push(submission),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => submit_open = false,
            }
            continue;
        }

        // 2. Lockstep tick barrier: exactly one `TickAdmissions` precedes every
        //    engine step (and every admission batch). Empty-request envelopes on
        //    pure-decode ticks cost one localhost TCP write per worker (~µs)
        //    against a ≥25 ms step. `engine.is_idle()` here is the pre-admission
        //    state; if it is idle AND nothing was drained, no step follows and
        //    no envelope is sent (workers park on recv symmetrically).
        if lockstep && (!drained.is_empty() || !engine.is_idle()) {
            let requests = drained
                .iter()
                .map(|submission| crate::multiproc_relay::WireRequest {
                    request_id: {
                        let id = next_request_id;
                        next_request_id += 1;
                        id
                    },
                    prompt_tokens: submission.prompt.clone(),
                    max_tokens: submission.max_tokens,
                    sampling: submission.sampling.clone(),
                    // Workers mirror rank-0 tokens; the constraint is applied
                    // once, on the rank that owns the matcher.
                    response_format: None,
                })
                .collect();
            if let Some(broadcast_tick) = tick_broadcast
                && let Err(err) = broadcast_tick(tick_seq, requests)
            {
                log::error!(
                    "infer-server multiproc tick #{tick_seq} broadcast failed; \
                     stopping rank-0 engine before local admission/step: {err:#}"
                );
                abort_pending(&mut pending, &streamers);
                publish_counters(&engine, &counters);
                return;
            }
            tick_seq += 1;
        }
        let admitted = drained.len();
        for submission in drained {
            admit_submission(&mut engine, &mut pending, &streamers, submission);
        }
        if trace_submit && admitted != 0 {
            log::info!(
                "[serve-engine] admitted={} active={} waiting={} pending={}",
                admitted,
                engine.active_count(),
                engine.waiting_count(),
                pending.len()
            );
        }

        // 3. If there is engine work, run one tick and deliver any completions.
        //    Looping here (rather than one tick per outer pass) keeps latency low
        //    without re-checking the submit channel between every micro-step.
        if !engine.is_idle() {
            let step_start = trace_submit.then(Instant::now);
            let active_before = engine.active_count();
            let waiting_before = engine.waiting_count();
            let pending_before = pending.len();
            if let Err(err) = engine.step() {
                log::error!("infer-server engine step failed: {err}");
                // Drop all back-channels so collectors observe the failure as a
                // closed channel rather than hanging forever.
                pending.clear();
                return;
            }
            // Cancel requests whose stream receiver hung up mid-decode so the
            // row frees its KV slot instead of decoding to max_tokens. Local
            // lane only: in lockstep mode a cancellation must arrive on every
            // rank via the rank-synchronized `CancelRequest` broadcast, or the
            // scheduler states desync.
            let dropped = std::mem::take(&mut *dropped_streams.borrow_mut());
            if !lockstep {
                for handle in dropped {
                    streamers.borrow_mut().remove(&handle);
                    if engine.cancel_request(handle) {
                        log::info!(
                            "[serve-engine] cancelled request {} (stream receiver dropped)",
                            handle.id()
                        );
                    }
                }
            }
            deliver_completions(&engine, &mut pending, &streamers);
            if let Some(start) = step_start {
                let step_ms = start.elapsed().as_secs_f64() * 1000.0;
                log::info!(
                    "[serve-engine] step_ms={step_ms:.1} active_before={active_before} waiting_before={waiting_before} pending_before={pending_before} active_after={} waiting_after={} pending_after={}",
                    engine.active_count(),
                    engine.waiting_count(),
                    pending.len()
                );
            }
            publish_counters(&engine, &counters);
            continue;
        }

        // Idle pass: the stepping branch above publishes on every tick, so this
        // is the only other place counters can go stale.
        publish_counters(&engine, &counters);

        if !submit_open {
            // Flush any straggler completions before leaving.
            deliver_completions(&engine, &mut pending, &streamers);
            return;
        }

        // 5. Idle but still serving: park on the submit channel instead of
        //    busy-spinning. The submission is carried to the next pass so it
        //    rides the same tick broadcast as try_recv'd ones.
        match submit_rx.recv_timeout(IDLE_PARK) {
            Ok(submission) => carry.push(submission),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => submit_open = false,
        }
    }
}

/// Run every queued control closure against the engine's executor without
/// blocking, returning how many ran. Each closure carries its own response
/// channel, so the loop only invokes it. A disconnected channel is benign (the
/// frontend dropped its `ServeHandle`); the loop's normal shutdown path still runs.
fn drain_control<E, K>(
    engine: &mut Engine<E, K>,
    control_rx: &Receiver<ControlMessage<E, K>>,
) -> usize
where
    E: BackendExecutor,
    K: KvPool,
{
    let mut ran = 0;
    while let Ok(closure) = control_rx.try_recv() {
        closure(engine);
        ran += 1;
    }
    ran
}

fn abort_pending(pending: &mut PendingCompletions, streamers: &Streamers) {
    pending.clear();
    streamers.borrow_mut().clear();
}

fn admit_submission<E, K>(
    engine: &mut Engine<E, K>,
    pending: &mut PendingCompletions,
    streamers: &Streamers,
    submission: Submission,
) where
    E: BackendExecutor,
    K: KvPool,
{
    // Multiproc lockstep: the per-tick `TickAdmissions` broadcast already
    // happened in the engine loop (step 2) before this admission batch, so
    // worker ranks admit these same requests at the same step index.
    let handle = engine.submit_request_with_options(
        submission.prompt,
        submission.max_tokens,
        RequestOptions {
            sampling: submission.sampling,
            grammar: submission.grammar,
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

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use std::time::Duration;

    use anyhow::bail;
    use infer_core::SchedulerConfig;
    use infer_plan::{ForwardPlan, SamplingParams, StepOutput};
    use infer_seam::{HostPagedKvPool, KvPool, PollResult};

    use super::*;

    struct CountingExecutor {
        submits: Arc<AtomicUsize>,
    }

    impl BackendExecutor for CountingExecutor {
        type Inflight = StepOutput;

        fn submit(
            &mut self,
            plan: &ForwardPlan,
            kv: &mut dyn KvPool,
        ) -> anyhow::Result<Self::Inflight> {
            self.submits.fetch_add(1, Ordering::SeqCst);
            let mut inner = crate::EchoExecutor;
            inner.submit(plan, kv)
        }

        fn poll(&mut self, inflight: Self::Inflight) -> anyhow::Result<PollResult<Self::Inflight>> {
            let mut inner = crate::EchoExecutor;
            inner.poll(inflight)
        }
    }

    #[test]
    fn quiesce_transition_aborts_preexisting_submissions() -> anyhow::Result<()> {
        let config = SchedulerConfig {
            num_slots: 1,
            max_prompt_tokens: 128,
            max_total_tokens: 256,
            ..SchedulerConfig::default()
        };
        let engine =
            Engine::with_config(crate::EchoExecutor, HostPagedKvPool::new(1, 16, 16), config);
        let (submit_tx, submit_rx) = mpsc::channel();
        let (control_tx, control_rx) =
            mpsc::channel::<ControlMessage<crate::EchoExecutor, HostPagedKvPool>>();
        let counters = Arc::new(Mutex::new(CounterSnapshot::default()));
        let shutdown = ServeShutdown::new();
        let (handle_tx, handle_rx) = mpsc::channel();
        let (completion_tx, _completion_rx) = mpsc::channel();
        submit_tx.send(Submission {
            prompt: vec![10, 11],
            max_tokens: 1,
            sampling: SamplingParams::default(),
            handle_tx,
            completion_tx,
            stream_tx: None,
            grammar: None,
        })?;
        control_tx
            .send(Box::new(|engine| {
                engine.quiesce();
            }))
            .map_err(|_| anyhow::anyhow!("control receiver closed"))?;
        drop(submit_tx);
        drop(control_tx);

        engine_loop_with_tick_broadcaster(engine, submit_rx, control_rx, counters, shutdown, None);

        assert!(
            handle_rx.recv_timeout(Duration::from_secs(1)).is_err(),
            "submission queued before quiesce must not survive cancel-all"
        );
        Ok(())
    }

    #[test]
    fn broadcast_failure_stops_before_local_admit_or_step() -> anyhow::Result<()> {
        let submits = Arc::new(AtomicUsize::new(0));
        let executor = CountingExecutor {
            submits: Arc::clone(&submits),
        };
        let config = SchedulerConfig {
            num_slots: 1,
            max_prompt_tokens: 128,
            max_total_tokens: 256,
            ..SchedulerConfig::default()
        };
        let engine = Engine::with_config(executor, HostPagedKvPool::new(1, 16, 16), config);
        let (submit_tx, submit_rx) = mpsc::channel();
        let (_control_tx, control_rx) = mpsc::channel();
        let counters = Arc::new(Mutex::new(CounterSnapshot::default()));
        let shutdown = ServeShutdown::new();

        let (handle_tx, handle_rx) = mpsc::channel();
        let (completion_tx, _completion_rx) = mpsc::channel();
        submit_tx.send(Submission {
            prompt: vec![10, 11],
            max_tokens: 1,
            sampling: SamplingParams::default(),
            handle_tx,
            completion_tx,
            stream_tx: None,
            grammar: None,
        })?;
        drop(submit_tx);

        let failing_broadcast = |seq: u64, requests: Vec<crate::multiproc_relay::WireRequest>| {
            assert_eq!(seq, 0);
            assert_eq!(requests.len(), 1);
            bail!("injected broadcast failure")
        };
        engine_loop_with_tick_broadcaster(
            engine,
            submit_rx,
            control_rx,
            counters,
            shutdown,
            Some(&failing_broadcast),
        );

        assert!(
            handle_rx.recv_timeout(Duration::from_secs(1)).is_err(),
            "rank-0 must not locally admit after a failed tick broadcast"
        );
        assert_eq!(
            submits.load(Ordering::SeqCst),
            0,
            "rank-0 must not step locally after a failed tick broadcast"
        );
        Ok(())
    }
}
