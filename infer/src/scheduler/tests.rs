use super::*;
use crate::events::{EngineEvent, EventSink};
use crate::sampler::SamplingParams;
use crate::server_engine::CompletionStreamDelta;
use crate::types::{InferenceMode, RequestEventKind, RequestId};
use std::sync::{Arc, Mutex};

fn make_request() -> IncomingRequest {
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<CompletionStreamDelta>();
    IncomingRequest {
        prompt: "hello".to_string(),
        prompt_tokens: None,
        max_tokens: 32,
        sampling: SamplingParams::default(),
        stop: None,
        speculative: None,
        priority: RequestPriority::Normal,
        session_id: None,
        delta_tx: tx,
        trace_context: None,
    }
}

#[test]
fn scheduler_config_default_valid() {
    SchedulerConfig::default().validate().unwrap();
}

#[test]
fn scheduler_config_zero_slots_invalid() {
    let cfg = SchedulerConfig {
        max_slots: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn scheduler_config_zero_chunk_invalid() {
    let cfg = SchedulerConfig {
        chunked_prefill_size: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn priority_ordering() {
    assert!(RequestPriority::High > RequestPriority::Normal);
    assert!(RequestPriority::Normal > RequestPriority::Low);
}

#[test]
fn priority_default_is_normal() {
    assert_eq!(RequestPriority::default(), RequestPriority::Normal);
}

#[tokio::test]
async fn submit_succeeds_when_queue_not_full() {
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    let handle = SchedulerHandle::with_max_waiting(tx, "test", 3);
    assert!(handle.submit(make_request()).is_ok());
    assert!(handle.submit(make_request()).is_ok());
    assert!(handle.submit(make_request()).is_ok());
    assert_eq!(handle.waiting_count(), 3);
}

#[tokio::test]
async fn submit_fails_when_queue_full() {
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    let handle = SchedulerHandle::with_max_waiting(tx, "test", 2);
    assert!(handle.submit(make_request()).is_ok());
    assert!(handle.submit(make_request()).is_ok());
    assert!(handle.submit(make_request()).is_err());
    assert!(handle.is_full());
}

#[tokio::test]
async fn consume_decrements_waiting_count() {
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    let handle = SchedulerHandle::with_max_waiting(tx, "test", 5);
    handle.submit(make_request()).unwrap();
    handle.submit(make_request()).unwrap();
    assert_eq!(handle.waiting_count(), 2);
    handle.consume_one();
    assert_eq!(handle.waiting_count(), 1);
    handle.submit(make_request()).unwrap();
    handle.submit(make_request()).unwrap();
    handle.submit(make_request()).unwrap();
    handle.submit(make_request()).unwrap();
    assert_eq!(handle.waiting_count(), 5);
}

#[tokio::test]
async fn unlimited_queue_never_rejects() {
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    let handle = SchedulerHandle::from_parts(tx, "test");
    for _ in 0..100 {
        assert!(handle.submit(make_request()).is_ok());
    }
    assert_eq!(handle.waiting_count(), 100);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_submit_does_not_oversubscribe_waiting_capacity() {
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    let handle = SchedulerHandle::with_max_waiting(tx, "test", 2);
    let barrier = Arc::new(tokio::sync::Barrier::new(17));
    let mut tasks = Vec::new();

    for _ in 0..16 {
        let handle = handle.clone();
        let barrier = Arc::clone(&barrier);
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            handle.submit(make_request()).is_ok()
        }));
    }

    barrier.wait().await;

    let mut successes = 0usize;
    for task in tasks {
        if task.await.expect("task join") {
            successes += 1;
        }
    }

    assert_eq!(successes, 2);
    assert_eq!(handle.waiting_count(), 2);
    assert!(handle.is_full());
}

fn make_batch_scheduler(
    num_gpu_blocks: usize,
    block_size: usize,
    chunk_size: usize,
) -> BatchScheduler {
    make_batch_scheduler_with_event_sink(
        num_gpu_blocks,
        block_size,
        chunk_size,
        Arc::new(RecordingEventSink::default()),
    )
}

fn make_batch_scheduler_with_event_sink(
    num_gpu_blocks: usize,
    block_size: usize,
    chunk_size: usize,
    event_sink: Arc<dyn EventSink>,
) -> BatchScheduler {
    let config = BatchSchedulerConfig {
        max_tokens_per_step: 4096,
        prefill_chunk_size: chunk_size,
        ..Default::default()
    };
    let bm = crate::block_manager::BlockManager::new(num_gpu_blocks, 0, block_size);
    BatchScheduler::with_event_sink(config, bm, event_sink)
}

/// Run one simulated step: process the logical plan without a real model.
/// Returns (num_prefilled_reqs, num_decoded_reqs).
fn sim_step(sched: &mut BatchScheduler) -> (usize, usize) {
    let plan = sched.schedule_step();
    let n_prefill = plan.prefill_rows.len();
    let to_finish: Vec<RequestId> = plan.decode_rows.iter().map(|row| row.req_id).collect();
    for id in &to_finish {
        sched.advance_decode(*id);
    }
    (n_prefill, to_finish.len())
}

fn req_ids(rows: &[LogicalDecodeRow]) -> Vec<RequestId> {
    rows.iter().map(|row| row.req_id).collect()
}

fn prefill_req_ids(rows: &[LogicalPrefillRow]) -> Vec<RequestId> {
    rows.iter().map(|row| row.req_id).collect()
}

fn expect_prefill_only(plan: LogicalServePlan) -> LogicalPrefillRow {
    assert!(
        plan.decode_rows.is_empty(),
        "expected no decode rows: {plan:?}"
    );
    assert_eq!(
        plan.prefill_rows.len(),
        1,
        "expected one prefill row: {plan:?}"
    );
    plan.prefill_rows.into_iter().next().unwrap()
}

fn expect_decode_only(plan: LogicalServePlan) -> Vec<LogicalDecodeRow> {
    assert!(
        plan.prefill_rows.is_empty(),
        "expected no prefill rows: {plan:?}"
    );
    assert!(
        !plan.decode_rows.is_empty(),
        "expected decode rows: {plan:?}"
    );
    plan.decode_rows
}

fn expect_mixed(plan: LogicalServePlan) -> (Vec<LogicalDecodeRow>, LogicalPrefillRow) {
    assert!(
        !plan.decode_rows.is_empty(),
        "expected decode rows: {plan:?}"
    );
    assert_eq!(
        plan.prefill_rows.len(),
        1,
        "expected one prefill row: {plan:?}"
    );
    let prefill = plan.prefill_rows.into_iter().next().unwrap();
    (plan.decode_rows, prefill)
}

#[test]
fn test_continuous_batching() {
    let mut sched = make_batch_scheduler(32, 4, 64);

    let id0 = sched.add_request(vec![1, 2, 3, 4], 8, RequestPriority::Normal);
    let id1 = sched.add_request(vec![5, 6, 7, 8], 8, RequestPriority::Normal);
    let _id2 = sched.add_request(vec![9, 10, 11, 12], 8, RequestPriority::Normal);

    let prefill = expect_prefill_only(sched.schedule_step());
    assert_eq!(prefill.req_id, id0);
    assert_eq!(prefill.input_tokens, vec![1, 2, 3, 4]);
    assert!(sched.is_running(id0));
    assert_eq!(sched.waiting_len(), 2);

    let plan = sched.schedule_step();
    if plan.decode_rows.is_empty() {
        let prefill = expect_prefill_only(plan);
        assert_eq!(prefill.req_id, id1);
    } else {
        let (decode, prefill) = expect_mixed(plan);
        assert_eq!(req_ids(&decode), vec![id0]);
        assert_eq!(prefill.req_id, id1);
        sched.advance_decode(id0);
    }

    sched.finish_request(id0);
    assert!(!sched.is_running(id0));

    for _ in 0..20 {
        sim_step(&mut sched);
        let running_ids: Vec<RequestId> = sched.running.keys().copied().collect();
        for id in running_ids {
            sched.finish_request(id);
        }
    }

    assert_eq!(sched.running_len(), 0);
    assert_eq!(sched.waiting_len(), 0);
}

#[test]
fn test_preemption() {
    let mut sched = make_batch_scheduler(5, 4, 16);

    let id0 = sched.add_request(vec![1, 2, 3, 4], 16, RequestPriority::Normal);
    let id1 = sched.add_request(vec![5, 6, 7, 8], 16, RequestPriority::Normal);
    let id2 = sched.add_request(vec![9, 10, 11, 12], 16, RequestPriority::Normal);

    assert_eq!(expect_prefill_only(sched.schedule_step()).req_id, id0);
    let plan = sched.schedule_step();
    if plan.decode_rows.is_empty() {
        assert_eq!(expect_prefill_only(plan).req_id, id1);
    } else {
        let (decode, prefill) = expect_mixed(plan);
        assert_eq!(req_ids(&decode), vec![id0]);
        assert_eq!(prefill.req_id, id1);
        sched.advance_decode(id0);
    }
    let plan = sched.schedule_step();
    if !plan.prefill_rows.is_empty() {
        assert_eq!(prefill_req_ids(&plan.prefill_rows), vec![id2]);
    }

    let free_before = sched.free_gpu_blocks();
    let plan = sched.schedule_step();
    if !plan.decode_rows.is_empty() {
        let decode_ids = req_ids(&plan.decode_rows);
        assert!(
            !decode_ids.contains(&id2),
            "id2 should be preempted, not decoding"
        );
        assert!(decode_ids.contains(&id0) || decode_ids.contains(&id1));
    }

    assert!(
        sched.is_waiting(id2) || !sched.is_running(id2),
        "id2 should have been preempted back to waiting"
    );
    assert!(
        sched.free_gpu_blocks() >= free_before || !sched.is_running(id2),
        "preemption should free blocks"
    );
}

#[test]
fn test_chunked_prefill() {
    let chunk_size = 8usize;
    let mut sched = make_batch_scheduler(8, 4, chunk_size);

    let prompt: Vec<u32> = (1u32..=20).collect();
    let id = sched.add_request(prompt, 4, RequestPriority::Normal);

    let prefill = expect_prefill_only(sched.schedule_step());
    assert_eq!(prefill.req_id, id);
    assert_eq!(prefill.input_tokens.len(), chunk_size);
    assert_eq!(prefill.input_tokens[0], 1);
    assert_eq!(prefill.input_tokens[chunk_size - 1], chunk_size as u32);
    assert_eq!(prefill.prompt_end, chunk_size);
    assert!(!sched.is_running(id));
    assert_eq!(sched.waiting_len(), 1);

    let prefill = expect_prefill_only(sched.schedule_step());
    assert_eq!(prefill.req_id, id);
    assert_eq!(prefill.input_tokens.len(), chunk_size);
    assert_eq!(prefill.input_tokens[0], chunk_size as u32 + 1);
    assert_eq!(prefill.prompt_end, chunk_size * 2);
    assert!(!sched.is_running(id));

    let prefill = expect_prefill_only(sched.schedule_step());
    assert_eq!(prefill.req_id, id);
    assert_eq!(prefill.input_tokens.len(), 4);
    assert_eq!(prefill.input_tokens[0], 17);
    assert_eq!(prefill.prompt_end, 20);
    assert!(sched.is_running(id));
    assert_eq!(sched.waiting_len(), 0);
    assert_eq!(sched.free_gpu_blocks(), 8 - 5);
}

#[test]
fn batch_scheduler_emits_prefill_started_once_for_chunked_request() {
    let sink = Arc::new(RecordingEventSink::default());
    let mut sched = make_batch_scheduler_with_event_sink(8, 4, 8, sink.clone());
    let id = sched.add_request((1u32..=20).collect(), 4, RequestPriority::Normal);

    let prefill = expect_prefill_only(sched.schedule_step());
    assert_eq!(prefill.req_id, id);
    assert_eq!(prefill.input_tokens.len(), 8);
    assert_eq!(
        recorded_events(sink.as_ref()),
        vec![
            EngineEvent {
                request_id: id,
                kind: RequestEventKind::Enqueued,
                mode: None,
            },
            EngineEvent {
                request_id: id,
                kind: RequestEventKind::PrefillStarted,
                mode: Some(InferenceMode::Prefill),
            },
        ]
    );

    let prefill = expect_prefill_only(sched.schedule_step());
    assert_eq!(prefill.req_id, id);
    assert_eq!(prefill.input_tokens.len(), 8);
    assert_eq!(
        recorded_events(sink.as_ref()),
        vec![
            EngineEvent {
                request_id: id,
                kind: RequestEventKind::Enqueued,
                mode: None,
            },
            EngineEvent {
                request_id: id,
                kind: RequestEventKind::PrefillStarted,
                mode: Some(InferenceMode::Prefill),
            },
        ]
    );

    let prefill = expect_prefill_only(sched.schedule_step());
    assert_eq!(prefill.req_id, id);
    assert_eq!(prefill.input_tokens.len(), 4);
    assert_eq!(
        recorded_events(sink.as_ref()),
        vec![
            EngineEvent {
                request_id: id,
                kind: RequestEventKind::Enqueued,
                mode: None,
            },
            EngineEvent {
                request_id: id,
                kind: RequestEventKind::PrefillStarted,
                mode: Some(InferenceMode::Prefill),
            },
        ]
    );

    let decode = expect_decode_only(sched.schedule_step());
    assert_eq!(req_ids(&decode), vec![id]);
    assert!(sched.advance_decode(id));
    assert_eq!(
        recorded_events(sink.as_ref()),
        vec![
            EngineEvent {
                request_id: id,
                kind: RequestEventKind::Enqueued,
                mode: None,
            },
            EngineEvent {
                request_id: id,
                kind: RequestEventKind::PrefillStarted,
                mode: Some(InferenceMode::Prefill),
            },
            EngineEvent {
                request_id: id,
                kind: RequestEventKind::DecodeStep,
                mode: Some(InferenceMode::Decode),
            },
        ]
    );
}

#[test]
fn batch_scheduler_cpu_policy_uses_decode_aware_chunking() {
    let mut sched = make_batch_scheduler(512, 4, 512);
    let req0 = sched.add_request((1..=32).collect(), 8, RequestPriority::Normal);
    let _req1 = sched.add_request((33..=512).collect(), 8, RequestPriority::Normal);

    // Step 1: prefill req0, it becomes running for decode.
    let _ = sched.schedule_step();

    // Step 2: the backend-agnostic CPU accounting scheduler still uses its own
    // decode-aware chunking policy. This is not the production CUDA runtime
    // contract, which now uses explicit token/request budgets.
    let (_decode, prefill) = expect_mixed(sched.schedule_step());
    assert_eq!(prefill.req_id, RequestId(req0.0 + 1));
    assert_eq!(prefill.input_tokens.len(), 64);
}

#[derive(Default)]
struct RecordingEventSink {
    events: Mutex<Vec<EngineEvent>>,
}

impl EventSink for RecordingEventSink {
    fn emit(&self, event: &EngineEvent) {
        self.events.lock().expect("poisoned").push(event.clone());
    }
}

fn recorded_events(sink: &RecordingEventSink) -> Vec<EngineEvent> {
    sink.events.lock().expect("poisoned").clone()
}

#[test]
fn batch_scheduler_emits_lifecycle_events_for_successful_request() {
    let sink = Arc::new(RecordingEventSink::default());
    let mut sched = make_batch_scheduler_with_event_sink(8, 4, 8, sink.clone());
    let req = sched.add_request(vec![1, 2, 3, 4], 8, RequestPriority::Normal);

    let prefill = expect_prefill_only(sched.schedule_step());
    assert_eq!(prefill.req_id, req);
    assert_eq!(prefill.input_tokens, vec![1, 2, 3, 4]);

    let decode = expect_decode_only(sched.schedule_step());
    assert_eq!(req_ids(&decode), vec![req]);
    assert!(sched.advance_decode(req));
    sched.finish_request(req);

    assert_eq!(
        recorded_events(sink.as_ref()),
        vec![
            EngineEvent {
                request_id: req,
                kind: RequestEventKind::Enqueued,
                mode: None,
            },
            EngineEvent {
                request_id: req,
                kind: RequestEventKind::PrefillStarted,
                mode: Some(InferenceMode::Prefill),
            },
            EngineEvent {
                request_id: req,
                kind: RequestEventKind::DecodeStep,
                mode: Some(InferenceMode::Decode),
            },
            EngineEvent {
                request_id: req,
                kind: RequestEventKind::Completed,
                mode: Some(InferenceMode::Decode),
            },
        ]
    );
}

#[test]
fn batch_scheduler_does_not_emit_prefill_event_when_admission_fails() {
    let sink = Arc::new(RecordingEventSink::default());
    let mut sched = make_batch_scheduler_with_event_sink(0, 4, 8, sink.clone());
    let req = sched.add_request(vec![1, 2, 3, 4], 8, RequestPriority::Normal);

    assert!(sched.schedule_step().is_idle());
    assert_eq!(
        recorded_events(sink.as_ref()),
        vec![EngineEvent {
            request_id: req,
            kind: RequestEventKind::Enqueued,
            mode: None,
        }]
    );
}

#[test]
fn batch_scheduler_emits_evicted_and_requeued_when_request_is_preempted() {
    let sink = Arc::new(RecordingEventSink::default());
    let mut sched = make_batch_scheduler_with_event_sink(5, 4, 16, sink.clone());

    let id0 = sched.add_request(vec![1, 2, 3, 4], 16, RequestPriority::Normal);
    let _id1 = sched.add_request(vec![5, 6, 7, 8], 16, RequestPriority::Normal);
    let id2 = sched.add_request(vec![9, 10, 11, 12], 16, RequestPriority::Normal);

    let _ = sched.schedule_step();
    let (decode, _) = expect_mixed(sched.schedule_step());
    assert_eq!(req_ids(&decode), vec![id0]);
    sched.advance_decode(id0);
    let _ = sched.schedule_step();
    let _ = sched.schedule_step();

    let events_for_preempted: Vec<EngineEvent> = recorded_events(sink.as_ref())
        .into_iter()
        .filter(|event| event.request_id == id2)
        .collect();

    assert_eq!(
        events_for_preempted,
        vec![
            EngineEvent {
                request_id: id2,
                kind: RequestEventKind::Enqueued,
                mode: None,
            },
            EngineEvent {
                request_id: id2,
                kind: RequestEventKind::PrefillStarted,
                mode: Some(InferenceMode::Prefill),
            },
            EngineEvent {
                request_id: id2,
                kind: RequestEventKind::Evicted,
                mode: Some(InferenceMode::Decode),
            },
            EngineEvent {
                request_id: id2,
                kind: RequestEventKind::Requeued,
                mode: None,
            },
        ]
    );
}
