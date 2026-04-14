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
        max_tokens: 32,
        sampling: SamplingParams::default(),
        stop: None,
        priority: RequestPriority::Normal,
        session_id: None,
        delta_tx: tx,
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
        prefill_chunk_size: 0,
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

#[test]
fn preemption_mode_default_is_recompute() {
    assert_eq!(PreemptionMode::default(), PreemptionMode::Recompute);
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
        preemption_mode: PreemptionMode::Recompute,
        ..Default::default()
    };
    let bm = crate::block_manager::BlockManager::new(num_gpu_blocks, 0, block_size);
    BatchScheduler::with_event_sink(config, bm, event_sink)
}

/// Run one simulated step: process the ScheduleDecision without a real model.
/// Returns (num_prefilled_reqs, num_decoded_reqs).
fn sim_step(sched: &mut BatchScheduler) -> (usize, usize) {
    match sched.schedule_step() {
        ScheduleDecision::Idle => (0, 0),
        ScheduleDecision::PrefillBatch(p) => (p.req_ids.len(), 0),
        ScheduleDecision::DecodeBatch(d) => {
            let to_finish = d.req_ids;
            for id in &to_finish {
                sched.advance_decode(*id);
            }
            (0, to_finish.len())
        }
        ScheduleDecision::Mixed { decode, prefill } => {
            let n_prefill = prefill.req_ids.len();
            let to_finish = decode.req_ids;
            for id in &to_finish {
                sched.advance_decode(*id);
            }
            (n_prefill, to_finish.len())
        }
    }
}

#[test]
fn test_continuous_batching() {
    let mut sched = make_batch_scheduler(32, 4, 64);

    let id0 = sched.add_request(vec![1, 2, 3, 4], 8, RequestPriority::Normal);
    let id1 = sched.add_request(vec![5, 6, 7, 8], 8, RequestPriority::Normal);
    let _id2 = sched.add_request(vec![9, 10, 11, 12], 8, RequestPriority::Normal);

    match sched.schedule_step() {
        ScheduleDecision::PrefillBatch(p) => {
            assert_eq!(p.req_ids, vec![id0]);
            assert_eq!(p.input_ids, vec![1, 2, 3, 4]);
        }
        other => panic!(
            "expected PrefillBatch, got {:?}",
            matches!(other, ScheduleDecision::Idle)
        ),
    }
    assert!(sched.is_running(id0));
    assert_eq!(sched.waiting_len(), 2);

    match sched.schedule_step() {
        ScheduleDecision::Mixed { decode, prefill } => {
            assert_eq!(decode.req_ids, vec![id0]);
            assert_eq!(prefill.req_ids, vec![id1]);
            sched.advance_decode(id0);
        }
        ScheduleDecision::PrefillBatch(p) => {
            assert_eq!(p.req_ids, vec![id1]);
        }
        other => panic!("unexpected: {:?}", matches!(other, ScheduleDecision::Idle)),
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

    match sched.schedule_step() {
        ScheduleDecision::PrefillBatch(p) => assert_eq!(p.req_ids, vec![id0]),
        _ => panic!("expected prefill id0"),
    }
    match sched.schedule_step() {
        ScheduleDecision::Mixed { decode, prefill } => {
            assert_eq!(decode.req_ids, vec![id0]);
            assert_eq!(prefill.req_ids, vec![id1]);
            sched.advance_decode(id0);
        }
        ScheduleDecision::PrefillBatch(p) => {
            assert_eq!(p.req_ids, vec![id1]);
        }
        _ => panic!("unexpected decision"),
    }
    match sched.schedule_step() {
        ScheduleDecision::Mixed { prefill, .. } => {
            assert_eq!(prefill.req_ids, vec![id2]);
        }
        ScheduleDecision::PrefillBatch(p) => {
            assert_eq!(p.req_ids, vec![id2]);
        }
        _ => {}
    }

    let free_before = sched.free_gpu_blocks();
    match sched.schedule_step() {
        ScheduleDecision::DecodeBatch(d) => {
            assert!(
                !d.req_ids.contains(&id2),
                "id2 should be preempted, not decoding"
            );
            assert!(d.req_ids.contains(&id0) || d.req_ids.contains(&id1));
        }
        ScheduleDecision::Mixed { decode, .. } => {
            assert!(!decode.req_ids.contains(&id2));
        }
        ScheduleDecision::Idle | ScheduleDecision::PrefillBatch(_) => {}
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

    match sched.schedule_step() {
        ScheduleDecision::PrefillBatch(p) => {
            assert_eq!(p.req_ids, vec![id]);
            assert_eq!(p.input_ids.len(), chunk_size);
            assert_eq!(p.input_ids[0], 1);
            assert_eq!(p.input_ids[chunk_size - 1], chunk_size as u32);
            assert_eq!(p.seq_lens, vec![chunk_size]);
        }
        other => panic!(
            "expected PrefillBatch chunk 1, got Idle={}",
            matches!(other, ScheduleDecision::Idle)
        ),
    }
    assert!(!sched.is_running(id));
    assert_eq!(sched.waiting_len(), 1);

    match sched.schedule_step() {
        ScheduleDecision::PrefillBatch(p) => {
            assert_eq!(p.req_ids, vec![id]);
            assert_eq!(p.input_ids.len(), chunk_size);
            assert_eq!(p.input_ids[0], chunk_size as u32 + 1);
            assert_eq!(p.seq_lens, vec![chunk_size * 2]);
        }
        other => panic!(
            "expected PrefillBatch chunk 2, got Idle={}",
            matches!(other, ScheduleDecision::Idle)
        ),
    }
    assert!(!sched.is_running(id));

    match sched.schedule_step() {
        ScheduleDecision::PrefillBatch(p) => {
            assert_eq!(p.req_ids, vec![id]);
            assert_eq!(p.input_ids.len(), 4);
            assert_eq!(p.input_ids[0], 17);
            assert_eq!(p.seq_lens, vec![20]);
        }
        other => panic!(
            "expected PrefillBatch final chunk, got Idle={}",
            matches!(other, ScheduleDecision::Idle)
        ),
    }
    assert!(sched.is_running(id));
    assert_eq!(sched.waiting_len(), 0);
    assert_eq!(sched.free_gpu_blocks(), 8 - 5);
}

#[test]
fn batch_scheduler_emits_prefill_started_once_for_chunked_request() {
    let sink = Arc::new(RecordingEventSink::default());
    let mut sched = make_batch_scheduler_with_event_sink(8, 4, 8, sink.clone());
    let id = sched.add_request((1u32..=20).collect(), 4, RequestPriority::Normal);

    match sched.schedule_step() {
        ScheduleDecision::PrefillBatch(p) => {
            assert_eq!(p.req_ids, vec![id]);
            assert_eq!(p.input_ids.len(), 8);
        }
        other => panic!(
            "expected first prefill chunk, got Idle={}",
            matches!(other, ScheduleDecision::Idle)
        ),
    }
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

    match sched.schedule_step() {
        ScheduleDecision::PrefillBatch(p) => {
            assert_eq!(p.req_ids, vec![id]);
            assert_eq!(p.input_ids.len(), 8);
        }
        other => panic!(
            "expected second prefill chunk, got Idle={}",
            matches!(other, ScheduleDecision::Idle)
        ),
    }
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

    match sched.schedule_step() {
        ScheduleDecision::PrefillBatch(p) => {
            assert_eq!(p.req_ids, vec![id]);
            assert_eq!(p.input_ids.len(), 4);
        }
        other => panic!(
            "expected final prefill chunk, got Idle={}",
            matches!(other, ScheduleDecision::Idle)
        ),
    }
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

    match sched.schedule_step() {
        ScheduleDecision::DecodeBatch(decode) => {
            assert_eq!(decode.req_ids, vec![id]);
        }
        other => panic!(
            "expected decode batch after final prefill, got {:?}",
            std::mem::discriminant(&other)
        ),
    }
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
fn batch_scheduler_uses_decode_aware_chunking_policy() {
    let mut sched = make_batch_scheduler(512, 4, 512);
    let req0 = sched.add_request((1..=32).collect(), 8, RequestPriority::Normal);
    let _req1 = sched.add_request((33..=512).collect(), 8, RequestPriority::Normal);

    // Step 1: prefill req0, it becomes running for decode.
    let _ = sched.schedule_step();

    // Step 2: mixed step, prefill chunk should be decode-capped by policy (64).
    match sched.schedule_step() {
        ScheduleDecision::Mixed { prefill, .. } => {
            assert_eq!(prefill.req_ids, vec![RequestId(req0.0 + 1)]);
            assert_eq!(prefill.input_ids.len(), 64);
        }
        other => panic!(
            "expected Mixed schedule decision, got {:?}",
            std::mem::discriminant(&other)
        ),
    }
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

    match sched.schedule_step() {
        ScheduleDecision::PrefillBatch(prefill) => {
            assert_eq!(prefill.req_ids, vec![req]);
            assert_eq!(prefill.input_ids, vec![1, 2, 3, 4]);
        }
        other => panic!(
            "expected PrefillBatch, got Idle={}",
            matches!(other, ScheduleDecision::Idle)
        ),
    }

    match sched.schedule_step() {
        ScheduleDecision::DecodeBatch(decode) => {
            assert_eq!(decode.req_ids, vec![req]);
        }
        other => panic!(
            "expected DecodeBatch, got {:?}",
            std::mem::discriminant(&other)
        ),
    }
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

    assert!(matches!(sched.schedule_step(), ScheduleDecision::Idle));
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
    match sched.schedule_step() {
        ScheduleDecision::Mixed { decode, .. } => {
            assert_eq!(decode.req_ids, vec![id0]);
            sched.advance_decode(id0);
        }
        other => panic!(
            "expected Mixed schedule decision, got {:?}",
            std::mem::discriminant(&other)
        ),
    }
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
