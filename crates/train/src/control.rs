//! Shared-state control plane for long-running training jobs.
//!
//! A `TrainingController` wraps the trainer's live counters plus a
//! cooperative `should_stop` flag. The trainer reads/writes through
//! `update` + `should_stop`; external drivers (HTTP handlers, tests)
//! read via `snapshot` and signal via `request_stop` / `request_save`.
//! Locking is coarse — one `Mutex` around a ~hundred-byte struct —
//! because iterations are orders of magnitude slower than the
//! contended section.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::atomic::{AtomicU64, AtomicUsize};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::metrics::{MetricSample, MetricSink, TrainEvent};

#[derive(Debug, Clone, Default)]
pub struct TrainingStatus {
    pub iter: usize,
    pub total_iters: usize,
    pub mean_reward: f32,
    pub best_reward: f32,
    pub last_kl: f32,
    pub last_loss: f32,
    pub wall_secs: f32,
    pub dropped_metrics: u64,
    pub started: bool,
    pub finished: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct TrainingRecord {
    pub seq: u64,
    pub ts_ms: u64,
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub strings: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub scalars: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub bools: BTreeMap<String, bool>,
}

#[derive(Debug, Default)]
pub struct TrainingController {
    status: Mutex<TrainingStatus>,
    records: Mutex<VecDeque<TrainingRecord>>,
    record_cap: AtomicUsize,
    next_seq: AtomicU64,
    stop: AtomicBool,
    save: AtomicBool,
}

impl TrainingController {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            status: Mutex::new(TrainingStatus::default()),
            records: Mutex::new(VecDeque::new()),
            record_cap: AtomicUsize::new(512),
            next_seq: AtomicU64::new(0),
            stop: AtomicBool::new(false),
            save: AtomicBool::new(false),
        })
    }

    pub fn set_record_capacity(&self, capacity: usize) {
        let capacity = capacity.max(1);
        self.record_cap.store(capacity, Ordering::Release);
        let mut guard = self.records.lock().expect("records poisoned");
        while guard.len() > capacity {
            guard.pop_front();
        }
    }

    pub fn snapshot(&self) -> TrainingStatus {
        self.status.lock().expect("status poisoned").clone()
    }

    pub fn update(&self, f: impl FnOnce(&mut TrainingStatus)) {
        let mut guard = self.status.lock().expect("status poisoned");
        f(&mut guard);
    }

    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::Release);
        self.record_status("operator_request", &[("stop_requested", true)]);
    }

    pub fn should_stop(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }

    /// External driver requests the trainer to flush a checkpoint at
    /// the next iteration boundary. Flag is edge-triggered — the
    /// trainer resets it via `take_save_request` once handled.
    pub fn request_save(&self) {
        self.save.store(true, Ordering::Release);
        self.record_status("operator_request", &[("save_requested", true)]);
    }

    pub fn take_save_request(&self) -> bool {
        self.save.swap(false, Ordering::AcqRel)
    }

    pub fn recent_records(&self, after_seq: Option<u64>) -> Vec<TrainingRecord> {
        let guard = self.records.lock().expect("records poisoned");
        guard
            .iter()
            .filter(|record| after_seq.is_none_or(|after| record.seq > after))
            .cloned()
            .collect()
    }

    pub fn latest_seq(&self) -> u64 {
        self.next_seq.load(Ordering::Acquire)
    }

    pub fn metric_sink(self: &Arc<Self>) -> ControllerSink {
        ControllerSink {
            controller: Arc::clone(self),
        }
    }

    fn record_metric(&self, sample: &MetricSample<'_>) {
        let record = TrainingRecord {
            seq: self.next_seq.fetch_add(1, Ordering::AcqRel) + 1,
            ts_ms: now_ms(),
            kind: "metric".to_string(),
            step: Some(sample.step),
            phase: Some(sample.phase.to_string()),
            strings: BTreeMap::new(),
            scalars: sample
                .fields
                .iter()
                .map(|(k, v)| ((*k).to_string(), *v))
                .collect(),
            bools: BTreeMap::new(),
        };
        self.update_status_from_record(&record);
        self.push_record(record);
    }

    fn record_event(&self, event: &TrainEvent<'_>) {
        let record = TrainingRecord {
            seq: self.next_seq.fetch_add(1, Ordering::AcqRel) + 1,
            ts_ms: now_ms(),
            kind: event.kind.to_string(),
            step: event.step,
            phase: None,
            strings: event
                .strings
                .iter()
                .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                .collect(),
            scalars: event
                .scalars
                .iter()
                .map(|(k, v)| ((*k).to_string(), *v))
                .collect(),
            bools: event
                .bools
                .iter()
                .map(|(k, v)| ((*k).to_string(), *v))
                .collect(),
        };
        self.update_status_from_record(&record);
        self.push_record(record);
    }

    fn record_status(&self, reason: &str, bools: &[(&str, bool)]) {
        let record = TrainingRecord {
            seq: self.next_seq.fetch_add(1, Ordering::AcqRel) + 1,
            ts_ms: now_ms(),
            kind: "status".to_string(),
            step: Some(self.snapshot().iter as u64),
            phase: None,
            strings: BTreeMap::from([("reason".to_string(), reason.to_string())]),
            scalars: BTreeMap::new(),
            bools: bools.iter().map(|(k, v)| ((*k).to_string(), *v)).collect(),
        };
        self.update_status_from_record(&record);
        self.push_record(record);
    }

    fn push_record(&self, record: TrainingRecord) {
        let cap = self.record_cap.load(Ordering::Acquire).max(1);
        let mut guard = self.records.lock().expect("records poisoned");
        guard.push_back(record);
        while guard.len() > cap {
            guard.pop_front();
        }
    }

    fn update_status_from_record(&self, record: &TrainingRecord) {
        let mut guard = self.status.lock().expect("status poisoned");
        if let Some(step) = record.step {
            guard.iter = step as usize;
        }
        if let Some(value) = record.scalars.get("loss") {
            guard.last_loss = *value as f32;
        }
        if let Some(value) = record.scalars.get("mean_reward") {
            guard.mean_reward = *value as f32;
        }
        if let Some(value) = record.scalars.get("best_reward") {
            guard.best_reward = *value as f32;
        }
        if let Some(value) = record.scalars.get("best_mean_reward") {
            guard.best_reward = *value as f32;
        }
        if let Some(value) = record.scalars.get("mean_kl") {
            guard.last_kl = *value as f32;
        }
        if let Some(value) = record.scalars.get("wall_secs") {
            guard.wall_secs = *value as f32;
        }
        if let Some(value) = record.scalars.get("dropped_metrics") {
            guard.dropped_metrics = (*value).max(0.0) as u64;
        }
        match record.kind.as_str() {
            "run_start" => guard.started = true,
            "run_end" => guard.finished = true,
            _ => {}
        }
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

pub struct ControllerSink {
    controller: Arc<TrainingController>,
}

impl MetricSink for ControllerSink {
    fn emit(&mut self, sample: &MetricSample<'_>) {
        self.controller.record_metric(sample);
    }

    fn event(&mut self, event: &TrainEvent<'_>) {
        self.controller.record_event(event);
    }
}
