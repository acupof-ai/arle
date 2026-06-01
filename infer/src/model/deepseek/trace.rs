//! DeepSeek V4 diagnostic operator trace aggregation.

use std::collections::BTreeMap;
use std::sync::{LazyLock, Mutex, OnceLock};

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct Dsv4OperatorTraceKey {
    phase: String,
    layer_idx: Option<usize>,
}

#[derive(Clone, Copy, Debug, Default)]
struct Dsv4OperatorTraceStats {
    calls: u64,
    total_us: u64,
    tokens: u64,
    min_us: u64,
    max_us: u64,
}

impl Dsv4OperatorTraceStats {
    fn observe(&mut self, elapsed_us: u64, tokens: usize) {
        self.calls = self.calls.saturating_add(1);
        self.total_us = self.total_us.saturating_add(elapsed_us);
        self.tokens = self.tokens.saturating_add(tokens as u64);
        self.max_us = self.max_us.max(elapsed_us);
        self.min_us = if self.min_us == 0 {
            elapsed_us
        } else {
            self.min_us.min(elapsed_us)
        };
    }

    fn delta_from(self, before: Self) -> Option<Self> {
        let calls = self.calls.saturating_sub(before.calls);
        let total_us = self.total_us.saturating_sub(before.total_us);
        let tokens = self.tokens.saturating_sub(before.tokens);
        if calls == 0 && total_us == 0 && tokens == 0 {
            return None;
        }
        Some(Self {
            calls,
            total_us,
            tokens,
            // Min/max cannot be subtracted from cumulative counters. Expose
            // the current process-global extrema with explicit field names.
            min_us: self.min_us,
            max_us: self.max_us,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Dsv4OperatorTraceSnapshot {
    entries: BTreeMap<Dsv4OperatorTraceKey, Dsv4OperatorTraceStats>,
}

#[derive(Clone, Debug)]
pub(crate) struct Dsv4OperatorTraceEntry {
    pub(crate) phase: String,
    pub(crate) layer_idx: Option<usize>,
    pub(crate) calls: u64,
    pub(crate) total_us: u64,
    pub(crate) avg_us: f64,
    pub(crate) tokens: u64,
    pub(crate) min_us_process_global: u64,
    pub(crate) max_us_process_global: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct Dsv4OperatorTraceDelta {
    pub(crate) operators: Vec<Dsv4OperatorTraceEntry>,
    pub(crate) layers: Vec<Dsv4OperatorTraceEntry>,
}

static DSV4_OPERATOR_TRACE_ENABLED: OnceLock<bool> = OnceLock::new();
static DSV4_OPERATOR_TRACE_EVENT_LOG: OnceLock<bool> = OnceLock::new();
static DSV4_OPERATOR_TRACE: LazyLock<
    Mutex<BTreeMap<Dsv4OperatorTraceKey, Dsv4OperatorTraceStats>>,
> = LazyLock::new(|| Mutex::new(BTreeMap::new()));

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|raw| !matches!(raw.as_str(), "0" | "false" | "FALSE" | "off" | "OFF"))
}

pub(crate) fn dsv4_operator_trace_enabled() -> bool {
    *DSV4_OPERATOR_TRACE_ENABLED.get_or_init(|| {
        env_truthy("ARLE_DSV4_OPERATOR_TRACE") || env_truthy("ARLE_DSV4_TRACE_LAYER")
    })
}

pub(crate) fn dsv4_trace_event_log_enabled() -> bool {
    *DSV4_OPERATOR_TRACE_EVENT_LOG.get_or_init(|| {
        env_truthy("ARLE_DSV4_TRACE_LAYER") || env_truthy("ARLE_DSV4_OPERATOR_TRACE_EVENTS")
    })
}

pub(crate) fn dsv4_operator_trace_snapshot() -> Option<Dsv4OperatorTraceSnapshot> {
    if !dsv4_operator_trace_enabled() {
        return None;
    }
    DSV4_OPERATOR_TRACE
        .lock()
        .ok()
        .map(|entries| Dsv4OperatorTraceSnapshot {
            entries: entries.clone(),
        })
}

pub(crate) fn record_dsv4_operator_trace(
    phase: &str,
    layer_idx: usize,
    tokens: usize,
    elapsed_us: u64,
) {
    if !dsv4_operator_trace_enabled() {
        return;
    }
    let Ok(mut trace) = DSV4_OPERATOR_TRACE.lock() else {
        return;
    };
    for layer_idx in [None, Some(layer_idx)] {
        let key = Dsv4OperatorTraceKey {
            phase: phase.to_string(),
            layer_idx,
        };
        trace.entry(key).or_default().observe(elapsed_us, tokens);
    }
}

pub(crate) fn dsv4_operator_trace_summary_since(
    start: &Dsv4OperatorTraceSnapshot,
) -> Option<Dsv4OperatorTraceDelta> {
    if !dsv4_operator_trace_enabled() {
        return None;
    }
    let trace = DSV4_OPERATOR_TRACE.lock().ok()?;
    let mut operators = Vec::new();
    let mut layers = Vec::new();
    for (key, current) in trace.iter() {
        let before = start.entries.get(key).copied().unwrap_or_default();
        let Some(delta) = (*current).delta_from(before) else {
            continue;
        };
        let entry = Dsv4OperatorTraceEntry {
            phase: key.phase.clone(),
            layer_idx: key.layer_idx,
            calls: delta.calls,
            total_us: delta.total_us,
            avg_us: if delta.calls > 0 {
                delta.total_us as f64 / delta.calls as f64
            } else {
                0.0
            },
            tokens: delta.tokens,
            min_us_process_global: delta.min_us,
            max_us_process_global: delta.max_us,
        };
        if key.layer_idx.is_some() {
            layers.push(entry);
        } else {
            operators.push(entry);
        }
    }
    let sort_entries = |entries: &mut Vec<Dsv4OperatorTraceEntry>| {
        entries.sort_by(|a, b| {
            b.total_us
                .cmp(&a.total_us)
                .then_with(|| b.calls.cmp(&a.calls))
                .then_with(|| a.phase.cmp(&b.phase))
                .then_with(|| a.layer_idx.cmp(&b.layer_idx))
        });
    };
    sort_entries(&mut operators);
    sort_entries(&mut layers);
    if operators.is_empty() && layers.is_empty() {
        return None;
    }
    Some(Dsv4OperatorTraceDelta { operators, layers })
}
