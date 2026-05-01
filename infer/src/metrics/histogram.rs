use std::fmt::Write as FmtWrite;

/// Fixed-bucket histogram for latency metrics.
///
/// Bucket boundaries follow a log-linear scale covering [1ms, 60s].
pub struct Histogram {
    /// Upper inclusive bounds for each bucket (seconds).
    buckets: Vec<f64>,
    /// Count of observations falling into each bucket (cumulative, like Prometheus).
    counts: Vec<u64>,
    sum: f64,
    count: u64,
}

/// Default latency buckets in seconds, covering 1ms … 60s.
/// Finer granularity in the 10–100ms range where ITL/TPOT typically falls.
pub const LATENCY_BUCKETS: &[f64] = &[
    0.001, 0.002, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050, 0.075, 0.100,
    0.150, 0.200, 0.300, 0.500, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0,
];

/// Speculative-step latency buckets in microseconds for `*_us` metrics.
pub(super) const MICROSECOND_BUCKETS: &[f64] = &[
    10.0,
    25.0,
    50.0,
    75.0,
    100.0,
    150.0,
    200.0,
    300.0,
    500.0,
    750.0,
    1_000.0,
    2_000.0,
    5_000.0,
    10_000.0,
    25_000.0,
    50_000.0,
    100_000.0,
    250_000.0,
    500_000.0,
    1_000_000.0,
];

pub(super) fn secs_to_micros(secs: f64) -> u64 {
    (secs.max(0.0) * 1_000_000.0).round() as u64
}

pub(super) fn micros_to_secs(micros: u64) -> f64 {
    micros as f64 / 1_000_000.0
}

impl Histogram {
    /// Create a new histogram with the given bucket boundaries (in seconds).
    /// Buckets are sorted ascending. Duplicate boundaries are de-duplicated by sort.
    pub fn new(buckets: &[f64]) -> Self {
        let mut sorted = buckets.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let counts = vec![0u64; sorted.len()];
        Self {
            buckets: sorted,
            counts,
            sum: 0.0,
            count: 0,
        }
    }

    /// Record one observation.
    pub fn observe(&mut self, value: f64) {
        self.sum += value;
        self.count += 1;
        for (i, &bound) in self.buckets.iter().enumerate() {
            if value <= bound {
                self.counts[i] += 1;
                break;
            }
        }
    }

    /// Render as Prometheus histogram lines.
    pub fn render(&self, name: &str, labels: &str) -> String {
        let mut out = String::new();
        let mut cumulative = 0u64;
        for (i, &bound) in self.buckets.iter().enumerate() {
            cumulative += self.counts[i];
            writeln!(
                out,
                "{name}_bucket{{{labels}le=\"{bound:.3}\"}} {cumulative}"
            )
            .unwrap();
        }
        writeln!(out, "{name}_bucket{{{labels}le=\"+Inf\"}} {}", self.count).unwrap();
        writeln!(out, "{name}_sum{{{labels}}} {:.6}", self.sum).unwrap();
        writeln!(out, "{name}_count{{{labels}}} {}", self.count).unwrap();
        out
    }

    pub fn sum(&self) -> f64 {
        self.sum
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    /// Estimate percentile using bucket counts.
    /// Returns `None` if no observations.
    pub fn percentile(&self, p: f64) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        let target = (p * self.count as f64).ceil() as u64;
        let mut cumulative = 0u64;
        for (i, &bound) in self.buckets.iter().enumerate() {
            cumulative += self.counts[i];
            if cumulative >= target {
                return Some(bound);
            }
        }
        self.buckets.last().copied()
    }
}

pub struct HistogramSet {
    pub queue_wait: Histogram,
    pub active_ttft: Histogram,
    pub service: Histogram,
    pub ttft: Histogram,
    pub tpot: Histogram,
    pub e2e: Histogram,
    pub scheduler_step: Histogram,
    pub spec_step_latency_us: Histogram,
}

impl HistogramSet {
    /// Create a new set of TTFT, TPOT, and E2E histograms using the default latency buckets.
    pub fn new() -> Self {
        Self {
            queue_wait: Histogram::new(LATENCY_BUCKETS),
            active_ttft: Histogram::new(LATENCY_BUCKETS),
            service: Histogram::new(LATENCY_BUCKETS),
            ttft: Histogram::new(LATENCY_BUCKETS),
            tpot: Histogram::new(LATENCY_BUCKETS),
            e2e: Histogram::new(LATENCY_BUCKETS),
            scheduler_step: Histogram::new(LATENCY_BUCKETS),
            spec_step_latency_us: Histogram::new(MICROSECOND_BUCKETS),
        }
    }
}

impl Default for HistogramSet {
    fn default() -> Self {
        Self::new()
    }
}
