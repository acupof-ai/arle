//! Batched CUDA graph pool for multi-request decode.
//!
//! The decode path in vLLM/sglang replays pre-captured CUDA graphs to
//! eliminate CPU-GPU dispatch overhead.  To support variable batch sizes,
//! a pool of graphs is maintained — one per "padded" batch size.  Requests
//! are padded to the next supported size and the matching graph is replayed.
//!
//! # CPU-verifiable parts
//!
//! - Batch-size padding arithmetic ([`pad_to_pool_size`], [`POOL_BATCH_SIZES`])
//! - Graph pool state tracking ([`GraphPool`])
//! - Warmup schedule generation ([`warmup_schedule`])
//!
//! # GPU stubs
//!
//! - Actual CUDA graph capture / replay is model-specific today. This module
//!   intentionally does not expose a generic capture entry that pretends to be
//!   implemented when it is not.

use std::collections::HashMap;

// ============================================================================
// Pool batch sizes
// ============================================================================

/// Supported decode batch sizes for CUDA graph compilation.
///
/// Graphs are compiled once per size; requests are padded to the next size up.
/// Larger than 128 falls through to eager (non-graph) execution.
pub const POOL_BATCH_SIZES: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128];

/// Largest supported graph batch size.
pub const MAX_GRAPH_BATCH_SIZE: usize = 128;

/// Round `n` up to the smallest supported pool batch size ≥ `n`.
///
/// Returns `None` if `n > MAX_GRAPH_BATCH_SIZE` (run eagerly instead).
pub fn pad_to_pool_size(n: usize) -> Option<usize> {
    POOL_BATCH_SIZES.iter().copied().find(|&s| n <= s)
}

/// Returns `true` if a CUDA graph will be used for this batch size.
pub fn is_graph_eligible(n: usize) -> bool {
    pad_to_pool_size(n).is_some()
}

/// Returns the number of padding slots needed to bring `n` up to the next pool size.
pub fn padding_slots(n: usize) -> usize {
    match pad_to_pool_size(n) {
        Some(padded) => padded - n,
        None => 0,
    }
}

// ============================================================================
// GraphCaptureState
// ============================================================================

/// Capture state of a single graph in the pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphCaptureState {
    /// Not yet captured.
    Uncaptured,
    /// Currently being captured (stream is in capture mode).
    Capturing,
    /// Successfully captured and ready for replay.
    Ready,
    /// Capture failed; will fall back to eager execution.
    Failed,
}

impl GraphCaptureState {
    pub fn is_ready(self) -> bool {
        matches!(self, Self::Ready)
    }
}

// ============================================================================
// GraphPool
// ============================================================================

/// CPU-side state tracking for the CUDA graph pool.
///
/// Tracks which batch sizes have been captured and how many times each graph
/// has been replayed.  The actual CUDA graph objects are managed by the GPU
/// layer (behind `#[cfg(feature = "cuda")]`).
pub struct GraphPool {
    states: HashMap<usize, GraphCaptureState>,
    replay_counts: HashMap<usize, u64>,
}

impl Default for GraphPool {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphPool {
    /// Create an empty pool.  No graphs are captured until [`warmup`] is called.
    pub fn new() -> Self {
        let states = POOL_BATCH_SIZES
            .iter()
            .map(|&bs| (bs, GraphCaptureState::Uncaptured))
            .collect();
        Self {
            states,
            replay_counts: HashMap::new(),
        }
    }

    /// Returns the capture state for the given batch size.
    ///
    /// Returns `None` if `batch_size` is not a pool size.
    pub fn state(&self, batch_size: usize) -> Option<GraphCaptureState> {
        self.states.get(&batch_size).copied()
    }

    /// Returns `true` if the graph for this batch size is ready for replay.
    pub fn is_ready(&self, batch_size: usize) -> bool {
        self.states
            .get(&batch_size)
            .copied()
            .is_some_and(GraphCaptureState::is_ready)
    }

    /// Mark a graph as currently being captured.
    pub fn mark_capturing(&mut self, batch_size: usize) {
        if let Some(s) = self.states.get_mut(&batch_size) {
            *s = GraphCaptureState::Capturing;
        }
    }

    /// Mark a graph as successfully captured and ready.
    pub fn mark_ready(&mut self, batch_size: usize) {
        if let Some(s) = self.states.get_mut(&batch_size) {
            *s = GraphCaptureState::Ready;
        }
    }

    /// Mark a graph capture as failed (will fall back to eager execution).
    pub fn mark_failed(&mut self, batch_size: usize) {
        if let Some(s) = self.states.get_mut(&batch_size) {
            *s = GraphCaptureState::Failed;
        }
    }

    /// Invalidate a graph (e.g. after weight update); must be re-captured.
    pub fn invalidate(&mut self, batch_size: usize) {
        if let Some(s) = self.states.get_mut(&batch_size) {
            *s = GraphCaptureState::Uncaptured;
        }
        self.replay_counts.remove(&batch_size);
    }

    /// Invalidate all graphs.
    pub fn invalidate_all(&mut self) {
        for s in self.states.values_mut() {
            *s = GraphCaptureState::Uncaptured;
        }
        self.replay_counts.clear();
    }

    /// Record a graph replay.
    pub fn record_replay(&mut self, batch_size: usize) {
        *self.replay_counts.entry(batch_size).or_insert(0) += 1;
    }

    /// Total replay count for a graph.
    pub fn replay_count(&self, batch_size: usize) -> u64 {
        self.replay_counts.get(&batch_size).copied().unwrap_or(0)
    }

    /// List all ready batch sizes, sorted ascending.
    pub fn ready_sizes(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self
            .states
            .iter()
            .filter(|(_, s)| s.is_ready())
            .map(|(bs, _)| *bs)
            .collect();
        v.sort_unstable();
        v
    }

    /// Number of graphs currently ready.
    pub fn num_ready(&self) -> usize {
        self.states.values().filter(|s| s.is_ready()).count()
    }

    /// Decide how to run a decode step for `n` requests.
    ///
    /// Returns the decision: which padded batch size to use and whether to
    /// use a graph or run eagerly.
    pub fn dispatch(&mut self, n: usize) -> DispatchDecision {
        match pad_to_pool_size(n) {
            None => DispatchDecision::Eager { actual: n },
            Some(padded) => {
                if self.is_ready(padded) {
                    self.record_replay(padded);
                    DispatchDecision::Graph {
                        padded,
                        actual: n,
                        padding: padded - n,
                    }
                } else {
                    DispatchDecision::Eager { actual: n }
                }
            }
        }
    }
}

/// How the scheduler should execute a decode batch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DispatchDecision {
    /// Use the pre-captured CUDA graph for `padded` batch size.
    Graph {
        /// Padded batch size (pool size to use).
        padded: usize,
        /// Actual number of active requests.
        actual: usize,
        /// Number of dummy padding slots.
        padding: usize,
    },
    /// Run eagerly (no graph available or batch too large).
    Eager {
        /// Actual number of active requests.
        actual: usize,
    },
}

impl DispatchDecision {
    pub fn is_graph(&self) -> bool {
        matches!(self, Self::Graph { .. })
    }

    pub fn actual_size(&self) -> usize {
        match self {
            Self::Graph { actual, .. } | Self::Eager { actual } => *actual,
        }
    }
}

// ============================================================================
// Warmup schedule
// ============================================================================

/// Generate the ordered list of batch sizes to warm up (capture graphs for).
///
/// Warmup always starts from the smallest batch size to ensure the graph
/// pool is populated for the most common case (batch size 1).
///
/// `max_batch` clips the schedule at the largest useful size.
pub fn warmup_schedule(max_batch: usize) -> Vec<usize> {
    POOL_BATCH_SIZES
        .iter()
        .copied()
        .filter(|&bs| bs <= max_batch)
        .collect()
}

// ============================================================================
// GPU capture stub
// ============================================================================

/// Generic graph-pool capture is intentionally unavailable.
///
/// The live CUDA decode path captures graphs in model-specific code where the
/// tensor layout and replay invariants are known. Callers should use that path
/// instead of treating [`GraphPool`] as a complete capture implementation.
#[allow(unused_variables)]
pub(crate) fn capture_decode_graph<F>(
    pool: &mut GraphPool,
    batch_size: usize,
    kernels: F,
) -> anyhow::Result<()>
where
    F: FnOnce() -> anyhow::Result<()>,
{
    anyhow::bail!(
        "generic CUDA graph capture is unavailable for batch_size={batch_size}; \
         use the model-specific capture path"
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_to_pool_size_exact() {
        assert_eq!(pad_to_pool_size(1), Some(1));
        assert_eq!(pad_to_pool_size(4), Some(4));
        assert_eq!(pad_to_pool_size(128), Some(128));
    }

    #[test]
    fn pad_to_pool_size_rounds_up() {
        assert_eq!(pad_to_pool_size(3), Some(4));
        assert_eq!(pad_to_pool_size(5), Some(8));
        assert_eq!(pad_to_pool_size(9), Some(16));
        assert_eq!(pad_to_pool_size(127), Some(128));
    }

    #[test]
    fn pad_to_pool_size_over_max_returns_none() {
        assert_eq!(pad_to_pool_size(129), None);
        assert_eq!(pad_to_pool_size(1000), None);
    }

    #[test]
    fn padding_slots_correct() {
        assert_eq!(padding_slots(1), 0);
        assert_eq!(padding_slots(3), 1); // rounds to 4
        assert_eq!(padding_slots(5), 3); // rounds to 8
        assert_eq!(padding_slots(256), 0); // over max, no padding
    }

    #[test]
    fn is_graph_eligible() {
        assert!(super::is_graph_eligible(1));
        assert!(super::is_graph_eligible(128));
        assert!(!super::is_graph_eligible(129));
        assert!(!super::is_graph_eligible(200));
    }

    #[test]
    fn graph_pool_initial_state() {
        let pool = GraphPool::new();
        for &bs in POOL_BATCH_SIZES {
            assert_eq!(pool.state(bs), Some(GraphCaptureState::Uncaptured));
        }
        assert_eq!(pool.num_ready(), 0);
        assert!(pool.ready_sizes().is_empty());
    }

    #[test]
    fn graph_pool_capture_lifecycle() {
        let mut pool = GraphPool::new();
        pool.mark_capturing(4);
        assert_eq!(pool.state(4), Some(GraphCaptureState::Capturing));
        pool.mark_ready(4);
        assert_eq!(pool.state(4), Some(GraphCaptureState::Ready));
        assert!(pool.is_ready(4));
        assert_eq!(pool.num_ready(), 1);
        assert_eq!(pool.ready_sizes(), vec![4]);
    }

    #[test]
    fn graph_pool_replay_counting() {
        let mut pool = GraphPool::new();
        pool.mark_ready(1);
        pool.record_replay(1);
        pool.record_replay(1);
        assert_eq!(pool.replay_count(1), 2);
        pool.invalidate(1);
        assert_eq!(pool.replay_count(1), 0);
        assert_eq!(pool.state(1), Some(GraphCaptureState::Uncaptured));
    }

    #[test]
    fn graph_pool_invalidate_all() {
        let mut pool = GraphPool::new();
        for &bs in POOL_BATCH_SIZES {
            pool.mark_ready(bs);
        }
        assert_eq!(pool.num_ready(), POOL_BATCH_SIZES.len());
        pool.invalidate_all();
        assert_eq!(pool.num_ready(), 0);
    }

    #[test]
    fn dispatch_uses_graph_when_ready() {
        let mut pool = GraphPool::new();
        pool.mark_ready(8);

        let d = pool.dispatch(6);
        assert!(d.is_graph());
        assert_eq!(d.actual_size(), 6);
        match d {
            DispatchDecision::Graph {
                padded,
                actual,
                padding,
            } => {
                assert_eq!(padded, 8);
                assert_eq!(actual, 6);
                assert_eq!(padding, 2);
            }
            _ => panic!("expected Graph"),
        }
    }

    #[test]
    fn dispatch_falls_back_to_eager_when_not_ready() {
        let mut pool = GraphPool::new(); // nothing captured
        let d = pool.dispatch(4);
        assert!(!d.is_graph());
        assert_eq!(d.actual_size(), 4);
    }

    #[test]
    fn dispatch_eager_for_oversized_batch() {
        let mut pool = GraphPool::new();
        for &bs in POOL_BATCH_SIZES {
            pool.mark_ready(bs);
        }
        let d = pool.dispatch(200);
        assert!(!d.is_graph()); // no graph for 200
        assert_eq!(d.actual_size(), 200);
    }

    #[test]
    fn warmup_schedule_respects_max() {
        let sched = warmup_schedule(16);
        assert_eq!(sched, vec![1, 2, 4, 8, 16]);

        let sched_all = warmup_schedule(MAX_GRAPH_BATCH_SIZE);
        assert_eq!(sched_all.len(), POOL_BATCH_SIZES.len());
    }

    #[test]
    fn warmup_schedule_empty_when_max_zero() {
        assert!(warmup_schedule(0).is_empty());
    }

    #[test]
    fn graph_pool_failed_state() {
        let mut pool = GraphPool::new();
        pool.mark_failed(32);
        assert_eq!(pool.state(32), Some(GraphCaptureState::Failed));
        assert!(!pool.is_ready(32));
    }

    #[test]
    fn capture_decode_graph_returns_error_instead_of_panicking() {
        let mut pool = GraphPool::new();
        let err = capture_decode_graph(&mut pool, 8, || Ok(())).expect_err("capture should fail");
        assert!(
            err.to_string()
                .contains("generic CUDA graph capture is unavailable")
        );
    }
}
