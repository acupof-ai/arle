//! CUDA Graph capture/replay for the decode path.
//!
//! First decode call captures the kernel sequence into a graph, later calls
//! replay it. The `AUTO_FREE_ON_LAUNCH` instantiate flag frees async-pool
//! allocations made during capture on launch.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use cudarc::driver::safe::{CudaGraph, CudaStream};
use cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
use cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
use log::debug;

/// CUDA Graph state for one fixed shape on the decode path.
///
/// The first [`Self::run_or_capture`] call captures the kernel closure into a
/// graph and launches it; every later call launches the captured graph only.
pub struct CudaGraphState {
    /// Compute stream the graph is captured on and replayed to (`end_capture`
    /// needs `&Arc<CudaStream>`, so the `Arc` is held here).
    stream: Arc<CudaStream>,
    graph: Option<CudaGraph>,
}

// SAFETY: the wrapped `CudaGraph` holds `!Send` handles (a CUDA graph must be
// captured, launched, and destroyed on its owning thread/context). The invariant:
// `CudaGraphState` lives inside the single CUDA executor driven from one blocking
// inference thread, so capture and replay always run there. Touching it from
// another thread would race the stream — undefined GPU behaviour.
unsafe impl Send for CudaGraphState {}

impl CudaGraphState {
    /// Create empty graph state bound to `stream`.
    #[must_use]
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self {
            stream,
            graph: None,
        }
    }

    /// Whether the graph for this shape has been captured yet.
    #[must_use]
    pub fn is_captured(&self) -> bool {
        self.graph.is_some()
    }

    /// Launch the captured graph, or capture it from `kernels` on the first call.
    ///
    /// `kernels` must be a pure GPU kernel sequence — no host/device sync, no
    /// host allocation — because the work is recorded into a replay-able graph.
    ///
    /// # Errors
    /// Propagates `begin_capture` / `end_capture` / `launch` driver errors.
    pub fn run_or_capture<F>(&mut self, kernels: F) -> Result<()>
    where
        F: FnOnce() -> Result<()>,
    {
        if let Some(graph) = &self.graph {
            graph
                .launch()
                .map_err(|e| anyhow::anyhow!("CUDA Graph launch failed: {e}"))?;
            return Ok(());
        }

        debug!("Capturing CUDA Graph for decode path...");
        self.stream
            .begin_capture(CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| anyhow::anyhow!("begin_capture failed: {e}"))?;

        kernels()?;

        let graph = self
            .stream
            .end_capture(CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .map_err(|e| anyhow::anyhow!("end_capture failed: {e}"))?;
        self.graph = graph;
        debug!("CUDA Graph captured successfully");

        if let Some(graph) = &self.graph {
            graph
                .launch()
                .map_err(|e| anyhow::anyhow!("CUDA Graph first launch failed: {e}"))?;
        }
        Ok(())
    }

    /// Pre-upload the captured graph so the first replay skips setup overhead.
    ///
    /// No-op when nothing has been captured yet.
    ///
    /// # Errors
    /// Propagates the `upload` driver error.
    pub fn upload(&self) -> Result<()> {
        if let Some(graph) = &self.graph {
            graph
                .upload()
                .map_err(|e| anyhow::anyhow!("CUDA Graph upload failed: {e}"))?;
        }
        Ok(())
    }
}

/// One captured decode graph keyed by its decode batch size. Wraps a
/// [`CudaGraphState`] so a later stage can carry per-bucket buffers alongside it.
pub struct CapturedDecodeGraph {
    /// Decode batch size (number of rows) this graph was captured for.
    pub batch_size: usize,
    /// The capture/replay state for this batch size.
    pub state: CudaGraphState,
}

impl CapturedDecodeGraph {
    /// Create an (uncaptured) decode graph for `batch_size` on `stream`.
    #[must_use]
    pub fn new(batch_size: usize, stream: Arc<CudaStream>) -> Self {
        Self {
            batch_size,
            state: CudaGraphState::new(stream),
        }
    }
}

/// Per-batch-size pool of captured graphs (one per batch size, reused across
/// steps). Holds the compute stream so new buckets can be created on demand.
pub struct GraphBucket {
    stream: Arc<CudaStream>,
    graphs: HashMap<usize, CapturedDecodeGraph>,
}

impl GraphBucket {
    /// Create an empty bucket pool bound to the decode compute `stream`.
    #[must_use]
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self {
            stream,
            graphs: HashMap::new(),
        }
    }

    /// Number of distinct decode batch sizes that have a graph slot.
    #[must_use]
    pub fn len(&self) -> usize {
        self.graphs.len()
    }

    /// Whether any decode batch size has a graph slot yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.graphs.is_empty()
    }

    /// Whether `batch_size` already has a captured graph.
    #[must_use]
    pub fn contains(&self, batch_size: usize) -> bool {
        self.graphs
            .get(&batch_size)
            .is_some_and(|g| g.state.is_captured())
    }

    /// Get or create the [`CapturedDecodeGraph`] for `batch_size`, creating an
    /// empty (uncaptured) entry on first request for that size.
    pub fn entry(&mut self, batch_size: usize) -> &mut CapturedDecodeGraph {
        let stream = self.stream.clone();
        self.graphs
            .entry(batch_size)
            .or_insert_with(|| CapturedDecodeGraph::new(batch_size, stream))
    }
}
