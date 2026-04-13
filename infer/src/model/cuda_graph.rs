use anyhow::Result;
use log::debug;

use cudarc::driver::safe::CudaGraph;
use cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
use cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;

use crate::backend::cuda::tensor::DeviceContext;

/// CUDA Graph state for decode path.
/// First decode call captures the graph; subsequent calls replay it.
pub(crate) struct CudaGraphState {
    graph: Option<CudaGraph>,
}

// SAFETY: `CudaGraphState` wraps a `cudarc::driver::safe::CudaGraph` which holds
// raw CUDA graph and instantiated-graph handles (`CUgraph`, `CUgraphExec`).
// These are `!Send` because CUDA graphs must be used from the thread/context
// that created them.
//
// Invariant upheld: `CudaGraphState` is stored inside model state structs
// (e.g. `Qwen3State`) which are exclusively accessed from the single blocking
// inference thread running `Scheduler::run()`.  Graph capture and replay both
// occur on that same thread with the same CUDA stream.
//
// Violation would mean: replaying or destroying the graph from a different thread
// would race with the owning thread's CUDA stream, causing undefined GPU behaviour.
unsafe impl Send for CudaGraphState {}

impl CudaGraphState {
    pub(crate) fn new() -> Self {
        Self { graph: None }
    }

    /// Run kernel closure directly, or capture into a graph and replay.
    ///
    /// `kernels` must be a pure GPU kernel sequence — no CPU-GPU sync, no allocation.
    pub(crate) fn run_or_capture<F>(&mut self, ctx: &DeviceContext, kernels: F) -> Result<()>
    where
        F: FnOnce() -> Result<()>,
    {
        if let Some(graph) = &self.graph {
            graph
                .launch()
                .map_err(|e| anyhow::anyhow!("CUDA Graph launch failed: {}", e))?;
        } else {
            debug!("Capturing CUDA Graph for decode path...");
            ctx.stream
                .begin_capture(CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .map_err(|e| anyhow::anyhow!("begin_capture failed: {}", e))?;

            kernels()?;

            self.graph = ctx
                .stream
                .end_capture(CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
                .map_err(|e| anyhow::anyhow!("end_capture failed: {}", e))?;
            debug!("CUDA Graph captured successfully");

            if let Some(ref graph) = self.graph {
                graph
                    .launch()
                    .map_err(|e| anyhow::anyhow!("CUDA Graph first launch failed: {}", e))?;
            }
        }
        Ok(())
    }
}
