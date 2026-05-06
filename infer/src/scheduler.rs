//! Multi-request scheduler with state pooling and decode-priority scheduling.
//!
//! Architecture:
//! ```text
//! HTTP Request → SchedulerHandle.submit() → channel → Scheduler.run()
//!                                                        ↓
//!                                              GPU (one forward at a time)
//!                                                        ↓
//!                                              CompletionStreamDelta → HTTP Response
//! ```
//!
//! The scheduler interleaves multiple requests on a single GPU by:
//! 1. Prioritizing decode steps (1 token each) over prefill
//! 2. Chunking long prefills (512 tokens) so decode can interleave
//! 3. Round-robin among active decode requests
//! 4. Starting new prefills only when no decode work is pending

mod batch;
pub mod forward_batch;
pub mod metrics;
pub mod plan;
pub mod policy;
mod types;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(test)]
mod tests;

pub use batch::{
    BatchScheduler, BatchSchedulerConfig, DecodeBatch, PendingRequest, PrefillBatch,
    RunningRequest, ScheduleDecision,
};
#[cfg(feature = "cuda")]
pub use cuda::Scheduler;
pub use forward_batch::{
    ForwardBatch, ForwardBatchKind, IntermediateTensorMeta, IntermediateTensors, TensorPayload,
};
pub use plan::{
    GeneratedToken, LogicalBatchShape, LogicalDecodeRow, LogicalPlanLowering, LogicalPrefillRow,
    LogicalServePlan, LogicalSparseDraftView, LogicalSpecDecodeRow, LogicalStepOutput,
};
pub use types::{
    DraftMode, IncomingRequest, RequestPriority, RequestSpecConfig, RuntimeEnvelopeOverrides,
    SchedulePolicy, SchedulerConfig, SchedulerFull, SchedulerHandle,
    pick_chunked_prefill_size_for_hbm,
};
