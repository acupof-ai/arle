use std::sync::Arc;

use std::fmt;

use crate::scheduler::{IncomingRequest, SchedulerHandle};

/// Error returned when a request cannot be submitted to a runtime handle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SubmitError;

impl fmt::Display for SubmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "request submission failed")
    }
}

impl std::error::Error for SubmitError {}

/// DFlash runtime status exposed through the `/v1/models` endpoint.
///
/// `draft_model` and `speculative_tokens` are the DFlash init-time constants
/// (one process → one draft pair). `acceptance_rate` is the rolling rate read
/// from `ServerMetrics::dflash_acceptance_rate()` at response time — it is
/// `None` until at least one speculative block has executed.
#[derive(Clone, Debug, PartialEq)]
pub struct DflashStatus {
    pub draft_model: String,
    pub speculative_tokens: usize,
}

/// Unified request-submission interface used by the HTTP layer.
pub trait RequestHandle: Send + Sync {
    fn submit(&self, req: IncomingRequest) -> Result<(), SubmitError>;
    fn model_id(&self) -> &str;

    /// DFlash init-time metadata, if speculative decode is active for this
    /// runtime. Default `None` — CUDA and non-DFlash Metal paths return it
    /// unchanged. The Metal scheduler wrappers override this when a draft
    /// model was successfully loaded.
    fn dflash_status(&self) -> Option<DflashStatus> {
        None
    }
}

impl RequestHandle for SchedulerHandle {
    fn submit(&self, req: IncomingRequest) -> Result<(), SubmitError> {
        SchedulerHandle::submit(self, req).map_err(|_| SubmitError)
    }

    fn model_id(&self) -> &str {
        SchedulerHandle::model_id(self)
    }
}

impl<T> RequestHandle for Arc<T>
where
    T: RequestHandle + ?Sized,
{
    fn submit(&self, req: IncomingRequest) -> Result<(), SubmitError> {
        (**self).submit(req)
    }

    fn model_id(&self) -> &str {
        (**self).model_id()
    }

    fn dflash_status(&self) -> Option<DflashStatus> {
        (**self).dflash_status()
    }
}
