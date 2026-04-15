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

/// Unified request-submission interface used by the HTTP layer.
pub trait RequestHandle: Send + Sync {
    fn submit(&self, req: IncomingRequest) -> Result<(), SubmitError>;
    fn model_id(&self) -> &str;
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
}
