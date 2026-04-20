use crate::http_server::sessions::{LoadResponseBody, SessionSnapshot, SessionSnapshotError};
use crate::types::BlockFingerprint;

/// High-level session persistence contract for HTTP session routes.
///
/// Implementations own the persistence mechanics. The HTTP layer only
/// asks for a snapshot save/load operation and never reaches into engine
/// internals such as disk stores or radix caches.
pub trait SessionPersistence: Send {
    fn save_session_snapshot(
        &self,
        _session_id: &str,
        _fingerprints: Option<&[BlockFingerprint]>,
    ) -> Result<SessionSnapshot, SessionSnapshotError> {
        Err(SessionSnapshotError::Unsupported(
            "session persistence is not implemented for this engine",
        ))
    }

    fn load_session_snapshot(
        &mut self,
        _snapshot: &SessionSnapshot,
    ) -> Result<LoadResponseBody, SessionSnapshotError> {
        Err(SessionSnapshotError::Unsupported(
            "session persistence is not implemented for this engine",
        ))
    }
}
