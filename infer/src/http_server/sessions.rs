use std::io;
use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{
        DefaultBodyLimit, Path, Request as AxumRequest, State,
        rejection::{BytesRejection, JsonRejection},
    },
    http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::RwLock;

use crate::session_persistence::SessionPersistence;
use crate::types::BlockFingerprint;

#[cfg(test)]
use crate::kv_tier::transport::DiskStore;
#[cfg(test)]
use crate::kv_tier::transport::disk::DiskBlockLocation;
#[cfg(test)]
use crate::prefix_cache::{BlockId, RadixCache, ReconcileReport};
#[cfg(test)]
use std::collections::HashMap;
#[cfg(test)]
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSnapshot {
    pub version: u16,
    pub session_id: String,
    pub kv_format_tag: u8,
    pub radix_bytes: Vec<u8>,
    pub persisted_blocks: Vec<PersistedBlockEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersistedBlockEntry {
    pub fingerprint_hex: String,
    pub location: SerializablePath,
    pub payload_len: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SerializablePath(pub String);

#[derive(Debug, Error)]
pub enum SessionSnapshotError {
    #[error("session snapshot I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("failed to serialize radix snapshot: {0}")]
    SerializeRadix(serde_json::Error),
    #[error("failed to deserialize radix snapshot: {0}")]
    DeserializeRadix(serde_json::Error),
    #[error("failed to serialize session manifest: {0}")]
    SerializeManifest(String),
    #[error("failed to deserialize session manifest: {0}")]
    DeserializeManifest(String),
    #[error("missing disk block {fingerprint_hex} at {path}")]
    MissingDiskBlock {
        fingerprint_hex: String,
        path: String,
    },
    #[error("disk block mismatch for {fingerprint_hex} at {path}: {reason}")]
    DiskBlockMismatch {
        fingerprint_hex: String,
        path: String,
        reason: String,
    },
    /// Loading a snapshot whose `kv_format_tag` does not match the
    /// currently-live pool. Covers two real hazards: (a) restoring a
    /// BF16 session into an INT8 pool, which would silently
    /// corrupt decode; (b) bumping the stable tag scheme and then
    /// reloading an older session snapshot. M4 review finding D1.
    #[error("kv format mismatch: snapshot={snapshot} live_pool={live}")]
    FormatMismatch { snapshot: u8, live: u8 },
    /// The caller's `allocate_block_id` callback ran out of capacity
    /// mid-restore (returned `None`). Covers M4 review finding D2 —
    /// pool exhaustion at reload time was previously a silent panic
    /// path. Now surfaces as a structured error so the HTTP wrapper
    /// can translate it into a 503 / retry.
    #[error("pool exhausted while minting block id for {fingerprint_hex}")]
    PoolExhausted { fingerprint_hex: String },
    #[error("{0}")]
    Unsupported(&'static str),
}

#[cfg(test)]
pub struct LoadedSession {
    pub radix: RadixCache,
    pub kv_payloads: HashMap<BlockFingerprint, Vec<u8>>,
    pub report: ReconcileReport,
}

#[cfg(test)]
fn load_snapshot_payloads(
    snapshot: &SessionSnapshot,
    expected_kv_format_tag: u8,
    disk: &DiskStore,
) -> Result<(RadixCache, HashMap<BlockFingerprint, Vec<u8>>), SessionSnapshotError> {
    if snapshot.version != 1 {
        return Err(SessionSnapshotError::DeserializeManifest(format!(
            "unsupported session snapshot version {}",
            snapshot.version
        )));
    }

    if snapshot.kv_format_tag != expected_kv_format_tag {
        return Err(SessionSnapshotError::FormatMismatch {
            snapshot: snapshot.kv_format_tag,
            live: expected_kv_format_tag,
        });
    }

    let radix: RadixCache = serde_json::from_slice(&snapshot.radix_bytes)
        .map_err(SessionSnapshotError::DeserializeRadix)?;
    let mut kv_payloads = HashMap::with_capacity(snapshot.persisted_blocks.len());

    for entry in &snapshot.persisted_blocks {
        let fingerprint = parse_fingerprint_hex(&entry.fingerprint_hex)?;
        let location = DiskBlockLocation {
            path: PathBuf::from(&entry.location.0),
            payload_len: entry.payload_len,
            fingerprint,
        };
        let payload = match disk.get_block(&location, Some(fingerprint)) {
            Ok(payload) => payload,
            Err(err) if err.kind() == io::ErrorKind::NotFound => {
                return Err(SessionSnapshotError::MissingDiskBlock {
                    fingerprint_hex: entry.fingerprint_hex.clone(),
                    path: entry.location.0.clone(),
                });
            }
            Err(err) if err.kind() == io::ErrorKind::InvalidData => {
                return Err(SessionSnapshotError::DiskBlockMismatch {
                    fingerprint_hex: entry.fingerprint_hex.clone(),
                    path: entry.location.0.clone(),
                    reason: err.to_string(),
                });
            }
            Err(err) => return Err(SessionSnapshotError::Io(err)),
        };

        let payload_len = payload.len() as u64;
        if payload_len != entry.payload_len {
            return Err(SessionSnapshotError::DiskBlockMismatch {
                fingerprint_hex: entry.fingerprint_hex.clone(),
                path: entry.location.0.clone(),
                reason: format!(
                    "payload length mismatch: manifest={} disk={payload_len}",
                    entry.payload_len
                ),
            });
        }

        kv_payloads.insert(fingerprint, payload);
    }

    Ok((radix, kv_payloads))
}

#[cfg(test)]
fn reconcile_loaded_session<F>(
    snapshot: &SessionSnapshot,
    mut radix: RadixCache,
    kv_payloads: HashMap<BlockFingerprint, Vec<u8>>,
    mut allocate_block_id: F,
) -> Result<LoadedSession, SessionSnapshotError>
where
    F: FnMut(BlockFingerprint) -> Option<BlockId>,
{
    let mut known = HashMap::with_capacity(snapshot.persisted_blocks.len());

    for entry in &snapshot.persisted_blocks {
        let fingerprint = parse_fingerprint_hex(&entry.fingerprint_hex)?;
        let block_id =
            allocate_block_id(fingerprint).ok_or_else(|| SessionSnapshotError::PoolExhausted {
                fingerprint_hex: entry.fingerprint_hex.clone(),
            })?;
        known.insert(fingerprint, block_id);
    }

    let report = radix.reconcile(&known);
    Ok(LoadedSession {
        radix,
        kv_payloads,
        report,
    })
}

#[cfg(test)]
pub fn save_session<F>(
    session_id: &str,
    kv_format_tag: u8,
    radix: &RadixCache,
    disk: &DiskStore,
    mut payload_for: F,
    fingerprints: &[BlockFingerprint],
) -> Result<SessionSnapshot, SessionSnapshotError>
where
    F: FnMut(BlockFingerprint) -> Option<Vec<u8>>,
{
    let radix_bytes = serde_json::to_vec(radix).map_err(SessionSnapshotError::SerializeRadix)?;
    let mut persisted_blocks = Vec::new();

    for &fingerprint in fingerprints {
        let Some(payload) = payload_for(fingerprint) else {
            continue;
        };

        let location = disk.put_block(fingerprint, kv_format_tag, &payload)?;
        persisted_blocks.push(PersistedBlockEntry {
            fingerprint_hex: fingerprint_to_hex(fingerprint),
            location: SerializablePath(location.path.to_string_lossy().into_owned()),
            payload_len: location.payload_len,
        });
    }

    Ok(SessionSnapshot {
        version: 1,
        session_id: session_id.to_string(),
        kv_format_tag,
        radix_bytes,
        persisted_blocks,
    })
}

#[cfg(test)]
pub fn load_session<F>(
    snapshot: &SessionSnapshot,
    expected_kv_format_tag: u8,
    disk: &DiskStore,
    allocate_block_id: F,
) -> Result<LoadedSession, SessionSnapshotError>
where
    F: FnMut(BlockFingerprint) -> Option<BlockId>,
{
    let (radix, kv_payloads) = load_snapshot_payloads(snapshot, expected_kv_format_tag, disk)?;
    reconcile_loaded_session(snapshot, radix, kv_payloads, allocate_block_id)
}

#[cfg(test)]
fn fingerprint_to_hex(fingerprint: BlockFingerprint) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    let mut out = String::with_capacity(32);
    for byte in fingerprint.0 {
        out.push(char::from(HEX[(byte >> 4) as usize]));
        out.push(char::from(HEX[(byte & 0x0f) as usize]));
    }
    out
}

fn parse_fingerprint_hex(hex: &str) -> Result<BlockFingerprint, SessionSnapshotError> {
    if hex.len() != 32 {
        return Err(SessionSnapshotError::DeserializeManifest(format!(
            "fingerprint hex must be 32 chars, got {}",
            hex.len()
        )));
    }

    let mut bytes = [0u8; 16];
    for (idx, chunk) in hex.as_bytes().chunks_exact(2).enumerate() {
        let hi = decode_hex_nibble(chunk[0]).ok_or_else(|| {
            SessionSnapshotError::DeserializeManifest(format!(
                "invalid fingerprint hex nibble '{}'",
                char::from(chunk[0])
            ))
        })?;
        let lo = decode_hex_nibble(chunk[1]).ok_or_else(|| {
            SessionSnapshotError::DeserializeManifest(format!(
                "invalid fingerprint hex nibble '{}'",
                char::from(chunk[1])
            ))
        })?;
        bytes[idx] = (hi << 4) | lo;
    }

    Ok(BlockFingerprint(bytes))
}

fn decode_hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct SaveRequestBody {
    #[serde(default)]
    pub fingerprints: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadResponseBody {
    pub remapped: usize,
    pub tombstoned: usize,
    pub orphans_cleared: usize,
    pub kv_payloads: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ErrorBody {
    pub error: String,
    pub kind: &'static str,
    pub expected: Option<u8>,
    pub got: Option<u8>,
}

#[derive(Debug)]
struct SessionRouteError {
    status: StatusCode,
    body: ErrorBody,
    headers: Vec<(HeaderName, HeaderValue)>,
}

impl SessionRouteError {
    #[must_use]
    fn with_header(mut self, name: HeaderName, value: HeaderValue) -> Self {
        self.headers.push((name, value));
        self
    }
}

impl IntoResponse for SessionRouteError {
    fn into_response(self) -> Response {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json; charset=utf-8"),
        );
        for (name, value) in self.headers {
            headers.insert(name, value);
        }
        (self.status, headers, Json(self.body)).into_response()
    }
}

fn error_body(
    error: impl Into<String>,
    kind: &'static str,
    expected: Option<u8>,
    got: Option<u8>,
) -> ErrorBody {
    ErrorBody {
        error: error.into(),
        kind,
        expected,
        got,
    }
}

fn session_error(
    status: StatusCode,
    error: impl Into<String>,
    kind: &'static str,
    expected: Option<u8>,
    got: Option<u8>,
) -> SessionRouteError {
    SessionRouteError {
        status,
        body: error_body(error, kind, expected, got),
        headers: Vec::new(),
    }
}

fn snapshot_error_to_response(err: &SessionSnapshotError) -> SessionRouteError {
    let (status, kind, expected, got) = match err {
        SessionSnapshotError::Io(_) => (StatusCode::INTERNAL_SERVER_ERROR, "io", None, None),
        SessionSnapshotError::SerializeRadix(_) | SessionSnapshotError::SerializeManifest(_) => {
            (StatusCode::INTERNAL_SERVER_ERROR, "serialize", None, None)
        }
        SessionSnapshotError::DeserializeRadix(_)
        | SessionSnapshotError::DeserializeManifest(_) => {
            (StatusCode::BAD_REQUEST, "deserialize", None, None)
        }
        SessionSnapshotError::MissingDiskBlock { .. } => {
            (StatusCode::NOT_FOUND, "missing_block", None, None)
        }
        SessionSnapshotError::DiskBlockMismatch { .. } => {
            (StatusCode::UNPROCESSABLE_ENTITY, "tampered", None, None)
        }
        SessionSnapshotError::FormatMismatch { snapshot, live } => (
            StatusCode::BAD_REQUEST,
            "format_mismatch",
            Some(*live),
            Some(*snapshot),
        ),
        SessionSnapshotError::PoolExhausted { .. } => (
            StatusCode::SERVICE_UNAVAILABLE,
            "pool_exhausted",
            None,
            None,
        ),
        SessionSnapshotError::Unsupported(_) => {
            (StatusCode::NOT_IMPLEMENTED, "unsupported", None, None)
        }
    };

    session_error(status, err.to_string(), kind, expected, got)
}

fn unsupported_response(message: impl Into<String>) -> SessionRouteError {
    session_error(
        StatusCode::NOT_IMPLEMENTED,
        message.into(),
        "unsupported",
        None,
        None,
    )
}

fn route_not_found_response(path: &str) -> SessionRouteError {
    session_error(
        StatusCode::NOT_FOUND,
        format!("Route `{path}` was not found"),
        "not_found",
        None,
        None,
    )
}

fn allow_header_value_for_path(path: &str) -> Option<HeaderValue> {
    let allow = if path.ends_with("/save") || path.ends_with("/load") {
        "POST"
    } else if path.ends_with("/manifest") {
        "GET, HEAD"
    } else if path.starts_with('/') && path.matches('/').count() == 1 {
        "DELETE"
    } else {
        return None;
    };
    Some(HeaderValue::from_static(allow))
}

fn method_not_allowed_response(method: &Method, path: &str) -> SessionRouteError {
    let error = session_error(
        StatusCode::METHOD_NOT_ALLOWED,
        format!("Method `{method}` is not allowed for `{path}`"),
        "method_not_allowed",
        None,
        None,
    );
    if let Some(allow) = allow_header_value_for_path(path) {
        error.with_header(header::ALLOW, allow)
    } else {
        error
    }
}

async fn route_not_found_handler(request: AxumRequest) -> SessionRouteError {
    route_not_found_response(request.uri().path())
}

async fn method_not_allowed_handler(request: AxumRequest) -> SessionRouteError {
    method_not_allowed_response(request.method(), request.uri().path())
}

fn bytes_rejection_to_response(err: &BytesRejection) -> SessionRouteError {
    let status = err.status();
    let body_text = err.body_text();
    let kind = if status == StatusCode::PAYLOAD_TOO_LARGE {
        "payload_too_large"
    } else {
        "invalid_body"
    };
    session_error(status, body_text, kind, None, None)
}

fn json_rejection_to_response(err: JsonRejection) -> SessionRouteError {
    match err {
        JsonRejection::MissingJsonContentType(_) => session_error(
            StatusCode::BAD_REQUEST,
            "Expected `Content-Type: application/json`",
            "invalid_json",
            None,
            None,
        ),
        JsonRejection::JsonSyntaxError(inner) => session_error(
            StatusCode::BAD_REQUEST,
            format!("Malformed JSON request body: {inner}"),
            "invalid_json",
            None,
            None,
        ),
        JsonRejection::JsonDataError(inner) => session_error(
            StatusCode::BAD_REQUEST,
            format!("Invalid JSON request body: {inner}"),
            "invalid_json",
            None,
            None,
        ),
        JsonRejection::BytesRejection(inner) => bytes_rejection_to_response(&inner),
        other => session_error(
            StatusCode::BAD_REQUEST,
            format!("Failed to decode JSON request body: {other}"),
            "invalid_json",
            None,
            None,
        ),
    }
}

fn parse_json_payload<T>(payload: Result<Json<T>, JsonRejection>) -> Result<T, SessionRouteError> {
    payload
        .map(|Json(value)| value)
        .map_err(json_rejection_to_response)
}

async fn handle_save<E: SessionPersistence>(
    Path(session_id): Path<String>,
    State(engine): State<Arc<RwLock<E>>>,
    payload: Result<Json<SaveRequestBody>, JsonRejection>,
) -> Result<Json<SessionSnapshot>, SessionRouteError> {
    let req = parse_json_payload(payload)?;
    let engine = engine.read().await;
    let fingerprints = if req.fingerprints.is_empty() {
        None
    } else {
        Some(
            req.fingerprints
                .iter()
                .map(|fingerprint| parse_fingerprint_hex(fingerprint))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|err| snapshot_error_to_response(&err))?,
        )
    };

    let snapshot = engine
        .save_session_snapshot(&session_id, fingerprints.as_deref())
        .map_err(|err| snapshot_error_to_response(&err))?;
    Ok(Json(snapshot))
}

async fn handle_load<E: SessionPersistence>(
    State(engine): State<Arc<RwLock<E>>>,
    payload: Result<Json<SessionSnapshot>, JsonRejection>,
) -> Result<Json<LoadResponseBody>, SessionRouteError> {
    let snapshot = parse_json_payload(payload)?;
    let mut engine = engine.write().await;
    let loaded = engine
        .load_session_snapshot(&snapshot)
        .map_err(|err| snapshot_error_to_response(&err))?;

    Ok(Json(loaded))
}

async fn handle_manifest(Path(_session_id): Path<String>) -> SessionRouteError {
    // TODO(2026-04-16): persist session manifests so GET /manifest can replay
    // the last saved snapshot for this session id.
    unsupported_response("session manifest persistence is not implemented yet")
}

async fn handle_delete(Path(_session_id): Path<String>) -> SessionRouteError {
    // TODO(2026-04-16): delete persisted session manifests and backing blocks
    // once the manifest storage path is wired.
    unsupported_response("session deletion is not implemented yet")
}

pub fn session_router<E>(engine: Arc<RwLock<E>>) -> Router
where
    E: SessionPersistence + Send + Sync + 'static,
{
    Router::new()
        .route("/{session_id}/save", post(handle_save::<E>))
        .route("/{session_id}/load", post(handle_load::<E>))
        .route("/{session_id}/manifest", get(handle_manifest))
        .route("/{session_id}", delete(handle_delete))
        .method_not_allowed_fallback(method_not_allowed_handler)
        .fallback(route_not_found_handler)
        .layer(DefaultBodyLimit::max(super::HTTP_REQUEST_BODY_LIMIT_BYTES))
        .with_state(engine)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;

    use axum::Router;
    use axum::body::{Body, to_bytes};
    use axum::http::{Request, StatusCode};
    use tempfile::TempDir;
    use tempfile::tempdir;
    use tokio::sync::RwLock;
    use tower::util::ServiceExt;

    use super::{
        LoadResponseBody, SaveRequestBody, SessionSnapshot, SessionSnapshotError, load_session,
        save_session, session_router,
    };
    use crate::kv_tier::transport::DiskStore;
    use crate::prefix_cache::{BlockId, RadixCache};
    use crate::session_persistence::SessionPersistence;
    use crate::types::BlockFingerprint;

    struct MockEngine {
        radix: RadixCache,
        disk: DiskStore,
        payloads: HashMap<BlockFingerprint, Vec<u8>>,
        next_block_id: u32,
        format_tag: u8,
        fingerprints: Vec<BlockFingerprint>,
    }

    impl SessionPersistence for MockEngine {
        fn save_session_snapshot(
            &self,
            session_id: &str,
            fingerprints: Option<&[BlockFingerprint]>,
        ) -> Result<SessionSnapshot, SessionSnapshotError> {
            let fingerprints: Vec<BlockFingerprint> =
                fingerprints.map_or_else(|| self.fingerprints.clone(), |items| items.to_vec());
            save_session(
                session_id,
                self.format_tag,
                &self.radix,
                &self.disk,
                |fingerprint| self.payloads.get(&fingerprint).cloned(),
                &fingerprints,
            )
        }

        fn load_session_snapshot(
            &mut self,
            snapshot: &SessionSnapshot,
        ) -> Result<LoadResponseBody, SessionSnapshotError> {
            let mut next_block_id = self.next_block_id;
            let loaded = load_session(snapshot, self.format_tag, &self.disk, |_| {
                let block_id = BlockId(next_block_id);
                next_block_id += 1;
                Some(block_id)
            })?;
            self.next_block_id = next_block_id;
            self.radix = loaded.radix;
            self.payloads = loaded.kv_payloads.clone();
            Ok(LoadResponseBody {
                remapped: loaded.report.remapped,
                tombstoned: loaded.report.tombstoned,
                orphans_cleared: loaded.report.orphans_cleared,
                kv_payloads: loaded.kv_payloads.len(),
            })
        }
    }

    fn mock_engine(tempdir: &TempDir, format_tag: u8) -> MockEngine {
        let disk = DiskStore::new(tempdir.path());
        let mut radix = RadixCache::new(4);
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        let fp_a = BlockFingerprint([0x91; 16]);
        let fp_b = BlockFingerprint([0x92; 16]);
        radix.insert_with_fingerprints(&tokens, &[BlockId(10), BlockId(20)], &[fp_a, fp_b]);

        MockEngine {
            radix,
            disk,
            payloads: HashMap::from([(fp_a, b"payload-a".to_vec()), (fp_b, b"payload-b".to_vec())]),
            next_block_id: 100,
            format_tag,
            fingerprints: vec![fp_a, fp_b],
        }
    }

    async fn response_json<T: serde::de::DeserializeOwned>(
        response: axum::response::Response,
    ) -> T {
        let bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read response body");
        serde_json::from_slice(&bytes).expect("deserialize response json")
    }

    fn mounted_session_app(engine: MockEngine) -> Router {
        Router::new().nest_service(
            "/v1/sessions",
            session_router(Arc::new(RwLock::new(engine))),
        )
    }

    #[test]
    fn save_then_load_round_trips_radix_and_payloads() {
        let dir = tempdir().expect("tempdir");
        let disk = crate::kv_tier::transport::DiskStore::new(dir.path());
        let mut radix = RadixCache::new(4);
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        let fp_a = BlockFingerprint([0x11; 16]);
        let fp_b = BlockFingerprint([0x22; 16]);
        let original_blocks = [BlockId(10), BlockId(20)];
        radix.insert_with_fingerprints(&tokens, &original_blocks, &[fp_a, fp_b]);

        let expected_payloads =
            HashMap::from([(fp_a, b"payload-a".to_vec()), (fp_b, b"payload-b".to_vec())]);
        let snapshot = save_session(
            "session-1",
            7,
            &radix,
            &disk,
            |fingerprint| expected_payloads.get(&fingerprint).cloned(),
            &[fp_a, fp_b],
        )
        .expect("save session");

        let mut next_block_id = 100u32;
        let mut loaded = load_session(&snapshot, 7, &disk, |_| {
            let block_id = BlockId(next_block_id);
            next_block_id += 1;
            Some(block_id)
        })
        .expect("load session");

        assert_eq!(loaded.report.remapped, 2);
        assert_eq!(loaded.report.tombstoned, 0);
        assert_eq!(loaded.kv_payloads, expected_payloads);

        let (matched_len, blocks) = loaded.radix.lookup(&tokens);
        assert_eq!(matched_len, 8);
        assert_eq!(blocks, vec![BlockId(100), BlockId(101)]);
        assert_ne!(blocks, original_blocks);
    }

    #[test]
    fn save_skips_blocks_with_no_payload() {
        let dir = tempdir().expect("tempdir");
        let disk = crate::kv_tier::transport::DiskStore::new(dir.path());
        let mut radix = RadixCache::new(4);
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        let fp_a = BlockFingerprint([0x33; 16]);
        let fp_b = BlockFingerprint([0x44; 16]);
        radix.insert_with_fingerprints(&tokens, &[BlockId(1), BlockId(2)], &[fp_a, fp_b]);

        let snapshot = save_session(
            "session-2",
            9,
            &radix,
            &disk,
            |fingerprint| (fingerprint == fp_a).then(|| b"payload-a".to_vec()),
            &[fp_a, fp_b],
        )
        .expect("save session");

        assert_eq!(snapshot.persisted_blocks.len(), 1);
        assert_eq!(snapshot.persisted_blocks[0].fingerprint_hex.len(), 32);

        let mut loaded =
            load_session(&snapshot, 9, &disk, |_| Some(BlockId(77))).expect("load session");
        assert_eq!(loaded.report.remapped, 1);
        assert_eq!(loaded.report.tombstoned, 1);
        assert_eq!(loaded.kv_payloads.len(), 1);
        assert_eq!(loaded.kv_payloads.get(&fp_a), Some(&b"payload-a".to_vec()));
        assert!(!loaded.kv_payloads.contains_key(&fp_b));

        let (matched_len, blocks) = loaded.radix.lookup(&tokens);
        assert_eq!(matched_len, 4);
        assert_eq!(blocks, vec![BlockId(77)]);
    }

    #[test]
    fn load_errors_on_tampered_disk_payload() {
        let dir = tempdir().expect("tempdir");
        let disk = crate::kv_tier::transport::DiskStore::new(dir.path());
        let mut radix = RadixCache::new(4);
        let tokens = [1, 2, 3, 4];
        let fingerprint = BlockFingerprint([0x55; 16]);
        radix.insert_with_fingerprints(&tokens, &[BlockId(9)], &[fingerprint]);

        let snapshot = save_session(
            "session-3",
            3,
            &radix,
            &disk,
            |fp| (fp == fingerprint).then(|| b"payload".to_vec()),
            &[fingerprint],
        )
        .expect("save session");

        let disk_path = PathBuf::from(&snapshot.persisted_blocks[0].location.0);
        fs::write(&disk_path, b"wrong-bytes").expect("tamper disk payload");

        match load_session(&snapshot, 3, &disk, |_| Some(BlockId(99))) {
            Err(SessionSnapshotError::DiskBlockMismatch { .. }) => {}
            Err(other) => panic!("expected DiskBlockMismatch, got {other:?}"),
            Ok(_) => panic!("tampered payload should fail"),
        }
    }

    #[test]
    fn load_rejects_kv_format_tag_mismatch() {
        let dir = tempdir().expect("tempdir");
        let disk = crate::kv_tier::transport::DiskStore::new(dir.path());
        let mut radix = RadixCache::new(4);
        let tokens = [1, 2, 3, 4];
        let fingerprint = BlockFingerprint([0x66; 16]);
        radix.insert_with_fingerprints(&tokens, &[BlockId(5)], &[fingerprint]);

        // Saved under tag=1 (BF16)
        let snapshot = save_session(
            "session-fmt",
            1,
            &radix,
            &disk,
            |fp| (fp == fingerprint).then(|| b"payload".to_vec()),
            &[fingerprint],
        )
        .expect("save session");

        // Live pool is tag=3 (INT8). Must refuse.
        match load_session(&snapshot, 3, &disk, |_| Some(BlockId(42))) {
            Err(SessionSnapshotError::FormatMismatch { snapshot, live }) => {
                assert_eq!(snapshot, 1);
                assert_eq!(live, 3);
            }
            Err(other) => panic!("expected FormatMismatch, got {other:?}"),
            Ok(_) => panic!("format-mismatched load should fail"),
        }
    }

    #[test]
    fn load_surfaces_pool_exhaustion() {
        let dir = tempdir().expect("tempdir");
        let disk = crate::kv_tier::transport::DiskStore::new(dir.path());
        let mut radix = RadixCache::new(4);
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        let fp_a = BlockFingerprint([0x77; 16]);
        let fp_b = BlockFingerprint([0x88; 16]);
        radix.insert_with_fingerprints(&tokens, &[BlockId(1), BlockId(2)], &[fp_a, fp_b]);

        let snapshot = save_session(
            "session-exhaust",
            1,
            &radix,
            &disk,
            |_| Some(b"payload".to_vec()),
            &[fp_a, fp_b],
        )
        .expect("save session");

        // Allocator runs out after the first block: fingerprint order
        // inside the snapshot isn't stable, so match against either
        // parsed fingerprint rather than hard-coding one.
        let mut minted = 0u32;
        match load_session(&snapshot, 1, &disk, |_| {
            if minted == 0 {
                minted = 1;
                Some(BlockId(500))
            } else {
                None
            }
        }) {
            Err(SessionSnapshotError::PoolExhausted { fingerprint_hex }) => {
                assert_eq!(fingerprint_hex.len(), 32);
            }
            Err(other) => panic!("expected PoolExhausted, got {other:?}"),
            Ok(_) => panic!("load with exhausted allocator should fail"),
        }
    }

    #[tokio::test]
    async fn save_route_returns_snapshot() {
        let dir = tempdir().expect("tempdir");
        let app = mounted_session_app(mock_engine(&dir, 1));

        let response = app
            .oneshot(
                Request::post("/v1/sessions/session-save/save")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&SaveRequestBody::default())
                            .expect("serialize save body"),
                    ))
                    .expect("build save request"),
            )
            .await
            .expect("save route response");

        assert_eq!(response.status(), StatusCode::OK);
        let snapshot: SessionSnapshot = response_json(response).await;
        assert_eq!(snapshot.session_id, "session-save");
        assert!(!snapshot.persisted_blocks.is_empty());
        assert!(!snapshot.radix_bytes.is_empty());
    }

    #[tokio::test]
    async fn load_route_refuses_format_mismatch() {
        let dir = tempdir().expect("tempdir");
        let source_engine = mock_engine(&dir, 1);
        let snapshot = save_session(
            "session-fmt",
            source_engine.format_tag,
            &source_engine.radix,
            &source_engine.disk,
            |fingerprint| source_engine.payloads.get(&fingerprint).cloned(),
            &source_engine.fingerprints,
        )
        .expect("save snapshot");
        let app = mounted_session_app(mock_engine(&dir, 3));

        let response = app
            .oneshot(
                Request::post("/v1/sessions/session-fmt/load")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&snapshot).expect("serialize snapshot"),
                    ))
                    .expect("build load request"),
            )
            .await
            .expect("load route response");

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body: serde_json::Value = response_json(response).await;
        assert_eq!(body["kind"], "format_mismatch");
        assert_eq!(body["expected"], 3);
        assert_eq!(body["got"], 1);
    }

    #[tokio::test]
    async fn load_route_refuses_tampered_payload() {
        let dir = tempdir().expect("tempdir");
        let source_engine = mock_engine(&dir, 1);
        let snapshot = save_session(
            "session-tampered",
            source_engine.format_tag,
            &source_engine.radix,
            &source_engine.disk,
            |fingerprint| source_engine.payloads.get(&fingerprint).cloned(),
            &source_engine.fingerprints,
        )
        .expect("save snapshot");
        let tampered_path = PathBuf::from(&snapshot.persisted_blocks[0].location.0);
        fs::write(&tampered_path, b"junk").expect("overwrite persisted block");

        let app = mounted_session_app(mock_engine(&dir, 1));
        let response = app
            .oneshot(
                Request::post("/v1/sessions/session-tampered/load")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&snapshot).expect("serialize snapshot"),
                    ))
                    .expect("build load request"),
            )
            .await
            .expect("load route response");

        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body: serde_json::Value = response_json(response).await;
        assert_eq!(body["kind"], "tampered");
    }

    #[tokio::test]
    async fn request_body_limit_enforced() {
        let dir = tempdir().expect("tempdir");
        let app = mounted_session_app(mock_engine(&dir, 1));
        let oversized_snapshot = SessionSnapshot {
            version: 1,
            session_id: "session-big".to_string(),
            kv_format_tag: 1,
            radix_bytes: vec![7u8; 10 * 1024 * 1024],
            persisted_blocks: Vec::new(),
        };

        let response = app
            .oneshot(
                Request::post("/v1/sessions/session-big/load")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&oversized_snapshot)
                            .expect("serialize oversized snapshot"),
                    ))
                    .expect("build oversized request"),
            )
            .await
            .expect("body-limit response");

        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let body: serde_json::Value = response_json(response).await;
        assert_eq!(body["kind"], "payload_too_large");
    }

    #[tokio::test]
    async fn load_route_rejects_malformed_json_with_structured_error() {
        let dir = tempdir().expect("tempdir");
        let app = mounted_session_app(mock_engine(&dir, 1));

        let response = app
            .oneshot(
                Request::post("/v1/sessions/session-bad/load")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"version":"oops"}"#))
                    .expect("build malformed request"),
            )
            .await
            .expect("malformed-json response");

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body: serde_json::Value = response_json(response).await;
        assert_eq!(body["kind"], "invalid_json");
    }

    #[tokio::test]
    async fn save_route_requires_json_content_type() {
        let dir = tempdir().expect("tempdir");
        let app = mounted_session_app(mock_engine(&dir, 1));

        let response = app
            .oneshot(
                Request::post("/v1/sessions/session-save/save")
                    .body(Body::from("{}"))
                    .expect("build missing-content-type request"),
            )
            .await
            .expect("missing-content-type response");

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body: serde_json::Value = response_json(response).await;
        assert_eq!(body["kind"], "invalid_json");
        assert!(
            body["error"]
                .as_str()
                .is_some_and(|error| error.contains("Content-Type")),
            "body={body}"
        );
    }

    #[tokio::test]
    async fn session_unknown_route_returns_structured_not_found() {
        let dir = tempdir().expect("tempdir");
        let app = mounted_session_app(mock_engine(&dir, 1));

        let response = app
            .oneshot(
                Request::get("/v1/sessions/session-missing/unknown")
                    .body(Body::empty())
                    .expect("build unknown-route request"),
            )
            .await
            .expect("unknown-route response");

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body: serde_json::Value = response_json(response).await;
        assert_eq!(body["kind"], "not_found");
        assert_eq!(
            body["error"],
            "Route `/session-missing/unknown` was not found"
        );
    }

    #[tokio::test]
    async fn session_wrong_method_returns_structured_method_not_allowed() {
        let dir = tempdir().expect("tempdir");
        let app = mounted_session_app(mock_engine(&dir, 1));

        let response = app
            .oneshot(
                Request::get("/v1/sessions/session-load/load")
                    .body(Body::empty())
                    .expect("build wrong-method request"),
            )
            .await
            .expect("wrong-method response");

        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
        assert_eq!(response.headers()["allow"], "POST");
        let body: serde_json::Value = response_json(response).await;
        assert_eq!(body["kind"], "method_not_allowed");
        assert_eq!(
            body["error"],
            "Method `GET` is not allowed for `/session-load/load`"
        );
    }

    #[tokio::test]
    async fn session_get_route_method_errors_include_allow_header() {
        let dir = tempdir().expect("tempdir");
        let app = mounted_session_app(mock_engine(&dir, 1));

        let response = app
            .oneshot(
                Request::post("/v1/sessions/session-manifest/manifest")
                    .body(Body::empty())
                    .expect("build manifest wrong-method request"),
            )
            .await
            .expect("manifest wrong-method response");

        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
        assert_eq!(response.headers()["allow"], "GET, HEAD");
    }
}
