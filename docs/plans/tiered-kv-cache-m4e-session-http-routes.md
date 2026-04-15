# Tiered KV Cache M4e — session HTTP routes

**Status**: Execution plan. Locally blocked (needs scheduler-owned
runtime state to wire the payload accessor); intended to be
executed in the remote L4 window after M4 a/b/c/d remote
acceptance signs off.

**Scope**: wrap the M4d pure-Rust `save_session` / `load_session`
functions in axum handlers and mount them on the existing `/v1`
router alongside `/v1/completions` + `/v1/chat/completions`. No new
radix / disk / fingerprint semantics — this batch is exclusively
HTTP plumbing on top of the already-shipped M4 core.

**Out of scope** for M4e (punt to a later batch):
- T1→T2 real byte spill through the coordinator (see
  `tiered-kv-cache-coordinator-real-byte-path.md`).
- Auth beyond the existing `AGENT_INFER_API_KEY` bearer gate.
- Streaming / chunked upload of very large snapshots.

---

## 1 · Surface

### Routes

| Method | Path | Body | Response | Error |
|--------|------|------|----------|-------|
| `POST` | `/v1/sessions/{id}/save` | `{ "fingerprints": ["<hex>", ...] }` (optional — empty = save all known) | `SessionSnapshot` JSON | 500 on disk IO, 409 if session actively decoding |
| `POST` | `/v1/sessions/{id}/load` | `SessionSnapshot` JSON | `{ "remapped": u, "tombstoned": u, "orphans_cleared": u, "kv_payloads": u }` | 400 format mismatch, 404 missing block, 422 tampered, 503 pool exhausted |
| `GET`  | `/v1/sessions/{id}/manifest` | — | last saved `SessionSnapshot` or 404 | — |
| `DELETE` | `/v1/sessions/{id}` | — | `{ "deleted_blocks": u }` | — |

All routes honor the existing `Authorization: Bearer <AGENT_INFER_API_KEY>`
gate when configured. Request bodies capped at 16 MiB via
`axum::extract::DefaultBodyLimit::max(16 * 1024 * 1024)` on the
sessions sub-router.

### Error mapping

`SessionSnapshotError` → HTTP status:

- `Io(_)` → 500
- `SerializeRadix(_)` / `SerializeManifest(_)` → 500
- `DeserializeRadix(_)` / `DeserializeManifest(_)` → 400
- `MissingDiskBlock { .. }` → 404
- `DiskBlockMismatch { .. }` → 422 (payload tampered or header invalid)
- `FormatMismatch { .. }` → 400 with `{ "expected": u8, "got": u8 }`
  body
- `PoolExhausted { .. }` → 503 with a `Retry-After: 5` header

---

## 2 · Files to touch

### `infer/src/server_engine.rs` (or `infer/src/backend/runtime.rs`
if that's where the `InferenceEngine` trait lives — check first)

Add two trait methods:

```rust
pub trait InferenceEngine: Send + Sync {
    // ... existing methods ...

    /// Enumerate currently-published BlockFingerprints for the
    /// session scope the caller wants to persist. The default impl
    /// returns empty; implementors that own a live `RadixCache`
    /// override to walk their fingerprints.
    fn session_fingerprints(&self, session_id: &str) -> Vec<BlockFingerprint> {
        Vec::new()
    }

    /// Read the raw KV payload bytes backing one fingerprint, for
    /// save. `None` means the fingerprint is known but the bytes
    /// are not available (e.g. evicted from T0 and T1 does not yet
    /// have them). The default impl returns None.
    fn read_block_payload(&self, fingerprint: BlockFingerprint) -> Option<Vec<u8>> {
        None
    }

    /// Install restored KV payloads into a fresh pool, returning
    /// a `BlockId` allocator closure. The default impl returns a
    /// no-op that fails every allocation.
    fn install_restored_kv(
        &mut self,
        payloads: &HashMap<BlockFingerprint, Vec<u8>>,
    ) -> Box<dyn FnMut(BlockFingerprint) -> Option<BlockId>> {
        Box::new(|_| None)
    }

    /// The engine's current `kv_format_tag`. Enforced on load.
    fn kv_format_tag(&self) -> u8 {
        0
    }
}
```

CUDA scheduler impl overrides all four. Metal impl can leave the
defaults (M4 is CUDA-only until the Metal wired-memory bindings
batch lands).

### `infer/src/scheduler/cuda/core.rs`

Implement the four trait methods on `Scheduler<M>`:

- `session_fingerprints`: walk `self.prefix_cache` and collect
  every `Node::fingerprint.is_some()` that has the matching
  `session_id`. Needs a new `pub fn fingerprints_for_session(&self,
  session_id: &SessionId) -> Vec<BlockFingerprint>` on `RadixCache`
  (keep visibility `pub(crate)` if you don't want to leak it).
- `read_block_payload`: look up `self.block_to_pages.get(bid)`
  (where `bid` is derived from the fingerprint via
  `self.prefix_cache.block_index` lookup), then `paged_kv_pool.
  copy_pages_to_host(&pages) -> Vec<u8>`. The copy_pages_to_host
  helper does NOT exist yet — add it as a thin wrapper around the
  existing page migration path. This is the only non-trivial new
  code on the scheduler side.
- `install_restored_kv`: alloc fresh pages via
  `paged_kv_pool.alloc_tokens`, write each payload to its page
  range, return a closure that pops from a prepared `Vec<BlockId>`.
- `kv_format_tag`: `self.paged_kv_pool.format.stable_tag().unwrap_or(0)`.

### `infer/src/http_server/sessions.rs`

Extend with the four axum handlers, wiring through the shared
engine handle:

```rust
pub fn session_router<E: InferenceEngine + 'static>() -> axum::Router<EngineHandle<E>> {
    Router::new()
        .route("/save", post(handle_save::<E>))
        .route("/load", post(handle_load::<E>))
        .route("/manifest", get(handle_get_manifest::<E>))
        .route("/", delete(handle_delete::<E>))
        .layer(DefaultBodyLimit::max(16 * 1024 * 1024))
}

async fn handle_save<E: InferenceEngine>(
    State(engine): State<EngineHandle<E>>,
    Path(session_id): Path<String>,
    Json(req): Json<SaveRequest>,
) -> Result<Json<SessionSnapshot>, (StatusCode, Json<ErrorBody>)> {
    let engine = engine.read().await;
    let disk = engine.disk_store();
    let fingerprints = if req.fingerprints.is_empty() {
        engine.session_fingerprints(&session_id)
    } else {
        req.fingerprints.iter()
            .map(|hex| parse_fingerprint_hex(hex))
            .collect::<Result<Vec<_>, _>>()
            .map_err(bad_request)?
    };
    let snapshot = save_session(
        &session_id,
        engine.kv_format_tag(),
        engine.radix_cache(),
        disk,
        |fp| engine.read_block_payload(fp),
        &fingerprints,
    )
    .map_err(snapshot_error_to_status)?;
    Ok(Json(snapshot))
}
```

...and similarly for load / manifest / delete.

The `EngineHandle` + `disk_store()` + `radix_cache()` accessors
are what the axum side needs but don't exist yet on the engine
trait. Plumb them as needed — probably via an `Arc<Mutex<...>>`
or `Arc<RwLock<...>>` wrapper in `server_engine.rs`.

### `infer/src/http_server.rs`

Register the session sub-router under `/v1/sessions/{session_id}`:

```rust
.nest("/v1/sessions/:session_id", sessions::session_router::<E>())
```

Make sure the nesting order puts the literal paths (`save`,
`load`, `manifest`) before any catch-all; axum's router is
order-sensitive for overlapping patterns.

### `docs/environment.md`

Document the new route set under the existing Metal/CUDA runtime
variables section. Not a new env var — just a link to this plan.

### `infer/src/http_server/AGENTS.md`

One-paragraph addition noting that session persistence is a
first-class OpenAI-v1-adjacent surface, and that the underlying
functions live in `sessions.rs` which was shipped in M4d.

---

## 3 · Work items

In order of dependency:

1. **Trait extension** (`InferenceEngine` + scheduler impl). This
   is the critical path: no route can land until the engine
   exposes `session_fingerprints` / `read_block_payload` /
   `install_restored_kv` / `kv_format_tag`.
2. **`copy_pages_to_host` helper** on `PagedKVPool`. This is
   where M4e actually touches CUDA kernel code — probably a
   simple wrapper around the existing `migrate_kv_range` path
   but with a `D2H` direction flag.
3. **axum handlers** in `sessions.rs`. Mechanical wiring on top
   of the shipped pure functions.
4. **Router registration** in `http_server.rs`.
5. **Integration test** in `infer/tests/session_http_routes.rs`:
   - Spin up a test server via `axum::test_server` (or
     `axum_test`)
   - Seed the engine with a small session
   - Hit `/save`, assert 200 + non-empty snapshot
   - Tamper the snapshot, hit `/load`, assert 422
   - Swap the `kv_format_tag`, hit `/load`, assert 400
   - Drain the engine's pool, hit `/load`, assert 503

---

## 4 · Acceptance

```bash
# Unit + module tests
cargo test -p infer --release --no-default-features --features no-cuda sessions
cargo test -p infer --release --no-default-features --features no-cuda http_server

# Integration test (needs the engine under test, so typically CUDA)
PEGAINFER_TEST_MODEL_PATH=models/Qwen3-4B \
  cargo test --release --test session_http_routes

# Full matrix
cargo test -p infer --release --no-default-features --features no-cuda
cargo check -p infer --tests --no-default-features --features cuda,no-cuda
cargo check -p infer --no-default-features --features metal
cargo fmt --all -- --check
cargo clippy -p infer --no-default-features --features no-cuda -- -D warnings
```

Sign-off requires:

- [ ] All 4 routes round-trip a real session against a live
      `Scheduler<Qwen3ForCausalLm>` on CUDA.
- [ ] `PoolExhausted` / `FormatMismatch` / `DiskBlockMismatch` all
      translate to the documented status codes.
- [ ] Request body size limit triggers 413 on a payload above 16
      MiB (drop in a deliberately oversized snapshot).
- [ ] No regression in `/v1/completions` latency on the same
      engine instance (the session router should not share state
      with the completions router beyond the engine handle).
- [ ] A new win note at
      `docs/experience/wins/<date>-tiered-kv-m4e-session-http-routes-local.md`
      (for the local landing) and a matching remote acceptance
      note if the route lands through a CUDA run.

---

## 5 · Open questions for the executor

1. **Engine handle concurrency**: is the existing `InferenceEngine`
   wrapper already `Arc<RwLock<...>>` or similar? Check
   `infer/src/server_engine.rs`'s existing axum wiring for
   `/v1/completions`. If there's no read lock, adding one for
   sessions may require upgrading the completions path too.
2. **`copy_pages_to_host` direction**: does the CUDA kernel layer
   already have a `cudaMemcpyDeviceToHost` variant in
   `crates/infer-cuda-kernels/src/paged_kv.rs`? If yes, thin
   wrapper. If no, one new FFI binding.
3. **`session_fingerprints` walk cost**: O(nodes) walk per save is
   fine for a few hundred blocks, probably not fine for a 30k-token
   session with 2k blocks. If it shows up in profiling, promote to
   a `HashMap<SessionId, Vec<BlockFingerprint>>` side-index on the
   scheduler.
4. **Idempotency-Key header**: M4e v1 does NOT honor it. Document
   as "coming in v2" and move on.
5. **`POST /load` partial failure**: if 15 of 16 blocks restore
   successfully and the 16th is missing, do we roll back the first
   15 or commit a partial restore? **Default: all-or-nothing** —
   any error on a block fails the whole load, the scheduler's
   freshly-allocated pages get released via the allocator closure
   returning an Err variant. Document this in the route's rustdoc
   so callers know.
