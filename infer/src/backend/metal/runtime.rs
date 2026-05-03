use std::collections::HashMap;
use std::io::Read;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use log::{error, info, warn};
use tokio::sync::mpsc;

use super::request_state::{
    DflashBatchOutcome, MetalMixedBatchResult, MetalRequestPhase as RuntimePhase,
    MetalRequestState, Qwen35PackedDecodeBatch, Qwen35PrefixSnapshot,
};
use super::scheduler::{
    MetalRequestPriority, MetalRuntimeRequestState, MetalScheduleStep, MetalScheduler,
    MetalSchedulerConfig,
};
use super::weights::MetalWeights;
use super::{MetalBackend, MetalBackendOptions};
use crate::backend::InferenceBackend;
use crate::backend::runtime::StopChunkProcessor;
use crate::kv_tier::transport::disk::{DiskBlockLocation, DiskStore};
use crate::metrics::ServerMetrics;
use crate::sampler::SamplingParams;
use crate::scheduler::{IncomingRequest, RequestPriority, SchedulerHandle};
use crate::server_engine::{CompletionStreamDelta, FinishReason, TokenUsage};
use crate::tokenizer::{IncrementalDecoder, Tokenizer};
use crate::types::{BlockFingerprint, InferenceMode, KvContentContext, RequestId, SessionId};

struct PendingMetalRequest {
    delta_tx: mpsc::UnboundedSender<CompletionStreamDelta>,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    sampling: SamplingParams,
    stop: Option<Vec<String>>,
    session_id: Option<SessionId>,
    enqueued_at: Instant,
}

impl PendingMetalRequest {
    fn from_incoming(
        tokenizer: &Tokenizer,
        incoming: IncomingRequest,
    ) -> Result<(Self, MetalRequestPriority)> {
        let prompt_tokens = tokenizer.encode(&incoming.prompt)?;
        if prompt_tokens.is_empty() {
            bail!("Metal scheduler request requires at least one prompt token");
        }
        Ok((
            Self {
                delta_tx: incoming.delta_tx,
                prompt_tokens,
                max_tokens: incoming.max_tokens,
                sampling: incoming.sampling,
                stop: incoming.stop,
                session_id: incoming.session_id,
                enqueued_at: Instant::now(),
            },
            map_request_priority(incoming.priority),
        ))
    }

    fn delta_closed(&self) -> bool {
        self.delta_tx.is_closed()
    }

    fn activate(
        self,
        backend: &'static MetalBackend,
        tokenizer: &'static Tokenizer,
        enable_dflash: bool,
    ) -> Result<ActiveMetalRequest> {
        ActiveMetalRequest::from_pending(backend, tokenizer, self, enable_dflash)
    }
}

struct ActiveMetalRequest {
    delta_tx: mpsc::UnboundedSender<CompletionStreamDelta>,
    request_state: MetalRequestState<'static>,
    decoder: IncrementalDecoder<'static>,
    stop_processor: StopChunkProcessor,
    session_id: Option<SessionId>,
    prompt_tokens: Vec<u32>,
    enqueued_at: Instant,
    admitted_at: Instant,
    first_token_at: Option<Instant>,
    /// Phase 2 trajectory token layer. Each `process_token` pushes the
    /// just-sampled id; whenever a text delta is actually sent
    /// (post stop-processor / decoder buffering), the pending IDs are
    /// drained onto that delta. Any IDs still pending at finish time
    /// ride on the final delta so the cumulative `response_token_ids`
    /// the consumer collates equals every generated token.
    pending_token_ids: Vec<u32>,
}

impl ActiveMetalRequest {
    fn from_pending(
        backend: &'static MetalBackend,
        tokenizer: &'static Tokenizer,
        pending: PendingMetalRequest,
        enable_dflash: bool,
    ) -> Result<Self> {
        let prompt_tokens = pending.prompt_tokens;
        let max_tokens = pending.max_tokens;
        let mut sampling = pending.sampling;
        sampling.max_new_tokens = Some(max_tokens);
        // Thread DFlash runtime into the request state so Qwen3StepDriver
        // can initialize speculative-decode state. Both refs are 'static
        // because the backend is leaked into the scheduler runtime thread.
        // SAFETY: `backend` was leaked to `'static` at runtime.rs:591 before
        // this function is called. The ptr-cast inside is sound.
        //
        // `enable_dflash=false` (caller sees concurrent sessions already
        // queued) skips the DFlash hidden-capture prefill too, saving the
        // full-prompt single-shot prefill cost — the request would have
        // been downgraded at the first decode step anyway.
        let dflash_ref = if enable_dflash {
            unsafe { backend.dflash_runtime_static() }
        } else {
            None
        };
        let request_state =
            backend.create_request_state_with_dflash(&prompt_tokens, &sampling, dflash_ref)?;
        Ok(Self {
            delta_tx: pending.delta_tx,
            request_state,
            decoder: tokenizer.incremental_decoder(),
            stop_processor: StopChunkProcessor::new(pending.stop.unwrap_or_default()),
            session_id: pending.session_id,
            prompt_tokens,
            enqueued_at: pending.enqueued_at,
            admitted_at: Instant::now(),
            first_token_at: None,
            pending_token_ids: Vec::new(),
        })
    }

    fn delta_closed(&self) -> bool {
        self.delta_tx.is_closed()
    }

    fn phase(&self) -> RuntimePhase {
        self.request_state.phase()
    }

    fn stop_hit(&self) -> bool {
        self.stop_processor.hit_stop()
    }

    fn prefill_chunk(&mut self, budget: usize) -> Result<Option<u32>> {
        let result = self.request_state.prefill_chunk(budget)?;
        if let Some(token) = result.emitted_token {
            self.process_token(token)?;
            Ok(Some(token))
        } else {
            Ok(None)
        }
    }

    fn decode_step(&mut self) -> Result<u32> {
        let token = self
            .request_state
            .decode_step()?
            .context("decode_step did not emit a token")?;
        self.process_token(token)?;
        Ok(token)
    }

    fn cancel(&mut self) -> Result<()> {
        self.request_state.cancel()
    }

    fn send_final_delta(&mut self) -> Result<()> {
        if let Some(tail) = self.decoder.finish()? {
            self.push_text_chunk(&tail)?;
        }
        if let Some(final_delta) = self.stop_processor.finish() {
            // Final stop-processor flush still belongs to the same
            // generation — drain any pending IDs onto it so they don't
            // need to ride the empty terminator delta below.
            send_text_delta_with_ids(
                &self.delta_tx,
                final_delta,
                std::mem::take(&mut self.pending_token_ids),
            )?;
        }

        let finish_reason = if self.stop_processor.hit_stop() {
            FinishReason::Stop
        } else {
            map_finish_reason(self.request_state.finish_reason())
        };
        let completion_tokens = self.request_state.generated_tokens();
        let prompt_tokens = self.prompt_tokens.len();
        let usage = TokenUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        };

        // Any IDs still pending (e.g. trailing tokens swallowed by the
        // stop processor's withheld suffix) ride on the terminator
        // delta. The collator on the consumer side sums every delta's
        // `token_ids` into `response_token_ids` — sum must equal
        // every token `process_token` saw.
        let _ = self.delta_tx.send(CompletionStreamDelta {
            text_delta: String::new(),
            finish_reason: Some(finish_reason),
            usage: Some(usage),
            logprob: None,
            token_ids: std::mem::take(&mut self.pending_token_ids),
        });
        Ok(())
    }

    fn process_token(&mut self, token_id: u32) -> Result<()> {
        if self.first_token_at.is_none() {
            self.first_token_at = Some(Instant::now());
        }
        // Record the token id BEFORE asking the incremental decoder for
        // text — the byte chunk may emit later (or never, if a stop
        // sequence withholds it), but the id always counts toward
        // `response_token_ids`.
        self.pending_token_ids.push(token_id);
        if let Some(chunk) = self.decoder.step(token_id)? {
            self.push_text_chunk(&chunk)?;
        }
        Ok(())
    }

    fn push_text_chunk(&mut self, chunk: &str) -> Result<()> {
        if let Some(delta) = self.stop_processor.push_chunk(chunk) {
            // Drain pending token IDs onto the delta we're about to
            // send. IDs still in the queue when no delta fires (the
            // decoder buffered, or stop withheld) wait until the next
            // emit or `send_final_delta`.
            let ids = std::mem::take(&mut self.pending_token_ids);
            send_text_delta_with_ids(&self.delta_tx, delta, ids)?;
        }
        Ok(())
    }

    fn prompt_len(&self) -> usize {
        self.prompt_tokens.len()
    }
}

const METAL_PREFIX_BLOCK_SIZE: usize = 16;
// In-memory prefix pool sized so that high-session-count agent traffic (e.g.
// the W3 64-warm-session workload) can keep the most-recently-published
// snapshot per session, instead of LRU-evicting them within seconds. The
// underlying KV+GDR arrays are MLX refcounts (no per-snapshot full copy), so
// the dominant cost is the MLX buffer footprint of the cached requests'
// resident state.
const METAL_PREFIX_POOL_MULTIPLIER: usize = 64;
const METAL_QWEN35_SNAPSHOT_KV_FORMAT_TAG: u8 = 0x35;
const METAL_QWEN35_SNAPSHOT_INDEX_PREFIX_BYTES: usize = 1024 * 1024;
const METRICS_REFRESH_INTERVAL: Duration = Duration::from_millis(40);

enum PrefillChunkOutcome {
    Progress {
        emitted_token: Option<u32>,
        runtime_finished: bool,
        stop_hit: bool,
    },
    ClientDropped,
    Failed(anyhow::Error),
}

#[derive(Debug, thiserror::Error)]
enum MetalStreamError {
    #[error("stream consumer dropped")]
    ConsumerDropped,
}

enum MetalLivePrefixRuntime {
    Qwen35(MetalQwen35PrefixRuntime),
}

/// Eviction footprint for an in-memory cached snapshot, in token-equivalent
/// units. The cache budget accounts the resident KV+GDR allocation, not just
/// the reusable prefix length: the live driver pre-allocates KV to
/// `prompt_len + max_new_tokens`, and the snapshot retains those full arrays
/// via `kv_capacity`. Counting only `token_ids.len()` would let a few
/// long-output requests pin many GB of KV while the cache thinks it has
/// room for more.
fn snapshot_footprint(snapshot: &Qwen35PrefixSnapshot) -> usize {
    let kv_cap = usize::try_from(snapshot.kv_capacity).unwrap_or(0);
    snapshot.token_ids.len().max(kv_cap)
}

struct MetalQwen35CachedPrefix {
    snapshot: Qwen35PrefixSnapshot,
    last_used_tick: u64,
}

struct MetalQwen35DiskPrefix {
    location: DiskBlockLocation,
    last_used_tick: u64,
}

struct MetalQwen35PrefixRuntime {
    entries: HashMap<Vec<u32>, MetalQwen35CachedPrefix>,
    disk_entries: HashMap<Vec<u32>, MetalQwen35DiskPrefix>,
    disk_store: Option<Arc<DiskStore>>,
    model_fingerprint: Vec<u8>,
    max_disk_bytes: Option<u64>,
    disk_high_watermark: f64,
    disk_low_watermark: f64,
    disk_fsync_each_block: bool,
    max_cached_tokens: usize,
    cached_tokens: usize,
    disk_bytes: u64,
    next_tick: u64,
    block_size: usize,
}

struct CachedQwen35DecodeBatch {
    req_ids: Vec<RequestId>,
    batch: Qwen35PackedDecodeBatch<'static>,
}

impl MetalLivePrefixRuntime {
    fn new(backend: &'static MetalBackend, config: &MetalSchedulerConfig) -> Result<Option<Self>> {
        let weights = backend.weights.as_ref().context("weights not loaded")?;
        let max_total_tokens = (config
            .max_running_requests
            .saturating_mul(config.max_batch_tokens)
            .saturating_mul(METAL_PREFIX_POOL_MULTIPLIER))
        .max(METAL_PREFIX_BLOCK_SIZE * 8);
        match weights {
            MetalWeights::Qwen3(_) => {
                info!(
                    "Metal live prefix cache disabled for Qwen3: long-prompt allocator stability takes priority over prompt-prefix reuse"
                );
                Ok(None)
            }
            MetalWeights::Qwen35(weights) => {
                if weights.cpp_model.is_none() {
                    info!(
                        "Metal live prefix cache disabled for Qwen3.6/Qwen3.5-MoE: snapshot replay requires the compiled Qwen3.5 step path"
                    );
                    return Ok(None);
                }
                info!(
                    "Metal live prefix cache enabled for Qwen3.5 snapshot replay: block_size={}, max_cached_tokens={}",
                    METAL_PREFIX_BLOCK_SIZE, max_total_tokens
                );
                let (
                    disk_store,
                    model_fingerprint,
                    max_disk_bytes,
                    disk_high_watermark,
                    disk_low_watermark,
                    disk_fsync_each_block,
                ) = if let Some(options) = backend.kv_disk_options.as_ref() {
                    let store = Arc::new(DiskStore::new(&options.dir));
                    store.create_root().with_context(|| {
                        format!("create Metal Qwen3.5 SSD KV dir {}", options.dir.display())
                    })?;
                    let model_fingerprint = metal_prefix_model_fingerprint(backend)?;
                    (
                        Some(store),
                        model_fingerprint,
                        options.max_bytes,
                        options.high_watermark,
                        options.low_watermark,
                        options.fsync_each_block,
                    )
                } else {
                    (None, Vec::new(), None, 0.90, 0.75, false)
                };
                Ok(Some(Self::Qwen35(MetalQwen35PrefixRuntime::new(
                    max_total_tokens,
                    METAL_PREFIX_BLOCK_SIZE,
                    disk_store,
                    model_fingerprint,
                    max_disk_bytes,
                    disk_high_watermark,
                    disk_low_watermark,
                    disk_fsync_each_block,
                )?)))
            }
        }
    }

    fn prepare_request(
        &mut self,
        request: &mut ActiveMetalRequest,
        metrics: &ServerMetrics,
    ) -> Result<()> {
        match self {
            MetalLivePrefixRuntime::Qwen35(runtime) => runtime.prepare_request(request, metrics),
        }
    }

    fn publish_prompt_prefix(&mut self, request: &mut ActiveMetalRequest) -> Result<()> {
        match self {
            MetalLivePrefixRuntime::Qwen35(runtime) => runtime.publish_prompt_prefix(request),
        }
    }
}

impl MetalQwen35PrefixRuntime {
    fn new(
        max_cached_tokens: usize,
        block_size: usize,
        disk_store: Option<Arc<DiskStore>>,
        model_fingerprint: Vec<u8>,
        max_disk_bytes: Option<u64>,
        disk_high_watermark: f64,
        disk_low_watermark: f64,
        disk_fsync_each_block: bool,
    ) -> Result<Self> {
        let mut runtime = Self {
            entries: HashMap::new(),
            disk_entries: HashMap::new(),
            disk_store,
            model_fingerprint,
            max_disk_bytes,
            disk_high_watermark,
            disk_low_watermark,
            disk_fsync_each_block,
            max_cached_tokens,
            cached_tokens: 0,
            disk_bytes: 0,
            next_tick: 1,
            block_size,
        };
        runtime.reconcile_disk_entries()?;
        if runtime.disk_store.is_some() {
            info!(
                "Metal Qwen3.5 SSD prefix cache indexed {} entries ({} bytes)",
                runtime.disk_entries.len(),
                runtime.disk_bytes
            );
        }
        Ok(runtime)
    }

    fn prepare_request(
        &mut self,
        request: &mut ActiveMetalRequest,
        metrics: &ServerMetrics,
    ) -> Result<()> {
        let prompt_len = request.prompt_tokens.len();
        if prompt_len < self.block_size {
            metrics.record_request_cache(request.session_id.as_ref(), 0, prompt_len, prompt_len);
            return Ok(());
        }
        if request.request_state.is_dflash_enabled() {
            metrics.record_request_cache(request.session_id.as_ref(), 0, prompt_len, prompt_len);
            return Ok(());
        }
        if !request.request_state.can_import_qwen35_prefix_snapshot() {
            metrics.record_request_cache(request.session_id.as_ref(), 0, prompt_len, prompt_len);
            return Ok(());
        }

        let memory_key = self.lookup_longest_prefix(&request.prompt_tokens);
        let disk_key = self.lookup_longest_disk_prefix(&request.prompt_tokens);
        let memory_len = memory_key.as_ref().map_or(0, Vec::len);
        let disk_len = disk_key.as_ref().map_or(0, Vec::len);

        let mut reused_tokens = 0;
        if disk_len > memory_len {
            if let Some(prefix_key) = disk_key.as_deref() {
                if self.try_import_disk_prefix_or_remove(prefix_key, request) {
                    reused_tokens = prefix_key.len();
                }
            }
            if reused_tokens == 0
                && let Some(prefix_key) = memory_key.as_deref()
                && self.try_import_memory_prefix(prefix_key, request)?
            {
                reused_tokens = prefix_key.len();
            }
        } else {
            if let Some(prefix_key) = memory_key.as_deref()
                && self.try_import_memory_prefix(prefix_key, request)?
            {
                reused_tokens = prefix_key.len();
            }
            if reused_tokens == 0
                && let Some(prefix_key) = disk_key.as_deref()
                && self.try_import_disk_prefix_or_remove(prefix_key, request)
            {
                reused_tokens = prefix_key.len();
            }
        }
        metrics.record_request_cache(
            request.session_id.as_ref(),
            reused_tokens,
            prompt_len,
            prompt_len.saturating_sub(reused_tokens),
        );
        Ok(())
    }

    fn publish_prompt_prefix(&mut self, request: &mut ActiveMetalRequest) -> Result<()> {
        if !request.request_state.can_import_qwen35_prefix_snapshot() {
            return Ok(());
        }

        if self.disk_store.is_some() {
            request
                .request_state
                .drain_qwen35_cpp_session()
                .context("drain Qwen3.5 C++ session before SSD prefix export")?;
            let snapshots = request
                .request_state
                .export_qwen35_disk_prompt_prefixes(self.block_size)
                .context("export Qwen3.5 prompt prefix snapshots for SSD")?;
            for snapshot in snapshots {
                if let Err(err) = self.persist_snapshot(&snapshot) {
                    warn!(
                        "Metal Qwen3.5 SSD prefix publish failed for {} tokens: {err:#}",
                        snapshot.token_ids.len()
                    );
                }
                self.insert_snapshot(snapshot);
            }
            return Ok(());
        }

        // In-memory tier: snapshot the live C++ session at the largest
        // block-aligned prompt prefix. Drains the session as a side effect; the
        // next decode/prefill tick re-attaches via `begin_session`.
        if let Some(snapshot) = request
            .request_state
            .export_qwen35_live_prefix_snapshot(self.block_size)
            .context("snapshot live Qwen3.5 prompt prefix")?
        {
            self.insert_snapshot(snapshot);
        }
        Ok(())
    }

    fn lookup_longest_prefix(&self, prompt_tokens: &[u32]) -> Option<Vec<u32>> {
        self.entries
            .keys()
            .filter(|tokens| {
                let prefix_len = tokens.len();
                prefix_len >= self.block_size
                    && prefix_len < prompt_tokens.len()
                    && prompt_tokens.starts_with(tokens.as_slice())
            })
            .max_by_key(|tokens| tokens.len())
            .cloned()
    }

    fn lookup_longest_disk_prefix(&self, prompt_tokens: &[u32]) -> Option<Vec<u32>> {
        self.disk_entries
            .keys()
            .filter(|tokens| {
                let prefix_len = tokens.len();
                prefix_len >= self.block_size
                    && prefix_len < prompt_tokens.len()
                    && prompt_tokens.starts_with(tokens.as_slice())
            })
            .max_by_key(|tokens| tokens.len())
            .cloned()
    }

    fn try_import_memory_prefix(
        &mut self,
        prefix_key: &[u32],
        request: &mut ActiveMetalRequest,
    ) -> Result<bool> {
        let imported = {
            let Some(snapshot) = self.entries.get(prefix_key).map(|entry| &entry.snapshot) else {
                return Ok(false);
            };
            request
                .request_state
                .import_qwen35_prefix_snapshot(snapshot, prefix_key.len())
                .context("import matched Qwen3.5 prefix snapshot into request state")?
        };
        if imported {
            self.touch(prefix_key);
        }
        Ok(imported)
    }

    fn try_import_disk_prefix_or_remove(
        &mut self,
        prefix_key: &[u32],
        request: &mut ActiveMetalRequest,
    ) -> bool {
        if !request.request_state.can_import_qwen35_prefix_snapshot() {
            warn!(
                "Metal Qwen3.5 SSD prefix import skipped for {} tokens: compiled step path unavailable",
                prefix_key.len()
            );
            return false;
        }
        match self.try_import_disk_prefix(prefix_key, request) {
            Ok(imported) => imported,
            Err(err) => {
                warn!(
                    "Metal Qwen3.5 SSD prefix import failed for {} tokens: {err:#}",
                    prefix_key.len()
                );
                self.remove_disk_entry(prefix_key);
                false
            }
        }
    }

    fn insert_snapshot(&mut self, snapshot: Qwen35PrefixSnapshot) {
        let token_count = snapshot.token_ids.len();
        if token_count < self.block_size {
            return;
        }
        // In-memory entries hold the live drained KV+GDR at exactly
        // `cache_len`, so block alignment is not required for correctness.
        // The disk-tier `persist_snapshot` path keeps its own alignment guard
        // because the on-disk format assumes block-aligned slices.
        let tick = self.bump_tick();
        if let Some(existing) = self.entries.get_mut(&snapshot.token_ids) {
            existing.last_used_tick = tick;
            return;
        }
        let footprint = snapshot_footprint(&snapshot);
        if footprint > self.max_cached_tokens {
            return;
        }

        self.ensure_capacity_for(footprint);
        let key = snapshot.token_ids.clone();
        self.cached_tokens += footprint;
        self.entries.insert(
            key,
            MetalQwen35CachedPrefix {
                snapshot,
                last_used_tick: tick,
            },
        );
    }

    fn persist_snapshot(&mut self, snapshot: &Qwen35PrefixSnapshot) -> Result<()> {
        let Some(store) = self.disk_store.clone() else {
            return Ok(());
        };
        let token_count = snapshot.token_ids.len();
        if token_count < self.block_size || !token_count.is_multiple_of(self.block_size) {
            return Ok(());
        }

        let key = snapshot.token_ids.clone();
        if self.disk_entries.contains_key(&key) {
            self.touch_disk(&key);
            return Ok(());
        }

        let estimated_payload_len = snapshot
            .estimated_disk_payload_len(&self.model_fingerprint)
            .context("estimate Qwen3.5 prefix snapshot size for SSD")?;
        if !self.ensure_disk_capacity_for(estimated_payload_len) {
            return Ok(());
        }

        let payload = snapshot
            .encode_for_disk(&self.model_fingerprint)
            .context("encode Qwen3.5 prefix snapshot for SSD")?;
        let payload_len =
            u64::try_from(payload.len()).context("Qwen3.5 prefix snapshot payload too large")?;
        if payload_len != estimated_payload_len && !self.ensure_disk_capacity_for(payload_len) {
            return Ok(());
        }

        let fingerprint = self.fingerprint_for_tokens(&key);
        let location = store
            .put_block_with_fsync(
                fingerprint,
                METAL_QWEN35_SNAPSHOT_KV_FORMAT_TAG,
                &payload,
                self.disk_fsync_each_block,
            )
            .context("write Qwen3.5 prefix snapshot to DiskStore")?;
        let tick = self.bump_tick();
        self.disk_bytes = self.disk_bytes.saturating_add(location.payload_len);
        self.disk_entries.insert(
            key,
            MetalQwen35DiskPrefix {
                location,
                last_used_tick: tick,
            },
        );
        Ok(())
    }

    fn try_import_disk_prefix(
        &mut self,
        prefix_key: &[u32],
        request: &mut ActiveMetalRequest,
    ) -> Result<bool> {
        let Some(store) = self.disk_store.clone() else {
            return Ok(false);
        };
        let Some(location) = self
            .disk_entries
            .get(prefix_key)
            .map(|entry| entry.location.clone())
        else {
            return Ok(false);
        };
        let expected = self.fingerprint_for_tokens(prefix_key);
        let payload = store
            .get_block(&location, Some(expected))
            .context("read Qwen3.5 prefix snapshot from DiskStore")?;
        let snapshot = Qwen35PrefixSnapshot::decode_from_disk(&payload, &self.model_fingerprint)
            .context("decode Qwen3.5 prefix snapshot from DiskStore")?;
        ensure!(
            snapshot.token_ids == prefix_key,
            "Qwen3.5 SSD prefix snapshot token key mismatch"
        );

        let imported = request
            .request_state
            .import_qwen35_prefix_snapshot(&snapshot, prefix_key.len())
            .context("import matched Qwen3.5 SSD prefix snapshot into request state")?;
        if imported {
            self.touch_disk(prefix_key);
            self.insert_snapshot(snapshot);
        }
        Ok(imported)
    }

    fn reconcile_disk_entries(&mut self) -> Result<()> {
        let Some(store) = self.disk_store.clone() else {
            return Ok(());
        };
        store
            .visit_block_payload_prefixes(
                METAL_QWEN35_SNAPSHOT_INDEX_PREFIX_BYTES,
                |location, payload| {
                    let token_ids = match Qwen35PrefixSnapshot::peek_disk_token_ids(
                        payload,
                        &self.model_fingerprint,
                    ) {
                        Ok(token_ids) => token_ids,
                        Err(err) => {
                            log::debug!(
                                "Metal Qwen3.5 SSD prefix cache ignored {}: {err:#}",
                                location.path.display()
                            );
                            if Qwen35PrefixSnapshot::looks_like_disk_payload(payload) {
                                delete_rejected_qwen35_disk_block(&store, &location);
                            }
                            return Ok(());
                        }
                    };
                    if token_ids.len() < self.block_size
                        || !token_ids.len().is_multiple_of(self.block_size)
                    {
                        delete_rejected_qwen35_disk_block(&store, &location);
                        return Ok(());
                    }
                    let expected = self.fingerprint_for_tokens(&token_ids);
                    if location.fingerprint != expected {
                        log::debug!(
                            "Metal Qwen3.5 SSD prefix cache ignored {}: fingerprint/token mismatch",
                            location.path.display()
                        );
                        delete_rejected_qwen35_disk_block(&store, &location);
                        return Ok(());
                    }
                    let tick = self.bump_tick();
                    self.disk_bytes = self.disk_bytes.saturating_add(location.payload_len);
                    self.disk_entries.insert(
                        token_ids,
                        MetalQwen35DiskPrefix {
                            location,
                            last_used_tick: tick,
                        },
                    );
                    Ok(())
                },
            )
            .context("scan Metal Qwen3.5 SSD prefix cache")?;
        self.ensure_disk_capacity_for(0);
        Ok(())
    }

    fn ensure_capacity_for(&mut self, needed_tokens: usize) {
        while self.cached_tokens.saturating_add(needed_tokens) > self.max_cached_tokens {
            let Some((lru_key, lru_footprint)) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used_tick)
                .map(|(tokens, entry)| (tokens.clone(), snapshot_footprint(&entry.snapshot)))
            else {
                break;
            };
            self.entries.remove(&lru_key);
            self.cached_tokens = self.cached_tokens.saturating_sub(lru_footprint);
        }
    }

    fn ensure_disk_capacity_for(&mut self, needed_bytes: u64) -> bool {
        let Some(max_bytes) = self.max_disk_bytes else {
            return true;
        };
        let high = watermark_bytes(max_bytes, self.disk_high_watermark);
        let low = watermark_bytes(max_bytes, self.disk_low_watermark);
        if needed_bytes > high {
            return false;
        }
        if self.disk_bytes.saturating_add(needed_bytes) <= high {
            return true;
        }

        let target = low.saturating_sub(needed_bytes);
        while self.disk_bytes > target {
            let Some(lru_key) = self
                .disk_entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used_tick)
                .map(|(tokens, _)| tokens.clone())
            else {
                break;
            };
            self.remove_disk_entry(&lru_key);
        }

        self.disk_bytes.saturating_add(needed_bytes) <= high
    }

    fn touch(&mut self, key: &[u32]) {
        let tick = self.bump_tick();
        if let Some(entry) = self.entries.get_mut(key) {
            entry.last_used_tick = tick;
        }
    }

    fn touch_disk(&mut self, key: &[u32]) {
        let tick = self.bump_tick();
        if let Some(entry) = self.disk_entries.get_mut(key) {
            entry.last_used_tick = tick;
        }
    }

    fn remove_disk_entry(&mut self, key: &[u32]) {
        let Some(entry) = self.disk_entries.remove(key) else {
            return;
        };
        self.disk_bytes = self.disk_bytes.saturating_sub(entry.location.payload_len);
        if let Some(store) = self.disk_store.as_ref()
            && let Err(err) = store.delete_block(&entry.location)
        {
            warn!(
                "Metal Qwen3.5 SSD prefix cache failed to delete {}: {err:#}",
                entry.location.path.display()
            );
        }
    }

    fn fingerprint_for_tokens(&self, token_ids: &[u32]) -> BlockFingerprint {
        BlockFingerprint::compute(
            KvContentContext {
                model_fingerprint: &self.model_fingerprint,
                kv_format_tag: METAL_QWEN35_SNAPSHOT_KV_FORMAT_TAG,
                parent: None,
            },
            token_ids,
        )
    }

    fn bump_tick(&mut self) -> u64 {
        let tick = self.next_tick;
        self.next_tick = self.next_tick.saturating_add(1);
        tick
    }
}

fn watermark_bytes(max_bytes: u64, watermark: f64) -> u64 {
    ((max_bytes as f64) * watermark).ceil() as u64
}

fn delete_rejected_qwen35_disk_block(store: &DiskStore, location: &DiskBlockLocation) {
    if let Err(err) = store.delete_block(location) {
        warn!(
            "Metal Qwen3.5 SSD prefix cache failed to delete rejected block {}: {err}",
            location.path.display()
        );
    }
}

fn metal_prefix_model_fingerprint(backend: &MetalBackend) -> Result<Vec<u8>> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"arle-metal-qwen35-prefix-v1\0");
    if let Some(config) = backend.config.as_ref() {
        hasher.update(b"config\0");
        hasher.update(format!("{config:?}").as_bytes());
    }
    if let Some(source_path) = backend.model_source_path.as_deref() {
        hasher.update(b"selected-source\0");
        hasher.update(source_path.to_string_lossy().as_bytes());
        hash_model_file_identity(&mut hasher, source_path)
            .with_context(|| format!("fingerprint selected model {}", source_path.display()))?;
    }
    let selected_file = backend
        .model_source_path
        .as_deref()
        .filter(|path| std::fs::metadata(path).is_ok_and(|metadata| metadata.is_file()));
    if let Some(model_dir) = backend.model_dir.as_deref() {
        hasher.update(b"model-root\0");
        hasher.update(model_dir.to_string_lossy().as_bytes());
        hash_model_tree_metadata(&mut hasher, model_dir, selected_file)
            .with_context(|| format!("fingerprint Metal model tree {}", model_dir.display()))?;
    }
    Ok(hasher.finalize().as_bytes().to_vec())
}

fn hash_model_tree_metadata(
    hasher: &mut blake3::Hasher,
    root: &Path,
    selected_file: Option<&Path>,
) -> Result<()> {
    let mut files = Vec::new();
    collect_model_files(root, root, selected_file, &mut files)?;
    files.sort();
    for relative in files {
        let path = root.join(&relative);
        hasher.update(b"file\0");
        hasher.update(relative.to_string_lossy().as_bytes());
        hash_model_file_identity(hasher, &path)?;
    }
    Ok(())
}

fn hash_model_file_identity(hasher: &mut blake3::Hasher, path: &Path) -> Result<()> {
    let metadata = std::fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    if !metadata.is_file() {
        hasher.update(b"not-file\0");
        return Ok(());
    }
    hasher.update(&metadata.len().to_le_bytes());
    if should_hash_model_file_contents(path) {
        hasher.update(b"content-blake3\0");
        let mut file =
            std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mut file_hasher = blake3::Hasher::new();
        let mut buffer = vec![0u8; 1024 * 1024];
        loop {
            let read = file
                .read(&mut buffer)
                .with_context(|| format!("read {}", path.display()))?;
            if read == 0 {
                break;
            }
            file_hasher.update(&buffer[..read]);
        }
        let file_hash = file_hasher.finalize();
        hasher.update(file_hash.as_bytes());
    }
    Ok(())
}

fn collect_model_files(
    root: &Path,
    path: &Path,
    selected_file: Option<&Path>,
    out: &mut Vec<PathBuf>,
) -> Result<()> {
    let metadata = std::fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    if metadata.is_file() {
        if should_include_model_tree_file(path, selected_file) {
            out.push(path.strip_prefix(root).unwrap_or(path).to_path_buf());
        }
        return Ok(());
    }
    if !metadata.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(path).with_context(|| format!("read {}", path.display()))? {
        let entry = entry?;
        collect_model_files(root, &entry.path(), selected_file, out)?;
    }
    Ok(())
}

fn should_include_model_tree_file(path: &Path, selected_file: Option<&Path>) -> bool {
    if !is_model_fingerprint_file(path) {
        return false;
    }
    if let Some(selected_file) = selected_file {
        if same_model_file(path, selected_file) || is_model_weight_file(path) {
            return false;
        }
    }
    true
}

fn is_model_fingerprint_file(path: &Path) -> bool {
    let Some(extension) = path.extension().and_then(|ext| ext.to_str()) else {
        return false;
    };
    matches!(extension, "json" | "safetensors" | "gguf" | "txt" | "model")
}

fn is_model_weight_file(path: &Path) -> bool {
    let Some(extension) = path.extension().and_then(|ext| ext.to_str()) else {
        return false;
    };
    matches!(extension, "safetensors" | "gguf")
}

fn same_model_file(left: &Path, right: &Path) -> bool {
    if left == right {
        return true;
    }
    match (std::fs::canonicalize(left), std::fs::canonicalize(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

fn should_hash_model_file_contents(path: &Path) -> bool {
    let Some(extension) = path.extension().and_then(|ext| ext.to_str()) else {
        return false;
    };
    matches!(extension, "json" | "safetensors" | "gguf" | "txt" | "model")
}

/// Spawn the first live Metal scheduler runtime.
///
/// This runtime uses the request-state API to interleave chunked prefill and
/// decode scheduling. Qwen3 decode batches are executed as one cross-request
/// GPU graph; unsupported decode batches fall back to request-by-request
/// execution inside the scheduler loop.
pub fn spawn_metal_scheduler_handle_from_path_with_options(
    model_path: &str,
    options: MetalBackendOptions,
    max_waiting: usize,
) -> Result<MetalSchedulerHandle> {
    let model_id = Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(model_path)
        .to_string();
    spawn_metal_scheduler_handle_from_path_with_options_and_metrics(
        model_path,
        options,
        max_waiting,
        ServerMetrics::new(&model_id),
    )
}

/// Wrapper that pairs a `SchedulerHandle` with DFlash init-time metadata for
/// HTTP-layer introspection (`/v1/models`).
///
/// The inner scheduler handle submits work exactly as the raw
/// `SchedulerHandle` does — this struct only adds a read-only side channel
/// for the DFlash draft id and speculative block size, captured at backend
/// load time. Acceptance rate is NOT stored here; it is read from the shared
/// `ServerMetrics` at response time (rolling counter).
#[derive(Clone)]
pub struct MetalSchedulerHandle {
    inner: SchedulerHandle,
    dflash_status: Option<crate::request_handle::DflashStatus>,
}

impl MetalSchedulerHandle {
    /// Borrow the underlying `SchedulerHandle` for callers that still expect
    /// the raw scheduler type (e.g. bench harness token-counter plumbing).
    pub fn inner(&self) -> &SchedulerHandle {
        &self.inner
    }
}

impl crate::request_handle::RequestHandle for MetalSchedulerHandle {
    fn submit(
        &self,
        req: IncomingRequest,
    ) -> std::result::Result<(), crate::request_handle::SubmitError> {
        SchedulerHandle::submit(&self.inner, req).map_err(|_| crate::request_handle::SubmitError)
    }

    fn model_id(&self) -> &str {
        SchedulerHandle::model_id(&self.inner)
    }

    fn dflash_status(&self) -> Option<crate::request_handle::DflashStatus> {
        self.dflash_status.clone()
    }
}

pub fn spawn_metal_scheduler_handle_from_path_with_options_and_metrics(
    model_path: &str,
    options: MetalBackendOptions,
    max_waiting: usize,
    metrics: ServerMetrics,
) -> Result<MetalSchedulerHandle> {
    // DFlash is now supported: Qwen3StepDriver's token-buffer pattern runs
    // speculative blocks inside decode_token, transparent to the scheduler.
    let mut backend = MetalBackend::with_options(options);
    backend.load(Path::new(model_path))?;

    // Snapshot DFlash metadata BEFORE the backend is leaked into the
    // scheduler thread. When DFlash is disabled at load time (either no
    // draft requested, or a compatibility check failed and fell back),
    // this reads `None` and the HTTP layer reports DFlash disabled —
    // matching the actual runtime state.
    let dflash_status =
        backend
            .dflash_runtime_ref()
            .map(|rt| crate::request_handle::DflashStatus {
                draft_model: rt.draft_model_id().to_string(),
                speculative_tokens: rt.block_size(),
            });

    let model_id = Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(model_path)
        .to_string();

    let (tx, rx) = mpsc::unbounded_channel();
    let waiting_count = Arc::new(AtomicUsize::new(0));
    // Forward the loaded tokenizer through the SchedulerHandle so the
    // server_engine layer's `tokenize()` (used by the v2 trajectory
    // exporter to mask tool tokens) actually returns IDs on Metal.
    // Without this, `RequestHandle::tokenizer_clone` returned None →
    // every Metal agent turn silently downgraded to `tokens: null`.
    // (codex Phase-2 P1)
    let mut handle =
        SchedulerHandle::with_shared_waiting_count(tx, &model_id, max_waiting, waiting_count);
    if let Some(tokenizer) = backend.tokenizer.as_ref() {
        handle = handle.with_tokenizer(tokenizer.clone());
    }

    let runtime_handle = handle.clone();
    std::thread::spawn(move || {
        // The runtime owns one backend instance for the process lifetime. The
        // request-state API currently borrows backend internals, so keep the
        // loaded backend stable inside the worker thread until the server exits.
        let backend: &'static MetalBackend = Box::leak(Box::new(backend));
        let Some(tokenizer) = backend.tokenizer.as_ref() else {
            error!("Metal scheduler runtime failed: model tokenizer not loaded");
            return;
        };
        let tokenizer: &'static Tokenizer = tokenizer;

        let result = catch_unwind(AssertUnwindSafe(|| {
            run_metal_scheduler_runtime(
                backend,
                tokenizer,
                rx,
                &runtime_handle,
                &metrics,
                MetalSchedulerConfig::default(),
            )
        }));

        match result {
            Ok(Ok(())) => {}
            Ok(Err(err)) => error!("Metal scheduler runtime failed: {err:#}"),
            Err(panic) => error!(
                "Metal scheduler runtime panicked: {}",
                super::panic_message(panic)
            ),
        }
    });

    Ok(MetalSchedulerHandle {
        inner: handle,
        dflash_status,
    })
}

pub fn spawn_metal_scheduler_handle_from_path(
    model_path: &str,
    max_waiting: usize,
) -> Result<MetalSchedulerHandle> {
    spawn_metal_scheduler_handle_from_path_with_options(
        model_path,
        MetalBackendOptions::default(),
        max_waiting,
    )
}

fn run_metal_scheduler_runtime(
    backend: &'static MetalBackend,
    tokenizer: &'static Tokenizer,
    mut request_rx: mpsc::UnboundedReceiver<IncomingRequest>,
    handle: &SchedulerHandle,
    metrics: &ServerMetrics,
    config: MetalSchedulerConfig,
) -> Result<()> {
    let mut prefix_runtime = MetalLivePrefixRuntime::new(backend, &config)?;
    let mut scheduler = MetalScheduler::new(config)?;
    let mut pending = HashMap::<RequestId, PendingMetalRequest>::new();
    let mut active = HashMap::<RequestId, ActiveMetalRequest>::new();
    let mut qwen35_decode_batch_cache: Option<CachedQwen35DecodeBatch> = None;
    let mut request_rx_closed = false;
    let mut last_metrics_refresh: Option<Instant> = None;

    info!("Metal scheduler runtime started");

    loop {
        drain_incoming_requests(
            tokenizer,
            handle,
            metrics,
            &mut request_rx,
            &mut request_rx_closed,
            &mut scheduler,
            &mut pending,
        );
        reap_closed_clients(handle, &mut scheduler, &mut pending, &mut active);
        maybe_refresh_runtime_metrics(
            metrics,
            handle,
            &scheduler,
            &pending,
            &active,
            &mut last_metrics_refresh,
            METRICS_REFRESH_INTERVAL,
        );

        if request_rx_closed && active.is_empty() && scheduler.waiting_len() == 0 {
            info!("Metal scheduler runtime shutting down: all handles dropped");
            break;
        }

        if active.is_empty() && scheduler.waiting_len() == 0 {
            if let Some(incoming) = request_rx.blocking_recv() {
                enqueue_request(
                    metrics,
                    tokenizer,
                    incoming,
                    handle,
                    &mut scheduler,
                    &mut pending,
                );
                // Admission is rare enough that an unconditional refresh
                // is fine — helps the first metrics scrape after idle.
                refresh_runtime_metrics(metrics, handle, &scheduler, &pending, &active);
                last_metrics_refresh = Some(Instant::now());
            } else {
                request_rx_closed = true;
                continue;
            }
        }

        let runtime_states = scheduler_runtime_states(&active);
        let step = scheduler.step(&runtime_states);
        if step.is_idle() {
            metrics.set_scheduler_step(0, 0, 0, 0, 0, 0);
            continue;
        }

        let scheduled_decode_rows =
            step.decode.as_ref().map_or(0, |batch| batch.req_ids.len()) as u64;
        let scheduled_prefill_rows = step.prefill.len() as u64;
        let scheduled_prefill_tokens = step
            .prefill
            .iter()
            .map(|prefill| prefill.input_tokens.len() as u64)
            .sum();
        let scheduled_rows = scheduled_decode_rows + scheduled_prefill_rows;
        metrics.set_scheduler_step(
            scheduled_rows,
            scheduled_decode_rows,
            scheduled_prefill_rows,
            scheduled_decode_rows,
            scheduled_prefill_tokens,
            scheduled_rows,
        );
        let step_started_at = Instant::now();

        guard_schedule_step(
            step,
            backend,
            tokenizer,
            handle,
            metrics,
            &mut prefix_runtime,
            &mut scheduler,
            &mut pending,
            &mut active,
            &mut qwen35_decode_batch_cache,
        );
        metrics.observe_scheduler_step(step_started_at.elapsed().as_secs_f64());

        maybe_refresh_runtime_metrics(
            metrics,
            handle,
            &scheduler,
            &pending,
            &active,
            &mut last_metrics_refresh,
            METRICS_REFRESH_INTERVAL,
        );
    }

    Ok(())
}

fn guard_prefill_chunk(
    req_id: RequestId,
    budget: usize,
    backend: &'static MetalBackend,
    tokenizer: &'static Tokenizer,
    handle: &SchedulerHandle,
    metrics: &ServerMetrics,
    prefix_runtime: &mut Option<MetalLivePrefixRuntime>,
    scheduler: &mut MetalScheduler,
    pending: &mut HashMap<RequestId, PendingMetalRequest>,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    let result = catch_unwind(AssertUnwindSafe(|| {
        execute_prefill_chunk(
            req_id,
            budget,
            backend,
            tokenizer,
            handle,
            metrics,
            prefix_runtime,
            scheduler,
            pending,
            active,
        );
    }));

    if let Err(panic) = result {
        error!(
            "Metal prefill chunk panicked for {:?}: {}",
            req_id,
            super::panic_message(panic)
        );
        metrics.record_request_failed();
        *prefix_runtime = None;
        abort_runtime_requests(&[req_id], scheduler, active);
    }
}

#[allow(clippy::too_many_arguments)]
fn guard_schedule_step(
    step: MetalScheduleStep,
    backend: &'static MetalBackend,
    tokenizer: &'static Tokenizer,
    handle: &SchedulerHandle,
    metrics: &ServerMetrics,
    prefix_runtime: &mut Option<MetalLivePrefixRuntime>,
    scheduler: &mut MetalScheduler,
    pending: &mut HashMap<RequestId, PendingMetalRequest>,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
    qwen35_decode_batch_cache: &mut Option<CachedQwen35DecodeBatch>,
) {
    // The planner emits 0-or-1 prefill rows in B2 commit 1; commit 3 lifts
    // this to up to `max_prefill_rows`. Until then, dispatch on the head row
    // and assert the invariant so a planner regression fails loudly.
    debug_assert!(
        step.prefill.len() <= 1,
        "B2 commit 1 dispatcher expects ≤1 prefill row; got {}",
        step.prefill.len()
    );
    let prefill_head = step.prefill.into_iter().next();
    match (step.decode, prefill_head) {
        (Some(batch), Some(prefill)) => {
            if !guard_mixed_batch(
                batch.req_ids.clone(),
                prefill.req_id,
                prefill.input_tokens.len(),
                backend,
                tokenizer,
                handle,
                metrics,
                prefix_runtime,
                scheduler,
                pending,
                active,
            ) {
                guard_decode_batch(
                    batch.req_ids,
                    metrics,
                    scheduler,
                    active,
                    qwen35_decode_batch_cache,
                );
                guard_prefill_chunk(
                    prefill.req_id,
                    prefill.input_tokens.len(),
                    backend,
                    tokenizer,
                    handle,
                    metrics,
                    prefix_runtime,
                    scheduler,
                    pending,
                    active,
                );
            }
        }
        (Some(batch), None) => {
            guard_decode_batch(
                batch.req_ids,
                metrics,
                scheduler,
                active,
                qwen35_decode_batch_cache,
            );
        }
        (None, Some(prefill)) => {
            guard_prefill_chunk(
                prefill.req_id,
                prefill.input_tokens.len(),
                backend,
                tokenizer,
                handle,
                metrics,
                prefix_runtime,
                scheduler,
                pending,
                active,
            );
        }
        (None, None) => {}
    }
}

fn guard_mixed_batch(
    decode_req_ids: Vec<RequestId>,
    prefill_req_id: RequestId,
    prefill_budget: usize,
    backend: &'static MetalBackend,
    tokenizer: &'static Tokenizer,
    handle: &SchedulerHandle,
    metrics: &ServerMetrics,
    prefix_runtime: &mut Option<MetalLivePrefixRuntime>,
    scheduler: &mut MetalScheduler,
    pending: &mut HashMap<RequestId, PendingMetalRequest>,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) -> bool {
    let mut panic_req_ids = decode_req_ids.clone();
    panic_req_ids.push(prefill_req_id);
    let result = catch_unwind(AssertUnwindSafe(|| {
        execute_mixed_batch(
            decode_req_ids,
            prefill_req_id,
            prefill_budget,
            backend,
            tokenizer,
            handle,
            metrics,
            prefix_runtime,
            scheduler,
            pending,
            active,
        )
    }));

    match result {
        Ok(handled) => handled,
        Err(panic) => {
            error!(
                "Metal mixed batch panicked for {:?}: {}",
                panic_req_ids,
                super::panic_message(panic)
            );
            metrics.record_request_failed();
            *prefix_runtime = None;
            abort_runtime_requests(&panic_req_ids, scheduler, active);
            true
        }
    }
}

fn guard_decode_batch(
    req_ids: Vec<RequestId>,
    metrics: &ServerMetrics,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
    qwen35_decode_batch_cache: &mut Option<CachedQwen35DecodeBatch>,
) {
    let panic_req_ids = req_ids.clone();
    let result = catch_unwind(AssertUnwindSafe(|| {
        execute_decode_batch(
            req_ids,
            metrics,
            scheduler,
            active,
            qwen35_decode_batch_cache,
        );
    }));

    if let Err(panic) = result {
        error!(
            "Metal decode batch panicked for {:?}: {}",
            panic_req_ids,
            super::panic_message(panic)
        );
        metrics.record_request_failed();
        *qwen35_decode_batch_cache = None;
        abort_runtime_requests(&panic_req_ids, scheduler, active);
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_mixed_batch(
    decode_req_ids: Vec<RequestId>,
    prefill_req_id: RequestId,
    prefill_budget: usize,
    backend: &'static MetalBackend,
    tokenizer: &'static Tokenizer,
    handle: &SchedulerHandle,
    metrics: &ServerMetrics,
    prefix_runtime: &mut Option<MetalLivePrefixRuntime>,
    scheduler: &mut MetalScheduler,
    pending: &mut HashMap<RequestId, PendingMetalRequest>,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) -> bool {
    if !active.contains_key(&prefill_req_id) {
        activate_pending_request(
            prefill_req_id,
            backend,
            tokenizer,
            handle,
            metrics,
            prefix_runtime,
            scheduler,
            pending,
            active,
        );
    }
    let Some(prefill_snapshot) = active.get(&prefill_req_id) else {
        return false;
    };
    if prefill_snapshot.delta_closed()
        || !prefill_snapshot.request_state.is_qwen3()
        || prefill_snapshot.request_state.is_dflash_enabled()
    {
        return false;
    }
    if !decode_req_ids.iter().all(|req_id| {
        active.get(req_id).is_some_and(|request| {
            !request.delta_closed()
                && request.request_state.is_qwen3()
                && !request.request_state.is_dflash_enabled()
        })
    }) {
        return false;
    }

    let mut decode_rows = Vec::with_capacity(decode_req_ids.len());
    for req_id in decode_req_ids {
        let Some(request) = active.remove(&req_id) else {
            warn!(
                "Metal mixed batch referenced missing decode request {:?}",
                req_id
            );
            scheduler.finish_request(req_id, None);
            continue;
        };
        if request.delta_closed() {
            scheduler.finish_request(req_id, request_mode(&request));
            continue;
        }
        decode_rows.push((req_id, request));
    }

    let Some(mut prefill_request) = active.remove(&prefill_req_id) else {
        for (req_id, request) in decode_rows {
            active.insert(req_id, request);
        }
        return false;
    };
    if prefill_request.delta_closed() {
        scheduler.finish_request(prefill_req_id, request_mode(&prefill_request));
        if let Err(err) = prefill_request.cancel() {
            warn!(
                "Metal request cancel failed for {:?}: {err:#}",
                prefill_req_id
            );
        }
        for (req_id, request) in decode_rows {
            active.insert(req_id, request);
        }
        return true;
    }

    let outcome = {
        let mut decode_refs: Vec<&mut MetalRequestState<'static>> = decode_rows
            .iter_mut()
            .map(|(_, request)| &mut request.request_state)
            .collect();
        MetalRequestState::try_mixed_batch(
            &mut decode_refs,
            &mut prefill_request.request_state,
            prefill_budget,
        )
    };

    let Some(MetalMixedBatchResult {
        decode_tokens,
        prefill,
    }) = (match outcome {
        Ok(result) => result,
        Err(err) => {
            error!("Metal mixed batch failed: {err:#}");
            metrics.record_request_failed();
            cancel_detached_request(prefill_req_id, prefill_request, scheduler);
            for (req_id, request) in decode_rows {
                cancel_detached_request(req_id, request, scheduler);
            }
            return true;
        }
    })
    else {
        active.insert(prefill_req_id, prefill_request);
        for (req_id, request) in decode_rows {
            active.insert(req_id, request);
        }
        return false;
    };

    // Mixed-batch decode rows ride the same batched GPU path as the
    // decode-only `execute_decode_batch` call site — count them in the
    // same Metal decode counters so dashboards don't undercount batched
    // throughput on mixed steps. (codex round-2 P2)
    if !decode_tokens.is_empty() {
        metrics.record_metal_decode_batch(decode_tokens.len());
    }

    for ((req_id, mut request), sampled_token) in decode_rows.into_iter().zip(decode_tokens) {
        if let Err(err) = request.process_token(sampled_token) {
            handle_detached_postprocess_error(
                "mixed decode",
                req_id,
                &err,
                request,
                metrics,
                scheduler,
            );
            continue;
        }
        finish_or_requeue_decoded_request(req_id, request, metrics, scheduler, active);
    }

    if let Some(sampled_token) = prefill.emitted_token {
        if let Err(err) = prefill_request.process_token(sampled_token) {
            handle_detached_postprocess_error(
                "mixed prefill",
                prefill_req_id,
                &err,
                prefill_request,
                metrics,
                scheduler,
            );
            return true;
        }
        if let Some(prefix_runtime) = prefix_runtime.as_mut()
            && let Err(err) = prefix_runtime.publish_prompt_prefix(&mut prefill_request)
        {
            warn!(
                "Metal live prefix publish failed for {:?}: {err:#}",
                prefill_req_id
            );
        }
    }

    if prefill_request.phase() == RuntimePhase::Finished || prefill_request.stop_hit() {
        finalize_detached_request(prefill_req_id, prefill_request, metrics, scheduler);
    } else {
        active.insert(prefill_req_id, prefill_request);
    }

    true
}

fn abort_runtime_requests(
    req_ids: &[RequestId],
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    for &req_id in req_ids {
        let mode = active.get(&req_id).and_then(request_mode);
        let _ = scheduler.finish_request(req_id, mode);
        if let Some(mut request) = active.remove(&req_id) {
            if let Err(err) = request.cancel() {
                warn!("Metal panic cleanup failed for {:?}: {err:#}", req_id);
            }
            drop(request);
        }
    }
}

fn maybe_refresh_runtime_metrics(
    metrics: &ServerMetrics,
    handle: &SchedulerHandle,
    scheduler: &MetalScheduler,
    pending: &HashMap<RequestId, PendingMetalRequest>,
    active: &HashMap<RequestId, ActiveMetalRequest>,
    last: &mut Option<Instant>,
    interval: Duration,
) {
    let now = Instant::now();
    if let Some(prev) = *last {
        if now.duration_since(prev) < interval {
            return;
        }
    }
    refresh_runtime_metrics(metrics, handle, scheduler, pending, active);
    *last = Some(now);
}

fn drain_incoming_requests(
    tokenizer: &'static Tokenizer,
    handle: &SchedulerHandle,
    metrics: &ServerMetrics,
    request_rx: &mut mpsc::UnboundedReceiver<IncomingRequest>,
    request_rx_closed: &mut bool,
    scheduler: &mut MetalScheduler,
    pending: &mut HashMap<RequestId, PendingMetalRequest>,
) {
    loop {
        match request_rx.try_recv() {
            Ok(incoming) => {
                enqueue_request(metrics, tokenizer, incoming, handle, scheduler, pending);
            }
            Err(mpsc::error::TryRecvError::Empty) => break,
            Err(mpsc::error::TryRecvError::Disconnected) => {
                *request_rx_closed = true;
                break;
            }
        }
    }
}

fn enqueue_request(
    metrics: &ServerMetrics,
    tokenizer: &'static Tokenizer,
    incoming: IncomingRequest,
    handle: &SchedulerHandle,
    scheduler: &mut MetalScheduler,
    pending: &mut HashMap<RequestId, PendingMetalRequest>,
) {
    if incoming.delta_tx.is_closed() {
        handle.consume_one();
        return;
    }

    let (pending_request, priority) = match PendingMetalRequest::from_incoming(tokenizer, incoming)
    {
        Ok(request) => request,
        Err(err) => {
            error!("Metal scheduler request init failed: {err:#}");
            metrics.record_request_failed();
            handle.consume_one();
            return;
        }
    };

    let req_id = match scheduler.submit(
        pending_request.prompt_tokens.clone(),
        pending_request.max_tokens,
        priority,
    ) {
        Ok(req_id) => req_id,
        Err(err) => {
            error!("Metal scheduler submit failed: {err}");
            metrics.record_request_failed();
            handle.consume_one();
            return;
        }
    };

    if pending.insert(req_id, pending_request).is_some() {
        warn!("Metal scheduler request id collision for {:?}", req_id);
    }
}

fn activate_pending_request(
    req_id: RequestId,
    backend: &'static MetalBackend,
    tokenizer: &'static Tokenizer,
    handle: &SchedulerHandle,
    metrics: &ServerMetrics,
    prefix_runtime: &mut Option<MetalLivePrefixRuntime>,
    scheduler: &mut MetalScheduler,
    pending: &mut HashMap<RequestId, PendingMetalRequest>,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    let Some(pending_request) = pending.remove(&req_id) else {
        warn!(
            "Metal prefill chunk referenced missing pending request {:?}",
            req_id
        );
        scheduler.finish_request(req_id, None);
        return;
    };

    if pending_request.delta_closed() {
        handle.consume_one();
        scheduler.finish_request(req_id, None);
        return;
    }

    // Always initialize DFlash when the backend has a draft model loaded;
    // concurrent DFlash rows are handled later in decode batching.
    let enable_dflash = true;
    let mut request = match pending_request.activate(backend, tokenizer, enable_dflash) {
        Ok(request) => request,
        Err(err) => {
            error!(
                "Metal scheduler activation failed for {:?}: {err:#}",
                req_id
            );
            metrics.record_request_failed();
            handle.consume_one();
            scheduler.finish_request(req_id, None);
            return;
        }
    };

    if let Some(prefix_runtime) = prefix_runtime.as_mut() {
        if let Err(err) = prefix_runtime.prepare_request(&mut request, metrics) {
            error!(
                "Metal prefix-cache activation failed for {:?}: {err:#}",
                req_id
            );
            metrics.record_request_failed();
            handle.consume_one();
            scheduler.finish_request(req_id, None);
            return;
        }
    }

    handle.consume_one();
    if active.insert(req_id, request).is_some() {
        warn!(
            "Metal scheduler activation overwrote an existing active request {:?}",
            req_id
        );
    }
}

fn execute_prefill_chunk(
    req_id: RequestId,
    mut budget: usize,
    backend: &'static MetalBackend,
    tokenizer: &'static Tokenizer,
    handle: &SchedulerHandle,
    metrics: &ServerMetrics,
    prefix_runtime: &mut Option<MetalLivePrefixRuntime>,
    scheduler: &mut MetalScheduler,
    pending: &mut HashMap<RequestId, PendingMetalRequest>,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    if !active.contains_key(&req_id) {
        activate_pending_request(
            req_id,
            backend,
            tokenizer,
            handle,
            metrics,
            prefix_runtime,
            scheduler,
            pending,
            active,
        );
    }
    if !active.contains_key(&req_id) {
        return;
    }

    if let Err((owner_req_id, err)) = drain_other_qwen35_cpp_sessions(req_id, active) {
        error!(
            "Metal prefill session handoff failed before prefilling {:?}: owner {:?}: {err:#}",
            req_id, owner_req_id
        );
        metrics.record_request_failed();
        if owner_req_id != req_id {
            cancel_request(owner_req_id, scheduler, active);
        }
        cancel_request(req_id, scheduler, active);
        return;
    }

    // DFlash requires full-prompt prefill in one shot because
    // `qwen3_forward_with_hidden_states` captures hidden states for all
    // positions — chunked KV-only prefill can't produce them. Override the
    // scheduler's chunk budget to process the entire remaining prompt.
    if let Some(request) = active.get(&req_id) {
        if request.request_state.is_dflash_enabled() {
            let remaining = request
                .request_state
                .prompt_len()
                .saturating_sub(request.request_state.prompt_progress());
            budget = budget.max(remaining);
        }
    }

    let outcome = {
        let Some(request) = active.get_mut(&req_id) else {
            warn!(
                "Metal prefill chunk referenced missing request {:?}",
                req_id
            );
            scheduler.finish_request(req_id, None);
            return;
        };

        if request.delta_closed() {
            PrefillChunkOutcome::ClientDropped
        } else {
            match request.prefill_chunk(budget) {
                Ok(emitted_token) => PrefillChunkOutcome::Progress {
                    emitted_token,
                    runtime_finished: request.phase() == RuntimePhase::Finished,
                    stop_hit: request.stop_hit(),
                },
                Err(err) => {
                    if request.delta_closed() {
                        PrefillChunkOutcome::ClientDropped
                    } else {
                        PrefillChunkOutcome::Failed(err)
                    }
                }
            }
        }
    };

    match outcome {
        PrefillChunkOutcome::Progress {
            emitted_token,
            runtime_finished,
            stop_hit,
        } => {
            if let Some(_token) = emitted_token {
                if let Some(prefix_runtime) = prefix_runtime.as_mut()
                    && let Some(request) = active.get_mut(&req_id)
                    && let Err(err) = prefix_runtime.publish_prompt_prefix(request)
                {
                    warn!("Metal live prefix publish failed for {:?}: {err:#}", req_id);
                }
            }

            if runtime_finished || stop_hit {
                finalize_request(req_id, metrics, scheduler, active);
            }
        }
        PrefillChunkOutcome::ClientDropped => cancel_request(req_id, scheduler, active),
        PrefillChunkOutcome::Failed(err) => {
            error!("Metal prefill chunk failed for {:?}: {err:#}", req_id);
            metrics.record_request_failed();
            cancel_request(req_id, scheduler, active);
        }
    }
}

fn drain_other_qwen35_cpp_sessions(
    prefill_req_id: RequestId,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) -> std::result::Result<(), (RequestId, anyhow::Error)> {
    for (req_id, request) in active.iter_mut() {
        if *req_id == prefill_req_id {
            continue;
        }
        let drained = request
            .request_state
            .drain_qwen35_cpp_session()
            .with_context(|| format!("drain Qwen3.5 C++ session for {req_id:?}"))
            .map_err(|err| (*req_id, err))?;
        if drained {
            log::debug!(
                "Metal drained Qwen3.5 C++ session for {:?} before prefilling {:?}",
                req_id,
                prefill_req_id
            );
        }
    }
    Ok(())
}

fn execute_decode_batch(
    req_ids: Vec<RequestId>,
    metrics: &ServerMetrics,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
    qwen35_decode_batch_cache: &mut Option<CachedQwen35DecodeBatch>,
) {
    if req_ids.is_empty() {
        return;
    }

    let mut staged = Vec::with_capacity(req_ids.len());
    for req_id in req_ids {
        let Some(request) = active.remove(&req_id) else {
            warn!("Metal decode batch referenced missing request {:?}", req_id);
            scheduler.finish_request(req_id, None);
            continue;
        };
        staged.push((req_id, request));
    }

    let mut open = Vec::with_capacity(staged.len());
    for (req_id, request) in staged {
        if request.delta_closed() {
            scheduler.finish_request(req_id, request_mode(&request));
            continue;
        }
        open.push((req_id, request));
    }

    // Round-3 codex findings on the partitioner are both closed:
    //   - [P2] "all-or-nothing DFlash demotion on buffered-speculative
    //     rows" — fixed at
    //     `request_state.rs::try_decode_qwen35_dflash_speculative_batch`
    //     (majority-equivalence-class per-row partition).
    //   - [P1] "plain-decode cache rollback on singleton fallback" —
    //     retracted; the `invalidate_*` sync on the `Ok(None)` arm is the
    //     only path that propagates `packed_kv_flat`/`packed_gdr_flat`
    //     updates into per-request state.
    // Partition into dflash_rows and plain_rows. Dispatch:
    //   - plain_rows (≥1): existing `execute_qwen35_packed_decode_batch`.
    //   - dflash_rows (≥2): new `execute_qwen35_dflash_packed_batch`.
    //   - dflash_rows (==1): fall through to the existing per-row
    //     `execute_decode_single` path (batched-stack overhead not worth it).
    let scheduled_open_len = open.len();
    let (dflash_requests, non_dflash): (Vec<_>, Vec<_>) = open
        .into_iter()
        .partition(|(_, request)| request.request_state.is_dflash_enabled());

    if dflash_requests.len() >= 2 {
        execute_qwen35_dflash_packed_batch(dflash_requests, metrics, scheduler, active);
    } else {
        if scheduled_open_len >= 2 && !dflash_requests.is_empty() {
            metrics.record_metal_decode_batch_fallback(dflash_requests.len());
        }
        for (req_id, request) in dflash_requests {
            execute_decode_single(req_id, request, metrics, scheduler, active);
        }
    }
    let mut open = non_dflash;

    let batch_result =
        match execute_qwen35_packed_decode_batch(&mut open, active, qwen35_decode_batch_cache) {
            Ok(Some(result)) => {
                metrics.record_metal_qwen35_packed_decode_batch(result.len());
                metrics.record_metal_decode_batch(result.len());
                Some(result)
            }
            Ok(None) => {
                invalidate_qwen35_decode_batch_cache(qwen35_decode_batch_cache, active, &mut open);
                let result = if open.is_empty() {
                    None
                } else {
                    let mut request_refs: Vec<&mut MetalRequestState<'static>> = open
                        .iter_mut()
                        .map(|(_, request)| &mut request.request_state)
                        .collect();
                    match MetalRequestState::decode_batch(&mut request_refs) {
                        Ok(result) => result,
                        Err(err) => {
                            error!("Metal batched decode failed: {err:#}");
                            metrics.record_request_failed();
                            for (req_id, request) in open {
                                cancel_detached_request(req_id, request, scheduler);
                            }
                            return;
                        }
                    }
                };
                if let Some(tokens) = result.as_ref() {
                    metrics.record_metal_decode_batch(tokens.len());
                }
                result
            }
            Err(err) => {
                error!("Metal packed Qwen3.5 decode failed: {err:#}");
                metrics.record_request_failed();
                invalidate_qwen35_decode_batch_cache(qwen35_decode_batch_cache, active, &mut open);
                for (req_id, request) in open {
                    cancel_detached_request(req_id, request, scheduler);
                }
                return;
            }
        };

    if let Some(sampled_tokens) = batch_result {
        for ((req_id, mut request), sampled_token) in open.into_iter().zip(sampled_tokens) {
            if let Err(err) = request.process_token(sampled_token) {
                handle_detached_postprocess_error(
                    "batched decode",
                    req_id,
                    &err,
                    request,
                    metrics,
                    scheduler,
                );
                continue;
            }
            finish_or_requeue_decoded_request(req_id, request, metrics, scheduler, active);
        }
        return;
    }

    if scheduled_open_len >= 1 && !open.is_empty() {
        metrics.record_metal_decode_batch_fallback(open.len());
    }
    for (req_id, request) in open {
        execute_decode_single(req_id, request, metrics, scheduler, active);
    }
}

/// Dispatch ≥2 DFlash-enabled Qwen3.5 rows through the batched speculative
/// block kernel. Mirrors `execute_qwen35_packed_decode_batch` in how sampled
/// tokens get fanned back into the scheduler via `process_token` +
/// `finish_or_requeue_decoded_request`.
///
/// No persistent cache struct (unlike the plain-decode path): the DFlash
/// verify batch re-stacks per-row target KV / GDR every tick, and the
/// scalar draft state already lives inside each `MetalRequestState`.
fn execute_qwen35_dflash_packed_batch(
    mut rows: Vec<(RequestId, ActiveMetalRequest)>,
    metrics: &ServerMetrics,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    if rows.len() < 2 {
        // Partition guard already filters on ≥2; defensive fallthrough only.
        if !rows.is_empty() {
            metrics.record_metal_decode_batch_fallback(rows.len());
        }
        for (req_id, request) in rows {
            execute_decode_single(req_id, request, metrics, scheduler, active);
        }
        return;
    }

    let outcome = {
        let mut request_refs: Vec<&mut MetalRequestState<'static>> = rows
            .iter_mut()
            .map(|(_, request)| &mut request.request_state)
            .collect();
        match MetalRequestState::try_decode_qwen35_dflash_speculative_batch(&mut request_refs) {
            Ok(Some(outcome)) => outcome,
            Ok(None) => {
                // <2 rows ready (wrong mode / phase / target_hidden not
                // captured / non-empty token_buffer / cross-row disagreement):
                // every row falls back to per-row single-path decode. Scalar
                // `decode_token` handles the stale-target_hidden, Rust-mode,
                // and buffered-drain cases cleanly.
                metrics.record_metal_decode_batch_fallback(rows.len());
                for (req_id, request) in rows {
                    execute_decode_single(req_id, request, metrics, scheduler, active);
                }
                return;
            }
            Err(err) => {
                error!("Metal Qwen3.5 DFlash batched decode failed: {err:#}");
                metrics.record_request_failed();
                for (req_id, request) in rows {
                    cancel_detached_request(req_id, request, scheduler);
                }
                return;
            }
        }
    };

    let DflashBatchOutcome {
        ready_indices,
        tokens: sampled,
    } = outcome;

    let dispatch_plan = match dflash_row_dispatch_plan(rows.len(), &ready_indices, sampled.len()) {
        Ok(plan) => plan,
        Err(err) => {
            error!(
                "Metal Qwen3.5 DFlash batched decode produced an invalid dispatch plan: {err:#}"
            );
            metrics.record_request_failed();
            for (req_id, request) in rows {
                cancel_detached_request(req_id, request, scheduler);
            }
            return;
        }
    };
    metrics.record_metal_decode_batch(ready_indices.len());
    let fallback_rows = rows.len().saturating_sub(ready_indices.len());
    if fallback_rows > 0 {
        metrics.record_metal_decode_batch_fallback(fallback_rows);
    }

    // Commit ready-row tokens and dispatch stale rows in the original scheduler
    // order (priority/arrival established by `build_decode_batch`).
    for ((req_id, mut request), dispatch) in rows.into_iter().zip(dispatch_plan) {
        if let DflashRowDispatch::Batched { sampled_index } = dispatch {
            let sampled_token = sampled[sampled_index];
            if let Err(err) = request.process_token(sampled_token) {
                handle_detached_postprocess_error(
                    "DFlash batched decode",
                    req_id,
                    &err,
                    request,
                    metrics,
                    scheduler,
                );
                continue;
            }
            finish_or_requeue_decoded_request(req_id, request, metrics, scheduler, active);
        } else {
            execute_decode_single(req_id, request, metrics, scheduler, active);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DflashRowDispatch {
    Batched { sampled_index: usize },
    ScalarFallback,
}

fn dflash_row_dispatch_plan(
    row_count: usize,
    ready_indices: &[usize],
    sampled_len: usize,
) -> Result<Vec<DflashRowDispatch>> {
    ensure!(
        sampled_len == ready_indices.len(),
        "expected {} sampled tokens, got {}",
        ready_indices.len(),
        sampled_len
    );

    let mut plan = vec![DflashRowDispatch::ScalarFallback; row_count];
    let mut previous = None;
    for (sampled_index, &row_index) in ready_indices.iter().enumerate() {
        ensure!(
            row_index < row_count,
            "ready row index {} out of range for {} rows",
            row_index,
            row_count
        );
        ensure!(
            previous.is_none_or(|prev| prev < row_index),
            "ready row indices must be sorted and unique: {:?}",
            ready_indices
        );
        plan[row_index] = DflashRowDispatch::Batched { sampled_index };
        previous = Some(row_index);
    }
    Ok(plan)
}

fn execute_qwen35_packed_decode_batch(
    open: &mut [(RequestId, ActiveMetalRequest)],
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
    cache: &mut Option<CachedQwen35DecodeBatch>,
) -> Result<Option<Vec<u32>>> {
    if open.is_empty() {
        return Ok(None);
    }

    let current_req_ids: Vec<RequestId> = open.iter().map(|(req_id, _)| *req_id).collect();

    if let Some(cached) = cache.as_mut() {
        if cached.req_ids != current_req_ids {
            if let Some(retained_rows) = retained_row_indices(&cached.req_ids, &current_req_ids) {
                cached.batch.retain_rows(&retained_rows)?;
                cached.req_ids.clone_from(&current_req_ids);
            } else if let Some(new_indices) = admit_row_indices(&cached.req_ids, &current_req_ids) {
                // Prefix-preserving grow: existing rows still first (in
                // order), new rows appended at the end. Admit when every new
                // row's own `cache_len` is `<= batch_cursor`. A row with
                // `cache_len < batch_cursor` gets left-padded up to the
                // cursor and receives its per-row RoPE offset via the
                // `rope_offsets` array passed through the bridge — so both
                // the attention mask and positional encoding stay correct.
                // A row with `cache_len > batch_cursor` would force the
                // cursor to bump and re-pad every existing row, which costs
                // more than a full rebuild, so we fall through to invalidate
                // in that case.
                let batch_cursor = cached.batch.batch_cache_len();
                let admittable = new_indices.iter().all(|&idx| {
                    open.get(idx)
                        .and_then(|(_, request)| request.request_state.qwen35_decode_cursor())
                        .is_some_and(|cache_len| cache_len <= batch_cursor)
                });
                if admittable {
                    let mut request_refs: Vec<&mut MetalRequestState<'static>> = open
                        .iter_mut()
                        .map(|(_, request)| &mut request.request_state)
                        .collect();
                    cached.batch.admit_rows(&mut request_refs, &new_indices)?;
                    cached.req_ids.clone_from(&current_req_ids);
                } else {
                    invalidate_qwen35_decode_batch_cache(cache, active, open);
                }
            } else {
                invalidate_qwen35_decode_batch_cache(cache, active, open);
            }
        }
    }

    if cache.is_none() {
        let mut request_refs: Vec<&mut MetalRequestState<'static>> = open
            .iter_mut()
            .map(|(_, request)| &mut request.request_state)
            .collect();
        let Some(batch) =
            MetalRequestState::try_build_qwen35_packed_decode_batch(&mut request_refs)?
        else {
            return Ok(None);
        };
        *cache = Some(CachedQwen35DecodeBatch {
            req_ids: current_req_ids.clone(),
            batch,
        });
    }

    let cached = cache
        .as_mut()
        .context("Qwen3.5 packed decode cache missing after build")?;
    if cached.req_ids != current_req_ids {
        invalidate_qwen35_decode_batch_cache(cache, active, open);
        return Ok(None);
    }

    let mut request_refs: Vec<&mut MetalRequestState<'static>> = open
        .iter_mut()
        .map(|(_, request)| &mut request.request_state)
        .collect();
    MetalRequestState::try_decode_qwen35_packed_batch(&mut request_refs, &mut cached.batch)
}

fn invalidate_qwen35_decode_batch_cache(
    cache: &mut Option<CachedQwen35DecodeBatch>,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
    open: &mut [(RequestId, ActiveMetalRequest)],
) {
    let Some(mut cached) = cache.take() else {
        return;
    };

    let mut row_indices = Vec::new();
    let mut state_ptrs = Vec::new();
    for (row_idx, req_id) in cached.req_ids.iter().enumerate() {
        if let Some((_, request)) = open.iter_mut().find(|(candidate, _)| candidate == req_id) {
            row_indices.push(row_idx);
            state_ptrs.push(&raw mut request.request_state);
            continue;
        }
        if let Some(request) = active.get_mut(req_id) {
            row_indices.push(row_idx);
            state_ptrs.push(&raw mut request.request_state);
        }
    }

    if row_indices.is_empty() {
        return;
    }

    if row_indices.len() != cached.req_ids.len() {
        if let Err(err) = cached.batch.retain_rows(&row_indices) {
            error!("Metal packed Qwen3.5 cache retain_rows failed during invalidate: {err:#}");
            return;
        }
    }

    let mut request_refs: Vec<&mut MetalRequestState<'static>> = state_ptrs
        .into_iter()
        .map(|ptr| unsafe { &mut *ptr })
        .collect();
    if let Err(err) =
        MetalRequestState::sync_qwen35_packed_decode_batch(&mut request_refs, &cached.batch)
    {
        error!("Metal packed Qwen3.5 cache sync failed during invalidate: {err:#}");
    }
}

fn retained_row_indices(
    previous_req_ids: &[RequestId],
    current_req_ids: &[RequestId],
) -> Option<Vec<usize>> {
    let mut indices = Vec::with_capacity(current_req_ids.len());
    let mut cursor = 0usize;
    for req_id in current_req_ids {
        let relative = previous_req_ids[cursor..]
            .iter()
            .position(|candidate| candidate == req_id)?;
        let absolute = cursor + relative;
        indices.push(absolute);
        cursor = absolute + 1;
    }
    Some(indices)
}

/// Prefix-preserving grow detector: if `current_req_ids` starts with
/// `previous_req_ids` in the exact same order, return the indices of the
/// new rows (the tail of `current_req_ids`). Otherwise return `None` and
/// the caller falls back to full invalidate.
///
/// We deliberately restrict to the prefix case rather than any supersequence
/// because `Qwen35PackedDecodeBatch::admit_rows` appends the new rows at the
/// end of the packed KV tensors — arbitrary splicing would require extra
/// `take_axis` reorders.
fn admit_row_indices(
    previous_req_ids: &[RequestId],
    current_req_ids: &[RequestId],
) -> Option<Vec<usize>> {
    if current_req_ids.len() <= previous_req_ids.len() {
        return None;
    }
    if &current_req_ids[..previous_req_ids.len()] != previous_req_ids {
        return None;
    }
    Some((previous_req_ids.len()..current_req_ids.len()).collect())
}

fn execute_decode_single(
    req_id: RequestId,
    mut request: ActiveMetalRequest,
    metrics: &ServerMetrics,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    enum Outcome {
        Progress {
            runtime_finished: bool,
            stop_hit: bool,
        },
        ClientDropped,
        Failed(anyhow::Error),
    }

    if let Err((owner_req_id, err)) = drain_other_qwen35_cpp_sessions(req_id, active) {
        error!(
            "Metal decode session handoff failed before decoding {:?}: owner {:?}: {err:#}",
            req_id, owner_req_id
        );
        metrics.record_request_failed();
        if owner_req_id != req_id {
            cancel_request(owner_req_id, scheduler, active);
        }
        cancel_detached_request(req_id, request, scheduler);
        return;
    }

    let outcome = if request.delta_closed() {
        Outcome::ClientDropped
    } else {
        metrics.record_metal_decode_scalar_row();
        match request.decode_step() {
            Ok(_sampled_token) => Outcome::Progress {
                runtime_finished: request.phase() == RuntimePhase::Finished,
                stop_hit: request.stop_hit(),
            },
            Err(err) => {
                if request.delta_closed() {
                    Outcome::ClientDropped
                } else {
                    Outcome::Failed(err)
                }
            }
        }
    };

    match outcome {
        Outcome::Progress {
            runtime_finished,
            stop_hit,
        } => {
            if runtime_finished || stop_hit {
                finalize_detached_request(req_id, request, metrics, scheduler);
            } else {
                active.insert(req_id, request);
            }
        }
        Outcome::ClientDropped => {
            scheduler.finish_request(req_id, request_mode(&request));
            if let Err(err) = request.cancel() {
                warn!("Metal request cancel failed for {:?}: {err:#}", req_id);
            }
            drop(request);
        }
        Outcome::Failed(err) => {
            error!("Metal decode step failed for {:?}: {err:#}", req_id);
            metrics.record_request_failed();
            cancel_detached_request(req_id, request, scheduler);
        }
    }
}

fn finish_or_requeue_decoded_request(
    req_id: RequestId,
    request: ActiveMetalRequest,
    metrics: &ServerMetrics,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    let runtime_finished = request.phase() == RuntimePhase::Finished;
    let stop_hit = request.stop_hit();
    if runtime_finished || stop_hit {
        finalize_detached_request(req_id, request, metrics, scheduler);
    } else {
        active.insert(req_id, request);
    }
}

fn handle_detached_postprocess_error(
    stage: &str,
    req_id: RequestId,
    request_err: &anyhow::Error,
    request: ActiveMetalRequest,
    metrics: &ServerMetrics,
    scheduler: &mut MetalScheduler,
) {
    if request.delta_closed() || is_stream_consumer_dropped(request_err) {
        info!("Metal {stage} client dropped for {:?}", req_id);
    } else {
        error!(
            "Metal {stage} post-process failed for {:?}: {request_err:#}",
            req_id
        );
        metrics.record_request_failed();
    }
    cancel_detached_request(req_id, request, scheduler);
}

fn is_stream_consumer_dropped(err: &anyhow::Error) -> bool {
    err.chain()
        .any(|cause| cause.downcast_ref::<MetalStreamError>().is_some())
}

fn cancel_detached_request(
    req_id: RequestId,
    mut request: ActiveMetalRequest,
    scheduler: &mut MetalScheduler,
) {
    scheduler.finish_request(req_id, request_mode(&request));
    if let Err(err) = request.cancel() {
        warn!("Metal request cancel failed for {:?}: {err:#}", req_id);
    }
    drop(request);
}

fn finalize_detached_request(
    req_id: RequestId,
    mut request: ActiveMetalRequest,
    metrics: &ServerMetrics,
    scheduler: &mut MetalScheduler,
) {
    scheduler.finish_request(req_id, Some(InferenceMode::Decode));
    record_request_completed(metrics, &request);
    if let Err(err) = request.send_final_delta() {
        warn!("Metal request final delta failed for {:?}: {err:#}", req_id);
    }
    drop(request);
}

fn reap_closed_clients(
    handle: &SchedulerHandle,
    scheduler: &mut MetalScheduler,
    pending: &mut HashMap<RequestId, PendingMetalRequest>,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    let pending_closed: Vec<_> = pending
        .iter()
        .filter_map(|(req_id, request)| request.delta_closed().then_some(*req_id))
        .collect();
    for req_id in pending_closed {
        handle.consume_one();
        scheduler.finish_request(req_id, None);
        pending.remove(&req_id);
    }

    let closed: Vec<_> = active
        .iter()
        .filter_map(|(req_id, request)| request.delta_closed().then_some(*req_id))
        .collect();

    for req_id in closed {
        cancel_request(req_id, scheduler, active);
    }
}

fn cancel_request(
    req_id: RequestId,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    let mode = active.get(&req_id).map(request_mode);
    scheduler.finish_request(req_id, mode.flatten());
    if let Some(mut request) = active.remove(&req_id) {
        if let Err(err) = request.cancel() {
            warn!("Metal request cancel failed for {:?}: {err:#}", req_id);
        }
        drop(request);
    }
}

fn finalize_request(
    req_id: RequestId,
    metrics: &ServerMetrics,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    scheduler.finish_request(req_id, Some(InferenceMode::Decode));
    let Some(mut request) = active.remove(&req_id) else {
        return;
    };
    record_request_completed(metrics, &request);
    if let Err(err) = request.cancel() {
        warn!("Metal request cleanup failed for {:?}: {err:#}", req_id);
    }
    if let Err(err) = request.send_final_delta()
        && !request.delta_closed()
    {
        warn!("Metal request final delta failed for {:?}: {err:#}", req_id);
    }
    drop(request);
}

fn record_request_completed(metrics: &ServerMetrics, request: &ActiveMetalRequest) {
    let completion_tokens = request.request_state.generated_tokens() as u64;
    let completed_at = Instant::now();
    let queue_wait_s = request
        .admitted_at
        .duration_since(request.enqueued_at)
        .as_secs_f64();
    let e2e_s = completed_at
        .duration_since(request.enqueued_at)
        .as_secs_f64();
    let active_ttft_s = request.first_token_at.map_or(0.0, |first| {
        first.duration_since(request.admitted_at).as_secs_f64()
    });
    let ttft_s = request.first_token_at.map_or(e2e_s, |first| {
        first.duration_since(request.enqueued_at).as_secs_f64()
    });
    let tpot_s = if completion_tokens > 1 {
        (e2e_s - ttft_s).max(0.0) / (completion_tokens - 1) as f64
    } else {
        0.0
    };
    metrics.record_request_completed_detailed(
        request.prompt_len() as u64,
        completion_tokens,
        queue_wait_s,
        active_ttft_s,
        ttft_s,
        tpot_s,
        e2e_s,
    );

    // Flush DFlash speculative decode metrics if this was a DFlash request.
    if let Some((blocks, accepted, drafted)) = request.request_state.dflash_block_stats() {
        for i in 0..blocks {
            metrics.record_dflash_block(accepted.get(i).copied().unwrap_or(0), drafted);
        }
    }
}

fn request_mode(request: &ActiveMetalRequest) -> Option<InferenceMode> {
    match request.phase() {
        RuntimePhase::Prefill => Some(InferenceMode::Prefill),
        RuntimePhase::Decode => Some(InferenceMode::Decode),
        RuntimePhase::Finished => None,
    }
}

fn scheduler_runtime_states(
    active: &HashMap<RequestId, ActiveMetalRequest>,
) -> Vec<MetalRuntimeRequestState> {
    active
        .iter()
        .filter(|(_, request)| request.phase() != RuntimePhase::Finished)
        .map(|(req_id, request)| MetalRuntimeRequestState {
            req_id: *req_id,
            phase: match request.phase() {
                RuntimePhase::Prefill => super::scheduler::MetalRequestPhase::Prefilling,
                RuntimePhase::Decode | RuntimePhase::Finished => {
                    super::scheduler::MetalRequestPhase::Decoding
                }
            },
            prompt_progress: request.request_state.prompt_progress(),
            generated_tokens: request.request_state.generated_tokens(),
            last_token: request.request_state.last_token(),
        })
        .collect()
}

fn refresh_runtime_metrics(
    metrics: &ServerMetrics,
    handle: &SchedulerHandle,
    _scheduler: &MetalScheduler,
    _pending: &HashMap<RequestId, PendingMetalRequest>,
    active: &HashMap<RequestId, ActiveMetalRequest>,
) {
    metrics.set_active(active.len() as u64);
    metrics.set_waiting(handle.waiting_count() as u64);
    let running_batch = active
        .values()
        .filter(|request| request.phase() == RuntimePhase::Decode)
        .count() as u64;
    let prefill_queue = active
        .values()
        .filter(|request| request.phase() == RuntimePhase::Prefill)
        .count() as u64;
    metrics.set_scheduler_occupancy(running_batch, prefill_queue);
    metrics.set_kv_coordinator(0, 0, 0, 0, false, false, 0, 0, 0, 0);
    metrics.set_tier_wait_seconds(0.0, 0.0);

    let (kv_used, kv_total) = active.values().fold((0u64, 0u64), |acc, request| {
        if let Some((used, total)) = request.request_state.kv_pool_usage() {
            (acc.0 + used as u64, acc.1 + total as u64)
        } else {
            acc
        }
    });
    metrics.set_kv_gpu_blocks(kv_total.saturating_sub(kv_used), kv_total);
    metrics.set_memory_bytes(
        super::mlx::active_memory_bytes(),
        super::mlx::peak_memory_bytes(),
        super::mlx::cache_memory_bytes(),
    );
}

fn map_request_priority(priority: RequestPriority) -> MetalRequestPriority {
    match priority {
        RequestPriority::Low => MetalRequestPriority::Low,
        RequestPriority::Normal => MetalRequestPriority::Normal,
        RequestPriority::High => MetalRequestPriority::High,
    }
}

fn map_finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("length") => FinishReason::Length,
        _ => FinishReason::Stop,
    }
}

fn send_text_delta_with_ids(
    delta_tx: &mpsc::UnboundedSender<CompletionStreamDelta>,
    text_delta: String,
    token_ids: Vec<u32>,
) -> Result<()> {
    // We still want to push token_ids even when the text delta is empty,
    // because the stop processor sometimes withholds bytes while the
    // decoder has already consumed token IDs that we must surface in
    // the trajectory. Empty text + empty IDs is the only case we drop.
    if text_delta.is_empty() && token_ids.is_empty() {
        return Ok(());
    }

    delta_tx
        .send(CompletionStreamDelta {
            text_delta,
            finish_reason: None,
            usage: None,
            logprob: None,
            token_ids,
        })
        .map_err(|_| MetalStreamError::ConsumerDropped.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::metal::mlx::MlxArray;
    use crate::test_support::metal_test_guard;
    use tempfile::tempdir;

    #[test]
    fn dflash_row_dispatch_plan_preserves_scheduler_order() {
        let plan = dflash_row_dispatch_plan(8, &[0, 2, 5], 3).expect("plan");

        assert_eq!(
            plan,
            vec![
                DflashRowDispatch::Batched { sampled_index: 0 },
                DflashRowDispatch::ScalarFallback,
                DflashRowDispatch::Batched { sampled_index: 1 },
                DflashRowDispatch::ScalarFallback,
                DflashRowDispatch::ScalarFallback,
                DflashRowDispatch::Batched { sampled_index: 2 },
                DflashRowDispatch::ScalarFallback,
                DflashRowDispatch::ScalarFallback,
            ]
        );
    }

    #[test]
    fn dflash_row_dispatch_plan_rejects_invalid_outcome_shape() {
        assert!(dflash_row_dispatch_plan(3, &[0, 2], 1).is_err());
        assert!(dflash_row_dispatch_plan(3, &[0, 3], 2).is_err());
        assert!(dflash_row_dispatch_plan(3, &[2, 1], 2).is_err());
        assert!(dflash_row_dispatch_plan(3, &[1, 1], 2).is_err());
    }

    #[test]
    fn stream_consumer_drop_detection_is_typed() {
        let dropped: anyhow::Error = MetalStreamError::ConsumerDropped.into();
        assert!(is_stream_consumer_dropped(&dropped));

        let other = anyhow::anyhow!("stream consumer dropped");
        assert!(!is_stream_consumer_dropped(&other));
    }

    #[test]
    fn qwen35_disk_prefix_runtime_reconciles_persisted_snapshot_headers() {
        let _guard = metal_test_guard();
        let dir = tempdir().expect("tempdir");
        let store = Arc::new(DiskStore::new(dir.path()));
        let model_fingerprint = b"qwen35-test-model".to_vec();
        let mut runtime = MetalQwen35PrefixRuntime::new(
            64,
            2,
            Some(store.clone()),
            model_fingerprint.clone(),
            None,
            0.90,
            0.75,
            false,
        )
        .expect("runtime");
        let snapshot = Qwen35PrefixSnapshot {
            token_ids: vec![11, 12],
            kv_flat: vec![MlxArray::from_slice_i32(&[1, 2], &[2])],
            gdr_flat: Vec::new(),
            cache_len: 2,
            kv_capacity: 2,
        };

        runtime.persist_snapshot(&snapshot).expect("persist");
        assert!(runtime.disk_entries.contains_key(&vec![11, 12]));
        let qwen35_fingerprint = runtime.fingerprint_for_tokens(&[11, 12]);
        let foreign_fingerprint = BlockFingerprint([0x7a; 16]);
        store
            .put_block_with_fsync(foreign_fingerprint, 0x99, b"not-a-qwen35-snapshot", false)
            .expect("persist foreign block");
        let disk_bytes = runtime.disk_bytes;
        assert!(disk_bytes > 0);

        let restored = MetalQwen35PrefixRuntime::new(
            64,
            2,
            Some(store.clone()),
            model_fingerprint,
            None,
            0.90,
            0.75,
            false,
        )
        .expect("restored runtime");
        assert!(restored.disk_entries.contains_key(&vec![11, 12]));
        assert_eq!(restored.disk_bytes, disk_bytes);

        let wrong_model = MetalQwen35PrefixRuntime::new(
            64,
            2,
            Some(store.clone()),
            b"other-model".to_vec(),
            None,
            0.90,
            0.75,
            false,
        )
        .expect("wrong-model runtime");
        assert!(wrong_model.disk_entries.is_empty());
        assert!(
            !store
                .contains_block(qwen35_fingerprint)
                .expect("stat stale block"),
            "wrong-model Qwen3.5 snapshot blocks should be discarded during reconciliation"
        );
        assert!(
            store
                .contains_block(foreign_fingerprint)
                .expect("stat foreign block"),
            "non-Qwen3.5 DiskStore blocks should not be deleted by Qwen3.5 reconciliation"
        );
    }

    #[test]
    fn metal_prefix_model_fingerprint_binds_selected_source_without_mtime() {
        let dir = tempdir().expect("tempdir");
        let gguf_a = dir.path().join("a.gguf");
        let gguf_b = dir.path().join("b.gguf");
        let tokenizer = dir.path().join("_gguf_tokenizer.json");
        std::fs::write(&gguf_a, b"same-size-a").expect("write a");
        std::fs::write(&gguf_b, b"same-size-b").expect("write b");
        std::fs::write(&tokenizer, br#"{"tokenizer":"stable"}"#).expect("write tokenizer");

        let mut backend = MetalBackend::with_options(MetalBackendOptions::default());
        backend.model_dir = Some(dir.path().to_path_buf());
        backend.model_source_path = Some(gguf_a.clone());
        let fp_a = metal_prefix_model_fingerprint(&backend).expect("fingerprint a");

        std::fs::write(&tokenizer, br#"{"tokenizer":"stable"}"#).expect("rewrite tokenizer");
        let fp_a_after_rewrite =
            metal_prefix_model_fingerprint(&backend).expect("fingerprint a after rewrite");
        assert_eq!(fp_a, fp_a_after_rewrite);

        std::fs::write(&gguf_b, b"same-size-z").expect("replace unrelated b same size");
        let fp_a_after_unrelated_weight =
            metal_prefix_model_fingerprint(&backend).expect("fingerprint a after unrelated b");
        assert_eq!(fp_a, fp_a_after_unrelated_weight);

        std::fs::write(&gguf_a, b"same-size-c").expect("replace a same size");
        let fp_a_replaced =
            metal_prefix_model_fingerprint(&backend).expect("fingerprint a after replacement");
        assert_ne!(fp_a, fp_a_replaced);

        backend.model_source_path = Some(gguf_b);
        let fp_b = metal_prefix_model_fingerprint(&backend).expect("fingerprint b");
        assert_ne!(fp_a, fp_b);
    }
}
