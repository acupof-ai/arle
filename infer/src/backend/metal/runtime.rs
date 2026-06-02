use std::collections::HashMap;
use std::io::Read;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
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
use crate::kv_tier::{
    BlockId, ChunkedSnapshotLocation, ChunkedSnapshotManifest, ChunkedSnapshotPartWrite,
    ChunkedSnapshotRead, ChunkedSnapshotStore, ChunkedSnapshotWrite, KvTierAdapter, Tier,
};
use crate::metrics::ServerMetrics;
use crate::model_arch::ModelArchInfo;
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
    /// Cooperative cancel flag (in-process CLI Ctrl-C path). When set, the
    /// runtime stops/reaps this request at the next tick/chunk boundary.
    cancel: Option<Arc<AtomicBool>>,
}

impl PendingMetalRequest {
    fn from_incoming(
        tokenizer: &Tokenizer,
        mut incoming: IncomingRequest,
    ) -> Result<(Self, MetalRequestPriority)> {
        let prompt_tokens = match incoming.prompt_tokens.take() {
            Some(tokens) => tokens,
            None => tokenizer.encode(&incoming.prompt)?,
        };
        ensure!(
            !prompt_tokens.is_empty(),
            "Metal scheduler request requires at least one prompt token"
        );
        Ok((
            Self {
                delta_tx: incoming.delta_tx,
                prompt_tokens,
                max_tokens: incoming.max_tokens,
                sampling: incoming.sampling,
                stop: incoming.stop,
                session_id: incoming.session_id,
                enqueued_at: Instant::now(),
                cancel: incoming.cancel,
            },
            map_request_priority(incoming.priority),
        ))
    }

    fn delta_closed(&self) -> bool {
        self.delta_tx.is_closed()
    }

    /// True when this request should stop generating: either the streaming
    /// consumer dropped (`delta_closed`) or the cooperative cancel flag is set
    /// (in-process CLI Ctrl-C). Drives reap/stop decisions at tick granularity.
    fn cancel_requested(&self) -> bool {
        self.delta_closed()
            || self
                .cancel
                .as_ref()
                .is_some_and(|c| c.load(Ordering::Relaxed))
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
    prefix_reused_tokens: usize,
    prefix_reuse_source: PrefixReuseSource,
    /// Full generated-token history for cache keys. This is deliberately
    /// separate from `pending_token_ids`, which is only a streaming transport
    /// buffer and may be drained before request finalization.
    generated_token_ids: Vec<u32>,
    /// Phase 2 trajectory token layer. Each `process_token` pushes the
    /// just-sampled id; whenever a text delta is actually sent
    /// (post stop-processor / decoder buffering), the pending IDs are
    /// drained onto that delta. Any IDs still pending at finish time
    /// ride on the final delta so the cumulative `response_token_ids`
    /// the consumer collates equals every generated token.
    pending_token_ids: Vec<u32>,
    /// Cooperative cancel flag carried over from the pending request. When set,
    /// decode/prefill pre-checks stop the request at the next token/chunk.
    cancel: Option<Arc<AtomicBool>>,
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
        let cancel = pending.cancel;
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
        let mtp_ref = if enable_dflash {
            unsafe { backend.mtp_runtime_static() }
        } else {
            None
        };
        let request_state = backend.create_request_state_with_specs(
            &prompt_tokens,
            &sampling,
            dflash_ref,
            mtp_ref,
        )?;
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
            prefix_reused_tokens: 0,
            prefix_reuse_source: PrefixReuseSource::None,
            generated_token_ids: Vec::new(),
            pending_token_ids: Vec::new(),
            cancel,
        })
    }

    fn delta_closed(&self) -> bool {
        self.delta_tx.is_closed()
    }

    /// True when this active request should stop generating: streaming consumer
    /// dropped (`delta_closed`) or the cooperative cancel flag is set (in-process
    /// CLI Ctrl-C). Drives prefill/decode pre-checks and reap.
    fn cancel_requested(&self) -> bool {
        self.delta_closed()
            || self
                .cancel
                .as_ref()
                .is_some_and(|c| c.load(Ordering::Relaxed))
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
            error: None,
        });
        Ok(())
    }

    fn process_token(&mut self, token_id: u32) -> Result<()> {
        if self.first_token_at.is_none() {
            self.first_token_at = Some(Instant::now());
        }
        self.generated_token_ids.push(token_id);
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

    fn materialized_session_tokens(&self, cache_len: usize) -> Result<Vec<u32>> {
        materialized_session_tokens_for_snapshot(
            &self.prompt_tokens,
            &self.generated_token_ids,
            cache_len,
        )
    }

    fn prompt_len(&self) -> usize {
        self.prompt_tokens.len()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PrefixReuseSource {
    None,
    Memory,
    Disk,
}

const METAL_PREFIX_BLOCK_SIZE: usize = 16;
const METAL_QWEN35_SNAPSHOT_KV_FORMAT_TAG: u8 = 0x35;
const METAL_QWEN35_CHUNKED_SNAPSHOT_NAMESPACE: &str = "metal-qwen35-prefix-v1";
const METRICS_REFRESH_INTERVAL: Duration = Duration::from_millis(40);

// SSD persist budget gate (M_e.14 memory-first prefix cache). A full-length
// snapshot is only worth persisting to SSD when re-prefilling the same prompt
// from scratch would cost more than reading the snapshot back, with margin.
//
//   worth_persist = prefill_cost_us > READBACK_US_PER_TOKEN * tokens * SAFETY
//
// `prefill_cost_us = first_token_at - admitted_at` (the wall-clock the live
// request already paid to reach token1). `READBACK_US_PER_TOKEN` is seeded
// from the M_e.13 measured `read_us + decode_us + import_us` at 2064 tokens
// (~48 µs/token); `SAFETY` leaves margin so we only persist clear wins.
// Both are env-tunable (`INFER_METAL_PREFIX_READBACK_US_PER_TOKEN`,
// `INFER_METAL_PREFIX_PERSIST_SAFETY`) following the `mlx.rs` parse idiom.
const METAL_PREFIX_READBACK_US_PER_TOKEN_DEFAULT: f64 = 48.0;
const METAL_PREFIX_PERSIST_SAFETY_DEFAULT: f64 = 2.0;
const METAL_PREFIX_SSD_PENDING_BYTES_DEFAULT: u64 = 1024 * 1024 * 1024;
const METAL_PREFIX_PERSIST_MIN_EXTENSION_TOKENS_DEFAULT: usize = 64;

fn metal_prefix_readback_us_per_token() -> f64 {
    std::env::var("INFER_METAL_PREFIX_READBACK_US_PER_TOKEN")
        .ok()
        .and_then(|raw| raw.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(METAL_PREFIX_READBACK_US_PER_TOKEN_DEFAULT)
}

fn metal_prefix_persist_safety() -> f64 {
    std::env::var("INFER_METAL_PREFIX_PERSIST_SAFETY")
        .ok()
        .and_then(|raw| raw.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(METAL_PREFIX_PERSIST_SAFETY_DEFAULT)
}

fn metal_prefix_ssd_pending_bytes_limit() -> u64 {
    std::env::var("INFER_METAL_PREFIX_SSD_PENDING_BYTES")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(METAL_PREFIX_SSD_PENDING_BYTES_DEFAULT)
}

fn metal_prefix_persist_min_extension_tokens() -> usize {
    std::env::var("INFER_METAL_PREFIX_PERSIST_MIN_EXTENSION_TOKENS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(METAL_PREFIX_PERSIST_MIN_EXTENSION_TOKENS_DEFAULT)
}

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

/// Resident byte budget for an in-memory cached snapshot. Count the retained
/// MLX KV+GDR arrays, not token length: the live driver pre-allocates KV to
/// `prompt_len + max_new_tokens`, and the snapshot keeps those arrays alive.
/// Token-equivalent accounting let large snapshots hide under an oversized
/// `max_batch_tokens * multiplier` budget.
fn snapshot_resident_bytes(snapshot: &Qwen35PrefixSnapshot) -> u64 {
    snapshot
        .kv_flat
        .iter()
        .chain(snapshot.gdr_flat.iter())
        .map(|array| array.nbytes() as u64)
        .sum()
}

fn materialized_session_tokens_for_snapshot(
    prompt_tokens: &[u32],
    generated_token_ids: &[u32],
    cache_len: usize,
) -> Result<Vec<u32>> {
    ensure!(
        cache_len >= prompt_tokens.len(),
        "materialized Qwen3.5 session cache_len {} is shorter than prompt {}",
        cache_len,
        prompt_tokens.len()
    );
    let generated_len = cache_len - prompt_tokens.len();
    ensure!(
        generated_len <= generated_token_ids.len(),
        "materialized Qwen3.5 session needs {} generated tokens, only {} recorded",
        generated_len,
        generated_token_ids.len()
    );

    let mut tokens = Vec::with_capacity(cache_len);
    tokens.extend_from_slice(prompt_tokens);
    tokens.extend_from_slice(&generated_token_ids[..generated_len]);
    Ok(tokens)
}

fn should_persist_metal_prefix_snapshot(
    prefill_cost_us: f64,
    prompt_len: usize,
    snapshot_tokens: usize,
    reused_tokens: usize,
    block_size: usize,
    readback_us_per_token: f64,
    safety: f64,
    min_extension_tokens: usize,
) -> bool {
    if let Some(extension_delta) =
        metal_prefix_extension_delta_tokens(prompt_len, snapshot_tokens, reused_tokens, block_size)
        && extension_delta < min_extension_tokens
    {
        return false;
    }
    if reused_tokens >= block_size && snapshot_tokens > reused_tokens {
        return true;
    }
    let prompt_len = prompt_len.max(1);
    let saved_prefill_us = prefill_cost_us * (snapshot_tokens as f64 / prompt_len as f64);
    let readback_cost_us = readback_us_per_token * snapshot_tokens as f64 * safety;
    saved_prefill_us > readback_cost_us
}

fn metal_prefix_extension_delta_tokens(
    prompt_len: usize,
    snapshot_tokens: usize,
    reused_tokens: usize,
    block_size: usize,
) -> Option<usize> {
    if reused_tokens >= block_size && snapshot_tokens > reused_tokens {
        return Some(snapshot_tokens - reused_tokens);
    }
    if snapshot_tokens > prompt_len {
        return Some(snapshot_tokens - prompt_len);
    }
    None
}

struct MetalQwen35CachedPrefix {
    snapshot: Qwen35PrefixSnapshot,
    last_used_tick: u64,
}

struct MetalQwen35DiskPrefix {
    location: ChunkedSnapshotLocation,
    last_used_tick: u64,
}

/// Shared SSD prefix index. Guarded by a `Mutex` because two threads touch it:
/// the serving thread (lookup / import-side `touch`/`remove` / startup
/// reconcile) and the dedicated SSD-writer thread (`persist` bookkeeping after
/// a budget-gated async write). Bytes-on-disk accounting and LRU eviction live
/// here so the two threads never disagree about disk occupancy. The actual file
/// I/O (`put`/`get`/`delete`) goes through `MetalTierAdapter` and is keyed by
/// per-block fingerprint, so concurrent reads/writes hit distinct files and do
/// not need the index lock held across the syscall.
struct MetalDiskPrefixIndex {
    entries: HashMap<Vec<u32>, MetalQwen35DiskPrefix>,
    disk_bytes: u64,
    max_disk_bytes: Option<u64>,
    high_watermark: f64,
    low_watermark: f64,
    next_tick: u64,
}

impl MetalDiskPrefixIndex {
    fn new(max_disk_bytes: Option<u64>, high_watermark: f64, low_watermark: f64) -> Self {
        Self {
            entries: HashMap::new(),
            disk_bytes: 0,
            max_disk_bytes,
            high_watermark,
            low_watermark,
            next_tick: 1,
        }
    }

    fn bump_tick(&mut self) -> u64 {
        let tick = self.next_tick;
        self.next_tick = self.next_tick.saturating_add(1);
        tick
    }

    fn contains(&self, key: &[u32]) -> bool {
        self.entries.contains_key(key)
    }

    fn location_for(&self, key: &[u32]) -> Option<ChunkedSnapshotLocation> {
        self.entries.get(key).map(|entry| entry.location.clone())
    }

    fn lookup_longest_prefix(&self, prompt_tokens: &[u32], block_size: usize) -> Option<Vec<u32>> {
        // Strict extension only. Exact-prompt reuse needs a separate state
        // transition from imported Prefill to Decode; the current prefill path
        // must still run the terminal prompt step to sample the first token.
        self.entries
            .keys()
            .filter(|tokens| {
                let prefix_len = tokens.len();
                prefix_len >= block_size
                    && prefix_len < prompt_tokens.len()
                    && prompt_tokens.starts_with(tokens.as_slice())
            })
            .max_by_key(|tokens| tokens.len())
            .cloned()
    }

    fn touch(&mut self, key: &[u32]) {
        let tick = self.bump_tick();
        if let Some(entry) = self.entries.get_mut(key) {
            entry.last_used_tick = tick;
        }
    }

    fn insert(&mut self, key: Vec<u32>, location: ChunkedSnapshotLocation) {
        let tick = self.bump_tick();
        self.disk_bytes = self.disk_bytes.saturating_add(location.payload_len);
        self.entries.insert(
            key,
            MetalQwen35DiskPrefix {
                location,
                last_used_tick: tick,
            },
        );
    }

    /// Remove an index entry. Returns the location so the caller can delete the
    /// on-disk file outside the lock (file I/O does not need the index held).
    fn remove(&mut self, key: &[u32]) -> Option<ChunkedSnapshotLocation> {
        let entry = self.entries.remove(key)?;
        self.disk_bytes = self.disk_bytes.saturating_sub(entry.location.payload_len);
        Some(entry.location)
    }

    /// Pop the least-recently-used entry to make room. Returns its location so
    /// the caller can delete the file outside the lock.
    fn pop_lru(&mut self) -> Option<ChunkedSnapshotLocation> {
        let lru_key = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_used_tick)
            .map(|(tokens, _)| tokens.clone())?;
        self.remove(&lru_key)
    }

    /// Decide whether `needed_bytes` fits under the high watermark, returning
    /// any LRU locations that must be deleted to make room. Mutates only the
    /// in-memory accounting; the caller deletes the returned files.
    fn reserve_capacity(&mut self, needed_bytes: u64) -> DiskCapacityDecision {
        let Some(max_bytes) = self.max_disk_bytes else {
            return DiskCapacityDecision {
                fits: true,
                evicted: Vec::new(),
            };
        };
        let high = watermark_bytes(max_bytes, self.high_watermark);
        let low = watermark_bytes(max_bytes, self.low_watermark);
        if needed_bytes > high {
            return DiskCapacityDecision {
                fits: false,
                evicted: Vec::new(),
            };
        }
        if self.disk_bytes.saturating_add(needed_bytes) <= high {
            return DiskCapacityDecision {
                fits: true,
                evicted: Vec::new(),
            };
        }

        let mut evicted = Vec::new();
        let target = low.saturating_sub(needed_bytes);
        while self.disk_bytes > target {
            let Some(location) = self.pop_lru() else {
                break;
            };
            evicted.push(location);
        }
        DiskCapacityDecision {
            fits: self.disk_bytes.saturating_add(needed_bytes) <= high,
            evicted,
        }
    }
}

struct DiskCapacityDecision {
    fits: bool,
    evicted: Vec<ChunkedSnapshotLocation>,
}

/// Job handed to the dedicated SSD-writer thread. The snapshot parts are already
/// encoded (`encode_chunked_for_disk` → `to_bytes`/`eval`) on the **serving** thread,
/// because that `eval` materializes resident MLX arrays and must not run
/// concurrently with the serving thread's decode `async_eval` (no global MLX
/// lock; dedicated GPU streams — see `feedback_mlx_async_eval_is_caller_thread`).
/// The writer therefore does **only** filesystem I/O + index bookkeeping.
struct DiskWriteJob {
    token_ids: Vec<u32>,
    manifest_id: BlockFingerprint,
    metadata: Vec<u8>,
    parts: Vec<ChunkedSnapshotPartWrite>,
    estimated_payload_len: u64,
}

struct PendingDiskPayload {
    pending_bytes: Arc<AtomicU64>,
    bytes: u64,
}

impl PendingDiskPayload {
    fn new(pending_bytes: Arc<AtomicU64>, bytes: u64) -> Self {
        Self {
            pending_bytes,
            bytes,
        }
    }
}

impl Drop for PendingDiskPayload {
    fn drop(&mut self) {
        if self.bytes > 0 {
            self.pending_bytes.fetch_sub(self.bytes, Ordering::AcqRel);
        }
    }
}

/// Owns the dedicated SSD-writer thread + its job channel. Dropping the handle
/// closes the channel; the writer drains remaining jobs and exits.
struct DiskWriteHandle {
    tx: Option<std_mpsc::Sender<DiskWriteJob>>,
    join: Option<std::thread::JoinHandle<()>>,
    pending_bytes: Arc<AtomicU64>,
    max_pending_bytes: u64,
}

impl DiskWriteHandle {
    fn spawn(
        index: Arc<Mutex<MetalDiskPrefixIndex>>,
        adapter: MetalTierAdapter,
        fsync_each_block: bool,
    ) -> Self {
        let (tx, rx) = std_mpsc::channel::<DiskWriteJob>();
        let pending_bytes = Arc::new(AtomicU64::new(0));
        let writer_pending_bytes = Arc::clone(&pending_bytes);
        let join = std::thread::Builder::new()
            .name("metal-prefix-ssd-writer".to_string())
            .spawn(move || {
                disk_writer_loop(rx, &index, &adapter, fsync_each_block, writer_pending_bytes);
            })
            .expect("spawn Metal prefix SSD-writer thread");
        Self {
            tx: Some(tx),
            join: Some(join),
            pending_bytes,
            max_pending_bytes: metal_prefix_ssd_pending_bytes_limit(),
        }
    }

    fn try_reserve_pending(&self, bytes: u64, token_count: usize) -> bool {
        if bytes == 0 {
            return true;
        }
        if bytes > self.max_pending_bytes {
            warn!(
                "Metal Qwen3.5 SSD prefix persist too large for pending queue; \
                 tokens={token_count} job_bytes={bytes} limit_bytes={}",
                self.max_pending_bytes
            );
            return false;
        }

        let mut current = self.pending_bytes.load(Ordering::Acquire);
        loop {
            let Some(next) = current.checked_add(bytes) else {
                warn!(
                    "Metal Qwen3.5 SSD prefix pending-byte counter overflow; \
                     tokens={token_count} job_bytes={bytes} current_bytes={current}"
                );
                return false;
            };
            if next > self.max_pending_bytes {
                warn!(
                    "Metal Qwen3.5 SSD prefix pending queue full; dropping persist job; \
                     tokens={token_count} job_bytes={bytes} pending_bytes={current} \
                     limit_bytes={}",
                    self.max_pending_bytes
                );
                return false;
            }
            match self.pending_bytes.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(observed) => current = observed,
            }
        }
    }

    fn release_pending(&self, bytes: u64) {
        if bytes > 0 {
            self.pending_bytes.fetch_sub(bytes, Ordering::AcqRel);
        }
    }

    fn submit_reserved(&self, job: DiskWriteJob) {
        let Some(tx) = self.tx.as_ref() else {
            self.release_pending(job.estimated_payload_len);
            return;
        };
        let payload_len = job.estimated_payload_len;
        if tx.send(job).is_err() {
            self.release_pending(payload_len);
            warn!("Metal Qwen3.5 SSD prefix writer thread is gone; dropping persist job");
        }
    }
}

impl Drop for DiskWriteHandle {
    fn drop(&mut self) {
        // Drop the sender FIRST so the writer's `recv()` returns `Err` and the
        // loop exits; only then is it safe to join. Joining before dropping the
        // sender would deadlock (the writer would block forever on `recv`).
        self.tx.take();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

/// SSD-writer thread body. The exclusive performer of disk **writes**: budget
/// has already cleared on the serving thread; here we reserve capacity, write
/// the block, and record it in the shared index. Reads/imports still happen on
/// the serving thread against the same `Arc<Mutex<MetalDiskPrefixIndex>>`.
fn disk_writer_loop(
    rx: std_mpsc::Receiver<DiskWriteJob>,
    index: &Arc<Mutex<MetalDiskPrefixIndex>>,
    adapter: &MetalTierAdapter,
    fsync_each_block: bool,
    pending_bytes: Arc<AtomicU64>,
) {
    let trace = std::env::var("INFER_M_E13_TRACE").is_ok();
    while let Ok(job) = rx.recv() {
        let _pending_payload =
            PendingDiskPayload::new(Arc::clone(&pending_bytes), job.estimated_payload_len);
        // Skip if a concurrent import already re-published this exact key.
        let reserve = {
            let mut guard = match index.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            if guard.contains(&job.token_ids) {
                guard.touch(&job.token_ids);
                continue;
            }
            guard.reserve_capacity(job.estimated_payload_len)
        };
        for location in &reserve.evicted {
            if let Err(err) = adapter.delete_disk_snapshot(location) {
                warn!(
                    "Metal Qwen3.5 SSD prefix cache failed to evict {}: {err:#}",
                    location.path.display()
                );
            }
        }
        if !reserve.fits {
            continue;
        }

        let t_write = std::time::Instant::now();
        let (location, stats) = match adapter.put_disk_snapshot(
            job.manifest_id,
            METAL_QWEN35_CHUNKED_SNAPSHOT_NAMESPACE,
            job.metadata,
            job.parts,
            fsync_each_block,
        ) {
            Ok(result) => result,
            Err(err) => {
                warn!(
                    "Metal Qwen3.5 SSD prefix publish failed for {} tokens: {err:#}",
                    job.token_ids.len()
                );
                continue;
            }
        };
        let write_us = t_write.elapsed().as_micros();
        let payload_len = location.payload_len;
        {
            let mut guard = match index.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            guard.insert(job.token_ids.clone(), location);
        }
        if trace {
            log::info!(
                "m_e13_trace ssd_writer persist: tokens={} payload_bytes={} chunks_written={} chunks_reused={} physical_chunk_bytes_written={} manifest_bytes_written={} write_us={}",
                job.token_ids.len(),
                payload_len,
                stats.chunks_written,
                stats.chunks_reused,
                stats.physical_chunk_bytes_written,
                stats.manifest_bytes_written,
                write_us,
            );
        }
    }
}

#[derive(Clone)]
struct MetalTierAdapter {
    snapshot_store: Option<Arc<ChunkedSnapshotStore>>,
    paged_pool_pressure: f64,
}

impl MetalTierAdapter {
    fn new(snapshot_store: Option<Arc<ChunkedSnapshotStore>>) -> Self {
        Self {
            snapshot_store,
            paged_pool_pressure: 0.0,
        }
    }

    fn with_paged_pool_pressure(mut self, pressure: f64) -> Self {
        self.set_paged_pool_pressure(pressure);
        self
    }

    fn set_paged_pool_pressure(&mut self, pressure: f64) {
        self.paged_pool_pressure = normalize_paged_pool_pressure(pressure);
    }

    fn has_disk_tier(&self) -> bool {
        self.snapshot_store.is_some()
    }

    fn put_disk_snapshot(
        &self,
        manifest_id: BlockFingerprint,
        namespace: &str,
        metadata: Vec<u8>,
        parts: Vec<ChunkedSnapshotPartWrite>,
        fsync_manifest: bool,
    ) -> Result<(
        ChunkedSnapshotLocation,
        crate::kv_tier::ChunkedSnapshotPutStats,
    )> {
        let store = self
            .snapshot_store
            .as_ref()
            .context("Metal T2 disk tier not configured")?;
        store
            .put_snapshot(
                ChunkedSnapshotWrite {
                    manifest_id,
                    namespace: namespace.to_string(),
                    metadata,
                    parts,
                },
                fsync_manifest,
            )
            .context("write snapshot through Metal T2 adapter")
    }

    fn get_disk_snapshot(
        &self,
        location: &ChunkedSnapshotLocation,
        expected_manifest_id: Option<BlockFingerprint>,
    ) -> Result<ChunkedSnapshotRead> {
        let store = self
            .snapshot_store
            .as_ref()
            .context("Metal T2 disk tier not configured")?;
        store
            .get_snapshot(location, expected_manifest_id)
            .context("read snapshot through Metal T2 adapter")
    }

    fn visit_disk_manifests(
        &self,
        visit: impl FnMut(ChunkedSnapshotLocation, &ChunkedSnapshotManifest) -> std::io::Result<()>,
    ) -> Result<()> {
        let Some(store) = self.snapshot_store.as_ref() else {
            return Ok(());
        };
        store
            .visit_manifests(visit)
            .context("scan Metal T2 adapter snapshot manifests")
    }

    fn delete_disk_snapshot(&self, location: &ChunkedSnapshotLocation) -> Result<()> {
        let store = self
            .snapshot_store
            .as_ref()
            .context("Metal T2 disk tier not configured")?;
        store
            .delete_snapshot(location)
            .context("delete snapshot manifest through Metal T2 adapter")?;
        let _ = store
            .collect_orphan_chunks()
            .context("collect orphan snapshot chunks through Metal T2 adapter")?;
        Ok(())
    }
}

impl KvTierAdapter for MetalTierAdapter {
    fn paged_pool_pressure(&self) -> f64 {
        self.paged_pool_pressure
    }

    fn submit_demote(&self, _block_id: BlockId) -> Result<()> {
        // Metal T2 is opt-in. With no disk store configured, demotion is a
        // no-op so the default backend behavior stays unchanged.
        Ok(())
    }

    fn submit_promote(&self, _block_id: BlockId, tier: Tier) -> Result<()> {
        match tier {
            Tier::Gpu | Tier::Disk => Ok(()),
            Tier::HostPinned => anyhow::bail!("Metal skips T1 HostPinned tier"),
            Tier::Remote => anyhow::bail!("Metal remote KV tier is not wired"),
        }
    }
}

fn normalize_paged_pool_pressure(pressure: f64) -> f64 {
    if pressure.is_finite() {
        pressure.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

struct MetalQwen35PrefixRuntime {
    entries: HashMap<Vec<u32>, MetalQwen35CachedPrefix>,
    /// Shared SSD index. Read on the serving thread (lookup/import/reconcile)
    /// and written on the SSD-writer thread (`persist` bookkeeping). The
    /// `Mutex` is the single source of disk-bytes accounting truth.
    disk_index: Arc<Mutex<MetalDiskPrefixIndex>>,
    /// Dedicated async SSD-writer thread. `None` when no disk tier is wired.
    disk_writer: Option<DiskWriteHandle>,
    tier_adapter: MetalTierAdapter,
    model_fingerprint: Vec<u8>,
    disk_fsync_each_block: bool,
    max_cached_bytes: u64,
    cached_bytes: u64,
    next_tick: u64,
    block_size: usize,
}

struct CachedQwen35DecodeBatch {
    req_ids: Vec<RequestId>,
    batch: Qwen35PackedDecodeBatch<'static>,
}

impl MetalLivePrefixRuntime {
    fn new(backend: &'static MetalBackend, _config: &MetalSchedulerConfig) -> Result<Option<Self>> {
        let weights = backend.weights.as_ref().context("weights not loaded")?;
        let max_cached_bytes = backend.kv_memory_max_bytes.unwrap_or(0);
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
                if max_cached_bytes == 0 {
                    info!(
                        "Metal live prefix memory snapshot replay disabled: block_size={}, max_cached_bytes=0; disk prefix tier still runs when configured",
                        METAL_PREFIX_BLOCK_SIZE
                    );
                } else {
                    info!(
                        "Metal live prefix cache enabled for Qwen3.5 snapshot replay: block_size={}, max_cached_bytes={}",
                        METAL_PREFIX_BLOCK_SIZE, max_cached_bytes
                    );
                }
                let (
                    disk_store,
                    model_fingerprint,
                    max_disk_bytes,
                    disk_high_watermark,
                    disk_low_watermark,
                    disk_fsync_each_block,
                ) = if let Some(options) = backend.kv_disk_options.as_ref() {
                    let store = Arc::new(ChunkedSnapshotStore::new(&options.dir));
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
                    max_cached_bytes,
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

    fn publish_completed_session_prefix(&mut self, request: &mut ActiveMetalRequest) -> Result<()> {
        match self {
            MetalLivePrefixRuntime::Qwen35(runtime) => {
                runtime.publish_completed_session_prefix(request)
            }
        }
    }

    fn set_paged_pool_pressure(&mut self, pressure: f64) {
        match self {
            MetalLivePrefixRuntime::Qwen35(runtime) => runtime.set_paged_pool_pressure(pressure),
        }
    }
}

impl MetalQwen35PrefixRuntime {
    fn new(
        max_cached_bytes: u64,
        block_size: usize,
        snapshot_store: Option<Arc<ChunkedSnapshotStore>>,
        model_fingerprint: Vec<u8>,
        max_disk_bytes: Option<u64>,
        disk_high_watermark: f64,
        disk_low_watermark: f64,
        disk_fsync_each_block: bool,
    ) -> Result<Self> {
        let tier_adapter = MetalTierAdapter::new(snapshot_store).with_paged_pool_pressure(0.0);
        let disk_index = Arc::new(Mutex::new(MetalDiskPrefixIndex::new(
            max_disk_bytes,
            disk_high_watermark,
            disk_low_watermark,
        )));
        let mut runtime = Self {
            entries: HashMap::new(),
            disk_index,
            disk_writer: None,
            tier_adapter,
            model_fingerprint,
            disk_fsync_each_block,
            max_cached_bytes,
            cached_bytes: 0,
            next_tick: 1,
            block_size,
        };
        runtime.reconcile_disk_entries()?;
        if runtime.tier_adapter.has_disk_tier() {
            let (entries, bytes) = {
                let guard = runtime.lock_disk_index();
                (guard.entries.len(), guard.disk_bytes)
            };
            info!("Metal Qwen3.5 SSD prefix cache indexed {entries} entries ({bytes} bytes)");
            // Spawn the async SSD-writer only when a disk tier exists. It shares
            // `disk_index` with the serving thread (the writer owns disk writes;
            // the serving thread owns reads + import-side touch/remove).
            runtime.disk_writer = Some(DiskWriteHandle::spawn(
                Arc::clone(&runtime.disk_index),
                runtime.tier_adapter.clone(),
                runtime.disk_fsync_each_block,
            ));
        }
        Ok(runtime)
    }

    fn lock_disk_index(&self) -> std::sync::MutexGuard<'_, MetalDiskPrefixIndex> {
        self.disk_index
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn prepare_request(
        &mut self,
        request: &mut ActiveMetalRequest,
        metrics: &ServerMetrics,
    ) -> Result<()> {
        let prompt_len = request.prompt_tokens.len();
        // M_e.10 trace probe — env-gated diagnostic to localize why
        // session_affinity_hit stays at 0 across multi-turn requests.
        // Set INFER_M_E10_TRACE=1 to log gate decisions + cache state.
        let trace = std::env::var("INFER_M_E10_TRACE").is_ok();
        if trace {
            let prompt_head: Vec<u32> = request.prompt_tokens.iter().take(8).copied().collect();
            log::info!(
                "m_e10_trace prepare_request: session={:?} \
                 prompt_len={} block_size={} dflash_enabled={} \
                 mtp_enabled={} \
                 can_import_snapshot={} entries_len={} \
                 entries_keys_len_sample={:?} prompt_head={:?}",
                &request.session_id,
                prompt_len,
                self.block_size,
                request.request_state.is_dflash_enabled(),
                request.request_state.is_mtp_enabled(),
                request.request_state.can_import_qwen35_prefix_snapshot(),
                self.entries.len(),
                self.entries
                    .keys()
                    .take(5)
                    .map(Vec::len)
                    .collect::<Vec<_>>(),
                prompt_head,
            );
        }
        if prompt_len < self.block_size {
            metrics.record_request_cache(request.session_id.as_ref(), 0, prompt_len, prompt_len);
            return Ok(());
        }
        if request.request_state.is_dflash_enabled() {
            metrics.record_request_cache(request.session_id.as_ref(), 0, prompt_len, prompt_len);
            return Ok(());
        }
        if request.request_state.is_mtp_enabled() {
            metrics.record_request_cache(request.session_id.as_ref(), 0, prompt_len, prompt_len);
            return Ok(());
        }
        if !request.request_state.can_import_qwen35_prefix_snapshot() {
            metrics.record_request_cache(request.session_id.as_ref(), 0, prompt_len, prompt_len);
            return Ok(());
        }

        let memory_key = self.lookup_longest_prefix(&request.prompt_tokens);
        let disk_key = self.lookup_longest_disk_prefix(&request.prompt_tokens);
        if trace {
            log::info!(
                "m_e10_trace lookup: session={:?} memory_match_len={:?} disk_match_len={:?}",
                &request.session_id,
                memory_key.as_ref().map(Vec::len),
                disk_key.as_ref().map(Vec::len),
            );
        }
        let memory_len = memory_key.as_ref().map_or(0, Vec::len);
        let disk_len = disk_key.as_ref().map_or(0, Vec::len);

        // M_e.13 diagnostic — `INFER_M_E13_FORCE_DISK=1` flips priority so disk
        // is tried first even when memory_len >= disk_len. Used to A/B test
        // whether the same-server in-memory short-circuit asymmetry is caused
        // by the in-memory import path itself. Revert if/when the asymmetry
        // is closed.
        let force_disk = std::env::var("INFER_M_E13_FORCE_DISK").is_ok();
        let mut reused_tokens = 0;
        let mut reuse_source = PrefixReuseSource::None;
        if force_disk || disk_len > memory_len {
            if let Some(prefix_key) = disk_key.as_deref() {
                if self.try_import_disk_prefix_or_remove(prefix_key, request) {
                    reused_tokens = prefix_key.len();
                    reuse_source = PrefixReuseSource::Disk;
                }
            }
            if reused_tokens == 0
                && let Some(prefix_key) = memory_key.as_deref()
                && self.try_import_memory_prefix(prefix_key, request)?
            {
                reused_tokens = prefix_key.len();
                reuse_source = PrefixReuseSource::Memory;
            }
        } else {
            if let Some(prefix_key) = memory_key.as_deref()
                && self.try_import_memory_prefix(prefix_key, request)?
            {
                reused_tokens = prefix_key.len();
                reuse_source = PrefixReuseSource::Memory;
            }
            if reused_tokens == 0
                && let Some(prefix_key) = disk_key.as_deref()
                && self.try_import_disk_prefix_or_remove(prefix_key, request)
            {
                reused_tokens = prefix_key.len();
                reuse_source = PrefixReuseSource::Disk;
            }
        }
        metrics.record_request_cache(
            request.session_id.as_ref(),
            reused_tokens,
            prompt_len,
            prompt_len.saturating_sub(reused_tokens),
        );
        request.prefix_reused_tokens = reused_tokens;
        request.prefix_reuse_source = reuse_source;
        Ok(())
    }

    /// Memory-first prefix publish (M_e.14). Runs on the serving thread between
    /// token1 and token2, so it must stay cheap:
    ///
    /// 1. Snapshot the **already-resident** full-length KV+GDR
    ///    (`export_qwen35_live_prefix_snapshot` → `export_drained_prefix_snapshot`,
    ///    no replay) and insert into the in-memory LRU. This is the only publish
    ///    work on the first-response path and is the same cheap clone the
    ///    in-memory tier always used.
    /// 2. If a disk tier is wired, apply the budget gate; on a clear win,
    ///    `encode_chunked_for_disk` (the `eval`/`to_bytes` stays on **this** thread for
    ///    MLX safety) and hand the encoded bytes to the async SSD-writer.
    ///
    /// The old synchronous replay disk-publish — which spun up a fresh replay
    /// `Qwen35StepDriver` and re-prefilled the whole prompt in blocks (512→1.9s,
    /// 2k→7.0s, 8k→30.5s on the serving thread) — is gone. SSD warm-restart
    /// reuse of shorter block-aligned prefixes depended on that replay and is
    /// intentionally dropped; in-memory warm reuse of the full-length snapshot
    /// is retained.
    fn publish_prompt_prefix(&mut self, request: &mut ActiveMetalRequest) -> Result<()> {
        let trace = std::env::var("INFER_M_E10_TRACE").is_ok();
        if !request.request_state.can_import_qwen35_prefix_snapshot() {
            if trace {
                log::info!(
                    "m_e10_trace publish: SKIP can_import=false session={:?}",
                    &request.session_id,
                );
            }
            return Ok(());
        }

        // Cheap full-length in-memory snapshot (drains the C++ session as a
        // side effect; the next decode/prefill tick re-attaches via
        // `begin_session`). `None` when shorter than one block.
        let Some(snapshot) = request
            .request_state
            .export_qwen35_live_prefix_snapshot(self.block_size)
            .context("snapshot live Qwen3.5 prompt prefix")?
        else {
            return Ok(());
        };

        // SSD budget gate + async persist. Only the in-memory snapshot is on the
        // first-response critical path; the disk write is off-thread.
        if self.tier_adapter.has_disk_tier() {
            self.maybe_enqueue_disk_persist(request, &snapshot, trace);
        }

        self.insert_snapshot(snapshot);
        Ok(())
    }

    /// Publish the longest currently materialized session prefix after request
    /// completion. This usually means `prompt + generated[..N-1]`: the final
    /// sampled token has not necessarily been fed back through the model, and
    /// running an extra decode at finish would put SSD persistence on the
    /// user-visible boundary. The next turn can still import this prefix and
    /// prefill the one-token tail plus the new user suffix.
    fn publish_completed_session_prefix(&mut self, request: &mut ActiveMetalRequest) -> Result<()> {
        let trace = std::env::var("INFER_M_E10_TRACE").is_ok();
        if !request.request_state.can_import_qwen35_prefix_snapshot() {
            return Ok(());
        }
        let Some(cache_len) = request.request_state.qwen35_live_cache_len() else {
            return Ok(());
        };
        if cache_len <= request.prompt_tokens.len() {
            return Ok(());
        }
        let token_ids = request
            .materialized_session_tokens(cache_len)
            .context("build Qwen3.5 completed-session snapshot token key")?;
        let Some(snapshot) = request
            .request_state
            .export_qwen35_live_session_snapshot(token_ids, self.block_size)
            .context("snapshot live Qwen3.5 completed session prefix")?
        else {
            return Ok(());
        };

        if self.tier_adapter.has_disk_tier() {
            self.maybe_enqueue_disk_persist(request, &snapshot, trace);
        }

        self.insert_snapshot(snapshot);
        Ok(())
    }

    /// Budget-gate a full-length snapshot for SSD and, on a clear win, encode it
    /// (on the serving thread, for MLX safety) and hand it to the async writer.
    /// Returns without blocking on disk I/O.
    fn maybe_enqueue_disk_persist(
        &self,
        request: &ActiveMetalRequest,
        snapshot: &Qwen35PrefixSnapshot,
        trace: bool,
    ) {
        let tokens = snapshot.token_ids.len();
        // Persist the resident full-length KV+GDR snapshot at exactly
        // `cache_len`. Unlike the old replay path (which produced block-aligned
        // prefix slices), this snapshot is the live drained state, so block
        // alignment is NOT required — the on-disk format only needs
        // `token_ids.len() == cache_len`. A future prompt that *extends* this
        // exact prefix can import it across restarts; the budget gate (below)
        // is the real control over what's worth persisting.
        if tokens < self.block_size {
            return;
        }
        let Some(writer) = self.disk_writer.as_ref() else {
            return;
        };

        // Budget: re-prefilling this prompt from scratch must cost more than
        // reading the snapshot back, with margin. Short prompts (cheap prefill)
        // fail the gate and are dropped — exactly the intended behavior.
        let prefill_cost_us = match request.first_token_at {
            Some(first) => first.duration_since(request.admitted_at).as_micros() as f64,
            None => return,
        };
        let readback_us_per_token = metal_prefix_readback_us_per_token();
        let safety = metal_prefix_persist_safety();
        let min_extension_tokens = metal_prefix_persist_min_extension_tokens();
        let readback_cost_us = readback_us_per_token * tokens as f64 * safety;
        let prompt_len = request.prompt_tokens.len().max(1);
        let saved_prefill_us = prefill_cost_us * (tokens as f64 / prompt_len as f64);
        let extends_cached_prefix = request.prefix_reused_tokens >= self.block_size
            && tokens > request.prefix_reused_tokens;
        let extension_delta_tokens = metal_prefix_extension_delta_tokens(
            request.prompt_tokens.len(),
            tokens,
            request.prefix_reused_tokens,
            self.block_size,
        );
        let worth_persist = should_persist_metal_prefix_snapshot(
            prefill_cost_us,
            request.prompt_tokens.len(),
            tokens,
            request.prefix_reused_tokens,
            self.block_size,
            readback_us_per_token,
            safety,
            min_extension_tokens,
        );
        if trace {
            log::info!(
                "m_e10_trace publish budget: tokens={} prefill_cost_us={:.0} \
                 saved_prefill_us={:.0} readback_cost_us={:.0} \
                 (per_token={} safety={}) reused_tokens={} extends_cached_prefix={} \
                 extension_delta_tokens={:?} min_extension_tokens={} worth_persist={}",
                tokens,
                prefill_cost_us,
                saved_prefill_us,
                readback_cost_us,
                readback_us_per_token,
                safety,
                request.prefix_reused_tokens,
                extends_cached_prefix,
                extension_delta_tokens,
                min_extension_tokens,
                worth_persist,
            );
        }
        if !worth_persist {
            return;
        }
        if request.prefix_reuse_source == PrefixReuseSource::Disk
            && request.prefix_reused_tokens >= self.block_size
            && tokens > request.prefix_reused_tokens
        {
            if trace {
                log::info!(
                    "m_e10_trace publish budget: skip disk-imported extension persist; tokens={} reused_tokens={}",
                    tokens,
                    request.prefix_reused_tokens,
                );
            }
            return;
        }

        // Skip if the index already holds this exact key (e.g. imported from
        // disk this lifetime). Avoids a redundant encode.
        if self.lock_disk_index().contains(&snapshot.token_ids) {
            return;
        }

        let estimated_payload_len =
            match snapshot.estimated_chunked_disk_payload_len(&self.model_fingerprint) {
                Ok(len) => len,
                Err(err) => {
                    warn!("Metal Qwen3.5 SSD prefix size estimate failed: {err:#}");
                    return;
                }
            };
        if !writer.try_reserve_pending(estimated_payload_len, tokens) {
            return;
        }
        // Encode here (serving thread): `encode_chunked_for_disk` → `to_bytes` calls
        // `eval`, which must not run concurrently with decode `async_eval` on a
        // separate thread (no global MLX lock). The resident full-length KV is
        // already materialized, so this eval is the cheap ~persist class, not a
        // replay.
        let (metadata, parts, actual_payload_len) =
            match snapshot.encode_chunked_for_disk(&self.model_fingerprint) {
                Ok(encoded) => encoded,
                Err(err) => {
                    writer.release_pending(estimated_payload_len);
                    warn!("Metal Qwen3.5 SSD prefix encode failed: {err:#}");
                    return;
                }
            };
        match actual_payload_len.cmp(&estimated_payload_len) {
            std::cmp::Ordering::Greater => {
                let delta = actual_payload_len - estimated_payload_len;
                if !writer.try_reserve_pending(delta, tokens) {
                    writer.release_pending(estimated_payload_len);
                    return;
                }
            }
            std::cmp::Ordering::Less => {
                writer.release_pending(estimated_payload_len - actual_payload_len);
            }
            std::cmp::Ordering::Equal => {}
        }
        let fingerprint = self.fingerprint_for_tokens(&snapshot.token_ids);
        writer.submit_reserved(DiskWriteJob {
            token_ids: snapshot.token_ids.clone(),
            manifest_id: fingerprint,
            metadata,
            parts,
            estimated_payload_len: actual_payload_len,
        });
    }

    fn set_paged_pool_pressure(&mut self, pressure: f64) {
        self.tier_adapter.set_paged_pool_pressure(pressure);
    }

    fn lookup_longest_prefix(&self, prompt_tokens: &[u32]) -> Option<Vec<u32>> {
        // Strict extension only; see `MetalDiskPrefixIndex::lookup_longest_prefix`.
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
        self.lock_disk_index()
            .lookup_longest_prefix(prompt_tokens, self.block_size)
    }

    fn try_import_memory_prefix(
        &mut self,
        prefix_key: &[u32],
        request: &mut ActiveMetalRequest,
    ) -> Result<bool> {
        let trace = std::env::var("INFER_M_E10_TRACE").is_ok();
        let trace13 = std::env::var("INFER_M_E13_TRACE").is_ok();
        let imported = {
            let Some(snapshot) = self.entries.get(prefix_key).map(|entry| &entry.snapshot) else {
                if trace {
                    log::info!(
                        "m_e10_trace try_import: SKIP entries.get returned None for key.len={} session={:?}",
                        prefix_key.len(),
                        &request.session_id,
                    );
                }
                return Ok(false);
            };
            if trace {
                log::info!(
                    "m_e10_trace try_import: snapshot found key.len={} snapshot.cache_len={} session={:?}",
                    prefix_key.len(),
                    snapshot.cache_len,
                    &request.session_id,
                );
            }
            let t_import_start = std::time::Instant::now();
            let result = request
                .request_state
                .import_qwen35_prefix_snapshot(snapshot, prefix_key.len());
            let t_import_us = t_import_start.elapsed().as_micros();
            if trace13 {
                log::info!(
                    "m_e13_trace try_import_memory_prefix: tokens={} import_us={} ok={}",
                    prefix_key.len(),
                    t_import_us,
                    result.is_ok(),
                );
            }
            match &result {
                Ok(b) if trace => log::info!(
                    "m_e10_trace import_qwen35_prefix_snapshot returned Ok({})",
                    b
                ),
                Err(e) if trace => {
                    log::info!("m_e10_trace import_qwen35_prefix_snapshot returned Err: {e:#}");
                }
                _ => {}
            }
            result.context("import matched Qwen3.5 prefix snapshot into request state")?
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
        // The SSD persist path keeps its own alignment guard
        // (`maybe_enqueue_disk_persist`) because the on-disk format assumes
        // block-aligned slices.
        let tick = self.bump_tick();
        if let Some(existing) = self.entries.get_mut(&snapshot.token_ids) {
            existing.last_used_tick = tick;
            return;
        }
        let footprint = snapshot_resident_bytes(&snapshot);
        if footprint > self.max_cached_bytes {
            return;
        }

        self.ensure_capacity_for(footprint);
        let key = snapshot.token_ids.clone();
        self.cached_bytes += footprint;
        self.entries.insert(
            key,
            MetalQwen35CachedPrefix {
                snapshot,
                last_used_tick: tick,
            },
        );
    }

    fn try_import_disk_prefix(
        &mut self,
        prefix_key: &[u32],
        request: &mut ActiveMetalRequest,
    ) -> Result<bool> {
        let trace = std::env::var("INFER_M_E13_TRACE").is_ok();
        if !self.tier_adapter.has_disk_tier() {
            return Ok(false);
        }
        let Some(location) = self.lock_disk_index().location_for(prefix_key) else {
            return Ok(false);
        };
        let expected = self.fingerprint_for_tokens(prefix_key);
        let t_read_start = std::time::Instant::now();
        let chunked = self
            .tier_adapter
            .get_disk_snapshot(&location, Some(expected))
            .context("read Qwen3.5 prefix snapshot from chunked store")?;
        let t_read_us = t_read_start.elapsed().as_micros();
        let payload_bytes = location.payload_len;

        let t_decode_start = std::time::Instant::now();
        let snapshot =
            Qwen35PrefixSnapshot::decode_chunked_from_disk(chunked, &self.model_fingerprint)
                .context("decode Qwen3.5 prefix snapshot from chunked store")?;
        let t_decode_us = t_decode_start.elapsed().as_micros();
        ensure!(
            snapshot.token_ids == prefix_key,
            "Qwen3.5 SSD prefix snapshot token key mismatch"
        );

        let t_import_start = std::time::Instant::now();
        let imported = request
            .request_state
            .import_qwen35_prefix_snapshot(&snapshot, prefix_key.len())
            .context("import matched Qwen3.5 SSD prefix snapshot into request state")?;
        let t_import_us = t_import_start.elapsed().as_micros();
        if trace {
            log::info!(
                "m_e13_trace try_import_disk_prefix: tokens={} payload_bytes={} read_us={} decode_us={} import_us={} imported={}",
                prefix_key.len(),
                payload_bytes,
                t_read_us,
                t_decode_us,
                t_import_us,
                imported,
            );
        }
        if imported {
            self.lock_disk_index().touch(prefix_key);
            self.insert_snapshot(snapshot);
        }
        Ok(imported)
    }

    /// Index existing on-disk snapshots at startup. Single-threaded — runs in
    /// `new()` before the SSD-writer thread is spawned — so it populates the
    /// shared index directly.
    fn reconcile_disk_entries(&mut self) -> Result<()> {
        if !self.tier_adapter.has_disk_tier() {
            return Ok(());
        }
        let adapter = self.tier_adapter.clone();
        let model_fingerprint = &self.model_fingerprint;
        let block_size = self.block_size;
        let mut accepted: Vec<(Vec<u32>, ChunkedSnapshotLocation)> = Vec::new();
        adapter
            .visit_disk_manifests(|location, manifest| {
                if manifest.namespace != METAL_QWEN35_CHUNKED_SNAPSHOT_NAMESPACE {
                    return Ok(());
                }
                let token_ids = match Qwen35PrefixSnapshot::peek_chunked_disk_token_ids(
                    &manifest.metadata,
                    model_fingerprint,
                ) {
                    Ok(token_ids) => token_ids,
                    Err(err) => {
                        log::debug!(
                            "Metal Qwen3.5 SSD prefix cache ignored {}: {err:#}",
                            location.path.display()
                        );
                        if Qwen35PrefixSnapshot::looks_like_chunked_metadata(&manifest.metadata) {
                            delete_rejected_qwen35_disk_snapshot(&adapter, &location);
                        }
                        return Ok(());
                    }
                };
                // Persisted snapshots are full-length live KV+GDR at exactly
                // `cache_len`; they need not be block-aligned (the old
                // replay path's alignment requirement is gone). Only reject
                // sub-block fragments.
                if token_ids.len() < block_size {
                    delete_rejected_qwen35_disk_snapshot(&adapter, &location);
                    return Ok(());
                }
                let expected = fingerprint_for_tokens(model_fingerprint, &token_ids);
                if location.manifest_id != expected {
                    log::debug!(
                        "Metal Qwen3.5 SSD prefix cache ignored {}: fingerprint/token mismatch",
                        location.path.display()
                    );
                    delete_rejected_qwen35_disk_snapshot(&adapter, &location);
                    return Ok(());
                }
                accepted.push((token_ids, location));
                Ok(())
            })
            .context("scan Metal Qwen3.5 SSD prefix cache")?;

        let evicted = {
            let mut index = self.lock_disk_index();
            for (token_ids, location) in accepted {
                index.insert(token_ids, location);
            }
            // Trim back under the high watermark; capture evictions to delete
            // their files outside the lock.
            index.reserve_capacity(0).evicted
        };
        for location in &evicted {
            if let Err(err) = self.tier_adapter.delete_disk_snapshot(location) {
                warn!(
                    "Metal Qwen3.5 SSD prefix cache failed to evict {}: {err:#}",
                    location.path.display()
                );
            }
        }
        Ok(())
    }

    fn ensure_capacity_for(&mut self, needed_bytes: u64) {
        while self.cached_bytes.saturating_add(needed_bytes) > self.max_cached_bytes {
            let Some((lru_key, lru_footprint)) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used_tick)
                .map(|(tokens, entry)| (tokens.clone(), snapshot_resident_bytes(&entry.snapshot)))
            else {
                break;
            };
            self.entries.remove(&lru_key);
            self.cached_bytes = self.cached_bytes.saturating_sub(lru_footprint);
        }
    }

    fn touch(&mut self, key: &[u32]) {
        let tick = self.bump_tick();
        if let Some(entry) = self.entries.get_mut(key) {
            entry.last_used_tick = tick;
        }
    }

    /// Drop a stale/corrupt SSD entry: remove it from the shared index, then
    /// delete the file outside the lock.
    fn remove_disk_entry(&mut self, key: &[u32]) {
        let location = self.lock_disk_index().remove(key);
        if let Some(location) = location
            && self.tier_adapter.has_disk_tier()
            && let Err(err) = self.tier_adapter.delete_disk_snapshot(&location)
        {
            warn!(
                "Metal Qwen3.5 SSD prefix cache failed to delete {}: {err:#}",
                location.path.display()
            );
        }
    }

    fn fingerprint_for_tokens(&self, token_ids: &[u32]) -> BlockFingerprint {
        fingerprint_for_tokens(&self.model_fingerprint, token_ids)
    }

    fn bump_tick(&mut self) -> u64 {
        let tick = self.next_tick;
        self.next_tick = self.next_tick.saturating_add(1);
        tick
    }

    /// Test-only synchronous persist: encode + write + index insert on the
    /// calling thread, mirroring `disk_writer_loop` without the channel hop so
    /// tests can assert on the index immediately. Production persists go through
    /// the async writer (`maybe_enqueue_disk_persist` → `DiskWriteHandle`).
    #[cfg(test)]
    fn persist_snapshot_blocking(&mut self, snapshot: &Qwen35PrefixSnapshot) -> Result<()> {
        if !self.tier_adapter.has_disk_tier() {
            return Ok(());
        }
        let token_count = snapshot.token_ids.len();
        if token_count < self.block_size {
            return Ok(());
        }
        let key = snapshot.token_ids.clone();
        let estimated_payload_len = snapshot
            .estimated_chunked_disk_payload_len(&self.model_fingerprint)
            .context("estimate Qwen3.5 prefix snapshot size for SSD")?;
        let reserve = self
            .lock_disk_index()
            .reserve_capacity(estimated_payload_len);
        for location in &reserve.evicted {
            let _ = self.tier_adapter.delete_disk_snapshot(location);
        }
        if !reserve.fits {
            return Ok(());
        }
        let (metadata, parts, actual_payload_len) = snapshot
            .encode_chunked_for_disk(&self.model_fingerprint)
            .context("encode Qwen3.5 prefix snapshot for SSD")?;
        let fingerprint = self.fingerprint_for_tokens(&key);
        let location = self
            .tier_adapter
            .put_disk_snapshot(
                fingerprint,
                METAL_QWEN35_CHUNKED_SNAPSHOT_NAMESPACE,
                metadata,
                parts,
                self.disk_fsync_each_block,
            )
            .context("write Qwen3.5 prefix snapshot to chunked store")?;
        let (location, _stats) = location;
        debug_assert_eq!(location.payload_len, actual_payload_len);
        self.lock_disk_index().insert(key, location);
        Ok(())
    }
}

fn fingerprint_for_tokens(model_fingerprint: &[u8], token_ids: &[u32]) -> BlockFingerprint {
    BlockFingerprint::compute(
        KvContentContext {
            model_fingerprint,
            kv_format_tag: METAL_QWEN35_SNAPSHOT_KV_FORMAT_TAG,
            parent: None,
        },
        token_ids,
    )
}

fn watermark_bytes(max_bytes: u64, watermark: f64) -> u64 {
    ((max_bytes as f64) * watermark).ceil() as u64
}

fn delete_rejected_qwen35_disk_snapshot(
    adapter: &MetalTierAdapter,
    location: &ChunkedSnapshotLocation,
) {
    if let Err(err) = adapter.delete_disk_snapshot(location) {
        warn!(
            "Metal Qwen3.5 SSD prefix cache failed to delete rejected snapshot {}: {err}",
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
        MetalSchedulerConfig::default(),
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

    fn tokenizer_clone(&self) -> Option<Tokenizer> {
        SchedulerHandle::tokenizer_clone(&self.inner)
    }

    /// Forward the inner `SchedulerHandle`'s server-metrics handle so
    /// `InferenceEngine::telemetry()` can project the unified
    /// `EngineTelemetry` snapshot for the Metal backend. Without this
    /// the trait default returned `None` and Metal silently lost its
    /// engine telemetry projection. (M1 unification)
    fn server_metrics(&self) -> Option<&crate::metrics::ServerMetrics> {
        SchedulerHandle::server_metrics(&self.inner)
    }
}

pub fn spawn_metal_scheduler_handle_from_path_with_options_and_metrics(
    model_path: &str,
    options: MetalBackendOptions,
    max_waiting: usize,
    metrics: ServerMetrics,
    scheduler_config: MetalSchedulerConfig,
) -> Result<MetalSchedulerHandle> {
    // DFlash is now supported: Qwen3StepDriver's token-buffer pattern runs
    // speculative blocks inside decode_token, transparent to the scheduler.
    let mut backend = MetalBackend::with_options(options);
    backend.load(Path::new(model_path))?;
    if let Some(config) = backend.config.as_ref() {
        metrics.set_model_arch(config.arch_summary());
    }

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
        SchedulerHandle::with_shared_waiting_count(tx, &model_id, max_waiting, waiting_count)
            .with_server_metrics(metrics.clone());
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
                scheduler_config,
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
            &mut prefix_runtime,
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
                refresh_runtime_metrics(
                    metrics,
                    handle,
                    &scheduler,
                    &pending,
                    &active,
                    &mut prefix_runtime,
                );
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
            &mut prefix_runtime,
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

// M_e.9 precondition counter — env-gated, accumulates mixed-batch
// dispatch outcomes across the run and emits a periodic summary so
// the bench can decide whether the M_e.9 generalize-to-Qwen3.5
// effort is on the hot path. Counters are atomic so the periodic
// dump is lock-free.
fn m_e9_precondition_record(succeeded: bool) {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};
    static FLAG: OnceLock<bool> = OnceLock::new();
    static MIXED_TICK_TOTAL: AtomicU64 = AtomicU64::new(0);
    static MIXED_TICK_FUSED: AtomicU64 = AtomicU64::new(0);
    let enabled = *FLAG.get_or_init(|| std::env::var("INFER_M_E9_PRECONDITION").is_ok());
    if !enabled {
        return;
    }
    let total = MIXED_TICK_TOTAL.fetch_add(1, Ordering::Relaxed) + 1;
    if succeeded {
        MIXED_TICK_FUSED.fetch_add(1, Ordering::Relaxed);
    }
    if total.is_multiple_of(50) {
        let fused = MIXED_TICK_FUSED.load(Ordering::Relaxed);
        let fallback = total - fused;
        let fallback_pct = (fallback as f64 / total as f64) * 100.0;
        log::info!(
            "m_e9_precondition: mixed_dispatch_ticks={} fused={} fallback={} fallback_pct={:.1}% (≥30% means M_e.9 is on hot path)",
            total,
            fused,
            fallback,
            fallback_pct
        );
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
            // M_e.9 precondition counters — measure how often the
            // dispatcher hits the (decode + prefill) case AND how
            // often it falls back to two sequential async_evals
            // because the model isn't Qwen3 (i.e. Qwen3.5/3.6 today).
            // Plan threshold: if Qwen3.5 fallback >= 30% of ticks
            // where (decode, prefill) coexist, M_e.9 is on the hot
            // path; <30% means deprioritize. Env-gated to keep
            // production output clean; turn on with
            // INFER_M_E9_PRECONDITION=1 during the bench tick.
            //
            // The is_qwen3() check at execute_mixed_batch:1685, :1693
            // is what actually drives the Qwen3.5 fallback rate; we
            // could short-circuit here with the same check, but
            // attributing the fallback only after guard_mixed_batch
            // returns false keeps the metric semantically correct
            // (any fallback reason — not just non-Qwen3 — increments).
            let succeeded = guard_mixed_batch(
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
            );
            m_e9_precondition_record(succeeded);
            if !succeeded {
                guard_decode_batch(
                    batch.req_ids,
                    metrics,
                    prefix_runtime,
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
                prefix_runtime,
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
    prefix_runtime: &mut Option<MetalLivePrefixRuntime>,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
    qwen35_decode_batch_cache: &mut Option<CachedQwen35DecodeBatch>,
) {
    let panic_req_ids = req_ids.clone();
    let result = catch_unwind(AssertUnwindSafe(|| {
        execute_decode_batch(
            req_ids,
            metrics,
            prefix_runtime,
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
    if !MixedBatchRequestEligibility::from_request(prefill_snapshot).is_supported() {
        return false;
    }
    if !decode_req_ids.iter().all(|req_id| {
        active.get(req_id).is_some_and(|request| {
            MixedBatchRequestEligibility::from_request(request).is_supported()
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
        if request.cancel_requested() {
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
    if prefill_request.cancel_requested() {
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
        finish_or_requeue_decoded_request(
            req_id,
            request,
            metrics,
            prefix_runtime,
            scheduler,
            active,
        );
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
        finalize_detached_request(
            prefill_req_id,
            prefill_request,
            metrics,
            prefix_runtime,
            scheduler,
        );
    } else {
        active.insert(prefill_req_id, prefill_request);
    }

    true
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MixedBatchRequestEligibility {
    cancel_requested: bool,
    is_qwen3: bool,
    is_dflash_enabled: bool,
}

impl MixedBatchRequestEligibility {
    fn from_request(request: &ActiveMetalRequest) -> Self {
        Self {
            cancel_requested: request.cancel_requested(),
            is_qwen3: request.request_state.is_qwen3(),
            is_dflash_enabled: request.request_state.is_dflash_enabled(),
        }
    }

    fn is_supported(self) -> bool {
        !self.cancel_requested && self.is_qwen3 && !self.is_dflash_enabled
    }
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
    prefix_runtime: &mut Option<MetalLivePrefixRuntime>,
    last: &mut Option<Instant>,
    interval: Duration,
) {
    let now = Instant::now();
    if let Some(prev) = *last {
        if now.duration_since(prev) < interval {
            return;
        }
    }
    refresh_runtime_metrics(metrics, handle, scheduler, pending, active, prefix_runtime);
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

    if pending_request.cancel_requested() {
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

        if request.cancel_requested() {
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
                finalize_request(req_id, metrics, prefix_runtime, scheduler, active);
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
    prefix_runtime: &mut Option<MetalLivePrefixRuntime>,
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
        if request.cancel_requested() {
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
    // Partition into speculative rows and plain rows. Dispatch:
    //   - plain_rows (≥1): existing `execute_qwen35_packed_decode_batch`.
    //   - dflash_rows (≥2): new `execute_qwen35_dflash_packed_batch`.
    //   - dflash_rows (==1): fall through to the existing per-row
    //     `execute_decode_single` path (batched-stack overhead not worth it).
    //   - mtp_rows: scalar only for now; the draft has no persistent KV and
    //     carries a per-row recurrent seed hidden.
    let scheduled_open_len = open.len();
    let (spec_requests, non_speculative): (Vec<_>, Vec<_>) =
        open.into_iter().partition(|(_, request)| {
            request.request_state.is_dflash_enabled() || request.request_state.is_mtp_enabled()
        });
    let (dflash_requests, mtp_requests): (Vec<_>, Vec<_>) = spec_requests
        .into_iter()
        .partition(|(_, request)| request.request_state.is_dflash_enabled());

    if dflash_requests.len() >= 2 {
        execute_qwen35_dflash_packed_batch(
            dflash_requests,
            metrics,
            prefix_runtime,
            scheduler,
            active,
        );
    } else {
        if scheduled_open_len >= 2 && !dflash_requests.is_empty() {
            metrics.record_metal_decode_batch_fallback(dflash_requests.len());
        }
        for (req_id, request) in dflash_requests {
            execute_decode_single(req_id, request, metrics, prefix_runtime, scheduler, active);
        }
    }
    for (req_id, request) in mtp_requests {
        metrics.record_metal_mtp_scalar_row();
        execute_decode_single(req_id, request, metrics, prefix_runtime, scheduler, active);
    }
    let mut open = non_speculative;

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
        // M_e.12 — capture row-ordered req_ids before consuming `open` so we
        // can detect mid-batch finishers and compact the packed-decode cache
        // in the SAME tick (instead of next-tick set-diff via
        // `invalidate_qwen35_decode_batch_cache`). Order matches both
        // `open` and the cache's `req_ids` (enforced at the cache-equality
        // check above), so position == cache row index.
        let original_req_ids: Vec<RequestId> = open.iter().map(|(req_id, _)| *req_id).collect();
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
            finish_or_requeue_decoded_request(
                req_id,
                request,
                metrics,
                prefix_runtime,
                scheduler,
                active,
            );
        }

        // Survivors are exactly the rows whose req_id is back in `active`
        // after `finish_or_requeue_decoded_request`. Finished/cancelled rows
        // got finalized or cancelled and are no longer keys. If any row is
        // missing, drop it from the cache before returning so the next tick
        // doesn't carry the dead row's KV slot or its `left_padding`.
        if let Some(cached) = qwen35_decode_batch_cache.as_mut() {
            let mut keep_row_indices: Vec<usize> = Vec::with_capacity(original_req_ids.len());
            let mut keep_req_ids: Vec<RequestId> = Vec::with_capacity(original_req_ids.len());
            for (row_idx, req_id) in original_req_ids.iter().enumerate() {
                if active.contains_key(req_id) {
                    keep_row_indices.push(row_idx);
                    keep_req_ids.push(*req_id);
                }
            }
            if keep_row_indices.len() < original_req_ids.len() {
                if keep_row_indices.is_empty() {
                    *qwen35_decode_batch_cache = None;
                } else if let Err(err) = cached.batch.retain_rows(&keep_row_indices, true) {
                    error!(
                        "Metal packed Qwen3.5 mid-batch compaction failed: {err:#}; invalidating cache"
                    );
                    *qwen35_decode_batch_cache = None;
                } else {
                    cached.req_ids = keep_req_ids;
                }
            }
        }
        return;
    }

    if scheduled_open_len >= 1 && !open.is_empty() {
        metrics.record_metal_decode_batch_fallback(open.len());
    }
    for (req_id, request) in open {
        execute_decode_single(req_id, request, metrics, prefix_runtime, scheduler, active);
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
    prefix_runtime: &mut Option<MetalLivePrefixRuntime>,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    if rows.len() < 2 {
        // Partition guard already filters on ≥2; defensive fallthrough only.
        if !rows.is_empty() {
            metrics.record_metal_decode_batch_fallback(rows.len());
        }
        for (req_id, request) in rows {
            execute_decode_single(req_id, request, metrics, prefix_runtime, scheduler, active);
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
                    execute_decode_single(
                        req_id,
                        request,
                        metrics,
                        prefix_runtime,
                        scheduler,
                        active,
                    );
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
            finish_or_requeue_decoded_request(
                req_id,
                request,
                metrics,
                prefix_runtime,
                scheduler,
                active,
            );
        } else {
            execute_decode_single(req_id, request, metrics, prefix_runtime, scheduler, active);
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
                cached.batch.retain_rows(&retained_rows, true)?;
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
        if let Err(err) = cached.batch.retain_rows(&row_indices, true) {
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
    prefix_runtime: &mut Option<MetalLivePrefixRuntime>,
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

    let outcome = if request.cancel_requested() {
        Outcome::ClientDropped
    } else {
        metrics.record_metal_decode_scalar_row();
        match request.decode_step() {
            Ok(_sampled_token) => Outcome::Progress {
                runtime_finished: request.phase() == RuntimePhase::Finished,
                stop_hit: request.stop_hit(),
            },
            Err(err) => {
                if request.cancel_requested() {
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
                finalize_detached_request(req_id, request, metrics, prefix_runtime, scheduler);
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
    prefix_runtime: &mut Option<MetalLivePrefixRuntime>,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    let runtime_finished = request.phase() == RuntimePhase::Finished;
    let stop_hit = request.stop_hit();
    if runtime_finished || stop_hit {
        finalize_detached_request(req_id, request, metrics, prefix_runtime, scheduler);
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

fn publish_completed_session_prefix(
    req_id: RequestId,
    prefix_runtime: &mut Option<MetalLivePrefixRuntime>,
    request: &mut ActiveMetalRequest,
) {
    let Some(prefix_runtime) = prefix_runtime.as_mut() else {
        return;
    };
    if let Err(err) = prefix_runtime.publish_completed_session_prefix(request) {
        warn!(
            "Metal completed-session prefix publish failed for {:?}: {err:#}",
            req_id
        );
    }
}

fn finalize_detached_request(
    req_id: RequestId,
    mut request: ActiveMetalRequest,
    metrics: &ServerMetrics,
    prefix_runtime: &mut Option<MetalLivePrefixRuntime>,
    scheduler: &mut MetalScheduler,
) {
    scheduler.finish_request(req_id, Some(InferenceMode::Decode));
    record_request_completed(metrics, &request);
    if let Err(err) = request.send_final_delta() {
        warn!("Metal request final delta failed for {:?}: {err:#}", req_id);
    }
    publish_completed_session_prefix(req_id, prefix_runtime, &mut request);
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
        .filter_map(|(req_id, request)| request.cancel_requested().then_some(*req_id))
        .collect();
    for req_id in pending_closed {
        handle.consume_one();
        scheduler.finish_request(req_id, None);
        pending.remove(&req_id);
    }

    let closed: Vec<_> = active
        .iter()
        .filter_map(|(req_id, request)| request.cancel_requested().then_some(*req_id))
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
    prefix_runtime: &mut Option<MetalLivePrefixRuntime>,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    scheduler.finish_request(req_id, Some(InferenceMode::Decode));
    let Some(mut request) = active.remove(&req_id) else {
        return;
    };
    record_request_completed(metrics, &request);
    if let Err(err) = request.send_final_delta()
        && !request.delta_closed()
    {
        warn!("Metal request final delta failed for {:?}: {err:#}", req_id);
    }
    publish_completed_session_prefix(req_id, prefix_runtime, &mut request);
    if let Err(err) = request.cancel() {
        warn!("Metal request cleanup failed for {:?}: {err:#}", req_id);
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
    if let Some((blocks, accepted, drafted)) = request.request_state.mtp_block_stats() {
        for i in 0..blocks {
            metrics.record_metal_mtp_block(accepted.get(i).copied().unwrap_or(0), drafted);
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
    prefix_runtime: &mut Option<MetalLivePrefixRuntime>,
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
    let pressure = if kv_total == 0 {
        0.0
    } else {
        kv_used as f64 / kv_total as f64
    };
    if let Some(prefix_runtime) = prefix_runtime.as_mut() {
        prefix_runtime.set_paged_pool_pressure(pressure);
    }
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
            error: None,
        })
        .map_err(|_| MetalStreamError::ConsumerDropped.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::metal::mlx::MlxArray;
    use crate::request_handle::RequestHandle;
    use crate::test_support::metal_test_guard;
    use tempfile::tempdir;
    use tokenizers::{
        Tokenizer as HfTokenizer, models::wordlevel::WordLevel,
        pre_tokenizers::whitespace::Whitespace,
    };

    fn test_word_tokenizer() -> (tempfile::TempDir, Tokenizer) {
        let dir = tempdir().expect("tempdir");
        let vocab = [
            ("<unk>".to_string(), 0u32),
            ("hello".to_string(), 1u32),
            ("world".to_string(), 2u32),
        ]
        .into_iter()
        .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".to_string())
            .build()
            .expect("wordlevel");
        let mut hf_tokenizer = HfTokenizer::new(model);
        hf_tokenizer.with_pre_tokenizer(Some(Whitespace));
        hf_tokenizer
            .save(dir.path().join("tokenizer.json"), false)
            .expect("save tokenizer");
        let tokenizer =
            Tokenizer::from_file(dir.path().to_str().expect("utf8 path")).expect("load tokenizer");
        (dir, tokenizer)
    }

    #[test]
    fn metal_handle_forwards_inner_tokenizer_clone() {
        let (_dir, tokenizer) = test_word_tokenizer();
        let (tx, _rx) = mpsc::unbounded_channel();
        let inner = SchedulerHandle::from_parts(tx, "metal-tokenizer-test")
            .with_tokenizer(tokenizer.clone());
        let handle = MetalSchedulerHandle {
            inner,
            dflash_status: None,
        };

        let forwarded = handle
            .tokenizer_clone()
            .expect("metal handle should forward tokenizer");
        assert_eq!(forwarded.encode("hello world").expect("encode"), vec![1, 2]);
    }

    #[test]
    fn pending_metal_request_uses_cached_prompt_tokens() {
        let (_dir, tokenizer) = test_word_tokenizer();
        let (delta_tx, _delta_rx) = mpsc::unbounded_channel();
        let incoming = IncomingRequest {
            prompt: "hello world".into(),
            prompt_tokens: Some(vec![42, 43]),
            max_tokens: 8,
            sampling: SamplingParams::default(),
            stop: None,
            speculative: None,
            priority: RequestPriority::High,
            session_id: None,
            ingress_numa_node: None,
            delta_tx,
            trace_context: None,
            distributed: None,
            cancel: None,
        };

        let (pending, priority) =
            PendingMetalRequest::from_incoming(&tokenizer, incoming).expect("pending request");
        assert_eq!(pending.prompt_tokens, vec![42, 43]);
        assert_eq!(priority, MetalRequestPriority::High);
    }

    #[test]
    fn disk_write_handle_bounds_pending_payload_bytes() {
        let (tx, _rx) = std_mpsc::channel();
        let handle = DiskWriteHandle {
            tx: Some(tx),
            join: None,
            pending_bytes: Arc::new(AtomicU64::new(0)),
            max_pending_bytes: 10,
        };

        assert!(handle.try_reserve_pending(6, 16));
        assert_eq!(handle.pending_bytes.load(Ordering::Acquire), 6);
        assert!(!handle.try_reserve_pending(5, 16));
        assert_eq!(handle.pending_bytes.load(Ordering::Acquire), 6);
        assert!(!handle.try_reserve_pending(11, 16));
        assert_eq!(handle.pending_bytes.load(Ordering::Acquire), 6);

        handle.release_pending(6);
        assert_eq!(handle.pending_bytes.load(Ordering::Acquire), 0);
    }

    #[test]
    fn materialized_session_tokens_use_only_committed_generated_prefix() {
        let tokens = materialized_session_tokens_for_snapshot(&[10, 11], &[20, 21, 22], 4)
            .expect("materialized key");
        assert_eq!(tokens, vec![10, 11, 20, 21]);

        let err = materialized_session_tokens_for_snapshot(&[10, 11], &[20], 4)
            .expect_err("missing generated token should fail");
        assert!(
            err.to_string().contains("only 1 recorded"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn metal_prefix_persist_gate_skips_small_session_extensions() {
        assert!(!should_persist_metal_prefix_snapshot(
            1.0,
            4096,
            4100,
            4096,
            METAL_PREFIX_BLOCK_SIZE,
            48.0,
            2.0,
            64,
        ));
        assert!(should_persist_metal_prefix_snapshot(
            1.0,
            4096,
            4160,
            4096,
            METAL_PREFIX_BLOCK_SIZE,
            48.0,
            2.0,
            64,
        ));
        assert!(!should_persist_metal_prefix_snapshot(
            100.0,
            128,
            128,
            0,
            METAL_PREFIX_BLOCK_SIZE,
            48.0,
            2.0,
            64,
        ));
        assert!(!should_persist_metal_prefix_snapshot(
            15_000_000.0,
            64,
            71,
            0,
            METAL_PREFIX_BLOCK_SIZE,
            48.0,
            2.0,
            64,
        ));
        assert!(should_persist_metal_prefix_snapshot(
            15_000_000.0,
            12_288,
            12_544,
            0,
            METAL_PREFIX_BLOCK_SIZE,
            48.0,
            2.0,
            64,
        ));
    }

    #[test]
    fn metal_prefix_extension_delta_tracks_reuse_then_prompt_tail() {
        assert_eq!(
            metal_prefix_extension_delta_tokens(4096, 4100, 4096, METAL_PREFIX_BLOCK_SIZE),
            Some(4)
        );
        assert_eq!(
            metal_prefix_extension_delta_tokens(64, 71, 0, METAL_PREFIX_BLOCK_SIZE),
            Some(7)
        );
        assert_eq!(
            metal_prefix_extension_delta_tokens(64, 64, 0, METAL_PREFIX_BLOCK_SIZE),
            None
        );
    }

    #[test]
    fn metal_disk_prefix_index_matches_only_strict_extensions() {
        let mut index = MetalDiskPrefixIndex::new(None, 0.90, 0.75);
        index.insert(
            vec![10, 11, 12],
            ChunkedSnapshotLocation {
                path: PathBuf::from("short"),
                payload_len: 30,
                manifest_id: BlockFingerprint([1; 16]),
            },
        );
        index.insert(
            vec![10, 11, 12, 13],
            ChunkedSnapshotLocation {
                path: PathBuf::from("long"),
                payload_len: 40,
                manifest_id: BlockFingerprint([2; 16]),
            },
        );

        assert_eq!(
            index.lookup_longest_prefix(&[10, 11, 12, 13, 14], 2),
            Some(vec![10, 11, 12, 13])
        );
        assert_eq!(
            index.lookup_longest_prefix(&[10, 11, 12, 13], 2),
            Some(vec![10, 11, 12])
        );
        assert_eq!(index.lookup_longest_prefix(&[10, 11, 12], 2), None);
    }

    #[test]
    fn disk_prefix_index_evicts_under_bounded_budget() {
        let mut index = MetalDiskPrefixIndex::new(Some(100), 0.90, 0.75);
        index.insert(
            vec![1],
            ChunkedSnapshotLocation {
                path: PathBuf::from("one"),
                payload_len: 30,
                manifest_id: BlockFingerprint([1; 16]),
            },
        );
        index.insert(
            vec![2],
            ChunkedSnapshotLocation {
                path: PathBuf::from("two"),
                payload_len: 30,
                manifest_id: BlockFingerprint([2; 16]),
            },
        );
        index.insert(
            vec![3],
            ChunkedSnapshotLocation {
                path: PathBuf::from("three"),
                payload_len: 20,
                manifest_id: BlockFingerprint([3; 16]),
            },
        );
        index.touch(&[2]);

        let decision = index.reserve_capacity(20);
        assert!(decision.fits);
        assert_eq!(index.disk_bytes, 50);
        assert_eq!(decision.evicted.len(), 1);
        assert_eq!(decision.evicted[0].path, PathBuf::from("one"));
        assert!(index.contains(&[2]));
        assert!(index.contains(&[3]));
    }

    fn qwen35_test_snapshot(tokens: &[u32], bytes: usize) -> Qwen35PrefixSnapshot {
        assert!(bytes.is_multiple_of(std::mem::size_of::<i32>()));
        let values = vec![0_i32; bytes / std::mem::size_of::<i32>()];
        Qwen35PrefixSnapshot {
            token_ids: tokens.to_vec(),
            kv_flat: vec![MlxArray::from_slice_i32(&values, &[values.len() as i32])],
            gdr_flat: Vec::new(),
            cache_len: tokens.len() as i32,
            kv_capacity: tokens.len() as i32,
        }
    }

    #[test]
    fn memory_prefix_runtime_evicts_under_byte_budget() {
        let _guard = metal_test_guard();
        let mut runtime =
            MetalQwen35PrefixRuntime::new(32, 2, None, Vec::new(), None, 0.90, 0.75, false)
                .expect("runtime");

        runtime.insert_snapshot(qwen35_test_snapshot(&[1, 2], 16));
        runtime.insert_snapshot(qwen35_test_snapshot(&[3, 4], 16));
        assert_eq!(runtime.cached_bytes, 32);

        runtime.insert_snapshot(qwen35_test_snapshot(&[5, 6], 16));
        assert_eq!(runtime.cached_bytes, 32);
        assert!(!runtime.entries.contains_key(&[1, 2][..]));
        assert!(runtime.entries.contains_key(&[3, 4][..]));
        assert!(runtime.entries.contains_key(&[5, 6][..]));
    }

    #[test]
    fn memory_prefix_runtime_drops_snapshot_larger_than_budget() {
        let _guard = metal_test_guard();
        let mut runtime =
            MetalQwen35PrefixRuntime::new(8, 2, None, Vec::new(), None, 0.90, 0.75, false)
                .expect("runtime");

        runtime.insert_snapshot(qwen35_test_snapshot(&[1, 2], 16));
        assert!(runtime.entries.is_empty());
        assert_eq!(runtime.cached_bytes, 0);
    }

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
    fn mixed_batch_eligibility_rejects_cooperative_cancel() {
        assert!(
            !MixedBatchRequestEligibility {
                cancel_requested: true,
                is_qwen3: true,
                is_dflash_enabled: false,
            }
            .is_supported()
        );
        assert!(
            MixedBatchRequestEligibility {
                cancel_requested: false,
                is_qwen3: true,
                is_dflash_enabled: false,
            }
            .is_supported()
        );
        assert!(
            !MixedBatchRequestEligibility {
                cancel_requested: false,
                is_qwen3: false,
                is_dflash_enabled: false,
            }
            .is_supported()
        );
        assert!(
            !MixedBatchRequestEligibility {
                cancel_requested: false,
                is_qwen3: true,
                is_dflash_enabled: true,
            }
            .is_supported()
        );
    }

    #[test]
    fn stream_consumer_drop_detection_is_typed() {
        let dropped: anyhow::Error = MetalStreamError::ConsumerDropped.into();
        assert!(is_stream_consumer_dropped(&dropped));

        let other = anyhow::anyhow!("stream consumer dropped");
        assert!(!is_stream_consumer_dropped(&other));
    }

    #[test]
    fn metal_tier_adapter_rejects_t1_and_allows_t2_noop() {
        let adapter = MetalTierAdapter::new(None).with_paged_pool_pressure(1.5);
        assert_eq!(adapter.paged_pool_pressure(), 1.0);
        assert!(adapter.submit_demote(BlockId(7)).is_ok());
        assert!(adapter.submit_promote(BlockId(7), Tier::Disk).is_ok());
        assert!(
            adapter
                .submit_promote(BlockId(7), Tier::HostPinned)
                .is_err()
        );
    }

    #[test]
    fn metal_tier_adapter_disk_snapshot_roundtrip_survives_restart() {
        let _guard = metal_test_guard();
        let dir = tempdir().expect("tempdir");
        let store = Arc::new(ChunkedSnapshotStore::new(dir.path()));
        let adapter = MetalTierAdapter::new(Some(store));
        let model_fingerprint = b"qwen35-adapter-test-model".to_vec();
        let snapshot = Qwen35PrefixSnapshot {
            token_ids: vec![21, 22],
            kv_flat: vec![MlxArray::from_slice_i32(&[3, 4], &[2])],
            gdr_flat: Vec::new(),
            cache_len: 2,
            kv_capacity: 2,
        };
        let (metadata, parts, _) = snapshot
            .encode_chunked_for_disk(&model_fingerprint)
            .expect("encode snapshot");
        let fingerprint = BlockFingerprint::compute(
            KvContentContext {
                model_fingerprint: &model_fingerprint,
                kv_format_tag: METAL_QWEN35_SNAPSHOT_KV_FORMAT_TAG,
                parent: None,
            },
            &snapshot.token_ids,
        );
        let (location, stats) = adapter
            .put_disk_snapshot(
                fingerprint,
                METAL_QWEN35_CHUNKED_SNAPSHOT_NAMESPACE,
                metadata,
                parts,
                false,
            )
            .expect("persist via adapter");
        assert!(stats.chunks_written > 0);

        let restarted_store = Arc::new(ChunkedSnapshotStore::new(dir.path()));
        let restarted = MetalTierAdapter::new(Some(restarted_store));
        let reloaded = restarted
            .get_disk_snapshot(&location, Some(fingerprint))
            .expect("reload via adapter");
        let decoded = Qwen35PrefixSnapshot::decode_chunked_from_disk(reloaded, &model_fingerprint)
            .expect("decode reloaded snapshot");
        assert_eq!(decoded.token_ids, vec![21, 22]);
        assert_eq!(decoded.cache_len, 2);
        assert_eq!(decoded.kv_capacity, 2);
    }

    #[test]
    fn qwen35_disk_prefix_runtime_reconciles_persisted_snapshot_headers() {
        let _guard = metal_test_guard();
        let dir = tempdir().expect("tempdir");
        let store = Arc::new(ChunkedSnapshotStore::new(dir.path()));
        let model_fingerprint = b"qwen35-test-model".to_vec();
        let mut runtime = MetalQwen35PrefixRuntime::new(
            1024 * 1024,
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

        runtime
            .persist_snapshot_blocking(&snapshot)
            .expect("persist");
        assert!(runtime.lock_disk_index().contains(&[11, 12]));
        let qwen35_fingerprint = runtime.fingerprint_for_tokens(&[11, 12]);
        let foreign_fingerprint = BlockFingerprint([0x7a; 16]);
        store
            .put_snapshot(
                ChunkedSnapshotWrite {
                    manifest_id: foreign_fingerprint,
                    namespace: "foreign-test-snapshot".into(),
                    metadata: b"not-a-qwen35-snapshot".to_vec(),
                    parts: Vec::new(),
                },
                false,
            )
            .expect("persist foreign block");
        let disk_bytes = runtime.lock_disk_index().disk_bytes;
        assert!(disk_bytes > 0);

        let restored = MetalQwen35PrefixRuntime::new(
            1024 * 1024,
            2,
            Some(store.clone()),
            model_fingerprint,
            None,
            0.90,
            0.75,
            false,
        )
        .expect("restored runtime");
        assert!(restored.lock_disk_index().contains(&[11, 12]));
        assert_eq!(restored.lock_disk_index().disk_bytes, disk_bytes);

        let wrong_model = MetalQwen35PrefixRuntime::new(
            1024 * 1024,
            2,
            Some(store.clone()),
            b"other-model".to_vec(),
            None,
            0.90,
            0.75,
            false,
        )
        .expect("wrong-model runtime");
        assert!(wrong_model.lock_disk_index().entries.is_empty());
        assert!(
            !store
                .manifest_path_for(qwen35_fingerprint)
                .try_exists()
                .expect("stat stale manifest"),
            "wrong-model Qwen3.5 snapshot manifests should be discarded during reconciliation"
        );
        assert!(
            store
                .manifest_path_for(foreign_fingerprint)
                .try_exists()
                .expect("stat foreign manifest"),
            "non-Qwen3.5 snapshot manifests should not be deleted by Qwen3.5 reconciliation"
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

    /// M_d.1 §3c — closes the documented silent-corruption hole for the
    /// Qwen3.5 SSD prefix cache. The existing model-tree walk already
    /// folds every `.json` file's bytes into the fingerprint, so a
    /// `tokenizer.json` content change MUST flip `fp` and a stale disk
    /// snapshot MUST then be rejected by `reconcile_disk_entries`.
    /// Pre-M_d.1 there was no test for this case, only for mtime-without-
    /// content invariance.
    #[test]
    fn metal_prefix_model_fingerprint_flips_on_tokenizer_content_change() {
        let dir = tempdir().expect("tempdir");
        let gguf = dir.path().join("model.gguf");
        let tokenizer = dir.path().join("_gguf_tokenizer.json");
        std::fs::write(&gguf, b"weights").expect("write gguf");
        std::fs::write(&tokenizer, br#"{"vocab":"v1"}"#).expect("write tokenizer v1");

        let mut backend = MetalBackend::with_options(MetalBackendOptions::default());
        backend.model_dir = Some(dir.path().to_path_buf());
        backend.model_source_path = Some(gguf.clone());
        let fp_v1 = metal_prefix_model_fingerprint(&backend).expect("fingerprint v1");

        std::fs::write(&tokenizer, br#"{"vocab":"v2"}"#).expect("rewrite tokenizer v2");
        let fp_v2 = metal_prefix_model_fingerprint(&backend).expect("fingerprint v2");
        assert_ne!(
            fp_v1, fp_v2,
            "tokenizer.json byte change must flip the model fingerprint so disk \
             reconcile drops stale prefix snapshots indexed under the old vocab"
        );
    }
}
