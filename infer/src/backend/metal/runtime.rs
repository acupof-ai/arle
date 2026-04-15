use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use log::{error, info, warn};
use tokio::sync::mpsc;

use super::kv_pool::MetalKVPool;
use super::prefix_cache::MetalPrefixCache;
use super::request_state::{
    MetalRequestPhase as RuntimePhase, MetalRequestState, Qwen3PrefixSnapshot, Qwen35PrefixSnapshot,
};
use super::scheduler::{
    MetalRequestPriority, MetalScheduleDecision, MetalScheduler, MetalSchedulerConfig,
};
use super::weights::MetalWeights;
use super::{MetalBackend, MetalBackendOptions};
use crate::backend::InferenceBackend;
use crate::metrics::ServerMetrics;
use crate::scheduler::{IncomingRequest, RequestPriority, SchedulerHandle};
use crate::server_engine::{CompletionStreamDelta, FinishReason, TokenUsage};
use crate::tokenizer::{IncrementalDecoder, Tokenizer};
use crate::types::RequestId;

struct ActiveMetalRequest {
    delta_tx: mpsc::UnboundedSender<CompletionStreamDelta>,
    request_state: MetalRequestState<'static>,
    decoder: IncrementalDecoder<'static>,
    stop_processor: StopChunkProcessor,
    prompt_tokens: Vec<u32>,
    admitted_at: Instant,
    first_token_at: Option<Instant>,
}

impl ActiveMetalRequest {
    fn from_incoming(
        backend: &'static MetalBackend,
        tokenizer: &'static Tokenizer,
        incoming: IncomingRequest,
    ) -> Result<(Vec<u32>, usize, Self)> {
        let prompt_tokens = tokenizer.encode(&incoming.prompt)?;
        let max_tokens = incoming.max_tokens;
        let request_state = backend.create_request_state(&prompt_tokens, &incoming.sampling)?;
        Ok((
            prompt_tokens.clone(),
            max_tokens,
            Self {
                delta_tx: incoming.delta_tx,
                request_state,
                decoder: tokenizer.incremental_decoder(),
                stop_processor: StopChunkProcessor::new(incoming.stop.unwrap_or_default()),
                prompt_tokens,
                admitted_at: Instant::now(),
                first_token_at: None,
            },
        ))
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
            send_text_delta(&self.delta_tx, final_delta)?;
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

        let _ = self.delta_tx.send(CompletionStreamDelta {
            text_delta: String::new(),
            finish_reason: Some(finish_reason),
            usage: Some(usage),
            logprob: None,
        });
        Ok(())
    }

    fn process_token(&mut self, token_id: u32) -> Result<()> {
        if self.first_token_at.is_none() {
            self.first_token_at = Some(Instant::now());
        }
        if let Some(chunk) = self.decoder.step(token_id)? {
            self.push_text_chunk(&chunk)?;
        }
        Ok(())
    }

    fn push_text_chunk(&mut self, chunk: &str) -> Result<()> {
        if let Some(delta) = self.stop_processor.push_chunk(chunk) {
            send_text_delta(&self.delta_tx, delta)?;
        }
        Ok(())
    }

    fn prompt_len(&self) -> usize {
        self.prompt_tokens.len()
    }
}

const METAL_PREFIX_BLOCK_SIZE: usize = 16;
const METAL_PREFIX_POOL_MULTIPLIER: usize = 4;

enum MetalLivePrefixRuntime {
    Qwen3(MetalQwen3PrefixRuntime),
    Qwen35(MetalQwen35PrefixRuntime),
}

struct MetalQwen3PrefixRuntime {
    pool: MetalKVPool,
    cache: MetalPrefixCache,
}

struct MetalQwen35CachedPrefix {
    snapshot: Qwen35PrefixSnapshot,
    last_used_tick: u64,
}

struct MetalQwen35PrefixRuntime {
    entries: HashMap<Vec<u32>, MetalQwen35CachedPrefix>,
    max_cached_tokens: usize,
    cached_tokens: usize,
    next_tick: u64,
    block_size: usize,
}

struct MetalPrefixAdmission {
    scheduler_prompt_tokens: Vec<u32>,
}

impl MetalLivePrefixRuntime {
    fn new(backend: &'static MetalBackend, config: &MetalSchedulerConfig) -> Result<Option<Self>> {
        let model_config = backend.config.as_ref().context("model not loaded")?;
        let weights = backend.weights.as_ref().context("weights not loaded")?;
        let max_total_tokens = (config
            .max_active_requests
            .saturating_mul(config.prefill_chunk_size)
            .saturating_mul(METAL_PREFIX_POOL_MULTIPLIER))
        .max(METAL_PREFIX_BLOCK_SIZE * 8);
        match weights {
            MetalWeights::Qwen3(weights) => {
                let kv_dtype = weights.layers[0].attention_inputs.kv_dtype();
                let pool = MetalKVPool::new(
                    model_config.num_hidden_layers,
                    model_config.num_key_value_heads,
                    model_config.head_dim,
                    max_total_tokens,
                    kv_dtype,
                )
                .context("create live Metal prefix KV pool")?;

                info!(
                    "Metal live prefix cache enabled for Qwen3: block_size={}, max_cached_tokens={}",
                    METAL_PREFIX_BLOCK_SIZE, max_total_tokens
                );

                Ok(Some(Self::Qwen3(MetalQwen3PrefixRuntime {
                    pool,
                    cache: MetalPrefixCache::new(METAL_PREFIX_BLOCK_SIZE),
                })))
            }
            MetalWeights::Qwen35(_) => {
                info!(
                    "Metal live prefix cache enabled for Qwen3.5 snapshot replay: block_size={}, max_cached_tokens={}",
                    METAL_PREFIX_BLOCK_SIZE, max_total_tokens
                );
                Ok(Some(Self::Qwen35(MetalQwen35PrefixRuntime::new(
                    max_total_tokens,
                    METAL_PREFIX_BLOCK_SIZE,
                ))))
            }
        }
    }

    fn prepare_request(
        &mut self,
        request: &mut ActiveMetalRequest,
        metrics: &ServerMetrics,
    ) -> Result<MetalPrefixAdmission> {
        match self {
            MetalLivePrefixRuntime::Qwen3(runtime) => runtime.prepare_request(request, metrics),
            MetalLivePrefixRuntime::Qwen35(runtime) => runtime.prepare_request(request, metrics),
        }
    }

    fn publish_prompt_prefix(&mut self, request: &ActiveMetalRequest) -> Result<()> {
        match self {
            MetalLivePrefixRuntime::Qwen3(runtime) => runtime.publish_prompt_prefix(request),
            MetalLivePrefixRuntime::Qwen35(runtime) => runtime.publish_prompt_prefix(request),
        }
    }
}

impl MetalQwen3PrefixRuntime {
    fn prepare_request(
        &mut self,
        request: &mut ActiveMetalRequest,
        metrics: &ServerMetrics,
    ) -> Result<MetalPrefixAdmission> {
        let prompt_tokens = &request.prompt_tokens;
        if prompt_tokens.len() < self.cache.block_size() {
            metrics.record_prefix_lookup(false);
            return Ok(MetalPrefixAdmission {
                scheduler_prompt_tokens: prompt_tokens.clone(),
            });
        }

        let hit = self
            .cache
            .lookup(prompt_tokens)
            .context("lookup Metal live prefix cache")?;
        let reusable_len = self.reusable_prefix_len(prompt_tokens.len(), hit.matched_len);
        let scheduler_prompt_tokens = if reusable_len > 0 {
            request
                .request_state
                .import_qwen3_prefix_from_pool(
                    &self.pool,
                    reusable_len,
                    &hit.slot_indices[..reusable_len],
                )
                .context("import matched Metal prefix into request state")?;
            prompt_tokens[reusable_len..].to_vec()
        } else {
            prompt_tokens.clone()
        };
        self.cache
            .release(&hit.slot_indices)
            .context("release Metal live prefix cache lookup pins")?;
        metrics.record_prefix_lookup(reusable_len > 0);

        Ok(MetalPrefixAdmission {
            scheduler_prompt_tokens,
        })
    }

    fn publish_prompt_prefix(&mut self, request: &ActiveMetalRequest) -> Result<()> {
        let publish_len =
            (request.prompt_len() / self.cache.block_size()) * self.cache.block_size();
        if publish_len == 0 {
            return Ok(());
        }

        let Some(snapshot) = request
            .request_state
            .export_qwen3_prompt_prefix(publish_len)
            .context("export Qwen3 prompt prefix for live prefix cache")?
        else {
            return Ok(());
        };

        self.publish_snapshot(snapshot)
    }

    fn publish_snapshot(&mut self, snapshot: Qwen3PrefixSnapshot) -> Result<()> {
        let full_len = snapshot.token_ids.len();
        if full_len == 0 {
            return Ok(());
        }

        let hit = self
            .cache
            .lookup(&snapshot.token_ids)
            .context("lookup cached Metal prefix before publish")?;
        let matched_len = hit.matched_len.min(full_len);
        if matched_len == full_len {
            self.cache
                .release(&hit.slot_indices)
                .context("release publish-time Metal prefix lookup pins")?;
            return Ok(());
        }

        let new_len = full_len - matched_len;
        self.ensure_capacity_for(new_len)?;
        let mut slot_indices = hit.slot_indices.clone();
        self.cache
            .release(&hit.slot_indices)
            .context("release partial publish-time Metal prefix lookup pins")?;

        let new_slots = self
            .pool
            .alloc_slots(new_len)
            .context("alloc Metal prefix cache slots for publish")?;
        for (layer_idx, (k_rows, v_rows)) in snapshot
            .k_rows_by_layer
            .iter()
            .zip(snapshot.v_rows_by_layer.iter())
            .enumerate()
        {
            let suffix_k = slice_rows(k_rows, matched_len, full_len)?;
            let suffix_v = slice_rows(v_rows, matched_len, full_len)?;
            self.pool
                .write_kv_slots(layer_idx, &new_slots, &suffix_k, &suffix_v)
                .with_context(|| format!("write Metal prefix cache rows for layer {layer_idx}"))?;
        }
        slot_indices.extend_from_slice(&new_slots);
        self.cache
            .insert(&snapshot.token_ids, &slot_indices)
            .context("insert published Qwen3 prompt into Metal prefix cache")?;
        Ok(())
    }

    fn ensure_capacity_for(&mut self, needed_tokens: usize) -> Result<()> {
        while self.pool.available_tokens() < needed_tokens {
            let freed = self.cache.evict(1);
            if freed.is_empty() {
                bail!(
                    "Metal live prefix cache out of space (need {}, available {})",
                    needed_tokens,
                    self.pool.available_tokens()
                );
            }
            self.pool
                .release_slots(&freed)
                .context("release evicted Metal prefix-cache slots")?;
        }
        Ok(())
    }

    fn reusable_prefix_len(&self, prompt_len: usize, matched_len: usize) -> usize {
        let block_size = self.cache.block_size();
        let aligned = matched_len / block_size * block_size;
        if aligned >= prompt_len {
            aligned.saturating_sub(block_size)
        } else {
            aligned
        }
    }
}

impl MetalQwen35PrefixRuntime {
    fn new(max_cached_tokens: usize, block_size: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_cached_tokens,
            cached_tokens: 0,
            next_tick: 1,
            block_size,
        }
    }

    fn prepare_request(
        &mut self,
        request: &mut ActiveMetalRequest,
        metrics: &ServerMetrics,
    ) -> Result<MetalPrefixAdmission> {
        let prompt_tokens = &request.prompt_tokens;
        if prompt_tokens.len() < self.block_size {
            metrics.record_prefix_lookup(false);
            return Ok(MetalPrefixAdmission {
                scheduler_prompt_tokens: prompt_tokens.clone(),
            });
        }

        let Some(prefix_key) = self.lookup_longest_prefix(prompt_tokens) else {
            metrics.record_prefix_lookup(false);
            return Ok(MetalPrefixAdmission {
                scheduler_prompt_tokens: prompt_tokens.clone(),
            });
        };

        let imported =
            if let Some(snapshot) = self.entries.get(&prefix_key).map(|entry| &entry.snapshot) {
                request
                    .request_state
                    .import_qwen35_prefix_snapshot(snapshot, prefix_key.len())
                    .context("import matched Qwen3.5 prefix snapshot into request state")?
            } else {
                false
            };

        metrics.record_prefix_lookup(imported);
        if !imported {
            return Ok(MetalPrefixAdmission {
                scheduler_prompt_tokens: prompt_tokens.clone(),
            });
        }

        self.touch(&prefix_key);
        Ok(MetalPrefixAdmission {
            scheduler_prompt_tokens: prompt_tokens[prefix_key.len()..].to_vec(),
        })
    }

    fn publish_prompt_prefix(&mut self, request: &ActiveMetalRequest) -> Result<()> {
        let snapshots = request
            .request_state
            .export_qwen35_prompt_prefixes(self.block_size)
            .context("export Qwen3.5 prompt prefix snapshots")?;
        for snapshot in snapshots {
            self.insert_snapshot(snapshot)?;
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

    fn insert_snapshot(&mut self, snapshot: Qwen35PrefixSnapshot) -> Result<()> {
        let token_count = snapshot.token_ids.len();
        if token_count < self.block_size || !token_count.is_multiple_of(self.block_size) {
            return Ok(());
        }
        let tick = self.bump_tick();
        if let Some(existing) = self.entries.get_mut(&snapshot.token_ids) {
            existing.last_used_tick = tick;
            return Ok(());
        }
        if token_count > self.max_cached_tokens {
            return Ok(());
        }

        self.ensure_capacity_for(token_count);
        let key = snapshot.token_ids.clone();
        self.cached_tokens += token_count;
        self.entries.insert(
            key,
            MetalQwen35CachedPrefix {
                snapshot,
                last_used_tick: tick,
            },
        );
        Ok(())
    }

    fn ensure_capacity_for(&mut self, needed_tokens: usize) {
        while self.cached_tokens.saturating_add(needed_tokens) > self.max_cached_tokens {
            let Some((lru_key, lru_tokens)) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used_tick)
                .map(|(tokens, entry)| (tokens.clone(), entry.snapshot.token_ids.len()))
            else {
                break;
            };
            self.entries.remove(&lru_key);
            self.cached_tokens = self.cached_tokens.saturating_sub(lru_tokens);
        }
    }

    fn touch(&mut self, key: &[u32]) {
        let tick = self.bump_tick();
        if let Some(entry) = self.entries.get_mut(key) {
            entry.last_used_tick = tick;
        }
    }

    fn bump_tick(&mut self) -> u64 {
        let tick = self.next_tick;
        self.next_tick = self.next_tick.saturating_add(1);
        tick
    }
}

fn slice_rows(
    array: &super::mlx::MlxArray,
    start: usize,
    end: usize,
) -> Result<super::mlx::MlxArray> {
    use super::mlx::slice;

    let end_i32 = i32::try_from(end).context("slice_rows end overflow")?;
    let start_i32 = i32::try_from(start).context("slice_rows start overflow")?;
    let width = *array
        .shape()
        .get(1)
        .context("slice_rows expects a rank-2 row-major array")?;
    Ok(slice(array, &[start_i32, 0], &[end_i32, width], &[1, 1]))
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
) -> Result<SchedulerHandle> {
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

pub fn spawn_metal_scheduler_handle_from_path_with_options_and_metrics(
    model_path: &str,
    options: MetalBackendOptions,
    max_waiting: usize,
    metrics: ServerMetrics,
) -> Result<SchedulerHandle> {
    if options.dflash.is_some() {
        bail!(
            "the live Metal scheduler runtime does not support DFlash yet; use the serial runtime path instead"
        );
    }

    let mut backend = MetalBackend::with_options(options);
    backend.load(Path::new(model_path))?;

    let model_id = Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(model_path)
        .to_string();

    let (tx, rx) = mpsc::unbounded_channel();
    let waiting_count = Arc::new(AtomicUsize::new(0));
    let handle =
        SchedulerHandle::with_shared_waiting_count(tx, &model_id, max_waiting, waiting_count);

    let runtime_handle = handle.clone();
    std::thread::spawn(move || {
        // The runtime owns one backend instance for the process lifetime. The
        // request-state API currently borrows backend internals, so keep the
        // loaded backend stable inside the worker thread until the server exits.
        let backend: &'static MetalBackend = Box::leak(Box::new(backend));
        let tokenizer = match backend.tokenizer.as_ref() {
            Some(tokenizer) => tokenizer,
            None => {
                error!("Metal scheduler runtime failed: model tokenizer not loaded");
                return;
            }
        };
        let tokenizer: &'static Tokenizer = tokenizer;

        let result = catch_unwind(AssertUnwindSafe(|| {
            run_metal_scheduler_runtime(
                backend,
                tokenizer,
                rx,
                runtime_handle,
                metrics,
                MetalSchedulerConfig {
                    max_waiting_requests: 0,
                    ..MetalSchedulerConfig::default()
                },
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

    Ok(handle)
}

pub fn spawn_metal_scheduler_handle_from_path(
    model_path: &str,
    max_waiting: usize,
) -> Result<SchedulerHandle> {
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
    handle: SchedulerHandle,
    metrics: ServerMetrics,
    config: MetalSchedulerConfig,
) -> Result<()> {
    let mut prefix_runtime = MetalLivePrefixRuntime::new(backend, &config)?;
    let mut scheduler = MetalScheduler::new(config)?;
    let mut active = HashMap::<RequestId, ActiveMetalRequest>::new();
    let mut request_rx_closed = false;

    info!("Metal scheduler runtime started");

    loop {
        drain_incoming_requests(
            backend,
            tokenizer,
            &handle,
            &metrics,
            &mut prefix_runtime,
            &mut request_rx,
            &mut request_rx_closed,
            &mut scheduler,
            &mut active,
        );
        reap_closed_clients(&mut scheduler, &mut active);
        refresh_runtime_metrics(&metrics, &handle, &scheduler, &active);

        if request_rx_closed && active.is_empty() && scheduler.waiting_len() == 0 {
            info!("Metal scheduler runtime shutting down: all handles dropped");
            break;
        }

        if active.is_empty() && scheduler.waiting_len() == 0 {
            match request_rx.blocking_recv() {
                Some(incoming) => {
                    handle.consume_one();
                    admit_request(
                        backend,
                        tokenizer,
                        incoming,
                        &metrics,
                        &mut prefix_runtime,
                        &mut scheduler,
                        &mut active,
                    );
                    refresh_runtime_metrics(&metrics, &handle, &scheduler, &active);
                }
                None => {
                    request_rx_closed = true;
                    continue;
                }
            }
        }

        match scheduler.step() {
            MetalScheduleDecision::Idle => continue,
            MetalScheduleDecision::PrefillChunk(prefill) => execute_prefill_chunk(
                prefill.req_id,
                prefill.input_tokens.len(),
                &metrics,
                &mut prefix_runtime,
                &mut scheduler,
                &mut active,
            ),
            MetalScheduleDecision::DecodeBatch(batch) => {
                execute_decode_batch(batch.req_ids, &metrics, &mut scheduler, &mut active);
            }
            MetalScheduleDecision::Mixed { decode, prefill } => {
                execute_decode_batch(decode.req_ids, &metrics, &mut scheduler, &mut active);
                execute_prefill_chunk(
                    prefill.req_id,
                    prefill.input_tokens.len(),
                    &metrics,
                    &mut prefix_runtime,
                    &mut scheduler,
                    &mut active,
                );
            }
        }

        refresh_runtime_metrics(&metrics, &handle, &scheduler, &active);
    }

    Ok(())
}

fn drain_incoming_requests(
    backend: &'static MetalBackend,
    tokenizer: &'static Tokenizer,
    handle: &SchedulerHandle,
    metrics: &ServerMetrics,
    prefix_runtime: &mut Option<MetalLivePrefixRuntime>,
    request_rx: &mut mpsc::UnboundedReceiver<IncomingRequest>,
    request_rx_closed: &mut bool,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    loop {
        match request_rx.try_recv() {
            Ok(incoming) => {
                handle.consume_one();
                admit_request(
                    backend,
                    tokenizer,
                    incoming,
                    metrics,
                    prefix_runtime,
                    scheduler,
                    active,
                );
            }
            Err(mpsc::error::TryRecvError::Empty) => break,
            Err(mpsc::error::TryRecvError::Disconnected) => {
                *request_rx_closed = true;
                break;
            }
        }
    }
}

fn admit_request(
    backend: &'static MetalBackend,
    tokenizer: &'static Tokenizer,
    incoming: IncomingRequest,
    metrics: &ServerMetrics,
    prefix_runtime: &mut Option<MetalLivePrefixRuntime>,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    if incoming.delta_tx.is_closed() {
        return;
    }

    let priority = map_request_priority(incoming.priority);
    let (prompt_tokens, max_tokens, mut request) =
        match ActiveMetalRequest::from_incoming(backend, tokenizer, incoming) {
            Ok(request) => request,
            Err(err) => {
                error!("Metal scheduler request init failed: {err:#}");
                metrics.record_request_failed();
                return;
            }
        };

    let scheduler_prompt_tokens = match prefix_runtime.as_mut() {
        Some(prefix_runtime) => match prefix_runtime.prepare_request(&mut request, metrics) {
            Ok(admission) => admission.scheduler_prompt_tokens,
            Err(err) => {
                error!("Metal prefix-cache admission failed: {err:#}");
                metrics.record_request_failed();
                return;
            }
        },
        None => prompt_tokens.clone(),
    };

    let req_id = match scheduler.submit(scheduler_prompt_tokens, max_tokens, priority) {
        Ok(req_id) => req_id,
        Err(err) => {
            error!("Metal scheduler submit failed: {err}");
            metrics.record_request_failed();
            return;
        }
    };

    if active.insert(req_id, request).is_some() {
        warn!("Metal scheduler request id collision for {:?}", req_id);
    }
}

fn execute_prefill_chunk(
    req_id: RequestId,
    budget: usize,
    metrics: &ServerMetrics,
    prefix_runtime: &mut Option<MetalLivePrefixRuntime>,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    enum Outcome {
        Progress {
            emitted_token: Option<u32>,
            runtime_finished: bool,
            stop_hit: bool,
        },
        ClientDropped,
        Failed(anyhow::Error),
    }

    let outcome = {
        let Some(request) = active.get_mut(&req_id) else {
            warn!(
                "Metal prefill chunk referenced missing request {:?}",
                req_id
            );
            scheduler.finish_request(req_id);
            return;
        };

        if request.delta_closed() {
            Outcome::ClientDropped
        } else {
            match request.prefill_chunk(budget) {
                Ok(emitted_token) => Outcome::Progress {
                    emitted_token,
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
        }
    };

    match outcome {
        Outcome::Progress {
            emitted_token,
            runtime_finished,
            stop_hit,
        } => {
            let mut scheduler_finished = false;
            if let Some(token) = emitted_token {
                if let Some(prefix_runtime) = prefix_runtime.as_mut()
                    && let Some(request) = active.get(&req_id)
                    && let Err(err) = prefix_runtime.publish_prompt_prefix(request)
                {
                    warn!("Metal live prefix publish failed for {:?}: {err:#}", req_id);
                }
                match scheduler.complete_prefill(req_id, token) {
                    Ok(done) => scheduler_finished = done,
                    Err(err) => {
                        error!("Metal complete_prefill failed for {:?}: {err}", req_id);
                        metrics.record_request_failed();
                        cancel_request(req_id, scheduler, active);
                        return;
                    }
                }
            }

            if runtime_finished || stop_hit || scheduler_finished {
                finalize_request(req_id, metrics, scheduler, active);
            }
        }
        Outcome::ClientDropped => cancel_request(req_id, scheduler, active),
        Outcome::Failed(err) => {
            error!("Metal prefill chunk failed for {:?}: {err:#}", req_id);
            metrics.record_request_failed();
            cancel_request(req_id, scheduler, active);
        }
    }
}

fn execute_decode_batch(
    req_ids: Vec<RequestId>,
    metrics: &ServerMetrics,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    if req_ids.is_empty() {
        return;
    }

    let mut staged = Vec::with_capacity(req_ids.len());
    for req_id in req_ids {
        let Some(request) = active.remove(&req_id) else {
            warn!("Metal decode batch referenced missing request {:?}", req_id);
            scheduler.finish_request(req_id);
            continue;
        };
        staged.push((req_id, request));
    }

    let mut open = Vec::with_capacity(staged.len());
    for (req_id, request) in staged {
        if request.delta_closed() {
            scheduler.finish_request(req_id);
            continue;
        }
        open.push((req_id, request));
    }

    let batch_result = if open.len() >= 2 {
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
    } else {
        None
    };

    if let Some(sampled_tokens) = batch_result {
        for ((req_id, mut request), sampled_token) in open.into_iter().zip(sampled_tokens) {
            if let Err(err) = request.process_token(sampled_token) {
                error!(
                    "Metal batched decode post-process failed for {:?}: {err:#}",
                    req_id
                );
                metrics.record_request_failed();
                cancel_detached_request(req_id, request, scheduler);
                continue;
            }
            finish_or_requeue_decoded_request(
                req_id,
                request,
                sampled_token,
                metrics,
                scheduler,
                active,
            );
        }
        return;
    }

    for (req_id, request) in open {
        execute_decode_single(req_id, request, metrics, scheduler, active);
    }
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
            sampled_token: u32,
            runtime_finished: bool,
            stop_hit: bool,
        },
        ClientDropped,
        Failed(anyhow::Error),
    }

    let outcome = if request.delta_closed() {
        Outcome::ClientDropped
    } else {
        match request.decode_step() {
            Ok(sampled_token) => Outcome::Progress {
                sampled_token,
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
            sampled_token,
            runtime_finished,
            stop_hit,
        } => {
            let scheduler_finished = match scheduler.advance_decode(req_id, sampled_token) {
                Ok(done) => done,
                Err(err) => {
                    error!("Metal advance_decode failed for {:?}: {err}", req_id);
                    metrics.record_request_failed();
                    cancel_detached_request(req_id, request, scheduler);
                    return;
                }
            };

            if runtime_finished || stop_hit || scheduler_finished {
                finalize_detached_request(req_id, request, metrics, scheduler);
            } else {
                active.insert(req_id, request);
            }
        }
        Outcome::ClientDropped => {
            scheduler.finish_request(req_id);
            if let Err(err) = request.cancel() {
                warn!("Metal request cancel failed for {:?}: {err:#}", req_id);
            }
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
    sampled_token: u32,
    metrics: &ServerMetrics,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    let runtime_finished = request.phase() == RuntimePhase::Finished;
    let stop_hit = request.stop_hit();
    let scheduler_finished = match scheduler.advance_decode(req_id, sampled_token) {
        Ok(done) => done,
        Err(err) => {
            error!("Metal advance_decode failed for {:?}: {err}", req_id);
            metrics.record_request_failed();
            cancel_detached_request(req_id, request, scheduler);
            return;
        }
    };

    if runtime_finished || stop_hit || scheduler_finished {
        finalize_detached_request(req_id, request, metrics, scheduler);
    } else {
        active.insert(req_id, request);
    }
}

fn cancel_detached_request(
    req_id: RequestId,
    mut request: ActiveMetalRequest,
    scheduler: &mut MetalScheduler,
) {
    scheduler.finish_request(req_id);
    if let Err(err) = request.cancel() {
        warn!("Metal request cancel failed for {:?}: {err:#}", req_id);
    }
}

fn finalize_detached_request(
    req_id: RequestId,
    mut request: ActiveMetalRequest,
    metrics: &ServerMetrics,
    scheduler: &mut MetalScheduler,
) {
    scheduler.finish_request(req_id);
    record_request_completed(metrics, &request);
    if let Err(err) = request.send_final_delta() {
        warn!("Metal request final delta failed for {:?}: {err:#}", req_id);
    }
}

fn reap_closed_clients(
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
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
    scheduler.finish_request(req_id);
    if let Some(mut request) = active.remove(&req_id)
        && let Err(err) = request.cancel()
    {
        warn!("Metal request cancel failed for {:?}: {err:#}", req_id);
    }
}

fn finalize_request(
    req_id: RequestId,
    metrics: &ServerMetrics,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    scheduler.finish_request(req_id);
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
}

fn record_request_completed(metrics: &ServerMetrics, request: &ActiveMetalRequest) {
    let completion_tokens = request.request_state.generated_tokens() as u64;
    let e2e_s = request.admitted_at.elapsed().as_secs_f64();
    let ttft_s = request
        .first_token_at
        .map(|first| first.duration_since(request.admitted_at).as_secs_f64())
        .unwrap_or(e2e_s);
    let tpot_s = if completion_tokens > 1 {
        (e2e_s - ttft_s).max(0.0) / (completion_tokens - 1) as f64
    } else {
        0.0
    };
    metrics.record_request_completed(
        request.prompt_len() as u64,
        completion_tokens,
        ttft_s,
        tpot_s,
        e2e_s,
    );
}

fn refresh_runtime_metrics(
    metrics: &ServerMetrics,
    handle: &SchedulerHandle,
    scheduler: &MetalScheduler,
    active: &HashMap<RequestId, ActiveMetalRequest>,
) {
    metrics.set_active(active.len() as u64);
    metrics.set_waiting((handle.waiting_count() + scheduler.waiting_len()) as u64);

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

fn send_text_delta(
    delta_tx: &mpsc::UnboundedSender<CompletionStreamDelta>,
    text_delta: String,
) -> Result<()> {
    if text_delta.is_empty() {
        return Ok(());
    }

    delta_tx
        .send(CompletionStreamDelta {
            text_delta,
            finish_reason: None,
            usage: None,
            logprob: None,
        })
        .map_err(|_| anyhow!("stream consumer dropped"))
}

struct StopChunkProcessor {
    accumulated: String,
    sent_len: usize,
    stops: Vec<String>,
    max_stop_len: usize,
    hit_stop: bool,
}

impl StopChunkProcessor {
    fn new(stops: Vec<String>) -> Self {
        let max_stop_len = stops.iter().map(String::len).max().unwrap_or(0);
        Self {
            accumulated: String::new(),
            sent_len: 0,
            stops,
            max_stop_len,
            hit_stop: false,
        }
    }

    fn push_chunk(&mut self, chunk: &str) -> Option<String> {
        if self.hit_stop {
            return None;
        }

        self.accumulated.push_str(chunk);

        if let Some(stop_pos) = self.find_earliest_stop() {
            let delta = self.accumulated[self.sent_len..stop_pos].to_string();
            self.sent_len = stop_pos;
            self.hit_stop = true;
            return Some(delta);
        }

        if self.max_stop_len <= 1 {
            return self.flush_ready(self.accumulated.len());
        }

        let safe_end = self
            .accumulated
            .len()
            .saturating_sub(self.max_stop_len.saturating_sub(1));
        let safe_end = clamp_char_boundary(&self.accumulated, safe_end);
        self.flush_ready(safe_end)
    }

    fn finish(&mut self) -> Option<String> {
        if self.hit_stop {
            return None;
        }
        self.flush_ready(self.accumulated.len())
    }

    fn hit_stop(&self) -> bool {
        self.hit_stop
    }

    fn flush_ready(&mut self, end: usize) -> Option<String> {
        if end <= self.sent_len {
            return None;
        }
        let delta = self.accumulated[self.sent_len..end].to_string();
        self.sent_len = end;
        Some(delta)
    }

    fn find_earliest_stop(&self) -> Option<usize> {
        let unsent = &self.accumulated[self.sent_len..];
        let mut earliest = None::<usize>;

        for stop in &self.stops {
            if stop.is_empty() {
                continue;
            }
            if let Some(pos) = unsent.find(stop) {
                let absolute = self.sent_len + pos;
                earliest = Some(match earliest {
                    None => absolute,
                    Some(existing) => existing.min(absolute),
                });
            }
        }

        earliest
    }
}

fn clamp_char_boundary(text: &str, mut idx: usize) -> usize {
    idx = idx.min(text.len());
    while idx > 0 && !text.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}
